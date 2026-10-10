"""
XVA of an interest rate swap under the Vasicek short-rate model.

Computes CVA, DVA and FVA of a payer swap by Monte Carlo, then studies
the effect of a collateral threshold (CSA) and of wrong-way risk.

    CVA = (1 - R_C) * sum_k  E[D(0,t_k) V+(t_k)] * dPD_C(t_k)
    DVA = (1 - R_B) * sum_k  E[D(0,t_k) V-(t_k)] * dPD_B(t_k)
    FVA = sum_k  ( s_b * E[D V+] - s_l * E[D V-] ) * dt

Usage:  python xva_swap.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- parameters
A, B_LONG, SIGMA, R0 = 0.10, 0.03, 0.01, 0.025   # Vasicek: dr = a(b - r)dt + sigma dW
MATURITY, DELTA = 10.0, 0.25                     # 10y swap, quarterly payments
NOTIONAL = 1.0
N_PATHS, SUBSTEPS = 40_000, 6                    # 6 simulation steps per quarter
SPREAD_C, REC_C = 0.0150, 0.40                   # counterparty CDS spread / recovery
SPREAD_B, REC_B = 0.0060, 0.40                   # bank CDS spread / recovery
FUND_BORROW, FUND_LEND = 0.0050, 0.0020          # funding spreads
SEED = 42


# ------------------------------------------------------------- Vasicek model
def bond_price(r, tau):
    """Zero-coupon bond price P(t, t+tau) given short rate r (closed form)."""
    b_fn = (1.0 - np.exp(-A * tau)) / A
    a_fn = np.exp((B_LONG - SIGMA**2 / (2 * A**2)) * (b_fn - tau)
                  - SIGMA**2 * b_fn**2 / (4 * A))
    return a_fn * np.exp(-b_fn * r)


def simulate_short_rate(n_paths, n_steps, dt, rng):
    """Exact simulation of Vasicek paths. Returns array (n_paths, n_steps+1)."""
    decay = np.exp(-A * dt)
    std = SIGMA * np.sqrt((1.0 - decay**2) / (2 * A))
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = R0
    half = n_paths // 2
    for i in range(n_steps):
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])                       # antithetic variates
        r[:, i + 1] = r[:, i] * decay + B_LONG * (1.0 - decay) + std * z
    return r


# --------------------------------------------------------------------- swap
def par_rate(pay_times):
    """Fixed rate making the swap worth zero at t = 0."""
    p = bond_price(R0, pay_times)
    return (1.0 - p[-1]) / (DELTA * p.sum())


def swap_values(r_pay, pay_times, fixed_rate):
    """Payer swap value on each path at each payment date (just after payment).

    V(t_k) = N * [ (1 - P(t_k, T_n)) - K * delta * sum_{j>k} P(t_k, T_j) ]
    """
    n = len(pay_times)
    values = np.zeros((r_pay.shape[0], n + 1))
    for k in range(n):                                    # k = 0 is today
        t_k = 0.0 if k == 0 else pay_times[k - 1]
        tau = pay_times[k:] - t_k                         # remaining payments
        p = bond_price(r_pay[:, k][:, None], tau[None, :])
        values[:, k] = NOTIONAL * ((1.0 - p[:, -1]) - fixed_rate * DELTA * p.sum(axis=1))
    return values                                         # V(T_n) = 0


# ---------------------------------------------------------------------- XVA
def default_increments(spread, recovery, grid):
    """Default probability on each interval, constant hazard rate = s / (1 - R)."""
    survival = np.exp(-spread / (1.0 - recovery) * grid)
    return survival[:-1] - survival[1:]


def xva(disc_pos, disc_neg, grid):
    """CVA, DVA, FVA from discounted positive / negative exposure profiles."""
    cva = (1 - REC_C) * np.sum(disc_pos[1:] * default_increments(SPREAD_C, REC_C, grid))
    dva = (1 - REC_B) * np.sum(disc_neg[1:] * default_increments(SPREAD_B, REC_B, grid))
    fva = np.sum((FUND_BORROW * disc_pos[1:] - FUND_LEND * disc_neg[1:]) * np.diff(grid))
    return cva, dva, fva


def cva_wrong_way(disc_factor, pos_exposure, r_pay, grid, beta):
    """CVA with a stochastic default intensity correlated with the short rate.

    lambda_t = lambda_0 * exp(beta * z_t - beta^2 / 2),  z_t = standardised r_t,
    so that E[lambda_t] = lambda_0 for every beta (same average credit quality).
    beta > 0: counterparty credit worsens when rates rise -> wrong-way risk
    for a payer swap.
    """
    lam0 = SPREAD_C / (1.0 - REC_C)
    std = r_pay.std(axis=0)
    std[0] = 1.0                                          # r_0 is deterministic
    z = (r_pay - r_pay.mean(axis=0)) / std
    lam = lam0 * np.exp(beta * z - 0.5 * beta**2)
    integral = np.cumsum(0.5 * (lam[:, 1:] + lam[:, :-1]) * np.diff(grid), axis=1)
    survival = np.hstack([np.ones((len(lam), 1)), np.exp(-integral)])
    d_pd = survival[:, :-1] - survival[:, 1:]
    return (1 - REC_C) * np.mean(np.sum(disc_factor[:, 1:] * pos_exposure[:, 1:] * d_pd, axis=1))


# --------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    n_pay = int(round(MATURITY / DELTA))
    pay_times = DELTA * np.arange(1, n_pay + 1)
    grid = np.concatenate([[0.0], pay_times])
    dt = DELTA / SUBSTEPS

    r = simulate_short_rate(N_PATHS, n_pay * SUBSTEPS, dt, rng)
    integral = np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * dt, axis=1)
    disc_full = np.hstack([np.ones((N_PATHS, 1)), np.exp(-integral)])
    r_pay, disc = r[:, ::SUBSTEPS], disc_full[:, ::SUBSTEPS]

    # --- validation: Monte Carlo discount factors vs closed form
    mc_bond, exact_bond = disc[:, -1].mean(), bond_price(R0, MATURITY)
    print(f"P(0,{MATURITY:.0f}y)  MC = {mc_bond:.5f}   closed form = {exact_bond:.5f}"
          f"   error = {abs(mc_bond - exact_bond) / exact_bond:.3%}")

    k_fix = par_rate(pay_times)
    v = swap_values(r_pay, pay_times, k_fix)
    pos, neg = np.maximum(v, 0.0), np.maximum(-v, 0.0)
    print(f"Par swap rate = {k_fix:.4%}")

    bps = 1e4 / NOTIONAL
    cva, dva, fva = xva((disc * pos).mean(axis=0), (disc * neg).mean(axis=0), grid)
    path_cva = (1 - REC_C) * np.sum((disc * pos)[:, 1:]
                                    * default_increments(SPREAD_C, REC_C, grid), axis=1)
    print("\n--- Uncollateralised (bps of notional) ---")
    half = N_PATHS // 2                                   # antithetic pairs are not independent:
    pair_cva = 0.5 * (path_cva[:half] + path_cva[half:])  # the std error is taken over pair averages
    print(f"CVA = {cva * bps:6.2f}  (MC std error {pair_cva.std(ddof=1) / np.sqrt(half) * bps:.2f})")
    print(f"DVA = {dva * bps:6.2f}")
    print(f"FVA = {fva * bps:6.2f}")
    print(f"All-in adjustment (-CVA + DVA - FVA) = {(-cva + dva - fva) * bps:6.2f}")

    print("\n--- Collateral: CSA threshold H (exposure capped at H) ---")
    for h in [0.05, 0.02, 0.01, 0.005]:
        cva_h, _, _ = xva((disc * np.minimum(pos, h * NOTIONAL)).mean(axis=0),
                          (disc * neg).mean(axis=0), grid)
        print(f"H = {h:5.1%} of notional   CVA = {cva_h * bps:6.2f}   reduction = {1 - cva_h / cva:5.1%}")

    print("\n--- Wrong-way risk: default intensity correlated with rates ---")
    for beta in [-0.5, 0.0, 0.5, 1.0]:
        c = cva_wrong_way(disc, pos, r_pay, grid, beta)
        print(f"beta = {beta:+.1f}   CVA = {c * bps:6.2f}   vs independent = {c / cva - 1:+6.1%}")

    # --- exposure profile plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, pos.mean(axis=0) * 100, label="EE  (expected positive exposure)")
    ax.plot(grid, -neg.mean(axis=0) * 100, label="ENE (expected negative exposure)")
    ax.plot(grid, np.percentile(v, 95, axis=0) * 100, "--", label="PFE 95%")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure (% of notional)")
    ax.set_title("10y payer swap - exposure profile under Vasicek")
    ax.legend()
    fig.tight_layout()
    fig.savefig("exposure_profile.png", dpi=150)
    print("\nSaved exposure_profile.png")


if __name__ == "__main__":
    main()
