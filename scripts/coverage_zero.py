"""Coverage of Rambachan-Roth honest confidence sets under a recording response.

Corollary 4.7 of theory/MAIN_RESULTS.md proves that a smoothness bound calibrated
from pre-treatment second differences converges to zero when the recording map
responds to treatment, so the resulting honest confidence set has asymptotic
coverage zero for the latent ATT.  This script checks that claim by simulation
and extends it to the relative-magnitudes class, which the theory notes do not
cover and which is the default choice in applied work.

The mechanism.  Under a treatment-responsive recording map the observationally
equivalent parallel-trends deviation is exactly zero before treatment (Theorem
4.6) and equal to the recording wedge afterwards.  Every pre-period difference a
calibration can read is therefore pure sampling noise, of order n^{-1/2}, while
the post-period wedge it is meant to bound is of order one.

Two calibrations are reported at each sample size.  "point" is what the
HonestDiD default does: read the largest pre-period difference off the estimated
coefficients.  "conservative" replaces that with a 95% upper confidence bound on
the same quantity, ignores the covariance between neighbouring coefficients so
the bound widens further, and then adds an unadjusted normal sampling term on
top of the full identified set.  The conservative interval is wider than the ARP
interval HonestDiD reports, so a coverage failure for it implies one for
HonestDiD.

Run:  python scripts/coverage_zero.py          (REPS env var overrides rep count)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from did_measurement.dgp import DGPConfig, generate_panel          # noqa: E402
from did_measurement.estimators import event_study_twfe            # noqa: E402

K_PRE, K_POST = 5, 5
ELL_STAR = 3           # horizon at which the latent ATT is targeted
Z = 1.959963984540054
MBARS = (0.5, 1.0, 2.0)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "theory",
                   "coverage_zero.json")


def config(n_units, seed, kappa):
    """One focal category with unit loading and full discretion, so the latent ATT
    is exactly 1.0 and the recording wedge is exactly kappa at every horizon."""
    return DGPConfig(
        n_units=n_units, n_periods=16, cohorts=(8,), cohort_shares=(0.5,),
        n_categories=1, include_bridge=False, discretion=(1.0,), theta_spread=0.0,
        tau_level=1.0, tau_shape="constant", kappa=kappa, kappa_shape="constant",
        map_type="additive", mu_scale=1.0, sigma_u=0.2, seed=seed,
    )


def fit(df):
    """Event study with ell = -1 as the omitted reference."""
    es = event_study_twfe(df, outcome="Y", k_pre=K_PRE, k_post=K_POST)
    tab = es["table"]
    coef = {int(r.ell): float(r.coef) for r in tab.itertuples()}
    se = {int(r.ell): float(r.se) for r in tab.itertuples()}
    coef[-1], se[-1] = 0.0, 0.0
    return coef, se


def pre_first_diff(coef, se, conservative):
    """Largest pre-period first difference |delta(l) - delta(l-1)|, the quantity a
    relative-magnitudes calibration reads off the data."""
    best = 0.0
    for ell in range(-K_PRE + 1, 0):
        d = abs(coef[ell] - coef[ell - 1])
        if conservative:
            d += Z * np.hypot(se[ell], se[ell - 1])
        best = max(best, d)
    return best


def pre_second_diff(coef, se, conservative):
    """Largest second difference computable from PRE-period coefficients alone.
    The second difference at ell = -1 involves delta(0) and is not available."""
    best = 0.0
    for ell in range(-K_PRE + 1, -1):
        d = abs(coef[ell + 1] - 2 * coef[ell] + coef[ell - 1])
        if conservative:
            d += Z * np.sqrt(se[ell + 1] ** 2 + 4 * se[ell] ** 2 + se[ell - 1] ** 2)
        best = max(best, d)
    return best


def rm_halfwidth(mbar, dpre):
    """delta(-1) = 0 and each of the ELL_STAR + 1 post-period first differences is
    bounded by mbar * dpre, so |delta(ELL_STAR)| <= (ELL_STAR + 1) * mbar * dpre."""
    return (ELL_STAR + 1) * mbar * dpre


def sd_interval(m, coef, b, sb):
    """Delta^SD(m) set at ELL_STAR: the linear extrapolation of the last two
    pre-period coefficients, widened by the curvature the class permits and by a
    normal sampling term."""
    lin = coef[-1] + (ELL_STAR + 1) * (coef[-1] - coef[-2])
    curv = m * (ELL_STAR + 1) * (ELL_STAR + 2) / 2.0
    pad = Z * sb
    return b - lin - curv - pad, b - lin + curv + pad


def run(n_units, kappa, reps, seed0):
    lat_true, wedge_true = 1.0, kappa
    acc = {("RM", mbar, mode): 0 for mbar in MBARS for mode in ("point", "cons")}
    acc[("SD", "precal", "point")] = 0
    acc[("SD", "precal", "cons")] = 0
    acc[("SD", "inst", "cons")] = 0
    covers_observed = 0
    dpre_p, dpre_c, mhat_p, hw = [], [], [], []

    for r in range(reps):
        df = generate_panel(config(n_units, seed0 + r, kappa))
        assert abs(df.attrs["true_latent_es"][0][ELL_STAR] - lat_true) < 1e-12
        assert abs(df.attrs["true_meas_es"][0][ELL_STAR] - wedge_true) < 1e-12
        coef, se = fit(df)
        b, sb = coef[ELL_STAR], se[ELL_STAR]

        covers_observed += (b - Z * sb <= lat_true + wedge_true <= b + Z * sb)

        for mode, cons in (("point", False), ("cons", True)):
            dpre = pre_first_diff(coef, se, cons)
            (dpre_p if mode == "point" else dpre_c).append(dpre)
            for mbar in MBARS:
                w = rm_halfwidth(mbar, dpre)
                pad = Z * sb
                acc[("RM", mbar, mode)] += (b - w - pad <= lat_true <= b + w + pad)
                if mbar == 1.0 and mode == "point":
                    hw.append(w + pad)

            mhat = pre_second_diff(coef, se, cons)
            if mode == "point":
                mhat_p.append(mhat)
            lo, hi = sd_interval(mhat, coef, b, sb)
            acc[("SD", "precal", mode)] += (lo <= lat_true <= hi)

        # smoothness bound motivated by the recording institution instead of pre-trends
        lo, hi = sd_interval(abs(wedge_true), coef, b, sb)
        acc[("SD", "inst", "cons")] += (lo <= lat_true <= hi)

    return {
        "n_units": n_units, "kappa": kappa, "reps": reps,
        "true_latent_att": lat_true, "true_measurement_wedge": wedge_true,
        "coverage_observed_effect_unadjusted": covers_observed / reps,
        "coverage_latent_RM_point": {str(m): acc[("RM", m, "point")] / reps for m in MBARS},
        "coverage_latent_RM_conservative": {str(m): acc[("RM", m, "cons")] / reps for m in MBARS},
        "coverage_latent_SD_precalibrated_point": acc[("SD", "precal", "point")] / reps,
        "coverage_latent_SD_precalibrated_conservative": acc[("SD", "precal", "cons")] / reps,
        "coverage_latent_SD_institutional": acc[("SD", "inst", "cons")] / reps,
        "mean_pre_first_diff_point": float(np.mean(dpre_p)),
        "mean_pre_first_diff_conservative": float(np.mean(dpre_c)),
        "mean_Mhat_second_diff_point": float(np.mean(mhat_p)),
        "mean_halfwidth_RM1_point": float(np.mean(hw)),
    }


def checkpoint(payload):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)


if __name__ == "__main__":
    base = int(os.environ.get("REPS", 400))
    grid = [(400, base), (1600, base), (6400, base), (25600, base // 2),
            (102400, base // 4), (409600, max(60, base // 8))]

    payload = {"recording": [], "placebo": [], "ell_star": ELL_STAR,
               "k_pre": K_PRE, "k_post": K_POST, "mbars": list(MBARS)}

    print("recording response active (kappa = 0.6); latent ATT = 1.0, wedge = 0.6", flush=True)
    print(f"{'N':>8} {'reps':>5} {'Dpre':>8} {'Mhat':>8} {'RM.5':>6} {'RM1':>6} "
          f"{'RM2':>6} {'RM1cons':>8} {'SDpre':>6} {'SDinst':>7} {'obs':>6}", flush=True)
    for n, reps in grid:
        res = run(n, 0.6, reps, seed0=10_000 + n)
        payload["recording"].append(res)
        checkpoint(payload)
        print(f"{n:>8} {reps:>5} {res['mean_pre_first_diff_point']:>8.4f} "
              f"{res['mean_Mhat_second_diff_point']:>8.4f} "
              f"{res['coverage_latent_RM_point']['0.5']:>6.3f} "
              f"{res['coverage_latent_RM_point']['1.0']:>6.3f} "
              f"{res['coverage_latent_RM_point']['2.0']:>6.3f} "
              f"{res['coverage_latent_RM_conservative']['1.0']:>8.3f} "
              f"{res['coverage_latent_SD_precalibrated_point']:>6.3f} "
              f"{res['coverage_latent_SD_institutional']:>7.3f} "
              f"{res['coverage_observed_effect_unadjusted']:>6.3f}", flush=True)

    print("\nplacebo: no recording response (kappa = 0), latent ATT = 1.0, wedge = 0",
          flush=True)
    for n in (1600, 25600, 409600):
        res = run(n, 0.0, max(60, base // 8), seed0=90_000 + n)
        payload["placebo"].append(res)
        checkpoint(payload)
        print(f"{n:>8} RM(1) cov={res['coverage_latent_RM_point']['1.0']:.3f}  "
              f"SD-precal cov={res['coverage_latent_SD_precalibrated_point']:.3f}  "
              f"obs cov={res['coverage_observed_effect_unadjusted']:.3f}", flush=True)

    print(f"\nwrote {os.path.abspath(OUT)}", flush=True)
