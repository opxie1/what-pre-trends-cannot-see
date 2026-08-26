"""Coverage of Rambachan-Roth honest confidence sets under a recording response,
computed with a reference implementation rather than a hand-rolled interval.

This repeats scripts/coverage_zero.py using the `diff-diff` port of HonestDiD
(v3.10.0).  For the smoothness class the port computes the optimal fixed-length
interval of Rambachan and Roth (2023, Section 4.1).  For the relative-magnitudes
class it solves the identified set by the union-of-polyhedra decomposition of
their Lemma 2.2 and then extends it by z * se; the port's own source calls that
choice conservative and notes that the ARP hybrid interval is not enabled.  A
coverage failure computed this way is therefore a bound on the coverage failure
of the interval the package would report with ARP switched on.

The smoothness bound M is calibrated the way the reference R package's exported
DeltaSD_upperBound_Mpre does it: the maximum over one-sided (1-alpha) upper
confidence bounds for the second differences of the observed pre-period
coefficients.  A second row uses a bound motivated by the recording institution
instead, which is the constructive alternative.

Run:  python scripts/coverage_zero_ref.py
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from did_measurement.dgp import DGPConfig, generate_panel          # noqa: E402
from did_measurement.estimators import event_study_twfe            # noqa: E402
from diff_diff.results import PeriodEffect, MultiPeriodDiDResults  # noqa: E402
from diff_diff.honest_did import compute_honest_did                # noqa: E402

K_PRE, K_POST = 5, 5
ELL_STAR = 3
Z = 1.959963984540054
MBARS = (0.5, 1.0, 2.0)
LATENT, WEDGE = 1.0, 0.6
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "theory",
                   "coverage_zero_ref.json")


def config(n_units, seed):
    return DGPConfig(
        n_units=n_units, n_periods=16, cohorts=(8,), cohort_shares=(0.5,),
        n_categories=1, include_bridge=False, discretion=(1.0,), theta_spread=0.0,
        tau_level=1.0, tau_shape="constant", kappa=WEDGE, kappa_shape="constant",
        map_type="additive", mu_scale=1.0, sigma_u=0.2, seed=seed,
    )


def fit(df):
    es = event_study_twfe(df, outcome="Y", k_pre=K_PRE, k_post=K_POST)
    tab = es["table"]
    labels = [int(x) for x in tab["ell"]]
    coef = np.asarray(tab["coef"], dtype=float)
    se = np.asarray(tab["se"], dtype=float)
    return labels, coef, se, np.asarray(es["vcov"], dtype=float)


def m_upper_bound_mpre(labels, coef, se, vcov, alpha=0.05):
    """Replicates HonestDiD::DeltaSD_upperBound_Mpre.

    The maximum over one-sided (1-alpha) upper confidence bounds for the second
    differences of the observed pre-period event-study coefficients.  Only
    coefficients at ell <= -2 enter; the reference period ell = -1 is normalised
    to zero and the second difference at ell = -1 involves a post-period
    coefficient, so a pre-period calibration cannot see it.
    """
    z = 1.6448536269514722 if abs(alpha - 0.05) < 1e-12 else None
    if z is None:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha)
    idx = {l: i for i, l in enumerate(labels)}
    pre = sorted([l for l in labels if l <= -2])
    best = 0.0
    for a, b, c in zip(pre, pre[1:], pre[2:]):
        w = np.zeros(len(labels))
        w[idx[c]] += 1.0
        w[idx[b]] += -2.0
        w[idx[a]] += 1.0
        val = float(w @ coef)
        sd = float(np.sqrt(max(w @ vcov @ w, 0.0)))
        best = max(best, abs(val) + z * sd)
    return best


def container(labels, coef, se, vcov):
    """Wrap the event study in the results object the reference port expects."""
    pre = [l for l in labels if l < 0]
    post = [l for l in labels if l >= 0]
    order = [labels.index(l) for l in pre + post]
    b = coef[order]
    V = vcov[np.ix_(order, order)]
    eff = {}
    for k, l in enumerate(pre + post):
        s = float(np.sqrt(max(V[k, k], 0.0)))
        eff[l] = PeriodEffect(period=l, effect=float(b[k]), se=s,
                              t_stat=float(b[k] / s) if s > 0 else 0.0,
                              p_value=1.0,
                              conf_int=(float(b[k] - Z * s), float(b[k] + Z * s)))
    return MultiPeriodDiDResults(
        period_effects=eff, avg_att=float(b[len(pre):].mean()), avg_se=1.0,
        avg_t_stat=0.0, avg_p_value=1.0, avg_conf_int=(0.0, 0.0),
        n_obs=0, n_treated=0, n_control=0,
        pre_periods=pre, post_periods=post, vcov=V,
    ), pre, post


def covers(res, target):
    lo, hi = res.ci_lb, res.ci_ub
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return bool(lo <= target <= hi)


def run(n_units, reps, seed0):
    keys = [("RM", m) for m in MBARS] + [("SD", "precal"), ("SD", "inst")]
    hit = {k: 0 for k in keys}
    seen = {k: 0 for k in keys}
    obs_hit = obs_seen = 0
    mhats = []

    for r in range(reps):
        df = generate_panel(config(n_units, seed0 + r))
        assert abs(df.attrs["true_latent_es"][0][ELL_STAR] - LATENT) < 1e-12
        assert abs(df.attrs["true_meas_es"][0][ELL_STAR] - WEDGE) < 1e-12
        labels, coef, se, vcov = fit(df)
        res_obj, pre, post = container(labels, coef, se, vcov)
        l_vec = np.zeros(len(post))
        l_vec[post.index(ELL_STAR)] = 1.0

        mhat = m_upper_bound_mpre(labels, coef, se, vcov)
        mhats.append(mhat)

        for k in keys:
            fam, par = k
            if fam == "RM":
                method, M = "relative_magnitude", par
            else:
                method, M = "smoothness", (mhat if par == "precal" else WEDGE)
            try:
                out = compute_honest_did(res_obj, method=method, M=float(M),
                                         alpha=0.05, l_vec=l_vec)
            except Exception:
                continue
            c = covers(out, LATENT)
            if c is None:
                continue
            seen[k] += 1
            hit[k] += c
            if k == ("RM", 1.0):
                co = covers(out, LATENT + WEDGE)
                if co is not None:
                    obs_seen += 1
                    obs_hit += co

    def rate(k):
        return (hit[k] / seen[k]) if seen[k] else float("nan")

    return {
        "n_units": n_units, "reps": reps,
        "mean_Mhat_upperBound_Mpre": float(np.mean(mhats)),
        "coverage_latent_RM": {str(m): rate(("RM", m)) for m in MBARS},
        "coverage_latent_SD_precalibrated": rate(("SD", "precal")),
        "coverage_latent_SD_institutional": rate(("SD", "inst")),
        "n_usable": {f"{k[0]}:{k[1]}": seen[k] for k in keys},
        "coverage_observed_effect_RM1": (obs_hit / obs_seen) if obs_seen else float("nan"),
    }


if __name__ == "__main__":
    base = int(os.environ.get("REPS", 200))
    if os.environ.get("BIGN"):
        grid = [(102400, max(40, base // 4)), (409600, max(20, base // 8))]
    else:
        grid = [(400, base), (1600, base), (6400, base),
                (25600, max(80, base // 2)), (102400, max(40, base // 4))]

    print("reference implementation: diff-diff HonestDiD port")
    print(f"latent ATT = {LATENT}, recording wedge = {WEDGE}, target horizon = {ELL_STAR}")
    print(f"{'N':>8} {'reps':>5} {'Mhat':>8} {'RM.5':>7} {'RM1':>7} {'RM2':>7} "
          f"{'SDpre':>7} {'SDinst':>7} {'obsRM1':>7}", flush=True)
    payload = {"runs": [], "latent": LATENT, "wedge": WEDGE, "ell_star": ELL_STAR}
    for n, reps in grid:
        res = run(n, reps, seed0=10_000 + n)
        payload["runs"].append(res)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"{n:>8} {reps:>5} {res['mean_Mhat_upperBound_Mpre']:>8.4f} "
              f"{res['coverage_latent_RM']['0.5']:>7.3f} "
              f"{res['coverage_latent_RM']['1.0']:>7.3f} "
              f"{res['coverage_latent_RM']['2.0']:>7.3f} "
              f"{res['coverage_latent_SD_precalibrated']:>7.3f} "
              f"{res['coverage_latent_SD_institutional']:>7.3f} "
              f"{res['coverage_observed_effect_RM1']:>7.3f}", flush=True)
    print(f"\nwrote {os.path.abspath(OUT)}", flush=True)
