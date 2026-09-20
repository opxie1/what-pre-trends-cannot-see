"""Does the blindness result survive staggered adoption?

Proposition 3 is proved for a single adoption cohort. The paper's scope section
concedes that the extension to staggered designs is not proved. This script asks
the question the proposition would answer, by running the coverage experiment of
Table 1 unchanged except for the adoption structure: three cohorts adopting at
periods 6, 8 and 10 with 40 percent of units never treated, in place of one
cohort adopting at period 8 with half the units never treated.

Everything else is held fixed, including the estimator, the reference
implementation of the honest confidence sets, the calibration of the smoothness
bound, the latent effect, the recording wedge and the target horizon, so the two
tables are comparable cell for cell.

What the proposition claims, if it extends: the deviation a recording response
induces is identically zero before treatment, so the pre period calibration
cannot see the wedge. Two observable consequences follow. The calibrated bound
Mhat is driven by sampling noise alone and shrinks toward zero as the panel
grows, and coverage of the latent effect collapses while coverage of the
recorded effect does not.

The staggered case is not a foregone conclusion. With several cohorts the
comparisons that identify a given cohort's pre period can involve units that are
already treated, and those units carry the recording wedge. If that channel
leaves a pre period signature, Mhat will not shrink and the blindness result is
weaker here than in the single cohort case. Whichever way it comes out is
reportable; the point of running it is that the paper currently does not know.

Run:  python scripts/oa_staggered.py
Env:  REPS (default 200) controls replications per sample size.
"""
import importlib.util
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
warnings.filterwarnings("ignore")

from did_measurement.dgp import DGPConfig, generate_panel          # noqa: E402

# Reuse the reference run's helpers verbatim so the two experiments cannot drift.
_spec = importlib.util.spec_from_file_location(
    "coverage_zero_ref", os.path.join(HERE, "coverage_zero_ref.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

from diff_diff.honest_did import compute_honest_did                # noqa: E402

K_PRE, K_POST = _ref.K_PRE, _ref.K_POST
ELL_STAR = _ref.ELL_STAR
MBARS = _ref.MBARS
LATENT, WEDGE = _ref.LATENT, _ref.WEDGE

COHORTS = (6, 8, 10)
SHARES = (0.2, 0.2, 0.2)          # 40 percent never treated

OUT = os.path.join(HERE, "..", "results", "coverage", "coverage_staggered.json")


def config(n_units, seed):
    """Table 1's design with the single cohort replaced by three."""
    return DGPConfig(
        n_units=n_units, n_periods=16, cohorts=COHORTS, cohort_shares=SHARES,
        n_categories=1, include_bridge=False, discretion=(1.0,), theta_spread=0.0,
        tau_level=1.0, tau_shape="constant", kappa=WEDGE, kappa_shape="constant",
        map_type="additive", mu_scale=1.0, sigma_u=0.2, seed=seed,
    )


def run(n_units, reps, seed0):
    keys = [("RM", m) for m in MBARS] + [("SD", "precal"), ("SD", "inst")]
    hit = {k: 0 for k in keys}
    seen = {k: 0 for k in keys}
    obs_hit = obs_seen = 0
    mhats = []
    pre_dev = []

    for r in range(reps):
        df = generate_panel(config(n_units, seed0 + r))
        assert abs(df.attrs["true_latent_es"][0][ELL_STAR] - LATENT) < 1e-12
        assert abs(df.attrs["true_meas_es"][0][ELL_STAR] - WEDGE) < 1e-12

        labels, coef, se, vcov = _ref.fit(df)
        res_obj, pre, post = _ref.container(labels, coef, se, vcov)
        l_vec = np.zeros(len(post))
        l_vec[post.index(ELL_STAR)] = 1.0

        mhat = _ref.m_upper_bound_mpre(labels, coef, se, vcov)
        mhats.append(mhat)
        # the largest pre period coefficient in absolute value, which the
        # recording response leaves at zero if the blindness result extends
        pre_dev.append(max(abs(coef[labels.index(l)]) for l in labels if l < 0))

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
            c = _ref.covers(out, LATENT)
            if c is None:
                continue
            seen[k] += 1
            hit[k] += c
            if k == ("RM", 1.0):
                co = _ref.covers(out, LATENT + WEDGE)
                if co is not None:
                    obs_seen += 1
                    obs_hit += co

    def rate(k):
        return (hit[k] / seen[k]) if seen[k] else float("nan")

    return {
        "n_units": n_units, "reps": reps,
        "cohorts": list(COHORTS), "cohort_shares": list(SHARES),
        "mean_Mhat_upperBound_Mpre": float(np.mean(mhats)),
        "mean_max_abs_pre_coef": float(np.mean(pre_dev)),
        "coverage_latent_RM": {str(m): rate(("RM", m)) for m in MBARS},
        "coverage_latent_SD_precalibrated": rate(("SD", "precal")),
        "coverage_latent_SD_institutional": rate(("SD", "inst")),
        "n_usable": {f"{k[0]}:{k[1]}": seen[k] for k in keys},
        "coverage_observed_effect_RM1": (obs_hit / obs_seen) if obs_seen else float("nan"),
    }


if __name__ == "__main__":
    base = int(os.environ.get("REPS", 200))
    maxn = int(os.environ.get("MAXN", 102400))
    grid = [(n, reps) for n, reps in
            [(400, base), (1600, base), (6400, base),
             (25600, max(80, base // 2)), (102400, max(40, base // 4))]
            if n <= maxn]

    print("staggered adoption: cohorts %s, shares %s, never treated %.0f%%"
          % (COHORTS, SHARES, 100 * (1 - sum(SHARES))))
    print("latent ATT = %g, recording wedge = %g, target horizon = %d"
          % (LATENT, WEDGE, ELL_STAR))
    print(f"{'N':>8} {'reps':>5} {'Mhat':>8} {'maxpre':>8} {'RM.5':>7} {'RM1':>7} "
          f"{'RM2':>7} {'SDpre':>7} {'SDinst':>7} {'obsRM1':>7}", flush=True)

    payload = {"runs": [], "latent": LATENT, "wedge": WEDGE,
               "ell_star": ELL_STAR, "cohorts": list(COHORTS),
               "cohort_shares": list(SHARES)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for n, reps in grid:
        res = run(n, reps, seed0=20_000 + n)
        payload["runs"].append(res)
        with open(OUT, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"{n:>8} {reps:>5} {res['mean_Mhat_upperBound_Mpre']:>8.4f} "
              f"{res['mean_max_abs_pre_coef']:>8.4f} "
              f"{res['coverage_latent_RM']['0.5']:>7.3f} "
              f"{res['coverage_latent_RM']['1.0']:>7.3f} "
              f"{res['coverage_latent_RM']['2.0']:>7.3f} "
              f"{res['coverage_latent_SD_precalibrated']:>7.3f} "
              f"{res['coverage_latent_SD_institutional']:>7.3f} "
              f"{res['coverage_observed_effect_RM1']:>7.3f}", flush=True)
    print(f"\nwrote {os.path.abspath(OUT)}", flush=True)
