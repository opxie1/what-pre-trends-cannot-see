"""Studentization under a failure of category exchangeability (Assumption 4).

Two experiments.

(A) The Gaussian limit experiment of Section 5, reproduced exactly, then rerun
    with a studentized statistic.  The limit experiment gives each category one
    scalar draw, so studentizing needs an auxiliary variance estimate; we supply
    omegahat_c = omega_c * chi2_nu / nu, independent across c, which is the
    stylized counterpart of the within-category residual variance in the panel.

(B) The same departure priced in the panel, with the CR1 statistic the paper
    actually uses.  The CR1 meat is quadratic in the realised residuals, so its
    variance estimate already tracks category-specific variances; (B) measures
    whether that is enough.

Writes results/inference/studentized_check.csv and .json.
"""
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from did_measurement import inference as inf          # noqa: E402
from theory_inference import make_cfg, panel          # noqa: E402

OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):                 # older working tree layout
    OUT = os.path.join(ROOT, "results", "theory", "inference")
ALPHA = 0.05
N_SIM_GAUSS = 20_000
RATIOS = (1.0, 2.0, 4.0, 9.0, 25.0)
CS = (5, 8)
PLACES = ("high_d", "low_d")
NUS = (10.0, 50.0, np.inf)
PANEL_REPS = 2000
N_PERM = 999


# ---------------------------------------------------------------- experiment A
def gauss_perm_size(omega, delta, n_sim=N_SIM_GAUSS, stat="wald",
                    studentize=False, nu=np.inf, seed=0):
    """Size of the label permutation test in the Gaussian limit experiment.

    studentize=False reproduces the homoskedastic covariance estimate reported
    in Section 5.  studentize=True divides by a variance estimate carrying a
    separate omegahat_c per category, which is what DiCiccio and Romano (2017)
    prescribe for a regression coefficient.
    """
    rng = np.random.default_rng(seed)
    D, _ = inf._perm_matrix(np.asarray(delta, float), None, rng)
    s2 = float((np.asarray(delta, float) ** 2).sum())
    omega = np.asarray(omega, float)
    sd = np.sqrt(omega)
    D2 = D ** 2                                                   # (C, P)
    v_homo = D2.sum(0) / s2 ** 2                                  # (P,) constant
    rej = 0
    for _ in range(n_sim):
        eps = sd * rng.standard_normal(len(sd))
        g = (eps @ D) / s2                                        # (P,)
        if studentize:
            if np.isinf(nu):
                omh = omega
            else:
                omh = omega * rng.chisquare(nu, size=len(omega)) / nu
            v = (omh @ D2) / s2 ** 2                              # (P,)
        else:
            v = v_homo
        t = g ** 2 / v if stat == "wald" else g / np.sqrt(v)
        rej += int((t >= t[0] - 1e-12).mean() <= ALPHA + 1e-12)
    return rej / n_sim


def run_gauss():
    rows = []
    for C in CS:
        dvec = np.linspace(0.1, 1.0, C)
        dl = dvec - dvec.mean()
        for r in RATIOS:
            for place in PLACES:
                om = np.ones(C)
                om[-1 if place == "high_d" else 0] = r
                for st in ("wald", "directed"):
                    rows.append(dict(
                        experiment="gaussian", C=C, var_ratio=r, inflated=place,
                        stat=st, calibration="homoskedastic", nu="",
                        size=gauss_perm_size(om, dl, stat=st)))
                    for nu in NUS:
                        rows.append(dict(
                            experiment="gaussian", C=C, var_ratio=r,
                            inflated=place, stat=st,
                            calibration="studentized",
                            nu=("inf" if np.isinf(nu) else str(int(nu))),
                            size=gauss_perm_size(om, dl, stat=st,
                                                 studentize=True, nu=nu)))
        print("  gaussian C=%d done" % C, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment B
def _panel_cell(args):
    C, ratio, place, rep = args
    cfg = make_cfg(G=51, T=120, C=C, seed=10_000 + rep)
    sig = np.ones(C + 1)
    if ratio != 1.0:
        # category 0 is the bridge; focal categories run 1..C in ascending d
        sig[C if place == "high_d" else 1] = np.sqrt(ratio)
    cfg = dataclasses.replace(cfg, sigma_eps_by_cat=tuple(sig))
    try:
        comp = inf.gradient_components(panel(cfg), outcome="logy",
                                       loading="triad")
    except Exception:
        return None
    out = {"resid_var_ratio": float(comp["resid_var_ratio"])}
    for st, key in (("wald", "perm_wald"), ("directed", "perm_dir")):
        out["p_" + key] = float(
            inf.permutation_test(comp, st, N_PERM, 10_000 + rep)["pvalue"])
    out["p_chi2"] = float(inf.chi2_wald(comp)["pvalue"])
    return out


def run_panel(reps=PANEL_REPS, workers=None):
    jobs = [(C, r, pl, rep) for C in CS for r in RATIOS for pl in PLACES
            for rep in range(reps)]
    print("  panel jobs: %d" % len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        res = list(ex.map(_panel_cell, jobs, chunksize=16))
    rows = []
    i = 0
    for C in CS:
        for r in RATIOS:
            for pl in PLACES:
                cell = [x for x in res[i:i + reps] if x is not None]
                i += reps
                n = len(cell)
                rv = float(np.median([x["resid_var_ratio"] for x in cell]))
                for key, st in (("perm_wald", "wald"), ("perm_dir", "directed"),
                                ("chi2", "chi2")):
                    p = np.array([x["p_" + key] for x in cell])
                    size = float((p <= ALPHA + 1e-12).mean())
                    rows.append(dict(
                        experiment="panel", C=C, var_ratio=r, inflated=pl,
                        stat=st,
                        calibration=("chi square" if key == "chi2"
                                     else "CR1 label permutation"),
                        nu="", size=size, reps=n,
                        mcse=float(np.sqrt(size * (1 - size) / max(n, 1))),
                        median_resid_var_ratio=rv))
        print("  panel C=%d done" % C, flush=True)
    return pd.DataFrame(rows)


def _series(df, ratios, **kw):
    """The size at each variance ratio for one (C, stat, calibration) cell."""
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return [round(float(q[q["var_ratio"] == r]["size"].iloc[0]), 4)
            for r in ratios]


def main():
    print("experiment A: Gaussian limit", flush=True)
    ga = run_gauss()
    print("experiment B: panel", flush=True)
    pa = run_panel()
    tab = pd.concat([ga, pa], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "studentized_check.csv"), index=False)

    S = {"alpha": ALPHA, "n_sim_gaussian": N_SIM_GAUSS,
         "panel_reps": PANEL_REPS, "n_perm": N_PERM,
         "ratios": list(RATIOS), "gaussian": {}, "panel": {}}
    for C in CS:
        for st in ("wald", "directed"):
            S["gaussian"]["C%d_%s_homoskedastic" % (C, st)] = _series(
                ga, RATIOS, C=C, stat=st, calibration="homoskedastic",
                inflated="high_d")
            for nu in ("inf", "50", "10"):
                S["gaussian"]["C%d_%s_studentized_nu%s" % (C, st, nu)] = _series(
                    ga, RATIOS, C=C, stat=st, calibration="studentized", nu=nu,
                    inflated="high_d")
            S["panel"]["C%d_%s" % (C, st)] = _series(
                pa, RATIOS, C=C, stat=st, inflated="high_d")
            S["panel"]["C%d_%s_low_d" % (C, st)] = _series(
                pa, RATIOS, C=C, stat=st, inflated="low_d")
        S["panel"]["C%d_chi2" % C] = _series(
            pa, RATIOS, C=C, stat="chi2", inflated="high_d")
        q = pa[(pa.C == C) & (pa.stat == "wald") & (pa.inflated == "high_d")]
        S["panel"]["C%d_median_resid_var_ratio" % C] = [
            round(float(q[q.var_ratio == r]["median_resid_var_ratio"].iloc[0]), 3)
            for r in RATIOS]
        S["panel"]["C%d_mcse_max" % C] = round(
            float(pa[(pa.C == C)]["mcse"].max()), 5)
    with open(os.path.join(OUT, "studentized_check.json"), "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
