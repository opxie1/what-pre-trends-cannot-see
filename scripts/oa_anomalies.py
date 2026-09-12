"""Three findings Section 7 reports without an explanation.

A. The measured chi square size falls from 0.393 at C = 3 to 0.356 at C = 8 at
   G = 51, where equation (5.2) predicts almost no movement.  Section 7
   attributes this to the loading normalisation, "whose estimate improves with
   C", without testing that.  The attribution is testable: rerun the same cells
   with loading="none", which removes the estimated normalisation entirely, and
   see whether the decline survives.

B. The size adjusted power of the two sided label permutation test is not
   monotone in kappa at C = 5.  Two candidate causes are separable.  The
   recording map is a log wedge, so the signal that reaches the gradient is
   log(1 + kappa d_c), concave in kappa; and the permutation reference set at
   C = 5 has only 120 points, so the attainable p values are coarse.  We
   measure the realised gradient signal against kappa to see whether the
   statistic itself turns over, which separates a signal effect from a
   reference set effect.

C. The Kolmogorov Smirnov test of p value uniformity rejects at G = 30 in the
   ideal experiment built to satisfy Assumption 3.  One rejection at the five
   percent level over several cells is what that test does one time in twenty.
   We rerun it over many independent seeds and report the rejection rate.

Writes results/inference/anomalies.csv and .json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from did_measurement import inference as inf          # noqa: E402
from theory_inference import (make_cfg, panel,            # noqa: E402
                              K_PRE, K_POST)

OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):                 # older working tree layout
    OUT = os.path.join(ROOT, "results", "theory", "inference")

ALPHA = 0.05
REPS_A = 2000
REPS_B = 600
N_PERM = 999


# ---------------------------------------------------------------- A
def _a_cell(args):
    C, loading, rep = args
    cfg = make_cfg(G=51, T=120, C=C, kappa=0.0, seed=70_000 + rep)
    try:
        comp = inf.gradient_components(panel(cfg), outcome="logy",
                                       k_pre=K_PRE, k_post=K_POST,
                                       loading=loading)
    except Exception:
        return None
    return float(inf.chi2_wald(comp)["pvalue"])


def run_a():
    rows = []
    jobs = [(C, ld, r) for C in (3, 5, 8) for ld in ("triad", "none")
            for r in range(REPS_A)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_a_cell, jobs, chunksize=16))
    i = 0
    for C in (3, 5, 8):
        for ld in ("triad", "none"):
            p = [x for x in res[i:i + REPS_A] if x is not None]
            i += REPS_A
            size = float((np.array(p) <= ALPHA + 1e-12).mean())
            rows.append(dict(part="A", C=C, loading=ld, reps=len(p),
                             chi2_size=round(size, 4),
                             mcse=round(float(np.sqrt(size * (1 - size)
                                                      / max(len(p), 1))), 4)))
            print("   A  C=%d loading=%-5s chi2 size %.4f (n=%d)"
                  % (C, ld, size, len(p)), flush=True)
    return rows


# ---------------------------------------------------------------- B
def _b_cell(args):
    C, kappa, rep = args
    cfg = make_cfg(G=51, T=120, C=C, kappa=kappa, seed=80_000 + rep)
    try:
        comp = inf.gradient_components(panel(cfg), outcome="logy",
                                       k_pre=K_PRE, k_post=K_POST,
                                       loading="triad")
    except Exception:
        return None
    # realised gradient on the post block at the identity labelling
    gP, VP = inf._gamma_V_block(comp, comp["delta"][:, None], comp["post"])
    g = gP[0]
    V = VP[0]
    r = inf.permutation_test(comp, "wald", N_PERM, 80_000 + rep)
    return {"mean_gamma": float(g.mean()),
            "wald": float(g @ np.linalg.pinv(V) @ g),
            "p": float(r["pvalue"])}


def run_b():
    rows = []
    KS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.80)
    jobs = [(5, k, r) for k in KS for r in range(REPS_B)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_b_cell, jobs, chunksize=8))
    i = 0
    cells = {}
    for k in KS:
        cells[k] = [x for x in res[i:i + REPS_B] if x is not None]
        i += REPS_B
    null_p = np.array([x["p"] for x in cells[0.0]])
    cut = float(np.quantile(null_p, ALPHA))
    for k in KS:
        c = cells[k]
        p = np.array([x["p"] for x in c])
        rows.append(dict(part="B", C=5, kappa=k, reps=len(c),
                         mean_gamma=round(float(np.mean([x["mean_gamma"]
                                                         for x in c])), 5),
                         median_wald=round(float(np.median([x["wald"]
                                                            for x in c])), 3),
                         power_size_adjusted=round(float((p <= cut + 1e-12).mean()), 4),
                         mcse=round(float(np.sqrt(0.25 / max(len(c), 1))), 4)))
        print("   B  kappa=%.2f  mean gamma %.5f  median Wald %8.3f  "
              "size adj power %.3f"
              % (k, rows[-1]["mean_gamma"], rows[-1]["median_wald"],
                 rows[-1]["power_size_adjusted"]), flush=True)
    return rows


# ---------------------------------------------------------------- C
def _c_cell(args):
    G, m, seed = args
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, m))
    Sig = A @ A.T + m * np.eye(m)
    eta = rng.multivariate_normal(np.zeros(m), Sig, size=G)
    bar = eta.mean(0)
    S = np.cov(eta, rowvar=False)
    T2 = G * bar @ np.linalg.solve(S, bar)
    F = (G - m) * T2 / ((G - 1) * m)
    return float(sps.f.sf(F, m, G - m))


def run_c():
    rows = []
    for G in (20, 30, 51, 100):
        m = 13
        if G - m < 1:
            continue
        rej = 0
        trials = 200
        per = 1000
        for t in range(trials):
            ps = [_c_cell((G, m, 900_000 + t * per + j)) for j in range(per)]
            d, pv = sps.kstest(ps, 'uniform')
            rej += pv <= ALPHA
        rate = rej / trials
        rows.append(dict(part="C", G=G, m=m, trials=trials, draws_per_trial=per,
                         ks_rejection_rate=round(rate, 4),
                         mcse=round(float(np.sqrt(ALPHA * (1 - ALPHA) / trials)), 4)))
        print("   C  G=%d  KS rejects uniformity in %.3f of %d independent trials"
              % (G, rate, trials), flush=True)
    return rows


def main():
    print("A: is the chi square decline across C the loading normalisation?",
          flush=True)
    rows = run_a()
    print("B: does the statistic itself turn over in kappa at C = 5?",
          flush=True)
    rows += run_b()
    print("C: how often does the KS test reject in the ideal experiment?",
          flush=True)
    rows += run_c()

    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "anomalies.csv"), index=False)
    with open(os.path.join(OUT, "anomalies.json"), "w") as f:
        json.dump({"alpha": ALPHA, "reps_A": REPS_A, "reps_B": REPS_B,
                   "rows": rows}, f, indent=2)
    print()
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
