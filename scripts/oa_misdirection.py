"""What misspecifying the response shape costs the directed test.

Theorem 3 is stated for a known shape k.  Section 8 proves that running the
test at a wrong shape k~ costs power and not level, and scales the
noncentrality by the cosine of the angle between k~ and k in the inner product
<a,b> = a'V^{-1}b.  Every power cell reported elsewhere in the paper uses the
true shape, so the objection that the gain is measured only under correct
specification is fair.

Two statistics are run, because they have different efficiency ratios and only
one of them is the object Section 8's cosine describes.

  directed      the fixed contrast statistic w'gamma / sqrt(w'Vw) reported in
                Tables 2 and 3.  Its noncentrality is kappa w'k / sqrt(w'Vw),
                so the loss relative to the optimum kappa sqrt(k'V^{-1}k) is
                    ratio = w'k / ( sqrt(w'Vw) sqrt(k'V^{-1}k) ),
                which is at most one by Cauchy Schwarz with equality exactly
                when w is proportional to V^{-1}k.

  directed_gls  the statistic of Theorem 3, k~'V^{-1}gamma / sqrt(k~'V^{-1}k~),
                whose noncentrality is kappa cos(theta) sqrt(k'V^{-1}k) with
                cos(theta) the V^{-1} cosine.  This is the one Section 8's
                formula is about.

Four shapes are run against a flat truth k(l) = 1: the truth itself, a response
growing with horizon, one that fades, and one that switches on only in the
second half of the post window.

Writes results/inference/misdirection.csv and .json.
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
if not os.path.isdir(OUT):  # working
    OUT = os.path.join(ROOT, "results", "theory", "inference")

ALPHA = 0.05
REPS = 600
N_PERM = 999
KAPPAS = (0.15, 0.30, 0.45)
CS = (5, 8)
STATS = ("directed", "directed_gls")
SHAPES = ("flat", "ramp", "decay", "late")


def shapes(m):
    l = np.arange(m, dtype=float)
    return {
        "flat":  np.ones(m),
        "ramp":  (l + 1) / (l + 1).mean(),
        "decay": (1.0 / (l + 1)) / (1.0 / (l + 1)).mean(),
        "late":  np.where(l >= m // 2, 1.0, 0.0) * (m / max(m - m // 2, 1)),
    }


def _cell(args):
    C, kappa, rep = args
    cfg = make_cfg(G=51, T=120, C=C, kappa=kappa, seed=50_000 + rep)
    try:
        comp = inf.gradient_components(panel(cfg), outcome="logy",
                                       k_pre=K_PRE, k_post=K_POST,
                                       loading="triad")
    except Exception:
        return None
    m = len(comp["post"])
    sh = shapes(m)
    out = {}
    for st in STATS:
        for name, w in sh.items():
            r = inf.permutation_test(comp, st, N_PERM, 50_000 + rep, w=w)
            out["p_%s_%s" % (st, name)] = float(r["pvalue"])

    # efficiency
    _, VP = inf._gamma_V_block(comp, comp["delta"][:, None], comp["post"])
    V = VP[0]
    Vi = np.linalg.pinv(V)
    k = sh["flat"]
    info = float(k @ Vi @ k)  # optimum
    root = np.sqrt(max(info, 1e-300))
    for name, w in sh.items():
        wVw = float(w @ V @ w)
        # contrast
        out["ratio_directed_" + name] = (
            float(w @ k) / (np.sqrt(max(wVw, 1e-300)) * root))
        # section
        num = float(w @ Vi @ k)
        den = float(np.sqrt(max((w @ Vi @ w) * info, 1e-300)))
        out["ratio_directed_gls_" + name] = num / den
    out["info"] = info
    return out


def run(kappa):
    jobs = [(C, kappa, r) for C in CS for r in range(REPS)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_cell, jobs, chunksize=8))
    by = {}
    i = 0
    for C in CS:
        by[C] = [x for x in res[i:i + REPS] if x is not None]
        i += REPS
    return by


def main():
    print("size cells (kappa = 0)", flush=True)
    null = run(0.0)
    alts = {}
    for kap in KAPPAS:
        print("power cells (kappa = %.2f)" % kap, flush=True)
        alts[kap] = run(kap)

    rows = []
    z = sps.norm.ppf(1 - ALPHA)
    for C in CS:
        for st in STATS:
            for kap in KAPPAS:
                n0, n1 = null[C], alts[kap][C]
                for name in SHAPES:
                    key = "p_%s_%s" % (st, name)
                    p0 = np.array([x[key] for x in n0])
                    p1 = np.array([x[key] for x in n1])
                    size = float((p0 <= ALPHA + 1e-12).mean())
                    cut = float(np.quantile(p0, ALPHA))
                    pw = float((p1 <= ALPHA + 1e-12).mean())
                    pwa = float((p1 <= cut + 1e-12).mean())
                    rk = "ratio_%s_%s" % (st, name)
                    ratio = float(np.median([x[rk] for x in n1]))
                    info = float(np.median([x["info"] for x in n1]))
                    pred = float(sps.norm.cdf(kap * ratio * np.sqrt(max(info, 0))
                                              - z))
                    rows.append(dict(
                        C=C, statistic=st, kappa=kap, shape=name, reps=len(n1),
                        size=round(size, 4), power=round(pw, 4),
                        power_size_adjusted=round(pwa, 4),
                        efficiency_ratio=round(ratio, 4),
                        predicted_power=round(pred, 4),
                        mcse=round(float(np.sqrt(0.25 / max(len(n1), 1))), 4)))
            print("   C=%d %-13s done" % (C, st), flush=True)

    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "misdirection.csv"), index=False)
    S = {"alpha": ALPHA, "reps": REPS, "kappas": list(KAPPAS),
         "n_perm": N_PERM,
         "rows": rows}
    with open(os.path.join(OUT, "misdirection.json"), "w") as f:
        json.dump(S, f, indent=2)
    print()
    for st in STATS:
        print("==== %s ====" % st)
        print(tab[tab.statistic == st].drop(columns=["statistic", "reps"])
              .to_string(index=False))
        print()


if __name__ == "__main__":
    main()
