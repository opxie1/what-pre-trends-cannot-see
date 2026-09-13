"""The proposed calibrations in designs the simulations of Section 7 avoid.

Referee points 9 and 13.  Section 7 puts one unit in every cluster, draws
Gaussian errors independent across periods, and makes the category loadings
exchangeable by construction.  Those choices are favourable to the procedures
the paper proposes and are not the state panels that motivate it.  This script
runs the same calibrations in four harder designs and adds the small sample
cluster robust variance the paper cites but never benchmarks against.

Designs, each at G = 51 clusters and C = 5 categories:

  baseline     one unit per cluster, iid Gaussian errors: Section 7's design
  clustered    five units per cluster with a cluster by period shock, so the
               clustering is doing work rather than decoration
  unequal      the same with cluster sizes spread geometrically from 2 to 16,
               which is the failure mode MacKinnon and Webb identify
  serial       AR(1) errors with rho = 0.7 within unit
  heavy        t5 errors, standardised, so only the tails differ

Variance estimators:

  CR1          the estimator the paper uses throughout
  CR2          the bias reduced linearisation of Bell and McCaffrey, whose
               residual adjustment is (I - H_gg)^{-1/2} per cluster

Calibrations: chi square, the exact Hotelling F of (5.2) at G and at G_eff, the
label permutation test, the restricted wild cluster bootstrap and the cluster
sign change test.

Writes results/inference/robust.csv and .json.
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

from did_measurement import inference as inf                 # noqa: E402
from theory_inference import (make_cfg, panel,               # noqa: E402
                              K_PRE, K_POST)

OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):                 # older working tree layout
    OUT = os.path.join(ROOT, "results", "theory", "inference")

ALPHA = 0.05
REPS = 1000
N_PERM = 999
N_BOOT = 499
N_SIGN = 999
G = 51
C = 5
T = 120

# name -> (config overrides, units, cluster column)
DESIGNS = {
    "baseline":  (dict(), G, "unit"),
    "clustered": (dict(n_clusters=G, sigma_cluster_time=0.6), 5 * G, "cluster"),
    "unequal":   (dict(n_clusters=G, sigma_cluster_time=0.6,
                       cluster_size_dispersion=2.0), 5 * G, "cluster"),
    "serial":    (dict(rho_eps=0.7), G, "unit"),
    "heavy":     (dict(eps_dist="t5"), G, "unit"),
}
VCOVS = ("CR1", "CR2")
METHODS = ("chi2", "hotelling_G", "hotelling_Geff", "perm_wald", "perm_dir",
           "wcb_wald", "signflip", "signflip_dir")


def _cell(args):
    name, vcov, rep = args
    over, n_units, ccol = DESIGNS[name]
    cfg = make_cfg(G=n_units, T=T, C=C, kappa=0.0, seed=120_000 + rep)
    cfg = dataclasses.replace(cfg, **over)
    try:
        df = panel(cfg)
        comp = inf.gradient_components(df, outcome="logy", k_pre=K_PRE,
                                       k_post=K_POST, loading="triad",
                                       cluster_col=ccol, vcov=vcov)
    except Exception:
        return None
    out = {"G_eff": float(inf.effective_clusters(comp)),
           "G_used": int(comp["G"]), "capped": int(comp.get("cr2_capped", 0))}
    try:
        out["p_chi2"] = float(inf.chi2_wald(comp)["pvalue"])
        out["p_hotelling_G"] = float(inf.hotelling_wald(comp)["pvalue"])
        out["p_hotelling_Geff"] = float(
            inf.hotelling_wald(comp, G_eff=out["G_eff"])["pvalue"])
        out["p_perm_wald"] = float(
            inf.permutation_test(comp, "wald", N_PERM, 120_000 + rep)["pvalue"])
        out["p_perm_dir"] = float(
            inf.permutation_test(comp, "directed", N_PERM, 120_000 + rep)["pvalue"])
        out["p_wcb_wald"] = float(
            inf.wild_cluster_bootstrap(comp, N_BOOT, 120_000 + rep, "wald")["pvalue"])
        out["p_signflip"] = float(
            inf.signflip_test(comp, N_SIGN, 120_000 + rep, "wald")["pvalue"])
        out["p_signflip_dir"] = float(
            inf.signflip_test(comp, N_SIGN, 120_000 + rep, "directed")["pvalue"])
    except Exception:
        return None
    return out


def main():
    jobs = [(n, v, r) for n in DESIGNS for v in VCOVS for r in range(REPS)]
    print("cells: %d designs x %d vcov x %d reps = %d panels"
          % (len(DESIGNS), len(VCOVS), REPS, len(jobs)), flush=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_cell, jobs, chunksize=8))

    rows = []
    i = 0
    for name in DESIGNS:
        for v in VCOVS:
            cell = [x for x in res[i:i + REPS] if x is not None]
            i += REPS
            n = len(cell)
            if n == 0:
                print("   *** %s %s: every replication failed" % (name, v))
                continue
            geff = float(np.median([x["G_eff"] for x in cell]))
            gused = int(np.median([x["G_used"] for x in cell]))
            capped = int(np.sum([x["capped"] for x in cell]))
            for m in METHODS:
                p = np.array([x["p_" + m] for x in cell])
                size = float((p <= ALPHA + 1e-12).mean())
                rows.append(dict(
                    design=name, vcov=v, method=m, reps=n,
                    size=round(size, 4),
                    mcse=round(float(np.sqrt(size * (1 - size) / n)), 4),
                    G_used=gused, G_eff=round(geff, 2), cr2_capped=capped))
            print("   %-10s %s  G=%d G_eff=%.1f  chi2 %.3f  perm %.3f  "
                  "wcb %.3f  sign %.3f"
                  % (name, v, gused, geff,
                     rows[-8]["size"], rows[-5]["size"],
                     rows[-3]["size"], rows[-2]["size"]), flush=True)

    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "robust.csv"), index=False)
    with open(os.path.join(OUT, "robust.json"), "w") as f:
        json.dump({"alpha": ALPHA, "reps": REPS, "G": G, "C": C, "T": T,
                   "rows": rows}, f, indent=2)
    print()
    piv = tab.pivot_table(index=["design", "vcov"], columns="method",
                          values="size")
    print(piv.to_string())


if __name__ == "__main__":
    main()
