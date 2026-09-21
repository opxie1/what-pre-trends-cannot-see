"""Two design diagnostics the paper (Sections 5 and 4) needs to state precisely.

  A  loading guard   how often the |rho_hat| <= RHO_GUARD category-dropping guard
                     fires over the size grid of scripts/theory_inference.py, and
                     the smallest estimated loading seen
  B  G_eff spread    G_eff(w) over the eigen-directions of [S^{-1}]_PP for the
                     headline design, against the equal-weight contrast

Run:  cd ruler-did && PYTHONPATH=src python scripts/theory_inference_guard.py
Out:  results/inference/loading_guard_check.csv
      results/inference/geff_eigen_sweep.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util
import numpy as np
import pandas as pd

from did_measurement import inference as inf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):  # working
    OUT = os.path.join(ROOT, "results", "theory", "inference")
CKPT = os.path.join(OUT, "checkpoints")
R = int(os.environ.get("TIG_REPS", "400"))
ONLY = os.environ.get("TIG_ONLY", "")

_spec = importlib.util.spec_from_file_location(
    "ti_main", os.path.join(ROOT, "scripts", "theory_inference.py"))
ti = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ti)

GRID = [(G, C) for G in (20, 30, 51, 100) for C in (3, 5, 8)]


def main():
    rows, sweep = [], []
    for G, C in GRID:
        seeds = pd.read_csv(os.path.join(CKPT, f"size_G{G}_C{C}.csv"))["seed"].to_numpy()[:R]
        fired = 0
        if ONLY == "sweep":
            seeds = seeds[:0]
        mn = np.inf
        cmin = C
        for s in seeds:
            df = ti.panel(ti.make_cfg(G=G, T=120, C=C, kappa=0.0,
                                      disc=(0.1, 1.0), seed=int(s)))
            comp = inf.gradient_components(df, outcome="logy", k_pre=ti.K_PRE,
                                           k_post=ti.K_POST, loading="triad")
            fired += int(comp["guard_fired"])
            mn = min(mn, float(np.abs(comp["rho_full"]).min()))
            cmin = min(cmin, int(comp["C"]))
        if ONLY != "sweep":
            rows.append(dict(G=G, C=C, reps=len(seeds), rho_guard=inf.RHO_GUARD,
                             guard_fired=fired, min_abs_rho_hat=mn, min_C_kept=cmin))
            print(rows[-1], flush=True)

        if C == 5:
            s0 = pd.read_csv(os.path.join(CKPT, f"size_G{G}_C{C}.csv"))["seed"].to_numpy()[0]
            df = ti.panel(ti.make_cfg(G=G, T=120, C=C, kappa=0.0,
                                      disc=(0.1, 1.0), seed=int(s0)))
            comp = inf.gradient_components(df, outcome="logy", k_pre=ti.K_PRE,
                                           k_post=ti.K_POST, loading="triad")
            idx = comp["post"]
            V = np.linalg.eigh(comp["Sinv"][np.ix_(idx, idx)])[1]
            ge = [inf.effective_clusters(comp, w=V[:, j]) for j in range(len(idx))]
            geq = inf.effective_clusters(comp)
            sweep.append(dict(
                G=G, C=C, m=len(idx), G_eff_equal_weight=geq,
                G_eff_eigen_min=float(min(ge)), G_eff_eigen_max=float(max(ge)),
                m_max_075_at_equal=inf.max_restrictions(geq, 0.075),
                m_max_075_at_min=inf.max_restrictions(float(min(ge)), 0.075),
                size_m13_at_equal=inf.predicted_wald_size(geq, 13, 0.05, 0.9989, "chi2"),
                size_m13_at_min=inf.predicted_wald_size(float(min(ge)), 13, 0.05, 0.9989, "chi2")))
            print(sweep[-1], flush=True)

    if ONLY != "sweep":
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "loading_guard_check.csv"), index=False)
    pd.DataFrame(sweep).to_csv(os.path.join(OUT, "geff_eigen_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
