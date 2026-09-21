"""Calibration check for the one factor diagnostic of oa_onefactor.py.

The diagnostic refers two statistics to a block bootstrap reference fitted
under a one factor null.  If that reference is right, then data generated from
a TRUE one factor model should reject at about the nominal rate.  This script
generates such data at the dimensions of each application design, including a
serially correlated factor and serially correlated idiosyncratic errors, and
reports the realised rejection rate of both statistics.

Writes results/inference/onefactor_calibration.csv.
"""
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

import oa_onefactor as OF                                       # noqa: E402

OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):  # working
    OUT = os.path.join(ROOT, "results", "theory", "inference")
REPS = 400
N_BOOT = 400  # bootstrap
ALPHA = 0.05


def ar1(n, rho, rng):
    e = rng.standard_normal(n)
    if rho == 0.0:
        return e
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def one_factor_draw(n, theta, sig, rho_f, rho_e, rng):
    """A true one factor array with AR(1) factor and AR(1) idiosyncratic parts."""
    F = ar1(n, rho_f, rng)
    E = np.column_stack([sig[c] * ar1(n, rho_e, rng) for c in range(len(theta))])
    return np.outer(F, theta) + E


def _one_rep(args):
    """One replication: draw true one factor data, refer it to its own reference."""
    n, theta, sig, rho_f, rho_e, r = args
    rng = np.random.default_rng(2024 + r)
    R = one_factor_draw(n, theta, sig, rho_f, rho_e, rng)
    obs = OF.stats(R)
    draws, _ = OF.null_draws(R, n_boot=N_BOOT, seed=r, mode="block")
    ev = np.array([d["ev21"] for d in draws])
    ts = np.array([d["triad_sd"] for d in draws])
    return (float((ev <= obs["ev21"]).mean()) <= ALPHA,
            float((ts >= obs["triad_sd"]).mean()) <= ALPHA)


def main():
    # loading
    cases = [
        ("crime", 1675, np.array([0.62, 0.48, 0.93, 0.84, 0.37, 0.64]),
         np.array([0.78, 0.88, 0.37, 0.54, 0.93, 0.77])),
        ("medicaid annual", 333, np.array([0.19, 0.47, 0.44, 0.13, 0.35]),
         np.array([0.98, 0.88, 0.90, 0.99, 0.94])),
        ("medicaid odd year", 183, np.array([0.36, 0.32, 0.30, 0.13, 0.26,
                                             0.06, 0.15]),
         np.array([0.93, 0.95, 0.96, 0.99, 0.97, 1.00, 0.99])),
    ]
    rows = []
    for name, n, theta, sig in cases:
        for rho_f, rho_e in ((0.0, 0.0), (0.7, 0.5)):
            jobs = [(n, theta, sig, rho_f, rho_e, r) for r in range(REPS)]
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
                res = list(ex.map(_one_rep, jobs, chunksize=4))
            rej_ev = sum(a for a, _ in res)
            rej_tr = sum(b for _, b in res)
            rows.append(dict(
                design=name, n=n, C=len(theta), rho_factor=rho_f,
                rho_idio=rho_e, reps=REPS,
                ev21_rejection=round(rej_ev / REPS, 4),
                triad_rejection=round(rej_tr / REPS, 4),
                mcse=round(float(np.sqrt(ALPHA * (1 - ALPHA) / REPS)), 4)))
            print("  %-18s rho_f=%.1f rho_e=%.1f  ev21 rejects %.3f  "
                  "triad rejects %.3f" % (name, rho_f, rho_e,
                                          rows[-1]["ev21_rejection"],
                                          rows[-1]["triad_rejection"]),
                  flush=True)
    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "onefactor_calibration.csv"), index=False)
    print()
    print(tab.to_string(index=False))
    with open(os.path.join(OUT, "onefactor_calibration.json"), "w") as f:
        json.dump({"alpha": ALPHA, "reps": REPS, "n_boot": N_BOOT,
                   "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
