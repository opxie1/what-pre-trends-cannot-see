"""Is the one factor premise of the triad normalization credible in the data?

Equation (6.1) is consistent for |theta_c| under a one factor structure, in
which the untreated cross category covariance satisfies Sigma_ck = theta_c
theta_k s^2 for every c != k.  The paper asserts that structure and never
checks it.  This script checks it in all three application designs.

Two statistics, both computed on the same untreated two way demeaned array that
gradient_components builds before it calls estimate_loadings_triad:

  ev21   the ratio of the first to the second eigenvalue of the cross category
         correlation matrix.  One dominant factor makes this large.

  triad  the dispersion of the single triad estimates.  For each category c and
         each pair k < l drawn from the others, a single triad identifies
         log|theta_c| up to a constant as (L_ck + L_cl - L_kl)/2, where L_ck =
         log|Sigma_ck|.  Equation (6.1) averages these.  Under one factor every
         triad gives the same answer, so the within c spread is a direct measure
         of the violation, in the units the estimator works in.  We report the
         largest within c standard deviation over c, and its exponential, which
         is the multiplicative disagreement between triads about rho_c.

Both are referred to a reference distribution generated under an exactly one
factor null fitted to the design, by a circular block bootstrap that keeps
the serial correlation of each series and breaks cross category dependence in
the idiosyncratic part.  An iid cell reference is reported alongside to show
how much the serial correlation matters.

WARNING.  The p-value columns in the output are NOT calibrated and the paper
does not report them.  oa_onefactor_calib.py generates data from a TRUE one
factor model at the dimensions of each design and finds that this reference
rejects between 0.090 and 0.995 on the eigenvalue ratio and between 0.090 and
0.178 on the triad dispersion, against a nominal 0.05.  Fitting the null to the
data biases both references.  The paper reports the descriptive columns only:
communality, ev1_share against the 1/C no-factor floor, and triad_mult.

Writes results/inference/onefactor_diagnostic.csv and .json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from did_measurement.estimators import demean_two_way          # noqa: E402
from did_measurement.inference import estimate_loadings_triad  # noqa: E402

OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):  # working
    OUT = os.path.join(ROOT, "results", "theory", "inference")
PROC = os.path.join(ROOT, "data", "processed")
CRIME_FOCAL = [1, 2, 3, 4, 5, 6]
MED_FOCAL = [1, 2, 3, 4, 5, 6, 7]
MED_ANNUAL = [1, 4, 5, 6, 7]
N_BOOT = 2000
COV_FLOOR = 1e-6


# fix_empirics_lib
def load_crime():
    return pd.read_csv(os.path.join(PROC, "crime_panel.csv"))


def load_medicaid():
    return pd.read_csv(os.path.join(PROC, "medicaid_panel.csv"))


def balanced_subset(df, cats):
    """Rows of df on the (unit, period) cells every one of cats observes."""
    sub = df[df["cat"].isin(cats)]
    keys = None
    for c in cats:
        k = set(map(tuple, sub.loc[sub["cat"] == c, ["unit", "period"]].to_numpy()))
        keys = k if keys is None else (keys & k)
    kk = list(map(tuple, sub[["unit", "period"]].to_numpy()))
    return sub[[x in keys for x in kk]].reset_index(drop=True)


# statistics
def untreated_array(df, cats, outcome):
    """The (n_untreated, C) two way demeaned array the triad estimator sees."""
    dat = balanced_subset(df, cats)
    dat = dat.sort_values(["cat", "unit", "period"])
    blocks = [dat[dat["cat"] == c] for c in sorted(cats)]
    ref = blocks[0]
    unit = ref["unit"].to_numpy()
    period = ref["period"].to_numpy()
    for b in blocks[1:]:
        if not (np.array_equal(b["unit"].to_numpy(), unit)
                and np.array_equal(b["period"].to_numpy(), period)):
            raise ValueError("category blocks do not share cells")
    Y = np.column_stack([b[outcome].to_numpy(float) for b in blocks])
    m = ref["D"].to_numpy(int) == 0
    R = demean_two_way(Y[m], unit[m], period[m])
    return R, unit[m], period[m]


def ev_ratio(Cov):
    """First to second eigenvalue of the implied correlation matrix."""
    s = np.sqrt(np.diag(Cov))
    Cor = Cov / np.outer(s, s)
    w = np.sort(np.linalg.eigvalsh(Cor))[::-1]
    return float(w[0] / w[1]), float(w[0] / w.sum()), w


def triad_spread(Cov):
    """Largest within category standard deviation of the single triad estimates."""
    C = Cov.shape[0]
    scale = float(np.median(np.abs(Cov[~np.eye(C, dtype=bool)])))
    floor = COV_FLOOR * max(scale, 1e-300)
    L = np.log(np.maximum(np.abs(Cov), floor))
    sds = np.empty(C)
    for c in range(C):
        others = [k for k in range(C) if k != c]
        vals = [0.5 * (L[c, k] + L[c, l] - L[k, l])
                for i, k in enumerate(others) for l in others[i + 1:]]
        sds[c] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return float(sds.max()), sds


def communality(Cov):
    """theta_c^2 / Sigma_cc, the share of category c's variance the factor carries.

    theta is taken from the same estimator the paper uses, so this reports what
    the normalization in Definition 2 is actually dividing by.
    """
    rho, _ = estimate_loadings_triad(Cov)
    C = Cov.shape[0]
    off = ~np.eye(C, dtype=bool)
    s2 = max(float(np.mean((Cov / np.outer(rho, rho))[off])), 1e-12)
    theta2 = (rho ** 2) * s2
    return np.clip(theta2 / np.diag(Cov), 0.0, 1.0)


def stats(R):
    Cov = np.cov(R, rowvar=False)
    e21, e1share, w = ev_ratio(Cov)
    ts, _ = triad_spread(Cov)
    h = communality(Cov)
    return {"ev21": e21, "ev1_share": e1share, "triad_sd": ts,
            "triad_mult": float(np.exp(ts)), "eigs": w.tolist(),
            "communality": h, "communality_median": float(np.median(h))}


# factor
def fit_one_factor(R):
    """theta, factor scores and idiosyncratic residuals under one factor."""
    Cov = np.cov(R, rowvar=False)
    rho, _ = estimate_loadings_triad(Cov)  # geometric
    # diagonal
    C = Cov.shape[0]
    off = ~np.eye(C, dtype=bool)
    s2 = float(np.mean((Cov / np.outer(rho, rho))[off]))
    s2 = max(s2, 1e-12)
    theta = rho * np.sqrt(s2)  # sigma_ck
    # variance
    sig2 = np.maximum(np.diag(Cov) - theta ** 2, 1e-12)
    wts = theta / sig2
    F = (R @ wts) / float(wts @ theta)
    E = R - np.outer(F, theta)
    return theta, F, E


def block_indices(n, starts_len, rng, block):
    """Circular block bootstrap row indices of length n, fixed block length."""
    nb = -(-n // block)  # ceil
    starts = rng.integers(0, starts_len, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]) % starts_len
    return idx.ravel()[:n]


def null_draws(R, n_boot=N_BOOT, seed=0, block=None, mode="block"):
    """Reference values of both statistics under an exactly one factor null."""
    rng = np.random.default_rng(seed)
    theta, F, E = fit_one_factor(R)
    n, C = R.shape
    if block is None:
        block = max(2, int(round(n ** (1.0 / 3.0))))
    out = []
    for _ in range(n_boot):
        if mode == "block":
            Fs = F[block_indices(n, n, rng, block)]
            Es = np.column_stack(
                [E[block_indices(n, n, rng, block), c] for c in range(C)])
        else:  # cells
            Fs = F[rng.integers(0, n, n)]
            Es = np.column_stack([E[rng.integers(0, n, n), c] for c in range(C)])
        out.append(stats(np.outer(Fs, theta) + Es))
    return out, block


# designs
def designs():
    cr = load_crime()
    md = load_medicaid()
    return [
        ("crime", "six offense categories", cr, CRIME_FOCAL, "y"),
        ("medicaid annual", "five annually asked conditions", md, MED_ANNUAL, "Y"),
        ("medicaid odd year", "seven conditions, odd years", md, MED_FOCAL, "Y"),
    ]


def main():
    rows = []
    S = {"n_boot": N_BOOT,
         "WARNING": ("the *_p columns are not calibrated references; see "
                     "onefactor_calibration.csv. The paper reports the "
                     "descriptive columns only."),
         "designs": {}}
    for key, label, df, cats, outcome in designs():
        R, unit, period = untreated_array(df, cats, outcome)
        obs = stats(R)
        rec = {"design": key, "label": label, "C": R.shape[1],
               "n_untreated_cells": int(R.shape[0]),
               "n_states": int(len(np.unique(unit))),
               "ev21": round(obs["ev21"], 3),
               "ev1_share": round(obs["ev1_share"], 4),
               "triad_sd": round(obs["triad_sd"], 4),
               "triad_mult": round(obs["triad_mult"], 4),
               "no_factor_ev1_share": round(1.0 / R.shape[1], 4),
               "communality_median": round(obs["communality_median"], 4),
               "communality_max": round(float(obs["communality"].max()), 4),
               "communality": [round(x, 4) for x in obs["communality"]],
               "eigenvalues": [round(x, 4) for x in obs["eigs"]]}
        for mode in ("block", "iid"):
            draws, block = null_draws(R, seed=7, mode=mode)
            ev = np.array([d["ev21"] for d in draws])
            ts = np.array([d["triad_sd"] for d in draws])
            # triad_sd
            rec[mode + "_block_len"] = block if mode == "block" else 1
            rec[mode + "_ev21_p"] = round(float((ev <= obs["ev21"]).mean()), 4)
            rec[mode + "_ev21_q05"] = round(float(np.quantile(ev, 0.05)), 3)
            rec[mode + "_triad_p"] = round(float((ts >= obs["triad_sd"]).mean()), 4)
            rec[mode + "_triad_q95"] = round(float(np.quantile(ts, 0.95)), 4)
        rows.append(rec)
        S["designs"][key] = rec
        print("  %-20s C=%d  n=%d  ev21=%.2f  triad_sd=%.3f  "
              "p(block): ev %.3f / triad %.3f"
              % (key, rec["C"], rec["n_untreated_cells"], rec["ev21"],
                 rec["triad_sd"], rec["block_ev21_p"], rec["block_triad_p"]),
              flush=True)

    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "onefactor_diagnostic.csv"), index=False)
    with open(os.path.join(OUT, "onefactor_diagnostic.json"), "w") as f:
        json.dump(S, f, indent=2)
    print()
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
