"""Exactness of the label-permutation test when the loading normalisation is not
S_C-equivariant.

Lemma 3 of theory/inference.md requires an order-invariant loading
normalisation, and the fast representation in `inference.permutation_test`
permutes only the centred label vector delta, holding the estimated loadings
fixed.  That is the randomisation distribution of the statistic ONLY when the
normalisation commutes with relabelling.  The log-triad estimator does; the
"mtest_iv" estimator does not, because it takes its reference category to be
argmin(d), so relabelling d moves the reference.  Both applications ship
mtest_iv.

`permutation_refit` computes the honest reference distribution for that case:
it relabels d in the raw panel, re-estimates the loadings, refits, and
recomputes the statistic, for every permutation in the orbit.  It is O(C!)
refits and is meant to be run once per headline specification, not inside a
Monte Carlo.

`equivariance_gap` reports how far the fast representation is from the refit
orbit, so the exactness claim can be stated with a number attached instead of
being asserted.
"""

import itertools
import math

import numpy as np

from .inference import gradient_components, _apply_stat, _orbit_multiplicity


def permutation_refit(df, stat="wald", loading="mtest_iv", idx="post",
                      alternative="greater", w=None, max_perm=5040, **kw):
    """Brute-force label-permutation test: relabel d in the panel and refit.

    df    : the long panel, one row per (unit, period, category).
    kw    : forwarded to `gradient_components` (outcome, k_pre, k_post,
            bridge_cat, use_bridge, cluster_col, rho_guard).
    idx   : "post" | "pre" | an explicit index array into the event-time labels.

    Returns the same fields as `inference.permutation_test`, plus `refit=True`.
    """
    base = gradient_components(df, loading=loading, **kw)
    cats = np.asarray(base["cats"])
    d_map = {int(c): float(v) for c, v in zip(cats, base["d_full"])}
    d_vals = np.array([d_map[int(c)] for c in cats], float)
    C0 = len(cats)
    total = math.factorial(C0)
    if total > max_perm:
        raise ValueError(f"{total} permutations exceeds max_perm={max_perm}")

    cat_col = df["cat"].to_numpy()
    vals = []
    for perm in itertools.permutations(range(C0)):
        dd = d_vals[list(perm)]
        newd = np.empty(len(df), float)
        for j, c in enumerate(cats):
            newd[cat_col == c] = dd[j]
        df2 = df.copy(deep=False)
        df2 = df2.assign(d=newd)
        df2.attrs = {}
        try:
            comp = gradient_components(df2, loading=loading, **kw)
        except ValueError:
            vals.append(np.nan)
            continue
        ii = comp["post"] if idx == "post" else (comp["pre"] if idx == "pre" else np.asarray(idx))
        D = comp["delta"][:, None]
        t = _apply_stat(comp, D, stat, w=w, idx=ii)
        vals.append(float(t[0]))
    vals = np.asarray(vals, float)
    ident = list(itertools.permutations(range(C0))).index(tuple(range(C0)))
    vals[[0, ident]] = vals[[ident, 0]]
    if stat.startswith("directed") and alternative == "less":
        vals = -vals
    good = np.isfinite(vals)
    obs = vals[0]
    p = float(np.mean(vals[good] >= obs - 1e-12))
    mult = _orbit_multiplicity(vals[good])
    return {"stat": float(obs), "pvalue": p, "n_perm": int(good.sum()),
            "n_failed": int((~good).sum()), "exhaustive": True, "refit": True,
            "min_pvalue": mult / int(good.sum()),
            "orbit_multiplicity": mult,
            "n_distinct": int(len(np.unique(np.round(vals[good], 9)))),
            "loading": loading, "draws": vals}


def equivariance_gap(df, stat="wald", loading="mtest_iv", **kw):
    """Distance between the fast (delta-permuting) orbit and the refit orbit.

    Reports the maximum absolute discrepancy between the two sorted orbits, the
    two p-values, and the Spearman rank correlation between them.  A zero gap is
    what `results/theory/inference/summary.json` certifies for loading="triad";
    the point of this function is to attach a number to the case the applications
    actually use.
    """
    from .inference import permutation_test

    base = gradient_components(df, loading=loading, **kw)
    fast = permutation_test(base, stat=stat, keep_draws=True)
    slow = permutation_refit(df, stat=stat, loading=loading, **kw)
    a = np.sort(np.asarray(fast["null_draws"], float))
    b = np.sort(np.asarray(slow["draws"], float))
    n = min(len(a), len(b))
    gap = float(np.max(np.abs(a[:n] - b[:n])))
    ra = np.argsort(np.argsort(a[:n])).astype(float)
    rb = np.argsort(np.argsort(b[:n])).astype(float)
    ca, cb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ca ** 2).sum()) * float((cb ** 2).sum()))
    return {"loading": loading, "max_abs_orbit_gap": gap,
            "p_fast": fast["pvalue"], "p_refit": slow["pvalue"],
            "p_gap": abs(fast["pvalue"] - slow["pvalue"]),
            "spearman_orbits": float(ca @ cb / den) if den > 0 else np.nan,
            "equivariant_claimed": bool(base.get("equivariant", True)),
            "stat": fast["stat"], "n_perm": slow["n_perm"]}
