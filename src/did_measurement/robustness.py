"""Robustness of the discretion-gradient apparatus to its two weakest inputs.

Two referee-fatal objections are addressed here, both about the running variable
d_c rather than about inference:

  (A6) The recording response is assumed PROPORTIONAL to discretion,
       m_c(l) = d_c y(l).  In the one application with power the data prefer a
       single-category dummy to a line in d (R2 0.957 vs 0.541) and dropping the
       banned category leaves 17.1% of the gradient.  `shape_test` turns that
       informal comparison into a selection-corrected test with an exact
       label-permutation calibration, and `loco_gradient` reports the
       leave-one-category-out profile as a first-class diagnostic rather than a
       robustness footnote.

  (d)  The index itself is never identified from anything.  `index_class_slope`
       reports the gradient over an explicit class of admissible codings, and
       `rank_gradient_test` supplies a statistic that uses only the ORDERING of
       d, so it is invariant to any strictly increasing recoding.

Nothing here needs a cardinal d except the functions that are explicitly about
cardinality, and every p-value is an exact label-permutation p-value with the
resolution floor computed from the realised orbit.
"""

import itertools
import math

import numpy as np


# permutation

def _orbit(stat_fn, y, d):
    """All C! values of stat_fn(y, d_permuted); index 0 is the identity."""
    d = np.asarray(d, float)
    C = len(d)
    perms = list(itertools.permutations(range(C)))
    vals = np.array([stat_fn(y, d[list(p)]) for p in perms], float)
    ident = perms.index(tuple(range(C)))
    vals[[0, ident]] = vals[[ident, 0]]
    return vals


def _perm_pvalue(vals, two_sided=False, tol=1e-9):
    obs = vals[0]
    M = len(vals)
    scale = max(1.0, float(np.max(np.abs(vals[np.isfinite(vals)]))))
    if two_sided:
        t = np.abs(vals)
        obs = abs(obs)
    else:
        t = vals
    p = float(np.mean(t >= obs - tol * scale))
    mx = np.nanmax(t)
    mult = int(np.sum(t >= mx - tol * scale))
    return {"pvalue": p, "stat": float(vals[0]), "n_perm": M,
            "min_pvalue": mult / M, "orbit_multiplicity": mult,
            "n_distinct": int(len(np.unique(np.round(t, 9)))),
            "rejection_attainable_05": bool(mult / M <= 0.05 + 1e-12)}


def _slope(y, d):
    d = np.asarray(d, float)
    dc = d - d.mean()
    s2 = float((dc ** 2).sum())
    if s2 <= 0:
        return 0.0
    return float(dc @ (np.asarray(y, float) - np.mean(y)) / s2)


def _lstsq(X, y):
    """Least squares via the normal equations and a pseudo-inverse.

    numpy's lstsq occasionally fails to converge on the tiny, badly scaled
    designs this module builds inside a C! permutation loop; the pseudo-inverse
    of X'X is stable there and identical whenever lstsq succeeds.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)


def _r2(y, X):
    y = np.asarray(y, float)
    beta = _lstsq(X, y)
    r = y - X @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    return (1.0 - float((r ** 2).sum()) / tss) if tss > 0 else np.nan, beta


# gradient

def rank_gradient_test(y, d, two_sided=False):
    """Exact permutation test of monotone association between y_c and d_c.

    Statistic: Spearman rank correlation.  Invariant to any strictly increasing
    recoding of d, so it is the statistic to report when the cardinal index is
    author-assigned.  Its price is that it cannot distinguish a graded response
    from one concentrated in the highest-d category -- use `shape_test` for that.
    """
    y = np.asarray(y, float)
    ry = np.argsort(np.argsort(y)).astype(float)

    def st(yy, dd):
        rd = np.argsort(np.argsort(dd)).astype(float)
        a, b = ry - ry.mean(), rd - rd.mean()
        den = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
        return float(a @ b / den) if den > 0 else 0.0

    out = _perm_pvalue(_orbit(st, y, d), two_sided=two_sided)
    out["statistic"] = "spearman"
    return out


# alternative

def shape_test(y, d):
    """Is the category profile graded in d, or a dummy for one category?

    Fits, on the C category-level summaries y_c:
        linear   y = a + g d                                   (2 parameters)
        dummy    y = a + h 1{c = c*}, c* chosen to maximise R2  (2 parameters)
    and reports both R2 values.  Because c* is selected on the same data the raw
    comparison is biased toward the dummy; the p-values below fix that by
    calibrating the R2 gap and the encompassing t on d against the exact
    label-permutation orbit, redoing the same selection inside every draw.

    p_dummy_better       : P(R2_dummy - R2_linear >= observed).  SMALL means the
                           response really is concentrated, which refutes A6.
    p_linear_encompassed : P(|t| on d in y ~ 1 + d + 1{c = c*} >= observed).
                           SMALL means the line in d survives conditioning on the
                           dummy, i.e. A6 has content beyond a two-group contrast.
    """
    y = np.asarray(y, float)
    C = len(y)
    d = np.asarray(d, float)
    one = np.ones(C)

    def fits(dd):
        r2_lin, _ = _r2(y, np.column_stack([one, dd]))
        best, cstar = -np.inf, 0
        for c in range(C):
            z = np.zeros(C)
            z[c] = 1.0
            r2, _ = _r2(y, np.column_stack([one, z]))
            if r2 > best:
                best, cstar = r2, c
        z = np.zeros(C)
        z[cstar] = 1.0
        X = np.column_stack([one, dd, z])
        if np.linalg.matrix_rank(X) < X.shape[1] or C <= X.shape[1]:
            tstat = np.nan
        else:
            beta = _lstsq(X, y)
            r = y - X @ beta
            s2 = float((r ** 2).sum()) / (C - X.shape[1])
            XtXi = np.linalg.pinv(X.T @ X)
            se = math.sqrt(max(s2 * XtXi[1, 1], 1e-300))
            tstat = float(beta[1] / se)
        return r2_lin, best, cstar, tstat

    r2_lin, r2_dum, cstar, tstat = fits(d)

    perms = list(itertools.permutations(range(C)))
    diffs, ts = [], []
    for p in perms:
        dd = d[list(p)]
        a, b, _, tt = fits(dd)
        diffs.append(b - a)
        ts.append(abs(tt) if tt == tt else 0.0)
    diffs = np.asarray(diffs, float)
    ts = np.asarray(ts, float)
    ident = perms.index(tuple(range(C)))
    diffs[[0, ident]] = diffs[[ident, 0]]
    ts[[0, ident]] = ts[[ident, 0]]

    pd_ = _perm_pvalue(diffs)
    pt_ = _perm_pvalue(ts)
    if pd_["pvalue"] <= 0.05:
        verdict = "A6 REFUTED: response concentrated in one category"
    elif r2_lin >= r2_dum:
        verdict = "A6 not rejected against the dummy alternative"
    else:
        verdict = "inconclusive: dummy fits better but not beyond chance"
    return {"r2_linear": r2_lin, "r2_dummy": r2_dum, "dummy_category": int(cstar),
            "r2_gap": float(r2_dum - r2_lin),
            "t_on_d_given_dummy": tstat,
            "p_dummy_better": pd_["pvalue"], "p_linear_encompassed": pt_["pvalue"],
            "min_pvalue": pd_["min_pvalue"], "n_perm": pd_["n_perm"],
            "verdict": verdict}


# category

def loco_gradient(y, d):
    """Leave-one-category-out gradient, as a ratio to the full-sample gradient.

    A graded response keeps every ratio near 1.  A one-category response shows
    one ratio near 0 (dropping the carrier) and the rest at or above 1.
    """
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    full = _slope(y, d)
    rows = []
    for c in range(len(y)):
        m = np.ones(len(y), bool)
        m[c] = False
        s = _slope(y[m], d[m])
        rows.append({"dropped": int(c), "d": float(d[c]), "slope": float(s),
                     "ratio": float(s / full) if full != 0 else np.nan})
    ratios = np.array([r["ratio"] for r in rows], float)
    return {"full_slope": float(full), "rows": rows,
            "min_ratio": float(np.nanmin(ratios)), "max_ratio": float(np.nanmax(ratios)),
            "carrier": int(np.nanargmin(ratios)),
            "single_category_driven": bool(np.nanmin(ratios) < 0.5)}


# sensitivity

def index_class_slope(y, d_variants, names=None, weights=None):
    """Gradient slope over an explicit class of admissible discretion codings."""
    y = np.asarray(y, float)
    V = [np.asarray(v, float) for v in d_variants]
    names = list(names) if names is not None else [f"variant_{i}" for i in range(len(V))]
    rows = []
    for nm, v in zip(names, V):
        if weights is None:
            s = _slope(y, v)
        else:
            w = np.asarray(weights, float)
            vc = v - np.average(v, weights=w)
            s = float((w * vc) @ (y - np.average(y, weights=w)) / (w * vc ** 2).sum())
        rows.append({"index": nm, "slope": float(s)})
    sl = np.array([r["slope"] for r in rows], float)
    pos = int((sl > 0).sum())
    neg = int((sl < 0).sum())
    return {"rows": rows, "min": float(sl.min()), "max": float(sl.max()),
            "range_ratio": float(np.abs(sl).max() / max(np.abs(sl).min(), 1e-300)),
            "sign_flips": bool(pos > 0 and neg > 0),
            "frac_same_sign_as_baseline": float(np.mean(np.sign(sl) == np.sign(sl[0])))}


def rank_recoding(d):
    """Pure rank coding of d on [0, 1] -- the coding that uses only the order."""
    d = np.asarray(d, float)
    r = np.argsort(np.argsort(d)).astype(float)
    return r / max(r.max(), 1.0)


def perturbation_ladder(d, sds=(0.05, 0.10, 0.20), n_draws=20, seed=0, clip=(0.0, 1.0)):
    """Random perturbations of an author-assigned index, as an admissible class."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    out, names = [], []
    for s in sds:
        for j in range(n_draws):
            out.append(np.clip(d + rng.normal(0, s, len(d)), clip[0], clip[1]))
            names.append(f"sd{s}_{j}")
    return out, names


# stability

def bridge_stability(decompositions):
    """Mandatory stability check across candidate bridges.

    decompositions : {bridge_name: array of measurement components, one per
    shared focal category}.  A decomposition that is a fact about the data must
    survive the choice of bridge; if signs flip, the pooled measurement share
    must not be quoted.
    """
    names = list(decompositions)
    M = np.array([np.asarray(decompositions[n], float) for n in names])
    base = M[0]
    rows = []
    for i in range(1, len(names)):
        flips = np.sign(M[i]) != np.sign(base)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(base) / np.maximum(np.abs(M[i]), 1e-300)
        rows.append({"bridge": names[i], "n_flip": int(flips.sum()),
                     "frac_flip": float(flips.mean()),
                     "max_abs_ratio": float(np.nanmax(ratio)),
                     "flip_index": np.flatnonzero(flips).tolist()})
    worst = max((r["frac_flip"] for r in rows), default=0.0)
    return {"reference_bridge": names[0], "rows": rows, "worst_frac_flip": worst,
            "passes": bool(worst == 0.0),
            "verdict": ("STABLE" if worst == 0.0 else
                        f"UNSTABLE: up to {worst:.0%} of components flip sign; "
                        "do not quote the pooled measurement share")}
