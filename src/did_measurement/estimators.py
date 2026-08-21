"""Event-study and group-time ATT estimators with cluster-robust inference.

All estimators operate on the long panel produced by dgp.generate_panel or on
real data shaped the same way: columns unit, period, cohort (-1 = never treated),
plus an outcome column. Estimation is per category unless rows are pre-filtered.

TWFE event studies use exact two-way demeaning (alternating projections), then
OLS on event-time dummies with cluster-robust (by unit) covariance. The default
DGP has effects homogeneous across cohorts, where TWFE ES is consistent; the
Callaway-Sant'Anna estimator is provided for heterogeneity-robust comparisons.

Callaway-Sant'Anna here is the unconditional (no covariates) version on cohort
means: ATT(g,t) = [m(g,t) - m(g,g-1)] - [m(ctrl,t) - m(ctrl,g-1)], control being
never-treated or not-yet-treated cohort means. All event-study horizons, leads
included, are long differences relative to g-1. SEs via unit-level bootstrap.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- helpers

def _codes(x):
    _, inv = np.unique(np.asarray(x), return_inverse=True)
    return inv, inv.max() + 1


def _group_demean_pass(A, codes, n_groups):
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    mx = 0.0
    for j in range(A.shape[1]):
        m = np.bincount(codes, weights=A[:, j], minlength=n_groups) / counts
        A[:, j] -= m[codes]
        mx = max(mx, float(np.abs(m).max()))
    return mx


def demean_two_way(A, g1, g2, tol=1e-9, max_iter=500):
    """Alternating projections: remove group means of g1 and g2 from columns of A."""
    A = np.ascontiguousarray(A, dtype=float).copy()
    c1, n1 = _codes(g1)
    c2, n2 = _codes(g2)
    for _ in range(max_iter):
        m1 = _group_demean_pass(A, c1, n1)
        m2 = _group_demean_pass(A, c2, n2)
        if max(m1, m2) < tol:
            break
    return A


def cluster_ols(X, y, cluster_ids):
    """OLS with CR1 cluster-robust covariance. Returns beta, vcov, n, G."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    cl, G = _codes(cluster_ids)
    Xu = X * u[:, None]
    agg = np.zeros((G, k))
    np.add.at(agg, cl, Xu)
    S = agg.T @ agg
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = adj * XtX_inv @ S @ XtX_inv
    return beta, V, n, G


def build_event_dummies(event_time, k_pre, k_post):
    """Event dummies for l in {-k_pre,...,-2, 0,...,k_post}, l=-1 omitted,
    endpoints binned (l <= -k_pre and l >= k_post). NaN event_time -> all zero.
    Returns (X, labels)."""
    ell = np.asarray(event_time, float)
    labels = [l for l in range(-k_pre, k_post + 1) if l != -1]
    X = np.zeros((len(ell), len(labels)))
    fin = np.isfinite(ell)
    e = np.where(fin, ell, 1e9)
    for j, l in enumerate(labels):
        if l == -k_pre:
            X[:, j] = fin & (e <= l)
        elif l == k_post:
            X[:, j] = fin & (e >= l)
        else:
            X[:, j] = fin & (e == l)
    return X, labels


# ---------------------------------------------------------------- TWFE ES

def event_study_twfe(df, outcome="Y", k_pre=5, k_post=6,
                     unit_col="unit", time_col="period", cluster_col=None):
    """TWFE event study on one category (or pre-stacked data with composite FE cols).

    Returns dict with 'table' (DataFrame: ell, coef, se), 'vcov', 'labels'.
    """
    cluster_col = cluster_col or unit_col
    Xd, labels = build_event_dummies(df["event_time"].to_numpy(), k_pre, k_post)
    yA = df[outcome].to_numpy(float)[:, None]
    M = np.column_stack([yA, Xd])
    M = demean_two_way(M, df[unit_col].to_numpy(), df[time_col].to_numpy())
    y_t, X_t = M[:, 0], M[:, 1:]
    beta, V, n, G = cluster_ols(X_t, y_t, df[cluster_col].to_numpy())
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    table = pd.DataFrame({"ell": labels, "coef": beta, "se": se})
    return {"table": table, "vcov": V, "labels": labels, "n": n, "n_clusters": G}


# ---------------------------------------------------------------- CS ATT(g,t)

def _cs_point(df, outcome, control):
    """ATT(g,t) matrix and event-study aggregation from cohort-by-period means."""
    m = df.groupby(["cohort", "period"], sort=True)[outcome].mean().unstack()
    sizes = df.groupby("cohort")["unit"].nunique()
    cohorts = [g for g in m.index if g >= 0]
    periods = list(m.columns)
    out = {}
    for g in cohorts:
        base = g - 1
        if base not in periods:
            continue
        for t in periods:
            if control == "never":
                if -1 not in m.index:
                    raise ValueError("no never-treated units for control='never'")
                ctrl = m.loc[-1, t] - m.loc[-1, base]
            else:  # not-yet-treated: cohorts with G > max(t, base), plus never
                elig = [h for h in m.index if h == -1 or h > max(t, g)]
                if not elig:
                    continue
                w = sizes.loc[elig]
                ctrl = ((m.loc[elig, t] - m.loc[elig, base]) * w).sum() / w.sum()
            out[(g, t)] = (m.loc[g, t] - m.loc[g, base]) - ctrl
    # event-study aggregation, cohort-size weighted
    es = {}
    for (g, t), v in out.items():
        l = t - g
        es.setdefault(l, []).append((v, sizes.loc[g]))
    es_agg = {l: float(np.average([v for v, _ in vw], weights=[w for _, w in vw]))
              for l, vw in es.items()}
    return out, es_agg


def att_gt_cs(df, outcome="Y", control="never", n_boot=0, seed=0):
    """Callaway-Sant'Anna (unconditional) ATT(g,t) + event-study aggregation.

    Returns dict: 'att_gt' {(g,t): est}, 'es' DataFrame(ell, coef[, se]) with SEs
    from a unit-level cluster bootstrap when n_boot > 0.
    """
    att, es = _cs_point(df, outcome, control)
    tab = pd.DataFrame(sorted(es.items()), columns=["ell", "coef"])
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        units = df["unit"].unique()
        gb = dict(tuple(df.groupby("unit", sort=False)))
        draws = {l: [] for l in es}
        for b in range(n_boot):
            samp = rng.choice(units, size=len(units), replace=True)
            parts = []
            for j, uu in enumerate(samp):
                p = gb[uu].copy()
                p["unit"] = j  # relabel so duplicates count separately
                parts.append(p)
            try:
                _, es_b = _cs_point(pd.concat(parts, ignore_index=True), outcome, control)
                for l in draws:
                    if l in es_b:
                        draws[l].append(es_b[l])
            except (ValueError, KeyError):
                continue
        tab["se"] = [np.std(draws[l], ddof=1) if len(draws[l]) > 10 else np.nan
                     for l in tab["ell"]]
    return {"att_gt": att, "es": tab}
