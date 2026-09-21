"""Finite-sample inference for the discretion-gradient test (theory/inference.md).

The pooled gradient regression of mtest.discretion_gradient_test stacks the C
outcome categories, absorbs unit-by-category and period-by-category fixed
effects, and regresses on event-time dummies plus event-time x centred
discretion interactions.  With a panel balanced across categories that
regression is *exactly* separable: writing b_c for the within-category
event-study coefficient vector and delta_c = d_c - mean(d),

    beta_hat = (1/C) sum_c b_c ,   gamma_hat = sum_c delta_c b_c / sum_c delta_c^2 ,

and the cluster-robust score of the gamma block for cluster g is exactly
sum_c delta_c Xt_g' e_gc, with e_c the *within-category* residual (Lemmas 1-2
of theory/inference.md).  Everything below is built on those category-level
"components", so label permutation, wild cluster bootstrap and cluster
sign-flip inference cost O(C) or O(G) per draw instead of a refit.

Procedures
  chi2_wald / f_wald      status quo (chi-square; and the stat/q ~ F(q,G-1) patch)
  permutation_test        randomisation over the discretion labels.  EXACT in
                          finite samples under category exchangeability
                          (Theorem 1) -- for ANY number of clusters
  wild_cluster_bootstrap  restricted (null-imposed) Rademacher wild cluster
                          bootstrap on the partialled-out regression
  signflip_test           cluster sign-change randomisation on restricted
                          scores (Canay-Romano-Shaikh approximate symmetry)

Statistics: "wald" (two-sided quadratic), "directed" (one-sided; locally most
powerful invariant against a known recording-path shape), "chibar" (one-sided
cone / chi-bar-squared LRT for an unknown non-negative shape).

Design and power tools: design_information, noncentrality, mde_kappa,
local_power_wald, local_power_permutation.
"""

import itertools
import math

import numpy as np
from scipy import stats, optimize

from .estimators import build_event_dummies, demean_two_way

__all__ = [
    "gradient_components", "estimate_loadings_triad",
    "chi2_wald", "f_wald", "permutation_test", "wild_cluster_bootstrap",
    "signflip_test", "gradient_inference",
    "design_information", "noncentrality", "chi2_noncentrality_for_power",
    "mde_kappa", "local_power_wald", "local_power_permutation",
    "rho_orbit", "nearest_label_correlation", "label_resolution",
    "perm_power_directed", "perm_inflation", "mde_kappa_permutation",
    "perm_beat_probability", "perm_power_markov_bound",
    "effective_clusters", "hotelling_wald", "predicted_wald_size",
]

RHO_GUARD = 0.1  # loading
COV_FLOOR = 1e-6  # covariance


# loading

def estimate_loadings_triad(Cov):
    """Relative loadings from the C x C untreated cross-category covariance.

    One-factor structure gives Cov_{ck} = theta_c theta_k s^2 for c != k, hence
    2 log|theta_c| = log|Cov_ck| + log|Cov_cm| - log|Cov_km| for any k != m both
    != c.  Averaging over all admissible pairs,

        a_c = mean_{k != c} L_ck - (1/2) mean_{k != m, both != c} L_km ,  L = log|Cov|,
        rho_c  propto  exp(a_c) .

    This estimator is *equivariant*: relabelling the categories relabels rho
    identically.  There is no reference category and no ordering convention,
    which is exactly what makes the label-permutation test exact (Lemma 3).
    Requires C >= 3 and loadings of a common sign (taken positive).  The
    returned vector has geometric mean one; only its direction matters, because
    all three statistics below are invariant to a common rescaling of the
    normalised outcomes.

    Returns (rho, share of negative off-diagonal covariances) -- the latter is a
    diagnostic for the common-sign assumption.
    """
    Cov = np.asarray(Cov, float)
    C = Cov.shape[0]
    off = ~np.eye(C, dtype=bool)
    if C < 3:
        return np.ones(C), 0.0
    A = np.abs(Cov)
    scale = A[off].max()
    A = np.maximum(A, COV_FLOOR * max(scale, 1e-300))
    L = np.log(A)
    neg_share = float((Cov[off] < 0).mean())
    a = np.empty(C)
    for c in range(C):
        idx = np.array([j for j in range(C) if j != c])
        sub = L[np.ix_(idx, idx)]
        m = ~np.eye(len(idx), dtype=bool)
        a[c] = L[c, idx].mean() - 0.5 * sub[m].mean()
    a -= a.mean()
    return np.exp(a), neg_share


def _cr2_transform(W, Sinv, E, tol=1e-10):
    """Apply the Bell-McCaffrey CR2 adjustment A_g = (I - H_gg)^{-1/2} to E.

    H_gg = W Sinv W' has rank at most q << n_g, so the inverse square root is
    built from the thin decomposition rather than from a dense n_g by n_g
    eigendecomposition.  With L = W Sinv^{1/2} = P diag(s) Q', H_gg = P diag(s^2) P'
    and (I - H_gg)^{-1/2} = I + P diag((1 - s^2)^{-1/2} - 1) P', which acts as the
    identity off the column space of W.

    Leverage of one or more would make the adjustment undefined; such directions
    are left unadjusted and counted in the returned `capped`.
    """
    w, U = np.linalg.eigh((Sinv + Sinv.T) / 2.0)
    w = np.where(w > tol * max(w.max(), tol), w, 0.0)
    Sinv_half = U @ (np.sqrt(w)[:, None] * U.T)
    L = W @ Sinv_half
    P, s, _ = np.linalg.svd(L, full_matrices=False)
    h = s ** 2
    capped = int((h >= 1.0 - 1e-12).sum())
    f = np.where(h < 1.0 - 1e-12, 1.0 / np.sqrt(np.maximum(1.0 - h, 1e-300)) - 1.0, 0.0)
    return E + P @ (f[:, None] * (P.T @ E)), capped


# extraction

def gradient_components(df, outcome="Y", k_pre=5, k_post=6, bridge_cat=0,
                        use_bridge=False, cluster_col="unit",
                        loading="triad", rho_guard=RHO_GUARD,
                        vcov="CR1"):
    """Category-level sufficient objects for the pooled gradient regression.

    Requires a panel balanced across categories: every category must be observed
    on the same set of (unit, period) cells (Assumption B of theory/inference.md).

    loading : "triad"    equivariant log-triad estimator (default)
              "none"     no normalisation (exact under exchangeable random loadings)
              "mtest_iv" reproduces mtest.discretion_gradient_test exactly

    Returns dict with
      B    (q, C)    within-category event-study coefficients b_c
      A    (G, q, C) cluster-by-category scores Xt_g' e_gc
      Sg   (G, q, q) cluster Gram matrices Xt_g' Xt_g
      S, Sinv (q, q) full Gram matrix and its pseudo-inverse
      d, delta (C,)  discretion index and its centred version
      labels, post, pre   event-time labels and index blocks
      rho, adj, n, G, n_cells, q, C, resid_var (C,)
    """
    cols = ["unit", "period", "cat", "d", "event_time", "D", outcome]
    if cluster_col not in cols:
        cols.append(cluster_col)
    if getattr(df, "attrs", None):
        # dictionaries
        df = df.copy(deep=False)
        df.attrs = {}
    dat = df if use_bridge else df[df["cat"] != bridge_cat]
    dat = dat.loc[:, cols].sort_values(["cat", cluster_col, "unit", "period"])

    cats = np.sort(dat["cat"].unique())
    C0 = len(cats)
    n_cells = len(dat) // C0
    if len(dat) != C0 * n_cells:
        raise ValueError("panel is not balanced across categories")

    blocks = [dat[dat["cat"] == c] for c in cats]
    ref = blocks[0]
    unit = ref["unit"].to_numpy()
    period = ref["period"].to_numpy()
    clus = ref[cluster_col].to_numpy()
    for b in blocks[1:]:
        if not (np.array_equal(b["unit"].to_numpy(), unit)
                and np.array_equal(b["period"].to_numpy(), period)):
            raise ValueError("category blocks do not share the same (unit, period) cells")
    d_all = np.array([float(b["d"].iloc[0]) for b in blocks])
    Y = np.column_stack([b[outcome].to_numpy(float) for b in blocks])  # (n_cells, C)
    Dind = ref["D"].to_numpy(int)
    ev = ref["event_time"].to_numpy(float)

    # loading
    neg_share = 0.0
    if loading == "none" or C0 < 3:
        rho = np.ones(C0)
    elif loading == "triad":
        m = Dind == 0
        R = demean_two_way(Y[m], unit[m], period[m])
        rho, neg_share = estimate_loadings_triad(np.cov(R, rowvar=False))
    elif loading == "mtest_iv":
        from .bridge import estimate_relative_loadings
        ref_cat = int(cats[int(np.argmin(d_all))])
        ld = estimate_relative_loadings(dat, outcome=outcome, bridge_cat=ref_cat)
        rho = np.array([1.0 if int(c) == ref_cat else float(ld["rho"][c]) for c in cats])
        neg_share = float((rho < 0).mean())
    else:
        raise ValueError(f"unknown loading option {loading!r}")

    keep = np.abs(rho) > rho_guard
    if keep.sum() < 3:
        raise ValueError("fewer than three categories survive the loading guard")
    rho_full = rho.copy()
    dropped = [{"cat": int(cats[i]), "d": float(d_all[i]), "rho": float(rho_full[i])}
               for i in range(C0) if not keep[i]]
    Y = Y[:, keep] / rho[keep]
    d = d_all[keep]
    rho = rho[keep]
    C = int(keep.sum())

    # demean
    X, labels = build_event_dummies(ev, k_pre, k_post)
    q = X.shape[1]
    M = demean_two_way(np.column_stack([X, Y]), unit, period)
    Xt, Yt = M[:, :q], M[:, q:]

    S = Xt.T @ Xt
    Sinv = np.linalg.pinv(S)
    B = Sinv @ (Xt.T @ Yt)  # (q, C)
    E = Yt - Xt @ B  # (n_cells, C)

    # cluster
    order = np.argsort(clus, kind="stable")
    Xs, Es, cs = Xt[order], E[order], clus[order]
    cl = np.unique(cs)
    G = len(cl)
    bounds = np.append(np.searchsorted(cs, cl), len(cs))
    A = np.empty((G, q, C))
    Sg = np.empty((G, q, q))
    n_capped = 0
    for g in range(G):
        sl = slice(bounds[g], bounds[g + 1])
        Eg = Es[sl]
        if vcov == "CR2":
            Eg, c = _cr2_transform(Xs[sl], Sinv, Eg)
            n_capped += c
        elif vcov != "CR1":
            raise ValueError(f"unknown vcov {vcov!r}")
        A[g] = Xs[sl].T @ Eg
        Sg[g] = Xs[sl].T @ Xs[sl]

    delta = d - d.mean()
    post = np.array([j for j, l in enumerate(labels) if l >= 0])
    pre = np.array([j for j, l in enumerate(labels) if l < -1])
    n = C * n_cells
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2 * q, 1))
    if vcov == "CR2":
        adj = 1.0  # residual
    n_unit = len(np.unique(unit))
    n_time = len(np.unique(period))
    dfree = max(n_cells - q - n_unit - n_time + 1, 1)
    resid_var = (E ** 2).sum(0) / dfree

    return {"B": B, "A": A, "Sg": Sg, "S": S, "Sinv": Sinv,
            "d": d, "delta": delta, "labels": labels, "post": post, "pre": pre,
            "rho": rho, "neg_cov_share": neg_share, "adj": adj, "n": n, "G": G,
            "n_cells": n_cells, "q": q, "C": C, "resid_var": resid_var,
            "clusters": cl,
            # guard
            "vcov": vcov, "cr2_capped": n_capped,
            "loading": loading, "rho_full": rho_full, "d_full": d_all,
            "cats": np.asarray(cats), "kept": keep, "dropped": dropped,
            "guard_fired": bool((~keep).any()), "C_full": int(C0),
            "equivariant": bool(loading in ("triad", "none")),
            "resid_var_ratio": float(resid_var.max() / max(resid_var.min(), 1e-300))}


# statistics

def _gamma_V_block(comp, D, idx):
    """gamma_hat and CR1 covariance on index block `idx`, batched over labels.

    D : (C, P) array of centred label vectors.  Returns gP (P,m), VP (P,m,m).
    """
    D = np.asarray(D, float)
    if D.ndim == 1:
        D = D[:, None]
    s2 = (D ** 2).sum(0)  # note
    gP = (comp["B"][idx] @ D / s2).T  # (P, m)
    Z = np.einsum("gqc,cp->pgq", comp["A"], D, optimize=True)  # (P,G,q)
    Zi = Z @ comp["Sinv"][:, idx]  # (P,G,m)
    meat = np.einsum("pgm,pgn->pmn", Zi, Zi, optimize=True)
    VP = comp["adj"] * meat / (s2 ** 2)[:, None, None]
    return gP, VP


def _quad_pinv(gP, VP):
    """Batched g' pinv(V) g."""
    w, U = np.linalg.eigh(VP)
    tol = np.maximum(w[:, -1:], 0) * VP.shape[-1] * np.finfo(float).eps
    inv = np.where(w > np.maximum(tol, 1e-300), 1.0 / np.where(w > 0, w, 1.0), 0.0)
    y = np.einsum("pmn,pm->pn", U, gP, optimize=True)
    return (inv * y ** 2).sum(1)


def _stat_directed(gP, VP, w):
    """One-sided statistic w'gamma / sd(w'gamma) with a FIXED contrast w."""
    num = gP @ w
    den = np.sqrt(np.maximum(np.einsum("i,pij,j->p", w, VP, w, optimize=True), 1e-300))
    return num / den


def _stat_directed_gls(gP, VP, k):
    """One-sided LMPI statistic k'V^{-1}gamma / sqrt(k'V^{-1}k) (Theorem 8).

    This is the locally most powerful invariant test against the rank-one
    alternative gamma = kappa k; the fixed-contrast version above is its
    identity-metric analogue and is optimal only when V is proportional to the
    identity on the post block.
    """
    w, U = np.linalg.eigh(VP)
    tol = np.maximum(w[:, -1:], 0) * VP.shape[-1] * np.finfo(float).eps
    inv = np.where(w > np.maximum(tol, 1e-300), 1.0 / np.where(w > 0, w, 1.0), 0.0)
    yk = np.einsum("pmn,n->pm", np.swapaxes(U, 1, 2), k, optimize=True)  # note
    yg = np.einsum("pmn,pn->pm", np.swapaxes(U, 1, 2), gP, optimize=True)
    num = (inv * yk * yg).sum(1)
    den = np.sqrt(np.maximum((inv * yk ** 2).sum(1), 1e-300))
    return num / den


def _stat_chibar(gP, VP):
    """Chi-bar-squared (orthant LRT) statistic: the drop in the V^{-1} quadratic
    form obtained by projecting gamma_hat onto the non-negative orthant."""
    P, m = gP.shape
    out = np.empty(P)
    w, U = np.linalg.eigh(VP)
    for p in range(P):
        ev = np.clip(w[p], 1e-14, None)
        R = (U[p] / np.sqrt(ev)) @ U[p].T  # note
        g = gP[p]
        rg = R @ g
        full = float(rg @ rg)
        h, _ = optimize.nnls(R, rg)
        res = rg - R @ h
        out[p] = full - float(res @ res)
    return out


def _stat_from(gP, VP, stat, w=None):
    """Dispatch on a batch of (gamma_post, V_post) pairs."""
    m = gP.shape[1]
    if stat == "wald":
        return _quad_pinv(gP, VP)
    if stat == "directed":
        return _stat_directed(gP, VP, np.ones(m) if w is None else np.asarray(w, float))
    if stat == "directed_gls":
        return _stat_directed_gls(gP, VP, np.ones(m) if w is None else np.asarray(w, float))
    if stat == "chibar":
        return _stat_chibar(gP, np.ascontiguousarray(VP))
    raise ValueError(f"unknown stat {stat!r}")


def _apply_stat(comp, D, stat, w=None, idx=None):
    idx = comp["post"] if idx is None else idx
    gP, VP = _gamma_V_block(comp, D, idx)
    return _stat_from(gP, VP, stat, w)


# status

def chi2_wald(comp, idx=None):
    """Wald statistic on the post-period interactions with a chi-square p-value."""
    idx = comp["post"] if idx is None else idx
    t = float(_apply_stat(comp, comp["delta"], "wald", idx=idx)[0])
    q = len(idx)
    return {"stat": t, "df": q, "pvalue": float(stats.chi2.sf(t, q))}


def f_wald(comp, idx=None):
    """Usual small-sample patch: stat/q referred to F(q, G-1)."""
    r = chi2_wald(comp, idx=idx)
    q, G = r["df"], comp["G"]
    return {"stat": r["stat"] / q, "df": (q, G - 1),
            "pvalue": float(stats.f.sf(r["stat"] / q, q, max(G - 1, 1)))}


def effective_clusters(comp, idx=None, w=None):
    """Carter-Schnepel-Steigerwald effective cluster count for the gamma block.

    gamma_g = a' S_g a with a = S^- [:, idx] w the design weight that cluster g
    carries in Var(w'gamma_hat) under homoskedasticity; the effective count is
    (sum_g gamma_g)^2 / sum_g gamma_g^2, equal to G when every cluster carries
    the same weight and strictly smaller otherwise.  Never-treated clusters
    carry almost no weight, so G_eff << G in a design with a large
    never-treated share.  Proposition 10 of theory/inference.md.
    """
    idx = comp["post"] if idx is None else idx
    m = len(idx)
    w = np.ones(m) if w is None else np.asarray(w, float)
    a = comp["Sinv"][:, idx] @ w
    gam = np.einsum("i,gij,j->g", a, comp["Sg"], a, optimize=True)
    s1, s2 = gam.sum(), (gam ** 2).sum()
    return float(s1 ** 2 / s2) if s2 > 0 else float(comp["G"])


def hotelling_wald(comp, idx=None, G_eff=None):
    """CR1 Wald referred to the exact Hotelling F law (Proposition 10).

    Under Gaussian, equally weighted cluster scores the CR1 Wald statistic
    satisfies W = omega2 * T^2 with omega2 = (n-2q)/(n-1) and T^2 Hotelling's
    with nu = G-1 degrees of freedom, so W (nu-m+1)/(omega2 nu m) ~ F(m, nu-m+1).
    Pass G_eff (e.g. effective_clusters(comp)) to use the effective count.
    """
    idx = comp["post"] if idx is None else idx
    m = len(idx)
    W = float(_apply_stat(comp, comp["delta"], "wald", idx=idx)[0])
    Ge = float(comp["G"] if G_eff is None else G_eff)
    n, q = comp["n"], comp["q"]
    om2 = (n - 2 * q) / (n - 1)
    nu = Ge - 1.0
    d2 = nu - m + 1.0
    if d2 < 1.0:
        return {"stat": W, "F": float("inf"), "df": (m, d2), "G_eff": Ge,
                "pvalue": 1.0, "degenerate": True}
    F = W * d2 / (om2 * nu * m)
    return {"stat": W, "F": float(F), "df": (m, d2), "G_eff": Ge,
            "pvalue": float(stats.f.sf(F, m, d2)), "degenerate": False}


def predicted_wald_size(G, m, alpha=0.05, omega2=1.0, calibration="chi2"):
    """Closed-form null rejection rate of the mis-calibrated CR1 Wald.

    Proposition 10(iii): with W = omega2 T^2 and T^2 ~ ((nu m)/(nu-m+1)) F,
    nu = G-1, the chi-square-calibrated test has size
    P(F(m, G-m) > (G-m) chi2_{m,1-alpha} / (omega2 (G-1) m)); the stat/m ~
    F(m, G-1) patch has size P(F(m,G-m) > (G-m) F_{m,G-1,1-alpha}/(omega2 (G-1))).
    G may be non-integer (an effective cluster count).
    """
    nu = float(G) - 1.0
    d2 = nu - m + 1.0
    if d2 < 1.0:
        return float("nan")
    if calibration == "chi2":
        crit = stats.chi2.ppf(1 - alpha, m)
        thr = crit * d2 / (omega2 * nu * m)
    elif calibration == "f":
        crit = stats.f.ppf(1 - alpha, m, max(nu, 1.0))
        thr = crit * d2 / (omega2 * nu)
    else:
        raise ValueError(calibration)
    return float(stats.f.sf(thr, m, d2))


def wald_size_expansion(G, m, alpha=0.05):
    """Small-m/G expansion of `predicted_wald_size` (Corollary 11(iii)).

    size = alpha + c f_m(c) (m + c) / (2 (G-1)) + O(G^-2), c = chi2_{m,1-alpha},
    f_m the chi-square density.  Accurate to <0.5pp when m/G <= 0.03; it
    understates the true distortion badly at the m/G ~ 1/4 of the leading
    application, which is the point of Corollary 11.
    """
    c = stats.chi2.ppf(1 - alpha, m)
    f = stats.chi2.pdf(c, m)
    return float(alpha + c * f * (m + c) / (2.0 * (float(G) - 1.0)))


def wald_size_normal_limit(G, m, alpha=0.05, omega2=1.0, refined=True):
    """Many-restriction limit of the chi-square-calibrated size (Corollary 11(ii)).

    With nu = G-1, d = nu-m+1, the limit along m -> infinity, m/nu -> 0,
    chi_m := sqrt(m/2) log(nu/d) -> chi is Phi(chi - z_{1-alpha}); since
    log(nu/d) ~ m/nu this is Phi(m^{3/2}/(sqrt(2)(G-1)) - z).  `refined=True`
    keeps the two finite-m terms that vanish in the limit (the exact log ratio
    and the variance inflation 1 + m/d); it is uniformly conservative.
    """
    z = stats.norm.ppf(1 - alpha)
    nu = float(G) - 1.0
    d = nu - m + 1.0
    if d < 1.0:
        return float("nan")
    if refined:
        mu = np.sqrt(m / 2.0) * (np.log(nu / d) + np.log(omega2))
        sd = np.sqrt(1.0 + m / d)
    else:
        mu = m ** 1.5 / (np.sqrt(2.0) * nu)
        sd = 1.0
    return float(stats.norm.sf((z - mu) / sd))


def max_restrictions(G_eff, target=0.075, alpha=0.05, omega2=1.0, m_max=200):
    """Largest post-window length m with chi-square Wald size <= target.

    Corollary 12.  Returns 0 when even m = 1 exceeds the target.
    """
    best = 0
    for m in range(1, int(m_max) + 1):
        if float(G_eff) - m < 1.0:
            break
        s = predicted_wald_size(G_eff, m, alpha, omega2, "chi2")
        if np.isfinite(s) and s <= target:
            best = m
        else:
            break
    return best


# permutation

def _perm_matrix(delta, n_perm, rng):
    """Columns are permuted label vectors, column 0 being the identity."""
    delta = np.asarray(delta, float)
    C = len(delta)
    if n_perm is None:
        n_perm = 999
    total = math.factorial(C)
    if total <= n_perm:
        D = np.column_stack([np.asarray(p, float) for p in itertools.permutations(delta)])
        j = int(np.argmax([np.array_equal(D[:, k], delta) for k in range(D.shape[1])]))
        D[:, [0, j]] = D[:, [j, 0]]
        return D, True
    D = np.empty((C, n_perm))
    D[:, 0] = delta
    for p in range(1, n_perm):
        D[:, p] = rng.permutation(delta)
    return D, False


def label_stabilizer_size(delta, decimals=12):
    """|Stab(delta)| = prod over tied label groups of (group size)!.

    Ties in the discretion index enlarge the stabiliser of the label vector, so
    the C! orbit collapses onto C!/|Stab| distinct assignments, each attained
    |Stab| times.  The smallest attainable one-sided p-value is then at least
    |Stab|/C!, not 1/C!.
    """
    v, cnt = np.unique(np.round(np.asarray(delta, float), decimals),
                       return_counts=True)
    out = 1
    for k in cnt:
        out *= math.factorial(int(k))
    return int(out)


def _orbit_multiplicity(t, tol=1e-9):
    """How many orbit coordinates tie with the maximum (the realised floor)."""
    t = np.asarray(t, float)
    scale = max(1.0, float(np.max(np.abs(t))))
    return int(np.sum(t >= t.max() - tol * scale))


def _injection_matrix(d_full, kept, n_perm, rng):
    """Reference set of INJECTIONS of the surviving arrays into the full label set.

    When the loading guard drops categories, permuting only the surviving
    d-values conditions on a multiset that is not ancillary.  Under category
    exchangeability the correct reference set for the C surviving arrays is the
    set of ordered injections from them into the full C0-vector of labels, of
    size C0!/(C0-C)!.  Column 0 is the realised assignment.
    """
    d_full = np.asarray(d_full, float)
    kept = np.asarray(kept, bool)
    C0 = len(d_full)
    C = int(kept.sum())
    obs = d_full[kept]
    total = math.factorial(C0) // math.factorial(C0 - C)
    if n_perm is None:
        n_perm = 999
    if total <= n_perm:
        cols = [np.asarray(p, float) for p in itertools.permutations(d_full, C)]
        D = np.column_stack([c - c.mean() for c in cols])
        j = int(np.argmax([np.allclose(cols[k], obs) for k in range(len(cols))]))
        D[:, [0, j]] = D[:, [j, 0]]
        return D, True
    D = np.empty((C, n_perm))
    D[:, 0] = obs - obs.mean()
    for p in range(1, n_perm):
        c = rng.choice(d_full, size=C, replace=False)
        D[:, p] = c - c.mean()
    return D, False


def permutation_test(comp, stat="wald", n_perm=None, seed=0, w=None,
                     alternative="greater", idx=None, keep_draws=False,
                     alpha=0.05, reference="labels"):
    """Randomisation test over permutations of the discretion labels.

    Exact in finite samples -- for any G, T and error dependence -- under
    Assumption CE of theory/inference.md: the C loading-normalised category data
    arrays are exchangeable.  Exhaustive whenever C! <= n_perm, otherwise a
    Monte Carlo randomisation test with the identity included (still exactly
    valid, Theorem 1(b)).

    `alternative` matters only for stat="directed": "greater" if the recording
    response is expected to raise the recorded outcome where discretion is high.
    """
    rng = np.random.default_rng(seed)
    if reference == "injection":
        if "d_full" not in comp:
            raise ValueError("comp lacks d_full; rebuild with gradient_components")
        D, exhaustive = _injection_matrix(comp["d_full"], comp["kept"], n_perm, rng)
    elif reference == "labels":
        D, exhaustive = _perm_matrix(comp["delta"], n_perm, rng)
    else:
        raise ValueError("reference must be 'labels' or 'injection'")
    t = _apply_stat(comp, D, stat, w=w, idx=idx)
    if stat.startswith("directed") and alternative == "less":
        t = -t
    obs = t[0]
    if exhaustive:
        p = float((t >= obs - 1e-12).mean())
    else:
        p = float((1.0 + (t[1:] >= obs - 1e-12).sum()) / len(t))
    # resolution
    mult = _orbit_multiplicity(t)
    n_distinct = int(len(np.unique(np.round(t, 9))))
    min_p = (mult / len(t)) if exhaustive else (1.0 / len(t))
    out = {"stat": float(obs), "pvalue": p, "n_perm": int(len(t)),
           "exhaustive": bool(exhaustive), "min_pvalue": float(min_p),
           "min_pvalue_naive": 1.0 / len(t),
           "orbit_multiplicity": mult, "n_distinct": n_distinct,
           "stabilizer": label_stabilizer_size(comp["delta"]),
           "rejection_attainable": bool(min_p <= alpha + 1e-12),
           "reference": reference,
           # reference
           "equivariant_loading": bool(comp.get("equivariant", True)),
           "exact_under_CE": bool(comp.get("equivariant", True)
                                  and not comp.get("guard_fired", False)),
           "guard_fired": bool(comp.get("guard_fired", False)),
           "n_dropped_by_guard": len(comp.get("dropped", []))}
    if keep_draws:
        out["null_draws"] = t
    return out


# bootstrap

def _restricted(comp):
    """Restricted (gamma_post = 0) fit; cluster scores of the gamma block.

    z_g^r = sum_c delta_c Xt_g'(y_c - Xt(beta_hat + delta_c gamma^r))
          = Z_g + (sum_c delta_c^2) Sg (gamma_hat - gamma^r).
    """
    delta = comp["delta"]
    s2 = float((delta ** 2).sum())
    gam = comp["B"] @ delta / s2
    pre = comp["pre"]
    gfull = np.zeros_like(gam)
    if len(pre):
        Spp = comp["S"][np.ix_(pre, pre)]
        gfull[pre] = np.linalg.solve(Spp, (comp["S"] @ gam)[pre])
    Z = np.einsum("gqc,c->gq", comp["A"], delta, optimize=True)
    Zr = Z + s2 * np.einsum("gqr,r->gq", comp["Sg"], gam - gfull, optimize=True)
    return {"gamma": gam, "gamma_r": gfull, "Zr": Zr, "s2": s2}


def wild_cluster_bootstrap(comp, n_boot=499, seed=0, stat="wald", w=None,
                           weights="rademacher"):
    """Restricted (null-imposed) wild cluster bootstrap, Cameron-Gelbach-Miller.

    Bootstrap outcome y*_c = Xt(beta_hat + delta_c gamma^r) + v_g e^r_gc with
    v_g i.i.d. Rademacher at the cluster level, applied to the partialled-out
    regression; each draw is refit unrestricted with the same CR1 covariance.
    Asymptotically valid as G -> infinity (Theorem 4); not exact at any fixed G.
    """
    rng = np.random.default_rng(seed)
    r = _restricted(comp)
    Zr, s2, G = r["Zr"], r["s2"], comp["G"]
    Si, P = comp["Sinv"], comp["post"]
    SiP = Si[:, P]
    # bootstrap
    Tg = np.einsum("qr,grs,sm->gqm", Si, comp["Sg"], SiP, optimize=True)

    Wt = np.empty((n_boot, G))
    Wt[0] = 1.0
    if weights == "rademacher":
        Wt[1:] = rng.choice([-1.0, 1.0], size=(n_boot - 1, G))
    else:
        Wt[1:] = rng.normal(size=(n_boot - 1, G))

    u = Wt @ Zr  # (B, q)
    gP = (u @ SiP) / s2  # (B, m)
    ZrP = Zr @ SiP  # (G, m)
    Zb = Wt[:, :, None] * ZrP[None, :, :] - np.einsum("gqm,pq->pgm", Tg, u, optimize=True)
    meat = np.einsum("pgm,pgn->pmn", Zb, Zb, optimize=True)
    VP = comp["adj"] * meat / s2 ** 2

    t = _stat_from(gP, VP, stat, w)
    obs = t[0]
    return {"stat": float(obs), "n_boot": int(n_boot),
            "pvalue": float((1.0 + (t[1:] >= obs - 1e-12).sum()) / n_boot)}


def signflip_test(comp, n_draws=999, seed=0, stat="wald", w=None, idx=None,
                  keep_draws=False):
    """Cluster sign-change randomisation test on restricted scores.

    zeta_g = (S^{-1} z_g^r)_post / sum_c delta_c^2 is cluster g's contribution to
    gamma_hat_post, and sum_g zeta_g = gamma_hat_post exactly.  The test
    randomises over the group {-1,+1}^G acting on the zeta_g.  Because the
    Rademacher weights leave sum_g zeta_g zeta_g' unchanged, the covariance is
    common to all draws and the statistic is a fixed quadratic form in
    sum_g eps_g zeta_g.  Exact when the cluster contributions are jointly
    sign-symmetric about zero under the null; asymptotically valid in the
    Canay-Romano-Shaikh approximate-symmetry sense (Theorem 3).  Its resolution
    is 2^{-G} rather than 1/C!, so unlike label permutation it works with C = 3.
    """
    rng = np.random.default_rng(seed)
    r = _restricted(comp)
    P = comp["post"] if idx is None else np.asarray(idx)
    zeta = (r["Zr"] @ comp["Sinv"][:, P]) / r["s2"]  # (G, m)
    G, m = zeta.shape
    E = np.empty((n_draws, G))
    E[0] = 1.0
    E[1:] = rng.choice([-1.0, 1.0], size=(n_draws - 1, G))
    gP = E @ zeta  # (B, m)
    VP = np.broadcast_to(comp["adj"] * (zeta.T @ zeta), (n_draws, m, m))
    t = _stat_from(gP, np.ascontiguousarray(VP), stat, w)
    obs = t[0]
    out = {"stat": float(obs), "n_draws": int(n_draws),
           "pvalue": float((1.0 + (t[1:] >= obs - 1e-12).sum()) / n_draws)}
    if keep_draws:
        out["null_draws"] = t
    return out


# convenience

_METHODS = {
    "chi2":         lambda c, kw: chi2_wald(c),
    "f":            lambda c, kw: f_wald(c),
    "hotelling":    lambda c, kw: hotelling_wald(c),
    "hotelling_geff": lambda c, kw: hotelling_wald(c, G_eff=effective_clusters(c)),
    "perm_wald":    lambda c, kw: permutation_test(c, "wald", kw["n_perm"], kw["seed"]),
    "perm_dir":     lambda c, kw: permutation_test(c, "directed", kw["n_perm"], kw["seed"]),
    "perm_dirgls":  lambda c, kw: permutation_test(c, "directed_gls", kw["n_perm"], kw["seed"]),
    "perm_chibar":  lambda c, kw: permutation_test(c, "chibar", kw["n_perm"], kw["seed"]),
    "wcb_wald":     lambda c, kw: wild_cluster_bootstrap(c, kw["n_boot"], kw["seed"], "wald"),
    "wcb_dir":      lambda c, kw: wild_cluster_bootstrap(c, kw["n_boot"], kw["seed"], "directed"),
    "signflip":     lambda c, kw: signflip_test(c, kw["n_draws"], kw["seed"], "wald"),
    "signflip_dir": lambda c, kw: signflip_test(c, kw["n_draws"], kw["seed"], "directed"),
}


def gradient_inference(df, methods=("chi2", "f", "perm_wald", "perm_dir",
                                    "wcb_wald", "signflip"),
                       outcome="Y", k_pre=5, k_post=6, bridge_cat=0,
                       use_bridge=False, cluster_col="unit", loading="triad",
                       n_perm=None, n_boot=499, n_draws=999, seed=0):
    """Run several gradient-test inference procedures on one panel.

    Returns {'components': comp, 'p': {method: pvalue}, 'detail': {method: dict}}.
    """
    comp = gradient_components(df, outcome=outcome, k_pre=k_pre, k_post=k_post,
                               bridge_cat=bridge_cat, use_bridge=use_bridge,
                               cluster_col=cluster_col, loading=loading)
    kw = {"n_perm": n_perm, "n_boot": n_boot, "n_draws": n_draws, "seed": seed}
    out, detail = {}, {}
    for m in methods:
        if m not in _METHODS:
            raise ValueError(f"unknown method {m!r}")
        r = _METHODS[m](comp, kw)
        out[m] = r["pvalue"]
        detail[m] = r
    return {"components": comp, "p": out, "detail": detail}


# power

def design_information(comp, k=None):
    """Path information I_k = k' ([S^{-1}]_{PP})^{-1} k on the post block.

    S = Xt'Xt is the demeaned event-dummy Gram matrix of ONE category, so I_k is
    a property of the treatment-timing design, T and the horizon window only --
    not of the outcome.  k is the recording-path shape over post horizons
    (default: constant).
    """
    P = comp["post"]
    W = np.linalg.pinv(comp["Sinv"][np.ix_(P, P)])
    k = np.ones(len(P)) if k is None else np.asarray(k, float)
    return {"I_k": float(k @ W @ k), "W": W, "k": k,
            "sum_delta2": float((comp["delta"] ** 2).sum())}


def noncentrality(comp, s, k=None, sigma2=None):
    """Wald non-centrality for a category-level signal vector s.

    Category-level model  E[b_c] = tau + s_c k.  Then
        E[gamma_hat] = k g,   g = sum_c delta_c s_c / sum_c delta_c^2 ,
        Var(gamma_hat) = vbar S^{-1},  vbar = sum_c delta_c^2 sigma_c^2 / (sum delta^2)^2,
        lambda = g^2 I_k / vbar .
    With homoskedastic categories and s_c = kappa d_c this collapses to
        lambda = (kappa^2 / sigma^2) (sum_c delta_c^2) I_k .
    sigma2 is a scalar or a length-C vector; default comp['resid_var'].
    """
    delta = comp["delta"]
    s = np.asarray(s, float)
    s2 = float((delta ** 2).sum())
    g = float(delta @ s / s2)
    if sigma2 is None:
        sigma2 = comp["resid_var"]
    sig = (np.full(len(delta), float(sigma2)) if np.ndim(sigma2) == 0
           else np.asarray(sigma2, float))
    vbar = float((delta ** 2 @ sig) / s2 ** 2)
    di = design_information(comp, k)
    return {"lambda": g ** 2 * di["I_k"] / vbar, "g": g, "vbar": vbar,
            "I_k": di["I_k"], "sum_delta2": s2}


def chi2_noncentrality_for_power(q, alpha=0.05, power=0.80):
    """lambda* solving P(chi2_q(lambda*) > chi2_{q,1-alpha}) = power."""
    crit = stats.chi2.ppf(1 - alpha, q)

    def f(lam):
        return stats.ncx2.sf(crit, q, lam) - power

    hi = max(4.0 * (q + 10.0), 10.0)
    while f(hi) < 0 and hi < 1e7:
        hi *= 2
    return float(optimize.brentq(f, 1e-8, hi))


def hotelling_crit(q, G_eff, alpha=0.05):
    """Exact Hotelling-T2 critical value on the chi-square scale.

    c_H = q (G_eff - 1) / (G_eff - q) * F_{q, G_eff - q, 1-alpha}.  Reduces to
    chi2_{q,1-alpha} as G_eff -> infinity and is the correctly sized reference
    point at the cluster counts this project actually has.
    """
    dfd = G_eff - q
    if dfd <= 0:
        return float("inf")
    return float(q * (G_eff - 1) / dfd * stats.f.ppf(1 - alpha, q, dfd))


def noncentrality_for_power_crit(crit, q, power=0.80):
    """lambda* solving P(chi2_q(lambda*) > crit) = power, for ANY critical value."""
    def f(lam):
        return stats.ncx2.sf(crit, q, lam) - power
    hi = max(4.0 * (q + 10.0), 10.0)
    while f(hi) < 0 and hi < 1e9:
        hi *= 2
    return float(optimize.brentq(f, 1e-8, hi))


def mde_kappa(comp, kind="wald", alpha=0.05, power=0.80, k=None, sigma2=None,
              wedge=None, calibration="chi2", G_eff=None, idx=None):
    """Minimum detectable measurement response kappa.

    Homoskedastic linear case (wedge=None, s_c = kappa d_c):
        kappa_MDE = sigma sqrt( lambda*(alpha,power,q) / (sum_c delta_c^2 I_k) )  [wald]
        kappa_MDE = sigma (z_{1-alpha} + z_power) / sqrt(sum_c delta_c^2 I_k)     [directed]
    `wedge` is an optional callable kappa -> s (length C) for nonlinear recording
    maps -- e.g. lambda kap: np.log1p(kap * comp['d']) under the multiplicative
    or Poisson map -- in which case lambda(kappa) = lambda* is solved numerically.
    """
    q = len(comp["post"])
    if kind == "wald":
        if calibration == "chi2":
            lam_star = chi2_noncentrality_for_power(q, alpha, power)
        elif calibration == "hotelling":
            # cluster
            Ge = float(effective_clusters(comp, idx=idx)) if G_eff is None else G_eff
            lam_star = noncentrality_for_power_crit(
                hotelling_crit(q, Ge, alpha), q, power)
        else:
            raise ValueError("calibration must be 'chi2' or 'hotelling' "
                             "(use mde_kappa_permutation for the exact test)")
    elif kind == "directed":
        if calibration == "chi2":
            lam_star = (stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) ** 2
        elif calibration == "hotelling":
            Ge = float(effective_clusters(comp, idx=idx)) if G_eff is None else G_eff
            dfd = max(Ge - 1.0, 1.0)
            lam_star = (stats.t.ppf(1 - alpha, dfd) + stats.norm.ppf(power)) ** 2
        else:
            raise ValueError(calibration)
    else:
        raise ValueError(kind)

    if wedge is None:
        info = noncentrality(comp, comp["d"], k=k, sigma2=sigma2)  # lambda
        return {"kappa": float(np.sqrt(lam_star / info["lambda"])),
                "lambda_star": lam_star, "lambda_at_1": info["lambda"],
                "I_k": info["I_k"], "sum_delta2": info["sum_delta2"]}

    def f(kap):
        return noncentrality(comp, wedge(kap), k=k, sigma2=sigma2)["lambda"] - lam_star

    hi = 1.0
    while f(hi) < 0 and hi < 1e3:
        hi *= 2
    return {"kappa": float(optimize.brentq(f, 1e-8, hi)), "lambda_star": lam_star}


def local_power_wald(lam, q, alpha=0.05):
    """P(chi2_q(lam) > chi2_{q,1-alpha})."""
    return float(stats.ncx2.sf(stats.chi2.ppf(1 - alpha, q), q, lam))


def local_power_permutation(delta, k, W, sigma_bar2, kappa, n_sim=5000,
                            alpha=0.05, stat="wald", seed=0, n_perm=None):
    """Local power of the label-permutation test in the Gaussian limit experiment.

    Draws eps_c ~ N(0, sigma_bar2 W^{-1}) i.i.d. across categories and forms, for
    every permutation pi,
        gamma(pi) = kappa k rho_pi + sum_c delta_{pi(c)} eps_c / sum delta^2 ,
        rho_pi = sum_c delta_{pi(c)} d_c / sum_c delta_c^2 ,
    which is the exact joint limiting law of the C! permuted estimators (Theorem
    6).  W = ([S^{-1}]_{PP})^{-1} is the post-block information matrix.
    """
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, float)
    C = len(delta)
    D, _ = _perm_matrix(delta, n_perm, rng)
    s2 = float((delta ** 2).sum())
    k = np.asarray(k, float)
    m = len(k)
    Winv = np.linalg.pinv(np.asarray(W, float))
    Vg = sigma_bar2 * Winv / s2
    Le = np.linalg.cholesky(sigma_bar2 * Winv + 1e-13 * np.eye(m))
    rho = (D.T @ (delta + delta.mean())) / s2  # delta_
    Vinv = np.linalg.pinv(Vg)
    kv = float(k @ Vg @ k)
    rej = 0
    for _ in range(n_sim):
        eps = Le @ rng.standard_normal((m, C))
        gam = kappa * np.outer(k, rho) + (eps @ D) / s2  # (m, P)
        if stat == "wald":
            t = np.einsum("mp,mn,np->p", gam, Vinv, gam, optimize=True)
        elif stat == "directed":
            t = (k @ gam) / np.sqrt(kv)
        else:
            raise ValueError(stat)
        rej += int((t >= t[0] - 1e-12).mean() <= alpha + 1e-12)
    return rej / n_sim


# power

def rho_orbit(delta, d=None, n_perm=None, seed=0):
    """The label-overlap coefficients rho_pi = sum_c delta_{pi(c)} d_c / sum delta^2.

    Returns (rho, Dperm) with rho[0] = 1 the identity.  Under the category-level
    limit experiment (7) the permuted estimator has mean kappa k rho_pi, so
    rho_pi is the fraction of the recording signal that survives relabelling
    (Theorem 6).  It is also exactly the correlation between the identity and
    the pi-th permuted directed statistic (Theorem 9).
    Enumerates all C! permutations when C! <= 50400, otherwise samples n_perm.
    """
    delta = np.asarray(delta, float)
    C = len(delta)
    s2 = float((delta ** 2).sum())
    d = delta if d is None else np.asarray(d, float)
    full = math.factorial(C)
    if n_perm is None and full <= 50400:
        P = np.array(list(itertools.permutations(range(C))), dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        n = full if n_perm is None else int(n_perm)
        P = np.empty((n, C), dtype=np.int64)
        P[0] = np.arange(C)
        for b in range(1, n):
            P[b] = rng.permutation(C)
    Dp = delta[P]  # (M, C)
    return Dp @ d / s2, Dp


def nearest_label_correlation(C):
    """Closed form max_{pi != id} rho_pi = 1 - 12 / (C (C^2 - 1)) for equally
    spaced discretion labels (Lemma 8); attained by the C-1 adjacent
    transpositions.  Independent of the spacing and the location of d."""
    return 1.0 - 12.0 / (C * (C * C - 1))


def label_resolution(C, alpha=0.05, even=True):
    """Smallest attainable p-value of the label-permutation test (Proposition 2)
    and whether a level-alpha rejection is attainable at all.
    `even=True` for a statistic invariant to d -> -d (the Wald and chi-bar
    statistics with equally spaced labels), False for a one-sided statistic.

    `J` is the COMPETITOR BUDGET: the p-value counts the identity itself, so
    p_hat = (1 + Z)/M with Z =  # rejects
    iff Z <= alpha M - 1, i.e. Z <= J with J = floor(alpha M) - 1.  J < 0 means
    no rejection is attainable (the resolution bound of Proposition 2)."""
    M = math.factorial(C)
    pmin = (2.0 if even else 1.0) / M
    return {"M": M, "p_min": pmin, "attainable": pmin <= alpha,
            "J": math.floor(alpha * M) - 1}


def perm_beat_probability(rho, tau):
    """P(t(pi) >= t(id) | t(id) = tau) = Phibar(tau sqrt((1-rho)/(1+rho))),
    the exact conditional beat probability of Theorem 9(ii)."""
    rho = np.asarray(rho, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.sqrt(np.clip((1.0 - rho) / (1.0 + rho), 0.0, np.inf))
    return stats.norm.sf(tau * psi)


def perm_power_markov_bound(mu, delta, d=None, alpha=0.05, **kw):
    """Rigorous lower bound on the power of the label-permutation test with the
    directed statistic (Corollary 8):

        power >= 1 - (1/(J+1)) sum_{pi != id} Phibar( mu sqrt((1-rho_pi)/2) ),
        J = floor(alpha M) - 1  (the competitor budget; the p-value counts the
        identity itself, so rejection needs Z <= alpha M - 1).

    Unconditional (Markov on the number of permutations that beat the identity);
    informative in the high-power regime only.  Returns the bound clipped to
    [0, 1] together with the expected beat count E[Z]."""
    rho, _ = rho_orbit(delta, d=d, **kw)
    M = len(rho)
    J = math.floor(alpha * M) - 1
    if J < 0:
        return {"bound": 0.0, "EZ": float("inf"), "J": J, "M": M}
    with np.errstate(divide="ignore"):
        EZ = float(stats.norm.sf(mu * np.sqrt(np.clip((1 - rho[1:]) / 2, 0, np.inf))).sum())
    return {"bound": float(np.clip(1.0 - EZ / (J + 1), 0.0, 1.0)),
            "EZ": EZ, "J": J, "M": M}


def perm_power_directed(mu, delta, d=None, alpha=0.05, n_sim=4000, seed=0,
                        n_perm=None):
    """Exact power of the label-permutation test with the directed statistic in
    the Gaussian limit experiment with KNOWN V and homoskedastic categories.

    Uses the sufficient representation of Theorem 9:
        t(pi) = mu rho_pi + sum_c delta_{pi(c)} u_c / ||delta||,  u ~ N(0, I_C),
    so the whole C!-dimensional field is driven by C standard normals.
    mu = kappa sqrt(sum_c delta_c^2) sqrt(I_k) / sigma is the noncentrality of
    the infeasible directed test (Corollary 3).
    """
    rho, Dp = rho_orbit(delta, d=d, n_perm=n_perm, seed=seed)
    M, C = Dp.shape
    sd = math.sqrt(float((np.asarray(delta, float) ** 2).sum()))
    thr = alpha * M + 1e-12
    rng = np.random.default_rng(seed + 1)
    base = mu * rho
    rej = 0
    block = max(1, int(4e6 // max(M, 1)))
    done = 0
    while done < n_sim:
        b = min(block, n_sim - done)
        U = rng.standard_normal((C, b))
        Tm = base[:, None] + (Dp @ U) / sd  # (M, b)
        rej += int(((Tm >= Tm[0] - 1e-12).sum(axis=0) <= thr).sum())
        done += b
    return rej / n_sim


def perm_inflation(delta, d=None, alpha=0.05, power=0.80, n_sim=4000, seed=0,
                   n_perm=None, mu_max=60.0):
    """Noncentrality inflation factor of label randomisation,

        e_perm = mu_perm(alpha, power) / (z_{1-alpha} + z_power),

    where mu_perm is the noncentrality at which the label-permutation test with
    the directed statistic attains `power`.  The permutation-calibrated MDE is
    e_perm times the infeasible directed MDE of Corollary 3.  Returns None for
    `mu_perm` when the resolution bound of Proposition 2 makes `power`
    unattainable (e.g. C <= 3)."""
    C = len(np.asarray(delta))
    res = label_resolution(C, alpha, even=False)
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    out = {"C": C, "M": res["M"], "J": res["J"], "z_benchmark": float(z),
           "max_rho": float(nearest_label_correlation(C)),
           "attainable": res["attainable"]}
    if not res["attainable"]:
        out.update(mu_perm=None, e_perm=None)
        return out

    def f(m):
        return perm_power_directed(m, delta, d=d, alpha=alpha, n_sim=n_sim,
                                   seed=seed, n_perm=n_perm) - power

    if f(mu_max) < 0:
        out.update(mu_perm=None, e_perm=None)
        return out
    mu = float(optimize.brentq(f, 1e-3, mu_max, xtol=2e-2))
    out.update(mu_perm=mu, e_perm=mu / float(z))
    return out


def mde_kappa_permutation(comp, alpha=0.05, power=0.80, k=None, sigma2=None,
                          wedge=None, n_sim=4000, seed=0):
    """MDE of the label-permutation-calibrated directed test.

    Solves lambda(kappa) = mu_perm^2 with mu_perm from perm_inflation, i.e. the
    Corollary 3 MDE inflated by e_perm.  With `wedge` the nonlinear recording
    map is inverted numerically exactly as in mde_kappa."""
    inf = perm_inflation(comp["delta"], d=comp["d"], alpha=alpha, power=power,
                         n_sim=n_sim, seed=seed)
    if inf["mu_perm"] is None:
        return {"kappa": None, **inf}
    lam_star = inf["mu_perm"] ** 2
    if wedge is None:
        info = noncentrality(comp, comp["d"], k=k, sigma2=sigma2)
        return {"kappa": float(np.sqrt(lam_star / info["lambda"])),
                "lambda_star": lam_star, **inf}

    def f(kap):
        return noncentrality(comp, wedge(kap), k=k, sigma2=sigma2)["lambda"] - lam_star

    hi = 1.0
    while f(hi) < 0 and hi < 1e3:
        hi *= 2
    return {"kappa": float(optimize.brentq(f, 1e-8, hi)),
            "lambda_star": lam_star, **inf}


# calibration

def anderson_rubin_set(comp, kvec=None, alpha=0.05, calibration="signflip",
                       n_draws=999, seed=0, idx=None):
    """Confidence set for kappa in the model gamma_post(l) = kappa * k(l).

    S(kappa0) = (ghat - kappa0 k)' V^{-1} (ghat - kappa0 k) is quadratic in
    kappa0, so the set {kappa0 : S(kappa0) <= crit} is a closed interval, the
    whole line, or EMPTY.  Emptiness is a specification test: no value of the
    recording response reconciles the estimated gradient path with the shape
    k(.).  Calibrations:
      "chi2"      chi2_{m,1-alpha}                (oversized at G = 51)
      "hotelling" hotelling_crit(m, G_eff, alpha) (conservative, correctly sized)
      "signflip"  (1-alpha) quantile of the cluster sign-flip null of the
                  CENTRED quadratic form, which is the reference distribution of
                  S(kappa0) at the true kappa0.
    """
    idx = comp["post"] if idx is None else np.asarray(idx)
    m = len(idx)
    kvec = np.ones(m) if kvec is None else np.asarray(kvec, float)
    if len(kvec) != m:
        raise ValueError("kvec must have one entry per post horizon")
    D = comp["delta"][:, None]
    gP, VP = _gamma_V_block(comp, D, idx)
    ghat, V = gP[0], VP[0]
    Vi = np.linalg.pinv(V)

    if calibration == "chi2":
        crit = float(stats.chi2.ppf(1 - alpha, m))
        Ge = np.nan
    elif calibration == "hotelling":
        Ge = float(effective_clusters(comp, idx=idx))
        crit = hotelling_crit(m, Ge, alpha)
    elif calibration == "signflip":
        sf = signflip_test(comp, n_draws=n_draws, seed=seed, stat="wald",
                           idx=idx, keep_draws=True)
        draws = np.asarray(sf["null_draws"]) if "null_draws" in sf else None
        if draws is None:
            raise RuntimeError("signflip_test did not return null draws")
        crit = float(np.quantile(draws, 1 - alpha))
        Ge = np.nan
    else:
        raise ValueError(calibration)

    a = float(kvec @ Vi @ kvec)
    b = float(kvec @ Vi @ ghat)
    c = float(ghat @ Vi @ ghat)
    out = {"stat_at_zero": c, "crit": crit, "calibration": calibration,
           "m": m, "a": a, "b": b, "c": c, "G_eff": Ge}
    if a <= 1e-14:
        out.update({"L": -np.inf, "U": np.inf, "empty": bool(c > crit),
                    "unbounded": True})
        return out
    disc = b * b - a * (c - crit)
    if disc < 0:
        out.update({"L": np.nan, "U": np.nan, "empty": True, "unbounded": False,
                    "min_stat": c - b * b / a, "argmin": b / a})
        return out
    r = math.sqrt(disc)
    out.update({"L": (b - r) / a, "U": (b + r) / a, "empty": False,
                "unbounded": False, "min_stat": c - b * b / a, "argmin": b / a,
                "contains_zero": bool((b - r) / a <= 0 <= (b + r) / a)})
    return out
