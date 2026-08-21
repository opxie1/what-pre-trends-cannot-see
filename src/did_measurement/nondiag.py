"""Identified set for the relative loadings when A10b fails (open item G3).

Lemma 6.3 point-identifies rho_c = theta_c / theta_r from untreated-period
cross-category covariances only under A10b: the untreated idiosyncratic
covariance Psi must be DIAGONAL.  The project's own overidentification test
rejects that restriction in all three applications, and the natural violation --
a records-management shock loading on the discretion index -- produces the exact
bias rho_hat_c = rho_c + varkappa_s d_c, which under perfect measurement
invariance manufactures the paper's headline wedge.

Nothing downstream of Lemma 6.3 can be believed while the identified set under a
non-diagonal Psi is uncharacterised.  This module characterises it for the
one-factor model with a bounded idiosyncratic cross-correlation.

MODEL.  On untreated cells the C x C residual covariance is
        S = V theta theta' + Psi,   V > 0,  Psi PSD,  psi_c = Psi_cc >= 0,
with the normalisation theta_r = 1, so theta_c = rho_c.

CLASS.  A10b(b), b in [0, 1]:  |Psi_ck| <= b sqrt(psi_c psi_k) for all c != k.
        b = 0 is A10b exactly.  b = 1 is no restriction at all beyond PSD-ness of
        each 2 x 2 block, so the family nests the assumption and its complete
        negation, and b indexes how much administrative co-movement in recording
        errors the reader is willing to allow.

RESULT R1 (pairwise outer set, closed form).  Fix the reference r and a category
c.  Using only the (r, c) block and psi_c, psi_r >= 0, the set of rho_c
consistent with S at a GIVEN V is the solution of

        (S_rc - V rho)^2 <= b^2 (S_rr - V)(S_cc - V rho^2),      V rho^2 <= S_cc,

a quadratic inequality with positive leading coefficient V(V + b^2 (S_rr - V)),
hence an interval.  The identified set for rho_c is the union over admissible V
in (0, S_rr].  At b = 0 it collapses to the single point S_rc / V with V pinned
by any third category, reproducing Lemma 6.3.

RESULT R2 (gradient estimand bounds).  The loading-adjusted gradient at horizon
l is gamma(l) = sum_c delta_c beta_c(l) / rho_c / sum_c delta_c^2.  Each summand
is monotone in its own rho_c on any interval not containing zero, and the
summands are separable, so over a product set the extremes of gamma are attained
at a vertex and are available in closed form.  This is the object a practitioner
needs after the A10 diagnostic rejects: an interval for the gradient rather than
nothing.

The functions marked "sharp" additionally impose every (c, k) block jointly and
are computed numerically; `rho_outer_set` is provably an outer set of them.
"""

import math

import numpy as np


# --------------------------------------------------------------------------
# R1: pairwise outer set for the relative loadings
# --------------------------------------------------------------------------

def _rho_interval_given_V(S, r, c, b, V):
    """Interval of rho_c consistent with the (r, c) block at a given factor variance."""
    S_rr, S_cc, S_rc = float(S[r, r]), float(S[c, c]), float(S[r, c])
    psi_r = S_rr - V
    if psi_r < -1e-12 or V <= 0:
        return None
    psi_r = max(psi_r, 0.0)
    Q = b * b * psi_r
    a2 = V * (V + Q)
    a1 = -2.0 * S_rc * V
    a0 = S_rc * S_rc - Q * S_cc
    if a2 <= 1e-300:
        return None
    disc = a1 * a1 - 4.0 * a2 * a0
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    lo, hi = (-a1 - sq) / (2 * a2), (-a1 + sq) / (2 * a2)
    cap = math.sqrt(max(S_cc, 0.0) / V)          # psi_c >= 0
    lo, hi = max(lo, -cap), min(hi, cap)
    if lo > hi:
        return None
    return lo, hi


def rho_outer_set(S, ref=0, b=0.0, n_grid=2001, v_lo=None, v_hi=None):
    """Outer identified set for rho_c = theta_c / theta_ref under A10b(b).

    Returns dict with per-category (lo, hi), the grid of factor variances that
    were admissible, and the value of b.  Empty categories carry NaN.
    """
    S = np.asarray(S, float)
    C = S.shape[0]
    S_rr = float(S[ref, ref])
    lo_v = 1e-8 * S_rr if v_lo is None else v_lo
    hi_v = S_rr if v_hi is None else v_hi
    Vs = np.linspace(lo_v, hi_v, int(n_grid))
    out = {}
    for c in range(C):
        if c == ref:
            out[c] = (1.0, 1.0)
            continue
        lo, hi = np.inf, -np.inf
        for V in Vs:
            iv = _rho_interval_given_V(S, ref, c, b, V)
            if iv is None:
                continue
            lo = min(lo, iv[0])
            hi = max(hi, iv[1])
        out[c] = (float(lo), float(hi)) if lo <= hi else (np.nan, np.nan)
    return {"rho_set": out, "b": float(b), "ref": int(ref),
            "V_range": (float(lo_v), float(hi_v)), "n_grid": int(n_grid),
            "widths": {c: (out[c][1] - out[c][0]) for c in out}}


def rho_lemma63_point(S, ref=0):
    """The Lemma 6.3 point estimate, i.e. the b = 0 member of the family.

    Under A10b every off-diagonal is S_ck = V theta_c theta_k, so for any triad
    (ref, c, k) the factor variance is V = S_rc S_rk / S_ck and rho_c = S_rc / V.
    Averaging over admissible instruments k is the shipped IV estimator.
    """
    S = np.asarray(S, float)
    C = S.shape[0]
    others = [c for c in range(C) if c != ref]
    Vs = []
    for i, c in enumerate(others):
        for k in others[i + 1:]:
            if abs(S[c, k]) > 1e-300:
                Vs.append(S[ref, c] * S[ref, k] / S[c, k])
    V = float(np.median(Vs)) if Vs else np.nan
    rho = {ref: 1.0}
    for c in others:
        rho[c] = float(S[ref, c] / V) if V and np.isfinite(V) else np.nan
    return {"rho": rho, "V": V, "n_triads": len(Vs)}


def rho_sharp_set(S, ref=0, b=0.0, n_grid=401, n_rho=801, rho_lim=None,
                  feas_tol=None):
    """Sharp set for rho under A10b(b), imposing EVERY (c, k) block jointly.

    Scans the scalar factor variance V; for each V the constraint on the vector
    rho is a system of pairwise inequalities, which is solved by first taking the
    per-pair-with-reference box and then removing rho_c values that admit no
    partner satisfying the (c, k) constraints for k != ref.  For C <= 5 the pass
    is exact up to the grid; for larger C it remains an outer set (documented,
    not silently claimed sharp).
    """
    S = np.asarray(S, float)
    C = S.shape[0]
    if b <= 1e-12:
        # the constraint is an equality and a grid cannot hit it; the b = 0
        # member of the family is exactly Lemma 6.3.
        pt = rho_lemma63_point(S, ref=ref)
        return {"rho_set": {c: (pt["rho"][c], pt["rho"][c]) for c in pt["rho"]},
                "b": 0.0, "ref": int(ref), "exact": True,
                "note": "b = 0 is Assumption A10b; the sharp set is the Lemma 6.3 point",
                "V": pt["V"]}
    S_rr = float(S[ref, ref])
    if feas_tol is None:
        feas_tol = 1e-6 * float(np.max(np.abs(S)))
    others = [c for c in range(C) if c != ref]
    # A uniform grid in V misses the narrow feasible band around the A10b value,
    # so refine around the Lemma 6.3 factor variance.  The b = 0 solution is
    # feasible for EVERY b >= 0 (the class is nested), so the sharp set must
    # contain it; seeding with it also guarantees monotonicity in b.
    pt0 = rho_lemma63_point(S, ref=ref)
    V0 = pt0["V"]
    Vs = [np.linspace(1e-6 * S_rr, S_rr, int(n_grid))]
    if np.isfinite(V0) and 0 < V0 <= S_rr:
        Vs.append(np.linspace(max(0.5 * V0, 1e-6 * S_rr), min(1.5 * V0, S_rr),
                              int(n_grid)))
    Vs = np.unique(np.concatenate(Vs))
    lim = rho_lim
    acc = {c: [np.inf, -np.inf] for c in others}
    if np.isfinite(V0) and 0 < V0 <= S_rr:
        ok0 = all(np.isfinite(pt0["rho"][c]) and
                  S[c, c] - V0 * pt0["rho"][c] ** 2 >= -1e-9 for c in others)
        if ok0:
            for c in others:
                acc[c][0] = min(acc[c][0], pt0["rho"][c])
                acc[c][1] = max(acc[c][1], pt0["rho"][c])
    exact = C <= 5
    for V in Vs:
        boxes = {}
        ok = True
        for c in others:
            iv = _rho_interval_given_V(S, ref, c, b, V)
            if iv is None:
                ok = False
                break
            boxes[c] = iv
        if not ok:
            continue
        if len(others) == 1:
            c = others[0]
            acc[c][0] = min(acc[c][0], boxes[c][0])
            acc[c][1] = max(acc[c][1], boxes[c][1])
            continue
        grids = {}
        for c in others:
            a, z = boxes[c]
            if lim is not None:
                a, z = max(a, -lim), min(z, lim)
            if a > z:
                ok = False
                break
            grids[c] = np.linspace(a, z, int(n_rho))
        if not ok:
            continue
        psi = {c: np.maximum(S[c, c] - V * grids[c] ** 2, 0.0) for c in others}
        psi_r = max(S_rr - V, 0.0)
        for c in others:
            feas = np.ones(len(grids[c]), bool)
            for k in others:
                if k == c:
                    continue
                # exists rho_k in its box with |S_ck - V rho_c rho_k| <= b sqrt(psi_c psi_k)
                A = S[c, k] - V * np.outer(grids[c], grids[k])
                B = b * np.sqrt(np.outer(psi[c], psi[k]))
                feas &= (np.abs(A) <= B + feas_tol).any(axis=1)
            if feas.any():
                acc[c][0] = min(acc[c][0], float(grids[c][feas].min()))
                acc[c][1] = max(acc[c][1], float(grids[c][feas].max()))
    rho_set = {ref: (1.0, 1.0)}
    for c in others:
        lo, hi = acc[c]
        rho_set[c] = (float(lo), float(hi)) if lo <= hi else (np.nan, np.nan)
    return {"rho_set": rho_set, "b": float(b), "ref": int(ref), "exact": bool(exact),
            "note": ("sharp up to the grid for C <= 5; outer set otherwise "
                     "(pairwise relaxation of the joint PSD constraint)")}


# --------------------------------------------------------------------------
# R2: gradient estimand bounds over a product set of loadings
# --------------------------------------------------------------------------

def gradient_bounds_over_rho(beta, d, rho_set, weights=None):
    """Bounds on the loading-adjusted discretion gradient over a box of loadings.

    beta     : (C,) recorded coefficients (one horizon) for the SAME categories
               indexed by rho_set.
    d        : (C,) discretion index.
    rho_set  : {c: (lo, hi)} from rho_outer_set / rho_sharp_set.

    gamma = sum_c delta_c (beta_c / rho_c) / sum_c delta_c^2.  Separable in rho,
    so the extremes are attained at a vertex.  An interval whose loading box
    straddles zero makes gamma unbounded -- reported as +/-inf rather than
    silently truncated.
    """
    beta = np.asarray(beta, float)
    d = np.asarray(d, float)
    C = len(beta)
    w = np.ones(C) if weights is None else np.asarray(weights, float)
    delta = d - np.average(d, weights=w)
    s2 = float((w * delta ** 2).sum())
    lo_tot = hi_tot = 0.0
    unbounded = False
    for c in range(C):
        a, z = rho_set[c]
        if not np.isfinite(a) or not np.isfinite(z):
            unbounded = True
            continue
        if a <= 0.0 <= z:
            unbounded = True
            continue
        cand = [w[c] * delta[c] * beta[c] / a, w[c] * delta[c] * beta[c] / z]
        lo_tot += min(cand)
        hi_tot += max(cand)
    if unbounded:
        return {"L": -np.inf, "U": np.inf, "unbounded": True, "sum_delta2": s2,
                "sign_determinate": False}
    L, U = lo_tot / s2, hi_tot / s2
    return {"L": float(L), "U": float(U), "width": float(U - L),
            "unbounded": False, "sum_delta2": s2,
            "contains_zero": bool(L <= 0 <= U),
            "sign_determinate": bool(not (L <= 0 <= U))}


def breakdown_b(S, beta, d, ref=0, b_hi=1.0, tol=1e-4, use_sharp=False):
    """Largest idiosyncratic cross-correlation b at which the gradient keeps its sign.

    The A10b analogue of a breakdown value: report it next to the estimated
    gradient so the reader can judge how much administrative co-movement the
    conclusion tolerates.  Returns 0.0 when the sign is already indeterminate at
    b = 0 and b_hi when it survives the whole class.
    """
    fset = rho_sharp_set if use_sharp else rho_outer_set

    def determinate(b):
        rs = fset(S, ref=ref, b=b)["rho_set"]
        return gradient_bounds_over_rho(beta, d, rs)["sign_determinate"]

    if not determinate(0.0):
        return {"breakdown_b": 0.0, "note": "sign indeterminate already at b = 0"}
    if determinate(b_hi):
        return {"breakdown_b": float(b_hi), "note": "sign survives the whole class"}
    lo, hi = 0.0, b_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if determinate(mid):
            lo = mid
        else:
            hi = mid
    return {"breakdown_b": float(lo), "note": "bisection on the sign-determinacy frontier"}


# --------------------------------------------------------------------------
# the parametric d-shaped alternative, for comparison
# --------------------------------------------------------------------------

def rho_bias_d_shock(rho, d, varkappa):
    """Closed-form A10b bias: rho_hat_c = rho_c + varkappa d_c (Result L3.4)."""
    return np.asarray(rho, float) + float(varkappa) * np.asarray(d, float)
