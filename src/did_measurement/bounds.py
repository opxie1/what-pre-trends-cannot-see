"""Partial identification under monotone measurement response.

Additive map: if the policy weakly raises recording of category c in proportion
to its discretion d_c (measurement response kappa >= 0), the recorded event-study
coefficient beta(l) identifies ATT*(l) + kappa * d_c only up to unknown kappa.
For a researcher-chosen cap kappa_bar: ATT*(l) in [beta(l) - kappa_bar * d_c, beta(l)]
(signs mirror for kappa <= 0). The breakdown value is the smallest kappa_bar at
which the conclusion of interest (sign or significance) is overturned.

Multiplicative map, log outcomes: recording rate r in (0, 1] with baseline r0;
monotone response means log r shifts within [0, -log r0]. Worst-case bounds on
the latent log effect: [beta_log + log(r0), beta_log].

Confidence intervals for the partially identified parameter use Imbens-Manski
(2004): [L - c*se_L, U + c*se_U] with c solving
    Phi(c + (U - L)/max(se_L, se_U)) - Phi(-c) = 1 - alpha.
"""

import numpy as np
from scipy import stats, optimize


def imbens_manski_c(L, U, se_L, se_U, alpha=0.05):
    """Imbens-Manski critical value for a CI on the identified set [L, U]."""
    se_max = max(se_L, se_U, 1e-12)
    delta = max(U - L, 0.0) / se_max

    def f(c):
        return stats.norm.cdf(c + delta) - stats.norm.cdf(-c) - (1 - alpha)

    z2 = stats.norm.ppf(1 - alpha / 2)
    if f(z2) >= 0:
        lo, hi = 1e-6, z2
    else:
        lo, hi = z2, 10.0
    try:
        return float(optimize.brentq(f, lo, hi))
    except ValueError:
        return float(z2)


def sensitivity_interval(coef, se, d_c, kappa_bar, sign=+1, alpha=0.05):
    """Identified set and IM confidence interval for ATT*(l) given |kappa| <= kappa_bar.

    sign=+1: policy weakly raises recording (kappa in [0, kappa_bar]);
    sign=-1: weakly lowers it;
    sign= 0: TWO-SIDED class, m_c(l) in [-kappa_bar d_c, +kappa_bar d_c], which
             imposes no direction on the recording response.  This is the class
             a researcher must use when the point-identified decomposition
             returns measurement components of BOTH signs -- the crime
             application's bridge decomposition returns negative wedges for all
             six focal categories while its bounds impose m_c >= 0, and those
             two statements are jointly refuting under sign=+1.
    """
    shift = kappa_bar * d_c
    if sign > 0:
        L, U = coef - shift, coef
    elif sign < 0:
        L, U = coef, coef + shift
    else:
        L, U = coef - shift, coef + shift
    c = imbens_manski_c(L, U, se, se, alpha)
    return {"L": L, "U": U, "ci_lo": L - c * se, "ci_hi": U + c * se,
            "kappa_bar": kappa_bar, "im_c": c}


def measurement_class_consistency(m_hat, m_bar, sign=+1, tol=1e-12):
    """Is a point-identified wedge vector inside the class the bounds impose?

    Returns the fraction of components outside M = {0 <= sign*m_c <= m_bar_c}
    (or |m_c| <= m_bar_c when sign = 0), the worst violation, and a `refuted`
    flag.  A True flag means the two identification routes in the same
    application contradict each other: by Theorem 5.1 the identified set built
    on M is empty at the estimated wedges, so reporting both side by side is
    incoherent.
    """
    m_hat = np.atleast_1d(np.asarray(m_hat, float))
    m_bar = np.broadcast_to(np.asarray(m_bar, float), m_hat.shape)
    if sign == 0:
        viol = np.maximum(np.abs(m_hat) - m_bar, 0.0)
    else:
        s = 1.0 if sign > 0 else -1.0
        viol = np.maximum(-s * m_hat, 0.0) + np.maximum(s * m_hat - m_bar, 0.0)
    out_of_class = viol > tol
    return {"frac_outside": float(out_of_class.mean()),
            "n_outside": int(out_of_class.sum()),
            "max_violation": float(viol.max()) if viol.size else 0.0,
            "sign": int(sign), "refuted": bool(out_of_class.any()),
            "per_component": viol.tolist()}


def index_class_bounds(coef, se, d_variants, kappa_bar, sign=+1, alpha=0.05):
    """Identified set over a CLASS of admissible discretion indices.

    d_variants : iterable of candidate values of d_c for this category (e.g. the
    NCVS 2018 coding, the 2019 coding, a pure rank coding, and a perturbation
    ladder).  The reported set is the union over the class, which is the honest
    object when d_c is not itself identified: Proposition 6.14 protects the SIZE
    of the gradient test under a mismeasured index but not the ESTIMAND, whose
    realised scale factor b ranges over 1.000 / 0.677 / 1.028 / -0.605 / -0.705
    across the project's own five codings (two of five flip sign).
    """
    dv = np.atleast_1d(np.asarray(d_variants, float))
    sets = [sensitivity_interval(coef, se, float(d), kappa_bar, sign=sign, alpha=alpha)
            for d in dv]
    L = min(x["L"] for x in sets)
    U = max(x["U"] for x in sets)
    c = imbens_manski_c(L, U, se, se, alpha)
    return {"L": L, "U": U, "width": U - L, "ci_lo": L - c * se, "ci_hi": U + c * se,
            "n_variants": len(dv), "d_min": float(dv.min()), "d_max": float(dv.max()),
            "per_variant": sets, "im_c": c,
            "width_ratio_to_point": (U - L) / max(sets[0]["U"] - sets[0]["L"], 1e-300)}


def breakdown_over_index_class(coef, se, d_variants, alpha=0.05, sign=+1):
    """Range of the breakdown value over a class of admissible indices."""
    dv = np.atleast_1d(np.asarray(d_variants, float))
    vals = [breakdown_kappa(coef, se, float(d), alpha=alpha, sign=sign)["point"]
            for d in dv]
    v = np.asarray(vals, float)
    finite = v[np.isfinite(v)]
    return {"min": float(finite.min()) if finite.size else np.inf,
            "max": float(finite.max()) if finite.size else np.inf,
            "ratio": float(finite.max() / finite.min()) if finite.size and finite.min() > 0 else np.inf,
            "values": vals}


def breakdown_kappa(coef, se, d_c, alpha=0.05, sign=+1):
    """Breakdown values of the measurement-response magnitude kappa.

    'point': kappa at which the identified set first includes 0.
    'stat':  kappa at which the IM confidence interval first includes 0
             (accounts for sampling noise; NaN when already insignificant at kappa=0).
    Defined for conclusions about a positive recorded effect when sign=+1
    (mirrored internally for negative coefficients).
    """
    if d_c <= 0:
        return {"point": np.inf, "stat": np.inf}
    s = np.sign(coef) if coef != 0 else 1.0
    b = abs(coef)
    point = b / d_c
    z = stats.norm.ppf(1 - alpha / 2)  # conservative
    stat_bd = (b - z * se) / d_c if b > z * se else np.nan
    return {"point": float(point), "stat": float(stat_bd) if stat_bd == stat_bd else np.nan,
            "direction": float(s)}


def recording_bounds(beta_log, se, r0, alpha=0.05):
    """Worst-case bounds on the latent log effect under monotone recording response.

    Recording rate rises from baseline r0 toward at most 1 in treated-post cells:
    the measurement component of the log event-study coefficient lies in
    [0, -log(r0)], so ATT*_log in [beta_log + log(r0), beta_log].
    """
    if not (0 < r0 <= 1):
        raise ValueError("r0 must be in (0, 1]")
    L, U = beta_log + np.log(r0), beta_log
    c = imbens_manski_c(L, U, se, se, alpha)
    return {"L": L, "U": U, "ci_lo": L - c * se, "ci_hi": U + c * se,
            "r0": r0, "im_c": c}


# cohort

_TOL = 1e-12


def _cohort_interval(beta_row, m_bar, rho, sign):
    """Sharp interval for one cohort's latent x at one horizon.

    Class M = {m : 0 <= sign*m_c <= m_bar_c} for sign = +/-1, and the two-sided
    class M = {m : |m_c| <= m_bar_c} for sign = 0.  Returns (L, U); L > U means
    the set is empty, i.e. the model is refuted at this cap.
    """
    lo, hi = -np.inf, np.inf
    for b, mb, r in zip(beta_row, m_bar, rho):
        if r == 0:
            continue  # loading
        if sign == 0:
            e1, e2 = (b - mb) / r, (b + mb) / r
        else:
            e1 = b / r
            e2 = (b - sign * mb) / r
        lo = max(lo, min(e1, e2))
        hi = min(hi, max(e1, e2))
    return lo, hi


def cohort_sharp_interval(beta, weights=None, m_bar=None, rho=None, sign=+1,
                          aggregate="minkowski"):
    """Identified set for a cohort-aggregated latent effect at one horizon.

    beta      : (G, C) recorded event-study coefficients, cohort by category.
    weights   : (G,) aggregation weights (default equal); normalized to sum 1.
    m_bar     : (C,) nonnegative caps on the measurement response, e.g. kappa_bar*d_c.
    rho       : (C,) relative loadings (default 1).
    sign      : +1 recording weakly raised, -1 weakly lowered.
    aggregate : 'minkowski' (SHARP, Corollary 1') or 'naive' (the shipped
                pre-fix behaviour: build the set from sum_g w_g beta^g).

    Returns dict with L, U, width, empty, per_cohort, aggregate, and — for
    convenience in comparisons — naive_L/naive_U/naive_width/width_ratio.
    """
    B = np.atleast_2d(np.asarray(beta, float))
    G, C = B.shape
    w = np.ones(G) / G if weights is None else np.asarray(weights, float)
    if w.shape != (G,):
        raise ValueError("weights must have one entry per cohort")
    w = w / w.sum()
    m_bar = np.zeros(C) if m_bar is None else np.broadcast_to(
        np.asarray(m_bar, float), (C,))
    if np.any(m_bar < 0):
        raise ValueError("m_bar must be nonnegative (sign carries the direction)")
    rho = np.ones(C) if rho is None else np.broadcast_to(np.asarray(rho, float), (C,))
    sign = 0.0 if sign == 0 else (1.0 if sign > 0 else -1.0)

    per = [_cohort_interval(B[g], m_bar, rho, sign) for g in range(G)]
    empty_g = [lo > hi + _TOL for lo, hi in per]
    Lm = float(sum(wg * lo for wg, (lo, hi) in zip(w, per)))
    Um = float(sum(wg * hi for wg, (lo, hi) in zip(w, per)))
    m_empty = any(empty_g)

    bbar = w @ B
    Ln, Un = _cohort_interval(bbar, m_bar, rho, sign)
    n_empty = Ln > Un + _TOL

    if aggregate == "minkowski":
        L, U, empty = Lm, Um, m_empty
    elif aggregate == "naive":
        L, U, empty = float(Ln), float(Un), bool(n_empty)
    else:
        raise ValueError("aggregate must be 'minkowski' or 'naive'")

    wn = max(Un - Ln, 0.0)
    wm = max(Um - Lm, 0.0)
    return {"L": L, "U": U, "width": max(U - L, 0.0), "empty": bool(empty),
            "per_cohort": [(float(a), float(b)) for a, b in per],
            "weights": w.tolist(), "aggregate": aggregate,
            "naive_L": float(Ln), "naive_U": float(Un), "naive_width": wn,
            "minkowski_L": Lm, "minkowski_U": Um, "minkowski_width": wm,
            "width_ratio": (wm / wn) if wn > _TOL else (1.0 if wm <= _TOL else np.inf)}


def aggregate_sensitivity_interval(beta, se, d, kappa_bar, weights=None, rho=None,
                                   sign=+1, alpha=0.05, aggregate="minkowski"):
    """cohort_sharp_interval with m_bar = kappa_bar * d, plus an Imbens-Manski CI.

    se is the standard error of the cohort-aggregated recorded coefficient; the
    same se is used at both endpoints, as in sensitivity_interval.
    """
    d = np.asarray(d, float)
    s = cohort_sharp_interval(beta, weights=weights, m_bar=kappa_bar * d, rho=rho,
                              sign=sign, aggregate=aggregate)
    c = imbens_manski_c(s["L"], s["U"], se, se, alpha)
    s.update({"ci_lo": s["L"] - c * se, "ci_hi": s["U"] + c * se,
              "kappa_bar": float(kappa_bar), "im_c": c})
    return s


def aggregate_breakdown_kappa(beta, d, weights=None, rho=None, sign=+1,
                              aggregate="minkowski", kappa_hi=1e6, tol=1e-10):
    """Smallest kappa_bar at which the aggregated identified set contains 0.

    The set is monotone in kappa_bar (it grows in the -sign direction), so the
    breakdown value is found by bisection.  Returns inf if 0 is never included
    below kappa_hi, 0.0 if the set already contains 0.
    """
    d = np.asarray(d, float)

    def contains_zero(kb):
        s = cohort_sharp_interval(beta, weights=weights, m_bar=kb * d, rho=rho,
                                  sign=sign, aggregate=aggregate)
        return (not s["empty"]) and (s["L"] <= 0.0 <= s["U"])

    if contains_zero(0.0):
        return 0.0
    if not contains_zero(kappa_hi):
        return float(np.inf)
    lo, hi = 0.0, kappa_hi
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if contains_zero(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)
