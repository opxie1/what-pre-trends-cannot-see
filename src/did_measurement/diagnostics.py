"""Testable diagnostics for the untreated-comovement assumption A10 (defect X4).

Lemma 6.3 identifies the relative loadings rho_c = theta_c/theta_{c*} from the
untreated cross-category covariance.  Its proof needs TWO restrictions that were
invisible in the source tracks:

  A10a (cross-lag orthogonality)  Cov(eta_{isc}, eta_{itc'}) = 0 for all c != c'
       and ALL s, t -- not merely s = t.  The residual maker subtracts a
       within-unit time average, so a purely lagged cross-category covariance
       contaminates the CONTEMPORANEOUS residual covariance.
  A10b (diagonal idiosyncratic covariance)  the untreated covariance is
       sigma_F^2 theta theta' + Psi with Psi DIAGONAL: no second common shock.
       A records-management shock loading on the discretion index d gives
       Sigma = sF2 theta theta' + sG2 d d' + Psi and biases rho_hat by exactly
       varkappa_s * d_c, manufacturing a d-collinear wedge under measurement
       invariance.

Both are population biases.  Both, however, have observable content: under A10
the whole array of untreated cross-category residual covariances, at EVERY lag,
is rank one with a COMMON loading vector,

        Lambda_h(c, c') = theta_c theta_{c'} V_F(h),    c != c',  h = 0, 1, ...,

which is overidentified as soon as C(C-1)/2 + H*C*(C-1) > C + H.  With one lag
this is satisfied already at C = 3, so the diagnostic is available in designs
where the across-instrument spread of Proposition 6.7(i) (which needs C >= 4)
is not.

This module implements:
  comovement_moments    -- Lambda_h and its unit-level influence functions
  comovement_test       -- minimum-distance overidentification test of A10
  discretion_shock_test -- a focused 1-df test of the A10b alternative
                           Sigma_offdiag = theta_c theta_k + xi d_c d_k, with the
                           xi-corrected relative loadings as a by-product

RECOMMENDED PRACTICE.  Run BOTH, at nominal 0.025 each:

  * comovement_test with the default weight='diag' and the sign-flip multiplier
    bootstrap p-value.  Simulated size 0.046-0.074 over N in {51, 100, 400},
    C in {3, 5}, lags 0-1, with or without a serially correlated factor and with
    or without own-category serial correlation (1,000 replications each).  Power
    1.000 against cross-lag leakage at lambda >= 0.5 (0.55-0.91 at lambda=0.25);
    against a d-loading shock it is the WEAKER instrument (0.086 at
    sigma_G^2=0.25 with an i.i.d. factor, 0.294 at rho_F=0.5, 0.97-1.00 at
    sigma_G^2 >= 0.5 with rho_F=0.5).
    weight='optimal' is far more powerful against the d-shock (0.76 at
    sigma_G^2=0.25) but its J statistic over-rejects badly whenever the moment
    count is not small relative to the number of clusters -- size 0.098 at M=30,
    N=400 and 0.813 at M=30, N=51, the same many-restrictions pathology the
    project documents for the discretion-gradient Wald test.  Use it as a
    descriptive statistic, not as a test.
  * discretion_shock_test, which targets the one alternative that mimics the
    paper's own object.  Size 0.039-0.065; power 0.996 at sigma_G^2 = 0.25.
    Its rejection is NOT proof of a d-shock: it also rejects under cross-lag
    leakage (0.79-1.00), where xi = 0 is true, so read it only conditional on
    the omnibus test not rejecting.
"""

import numpy as np
from scipy import stats, optimize

from .estimators import demean_two_way

__all__ = ["comovement_moments", "moments_from_cube", "comovement_test", "discretion_shock_test",
           "robust_relative_loadings"]


# moments

def _residual_cube(df, outcome="Y", untreated_only=True):
    """Residualize on unit-by-cat and period-by-cat FE; return (units, T, C) cube
    with NaN where the cell is absent, plus unit and category labels."""
    dat = df[df["D"] == 0] if untreated_only else df
    dat = dat.reset_index(drop=True)
    cats = np.sort(dat["cat"].unique())
    span = int(cats.max()) + 1
    uc = dat["unit"].to_numpy() * span + dat["cat"].to_numpy()
    pc = dat["period"].to_numpy() * span + dat["cat"].to_numpy()
    R = demean_two_way(dat[[outcome]].to_numpy(float), uc, pc)[:, 0]
    units, ui = np.unique(dat["unit"].to_numpy(), return_inverse=True)
    pers, pi = np.unique(dat["period"].to_numpy(), return_inverse=True)
    ci = np.searchsorted(cats, dat["cat"].to_numpy())
    cube = np.full((len(units), len(pers), len(cats)), np.nan)
    cube[ui, pi, ci] = R
    return cube, units, cats


def moments_from_cube(cube, cats=None, max_lag=1, symmetrize=False, demean=False):
    """Same moments as comovement_moments, from an already-built (N, T, C) cube.

    demean=True applies the balanced-panel two-way (unit, period) within-category
    projection; use it when the cube is raw rather than already residualized.
    """
    cube = np.asarray(cube, float)
    N, T, C = cube.shape
    if cats is None:
        cats = np.arange(C)
    if demean:
        cube = (cube - np.nanmean(cube, axis=1, keepdims=True)
                - np.nanmean(cube, axis=0, keepdims=True)
                + np.nanmean(cube, axis=(0, 1), keepdims=True))
    if C < 3:
        raise ValueError("comovement diagnostics need at least 3 categories")
    keys, num, cnt = [], [], []
    for h in range(max_lag + 1):
        A = cube[:, h:, :]
        B = cube[:, :T - h, :]
        pairs = [(c, k) for c in range(C) for k in range(C) if c != k]
        if h == 0 or symmetrize:
            pairs = [(c, k) for c, k in pairs if c < k]
        for c, k in pairs:
            if h == 0 or not symmetrize:
                p = A[:, :, c] * B[:, :, k]
            else:
                p = 0.5 * (A[:, :, c] * B[:, :, k] + A[:, :, k] * B[:, :, c])
            ok = np.isfinite(p)
            num.append(np.where(ok, p, 0.0).sum(axis=1))
            cnt.append(ok.sum(axis=1).astype(float))
            keys.append((h, int(c), int(k)))
    num = np.array(num).T  # note
    cnt = np.array(cnt).T
    tot = cnt.sum(axis=0)
    if np.any(tot <= 0):
        raise ValueError("a category pair has no usable untreated observations")
    g = num.sum(axis=0) / tot
    psi = (num - cnt * g) / (tot / N)  # linearization
    return {"g": g, "psi": psi, "keys": keys, "cats": np.asarray(cats),
            "n_units": N, "n_pairs": len(keys), "max_lag": max_lag}


def comovement_moments(df, outcome="Y", max_lag=1, symmetrize=False):
    """Untreated cross-category residual covariances at lags 0..max_lag.

    Returns dict:
      'g'      (M,) stacked Lambda_h(c, k), c != k
      'psi'    (N, M) unit-level influence functions, mean zero, so that
               g_hat - g_0 = mean_i psi_i + o_p(N^{-1/2})
      'keys'   list of (h, c_index, k_index)
      'cats'   category labels, 'n_units'
    """
    cube, _units, cats = _residual_cube(df, outcome)
    return moments_from_cube(cube, cats, max_lag, symmetrize, demean=False)


# model

def _model(keys, C, theta, V):
    """s(pi): Lambda_h(c,k) = theta_c theta_k V_h, with V_0 = 1."""
    out = np.empty(len(keys))
    for j, (h, c, k) in enumerate(keys):
        out[j] = theta[c] * theta[k] * (1.0 if h == 0 else V[h - 1])
    return out


def _jac(keys, C, theta, V):
    H = len(V)
    J = np.zeros((len(keys), C + H))
    for j, (h, c, k) in enumerate(keys):
        v = 1.0 if h == 0 else V[h - 1]
        J[j, c] += theta[k] * v
        J[j, k] += theta[c] * v
        if h > 0:
            J[j, C + h - 1] = theta[c] * theta[k]
    return J


def _start(g, keys, C, H):
    """Closed-form start: theta from lag-0 triples, V_h from a lag-h/lag-0 ratio."""
    S = np.full((C, C), np.nan)
    for j, (h, c, k) in enumerate(keys):
        if h == 0:
            S[c, k] = S[k, c] = g[j]
    theta = np.ones(C)
    for c in range(C):
        cand = []
        for k in range(C):
            for m in range(C):
                if len({c, k, m}) == 3 and np.isfinite(S[k, m]) and abs(S[k, m]) > 1e-12:
                    x = S[c, k] * S[c, m] / S[k, m]
                    if x > 0:
                        cand.append(np.sqrt(x))
        theta[c] = float(np.median(cand)) if cand else 1.0
    for c in range(1, C):
        if np.isfinite(S[0, c]) and S[0, c] < 0:
            theta[c] = -theta[c]
    V = []
    for h in range(1, H + 1):
        num, den = [], []
        for j, (hh, c, k) in enumerate(keys):
            if hh == h:
                num.append(g[j])
                den.append(theta[c] * theta[k])
        num = np.array(num)
        den = np.array(den)
        V.append(float(np.sum(num * den) / max(np.sum(den ** 2), 1e-12)))
    return theta, np.array(V)


def _md_fit(g, W_half, keys, C, H, theta0, V0):
    def resid(p):
        return W_half @ (g - _model(keys, C, p[:C], p[C:]))

    p0 = np.concatenate([theta0, V0])
    sol = optimize.least_squares(resid, p0, method="lm", max_nfev=20000)
    return sol.x, float(sol.cost * 2.0)


def _omega_half_inv(Omega, rcond=1e-8):
    """Omega^{-1/2} (symmetric), pseudo-inverting near-null directions."""
    ev, U = np.linalg.eigh(0.5 * (Omega + Omega.T))
    keep = ev > rcond * max(ev.max(), 1e-300)
    ev2 = np.where(keep, ev, np.inf)
    return (U * (1.0 / np.sqrt(ev2))) @ U.T, int(keep.sum())


def comovement_test(df, outcome="Y", max_lag=1, symmetrize=False, n_boot=999,
                    seed=0, moments=None, weight="diag"):
    """Minimum-distance overidentification test of A10 (A10a and A10b jointly).

    H0: there exist theta in R^C and V_1..V_H with
        Lambda_h(c, k) = theta_c theta_k V_h  for all c != k, all h <= max_lag
        (V_0 = 1).  Implied by A10 plus the one-factor untreated law.
    H1: any departure -- a second common shock (non-diagonal Psi, A10b) or a
        nonzero cross-category covariance at a nonzero lag (A10a).

    Statistic: J = N min_pi (g - s(pi))' Omega^{-1} (g - s(pi)), Omega the
    unit-clustered covariance of the moment influence functions.  Reference
    distributions: chi2 with df = M - C - H, and a Rademacher multiplier
    (sign-flip) bootstrap of the asymptotically equivalent quadratic form
    N g*' [Om^-1 - Om^-1 J (J' Om^-1 J)^-1 J' Om^-1] g*, the few-cluster-safe
    version.

    Returns dict: stat, df, pvalue_chi2, pvalue_boot, crit_boot_95, theta_hat,
    V_hat, n_units, n_moments.
    """
    mm = moments or comovement_moments(df, outcome, max_lag, symmetrize)
    g, psi, keys = mm["g"], mm["psi"], mm["keys"]
    N, M = psi.shape
    C, H = len(mm["cats"]), max_lag
    P = C + H
    if M <= P:
        return {"stat": np.nan, "df": M - P, "pvalue_chi2": np.nan,
                "pvalue_boot": np.nan, "crit_boot_95": np.nan,
                "theta_hat": None, "V_hat": None, "n_units": N, "n_moments": M,
                "pvalue": np.nan, "calibration": "none available",
                "note": "not overidentified: M <= C + H"}
    Omega = (psi.T @ psi) / N
    if weight == "optimal":
        Wh, rk = _omega_half_inv(Omega)
    elif weight == "diag":
        # bootstrap
        Wh = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Omega), 1e-300)))
        rk = M
    else:
        raise ValueError("weight must be 'optimal' or 'diag'")
    th0, V0 = _start(g, keys, C, H)
    p, obj = _md_fit(g, Wh, keys, C, H, th0, V0)
    stat = float(N * obj)
    dfree = M - P
    Jm = _jac(keys, C, p[:C], p[C:])
    A = Wh @ Jm
    Pm = A @ np.linalg.pinv(A.T @ A) @ A.T
    Q = Wh @ (np.eye(M) - Pm) @ Wh
    out = {"stat": stat, "df": int(dfree), "weight": weight,
           "pvalue_chi2": (float(stats.chi2.sf(stat, dfree))
                           if weight == "optimal" else np.nan),
           "theta_hat": p[:C].tolist(), "V_hat": p[C:].tolist(),
           "n_units": N, "n_moments": M, "omega_rank": rk}
    if weight == "diag":
        # satterthwaite
        QO = Q @ Omega
        tr1 = float(np.trace(QO))
        tr2 = float(np.trace(QO @ QO))
        if tr2 > 0:
            nu = tr1 ** 2 / tr2
            out["df_satterthwaite"] = nu
            out["pvalue_satterthwaite"] = float(
                stats.chi2.sf(stat * nu / max(tr1, 1e-300), nu))
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        W = rng.choice([-1.0, 1.0], size=(n_boot, N))
        GS = (W @ psi) / N  # n_boot
        boot = N * np.einsum("bm,mn,bn->b", GS, Q, GS)
        out["pvalue_boot"] = float((1.0 + np.sum(boot >= stat)) / (1.0 + n_boot))
        out["crit_boot_95"] = float(np.quantile(boot, 0.95))
    else:
        out["pvalue_boot"] = np.nan
        out["crit_boot_95"] = np.nan
    _preregistered_pvalue(out)
    return out


def _preregistered_pvalue(out):
    """The single p-value to report: max of the two valid calibrations.

    The fixed-diagonal-weight statistic admits two asymptotically valid
    references, the Rademacher multiplier bootstrap and the Satterthwaite
    chi-square, and both are correctly sized in the L3.9 grid (0.046-0.074 and
    0.043-0.078).  In the applications they disagree by up to an order of
    magnitude (medicaid annual5: 0.022 against 0.112), and reporting whichever
    is smaller is a selection the file cannot defend.  Because each is
    individually valid, their MAXIMUM is also a valid p-value --
    P(max <= a) <= min_i P(p_i <= a) <= a -- and it is the only choice that does
    not require ranking the two references.  It is therefore the pre-registered
    reported p-value, and both inputs stay in the output so a reader can see the
    disagreement.  With the optimal weight there is only one reference and
    `pvalue` is `pvalue_chi2`; that statistic is oversized whenever M/N is not
    tiny (0.81 at M=30, N=51) and is a descriptive quantity, not a test.
    """
    cand = [out.get("pvalue_boot"), out.get("pvalue_satterthwaite"),
            out.get("pvalue_chi2")]
    cand = [float(p) for p in cand if p is not None and np.isfinite(p)]
    if not cand:
        out["pvalue"] = np.nan
        out["calibration"] = "none available"
        return out
    if out.get("weight") == "optimal":
        out["pvalue"] = float(out.get("pvalue_chi2", cand[0]))
        out["calibration"] = "chi2 (optimal weight; oversized unless M/N tiny)"
        return out
    out["pvalue"] = float(max(cand))
    out["calibration"] = "max(signflip bootstrap, Satterthwaite chi2)"
    out["pvalue_spread"] = float(max(cand) - min(cand))
    return out


# focused

def discretion_shock_test(df, d, outcome="Y", bridge_cat=0, n_boot=999, seed=0,
                          moments=None):
    """Focused test of the records-management alternative (A10b failure).

    Model on the lag-0 untreated off-diagonal covariances:
        S_ck = theta_c theta_k + xi d_c d_k,    c != k,
    i.e. Sigma = sF2 theta theta' + sG2 d d' + Psi with xi = sG2/sF2 after the
    V_F = 1 normalization.  H0: xi = 0 (A10b holds).  One extra parameter, so the
    fit is overidentified whenever C(C-1)/2 >= C + 2, i.e. C >= 5; xi is
    just-identified at C = 4 and testable there too.

    Returns dict with xi_hat, se_xi (unit-clustered, minimum-distance), z,
    pvalue, pvalue_boot, and rho_corrected = theta_c/theta_bridge from the
    xi-augmented fit -- the A10b-robust replacement for
    bridge.estimate_relative_loadings.
    """
    mm = moments or comovement_moments(df, outcome, max_lag=0)
    g, psi, keys = mm["g"], mm["psi"], mm["keys"]
    cats = list(mm["cats"])
    d = np.asarray(d, float)
    if len(d) != len(cats):
        raise ValueError("d must have one entry per category present")
    N, M = psi.shape
    C = len(cats)
    P = C + 1
    if M < P:
        return {"xi_hat": np.nan, "se_xi": np.nan, "z": np.nan, "pvalue": np.nan,
                "pvalue_boot": np.nan, "rho_corrected": None,
                "note": "needs C >= 4 for an identified xi"}
    Omega = (psi.T @ psi) / N
    Wh = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Omega), 1e-300)))

    def s_of(p):
        th, xi = p[:C], p[C]
        return np.array([th[c] * th[k] + xi * d[c] * d[k] for (_, c, k) in keys])

    def jac(p):
        th = p[:C]
        Jm = np.zeros((M, P))
        for j, (_, c, k) in enumerate(keys):
            Jm[j, c] += th[k]
            Jm[j, k] += th[c]
            Jm[j, C] = d[c] * d[k]
        return Jm

    th0, _ = _start(g, keys, C, 0)
    sol = optimize.least_squares(lambda p: Wh @ (g - s_of(p)),
                                 np.concatenate([th0, [0.0]]),
                                 method="lm", max_nfev=20000)
    p = sol.x
    Jm = jac(p)
    A = Wh @ Jm
    AtA_inv = np.linalg.pinv(A.T @ A)
    Bm_ = AtA_inv @ A.T @ Wh  # linear
    Vp = (Bm_ @ Omega @ Bm_.T) / N  # sandwich
    se = float(np.sqrt(max(Vp[C, C], 0.0)))
    xi = float(p[C])
    z = xi / se if se > 0 else np.nan
    out = {"xi_hat": xi, "se_xi": se, "z": float(z),
           "pvalue": float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan,
           "theta_hat": p[:C].tolist(), "n_units": N, "n_moments": M}
    if n_boot and n_boot > 0 and se > 0:
        rng = np.random.default_rng(seed)
        W = rng.choice([-1.0, 1.0], size=(n_boot, N))
        GS = (W @ psi) / N
        DB = GS @ Bm_.T
        zb = DB[:, C] / se
        out["pvalue_boot"] = float((1.0 + np.sum(np.abs(zb) >= abs(z)))
                                   / (1.0 + n_boot))
    else:
        out["pvalue_boot"] = np.nan
    bi = int(np.argmin(np.abs(np.asarray(cats) - bridge_cat)))
    th = p[:C]
    out["rho_corrected"] = {int(cats[c]): float(th[c] / th[bi])
                            for c in range(C) if c != bi}
    # bootstrap
    cand = [q for q in (out.get("pvalue"), out.get("pvalue_boot"))
            if q is not None and np.isfinite(q)]
    out["pvalue_normal"] = out.get("pvalue")
    out["pvalue_preregistered"] = float(max(cand)) if cand else np.nan
    out["calibration"] = "max(normal, signflip bootstrap)"
    return out


def robust_relative_loadings(df, d, outcome="Y", bridge_cat=0):
    """A10b-robust relative loadings: the theta of the xi-augmented fit.

    Consistent when the untreated covariance is sF2 theta theta' + sG2 d d' + Psi
    (a records-management shock loading on discretion), where the shipped
    cross-category IV bridge.estimate_relative_loadings is biased by
    varkappa_s * d_c.  It does NOT repair a violation of A10a.
    """
    r = discretion_shock_test(df, d, outcome=outcome, bridge_cat=bridge_cat,
                              n_boot=0)
    return {"rho": r["rho_corrected"], "xi_hat": r["xi_hat"], "se_xi": r["se_xi"]}
