"""Corrected efficiency formulas (fix for defect X3).

Two corrections carried over from `theory/efficiency_review.md` (R6, R8) into
executable code.  Neither object existed in `src/` before; the false versions
are kept behind `corrected=False` so the size of the error is reproducible.

R8 -- the orthogonal / minimum-distance loading score.  As printed,
    Phi^MD_i = -(e_c - e_0)' (J' Om^-1 J)^-1 J' Om^-1 phi^V_i
is the influence function of log rho_c, whereas the efficient influence function
decomposition psi_beta = rho_c psi_b + b_0 psi_rho uses psi_rho = rho_c psi_logrho.
The printed score is therefore short by a factor rho_c, so the LOADING CHANNEL of
the variance, b_0^2 V_rho = rho_c^2 b_0^2 V_logrho, is reported as b_0^2 V_logrho:
too small by the factor rho_c^2.

R6 -- the M2 (pooled-category) variance.  The cross-category regression
    beta_c = theta_c tau + d_c varkappa
has errors that are NOT independent across c: every beta_hat_c loads the common
factor, so Var(beta_hat) = (A/N)(sigma_v^2 theta theta' + Psi), rank one plus
diagonal.  The corrected GLS variance is

    N V^M2_tau = A [ Z/(sv2 (1+Z)) - S_td^2 / ((1+Z)^2 S_dd - sv2 (1+Z) S_td^2) ]^-1

with zeta_a = sv2 theta_a^2 / psi_a, Z = sum_a zeta_a, S_td = sum_c theta_c d_c/psi_c,
S_dd = sum_c d_c^2/psi_c.
"""

import numpy as np

__all__ = ["loading_channel_variance", "psi_rho_from_log", "eif_variance",
           "bridge_delta_se", "m2_variance", "m2_variance_gls", "m2_zeta"]


# ----------------------------------------------------------------- R8

def psi_rho_from_log(psi_log_rho, rho):
    """psi_rho = rho * psi_{log rho}.  The printed Phi^MD omits the factor rho."""
    return float(rho) * np.asarray(psi_log_rho, float)


def loading_channel_variance(b0, V_log_rho, rho, corrected=True):
    """Variance contribution of the estimated loading to beta_c = rho_c * b_0.

    corrected=True : b0^2 * rho^2 * V_log_rho  (= b0^2 V_rho)   [Theorem 7.9(ii) fixed]
    corrected=False: b0^2 * V_log_rho                            [as printed]
    """
    base = float(b0) ** 2 * float(V_log_rho)
    return base * (float(rho) ** 2 if corrected else 1.0)


def eif_variance(rho, b0, psi_b, psi_log_rho, corrected=True):
    """Mean square of psi_beta = rho psi_b + b0 psi_rho, with psi_rho = rho psi_logrho."""
    psi_b = np.asarray(psi_b, float)
    pl = np.asarray(psi_log_rho, float)
    psi = float(rho) * psi_b + float(b0) * (float(rho) if corrected else 1.0) * pl
    return float(np.mean(psi ** 2))


def bridge_delta_se(rho, b0, var_b0, var_rho, cov=0.0, corrected=True):
    """Delta-method SE of beta_c = rho * b0.

    corrected=True uses Var(rho); corrected=False reproduces the printed score,
    which supplies Var(log rho) = Var(rho)/rho^2 in its place.
    """
    vr = float(var_rho) if corrected else float(var_rho) / float(rho) ** 2
    v = float(rho) ** 2 * float(var_b0) + float(b0) ** 2 * vr \
        + 2.0 * float(rho) * float(b0) * float(cov)
    return float(np.sqrt(max(v, 0.0)))


# ----------------------------------------------------------------- R6

def m2_zeta(theta, psi, sigma_v2):
    """Per-category signal-to-noise ratios zeta_a = sigma_v^2 theta_a^2 / psi_a."""
    theta = np.asarray(theta, float)
    psi = np.asarray(psi, float)
    return float(sigma_v2) * theta ** 2 / psi


def m2_variance(theta, d, psi, sigma_v2, A=1.0, corrected=True):
    """N * Var(tau_hat) for the pooled-category (M2) estimator.

    corrected=True : the GLS variance under Var(beta_hat) = (A/N)(sv2 th th' + Psi)
                     (Proposition 7.6', closed form).
    corrected=False: the printed formula, which assumes the category errors are
                     independent, i.e. Var(beta_hat) = (A/N) Psi.
    """
    theta = np.asarray(theta, float)
    d = np.asarray(d, float)
    psi = np.asarray(psi, float)
    sv2 = float(sigma_v2)
    if not corrected:
        return m2_variance_gls(theta, d, np.diag(psi), A)
    Z = float(np.sum(m2_zeta(theta, psi, sv2)))
    S_td = float(np.sum(theta * d / psi))
    S_dd = float(np.sum(d ** 2 / psi))
    first = Z / (sv2 * (1.0 + Z))
    den = (1.0 + Z) ** 2 * S_dd - sv2 * (1.0 + Z) * S_td ** 2
    second = 0.0 if abs(den) < 1e-300 else S_td ** 2 / den
    br = first - second
    if br <= 0:
        return float(np.inf)
    return float(A) / br


def m2_variance_gls(theta, d, Omega, A=1.0):
    """N * Var(tau_hat) by direct GLS inversion with error covariance (A/N) Omega."""
    theta = np.asarray(theta, float)
    d = np.asarray(d, float)
    X = np.column_stack([theta, d])
    Oi = np.linalg.pinv(np.asarray(Omega, float))
    M = X.T @ Oi @ X
    if abs(np.linalg.det(M)) < 1e-300:
        return float(np.inf)
    return float(A) * float(np.linalg.inv(M)[0, 0])
