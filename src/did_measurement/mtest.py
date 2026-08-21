"""Falsification tests: the discretion-gradient test and standard pre-trend tests.

Discretion-gradient test (project Theorem 2). Under measurement invariance plus
mean-independence of latent treatment effects from the recording-discretion index
d_c, event-study coefficients do not vary systematically with d_c across outcome
categories at a fixed event time. Treatment-induced measurement change loads on
the d_c gradient; parallel-trends violations load on event time but not on d_c.
The two failure modes therefore have (approximately) orthogonal signatures.

Implementation: stack focal categories; absorb unit-by-category and
period-by-category fixed effects (each category keeps its own trends and level);
regress the outcome on common event-time dummies plus event-time dummies
interacted with the centered discretion index; Wald-test all post-period
interactions jointly, clustering by unit.

Loading adjustment (default ON). With heterogeneous factor loadings theta_c, the
common event dummies force a common latent effect, and any *realized* correlation
between theta_c and d_c across the (few) categories loads onto the interaction,
over-rejecting under measurement invariance. The fix: estimate relative loadings
rho_c from untreated comovement (cross-category IV, see bridge.py), normalize the
outcome to Y/rho_c, and run the test on the normalized outcome, whose latent
event-study effects are equal across categories by construction under the
one-factor model. loading_adjust=False recovers the naive test (kept so its
over-rejection can be quantified in simulations).
"""

import numpy as np
import pandas as pd
from scipy import stats

from .estimators import build_event_dummies, demean_two_way, cluster_ols
from .bridge import estimate_relative_loadings

RHO_GUARD = 0.1  # drop categories whose |estimated loading| falls below this


def discretion_gradient_test(df, outcome="Y", k_pre=5, k_post=6,
                             bridge_cat=0, use_bridge=False, cluster_col="unit",
                             loading_adjust=True):
    """Joint Wald test that post-period event-time x discretion interactions are zero.

    Returns dict: 'stat', 'df', 'pvalue', 'slopes' (per-horizon interaction
    coefficients with SEs), 'pre_slopes_pvalue' (same test on leads, a placebo),
    'rho' (loading normalization used, if any).
    """
    dat = df if use_bridge else df[df["cat"] != bridge_cat]
    dat = dat.reset_index(drop=True)

    rho_used = None
    ycol = outcome
    if loading_adjust and dat["cat"].nunique() >= 3:
        ref = dat.groupby("cat")["d"].first().idxmin()
        ld = estimate_relative_loadings(dat, outcome=outcome, bridge_cat=ref)
        rho = {int(ref): 1.0, **{int(c): v for c, v in ld["rho"].items()}}
        scale = dat["cat"].map(rho).astype(float)
        keep = scale.abs() > RHO_GUARD
        dat = dat.loc[keep].reset_index(drop=True)
        scale = scale.loc[keep].reset_index(drop=True)
        dat = dat.copy()
        dat["_y_adj"] = dat[outcome].to_numpy() / scale.to_numpy()
        ycol = "_y_adj"
        rho_used = rho
    n_cat_span = int(dat["cat"].max()) + 1
    unit_cat = dat["unit"].to_numpy() * n_cat_span + dat["cat"].to_numpy()
    period_cat = dat["period"].to_numpy() * n_cat_span + dat["cat"].to_numpy()

    Xd, labels = build_event_dummies(dat["event_time"].to_numpy(), k_pre, k_post)
    d_center = dat["d"].to_numpy() - dat["d"].mean()
    Xint = Xd * d_center[:, None]
    y = dat[ycol].to_numpy(float)[:, None]

    M = np.column_stack([y, Xd, Xint])
    M = demean_two_way(M, unit_cat, period_cat)
    y_t = M[:, 0]
    X_t = M[:, 1:]
    beta, V, n, G = cluster_ols(X_t, y_t, dat[cluster_col].to_numpy())

    k = len(labels)
    int_idx = np.arange(k, 2 * k)                       # interaction block
    post_pos = [j for j, l in enumerate(labels) if l >= 0]
    pre_pos = [j for j, l in enumerate(labels) if l < -1]

    def _wald(pos):
        idx = int_idx[pos]
        b = beta[idx]
        Vb = V[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.pinv(Vb) @ b)
        q = len(idx)
        return stat, q, float(stats.chi2.sf(stat, q))

    stat, q, pval = _wald(post_pos)
    _, _, pval_pre = _wald(pre_pos) if pre_pos else (np.nan, 0, np.nan)

    slopes = pd.DataFrame({
        "ell": labels,
        "slope": beta[int_idx],
        "se": np.sqrt(np.clip(np.diag(V)[int_idx], 0, None)),
        "main": beta[:k],
    })
    return {"stat": stat, "df": q, "pvalue": pval,
            "pre_slopes_pvalue": pval_pre, "slopes": slopes,
            "n": n, "n_clusters": G, "rho": rho_used}


def pretrend_test(es_result, max_leads=None):
    """Joint Wald test that event-study leads (ell < -1) are zero.

    es_result: output of estimators.event_study_twfe.
    """
    labels = es_result["labels"]
    V = es_result["vcov"]
    coefs = es_result["table"]["coef"].to_numpy()
    lead_idx = [j for j, l in enumerate(labels) if l < -1]
    if max_leads is not None:
        lead_idx = lead_idx[-max_leads:]
    if not lead_idx:
        return {"stat": np.nan, "df": 0, "pvalue": np.nan}
    b = coefs[lead_idx]
    Vb = V[np.ix_(lead_idx, lead_idx)]
    stat = float(b @ np.linalg.pinv(Vb) @ b)
    q = len(lead_idx)
    return {"stat": stat, "df": q, "pvalue": float(stats.chi2.sf(stat, q))}
