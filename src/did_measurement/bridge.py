"""Bridge-outcome identification (project Theorem 3, two-step implementation).

A bridge outcome (category 0 by convention) is recorded by a technology that
does not respond to treatment (d_0 = 0) and loads on the same latent factor as
the focal categories. Under loading stability, the latent event-study effect of
category c is rho_c * ES_bridge(l), with rho_c = theta_c / theta_bridge, and the
measurement response of category c is ES_c(l) - rho_c * ES_bridge(l).

Loadings are estimated from untreated observations (pre-period and never-treated)
after removing unit-by-category and period-by-category fixed effects. The direct
regression of category residuals on bridge residuals is attenuated by the
bridge's idiosyncratic noise, so the default estimator instruments with a third
category:  rho_c = cov(res_c, res_k) / cov(res_0, res_k), averaged over k != c, 0.
This is consistent under the one-factor structure; the minimum |denominator| is
reported as a weak-bridge diagnostic.
"""

import numpy as np
import pandas as pd

from .estimators import demean_two_way, event_study_twfe, att_gt_cs


def _residual_wide(df, outcome):
    """Residualize outcome on unit-by-cat and period-by-cat FE using untreated rows;
    return wide array res[(unit, period), cat]."""
    dat = df[df["D"] == 0].reset_index(drop=True)
    n_cat_span = int(dat["cat"].max()) + 1
    uc = dat["unit"].to_numpy() * n_cat_span + dat["cat"].to_numpy()
    pc = dat["period"].to_numpy() * n_cat_span + dat["cat"].to_numpy()
    R = demean_two_way(dat[[outcome]].to_numpy(float), uc, pc)
    out = dat[["unit", "period", "cat"]].copy()
    out["res"] = R[:, 0]
    return out.pivot_table(index=["unit", "period"], columns="cat", values="res")


def estimate_relative_loadings(df, outcome="Y", bridge_cat=0, method="iv"):
    """Estimate rho_c = theta_c / theta_bridge from untreated comovement.

    method='iv' (default): cross-category instrumenting, consistent under the
    one-factor model. method='ols': direct projection on the bridge (attenuated;
    for diagnostics). Returns dict: 'rho' {cat: est}, 'min_denom', 'method'.
    """
    wide = _residual_wide(df, outcome)
    cats = [c for c in wide.columns if c != bridge_cat]
    cov = wide.cov()
    rho, denoms = {}, []
    for c in cats:
        if method == "ols" or len(cats) < 2:
            denom = cov.loc[bridge_cat, bridge_cat]
            rho[c] = float(cov.loc[c, bridge_cat] / denom)
            denoms.append(abs(denom))
        else:
            ratios, ws = [], []
            for k in cats:
                if k == c:
                    continue
                denom = cov.loc[bridge_cat, k]
                denoms.append(abs(denom))
                if abs(denom) > 1e-12:
                    ratios.append(cov.loc[c, k] / denom)
                    ws.append(abs(denom))
            rho[c] = float(np.average(ratios, weights=ws)) if ratios else np.nan
    return {"rho": rho, "min_denom": float(min(denoms)) if denoms else np.nan,
            "method": method if len(cats) >= 2 else "ols"}


def cohort_event_studies(df, outcome="Y", control="never"):
    """Per-cohort, per-category Callaway-Sant'Anna event studies plus the weights
    the aggregated event study uses.

    Fix for defect X2: the sharp identified set for a cohort-aggregated
    coefficient is the Minkowski average of the per-cohort sets
    (bounds.cohort_sharp_interval), which needs beta^g_c(l) cohort by cohort.
    The shipped aggregated event study only returns sum_g w_g(l) beta^g_c(l).

    Returns dict:
      'beta'   {(cat, cohort, ell): coef}
      'weights'{(ell, cohort): w}  cohort-size weights, summing to 1 over cohorts
      'cohorts' sorted list of treated cohorts.
    """
    beta, wts, cohorts = {}, {}, set()
    sizes = df.groupby("cohort")["unit"].nunique()
    for c, sub in df.groupby("cat"):
        att = att_gt_cs(sub, outcome=outcome, control=control)["att_gt"]
        for (g, t), v in att.items():
            beta[(int(c), int(g), int(t - g))] = float(v)
            cohorts.add(int(g))
    for (_, g, l) in beta:
        wts.setdefault(l, {})[g] = float(sizes.loc[g])
    weights = {}
    for l, dd in wts.items():
        tot = sum(dd.values())
        for g, v in dd.items():
            weights[(l, g)] = v / tot
    return {"beta": beta, "weights": weights, "cohorts": sorted(cohorts)}


def cohort_beta_matrix(ces, ell, cats):
    """(G, C) matrix of beta^g_c(ell) and the (G,) weight vector, for the cohorts
    at which every requested category is observed at horizon ell."""
    gs = [g for g in ces["cohorts"]
          if all((c, g, ell) in ces["beta"] for c in cats)]
    if not gs:
        return None, None, []
    B = np.array([[ces["beta"][(c, g, ell)] for c in cats] for g in gs], float)
    w = np.array([ces["weights"][(ell, g)] for g in gs], float)
    return B, w / w.sum(), gs


def _es_by_cat(df, outcome, estimator, k_pre, k_post, control):
    out = {}
    for c, sub in df.groupby("cat"):
        if estimator == "twfe":
            r = event_study_twfe(sub, outcome=outcome, k_pre=k_pre, k_post=k_post)
            out[c] = r["table"].set_index("ell")["coef"]
        else:
            r = att_gt_cs(sub, outcome=outcome, control=control)
            out[c] = r["es"].set_index("ell")["coef"]
    return out


def bridge_decomposition(df, outcome="Y", bridge_cat=0, estimator="cs",
                         k_pre=5, k_post=6, control="never",
                         n_boot=0, seed=0, loading_method="iv"):
    """Decompose recorded event-study effects into latent and measurement parts.

    Returns dict:
      'table': DataFrame(cat, ell, es_recorded, es_latent, es_measurement[, ses])
      'rho', 'min_denom'.
    Bootstrap (unit-level, n_boot > 0) re-estimates loadings and event studies
    per draw, so the SEs reflect both steps.
    """
    def _once(d):
        ld = estimate_relative_loadings(d, outcome, bridge_cat, loading_method)
        es = _es_by_cat(d, outcome, estimator, k_pre, k_post, control)
        rows = []
        for c, s in es.items():
            if c == bridge_cat:
                continue
            for l, v in s.items():
                lat = ld["rho"].get(c, np.nan) * es[bridge_cat].get(l, np.nan)
                rows.append((c, l, v, lat, v - lat))
        t = pd.DataFrame(rows, columns=["cat", "ell", "es_recorded",
                                        "es_latent", "es_measurement"])
        return t, ld

    table, ld = _once(df)
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        units = df["unit"].unique()
        gb = dict(tuple(df.groupby("unit", sort=False)))
        stack = []
        for b in range(n_boot):
            samp = rng.choice(units, size=len(units), replace=True)
            parts = []
            for j, uu in enumerate(samp):
                p = gb[uu].copy()
                p["unit"] = j
                parts.append(p)
            try:
                tb, _ = _once(pd.concat(parts, ignore_index=True))
                stack.append(tb.set_index(["cat", "ell"]))
            except (ValueError, KeyError, np.linalg.LinAlgError):
                continue
        if stack:
            allb = pd.concat(stack, keys=range(len(stack)))
            sds = allb.groupby(level=["cat", "ell"]).std(ddof=1)
            table = table.set_index(["cat", "ell"])
            for col in ["es_recorded", "es_latent", "es_measurement"]:
                table["se_" + col] = sds[col]
            table = table.reset_index()
    return {"table": table, "rho": ld["rho"], "min_denom": ld["min_denom"]}


def bridge_decomposition_by_cohort(df, outcome="Y", bridge_cat=0, control="never",
                                   loading_method="iv"):
    """Cohort-by-cohort bridge decomposition (fix for defect X2, point-identified case).

    m_{c,g}(l) = beta^g_c(l) - rho_c beta^g_b(l), then aggregated with the CS
    weights.  Because rho is common across cohorts (it is estimated from
    untreated cells) the map beta -> m is LINEAR, so cohort-then-average and
    average-then-decompose coincide exactly; this function returns both, and
    'max_abs_gap' is the numerical check.  Non-sharpness of naive aggregation
    bites only where the measurement class M enters as an INEQUALITY (bounds),
    not here.  The per-cohort table is what bounds.cohort_sharp_interval needs.

    Returns dict: 'table' (cat, cohort, ell, es_recorded, es_latent,
    es_measurement, weight), 'aggregated' (cat, ell, es_* by both routes),
    'rho', 'min_denom', 'max_abs_gap'.
    """
    ld = estimate_relative_loadings(df, outcome, bridge_cat, loading_method)
    ces = cohort_event_studies(df, outcome=outcome, control=control)
    rows = []
    for (c, g, l), v in ces["beta"].items():
        if c == bridge_cat:
            continue
        vb = ces["beta"].get((bridge_cat, g, l), np.nan)
        r = ld["rho"].get(c, np.nan)
        rows.append((c, g, l, v, r * vb, v - r * vb, ces["weights"][(l, g)]))
    t = pd.DataFrame(rows, columns=["cat", "cohort", "ell", "es_recorded",
                                    "es_latent", "es_measurement", "weight"])
    agg = (t.assign(**{k: t[k] * t["weight"] for k in
                       ["es_recorded", "es_latent", "es_measurement"]})
             .groupby(["cat", "ell"])[["es_recorded", "es_latent",
                                       "es_measurement", "weight"]].sum())
    for k in ["es_recorded", "es_latent", "es_measurement"]:
        agg[k] = agg[k] / agg["weight"]
    agg = agg.drop(columns="weight").reset_index()
    # average-then-decompose route, for the equality check
    direct = agg.copy()
    direct["es_latent"] = [
        ld["rho"].get(int(r.cat), np.nan) *
        sum(ces["weights"][(int(r.ell), g)] * ces["beta"][(bridge_cat, g, int(r.ell))]
            for g in ces["cohorts"] if (bridge_cat, g, int(r.ell)) in ces["beta"])
        for r in agg.itertuples()]
    direct["es_measurement"] = direct["es_recorded"] - direct["es_latent"]
    gap = float(np.nanmax(np.abs(
        agg[["es_latent", "es_measurement"]].to_numpy()
        - direct[["es_latent", "es_measurement"]].to_numpy())))
    return {"table": t, "aggregated": agg, "aggregated_direct": direct,
            "rho": ld["rho"], "min_denom": ld["min_denom"], "max_abs_gap": gap}
