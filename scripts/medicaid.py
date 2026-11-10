"""Medicaid expansion study: diagnosed conditions vs diagnosis-free outcomes.

Real-data application of the DiD-with-measurement pipeline. Panel: 51 states
(incl. DC) x 2011-2024 x 10 BRFSS/NCHS categories. Treatment: ACA Medicaid
expansion (KFF adoption table; cohort = first calendar year with >= 6 months
of coverage). Outcome: crude prevalence (pp). Two bridges are run and judged by
an untreated-comovement diagnostic: NCHS diabetes mortality (external, d=0,
2011-2017) and BRFSS obesity (internal, d=0.05). Stages checkpoint to CSV so
partial runs resume. Run: PYTHONPATH=src python scripts/medicaid.py"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medicaid_lib as ml
from crime_lib import verify_fast_cs, cs_es_fast
from did_measurement.estimators import event_study_twfe, att_gt_cs
from did_measurement.mtest import discretion_gradient_test
from did_measurement.bounds import (sensitivity_interval, breakdown_kappa,
                                    recording_bounds)

# Paths, bootstrap sizes, event windows, category sets and the reporting layer
# live in medicaid_report (same study, split only for the file-size limit).
from medicaid_report import (  # noqa: E402
    ROOT, RAW, RES, PANEL, B_MAIN, B_ROB, B_YLOG, POST_W, BRIDGE_W,
    K_PRE, K_POST, KT_PRE, KT_POST, SEED, R0_DIABETES, CATS_ALL, BRIDGE_SET,
    FOCAL, BRFSS, OBES, DIAGNOSED, AUX, NAMES, D_IDX, ROLES, TXT, CI_TOPICS,
    noise_decomposition, make_figures, summarize)


# ---------------------------------------------------------------- stage 0

def stage0_build():
    df, missing = ml.build_panel(RAW, PANEL)
    pd.DataFrame(
        [(a, ml.EXPANSION_EFFECTIVE.get(a, ""), ml.cohort_of(a), int(a in ml.EARLY_EXPANDERS),
          int(a in ml.NEVER_STATES)) for a in ml.ABBRS],
        columns=["abbr", "expansion_effective", "cohort", "early_expander", "never"],
    ).to_csv(os.path.join(RES, "expansion_cohorts.csv"), index=False)
    pd.DataFrame(
        [(c, NAMES[c], D_IDX[c], ROLES[c], ml.D_RATIONALE[NAMES[c]]) for c in CATS_ALL],
        columns=["cat", "cat_name", "d", "role", "rationale"],
    ).to_csv(os.path.join(RES, "discretion_index.csv"), index=False)
    write_manifest(missing)
    return df


def _integrity_checks():
    rows = json.load(open(os.path.join(RAW, "nchs_diabetes_deaths.json")))
    us = {r["year"]: float(r["deaths"]) for r in rows if r["state"] == "United States"}
    assert us["2017"] == 83564 and us["2016"] == 80058
    dia = json.load(open(os.path.join(RAW, "brfss_diabetes.json")))
    v = [float(r["data_value"]) for r in dia
         if r.get("locationabbr") == "US" and r["year"] == "2019" and r["response"] == "Yes"]
    assert v == [10.8]
    return {"nchs_us_diabetes_deaths_2017": 83564, "nchs_us_diabetes_deaths_2016": 80058,
            "brfss_us_2019_diabetes_crude_pct": 10.8,
            "note": ("US 2016/2017 diabetes death counts and US 2019 crude diagnosed-diabetes prevalence "
                     "match CDC-published figures exactly.")}


def write_manifest(missing):
    man_path = os.path.join(ROOT, "data", "data_manifest.json")
    man = {}
    if os.path.exists(man_path):
        try:
            man = json.load(open(man_path))
        except json.JSONDecodeError:
            man = {}
    man["medicaid"] = {
        **ml.MANIFEST_TEXT,
        "integrity_checks": _integrity_checks(),
        "expansion_effective": ml.EXPANSION_EFFECTIVE,
        "never_states": ml.NEVER_STATES,
        "early_partial_expanders": ml.EARLY_EXPANDERS,
        "cohort_rule": ml.COHORT_RULE,
        "cohorts": {a: ml.cohort_of(a) for a in ml.ABBRS},
        "discretion_d": {NAMES[c]: (None if not np.isfinite(D_IDX[c])
                                    else D_IDX[c]) for c in CATS_ALL},
        "discretion_rationale": ml.D_RATIONALE,
        "missing_cells": sorted(set(missing)),
        "r0_diabetes": {"value": R0_DIABETES, "source": ml.R0_SOURCE},
    }
    json.dump(man, open(man_path, "w"), indent=1)


# ---------------------------------------------------------------- stage 1

def stage1_boot(df):
    """Two state-cluster bootstraps: the external mortality bridge (cat 0) and
    the internal obesity bridge (cat 5, d=0.05), plus the untreated comovement
    matrix that adjudicates which of the two is identified at all."""
    keys = ("es_cs", "postmean_cs", "bridge_rho", "bridge_decomposition",
            "bridge_shares", "gradient_slope_boot", "bridge2_rho",
            "bridge2_decomposition", "bridge2_shares", "bridge_comovement",
            "bridge_comovement_n", "postmean_odd_cs", "gradient_slope_odd_boot")
    paths = {k: os.path.join(RES, f"{k}.csv") for k in keys}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: pd.read_csv(p) for k, p in paths.items()}
    verify_fast_cs(df, cats=[0, 1, 2, 6], outcome="Y", control="notyet")
    r = ml.boot_unified(df, CATS_ALL, BRIDGE_SET, FOCAL, DIAGNOSED, D_IDX,
                        outcome="Y", B=B_MAIN, seed=SEED, bridge_cat=0,
                        post_window=POST_W, bridge_window=BRIDGE_W)
    r2 = ml.boot_unified(df[df["cat"].isin(BRFSS)].reset_index(drop=True),
                         BRFSS, BRFSS, [c for c in BRFSS if c != OBES],
                         DIAGNOSED, D_IDX, outcome="Y", B=B_MAIN,
                         seed=SEED + 40, bridge_cat=OBES,
                         post_window=POST_W, bridge_window=POST_W)
    corr, npair = ml.residual_corr(df, BRIDGE_SET)
    out = {"es_cs": r["es"], "postmean_cs": r["pm"], "bridge_rho": r["rho"],
           "bridge_decomposition": r["dec"], "bridge_shares": r["share"],
           "gradient_slope_boot": pd.DataFrame([r["slope"]]),
           "bridge2_rho": r2["rho"], "bridge2_decomposition": r2["dec"],
           "bridge2_shares": r2["share"],
           "bridge_comovement": corr.reset_index(names="cat_name"),
           "bridge_comovement_n": npair.reset_index(names="cat_name"),
           "postmean_odd_cs": r["pm_odd"],
           "gradient_slope_odd_boot": pd.DataFrame([r["slope_odd"]])}
    for k in keys[:5] + keys[6:9] + ("postmean_odd_cs",):
        out[k].insert(1, "cat_name", out[k]["cat"].map(NAMES))
    for k, p in paths.items():
        out[k].to_csv(p, index=False)
    return out


# ---------------------------------------------------------------- stage 2

def _wald_rank(es_result):
    """Joint Wald on leads (ell < -1) with rank-based df (guards degenerate
    lead dummies in odd-year categories); equals mtest.pretrend_test when
    the lead covariance block has full rank."""
    labels, V = es_result["labels"], es_result["vcov"]
    b = es_result["table"]["coef"].to_numpy()
    idx = [j for j, l in enumerate(labels) if l < -1]
    Vb = V[np.ix_(idx, idx)]
    bb = b[idx]
    stat = float(bb @ np.linalg.pinv(Vb) @ bb)
    q = int(np.linalg.matrix_rank(Vb, tol=1e-12 * max(np.trace(Vb), 1e-30)))
    return stat, q, float(stats.chi2.sf(stat, max(q, 1)))


def stage2_twfe(df):
    tw_csv = os.path.join(RES, "es_twfe.csv")
    pt_csv = os.path.join(RES, "pretrend_tests.csv")
    if os.path.exists(tw_csv) and os.path.exists(pt_csv):
        return pd.read_csv(tw_csv), pd.read_csv(pt_csv)
    tw_rows, pt_rows = [], []
    for c in CATS_ALL:
        sub = df[df["cat"] == c]
        r = event_study_twfe(sub, outcome="Y", k_pre=KT_PRE, k_post=KT_POST)
        t = r["table"].copy()
        t.insert(0, "cat_name", NAMES[c])
        t.insert(0, "cat", c)
        obs = set(np.unique(sub["event_time"].dropna().clip(-KT_PRE, KT_POST)))
        tw_rows.append(t[t["ell"].isin(obs)])
        # full window bins leads at -6, so the joint test is driven by event
        # times that exist only for the 1-3 states adopting in 2022-2024;
        # the short window tests the pre-period identification rests on
        for tag, kp in (("full_k6", KT_PRE), ("short_k4", 4)):
            rr = r if kp == KT_PRE else event_study_twfe(
                sub, outcome="Y", k_pre=kp, k_post=KT_POST)
            stat, q, p = _wald_rank(rr)
            pt_rows.append((c, NAMES[c], tag, stat, q, p))
    tw_tab = pd.concat(tw_rows, ignore_index=True)
    pt_tab = pd.DataFrame(pt_rows, columns=["cat", "cat_name", "window",
                                            "stat", "df_rank", "pvalue"])
    tw_tab.to_csv(tw_csv, index=False)
    pt_tab.to_csv(pt_csv, index=False)
    return tw_tab, pt_tab


# ---------------------------------------------------------------- stage 3

def _grad(dat, outcome, adj):
    return discretion_gradient_test(dat, outcome=outcome, k_pre=K_PRE,
                                    k_post=K_POST, bridge_cat=0,
                                    use_bridge=False, loading_adjust=adj)


def stage3_gradient(df):
    out_csv = os.path.join(RES, "gradient_test.csv")
    sl_csv = os.path.join(RES, "gradient_slopes.csv")
    if os.path.exists(out_csv) and os.path.exists(sl_csv):
        return pd.read_csv(out_csv), pd.read_csv(sl_csv)
    foc = df[df["cat"].isin(FOCAL)].reset_index(drop=True)
    rows, slopes = [], []
    for outcome in ("Y", "ylog"):
        for adj in (True, False):
            r = _grad(foc, outcome, adj)
            tag = "adjusted" if adj else "naive"
            rows.append((outcome, tag, r["stat"], r["df"], r["pvalue"], r["pre_slopes_pvalue"],
                         r["n_clusters"],
                         json.dumps({NAMES[int(k)]: round(v, 4) for k, v in (r["rho"] or {}).items()})))
            if outcome == "Y":
                s = r["slopes"].copy()
                s.insert(0, "version", tag)
                slopes.append(s)
    tab = pd.DataFrame(rows, columns=["outcome", "version", "stat", "df", "pvalue",
                                      "pre_slopes_pvalue", "n_clusters", "rho"])
    sl = pd.concat(slopes, ignore_index=True)
    tab.to_csv(out_csv, index=False)
    sl.to_csv(sl_csv, index=False)
    return tab, sl


# ---------------------------------------------------------------- stage 4

def stage4_bounds(df, pm_tab):
    b_csv = os.path.join(RES, "bounds.csv")
    s_csv = os.path.join(RES, "sensitivity_diabetes.csv")
    if os.path.exists(b_csv) and os.path.exists(s_csv):
        return pd.read_csv(b_csv), pd.read_csv(s_csv)
    pm = pm_tab.set_index("cat")
    es = pd.read_csv(os.path.join(RES, "es_cs.csv")).set_index(["cat", "ell"])
    # diabetes log outcome for the multiplicative recording bounds
    sub = df[df["cat"] == 1].reset_index(drop=True)
    pm_log, pm_log_se, _ = ml.boot_postmean(sub, outcome="ylog", B=B_YLOG,
                                            seed=SEED + 3, post_window=POST_W)
    rb = recording_bounds(pm_log, pm_log_se, R0_DIABETES)

    targets = [(c, "postmean_0_5", float(pm.loc[c, "post_mean"]), float(pm.loc[c, "se"]))
               for c in DIAGNOSED + [5]]
    targets.insert(1, (1, "ell_0", float(es.loc[(1, 0), "coef"]), float(es.loc[(1, 0), "se"])))
    rows = []
    for c, tgt, coef, se in targets:
        bk = breakdown_kappa(coef, se, D_IDX[c])
        rows.append((c, NAMES[c], tgt, D_IDX[c], coef, se, bk["point"], bk["stat"]))
    bounds = pd.DataFrame(rows, columns=["cat", "cat_name", "target", "d", "coef", "se",
                                         "breakdown_kappa_point", "breakdown_kappa_stat"])
    for col, val in [("rb_log_L", rb["L"]), ("rb_log_U", rb["U"]), ("rb_log_ci_lo", rb["ci_lo"]),
                     ("rb_log_ci_hi", rb["ci_hi"]), ("rb_r0", R0_DIABETES),
                     ("pm_log", pm_log), ("pm_log_se", pm_log_se)]:
        bounds[col] = np.nan
        bounds.loc[(bounds["cat"] == 1) & (bounds["target"] == "postmean_0_5"), col] = val
    coef, se = float(pm.loc[1, "post_mean"]), float(pm.loc[1, "se"])
    grid = np.round(np.arange(0.0, 0.601, 0.025), 3)
    sens = pd.DataFrame([{**{"kappa_bar": k}, **sensitivity_interval(coef, se, D_IDX[1], k)}
                         for k in grid])[["kappa_bar", "L", "U", "ci_lo", "ci_hi", "im_c"]]
    bounds.to_csv(b_csv, index=False)
    sens.to_csv(s_csv, index=False)
    return bounds, sens


# ---------------------------------------------------------------- stage 5

def _boot_pm_never(df, cats, B, seed):
    """Post-window CS means using ONLY never-expanded states as controls
    (the main pass uses not-yet-treated, which adds late expanders to the
    comparison). State-cluster bootstrap; draws without a never-treated state
    are dropped. Returns (point {cat: pm}, draws {cat: [pm, ...]})."""
    lo, hi = POST_W
    dat = df[df["cat"].isin(cats)].reset_index(drop=True)

    def _pm_all(d):
        out = {}
        for c in cats:
            try:
                out[c] = ml._postmean(cs_es_fast(d[d["cat"] == c], "Y",
                                                 "never"), lo, hi)
            except (ValueError, KeyError, np.linalg.LinAlgError):
                out[c] = np.nan
        return out

    point = _pm_all(dat)
    rng = np.random.default_rng(seed)
    units, draw = ml.make_resampler(dat)
    draws = {c: [] for c in cats}
    for _ in range(B):
        pm_b = _pm_all(draw(rng.choice(units, size=len(units), replace=True)))
        for c in cats:
            if np.isfinite(pm_b[c]):
                draws[c].append(pm_b[c])
    return point, draws


def stage5_robust(df):
    rb_csv = os.path.join(RES, "robustness.csv")
    rg_csv = os.path.join(RES, "robustness_gradient.csv")
    if os.path.exists(rb_csv) and os.path.exists(rg_csv):
        return pd.read_csv(rb_csv), pd.read_csv(rg_csv)
    early_units = set(df.loc[df["abbr"].isin(ml.EARLY_EXPANDERS), "unit"])
    pre = df[df["period"] <= 2019].reset_index(drop=True)
    noearly = df[~df["unit"].isin(early_units)].reset_index(drop=True)
    rows = []
    for i, (tag, dat) in enumerate([("pre2020", pre),
                                    ("excl_early_expanders", noearly)]):
        r = ml.boot_unified(dat, BRIDGE_SET, BRIDGE_SET, FOCAL, DIAGNOSED,
                            D_IDX, outcome="Y", B=B_ROB,
                            seed=SEED + 10 + i, bridge_cat=0,
                            post_window=POST_W, bridge_window=BRIDGE_W)
        for _, x in r["pm"].iterrows():
            rows.append((tag, int(x["cat"]), NAMES[int(x["cat"])],
                         x["post_mean"], x["se"], int(x["n_boot"])))
        sl = r["slope"]
        rows.append((tag, -1, "gradient_slope_pp_per_d", sl["point"],
                     sl["se"], sl["n_draws"]))
    point, draws = _boot_pm_never(df, BRIDGE_SET, B_ROB, SEED + 20)
    for c in BRIDGE_SET:
        se = (float(np.std(draws[c], ddof=1)) if len(draws[c]) > 25
              else np.nan)
        rows.append(("never_control", c, NAMES[c], point[c], se,
                     len(draws[c])))
    rb = pd.DataFrame(rows, columns=["variant", "cat", "cat_name",
                                     "post_mean", "se", "n_boot"])
    grows = []
    for tag, dat in (("pre2020", pre[pre["cat"].isin(FOCAL)]),
                     ("excl_early_expanders",
                      noearly[noearly["cat"].isin(FOCAL)]),
                     ("drop_no_exercise",
                      df[df["cat"].isin([1, 2, 3, 4, 5, 6])]),
                     ("diagnosed_vs_pure_selfreport",
                      df[df["cat"].isin([1, 4, 5, 6, 7])])):
        dat = dat.reset_index(drop=True)
        for adj in (True, False):
            g = _grad(dat, "Y", adj)
            grows.append((tag, "adjusted" if adj else "naive", g["stat"],
                          g["df"], g["pvalue"], g["pre_slopes_pvalue"]))
    rg = pd.DataFrame(grows, columns=["variant", "version", "stat", "df",
                                      "pvalue", "pre_slopes_pvalue"])
    rb.to_csv(rb_csv, index=False)
    rg.to_csv(rg_csv, index=False)
    return rb, rg


# ---------------------------------------------------------------- stage 6

def stage6_power(df, pm_tab, pmo_tab):
    """Is the null informative? Compare each condition's minimum detectable
    effect with the recording response the mechanism can arithmetically
    deliver: d_c x (insured gain) x (pool of unaware cases). See
    manifest summary_text.power_note for the assumptions."""
    p_csv = os.path.join(RES, "power_mde.csv")
    if os.path.exists(p_csv):
        return pd.read_csv(p_csv)
    base = df[(df["D"] == 0) & (df["cohort"] > 0)].groupby("cat")["Y"].mean()
    pm, pmo = pm_tab.set_index("cat"), pmo_tab.set_index("cat")
    dcov = float(pm.loc[8, "post_mean"]) / 100.0     # insured gain, pop share
    u = 1.0 - R0_DIABETES                            # unaware share (sourced)
    rows = []
    for c in FOCAL:
        se, d = float(pm.loc[c, "se"]), D_IDX[c]
        pool = float(base[c]) * u / (1.0 - u)        # unaware cases, pp
        rows.append((c, NAMES[c], d, float(base[c]), float(pm.loc[c, "post_mean"]),
                     se, 1.96 * se, 2.802 * se, d * dcov * pool,
                     3.0 * d * dcov * pool, float(pmo.loc[c, "post_mean"]),
                     float(pmo.loc[c, "se"])))
    t = pd.DataFrame(rows, columns=[
        "cat", "cat_name", "d", "baseline_prev_pp", "post_mean", "se",
        "mde95_pp", "mde80_pp", "implied_pp", "implied_pp_3x",
        "post_mean_odd_ell", "se_odd_ell"])
    t["power_ratio_mde80_over_implied"] = t["mde80_pp"] / t["implied_pp"]
    t.to_csv(p_csv, index=False)
    return t



# ---------------------------------------------------------------- stage 7

def _fetch_ci():
    """BRFSS design-based 95% confidence limits per state-year cell (columns
    the main extract did not select). Cached; refetched only if absent."""
    path = os.path.join(RAW, "brfss_ci.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = []
    for topic in CI_TOPICS.values():
        q = urllib.parse.urlencode({
            "$select": ("year,locationabbr,topic,response,data_value,"
                        "confidence_limit_low,confidence_limit_high"),
            "$where": (f"topic='{topic}' AND break_out='Overall' AND "
                       "data_value_type='Crude Prevalence'"),
            "$limit": "50000"})
        rows += json.load(urllib.request.urlopen(
            f"https://data.cdc.gov/resource/dttw-5yxu.json?{q}", timeout=180))
    json.dump(rows, open(path, "w"))
    return rows


def _attach_se_samp(df):
    """Merge the design-based sampling SE (pp) of each BRFSS cell onto the
    panel. Asserts the CI extract's data_value reproduces the panel's Y, so a
    silent response/label mismatch cannot pass as a valid merge."""
    rows = _fetch_ci()
    se, val = {}, {}
    for key, topic in CI_TOPICS.items():
        for r in ml._select_rows(key, [x for x in rows if x["topic"] == topic]):
            lo, hi = r.get("confidence_limit_low"), r.get("confidence_limit_high")
            if lo is None or hi is None:
                continue
            k = (key, r["locationabbr"], int(r["year"]))
            se[k] = (float(hi) - float(lo)) / (2 * 1.959964)
            val[k] = float(r["data_value"])
    kk = df["cat"].map({c: ml.CATS[c][3] for c in ml.CATS})
    keys = list(zip(kk, df["abbr"], df["period"]))
    out = df.copy()
    out["se_samp"] = [se.get(k, np.nan) for k in keys]
    chk = np.array([val.get(k, np.nan) for k in keys]) - out["Y"].to_numpy()
    assert np.nanmax(np.abs(chk)) < 1e-8, "CI extract does not match panel Y"
    assert out.loc[out["cat"] != 0, "se_samp"].notna().all()
    return out


def stage7_sampling(df):
    """How much of the state-year variation BRFSS reports is survey sampling
    noise, and would the design work without it? Splits each category's
    within-residual variance into design-based sampling variance and true
    state-year variation, recomputes the MDE at zero sampling noise, and
    checks whether inverse-variance weighting buys any precision."""
    s_csv = os.path.join(RES, "sampling_noise.csv")
    c_csv = os.path.join(RES, "comovement_noise_corrected.csv")
    if os.path.exists(s_csv) and os.path.exists(c_csv):
        return pd.read_csv(s_csv), pd.read_csv(c_csv)
    dfe = _attach_se_samp(df)
    pm = pd.read_csv(os.path.join(RES, "postmean_cs.csv")).set_index("cat")
    rows = {}
    for c in sorted(dfe.loc[dfe["se_samp"].notna(), "cat"].unique()):
        sub = dfe[dfe["cat"] == c].reset_index(drop=True)
        v_res, v_samp, twfe = noise_decomposition(sub, POST_W, KT_PRE, KT_POST)
        share = min(v_samp / v_res, 1.0)
        cse = float(pm.loc[c, "se"])
        rows[c] = dict(cat=c, cat_name=NAMES[c], d=D_IDX[c],
                       sd_samp_pp=np.sqrt(v_samp), sd_resid_pp=np.sqrt(v_res),
                       noise_share_of_var=share,
                       true_state_sd_pp=np.sqrt(max(v_res - v_samp, 0.0)),
                       cs_se=cse, cs_se_noisefree=cse * np.sqrt(1 - share),
                       mde80_noisefree_pp=2.802 * cse * np.sqrt(1 - share),
                       **twfe)
    tab = pd.DataFrame(rows.values())
    imp = pd.read_csv(os.path.join(RES, "power_mde.csv")).set_index("cat")
    tab["implied_pp"] = tab["cat"].map(imp["implied_pp"])
    tab["mde80_noisefree_over_implied"] = (tab["mde80_noisefree_pp"]
                                           / tab["implied_pp"])
    # observed untreated comovement is attenuated by independent survey noise
    # in both series; divide it out to see what the loadings could ever be
    corr = pd.read_csv(os.path.join(RES, "bridge_comovement.csv")
                       ).set_index("cat_name")
    sh = {NAMES[c]: rows[c]["noise_share_of_var"] for c in rows}
    # NCHS death counts are a near-census, so the mortality bridge carries no
    # survey sampling noise of its own; entering it at share 0 tests whether
    # its zero comovement is itself only attenuation from the BRFSS side
    sh[NAMES[0]] = 0.0
    crows = []
    for a in corr.index:
        for b in corr.columns:
            if a >= b or a not in sh or b not in sh:
                continue
            att = np.sqrt((1 - sh[a]) * (1 - sh[b]))
            crows.append((a, b, float(corr.loc[a, b]), att,
                          float(corr.loc[a, b]) / att if att > 0 else np.nan))
    cc = pd.DataFrame(crows, columns=["cat_a", "cat_b", "corr_observed",
                                      "attenuation_factor", "corr_noise_corrected"])
    tab.to_csv(s_csv, index=False)
    cc.to_csv(c_csv, index=False)
    return tab, cc


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.dirname(PANEL), exist_ok=True)

    def tick(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    df = stage0_build()   # deterministic rebuild from cached raw files
    tick(f"panel {df.shape}, {df['unit'].nunique()} states")
    tabs = stage1_boot(df); tick("stage1 unified bootstrap")
    tw_tab, pt_tab = stage2_twfe(df); tick("stage2 TWFE + pretrends")
    grad_tab, _ = stage3_gradient(df); tick("stage3 gradient")
    bounds, sens = stage4_bounds(df, tabs["postmean_cs"]); tick("stage4 bounds")
    rb_tab, rg_tab = stage5_robust(df); tick("stage5 robustness")
    power = stage6_power(df, tabs["postmean_cs"], tabs["postmean_odd_cs"])
    tick("stage6 power")
    noise, ncorr = stage7_sampling(df); tick("stage7 sampling noise")
    make_figures(tabs, tw_tab, sens, power, noise)
    summary = summarize(df, tabs, pt_tab, grad_tab, rb_tab, rg_tab, bounds,
                        power, noise, ncorr, t0)
    tick("summary written")
    print(json.dumps({k: summary[k] for k in
                      ("headline", "power_and_mde", "common_support_odd_ell")},
                     indent=1))


if __name__ == "__main__":
    main()
