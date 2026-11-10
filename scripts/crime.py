"""Crime recording study: staggered state transition from UCR-SRS to NIBRS.

Panel: 51 states x 1979-2024 x 7 index offenses (homicide = low-discretion
bridge). Treatment: first year >= 50% of a state's FBI CDE agency roster
reports NIBRS. Outcome: log((count + 0.5) / population * 100000). Stages
checkpoint to CSV. Run: PYTHONPATH=src python scripts/crime.py"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crime_lib import (CATS, NCVS_REPORTED, CORE_TYPES, CAVEATS, TEXT,
                       share_table, discretion_index, build_panel, verify_fast_cs,
                       bootstrap_all, load_adoption, apply_cohorts,
                       run_adoption_variants, mc_se_of_sd, fig_event_study,
                       fig_gradient, fig_adoption, fig_bridge)
from did_measurement.estimators import event_study_twfe
from did_measurement.mtest import discretion_gradient_test, pretrend_test
from did_measurement.bounds import recording_bounds, breakdown_kappa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "crime")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results", "empirics", "crime")
PANEL = os.path.join(PROC, "crime_panel.csv")

B_BOOT = 300        # state bootstrap draws (shared by ES, post-means, bridge)
B_ROB = 200         # bootstrap draws for the pre-2020 robustness subsample
POST_W = (0, 5)     # post window (event years) for headline aggregates
K_PRE, K_POST = 6, 8
SEED = 71
NAMES = {c: nm for c, (_, nm) in CATS.items()}
FOCAL = [1, 2, 3, 4, 5, 6]

# Measured fresh-compute cost per stage, so runtime_min from a checkpoint-
# resumed execution is not mistaken for the cost of the whole study.
RUNTIME_NOTE = TEXT["runtime_note"]


def stage0_build():
    df, adopt, shares = build_panel(os.path.join(RAW, "state-trends.json"),
                                    os.path.join(RAW, "agencies"), PANEL)
    adopt.to_csv(os.path.join(RES, "nibrs_adoption.csv"), index=False)
    d_idx, _ = discretion_index()
    pd.DataFrame({
        "cat": list(CATS), "cat_name": [NAMES[c] for c in CATS],
        "ncvs_pct_reported_2018": [np.nan] + [NCVS_REPORTED[c][0] for c in FOCAL],
        "ncvs_pct_reported_2019": [np.nan] + [NCVS_REPORTED[c][1] for c in FOCAL],
        "d": [d_idx[c] for c in CATS],
    }).to_csv(os.path.join(RES, "discretion_index.csv"), index=False)
    fig_adoption(shares, os.path.join(RES, "fig_adoption.png"))
    write_manifest(adopt)
    return df


def write_manifest(adopt):
    coh = {r["abbr"]: int(r["cohort"]) for _, r in adopt.iterrows()}
    d_idx, _ = discretion_index()
    man_path = os.path.join(ROOT, "data", "data_manifest.json")
    man = {}
    if os.path.exists(man_path):
        try:
            man = json.load(open(man_path))
        except json.JSONDecodeError:
            man = {}
    man["crime"] = {
        "design_chosen": TEXT["design_chosen"],
        "rejected_alternatives": TEXT["rejected_alternatives"],
        "sources": TEXT["sources"],
        "ncvs_pct_reported_2018_2019": {NAMES[c]: list(NCVS_REPORTED[c])
                                        for c in FOCAL},
        "discretion_d": {NAMES[c]: round(d_idx[c], 4) for c in CATS},
        "cohorts": coh,
        "repairs": TEXT["repairs"],
        "caveats": CAVEATS,
    }
    json.dump(man, open(man_path, "w"), indent=1)


def stage1_boot(df):
    """CS event studies, post-means, bridge decomposition: one state bootstrap."""
    paths = {k: os.path.join(RES, f"{k}.csv") for k in
             ("es_cs", "postmean_cs", "bridge_decomposition", "bridge_rho")}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: pd.read_csv(p) for k, p in paths.items()}, None
    verify_fast_cs(df, cats=[0, 6])   # fast CS == library CS on real data
    r = bootstrap_all(df, list(CATS), outcome="y", control="notyet", B=B_BOOT,
                      seed=SEED, bridge_cat=0, post_window=POST_W)
    out = {"es_cs": r["es"], "postmean_cs": r["postmean"],
           "bridge_decomposition": r["dec"], "bridge_rho": r["rho"]}
    out["bridge_decomposition"].insert(
        1, "cat_name", out["bridge_decomposition"]["cat"].map(NAMES))
    out["bridge_rho"].insert(1, "cat_name", out["bridge_rho"]["cat"].map(NAMES))
    for k, p in paths.items():
        out[k].to_csv(p, index=False)
    return out, r


def stage2_twfe(df):
    tw_csv = os.path.join(RES, "es_twfe.csv")
    pt_csv = os.path.join(RES, "pretrend_tests.csv")
    if os.path.exists(tw_csv) and os.path.exists(pt_csv):
        return pd.read_csv(tw_csv), pd.read_csv(pt_csv)
    tw_rows, pt_rows = [], []
    for c in CATS:
        sub = df[df["cat"] == c]
        r = event_study_twfe(sub, outcome="y", k_pre=K_PRE, k_post=K_POST)
        t = r["table"].copy()
        t.insert(0, "cat", c)
        tw_rows.append(t)
        pt = pretrend_test(r)
        pt_rows.append((c, NAMES[c], pt["stat"], pt["df"], pt["pvalue"]))
    tw_tab = pd.concat(tw_rows, ignore_index=True)
    pt_tab = pd.DataFrame(pt_rows, columns=["cat", "cat_name", "stat", "df",
                                            "pvalue"])
    tw_tab.to_csv(tw_csv, index=False)
    pt_tab.to_csv(pt_csv, index=False)
    return tw_tab, pt_tab


def stage3_gradient(df):
    out_csv = os.path.join(RES, "gradient_test.csv")
    sl_csv = os.path.join(RES, "gradient_slopes.csv")
    if os.path.exists(out_csv) and os.path.exists(sl_csv):
        return pd.read_csv(out_csv), pd.read_csv(sl_csv)
    rows, slopes = [], []
    for adj in (True, False):
        r = discretion_gradient_test(df, outcome="y", k_pre=K_PRE, k_post=K_POST,
                                     bridge_cat=0, use_bridge=False,
                                     loading_adjust=adj)
        rows.append(("adjusted" if adj else "naive", r["stat"], r["df"],
                     r["pvalue"], r["pre_slopes_pvalue"], r["n_clusters"],
                     json.dumps(r["rho"]) if r["rho"] else ""))
        s = r["slopes"].copy()
        s.insert(0, "version", "adjusted" if adj else "naive")
        slopes.append(s)
    tab = pd.DataFrame(rows, columns=["version", "stat", "df", "pvalue",
                                      "pre_slopes_pvalue", "n_clusters", "rho"])
    sl = pd.concat(slopes, ignore_index=True)
    tab.to_csv(out_csv, index=False)
    sl.to_csv(sl_csv, index=False)
    return tab, sl


def stage4_bounds(pm_tab):
    out_csv = os.path.join(RES, "bounds.csv")
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    d_idx, rates = discretion_index()
    rows = []
    for _, r in pm_tab.iterrows():
        c = int(r["cat"])
        if c == 0:
            continue
        coef, se = float(r["post_mean"]), float(r["se"])
        rb = recording_bounds(coef, se, rates[c])
        bk = breakdown_kappa(coef, se, d_idx[c])
        rows.append((c, NAMES[c], d_idx[c], coef, se, rates[c], rb["L"], rb["U"],
                     rb["ci_lo"], rb["ci_hi"], bk["point"], bk["stat"]))
    tab = pd.DataFrame(rows, columns=["cat", "cat_name", "d", "post_mean", "se",
                                      "r0_ncvs", "bound_L", "bound_U",
                                      "im_ci_lo", "im_ci_hi",
                                      "breakdown_kappa_point",
                                      "breakdown_kappa_stat"])
    tab.to_csv(out_csv, index=False)
    return tab


def stage5_robust(df):
    """(a) Pre-2020 subsample (drops COVID homicide spike + NIBRS-only
    estimation era; 2020+ cohorts become not-yet controls): post-means and
    bridge decomposition with their own bootstrap. (b) Gradient-test variants
    excluding the categories with failing pretrends (rape, larceny)."""
    rb_csv = os.path.join(RES, "robustness.csv")
    rg_csv = os.path.join(RES, "robustness_gradient.csv")
    if os.path.exists(rb_csv) and os.path.exists(rg_csv):
        return pd.read_csv(rb_csv), pd.read_csv(rg_csv)
    pre = df[df["period"] <= 2019].reset_index(drop=True)
    r = bootstrap_all(pre, list(CATS), outcome="y", control="notyet",
                      B=B_ROB, seed=SEED + 1, bridge_cat=0, post_window=POST_W)
    dec, pm = r["dec"], r["postmean"]
    post = dec[(dec["ell"] >= POST_W[0]) & (dec["ell"] <= POST_W[1])]
    rows = []
    for _, x in pm.iterrows():
        c = int(x["cat"])
        g = post[post["cat"] == c]
        rec, meas = g["es_recorded"].sum(), g["es_measurement"].sum()
        rows.append(("pre2020", c, NAMES[c], x["post_mean"], x["se"],
                     g["es_latent"].mean() if len(g) else np.nan,
                     g["es_measurement"].mean() if len(g) else np.nan,
                     meas / rec if abs(rec) > 1e-9 else np.nan,
                     np.nan, np.nan))
    rec_s, meas_s = post["es_recorded"].sum(), post["es_measurement"].sum()
    sd = np.asarray(r["share_draws"])
    q = (np.percentile(sd, [2.5, 97.5]) if len(sd) > 20
         else [np.nan, np.nan])
    rows.append(("pre2020", -1, "pooled", np.nan, np.nan, np.nan, np.nan,
                 meas_s / rec_s if abs(rec_s) > 1e-9 else np.nan, q[0], q[1]))
    rb = pd.DataFrame(rows, columns=["variant", "cat", "cat_name", "post_mean",
                                     "se", "latent_mean", "measurement_mean",
                                     "measurement_share", "share_boot_p2_5",
                                     "share_boot_p97_5"])
    grows = []
    for tag, dat in (("pre2020", pre),
                     ("no_rape", df[df["cat"] != 5]),
                     ("no_rape_larceny", df[~df["cat"].isin([5, 6])])):
        for adj in (True, False):
            g = discretion_gradient_test(dat, outcome="y", k_pre=K_PRE,
                                         k_post=K_POST, bridge_cat=0,
                                         use_bridge=False, loading_adjust=adj)
            grows.append((tag, "adjusted" if adj else "naive", g["stat"],
                          g["df"], g["pvalue"], g["pre_slopes_pvalue"]))
    rg = pd.DataFrame(grows, columns=["variant", "version", "stat", "df",
                                      "pvalue", "pre_slopes_pvalue"])
    rb.to_csv(rb_csv, index=False)
    rg.to_csv(rg_csv, index=False)
    return rb, rg


ADOPT_VARIANTS = [
    ("baseline", {}),
    ("cohort_plus1", {"shift": 1}),
    ("threshold_0.8", {"threshold": 0.8}),
    ("core_agency_types", {"types": CORE_TYPES}),
    ("drop_never_treated", {"drop_never": True}),
    ("core_types_pre2020", {"types": CORE_TYPES, "max_year": 2019}),
]


def stage6_adoption(df):
    """Treatment-timing robustness. The baseline cohort rule counts every entry
    on the CDE roster, including the 17% with no NIBRS start date, which
    dilutes the share and can misclassify states (Hawaii: all 4 county/city
    police departments adopted by 2021, but 9 dateless state entries hold the
    share at 0.31, coding it never-treated)."""
    paths = {k: os.path.join(RES, f"adoption_variants{s}.csv") for k, s in
             (("pm", ""), ("gt", "_gradient"), ("coh", "_cohorts"))}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: pd.read_csv(p) for k, p in paths.items()}
    pm, gt, coh = run_adoption_variants(df, os.path.join(RAW, "agencies"),
                                        list(CATS), ADOPT_VARIANTS,
                                        post_window=POST_W, k_pre=K_PRE,
                                        k_post=K_POST)
    pm.insert(2, "cat_name", pm["cat"].map(NAMES))
    out = {"pm": pm, "gt": gt, "coh": coh}
    for k, p in paths.items():
        out[k].to_csv(p, index=False)
    return out


def stage7_mcse(df):
    """Monte Carlo error of the B=300 state-bootstrap SEs: how much of the
    reported precision is simulation noise rather than sampling uncertainty.
    Re-runs the stage-1 bootstrap at the same seed (reproduces it exactly) to
    recover the draws, then bootstraps the draws to get the MC SE of each SE."""
    out_csv = os.path.join(RES, "mc_se.csv")
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    r = bootstrap_all(df, list(CATS), outcome="y", control="notyet", B=B_BOOT,
                      seed=SEED, bridge_cat=0, post_window=POST_W)
    pm = r["postmean"].set_index("cat")
    rows = []
    for c in CATS:
        sd, mc, approx = mc_se_of_sd(r["pm_draws"][c])
        coef = float(pm.loc[c, "post_mean"])
        rows.append((c, NAMES[c], coef, sd, mc, approx, coef / sd,
                     abs(coef) * mc / sd ** 2, int(len(r["pm_draws"][c]))))
    tab = pd.DataFrame(rows, columns=["cat", "cat_name", "post_mean", "boot_se",
                                      "mc_se_of_se", "mc_se_analytic",
                                      "t_stat", "mc_se_of_t", "n_draws"])
    tab.to_csv(out_csv, index=False)
    return tab


def _bridge_spec(dat, tag, bridge_cat, seed):
    """Bootstrap one bridge/timing specification and checkpoint its tables:
    post-means, bridge decomposition, loadings, and post-window measurement
    shares. Used for the MVT bridge (stage 8) and the core-agency-types
    adoption timing (stage 9), which stage 6 covers by point estimate only."""
    paths = {k: os.path.join(RES, f"{tag}_{k}.csv")
             for k in ("postmean", "decomposition", "rho", "shares")}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: pd.read_csv(p) for k, p in paths.items()}
    r = bootstrap_all(dat, list(CATS), outcome="y", control="notyet", B=B_BOOT,
                      seed=seed, bridge_cat=bridge_cat, post_window=POST_W)
    out = {"postmean": r["postmean"], "decomposition": r["dec"], "rho": r["rho"]}
    for t in out.values():
        t.insert(1, "cat_name", t["cat"].map(NAMES))
    out["shares"] = share_table(out["decomposition"], r["share_draws"], NAMES,
                                POST_W, tag)
    for k, p in paths.items():
        out[k].to_csv(p, index=False)
    return out


def _bsum(tab, note):
    """summary.json block for one specification produced by _bridge_spec."""
    rho, sh, pm = tab["rho"], tab["shares"], tab["postmean"]
    pooled = sh[sh["cat"] == -1].iloc[0]
    foc = sh[sh["cat"] >= 0]
    return {
        "note": note,
        "post_mean": {r["cat_name"]: {"coef": round(float(r["post_mean"]), 4),
                                      "se": round(float(r["se"]), 4)}
                      for _, r in pm.iterrows()},
        "rho": {r["cat_name"]: round(float(r["rho"]), 4)
                for _, r in rho.iterrows()},
        "min_denom": round(float(rho["min_denom"].iloc[0]), 6),
        "measurement_mean": {r["cat_name"]: round(float(r["measurement_mean"]), 4)
                             for _, r in foc.iterrows()},
        "measurement_share_pooled": round(float(pooled["measurement_share"]), 4),
        "pooled_share_boot_ci95": [round(float(pooled["share_boot_p2_5"]), 4),
                                   round(float(pooled["share_boot_p97_5"]), 4)],
    }


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(PROC, exist_ok=True)
    adopt_csv = os.path.join(RES, "nibrs_adoption.csv")
    if os.path.exists(PANEL) and os.path.exists(adopt_csv):
        df = pd.read_csv(PANEL)
        adopt = pd.read_csv(adopt_csv)
        write_manifest(adopt)
    else:
        df = stage0_build()   # deterministic rebuild from cached raw files
        adopt = pd.read_csv(adopt_csv)
    print(f"[{time.time()-t0:6.1f}s] panel {df.shape}, "
          f"never-treated: {sorted(adopt[adopt.cohort < 0].abbr)}")

    tabs, _ = stage1_boot(df)
    es_tab, pm_tab = tabs["es_cs"], tabs["postmean_cs"]
    dec, rho = tabs["bridge_decomposition"], tabs["bridge_rho"]
    print(f"[{time.time()-t0:6.1f}s] stage1 bootstrap done")

    tw_tab, pt_tab = stage2_twfe(df)
    for c in CATS:
        fig_event_study(es_tab[es_tab["cat"] == c], tw_tab[tw_tab["cat"] == c],
                        NAMES[c], os.path.join(RES, f"fig_es_{NAMES[c]}.png"))
    d_idx, _ = discretion_index()
    fig_gradient(pm_tab, d_idx, NAMES, os.path.join(RES, "fig_gradient.png"))
    fig_bridge(dec, NAMES, os.path.join(RES, "fig_bridge.png"), POST_W)

    grad_tab, _ = stage3_gradient(df)
    print(f"[{time.time()-t0:6.1f}s] stage3 gradient done")
    bounds_tab = stage4_bounds(pm_tab)
    rb_tab, rg_tab = stage5_robust(df)
    print(f"[{time.time()-t0:6.1f}s] stage5 robustness done")
    av = stage6_adoption(df)
    print(f"[{time.time()-t0:6.1f}s] stage6 adoption variants done")
    mc_tab = stage7_mcse(df)
    print(f"[{time.time()-t0:6.1f}s] stage7 MC error done")
    mvt = _bridge_spec(df, "bridge2", bridge_cat=1, seed=SEED + 2)
    ct, _ = load_adoption(os.path.join(RAW, "agencies"), types=CORE_TYPES)
    ct_dat = apply_cohorts(df, dict(zip(ct["abbr"], ct["cohort"].astype(int))))
    pref = _bridge_spec(ct_dat, "preferred", bridge_cat=0, seed=SEED + 3)
    print(f"[{time.time()-t0:6.1f}s] stage8 MVT bridge + stage9 core-types done")
    # the stage-7 re-run reproduces the stage-1 bootstrap at the same seed
    se_repro = float((mc_tab.set_index("cat")["boot_se"]
                      - pm_tab.set_index("cat")["se"]).abs().max())

    post = dec[(dec["ell"] >= POST_W[0]) & (dec["ell"] <= POST_W[1])]
    shares = {}
    for c, g in post.groupby("cat"):
        rec, meas = g["es_recorded"].sum(), g["es_measurement"].sum()
        shares[NAMES[int(c)]] = {
            "recorded_mean": round(float(g["es_recorded"].mean()), 4),
            "latent_mean": round(float(g["es_latent"].mean()), 4),
            "measurement_mean": round(float(g["es_measurement"].mean()), 4),
            "measurement_share": (round(float(meas / rec), 4)
                                  if abs(rec) > 1e-9 else None),
        }
    pooled = float(post["es_measurement"].sum() / post["es_recorded"].sum())

    grad = grad_tab.set_index("version")
    summary = {
        "design": "UCR-SRS to NIBRS staggered state transition, 1979-2024",
        "panel": {"n_states": int(df["unit"].nunique()),
                  "years": [int(df["period"].min()), int(df["period"].max())],
                  "n_categories": len(CATS),
                  "cohort_range": [int(adopt[adopt.cohort > 0].cohort.min()),
                                   int(adopt[adopt.cohort > 0].cohort.max())],
                  "never_treated": sorted(adopt[adopt.cohort < 0].abbr.tolist()),
                  "n_clusters": int(df["unit"].nunique())},
        "discretion_d": {NAMES[c]: round(d_idx[c], 4) for c in CATS},
        "recorded_post_mean_log_effect": {
            NAMES[int(r["cat"])]: {"coef": round(float(r["post_mean"]), 4),
                                   "se": round(float(r["se"]), 4)}
            for _, r in pm_tab.iterrows()},
        # 4 sig digits: 4-dp rounding printed rape's p (3.6e-05) as exactly 0.0
        "pretrend_pvalues_twfe": {r["cat_name"]: float(f'{float(r["pvalue"]):.4g}')
                                  for _, r in pt_tab.iterrows()},
        "gradient_test": {
            v: {"stat": round(float(grad.loc[v, "stat"]), 3),
                "df": int(grad.loc[v, "df"]),
                "pvalue": float(grad.loc[v, "pvalue"]),
                "pre_placebo_pvalue": float(grad.loc[v, "pre_slopes_pvalue"])}
            for v in ("adjusted", "naive")},
        "bounds": {r["cat_name"]: {
            "post_mean": round(float(r["post_mean"]), 4),
            "r0_ncvs": round(float(r["r0_ncvs"]), 4),
            "identified_set": [round(float(r["bound_L"]), 4),
                               round(float(r["bound_U"]), 4)],
            "im_ci": [round(float(r["im_ci_lo"]), 4),
                      round(float(r["im_ci_hi"]), 4)],
            "breakdown_kappa_point": round(float(r["breakdown_kappa_point"]), 4),
            "breakdown_kappa_stat":
                (None if pd.isna(r["breakdown_kappa_stat"])
                 else round(float(r["breakdown_kappa_stat"]), 4))}
            for _, r in bounds_tab.iterrows()},
        "bridge": {"rho": {r["cat_name"]: round(float(r["rho"]), 4)
                           for _, r in rho.iterrows()},
                   "min_denom": float(rho["min_denom"].iloc[0]),
                   "measurement_by_cat": shares,
                   "measurement_share_pooled": round(pooled, 4),
                   "pooled_recorded_sum": round(float(post["es_recorded"]
                                                      .sum()), 4),
                   "pooled_measurement_sum": round(float(post["es_measurement"]
                                                         .sum()), 4)},
        "bridge_mvt": _bsum(mvt, TEXT["bridge_mvt_note"]),
        "preferred_core_types": _bsum(pref, TEXT["preferred_note"]),
        "robustness": {
            "pre2020_post_mean": {
                r["cat_name"]: {"coef": round(float(r["post_mean"]), 4),
                                "se": round(float(r["se"]), 4)}
                for _, r in rb_tab.iterrows() if int(r["cat"]) >= 0},
            "pre2020_measurement_share": {
                r["cat_name"]: (None if pd.isna(r["measurement_share"])
                                else round(float(r["measurement_share"]), 4))
                for _, r in rb_tab.iterrows() if int(r["cat"]) > 0},
            "pre2020_measurement_share_pooled":
                round(float(rb_tab.loc[rb_tab.cat == -1,
                                       "measurement_share"].iloc[0]), 4),
            "pre2020_pooled_share_boot_ci95": [
                round(float(rb_tab.loc[rb_tab.cat == -1,
                                       "share_boot_p2_5"].iloc[0]), 4),
                round(float(rb_tab.loc[rb_tab.cat == -1,
                                       "share_boot_p97_5"].iloc[0]), 4)],
            "gradient_variants": {
                f'{r["variant"]}_{r["version"]}': {
                    "pvalue": float(r["pvalue"]),
                    "pre_placebo_pvalue": float(r["pre_slopes_pvalue"])}
                for _, r in rg_tab.iterrows()},
            "adoption_rule_post_mean": {
                v: {NAMES[int(r["cat"])]: round(float(r["post_mean"]), 4)
                    for _, r in g.iterrows()}
                for v, g in av["pm"].groupby("variant")},
            "adoption_rule_n_never_treated": {
                v: int(g["n_never_treated"].iloc[0])
                for v, g in av["pm"].groupby("variant")},
            "adoption_rule_gradient": {
                f'{r["variant"]}_{r["version"]}': {
                    "pvalue": float(r["pvalue"]),
                    "pre_placebo_pvalue": float(r["pre_placebo_pvalue"]),
                    "mean_post_slope": round(float(r["mean_post_slope"]), 4)}
                for _, r in av["gt"].iterrows()},
            "hawaii_cohort_by_rule": {
                v: int(g.loc[g["abbr"] == "HI", "cohort"].iloc[0])
                for v, g in av["coh"].groupby("variant")},
        },
        "monte_carlo_error": {
            "note": ("MC SE of each B=300 bootstrap SE, by resampling the "
                     "draws; mc_se_of_t is the induced MC SE of the t stat."),
            "by_cat": {r["cat_name"]: {
                "boot_se": round(float(r["boot_se"]), 4),
                "mc_se_of_se": round(float(r["mc_se_of_se"]), 4),
                "t_stat": round(float(r["t_stat"]), 2),
                "mc_se_of_t": round(float(r["mc_se_of_t"]), 3)}
                for _, r in mc_tab.iterrows()},
            "stage1_bootstrap_reproduced_max_se_diff": se_repro},
        "gradient_post_slopes_adjusted": {
            str(int(r["ell"])): round(float(r["slope"]), 4)
            for _, r in pd.read_csv(os.path.join(RES, "gradient_slopes.csv"))
            .query("version == 'adjusted' and ell >= 0").iterrows()},
        "reps_used": {"state_bootstrap": B_BOOT,
                      "robustness_bootstrap": B_ROB},
        "post_window": list(POST_W),
        "runtime_min": round((time.time() - t0) / 60.0, 2),
        "runtime_note": RUNTIME_NOTE,
        "runtime_cold_total_min": 9.4,
        "caveats": json.load(open(os.path.join(
            ROOT, "data", "data_manifest.json")))["crime"]["caveats"],
    }
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(f"[{time.time()-t0:6.1f}s] summary written")
    print(json.dumps({"gradient_test": summary["gradient_test"],
                      "measurement_share_pooled":
                          summary["bridge"]["measurement_share_pooled"]},
                     indent=1))


if __name__ == "__main__":
    main()
