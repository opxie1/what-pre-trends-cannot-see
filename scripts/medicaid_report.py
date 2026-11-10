"""Shared study constants and the reporting layer for the Medicaid study.

Holds the paths, bootstrap sizes, event windows and category sets used by
scripts/medicaid.py, plus the two functions that turn the stage tables into
PNG figures and results/empirics/medicaid/summary.json. Split out of
medicaid.py only to keep every file under the 500-line limit; it imports
medicaid_lib but never medicaid, so the import graph stays acyclic."""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medicaid_lib as ml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "medicaid")
RES = os.path.join(ROOT, "results", "empirics", "medicaid")
PANEL = os.path.join(ROOT, "data", "processed", "medicaid_panel.csv")

B_MAIN = 400            # state bootstrap draws, main unified pass
B_ROB = 150             # draws per robustness subsample
B_YLOG = 250            # draws for the diabetes log-outcome series
POST_W = (0, 5)         # post window (event years) for headline aggregates
BRIDGE_W = (0, 3)       # bridge decomposition window (mortality ends 2017)
K_PRE, K_POST = 5, 6    # gradient-test event window (binned endpoints)
KT_PRE, KT_POST = 6, 8  # TWFE event window
SEED = 42
R0_DIABETES = 0.787     # diagnosed share of total diabetes (CDC NDSR 2020)

CATS_ALL = list(ml.CATS)        # 0..9
BRIDGE_SET = list(range(8))     # mortality bridge + 7 focal
FOCAL = list(range(1, 8))       # BRFSS focal categories
BRFSS = list(range(1, 8))       # BRFSS-only set (excludes mortality cat 0)
OBES = 5                        # obesity: internal low-discretion bridge
DIAGNOSED = [1, 2, 3, 4]        # diagnosis-dependent conditions
AUX = [8, 9]                    # insured, checkup (mechanism)
NAMES, D_IDX, ROLES, TXT = ml.NAMES, ml.D_IDX, ml.ROLES, ml.TXT


# raw-file key -> BRFSS topic string, for the confidence-limit extract that
# stage 7 uses to measure each cell's survey sampling error
CI_TOPICS = {"diabetes": "Diabetes", "hypertension": "High Blood Pressure",
             "cholesterol": "Cholesterol High", "depression": "Depression",
             "bmi": "BMI Categories", "smoking": "Current Smoker Status",
             "exercise": "Exercise", "coverage": "Under 65 Coverage",
             "checkup": "Last Checkup"}


def _wdemean(M, groups, w, n_iter=200, tol=1e-12):
    """Weighted alternating projections: remove w-weighted group means of every
    column of M for each grouping in `groups`."""
    M = np.array(M, dtype=float)
    codes = []
    for g in groups:
        _, inv = np.unique(np.asarray(g), return_inverse=True)
        codes.append((inv, np.bincount(inv, weights=w)))
    for _ in range(n_iter):
        mx = 0.0
        for inv, sw in codes:
            for j in range(M.shape[1]):
                m = np.bincount(inv, weights=w * M[:, j]) / sw
                M[:, j] -= m[inv]
                mx = max(mx, float(np.abs(m).max()))
        if mx < tol:
            break
    return M


def noise_decomposition(sub, post_window, k_pre, k_post):
    """For one category: (residual variance after state and year FE, mean
    design-based sampling variance of a cell, {unweighted, inverse-variance
    weighted} TWFE post-window means with state-clustered SEs).

    The sampling variance is measured, not assumed: it is the square of the
    SE implied by CDC's published design-based confidence limits for that
    state-year cell, so the split is not a modelling choice."""
    from did_measurement.estimators import build_event_dummies, cluster_ols

    y = sub["Y"].to_numpy(float)
    ss = sub["se_samp"].to_numpy(float)
    unit, per = sub["unit"].to_numpy(), sub["period"].to_numpy()
    res = _wdemean(y[:, None], (unit, per), np.ones(len(y)))[:, 0]
    # the two-way FE projection costs n_units + n_periods - 1 degrees of
    # freedom, so E[res'res] = (n - k_fe) * (v_true + v_samp). Dividing by
    # n - 1 instead would understate the denominator and overstate the
    # sampling-noise share (by ~6 pp at the median focal series here).
    k_fe = len(np.unique(unit)) + len(np.unique(per)) - 1
    v_res = float(res @ res / max(len(res) - k_fe, 1))
    v_samp = float(np.mean(ss ** 2))

    Xd, lab = build_event_dummies(sub["event_time"].to_numpy(), k_pre, k_post)
    a = np.zeros(len(lab))
    idx = [j for j, l in enumerate(lab) if post_window[0] <= l <= post_window[1]]
    a[idx] = 1.0 / len(idx)
    M0 = np.column_stack([y[:, None], Xd])
    out = {}
    for tag, w in (("unw", np.ones(len(y))), ("wls", 1.0 / ss ** 2)):
        w = w / w.mean()
        rt = np.sqrt(w)
        M = _wdemean(M0, (unit, per), w)
        b, V, _, _ = cluster_ols(M[:, 1:] * rt[:, None], M[:, 0] * rt, unit)
        out[f"twfe_{tag}"] = float(a @ b)
        out[f"twfe_{tag}_se"] = float(np.sqrt(max(a @ V @ a, 0.0)))
    wv = 1.0 / pd.Series(ss ** 2, index=unit).groupby(level=0).mean()
    out["n_eff_clusters"] = float(wv.sum() ** 2 / (wv ** 2).sum())
    return v_res, v_samp, out


def _r(x, n=4):
    """Round for JSON. NaN and +/-inf become null so summary.json stays
    strictly valid JSON (inf arises e.g. in mde/implied ratios at d_c = 0)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, n) if np.isfinite(v) else None


def _f(x):
    """Full-precision float for JSON (p-values); null if not finite."""
    v = float(x)
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------- figures

def fig_sampling_noise(noise, path):
    """Left: how much of each series' state-year variation is survey sampling
    noise. Right: the MDE with and without that noise, against the recording
    response the mechanism implies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = noise[noise["d"].notna()].sort_values("noise_share_of_var")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    yy = np.arange(len(t))
    ax[0].barh(yy, t["noise_share_of_var"], color="0.35")
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels(t["cat_name"], fontsize=8)
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("share of within-residual variance that is BRFSS\n"
                     "sampling noise (design-based CIs)", fontsize=9)
    ax[0].set_title("Survey noise dominates the state-year panel", fontsize=10)

    u = t[t["implied_pp"] > 0].sort_values("cat_name")
    xx = np.arange(len(u))
    ax[1].bar(xx - 0.26, u["cs_se"] * 2.802, 0.26, label="MDE80, as observed",
              color="0.25")
    ax[1].bar(xx, u["mde80_noisefree_pp"], 0.26,
              label="MDE80, zero sampling noise", color="0.6")
    ax[1].bar(xx + 0.26, u["implied_pp"], 0.26,
              label="mechanism-implied effect", color="0.85", edgecolor="0.3")
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(u["cat_name"], rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("pp", fontsize=9)
    ax[1].set_title("Even noise-free, the design cannot reach the mechanism",
                    fontsize=10)
    ax[1].legend(fontsize=7.5, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(tabs, tw_tab, sens, power, noise=None):
    es, pm = tabs["es_cs"], tabs["postmean_cs"]
    for c in BRIDGE_SET:
        ylab = ("age-adjusted deaths per 100k, effect" if c == 0
                else "crude prevalence, effect (pp)")
        ml.fig_es(es[es["cat"] == c], tw_tab[tw_tab["cat"] == c], NAMES[c],
                  ylab, os.path.join(RES, f"fig_es_{NAMES[c]}.png"))
    ml.fig_gradient(pm, os.path.join(RES, "fig_gradient.png"))
    ml.fig_first_stage(es[es["cat"] == 8], es[es["cat"] == 9],
                       os.path.join(RES, "fig_first_stage.png"))
    ml.fig_bridge(tabs["bridge_decomposition"],
                  os.path.join(RES, "fig_bridge.png"), BRIDGE_W, "mortality")
    ml.fig_bridge(tabs["bridge2_decomposition"],
                  os.path.join(RES, "fig_bridge_obesity.png"), POST_W,
                  "obesity")
    ml.fig_comovement(tabs["bridge_comovement"].set_index("cat_name"),
                      os.path.join(RES, "fig_comovement.png"))
    ml.fig_sensitivity(sens, os.path.join(RES,
                                          "fig_sensitivity_diabetes.png"))
    ml.fig_power(power, os.path.join(RES, "fig_power.png"))
    if noise is not None:
        fig_sampling_noise(noise, os.path.join(RES, "fig_sampling_noise.png"))


# ---------------------------------------------------------------- summary

def _verdict(slope, pw):
    """Headline verdict, read off the numbers rather than asserted: is the
    post-mean-on-d gradient significantly positive, and if not, can the design
    even detect the recording response the mechanism implies?"""
    b, se = float(slope["point"]), float(slope["se"])
    sig = np.isfinite(se) and se > 0 and abs(b) > 1.96 * se
    powered = bool((pw.loc[DIAGNOSED,
                           "power_ratio_mde80_over_implied"] < 1).any())
    if sig and b > 0:
        return "SUPPORTED: positive, significant discretion gradient"
    if sig:
        return "REJECTED WITH THE WRONG SIGN: significant negative gradient"
    return ("NOT SUPPORTED: gradient indistinguishable from zero" if powered
            else "NOT SUPPORTED, BUT THE DESIGN CANNOT DETECT IT EITHER")


def _grad_diagnostic(grad_tab, rg_tab, corr):
    """Is the loading-adjusted gradient test identified in this data?

    The adjustment divides each series by rho_c estimated from untreated
    comovement with the lowest-discretion reference category. If the reference
    barely comoves with the focal series, rho is a ratio of noise: the test
    statistic is then an artifact. Diagnosed by (i) the reference's untreated
    residual correlations, (ii) sign flips in rho between levels and logs,
    (iii) the spread of the adjusted p-value across specifications."""
    ref = NAMES[min(FOCAL, key=lambda c: (D_IDX[c], c))]
    g = grad_tab.set_index(["outcome", "version"])
    rho = {o: json.loads(g.loc[(o, "adjusted"), "rho"]) for o in ("Y", "ylog")}
    shared = [k for k in rho["Y"] if k in rho["ylog"] and k != ref]
    flips = sum(1 for k in shared
                if np.sign(rho["Y"][k]) != np.sign(rho["ylog"][k]))
    mx = max(abs(float(corr.loc[ref, NAMES[c]])) for c in DIAGNOSED)
    p = {v: sorted(list(g.xs(v, level="version")["pvalue"]) +
                   list(rg_tab.loc[rg_tab["version"] == v, "pvalue"]))
         for v in ("adjusted", "naive")}
    unstable = mx < 0.10 or flips >= 2
    verdict = (
        f"NOT IDENTIFIED: the reference series ({ref}, d=0) has |corr| <= "
        f"{mx:.3f} with every diagnosed condition in untreated residuals, so "
        f"the rho are ratios of noise; {flips} of {len(shared)} flip sign "
        f"between levels and logs, and the adjusted p-value ranges over "
        f"[{p['adjusted'][0]:.3f}, {p['adjusted'][-1]:.3f}] across the 6 "
        f"specifications while the naive p-value stays in "
        f"[{p['naive'][0]:.3f}, {p['naive'][-1]:.3f}]. The marginal adjusted "
        f"rejection on Y is an artifact of the normalization, not a "
        f"discretion gradient; read the naive test and the post-mean-on-d "
        f"slope instead." if unstable else
        f"loadings identified (|corr| up to {mx:.3f}, {flips} sign flips)")
    return {"loading_reference": ref,
            "reference_untreated_corr": {c: _r(corr.loc[ref, c], 3)
                                         for c in corr.columns if c != ref},
            "max_abs_corr_reference_vs_diagnosed": _r(mx, 3),
            "rho_levels": rho["Y"], "rho_logs": rho["ylog"],
            "rho_sign_flips_levels_vs_logs": int(flips),
            "adjusted_pvalue_range_6_specs": [_r(p["adjusted"][0]),
                                              _r(p["adjusted"][-1])],
            "naive_pvalue_range_6_specs": [_r(p["naive"][0]),
                                           _r(p["naive"][-1])],
            "adjusted_test_unreliable": bool(unstable),
            "verdict": verdict}


# stage-7 per-category columns carried into summary.json
COLS = ("d", "sd_samp_pp", "sd_resid_pp", "noise_share_of_var",
        "true_state_sd_pp", "cs_se", "cs_se_noisefree", "mde80_noisefree_pp",
        "implied_pp", "mde80_noisefree_over_implied")


def _noise_section(noise, ncorr):
    """Stage 7: split state-year variation into BRFSS design-based sampling
    noise and true state variation, then settle the two questions that split
    decides -- could the design ever detect the mechanism-implied recording
    response, and is the zero comovement that killed the loadings real?"""
    n = noise.set_index("cat_name")
    diag = [NAMES[c] for c in DIAGNOSED if NAMES[c] in n.index]
    ratio = {c: float(n.loc[c, "mde80_noisefree_over_implied"]) for c in diag}
    best = min(ratio.values())
    med = float(n.loc[[NAMES[c] for c in FOCAL], "noise_share_of_var"].median())
    wls = float((n["twfe_wls_se"] / n["twfe_unw_se"]).median())

    cm = {frozenset((r["cat_a"], r["cat_b"])): _r(r["corr_noise_corrected"], 3)
          for _, r in ncorr.iterrows()}
    ref = NAMES[min(FOCAL, key=lambda c: (D_IDX[c], c))]
    mort, refc, obes = ({c: cm.get(frozenset((k, c))) for c in diag}
                        for k in (NAMES[0], ref, NAMES[OBES]))
    mx = max(abs(v) for v in mort.values() if v is not None)
    # the reference's disattenuated correlations are not uniformly small, so
    # report the median and name the one condition that drives the maximum
    av = {k: abs(v) for k, v in refc.items() if v is not None}
    medr, argr = float(np.median(list(av.values()))), max(av, key=av.get)
    return {
        "definition": (
            "sampling variance is measured, not assumed: the square of the SE "
            "implied by CDC's own design-based 95% confidence limits per "
            "state-year cell. Residual variance is after state and year FE "
            "within category, with the degrees of freedom absorbed by those "
            "fixed effects netted out. Mortality carries no survey sampling error "
            "(NCHS deaths are a near-census); it disattenuates at share 0."),
        "median_noise_share_of_variance_focal": _r(med, 3),
        "by_category": {r["cat_name"]: {k: _r(r[k], 3) for k in COLS}
                        for _, r in noise.iterrows()},
        "inverse_variance_weighting": {
            "median_se_ratio_wls_over_unweighted": _r(wls, 3),
            "verdict": (
                "weighting cells by 1/sampling-variance does not buy "
                f"precision (median SE ratio {wls:.2f}); the binding "
                "constraint is 51 state clusters, not cell weights."
                if wls > 0.9 else
                f"weighting reduces the SE by a median factor {wls:.2f}")},
        "noise_corrected_comovement": {
            "note": ("observed untreated correlation divided by "
                     "sqrt((1-share_a)(1-share_b)); the ceiling any loading "
                     "estimate could reach if survey noise were removed"),
            "mortality_vs_diagnosed": mort,
            f"{ref}_reference_vs_diagnosed": refc,
            "obesity_vs_diagnosed": obes,
            "verdict": (
                f"the two loading failures are real, not attenuation: even "
                f"disattenuated, the mortality bridge reaches |r| <= {mx:.3f} "
                f"against every diagnosed condition, and the {ref} reference "
                f"has median |r| = {medr:.3f} across them (its only non-"
                f"trivial partner is {argr}, |r| = {av[argr]:.2f}), while the "
                f"diagnosed conditions comove with obesity at r up to "
                f"{max(v for v in obes.values() if v is not None):.2f}. "
                "Obesity is the only bridge with a denominator.")},
        "power_verdict": (
            "DESIGN-LEVEL, NOT DATA-QUALITY: even at zero BRFSS sampling "
            "noise the 80%-power MDE would still exceed the mechanism-implied "
            f"recording response by a factor of {best:.1f} for the best-placed "
            f"diagnosed condition ({min(ratio, key=ratio.get)}), so the null "
            "is a power failure of the state-year aggregate design and cannot "
            "be read as evidence against a recording response."
            if best > 1 else
            f"at zero sampling noise the design would be powered "
            f"(MDE80/implied = {best:.2f}), making the null informative."),
        "min_mde80_noisefree_over_implied": _r(best, 2)}


def summarize(df, tabs, pt_tab, grad_tab, rb_tab, rg_tab, bounds, power,
              noise, ncorr, t0):
    pm, pw = tabs["postmean_cs"].set_index("cat"), power.set_index("cat")
    share = tabs["bridge_shares"].set_index("cat")
    sh2 = tabs["bridge2_shares"].set_index("cat")
    rho, rho2 = tabs["bridge_rho"], tabs["bridge2_rho"]
    slope = tabs["gradient_slope_boot"].iloc[0]
    slope_o = tabs["gradient_slope_odd_boot"].iloc[0]
    grad = grad_tab.set_index(["outcome", "version"])
    corr = tabs["bridge_comovement"].set_index("cat_name")
    bd = bounds.set_index(["cat", "target"])
    dia, dia0 = bd.loc[(1, "postmean_0_5")], bd.loc[(1, "ell_0")]
    gdiag = _grad_diagnostic(grad_tab, rg_tab, corr)
    nsec = _noise_section(noise, ncorr)

    def _pmrow(c):
        return {"coef": _r(pm.loc[c, "post_mean"]), "se": _r(pm.loc[c, "se"])}

    def _shrow(t, c):
        return {"recorded_mean": _r(t.loc[c, "recorded_mean"]),
                "latent_mean": _r(t.loc[c, "latent_mean"]),
                "share_point": _r(t.loc[c, "share_point"]),
                "share_boot_ci95": [_r(t.loc[c, "share_p2_5"]),
                                    _r(t.loc[c, "share_p97_5"])]}

    def _bridge(cat, win, rho_tab, share_tab, verdict):
        # the share (recorded - latent)/recorded divides by a recorded effect
        # that is itself indistinguishable from zero here, so record how wide
        # the bootstrap CI is rather than presenting the point share as a fact
        w = [abs(share_tab.loc[c, "share_p97_5"] - share_tab.loc[c, "share_p2_5"])
             for c in DIAGNOSED]
        wmin = min([x for x in w if np.isfinite(x)], default=np.nan)
        bad = np.isfinite(wmin) and wmin > 2.0
        return {"bridge_outcome": NAMES[cat], "d_bridge": D_IDX[cat],
                "window": list(win),
                "rho_iv": {r["cat_name"]: _r(r["rho"])
                           for _, r in rho_tab.iterrows()},
                "min_denom": _f(rho_tab["min_denom"].iloc[0]),
                "measurement_share_diagnosed": {NAMES[c]: _shrow(share_tab, c)
                                                for c in DIAGNOSED},
                "shares_uninformative": bool(bad),
                "share_ci95_width_min": _r(wmin, 2),
                "share_note": (
                    "UNINFORMATIVE: the narrowest bootstrap 95% CI on the "
                    f"measurement share spans {wmin:.1f} (i.e. {100*wmin:.0f} "
                    "percentage points of share). The share divides by a "
                    "recorded effect that is itself indistinguishable from "
                    "zero, so no measurement share is estimable here in either "
                    "direction." if bad else "share CIs are finite and usable"),
                "verdict": TXT[verdict]}

    summary = {
        "design": TXT["design"],
        "headline": {
            "expected_pattern": TXT["expected_pattern"],
            "found": _verdict(slope, pw),
            "gradient_slope_pp_per_unit_d": _r(slope["point"]),
            "gradient_slope_se": _r(slope["se"]),
            "direction": TXT["direction"],
            "diabetes_diagnosed_post_mean_pp": _r(pm.loc[1, "post_mean"]),
            "first_stage_insured_pp": _r(pm.loc[8, "post_mean"]),
            "diabetes_mde80_pp": _r(pw.loc[1, "mde80_pp"]),
            "diabetes_mechanism_implied_pp": _r(pw.loc[1, "implied_pp"]),
            "gradient_test_pvalue_naive_Y": _f(
                grad_tab.set_index(["outcome", "version"]).loc[
                    ("Y", "naive"), "pvalue"]),
            "loading_adjusted_test_unreliable_here":
                gdiag["adjusted_test_unreliable"],
            "sampling_noise_share_of_state_year_variance":
                nsec["median_noise_share_of_variance_focal"],
            "min_mde80_noisefree_over_implied":
                nsec["min_mde80_noisefree_over_implied"],
            "interpretation": TXT["interpretation"]},
        "panel": {
            "n_states": int(df["unit"].nunique()),
            "years": [int(df["period"].min()), int(df["period"].max())],
            "n_categories": len(CATS_ALL),
            "never_treated": sorted(df.loc[df["cohort"] == -1,
                                           "abbr"].unique().tolist()),
            "cohorts": {str(k): int(v) for k, v in
                        df[df["cat"] == 1].groupby(
                            "cohort")["unit"].nunique().items()}},
        "discretion_d": {NAMES[c]: D_IDX[c] for c in BRIDGE_SET},
        "first_stage_post_mean_pp": {NAMES[c]: _pmrow(c) for c in AUX},
        "recorded_post_mean": {NAMES[c]: _pmrow(c) for c in BRIDGE_SET},
        "recorded_post_mean_units": TXT["units"],
        "pretrend_pvalues_twfe": {
            w: {r["cat_name"]: _r(r["pvalue"]) for _, r in g.iterrows()}
            for w, g in pt_tab.groupby("window")},
        "pretrend_note": TXT["short_pretrend_note"],
        "power_and_mde": {
            "note": TXT["power_note"],
            "insured_gain_pp": _r(pm.loc[8, "post_mean"]),
            "unaware_share_u": round(1 - R0_DIABETES, 3),
            "by_condition": {
                r["cat_name"]: {
                    "d": r["d"], "baseline_prev_pp": _r(r["baseline_prev_pp"], 2),
                    "post_mean_pp": _r(r["post_mean"]), "se": _r(r["se"]),
                    "mde80_pp": _r(r["mde80_pp"]), "implied_pp": _r(r["implied_pp"]),
                    "implied_pp_3x": _r(r["implied_pp_3x"]),
                    "mde80_over_implied": _r(r["power_ratio_mde80_over_implied"], 1)}
                for _, r in pw.reset_index().iterrows()}},
        "sampling_noise": nsec,
        "common_support_odd_ell": {
            "note": TXT["odd_ell_note"],
            "post_mean_pp": {r["cat_name"]: {"coef": _r(r["post_mean"]),
                                             "se": _r(r["se"])}
                             for _, r in tabs["postmean_odd_cs"].iterrows()},
            "gradient_slope_pp_per_unit_d": _r(slope_o["point"]),
            "gradient_slope_se": _r(slope_o["se"]),
            "boot_ci95": [_r(slope_o["p2_5"]), _r(slope_o["p97_5"])]},
        "gradient_test": {
            f"{o}_{v}": {"stat": _r(grad.loc[(o, v), "stat"], 3),
                         "df": int(grad.loc[(o, v), "df"]),
                         "pvalue": _f(grad.loc[(o, v), "pvalue"]),
                         "pre_placebo_pvalue": _f(
                             grad.loc[(o, v), "pre_slopes_pvalue"])}
            for o in ("Y", "ylog") for v in ("adjusted", "naive")},
        "gradient_test_diagnostic": gdiag,
        "gradient_slope_postmean_on_d": {
            "pp_per_unit_d": _r(slope["point"]), "se": _r(slope["se"]),
            "boot_ci95": [_r(slope["p2_5"]), _r(slope["p97_5"])]},
        "bounds_diabetes": {
            "post_mean_pp": _r(dia["coef"]), "se": _r(dia["se"]),
            "breakdown_kappa_point_pp": _r(dia["breakdown_kappa_point"]),
            "breakdown_kappa_stat_pp": _r(dia["breakdown_kappa_stat"]),
            "ell0_coef_pp": _r(dia0["coef"]), "ell0_se": _r(dia0["se"]),
            "recording_bounds_log": {
                "r0": R0_DIABETES, "beta_log": _r(dia["pm_log"]),
                "se": _r(dia["pm_log_se"]),
                "identified_set": [_r(dia["rb_log_L"]), _r(dia["rb_log_U"])],
                "im_ci": [_r(dia["rb_log_ci_lo"]), _r(dia["rb_log_ci_hi"])]}},
        "breakdown_kappa_by_condition": {
            r["cat_name"]: {"coef": _r(r["coef"]),
                            "point": _r(r["breakdown_kappa_point"]),
                            "stat": _r(r["breakdown_kappa_stat"])}
            for _, r in bounds.iterrows() if r["target"] == "postmean_0_5"},
        "bridge_comovement": {
            "definition": TXT["comovement_definition"],
            "mortality_vs_focal": {c: _r(corr.loc[NAMES[0], c], 3)
                                   for c in corr.columns if c != NAMES[0]},
            "obesity_vs_diagnosed": {NAMES[c]: _r(corr.loc[NAMES[OBES],
                                                           NAMES[c]], 3)
                                     for c in DIAGNOSED},
            "verdict": TXT["comovement_verdict"]},
        "bridge_mortality": _bridge(0, BRIDGE_W, rho, share, "mortality_verdict"),
        "bridge_obesity": _bridge(OBES, POST_W, rho2, sh2, "obesity_verdict"),
        "robustness": {
            "post_mean_pp": {
                v: {r["cat_name"]: {"coef": _r(r["post_mean"]),
                                    "se": _r(r["se"])} for _, r in g.iterrows()}
                for v, g in rb_tab.groupby("variant")},
            "gradient_variants": {
                f'{r["variant"]}_{r["version"]}': {
                    "pvalue": _f(r["pvalue"]),
                    "pre_placebo_pvalue": _f(r["pre_slopes_pvalue"])}
                for _, r in rg_tab.iterrows()}},
        "reps_used": {"main_state_bootstrap": B_MAIN,
                      "robustness_bootstrap_per_variant": B_ROB,
                      "diabetes_ylog_bootstrap": B_YLOG},
        "post_window": list(POST_W),
        "runtime_min": round((time.time() - t0) / 60.0, 2),
        "runtime_note": (
            "wall time of this invocation of scripts/medicaid.py only. Stages "
            "whose checkpoint CSVs already exist are read, not recomputed, so a "
            "resumed run reports far less than a rebuild; a from-scratch run of "
            "all seven stages takes about 4.3 min. Neither figure includes the "
            "one-off CDC/NCHS API downloads, which are cached under "
            "data/raw/medicaid/."),
        "caveats": json.load(open(os.path.join(
            ROOT, "data", "data_manifest.json")))["medicaid"]["caveats"],
    }
    # allow_nan=False: fail loudly rather than emit NaN/Infinity, which are
    # not valid JSON and silently break strict parsers downstream.
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1,
              allow_nan=False)
    return summary
