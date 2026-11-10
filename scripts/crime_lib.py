"""Helpers for the crime (NIBRS transition) empirical study: panel build, a
vectorized Callaway-Sant'Anna point estimator (verified at runtime to match
did_measurement.estimators.att_gt_cs exactly), a unified state-cluster
bootstrap (event studies + post-means + bridge decomposition), and figures.
Estimation uses only observed columns (unit, period, cohort, cat, d, D,
event_time, Y/y, exposure)."""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from did_measurement.estimators import att_gt_cs
from did_measurement.bridge import estimate_relative_loadings
from did_measurement.mtest import discretion_gradient_test

# agency types that actually report index crime (excludes 'Other State Agency',
# 'Other', 'University or College', 'Tribal'); used by the adoption-rule
# robustness variants, where dateless roster entries otherwise dilute the share
CORE_TYPES = ("City", "County", "State Police")

# Static manifest/summary prose (design choice, rejected alternatives, sources,
# repairs, runtime note, caveats), kept as data in crime_caveats.json because it
# is documentation text, not logic, and would otherwise push these files past
# the 500-line limit. Detail behind each caveat is recoverable from the stored
# tables (adoption_variants*.csv, gradient_slopes.csv, mc_se.csv, review.json).
TEXT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "crime_caveats.json")))
CAVEATS = TEXT["caveats"]

# category map: cat code -> (json field, display name)
CATS = {
    0: ("homicide", "homicide"),
    1: ("motorVehicleTheft", "motor_vehicle_theft"),
    2: ("aggravatedAssault", "aggravated_assault"),
    3: ("robbery", "robbery"),
    4: ("burglary", "burglary"),
    5: ("rape", "rape"),
    6: ("larceny", "larceny"),
}

# NCVS percent of victimizations reported to police, 2018 / 2019
# (BJS, Criminal Victimization 2019, NCJ 255113, Table 6). Averaged for stability.
NCVS_REPORTED = {
    1: (78.6, 79.5),   # motor-vehicle theft
    2: (60.5, 52.1),   # aggravated assault
    3: (62.6, 46.6),   # robbery
    4: (47.5, 51.4),   # burglary (subcategory, not burglary/trespassing)
    5: (24.9, 33.9),   # rape/sexual assault
    6: (28.6, 26.8),   # other theft ~ UCR larceny
}


def discretion_index():
    """d_c = 1 - mean reporting rate, rescaled so max focal = 1; homicide d = 0."""
    r = {c: (a + b) / 2.0 / 100.0 for c, (a, b) in NCVS_REPORTED.items()}
    d_raw = {c: 1.0 - v for c, v in r.items()}
    mx = max(d_raw.values())
    d = {0: 0.0, **{c: v / mx for c, v in d_raw.items()}}
    return d, r


def load_adoption(agency_dir, threshold=0.5, types=None, shift=0):
    """Per-state NIBRS adoption from FBI CDE agency rosters. cohort = first year
    >= threshold of listed agencies have nibrs_start_date <= that year; -1 if
    never reached by 2024. types restricts the roster to these agency_type_name
    values (None = all; CORE_TYPES drops dateless 'Other State Agency' entries).
    shift adds years to every cohort (roster dates lead actual submission)."""
    rows, shares = [], {}
    for fn in sorted(os.listdir(agency_dir)):
        if not fn.endswith(".json"):
            continue
        st = fn[:-5]
        j = json.load(open(os.path.join(agency_dir, fn)))
        ags = [a for county in j.values() for a in county]
        if types is not None:
            ags = [a for a in ags if a.get("agency_type_name") in types]
        yrs = np.array([int(a["nibrs_start_date"][:4]) if a.get("nibrs_start_date")
                        else 9999 for a in ags])
        n = len(yrs)
        if n == 0:
            rows.append((st, 0, -1, np.nan))
            shares[st] = {y: 0.0 for y in range(1985, 2025)}
            continue
        path = {y: float((yrs <= y).sum()) / n for y in range(1985, 2025)}
        cohort = -1
        for y in range(1985, 2025):
            if path[y] >= threshold:
                cohort = y + shift
                break
        if cohort > 2024:
            cohort = -1            # shifted past the sample: never-treated
        rows.append((st, n, cohort, path[2024]))
        shares[st] = path
    tab = pd.DataFrame(rows, columns=["abbr", "n_agencies", "cohort", "share_2024"])
    return tab, shares


def build_panel(trends_path, agency_dir, out_csv):
    """State x year x category long panel with log recorded rates."""
    raw = json.load(open(trends_path))
    # repair: ND's 2022 row is misfiled under NC in the source feed
    nc = next(s for s in raw if s["abbr"] == "NC")
    nd = next(s for s in raw if s["abbr"] == "ND")
    dups = [d for d in nc["years"] if d["year"] == 2022]
    if len(dups) == 2 and not any(d["year"] == 2022 for d in nd["years"]):
        stray = min(dups, key=lambda d: d["population"])  # ND pop ~0.78M
        nc["years"].remove(stray)
        nd["years"].append(stray)

    adopt, shares = load_adoption(agency_dir)
    adopt = adopt.set_index("abbr")
    d_idx, _ = discretion_index()

    recs = []
    abbrs = sorted(s["abbr"] for s in raw)
    code = {a: i for i, a in enumerate(abbrs)}
    for s in raw:
        ab = s["abbr"]
        g = int(adopt.loc[ab, "cohort"])
        for yr in s["years"]:
            y = yr["year"]
            for c, (field, name) in CATS.items():
                cnt = float(yr[field])
                pop = float(yr["population"])
                recs.append((code[ab], ab, y, g, c, name, d_idx[c],
                             int(g > 0 and y >= g),
                             (y - g) if g > 0 else np.nan,
                             cnt, pop,
                             np.log((cnt + 0.5) / pop * 1e5)))
    df = pd.DataFrame(recs, columns=["unit", "abbr", "period", "cohort", "cat",
                                     "cat_name", "d", "D", "event_time",
                                     "Y", "exposure", "y"])
    df = df.sort_values(["cat", "unit", "period"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df, adopt.reset_index(), shares


# ------------------------------------------------- fast CS (verified vs library)

def cs_es_fast(sub, outcome="y", control="notyet"):
    """Vectorized replica of estimators._cs_point + ES aggregation.

    Same semantics: base period g-1, controls are never-treated only
    (control='never') or cohorts h > max(t, g) plus never-treated
    (control='notyet'), cohort-size weights, ES = size-weighted mean over (g,t)
    at each ell. Returns dict ell -> coef. Verified against att_gt_cs at runtime
    (see verify_fast_cs)."""
    if control not in ("notyet", "never"):
        raise ValueError(f"unknown control scheme {control!r}")
    m = sub.groupby(["cohort", "period"], sort=True)[outcome].mean().unstack()
    sizes = sub.groupby("cohort")["unit"].nunique()
    periods = np.asarray(m.columns)
    cohs = np.asarray([g for g in m.index if g >= 0])
    if len(cohs) == 0:
        return {}
    Mg = m.loc[cohs].to_numpy()                      # K x T
    wg = sizes.loc[cohs].to_numpy(float)
    has_never = -1 in m.index
    Mn = m.loc[-1].to_numpy() if has_never else None
    wn = float(sizes.loc[-1]) if has_never else 0.0
    if control == "never" and not has_never:
        raise ValueError("no never-treated units for control='never'")
    WM = wg[:, None] * Mg
    SS = np.vstack([np.cumsum(WM[::-1], axis=0)[::-1],
                    np.zeros((1, len(periods)))])    # SS[j] = sum_{k>=j} w*M
    SW = np.concatenate([np.cumsum(wg[::-1])[::-1], [0.0]])
    pos = {int(p): i for i, p in enumerate(periods)}
    es_num, es_den = {}, {}
    for gi, g in enumerate(cohs):
        b = pos.get(int(g) - 1)
        if b is None:
            continue
        for ti, t in enumerate(periods):
            if control == "never":
                num, den = 0.0, 0.0
            else:
                j = int(np.searchsorted(cohs, max(int(t), int(g)), side="right"))
                num = SS[j, ti] - SS[j, b]
                den = SW[j]
            if has_never:
                num += wn * (Mn[ti] - Mn[b])
                den += wn
            if den <= 0:
                continue
            att = (Mg[gi, ti] - Mg[gi, b]) - num / den
            l = int(t) - int(g)
            es_num[l] = es_num.get(l, 0.0) + att * wg[gi]
            es_den[l] = es_den.get(l, 0.0) + wg[gi]
    return {l: es_num[l] / es_den[l] for l in sorted(es_num)}


def verify_fast_cs(df, cats, outcome="y", control="notyet", tol=1e-10):
    """Assert cs_es_fast == att_gt_cs on the real panel for given categories."""
    for c in cats:
        sub = df[df["cat"] == c]
        slow = att_gt_cs(sub, outcome=outcome, control=control, n_boot=0)["es"]
        slow = dict(zip(slow["ell"].astype(int), slow["coef"]))
        fast = cs_es_fast(sub, outcome=outcome, control=control)
        keys = set(slow) | set(fast)
        diff = max(abs(slow[k] - fast[k]) for k in keys)
        assert diff < tol, f"fast CS mismatch cat {c}: {diff}"
    return True


# ------------------------------------------------- unified state bootstrap

def _decompose(es_by_cat, rho, bridge_cat=0):
    rows = []
    br = es_by_cat.get(bridge_cat, {})
    for c, es in es_by_cat.items():
        if c == bridge_cat:
            continue
        for l, v in es.items():
            lat = rho.get(c, np.nan) * br.get(l, np.nan)
            rows.append((c, l, v, lat, v - lat))
    return rows


def bootstrap_all(df, cats, outcome="y", control="notyet", B=300, seed=71,
                  bridge_cat=0, post_window=(0, 5), ell_lo=-12, ell_hi=12,
                  loading_method="iv"):
    """One state-cluster bootstrap feeding all deliverables.

    Per draw: resample 51 states with replacement, recompute (a) CS event
    studies per category (fast estimator), (b) post-window means, (c) relative
    loadings + bridge decomposition, (d) pooled measurement share.
    Returns dict of tables and draw arrays."""
    # point estimates via the fast estimator (verified against att_gt_cs by
    # verify_fast_cs before this is called)
    point = {c: cs_es_fast(df[df["cat"] == c], outcome, control) for c in cats}
    ld = estimate_relative_loadings(df, outcome=outcome, bridge_cat=bridge_cat,
                                    method=loading_method)
    rho_pt, min_denom = ld["rho"], ld["min_denom"]
    dec_pt = _decompose(point, rho_pt, bridge_cat)

    rng = np.random.default_rng(seed)
    units = np.sort(df["unit"].unique())
    gb = {u: g for u, g in df.groupby("unit", sort=False)}
    es_draws = {c: {} for c in cats}
    pm_draws = {c: [] for c in cats}
    dec_draws = {}      # (cat, ell) -> list of (rec, lat, meas)
    share_draws = []
    rho_draws = {c: [] for c in cats if c != bridge_cat}
    lo, hi = post_window
    for b in range(B):
        samp = rng.choice(units, size=len(units), replace=True)
        parts = []
        for j, u in enumerate(samp):
            p = gb[u].copy()
            p["unit"] = j
            parts.append(p)
        dfb = pd.concat(parts, ignore_index=True)
        es_b = {}
        for c in cats:
            es_b[c] = cs_es_fast(dfb[dfb["cat"] == c], outcome, control)
            for l, v in es_b[c].items():
                if np.isfinite(v):
                    es_draws[c].setdefault(l, []).append(v)
            post = [v for l, v in es_b[c].items() if lo <= l <= hi
                    and np.isfinite(v)]
            if post:
                pm_draws[c].append(float(np.mean(post)))
        try:
            ld_b = estimate_relative_loadings(dfb, outcome=outcome,
                                              bridge_cat=bridge_cat,
                                              method=loading_method)
        except (ValueError, KeyError, np.linalg.LinAlgError):
            continue
        for c, v in ld_b["rho"].items():
            rho_draws[int(c)].append(v)
        rec_s = meas_s = 0.0
        for (c, l, rec, lat, meas) in _decompose(es_b, ld_b["rho"], bridge_cat):
            if np.isfinite(rec) and np.isfinite(meas):
                dec_draws.setdefault((c, l), []).append((rec, lat, meas))
                if lo <= l <= hi:
                    rec_s += rec
                    meas_s += meas
        if abs(rec_s) > 1e-9:
            share_draws.append(meas_s / rec_s)

    def _se(v, min_n=20):
        return float(np.std(v, ddof=1)) if len(v) > min_n else np.nan

    es_rows = [(c, l, point[c][l], _se(es_draws[c].get(l, [])),
                len(es_draws[c].get(l, [])))
               for c in cats for l in sorted(point[c]) if ell_lo <= l <= ell_hi]
    es_tab = pd.DataFrame(es_rows, columns=["cat", "ell", "coef", "se", "n_boot"])
    pm_rows = []
    for c in cats:
        post = [v for l, v in point[c].items() if lo <= l <= hi
                and np.isfinite(v)]
        pm_rows.append((c, float(np.mean(post)) if post else np.nan,
                        _se(pm_draws[c]), len(pm_draws[c])))
    pm_tab = pd.DataFrame(pm_rows, columns=["cat", "post_mean", "se", "n_boot"])
    dec_rows = []
    for (c, l, rec, lat, meas) in dec_pt:
        if not (ell_lo <= l <= ell_hi):
            continue
        dd = np.asarray(dec_draws.get((c, l), np.empty((0, 3))))
        ses = dd.std(axis=0, ddof=1) if len(dd) > 20 else [np.nan] * 3
        dec_rows.append((c, l, rec, lat, meas, ses[0], ses[1], ses[2]))
    dec_tab = pd.DataFrame(dec_rows, columns=[
        "cat", "ell", "es_recorded", "es_latent", "es_measurement",
        "se_es_recorded", "se_es_latent", "se_es_measurement"])
    rho_tab = pd.DataFrame({
        "cat": [int(c) for c in rho_pt],
        "rho": [rho_pt[c] for c in rho_pt],
        "se": [_se(rho_draws[int(c)]) for c in rho_pt]})
    rho_tab["min_denom"] = min_denom
    share_arr = np.asarray(share_draws)
    return {"es": es_tab, "postmean": pm_tab, "dec": dec_tab, "rho": rho_tab,
            "pm_draws": {c: np.asarray(v) for c, v in pm_draws.items()},
            "share_draws": share_arr, "n_draws": B}


def share_table(dec, share_draws, names, post_window=(0, 5), tag="spec"):
    """Post-window measurement decomposition per category plus the pooled share.

    Shares divide by the recorded effect, so they are only interpretable where
    that denominator is away from zero; the measurement LEVEL (log points) is
    the stable object. The pooled row carries the bootstrap percentile CI."""
    t = dec[(dec["ell"] >= post_window[0]) & (dec["ell"] <= post_window[1])]
    rows = []
    for c, g in t.groupby("cat"):
        rec, meas = g["es_recorded"].sum(), g["es_measurement"].sum()
        rows.append((tag, int(c), names[int(c)], g["es_recorded"].mean(),
                     g["es_latent"].mean(), g["es_measurement"].mean(),
                     meas / rec if abs(rec) > 1e-9 else np.nan, np.nan, np.nan))
    rec_s, meas_s = t["es_recorded"].sum(), t["es_measurement"].sum()
    sd = np.asarray([x for x in share_draws if np.isfinite(x)])
    q = np.percentile(sd, [2.5, 97.5]) if len(sd) > 20 else [np.nan, np.nan]
    rows.append((tag, -1, "pooled", np.nan, np.nan, np.nan,
                 meas_s / rec_s if abs(rec_s) > 1e-9 else np.nan, q[0], q[1]))
    return pd.DataFrame(rows, columns=["variant", "cat", "cat_name",
                                       "recorded_mean", "latent_mean",
                                       "measurement_mean", "measurement_share",
                                       "share_boot_p2_5", "share_boot_p97_5"])


# ------------------------------------------- adoption-rule robustness

def apply_cohorts(df, coh):
    """Re-derive cohort/D/event_time under an alternative adoption rule."""
    out = df.copy()
    g = out["abbr"].map(coh).astype(float)
    out["cohort"] = g.astype(int)
    out["D"] = ((g > 0) & (out["period"] >= g)).astype(int)
    out["event_time"] = np.where(g > 0, out["period"] - g, np.nan)
    return out


def run_adoption_variants(df, agency_dir, cats, variants, post_window=(0, 5),
                          k_pre=6, k_post=8, outcome="y", control="notyet"):
    """Point post-means and gradient tests under alternative treatment-timing
    definitions (deterministic, no bootstrap): how much the headline numbers
    depend on the 50%-of-full-roster adoption rule. variants is a list of
    (tag, spec) where spec may set threshold, types, shift, drop_never,
    max_year."""
    lo, hi = post_window
    pm_rows, g_rows, coh_rows = [], [], []
    for tag, spec in variants:
        tab, _ = load_adoption(agency_dir, threshold=spec.get("threshold", 0.5),
                               types=spec.get("types"), shift=spec.get("shift", 0))
        coh = dict(zip(tab["abbr"], tab["cohort"].astype(int)))
        dat = apply_cohorts(df, coh)
        if spec.get("max_year"):
            dat = dat[dat["period"] <= spec["max_year"]]
        if spec.get("drop_never"):
            dat = dat[dat["cohort"] > 0]
        dat = dat.reset_index(drop=True)
        n_never = int((dat.drop_duplicates("unit")["cohort"] < 0).sum())
        for st, gg in sorted(coh.items()):
            coh_rows.append((tag, st, int(gg)))
        for c in cats:
            es = cs_es_fast(dat[dat["cat"] == c], outcome, control)
            post = [v for l, v in es.items() if lo <= l <= hi and np.isfinite(v)]
            pm_rows.append((tag, c, float(np.mean(post)) if post else np.nan,
                            n_never, int(dat["unit"].nunique())))
        for adj in (True, False):
            r = discretion_gradient_test(dat, outcome=outcome, k_pre=k_pre,
                                         k_post=k_post, bridge_cat=0,
                                         use_bridge=False, loading_adjust=adj)
            sl = r["slopes"]
            postsl = sl[(sl["ell"] >= lo) & (sl["ell"] <= hi)]["slope"]
            g_rows.append((tag, "adjusted" if adj else "naive", r["stat"],
                           r["df"], r["pvalue"], r["pre_slopes_pvalue"],
                           float(postsl.mean()) if len(postsl) else np.nan))
    pm = pd.DataFrame(pm_rows, columns=["variant", "cat", "post_mean",
                                        "n_never_treated", "n_states"])
    gt = pd.DataFrame(g_rows, columns=["variant", "version", "stat", "df",
                                       "pvalue", "pre_placebo_pvalue",
                                       "mean_post_slope"])
    ct = pd.DataFrame(coh_rows, columns=["variant", "abbr", "cohort"])
    return pm, gt, ct


# ------------------------------------------- Monte Carlo error of bootstrap SEs

def mc_se_of_sd(draws, R=1000, seed=913):
    """MC standard error of a bootstrap SD, by resampling the B draws. Returns
    (sd, mc_se_of_sd, kurtosis-based analytic approximation)."""
    v = np.asarray([x for x in draws if np.isfinite(x)])
    B = len(v)
    if B < 20:
        return np.nan, np.nan, np.nan
    sd = float(v.std(ddof=1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(R, B))
    sds = v[idx].std(axis=1, ddof=1)
    z = (v - v.mean()) / sd
    kurt = float((z ** 4).mean())
    return sd, float(sds.std(ddof=1)), float(sd * np.sqrt((kurt - 1) / (4 * B)))


# ------------------------------------------------- figures

def fig_event_study(es_cat, twfe_cat, name, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    e = es_cat.dropna(subset=["coef"])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(-0.5, color="0.8", lw=0.8, ls="--")
    ax.errorbar(e["ell"], e["coef"], yerr=1.96 * e["se"], fmt="o", ms=3.5,
                lw=1, capsize=2, color="tab:blue", label="CS (not-yet control)")
    if twfe_cat is not None:
        ax.plot(twfe_cat["ell"], twfe_cat["coef"], "s--", ms=3, lw=0.8,
                color="tab:orange", alpha=0.8, label="TWFE (binned)")
    ax.set_xlabel("years since NIBRS transition (50% of agencies)")
    ax.set_ylabel("log recorded rate, effect")
    ax.set_title(name)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gradient(pm, d_idx, names, path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(0, color="0.6", lw=0.8)
    for _, r in pm.iterrows():
        c = int(r["cat"])
        x = d_idx[c]
        col = "tab:red" if c == 0 else "tab:blue"
        ax.errorbar([x], [r["post_mean"]], yerr=[1.96 * r["se"]], fmt="o",
                    color=col, ms=5, capsize=3)
        ax.annotate(names[c], (x, r["post_mean"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    foc = pm[pm["cat"] != 0]
    xs = np.array([d_idx[int(c)] for c in foc["cat"]])
    ys = foc["post_mean"].to_numpy()
    b = np.polyfit(xs, ys, 1)
    xx = np.linspace(0, 1.05, 20)
    ax.plot(xx, np.polyval(b, xx), "-", color="0.4", lw=1,
            label=f"focal fit: slope={b[0]:.3f}")
    ax.set_xlabel("discretion index d_c (1 - NCVS reporting rate, rescaled)")
    ax.set_ylabel("mean recorded effect, ell 0-5 (log points)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_adoption(shares, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for st, p in shares.items():
        ys = sorted(p)
        ax.plot(ys, [p[y] for y in ys], color="0.7", lw=0.7)
    ax.axhline(0.5, color="tab:red", lw=0.9, ls="--", label="cohort threshold")
    ax.set_xlabel("year")
    ax.set_ylabel("share of agencies reporting NIBRS")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bridge(dec, names, path, post_window=(0, 5)):
    t = dec[(dec["ell"] >= post_window[0]) & (dec["ell"] <= post_window[1])]
    agg = t.groupby("cat")[["es_recorded", "es_latent", "es_measurement"]].mean()
    cats = list(agg.index)
    x = np.arange(len(cats))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.bar(x - w, agg["es_recorded"], w, label="recorded", color="tab:blue")
    ax.bar(x, agg["es_latent"], w, label="latent (bridge)", color="tab:green")
    ax.bar(x + w, agg["es_measurement"], w, label="measurement", color="tab:red")
    ax.set_xticks(x, [names[int(c)] for c in cats], rotation=20, fontsize=8)
    ax.set_ylabel("mean effect, ell 0-5 (log points)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
