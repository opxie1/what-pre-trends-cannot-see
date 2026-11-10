"""Helpers for the Medicaid expansion (BRFSS) study: panel build from cached
CDC Socrata extracts, ACA expansion cohorts, discretion grading, state-cluster
bootstraps and figures. Generic CS machinery (cs_es_fast) comes from crime_lib
and is runtime-verified against did_measurement.estimators.att_gt_cs.
Estimation uses only observed columns (unit, period, cohort, cat, d, D,
event_time, Y, ylog, n_sample)."""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crime_lib import cs_es_fast  # generic; verified vs library at runtime
from did_measurement.bridge import estimate_relative_loadings, _residual_wide

# ---------------------------------------------------------------- states

STATE_ABBR = dict(p.split(":") for p in (
    "Alabama:AL Alaska:AK Arizona:AZ Arkansas:AR California:CA Colorado:CO "
    "Connecticut:CT Delaware:DE Florida:FL Georgia:GA Hawaii:HI Idaho:ID "
    "Illinois:IL Indiana:IN Iowa:IA Kansas:KS Kentucky:KY Louisiana:LA "
    "Maine:ME Maryland:MD Massachusetts:MA Michigan:MI Minnesota:MN "
    "Mississippi:MS Missouri:MO Montana:MT Nebraska:NE Nevada:NV "
    "Ohio:OH Oklahoma:OK Oregon:OR Pennsylvania:PA Vermont:VT Texas:TX "
    "Tennessee:TN Utah:UT Virginia:VA Washington:WA Wisconsin:WI Wyoming:WY"
).split())
STATE_ABBR.update({"District of Columbia": "DC", "New Hampshire": "NH",
                   "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
                   "North Carolina": "NC", "North Dakota": "ND",
                   "Rhode Island": "RI", "South Carolina": "SC",
                   "South Dakota": "SD", "West Virginia": "WV"})
ABBRS = sorted(STATE_ABBR.values())
assert len(ABBRS) == 51, len(ABBRS)

# ---------------------------------------------- ACA Medicaid expansion dates
# Standard KFF adoption table (coverage effective dates), verified against the
# Wikipedia "Medicaid expansion" snapshot and healthinsurance.org state pages
# cached in data/raw/medicaid/sources/ (see data_manifest.json).

EXPANSION_EFFECTIVE = dict(p.split("=") for p in (
    "AZ=2014-01-01 AR=2014-01-01 CA=2014-01-01 CO=2014-01-01 CT=2014-01-01 "
    "DE=2014-01-01 DC=2014-01-01 HI=2014-01-01 IL=2014-01-01 IA=2014-01-01 "
    "KY=2014-01-01 MD=2014-01-01 MA=2014-01-01 MI=2014-04-01 MN=2014-01-01 "
    "NV=2014-01-01 NJ=2014-01-01 NM=2014-01-01 NY=2014-01-01 ND=2014-01-01 "
    "OH=2014-01-01 OR=2014-01-01 RI=2014-01-01 VT=2014-01-01 WA=2014-01-01 "
    "WV=2014-01-01 NH=2014-08-15 PA=2015-01-01 IN=2015-02-01 AK=2015-09-01 "
    "MT=2016-01-01 LA=2016-07-01 VA=2019-01-01 ME=2019-01-10 ID=2020-01-01 "
    "UT=2020-01-01 NE=2020-10-01 OK=2021-07-01 MO=2021-10-01 SD=2023-07-01 "
    "NC=2023-12-01").split())
assert len(EXPANSION_EFFECTIVE) == 41, len(EXPANSION_EFFECTIVE)
# no ACA expansion through 2024 (WI covers childless adults to 100% FPL
# outside the ACA expansion; classified never)
NEVER_STATES = ["AL", "FL", "GA", "KS", "MS", "SC", "TN", "TX", "WI", "WY"]
EARLY_EXPANDERS = ["CA", "CT", "DC", "MN", "NJ", "WA"]  # partial 1115, 2010-11

COHORT_RULE = ("cohort = calendar year of the coverage effective date when "
               "coverage began on or before July 1 (>= 6 months treated in "
               "that year), else the next calendar year")


def cohort_of(abbr):
    if abbr in NEVER_STATES:
        return -1
    y, m, d = (int(x) for x in EXPANSION_EFFECTIVE[abbr].split("-"))
    return y if (m, d) <= (7, 1) else y + 1


# ---------------------------------------------------------------- categories
# cat -> (name, discretion d, role, raw file key)
CATS = {
    0: ("diabetes_mortality", 0.00, "bridge", "nchs"),
    1: ("diabetes_diagnosed", 1.00, "focal", "diabetes"),
    2: ("hypertension_diagnosed", 0.90, "focal", "hypertension"),
    3: ("cholesterol_diagnosed", 1.00, "focal", "cholesterol"),
    4: ("depression_diagnosed", 0.80, "focal", "depression"),
    5: ("obesity_bmi_selfreport", 0.05, "focal", "bmi"),
    6: ("current_smoking", 0.00, "focal", "smoking"),
    7: ("no_exercise_past_month", 0.00, "focal", "exercise"),
    8: ("insured_18_64", np.nan, "aux", "coverage"),
    9: ("checkup_past_year", np.nan, "aux", "checkup"),
}
NAMES = {c: nm for c, (nm, _, _, _) in CATS.items()}
D_IDX = {c: d for c, (_, d, _, _) in CATS.items()}
ROLES = {c: r for c, (_, _, r, _) in CATS.items()}

# Static provenance prose (sources, discretion rationale, caveats, rejected
# bridge alternatives) lives beside the cached extracts it documents.
_DOC = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "raw", "medicaid",
    "manifest_text.json")))
D_RATIONALE = _DOC["discretion_rationale"]
TXT = _DOC["summary_text"]
MANIFEST_TEXT = {k: v for k, v in _DOC.items()
                 if k not in ("discretion_rationale", "r0_diabetes_source",
                              "_note", "summary_text")}
R0_SOURCE = _DOC["r0_diabetes_source"]


def _select_rows(key, rows):
    """Pick the relevant response rows for one BRFSS topic extract."""
    if key == "bmi":
        return [r for r in rows if r["response"].lower().startswith("obese")]
    # coverage: question redesigned in 2021, response label renamed
    ok = {"exercise": ("No",), "checkup": ("Within the past year",),
          "coverage": ("Yes", "Have some form of health insurance")}.get(
              key, ("Yes",))
    return [r for r in rows if r["response"] in ok]


def build_panel(raw_dir, out_csv):
    """State x year x cat long panel. Y = crude prevalence (pp) for BRFSS cats;
    age-adjusted deaths per 100k for the mortality bridge."""
    code = {a: i for i, a in enumerate(ABBRS)}
    cohort = {a: cohort_of(a) for a in ABBRS}
    recs, missing = [], []
    for c, (name, d, role, key) in CATS.items():
        cells = {}
        if key == "nchs":
            for r in json.load(open(os.path.join(
                    raw_dir, "nchs_diabetes_deaths.json"))):
                ab, y = STATE_ABBR.get(r["state"]), int(r["year"])
                if ab is not None and 2011 <= y <= 2017:
                    cells[(ab, y)] = (float(r["aadr"]), float(r["deaths"]))
            years = list(range(2011, 2018))
        else:
            rows = json.load(open(os.path.join(raw_dir, f"brfss_{key}.json")))
            for r in _select_rows(key, rows):
                ab = r.get("locationabbr")
                if ab not in code or "data_value" not in r:
                    continue
                k = (ab, int(r["year"]))
                assert k not in cells, f"duplicate cell {name} {k}"
                cells[k] = (float(r["data_value"]),
                            float(r.get("sample_size", "nan") or "nan"))
            years = sorted({y for _, y in cells})
        for ab in ABBRS:
            for y in years:
                if (ab, y) in cells:
                    v, n = cells[(ab, y)]
                    recs.append((code[ab], ab, y, cohort[ab], c, name, d,
                                 role, v, n))
                else:
                    missing.append((name, ab, y))
    df = pd.DataFrame(recs, columns=["unit", "abbr", "period", "cohort",
                                     "cat", "cat_name", "d", "role",
                                     "Y", "n_sample"])
    g = df["cohort"].to_numpy()
    y = df["period"].to_numpy()
    df["D"] = ((g > 0) & (y >= g)).astype(int)
    df["event_time"] = np.where(g > 0, y - g, np.nan)
    df["ylog"] = np.log(df["Y"])
    df = df.sort_values(["cat", "unit", "period"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df, missing


# ------------------------------------------------- state resampling

def make_resampler(df):
    """Return (units, draw); draw(samp) rebuilds the panel from a state sample,
    relabelling units 0..n-1 so repeat draws count as distinct clusters.
    Row-index take, not copy+concat: 10x faster, byte-identical output."""
    u_arr = df["unit"].to_numpy()
    units = np.sort(df["unit"].unique())
    rows = {u: np.flatnonzero(u_arr == u) for u in units}
    cnt = {u: len(rows[u]) for u in units}
    ucol = list(df.columns).index("unit")

    def draw(samp):
        out = df.take(np.concatenate([rows[u] for u in samp]))
        out = out.reset_index(drop=True)
        out.isetitem(ucol, np.repeat(np.arange(len(samp)),
                                     [cnt[u] for u in samp]))
        return out

    return units, draw


# ------------------------------------------------- single-series bootstrap

def boot_postmean(sub, outcome="Y", control="notyet", B=500, seed=0,
                  post_window=(0, 5), min_n=20):
    """State bootstrap of one category's CS post-window mean effect.
    Returns (postmean, se, n_draws)."""
    lo, hi = post_window
    pm = _postmean(cs_es_fast(sub, outcome, control), lo, hi)
    rng = np.random.default_rng(seed)
    units, draw = make_resampler(sub)
    draws = []
    for _ in range(B):
        samp = rng.choice(units, size=len(units), replace=True)
        v = _postmean(cs_es_fast(draw(samp), outcome, control), lo, hi)
        if np.isfinite(v):
            draws.append(v)
    se = float(np.std(draws, ddof=1)) if len(draws) > min_n else np.nan
    return pm, se, len(draws)


# ------------------------------------------------- unified state bootstrap

def residual_corr(df, cats, outcome="Y"):
    """Untreated (D==0) comovement after unit-by-cat and period-by-cat FE: the
    variation bridge loadings rho_c = cov(res_c,res_k)/cov(res_bridge,res_k)
    are read off. Near-zero bridge covariances mean rho is a ratio of noise,
    i.e. the bridge is not identified. Returns (corr, n_pairwise)."""
    w = _residual_wide(df[df["cat"].isin(cats)], outcome)
    w.columns = [NAMES[int(c)] for c in w.columns]
    corr = w.corr()
    n = pd.DataFrame(
        [[int((w[a].notna() & w[b].notna()).sum()) for b in w.columns]
         for a in w.columns], index=w.columns, columns=w.columns)
    return corr, n


def _postmean(es, lo, hi, parity=None):
    """Mean event-study effect over [lo,hi]; parity=1 keeps odd event times
    only, the common support shared with the odd-year-only questions."""
    v = [x for l, x in es.items() if lo <= l <= hi and np.isfinite(x)
         and (parity is None or l % 2 == parity)]
    return float(np.mean(v)) if v else np.nan


def _share(es_c, es_br, rho_c, lo, hi):
    """Measurement share over [lo,hi]: mean(rec - rho*bridge)/mean(rec)."""
    ells = [l for l in es_c if lo <= l <= hi and np.isfinite(es_c[l])
            and np.isfinite(es_br.get(l, np.nan))]
    if not ells or not np.isfinite(rho_c):
        return np.nan, np.nan, np.nan
    rec = float(np.mean([es_c[l] for l in ells]))
    lat = float(np.mean([rho_c * es_br[l] for l in ells]))
    sh = (rec - lat) / rec if abs(rec) > 1e-9 else np.nan
    return rec, lat, sh


def boot_unified(df, cats_all, bridge_set, focal, diagnosed, d_idx,
                 outcome="Y", B=400, seed=7, bridge_cat=0,
                 post_window=(0, 5), bridge_window=(0, 3), min_n=25):
    """One state-cluster bootstrap feeding every deliverable (not-yet CS).
    Per draw: CS event studies for all cats, post-window means, the gradient
    slope of focal post-means on d_c, IV loadings vs `bridge_cat`, bridge
    decomposition and measurement shares. Returns tables + draws."""
    lo, hi = post_window
    blo, bhi = bridge_window

    def _es_all(d):
        return {c: cs_es_fast(d[d["cat"] == c], outcome) for c in cats_all}

    point = _es_all(df)
    ld = estimate_relative_loadings(df[df["cat"].isin(bridge_set)],
                                    outcome=outcome, bridge_cat=bridge_cat)
    rho_pt, min_denom = ld["rho"], ld["min_denom"]
    # parity None = all event times; parity 1 = odd only, the common support
    # shared with the odd-year questions (hypertension, cholesterol)
    PAR = (None, 1)
    pm_pt = {p: {c: _postmean(point[c], lo, hi, p) for c in cats_all}
             for p in PAR}
    xs = np.array([d_idx[c] for c in focal])
    slope_pt = {p: float(np.polyfit(xs, [pm_pt[p][c] for c in focal], 1)[0])
                for p in PAR}

    rng = np.random.default_rng(seed)
    units, draw = make_resampler(df)
    es_draws = {c: {} for c in cats_all}
    pm_draws = {p: {c: [] for c in cats_all} for p in PAR}
    slope_draws = {p: [] for p in PAR}
    rho_draws = {c: [] for c in rho_pt}
    dec_draws, share_draws = {}, {c: [] for c in diagnosed}
    for _ in range(B):
        samp = rng.choice(units, size=len(units), replace=True)
        dfb = draw(samp)
        es_b = _es_all(dfb)
        for c in cats_all:
            for l, v in es_b[c].items():
                if np.isfinite(v):
                    es_draws[c].setdefault(l, []).append(v)
        for p in PAR:
            pmp = {c: _postmean(es_b[c], lo, hi, p) for c in cats_all}
            for c in cats_all:
                if np.isfinite(pmp[c]):
                    pm_draws[p][c].append(pmp[c])
            ys = np.array([pmp[c] for c in focal])
            if np.all(np.isfinite(ys)):
                slope_draws[p].append(float(np.polyfit(xs, ys, 1)[0]))
        try:
            ld_b = estimate_relative_loadings(
                dfb[dfb["cat"].isin(bridge_set)], outcome=outcome,
                bridge_cat=bridge_cat)
        except (ValueError, KeyError, np.linalg.LinAlgError):
            continue
        for c, v in ld_b["rho"].items():
            if np.isfinite(v):
                rho_draws[c].append(v)
        for c in focal:
            rho_c = ld_b["rho"].get(c, np.nan)
            for l, v in es_b[c].items():
                lat = rho_c * es_b[bridge_cat].get(l, np.nan)
                if np.isfinite(v) and np.isfinite(lat):
                    dec_draws.setdefault((c, l), []).append((v, lat, v - lat))
            if c in diagnosed:
                _, _, sh = _share(es_b[c], es_b[bridge_cat], rho_c, blo, bhi)
                if np.isfinite(sh):
                    share_draws[c].append(sh)

    def _se(v):
        return float(np.std(v, ddof=1)) if len(v) > min_n else np.nan

    es_tab = pd.DataFrame(
        [(c, l, point[c][l], _se(es_draws[c].get(l, [])),
          len(es_draws[c].get(l, [])))
         for c in cats_all for l in sorted(point[c])],
        columns=["cat", "ell", "coef", "se", "n_boot"])
    def _pm_tab(p):
        return pd.DataFrame(
            [(c, pm_pt[p][c], _se(pm_draws[p][c]), len(pm_draws[p][c]))
             for c in cats_all],
            columns=["cat", "post_mean", "se", "n_boot"])
    rho_tab = pd.DataFrame(
        [(c, rho_pt[c], _se(rho_draws[c]), min_denom) for c in rho_pt],
        columns=["cat", "rho", "se", "min_denom"])
    def _dec_row(c, l):
        v = point[c][l]
        lat = rho_pt.get(c, np.nan) * point[bridge_cat].get(l, np.nan)
        dd = np.asarray(dec_draws.get((c, l), np.empty((0, 3))))
        s = dd.std(axis=0, ddof=1) if len(dd) > min_n else [np.nan] * 3
        return (c, l, v, lat, v - lat, s[0], s[1], s[2])

    dec_tab = pd.DataFrame(
        [_dec_row(c, l) for c in focal for l in sorted(point[c])], columns=[
            "cat", "ell", "es_recorded", "es_latent", "es_measurement",
            "se_es_recorded", "se_es_latent", "se_es_measurement"])

    def _sh_row(c):
        rec, lat, sh = _share(point[c], point[bridge_cat],
                              rho_pt.get(c, np.nan), blo, bhi)
        d = np.asarray(share_draws[c])
        q = np.percentile(d, [2.5, 50, 97.5]) if len(d) > min_n else [np.nan] * 3
        return (c, rec, lat, sh, q[0], q[1], q[2], len(d))

    share_tab = pd.DataFrame([_sh_row(c) for c in diagnosed], columns=[
        "cat", "recorded_mean", "latent_mean", "share_point",
        "share_p2_5", "share_p50", "share_p97_5", "n_draws"])
    def _slope(p):
        sd = np.asarray(slope_draws[p])
        ok = len(sd) > min_n
        return {"point": slope_pt[p], "se": _se(slope_draws[p]),
                "p2_5": float(np.percentile(sd, 2.5)) if ok else np.nan,
                "p97_5": float(np.percentile(sd, 97.5)) if ok else np.nan,
                "n_draws": len(sd)}

    return {"es": es_tab, "pm": _pm_tab(None), "pm_odd": _pm_tab(1),
            "rho": rho_tab, "dec": dec_tab, "share": share_tab,
            "slope": _slope(None), "slope_odd": _slope(1), "B": B}


# ---------------------------------------------------------------- figures

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _es_ax(ax, tab, color, label=None, xlim=(-8, 10)):
    e = tab.dropna(subset=["coef"])
    e = e[(e["ell"] >= xlim[0]) & (e["ell"] <= xlim[1])]
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(-0.5, color="0.8", lw=0.8, ls="--")
    ax.errorbar(e["ell"], e["coef"],
                yerr=np.where(np.isfinite(e["se"]), 1.96 * e["se"], 0.0),
                fmt="o", ms=3.5, lw=1, capsize=2, color=color, label=label)


def fig_es(es_cat, twfe_cat, name, ylab, path, xlim=(-8, 10)):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _es_ax(ax, es_cat, "tab:blue", "CS (not-yet control)", xlim)
    if twfe_cat is not None:
        t = twfe_cat[(twfe_cat["ell"] >= xlim[0]) &
                     (twfe_cat["ell"] <= xlim[1])]
        ax.plot(t["ell"], t["coef"], "s--", ms=3, lw=0.8,
                color="tab:orange", alpha=0.8, label="TWFE (binned)")
    ax.set_xlabel("years since Medicaid expansion")
    ax.set_ylabel(ylab)
    ax.set_title(name)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_gradient(pm, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0, color="0.6", lw=0.8)
    seen = {}   # stagger labels: several categories share d = 0 and d = 1
    for _, r in pm[pm["cat"].map(ROLES) != "aux"].iterrows():
        c, x = int(r["cat"]), D_IDX[int(r["cat"])]
        k = seen.get(x, 0)
        seen[x] = k + 1
        ax.errorbar([x], [r["post_mean"]], yerr=[1.96 * r["se"]], fmt="o",
                    color="tab:red" if c == 0 else "tab:blue", ms=5, capsize=3)
        left = x < 0.5
        ax.annotate(NAMES[c], (x, r["post_mean"]), fontsize=7,
                    ha="left" if left else "right",
                    textcoords="offset points",
                    xytext=(7 if left else -7, 6 + 11 * k))
    foc = pm[pm["cat"].map(ROLES) == "focal"]
    b = np.polyfit(np.array([D_IDX[int(c)] for c in foc["cat"]]),
                   foc["post_mean"].to_numpy(), 1)
    xx = np.linspace(-0.02, 1.05, 20)
    ax.plot(xx, np.polyval(b, xx), "-", color="0.4", lw=1,
            label=f"focal fit: slope={b[0]:.3f} pp per unit d")
    ax.set_xlabel("diagnosis-dependence index d_c")
    ax.set_ylabel("mean effect, ell 0-5 (pp; bridge: deaths/100k)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_power(tab, path):
    """What the design can detect vs what the recording channel can deliver."""
    t = tab[tab["d"] > 0].sort_values("d", ascending=False)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - 0.2, t["mde80_pp"], 0.4, color="0.6",
           label="minimum detectable effect (80% power, 5%)")
    ax.bar(x + 0.2, t["implied_pp"], 0.4, color="tab:red",
           label="mechanism upper bound (all newly insured diagnosed)")
    ax.set_yscale("log")
    ax.set_xticks(x, [n.replace("_", "\n") for n in t["cat_name"]], fontsize=7)
    ax.set_ylabel("effect size (pp)")
    ax.set_title("what the design can detect vs what recording can deliver",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_first_stage(cov_tab, chk_tab, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for ax, tab, ttl in [(axes[0], cov_tab, "insured, adults 18-64"),
                         (axes[1], chk_tab, "routine checkup past year")]:
        _es_ax(ax, tab, "tab:green")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("years since expansion")
    axes[0].set_ylabel("CS effect (pp)")
    _save(fig, path)


def fig_sensitivity(sens, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="0.6", lw=0.8)
    k = sens["kappa_bar"]
    ax.fill_between(k, sens["L"], sens["U"], alpha=0.35, color="tab:blue",
                    label="identified set")
    ax.plot(k, sens["ci_lo"], "--", color="tab:blue", lw=1)
    ax.plot(k, sens["ci_hi"], "--", color="tab:blue", lw=1,
            label="Imbens-Manski 95% CI")
    ax.set_xlabel("kappa_bar: max recording response (pp at d=1)")
    ax.set_ylabel("diabetes diagnosed, mean effect ell 0-5 (pp)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_bridge(dec, path, post_window=(0, 3), bridge_name="mortality"):
    t = dec[(dec["ell"] >= post_window[0]) & (dec["ell"] <= post_window[1])
            & np.isfinite(dec["es_latent"])]
    agg = t.groupby("cat")[["es_recorded", "es_latent",
                            "es_measurement"]].mean()
    x, w = np.arange(len(agg)), 0.27
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0, color="0.6", lw=0.8)
    for dx, col, lab in [(-w, "es_recorded", "recorded"),
                         (0, "es_latent", f"latent ({bridge_name} bridge)"),
                         (w, "es_measurement", "measurement")]:
        ax.bar(x + dx, agg[col], w, label=lab)
    ax.set_xticks(x, [NAMES[int(c)] for c in agg.index], rotation=20,
                  fontsize=7)
    ax.set_ylabel(f"mean effect, ell {post_window[0]}-{post_window[1]} (pp)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_comovement(corr, path):
    """Untreated residual correlations: the bridge's identifying variation."""
    m, lab = corr.to_numpy(), [c.replace("_", " ") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="0.15")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("untreated residual comovement (unit-x-cat, period-x-cat FE)",
                 fontsize=9)
    _save(fig, path)
