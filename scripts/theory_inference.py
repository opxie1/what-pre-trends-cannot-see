"""Verification for theory/inference.md (T4: exact few-cluster inference).

Reproduces the size failure of the chi-square discretion-gradient Wald test and
demonstrates the fixes:

  A  size      G in {20,30,51,100} x C in {3,5,8}, kappa = 0, 2000 reps/cell
  B  power     G = 51, C in {3,5,8}, kappa grid, raw and size-adjusted
  C  design    discretion spread (narrow vs wide) x T in {60,120}
  E  reference single omitted pre-period (l = -1) vs the whole pre-block
  V  exactness deterministic checks: the category-level reduction reproduces
               mtest.discretion_gradient_test to machine precision, and the fast
               permutation distribution equals brute-force relabel-and-refit
  M  MDE       closed-form kappa_MDE vs the simulated size-adjusted MDE

Run repeatedly until it prints COMPLETE (per-cell checkpoints resume):
    cd ruler-did && PYTHONPATH=src python scripts/theory_inference.py
Env: TI_BUDGET_S (default 100000), TI_WORKERS, TI_SMOKE=1.
Outputs -> results/inference/.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from did_measurement.dgp import DGPConfig, generate_panel
from did_measurement.mtest import discretion_gradient_test
from did_measurement.estimators import build_event_dummies, demean_two_way
from did_measurement import inference as inf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):                 # older working tree layout
    OUT = os.path.join(ROOT, "results", "theory", "inference")
SMOKE = os.environ.get("TI_SMOKE", "") == "1"
if SMOKE:
    OUT = OUT + "_smoke"
CKPT = os.path.join(OUT, "checkpoints")

ALPHA = 0.05
K_PRE, K_POST = 6, 12
N_PERM, N_BOOT, N_SIGN = 999, 499, 999
METHODS = ("chi2", "f", "perm_wald", "perm_dir", "perm_dirgls", "perm_chibar",
           "wcb_wald", "wcb_dir", "signflip", "signflip_dir")
LABELS = {
    "chi2": r"Wald, $\chi^2_q$ (status quo)",
    "f": r"Wald/q, $F(q,G-1)$",
    "perm_wald": "label perm., Wald",
    "perm_dir": "label perm., directed",
    "perm_dirgls": "label perm., directed-GLS (LMPI)",
    "perm_chibar": r"label perm., $\bar\chi^2$",
    "wcb_wald": "wild cluster boot., Wald",
    "wcb_dir": "wild cluster boot., directed",
    "signflip": "cluster sign-flip, Wald",
    "signflip_dir": "cluster sign-flip, directed",
}
REPS = {"size": 2000, "power": 600, "design": 500, "ref": 600}
if SMOKE:
    REPS = {k: 8 for k in REPS}


# ------------------------------------------------------------------ DGP
def make_cfg(G=51, T=120, C=5, kappa=0.0, disc=(0.1, 1.0), seed=0):
    cohorts = (48, 72, 96) if T >= 120 else (T * 2 // 5, T * 3 // 5, T * 4 // 5)
    d = tuple(np.round(np.linspace(disc[0], disc[1], C), 6))
    return DGPConfig(n_units=G, n_periods=T, cohorts=cohorts,
                     cohort_shares=(0.2, 0.2, 0.2), n_categories=C,
                     include_bridge=True, discretion=d, theta_spread=0.3,
                     tau_level=0.4, tau_shape="constant", kappa=kappa,
                     kappa_shape="constant", map_type="poisson", mu_scale=0.5,
                     r_base=0.6, exposure=500_000.0, latent_scale=0.25, seed=seed)


def panel(cfg):
    df = generate_panel(cfg)
    df.attrs = {}
    df["logy"] = np.log((df["Y"] + 0.5) / df["exposure"])
    return df


def run_methods(comp, seed, methods=METHODS):
    """p-values for every method from one component object."""
    C = comp["C"]
    out = {}
    for m in methods:
        if m == "chi2":
            r = inf.chi2_wald(comp)
        elif m == "f":
            r = inf.f_wald(comp)
        elif m == "perm_wald":
            r = inf.permutation_test(comp, "wald", N_PERM, seed)
        elif m == "perm_dir":
            r = inf.permutation_test(comp, "directed", N_PERM, seed)
        elif m == "perm_dirgls":
            r = inf.permutation_test(comp, "directed_gls", N_PERM, seed)
        elif m == "perm_chibar":
            r = inf.permutation_test(comp, "chibar", 199 if C >= 8 else N_PERM, seed)
        elif m == "wcb_wald":
            r = inf.wild_cluster_bootstrap(comp, N_BOOT, seed, "wald")
        elif m == "wcb_dir":
            r = inf.wild_cluster_bootstrap(comp, N_BOOT, seed, "directed")
        elif m == "signflip":
            r = inf.signflip_test(comp, N_SIGN, seed, "wald")
        elif m == "signflip_dir":
            r = inf.signflip_test(comp, N_SIGN, seed, "directed")
        else:
            raise ValueError(m)
        out["p_" + m] = float(r["pvalue"])
    return out


# ------------------------------------------------------------------ job grid
def build_jobs():
    jobs = []

    def add(part, key, meta, reps, **cfg):
        jobs.append(dict(part=part, key=key, meta=meta, reps=reps, cfg=cfg))

    for G in (20, 30, 51, 100):
        for C in (3, 5, 8):
            add("size", f"size_G{G}_C{C}", dict(G=G, C=C, T=120, kappa=0.0,
                                                spread="wide", k_pre=K_PRE),
                REPS["size"], G=G, T=120, C=C, kappa=0.0, disc=(0.1, 1.0))

    for C in (3, 5, 8):
        for kap in (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80):
            add("power", f"pow_C{C}_k{kap}", dict(G=51, C=C, T=120, kappa=kap,
                                                  spread="wide", k_pre=K_PRE),
                REPS["power"], G=51, T=120, C=C, kappa=kap, disc=(0.1, 1.0))

    for T in (60, 120):
        for sname, dd in (("narrow", (0.4, 0.6)), ("wide", (0.1, 1.0))):
            for kap in (0.0, 0.15, 0.30, 0.50, 0.80):
                add("design", f"des_T{T}_{sname}_k{kap}",
                    dict(G=51, C=5, T=T, kappa=kap, spread=sname, k_pre=K_PRE),
                    REPS["design"], G=51, T=T, C=5, kappa=kap, disc=dd)

    for kp in (1, K_PRE):
        for kap in (0.0, 0.10, 0.20, 0.30):
            add("ref", f"ref_kpre{kp}_k{kap}",
                dict(G=51, C=5, T=120, kappa=kap, spread="wide", k_pre=kp),
                REPS["ref"], G=51, T=120, C=5, kappa=kap, disc=(0.1, 1.0),
                k_pre=kp)
    return jobs


def run_chunk(args):
    job, r0, r1 = args
    cfgkw = dict(job["cfg"])
    k_pre = cfgkw.pop("k_pre", K_PRE)
    rows = []
    for rep in range(r0, r1):
        seed = zlib.crc32(f"ti:{job['key']}:{rep}".encode()) & 0x7FFFFFFF
        row = dict(job["meta"]); row.update(rep=rep, seed=seed, err=0)
        try:
            df = panel(make_cfg(seed=seed, **cfgkw))
            comp = inf.gradient_components(df, outcome="logy", k_pre=k_pre,
                                           k_post=K_POST, loading="triad")
            row.update(run_methods(comp, seed))
            row["sigma2"] = float(np.mean(comp["resid_var"]))
            row["sigma2_w"] = float((comp["delta"] ** 2 @ comp["resid_var"])
                                    / (comp["delta"] ** 2).sum())
            row["neg_cov"] = comp["neg_cov_share"]
        except Exception:
            row["err"] = 1
        rows.append(row)
    return rows


# ------------------------------------------------------------------ analytics
def design_objects(G=51, T=120, C=5, disc=(0.1, 1.0), k_pre=K_PRE, seed=7):
    """Design-only objects (Sigma_delta2, I_k, q) plus a sigma^2 estimate."""
    df = panel(make_cfg(G=G, T=T, C=C, kappa=0.0, disc=disc, seed=seed))
    comp = inf.gradient_components(df, outcome="logy", k_pre=k_pre,
                                   k_post=K_POST, loading="triad")
    di = inf.design_information(comp)
    return comp, di


def closed_form_mde(comp, sigma2, kind, alpha=ALPHA, power=0.80, nonlinear=True):
    wedge = (lambda kap: np.log1p(kap * comp["d"])) if nonlinear else None
    return inf.mde_kappa(comp, kind=kind, alpha=alpha, power=power,
                         sigma2=sigma2, wedge=wedge)["kappa"]


def canonical_information(N=1000, T=24, g=12, p=0.5, k_pre=None, k_post=None):
    """I_k for the single-cohort canonical design, numerically, to check Lemma 8."""
    n_tr = int(round(p * N))
    unit = np.repeat(np.arange(N), T)
    period = np.tile(np.arange(T), N)
    coh = np.where(np.repeat(np.arange(N), T) < n_tr, g, -1)
    ev = np.where(coh >= 0, period - g, np.nan)
    kp = g if k_pre is None else k_pre
    kq = T - g - 1 if k_post is None else k_post
    X, labels = build_event_dummies(ev, kp, kq)
    Xt = demean_two_way(X, unit, period)
    S = Xt.T @ Xt
    P = np.array([j for j, l in enumerate(labels) if l >= 0])
    W = np.linalg.pinv(np.linalg.pinv(S)[np.ix_(P, P)])
    k = np.ones(len(P))
    return float(k @ W @ k), len(P), N, p


# ------------------------------------------------------------------ aggregate
def _rej(p, cut=ALPHA):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan, np.nan, 0
    r = float((p <= cut).mean())
    return r, float(np.sqrt(max(r * (1 - r), 1e-12) / len(p))), len(p)


def _load(part, jobs):
    fr = [pd.read_csv(os.path.join(CKPT, j["key"] + ".csv"))
          for j in jobs if j["part"] == part]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def aggregate(jobs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = {"alpha": ALPHA, "reps": REPS, "n_perm": N_PERM, "n_boot": N_BOOT,
         "n_signflip": N_SIGN, "k_pre": K_PRE, "k_post": K_POST,
         "design": ("51 states x T=120 months, cohorts (48,72,96) at 20% each, "
                    "40% never treated, Poisson recording map, outcome "
                    "log((Y+0.5)/exposure); matches results/sims/e_calibrated"),
         "caveats": []}

    # ---------------- V. exactness checks -------------------------------
    df = panel(make_cfg(G=51, T=60, C=5, kappa=0.0, seed=3))
    c_iv = inf.gradient_components(df, outcome="logy", k_pre=K_PRE,
                                   k_post=K_POST, loading="mtest_iv")
    w_new = inf.chi2_wald(c_iv)["stat"]
    w_old = discretion_gradient_test(df, outcome="logy", k_pre=K_PRE,
                                     k_post=K_POST)["stat"]
    comp = inf.gradient_components(df, outcome="logy", k_pre=K_PRE,
                                   k_post=K_POST, loading="triad")
    dvals = np.round(np.linspace(0.1, 1.0, 5), 6)
    brute = []
    for perm in itertools.permutations(range(5)):
        d2 = df["cat"].map({0: 0.0, **{c + 1: dvals[perm[c]] for c in range(5)}})
        dd = df.copy(); dd["d"] = d2.astype(float)
        brute.append(inf.chi2_wald(inf.gradient_components(
            dd, outcome="logy", k_pre=K_PRE, k_post=K_POST, loading="triad"))["stat"])
    D, _ = inf._perm_matrix(comp["delta"], None, np.random.default_rng(0))
    fast = inf._apply_stat(comp, D, "wald")
    # Definition 4 / Lemma 2: at the identity Rademacher draw (v == 1) the
    # bootstrap sample IS the data, so z*_g = z_g and the bootstrap Wald must
    # equal chi2_wald exactly.  Likewise sum_g zeta_g = gamma_hat_post (eq. 6).
    w_obs = inf.chi2_wald(comp)["stat"]
    w_wcb = inf.wild_cluster_bootstrap(comp, 3, 0, "wald")["stat"]
    _r = inf._restricted(comp)
    _zeta = (_r["Zr"] @ comp["Sinv"][:, comp["post"]]) / _r["s2"]
    _gam = (comp["B"] @ comp["delta"] / (comp["delta"] ** 2).sum())[comp["post"]]
    S["exactness_checks"] = {
        "reduction_vs_mtest_abs_diff": float(abs(w_new - w_old)),
        "reduction_vs_mtest_rel_diff": float(abs(w_new - w_old) / max(abs(w_old), 1e-12)),
        "permutation_vs_brute_refit_max_abs_diff":
            float(np.abs(np.sort(np.array(brute)) - np.sort(fast)).max()),
        "wcb_identity_draw_vs_wald_abs_diff": float(abs(w_wcb - w_obs)),
        "wcb_identity_draw_vs_wald_rel_diff":
            float(abs(w_wcb - w_obs) / max(abs(w_obs), 1e-12)),
        "signflip_zeta_sum_vs_gamma_max_abs_diff":
            float(np.abs(_zeta.sum(0) - _gam).max()),
        "note": ("Lemma 1-2: the category-level reduction reproduces the pooled "
                 "gradient Wald statistic exactly; Lemma 3: permuting the labels "
                 "in the fast representation equals relabelling d in the data and "
                 "refitting (loadings re-estimated) to machine precision; "
                 "Definition 4: the wild-cluster-bootstrap statistic at the "
                 "identity draw v == 1 reproduces the observed Wald exactly; "
                 "equation (6): the cluster contributions zeta_g sum to "
                 "gamma_hat on the post block."),
    }

    # ---------------- A. size grid --------------------------------------
    d = _load("size", jobs)
    rows = []
    for (G, C), g in d.groupby(["G", "C"]):
        for m in METHODS:
            r, se, n = _rej(g["p_" + m])
            rows.append(dict(G=G, C=C, method=m, reps=n, size=r, mcse=se,
                             abs_dev_pp=100 * abs(r - ALPHA)))
    size_tab = pd.DataFrame(rows)
    size_tab.to_csv(os.path.join(OUT, "size_table.csv"), index=False)
    S["size"] = {f"G{int(G)}_C{int(C)}_{m}": round(r, 4) for G, C, m, r
                 in size_tab[["G", "C", "method", "size"]].itertuples(index=False)}
    hdl = size_tab[(size_tab.G == 51) & (size_tab.C == 5)].set_index("method")["size"]
    S["headline_G51_C5"] = {m: round(float(hdl[m]), 4) for m in METHODS}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, C in zip(axes, (3, 5, 8)):
        sub = size_tab[size_tab.C == C]
        for m in METHODS:
            g = sub[sub.method == m].sort_values("G")
            ls = "-" if m.startswith(("perm", "signflip")) else "--"
            ax.plot(g.G, g["size"], ls, marker="o", ms=3.5, lw=1.3, label=LABELS[m])
        ax.axhline(ALPHA, color="k", lw=0.9, ls=":")
        ax.set_xlabel("clusters $G$")
        ax.set_title(f"$C = {C}$ categories", fontsize=10)
        ax.set_ylim(0, 0.55)
    axes[0].set_ylabel("rejection rate under $H_0$ (nominal 5%)")
    axes[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.suptitle("Size of the discretion-gradient test, $\\kappa = 0$, "
                 f"{REPS['size']} replications per cell", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "size_grid.png"), dpi=150)
    plt.close(fig)

    # null critical cutoffs (for size adjustment), by (G, C, method)
    cuts = {}
    for (G, C), g in d.groupby(["G", "C"]):
        for m in METHODS:
            p = g["p_" + m].dropna().to_numpy()
            cuts[(int(G), int(C), m)] = float(np.quantile(p, ALPHA, method="lower"))

    # ---------------- B. power ------------------------------------------
    d = _load("power", jobs)
    rows = []
    for (C, kap), g in d.groupby(["C", "kappa"]):
        for m in METHODS:
            r, se, n = _rej(g["p_" + m])
            ra, _, _ = _rej(g["p_" + m], cut=cuts[(51, int(C), m)])
            rows.append(dict(C=C, kappa=kap, method=m, reps=n, power=r, mcse=se,
                             power_sizeadj=ra))
    for (G, C), g in _load("size", jobs).groupby(["G", "C"]):
        if G != 51:
            continue
        for m in METHODS:
            r, se, n = _rej(g["p_" + m])
            rows.append(dict(C=C, kappa=0.0, method=m, reps=n, power=r, mcse=se,
                             power_sizeadj=ALPHA))
    pow_tab = pd.DataFrame(rows).sort_values(["C", "method", "kappa"])
    pow_tab.to_csv(os.path.join(OUT, "power_curves.csv"), index=False)
    S["power_sizeadj_G51"] = {
        f"C{int(C)}_k{k}_{m}": round(v, 4) for C, k, m, v in
        pow_tab[["C", "kappa", "method", "power_sizeadj"]].itertuples(index=False)}

    show = ("chi2", "perm_wald", "perm_dir", "perm_dirgls", "perm_chibar",
            "wcb_wald", "signflip", "signflip_dir")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, C in zip(axes, (3, 5, 8)):
        sub = pow_tab[pow_tab.C == C]
        for m in show:
            g = sub[sub.method == m].sort_values("kappa")
            ax.plot(g.kappa, g.power_sizeadj, marker="o", ms=3.5, lw=1.3,
                    label=LABELS[m])
        ax.axhline(ALPHA, color="k", lw=0.9, ls=":")
        ax.axhline(0.80, color="gray", lw=0.7, ls=":")
        ax.set_xlabel(r"measurement response $\kappa$")
        ax.set_title(f"$C = {C}$", fontsize=10)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("size-adjusted power")
    axes[2].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("Size-adjusted power, $G = 51$ clusters, $T = 120$", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "power_curves.png"), dpi=150)
    plt.close(fig)

    # ---------------- C/E. design and reference block --------------------
    d = _load("design", jobs)
    rows = []
    for (T, sp, kap), g in d.groupby(["T", "spread", "kappa"]):
        for m in METHODS:
            key = (T, sp, m)
            cut = np.quantile(
                d[(d["T"] == T) & (d["spread"] == sp) & (d["kappa"] == 0.0)]["p_" + m]
                .dropna().to_numpy(), ALPHA, method="lower")
            r, se, n = _rej(g["p_" + m])
            ra, _, _ = _rej(g["p_" + m], cut=cut)
            rows.append(dict(T=T, spread=sp, kappa=kap, method=m, reps=n,
                             power=r, mcse=se, power_sizeadj=ra))
    des_tab = pd.DataFrame(rows)
    des_tab.to_csv(os.path.join(OUT, "design_table.csv"), index=False)

    d = _load("ref", jobs)
    rows = []
    for (kp, kap), g in d.groupby(["k_pre", "kappa"]):
        for m in METHODS:
            cut = np.quantile(d[(d["k_pre"] == kp) & (d["kappa"] == 0.0)]["p_" + m]
                              .dropna().to_numpy(), ALPHA, method="lower")
            r, se, n = _rej(g["p_" + m])
            ra, _, _ = _rej(g["p_" + m], cut=cut)
            rows.append(dict(k_pre=kp, kappa=kap, method=m, reps=n, power=r,
                             mcse=se, power_sizeadj=ra))
    ref_tab = pd.DataFrame(rows)
    ref_tab.to_csv(os.path.join(OUT, "reference_block_table.csv"), index=False)
    S["reference_block"] = {
        f"kpre{int(kp)}_k{kap}_{m}": round(v, 4) for kp, kap, m, v in
        ref_tab[["k_pre", "kappa", "method", "power_sizeadj"]].itertuples(index=False)
        if m in ("perm_wald", "perm_dir", "signflip_dir")}

    # ---------------- M. closed-form MDE vs simulation -------------------
    sig_by = {}
    for part, keys in (("size", ["G", "C"]), ("design", ["T", "spread"]),
                       ("ref", ["k_pre"])):
        dd = _load(part, jobs)
        dd = dd[dd["kappa"] == 0.0] if "kappa" in dd else dd
        if len(dd):
            sig_by[part] = dd.groupby(keys)["sigma2_w"].mean()

    mde_rows = []
    configs = [("C3_T120_wide", dict(C=3, T=120, disc=(0.1, 1.0), k_pre=K_PRE),
                ("power", 3)),
               ("C5_T120_wide", dict(C=5, T=120, disc=(0.1, 1.0), k_pre=K_PRE),
                ("power", 5)),
               ("C8_T120_wide", dict(C=8, T=120, disc=(0.1, 1.0), k_pre=K_PRE),
                ("power", 8)),
               ("C5_T60_wide", dict(C=5, T=60, disc=(0.1, 1.0), k_pre=K_PRE),
                ("design", (60, "wide"))),
               ("C5_T60_narrow", dict(C=5, T=60, disc=(0.4, 0.6), k_pre=K_PRE),
                ("design", (60, "narrow"))),
               ("C5_T120_narrow", dict(C=5, T=120, disc=(0.4, 0.6), k_pre=K_PRE),
                ("design", (120, "narrow"))),
               ("C5_T120_wide_kpre1", dict(C=5, T=120, disc=(0.1, 1.0), k_pre=1),
                ("ref", 1))]
    for name, kw, (part, skey) in configs:
        comp, di = design_objects(G=51, **kw)
        if part == "size":
            s2 = float(sig_by["size"].loc[(51, kw["C"])])
        elif part == "power":
            s2 = float(sig_by["size"].loc[(51, skey)])
        elif part == "design":
            s2 = float(sig_by["design"].loc[skey])
        else:
            s2 = float(sig_by["ref"].loc[skey])
        row = dict(config=name, C=kw["C"], T=kw["T"], k_pre=kw["k_pre"],
                   sum_delta2=di["sum_delta2"], I_k=di["I_k"], q_post=len(comp["post"]),
                   sigma2=s2,
                   lambda_star_wald=inf.chi2_noncentrality_for_power(
                       len(comp["post"]), ALPHA, 0.80),
                   mde_wald_linear=closed_form_mde(comp, s2, "wald", nonlinear=False),
                   mde_wald=closed_form_mde(comp, s2, "wald"),
                   mde_directed=closed_form_mde(comp, s2, "directed"))
        mde_rows.append(row)
    mde_tab = pd.DataFrame(mde_rows)

    # simulated size-adjusted MDE by inverting the power curves
    def invert(ks, ps, target=0.80):
        ks, ps = np.asarray(ks, float), np.asarray(ps, float)
        o = np.argsort(ks); ks, ps = ks[o], ps[o]
        for i in range(1, len(ks)):
            if ps[i - 1] < target <= ps[i]:
                return float(ks[i - 1] + (target - ps[i - 1]) *
                             (ks[i] - ks[i - 1]) / (ps[i] - ps[i - 1]))
        return np.nan

    sim = {}
    for C in (3, 5, 8):
        for m in ("perm_wald", "perm_dir", "signflip", "signflip_dir"):
            g = pow_tab[(pow_tab.C == C) & (pow_tab.method == m)]
            sim[(f"C{C}_T120_wide", m)] = invert(g.kappa, g.power_sizeadj)
    for T in (60, 120):
        for sp in ("narrow", "wide"):
            for m in ("perm_wald", "perm_dir", "signflip", "signflip_dir"):
                g = des_tab[(des_tab["T"] == T) & (des_tab.spread == sp)
                            & (des_tab.method == m)]
                sim[(f"C5_T{T}_{sp}", m)] = invert(g.kappa, g.power_sizeadj)
    for kp in (1, K_PRE):
        for m in ("perm_wald", "perm_dir", "signflip", "signflip_dir"):
            g = ref_tab[(ref_tab.k_pre == kp) & (ref_tab.method == m)]
            nm = "C5_T120_wide_kpre1" if kp == 1 else "C5_T120_wide_kpre6"
            sim[(nm, m)] = invert(g.kappa, g.power_sizeadj)
    mde_tab["mde_sim_perm_wald"] = [sim.get((c, "perm_wald"), np.nan)
                                    for c in mde_tab.config]
    mde_tab["mde_sim_perm_dir"] = [sim.get((c, "perm_dir"), np.nan)
                                   for c in mde_tab.config]
    mde_tab["mde_sim_signflip_dir"] = [sim.get((c, "signflip_dir"), np.nan)
                                       for c in mde_tab.config]
    mde_tab.to_csv(os.path.join(OUT, "mde_table.csv"), index=False)
    S["mde"] = json.loads(mde_tab.to_json(orient="records"))

    ok = mde_tab.dropna(subset=["mde_sim_perm_wald"])
    if len(ok):
        rel = np.abs(ok.mde_wald - ok.mde_sim_perm_wald) / ok.mde_sim_perm_wald
        S["mde_closed_form_vs_sim"] = {
            "n_configs": int(len(ok)),
            "median_rel_error_wald": float(np.median(rel)),
            "max_rel_error_wald": float(rel.max())}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot([0, 1.2], [0, 1.2], "k:", lw=0.9)
    ax.scatter(mde_tab.mde_wald, mde_tab.mde_sim_perm_wald, s=34,
               label="Wald (label perm.)")
    ax.scatter(mde_tab.mde_directed, mde_tab.mde_sim_perm_dir, s=34, marker="s",
               label="directed (label perm.)")
    for _, r in mde_tab.iterrows():
        if np.isfinite(r.mde_sim_perm_wald):
            ax.annotate(r.config.replace("C5_T", "").replace("_wide", "w")
                        .replace("_narrow", "n"),
                        (r.mde_wald, r.mde_sim_perm_wald), fontsize=6,
                        xytext=(3, -6), textcoords="offset points")
    ax.set_xlabel(r"closed-form $\kappa_{MDE}$")
    ax.set_ylabel(r"simulated size-adjusted $\kappa_{MDE}$")
    ax.set_title("Closed-form MDE vs simulation (80% power)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    comp5, _ = design_objects(G=51, C=5, T=120, disc=(0.1, 1.0))
    s2 = float(sig_by["size"].loc[(51, 5)])
    spreads = np.linspace(0.05, 1.0, 40)
    for C in (3, 5, 8, 12):
        vals = []
        for w in spreads:
            dvec = np.linspace(0.55 - w / 2, 0.55 + w / 2, C)
            sd2 = float(((dvec - dvec.mean()) ** 2).sum())
            lam = inf.chi2_noncentrality_for_power(len(comp5["post"]), ALPHA, 0.80)
            di = inf.design_information(comp5)
            vals.append(np.sqrt(s2 * lam / (sd2 * di["I_k"])))
        ax.plot(spreads, vals, lw=1.4, label=f"C = {C}")
    ax.axhspan(0.05, 0.15, color="tab:red", alpha=0.12)
    ax.text(0.5, 0.10, "realistic $\\kappa$", fontsize=7, color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("discretion spread  $\\max d - \\min d$")
    ax.set_ylabel(r"$\kappa_{MDE}$ (Wald, 80% power)")
    ax.set_title("Design rule, $G=51$, $T=120$", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mde_design.png"), dpi=150)
    plt.close(fig)

    # ---------------- canonical design lemma ----------------------------
    checks = []
    for T0mode, kp in (("single", None), ("block", 1)):
        for T, g, p in ((24, 12, 0.5), (40, 20, 0.4), (60, 30, 0.5)):
            I, qp, N, pp = canonical_information(N=400, T=T, g=g, p=p, k_pre=kp)
            T0 = 1 if T0mode == "single" else g
            pred = N * pp * (1 - pp) * qp * T0 / (T0 + qp)
            checks.append(dict(reference=T0mode, T=T, g=g, p=p, q_post=qp,
                               I_k_numeric=I, I_k_closed_form=pred,
                               rel_err=abs(I - pred) / pred))
    can = pd.DataFrame(checks)
    can.to_csv(os.path.join(OUT, "canonical_information.csv"), index=False)
    S["canonical_information_max_rel_err"] = float(can.rel_err.max())

    # ---------------- H. exchangeability failure (Gaussian limit) --------
    # Proposition 6: with category-specific idiosyncratic variances the limit of
    # the category deviations is NOT exchangeable, so Theorem 1 does not apply.
    # Quantify the resulting size distortion in the Gaussian limit experiment.
    def gauss_perm_size(omega, delta, n_sim=20000, stat="wald", seed=0):
        rng = np.random.default_rng(seed)
        D, _ = inf._perm_matrix(np.asarray(delta, float), None, rng)
        s2 = float((np.asarray(delta, float) ** 2).sum())
        sd = np.sqrt(np.asarray(omega, float))
        rej = 0
        for _ in range(n_sim):
            eps = sd * rng.standard_normal(len(sd))
            g = (eps @ D) / s2                                   # (P,)
            # CR1-style variance estimate that is blind to heteroskedasticity:
            # v(pi) propto sum_c delta_pi(c)^2 * omegahat, omegahat common
            v = (D ** 2).sum(0) / s2 ** 2
            t = g ** 2 / v if stat == "wald" else g / np.sqrt(v)
            rej += int((t >= t[0] - 1e-12).mean() <= ALPHA + 1e-12)
        return rej / n_sim

    het_rows = []
    for C in (5, 8):
        dvec = np.linspace(0.1, 1.0, C)
        dl = dvec - dvec.mean()
        for r in (1.0, 2.0, 4.0, 9.0, 25.0):
            for place in ("high_d", "low_d"):
                om = np.ones(C)
                om[-1 if place == "high_d" else 0] = r
                for st in ("wald", "directed"):
                    het_rows.append(dict(C=C, var_ratio=r, inflated=place, stat=st,
                                         size=gauss_perm_size(om, dl, stat=st)))
    het = pd.DataFrame(het_rows)
    het.to_csv(os.path.join(OUT, "exchangeability_failure.csv"), index=False)
    S["exchangeability_failure"] = {
        f"C{int(C)}_ratio{r}_{pl}_{st}": round(v, 4) for C, r, pl, st, v in
        het[["C", "var_ratio", "inflated", "stat", "size"]].itertuples(index=False)}

    # ---------------- L. Theorem 6: local power of the permutation test ---
    comp5, di5 = design_objects(G=51, C=5, T=120, disc=(0.1, 1.0))
    s2_5 = float(sig_by["size"].loc[(51, 5)])
    lp_rows = []
    for kap in (0.0, 0.15, 0.30, 0.45, 0.60, 0.80):
        sig = np.log1p(kap * comp5["d"])
        gg = float(comp5["delta"] @ sig / (comp5["delta"] ** 2).sum())
        lam = inf.noncentrality(comp5, sig, sigma2=s2_5)["lambda"]
        lp_rows.append(dict(
            kappa=kap, g=gg, lam=lam,
            # linearised: signal g*1 per unit of rho_pi (Theorem 6)
            theory_perm_wald=inf.local_power_permutation(
                comp5["delta"], np.ones(len(comp5["post"])), di5["W"], s2_5, gg,
                n_sim=3000, alpha=ALPHA, stat="wald", seed=1),
            theory_perm_dir=inf.local_power_permutation(
                comp5["delta"], np.ones(len(comp5["post"])), di5["W"], s2_5, gg,
                n_sim=3000, alpha=ALPHA, stat="directed", seed=1),
            theory_infeasible_chi2=inf.local_power_wald(lam, len(comp5["post"]), ALPHA),
            sim_perm_wald=float(pow_tab[(pow_tab.C == 5) & (pow_tab.kappa == kap)
                                        & (pow_tab.method == "perm_wald")]
                                ["power_sizeadj"].mean())))
    lp = pd.DataFrame(lp_rows)
    lp.to_csv(os.path.join(OUT, "local_power_check.csv"), index=False)
    S["local_power_check"] = json.loads(lp.to_json(orient="records"))

    # ---------------- caveats -------------------------------------------
    S["caveats"] = [
        "The label-permutation test needs C! >= 1/alpha distinct permuted values. "
        "With C = 3 the two-sided Wald takes at most 3 distinct values over the 6 "
        "permutations (equally spaced d makes reversal a sign flip), so its "
        "attainable p-values are {1/3, 2/3, 1} and it CANNOT reject at 5%; the "
        "one-sided directed statistic attains 1/6 = 0.167, still above 5%. "
        "Size 0.000 at C = 3 in size_table.csv is that resolution bound, not a "
        "coding artefact. Use the cluster sign-flip or wild-bootstrap tests when "
        "C <= 4.",
        "Exactness of the label-permutation test requires exchangeability of the "
        "loading-normalised category arrays. The DGP satisfies it by construction "
        "(loadings are a uniformly random permutation of a fixed set, drawn "
        "independently of d, and the idiosyncratic errors are i.i.d. across "
        "categories). Cells with category-specific error variances or "
        "block-correlated categories are NOT simulated here; theory/inference.md "
        "Section 3.4 gives the asymptotic (approximate-symmetry) fallback.",
        "The wild cluster bootstrap is applied to the partialled-out regression "
        "(fixed effects removed once, weights applied to the residualised "
        "residuals) rather than re-absorbing the fixed effects on every draw. The "
        "difference is O_p(1/G) and standard in partialling-out implementations, "
        "but it is an implementation choice, not the literal CGM algorithm.",
        "Size adjustment uses the alpha-quantile of the p-value distribution from "
        "the matching kappa = 0 cell of the SAME configuration (no borrowing "
        "across designs). Size-adjusted power at kappa = 0 is 0.05 by construction.",
        "MDE inversion is linear interpolation on a coarse kappa grid with "
        "500-600 replications per cell (MCSE <= 0.02), so simulated MDEs are "
        "accurate to roughly +/- 0.05; read the closed-form/simulation comparison "
        "as a check on the formula's shape and level, not to two decimals.",
        "Under the Poisson recording map the wedge is log1p(kappa d_c), not "
        "kappa d_c; the closed-form MDE column 'mde_wald' solves the nonlinear "
        "version and 'mde_wald_linear' the linearised one. At kappa ~ 0.5 the two "
        "differ by roughly 15-20%.",
        "sigma^2 is the delta^2-weighted mean of the category residual variances "
        "of the loading-normalised outcome, estimated on the kappa = 0 cells. In "
        "an application it would come from pilot data; it is the only quantity in "
        "the MDE formula that is not a pure design object.",
    ]
    n_err = int(sum(pd.read_csv(os.path.join(CKPT, j["key"] + ".csv"))["err"].sum()
                    for j in jobs))
    if n_err:
        S["caveats"].append(f"{n_err} replication-level failures recorded as NaN.")
    rt = os.path.join(CKPT, "_runtime.log")
    if os.path.exists(rt):
        with open(rt) as fh:
            S["runtime_min_total"] = round(sum(float(x) for x in fh.read().split()) / 60, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=2)
    print("Aggregation done.")
    print(json.dumps(S["headline_G51_C5"], indent=1))
    print(json.dumps(S["exactness_checks"], indent=1))


# ------------------------------------------------------------------ driver
def main():
    t0 = time.time()
    budget = float(os.environ.get("TI_BUDGET_S", 100000))
    workers = int(os.environ.get("TI_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
    os.makedirs(CKPT, exist_ok=True)
    jobs = build_jobs()
    pending = [j for j in jobs
               if not os.path.exists(os.path.join(CKPT, j["key"] + ".csv"))]
    print(f"{len(jobs)} cells, {len(pending)} pending; workers={workers}")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for job in pending:
                if time.time() - t0 > budget:
                    print("Budget reached; re-run to resume.")
                    break
                reps = job["reps"]
                step = max(5, int(np.ceil(reps / (workers * 3))))
                tasks = [(job, r, min(r + step, reps)) for r in range(0, reps, step)]
                rows = [r for ch in ex.map(run_chunk, tasks) for r in ch]
                pd.DataFrame(rows).sort_values("rep").to_csv(
                    os.path.join(CKPT, job["key"] + ".csv"), index=False)
                print(f"  [{job['key']}] {len(rows)} reps "
                      f"({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(CKPT, "_runtime.log"), "a") as fh:
        fh.write(f"{time.time() - t0:.1f}\n")
    left = [j for j in jobs
            if not os.path.exists(os.path.join(CKPT, j["key"] + ".csv"))]
    if left:
        print(f"NOT COMPLETE: {len(left)} cells remain.")
    else:
        aggregate(jobs)
        print("COMPLETE")


if __name__ == "__main__":
    main()
