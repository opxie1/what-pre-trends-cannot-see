"""Smoke tests for the did_measurement core: closed-form checks on large panels.

Run from the project root:  PYTHONPATH=src python tests/test_core.py
Every check prints PASS/FAIL and the script exits nonzero on any failure.
"""

import sys
import numpy as np
import pandas as pd

from did_measurement import (
    DGPConfig, generate_panel, event_study_twfe, att_gt_cs,
    discretion_gradient_test, pretrend_test,
    sensitivity_interval, breakdown_kappa, recording_bounds,
    estimate_relative_loadings, bridge_decomposition,
    cohort_sharp_interval, aggregate_breakdown_kappa,
    aggregate_sensitivity_interval,
    cohort_event_studies, cohort_beta_matrix, bridge_decomposition_by_cohort,
    comovement_test, discretion_shock_test,
    m2_variance, m2_variance_gls, loading_channel_variance, bridge_delta_se,
)
from did_measurement.diagnostics import moments_from_cube

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def post_mean(es_table, lmax=3):
    t = es_table
    return t.loc[(t.ell >= 0) & (t.ell <= lmax), "coef"].mean()


# ---------------------------------------------------------------- 1. additive closed form
cfg = DGPConfig(n_units=800, n_categories=3, discretion=(0.0, 0.5, 1.0),
                include_bridge=False, theta_spread=0.0, tau_level=1.0,
                kappa=0.8, sigma_eps=0.5, seed=11)
df = generate_panel(cfg)
for c, dc in [(0, 0.0), (1, 0.5), (2, 1.0)]:
    es = event_study_twfe(df[df.cat == c], k_pre=5, k_post=5)
    got = post_mean(es["table"])
    want = 1.0 + 0.8 * dc
    check(f"additive bias formula cat{c}", abs(got - want) < 0.10,
          f"got {got:.3f} want {want:.3f}")

# leads should be ~0 (no PT violation)
es0 = event_study_twfe(df[df.cat == 2], k_pre=5, k_post=5)
pt = pretrend_test(es0)
check("pretrend test does not reject under PT", pt["pvalue"] > 0.01,
      f"p={pt['pvalue']:.3f}")

# ---------------------------------------------------------------- 2. Theorem 1: FE absorb time-invariant distortions
cfg = DGPConfig(n_units=800, n_categories=2, discretion=(0.3, 1.0),
                include_bridge=False, theta_spread=0.0, tau_level=1.0,
                kappa=0.0, mu_scale=50.0, sigma_eps=0.5, seed=12)
df = generate_panel(cfg)
es = event_study_twfe(df[df.cat == 1], k_pre=5, k_post=5)
got = post_mean(es["table"])
check("Theorem 1: huge time-invariant distortion, zero bias", abs(got - 1.0) < 0.10,
      f"got {got:.3f} want 1.000")

# ---------------------------------------------------------------- 3. multiplicative closed form (log outcome)
cfg = DGPConfig(n_units=800, n_categories=2, discretion=(0.2, 1.0),
                include_bridge=False, theta_spread=0.0, tau_level=1.0,
                kappa=0.5, map_type="multiplicative", sigma_eps=0.5, seed=13)
df = generate_panel(cfg)
df["logY"] = np.log(df["Y"])
for c, dc in [(0, 0.2), (1, 1.0)]:
    es = event_study_twfe(df[df.cat == c], outcome="logY", k_pre=5, k_post=5)
    got = post_mean(es["table"])
    want = cfg.latent_scale * 1.0 + np.log1p(0.5 * dc)
    check(f"multiplicative log bias formula cat{c}", abs(got - want) < 0.10,
          f"got {got:.3f} want {want:.3f}")

# ---------------------------------------------------------------- 4. CS matches TWFE under homogeneous effects
cfg = DGPConfig(n_units=800, n_categories=1, discretion=(0.5,), include_bridge=False,
                theta_spread=0.0, tau_level=1.0, kappa=0.6, sigma_eps=0.5, seed=14)
df = generate_panel(cfg)
es_t = event_study_twfe(df, k_pre=5, k_post=5)["table"].set_index("ell")["coef"]
es_c = att_gt_cs(df, control="never")["es"].set_index("ell")["coef"]
common = [l for l in range(0, 5) if l in es_t.index and l in es_c.index]
diff = max(abs(es_t[l] - es_c[l]) for l in common)
check("CS ~= TWFE under homogeneous effects", diff < 0.08, f"max diff {diff:.3f}")
cs_ny = att_gt_cs(df, control="notyet")["es"].set_index("ell")["coef"]
diff2 = max(abs(es_t[l] - cs_ny[l]) for l in common if l in cs_ny.index)
check("CS not-yet-treated control also close", diff2 < 0.12, f"max diff {diff2:.3f}")

# ---------------------------------------------------------------- 5. discretion-gradient test: size and power
cfg0 = DGPConfig(n_units=800, n_categories=4, include_bridge=False,
                 theta_spread=0.3, kappa=0.0, sigma_eps=0.5, seed=15)
df0 = generate_panel(cfg0)
t0 = discretion_gradient_test(df0)
check("gradient test (loading-adjusted): no rejection under invariance",
      t0["pvalue"] > 0.01, f"p={t0['pvalue']:.3f}")
t0n = discretion_gradient_test(df0, loading_adjust=False)
check("gradient test (naive): over-rejects when realized loadings correlate with d",
      t0n["pvalue"] < 0.05, f"p={t0n['pvalue']:.4f} (documents why adjustment is needed)")
cfg1 = DGPConfig(n_units=800, n_categories=4, include_bridge=False,
                 theta_spread=0.3, kappa=1.5, sigma_eps=0.5, seed=16)
t1 = discretion_gradient_test(generate_panel(cfg1))
check("gradient test: rejects under measurement change", t1["pvalue"] < 0.01,
      f"p={t1['pvalue']:.4f}")

# orthogonality: PT violation without measurement change -> gradient test quiet, pretrend loud
cfgv = DGPConfig(n_units=800, n_categories=4, include_bridge=False, theta_spread=0.0,
                 kappa=0.0, trend_break=0.06, sigma_eps=0.5, seed=17)
dfv = generate_panel(cfgv)
tg = discretion_gradient_test(dfv)
esv = event_study_twfe(dfv[dfv.cat == 3], k_pre=5, k_post=5)
ptv = pretrend_test(esv)
check("orthogonality: PT violation -> pretrend rejects", ptv["pvalue"] < 0.01,
      f"p={ptv['pvalue']:.4f}")
check("orthogonality: PT violation -> gradient test does NOT reject",
      tg["pvalue"] > 0.01, f"p={tg['pvalue']:.3f}")

# ---------------------------------------------------------------- 6. bridge decomposition
cfg = DGPConfig(n_units=1000, n_categories=4, include_bridge=True, theta_spread=0.3,
                tau_level=1.0, kappa=1.0, sigma_eps=0.5, seed=18)
df = generate_panel(cfg)
theta = df.groupby("cat")["theta"].first()
dvec = df.groupby("cat")["d"].first()
ld = estimate_relative_loadings(df)
rho_err = max(abs(ld["rho"][c] - theta[c]) for c in ld["rho"])
check("bridge loadings recovered (IV method)", rho_err < 0.15, f"max err {rho_err:.3f}")
bd = bridge_decomposition(df, estimator="cs", control="never")
tab = bd["table"]
row = tab[(tab.ell >= 0) & (tab.ell <= 3)].groupby("cat")[["es_latent", "es_measurement"]].mean()
lat_err = max(abs(row.loc[c, "es_latent"] - theta[c] * 1.0) for c in row.index)
meas_err = max(abs(row.loc[c, "es_measurement"] - 1.0 * dvec[c]) for c in row.index)
check("bridge: latent component recovered", lat_err < 0.15, f"max err {lat_err:.3f}")
check("bridge: measurement component recovered", meas_err < 0.15, f"max err {meas_err:.3f}")

# ---------------------------------------------------------------- 7. bounds sanity
si = sensitivity_interval(coef=1.5, se=0.1, d_c=1.0, kappa_bar=0.8)
check("sensitivity interval brackets", si["L"] < si["U"] and si["ci_lo"] < si["L"]
      and si["ci_hi"] > si["U"], f"[{si['ci_lo']:.2f},{si['ci_hi']:.2f}]")
bk = breakdown_kappa(coef=1.5, se=0.1, d_c=1.0)
check("breakdown ordering stat < point", bk["stat"] < bk["point"],
      f"stat {bk['stat']:.3f} point {bk['point']:.3f}")
rb = recording_bounds(beta_log=0.4, se=0.05, r0=0.6)
check("recording bounds contain latent truth in closed form",
      rb["L"] <= 0.4 + np.log(0.6) + 1e-9 and rb["U"] >= 0.4 - 1e-9,
      f"[{rb['L']:.3f},{rb['U']:.3f}]")

# ---------------------------------------------------------------- 8. X2: cohort
# aggregation -- the sharp set is the MINKOWSKI AVERAGE of per-cohort sets
Bc = np.array([[0.30, 0.00], [0.00, 0.30]])
sc = cohort_sharp_interval(Bc, weights=[0.5, 0.5], m_bar=[0.5, 0.5])
check("X2: sharpness_review F2 counterexample reproduced",
      abs(sc["minkowski_L"] + 0.2) < 1e-12 and abs(sc["minkowski_U"]) < 1e-12
      and abs(sc["naive_L"] + 0.35) < 1e-12 and abs(sc["naive_U"] - 0.15) < 1e-12,
      f"sharp [{sc['minkowski_L']:.3f},{sc['minkowski_U']:.3f}] "
      f"naive [{sc['naive_L']:.3f},{sc['naive_U']:.3f}] ratio {sc['width_ratio']:.2f}")

rngx = np.random.default_rng(7)
nest, strict, c1_gap = 0, 0, 0.0
for _ in range(400):
    G_, C_ = int(rngx.integers(2, 5)), int(rngx.integers(1, 6))
    rho_ = rngx.uniform(0.4, 1.6, C_)
    mb_ = rngx.uniform(0.05, 1.5, C_)
    x_ = rngx.uniform(-1, 1, G_)
    Bx = np.array([[rho_[c] * x_[g] + rngx.uniform(0, mb_[c]) for c in range(C_)]
                   for g in range(G_)])
    w_ = rngx.dirichlet(np.ones(G_))
    r_ = cohort_sharp_interval(Bx, w_, mb_, rho_)
    nest += (r_["minkowski_L"] >= r_["naive_L"] - 1e-9
             and r_["minkowski_U"] <= r_["naive_U"] + 1e-9)
    strict += r_["width_ratio"] < 1 - 1e-9
    if C_ == 1:
        c1_gap = max(c1_gap, abs(r_["minkowski_L"] - r_["naive_L"]),
                     abs(r_["minkowski_U"] - r_["naive_U"]))
check("X2: Minkowski set is contained in the naive set in every design",
      nest == 400, f"{nest}/400")
check("X2: the two sets differ strictly in a large share of designs",
      strict > 200, f"{strict}/400 strict")
check("X2: with one category cohort aggregation is harmless", c1_gap < 1e-12,
      f"max gap {c1_gap:.2e}")

cfgx = DGPConfig(n_units=1200, n_periods=16, cohorts=(8, 11), n_categories=3,
                 include_bridge=True, theta_spread=0.2, tau_level=1.0,
                 kappa=0.7, sigma_eps=0.5, seed=21)
dfx = generate_panel(cfgx)
ces = cohort_event_studies(dfx, control="never")
es_agg = att_gt_cs(dfx[dfx.cat == 1], control="never")["es"].set_index("ell")["coef"]
recon = {}
for l in [0, 1, 2]:
    gs = [g for g in ces["cohorts"] if (1, g, l) in ces["beta"]]
    recon[l] = sum(ces["weights"][(l, g)] * ces["beta"][(1, g, l)] for g in gs)
agg_err = max(abs(recon[l] - es_agg[l]) for l in recon)
check("X2: per-cohort betas reaggregate to the CS event study", agg_err < 1e-10,
      f"max err {agg_err:.2e}")
Bm, wv, gsv = cohort_beta_matrix(ces, 1, [0, 1, 2])
check("X2: cohort_beta_matrix returns one row per cohort",
      Bm is not None and Bm.shape == (len(gsv), 3) and abs(wv.sum() - 1) < 1e-12,
      f"shape {None if Bm is None else Bm.shape}")
rngb = np.random.default_rng(33)
bad_bk, strict_bk, ex = 0, 0, None
for _ in range(300):
    G_, C_ = int(rngb.integers(2, 5)), int(rngb.integers(2, 6))
    d_b = np.sort(rngb.uniform(0.05, 1.0, C_))
    kb_ = rngb.uniform(0.3, 1.5)
    x_b = rngb.uniform(0.05, 0.8, G_)
    Bb = np.array([[x_b[g] + rngb.uniform(0, kb_ * d_b[c]) for c in range(C_)]
                   for g in range(G_)])
    w_b = rngb.dirichlet(np.ones(G_))
    s_bk = aggregate_breakdown_kappa(Bb, d_b, w_b)
    n_bk = aggregate_breakdown_kappa(Bb, d_b, w_b, aggregate="naive")
    if np.isfinite(s_bk) and np.isfinite(n_bk):
        bad_bk += s_bk < n_bk - 1e-7
        if s_bk > n_bk + 1e-7:
            strict_bk += 1
            ex = (s_bk, n_bk)
check("X2: sharp breakdown kappa is never below the naive one, and is often above",
      bad_bk == 0 and strict_bk > 0,
      f"{bad_bk} violations, {strict_bk}/300 strictly larger, e.g. {ex}")
asi = aggregate_sensitivity_interval(Bc, se=0.05, d=np.array([1.0, 1.0]),
                                     kappa_bar=0.5, weights=[0.5, 0.5])
asi_n = aggregate_sensitivity_interval(Bc, se=0.05, d=np.array([1.0, 1.0]),
                                       kappa_bar=0.5, weights=[0.5, 0.5],
                                       aggregate="naive")
check("X2: aggregate IM confidence interval is nested inside the naive one",
      asi["ci_lo"] > asi_n["ci_lo"] and asi["ci_hi"] < asi_n["ci_hi"]
      and asi["ci_lo"] < asi["L"] and asi["ci_hi"] > asi["U"],
      f"sharp CI [{asi['ci_lo']:.3f},{asi['ci_hi']:.3f}] "
      f"naive CI [{asi_n['ci_lo']:.3f},{asi_n['ci_hi']:.3f}]")
bdc = bridge_decomposition_by_cohort(dfx)
check("X2: bridge point decomposition aggregates exactly (linear in beta)",
      bdc["max_abs_gap"] < 1e-9, f"gap {bdc['max_abs_gap']:.2e}")

# ---------------------------------------------------------------- 9. X3: corrected
# efficiency formulas
rnge = np.random.default_rng(9)
m2_err, m2_small = 0.0, 0
for _ in range(200):
    C_ = int(rnge.integers(3, 7))
    th_ = rnge.uniform(0.3, 2.0, C_)
    d_ = rnge.uniform(0.0, 1.0, C_)
    ps_ = rnge.uniform(0.2, 2.0, C_)
    sv_ = float(rnge.uniform(0.05, 1.0))
    vc = m2_variance(th_, d_, ps_, sv_, 3.0, corrected=True)
    vg = m2_variance_gls(th_, d_, sv_ * np.outer(th_, th_) + np.diag(ps_), 3.0)
    vb = m2_variance(th_, d_, ps_, sv_, 3.0, corrected=False)
    m2_err = max(m2_err, abs(vc - vg) / abs(vg))
    m2_small += vb < vc
check("X3: corrected M2 closed form matches direct GLS inversion", m2_err < 1e-8,
      f"max rel err {m2_err:.2e}")
check("X3: the printed (independence) M2 variance is too small in every design",
      m2_small == 200, f"{m2_small}/200")
vr = (loading_channel_variance(1.0, 1.0, 1.3, corrected=True)
      / loading_channel_variance(1.0, 1.0, 1.3, corrected=False))
check("X3: Phi^MD correction scales the loading channel by rho^2",
      abs(vr - 1.69) < 1e-12, f"ratio {vr:.4f}")
se_ok = bridge_delta_se(2.5, 1.0, 0.01, 0.04, 0.0, corrected=True)
se_bad = bridge_delta_se(2.5, 1.0, 0.01, 0.04, 0.0, corrected=False)
check("X3: printed score understates the bridge SE when rho > 1", se_bad < se_ok,
      f"printed {se_bad:.4f} corrected {se_ok:.4f}")

# ---------------------------------------------------------------- 10. X4: the A10
# comovement diagnostic
th_d = np.array([1.0, 0.8, 1.2, 0.9, 1.4])
d_d = np.array([0.0, 0.3, 0.6, 1.0, 0.45])


def _cube(seed, sG2=0.0, lam=0.0, rho_F=0.5, N=400, T=8):
    rg = np.random.default_rng(seed)
    F = np.empty((N, T))
    F[:, 0] = rg.normal(size=N)
    for t in range(1, T):
        F[:, t] = rho_F * F[:, t - 1] + np.sqrt(1 - rho_F ** 2) * rg.normal(size=N)
    e = rg.normal(scale=0.8, size=(N, T, 5))
    if lam:
        tot = e[:, :, 1].sum(axis=1, keepdims=True)
        e[:, :, 2] = e[:, :, 2] + lam * (tot - e[:, :, 1])
    cube = th_d[None, None, :] * F[:, :, None] + e
    if sG2:
        sh = rg.normal(size=(N, T))
        cube = cube + np.sqrt(sG2) * sh[:, :, None] * d_d[None, None, :]
    return cube


r_null = comovement_test(None, moments=moments_from_cube(_cube(1), max_lag=1,
                                                         demean=True),
                         max_lag=1, n_boot=299, seed=1)
check("X4: A10 diagnostic does not reject under A10 (serially correlated factor)",
      r_null["pvalue_boot"] > 0.10 and r_null["df"] == 24,
      f"p_boot={r_null['pvalue_boot']:.3f} p_chi2={r_null['pvalue_chi2']:.3f} "
      f"df={r_null['df']}")
r_b = comovement_test(None, moments=moments_from_cube(_cube(2, sG2=0.5, N=800),
                                                      max_lag=1, demean=True),
                      max_lag=1, n_boot=299, seed=2)
check("X4: A10 diagnostic rejects a discretion-loading common shock (A10b)",
      r_b["pvalue_boot"] < 0.05, f"p_boot={r_b['pvalue_boot']:.4f}")
r_b_opt = comovement_test(None, moments=moments_from_cube(_cube(2, sG2=0.25),
                                                          max_lag=1, demean=True),
                          max_lag=1, n_boot=299, seed=2, weight="optimal")
check("X4: the optimal-weight variant is available and more powerful (but oversized)",
      np.isfinite(r_b_opt["pvalue_chi2"]) and r_b_opt["pvalue_boot"] < 0.05,
      f"p_boot={r_b_opt['pvalue_boot']:.4f} p_chi2={r_b_opt['pvalue_chi2']:.4f}")
r_a = comovement_test(None, moments=moments_from_cube(_cube(3, lam=0.5), max_lag=1,
                                                      demean=True),
                      max_lag=1, n_boot=299, seed=3)
check("X4: A10 diagnostic rejects cross-lag leakage (A10a)",
      r_a["pvalue_boot"] < 0.05, f"p_boot={r_a['pvalue_boot']:.4f}")

cb = _cube(4, sG2=0.25, N=1500)
m0 = moments_from_cube(cb, max_lag=0, demean=True)
fs = discretion_shock_test(None, d_d, moments=m0, n_boot=0)
S = np.zeros((5, 5))
for j, (_, c, k) in enumerate(m0["keys"]):
    S[c, k] = S[k, c] = m0["g"][j]
rho_iv = np.array([np.average(
    [S[c, k] / S[0, k] for k in range(5) if k not in (c, 0)],
    weights=[abs(S[0, k]) for k in range(5) if k not in (c, 0)])
    for c in range(1, 5)])
rho_rb = np.array([fs["rho_corrected"][c] for c in range(1, 5)])
tr = (th_d / th_d[0])[1:]
check("X4: focused test detects the records-management shock",
      fs["pvalue"] < 0.01 and abs(fs["xi_hat"] - 0.25) < 0.12,
      f"xi_hat={fs['xi_hat']:.3f} (true 0.25) p={fs['pvalue']:.2e}")
check("X4: xi-corrected loadings beat the shipped IV under an A10b violation",
      np.max(np.abs(rho_rb - tr)) < np.max(np.abs(rho_iv - tr)),
      f"corrected {np.max(np.abs(rho_rb - tr)):.4f} vs IV "
      f"{np.max(np.abs(rho_iv - tr)):.4f}")

# two discretion-free bridges make rho exactly immune to A10b
d2 = d_d.copy()
d2[4] = 0.0
Sig2 = np.outer(th_d, th_d) + 0.25 * np.outer(d2, d2) + np.diag([0.5] * 5)
keep2 = [c for c in range(5) if c != 4]          # 4 is the instrument itself
rho_2b = np.array([Sig2[c, 4] / Sig2[0, 4] for c in keep2])
err_2b = float(np.max(np.abs(rho_2b - (th_d / th_d[0])[keep2])))
check("X4: a second discretion-free bridge is exactly immune to A10b",
      err_2b < 1e-12, f"max err {err_2b:.2e}")

# ---------------------------------------------------------------- R. referee-response track
#   nondiag (identified set for rho when A10b fails), robustness (A6 and the
#   discretion index), exactness (non-equivariant loading normalisation), and
#   the inference additions: realised-orbit resolution floors, the injection
#   reference set, the Hotelling/permutation MDE calibrations, the validity
#   region of the Corollary 8.4 expansion, and Anderson-Rubin sets.

from did_measurement.nondiag import (rho_outer_set, rho_sharp_set,
                                     rho_lemma63_point, gradient_bounds_over_rho,
                                     breakdown_b, rho_bias_d_shock)
from did_measurement.robustness import (rank_gradient_test, shape_test,
                                        loco_gradient, index_class_slope,
                                        rank_recoding, bridge_stability)
from did_measurement.exactness import equivariance_gap
from did_measurement.inference import (gradient_components, permutation_test,
                                       label_stabilizer_size, anderson_rubin_set,
                                       mde_kappa, hotelling_crit,
                                       wald_size_expansion, predicted_wald_size)
from did_measurement.bounds import (measurement_class_consistency,
                                    sensitivity_interval)
from scipy import stats as _st

# ---- R1  at b = 0 the sharp set is the Lemma 6.3 point, and it is the truth
_th = np.array([1.0, 1.7, 0.6, 1.2])
_V, _psi = 0.8, np.array([0.5, 0.7, 0.4, 0.6])
_S = _V * np.outer(_th, _th) + np.diag(_psi)
_pt = rho_lemma63_point(_S, ref=0)["rho"]
_err63 = max(abs(_pt[c] - _th[c]) for c in range(4))
check("R1: Lemma 6.3 point recovers rho exactly under A10b",
      _err63 < 1e-10, f"max err {_err63:.2e}")
_sh0 = rho_sharp_set(_S, ref=0, b=0.0)["rho_set"]
_err0 = max(max(abs(_sh0[c][0] - _th[c]), abs(_sh0[c][1] - _th[c])) for c in range(4))
check("R1: the b = 0 sharp set collapses to that point",
      _err0 < 5e-3, f"max endpoint err {_err0:.2e}")

# ---- R2  nested in the outer set, covers the truth, widens in b
_w = []
_inside = _covers = True
for _b in (0.0, 0.05, 0.10, 0.25):
    _s = rho_sharp_set(_S, ref=0, b=_b)["rho_set"]
    _o = rho_outer_set(_S, ref=0, b=_b)["rho_set"]
    _w.append(_s[1][1] - _s[1][0])
    _inside = _inside and all(_o[c][0] - 1e-6 <= _s[c][0] and _s[c][1] <= _o[c][1] + 1e-6
                              for c in range(4))
    _covers = _covers and all(_s[c][0] - 1e-6 <= _th[c] <= _s[c][1] + 1e-6
                              for c in range(4))
check("R2: sharp set nested in the outer set and covering the truth at every b",
      _inside and _covers, f"nested {_inside}, covers {_covers}")
check("R2: the sharp set widens monotonically in b",
      all(_w[i] <= _w[i + 1] + 1e-9 for i in range(len(_w) - 1)),
      f"widths {[round(x, 4) for x in _w]}")

# ---- R3  a genuinely non-diagonal Psi: the point estimate is biased, the set covers
_b_true = 0.3
_Psi = np.diag(_psi) + _b_true * np.sqrt(np.outer(_psi, _psi)) * (1 - np.eye(4))
_S2 = _V * np.outer(_th, _th) + _Psi
_o2 = rho_outer_set(_S2, ref=0, b=_b_true)["rho_set"]
_cov2 = all(_o2[c][0] - 1e-6 <= _th[c] <= _o2[c][1] + 1e-6 for c in range(4))
_pt2 = rho_lemma63_point(_S2, ref=0)["rho"]
_bias2 = max(abs(_pt2[c] - _th[c]) for c in range(4))
check("R3: under a non-diagonal Psi Lemma 6.3 is biased but the A10b(b) set covers",
      _cov2 and _bias2 > 0.02, f"Lemma 6.3 bias {_bias2:.4f}, set covers {_cov2}")

# ---- R4  the closed-form d-shock bias, and the gradient bounds
_rb = rho_bias_d_shock([1.0, 1.5, 2.0], [0.0, 0.5, 1.0], 0.4)
check("R4: rho_hat = rho + varkappa d exactly",
      np.allclose(_rb, [1.0, 1.7, 2.4]), f"{np.round(_rb, 6).tolist()}")
_beta = np.array([0.02, -0.01, 0.03, -0.04])
_dd = np.array([0.0, 0.4, 0.7, 1.0])
_deg = {c: (_th[c], _th[c]) for c in range(4)}
_g_deg = gradient_bounds_over_rho(_beta, _dd, _deg)
_delta = _dd - _dd.mean()
_g_direct = float((_delta * (_beta / _th)).sum() / (_delta ** 2).sum())
check("R4: a degenerate loading box reproduces the point gradient",
      abs(_g_deg["L"] - _g_direct) < 1e-12 and abs(_g_deg["U"] - _g_direct) < 1e-12,
      f"{_g_deg['L']:.8f} vs {_g_direct:.8f}")
_g_wide = gradient_bounds_over_rho(_beta, _dd,
                                   {c: (_th[c] * 0.8, _th[c] * 1.2) for c in range(4)})
check("R4: widening the loading box widens the gradient interval",
      _g_wide["L"] <= _g_deg["L"] + 1e-12 and _g_wide["U"] >= _g_deg["U"] - 1e-12,
      f"[{_g_wide['L']:.4f}, {_g_wide['U']:.4f}]")
_g_zero = gradient_bounds_over_rho(_beta, _dd, {0: (1.0, 1.0), 1: (-0.5, 0.5),
                                                2: (0.5, 1.0), 3: (0.5, 1.0)})
check("R4: a loading interval straddling zero returns an unbounded gradient",
      _g_zero["unbounded"] and not _g_zero["sign_determinate"], "")

# ---- R5  breakdown value of b
_bd = breakdown_b(_S, _beta, _dd, ref=0, b_hi=1.0, tol=1e-3)
_bstar = _bd["breakdown_b"]
_ok5 = 0.0 <= _bstar <= 1.0
if _bstar > 0.0:
    _ok5 = _ok5 and gradient_bounds_over_rho(
        _beta, _dd,
        rho_outer_set(_S, ref=0, b=max(_bstar - 2e-3, 0.0))["rho_set"]
    )["sign_determinate"]
check("R5: breakdown_b lies in [0, 1] and the sign is determinate just below it",
      _ok5, f"b* = {_bstar:.4f} ({_bd['note']})")

# ---- R6  the resolution floor is |Stab| x multiplicity / C!, not 1/C!
check("R6: label_stabilizer_size counts tied-label factorials",
      label_stabilizer_size([1.0, 1.0, 0.5, 0.0]) == 2
      and label_stabilizer_size([1.0, 1.0, 0.0, 0.0]) == 4
      and label_stabilizer_size([0.1, 0.2, 0.3]) == 1, "")

cfgR = DGPConfig(n_units=120, n_categories=4, discretion=(0.0, 0.5, 0.5, 1.0),
                 include_bridge=False, theta_spread=0.0, tau_level=0.5,
                 kappa=0.0, sigma_eps=0.5, seed=707)
dfR = generate_panel(cfgR)
compR = gradient_components(dfR, outcome="Y", k_pre=3, k_post=3, loading="none")
prR = permutation_test(compR, stat="wald")
check("R6: with tied labels the realised floor exceeds the naive 1/C!",
      prR["min_pvalue"] > prR["min_pvalue_naive"] + 1e-12
      and prR["min_pvalue"] >= prR["stabilizer"] / prR["n_perm"] - 1e-12,
      f"realised {prR['min_pvalue']:.4f} vs naive {prR['min_pvalue_naive']:.4f}, "
      f"|Stab| = {prR['stabilizer']}, distinct {prR['n_distinct']}/{prR['n_perm']}")
check("R6: rejection_attainable is a computed function of the realised floor",
      prR["rejection_attainable"] == (prR["min_pvalue"] <= 0.05 + 1e-12), "")

# ---- R7  the injection reference set when the loading guard fires
cfgI = DGPConfig(n_units=120, n_categories=5, discretion=(0.0, 0.25, 0.5, 0.75, 1.0),
                 include_bridge=False, theta_spread=0.0, tau_level=0.5,
                 kappa=0.0, sigma_eps=0.5, seed=808)
dfI = generate_panel(cfgI)
compI = dict(gradient_components(dfI, outcome="Y", k_pre=3, k_post=3, loading="none"))
_dfull = np.asarray(compI["d"], float)
compI["d_full"] = _dfull
compI["kept"] = np.array([True, True, True, False])
compI["B"] = np.asarray(compI["B"])[:, :3]
compI["A"] = np.asarray(compI["A"])[:, :, :3]
compI["d"] = _dfull[:3]
compI["delta"] = compI["d"] - compI["d"].mean()
compI["C"] = 3
compI["rho"] = np.asarray(compI["rho"])[:3]
compI["resid_var"] = np.asarray(compI["resid_var"])[:3]
prI = permutation_test(compI, stat="wald", reference="injection")
prL = permutation_test(compI, stat="wald", reference="labels")
_C0 = len(compI["d_full"])
check("R7: the injection reference set has C0!/(C0-C)! members, the label set C!",
      prI["n_perm"] == _C0 * (_C0 - 1) * (_C0 - 2) and prL["n_perm"] == 6,
      f"C0 = {_C0}: injection n = {prI['n_perm']}, labels n = {prL['n_perm']}; "
      f"p {prI['pvalue']:.4f} vs {prL['pvalue']:.4f}")

# ---- R8  the Corollary 8.4 expansion, inside and outside its validity region
_inside_err, _outside_err = [], []
for _G in (30, 51, 100, 200, 400):
    for _m in (1, 3, 5, 9, 13, 20):
        _ex = predicted_wald_size(_G, _m)
        _ap = wald_size_expansion(_G, _m)
        (_inside_err if _m / (_G - 1) <= 0.10 else _outside_err).append(abs(_ap - _ex))
check("R8: the size expansion is accurate to 1pp only inside H/(J-1) <= 0.10",
      max(_inside_err) <= 0.01 and max(_outside_err) > 0.05,
      f"max |error| inside {max(_inside_err):.4f}, outside {max(_outside_err):.4f}")
check("R8: at the paper's own design the expansion understates the exact size",
      wald_size_expansion(51, 13) < predicted_wald_size(51, 13) - 0.05,
      f"expansion {wald_size_expansion(51, 13):.4f} vs exact "
      f"{predicted_wald_size(51, 13):.4f}")

# ---- R9  MDE priced at a critical value the project does not refute
compM = gradient_components(dfI, outcome="Y", k_pre=3, k_post=3, loading="none")
_m_chi = mde_kappa(compM, calibration="chi2")["kappa"]
_m_hot = mde_kappa(compM, calibration="hotelling")["kappa"]
check("R9: the Hotelling critical value exceeds the chi-square one and the MDE "
      "rises with it",
      hotelling_crit(4, 45.9) > _st.chi2.ppf(0.95, 4) and _m_hot > _m_chi,
      f"MDE chi2 {_m_chi:.4f} -> Hotelling {_m_hot:.4f} "
      f"(+{100 * (_m_hot / _m_chi - 1):.1f}%)")

# ---- R10  Anderson-Rubin sets
_ar_c = anderson_rubin_set(compM, calibration="chi2")
_ar_h = anderson_rubin_set(compM, calibration="hotelling")
_nested = (not _ar_c["empty"] and not _ar_h["empty"]
           and _ar_h["L"] <= _ar_c["L"] + 1e-9 and _ar_h["U"] >= _ar_c["U"] - 1e-9)
check("R10: the AR set at the larger critical value contains the smaller one",
      _nested or (_ar_c["empty"] and not _ar_h["empty"]),
      f"chi2 [{_ar_c.get('L')}, {_ar_c.get('U')}], hotelling "
      f"[{_ar_h.get('L')}, {_ar_h.get('U')}]")
_ar_tiny = anderson_rubin_set(compM, calibration="chi2", alpha=1 - 1e-9)
check("R10: an AR set is EMPTY exactly when the minimised fit criterion exceeds "
      "the critical value",
      _ar_tiny["empty"] == (_ar_tiny["min_stat"] > _ar_tiny["crit"]),
      f"min_stat {_ar_tiny['min_stat']:.3f} crit {_ar_tiny['crit']:.3f} "
      f"empty {_ar_tiny['empty']}")

# ---- R11  the ordinal gradient test is invariant to any increasing recoding
_yv = np.array([0.02, -0.01, 0.05, 0.03, -0.02])
_dv = np.array([0.0, 0.2, 0.4, 0.8, 1.0])
_r1 = rank_gradient_test(_yv, _dv, two_sided=True)
_r2 = rank_gradient_test(_yv, np.exp(3 * _dv), two_sided=True)
_r3 = rank_gradient_test(_yv, rank_recoding(_dv), two_sided=True)
check("R11: the rank gradient test is invariant to strictly increasing recodings",
      abs(_r1["pvalue"] - _r2["pvalue"]) < 1e-12
      and abs(_r1["pvalue"] - _r3["pvalue"]) < 1e-12,
      f"p = {_r1['pvalue']:.4f} under all three codings")

# ---- R12  the shape test, and leave-one-category-out
_y_line = 0.1 * _dv - 0.03
_sh_line = shape_test(_y_line, _dv)
check("R12: an exactly linear profile gives R2_linear = 1 and no dummy advantage",
      abs(_sh_line["r2_linear"] - 1.0) < 1e-10
      and _sh_line["r2_dummy"] <= _sh_line["r2_linear"] + 1e-10,
      f"R2 linear {_sh_line['r2_linear']:.6f}, dummy {_sh_line['r2_dummy']:.6f}")
_y_dum = np.array([0.0, 0.0, 0.0, 0.0, -0.8])
_sh_dum = shape_test(_y_dum, _dv)
_lc_dum = loco_gradient(_y_dum, _dv)
check("R12: a one-category response is flagged by both the shape test and LOCO",
      _sh_dum["r2_dummy"] > _sh_dum["r2_linear"]
      and _lc_dum["single_category_driven"] and _lc_dum["min_ratio"] < 0.2,
      f"R2 dummy {_sh_dum['r2_dummy']:.3f} vs linear {_sh_dum['r2_linear']:.3f}; "
      f"LOCO min ratio {_lc_dum['min_ratio']:.3f}")

# ---- R13  index-class slope and bridge stability
_ic = index_class_slope(_y_dum, [_dv, _dv ** 2, rank_recoding(_dv)],
                        names=["shipped", "squared", "rank"])
check("R13: index_class_slope reports every coding and its range",
      len(_ic["rows"]) == 3 and _ic["range_ratio"] >= 1.0,
      f"range {_ic['min']:.4f} to {_ic['max']:.4f}, flips {_ic['sign_flips']}")
_bs = bridge_stability({"homicide": np.array([-0.10, -0.17, -0.30]),
                        "motor_vehicle_theft": np.array([+0.03, +0.02, -0.03])})
check("R13: bridge_stability counts sign flips against the reference bridge",
      _bs["rows"][0]["n_flip"] == 2 and not _bs["passes"],
      f"{_bs['rows'][0]['n_flip']} of 3 flip")

# ---- R14  the measurement class the bounds impose must contain the point wedge
_m_hat = np.array([-0.218, -0.070, -0.273, -0.140])
_dcs = np.array([0.29, 0.60, 0.63, 0.70])
_one = measurement_class_consistency(_m_hat, 0.25 * _dcs, sign=1)
_two_big = measurement_class_consistency(_m_hat, 1.0 * _dcs, sign=0)
_si1 = sensitivity_interval(0.02, 0.01, 0.6, 0.5, sign=1)
_si0 = sensitivity_interval(0.02, 0.01, 0.6, 0.5, sign=0)
check("R14: a wholly negative wedge refutes the one-sided class at every cap",
      _one["refuted"] and _one["n_outside"] == 4,
      f"{_one['n_outside']} of 4 outside, max violation {_one['max_violation']:.4f}")
check("R14: the two-sided class at a large cap is not refuted, and the two-sided "
      "interval is exactly twice as wide",
      (not _two_big["refuted"])
      and abs((_si0["U"] - _si0["L"]) / (_si1["U"] - _si1["L"]) - 2.0) < 1e-9,
      f"width ratio {(_si0['U'] - _si0['L']) / (_si1['U'] - _si1['L']):.6f}")

# ---- R15  exactness of the fast permutation representation
_eq_tri = equivariance_gap(dfI, stat="wald", loading="triad", outcome="Y",
                           k_pre=3, k_post=3)
check("R15: the equivariant (triad) normalisation reproduces the refit orbit",
      _eq_tri["max_abs_orbit_gap"] < 1e-8 and _eq_tri["equivariant_claimed"],
      f"gap {_eq_tri['max_abs_orbit_gap']:.2e}, p_fast {_eq_tri['p_fast']:.4f} "
      f"= p_refit {_eq_tri['p_refit']:.4f}")
_eq_iv = equivariance_gap(dfI, stat="wald", loading="mtest_iv", outcome="Y",
                          k_pre=3, k_post=3)
check("R15: the shipped mtest_iv normalisation is NOT equivariant, and the gap is "
      "measured rather than assumed away",
      (not _eq_iv["equivariant_claimed"]) and _eq_iv["max_abs_orbit_gap"] > 1e-8,
      f"gap {_eq_iv['max_abs_orbit_gap']:.4f} on a statistic of {_eq_iv['stat']:.3f}; "
      f"p_fast {_eq_iv['p_fast']:.4f} vs p_refit {_eq_iv['p_refit']:.4f}")

# ---- R16  the A10 diagnostic reports ONE pre-registered p-value
_pre = comovement_test(None, moments=moments_from_cube(_cube(4, sG2=0.25, N=800),
                                                       max_lag=1, demean=True),
                       max_lag=1, n_boot=299, seed=5)
check("R16: the pre-registered A10 p-value is the max of the two valid "
      "calibrations, and both stay in the output",
      abs(_pre["pvalue"] - max(_pre["pvalue_boot"], _pre["pvalue_satterthwaite"]))
      < 1e-12 and _pre["calibration"].startswith("max("),
      f"boot {_pre['pvalue_boot']:.4f}, Satterthwaite "
      f"{_pre['pvalue_satterthwaite']:.4f}, reported {_pre['pvalue']:.4f}")


# ---- R17-R19  the entry point a reader of the paper calls
from did_measurement import screen_design, falsification_test   # noqa: E402
from did_measurement.inference import effective_clusters        # noqa: E402

_api_cfg = dict(n_units=240, n_periods=24, cohorts=(8, 12, 16),
                cohort_shares=(0.2, 0.2, 0.2), tau_level=1.0, kappa=0.0)
_api8 = generate_panel(DGPConfig(
    n_categories=8, discretion=tuple(0.1 + 0.9 * i / 7 for i in range(8)),
    seed=11, **_api_cfg))
_api3 = generate_panel(DGPConfig(
    n_categories=3, discretion=(0.1, 0.5, 1.0), seed=11, **_api_cfg))

_v8 = screen_design(_api8, kappa_inst=0.30, k_pre=5, k_post=5)
_comp8 = gradient_components(_api8, k_pre=5, k_post=5)
check("R17: the design screen reports the same effective cluster count and "
      "number of restrictions as the components it is built from",
      abs(_v8.G_eff - effective_clusters(_comp8)) < 1e-9
      and _v8.m == len(_comp8["post"]) and _v8.C == int(_comp8["C"]),
      f"G_eff {_v8.G_eff:.3f}, m {_v8.m}, C {_v8.C}")

_v3 = screen_design(_api3, kappa_inst=0.30, k_pre=5, k_post=5)
check("R18: the resolution floor 2/C! rules the label permutation test out at "
      "C = 3 and leaves it usable at C = 8",
      (not _v3.label_permutation_usable) and _v8.label_permutation_usable
      and abs(_v3.resolution_floor - 2.0 / 6.0) < 1e-12,
      f"floor at C=3 is {_v3.resolution_floor:.4f}, at C=8 "
      f"{_v8.resolution_floor:.6f}")

_r8 = falsification_test(_api8, kappa_inst=0.30, k_pre=5, k_post=5, seed=3)
_r3 = falsification_test(_api3, kappa_inst=0.30, k_pre=5, k_post=5, seed=3)
check("R19: the reported p-value is a randomisation calibration, never the chi "
      "square one, and the floor switches C = 3 to the bootstrap",
      "chi" not in _r8.calibration.lower() and "chi" not in _r3.calibration.lower()
      and _r8.pvalue == _r8.all_pvalues["perm_wald"]
      and _r3.pvalue == _r3.all_pvalues["wcb_wald"],
      f"C=8 via {_r8.calibration.split(',')[0]}, "
      f"C=3 via {_r3.calibration.split(',')[0]}")

# ----------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
