"""Two-layer panel DGP: latent outcomes plus a recording map that may respond to treatment.

Latent layer (never observed directly):
    F[i,t]       = f_i + g_t + tau(t - G_i) * 1{t >= G_i} + trend_break * t * 1{G_i finite} + v_it
    Ystar[i,t,c] = alpha[i,c] + lam[t,c] + theta_c * F[i,t] + eps[i,t,c]

Recording layer (observed):
    additive:        Y = Ystar + mu[i,c] + kappa * kpath(l) * d_c * D + u
    multiplicative:  Y = r * exp(Ystar), r = r0[i,c] * (1 + kappa * kpath(l) * d_c * D), clipped to (0.02, 1]
                     so log Y = log r0 + log(1 + kappa*kpath*d_c*D) + Ystar
    poisson:         Y ~ Poisson(exposure * r * exp(Ystar))   (recording thins counts)

Category c = 0 is the bridge outcome when include_bridge=True: d_0 = 0 (recording
technology invariant to treatment), loading theta_bridge on the same latent factor.

Identification-relevant structure:
  * mu[i,c] / r0[i,c] are time-invariant unit-by-category recording distortions.
    Theorem 1 of the project: they are absorbed by unit(-by-category) fixed effects,
    so only the treatment-induced shift (kappa != 0) biases DiD on recorded outcomes.
  * The bias of a recorded-outcome event study at horizon l for category c is
    kappa * kpath(l) * d_c (additive map) or log(1 + kappa*kpath(l)*d_c) (multiplicative),
    stored in df.attrs["true_meas_es"] for verification against closed forms.
  * True latent event-study effects theta_c * tau(l) are in df.attrs["true_latent_es"].

theta_d_corr tilts loadings toward the discretion index (theta_c increases in d_c),
violating the mean-independence assumption maintained by the discretion-gradient test.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

CLIP_LO, CLIP_HI = 0.02, 1.0


@dataclass
class DGPConfig:
    n_units: int = 400
    n_periods: int = 16
    cohorts: tuple = (8, 11)            # adoption periods, 0-indexed
    cohort_shares: tuple = (0.3, 0.3)   # shares of units per cohort; remainder never treated
    n_categories: int = 4               # focal outcome categories (bridge is extra)
    include_bridge: bool = True
    discretion: tuple = None            # d_c for focal cats; default linspace(0.2, 1.0)
    theta_bridge: float = 1.0
    theta_spread: float = 0.3           # focal loadings ~ evenly spaced in 1 +/- spread
    theta_d_corr: float = 0.0           # >0 tilts loadings to increase with discretion
    tau_level: float = 1.0              # latent effect size on the factor
    tau_shape: str = "constant"         # "constant" | "ramp" (phases in over 3 periods)
    kappa: float = 0.0                  # measurement-response scale
    kappa_shape: str = "constant"       # "constant" | "ramp"
    map_type: str = "additive"          # "additive" | "multiplicative" | "poisson"
    mu_scale: float = 1.0               # sd of time-invariant recording distortions
    trend_break: float = 0.0            # differential linear trend for eventual adopters
    sigma_unit: float = 1.0
    sigma_time: float = 0.5
    sigma_factor: float = 0.5
    sigma_eps: float = 1.0
    sigma_eps_bridge: float = None      # override bridge noise (weak-bridge scenarios)
    sigma_eps_by_cat: tuple = None      # per-category idiosyncratic sd, INCLUDING the
                                        # bridge when include_bridge (length n_cat).
                                        # Breaks Assumption CE (category exchangeability)
                                        # of theory/inference.md; used to price the
                                        # label-permutation test at each application's
                                        # own realised residual-variance spread.
    recording_profile: tuple = None     # per-category recording-response profile p_c
                                        # replacing d_c in the recording map (length
                                        # n_cat).  p_c = d_c reproduces A6; a 0/1 vector
                                        # gives the single-category ("dummy") response
                                        # that the CA application's shape test prefers.
    n_clusters: int = None              # units per cluster > 1 when set; a
                                        # `cluster` column is added and
                                        # gradient_components should be called
                                        # with cluster_col="cluster"
    cluster_size_dispersion: float = 0.0  # 0 equal sizes, larger is more unequal
    sigma_cluster_time: float = 0.0     # cluster-by-period shock, the source of
                                        # within-cluster dependence
    rho_eps: float = 0.0                # AR(1) on the idiosyncratic error in t
    eps_dist: str = "normal"            # "normal" | "t5" (standardised)
    sigma_u: float = 0.2                # recording noise, additive map only
    r_base: float = 0.6                 # multiplicative/poisson baseline recording rate
    exposure: float = 2000.0            # poisson exposure
    latent_scale: float = 0.25          # index scale for exp() maps to keep rates sane
    seed: int = 0


def _cluster_assignment(N, n_clusters, dispersion, rng):
    """Unit to cluster map with sizes spread geometrically at fixed total N."""
    if n_clusters is None or n_clusters >= N:
        return np.arange(N), np.ones(N, dtype=int)
    w = np.exp(dispersion * np.linspace(0.0, 1.0, n_clusters))
    sizes = np.maximum(1, np.round(w / w.sum() * N).astype(int))
    # repair rounding so the sizes sum to N exactly
    while sizes.sum() > N:
        sizes[int(np.argmax(sizes))] -= 1
    while sizes.sum() < N:
        sizes[int(np.argmin(sizes))] += 1
    lab = np.repeat(np.arange(n_clusters), sizes)
    return lab, sizes


def _ar1(shape, rho, rng, dist="normal"):
    """AR(1) in the last axis with unit stationary variance."""
    n, T = shape
    if dist == "t5":
        nu = 5.0
        z = rng.standard_t(nu, size=(n, T)) / np.sqrt(nu / (nu - 2.0))
    elif dist == "normal":
        z = rng.standard_normal((n, T))
    else:
        raise ValueError(f"unknown eps_dist {dist!r}")
    if rho == 0.0:
        return z
    out = np.empty((n, T))
    out[:, 0] = z[:, 0]
    s = np.sqrt(1.0 - rho ** 2)
    for j in range(1, T):
        out[:, j] = rho * out[:, j - 1] + s * z[:, j]
    return out


def _tau_path(cfg, n_h):
    if cfg.tau_shape == "ramp":
        return cfg.tau_level * np.minimum(1.0, (np.arange(n_h) + 1) / 3.0)
    return cfg.tau_level * np.ones(n_h)


def _kappa_path(cfg, n_h):
    if cfg.kappa_shape == "ramp":
        return np.minimum(1.0, (np.arange(n_h) + 1) / 3.0)
    return np.ones(n_h)


def generate_panel(cfg: DGPConfig) -> pd.DataFrame:
    """Simulate one panel. Returns long DataFrame with one row per (unit, period, category).

    Columns: unit, period, cohort (-1 = never treated), cat (0 = bridge if present),
    d, theta, D, event_time (NaN for never treated), Ystar (latent index), Y.
    Poisson map adds 'exposure'. Multiplicative/poisson users analyze log outcomes.
    """
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_units, cfg.n_periods

    shares = np.asarray(cfg.cohort_shares, dtype=float)
    if shares.sum() > 1.0 + 1e-9:
        raise ValueError("cohort shares exceed 1")
    n_per = np.floor(shares * N).astype(int)
    G = np.full(N, -1, dtype=int)  # -1 = never treated
    pos = 0
    for g, k in zip(cfg.cohorts, n_per):
        G[pos:pos + k] = g
        pos += k
    rng.shuffle(G)

    # --- categories ---
    C = cfg.n_categories
    d_focal = np.asarray(cfg.discretion if cfg.discretion is not None
                         else np.linspace(0.2, 1.0, C), dtype=float)
    th_focal = 1.0 + cfg.theta_spread * np.linspace(-1, 1, C)
    rng.shuffle(th_focal)  # loadings unrelated to discretion by default
    if cfg.theta_d_corr != 0.0:
        z = (d_focal - d_focal.mean()) / max(d_focal.std(), 1e-12)
        th_focal = 1.0 + cfg.theta_d_corr * 0.3 * z  # deterministic tilt
    if cfg.include_bridge:
        d_all = np.concatenate([[0.0], d_focal])
        th_all = np.concatenate([[cfg.theta_bridge], th_focal])
    else:
        d_all, th_all = d_focal, th_focal
    n_cat = len(d_all)

    # --- cluster structure ---
    clus_lab, clus_sizes = _cluster_assignment(
        N, cfg.n_clusters, cfg.cluster_size_dispersion, rng)
    n_clus = len(clus_sizes)
    if cfg.sigma_cluster_time > 0.0:
        ct = rng.normal(0, cfg.sigma_cluster_time, (n_clus, T))[clus_lab]
    else:
        ct = None

    # --- latent factor F[i,t] ---
    f_i = rng.normal(0, cfg.sigma_unit, N)
    g_t = rng.normal(0, cfg.sigma_time, T)
    v = rng.normal(0, cfg.sigma_factor, (N, T))
    per = np.arange(T)
    ell = per[None, :] - G[:, None]                      # event time
    treated_mask = (G[:, None] >= 0) & (ell >= 0)        # D[i,t]
    n_h = T  # max horizons
    tau = _tau_path(cfg, n_h)
    kpath = _kappa_path(cfg, n_h)
    tau_it = np.where(treated_mask, tau[np.clip(ell, 0, n_h - 1)], 0.0)
    drift = cfg.trend_break * per[None, :] * (G[:, None] >= 0)
    F = f_i[:, None] + g_t[None, :] + tau_it + drift + v

    # --- assemble long panel ---
    sig_eps = np.full(n_cat, cfg.sigma_eps, dtype=float)
    if cfg.include_bridge and cfg.sigma_eps_bridge is not None:
        sig_eps[0] = cfg.sigma_eps_bridge
    if cfg.sigma_eps_by_cat is not None:
        sig_eps = np.asarray(cfg.sigma_eps_by_cat, dtype=float)
        if sig_eps.shape != (n_cat,):
            raise ValueError(f"sigma_eps_by_cat must have {n_cat} entries")
        if np.any(sig_eps < 0):
            raise ValueError("sigma_eps_by_cat must be nonnegative")

    # recording-response profile: p_c multiplies kappa in the recording map.
    # Default p_c = d_c is Assumption A6 (discretion-proportional response).
    if cfg.recording_profile is None:
        prof = d_all.copy()
    else:
        prof = np.asarray(cfg.recording_profile, dtype=float)
        if prof.shape != (n_cat,):
            raise ValueError(f"recording_profile must have {n_cat} entries")

    rows = []
    D_flat = treated_mask.astype(int)
    for c in range(n_cat):
        alpha = rng.normal(0, cfg.sigma_unit, N)
        lam = rng.normal(0, cfg.sigma_time, T)
        eps = sig_eps[c] * _ar1((N, T), cfg.rho_eps, rng, cfg.eps_dist)
        if ct is not None:
            eps = eps + ct
        ystar = alpha[:, None] + lam[None, :] + th_all[c] * F + eps
        mu = rng.normal(0, cfg.mu_scale, N)              # time-invariant recording distortion
        kap_it = cfg.kappa * np.where(treated_mask, kpath[np.clip(ell, 0, n_h - 1)], 0.0) * prof[c]

        if cfg.map_type == "additive":
            u = rng.normal(0, cfg.sigma_u, (N, T))
            y = ystar + mu[:, None] + kap_it + u
        elif cfg.map_type in ("multiplicative", "poisson"):
            r0 = np.clip(cfg.r_base * np.exp(0.2 * mu), CLIP_LO, CLIP_HI)
            r = np.clip(r0[:, None] * (1.0 + kap_it), CLIP_LO, CLIP_HI)
            rate = np.exp(cfg.latent_scale * ystar)
            if cfg.map_type == "multiplicative":
                y = r * rate
            else:
                y = rng.poisson(cfg.exposure * r * rate / 1000.0).astype(float)
        else:
            raise ValueError(f"unknown map_type {cfg.map_type}")

        df_c = pd.DataFrame({
            "unit": np.repeat(np.arange(N), T),
            "cluster": np.repeat(clus_lab, T),
            "period": np.tile(per, N),
            "cohort": np.repeat(G, T),
            "cat": c,
            "d": d_all[c],
            "theta": th_all[c],
            "D": D_flat.ravel(),
            "event_time": np.where(np.repeat(G, T) >= 0, ell.ravel(), np.nan),
            "Ystar": ystar.ravel(),
            "F": F.ravel(),   # latent factor: simulation diagnostics ONLY, never estimation
            "Y": y.ravel(),
        })
        rows.append(df_c)

    df = pd.concat(rows, ignore_index=True)
    if cfg.map_type == "poisson":
        df["exposure"] = cfg.exposure

    # --- store truth for verification ---
    horizons = list(range(0, T - min(cfg.cohorts)))
    true_latent, true_meas = {}, {}
    for ci, (dc, thc) in enumerate(zip(prof, th_all)):
        if cfg.map_type == "additive":
            scale = 1.0
            meas = {h: cfg.kappa * kpath[h] * dc for h in horizons}
            lat = {h: thc * tau[h] for h in horizons}
        else:
            meas = {h: float(np.log1p(cfg.kappa * kpath[h] * dc)) for h in horizons}
            lat = {h: cfg.latent_scale * thc * tau[h] for h in horizons}
        true_latent[ci], true_meas[ci] = lat, meas
    df.attrs["true_latent_es"] = true_latent
    df.attrs["true_meas_es"] = true_meas
    df.attrs["config"] = cfg
    df.attrs["recording_profile"] = prof
    df.attrs["sigma_eps_by_cat"] = sig_eps
    df.attrs["cluster_sizes"] = clus_sizes
    return df
