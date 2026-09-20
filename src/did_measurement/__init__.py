"""did_measurement: DiD with treatment-induced measurement change.

Core library for the project "Did the Policy Change the Outcome or the Ruler?"
Two-layer panel model: latent outcomes Y* plus a recording map m_t(D, Y*).
"""

from .dgp import DGPConfig, generate_panel
from .estimators import event_study_twfe, att_gt_cs, cluster_ols
from .mtest import discretion_gradient_test, pretrend_test
from .bounds import (sensitivity_interval, breakdown_kappa, recording_bounds,
                     cohort_sharp_interval, aggregate_sensitivity_interval,
                     aggregate_breakdown_kappa, measurement_class_consistency,
                     index_class_bounds, breakdown_over_index_class)
from .bridge import (estimate_relative_loadings, bridge_decomposition,
                     cohort_event_studies, cohort_beta_matrix,
                     bridge_decomposition_by_cohort)
from .diagnostics import (comovement_moments, comovement_test,
                          discretion_shock_test, robust_relative_loadings)
from .efficiency import (loading_channel_variance, psi_rho_from_log,
                         eif_variance, bridge_delta_se, m2_variance,
                         m2_variance_gls, m2_zeta)
from .inference import (
    gradient_components, gradient_inference, permutation_test,
    wild_cluster_bootstrap, signflip_test, chi2_wald, f_wald,
    design_information, noncentrality, mde_kappa,
    hotelling_wald, effective_clusters, predicted_wald_size,
    wald_size_expansion, wald_size_normal_limit, max_restrictions,
    label_stabilizer_size, hotelling_crit, noncentrality_for_power_crit,
    anderson_rubin_set,
)
from .robustness import (rank_gradient_test, shape_test, loco_gradient,
                         index_class_slope, rank_recoding, perturbation_ladder,
                         bridge_stability)
from .nondiag import (rho_outer_set, rho_sharp_set, gradient_bounds_over_rho,
                      breakdown_b, rho_bias_d_shock, rho_lemma63_point)
from .exactness import permutation_refit, equivariance_gap
from .api import (screen_design, falsification_test,
                  DesignVerdict, TestResult)

__version__ = "1.0.0"

__all__ = [
    "screen_design", "falsification_test",
    "DesignVerdict", "TestResult",
    "gradient_components", "gradient_inference", "permutation_test",
    "wild_cluster_bootstrap", "signflip_test", "chi2_wald", "f_wald",
    "design_information", "noncentrality", "mde_kappa",
    "hotelling_wald", "effective_clusters", "predicted_wald_size",
    "wald_size_expansion", "wald_size_normal_limit", "max_restrictions",
    "label_stabilizer_size", "hotelling_crit", "noncentrality_for_power_crit",
    "anderson_rubin_set",
    "rank_gradient_test", "shape_test", "loco_gradient", "index_class_slope",
    "rank_recoding", "perturbation_ladder", "bridge_stability",
    "rho_outer_set", "rho_sharp_set", "gradient_bounds_over_rho",
    "breakdown_b", "rho_bias_d_shock", "rho_lemma63_point",
    "permutation_refit", "equivariance_gap",
    "measurement_class_consistency", "index_class_bounds",
    "breakdown_over_index_class",
    "DGPConfig", "generate_panel",
    "event_study_twfe", "att_gt_cs", "cluster_ols",
    "discretion_gradient_test", "pretrend_test",
    "sensitivity_interval", "breakdown_kappa", "recording_bounds",
    "cohort_sharp_interval", "aggregate_sensitivity_interval",
    "aggregate_breakdown_kappa",
    "estimate_relative_loadings", "bridge_decomposition",
    "cohort_event_studies", "cohort_beta_matrix",
    "bridge_decomposition_by_cohort",
    "comovement_moments", "comovement_test", "discretion_shock_test",
    "robust_relative_loadings",
    "loading_channel_variance", "psi_rho_from_log", "eif_variance",
    "bridge_delta_se", "m2_variance", "m2_variance_gls", "m2_zeta",
]
