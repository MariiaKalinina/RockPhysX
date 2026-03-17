"""Sensitivity-analysis helpers for dissertation workflows."""

from rockphysx.analysis.thermal_sensitivity import (
    LocalSensitivityResult,
    MisfitGridResult,
    build_thermal_observations,
    compute_alpha_lambda_misfit_grid,
    compute_local_sensitivities,
    profile_misfit_over_alpha,
    propagate_log_variance,
    propagate_relative_uncertainty_independent,
    thermal_misfit,
)

__all__ = [
    "LocalSensitivityResult",
    "MisfitGridResult",
    "build_thermal_observations",
    "compute_alpha_lambda_misfit_grid",
    "compute_local_sensitivities",
    "profile_misfit_over_alpha",
    "propagate_log_variance",
    "propagate_relative_uncertainty_independent",
    "thermal_misfit",
]
