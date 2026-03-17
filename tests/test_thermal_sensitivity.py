from __future__ import annotations

import numpy as np

from rockphysx.analysis.thermal_sensitivity import (
    build_thermal_observations,
    compute_alpha_lambda_misfit_grid,
    compute_local_sensitivities,
    propagate_log_variance,
    propagate_relative_uncertainty_independent,
)
from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver


def build_sample() -> SampleDescription:
    calcite = MineralPhase(
        name="calcite",
        volume_fraction=1.0,
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.71,
        thermal_conductivity_wmk=3.6,
        electrical_conductivity_sm=1e-10,
    )
    return SampleDescription(
        name="synthetic_carbonate",
        porosity=0.14,
        minerals=[calcite],
        fluids={
            SaturationState.DRY: FluidPhase.air(),
            SaturationState.BRINE: FluidPhase.brine(),
            SaturationState.OIL: FluidPhase.oil(),
        },
        matrix=MatrixProperties(
            bulk_modulus_gpa=70.0,
            shear_modulus_gpa=32.0,
            density_gcc=2.71,
            thermal_conductivity_wmk=3.8,
            electrical_conductivity_sm=1e-10,
        ),
        microstructure=MicrostructureParameters(aspect_ratio=0.08, connectivity=0.8),
    )


def test_local_sensitivities_are_finite_for_all_states() -> None:
    sample = build_sample()
    results = compute_local_sensitivities(
        sample,
        [SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL],
        epsilon=0.01,
        model="gsa",
    )
    assert len(results) == 12
    assert all(np.isfinite(item.normalized_sensitivity) for item in results)
    assert all(np.isfinite(item.absolute_sensitivity) for item in results)


def test_uncertainty_propagation_helpers_return_positive_values() -> None:
    rel_std = propagate_relative_uncertainty_independent(
        {
            "porosity": -0.8,
            "matrix_conductivity": 0.9,
            "fluid_conductivity": 0.05,
            "aspect_ratio": -0.2,
        },
        {
            "porosity": 0.05,
            "matrix_conductivity": 0.10,
            "fluid_conductivity": 0.03,
            "aspect_ratio": 0.20,
        },
    )
    assert rel_std > 0.0

    variance = propagate_log_variance(
        [0.2, 0.5, 0.1, -0.3],
        [
            [0.01, 0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0, 0.0],
            [0.0, 0.0, 0.0025, 0.0],
            [0.0, 0.0, 0.0, 0.09],
        ],
    )
    assert variance > 0.0


def test_combined_misfit_grid_minimum_recovers_synthetic_alpha_and_lambda_m() -> None:
    sample = build_sample()
    solver = ForwardSolver()

    measured = {
        sat: solver.predict("thermal_conductivity", sample, sat, model="gsa")
        for sat in (SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL)
    }
    observations = build_thermal_observations(measured, model="gsa")

    alpha_values = np.logspace(-2, 0, 9)
    lambda_m_values = np.linspace(3.4, 4.2, 9)
    grid = compute_alpha_lambda_misfit_grid(
        sample,
        observations,
        alpha_values,
        lambda_m_values,
        normalized=True,
        power=2,
        model="gsa",
    )

    min_index = np.unravel_index(np.argmin(grid.misfit), grid.misfit.shape)
    recovered_lambda_m = grid.matrix_conductivity_values[min_index[0]]
    recovered_alpha = grid.alpha_values[min_index[1]]

    assert abs(recovered_lambda_m - sample.matrix_properties.thermal_conductivity_wmk) <= 0.15
    assert abs(np.log10(recovered_alpha) - np.log10(sample.microstructure.aspect_ratio)) <= 0.15
