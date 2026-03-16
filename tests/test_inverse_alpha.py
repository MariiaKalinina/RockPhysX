import pytest

from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver
from rockphysx.inverse.objective import Observation
from rockphysx.inverse.parametrization_alpha import calibrate_constant_aspect_ratio


def make_sample(aspect_ratio: float = 0.12):
    quartz = MineralPhase(
        name="quartz",
        volume_fraction=1.0,
        bulk_modulus_gpa=37.0,
        shear_modulus_gpa=44.0,
        density_gcc=2.65,
        thermal_conductivity_wmk=6.5,
        electrical_conductivity_sm=1e-8,
    )
    matrix = MatrixProperties.from_minerals([quartz])
    fluids = {
        SaturationState.DRY: FluidPhase.air(),
        SaturationState.BRINE: FluidPhase.brine(),
        SaturationState.OIL: FluidPhase.oil(),
    }
    return SampleDescription(
        name="synthetic_sandstone",
        porosity=0.18,
        minerals=[quartz],
        matrix=matrix,
        fluids=fluids,
        microstructure=MicrostructureParameters(aspect_ratio=aspect_ratio, connectivity=0.9),
    )


def test_constant_alpha_round_trip():
    true_alpha = 0.12
    sample = make_sample(aspect_ratio=true_alpha)
    solver = ForwardSolver()

    thermal_dry = solver.predict("thermal_conductivity", sample, SaturationState.DRY)
    thermal_brine = solver.predict("thermal_conductivity", sample, SaturationState.BRINE)

    inversion_sample = make_sample(aspect_ratio=0.5)
    observations = [
        Observation("thermal_conductivity", SaturationState.DRY, thermal_dry),
        Observation("thermal_conductivity", SaturationState.BRINE, thermal_brine),
    ]
    result = calibrate_constant_aspect_ratio(inversion_sample, observations, solver=solver)

    assert result.alpha_hat == pytest.approx(true_alpha, rel=0.15)
    assert result.objective_value < 1e-8
