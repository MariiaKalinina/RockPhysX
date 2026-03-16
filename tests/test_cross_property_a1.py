import pytest

from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.cross_property.approach_a1 import thermal_only_calibration_then_predict
from rockphysx.forward.solver import ForwardSolver


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


def test_cross_property_a1_matches_direct_forward_prediction():
    true_alpha = 0.10
    sample = make_sample(aspect_ratio=true_alpha)
    solver = ForwardSolver()

    measured = {
        SaturationState.DRY: solver.predict("thermal_conductivity", sample, SaturationState.DRY),
        SaturationState.BRINE: solver.predict("thermal_conductivity", sample, SaturationState.BRINE),
    }

    result = thermal_only_calibration_then_predict(
        make_sample(aspect_ratio=0.5),
        measured,
        target_properties=("electrical_conductivity",),
        target_saturations=(SaturationState.BRINE,),
        solver=solver,
    )

    direct = solver.predict("electrical_conductivity", sample, SaturationState.BRINE)
    predicted = result.predictions[("electrical_conductivity", SaturationState.BRINE)]

    assert result.calibration.alpha_hat == pytest.approx(true_alpha, rel=0.15)
    assert predicted == pytest.approx(direct, rel=0.15)
