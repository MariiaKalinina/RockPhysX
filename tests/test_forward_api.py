import pytest

from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
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


def test_forward_solver_basic_properties_are_positive():
    sample = make_sample()
    solver = ForwardSolver()

    thermal = solver.predict("thermal_conductivity", sample, SaturationState.BRINE)
    sigma = solver.predict("electrical_conductivity", sample, SaturationState.BRINE)
    vp = solver.predict("vp", sample, SaturationState.BRINE)
    vs = solver.predict("vs", sample, SaturationState.BRINE)
    permeability = solver.predict("permeability", sample, SaturationState.BRINE)

    assert thermal > 0.0
    assert sigma > 0.0
    assert vp > vs > 0.0
    assert permeability > 0.0


def test_resistivity_is_inverse_of_conductivity():
    sample = make_sample()
    solver = ForwardSolver()

    sigma = solver.predict("electrical_conductivity", sample, SaturationState.BRINE)
    resistivity = solver.predict("electrical_resistivity", sample, SaturationState.BRINE)

    assert resistivity == pytest.approx(1.0 / sigma)
