from thesis_rp.core.parameters import FluidPhase, MicrostructureParameters, MineralPhase
from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState
from thesis_rp.forward.solver import ForwardSolver


"""
Scheme of the forward model workflow
------------------------------------

Input:
    phi      -> porosity
    X_m      -> mineral matrix properties
    X_fl     -> pore fluid properties
    theta    -> microstructural parameters
    F_p      -> selected rock-physics / EMT model

Forward problem:
    X*_p = F_p(phi, X_m, X_fl, theta)

This example runs one forward prediction for:
    p = thermal conductivity
    model = GSA
    saturation = brine
"""


quartz = MineralPhase(
    name="quartz",
    volume_fraction=0.85,
    bulk_modulus_gpa=37.0,
    shear_modulus_gpa=44.0,
    density_gcc=2.65,
    thermal_conductivity_wmk=6.5,
    electrical_conductivity_sm=1e-12,
)

clay = MineralPhase(
    name="clay",
    volume_fraction=0.15,
    bulk_modulus_gpa=20.0,
    shear_modulus_gpa=7.0,
    density_gcc=2.58,
    thermal_conductivity_wmk=2.0,
    electrical_conductivity_sm=1e-8,
)

sample = SampleDescription(
    name="synthetic_sandstone",
    porosity=0.18,
    minerals=[quartz, clay],
    fluids={
        SaturationState.DRY: FluidPhase.air(),
        SaturationState.BRINE: FluidPhase.brine(),
        SaturationState.OIL: FluidPhase.oil(),
    },
    microstructure=MicrostructureParameters(
        aspect_ratio=0.12,
        connectivity=0.65,
        orientation="isotropic",
        topology="intergranular",
    ),
)

solver = ForwardSolver()

result = solver.predict(
    "thermal_conductivity",
    sample,
    SaturationState.BRINE,
    model="gsa",
)

print("Thermal conductivity:", result)