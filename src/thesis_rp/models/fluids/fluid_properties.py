from __future__ import annotations

from thesis_rp.core.parameters import FluidPhase

AIR = FluidPhase(
    name="air",
    bulk_modulus_gpa=1.0e-4,
    density_gcc=0.0012,
    thermal_conductivity_wmk=0.026,
    electrical_conductivity_sm=1.0e-12,
    viscosity_pas=1.8e-5,
)

BRINE = FluidPhase(
    name="brine",
    bulk_modulus_gpa=2.6,
    density_gcc=1.03,
    thermal_conductivity_wmk=0.60,
    electrical_conductivity_sm=3.0,
    viscosity_pas=1.0e-3,
)

OIL = FluidPhase(
    name="oil",
    bulk_modulus_gpa=1.1,
    density_gcc=0.80,
    thermal_conductivity_wmk=0.13,
    electrical_conductivity_sm=1.0e-6,
    viscosity_pas=5.0e-3,
)
