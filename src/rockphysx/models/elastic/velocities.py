from __future__ import annotations

import math


def velocities_from_moduli(
    bulk_modulus_gpa: float,
    shear_modulus_gpa: float,
    density_gcc: float,
) -> tuple[float, float]:
    """Convert moduli in GPa and density in g/cc to seismic velocities in m/s."""
    vp = math.sqrt((bulk_modulus_gpa + 4.0 * shear_modulus_gpa / 3.0) / density_gcc) * 1000.0
    vs = math.sqrt(shear_modulus_gpa / density_gcc) * 1000.0
    return vp, vs
