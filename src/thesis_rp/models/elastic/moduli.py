from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ElasticState:
    """Elastic state container."""

    bulk_modulus_gpa: float
    shear_modulus_gpa: float
    density_gcc: float
    vp_mps: float
    vs_mps: float


def critical_porosity_dry_moduli(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    critical_porosity: float = 0.40,
) -> tuple[float, float]:
    """Critical-porosity dry moduli baseline.

    Attribution
    -----------
    This formula is reimplemented from the `EM.cripor` relation in the public
    `rockphypy` package, where dry moduli scale linearly as `1 - phi / phi_c`.
    """
    if porosity >= critical_porosity:
        return 0.0, 0.0
    factor = 1.0 - porosity / critical_porosity
    return matrix_bulk_gpa * factor, matrix_shear_gpa * factor


def gassmann_saturation(
    dry_bulk_gpa: float,
    dry_shear_gpa: float,
    matrix_bulk_gpa: float,
    fluid_bulk_gpa: float,
    porosity: float,
) -> tuple[float, float]:
    """Low-frequency Gassmann fluid substitution.

    Attribution
    -----------
    This is a clean reimplementation of the `Fluid.Gassmann` relation documented
    in `rockphypy`. Shear modulus is unchanged in the low-frequency Gassmann limit.
    """
    if porosity <= 0.0:
        return dry_bulk_gpa, dry_shear_gpa
    a_term = (1.0 - dry_bulk_gpa / matrix_bulk_gpa) ** 2
    b_term = porosity / fluid_bulk_gpa + (1.0 - porosity) / matrix_bulk_gpa - dry_bulk_gpa / (matrix_bulk_gpa**2)
    saturated_bulk = dry_bulk_gpa + a_term / b_term
    saturated_shear = dry_shear_gpa
    return saturated_bulk, saturated_shear


def saturated_elastic_properties(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    matrix_density_gcc: float,
    fluid_bulk_gpa: float,
    fluid_density_gcc: float,
    porosity: float,
    critical_porosity: float = 0.40,
):
    """Convenience function returning saturated moduli and velocities."""
    from thesis_rp.models.elastic.velocities import velocities_from_moduli

    dry_bulk, dry_shear = critical_porosity_dry_moduli(
        matrix_bulk_gpa,
        matrix_shear_gpa,
        porosity,
        critical_porosity=critical_porosity,
    )
    sat_bulk, sat_shear = gassmann_saturation(
        dry_bulk,
        dry_shear,
        matrix_bulk_gpa,
        fluid_bulk_gpa,
        porosity,
    )
    density = matrix_density_gcc * (1.0 - porosity) + fluid_density_gcc * porosity
    vp, vs = velocities_from_moduli(sat_bulk, sat_shear, density)
    return ElasticState(
        bulk_modulus_gpa=sat_bulk,
        shear_modulus_gpa=sat_shear,
        density_gcc=density,
        vp_mps=vp,
        vs_mps=vs,
    )
