from __future__ import annotations

from typing import Sequence

import numpy as np

from rockphysx.utils.validation import normalize_fractions


def vrh_average(volume_fractions: Sequence[float], values: Sequence[float]) -> tuple[float, float, float]:
    """Return Voigt, Reuss, and Hill averages for scalar phase properties."""
    fractions = normalize_fractions(volume_fractions)
    arr = np.asarray(values, dtype=float)
    voigt = float(np.dot(fractions, arr))
    reuss = float(1.0 / np.dot(fractions, 1.0 / arr))
    hill = 0.5 * (voigt + reuss)
    return voigt, reuss, hill


def geometric_average(volume_fractions: Sequence[float], values: Sequence[float]) -> float:
    """Return the logarithmic/geometric mixture average."""
    fractions = normalize_fractions(volume_fractions)
    arr = np.asarray(values, dtype=float)
    return float(np.exp(np.dot(fractions, np.log(arr))))


def compute_matrix_properties_from_minerals(minerals: Sequence[object]):
    """Compute effective matrix properties from mineral end-members.

    Bulk and shear moduli are mixed using the Voigt-Reuss-Hill average.
    Density is mixed linearly by mass balance.
    Thermal and electrical conductivities are mixed using the geometric mean
    as a pragmatic first-pass transport average for the solid matrix.
    """
    from rockphysx.core.parameters import MatrixProperties

    fractions = [mineral.volume_fraction for mineral in minerals]
    bulk = [mineral.bulk_modulus_gpa for mineral in minerals]
    shear = [mineral.shear_modulus_gpa for mineral in minerals]
    density = [mineral.density_gcc for mineral in minerals]
    thermal = [mineral.thermal_conductivity_wmk for mineral in minerals]
    electrical = [mineral.electrical_conductivity_sm for mineral in minerals]

    _, _, bulk_hill = vrh_average(fractions, bulk)
    _, _, shear_hill = vrh_average(fractions, shear)
    fractions_arr = normalize_fractions(fractions)
    density_mix = float(np.dot(fractions_arr, np.asarray(density, dtype=float)))
    thermal_mix = geometric_average(fractions, thermal)
    electrical_mix = geometric_average(fractions, electrical)

    return MatrixProperties(
        bulk_modulus_gpa=bulk_hill,
        shear_modulus_gpa=shear_hill,
        density_gcc=density_mix,
        thermal_conductivity_wmk=thermal_mix,
        electrical_conductivity_sm=electrical_mix,
    )
