from __future__ import annotations


def maxwell_garnett_isotropic(
    matrix_value: float,
    inclusion_value: float,
    inclusion_fraction: float,
) -> float:
    """Classic isotropic Maxwell-Garnett relation for spherical inclusions."""
    numerator = inclusion_value + 2.0 * matrix_value + 2.0 * inclusion_fraction * (inclusion_value - matrix_value)
    denominator = inclusion_value + 2.0 * matrix_value - inclusion_fraction * (inclusion_value - matrix_value)
    return float(matrix_value * numerator / denominator)
