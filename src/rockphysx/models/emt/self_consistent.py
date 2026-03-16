from __future__ import annotations

from typing import Sequence

import numpy as np

from rockphysx.utils.validation import normalize_fractions


def spheroidal_depolarization_factor(aspect_ratio: float) -> float:
    """Return the principal depolarization factor for a spheroid.

    The implementation follows the same branch logic as the user's notebook-based
    GSA prototype: oblate pores (`alpha < 1`), spherical pores (`alpha = 1`),
    and prolate inclusions (`alpha > 1`).
    """
    alpha = float(aspect_ratio)
    if np.isclose(alpha, 1.0):
        return 1.0 / 3.0
    if alpha < 1.0:
        t1 = alpha**2
        t2 = 1.0 / t1
        t4 = np.sqrt(t2 - 1.0)
        t5 = np.arctan(t4)
        return float(t2 * (t4 - t5) / (t4**3))
    t1 = alpha**2
    t2 = 1.0 / t1
    t4 = np.sqrt(1.0 - t2)
    t6 = np.log((1.0 + t4) / (1.0 - t4))
    return float(t2 * (0.5 * t6 - t4) / (t4**3))


def gsa_effective_property(
    volume_fractions: Sequence[float],
    phase_values: Sequence[float],
    aspect_ratios: Sequence[float],
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Iterative generalized self-consistent estimate for scalar transport properties.

    Notes
    -----
    This function is a clean reimplementation of the scalar idea in the uploaded
    notebook's `GSAModel` and is intended as a dissertation-oriented transport
    backbone for thermal and electrical forward problems.
    """
    phi = normalize_fractions(volume_fractions)
    values = np.asarray(phase_values, dtype=float)
    alpha = np.asarray(aspect_ratios, dtype=float)
    if not (len(phi) == len(values) == len(alpha)):
        raise ValueError("volume_fractions, phase_values, and aspect_ratios must have the same length.")

    x_eff = float(np.average(values, weights=phi))
    for _ in range(max_iter):
        numerator = 0.0
        denominator = 0.0
        for p_i, x_i, a_i in zip(phi, values, alpha, strict=True):
            f = spheroidal_depolarization_factor(float(a_i))
            d_long = x_eff * (1.0 - f) + x_i * f
            d_trans = x_eff * (1.0 + f) / 2.0 + x_i * (1.0 - f) / 2.0
            inv_d_num = (x_i / d_long + 2.0 * x_i / d_trans) / 3.0
            inv_d_den = (1.0 / d_long + 2.0 / d_trans) / 3.0
            numerator += p_i * inv_d_num
            denominator += p_i * inv_d_den
        x_new = numerator / denominator
        if abs(x_new - x_eff) < tol:
            return float(x_new)
        x_eff = float(x_new)

    raise RuntimeError(f"GSA iteration did not converge within {max_iter} iterations.")
