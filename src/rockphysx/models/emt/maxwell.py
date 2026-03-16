from __future__ import annotations

from typing import Sequence

import numpy as np

from rockphysx.utils.validation import normalize_fractions


def validate_inputs_maxwell_garnett(
    volume_fractions: Sequence[float],
    thermal_conductivities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate inputs for the Maxwell-Garnett thermal-conductivity model.
    """
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(thermal_conductivities, dtype=float)

    if len(phi) != len(lam):
        raise ValueError("Volume fractions and thermal conductivities must have the same length.")
    if np.any(phi < 0.0) or np.any(lam <= 0.0):
        raise ValueError("Volume fractions must be non-negative and thermal conductivities must be positive.")

    return phi, lam


def maxwell_garnett_ncomponent(
    volume_fractions: Sequence[float],
    thermal_conductivities: Sequence[float],
    matrix_index: int = 0,
    depolarization_factors: Sequence[Sequence[float]] | tuple[float, float, float] | None = None,
) -> float | tuple[float, float, float]:
    """
    Generalized N-component Maxwell-Garnett effective-medium model.
    """
    phi, lam = validate_inputs_maxwell_garnett(volume_fractions, thermal_conductivities)
    lam0 = lam[matrix_index]

    if depolarization_factors is None:
        L = np.array([(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)] * (len(phi) - 1), dtype=float)
    elif isinstance(depolarization_factors, tuple):
        L = np.array([depolarization_factors] * (len(phi) - 1), dtype=float)
    else:
        L = np.asarray(depolarization_factors, dtype=float)

    phi_incl = np.delete(phi, matrix_index)
    lam_incl = np.delete(lam, matrix_index)

    lambda_eff = []
    for axis in range(3):
        L_axis = L[:, axis]
        beta = (lam_incl - lam0) / (lam_incl + L_axis * lam0 + 1e-10)
        sum_phi_beta = np.sum(phi_incl * beta)
        lambda_axis = lam0 * (1.0 + sum_phi_beta / (1.0 - sum_phi_beta + 1e-10))
        lambda_eff.append(float(lambda_axis))

    if np.allclose(L[:, 0], L[:, 1]) and np.allclose(L[:, 0], L[:, 2]):
        return float(lambda_eff[0])

    return tuple(lambda_eff)


def Maxwell(
    phi: Sequence[float],
    lambda_i: Sequence[float],
    alpha_i: Sequence[float] | None = None,
) -> float:
    """
    Notebook-style Maxwell wrapper.

    Matrix is assumed to be phase index 0.
    Aspect ratios are ignored here and spherical inclusions are enforced.
    """
    phi, lambda_i = validate_inputs_maxwell_garnett(phi, lambda_i)

    result = maxwell_garnett_ncomponent(
        volume_fractions=phi,
        thermal_conductivities=lambda_i,
        matrix_index=0,
        depolarization_factors=None,
    )

    return float(result) if isinstance(result, (float, int)) else float(np.mean(result))


def maxwell_garnett_isotropic(
    matrix_value: float,
    inclusion_value: float,
    inclusion_fraction: float,
) -> float:
    """
    Repo forward-solver compatibility wrapper.

    Converts the classic matrix/inclusion call into the notebook-style
    Maxwell implementation.
    """
    phi = [1.0 - inclusion_fraction, inclusion_fraction]
    lam = [matrix_value, inclusion_value]
    alpha = [1.0, 1.0]
    return Maxwell(phi, lam, alpha)