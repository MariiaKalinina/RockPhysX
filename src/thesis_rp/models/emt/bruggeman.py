from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import root_scalar

from thesis_rp.utils.validation import normalize_fractions


def bruggeman_isotropic(volume_fractions: Sequence[float], values: Sequence[float]) -> float:
    """Solve the isotropic Bruggeman equation for scalar properties."""
    phi = normalize_fractions(volume_fractions)
    val = np.asarray(values, dtype=float)

    def equation(x: float) -> float:
        return float(np.sum(phi * (val - x) / (val + 2.0 * x)))

    bracket = (float(np.min(val)) * 1e-6, float(np.max(val)) * 1e6)
    result = root_scalar(equation, bracket=bracket, method="brentq")
    if not result.converged:
        raise RuntimeError("Bruggeman solver did not converge.")
    return float(result.root)
