from __future__ import annotations

from collections.abc import Callable

from scipy.optimize import minimize_scalar


def bounded_scalar_minimize(
    objective: Callable[[float], float],
    bounds: tuple[float, float],
):
    """Bounded scalar minimization wrapper."""
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result
