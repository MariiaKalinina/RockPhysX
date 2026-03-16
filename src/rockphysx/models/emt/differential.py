from __future__ import annotations

from scipy.integrate import solve_ivp

from rockphysx.models.emt.maxwell import maxwell_garnett_isotropic


def differential_scheme_scalar(
    matrix_value: float,
    inclusion_value: float,
    inclusion_fraction: float,
    *,
    n_steps: int = 100,
) -> float:
    """Simple differential effective-medium integration using Maxwell increments."""

    def rhs(phi, y):
        current = float(y[0])
        increment = 1.0 / max(1.0 - phi, 1e-8)
        mixed = maxwell_garnett_isotropic(current, inclusion_value, 1.0 / n_steps)
        return [increment * (mixed - current) * n_steps]

    result = solve_ivp(rhs, (0.0, inclusion_fraction), [matrix_value], max_step=max(inclusion_fraction / n_steps, 1e-4))
    return float(result.y[0, -1])
