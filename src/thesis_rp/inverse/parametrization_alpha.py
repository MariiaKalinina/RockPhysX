from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from thesis_rp.core.sample import SampleDescription
from thesis_rp.forward.solver import ForwardSolver
from thesis_rp.inverse.objective import Observation, weighted_misfit
from thesis_rp.inverse.optimizers import bounded_scalar_minimize


@dataclass(slots=True, frozen=True)
class AlphaCalibrationResult:
    """Result of constant-aspect-ratio inversion."""

    alpha_hat: float
    objective_value: float
    n_constraints: int


def calibrate_constant_aspect_ratio(
    sample: SampleDescription,
    observations: Sequence[Observation],
    *,
    bounds: tuple[float, float] = (1e-3, 1.0),
    solver: ForwardSolver | None = None,
) -> AlphaCalibrationResult:
    """Estimate the constant effective aspect ratio `alpha`."""
    if not observations:
        raise ValueError("At least one observation is required.")
    forward = solver or ForwardSolver()

    result = bounded_scalar_minimize(
        lambda alpha: weighted_misfit(alpha, sample, observations, solver=forward),
        bounds=bounds,
    )
    return AlphaCalibrationResult(
        alpha_hat=float(result.x),
        objective_value=float(result.fun),
        n_constraints=len(observations),
    )
