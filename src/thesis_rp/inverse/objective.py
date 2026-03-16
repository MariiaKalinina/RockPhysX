from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from thesis_rp.core.parameters import MicrostructureParameters
from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState
from thesis_rp.forward.solver import ForwardSolver


@dataclass(slots=True, frozen=True)
class Observation:
    """Single inversion constraint."""

    property_name: str
    saturation: SaturationState
    measured_value: float
    weight: float = 1.0
    model: str | None = None


def weighted_misfit(
    aspect_ratio: float,
    sample: SampleDescription,
    observations: Sequence[Observation],
    *,
    solver: ForwardSolver | None = None,
) -> float:
    """Weighted least-squares misfit for constant-aspect-ratio inversion."""
    forward = solver or ForwardSolver()
    micro = replace(sample.microstructure, aspect_ratio=aspect_ratio)
    total = 0.0
    for obs in observations:
        predicted = forward.predict(
            obs.property_name,
            sample,
            obs.saturation,
            micro,
            model=obs.model,
        )
        residual = predicted - obs.measured_value
        total += obs.weight * residual**2
    return float(total)
