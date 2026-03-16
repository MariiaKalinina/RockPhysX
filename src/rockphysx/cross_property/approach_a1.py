from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver
from rockphysx.inverse.objective import Observation
from rockphysx.inverse.parametrization_alpha import AlphaCalibrationResult, calibrate_constant_aspect_ratio


@dataclass(slots=True, frozen=True)
class CrossPropertyA1Result:
    """Result of thermal-only calibration followed by forward prediction."""

    calibration: AlphaCalibrationResult
    predictions: dict[tuple[str, SaturationState], float]


def thermal_only_calibration_then_predict(
    sample: SampleDescription,
    measured_thermal: Mapping[SaturationState, float],
    *,
    target_properties: Sequence[str] = ("electrical_conductivity", "electrical_resistivity"),
    target_saturations: Sequence[SaturationState] = (SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL),
    bounds: tuple[float, float] = (1e-3, 1.0),
    solver: ForwardSolver | None = None,
) -> CrossPropertyA1Result:
    """Approach A1: calibrate `alpha` from thermal conductivity only and predict other properties."""
    forward = solver or ForwardSolver()
    observations = [
        Observation(
            property_name="thermal_conductivity",
            saturation=saturation,
            measured_value=value,
            weight=1.0,
            model="gsa",
        )
        for saturation, value in measured_thermal.items()
    ]
    calibration = calibrate_constant_aspect_ratio(sample, observations, bounds=bounds, solver=forward)

    predictions: dict[tuple[str, SaturationState], float] = {}
    for property_name in target_properties:
        for saturation in target_saturations:
            predictions[(property_name, saturation)] = forward.predict(
                property_name,
                sample,
                saturation,
                microstructure=sample.microstructure.__class__(
                    aspect_ratio=calibration.alpha_hat,
                    orientation=sample.microstructure.orientation,
                    connectivity=sample.microstructure.connectivity,
                    topology=sample.microstructure.topology,
                    metadata=sample.microstructure.metadata,
                ),
            )
    return CrossPropertyA1Result(calibration=calibration, predictions=predictions)
