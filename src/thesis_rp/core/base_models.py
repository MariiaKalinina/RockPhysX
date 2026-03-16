from __future__ import annotations

from abc import ABC, abstractmethod

from thesis_rp.core.parameters import MicrostructureParameters
from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState


class BaseForwardModel(ABC):
    """Abstract base class for forward property models."""

    property_name: str

    @abstractmethod
    def predict(
        self,
        sample: SampleDescription,
        saturation: SaturationState,
        microstructure: MicrostructureParameters,
    ) -> float:
        """Return a predicted effective property."""
