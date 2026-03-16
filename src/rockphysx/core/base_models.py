from __future__ import annotations

from abc import ABC, abstractmethod

from rockphysx.core.parameters import MicrostructureParameters
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState


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
