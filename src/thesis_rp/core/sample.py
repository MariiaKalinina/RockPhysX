from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from thesis_rp.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from thesis_rp.core.saturation import SaturationState
from thesis_rp.utils.validation import ensure_fraction


@dataclass(slots=True)
class SampleDescription:
    """High-level sample description used throughout the repository."""

    name: str
    porosity: float
    minerals: Sequence[MineralPhase]
    fluids: Mapping[SaturationState, FluidPhase]
    microstructure: MicrostructureParameters = field(default_factory=MicrostructureParameters)
    matrix: MatrixProperties | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        ensure_fraction(self.porosity, "porosity")
        if not self.minerals:
            raise ValueError("At least one mineral phase is required.")
        total_solid_fraction = sum(mineral.volume_fraction for mineral in self.minerals)
        if abs(total_solid_fraction - 1.0) > 1e-6:
            raise ValueError(
                "Mineral volume fractions must sum to 1 within the solid phase; "
                f"got {total_solid_fraction:.6f}."
            )

    @property
    def matrix_properties(self) -> MatrixProperties:
        """Return explicit matrix properties or compute them from mineral phases."""
        if self.matrix is not None:
            return self.matrix
        return MatrixProperties.from_minerals(self.minerals)

    def fluid_for(self, saturation: SaturationState) -> FluidPhase:
        """Return fluid properties for the requested saturation state."""
        try:
            return self.fluids[saturation]
        except KeyError as exc:
            raise KeyError(f"Fluid properties for saturation {saturation!s} are not available.") from exc
