from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from rockphysx.models.matrix.mineral_mixing import compute_matrix_properties_from_minerals
from rockphysx.utils.validation import ensure_fraction, ensure_positive


@dataclass(slots=True, frozen=True)
class MineralPhase:
    """Mineral end-member properties used to construct the solid matrix.

    Volume fractions are interpreted within the *solid* fraction and are therefore
    expected to sum to one across all minerals in a sample.
    """

    name: str
    volume_fraction: float
    bulk_modulus_gpa: float
    shear_modulus_gpa: float
    density_gcc: float
    thermal_conductivity_wmk: float
    electrical_conductivity_sm: float = 1e-10

    def __post_init__(self) -> None:
        ensure_fraction(self.volume_fraction, "volume_fraction")
        ensure_positive(self.bulk_modulus_gpa, "bulk_modulus_gpa")
        ensure_positive(self.shear_modulus_gpa, "shear_modulus_gpa")
        ensure_positive(self.density_gcc, "density_gcc")
        ensure_positive(self.thermal_conductivity_wmk, "thermal_conductivity_wmk")
        ensure_positive(self.electrical_conductivity_sm, "electrical_conductivity_sm")


@dataclass(slots=True, frozen=True)
class FluidPhase:
    """Fluid properties for saturation scenarios."""

    name: str
    bulk_modulus_gpa: float
    density_gcc: float
    thermal_conductivity_wmk: float
    electrical_conductivity_sm: float
    viscosity_pas: float | None = None

    def __post_init__(self) -> None:
        ensure_positive(self.bulk_modulus_gpa, "bulk_modulus_gpa", allow_zero=True)
        ensure_positive(self.density_gcc, "density_gcc")
        ensure_positive(self.thermal_conductivity_wmk, "thermal_conductivity_wmk")
        ensure_positive(self.electrical_conductivity_sm, "electrical_conductivity_sm", allow_zero=True)
        if self.viscosity_pas is not None:
            ensure_positive(self.viscosity_pas, "viscosity_pas")

    @staticmethod
    def air() -> "FluidPhase":
        from rockphysx.models.fluids.fluid_properties import AIR
        return AIR

    @staticmethod
    def brine() -> "FluidPhase":
        from rockphysx.models.fluids.fluid_properties import BRINE
        return BRINE

    @staticmethod
    def oil() -> "FluidPhase":
        from rockphysx.models.fluids.fluid_properties import OIL
        return OIL


@dataclass(slots=True, frozen=True)
class MatrixProperties:
    """Effective properties of the mineral matrix."""

    bulk_modulus_gpa: float
    shear_modulus_gpa: float
    density_gcc: float
    thermal_conductivity_wmk: float
    electrical_conductivity_sm: float = 1e-10

    def __post_init__(self) -> None:
        ensure_positive(self.bulk_modulus_gpa, "bulk_modulus_gpa")
        ensure_positive(self.shear_modulus_gpa, "shear_modulus_gpa")
        ensure_positive(self.density_gcc, "density_gcc")
        ensure_positive(self.thermal_conductivity_wmk, "thermal_conductivity_wmk")
        ensure_positive(self.electrical_conductivity_sm, "electrical_conductivity_sm", allow_zero=True)

    @staticmethod
    def from_minerals(minerals: Sequence[MineralPhase]) -> "MatrixProperties":
        return compute_matrix_properties_from_minerals(minerals)


@dataclass(slots=True, frozen=True)
class MicrostructureParameters:
    """Shared microstructural parameters across forward, inverse, and cross-property workflows.

    Parameters
    ----------
    aspect_ratio:
        Effective pore or inclusion aspect ratio. Values below one correspond to
        crack-like or oblate pores; unity corresponds to spheres; values above one
        correspond to prolate inclusions.
    orientation:
        Orientation descriptor retained for future anisotropic workflows.
    connectivity:
        Connectivity/topology factor in [0, 1]. In this first pass it enters
        electrical conductivity and permeability calculations directly.
    topology:
        Free-text description of pore topology.
    """

    aspect_ratio: float = 1.0
    orientation: str = "isotropic"
    connectivity: float = 1.0
    topology: str = "intergranular"
    metadata: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_positive(self.aspect_ratio, "aspect_ratio")
        ensure_fraction(self.connectivity, "connectivity")


@dataclass(slots=True, frozen=True)
class AspectRatioBetaDistribution:
    """Bounded beta-distribution parameterization for future inverse workflows."""

    gamma: float
    delta: float
    lower: float = 1e-3
    upper: float = 1.0

    def __post_init__(self) -> None:
        ensure_positive(self.gamma, "gamma")
        ensure_positive(self.delta, "delta")
        ensure_positive(self.lower, "lower")
        ensure_positive(self.upper, "upper")
        if self.lower >= self.upper:
            raise ValueError("lower must be smaller than upper.")
