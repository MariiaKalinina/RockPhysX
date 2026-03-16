"""Dissertation-oriented rock physics modeling package."""

from thesis_rp.core.parameters import (
    FluidPhase,
    MatrixProperties,
    MicrostructureParameters,
    MineralPhase,
)
from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState
from thesis_rp.forward.solver import ForwardSolver

__all__ = [
    "FluidPhase",
    "MatrixProperties",
    "MicrostructureParameters",
    "MineralPhase",
    "SampleDescription",
    "SaturationState",
    "ForwardSolver",
]

__version__ = "0.1.0"
