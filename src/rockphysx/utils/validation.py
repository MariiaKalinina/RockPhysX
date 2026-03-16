from __future__ import annotations

from typing import Iterable

import numpy as np


def ensure_fraction(value: float, name: str) -> float:
    """Validate that a scalar is in the closed interval [0, 1]."""
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")
    return float(value)


def ensure_positive(value: float, name: str, allow_zero: bool = False) -> float:
    """Validate that a scalar is strictly positive unless zero is explicitly allowed."""
    scalar = float(value)
    if allow_zero:
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value!r}.")
    else:
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}.")
    return scalar


def normalize_fractions(values: Iterable[float], *, atol: float = 1e-8) -> np.ndarray:
    """Normalize non-negative fractions to unit sum."""
    arr = np.asarray(list(values), dtype=float)
    if np.any(arr < 0.0):
        raise ValueError("Fractions must be non-negative.")
    total = float(arr.sum())
    if total <= atol:
        raise ValueError("Fractions sum to zero.")
    return arr / total


def validate_equal_lengths(*arrays: Iterable[object]) -> None:
    """Validate that all iterables have identical lengths."""
    lengths = {len(list(arr)) for arr in arrays}
    if len(lengths) > 1:
        raise ValueError(f"Length mismatch detected: {sorted(lengths)!r}.")
