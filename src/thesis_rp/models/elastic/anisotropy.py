from __future__ import annotations


def anisotropy_ratio(parallel_value: float, perpendicular_value: float) -> float:
    """Return a simple anisotropy ratio `parallel / perpendicular`."""
    if perpendicular_value == 0.0:
        raise ValueError("perpendicular_value must be non-zero.")
    return parallel_value / perpendicular_value
