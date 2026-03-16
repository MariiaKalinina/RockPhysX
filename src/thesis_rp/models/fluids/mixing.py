from __future__ import annotations

from typing import Sequence

import numpy as np

from thesis_rp.utils.validation import normalize_fractions


def wood_bulk_modulus(volume_fractions: Sequence[float], bulk_moduli_gpa: Sequence[float]) -> float:
    """Wood/Reuss average for fluid bulk modulus."""
    fractions = normalize_fractions(volume_fractions)
    values = np.asarray(bulk_moduli_gpa, dtype=float)
    return float(1.0 / np.dot(fractions, 1.0 / values))
