from __future__ import annotations

from typing import Sequence

import numpy as np

from thesis_rp.utils.validation import normalize_fractions


def wiener_upper(volume_fractions: Sequence[float], values: Sequence[float]) -> float:
    """Arithmetic upper Wiener bound for scalar transport properties."""
    fractions = normalize_fractions(volume_fractions)
    arr = np.asarray(values, dtype=float)
    return float(np.dot(fractions, arr))


def wiener_lower(volume_fractions: Sequence[float], values: Sequence[float]) -> float:
    """Harmonic lower Wiener bound for scalar transport properties."""
    fractions = normalize_fractions(volume_fractions)
    arr = np.asarray(values, dtype=float)
    return float(1.0 / np.dot(fractions, 1.0 / arr))


def geometric_mean(volume_fractions: Sequence[float], values: Sequence[float]) -> float:
    """Lichtenecker/logarithmic mixture rule."""
    fractions = normalize_fractions(volume_fractions)
    arr = np.asarray(values, dtype=float)
    return float(np.exp(np.dot(fractions, np.log(arr))))


def hashin_shtrikman_lower(volume_fractions: Sequence[float], conductivities: Sequence[float]) -> float:
    """Lower Hashin-Shtrikman bound for scalar conductivity-like properties."""
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(conductivities, dtype=float)
    idx = int(np.argmin(lam))
    lam_min = lam[idx]
    total = 0.0
    for i, (p_i, lam_i) in enumerate(zip(phi, lam, strict=True)):
        if i == idx:
            continue
        total += p_i / (lam_i - lam_min)
    return float(lam_min + (1.0 - phi[idx]) / ((3.0 * lam_min) ** -1 + total))


def hashin_shtrikman_upper(volume_fractions: Sequence[float], conductivities: Sequence[float]) -> float:
    """Upper Hashin-Shtrikman bound for scalar conductivity-like properties."""
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(conductivities, dtype=float)
    idx = int(np.argmax(lam))
    lam_max = lam[idx]
    total = 0.0
    for i, (p_i, lam_i) in enumerate(zip(phi, lam, strict=True)):
        if i == idx:
            continue
        total += p_i / (lam_i - lam_max)
    return float(lam_max + (1.0 - phi[idx]) / ((3.0 * lam_max) ** -1 + total))
