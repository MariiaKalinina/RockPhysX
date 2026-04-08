from __future__ import annotations

"""
Mori–Tanaka (MT) effective medium for isotropic elastic moduli (spheroidal pores).

We implement the common MT "matrix-known" form used in the user-provided paper:

    Σ_i x_i (K_i - K_eff) P_i^0 = 0
    Σ_i x_i (μ_i - μ_eff) Q_i^0 = 0

where P_i^0 and Q_i^0 are geometric factors computed in the *matrix* host (index 0).

With fixed P_i^0, Q_i^0 this is linear in K_eff, μ_eff, so:

    K_eff = (Σ_i x_i K_i P_i^0) / (Σ_i x_i P_i^0)
    μ_eff = (Σ_i x_i μ_i Q_i^0) / (Σ_i x_i Q_i^0)
"""

from typing import Sequence

import numpy as np

from rockphysx.models.emt.gsa_elastic import berryman_PQ_isotropic_host
from rockphysx.utils.validation import ensure_fraction, normalize_fractions


def mori_tanaka_effective_moduli(
    volume_fractions: Sequence[float],
    bulk_moduli_gpa: Sequence[float],
    shear_moduli_gpa: Sequence[float],
    aspect_ratios: Sequence[float],
    *,
    matrix_index: int = 0,
) -> tuple[float, float]:
    """
    Multi-phase MT effective moduli for isotropic composites with a designated matrix host.
    """
    c = normalize_fractions(volume_fractions)
    Kp = np.asarray(list(bulk_moduli_gpa), float)
    Gp = np.asarray(list(shear_moduli_gpa), float)
    ar = np.asarray(list(aspect_ratios), float)
    if not (len(c) == len(Kp) == len(Gp) == len(ar)):
        raise ValueError("volume_fractions, bulk_moduli_gpa, shear_moduli_gpa, aspect_ratios must have same length.")
    if not (0 <= int(matrix_index) < len(c)):
        raise ValueError("matrix_index out of range.")

    K0 = float(Kp[int(matrix_index)])
    G0 = float(Gp[int(matrix_index)])
    if not (np.isfinite(K0) and np.isfinite(G0) and K0 > 0.0 and G0 > 0.0):
        raise ValueError("Matrix moduli must be positive and finite.")

    P0 = np.zeros_like(Kp, float)
    Q0 = np.zeros_like(Gp, float)
    for i in range(len(c)):
        if i == int(matrix_index):
            P0[i] = 1.0
            Q0[i] = 1.0
            continue
        P0[i], Q0[i] = berryman_PQ_isotropic_host(
            host_bulk_gpa=K0,
            host_shear_gpa=G0,
            inclusion_bulk_gpa=float(Kp[i]),
            inclusion_shear_gpa=float(Gp[i]),
            aspect_ratio=float(ar[i]),
        )

    denomK = float(np.sum(c * P0))
    denomG = float(np.sum(c * Q0))
    if abs(denomK) <= 1e-18 or abs(denomG) <= 1e-18:
        raise RuntimeError("MT denominator too small (non-physical parameter regime).")

    K_eff = float(np.sum(c * Kp * P0) / denomK)
    G_eff = float(np.sum(c * Gp * Q0) / denomG)

    if not (np.isfinite(K_eff) and np.isfinite(G_eff)) or K_eff <= 0.0 or G_eff <= 0.0:
        raise RuntimeError("MT returned non-physical effective moduli.")

    return float(K_eff), float(G_eff)


def mt_elastic_pores(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    *,
    aspect_ratio: float,
    pore_bulk_gpa: float = 0.0,
) -> tuple[float, float]:
    """
    Convenience wrapper: MT for a solid matrix with spheroidal pores.
    """
    phi = ensure_fraction(porosity, "porosity")
    Km = float(matrix_bulk_gpa)
    Gm = float(matrix_shear_gpa)
    if not (np.isfinite(Km) and np.isfinite(Gm) and Km > 0.0 and Gm > 0.0):
        raise ValueError("Matrix moduli must be positive and finite.")
    Ki = float(pore_bulk_gpa)
    if not (np.isfinite(Ki) and Ki >= 0.0):
        raise ValueError("pore_bulk_gpa must be non-negative and finite.")
    ar = float(aspect_ratio)
    if not (np.isfinite(ar) and ar > 0.0):
        raise ValueError("aspect_ratio must be positive and finite.")

    return mori_tanaka_effective_moduli(
        volume_fractions=[1.0 - phi, phi],
        bulk_moduli_gpa=[Km, Ki],
        shear_moduli_gpa=[Gm, 0.0],
        aspect_ratios=[1.0, ar],
        matrix_index=0,
    )

