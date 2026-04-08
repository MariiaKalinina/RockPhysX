from __future__ import annotations

"""
Kuster–Toksöz (KT) effective medium for isotropic elastic moduli.

This follows the common first-order scattering / KT formulation written in an
implicit "mixing-law" form, but for a two-phase workflow (matrix + pores) it is
algebraically solvable.

Reference form (as in the user-provided paper screenshot)
---------------------------------------------------------
Bulk:
    (K_eff - K0) * (K0 + 4/3 μ0) / (K_eff + 4/3 μ0) = Σ_i x_i (K_i - K0) P_i^0

Shear:
    (μ_eff - μ0) * (μ0 + ζ0) / (μ_eff + ζ0) = Σ_i x_i (μ_i - μ0) Q_i^0

where ζ0 = μ0 (9K0 + 8μ0) / (6 (K0 + 2μ0)), and P_i^0, Q_i^0 are the geometric
factors computed in the *matrix* host (index 0).
"""

from typing import Sequence

import numpy as np

from rockphysx.models.emt.gsa_elastic import berryman_PQ_isotropic_host
from rockphysx.utils.validation import ensure_fraction, normalize_fractions


def _zeta_sphere(K: float, G: float) -> float:
    K = float(K)
    G = float(G)
    denom = 6.0 * (K + 2.0 * G)
    if denom <= 0.0 or not np.isfinite(denom):
        return float("inf")
    return float(G * (9.0 * K + 8.0 * G) / denom)


def kuster_toksoz_effective_moduli(
    volume_fractions: Sequence[float],
    bulk_moduli_gpa: Sequence[float],
    shear_moduli_gpa: Sequence[float],
    aspect_ratios: Sequence[float],
    *,
    matrix_index: int = 0,
) -> tuple[float, float]:
    """
    Multi-phase KT effective moduli for isotropic composites.

    Parameters
    ----------
    volume_fractions
        Phase volume fractions (any non-negative; normalized internally).
    bulk_moduli_gpa, shear_moduli_gpa
        Phase bulk and shear moduli (GPa).
    aspect_ratios
        Spheroidal aspect ratio for each phase representation.
        For KT the *matrix host* is fixed, so aspect ratio for the matrix phase
        is irrelevant; you can pass 1.0.
    matrix_index
        Which phase is used as the matrix host (default 0).
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

    # Geometric factors in the *matrix* host.
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

    # RHS sums exclude the matrix term (because K_i-K0 and G_i-G0 = 0 there anyway).
    rhsK = float(np.sum(c * (Kp - K0) * P0))
    rhsG = float(np.sum(c * (Gp - G0) * Q0))

    A = K0 + 4.0 * G0 / 3.0
    zeta0 = _zeta_sphere(K0, G0)
    B = G0 + zeta0

    # Solve the KT implicit fractions for K_eff, G_eff.
    denomK = A - rhsK
    denomG = B - rhsG
    if abs(denomK) <= 1e-18 or abs(denomG) <= 1e-18:
        raise RuntimeError("KT denominator too small (non-physical parameter regime).")

    K_eff = (A * K0 + rhsK * (A - K0)) / denomK
    G_eff = (B * G0 + rhsG * (B - G0)) / denomG

    if not (np.isfinite(K_eff) and np.isfinite(G_eff)) or K_eff <= 0.0 or G_eff <= 0.0:
        raise RuntimeError("KT returned non-physical effective moduli.")

    return float(K_eff), float(G_eff)


def kt_elastic_pores(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    *,
    aspect_ratio: float,
    pore_bulk_gpa: float = 0.0,
) -> tuple[float, float]:
    """
    Convenience wrapper: isotropic KT for a solid matrix with spheroidal pores.

    Notes
    -----
    - Pore shear modulus is assumed zero.
    - `pore_bulk_gpa` can be 0 (dry) or a fluid bulk modulus.
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

    return kuster_toksoz_effective_moduli(
        volume_fractions=[1.0 - phi, phi],
        bulk_moduli_gpa=[Km, Ki],
        shear_moduli_gpa=[Gm, 0.0],
        aspect_ratios=[1.0, ar],
        matrix_index=0,
    )

