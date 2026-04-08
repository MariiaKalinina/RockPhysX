from __future__ import annotations

"""
General Singular Approximation (GSA) for isotropic elastic properties.

Scope
-----
This module implements the *isotropic, randomly oriented* spheroidal-inclusion
case used throughout the dissertation workflows in this repository.

For this isotropic setting, the 4th-order GSA operators reduce to two scalar
operators acting on the bulk and shear parts. Those operators can be written in
terms of Berryman's geometric strain-concentration factors (P, Q) for a
spheroidal inclusion in an isotropic comparison medium (K_c, G_c).

With comparison-body moduli (K_c, G_c), the isotropic GSA update is:

    K* =  (Σ_i c_i K_i P_i) / (Σ_i c_i P_i)
    G* =  (Σ_i c_i G_i Q_i) / (Σ_i c_i Q_i)

where (P_i, Q_i) depend on the inclusion aspect ratio and on the contrast
between phase i and the comparison body.

Comparison-body strategies
--------------------------
- "matrix": use one chosen phase as comparison body (one-shot).
- "user_defined": user-provided (K_c, G_c) (one-shot).
- "self_consistent": set (K_c, G_c) = (K*, G*) and solve by fixed-point.

Note
----
You shared a Fortran implementation that computes the singular Green tensor for
VTI comparison bodies and then isotropizes it for random orientations. For our
current thesis use case (isotropic random), the (P, Q) formulation below is the
most direct and numerically robust equivalent.
"""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from rockphysx.utils.validation import ensure_fraction


@dataclass(slots=True, frozen=True)
class ElasticPhase:
    name: str
    volume_fraction: float
    bulk_modulus_gpa: float
    shear_modulus_gpa: float
    aspect_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.volume_fraction) <= 1.0):
            raise ValueError("volume_fraction must lie in [0, 1].")
        K = float(self.bulk_modulus_gpa)
        G = float(self.shear_modulus_gpa)
        if not np.isfinite(K) or K < 0.0:
            raise ValueError("bulk_modulus_gpa must be non-negative and finite.")
        if not np.isfinite(G) or G < 0.0:
            raise ValueError("shear_modulus_gpa must be non-negative and finite.")
        ar = float(self.aspect_ratio)
        if not np.isfinite(ar) or ar <= 0.0:
            raise ValueError("aspect_ratio must be positive and finite.")


@dataclass(slots=True, frozen=True)
class ComparisonBody:
    kind: Literal["matrix", "self_consistent", "user_defined"]
    matrix_index: int = 0
    user_bulk_gpa: float | None = None
    user_shear_gpa: float | None = None


def validate_phases(phases: Iterable[ElasticPhase]) -> list[ElasticPhase]:
    phases = list(phases)
    if len(phases) < 2:
        raise ValueError("At least two phases are required.")
    s = float(sum(p.volume_fraction for p in phases))
    if not np.isclose(s, 1.0, atol=1e-8):
        raise ValueError(f"Volume fractions must sum to 1.0, got {s!r}.")
    return phases


def _poisson_from_KG(K: float, G: float) -> float:
    K = float(K)
    G = float(G)
    denom = 2.0 * (3.0 * K + G)
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.25
    return float((3.0 * K - 2.0 * G) / denom)


def _zeta_sphere(K: float, G: float) -> float:
    K = float(K)
    G = float(G)
    denom = 6.0 * (K + 2.0 * G)
    if denom <= 0.0 or not np.isfinite(denom):
        return float("inf")
    return float(G * (9.0 * K + 8.0 * G) / denom)


def berryman_PQ_isotropic_host(
    host_bulk_gpa: float,
    host_shear_gpa: float,
    inclusion_bulk_gpa: float,
    inclusion_shear_gpa: float,
    aspect_ratio: float,
) -> tuple[float, float]:
    """
    Geometric strain concentration factors (P, Q) for a spheroidal inclusion.

    This is the same algebra as the "Irina" reference implementation you
    provided, used here as a stable isotropic-random backend.
    """
    Kc = float(host_bulk_gpa)
    Gc = float(host_shear_gpa)
    Ki = float(inclusion_bulk_gpa)
    Gi = float(inclusion_shear_gpa)
    ar = float(aspect_ratio)

    Kc = max(Kc, 1e-18)
    Gc = max(Gc, 1e-18)

    if np.isclose(ar, 1.0):
        P = (Kc + 4.0 * Gc / 3.0) / (Ki + 4.0 * Gc / 3.0)
        zeta = _zeta_sphere(Kc, Gc)
        Q = (Gc + zeta) / (Gi + zeta)
        return float(P), float(Q)

    if ar >= 1.0:
        theta = ar / ((ar * ar - 1.0) ** (3.0 / 2.0)) * (ar * (ar * ar - 1.0) ** 0.5 - np.arccosh(ar))
    else:
        theta = ar / ((1.0 - ar * ar) ** (3.0 / 2.0)) * (np.arccos(ar) - ar * (1.0 - ar * ar) ** 0.5)

    t = float(theta)
    nu = _poisson_from_KG(Kc, Gc)
    R = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))

    A = Gi / Gc - 1.0
    B = (1.0 / 3.0) * (Ki / Kc - Gi / Gc)

    f = (ar * ar) / (1.0 - ar * ar) * (3.0 * t - 2.0)

    F9 = A * ((R - 1.0) * f - R * t) + B * t * (3.0 - 4.0 * R)
    F8 = A * (1.0 - 2.0 * R + 0.5 * f * (R - 1.0) + 0.5 * t * (5.0 * R - 3.0)) + B * (1.0 - t) * (3.0 - 4.0 * R)
    F7 = 2.0 + 0.25 * A * (3.0 * f + 9.0 * t - R * (3.0 * f + 5.0 * t)) + B * t * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + f - R * (f + t)) + B * (1.0 - t) * (3.0 - 4.0 * R)
    F5 = A * (-f + R * (f + t - 4.0 / 3.0)) + B * t * (3.0 - 4.0 * R)
    F4 = 1.0 + 0.25 * A * (f + 3.0 * t - R * (f - t))
    F3 = 1.0 + A * (1.0 - (f + 1.5 * t) + R * (f + t))
    F2 = (
        1.0
        + A * (1.0 + 1.5 * (f + t) - 0.5 * R * (3.0 * f + 5.0 * t))
        + B * (3.0 - 4.0 * R)
        + 0.5 * A * (A + 3.0 * B) * (3.0 - 4.0 * R) * (f + t - R * (f - t + 2.0 * t * t))
    )
    F1 = 1.0 + A * (1.5 * (f + t) - R * (1.5 * f + 2.5 * t - 4.0 / 3.0))

    Tiijj = 3.0 * F1 / F2
    Tijij = 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    P = (1.0 / 3.0) * Tiijj
    Q = (1.0 / 5.0) * Tijij
    return float(P), float(Q)


def compute_comparison_body(
    phases: list[ElasticPhase],
    comparison_body: ComparisonBody,
    current_effective_gpa: tuple[float, float] | None = None,
) -> tuple[float, float]:
    phases = validate_phases(phases)

    if comparison_body.kind == "matrix":
        p = phases[int(comparison_body.matrix_index)]
        return float(p.bulk_modulus_gpa), float(p.shear_modulus_gpa)

    if comparison_body.kind == "user_defined":
        if comparison_body.user_bulk_gpa is None or comparison_body.user_shear_gpa is None:
            raise ValueError("user_defined requires user_bulk_gpa and user_shear_gpa.")
        Kc = float(comparison_body.user_bulk_gpa)
        Gc = float(comparison_body.user_shear_gpa)
        if not np.isfinite(Kc) or Kc <= 0.0:
            raise ValueError("user_bulk_gpa must be positive and finite.")
        if not np.isfinite(Gc) or Gc <= 0.0:
            raise ValueError("user_shear_gpa must be positive and finite.")
        return Kc, Gc

    if comparison_body.kind == "self_consistent":
        if current_effective_gpa is None:
            raise ValueError("current_effective_gpa is required for self_consistent comparison body.")
        return float(current_effective_gpa[0]), float(current_effective_gpa[1])

    raise ValueError(f"Unknown comparison-body strategy {comparison_body.kind!r}.")


def homogenize_elastic_gsa_isotropic_random(
    phases: Iterable[ElasticPhase],
    comparison_body: ComparisonBody,
    *,
    tol: float = 1e-10,
    max_iter: int = 2000,
    relaxation: float = 1.0,
    initial_guess_gpa: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Isotropic-random elastic GSA estimate of (K_eff, G_eff) in GPa.

    For "matrix" and "user_defined" comparison bodies, the estimate is computed
    in a single pass. For "self_consistent", fixed-point iteration is used.
    """
    phases = validate_phases(phases)

    omega = float(relaxation)
    if not np.isfinite(omega) or omega <= 0.0 or omega > 1.0:
        raise ValueError("relaxation must lie in (0, 1].")

    if comparison_body.kind != "self_consistent":
        Kc, Gc = compute_comparison_body(phases, comparison_body)
        numK = 0.0
        denK = 0.0
        numG = 0.0
        denG = 0.0
        for p in phases:
            P, Q = berryman_PQ_isotropic_host(Kc, Gc, p.bulk_modulus_gpa, p.shear_modulus_gpa, p.aspect_ratio)
            c = float(p.volume_fraction)
            numK += c * float(p.bulk_modulus_gpa) * P
            denK += c * P
            numG += c * float(p.shear_modulus_gpa) * Q
            denG += c * Q
        return float(numK / denK), float(numG / denG)

    # Self-consistent: iterate.
    if initial_guess_gpa is None:
        K = float(sum(p.volume_fraction * p.bulk_modulus_gpa for p in phases))
        G = float(sum(p.volume_fraction * p.shear_modulus_gpa for p in phases))
    else:
        K = float(initial_guess_gpa[0])
        G = float(initial_guess_gpa[1])

    K = max(K, 1e-18)
    G = max(G, 1e-18)

    for _ in range(int(max_iter)):
        Kc, Gc = compute_comparison_body(phases, comparison_body, (K, G))

        numK = 0.0
        denK = 0.0
        numG = 0.0
        denG = 0.0
        for p in phases:
            P, Q = berryman_PQ_isotropic_host(Kc, Gc, p.bulk_modulus_gpa, p.shear_modulus_gpa, p.aspect_ratio)
            c = float(p.volume_fraction)
            numK += c * float(p.bulk_modulus_gpa) * P
            denK += c * P
            numG += c * float(p.shear_modulus_gpa) * Q
            denG += c * Q

        K_raw = float(numK / denK)
        G_raw = float(numG / denG)

        K_new = (1.0 - omega) * K + omega * K_raw
        G_new = (1.0 - omega) * G + omega * G_raw

        if abs(K_new - K) <= tol * max(1.0, abs(K)) and abs(G_new - G) <= tol * max(1.0, abs(G)):
            return float(max(K_new, 0.0)), float(max(G_new, 0.0))

        K, G = float(max(K_new, 1e-18)), float(max(G_new, 1e-18))

    raise RuntimeError(f"Elastic GSA self-consistent iteration did not converge within {max_iter} iterations.")


def make_phase(
    name: str,
    volume_fraction: float,
    bulk_modulus_gpa: float,
    shear_modulus_gpa: float,
    *,
    aspect_ratio: float = 1.0,
) -> ElasticPhase:
    return ElasticPhase(
        name=name,
        volume_fraction=float(volume_fraction),
        bulk_modulus_gpa=float(bulk_modulus_gpa),
        shear_modulus_gpa=float(shear_modulus_gpa),
        aspect_ratio=float(aspect_ratio),
    )


def two_phase_elastic_isotropic(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    inclusion_bulk_gpa: float,
    inclusion_shear_gpa: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float = 1.0,
    comparison: Literal["matrix", "self_consistent", "user_defined"] = "matrix",
    user_bulk_gpa: float | None = None,
    user_shear_gpa: float | None = None,
    **kwargs,
) -> tuple[float, float]:
    phi = ensure_fraction(inclusion_fraction, "inclusion_fraction")
    phases = [
        make_phase("matrix", 1.0 - phi, matrix_bulk_gpa, matrix_shear_gpa, aspect_ratio=1.0),
        make_phase("inclusion", phi, inclusion_bulk_gpa, inclusion_shear_gpa, aspect_ratio=aspect_ratio),
    ]
    body = ComparisonBody(kind=comparison, user_bulk_gpa=user_bulk_gpa, user_shear_gpa=user_shear_gpa)
    return homogenize_elastic_gsa_isotropic_random(phases, body, **kwargs)


__all__ = [
    "ElasticPhase",
    "ComparisonBody",
    "validate_phases",
    "berryman_PQ_isotropic_host",
    "compute_comparison_body",
    "homogenize_elastic_gsa_isotropic_random",
    "make_phase",
    "two_phase_elastic_isotropic",
]

