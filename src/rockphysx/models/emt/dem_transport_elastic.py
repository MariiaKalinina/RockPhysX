
from __future__ import annotations

"""
Differential effective medium (DEM) models for isotropic transport and elastic properties.

Implemented models
------------------
1) Two-phase isotropic DEM for thermal conductivity:
       dκ*/dφ = 3 κ* (κ2 - κ*) R(*2) / (1 - φ)
   where, for randomly oriented spheroidal inclusions,
       R(*2) = (1/9) * [4 / (κ* + κ2 + L(κ* - κ2))
                      + 1 / (κ* - L(κ* - κ2))]

   This is the thermal-conductivity reinterpretation of the DEM conductivity model,
   relying on the Laplace-equation equivalence used by Cilli & Chapman (2021).

2) Two-phase isotropic DEM for elastic bulk and shear moduli:
       dK*/dφ = (K2 - K*) P(*2) / (1 - φ)
       dG*/dφ = (G2 - G*) Q(*2) / (1 - φ)

   where P(*2) and Q(*2) are the geometric strain concentration factors for
   randomly oriented spheroidal inclusions as given by Berryman (1980) and used
   in Berryman (1992), Cilli & Chapman (2021).

Notes
-----
- Phase 1 is the host/background/matrix.
- Phase 2 is the inclusion phase.
- φ is inclusion volume fraction.
- α is inclusion aspect ratio:
    α < 1 : oblate / crack-like
    α = 1 : sphere
    α > 1 : prolate / needle-like

This implementation is intended as a clean, reusable forward model for inversion workflows.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _ensure_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _ensure_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return value


def _ensure_fraction(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


# ---------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------

def spheroidal_depolarization_factor(aspect_ratio: float) -> float:
    """
    Principal depolarization factor L for a spheroid.

    Returns the factor along the symmetry axis.
    """
    alpha = _ensure_positive(aspect_ratio, "aspect_ratio")

    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-10):
        return 1.0 / 3.0

    if alpha < 1.0:
        # Oblate spheroid
        inv_a2 = 1.0 / (alpha * alpha)
        root = np.sqrt(inv_a2 - 1.0)
        return float(inv_a2 * (root - np.arctan(root)) / (root**3))

    # Prolate spheroid
    inv_a2 = 1.0 / (alpha * alpha)
    root = np.sqrt(1.0 - inv_a2)
    return float(inv_a2 * (0.5 * np.log((1.0 + root) / (1.0 - root)) - root) / (root**3))


# ---------------------------------------------------------------------
# Thermal DEM
# ---------------------------------------------------------------------

def dem_thermal_geometric_factor(
    kappa_eff: float,
    kappa_inclusion: float,
    aspect_ratio: float,
) -> float:
    """
    Geometric factor R(*2) for isotropic thermal DEM with randomly oriented spheroids.
    """
    k_eff = _ensure_positive(kappa_eff, "kappa_eff")
    k_inc = _ensure_positive(kappa_inclusion, "kappa_inclusion")
    L = spheroidal_depolarization_factor(aspect_ratio)

    denom_trans = k_eff + k_inc + L * (k_eff - k_inc)
    denom_axial = k_eff - L * (k_eff - k_inc)

    if denom_trans <= 0.0 or denom_axial <= 0.0:
        raise RuntimeError("Non-physical thermal DEM denominator encountered.")

    return float((4.0 / denom_trans + 1.0 / denom_axial) / 9.0)


def dem_thermal_rhs(
    phi: float,
    kappa_eff: float,
    *,
    kappa_inclusion: float,
    aspect_ratio: float,
) -> float:
    """
    Right-hand side of the thermal DEM ODE.
    """
    R = dem_thermal_geometric_factor(
        kappa_eff=kappa_eff,
        kappa_inclusion=kappa_inclusion,
        aspect_ratio=aspect_ratio,
    )
    return float(3.0 * kappa_eff * (kappa_inclusion - kappa_eff) * R / (1.0 - phi))


def dem_thermal_conductivity(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float,
    n_steps: int = 400,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> float:
    """
    Solve the classical two-phase isotropic DEM model for thermal conductivity.
    """
    k1 = _ensure_positive(matrix_conductivity, "matrix_conductivity")
    k2 = _ensure_positive(inclusion_conductivity, "inclusion_conductivity")
    phi_target = _ensure_fraction(inclusion_fraction, "inclusion_fraction")

    if np.isclose(phi_target, 0.0):
        return float(k1)

    phi_target = min(phi_target, 1.0 - 1e-8)

    sol = solve_ivp(
        fun=lambda phi, y: [
            dem_thermal_rhs(
                phi,
                y[0],
                kappa_inclusion=k2,
                aspect_ratio=aspect_ratio,
            )
        ],
        t_span=(0.0, phi_target),
        y0=[k1],
        t_eval=np.linspace(0.0, phi_target, n_steps),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"Thermal DEM integration failed: {sol.message}")

    value = float(sol.y[0, -1])
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("Thermal DEM returned a non-physical value.")

    return value


# ---------------------------------------------------------------------
# Elastic DEM
# ---------------------------------------------------------------------

def _theta_f_for_spheroid(aspect_ratio: float) -> Tuple[float, float]:
    """
    Auxiliary Berryman shape functions theta and f for a spheroid.
    """
    alpha = _ensure_positive(aspect_ratio, "aspect_ratio")

    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-8):
        theta = 2.0 / 3.0
        f = 0.0
        return theta, f

    if alpha < 1.0:
        # Oblate spheroid
        root = np.sqrt(1.0 - alpha**2)
        theta = alpha / (root**3) * (np.arccos(alpha) - alpha * root)
    else:
        # Prolate spheroid
        root = np.sqrt(alpha**2 - 1.0)
        theta = alpha / (root**3) * (alpha * root - np.arccosh(alpha))

    f = alpha**2 * (3.0 * theta - 2.0) / (1.0 - alpha**2)
    return float(theta), float(f)


def berryman_pq(
    matrix_bulk: float,
    matrix_shear: float,
    inclusion_bulk: float,
    inclusion_shear: float,
    aspect_ratio: float,
) -> Tuple[float, float]:
    """
    Berryman geometric strain concentration factors P and Q for
    randomly oriented spheroidal inclusions in an isotropic host.

    Parameters follow the natural order:
        matrix_bulk, matrix_shear, inclusion_bulk, inclusion_shear, aspect_ratio

    Important:
    ----------
    This order must be respected in the DEM ODE. A common bug is accidentally swapping
    bulk and shear arguments when calling the function.
    """
    Km = _ensure_positive(matrix_bulk, "matrix_bulk")
    Gm = _ensure_positive(matrix_shear, "matrix_shear")
    Ki = _ensure_nonnegative(inclusion_bulk, "inclusion_bulk")
    Gi = _ensure_nonnegative(inclusion_shear, "inclusion_shear")
    alpha = _ensure_positive(aspect_ratio, "aspect_ratio")

    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-8):
        P = (Km + 4.0 * Gm / 3.0) / (Ki + 4.0 * Gm / 3.0)
        zeta = (Gm / 6.0) * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm)
        Q = (Gm + zeta) / (Gi + zeta)
        return float(P), float(Q)

    theta, f = _theta_f_for_spheroid(alpha)

    A = Gi / Gm - 1.0
    B = (Ki / Km - Gi / Gm) / 3.0
    R = Gm / (Km + 4.0 * Gm / 3.0)

    F1 = 1.0 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = (
        1.0
        + A * (1.0 + 1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta))
        + B * (3.0 - 4.0 * R)
        + A * (A + 3.0 * B) * (1.5 - 2.0 * R) * (f + theta - R * (f - theta + 2.0 * theta**2))
    )
    F3 = 1.0 + A * (1.0 - f - 1.5 * theta + R * (f + theta))
    F4 = 1.0 + 0.25 * A * (f + 3.0 * theta - R * (f - theta))
    F5 = A * (-f + R * (f + theta - 4.0 / 3.0)) + B * theta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + f - R * (f + theta)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F7 = 2.0 + 0.25 * A * (3.0 * f + 9.0 * theta - R * (3.0 * f + 5.0 * theta)) + B * theta * (3.0 - 4.0 * R)
    F8 = A * (1.0 - 2.0 * R + 0.5 * f * (R - 1.0) + 0.5 * theta * (5.0 * R - 3.0)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F9 = A * ((R - 1.0) * f - R * theta) + B * theta * (3.0 - 4.0 * R)

    if np.isclose(F2, 0.0) or np.isclose(F3, 0.0) or np.isclose(F4, 0.0):
        raise RuntimeError("Degenerate elastic DEM denominator encountered in P/Q evaluation.")

    Tiijj = 3.0 * F1 / F2
    Tijij = Tiijj / 3.0 + 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)

    P = Tiijj / 3.0
    Q = (Tijij - P) / 5.0

    if not np.isfinite(P) or not np.isfinite(Q) or P <= 0.0 or Q <= 0.0:
        raise RuntimeError("Elastic DEM produced non-physical P/Q factors.")

    return float(P), float(Q)


@dataclass(frozen=True)
class ElasticDemResult:
    bulk_modulus: float
    shear_modulus: float
    density: float | None = None
    vp: float | None = None
    vs: float | None = None


def dem_elastic_rhs(
    phi: float,
    y: np.ndarray,
    *,
    inclusion_bulk: float,
    inclusion_shear: float,
    aspect_ratio: float,
) -> np.ndarray:
    """
    Right-hand side of the coupled isotropic elastic DEM ODE.
    """
    k_eff, g_eff = map(float, y)

    if k_eff <= 0.0 or g_eff <= 0.0:
        raise RuntimeError("Effective elastic moduli became non-physical during integration.")

    P, Q = berryman_pq(
        matrix_bulk=k_eff,
        matrix_shear=g_eff,
        inclusion_bulk=inclusion_bulk,
        inclusion_shear=inclusion_shear,
        aspect_ratio=aspect_ratio,
    )

    return np.array(
        [
            (inclusion_bulk - k_eff) * P / (1.0 - phi),
            (inclusion_shear - g_eff) * Q / (1.0 - phi),
        ],
        dtype=float,
    )


def dem_elastic_moduli(
    matrix_bulk: float,
    matrix_shear: float,
    inclusion_bulk: float,
    inclusion_shear: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float,
    n_steps: int = 500,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> ElasticDemResult:
    """
    Solve the classical two-phase isotropic DEM model for bulk and shear moduli.
    """
    K1 = _ensure_positive(matrix_bulk, "matrix_bulk")
    G1 = _ensure_positive(matrix_shear, "matrix_shear")
    K2 = _ensure_nonnegative(inclusion_bulk, "inclusion_bulk")
    G2 = _ensure_nonnegative(inclusion_shear, "inclusion_shear")
    phi_target = _ensure_fraction(inclusion_fraction, "inclusion_fraction")

    if np.isclose(phi_target, 0.0):
        return ElasticDemResult(bulk_modulus=float(K1), shear_modulus=float(G1))

    phi_target = min(phi_target, 1.0 - 1e-8)

    sol = solve_ivp(
        fun=lambda phi, y: dem_elastic_rhs(
            phi,
            y,
            inclusion_bulk=K2,
            inclusion_shear=G2,
            aspect_ratio=aspect_ratio,
        ),
        t_span=(0.0, phi_target),
        y0=np.array([K1, G1], dtype=float),
        t_eval=np.linspace(0.0, phi_target, n_steps),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"Elastic DEM integration failed: {sol.message}")

    K_eff = float(sol.y[0, -1])
    G_eff = float(sol.y[1, -1])

    if not np.isfinite(K_eff) or not np.isfinite(G_eff) or K_eff <= 0.0 or G_eff <= 0.0:
        raise RuntimeError("Elastic DEM returned non-physical effective moduli.")

    return ElasticDemResult(bulk_modulus=K_eff, shear_modulus=G_eff)


# ---------------------------------------------------------------------
# Optional density / velocity helpers
# ---------------------------------------------------------------------

def volume_average_density(
    matrix_density: float,
    inclusion_density: float,
    inclusion_fraction: float,
) -> float:
    """
    Simple linear mixture density.
    """
    rho_m = _ensure_positive(matrix_density, "matrix_density")
    rho_i = _ensure_nonnegative(inclusion_density, "inclusion_density")
    phi = _ensure_fraction(inclusion_fraction, "inclusion_fraction")
    return float((1.0 - phi) * rho_m + phi * rho_i)


def velocities_from_moduli(
    bulk_modulus: float,
    shear_modulus: float,
    density: float,
) -> Tuple[float, float]:
    """
    Convert isotropic bulk/shear moduli and density to Vp and Vs.
    """
    K = _ensure_positive(bulk_modulus, "bulk_modulus")
    G = _ensure_positive(shear_modulus, "shear_modulus")
    rho = _ensure_positive(density, "density")

    vp = math.sqrt((K + 4.0 * G / 3.0) / rho)
    vs = math.sqrt(G / rho)
    return float(vp), float(vs)


def dem_elastic_velocities(
    matrix_bulk: float,
    matrix_shear: float,
    inclusion_bulk: float,
    inclusion_shear: float,
    matrix_density: float,
    inclusion_density: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float,
    n_steps: int = 500,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> ElasticDemResult:
    """
    Solve elastic DEM and return K, G, density, Vp, Vs.
    """
    result = dem_elastic_moduli(
        matrix_bulk=matrix_bulk,
        matrix_shear=matrix_shear,
        inclusion_bulk=inclusion_bulk,
        inclusion_shear=inclusion_shear,
        inclusion_fraction=inclusion_fraction,
        aspect_ratio=aspect_ratio,
        n_steps=n_steps,
        rtol=rtol,
        atol=atol,
    )

    rho_eff = volume_average_density(
        matrix_density=matrix_density,
        inclusion_density=inclusion_density,
        inclusion_fraction=inclusion_fraction,
    )
    vp, vs = velocities_from_moduli(result.bulk_modulus, result.shear_modulus, rho_eff)

    return ElasticDemResult(
        bulk_modulus=result.bulk_modulus,
        shear_modulus=result.shear_modulus,
        density=rho_eff,
        vp=vp,
        vs=vs,
    )


# ---------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Thermal example
    k_eff = dem_thermal_conductivity(
        matrix_conductivity=3.0,
        inclusion_conductivity=0.13,
        inclusion_fraction=0.25,
        aspect_ratio=0.1,
    )
    print(f"Thermal DEM: k_eff = {k_eff:.6f}")

    # Elastic example
    elastic = dem_elastic_velocities(
        matrix_bulk=70.0e9,
        matrix_shear=30.0e9,
        inclusion_bulk=2.2e9,
        inclusion_shear=0.0,
        matrix_density=2650.0,
        inclusion_density=1000.0,
        inclusion_fraction=0.20,
        aspect_ratio=0.1,
    )
    print(
        "Elastic DEM: "
        f"K_eff = {elastic.bulk_modulus:.6e}, "
        f"G_eff = {elastic.shear_modulus:.6e}, "
        f"Vp = {elastic.vp:.3f}, Vs = {elastic.vs:.3f}"
    )
