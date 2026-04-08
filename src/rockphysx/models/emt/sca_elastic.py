from __future__ import annotations

"""
Elastic self-consistent (SCA/CPA) model for isotropic effective moduli.

This module implements a practical isotropic Berryman-style self-consistent /
coherent-potential approximation (CPA) for bulk and shear moduli.

Scope (important)
-----------------
This implementation targets the common dissertation workflow case:
two-phase mixtures (solid matrix + pore/fluid phase) where the pore phase has
zero shear modulus.

The full Berryman (1995) spheroidal-inclusion CPA involves more general
Eshelby/P-tensor machinery. Here we implement the widely used *spherical*
isotropic self-consistent system, solved iteratively in (K, G).

References
----------
- Berryman, J. G. (1995), "Mixture theories for rock properties"
- Mavko, Mukerji, Dvorkin (1998), "The Rock Physics Handbook"
"""

from typing import Sequence

import numpy as np

from rockphysx.utils.validation import ensure_fraction, normalize_fractions


def _ensure_non_negative(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}.")
    return v


def _zeta_sphere(K: float, G: float) -> float:
    """
    Spherical shear-polarization factor for isotropic elasticity.

    This is the standard combination that appears in isotropic HS bounds and
    self-consistent / CPA formulas for spherical inclusions:

        zeta = G * (9K + 8G) / (6 (K + 2G))
    """
    K = float(K)
    G = float(G)
    denom = 6.0 * (K + 2.0 * G)
    if denom <= 0.0:
        # Should not happen for physical moduli, but avoid division-by-zero.
        return float("inf")
    return float(G * (9.0 * K + 8.0 * G) / denom)


def berryman_self_consistent_moduli(
    volume_fractions: Sequence[float],
    bulk_moduli_gpa: Sequence[float],
    shear_moduli_gpa: Sequence[float],
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[float, float]:
    """
    Berryman-style isotropic self-consistent (CPA) estimate of (K_eff, G_eff).

    Parameters
    ----------
    volume_fractions
        Phase volume fractions (any non-negative values; normalized internally).
    bulk_moduli_gpa, shear_moduli_gpa
        Phase bulk and shear moduli in GPa. Shear may be zero (fluids/pores).
    tol, max_iter
        Iteration tolerance and maximum iterations.

    Returns
    -------
    (K_eff_gpa, G_eff_gpa)
    """
    c = normalize_fractions(volume_fractions)
    Kp = np.asarray(list(bulk_moduli_gpa), dtype=float)
    Gp = np.asarray(list(shear_moduli_gpa), dtype=float)
    if not (len(c) == len(Kp) == len(Gp)):
        raise ValueError("volume_fractions, bulk_moduli_gpa and shear_moduli_gpa must have the same length.")

    for i in range(len(Kp)):
        _ensure_non_negative(Kp[i], f"bulk_moduli_gpa[{i}]")
        _ensure_non_negative(Gp[i], f"shear_moduli_gpa[{i}]")

    # Initial guess: Voigt average (stable, positive).
    K = float(np.sum(c * Kp))
    G = float(np.sum(c * Gp))

    # Guard against exactly zero (e.g., all-fluid) to keep denominators finite.
    K = max(K, 1e-18)
    G = max(G, 1e-18)

    for _ in range(int(max_iter)):
        alpha = 4.0 * G / 3.0
        zeta = _zeta_sphere(K, G)

        denomK = Kp + alpha
        denomG = Gp + zeta

        # Avoid division by zero for pathological inputs.
        denomK = np.maximum(denomK, 1e-18)
        denomG = np.maximum(denomG, 1e-18)

        K_new = float(np.sum(c * Kp / denomK) / np.sum(c / denomK))
        G_new = float(np.sum(c * Gp / denomG) / np.sum(c / denomG))

        if abs(K_new - K) <= tol * max(1.0, abs(K)) and abs(G_new - G) <= tol * max(1.0, abs(G)):
            return float(K_new), float(G_new)

        K, G = K_new, max(G_new, 1e-18)

    raise RuntimeError(f"Self-consistent iteration did not converge within {max_iter} iterations.")


def sca_elastic_two_phase(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    inclusion_bulk_gpa: float,
    inclusion_shear_gpa: float,
    inclusion_fraction: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[float, float]:
    """
    Convenience wrapper for a two-phase isotropic self-consistent elastic model.
    """
    phi = ensure_fraction(inclusion_fraction, "inclusion_fraction")
    Km = _ensure_non_negative(matrix_bulk_gpa, "matrix_bulk_gpa")
    Gm = _ensure_non_negative(matrix_shear_gpa, "matrix_shear_gpa")
    Ki = _ensure_non_negative(inclusion_bulk_gpa, "inclusion_bulk_gpa")
    Gi = _ensure_non_negative(inclusion_shear_gpa, "inclusion_shear_gpa")
    return berryman_self_consistent_moduli(
        [1.0 - phi, phi],
        [Km, Ki],
        [Gm, Gi],
        tol=tol,
        max_iter=max_iter,
    )


def sca_elastic_pores(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    *,
    pore_bulk_gpa: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[float, float]:
    """
    Two-phase self-consistent elastic moduli for spheroidal pores approximated as spherical.

    Notes
    -----
    - Pore shear modulus is assumed zero.
    - ``pore_bulk_gpa`` may be 0 (dry pores) or a fluid bulk modulus.
    """
    phi = ensure_fraction(porosity, "porosity")
    return sca_elastic_two_phase(
        matrix_bulk_gpa=matrix_bulk_gpa,
        matrix_shear_gpa=matrix_shear_gpa,
        inclusion_bulk_gpa=pore_bulk_gpa,
        inclusion_shear_gpa=0.0,
        inclusion_fraction=phi,
        tol=tol,
        max_iter=max_iter,
    )


# -----------------------------------------------------------------------------
# Berryman self-consistent for spheroidal inclusions (P/Q formulation)
# -----------------------------------------------------------------------------

def _berryman_prolate_common(
    K_i: float,
    G_i: float,
    aspect_ratio: float,
    K_it: float,
    G_it: float,
) -> tuple[float, float]:
    """Return (P, Q) factors for a prolate spheroid (aspect_ratio > 1)."""
    ar = float(aspect_ratio)
    if ar <= 1.0:
        raise ValueError("Prolate branch requires aspect_ratio > 1.")

    # Same algebra as in the user-provided Berryman_SC implementation.
    func_teta = (ar * (ar**2 - 1.0) ** (-1.5)) * (ar * (ar**2 - 1.0) ** 0.5 - np.arccosh(ar))
    func_f = ar**2 * (1.0 - ar**2) ** (-1.0) * (3.0 * func_teta - 2.0)

    v_m = (3.0 * K_it - 2.0 * G_it) / (6.0 * K_it + 2.0 * G_it)
    R = (1.0 - 2.0 * v_m) / (2.0 * (1.0 - v_m))

    A = G_i / G_it - 1.0
    B = (1.0 / 3.0) * ((K_i / K_it) - (G_i / G_it))

    F1 = 1.0 + A * (1.5 * (func_f + func_teta) - R * (1.5 * func_f + 2.5 * func_teta - 4.0 / 3.0))
    F2 = (
        1.0
        + A * (1.0 + 1.5 * (func_f + func_teta) - 0.5 * R * (3.0 * func_f + 5.0 * func_teta))
        + B * (3.0 - 4.0 * R)
        + 0.5
        * A
        * (A + 3.0 * B)
        * (3.0 - 4.0 * R)
        * (func_f + func_teta - R * (func_f - func_teta + 2.0 * func_teta**2))
    )
    F3 = 1.0 + A * (1.0 - (func_f + 1.5 * func_teta) + R * (func_f + func_teta))
    F4 = 1.0 + 0.25 * A * (func_f + 3.0 * func_teta - R * (func_f - func_teta))
    F5 = A * (-func_f + R * (func_f + func_teta - 4.0 / 3.0)) + B * func_teta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + func_f - R * (func_f + func_teta)) + B * (1.0 - func_teta) * (3.0 - 4.0 * R)
    F7 = (
        2.0
        + 0.25 * A * (3.0 * func_f + 9.0 * func_teta - R * (3.0 * func_f + 5.0 * func_teta))
        + B * func_teta * (3.0 - 4.0 * R)
    )
    F8 = A * (1.0 - 2.0 * R + 0.5 * func_f * (R - 1.0) + 0.5 * func_teta * (5.0 * R - 3.0)) + B * (1.0 - func_teta) * (3.0 - 4.0 * R)
    F9 = A * ((R - 1.0) * func_f - R * func_teta) + B * func_teta * (3.0 - 4.0 * R)

    P = (1.0 / 3.0) * (3.0 * F1 / F2)
    Q = (1.0 / 5.0) * (2.0 / F3 + 1.0 / F4 + ((F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)))
    return float(P), float(Q)


def _berryman_oblate_common(
    K_i: float,
    G_i: float,
    aspect_ratio: float,
    K_it: float,
    G_it: float,
) -> tuple[float, float]:
    """Return (P, Q) factors for an oblate spheroid (aspect_ratio < 1)."""
    ar = float(aspect_ratio)
    if ar >= 1.0:
        raise ValueError("Oblate branch requires aspect_ratio < 1.")

    func_teta = ar / (1.0 - ar**2) ** 1.5 * (np.arccos(ar) - ar * (1.0 - ar**2) ** 0.5)
    func_f = (ar**2 / (1.0 - ar**2)) * (3.0 * func_teta - 2.0)

    v_m = (3.0 * K_it - 2.0 * G_it) / (6.0 * K_it + 2.0 * G_it)
    R = (1.0 - 2.0 * v_m) / (2.0 * (1.0 - v_m))

    A = (G_i / G_it) - 1.0
    B = (1.0 / 3.0) * ((K_i / K_it) - (G_i / G_it))

    F1 = 1.0 + A * (1.5 * (func_f + func_teta) - R * (1.5 * func_f + 2.5 * func_teta - 4.0 / 3.0))
    F2 = (
        1.0
        + A * (1.0 + 1.5 * (func_f + func_teta) - 0.5 * R * (3.0 * func_f + 5.0 * func_teta))
        + B * (3.0 - 4.0 * R)
        + 0.5
        * A
        * (A + 3.0 * B)
        * (3.0 - 4.0 * R)
        * (func_f + func_teta - R * (func_f - func_teta + 2.0 * func_teta**2))
    )
    F3 = 1.0 + A * (1.0 - (func_f + 1.5 * func_teta) + R * (func_f + func_teta))
    F4 = 1.0 + 0.25 * A * (func_f + 3.0 * func_teta - R * (func_f - func_teta))
    F5 = A * (-func_f + R * (func_f + func_teta - 4.0 / 3.0)) + B * func_teta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + func_f - R * (func_f + func_teta)) + B * (1.0 - func_teta) * (3.0 - 4.0 * R)
    F7 = (
        2.0
        + 0.25 * A * (3.0 * func_f + 9.0 * func_teta - R * (3.0 * func_f + 5.0 * func_teta))
        + B * func_teta * (3.0 - 4.0 * R)
    )
    F8 = A * (1.0 - 2.0 * R + 0.5 * func_f * (R - 1.0) + 0.5 * func_teta * (5.0 * R - 3.0)) + B * (1.0 - func_teta) * (3.0 - 4.0 * R)
    F9 = A * ((R - 1.0) * func_f - R * func_teta) + B * func_teta * (3.0 - 4.0 * R)

    P = (1.0 / 3.0) * (3.0 * F1 / F2)
    Q = (1.0 / 5.0) * (2.0 / F3 + 1.0 / F4 + ((F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)))
    return float(P), float(Q)


def berryman_self_consistent_spheroidal_two_phase(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    inclusion_bulk_gpa: float,
    inclusion_shear_gpa: float,
    inclusion_fraction: float,
    *,
    matrix_aspect_ratio: float = 1.0,
    inclusion_aspect_ratio: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 2000,
    relaxation: float = 1.0,
    initial_guess_gpa: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Berryman self-consistent (CPA) for spheroidal phases using P/Q factors.

    This follows the common isotropic spheroidal-inclusion SC formulas where each
    phase is represented by a randomly oriented spheroid with given aspect ratio.
    """
    phi = ensure_fraction(inclusion_fraction, "inclusion_fraction")
    Km = _ensure_non_negative(matrix_bulk_gpa, "matrix_bulk_gpa")
    Gm = _ensure_non_negative(matrix_shear_gpa, "matrix_shear_gpa")
    Ki = _ensure_non_negative(inclusion_bulk_gpa, "inclusion_bulk_gpa")
    Gi = _ensure_non_negative(inclusion_shear_gpa, "inclusion_shear_gpa")

    ar_m = float(matrix_aspect_ratio)
    ar_i = float(inclusion_aspect_ratio)
    if not np.isfinite(ar_m) or ar_m <= 0.0:
        raise ValueError("matrix_aspect_ratio must be positive and finite.")
    if not np.isfinite(ar_i) or ar_i <= 0.0:
        raise ValueError("inclusion_aspect_ratio must be positive and finite.")
    omega = float(relaxation)
    if not np.isfinite(omega) or omega <= 0.0 or omega > 1.0:
        raise ValueError("relaxation must lie in (0, 1].")

    c_m = 1.0 - phi
    c_i = phi

    # Initial guess: start from the matrix phase unless the caller provides a warm start.
    # Warm starts are important when tracing a continuous branch vs porosity to avoid
    # fixed-point switching for crack-like inclusions.
    if initial_guess_gpa is None:
        K0 = max(float(Km), 1e-18)
        G0 = max(float(Gm), 1e-18)
    else:
        K0 = max(_ensure_non_negative(initial_guess_gpa[0], "initial_guess_gpa[0]"), 1e-18)
        G0 = max(_ensure_non_negative(initial_guess_gpa[1], "initial_guess_gpa[1]"), 1e-18)

    def _run_fixed_point(*, omega_run: float, K_start: float, G_start: float, n_iter: int) -> tuple[bool, float, float]:
        K = float(K_start)
        G = float(G_start)
        for _ in range(int(n_iter)):
            K_it = max(float(K), 1e-18)
            G_it = max(float(G), 1e-18)

            def pq_for_phase(Kp: float, Gp: float, ar: float) -> tuple[float, float]:
                """
                Geometric strain concentration factors (P, Q).

                Uses the same closed-form expressions as the "Irina" reference code:
                a direct P/Q implementation with stable spherical limiting forms.
                """
                ar0 = float(ar)
                if not np.isfinite(ar0) or ar0 <= 0.0:
                    raise ValueError("aspect ratio must be positive and finite.")

                # Spherical limit (needed because the matrix phase is typically alpha=1).
                if np.isclose(ar0, 1.0):
                    P = (K_it + 4.0 * G_it / 3.0) / (float(Kp) + 4.0 * G_it / 3.0)
                    zeta = _zeta_sphere(K_it, G_it)
                    Q = (G_it + zeta) / (float(Gp) + zeta)
                    return float(P), float(Q)

                # Use the exact theta branches.
                if ar0 >= 1.0:
                    theta = ar0 / ((ar0 * ar0 - 1.0) ** (3.0 / 2.0)) * (
                        ar0 * (ar0 * ar0 - 1.0) ** 0.5 - np.arccosh(ar0)
                    )
                else:
                    theta = ar0 / ((1.0 - ar0 * ar0) ** (3.0 / 2.0)) * (np.arccos(ar0) - ar0 * (1.0 - ar0 * ar0) ** 0.5)

                t = float(theta)
                nu_eff = _poisson_from_KG(K_it, G_it)
                R = (1.0 - 2.0 * nu_eff) / (2.0 * (1.0 - nu_eff))

                A = float(Gp) / G_it - 1.0
                B = (1.0 / 3.0) * (float(Kp) / K_it - float(Gp) / G_it)

                f = (ar0 * ar0) / (1.0 - ar0 * ar0) * (3.0 * t - 2.0)

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
                Tijij_min = 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
                P = (1.0 / 3.0) * Tiijj
                Q = (1.0 / 5.0) * Tijij_min
                return float(P), float(Q)

            Pm, Qm = pq_for_phase(Km, Gm, ar_m)
            Pi, Qi = pq_for_phase(Ki, Gi, ar_i)

            denomK = c_m * Pm + c_i * Pi
            denomG = c_m * Qm + c_i * Qi
            if abs(denomK) <= 1e-18 or abs(denomG) <= 1e-18:
                return False, float("nan"), float("nan")

            K_raw = (c_m * Km * Pm + c_i * Ki * Pi) / denomK
            G_raw = (c_m * Gm * Qm + c_i * Gi * Qi) / denomG

            K_new = (1.0 - omega_run) * K + omega_run * K_raw
            G_new = (1.0 - omega_run) * G + omega_run * G_raw

            if not (np.isfinite(K_new) and np.isfinite(G_new)):
                return False, float("nan"), float("nan")

            if abs(K_new - K) <= tol * max(1.0, abs(K)) and abs(G_new - G) <= tol * max(1.0, abs(G)):
                return True, float(K_new), float(max(G_new, 0.0))

            K = float(max(K_new, 1e-18))
            G = float(max(G_new, 1e-18))

        return False, float(K), float(G)

    # Try the requested relaxation first.
    ok, K_eff, G_eff = _run_fixed_point(omega_run=omega, K_start=K0, G_start=G0, n_iter=max_iter)
    if ok:
        return float(K_eff), float(G_eff)

    # If user requested damping, don't override it.
    if omega < 1.0:
        raise RuntimeError(f"Spheroidal self-consistent iteration did not converge within {max_iter} iterations.")

    # Fallback: under-relaxed fixed-point to avoid limit cycles (thin cracks).
    ok, K_eff, G_eff = _run_fixed_point(omega_run=0.5, K_start=K0, G_start=G0, n_iter=max_iter)
    if ok:
        return float(K_eff), float(G_eff)

    raise RuntimeError(f"Spheroidal self-consistent iteration did not converge within {max_iter} iterations.")


def berryman_self_consistent_spheroidal_pores(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    *,
    pore_bulk_gpa: float = 0.0,
    aspect_ratio: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 2000,
    relaxation: float = 1.0,
    initial_guess_gpa: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Convenience wrapper: solid matrix (sphere) + spheroidal pores/fluids.

    Notes
    -----
    - Pore shear is assumed zero (fluid/void).
    - ``aspect_ratio`` is applied to the pore/fluid phase.
    """
    phi = ensure_fraction(porosity, "porosity")
    return berryman_self_consistent_spheroidal_two_phase(
        matrix_bulk_gpa=matrix_bulk_gpa,
        matrix_shear_gpa=matrix_shear_gpa,
        inclusion_bulk_gpa=_ensure_non_negative(pore_bulk_gpa, "pore_bulk_gpa"),
        inclusion_shear_gpa=0.0,
        inclusion_fraction=phi,
        matrix_aspect_ratio=1.0,
        inclusion_aspect_ratio=aspect_ratio,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
        initial_guess_gpa=initial_guess_gpa,
    )


# -----------------------------------------------------------------------------
# O'Connell & Budiansky (penny-shaped cracks) self-consistent approximations
# -----------------------------------------------------------------------------

def penny_crack_density_from_porosity(porosity: float, aspect_ratio: float) -> float:
    """
    Convert porosity (void volume fraction) to crack density for penny-shaped cracks.

    For penny-shaped cracks of radius a and half-thickness c (aspect ratio alpha=c/a),
    crack volume is V = (4/3) π a^2 c. With crack density parameter ε = n a^3:

        φ = n * (4/3) π a^3 α = (4/3) π α ε
        ε = 3 φ / (4 π α)
    """
    phi = ensure_fraction(porosity, "porosity")
    alpha = float(aspect_ratio)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite.")
    return float(3.0 * phi / (4.0 * np.pi * alpha))


def _bracket_root(func, x_min: float, x_max: float, n_grid: int = 200) -> tuple[float, float]:
    xs = np.linspace(float(x_min), float(x_max), int(n_grid))
    last_x: float | None = None
    last_f: float | None = None
    for x in xs:
        fx = float(func(float(x)))
        if not np.isfinite(fx):
            continue
        if fx == 0.0:
            return float(x), float(x)
        if last_f is not None and (last_f > 0) != (fx > 0):
            return float(last_x), float(x)  # type: ignore[arg-type]
        last_x, last_f = float(x), float(fx)
    raise RuntimeError("Could not bracket a root in the requested interval.")


def _bisect(func, a: float, b: float, *, tol: float = 1e-10, max_iter: int = 200) -> float:
    fa = float(func(a))
    fb = float(func(b))
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if (fa > 0) == (fb > 0):
        raise RuntimeError("Bisection requires a bracketing interval.")
    lo, hi = float(a), float(b)
    flo, fhi = fa, fb
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        fmid = float(func(mid))
        if not np.isfinite(fmid):
            # Nudge away from singularities.
            mid = np.nextafter(mid, hi)
            fmid = float(func(mid))
        if abs(fmid) <= tol:
            return float(mid)
        if (flo > 0) == (fmid > 0):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
        if abs(hi - lo) <= tol * max(1.0, abs(mid)):
            return float(0.5 * (lo + hi))
    return float(0.5 * (lo + hi))


def _poisson_from_KG(K: float, G: float) -> float:
    K = float(K)
    G = float(G)
    denom = 2.0 * (3.0 * K + G)
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.25
    return float((3.0 * K - 2.0 * G) / denom)


def oc_budiansky_sc_penny_cracks_dry(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    crack_density: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, float]:
    """
    Self-consistent effective (K, G) for randomly oriented dry penny-shaped cracks.

    Implements Eqs. (4.11.1)-(4.11.3) from the provided excerpt.
    """
    K = _ensure_non_negative(matrix_bulk_gpa, "matrix_bulk_gpa")
    G = _ensure_non_negative(matrix_shear_gpa, "matrix_shear_gpa")
    eps = float(crack_density)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("crack_density must be non-negative and finite.")
    if eps == 0.0:
        return float(K), float(G)

    nu = _poisson_from_KG(K, G)

    def eps_from_nu_sc(nu_sc: float) -> float:
        nu_sc = float(nu_sc)
        num = (nu - nu_sc) * (2.0 - nu_sc)
        den = (1.0 - nu_sc**2) * (10.0 * nu - nu_sc - 5.0 * nu * nu_sc)
        if den == 0.0:
            return float("nan")
        return float((45.0 / 16.0) * num / den)

    def f(nu_sc: float) -> float:
        return eps_from_nu_sc(nu_sc) - eps

    a, b = _bracket_root(f, -0.999, 0.499, n_grid=400)
    nu_sc = _bisect(f, a, b, tol=tol, max_iter=max_iter)

    A = (1.0 - nu_sc**2) / (1.0 - 2.0 * nu_sc)
    K_sc = K * (1.0 - (16.0 / 9.0) * A * eps)
    G_sc = G * (1.0 - (32.0 / 45.0) * ((1.0 - nu_sc) * (5.0 - nu_sc) / (2.0 - nu_sc)) * eps)
    return float(max(K_sc, 0.0)), float(max(G_sc, 0.0))


def oc_budiansky_sc_penny_cracks_fluid(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    crack_density: float,
    *,
    aspect_ratio: float,
    fluid_bulk_gpa: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, float]:
    """
    Self-consistent effective (K, G) for fluid-saturated penny-shaped cracks.

    Implements Eqs. (4.11.8)-(4.11.11) from the provided excerpt.

    Notes
    -----
    - The parameter ω is ω = K_fluid / (α K_matrix).
    - The auxiliary parameter D and ν_SC are coupled and must be solved
      consistently (O'Connell & Budiansky style). Here we eliminate D by solving
      the quadratic relation for D at a given ν_SC, then solve for ν_SC by
      bracketing/bisection on the remaining equation.
    """
    K = _ensure_non_negative(matrix_bulk_gpa, "matrix_bulk_gpa")
    G = _ensure_non_negative(matrix_shear_gpa, "matrix_shear_gpa")
    eps = float(crack_density)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("crack_density must be non-negative and finite.")
    if eps == 0.0:
        return float(K), float(G)

    alpha = float(aspect_ratio)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite.")
    Kf = _ensure_non_negative(fluid_bulk_gpa, "fluid_bulk_gpa")
    omega = float(Kf / (alpha * K)) if alpha * K > 0 else float("inf")

    nu = _poisson_from_KG(K, G)

    def solve_D(nu_sc: float) -> float:
        # O'Connell & Budiansky: quadratic relation for D (see user's reference code).
        nu_sc = float(nu_sc)
        # A0 = 9/16 * (1 - 2 ν_SC) / (1 - ν_0^2)
        denom = 1.0 - nu**2
        if denom <= 0.0:
            return float("nan")
        A0 = (9.0 / 16.0) * ((1.0 - 2.0 * nu_sc) / denom)
        # eps * D^2 - (eps + A0 + 3 ω/(4π)) D + A0 = 0
        aa = float(eps)
        bb = float(-(eps + A0 + (3.0 * omega) / (4.0 * np.pi)))
        cc = float(A0)
        disc = bb * bb - 4.0 * aa * cc
        if not np.isfinite(disc) or disc < 0.0 or aa == 0.0:
            return float("nan")
        s = float(np.sqrt(disc))
        r1 = float((-bb + s) / (2.0 * aa))
        r2 = float((-bb - s) / (2.0 * aa))
        cand = [r for r in (r1, r2) if np.isfinite(r) and 0.0 < r <= 1.0]
        if not cand:
            return float("nan")
        return float(max(cand))

    def eps_from_nu_sc(nu_sc: float) -> float:
        nu_sc = float(nu_sc)
        D = solve_D(nu_sc)
        if not np.isfinite(D):
            return float("nan")
        num = (nu - nu_sc) * (2.0 - nu_sc)
        den = (1.0 - nu_sc**2) * (D * (1.0 + 3.0 * nu) * (2.0 - nu_sc) - 2.0 * (1.0 - 2.0 * nu))
        if den == 0.0:
            return float("nan")
        return float((45.0 / 16.0) * num / den)

    def f(nu_sc: float) -> float:
        return eps_from_nu_sc(nu_sc) - eps

    try:
        a, b = _bracket_root(f, -0.999, 0.499, n_grid=800)
        nu_sc = _bisect(f, a, b, tol=tol, max_iter=max_iter)
    except RuntimeError:
        # At very high crack densities there may be no stable SC solution in the
        # admissible ν range. For plotting/sweeps, fall back to a fluid-like limit.
        return float(Kf), 0.0

    D = solve_D(nu_sc)
    A = (1.0 - nu_sc**2) / (1.0 - 2.0 * nu_sc)
    K_sc = K * (1.0 - (16.0 / 9.0) * A * D * eps)
    G_sc = G * (1.0 - (32.0 / 45.0) * ((1.0 - nu_sc) / (2.0 - nu_sc)) * (D + 3.0 / (2.0 - nu_sc)) * eps)

    # If the SC solution loses shear stability (G_sc <= 0) or goes negative, fall back
    # to a fluid-like limit (K -> Kf, G -> 0). This avoids non-physical bulk increases
    # at high crack densities where the OCB SC assumptions break down.
    if not (np.isfinite(K_sc) and np.isfinite(G_sc)) or K_sc <= 0.0 or G_sc <= 0.0:
        return float(Kf), 0.0

    return float(K_sc), float(G_sc)


def oc_budiansky_sc_penny_cracks_from_phi_alpha(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    aspect_ratio: float,
    *,
    fluid_bulk_gpa: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, float]:
    """
    Convenience wrapper: use (φ, α) to compute ε, then apply O'Connell-Budiansky SC.

    If ``fluid_bulk_gpa`` is None or 0, uses the dry-crack equations.
    Otherwise uses the fluid-saturated equations with ω = K_f / (α K).
    """
    eps = penny_crack_density_from_porosity(porosity, aspect_ratio)
    if fluid_bulk_gpa is None or float(fluid_bulk_gpa) <= 0.0:
        return oc_budiansky_sc_penny_cracks_dry(
            matrix_bulk_gpa,
            matrix_shear_gpa,
            eps,
            tol=tol,
            max_iter=max_iter,
        )
    return oc_budiansky_sc_penny_cracks_fluid(
        matrix_bulk_gpa,
        matrix_shear_gpa,
        eps,
        aspect_ratio=aspect_ratio,
        fluid_bulk_gpa=float(fluid_bulk_gpa),
        tol=tol,
        max_iter=max_iter,
    )


def sca_elastic_crack_like_pores(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    *,
    aspect_ratio: float,
    fluid_bulk_gpa: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, float]:
    """
    Self-consistent elastic moduli for crack-like pores (aspect_ratio << 1).

    This is a thin-crack specialization: randomly oriented penny-shaped cracks
    modeled with the O'Connell & Budiansky self-consistent approximation.

    Parameters
    ----------
    matrix_bulk_gpa, matrix_shear_gpa
        Uncracked solid (matrix) bulk and shear moduli in GPa.
    porosity
        Crack porosity (void volume fraction) in [0, 1].
    aspect_ratio
        Crack aspect ratio α = c/a (half-thickness / radius). For penny cracks
        α is typically very small (e.g., 1e-4 ... 1e-2).
    fluid_bulk_gpa
        If None or <= 0 -> dry cracks. Otherwise, fluid-saturated crack
        equations are used.
    tol, max_iter
        Root-finding tolerances for the SC equations.

    Returns
    -------
    (K_eff_gpa, G_eff_gpa)
        Effective bulk and shear moduli (GPa).
    """
    alpha = float(aspect_ratio)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite.")
    if alpha > 0.2:
        raise ValueError(
            "sca_elastic_crack_like_pores is intended for crack-like pores (aspect_ratio << 1). "
            f"Got aspect_ratio={alpha}."
        )
    return oc_budiansky_sc_penny_cracks_from_phi_alpha(
        matrix_bulk_gpa=matrix_bulk_gpa,
        matrix_shear_gpa=matrix_shear_gpa,
        porosity=porosity,
        aspect_ratio=alpha,
        fluid_bulk_gpa=fluid_bulk_gpa,
        tol=tol,
        max_iter=max_iter,
    )


def sca_elastic_pores_unified(
    matrix_bulk_gpa: float,
    matrix_shear_gpa: float,
    porosity: float,
    *,
    aspect_ratio: float = 1.0,
    pore_bulk_gpa: float = 0.0,
    crack_like_threshold: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 2000,
    relaxation: float = 1.0,
    initial_guess_gpa: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Unified elastic self-consistent model for pores across aspect-ratio regimes.

    Goal
    ----
    Provide a *single* elastic SC entry point for dissertation workflows:
    - for crack-like pores (alpha << 1): use O'Connell–Budiansky SC for penny cracks
    - otherwise: use Berryman spheroidal-inclusion SC (P/Q formulation)

    Parameters
    ----------
    matrix_bulk_gpa, matrix_shear_gpa
        Uncracked matrix bulk/shear moduli (GPa).
    porosity
        Porosity (fraction).
    aspect_ratio
        Pore aspect ratio alpha.
    pore_bulk_gpa
        Pore/fluid bulk modulus (GPa). If 0 -> dry pores; otherwise fluid-saturated.
    crack_like_threshold
        If > 0 and aspect_ratio <= threshold, treat pores as penny cracks.
        Default is 0.0 (disabled) because penny-crack SC assumes crack porosity
        (small φ) rather than arbitrary total porosity.
    tol, max_iter, relaxation
        Numerical controls passed to the underlying SC solvers.
    initial_guess_gpa
        Optional warm start (K, G) for the spheroidal Berryman branch.

    Returns
    -------
    (K_eff_gpa, G_eff_gpa)
    """
    alpha = float(aspect_ratio)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite.")

    if alpha <= float(crack_like_threshold):
        return sca_elastic_crack_like_pores(
            matrix_bulk_gpa=matrix_bulk_gpa,
            matrix_shear_gpa=matrix_shear_gpa,
            porosity=porosity,
            aspect_ratio=alpha,
            fluid_bulk_gpa=float(pore_bulk_gpa),
            tol=tol,
            max_iter=min(int(max_iter), 500),
        )

    return berryman_self_consistent_spheroidal_pores(
        matrix_bulk_gpa=matrix_bulk_gpa,
        matrix_shear_gpa=matrix_shear_gpa,
        porosity=porosity,
        pore_bulk_gpa=float(pore_bulk_gpa),
        aspect_ratio=alpha,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
        initial_guess_gpa=initial_guess_gpa,
    )
