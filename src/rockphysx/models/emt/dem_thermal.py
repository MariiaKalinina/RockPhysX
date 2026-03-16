from __future__ import annotations

"""
Differential effective medium (DEM) model for thermal conductivity.

Source / attribution
--------------------
This implementation follows Appendix A ("Cross-Property DEM Modelling for
Thermal Conductivity") of:

Cilli, P. A. and Chapman, M. (2020), preprint:
"Linking elastic and electrical properties of rocks using cross-property DEM"

Thermal DEM equations used here:
    dκ*/dφ = 3 κ* (κ2 - κ*) T(*2) / (1 - φ)
    T(*2) = (1/9) [ 4 / (κ* + κ2 + L(κ* - κ2))
                  + 1 / (κ* - L(κ* - κ2)) ]

Boundary condition:
    κ*(φ=0) = κ1

Notes
-----
- This is a 2-phase isotropic DEM implementation.
- κ1 is the background / host conductivity.
- κ2 is the inclusion conductivity.
- φ is the inclusion volume fraction.
- For porous rocks, κ1 is typically the matrix conductivity,
  κ2 the pore-fluid conductivity, and φ the porosity.
"""

import numpy as np
from scipy.integrate import solve_ivp

from rockphysx.utils.validation import ensure_fraction, ensure_positive


def spheroidal_depolarization_factor(aspect_ratio: float) -> float:
    """
    Return the principal depolarization factor L for a spheroid.

    Parameters
    ----------
    aspect_ratio
        Spheroidal aspect ratio alpha.
        - alpha < 1 : oblate / disk-like
        - alpha = 1 : sphere
        - alpha > 1 : prolate / needle-like
    """
    alpha = ensure_positive(aspect_ratio, "aspect_ratio")

    if np.isclose(alpha, 1.0):
        return 1.0 / 3.0

    if alpha < 1.0:
        t1 = alpha**2
        t2 = 1.0 / t1
        t4 = np.sqrt(t2 - 1.0)
        t5 = np.arctan(t4)
        t8 = t4**2
        return float(t2 * (t4 - t5) / t8 / t4)

    t1 = alpha**2
    t2 = 1.0 / t1
    t4 = np.sqrt(1.0 - t2)
    t6 = np.log(1.0 + t4)
    t9 = np.log(1.0 - t4)
    t13 = t4**2
    return float(t2 * (t6 / 2.0 - t9 / 2.0 - t4) / t13 / t4)


def dem_thermal_geometric_function(
    kappa_eff: float,
    kappa_inclusion: float,
    aspect_ratio: float,
) -> float:
    """
    Geometric function T(*2) for isotropic spheroidal inclusions.

    Implements the Appendix-A expression:
        T(*2) = (1/9) [ 4 / (κ* + κ2 + L(κ* - κ2))
                      + 1 / (κ* - L(κ* - κ2)) ]
    """
    k_eff = ensure_positive(kappa_eff, "kappa_eff")
    k2 = ensure_positive(kappa_inclusion, "kappa_inclusion")
    L = spheroidal_depolarization_factor(aspect_ratio)

    term1 = 4.0 / (k_eff + k2 + L * (k_eff - k2))
    term2 = 1.0 / (k_eff - L * (k_eff - k2))
    return float((term1 + term2) / 9.0)


def dem_thermal_rhs(
    phi: float,
    kappa_eff: float,
    *,
    kappa_inclusion: float,
    aspect_ratio: float,
) -> float:
    """
    Right-hand side of the thermal DEM ODE.

    dκ*/dφ = 3 κ* (κ2 - κ*) T(*2) / (1 - φ)
    """
    if phi >= 1.0:
        raise ValueError("phi must stay below 1 in DEM integration.")

    T_star2 = dem_thermal_geometric_function(
        kappa_eff=kappa_eff,
        kappa_inclusion=kappa_inclusion,
        aspect_ratio=aspect_ratio,
    )
    return float(3.0 * kappa_eff * (kappa_inclusion - kappa_eff) * T_star2 / (1.0 - phi))


def dem_thermal_conductivity(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float,
    n_steps: int = 400,
) -> float:
    """
    Solve the 2-phase isotropic thermal DEM model.

    Parameters
    ----------
    matrix_conductivity
        Background / host conductivity κ1.
    inclusion_conductivity
        Inclusion conductivity κ2.
    inclusion_fraction
        Inclusion volume fraction φ.
    aspect_ratio
        Spheroidal inclusion aspect ratio.
    n_steps
        Integration resolution.

    Returns
    -------
    float
        Effective thermal conductivity κ*(φ).

    Notes
    -----
    Boundary condition:
        κ*(φ=0) = κ1
    """
    k1 = ensure_positive(matrix_conductivity, "matrix_conductivity")
    k2 = ensure_positive(inclusion_conductivity, "inclusion_conductivity")
    phi_target = ensure_fraction(inclusion_fraction, "inclusion_fraction")

    if np.isclose(phi_target, 0.0):
        return float(k1)

    # Avoid the singular point exactly at phi = 1
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
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"DEM thermal conductivity integration failed: {sol.message}")

    value = float(sol.y[0, -1])
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("DEM thermal conductivity solver returned a non-physical value.")

    return value