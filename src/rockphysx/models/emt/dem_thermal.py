from __future__ import annotations

"""
Two-phase isotropic DEM model for thermal conductivity.

Source / attribution
--------------------
This implementation follows Appendix A of:

Cilli, P. A. and Chapman, M. (2020), preprint
"Linking elastic and electrical properties of rocks using cross-property DEM"

Thermal DEM equations:
    dκ*/dφ = 3 κ* (κ2 - κ*) T(*2) / (1 - φ)
    T(*2) = (1/9) [ 4 / (κ* + κ2 + L(κ* - κ2))
                  + 1 / (κ* - L(κ* - κ2)) ]
with boundary condition:
    κ*(φ = 0) = κ1

Notes
-----
- κ1 = background / matrix conductivity
- κ2 = inclusion conductivity
- φ  = inclusion volume fraction
- L  = spheroidal depolarization factor from aspect ratio α

This file intentionally implements the classical two-phase DEM only.
The generalized multicomponent/backbone DEM is implemented separately.
"""

import numpy as np
from scipy.integrate import solve_ivp

from rockphysx.utils.validation import ensure_fraction, ensure_positive


def spheroidal_depolarization_factor(aspect_ratio: float) -> float:
    """
    Principal depolarization factor L for a spheroid.
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
    Geometric function T(*2) used in the two-phase thermal DEM equation.
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
    Right-hand side of the two-phase thermal DEM ODE.
    """
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
    Solve the classical 2-phase isotropic DEM model for thermal conductivity.

    Parameters
    ----------
    matrix_conductivity
        Background / host conductivity κ1.
    inclusion_conductivity
        Inclusion conductivity κ2.
    inclusion_fraction
        Inclusion volume fraction φ.
    aspect_ratio
        Spheroidal inclusion aspect ratio α.
    n_steps
        Resolution for numerical integration.

    Returns
    -------
    float
        Effective thermal conductivity κ*(φ).
    """
    k1 = ensure_positive(matrix_conductivity, "matrix_conductivity")
    k2 = ensure_positive(inclusion_conductivity, "inclusion_conductivity")
    phi_target = ensure_fraction(inclusion_fraction, "inclusion_fraction")

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
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"DEM integration failed: {sol.message}")

    value = float(sol.y[0, -1])
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("DEM solver returned a non-physical value.")

    return value

"""
Example usage:

from rockphysx.models.emt.dem_thermal import dem_thermal_conductivity

k_dem = dem_thermal_conductivity(
    matrix_conductivity=3.0,
    inclusion_conductivity=0.13,
    inclusion_fraction=0.25,
    aspect_ratio=0.1,
)

"""