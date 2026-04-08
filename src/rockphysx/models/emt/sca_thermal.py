from __future__ import annotations

"""
Thermal-conductivity model for randomly arranged inclusions.

Source / attribution
--------------------
This implementation follows the isotropic random-inclusion / generalized
Clausius-Mossotti thermal-conductivity relations reproduced in the user-provided
excerpt:

- Berryman, J. G. (1995)
- Mavko, Mukerji, and Dvorkin (1998)

Specifically, the implementation follows the equations shown as:
- Eq. (9.52)
- Eq. (9.53)
- Eq. (9.54)
and the shape table shown as Table 9.17.

Important note
--------------
This module is kept separate from ``gsa_thermal.py`` on purpose.

In RockPhysX:
- this file implements the isotropic random-inclusion / SCA-style
  Clausius-Mossotti thermal model used for randomly arranged inclusions;
- ``gsa_thermal.py`` implements the separate GSA thermal-conductivity model.

These are treated as different models in the dissertation workflow.
"""

from typing import Iterable, Sequence

import numpy as np

from rockphysx.utils.validation import ensure_fraction, ensure_positive, normalize_fractions


# ---------------------------------------------------------------------
# Depolarization factors
# ---------------------------------------------------------------------

def spheroidal_depolarization_factor(aspect_ratio: float) -> float:
    """
    Return the depolarization exponent along the symmetry axis of a spheroid.

    Parameters
    ----------
    aspect_ratio
        Spheroidal aspect ratio alpha.
        - alpha < 1 : oblate / disk-like
        - alpha = 1 : sphere
        - alpha > 1 : prolate / needle-like

    Returns
    -------
    float
        Symmetry-axis depolarization factor.

    Notes
    -----
    This is the same spheroidal depolarization-factor formula already used in the
    thermal GSA implementation, but here it is used inside the isotropic
    randomly-arranged-inclusions model through Eq. (9.54).
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


def spheroidal_depolarization_triplet(aspect_ratio: float) -> tuple[float, float, float]:
    """
    Return the three depolarization exponents (L_a, L_b, L_c) for a spheroid.

    Returns
    -------
    tuple[float, float, float]
        Two equal transverse factors and one symmetry-axis factor.

    Notes
    -----
    For the isotropic random-inclusion model, the ordering of the three axes
    does not matter, because Eq. (9.54) sums over all principal axes.
    """
    lc = spheroidal_depolarization_factor(aspect_ratio)
    la = 0.5 * (1.0 - lc)
    lb = la
    return float(la), float(lb), float(lc)


def depolarization_triplet_from_shape(shape: str) -> tuple[float, float, float]:
    """
    Return exact Table-9.17 depolarization exponents for selected ideal shapes.

    Supported shapes
    ----------------
    - "sphere" -> (1/3, 1/3, 1/3)
    - "needle" -> (0, 1/2, 1/2)
    - "disk"   -> (1, 0, 0)
    """
    key = shape.strip().lower()

    if key == "sphere":
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    if key == "needle":
        return (0.0, 0.5, 0.5)
    if key == "disk":
        return (1.0, 0.0, 0.0)

    raise ValueError(f"Unsupported shape {shape!r}. Use 'sphere', 'needle', or 'disk'.")


# ---------------------------------------------------------------------
# Random-inclusion parameter R^{mi}
# ---------------------------------------------------------------------

def random_inclusion_r_parameter(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    depolarization_factors: Sequence[float],
) -> float:
    """
    Compute the random-inclusion parameter R^{mi} from Eq. (9.54).

    Eq. (9.54):
        R^{mi} = (1/9) * sum_k 1 / (L_k * lambda_i + (1 - L_k) * lambda_m)

    Parameters
    ----------
    matrix_conductivity
        Host / matrix conductivity lambda_m.
    inclusion_conductivity
        Inclusion conductivity lambda_i.
    depolarization_factors
        Three principal depolarization exponents (L_a, L_b, L_c).

    Returns
    -------
    float
        Random-inclusion parameter R^{mi}.
    """
    lambda_m = ensure_positive(matrix_conductivity, "matrix_conductivity")
    lambda_i = ensure_positive(inclusion_conductivity, "inclusion_conductivity")

    L = np.asarray(list(depolarization_factors), dtype=float)
    if L.shape != (3,):
        raise ValueError("depolarization_factors must contain exactly three values.")
    if np.any(L < 0.0) or np.any(L > 1.0):
        raise ValueError("Each depolarization factor must lie in [0, 1].")
    if not np.isclose(np.sum(L), 1.0, atol=1e-8):
        raise ValueError("Depolarization factors must sum to 1.")

    denom = L * lambda_i + (1.0 - L) * lambda_m
    return float((1.0 / 9.0) * np.sum(1.0 / denom))


# ---------------------------------------------------------------------
# Effective conductivity
# ---------------------------------------------------------------------

def sca_effective_conductivity(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float | None = None,
    depolarization_factors: Sequence[float] | None = None,
) -> float:
    """
    Effective thermal conductivity for randomly arranged inclusions.

    Parameters
    ----------
    matrix_conductivity
        Host / matrix conductivity lambda_m.
    inclusion_conductivity
        Inclusion conductivity lambda_i.
    inclusion_fraction
        Inclusion volume fraction V_i.
        In porous-rock usage, this is usually porosity phi.
    aspect_ratio
        Optional spheroidal aspect ratio used to derive (L_a, L_b, L_c).
    depolarization_factors
        Optional explicit depolarization exponents.
        If given, these override ``aspect_ratio``.

    Returns
    -------
    float
        Effective thermal conductivity.

    Notes
    -----
    This implements the algebraic solution of Eq. (9.52), i.e. Eq. (9.53).

    The code uses the equivalent numerically convenient form:

        lambda_eff = lambda_m * (1 + 2 * V_i * R^{mi} * (lambda_i - lambda_m)) \
                               / (1 -     V_i * R^{mi} * (lambda_i - lambda_m))

    which is algebraically identical to the form in the excerpt.
    """
    lambda_m = ensure_positive(matrix_conductivity, "matrix_conductivity")
    lambda_i = ensure_positive(inclusion_conductivity, "inclusion_conductivity")
    vi = ensure_fraction(inclusion_fraction, "inclusion_fraction")

    if depolarization_factors is None:
        alpha = 1.0 if aspect_ratio is None else aspect_ratio
        depolarization_factors = spheroidal_depolarization_triplet(alpha)

    r_mi = random_inclusion_r_parameter(
        lambda_m,
        lambda_i,
        depolarization_factors,
    )

    delta = lambda_i - lambda_m
    denominator = 1.0 - vi * r_mi * delta

    if denominator <= 0.0 or np.isclose(denominator, 0.0, atol=1e-12):
        raise ValueError(
            "The SCA/random-inclusion denominator is non-positive or too close to zero. "
            "Check the conductivity contrast and inclusion fraction."
        )

    return float(lambda_m * (1.0 + 2.0 * vi * r_mi * delta) / denominator)


def sca_effective_conductivity_by_shape(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    inclusion_fraction: float,
    *,
    shape: str,
) -> float:
    """
    Convenience wrapper using exact Table-9.17 depolarization exponents.

    Examples
    --------
    shape="sphere"
    shape="needle"
    shape="disk"
    """
    return sca_effective_conductivity(
        matrix_conductivity,
        inclusion_conductivity,
        inclusion_fraction,
        depolarization_factors=depolarization_triplet_from_shape(shape),
    )


# ---------------------------------------------------------------------
# Self-consistent transport (Torquato-style)
# ---------------------------------------------------------------------

def _field_concentration_factor_torquato(
    eps_m: float,
    eps_i: float,
    depolarization_factors: Sequence[float],
) -> float:
    """
    Torquato (2002) field concentration factor R^{mi}.

        R^{mi} = (eps_m/3) * sum_{j=a,b,c} 1 / (eps_m + L_j (eps_i - eps_m))
    """
    eps_m = ensure_positive(eps_m, "eps_m")
    eps_i = ensure_positive(eps_i, "eps_i", allow_zero=True)

    L = np.asarray(list(depolarization_factors), dtype=float)
    if L.shape != (3,):
        raise ValueError("depolarization_factors must contain exactly three values.")
    if np.any(L < 0.0) or np.any(L > 1.0):
        raise ValueError("Each depolarization factor must lie in [0, 1].")
    if not np.isclose(np.sum(L), 1.0, atol=1e-8):
        raise ValueError("Depolarization factors must sum to 1.")

    denom = eps_m + L * (eps_i - eps_m)  # = eps_m*(1-L) + L*eps_i
    denom = np.maximum(denom, 1e-30)
    return float((eps_m / 3.0) * np.sum(1.0 / denom))


def sca_self_consistent_effective_conductivity(
    volume_fractions: Sequence[float],
    conductivities: Sequence[float],
    *,
    depolarization_factors_list: Sequence[Sequence[float]],
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """
    Self-consistent (SC) effective conductivity for randomly oriented ellipsoids.

    This solves the implicit Torquato (2002) SC equation:

        sum_i f_i (eps_i - eps_sc) R_i^*(eps_sc) = 0

    where the star indicates eps_m -> eps_sc in the concentration factor.
    """
    f = normalize_fractions(volume_fractions)
    eps = np.asarray(list(conductivities), dtype=float)
    if eps.shape != f.shape:
        raise ValueError("volume_fractions and conductivities must have the same length.")
    if np.any(~np.isfinite(eps)) or np.any(eps < 0.0):
        raise ValueError("conductivities must be finite and non-negative.")
    if len(depolarization_factors_list) != len(f):
        raise ValueError("depolarization_factors_list must have the same length as phases.")

    eps_min = float(np.min(eps))
    eps_max = float(np.max(eps))
    if eps_max <= 0.0:
        return 0.0

    def F(x: float) -> float:
        x = float(x)
        if x <= 0.0 or not np.isfinite(x):
            return float("nan")
        s = 0.0
        for fi, ei, L in zip(f, eps, depolarization_factors_list, strict=True):
            Ri = _field_concentration_factor_torquato(x, float(ei), L)
            s += float(fi) * (float(ei) - x) * Ri
        return float(s)

    # Prefer bracketing inside [min, max] when possible.
    a = max(eps_min, 1e-30)
    b = max(eps_max, a * 10.0)
    fa = F(a)
    fb = F(b)

    # Expand bracket if needed (log space).
    for _ in range(80):
        if np.isfinite(fa) and np.isfinite(fb) and (fa == 0.0 or fb == 0.0 or (fa > 0) != (fb > 0)):
            break
        # If signs are equal or NaN, expand outward.
        a = max(a / 10.0, 1e-30)
        b = b * 10.0
        fa = F(a)
        fb = F(b)
    else:
        raise RuntimeError("Could not bracket SC conductivity root; check phase conductivities and shapes.")

    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)

    lo, hi = float(a), float(b)
    flo, fhi = float(fa), float(fb)
    if (flo > 0) == (fhi > 0):
        raise RuntimeError("Could not bracket SC conductivity root; check phase conductivities and shapes.")

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        fmid = F(mid)
        if not np.isfinite(fmid):
            mid = np.nextafter(mid, hi)
            fmid = F(mid)
        if abs(fmid) <= tol:
            return float(mid)
        if (flo > 0) == (fmid > 0):
            lo, flo = mid, float(fmid)
        else:
            hi, fhi = mid, float(fmid)
        if abs(hi - lo) <= tol * max(1.0, abs(mid)):
            return float(0.5 * (lo + hi))

    return float(0.5 * (lo + hi))


def sca_sc_effective_conductivity(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    inclusion_fraction: float,
    *,
    aspect_ratio: float | None = None,
    depolarization_factors: Sequence[float] | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """
    Two-phase SC conductivity (Torquato 2002) for matrix + ellipsoidal inclusions.

    Notes
    -----
    - Matrix phase is treated as spherical (L=1/3,1/3,1/3).
    - Inclusion phase uses either ``depolarization_factors`` or ``aspect_ratio``.
    """
    eps_m = ensure_positive(matrix_conductivity, "matrix_conductivity", allow_zero=True)
    eps_i = ensure_positive(inclusion_conductivity, "inclusion_conductivity", allow_zero=True)
    vi = ensure_fraction(inclusion_fraction, "inclusion_fraction")

    if depolarization_factors is None:
        alpha = 1.0 if aspect_ratio is None else aspect_ratio
        depolarization_factors = spheroidal_depolarization_triplet(alpha)

    return sca_self_consistent_effective_conductivity(
        [1.0 - vi, vi],
        [eps_m, eps_i],
        depolarization_factors_list=[(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), depolarization_factors],
        tol=tol,
        max_iter=max_iter,
    )
