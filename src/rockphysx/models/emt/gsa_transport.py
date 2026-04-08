from __future__ import annotations

"""General Singular Approximation (GSA) transport model.

This module implements a reproducible tensor GSA backend for transport properties
(thermal conductivity, electrical conductivity, permeability) in isotropic and
transversely isotropic media. The implementation follows the Bayuk–Chesnokov
form of the transport-property GSA equation,

    T* = < T (I - g (T - Tc))^-1 >  < (I - g (T - Tc))^-1 >^-1,

where Tc is the comparison-body tensor and g is the singular Green-tensor term.

Design goals
------------
- multi-component mixtures: any number of phases;
- isotropic or TI effective medium;
- explicit comparison-body strategy;
- axisymmetric pore/crack shapes (sphere/spheroid) as the primary use case;
- numerical angular integration for g-tensor, so the same backend works for
  isotropic and anisotropic comparison bodies.

Important modeling note
-----------------------
This implementation is *strictly tensorial* and computes the g-tensor from a
numerical angular integral. For isotropic random mixtures, the scalar property
can be extracted as tr(T*)/3. For TI mixtures, the result is returned as
parallel/perpendicular components relative to the symmetry axis.
"""

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable, Literal

import math
import numpy as np
from numpy.linalg import inv, norm

ArrayLike = np.ndarray


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Shape:
    """Inclusion shape.

    kind
        "sphere" or "spheroid". For spheroid, the symmetry axis is the third
        semi-axis a3 in the local inclusion frame.
    semi_axes
        Semi-axes (a1, a2, a3). For a sphere all are equal. For a spheroid,
        a1 = a2 != a3.
    """

    kind: Literal["sphere", "spheroid"]
    semi_axes: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.semi_axes) != 3:
            raise ValueError("semi_axes must contain exactly three values")
        if any(a <= 0 for a in self.semi_axes):
            raise ValueError("semi_axes must be strictly positive")
        if self.kind == "sphere":
            a0 = self.semi_axes[0]
            if not np.allclose(self.semi_axes, (a0, a0, a0), atol=1e-12):
                raise ValueError("sphere requires equal semi_axes")
        if self.kind == "spheroid":
            if not np.isclose(self.semi_axes[0], self.semi_axes[1], atol=1e-12):
                raise ValueError("spheroid requires a1 = a2")

    @property
    def aspect_ratio(self) -> float:
        """Return axisymmetric aspect ratio alpha = a3 / a1."""
        a1, _, a3 = self.semi_axes
        return float(a3 / a1)

    @classmethod
    def sphere(cls, radius: float = 1.0) -> "Shape":
        return cls(kind="sphere", semi_axes=(radius, radius, radius))

    @classmethod
    def spheroid(cls, aspect_ratio: float, radius: float = 1.0) -> "Shape":
        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        return cls(kind="spheroid", semi_axes=(radius, radius, radius * aspect_ratio))


@dataclass(slots=True, frozen=True)
class OrientationDistribution:
    """Orientation distribution of a phase.

    kind
        - "random": isotropic random orientation.
        - "aligned": all inclusion symmetry axes parallel to symmetry_axis.
        - "ti_fabric": transversely isotropic ODF controlled by concentration.
          concentration = 0 gives random orientation, large values approach the
          aligned case.
    symmetry_axis
        Laboratory-frame symmetry axis for aligned / TI orientation.
    concentration
        Orientation concentration parameter for "ti_fabric".
    """

    kind: Literal["random", "aligned", "ti_fabric"]
    symmetry_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    concentration: float | None = None

    def __post_init__(self) -> None:
        axis = np.asarray(self.symmetry_axis, dtype=float)
        if axis.shape != (3,) or norm(axis) == 0.0:
            raise ValueError("symmetry_axis must be a non-zero 3-vector")
        if self.kind == "ti_fabric" and (self.concentration is None or self.concentration < 0):
            raise ValueError("ti_fabric requires concentration >= 0")


@dataclass(slots=True, frozen=True)
class Phase:
    """One component of the heterogeneous medium."""

    name: str
    volume_fraction: float
    property_tensor: ArrayLike  # 3x3 transport tensor
    shape: Shape
    orientation: OrientationDistribution

    def __post_init__(self) -> None:
        if not (0.0 <= self.volume_fraction <= 1.0):
            raise ValueError("volume_fraction must lie in [0, 1]")
        tensor = np.asarray(self.property_tensor, dtype=float)
        if tensor.shape != (3, 3):
            raise ValueError("property_tensor must be a 3x3 array")
        if not np.allclose(tensor, tensor.T, atol=1e-12):
            raise ValueError("property_tensor must be symmetric")
        if np.any(np.linalg.eigvalsh(tensor) <= 0.0):
            raise ValueError("property_tensor must be positive definite")


@dataclass(slots=True, frozen=True)
class ComparisonBody:
    """Comparison-body closure strategy."""

    kind: Literal["matrix", "self_consistent", "bayuk_linear_mix", "user_defined"]
    matrix_index: int = 0
    k_connectivity: float | None = None
    user_tensor: ArrayLike | None = None


@dataclass(slots=True, frozen=True)
class TITransportResult:
    lambda_parallel: float
    lambda_perpendicular: float

    @property
    def tensor(self) -> np.ndarray:
        return np.diag([
            self.lambda_perpendicular,
            self.lambda_perpendicular,
            self.lambda_parallel,
        ])

    @property
    def ratio(self) -> float:
        return float(self.lambda_parallel / self.lambda_perpendicular)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def depolarization_factor_spheroid(alpha: float) -> float:
    """Axisymmetric depolarization factor along the symmetry axis."""
    if alpha <= 0:
        raise ValueError("aspect_ratio must be positive")

    if np.isclose(alpha, 1.0):
        return 1.0 / 3.0

    if alpha < 1.0:  # oblate / crack-like
        t1 = alpha ** 2
        t2 = 1.0 / t1
        t4 = np.sqrt(t2 - 1.0)
        t5 = np.arctan(t4)
        return t2 * (t4 - t5) / (t4 ** 3)

    # alpha > 1.0, prolate
    t1 = alpha ** 2
    t2 = 1.0 / t1
    t4 = np.sqrt(1.0 - t2)
    t6 = np.log((1.0 + t4) / (1.0 - t4))
    return t2 * (0.5 * t6 - t4) / (t4 ** 3)

def _phase_scalar_random_axisymmetric(
    phase_value: float,
    comparison_value: float,
    aspect_ratio: float,
) -> tuple[float, float]:
    """
    Scalar phase operators for isotropic medium with randomly oriented
    axisymmetric spheroidal inclusions.
    """
    F = depolarization_factor_spheroid(aspect_ratio)

    d_long = comparison_value * (1.0 - F) + phase_value * F
    d_trans = comparison_value * (1.0 + F) / 2.0 + phase_value * (1.0 - F) / 2.0

    m_i = (1.0 / d_long + 2.0 / d_trans) / 3.0
    a_i = phase_value * m_i
    return a_i, m_i

def _all_phases_isotropic_random(phases: list[Phase]) -> bool:
    for p in phases:
        if p.orientation.kind != "random":
            return False
        if not np.allclose(p.property_tensor, np.eye(3) * p.property_tensor[0, 0], atol=1e-12):
            return False
        if p.shape.kind not in {"sphere", "spheroid"}:
            return False
    return True

def homogenize_transport_gsa_isotropic_random(
    phases: Iterable[Phase],
    comparison_body: ComparisonBody,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    initial_effective: float | None = None,
) -> float:
    """
    Special isotropic GSA branch for randomly oriented axisymmetric inclusions.
    This preserves aspect-ratio dependence.
    """
    phases = validate_phases(phases)

    if not _all_phases_isotropic_random(phases):
        raise ValueError("This branch requires isotropic tensors and random orientations only.")

    if initial_effective is None:
        current_eff = sum(p.volume_fraction * p.property_tensor[0, 0] for p in phases)
    else:
        current_eff = float(initial_effective)

    for _ in range(max_iter):
        if comparison_body.kind == "self_consistent":
            Tc = current_eff
        else:
            Tc_tensor = compute_comparison_body(phases, comparison_body, np.eye(3) * current_eff)
            Tc = float(Tc_tensor[0, 0])

        num = 0.0
        den = 0.0

        for p in phases:
            phase_value = float(p.property_tensor[0, 0])
            alpha = p.shape.aspect_ratio
            a_i, m_i = _phase_scalar_random_axisymmetric(phase_value, Tc, alpha)
            c = p.volume_fraction
            num += c * a_i
            den += c * m_i

        new_eff = num / den

        if comparison_body.kind != "self_consistent":
            return float(new_eff)

        if abs(new_eff - current_eff) < tol:
            return float(new_eff)

        current_eff = new_eff

    raise RuntimeError("Isotropic random GSA did not converge.")

def as_tensor(value: float | ArrayLike) -> np.ndarray:
    """Convert a scalar or array-like value to a symmetric 3x3 tensor."""
    if np.isscalar(value):
        scalar = float(value)
        if scalar <= 0.0:
            raise ValueError("scalar transport property must be positive")
        return np.eye(3) * scalar
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError("tensor value must be 3x3")
    if not np.allclose(arr, arr.T, atol=1e-12):
        raise ValueError("tensor value must be symmetric")
    if np.any(np.linalg.eigvalsh(arr) <= 0.0):
        raise ValueError("tensor value must be positive definite")
    return arr


def validate_phases(phases: Iterable[Phase]) -> list[Phase]:
    phases = list(phases)
    if len(phases) < 2:
        raise ValueError("At least two phases are required")
    phi_sum = sum(p.volume_fraction for p in phases)
    if not np.isclose(phi_sum, 1.0, atol=1e-8):
        raise ValueError(f"Volume fractions must sum to 1.0, got {phi_sum!r}")
    return phases


def unit_vector(v: ArrayLike) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = norm(v)
    if n == 0.0:
        raise ValueError("zero vector cannot be normalized")
    return v / n


def rotation_from_z(axis: ArrayLike) -> np.ndarray:
    """Rotation matrix that maps local ez onto the given axis."""
    z = np.array([0.0, 0.0, 1.0])
    a = unit_vector(axis)
    if np.allclose(a, z):
        return np.eye(3)
    if np.allclose(a, -z):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, a)
    s = norm(v)
    c = float(np.dot(z, a))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s**2))


# -----------------------------------------------------------------------------
# Angular quadrature and orientation sampling
# -----------------------------------------------------------------------------


def fibonacci_sphere(n: int) -> np.ndarray:
    """Quasi-uniform points on the unit sphere."""
    if n < 1:
        raise ValueError("n must be >= 1")
    pts = np.zeros((n, 3), dtype=float)
    phi = pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * i / max(n - 1, 1)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i] = [np.cos(theta) * r, y, np.sin(theta) * r]
    return pts


def sample_orientation_axes(odf: OrientationDistribution, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return orientation axes and weights.

    For axisymmetric inclusions only the symmetry axis matters.
    """
    if odf.kind == "aligned":
        return unit_vector(odf.symmetry_axis).reshape(1, 3), np.array([1.0])

    axes = fibonacci_sphere(n)

    if odf.kind == "random":
        weights = np.full(n, 1.0 / n)
        return axes, weights

    # TI ODF around symmetry_axis using a simple Fisher-like weight exp(k (a·n)^2)
    sym = unit_vector(odf.symmetry_axis)
    kappa = float(odf.concentration or 0.0)
    dots = axes @ sym
    weights = np.exp(kappa * dots * dots)
    weights /= np.sum(weights)
    return axes, weights


# -----------------------------------------------------------------------------
# g-tensor
# -----------------------------------------------------------------------------


def g_tensor_transport_numeric(
    comparison_tensor: np.ndarray,
    shape: Shape,
    *,
    orientation_axis: ArrayLike = (0.0, 0.0, 1.0),
    n_theta: int | None = None,
    n_phi: int | None = None,
    theta_quadrature: Literal["uniform", "gauss"] = "uniform",
) -> np.ndarray:
    """Numerical g-tensor for transport properties.

    Reproducible method based on the Bayuk–Chesnokov transport-property integral.
    The inclusion local frame is defined by the shape semi-axes; the local symmetry
    axis (third semi-axis) is then rotated onto orientation_axis.
    """
    Tc = as_tensor(comparison_tensor)
    a1, a2, a3 = shape.semi_axes
    R = rotation_from_z(orientation_axis)

    def _recommend_quadrature(aspect_ratio: float) -> tuple[int, int]:
        # Heuristic: resolution required grows with anisotropy of the inclusion shape.
        # For crack-like spheroids (alpha << 1) the integrand becomes sharply peaked
        # near the poles; similarly for needle-like (alpha >> 1) near the equator.
        beta = max(float(aspect_ratio), 1.0 / float(aspect_ratio))
        if beta <= 1.5:
            return (60, 120)
        if beta <= 10.0:
            return (120, 240)
        if beta <= 100.0:
            return (240, 480)
        return (400, 800)

    if n_theta is None or n_phi is None:
        rt, rp = _recommend_quadrature(shape.aspect_ratio)
        n_theta = rt if n_theta is None else int(n_theta)
        n_phi = rp if n_phi is None else int(n_phi)
    else:
        n_theta = int(n_theta)
        n_phi = int(n_phi)

    phi_vals = np.linspace(0.0, 2.0 * pi, int(n_phi), endpoint=False)
    dphi = 2.0 * pi / float(n_phi)

    g = np.zeros((3, 3), dtype=float)

    if theta_quadrature not in {"uniform", "gauss"}:
        raise ValueError("theta_quadrature must be 'uniform' or 'gauss'.")

    if theta_quadrature == "uniform":
        theta_vals = np.linspace(0.0, pi, int(n_theta))
        dtheta = theta_vals[1] - theta_vals[0] if int(n_theta) > 1 else pi
        for theta in theta_vals:
            st = sin(theta)
            ct = cos(theta)
            for phi in phi_vals:
                cp = cos(phi)
                sp = sin(phi)
                # scaled direction in the local ellipsoid frame
                m_local = np.array([st * cp / a1, st * sp / a2, ct / a3], dtype=float)
                m = R @ m_local
                A = float(m @ Tc @ m)
                if A <= 0.0:
                    raise RuntimeError("Non-positive A encountered in g-tensor integration")
                g += np.outer(m, m) / A * st * dtheta * dphi
    else:
        # Gauss-Legendre in u = cos(theta), u ∈ [-1, 1].
        # sin(theta) dtheta = du, so the integral becomes:
        #   g = (1/4π) ∫_{0..2π} ∫_{-1..1} (m m^T)/A du dφ
        u, wu = np.polynomial.legendre.leggauss(int(n_theta))
        for ui, wi in zip(u, wu, strict=True):
            ct = float(ui)
            st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
            for phi in phi_vals:
                cp = cos(phi)
                sp = sin(phi)
                m_local = np.array([st * cp / a1, st * sp / a2, ct / a3], dtype=float)
                m = R @ m_local
                A = float(m @ Tc @ m)
                if A <= 0.0:
                    raise RuntimeError("Non-positive A encountered in g-tensor integration")
                g += np.outer(m, m) / A * float(wi) * dphi

    return g / (4.0 * pi)


def orientation_averaged_g_tensor(
    comparison_tensor: np.ndarray,
    shape: Shape,
    orientation: OrientationDistribution,
    *,
    n_orientation: int = 80,
    n_theta: int | None = None,
    n_phi: int | None = None,
    theta_quadrature: Literal["uniform", "gauss"] = "uniform",
) -> np.ndarray:
    axes, weights = sample_orientation_axes(orientation, n_orientation)
    g_avg = np.zeros((3, 3), dtype=float)
    for axis, w in zip(axes, weights):
        g_axis = g_tensor_transport_numeric(
            comparison_tensor,
            shape,
            orientation_axis=axis,
            n_theta=n_theta,
            n_phi=n_phi,
            theta_quadrature=theta_quadrature,
        )
        g_avg += float(w) * g_axis
    return g_avg


# -----------------------------------------------------------------------------
# Comparison body
# -----------------------------------------------------------------------------


def compute_comparison_body(
    phases: list[Phase],
    strategy: ComparisonBody,
    current_effective: np.ndarray | None = None,
) -> np.ndarray:
    phases = validate_phases(phases)

    if strategy.kind == "matrix":
        return phases[strategy.matrix_index].property_tensor

    if strategy.kind == "self_consistent":
        if current_effective is None:
            raise ValueError("current_effective is required for self_consistent strategy")
        return as_tensor(current_effective)

    if strategy.kind == "bayuk_linear_mix":
        if strategy.k_connectivity is None:
            raise ValueError("k_connectivity is required for bayuk_linear_mix")
        matrix_phase = phases[strategy.matrix_index]
        Tm = matrix_phase.property_tensor
        inclusion_phases = [p for i, p in enumerate(phases) if i != strategy.matrix_index]
        Vi = sum(p.volume_fraction for p in inclusion_phases)
        if Vi <= 0.0:
            return Tm
        Ti = sum(p.volume_fraction * p.property_tensor for p in inclusion_phases) / Vi
        k = float(strategy.k_connectivity)
        return (1.0 - k * Vi) * Tm + (k * Vi) * Ti

    if strategy.kind == "user_defined":
        if strategy.user_tensor is None:
            raise ValueError("user_tensor is required for user_defined strategy")
        return as_tensor(strategy.user_tensor)

    raise ValueError(f"Unknown comparison-body strategy {strategy.kind!r}")


# -----------------------------------------------------------------------------
# Homogenization core
# -----------------------------------------------------------------------------


def phase_operators(
    phase_tensor: np.ndarray,
    comparison_tensor: np.ndarray,
    g_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = phase_tensor - comparison_tensor
    # M = inv(np.eye(3) - g_tensor @ delta)
    M = inv(np.eye(3) + g_tensor @ delta)
    A = phase_tensor @ M
    return A, M


def homogenize_transport_gsa(
    phases: Iterable[Phase],
    comparison_body: ComparisonBody,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    n_orientation: int = 80,
    n_theta: int | None = None,
    n_phi: int | None = None,
    theta_quadrature: Literal["uniform", "gauss"] = "uniform",
    initial_effective: np.ndarray | None = None,
) -> np.ndarray:
    """General N-phase tensor GSA for transport properties.

    Supports multi-component isotropic and TI media.
    For `comparison_body.kind == "self_consistent"`, the body is updated by fixed-
    point iteration until convergence.
    """
    phases = validate_phases(phases)

    if initial_effective is None:
        current_effective = sum(p.volume_fraction * p.property_tensor for p in phases)
    else:
        current_effective = as_tensor(initial_effective)

    for _ in range(max_iter):
        Tc = compute_comparison_body(phases, comparison_body, current_effective)
        num = np.zeros((3, 3), dtype=float)
        den = np.zeros((3, 3), dtype=float)

        for phase in phases:
            g = orientation_averaged_g_tensor(
                Tc,
                phase.shape,
                phase.orientation,
                n_orientation=n_orientation,
                n_theta=n_theta,
                n_phi=n_phi,
                theta_quadrature=theta_quadrature,
            )
            A, M = phase_operators(phase.property_tensor, Tc, g)
            c = phase.volume_fraction
            num += c * A
            den += c * M

        T_new = num @ inv(den)

        if comparison_body.kind != "self_consistent":
            return T_new

        if np.allclose(T_new, current_effective, atol=tol, rtol=0.0):
            return T_new

        current_effective = T_new

    raise RuntimeError("GSA self-consistent iteration did not converge")


# -----------------------------------------------------------------------------
# User-level wrappers
# -----------------------------------------------------------------------------


def isotropic_scalar_from_tensor(tensor: np.ndarray) -> float:
    return float(np.trace(tensor) / 3.0)


def gsa_transport_isotropic(
    phases: Iterable[Phase],
    comparison_body: ComparisonBody,
    **kwargs,
) -> float:
    phase_list = validate_phases(phases)

    if _all_phases_isotropic_random(phase_list):
        # The isotropic-random scalar branch does not use angular/orientation
        # quadrature parameters but callers may pass them for API uniformity.
        allowed = {"max_iter", "tol", "initial_effective"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return homogenize_transport_gsa_isotropic_random(
            phase_list,
            comparison_body,
            **kwargs,
        )

    T = homogenize_transport_gsa(phase_list, comparison_body, **kwargs)
    return isotropic_scalar_from_tensor(T)


def gsa_transport_ti(
    phases: Iterable[Phase],
    comparison_body: ComparisonBody,
    **kwargs,
) -> TITransportResult:
    T = homogenize_transport_gsa(phases, comparison_body, **kwargs)
    return TITransportResult(
        lambda_parallel=float(T[2, 2]),
        lambda_perpendicular=float(0.5 * (T[0, 0] + T[1, 1])),
    )


# -----------------------------------------------------------------------------
# Convenience constructors for common thermal-conductivity cases
# -----------------------------------------------------------------------------


def make_phase(
    name: str,
    volume_fraction: float,
    property_value: float | ArrayLike,
    *,
    aspect_ratio: float = 1.0,
    orientation: Literal["random", "aligned"] = "random",
    symmetry_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Phase:
    tensor = as_tensor(property_value)
    shape = Shape.sphere() if np.isclose(aspect_ratio, 1.0) else Shape.spheroid(aspect_ratio)
    odf = OrientationDistribution(kind=orientation, symmetry_axis=symmetry_axis)
    return Phase(name=name, volume_fraction=volume_fraction, property_tensor=tensor, shape=shape, orientation=odf)


def two_phase_thermal_isotropic(
    matrix_value: float,
    inclusion_value: float,
    porosity: float,
    *,
    aspect_ratio: float,
    comparison: Literal["matrix", "self_consistent", "bayuk_linear_mix"] = "matrix",
    k_connectivity: float | None = None,
    matrix_index: int = 0,
    **kwargs,
) -> float:
    solid = 1.0 - porosity
    phases = [
        make_phase("matrix", solid, matrix_value, aspect_ratio=1.0, orientation="random"),
        make_phase("inclusion", porosity, inclusion_value, aspect_ratio=aspect_ratio, orientation="random"),
    ]
    body = ComparisonBody(kind=comparison, matrix_index=matrix_index, k_connectivity=k_connectivity)
    return gsa_transport_isotropic(phases, body, **kwargs)


def two_phase_thermal_ti(
    matrix_value: float,
    inclusion_value: float,
    porosity: float,
    *,
    aspect_ratio: float,
    comparison: Literal["matrix", "self_consistent", "bayuk_linear_mix"] = "matrix",
    k_connectivity: float | None = None,
    symmetry_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    **kwargs,
) -> TITransportResult:
    solid = 1.0 - porosity
    phases = [
        make_phase("matrix", solid, matrix_value, aspect_ratio=1.0, orientation="random"),
        make_phase(
            "inclusion",
            porosity,
            inclusion_value,
            aspect_ratio=aspect_ratio,
            orientation="aligned",
            symmetry_axis=symmetry_axis,
        ),
    ]
    body = ComparisonBody(kind=comparison, matrix_index=0, k_connectivity=k_connectivity)
    return gsa_transport_ti(phases, body, **kwargs)


# -----------------------------------------------------------------------------
# Benchmark: self-consistent model (explicitly not named GSA)
# -----------------------------------------------------------------------------


def self_consistent_transport(
    phases: Iterable[Phase],
    **kwargs,
) -> np.ndarray:
    body = ComparisonBody(kind="self_consistent")
    return homogenize_transport_gsa(phases, body, **kwargs)


__all__ = [
    "Shape",
    "OrientationDistribution",
    "Phase",
    "ComparisonBody",
    "TITransportResult",
    "as_tensor",
    "compute_comparison_body",
    "g_tensor_transport_numeric",
    "orientation_averaged_g_tensor",
    "homogenize_transport_gsa",
    "gsa_transport_isotropic",
    "gsa_transport_ti",
    "two_phase_thermal_isotropic",
    "two_phase_thermal_ti",
    "self_consistent_transport",
    "make_phase",
]
