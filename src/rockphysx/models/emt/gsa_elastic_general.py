from __future__ import annotations

"""
General elastic GSA / OSP utilities (numeric Green tensor).

Motivation
----------
The existing elastic GSA implementation in this repo relies on an uploaded Fortran
kernel `GREEN_ANAL_VTI` that computes the local Green tensor for a *VTI* comparison
body. That necessarily constrains the GSA step to TI/VTI forms.

To reproduce dissertation workflows that include:
- lower symmetry single-crystal stiffness (trigonal, monoclinic, ...),
- coordinate rotations (Euler angles),
- ODF-based orientation averaging,
we need a Green tensor implementation that works for a general anisotropic
comparison stiffness.

This module implements a numerical Green-tensor integral for elasticity in the
spirit of the dissertation equations (2-29)–(2-32).

Scope (current)
---------------
- Ellipsoidal inclusion shape (semi-axes a1,a2,a3).
- General anisotropic comparison stiffness C^c (engineering Voigt 6x6 input).
- Aligned inclusion orientation: the ellipsoid local a3 axis rotated onto a
  specified global axis (orientation_axis), with optional extra rotation matrix.
- Case-1 GSA closure with comparison body = matrix (solid inclusions in solid matrix),
  producing a full 6x6 stiffness matrix (engineering Voigt) that can contain
  low-symmetry terms after rotations.

Notes
-----
- Internally all algebra uses Mandel 6x6 to avoid the Voigt shear scaling issues
  discussed in the dissertation (their coefficient matrix A(m,n)). Mandel is an
  orthonormal basis for symmetric tensors, so matrix inverses correspond to the
  physical 4th-rank inverses.
"""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from . import gsa_elastic_random_isotropic as core


@dataclass(frozen=True)
class EllipsoidShape:
    a1: float
    a2: float
    a3: float


@dataclass(frozen=True)
class OrientationDistributionFunction:
    """
    Orientation distribution function (ODF) for Euler-angle averaging.

    Euler convention follows the dissertation z-x'-z'' convention:
        R = B(psi) C(theta) D(phi)

    Supported kinds
    ---------------
    - "aligned": delta-function (single orientation, identity).
    - "random_3d": uniform random distribution on SO(3).
    - "ti_gaussian": transversely isotropic distribution around the z-axis:
        phi ~ U[0, 2π), psi ~ U[0, 2π)
        theta weighted by exp(-(theta-mean)^2/(2 sigma^2)).
    """

    kind: Literal["aligned", "random_3d", "ti_gaussian"] = "aligned"
    mean_tilt_deg: float = 0.0
    sigma_tilt_deg: float = 10.0


def _unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("orientation_axis must be a non-zero finite vector.")
    return v / n


def _rotation_from_z(target_axis: np.ndarray) -> np.ndarray:
    """
    Rotation matrix that maps local z-axis onto target_axis.

    This matches the helper used in gsa_transport.py.
    """
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    t = _unit_vector(target_axis)
    if np.allclose(t, z):
        return np.eye(3)
    if np.allclose(t, -z):
        # 180° rotation about x
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

    v = np.cross(z, t)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, t))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R


def rotation_matrix_zxz(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Euler rotation matrix for the z-x'-z'' convention (dissertation).

    Parameters are in radians.
    """
    D = np.array(
        [
            [np.cos(phi), np.sin(phi), 0.0],
            [-np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    C = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), np.sin(theta)],
            [0.0, -np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [np.cos(psi), np.sin(psi), 0.0],
            [-np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return B @ C @ D


def sample_rotation_matrices(
    odf: OrientationDistributionFunction,
    *,
    n: int = 2000,
    rng: np.random.Generator | None = None,
    n_phi: int = 80,
    n_theta: int = 80,
    n_psi: int = 80,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Return rotation matrices and associated weights for ODF averaging.

    - "random_3d": Monte-Carlo uniform sampling on SO(3) with equal weights.
    - "ti_gaussian": tensor-product quadrature in (phi, theta, psi) with
      weight exp(-(theta-mean)^2/(2 sigma^2)) * sin(theta).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    kind = str(odf.kind)
    if kind == "aligned":
        return [np.eye(3, dtype=float)], np.array([1.0], dtype=float)

    if kind == "random_3d":
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive for random_3d.")
        # Uniform SO(3): sample unit quaternions (uniform on S^3).
        q = rng.normal(size=(n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R_list: list[np.ndarray] = []
        for wi, xi, yi, zi in zip(w, x, y, z, strict=True):
            R_list.append(
                np.array(
                    [
                        [1 - 2 * (yi * yi + zi * zi), 2 * (xi * yi + wi * zi), 2 * (xi * zi - wi * yi)],
                        [2 * (xi * yi - wi * zi), 1 - 2 * (xi * xi + zi * zi), 2 * (yi * zi + wi * xi)],
                        [2 * (xi * zi + wi * yi), 2 * (yi * zi - wi * xi), 1 - 2 * (xi * xi + yi * yi)],
                    ],
                    dtype=float,
                )
            )
        weights = np.full(len(R_list), 1.0 / len(R_list), dtype=float)
        return R_list, weights

    if kind == "ti_gaussian":
        mu = np.deg2rad(float(odf.mean_tilt_deg))
        sig = np.deg2rad(float(odf.sigma_tilt_deg))
        if not np.isfinite(sig) or sig <= 0.0:
            raise ValueError("sigma_tilt_deg must be positive for ti_gaussian.")

        phi_vals = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
        psi_vals = np.linspace(0.0, 2.0 * np.pi, int(n_psi), endpoint=False)
        theta_vals = np.linspace(0.0, np.pi, int(n_theta))
        dphi = 2.0 * np.pi / len(phi_vals)
        dpsi = 2.0 * np.pi / len(psi_vals)
        dtheta = theta_vals[1] - theta_vals[0] if len(theta_vals) > 1 else np.pi

        R_list: list[np.ndarray] = []
        w_list: list[float] = []
        for th in theta_vals:
            st = float(np.sin(th))
            wt = float(np.exp(-0.5 * ((th - mu) / sig) ** 2))
            for ph in phi_vals:
                for ps in psi_vals:
                    R_list.append(rotation_matrix_zxz(ph, th, ps))
                    w_list.append(wt * st * dphi * dtheta * dpsi)

        weights = np.asarray(w_list, dtype=float)
        s = float(np.sum(weights))
        if not np.isfinite(s) or s <= 0.0:
            raise RuntimeError("ODF weights sum is invalid.")
        weights = weights / s
        return R_list, weights

    raise ValueError(f"Unsupported ODF kind: {odf.kind!r}. Use 'aligned', 'random_3d', or 'ti_gaussian'.")


def orientation_average_stiffness_voigt66(
    C_local_voigt66: np.ndarray,
    *,
    odf: OrientationDistributionFunction,
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute <C^T> under an ODF (engineering Voigt 6x6, Pa).

    This matches the dissertation-style Euler-angle average:
      <X> = ∫∫∫ F(phi,theta,psi) X^T(phi,theta,psi) sin(theta) dphi dtheta dpsi
    """
    C = np.asarray(C_local_voigt66, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C_local_voigt66 must be (6,6).")
    R_list, weights = sample_rotation_matrices(odf, n=n, rng=rng)
    out = np.zeros((6, 6), dtype=float)
    for R, w in zip(R_list, weights, strict=True):
        out += float(w) * rotate_stiffness_voigt66(C, R)
    return out


def orientation_averaged_green_tensor_elastic_numeric(
    comparison_stiffness_voigt66: np.ndarray,
    shape: EllipsoidShape,
    *,
    odf: OrientationDistributionFunction,
    n_orientation: int = 500,
    rng: np.random.Generator | None = None,
    # inner Green integral resolution:
    n_theta: int = 60,
    n_phi: int = 120,
) -> np.ndarray:
    """
    Orientation-average the elastic Green tensor over Euler angles.

    This matches the dissertation-style averaging:
        <g> = ∫ F(phi,theta,psi) g(phi,theta,psi) sin(theta) dphi dtheta dpsi

    Notes
    -----
    - This is expensive: it nests an orientation loop around the angular Green integral.
    - For spheroids (a1=a2) the dependence on the third Euler angle is weak/none,
      but we keep the general form for completeness.
    """
    R_list, weights = sample_rotation_matrices(odf, n=n_orientation, rng=rng)
    g = np.zeros((3, 3, 3, 3), dtype=float)
    for R, w in zip(R_list, weights, strict=True):
        gi = green_tensor_elastic_numeric(
            comparison_stiffness_voigt66=comparison_stiffness_voigt66,
            shape=shape,
            orientation_rotation=R,
            n_theta=int(n_theta),
            n_phi=int(n_phi),
        )
        g += float(w) * gi
    return g


def _lambda_matrix(Cc_ijkl: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Λ_{ik} = C^c_{ijkl} m_j m_l

    (Christoffel-like matrix for the comparison stiffness, using the ellipsoid-scaled direction m)
    """
    return np.einsum("ijkl,j,l->ik", Cc_ijkl, m, m, optimize=True)


def green_tensor_elastic_numeric(
    comparison_stiffness_voigt66: np.ndarray,
    shape: EllipsoidShape,
    *,
    orientation_axis: Iterable[float] = (0.0, 0.0, 1.0),
    extra_rotation: np.ndarray | None = None,
    orientation_rotation: np.ndarray | None = None,
    n_theta: int = 60,
    n_phi: int = 120,
) -> np.ndarray:
    """
    Numerical 4th-rank Green tensor g_ijkl for elasticity.

    Parameters
    ----------
    comparison_stiffness_voigt66:
        Engineering Voigt 6x6 stiffness of the comparison body, in Pa.
    shape:
        Ellipsoid semi-axes (a1,a2,a3). Only ratios matter.
    orientation_axis:
        Global axis onto which the ellipsoid local a3 axis is rotated.
        Ignored if orientation_rotation is provided.
    extra_rotation:
        Optional 3x3 rotation applied after the axis alignment (useful for Euler rotations).
        Ignored if orientation_rotation is provided.
    orientation_rotation:
        Optional full 3x3 rotation mapping the ellipsoid local axes to the global axes.
        Use this for full Euler-angle/ODF workflows and triaxial ellipsoids.
    n_theta, n_phi:
        Angular quadrature resolution.

    Returns
    -------
    g_ijkl (3,3,3,3)
    """
    Cc = np.asarray(comparison_stiffness_voigt66, dtype=float)
    if Cc.shape != (6, 6):
        raise ValueError("comparison_stiffness_voigt66 must have shape (6,6).")

    a1 = float(shape.a1)
    a2 = float(shape.a2)
    a3 = float(shape.a3)
    if not (a1 > 0 and a2 > 0 and a3 > 0):
        raise ValueError("Ellipsoid semi-axes must be positive.")

    # Voigt -> Mandel -> tensor4
    Cc_mandel = core.voigt66_to_mandel66(Cc)
    Cc_ijkl = core.mandel_to_tensor4(Cc_mandel)

    if orientation_rotation is not None:
        R = np.asarray(orientation_rotation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("orientation_rotation must be (3,3).")
    else:
        R = _rotation_from_z(np.asarray(list(orientation_axis), dtype=float))
        if extra_rotation is not None:
            extra_rotation = np.asarray(extra_rotation, dtype=float)
            if extra_rotation.shape != (3, 3):
                raise ValueError("extra_rotation must be (3,3).")
            R = extra_rotation @ R

    theta_vals = np.linspace(0.0, np.pi, int(n_theta))
    phi_vals = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    dtheta = theta_vals[1] - theta_vals[0] if len(theta_vals) > 1 else np.pi
    dphi = 2.0 * np.pi / len(phi_vals)

    # Accumulate a_{kj,il} = -1/(4π) ∫ m_k m_j Λ^{-1}_{il} dΩ
    a = np.zeros((3, 3, 3, 3), dtype=float)

    for th in theta_vals:
        st = float(np.sin(th))
        ct = float(np.cos(th))
        for ph in phi_vals:
            cp = float(np.cos(ph))
            sp = float(np.sin(ph))
            # ellipsoid-scaled direction in local frame (as in transport integral)
            m_local = np.array([st * cp / a1, st * sp / a2, ct / a3], dtype=float)
            m = R @ m_local

            Lam = _lambda_matrix(Cc_ijkl, m)
            # Symmetrize to reduce numerical noise.
            Lam = 0.5 * (Lam + Lam.T)
            try:
                Lam_inv = np.linalg.inv(Lam)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Singular Λ encountered in Green-tensor integration.") from exc

            w = st * dtheta * dphi
            # outer(m,m) gives m_k m_j
            mm = np.outer(m, m)
            # a_{k j i l} += mm_{k j} * Lam_inv_{i l}
            a += np.einsum("kj,il->kjil", mm, Lam_inv, optimize=True) * w

    a *= -(1.0 / (4.0 * np.pi))

    # g_{ijkl} = (a_{kj,il} + a_{ki,jl} + a_{lj,ik} + a_{li,jk}) / 4
    g = (
        np.transpose(a, (0, 1, 2, 3))
        + np.transpose(a, (0, 2, 1, 3))
        + np.transpose(a, (3, 1, 2, 0))
        + np.transpose(a, (3, 2, 1, 0))
    ) / 4.0
    return g


def gsa_case1_effective_stiffness(
    *,
    matrix_C_voigt66: np.ndarray,
    inclusion_C_voigt66_list: list[np.ndarray],
    fractions: np.ndarray,
    inclusion_shape: EllipsoidShape,
    orientation_axis: Iterable[float] = (0.0, 0.0, 1.0),
    extra_rotation: np.ndarray | None = None,
    sign: int = -1,
    n_theta: int = 60,
    n_phi: int = 120,
) -> np.ndarray:
    """
    Case-1 style elastic GSA: solid inclusions in solid matrix, comparison body = matrix.

    Returns
    -------
    C_eff_voigt66 (engineering Voigt), Pa
    """
    C0_voigt = np.asarray(matrix_C_voigt66, dtype=float)
    if C0_voigt.shape != (6, 6):
        raise ValueError("matrix_C_voigt66 must be (6,6).")

    fr = np.asarray(fractions, dtype=float).reshape(-1)
    if fr.size != 1 + len(inclusion_C_voigt66_list):
        raise ValueError("fractions length must equal 1 + number of inclusions.")
    if np.any(fr < 0.0):
        raise ValueError("fractions must be non-negative.")
    s = float(np.sum(fr))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Sum of fractions must be positive.")
    fr = fr / s

    if sign not in (-1, +1):
        raise ValueError("sign must be -1 or +1.")

    # Comparison body = matrix.
    C0_mandel = core.voigt66_to_mandel66(C0_voigt)
    I6 = np.eye(6, dtype=float)

    g_ijkl = green_tensor_elastic_numeric(
        comparison_stiffness_voigt66=C0_voigt,
        shape=inclusion_shape,
        orientation_axis=orientation_axis,
        extra_rotation=extra_rotation,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    G0_66 = core.tensor4_to_mandel(g_ijkl)

    # Matrix concentration tensor is identity because ΔC=0.
    C_list = [C0_mandel]
    A_list = [I6]

    for Ci_voigt in inclusion_C_voigt66_list:
        Ci_voigt = np.asarray(Ci_voigt, dtype=float)
        if Ci_voigt.shape != (6, 6):
            raise ValueError("Each inclusion stiffness must be (6,6).")
        Ci_mandel = core.voigt66_to_mandel66(Ci_voigt)
        dC = Ci_mandel - C0_mandel
        A = np.linalg.inv(I6 + float(sign) * (G0_66 @ dC))
        C_list.append(Ci_mandel)
        A_list.append(A)

    num = np.zeros((6, 6), dtype=float)
    den = np.zeros((6, 6), dtype=float)
    for fi, Ci, Ai in zip(fr, C_list, A_list, strict=True):
        num += float(fi) * (Ci @ Ai)
        den += float(fi) * Ai

    C_eff_mandel = num @ np.linalg.inv(den)
    return core.mandel66_to_voigt66(C_eff_mandel)


__all__ = [
    "EllipsoidShape",
    "OrientationDistributionFunction",
    "rotation_matrix_zxz",
    "sample_rotation_matrices",
    "orientation_average_stiffness_voigt66",
    "orientation_averaged_green_tensor_elastic_numeric",
    "green_tensor_elastic_numeric",
    "gsa_case1_effective_stiffness",
    "rotate_stiffness_voigt66",
]


def rotate_stiffness_voigt66(C_voigt66: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate a stiffness tensor given in engineering Voigt 6x6.

    C'_{ijkl} = R_{im} R_{jn} R_{kp} R_{lq} C_{mnpq}
    """
    C_voigt66 = np.asarray(C_voigt66, dtype=float)
    if C_voigt66.shape != (6, 6):
        raise ValueError("C_voigt66 must have shape (6,6).")
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must have shape (3,3).")
    C_mandel = core.voigt66_to_mandel66(C_voigt66)
    C_ijkl = core.mandel_to_tensor4(C_mandel)
    Cp = np.einsum("im,jn,kp,lq,mnpq->ijkl", R, R, R, R, C_ijkl, optimize=True)
    Cp_mandel = core.tensor4_to_mandel(Cp)
    return core.mandel66_to_voigt66(Cp_mandel)
