
"""
Strict-ish elastic Mori–Tanaka module for isotropic matrix + isotropic fluid-filled
spheroidal pores, inspired by the tensor architecture used in homopy / Benveniste.

Important scope notes
---------------------
1. This module is for *elastic* properties only. Thermal conductivity is handled separately.
2. The implementation uses Eshelby components for a spheroid embedded in an isotropic matrix,
   following the same component structure used in homopy for the prolate branch, but extended
   with a custom oblate branch for aspect ratios a < 1.
3. The final effective stiffness is orientation-averaged for *random isotropic orientation*.
4. The code is intended as a clean, auditable module for rock-physics experimentation, not as a
   drop-in replacement for every composite-mechanics package.

References used as design anchors
---------------------------------
- Benveniste (1987): Mori–Tanaka direct formulation for effective stiffness.
- homopy repository: tensor architecture / orientation averaging / Mandel mapping style.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


ArrayLike = np.ndarray


# -----------------------------------------------------------------------------
# Basic tensor helpers (Mandel notation <-> 4th-order tensor)
# -----------------------------------------------------------------------------

class TensorTools:
    """Minimal tensor helpers inspired by homopy's tensor architecture."""

    SQRT2 = math.sqrt(2.0)
    SQRT2_INV = 1.0 / math.sqrt(2.0)

    # Mandel mapping order:
    # 0 -> 11, 1 -> 22, 2 -> 33, 3 -> 23, 4 -> 13, 5 -> 12
    PAIRS = (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    )

    @staticmethod
    def mandel_factor(i: int) -> float:
        return 1.0 if i < 3 else TensorTools.SQRT2

    @staticmethod
    def tensor2mandel(C: ArrayLike) -> ArrayLike:
        """Convert 4th-order tensor C_ijkl to Mandel 6x6 matrix."""
        M = np.zeros((6, 6), dtype=float)
        for a, (i, j) in enumerate(TensorTools.PAIRS):
            fa = TensorTools.mandel_factor(a)
            for b, (k, l) in enumerate(TensorTools.PAIRS):
                fb = TensorTools.mandel_factor(b)
                M[a, b] = fa * fb * C[i, j, k, l]
        return M

    @staticmethod
    def mandel2tensor(M: ArrayLike) -> ArrayLike:
        """Convert Mandel 6x6 matrix to a fully symmetric 4th-order tensor."""
        C = np.zeros((3, 3, 3, 3), dtype=float)
        for a, (i, j) in enumerate(TensorTools.PAIRS):
            fa = TensorTools.mandel_factor(a)
            for b, (k, l) in enumerate(TensorTools.PAIRS):
                fb = TensorTools.mandel_factor(b)
                val = M[a, b] / (fa * fb)
                C[i, j, k, l] = val
                C[j, i, k, l] = val
                C[i, j, l, k] = val
                C[j, i, l, k] = val
        return C

    @staticmethod
    def tensor_product(A: ArrayLike, B: ArrayLike) -> ArrayLike:
        """Matrix product in Mandel representation."""
        return A @ B

    @staticmethod
    def random_isotropic_N2() -> ArrayLike:
        return np.eye(3) / 3.0

    @staticmethod
    def random_isotropic_N4() -> ArrayLike:
        eye = np.eye(3)
        N4 = np.zeros((3, 3, 3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        N4[i, j, k, l] = (
                            eye[i, j] * eye[k, l]
                            + eye[i, k] * eye[j, l]
                            + eye[i, l] * eye[j, k]
                        ) / 15.0
        return N4

    @staticmethod
    def orientation_average(tensor3333: ArrayLike, N2: ArrayLike, N4: ArrayLike) -> ArrayLike:
        """
        Advani–Tucker style orientation average for a transversely isotropic 4th-order tensor.
        This is the same high-level idea used in homopy.
        """
        C = TensorTools.tensor2mandel(tensor3333)
        B1 = C[0, 0] + C[2, 2] - 2.0 * C[1, 2] - 4.0 * C[4, 4]
        B2 = C[1, 2] - C[0, 1]
        B3 = C[4, 4] - 0.5 * (C[0, 0] - C[0, 1])
        B4 = C[0, 1]
        B5 = 0.5 * (C[0, 0] - C[0, 1])

        code = {
            0: (0, 0),
            1: (1, 1),
            2: (2, 2),
            3: (1, 2),
            4: (0, 2),
            5: (0, 1),
        }

        C_avg = np.zeros((6, 6), dtype=float)
        for i in range(6):
            for j in range(6):
                ii, ij = code[i]
                ji, jj = code[j]

                term1 = B1 * N4[ii, ij, ji, jj]
                term2 = B2 * (
                    N2[ii, ij] * float(ji == jj) +
                    N2[ji, jj] * float(ii == ij)
                )
                term3 = B3 * (
                    N2[ii, ji] * float(ij == jj) +
                    N2[ii, jj] * float(ij == ji) +
                    N2[ij, ji] * float(ii == jj) +
                    N2[ij, jj] * float(ii == ji)
                )
                term4 = B4 * float(ii == ij) * float(ji == jj)
                term5 = B5 * (
                    float(ii == ji) * float(ij == jj) +
                    float(ii == jj) * float(ij == ji)
                )
                C_avg[i, j] = term1 + term2 + term3 + term4 + term5

        return TensorTools.mandel2tensor(C_avg)


# -----------------------------------------------------------------------------
# Isotropic stiffness
# -----------------------------------------------------------------------------

def isotropic_stiffness_from_KG(K: float, G: float) -> ArrayLike:
    """
    Return isotropic stiffness tensor in Mandel notation from bulk/shear moduli in Pa.
    """
    lam = K - 2.0 * G / 3.0
    C = np.zeros((3, 3, 3, 3), dtype=float)
    eye = np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (
                        lam * eye[i, j] * eye[k, l]
                        + G * (eye[i, k] * eye[j, l] + eye[i, l] * eye[j, k])
                    )
    return TensorTools.tensor2mandel(C)


def poisson_from_KG(K: float, G: float) -> float:
    return (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))


def bulk_shear_from_stiffness_tensor(C3333: ArrayLike) -> tuple[float, float]:
    """
    Extract isotropic-equivalent bulk and shear moduli from a 4th-order tensor using
    invariant projections:

        K = C_iijj / 9
        G = (C_ijij - C_iijj / 3) / 10

    For an exactly isotropic tensor this is exact.
    """
    c_iijj = 0.0
    c_ijij = 0.0
    for i in range(3):
        for j in range(3):
            c_iijj += C3333[i, i, j, j]
            c_ijij += C3333[i, j, i, j]
    K = c_iijj / 9.0
    G = (c_ijij - c_iijj / 3.0) / 10.0
    return float(K), float(G)


# -----------------------------------------------------------------------------
# Eshelby tensor for spheroids in isotropic matrix
# -----------------------------------------------------------------------------

def _g_spheroid(a_ratio: float) -> float:
    """
    Shape function used in the homopy / Tandon–Weng-style Eshelby components.

    a_ratio > 1  : prolate
    a_ratio = 1  : sphere
    a_ratio < 1  : oblate  (custom branch added here)
    """
    a = float(a_ratio)
    if a <= 0.0:
        raise ValueError("Aspect ratio must be positive.")

    if abs(a - 1.0) < 1e-10:
        return 2.0 / 3.0

    if a > 1.0:
        a2 = a * a
        return a / ((a2 - 1.0) ** 1.5) * (a * math.sqrt(a2 - 1.0) - math.acosh(a))

    # oblate branch
    b2 = 1.0 - a * a
    b = math.sqrt(max(b2, 0.0))
    return a / (b2 ** 1.5) * (math.acos(a) - a * b)


def eshelby_spheroid_isotropic_matrix(
    a_ratio: float,
    nu_matrix: float,
    return_dim: Literal["66", "3333"] = "66",
) -> ArrayLike:
    """
    Eshelby tensor for a spheroid embedded in an isotropic matrix.

    This follows the same component structure as homopy's spheroid/ellipsoid branch,
    but extends it to aspect ratios a < 1 via the oblate branch of g(a).

    Axis convention:
    - The spheroid symmetry axis is the x1-axis in the local inclusion frame.
    - Random isotropic orientation averaging is performed later at the stiffness level.
    """
    nu = float(nu_matrix)
    a = float(a_ratio)

    if not (-1.0 < nu < 0.5):
        raise ValueError("Matrix Poisson ratio must lie in (-1, 0.5).")

    S = np.zeros((3, 3, 3, 3), dtype=float)

    if abs(a - 1.0) < 1e-10:
        eye = np.eye(3)
        fac = 1.0 / (15.0 * (1.0 - nu))
        S = fac * (
            (5.0 * nu - 1.0) * np.einsum("ij,kl->ijkl", eye, eye)
            + (4.0 - 5.0 * nu)
            * (
                np.einsum("ik,jl->ijkl", eye, eye)
                + np.einsum("il,jk->ijkl", eye, eye)
            )
        )
        return TensorTools.tensor2mandel(S) if return_dim == "66" else S

    a2 = a * a
    g = _g_spheroid(a)

    # These are the same component formulas used by homopy for spheroids,
    # now evaluated with the custom oblate/prolate g(a).
    # Local frame: symmetry axis along x1.
    S[0, 0, 0, 0] = (
        1.0 / (2.0 * (1.0 - nu))
        * (
            1.0
            - 2.0 * nu
            + (3.0 * a2 - 1.0) / (a2 - 1.0)
            - (1.0 - 2.0 * nu + 3.0 * a2 / (a2 - 1.0)) * g
        )
    )

    S[1, 1, 1, 1] = S[2, 2, 2, 2] = (
        3.0 / (8.0 * (1.0 - nu)) * a2 / (a2 - 1.0)
        + 1.0 / (4.0 * (1.0 - nu))
        * (1.0 - 2.0 * nu - 9.0 / (4.0 * (a2 - 1.0))) * g
    )

    S[1, 1, 2, 2] = S[2, 2, 1, 1] = (
        1.0 / (4.0 * (1.0 - nu))
        * (
            a2 / (2.0 * (a2 - 1.0))
            - (1.0 - 2.0 * nu + 3.0 / (4.0 * (a2 - 1.0))) * g
        )
    )

    S[1, 1, 0, 0] = S[2, 2, 0, 0] = (
        -1.0 / (2.0 * (1.0 - nu)) * a2 / (a2 - 1.0)
        + 1.0 / (4.0 * (1.0 - nu))
        * (3.0 * a2 / (a2 - 1.0) - (1.0 - 2.0 * nu)) * g
    )

    S[0, 0, 1, 1] = S[0, 0, 2, 2] = (
        -1.0 / (2.0 * (1.0 - nu)) * (1.0 - 2.0 * nu + 1.0 / (a2 - 1.0))
        + 1.0 / (2.0 * (1.0 - nu))
        * (1.0 - 2.0 * nu + 3.0 / (2.0 * (a2 - 1.0))) * g
    )

    shear23 = (
        1.0 / (4.0 * (1.0 - nu))
        * (
            a2 / (2.0 * (a2 - 1.0))
            + (1.0 - 2.0 * nu - 3.0 / (4.0 * (a2 - 1.0))) * g
        )
    )
    S[1, 2, 1, 2] = S[2, 1, 2, 1] = S[2, 1, 1, 2] = S[1, 2, 2, 1] = shear23

    shear12 = (
        1.0 / (4.0 * (1.0 - nu))
        * (
            1.0
            - 2.0 * nu
            - (a2 + 1.0) / (a2 - 1.0)
            - 0.5 * (1.0 - 2.0 * nu - 3.0 * (a2 + 1.0) / (a2 - 1.0)) * g
        )
    )
    S[0, 1, 0, 1] = S[0, 2, 0, 2] = S[1, 0, 1, 0] = S[1, 0, 0, 1] = \
    S[0, 1, 1, 0] = S[2, 0, 2, 0] = S[2, 0, 0, 2] = S[0, 2, 2, 0] = shear12

    return TensorTools.tensor2mandel(S) if return_dim == "66" else S


# -----------------------------------------------------------------------------
# Mori–Tanaka homogenization
# -----------------------------------------------------------------------------

@dataclass
class ElasticPhase:
    K: float  # bulk modulus [Pa]
    G: float  # shear modulus [Pa]

    @property
    def stiffness66(self) -> ArrayLike:
        return isotropic_stiffness_from_KG(self.K, self.G)

    @property
    def nu(self) -> float:
        return poisson_from_KG(self.K, self.G)


@dataclass
class StrictMTPoreResult:
    Ceff66: ArrayLike
    Ceff3333: ArrayLike
    K_eff: float
    G_eff: float
    vp_m_s: float
    vs_m_s: float
    rho_eff_kg_m3: float


def strict_mt_elastic_random_spheroidal_pores(
    phi: float,
    a_ratio: float,
    matrix: ElasticPhase,
    inclusion: ElasticPhase,
    rho_matrix_kg_m3: float,
    rho_inclusion_kg_m3: float,
) -> StrictMTPoreResult:
    """
    Strict-ish elastic Mori–Tanaka for isotropic matrix + isotropic spheroidal pores/inclusions
    with random isotropic orientation.

    Parameters
    ----------
    phi : inclusion volume fraction / porosity
    a_ratio : spheroid aspect ratio c/a
              <1 oblate (crack-like), =1 sphere, >1 prolate
    matrix : matrix elastic phase (Pa)
    inclusion : pore/fluid elastic phase (Pa)
    rho_matrix_kg_m3, rho_inclusion_kg_m3 : densities used only for Vp/Vs

    Returns
    -------
    StrictMTPoreResult
    """
    if not (0.0 <= phi < 1.0):
        raise ValueError("phi must be in [0,1).")
    if a_ratio <= 0.0:
        raise ValueError("a_ratio must be positive.")

    Cm = matrix.stiffness66
    Ci = inclusion.stiffness66
    dC = Ci - Cm
    Cm_inv = np.linalg.inv(Cm)

    S66 = eshelby_spheroid_isotropic_matrix(a_ratio=a_ratio, nu_matrix=matrix.nu, return_dim="66")
    I6 = np.eye(6)

    # Dilute concentration tensor in the local inclusion frame
    A_dil = np.linalg.inv(I6 + TensorTools.tensor_product(S66, TensorTools.tensor_product(Cm_inv, dC)))

    # Orientation-average the weighted strain concentration tensor
    weighted_A_local_66 = TensorTools.tensor_product(dC, A_dil)
    weighted_A_local_3333 = TensorTools.mandel2tensor(weighted_A_local_66)

    N2 = TensorTools.random_isotropic_N2()
    N4 = TensorTools.random_isotropic_N4()
    weighted_A_avg_3333 = TensorTools.orientation_average(weighted_A_local_3333, N2, N4)
    weighted_A_avg_66 = TensorTools.tensor2mandel(weighted_A_avg_3333)

    # Remove weighting by dC (same logic as homopy / Benveniste implementation)
    dC_inv = np.linalg.inv(dC)
    A_avg = TensorTools.tensor_product(dC_inv, weighted_A_avg_66)

    # Standard single-inclusion Mori–Tanaka formula
    A_mt = TensorTools.tensor_product(
        A_avg,
        np.linalg.inv((1.0 - phi) * I6 + phi * A_avg),
    )

    Ceff66 = Cm + phi * TensorTools.tensor_product(dC, A_mt)
    Ceff3333 = TensorTools.mandel2tensor(Ceff66)

    K_eff, G_eff = bulk_shear_from_stiffness_tensor(Ceff3333)

    rho_eff = (1.0 - phi) * rho_matrix_kg_m3 + phi * rho_inclusion_kg_m3
    vp = math.sqrt(max((K_eff + 4.0 * G_eff / 3.0) / rho_eff, 1e-30))
    vs = math.sqrt(max(G_eff / rho_eff, 1e-30))

    return StrictMTPoreResult(
        Ceff66=Ceff66,
        Ceff3333=Ceff3333,
        K_eff=K_eff,
        G_eff=G_eff,
        vp_m_s=vp,
        vs_m_s=vs,
        rho_eff_kg_m3=rho_eff,
    )


# -----------------------------------------------------------------------------
# Convenience helper for fluid-filled pores in rocks
# -----------------------------------------------------------------------------

def fluid_phase_from_bulk(K_fluid_pa: float, rho_fluid_kg_m3: float) -> tuple[ElasticPhase, float]:
    """
    Fluid phase with G ~= 0.
    """
    return ElasticPhase(K=K_fluid_pa, G=1e-9), rho_fluid_kg_m3


if __name__ == "__main__":
    # Small smoke test
    Km = 76.8e9
    Gm = 32.0e9
    Kf = 2.2e9
    rho_m = 2710.0
    rho_f = 1000.0
    phi = 0.15

    matrix = ElasticPhase(K=Km, G=Gm)
    fluid = ElasticPhase(K=Kf, G=1e-9)

    rows = []
    for ar in [0.1, 1.0, 10.0]:
        res = strict_mt_elastic_random_spheroidal_pores(
            phi=phi,
            a_ratio=ar,
            matrix=matrix,
            inclusion=fluid,
            rho_matrix_kg_m3=rho_m,
            rho_inclusion_kg_m3=rho_f,
        )
        rows.append({
            "aspect_ratio": ar,
            "K_eff_GPa": res.K_eff / 1e9,
            "G_eff_GPa": res.G_eff / 1e9,
            "Vp_m_s": res.vp_m_s,
            "Vs_m_s": res.vs_m_s,
        })

    import pandas as pd
    print(pd.DataFrame(rows))
