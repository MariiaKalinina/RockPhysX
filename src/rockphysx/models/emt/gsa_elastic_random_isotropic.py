from __future__ import annotations

"""
Elastic GSA / OSP module for isotropic random spheroidal pores.

Scope
-----
This module implements the *isotropic random* branch needed for the dissertation:
- isotropic matrix / isotropic inclusion phases,
- spheroidal pore shape parameter via a single aspect ratio,
- random orientation averaging,
- effective isotropic outputs K_eff, G_eff, Vp, Vs.

Implementation strategy
-----------------------
The module uses the user-provided Fortran kernels as the authoritative backend:
- GREEN_ANAL_VTI.f90    -> local 4th-order Green/polarization tensor in the VTI frame
- subr_IZOTR.f90        -> isotropic random orientation average of a 4th-order tensor

The Python side provides:
1. build/compile helpers for the Fortran backend,
2. tensor <-> Mandel helpers,
3. isotropic stiffness utilities,
4. GSA closure for a two-phase isotropic random medium.

Current closure
---------------
This first working implementation uses the *matrix comparison body* closure:
    C0 = Cm

Then for the inclusion phase:
    A_p^loc = [I + G0 : (C_p - C0)]^{-1}

The random isotropic average is:
    A_p = < A_p^loc >_Omega

The effective stiffness is assembled as:
    C* = [ (1-phi) C_m + phi C_p : A_p ] : [ (1-phi) I + phi A_p ]^{-1}

This matches the transport GSA structure already used in the project:
    < X A > < A >^{-1}

Notes
-----
- The module intentionally focuses on the isotropic random case only.
- The comparison-body choice is exposed, but only "matrix" is implemented here.
- Random averaging is performed by the uploaded Fortran izotr routine.
- The Fortran kernels are compiled on first use with gfortran.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import ctypes
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np


# -----------------------------------------------------------------------------
# Basic tensor helpers (Mandel notation <-> 4th-order tensor)
# -----------------------------------------------------------------------------

_PAIRS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)


def _mandel_factor(i: int) -> float:
    return 1.0 if i < 3 else math.sqrt(2.0)

def voigt66_to_mandel66(C: np.ndarray) -> np.ndarray:
    """
    Convert a 6x6 matrix from engineering-Voigt to Mandel basis.

    In Mandel notation shear components use sqrt(2) scaling.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Expected a 6x6 matrix.")
    f = np.array([_mandel_factor(i) for i in range(6)], dtype=float)
    return (f[:, None] * C) * f[None, :]


def mandel66_to_voigt66(C: np.ndarray) -> np.ndarray:
    """
    Convert a 6x6 matrix from Mandel basis to engineering-Voigt basis.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Expected a 6x6 matrix.")
    f = np.array([_mandel_factor(i) for i in range(6)], dtype=float)
    return C / (f[:, None] * f[None, :])



def tensor4_to_mandel(C: np.ndarray) -> np.ndarray:
    M = np.zeros((6, 6), dtype=float)
    for a, (i, j) in enumerate(_PAIRS):
        fa = _mandel_factor(a)
        for b, (k, l) in enumerate(_PAIRS):
            fb = _mandel_factor(b)
            M[a, b] = fa * fb * C[i, j, k, l]
    return M



def mandel_to_tensor4(M: np.ndarray) -> np.ndarray:
    C = np.zeros((3, 3, 3, 3), dtype=float)
    for a, (i, j) in enumerate(_PAIRS):
        fa = _mandel_factor(a)
        for b, (k, l) in enumerate(_PAIRS):
            fb = _mandel_factor(b)
            val = M[a, b] / (fa * fb)
            C[i, j, k, l] = val
            C[j, i, k, l] = val
            C[i, j, l, k] = val
            C[j, i, l, k] = val
    return C


# -----------------------------------------------------------------------------
# Elastic phase and isotropic stiffness helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ElasticPhase:
    K: float  # Pa
    G: float  # Pa


@dataclass(frozen=True)
class ElasticEffectiveResult:
    C_eff_66: np.ndarray
    C_eff_3333: np.ndarray
    K_eff: float
    G_eff: float
    rho_eff_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    comparison_body: str



def isotropic_stiffness_66_from_KG(K: float, G: float) -> np.ndarray:
    K = float(K)
    G = float(G)
    if K <= 0.0 or G < 0.0:
        raise ValueError("K must be positive and G must be non-negative.")
    lam = K - 2.0 * G / 3.0
    C = np.zeros((6, 6), dtype=float)
    C[:3, :3] = lam
    np.fill_diagonal(C[:3, :3], lam + 2.0 * G)
    np.fill_diagonal(C[3:, 3:], G)
    return C



def isotropic_tensor4_from_KG(K: float, G: float) -> np.ndarray:
    lam = K - 2.0 * G / 3.0
    eye = np.eye(3)
    C = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (
                        lam * eye[i, j] * eye[k, l]
                        + G * (eye[i, k] * eye[j, l] + eye[i, l] * eye[j, k])
                    )
    return C



def bulk_shear_from_tensor4(C: np.ndarray) -> tuple[float, float]:
    """
    Exact for isotropic tensors:
        K = C_iijj / 9
        G = (C_ijij - C_iijj/3) / 10
    """
    c_iijj = 0.0
    c_ijij = 0.0
    for i in range(3):
        for j in range(3):
            c_iijj += C[i, i, j, j]
            c_ijij += C[i, j, i, j]
    K = c_iijj / 9.0
    G = (c_ijij - c_iijj / 3.0) / 10.0
    return float(K), float(G)



def velocities_from_KG_rho(K: float, G: float, rho: float) -> tuple[float, float]:
    K = float(K)
    G = float(G)
    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if K <= 0.0 or G < 0.0:
        raise ValueError("K must be positive and G must be non-negative.")
    vp = math.sqrt((K + 4.0 * G / 3.0) / rho)
    vs = math.sqrt(max(G, 0.0) / rho)
    return float(vp), float(vs)


# -----------------------------------------------------------------------------
# Fortran backend builder
# -----------------------------------------------------------------------------

@dataclass
class _Backend:
    lib: ctypes.CDLL
    green_vti: object
    izotr: object


def _find_repo_root(start: Path) -> Path | None:
    """
    Best-effort repository root discovery (looks for pyproject.toml or .git).
    """
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _resolve_fortran_path(path: str | Path, *, label: str, must_exist: bool = True) -> Path:
    """
    Resolve a Fortran source path robustly.

    Accepts:
    - absolute paths,
    - relative paths (searched relative to CWD, repo root, and this module dir),
    - bare filenames (searched in the same locations).
    """
    p_in = Path(path).expanduser()
    if p_in.is_absolute():
        p = p_in
        if must_exist and not p.exists():
            raise FileNotFoundError(f"{label} Fortran file not found: {p}")
        return p.resolve()

    search_dirs: list[Path] = []
    search_dirs.append(Path.cwd())
    mod_dir = Path(__file__).resolve().parent
    search_dirs.append(mod_dir)
    repo = _find_repo_root(mod_dir)
    if repo is not None:
        search_dirs.append(repo)

    # Try direct relative to each search dir.
    for d in search_dirs:
        cand = (d / p_in).resolve()
        if cand.exists():
            return cand

    # If it's a bare filename, try a couple of common subfolders.
    if len(p_in.parts) == 1:
        for d in search_dirs:
            for sub in ("fortran", "src", "test_new", "tests"):
                cand = (d / sub / p_in.name).resolve()
                if cand.exists():
                    return cand

    if must_exist:
        msg = [f"{label} Fortran file not found: {p_in}"]
        msg.append("Searched in:")
        for d in search_dirs:
            msg.append(f"  - {d}")
        raise FileNotFoundError("\n".join(msg))
    return (search_dirs[0] / p_in).resolve()



def _write_support_sources(workdir: Path) -> tuple[Path, Path, Path]:
    glbls = workdir / "glbls.f90"
    glbls.write_text(
        """
module glbls
  implicit none
  real(8) :: aa1 = 1d0
  real(8) :: aa2 = 1d0
  real(8) :: pi = 3.1415926535897932384626433832795d0
  integer :: ind_err = 0
end module glbls
""".strip() + "\n",
        encoding="utf-8",
    )

    izotr_fixed = workdir / "subr_IZOTR_fixed.f90"
    izotr_fixed.write_text(
        """
subroutine izotr(c,c_iz)
  use glbls
  implicit none
  real(8), intent(in) :: c(3,3,3,3)
  real(8), intent(out) :: c_iz(3,3,3,3)
  real(8) :: c11,c22,c33,c12,c13,c23,c44,c55,c66,t1,t20

  c_iz = 0d0

  c11=c(1,1,1,1)
  c22=c(2,2,2,2)
  c33=c(3,3,3,3)
  c12=c(1,1,2,2)
  c13=c(1,1,3,3)
  c23=c(2,2,3,3)
  c44=c(2,3,2,3)
  c55=c(1,3,1,3)
  c66=c(1,2,1,2)

  t1 = 0.3141593D1**2
  t20 = 32D0/15D0*c44*t1+8D0/5D0*c22*t1+16D0/15D0*c23*t1+32D0/15D0*c55*t1+8D0/5D0*c11*t1+32D0/15D0*c66*t1+16D0/15D0*c13*t1+16D0/15D0*c12*t1+8D0/5D0*c33*t1
  c_iz(1,1,1,1)=t20/8D0/pi/pi

  t1 = 0.3141593D1**2
  t20 = 32D0/15D0*c12*t1+8D0/15D0*c11*t1+8D0/15D0*c22*t1-16D0/15D0*c66*t1+32D0/15D0*c13*t1+32D0/15D0*c23*t1-16D0/15D0*c44*t1+8D0/15D0*c33*t1-16D0/15D0*c55*t1
  c_iz(1,1,2,2)=t20/8D0/pi/pi
  c_iz(2,2,1,1)=c_iz(1,1,2,2)

  t1 = 0.3141593D1**2
  t20 = -8D0/15D0*c12*t1+8D0/15D0*c22*t1+8D0/5D0*c44*t1+8D0/5D0*c55*t1+8D0/15D0*c11*t1+8D0/5D0*c66*t1-8D0/15D0*c13*t1-8D0/15D0*c23*t1+8D0/15D0*c33*t1
  c_iz(1,3,1,3)=t20/8D0/pi/pi

  c_iz(2,2,2,2)=c_iz(1,1,1,1)
  c_iz(3,3,3,3)=c_iz(1,1,1,1)
  c_iz(1,1,3,3)=c_iz(1,1,2,2)
  c_iz(2,2,3,3)=c_iz(1,1,2,2)
  c_iz(3,3,1,1)=c_iz(1,1,2,2)
  c_iz(3,3,2,2)=c_iz(1,1,2,2)

  c_iz(1,3,3,1)=c_iz(1,3,1,3)
  c_iz(3,1,3,1)=c_iz(1,3,1,3)
  c_iz(3,1,1,3)=c_iz(1,3,1,3)
  c_iz(1,2,1,2)=c_iz(1,3,1,3)
  c_iz(1,2,2,1)=c_iz(1,3,1,3)
  c_iz(2,1,1,2)=c_iz(1,3,1,3)
  c_iz(2,1,2,1)=c_iz(1,3,1,3)
  c_iz(2,3,2,3)=c_iz(1,3,1,3)
  c_iz(2,3,3,2)=c_iz(1,3,1,3)
  c_iz(3,2,2,3)=c_iz(1,3,1,3)
  c_iz(3,2,3,2)=c_iz(1,3,1,3)
end subroutine izotr
""".strip() + "\n",
        encoding="utf-8",
    )

    wrapper = workdir / "gsa_wrapper.f90"
    wrapper.write_text(
        """
module gsa_wrapper_mod
  use glbls
  implicit none
contains
  subroutine green_vti_wrap(a1, a2, cc, greent)
    real(8), intent(in) :: a1, a2
    real(8), intent(in) :: cc(6,6)
    real(8), intent(out) :: greent(3,3,3,3)
    aa1 = a1
    aa2 = a2
    call GREEN_ANAL_VTI(cc, greent)
  end subroutine green_vti_wrap

  subroutine izotr_wrap(c, c_iz)
    real(8), intent(in) :: c(3,3,3,3)
    real(8), intent(out) :: c_iz(3,3,3,3)
    call izotr(c, c_iz)
  end subroutine izotr_wrap
end module gsa_wrapper_mod
""".strip() + "\n",
        encoding="utf-8",
    )
    return glbls, izotr_fixed, wrapper



def build_backend(
    *,
    green_fortran: str | Path = "GREEN_ANAL_VTI.f90",
    izotr_fortran: str | Path | None = None,
    output_library: str | Path | None = None,
    force_rebuild: bool = False,
) -> _Backend:
    """
    Compile the uploaded Fortran kernels into a shared library and return ctypes wrappers.
    """
    green_fortran = _resolve_fortran_path(green_fortran, label="GREEN", must_exist=True)
    # NOTE: izotr_fortran is kept for backward compatibility, but the build uses
    # an internal, gfortran-compatible `izotr` implementation written by
    # `_write_support_sources()`. We therefore do not require an external
    # `subr_IZOTR.f90` file to exist.
    if izotr_fortran is not None:
        _ = _resolve_fortran_path(izotr_fortran, label="IZOTR", must_exist=True)

    if output_library is None:
        output_library = green_fortran.with_name("libgsa_elastic_fortran.so")
    output_library = Path(output_library).expanduser().resolve()

    if force_rebuild or not output_library.exists():
        with tempfile.TemporaryDirectory(prefix="gsa_build_") as td:
            wd = Path(td)
            glbls, izotr_fixed, wrapper = _write_support_sources(wd)

            compiler = os.environ.get("GFORTRAN", "gfortran")
            compiler_path = shutil.which(compiler)
            if compiler_path is None:
                raise RuntimeError(
                    "Fortran compiler not found (needed to build the elastic GSA backend).\n"
                    f"Tried: {compiler!r} (set via $GFORTRAN)\n\n"
                    "Install options (macOS):\n"
                    "  - Conda (recommended): `conda install -c conda-forge gfortran`\n"
                    "    (or: `conda install -c conda-forge fortran-compiler`)\n"
                    "  - Homebrew: `brew install gcc` (provides `gfortran`)\n\n"
                    "After installation, re-run with `--force-rebuild-backend`."
                )

            cmd = [
                compiler_path,
                "-shared",
                "-fPIC",
                "-O2",
                str(glbls),
                str(green_fortran),
                str(izotr_fixed),
                str(wrapper),
                "-o",
                str(output_library),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Failed to invoke the Fortran compiler while building the GSA backend.\n"
                    f"Command: {' '.join(map(str, cmd))}\n"
                    "Hint: ensure `gfortran` is installed and on PATH, or set $GFORTRAN to its full path."
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Failed to compile the Fortran GSA backend.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{exc.stdout}\n\n"
                    f"stderr:\n{exc.stderr}"
                ) from exc

    lib = ctypes.CDLL(str(output_library))

    green_vti = getattr(lib, "__gsa_wrapper_mod_MOD_green_vti_wrap")
    green_vti.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.float64, shape=(6, 6), flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, shape=(3, 3, 3, 3), flags="C_CONTIGUOUS"),
    ]
    green_vti.restype = None

    izotr = getattr(lib, "__gsa_wrapper_mod_MOD_izotr_wrap")
    izotr.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, shape=(3, 3, 3, 3), flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, shape=(3, 3, 3, 3), flags="C_CONTIGUOUS"),
    ]
    izotr.restype = None

    return _Backend(lib=lib, green_vti=green_vti, izotr=izotr)


# -----------------------------------------------------------------------------
# Fortran-backed operators
# -----------------------------------------------------------------------------


def green_tensor_vti_fortran(C0_66: np.ndarray, aspect_ratio: float, backend: _Backend) -> np.ndarray:
    """
    Return the local 4th-order Green tensor from the uploaded Fortran kernel.

    The uploaded Fortran uses two equal shape parameters aa1, aa2 for the VTI/spheroidal case.

    IMPORTANT: in the original Fortran, aa1/aa2 correspond to the ratio of the
    transverse semi-axis to the symmetry-axis semi-axis (a1/a3). For our
    convention ``aspect_ratio = a3/a1``, the correct mapping is:

        aa1 = aa2 = a1/a3 = 1/aspect_ratio
    """
    C0_66 = np.ascontiguousarray(C0_66, dtype=np.float64)
    out = np.zeros((3, 3, 3, 3), dtype=np.float64)
    ar = float(aspect_ratio)
    if not np.isfinite(ar) or ar <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite.")
    aa = 1.0 / ar
    a1 = ctypes.c_double(float(aa))
    a2 = ctypes.c_double(float(aa))
    backend.green_vti(ctypes.byref(a1), ctypes.byref(a2), C0_66, out)
    return out



def izotr_average_fortran(T: np.ndarray, backend: _Backend) -> np.ndarray:
    Tin = np.ascontiguousarray(T, dtype=np.float64)
    Tout = np.zeros((3, 3, 3, 3), dtype=np.float64)
    backend.izotr(Tin, Tout)
    return Tout


# -----------------------------------------------------------------------------
# GSA / OSP elastic closure
# -----------------------------------------------------------------------------


def local_concentration_tensor_mandel(
    phase: ElasticPhase,
    comparison_phase: ElasticPhase,
    aspect_ratio: float,
    backend: _Backend,
    *,
    sign: int = -1,
) -> np.ndarray:
    """
    Local concentration tensor in Mandel notation:
        A_loc = [I - G0 : (C_i - C0)]^{-1}

    Notes
    -----
    The Fortran Green operator returned by ``GREEN_ANAL_VTI`` is negative for a
    stable comparison medium. Using the conventional minus sign yields strain
    localization factors > 1 for soft pores/fluids, which matches physical
    expectations and reference OSP/GSA implementations.
    """
    C0_66 = isotropic_stiffness_66_from_KG(comparison_phase.K, comparison_phase.G)
    C0_3333 = isotropic_tensor4_from_KG(comparison_phase.K, comparison_phase.G)
    Ci_3333 = isotropic_tensor4_from_KG(phase.K, phase.G)
    dC = Ci_3333 - C0_3333

    G0_3333 = green_tensor_vti_fortran(C0_66, aspect_ratio, backend)
    G0_66 = tensor4_to_mandel(G0_3333)
    dC_66 = tensor4_to_mandel(dC)

    if sign not in (-1, +1):
        raise ValueError("sign must be +1 or -1.")
    A_loc = np.linalg.inv(np.eye(6) + float(sign) * (G0_66 @ dC_66))
    return A_loc



def random_isotropic_average_mandel(A_loc_66: np.ndarray, backend: _Backend) -> np.ndarray:
    A_loc_3333 = mandel_to_tensor4(A_loc_66)
    A_iso_3333 = izotr_average_fortran(A_loc_3333, backend)
    return tensor4_to_mandel(A_iso_3333)



def gsa_effective_stiffness_random_two_phase(
    *,
    phi: float,
    matrix: ElasticPhase,
    inclusion: ElasticPhase,
    pore_aspect_ratio: float,
    backend: _Backend,
    comparison_body: Literal["matrix", "bayuk_linear_mix"] = "matrix",
    k_connectivity: float | None = None,
    sign: int = -1,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Two-phase elastic GSA / OSP closure for isotropic random spheroidal inclusions.

    Supported comparison bodies:
        - "matrix": C0 = Cm (one-shot).
        - "bayuk_linear_mix": C0 = (1-f) Cm + f Cp with f = k * phi (clipped to [0, 1]).

    Effective stiffness:
        C* = [ (1-phi) C_m + phi C_p : A_p ] : [ (1-phi) I + phi A_p ]^{-1}

    with A_p being the random isotropic average of the local concentration tensor.
    """
    phi = float(phi)
    if not (0.0 <= phi <= 1.0):
        raise ValueError(f"phi must lie in [0, 1], got {phi!r}")
    if pore_aspect_ratio <= 0.0:
        raise ValueError("pore_aspect_ratio must be positive.")
    if comparison_body not in ("matrix", "bayuk_linear_mix"):
        raise ValueError("comparison_body must be 'matrix' or 'bayuk_linear_mix'.")
    if comparison_body == "bayuk_linear_mix":
        if k_connectivity is None:
            raise ValueError("k_connectivity is required for comparison_body='bayuk_linear_mix'.")
        k = float(k_connectivity)
        if not np.isfinite(k) or k < 0.0:
            raise ValueError("k_connectivity must be non-negative and finite.")

    # Engineering-Voigt stiffness (for bookkeeping / plotting); convert to Mandel
    # for all algebra with A and G operators.
    C_m_voigt = isotropic_stiffness_66_from_KG(matrix.K, matrix.G)
    C_p_voigt = isotropic_stiffness_66_from_KG(inclusion.K, inclusion.G)
    C_m_66 = voigt66_to_mandel66(C_m_voigt)
    C_p_66 = voigt66_to_mandel66(C_p_voigt)

    if comparison_body == "matrix":
        # For matrix comparison body the matrix phase concentration tensor is identity.
        A_m_66 = np.eye(6, dtype=float)
        A_p_loc_66 = local_concentration_tensor_mandel(
            phase=inclusion,
            comparison_phase=matrix,
            aspect_ratio=pore_aspect_ratio,
            backend=backend,
            sign=sign,
        )
        A_p_66 = random_isotropic_average_mandel(A_p_loc_66, backend)
    else:
        # Bayuk / friability-style linear-mix comparison body:
        #   C0 = (1-f) Cm + f Cp,  f = k * phi, clipped.
        f = float(np.clip(float(k_connectivity) * phi, 0.0, 1.0))
        Kc = (1.0 - f) * float(matrix.K) + f * float(inclusion.K)
        Gc = (1.0 - f) * float(matrix.G) + f * float(inclusion.G)
        comparison = ElasticPhase(K=Kc, G=Gc)

        A_m_loc_66 = local_concentration_tensor_mandel(
            phase=matrix,
            comparison_phase=comparison,
            aspect_ratio=pore_aspect_ratio,
            backend=backend,
            sign=sign,
        )
        A_p_loc_66 = local_concentration_tensor_mandel(
            phase=inclusion,
            comparison_phase=comparison,
            aspect_ratio=pore_aspect_ratio,
            backend=backend,
            sign=sign,
        )
        A_m_66 = random_isotropic_average_mandel(A_m_loc_66, backend)
        A_p_66 = random_isotropic_average_mandel(A_p_loc_66, backend)

    numerator = (1.0 - phi) * (C_m_66 @ A_m_66) + phi * (C_p_66 @ A_p_66)
    denominator = (1.0 - phi) * A_m_66 + phi * A_p_66
    C_eff_mandel = numerator @ np.linalg.inv(denominator)
    C_eff_66 = mandel66_to_voigt66(C_eff_mandel)
    C_eff_3333 = mandel_to_tensor4(C_eff_mandel)
    K_eff, G_eff = bulk_shear_from_tensor4(C_eff_3333)
    return C_eff_66, C_eff_3333, K_eff, G_eff


def gsa_effective_stiffness_aligned_vti_two_phase(
    *,
    phi: float,
    matrix: ElasticPhase,
    inclusion: ElasticPhase,
    pore_aspect_ratio: float,
    backend: _Backend,
    comparison_body: Literal["matrix", "self_consistent", "bayuk_linear_mix"] = "matrix",
    k_connectivity: float | None = None,
    sign: int = -1,
    tol: float = 1e-10,
    max_iter: int = 300,
    relaxation: float = 1.0,
    initial_C0_voigt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-phase elastic GSA for aligned spheroidal inclusions (VTI result).

    This corresponds to the "однонаправленные пустоты" case: all spheroidal
    pores share the same symmetry axis (local axis 3 in GREEN_ANAL_VTI).

    Supported comparison bodies:
        - "matrix": C0 = Cm (one-shot).
        - "bayuk_linear_mix": C0 = (1-f) Cm + f Cp with f = k * phi (clipped to [0, 1]).
        - "self_consistent": C0 = C* solved by fixed-point.

    Effective stiffness:
        C* = [ (1-phi) C_m + phi C_p : A_p ] : [ (1-phi) I + phi A_p ]^{-1}

    where A_p is the *local* concentration tensor (no isotropic random average).
    """
    phi = float(phi)
    if not (0.0 <= phi <= 1.0):
        raise ValueError(f"phi must lie in [0, 1], got {phi!r}")
    if pore_aspect_ratio <= 0.0:
        raise ValueError("pore_aspect_ratio must be positive.")
    if sign not in (-1, +1):
        raise ValueError("sign must be +1 or -1.")
    omega = float(relaxation)
    if not np.isfinite(omega) or omega <= 0.0 or omega > 1.0:
        raise ValueError("relaxation must lie in (0, 1].")

    C_m_voigt = isotropic_stiffness_66_from_KG(matrix.K, matrix.G)
    C_p_voigt = isotropic_stiffness_66_from_KG(inclusion.K, inclusion.G)
    C_m_66 = voigt66_to_mandel66(C_m_voigt)
    C_p_66 = voigt66_to_mandel66(C_p_voigt)
    A_m_66 = np.eye(6, dtype=float)

    if comparison_body == "matrix":
        A_p_loc_66 = local_concentration_tensor_mandel(
            phase=inclusion,
            comparison_phase=matrix,
            aspect_ratio=pore_aspect_ratio,
            backend=backend,
            sign=sign,
        )
        numerator = (1.0 - phi) * (C_m_66 @ A_m_66) + phi * (C_p_66 @ A_p_loc_66)
        denominator = (1.0 - phi) * A_m_66 + phi * A_p_loc_66
        C_eff_mandel = numerator @ np.linalg.inv(denominator)
        C_eff_66 = mandel66_to_voigt66(C_eff_mandel)
        return C_eff_66, mandel_to_tensor4(C_eff_mandel)

    if comparison_body == "bayuk_linear_mix":
        if k_connectivity is None:
            raise ValueError("k_connectivity is required for comparison_body='bayuk_linear_mix'.")
        k = float(k_connectivity)
        if not np.isfinite(k) or k < 0.0:
            raise ValueError("k_connectivity must be non-negative and finite.")

        f = float(np.clip(k * phi, 0.0, 1.0))
        C0_mandel = (1.0 - f) * C_m_66 + f * C_p_66
        C0_voigt = mandel66_to_voigt66(C0_mandel)
        C0_3333 = mandel_to_tensor4(C0_mandel)

        G0_3333 = green_tensor_vti_fortran(C0_voigt, pore_aspect_ratio, backend)
        G0_66 = tensor4_to_mandel(G0_3333)

        C_m_3333 = isotropic_tensor4_from_KG(matrix.K, matrix.G)
        C_p_3333 = isotropic_tensor4_from_KG(inclusion.K, inclusion.G)

        I6 = np.eye(6, dtype=float)
        dC_m_66 = tensor4_to_mandel(C_m_3333 - C0_3333)
        dC_p_66 = tensor4_to_mandel(C_p_3333 - C0_3333)

        A_m_loc_66 = np.linalg.inv(I6 + float(sign) * (G0_66 @ dC_m_66))
        A_p_loc_66 = np.linalg.inv(I6 + float(sign) * (G0_66 @ dC_p_66))

        numerator = (1.0 - phi) * (C_m_66 @ A_m_loc_66) + phi * (C_p_66 @ A_p_loc_66)
        denominator = (1.0 - phi) * A_m_loc_66 + phi * A_p_loc_66
        C_eff_mandel = numerator @ np.linalg.inv(denominator)
        C_eff_66 = mandel66_to_voigt66(C_eff_mandel)
        return C_eff_66, mandel_to_tensor4(C_eff_mandel)

    # Self-consistent aligned VTI: iterate comparison body stiffness (full VTI).
    #
    # Fixed-point form:
    #   Given C0 (comparison body), compute phase local concentration tensors
    #   A_m(C0), A_p(C0), then update
    #       C_new = < C_i : A_i > : < A_i >^{-1}.
    #
    # This is the aligned-pore analogue of the GSA / coherent-potential update.
    if initial_C0_voigt is not None:
        C0v = np.asarray(initial_C0_voigt, dtype=float)
        if C0v.shape != (6, 6):
            raise ValueError("initial_C0_voigt must have shape (6,6).")
        C0_mandel = voigt66_to_mandel66(C0v)
    else:
        C0_mandel = np.array(C_m_66, dtype=float)  # initial guess: matrix stiffness
    I6 = np.eye(6, dtype=float)

    C_m_3333 = isotropic_tensor4_from_KG(matrix.K, matrix.G)
    C_p_3333 = isotropic_tensor4_from_KG(inclusion.K, inclusion.G)

    for _ in range(int(max_iter)):
        # Comparison stiffness to feed the Fortran Green kernel must be in
        # engineering Voigt; keep all algebra in Mandel.
        C0_voigt = mandel66_to_voigt66(C0_mandel)
        C0_3333 = mandel_to_tensor4(C0_mandel)

        G0_3333 = green_tensor_vti_fortran(C0_voigt, pore_aspect_ratio, backend)
        G0_66 = tensor4_to_mandel(G0_3333)

        dC_m_66 = tensor4_to_mandel(C_m_3333 - C0_3333)
        dC_p_66 = tensor4_to_mandel(C_p_3333 - C0_3333)

        A_m_loc_66 = np.linalg.inv(I6 + float(sign) * (G0_66 @ dC_m_66))
        A_p_loc_66 = np.linalg.inv(I6 + float(sign) * (G0_66 @ dC_p_66))

        numerator = (1.0 - phi) * (C_m_66 @ A_m_loc_66) + phi * (C_p_66 @ A_p_loc_66)
        denominator = (1.0 - phi) * A_m_loc_66 + phi * A_p_loc_66
        C_new = numerator @ np.linalg.inv(denominator)  # Mandel

        C_upd = (1.0 - omega) * C0_mandel + omega * C_new
        delta = float(np.max(np.abs(C_upd - C0_mandel)))
        if delta <= float(tol):
            C_eff_voigt = mandel66_to_voigt66(C_upd)
            return C_eff_voigt, mandel_to_tensor4(C_upd)
        C0_mandel = C_upd

    raise RuntimeError(
        "Aligned VTI self-consistent iteration did not converge. "
        "Try smaller `relaxation` (e.g. 0.2-0.5) and/or a larger `max_iter`."
    )



def gsa_elastic_random_spheroidal_pores(
    phi: float,
    a_ratio: float,
    matrix: ElasticPhase,
    inclusion: ElasticPhase,
    rho_matrix_kg_m3: float,
    rho_inclusion_kg_m3: float,
    *,
    comparison_body: Literal["matrix"] = "matrix",
    backend: Optional[_Backend] = None,
    green_fortran: str | Path = "GREEN_ANAL_VTI.f90",
    izotr_fortran: str | Path | None = None,
    output_library: str | Path | None = None,
    force_rebuild_backend: bool = False,
) -> ElasticEffectiveResult:
    """
    Main user-facing function for the dissertation branch.

    Parameters
    ----------
    phi
        Inclusion fraction / porosity.
    a_ratio
        Spheroidal aspect ratio of pores.
    matrix, inclusion
        Elastic phases in Pa.
    rho_matrix_kg_m3, rho_inclusion_kg_m3
        Phase densities for velocity calculation.
    comparison_body
        Only "matrix" is implemented here.
    backend
        Optional already-built Fortran backend.
    """
    if backend is None:
        backend = build_backend(
            green_fortran=green_fortran,
            izotr_fortran=izotr_fortran,
            output_library=output_library,
            force_rebuild=force_rebuild_backend,
        )

    C_eff_66, C_eff_3333, K_eff, G_eff = gsa_effective_stiffness_random_two_phase(
        phi=phi,
        matrix=matrix,
        inclusion=inclusion,
        pore_aspect_ratio=a_ratio,
        backend=backend,
        comparison_body=comparison_body,
    )

    rho_eff = (1.0 - float(phi)) * float(rho_matrix_kg_m3) + float(phi) * float(rho_inclusion_kg_m3)
    vp, vs = velocities_from_KG_rho(K_eff, G_eff, rho_eff)

    return ElasticEffectiveResult(
        C_eff_66=C_eff_66,
        C_eff_3333=C_eff_3333,
        K_eff=K_eff,
        G_eff=G_eff,
        rho_eff_kg_m3=rho_eff,
        vp_m_s=vp,
        vs_m_s=vs,
        comparison_body=comparison_body,
    )


__all__ = [
    "ElasticPhase",
    "ElasticEffectiveResult",
    "build_backend",
    "green_tensor_vti_fortran",
    "izotr_average_fortran",
    "gsa_effective_stiffness_random_two_phase",
    "gsa_effective_stiffness_aligned_vti_two_phase",
    "gsa_elastic_random_spheroidal_pores",
    "isotropic_stiffness_66_from_KG",
    "isotropic_tensor4_from_KG",
    "bulk_shear_from_tensor4",
    "velocities_from_KG_rho",
]
