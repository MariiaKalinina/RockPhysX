from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa
from rockphysx.models.emt import gsa_elastic_general as gsa_gen
from vti_phase_velocities_vs_angle_check import vti_phase_velocities_xz_plane
from gsa_polycrystal_case1_solid_inclusions_in_clay_matrix import (
    gsa_case1_matrix_comparison_effective_stiffness,
)


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _isotropic_voigt66_from_KG(K_gpa: float, G_gpa: float) -> np.ndarray:
    return gsa.isotropic_stiffness_66_from_KG(float(K_gpa) * 1e9, float(G_gpa) * 1e9)


def _trigonal_quartz_voigt66_from_table(*, C11: float, C12: float, C13: float, C14: float, C33: float, C44: float, C66: float) -> np.ndarray:
    C = np.zeros((6, 6), dtype=float)
    C11 *= 1e9
    C12 *= 1e9
    C13 *= 1e9
    C14 *= 1e9
    C33 *= 1e9
    C44 *= 1e9
    C66 *= 1e9
    C[0, 0] = C11
    C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C13
    C[3, 3] = C44
    C[4, 4] = C44
    C[5, 5] = C66
    C[0, 3] = C[3, 0] = C14
    C[1, 3] = C[3, 1] = -C14
    C[4, 5] = C[5, 4] = C14
    return C


def _vti_from_any_stiffness(C_voigt66: np.ndarray) -> np.ndarray:
    """
    Project a general anisotropic stiffness to a VTI-form matrix by taking
    (C11,C33,C13,C44,C66) from the input and enforcing C12 = C11 - 2*C66,
    and all other off-diagonals = 0.

    This ensures we can feed a VTI comparison body into GREEN_ANAL_VTI.
    """
    C = np.asarray(C_voigt66, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C_voigt66 must be (6,6)")
    C11 = float(C[0, 0])
    C33 = float(C[2, 2])
    C13 = float(C[0, 2])
    C44 = float(C[3, 3])
    C66 = float(C[5, 5])
    C12 = C11 - 2.0 * C66
    out = np.zeros((6, 6), dtype=float)
    out[0, 0] = out[1, 1] = C11
    out[2, 2] = C33
    out[0, 1] = out[1, 0] = C12
    out[0, 2] = out[2, 0] = C13
    out[1, 2] = out[2, 1] = C13
    out[3, 3] = out[4, 4] = C44
    out[5, 5] = C66
    return out


def _rotation_matrix_zxz(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """
    Euler rotation matrix using the z-x'-z'' convention (as in the dissertation screenshot).

    A = B C D in their notation:
      D: rotation about z by phi
      C: rotation about x' by theta
      B: rotation about z'' by psi
    """
    phi = np.deg2rad(float(phi_deg))
    theta = np.deg2rad(float(theta_deg))
    psi = np.deg2rad(float(psi_deg))

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


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    return np.array(
        [
            [np.cos(a), np.sin(a), 0.0],
            [-np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rotation_matrix_y(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    return np.array(
        [
            [np.cos(a), 0.0, -np.sin(a)],
            [0.0, 1.0, 0.0],
            [np.sin(a), 0.0, np.cos(a)],
        ],
        dtype=float,
    )


def _tilt_rotation_matrix(*, tilt_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """
    Build a "tilted TI" style rotation.

    The stiffness computed by GSA here is typically VTI in the local coordinate
    system (symmetry axis aligned with z). Rotating the stiffness into a global
    coordinate system where the symmetry axis is tilted introduces small
    off-diagonal Voigt terms (e.g., C15/C25/C35/C46), matching the dissertation
    style matrices.

    We parameterize the tilt by:
      azimuth_deg: direction of tilt in the x-y plane (rotation about z)
      tilt_deg: tilt angle away from the z axis (rotation about y in the tilted frame)
    """
    Rz = _rotation_matrix_z(float(azimuth_deg))
    Ry = _rotation_matrix_y(float(tilt_deg))
    return Rz @ Ry @ Rz.T


def rotate_stiffness_voigt66(C_voigt66: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate a stiffness tensor given in engineering Voigt 6x6.

    Implementation:
      Voigt -> Mandel -> 4th-rank tensor -> rotate -> Mandel -> Voigt
    """
    C_voigt66 = np.asarray(C_voigt66, dtype=float)
    if C_voigt66.shape != (6, 6):
        raise ValueError("C_voigt66 must have shape (6,6).")
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must have shape (3,3).")

    C_mandel = gsa.voigt66_to_mandel66(C_voigt66)
    C_ijkl = gsa.mandel_to_tensor4(C_mandel)
    Cp = np.einsum("im,jn,kp,lq,mnpq->ijkl", R, R, R, R, C_ijkl, optimize=True)
    Cp_mandel = gsa.tensor4_to_mandel(Cp)
    return gsa.mandel66_to_voigt66(Cp_mandel)


def _hex_voigt66(*, C11: float, C12: float | None, C13: float, C33: float, C44: float, C66: float | None) -> np.ndarray:
    """
    Hexagonal stiffness matrix (engineering Voigt), axis-3 is symmetry axis.

    If C12 is None, it is derived from C66 via: C66 = (C11 - C12)/2.
    If C66 is None, it is derived from C12 via the same relation.
    """
    C11_pa = float(C11) * 1e9
    C13_pa = float(C13) * 1e9
    C33_pa = float(C33) * 1e9
    C44_pa = float(C44) * 1e9

    if C12 is None and C66 is None:
        raise ValueError("Provide at least one of C12 or C66 for hexagonal stiffness.")
    if C12 is None:
        C66_pa = float(C66) * 1e9
        C12_pa = C11_pa - 2.0 * C66_pa
    else:
        C12_pa = float(C12) * 1e9
        if C66 is None:
            C66_pa = 0.5 * (C11_pa - C12_pa)
        else:
            C66_pa = float(C66) * 1e9

    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = C11_pa
    C[1, 1] = C11_pa
    C[2, 2] = C33_pa
    C[0, 1] = C[1, 0] = C12_pa
    C[0, 2] = C[2, 0] = C13_pa
    C[1, 2] = C[2, 1] = C13_pa
    C[3, 3] = C44_pa
    C[4, 4] = C44_pa
    C[5, 5] = C66_pa
    return C


def _monoclinic_voigt66(
    *,
    C11: float,
    C22: float,
    C33: float,
    C44: float,
    C55: float,
    C66: float,
    C12: float,
    C13: float,
    C23: float,
    C15: float = 0.0,
    C25: float = 0.0,
    C35: float = 0.0,
    C46: float = 0.0,
) -> np.ndarray:
    """
    Minimal monoclinic (engineering Voigt) using the constants visible in the table.

    We assume the conventional form with non-zeros:
      C15, C25, C35, C46 (and their symmetric counterparts).
    Other off-diagonals are set to 0.
    """
    def gpa(x: float) -> float:
        return float(x) * 1e9

    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = gpa(C11)
    C[1, 1] = gpa(C22)
    C[2, 2] = gpa(C33)
    C[3, 3] = gpa(C44)
    C[4, 4] = gpa(C55)
    C[5, 5] = gpa(C66)
    C[0, 1] = C[1, 0] = gpa(C12)
    C[0, 2] = C[2, 0] = gpa(C13)
    C[1, 2] = C[2, 1] = gpa(C23)

    C[0, 4] = C[4, 0] = gpa(C15)  # 15
    C[1, 4] = C[4, 1] = gpa(C25)  # 25
    C[2, 4] = C[4, 2] = gpa(C35)  # 35
    C[3, 5] = C[5, 3] = gpa(C46)  # 46
    return C


def _polycrystal_KG_vrh_from_C(C_voigt66: np.ndarray) -> tuple[float, float]:
    """
    Voigt-Reuss-Hill isotropic moduli for a (randomly oriented) polycrystal.

    Uses standard bounds from stiffness C and compliance S=C^{-1}.
    Returns (K_H, G_H) in GPa.
    """
    C = np.asarray(C_voigt66, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C_voigt66 must be (6,6).")
    # Voigt bounds
    C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
    C12, C13, C23 = C[0, 1], C[0, 2], C[1, 2]
    C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
    K_V = (C11 + C22 + C33 + 2.0 * (C12 + C13 + C23)) / 9.0
    G_V = (C11 + C22 + C33 - (C12 + C13 + C23) + 3.0 * (C44 + C55 + C66)) / 15.0

    # Reuss bounds
    S = np.linalg.inv(C)
    S11, S22, S33 = S[0, 0], S[1, 1], S[2, 2]
    S12, S13, S23 = S[0, 1], S[0, 2], S[1, 2]
    S44, S55, S66 = S[3, 3], S[4, 4], S[5, 5]
    K_R = 1.0 / (S11 + S22 + S33 + 2.0 * (S12 + S13 + S23))
    G_R = 15.0 / (4.0 * (S11 + S22 + S33) - 4.0 * (S12 + S13 + S23) + 3.0 * (S44 + S55 + S66))

    K_H = 0.5 * (K_V + K_R)
    G_H = 0.5 * (G_V + G_R)
    return float(K_H / 1e9), float(G_H / 1e9)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Barnett shale core A: GSA case-1 demo (clays as matrix, other minerals as isolated solid inclusions)."
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out-name", type=str, default="barnett_coreA_case1_solid_polycrystal.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--green-fortran", type=Path, default=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"))

    ap.add_argument("--inclusion-alpha", type=float, default=1.0, help="Aspect ratio of solid inclusions (default: sphere).")
    ap.add_argument("--sign", choices=("+", "-"), default="-")
    ap.add_argument("--angles", type=str, default="0,45,90", help="Angles (deg) to report, comma-separated.")
    ap.add_argument("--plot-angle-curves", action="store_true", help="Also plot V(θ) curves 0..90°.")
    ap.add_argument(
        "--dump-ceff",
        action="store_true",
        help="Save the effective stiffness tensor (Voigt 6x6, GPa) next to the plot.",
    )
    ap.add_argument(
        "--rotate-euler",
        type=str,
        default="",
        help="Optional Euler rotation 'phi,theta,psi' in degrees (z-x'-z'' convention) applied to C_eff.",
    )
    ap.add_argument(
        "--tilt-deg",
        type=float,
        default=None,
        help="Optional convenience tilt (degrees) applied to C_eff. Use this instead of --rotate-euler to get small C15/C25/C35/C46 terms (tilted-TI style).",
    )
    ap.add_argument(
        "--tilt-azimuth-deg",
        type=float,
        default=0.0,
        help="Azimuth (degrees) of the tilt direction in the x-y plane (used with --tilt-deg).",
    )
    ap.add_argument(
        "--use-numeric-green",
        action="store_true",
        help="Use numeric elastic Green tensor (general anisotropy) instead of GREEN_ANAL_VTI (VTI kernel).",
    )
    ap.add_argument("--n-theta", type=int, default=60, help="Theta quadrature for numeric Green (if enabled).")
    ap.add_argument("--n-phi", type=int, default=120, help="Phi quadrature for numeric Green (if enabled).")
    ap.add_argument(
        "--clay-odf",
        choices=("aligned", "random_3d", "ti_gaussian"),
        default="aligned",
        help="Orientation distribution for clay minerals when building the clay comparison body.",
    )
    ap.add_argument(
        "--clay-odf-n",
        type=int,
        default=2000,
        help="Number of orientation samples for clay ODF when clay-odf=random_3d (Monte Carlo).",
    )
    ap.add_argument(
        "--clay-odf-mean-tilt-deg",
        type=float,
        default=0.0,
        help="Mean tilt (deg) for clay-odf=ti_gaussian (about the symmetry axis).",
    )
    ap.add_argument(
        "--clay-odf-sigma-tilt-deg",
        type=float,
        default=10.0,
        help="Tilt sigma (deg) for clay-odf=ti_gaussian.",
    )
    ap.add_argument(
        "--clay-project-to-vti",
        action="store_true",
        help="Project each clay stiffness tensor to VTI form before ODF averaging (matches dissertation text 'TI stiffness is used for clay minerals').",
    )

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = None
    if not bool(args.use_numeric_green):
        backend = gsa.build_backend(
            green_fortran=args.green_fortran,
            output_library=out_dir / "libgsa_elastic_fortran.so",
            force_rebuild=False,
        )
    sign = (-1 if args.sign == "-" else +1)

    # ---------------------------------------------------------------------
    # Tables from Jiang dissertation screenshots (Barnett):
    # - Table 3-2: mass % composition, core A
    # - Table 3-3: densities (g/cc)
    # - Table 3-4: elastic constants (GPa) and/or (K, μ) for minerals
    #
    # Notes:
    # - We treat "Sulfates & Halite" as halite for this demo.
    # - Many clay minerals in Table 3-4 have stiffness constants but no (K, μ)
    #   row values; we do not attempt full anisotropic polycrystal averaging here.
    #   Instead we use the provided (K, μ) when available; otherwise you should
    #   switch to a full single-crystal->polycrystal averaging method.
    # ---------------------------------------------------------------------

    comp_mass_pct = {
        "Quartz": 57.0,
        "Orthoclase": 0.0,
        "Albite": 4.0,
        "Pyrite": 3.0,
        "Calcite": 8.0,
        "Dolomite": 2.0,
        "Aragonite": 1.0,
        "Siderite": 1.0,
        "Halite": 5.0,  # proxy for "Sulfates&Halite"
        "Apatite": 1.0,
        # Clay breakdown (sums to Total Clay = 18)
        "Smectite": 2.0,
        "Illite": 9.0,
        "MixedLayer": 4.0,
        "Kaolinite": 1.0,
        "Mica": 2.0,
        "Chlorite": 0.0,
    }

    rho_gcc = {
        "Quartz": 2.65,
        "Orthoclase": 2.56,
        "Albite": 2.62,
        "Pyrite": 5.016,
        "Calcite": 2.712,
        "Dolomite": 2.87,
        "Aragonite": 2.93,
        "Siderite": 3.96,
        "Halite": 2.16,
        "Apatite": 3.146,
        "Smectite": 2.29,
        "Illite": 2.79,
        "MixedLayer": 2.60,
        "Kaolinite": 2.52,
        "Mica": 2.844,
        "Chlorite": 2.68,
    }

    # Prefer provided isotropic (K, μ) from Table 3-4 when present (GPa).
    iso_KG_gpa = {
        "Quartz": (38.0, 44.0),
        "Orthoclase": (55.4, 28.1),
        "Albite": (56.9, 28.6),
        "Pyrite": (142.7, 125.7),
        "Calcite": (73.3, 32.0),
        "Dolomite": (94.9, 45.7),
        "Aragonite": (46.9, 38.5),
        "Siderite": (124.0, 51.0),
        "Halite": (24.9, 14.7),
        # NOTE: Table 3-4 screenshot provides K, μ only up to Apatite; clays usually do not have K, μ there.
        "Apatite": (80.4, 45.6),
    }

    # Quartz single-crystal constants (GPa) from Table 3-4 (used only if you want)
    # Clay single-crystal stiffness tensors from Table 3-4 (GPa).
    #
    # The screenshot table includes (at least) Illite, Kaolinite, Mica, Chlorite.
    # Smectite and MixedLayer are not clearly visible there; for this demo we
    # approximate them by Illite.
    #
    # If you want strict reproduction, we should OCR the dissertation table or
    # load it from a CSV rather than guessing.
    clay_C = {
        # Illite (Hex.)
        "Illite": _hex_voigt66(C11=140.0, C12=13.0, C13=69.0, C33=180.0, C44=36.2, C66=None),
        # Kaolinite (Hex.) — using values visible in the table image
        "Kaolinite": _hex_voigt66(C11=171.5, C12=None, C13=27.1, C33=52.6, C44=14.8, C66=66.3),
        # Mica (Mono.) — using values visible in the table image
        "Mica": _monoclinic_voigt66(
            C11=184.3,
            C22=178.4,
            C33=59.1,
            C44=16.0,
            C55=17.6,
            C66=72.4,
            C12=48.3,
            C13=23.8,
            C23=21.7,
            C15=-2.0,
            C25=3.9,
            C35=1.2,
            C46=0.5,
        ),
        # Chlorite (Hex.)
        "Chlorite": _hex_voigt66(C11=181.8, C12=None, C13=20.3, C33=106.8, C44=11.4, C66=62.5),
    }

    # Quartz single-crystal constants (GPa) from Table 3-4 (kept for future anisotropic inclusion extension).
    _ = _trigonal_quartz_voigt66_from_table(
        C11=86.0, C12=7.4, C13=11.91, C14=-18.04, C33=105.75, C44=58.2, C66=39.3
    )

    # ---------------------------------------------------------------------
    # Convert mass% -> volume fractions using densities (solid-only)
    # ---------------------------------------------------------------------
    names = list(comp_mass_pct.keys())
    m = np.array([float(comp_mass_pct[n]) for n in names], dtype=float)
    r = np.array([float(rho_gcc[n]) for n in names], dtype=float)
    # volume proportional to mass/density
    v = m / r
    vfrac = v / np.sum(v)

    # Define clay matrix as the *combined* clay group (Smectite+Illite+MixedLayer+Kaolinite+Mica+Chlorite).
    clay_names = ["Smectite", "Illite", "MixedLayer", "Kaolinite", "Mica", "Chlorite"]
    clay_mask = np.array([n in clay_names for n in names], dtype=bool)
    other_mask = ~clay_mask

    clay_vfrac_total = float(np.sum(vfrac[clay_mask]))
    if clay_vfrac_total <= 0.0:
        raise ValueError("Core A has zero clay fraction in the provided table.")

    # Build clay matrix stiffness following the dissertation text:
    # - only clay minerals are aligned;
    # - TI stiffness used for clay minerals except smectite and mixed layer;
    #   smectite and mixed layer use isotropic (K, μ).
    clay_vols = vfrac[clay_mask]
    clay_vols = clay_vols / np.sum(clay_vols)

    # Isotropic clay properties (GPa) from the provided page:
    #   smectite: k=7.0, μ=3.9
    #   mixed layer: k=21.4, μ=6.7
    smectite_KG = (7.0, 3.9)
    mixedlayer_KG = (21.4, 6.7)

    C_clay = np.zeros((6, 6), dtype=float)
    clay_used_names: list[str] = []
    clay_odf = gsa_gen.OrientationDistributionFunction(
        kind=str(args.clay_odf),
        mean_tilt_deg=float(args.clay_odf_mean_tilt_deg),
        sigma_tilt_deg=float(args.clay_odf_sigma_tilt_deg),
    )
    for n, f in zip(np.array(names)[clay_mask], clay_vols, strict=True):
        nm = str(n)
        if float(f) <= 0.0:
            continue
        if nm == "Smectite":
            Ci = _isotropic_voigt66_from_KG(*smectite_KG)
        elif nm == "MixedLayer":
            Ci = _isotropic_voigt66_from_KG(*mixedlayer_KG)
        else:
            if nm not in clay_C:
                raise ValueError(f"Missing clay stiffness tensor for {nm!r}.")
            Ci = np.asarray(clay_C[nm], dtype=float)
            if bool(args.clay_project_to_vti):
                # Illite/Kaolinite/Chlorite are already TI (hex). Mica is monoclinic in Table 3-4,
                # but the text says TI stiffness is used for clay minerals, so project to VTI.
                Ci = _vti_from_any_stiffness(Ci)
            # ODF Euler-angle average of each clay mineral stiffness:
            #   <C> = ∫ F(phi,theta,psi) C^T sin(theta) dphi dtheta dpsi
            Ci = gsa_gen.orientation_average_stiffness_voigt66(
                Ci,
                odf=clay_odf,
                n=int(args.clay_odf_n),
            )
        C_clay += float(f) * Ci
        clay_used_names.append(nm)

    # Build isotropic inclusions (non-clays) from Table 3-4 K,μ
    inclusion_names = list(np.array(names)[other_mask])
    inclusion_vfrac = vfrac[other_mask]
    C_incs: list[np.ndarray] = []
    rho_incs_gcc: list[float] = []
    inc_used_names: list[str] = []
    inc_used_vf: list[float] = []
    for n, vf in zip(inclusion_names, inclusion_vfrac, strict=True):
        if float(vf) <= 0.0:
            continue
        if n not in iso_KG_gpa:
            # skip minerals without K,μ (in core A: should not happen for non-clays with this table)
            continue
        K, G = iso_KG_gpa[n]
        C_incs.append(_isotropic_voigt66_from_KG(K, G))
        rho_incs_gcc.append(float(rho_gcc[n]))
        inc_used_names.append(str(n))
        inc_used_vf.append(float(vf))

    # Fractions vector for GSA case-1: [clay_matrix, inclusions...], volume fractions renormalized to 1.
    f_clay = float(clay_vfrac_total)
    inc_vf = np.array(inc_used_vf, dtype=float)
    fr = np.concatenate([[f_clay], inc_vf], axis=0)
    fr = fr / np.sum(fr)

    # Effective density (g/cc -> kg/m^3) by volume mixture (solid-only).
    rho_eff = float(np.sum(vfrac * r) * 1e3)

    if bool(args.use_numeric_green):
        shape = gsa_gen.EllipsoidShape(a1=1.0, a2=1.0, a3=float(args.inclusion_alpha))
        C_eff = gsa_gen.gsa_case1_effective_stiffness(
            matrix_C_voigt66=C_clay,
            inclusion_C_voigt66_list=C_incs,
            fractions=fr,
            inclusion_shape=shape,
            orientation_axis=(0.0, 0.0, 1.0),
            sign=sign,
            n_theta=int(args.n_theta),
            n_phi=int(args.n_phi),
        )
    else:
        assert backend is not None
        C_eff = gsa_case1_matrix_comparison_effective_stiffness(
            matrix_C_voigt66=C_clay,
            inclusion_C_voigt66_list=C_incs,
            fractions=fr,
            inclusion_aspect_ratio=float(args.inclusion_alpha),
            backend=backend,
            sign=sign,
        )

    C_eff_rot = None
    rotate_euler = str(args.rotate_euler).strip()
    if rotate_euler and args.tilt_deg is not None:
        raise ValueError("Use either --rotate-euler or --tilt-deg, not both.")
    if rotate_euler:
        phi_deg, theta_deg, psi_deg = [float(x.strip()) for x in rotate_euler.split(",")]
        R = _rotation_matrix_zxz(phi_deg, theta_deg, psi_deg)
        C_eff_rot = rotate_stiffness_voigt66(C_eff, R)
    elif args.tilt_deg is not None:
        R = _tilt_rotation_matrix(tilt_deg=float(args.tilt_deg), azimuth_deg=float(args.tilt_azimuth_deg))
        C_eff_rot = rotate_stiffness_voigt66(C_eff, R)

    # Report velocities at requested angles using Christoffel (works for isotropic too).
    angles = np.array([float(x.strip()) for x in str(args.angles).split(",") if x.strip()], dtype=float)
    C_for_vel = C_eff if C_eff_rot is None else C_eff_rot
    vP, vS1, vS2 = vti_phase_velocities_xz_plane(C_for_vel, rho_eff, angles)

    print("Barnett core A (solid phase only) - GSA case-1 demo")
    print(
        "Clay matrix: volume average of clay minerals (each optionally ODF-averaged by Euler angles); "
        "smectite/mixed-layer isotropic from text."
    )
    print(f"Clay minerals used: {', '.join(clay_used_names)}")
    print(
        f"Clay ODF: kind={args.clay_odf}, mean_tilt_deg={float(args.clay_odf_mean_tilt_deg):g}, "
        f"sigma_tilt_deg={float(args.clay_odf_sigma_tilt_deg):g}, n={int(args.clay_odf_n)}"
        + (", project_to_vti=True" if bool(args.clay_project_to_vti) else ", project_to_vti=False")
    )
    print(f"Solid density rho_eff = {rho_eff:.1f} kg/m^3")
    print("Inclusions used (from Table 3-4 K,μ): " + ", ".join(inc_used_names))
    print("\nC_eff (engineering Voigt, 6x6) in GPa:")
    C_eff_gpa = C_eff / 1e9
    for row in C_eff_gpa:
        print("  " + "  ".join(f"{v:9.3f}" for v in row))
    if C_eff_rot is not None:
        print("\nC_eff_rotated (engineering Voigt, 6x6) in GPa:")
        for row in (C_eff_rot / 1e9):
            print("  " + "  ".join(f"{v:9.3f}" for v in row))
    for th, vp, vs1, vs2 in zip(angles, vP, vS1, vS2, strict=True):
        print(f"theta={th:5.1f} deg:  Vp={vp:6.3f} km/s,  Vs1={vs1:6.3f} km/s,  Vs2={vs2:6.3f} km/s")

    # Optional angle curves plot (0..90)
    import matplotlib.pyplot as plt

    dense = np.linspace(0.0, 90.0, 181)
    vP_d, vS1_d, vS2_d = vti_phase_velocities_xz_plane(C_for_vel, rho_eff, dense)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8), constrained_layout=True)
        ax.plot(dense, vP_d, color="red", lw=2.4, label="qP")
        ax.plot(dense, vS1_d, color="blue", lw=2.4, label="qS1")
        ax.plot(dense, vS2_d, color="black", lw=2.4, label="qS2 (SH)")
        ax.set_xlabel("Angle of wave incidence to symmetry axis (Degree)")
        ax.set_ylabel("Wave velocities (km/s)")
        ax.set_xlim(0.0, 90.0)
        ax.set_ylim(2.0, 6.0)
        ax.grid(True, alpha=0.25)
        ax.set_title("Barnett core A solid: case-1 demo (clay matrix + isolated inclusions)")
        ax.legend(frameon=False, loc="best")

        out_png = out_dir / str(args.out_name)
        if out_png.suffix.lower() != ".png":
            out_png = out_png.with_suffix(".png")
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")

    if bool(args.dump_ceff):
        out_txt = out_png.with_name(out_png.stem + "_Ceff_voigt_gpa.txt")
        lines: list[str] = []
        lines.append("Barnett core A (solid, no pores) - effective stiffness C_eff (Voigt 6x6) [GPa]")
        lines.append(f"rho_eff [g/cc] = {rho_eff/1e3:.3f}")
        if rotate_euler:
            lines.append(f"rotation_euler_zxz_deg = {rotate_euler}")
        if args.tilt_deg is not None:
            lines.append(f"tilt_deg = {float(args.tilt_deg):.6f}")
            lines.append(f"tilt_azimuth_deg = {float(args.tilt_azimuth_deg):.6f}")
        lines.append("")
        for row in (C_for_vel / 1e9):
            lines.append("  ".join(f"{v:0.6f}" for v in row))
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
