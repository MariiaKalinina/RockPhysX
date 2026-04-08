from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa
from vti_phase_velocities_vs_angle_check import vti_phase_velocities_xz_plane


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _vti66_from_constants(
    *,
    C11_gpa: float,
    C12_gpa: float,
    C13_gpa: float,
    C33_gpa: float,
    C44_gpa: float,
    C66_gpa: float,
    C14_gpa: float = 0.0,
) -> np.ndarray:
    """
    Build a 6x6 stiffness matrix (engineering Voigt) for a VTI or (optional) trigonal medium.

    - For VTI set C14=0.
    - For trigonal (quartz-like) set C14!=0, which implies C24=-C14 and C56=C14 in
      the conventional symmetry-axis coordinate system.
    """
    C11 = float(C11_gpa) * 1e9
    C12 = float(C12_gpa) * 1e9
    C13 = float(C13_gpa) * 1e9
    C33 = float(C33_gpa) * 1e9
    C44 = float(C44_gpa) * 1e9
    C66 = float(C66_gpa) * 1e9
    C14 = float(C14_gpa) * 1e9

    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = C11
    C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C13
    C[3, 3] = C44
    C[4, 4] = C44
    C[5, 5] = C66

    if C14 != 0.0:
        C[0, 3] = C[3, 0] = C14
        C[1, 3] = C[3, 1] = -C14
        C[4, 5] = C[5, 4] = C14

    return C


def _parse_inclusion(s: str) -> dict[str, float | str]:
    """
    Parse:
      name,f,rho_gcc,C11,C12,C13,C33,C44,C66[,C14]
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) not in (10, 11):
        raise ValueError(
            "Invalid --inclusion format. Expected: "
            "name,f,rho_gcc,C11,C12,C13,C33,C44,C66[,C14]"
        )
    name = parts[0]
    f = float(parts[1])
    rho = float(parts[2])
    C11 = float(parts[3])
    C12 = float(parts[4])
    C13 = float(parts[5])
    C33 = float(parts[6])
    C44 = float(parts[7])
    if len(parts) == 10:
        C66 = float(parts[8])
        C14 = 0.0
    else:
        C66 = float(parts[8])
        C14 = float(parts[9])
    return {
        "name": name,
        "f": f,
        "rho_gcc": rho,
        "C11_gpa": C11,
        "C12_gpa": C12,
        "C13_gpa": C13,
        "C33_gpa": C33,
        "C44_gpa": C44,
        "C66_gpa": C66,
        "C14_gpa": C14,
    }


def gsa_case1_matrix_comparison_effective_stiffness(
    *,
    matrix_C_voigt66: np.ndarray,
    inclusion_C_voigt66_list: list[np.ndarray],
    fractions: np.ndarray,
    inclusion_aspect_ratio: float,
    backend: gsa._Backend,
    sign: int = -1,
) -> np.ndarray:
    """
    Case-1 style: solid inclusions in solid matrix, comparison body = matrix.

    Effective stiffness (Mandel algebra):
      C* = < C_i : A_i > : < A_i >^{-1}
    where:
      A_i = (I + sign * G0 : (C_i - C0))^{-1}, C0 = C_matrix

    For the matrix phase, C_i=C0 -> A_m = I.
    """
    C0_voigt = np.asarray(matrix_C_voigt66, dtype=float)
    if C0_voigt.shape != (6, 6):
        raise ValueError("matrix_C_voigt66 must have shape (6,6).")
    if sign not in (-1, +1):
        raise ValueError("sign must be +1 or -1.")

    fr = np.asarray(fractions, dtype=float)
    if fr.ndim != 1:
        raise ValueError("fractions must be 1D.")
    if np.any(fr < 0.0):
        raise ValueError("fractions must be non-negative.")
    s = float(np.sum(fr))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Sum of fractions must be positive.")
    fr = fr / s

    # Comparison body = matrix.
    C0_mandel = gsa.voigt66_to_mandel66(C0_voigt)
    C0_3333 = gsa.mandel_to_tensor4(C0_mandel)

    # Green tensor for the inclusion shape in the comparison body.
    G0_3333 = gsa.green_tensor_vti_fortran(C0_voigt, float(inclusion_aspect_ratio), backend)
    G0_66 = gsa.tensor4_to_mandel(G0_3333)
    I6 = np.eye(6, dtype=float)

    # Build list of phases: matrix first, then inclusions.
    C_list_mandel: list[np.ndarray] = [C0_mandel]
    A_list: list[np.ndarray] = [I6]  # A_matrix = I

    for Ci_voigt in inclusion_C_voigt66_list:
        Ci_voigt = np.asarray(Ci_voigt, dtype=float)
        if Ci_voigt.shape != (6, 6):
            raise ValueError("Each inclusion stiffness must have shape (6,6).")
        Ci_mandel = gsa.voigt66_to_mandel66(Ci_voigt)
        dC = Ci_mandel - C0_mandel
        A = np.linalg.inv(I6 + float(sign) * (G0_66 @ dC))
        C_list_mandel.append(Ci_mandel)
        A_list.append(A)

    if fr.size != len(C_list_mandel):
        raise ValueError("fractions length must equal 1 + number of inclusions.")

    num = np.zeros((6, 6), dtype=float)
    den = np.zeros((6, 6), dtype=float)
    for fi, Ci, Ai in zip(fr, C_list_mandel, A_list, strict=True):
        num += float(fi) * (Ci @ Ai)
        den += float(fi) * Ai

    C_eff_mandel = num @ np.linalg.inv(den)
    return gsa.mandel66_to_voigt66(C_eff_mandel)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "GSA case-1 demo: polycrystalline solid matrix as clay (matrix) + isolated solid inclusions.\n"
            "Comparison body is the matrix stiffness (friability of inclusions effectively 0)."
        )
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out-name", type=str, default="gsa_polycrystal_case1_angles.png")
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--inclusion-alpha", type=float, default=1.0, help="Aspect ratio of inclusions (default: sphere)")
    ap.add_argument("--sign", choices=("+", "-"), default="-", help="Operator sign in A = (I ± G0·ΔC)^-1.")
    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))

    # Matrix (clay) stiffness and density
    ap.add_argument("--matrix-name", type=str, default="clay")
    ap.add_argument("--matrix-f", type=float, default=0.6, help="Matrix volume fraction (will be renormalized).")
    ap.add_argument("--matrix-rho-gcc", type=float, default=2.60)
    ap.add_argument("--matrix-C11-gpa", type=float, required=True)
    ap.add_argument("--matrix-C12-gpa", type=float, required=True)
    ap.add_argument("--matrix-C13-gpa", type=float, required=True)
    ap.add_argument("--matrix-C33-gpa", type=float, required=True)
    ap.add_argument("--matrix-C44-gpa", type=float, required=True)
    ap.add_argument("--matrix-C66-gpa", type=float, required=True)
    ap.add_argument("--matrix-C14-gpa", type=float, default=0.0)

    ap.add_argument(
        "--inclusion",
        action="append",
        default=[],
        help="Repeatable. Format: name,f,rho_gcc,C11,C12,C13,C33,C44,C66[,C14]",
    )
    ap.add_argument("--angles", type=str, default="0,45,90", help="Comma-separated angles in degrees.")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=False,
    )

    C_matrix = _vti66_from_constants(
        C11_gpa=args.matrix_C11_gpa,
        C12_gpa=args.matrix_C12_gpa,
        C13_gpa=args.matrix_C13_gpa,
        C33_gpa=args.matrix_C33_gpa,
        C44_gpa=args.matrix_C44_gpa,
        C66_gpa=args.matrix_C66_gpa,
        C14_gpa=args.matrix_C14_gpa,
    )

    inc_specs = [_parse_inclusion(s) for s in list(args.inclusion)]
    C_incs: list[np.ndarray] = []
    fracs = [float(args.matrix_f)]
    rhos = [float(args.matrix_rho_gcc)]

    for spec in inc_specs:
        C_incs.append(
            _vti66_from_constants(
                C11_gpa=float(spec["C11_gpa"]),
                C12_gpa=float(spec["C12_gpa"]),
                C13_gpa=float(spec["C13_gpa"]),
                C33_gpa=float(spec["C33_gpa"]),
                C44_gpa=float(spec["C44_gpa"]),
                C66_gpa=float(spec["C66_gpa"]),
                C14_gpa=float(spec["C14_gpa"]),
            )
        )
        fracs.append(float(spec["f"]))
        rhos.append(float(spec["rho_gcc"]))

    fractions = np.asarray(fracs, dtype=float)
    rho_eff = float(np.sum((fractions / np.sum(fractions)) * (np.asarray(rhos, dtype=float) * 1e3)))

    sign = (-1 if args.sign == "-" else +1)
    C_eff = gsa_case1_matrix_comparison_effective_stiffness(
        matrix_C_voigt66=C_matrix,
        inclusion_C_voigt66_list=C_incs,
        fractions=fractions,
        inclusion_aspect_ratio=float(args.inclusion_alpha),
        backend=backend,
        sign=sign,
    )

    angles = np.array([float(x.strip()) for x in str(args.angles).split(",") if x.strip()], dtype=float)
    vP, vS1, vS2 = vti_phase_velocities_xz_plane(C_eff, rho_eff, angles)

    # Print numeric check table.
    print("Effective solid matrix (GSA case-1, comparison body = clay)")
    print(f"rho_eff = {rho_eff:.1f} kg/m^3")
    for th, vp, vs1, vs2 in zip(angles, vP, vS1, vS2, strict=True):
        print(f"theta={th:5.1f} deg:  Vp={vp:6.3f} km/s,  Vs1={vs1:6.3f} km/s,  Vs2={vs2:6.3f} km/s")

    # Plot (single panel) to compare with references like Fig. 3-6/3-7 in dissertation.
    import matplotlib.pyplot as plt

    dense_angles = np.linspace(0.0, 90.0, 181)
    vP_d, vS1_d, vS2_d = vti_phase_velocities_xz_plane(C_eff, rho_eff, dense_angles)

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
        ax.plot(dense_angles, vP_d, color="red", lw=2.4, label="qP")
        ax.plot(dense_angles, vS1_d, color="blue", lw=2.4, label="qS1")
        ax.plot(dense_angles, vS2_d, color="black", lw=2.4, label="qS2 (SH)")
        ax.set_xlabel("Angle of wave incidence to symmetry axis (Degree)")
        ax.set_ylabel("Wave velocities (km/s)")
        ax.set_xlim(0.0, 90.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")

        out_png = out_dir / str(args.out_name)
        if out_png.suffix.lower() != ".png":
            out_png = out_png.with_suffix(".png")
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
