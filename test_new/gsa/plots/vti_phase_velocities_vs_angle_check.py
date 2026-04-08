from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _matrix_moduli_from_vp_vs_rho(vp_kms: float, vs_kms: float, rho_gcc: float) -> tuple[float, float]:
    vp = float(vp_kms) * 1e3
    vs = float(vs_kms) * 1e3
    rho = float(rho_gcc) * 1e3
    K = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
    G = rho * (vs * vs)
    return float(K), float(G)


def _christoffel_matrix(C_ijkl: np.ndarray, n: np.ndarray) -> np.ndarray:
    # Christoffel matrix:
    #   Γ_{ik} = C_{ijkl} n_j n_l
    # (contract over the 2nd and 4th indices).
    return np.einsum("ijkl,j,l->ik", C_ijkl, n, n, optimize=True)


def vti_phase_velocities_xz_plane(
    C_voigt66_pa: np.ndarray,
    rho_kg_m3: float,
    angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase velocities vs incidence angle (0..90°) to symmetry axis for VTI medium.

    We assume propagation in the XZ plane, with symmetry axis along Z (axis 3).
    Angle θ is measured from the symmetry axis:
        θ=0°  -> along Z
        θ=90° -> along X

    Returns
    -------
    vP, vSV, vSH : arrays in km/s
    """
    C_voigt66_pa = np.asarray(C_voigt66_pa, dtype=float)
    if C_voigt66_pa.shape != (6, 6):
        raise ValueError("C_voigt66_pa must have shape (6,6).")
    rho = float(rho_kg_m3)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho_kg_m3 must be positive and finite.")

    # Convert engineering-Voigt -> Mandel -> full tensor C_ijkl.
    C_mandel = gsa.voigt66_to_mandel66(C_voigt66_pa)
    C_ijkl = gsa.mandel_to_tensor4(C_mandel)

    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    vP = np.full_like(theta, np.nan, dtype=float)
    vS1 = np.full_like(theta, np.nan, dtype=float)
    vS2 = np.full_like(theta, np.nan, dtype=float)

    prev_vecs: np.ndarray | None = None

    def _best_perm(scores: np.ndarray) -> tuple[int, int, int]:
        # scores[a,b] = similarity between new mode a and previous mode b
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        best = perms[0]
        best_val = -1.0
        for p in perms:
            val = float(scores[0, p[0]] + scores[1, p[1]] + scores[2, p[2]])
            if val > best_val:
                best_val = val
                best = p
        return best

    for i, th in enumerate(theta):
        n = np.array([np.sin(th), 0.0, np.cos(th)], dtype=float)
        Gamma = _christoffel_matrix(C_ijkl, n) / rho
        Gamma = 0.5 * (Gamma + Gamma.T)
        w, V = np.linalg.eigh(Gamma)  # ascending eigenvalues
        # For stable media w should be >=0; guard tiny negatives.
        w = np.clip(w, 0.0, None)
        speeds = np.sqrt(w) / 1e3  # km/s

        if prev_vecs is None:
            # Initialization: pick SH as most y-polarized, P as fastest of the rest.
            y_comp = np.abs(V[1, :])
            sh_idx = int(np.argmax(y_comp))
            other = [j for j in range(3) if j != sh_idx]
            if speeds[other[0]] >= speeds[other[1]]:
                p_idx, sv_idx = other[0], other[1]
            else:
                p_idx, sv_idx = other[1], other[0]
            idxs = [p_idx, sv_idx, sh_idx]  # (P, S1, S2)
        else:
            # Track modes by polarization continuity (maximize |dot| with previous eigenvectors).
            sims = np.abs(V.T @ prev_vecs)  # new(3)xprev(3)
            perm = _best_perm(sims)
            # perm maps new-index -> prev-index, so invert to get order by prev slots
            inv = np.argsort(np.array(perm))
            idxs = [int(inv[0]), int(inv[1]), int(inv[2])]  # (P, S1, S2) consistent with previous

        vP[i] = float(speeds[idxs[0]])
        vS1[i] = float(speeds[idxs[1]])
        vS2[i] = float(speeds[idxs[2]])

        # Enforce the physical labeling: qP is always the fastest branch.
        # (For isotropic or near-degenerate cases, eigenvector tracking can
        # arbitrarily swap labels; this keeps qP consistent.)
        fastest = int(np.argmax(speeds))
        if fastest != idxs[0]:
            if fastest == idxs[1]:
                idxs = [idxs[1], idxs[0], idxs[2]]
            elif fastest == idxs[2]:
                idxs = [idxs[2], idxs[1], idxs[0]]
            vP[i] = float(speeds[idxs[0]])
            vS1[i] = float(speeds[idxs[1]])
            vS2[i] = float(speeds[idxs[2]])

        prev_vecs = V[:, idxs]  # store tracked polarization vectors

    return vP, vS1, vS2


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check plot: phase velocities vs incidence angle to symmetry axis (Christoffel)."
    )
    ap.add_argument("--n-angles", type=int, default=181, help="Number of angle points between 0 and 90 degrees")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Output PNG filename (optional). If omitted, a name is generated from the inputs.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title. If omitted, a compact title is auto-generated.",
    )

    ap.add_argument(
        "--source",
        choices=("gsa_aligned", "manual_vti", "manual_trigonal_quartz"),
        default="gsa_aligned",
        help="How to obtain the VTI stiffness: compute via aligned GSA or use manual Cij inputs.",
    )

    # Manual VTI inputs (GPa, g/cc)
    ap.add_argument("--rho-gcc", type=float, default=2.71, help="Density (g/cc) for manual_vti")
    ap.add_argument("--C11-gpa", type=float, default=90.0, help="VTI stiffness C11 (GPa) for manual_vti")
    ap.add_argument("--C33-gpa", type=float, default=90.0, help="VTI stiffness C33 (GPa) for manual_vti")
    ap.add_argument("--C13-gpa", type=float, default=30.0, help="VTI stiffness C13 (GPa) for manual_vti")
    ap.add_argument("--C44-gpa", type=float, default=30.0, help="VTI stiffness C44 (GPa) for manual_vti")
    ap.add_argument("--C66-gpa", type=float, default=30.0, help="VTI stiffness C66 (GPa) for manual_vti")

    # Trigonal quartz (class 32/3m style constants, GPa, g/cc)
    ap.add_argument("--quartz-rho-gcc", type=float, default=2.65, help="Quartz density (g/cc)")
    ap.add_argument("--quartz-C11-gpa", type=float, default=86.0, help="Quartz C11 (GPa)")
    ap.add_argument("--quartz-C12-gpa", type=float, default=7.4, help="Quartz C12 (GPa)")
    ap.add_argument("--quartz-C13-gpa", type=float, default=11.91, help="Quartz C13 (GPa)")
    ap.add_argument("--quartz-C14-gpa", type=float, default=-18.04, help="Quartz C14 (GPa)")
    ap.add_argument("--quartz-C33-gpa", type=float, default=105.75, help="Quartz C33 (GPa)")
    ap.add_argument("--quartz-C44-gpa", type=float, default=58.2, help="Quartz C44 (GPa)")
    ap.add_argument("--quartz-C66-gpa", type=float, default=39.3, help="Quartz C66 (GPa)")

    # GSA aligned inputs (defaults: limestone matrix + water pores)
    ap.add_argument("--phi", type=float, default=0.10, help="Porosity for gsa_aligned")
    ap.add_argument("--alpha", type=float, default=1e-2, help="Aspect ratio α=a3/a1 for gsa_aligned")
    ap.add_argument("--matrix-vp-kms", type=float, default=5.8, help="Matrix VP (km/s)")
    ap.add_argument("--matrix-vs-kms", type=float, default=3.2, help="Matrix VS (km/s)")
    ap.add_argument("--matrix-rho-gcc", type=float, default=2.71, help="Matrix density (g/cc)")
    ap.add_argument("--fluid-K-gpa", type=float, default=2.2, help="Fluid bulk modulus (GPa) for pores")
    ap.add_argument("--fluid-rho-gcc", type=float, default=1.0, help="Fluid density (g/cc) for pores")
    ap.add_argument("--comparison-body", choices=("matrix", "bayuk_linear_mix"), default="matrix")
    ap.add_argument("--k-connectivity", type=float, default=3.0, help="k for comparison_body='bayuk_linear_mix'.")
    ap.add_argument("--sign", choices=("+", "-"), default="-", help="Operator sign in A = (I ± G0·ΔC)^-1.")
    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--force-rebuild-backend", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    angles = np.linspace(0.0, 90.0, int(args.n_angles))

    if args.source == "manual_vti":
        rho = float(args.rho_gcc) * 1e3
        C = np.zeros((6, 6), dtype=float)
        C[0, 0] = float(args.C11_gpa) * 1e9
        C[1, 1] = float(args.C11_gpa) * 1e9
        C[2, 2] = float(args.C33_gpa) * 1e9
        C[0, 1] = C[1, 0] = float(args.C11_gpa - 2.0 * args.C66_gpa) * 1e9
        C[0, 2] = C[2, 0] = float(args.C13_gpa) * 1e9
        C[1, 2] = C[2, 1] = float(args.C13_gpa) * 1e9
        C[3, 3] = float(args.C44_gpa) * 1e9
        C[4, 4] = float(args.C44_gpa) * 1e9
        C[5, 5] = float(args.C66_gpa) * 1e9
    elif args.source == "manual_trigonal_quartz":
        # Build full 6x6 Voigt stiffness for trigonal quartz with c-axis along axis-3.
        # Voigt order: (11,22,33,23,13,12) == (xx,yy,zz,yz,xz,xy)
        rho = float(args.quartz_rho_gcc) * 1e3
        C11 = float(args.quartz_C11_gpa) * 1e9
        C12 = float(args.quartz_C12_gpa) * 1e9
        C13 = float(args.quartz_C13_gpa) * 1e9
        C14 = float(args.quartz_C14_gpa) * 1e9
        C33 = float(args.quartz_C33_gpa) * 1e9
        C44 = float(args.quartz_C44_gpa) * 1e9
        C66 = float(args.quartz_C66_gpa) * 1e9

        C = np.zeros((6, 6), dtype=float)
        # Main symmetric block
        C[0, 0] = C11
        C[1, 1] = C11
        C[2, 2] = C33
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C13
        C[1, 2] = C[2, 1] = C13
        # Shear terms
        C[3, 3] = C44
        C[4, 4] = C44
        C[5, 5] = C66

        # Trigonal coupling terms (matching the screenshot table):
        # C14 at (1,4), C24 at (2,4) with opposite sign, and C56 equals C14.
        # Note: many references use C24 = -C14 and C56 = C14.
        C[0, 3] = C[3, 0] = C14
        C[1, 3] = C[3, 1] = -C14
        C[4, 5] = C[5, 4] = C14
    else:
        phi = float(args.phi)
        alpha = float(args.alpha)
        if not (0.0 <= phi < 1.0):
            raise ValueError("--phi must lie in [0,1).")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("--alpha must be positive and finite.")

        Km_pa, Gm_pa = _matrix_moduli_from_vp_vs_rho(args.matrix_vp_kms, args.matrix_vs_kms, args.matrix_rho_gcc)
        matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)
        pores = gsa.ElasticPhase(K=float(args.fluid_K_gpa) * 1e9, G=1e-9)

        backend = gsa.build_backend(
            green_fortran=args.green_fortran,
            output_library=out_dir / "libgsa_elastic_fortran.so",
            force_rebuild=bool(args.force_rebuild_backend),
        )
        sign = (-1 if args.sign == "-" else +1)

        C, _ = gsa.gsa_effective_stiffness_aligned_vti_two_phase(
            phi=phi,
            matrix=matrix,
            inclusion=pores,
            pore_aspect_ratio=alpha,
            backend=backend,
            comparison_body=str(args.comparison_body),
            k_connectivity=(float(args.k_connectivity) if str(args.comparison_body) == "bayuk_linear_mix" else None),
            sign=sign,
        )
        rho_m = float(args.matrix_rho_gcc) * 1e3
        rho_f = float(args.fluid_rho_gcc) * 1e3
        rho = (1.0 - phi) * rho_m + phi * rho_f

    vP, vSV, vSH = vti_phase_velocities_xz_plane(C, rho, angles)

    import matplotlib.pyplot as plt

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
        ax.plot(angles, vP, color="red", lw=2.4, label="qP")
        ax.plot(angles, vSV, color="blue", lw=2.4, label="qS1")
        ax.plot(angles, vSH, color="black", lw=2.4, label="qS2 (SH)")

        ax.set_xlabel("Angle of wave incidence to symmetry axis (Degree)")
        ax.set_ylabel("Wave velocities (km/s)")
        ax.set_xlim(0.0, 90.0)
        ax.grid(True, alpha=0.25)
        if str(args.title).strip():
            ax.set_title(str(args.title).strip())
        else:
            title = "VTI phase velocities (Christoffel), XZ-plane"
            if args.source == "gsa_aligned":
                title += rf", $\phi={float(args.phi):.2f}$, $\alpha={float(args.alpha):g}$"
            elif args.source == "manual_trigonal_quartz":
                title = "Quartz phase velocities (Christoffel), XZ-plane"
            ax.set_title(title)
        ax.legend(frameon=False, loc="best")

        out_name = str(args.out_name).strip()
        if not out_name:
            if args.source == "manual_vti":
                out_name = "vti_phase_velocities_vs_angle_manual.png"
            elif args.source == "manual_trigonal_quartz":
                out_name = "quartz_phase_velocities_vs_angle.png"
            else:
                out_name = f"vti_phase_velocities_vs_angle_phi_{float(args.phi):.2f}_alpha_{float(args.alpha):g}.png"
        if not out_name.lower().endswith(".png"):
            out_name += ".png"
        out_png = out_dir / out_name
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
