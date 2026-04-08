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


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic GSA (isotropic random): stiffness component maps vs (phi, log10(alpha)).")
    ap.add_argument("--phi-max", type=float, default=0.35, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")
    ap.add_argument("--log10a-min", type=float, default=-4.0, help="Minimum log10(alpha)")
    ap.add_argument("--log10a-max", type=float, default=0.0, help="Maximum log10(alpha)")
    ap.add_argument("--n-alpha", type=int, default=81, help="Number of alpha points")

    ap.add_argument("--matrix-K-gpa", type=float, default=76.0, help="Matrix bulk modulus (GPa)")
    ap.add_argument("--matrix-G-gpa", type=float, default=32.0, help="Matrix shear modulus (GPa)")
    ap.add_argument("--matrix-rho", type=float, default=2720.0, help="Matrix density (kg/m^3)")

    ap.add_argument("--pore-K-gpa", type=float, default=0.0, help="Pore/fluid bulk modulus (GPa)")
    ap.add_argument("--pore-G-gpa", type=float, default=0.0, help="Pore/fluid shear modulus (GPa)")
    ap.add_argument("--pore-rho", type=float, default=0.0, help="Pore/fluid density (kg/m^3)")

    ap.add_argument(
        "--velocity-density",
        choices=("effective", "matrix"),
        default="matrix",
        help="How to compute velocities: use effective bulk density (physical) or fixed matrix density (diagnostic).",
    )

    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    args = ap.parse_args()

    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))
    log10a = np.linspace(float(args.log10a_min), float(args.log10a_max), int(args.n_alpha))
    alpha = 10.0 ** log10a

    Km_pa = float(args.matrix_K_gpa) * 1e9
    Gm_pa = float(args.matrix_G_gpa) * 1e9
    Kp_pa = max(float(args.pore_K_gpa) * 1e9, 1e-9)
    Gp_pa = max(float(args.pore_G_gpa) * 1e9, 1e-9)

    matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)
    pores = gsa.ElasticPhase(K=Kp_pa, G=Gp_pa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )

    # Grids (alpha index, phi index)
    C11 = np.full((len(alpha), len(phi)), np.nan, dtype=float)
    C12 = np.full((len(alpha), len(phi)), np.nan, dtype=float)
    C44 = np.full((len(alpha), len(phi)), np.nan, dtype=float)
    Keff = np.full((len(alpha), len(phi)), np.nan, dtype=float)
    Geff = np.full((len(alpha), len(phi)), np.nan, dtype=float)
    Vp = np.full((len(alpha), len(phi)), np.nan, dtype=float)
    Vs = np.full((len(alpha), len(phi)), np.nan, dtype=float)

    for ia, a in enumerate(alpha):
        for ip, ph in enumerate(phi):
            res = gsa.gsa_elastic_random_spheroidal_pores(
                phi=float(ph),
                a_ratio=float(a),
                matrix=matrix,
                inclusion=pores,
                rho_matrix_kg_m3=float(args.matrix_rho),
                rho_inclusion_kg_m3=float(args.pore_rho),
                backend=backend,
                green_fortran=args.green_fortran,
            )
            C_eff = np.asarray(res.C_eff_66, dtype=float)
            C11[ia, ip] = float(C_eff[0, 0])
            C12[ia, ip] = float(C_eff[0, 1])
            C44[ia, ip] = float(C_eff[3, 3])
            Keff[ia, ip] = float(res.K_eff)
            Geff[ia, ip] = float(res.G_eff)

            if args.velocity_density == "matrix":
                vp_i, vs_i = gsa.velocities_from_KG_rho(res.K_eff, res.G_eff, float(args.matrix_rho))
                Vp[ia, ip] = float(vp_i)
                Vs[ia, ip] = float(vs_i)
            else:
                Vp[ia, ip] = float(res.vp_m_s)
                Vs[ia, ip] = float(res.vs_m_s)

    # Convert for plotting
    C11_gpa = C11 / 1e9
    C12_gpa = C12 / 1e9
    C44_gpa = C44 / 1e9
    Keff_gpa = Keff / 1e9
    Geff_gpa = Geff / 1e9
    Vp_kms = Vp / 1e3
    Vs_kms = Vs / 1e3

    print("Elastic GSA (isotropic random): stiffness maps computed")
    print(f"phi range: [{phi[0]:.3f}, {phi[-1]:.3f}]  n={len(phi)}")
    print(f"log10(alpha) range: [{log10a[0]:.1f}, {log10a[-1]:.1f}]  n={len(log10a)}")

    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    X, Y = np.meshgrid(phi, log10a)  # X: phi, Y: log10(alpha)

    def _panel(fig, ax, Z, title, cbar_label):
        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="cividis")
        ax.set_title(title)
        ax.set_xlabel("Porosity, φ (fraction)")
        ax.set_ylabel(r"$\log_{10}(\alpha)$")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)
        _panel(fig, axes[0, 0], C11_gpa, r"$C_{11}$", "GPa")
        _panel(fig, axes[0, 1], C12_gpa, r"$C_{12}$", "GPa")
        _panel(fig, axes[0, 2], C44_gpa, r"$C_{44}$", "GPa")
        _panel(fig, axes[1, 0], Keff_gpa, r"$K_{\mathrm{eff}}$", "GPa")
        _panel(fig, axes[1, 1], Geff_gpa, r"$G_{\mathrm{eff}}$", "GPa")
        _panel(fig, axes[1, 2], Vp_kms, r"$V_P$", "km/s")

        out_png = out_dir / "gsa_elastic_random_isotropic_maps_phi_log10alpha.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    # Second figure for Vs only (keeps colorbar readable)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.8), constrained_layout=True)
        im = ax.pcolormesh(X, Y, Vs_kms, shading="auto", cmap="cividis")
        ax.set_xlabel("Porosity, φ (fraction)")
        ax.set_ylabel(r"$\log_{10}(\alpha)$")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("km/s")

        out_png = out_dir / "gsa_elastic_random_isotropic_map_vs_phi_log10alpha_Vs.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_dir / 'gsa_elastic_random_isotropic_maps_phi_log10alpha.png'}")
    print(f"Saved: {out_dir / 'gsa_elastic_random_isotropic_map_vs_phi_log10alpha_Vs.png'}")


if __name__ == "__main__":
    main()

