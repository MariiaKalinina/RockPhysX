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


def _parse_aspect_ratios(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("At least one aspect ratio is required.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic GSA (isotropic random) sweep vs porosity.")
    ap.add_argument(
        "--aspect-ratios",
        nargs="+",
        default=["1e-4", "1e-3", "1e-2", "1e-1", "1"],
        help="Pore aspect ratios α (space-separated), e.g. 1e-4 1e-3 1e-2 1e-1 1",
    )
    ap.add_argument("--phi-max", type=float, default=0.35, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")

    ap.add_argument("--matrix-K-gpa", type=float, default=76.0, help="Matrix bulk modulus (GPa)")
    ap.add_argument("--matrix-G-gpa", type=float, default=32.0, help="Matrix shear modulus (GPa)")
    ap.add_argument("--matrix-rho", type=float, default=2720.0, help="Matrix density (kg/m^3)")

    ap.add_argument(
        "--pore-K-gpa",
        type=float,
        default=0.0,
        help="Pore/fluid bulk modulus (GPa). Use 0 for dry pores (internally clamped to epsilon).",
    )
    ap.add_argument(
        "--pore-G-gpa",
        type=float,
        default=0.0,
        help="Pore/fluid shear modulus (GPa). Use 0 for fluids/pores (internally clamped to epsilon).",
    )
    ap.add_argument("--pore-rho", type=float, default=1.2, help="Pore/fluid density (kg/m^3)")
    ap.add_argument(
        "--velocity-density",
        choices=("effective", "matrix"),
        default="effective",
        help="How to compute velocities: use effective bulk density (physical) or fixed matrix density (diagnostic).",
    )

    ap.add_argument(
        "--green-fortran",
        type=Path,
        default=Path("GREEN_ANAL_VTI.f90"),
        help="Path to GREEN_ANAL_VTI.f90 (absolute or relative; searched relative to repo/module if needed).",
    )
    ap.add_argument(
        "--izotr-fortran",
        type=Path,
        default=None,
        help="Optional path to subr_IZOTR.f90 (not required; built-in gfortran-compatible izotr is used).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    args = ap.parse_args()

    aspect_ratios = _parse_aspect_ratios(list(args.aspect_ratios))

    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))

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
        izotr_fortran=args.izotr_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )

    curves: dict[float, dict[str, np.ndarray]] = {}
    for ar in aspect_ratios:
        K_eff = np.full_like(phi, np.nan, dtype=float)
        G_eff = np.full_like(phi, np.nan, dtype=float)
        vp = np.full_like(phi, np.nan, dtype=float)
        vs = np.full_like(phi, np.nan, dtype=float)

        for i, ph in enumerate(phi):
            res = gsa.gsa_elastic_random_spheroidal_pores(
                phi=float(ph),
                a_ratio=float(ar),
                matrix=matrix,
                inclusion=pores,
                rho_matrix_kg_m3=float(args.matrix_rho),
                rho_inclusion_kg_m3=float(args.pore_rho),
                backend=backend,
                green_fortran=args.green_fortran,
                izotr_fortran=args.izotr_fortran,
            )
            K_eff[i] = float(res.K_eff)
            G_eff[i] = float(res.G_eff)
            if args.velocity_density == "matrix":
                vp_i, vs_i = gsa.velocities_from_KG_rho(res.K_eff, res.G_eff, float(args.matrix_rho))
                vp[i] = float(vp_i)
                vs[i] = float(vs_i)
            else:
                vp[i] = float(res.vp_m_s)
                vs[i] = float(res.vs_m_s)

        curves[float(ar)] = {"K_eff": K_eff, "G_eff": G_eff, "vp": vp, "vs": vs}

    # Terminal summary
    print("Elastic GSA (isotropic random): sweep vs porosity")
    print(f"Matrix: K={args.matrix_K_gpa:.2f} GPa, G={args.matrix_G_gpa:.2f} GPa, rho={args.matrix_rho:.0f} kg/m3")
    print(f"Pores/fluid (inputs): K={args.pore_K_gpa:.2g} GPa, G={args.pore_G_gpa:.2g} GPa, rho={args.pore_rho:.2g} kg/m3")
    if args.velocity_density == "matrix":
        print("Velocity mode: using fixed matrix density (diagnostic).")
    else:
        print("Velocity mode: using effective bulk density (physical).")
    print(f"Aspect ratios: {', '.join(f'{a:g}' for a in aspect_ratios)}")

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

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6), sharex=True, constrained_layout=True)
        axK, axG, axVp, axVs = axes.ravel()

        for ar in aspect_ratios:
            c = curves[float(ar)]
            ar_f = float(ar)
            if any(np.isclose(ar_f, v) for v in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
                exp = int(round(np.log10(ar_f))) if ar_f > 0 else 0
                label = rf"$\alpha = 10^{{{exp}}}$" if not np.isclose(ar_f, 1.0) else r"$\alpha = 10^{0}$"
            else:
                label = rf"$\alpha={ar_f:g}$"
            axK.plot(phi, c["K_eff"] / 1e9, lw=2.2, label=label)
            axG.plot(phi, c["G_eff"] / 1e9, lw=2.2)
            axVp.plot(phi, c["vp"] / 1e3, lw=2.2)
            axVs.plot(phi, c["vs"] / 1e3, lw=2.2)

        axK.set_ylabel("Keff (GPa)")
        axG.set_ylabel("Geff (GPa)")
        axVp.set_ylabel("Vp (km/s)")
        axVs.set_ylabel("Vs (km/s)")
        for ax in (axK, axG, axVp, axVs):
            ax.grid(True, alpha=0.25)
        for ax in (axVp, axVs):
            ax.set_xlabel("Porosity, φ (fraction)")

        handles, labels = axK.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, -0.02))

        out_png = out_dir / "gsa_elastic_random_isotropic_vs_phi.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
