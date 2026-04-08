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


def _parse_phi(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("At least one porosity value is required.")
    for ph in out:
        if not (0.0 <= ph <= 1.0):
            raise ValueError(f"Porosity must be in [0,1], got {ph}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic GSA (isotropic random): sweep vs log10(alpha) at fixed porosities.")
    ap.add_argument(
        "--phi-values",
        nargs="+",
        default=["0.05", "0.10", "0.20", "0.30"],
        help="Porosity values (fractions), e.g. 0.05 0.10 0.20 0.30",
    )
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

    phi_values = _parse_phi(list(args.phi_values))
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

    curves: dict[float, dict[str, np.ndarray]] = {}
    for ph in phi_values:
        Keff = np.full_like(alpha, np.nan, dtype=float)
        Geff = np.full_like(alpha, np.nan, dtype=float)
        vp = np.full_like(alpha, np.nan, dtype=float)
        vs = np.full_like(alpha, np.nan, dtype=float)
        for i, a in enumerate(alpha):
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
            Keff[i] = float(res.K_eff)
            Geff[i] = float(res.G_eff)
            if args.velocity_density == "matrix":
                vp_i, vs_i = gsa.velocities_from_KG_rho(res.K_eff, res.G_eff, float(args.matrix_rho))
                vp[i] = float(vp_i)
                vs[i] = float(vs_i)
            else:
                vp[i] = float(res.vp_m_s)
                vs[i] = float(res.vs_m_s)
        curves[float(ph)] = {"Keff": Keff, "Geff": Geff, "vp": vp, "vs": vs}

    print("Elastic GSA (isotropic random): sweep vs log10(alpha)")
    print(f"phi values: {', '.join(f'{p:.3f}' for p in phi_values)}")
    print(f"log10(alpha) range: [{log10a[0]:.1f}, {log10a[-1]:.1f}]  n={len(log10a)}")
    print(f"Velocity mode: {args.velocity_density}")

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

        for ph in phi_values:
            c = curves[float(ph)]
            label = rf"$\phi={float(ph):.2f}$"
            axK.plot(log10a, c["Keff"] / 1e9, lw=2.2, label=label)
            axG.plot(log10a, c["Geff"] / 1e9, lw=2.2)
            axVp.plot(log10a, c["vp"] / 1e3, lw=2.2)
            axVs.plot(log10a, c["vs"] / 1e3, lw=2.2)

        axK.set_ylabel("Keff (GPa)")
        axG.set_ylabel("Geff (GPa)")
        axVp.set_ylabel("Vp (km/s)")
        axVs.set_ylabel("Vs (km/s)")
        for ax in (axK, axG, axVp, axVs):
            ax.grid(True, alpha=0.25)
        for ax in (axVp, axVs):
            ax.set_xlabel(r"$\log_{10}(\alpha)$")

        handles, labels = axK.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=True, bbox_to_anchor=(0.5, -0.02))

        out_png = out_dir / "gsa_elastic_random_isotropic_vs_log10_alpha.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

