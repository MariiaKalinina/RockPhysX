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


def _compute_curves(
    *,
    phi: np.ndarray,
    aspect_ratios: list[float],
    matrix: gsa.ElasticPhase,
    pores: gsa.ElasticPhase,
    rho_matrix_kg_m3: float,
    rho_pore_kg_m3: float,
    velocity_density: str,
    backend: gsa._Backend,
    green_fortran: Path,
) -> dict[float, dict[str, np.ndarray]]:
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
                rho_matrix_kg_m3=float(rho_matrix_kg_m3),
                rho_inclusion_kg_m3=float(rho_pore_kg_m3),
                backend=backend,
                green_fortran=green_fortran,
            )
            K_eff[i] = float(res.K_eff)
            G_eff[i] = float(res.G_eff)
            if velocity_density == "matrix":
                vp_i, vs_i = gsa.velocities_from_KG_rho(res.K_eff, res.G_eff, float(rho_matrix_kg_m3))
                vp[i] = float(vp_i)
                vs[i] = float(vs_i)
            else:
                vp[i] = float(res.vp_m_s)
                vs[i] = float(res.vs_m_s)

        curves[float(ar)] = {"K_eff": K_eff, "G_eff": G_eff, "vp": vp, "vs": vs}
    return curves


def _alpha_label(ar: float) -> str:
    ar_f = float(ar)
    if any(np.isclose(ar_f, v) for v in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
        exp = int(round(np.log10(ar_f))) if ar_f > 0 else 0
        return rf"$\alpha = 10^{{{exp}}}$" if not np.isclose(ar_f, 1.0) else r"$\alpha = 10^{0}$"
    return rf"$\alpha={ar_f:g}$"


def _plot_2x2(
    *,
    out_png: Path,
    phi: np.ndarray,
    curves: dict[float, dict[str, np.ndarray]],
    aspect_ratios: list[float],
    title: str,
    dpi: int,
) -> None:
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
            label = _alpha_label(float(ar))
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

        if title:
            fig.suptitle(title)

        handles, labels = axK.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, -0.02))

        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic GSA (isotropic random): calcite matrix with water/gas pores.")
    ap.add_argument(
        "--aspect-ratios",
        nargs="+",
        default=["1e-4", "1e-3", "1e-2", "1e-1", "1"],
        help="Pore aspect ratios α (space-separated), e.g. 1e-4 1e-3 1e-2 1e-1 1",
    )
    ap.add_argument("--phi-max", type=float, default=0.35, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")

    # Calcite (defaults aligned with what you used in prior plots)
    ap.add_argument("--matrix-K-gpa", type=float, default=76.0, help="Calcite bulk modulus (GPa)")
    ap.add_argument("--matrix-G-gpa", type=float, default=32.0, help="Calcite shear modulus (GPa)")
    ap.add_argument("--matrix-rho", type=float, default=2710.0, help="Calcite density (kg/m^3)")

    # Water
    ap.add_argument("--water-K-gpa", type=float, default=2.2, help="Water bulk modulus (GPa)")
    ap.add_argument("--water-rho", type=float, default=1000.0, help="Water density (kg/m^3)")

    # Gas (bulk modulus depends on pressure; default is a typical soft-gas placeholder)
    ap.add_argument("--gas-K-gpa", type=float, default=0.02, help="Gas bulk modulus (GPa)")
    ap.add_argument("--gas-rho", type=float, default=1.2, help="Gas density (kg/m^3)")

    ap.add_argument(
        "--velocity-density",
        choices=("effective", "matrix"),
        default="effective",
        help="How to compute velocities: use effective bulk density (physical) or fixed matrix density (diagnostic).",
    )

    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    args = ap.parse_args()

    aspect_ratios = _parse_aspect_ratios(list(args.aspect_ratios))
    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))

    matrix = gsa.ElasticPhase(K=float(args.matrix_K_gpa) * 1e9, G=float(args.matrix_G_gpa) * 1e9)
    water = gsa.ElasticPhase(K=max(float(args.water_K_gpa) * 1e9, 1e-9), G=1e-9)
    gas = gsa.ElasticPhase(K=max(float(args.gas_K_gpa) * 1e9, 1e-9), G=1e-9)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )

    curves_water = _compute_curves(
        phi=phi,
        aspect_ratios=aspect_ratios,
        matrix=matrix,
        pores=water,
        rho_matrix_kg_m3=float(args.matrix_rho),
        rho_pore_kg_m3=float(args.water_rho),
        velocity_density=str(args.velocity_density),
        backend=backend,
        green_fortran=args.green_fortran,
    )
    curves_gas = _compute_curves(
        phi=phi,
        aspect_ratios=aspect_ratios,
        matrix=matrix,
        pores=gas,
        rho_matrix_kg_m3=float(args.matrix_rho),
        rho_pore_kg_m3=float(args.gas_rho),
        velocity_density=str(args.velocity_density),
        backend=backend,
        green_fortran=args.green_fortran,
    )

    title_water = f"Calcite + water-filled pores ({args.velocity_density} density)"
    title_gas = f"Calcite + gas-filled pores ({args.velocity_density} density)"

    out_water = out_dir / "gsa_elastic_calcite_water_vs_phi.png"
    out_gas = out_dir / "gsa_elastic_calcite_gas_vs_phi.png"

    _plot_2x2(out_png=out_water, phi=phi, curves=curves_water, aspect_ratios=aspect_ratios, title=title_water, dpi=int(args.dpi))
    _plot_2x2(out_png=out_gas, phi=phi, curves=curves_gas, aspect_ratios=aspect_ratios, title=title_gas, dpi=int(args.dpi))

    print(f"Saved: {out_water}")
    print(f"Saved: {out_gas}")


if __name__ == "__main__":
    main()

