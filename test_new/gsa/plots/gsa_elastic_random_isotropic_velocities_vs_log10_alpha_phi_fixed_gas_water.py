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


def _parse_aspect_ratios(values: list[str]) -> np.ndarray:
    out: list[float] = []
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("At least one aspect ratio is required.")
    ar = np.asarray(out, dtype=float)
    if np.any(~np.isfinite(ar)) or np.any(ar <= 0.0):
        raise ValueError("All aspect ratios must be positive and finite.")
    return np.sort(ar)  # ascending for x=log10(alpha)


def _matrix_moduli_from_vp_vs_rho(vp_kms: float, vs_kms: float, rho_gcc: float) -> tuple[float, float]:
    vp = float(vp_kms) * 1e3
    vs = float(vs_kms) * 1e3
    rho = float(rho_gcc) * 1e3
    K = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
    G = rho * (vs * vs)
    return float(K), float(G)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Elastic GSA (isotropic random): Vp, Vs vs log10(alpha) for a fixed porosity, gas vs water."
    )
    ap.add_argument(
        "--aspect-ratios",
        nargs="+",
        default=["1e-4", "1e-3", "1e-2", "1e-1", "1"],
        help="Aspect ratios α (space-separated), e.g. 1e-4 1e-3 1e-2 1e-1 1",
    )
    ap.add_argument("--alpha-min", type=float, default=1e-4, help="Minimum α for auto grid (used with --n-alpha)")
    ap.add_argument("--alpha-max", type=float, default=1.0, help="Maximum α for auto grid (used with --n-alpha)")
    ap.add_argument(
        "--n-alpha",
        type=int,
        default=0,
        help="If >0, ignore --aspect-ratios and generate log-spaced α grid with this many points.",
    )
    ap.add_argument("--phi-fixed", type=float, default=0.10, help="Fixed porosity (fraction), e.g. 0.10")

    ap.add_argument("--matrix-vp-kms", type=float, default=5.8, help="Matrix VP (km/s)")
    ap.add_argument("--matrix-vs-kms", type=float, default=3.2, help="Matrix VS (km/s)")
    ap.add_argument("--matrix-rho-gcc", type=float, default=2.71, help="Matrix density (g/cm^3)")

    ap.add_argument("--gas-K-gpa", type=float, default=0.02, help="Gas bulk modulus (GPa)")
    ap.add_argument("--gas-rho-gcc", type=float, default=0.0012, help="Gas density (g/cm^3)")
    ap.add_argument("--water-K-gpa", type=float, default=2.2, help="Water bulk modulus (GPa)")
    ap.add_argument("--water-rho-gcc", type=float, default=1.0, help="Water density (g/cm^3)")

    ap.add_argument(
        "--comparison-body",
        choices=("matrix", "bayuk_linear_mix"),
        default="matrix",
        help="Comparison body strategy used in the isotropic-random solver.",
    )
    ap.add_argument("--k-connectivity", type=float, default=3.0, help="k for comparison_body='bayuk_linear_mix'.")
    ap.add_argument("--sign", choices=("+", "-"), default="-", help="Operator sign in A = (I ± G0·ΔC)^-1.")

    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    args = ap.parse_args()

    phi = float(args.phi_fixed)
    if not (0.0 <= phi <= 1.0):
        raise ValueError("--phi-fixed must lie in [0, 1].")

    if int(args.n_alpha) > 0:
        a_min = float(args.alpha_min)
        a_max = float(args.alpha_max)
        if not np.isfinite(a_min) or not np.isfinite(a_max) or a_min <= 0.0 or a_max <= 0.0:
            raise ValueError("--alpha-min/--alpha-max must be positive and finite.")
        if a_min >= a_max:
            raise ValueError("--alpha-min must be < --alpha-max.")
        aspect_ratios = np.logspace(np.log10(a_min), np.log10(a_max), int(args.n_alpha))
    else:
        aspect_ratios = _parse_aspect_ratios(list(args.aspect_ratios))
    x = np.log10(aspect_ratios)

    Km_pa, Gm_pa = _matrix_moduli_from_vp_vs_rho(args.matrix_vp_kms, args.matrix_vs_kms, args.matrix_rho_gcc)
    rho_m = float(args.matrix_rho_gcc) * 1e3
    matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)

    fluids = {
        "Water": {"K": float(args.water_K_gpa) * 1e9, "rho": float(args.water_rho_gcc) * 1e3, "color": "C0"},
        "Gas": {"K": float(args.gas_K_gpa) * 1e9, "rho": float(args.gas_rho_gcc) * 1e3, "color": "C3"},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )
    sign = (-1 if args.sign == "-" else +1)

    series: dict[str, dict[str, np.ndarray]] = {}
    for fluid_name, fd in fluids.items():
        pores = gsa.ElasticPhase(K=float(fd["K"]), G=1e-9)
        rho_f = float(fd["rho"])
        vp = np.full_like(aspect_ratios, np.nan, dtype=float)
        vs = np.full_like(aspect_ratios, np.nan, dtype=float)
        for i, ar in enumerate(aspect_ratios):
            _, _, K_eff, G_eff = gsa.gsa_effective_stiffness_random_two_phase(
                phi=phi,
                matrix=matrix,
                inclusion=pores,
                pore_aspect_ratio=float(ar),
                backend=backend,
                comparison_body=str(args.comparison_body),
                k_connectivity=(float(args.k_connectivity) if str(args.comparison_body) == "bayuk_linear_mix" else None),
                sign=sign,
            )
            rho_eff = (1.0 - phi) * rho_m + phi * rho_f
            vp[i] = float(np.sqrt(max(K_eff + 4.0 * G_eff / 3.0, 0.0) / rho_eff) / 1e3)
            vs[i] = float(np.sqrt(max(G_eff, 0.0) / rho_eff) / 1e3)
        series[fluid_name] = {"vp": vp, "vs": vs, "color": np.array([fd["color"]])}

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
        fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.6), constrained_layout=True)
        for fluid_name, d in series.items():
            color = str(d["color"][0])
            ax.plot(x, d["vp"], color=color, lw=2.3, ls="-", label=f"{fluid_name} $V_P$")
            ax.plot(x, d["vs"], color=color, lw=2.3, ls="--", label=f"{fluid_name} $V_S$")

        ax.set_xlabel(r"$\log_{10}(\alpha)$")
        ax.set_ylabel("km/s")
        ax.grid(True, alpha=0.25)
        ax.set_title(rf"Isotropic random GSA, $\phi={phi:.2f}$")
        ax.legend(frameon=False, ncol=2, loc="best")

        out_png = out_dir / f"gsa_elastic_random_isotropic_velocities_vs_log10_alpha_phi_{phi:.2f}_gas_water.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
