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
    return sorted(out, reverse=True)  # 1 -> 1e-4


def _alpha_math_label(ar: float) -> str:
    ar_f = float(ar)
    if ar_f <= 0 or not np.isfinite(ar_f):
        return "?"
    exp = np.log10(ar_f)
    if np.isclose(exp, round(exp), atol=1e-12):
        e = int(round(exp))
        return rf"10^{{{e}}}"
    return rf"{ar_f:g}"


def _matrix_moduli_from_vp_vs_rho(vp_kms: float, vs_kms: float, rho_gcc: float) -> tuple[float, float]:
    vp = float(vp_kms) * 1e3
    vs = float(vs_kms) * 1e3
    rho = float(rho_gcc) * 1e3
    K = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
    G = rho * (vs * vs)
    return float(K), float(G)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Elastic GSA (aligned VTI): compare stiffness and velocities vs porosity "
            "for different aspect ratios and for gas vs water in pores."
        )
    )
    ap.add_argument(
        "--aspect-ratios",
        nargs="+",
        default=["1", "1e-1", "1e-2", "1e-3", "1e-4"],
        help="Pore aspect ratios α (space-separated), e.g. 1 1e-1 1e-2 1e-3 1e-4",
    )
    ap.add_argument("--phi-max", type=float, default=0.30, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")
    ap.add_argument("--phi-units", choices=("fraction", "percent"), default="percent")

    # Matrix (limestone) default is the paper-like Vp/Vs/rho used earlier.
    ap.add_argument("--matrix-vp-kms", type=float, default=5.8, help="Matrix VP (km/s)")
    ap.add_argument("--matrix-vs-kms", type=float, default=3.2, help="Matrix VS (km/s)")
    ap.add_argument("--matrix-rho-gcc", type=float, default=2.71, help="Matrix density (g/cm^3)")

    # Fluids
    ap.add_argument("--gas-K-gpa", type=float, default=0.02, help="Gas bulk modulus (GPa)")
    ap.add_argument("--gas-rho-gcc", type=float, default=0.0012, help="Gas density (g/cm^3); 1.2 kg/m^3 -> 0.0012")
    ap.add_argument("--water-K-gpa", type=float, default=2.2, help="Water bulk modulus (GPa)")
    ap.add_argument("--water-rho-gcc", type=float, default=1.0, help="Water density (g/cm^3)")

    ap.add_argument(
        "--comparison-body",
        choices=("matrix", "bayuk_linear_mix"),
        default="matrix",
        help="Comparison body strategy used in the aligned VTI solver.",
    )
    ap.add_argument(
        "--k-connectivity",
        type=float,
        default=3.0,
        help="k for comparison_body='bayuk_linear_mix' (f = k*phi, clipped).",
    )
    ap.add_argument(
        "--sign",
        choices=("+", "-"),
        default="-",
        help="Operator sign in A = (I ± G0·ΔC)^-1. For elastic pores/fluids use '-'.",
    )

    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    args = ap.parse_args()

    aspect_ratios = _parse_aspect_ratios(list(args.aspect_ratios))
    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))
    phi_x = phi * 100.0 if args.phi_units == "percent" else phi
    xlab = "Porosity, φ (%)" if args.phi_units == "percent" else "Porosity, φ (fraction)"

    Km_pa, Gm_pa = _matrix_moduli_from_vp_vs_rho(args.matrix_vp_kms, args.matrix_vs_kms, args.matrix_rho_gcc)
    rho_m = float(args.matrix_rho_gcc) * 1e3
    matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)

    # Pore fluids: take shear ~0
    fluids = {
        "Gas": {"K": float(args.gas_K_gpa) * 1e9, "rho": float(args.gas_rho_gcc) * 1e3},
        "Water": {"K": float(args.water_K_gpa) * 1e9, "rho": float(args.water_rho_gcc) * 1e3},
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

    # Compute results: dict[fluid][alpha][series]
    results: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    for fluid_name, fd in fluids.items():
        pores = gsa.ElasticPhase(K=float(fd["K"]), G=1e-9)
        rho_f = float(fd["rho"])
        per_alpha: dict[float, dict[str, np.ndarray]] = {}
        for alpha in aspect_ratios:
            C11 = np.full_like(phi, np.nan, dtype=float)
            C33 = np.full_like(phi, np.nan, dtype=float)
            C44 = np.full_like(phi, np.nan, dtype=float)
            C66 = np.full_like(phi, np.nan, dtype=float)
            vp_h = np.full_like(phi, np.nan, dtype=float)
            vp_v = np.full_like(phi, np.nan, dtype=float)
            vs_h = np.full_like(phi, np.nan, dtype=float)
            vs_v = np.full_like(phi, np.nan, dtype=float)

            for i, ph in enumerate(phi):
                C_eff_66, _ = gsa.gsa_effective_stiffness_aligned_vti_two_phase(
                    phi=float(ph),
                    matrix=matrix,
                    inclusion=pores,
                    pore_aspect_ratio=float(alpha),
                    backend=backend,
                    comparison_body=str(args.comparison_body),
                    k_connectivity=(float(args.k_connectivity) if str(args.comparison_body) == "bayuk_linear_mix" else None),
                    sign=sign,
                )

                C_pa = np.asarray(C_eff_66, dtype=float)
                rho_eff = (1.0 - float(ph)) * rho_m + float(ph) * rho_f

                C11[i] = float(C_pa[0, 0] / 1e9)
                C33[i] = float(C_pa[2, 2] / 1e9)
                C44[i] = float(C_pa[3, 3] / 1e9)
                C66[i] = float(C_pa[5, 5] / 1e9)

                # TI axis: 3 is normal to crack plane.
                vp_h[i] = float(np.sqrt(max(C_pa[0, 0], 0.0) / rho_eff) / 1e3)
                vp_v[i] = float(np.sqrt(max(C_pa[2, 2], 0.0) / rho_eff) / 1e3)
                vs_v[i] = float(np.sqrt(max(C_pa[3, 3], 0.0) / rho_eff) / 1e3)
                vs_h[i] = float(np.sqrt(max(C_pa[5, 5], 0.0) / rho_eff) / 1e3)

            per_alpha[float(alpha)] = {
                "C11": C11,
                "C33": C33,
                "C44": C44,
                "C66": C66,
                "vp_h": vp_h,
                "vp_v": vp_v,
                "vs_h": vs_h,
                "vs_v": vs_v,
            }

        results[fluid_name] = per_alpha

    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }

    # ------------------------------------------------------------------
    # Figure 1: stiffness comparison (panels by alpha)
    # color by fluid: Water=blue, Gas=red; linestyle by component
    # ------------------------------------------------------------------
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharex=True, constrained_layout=True)
        axs = axes.ravel()
        fluid_color = {"Water": "C0", "Gas": "C3"}  # blue, red
        comp_style = {"C11": "-", "C33": "--", "C44": ":", "C66": "-."}

        for i_ax, alpha in enumerate(aspect_ratios):
            ax = axs[i_ax]
            alpha_label = _alpha_math_label(alpha)
            ax.set_title(rf"$\alpha={alpha_label}$")
            ax.grid(False)

            for name in ("Water", "Gas"):
                comp = results[name][float(alpha)]
                col = fluid_color[name]
                if np.isclose(float(alpha), 1.0):
                    ax.plot(phi_x, comp["C11"], color=col, lw=2.0, ls=comp_style["C11"], label=f"{name} $C_{{11}}$")
                    ax.plot(phi_x, comp["C44"], color=col, lw=2.0, ls=comp_style["C44"], label=f"{name} $C_{{44}}$")
                else:
                    ax.plot(phi_x, comp["C11"], color=col, lw=2.0, ls=comp_style["C11"], label=f"{name} $C_{{11}}$")
                    ax.plot(phi_x, comp["C33"], color=col, lw=2.0, ls=comp_style["C33"], label=f"{name} $C_{{33}}$")
                    ax.plot(phi_x, comp["C44"], color=col, lw=2.0, ls=comp_style["C44"], label=f"{name} $C_{{44}}$")
                    ax.plot(phi_x, comp["C66"], color=col, lw=2.0, ls=comp_style["C66"], label=f"{name} $C_{{66}}$")

            ax.set_ylabel(r"$C_{ij}$, GPa")
            ax.set_xlim(phi_x[0], phi_x[-1])

        # Use the last (empty) panel for a readable legend (avoid overlap).
        legend_ax = axs[-1]
        legend_ax.axis("off")
        for ax in axes[1, :]:
            ax.set_xlabel(xlab)

        handles, labels = axs[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", frameon=False, ncol=1, handlelength=3.0)

        out_png = out_dir / "gsa_elastic_aligned_vti_stiffness_gas_vs_water_by_alpha.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: velocity comparison (Vp_h, Vp_v, Vs_h, Vs_v), panels by alpha
    # ------------------------------------------------------------------
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharex=True, constrained_layout=True)
        axs = axes.ravel()
        # Match the requested paper-like styling:
        # - color by wave type (Vp_h, Vp_v, Vs_h, Vs_v)
        # - linestyle by fluid (Water solid, Gas dashed)
        comp_color = {"vp_h": "C0", "vp_v": "C3", "vs_h": "C2", "vs_v": "C4"}  # blue, red, green, purple
        fluid_style = {"Water": "-", "Gas": "--"}
        for i_ax, alpha in enumerate(aspect_ratios):
            ax = axs[i_ax]
            alpha_label = _alpha_math_label(alpha)
            ax.set_title(rf"$\alpha={alpha_label}$")
            ax.grid(False)

            for name in ("Water", "Gas"):
                comp = results[name][float(alpha)]
                ls = fluid_style[name]
                ax.plot(phi_x, comp["vp_h"], color=comp_color["vp_h"], lw=2.2, ls=ls, label=f"{name} $V_{{P,h}}$")
                ax.plot(phi_x, comp["vp_v"], color=comp_color["vp_v"], lw=2.2, ls=ls, label=f"{name} $V_{{P,v}}$")
                ax.plot(phi_x, comp["vs_h"], color=comp_color["vs_h"], lw=2.2, ls=ls, label=f"{name} $V_{{S,h}}$")
                ax.plot(phi_x, comp["vs_v"], color=comp_color["vs_v"], lw=2.2, ls=ls, label=f"{name} $V_{{S,v}}$")

            ax.set_ylabel("km/s")
            ax.set_xlim(phi_x[0], phi_x[-1])
            ax.margins(y=0.05)

        legend_ax = axs[-1]
        legend_ax.axis("off")
        for ax in axes[1, :]:
            ax.set_xlabel(xlab)

        handles, labels = axs[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", frameon=False, ncol=1, handlelength=3.0)

        out_png = out_dir / "gsa_elastic_aligned_vti_velocities_gas_vs_water_by_alpha.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_dir / 'gsa_elastic_aligned_vti_stiffness_gas_vs_water_by_alpha.png'}")
    print(f"Saved: {out_dir / 'gsa_elastic_aligned_vti_velocities_gas_vs_water_by_alpha.png'}")


if __name__ == "__main__":
    main()
