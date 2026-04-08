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


def _alpha_math_label(ar: float) -> str:
    ar_f = float(ar)
    if ar_f <= 0 or not np.isfinite(ar_f):
        return "?"
    exp = np.log10(ar_f)
    if np.isclose(exp, round(exp), atol=1e-12):
        e = int(round(exp))
        return rf"10^{{{e}}}"
    return rf"{ar_f:g}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Elastic GSA (aligned VTI): stiffness components vs porosity (paper-style annotations)."
    )
    ap.add_argument(
        "--aspect-ratios",
        nargs="+",
        default=["1e-4", "1e-3", "1e-2", "1e-1", "1"],
        help="Pore aspect ratios α (space-separated), e.g. 1e-4 1e-3 1e-2 1e-1 1",
    )
    ap.add_argument(
        "--alpha-definition",
        choices=("a3_over_a1", "a1_over_a3"),
        default="a3_over_a1",
        help=(
            "How to interpret the provided α values. "
            "Use 'a3_over_a1' when α is thickness/diameter (crack-like pores: α<<1), "
            "which also ensures C33<C11 for aligned oblate pores. "
            "Use 'a1_over_a3' only if your source explicitly defines α as a1/a3."
        ),
    )
    ap.add_argument(
        "--alpha-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative factor applied to α after definition conversion (use only to match a reference figure).",
    )
    ap.add_argument("--phi-max", type=float, default=0.30, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")
    ap.add_argument("--phi-units", choices=("fraction", "percent"), default="percent")

    # Either specify matrix moduli directly, or via (Vp, Vs, rho) as in the paper.
    ap.add_argument("--matrix-K-gpa", type=float, default=76.0, help="Matrix bulk modulus (GPa) [used if Vp/Vs/rho not provided]")
    ap.add_argument("--matrix-G-gpa", type=float, default=32.0, help="Matrix shear modulus (GPa) [used if Vp/Vs/rho not provided]")
    ap.add_argument("--matrix-vp-kms", type=float, default=5.8, help="Matrix VP (km/s) [paper default]")
    ap.add_argument("--matrix-vs-kms", type=float, default=3.2, help="Matrix VS (km/s) [paper default]")
    ap.add_argument("--matrix-rho-gcc", type=float, default=2.71, help="Matrix density (g/cm^3) [paper default]")

    # Fluid/pore phase: either specify K directly, or via Vp and rho (G=0).
    ap.add_argument(
        "--fluid-mode",
        choices=("vp_rho", "K"),
        default="vp_rho",
        help="How to define the pore/fluid bulk modulus: via (Vp,rho) or directly via K.",
    )
    ap.add_argument("--pore-K-gpa", type=float, default=0.02, help="Pore/fluid bulk modulus (GPa) [used if fluid Vp/rho not provided]")
    ap.add_argument("--fluid-vp-kms", type=float, default=0.2, help="Fluid VP (km/s) [paper default: gas-condensate]")
    ap.add_argument("--fluid-rho-gcc", type=float, default=0.65, help="Fluid density (g/cm^3) [paper default: gas-condensate]")
    ap.add_argument(
        "--comparison-body",
        choices=("matrix", "bayuk_linear_mix", "self_consistent"),
        default="matrix",
        help=(
            "Comparison body strategy. "
            "'matrix' uses C0=Cm (one-shot). "
            "'bayuk_linear_mix' uses C0=(1-f)Cm+fCp with f=k*phi (clipped). "
            "'self_consistent' solves C0=C* by fixed-point."
        ),
    )
    ap.add_argument(
        "--k-connectivity",
        type=float,
        default=3.0,
        help="Connectivity/friability slope k used only for comparison_body='bayuk_linear_mix' (f = k*phi, clipped to [0,1]).",
    )
    ap.add_argument(
        "--sign",
        choices=("+", "-"),
        default="-",
        help="Operator sign in A = (I ± G0·ΔC)^-1. For elastic pores/fluids the conventional choice is '-'.",
    )
    ap.add_argument("--sc-max-iter", type=int, default=300, help="Max iterations for self_consistent mode")
    ap.add_argument("--sc-tol", type=float, default=1e-10, help="Abs tolerance for self_consistent mode")
    ap.add_argument("--sc-relax", type=float, default=1.0, help="Damping/relaxation for self_consistent mode (0..1)")
    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    ap.add_argument(
        "--export-per-alpha",
        action="store_true",
        help="Also export separate figures per alpha (alpha=1: C11,C44; others: C11,C33,C44,C66).",
    )
    ap.add_argument(
        "--ymax",
        type=float,
        default=0.0,
        help="Upper y-limit in GPa for all panels (set <=0 to auto). Default: auto.",
    )
    args = ap.parse_args()

    # For paper-style figure, keep panels ordered from round pores (α=1)
    # to crack-like pores (α→0), i.e. descending α.
    aspect_ratios = sorted(_parse_aspect_ratios(list(args.aspect_ratios)), reverse=True)
    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))
    phi_x = phi * 100.0 if args.phi_units == "percent" else phi

    # Build matrix moduli.
    if args.matrix_vp_kms is not None and args.matrix_vs_kms is not None and args.matrix_rho_gcc is not None:
        vp_m = float(args.matrix_vp_kms) * 1e3
        vs_m = float(args.matrix_vs_kms) * 1e3
        rho_m = float(args.matrix_rho_gcc) * 1e3
        Km_pa = rho_m * (vp_m * vp_m - (4.0 / 3.0) * vs_m * vs_m)
        Gm_pa = rho_m * (vs_m * vs_m)
    else:
        Km_pa = float(args.matrix_K_gpa) * 1e9
        Gm_pa = float(args.matrix_G_gpa) * 1e9

    # Build pore/fluid moduli (shear ~ 0).
    if args.fluid_mode == "vp_rho":
        vp_f = float(args.fluid_vp_kms) * 1e3
        rho_f = float(args.fluid_rho_gcc) * 1e3
        Kp_pa = max(rho_f * vp_f * vp_f, 1e-9)
    else:
        Kp_pa = max(float(args.pore_K_gpa) * 1e9, 1e-9)
    Gp_pa = 1e-9

    matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)
    pores = gsa.ElasticPhase(K=Kp_pa, G=Gp_pa)

    # Echo actual elastic parameters (useful for matching papers).
    vp_m = (Km_pa + 4.0 * Gm_pa / 3.0) ** 0.5 / (float(args.matrix_rho_gcc) * 1e3) ** 0.5
    vs_m = (Gm_pa) ** 0.5 / (float(args.matrix_rho_gcc) * 1e3) ** 0.5
    print(
        "Matrix:",
        f"K={Km_pa/1e9:.3f} GPa, G={Gm_pa/1e9:.3f} GPa, "
        f"rho={float(args.matrix_rho_gcc):.3f} g/cc, "
        f"Vp={vp_m/1e3:.3f} km/s, Vs={vs_m/1e3:.3f} km/s",
    )
    print(
        "Fluid/pore:",
        (
            f"K={Kp_pa/1e9:.4f} GPa, G~0, rho={float(args.fluid_rho_gcc):.3f} g/cc, Vp={float(args.fluid_vp_kms):.3f} km/s"
            if args.fluid_mode == "vp_rho"
            else (
                f"K={Kp_pa/1e9:.4f} GPa, G~0, rho={float(args.fluid_rho_gcc):.3f} g/cc, "
                f"Vp_equiv={((Kp_pa)/(float(args.fluid_rho_gcc)*1e3))**0.5/1e3:.3f} km/s (from K, rho)"
            )
        ),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )

    def _to_model_aspect_ratio(alpha_user: float) -> float:
        a = float(alpha_user)
        if args.alpha_definition == "a3_over_a1":
            ar = a
        # user provides a1/a3 -> model expects a3/a1
        else:
            ar = 1.0 / a
        return float(args.alpha_scale) * ar

    # curves per alpha (all in GPa); keep alpha labels as user provided
    curves: dict[float, dict[str, np.ndarray]] = {}
    for alpha_user in aspect_ratios:
        ar_model = _to_model_aspect_ratio(float(alpha_user))
        C11 = np.full_like(phi, np.nan, dtype=float)
        C33 = np.full_like(phi, np.nan, dtype=float)
        C13 = np.full_like(phi, np.nan, dtype=float)
        C44 = np.full_like(phi, np.nan, dtype=float)
        C66 = np.full_like(phi, np.nan, dtype=float)

        # In self-consistent mode use continuation in porosity to improve convergence.
        C0_guess: np.ndarray | None = None
        for i, ph in enumerate(phi):
            C_eff_66, _ = gsa.gsa_effective_stiffness_aligned_vti_two_phase(
                phi=float(ph),
                matrix=matrix,
                inclusion=pores,
                pore_aspect_ratio=float(ar_model),
                backend=backend,
                comparison_body=str(args.comparison_body),
                k_connectivity=(float(args.k_connectivity) if str(args.comparison_body) == "bayuk_linear_mix" else None),
                sign=(-1 if args.sign == "-" else +1),
                max_iter=int(args.sc_max_iter),
                tol=float(args.sc_tol),
                relaxation=float(args.sc_relax),
                initial_C0_voigt=C0_guess if str(args.comparison_body) == "self_consistent" else None,
            )
            if str(args.comparison_body) == "self_consistent":
                C0_guess = np.asarray(C_eff_66, dtype=float)
            C = np.asarray(C_eff_66, dtype=float) / 1e9
            C11[i] = float(C[0, 0])
            C33[i] = float(C[2, 2])
            C13[i] = float(C[0, 2])
            C44[i] = float(C[3, 3])
            C66[i] = float(C[5, 5])

        curves[float(alpha_user)] = {"C11": C11, "C33": C33, "C13": C13, "C44": C44, "C66": C66}

    import matplotlib.pyplot as plt

    # Choose a shared y-limit so nothing is clipped.
    if float(args.ymax) > 0:
        y_top = float(args.ymax)
    else:
        max_vals: list[float] = []
        for ar in aspect_ratios:
            comp = curves[float(ar)]
            if np.isclose(float(ar), 1.0):
                series = [comp["C11"], comp["C44"]]
            else:
                series = [comp["C11"], comp["C33"], comp["C44"], comp["C66"]]
            for s in series:
                if np.size(s):
                    max_vals.append(float(np.nanmax(s)))
        m = float(np.nanmax(max_vals)) if max_vals else 1.0
        # add headroom and round to a nice tick (10 GPa)
        y_top = float(np.ceil(1.06 * m / 10.0) * 10.0)
        y_top = max(y_top, 10.0)

    # Paper-like minimal style: monochrome lines + different dashes, labels on curves.
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }

    def _common_axis_style(ax):
        ax.grid(False)
        ax.set_ylabel(r"$C_{ij}$, ГПа")
        ax.set_xlim(phi_x[0], phi_x[-1])
        ax.set_ylim(0.0, y_top)
        ax.margins(y=0.02)

    xlab = "Porosity, φ (%)" if args.phi_units == "percent" else "Porosity, φ (fraction)"

    def _save_grid_by_alpha() -> Path:
        """
        One figure, panels by alpha (as in the paper screenshot).

        Layout: 2x3 with the last panel empty for 5 aspect ratios.
        """
        with plt.rc_context(rc):
            fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True, sharey=True, constrained_layout=True)
            axs = axes.ravel()

            # Colors consistent with typical paper plots
            color_C11 = "#0047AB"  # blue
            color_C33 = "#D62728"  # red
            color_C44 = "#00A650"  # green
            color_C66 = "#6A5ACD"  # purple-ish

            for ax in axs:
                ax.grid(False)

            for idx, ar in enumerate(aspect_ratios[:5]):
                ax = axs[idx]
                _common_axis_style(ax)
                ax.set_xlabel("Пористость, %" if args.phi_units == "percent" else xlab)

                comp = curves[float(ar)]

                # alpha=1 panel: only C11 and C44 (as requested)
                if np.isclose(float(ar), 1.0):
                    ax.plot(phi_x, comp["C11"], color=color_C11, lw=2.0, label=f"ceff_11_{float(ar):g}")
                    ax.plot(phi_x, comp["C44"], color=color_C33, lw=2.0, label=f"ceff_44_{float(ar):g}")
                else:
                    # panels for alpha=1e-1..1e-4: C11, C33, C44, C66
                    tag = f"{float(ar):g}"
                    ax.plot(phi_x, comp["C11"], color=color_C11, lw=2.0, label=f"ceff_11_{tag}")
                    ax.plot(phi_x, comp["C33"], color=color_C33, lw=2.0, label=f"ceff_33_{tag}")
                    ax.plot(phi_x, comp["C44"], color=color_C44, lw=2.0, label=f"ceff_44_{tag}")
                    ax.plot(phi_x, comp["C66"], color=color_C66, lw=2.0, label=f"ceff_66_{tag}")

                ax.legend(loc="upper right", frameon=False, handlelength=2.8)

            # Hide any remaining axes
            for j in range(min(5, len(aspect_ratios)), len(axs)):
                axs[j].axis("off")

            out_png = out_dir / "gsa_elastic_aligned_vti_components_by_alpha_single_figure.png"
            fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
            plt.close(fig)
        return out_png

    def _slug_alpha(a: float) -> str:
        s = f"{a:.0e}".replace("+", "").replace("e-0", "e-").replace("e0", "e0")
        return s

    def _save_per_alpha(a: float) -> Path:
        """
        Requested exports:
        - alpha=1: C11, C44
        - other: C11, C33, C44, C66
        """
        a = float(a)
        comp = curves[a]
        with plt.rc_context(rc):
            if np.isclose(a, 1.0):
                fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharex=True, constrained_layout=True)
                ax1, ax2 = axes
                ax1.set_title(r"$C_{11}$")
                _common_axis_style(ax1)
                ax1.plot(phi_x, comp["C11"], color="black", lw=2.0)
                ax2.set_title(r"$C_{44}$")
                _common_axis_style(ax2)
                ax2.plot(phi_x, comp["C44"], color="black", lw=2.0)
                for ax in axes:
                    ax.set_xlabel(xlab)
                fig.suptitle(rf"Aligned pores: $\alpha={_alpha_math_label(a)}$")
            else:
                fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), sharex=True, constrained_layout=True)
                ax11, ax33, ax44, ax66 = axes.ravel()
                ax11.set_title(r"$C_{11}$")
                _common_axis_style(ax11)
                ax11.plot(phi_x, comp["C11"], color="black", lw=2.0)
                ax33.set_title(r"$C_{33}$")
                _common_axis_style(ax33)
                ax33.plot(phi_x, comp["C33"], color="black", lw=2.0)
                ax44.set_title(r"$C_{44}$")
                _common_axis_style(ax44)
                ax44.plot(phi_x, comp["C44"], color="black", lw=2.0)
                ax66.set_title(r"$C_{66}$")
                _common_axis_style(ax66)
                ax66.plot(phi_x, comp["C66"], color="black", lw=2.0)
                for ax in (ax44, ax66):
                    ax.set_xlabel(xlab)
                fig.suptitle(rf"Aligned pores: $\alpha={_alpha_math_label(a)}$")

            out_png = out_dir / f"gsa_elastic_aligned_vti_components_alpha_{_slug_alpha(a)}.png"
            fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
            plt.close(fig)
        return out_png

    # Save the requested single-figure layout (panels by alpha).
    out_png = _save_grid_by_alpha()
    print(f"Saved: {out_png}")
    if args.export_per_alpha:
        for a in aspect_ratios:
            p = _save_per_alpha(float(a))
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
