from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rockphysx.models.emt import gsa_transport


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _tilted_axis(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    tilt = np.deg2rad(float(tilt_deg))
    az = np.deg2rad(float(azimuth_deg))
    v = np.array([np.sin(tilt), 0.0, np.cos(tilt)], dtype=float)
    Rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return Rz @ v


def _g_tensor_cached(
    *,
    matrix_value: float,
    aspect_ratio: float,
    axis: tuple[float, float, float],
    g_cache: dict[float, np.ndarray],
    n_theta: int | None,
    n_phi: int | None,
    theta_quadrature: str,
) -> np.ndarray:
    a = float(aspect_ratio)
    if a not in g_cache:
        Tc = np.eye(3) * float(matrix_value)
        shape = gsa_transport.Shape.sphere() if np.isclose(a, 1.0) else gsa_transport.Shape.spheroid(a)
        g_cache[a] = gsa_transport.g_tensor_transport_numeric(
            Tc,
            shape,
            orientation_axis=axis,
            n_theta=n_theta,
            n_phi=n_phi,
            theta_quadrature=theta_quadrature,  # type: ignore[arg-type]
        )
    return g_cache[a]


def _gsa_two_phase_matrix_comparison_tensor_from_g(
    *,
    matrix_value: float,
    inclusion_value: float,
    porosity: float,
    g: np.ndarray,
) -> np.ndarray:
    """
    Fast 2-phase tensor GSA for transport with comparison body = matrix.

    Uses precomputed Green tensor `g` (depends on Tc=matrix_value, aspect_ratio, axis).
    """
    phi = float(porosity)
    if not (0.0 <= phi <= 1.0):
        raise ValueError("porosity must be in [0,1]")
    Tm = np.eye(3) * float(matrix_value)
    Ti = np.eye(3) * float(inclusion_value)

    I = np.eye(3)
    # matrix phase: delta=0 => M=I, A=Tm
    M_m = I
    A_m = Tm
    # inclusion phase:
    d = Ti - Tm
    M_i = np.linalg.inv(I + g @ d)
    A_i = Ti @ M_i

    num = (1.0 - phi) * A_m + phi * A_i
    den = (1.0 - phi) * M_m + phi * M_i
    return num @ np.linalg.inv(den)


@dataclass(frozen=True)
class Fluid:
    name: str
    k_w_mk: float
    color: str


def _parse_alphas(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("--alphas must contain at least one value")
    for a in out:
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"Invalid alpha {a!r}; aspect ratios must be positive and finite.")
    return out


def _alpha_title(alpha: float) -> str:
    loga = np.log10(float(alpha))
    if np.isclose(loga, round(loga), atol=1e-12):
        return rf"$\alpha = 10^{{{int(round(loga))}}}$"
    return rf"$\alpha = {alpha:g}$"


def _plot_by_alpha(
    *,
    out_png: Path,
    title: str,
    phi: np.ndarray,
    alphas: list[float],
    curves_by_alpha: dict[float, dict[str, dict[str, np.ndarray]]],
    ylabel: str,
    legend_items: list[tuple[str, str, str]],
    ylim: tuple[float | None, float | None],
) -> None:
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

    phi_pct = np.asarray(phi, dtype=float) * 100.0

    n_panels = len(alphas)
    nrows, ncols = 2, 3
    if n_panels > 5:
        raise ValueError("This plot layout supports up to 5 alpha panels (2x3 grid with one legend panel).")

    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 7.2), constrained_layout=True)
        axes = np.asarray(axes)

        for i, alpha in enumerate(alphas):
            r = i // ncols
            c = i % ncols
            ax = axes[r, c]
            ax.set_title(_alpha_title(alpha))
            ax.set_xlabel(r"Porosity, $\phi$ (%)")
            ax.set_ylabel(ylabel)
            if ylim[0] is not None or ylim[1] is not None:
                ax.set_ylim(ylim)

            data = curves_by_alpha[float(alpha)]
            for lab, color, linestyle in legend_items:
                # lab like "Water ⟂" (a unique key), but use stored series keys in data.
                series = data[lab]
                ax.plot(phi_pct, series["y"], color=color, linestyle=linestyle, linewidth=2.2)

        # Legend panel (last subplot)
        leg_ax = axes[1, 2]
        leg_ax.axis("off")
        handles = []
        labels = []
        for lab, color, linestyle in legend_items:
            (h,) = leg_ax.plot([], [], color=color, linestyle=linestyle, linewidth=2.6)
            handles.append(h)
            labels.append(lab)
        leg_ax.legend(handles, labels, loc="center", frameon=False, ncol=1)

        # Hide any unused panels (except legend panel)
        for j in range(n_panels, 5):
            rr = j // ncols
            cc = j % ncols
            axes[rr, cc].axis("off")

        fig.suptitle(title)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Thermal conductivity (GSA transport): curves vs porosity in 5 alpha-panels "
            "for gas/water/oil; isotropic, aligned VTI, and tilted TTI."
        )
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--phi-max", type=float, default=0.30)
    ap.add_argument("--phi-n", type=int, default=151)
    ap.add_argument(
        "--alphas",
        type=str,
        default="1,1e-1,1e-2,1e-3,1e-4",
        help="Comma-separated aspect ratios, e.g. '1,1e-1,1e-2,1e-3,1e-4'.",
    )

    ap.add_argument("--matrix-k", type=float, default=2.85, help="Matrix thermal conductivity (W/m/K).")
    ap.add_argument("--gas-k", type=float, default=0.03, help="Gas thermal conductivity (W/m/K).")
    ap.add_argument("--water-k", type=float, default=0.60, help="Water thermal conductivity (W/m/K).")
    ap.add_argument("--oil-k", type=float, default=0.15, help="Oil thermal conductivity (W/m/K).")

    ap.add_argument("--cases", type=str, default="iso,vti,tti", help="Comma-separated: iso,vti,tti")

    ap.add_argument("--n-theta", type=int, default=None, help="Override theta quadrature nodes (else auto).")
    ap.add_argument("--n-phi-ang", type=int, default=None, help="Override phi quadrature nodes (else auto).")
    ap.add_argument("--theta-quadrature", choices=("uniform", "gauss"), default="uniform")

    ap.add_argument("--tilt-deg", type=float, default=20.0, help="TTI: tilt angle of symmetry axis (deg).")
    ap.add_argument("--tilt-azimuth-deg", type=float, default=0.0, help="TTI: tilt azimuth (deg).")

    ap.add_argument("--ylim-min", type=float, default=None, help="Optional y-axis min.")
    ap.add_argument("--ylim-max", type=float, default=None, help="Optional y-axis max.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    phi_max = float(args.phi_max)
    if not np.isfinite(phi_max) or phi_max <= 0.0 or phi_max > 1.0:
        raise ValueError("--phi-max must be in (0,1].")
    phi = np.linspace(0.0, float(phi_max), int(args.phi_n))

    alphas = _parse_alphas(str(args.alphas))
    if len(alphas) > 5:
        raise ValueError("--alphas: maximum 5 values for this figure layout.")

    matrix_k = float(args.matrix_k)
    fluids = [
        Fluid("Water", float(args.water_k), "C0"),
        Fluid("Gas", float(args.gas_k), "C3"),
        Fluid("Oil", float(args.oil_k), "C2"),
    ]

    n_theta = args.n_theta if args.n_theta is None else int(args.n_theta)
    n_phi_ang = args.n_phi_ang if args.n_phi_ang is None else int(args.n_phi_ang)
    theta_quad = str(args.theta_quadrature)

    ylim = (args.ylim_min, args.ylim_max)

    cases = {c.strip().lower() for c in str(args.cases).split(",") if c.strip()}

    if "iso" in cases:
        curves_by_alpha: dict[float, dict[str, dict[str, np.ndarray]]] = {}
        legend_items: list[tuple[str, str, str]] = []
        for fl in fluids:
            legend_items.append((fl.name, fl.color, "-"))

        for alpha in alphas:
            data: dict[str, dict[str, np.ndarray]] = {}
            for fl in fluids:
                y = np.array(
                    [
                        gsa_transport.two_phase_thermal_isotropic(
                            matrix_k,
                            fl.k_w_mk,
                            float(ph),
                            aspect_ratio=float(alpha),
                            comparison="matrix",
                            max_iter=1,
                        )
                        for ph in phi
                    ],
                    dtype=float,
                )
                data[fl.name] = {"y": y}
            curves_by_alpha[float(alpha)] = data

        _plot_by_alpha(
            out_png=out_dir / "tc_vs_porosity_by_alpha_iso_gas_water_oil.png",
            title="Thermal conductivity (isotropic random GSA): $\\lambda_{eff}(\\phi)$",
            phi=phi,
            alphas=alphas,
            curves_by_alpha=curves_by_alpha,
            ylabel=r"$\lambda_{eff}$ (W/m/K)",
            legend_items=legend_items,
            ylim=ylim,
        )

    if "vti" in cases:
        curves_by_alpha = {}
        legend_items = []
        for fl in fluids:
            legend_items.append((f"{fl.name} $\\lambda_\\parallel$", fl.color, "-"))
            legend_items.append((f"{fl.name} $\\lambda_\\perp$", fl.color, "--"))

        axis = (0.0, 0.0, 1.0)
        g_cache: dict[float, np.ndarray] = {}
        for alpha in alphas:
            g = _g_tensor_cached(
                matrix_value=matrix_k,
                aspect_ratio=float(alpha),
                axis=axis,
                g_cache=g_cache,
                n_theta=n_theta,
                n_phi=n_phi_ang,
                theta_quadrature=theta_quad,
            )
            data = {}
            for fl in fluids:
                t_par = np.full_like(phi, np.nan, dtype=float)
                t_perp = np.full_like(phi, np.nan, dtype=float)
                for i, ph in enumerate(phi):
                    T = _gsa_two_phase_matrix_comparison_tensor_from_g(
                        matrix_value=matrix_k,
                        inclusion_value=fl.k_w_mk,
                        porosity=float(ph),
                        g=g,
                    )
                    t_par[i] = float(T[2, 2])
                    t_perp[i] = float(0.5 * (T[0, 0] + T[1, 1]))
                data[f"{fl.name} $\\lambda_\\parallel$"] = {"y": t_par}
                data[f"{fl.name} $\\lambda_\\perp$"] = {"y": t_perp}
            curves_by_alpha[float(alpha)] = data

        _plot_by_alpha(
            out_png=out_dir / "tc_vs_porosity_by_alpha_vti_gas_water_oil.png",
            title="Thermal conductivity (aligned VTI GSA): components vs porosity",
            phi=phi,
            alphas=alphas,
            curves_by_alpha=curves_by_alpha,
            ylabel=r"$\lambda$ (W/m/K)",
            legend_items=legend_items,
            ylim=ylim,
        )

    if "tti" in cases:
        curves_by_alpha = {}
        legend_items = []
        for fl in fluids:
            legend_items.append((f"{fl.name} $T_{{zz}}$", fl.color, "-"))
            legend_items.append((f"{fl.name} $T_{{xx}}$", fl.color, "--"))

        tilt_axis = _tilted_axis(float(args.tilt_deg), float(args.tilt_azimuth_deg))
        axis = (float(tilt_axis[0]), float(tilt_axis[1]), float(tilt_axis[2]))
        g_cache = {}
        for alpha in alphas:
            g = _g_tensor_cached(
                matrix_value=matrix_k,
                aspect_ratio=float(alpha),
                axis=axis,
                g_cache=g_cache,
                n_theta=n_theta,
                n_phi=n_phi_ang,
                theta_quadrature=theta_quad,
            )
            data = {}
            for fl in fluids:
                t_zz = np.full_like(phi, np.nan, dtype=float)
                t_xx = np.full_like(phi, np.nan, dtype=float)
                for i, ph in enumerate(phi):
                    T = _gsa_two_phase_matrix_comparison_tensor_from_g(
                        matrix_value=matrix_k,
                        inclusion_value=fl.k_w_mk,
                        porosity=float(ph),
                        g=g,
                    )
                    t_zz[i] = float(T[2, 2])
                    t_xx[i] = float(T[0, 0])
                data[f"{fl.name} $T_{{zz}}$"] = {"y": t_zz}
                data[f"{fl.name} $T_{{xx}}$"] = {"y": t_xx}
            curves_by_alpha[float(alpha)] = data

        _plot_by_alpha(
            out_png=out_dir / "tc_vs_porosity_by_alpha_tti_gas_water_oil.png",
            title=f"Thermal conductivity (tilted TTI GSA): tilt={float(args.tilt_deg):g}°, az={float(args.tilt_azimuth_deg):g}°",
            phi=phi,
            alphas=alphas,
            curves_by_alpha=curves_by_alpha,
            ylabel=r"$T$ (W/m/K)",
            legend_items=legend_items,
            ylim=ylim,
        )

    print(f"Saved PNGs to: {out_dir}")


if __name__ == "__main__":
    main()
