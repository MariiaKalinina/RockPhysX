from __future__ import annotations

import argparse
import os
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


def _g_tensor_cached(
    *,
    matrix_k: float,
    aspect_ratio: float,
    axis: tuple[float, float, float],
    g_cache: dict[float, np.ndarray],
    n_theta: int | None,
    n_phi: int | None,
    theta_quadrature: str,
) -> np.ndarray:
    a = float(aspect_ratio)
    if a not in g_cache:
        Tc = np.eye(3) * float(matrix_k)
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


def _gsa_two_phase_matrix_comparison_tensor(
    *,
    matrix_k: float,
    inclusion_k: float,
    porosity: float,
    g: np.ndarray,
) -> np.ndarray:
    """
    2-phase tensor GSA for transport with comparison body = matrix (1 iteration closed form).
    """
    phi = float(porosity)
    if not (0.0 <= phi <= 1.0):
        raise ValueError("porosity must be in [0,1]")
    Tm = np.eye(3) * float(matrix_k)
    Ti = np.eye(3) * float(inclusion_k)

    I = np.eye(3)
    d = Ti - Tm
    M_i = np.linalg.inv(I + g @ d)
    A_i = Ti @ M_i

    num = (1.0 - phi) * Tm + phi * A_i
    den = (1.0 - phi) * I + phi * M_i
    return num @ np.linalg.inv(den)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Example-style TI anisotropy sensitivity for thermal conductivity (GSA transport): "
            "isotropic calcite matrix with aligned crack-like pores filled with oil."
        )
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out-png", type=Path, default=None, help="Optional explicit output PNG path.")

    ap.add_argument("--matrix-k", type=float, default=3.59, help="Calcite thermal conductivity (W/m/K).")
    ap.add_argument("--oil-k", type=float, default=0.15, help="Oil thermal conductivity (W/m/K).")

    ap.add_argument("--phi-max", type=float, default=0.30, help="Max porosity for the map (fraction).")
    ap.add_argument("--phi-n", type=int, default=61, help="Porosity grid size for the map.")
    ap.add_argument("--phi-curves", type=str, default="0.05,0.10,0.20", help="Comma-separated porosities for curves.")

    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=81)

    ap.add_argument("--n-theta", type=int, default=None, help="Override theta quadrature (else auto).")
    ap.add_argument("--n-phi-ang", type=int, default=None, help="Override phi quadrature (else auto).")
    ap.add_argument("--theta-quadrature", choices=("uniform", "gauss"), default="uniform")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    matrix_k = float(args.matrix_k)
    oil_k = float(args.oil_k)

    phi_max = float(args.phi_max)
    if not (0.0 < phi_max <= 1.0):
        raise ValueError("--phi-max must be in (0,1].")

    phi_grid = np.linspace(0.0, phi_max, int(args.phi_n))
    phi_curves = [float(x.strip()) for x in str(args.phi_curves).split(",") if x.strip()]
    for p in phi_curves:
        if not (0.0 < p < 1.0):
            raise ValueError("--phi-curves values must be in (0,1).")

    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    if not (0.0 < alpha_min < alpha_max):
        raise ValueError("--alpha-min must be >0 and < --alpha-max.")
    alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), int(args.alpha_n))

    # Aligned cracks: symmetry axis is the crack normal (z). Bedding plane = x-y.
    # So: lambda_bedding = T_xx (=T_yy), lambda_normal = T_zz.
    axis = (0.0, 0.0, 1.0)

    n_theta = args.n_theta if args.n_theta is None else int(args.n_theta)
    n_phi_ang = args.n_phi_ang if args.n_phi_ang is None else int(args.n_phi_ang)
    theta_quad = str(args.theta_quadrature)

    g_cache: dict[float, np.ndarray] = {}
    g_by_alpha = {
        float(a): _g_tensor_cached(
            matrix_k=matrix_k,
            aspect_ratio=float(a),
            axis=axis,
            g_cache=g_cache,
            n_theta=n_theta,
            n_phi=n_phi_ang,
            theta_quadrature=theta_quad,
        )
        for a in alpha
    }

    # Curves: for fixed porosity values, sweep alpha.
    k_normal_curves: dict[float, np.ndarray] = {}
    k_bedding_curves: dict[float, np.ndarray] = {}
    for p in phi_curves:
        kn = np.full_like(alpha, np.nan, dtype=float)
        kb = np.full_like(alpha, np.nan, dtype=float)
        for i, a in enumerate(alpha):
            T = _gsa_two_phase_matrix_comparison_tensor(
                matrix_k=matrix_k,
                inclusion_k=oil_k,
                porosity=float(p),
                g=g_by_alpha[float(a)],
            )
            kn[i] = float(T[2, 2])
            kb[i] = float(0.5 * (T[0, 0] + T[1, 1]))
        k_normal_curves[float(p)] = kn
        k_bedding_curves[float(p)] = kb

    # Map: anisotropy ratio vs (phi, alpha)
    ratio = np.full((phi_grid.size, alpha.size), np.nan, dtype=float)
    for ip, p in enumerate(phi_grid):
        for ia, a in enumerate(alpha):
            T = _gsa_two_phase_matrix_comparison_tensor(
                matrix_k=matrix_k,
                inclusion_k=oil_k,
                porosity=float(p),
                g=g_by_alpha[float(a)],
            )
            kn = float(T[2, 2])
            kb = float(0.5 * (T[0, 0] + T[1, 1]))
            ratio[ip, ia] = kb / kn if kn > 0 else np.nan

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

    out_png = args.out_png
    if out_png is None:
        out_png = out_dir / "gsa_tc_ti_anisotropy_calcite_oil.png"
    out_png = Path(out_png)

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(13.8, 5.6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.0))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # (a) Curves
        colors = ["C0", "C2", "C4", "C6", "C8"]
        for j, p in enumerate(phi_curves):
            c = colors[j % len(colors)]
            ax1.plot(alpha, k_normal_curves[float(p)], color=c, lw=2.6, label=rf"$\lambda_{{normal}},\ \phi={p:.2f}$")
            ax1.plot(alpha, k_bedding_curves[float(p)], color=c, lw=2.2, ls="--", label=rf"$\lambda_{{bedding}},\ \phi={p:.2f}$")

        ax1.set_xscale("log")
        ax1.set_xlabel(r"Aspect ratio $\alpha$")
        ax1.set_ylabel(r"Thermal conductivity (W m$^{-1}$ K$^{-1}$)")
        ax1.set_title(r"(a) TI components for aligned oil cracks")
        ax1.grid(True, which="both", alpha=0.35)
        ax1.legend(ncol=2, frameon=False, loc="lower right")

        # (b) Map
        X, Y = np.meshgrid(alpha, phi_grid, indexing="xy")
        # Choose levels similar density as the example.
        z = ratio
        vmin = np.nanmin(z[np.isfinite(z)])
        vmax = np.nanmax(z[np.isfinite(z)])
        # keep color scale readable if spikes exist
        vmax_plot = float(np.nanpercentile(z[np.isfinite(z)], 99.5)) if np.isfinite(vmax) else 1.0
        vmin_plot = max(1.0, float(vmin)) if np.isfinite(vmin) else 1.0
        levels = np.linspace(vmin_plot, vmax_plot, 16)
        cf = ax2.contourf(X, Y, z, levels=levels, cmap="viridis", extend="max")
        ax2.contour(X, Y, z, levels=levels[::2], colors="k", linewidths=0.6, alpha=0.55)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"Aspect ratio $\alpha$")
        ax2.set_ylabel(r"Porosity $\phi$")
        ax2.set_title(r"(b) Anisotropy ratio $\lambda_{bedding}/\lambda_{normal}$")
        cb = fig.colorbar(cf, ax=ax2, shrink=0.96, pad=0.02)
        cb.set_label(r"$\lambda_{bedding}/\lambda_{normal}$")

        fig.suptitle("TI anisotropy sensitivity study for aligned crack fabric (calcite matrix, oil-filled cracks)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

