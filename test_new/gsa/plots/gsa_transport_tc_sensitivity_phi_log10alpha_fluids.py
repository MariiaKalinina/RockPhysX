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
    # Start from z, tilt by tilt_deg towards +x (about y), then rotate around z by azimuth.
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
    n_theta: int | None,
    n_phi: int | None,
    theta_quadrature: str,
) -> np.ndarray:
    Tc = np.eye(3) * float(matrix_value)
    shape = gsa_transport.Shape.sphere() if np.isclose(aspect_ratio, 1.0) else gsa_transport.Shape.spheroid(aspect_ratio)
    return gsa_transport.g_tensor_transport_numeric(
        Tc,
        shape,
        orientation_axis=axis,
        n_theta=n_theta,
        n_phi=n_phi,
        theta_quadrature=theta_quadrature,  # type: ignore[arg-type]
    )


def _gsa_two_phase_matrix_comparison_tensor(
    *,
    matrix_value: float,
    inclusion_value: float,
    porosity: float,
    aspect_ratio: float,
    axis: tuple[float, float, float],
    g_cache: dict[float, np.ndarray],
    n_theta: int | None,
    n_phi: int | None,
    theta_quadrature: str,
) -> np.ndarray:
    """
    Fast 2-phase tensor GSA for transport with comparison body = matrix.

    Uses cached g-tensor that depends only on (Tc=matrix_value, aspect_ratio, axis).
    """
    phi = float(porosity)
    if not (0.0 <= phi <= 1.0):
        raise ValueError("porosity must be in [0,1]")
    Tm = np.eye(3) * float(matrix_value)
    Ti = np.eye(3) * float(inclusion_value)

    if float(aspect_ratio) not in g_cache:
        g_cache[float(aspect_ratio)] = _g_tensor_cached(
            matrix_value=matrix_value,
            aspect_ratio=float(aspect_ratio),
            axis=axis,
            n_theta=n_theta,
            n_phi=n_phi,
            theta_quadrature=theta_quadrature,
        )
    g = g_cache[float(aspect_ratio)]

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


def _scalar_from_tensor(T: np.ndarray) -> float:
    return float(np.trace(T) / 3.0)


@dataclass(frozen=True)
class Fluid:
    name: str
    k_w_mk: float
    color: str


def _grid_phi_alpha(
    *,
    phi_max: float,
    phi_n: int,
    alpha_min: float,
    alpha_max: float,
    alpha_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, float(phi_max), int(phi_n))
    loga = np.linspace(np.log10(float(alpha_min)), np.log10(float(alpha_max)), int(alpha_n))
    alpha = 10.0 ** loga
    # meshgrid with x=phi, y=loga
    PHI, LOGA = np.meshgrid(phi, loga, indexing="xy")
    ALPHA = 10.0 ** LOGA
    return phi, loga, PHI, ALPHA


def _plot_maps(
    *,
    out_png: Path,
    title: str,
    x_phi: np.ndarray,
    y_loga: np.ndarray,
    panels: list[tuple[str, np.ndarray, str]],
    ylabel: str = r"$\log_{10}(\alpha)$",
    xlabel: str = r"Porosity, $\phi$ (%)",
    vmin: float | None = None,
    vmax: float | None = None,
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
        "legend.fontsize": 10,
    }

    n = len(panels)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.4 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        X = x_phi * 100.0

        for i, (lab, Z, cmap) in enumerate(panels):
            ax = axes[i // ncols][i % ncols]
            im = ax.pcolormesh(X, y_loga, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(lab)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, shrink=0.92)

        # hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(title)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sensitivity maps for thermal conductivity (GSA transport): porosity vs log10(aspect ratio) for gas/water/oil; isotropy, VTI, and TTI.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--phi-max", type=float, default=0.35)
    ap.add_argument("--phi-n", type=int, default=61)
    ap.add_argument(
        "--alpha-min",
        type=float,
        default=1e-3,
        help="Minimum aspect ratio. Note: alpha << 1 requires dense angular quadrature and can be slow.",
    )
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=41)
    ap.add_argument("--matrix-k", type=float, default=2.85, help="Matrix thermal conductivity (W/m/K).")

    ap.add_argument("--gas-k", type=float, default=0.03, help="Gas thermal conductivity (W/m/K).")
    ap.add_argument("--water-k", type=float, default=0.60, help="Water thermal conductivity (W/m/K).")
    ap.add_argument("--oil-k", type=float, default=0.15, help="Oil thermal conductivity (W/m/K).")

    ap.add_argument("--cases", type=str, default="iso,vti,tti", help="Comma-separated: iso,vti,tti")

    ap.add_argument("--n-theta", type=int, default=None, help="Override theta quadrature nodes (else auto).")
    ap.add_argument("--n-phi", type=int, default=None, help="Override phi quadrature nodes (else auto).")
    ap.add_argument("--theta-quadrature", choices=("uniform", "gauss"), default="uniform")

    ap.add_argument("--tilt-deg", type=float, default=20.0, help="TTI: tilt angle of symmetry axis (deg).")
    ap.add_argument("--tilt-azimuth-deg", type=float, default=0.0, help="TTI: tilt azimuth (deg).")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    fluids = [
        Fluid("Gas", float(args.gas_k), "Reds"),
        Fluid("Water", float(args.water_k), "Blues"),
        Fluid("Oil", float(args.oil_k), "Purples"),
    ]

    cases = {c.strip().lower() for c in str(args.cases).split(",") if c.strip()}
    phi, loga, PHI, ALPHA = _grid_phi_alpha(
        phi_max=float(args.phi_max),
        phi_n=int(args.phi_n),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        alpha_n=int(args.alpha_n),
    )

    matrix_k = float(args.matrix_k)
    n_theta = args.n_theta if args.n_theta is None else int(args.n_theta)
    n_phi = args.n_phi if args.n_phi is None else int(args.n_phi)
    theta_quad = str(args.theta_quadrature)

    if "iso" in cases:
        # isotropic: random orientations, scalar output.
        panels = []
        for fl in fluids:
            Z = np.zeros_like(PHI, dtype=float)
            for iy in range(Z.shape[0]):
                for ix in range(Z.shape[1]):
                    Z[iy, ix] = gsa_transport.two_phase_thermal_isotropic(
                        matrix_k,
                        fl.k_w_mk,
                        float(PHI[iy, ix]),
                        aspect_ratio=float(ALPHA[iy, ix]),
                        comparison="matrix",
                        max_iter=1,
                    )
            panels.append((f"{fl.name}: $\\lambda_{{eff}}$ (W/m/K)", Z, fl.color))
        _plot_maps(
            out_png=out_dir / "tc_sensitivity_iso_phi_log10alpha_gas_water_oil.png",
            title="Thermal conductivity (isotropic random GSA): $\\lambda_{eff}$",
            x_phi=phi,
            y_loga=loga,
            panels=panels,
        )

    if "vti" in cases:
        # VTI: aligned symmetry axis = z, return (perp, parallel, ratio).
        panels = []
        g_cache: dict[float, np.ndarray] = {}
        axis = (0.0, 0.0, 1.0)
        for fl in fluids:
            k_perp = np.zeros_like(PHI, dtype=float)
            k_par = np.zeros_like(PHI, dtype=float)
            ratio = np.zeros_like(PHI, dtype=float)
            for iy in range(PHI.shape[0]):
                for ix in range(PHI.shape[1]):
                    T = _gsa_two_phase_matrix_comparison_tensor(
                        matrix_value=matrix_k,
                        inclusion_value=fl.k_w_mk,
                        porosity=float(PHI[iy, ix]),
                        aspect_ratio=float(ALPHA[iy, ix]),
                        axis=axis,
                        g_cache=g_cache,
                        n_theta=n_theta,
                        n_phi=n_phi,
                        theta_quadrature=theta_quad,
                    )
                    k_par[iy, ix] = float(T[2, 2])
                    k_perp[iy, ix] = float(0.5 * (T[0, 0] + T[1, 1]))
                    ratio[iy, ix] = k_par[iy, ix] / k_perp[iy, ix]

            panels.append((f"{fl.name}: $\\lambda_\\perp$ (W/m/K)", k_perp, fl.color))
            panels.append((f"{fl.name}: $\\lambda_\\parallel$ (W/m/K)", k_par, fl.color))
            panels.append((f"{fl.name}: $\\lambda_\\parallel/\\lambda_\\perp$", ratio, "viridis"))

        _plot_maps(
            out_png=out_dir / "tc_sensitivity_vti_phi_log10alpha_gas_water_oil.png",
            title="Thermal conductivity (aligned VTI GSA): components vs $\\phi$ and $\\log_{10}(\\alpha)$",
            x_phi=phi,
            y_loga=loga,
            panels=panels,
        )

    if "tti" in cases:
        # TTI: same aligned pores, but symmetry axis tilted in the lab frame.
        panels = []
        tilt_axis = _tilted_axis(float(args.tilt_deg), float(args.tilt_azimuth_deg))
        axis = (float(tilt_axis[0]), float(tilt_axis[1]), float(tilt_axis[2]))
        g_cache: dict[float, np.ndarray] = {}
        for fl in fluids:
            t_xx = np.zeros_like(PHI, dtype=float)
            t_zz = np.zeros_like(PHI, dtype=float)
            t_xz = np.zeros_like(PHI, dtype=float)
            for iy in range(PHI.shape[0]):
                for ix in range(PHI.shape[1]):
                    T = _gsa_two_phase_matrix_comparison_tensor(
                        matrix_value=matrix_k,
                        inclusion_value=fl.k_w_mk,
                        porosity=float(PHI[iy, ix]),
                        aspect_ratio=float(ALPHA[iy, ix]),
                        axis=axis,
                        g_cache=g_cache,
                        n_theta=n_theta,
                        n_phi=n_phi,
                        theta_quadrature=theta_quad,
                    )
                    t_xx[iy, ix] = float(T[0, 0])
                    t_zz[iy, ix] = float(T[2, 2])
                    t_xz[iy, ix] = float(T[0, 2])

            panels.append((f"{fl.name}: $T_{{xx}}$ (W/m/K)", t_xx, fl.color))
            panels.append((f"{fl.name}: $T_{{zz}}$ (W/m/K)", t_zz, fl.color))
            panels.append((f"{fl.name}: $T_{{xz}}$ (W/m/K)", t_xz, "coolwarm"))

        _plot_maps(
            out_png=out_dir / "tc_sensitivity_tti_phi_log10alpha_gas_water_oil.png",
            title=f"Thermal conductivity (tilted TTI GSA): tilt={float(args.tilt_deg):g}°, az={float(args.tilt_azimuth_deg):g}°",
            x_phi=phi,
            y_loga=loga,
            panels=panels,
        )

    print(f"Saved PNGs to: {out_dir}")


if __name__ == "__main__":
    main()
