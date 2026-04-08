from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlotConfig:
    phi_max: float = 0.35
    n_phi: int = 41
    ar_min: float = 1e-4
    ar_max: float = 1.0
    n_ar: int = 61
    dpi: int = 300

    # Thermal conductivity [W/(m·K)]
    tc_matrix: float = 2.86
    tc_fluid: float = 0.60

    # Electrical conductivity [S/m]
    ec_matrix: float = 1e-4
    ec_fluid: float = 5.0


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def depolarization_factors_spheroid(aspect_ratio: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(aspect_ratio, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("Aspect ratio must be positive and finite.")

    n3 = np.empty_like(r)
    sphere = np.isclose(r, 1.0)
    oblate = r < 1.0
    prolate = r > 1.0

    n3[sphere] = 1.0 / 3.0

    rr = r[oblate]
    if rr.size > 0:
        xi = np.sqrt(np.maximum(1.0 / (rr * rr) - 1.0, 0.0))
        n3[oblate] = ((1.0 + xi * xi) / (xi**3)) * (xi - np.arctan(xi))

    rr = r[prolate]
    if rr.size > 0:
        e = np.sqrt(np.maximum(1.0 - 1.0 / (rr * rr), 0.0))
        n3[prolate] = ((1.0 - e * e) / (2.0 * e**3)) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)

    n1 = (1.0 - n3) / 2.0
    return n1, n3


def mt_transport_single_aspect_ratio(
    phi: float,
    aspect_ratio: float | np.ndarray,
    prop_matrix: float,
    prop_fluid: float,
) -> float | np.ndarray:
    """
    Mori–Tanaka transport (scalar property) for isotropic matrix with randomly oriented spheroidal inclusions.

    This matches the structure you used for thermal conductivity:
      - compute spheroid depolarization factors (n1, n3)
      - compute directional concentration factors a1, a3
      - orientation-average: a_bar = (2 a1 + a3)/3
      - Mori–Tanaka update for scalar transport
    """
    phi = float(phi)
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"Porosity must be in [0,1). Got {phi}")

    r = np.asarray(aspect_ratio, dtype=float)
    n1, n3 = depolarization_factors_spheroid(r)

    dm = float(prop_matrix)
    df = float(prop_fluid)
    delta = df - dm

    a1 = dm / (dm + n1 * delta)
    a3 = dm / (dm + n3 * delta)
    a_bar = (2.0 * a1 + a3) / 3.0

    c_i = phi
    c_m = 1.0 - phi
    value = dm + c_i * delta * a_bar / (c_m + c_i * a_bar)

    if np.ndim(value) == 0:
        return float(value)
    return value


def compute_parametric_curves(cfg: PlotConfig) -> dict[str, np.ndarray]:
    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar)

    tc_eff = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    ec_eff = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)

    for i, ar in enumerate(a_ratio):
        for j, ph in enumerate(phi):
            tc_eff[i, j] = float(
                mt_transport_single_aspect_ratio(
                    phi=float(ph),
                    aspect_ratio=float(ar),
                    prop_matrix=cfg.tc_matrix,
                    prop_fluid=cfg.tc_fluid,
                )
            )
            ec_eff[i, j] = float(
                mt_transport_single_aspect_ratio(
                    phi=float(ph),
                    aspect_ratio=float(ar),
                    prop_matrix=cfg.ec_matrix,
                    prop_fluid=cfg.ec_fluid,
                )
            )

    return {"phi": phi, "a_ratio": a_ratio, "tc_eff": tc_eff, "ec_eff": ec_eff}


def save_parametric_png(maps: dict[str, np.ndarray], cfg: PlotConfig, out_dir: Path) -> Path:
    _configure_matplotlib_env(out_dir)
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (e.g., `pip install matplotlib`)."
        ) from e

    phi = maps["phi"]
    a_ratio = maps["a_ratio"]
    log10_ar = np.log10(a_ratio)

    norm = mpl.colors.Normalize(vmin=float(log10_ar.min()), vmax=float(log10_ar.max()))
    cmap = mpl.colormaps["turbo"]
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), sharex=True, constrained_layout=True)

    for i, lg in enumerate(log10_ar):
        color = cmap(norm(float(lg)))
        axes[0].plot(phi, maps["tc_eff"][i, :], color=color, lw=1.0, alpha=0.85)
        axes[1].plot(phi, maps["ec_eff"][i, :], color=color, lw=1.0, alpha=0.85)

    axes[0].set_title("Thermal conductivity (Mori–Tanaka, spheroidal pores)")
    axes[0].set_ylabel("λeff, W/(m·K)")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Electrical conductivity (Mori–Tanaka, spheroidal pores)")
    axes[1].set_ylabel("σeff, S/m")
    axes[1].set_xlabel("Porosity, φ (fraction)")
    axes[1].grid(True, alpha=0.25)

    # Electrical conductivity typically spans orders of magnitude.
    if np.nanmin(maps["ec_eff"]) > 0:
        axes[1].set_yscale("log")

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01)
    cbar.set_label("log10(α)")

    subtitle = (
        f"λm={cfg.tc_matrix:g}, λf={cfg.tc_fluid:g}  |  "
        f"σm={cfg.ec_matrix:g}, σf={cfg.ec_fluid:g}"
    )
    fig.suptitle(subtitle, y=1.01, fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mt_transport_phi_parametric.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Parametric transport plots vs porosity with aspect ratio as parameter.")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "mt_transport_plots")
    p.add_argument("--phi-max", type=float, default=PlotConfig.phi_max)
    p.add_argument("--n-phi", type=int, default=PlotConfig.n_phi)
    p.add_argument("--ar-min", type=float, default=PlotConfig.ar_min)
    p.add_argument("--ar-max", type=float, default=PlotConfig.ar_max)
    p.add_argument("--n-ar", type=int, default=PlotConfig.n_ar)
    p.add_argument("--dpi", type=int, default=PlotConfig.dpi)

    p.add_argument("--tc-matrix", type=float, default=PlotConfig.tc_matrix, help="Matrix thermal conductivity.")
    p.add_argument("--tc-fluid", type=float, default=PlotConfig.tc_fluid, help="Fluid thermal conductivity.")
    p.add_argument("--ec-matrix", type=float, default=PlotConfig.ec_matrix, help="Matrix electrical conductivity.")
    p.add_argument("--ec-fluid", type=float, default=PlotConfig.ec_fluid, help="Fluid electrical conductivity.")
    args = p.parse_args()

    cfg = PlotConfig(
        phi_max=float(args.phi_max),
        n_phi=int(args.n_phi),
        ar_min=float(args.ar_min),
        ar_max=float(args.ar_max),
        n_ar=int(args.n_ar),
        dpi=int(args.dpi),
        tc_matrix=float(args.tc_matrix),
        tc_fluid=float(args.tc_fluid),
        ec_matrix=float(args.ec_matrix),
        ec_fluid=float(args.ec_fluid),
    )

    maps = compute_parametric_curves(cfg)
    out_path = save_parametric_png(maps, cfg, args.out_dir)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

