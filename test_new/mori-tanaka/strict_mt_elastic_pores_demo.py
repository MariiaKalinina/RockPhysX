from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import argparse
import numpy as np

from strict_mt_elastic_pores import ElasticPhase, strict_mt_elastic_random_spheroidal_pores


@dataclass(frozen=True)
class DemoConfig:
    # matrix + fluid phases
    Km_pa: float = 76.8e9
    Gm_pa: float = 32.0e9
    Kf_pa: float = 2.2e9
    rho_m_kg_m3: float = 2710.0
    rho_f_kg_m3: float = 1000.0

    # sweep grid
    phi_max: float = 0.35
    n_phi: int = 41
    ar_min: float = 1e-4
    ar_max: float = 1.0
    n_ar: int = 61

    # output
    dpi: int = 300


def _configure_matplotlib_env(out_dir: Path) -> None:
    # In sandboxed / read-only home environments, Matplotlib + fontconfig caches
    # may not be writable. Force local writable cache/config directories.
    import os

    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _edges_from_centers_linear(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D array with at least 2 points.")
    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        raise ValueError("x must be evenly spaced for linear edges.")
    d = float(dx[0])
    edges = np.concatenate(([x[0] - d / 2.0], x + d / 2.0))
    return edges


def _edges_from_centers_geom(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D array with at least 2 points.")
    r = x[1] / x[0]
    if not np.allclose(x[1:] / x[:-1], r):
        raise ValueError("x must be geometrically spaced for geometric edges.")
    s = float(np.sqrt(r))
    edges = np.concatenate(([x[0] / s], x * s))
    return edges


def compute_property_maps(cfg: DemoConfig) -> dict[str, np.ndarray]:
    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar)

    matrix = ElasticPhase(K=cfg.Km_pa, G=cfg.Gm_pa)
    fluid = ElasticPhase(K=cfg.Kf_pa, G=1e-9)

    K_eff_gpa = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    G_eff_gpa = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    vp_km_s = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    vs_km_s = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)

    for i, ar in enumerate(a_ratio):
        for j, ph in enumerate(phi):
            res = strict_mt_elastic_random_spheroidal_pores(
                phi=float(ph),
                a_ratio=float(ar),
                matrix=matrix,
                inclusion=fluid,
                rho_matrix_kg_m3=cfg.rho_m_kg_m3,
                rho_inclusion_kg_m3=cfg.rho_f_kg_m3,
            )
            K_eff_gpa[i, j] = res.K_eff / 1e9
            G_eff_gpa[i, j] = res.G_eff / 1e9
            vp_km_s[i, j] = res.vp_m_s / 1e3
            vs_km_s[i, j] = res.vs_m_s / 1e3

    return {
        "phi": phi,
        "a_ratio": a_ratio,
        "K_eff_gpa": K_eff_gpa,
        "G_eff_gpa": G_eff_gpa,
        "vp_km_s": vp_km_s,
        "vs_km_s": vs_km_s,
    }


def save_property_maps_png(maps: dict[str, np.ndarray], cfg: DemoConfig, out_dir: Path) -> Path:
    _configure_matplotlib_env(out_dir)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (e.g., `pip install matplotlib`)."
        ) from e

    phi = maps["phi"]
    a_ratio = maps["a_ratio"]

    phi_edges = _edges_from_centers_linear(phi)
    phi_edges[0] = max(phi_edges[0], 0.0)
    ar_edges = _edges_from_centers_geom(a_ratio)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    panels = [
        ("Keff (GPa)", maps["K_eff_gpa"], "viridis"),
        ("Geff (GPa)", maps["G_eff_gpa"], "viridis"),
        ("Vp (km/s)", maps["vp_km_s"], "plasma"),
        ("Vs (km/s)", maps["vs_km_s"], "plasma"),
    ]

    for ax, (title, Z, cmap) in zip(axes.ravel(), panels, strict=True):
        m = ax.pcolormesh(phi_edges, ar_edges, Z, shading="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Porosity, φ (fraction)")
        ax.set_ylabel("Aspect ratio, α (log scale)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        cbar = fig.colorbar(m, ax=ax, pad=0.01)
        cbar.ax.tick_params(labelsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mt_elastic_pores_maps.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_phi_parametric_png(maps: dict[str, np.ndarray], cfg: DemoConfig, out_dir: Path) -> Path:
    """
    Plot properties vs porosity (x), with aspect ratio as the *parameter*.

    Each line is a fixed aspect ratio α, colored by log10(α).
    """
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

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    panels = [
        ("Keff (GPa)", maps["K_eff_gpa"]),
        ("Geff (GPa)", maps["G_eff_gpa"]),
        ("Vp (km/s)", maps["vp_km_s"]),
        ("Vs (km/s)", maps["vs_km_s"]),
    ]

    for ax, (title, Z) in zip(axes.ravel(), panels, strict=True):
        for i, lg in enumerate(log10_ar):
            ax.plot(phi, Z[i, :], color=cmap(norm(float(lg))), lw=1.0, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Porosity, φ (fraction)")
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("Keff (GPa)")
    axes[0, 1].set_ylabel("Geff (GPa)")
    axes[1, 0].set_ylabel("Vp (km/s)")
    axes[1, 1].set_ylabel("Vs (km/s)")

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01)
    cbar.set_label("log10(α)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mt_elastic_pores_phi_parametric.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: elastic Mori–Tanaka for fluid-filled spheroidal pores (random orientation).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="Directory to save PNG plots.",
    )
    parser.add_argument("--phi-max", type=float, default=DemoConfig.phi_max)
    parser.add_argument("--n-phi", type=int, default=DemoConfig.n_phi)
    parser.add_argument("--ar-min", type=float, default=DemoConfig.ar_min)
    parser.add_argument("--ar-max", type=float, default=DemoConfig.ar_max)
    parser.add_argument("--n-ar", type=int, default=DemoConfig.n_ar)
    parser.add_argument("--dpi", type=int, default=DemoConfig.dpi)
    args = parser.parse_args()

    cfg = DemoConfig(
        phi_max=float(args.phi_max),
        n_phi=int(args.n_phi),
        ar_min=float(args.ar_min),
        ar_max=float(args.ar_max),
        n_ar=int(args.n_ar),
        dpi=int(args.dpi),
    )

    matrix = ElasticPhase(K=cfg.Km_pa, G=cfg.Gm_pa)
    fluid = ElasticPhase(K=cfg.Kf_pa, G=1e-9)

    print("Single-phi sweep (phi=0.15) for selected aspect ratios:")
    for ar in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        res = strict_mt_elastic_random_spheroidal_pores(
            phi=0.15,
            a_ratio=ar,
            matrix=matrix,
            inclusion=fluid,
            rho_matrix_kg_m3=cfg.rho_m_kg_m3,
            rho_inclusion_kg_m3=cfg.rho_f_kg_m3,
        )
        print(
            f"  AR={ar:.3f}  "
            f"Keff={res.K_eff/1e9:.3f} GPa  "
            f"Geff={res.G_eff/1e9:.3f} GPa  "
            f"Vp={res.vp_m_s:.1f} m/s  "
            f"Vs={res.vs_m_s:.1f} m/s"
        )

    maps = compute_property_maps(cfg)
    out_path_maps = save_property_maps_png(maps, cfg, args.out_dir)
    out_path_parametric = save_phi_parametric_png(maps, cfg, args.out_dir)
    print(f"\nSaved: {out_path_maps}")
    print(f"Saved: {out_path_parametric}")


if __name__ == "__main__":
    main()
