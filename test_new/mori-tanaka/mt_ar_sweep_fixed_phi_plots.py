from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

import sys

_ELASTIC_DIR = Path(__file__).resolve().parent / "mori-tanaka"
sys.path.insert(0, str(_ELASTIC_DIR))

from strict_mt_elastic_pores import ElasticPhase, strict_mt_elastic_random_spheroidal_pores  # noqa: E402


@dataclass(frozen=True)
class Config:
    # sweep
    ar_min: float = 1e-4
    ar_max: float = 1.0
    n_ar: int = 301
    phi_list: tuple[float, ...] = (0.05, 0.10, 0.20)

    # elastic [Pa] + densities
    Km_pa: float = 76.8e9
    Gm_pa: float = 32.0e9
    Kf_pa: float = 2.2e9
    rho_m_kg_m3: float = 2710.0
    rho_f_kg_m3: float = 1000.0

    # thermal conductivity [W/(m·K)]
    tc_matrix: float = 2.86
    tc_fluid: float = 0.60

    # electrical conductivity [S/m]
    ec_matrix: float = 1e-4
    ec_fluid: float = 5.0

    dpi: int = 300


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


def compute_curves(cfg: Config) -> dict[str, np.ndarray]:
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar)
    log10_ar = np.log10(a_ratio)
    phi = np.asarray(cfg.phi_list, dtype=float)

    # transport: (n_phi, n_ar)
    tc = np.full((phi.size, a_ratio.size), np.nan, dtype=float)
    ec = np.full((phi.size, a_ratio.size), np.nan, dtype=float)

    for i, ph in enumerate(phi):
        tc[i, :] = mt_transport_single_aspect_ratio(ph, a_ratio, cfg.tc_matrix, cfg.tc_fluid)
        ec[i, :] = mt_transport_single_aspect_ratio(ph, a_ratio, cfg.ec_matrix, cfg.ec_fluid)

    # elastic: (n_phi, n_ar)
    Keff = np.full((phi.size, a_ratio.size), np.nan, dtype=float)
    Geff = np.full((phi.size, a_ratio.size), np.nan, dtype=float)
    Vp = np.full((phi.size, a_ratio.size), np.nan, dtype=float)
    Vs = np.full((phi.size, a_ratio.size), np.nan, dtype=float)

    matrix = ElasticPhase(K=cfg.Km_pa, G=cfg.Gm_pa)
    fluid = ElasticPhase(K=cfg.Kf_pa, G=1e-9)
    for i, ph in enumerate(phi):
        for j, ar in enumerate(a_ratio):
            res = strict_mt_elastic_random_spheroidal_pores(
                phi=float(ph),
                a_ratio=float(ar),
                matrix=matrix,
                inclusion=fluid,
                rho_matrix_kg_m3=cfg.rho_m_kg_m3,
                rho_inclusion_kg_m3=cfg.rho_f_kg_m3,
            )
            Keff[i, j] = res.K_eff / 1e9
            Geff[i, j] = res.G_eff / 1e9
            Vp[i, j] = res.vp_m_s / 1e3
            Vs[i, j] = res.vs_m_s / 1e3

    return {
        "a_ratio": a_ratio,
        "log10_ar": log10_ar,
        "phi": phi,
        "tc": tc,
        "ec": ec,
        "Keff": Keff,
        "Geff": Geff,
        "Vp": Vp,
        "Vs": Vs,
    }


def save_png(curves: dict[str, np.ndarray], cfg: Config, out_dir: Path) -> Path:
    _configure_matplotlib_env(out_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (e.g., `pip install matplotlib`)."
        ) from e

    x = curves["log10_ar"]
    phi = curves["phi"]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True, constrained_layout=True)
    panels = [
        ("Thermal conductivity", "λeff, W/(m·K)", curves["tc"], False),
        ("Electrical conductivity", "σeff, S/m", curves["ec"], True),
        ("Bulk modulus", "Keff, GPa", curves["Keff"], False),
        ("Shear modulus", "Geff, GPa", curves["Geff"], False),
        ("P-wave velocity", "Vp, km/s", curves["Vp"], False),
        ("S-wave velocity", "Vs, km/s", curves["Vs"], False),
    ]

    for ax, (title, ylab, Y, logy) in zip(axes.ravel(), panels, strict=True):
        for i, ph in enumerate(phi):
            ax.plot(x, Y[i, :], lw=2.0, label=f"φ={ph*100:.0f}%")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        if logy and np.nanmin(Y) > 0:
            ax.set_yscale("log")

    for ax in axes[-1, :]:
        ax.set_xlabel("log10(α)")

    # Legend below the plot (outside)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(phi), frameon=True, bbox_to_anchor=(0.5, -0.01))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mt_tc_elastic_ec_vs_log10_ar_fixed_phi.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_phi_list(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --phi list.")
    phi = []
    for p in parts:
        v = float(p)
        if v > 1.0:  # assume user provided percent
            v = v / 100.0
        phi.append(v)
    return tuple(phi)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare TC/EC/elastic vs log10(aspect ratio) for fixed porosity values.",
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "mt_transport_plots")
    p.add_argument("--ar-min", type=float, default=Config.ar_min)
    p.add_argument("--ar-max", type=float, default=Config.ar_max)
    p.add_argument("--n-ar", type=int, default=Config.n_ar)
    p.add_argument("--phi", type=str, default="5,10,20", help="Porosity values (%, or fractions) e.g. 5,10,20")
    p.add_argument("--dpi", type=int, default=Config.dpi)
    args = p.parse_args()

    cfg = Config(
        ar_min=float(args.ar_min),
        ar_max=float(args.ar_max),
        n_ar=int(args.n_ar),
        phi_list=_parse_phi_list(args.phi),
        dpi=int(args.dpi),
    )

    curves = compute_curves(cfg)
    out_path = save_png(curves, cfg, args.out_dir)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
