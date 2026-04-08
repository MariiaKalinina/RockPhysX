from __future__ import annotations

"""
Crack-like elastic SCA maps (porosity vs aspect ratio, alpha << 1).

This script generates heatmaps for the thin-crack elastic self-consistent model
(O'Connell–Budiansky penny-shaped cracks) over a (phi, alpha) grid.

Outputs (PNG + PDF) are saved next to this file by default.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rockphysx.models.emt.sca_elastic import sca_elastic_crack_like_pores  # noqa: E402


@dataclass(frozen=True)
class Config:
    out_dir: Path = Path(__file__).resolve().parent
    dpi: int = 300

    # Grid
    phi_max: float = 0.35
    n_phi: int = 101
    ar_min: float = 1e-4
    ar_max: float = 1e-2
    n_ar: int = 61

    # Matrix + fluid
    Km_gpa: float = 76.8
    Gm_gpa: float = 32.0
    Kf_gpa: float = 2.2
    rho_m_kg_m3: float = 2710.0
    rho_f_kg_m3: float = 1000.0


def _configure_matplotlib_env(out_dir: Path) -> None:
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
    return np.concatenate(([x[0] / s], x * s))


def _vp_vs_from_KG_rho(K_gpa: float, G_gpa: float, rho: float) -> tuple[float, float]:
    K = float(K_gpa) * 1e9
    G = float(G_gpa) * 1e9
    rho = float(rho)
    vp = np.sqrt(max((K + 4.0 * G / 3.0) / rho, 0.0)) / 1e3
    vs = np.sqrt(max(G / rho, 0.0)) / 1e3
    return float(vp), float(vs)


def compute_maps(cfg: Config) -> dict[str, np.ndarray]:
    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar)

    # Arrays are (n_ar, n_phi)
    out: dict[str, np.ndarray] = {"phi": phi, "a_ratio": a_ratio}
    for state in ["dry", "brine"]:
        out[f"K_{state}_gpa"] = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
        out[f"G_{state}_gpa"] = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
        out[f"vp_{state}_km_s"] = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
        out[f"vs_{state}_km_s"] = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)

    for i, ar in enumerate(a_ratio):
        for j, ph in enumerate(phi):
            rho_eff = (1.0 - float(ph)) * cfg.rho_m_kg_m3 + float(ph) * cfg.rho_f_kg_m3

            # Dry cracks
            try:
                K0, G0 = sca_elastic_crack_like_pores(
                    cfg.Km_gpa,
                    cfg.Gm_gpa,
                    float(ph),
                    aspect_ratio=float(ar),
                    fluid_bulk_gpa=0.0,
                )
                out["K_dry_gpa"][i, j] = K0
                out["G_dry_gpa"][i, j] = G0
                out["vp_dry_km_s"][i, j], out["vs_dry_km_s"][i, j] = _vp_vs_from_KG_rho(K0, G0, rho_eff)
            except Exception:
                pass

            # Brine-saturated cracks
            try:
                K1, G1 = sca_elastic_crack_like_pores(
                    cfg.Km_gpa,
                    cfg.Gm_gpa,
                    float(ph),
                    aspect_ratio=float(ar),
                    fluid_bulk_gpa=cfg.Kf_gpa,
                )
                out["K_brine_gpa"][i, j] = K1
                out["G_brine_gpa"][i, j] = G1
                out["vp_brine_km_s"][i, j], out["vs_brine_km_s"][i, j] = _vp_vs_from_KG_rho(K1, G1, rho_eff)
            except Exception:
                pass

    return out


def save_maps(maps: dict[str, np.ndarray], cfg: Config) -> None:
    _configure_matplotlib_env(cfg.out_dir)
    import matplotlib.pyplot as plt  # noqa: E402

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
    plt.rcParams.update(rc)

    phi = maps["phi"]
    a_ratio = maps["a_ratio"]
    phi_edges = _edges_from_centers_linear(phi)
    phi_edges[0] = max(phi_edges[0], 0.0)
    phi_edges[-1] = min(phi_edges[-1], cfg.phi_max)
    ar_edges = _edges_from_centers_geom(a_ratio)

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.8), sharex=True, sharey=True)

    panels = [
        ("K_dry_gpa", r"$K_{\mathrm{eff}}$ (GPa)"),
        ("G_dry_gpa", r"$G_{\mathrm{eff}}$ (GPa)"),
        ("vp_dry_km_s", r"$V_P$ (km/s)"),
        ("vs_dry_km_s", r"$V_S$ (km/s)"),
        ("K_brine_gpa", r"$K_{\mathrm{eff}}$ (GPa)"),
        ("G_brine_gpa", r"$G_{\mathrm{eff}}$ (GPa)"),
        ("vp_brine_km_s", r"$V_P$ (km/s)"),
        ("vs_brine_km_s", r"$V_S$ (km/s)"),
    ]

    # Column-wise color scales by property (share between dry/brine row).
    prop_keys = ["K", "G", "vp", "vs"]
    vmins: dict[str, float] = {}
    vmaxs: dict[str, float] = {}
    for prop in prop_keys:
        vals = []
        for state in ["dry", "brine"]:
            arr = maps[f"{prop}_{state}_km_s"] if prop in {"vp", "vs"} else maps[f"{prop}_{state}_gpa"]
            vals.append(arr[np.isfinite(arr)])
        allv = np.concatenate(vals) if any(v.size for v in vals) else np.array([])
        if allv.size == 0:
            vmins[prop], vmaxs[prop] = 0.0, 1.0
        else:
            vmins[prop], vmaxs[prop] = float(np.nanmin(allv)), float(np.nanmax(allv))

    cm = "viridis"
    for idx, (key, title) in enumerate(panels):
        r = 0 if idx < 4 else 1
        c = idx % 4
        ax = axes[r, c]
        prop = prop_keys[c]
        Z = maps[key]
        im = ax.pcolormesh(
            phi_edges,
            ar_edges,
            Z,
            shading="auto",
            cmap=cm,
            vmin=vmins[prop],
            vmax=vmaxs[prop],
        )
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.15)
        if r == 0:
            ax.set_title(title)
        if c == 0:
            ax.set_ylabel(r"Aspect ratio, $\alpha$")
        if r == 1:
            ax.set_xlabel(r"Porosity, $\phi$ (fraction)")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=10)

    # Row labels on the left margin
    fig.text(0.005, 0.73, "Dry cracks", rotation=90, va="center", ha="left")
    fig.text(0.005, 0.27, "Brine-saturated cracks", rotation=90, va="center", ha="left")

    fig.tight_layout(rect=(0.02, 0.02, 1.0, 1.0))
    out_base = cfg.out_dir / "sca_elastic_crack_like_maps"
    fig.savefig(out_base.with_suffix(".png"), dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base.with_suffix('.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Crack-like elastic SCA maps (phi vs alpha, alpha << 1).")
    p.add_argument("--out-dir", type=Path, default=Config.out_dir)
    p.add_argument("--phi-max", type=float, default=Config.phi_max)
    p.add_argument("--n-phi", type=int, default=Config.n_phi)
    p.add_argument("--ar-min", type=float, default=Config.ar_min)
    p.add_argument("--ar-max", type=float, default=Config.ar_max)
    p.add_argument("--n-ar", type=int, default=Config.n_ar)
    p.add_argument("--Km-gpa", type=float, default=Config.Km_gpa)
    p.add_argument("--Gm-gpa", type=float, default=Config.Gm_gpa)
    p.add_argument("--Kf-gpa", type=float, default=Config.Kf_gpa)
    p.add_argument("--rho-m", type=float, default=Config.rho_m_kg_m3)
    p.add_argument("--rho-f", type=float, default=Config.rho_f_kg_m3)
    p.add_argument("--dpi", type=int, default=Config.dpi)
    args = p.parse_args()

    cfg = Config(
        out_dir=Path(args.out_dir),
        phi_max=float(args.phi_max),
        n_phi=int(args.n_phi),
        ar_min=float(args.ar_min),
        ar_max=float(args.ar_max),
        n_ar=int(args.n_ar),
        Km_gpa=float(args.Km_gpa),
        Gm_gpa=float(args.Gm_gpa),
        Kf_gpa=float(args.Kf_gpa),
        rho_m_kg_m3=float(args.rho_m),
        rho_f_kg_m3=float(args.rho_f),
        dpi=int(args.dpi),
    )

    maps = compute_maps(cfg)
    save_maps(maps, cfg)


if __name__ == "__main__":
    main()
