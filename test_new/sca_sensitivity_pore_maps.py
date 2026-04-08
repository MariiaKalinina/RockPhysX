from __future__ import annotations

"""
Sensitivity plots for the SCA/CPA-style models (porosity vs aspect ratio).

Outputs (PNG)
-------------
1) sca_tc_ec_elastic_maps.*
   3x2 heatmaps: lambda_eff, sigma_eff, K_eff, G_eff, Vp, Vs
   axes: porosity (linear) vs aspect ratio (log scale)

2) sca_tc_ec_elastic_vs_log10_ar_fixed_phi.*
   3x2 line plots vs log10(alpha) for a few fixed porosity values.

Notes
-----
- Transport properties use RockPhysX SCA thermal model (random-inclusion /
  generalized Clausius–Mossotti form) and are also valid for electrical
  conductivity as a scalar transport property.
- Optionally, TC can also be computed with the self-consistent (SC) transport
  closure (Torquato/Bruggeman style) for consistency with EC.
- Elastic properties use an isotropic self-consistent fixed-point iteration
  in stiffness space, where:
    * the pore phase is a randomly oriented spheroid with aspect ratio alpha
    * the solid phase is treated as spherical (alpha=1)
  This gives a practical "Berryman self-consistent / CPA" sensitivity map that
  depends on aspect ratio.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rockphysx.models.emt.sca_thermal import sca_effective_conductivity, sca_sc_effective_conductivity  # noqa: E402

from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores, sca_elastic_pores_unified  # noqa: E402
from rockphysx.models.emt.sca_elastic import oc_budiansky_sc_penny_cracks_from_phi_alpha  # noqa: E402


@dataclass(frozen=True)
class Config:
    # sweep grid
    phi_max: float = 1
    n_phi: int = 101
    ar_min: float = 1e-4
    ar_max: float = 1.0
    n_ar: int = 61

    # matrix + pore phases (elastic; Pa + densities)
    Km_pa: float = 76.8e9
    Gm_pa: float = 32.0e9
    Kf_pa: float = 2.2e9
    rho_m_kg_m3: float = 2710.0
    rho_f_kg_m3: float = 1000.0

    # transport phases
    tc_matrix: float = 2.86
    tc_fluid: float = 0.60
    ec_matrix: float = 1e-6
    ec_fluid: float = 3.5

    # elastic SC iteration
    tol: float = 1e-10
    max_iter: int = 1000

    # fixed-phi line plot
    phi_list: tuple[float, ...] = (0.05, 0.10, 0.20)

    dpi: int = 300


def _elastic_effective_moduli(
    cfg: Config,
    *,
    porosity: float,
    aspect_ratio: float,
    elastic_model: str,
) -> tuple[float, float]:
    model = elastic_model.strip().lower()
    if model == "auto":
        return sca_elastic_pores_unified(
            matrix_bulk_gpa=cfg.Km_pa / 1e9,
            matrix_shear_gpa=cfg.Gm_pa / 1e9,
            porosity=float(porosity),
            aspect_ratio=float(aspect_ratio),
            pore_bulk_gpa=cfg.Kf_pa / 1e9,
            crack_like_threshold=0.0,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            relaxation=0.2,
        )
    if model == "berryman":
        return berryman_self_consistent_spheroidal_pores(
            matrix_bulk_gpa=cfg.Km_pa / 1e9,
            matrix_shear_gpa=cfg.Gm_pa / 1e9,
            porosity=float(porosity),
            pore_bulk_gpa=cfg.Kf_pa / 1e9,
            aspect_ratio=float(aspect_ratio),
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            relaxation=0.2,
        )
    if model == "penny":
        return oc_budiansky_sc_penny_cracks_from_phi_alpha(
            matrix_bulk_gpa=cfg.Km_pa / 1e9,
            matrix_shear_gpa=cfg.Gm_pa / 1e9,
            porosity=float(porosity),
            aspect_ratio=float(aspect_ratio),
            fluid_bulk_gpa=cfg.Kf_pa / 1e9,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
        )
    raise ValueError(f"Unknown elastic model {elastic_model!r}. Use 'berryman' or 'penny'.")


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
    edges = np.concatenate(([x[0] / s], x * s))
    return edges


def compute_maps(cfg: Config, *, elastic_model: str) -> dict[str, np.ndarray]:
    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar)

    # transport maps: (n_ar, n_phi)
    tc = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    ec = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    for i, ar in enumerate(a_ratio):
        for j, ph in enumerate(phi):
            try:
                # TC closure can be either random-inclusion SCA or self-consistent transport.
                if getattr(cfg, "tc_model", "random") == "sc":
                    tc[i, j] = sca_sc_effective_conductivity(
                        cfg.tc_matrix,
                        cfg.tc_fluid,
                        float(ph),
                        aspect_ratio=float(ar),
                    )
                else:
                    tc[i, j] = sca_effective_conductivity(
                        cfg.tc_matrix,
                        cfg.tc_fluid,
                        float(ph),
                        aspect_ratio=float(ar),
                    )
            except ValueError:
                tc[i, j] = np.nan
            try:
                # Electrical uses Torquato self-consistent transport to avoid non-physical spikes
                # for high contrast + crack-like aspect ratios.
                ec[i, j] = sca_sc_effective_conductivity(cfg.ec_matrix, cfg.ec_fluid, float(ph), aspect_ratio=float(ar))
            except ValueError:
                ec[i, j] = np.nan

    # elastic maps
    K_eff_gpa = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    G_eff_gpa = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    vp_km_s = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)
    vs_km_s = np.full((cfg.n_ar, cfg.n_phi), np.nan, dtype=float)

    for i, ar in enumerate(a_ratio):
        warm_start: tuple[float, float] | None = None
        for j, ph in enumerate(phi):
            try:
                if elastic_model.strip().lower() == "berryman":
                    K_gpa, G_gpa = berryman_self_consistent_spheroidal_pores(
                        matrix_bulk_gpa=cfg.Km_pa / 1e9,
                        matrix_shear_gpa=cfg.Gm_pa / 1e9,
                        porosity=float(ph),
                        pore_bulk_gpa=cfg.Kf_pa / 1e9,
                        aspect_ratio=float(ar),
                        tol=cfg.tol,
                        max_iter=cfg.max_iter,
                        relaxation=0.2,
                        initial_guess_gpa=warm_start,
                    )
                else:
                    K_gpa, G_gpa = _elastic_effective_moduli(cfg, porosity=float(ph), aspect_ratio=float(ar), elastic_model=elastic_model)
            except RuntimeError:
                continue

            K_eff_gpa[i, j] = K_gpa
            G_eff_gpa[i, j] = G_gpa
            warm_start = (float(K_gpa), float(G_gpa))
            K_eff = K_gpa * 1e9
            G_eff = G_gpa * 1e9
            rho_eff = (1.0 - float(ph)) * cfg.rho_m_kg_m3 + float(ph) * cfg.rho_f_kg_m3
            vp = np.sqrt(max((K_eff + 4.0 * G_eff / 3.0) / rho_eff, 1e-30))
            vs = np.sqrt(max(G_eff / rho_eff, 1e-30))
            vp_km_s[i, j] = vp / 1e3
            vs_km_s[i, j] = vs / 1e3

    return {
        "phi": phi,
        "a_ratio": a_ratio,
        "tc": tc,
        "ec": ec,
        "K_eff_gpa": K_eff_gpa,
        "G_eff_gpa": G_eff_gpa,
        "vp_km_s": vp_km_s,
        "vs_km_s": vs_km_s,
    }


def _tc_suffix(cfg: Config) -> str:
    return "_tc_sc" if getattr(cfg, "tc_model", "random") == "sc" else ""


def save_maps(maps: dict[str, np.ndarray], cfg: Config, out_dir: Path) -> None:
    _configure_matplotlib_env(out_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    phi = maps["phi"]
    a_ratio = maps["a_ratio"]
    phi_edges = _edges_from_centers_linear(phi)
    phi_edges[0] = max(phi_edges[0], 0.0)
    ar_edges = _edges_from_centers_geom(a_ratio)

    panels = [
        (r"$\lambda_{\mathrm{eff}}$ (W/(m$\cdot$K))", maps["tc"], "viridis", False),
        (r"$\sigma_{\mathrm{eff}}$ (S/m)", maps["ec"], "viridis", True),
        ("Keff (GPa)", maps["K_eff_gpa"], "viridis", False),
        ("Geff (GPa)", maps["G_eff_gpa"], "viridis", False),
        ("Vp (km/s)", maps["vp_km_s"], "plasma", False),
        ("Vs (km/s)", maps["vs_km_s"], "plasma", False),
    ]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 2, figsize=(12.6, 11.0), constrained_layout=True)
        for ax, (title, Z, cmap, logz) in zip(axes.ravel(), panels, strict=True):
            Zp = Z.copy()
            if logz:
                Zp = np.where(Zp > 0, Zp, np.nan)
            m = ax.pcolormesh(phi_edges, ar_edges, Zp, shading="auto", cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("Porosity, φ (fraction)")
            ax.set_ylabel("Aspect ratio, α (log scale)")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            cbar = fig.colorbar(m, ax=ax, pad=0.01)
            if logz:
                cbar.ax.set_yscale("log")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"sca_tc_ec_elastic_maps{_tc_suffix(cfg)}.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)


def save_fixed_phi_curves(maps: dict[str, np.ndarray], cfg: Config, out_dir: Path, *, elastic_model: str) -> None:
    _configure_matplotlib_env(out_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    phi_grid = np.asarray(cfg.phi_list, dtype=float)
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar * 3)
    x = np.log10(a_ratio)

    # transport curves (n_phi, n_ar)
    tc = np.full((phi_grid.size, a_ratio.size), np.nan, dtype=float)
    ec = np.full((phi_grid.size, a_ratio.size), np.nan, dtype=float)
    for i, ph in enumerate(phi_grid):
        for j, ar in enumerate(a_ratio):
            try:
                if getattr(cfg, "tc_model", "random") == "sc":
                    tc[i, j] = sca_sc_effective_conductivity(cfg.tc_matrix, cfg.tc_fluid, float(ph), aspect_ratio=float(ar))
                else:
                    tc[i, j] = sca_effective_conductivity(cfg.tc_matrix, cfg.tc_fluid, float(ph), aspect_ratio=float(ar))
            except ValueError:
                tc[i, j] = np.nan
            try:
                ec[i, j] = sca_sc_effective_conductivity(cfg.ec_matrix, cfg.ec_fluid, float(ph), aspect_ratio=float(ar))
            except ValueError:
                ec[i, j] = np.nan

    # elastic curves
    K = np.full((phi_grid.size, a_ratio.size), np.nan, dtype=float)
    G = np.full((phi_grid.size, a_ratio.size), np.nan, dtype=float)
    Vp = np.full((phi_grid.size, a_ratio.size), np.nan, dtype=float)
    Vs = np.full((phi_grid.size, a_ratio.size), np.nan, dtype=float)
    for i, ph in enumerate(phi_grid):
        warm_start: tuple[float, float] | None = None
        for j, ar in enumerate(a_ratio):
            try:
                if elastic_model.strip().lower() == "berryman":
                    K_gpa, G_gpa = berryman_self_consistent_spheroidal_pores(
                        matrix_bulk_gpa=cfg.Km_pa / 1e9,
                        matrix_shear_gpa=cfg.Gm_pa / 1e9,
                        porosity=float(ph),
                        pore_bulk_gpa=cfg.Kf_pa / 1e9,
                        aspect_ratio=float(ar),
                        tol=cfg.tol,
                        max_iter=cfg.max_iter,
                        relaxation=0.2,
                        initial_guess_gpa=warm_start,
                    )
                else:
                    K_gpa, G_gpa = _elastic_effective_moduli(cfg, porosity=float(ph), aspect_ratio=float(ar), elastic_model=elastic_model)
            except RuntimeError:
                continue
            K[i, j] = K_gpa
            G[i, j] = G_gpa
            warm_start = (float(K_gpa), float(G_gpa))
            K_eff = K_gpa * 1e9
            G_eff = G_gpa * 1e9
            rho_eff = (1.0 - float(ph)) * cfg.rho_m_kg_m3 + float(ph) * cfg.rho_f_kg_m3
            vp = np.sqrt(max((K_eff + 4.0 * G_eff / 3.0) / rho_eff, 1e-30))
            vs = np.sqrt(max(G_eff / rho_eff, 1e-30))
            Vp[i, j] = vp / 1e3
            Vs[i, j] = vs / 1e3

    panels = [
        ("Thermal conductivity", r"$\lambda_{\mathrm{eff}}$, W/(m$\cdot$K)", tc, False),
        ("Electrical conductivity", r"$\sigma_{\mathrm{eff}}$, S/m", ec, True),
        ("Bulk modulus", "Keff, GPa", K, False),
        ("Shear modulus", "Geff, GPa", G, False),
        ("P-wave velocity", "Vp, km/s", Vp, False),
        ("S-wave velocity", "Vs, km/s", Vs, False),
    ]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 2, figsize=(12.6, 11.0), sharex=True, constrained_layout=True)
        for ax, (title, ylab, Y, logy) in zip(axes.ravel(), panels, strict=True):
            for i, ph in enumerate(phi_grid):
                ax.plot(x, Y[i, :], lw=2.0, label=f"φ={ph*100:.0f}%")
            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.25)
            if logy and np.nanmin(Y) > 0:
                ax.set_yscale("log")
        for ax in axes[-1, :]:
            ax.set_xlabel("log10(α)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(phi_grid), frameon=True, bbox_to_anchor=(0.5, -0.01))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"sca_tc_ec_elastic_vs_log10_ar_fixed_phi{_tc_suffix(cfg)}.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)


def save_fixed_alpha_curves(cfg: Config, out_dir: Path, *, alpha_list: tuple[float, ...], elastic_model: str) -> None:
    """
    Plot properties vs porosity (x), with aspect ratio alpha as the *parameter*.

    Each line is a fixed aspect ratio alpha.
    """
    _configure_matplotlib_env(out_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    alpha = np.asarray(alpha_list, dtype=float)

    # transport curves (n_alpha, n_phi)
    tc = np.full((alpha.size, phi.size), np.nan, dtype=float)
    ec = np.full((alpha.size, phi.size), np.nan, dtype=float)
    for i, ar in enumerate(alpha):
        for j, ph in enumerate(phi):
            try:
                if getattr(cfg, "tc_model", "random") == "sc":
                    tc[i, j] = sca_sc_effective_conductivity(cfg.tc_matrix, cfg.tc_fluid, float(ph), aspect_ratio=float(ar))
                else:
                    tc[i, j] = sca_effective_conductivity(cfg.tc_matrix, cfg.tc_fluid, float(ph), aspect_ratio=float(ar))
            except ValueError:
                tc[i, j] = np.nan
            try:
                ec[i, j] = sca_sc_effective_conductivity(cfg.ec_matrix, cfg.ec_fluid, float(ph), aspect_ratio=float(ar))
            except ValueError:
                ec[i, j] = np.nan

    # elastic curves
    K = np.full((alpha.size, phi.size), np.nan, dtype=float)
    G = np.full((alpha.size, phi.size), np.nan, dtype=float)
    Vp = np.full((alpha.size, phi.size), np.nan, dtype=float)
    Vs = np.full((alpha.size, phi.size), np.nan, dtype=float)

    for i, ar in enumerate(alpha):
        warm_start: tuple[float, float] | None = None
        for j, ph in enumerate(phi):
            try:
                if elastic_model.strip().lower() == "berryman":
                    # Spheroidal self-consistent iterations can converge very slowly for crack-like
                    # aspect ratios. For these curves (few points), use a looser tolerance and more
                    # iterations to avoid breaks around convergence-sensitive porosity ranges.
                    elastic_tol = max(float(cfg.tol), 1e-8)
                    elastic_max_iter = max(int(cfg.max_iter), 6000)
                    K_gpa, G_gpa = berryman_self_consistent_spheroidal_pores(
                        matrix_bulk_gpa=cfg.Km_pa / 1e9,
                        matrix_shear_gpa=cfg.Gm_pa / 1e9,
                        porosity=float(ph),
                        pore_bulk_gpa=cfg.Kf_pa / 1e9,
                        aspect_ratio=float(ar),
                        tol=elastic_tol,
                        max_iter=elastic_max_iter,
                        initial_guess_gpa=warm_start,
                    )
                else:
                    K_gpa, G_gpa = _elastic_effective_moduli(cfg, porosity=float(ph), aspect_ratio=float(ar), elastic_model=elastic_model)
            except RuntimeError:
                continue
            K[i, j] = K_gpa
            G[i, j] = G_gpa
            warm_start = (float(K_gpa), float(G_gpa))
            K_eff = K_gpa * 1e9
            G_eff = G_gpa * 1e9
            rho_eff = (1.0 - float(ph)) * cfg.rho_m_kg_m3 + float(ph) * cfg.rho_f_kg_m3
            vp = np.sqrt(max((K_eff + 4.0 * G_eff / 3.0) / rho_eff, 1e-30))
            vs = np.sqrt(max(G_eff / rho_eff, 1e-30))
            Vp[i, j] = vp / 1e3
            Vs[i, j] = vs / 1e3

    panels = [
        ("Thermal conductivity", r"$\lambda_{\mathrm{eff}}$, W/(m$\cdot$K)", tc, False),
        ("Electrical conductivity", r"$\sigma_{\mathrm{eff}}$, S/m", ec, True),
        ("Bulk modulus", "Keff, GPa", K, False),
        ("Shear modulus", "Geff, GPa", G, False),
        ("P-wave velocity", "Vp, km/s", Vp, False),
        ("S-wave velocity", "Vs, km/s", Vs, False),
    ]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 2, figsize=(12.6, 11.0), sharex=True, constrained_layout=True)
        for ax, (title, ylab, Y, logy) in zip(axes.ravel(), panels, strict=True):
            for i, ar in enumerate(alpha):
                if ar in {1e-4, 1e-3, 1e-2, 1e-1, 1.0}:
                    exp = int(np.round(np.log10(ar)))
                    lab = rf"$\alpha=10^{{{exp}}}$"
                else:
                    lab = rf"$\alpha={ar:g}$"
                ax.plot(phi, Y[i, :], lw=2.0, label=lab)
            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.25)
            if logy and np.nanmin(Y) > 0:
                ax.set_yscale("log")
        for ax in axes[-1, :]:
            ax.set_xlabel("Porosity, φ (fraction)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(alpha), frameon=True, bbox_to_anchor=(0.5, -0.01))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"sca_tc_ec_elastic_vs_porosity_fixed_alpha{_tc_suffix(cfg)}.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)


def _parse_phi_list(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --phi list.")
    phi = []
    for p in parts:
        v = float(p)
        if v > 1.0:
            v = v / 100.0
        phi.append(v)
    return tuple(phi)


def _parse_alpha_list(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --alpha list.")
    out = []
    for p in parts:
        out.append(float(p))
    return tuple(out)


def main() -> None:
    p = argparse.ArgumentParser(description="SCA sensitivity maps: porosity vs aspect ratio (thermal/electrical/elastic).")
    p.add_argument("--out-dir", type=Path, default=Path("test_new/sca/plots"))
    p.add_argument("--phi-max", type=float, default=Config.phi_max)
    p.add_argument("--n-phi", type=int, default=Config.n_phi)
    p.add_argument("--ar-min", type=float, default=Config.ar_min)
    p.add_argument("--ar-max", type=float, default=Config.ar_max)
    p.add_argument("--n-ar", type=int, default=Config.n_ar)
    p.add_argument("--phi", type=str, default="5,10,20", help="Fixed-phi list for line plots (%, or fractions)")
    p.add_argument(
        "--alpha-list",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1,1",
        help="Fixed aspect ratios for porosity curves, e.g. 1e-4,1e-3,1e-2,1e-1,1",
    )
    p.add_argument("--max-iter", type=int, default=Config.max_iter)
    p.add_argument("--dpi", type=int, default=Config.dpi)
    p.add_argument("--elastic-model", type=str, default="auto", help="Elastic model: auto | berryman | penny")
    p.add_argument(
        "--tc-model",
        type=str,
        default="random",
        choices=["random", "sc"],
        help="TC closure: random (SCA random-inclusion) or sc (self-consistent transport).",
    )
    args = p.parse_args()

    cfg = Config(
        phi_max=float(args.phi_max),
        n_phi=int(args.n_phi),
        ar_min=float(args.ar_min),
        ar_max=float(args.ar_max),
        n_ar=int(args.n_ar),
        phi_list=_parse_phi_list(args.phi),
        max_iter=int(args.max_iter),
        dpi=int(args.dpi),
    )
    # Attach option without changing the dataclass definition
    object.__setattr__(cfg, "tc_model", str(args.tc_model))

    elastic_model = str(args.elastic_model)
    maps = compute_maps(cfg, elastic_model=elastic_model)
    save_maps(maps, cfg, args.out_dir)
    save_fixed_phi_curves(maps, cfg, args.out_dir, elastic_model=elastic_model)
    save_fixed_alpha_curves(cfg, args.out_dir, alpha_list=_parse_alpha_list(args.alpha_list), elastic_model=elastic_model)
    print(f"Saved plots to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
