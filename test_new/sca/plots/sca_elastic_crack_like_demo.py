from __future__ import annotations

"""
Crack-like elastic self-consistent demo (aspect_ratio << 1).

This script focuses on the physically relevant thin-crack regime where the
O'Connell–Budiansky self-consistent approximation is the preferred elastic SCA.

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
    phi_max: float = 0.35
    n_phi: int = 151
    alpha_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2)

    Km_gpa: float = 76.8
    Gm_gpa: float = 32.0
    rho_m_kg_m3: float = 2710.0
    rho_f_kg_m3: float = 1000.0
    Kf_gpa: float = 2.2  # brine-like


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _vp_vs_from_KG_rho(K_gpa: float, G_gpa: float, rho: float) -> tuple[float, float]:
    K = float(K_gpa) * 1e9
    G = float(G_gpa) * 1e9
    rho = float(rho)
    vp = np.sqrt(max((K + 4.0 * G / 3.0) / rho, 0.0)) / 1e3
    vs = np.sqrt(max(G / rho, 0.0)) / 1e3
    return float(vp), float(vs)


def main() -> None:
    p = argparse.ArgumentParser(description="Elastic SC demo for crack-like pores (alpha << 1).")
    p.add_argument("--out-dir", type=Path, default=Config.out_dir)
    p.add_argument("--phi-max", type=float, default=Config.phi_max)
    p.add_argument("--n-phi", type=int, default=Config.n_phi)
    p.add_argument("--alpha-list", type=str, default="1e-4,1e-3,1e-2")
    p.add_argument("--dpi", type=int, default=Config.dpi)
    args = p.parse_args()

    alpha_list = tuple(float(x.strip()) for x in str(args.alpha_list).split(",") if x.strip())
    cfg = Config(out_dir=Path(args.out_dir), phi_max=float(args.phi_max), n_phi=int(args.n_phi), alpha_list=alpha_list, dpi=int(args.dpi))

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
        "legend.fontsize": 11,
    }
    plt.rcParams.update(rc)

    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)
    (axK, axG), (axVp, axVs) = axes

    for idx, alpha in enumerate(cfg.alpha_list):
        c = colors[idx % len(colors)]

        K_dry = np.full_like(phi, np.nan, dtype=float)
        G_dry = np.full_like(phi, np.nan, dtype=float)
        vp_dry = np.full_like(phi, np.nan, dtype=float)
        vs_dry = np.full_like(phi, np.nan, dtype=float)

        K_wet = np.full_like(phi, np.nan, dtype=float)
        G_wet = np.full_like(phi, np.nan, dtype=float)
        vp_wet = np.full_like(phi, np.nan, dtype=float)
        vs_wet = np.full_like(phi, np.nan, dtype=float)

        for j, ph in enumerate(phi):
            rho_eff = (1.0 - float(ph)) * cfg.rho_m_kg_m3 + float(ph) * cfg.rho_f_kg_m3

            K0, G0 = sca_elastic_crack_like_pores(cfg.Km_gpa, cfg.Gm_gpa, float(ph), aspect_ratio=float(alpha), fluid_bulk_gpa=0.0)
            K_dry[j] = K0
            G_dry[j] = G0
            vp_dry[j], vs_dry[j] = _vp_vs_from_KG_rho(K0, G0, rho_eff)

            K1, G1 = sca_elastic_crack_like_pores(cfg.Km_gpa, cfg.Gm_gpa, float(ph), aspect_ratio=float(alpha), fluid_bulk_gpa=cfg.Kf_gpa)
            K_wet[j] = K1
            G_wet[j] = G1
            vp_wet[j], vs_wet[j] = _vp_vs_from_KG_rho(K1, G1, rho_eff)

        label = rf"$\alpha={alpha:g}$"
        axK.plot(phi, K_dry, color=c, lw=2.0, label=label)
        axK.plot(phi, K_wet, color=c, lw=2.0, ls="--")

        axG.plot(phi, G_dry, color=c, lw=2.0)
        axG.plot(phi, G_wet, color=c, lw=2.0, ls="--")

        axVp.plot(phi, vp_dry, color=c, lw=2.0)
        axVp.plot(phi, vp_wet, color=c, lw=2.0, ls="--")

        axVs.plot(phi, vs_dry, color=c, lw=2.0)
        axVs.plot(phi, vs_wet, color=c, lw=2.0, ls="--")

    axK.set_ylabel(r"$K_{\mathrm{eff}}$ (GPa)")
    axG.set_ylabel(r"$G_{\mathrm{eff}}$ (GPa)")
    axVp.set_ylabel(r"$V_P$ (km/s)")
    axVs.set_ylabel(r"$V_S$ (km/s)")

    axVp.set_xlabel(r"Porosity, $\phi$ (fraction)")
    axVs.set_xlabel(r"Porosity, $\phi$ (fraction)")

    for ax in [axK, axG, axVp, axVs]:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.0, cfg.phi_max)

    # Legend: alpha lines + linestyle meaning (dry vs brine-sat)
    handles, labels = axK.get_legend_handles_labels()
    style_handles = [
        plt.Line2D([], [], color="k", lw=2.0, ls="-", label="Dry cracks"),
        plt.Line2D([], [], color="k", lw=2.0, ls="--", label="Brine-saturated cracks"),
    ]
    fig.legend(
        handles + style_handles,
        labels + ["Dry cracks", "Brine-saturated cracks"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    out_base = cfg.out_dir / "sca_elastic_crack_like_vs_porosity"
    fig.savefig(out_base.with_suffix(".png"), dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
