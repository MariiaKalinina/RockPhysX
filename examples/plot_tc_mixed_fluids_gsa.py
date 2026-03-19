from __future__ import annotations

"""Reproduce mixed-fluid GSA thermal-conductivity atlas figures using RockPhysX.

Figure 1:
    Effective thermal conductivity λ* for two-component pore-fluid mixtures
    as a function of porosity and saturation at fixed aspect ratios.

Figure 2:
    Relative discrepancy (increase) in λ* for mixed saturation relative to the
    low-conductivity end-member rock at fixed porosities, plotted versus
    log10(aspect ratio) and saturation.

Repo API used:
    rockphysx.models.emt.gsa_transport.two_phase_thermal_isotropic
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.models.emt.gsa_transport import two_phase_thermal_isotropic


@dataclass(frozen=True)
class FluidPair:
    name: str
    ylabel: str
    low_name: str
    high_name: str
    low_tc: float
    high_tc: float

    def mixture_tc(self, sat_high: np.ndarray) -> np.ndarray:
        """Lichtenecker mixing law for a two-component fluid mixture.

        sat_high is the saturation of the higher-conductivity component for the
        pair definition used in the figure labels.
        """
        sat_high = np.asarray(sat_high, dtype=float)
        sat_low = 1.0 - sat_high
        return (self.low_tc ** sat_low) * (self.high_tc ** sat_high)

    # def mixture_tc(self, sat_first: np.ndarray) -> np.ndarray:
    #     """Lichtenecker mixing law for a two-component fluid mixture.

    #     sat_first is the saturation of the first fluid named in the pair label.
    #     Thus:
    #     - Gas-Brine: sat_first = S_g
    #     - Gas-Oil:   sat_first = S_g
    #     - Oil-Brine: sat_first = S_o
    #     """
    #     sat_first = np.asarray(sat_first, dtype=float)
    #     sat_second = 1.0 - sat_first
    #     return (self.low_tc ** sat_first) * (self.high_tc ** sat_second)


PAIRS = [
    FluidPair(
        name="Gas-Brine",
        ylabel="Gas-Brine\nWater Saturation ($S_w$)",
        low_name="gas",
        high_name="brine",
        low_tc=0.025,
        high_tc=0.60,
    ),
    FluidPair(
        name="Gas-Oil",
        ylabel="Gas-Oil\nOil Saturation ($S_o$)",
        low_name="gas",
        high_name="oil",
        low_tc=0.025,
        high_tc=0.13,
    ),
    FluidPair(
        name="Oil-Brine",
        ylabel="Oil-Brine\nWater Saturation ($S_w$)",
        low_name="oil",
        high_name="brine",
        low_tc=0.13,
        high_tc=0.60,
    ),
]


def effective_tc(matrix_tc: float, pore_tc: np.ndarray, porosity: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized helper around RockPhysX two-phase isotropic thermal wrapper."""
    # Broadcast manually because RockPhysX wrapper expects scalars.
    p_b, f_b = np.broadcast_arrays(np.asarray(porosity, dtype=float), np.asarray(pore_tc, dtype=float))
    out = np.empty_like(p_b, dtype=float)
    it = np.nditer([p_b, f_b, out], flags=["multi_index"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for phi, lam_f, dest in it:
        dest[...] = two_phase_thermal_isotropic(
            matrix_value=float(matrix_tc),
            inclusion_value=float(lam_f),
            porosity=float(phi),
            aspect_ratio=float(alpha),
            comparison="matrix",
        )
    return out


def mixed_effective_tc(pair: FluidPair, matrix_tc: float, porosity: np.ndarray, sat_high: np.ndarray, alpha: float) -> np.ndarray:
    pore_tc = pair.mixture_tc(sat_high)
    return effective_tc(matrix_tc, pore_tc, porosity, alpha)


def relative_increase_percent(pair: FluidPair, matrix_tc: float, porosity: float, sat_high: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Relative increase from the low-conductivity end-member rock.

    For gas-brine and gas-oil, the reference is the gas-filled rock.
    For oil-brine, the reference is the oil-filled rock.
    """
    sat_grid, alpha_grid = np.broadcast_arrays(np.asarray(sat_high, dtype=float), np.asarray(alpha, dtype=float))
    lam_mix = np.empty_like(sat_grid, dtype=float)
    lam_ref = np.empty_like(sat_grid, dtype=float)

    for idx in np.ndindex(sat_grid.shape):
        a = float(alpha_grid[idx])
        s = float(sat_grid[idx])
        lam_mix[idx] = mixed_effective_tc(pair, matrix_tc, float(porosity), s, a)
        lam_ref[idx] = effective_tc(matrix_tc, pair.low_tc, float(porosity), a)

    return 100.0 * (lam_mix - lam_ref) / lam_ref

def make_porosity_figure(
    outpath: Path,
    matrix_tc: float,
    alphas: list[float],
    phi_max: float,
    n_phi: int,
    n_sat: int,
) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    phi = np.linspace(0.0, phi_max, n_phi)
    sat = np.linspace(0.0, 1.0, n_sat)
    PHI, SAT = np.meshgrid(phi, sat)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(14, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # --- one common color scale for all panels ---
    vmin = 0.51
    vmax = float(matrix_tc)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # same style as your reference: low = blue, high = red
    cmap = plt.get_cmap("RdYlBu_r")

    # contour labels
    contour_levels = np.array([0.78, 1.06, 1.34, 1.62, 1.89, 2.17, 2.45, 2.72])

    for r, pair in enumerate(PAIRS):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            Z = mixed_effective_tc(pair, matrix_tc, PHI, SAT, alpha)

            ax.contourf(
                PHI,
                SAT,
                Z,
                levels=60,
                cmap=cmap,
                norm=norm,
            )

            cs = ax.contour(
                PHI,
                SAT,
                Z,
                levels=contour_levels,
                colors="k",
                linewidths=1.4,
            )
            ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")

            if r == 0:
                ax.set_title(rf"$\alpha$ = {alpha:g}", fontsize=18, pad=10)
            if c == 0:
                ax.set_ylabel(pair.ylabel, fontsize=15)
            if r == 2:
                ax.set_xlabel("Porosity ($\phi$)", fontsize=15)

            ax.set_xlim(phi.min(), phi.max())
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.10)

    # --- single shared colorbar for all subplots ---
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        location="right",
        shrink=0.95,
        pad=0.03,
    )
    cbar.set_label(
        r"Effective thermal conductivity $\lambda^*$ (W m$^{-1}$ K$^{-1}$)",
        fontsize=13,
    )
    cbar.set_ticks([0.78, 1.06, 1.34, 1.62, 1.89, 2.17, 2.45, 2.72, 2.95])
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

def make_error_figure(
    outpath: Path,
    matrix_tc: float,
    phi_values: list[float],
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
    n_sat: int,
) -> None:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    sat = np.linspace(0.0, 1.0, n_sat)
    ALOG, SAT = np.meshgrid(np.log10(alphas), sat)
    ALPHA = 10.0 ** ALOG

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)

    major_levels = [0, 1, 5, 10, 50, 100, 150, 200]
    minor_levels = np.array([0.5, 2, 3, 4, 7.5, 20, 30, 40, 75, 125, 175])

    for r, pair in enumerate(PAIRS):
        for c, phi in enumerate(phi_values):
            ax = axes[r, c]
            Z = relative_increase_percent(pair, matrix_tc, phi, SAT, ALPHA)
            ax.contour(ALOG, SAT, Z, levels=minor_levels, colors="0.75", linewidths=0.8)
            cs = ax.contour(ALOG, SAT, Z, levels=major_levels, colors="0.15", linewidths=1.8)
            ax.clabel(cs, inline=True, fontsize=10, fmt=lambda v: f"{v:.1f}%")

            if r == 0:
                ax.set_title(rf"$\phi$ = {phi*100:.1f}%", fontsize=18, pad=10)
            if c == 0:
                ax.set_ylabel(pair.ylabel, fontsize=15)
            if r == 2:
                ax.set_xlabel(r"$\log_{10}$(Aspect Ratio $\alpha$)", fontsize=15)

            ax.set_xlim(np.log10(alpha_min), np.log10(alpha_max))
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.12)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(outdir: Path, args: argparse.Namespace) -> None:
    text = f"""Mixed-fluid GSA atlas parameters
===============================
matrix_tc = {args.matrix_tc}
air_tc = {args.air_tc}
oil_tc = {args.oil_tc}
brine_tc = {args.brine_tc}
porosity_alphas = {args.alphas}
phi_max = {args.phi_max}
phi_fixed = {args.error_porosity}
alpha_range = [{args.alpha_min}, {args.alpha_max}]
comparison_body = matrix
mixture_rule = Lichtenecker geometric mixing
"""
    (outdir / "summary.txt").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce mixed-fluid GSA thermal-conductivity figures with RockPhysX.")
    p.add_argument("--outdir", default="results/mixed_fluid_gsa", help="Output directory")
    p.add_argument("--matrix-tc", type=float, default=3.0, help="Matrix thermal conductivity, W m^-1 K^-1")
    p.add_argument("--air-tc", type=float, default=0.025)
    p.add_argument("--oil-tc", type=float, default=0.13)
    p.add_argument("--brine-tc", type=float, default=0.60)
    p.add_argument("--alphas", type=float, nargs=3, default=[0.01, 0.1, 1.0], help="Aspect ratios for the porosity figure")
    p.add_argument("--phi-max", type=float, default=0.25)
    p.add_argument("--n-phi", type=int, default=251)
    p.add_argument("--n-sat", type=int, default=201)
    p.add_argument("--error-porosity", type=float, nargs=3, default=[0.015, 0.105, 0.245], help="Fixed porosities for the discrepancy figure")
    p.add_argument("--alpha-min", type=float, default=0.01)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alpha", type=int, default=241)
    return p.parse_args()


def main() -> None:
    global PAIRS
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Update pair conductivities from CLI to keep one source of truth.
    PAIRS = [
        FluidPair(PAIRS[0].name, PAIRS[0].ylabel, PAIRS[0].low_name, PAIRS[0].high_name, args.air_tc, args.brine_tc),
        FluidPair(PAIRS[1].name, PAIRS[1].ylabel, PAIRS[1].low_name, PAIRS[1].high_name, args.air_tc, args.oil_tc),
        FluidPair(PAIRS[2].name, PAIRS[2].ylabel, PAIRS[2].low_name, PAIRS[2].high_name, args.oil_tc, args.brine_tc),
    ]

    make_porosity_figure(
        outdir / "ch4_carb_mix_porosity_reproduced.png",
        matrix_tc=args.matrix_tc,
        alphas=list(args.alphas),
        phi_max=args.phi_max,
        n_phi=args.n_phi,
        n_sat=args.n_sat,
    )

    make_error_figure(
        outdir / "ch4_carb_mix_error_reproduced.png",
        matrix_tc=args.matrix_tc,
        phi_values=list(args.error_porosity),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_alpha=args.n_alpha,
        n_sat=args.n_sat,
    )

    write_summary(outdir, args)
    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
