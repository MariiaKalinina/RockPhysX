from __future__ import annotations

"""Mixed-fluid GSA thermal-conductivity atlas with temperature-dependent phases.

Outputs:
1) ch4_carb_mix_porosity_reproduced.png
2) ch4_carb_mix_error_reproduced.png
3) ch4_temp_component_functions.png
4) ch4_carb_mix_temperature_effect.png

Notes
-----
- Effective conductivity is computed with RockPhysX isotropic GSA wrapper
  `two_phase_thermal_isotropic(...)` for a matrix + randomly oriented pore system.
- Matrix conductivity is temperature-corrected with the Sekiguchi-Waples-style
  relation printed in the basin-modeling book.
- Fluid conductivities use the book's pore-fluid functions. The book text for
  water (3.14) is internally inconsistent (threshold in degC, but says T in K).
  By default this script uses the physically consistent interpretation:
      water equation with T in degC,
      oil/gas equations with T in K.
  Use --strict-book-water-kelvin to follow the printed text literally.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from rockphysx.models.emt.gsa_transport import two_phase_thermal_isotropic

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "0.15",
    "axes.linewidth": 1.0,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "grid.color": "0.75",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.35,
})

@dataclass(frozen=True)
class FluidPair:
    name: str
    ylabel: str
    low_name: str
    high_name: str
    low_tc: float
    high_tc: float

    def mixture_tc(self, sat_high: np.ndarray) -> np.ndarray:
        sat_high = np.asarray(sat_high, dtype=float)
        sat_low = 1.0 - sat_high
        return (self.low_tc ** sat_low) * (self.high_tc ** sat_high)


PAIRS = [
    FluidPair("Gas-Brine", "Gas-Brine\nWater Saturation ($S_w$)", "gas", "brine", 0.025, 0.60),
    FluidPair("Gas-Oil", "Gas-Oil\nOil Saturation ($S_o$)", "gas", "oil", 0.025, 0.13),
    FluidPair("Oil-Brine", "Oil-Brine\nWater Saturation ($S_w$)", "oil", "brine", 0.13, 0.60),
]


# ------------------------- temperature functions -------------------------

def lambda_matrix_sw(T_c: np.ndarray, lambda20: float) -> np.ndarray:
    """Sekiguchi-Waples-style matrix/mineral temperature correction from Eq. (3.10).

    λ(T) = 358 * (1.0227 λ20 - 1.882) * (1/T - 0.00068) + 1.84
    with λ in W/m/K and T in K.
    """
    T_k = np.asarray(T_c, dtype=float) + 273.15
    return 358.0 * (1.0227 * lambda20 - 1.882) * ((1.0 / T_k) - 0.00068) + 1.84


def lambda_water_314(T_c: np.ndarray, strict_book_water_kelvin: bool = False) -> np.ndarray:
    """Book Eq. (3.14) for water.

    Default: use T in degC, because the printed book text is internally inconsistent
    (threshold given in degC, variable unit stated as K). This choice gives physically
    reasonable room-temperature values. Set strict_book_water_kelvin=True to use the
    printed 'T in Kelvin' text literally.
    """
    T_c = np.asarray(T_c, dtype=float)
    if strict_book_water_kelvin:
        T = T_c + 273.15
    else:
        T = T_c
    out = np.where(
        T_c < 137.0,
        0.565 - 1.88e-3 * T - 7.23e-6 * T**2,
        0.602 - 1.31e-3 * T - 5.14e-6 * T**2,
    )
    return out


def lambda_oil_315(T_c: np.ndarray) -> np.ndarray:
    """Book Eq. (3.15) for liquid petroleum; T in Kelvin."""
    T_c = np.asarray(T_c, dtype=float)
    T_k = T_c + 273.15
    out = np.where(T_c < 240.0, 0.2389 - 4.593e-4 * T_k + 2.676e-7 * T_k**2, 0.075)
    return out


def lambda_gas_315(T_c: np.ndarray) -> np.ndarray:
    """Book Eq. (3.15) for vapor petroleum / gas; T in Kelvin."""
    T_c = np.asarray(T_c, dtype=float)
    T_k = T_c + 273.15
    out = np.where(T_c < 120.0, -0.0969 + 4.37e-4 * T_k, 0.075)
    return out


# ------------------------- GSA helpers -------------------------

def effective_tc_scalar(matrix_tc: float, pore_tc: float, porosity: float, alpha: float) -> float:
    return float(
        two_phase_thermal_isotropic(
            matrix_value=float(matrix_tc),
            inclusion_value=float(pore_tc),
            porosity=float(porosity),
            aspect_ratio=float(alpha),
            comparison="matrix",
        )
    )


def mixed_effective_tc_grid(pair: FluidPair, matrix_tc: float, phi: np.ndarray, sat: np.ndarray, alpha: float) -> np.ndarray:
    Z = np.empty((sat.size, phi.size), dtype=float)
    for i, s in enumerate(sat):
        pore_tc = float(pair.mixture_tc(float(s)))
        for j, p in enumerate(phi):
            Z[i, j] = effective_tc_scalar(matrix_tc, pore_tc, float(p), float(alpha))
    return Z


def mixed_effective_tc_grid_temperature(
    pair_name: str,
    matrix_lambda20: float,
    phi: float,
    sat_high: float,
    alpha: float,
    T_c: np.ndarray,
    strict_book_water_kelvin: bool = False,
) -> np.ndarray:
    T_c = np.asarray(T_c, dtype=float)
    lam_M = lambda_matrix_sw(T_c, matrix_lambda20)
    lam_w = lambda_water_314(T_c, strict_book_water_kelvin=strict_book_water_kelvin)
    lam_o = lambda_oil_315(T_c)
    lam_g = lambda_gas_315(T_c)

    if pair_name == "Gas-Brine":
        lam_mix = (lam_g ** (1.0 - sat_high)) * (lam_w ** sat_high)
    elif pair_name == "Gas-Oil":
        lam_mix = (lam_g ** (1.0 - sat_high)) * (lam_o ** sat_high)
    elif pair_name == "Oil-Brine":
        lam_mix = (lam_o ** (1.0 - sat_high)) * (lam_w ** sat_high)
    else:
        raise ValueError(f"Unknown pair: {pair_name}")

    out = np.empty_like(T_c, dtype=float)
    for i, t in enumerate(T_c):
        out[i] = effective_tc_scalar(float(lam_M[i]), float(lam_mix[i]), float(phi), float(alpha))
    return out


def relative_increase_percent(pair: FluidPair, matrix_tc: float, porosity: float, sat_high: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    sat_grid, alpha_grid = np.broadcast_arrays(np.asarray(sat_high, dtype=float), np.asarray(alpha, dtype=float))
    lam_mix = np.empty_like(sat_grid, dtype=float)
    lam_ref = np.empty_like(sat_grid, dtype=float)

    for idx in np.ndindex(sat_grid.shape):
        a = float(alpha_grid[idx])
        s = float(sat_grid[idx])
        lam_mix[idx] = effective_tc_scalar(matrix_tc, float(pair.mixture_tc(s)), float(porosity), a)
        lam_ref[idx] = effective_tc_scalar(matrix_tc, float(pair.low_tc), float(porosity), a)

    return 100.0 * (lam_mix - lam_ref) / lam_ref


# ------------------------- figures -------------------------

def make_porosity_figure(outpath: Path, matrix_tc: float, alphas: list[float], phi_max: float, n_phi: int, n_sat: int) -> None:
    phi = np.linspace(0.0, phi_max, n_phi)
    sat = np.linspace(0.0, 1.0, n_sat)
    PHI, SAT = np.meshgrid(phi, sat)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    vmin = 0.51
    vmax = float(matrix_tc)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdYlBu_r")
    contour_levels = np.array([0.78, 1.06, 1.34, 1.62, 1.89, 2.17, 2.45, 2.72])

    for r, pair in enumerate(PAIRS):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            Z = mixed_effective_tc_grid(pair, matrix_tc, phi, sat, alpha)
            ax.contourf(PHI, SAT, Z, levels=60, cmap=cmap, norm=norm)
            cs = ax.contour(PHI, SAT, Z, levels=contour_levels, colors="k", linewidths=1.4)
            ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")

            if r == 0:
                ax.set_title(rf"$\alpha$ = {alpha:g}", fontsize=18, pad=10)
            if c == 0:
                ax.set_ylabel(pair.ylabel, fontsize=15)
            if r == 2:
                ax.set_xlabel("Porosity ($\\phi$)", fontsize=15)

            ax.set_xlim(phi.min(), phi.max())
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.10)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.95, pad=0.03)
    cbar.set_label(r"Effective thermal conductivity $\lambda^*$ (W m$^{-1}$ K$^{-1}$)", fontsize=13)
    cbar.set_ticks([0.78, 1.06, 1.34, 1.62, 1.89, 2.17, 2.45, 2.72, 2.95])
    cbar.ax.tick_params(labelsize=11)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_error_figure(outpath: Path, matrix_tc: float, phi_values: list[float], alpha_min: float, alpha_max: float, n_alpha: int, n_sat: int) -> None:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    sat = np.linspace(0.0, 1.0, n_sat)
    ALOG, SAT = np.meshgrid(np.log10(alphas), sat)
    ALPHA = 10.0 ** ALOG

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
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
                ax.set_title(rf"$\phi$ = {phi*100:.1f}\%", fontsize=18, pad=10)
            if c == 0:
                ax.set_ylabel(pair.ylabel, fontsize=15)
            if r == 2:
                ax.set_xlabel(r"$\log_{10}$(Aspect Ratio $\alpha$)", fontsize=15)

            ax.set_xlim(np.log10(alpha_min), np.log10(alpha_max))
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.12)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

def make_component_temperature_figure(
    outpath: Path,
    matrix_lambda20: float,
    t_min_c: float,
    t_max_c: float,
    n_t: int,
    strict_book_water_kelvin: bool = False,
) -> None:
    T_c = np.linspace(t_min_c, t_max_c, n_t)
    lam_M = lambda_matrix_sw(T_c, matrix_lambda20)
    lam_w = lambda_water_314(T_c, strict_book_water_kelvin=strict_book_water_kelvin)
    lam_o = lambda_oil_315(T_c)
    lam_g = lambda_gas_315(T_c)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.6, 7.8),
        sharex=True,
        constrained_layout=True
    )

    red = "#b22222"
    blue = "#1f4e79"
    black = "#111111"

    # -------- upper panel: matrix --------
    ax = axes[0]
    ax.plot(T_c, lam_M, color=black, lw=2.6)

    ax.set_ylabel(r"$\lambda_M(T)$ (W m$^{-1}$ K$^{-1}$)")
    # ax.set_title("Temperature-dependent thermal conductivity of matrix and pore fluids", pad=10)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.15)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ax.text(
    #     0.02, 0.92,
    #     rf"(a) Matrix, $\lambda_{{M,20}}={matrix_lambda20:.2f}$ W m$^{{-1}}$ K$^{{-1}}$",
    #     transform=ax.transAxes,
    #     ha="left", va="top", fontsize=12
    # )

    # -------- lower panel: fluids --------
    ax = axes[1]
    ax.plot(T_c, lam_w, color=blue, lw=2.4, label=r"$\lambda_{brine}(T)$")
    ax.plot(T_c, lam_o, color=red, lw=2.4, label=r"$\lambda_{oil}(T)$")
    ax.plot(T_c, lam_g, color=black, lw=2.4, label=r"$\lambda_{gas}(T)$")

    ax.set_xlabel(r"Temperature ($^\circ$C)")
    ax.set_ylabel(r"$\lambda_f(T)$ (W m$^{-1}$ K$^{-1}$)")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.15)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.legend(loc="upper right", ncol=3, handlelength=2.8)

    # ax.text(
    #     0.02, 0.92,
    #     "(b) Pore-fluid functions",
    #     transform=ax.transAxes,
    #     ha="left", va="top", fontsize=12
    # )

    for ax in axes:
        ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.0)
        ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_temperature_effect_figure(
    outpath: Path,
    matrix_lambda20: float,
    alphas: list[float],
    phi_ref: float,
    sat_levels: list[float],
    t_min_c: float,
    t_max_c: float,
    n_t: int,
    strict_book_water_kelvin: bool = False,
) -> None:
    T_c = np.linspace(t_min_c, t_max_c, n_t)

    fig, axes = plt.subplots(
        3, 3,
        figsize=(13.8, 10.0),
        sharex=True,
        sharey=False,
        constrained_layout=True
    )

    red = "#b22222"
    blue = "#1f4e79"
    black = "#111111"

    # fixed colors for saturation curves
    sat_palette = [black, blue, red]
    sat_alpha = 0.95

    for r, pair in enumerate(PAIRS):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]

            for k, sat in enumerate(sat_levels):
                lam_eff = mixed_effective_tc_grid_temperature(
                    pair.name,
                    matrix_lambda20,
                    phi_ref,
                    sat,
                    alpha,
                    T_c,
                    strict_book_water_kelvin=strict_book_water_kelvin,
                )

                color = sat_palette[k % len(sat_palette)]
                ax.plot(
                    T_c,
                    lam_eff,
                    color=color,
                    lw=2.2,
                    alpha=sat_alpha,
                    label=rf"$S={sat:.1f}$"
                )

            if r == 0:
                ax.set_title(rf"$\alpha = {alpha:g}$", pad=8)

            if c == 0:
                ax.set_ylabel(
                    pair.name + "\n" + r"$\lambda^*(T)$ (W m$^{-1}$ K$^{-1}$)",
                    fontsize=12
                )

            if r == 2:
                ax.set_xlabel(r"Temperature ($^\circ$C)")

            # ticks and grid
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            ax.grid(True, which="major", alpha=0.28)
            ax.grid(True, which="minor", alpha=0.12)

            ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.0)
            ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # # subtle panel label
            # ax.text(
            #     0.03, 0.95,
            #     f"({chr(97 + r*3 + c)})",
            #     transform=ax.transAxes,
            #     ha="left", va="top", fontsize=11
            # )

    # one shared legend
    handles = [
        plt.Line2D([0], [0], color=black, lw=2.2, alpha=0.95, label=rf"$S={sat_levels[0]:.1f}$"),
        plt.Line2D([0], [0], color=blue,  lw=2.2, alpha=0.95, label=rf"$S={sat_levels[1]:.1f}$"),
        plt.Line2D([0], [0], color=red,   lw=2.2, alpha=0.95, label=rf"$S={sat_levels[2]:.1f}$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False
    )

    fig.suptitle(
        rf"Temperature dependence of effective thermal conductivity at fixed porosity $\phi={phi_ref*100:.1f}\%$",
        fontsize=15,
        y=1.04
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_component_relative_change_figure(
    outpath: Path,
    matrix_lambda20: float,
    t_min_c: float,
    t_max_c: float,
    n_t: int,
    strict_book_water_kelvin: bool = False,
) -> None:
    T_c = np.linspace(t_min_c, t_max_c, n_t)

    lam_M = lambda_matrix_sw(T_c, matrix_lambda20)
    lam_w = lambda_water_314(T_c, strict_book_water_kelvin=strict_book_water_kelvin)
    lam_o = lambda_oil_315(T_c)
    lam_g = lambda_gas_315(T_c)

    # reference at 20 C
    lam_M_20 = float(lambda_matrix_sw(np.array([20.0]), matrix_lambda20)[0])
    lam_w_20 = float(lambda_water_314(np.array([20.0]), strict_book_water_kelvin=strict_book_water_kelvin)[0])
    lam_o_20 = float(lambda_oil_315(np.array([20.0]))[0])
    lam_g_20 = float(lambda_gas_315(np.array([20.0]))[0])

    dM = 100.0 * (lam_M - lam_M_20) / lam_M_20
    dw = 100.0 * (lam_w - lam_w_20) / lam_w_20
    do = 100.0 * (lam_o - lam_o_20) / lam_o_20
    dg = 100.0 * (lam_g - lam_g_20) / lam_g_20

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.6, 7.8),
        sharex=True,
        constrained_layout=True
    )

    red = "#b22222"
    blue = "#1f4e79"
    black = "#111111"

    # -------- upper panel: matrix relative change --------
    ax = axes[0]
    ax.plot(T_c, dM, color=black, lw=2.6)
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--", alpha=0.5)

    ax.set_ylabel(r"$\delta\lambda_M(T)$ (%)")
    # ax.set_title(r"Relative change with respect to $20^\circ$C", pad=10)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.15)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ax.text(
    #     0.02, 0.92,
    #     "(a) Matrix",
    #     transform=ax.transAxes,
    #     ha="right", va="top", fontsize=12
    # )

    # -------- lower panel: fluid relative change --------
    ax = axes[1]
    ax.plot(T_c, dw, color=blue, lw=2.4, label=r"$\delta\lambda_{brine}(T)$")
    ax.plot(T_c, do, color=red, lw=2.4, label=r"$\delta\lambda_{oil}(T)$")
    ax.plot(T_c, dg, color=black, lw=2.4, label=r"$\delta\lambda_{gas}(T)$")
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--", alpha=0.5)

    ax.set_xlabel(r"Temperature ($^\circ$C)")
    ax.set_ylabel(r"$\delta\lambda_f(T)$ (%)")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.15)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.legend(loc="upper left", ncol=3, handlelength=2.8)

    # ax.text(
    #     0.02, 0.92,
    #     "(b) Pore fluids",
    #     transform=ax.transAxes,
    #     ha="left", va="top", fontsize=12
    # )

    for ax in axes:
        ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.0)
        ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed-fluid GSA atlas with temperature-dependent matrix and fluid conductivities")
    p.add_argument("--outdir", default="results/mixed_fluid_gsa_temp", help="Output directory")

    # Reference conductivities at 20 C or room-temperature defaults
    p.add_argument("--matrix-tc", type=float, default=3.0, help="Reference matrix conductivity λ20 for Eq. (3.10), W m^-1 K^-1")
    p.add_argument("--air-tc", type=float, default=0.025, help="Reference gas conductivity used in the original isothermal atlas")
    p.add_argument("--oil-tc", type=float, default=0.13, help="Reference oil conductivity used in the original isothermal atlas")
    p.add_argument("--brine-tc", type=float, default=0.60, help="Reference brine/water conductivity used in the original isothermal atlas")

    # Original figures
    p.add_argument("--alphas", type=float, nargs=3, default=[0.01, 0.1, 1.0], help="Aspect ratios for the porosity figure")
    p.add_argument("--phi-max", type=float, default=0.25)
    p.add_argument("--n-phi", type=int, default=121)
    p.add_argument("--n-sat", type=int, default=121)
    p.add_argument("--error-porosity", type=float, nargs=3, default=[0.015, 0.105, 0.245], help="Fixed porosities for the discrepancy figure")
    p.add_argument("--alpha-min", type=float, default=0.01)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alpha", type=int, default=121)

    # Temperature figures
    p.add_argument("--t-min-c", type=float, default=20.0)
    p.add_argument("--t-max-c", type=float, default=120.0)
    p.add_argument("--n-temp", type=int, default=241)
    p.add_argument("--temp-phi", type=float, default=0.105, help="Fixed porosity for the temperature-effect figure")
    p.add_argument("--temp-sats", type=float, nargs="*", default=[0.0, 0.5, 1.0], help="Saturation levels for temperature-effect figure")
    p.add_argument("--strict-book-water-kelvin", action="store_true", help="Use the printed 'T in Kelvin' text literally for Eq. (3.14)")
    return p.parse_args()


def write_summary(outdir: Path, args: argparse.Namespace) -> None:
    text = f"""Mixed-fluid GSA atlas with temperature-dependent phases
=================================================
matrix_lambda20 = {args.matrix_tc}
air_tc_isothermal = {args.air_tc}
oil_tc_isothermal = {args.oil_tc}
brine_tc_isothermal = {args.brine_tc}
porosity_alphas = {args.alphas}
phi_max = {args.phi_max}
phi_fixed_error = {args.error_porosity}
alpha_range = [{args.alpha_min}, {args.alpha_max}]
temperature_range_C = [{args.t_min_c}, {args.t_max_c}]
temperature_phi = {args.temp_phi}
temperature_sats = {args.temp_sats}
strict_book_water_kelvin = {args.strict_book_water_kelvin}
comparison_body = matrix
mixture_rule = Lichtenecker / geometrical average
matrix_temperature_model = Sekiguchi-Waples Eq. (3.10)
fluid_temperature_model = book Eqs. (3.14), (3.15)
"""
    (outdir / "summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    global PAIRS
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    if args.t_max_c > 130.0:
        raise ValueError(
            "This version is limited to 130°C because only the low-temperature water equation (3.14) is used."
        )

    # Update isothermal reference pairs from CLI for the original two figures.
    PAIRS = [
        FluidPair(PAIRS[0].name, PAIRS[0].ylabel, PAIRS[0].low_name, PAIRS[0].high_name, args.air_tc, args.brine_tc),
        FluidPair(PAIRS[1].name, PAIRS[1].ylabel, PAIRS[1].low_name, PAIRS[1].high_name, args.air_tc, args.oil_tc),
        FluidPair(PAIRS[2].name, PAIRS[2].ylabel, PAIRS[2].low_name, PAIRS[2].high_name, args.oil_tc, args.brine_tc),
    ]

    # make_porosity_figure(
    #     outdir / "ch4_carb_mix_porosity_reproduced.png",
    #     matrix_tc=args.matrix_tc,
    #     alphas=list(args.alphas),
    #     phi_max=args.phi_max,
    #     n_phi=args.n_phi,
    #     n_sat=args.n_sat,
    # )

    # make_error_figure(
    #     outdir / "ch4_carb_mix_error_reproduced.png",
    #     matrix_tc=args.matrix_tc,
    #     phi_values=list(args.error_porosity),
    #     alpha_min=args.alpha_min,
    #     alpha_max=args.alpha_max,
    #     n_alpha=args.n_alpha,
    #     n_sat=args.n_sat,
    # )

    make_component_temperature_figure(
        outdir / "ch4_temp_component_functions.png",
        matrix_lambda20=args.matrix_tc,
        t_min_c=args.t_min_c,
        t_max_c=args.t_max_c,
        n_t=args.n_temp,
        strict_book_water_kelvin=args.strict_book_water_kelvin,
    )

    make_temperature_effect_figure(
        outdir / "ch4_carb_mix_temperature_effect.png",
        matrix_lambda20=args.matrix_tc,
        alphas=list(args.alphas),
        phi_ref=args.temp_phi,
        sat_levels=list(args.temp_sats),
        t_min_c=args.t_min_c,
        t_max_c=args.t_max_c,
        n_t=args.n_temp,
        strict_book_water_kelvin=args.strict_book_water_kelvin,
    )

    make_component_relative_change_figure(
    outdir / "ch4_temp_component_relchange.png",
    matrix_lambda20=args.matrix_tc,
    t_min_c=args.t_min_c,
    t_max_c=args.t_max_c,
    n_t=args.n_temp,
    strict_book_water_kelvin=args.strict_book_water_kelvin,
)

    write_summary(outdir, args)
    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
