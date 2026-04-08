from __future__ import annotations

"""Mixed-fluid GSA thermal-conductivity atlases with temperature dependence.

What is generated
-----------------
1) Isothermal reference figures at 20 C:
   - effective TC atlas versus porosity/saturation for several aspect ratios
   - relative discrepancy atlas versus log10(alpha)/saturation for fixed porosities

2) Temperature-dependent component functions:
   - lambda_M(T) by Sekiguchi-Waples Eq. (3.10)
   - lambda_w(T) by low-temperature branch of Eq. (3.14), limited to 130 C
   - lambda_o(T), lambda_g(T) by Eq. (3.15)

3) Temperature-aware atlases at selected temperatures:
   - effective TC atlas at each temperature
   - relative change atlas: 100*(lambda*(T)-lambda*(20C))/lambda*(20C)
     for fixed porosities, versus log10(alpha) and saturation

The EMT/GSA forward model is the isotropic two-phase matrix + random-oriented pore model
implemented in RockPhysX via two_phase_thermal_isotropic(...).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator

from rockphysx.models.emt.gsa_transport import two_phase_thermal_isotropic


plt.rcParams["contour.negative_linestyle"] = "solid"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


@dataclass(frozen=True)
class FluidPair:
    name: str
    ylabel: str
    low_name: str
    high_name: str
    low_tc_20: float
    high_tc_20: float

    def mixture_tc(self, sat_high: np.ndarray, t_c: float) -> np.ndarray:
        """Lichtenecker geometric mixing for a two-component pore-fluid mixture.

        sat_high is the saturation of the higher-conductivity component for the pair label:
          - Gas-Brine: sat_high = S_w (brine saturation)
          - Gas-Oil:   sat_high = S_o (oil saturation)
          - Oil-Brine: sat_high = S_w (brine saturation)
        """
        sat_high = np.asarray(sat_high, dtype=float)
        sat_low = 1.0 - sat_high
        lam_low = fluid_tc_by_name(self.low_name, t_c)
        lam_high = fluid_tc_by_name(self.high_name, t_c)
        return (lam_low ** sat_low) * (lam_high ** sat_high)


PAIRS = [
    FluidPair("Gas-Brine", "Gas-Brine\nWater Saturation ($S_w$)", "gas", "brine", 0.025, 0.60),
    FluidPair("Gas-Oil", "Gas-Oil\nOil Saturation ($S_o$)", "gas", "oil", 0.025, 0.13),
    FluidPair("Oil-Brine", "Oil-Brine\nWater Saturation ($S_w$)", "oil", "brine", 0.13, 0.60),
]


# --------------------------
# Temperature-dependent phase functions
# --------------------------

def lambda_matrix_sw(t_c: float, lambda20: float) -> float:
    """Sekiguchi-Waples-type matrix correction (Eq. 3.10).

    Parameters
    ----------
    t_c : float
        Temperature in degC.
    lambda20 : float
        Matrix thermal conductivity at 20 C, W m^-1 K^-1.
    """
    t_k = float(t_c) + 273.15
    return 358.0 * (1.0227 * float(lambda20) - 1.882) * ((1.0 / t_k) - 0.00068) + 1.84


def lambda_water_lowT_eq314(t_c: float) -> float:
    """Water thermal conductivity by low-temperature branch of Eq. (3.14).

    This script intentionally limits use to <= 130 C.
    """
    t_c = float(t_c)
    if not (0.0 <= t_c <= 130.0):
        raise ValueError("lambda_water_lowT_eq314 is restricted to 0..130 C in this script")
    return 0.565 - 1.88e-3 * t_c - 7.23e-6 * t_c * t_c


def lambda_oil_eq315(t_c: float) -> float:
    t_k = float(t_c) + 273.15
    return 0.2389 - 4.593e-4 * t_k + 2.676e-7 * t_k * t_k


def lambda_gas_eq315(t_c: float) -> float:
    t_k = float(t_c) + 273.15
    return -0.0969 + 4.37e-4 * t_k


def fluid_tc_by_name(name: str, t_c: float) -> float:
    key = name.lower()
    if key in {"water", "brine"}:
        return lambda_water_lowT_eq314(t_c)
    if key == "oil":
        return lambda_oil_eq315(t_c)
    if key == "gas":
        return lambda_gas_eq315(t_c)
    if key == "air":
        return lambda_gas_eq315(t_c)
    raise ValueError(f"Unknown fluid name: {name}")


# --------------------------
# EMT helpers
# --------------------------

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


def mixed_effective_tc_grid(pair: FluidPair, matrix_tc20: float, phi: np.ndarray, sat: np.ndarray, alpha: float, t_c: float) -> np.ndarray:
    z = np.empty((sat.size, phi.size), dtype=float)
    lam_m = lambda_matrix_sw(t_c, matrix_tc20)
    for i, s in enumerate(sat):
        pore_tc = float(pair.mixture_tc(float(s), t_c))
        for j, p in enumerate(phi):
            z[i, j] = effective_tc_scalar(lam_m, pore_tc, float(p), float(alpha))
    return z


def relative_increase_percent_isothermal(pair: FluidPair, matrix_tc: float, porosity: float, sat_high: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Original discrepancy atlas at fixed temperature using the lower-conductivity end-member reference."""
    sat_grid, alpha_grid = np.broadcast_arrays(np.asarray(sat_high, dtype=float), np.asarray(alpha, dtype=float))
    lam_mix = np.empty_like(sat_grid, dtype=float)
    lam_ref = np.empty_like(sat_grid, dtype=float)

    for idx in np.ndindex(sat_grid.shape):
        a = float(alpha_grid[idx])
        s = float(sat_grid[idx])
        lam_mix[idx] = effective_tc_scalar(matrix_tc, float((pair.low_tc_20 ** (1.0 - s)) * (pair.high_tc_20 ** s)), float(porosity), a)
        lam_ref[idx] = effective_tc_scalar(matrix_tc, float(pair.low_tc_20), float(porosity), a)

    return 100.0 * (lam_mix - lam_ref) / lam_ref


def relative_temp_change_percent(pair: FluidPair, matrix_tc20: float, porosity: float, sat_high: np.ndarray, alpha: np.ndarray, t_c: float, tref_c: float = 20.0) -> np.ndarray:
    """(lambda*(T)-lambda*(Tref))/lambda*(Tref) for a mixed-fluid rock."""
    sat_grid, alpha_grid = np.broadcast_arrays(np.asarray(sat_high, dtype=float), np.asarray(alpha, dtype=float))
    out = np.empty_like(sat_grid, dtype=float)

    lam_m_t = lambda_matrix_sw(t_c, matrix_tc20)
    lam_m_ref = lambda_matrix_sw(tref_c, matrix_tc20)

    for idx in np.ndindex(sat_grid.shape):
        a = float(alpha_grid[idx])
        s = float(sat_grid[idx])
        lam_mix_t = float(pair.mixture_tc(s, t_c))
        lam_mix_ref = float(pair.mixture_tc(s, tref_c))
        lam_t = effective_tc_scalar(lam_m_t, lam_mix_t, float(porosity), a)
        lam_ref = effective_tc_scalar(lam_m_ref, lam_mix_ref, float(porosity), a)
        out[idx] = 100.0 * (lam_t - lam_ref) / lam_ref

    return out


# --------------------------
# Plotting
# --------------------------

def make_component_relchange_figure(
    outpath: Path,
    matrix_tc20: float,
    t_min_c: float,
    t_max_c: float,
    n_t: int,
) -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    t_c = np.linspace(t_min_c, t_max_c, n_t)

    lam_m = np.array([lambda_matrix_sw(t, matrix_tc20) for t in t_c])
    lam_w = np.array([lambda_water_lowT_eq314(t) for t in t_c])
    lam_o = np.array([lambda_oil_eq315(t) for t in t_c])
    lam_g = np.array([lambda_gas_eq315(t) for t in t_c])

    lam_m_20 = lambda_matrix_sw(20.0, matrix_tc20)
    lam_w_20 = lambda_water_lowT_eq314(20.0)
    lam_o_20 = lambda_oil_eq315(20.0)
    lam_g_20 = lambda_gas_eq315(20.0)

    d_m = 100.0 * (lam_m - lam_m_20) / lam_m_20
    d_w = 100.0 * (lam_w - lam_w_20) / lam_w_20
    d_o = 100.0 * (lam_o - lam_o_20) / lam_o_20
    d_g = 100.0 * (lam_g - lam_g_20) / lam_g_20

    blue = "#1f4e79"
    red = "#b22222"
    black = "#111111"

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.8, 7.6),
        sharex=True,
        constrained_layout=True
    )

    # -------- upper panel: matrix --------
    ax = axes[0]
    ax.plot(t_c, d_m, color=black, lw=2.6)
    ax.axhline(0.0, color="0.35", lw=1.0, ls="--", alpha=0.5)

    ax.set_ylabel(r"$\delta\lambda_M(T)$ (%)")
    # ax.set_title(r"Relative change with respect to $20^\circ$C", pad=10)
    ax.text(0.02, 0.94, "(a) Matrix", transform=ax.transAxes, ha="left", va="top", fontsize=12)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # more major/minor lines for easier reading
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(True, which="major", color="0.70", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", color="0.82", linewidth=0.5, alpha=0.25)

    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=3.5, width=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------- lower panel: pore fluids --------
    ax = axes[1]
    ax.plot(t_c, d_w, color=blue, lw=2.4, label=r"$\Delta\lambda_{\mathrm{brine}}(T)$")
    ax.plot(t_c, d_o, color=red, lw=2.4, label=r"$\Delta\lambda_{\mathrm{oil}}(T)$")
    ax.plot(t_c, d_g, color=black, lw=2.4, label=r"$\Delta\lambda_{\mathrm{gas}}(T)$")
    ax.axhline(0.0, color="0.35", lw=1.0, ls="--", alpha=0.5)

    ax.set_xlabel(r"Temperature ($^\circ$C)")
    ax.set_ylabel(r"$\Delta\lambda_f(T)$ (%)")
    ax.text(0.02, 0.94, "(b) Pore fluids", transform=ax.transAxes, ha="left", va="top", fontsize=12)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # especially for gas: denser y-axis structure
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.grid(True, which="major", color="0.70", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", color="0.82", linewidth=0.5, alpha=0.25)

    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=3.5, width=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left", ncol=1, frameon=False, handlelength=2.8)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_porosity_figure(outpath: Path, matrix_tc: float, alphas: list[float], phi_max: float, n_phi: int, n_sat: int) -> None:
    phi = np.linspace(0.0, phi_max, n_phi)
    sat = np.linspace(0.0, 1.0, n_sat)
    phi_grid, sat_grid = np.meshgrid(phi, sat)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)

    norm = Normalize(vmin=0.51, vmax=float(matrix_tc))
    cmap = plt.get_cmap("RdYlBu_r")
    contour_levels = np.array([0.78, 1.06, 1.34, 1.62, 1.89, 2.17, 2.45, 2.72])

    for r, pair in enumerate(PAIRS):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            z = np.empty_like(phi_grid, dtype=float)
            for i, s in enumerate(sat):
                pore_tc = float((pair.low_tc_20 ** (1.0 - s)) * (pair.high_tc_20 ** s))
                for j, p in enumerate(phi):
                    z[i, j] = effective_tc_scalar(matrix_tc, pore_tc, p, alpha)

            ax.contourf(phi_grid, sat_grid, z, levels=60, cmap=cmap, norm=norm)
            cs = ax.contour(phi_grid, sat_grid, z, levels=contour_levels, colors="k", linewidths=1.4)
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
    alog, sat_grid = np.meshgrid(np.log10(alphas), sat)
    alpha_grid = 10.0 ** alog

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    major_levels = [0, 1, 5, 10, 50, 100, 150, 200]
    minor_levels = np.array([0.5, 2, 3, 4, 7.5, 20, 30, 40, 75, 125, 175])

    for r, pair in enumerate(PAIRS):
        for c, phi in enumerate(phi_values):
            ax = axes[r, c]
            z = relative_increase_percent_isothermal(pair, matrix_tc, phi, sat_grid, alpha_grid)
            ax.contour(alog, sat_grid, z, levels=minor_levels, colors="0.75", linewidths=0.8)
            cs = ax.contour(alog, sat_grid, z, levels=major_levels, colors="0.15", linewidths=1.8)
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


def make_component_temperature_figure(outpath: Path, matrix_tc20: float, t_min_c: float, t_max_c: float, n_t: int) -> None:
    t_c = np.linspace(t_min_c, t_max_c, n_t)
    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)

    ax.plot(t_c, [lambda_matrix_sw(t, matrix_tc20) for t in t_c], label=r"$\lambda_M(T)$ (Eq. 3.10)", linewidth=2.2)
    ax.plot(t_c, [lambda_water_lowT_eq314(t) for t in t_c], label=r"$\lambda_w(T)$ (Eq. 3.14)", linewidth=2.2)
    ax.plot(t_c, [lambda_oil_eq315(t) for t in t_c], label=r"$\lambda_o(T)$ (Eq. 3.15)", linewidth=2.2)
    ax.plot(t_c, [lambda_gas_eq315(t) for t in t_c], label=r"$\lambda_g(T)$ (Eq. 3.15)", linewidth=2.2)

    ax.set_xlabel(r"Temperature ($^{\circ}$C)")
    ax.set_ylabel(r"Thermal conductivity (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Temperature-dependent matrix and pore-fluid thermal conductivities")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_porosity_figure_temperature(outpath: Path, matrix_tc20: float, alphas: list[float], phi_max: float, n_phi: int, n_sat: int, t_c: float) -> None:
    phi = np.linspace(0.0, phi_max, n_phi)
    sat = np.linspace(0.0, 1.0, n_sat)
    phi_grid, sat_grid = np.meshgrid(phi, sat)

    all_panels = []
    for pair in PAIRS:
        for alpha in alphas:
            all_panels.append(mixed_effective_tc_grid(pair, matrix_tc20, phi, sat, alpha, t_c))
    vmin = min(np.min(z) for z in all_panels)
    vmax = max(np.max(z) for z in all_panels)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdYlBu_r")

    # pick contour levels from global range for consistency
    contour_levels = np.linspace(vmin, vmax, 8)

    k = 0
    for r, pair in enumerate(PAIRS):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            z = all_panels[k]
            k += 1

            ax.contourf(phi_grid, sat_grid, z, levels=60, cmap=cmap, norm=norm)
            cs = ax.contour(phi_grid, sat_grid, z, levels=contour_levels, colors="k", linewidths=1.2)
            ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")

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
    cbar.set_label(rf"$\lambda^*(T={t_c:.0f}^\circ\mathrm{{C}})$ (W m$^{{-1}}$ K$^{{-1}}$)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_temp_relchange_atlas(outpath: Path, matrix_tc20: float, phi_values: list[float], alpha_min: float, alpha_max: float, n_alpha: int, n_sat: int, t_c: float, tref_c: float = 20.0) -> None:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    sat = np.linspace(0.0, 1.0, n_sat)
    alog, sat_grid = np.meshgrid(np.log10(alphas), sat)
    alpha_grid = 10.0 ** alog

    all_panels = []
    for pair in PAIRS:
        for phi in phi_values:
            all_panels.append(relative_temp_change_percent(pair, matrix_tc20, phi, sat_grid, alpha_grid, t_c=t_c, tref_c=tref_c))

    vmin = min(float(np.min(z)) for z in all_panels)
    vmax = max(float(np.max(z)) for z in all_panels)
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")

    # choose symmetric or simple levels, including zero if crossed
    if vmin < 0.0 < vmax:
        m = max(abs(vmin), abs(vmax))
        fill_levels = np.linspace(-m, m, 61)
        contour_levels = np.array([-m, -0.75*m, -0.5*m, -0.25*m, 0.0, 0.25*m, 0.5*m, 0.75*m, m])
    else:
        fill_levels = np.linspace(vmin, vmax, 61)
        contour_levels = np.linspace(vmin, vmax, 9)

    k = 0
    for r, pair in enumerate(PAIRS):
        for c, phi in enumerate(phi_values):
            ax = axes[r, c]
            z = all_panels[k]
            k += 1

            ax.contourf(alog, sat_grid, z, levels=fill_levels, cmap=cmap, norm=norm)
            cs = ax.contour(alog, sat_grid, z, levels=contour_levels, colors="k", linewidths=1.2)
            ax.clabel(cs, inline=True, fontsize=9, fmt=lambda v: f"{v:.1f}%")

            if r == 0:
                ax.set_title(rf"$\phi$ = {phi*100:.1f}%", fontsize=18, pad=10)
            if c == 0:
                ax.set_ylabel(pair.ylabel, fontsize=15)
            if r == 2:
                ax.set_xlabel(r"$\log_{10}$(Aspect Ratio $\alpha$)", fontsize=15)

            ax.set_xlim(np.log10(alpha_min), np.log10(alpha_max))
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.95, pad=0.03)
    cbar.set_label(rf"Relative change: $(\lambda^*(T)-\lambda^*(20^\circ C))/\lambda^*(20^\circ C)*100$, T={t_c:.0f}$^\circ$C (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# CLI and summary
# --------------------------

def write_summary(outdir: Path, args: argparse.Namespace) -> None:
    text = f"""Mixed-fluid GSA atlas parameters (temperature-aware version)
===================================================
matrix_tc20 = {args.matrix_tc}
porosity_alphas = {args.alphas}
phi_max = {args.phi_max}
phi_fixed = {args.error_porosity}
alpha_range = [{args.alpha_min}, {args.alpha_max}]
atlas_temperatures_C = {args.atlas_temperatures}
T_range_components_C = [{args.t_min_c}, {args.t_max_c}]
water_model = Eq. 3.14 low-temperature branch only (restricted to <=130 C)
matrix_model = Sekiguchi-Waples Eq. 3.10
fluid_mix_rule = Lichtenecker geometric mixing
relchange_reference = lambda*(20 C)
"""
    (outdir / "summary.txt").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed-fluid GSA atlases with temperature-dependent matrix and fluids.")
    p.add_argument("--outdir", default="results/mixed_fluid_gsa_temp", help="Output directory")
    p.add_argument("--matrix-tc", type=float, default=3.0, help="Matrix thermal conductivity at 20 C, W m^-1 K^-1")
    p.add_argument("--alphas", type=float, nargs=3, default=[0.01, 0.1, 1.0], help="Aspect ratios for the porosity figure")
    p.add_argument("--phi-max", type=float, default=0.25)
    p.add_argument("--n-phi", type=int, default=151)
    p.add_argument("--n-sat", type=int, default=121)
    p.add_argument("--error-porosity", type=float, nargs=3, default=[0.015, 0.105, 0.245], help="Fixed porosities for atlas panels")
    p.add_argument("--alpha-min", type=float, default=0.01)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alpha", type=int, default=121)
    p.add_argument("--t-min-c", type=float, default=20.0)
    p.add_argument("--t-max-c", type=float, default=130.0)
    p.add_argument("--n-t", type=int, default=300)
    p.add_argument("--atlas-temperatures", type=float, nargs="*", default=[20.0, 50.0, 100.0], help="Temperatures for temperature-aware atlases")
    args = p.parse_args()
    if args.t_max_c > 130.0:
        p.error("This script intentionally restricts water Eq. (3.14) to <=130 C.")
    for t in args.atlas_temperatures:
        if t > 130.0:
            p.error("All atlas temperatures must be <=130 C in this script.")
    return args


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # original isothermal 20 C figures
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

    # # component functions vs temperature
    # make_component_temperature_figure(
    #     outdir / "ch4_temp_component_functions.png",
    #     matrix_tc20=args.matrix_tc,
    #     t_min_c=args.t_min_c,
    #     t_max_c=args.t_max_c,
    #     n_t=args.n_t,
    # )

    make_component_relchange_figure(
    outdir / "ch4_temp_component_relchange.png",
    matrix_tc20=args.matrix_tc,
    t_min_c=args.t_min_c,
    t_max_c=args.t_max_c,
    n_t=args.n_t,
)

    # temperature-aware atlases
    for t_c in args.atlas_temperatures:
        t_label = f"{int(round(t_c)):03d}C"
        make_porosity_figure_temperature(
            outdir / f"ch4_carb_mix_porosity_temp_{t_label}.png",
            matrix_tc20=args.matrix_tc,
            alphas=list(args.alphas),
            phi_max=args.phi_max,
            n_phi=args.n_phi,
            n_sat=args.n_sat,
            t_c=float(t_c),
        )
        if abs(float(t_c) - 20.0) > 1e-9:
            make_temp_relchange_atlas(
                outdir / f"ch4_carb_mix_relchange_temp_{t_label}.png",
                matrix_tc20=args.matrix_tc,
                phi_values=list(args.error_porosity),
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                n_alpha=args.n_alpha,
                n_sat=args.n_sat,
                t_c=float(t_c),
                tref_c=20.0,
            )

    write_summary(outdir, args)
    print(f"Saved results to: {outdir}")


def style_scientific_axis(ax, x_major=None, y_major=None, x_minor=2, y_minor=2):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))

    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))

    ax.grid(True, which="major", color="0.70", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", color="0.82", linewidth=0.5, alpha=0.25)

    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=3.5, width=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


if __name__ == "__main__":
    main()
