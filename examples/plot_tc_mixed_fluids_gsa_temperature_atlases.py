
from __future__ import annotations

"""Mixed-fluid GSA thermal-conductivity atlas with temperature-dependent phase conductivities.

This script keeps the original isothermal mixed-fluid atlases and adds:
1) informative phase-conductivity-vs-temperature plots,
2) temperature-aware porosity atlases, and
3) temperature-aware discrepancy atlases analogous to the original Figure 4.14 style.

Temperature models:
- Matrix conductivity: Sekiguchi-Waples Eq. (3.10)
- Water conductivity:  Eq. (3.14), low-temperature branch only (used up to 130 C)
- Oil and gas:         Eq. (3.15)
- Mixed pore fluids:   geometrical average (Lichtenecker rule)

Designed for RockPhysX isotropic GSA wrapper:
    rockphysx.models.emt.gsa_transport.two_phase_thermal_isotropic
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
        """Geometric average for isothermal pore-fluid mixture.

        sat_high is saturation of the higher-conductivity component named in ylabel.
        """
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
    """Sekiguchi-Waples matrix correction, Eq. (3.10), T in Kelvin."""
    T_k = np.asarray(T_c, dtype=float) + 273.15
    return 358.0 * (1.0227 * lambda20 - 1.882) * ((1.0 / T_k) - 0.00068) + 1.84


def lambda_water_314_low(T_c: np.ndarray) -> np.ndarray:
    """Water conductivity from Eq. (3.14), low-temperature branch only.

    The script is intentionally limited to 130 C, so the high-temperature branch
    is not used here.
    """
    T_c = np.asarray(T_c, dtype=float)
    return 0.565 - 1.88e-3 * T_c - 7.23e-6 * T_c**2


def lambda_oil_315(T_c: np.ndarray) -> np.ndarray:
    """Liquid petroleum conductivity from Eq. (3.15), T in Kelvin."""
    T_c = np.asarray(T_c, dtype=float)
    T_k = T_c + 273.15
    return np.where(T_c < 240.0, 0.2389 - 4.593e-4 * T_k + 2.676e-7 * T_k**2, 0.075)


def lambda_gas_315(T_c: np.ndarray) -> np.ndarray:
    """Gas conductivity from Eq. (3.15), T in Kelvin."""
    T_c = np.asarray(T_c, dtype=float)
    T_k = T_c + 273.15
    return np.where(T_c < 120.0, -0.0969 + 4.37e-4 * T_k, 0.075)


def mixture_tc_temperature(pair_name: str, sat_high: np.ndarray, T_c: np.ndarray) -> np.ndarray:
    """Temperature-dependent fluid-mixture conductivity from the geometrical average."""
    sat_high = np.asarray(sat_high, dtype=float)
    T_c = np.asarray(T_c, dtype=float)

    lam_w = lambda_water_314_low(T_c)
    lam_o = lambda_oil_315(T_c)
    lam_g = lambda_gas_315(T_c)

    if pair_name == "Gas-Brine":
        return (lam_g ** (1.0 - sat_high)) * (lam_w ** sat_high)
    if pair_name == "Gas-Oil":
        return (lam_g ** (1.0 - sat_high)) * (lam_o ** sat_high)
    if pair_name == "Oil-Brine":
        return (lam_o ** (1.0 - sat_high)) * (lam_w ** sat_high)
    raise ValueError(f"Unknown pair: {pair_name}")


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
    """Isothermal atlas helper."""
    Z = np.empty((sat.size, phi.size), dtype=float)
    for i, s in enumerate(sat):
        pore_tc = float(pair.mixture_tc(float(s)))
        for j, p in enumerate(phi):
            Z[i, j] = effective_tc_scalar(matrix_tc, pore_tc, float(p), float(alpha))
    return Z


def mixed_effective_tc_grid_temperature(pair_name: str, matrix_lambda20: float, phi: np.ndarray, sat: np.ndarray, alpha: float, T_c: float) -> np.ndarray:
    """Temperature-aware atlas helper at one fixed temperature T_c."""
    Z = np.empty((sat.size, phi.size), dtype=float)
    lam_M = float(lambda_matrix_sw(T_c, matrix_lambda20))
    for i, s in enumerate(sat):
        pore_tc = float(mixture_tc_temperature(pair_name, float(s), float(T_c)))
        for j, p in enumerate(phi):
            Z[i, j] = effective_tc_scalar(lam_M, pore_tc, float(p), float(alpha))
    return Z


def relative_increase_percent(pair: FluidPair, matrix_tc: float, porosity: float, sat_high: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Original isothermal discrepancy relative to the lower-conductivity end-member."""
    sat_grid, alpha_grid = np.broadcast_arrays(np.asarray(sat_high, dtype=float), np.asarray(alpha, dtype=float))
    lam_mix = np.empty_like(sat_grid, dtype=float)
    lam_ref = np.empty_like(sat_grid, dtype=float)

    for idx in np.ndindex(sat_grid.shape):
        a = float(alpha_grid[idx])
        s = float(sat_grid[idx])
        lam_mix[idx] = effective_tc_scalar(matrix_tc, float(pair.mixture_tc(s)), float(porosity), a)
        lam_ref[idx] = effective_tc_scalar(matrix_tc, float(pair.low_tc), float(porosity), a)

    return 100.0 * (lam_mix - lam_ref) / lam_ref


def relative_increase_percent_temperature(pair_name: str, matrix_lambda20: float, porosity: float, sat_high: np.ndarray, alpha: np.ndarray, T_c: float) -> np.ndarray:
    """Temperature-aware discrepancy relative to the lower-conductivity end-member rock at the same T."""
    sat_grid, alpha_grid = np.broadcast_arrays(np.asarray(sat_high, dtype=float), np.asarray(alpha, dtype=float))
    lam_mix = np.empty_like(sat_grid, dtype=float)
    lam_ref = np.empty_like(sat_grid, dtype=float)

    lam_M = float(lambda_matrix_sw(T_c, matrix_lambda20))
    if pair_name == "Gas-Brine":
        lam_ref_fluid = float(lambda_gas_315(T_c))
    elif pair_name == "Gas-Oil":
        lam_ref_fluid = float(lambda_gas_315(T_c))
    elif pair_name == "Oil-Brine":
        lam_ref_fluid = float(lambda_oil_315(T_c))
    else:
        raise ValueError(f"Unknown pair: {pair_name}")

    for idx in np.ndindex(sat_grid.shape):
        a = float(alpha_grid[idx])
        s = float(sat_grid[idx])
        lam_mix_fluid = float(mixture_tc_temperature(pair_name, s, T_c))
        lam_mix[idx] = effective_tc_scalar(lam_M, lam_mix_fluid, float(porosity), a)
        lam_ref[idx] = effective_tc_scalar(lam_M, lam_ref_fluid, float(porosity), a)

    return 100.0 * (lam_mix - lam_ref) / lam_ref


# ------------------------- figure builders -------------------------

def _common_colorbar(fig, axes, cmap, norm, ticks, label):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.95, pad=0.03)
    cbar.set_label(label, fontsize=13)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=11)


def make_porosity_figure(outpath: Path, matrix_tc: float, alphas: list[float], phi_max: float, n_phi: int, n_sat: int) -> None:
    """Original isothermal porosity atlas."""
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
            cs = ax.contour(PHI, SAT, Z, levels=contour_levels, colors="k", linewidths=1.3)
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

    _common_colorbar(
        fig, axes, cmap, norm,
        ticks=[0.78, 1.06, 1.34, 1.62, 1.89, 2.17, 2.45, 2.72, 2.95],
        label=r"Effective thermal conductivity $\lambda^*$ (W m$^{-1}$ K$^{-1}$)"
    )
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_error_figure(outpath: Path, matrix_tc: float, phi_values: list[float], alpha_min: float, alpha_max: float, n_alpha: int, n_sat: int) -> None:
    """Original isothermal discrepancy atlas."""
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


def make_component_temperature_figure(outpath: Path, matrix_lambda20: float, t_min_c: float, t_max_c: float, n_t: int) -> None:
    T_c = np.linspace(t_min_c, t_max_c, n_t)
    lam_M = lambda_matrix_sw(T_c, matrix_lambda20)
    lam_w = lambda_water_314_low(T_c)
    lam_o = lambda_oil_315(T_c)
    lam_g = lambda_gas_315(T_c)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)

    axes[0].plot(T_c, lam_M, lw=2.5, label=rf"$\lambda_M(T)$, Eq. (3.10), $\lambda_{{20}}={matrix_lambda20:.2f}$")
    axes[0].set_ylabel(r"Matrix thermal conductivity $\lambda_M$ (W m$^{-1}$ K$^{-1}$)")
    axes[0].grid(alpha=0.20)
    axes[0].legend(frameon=False)

    axes[1].plot(T_c, lam_w, lw=2.2, label=r"$\lambda_w(T)$, Eq. (3.14), low-$T$ branch")
    axes[1].plot(T_c, lam_o, lw=2.2, label=r"$\lambda_o(T)$, Eq. (3.15)")
    axes[1].plot(T_c, lam_g, lw=2.2, label=r"$\lambda_g(T)$, Eq. (3.15)")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel(r"Fluid thermal conductivity $\lambda_f$ (W m$^{-1}$ K$^{-1}$)")
    axes[1].grid(alpha=0.20)
    axes[1].legend(frameon=False)
    axes[0].set_title("Temperature-dependent phase conductivities")

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_temperature_porosity_atlas(outpath: Path, matrix_lambda20: float, alphas: list[float], phi_max: float, n_phi: int, n_sat: int, T_c: float) -> None:
    """Temperature-aware counterpart of the porosity atlas at one fixed temperature."""
    phi = np.linspace(0.0, phi_max, n_phi)
    sat = np.linspace(0.0, 1.0, n_sat)
    PHI, SAT = np.meshgrid(phi, sat)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    lamM = float(lambda_matrix_sw(T_c, matrix_lambda20))
    # global color scale per atlas
    norm = Normalize(vmin=0.51, vmax=max(lamM, 0.55))
    cmap = plt.get_cmap("RdYlBu_r")

    # temperature-specific contour levels based on matrix conductivity
    contour_levels = np.linspace(max(0.6, 0.7*lamM), 0.95*lamM, 6)

    for r, pair in enumerate(PAIRS):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            Z = mixed_effective_tc_grid_temperature(pair.name, matrix_lambda20, phi, sat, alpha, T_c)
            ax.contourf(PHI, SAT, Z, levels=60, cmap=cmap, norm=norm)
            cs = ax.contour(PHI, SAT, Z, levels=contour_levels, colors="k", linewidths=1.3)
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

    fig.suptitle(rf"Temperature-aware mixed-fluid GSA atlas at $T$ = {T_c:.0f}$^\circ$C", fontsize=16)
    ticks = np.linspace(norm.vmin, norm.vmax, 8)
    _common_colorbar(
        fig, axes, cmap, norm,
        ticks=np.round(ticks, 2),
        label=r"Effective thermal conductivity $\lambda^*$ (W m$^{-1}$ K$^{-1}$)"
    )
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_temperature_error_atlas(outpath: Path, matrix_lambda20: float, phi_values: list[float], alpha_min: float, alpha_max: float, n_alpha: int, n_sat: int, T_c: float) -> None:
    """Temperature-aware counterpart of the discrepancy atlas at one fixed temperature."""
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
            Z = relative_increase_percent_temperature(pair.name, matrix_lambda20, phi, SAT, ALPHA, T_c)
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

    fig.suptitle(rf"Temperature-aware relative discrepancy atlas at $T$ = {T_c:.0f}$^\circ$C", fontsize=16)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------- CLI / main -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed-fluid GSA atlases with temperature-dependent phases")
    p.add_argument("--outdir", default="results/mixed_fluid_gsa_temp", help="Output directory")

    # original/isothermal inputs
    p.add_argument("--matrix-tc", type=float, default=3.0, help="Reference matrix conductivity λ20, W m^-1 K^-1")
    p.add_argument("--air-tc", type=float, default=0.025, help="Isothermal gas conductivity for baseline atlases")
    p.add_argument("--oil-tc", type=float, default=0.13, help="Isothermal oil conductivity for baseline atlases")
    p.add_argument("--brine-tc", type=float, default=0.60, help="Isothermal brine conductivity for baseline atlases")

    p.add_argument("--alphas", type=float, nargs=3, default=[0.01, 0.1, 1.0], help="Aspect ratios for porosity atlases")
    p.add_argument("--phi-max", type=float, default=0.25)
    p.add_argument("--n-phi", type=int, default=121)
    p.add_argument("--n-sat", type=int, default=121)
    p.add_argument("--error-porosity", type=float, nargs=3, default=[0.015, 0.105, 0.245], help="Fixed porosities for discrepancy atlases")
    p.add_argument("--alpha-min", type=float, default=0.01)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alpha", type=int, default=121)

    # temperature settings
    p.add_argument("--t-min-c", type=float, default=20.0)
    p.add_argument("--t-max-c", type=float, default=130.0)
    p.add_argument("--n-temp", type=int, default=221)
    p.add_argument("--atlas-temperatures", type=float, nargs="*", default=[20.0, 75.0, 130.0], help="Fixed temperatures for temperature-aware atlases")
    return p.parse_args()


def write_summary(outdir: Path, args: argparse.Namespace) -> None:
    text = f"""Mixed-fluid GSA atlases with temperature-dependent phases
=====================================================
matrix_lambda20 = {args.matrix_tc}
air_tc_isothermal = {args.air_tc}
oil_tc_isothermal = {args.oil_tc}
brine_tc_isothermal = {args.brine_tc}
porosity_alphas = {args.alphas}
phi_max = {args.phi_max}
phi_fixed_error = {args.error_porosity}
alpha_range = [{args.alpha_min}, {args.alpha_max}]
temperature_range_C = [{args.t_min_c}, {args.t_max_c}]
atlas_temperatures_C = {args.atlas_temperatures}
comparison_body = matrix
mixture_rule = Lichtenecker / geometrical average
matrix_temperature_model = Sekiguchi-Waples Eq. (3.10)
water_temperature_model = Eq. (3.14), low-temperature branch only
oil_gas_temperature_model = Eq. (3.15)
"""
    (outdir / "summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    global PAIRS
    args = parse_args()

    if args.t_max_c > 130.0:
        raise ValueError("This version is intentionally limited to 130 °C because only the low-temperature water branch is used.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Update baseline isothermal pairs from CLI
    PAIRS = [
        FluidPair(PAIRS[0].name, PAIRS[0].ylabel, PAIRS[0].low_name, PAIRS[0].high_name, args.air_tc, args.brine_tc),
        FluidPair(PAIRS[1].name, PAIRS[1].ylabel, PAIRS[1].low_name, PAIRS[1].high_name, args.air_tc, args.oil_tc),
        FluidPair(PAIRS[2].name, PAIRS[2].ylabel, PAIRS[2].low_name, PAIRS[2].high_name, args.oil_tc, args.brine_tc),
    ]

    # Original baseline figures
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

    # New temperature-function figure
    make_component_temperature_figure(
        outdir / "ch4_temp_component_functions.png",
        matrix_lambda20=args.matrix_tc,
        t_min_c=args.t_min_c,
        t_max_c=args.t_max_c,
        n_t=args.n_temp,
    )

    # New temperature-aware atlas figures at selected temperatures
    for T_c in args.atlas_temperatures:
        tag = f"{int(round(T_c)):03d}C"
        make_temperature_porosity_atlas(
            outdir / f"ch4_carb_mix_porosity_temp_{tag}.png",
            matrix_lambda20=args.matrix_tc,
            alphas=list(args.alphas),
            phi_max=args.phi_max,
            n_phi=args.n_phi,
            n_sat=args.n_sat,
            T_c=float(T_c),
        )
        make_temperature_error_atlas(
            outdir / f"ch4_carb_mix_error_temp_{tag}.png",
            matrix_lambda20=args.matrix_tc,
            phi_values=list(args.error_porosity),
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            n_alpha=args.n_alpha,
            n_sat=args.n_sat,
            T_c=float(T_c),
        )

    write_summary(outdir, args)
    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
