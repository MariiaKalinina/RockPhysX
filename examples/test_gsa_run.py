from __future__ import annotations

import rockphysx.models.emt.gsa_transport

# lam_iso = rockphysx.models.emt.gsa_transport.two_phase_thermal_isotropic(
#     matrix_value=3.0,
#     inclusion_value=0.025,
#     porosity=0.12,
#     aspect_ratio=0.1,
#     comparison="matrix",
# )

# print("Isotropic lambda* =", lam_iso)

# lam_ti = rockphysx.models.emt.gsa_transport.two_phase_thermal_ti(
#     matrix_value=3.0,
#     inclusion_value=0.025,
#     porosity=0.12,
#     aspect_ratio=0.01,
#     comparison="matrix",
# )

# print("lambda_parallel =", lam_ti.lambda_parallel)
# print("lambda_perpendicular =", lam_ti.lambda_perpendicular)
# print("anisotropy ratio =", lam_ti.ratio)
#

# def matrix_oil_pores_gas_cracks_parallel_to_bedding(
#     matrix_tc: float,
#     oil_tc: float,
#     gas_tc: float,
#     phi_oil_pores: float,
#     phi_gas_cracks: float,
#     alpha_oil_pores: float,
#     alpha_gas_cracks: float,
#     symmetry_axis=(0.0, 0.0, 1.0),
# ):
#     import rockphysx.models.emt.gsa_transport as gsa

#     lambda_bg = gsa.two_phase_thermal_isotropic(
#         matrix_value=matrix_tc,
#         inclusion_value=oil_tc,
#         porosity=phi_oil_pores,
#         aspect_ratio=alpha_oil_pores,
#         comparison="matrix",
#     )

#     lam_ti = gsa.two_phase_thermal_ti(
#         matrix_value=lambda_bg,
#         inclusion_value=gas_tc,
#         porosity=phi_gas_cracks,
#         aspect_ratio=alpha_gas_cracks,
#         comparison="matrix",
#         symmetry_axis=symmetry_axis,
#     )

#     return {
#         "lambda_background": lambda_bg,
#         "lambda_normal": lam_ti.lambda_parallel,
#         "lambda_bedding": lam_ti.lambda_perpendicular,
#         "anisotropy_ratio": lam_ti.lambda_perpendicular / lam_ti.lambda_parallel,
#         "tensor": lam_ti.tensor,
#     }


# result = matrix_oil_pores_gas_cracks_parallel_to_bedding(
#     matrix_tc=3.00,
#     oil_tc=0.13,
#     gas_tc=0.025,
#     phi_oil_pores=0.12,
#     phi_gas_cracks=0.03,
#     alpha_oil_pores=0.20,
#     alpha_gas_cracks=0.001,
# )

# print(result)

#



"""Sensitivity-study examples for the tensor GSA transport model.

This example script is intended to be copied into:
    examples/test_gsa_run.py

It generates three thesis-style figure sets:
1. lambda*(alpha) and lambda*(phi) for dry / oil / brine states.
2. 2D response maps lambda*(alpha, phi) for dry / oil / brine.
3. TI anisotropy study: lambda_parallel, lambda_perpendicular and
   anisotropy ratio lambda_parallel / lambda_perpendicular.

Run from the repository root, for example:
    PYTHONPATH=src python examples/test_gsa_run.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.models.emt.gsa_transport import (
    two_phase_thermal_isotropic,
    two_phase_thermal_ti,
)


MATRIX_TC = 3.00
SATURATION_TC = {
    "Dry (air)": 0.025,
    "Oil": 0.13,
    "Brine": 0.60,
}

PHI_REF = 0.03
ALPHA_REF = 0.001
COMPARISON_BODY = "matrix"
OUTPUT_DIR = Path("figures/thermal")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")


# -----------------------------------------------------------------------------
# Example 1
# -----------------------------------------------------------------------------


def plot_example1_saturation_curves() -> None:
    """Plot lambda*(alpha) and lambda*(phi) for dry / oil / brine."""
    alpha_grid = np.logspace(-4, 0, 5)
    phi_grid = np.linspace(0.01, 0.35, 5)
    print(alpha_grid, phi_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Panel A: lambda*(alpha) at fixed porosity
    ax = axes[0]
    for label, fluid_tc in SATURATION_TC.items():
        values = [
            rockphysx.models.emt.gsa_transport.two_phase_thermal_isotropic(
                matrix_value=MATRIX_TC,
                inclusion_value=fluid_tc,
                porosity=PHI_REF,
                aspect_ratio=float(alpha),
                comparison=COMPARISON_BODY,
            )
            for alpha in alpha_grid
        ]
        ax.plot(alpha_grid, values, linewidth=2.2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"Aspect ratio $\alpha$")
    ax.set_ylabel(r"Effective thermal conductivity $\lambda^*$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(rf"(a) $\lambda^*(\alpha)$ at fixed porosity $\phi={PHI_REF:.2f}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    # Panel B: lambda*(phi) at fixed aspect ratio
    ax = axes[1]
    for label, fluid_tc in SATURATION_TC.items():
        values = [
            rockphysx.models.emt.gsa_transport.two_phase_thermal_isotropic(
                matrix_value=MATRIX_TC,
                inclusion_value=fluid_tc,
                porosity=float(phi),
                aspect_ratio=ALPHA_REF,
                comparison=COMPARISON_BODY,
            )
            for phi in phi_grid
        ]
        ax.plot(phi_grid, values, linewidth=2.2, label=label)

    ax.set_xlabel(r"Porosity $\phi$")
    ax.set_ylabel(r"Effective thermal conductivity $\lambda^*$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(rf"(b) $\lambda^*(\phi)$ at fixed aspect ratio $\alpha={ALPHA_REF:.2f}$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.suptitle("Example 1. Sensitivity of effective thermal conductivity to aspect ratio and porosity", y=1.02)
    fig.tight_layout()
    save_figure(fig, "test_gsa_example1_saturation_curves")


# -----------------------------------------------------------------------------
# Example 2
# -----------------------------------------------------------------------------


def plot_example2_response_maps() -> None:
    """Plot lambda*(alpha, phi) maps for dry / oil / brine."""
    alpha_grid = np.logspace(-4, 0, 5)
    phi_grid = np.linspace(0.01, 0.35, 5)
    A, P = np.meshgrid(alpha_grid, phi_grid)


    state_maps: dict[str, np.ndarray] = {}
    vmin = np.inf
    vmax = -np.inf

    for label, fluid_tc in SATURATION_TC.items():
        Z = np.empty_like(A)
        for i in range(P.shape[0]):
            for j in range(A.shape[1]):
                Z[i, j] = rockphysx.models.emt.gsa_transport.two_phase_thermal_isotropic(
                    matrix_value=MATRIX_TC,
                    inclusion_value=fluid_tc,
                    porosity=float(P[i, j]),
                    aspect_ratio=float(A[i, j]),
                    comparison=COMPARISON_BODY,
                )
        state_maps[label] = Z
        vmin = min(vmin, float(np.min(Z)))
        vmax = max(vmax, float(np.max(Z)))
        print(i, j, label, vmin, vmax)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)

    contour = None
    for ax, (label, Z) in zip(axes, state_maps.items(), strict=True):
        levels = np.linspace(vmin, vmax, 25)
        contour = ax.contourf(A, P, Z, levels=levels)
        ax.contour(A, P, Z, levels=levels[::3], colors="black", linewidths=0.5, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel(r"Aspect ratio $\alpha$")
    axes[0].set_ylabel(r"Porosity $\phi$")

    cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(r"$\lambda^*$ (W m$^{-1}$ K$^{-1}$)")

    fig.suptitle(r"Example 2. Response surfaces $\lambda^*(\alpha,\phi)$ for different saturation states", y=1.02)
    fig.tight_layout()
    save_figure(fig, "test_gsa_example2_response_maps")


# -----------------------------------------------------------------------------
# Example 3
# -----------------------------------------------------------------------------


def plot_example3_ti_anisotropy() -> None:
    alpha_grid = np.logspace(-4, 0, 11)
    porosity_values = [0.05, 0.10, 0.20]
    fluid_tc = SATURATION_TC["Dry (air)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    for phi in porosity_values:
        lam_normal = []
        lam_bedding = []

        for alpha in alpha_grid:
            result = rockphysx.models.emt.gsa_transport.two_phase_thermal_ti(
                matrix_value=MATRIX_TC,
                inclusion_value=fluid_tc,
                porosity=phi,
                aspect_ratio=float(alpha),
                comparison=COMPARISON_BODY,
                symmetry_axis=(0.0, 0.0, 1.0),
            )
            lam_normal.append(result.lambda_parallel)
            lam_bedding.append(result.lambda_perpendicular)

        ax.plot(alpha_grid, lam_normal, linewidth=2.2, label=rf"$\lambda_{{normal}}$, $\phi={phi:.2f}$")
        ax.plot(alpha_grid, lam_bedding, linewidth=1.6, linestyle="--", label=rf"$\lambda_{{bedding}}$, $\phi={phi:.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Aspect ratio $\alpha$")
    ax.set_ylabel(r"Thermal conductivity (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(r"(a) TI components for aligned gas cracks")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    alpha_map = np.logspace(-4, 0, 10)
    phi_map = np.linspace(0.01, 0.30, 10)
    A, P = np.meshgrid(alpha_map, phi_map)
    ratio = np.empty_like(A)

    for i in range(P.shape[0]):
        for j in range(A.shape[1]):
            result = rockphysx.models.emt.gsa_transport.two_phase_thermal_ti(
                matrix_value=MATRIX_TC,
                inclusion_value=fluid_tc,
                porosity=float(P[i, j]),
                aspect_ratio=float(A[i, j]),
                comparison=COMPARISON_BODY,
                symmetry_axis=(0.0, 0.0, 1.0),
            )
            ratio[i, j] = result.lambda_perpendicular / result.lambda_parallel

    levels = np.linspace(float(np.min(ratio)), float(np.max(ratio)), 25)
    contour = axes[1].contourf(A, P, ratio, levels=levels)
    axes[1].contour(A, P, ratio, levels=levels[::3], colors="black", linewidths=0.5, alpha=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Aspect ratio $\alpha$")
    axes[1].set_ylabel(r"Porosity $\phi$")
    axes[1].set_title(r"(b) Anisotropy ratio $\lambda_{bedding}/\lambda_{normal}$")

    cbar = fig.colorbar(contour, ax=axes[1], shrink=0.92)
    cbar.set_label(r"$\lambda_{bedding}/\lambda_{normal}$")

    fig.suptitle("Example 3. TI anisotropy sensitivity study for aligned crack fabric", y=1.02)
    fig.tight_layout()
    save_figure(fig, "test_gsa_example3_ti_anisotropy")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Bayuk-style thermal-conductivity plots
# -----------------------------------------------------------------------------


def plot_bayuk_style_tc_ratio_vs_porosity() -> None:
    """
    Bayuk-style analogue of Fig. 3 for thermal conductivity:
    ratio lambda_max / lambda_min versus porosity
    for disks (alpha=0.01) and channels (alpha=3.0).
    """
    # porosity_grid = np.logspace(-4, np.log10(0.35), 10)
    porosity_grid = np.linspace(0.01, 0.50, 10)

    fluid_cases = {
        "Brine": 0.60,
        "Oil": 0.13,
        "Air": 0.025,
    }

    shape_cases = {
        "channels": {"alpha": 3.0, "linestyle": "-"},
        "disks": {"alpha": 0.01, "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    for fluid_label, fluid_tc in fluid_cases.items():
        for shape_label, shape_cfg in shape_cases.items():
            ratio = []

            for phi in porosity_grid:
                result = rockphysx.models.emt.gsa_transport.two_phase_thermal_ti(
                    matrix_value=3.0,
                    inclusion_value=fluid_tc,
                    porosity=float(phi),
                    aspect_ratio=float(shape_cfg["alpha"]),
                    comparison="matrix",
                    symmetry_axis=(0.0, 0.0, 1.0),
                )

                lam_normal = result.lambda_parallel
                lam_bedding = result.lambda_perpendicular

                # lam_max = max(lam_normal, lam_bedding)
                # lam_min = min(lam_normal, lam_bedding)
                # ratio.append(lam_max / lam_min)
                ratio.append(lam_bedding)

            ax.plot(
                porosity_grid,
                ratio,
                linestyle=shape_cfg["linestyle"],
                linewidth=2.2,
                label=f"{fluid_label}, {shape_label}",
            )

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"Porosity $\phi$")
    ax.set_ylabel(r"$\lambda_{\max}/\lambda_{\min}$")
    ax.set_title(
        "Bayuk-style TC plot: ratio of maximal and minimal thermal conductivity\n"
        "solid = channels ($\\alpha=3$), dashed = disks ($\\alpha=0.01$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    save_figure(fig, "test_gsa_bayuk_style_tc_ratio_vs_porosity")


def plot_bayuk_style_tc_angle_dependence() -> None:
        # """
        # Directional thermal conductivity lambda(theta) in the plane normal
        # to the crack plane, for gas / oil / brine inclusions.

        # theta = 0 deg  -> normal to bedding / crack plane
        # theta = 90 deg -> in bedding / crack plane
        # """
        theta_deg = np.linspace(0.0, 189.0, 15)
        theta_rad = np.deg2rad(theta_deg)

        # choose one representative TI state
        phi = 0.10
        alpha = 0.01

        fluid_cases = {
            "Brine": 0.60,
            "Oil": 0.13,
            "Air": 0.025,
        }

        fig, ax = plt.subplots(figsize=(8.0, 5.8))

        for fluid_label, fluid_tc in fluid_cases.items():
            result = rockphysx.models.emt.gsa_transport.two_phase_thermal_ti(
                matrix_value=3.0,
                inclusion_value=fluid_tc,
                porosity=phi,
                aspect_ratio=alpha,
                comparison="matrix",
                symmetry_axis=(0.0, 0.0, 1.0),
            )

            T = result.tensor
            lambda_theta = []

            for th in theta_rad:
                n = np.array([np.sin(th), 0.0, np.cos(th)], dtype=float)
                lam_dir = float(n @ T @ n)
                lambda_theta.append(lam_dir)

            ax.plot(theta_deg, lambda_theta, linewidth=2.2, label=fluid_label)

        ax.set_xlabel(r"Angle $\theta$ (deg)")
        ax.set_ylabel(r"Directional thermal conductivity $\lambda(\theta)$ (W m$^{-1}$ K$^{-1}$)")
        ax.set_title(
            r"Bayuk-style TC angular plot: $\lambda(\theta)$ in the plane normal to crack plane"
            "\n"
            rf"fixed porosity $\phi={phi:.2f}$, fixed aspect ratio $\alpha={alpha:.2f}$"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        fig.tight_layout()
        save_figure(fig, "test_gsa_bayuk_style_tc_angle_dependence")

def main() -> None:
    # plot_example1_saturation_curves()
    # plot_example2_response_maps()
    #plot_example3_ti_anisotropy()

    plot_bayuk_style_tc_ratio_vs_porosity()
    # plot_bayuk_style_tc_angle_dependence()
    plt.show()


if __name__ == "__main__":
    main()
