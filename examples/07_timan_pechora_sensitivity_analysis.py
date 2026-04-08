from __future__ import annotations

"""
Template sensitivity-analysis workflow for Timan-Pechora carbonates.

The numerical values in this script are illustrative placeholders so the example
runs out of the box. Replace the representative sample properties, lambda_M
interval, and measured thermal conductivities with the actual values used in the
thesis subsection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.analysis.thermal_sensitivity import (
    build_thermal_observations,
    compute_alpha_lambda_misfit_grid,
    compute_local_sensitivities,
    profile_misfit_over_alpha,
)
from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState

# from rockphysx.models.emt.sca_thermal import sca_effective_conductivity
# from rockphysx.models.emt.bruggeman import bruggeman_isotropic

# MODEL = "gsa"
# MODEL = "sca"
MODEL = "sca"
EPSILON = 0.01


def build_representative_sample() -> SampleDescription:
    """Illustrative representative carbonate sample.

    Replace these placeholder values with the Timan-Pechora representative or
    sample-specific values used in the thesis analysis.
    """
    calcite = MineralPhase(
        name="calcite",
        volume_fraction=1.0,
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.71,
        thermal_conductivity_wmk=3.00,
        electrical_conductivity_sm=1e-10,
    )
    dolomite = MineralPhase(
        name="dolomite",
        volume_fraction=0.20,
        bulk_modulus_gpa=94.0,
        shear_modulus_gpa=45.0,
        density_gcc=2.87,
        thermal_conductivity_wmk=5.51,
        electrical_conductivity_sm=1e-10,
    )

    return SampleDescription(
        name="timan_pechora_representative",
        porosity=0.105,
        minerals=[calcite],
        fluids={
            SaturationState.DRY: FluidPhase.air(),
            SaturationState.BRINE: FluidPhase.brine(),
            SaturationState.OIL: FluidPhase.oil(),
        },
        matrix=MatrixProperties(
            bulk_modulus_gpa=74.8,
            shear_modulus_gpa=34.6,
            density_gcc=2.74,
            thermal_conductivity_wmk=4.15,
            electrical_conductivity_sm=1e-10,
        ),
        microstructure=MicrostructureParameters(
            aspect_ratio=0.1,
            connectivity=1.0,
            orientation="isotropic",
            topology="vuggy-intercrystalline",
        ),
        notes="Replace placeholder values with Timan-Pechora representative inputs.",
    )


def build_placeholder_measurements(sample: SampleDescription) -> dict[SaturationState, float]:
    """Illustrative measurements.

    Replace with measured lambda* values for dry, brine, and kerosene states.
    """
    # Slightly perturbed around a plausible GSA forward response.
    from rockphysx.forward.solver import ForwardSolver

    solver = ForwardSolver()
    return {
        sat: solver.predict("thermal_conductivity", sample, sat, model=MODEL)
        for sat in (
            SaturationState.DRY,
            SaturationState.BRINE,
            SaturationState.OIL,
        )
    }

def plot_sensitivity_bars(sample: SampleDescription, output_dir: Path) -> None:
    saturations = [SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL]
    results = compute_local_sensitivities(sample, saturations, epsilon=EPSILON, model=MODEL)

    parameter_order = ["porosity", "matrix_conductivity", "fluid_conductivity", "aspect_ratio"]
    x = np.arange(len(parameter_order))
    width = 0.24

    bar_colors = {
        SaturationState.DRY: "red",
        SaturationState.BRINE: "blue",
        SaturationState.OIL: "green",
    }


    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for idx, saturation in enumerate(saturations):
        subset = [
            next(r for r in results if r.saturation == saturation and r.parameter == p)
            for p in parameter_order
        ]
        ax.bar(
            x + (idx - 1) * width,
            [item.normalized_sensitivity for item in subset],
            width=width,
            label=saturation.value,
            color=bar_colors[saturation],
            alpha=0.5,
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\phi$", r"$\lambda_M$", r"$\lambda_f$", r"$\alpha$"], fontsize=12)
    ax.set_ylabel(r"Normalized sensitivity coefficient $S_p$")
    ax.set_title("Timan-Pechora carbonates: local normalized thermal sensitivities")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "07_timan_pechora_sensitivities.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "07_timan_pechora_sensitivities.png", bbox_inches="tight")

def plot_misfit_maps(sample: SampleDescription, measured: dict[SaturationState, float], output_dir: Path) -> None:
    alpha_grid = np.logspace(-2, 0, 120)
    lambda_m_grid = np.linspace(2.7, 3.6, 120)
    # lambda_m_grid = np.linspace(2.90, 3.05, 120)
 

    observation_sets = {
        "dry": build_thermal_observations({SaturationState.DRY: measured[SaturationState.DRY]}, model=MODEL),
        "brine": build_thermal_observations({SaturationState.BRINE: measured[SaturationState.BRINE]}, model=MODEL),
        "kerosene": build_thermal_observations({SaturationState.OIL: measured[SaturationState.OIL]}, model=MODEL),
        "combined": build_thermal_observations(measured, model=MODEL),
    }

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.18, hspace=0.28)

    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])

    cax = fig.add_subplot(gs[:, 2])

    for ax, (title, observations) in zip(axes.flat, observation_sets.items(), strict=True):
        grid = compute_alpha_lambda_misfit_grid(
            sample,
            observations,
            alpha_grid,
            lambda_m_grid,
            normalized=True,
            power=2,
            model=MODEL,
        )

        levels = np.linspace(np.min(grid.misfit), np.max(grid.misfit), 25)

        contour = ax.contourf(
            grid.alpha_values,
            grid.matrix_conductivity_values,
            grid.misfit,
            levels=levels,
        )

        contour_lines = ax.contour(
            grid.alpha_values,
            grid.matrix_conductivity_values,
            grid.misfit,
            levels=levels[::2],   # every second level for clarity
            colors="black",
            linewidths=0.6,
            alpha=0.7,
        )

        ax.clabel(
            contour_lines,
            fmt="%.2f",
            fontsize=8,
            inline=True,
        )

    
        ax.set_xscale("log")
        ax.set_title(title.capitalize())
        ax.set_xlabel(r"Aspect ratio $\alpha$")
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel(r"Matrix conductivity $\lambda_M$ (W/(m$\cdot$K))")
        else:
            ax.set_ylabel("")

    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label("Normalized least-squares misfit")

    # fig.suptitle(
    #     r"Timan-Pechora carbonates: misfit maps in $(\alpha, \lambda_M)$ space",
    #     y=0.97,
    # )

    fig.savefig(output_dir / "07_timan_pechora_misfit_maps.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "07_timan_pechora_misfit_maps.png", bbox_inches="tight")



def plot_profile_misfits(sample: SampleDescription, measured: dict[SaturationState, float], output_dir: Path) -> None:
    observations = build_thermal_observations(measured, model=MODEL)
    alpha_grid = np.logspace(-2, 1, 200)
    lambda_m_values = [2.7, 3.0, 3.6]
    # lambda_m_values = [2.90, 2.98, 3.05]
    profiles = profile_misfit_over_alpha(
        sample,
        observations,
        alpha_grid,
        lambda_m_values,
        normalized=True,
        power=2,
        model=MODEL,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for lambda_m, profile in profiles.items():
        ax.plot(alpha_grid, profile, linewidth=2.0, label=rf"$\lambda_M={lambda_m:.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Aspect ratio $\alpha$")
    ax.set_ylabel("Normalized least-squares misfit")
    # ax.set_title("Timan-Pechora carbonates: profile misfit versus aspect ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "07_timan_pechora_profile_misfit.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "07_timan_pechora_profile_misfit.png", bbox_inches="tight")



def main() -> None:
    output_dir = Path("figures/thermal")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = build_representative_sample()
    measured = build_placeholder_measurements(sample)

    plot_sensitivity_bars(sample, output_dir)
    plot_misfit_maps(sample, measured, output_dir)
    plot_profile_misfits(sample, measured, output_dir)
    plt.show()


if __name__ == "__main__":
    main()
