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

MODEL = "gsa"
EPSILON = 0.01


def build_representative_sample() -> SampleDescription:
    """Illustrative representative carbonate sample.

    Replace these placeholder values with the Timan-Pechora representative or
    sample-specific values used in the thesis analysis.
    """
    calcite = MineralPhase(
        name="calcite",
        volume_fraction=0.80,
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.71,
        thermal_conductivity_wmk=3.59,
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
        porosity=0.12,
        minerals=[calcite, dolomite],
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
            aspect_ratio=0.06,
            connectivity=0.75,
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
        sat: solver.predict("thermal_conductivity", sample, sat, model=MODEL) * factor
        for sat, factor in {
            SaturationState.DRY: 0.995,
            SaturationState.BRINE: 1.003,
            SaturationState.OIL: 1.001,
        }.items()
    }



def plot_sensitivity_bars(sample: SampleDescription, output_dir: Path) -> None:
    saturations = [SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL]
    results = compute_local_sensitivities(sample, saturations, epsilon=EPSILON, model=MODEL)

    parameter_order = ["porosity", "matrix_conductivity", "fluid_conductivity", "aspect_ratio"]
    x = np.arange(len(parameter_order))
    width = 0.24

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
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\phi$", r"$\lambda_M$", r"$\lambda_f$", r"$\alpha$"], fontsize=12)
    ax.set_ylabel(r"Normalized sensitivity $S_p = \partial \ln \lambda^* / \partial \ln p$")
    ax.set_title("Timan-Pechora carbonates: local normalized thermal sensitivities")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "07_timan_pechora_sensitivities.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "07_timan_pechora_sensitivities.pdf", bbox_inches="tight")



def plot_misfit_maps(sample: SampleDescription, measured: dict[SaturationState, float], output_dir: Path) -> None:
    alpha_grid = np.logspace(-3, 0, 120)
    lambda_m_grid = np.linspace(3.6, 4.8, 120)

    observation_sets = {
        "dry": build_thermal_observations({SaturationState.DRY: measured[SaturationState.DRY]}, model=MODEL),
        "brine": build_thermal_observations({SaturationState.BRINE: measured[SaturationState.BRINE]}, model=MODEL),
        "kerosene": build_thermal_observations({SaturationState.OIL: measured[SaturationState.OIL]}, model=MODEL),
        "combined": build_thermal_observations(measured, model=MODEL),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
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
        contour = ax.contourf(
            grid.alpha_values,
            grid.matrix_conductivity_values,
            grid.misfit,
            levels=25,
        )
        ax.set_xscale("log")
        ax.set_title(title.capitalize())
        ax.set_xlabel(r"Aspect ratio $\alpha$")
        ax.set_ylabel(r"Matrix conductivity $\lambda_M$ (W/(m·K))")

    cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Normalized least-squares misfit")
    fig.suptitle(r"Timan-Pechora carbonates: misfit maps in $(\alpha, \lambda_M)$ space", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "07_timan_pechora_misfit_maps.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "07_timan_pechora_misfit_maps.pdf", bbox_inches="tight")



def plot_profile_misfits(sample: SampleDescription, measured: dict[SaturationState, float], output_dir: Path) -> None:
    observations = build_thermal_observations(measured, model=MODEL)
    alpha_grid = np.logspace(-3, 0, 200)
    lambda_m_values = [3.8, 4.15, 4.5]
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
    ax.set_title("Timan-Pechora carbonates: profile misfit versus aspect ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "07_timan_pechora_profile_misfit.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "07_timan_pechora_profile_misfit.pdf", bbox_inches="tight")



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
