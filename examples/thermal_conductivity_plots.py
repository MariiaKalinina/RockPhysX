from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from thesis_rp.core.parameters import FluidPhase, MicrostructureParameters, MineralPhase
from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState
from thesis_rp.utils.plotting import (
    plot_alpha_calibration_misfit,
    plot_saturation_comparison,
    plot_thermal_model_comparison,
    plot_thermal_vs_aspect_ratio,
    save_figure,
)


def build_sample() -> SampleDescription:
    quartz = MineralPhase(
        name="quartz",
        volume_fraction=0.85,
        bulk_modulus_gpa=37.0,
        shear_modulus_gpa=44.0,
        density_gcc=2.65,
        thermal_conductivity_wmk=6.5,
        electrical_conductivity_sm=1e-12,
    )
    clay = MineralPhase(
        name="clay",
        volume_fraction=0.15,
        bulk_modulus_gpa=20.0,
        shear_modulus_gpa=7.0,
        density_gcc=2.58,
        thermal_conductivity_wmk=2.0,
        electrical_conductivity_sm=1e-8,
    )

    return SampleDescription(
        name="synthetic_sandstone",
        porosity=0.18,
        minerals=[quartz, clay],
        fluids={
            SaturationState.DRY: FluidPhase.air(),
            SaturationState.BRINE: FluidPhase.brine(),
            SaturationState.OIL: FluidPhase.oil(),
        },
        microstructure=MicrostructureParameters(
            aspect_ratio=0.12,
            connectivity=0.65,
            orientation="isotropic",
            topology="intergranular",
        ),
    )


def main() -> None:
    sample = build_sample()

    porosity = np.linspace(0.02, 0.35, 80)
    alpha = np.linspace(0.01, 0.8, 120)

    ax1 = plot_thermal_model_comparison(
        sample,
        SaturationState.DRY,
        porosity,
        models=("maxwell", "bruggeman", "gsa"),
    )
    save_figure(ax1.figure, "figures/thermal_model_comparison_dry.png")

    ax2 = plot_saturation_comparison(
        sample,
        porosity,
        model="gsa",
    )
    save_figure(ax2.figure, "figures/thermal_saturation_comparison.png")

    ax3 = plot_thermal_vs_aspect_ratio(
        sample,
        SaturationState.BRINE,
        alpha,
        model="gsa",
    )
    save_figure(ax3.figure, "figures/thermal_vs_aspect_ratio_brine.png")

    measured_thermal = {
        SaturationState.DRY: 2.55,
        SaturationState.BRINE: 3.35,
        SaturationState.OIL: 2.95,
    }
    ax4 = plot_alpha_calibration_misfit(
        sample,
        measured_thermal,
        alpha_grid=alpha,
    )
    save_figure(ax4.figure, "figures/alpha_calibration_misfit.png")

    plt.show()


if __name__ == "__main__":
    main()