"""
Example 1
---------
How do pore fluid and pore shape control effective thermal conductivity of sandstone?

Forward problem:
    lambda* = F_lambda(phi, X_m, X_fl, theta)

Mixed pore-fluid thermal conductivity:
    lambda_mix = prod_i lambda_i ** v_i

Notes
-----
Aspect-ratio lines are defined on a log10 scale:
    alpha = 10^-4, 10^-3, 10^-2, 10^-1, 10^0, 10^1

The original request included "0", but zero cannot appear on a logarithmic
aspect-ratio scale, so it is interpreted here as 10^0 = 1.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.core.parameters import FluidPhase, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver
from rockphysx.models.fluids.mixing import mix_fluid_phases


ALPHA_LINES = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], dtype=float)


def alpha_label(alpha: float) -> str:
    exponent = int(round(np.log10(alpha)))
    return rf"$10^{{{exponent}}}$"


def save_figure(fig: plt.Figure, stem: str) -> None:
    outdir = Path("figures/thermal")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def build_sandstone_sample(porosity: float = 0.18) -> SampleDescription:
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
        porosity=porosity,
        minerals=[quartz, clay],
        fluids={
            SaturationState.DRY: FluidPhase.air(),
            SaturationState.BRINE: FluidPhase.brine(),
            SaturationState.OIL: FluidPhase.oil(),
        },
        microstructure=MicrostructureParameters(
            aspect_ratio=0.1,
            connectivity=0.65,
            orientation="isotropic",
            topology="intergranular",
        ),
        notes="Example 1: TC sensitivity to porosity, pore fluid, and pore shape.",
    )


def make_gas_brine_mixture() -> FluidPhase:
    return mix_fluid_phases(
        [FluidPhase.air(), FluidPhase.brine()],
        [0.5, 0.5],
        name="gas-brine 50/50",
    )


def panel_definitions(base_sample: SampleDescription):
    gas_brine = make_gas_brine_mixture()

    mixed_fluids = dict(base_sample.fluids)
    mixed_fluids[SaturationState.BRINE] = gas_brine
    mixed_sample = replace(base_sample, fluids=mixed_fluids)

    return [
        ("Dry (air)", base_sample, SaturationState.DRY),
        ("Brine", base_sample, SaturationState.BRINE),
        ("Oil", base_sample, SaturationState.OIL),
        ("Gas-brine mixture (50/50)", mixed_sample, SaturationState.BRINE),
    ]


def main() -> None:
    solver = ForwardSolver()
    base_sample = build_sandstone_sample()

    porosity_grid = np.linspace(0.0, 0.35, 150)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for ax, (title, panel_sample, saturation) in zip(axes.flat, panel_definitions(base_sample), strict=True):
        for alpha in ALPHA_LINES:
            values = []
            for phi in porosity_grid:
                varied_sample = replace(
                    panel_sample,
                    porosity=float(phi),
                    microstructure=replace(panel_sample.microstructure, aspect_ratio=float(alpha)),
                )
                values.append(
                    solver.predict(
                        "thermal_conductivity",
                        varied_sample,
                        saturation,
                        # model="gsa",
                        model="sca",
                    )
                )

            ax.plot(
                porosity_grid,
                values,
                linewidth=2.0,
                label=alpha_label(alpha),
            )

        ax.set_title(title)
        ax.set_xlabel("Porosity, fraction")
        ax.set_ylabel("Thermal conductivity, W/(m·K)")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=r"Aspect ratio $\alpha$",
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Sensitivity of effective thermal conductivity to porosity, pore fluid, and pore shape\n"
        "GSA model, synthetic sandstone",
        y=1.02,
    )

    save_figure(fig, "01_tc_sensitivity_porosity_saturation_shape")
    plt.show()


if __name__ == "__main__":
    main()