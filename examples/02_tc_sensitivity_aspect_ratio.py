"""
Example 2
---------
How sensitive is effective thermal conductivity to pore aspect ratio alpha?

Forward problem:
    lambda* = F_lambda(phi, X_m, X_fl, theta)

Mixed pore-fluid thermal conductivity:
    lambda_mix = prod_i lambda_i ** v_i

Design of the sensitivity study
-------------------------------
- x-axis: aspect ratio alpha on a log10 scale
- y-axis: effective thermal conductivity
- panels: dry, brine, oil, gas-brine mixture
- curves: several porosity values

This example isolates the role of pore shape while still showing its interaction
with porosity and pore-fluid state.
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


POROSITY_LEVELS = [0.05, 0.15, 0.25, 0.35]
ALPHA_GRID = np.logspace(-4, 1, 300)


def save_figure(fig: plt.Figure, stem: str) -> None:
    outdir = Path("figures/thermal")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")


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
        notes="Example 2: TC sensitivity to aspect ratio alpha.",
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for ax, (title, panel_sample, saturation) in zip(axes.flat, panel_definitions(base_sample), strict=True):
        for phi in POROSITY_LEVELS:
            values = []
            for alpha in ALPHA_GRID:
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
                        model="gsa",
                    )
                )

            ax.plot(
                ALPHA_GRID,
                values,
                linewidth=2.0,
                label=rf"$\phi = {phi:.2f}$",
            )

        ax.set_xscale("log")
        ax.axvline(1.0, linestyle=":", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel(r"Aspect ratio $\alpha$")
        ax.set_ylabel("Thermal conductivity, W/(m·K)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Porosity",
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Sensitivity of effective thermal conductivity to aspect ratio alpha\n"
        "GSA model, synthetic sandstone",
        y=1.02,
    )

    save_figure(fig, "02_tc_sensitivity_aspect_ratio")
    plt.show()


if __name__ == "__main__":
    main()
