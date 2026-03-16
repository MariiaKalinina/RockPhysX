from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.core.parameters import FluidPhase, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver
from rockphysx.inverse.objective import Observation, weighted_misfit
from rockphysx.inverse.parametrization_alpha import calibrate_constant_aspect_ratio


TRUE_ALPHA = 1e-2
ALPHA_GRID = np.logspace(-4, 1, 500)
BOUNDS = (1e-4, 10.0)


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
            aspect_ratio=TRUE_ALPHA,
            connectivity=0.65,
            orientation="isotropic",
            topology="intergranular",
        ),
    )


def build_synthetic_observations(sample: SampleDescription, solver: ForwardSolver) -> dict[SaturationState, float]:
    truth_sample = replace(sample, microstructure=replace(sample.microstructure, aspect_ratio=TRUE_ALPHA))
    return {
        sat: solver.predict("thermal_conductivity", truth_sample, sat, model="gsa")
        for sat in (SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL)
    }


def make_observation_sets(measured: dict[SaturationState, float]) -> dict[str, list[Observation]]:
    dry = [Observation("thermal_conductivity", SaturationState.DRY, measured[SaturationState.DRY], 1.0, "gsa")]
    brine = [Observation("thermal_conductivity", SaturationState.BRINE, measured[SaturationState.BRINE], 1.0, "gsa")]
    oil = [Observation("thermal_conductivity", SaturationState.OIL, measured[SaturationState.OIL], 1.0, "gsa")]
    joint = dry + brine + oil
    return {
        "Dry only": dry,
        "Brine only": brine,
        "Oil only": oil,
        "Joint dry+brine+oil": joint,
    }


def main() -> None:
    solver = ForwardSolver()
    sample = build_sandstone_sample()
    measured = build_synthetic_observations(sample, solver)
    observation_sets = make_observation_sets(measured)

    fig, ax = plt.subplots(figsize=(8, 5.2))

    for label, observations in observation_sets.items():
        misfit_values = [weighted_misfit(alpha, sample, observations, solver=solver) for alpha in ALPHA_GRID]
        fit = calibrate_constant_aspect_ratio(sample, observations, bounds=BOUNDS, solver=solver)
        ax.plot(ALPHA_GRID, misfit_values, linewidth=2.0, label=f"{label} (best = {fit.alpha_hat:.2e})")

    ax.axvline(TRUE_ALPHA, linestyle="--", linewidth=1.5, label=rf"true $\alpha$ = {TRUE_ALPHA:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Aspect ratio $\alpha$")
    ax.set_ylabel("Weighted least-squares misfit")
    ax.set_title("Identifiability of effective aspect ratio from thermal conductivity data")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    save_figure(fig, "03_tc_inverse_identifiability")
    plt.show()


if __name__ == "__main__":
    main()