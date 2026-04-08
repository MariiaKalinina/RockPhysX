from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.core.parameters import FluidPhase, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.models.fluids.mixing import mix_fluid_phases
from rockphysx.models.transport.thermal_tensor import thermal_conductivity_tensor_gsa


ALPHA_GRID = np.logspace(-4, 1, 70)
ORIENTATION_GRID = np.linspace(0.0, 1.0, 50)


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


def compute_surfaces(sample: SampleDescription, saturation: SaturationState):
    matrix_tc = sample.matrix_properties.thermal_conductivity_wmk
    fluid_tc = sample.fluid_for(saturation).thermal_conductivity_wmk

    lambda_parallel = np.empty((len(ORIENTATION_GRID), len(ALPHA_GRID)), dtype=float)
    lambda_perpendicular = np.empty((len(ORIENTATION_GRID), len(ALPHA_GRID)), dtype=float)

    for i, s in enumerate(ORIENTATION_GRID):
        for j, alpha in enumerate(ALPHA_GRID):
            result = thermal_conductivity_tensor_gsa(
                matrix_tc,
                fluid_tc,
                sample.porosity,
                aspect_ratio=float(alpha),
                orientation_order=float(s),
            )
            lambda_parallel[i, j] = result.lambda_parallel
            lambda_perpendicular[i, j] = result.lambda_perpendicular

    return lambda_parallel, lambda_perpendicular


def apply_log_alpha_ticks(ax) -> None:
    tick_positions = [-4, -3, -2, -1, 0, 1]
    tick_labels = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def make_surface_figure(base_sample: SampleDescription, *, component: str, stem: str) -> None:
    fig = plt.figure(figsize=(14, 10))
    x_mesh, y_mesh = np.meshgrid(np.log10(ALPHA_GRID), ORIENTATION_GRID)

    for index, (title, sample, saturation) in enumerate(panel_definitions(base_sample), start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        lambda_parallel, lambda_perpendicular = compute_surfaces(sample, saturation)

        if component == "parallel":
            z_mesh = lambda_parallel
            z_label = r"$\lambda_{\parallel}$, W/(m·K)"
            title_suffix = r"$\lambda_{\parallel}$"
        else:
            z_mesh = lambda_perpendicular
            z_label = r"$\lambda_{\perp}$, W/(m·K)"
            title_suffix = r"$\lambda_{\perp}$"

        ax.plot_surface(x_mesh, y_mesh, z_mesh, rstride=1, cstride=1, linewidth=0.0, antialiased=True, alpha=0.95)
        ax.set_title(f"{title}\n{title_suffix}")
        ax.set_xlabel(r"log$_{10}(\alpha)$")
        ax.set_ylabel("Orientation order S")
        ax.set_zlabel(z_label)
        apply_log_alpha_ticks(ax)

    fig.suptitle(
        "Tensor GSA sensitivity study: preferred pore orientation and pore shape\n"
        + (r"Surface of $\lambda_{\parallel}$" if component == "parallel" else r"Surface of $\lambda_{\perp}$"),
        y=0.98,
    )
    save_figure(fig, stem)


def main() -> None:
    base_sample = build_sandstone_sample()
    make_surface_figure(base_sample, component="parallel", stem="04_tc_anisotropy_orientation_surface_parallel")
    make_surface_figure(base_sample, component="perpendicular", stem="04_tc_anisotropy_orientation_surface_perpendicular")
    plt.show()


if __name__ == "__main__":
    main()
