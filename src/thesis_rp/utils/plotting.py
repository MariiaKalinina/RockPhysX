from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState
from thesis_rp.forward.solver import ForwardSolver
from thesis_rp.inverse.objective import Observation
from thesis_rp.inverse.parametrization_alpha import calibrate_constant_aspect_ratio


def plot_thermal_model_comparison(
    sample: SampleDescription,
    saturation: SaturationState,
    porosity_grid: Sequence[float] | np.ndarray,
    *,
    models: Sequence[str] = ("maxwell", "bruggeman", "gsa"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot thermal conductivity versus porosity for several EMT models."""
    solver = ForwardSolver()
    porosity = np.asarray(porosity_grid, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for model in models:
        values = []
        for phi in porosity:
            varied_sample = replace(sample, porosity=float(phi))
            values.append(
                solver.predict(
                    "thermal_conductivity",
                    varied_sample,
                    saturation,
                    model=model,
                )
            )
        ax.plot(porosity, values, label=model)

    ax.set_xlabel("Porosity, fraction")
    ax.set_ylabel("Thermal conductivity, W/(m·K)")
    ax.set_title(f"Thermal conductivity model comparison ({saturation.value})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return ax


def plot_thermal_vs_aspect_ratio(
    sample: SampleDescription,
    saturation: SaturationState,
    aspect_ratio_grid: Sequence[float] | np.ndarray,
    *,
    model: str = "gsa",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot thermal conductivity versus aspect ratio alpha."""
    solver = ForwardSolver()
    alpha = np.asarray(aspect_ratio_grid, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    values = []
    for a in alpha:
        micro = replace(sample.microstructure, aspect_ratio=float(a))
        values.append(
            solver.predict(
                "thermal_conductivity",
                sample,
                saturation,
                microstructure=micro,
                model=model,
            )
        )

    ax.plot(alpha, values, label=f"{model}, {saturation.value}")
    ax.set_xlabel("Aspect ratio α")
    ax.set_ylabel("Thermal conductivity, W/(m·K)")
    ax.set_title("Thermal conductivity sensitivity to aspect ratio")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return ax


def plot_saturation_comparison(
    sample: SampleDescription,
    porosity_grid: Sequence[float] | np.ndarray,
    *,
    model: str = "gsa",
    saturations: Sequence[SaturationState] = (
        SaturationState.DRY,
        SaturationState.BRINE,
        SaturationState.OIL,
    ),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot thermal conductivity versus porosity for several saturation states."""
    solver = ForwardSolver()
    porosity = np.asarray(porosity_grid, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for saturation in saturations:
        values = []
        for phi in porosity:
            varied_sample = replace(sample, porosity=float(phi))
            values.append(
                solver.predict(
                    "thermal_conductivity",
                    varied_sample,
                    saturation,
                    model=model,
                )
            )
        ax.plot(porosity, values, label=saturation.value)

    ax.set_xlabel("Porosity, fraction")
    ax.set_ylabel("Thermal conductivity, W/(m·K)")
    ax.set_title(f"Thermal conductivity vs porosity ({model})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return ax


def plot_alpha_calibration_misfit(
    sample: SampleDescription,
    measured_thermal: dict[SaturationState, float],
    *,
    alpha_grid: Sequence[float] | np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot weighted least-squares misfit as a function of aspect ratio alpha."""
    solver = ForwardSolver()
    alpha_values = np.asarray(alpha_grid, dtype=float)

    observations = [
        Observation(
            property_name="thermal_conductivity",
            saturation=sat,
            measured_value=value,
            weight=1.0,
            model="gsa",
        )
        for sat, value in measured_thermal.items()
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    misfit = []
    for alpha in alpha_values:
        total = 0.0
        micro = replace(sample.microstructure, aspect_ratio=float(alpha))
        for obs in observations:
            pred = solver.predict(
                obs.property_name,
                sample,
                obs.saturation,
                microstructure=micro,
                model=obs.model,
            )
            total += obs.weight * (pred - obs.measured_value) ** 2
        misfit.append(total)

    result = calibrate_constant_aspect_ratio(sample, observations, solver=solver)

    ax.plot(alpha_values, misfit, label="misfit")
    ax.axvline(result.alpha_hat, linestyle="--", label=f"best α = {result.alpha_hat:.4f}")
    ax.set_xlabel("Aspect ratio α")
    ax.set_ylabel("Weighted least-squares misfit")
    ax.set_title("Thermal-only inversion objective")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return ax


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 300) -> None:
    """Save a figure to disk with publication-friendly defaults."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")