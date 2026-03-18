from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.inverse.bayesian_thermal import (
    ThermalDatum,
    compute_bayesian_posterior_grid,
    marginal_alpha,
    marginal_lambda_m,
    predict_tc_for_datum,
)
from rockphysx.utils.data_loading import (
    read_timan_pechora_tc_excel,
    write_posterior_summary_excel,
)

MODEL = "gsa"


# ---------------------------------------------------------------------
# Scientific plotting style
# ---------------------------------------------------------------------
def apply_scientific_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "grid.color": "#9a9a9a",
            "legend.frameon": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )


def style_axis(ax: plt.Axes, *, logx: bool = False, minor_grid: bool = False) -> None:
    if logx:
        ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    if minor_grid:
        ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)
    ax.tick_params(direction="out", length=4, width=0.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------
# Sample/data helpers
# ---------------------------------------------------------------------
def build_sample_from_row(sample_name: str, porosity: float) -> SampleDescription:
    calcite = MineralPhase(
        name="calcite",
        volume_fraction=1.00,
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.71,
        thermal_conductivity_wmk=3.59,
        electrical_conductivity_sm=1e-10,
    )
    # dolomite = MineralPhase(
    #     name="dolomite",
    #     volume_fraction=0.20,
    #     bulk_modulus_gpa=94.0,
    #     shear_modulus_gpa=45.0,
    #     density_gcc=2.87,
    #     thermal_conductivity_wmk=5.51,
    #     electrical_conductivity_sm=1e-10,
    # )

    return SampleDescription(
        name=sample_name,
        porosity=porosity,
        minerals=[calcite],
        fluids={},
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
    )


def fluid_tc_by_state() -> dict[str, float]:
    return {
        "dry": FluidPhase.air().thermal_conductivity_wmk,
        "oil": FluidPhase.oil().thermal_conductivity_wmk,
        "brine_0_6": FluidPhase.brine().thermal_conductivity_wmk,
        "brine_6": FluidPhase.brine().thermal_conductivity_wmk,
        "brine_60": FluidPhase.brine().thermal_conductivity_wmk,
        "brine_180": FluidPhase.brine().thermal_conductivity_wmk,
    }


def build_thermal_data_for_sample(df_long: pd.DataFrame, sample_name: str) -> tuple[SampleDescription, list[ThermalDatum]]:
    subset = df_long[df_long["sample"] == sample_name].copy()
    if subset.empty:
        raise ValueError(f"Sample {sample_name!r} not found in Excel data.")

    porosity = float(subset["porosity"].iloc[0])
    sample = build_sample_from_row(sample_name, porosity)

    fluid_map = fluid_tc_by_state()
    data = [
        ThermalDatum(
            label=row["state"],
            measured_tc=float(row["measured_tc"]),
            fluid_conductivity=float(fluid_map[row["state"]]),
            relative_sigma=0.03,
            weight=1.0,
        )
        for _, row in subset.iterrows()
    ]
    return sample, data


# ---------------------------------------------------------------------
# Posterior plotting
# ---------------------------------------------------------------------
def compute_hpd_like_levels(posterior: np.ndarray, probs: tuple[float, ...] = (0.50, 0.80, 0.95)) -> list[float]:
    """
    Approximate HPD contour levels from a discrete posterior grid.
    """
    flat = posterior.ravel()
    order = np.argsort(flat)[::-1]
    sorted_post = flat[order]
    csum = np.cumsum(sorted_post)
    csum /= csum[-1]

    levels = []
    for p in probs:
        idx = np.searchsorted(csum, p)
        idx = min(idx, len(sorted_post) - 1)
        levels.append(sorted_post[idx])

    # contour wants increasing levels
    return sorted(set(levels))


def plot_joint_and_marginals(result, output_dir: Path, stem: str) -> None:
    fig = plt.figure(figsize=(11.0, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1.7, 1.0])

    ax_joint = fig.add_subplot(gs[:, 0])
    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_lambda = fig.add_subplot(gs[1, 1])

    X, Y = np.meshgrid(result.log10_alpha_values, result.matrix_conductivity_values)

    levels = compute_hpd_like_levels(result.posterior, probs=(0.50, 0.80, 0.95))
    if len(levels) >= 2:
        cs = ax_joint.contour(
            X,
            Y,
            result.posterior,
            levels=levels,
            colors=["#1f77b4", "#2ca02c", "#d62728"][: len(levels)],
            linewidths=[1.8, 1.6, 1.4][: len(levels)],
        )
        fmt = {level: label for level, label in zip(levels, ["50%", "80%", "95%"][-len(levels):])}
        ax_joint.clabel(cs, inline=True, fontsize=9, fmt=fmt)

    ax_joint.plot(
        result.map_log10_alpha,
        result.map_matrix_conductivity,
        marker="*",
        color="crimson",
        markersize=11,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
    )

    ax_joint.set_xlabel(r"$\log_{10}\alpha$")
    ax_joint.set_ylabel(r"$\lambda_M$ (W/(m$\cdot$K))")
    ax_joint.set_title("Joint posterior")
    style_axis(ax_joint)
    add_panel_label(ax_joint, "(a)")

    p_alpha = marginal_alpha(result)
    ax_alpha.plot(result.log10_alpha_values, p_alpha, color="#1f77b4", linewidth=2.2)
    ax_alpha.axvline(result.map_log10_alpha, color="crimson", linestyle="--", linewidth=1.0)
    ax_alpha.fill_between(result.log10_alpha_values, 0, p_alpha, color="#1f77b4", alpha=0.15)
    ax_alpha.set_xlabel(r"$\log_{10}\alpha$")
    ax_alpha.set_ylabel("Relative posterior density")
    ax_alpha.set_title(r"Marginal posterior of $\alpha$")
    style_axis(ax_alpha)
    add_panel_label(ax_alpha, "(b)")

    p_lm = marginal_lambda_m(result)
    ax_lambda.plot(result.matrix_conductivity_values, p_lm, color="#2ca02c", linewidth=2.2)
    ax_lambda.axvline(result.map_matrix_conductivity, color="crimson", linestyle="--", linewidth=1.0)
    ax_lambda.fill_between(result.matrix_conductivity_values, 0, p_lm, color="#2ca02c", alpha=0.15)
    ax_lambda.set_xlabel(r"$\lambda_M$ (W/(m$\cdot$K))")
    ax_lambda.set_ylabel("Relative posterior density")
    ax_lambda.set_title(r"Marginal posterior of $\lambda_M$")
    style_axis(ax_lambda)
    add_panel_label(ax_lambda, "(c)")

    save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------
# Posterior predictive plot
# ---------------------------------------------------------------------
def draw_posterior_samples(result, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    flat = result.posterior.ravel()
    idx = np.random.choice(len(flat), size=n, p=flat)
    iy, ix = np.unravel_index(idx, result.posterior.shape)
    sampled_log10_alpha = result.log10_alpha_values[ix]
    sampled_lambda_m = result.matrix_conductivity_values[iy]
    return sampled_log10_alpha, sampled_lambda_m


def plot_posterior_predictive(sample: SampleDescription, data: list[ThermalDatum], result, output_dir: Path, stem: str) -> None:
    sampled_log10_alpha, sampled_lambda_m = draw_posterior_samples(result, n=500)

    labels = [d.label for d in data]
    measured = np.array([d.measured_tc for d in data], dtype=float)

    pred_samples = []
    for u, lm in zip(sampled_log10_alpha, sampled_lambda_m, strict=True):
        alpha = 10.0 ** u
        pred = [
            predict_tc_for_datum(
                sample,
                d,
                aspect_ratio=alpha,
                matrix_conductivity=float(lm),
                model=MODEL,
            )
            for d in data
        ]
        pred_samples.append(pred)

    pred_samples = np.asarray(pred_samples, dtype=float)
    q05 = np.quantile(pred_samples, 0.05, axis=0)
    q50 = np.quantile(pred_samples, 0.50, axis=0)
    q95 = np.quantile(pred_samples, 0.95, axis=0)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

    ax.errorbar(
        x,
        q50,
        yerr=np.vstack([q50 - q05, q95 - q50]),
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.5,
        capsize=4,
        label="Posterior predictive median and 90% interval",
    )
    ax.scatter(x, measured, color="crimson", marker="s", s=42, zorder=5, label="Measured thermal conductivity")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Thermal conductivity $\lambda^*$ (W/(m$\cdot$K))")
    ax.set_title("Posterior predictive check")
    style_axis(ax)
    ax.legend(loc="best")
    add_panel_label(ax, "(d)")

    save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------
# Experimental distribution plot from Excel
# ---------------------------------------------------------------------
def plot_state_distribution(df_long: pd.DataFrame, output_dir: Path, stem: str) -> None:
    states_order = ["dry", "oil", "brine_0_6", "brine_6", "brine_60", "brine_180"]
    data_by_state = [df_long.loc[df_long["state"] == s, "measured_tc"].dropna().values for s in states_order]
    data_by_state = [d for d in data_by_state if len(d) > 0]
    labels = [s for s in states_order if len(df_long.loc[df_long["state"] == s]) > 0]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)

    parts = ax.violinplot(
        data_by_state,
        positions=np.arange(1, len(data_by_state) + 1),
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor("#6baed6")
        body.set_edgecolor("black")
        body.set_alpha(0.35)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.2)

    # jittered points
    rng = np.random.default_rng(42)
    for i, values in enumerate(data_by_state, start=1):
        jitter = rng.normal(0.0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=16,
            color="#1f77b4",
            alpha=0.65,
            edgecolors="none",
            zorder=4,
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Measured thermal conductivity $\lambda^*$ (W/(m$\cdot$K))")
    ax.set_title("Experimental thermal-conductivity distribution by state")
    style_axis(ax)
    add_panel_label(ax, "(e)")

    save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    apply_scientific_style()

    excel_path = Path("Tver_ver1.xlsx")  # replace with your actual workbook path
    output_dir = Path("figures/thermal")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_long = read_timan_pechora_tc_excel(excel_path, sheet_name=0)

    sample_name = str(df_long["sample"].iloc[0])
    sample, data = build_thermal_data_for_sample(df_long, sample_name)

    log10_alpha_values = np.linspace(-4.0, 1.0, 220)
    lambda_m_values = np.linspace(2.8, 4.8, 220)

    result = compute_bayesian_posterior_grid(
        sample,
        data,
        log10_alpha_values,
        lambda_m_values,
        log10_alpha_min=-4.0,
        log10_alpha_max=1.0,
        lambda_m_min=2.8,
        lambda_m_max=4.8,
        model=MODEL,
    )

    plot_joint_and_marginals(result, output_dir, "11_bayesian_timan_pechora_posterior")
    plot_posterior_predictive(sample, data, result, output_dir, "11_bayesian_timan_pechora_predictive")
    plot_state_distribution(df_long, output_dir, "11_bayesian_timan_pechora_state_distribution")

    summary_df = pd.DataFrame(
        [
            {
                "sample": sample.name,
                "map_log10_alpha": result.map_log10_alpha,
                "map_alpha": result.map_alpha,
                "map_lambda_M": result.map_matrix_conductivity,
            }
        ]
    )
    write_posterior_summary_excel(
        summary_df,
        output_dir / "11_bayesian_timan_pechora_summary.xlsx",
    )

    plt.show()


if __name__ == "__main__":
    main()