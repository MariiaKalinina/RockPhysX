from __future__ import annotations

"""Forward comparison of a basic two-phase sandstone TC model and a one-stage dual-porosity model.

This script creates FIVE figure files:
1. lambda_vs_porosity_comparison.png
2. delta_lambda_percent_vs_porosity.png
3. heatmap_delta_vs_phi_total_phi_mc.png
4. heatmap_delta_vs_alpha_pore_phi_mc.png
5. saturation_contrast_vs_porosity.png

Run from the RockPhysX repository root, for example:

    PYTHONPATH=src python examples/comparison_basic_vs_dual_sandstone_five_plots.py \
        --phi-min 0.05 \
        --phi-max 0.30 \
        --n-phi 101 \
        --matrix-tc 4.5 \
        --alpha-pore 0.15 \
        --alpha-basic 0.15 \
        --phi-mc-abs 0.0025 \
        --alpha-mc 1e-3 \
        --ic-phase-tc 0.025 \
        --outdir results/basic_vs_dual

Interpretation:
    This is a forward-structure comparison only. It shows whether the two models
    predict different thermal conductivities under matched assumptions. It does
    not, by itself, prove that one model is better than the other.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rockphysx.core.saturation import SaturationState
from rockphysx.models.emt.gsa_transport import (
    ComparisonBody,
    gsa_transport_isotropic,
    make_phase,
    two_phase_thermal_isotropic,
)


STATE_ORDER: list[tuple[str, SaturationState]] = [
    ("dry", SaturationState.DRY),
    ("oil", SaturationState.OIL),
    ("brine", SaturationState.BRINE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("results/basic_vs_dual"))

    # Main porosity sweep for line plots.
    p.add_argument("--phi-min", type=float, default=0.05, help="Minimum total porosity fraction")
    p.add_argument("--phi-max", type=float, default=0.30, help="Maximum total porosity fraction")
    p.add_argument("--n-phi", type=int, default=101, help="Number of porosity points")

    # Shared matrix and fluid properties.
    p.add_argument("--matrix-tc", type=float, default=4.5, help="Matrix thermal conductivity [W m^-1 K^-1]")
    p.add_argument("--air-tc", type=float, default=0.025, help="Dry-state fluid TC")
    p.add_argument("--oil-tc", type=float, default=0.13, help="Oil/kerosene TC")
    p.add_argument("--brine-tc", type=float, default=0.60, help="Brine TC")

    # Basic-model geometry.
    p.add_argument(
        "--alpha-basic",
        type=float,
        default=None,
        help="Effective aspect ratio for the basic model. Default: use --alpha-pore for a fair comparison.",
    )

    # Dual-model geometry.
    p.add_argument("--alpha-pore", type=float, default=0.15, help="Intergranular-pore aspect ratio")
    p.add_argument("--alpha-mc", type=float, default=1e-3, help="IC microcrack aspect ratio")
    p.add_argument(
        "--phi-mc-abs",
        type=float,
        default=0.0025,
        help="Absolute IC microcrack porosity fraction used in the line-plot comparison",
    )
    p.add_argument(
        "--ic-phase-tc",
        type=float,
        default=0.025,
        help="Thermal conductivity assigned to the IC microcrack fill phase",
    )

    # Comparison-body choice.
    p.add_argument(
        "--comparison",
        choices=["matrix", "self_consistent", "bayuk_linear_mix"],
        default="matrix",
        help="Comparison-body closure passed to RockPhysX GSA wrappers",
    )

    # Heatmap 3: (phi_total, phi_mc)
    p.add_argument("--phi-mc-min", type=float, default=0.0005, help="Minimum IC porosity for heatmaps")
    p.add_argument("--phi-mc-max", type=float, default=0.0060, help="Maximum IC porosity for heatmaps")
    p.add_argument("--n-phi-mc", type=int, default=81, help="Number of IC porosity points for heatmaps")

    # Heatmap 4: (alpha_pore, phi_mc) at a fixed representative total porosity.
    p.add_argument("--alpha-pore-min", type=float, default=0.05, help="Minimum alpha_pore for heatmap 4")
    p.add_argument("--alpha-pore-max", type=float, default=0.30, help="Maximum alpha_pore for heatmap 4")
    p.add_argument("--n-alpha-pore", type=int, default=81, help="Number of alpha_pore points for heatmap 4")
    p.add_argument(
        "--phi-representative",
        type=float,
        default=0.18,
        help="Representative total porosity used in the (alpha_pore, phi_mc) heatmap",
    )
    return p.parse_args()



def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.phi_min < 1.0 and 0.0 < args.phi_max < 1.0 and args.phi_min < args.phi_max):
        raise ValueError("Porosity bounds must satisfy 0 < phi_min < phi_max < 1.")
    if args.n_phi < 3:
        raise ValueError("n_phi must be at least 3.")
    if args.matrix_tc <= 0.0 or args.air_tc <= 0.0 or args.oil_tc <= 0.0 or args.brine_tc <= 0.0 or args.ic_phase_tc <= 0.0:
        raise ValueError("All thermal conductivities must be positive.")
    if args.alpha_pore <= 0.0 or args.alpha_mc <= 0.0:
        raise ValueError("Aspect ratios must be positive.")
    if args.alpha_basic is not None and args.alpha_basic <= 0.0:
        raise ValueError("alpha_basic must be positive.")
    if args.phi_mc_abs <= 0.0:
        raise ValueError("phi_mc_abs must be positive.")
    if args.phi_mc_abs >= args.phi_min:
        raise ValueError("phi_mc_abs must be smaller than phi_min so that phi_pore = phi_total - phi_mc stays positive on the sweep.")
    if not (0.0 < args.phi_mc_min < args.phi_mc_max < args.phi_max):
        raise ValueError("Heatmap IC-porosity bounds must satisfy 0 < phi_mc_min < phi_mc_max < phi_max.")
    if args.n_phi_mc < 3:
        raise ValueError("n_phi_mc must be at least 3.")
    if not (0.0 < args.alpha_pore_min < args.alpha_pore_max):
        raise ValueError("alpha_pore heatmap bounds must satisfy 0 < alpha_pore_min < alpha_pore_max.")
    if args.n_alpha_pore < 3:
        raise ValueError("n_alpha_pore must be at least 3.")
    if not (args.phi_min <= args.phi_representative <= args.phi_max):
        raise ValueError("phi_representative must lie within [phi_min, phi_max].")
    if args.phi_mc_max >= args.phi_representative:
        raise ValueError("phi_mc_max must be smaller than phi_representative for the (alpha_pore, phi_mc) heatmap.")



def fluid_tc_by_state(args: argparse.Namespace, state: SaturationState) -> float:
    if state == SaturationState.DRY:
        return float(args.air_tc)
    if state == SaturationState.OIL:
        return float(args.oil_tc)
    if state == SaturationState.BRINE:
        return float(args.brine_tc)
    raise ValueError(f"Unsupported saturation state: {state!r}")



def basic_tc(*, phi_total: float, matrix_tc: float, fluid_tc: float, alpha_basic: float, comparison: str) -> float:
    return float(
        two_phase_thermal_isotropic(
            matrix_value=float(matrix_tc),
            inclusion_value=float(fluid_tc),
            porosity=float(phi_total),
            aspect_ratio=float(alpha_basic),
            comparison=comparison,
        )
    )



def dual_tc(
    *,
    phi_total: float,
    matrix_tc: float,
    fluid_tc: float,
    alpha_pore: float,
    phi_mc_abs: float,
    alpha_mc: float,
    ic_phase_tc: float,
    comparison: str,
) -> float:
    if phi_mc_abs <= 0.0 or phi_mc_abs >= phi_total:
        raise ValueError(f"phi_mc_abs must lie in (0, phi_total). Got phi_mc_abs={phi_mc_abs}, phi_total={phi_total}.")

    phi_pore = float(phi_total - phi_mc_abs)
    phi_matrix = float(1.0 - phi_total)
    phase_sum = phi_matrix + phi_pore + float(phi_mc_abs)
    if not np.isclose(phase_sum, 1.0, atol=1e-10):
        raise ValueError(f"Phase fractions must sum to 1.0, got {phase_sum}.")

    phases = [
        make_phase("matrix", phi_matrix, matrix_tc, aspect_ratio=1.0, orientation="random"),
        make_phase("pore_fluid", phi_pore, fluid_tc, aspect_ratio=alpha_pore, orientation="random"),
        make_phase("ic_microcrack", float(phi_mc_abs), ic_phase_tc, aspect_ratio=alpha_mc, orientation="random"),
    ]
    body = ComparisonBody(kind=comparison, matrix_index=0)
    return float(gsa_transport_isotropic(phases, body))



def alpha_basic_value(args: argparse.Namespace) -> float:
    return float(args.alpha_pore if args.alpha_basic is None else args.alpha_basic)



def build_curve_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    alpha_basic = alpha_basic_value(args)
    phi_values = np.linspace(float(args.phi_min), float(args.phi_max), int(args.n_phi))

    rows: list[dict[str, float | str]] = []
    for state_name, state in STATE_ORDER:
        fluid_tc = fluid_tc_by_state(args, state)
        for phi_total in phi_values:
            lam_basic = basic_tc(
                phi_total=float(phi_total),
                matrix_tc=float(args.matrix_tc),
                fluid_tc=fluid_tc,
                alpha_basic=alpha_basic,
                comparison=args.comparison,
            )
            lam_dual = dual_tc(
                phi_total=float(phi_total),
                matrix_tc=float(args.matrix_tc),
                fluid_tc=fluid_tc,
                alpha_pore=float(args.alpha_pore),
                phi_mc_abs=float(args.phi_mc_abs),
                alpha_mc=float(args.alpha_mc),
                ic_phase_tc=float(args.ic_phase_tc),
                comparison=args.comparison,
            )
            rows.append(
                {
                    "state": state_name,
                    "phi_total": float(phi_total),
                    "phi_mc_abs": float(args.phi_mc_abs),
                    "phi_pore": float(phi_total - args.phi_mc_abs),
                    "lambda_basic": float(lam_basic),
                    "lambda_dual": float(lam_dual),
                    "delta_lambda_abs": float(lam_dual - lam_basic),
                    "delta_lambda_pct": float(100.0 * (lam_dual - lam_basic) / lam_basic),
                    "alpha_basic": float(alpha_basic),
                    "alpha_pore": float(args.alpha_pore),
                    "alpha_mc": float(args.alpha_mc),
                }
            )
    return pd.DataFrame(rows)



def build_phi_phi_mc_heatmap_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    alpha_basic = alpha_basic_value(args)
    phi_values = np.linspace(float(args.phi_min), float(args.phi_max), int(args.n_phi))
    phi_mc_values = np.linspace(float(args.phi_mc_min), float(args.phi_mc_max), int(args.n_phi_mc))

    rows: list[dict[str, float | str]] = []
    for state_name, state in STATE_ORDER:
        fluid_tc = fluid_tc_by_state(args, state)
        for phi_total in phi_values:
            lam_basic = basic_tc(
                phi_total=float(phi_total),
                matrix_tc=float(args.matrix_tc),
                fluid_tc=fluid_tc,
                alpha_basic=alpha_basic,
                comparison=args.comparison,
            )
            for phi_mc_abs in phi_mc_values:
                if phi_mc_abs >= phi_total:
                    continue
                lam_dual = dual_tc(
                    phi_total=float(phi_total),
                    matrix_tc=float(args.matrix_tc),
                    fluid_tc=fluid_tc,
                    alpha_pore=float(args.alpha_pore),
                    phi_mc_abs=float(phi_mc_abs),
                    alpha_mc=float(args.alpha_mc),
                    ic_phase_tc=float(args.ic_phase_tc),
                    comparison=args.comparison,
                )
                rows.append(
                    {
                        "state": state_name,
                        "phi_total": float(phi_total),
                        "phi_mc_abs": float(phi_mc_abs),
                        "delta_lambda_pct": float(100.0 * (lam_dual - lam_basic) / lam_basic),
                    }
                )
    return pd.DataFrame(rows)



def build_alpha_phi_mc_heatmap_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    alpha_basic = alpha_basic_value(args)
    alpha_values = np.linspace(float(args.alpha_pore_min), float(args.alpha_pore_max), int(args.n_alpha_pore))
    phi_mc_values = np.linspace(float(args.phi_mc_min), float(args.phi_mc_max), int(args.n_phi_mc))
    phi_total = float(args.phi_representative)

    rows: list[dict[str, float | str]] = []
    for state_name, state in STATE_ORDER:
        fluid_tc = fluid_tc_by_state(args, state)
        lam_basic = basic_tc(
            phi_total=phi_total,
            matrix_tc=float(args.matrix_tc),
            fluid_tc=fluid_tc,
            alpha_basic=alpha_basic,
            comparison=args.comparison,
        )
        for alpha_pore in alpha_values:
            for phi_mc_abs in phi_mc_values:
                if phi_mc_abs >= phi_total:
                    continue
                lam_dual = dual_tc(
                    phi_total=phi_total,
                    matrix_tc=float(args.matrix_tc),
                    fluid_tc=fluid_tc,
                    alpha_pore=float(alpha_pore),
                    phi_mc_abs=float(phi_mc_abs),
                    alpha_mc=float(args.alpha_mc),
                    ic_phase_tc=float(args.ic_phase_tc),
                    comparison=args.comparison,
                )
                rows.append(
                    {
                        "state": state_name,
                        "phi_total": phi_total,
                        "alpha_pore": float(alpha_pore),
                        "phi_mc_abs": float(phi_mc_abs),
                        "delta_lambda_pct": float(100.0 * (lam_dual - lam_basic) / lam_basic),
                    }
                )
    return pd.DataFrame(rows)



def build_saturation_contrast_dataframe(df_curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    phi_values = np.sort(df_curves["phi_total"].unique())
    for phi_total in phi_values:
        sub = df_curves[np.isclose(df_curves["phi_total"], phi_total)].copy()
        by_state = {row["state"]: row for _, row in sub.iterrows()}

        lam_b_dry = float(by_state["dry"]["lambda_basic"])
        lam_b_oil = float(by_state["oil"]["lambda_basic"])
        lam_b_brine = float(by_state["brine"]["lambda_basic"])
        lam_d_dry = float(by_state["dry"]["lambda_dual"])
        lam_d_oil = float(by_state["oil"]["lambda_dual"])
        lam_d_brine = float(by_state["brine"]["lambda_dual"])

        rows.extend(
            [
                {
                    "contrast": "brine/dry",
                    "phi_total": float(phi_total),
                    "basic": lam_b_brine / lam_b_dry,
                    "dual": lam_d_brine / lam_d_dry,
                },
                {
                    "contrast": "oil/dry",
                    "phi_total": float(phi_total),
                    "basic": lam_b_oil / lam_b_dry,
                    "dual": lam_d_oil / lam_d_dry,
                },
                {
                    "contrast": "brine/oil",
                    "phi_total": float(phi_total),
                    "basic": lam_b_brine / lam_b_oil,
                    "dual": lam_d_brine / lam_d_oil,
                },
            ]
        )
    return pd.DataFrame(rows)



def plot_lambda_vs_porosity(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharex=True, sharey=False)
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        sub = df[df["state"] == state_name].copy()
        ax.plot(sub["phi_total"], sub["lambda_basic"], label="Basic: matrix + void space")
        ax.plot(sub["phi_total"], sub["lambda_dual"], label="Dual: matrix + pores + microcracks")
        ax.set_title(state_name.capitalize())
        ax.set_xlabel("Total porosity, fraction")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"Effective thermal conductivity, W m$^{-1}$ K$^{-1}$")
    axes[0].legend(loc="best")
    fig.suptitle("Plot 1. Forward-model comparison: thermal conductivity vs porosity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_delta_vs_porosity(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharex=True, sharey=True)
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        sub = df[df["state"] == state_name].copy()
        ax.plot(sub["phi_total"], sub["delta_lambda_pct"])
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(state_name.capitalize())
        ax.set_xlabel("Total porosity, fraction")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"100 (\lambda$_{dual}$ - \lambda$_{basic}$) / \lambda$_{basic}$, %")
    fig.suptitle("Plot 2. Relative difference between dual and basic model predictions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def _heatmap_arrays(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index=ycol, columns=xcol, values="delta_lambda_pct")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    return x, y, z



def plot_phi_phi_mc_heatmaps(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True, sharey=True)
    mesh = None
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        sub = df[df["state"] == state_name].copy()
        x, y, z = _heatmap_arrays(sub, "phi_total", "phi_mc_abs")
        mesh = ax.pcolormesh(x, y, z, shading="auto")
        ax.set_title(state_name.capitalize())
        ax.set_xlabel("Total porosity, fraction")
        ax.grid(False)
    axes[0].set_ylabel("IC microcrack porosity, fraction")
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.96)
        cb.set_label(r"100 (\lambda$_{dual}$ - \lambda$_{basic}$) / \lambda$_{basic}$, %")
    fig.suptitle("Plot 3. Relative difference over (total porosity, IC porosity)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_alpha_phi_mc_heatmaps(df: pd.DataFrame, outpath: Path) -> None:
    phi_total = float(df["phi_total"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True, sharey=True)
    mesh = None
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        sub = df[df["state"] == state_name].copy()
        x, y, z = _heatmap_arrays(sub, "alpha_pore", "phi_mc_abs")
        mesh = ax.pcolormesh(x, y, z, shading="auto")
        ax.set_title(state_name.capitalize())
        ax.set_xlabel(r"Intergranular-pore aspect ratio, $\alpha_{pore}$")
        ax.grid(False)
    axes[0].set_ylabel("IC microcrack porosity, fraction")
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.96)
        cb.set_label(r"100 (\lambda$_{dual}$ - \lambda$_{basic}$) / \lambda$_{basic}$, %")
    fig.suptitle(fr"Plot 4. Relative difference over ($\alpha_{{pore}}$, IC porosity) at $\phi_{{total}}$ = {phi_total:.3f}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_saturation_contrasts(df: pd.DataFrame, outpath: Path) -> None:
    contrast_order = ["brine/dry", "oil/dry", "brine/oil"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), sharex=True, sharey=False)
    for ax, contrast_name in zip(axes, contrast_order):
        sub = df[df["contrast"] == contrast_name].copy()
        ax.plot(sub["phi_total"], sub["basic"], label="Basic model")
        ax.plot(sub["phi_total"], sub["dual"], label="Dual model")
        ax.set_title(contrast_name)
        ax.set_xlabel("Total porosity, fraction")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Saturation-contrast ratio")
    axes[0].legend(loc="best")
    fig.suptitle("Plot 5. Saturation-contrast ratios vs porosity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def write_summary(args: argparse.Namespace, outpath: Path) -> None:
    alpha_basic = alpha_basic_value(args)
    lines = [
        "Forward comparison of basic vs dual-porosity sandstone models",
        "",
        "Basic model:",
        "  matrix + total void space",
        f"  alpha_basic = {alpha_basic:.6g}",
        "",
        "Dual model:",
        "  matrix + intergranular pores + IC microcracks (embedded together)",
        f"  alpha_pore = {float(args.alpha_pore):.6g}",
        f"  alpha_mc = {float(args.alpha_mc):.6g}",
        f"  phi_mc_abs = {float(args.phi_mc_abs):.6g}",
        f"  ic_phase_tc = {float(args.ic_phase_tc):.6g} W m^-1 K^-1",
        "",
        "Shared properties:",
        f"  matrix_tc = {float(args.matrix_tc):.6g} W m^-1 K^-1",
        f"  air_tc = {float(args.air_tc):.6g} W m^-1 K^-1",
        f"  oil_tc = {float(args.oil_tc):.6g} W m^-1 K^-1",
        f"  brine_tc = {float(args.brine_tc):.6g} W m^-1 K^-1",
        f"  comparison_body = {args.comparison}",
        "",
        "Main line-plot sweep:",
        f"  phi_total in [{float(args.phi_min):.6g}, {float(args.phi_max):.6g}] with n = {int(args.n_phi)}",
        "",
        "Heatmap sweep 1:",
        f"  phi_mc in [{float(args.phi_mc_min):.6g}, {float(args.phi_mc_max):.6g}] with n = {int(args.n_phi_mc)}",
        "",
        "Heatmap sweep 2:",
        f"  alpha_pore in [{float(args.alpha_pore_min):.6g}, {float(args.alpha_pore_max):.6g}] with n = {int(args.n_alpha_pore)}",
        f"  phi_representative = {float(args.phi_representative):.6g}",
        "",
        "Figures produced:",
        "  1) lambda_vs_porosity_comparison.png",
        "  2) delta_lambda_percent_vs_porosity.png",
        "  3) heatmap_delta_vs_phi_total_phi_mc.png",
        "  4) heatmap_delta_vs_alpha_pore_phi_mc.png",
        "  5) saturation_contrast_vs_porosity.png",
        "",
        "Interpretation:",
        "  These plots show how much the extra IC microcrack subdomain changes the forward thermal-conductivity prediction.",
        "  They do not show which model is closer to experiment.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    args = parse_args()
    validate_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_curves = build_curve_dataframe(args)
    df_curves.to_csv(args.outdir / "comparison_curves.csv", index=False)

    df_phi_phi_mc = build_phi_phi_mc_heatmap_dataframe(args)
    df_phi_phi_mc.to_csv(args.outdir / "heatmap_phi_total_phi_mc.csv", index=False)

    df_alpha_phi_mc = build_alpha_phi_mc_heatmap_dataframe(args)
    df_alpha_phi_mc.to_csv(args.outdir / "heatmap_alpha_pore_phi_mc.csv", index=False)

    df_contrasts = build_saturation_contrast_dataframe(df_curves)
    df_contrasts.to_csv(args.outdir / "saturation_contrasts.csv", index=False)

    plot_lambda_vs_porosity(df_curves, args.outdir / "lambda_vs_porosity_comparison.png")
    plot_delta_vs_porosity(df_curves, args.outdir / "delta_lambda_percent_vs_porosity.png")
    plot_phi_phi_mc_heatmaps(df_phi_phi_mc, args.outdir / "heatmap_delta_vs_phi_total_phi_mc.png")
    plot_alpha_phi_mc_heatmaps(df_alpha_phi_mc, args.outdir / "heatmap_delta_vs_alpha_pore_phi_mc.png")
    plot_saturation_contrasts(df_contrasts, args.outdir / "saturation_contrast_vs_porosity.png")
    write_summary(args, args.outdir / "summary.txt")

    print(f"Saved results to: {args.outdir}")


if __name__ == "__main__":
    main()
