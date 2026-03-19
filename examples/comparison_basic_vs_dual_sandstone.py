from __future__ import annotations

"""Forward comparison of a basic two-phase TC model and a one-stage dual-porosity model.

Run from the RockPhysX repository root, for example:

    PYTHONPATH=src python examples/comparison_basic_vs_dual_sandstone.py \
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

What it does
------------
1. Compares the forward predictions of:
   - a basic two-phase model: matrix + total void space;
   - a one-stage dual-porosity model: matrix + intergranular pores + IC microcracks.
2. Creates three-panel lambda-vs-porosity plots for dry / oil / brine.
3. Creates three-panel relative-difference plots (dual vs basic) for dry / oil / brine.
4. Optionally creates a dry-state heatmap of relative difference over (phi_total, phi_mc).

Model logic
-----------
Basic model:
    lambda_basic = GSA(matrix, fluid, total_porosity, alpha_basic)

Dual model:
    total porosity = phi_pore + phi_mc
    lambda_dual = GSA(matrix + pore_fluid + IC_microcrack together)

Important interpretation:
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

    # Porosity grid for the comparison curves.
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
        help="Absolute IC microcrack porosity fraction used in the curve comparison",
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

    # Optional heatmap in (phi_total, phi_mc) for the dry state.
    p.add_argument("--make-heatmap", action="store_true", help="Also create a dry-state heatmap over (phi_total, phi_mc)")
    p.add_argument("--phi-mc-min", type=float, default=0.0005)
    p.add_argument("--phi-mc-max", type=float, default=0.0060)
    p.add_argument("--n-phi-mc", type=int, default=81)
    return p.parse_args()



def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.phi_min < 1.0 and 0.0 < args.phi_max < 1.0 and args.phi_min < args.phi_max):
        raise ValueError("Porosity bounds must satisfy 0 < phi_min < phi_max < 1.")
    if args.n_phi < 3:
        raise ValueError("n_phi must be at least 3.")
    if args.alpha_pore <= 0.0 or args.alpha_mc <= 0.0:
        raise ValueError("Aspect ratios must be positive.")
    if args.matrix_tc <= 0.0 or args.air_tc <= 0.0 or args.oil_tc <= 0.0 or args.brine_tc <= 0.0 or args.ic_phase_tc <= 0.0:
        raise ValueError("All thermal conductivities must be positive.")
    if args.phi_mc_abs <= 0.0:
        raise ValueError("phi_mc_abs must be positive.")
    if args.phi_mc_abs >= args.phi_min:
        raise ValueError("phi_mc_abs must be smaller than phi_min so that phi_pore = phi_total - phi_mc stays positive.")
    if args.make_heatmap:
        if not (0.0 < args.phi_mc_min < args.phi_mc_max):
            raise ValueError("Heatmap phi_mc bounds must satisfy 0 < phi_mc_min < phi_mc_max.")
        if args.phi_mc_max >= args.phi_max:
            raise ValueError("phi_mc_max must be smaller than phi_max.")
        if args.n_phi_mc < 3:
            raise ValueError("n_phi_mc must be at least 3.")



def fluid_tc_by_state(args: argparse.Namespace, state: SaturationState) -> float:
    if state == SaturationState.DRY:
        return float(args.air_tc)
    if state == SaturationState.OIL:
        return float(args.oil_tc)
    if state == SaturationState.BRINE:
        return float(args.brine_tc)
    raise ValueError(f"Unsupported saturation state: {state!r}")



def basic_tc(
    *,
    phi_total: float,
    matrix_tc: float,
    fluid_tc: float,
    alpha_basic: float,
    comparison: str,
) -> float:
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



def build_curve_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    alpha_basic = float(args.alpha_pore if args.alpha_basic is None else args.alpha_basic)
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
            delta_abs = lam_dual - lam_basic
            delta_pct = 100.0 * delta_abs / lam_basic
            rows.append(
                {
                    "state": state_name,
                    "phi_total": float(phi_total),
                    "phi_mc_abs": float(args.phi_mc_abs),
                    "phi_pore": float(phi_total - args.phi_mc_abs),
                    "lambda_basic": float(lam_basic),
                    "lambda_dual": float(lam_dual),
                    "delta_lambda_abs": float(delta_abs),
                    "delta_lambda_pct": float(delta_pct),
                    "alpha_basic": float(alpha_basic),
                    "alpha_pore": float(args.alpha_pore),
                    "alpha_mc": float(args.alpha_mc),
                }
            )
    return pd.DataFrame(rows)



def plot_lambda_vs_porosity(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=False)
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        sub = df[df["state"] == state_name].copy()
        ax.plot(sub["phi_total"], sub["lambda_basic"], label="Basic: matrix + void space")
        ax.plot(sub["phi_total"], sub["lambda_dual"], label="Dual: matrix + pores + microcracks")
        ax.set_title(state_name.capitalize())
        ax.set_xlabel("Total porosity, fraction")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"Effective thermal conductivity, W m$^{-1}$ K$^{-1}$")
    axes[0].legend(loc="best")
    fig.suptitle("Forward-model comparison: thermal conductivity vs porosity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_delta_vs_porosity(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        sub = df[df["state"] == state_name].copy()
        ax.plot(sub["phi_total"], sub["delta_lambda_pct"])
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(state_name.capitalize())
        ax.set_xlabel("Total porosity, fraction")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"100 (\lambda_dual - \lambda_basic) / \lambda_basic, %")
    fig.suptitle("Relative difference between dual and basic model predictions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def build_heatmap_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    alpha_basic = float(args.alpha_pore if args.alpha_basic is None else args.alpha_basic)
    phi_values = np.linspace(float(args.phi_min), float(args.phi_max), int(args.n_phi))
    phi_mc_values = np.linspace(float(args.phi_mc_min), float(args.phi_mc_max), int(args.n_phi_mc))

    rows: list[dict[str, float]] = []
    fluid_tc = float(args.air_tc)  # dry state heatmap by design
    for phi_total in phi_values:
        for phi_mc_abs in phi_mc_values:
            if phi_mc_abs >= phi_total:
                continue
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
                phi_mc_abs=float(phi_mc_abs),
                alpha_mc=float(args.alpha_mc),
                ic_phase_tc=float(args.ic_phase_tc),
                comparison=args.comparison,
            )
            rows.append(
                {
                    "phi_total": float(phi_total),
                    "phi_mc_abs": float(phi_mc_abs),
                    "delta_lambda_pct": float(100.0 * (lam_dual - lam_basic) / lam_basic),
                }
            )
    return pd.DataFrame(rows)



def plot_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    pivot = df.pivot(index="phi_mc_abs", columns="phi_total", values="delta_lambda_pct")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    mesh = ax.pcolormesh(x, y, z, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"100 (\lambda_dual - \lambda_basic) / \lambda_basic, %")
    ax.set_xlabel("Total porosity, fraction")
    ax.set_ylabel("IC microcrack porosity, fraction")
    ax.set_title("Dry-state relative difference: dual vs basic")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def write_summary(args: argparse.Namespace, outpath: Path) -> None:
    alpha_basic = float(args.alpha_pore if args.alpha_basic is None else args.alpha_basic)
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
        "Porosity sweep:",
        f"  phi_total in [{float(args.phi_min):.6g}, {float(args.phi_max):.6g}] with n = {int(args.n_phi)}",
        "",
        "Interpretation:",
        "  The plots show how much the extra IC microcrack subdomain changes the forward thermal-conductivity prediction.",
        "  They do not show which model is closer to experiment.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    args = parse_args()
    validate_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = build_curve_dataframe(args)
    df.to_csv(args.outdir / "comparison_curves.csv", index=False)
    plot_lambda_vs_porosity(df, args.outdir / "lambda_vs_porosity_comparison.png")
    plot_delta_vs_porosity(df, args.outdir / "delta_lambda_percent_vs_porosity.png")
    write_summary(args, args.outdir / "summary.txt")

    if args.make_heatmap:
        hdf = build_heatmap_dataframe(args)
        hdf.to_csv(args.outdir / "dry_state_heatmap.csv", index=False)
        plot_heatmap(hdf, args.outdir / "dry_state_heatmap_delta_percent.png")

    print(f"Saved results to: {args.outdir}")


if __name__ == "__main__":
    main()
