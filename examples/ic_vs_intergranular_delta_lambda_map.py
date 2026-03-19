#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.models.emt.gsa_transport import (
    ComparisonBody,
    gsa_transport_isotropic,
    make_phase,
    two_phase_thermal_isotropic,
)


def dual_one_stage_tc(
    *,
    matrix_tc: float,
    fluid_tc: float,
    phi_intergranular: float,
    phi_ic: float,
    alpha_pore: float,
    alpha_mc: float,
    ic_phase_tc: float,
    comparison: str = "matrix",
) -> float:
    phi_total = phi_intergranular + phi_ic
    if phi_total <= 0.0:
        raise ValueError("phi_intergranular + phi_ic must be positive")
    if phi_total >= 1.0:
        raise ValueError("phi_intergranular + phi_ic must be < 1")

    phi_matrix = 1.0 - phi_total
    phases = [
        make_phase("matrix", phi_matrix, matrix_tc, aspect_ratio=1.0, orientation="random"),
        make_phase(
            "intergranular_pores",
            phi_intergranular,
            fluid_tc,
            aspect_ratio=alpha_pore,
            orientation="random",
        ),
        make_phase(
            "ic_microcracks",
            phi_ic,
            ic_phase_tc,
            aspect_ratio=alpha_mc,
            orientation="random",
        ),
    ]
    body = ComparisonBody(kind=comparison, matrix_index=0)
    return float(gsa_transport_isotropic(phases, body))


def basic_tc(
    *,
    matrix_tc: float,
    fluid_tc: float,
    phi_intergranular: float,
    phi_ic: float,
    alpha_basic: float,
    comparison: str = "matrix",
) -> float:
    phi_total = phi_intergranular + phi_ic
    return float(
        two_phase_thermal_isotropic(
            matrix_value=matrix_tc,
            inclusion_value=fluid_tc,
            porosity=phi_total,
            aspect_ratio=alpha_basic,
            comparison=comparison,
        )
    )


STATE_TO_TC = {
    "dry": 0.025,
    "oil": 0.13,
    "brine": 0.60,
}


def build_map(
    *,
    matrix_tc: float,
    fluid_tc: float,
    alpha_pore: float,
    alpha_basic: float,
    alpha_mc: float,
    ic_phase_tc: float,
    phi_intergranular_values: np.ndarray,
    phi_ic_values: np.ndarray,
    comparison: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(phi_intergranular_values, phi_ic_values)
    zz = np.full_like(xx, np.nan, dtype=float)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            phi_inter = float(xx[i, j])
            phi_ic = float(yy[i, j])
            phi_total = phi_inter + phi_ic
            if phi_total <= 0.0 or phi_total >= 0.999:
                continue

            lam_basic = basic_tc(
                matrix_tc=matrix_tc,
                fluid_tc=fluid_tc,
                phi_intergranular=phi_inter,
                phi_ic=phi_ic,
                alpha_basic=alpha_basic,
                comparison=comparison,
            )
            lam_dual = dual_one_stage_tc(
                matrix_tc=matrix_tc,
                fluid_tc=fluid_tc,
                phi_intergranular=phi_inter,
                phi_ic=phi_ic,
                alpha_pore=alpha_pore,
                alpha_mc=alpha_mc,
                ic_phase_tc=ic_phase_tc,
                comparison=comparison,
            )
            zz[i, j] = 100.0 * (lam_dual - lam_basic) / lam_basic

    return xx, yy, zz


def plot_three_states(
    *,
    matrix_tc: float,
    alpha_pore: float,
    alpha_basic: float,
    alpha_mc: float,
    ic_phase_tc: float,
    phi_intergranular_values: np.ndarray,
    phi_ic_values: np.ndarray,
    comparison: str,
    outpath: Path,
) -> None:
    states = ["dry", "oil", "brine"]
    maps = []
    finite_vals = []

    for state in states:
        xx, yy, zz = build_map(
            matrix_tc=matrix_tc,
            fluid_tc=STATE_TO_TC[state],
            alpha_pore=alpha_pore,
            alpha_basic=alpha_basic,
            alpha_mc=alpha_mc,
            ic_phase_tc=ic_phase_tc,
            phi_intergranular_values=phi_intergranular_values,
            phi_ic_values=phi_ic_values,
            comparison=comparison,
        )
        maps.append((state, xx, yy, zz))
        finite = zz[np.isfinite(zz)]
        if finite.size:
            finite_vals.append(finite)

    if finite_vals:
        all_vals = np.concatenate(finite_vals)
        vmax = float(np.max(np.abs(all_vals)))
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    mappable = None

    for ax, (state, xx, yy, zz) in zip(axes, maps):
        mappable = ax.pcolormesh(
            xx,
            yy,
            zz,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
        )
        total_porosity = xx + yy
        ax.contour(
            xx,
            yy,
            total_porosity,
            levels=[0.10, 0.15, 0.20, 0.25, 0.30],
            colors="k",
            linewidths=0.5,
            alpha=0.45,
        )
        ax.set_title(state.capitalize())
        ax.set_xlabel("Intergranular porosity")
        ax.set_ylabel("IC porosity")

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.95)
    cbar.set_label(r"$100(\lambda_{dual}-\lambda_{basic})/\lambda_{basic}$, %")
    fig.suptitle("Relative difference between dual-porosity and basic EMT models")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_flat_csv(
    *,
    matrix_tc: float,
    alpha_pore: float,
    alpha_basic: float,
    alpha_mc: float,
    ic_phase_tc: float,
    phi_intergranular_values: np.ndarray,
    phi_ic_values: np.ndarray,
    comparison: str,
    outpath: Path,
) -> None:
    rows = [
        "state,phi_intergranular,phi_ic,phi_total,delta_lambda_percent"
    ]
    for state in ("dry", "oil", "brine"):
        _, _, zz = build_map(
            matrix_tc=matrix_tc,
            fluid_tc=STATE_TO_TC[state],
            alpha_pore=alpha_pore,
            alpha_basic=alpha_basic,
            alpha_mc=alpha_mc,
            ic_phase_tc=ic_phase_tc,
            phi_intergranular_values=phi_intergranular_values,
            phi_ic_values=phi_ic_values,
            comparison=comparison,
        )
        for i, phi_ic in enumerate(phi_ic_values):
            for j, phi_inter in enumerate(phi_intergranular_values):
                val = zz[i, j]
                if np.isfinite(val):
                    rows.append(
                        f"{state},{phi_inter:.8f},{phi_ic:.8f},{phi_inter + phi_ic:.8f},{val:.8f}"
                    )
    outpath.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "2D map of relative difference in thermal conductivity between "
            "the basic matrix+void-space model and the one-stage dual-porosity model."
        )
    )
    parser.add_argument("--outdir", default="results/ic_vs_intergranular_delta_lambda")
    parser.add_argument("--matrix-tc", type=float, default=4.5)
    parser.add_argument("--alpha-pore", type=float, default=0.15)
    parser.add_argument(
        "--alpha-basic",
        type=float,
        default=None,
        help="If omitted, alpha_basic = alpha_pore for a fair comparison.",
    )
    parser.add_argument("--alpha-mc", type=float, default=1e-3)
    parser.add_argument("--ic-phase-tc", type=float, default=0.025)
    parser.add_argument("--comparison", default="matrix", choices=["matrix", "self_consistent", "bayuk_linear_mix"])
    parser.add_argument("--phi-inter-min", type=float, default=0.03)
    parser.add_argument("--phi-inter-max", type=float, default=0.25)
    parser.add_argument("--n-inter", type=int, default=120)
    parser.add_argument("--phi-ic-min", type=float, default=0.0)
    parser.add_argument("--phi-ic-max", type=float, default=0.010)
    parser.add_argument("--n-ic", type=int, default=120)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    alpha_basic = args.alpha_pore if args.alpha_basic is None else args.alpha_basic
    phi_inter_vals = np.linspace(args.phi_inter_min, args.phi_inter_max, args.n_inter)
    phi_ic_vals = np.linspace(args.phi_ic_min, args.phi_ic_max, args.n_ic)

    plot_three_states(
        matrix_tc=args.matrix_tc,
        alpha_pore=args.alpha_pore,
        alpha_basic=alpha_basic,
        alpha_mc=args.alpha_mc,
        ic_phase_tc=args.ic_phase_tc,
        phi_intergranular_values=phi_inter_vals,
        phi_ic_values=phi_ic_vals,
        comparison=args.comparison,
        outpath=outdir / "delta_lambda_intergranular_vs_ic_porosity.png",
    )

    save_flat_csv(
        matrix_tc=args.matrix_tc,
        alpha_pore=args.alpha_pore,
        alpha_basic=alpha_basic,
        alpha_mc=args.alpha_mc,
        ic_phase_tc=args.ic_phase_tc,
        phi_intergranular_values=phi_inter_vals,
        phi_ic_values=phi_ic_vals,
        comparison=args.comparison,
        outpath=outdir / "delta_lambda_intergranular_vs_ic_porosity.csv",
    )

    (outdir / "summary.txt").write_text(
        "\n".join(
            [
                "2D relative-difference map between the basic and dual-porosity EMT models.",
                f"matrix_tc = {args.matrix_tc}",
                f"alpha_pore = {args.alpha_pore}",
                f"alpha_basic = {alpha_basic}",
                f"alpha_mc = {args.alpha_mc}",
                f"ic_phase_tc = {args.ic_phase_tc}",
                f"comparison = {args.comparison}",
                f"phi_intergranular range = [{args.phi_inter_min}, {args.phi_inter_max}]",
                f"phi_ic range = [{args.phi_ic_min}, {args.phi_ic_max}]",
                "Color = 100 * (lambda_dual - lambda_basic) / lambda_basic, in percent.",
                "Basic model uses total porosity = phi_intergranular + phi_ic as a single void phase.",
                "Dual model mixes matrix, intergranular pores, and IC microcracks in one GSA step.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
