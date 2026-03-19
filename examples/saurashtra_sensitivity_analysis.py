from __future__ import annotations

"""Sensitivity analysis for the Saurashtra sandstone one-stage TC model.

Designed to be run from the RockPhysX repository root:

    PYTHONPATH=src python examples/saurashtra_sensitivity_analysis.py \
        --phi-total 0.18 \
        --matrix-tc 4.5 \
        --alpha-pore 0.15 \
        --phi-mc 0.0025 \
        --alpha-mc 1e-3 \
        --ic-phase-tc 0.025 \
        --outdir results/saurashtra_sensitivity

What this script does
---------------------
1. Computes local normalized sensitivities of lambda* in dry / oil / brine states.
2. Builds response maps in (alpha_pore, phi_mc).
3. Optionally builds misfit maps if measured TC values are provided.
4. Runs a robustness sweep over alpha_mc.

Modeling assumption used here
-----------------------------
The sandstone model is implemented as a one-stage isotropic GSA mixture in which
all three constituents are embedded simultaneously:

    matrix + intergranular pores + IC microcrack domain

The total porosity is partitioned into:
    phi_pore = phi_total - phi_mc
    phi_mc   = IC microcrack porosity

The intergranular pores are assigned the current saturation-state fluid thermal
conductivity, while the IC microcrack domain is filled by a phase of fixed thermal
conductivity (`ic_phase_tc`).
"""

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.models.emt.gsa_transport import ComparisonBody, gsa_transport_isotropic, make_phase


EPS = 1e-12
STATE_ORDER: list[tuple[str, SaturationState]] = [
    ("dry", SaturationState.DRY),
    ("oil", SaturationState.OIL),
    ("brine", SaturationState.BRINE),
]


@dataclass(frozen=True)
class DualDomainMicrostructure:
    alpha_pore: float
    alpha_mc: float
    phi_mc: float
    ic_phase_tc: float


@dataclass(frozen=True)
class DualDomainSample:
    sample: SampleDescription
    dual: DualDomainMicrostructure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("results/saurashtra_sensitivity"))

    # Representative sandstone plug parameters.
    p.add_argument("--name", default="saurashtra_reference")
    p.add_argument("--phi-total", type=float, required=True, help="Total porosity fraction, e.g. 0.18")
    p.add_argument("--matrix-tc", type=float, required=True, help="Matrix thermal conductivity [W m^-1 K^-1]")
    p.add_argument("--alpha-pore", type=float, required=True, help="Aspect ratio of intergranular pores")
    p.add_argument("--phi-mc", type=float, required=True, help="Absolute microcrack porosity fraction, e.g. 0.002")
    p.add_argument("--alpha-mc", type=float, default=1e-3, help="Aspect ratio of the IC microcrack domain")
    p.add_argument("--ic-phase-tc", type=float, default=0.025, help="Fixed TC of the IC-domain fill phase")

    # State-dependent fluid thermal conductivities.
    p.add_argument("--air-tc", type=float, default=0.025)
    p.add_argument("--oil-tc", type=float, default=0.13)
    p.add_argument("--brine-tc", type=float, default=0.60)

    # Local finite-difference perturbation.
    p.add_argument("--rel-step", type=float, default=0.02, help="Relative perturbation for local sensitivities")

    # Response-map bounds.
    p.add_argument("--alpha-pore-min", type=float, default=0.03)
    p.add_argument("--alpha-pore-max", type=float, default=0.30)
    p.add_argument("--phi-mc-min", type=float, default=0.0005)
    p.add_argument("--phi-mc-max", type=float, default=0.0060)
    p.add_argument("--n-grid", type=int, default=81)

    # Optional measured TC values: if given, misfit maps are produced.
    p.add_argument("--tc-meas-dry", type=float, default=None)
    p.add_argument("--tc-meas-oil", type=float, default=None)
    p.add_argument("--tc-meas-brine", type=float, default=None)

    # Optional weights for multi-state misfit.
    p.add_argument("--w-dry", type=float, default=1.0)
    p.add_argument("--w-oil", type=float, default=1.0)
    p.add_argument("--w-brine", type=float, default=1.0)

    # alpha_mc robustness sweep.
    p.add_argument("--alpha-mc-sweep-min", type=float, default=1e-4)
    p.add_argument("--alpha-mc-sweep-max", type=float, default=1e-3)
    p.add_argument("--alpha-mc-sweep-n", type=int, default=31)
    return p.parse_args()


def make_matrix_properties(matrix_tc: float) -> MatrixProperties:
    return MatrixProperties(
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.65,
        thermal_conductivity_wmk=float(matrix_tc),
        electrical_conductivity_sm=1e-10,
    )


def make_dummy_mineral(matrix_tc: float) -> MineralPhase:
    return MineralPhase(
        name="matrix",
        volume_fraction=1.0,
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.65,
        thermal_conductivity_wmk=float(matrix_tc),
        electrical_conductivity_sm=1e-10,
    )


def make_fluids(air_tc: float, oil_tc: float, brine_tc: float) -> dict[SaturationState, FluidPhase]:
    return {
        SaturationState.DRY: replace(FluidPhase.air(), thermal_conductivity_wmk=float(air_tc)),
        SaturationState.OIL: replace(FluidPhase.oil(), thermal_conductivity_wmk=float(oil_tc)),
        SaturationState.BRINE: replace(FluidPhase.brine(), thermal_conductivity_wmk=float(brine_tc)),
    }


def build_dual_domain_sample(args: argparse.Namespace) -> DualDomainSample:
    sample = SampleDescription(
        name=args.name,
        porosity=float(args.phi_total),
        minerals=[make_dummy_mineral(args.matrix_tc)],
        matrix=make_matrix_properties(args.matrix_tc),
        fluids=make_fluids(args.air_tc, args.oil_tc, args.brine_tc),
        microstructure=MicrostructureParameters(
            aspect_ratio=float(args.alpha_pore),
            connectivity=1.0,
            orientation="isotropic",
            topology="intergranular",
        ),
    )
    dual = DualDomainMicrostructure(
        alpha_pore=float(args.alpha_pore),
        alpha_mc=float(args.alpha_mc),
        phi_mc=float(args.phi_mc),
        ic_phase_tc=float(args.ic_phase_tc),
    )
    validate_inputs(sample, dual)
    return DualDomainSample(sample=sample, dual=dual)


def validate_inputs(sample: SampleDescription, dual: DualDomainMicrostructure) -> None:
    phi_total = float(sample.porosity)
    if not (0.0 < phi_total < 1.0):
        raise ValueError(f"phi_total must lie in (0, 1). Got {phi_total}.")
    if not (0.0 < dual.phi_mc < phi_total):
        raise ValueError(f"phi_mc must lie in (0, phi_total). Got phi_mc={dual.phi_mc}, phi_total={phi_total}.")
    for name, value in {
        "alpha_pore": dual.alpha_pore,
        "alpha_mc": dual.alpha_mc,
        "matrix_tc": sample.matrix.thermal_conductivity_wmk,
        "ic_phase_tc": dual.ic_phase_tc,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive. Got {value}.")


def dual_domain_tc(
    sample: SampleDescription,
    dual: DualDomainMicrostructure,
    state: SaturationState,
    *,
    phi_total: float | None = None,
    matrix_tc: float | None = None,
    fluid_tc: float | None = None,
    alpha_pore: float | None = None,
    phi_mc: float | None = None,
    alpha_mc: float | None = None,
    ic_phase_tc: float | None = None,
) -> float:
    """One-stage effective TC: matrix + pores + IC microcracks together.

    Assumption:
    - total porosity = intergranular pore porosity + IC microcrack porosity;
    - all phases are embedded simultaneously in a single isotropic random GSA mixture;
    - pore fluid follows the saturation state, while the IC domain uses a fixed fill conductivity.
    """
    phi_total_v = float(sample.porosity if phi_total is None else phi_total)
    matrix_tc_v = float(sample.matrix.thermal_conductivity_wmk if matrix_tc is None else matrix_tc)
    fluid_tc_v = float(sample.fluids[state].thermal_conductivity_wmk if fluid_tc is None else fluid_tc)
    alpha_pore_v = float(dual.alpha_pore if alpha_pore is None else alpha_pore)
    phi_mc_v = float(dual.phi_mc if phi_mc is None else phi_mc)
    alpha_mc_v = float(dual.alpha_mc if alpha_mc is None else alpha_mc)
    ic_phase_tc_v = float(dual.ic_phase_tc if ic_phase_tc is None else ic_phase_tc)

    if phi_total_v <= 0.0 or phi_total_v >= 1.0:
        raise ValueError(f"phi_total must lie in (0, 1). Got {phi_total_v}.")
    if phi_mc_v <= 0.0 or phi_mc_v >= phi_total_v:
        raise ValueError(f"phi_mc must lie in (0, phi_total). Got phi_mc={phi_mc_v}, phi_total={phi_total_v}.")
    if min(matrix_tc_v, fluid_tc_v, alpha_pore_v, alpha_mc_v, ic_phase_tc_v) <= 0.0:
        raise ValueError("All conductivities and aspect ratios must be positive.")

    phi_pore_abs = max(phi_total_v - phi_mc_v, EPS)
    phi_matrix = 1.0 - phi_total_v
    phi_sum = phi_matrix + phi_pore_abs + phi_mc_v
    if not np.isclose(phi_sum, 1.0, atol=1e-10):
        raise ValueError(f"Phase fractions must sum to 1.0, got {phi_sum}.")

    phases = [
        make_phase("matrix", phi_matrix, matrix_tc_v, aspect_ratio=1.0, orientation="random"),
        make_phase("pore_fluid", phi_pore_abs, fluid_tc_v, aspect_ratio=alpha_pore_v, orientation="random"),
        make_phase("ic_microcrack", phi_mc_v, ic_phase_tc_v, aspect_ratio=alpha_mc_v, orientation="random"),
    ]
    body = ComparisonBody(kind="matrix", matrix_index=0)
    return float(gsa_transport_isotropic(phases, body))


def predict_states(dds: DualDomainSample) -> dict[str, float]:
    return {
        state_name: dual_domain_tc(dds.sample, dds.dual, state)
        for state_name, state in STATE_ORDER
    }


def safe_perturb(value: float, rel_step: float, lower: float, upper: float) -> tuple[float, float]:
    minus = value * (1.0 - rel_step)
    plus = value * (1.0 + rel_step)
    minus = max(minus, lower)
    plus = min(plus, upper)
    if not (lower < minus < plus < upper):
        # fallback with a tighter step if value is too close to a bound
        half_span = min(value - lower, upper - value) * 0.45
        minus = value - half_span
        plus = value + half_span
    if not (lower < minus < plus < upper):
        raise ValueError(f"Cannot perturb value={value} within bounds ({lower}, {upper}).")
    return minus, plus


def local_log_sensitivity(
    dds: DualDomainSample,
    state: SaturationState,
    parameter: str,
    rel_step: float,
) -> float:
    sample = dds.sample
    dual = dds.dual
    phi_total = float(sample.porosity)
    matrix_tc = float(sample.matrix.thermal_conductivity_wmk)
    fluid_tc = float(sample.fluids[state].thermal_conductivity_wmk)
    alpha_pore = float(dual.alpha_pore)
    phi_mc = float(dual.phi_mc)

    bounds = {
        "phi_total": (EPS, 1.0 - EPS, phi_total),
        "matrix_tc": (EPS, math.inf, matrix_tc),
        "fluid_tc": (EPS, math.inf, fluid_tc),
        "alpha_pore": (EPS, 1.0 - EPS, alpha_pore),
        "phi_mc": (EPS, phi_total - EPS, phi_mc),
    }
    if parameter not in bounds:
        raise KeyError(f"Unsupported parameter '{parameter}'.")

    lower, upper, value = bounds[parameter]
    upper_finite = upper if math.isfinite(upper) else value * (1.0 + 50.0 * rel_step)
    minus, plus = safe_perturb(value, rel_step, lower, upper_finite)

    kwargs_minus: dict[str, float] = {}
    kwargs_plus: dict[str, float] = {}
    if parameter == "phi_total":
        # keep phi_mc inside total porosity and preserve the ratio phi_mc / phi_total
        ratio = phi_mc / phi_total
        kwargs_minus = {"phi_total": minus, "phi_mc": min(max(ratio * minus, EPS), minus - EPS)}
        kwargs_plus = {"phi_total": plus, "phi_mc": min(max(ratio * plus, EPS), plus - EPS)}
    elif parameter == "matrix_tc":
        kwargs_minus = {"matrix_tc": minus}
        kwargs_plus = {"matrix_tc": plus}
    elif parameter == "fluid_tc":
        kwargs_minus = {"fluid_tc": minus}
        kwargs_plus = {"fluid_tc": plus}
    elif parameter == "alpha_pore":
        kwargs_minus = {"alpha_pore": minus}
        kwargs_plus = {"alpha_pore": plus}
    elif parameter == "phi_mc":
        kwargs_minus = {"phi_mc": minus}
        kwargs_plus = {"phi_mc": plus}

    y_minus = dual_domain_tc(sample, dual, state, **kwargs_minus)
    y_plus = dual_domain_tc(sample, dual, state, **kwargs_plus)
    return (math.log(y_plus) - math.log(y_minus)) / (math.log(plus) - math.log(minus))


def sensitivity_table(dds: DualDomainSample, rel_step: float) -> pd.DataFrame:
    parameters = ["phi_total", "matrix_tc", "fluid_tc", "alpha_pore", "phi_mc"]
    rows: list[dict[str, float | str]] = []
    for state_name, state in STATE_ORDER:
        for parameter in parameters:
            rows.append(
                {
                    "state": state_name,
                    "parameter": parameter,
                    "S": local_log_sensitivity(dds, state, parameter, rel_step),
                }
            )
    return pd.DataFrame(rows)


def plot_sensitivity_bars(df: pd.DataFrame, outpath: Path) -> None:
    states = [s for s, _ in STATE_ORDER]
    parameters = ["phi_total", "matrix_tc", "fluid_tc", "alpha_pore", "phi_mc"]
    x = np.arange(len(parameters), dtype=float)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, state in enumerate(states):
        subset = df[df["state"] == state].set_index("parameter").loc[parameters]
        ax.bar(x + (i - 1) * width, subset["S"].to_numpy(), width=width, label=state)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\phi$", r"$\lambda_M$", r"$\lambda_i$", r"$\alpha_{pore}$", r"$\phi_{mc}$"])
    ax.set_ylabel(r"Local normalized sensitivity $S_p = \partial \ln \lambda^* / \partial \ln p$")
    ax.set_title("Saurashtra sandstone: local TC sensitivities")
    ax.legend(title="state")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def response_maps(
    dds: DualDomainSample,
    alpha_pore_bounds: tuple[float, float],
    phi_mc_bounds: tuple[float, float],
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    alpha_vals = np.linspace(alpha_pore_bounds[0], alpha_pore_bounds[1], n_grid)
    phi_mc_vals = np.linspace(phi_mc_bounds[0], phi_mc_bounds[1], n_grid)
    A, P = np.meshgrid(alpha_vals, phi_mc_vals)
    out: dict[str, np.ndarray] = {}
    for state_name, state in STATE_ORDER:
        Z = np.empty_like(A, dtype=float)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                phi_mc = min(P[i, j], float(dds.sample.porosity) - 1e-9)
                Z[i, j] = dual_domain_tc(
                    dds.sample,
                    dds.dual,
                    state,
                    alpha_pore=float(A[i, j]),
                    phi_mc=float(phi_mc),
                )
        out[state_name] = Z
    return A, P, out


def plot_response_maps(A: np.ndarray, P: np.ndarray, maps: dict[str, np.ndarray], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (state_name, _) in zip(axes, STATE_ORDER):
        im = ax.contourf(A, 100.0 * P, maps[state_name], levels=20)
        ax.set_title(state_name)
        ax.set_xlabel(r"$\alpha_{pore}$")
        ax.set_ylabel(r"$\phi_{mc}$, %")
        fig.colorbar(im, ax=ax, label=r"$\lambda^*$, W m$^{-1}$ K$^{-1}$")
    fig.suptitle("Response surfaces in $(\\alpha_{pore}, \\phi_{mc})$")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def measured_tc_dict(args: argparse.Namespace) -> dict[str, float] | None:
    values = {
        "dry": args.tc_meas_dry,
        "oil": args.tc_meas_oil,
        "brine": args.tc_meas_brine,
    }
    if all(v is None for v in values.values()):
        return None
    if any(v is None for v in values.values()):
        raise ValueError("Provide either all measured TC values (--tc-meas-dry/oil/brine) or none.")
    return {k: float(v) for k, v in values.items()}


def misfit_maps(
    maps: dict[str, np.ndarray],
    measured: dict[str, float],
    weights: dict[str, float],
) -> dict[str, np.ndarray]:
    misfits: dict[str, np.ndarray] = {}
    total = None
    for state_name in measured:
        rel = (maps[state_name] - measured[state_name]) / measured[state_name]
        M = weights[state_name] * rel * rel
        misfits[state_name] = M
        total = M if total is None else total + M
    if total is None:
        raise ValueError("No measured values were supplied.")
    misfits["joint"] = total
    return misfits


def plot_misfit_maps(A: np.ndarray, P: np.ndarray, misfits: dict[str, np.ndarray], outpath: Path) -> None:
    panels = ["dry", "oil", "brine", "joint"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for ax, key in zip(axes.flat, panels):
        im = ax.contourf(A, 100.0 * P, misfits[key], levels=20)
        ax.set_title(key)
        ax.set_xlabel(r"$\alpha_{pore}$")
        ax.set_ylabel(r"$\phi_{mc}$, %")
        fig.colorbar(im, ax=ax, label="normalized squared misfit")
    fig.suptitle("Misfit maps in $(\\alpha_{pore}, \\phi_{mc})$")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def alpha_mc_robustness(dds: DualDomainSample, alpha_min: float, alpha_max: float, n: int) -> pd.DataFrame:
    alpha_vals = np.geomspace(alpha_min, alpha_max, n)
    rows: list[dict[str, float | str]] = []
    for alpha_mc in alpha_vals:
        for state_name, state in STATE_ORDER:
            rows.append(
                {
                    "state": state_name,
                    "alpha_mc": float(alpha_mc),
                    "lambda_star": dual_domain_tc(dds.sample, dds.dual, state, alpha_mc=float(alpha_mc)),
                }
            )
    return pd.DataFrame(rows)


def plot_alpha_mc_robustness(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for state_name, _ in STATE_ORDER:
        sub = df[df["state"] == state_name]
        ax.plot(sub["alpha_mc"], sub["lambda_star"], marker="o", markersize=3, label=state_name)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha_{mc}$")
    ax.set_ylabel(r"$\lambda^*$, W m$^{-1}$ K$^{-1}$")
    ax.set_title(r"Robustness of $\lambda^*$ to IC microcrack aspect ratio")
    ax.legend(title="state")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(dds: DualDomainSample, pred: dict[str, float], outpath: Path) -> None:
    phi_total = float(dds.sample.porosity)
    phi_mc = float(dds.dual.phi_mc)
    phi_pore = phi_total - phi_mc
    lines = [
        f"sample = {dds.sample.name}",
        f"phi_total = {phi_total:.6f}",
        f"phi_pore_abs = {phi_pore:.6f}",
        f"phi_mc_abs = {phi_mc:.6f}",
        f"matrix_tc = {dds.sample.matrix.thermal_conductivity_wmk:.6f}",
        f"alpha_pore = {dds.dual.alpha_pore:.6f}",
        f"alpha_mc = {dds.dual.alpha_mc:.6e}",
        f"ic_phase_tc = {dds.dual.ic_phase_tc:.6f}",
        "predicted_tc:",
    ]
    for state_name in [s for s, _ in STATE_ORDER]:
        lines.append(f"  {state_name}: {pred[state_name]:.6f}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dds = build_dual_domain_sample(args)
    pred = predict_states(dds)
    write_summary(dds, pred, args.outdir / "summary.txt")

    sens = sensitivity_table(dds, rel_step=float(args.rel_step))
    sens.to_csv(args.outdir / "local_sensitivities.csv", index=False)
    plot_sensitivity_bars(sens, args.outdir / "local_sensitivities.png")

    A, P, maps = response_maps(
        dds,
        alpha_pore_bounds=(float(args.alpha_pore_min), float(args.alpha_pore_max)),
        phi_mc_bounds=(float(args.phi_mc_min), float(args.phi_mc_max)),
        n_grid=int(args.n_grid),
    )
    plot_response_maps(A, P, maps, args.outdir / "response_maps_alpha_pore_phi_mc.png")

    measured = measured_tc_dict(args)
    if measured is not None:
        weights = {"dry": float(args.w_dry), "oil": float(args.w_oil), "brine": float(args.w_brine)}
        misfits = misfit_maps(maps, measured, weights)
        plot_misfit_maps(A, P, misfits, args.outdir / "misfit_maps_alpha_pore_phi_mc.png")
        pd.DataFrame({k: v.ravel() for k, v in misfits.items()}).to_csv(args.outdir / "misfit_maps_flat.csv", index=False)

    robustness = alpha_mc_robustness(
        dds,
        alpha_min=float(args.alpha_mc_sweep_min),
        alpha_max=float(args.alpha_mc_sweep_max),
        n=int(args.alpha_mc_sweep_n),
    )
    robustness.to_csv(args.outdir / "alpha_mc_robustness.csv", index=False)
    plot_alpha_mc_robustness(robustness, args.outdir / "alpha_mc_robustness.png")

    print(f"Done. Results written to: {args.outdir}")


if __name__ == "__main__":
    main()
