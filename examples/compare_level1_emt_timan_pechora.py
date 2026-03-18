from __future__ import annotations

"""Compare Level-1 EMT thermal-conductivity schemes against Timan-Pechora data.

This script is designed for the RockPhysX repository. It reads the Timan-Pechora
spreadsheet, uses one common aspect ratio per sample fitted jointly across three
states (dry, oil, and 60 g/L brine), and reports thesis-style comparison metrics:

- MAPE (%)
- RMSE (W m^-1 K^-1)
- Bias (%)
- MaxAE (%)
- w95(AR) = AR_97.5 - AR_2.5

Modeling assumptions used here
------------------------------
- Matrix thermal conductivity: 3.00 W m^-1 K^-1
- Dry / air thermal conductivity: 0.025 W m^-1 K^-1
- Oil thermal conductivity: 0.13 W m^-1 K^-1
- Brine (60 g/L) thermal conductivity: 0.60 W m^-1 K^-1

Inversion strategy
------------------
For GSA and SCA, one effective aspect ratio alpha is fitted per sample by solving

    alpha_hat = arg min_alpha sum_i w_i |X_calc_i(alpha) - X_meas_i|

across the three selected saturation states. Bruggeman does not use an aspect-ratio
parameter in the current RockPhysX implementation, so w95(AR) is reported as NaN.

Run from the repository root, for example:

    PYTHONPATH=src python examples/thermal/08_compare_level1_emt_timan_pechora.py \
        --excel /path/to/Tver_ver1.xlsx \
        --outdir results/level1_emt_comparison
"""

from dataclasses import replace
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver
from rockphysx.models.emt.bruggeman import bruggeman_isotropic


DEFAULT_MATRIX_TC = 2.98
DEFAULT_AIR_TC = 0.025
DEFAULT_OIL_TC = 0.13
DEFAULT_brine_avg_TC = 0.60
DEFAULT_ALPHA_BOUNDS = (1e-4, 1.0)
DEFAULT_SHEET = "All properties_data"

STATE_ORDER = [
    ("dry", SaturationState.DRY, "TC air", DEFAULT_AIR_TC),
    ("oil", SaturationState.OIL, "TC oil", DEFAULT_OIL_TC),
    ("brine_avg", SaturationState.BRINE, "TC 60", DEFAULT_brine_avg_TC),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, required=True, help="Path to the Timan-Pechora Excel workbook.")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="Sheet name to read. Default: %(default)s")
    parser.add_argument("--outdir", type=Path, default=Path("results/level1_emt_comparison"), help="Output directory.")
    parser.add_argument("--matrix-tc", type=float, default=DEFAULT_MATRIX_TC)
    parser.add_argument("--air-tc", type=float, default=DEFAULT_AIR_TC)
    parser.add_argument("--oil-tc", type=float, default=DEFAULT_OIL_TC)
    parser.add_argument("--brine_avg-tc", type=float, default=DEFAULT_brine_avg_TC)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_BOUNDS[0])
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_BOUNDS[1])
    return parser.parse_args()



def make_matrix_properties(matrix_tc: float) -> MatrixProperties:
    return MatrixProperties(
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.71,
        thermal_conductivity_wmk=matrix_tc,
        electrical_conductivity_sm=1e-10,
    )



def make_fluids(air_tc: float, oil_tc: float, brine_tc: float) -> dict[SaturationState, FluidPhase]:
    return {
        SaturationState.DRY: replace(FluidPhase.air(), thermal_conductivity_wmk=air_tc),
        SaturationState.OIL: replace(FluidPhase.oil(), thermal_conductivity_wmk=oil_tc),
        SaturationState.BRINE: replace(FluidPhase.brine(), thermal_conductivity_wmk=brine_tc),
    }



def build_sample(porosity_fraction: float, matrix_tc: float, alpha: float, *, air_tc: float, oil_tc: float, brine_tc: float) -> SampleDescription:
    dummy_mineral = MineralPhase(
        name="matrix",
        volume_fraction=1.0,
        bulk_modulus_gpa=70.0,
        shear_modulus_gpa=32.0,
        density_gcc=2.71,
        thermal_conductivity_wmk=matrix_tc,
        electrical_conductivity_sm=1e-10,
    )
    return SampleDescription(
        name="timan_pechora_sample",
        porosity=float(porosity_fraction),
        minerals=[dummy_mineral],
        fluids=make_fluids(air_tc, oil_tc, brine_tc),
        matrix=make_matrix_properties(matrix_tc),
        microstructure=MicrostructureParameters(
            aspect_ratio=float(alpha),
            connectivity=1.0,
            orientation="isotropic",
            topology="intergranular",
        ),
    )



def load_dataset(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    brine_cols = ["TC 0,6", "TC 6", "TC 60", "TC 180"]
    required = ["Sample", "Porosity,%", "TC air", "TC oil"] + brine_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df[required].copy()

    # Среднее по всем доступным brine-состояниям для каждого образца
    out["tc_brine_avg"] = out[brine_cols].mean(axis=1, skipna=True)

    out = out.rename(
        columns={
            "Porosity,%": "porosity_pct",
            "TC air": "tc_air",
            "TC oil": "tc_oil",
        }
    )

    out = out.dropna(subset=["Sample", "porosity_pct", "tc_air", "tc_oil", "tc_brine_avg"])
    out["porosity"] = out["porosity_pct"] / 100.0
    out = out[(out["porosity"] > 0.0) & (out["porosity"] < 1.0)]
    out = out[(out[["tc_air", "tc_oil", "tc_brine_avg"]] > 0.0).all(axis=1)]
    out = out.reset_index(drop=True)

    print(f"Mean brine TC across all salinities and all samples: {out['tc_brine_avg'].mean():.4f} W m^-1 K^-1")
    return out



# def predict_three_states(sample: SampleDescription, model: str, solver: ForwardSolver) -> dict[str, float]:
#     return {
#         "dry": float(solver.predict("thermal_conductivity", sample, SaturationState.DRY, model=model)),
#         "oil": float(solver.predict("thermal_conductivity", sample, SaturationState.OIL, model=model)),
#         "brine_avg": float(solver.predict("thermal_conductivity", sample, SaturationState.BRINE, model=model)),
#     }
def predict_three_states(sample: SampleDescription, model: str, solver: ForwardSolver) -> dict[str, float]:
    return {
        "dry": float(solver.predict("thermal_conductivity", sample, SaturationState.DRY, model=model)),
        "oil": float(solver.predict("thermal_conductivity", sample, SaturationState.OIL, model=model)),
        "brine_avg": float(solver.predict("thermal_conductivity", sample, SaturationState.BRINE, model=model)),
    }



# def predict_three_states_bruggeman(porosity: float, matrix_tc: float, *, air_tc: float, oil_tc: float, brine_tc: float) -> dict[str, float]:
#     solid = 1.0 - porosity
#     return {
#         "dry": float(bruggeman_isotropic([solid, porosity], [matrix_tc, air_tc])),
#         "oil": float(bruggeman_isotropic([solid, porosity], [matrix_tc, oil_tc])),
#         "brine_avg": float(bruggeman_isotropic([solid, porosity], [matrix_tc, brine_tc])),
#     }

def predict_three_states_bruggeman(
    porosity: float,
    matrix_tc: float,
    *,
    air_tc: float,
    oil_tc: float,
    brine_tc: float,
) -> dict[str, float]:
    solid = 1.0 - porosity
    return {
        "dry": float(bruggeman_isotropic([solid, porosity], [matrix_tc, air_tc])),
        "oil": float(bruggeman_isotropic([solid, porosity], [matrix_tc, oil_tc])),
        "brine_avg": float(bruggeman_isotropic([solid, porosity], [matrix_tc, brine_tc])),
    }



def objective_alpha(log10_alpha: float, porosity: float, measured: dict[str, float], model: str, solver: ForwardSolver, *, matrix_tc: float, air_tc: float, oil_tc: float, brine_tc: float, weights: dict[str, float]) -> float:
    alpha = 10.0 ** log10_alpha
    sample = build_sample(porosity, matrix_tc, alpha, air_tc=air_tc, oil_tc=oil_tc, brine_tc=brine_tc)
    predicted = predict_three_states(sample, model, solver)
    return float(sum(weights[state] * abs(predicted[state] - measured[state]) for state in measured))


def fit_alpha_for_sample(
    porosity: float,
    measured: dict[str, float],
    model: str,
    solver: ForwardSolver,
    *,
    alpha_min: float,
    alpha_max: float,
    matrix_tc: float,
    air_tc: float,
    oil_tc: float,
    brine_tc: float,
    weights: dict[str, float],
) -> tuple[float, dict[str, float], float]:
    def objective(log10_alpha: float) -> float:
        return objective_alpha(
            log10_alpha,
            porosity,
            measured,
            model,
            solver,
            matrix_tc=matrix_tc,
            air_tc=air_tc,
            oil_tc=oil_tc,
            brine_tc=brine_tc,
            weights=weights,
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log10(alpha_min), math.log10(alpha_max)),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(f"Aspect-ratio optimization failed: {result.message}")

    alpha_hat = 10.0 ** float(result.x)
    sample = build_sample(
        porosity,
        matrix_tc,
        alpha_hat,
        air_tc=air_tc,
        oil_tc=oil_tc,
        brine_tc=brine_tc,
    )
    predicted = predict_three_states(sample, model, solver)
    return alpha_hat, predicted, float(result.fun)



def evaluate_models(df: pd.DataFrame, *, matrix_tc: float, air_tc: float, oil_tc: float, brine_tc: float, alpha_min: float, alpha_max: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    solver = ForwardSolver()
    weights = {"dry": 1.0, "oil": 1.0, "brine_avg": 1.0}

    sample_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        sample_id = row["Sample"]
        porosity = float(row["porosity"])
        # measured = {
        #     "dry": float(row["tc_air"]),
        #     "oil": float(row["tc_oil"]),
        #     "brine_avg": float(row["tc_60"]),
        # }
        measured = {
            "dry": float(row["tc_air"]),
            "oil": float(row["tc_oil"]),
            "brine_avg": float(row["tc_brine_avg"]),
}

        # GSA / SCA: fit one alpha across all three states
        for model in ("gsa", "sca"):
            alpha_hat, predicted, objective_value = fit_alpha_for_sample(
                porosity,
                measured,
                model,
                solver,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                matrix_tc=matrix_tc,
                air_tc=air_tc,
                oil_tc=oil_tc,
                brine_tc=brine_tc,
                weights=weights,
            )
            sample_rows.append(
                {
                    "sample": sample_id,
                    "porosity": porosity,
                    "model": model.upper(),
                    "alpha_fit": alpha_hat,
                    "objective_L1": objective_value,
                }
            )
            for state in ("dry", "oil", "brine_avg"):
                point_rows.append(
                    {
                        "sample": sample_id,
                        "porosity": porosity,
                        "model": model.upper(),
                        "state": state,
                        "alpha_fit": alpha_hat,
                        "measured": measured[state],
                        "predicted": predicted[state],
                    }
                )

        # Bruggeman: direct prediction, no AR parameter
        predicted_br = predict_three_states_bruggeman(porosity, matrix_tc, air_tc=air_tc, oil_tc=oil_tc, brine_tc=brine_tc)
        sample_rows.append(
            {
                "sample": sample_id,
                "porosity": porosity,
                "model": "BRUGGEMAN",
                "alpha_fit": np.nan,
                "objective_L1": sum(abs(predicted_br[s] - measured[s]) for s in measured),
            }
        )
        for state in ("dry", "oil", "brine_avg"):
            point_rows.append(
                {
                    "sample": sample_id,
                    "porosity": porosity,
                    "model": "BRUGGEMAN",
                    "state": state,
                    "alpha_fit": np.nan,
                    "measured": measured[state],
                    "predicted": predicted_br[state],
                }
            )

    points = pd.DataFrame(point_rows)
    points["error"] = points["predicted"] - points["measured"]
    points["ape_pct"] = (points["error"].abs() / points["measured"]) * 100.0
    points["signed_pct"] = (points["error"] / points["measured"]) * 100.0

    per_sample = pd.DataFrame(sample_rows)

    summary_rows = []
    for model, grp in points.groupby("model", sort=False):
        alpha_grp = per_sample.loc[(per_sample["model"] == model) & per_sample["alpha_fit"].notna(), "alpha_fit"]
        if len(alpha_grp) > 0:
            w95 = float(np.percentile(alpha_grp, 97.5) - np.percentile(alpha_grp, 2.5))
        else:
            w95 = np.nan
        summary_rows.append(
            {
                "Model": model,
                "MAPE (%)": float(grp["ape_pct"].mean()),
                "RMSE (W m^-1 K^-1)": float(np.sqrt(np.mean(grp["error"] ** 2))),
                "Bias (%)": float(grp["signed_pct"].mean()),
                "MaxAE (%)": float(grp["ape_pct"].max()),
                "w95(AR)": w95,
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary, per_sample, points



def by_state_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, state), grp in points.groupby(["model", "state"], sort=False):
        rows.append(
            {
                "model": model,
                "state": state,
                "n": int(len(grp)),
                "MAPE (%)": float(grp["ape_pct"].mean()),
                "RMSE (W m^-1 K^-1)": float(np.sqrt(np.mean(grp["error"] ** 2))),
                "Bias (%)": float(grp["signed_pct"].mean()),
                "MaxAE (%)": float(grp["ape_pct"].max()),
            }
        )
    return pd.DataFrame(rows)

def alpha_fit_summary(per_sample: pd.DataFrame) -> pd.DataFrame:
    alpha_df = per_sample.loc[per_sample["alpha_fit"].notna()].copy()

    rows = []
    for model, grp in alpha_df.groupby("model", sort=False):
        values = grp["alpha_fit"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "n": int(len(values)),
                "AR_min": float(np.min(values)),
                "AR_mean": float(np.mean(values)),
                "AR_median": float(np.median(values)),
                "AR_max": float(np.max(values)),
                "AR_p2.5": float(np.percentile(values, 2.5)),
                "AR_p97.5": float(np.percentile(values, 97.5)),
                "w95(AR)": float(np.percentile(values, 97.5) - np.percentile(values, 2.5)),
            }
        )

    return pd.DataFrame(rows)



def format_markdown_table(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for col in ["MAPE (%)", "RMSE (W m^-1 K^-1)", "Bias (%)", "MaxAE (%)", "w95(AR)"]:
        display[col] = display[col].map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
    return display.to_markdown(index=False)



def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.excel, args.sheet)
    summary, per_sample, points = evaluate_models(
        df,
        matrix_tc=args.matrix_tc,
        air_tc=args.air_tc,
        oil_tc=args.oil_tc,
        brine_tc=args.brine_avg_tc,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )
    state_table = by_state_summary(points)
    alpha_table = alpha_fit_summary(per_sample)

    summary_csv = args.outdir / "table12_level1_emt_comparison.csv"
    sample_csv = args.outdir / "table12_per_sample_alpha_fits.csv"
    points_csv = args.outdir / "table12_point_errors.csv"
    state_csv = args.outdir / "table12_by_state.csv"
    alpha_csv = args.outdir / "table12_alpha_summary.csv"

    summary.to_csv(summary_csv, index=False)
    per_sample.to_csv(sample_csv, index=False)
    points.to_csv(points_csv, index=False)
    state_table.to_csv(state_csv, index=False)
    alpha_table.to_csv(alpha_csv, index=False)

    xlsx_path = args.outdir / "table12_level1_emt_comparison.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        per_sample.to_excel(writer, sheet_name="AR_Fits", index=False)
        points.to_excel(writer, sheet_name="Point_Errors", index=False)
        state_table.to_excel(writer, sheet_name="By_State", index=False)
        alpha_table.to_excel(writer, sheet_name="Alpha_Summary", index=False)

    md_path = args.outdir / "table12_level1_emt_comparison.md"
    notes = (
        "Notes: MAPE = mean absolute percentage error; RMSE = root mean squared error; "
        "Bias = mean signed percentage error; MaxAE = maximum absolute percentage error; "
        "w95(AR) = AR97.5 − AR2.5. Bruggeman has no aspect-ratio parameter in the current implementation, "
        "so w95(AR) is reported as n/a.\n"
    )
    md_text = (
        "Table 12 — Comparison of Level-1 EMT schemes for thermal-conductivity prediction "
        "in Timan–Pechora carbonates\n\n"
        + format_markdown_table(summary)
        + "\n\n"
        + notes
    )
    md_path.write_text(md_text, encoding="utf-8")

    print(md_text)
    print("\nAspect-ratio summary by model:\n")
    print(alpha_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
