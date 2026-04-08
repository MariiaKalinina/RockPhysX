from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit, betaln


INPUT_XLSX = Path(r"/mnt/data/data_restructured_for_MT_v2.xlsx")
OUTPUT_XLSX = Path(r"/mnt/data/mt_tc_m1_m2_results.xlsx")

SHEET_MEASUREMENTS = "measurements_long"
SHEET_CONSTANTS = "sample_constants"
SHEET_UNCERTAINTY = "uncertainty"

# Smaller quadrature grid for a fast first implementation
N_QUAD = 61
U_GRID = (np.arange(N_QUAD) + 0.5) / N_QUAD
ASPECT_RATIO_GRID = 10.0 ** (-4.0 + 4.0 * U_GRID)


@dataclass
class MTConfig:
    lambda_matrix: float
    lambda_water: float
    lambda_gas: float
    tc_rel_sigma: float


def load_workbook_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    measurements = pd.read_excel(path, sheet_name=SHEET_MEASUREMENTS)
    constants = pd.read_excel(path, sheet_name=SHEET_CONSTANTS)
    uncertainty = pd.read_excel(path, sheet_name=SHEET_UNCERTAINTY)

    if "phi_frac" in measurements.columns and "phi_pct" in measurements.columns:
        measurements["phi_frac"] = measurements["phi_frac"].where(
            measurements["phi_frac"].notna(), measurements["phi_pct"] / 100.0
        )

    for col in ["stage", "fluid_state", "field", "well"]:
        if col in measurements.columns:
            measurements[col] = measurements[col].astype(str).str.strip()

    return measurements, constants, uncertainty


def scalar_constant(constants_df: pd.DataFrame, parameter: str, fallback: float | None = None) -> float:
    row = constants_df.loc[constants_df["parameter"] == parameter]
    if row.empty:
        if fallback is None:
            raise KeyError(f"Constant '{parameter}' not found in sample_constants.")
        return float(fallback)
    return float(row["default_value"].iloc[0])


def relative_uncertainty(unc_df: pd.DataFrame, property_name: str, fallback: float | None = None) -> float:
    row = unc_df.loc[unc_df["property"] == property_name]
    if row.empty:
        if fallback is None:
            raise KeyError(f"Uncertainty for '{property_name}' not found.")
        return float(fallback)
    return float(row["relative_sigma"].iloc[0])


def build_config(constants_df: pd.DataFrame, uncertainty_df: pd.DataFrame) -> MTConfig:
    return MTConfig(
        lambda_matrix=scalar_constant(constants_df, "lambda_matrix_w_mk", 2.86),
        lambda_water=scalar_constant(constants_df, "lambda_water_w_mk", 0.60),
        lambda_gas=scalar_constant(constants_df, "lambda_gas_w_mk", 0.025),
        tc_rel_sigma=relative_uncertainty(uncertainty_df, "tc_w_mk", 0.025),
    )


def build_tc_dataset(measurements_df: pd.DataFrame, config: MTConfig) -> pd.DataFrame:
    df = measurements_df.copy()
    required = ["lab_sample_id", "stage", "fluid_state", "phi_frac", "tc_w_mk"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Measurements sheet is missing required columns: {missing_cols}")

    df = df.dropna(subset=required).copy()
    df["sample_id"] = df["lab_sample_id"].astype(str)
    df["phi_frac"] = pd.to_numeric(df["phi_frac"], errors="coerce")
    df["tc_w_mk"] = pd.to_numeric(df["tc_w_mk"], errors="coerce")
    df["lambda_fluid_w_mk"] = np.where(
        df["fluid_state"].str.lower() == "dry",
        config.lambda_gas,
        config.lambda_water,
    )

    keep_cols = [
        "sample_id", "lab_sample_id", "field", "well", "depth_m", "tg_position_mm",
        "stage", "fluid_state", "phi_frac", "phi_pct", "tc_w_mk", "lambda_fluid_w_mk"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].sort_values(["sample_id", "stage", "fluid_state"]).reset_index(drop=True)


def depolarization_factors_spheroid(aspect_ratio):
    r = np.asarray(aspect_ratio, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("Aspect ratio must be positive and finite.")

    n3 = np.empty_like(r)
    sphere = np.isclose(r, 1.0)
    oblate = r < 1.0
    prolate = r > 1.0

    n3[sphere] = 1.0 / 3.0

    rr = r[oblate]
    if rr.size > 0:
        xi = np.sqrt(np.maximum(1.0 / (rr * rr) - 1.0, 0.0))
        n3[oblate] = ((1.0 + xi * xi) / (xi ** 3)) * (xi - np.arctan(xi))

    rr = r[prolate]
    if rr.size > 0:
        e = np.sqrt(np.maximum(1.0 - 1.0 / (rr * rr), 0.0))
        n3[prolate] = ((1.0 - e * e) / (2.0 * e ** 3)) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)

    n1 = (1.0 - n3) / 2.0
    return n1, n3


# Precompute depolarization factors for the beta model grid
N1_GRID, N3_GRID = depolarization_factors_spheroid(ASPECT_RATIO_GRID)


def mt_tc_single_aspect_ratio(phi: float, aspect_ratio: float | np.ndarray, lambda_matrix: float, lambda_fluid: float):
    phi = float(phi)
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"Porosity must be in [0,1). Got {phi}")

    r = np.asarray(aspect_ratio, dtype=float)
    n1, n3 = depolarization_factors_spheroid(r)
    a1 = lambda_matrix / (lambda_matrix + n1 * (lambda_fluid - lambda_matrix))
    a3 = lambda_matrix / (lambda_matrix + n3 * (lambda_fluid - lambda_matrix))
    a_bar = (2.0 * a1 + a3) / 3.0

    c_i = phi
    c_m = 1.0 - phi
    delta = lambda_fluid - lambda_matrix
    value = lambda_matrix + c_i * delta * a_bar / (c_m + c_i * a_bar)

    if np.ndim(value) == 0:
        return float(value)
    return value


def beta_weights(m: float, kappa: float) -> np.ndarray:
    if not (0.0 < m < 1.0):
        raise ValueError(f"m must be in (0,1). Got {m}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive. Got {kappa}")

    a = max(m * kappa, 1e-8)
    b = max((1.0 - m) * kappa, 1e-8)
    logw = (a - 1.0) * np.log(U_GRID) + (b - 1.0) * np.log(1.0 - U_GRID) - betaln(a, b)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    return w / np.sum(w)


def mt_tc_beta_aspect_ratio(phi: float, m: float, kappa: float, lambda_matrix: float, lambda_fluid: float) -> float:
    w = beta_weights(m, kappa)
    values = mt_tc_single_aspect_ratio(
        phi=phi,
        aspect_ratio=ASPECT_RATIO_GRID,
        lambda_matrix=lambda_matrix,
        lambda_fluid=lambda_fluid,
    )
    return float(np.sum(w * values))


def unpack_m1_params(x: np.ndarray) -> dict[str, float]:
    q_before = expit(x[0])
    q_after = expit(x[0] + x[1])
    z_before = -4.0 + 4.0 * q_before
    z_after = -4.0 + 4.0 * q_after
    ar_before = 10.0 ** z_before
    ar_after = 10.0 ** z_after
    return {
        "q_before": float(q_before),
        "q_after": float(q_after),
        "z_before": float(z_before),
        "z_after": float(z_after),
        "delta_z": float(z_after - z_before),
        "ar_before": float(ar_before),
        "ar_after": float(ar_after),
        "ar_ratio_after_before": float(ar_after / ar_before),
    }


def unpack_m2_params(x: np.ndarray) -> dict[str, float]:
    m_before = float(expit(x[0]))
    m_after = float(expit(x[0] + x[1]))
    kappa_before = float(2.0 + np.exp(x[2]))
    kappa_after = float(kappa_before * np.exp(x[3]))
    return {
        "m_before": m_before,
        "m_after": m_after,
        "delta_m": float(m_after - m_before),
        "delta_logit_m": float(x[1]),
        "kappa_before": kappa_before,
        "kappa_after": kappa_after,
        "delta_kappa": float(kappa_after - kappa_before),
        "kappa_ratio_after_before": float(kappa_after / kappa_before),
        "delta_log_kappa": float(x[3]),
    }


def lognormal_nll(obs: float, pred: float, sigma_rel: float) -> float:
    sigma_log = math.log1p(sigma_rel)
    return 0.5 * ((math.log(obs) - math.log(pred)) / sigma_log) ** 2 + math.log(sigma_log)


def nlp_m1(x: np.ndarray, sample_df: pd.DataFrame, config: MTConfig) -> float:
    p = unpack_m1_params(x)
    nlp = 0.0
    for _, row in sample_df.iterrows():
        ar = p["ar_before"] if row["stage"] == "before" else p["ar_after"]
        pred = mt_tc_single_aspect_ratio(row["phi_frac"], ar, config.lambda_matrix, row["lambda_fluid_w_mk"])
        if pred <= 0 or not np.isfinite(pred):
            return 1e12
        nlp += lognormal_nll(float(row["tc_w_mk"]), float(pred), config.tc_rel_sigma)
    nlp += 0.5 * (x[0] / 2.0) ** 2 + 0.5 * (x[1] / 1.5) ** 2
    return float(nlp)


def nlp_m2(x: np.ndarray, sample_df: pd.DataFrame, config: MTConfig) -> float:
    p = unpack_m2_params(x)
    nlp = 0.0
    for _, row in sample_df.iterrows():
        if row["stage"] == "before":
            pred = mt_tc_beta_aspect_ratio(row["phi_frac"], p["m_before"], p["kappa_before"], config.lambda_matrix, row["lambda_fluid_w_mk"])
        else:
            pred = mt_tc_beta_aspect_ratio(row["phi_frac"], p["m_after"], p["kappa_after"], config.lambda_matrix, row["lambda_fluid_w_mk"])
        if pred <= 0 or not np.isfinite(pred):
            return 1e12
        nlp += lognormal_nll(float(row["tc_w_mk"]), float(pred), config.tc_rel_sigma)
    nlp += 0.5 * (x[0] / 2.0) ** 2
    nlp += 0.5 * (x[1] / 1.5) ** 2
    nlp += 0.5 * ((x[2] - math.log(8.0)) / 1.0) ** 2
    nlp += 0.5 * (x[3] / 0.8) ** 2
    return float(nlp)


def multi_start_optimize(fun, starts: Iterable[np.ndarray]):
    best = None
    for x0 in starts:
        res = minimize(fun, x0=np.asarray(x0, dtype=float), method="Powell",
                       options={"maxiter": 80, "xtol": 1e-3, "ftol": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    return best


def fit_sample_m1(sample_df: pd.DataFrame, config: MTConfig) -> dict[str, float]:
    starts = [
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.5]),
        np.array([1.0, -0.5]),
    ]
    res = multi_start_optimize(lambda x: nlp_m1(x, sample_df, config), starts)
    out = {
        "sample_id": sample_df["sample_id"].iloc[0],
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "objective_nlp": float(res.fun),
        "n_obs": int(len(sample_df)),
    }
    out.update(unpack_m1_params(res.x))
    return out


def fit_sample_m2(sample_df: pd.DataFrame, config: MTConfig) -> dict[str, float]:
    starts = [
        np.array([0.0, 0.0, math.log(8.0), 0.0]),
        np.array([-1.0, 0.5, math.log(6.0), 0.0]),
        np.array([1.0, -0.5, math.log(12.0), 0.1]),
    ]
    res = multi_start_optimize(lambda x: nlp_m2(x, sample_df, config), starts)
    out = {
        "sample_id": sample_df["sample_id"].iloc[0],
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "objective_nlp": float(res.fun),
        "n_obs": int(len(sample_df)),
    }
    out.update(unpack_m2_params(res.x))
    return out


def predict_sample_m1(sample_df: pd.DataFrame, fit_row: dict[str, float], config: MTConfig) -> pd.DataFrame:
    rows = []
    for _, row in sample_df.iterrows():
        ar = fit_row["ar_before"] if row["stage"] == "before" else fit_row["ar_after"]
        pred = mt_tc_single_aspect_ratio(row["phi_frac"], ar, config.lambda_matrix, row["lambda_fluid_w_mk"])
        rows.append({
            "sample_id": row["sample_id"],
            "stage": row["stage"],
            "fluid_state": row["fluid_state"],
            "phi_frac": row["phi_frac"],
            "lambda_fluid_w_mk": row["lambda_fluid_w_mk"],
            "tc_obs_w_mk": row["tc_w_mk"],
            "tc_pred_w_mk": pred,
            "residual_w_mk": pred - row["tc_w_mk"],
            "log_residual": math.log(pred) - math.log(row["tc_w_mk"]),
            "model": "M1_single_effective_AR",
        })
    return pd.DataFrame(rows)


def predict_sample_m2(sample_df: pd.DataFrame, fit_row: dict[str, float], config: MTConfig) -> pd.DataFrame:
    rows = []
    for _, row in sample_df.iterrows():
        if row["stage"] == "before":
            pred = mt_tc_beta_aspect_ratio(row["phi_frac"], fit_row["m_before"], fit_row["kappa_before"], config.lambda_matrix, row["lambda_fluid_w_mk"])
        else:
            pred = mt_tc_beta_aspect_ratio(row["phi_frac"], fit_row["m_after"], fit_row["kappa_after"], config.lambda_matrix, row["lambda_fluid_w_mk"])
        rows.append({
            "sample_id": row["sample_id"],
            "stage": row["stage"],
            "fluid_state": row["fluid_state"],
            "phi_frac": row["phi_frac"],
            "lambda_fluid_w_mk": row["lambda_fluid_w_mk"],
            "tc_obs_w_mk": row["tc_w_mk"],
            "tc_pred_w_mk": pred,
            "residual_w_mk": pred - row["tc_w_mk"],
            "log_residual": math.log(pred) - math.log(row["tc_w_mk"]),
            "model": "M2_beta_AR_distribution",
        })
    return pd.DataFrame(rows)


def main() -> None:
    measurements, constants, uncertainty = load_workbook_data(INPUT_XLSX)
    config = build_config(constants, uncertainty)
    tc_df = build_tc_dataset(measurements, config)

    m1_rows, m2_rows, pred_frames = [], [], []
    for sample_id, g in tc_df.groupby("sample_id"):
        g = g.sort_values(["stage", "fluid_state"]).reset_index(drop=True)
        fit1 = fit_sample_m1(g, config)
        fit2 = fit_sample_m2(g, config)
        m1_rows.append(fit1)
        m2_rows.append(fit2)
        pred_frames.append(predict_sample_m1(g, fit1, config))
        pred_frames.append(predict_sample_m2(g, fit2, config))

    m1_df = pd.DataFrame(m1_rows).sort_values("sample_id")
    m2_df = pd.DataFrame(m2_rows).sort_values("sample_id")
    pred_df = pd.concat(pred_frames, ignore_index=True)

    fit_quality = (
        pred_df.groupby("model")
        .agg(
            n_obs=("tc_obs_w_mk", "size"),
            rmse_w_mk=("residual_w_mk", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mean_abs_w_mk=("residual_w_mk", lambda x: float(np.mean(np.abs(x)))),
            rmse_log=("log_residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )

    config_df = pd.DataFrame(
        [
            ("lambda_matrix_w_mk", config.lambda_matrix),
            ("lambda_water_w_mk", config.lambda_water),
            ("lambda_gas_w_mk", config.lambda_gas),
            ("tc_rel_sigma", config.tc_rel_sigma),
            ("n_quad_beta_model", N_QUAD),
        ],
        columns=["parameter", "value"],
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        tc_df.to_excel(writer, sheet_name="tc_dataset", index=False)
        m1_df.to_excel(writer, sheet_name="M1_fits", index=False)
        m2_df.to_excel(writer, sheet_name="M2_fits", index=False)
        pred_df.to_excel(writer, sheet_name="predictions", index=False)
        fit_quality.to_excel(writer, sheet_name="fit_quality", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

    print(f"Saved results to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
