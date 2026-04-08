from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, betaln

MODULE_DIR = Path(r"/mnt/data")
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from strict_mt_elastic_pores import ElasticPhase, strict_mt_elastic_random_spheroidal_pores

DATA_XLSX = Path(r"/mnt/data/data_restructured_for_MT_v2.xlsx")
PREV_XLSX = Path(r"/mnt/data/mt_tc_m1_m2_results.xlsx")
RESULT_XLSX = Path(r"/mnt/data/strict_mt_tc_vp_vs_inversion_fast_results.xlsx")

SHEET_MEASUREMENTS = "measurements_long"
SHEET_CONSTANTS = "sample_constants"
SHEET_UNCERTAINTY = "uncertainty"

OBJECTIVE = "rel_l1"

# Shared AR grid for precomputation and M2 quadrature
N_GRID = 31
U_GRID = (np.arange(N_GRID) + 0.5) / N_GRID
Z_GRID = -4.0 + 4.0 * U_GRID
AR_GRID = 10.0 ** Z_GRID
NU_STUDENT = 4.0


def load_inputs():
    measurements = pd.read_excel(DATA_XLSX, sheet_name=SHEET_MEASUREMENTS)
    constants = pd.read_excel(DATA_XLSX, sheet_name=SHEET_CONSTANTS)
    uncertainty = pd.read_excel(DATA_XLSX, sheet_name=SHEET_UNCERTAINTY)
    prev_m1 = pd.read_excel(PREV_XLSX, sheet_name="M1_fits")
    prev_m2 = pd.read_excel(PREV_XLSX, sheet_name="M2_fits")

    if "phi_frac" in measurements.columns and "phi_pct" in measurements.columns:
        measurements["phi_frac"] = measurements["phi_frac"].where(
            measurements["phi_frac"].notna(), measurements["phi_pct"] / 100.0
        )
    for col in ["stage", "fluid_state"]:
        if col in measurements.columns:
            measurements[col] = measurements[col].astype(str).str.strip()
    return measurements, constants, uncertainty, prev_m1, prev_m2


def scalar_constant(constants_df: pd.DataFrame, parameter: str, fallback: float | None = None) -> float:
    row = constants_df.loc[constants_df["parameter"] == parameter]
    if row.empty:
        if fallback is None:
            raise KeyError(f"Constant '{parameter}' not found.")
        return float(fallback)
    return float(row["default_value"].iloc[0])


def relative_uncertainty(unc_df: pd.DataFrame, property_name: str, fallback: float | None = None) -> float:
    row = unc_df.loc[unc_df["property"] == property_name]
    if row.empty:
        if fallback is None:
            raise KeyError(f"Uncertainty for '{property_name}' not found.")
        return float(fallback)
    return float(row["relative_sigma"].iloc[0])


def build_dataset(measurements_df: pd.DataFrame) -> pd.DataFrame:
    req = ["lab_sample_id", "stage", "fluid_state", "phi_frac", "tc_w_mk", "vp_m_s", "vs_m_s", "bulk_density_g_cm3"]
    df = measurements_df.dropna(subset=req).copy()
    df["sample_id"] = df["lab_sample_id"].astype(str)
    df["phi_frac"] = pd.to_numeric(df["phi_frac"], errors="coerce")
    df["tc_w_mk"] = pd.to_numeric(df["tc_w_mk"], errors="coerce")
    df["vp_m_s"] = pd.to_numeric(df["vp_m_s"], errors="coerce")
    df["vs_m_s"] = pd.to_numeric(df["vs_m_s"], errors="coerce")
    df["rho_bulk_kg_m3"] = pd.to_numeric(df["bulk_density_g_cm3"], errors="coerce") * 1000.0

    keep_cols = [
        "sample_id", "lab_sample_id", "stage", "fluid_state",
        "phi_frac", "phi_pct", "rho_bulk_kg_m3",
        "tc_w_mk", "vp_m_s", "vs_m_s",
        "field", "well", "depth_m", "tg_position_mm"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].sort_values(["sample_id", "stage", "fluid_state"]).reset_index(drop=True)


def build_property_inputs(row: pd.Series, constants_df: pd.DataFrame):
    lambda_matrix = scalar_constant(constants_df, "lambda_matrix_w_mk", 2.86)
    is_dry = str(row["fluid_state"]).lower() == "dry"
    lambda_fluid = scalar_constant(constants_df, "lambda_gas_w_mk", 0.025) if is_dry else scalar_constant(constants_df, "lambda_water_w_mk", 0.60)

    Km = scalar_constant(constants_df, "k_matrix_gpa", 76.8) * 1e9
    Gm = scalar_constant(constants_df, "g_matrix_gpa", 32.0) * 1e9
    Kf = scalar_constant(constants_df, "k_gas_gpa", 0.00014) * 1e9 if is_dry else scalar_constant(constants_df, "k_water_gpa", 2.2) * 1e9
    Gf = 1e-9
    rho_m = scalar_constant(constants_df, "rho_matrix_kg_m3", 2710.0)
    rho_f = 1.2 if is_dry else 1000.0
    return lambda_matrix, lambda_fluid, Km, Gm, Kf, Gf, rho_m, rho_f


def mt_tc_single(phi: float, a_ratio: float, lambda_matrix: float, lambda_fluid: float) -> float:
    r = float(a_ratio)
    if abs(r - 1.0) < 1e-12:
        n3 = 1.0 / 3.0
    elif r < 1.0:
        xi = math.sqrt(max(1.0 / (r * r) - 1.0, 0.0))
        n3 = ((1.0 + xi * xi) / (xi ** 3)) * (xi - math.atan(xi))
    else:
        e = math.sqrt(max(1.0 - 1.0 / (r * r), 0.0))
        n3 = ((1.0 - e * e) / (2.0 * e ** 3)) * (math.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
    n1 = (1.0 - n3) / 2.0
    a1 = lambda_matrix / (lambda_matrix + n1 * (lambda_fluid - lambda_matrix))
    a3 = lambda_matrix / (lambda_matrix + n3 * (lambda_fluid - lambda_matrix))
    a_bar = (2.0 * a1 + a3) / 3.0
    c_i = phi
    c_m = 1.0 - phi
    delta = lambda_fluid - lambda_matrix
    return float(lambda_matrix + c_i * delta * a_bar / (c_m + c_i * a_bar))


def precompute_grids(df: pd.DataFrame, constants_df: pd.DataFrame):
    recs = []
    for idx, row in df.iterrows():
        lambda_matrix, lambda_fluid, Km, Gm, Kf, Gf, rho_m, rho_f = build_property_inputs(row, constants_df)
        tc_vals = []
        vp_vals = []
        vs_vals = []
        K_vals = []
        G_vals = []
        for ar in AR_GRID:
            tc_vals.append(mt_tc_single(row["phi_frac"], float(ar), lambda_matrix, lambda_fluid))
            elastic = strict_mt_elastic_random_spheroidal_pores(
                phi=float(row["phi_frac"]),
                a_ratio=float(ar),
                matrix=ElasticPhase(K=Km, G=Gm),
                inclusion=ElasticPhase(K=Kf, G=Gf),
                rho_matrix_kg_m3=float(rho_m),
                rho_inclusion_kg_m3=float(rho_f),
            )
            vp_vals.append(elastic.vp_m_s)
            vs_vals.append(elastic.vs_m_s)
            K_vals.append(elastic.K_eff / 1e9)
            G_vals.append(elastic.G_eff / 1e9)
        recs.append({
            "obs_index": int(idx),
            "tc_grid": np.array(tc_vals, dtype=float),
            "vp_grid": np.array(vp_vals, dtype=float),
            "vs_grid": np.array(vs_vals, dtype=float),
            "K_grid": np.array(K_vals, dtype=float),
            "G_grid": np.array(G_vals, dtype=float),
        })
    return pd.DataFrame(recs)


def beta_weights(m: float, kappa: float) -> np.ndarray:
    a = max(m * kappa, 1e-8)
    b = max((1.0 - m) * kappa, 1e-8)
    logw = (a - 1.0) * np.log(U_GRID) + (b - 1.0) * np.log(1.0 - U_GRID) - betaln(a, b)
    logw -= np.max(logw)
    w = np.exp(logw)
    return w / np.sum(w)


def interp_on_z(grid_vals: np.ndarray, z_value: float) -> float:
    return float(np.interp(z_value, Z_GRID, grid_vals))


def unpack_m1_params(x):
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


def unpack_m2_params(x):
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


def objective_term(obs: float, pred: float, sigma_rel: float, kind: str) -> float:
    rel = (pred - obs) / obs
    sigma_log = math.log1p(sigma_rel)
    log_res = (math.log(pred) - math.log(obs)) / sigma_log
    if kind == "rel_l1":
        return abs(rel) / sigma_rel
    if kind == "rel_l2":
        return (rel / sigma_rel) ** 2
    if kind == "log_gaussian":
        return 0.5 * log_res ** 2 + math.log(sigma_log)
    if kind == "log_student_t":
        return 0.5 * (NU_STUDENT + 1.0) * math.log(1.0 + (log_res ** 2) / NU_STUDENT)
    raise ValueError(kind)


def reg_m1(x):
    return 0.5 * (x[0] / 2.0) ** 2 + 0.5 * (x[1] / 1.5) ** 2


def reg_m2(x):
    return (
        0.5 * (x[0] / 2.0) ** 2 +
        0.5 * (x[1] / 1.5) ** 2 +
        0.5 * ((x[2] - math.log(8.0)) / 1.0) ** 2 +
        0.5 * (x[3] / 0.8) ** 2
    )


def sample_prediction_m1(sample_df, grid_df, fit_pars):
    rows = []
    for _, row in sample_df.iterrows():
        grow = grid_df.loc[grid_df["obs_index"] == int(row["obs_index"])].iloc[0]
        z = fit_pars["z_before"] if row["stage"] == "before" else fit_pars["z_after"]
        tc = interp_on_z(grow["tc_grid"], z)
        vp = interp_on_z(grow["vp_grid"], z)
        vs = interp_on_z(grow["vs_grid"], z)
        Ke = interp_on_z(grow["K_grid"], z)
        Ge = interp_on_z(grow["G_grid"], z)
        rows.append((tc, vp, vs, Ke, Ge))
    return rows


def sample_prediction_m2(sample_df, grid_df, fit_pars):
    rows = []
    for _, row in sample_df.iterrows():
        grow = grid_df.loc[grid_df["obs_index"] == int(row["obs_index"])].iloc[0]
        if row["stage"] == "before":
            w = beta_weights(fit_pars["m_before"], fit_pars["kappa_before"])
        else:
            w = beta_weights(fit_pars["m_after"], fit_pars["kappa_after"])
        tc = float(np.sum(w * grow["tc_grid"]))
        vp = float(np.sum(w * grow["vp_grid"]))
        vs = float(np.sum(w * grow["vs_grid"]))
        Ke = float(np.sum(w * grow["K_grid"]))
        Ge = float(np.sum(w * grow["G_grid"]))
        rows.append((tc, vp, vs, Ke, Ge))
    return rows


def loss_m1(x, sample_df, grid_df, unc_df):
    p = unpack_m1_params(x)
    tc_sigma = relative_uncertainty(unc_df, "tc_w_mk", 0.025)
    vp_sigma = relative_uncertainty(unc_df, "vp_m_s", 0.05)
    vs_sigma = relative_uncertainty(unc_df, "vs_m_s", 0.05)
    preds = sample_prediction_m1(sample_df, grid_df, p)
    tc_terms, vp_terms, vs_terms = [], [], []
    for (_, row), (tc_pred, vp_pred, vs_pred, _, _) in zip(sample_df.iterrows(), preds):
        tc_terms.append(objective_term(float(row["tc_w_mk"]), tc_pred, tc_sigma, OBJECTIVE))
        vp_terms.append(objective_term(float(row["vp_m_s"]), vp_pred, vp_sigma, OBJECTIVE))
        vs_terms.append(objective_term(float(row["vs_m_s"]), vs_pred, vs_sigma, OBJECTIVE))
    return float(np.mean(tc_terms) + np.mean(vp_terms) + np.mean(vs_terms) + reg_m1(x))


def loss_m2(x, sample_df, grid_df, unc_df):
    p = unpack_m2_params(x)
    tc_sigma = relative_uncertainty(unc_df, "tc_w_mk", 0.025)
    vp_sigma = relative_uncertainty(unc_df, "vp_m_s", 0.05)
    vs_sigma = relative_uncertainty(unc_df, "vs_m_s", 0.05)
    preds = sample_prediction_m2(sample_df, grid_df, p)
    tc_terms, vp_terms, vs_terms = [], [], []
    for (_, row), (tc_pred, vp_pred, vs_pred, _, _) in zip(sample_df.iterrows(), preds):
        tc_terms.append(objective_term(float(row["tc_w_mk"]), tc_pred, tc_sigma, OBJECTIVE))
        vp_terms.append(objective_term(float(row["vp_m_s"]), vp_pred, vp_sigma, OBJECTIVE))
        vs_terms.append(objective_term(float(row["vs_m_s"]), vs_pred, vs_sigma, OBJECTIVE))
    return float(np.mean(tc_terms) + np.mean(vp_terms) + np.mean(vs_terms) + reg_m2(x))


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def warm_start_m1(prev_row):
    q_before = float(prev_row["q_before"])
    q_after = float(prev_row["q_after"])
    x1 = logit(q_before)
    x2 = logit(q_after) - x1
    return np.array([x1, x2], dtype=float)


def warm_start_m2(prev_row):
    m_before = float(prev_row["m_before"])
    m_after = float(prev_row["m_after"])
    kappa_before = float(prev_row["kappa_before"])
    kappa_after = float(prev_row["kappa_after"])
    x1 = logit(m_before)
    x2 = logit(m_after) - x1
    x3 = math.log(max(kappa_before - 2.0, 1e-6))
    x4 = math.log(max(kappa_after / kappa_before, 1e-6))
    return np.array([x1, x2, x3, x4], dtype=float)


def optimize(fun, starts):
    best = None
    for x0 in starts:
        res = minimize(fun, x0=np.asarray(x0, dtype=float), method="Powell",
                       options={"maxiter": 35, "xtol": 1e-3, "ftol": 1e-3, "disp": False})
        if best is None or res.fun < best.fun:
            best = res
    return best


def main():
    measurements, constants, uncertainty, prev_m1, prev_m2 = load_inputs()
    joint_df = build_dataset(measurements)
    joint_df = joint_df.reset_index(drop=True)
    joint_df["obs_index"] = joint_df.index.astype(int)

    grid_df = precompute_grids(joint_df, constants)

    prev_m1["sample_id"] = prev_m1["sample_id"].astype(str)
    prev_m2["sample_id"] = prev_m2["sample_id"].astype(str)

    m1_rows, m2_rows, pred_frames = [], [], []

    for sample_id, g in joint_df.groupby("sample_id"):
        g = g.sort_values(["stage", "fluid_state"]).reset_index(drop=True)
        prev_row_m1 = prev_m1.loc[prev_m1["sample_id"] == sample_id].iloc[0]
        prev_row_m2 = prev_m2.loc[prev_m2["sample_id"] == sample_id].iloc[0]

        res1 = optimize(lambda x: loss_m1(x, g, grid_df, uncertainty),
                        [warm_start_m1(prev_row_m1), np.array([0.0, 0.0])])
        fit1 = {"sample_id": sample_id, "objective": OBJECTIVE, "success": bool(res1.success),
                 "status": int(res1.status), "objective_value": float(res1.fun),
                 "n_obs_per_property": int(len(g))}
        fit1.update(unpack_m1_params(res1.x))
        m1_rows.append(fit1)

        res2 = optimize(lambda x: loss_m2(x, g, grid_df, uncertainty),
                        [warm_start_m2(prev_row_m2), np.array([0.0, 0.0, math.log(8.0), 0.0])])
        fit2 = {"sample_id": sample_id, "objective": OBJECTIVE, "success": bool(res2.success),
                 "status": int(res2.status), "objective_value": float(res2.fun),
                 "n_obs_per_property": int(len(g))}
        fit2.update(unpack_m2_params(res2.x))
        m2_rows.append(fit2)

        preds1 = sample_prediction_m1(g, grid_df, fit1)
        preds2 = sample_prediction_m2(g, grid_df, fit2)

        for (_, row), vals in zip(g.iterrows(), preds1):
            tc_pred, vp_pred, vs_pred, Ke, Ge = vals
            for prop, obs, pred in [("tc_w_mk", row["tc_w_mk"], tc_pred), ("vp_m_s", row["vp_m_s"], vp_pred), ("vs_m_s", row["vs_m_s"], vs_pred)]:
                pred_frames.append({
                    "sample_id": sample_id, "stage": row["stage"], "fluid_state": row["fluid_state"],
                    "model": "M1", "property": prop, "objective": OBJECTIVE,
                    "obs": float(obs), "pred": float(pred), "residual": float(pred - obs),
                    "abs_rel_error": abs(float(pred - obs)) / float(obs),
                    "log_error": math.log(float(pred)) - math.log(float(obs)),
                    "K_eff_gpa": Ke, "G_eff_gpa": Ge
                })

        for (_, row), vals in zip(g.iterrows(), preds2):
            tc_pred, vp_pred, vs_pred, Ke, Ge = vals
            for prop, obs, pred in [("tc_w_mk", row["tc_w_mk"], tc_pred), ("vp_m_s", row["vp_m_s"], vp_pred), ("vs_m_s", row["vs_m_s"], vs_pred)]:
                pred_frames.append({
                    "sample_id": sample_id, "stage": row["stage"], "fluid_state": row["fluid_state"],
                    "model": "M2", "property": prop, "objective": OBJECTIVE,
                    "obs": float(obs), "pred": float(pred), "residual": float(pred - obs),
                    "abs_rel_error": abs(float(pred - obs)) / float(obs),
                    "log_error": math.log(float(pred)) - math.log(float(obs)),
                    "K_eff_gpa": Ke, "G_eff_gpa": Ge
                })

    m1_df = pd.DataFrame(m1_rows).sort_values("sample_id")
    m2_df = pd.DataFrame(m2_rows).sort_values("sample_id")
    pred_df = pd.DataFrame(pred_frames)

    overall_metrics = (
        pred_df.groupby(["model", "property"])
        .agg(
            n_obs=("obs", "size"),
            rmse=("residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("residual", lambda x: float(np.mean(np.abs(x)))),
            mare=("abs_rel_error", "mean"),
            rmse_log=("log_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )

    sample_metrics = (
        pred_df.groupby(["sample_id", "model", "property"])
        .agg(
            mare=("abs_rel_error", "mean"),
            rmse_log=("log_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )

    assumptions = pd.DataFrame([
        ("objective", OBJECTIVE),
        ("elastic_module", "strict_mt_elastic_pores.py"),
        ("thermal_block", "scalar MT thermal block with same aspect ratio parameter"),
        ("M2_quadrature_points", N_GRID),
        ("acceleration", "Precomputed AR-grid predictions per observation; optimization uses interpolation/weighted sums."),
    ], columns=["parameter", "value"])

    with pd.ExcelWriter(RESULT_XLSX, engine="openpyxl") as writer:
        joint_df.to_excel(writer, sheet_name="joint_dataset", index=False)
        grid_df.to_excel(writer, sheet_name="precomputed_grids", index=False)
        m1_df.to_excel(writer, sheet_name="M1_fits", index=False)
        m2_df.to_excel(writer, sheet_name="M2_fits", index=False)
        pred_df.to_excel(writer, sheet_name="predictions", index=False)
        overall_metrics.to_excel(writer, sheet_name="overall_metrics", index=False)
        sample_metrics.to_excel(writer, sheet_name="sample_metrics", index=False)
        assumptions.to_excel(writer, sheet_name="assumptions", index=False)

    print(f"Saved results to: {RESULT_XLSX}")


if __name__ == "__main__":
    main()
