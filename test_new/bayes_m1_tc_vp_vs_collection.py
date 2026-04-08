
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd()

# make matplotlib cache writable before importing pyplot
_MPL = _THIS_DIR / ".mplconfig_bayes_m1"
_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))

import matplotlib.pyplot as plt  # noqa: E402

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# strict MT elastic backend lives in `test_new/mori-tanaka/`
_BACKEND_DIR = _THIS_DIR / "mori-tanaka"
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from strict_mt_elastic_pores import ElasticPhase, strict_mt_elastic_random_spheroidal_pores  # noqa: E402


DATA_XLSX = _THIS_DIR / "data_restructured_for_MT_v2.xlsx"
OUT_XLSX = _THIS_DIR / "bayes_m1_tc_vp_vs_collection_results.xlsx"
SUMMARY_MD = _THIS_DIR / "bayes_m1_tc_vp_vs_collection_summary.md"
PLOT_DIR = _THIS_DIR / "bayes_m1_tc_vp_vs_collection_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_DIR = PLOT_DIR / "collection_exports"
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

N_Z = 121
Z_GRID = np.linspace(-4.0, 0.0, N_Z)
AR_GRID = 10.0 ** Z_GRID

SIGMA_DELTA_PRIOR = 1.0
PRACTICAL_DELTA = 0.10
N_DRAWS = 12000
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Draft nuisance priors (v1): marginalized per-sample (not pooled across sample)
# User provided: lambda_M range 2.80–2.92 W/(m·K)
# For K_M, G_M: derive reference from Vp/Vs and density, and use lognormal widths.
# ---------------------------------------------------------------------------

LAMBDA_M_MIN = 2.80
LAMBDA_M_MAX = 2.92

VP_REF_M_S = 6800.0
VS_REF_M_S = 3800.0
RHO_REF_KG_M3 = 2725.0  # mid of 2.72–2.73 g/cm3

G_REF_PA = RHO_REF_KG_M3 * VS_REF_M_S**2
K_REF_PA = RHO_REF_KG_M3 * VP_REF_M_S**2 - 4.0 * G_REF_PA / 3.0

# Vs, Vp ±5% => modulus uncertainty ~10% (since ~V^2). Use slightly wider K.
KM_REL_SIGMA = 0.12
GM_REL_SIGMA = 0.10


def delta_prior_logpdf(
    dz: np.ndarray,
    *,
    mode: str,
    sigma: float,
    box_min: float,
    box_max: float,
) -> np.ndarray:
    """
    Prior on delta_z = z_after - z_before evaluated on a grid.

    mode:
      - "gaussian": Normal(0, sigma^2)
      - "none": uniform (returns zeros)
      - "box": uniform over [box_min, box_max] (returns 0 in-range, -inf outside)
    """
    mode = str(mode).lower().strip()
    if mode == "none":
        return np.zeros_like(dz, dtype=float)
    if mode == "box":
        lo = float(min(box_min, box_max))
        hi = float(max(box_min, box_max))
        lp = np.zeros_like(dz, dtype=float)
        lp[(dz < lo) | (dz > hi)] = -np.inf
        return lp
    if mode == "gaussian":
        sig = float(sigma)
        if sig <= 0:
            raise ValueError("sigma must be positive for gaussian delta prior.")
        return -0.5 * (dz / sig) ** 2 - math.log(sig)
    raise ValueError(f"Unknown delta prior mode: {mode}")


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


def depolarization_factors_spheroid(aspect_ratio: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(aspect_ratio, dtype=float)
    n3 = np.empty_like(r)
    sphere = np.isclose(r, 1.0)
    oblate = r < 1.0
    prolate = r > 1.0
    n3[sphere] = 1.0 / 3.0

    rr = r[oblate]
    if rr.size > 0:
        xi = np.sqrt(np.maximum(1.0 / (rr * rr) - 1.0, 0.0))
        n3[oblate] = ((1.0 + xi * xi) / (xi**3)) * (xi - np.arctan(xi))

    rr = r[prolate]
    if rr.size > 0:
        e = np.sqrt(np.maximum(1.0 - 1.0 / (rr * rr), 0.0))
        n3[prolate] = ((1.0 - e * e) / (2.0 * e**3)) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)

    n1 = (1.0 - n3) / 2.0
    return n1, n3


N1_AR_GRID, N3_AR_GRID = depolarization_factors_spheroid(AR_GRID)


def mt_tc_single(phi: float, a_ratio: float, lambda_matrix: float, lambda_fluid: float) -> float:
    n1, n3 = depolarization_factors_spheroid(np.array([a_ratio], dtype=float))
    n1 = float(n1[0]); n3 = float(n3[0])
    dm = float(lambda_matrix); df = float(lambda_fluid)
    delta = df - dm
    a1 = dm / (dm + n1 * delta)
    a3 = dm / (dm + n3 * delta)
    a_bar = (2.0 * a1 + a3) / 3.0
    c_i = float(phi); c_m = 1.0 - c_i
    return float(dm + c_i * delta * a_bar / (c_m + c_i * a_bar))


def mt_tc_grid(phi: float, lambda_matrix: float, lambda_fluid: float) -> np.ndarray:
    """
    Vectorized MT transport prediction over the global AR_GRID using precomputed
    depolarization factors.
    """
    dm = float(lambda_matrix)
    df = float(lambda_fluid)
    delta = df - dm
    a1 = dm / (dm + N1_AR_GRID * delta)
    a3 = dm / (dm + N3_AR_GRID * delta)
    a_bar = (2.0 * a1 + a3) / 3.0
    c_i = float(phi)
    c_m = 1.0 - c_i
    return dm + c_i * delta * a_bar / (c_m + c_i * a_bar)


def load_all():
    measurements = pd.read_excel(DATA_XLSX, sheet_name="measurements_long")
    constants = pd.read_excel(DATA_XLSX, sheet_name="sample_constants")
    uncertainty = pd.read_excel(DATA_XLSX, sheet_name="uncertainty")

    if "phi_frac" in measurements.columns and "phi_pct" in measurements.columns:
        measurements["phi_frac"] = measurements["phi_frac"].where(
            measurements["phi_frac"].notna(), measurements["phi_pct"] / 100.0
        )
    for col in ["stage", "fluid_state"]:
        measurements[col] = measurements[col].astype(str).str.strip()

    req = ["lab_sample_id", "stage", "fluid_state", "phi_frac", "tc_w_mk", "vp_m_s", "vs_m_s", "bulk_density_g_cm3"]
    df = measurements.dropna(subset=req).copy()
    df["sample_id"] = df["lab_sample_id"].astype(str)
    df["phi_frac"] = pd.to_numeric(df["phi_frac"], errors="coerce")
    df["tc_w_mk"] = pd.to_numeric(df["tc_w_mk"], errors="coerce")
    df["vp_m_s"] = pd.to_numeric(df["vp_m_s"], errors="coerce")
    df["vs_m_s"] = pd.to_numeric(df["vs_m_s"], errors="coerce")
    df["rho_bulk_kg_m3"] = pd.to_numeric(df["bulk_density_g_cm3"], errors="coerce") * 1000.0
    keep = ["sample_id", "lab_sample_id", "stage", "fluid_state", "phi_frac", "phi_pct", "rho_bulk_kg_m3",
            "tc_w_mk", "vp_m_s", "vs_m_s", "field", "well", "depth_m", "tg_position_mm"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["sample_id", "stage", "fluid_state"]).reset_index(drop=True)
    df["obs_index"] = np.arange(len(df), dtype=int)
    return df, constants, uncertainty


def sample_positions_mm(joint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean TG position per sample (mm), taken from measurement table if available.
    """
    if "tg_position_mm" not in joint_df.columns:
        return pd.DataFrame({"sample_id": sorted(joint_df["sample_id"].astype(str).unique().tolist())}).assign(tg_position_mm=np.nan)
    g = joint_df.copy()
    g["sample_id"] = g["sample_id"].astype(str)
    g["tg_position_mm"] = pd.to_numeric(g["tg_position_mm"], errors="coerce")
    out = g.groupby("sample_id", as_index=False)["tg_position_mm"].mean()
    return out


class Zone:
    def __init__(self, zone_name: str, temperature_c: float, start_mm: float, end_mm: float):
        self.zone_name = str(zone_name)
        self.temperature_c = float(temperature_c)
        self.start_mm = float(start_mm)
        self.end_mm = float(end_mm)


DEFAULT_ZONE8 = Zone(zone_name="Зона 8", temperature_c=306.705, start_mm=612.0, end_mm=688.0)


def _normalize_sample_id_str(x: object) -> str:
    try:
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf).rstrip("0").rstrip(".")
    except Exception:
        return str(x).replace(",", ".").strip()


def load_zones_from_metadata(path: Path) -> list[Zone]:
    md = pd.read_excel(path, sheet_name="experiment_metadata")
    md = md[md["section"].astype(str).str.strip() == "heating_zone"].copy()
    md["parameter"] = md["parameter"].astype(str).str.strip()

    zones: list[Zone] = []
    current: dict[str, float | str | None] = {}
    for _, r in md.iterrows():
        param = str(r["parameter"])
        value = r["value"]
        if param == "zone_name":
            if current:
                zones.append(
                    Zone(
                        zone_name=str(current["zone_name"]),
                        temperature_c=float(current.get("temperature_c", np.nan)),
                        start_mm=float(current.get("start_mm", 0.0)),
                        end_mm=float(current.get("end_mm", np.nan)),
                    )
                )
            current = {
                "zone_name": str(value),
                "start_mm": float(zones[-1].end_mm) if zones else 0.0,  # type: ignore[attr-defined]
                "end_mm": None,
            }
        elif param == "temperature":
            current["temperature_c"] = float(value)
        elif param == "distance":
            current["end_mm"] = float(value)

    if current:
        zones.append(
            Zone(
                zone_name=str(current["zone_name"]),
                temperature_c=float(current.get("temperature_c", np.nan)),
                start_mm=float(current.get("start_mm", 0.0)),
                end_mm=float(current.get("end_mm", np.nan)),
            )
        )

    zones = sorted(zones, key=lambda z: z.end_mm)
    if zones and all(z.zone_name != "Зона 8" for z in zones):
        last = zones[-1]
        if last.zone_name == "Зона 7" and last.end_mm <= DEFAULT_ZONE8.start_mm + 1e-6:
            zones.append(DEFAULT_ZONE8)
    return zones


def assign_zone(position_mm: float, zones: list[Zone]) -> Zone | None:
    if not np.isfinite(position_mm) or not zones:
        return None
    if position_mm < zones[0].start_mm:
        return zones[0]
    if position_mm > zones[-1].end_mm:
        return zones[-1]
    for z in zones:
        if z.start_mm <= position_mm <= z.end_mm:
            return z
    return None


def load_sample_positions_from_raw(path: Path, sheet_name: str = "raw_original") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    sample_col_candidates = ["Лаб. номер образца", "lab_sample_id", "sample_id"]
    pos_col_candidates = ["Расположение в ТГ, мм", "tg_position_mm"]
    sample_col = next((c for c in sample_col_candidates if c in df.columns), None)
    pos_col = next((c for c in pos_col_candidates if c in df.columns), None)
    if sample_col is None or pos_col is None:
        raise ValueError(f"Cannot find sample_id/position columns in sheet '{sheet_name}'.")
    out = df[[sample_col, pos_col]].copy()
    out = out.rename(columns={sample_col: "sample_id", pos_col: "tg_position_mm"})
    out["sample_id"] = out["sample_id"].map(_normalize_sample_id_str)
    out["tg_position_mm"] = pd.to_numeric(out["tg_position_mm"], errors="coerce")
    out = out.dropna(subset=["sample_id"]).copy()
    # raw_original contains note rows and non-sample numeric entries (e.g. 0.26, 0.32).
    # Keep only plausible numeric sample ids (>= 1.0).
    s = out["sample_id"].astype(str).str.replace(",", ".", regex=False).str.strip()
    mask_num = s.str.match(r"^\d+(\.\d+)?$")
    out = out.loc[mask_num].copy()
    out["sample_id"] = s.loc[mask_num]
    out["sample_id_num"] = pd.to_numeric(out["sample_id"], errors="coerce")
    out = out.loc[out["sample_id_num"].notna() & (out["sample_id_num"] >= 1.0)].copy()
    out = (
        out.sort_values(["sample_id"])
        .groupby("sample_id", as_index=False)
        .agg(tg_position_mm=("tg_position_mm", "first"))
    )
    return out


def build_property_inputs(
    row: pd.Series,
    constants_df: pd.DataFrame,
    *,
    lambda_matrix: float | None = None,
    Km: float | None = None,
    Gm: float | None = None,
):
    lambda_matrix = float(lambda_matrix) if lambda_matrix is not None else scalar_constant(constants_df, "lambda_matrix_w_mk", 2.86)
    is_dry = str(row["fluid_state"]).lower() == "dry"
    lambda_fluid = scalar_constant(constants_df, "lambda_gas_w_mk", 0.025) if is_dry else scalar_constant(constants_df, "lambda_water_w_mk", 0.60)
    Km = float(Km) if Km is not None else scalar_constant(constants_df, "k_matrix_gpa", 76.8) * 1e9
    Gm = float(Gm) if Gm is not None else scalar_constant(constants_df, "g_matrix_gpa", 32.0) * 1e9
    Kf = scalar_constant(constants_df, "k_gas_gpa", 0.00014) * 1e9 if is_dry else scalar_constant(constants_df, "k_water_gpa", 2.2) * 1e9
    Gf = 1e-9
    rho_m = scalar_constant(constants_df, "rho_matrix_kg_m3", 2710.0)
    rho_f = 1.2 if is_dry else 1000.0
    return lambda_matrix, lambda_fluid, Km, Gm, Kf, Gf, rho_m, rho_f


def precompute_prediction_grids(df: pd.DataFrame, constants_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    cache = {}
    for _, row in df.iterrows():
        lambda_matrix, lambda_fluid, Km, Gm, Kf, Gf, rho_m, rho_f = build_property_inputs(row, constants_df)
        tc_vals, vp_vals, vs_vals = [], [], []
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
        cache[int(row["obs_index"])] = {
            "tc_grid": np.array(tc_vals, float),
            "vp_grid": np.array(vp_vals, float),
            "vs_grid": np.array(vs_vals, float),
        }
    return cache


def gaussian_loglike_logspace(obs: float, pred_grid: np.ndarray, sigma_rel: float) -> np.ndarray:
    sigma_log = math.log1p(float(sigma_rel))
    return -0.5 * ((math.log(float(obs)) - np.log(pred_grid)) / sigma_log) ** 2 - math.log(sigma_log)


def build_sample_stage_loglikes(sample_df: pd.DataFrame, grid_cache: dict[int, dict[str, np.ndarray]], uncertainty_df: pd.DataFrame):
    tc_sigma = relative_uncertainty(uncertainty_df, "tc_w_mk", 0.025)
    vp_sigma = relative_uncertainty(uncertainty_df, "vp_m_s", 0.05)
    vs_sigma = relative_uncertainty(uncertainty_df, "vs_m_s", 0.05)

    ll_before = np.zeros_like(Z_GRID)
    ll_after = np.zeros_like(Z_GRID)

    for _, row in sample_df.iterrows():
        g = grid_cache[int(row["obs_index"])]
        ll = (
            gaussian_loglike_logspace(row["tc_w_mk"], g["tc_grid"], tc_sigma)
            + gaussian_loglike_logspace(row["vp_m_s"], g["vp_grid"], vp_sigma)
            + gaussian_loglike_logspace(row["vs_m_s"], g["vs_grid"], vs_sigma)
        )
        if row["stage"] == "before":
            ll_before += ll
        else:
            ll_after += ll

    return ll_before, ll_after


def posterior_grid(ll_before: np.ndarray, ll_after: np.ndarray):
    dz = Z_GRID[None, :] - Z_GRID[:, None]
    log_post = ll_before[:, None] + ll_after[None, :] + delta_prior_logpdf(
        dz,
        mode="gaussian",
        sigma=SIGMA_DELTA_PRIOR,
        box_min=-np.inf,
        box_max=np.inf,
    )
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.sum(post)
    return post


def _lognormal_from_median(median: float, rel_sigma: float, size: int) -> np.ndarray:
    sigma_log = math.log1p(float(rel_sigma))
    mu_log = math.log(float(median))
    return np.exp(RNG.normal(mu_log, sigma_log, size=size))


def _sample_nuisance(n: int) -> dict[str, np.ndarray]:
    """
    Draw nuisance parameters from priors:
      - lambda_M ~ Uniform([2.80, 2.92])
      - Km, Gm ~ LogNormal around (K_REF, G_REF) with rel widths
    """
    # Use stratified sampling (lower MC noise than plain RNG) when possible.
    u = (np.arange(n, dtype=float) + 0.5) / float(n)
    u1 = RNG.permutation(u)
    u2 = RNG.permutation(u)
    u3 = RNG.permutation(u)

    lam_m = LAMBDA_M_MIN + (LAMBDA_M_MAX - LAMBDA_M_MIN) * u1

    try:
        from scipy.stats import norm  # type: ignore[import-not-found]
        z2 = norm.ppf(u2)
        z3 = norm.ppf(u3)
        Km = np.exp(math.log(K_REF_PA) + math.log1p(KM_REL_SIGMA) * z2)
        Gm = np.exp(math.log(G_REF_PA) + math.log1p(GM_REL_SIGMA) * z3)
    except Exception:
        Km = _lognormal_from_median(K_REF_PA, KM_REL_SIGMA, size=n)
        Gm = _lognormal_from_median(G_REF_PA, GM_REL_SIGMA, size=n)

    return {"lambda_m": lam_m, "Km": Km, "Gm": Gm}


def posterior_grid_marginalized_nuisance(
    sample_df: pd.DataFrame,
    constants_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    n_nuisance: int,
    *,
    delta_prior_mode: str,
    sigma_delta: float,
    delta_box_min: float,
    delta_box_max: float,
) -> np.ndarray:
    """
    Monte-Carlo marginalization over nuisance parameters per-sample:

      p(zb, za | data) ∝ E_theta [ p(data | zb, za, theta) ] * p(za-zb)

    where theta = (lambda_M, Km, Gm) drawn from their priors.
    """
    tc_sigma = relative_uncertainty(uncertainty_df, "tc_w_mk", 0.025)
    vp_sigma = relative_uncertainty(uncertainty_df, "vp_m_s", 0.05)
    vs_sigma = relative_uncertainty(uncertainty_df, "vs_m_s", 0.05)

    theta = _sample_nuisance(int(n_nuisance))
    rows = [r for _, r in sample_df.iterrows()]

    ll_before_s = np.zeros((n_nuisance, N_Z), dtype=float)
    ll_after_s = np.zeros((n_nuisance, N_Z), dtype=float)

    for s in range(n_nuisance):
        lam_m = float(theta["lambda_m"][s])
        Km = float(theta["Km"][s])
        Gm = float(theta["Gm"][s])
        for row in rows:
            lambda_matrix, lambda_fluid, Km_s, Gm_s, Kf, Gf, rho_m, rho_f = build_property_inputs(
                row,
                constants_df,
                lambda_matrix=lam_m,
                Km=Km,
                Gm=Gm,
            )

            tc_pred = mt_tc_grid(float(row["phi_frac"]), lambda_matrix, lambda_fluid)

            matrix = ElasticPhase(K=Km_s, G=Gm_s)
            inclusion = ElasticPhase(K=Kf, G=Gf)
            vp_pred = np.empty(N_Z, dtype=float)
            vs_pred = np.empty(N_Z, dtype=float)
            for i, ar in enumerate(AR_GRID):
                elastic = strict_mt_elastic_random_spheroidal_pores(
                    phi=float(row["phi_frac"]),
                    a_ratio=float(ar),
                    matrix=matrix,
                    inclusion=inclusion,
                    rho_matrix_kg_m3=float(rho_m),
                    rho_inclusion_kg_m3=float(rho_f),
                )
                vp_pred[i] = elastic.vp_m_s
                vs_pred[i] = elastic.vs_m_s

            ll = (
                gaussian_loglike_logspace(float(row["tc_w_mk"]), tc_pred, tc_sigma)
                + gaussian_loglike_logspace(float(row["vp_m_s"]), vp_pred, vp_sigma)
                + gaussian_loglike_logspace(float(row["vs_m_s"]), vs_pred, vs_sigma)
            )
            if row["stage"] == "before":
                ll_before_s[s, :] += ll
            else:
                ll_after_s[s, :] += ll

    # log-mean-exp across nuisance draws on the joint grid
    log_like_mean = None
    for s in range(n_nuisance):
        L = ll_before_s[s, :, None] + ll_after_s[s, None, :]
        if log_like_mean is None:
            log_like_mean = L
        else:
            log_like_mean = np.logaddexp(log_like_mean, L)
    assert log_like_mean is not None
    log_like_mean = log_like_mean - math.log(float(n_nuisance))

    dz = Z_GRID[None, :] - Z_GRID[:, None]
    log_post = log_like_mean + delta_prior_logpdf(
        dz,
        mode=delta_prior_mode,
        sigma=sigma_delta,
        box_min=delta_box_min,
        box_max=delta_box_max,
    )
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.sum(post)
    return post


def smooth_gaussian_1d(y: np.ndarray, sigma_pts: float) -> np.ndarray:
    """
    Small Gaussian smoothing on an evenly spaced grid.
    sigma_pts is in *grid points* (e.g., 1.5–3.0).
    """
    y = np.asarray(y, dtype=float)
    if sigma_pts <= 0:
        return y
    half = int(max(3, math.ceil(4.0 * sigma_pts)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma_pts)) ** 2)
    k = k / np.sum(k)
    y_pad = np.pad(y, (half, half), mode="edge")
    return np.convolve(y_pad, k, mode="valid")


def _latex_escape(s: object) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def df_to_latex_booktabs(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    out_path: Path,
    float_fmt: str = "{:.3f}",
) -> None:
    cols = list(df.columns)
    align = []
    for c in cols:
        align.append("r" if pd.api.types.is_numeric_dtype(df[c]) else "l")
    lines: list[str] = []
    lines.append("\\begin{table}[ht!]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append("\\begin{tabular}{%s}" % ("".join(align)))
    lines.append("\\toprule")
    lines.append(" & ".join([f"\\textbf{{{_latex_escape(c)}}}" for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                vals.append(float_fmt.format(float(v)))
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_collection_tables(sample_summary: pd.DataFrame, decision: pd.DataFrame) -> None:
    # Save CSV (easy to reuse)
    sample_summary.to_csv(COLLECTION_DIR / "sample_summary.csv", index=False)
    decision.to_csv(COLLECTION_DIR / "decision.csv", index=False)

    # Compact thesis table (key columns)
    tab = sample_summary.copy()
    keep = [
        "sample_id",
        "delta_z_median",
        "delta_z_q2.5",
        "delta_z_q97.5",
        "p_delta_lt_0",
        "p_abs_delta_gt_practical",
    ]
    keep = [c for c in keep if c in tab.columns]
    tab = tab[keep].copy()
    tab = tab.sort_values("p_delta_lt_0", ascending=False)
    df_to_latex_booktabs(
        tab,
        caption="Per-sample posterior summary for $\\Delta z = z_{after}-z_{before}$.",
        label="tab:bayes_m1_delta_z_per_sample",
        out_path=COLLECTION_DIR / "sample_delta_z_table.tex",
        float_fmt="{:.3f}",
    )

    # Also export decision sheet as a tiny LaTeX table
    df_to_latex_booktabs(
        decision,
        caption="Bayesian inversion settings (collection run).",
        label="tab:bayes_m1_settings",
        out_path=COLLECTION_DIR / "decision_table.tex",
        float_fmt="{:.3f}",
    )


def export_positions_and_temperature(joint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Export sample positions and derived temperature (via metadata zones).
    """
    zones = load_zones_from_metadata(DATA_XLSX)
    pos = load_sample_positions_from_raw(DATA_XLSX, sheet_name="raw_original")
    valid_ids = {_normalize_sample_id_str(v) for v in joint_df["sample_id"].astype(str).tolist()}
    pos = pos[pos["sample_id"].astype(str).isin(valid_ids)].copy()

    zone_name: list[str] = []
    zone_temp: list[float] = []
    for mm in pos["tg_position_mm"].to_numpy(dtype=float):
        z = assign_zone(float(mm), zones)
        zone_name.append(z.zone_name if z else "unknown")
        zone_temp.append(float(z.temperature_c) if z else float("nan"))
    pos["zone_name"] = zone_name
    pos["temperature_c"] = zone_temp

    pos.to_csv(COLLECTION_DIR / "sample_positions_temperature.csv", index=False)
    return pos


def make_collection_delta_z_ci_plot(sample_summary: pd.DataFrame) -> Path:
    """
    Thesis-ready plot: median Δz with 95% CI across samples.
    """
    fig, ax = plt.subplots(figsize=(11.5, 5.4), constrained_layout=True)
    df = sample_summary.sort_values("delta_z_median").reset_index(drop=True)
    x = np.arange(len(df), dtype=int)
    y = df["delta_z_median"].to_numpy(dtype=float)
    ylo = df["delta_z_q2.5"].to_numpy(dtype=float)
    yhi = df["delta_z_q97.5"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=[y - ylo, yhi - y],
        fmt="o",
        color="#1f77b4",
        ecolor="0.35",
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        alpha=0.9,
    )
    ax.axhline(0.0, color="k", ls="--", lw=1.1, alpha=0.7)
    ax.axhline(-PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axhline(PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample_id"].astype(str).tolist(), rotation=75, ha="right")
    ax.set_ylabel(r"$\Delta z = z_{after}-z_{before}$ (median $\pm$ 95\% CI)")
    ax.set_title("Collection summary: posterior change in aspect ratio (log-scale)")
    ax.grid(True, alpha=0.25, axis="y")
    out = COLLECTION_DIR / "collection_delta_z_ci.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_collection_alpha_pct_change_plot(sample_summary: pd.DataFrame) -> Path:
    """
    Thesis-ready plot in an intuitive scale:
      percent_change = 100*(alpha_after/alpha_before - 1) = 100*(10**Δz - 1)

    Point color encodes P(Δz < 0 | data) so probabilities stay on the figure.
    """
    fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    df = sample_summary.copy()

    # Join positions + derive temperatures so x-axis follows tube layout.
    # If positions are missing, fall back to sorting by delta_z.
    pos_path = COLLECTION_DIR / "sample_positions_temperature.csv"
    if pos_path.exists():
        pos_df = pd.read_csv(pos_path)
    else:
        pos_df = pd.DataFrame(columns=["sample_id", "tg_position_mm", "temperature_c"])

    if "tg_position_mm" in pos_df.columns and pos_df["tg_position_mm"].notna().any():
        pos_df = pos_df.copy()
        pos_df["sample_id"] = pos_df["sample_id"].astype(str)
        df = df.merge(pos_df[["sample_id", "tg_position_mm", "temperature_c"]], on="sample_id", how="left")
        df = df.sort_values("tg_position_mm").reset_index(drop=True)
    else:
        df = df.sort_values("delta_z_median").reset_index(drop=True)
        df["tg_position_mm"] = np.nan
        df["temperature_c"] = np.nan

    dz_med = df["delta_z_median"].to_numpy(dtype=float)
    dz_lo = df["delta_z_q2.5"].to_numpy(dtype=float)
    dz_hi = df["delta_z_q97.5"].to_numpy(dtype=float)
    pneg = df["p_delta_lt_0"].to_numpy(dtype=float)

    pct = 100.0 * (10.0 ** dz_med - 1.0)
    pct_lo = 100.0 * (10.0 ** dz_lo - 1.0)
    pct_hi = 100.0 * (10.0 ** dz_hi - 1.0)

    x = np.arange(len(df), dtype=float)

    # Color by probability
    try:
        import matplotlib as mpl
    except Exception:  # pragma: no cover
        mpl = None
    cmap = plt.get_cmap("viridis")
    if mpl is not None:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        colors = cmap(norm(pneg))
    else:
        colors = "C0"

    ax.errorbar(
        x,
        pct,
        yerr=[pct - pct_lo, pct_hi - pct],
        fmt="none",
        ecolor="0.35",
        elinewidth=1.2,
        capsize=3,
        alpha=0.9,
        zorder=1,
    )
    sc = ax.scatter(
        x,
        pct,
        s=46,
        c=pneg,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
        zorder=2,
    )

    ax.axhline(0.0, color="k", ls="--", lw=1.1, alpha=0.7)
    # Practical-change threshold in percent scale
    pct_pr = 100.0 * (10.0 ** PRACTICAL_DELTA - 1.0)
    pct_mn = 100.0 * (10.0 ** (-PRACTICAL_DELTA) - 1.0)
    ax.axhline(pct_mn, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axhline(pct_pr, color="0.35", ls=":", lw=1.2, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["sample_id"].astype(str).tolist(), rotation=75, ha="right")
    ax.set_ylabel(r"$100\cdot(\alpha_{after}/\alpha_{before}-1)$, \%  (median $\pm$ 95\% CI)")
    ax.set_title("Collection summary: percent change in aspect ratio (posterior)")
    ax.grid(True, alpha=0.25, axis="y")

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label(r"$P(\Delta z<0\mid data)$")

    # Temperature profile on a secondary axis (if available)
    if np.isfinite(df["temperature_c"].to_numpy(dtype=float)).any():
        ax2 = ax.twinx()
        t = df["temperature_c"].to_numpy(dtype=float)
        ax2.plot(x, t, color="0.2", lw=2.0, marker="o", markersize=4.5, alpha=0.85)
        ax2.set_ylabel("Temperature (°C)")
        ax2.grid(False)

    out = COLLECTION_DIR / "collection_alpha_pct_change_ci.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _zone_label(zone_name: str) -> str:
    z = str(zone_name)
    if z.lower().startswith("зона"):
        # "Зона 3" -> "Zone 3"
        return "Zone " + z.split()[-1]
    return z


def make_zone_boxplot_alpha_pct_change(sample_summary: pd.DataFrame) -> Path:
    """
    Aggregate samples by heating zone and show distribution of posterior median percent change.
    Adds temperature profile on the right axis.
    """
    pos_path = COLLECTION_DIR / "sample_positions_temperature.csv"
    if not pos_path.exists():
        raise FileNotFoundError(f"Missing {pos_path}. Run the main script once to export positions.")
    pos_df = pd.read_csv(pos_path)
    pos_df["sample_id"] = pos_df["sample_id"].astype(str)

    df = sample_summary.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.merge(pos_df[["sample_id", "zone_name", "temperature_c", "tg_position_mm"]], on="sample_id", how="left")

    # percent change from Δz median/CI
    df["pct_median"] = 100.0 * (10.0 ** df["delta_z_median"].astype(float) - 1.0)
    df["pct_q2.5"] = 100.0 * (10.0 ** df["delta_z_q2.5"].astype(float) - 1.0)
    df["pct_q97.5"] = 100.0 * (10.0 ** df["delta_z_q97.5"].astype(float) - 1.0)

    # Order zones by their tube position (mean)
    zone_order = (
        df.dropna(subset=["zone_name"])
        .groupby("zone_name", as_index=False)["tg_position_mm"]
        .mean()
        .sort_values("tg_position_mm")["zone_name"]
        .tolist()
    )
    zone_order = [z for z in zone_order if str(z).lower() != "ign" and str(z).lower() != "unknown"]

    data = [df.loc[df["zone_name"] == z, "pct_median"].dropna().to_numpy(dtype=float) for z in zone_order]
    temps = [float(df.loc[df["zone_name"] == z, "temperature_c"].dropna().iloc[0]) if df.loc[df["zone_name"] == z, "temperature_c"].notna().any() else float("nan") for z in zone_order]

    # Match dissertation style (bigger fonts + clean layout)
    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(13.2, 6.4), constrained_layout=False)

    # Interpretation bands (match earlier dissertation style)
    ax.set_facecolor("#fff4e6")  # warm paper background
    ax.axhspan(-100.0, -20.0, color="#f8d7da", alpha=0.45, zorder=0)
    ax.axhspan(-20.0, -5.0, color="#fdebd0", alpha=0.45, zorder=0)
    ax.axhspan(-5.0, 5.0, color="#d4edda", alpha=0.45, zorder=0)

    bp = ax.boxplot(
        data,
        tick_labels=[_zone_label(z) for z in zone_order],
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    for b in bp["boxes"]:
        b.set_facecolor("#9bd0ff")  # similar to reference blue
        b.set_alpha(0.85)
        b.set_edgecolor("0.25")
    for elem in ["whiskers", "caps", "medians", "means"]:
        for l in bp[elem]:
            l.set_color("0.25")
            l.set_linewidth(1.2)

    # Overlay sample points (small black markers like the reference figure)
    x_positions = np.arange(1, len(zone_order) + 1, dtype=float)
    for i, z in enumerate(zone_order):
        g = df[df["zone_name"] == z].dropna(subset=["pct_median"])
        if g.empty:
            continue
        jitter = (RNG.random(len(g)) - 0.5) * 0.18
        ax.scatter(
            np.full(len(g), x_positions[i]) + jitter,
            g["pct_median"].to_numpy(dtype=float),
            s=24,
            alpha=0.75,
            color="0.15",
            edgecolors="none",
            zorder=3,
        )

    ax.axhline(0.0, color="k", ls="--", lw=1.1, alpha=0.7)
    ax.set_ylim(-60, 10)
    ax.set_ylabel(r"$100\cdot(\alpha_{after}/\alpha_{before}-1)$, \%")
    ax.set_title("Aspect-ratio change by heating zone (with temperature profile)")
    ax.grid(True, alpha=0.25, axis="y")

    # Temperature profile on right axis
    ax2 = ax.twinx()
    ax2.plot(x_positions, temps, color="0.2", lw=2.2, marker="o", markersize=5.0, alpha=0.85, label="Zone temperature")
    ax2.set_ylabel("Temperature (°C)")

    # Legends below plot (outside the axes, like the reference)
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    band_handles = [
        mpatches.Patch(color="#f8d7da", alpha=0.45, label="< −20%: strong crack-like shift"),
        mpatches.Patch(color="#fdebd0", alpha=0.45, label="−20%…−5%: moderate crack-like shift"),
        mpatches.Patch(color="#d4edda", alpha=0.45, label="−5%…+5%: ~no change"),
    ]
    box_handle = mpatches.Patch(color="#9bd0ff", alpha=0.85, label="Bayes M1: % change")
    temp_handle = mlines.Line2D([0], [0], color="0.2", lw=2.2, marker="o", label="Zone temperature profile")
    # Reserve space for legend under axes and place legend below plot.
    fig.subplots_adjust(bottom=0.23)
    fig.legend(
        handles=[box_handle, temp_handle, *band_handles],
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    out = COLLECTION_DIR / "zone_boxplot_alpha_pct_change.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def export_zone_membership_tables(sample_summary: pd.DataFrame) -> None:
    """
    Export counts and sample-id lists by heating zone for dissertation tables.
    """
    pos_path = COLLECTION_DIR / "sample_positions_temperature.csv"
    if not pos_path.exists():
        return
    pos_df = pd.read_csv(pos_path)
    pos_df["sample_id"] = pos_df["sample_id"].astype(str)

    df = sample_summary.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.merge(pos_df[["sample_id", "zone_name", "temperature_c", "tg_position_mm"]], on="sample_id", how="left")

    # Build table
    rows = []
    for zone, g in df.groupby("zone_name", dropna=False):
        if pd.isna(zone):
            zone = "unknown"
        g = g.sort_values("tg_position_mm")
        ids = g["sample_id"].astype(str).tolist()
        rows.append(
            {
                "zone": _zone_label(zone),
                "temperature_c": float(g["temperature_c"].dropna().iloc[0]) if g["temperature_c"].notna().any() else float("nan"),
                "n_samples": int(len(ids)),
                "sample_ids": ", ".join(ids),
            }
        )
    tab = pd.DataFrame(rows).sort_values(["temperature_c", "zone"])
    tab.to_csv(COLLECTION_DIR / "zone_membership.csv", index=False)
    df_to_latex_booktabs(
        tab,
        caption="Number of samples and IDs by heating zone.",
        label="tab:zone_membership",
        out_path=COLLECTION_DIR / "zone_membership.tex",
        float_fmt="{:.1f}",
    )

def sample_from_posterior_grid(post: np.ndarray, n_draws: int = N_DRAWS):
    flat = post.ravel()
    idx = RNG.choice(flat.size, size=n_draws, replace=True, p=flat)
    i_before, i_after = np.unravel_index(idx, post.shape)
    z_before = Z_GRID[i_before]
    z_after = Z_GRID[i_after]
    delta_z = z_after - z_before
    ao_before = 10.0 ** z_before
    ao_after = 10.0 ** z_after
    ratio = ao_after / ao_before
    return {
        "z_before": z_before,
        "z_after": z_after,
        "delta_z": delta_z,
        "ao_before": ao_before,
        "ao_after": ao_after,
        "ratio": ratio,
    }


def summarize_draws(draws: dict[str, np.ndarray], sample_id: str) -> pd.DataFrame:
    rows = []
    for name, arr in draws.items():
        q2, q50, q97 = np.quantile(arr, [0.025, 0.5, 0.975])
        rows.append({
            "sample_id": sample_id,
            "quantity": name,
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)),
            "q2.5": float(q2),
            "median": float(q50),
            "q97.5": float(q97),
        })
    return pd.DataFrame(rows)


def summarize_sample(sample_id: str, sample_df: pd.DataFrame, post: np.ndarray, draws: dict[str, np.ndarray]) -> dict[str, object]:
    obs_stage = sample_df.groupby("stage")[["tc_w_mk", "vp_m_s", "vs_m_s"]].mean()
    ib, ia = np.unravel_index(np.argmax(post), post.shape)
    return {
        "sample_id": sample_id,
        "phi_before": float(sample_df.loc[sample_df["stage"] == "before", "phi_frac"].mean()),
        "phi_after": float(sample_df.loc[sample_df["stage"] == "after", "phi_frac"].mean()),
        "obs_tc_after_minus_before": float(obs_stage.loc["after", "tc_w_mk"] - obs_stage.loc["before", "tc_w_mk"]),
        "obs_vp_after_minus_before": float(obs_stage.loc["after", "vp_m_s"] - obs_stage.loc["before", "vp_m_s"]),
        "obs_vs_after_minus_before": float(obs_stage.loc["after", "vs_m_s"] - obs_stage.loc["before", "vs_m_s"]),
        "z_before_map": float(Z_GRID[ib]),
        "z_after_map": float(Z_GRID[ia]),
        "ao_before_median": float(np.median(draws["ao_before"])),
        "ao_after_median": float(np.median(draws["ao_after"])),
        "ratio_median": float(np.median(draws["ratio"])),
        "delta_z_median": float(np.median(draws["delta_z"])),
        "delta_z_q2.5": float(np.quantile(draws["delta_z"], 0.025)),
        "delta_z_q97.5": float(np.quantile(draws["delta_z"], 0.975)),
        "p_delta_lt_0": float(np.mean(draws["delta_z"] < 0.0)),
        "p_delta_gt_0": float(np.mean(draws["delta_z"] > 0.0)),
        "p_abs_delta_gt_practical": float(np.mean(np.abs(draws["delta_z"]) > PRACTICAL_DELTA)),
    }


def classify_support(row):
    pneg = row["p_delta_lt_0"]
    pchange = row["p_abs_delta_gt_practical"]
    if pneg >= 0.95 and pchange >= 0.90:
        return "сильная поддержка crack-like shift"
    if pneg >= 0.80 and pchange >= 0.70:
        return "умеренная поддержка"
    if 0.35 <= pneg <= 0.65:
        return "неопределённый знак изменения"
    return "слабая поддержка"


def make_collection_plot(summary_df: pd.DataFrame):
    df = summary_df.sort_values("p_delta_lt_0", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.bar(df["sample_id"], df["p_delta_lt_0"])
    ax.axhline(0.95, linestyle="--", linewidth=1.2, label="0.95")
    ax.axhline(0.80, linestyle=":", linewidth=1.2, label="0.80")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("P(Δz < 0 | data)")
    ax.set_title("Posterior probability of crack-like shift by sample")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    out = PLOT_DIR / "collection_probability_crack_like_shift.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_sample_plot(sample_id: str, post: np.ndarray, draws: dict[str, np.ndarray]):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel 1: Joint posterior (contours) in (z_before, z_after)
    # ------------------------------------------------------------------
    ax = axes[0]
    X, Y = np.meshgrid(Z_GRID, Z_GRID)  # x=z_after, y=z_before

    # Contour levels: evenly spaced between small positive and max
    vmin = float(np.quantile(post[post > 0], 0.85)) if np.any(post > 0) else float(np.min(post))
    vmax = float(np.max(post))
    levels = np.linspace(vmin, vmax, 6)

    cs = ax.contour(X, Y, post, levels=levels, cmap="viridis", linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=10, fmt="%.3f")

    # Highlight the highest-density region with a bold yellow contour (top level)
    try:
        ax.contour(X, Y, post, levels=[levels[-2]], colors=["#ffd400"], linewidths=2.6)
    except Exception:
        pass

    ib, ia = np.unravel_index(np.argmax(post), post.shape)
    ax.plot([Z_GRID[ia]], [Z_GRID[ib]], marker="*", markersize=14, color="crimson", zorder=5)

    ax.set_title("Joint posterior")
    ax.set_xlabel(r"$\log_{10}\alpha_{after}$")
    ax.set_ylabel(r"$\log_{10}\alpha_{before}$")
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Panel 2: Marginal posterior of log10(alpha_before)
    # ------------------------------------------------------------------
    ax = axes[1]
    m_before = post.sum(axis=1)
    area = float(np.trapz(m_before, Z_GRID))
    if area > 0:
        m_before = m_before / area
    m_before = smooth_gaussian_1d(m_before, sigma_pts=2.0)
    ax.plot(Z_GRID, m_before, color="#1f77b4", lw=3.0)
    ax.fill_between(Z_GRID, 0.0, m_before, color="#1f77b4", alpha=0.12, linewidth=0)
    mean_before = float(np.mean(draws["z_before"]))
    ax.axvline(mean_before, color="red", ls="--", lw=1.4)
    q2, q97 = np.quantile(draws["z_before"], [0.025, 0.975])
    ax.axvline(float(q2), color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axvline(float(q97), color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.set_title(r"Marginal posterior of $\alpha_{before}$")
    ax.set_xlabel(r"$\log_{10}\alpha$")
    ax.set_ylabel("Posterior density (relative)")
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Panel 3: Marginal posterior of log10(alpha_after)
    # ------------------------------------------------------------------
    ax = axes[2]
    m_after = post.sum(axis=0)
    area = float(np.trapz(m_after, Z_GRID))
    if area > 0:
        m_after = m_after / area
    m_after = smooth_gaussian_1d(m_after, sigma_pts=2.0)
    ax.plot(Z_GRID, m_after, color="#1f77b4", lw=3.0)
    ax.fill_between(Z_GRID, 0.0, m_after, color="#1f77b4", alpha=0.12, linewidth=0)
    mean_after = float(np.mean(draws["z_after"]))
    ax.axvline(mean_after, color="red", ls="--", lw=1.4)
    q2, q97 = np.quantile(draws["z_after"], [0.025, 0.975])
    ax.axvline(float(q2), color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axvline(float(q97), color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.set_title(r"Marginal posterior of $\alpha_{after}$")
    ax.set_xlabel(r"$\log_{10}\alpha$")
    ax.set_ylabel("Posterior density (relative)")
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"Bayesian inversion (TC+Vp+Vs) — sample {sample_id}", y=1.03, fontsize=16)

    out = PLOT_DIR / f"sample_{sample_id}_posterior_styled.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize_delta_draws(draws: dict[str, np.ndarray]) -> dict[str, float]:
    dz = np.asarray(draws["delta_z"], dtype=float)
    q2, q50, q97 = np.quantile(dz, [0.025, 0.5, 0.975])
    return {
        "delta_z_mean": float(np.mean(dz)),
        "delta_z_median": float(q50),
        "delta_z_q2.5": float(q2),
        "delta_z_q97.5": float(q97),
        "p_delta_lt_0": float(np.mean(dz < 0.0)),
        "p_abs_delta_gt_practical": float(np.mean(np.abs(dz) > PRACTICAL_DELTA)),
    }


def make_prior_sensitivity_plot(
    sample_id: str,
    posts: dict[str, np.ndarray],
    draws_by_mode: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    *,
    dpi: int = 300,
) -> Path:
    """
    Compare how the delta prior changes the posterior for one sample.
    Produces a clean, thesis-ready overlay for Δz.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)

    # Panel 1: Δz marginal overlay (from posterior grid)
    ax = axes[0]
    colors = {"gaussian": "#1f77b4", "none": "#ff7f0e", "box": "#2ca02c"}
    labels = {"gaussian": "Gaussian prior", "none": "Uniform (no prior)", "box": "Box prior"}
    for mode, post in posts.items():
        m_dz = post.sum(axis=0)  # marginal over z_before -> distribution of z_after index, not dz
        # Better: compute Δz histogram from draws
        dz = np.asarray(draws_by_mode[mode]["delta_z"], dtype=float)
        bins = np.linspace(float(dz.min()), float(dz.max()), 60)
        hist, edges = np.histogram(dz, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist = smooth_gaussian_1d(hist, sigma_pts=1.2)
        ax.plot(centers, hist, lw=2.8, color=colors.get(mode, "0.3"), label=labels.get(mode, mode))
    ax.axvline(0.0, color="k", ls="--", lw=1.2, alpha=0.7)
    ax.axvline(-PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axvline(PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.set_title(r"Posterior of $\Delta z$ (sensitivity to prior)")
    ax.set_xlabel(r"$\Delta z = z_{after} - z_{before}$")
    ax.set_ylabel("Density (relative)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    # Panel 2: Joint posterior contours for the three modes (small multiples)
    ax = axes[1]
    X, Y = np.meshgrid(Z_GRID, Z_GRID)
    for mode, post in posts.items():
        vmax = float(np.max(post))
        vmin = float(np.quantile(post[post > 0], 0.85)) if np.any(post > 0) else float(np.min(post))
        levels = np.linspace(vmin, vmax, 5)
        ax.contour(X, Y, post, levels=levels, colors=[colors.get(mode, "0.3")], linewidths=1.6, alpha=0.9)
    ax.set_title("Joint posterior contours")
    ax.set_xlabel(r"$\log_{10}\alpha_{after}$")
    ax.set_ylabel(r"$\log_{10}\alpha_{before}$")
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"Prior sensitivity — sample {sample_id}", y=1.03, fontsize=15)
    out = out_dir / f"prior_sensitivity_sample_{sample_id}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def make_prior_sensitivity_collection_plot(comp: pd.DataFrame, out_dir: Path) -> Path:
    """
    Thesis-ready comparison across priors for many samples.
    Expects columns from prior_sensitivity_summary.csv.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    if comp.empty:
        raise ValueError("Empty prior sensitivity table.")

    df = comp.copy()
    df["prior"] = df["prior"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)

    order = (
        df[df["prior"] == "gaussian"]
        .sort_values("delta_z_median")["sample_id"]
        .tolist()
    )
    if not order:
        order = sorted(df["sample_id"].unique().tolist())

    priors = ["gaussian", "none", "box"]
    colors = {"gaussian": "#1f77b4", "none": "#ff7f0e", "box": "#2ca02c"}
    labels = {"gaussian": "Gaussian", "none": "Uniform", "box": "Box"}

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.5), sharex=True, constrained_layout=True)

    # Panel 1: delta_z median with 95% CI per prior (overlay)
    ax = axes[0]
    x = np.arange(len(order), dtype=float)
    width = 0.22
    offsets = {"gaussian": -width, "none": 0.0, "box": width}
    for prior in priors:
        g = df[df["prior"] == prior].set_index("sample_id").reindex(order)
        y = g["delta_z_median"].to_numpy(dtype=float)
        ylo = g["delta_z_q2.5"].to_numpy(dtype=float)
        yhi = g["delta_z_q97.5"].to_numpy(dtype=float)
        ax.errorbar(
            x + offsets[prior],
            y,
            yerr=[y - ylo, yhi - y],
            fmt="o",
            markersize=4.5,
            lw=1.0,
            capsize=2.5,
            color=colors[prior],
            ecolor=colors[prior],
            alpha=0.9,
            label=labels[prior],
        )
    ax.axhline(0.0, color="k", ls="--", lw=1.1, alpha=0.7)
    ax.axhline(-PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axhline(PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.set_ylabel(r"$\Delta z$ (median $\pm$ 95\% CI)")
    ax.set_title("Prior sensitivity across collection")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=True, ncol=3, loc="upper right")

    # Panel 2: probabilities
    ax = axes[1]
    for prior in priors:
        g = df[df["prior"] == prior].set_index("sample_id").reindex(order)
        y = g["p_delta_lt_0"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=4.0, lw=1.8, color=colors[prior], alpha=0.9, label=labels[prior])
    ax.axhline(0.95, color="0.35", ls="--", lw=1.1, alpha=0.8)
    ax.axhline(0.80, color="0.35", ls=":", lw=1.1, alpha=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(r"$P(\Delta z<0\mid data)$")
    ax.set_xlabel("Sample ID")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=75, ha="right")
    ax.grid(True, alpha=0.25, axis="y")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "prior_sensitivity_collection.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_prior_sensitivity_overview_plot(comp: pd.DataFrame, out_dir: Path) -> Path:
    """
    High-level "one page" overview across samples:
      - distributions of P(Δz<0)
      - distributions of Δz (median)
      - counts above common evidence thresholds
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    df = comp.copy()
    if df.empty:
        raise ValueError("Empty prior sensitivity table.")

    priors = ["gaussian", "none", "box"]
    colors = {"gaussian": "#1f77b4", "none": "#ff7f0e", "box": "#2ca02c"}
    labels = {"gaussian": "Gaussian", "none": "Uniform", "box": "Box"}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

    # Panel A: histogram of P(Δz<0)
    ax = axes[0, 0]
    bins = np.linspace(0.0, 1.0, 21)
    for p in priors:
        x = df.loc[df["prior"] == p, "p_delta_lt_0"].to_numpy(dtype=float)
        ax.hist(x, bins=bins, alpha=0.35, color=colors[p], label=labels[p], edgecolor="none")
    ax.axvline(0.95, color="0.35", ls="--", lw=1.1, alpha=0.8)
    ax.axvline(0.80, color="0.35", ls=":", lw=1.1, alpha=0.8)
    ax.set_xlabel(r"$P(\Delta z<0\mid data)$")
    ax.set_ylabel("Count")
    ax.set_title("Evidence distribution across samples")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=True)

    # Panel B: histogram of Δz median
    ax = axes[0, 1]
    all_vals = df["delta_z_median"].to_numpy(dtype=float)
    lo = float(np.nanmin(all_vals))
    hi = float(np.nanmax(all_vals))
    bins = np.linspace(lo, hi, 22) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 20
    for p in priors:
        x = df.loc[df["prior"] == p, "delta_z_median"].to_numpy(dtype=float)
        ax.hist(x, bins=bins, alpha=0.35, color=colors[p], label=labels[p], edgecolor="none")
    ax.axvline(0.0, color="k", ls="--", lw=1.1, alpha=0.7)
    ax.axvline(-PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.axvline(PRACTICAL_DELTA, color="0.35", ls=":", lw=1.2, alpha=0.9)
    ax.set_xlabel(r"$\Delta z$ median")
    ax.set_ylabel("Count")
    ax.set_title(r"Posterior shift magnitude ($\Delta z$)")
    ax.grid(True, alpha=0.25, axis="y")

    # Panel C: counts above thresholds
    ax = axes[1, 0]
    x = np.arange(len(priors), dtype=float)
    n95 = []
    n80 = []
    for p in priors:
        s = df[df["prior"] == p]["p_delta_lt_0"].to_numpy(dtype=float)
        n95.append(int(np.sum(s >= 0.95)))
        n80.append(int(np.sum(s >= 0.80)))
    ax.bar(x - 0.18, n95, width=0.36, color="#4c72b0", alpha=0.9, label="≥ 0.95")
    ax.bar(x + 0.18, n80, width=0.36, color="#55a868", alpha=0.9, label="≥ 0.80")
    ax.set_xticks(x, [labels[p] for p in priors])
    ax.set_ylabel("Number of samples")
    ax.set_title(r"Counts above $P(\Delta z<0)$ thresholds")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=True)

    # Panel D: practical-change probabilities
    ax = axes[1, 1]
    thresh = 0.90
    counts = []
    for p in priors:
        s = df[df["prior"] == p]["p_abs_delta_gt_practical"].to_numpy(dtype=float)
        counts.append(int(np.sum(s >= thresh)))
    ax.bar(x, counts, width=0.55, color=[colors[p] for p in priors], alpha=0.9)
    ax.set_xticks(x, [labels[p] for p in priors])
    ax.set_ylabel("Number of samples")
    ax.set_title(rf"Counts with $P(|\Delta z|>{PRACTICAL_DELTA}) \geq {thresh:.2f}$")
    ax.grid(True, alpha=0.25, axis="y")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "prior_sensitivity_overview.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Bayesian M1 per-sample inversion (TC+Vp+Vs).")
    ap.add_argument(
        "--nuisance-draws",
        type=int,
        default=700,
        help="If >0, marginalize (lambda_M, K_M, G_M) with this many prior draws per sample.",
    )
    ap.add_argument(
        "--demo-sample-ids",
        nargs="*",
        default=["22.1", "24.2"],
        help="Sample IDs to render styled posterior plots for.",
    )
    ap.add_argument(
        "--delta-prior",
        choices=["gaussian", "none", "box"],
        default="gaussian",
        help="Prior on Δz=z_after−z_before.",
    )
    ap.add_argument(
        "--sigma-delta",
        type=float,
        default=SIGMA_DELTA_PRIOR,
        help="Sigma for gaussian Δz prior (only used when --delta-prior gaussian).",
    )
    ap.add_argument(
        "--delta-min",
        type=float,
        default=-0.5,
        help="Min Δz for box prior (only used when --delta-prior box).",
    )
    ap.add_argument(
        "--delta-max",
        type=float,
        default=0.5,
        help="Max Δz for box prior (only used when --delta-prior box).",
    )
    ap.add_argument(
        "--prior-sensitivity-sample-ids",
        nargs="*",
        default=[],
        help="If provided, runs 3 priors (gaussian/none/box) only for these sample IDs and saves a comparison table+plot.",
    )
    args = ap.parse_args()

    df, constants, uncertainty = load_all()
    grid_cache = None if int(args.nuisance_draws) > 0 else precompute_prediction_grids(df, constants)

    # Prior-sensitivity mode: run only a small set of samples under multiple priors
    if args.prior_sensitivity_sample_ids:
        out_dir = PLOT_DIR / "prior_sensitivity"
        rows = []
        for sid in args.prior_sensitivity_sample_ids:
            sample_df = df[df["sample_id"].astype(str) == str(sid)].sort_values(["stage", "fluid_state"]).copy()
            if sample_df.empty:
                continue

            posts = {}
            draws_by_mode = {}
            for mode in ["gaussian", "none", "box"]:
                post = posterior_grid_marginalized_nuisance(
                    sample_df=sample_df,
                    constants_df=constants,
                    uncertainty_df=uncertainty,
                    n_nuisance=int(args.nuisance_draws),
                    delta_prior_mode=mode,
                    sigma_delta=float(args.sigma_delta),
                    delta_box_min=float(args.delta_min),
                    delta_box_max=float(args.delta_max),
                )
                draws = sample_from_posterior_grid(post, n_draws=N_DRAWS)
                posts[mode] = post
                draws_by_mode[mode] = draws
                s = summarize_delta_draws(draws)
                rows.append({"sample_id": str(sid), "prior": mode, **s})

            make_prior_sensitivity_plot(str(sid), posts, draws_by_mode, out_dir, dpi=300)

        out_dir.mkdir(parents=True, exist_ok=True)
        comp = pd.DataFrame(rows)
        comp.to_csv(out_dir / "prior_sensitivity_summary.csv", index=False)
        try:
            out_plot = make_prior_sensitivity_collection_plot(comp, out_dir)
            print(f"Saved: {out_plot}")
        except Exception:
            pass
        try:
            out_plot = make_prior_sensitivity_overview_plot(comp, out_dir)
            print(f"Saved: {out_plot}")
        except Exception:
            pass
        print(f"Saved: {out_dir / 'prior_sensitivity_summary.csv'}")
        return

    per_quantity_summaries = []
    sample_summary_rows = []
    posterior_maps = {}

    for sample_id, sample_df in df.groupby("sample_id"):
        sample_df = sample_df.sort_values(["stage", "fluid_state"]).copy()
        if int(args.nuisance_draws) > 0:
            post = posterior_grid_marginalized_nuisance(
                sample_df=sample_df,
                constants_df=constants,
                uncertainty_df=uncertainty,
                n_nuisance=int(args.nuisance_draws),
                delta_prior_mode=str(args.delta_prior),
                sigma_delta=float(args.sigma_delta),
                delta_box_min=float(args.delta_min),
                delta_box_max=float(args.delta_max),
            )
        else:
            assert grid_cache is not None
            ll_before, ll_after = build_sample_stage_loglikes(sample_df, grid_cache, uncertainty)
            # fixed-constants path keeps gaussian prior for Δz (legacy); for uniform/box, use nuisance mode
            post = posterior_grid(ll_before, ll_after)
        draws = sample_from_posterior_grid(post, n_draws=N_DRAWS)

        posterior_maps[str(sample_id)] = post
        per_quantity_summaries.append(summarize_draws(draws, str(sample_id)))
        sample_summary_rows.append(summarize_sample(str(sample_id), sample_df, post, draws))

    posterior_summary = pd.concat(per_quantity_summaries, ignore_index=True)
    sample_summary = pd.DataFrame(sample_summary_rows)
    sample_summary["comment"] = sample_summary.apply(classify_support, axis=1)

    decision = pd.DataFrame([
        {"item": "analysis_type", "value": "Independent grid-Bayes M1 per sample"},
        {"item": "data", "value": "TC + Vp + Vs"},
        {"item": "parameters", "value": "z_before, z_after (derived delta_z, AO_before, AO_after, ratio)"},
        {"item": "prior", "value": f"uniform box z∈[-4,0] + weak Gaussian prior on delta_z with sigma={SIGMA_DELTA_PRIOR}"},
        {"item": "likelihood", "value": "Gaussian in log-space using experimental relative uncertainties"},
        {"item": "practical_delta", "value": PRACTICAL_DELTA},
        {"item": "nuisance_mode", "value": f"MC-marginalized per-sample, draws={int(args.nuisance_draws)}" if int(args.nuisance_draws) > 0 else "fixed constants"},
        {"item": "lambda_M_prior", "value": f"Uniform({LAMBDA_M_MIN:.2f}, {LAMBDA_M_MAX:.2f}) W/(m·K)"},
        {"item": "K_M_prior", "value": f"LogNormal(median={K_REF_PA/1e9:.1f} GPa, rel_sigma={KM_REL_SIGMA:.2f})"},
        {"item": "G_M_prior", "value": f"LogNormal(median={G_REF_PA/1e9:.1f} GPa, rel_sigma={GM_REL_SIGMA:.2f})"},
    ])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="joint_dataset", index=False)
        posterior_summary.to_excel(writer, sheet_name="posterior_summary_by_quantity", index=False)
        sample_summary.to_excel(writer, sheet_name="sample_summary", index=False)
        decision.to_excel(writer, sheet_name="decision", index=False)

    collection_plot = make_collection_plot(sample_summary)
    export_collection_tables(sample_summary, decision)
    export_positions_and_temperature(df)
    export_zone_membership_tables(sample_summary)
    delta_ci_plot = make_collection_delta_z_ci_plot(sample_summary)
    pct_ci_plot = make_collection_alpha_pct_change_plot(sample_summary)
    zone_plot = make_zone_boxplot_alpha_pct_change(sample_summary)
    demo_plots = []
    for sid in list(args.demo_sample_ids):
        if sid in posterior_maps:
            post = posterior_maps[sid]
            draws = sample_from_posterior_grid(post, n_draws=N_DRAWS)
            demo_plots.append(make_sample_plot(sid, post, draws))

    lines = []
    lines.append("# Bayesian M1 (independent per-sample) — collection summary\n")
    lines.append(f"- Prior on delta_z: Normal(0, {SIGMA_DELTA_PRIOR}^2), truncated by z_before,z_after ∈ [-4,0]\n")
    lines.append(f"- Practical-change threshold: |Δz| > {PRACTICAL_DELTA}\n")
    strong = int((sample_summary["p_delta_lt_0"] >= 0.95).sum())
    moderate = int((sample_summary["p_delta_lt_0"] >= 0.80).sum())
    lines.append(f"- Samples with P(Δz < 0 | data) ≥ 0.95: **{strong}**\n")
    lines.append(f"- Samples with P(Δz < 0 | data) ≥ 0.80: **{moderate}**\n")
    SUMMARY_MD.write_text("".join(lines), encoding="utf-8")

    print(f"Saved workbook: {OUT_XLSX}")
    print(f"Saved summary: {SUMMARY_MD}")
    print(f"Saved plot: {collection_plot}")
    print(f"Saved plot: {delta_ci_plot}")
    print(f"Saved plot: {pct_ci_plot}")
    print(f"Saved plot: {zone_plot}")
    print(f"Saved exports: {COLLECTION_DIR}")
    for p in demo_plots:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
