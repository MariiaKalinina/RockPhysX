from __future__ import annotations

"""
Fifth-look anisotropic thermal-conductivity workflow for Bazhenov samples.

New physics:
1) Split OM into two morphologically distinct subphases:
   - lenticular OM: aligned, weakly porous, controls anisotropy;
   - patchy OM: inclusion-like, more porous, controls weakening via OM porosity.
2) After tube experiment:
   - delta_f_lens_after <= 0 (lenses can stay constant or decrease);
   - patchy OM porosity can increase;
   - shrinkage / crack-like pores appear in OM;
   - matrix-host microcracks appear.

This is still a screening model, not a final inversion.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_sheet_with_columns(xlsx_path: Path, required_cols: list[str], preferred: str | None = None) -> str:
    """Return the first sheet containing all required columns.

    This makes the workflow robust to different sheet names and gives a much
    clearer error than pandas/openpyxl default messages.
    """
    xls = pd.ExcelFile(xlsx_path)
    sheet_names = list(xls.sheet_names)
    if preferred and preferred in sheet_names:
        probe = pd.read_excel(xlsx_path, sheet_name=preferred, nrows=3)
        if all(c in probe.columns for c in required_cols):
            return preferred
    for s in sheet_names:
        try:
            probe = pd.read_excel(xlsx_path, sheet_name=s, nrows=3)
        except Exception:
            continue
        if all(c in probe.columns for c in required_cols):
            return s
    raise ValueError(
        f"Could not find a worksheet with columns {required_cols!r} in {xlsx_path.name}. Available sheets: {sheet_names}"
    )

# ---------------------------- utilities ----------------------------

def _to_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace(",", ".").replace("−", "-").strip()
        if value == "":
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def depolarization_factor_spheroid(alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("aspect ratio must be > 0")
    if np.isclose(alpha, 1.0):
        return 1.0 / 3.0
    if alpha < 1.0:
        t1 = alpha**2
        t2 = 1.0 / t1
        t4 = np.sqrt(t2 - 1.0)
        t5 = np.arctan(t4)
        return float(t2 * (t4 - t5) / (t4**3))
    t1 = alpha**2
    t2 = 1.0 / t1
    t4 = np.sqrt(1.0 - t2)
    t6 = np.log((1.0 + t4) / (1.0 - t4))
    return float(t2 * (0.5 * t6 - t4) / (t4**3))


def mt_directional_update(k_host: float, k_inc: float, c_inc: float, alpha: float, axis: str) -> float:
    c_inc = float(c_inc)
    if c_inc <= 0:
        return float(k_host)
    if c_inc >= 1.0:
        return float(k_inc)

    n3 = depolarization_factor_spheroid(alpha)
    n1 = (1.0 - n3) / 2.0
    n = n3 if axis == "33" else n1

    denom = k_host + n * (k_inc - k_host)
    denom = max(denom, 1e-12)
    a = k_host / denom
    return float(k_host + c_inc * (k_inc - k_host) * a / ((1.0 - c_inc) + c_inc * a))


def layered_average(k_host_11: float, k_host_33: float, k_lens: float, f_lens_bulk: float) -> Tuple[float, float]:
    f = float(np.clip(f_lens_bulk, 0.0, 0.999999))
    k11 = (1.0 - f) * k_host_11 + f * k_lens
    denom = (1.0 - f) / max(k_host_33, 1e-12) + f / max(k_lens, 1e-12)
    k33 = 1.0 / denom
    return float(k11), float(k33)


def relative_misfit(obs11, obs33, mod11, mod33) -> float:
    e11 = (mod11 - obs11) / max(obs11, 1e-9)
    e33 = (mod33 - obs33) / max(obs33, 1e-9)
    return float(e11 * e11 + e33 * e33)


# ---------------------------- assumptions ----------------------------

DENSITY = {
    "quartz": 2.65,
    "pyrite": 5.02,
    "feldspar": 2.62,
    "dolomite": 2.87,
    "calcite": 2.71,
    "illite": 2.58,
    "kaolinite": 2.60,
    "anhydrite": 2.98,
}

THERMAL = {
    "clay_11": 2.0,
    "clay_33": 1.0,
    "quartz": 6.5,
    "pyrite": 18.0,
    "feldspar": 2.3,
    "dolomite": 5.5,
    "calcite": 3.6,
    "anhydrite": 5.0,
    "pore": 0.026,
}


# ---------------------------- data prep ----------------------------

def load_before_curated(path: Path) -> pd.DataFrame:
    required = ["sample_id", "depth_m", "lithotype", "lambda11_dry", "lambda33_dry", "phi_total_frac", "TOC_wt"]
    sheet = _find_sheet_with_columns(path, required_cols=required, preferred="input")
    df = pd.read_excel(path, sheet_name=sheet)
    out = df[[
        "sample_id", "depth_m", "lithotype",
        "lambda11_dry", "lambda33_dry",
        "phi_total_frac", "TOC_wt"
    ]].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["state"] = "before"
    out = out.rename(columns={
        "lambda11_dry": "lambda11_obs",
        "lambda33_dry": "lambda33_obs",
    })
    return out


def load_after_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Samples_raw", header=None)
    rows = []
    for idx in range(2, raw.shape[0]):
        sample_id = raw.iat[idx, 1]
        if pd.isna(sample_id):
            continue
        phi_raw = _to_float(raw.iat[idx, 31])
        rows.append({
            "sample_id": str(sample_id),
            "lithotype": raw.iat[idx, 3],
            "lambda11_obs": _to_float(raw.iat[idx, 24]),
            "lambda33_obs": _to_float(raw.iat[idx, 25]),
            "TOC_wt": _to_float(raw.iat[idx, 30]),
            "phi_total_frac": (phi_raw / 100.0 if np.isfinite(phi_raw) and phi_raw > 1.0 else phi_raw),
            "state": "after",
        })
    return pd.DataFrame(rows)


def load_xrd_before_imputed(path: Path) -> pd.DataFrame:
    xrd = pd.read_excel(path, sheet_name="XRD_raw")
    xrd.columns = [
        "sample_id", "lithotype", "state",
        "Quartz_pct", "Pyrite_pct", "Feldspar_pct", "Dolomite_pct",
        "Calcite_pct", "Illite_pct", "Kaolinite_pct", "Anhydrite_pct", "Sum"
    ]
    xrd["sample_id"] = xrd["sample_id"].astype(str)
    for c in xrd.columns[3:]:
        xrd[c] = pd.to_numeric(xrd[c].astype(str).str.replace(",", "."), errors="coerce")

    xrd = xrd[xrd["state"].eq("before")].copy()

    mineral_cols = [
        "Quartz_pct", "Pyrite_pct", "Feldspar_pct", "Dolomite_pct",
        "Calcite_pct", "Illite_pct", "Kaolinite_pct", "Anhydrite_pct"
    ]
    litho_means = xrd.groupby("lithotype")[mineral_cols].mean(numeric_only=True)

    def fill_row(row):
        complete = row[mineral_cols].notna().sum() == len(mineral_cols)
        if complete:
            row["xrd_status"] = "original"
            return row
        if row["lithotype"] in litho_means.index:
            means = litho_means.loc[row["lithotype"]]
            for c in mineral_cols:
                if pd.isna(row[c]):
                    row[c] = means[c]
            row["xrd_status"] = "imputed_lithotype_mean"
        else:
            row["xrd_status"] = "missing_lithotype_mean"
        return row

    xrd = xrd.apply(fill_row, axis=1)

    xrd["clay_pct"] = xrd["Illite_pct"].fillna(0.0) + xrd["Kaolinite_pct"].fillna(0.0)
    xrd["quartz_pct"] = xrd["Quartz_pct"].fillna(0.0)
    xrd["carb_pct"] = xrd["Calcite_pct"].fillna(0.0) + xrd["Dolomite_pct"].fillna(0.0)
    xrd["pyrite_pct"] = xrd["Pyrite_pct"].fillna(0.0)
    xrd["feldspar_pct"] = xrd["Feldspar_pct"].fillna(0.0)
    xrd["anhydrite_pct"] = xrd["Anhydrite_pct"].fillna(0.0)

    return xrd[[
        "sample_id", "lithotype", "xrd_status",
        "quartz_pct", "pyrite_pct", "feldspar_pct",
        "carb_pct", "clay_pct", "anhydrite_pct",
        "Calcite_pct", "Dolomite_pct", "Illite_pct", "Kaolinite_pct"
    ]].copy()


def build_two_state_dataset(curated_before_path: Path, raw_after_path: Path) -> pd.DataFrame:
    before = load_before_curated(curated_before_path)
    after = load_after_raw(raw_after_path)
    xrd = load_xrd_before_imputed(raw_after_path)

    common_ids = sorted(set(before["sample_id"]) & set(after["sample_id"]) & set(xrd["sample_id"]))
    before = before[before["sample_id"].isin(common_ids)].copy()
    after = after[after["sample_id"].isin(common_ids)].copy()
    xrd = xrd[xrd["sample_id"].isin(common_ids)].copy()

    long_df = pd.concat([before, after], ignore_index=True)
    long_df = long_df.merge(xrd, on=["sample_id", "lithotype"], how="left")

    depth_map = before.set_index("sample_id")["depth_m"].to_dict()
    long_df["depth_m"] = long_df["sample_id"].map(depth_map)
    long_df = long_df.dropna(subset=["lambda11_obs", "lambda33_obs", "TOC_wt", "phi_total_frac"])
    return long_df.reset_index(drop=True)


# ---------------------------- TOC -> OM ----------------------------

def mineral_mixture_density(row: pd.Series) -> float:
    x = {
        "quartz": row["quartz_pct"],
        "pyrite": row["pyrite_pct"],
        "feldspar": row["feldspar_pct"],
        "anhydrite": row["anhydrite_pct"],
        "illite": row["Illite_pct"],
        "kaolinite": row["Kaolinite_pct"],
        "calcite": row["Calcite_pct"],
        "dolomite": row["Dolomite_pct"],
    }
    total = sum(v for v in x.values() if pd.notna(v))
    if total <= 0:
        return np.nan

    inv_rho = 0.0
    for key, wt in x.items():
        wt = 0.0 if pd.isna(wt) else float(wt)
        if wt <= 0:
            continue
        inv_rho += (wt / total) / DENSITY[key]
    return 1.0 / inv_rho


def add_om_bulk_volume(df: pd.DataFrame, rho_om: float, k_om_toc: float) -> pd.DataFrame:
    out = df.copy()
    out["rho_min"] = out.apply(mineral_mixture_density, axis=1)
    out["toc_frac"] = out["TOC_wt"] / 100.0
    out["w_om"] = np.clip(k_om_toc * out["toc_frac"], 0.0, 0.95)

    denom = (out["w_om"] / rho_om) + ((1.0 - out["w_om"]) / out["rho_min"])
    out["v_om_solid"] = np.where(denom > 0, (out["w_om"] / rho_om) / denom, np.nan)
    out["V_om_bulk"] = (1.0 - out["phi_total_frac"]) * out["v_om_solid"]
    out["V_min_total_bulk"] = (1.0 - out["phi_total_frac"]) * (1.0 - out["v_om_solid"])

    group_cols = ["quartz_pct", "pyrite_pct", "feldspar_pct", "carb_pct", "clay_pct", "anhydrite_pct"]
    gsum = out[group_cols].sum(axis=1)
    for gc in group_cols:
        out["V_" + gc.replace("_pct", "")] = np.where(
            gsum > 0,
            out["V_min_total_bulk"] * out[gc] / gsum,
            np.nan,
        )

    out["closure_check"] = (
        out["V_om_bulk"]
        + out["V_quartz"]
        + out["V_pyrite"]
        + out["V_feldspar"]
        + out["V_carb"]
        + out["V_clay"]
        + out["V_anhydrite"]
        + out["phi_total_frac"]
    )
    return out


# ---------------------------- forward model ----------------------------

@dataclass
class BaseParams:
    f_lens_before: float
    phi_patch_share_before: float
    alpha_matrix_pore: float
    alpha_spongy: float
    alpha_lens: float


@dataclass
class AfterParams:
    lambda_om_after: float
    delta_f_lens_after: float  # should be <= 0 in grid
    delta_phi_patch_share_after: float  # >= 0 in grid
    phi_shrink_after: float
    alpha_shrink_after: float
    phi_cr_m_after: float
    alpha_cr_m_after: float
    crack_mix_after: float


def _host_with_patchy_om_and_pores(
    row: pd.Series,
    lambda_om_solid: float,
    f_lens: float,
    phi_patch_share: float,
    alpha_matrix_pore: float,
    alpha_spongy: float,
    extra_phi_shrink: float = 0.0,
    alpha_shrink: float = 1e-3,
) -> tuple[float, float, float, float, float]:
    """Build host before matrix cracks and before layered insertion of lenticular OM.

    Returns
    -------
    k11, k33, V_lens_bulk, V_patch_bulk, phi_matrix
    """
    f_lens = float(np.clip(f_lens, 0.0, 0.95))
    phi_total = float(np.clip(row["phi_total_frac"], 0.0, 0.8))
    phi_patch = float(np.clip(phi_patch_share, 0.0, 1.0)) * phi_total
    phi_matrix = max(phi_total - phi_patch, 0.0)

    V_om_bulk = max(float(row["V_om_bulk"]), 0.0)
    V_lens_bulk = f_lens * V_om_bulk
    V_patch_bulk = max(V_om_bulk - V_lens_bulk, 0.0)

    comps = {
        "clay": max(float(row["V_clay"]), 0.0),
        "quartz": max(float(row["V_quartz"]), 0.0),
        "carb": max(float(row["V_carb"]), 0.0),
        "pyrite": max(float(row["V_pyrite"]), 0.0),
        "feldspar": max(float(row["V_feldspar"]), 0.0),
        "anhydrite": max(float(row["V_anhydrite"]), 0.0),
        "om_patch": V_patch_bulk,
        "pore_m": phi_matrix,
        "pore_sp": phi_patch,
        "pore_sh": max(float(extra_phi_shrink), 0.0),
    }
    host_total = sum(comps.values())
    if host_total <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    k11 = THERMAL["clay_11"]
    k33 = THERMAL["clay_33"]
    built = max(comps["clay"] / host_total, 1e-9)

    ordered = [
        ("quartz", THERMAL["quartz"], 1.0),
        ("carb", THERMAL["calcite"], 1.0),
        ("pyrite", THERMAL["pyrite"], 1.0),
        ("feldspar", THERMAL["feldspar"], 1.0),
        ("anhydrite", THERMAL["anhydrite"], 1.0),
        ("om_patch", lambda_om_solid, 1.0),
        ("pore_m", THERMAL["pore"], alpha_matrix_pore),
        ("pore_sp", THERMAL["pore"], alpha_spongy),
        ("pore_sh", THERMAL["pore"], alpha_shrink),
    ]

    for name, k_inc, alpha in ordered:
        v = comps[name] / host_total
        if v <= 0:
            continue
        c_rel = v / (built + v)
        k11 = mt_directional_update(k11, k_inc, c_rel, alpha, axis="11")
        k33 = mt_directional_update(k33, k_inc, c_rel, alpha, axis="33")
        built += v

    return float(k11), float(k33), float(V_lens_bulk), float(V_patch_bulk), float(phi_matrix)


def _apply_matrix_cracks(k11: float, k33: float, phi_cr: float, alpha_cr: float, crack_mix: float) -> tuple[float, float]:
    if phi_cr <= 0:
        return float(k11), float(k33)

    pore_k = THERMAL["pore"]
    mix = float(np.clip(crack_mix, 0.0, 1.0))

    # aligned cracks: strongest VTI effect
    k11_al = mt_directional_update(k11, pore_k, phi_cr, alpha_cr, axis="11")
    k33_al = mt_directional_update(k33, pore_k, phi_cr, alpha_cr, axis="33")

    # random crack network approximation: isotropized host updated once
    k_iso = (2.0 * k11 + k33) / 3.0
    k_iso2 = mt_directional_update(k_iso, pore_k, phi_cr, alpha_cr, axis="11")
    k11_rand = k_iso2
    k33_rand = k_iso2

    out11 = mix * k11_al + (1.0 - mix) * k11_rand
    out33 = mix * k33_al + (1.0 - mix) * k33_rand
    return float(out11), float(out33)


def predict_before(row: pd.Series, lambda_om_before: float, params: BaseParams) -> tuple[float, float, dict]:
    host11, host33, V_lens_bulk, V_patch_bulk, phi_matrix = _host_with_patchy_om_and_pores(
        row,
        lambda_om_solid=lambda_om_before,
        f_lens=params.f_lens_before,
        phi_patch_share=params.phi_patch_share_before,
        alpha_matrix_pore=params.alpha_matrix_pore,
        alpha_spongy=params.alpha_spongy,
    )
    if not np.isfinite(host11) or not np.isfinite(host33):
        return np.nan, np.nan, {}

    mod11, mod33 = layered_average(host11, host33, lambda_om_before, V_lens_bulk)
    aux = {
        "V_lens_bulk_before": V_lens_bulk,
        "V_patch_bulk_before": V_patch_bulk,
        "phi_matrix_before": phi_matrix,
        "phi_patch_before": params.phi_patch_share_before * float(row["phi_total_frac"]),
    }
    return mod11, mod33, aux


def predict_after(row: pd.Series, base: BaseParams, aft: AfterParams) -> tuple[float, float, dict]:
    f_after = float(np.clip(base.f_lens_before + aft.delta_f_lens_after, 0.0, 0.95))
    phi_patch_share_after = float(np.clip(base.phi_patch_share_before + aft.delta_phi_patch_share_after, 0.0, 0.98))

    host11, host33, V_lens_bulk, V_patch_bulk, phi_matrix = _host_with_patchy_om_and_pores(
        row,
        lambda_om_solid=aft.lambda_om_after,
        f_lens=f_after,
        phi_patch_share=phi_patch_share_after,
        alpha_matrix_pore=base.alpha_matrix_pore,
        alpha_spongy=base.alpha_spongy,
        extra_phi_shrink=aft.phi_shrink_after,
        alpha_shrink=aft.alpha_shrink_after,
    )
    if not np.isfinite(host11) or not np.isfinite(host33):
        return np.nan, np.nan, {}

    host11, host33 = _apply_matrix_cracks(
        host11,
        host33,
        phi_cr=aft.phi_cr_m_after,
        alpha_cr=aft.alpha_cr_m_after,
        crack_mix=aft.crack_mix_after,
    )

    mod11, mod33 = layered_average(host11, host33, aft.lambda_om_after, V_lens_bulk)
    aux = {
        "f_lens_after": f_after,
        "phi_patch_share_after": phi_patch_share_after,
        "V_lens_bulk_after": V_lens_bulk,
        "V_patch_bulk_after": V_patch_bulk,
        "phi_matrix_after": phi_matrix,
        "phi_patch_after": phi_patch_share_after * float(row["phi_total_frac"]),
    }
    return mod11, mod33, aux


# ---------------------------- fitting ----------------------------

def random_search_sample_pair(
    row_before: pd.Series,
    row_after: pd.Series,
    lambda_om_before: float,
    after_params: AfterParams,
    n_trials: int,
    rng: np.random.Generator,
):
    best = None
    best_loss = np.inf

    for _ in range(n_trials):
        base = BaseParams(
            f_lens_before=float(rng.uniform(0.10, 0.85)),
            phi_patch_share_before=float(rng.uniform(0.15, 0.95)),
            alpha_matrix_pore=float(10.0 ** rng.uniform(-3.3, -0.1)),
            alpha_spongy=float(10.0 ** rng.uniform(-0.3, 0.0)),  # ~0.5 to 1.0
            alpha_lens=float(10.0 ** rng.uniform(-2.0, -0.5)),
        )
        mod11_b, mod33_b, aux_b = predict_before(row_before, lambda_om_before, base)
        mod11_a, mod33_a, aux_a = predict_after(row_after, base, after_params)

        if not all(np.isfinite(x) for x in [mod11_b, mod33_b, mod11_a, mod33_a]):
            continue

        loss_b = relative_misfit(row_before["lambda11_obs"], row_before["lambda33_obs"], mod11_b, mod33_b)
        loss_a = relative_misfit(row_after["lambda11_obs"], row_after["lambda33_obs"], mod11_a, mod33_a)
        total = 0.5 * (loss_b + loss_a)

        if total < best_loss:
            best_loss = total
            best = {
                "f_lens_before": base.f_lens_before,
                "phi_patch_share_before": base.phi_patch_share_before,
                "alpha_matrix_pore": base.alpha_matrix_pore,
                "alpha_spongy": base.alpha_spongy,
                "alpha_lens": base.alpha_lens,
                **aux_b,
                **aux_a,
                "lambda11_mod_before": mod11_b,
                "lambda33_mod_before": mod33_b,
                "loss_before": loss_b,
                "lambda11_mod_after": mod11_a,
                "lambda33_mod_after": mod33_a,
                "loss_after": loss_a,
                "loss_total": total,
            }

    return best


def run_screening(
    df: pd.DataFrame,
    rho_grid: Iterable[float],
    k_grid: Iterable[float],
    lambda_om_before_grid: Iterable[float],
    lambda_om_after_grid: Iterable[float],
    delta_f_lens_after_grid: Iterable[float],
    delta_phi_patch_share_after_grid: Iterable[float],
    phi_shrink_after_grid: Iterable[float],
    alpha_shrink_after_grid: Iterable[float],
    phi_cr_m_after_grid: Iterable[float],
    alpha_cr_m_after_grid: Iterable[float],
    crack_mix_after_grid: Iterable[float],
    n_trials: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    scenario_rows = []
    best_detail_df = None
    best_prepared_df = None
    best_loss = np.inf

    samples = sorted(df["sample_id"].unique())
    for rho_om in rho_grid:
        for k_om_toc in k_grid:
            prepared = add_om_bulk_volume(df, rho_om=rho_om, k_om_toc=k_om_toc)
            before_map = {r["sample_id"]: r for _, r in prepared[prepared["state"] == "before"].iterrows()}
            after_map = {r["sample_id"]: r for _, r in prepared[prepared["state"] == "after"].iterrows()}

            for lambda_om_before in lambda_om_before_grid:
                for lambda_om_after in lambda_om_after_grid:
                    for delta_f_lens_after in delta_f_lens_after_grid:
                        for delta_phi_patch_share_after in delta_phi_patch_share_after_grid:
                            for phi_shrink_after in phi_shrink_after_grid:
                                for alpha_shrink_after in alpha_shrink_after_grid:
                                    for phi_cr_m_after in phi_cr_m_after_grid:
                                        for alpha_cr_m_after in alpha_cr_m_after_grid:
                                            for crack_mix_after in crack_mix_after_grid:
                                                aft = AfterParams(
                                                    lambda_om_after=float(lambda_om_after),
                                                    delta_f_lens_after=float(delta_f_lens_after),
                                                    delta_phi_patch_share_after=float(delta_phi_patch_share_after),
                                                    phi_shrink_after=float(phi_shrink_after),
                                                    alpha_shrink_after=float(alpha_shrink_after),
                                                    phi_cr_m_after=float(phi_cr_m_after),
                                                    alpha_cr_m_after=float(alpha_cr_m_after),
                                                    crack_mix_after=float(crack_mix_after),
                                                )
                                                details = []
                                                total_loss = 0.0
                                                valid_n = 0
                                                for sid in samples:
                                                    if sid not in before_map or sid not in after_map:
                                                        continue
                                                    fit = random_search_sample_pair(
                                                        before_map[sid],
                                                        after_map[sid],
                                                        lambda_om_before=float(lambda_om_before),
                                                        after_params=aft,
                                                        n_trials=n_trials,
                                                        rng=rng,
                                                    )
                                                    if fit is None:
                                                        continue
                                                    valid_n += 1
                                                    total_loss += fit["loss_total"]
                                                    details.append({
                                                        "sample_id": sid,
                                                        "lithotype": before_map[sid]["lithotype"],
                                                        "depth_m": before_map[sid]["depth_m"],
                                                        "rho_om": rho_om,
                                                        "k_om_toc": k_om_toc,
                                                        "lambda_om_before": lambda_om_before,
                                                        "lambda_om_after": lambda_om_after,
                                                        "delta_f_lens_after": delta_f_lens_after,
                                                        "delta_phi_patch_share_after": delta_phi_patch_share_after,
                                                        "phi_shrink_after": phi_shrink_after,
                                                        "alpha_shrink_after": alpha_shrink_after,
                                                        "phi_cr_m_after": phi_cr_m_after,
                                                        "alpha_cr_m_after": alpha_cr_m_after,
                                                        "crack_mix_after": crack_mix_after,
                                                        **fit,
                                                    })
                                                mean_loss = total_loss / max(valid_n, 1)
                                                scenario_rows.append({
                                                    "rho_om": rho_om,
                                                    "k_om_toc": k_om_toc,
                                                    "lambda_om_before": lambda_om_before,
                                                    "lambda_om_after": lambda_om_after,
                                                    "delta_f_lens_after": delta_f_lens_after,
                                                    "delta_phi_patch_share_after": delta_phi_patch_share_after,
                                                    "phi_shrink_after": phi_shrink_after,
                                                    "alpha_shrink_after": alpha_shrink_after,
                                                    "phi_cr_m_after": phi_cr_m_after,
                                                    "alpha_cr_m_after": alpha_cr_m_after,
                                                    "crack_mix_after": crack_mix_after,
                                                    "n_pairs": valid_n,
                                                    "mean_loss": mean_loss,
                                                })
                                                if valid_n > 0 and mean_loss < best_loss:
                                                    best_loss = mean_loss
                                                    best_detail_df = pd.DataFrame(details)
                                                    best_prepared_df = prepared.copy()

    scenario_df = pd.DataFrame(scenario_rows).sort_values("mean_loss", ascending=True).reset_index(drop=True)
    return scenario_df, best_detail_df, best_prepared_df


# ---------------------------- outputs ----------------------------

def assemble_best_pair_long(best_detail_df: pd.DataFrame, prepared_df: pd.DataFrame) -> pd.DataFrame:
    before = prepared_df[prepared_df["state"] == "before"][["sample_id", "lambda11_obs", "lambda33_obs", "TOC_wt", "phi_total_frac", "V_om_bulk"]].copy()
    after = prepared_df[prepared_df["state"] == "after"][["sample_id", "lambda11_obs", "lambda33_obs", "TOC_wt", "phi_total_frac", "V_om_bulk"]].copy()
    bmap = before.set_index("sample_id")
    amap = after.set_index("sample_id")

    rows = []
    for _, r in best_detail_df.iterrows():
        sid = r["sample_id"]
        br = bmap.loc[sid]
        ar = amap.loc[sid]
        rows.append({
            "sample_id": sid, "state": "before", "lithotype": r["lithotype"], "depth_m": r["depth_m"],
            "rho_om": r["rho_om"], "k_om_toc": r["k_om_toc"],
            "lambda_om_before": r["lambda_om_before"], "lambda_om_after": r["lambda_om_after"],
            "TOC_wt": br["TOC_wt"], "phi_total_frac": br["phi_total_frac"], "V_om_bulk": br["V_om_bulk"],
            "lambda11_obs": br["lambda11_obs"], "lambda33_obs": br["lambda33_obs"],
            "lambda11_mod": r["lambda11_mod_before"], "lambda33_mod": r["lambda33_mod_before"],
            "loss": r["loss_before"],
            "f_lens_before": r["f_lens_before"], "f_lens_after": r["f_lens_after"],
            "phi_patch_share_before": r["phi_patch_share_before"], "phi_patch_share_after": r["phi_patch_share_after"],
            "phi_patch_before": r["phi_patch_before"], "phi_patch_after": r["phi_patch_after"],
            "phi_matrix_before": r["phi_matrix_before"], "phi_matrix_after": r["phi_matrix_after"],
            "V_lens_bulk_before": r["V_lens_bulk_before"], "V_patch_bulk_before": r["V_patch_bulk_before"],
            "phi_shrink_after": r.get("phi_shrink_after", np.nan),
            "phi_cr_m_after": r.get("phi_cr_m_after", np.nan),
        })
        rows.append({
            "sample_id": sid, "state": "after", "lithotype": r["lithotype"], "depth_m": r["depth_m"],
            "rho_om": r["rho_om"], "k_om_toc": r["k_om_toc"],
            "lambda_om_before": r["lambda_om_before"], "lambda_om_after": r["lambda_om_after"],
            "TOC_wt": ar["TOC_wt"], "phi_total_frac": ar["phi_total_frac"], "V_om_bulk": ar["V_om_bulk"],
            "lambda11_obs": ar["lambda11_obs"], "lambda33_obs": ar["lambda33_obs"],
            "lambda11_mod": r["lambda11_mod_after"], "lambda33_mod": r["lambda33_mod_after"],
            "loss": r["loss_after"],
            "f_lens_before": r["f_lens_before"], "f_lens_after": r["f_lens_after"],
            "phi_patch_share_before": r["phi_patch_share_before"], "phi_patch_share_after": r["phi_patch_share_after"],
            "phi_patch_before": r["phi_patch_before"], "phi_patch_after": r["phi_patch_after"],
            "phi_matrix_before": r["phi_matrix_before"], "phi_matrix_after": r["phi_matrix_after"],
            "V_lens_bulk_after": r["V_lens_bulk_after"], "V_patch_bulk_after": r["V_patch_bulk_after"],
            "phi_shrink_after": r.get("phi_shrink_after", np.nan),
            "phi_cr_m_after": r.get("phi_cr_m_after", np.nan),
        })
    out = pd.DataFrame(rows)
    out["A_obs"] = out["lambda11_obs"] / out["lambda33_obs"]
    out["A_mod"] = out["lambda11_mod"] / out["lambda33_mod"]
    return out


def save_basic_plots(best_pair_long: pd.DataFrame, scenario_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for state in ["before", "after"]:
        sub = best_pair_long[best_pair_long["state"] == state]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, comp in zip(axes, ["11", "33"], strict=True):
            obs = sub[f"lambda{comp}_obs"]
            mod = sub[f"lambda{comp}_mod"]
            ax.scatter(obs, mod)
            mn = min(obs.min(), mod.min())
            mx = max(obs.max(), mod.max())
            ax.plot([mn, mx], [mn, mx], "k--", lw=1)
            ax.set_xlabel(f"Observed λ{comp}")
            ax.set_ylabel(f"Modeled λ{comp}")
            ax.set_title(f"{state}: λ{comp}")
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{state}_obs_vs_mod.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    top = scenario_df.head(min(120, len(scenario_df))).copy()
    sc = ax.scatter(top["delta_f_lens_after"], top["mean_loss"], c=top["delta_phi_patch_share_after"], s=45)
    ax.set_xlabel("delta_f_lens_after")
    ax.set_ylabel("mean_loss")
    ax.set_title("Top scenarios: lens decrease vs loss")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("delta_phi_patch_share_after")
    fig.tight_layout()
    fig.savefig(out_dir / "top_scenarios_lens_vs_patchgrowth.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grp = best_pair_long.groupby("state")["loss"].mean()
    ax.bar(grp.index, grp.values)
    ax.set_ylabel("Mean loss")
    ax.set_title("Best scenario mean loss by state")
    fig.tight_layout()
    fig.savefig(out_dir / "best_state_mean_loss.png", dpi=200)
    plt.close(fig)


# ---------------------------- main ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before-xlsx", type=Path, default=Path("/mnt/data/forward_anisotropic_results_v2 (1).xlsx"))
    p.add_argument("--raw-xlsx", type=Path, default=Path("/mnt/data/fd8a84c5-d16a-406f-b3fb-e90d5a0e7f66.xlsx"))
    p.add_argument("--out-dir", type=Path, default=Path("/mnt/data/bazhenov_fifthlook_outputs"))
    p.add_argument("--n-trials", type=int, default=140)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = build_two_state_dataset(args.before_xlsx, args.raw_xlsx)

    # keep OM property in a physical, narrow range; let geometry do the work.
    rho_grid = [1.10, 1.20]
    k_grid = [1.0]
    lambda_om_before_grid = [0.18, 0.22]
    lambda_om_after_grid = [0.16, 0.20]

    # physics-informed post-tube ranges
    delta_f_lens_after_grid = [-0.50, -0.30, -0.15, 0.0]  # lenses do not increase
    delta_phi_patch_share_after_grid = [0.00, 0.10, 0.20, 0.30]  # patchy OM becomes more porous
    phi_shrink_after_grid = [0.00, 0.01, 0.02, 0.04]  # crack-like pores inside OM
    alpha_shrink_after_grid = [1e-4, 1e-3]
    phi_cr_m_after_grid = [0.00, 0.01, 0.02, 0.04]  # matrix cracks
    alpha_cr_m_after_grid = [1e-4, 1e-3]
    crack_mix_after_grid = [0.0, 0.5, 1.0]

    scenario_df, best_detail_df, best_prepared_df = run_screening(
        df,
        rho_grid=rho_grid,
        k_grid=k_grid,
        lambda_om_before_grid=lambda_om_before_grid,
        lambda_om_after_grid=lambda_om_after_grid,
        delta_f_lens_after_grid=delta_f_lens_after_grid,
        delta_phi_patch_share_after_grid=delta_phi_patch_share_after_grid,
        phi_shrink_after_grid=phi_shrink_after_grid,
        alpha_shrink_after_grid=alpha_shrink_after_grid,
        phi_cr_m_after_grid=phi_cr_m_after_grid,
        alpha_cr_m_after_grid=alpha_cr_m_after_grid,
        crack_mix_after_grid=crack_mix_after_grid,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(args.out_dir / "scenario_ranking.csv", index=False)

    best_pair_long = assemble_best_pair_long(best_detail_df, best_prepared_df)
    best_pair_long.to_csv(args.out_dir / "best_scenario_sample_fits.csv", index=False)
    best_prepared_df.to_csv(args.out_dir / "prepared_dataset_best_scenario.csv", index=False)
    save_basic_plots(best_pair_long, scenario_df, args.out_dir)

    top = scenario_df.iloc[0]
    with open(args.out_dir / "best_scenario.txt", "w", encoding="utf-8") as f:
        f.write("Best fifth-look scenario with split OM physics\n")
        f.write("=" * 68 + "\n")
        for col in [
            "rho_om", "k_om_toc", "lambda_om_before", "lambda_om_after",
            "delta_f_lens_after", "delta_phi_patch_share_after",
            "phi_shrink_after", "alpha_shrink_after",
            "phi_cr_m_after", "alpha_cr_m_after", "crack_mix_after",
            "mean_loss", "n_pairs",
        ]:
            f.write(f"{col}: {top[col]}\n")

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
