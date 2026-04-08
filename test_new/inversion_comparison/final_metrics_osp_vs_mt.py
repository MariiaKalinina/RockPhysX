from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_num(a: pd.Series) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(float)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    d = (y[m] - x[m]).astype(float)
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(d**2)))


def _mean_diff(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    d = (y[m] - x[m]).astype(float)
    if d.size == 0:
        return float("nan")
    return float(np.mean(d))


def _sign_match(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) == 0:
        return float("nan")
    return float(np.mean(np.sign(x[m]) == np.sign(y[m])))


def _canonical_property(name: str) -> str:
    key = str(name).strip().lower()
    return {"tc_w_mk": "tc", "vp_m_s": "vp", "vs_m_s": "vs"}.get(key, key)


def _misfit_pct(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(pred - obs) / np.abs(obs) * 100.0


def _median_misfits_joint_m1(
    *,
    osp_m1_misfit_path: Path,
    mt_joint_predictions_path: Path,
) -> pd.DataFrame:
    osp = pd.read_excel(osp_m1_misfit_path, sheet_name="misfit_detail")
    osp = osp[osp["mode"].astype(str) == "tc_vp_vs"].copy()
    osp["property"] = osp["property"].map(_canonical_property)
    osp["misfit_pct"] = pd.to_numeric(osp["misfit_pct"], errors="coerce").astype(float)
    osp["method"] = "OSP"

    mt = pd.read_excel(mt_joint_predictions_path, sheet_name="predictions")
    mt = mt[mt["model"].astype(str).str.upper() == "M1"].copy()
    mt["property"] = mt["property"].map(_canonical_property)
    obs = _to_num(mt["obs"])
    pred = _to_num(mt["pred"])
    mt["misfit_pct"] = _misfit_pct(pred, obs)
    mt["method"] = "MT"

    use = pd.concat(
        [
            osp[["method", "property", "stage", "fluid_state", "misfit_pct"]],
            mt[["method", "property", "stage", "fluid_state", "misfit_pct"]],
        ],
        ignore_index=True,
    )
    out = (
        use.groupby(["method", "property", "stage", "fluid_state"])["misfit_pct"]
        .median()
        .reset_index()
    )
    return out.sort_values(["property", "stage", "fluid_state", "method"]).reset_index(drop=True)


def _median_misfits_tc_only(
    *,
    osp_m1_misfit_path: Path,
    mt_tc_only_predictions_path: Path,
) -> pd.DataFrame:
    osp = pd.read_excel(osp_m1_misfit_path, sheet_name="misfit_detail")
    osp = osp[osp["mode"].astype(str) == "tc_only"].copy()
    osp["property"] = osp["property"].map(_canonical_property)
    osp["misfit_pct"] = pd.to_numeric(osp["misfit_pct"], errors="coerce").astype(float)
    osp["method"] = "OSP"

    mt = pd.read_excel(mt_tc_only_predictions_path, sheet_name="predictions")
    mt = mt.rename(columns={"tc_obs_w_mk": "obs", "tc_pred_w_mk": "pred"})
    obs = _to_num(mt["obs"])
    pred = _to_num(mt["pred"])
    mt["misfit_pct"] = _misfit_pct(pred, obs)
    mt["property"] = "tc"
    mt["method"] = "MT"

    use = pd.concat(
        [
            osp[["method", "property", "stage", "fluid_state", "misfit_pct"]],
            mt[["method", "property", "stage", "fluid_state", "misfit_pct"]],
        ],
        ignore_index=True,
    )
    out = (
        use.groupby(["method", "property", "stage", "fluid_state"])["misfit_pct"]
        .median()
        .reset_index()
    )
    return out.sort_values(["stage", "fluid_state", "method"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final comparison metrics: OSP (GSA) vs Mori–Tanaka.")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--comparison-xlsx", type=Path, default=Path("test_new/inversion_comparison/osp_vs_mt_comparison.xlsx"))
    ap.add_argument("--osp-bayes-m1", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m1_tc_elastic_results.xlsx"))
    ap.add_argument("--mt-tc-only", type=Path, default=Path("test_new/mt_tc_m1_m2_results.xlsx"))
    ap.add_argument("--mt-joint", type=Path, default=Path("test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(args.comparison_xlsx)
    tc_only = xl.parse("tc_only")
    tc_vp_vs = xl.parse("tc_vp_vs")

    metrics_rows: list[dict[str, object]] = []

    # Δz comparisons
    def add_row(branch: str, model: str, xcol: str, ycol: str, *, df: pd.DataFrame, kind: str) -> None:
        x = _to_num(df[xcol])
        y = _to_num(df[ycol])
        m = np.isfinite(x) & np.isfinite(y)
        metrics_rows.append(
            {
                "branch": branch,
                "model": model,
                "metric": kind,
                "n": int(np.count_nonzero(m)),
                "corr": _corr(x, y),
                "rmse": _rmse(x, y),
                "mean_mt": float(np.nanmean(x)),
                "mean_osp": float(np.nanmean(y)),
                "mean_diff_osp_minus_mt": _mean_diff(x, y),
                "sign_match_frac": _sign_match(x, y),
            }
        )

    add_row("TC-only", "M1", "mt_tc_only_m1_delta_z", "osp_m1_delta_z_p50", df=tc_only, kind="Δz")
    add_row("TC-only", "M2", "mt_tc_only_m2_delta_z_p50", "osp_m2_delta_z_p50", df=tc_only, kind="Δz")
    add_row("TC+Vp+Vs", "M1", "mt_tc_vp_vs_m1_delta_z", "osp_m1_delta_z_p50", df=tc_vp_vs, kind="Δz (MT deterministic)")
    add_row("TC+Vp+Vs", "M2", "mt_tc_vp_vs_m2_delta_z_p50", "osp_m2_delta_z_p50", df=tc_vp_vs, kind="Δz (MT deterministic)")
    add_row("TC+Vp+Vs", "M1", "mt_bayes_m1_delta_z_median", "osp_m1_delta_z_p50", df=tc_vp_vs, kind="Δz (MT Bayesian median)")

    # Probability comparisons (Bayesian M1 joint)
    add_row("TC+Vp+Vs", "M1", "mt_bayes_m1_p_delta_z_lt_0", "osp_m1_p_delta_z_lt_0", df=tc_vp_vs, kind="P(Δz<0)")

    metrics = pd.DataFrame(metrics_rows)

    # Median misfits
    misfit_tc_only = _median_misfits_tc_only(
        osp_m1_misfit_path=Path(args.osp_bayes_m1),
        mt_tc_only_predictions_path=Path(args.mt_tc_only),
    )
    misfit_joint = _median_misfits_joint_m1(
        osp_m1_misfit_path=Path(args.osp_bayes_m1),
        mt_joint_predictions_path=Path(args.mt_joint),
    )

    out_xlsx = out_dir / "final_metrics_osp_vs_mt.xlsx"
    with pd.ExcelWriter(out_xlsx) as xw:
        metrics.to_excel(xw, sheet_name="metrics", index=False)
        misfit_tc_only.to_excel(xw, sheet_name="median_misfit_tc_only", index=False)
        misfit_joint.to_excel(xw, sheet_name="median_misfit_joint_m1", index=False)

    out_csv = out_dir / "final_metrics_osp_vs_mt_metrics.csv"
    metrics.to_csv(out_csv, index=False)

    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

