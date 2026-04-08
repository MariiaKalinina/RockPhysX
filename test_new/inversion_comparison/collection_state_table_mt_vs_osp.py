from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _canon(s: object) -> str:
    return str(s).strip().lower()


def _canon_property(s: object) -> str:
    key = _canon(s)
    return {
        "tc_w_mk": "tc",
        "vp_m_s": "vp",
        "vs_m_s": "vs",
        "tc_dry": "tc",
        "tc_wet": "tc",
        "vp_dry": "vp",
        "vp_wet": "vp",
        "vs_dry": "vs",
        "vs_wet": "vs",
    }.get(key, key)


def _canon_fluid_state(category_or_fluid: object) -> str:
    key = _canon(category_or_fluid)
    if key in {"dry", "wet"}:
        return key
    if key.endswith("_dry"):
        return "dry"
    if key.endswith("_wet"):
        return "wet"
    return key


def _canon_stage(s: object) -> str:
    key = _canon(s)
    if key in {"before", "after"}:
        return key
    return key


def _load_osp_misfits(path: Path) -> pd.DataFrame:
    """
    OSP (GSA) results: `test_new/gsa/plots/gsa_tc_elastic_m1_m2_results.xlsx`
    sheet `misfit_detail`.
    """
    df = pd.read_excel(path, sheet_name="misfit_detail")
    df = df.copy()
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce").astype(float)
    df["branch"] = df["mode"].map(_canon)
    df["model"] = df["model"].astype(str).str.strip().str.upper()
    df["stage"] = df["stage"].map(_canon_stage)
    df["fluid_state"] = df["fluid_state"].map(_canon_fluid_state)
    df["property"] = df["category"].map(_canon_property)
    df["misfit_pct"] = pd.to_numeric(df["misfit_pct"], errors="coerce").astype(float)
    df["method"] = "OSP"
    return df[["lab_sample_id", "method", "branch", "model", "stage", "fluid_state", "property", "misfit_pct"]].copy()


def _load_mt_tc_only_misfits(path: Path) -> pd.DataFrame:
    """
    MT TC-only results: `test_new/mt_tc_m1_m2_results.xlsx`, sheet `predictions`.
    Contains both M1 and M2 for TC only.
    """
    df = pd.read_excel(path, sheet_name="predictions")
    df = df.copy()
    df["lab_sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].map(_canon_stage)
    df["fluid_state"] = df["fluid_state"].map(_canon_fluid_state)
    df["property"] = "tc"
    df["misfit_pct"] = (
        (pd.to_numeric(df["tc_pred_w_mk"], errors="coerce") - pd.to_numeric(df["tc_obs_w_mk"], errors="coerce"))
        .abs()
        .to_numpy(float)
    )
    df["misfit_pct"] = df["misfit_pct"] / pd.to_numeric(df["tc_obs_w_mk"], errors="coerce").abs() * 100.0
    df["model"] = df["model"].astype(str).map(_canon)
    df["model"] = df["model"].map(
        {"m1_single_effective_ar": "M1", "m2_beta_ar_distribution": "M2"}
    ).fillna(df["model"].str.upper())
    df["branch"] = "tc_only"
    df["method"] = "MT"
    return df[["lab_sample_id", "method", "branch", "model", "stage", "fluid_state", "property", "misfit_pct"]].copy()


def _load_mt_joint_misfits(path: Path) -> pd.DataFrame:
    """
    MT joint (TC+Vp+Vs) results: `test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx`, sheet `predictions`.
    Contains M1 and M2 for {tc,vp,vs}.
    """
    df = pd.read_excel(path, sheet_name="predictions")
    df = df.copy()
    df["lab_sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].map(_canon_stage)
    df["fluid_state"] = df["fluid_state"].map(_canon_fluid_state)
    df["model"] = df["model"].astype(str).str.strip().str.upper()
    df["property"] = df["property"].map(_canon_property)
    obs = pd.to_numeric(df["obs"], errors="coerce").astype(float).to_numpy(float)
    pred = pd.to_numeric(df["pred"], errors="coerce").astype(float).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["misfit_pct"] = np.abs(pred - obs) / np.abs(obs) * 100.0
    df["branch"] = "tc_vp_vs"
    df["method"] = "MT"
    return df[["lab_sample_id", "method", "branch", "model", "stage", "fluid_state", "property", "misfit_pct"]].copy()


def _load_delta_z_table(path: Path) -> pd.DataFrame:
    """
    Per-sample alpha / delta_z comparison: `test_new/inversion_comparison/osp_vs_mt_comparison.xlsx`.
    """
    out = []
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        df = xl.parse(sh).copy()
        df["branch"] = _canon(sh)
        df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce").astype(float)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build collection-wide comparison table by state (before/after × dry/wet) for MT vs OSP."
    )
    ap.add_argument(
        "--osp-xlsx",
        type=Path,
        default=Path("test_new/gsa/plots/gsa_tc_elastic_m1_m2_results.xlsx"),
    )
    ap.add_argument(
        "--mt-tc-only-xlsx",
        type=Path,
        default=Path("test_new/mt_tc_m1_m2_results.xlsx"),
    )
    ap.add_argument(
        "--mt-joint-xlsx",
        type=Path,
        default=Path("test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx"),
    )
    ap.add_argument(
        "--delta-z-xlsx",
        type=Path,
        default=Path("test_new/inversion_comparison/osp_vs_mt_comparison.xlsx"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    osp = _load_osp_misfits(Path(args.osp_xlsx))
    mt_tc = _load_mt_tc_only_misfits(Path(args.mt_tc_only_xlsx))
    mt_joint = _load_mt_joint_misfits(Path(args.mt_joint_xlsx))

    long_df = pd.concat([osp, mt_tc, mt_joint], ignore_index=True)
    long_df = long_df.dropna(subset=["lab_sample_id", "misfit_pct"])
    long_df["lab_sample_id"] = long_df["lab_sample_id"].astype(float)
    long_df["misfit_pct"] = long_df["misfit_pct"].astype(float)

    # Keep canonical ordering.
    long_df["property"] = long_df["property"].map(_canon_property)
    long_df = long_df[long_df["property"].isin(["tc", "vp", "vs"])].copy()
    long_df["branch"] = long_df["branch"].map(_canon)
    long_df["stage"] = long_df["stage"].map(_canon_stage)
    long_df["fluid_state"] = long_df["fluid_state"].map(_canon_fluid_state)
    long_df["model"] = long_df["model"].astype(str).str.strip().str.upper()
    long_df["method"] = long_df["method"].astype(str).str.strip().str.upper()

    # Summary stats by group
    summary = (
        long_df.groupby(["method", "branch", "model", "property", "stage", "fluid_state"])["misfit_pct"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values(["branch", "model", "property", "stage", "fluid_state", "method"])
        .reset_index(drop=True)
    )

    # Per-sample wide pivot for convenient reading
    wide = long_df.pivot_table(
        index=["lab_sample_id"],
        columns=["method", "branch", "model", "property", "stage", "fluid_state"],
        values="misfit_pct",
        aggfunc="first",
    )
    wide.columns = ["__".join(map(str, col)) for col in wide.columns.to_flat_index()]
    wide = wide.reset_index().sort_values("lab_sample_id").reset_index(drop=True)

    # Join Δz / alpha table (optional convenience)
    dz = _load_delta_z_table(Path(args.delta_z_xlsx))
    dz = dz.copy()
    dz["branch"] = dz["branch"].map(_canon)

    out_xlsx = out_dir / "collection_state_comparison_mt_vs_osp.xlsx"
    out_csv_long = out_dir / "collection_state_comparison_mt_vs_osp_long.csv"

    with pd.ExcelWriter(out_xlsx) as xw:
        long_df.to_excel(xw, sheet_name="misfit_long", index=False)
        summary.to_excel(xw, sheet_name="misfit_summary", index=False)
        wide.to_excel(xw, sheet_name="misfit_wide", index=False)
        dz.to_excel(xw, sheet_name="delta_z_table", index=False)

    long_df.to_csv(out_csv_long, index=False)

    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_csv_long}")


if __name__ == "__main__":
    main()

