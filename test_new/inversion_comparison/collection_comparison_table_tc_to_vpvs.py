from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _q(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def _rate_le(a: np.ndarray, thr: float) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a <= float(thr)))


def _canonical(s: str) -> str:
    return str(s).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collection-level comparison table (by state) for Vp/Vs forecast using TC-inverted alpha."
    )
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("test_new/inversion_comparison/vp_vs_pred_from_tc_alpha_misfit_by_sample.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--thr-pct", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].map(_canonical)
    df["fluid_state"] = df["fluid_state"].map(_canonical)
    df["method"] = df["method"].astype(str).str.strip().str.upper()

    # Build long table of p50 misfit per property
    long_rows: list[dict[str, object]] = []
    for prop in ["vp", "vs", "agg"]:
        for _, r in df.iterrows():
            long_rows.append(
                {
                    "lab_sample_id": float(r["lab_sample_id"]),
                    "stage": str(r["stage"]),
                    "fluid_state": str(r["fluid_state"]),
                    "method": str(r["method"]),
                    "property": prop,
                    "misfit_p50": float(r[f"{prop}_misfit_pct"]),
                    "misfit_p05": float(r[f"{prop}_misfit_pct_p05"]),
                    "misfit_p95": float(r[f"{prop}_misfit_pct_p95"]),
                }
            )
    long_df = pd.DataFrame(long_rows)

    # Aggregate stats per (stage, fluid_state, property, method)
    out_rows: list[dict[str, object]] = []
    for (stage, fluid, prop, method), g in long_df.groupby(["stage", "fluid_state", "property", "method"]):
        p50 = pd.to_numeric(g["misfit_p50"], errors="coerce").to_numpy(float)
        p05 = pd.to_numeric(g["misfit_p05"], errors="coerce").to_numpy(float)
        p95 = pd.to_numeric(g["misfit_p95"], errors="coerce").to_numpy(float)

        out_rows.append(
            {
                "stage": stage,
                "fluid_state": fluid,
                "property": prop,
                "method": method,
                "n": int(np.count_nonzero(np.isfinite(p50))),
                "misfit_median_pct": _q(p50, 50),
                "misfit_mean_pct": float(np.nanmean(p50)),
                "misfit_p05_pct": _q(p50, 5),
                "misfit_p95_pct": _q(p50, 95),
                "band_lo_median_pct": _q(p05, 50),
                "band_hi_median_pct": _q(p95, 50),
                "success_rate_median_le_thr": _rate_le(p50, float(args.thr_pct)),
                "success_rate_upper_le_thr": _rate_le(p95, float(args.thr_pct)),
            }
        )

    summary_long = pd.DataFrame(out_rows).sort_values(
        ["property", "stage", "fluid_state", "method"]
    ).reset_index(drop=True)

    # Wide view for easier reading: index=(property, stage, fluid_state), columns=method×metric
    wide = summary_long.pivot_table(
        index=["property", "stage", "fluid_state"],
        columns=["method"],
        values=[
            "n",
            "misfit_median_pct",
            "misfit_mean_pct",
            "misfit_p05_pct",
            "misfit_p95_pct",
            "band_lo_median_pct",
            "band_hi_median_pct",
            "success_rate_median_le_thr",
            "success_rate_upper_le_thr",
        ],
        aggfunc="first",
    )
    wide.columns = ["_".join([a, b]) for (a, b) in wide.columns.to_flat_index()]
    wide = wide.reset_index().sort_values(["property", "stage", "fluid_state"]).reset_index(drop=True)

    # Differences (OSP - MT) for the key metrics
    def diff(col: str) -> pd.Series:
        a = pd.to_numeric(wide.get(f"{col}_OSP"), errors="coerce")
        b = pd.to_numeric(wide.get(f"{col}_MT"), errors="coerce")
        return a - b

    diff_df = wide[["property", "stage", "fluid_state"]].copy()
    for col in ["misfit_median_pct", "misfit_mean_pct", "success_rate_median_le_thr"]:
        diff_df[f"{col}_OSP_minus_MT"] = diff(col)
    diff_df = diff_df.sort_values(["property", "stage", "fluid_state"]).reset_index(drop=True)

    out_xlsx = out_dir / "collection_comparison_tc_to_vpvs_by_state.xlsx"
    with pd.ExcelWriter(out_xlsx) as xw:
        summary_long.to_excel(xw, sheet_name="summary_long", index=False)
        wide.to_excel(xw, sheet_name="summary_wide", index=False)
        diff_df.to_excel(xw, sheet_name="osp_minus_mt", index=False)

    out_csv = out_dir / "collection_comparison_tc_to_vpvs_by_state_summary_wide.csv"
    wide.to_csv(out_csv, index=False)

    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

