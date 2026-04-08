from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(float)


def _q(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def _agg_stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size >= 2 else 0.0,
        "p05": _q(a, 5),
        "p50": _q(a, 50),
        "p95": _q(a, 95),
    }


def _threshold_rate(a: np.ndarray, thr: float) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a <= float(thr)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Vp/Vs prediction misfit statistics (from TC-inverted alpha).")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("test_new/inversion_comparison/vp_vs_pred_from_tc_alpha_misfit_by_sample.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--thr-pct", type=float, default=5.0, help="Threshold for misfit% success rate.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    # Ensure canonical types
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].astype(str).str.strip().str.lower()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip().str.lower()
    df["method"] = df["method"].astype(str).str.strip().str.upper()

    # Long-format rows for easy groupby
    records: list[dict[str, object]] = []
    for prop in ["vp", "vs", "agg"]:
        v = _to_num(df[f"{prop}_misfit_pct"])
        v05 = _to_num(df[f"{prop}_misfit_pct_p05"])
        v95 = _to_num(df[f"{prop}_misfit_pct_p95"])
        for i in range(len(df)):
            records.append(
                {
                    "lab_sample_id": float(df["lab_sample_id"].iloc[i]),
                    "stage": str(df["stage"].iloc[i]),
                    "fluid_state": str(df["fluid_state"].iloc[i]),
                    "method": str(df["method"].iloc[i]),
                    "property": prop,
                    "misfit_p50": float(v[i]) if np.isfinite(v[i]) else np.nan,
                    "misfit_p05": float(v05[i]) if np.isfinite(v05[i]) else np.nan,
                    "misfit_p95": float(v95[i]) if np.isfinite(v95[i]) else np.nan,
                }
            )

    long_df = pd.DataFrame.from_records(records)

    rows: list[dict[str, object]] = []
    group_cols = ["property", "method", "stage", "fluid_state"]
    for (prop, method, stage, fluid), g in long_df.groupby(group_cols):
        stats_p50 = _agg_stats(_to_num(g["misfit_p50"]))
        stats_p05 = _agg_stats(_to_num(g["misfit_p05"]))
        stats_p95 = _agg_stats(_to_num(g["misfit_p95"]))
        rows.append(
            {
                "property": prop,
                "method": method,
                "stage": stage,
                "fluid_state": fluid,
                "n": stats_p50["n"],
                "misfit_p50_mean": stats_p50["mean"],
                "misfit_p50_std": stats_p50["std"],
                "misfit_p50_p05": stats_p50["p05"],
                "misfit_p50_p50": stats_p50["p50"],
                "misfit_p50_p95": stats_p50["p95"],
                "misfit_p05_p50": stats_p05["p50"],
                "misfit_p95_p50": stats_p95["p50"],
                "success_rate_p50_le_thr": _threshold_rate(_to_num(g["misfit_p50"]), float(args.thr_pct)),
                "success_rate_p95_le_thr": _threshold_rate(_to_num(g["misfit_p95"]), float(args.thr_pct)),
            }
        )

    summary = pd.DataFrame.from_records(rows)
    summary = summary.sort_values(["property", "stage", "fluid_state", "method"]).reset_index(drop=True)

    # Also: per-sample wide table (quick scan)
    wide = long_df.pivot_table(
        index=["lab_sample_id", "stage", "fluid_state"],
        columns=["property", "method"],
        values=["misfit_p50", "misfit_p05", "misfit_p95"],
        aggfunc="first",
    )
    wide.columns = ["_".join([a, b, c]).replace("misfit_", "") for (a, b, c) in wide.columns.to_flat_index()]
    wide = wide.reset_index().sort_values(["lab_sample_id", "stage", "fluid_state"]).reset_index(drop=True)

    out_xlsx = out_dir / "vp_vs_prediction_stats_from_tc_alpha.xlsx"
    with pd.ExcelWriter(out_xlsx) as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        wide.to_excel(xw, sheet_name="per_sample", index=False)

    out_csv = out_dir / "vp_vs_prediction_stats_from_tc_alpha_summary.csv"
    summary.to_csv(out_csv, index=False)

    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

