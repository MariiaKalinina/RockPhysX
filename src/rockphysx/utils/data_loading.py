from __future__ import annotations

from pathlib import Path

import pandas as pd


BRINE_COLS = ["TC 0,6", "TC 6", "TC 60", "TC 180"]
REQUIRED_TC_COLUMNS = ["Sample", "Porosity,%", "TC air", "TC oil"] + BRINE_COLS


def read_timan_pechora_tc_excel(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """
    Read Timan-Pechora thermal-conductivity measurements from Excel.

    Expected wide-format columns:
        Sample
        Porosity,%
        TC air
        TC oil
        TC 0,6
        TC 6
        TC 60
        TC 180

    Returns
    -------
    pd.DataFrame
        Long-format table with columns:
        sample, porosity, state, measured_tc
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)

    missing = [c for c in REQUIRED_TC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        sample = row["Sample"]
        porosity = float(row["Porosity,%"]) / 100.0

        state_map = {
            "dry": row["TC air"],
            "oil": row["TC oil"],
            "brine_0_6": row["TC 0,6"],
            "brine_6": row["TC 6"],
            "brine_60": row["TC 60"],
            "brine_180": row["TC 180"],
        }

        for state, value in state_map.items():
            if pd.notna(value):
                records.append(
                    {
                        "sample": str(sample),
                        "porosity": porosity,
                        "state": state,
                        "measured_tc": float(value),
                    }
                )

    return pd.DataFrame.from_records(records)


def write_posterior_summary_excel(
    posterior_df: pd.DataFrame,
    output_path: str | Path,
    *,
    summary_sheet: str = "posterior_summary",
) -> None:
    """
    Write posterior summary results to Excel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        posterior_df.to_excel(writer, sheet_name=summary_sheet, index=False)