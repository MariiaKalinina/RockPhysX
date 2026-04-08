from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_lambda(v: float) -> str:
    return f"{float(v):.2f}"


def _fmt_int(v: float) -> str:
    return str(int(np.rint(float(v))))


def _fmt_sigma(v: float) -> str:
    v = float(v)
    if not np.isfinite(v):
        return ""
    if v == 0.0:
        return "0"
    if 1e-3 <= abs(v) <= 1e3:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _fmt_float(v: float, nd: int = 2) -> str:
    v = float(v)
    return "" if not np.isfinite(v) else f"{v:.{nd}f}"


def _latex_escape(s: object) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _df_to_latex_booktabs(df: pd.DataFrame, *, caption: str, label: str) -> str:
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
                vals.append(_fmt_float(float(v), 4))
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a thesis-ready table of matrix (p50/p05/p95) and fixed fluid properties.")
    ap.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("test_new/hs_matrix_feasible_set_with_resistivity"),
        help="Directory containing feasible_minmax_summary.csv from the HS-feasible scan.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/matrix_fluid_params"))

    # Fixed pore-phase properties used elsewhere in the repo (defaults match plotting scripts)
    ap.add_argument("--lambda-air", type=float, default=0.026)
    ap.add_argument("--lambda-brine", type=float, default=0.60)
    ap.add_argument("--K-air-gpa", type=float, default=1e-4)
    ap.add_argument("--K-brine-gpa", type=float, default=2.2)
    ap.add_argument("--rho-air", type=float, default=1.2)
    ap.add_argument("--rho-brine", type=float, default=1030.0)

    # Brine resistivity (20 g/L NaCl) before/after MPCT (Ohm·m)
    ap.add_argument("--brine-res-before", type=float, default=0.32)
    ap.add_argument("--brine-res-after", type=float, default=0.26)

    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feasible_path = scan_dir / "feasible_minmax_summary.csv"
    if not feasible_path.exists():
        raise FileNotFoundError(f"Missing {feasible_path}")
    feasible = pd.read_csv(feasible_path)

    # Map parameter names to pretty labels and formatting
    param_specs = {
        "lambda_M": (r"$\lambda_M$", "W/(m·K)", _fmt_lambda),
        "vp_M_m_s": (r"$V_{P,M}$", "m/s", _fmt_int),
        "vs_M_m_s": (r"$V_{S,M}$", "m/s", _fmt_int),
        "sigmaM_S_m": (r"$\sigma_M$", "S/m", _fmt_sigma),
        "log10_sigmaM": (r"$\\log_{10}\\sigma_M$", "", lambda v: _fmt_float(v, 2)),
    }

    matrix_rows: list[dict[str, object]] = []
    for key, (label, unit, fmt) in param_specs.items():
        row = feasible[feasible["param"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        matrix_rows.append(
            {
                "Category": "Matrix (HS-feasible set)",
                "Parameter": label,
                "p05": fmt(r["p05"]),
                "p50": fmt(r["p50"]),
                "p95": fmt(r["p95"]),
                "Unit": unit,
            }
        )

    # Fixed fluids
    brine_sigma_before = 1.0 / max(float(args.brine_res_before), 1e-12)
    brine_sigma_after = 1.0 / max(float(args.brine_res_after), 1e-12)
    fluid_rows = [
        # Air
        {"Category": "Pore fluid (fixed)", "Parameter": r"$\lambda_{\mathrm{air}}$", "p05": "", "p50": _fmt_lambda(args.lambda_air), "p95": "", "Unit": "W/(m·K)"},
        {"Category": "Pore fluid (fixed)", "Parameter": r"$K_{\mathrm{air}}$", "p05": "", "p50": _fmt_float(args.K_air_gpa, 4), "p95": "", "Unit": "GPa"},
        {"Category": "Pore fluid (fixed)", "Parameter": r"$\rho_{\mathrm{air}}$", "p05": "", "p50": _fmt_float(args.rho_air, 1), "p95": "", "Unit": "kg/m$^3$"},
        # Brine
        {"Category": "Pore fluid (fixed)", "Parameter": r"$\lambda_{\mathrm{brine}}$", "p05": "", "p50": _fmt_lambda(args.lambda_brine), "p95": "", "Unit": "W/(m·K)"},
        {"Category": "Pore fluid (fixed)", "Parameter": r"$K_{\mathrm{brine}}$", "p05": "", "p50": _fmt_float(args.K_brine_gpa, 2), "p95": "", "Unit": "GPa"},
        {"Category": "Pore fluid (fixed)", "Parameter": r"$\rho_{\mathrm{brine}}$", "p05": "", "p50": _fmt_float(args.rho_brine, 0), "p95": "", "Unit": "kg/m$^3$"},
        # Brine resistivity / conductivity per stage
        {
            "Category": "Brine (20 g/L NaCl)",
            "Parameter": r"$R_f$ (before MPCT)",
            "p05": "",
            "p50": _fmt_float(args.brine_res_before, 2),
            "p95": "",
            "Unit": r"$\Omega\cdot$m",
        },
        {
            "Category": "Brine (20 g/L NaCl)",
            "Parameter": r"$\sigma_f$ (before MPCT)",
            "p05": "",
            "p50": _fmt_sigma(brine_sigma_before),
            "p95": "",
            "Unit": "S/m",
        },
        {
            "Category": "Brine (20 g/L NaCl)",
            "Parameter": r"$R_f$ (after MPCT)",
            "p05": "",
            "p50": _fmt_float(args.brine_res_after, 2),
            "p95": "",
            "Unit": r"$\Omega\cdot$m",
        },
        {
            "Category": "Brine (20 g/L NaCl)",
            "Parameter": r"$\sigma_f$ (after MPCT)",
            "p05": "",
            "p50": _fmt_sigma(brine_sigma_after),
            "p95": "",
            "Unit": "S/m",
        },
    ]

    out_df = pd.DataFrame(matrix_rows + fluid_rows, columns=["Category", "Parameter", "p05", "p50", "p95", "Unit"])

    out_csv = out_dir / "matrix_and_fluids_table.csv"
    out_tex = out_dir / "matrix_and_fluids_table.tex"
    out_df.to_csv(out_csv, index=False)

    caption = "Matrix parameters inferred from the HS-feasible set (p05/p50/p95) and fixed pore-fluid properties used in forward modelling."
    label = "tab:matrix_and_fluids"
    out_tex.write_text(_df_to_latex_booktabs(out_df, caption=caption, label=label), encoding="utf-8")

    print(f"Saved: {out_csv}")  # noqa: T201
    print(f"Saved: {out_tex}")  # noqa: T201
    print("\nTable preview:")  # noqa: T201
    print(out_df.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()

