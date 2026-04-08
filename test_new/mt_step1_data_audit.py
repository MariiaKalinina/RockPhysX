from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / "data_restructured_for_MT_v2.xlsx"
OUT_DIR = BASE_DIR / "mt_step1_outputs"
OUT_DIR.mkdir(exist_ok=True)

STAGE_ORDER = ["before", "after"]
FLUID_ORDER = ["dry", "wet"]


def _to_float_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def statcell(series: pd.Series) -> str:
    s = _to_float_series(series).dropna()
    if s.empty:
        return "--"
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    vmin = float(s.min())
    vmax = float(s.max())
    if np.isnan(std):
        return f"{mean:.3g}\n{vmin:.3g}...{vmax:.3g}"
    return f"{mean:.3g} ({std:.2g})\n{vmin:.3g}...{vmax:.3g}"


def _print_multiline_table(rows: list[list[str]], *, header: list[str]) -> None:
    """Print a small table to terminal with multiline cells."""
    table = [header] + rows
    # split cells into lines
    split = [[[line for line in str(cell).splitlines()] for cell in r] for r in table]
    ncols = len(header)
    widths = [0] * ncols
    for r in split:
        for j in range(ncols):
            widths[j] = max(widths[j], max((len(line) for line in r[j]), default=0))

    def sep(ch: str = "-") -> str:
        return "+".join([ch * (w + 2) for w in widths])

    print(sep("="))
    # header
    header_lines = max(len(c) for c in split[0])
    for i in range(header_lines):
        parts = []
        for j in range(ncols):
            line = split[0][j][i] if i < len(split[0][j]) else ""
            parts.append(f" {line:<{widths[j]}} ")
        print("|".join(parts))
    print(sep("="))
    # body
    for r in split[1:]:
        h = max(len(c) for c in r)
        for i in range(h):
            parts = []
            for j in range(ncols):
                line = r[j][i] if i < len(r[j]) else ""
                parts.append(f" {line:<{widths[j]}} ")
            print("|".join(parts))
        print(sep("-"))


def make_summary_table(df: pd.DataFrame) -> list[list[str]]:
    props = [
        ("Effective porosity, %", r"$\\phi$", "phi_pct", lambda s: s),
        ("Permeability, mD", r"$k$", "permeability_md_modeled", lambda s: s),
        ("Bulk density, g/cm3", r"$\\rho$", "bulk_density_g_cm3", lambda s: s),
        ("Thermal conductivity, W/(m·K)", r"$\\lambda$", "tc_w_mk", lambda s: s),
        ("Elastic wave velocity, km/s (Vp)", r"$V_P$", "vp_m_s", lambda s: _to_float_series(s) / 1000.0),
        ("Elastic wave velocity, km/s (Vs)", r"$V_S$", "vs_m_s", lambda s: _to_float_series(s) / 1000.0),
        ("Electrical resistivity, Ohm·m", r"$R$", "resistivity_ohm_m", lambda s: s),
    ]

    rows: list[list[str]] = []
    for name, symbol, col, transform in props:
        if col not in df.columns:
            continue
        row = [name, symbol]
        for stage in STAGE_ORDER:
            for fluid in FLUID_ORDER:
                g = df[(df["stage"] == stage) & (df["fluid_state"] == fluid)]
                row.append(statcell(transform(g[col])) if not g.empty else "--")
        rows.append(row)
    return rows


def bootstrap_ci_mean(values: np.ndarray, *, n_boot: int = 5000, ci: float = 0.94, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def make_effect_table(wide: pd.DataFrame) -> list[list[str]]:
    # wide has columns like <prop>_before, <prop>_after, delta_<prop>, rel_change_<prop>
    definitions = [
        ("phi_pct", "Δphi, %", lambda s: s),
        ("tc_w_mk", "Δλ, W/(m·K)", lambda s: s),
        ("vp_m_s", "ΔVp, km/s", lambda s: _to_float_series(s) / 1000.0),
        ("vs_m_s", "ΔVs, km/s", lambda s: _to_float_series(s) / 1000.0),
        ("resistivity_ohm_m", "ΔR, Ohm·m", lambda s: s),
    ]

    rows: list[list[str]] = []
    for fluid in FLUID_ORDER:
        g = wide[wide["fluid_state"] == fluid].copy()
        if g.empty:
            continue
        for prop, label, transform in definitions:
            before = f"{prop}_before"
            after = f"{prop}_after"
            if before not in g.columns or after not in g.columns:
                continue
            b = transform(g[before])
            a = transform(g[after])
            delta = _to_float_series(a) - _to_float_series(b)
            delta = delta.dropna()
            if delta.empty:
                continue
            n = int(delta.size)
            mean = float(delta.mean())
            std = float(delta.std(ddof=1)) if n > 1 else float("nan")
            med = float(delta.median())
            p_pos = float(np.mean(delta > 0.0))
            lo, hi = bootstrap_ci_mean(delta.to_numpy(), ci=0.94, seed=42)
            if np.isfinite(lo) and lo > 0:
                concl = "systematic increase"
            elif np.isfinite(hi) and hi < 0:
                concl = "systematic decrease"
            else:
                concl = "no clear effect"
            rows.append(
                [
                    str(fluid),
                    label,
                    str(n),
                    f"{mean:.3g} ({std:.2g})" if np.isfinite(std) else f"{mean:.3g}",
                    f"{med:.3g}",
                    f"{100*p_pos:.1f}%",
                    f"[{lo:.3g}, {hi:.3g}]",
                    concl,
                ]
            )
    return rows


def load_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="measurements_long")
    required = {"lab_sample_id", "stage", "fluid_state"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input sheet 'measurements_long' is missing required columns: {missing}")
    if "phi_frac" in df.columns and "phi_pct" in df.columns:
        df["phi_frac"] = df["phi_frac"].where(df["phi_frac"].notna(), df["phi_pct"] / 100.0)
    for col in ["stage", "fluid_state", "lab_sample_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def summarize_measurements(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n_rows = int(len(df))
    n_samples = int(df["lab_sample_id"].nunique(dropna=False))
    n_stages = int(df["stage"].nunique(dropna=False))
    n_fluids = int(df["fluid_state"].nunique(dropna=False))

    overview = pd.DataFrame(
        [
            {"metric": "n_rows", "value": n_rows},
            {"metric": "n_samples", "value": n_samples},
            {"metric": "n_stages", "value": n_stages},
            {"metric": "n_fluids", "value": n_fluids},
        ]
    )

    counts_stage_fluid = (
        df.groupby(["stage", "fluid_state"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["stage", "fluid_state"])
    )

    numeric_cols = [c for c in ["phi_pct", "tc_w_mk", "vp_m_s", "vs_m_s", "resistivity_ohm_m"] if c in df.columns]
    overall_describe = df[numeric_cols].describe(include="all").T.reset_index().rename(columns={"index": "column"})

    by_group_rows: list[pd.DataFrame] = []
    for (stage, fluid), g in df.groupby(["stage", "fluid_state"], dropna=False):
        if g.empty:
            continue
        desc = g[numeric_cols].describe(include="all").T
        desc.insert(0, "fluid_state", str(fluid))
        desc.insert(0, "stage", str(stage))
        by_group_rows.append(desc.reset_index().rename(columns={"index": "column"}))
    group_describe = pd.concat(by_group_rows, ignore_index=True) if by_group_rows else pd.DataFrame()

    return {
        "overview": overview,
        "counts_stage_fluid": counts_stage_fluid,
        "describe_overall": overall_describe,
        "describe_by_stage_fluid": group_describe,
    }


def audit_measurements(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    key_cols = ["lab_sample_id", "stage", "fluid_state"]
    missing = (
        df.isna().sum().rename("missing_count").to_frame()
        .assign(missing_share=lambda x: x["missing_count"] / len(df))
        .reset_index().rename(columns={"index": "column"})
        .sort_values(["missing_count", "column"], ascending=[False, True])
    )

    duplicates = df[df.duplicated(subset=key_cols, keep=False)].copy()

    expected_pairs = {("before", "dry"), ("before", "wet"), ("after", "dry"), ("after", "wet")}
    pair_rows = []
    for sample_id, g in df.groupby("lab_sample_id"):
        observed = set(zip(g["stage"], g["fluid_state"]))
        pair_rows.append(
            {
                "lab_sample_id": sample_id,
                "pairing_complete": observed == expected_pairs,
                "has_before_dry": ("before", "dry") in observed,
                "has_before_wet": ("before", "wet") in observed,
                "has_after_dry": ("after", "dry") in observed,
                "has_after_wet": ("after", "wet") in observed,
            }
        )
    pairing = pd.DataFrame(pair_rows).sort_values("lab_sample_id")

    wide = df.pivot_table(
        index=["lab_sample_id", "fluid_state"],
        columns="stage",
        values=["phi_pct", "tc_w_mk", "vp_m_s", "vs_m_s", "resistivity_ohm_m"],
        aggfunc="first",
    )
    wide.columns = [f"{prop}_{stage}" for prop, stage in wide.columns]
    wide = wide.reset_index()

    for prop in ["phi_pct", "tc_w_mk", "vp_m_s", "vs_m_s", "resistivity_ohm_m"]:
        b = f"{prop}_before"
        a = f"{prop}_after"
        if b in wide.columns and a in wide.columns:
            wide[f"delta_{prop}"] = wide[a] - wide[b]
            with np.errstate(divide="ignore", invalid="ignore"):
                wide[f"rel_change_{prop}"] = (wide[a] - wide[b]) / wide[b]

    # Delta summaries (overall and per fluid_state)
    delta_cols = [c for c in wide.columns if c.startswith("delta_") or c.startswith("rel_change_")]
    delta_overall = wide[delta_cols].describe(include="all").T.reset_index().rename(columns={"index": "column"})
    delta_by_fluid_rows: list[pd.DataFrame] = []
    for fluid, g in wide.groupby("fluid_state", dropna=False):
        desc = g[delta_cols].describe(include="all").T
        desc.insert(0, "fluid_state", str(fluid))
        delta_by_fluid_rows.append(desc.reset_index().rename(columns={"index": "column"}))
    delta_by_fluid = pd.concat(delta_by_fluid_rows, ignore_index=True) if delta_by_fluid_rows else pd.DataFrame()

    return {
        "missingness": missing,
        "duplicates": duplicates,
        "pairing": pairing,
        "wide_deltas": wide,
        "delta_describe_overall": delta_overall,
        "delta_describe_by_fluid": delta_by_fluid,
    }


def make_plots(wide: pd.DataFrame, out_dir: Path) -> None:
    props = ["tc_w_mk", "vp_m_s", "vs_m_s", "resistivity_ohm_m"]
    for prop in props:
        before, after = f"{prop}_before", f"{prop}_after"
        if before not in wide.columns or after not in wide.columns:
            continue
        tmp = wide[[before, after, "fluid_state"]].dropna()
        if tmp.empty:
            continue

        plt.figure(figsize=(6.5, 5.5))
        for fluid in sorted(tmp["fluid_state"].dropna().unique()):
            g = tmp[tmp["fluid_state"] == fluid]
            plt.scatter(g[before], g[after], label=str(fluid), alpha=0.85)
        lo = float(np.nanmin([tmp[before].min(), tmp[after].min()]))
        hi = float(np.nanmax([tmp[before].max(), tmp[after].max()]))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel(f"Before: {prop}")
        plt.ylabel(f"After: {prop}")
        plt.title(f"Before vs After — {prop}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"before_after_{prop}.png", dpi=180)
        plt.close()

    # Delta distributions
    for prop in props:
        delta = f"delta_{prop}"
        if delta not in wide.columns:
            continue
        tmp = wide[[delta, "fluid_state"]].dropna()
        if tmp.empty:
            continue
        plt.figure(figsize=(7.0, 4.8))
        bins = 20
        for fluid in sorted(tmp["fluid_state"].dropna().unique()):
            g = tmp[tmp["fluid_state"] == fluid][delta].dropna()
            if g.empty:
                continue
            plt.hist(g, bins=bins, alpha=0.55, label=str(fluid), density=True)
        plt.axvline(0.0, linestyle="--", linewidth=1.2, color="k")
        plt.xlabel(f"Δ {prop} (after − before)")
        plt.ylabel("Density")
        plt.title(f"Delta distribution — {prop}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"delta_hist_{prop}.png", dpi=180)
        plt.close()

    # Bland–Altman plots
    for prop in props:
        before, after = f"{prop}_before", f"{prop}_after"
        if before not in wide.columns or after not in wide.columns:
            continue
        tmp = wide[[before, after, "fluid_state"]].dropna()
        if tmp.empty:
            continue
        mean = 0.5 * (tmp[before] + tmp[after])
        diff = tmp[after] - tmp[before]
        plt.figure(figsize=(7.0, 5.0))
        for fluid in sorted(tmp["fluid_state"].dropna().unique()):
            idx = tmp["fluid_state"] == fluid
            plt.scatter(mean[idx], diff[idx], label=str(fluid), alpha=0.85)
        bias = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
        plt.axhline(bias, color="k", linewidth=1.2, label="bias")
        if np.isfinite(sd):
            plt.axhline(bias + 1.96 * sd, color="k", linestyle="--", linewidth=1.0)
            plt.axhline(bias - 1.96 * sd, color="k", linestyle="--", linewidth=1.0)
        plt.axhline(0.0, color="k", linestyle=":", linewidth=1.0)
        plt.xlabel(f"Mean ({prop}) = (before+after)/2")
        plt.ylabel(f"Difference ({prop}) = after−before")
        plt.title(f"Bland–Altman — {prop}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"bland_altman_{prop}.png", dpi=180)
        plt.close()

    # Spaghetti / slopegraph per sample
    for prop in props:
        before, after = f"{prop}_before", f"{prop}_after"
        if before not in wide.columns or after not in wide.columns:
            continue
        tmp = wide[["lab_sample_id", "fluid_state", before, after]].dropna()
        if tmp.empty:
            continue
        plt.figure(figsize=(7.0, 5.0))
        x = np.array([0, 1])
        for fluid in sorted(tmp["fluid_state"].dropna().unique()):
            g = tmp[tmp["fluid_state"] == fluid]
            for _, row in g.iterrows():
                plt.plot(x, [row[before], row[after]], alpha=0.35)
        plt.xticks([0, 1], ["before", "after"])
        plt.ylabel(prop)
        plt.title(f"Per-sample change (slopegraph) — {prop}")
        plt.tight_layout()
        plt.savefig(out_dir / f"slopegraph_{prop}.png", dpi=180)
        plt.close()

    # Delta–delta correlations (quick scan)
    delta_cols = [c for c in wide.columns if c.startswith("delta_")]
    if delta_cols:
        corr = wide[delta_cols].corr(numeric_only=True)
        plt.figure(figsize=(6.8, 5.8))
        im = plt.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="corr")
        labels = [c.replace("delta_", "Δ") for c in delta_cols]
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Correlation of deltas")
        plt.tight_layout()
        plt.savefig(out_dir / "delta_correlation_heatmap.png", dpi=200)
        plt.close()


def main() -> None:
    if not INPUT_XLSX.exists():
        candidates = sorted(Path(".").glob("**/*.xlsx"))
        hint = "\n".join(f"- {p}" for p in candidates[:20])
        raise FileNotFoundError(
            f"Missing input XLSX: {INPUT_XLSX} (resolved to {INPUT_XLSX.resolve()}).\n"
            f"First XLSX candidates under repo root:\n{hint}"
        )
    df = load_measurements(INPUT_XLSX)
    summary = summarize_measurements(df)
    audit = audit_measurements(df)
    report_path = OUT_DIR / "mt_step1_audit_report.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        for name, table in summary.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
        for name, table in audit.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    make_plots(audit["wide_deltas"], OUT_DIR)
    print(f"Saved report to: {report_path}")
    print(f"Saved plots to: {OUT_DIR}")

    # Terminal tables for quick write-up
    header = [
        "Physical property",
        "Symbol",
        "Before / dry",
        "Before / wet",
        "After / dry",
        "After / wet",
    ]
    print("\nSUMMARY (mean (std) on first line; min...max on second line)")
    _print_multiline_table(make_summary_table(df), header=header)

    effect_header = ["Fluid", "Property", "n", "mean( sd )", "median", "%Δ>0", "94% CI mean", "Conclusion"]
    print("\nBEFORE→AFTER EFFECT (paired deltas per sample)")
    _print_multiline_table(make_effect_table(audit["wide_deltas"]), header=effect_header)


if __name__ == "__main__":
    main()
