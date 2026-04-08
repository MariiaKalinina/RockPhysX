from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_XLSX = BASE_DIR / "data_restructured_for_MT_v2.xlsx"
DEFAULT_RESULTS_XLSX = BASE_DIR / "mt_tc_m1_m2_results.xlsx"
DEFAULT_OUT_DIR = BASE_DIR / "mt_tc_m1_m2_plots" / "zones"
DEFAULT_POSITIONS_SHEET = "raw_original"


@dataclass(frozen=True)
class Zone:
    zone_name: str
    temperature_c: float
    duration_h: float | None
    start_mm: float
    end_mm: float


# The current dataset's TG tube zone table (as provided by the user) includes Zone 8.
# Some versions of `experiment_metadata` may stop at Zone 7; in that case we append Zone 8
# so samples beyond 612 mm are not incorrectly assigned to Zone 7.
DEFAULT_ZONE8 = Zone(zone_name="Зона 8", temperature_c=306.705, duration_h=0.134, start_mm=612.0, end_mm=688.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare M1/M2 inversion results vs heating temperature zones.")
    p.add_argument("--data-xlsx", type=Path, default=DEFAULT_DATA_XLSX, help="Path to data_restructured_for_MT_v2.xlsx")
    p.add_argument("--results-xlsx", type=Path, default=DEFAULT_RESULTS_XLSX, help="Path to mt_tc_m1_m2_results.xlsx")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder for plots/CSVs")
    p.add_argument("--positions-sheet", default=DEFAULT_POSITIONS_SHEET, help="Sheet used for tube positions (default: raw_original)")
    p.add_argument("--include-ign", action="store_true", help="Include the 'Ign' zone in plots/tables")
    return p.parse_args()


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
                        duration_h=(None if current.get("duration_h") is None else float(current["duration_h"])),  # type: ignore[index]
                        start_mm=float(current.get("start_mm", 0.0)),
                        end_mm=float(current.get("end_mm", np.nan)),
                    )
                )
            current = {"zone_name": str(value), "start_mm": float(zones[-1].end_mm) if zones else 0.0, "end_mm": None}
        elif param == "temperature":
            current["temperature_c"] = float(value)
        elif param == "duration":
            current["duration_h"] = float(value)
        elif param == "distance":
            current["end_mm"] = float(value)

    if current:
        zones.append(
            Zone(
                zone_name=str(current["zone_name"]),
                temperature_c=float(current.get("temperature_c", np.nan)),
                duration_h=(None if current.get("duration_h") is None else float(current["duration_h"])),  # type: ignore[index]
                start_mm=float(current.get("start_mm", 0.0)),
                end_mm=float(current.get("end_mm", np.nan)),
            )
        )

    # sanity: sort by end_mm
    zones = sorted(zones, key=lambda z: z.end_mm)

    # Append Zone 8 if missing and the metadata ends at Zone 7.
    # This matches the zone table used in the dissertation figures.
    if zones and all(z.zone_name != "Зона 8" for z in zones):
        last = zones[-1]
        if last.zone_name == "Зона 7" and last.end_mm <= DEFAULT_ZONE8.start_mm + 1e-6:
            zones.append(DEFAULT_ZONE8)
    return zones


def assign_zone(position_mm: float, zones: list[Zone]) -> Zone | None:
    if not np.isfinite(position_mm):
        return None
    if not zones:
        return None
    # Clamp out-of-range positions to nearest zone to avoid losing samples that
    # lie slightly outside the documented distance range.
    if position_mm < zones[0].start_mm:
        return zones[0]
    if position_mm > zones[-1].end_mm:
        return zones[-1]
    for z in zones:
        if z.start_mm <= position_mm <= z.end_mm:
            return z
    return None


def load_sample_positions(data_xlsx: Path, sheet_name: str) -> pd.DataFrame:
    """Load per-sample tube position (mm).

    User request: prefer `raw_original` where positions were entered originally.
    Falls back to `sample_stage` if needed.
    """
    df = pd.read_excel(data_xlsx, sheet_name=sheet_name)

    # Column candidates (Russian + English)
    sample_col_candidates = ["Лаб. номер образца", "lab_sample_id", "sample_id"]
    pos_col_candidates = ["Расположение в ТГ, мм", "tg_position_mm"]

    sample_col = next((c for c in sample_col_candidates if c in df.columns), None)
    pos_col = next((c for c in pos_col_candidates if c in df.columns), None)

    if sample_col is None or pos_col is None:
        # try fallback sheet
        if sheet_name != "sample_stage":
            return load_sample_positions(data_xlsx, "sample_stage")
        raise ValueError(
            f"Cannot find required columns in sheet '{sheet_name}'. "
            f"Need one of {sample_col_candidates} and one of {pos_col_candidates}."
        )

    out = df[[sample_col, pos_col]].copy()
    out = out.rename(columns={sample_col: "sample_id", pos_col: "tg_position_mm"})
    out["sample_id"] = pd.to_numeric(out["sample_id"], errors="coerce")
    out["tg_position_mm"] = pd.to_numeric(out["tg_position_mm"], errors="coerce")
    out = out.dropna(subset=["sample_id"]).copy()

    # If there are duplicates (raw_original has one row per sample), keep first non-null position
    out = (
        out.sort_values(["sample_id"])
        .groupby("sample_id", as_index=False)
        .agg(tg_position_mm=("tg_position_mm", "first"))
    )
    return out


def beta_mean_aspect_ratio(a: float, b: float, *, n: int = 800) -> float:
    """E[AR] where u~Beta(a,b) and AR=10^(-4+4u)."""
    # integrate over z = log10(AR) in [-4,0], u=(z+4)/4
    z = np.linspace(-4.0, 0.0, n)
    u = (z + 4.0) / 4.0
    # beta pdf in u, transformed to density in z
    from scipy.stats import beta as beta_dist

    pz = 0.25 * beta_dist.pdf(u, a, b)
    dz = float(z[1] - z[0]) if n > 1 else 1.0
    ar = 10.0 ** z
    denom = float(np.sum(pz) * dz)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.sum(ar * pz) * dz / denom)


def build_param_table(
    results_xlsx: Path,
    data_xlsx: Path,
    zones: list[Zone],
    *,
    include_ign: bool,
    positions_sheet: str,
) -> pd.DataFrame:
    m1 = pd.read_excel(results_xlsx, sheet_name="M1_fits")
    m2 = pd.read_excel(results_xlsx, sheet_name="M2_fits")

    m1["sample_id"] = pd.to_numeric(m1["sample_id"], errors="coerce")
    m2["sample_id"] = pd.to_numeric(m2["sample_id"], errors="coerce")
    pos = load_sample_positions(data_xlsx, positions_sheet)

    df = m1.merge(m2, on="sample_id", suffixes=("_m1", "_m2")).merge(pos, on="sample_id", how="left")

    zone_names: list[str] = []
    zone_t: list[float] = []
    for mm in df["tg_position_mm"].to_numpy(dtype=float):
        z = assign_zone(float(mm), zones)
        zone_names.append(z.zone_name if z else "unknown")
        zone_t.append(float(z.temperature_c) if z else float("nan"))
    df["zone_name"] = zone_names
    df["zone_temp_c"] = zone_t

    # compute mean AR for M2 before/after
    a_before = df["m_before"] * df["kappa_before"]
    b_before = (1.0 - df["m_before"]) * df["kappa_before"]
    a_after = df["m_after"] * df["kappa_after"]
    b_after = (1.0 - df["m_after"]) * df["kappa_after"]
    df["m2_mean_ar_before"] = [beta_mean_aspect_ratio(float(a), float(b)) for a, b in zip(a_before, b_before)]
    df["m2_mean_ar_after"] = [beta_mean_aspect_ratio(float(a), float(b)) for a, b in zip(a_after, b_after)]
    df["m2_mean_ar_ratio"] = df["m2_mean_ar_after"] / df["m2_mean_ar_before"]

    # M1 already has ar_before/ar_after
    df["m1_ar_ratio"] = df["ar_ratio_after_before"]
    df["m1_ar_pct_change"] = 100.0 * (df["m1_ar_ratio"] - 1.0)
    df["m2_ar_pct_change"] = 100.0 * (df["m2_mean_ar_ratio"] - 1.0)

    if not include_ign:
        df = df[df["zone_name"].astype(str) != "Ign"].copy()
    return df


def build_error_table(
    results_xlsx: Path,
    data_xlsx: Path,
    zones: list[Zone],
    *,
    include_ign: bool,
    positions_sheet: str,
) -> pd.DataFrame:
    pred = pd.read_excel(results_xlsx, sheet_name="predictions")
    ds = pd.read_excel(results_xlsx, sheet_name="tc_dataset")

    for col in ["sample_id", "phi_frac"]:
        if col in pred.columns:
            pred[col] = pd.to_numeric(pred[col], errors="coerce")
        if col in ds.columns:
            ds[col] = pd.to_numeric(ds[col], errors="coerce")

    # Prefer tube positions from the user-requested sheet (raw_original).
    pos = load_sample_positions(data_xlsx, positions_sheet)
    merged = pred.merge(pos, on="sample_id", how="left")
    zone_names: list[str] = []
    zone_t: list[float] = []
    for mm in merged["tg_position_mm"].to_numpy(dtype=float):
        z = assign_zone(float(mm), zones)
        zone_names.append(z.zone_name if z else "unknown")
        zone_t.append(float(z.temperature_c) if z else float("nan"))
    merged["zone_name"] = zone_names
    merged["zone_temp_c"] = zone_t

    merged["tc_obs_w_mk"] = pd.to_numeric(merged["tc_obs_w_mk"], errors="coerce")
    merged["tc_pred_w_mk"] = pd.to_numeric(merged["tc_pred_w_mk"], errors="coerce")
    merged["residual_w_mk"] = pd.to_numeric(merged["residual_w_mk"], errors="coerce")
    merged["rel_error"] = merged["residual_w_mk"] / merged["tc_obs_w_mk"]
    merged["abs_rel_error"] = merged["rel_error"].abs()

    if not include_ign:
        merged = merged[merged["zone_name"].astype(str) != "Ign"].copy()
    return merged


def plot_ratio_vs_temp(df: pd.DataFrame, out_dir: Path) -> None:
    tmp = df.dropna(subset=["zone_temp_c", "m1_ar_pct_change", "m2_ar_pct_change"]).copy()
    if tmp.empty:
        return
    plt.figure(figsize=(7.6, 5.8))
    plt.scatter(tmp["zone_temp_c"], tmp["m1_ar_pct_change"], label="M1: 100·(α_after/α_before − 1)", s=55, alpha=0.9)
    plt.scatter(tmp["zone_temp_c"], tmp["m2_ar_pct_change"], label="M2: 100·(mean α_after/mean α_before − 1)", s=55, alpha=0.9)
    for _, r in tmp.iterrows():
        plt.text(float(r["zone_temp_c"]) + 0.3, float(r["m2_ar_pct_change"]), f'{float(r["sample_id"]):g}', fontsize=8, alpha=0.7)
    plt.axhline(0.0, linestyle="--", color="k", linewidth=1.0)
    plt.xlabel("Zone temperature (°C)")
    plt.ylabel(r"Aspect-ratio change  $100\cdot(\alpha_{\mathrm{after}}/\alpha_{\mathrm{before}} - 1)$  (%)")
    plt.title("Aspect-ratio change vs heating temperature")
    plt.grid(alpha=0.25)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "ar_ratio_vs_temperature.png", dpi=300)
    plt.close()


def plot_ratio_by_zone(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
        }
    )

    tmp = df.copy()
    zone_meta = tmp[["zone_name", "zone_temp_c"]].drop_duplicates().copy()

    def _zone_index(name: str) -> float:
        # "Зона 1".."Зона 8" -> 1..8, unknown -> inf
        parts = str(name).strip().split()
        if len(parts) >= 2 and parts[0].lower().startswith("зона"):
            try:
                return float(parts[1])
            except Exception:
                return float("inf")
        return float("inf")

    zone_meta["zone_idx"] = zone_meta["zone_name"].map(_zone_index)
    zone_meta = zone_meta.sort_values(["zone_idx", "zone_name"]).reset_index(drop=True)
    order = zone_meta["zone_name"].tolist()
    if not order:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 5.9))
    data_m1 = [tmp.loc[tmp["zone_name"] == z, "m1_ar_pct_change"].dropna().to_numpy(dtype=float) for z in order]
    data_m2 = [tmp.loc[tmp["zone_name"] == z, "m2_ar_pct_change"].dropna().to_numpy(dtype=float) for z in order]

    positions = np.arange(len(order))

    # Interpretation bands for percent change (colored intervals)
    all_vals = np.concatenate(
        [np.concatenate([d for d in data_m1 if d.size]), np.concatenate([d for d in data_m2 if d.size])]
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        y_lo = float(np.nanmin(all_vals)) - 3.0
        y_hi = float(np.nanmax(all_vals)) + 3.0
    else:
        y_lo, y_hi = -30.0, 30.0

    bands = [
        (-1e9, -20.0, "#ffd6d6", r"$< -20\%$: strong crack-like shift"),
        (-20.0, -5.0, "#ffe8cc", r"$-20\%$ to $-5\%$: moderate crack-like shift"),
        (-5.0, 5.0, "#d3f9d8", r"$-5\%$ to $+5\%$: ~no change"),
    ]
    band_handles = []
    for a, b, color, label in bands:
        ax.axhspan(a, b, color=color, alpha=0.55, zorder=0)
        band_handles.append(plt.Line2D([0], [0], color=color, linewidth=10, alpha=0.7, label=label))
    ax.boxplot(data_m1, positions=positions - 0.15, widths=0.25, patch_artist=True, boxprops=dict(facecolor="#74c0fc", alpha=0.7))
    ax.boxplot(data_m2, positions=positions + 0.15, widths=0.25, patch_artist=True, boxprops=dict(facecolor="#ff8787", alpha=0.6))
    ax.axhline(0.0, linestyle="--", color="k", linewidth=1.0)
    ax.set_xticks(positions)
    def _zone_label(name: str) -> str:
        s = str(name).strip()
        if s.lower().startswith("зона"):
            parts = s.split()
            return f"Zone {parts[1]}" if len(parts) >= 2 else "Zone"
        if s.lower() == "ign":
            return "Ign"
        return s

    ax.set_xticklabels([_zone_label(z) for z in order], rotation=0)
    ax.set_ylabel(r"Aspect-ratio change  $100\cdot(\alpha_{\mathrm{after}}/\alpha_{\mathrm{before}} - 1)$  (%)")
    ax.set_title("Aspect-ratio change by heating zone (with temperature profile)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(y_lo, y_hi)

    # Right axis: temperature per zone as scatter + line
    ax2 = ax.twinx()
    temps = zone_meta.set_index("zone_name").reindex(order)["zone_temp_c"].to_numpy(dtype=float)
    ax2.plot(positions, temps, color="0.25", linewidth=1.6, alpha=0.9, zorder=2)
    ax2.scatter(positions, temps, color="0.15", s=36, zorder=3, label="Zone temperature")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_ylim(float(np.nanmin(temps)) - 10.0, float(np.nanmax(temps)) + 10.0)

    legend_handles = [
        plt.Line2D([0], [0], color="#74c0fc", linewidth=8, alpha=0.7, label="M1: % change"),
        plt.Line2D([0], [0], color="#ff8787", linewidth=8, alpha=0.6, label="M2: % change"),
        plt.Line2D([0], [0], color="0.25", linewidth=2, alpha=0.9, label="Zone temperature profile"),
        *band_handles,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "ar_ratio_by_zone_boxplot.png", dpi=300)
    plt.close(fig)


def plot_error_by_zone(err: pd.DataFrame, out_dir: Path) -> None:
    tmp = err.dropna(subset=["zone_name", "abs_rel_error", "model"]).copy()
    if tmp.empty:
        return
    order = (
        tmp[["zone_name", "zone_temp_c"]]
        .drop_duplicates()
        .sort_values(["zone_temp_c", "zone_name"])
        .loc[:, "zone_name"]
        .tolist()
    )
    models = sorted(pd.unique(tmp["model"].astype(str)))
    fig, axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 5.2), squeeze=False)
    for i, model in enumerate(models):
        ax = axes[0][i]
        g = tmp[tmp["model"].astype(str) == model]
        data = [g.loc[g["zone_name"] == z, "abs_rel_error"].dropna().to_numpy(dtype=float) for z in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title(model)
        ax.set_ylabel("|relative error|")
        ax.set_ylim(0.0, max(0.15, float(np.nanmax(g["abs_rel_error"])) * 1.1))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("TC prediction error by heating zone", fontsize=13, weight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "tc_abs_rel_error_by_zone.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zones = load_zones_from_metadata(args.data_xlsx)
    params = build_param_table(
        args.results_xlsx,
        args.data_xlsx,
        zones,
        include_ign=args.include_ign,
        positions_sheet=args.positions_sheet,
    )
    errors = build_error_table(
        args.results_xlsx,
        args.data_xlsx,
        zones,
        include_ign=args.include_ign,
        positions_sheet=args.positions_sheet,
    )

    params.to_csv(args.out_dir / "zone_param_table.csv", index=False)
    errors.to_csv(args.out_dir / "zone_error_table.csv", index=False)

    plot_ratio_vs_temp(params, args.out_dir)
    plot_ratio_by_zone(params, args.out_dir)
    plot_error_by_zone(errors, args.out_dir)

    print(f"Saved zone analysis to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
