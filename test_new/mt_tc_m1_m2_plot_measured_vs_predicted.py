from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = BASE_DIR / "mt_tc_m1_m2_results.xlsx"
DEFAULT_OUT_DIR = BASE_DIR / "mt_tc_m1_m2_plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot measured vs predicted thermal conductivity from mt_tc_m1_m2_results.xlsx.")
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Path to mt_tc_m1_m2_results.xlsx")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder for PNGs")
    p.add_argument("--log", action="store_true", help="Use log10 axes for the parity plots")
    return p.parse_args()


def _require_cols(df: pd.DataFrame, cols: list[str], sheet: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing required columns: {missing}")


def load_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not path.exists():
        raise FileNotFoundError(f"Missing results XLSX: {path} (resolved to {path.resolve()})")
    predictions = pd.read_excel(path, sheet_name="predictions")
    fit_quality = None
    try:
        fit_quality = pd.read_excel(path, sheet_name="fit_quality")
    except Exception:
        fit_quality = None
    return predictions, fit_quality


def model_metrics(fit_quality: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if fit_quality is None or fit_quality.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in fit_quality.iterrows():
        name = str(row.get("model", ""))
        if not name:
            continue
        out[name] = {
            "rmse_w_mk": float(row.get("rmse_w_mk", np.nan)),
            "mean_abs_w_mk": float(row.get("mean_abs_w_mk", np.nan)),
            "rmse_log": float(row.get("rmse_log", np.nan)),
            "n_obs": float(row.get("n_obs", np.nan)),
        }
    return out


def _category(stage: str, fluid: str) -> str:
    return f"{stage}/{fluid}"


def add_relative_error_bands(
    ax: plt.Axes,
    *,
    lo: float,
    hi: float,
    log_axes: bool,
) -> list[Patch]:
    """Add shaded relative-error bands around the 1:1 line."""
    if log_axes:
        x = np.logspace(np.log10(lo), np.log10(hi), 200)
    else:
        x = np.linspace(lo, hi, 200)

    bands = [
        (0.10, "#ff6b6b", 0.12, "±10%"),
        (0.05, "#ffa94d", 0.14, "±5%"),
        (0.025, "#51cf66", 0.16, "±2.5%"),
    ]

    patches: list[Patch] = []
    for eps, color, alpha, label in bands:
        ax.fill_between(
            x,
            (1.0 - eps) * x,
            (1.0 + eps) * x,
            color=color,
            alpha=alpha,
            linewidth=0.0,
            zorder=1,
        )
        patches.append(Patch(facecolor=color, edgecolor="none", alpha=alpha, label=label))
    return patches


def parity_plot(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    log_axes: bool,
    metrics: dict[str, dict[str, float]] | None = None,
) -> None:
    models = list(pd.unique(df["model"]))
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.6), squeeze=False)

    palette = {
        "before/dry": "#1f77b4",
        "before/wet": "#2ca02c",
        "after/dry": "#ff7f0e",
        "after/wet": "#d62728",
    }

    for i, model in enumerate(models):
        ax = axes[0][i]
        g = df[df["model"] == model].copy()
        g["cat"] = [_category(s, f) for s, f in zip(g["stage"].astype(str), g["fluid_state"].astype(str))]

        x = pd.to_numeric(g["tc_obs_w_mk"], errors="coerce").astype(float)
        y = pd.to_numeric(g["tc_pred_w_mk"], errors="coerce").astype(float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[valid]
        y = y[valid]
        cats = g.loc[valid, "cat"].astype(str)

        for cat in sorted(pd.unique(cats)):
            idx = cats == cat
            ax.scatter(
                x[idx],
                y[idx],
                s=42,
                alpha=0.85,
                label=cat,
                color=palette.get(cat, None),
                edgecolors="none",
            )

        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo = max(lo * 0.9, 1e-6)
            hi = hi * 1.1
        else:
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            lo -= pad
            hi += pad

        band_patches = add_relative_error_bands(ax, lo=lo, hi=hi, log_axes=log_axes)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="k", linewidth=1.2, label="1:1", zorder=4)
        for eps, ls in [(0.025, ":"), (0.05, "--"), (0.10, "-.")]:
            ax.plot([lo, hi], [(1.0 + eps) * lo, (1.0 + eps) * hi], linestyle=ls, color="0.35", alpha=0.85, linewidth=1.0, zorder=3)
            ax.plot([lo, hi], [(1.0 - eps) * lo, (1.0 - eps) * hi], linestyle=ls, color="0.35", alpha=0.85, linewidth=1.0, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_title(str(model))
        ax.set_xlabel("Measured TC, W/(m·K)")
        ax.set_ylabel("Predicted TC, W/(m·K)")
        ax.grid(True, which="both", alpha=0.25)

        if metrics and str(model) in metrics:
            m = metrics[str(model)]
            rmse = m.get("rmse_w_mk", np.nan)
            mae = m.get("mean_abs_w_mk", np.nan)
            ax.text(
                0.03,
                0.97,
                f"RMSE={rmse:.3g}\nMAE={mae:.3g}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

        # Keep two separate legends: (1) relative error bands + 1:1, (2) categories
        one_to_one = Line2D([0], [0], linestyle="--", color="k", linewidth=1.2, label="1:1")
        leg1 = ax.legend(handles=band_patches + [one_to_one], loc="upper left", frameon=True, fontsize=9)
        ax.add_artist(leg1)
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parity_by_fluid_plot(
    df: pd.DataFrame,
    *,
    out_path: Path,
    log_axes: bool,
) -> None:
    models = list(pd.unique(df["model"]))
    fluids = list(pd.unique(df["fluid_state"]))
    nrows = len(models)
    ncols = len(fluids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)

    for i, model in enumerate(models):
        for j, fluid in enumerate(fluids):
            ax = axes[i][j]
            g = df[(df["model"] == model) & (df["fluid_state"] == fluid)].copy()
            x = pd.to_numeric(g["tc_obs_w_mk"], errors="coerce").astype(float)
            y = pd.to_numeric(g["tc_pred_w_mk"], errors="coerce").astype(float)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x = x[valid]
            y = y[valid]
            stage = g.loc[valid, "stage"].astype(str)

            for st in sorted(pd.unique(stage)):
                idx = stage == st
                ax.scatter(x[idx], y[idx], s=42, alpha=0.85, label=str(st))

            if x.empty:
                ax.axis("off")
                continue

            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
                lo = max(lo * 0.9, 1e-6)
                hi = hi * 1.1
            else:
                pad = 0.05 * (hi - lo) if hi > lo else 0.1
                lo -= pad
                hi += pad
            add_relative_error_bands(ax, lo=lo, hi=hi, log_axes=log_axes)
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="k", linewidth=1.0, zorder=4)
            for eps, ls in [(0.025, ":"), (0.05, "--"), (0.10, "-.")]:
                ax.plot([lo, hi], [(1.0 + eps) * lo, (1.0 + eps) * hi], linestyle=ls, color="0.35", alpha=0.85, linewidth=0.9, zorder=3)
                ax.plot([lo, hi], [(1.0 - eps) * lo, (1.0 - eps) * hi], linestyle=ls, color="0.35", alpha=0.85, linewidth=0.9, zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{model} — {fluid}")
            ax.set_xlabel("Measured")
            ax.set_ylabel("Predicted")
            ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Measured vs predicted TC (split by fluid)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pred, fit_q = load_tables(args.xlsx)
    _require_cols(pred, ["model", "stage", "fluid_state", "tc_obs_w_mk", "tc_pred_w_mk"], "predictions")

    m = model_metrics(fit_q)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parity_plot(
        pred,
        out_path=args.out_dir / ("tc_measured_vs_predicted_log.png" if args.log else "tc_measured_vs_predicted.png"),
        title="Thermal conductivity: measured vs predicted",
        log_axes=args.log,
        metrics=m,
    )
    parity_by_fluid_plot(
        pred,
        out_path=args.out_dir / ("tc_measured_vs_predicted_by_fluid_log.png" if args.log else "tc_measured_vs_predicted_by_fluid.png"),
        log_axes=args.log,
    )
    print(f"Saved plots to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
