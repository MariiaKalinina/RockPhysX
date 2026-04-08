from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _rel_error_bands(ax, x_min: float, x_max: float, bands: list[float]) -> None:
    x = np.linspace(x_min, x_max, 200)
    # Neutral greys so bands don't compete with point colors.
    # Make ±5% clearly visible by using stronger contrast and thin boundary lines.
    colors = {
        0.10: ("#d8d8d8", 0.25),
        0.05: ("#bdbdbd", 0.35),
        0.025: ("#a0a0a0", 0.45),
    }
    for p in sorted(bands, reverse=True):
        c, a = colors.get(p, ("#dddddd", 0.25))
        ax.fill_between(x, x * (1.0 - p), x * (1.0 + p), color=c, alpha=a, linewidth=0)
        if abs(p - 0.05) < 1e-12:
            ax.plot(x, x * (1.0 - p), color="0.35", lw=0.9, ls="--", alpha=0.8)
            ax.plot(x, x * (1.0 + p), color="0.35", lw=0.9, ls="--", alpha=0.8)


def _property_label(prop: str) -> str:
    return {
        "tc_w_mk": "Thermal conductivity, λ (W/(m·K))",
        "vp_m_s": "P-wave velocity, Vp (m/s)",
        "vs_m_s": "S-wave velocity, Vs (m/s)",
    }.get(prop, prop)


def main() -> None:
    p = argparse.ArgumentParser(description="Parity plots for strict MT joint inversion (measured vs predicted).")
    p.add_argument(
        "--xlsx",
        type=Path,
        default=Path("test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("test_new/strict_mt_tc_vp_vs_plots"))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="predictions")
    for col in ["obs", "pred", "abs_rel_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["obs", "pred", "model", "property", "stage", "fluid_state"]).copy()

    models = ["M1", "M2"]
    properties = ["tc_w_mk", "vp_m_s", "vs_m_s"]

    # User preference:
    # - color encodes fluid: dry=red, wet=blue
    # - marker encodes stage: before/after
    fluid_colors = {"dry": "#d62728", "wet": "#1f77b4"}
    stage_markers = {"before": "o", "after": "^"}

    _configure_matplotlib_env(args.out_dir)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    fig, axes = plt.subplots(
        nrows=len(properties),
        ncols=len(models),
        figsize=(13.5, 11.0),
        constrained_layout=True,
    )

    for r, prop in enumerate(properties):
        for c, model in enumerate(models):
            ax = axes[r, c]
            g = df[(df["property"] == prop) & (df["model"] == model)]
            if g.empty:
                ax.set_axis_off()
                continue

            x = g["obs"].to_numpy(dtype=float)
            y = g["pred"].to_numpy(dtype=float)
            lo = float(np.nanmin(np.concatenate([x, y])))
            hi = float(np.nanmax(np.concatenate([x, y])))
            if not np.isfinite(lo) or not np.isfinite(hi):
                ax.set_axis_off()
                continue

            pad = 0.03 * (hi - lo) if hi > lo else 1.0
            x_min, x_max = max(lo - pad, 0.0), hi + pad

            _rel_error_bands(ax, x_min, x_max, bands=[0.025, 0.05, 0.10])
            ax.plot([x_min, x_max], [x_min, x_max], color="k", lw=1.2, alpha=0.65)

            for stage in ["before", "after"]:
                for fluid in ["dry", "wet"]:
                    gg = g[(g["stage"] == stage) & (g["fluid_state"] == fluid)]
                    if gg.empty:
                        continue
                    ax.scatter(
                        gg["obs"],
                        gg["pred"],
                        s=30,
                        alpha=0.85,
                        c=fluid_colors.get(fluid, "0.3"),
                        marker=stage_markers.get(stage, "o"),
                        edgecolors="none",
                    )

            mare = float(np.nanmean(g["abs_rel_error"])) if g["abs_rel_error"].notna().any() else float("nan")
            ax.text(
                0.02,
                0.96,
                f"MARE={mare*100:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.grid(True, alpha=0.25)

            if r == 0:
                ax.set_title(f"{model}")
            if c == 0:
                ax.set_ylabel(_property_label(prop))
            if r == len(properties) - 1:
                ax.set_xlabel("Measured")

    # Figure-level legend (under plot)
    band_handles = [
        mpatches.Patch(color="#d8d8d8", alpha=0.25, label="±10%"),
        mpatches.Patch(color="#bdbdbd", alpha=0.35, label="±5%"),
        mpatches.Patch(color="#a0a0a0", alpha=0.45, label="±2.5%"),
    ]
    stage_handles = [
        mlines.Line2D([], [], color="0.25", marker=stage_markers["before"], linestyle="None", label="Before"),
        mlines.Line2D([], [], color="0.25", marker=stage_markers["after"], linestyle="None", label="After"),
    ]
    fluid_handles = [
        mlines.Line2D([], [], color=fluid_colors["dry"], marker="o", linestyle="None", label="Dry"),
        mlines.Line2D([], [], color=fluid_colors["wet"], marker="o", linestyle="None", label="Wet"),
    ]

    fig.legend(
        handles=band_handles + stage_handles + fluid_handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "measured_vs_predicted_parity.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
