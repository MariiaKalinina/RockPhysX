from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = BASE_DIR / "mt_tc_m1_m2_results.xlsx"

REGIONS = [
    (1e-4, 1e-2, "#ffd1e8", "Microcracks"),
    (1e-2, 1e-1, "#ffe6bf", "Crack-like pores"),
    (1e-1, 1.0, "#d9ffd4", "Interparticle pores"),
]

BEFORE_LINE = "#0b51ff"
AFTER_LINE = "#8b0000"
BEFORE_FILL = "#55b6e8"
AFTER_FILL = "#e35b8f"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot beta-distribution of log10(aspect ratio) for one sample (M2 fit).")
    p.add_argument("sample_id", nargs="+", help="One or more sample ids as in the Excel sheet (e.g. 11.2 12 15).")
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Path to mt_tc_m1_m2_results.xlsx")
    p.add_argument("--sheet", default="M2_fits", help="Sheet name with M2 fitted parameters.")
    p.add_argument("--out", type=Path, default=None, help="Optional path to save PNG instead of showing a window.")
    p.add_argument("--cols", type=int, default=2, help="Number of subplot columns when plotting multiple samples.")
    return p.parse_args()


def load_m2_fits(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results XLSX: {path} (resolved to {path.resolve()})")
    return pd.read_excel(path, sheet_name=sheet_name)


def get_row_for_sample(m2: pd.DataFrame, sample_id: float) -> pd.Series:
    if "sample_id" not in m2.columns:
        raise ValueError("Expected column 'sample_id' in M2_fits sheet.")
    sample_ids = pd.to_numeric(m2["sample_id"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(sample_ids) & np.isclose(sample_ids, float(sample_id), atol=1e-9)
    matches = m2.loc[mask]
    if matches.empty:
        available = ", ".join(str(x) for x in pd.unique(m2["sample_id"]))
        raise KeyError(f"sample_id={sample_id} not found. Available sample_id values: {available}")
    return matches.iloc[0]


def _compute_curves(row: pd.Series) -> tuple[float, float, float, float]:
    m_before = float(row["m_before"])
    k_before = float(row["kappa_before"])
    m_after = float(row["m_after"])
    k_after = float(row["kappa_after"])

    a_before = m_before * k_before
    b_before = (1 - m_before) * k_before
    a_after = m_after * k_after
    b_after = (1 - m_after) * k_after
    return a_before, b_before, a_after, b_after


def _pdf_over_log10_alpha(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Density over z = log10(alpha) for alpha in [1e-4, 1]."""
    u = (z + 4) / 4  # map z∈[-4,0] -> u∈[0,1]
    return 0.25 * beta.pdf(u, a, b)


def _mean_alpha(z: np.ndarray, pz: np.ndarray) -> float:
    dz = float(z[1] - z[0]) if z.size > 1 else 1.0
    alpha = 10.0 ** z
    return float(np.sum(alpha * pz) * dz)


def plot_one_sample(
    ax: plt.Axes,
    *,
    sample_id: float,
    a_before: float,
    b_before: float,
    a_after: float,
    b_after: float,
    z: np.ndarray,
) -> tuple[float, float]:
    pz_before = _pdf_over_log10_alpha(z, a_before, b_before)
    pz_after = _pdf_over_log10_alpha(z, a_after, b_after)
    mean_before = _mean_alpha(z, pz_before)
    mean_after = _mean_alpha(z, pz_after)

    x = 10.0 ** z
    y_before = pz_before / float(np.max(pz_before)) if np.max(pz_before) > 0 else pz_before
    y_after = pz_after / float(np.max(pz_after)) if np.max(pz_after) > 0 else pz_after

    # background regions
    for x0, x1, color, _label in REGIONS:
        ax.axvspan(x0, x1, color=color, alpha=0.55, zorder=0)

    # filled densities
    ax.fill_between(x, 0.0, y_before, color=BEFORE_FILL, alpha=0.45, zorder=2)
    ax.fill_between(x, 0.0, y_after, color=AFTER_FILL, alpha=0.45, zorder=3)
    ax.plot(x, y_before, color=BEFORE_LINE, linewidth=2.2, zorder=4)
    ax.plot(x, y_after, color=AFTER_LINE, linewidth=2.2, zorder=5)

    # mean lines
    ax.axvline(mean_before, color=BEFORE_LINE, linestyle="--", linewidth=1.6, zorder=6)
    ax.axvline(mean_after, color=AFTER_LINE, linestyle="--", linewidth=1.6, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Aspect Ratio (α) - Log Scale")
    ax.set_ylabel("Normalized Probability Density")
    ax.set_title(f"Sample {sample_id:g}", fontsize=13, weight="bold")
    return mean_before, mean_after


def main() -> None:
    args = parse_args()
    sample_ids = [float(x) for x in args.sample_id]
    m2 = load_m2_fits(args.xlsx, args.sheet)

    # grid for z = log10(aspect ratio)
    z = np.linspace(-4, 0, 400)

    if len(sample_ids) == 1:
        sample_id = sample_ids[0]
        row = get_row_for_sample(m2, sample_id)
        a_before, b_before, a_after, b_after = _compute_curves(row)

        fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.3))
        mean_before, mean_after = plot_one_sample(
            ax,
            sample_id=sample_id,
            a_before=a_before,
            b_before=b_before,
            a_after=a_after,
            b_after=b_after,
            z=z,
        )

        print(f"sample_id: {sample_id:g}")
        print(f"before: alpha={a_before:.6g}, beta={b_before:.6g}, mean_aspect_ratio={mean_before:.4g}")
        print(f"after:  alpha={a_after:.6g}, beta={b_after:.6g}, mean_aspect_ratio={mean_after:.4g}")

        legend_items = [
            Patch(facecolor=BEFORE_FILL, edgecolor=BEFORE_FILL, alpha=0.45, label="Before: Dominant interparticle pores"),
            Patch(facecolor=AFTER_FILL, edgecolor=AFTER_FILL, alpha=0.45, label="After: Increased microcracks"),
            Line2D([0], [0], color=BEFORE_LINE, linestyle="--", linewidth=1.6, label=f"Mean before: {mean_before:.3g}"),
            Line2D([0], [0], color=AFTER_LINE, linestyle="--", linewidth=1.6, label=f"Mean after: {mean_after:.3g}"),
        ] + [Patch(facecolor=c, edgecolor=c, alpha=0.55, label=lab) for _, _, c, lab in REGIONS]
        ax.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        fig.tight_layout()
    else:
        cols = max(1, int(args.cols))
        rows = int(np.ceil(len(sample_ids) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.8 * cols, 4.8 * rows), squeeze=False)
        for i, sample_id in enumerate(sample_ids):
            ax = axes[i // cols][i % cols]
            row = get_row_for_sample(m2, sample_id)
            a_before, b_before, a_after, b_after = _compute_curves(row)

            mean_before, mean_after = plot_one_sample(
                ax,
                sample_id=sample_id,
                a_before=a_before,
                b_before=b_before,
                a_after=a_after,
                b_after=b_after,
                z=z,
            )

            print(f"sample_id: {sample_id:g}")
            print(f"before: alpha={a_before:.6g}, beta={b_before:.6g}, mean_aspect_ratio={mean_before:.4g}")
            print(f"after:  alpha={a_after:.6g}, beta={b_after:.6g}, mean_aspect_ratio={mean_after:.4g}")

        # hide unused axes
        for j in range(len(sample_ids), rows * cols):
            axes[j // cols][j % cols].axis("off")
        legend_items = [
            Patch(facecolor=BEFORE_FILL, edgecolor=BEFORE_FILL, alpha=0.45, label="Before: Dominant interparticle pores"),
            Patch(facecolor=AFTER_FILL, edgecolor=AFTER_FILL, alpha=0.45, label="After: Increased microcracks"),
            Line2D([0], [0], color=BEFORE_LINE, linestyle="--", linewidth=1.6, label="Mean before"),
            Line2D([0], [0], color=AFTER_LINE, linestyle="--", linewidth=1.6, label="Mean after"),
        ] + [Patch(facecolor=c, edgecolor=c, alpha=0.55, label=lab) for _, _, c, lab in REGIONS]
        fig.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
        fig.suptitle("Aspect ratio (M2 beta fit): before vs after", fontsize=14, weight="bold")
        fig.tight_layout(rect=(0, 0, 0.88, 0.95))

    if args.out is not None:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved figure to: {out_path.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
