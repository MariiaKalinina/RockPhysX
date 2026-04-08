from __future__ import annotations

import argparse
from dataclasses import dataclass
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


def _normalize_sample_id(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(float(x)))
        return str(float(x)).rstrip("0").rstrip(".")
    return str(x).strip()


def beta_expected_aspect_ratio(
    m: float,
    kappa: float,
    n_quad: int = 401,
) -> float:
    """
    Expected aspect ratio for the M2 parameterization used in mt_tc_m1_m2:

      U ~ Beta(a, b),  a = m*kappa, b = (1-m)*kappa
      z = -4 + 4U
      α = 10^z

    Uses midpoint quadrature in U-space.
    """
    m = float(m)
    kappa = float(kappa)
    if not (0.0 < m < 1.0):
        return float("nan")
    if not (kappa > 0):
        return float("nan")

    a = max(m * kappa, 1e-8)
    b = max((1.0 - m) * kappa, 1e-8)

    u = (np.arange(n_quad, dtype=float) + 0.5) / float(n_quad)
    logw = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w = w / np.sum(w)

    aspect_ratio = 10.0 ** (-4.0 + 4.0 * u)
    return float(np.sum(w * aspect_ratio))


@dataclass(frozen=True)
class Inputs:
    tc_only_xlsx: Path
    joint_xlsx: Path
    out_dir: Path
    n_quad: int = 401
    dpi: int = 300


def _load_fits_tc_only(path: Path, n_quad: int) -> pd.DataFrame:
    m1 = pd.read_excel(path, sheet_name="M1_fits")
    m2 = pd.read_excel(path, sheet_name="M2_fits")

    m1 = m1.copy()
    m1["sample_id"] = m1["sample_id"].map(_normalize_sample_id)
    m1["model"] = "M1"
    m1["method"] = "TC only"
    m1["ar_before"] = pd.to_numeric(m1["ar_before"], errors="coerce")
    m1["ar_after"] = pd.to_numeric(m1["ar_after"], errors="coerce")

    m2 = m2.copy()
    m2["sample_id"] = m2["sample_id"].map(_normalize_sample_id)
    m2["model"] = "M2"
    m2["method"] = "TC only"
    m2["m_before"] = pd.to_numeric(m2["m_before"], errors="coerce")
    m2["m_after"] = pd.to_numeric(m2["m_after"], errors="coerce")
    m2["kappa_before"] = pd.to_numeric(m2["kappa_before"], errors="coerce")
    m2["kappa_after"] = pd.to_numeric(m2["kappa_after"], errors="coerce")
    m2["ar_before"] = [beta_expected_aspect_ratio(m, k, n_quad=n_quad) for m, k in zip(m2["m_before"], m2["kappa_before"], strict=False)]
    m2["ar_after"] = [beta_expected_aspect_ratio(m, k, n_quad=n_quad) for m, k in zip(m2["m_after"], m2["kappa_after"], strict=False)]

    keep = ["sample_id", "model", "method", "ar_before", "ar_after"]
    return pd.concat([m1[keep], m2[keep]], ignore_index=True)


def _load_fits_joint(path: Path, n_quad: int) -> pd.DataFrame:
    m1 = pd.read_excel(path, sheet_name="M1_fits")
    m2 = pd.read_excel(path, sheet_name="M2_fits")

    m1 = m1.copy()
    m1["sample_id"] = m1["sample_id"].map(_normalize_sample_id)
    m1["model"] = "M1"
    m1["method"] = "TC+Vp+Vs"
    m1["ar_before"] = pd.to_numeric(m1["ar_before"], errors="coerce")
    m1["ar_after"] = pd.to_numeric(m1["ar_after"], errors="coerce")

    m2 = m2.copy()
    m2["sample_id"] = m2["sample_id"].map(_normalize_sample_id)
    m2["model"] = "M2"
    m2["method"] = "TC+Vp+Vs"
    m2["m_before"] = pd.to_numeric(m2["m_before"], errors="coerce")
    m2["m_after"] = pd.to_numeric(m2["m_after"], errors="coerce")
    m2["kappa_before"] = pd.to_numeric(m2["kappa_before"], errors="coerce")
    m2["kappa_after"] = pd.to_numeric(m2["kappa_after"], errors="coerce")
    m2["ar_before"] = [beta_expected_aspect_ratio(m, k, n_quad=n_quad) for m, k in zip(m2["m_before"], m2["kappa_before"], strict=False)]
    m2["ar_after"] = [beta_expected_aspect_ratio(m, k, n_quad=n_quad) for m, k in zip(m2["m_after"], m2["kappa_after"], strict=False)]

    keep = ["sample_id", "model", "method", "ar_before", "ar_after"]
    return pd.concat([m1[keep], m2[keep]], ignore_index=True)


def build_comparison_table(inputs: Inputs) -> pd.DataFrame:
    tc_only = _load_fits_tc_only(inputs.tc_only_xlsx, inputs.n_quad)
    joint = _load_fits_joint(inputs.joint_xlsx, inputs.n_quad)
    df = pd.concat([tc_only, joint], ignore_index=True)

    df["log10_ar_before"] = np.log10(df["ar_before"].astype(float))
    df["log10_ar_after"] = np.log10(df["ar_after"].astype(float))

    df["ar_ratio_after_before"] = df["ar_after"] / df["ar_before"]
    df["ar_pct_change"] = 100.0 * (df["ar_ratio_after_before"] - 1.0)
    df["delta_log10_ar"] = df["log10_ar_after"] - df["log10_ar_before"]

    return df.sort_values(["model", "sample_id", "method"]).reset_index(drop=True)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        ("ar_before", "α_before"),
        ("ar_after", "α_after"),
        ("ar_pct_change", "100·(α_after/α_before−1), %"),
        ("delta_log10_ar", "Δlog10(α)"),
    ]
    for (model, method), g in df.groupby(["model", "method"], sort=True):
        for col, label in metrics:
            x = pd.to_numeric(g[col], errors="coerce").dropna()
            if x.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "metric": label,
                    "n": int(x.size),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
                    "median": float(x.median()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }
            )
    return pd.DataFrame(rows)


def _paired_method_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    pivot = df.pivot_table(index=["sample_id", "model"], columns="method", values=col, aggfunc="first")
    if "TC only" not in pivot.columns or "TC+Vp+Vs" not in pivot.columns:
        return pd.DataFrame(columns=["sample_id", "model", "tc_only", "joint", "joint_minus_tc_only"])
    out = pivot.reset_index().rename(columns={"TC only": "tc_only", "TC+Vp+Vs": "joint"})
    out["joint_minus_tc_only"] = out["joint"] - out["tc_only"]
    return out


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison_per_sample.csv", index=False)

    summary = build_summary_table(df)
    summary.to_csv(out_dir / "comparison_summary.csv", index=False)

    diffs = []
    for col, name in [
        ("log10_ar_before", "diff_log10_ar_before"),
        ("log10_ar_after", "diff_log10_ar_after"),
        ("ar_pct_change", "diff_ar_pct_change"),
        ("delta_log10_ar", "diff_delta_log10_ar"),
    ]:
        d = _paired_method_diff(df, col)
        d["metric"] = name
        diffs.append(d)
    pd.concat(diffs, ignore_index=True).to_csv(out_dir / "comparison_paired_differences.csv", index=False)

    # Optional paired significance (Wilcoxon) for method differences
    try:
        from scipy.stats import wilcoxon  # type: ignore[import-not-found]
    except Exception:
        return

    rows = []
    for model in sorted(df["model"].dropna().unique().tolist()):
        g = df[df["model"] == model]
        for col, label in [
            ("log10_ar_before", "log10(α_before)"),
            ("log10_ar_after", "log10(α_after)"),
            ("ar_pct_change", "100·(α_after/α_before−1), %"),
            ("delta_log10_ar", "Δlog10(α)"),
        ]:
            d = _paired_method_diff(g, col).dropna(subset=["tc_only", "joint"])
            if d.empty:
                continue
            diff = (d["joint"] - d["tc_only"]).to_numpy(dtype=float)
            if diff.size < 3:
                continue
            try:
                stat, pval = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            except Exception:
                continue
            rows.append(
                {
                    "model": model,
                    "metric": label,
                    "n_pairs": int(diff.size),
                    "median_diff": float(np.median(diff)),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(pval),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "comparison_significance_wilcoxon.csv", index=False)


def save_plots(df: pd.DataFrame, inputs: Inputs) -> None:
    out_dir = inputs.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    def _scatter_panel(ax, g: pd.DataFrame, which: str, title: str) -> None:
        pivot = g.pivot_table(index="sample_id", columns="method", values=which, aggfunc="first")
        pivot = pivot.dropna()
        if pivot.empty:
            ax.set_axis_off()
            return
        x = pivot["TC only"].to_numpy()
        y = pivot["TC+Vp+Vs"].to_numpy()
        ax.scatter(x, y, s=35, alpha=0.85)
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        ax.plot([lo, hi], [lo, hi], color="k", lw=1.2, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("TC only")
        ax.set_ylabel("TC+Vp+Vs")
        ax.grid(True, alpha=0.25)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    for i, model in enumerate(["M1", "M2"]):
        g = df[df["model"] == model]
        _scatter_panel(axes[i, 0], g, "log10_ar_before", f"{model}: log10(α_before)")
        _scatter_panel(axes[i, 1], g, "log10_ar_after", f"{model}: log10(α_after)")
    out_scatter = out_dir / "scatter_tc_only_vs_joint_log10_ar.png"
    fig.savefig(out_scatter, dpi=inputs.dpi, bbox_inches="tight")
    plt.close(fig)

    # Percent-change comparison (boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["M1", "M2"], strict=True):
        g = df[df["model"] == model].copy()
        data = [
            g.loc[g["method"] == "TC only", "ar_pct_change"].dropna().to_numpy(),
            g.loc[g["method"] == "TC+Vp+Vs", "ar_pct_change"].dropna().to_numpy(),
        ]
        ax.boxplot(
            data,
            tick_labels=["TC only", "TC+Vp+Vs"],
            showmeans=True,
            meanline=True,
        )
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
        ax.set_title(f"{model}: % change in α")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("100·(α_after/α_before − 1), %")
    out_box = out_dir / "boxplot_ar_pct_change_tc_only_vs_joint.png"
    fig.savefig(out_box, dpi=inputs.dpi, bbox_inches="tight")
    plt.close(fig)

    # Paired lines (slopegraph-style) for percent change
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True, constrained_layout=True)
    x_positions = {"TC only": 0, "TC+Vp+Vs": 1}
    for ax, model in zip(axes, ["M1", "M2"], strict=True):
        g = df[df["model"] == model]
        pivot = g.pivot_table(index="sample_id", columns="method", values="ar_pct_change", aggfunc="first")
        pivot = pivot.dropna()
        for _, row in pivot.iterrows():
            ax.plot(
                [x_positions["TC only"], x_positions["TC+Vp+Vs"]],
                [row["TC only"], row["TC+Vp+Vs"]],
                color="0.35",
                alpha=0.55,
                lw=1.2,
            )
        ax.scatter(np.full(pivot.shape[0], x_positions["TC only"]), pivot["TC only"], s=18, alpha=0.8)
        ax.scatter(np.full(pivot.shape[0], x_positions["TC+Vp+Vs"]), pivot["TC+Vp+Vs"], s=18, alpha=0.8)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
        ax.set_xticks([0, 1], ["TC only", "TC+Vp+Vs"])
        ax.set_title(f"{model}: paired % change in α")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("100·(α_after/α_before − 1), %")
    out_slope = out_dir / "paired_ar_pct_change_tc_only_vs_joint.png"
    fig.savefig(out_slope, dpi=inputs.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare inversion results: TC-only (mt_tc_m1_m2) vs joint (TC+Vp+Vs).",
    )
    p.add_argument("--tc-only-xlsx", type=Path, default=Path("test_new/mt_tc_m1_m2_results.xlsx"))
    p.add_argument("--joint-xlsx", type=Path, default=Path("test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx"))
    p.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    p.add_argument("--n-quad", type=int, default=401, help="Quadrature points for M2 expected aspect ratio.")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    inputs = Inputs(
        tc_only_xlsx=args.tc_only_xlsx,
        joint_xlsx=args.joint_xlsx,
        out_dir=args.out_dir,
        n_quad=int(args.n_quad),
        dpi=int(args.dpi),
    )

    df = build_comparison_table(inputs)
    save_tables(df, inputs.out_dir)
    save_plots(df, inputs)
    print(f"Saved tables/plots in: {inputs.out_dir}")


if __name__ == "__main__":
    main()
