from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_log10(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    return float(np.log10(x))


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _beta_pdf_over_z(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Density over z=log10(alpha) where alpha in [1e-4, 1] (z in [-4,0]).
    Beta is defined on u in [0,1] with u=(z+4)/4.
    """
    # scipy may not be available in every runtime; implement log-form directly.
    u = (z + 4.0) / 4.0
    eps = 1e-12
    u = np.clip(u, eps, 1.0 - eps)
    a = max(float(a), eps)
    b = max(float(b), eps)
    # log Beta(a,b) via gammaln
    from math import lgamma

    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    log_pdf_u = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u) - logB
    # change of variables: u=(z+4)/4 => du/dz = 1/4
    return 0.25 * np.exp(log_pdf_u)


def _beta_quantile_alpha_p50(m: float, kappa: float, *, z_grid: np.ndarray) -> float:
    m = float(m)
    kappa = float(kappa)
    a = max(m * kappa, 1e-8)
    b = max((1.0 - m) * kappa, 1e-8)
    pz = _beta_pdf_over_z(z_grid, a, b)
    dz = float(z_grid[1] - z_grid[0])
    # normalize (safety)
    s = float(np.trapz(pz, dx=dz))
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    pz = pz / s
    cdf = np.cumsum(pz) * dz
    j = int(np.searchsorted(cdf, 0.50, side="left"))
    j = min(max(j, 0), z_grid.size - 1)
    z50 = float(z_grid[j])
    return float(10.0 ** z50)


def _load_gsa_bayes_m1(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="summary")
    df["lab_sample_id"] = _to_float_series(df["lab_sample_id"])
    df["mode"] = df["mode"].astype(str)
    keep = [
        "lab_sample_id",
        "mode",
        "alpha_before_p50",
        "alpha_after_p50",
        "delta_z_p50",
        "p_delta_z_lt_0",
        "p_delta_z_lt_m0p05",
        "p_delta_z_lt_m0p1",
        "p_delta_z_lt_m0p2",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.rename(columns={c: f"osp_m1_{c}" for c in out.columns if c not in {"lab_sample_id", "mode"}})
    return out


def _load_gsa_bayes_m2(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="summary")
    df["lab_sample_id"] = _to_float_series(df["lab_sample_id"])
    df["mode"] = df["mode"].astype(str)
    keep = [
        "lab_sample_id",
        "mode",
        "alpha_before_p50",
        "alpha_after_p50",
        "delta_z_p50",
        "p_delta_z_lt_0",
        "p_delta_z_lt_m0p05",
        "p_delta_z_lt_m0p1",
        "p_delta_z_lt_m0p2",
        "m_before_map",
        "kappa_before_map",
        "m_after_map",
        "kappa_after_map",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.rename(columns={c: f"osp_m2_{c}" for c in out.columns if c not in {"lab_sample_id", "mode"}})
    return out


def _load_mt_tc_only(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    m1 = xl.parse("M1_fits")
    m2 = xl.parse("M2_fits")
    m1["lab_sample_id"] = _to_float_series(m1["sample_id"])
    m1 = m1[["lab_sample_id", "delta_z", "ar_before", "ar_after"]].copy()
    m1 = m1.rename(
        columns={
            "delta_z": "mt_tc_only_m1_delta_z",
            "ar_before": "mt_tc_only_m1_alpha_before",
            "ar_after": "mt_tc_only_m1_alpha_after",
        }
    )
    m2["lab_sample_id"] = _to_float_series(m2["sample_id"])
    m2 = m2[["lab_sample_id", "m_before", "kappa_before", "m_after", "kappa_after"]].copy()
    return m1, m2


def _load_mt_tc_vp_vs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    m1 = xl.parse("M1_fits")
    m2 = xl.parse("M2_fits")
    m1["lab_sample_id"] = _to_float_series(m1["sample_id"])
    m1 = m1[["lab_sample_id", "delta_z", "ar_before", "ar_after"]].copy()
    m1 = m1.rename(
        columns={
            "delta_z": "mt_tc_vp_vs_m1_delta_z",
            "ar_before": "mt_tc_vp_vs_m1_alpha_before",
            "ar_after": "mt_tc_vp_vs_m1_alpha_after",
        }
    )
    m2["lab_sample_id"] = _to_float_series(m2["sample_id"])
    m2 = m2[["lab_sample_id", "m_before", "kappa_before", "m_after", "kappa_after"]].copy()
    return m1, m2


def _load_mt_bayes_m1(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="sample_summary")
    # This file uses `sample_id` (same numeric ids as lab_sample_id).
    df["lab_sample_id"] = _to_float_series(df["sample_id"])
    out = df[["lab_sample_id", "ao_before_median", "ao_after_median", "delta_z_median", "delta_z_q2.5", "delta_z_q97.5", "p_delta_lt_0"]].copy()
    out = out.rename(
        columns={
            "ao_before_median": "mt_bayes_m1_alpha_before_median",
            "ao_after_median": "mt_bayes_m1_alpha_after_median",
            "delta_z_median": "mt_bayes_m1_delta_z_median",
            "delta_z_q2.5": "mt_bayes_m1_delta_z_q2p5",
            "delta_z_q97.5": "mt_bayes_m1_delta_z_q97p5",
            "p_delta_lt_0": "mt_bayes_m1_p_delta_z_lt_0",
        }
    )
    return out


def _derive_mt_m2_quantiles(m2: pd.DataFrame, *, prefix: str, z_grid: np.ndarray) -> pd.DataFrame:
    df = m2.copy()
    for c in ["m_before", "kappa_before", "m_after", "kappa_after"]:
        df[c] = _to_float_series(df[c])
    a_before = []
    a_after = []
    dz = []
    for _, r in df.iterrows():
        ab = _beta_quantile_alpha_p50(float(r["m_before"]), float(r["kappa_before"]), z_grid=z_grid)
        aa = _beta_quantile_alpha_p50(float(r["m_after"]), float(r["kappa_after"]), z_grid=z_grid)
        a_before.append(ab)
        a_after.append(aa)
        dz.append(_safe_log10(aa) - _safe_log10(ab))
    out = pd.DataFrame(
        {
            "lab_sample_id": df["lab_sample_id"].to_numpy(float),
            f"{prefix}_m2_alpha_before_p50": np.array(a_before, float),
            f"{prefix}_m2_alpha_after_p50": np.array(a_after, float),
            f"{prefix}_m2_delta_z_p50": np.array(dz, float),
            f"{prefix}_m2_m_before": df["m_before"].to_numpy(float),
            f"{prefix}_m2_kappa_before": df["kappa_before"].to_numpy(float),
            f"{prefix}_m2_m_after": df["m_after"].to_numpy(float),
            f"{prefix}_m2_kappa_after": df["kappa_after"].to_numpy(float),
        }
    )
    return out


def _scatter(out_png: Path, df: pd.DataFrame, *, x: str, y: str, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    xx = pd.to_numeric(df[x], errors="coerce").to_numpy(float)
    yy = pd.to_numeric(df[y], errors="coerce").to_numpy(float)
    m = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[m]
    yy = yy[m]
    if xx.size < 2:
        return

    lo = float(np.nanmin(np.concatenate([xx, yy])))
    hi = float(np.nanmax(np.concatenate([xx, yy])))
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.4), constrained_layout=True)
        ax.scatter(xx, yy, s=55, alpha=0.85, color="C0")
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.4, color="0.25", alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare OSP (GSA) vs Mori–Tanaka inversion results.")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--osp-bayes-m1", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m1_tc_elastic_results.xlsx"))
    ap.add_argument("--osp-bayes-m2", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m2_tc_elastic_results.xlsx"))
    ap.add_argument("--mt-tc-only", type=Path, default=Path("test_new/mt_tc_m1_m2_results.xlsx"))
    ap.add_argument("--mt-tc-vp-vs", type=Path, default=Path("test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx"))
    ap.add_argument("--mt-bayes-m1", type=Path, default=Path("test_new/bayes_m1_tc_vp_vs_collection_results.xlsx"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # OSP
    osp_m1 = _load_gsa_bayes_m1(Path(args.osp_bayes_m1))
    osp_m2 = _load_gsa_bayes_m2(Path(args.osp_bayes_m2))

    # MT (TC-only deterministic)
    mt_tc_m1, mt_tc_m2_raw = _load_mt_tc_only(Path(args.mt_tc_only))
    # MT (TC+Vp+Vs deterministic)
    mt_joint_m1, mt_joint_m2_raw = _load_mt_tc_vp_vs(Path(args.mt_tc_vp_vs))
    # MT Bayesian M1 (joint)
    mt_bayes_m1 = _load_mt_bayes_m1(Path(args.mt_bayes_m1))

    z_grid = np.linspace(-4.0, 0.0, 4001)
    mt_tc_m2 = _derive_mt_m2_quantiles(mt_tc_m2_raw, prefix="mt_tc_only", z_grid=z_grid)
    mt_joint_m2 = _derive_mt_m2_quantiles(mt_joint_m2_raw, prefix="mt_tc_vp_vs", z_grid=z_grid)

    # Merge per branch
    tc_only = (
        osp_m1[osp_m1["mode"] == "tc_only"]
        .merge(osp_m2[osp_m2["mode"] == "tc_only"], on=["lab_sample_id", "mode"], how="outer")
        .merge(mt_tc_m1, on="lab_sample_id", how="left")
        .merge(mt_tc_m2, on="lab_sample_id", how="left")
    )
    tc_vp_vs = (
        osp_m1[osp_m1["mode"] == "tc_vp_vs"]
        .merge(osp_m2[osp_m2["mode"] == "tc_vp_vs"], on=["lab_sample_id", "mode"], how="outer")
        .merge(mt_joint_m1, on="lab_sample_id", how="left")
        .merge(mt_joint_m2, on="lab_sample_id", how="left")
        .merge(mt_bayes_m1, on="lab_sample_id", how="left")
    )

    out_xlsx = out_dir / "osp_vs_mt_comparison.xlsx"
    with pd.ExcelWriter(out_xlsx) as xw:
        tc_only.to_excel(xw, sheet_name="tc_only", index=False)
        tc_vp_vs.to_excel(xw, sheet_name="tc_vp_vs", index=False)

    # Plots: Δz comparison
    _scatter(
        out_dir / "delta_z_tc_only_m1_osp_vs_mt.png",
        tc_only,
        x="mt_tc_only_m1_delta_z",
        y="osp_m1_delta_z_p50",
        title="Δz comparison (TC-only, M1)",
        xlabel="MT Δz (deterministic)",
        ylabel="OSP Δz (Bayes p50)",
    )
    _scatter(
        out_dir / "delta_z_tc_only_m2_osp_vs_mt.png",
        tc_only,
        x="mt_tc_only_m2_delta_z_p50",
        y="osp_m2_delta_z_p50",
        title="Δz comparison (TC-only, M2)",
        xlabel="MT Δz (deterministic, beta p50)",
        ylabel="OSP Δz (Bayes p50)",
    )
    _scatter(
        out_dir / "delta_z_tc_vp_vs_m1_osp_vs_mt.png",
        tc_vp_vs,
        x="mt_tc_vp_vs_m1_delta_z",
        y="osp_m1_delta_z_p50",
        title="Δz comparison (TC+Vp+Vs, M1)",
        xlabel="MT Δz (deterministic)",
        ylabel="OSP Δz (Bayes p50)",
    )
    _scatter(
        out_dir / "delta_z_tc_vp_vs_m2_osp_vs_mt.png",
        tc_vp_vs,
        x="mt_tc_vp_vs_m2_delta_z_p50",
        y="osp_m2_delta_z_p50",
        title="Δz comparison (TC+Vp+Vs, M2)",
        xlabel="MT Δz (deterministic, beta p50)",
        ylabel="OSP Δz (Bayes p50)",
    )
    # MT Bayesian M1 vs OSP Bayesian M1
    _scatter(
        out_dir / "delta_z_tc_vp_vs_m1_osp_vs_mt_bayes.png",
        tc_vp_vs,
        x="mt_bayes_m1_delta_z_median",
        y="osp_m1_delta_z_p50",
        title="Δz comparison (TC+Vp+Vs, M1 Bayesian)",
        xlabel="MT Δz (Bayes median)",
        ylabel="OSP Δz (Bayes p50)",
    )

    print(f"Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
