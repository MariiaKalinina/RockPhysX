from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa_elastic

import importlib.util
import sys


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


@lru_cache(maxsize=1)
def _load_mt_elastic_module() -> object:
    path = Path("test_new/mori-tanaka/strict_mt_elastic_pores.py").resolve()
    spec = importlib.util.spec_from_file_location("strict_mt_elastic_pores", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load MT elastic module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[misc]
    return mod


def _vp_vs_from_KG_rho(K_pa: float, G_pa: float, rho_kg_m3: float) -> tuple[float, float]:
    vp, vs = gsa_elastic.velocities_from_KG_rho(float(K_pa), float(G_pa), float(rho_kg_m3))
    return float(vp), float(vs)


def _predict_vp_vs_mt(
    *,
    matrix_K_pa: float,
    matrix_G_pa: float,
    fluid_K_pa: float,
    phi: float,
    alpha: float,
    rho_bulk_kg_m3: float,
) -> tuple[float, float]:
    mt = _load_mt_elastic_module()
    ElasticPhase = getattr(mt, "ElasticPhase")
    strict_mt_elastic_random_spheroidal_pores = getattr(mt, "strict_mt_elastic_random_spheroidal_pores")
    res = strict_mt_elastic_random_spheroidal_pores(
        phi=float(phi),
        a_ratio=float(alpha),
        matrix=ElasticPhase(K=float(matrix_K_pa), G=float(matrix_G_pa)),
        inclusion=ElasticPhase(K=float(fluid_K_pa), G=1e-9),
        rho_matrix_kg_m3=2700.0,
        rho_inclusion_kg_m3=1000.0,
    )
    return _vp_vs_from_KG_rho(float(res.K_eff), float(res.G_eff), float(rho_bulk_kg_m3))


def _predict_vp_vs_osp(
    *,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
    matrix_K_pa: float,
    matrix_G_pa: float,
    fluid_K_pa: float,
    phi: float,
    alpha: float,
    rho_bulk_kg_m3: float,
) -> tuple[float, float]:
    matrix = gsa_elastic.ElasticPhase(K=float(matrix_K_pa), G=float(matrix_G_pa))
    inclusion = gsa_elastic.ElasticPhase(K=float(fluid_K_pa), G=1e-9)
    _, _, K_eff, G_eff = gsa_elastic.gsa_effective_stiffness_random_two_phase(
        phi=float(phi),
        matrix=matrix,
        inclusion=inclusion,
        pore_aspect_ratio=float(alpha),
        backend=backend,
        comparison_body="matrix",
        k_connectivity=None,
        sign=-1,
    )
    return _vp_vs_from_KG_rho(float(K_eff), float(G_eff), float(rho_bulk_kg_m3))


@dataclass(frozen=True)
class AlphaStage:
    before: float
    after: float


def _load_alpha_from_tc_only_mt_m1(path: Path) -> dict[float, AlphaStage]:
    df = pd.read_excel(path, sheet_name="M1_fits")
    df["lab_sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype(float)
    out: dict[float, AlphaStage] = {}
    for _, r in df.iterrows():
        sid = float(r["lab_sample_id"])
        out[sid] = AlphaStage(before=float(r["ar_before"]), after=float(r["ar_after"]))
    return out


def _load_alpha_from_tc_only_osp_m1(path: Path) -> dict[float, AlphaStage]:
    df = pd.read_excel(path, sheet_name="summary")
    df = df[df["mode"].astype(str) == "tc_only"].copy()
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce").astype(float)
    out: dict[float, AlphaStage] = {}
    for _, r in df.iterrows():
        sid = float(r["lab_sample_id"])
        out[sid] = AlphaStage(before=float(r["alpha_before_p50"]), after=float(r["alpha_after_p50"]))
    return out


def _load_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="measurements_long")
    # Some datasets carry porosity only as phi_pct. Fill phi_frac from it.
    if "phi_frac" in df.columns and "phi_pct" in df.columns:
        df["phi_frac"] = df["phi_frac"].where(df["phi_frac"].notna(), pd.to_numeric(df["phi_pct"], errors="coerce") / 100.0)
    need = ["lab_sample_id", "stage", "fluid_state", "phi_frac", "vp_m_s", "vs_m_s", "bulk_density_g_cm3"]
    df = df.dropna(subset=need).copy()
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].astype(str).str.strip().str.lower()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip().str.lower()
    df["phi_frac"] = pd.to_numeric(df["phi_frac"], errors="coerce").astype(float)
    df["vp_m_s"] = pd.to_numeric(df["vp_m_s"], errors="coerce").astype(float)
    df["vs_m_s"] = pd.to_numeric(df["vs_m_s"], errors="coerce").astype(float)
    df["rho_bulk_kg_m3"] = pd.to_numeric(df["bulk_density_g_cm3"], errors="coerce").astype(float) * 1000.0
    return df.reset_index(drop=True)


def _kg_grid_from_hs(n: int) -> tuple[np.ndarray, np.ndarray]:
    # HS-feasible intervals from your table (derived from VP/VS bounds at rho=2720).
    K_min, K_max = 46.43, 80.29  # GPa
    G_min, G_max = 27.03, 41.55  # GPa
    K = np.linspace(K_min, K_max, int(n)) * 1e9
    G = np.linspace(G_min, G_max, int(n)) * 1e9
    KK, GG = np.meshgrid(K, G, indexing="xy")
    return KK.ravel(), GG.ravel()


def _fluid_K_pa(fluid_state: str) -> float:
    # Match your fixed pore-fluid properties (gas vs brine/water).
    fs = str(fluid_state).lower()
    if fs == "dry":
        return 0.00014 * 1e9
    return 2.20 * 1e9


def _summarize_dist(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.percentile(x, 5)), float(np.percentile(x, 50)), float(np.percentile(x, 95))


def _style_rc() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }


def _barplot_panels(
    out_png: Path,
    rows: pd.DataFrame,
    *,
    value_col: str,
    y_label: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    stages = ["before", "after"]
    fluids = ["dry", "wet"]

    with plt.rc_context(_style_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(16.5, 8.8), constrained_layout=True, sharey=True)

        colors = {"MT": "#8da0cb", "OSP": "#fc8d62"}  # pastel blue / orange

        for i, stage in enumerate(stages):
            for j, fluid in enumerate(fluids):
                ax = axes[i, j]
                sub = rows[(rows["stage"] == stage) & (rows["fluid_state"] == fluid)].copy()
                sub = sub.sort_values("lab_sample_id")

                x = np.arange(len(sub["lab_sample_id"].unique()))
                # group bars: MT and OSP side by side
                width = 0.40

                for k, method in enumerate(["MT", "OSP"]):
                    ss = sub[sub["method"] == method].copy()
                    ss = ss.sort_values("lab_sample_id")
                    y = ss[value_col].to_numpy(float)
                    lo = ss[f"{value_col}_p05"].to_numpy(float)
                    hi = ss[f"{value_col}_p95"].to_numpy(float)
                    yerr = np.vstack([y - lo, hi - y])
                    ax.bar(
                        x + (k - 0.5) * width,
                        y,
                        width=width,
                        color=colors[method],
                        edgecolor="0.35",
                        linewidth=0.8,
                        alpha=0.9,
                        label=method if (i == 0 and j == 0) else None,
                    )
                    ax.errorbar(
                        x + (k - 0.5) * width,
                        y,
                        yerr=yerr,
                        fmt="none",
                        ecolor="0.25",
                        elinewidth=1.0,
                        capsize=2.5,
                        alpha=0.9,
                    )

                ax.set_title(f"{stage}, {fluid}")
                ax.grid(True, axis="y", alpha=0.25)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{v:g}" for v in sub["lab_sample_id"].unique()], rotation=60, ha="right")
                if j == 0:
                    ax.set_ylabel(y_label)

        axes[0, 0].legend(loc="upper right", frameon=True)
        fig.suptitle(title)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict Vp/Vs from TC-inverted alpha (single microstructure) with HS matrix (K,G) grid.")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--mt-tc-only-results", type=Path, default=Path("test_new/mt_tc_m1_m2_results.xlsx"))
    ap.add_argument("--osp-tc-only-results", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m1_tc_elastic_results.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--matrix-grid-n", type=int, default=5)
    ap.add_argument(
        "--osp-backend-so",
        type=Path,
        default=Path("test_new/gsa/plots/libgsa_elastic_fortran.so"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    # Load alphas (TC-only inversion) for each method
    alpha_mt = _load_alpha_from_tc_only_mt_m1(Path(args.mt_tc_only_results))
    alpha_osp = _load_alpha_from_tc_only_osp_m1(Path(args.osp_tc_only_results))

    df = _load_measurements(Path(args.data_xlsx))

    # Build OSP backend (load .so; avoid rebuild)
    backend = gsa_elastic.build_backend(
        green_fortran=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"),
        output_library=Path(args.osp_backend_so),
        force_rebuild=False,
    )

    K_grid, G_grid = _kg_grid_from_hs(int(args.matrix_grid_n))

    recs: list[dict[str, object]] = []

    for _, r in df.iterrows():
        sid = float(r["lab_sample_id"])
        stage = str(r["stage"])
        fluid_state = str(r["fluid_state"])
        phi = float(r["phi_frac"])
        rho = float(r["rho_bulk_kg_m3"])
        vp_obs = float(r["vp_m_s"])
        vs_obs = float(r["vs_m_s"])
        Kf = _fluid_K_pa(fluid_state)

        # Select alpha for stage
        if sid in alpha_mt:
            a = alpha_mt[sid].before if stage == "before" else alpha_mt[sid].after
        else:
            a = float("nan")
        if sid in alpha_osp:
            a2 = alpha_osp[sid].before if stage == "before" else alpha_osp[sid].after
        else:
            a2 = float("nan")

        # Predictions across matrix grid (for uncertainty bands)
        if np.isfinite(a):
            vp_preds = []
            vs_preds = []
            for Km, Gm in zip(K_grid, G_grid, strict=False):
                vp_p, vs_p = _predict_vp_vs_mt(
                    matrix_K_pa=float(Km),
                    matrix_G_pa=float(Gm),
                    fluid_K_pa=float(Kf),
                    phi=phi,
                    alpha=float(a),
                    rho_bulk_kg_m3=rho,
                )
                vp_preds.append(vp_p)
                vs_preds.append(vs_p)
            vp_preds = np.asarray(vp_preds, float)
            vs_preds = np.asarray(vs_preds, float)
            vp_mis = np.abs(vp_preds - vp_obs) / vp_obs * 100.0
            vs_mis = np.abs(vs_preds - vs_obs) / vs_obs * 100.0
            vp_p05, vp_p50, vp_p95 = _summarize_dist(vp_mis)
            vs_p05, vs_p50, vs_p95 = _summarize_dist(vs_mis)
            agg = np.sqrt(0.5 * (vp_mis**2 + vs_mis**2))
            agg_p05, agg_p50, agg_p95 = _summarize_dist(agg)
            recs.append(
                {
                    "lab_sample_id": sid,
                    "stage": stage,
                    "fluid_state": fluid_state,
                    "method": "MT",
                    "alpha_used": float(a),
                    "vp_misfit_pct": vp_p50,
                    "vp_misfit_pct_p05": vp_p05,
                    "vp_misfit_pct_p95": vp_p95,
                    "vs_misfit_pct": vs_p50,
                    "vs_misfit_pct_p05": vs_p05,
                    "vs_misfit_pct_p95": vs_p95,
                    "agg_misfit_pct": agg_p50,
                    "agg_misfit_pct_p05": agg_p05,
                    "agg_misfit_pct_p95": agg_p95,
                }
            )

        if np.isfinite(a2):
            vp_preds = []
            vs_preds = []
            for Km, Gm in zip(K_grid, G_grid, strict=False):
                vp_p, vs_p = _predict_vp_vs_osp(
                    backend=backend,
                    matrix_K_pa=float(Km),
                    matrix_G_pa=float(Gm),
                    fluid_K_pa=float(Kf),
                    phi=phi,
                    alpha=float(a2),
                    rho_bulk_kg_m3=rho,
                )
                vp_preds.append(vp_p)
                vs_preds.append(vs_p)
            vp_preds = np.asarray(vp_preds, float)
            vs_preds = np.asarray(vs_preds, float)
            vp_mis = np.abs(vp_preds - vp_obs) / vp_obs * 100.0
            vs_mis = np.abs(vs_preds - vs_obs) / vs_obs * 100.0
            vp_p05, vp_p50, vp_p95 = _summarize_dist(vp_mis)
            vs_p05, vs_p50, vs_p95 = _summarize_dist(vs_mis)
            agg = np.sqrt(0.5 * (vp_mis**2 + vs_mis**2))
            agg_p05, agg_p50, agg_p95 = _summarize_dist(agg)
            recs.append(
                {
                    "lab_sample_id": sid,
                    "stage": stage,
                    "fluid_state": fluid_state,
                    "method": "OSP",
                    "alpha_used": float(a2),
                    "vp_misfit_pct": vp_p50,
                    "vp_misfit_pct_p05": vp_p05,
                    "vp_misfit_pct_p95": vp_p95,
                    "vs_misfit_pct": vs_p50,
                    "vs_misfit_pct_p05": vs_p05,
                    "vs_misfit_pct_p95": vs_p95,
                    "agg_misfit_pct": agg_p50,
                    "agg_misfit_pct_p05": agg_p05,
                    "agg_misfit_pct_p95": agg_p95,
                }
            )

    out_df = pd.DataFrame.from_records(recs)
    out_csv = out_dir / "vp_vs_pred_from_tc_alpha_misfit_by_sample.csv"
    out_df.to_csv(out_csv, index=False)

    # Figures: Vp misfit, Vs misfit, aggregate misfit
    _barplot_panels(
        out_dir / "vp_misfit_from_tc_alpha_bars_mt_vs_osp.png",
        out_df.rename(columns={"vp_misfit_pct_p05": "vp_misfit_pct_p05", "vp_misfit_pct_p95": "vp_misfit_pct_p95"}),
        value_col="vp_misfit_pct",
        y_label="Vp misfit (%)",
        title="Vp prediction error using TC-inverted α (HS matrix K,G grid)",
    )
    _barplot_panels(
        out_dir / "vs_misfit_from_tc_alpha_bars_mt_vs_osp.png",
        out_df,
        value_col="vs_misfit_pct",
        y_label="Vs misfit (%)",
        title="Vs prediction error using TC-inverted α (HS matrix K,G grid)",
    )
    _barplot_panels(
        out_dir / "agg_misfit_from_tc_alpha_bars_mt_vs_osp.png",
        out_df,
        value_col="agg_misfit_pct",
        y_label="Aggregate misfit (%)",
        title="Aggregate (Vp,Vs) prediction error using TC-inverted α (HS matrix K,G grid)",
    )

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
