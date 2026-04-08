from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa_elastic
from rockphysx.models.emt import gsa_transport


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _parse_samples(s: str) -> list[float]:
    s = str(s).strip().lower()
    if s in {"all", "*"}:
        return []
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("--samples must contain at least one lab_sample_id (or 'all').")
    return out


def _phi_from_row(row: pd.Series) -> float:
    if pd.notna(row.get("phi_frac")):
        return float(row["phi_frac"])
    if pd.notna(row.get("phi_pct")):
        return float(row["phi_pct"]) / 100.0
    raise ValueError("Row has neither phi_frac nor phi_pct.")


@dataclass(frozen=True)
class MatrixParams:
    lambda_w_mk: float
    vp_m_s: float
    vs_m_s: float
    rho_kg_m3: float


@dataclass(frozen=True)
class FluidParams:
    name: str
    lambda_w_mk: float
    K_gpa: float
    rho_kg_m3: float


def _matrix_KG_from_vp_vs_rho(vp_m_s: float, vs_m_s: float, rho_kg_m3: float) -> tuple[float, float]:
    vp = float(vp_m_s)
    vs = float(vs_m_s)
    rho = float(rho_kg_m3)
    G = rho * vs * vs
    K = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
    return float(K), float(G)


def _predict_tc_isotropic(matrix_lambda: float, fluid_lambda: float, phi: float, alpha: float) -> float:
    return float(
        gsa_transport.two_phase_thermal_isotropic(
            float(matrix_lambda),
            float(fluid_lambda),
            float(phi),
            aspect_ratio=float(alpha),
            comparison="matrix",
            max_iter=1,
        )
    )


def _predict_elastic_vp_vs(
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
    vp, vs = gsa_elastic.velocities_from_KG_rho(K_eff, G_eff, float(rho_bulk_kg_m3))
    return float(vp), float(vs)


def _rel_l2(errors: list[float]) -> float:
    if not errors:
        return float("nan")
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.nanmean(e * e)))

def _rel_l2_weighted(errors: list[float], sigmas: list[float]) -> float:
    if not errors:
        return float("nan")
    e = np.asarray(errors, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if e.shape != s.shape:
        raise ValueError("errors and sigmas must have same length")
    s = np.clip(s, 1e-12, np.inf)
    z = e / s
    return float(np.sqrt(np.nanmean(z * z)))


def _combine_stage_objectives(J_before: float, J_after: float) -> float:
    # stage objectives are already RMS-like; combine symmetrically
    return float(np.sqrt(0.5 * (float(J_before) ** 2 + float(J_after) ** 2)))


def _u_from_alpha(alpha: np.ndarray, alpha_min: float, alpha_max: float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    lo = np.log10(float(alpha_min))
    hi = np.log10(float(alpha_max))
    return (np.log10(a) - lo) / (hi - lo)


def _beta_weights(u: np.ndarray, m: float, kappa: float) -> np.ndarray:
    # u in (0,1), m in (0,1), kappa > 0
    u = np.asarray(u, dtype=float)
    m = float(m)
    kappa = float(kappa)
    a = max(m * kappa, 1e-8)
    b = max((1.0 - m) * kappa, 1e-8)
    logw = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    return w / np.sum(w)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have same shape")
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    j = int(np.searchsorted(cw, q, side="left"))
    j = min(max(j, 0), v.size - 1)
    return float(v[j])


@dataclass(frozen=True)
class Observations:
    tc_w_mk: float | None
    vp_m_s: float | None
    vs_m_s: float | None
    rho_bulk_kg_m3: float | None
    phi: float


def _collect_obs(rows: pd.DataFrame, *, stage: str, fluid_state: str) -> Observations | None:
    sub = rows[(rows["stage"] == stage) & (rows["fluid_state"] == fluid_state)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    tc = float(r["tc_w_mk"]) if pd.notna(r.get("tc_w_mk")) else None
    vp = float(r["vp_m_s"]) if pd.notna(r.get("vp_m_s")) else None
    vs = float(r["vs_m_s"]) if pd.notna(r.get("vs_m_s")) else None
    rho = float(r["bulk_density_g_cm3"]) * 1e3 if pd.notna(r.get("bulk_density_g_cm3")) else None
    phi = _phi_from_row(r)
    return Observations(tc_w_mk=tc, vp_m_s=vp, vs_m_s=vs, rho_bulk_kg_m3=rho, phi=phi)


@dataclass
class ForwardCache:
    # predictions per alpha for each stage and fluid_state
    alphas: np.ndarray
    stage: str
    phi: float
    tc_dry: np.ndarray
    tc_wet: np.ndarray
    vp_dry: np.ndarray
    vs_dry: np.ndarray
    vp_wet: np.ndarray
    vs_wet: np.ndarray


def _build_forward_cache_for_stage(
    *,
    rows: pd.DataFrame,
    stage: str,
    alphas: np.ndarray,
    matrix: MatrixParams,
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
) -> ForwardCache | None:
    # choose phi from any available row in this stage
    sub_stage = rows[rows["stage"] == stage]
    if sub_stage.empty:
        return None
    phi = _phi_from_row(sub_stage.iloc[0])

    K_m, G_m = _matrix_KG_from_vp_vs_rho(matrix.vp_m_s, matrix.vs_m_s, matrix.rho_kg_m3)

    tc_dry = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_dry.lambda_w_mk, phi, float(a)) for a in alphas])
    tc_wet = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_wet.lambda_w_mk, phi, float(a)) for a in alphas])

    # density differs by stage+fluid_state, so use observed rho per state if present; otherwise fallback to matrix rho
    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")
    rho_dry = float(obs_dry.rho_bulk_kg_m3) if (obs_dry is not None and obs_dry.rho_bulk_kg_m3 is not None) else float(matrix.rho_kg_m3)
    rho_wet = float(obs_wet.rho_bulk_kg_m3) if (obs_wet is not None and obs_wet.rho_bulk_kg_m3 is not None) else float(matrix.rho_kg_m3)

    vp_dry = np.full_like(alphas, np.nan, dtype=float)
    vs_dry = np.full_like(alphas, np.nan, dtype=float)
    vp_wet = np.full_like(alphas, np.nan, dtype=float)
    vs_wet = np.full_like(alphas, np.nan, dtype=float)
    for i, a in enumerate(alphas):
        vp_dry[i], vs_dry[i] = _predict_elastic_vp_vs(
            backend=backend,
            matrix_K_pa=float(K_m),
            matrix_G_pa=float(G_m),
            fluid_K_pa=float(fluid_dry.K_gpa) * 1e9,
            phi=phi,
            alpha=float(a),
            rho_bulk_kg_m3=rho_dry,
        )
        vp_wet[i], vs_wet[i] = _predict_elastic_vp_vs(
            backend=backend,
            matrix_K_pa=float(K_m),
            matrix_G_pa=float(G_m),
            fluid_K_pa=float(fluid_wet.K_gpa) * 1e9,
            phi=phi,
            alpha=float(a),
            rho_bulk_kg_m3=rho_wet,
        )

    return ForwardCache(
        alphas=np.asarray(alphas, dtype=float),
        stage=str(stage),
        phi=float(phi),
        tc_dry=tc_dry,
        tc_wet=tc_wet,
        vp_dry=vp_dry,
        vs_dry=vs_dry,
        vp_wet=vp_wet,
        vs_wet=vs_wet,
    )


def _objective_m1_for_stage(
    *,
    cache: ForwardCache,
    rows: pd.DataFrame,
    stage: str,
    mode: str,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
) -> np.ndarray:
    # returns J(alpha_i) array
    y = np.full_like(cache.alphas, np.nan, dtype=float)
    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")

    for i in range(cache.alphas.size):
        rel_tc: list[float] = []
        rel_all: list[float] = []
        sig_tc: list[float] = []
        sig_all: list[float] = []

        for fs, obs, tc_pred, vp_pred, vs_pred in (
            ("dry", obs_dry, cache.tc_dry[i], cache.vp_dry[i], cache.vs_dry[i]),
            ("wet", obs_wet, cache.tc_wet[i], cache.vp_wet[i], cache.vs_wet[i]),
        ):
            if obs is None:
                continue

            # TC
            if obs.tc_w_mk is not None:
                rel = abs(float(tc_pred) - float(obs.tc_w_mk)) / max(float(obs.tc_w_mk), 1e-12)
                rel_tc.append(rel)
                rel_all.append(rel)
                sig_tc.append(float(tc_sigma_rel))
                sig_all.append(float(tc_sigma_rel))

            if mode == "tc_vp_vs":
                if obs.vp_m_s is not None:
                    rel_all.append(abs(float(vp_pred) - float(obs.vp_m_s)) / max(float(obs.vp_m_s), 1e-12))
                    sig_all.append(float(vp_sigma_rel))
                if obs.vs_m_s is not None:
                    rel_all.append(abs(float(vs_pred) - float(obs.vs_m_s)) / max(float(obs.vs_m_s), 1e-12))
                    sig_all.append(float(vs_sigma_rel))

        if mode == "tc_only":
            y[i] = _rel_l2_weighted(rel_tc, sig_tc)
        else:
            y[i] = _rel_l2_weighted(rel_all, sig_all)

    return y


def _objective_m2_for_stage(
    *,
    cache: ForwardCache,
    rows: pd.DataFrame,
    stage: str,
    mode: str,
    weights: np.ndarray,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
) -> float:
    # weighted average predictions across alpha grid
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")

    rel_tc: list[float] = []
    rel_all: list[float] = []
    sig_tc: list[float] = []
    sig_all: list[float] = []

    pred_tc_dry = float(np.sum(w * cache.tc_dry))
    pred_tc_wet = float(np.sum(w * cache.tc_wet))
    pred_vp_dry = float(np.sum(w * cache.vp_dry))
    pred_vs_dry = float(np.sum(w * cache.vs_dry))
    pred_vp_wet = float(np.sum(w * cache.vp_wet))
    pred_vs_wet = float(np.sum(w * cache.vs_wet))

    for fs, obs, tc_pred, vp_pred, vs_pred in (
        ("dry", obs_dry, pred_tc_dry, pred_vp_dry, pred_vs_dry),
        ("wet", obs_wet, pred_tc_wet, pred_vp_wet, pred_vs_wet),
    ):
        if obs is None:
            continue
        if obs.tc_w_mk is not None:
            rel = abs(float(tc_pred) - float(obs.tc_w_mk)) / max(float(obs.tc_w_mk), 1e-12)
            rel_tc.append(rel)
            rel_all.append(rel)
            sig_tc.append(float(tc_sigma_rel))
            sig_all.append(float(tc_sigma_rel))
        if mode == "tc_vp_vs":
            if obs.vp_m_s is not None:
                rel_all.append(abs(float(vp_pred) - float(obs.vp_m_s)) / max(float(obs.vp_m_s), 1e-12))
                sig_all.append(float(vp_sigma_rel))
            if obs.vs_m_s is not None:
                rel_all.append(abs(float(vs_pred) - float(obs.vs_m_s)) / max(float(obs.vs_m_s), 1e-12))
                sig_all.append(float(vs_sigma_rel))

    if mode == "tc_only":
        return _rel_l2_weighted(rel_tc, sig_tc)
    return _rel_l2_weighted(rel_all, sig_all)


def _invert_m2_grid(
    *,
    cache: ForwardCache,
    rows: pd.DataFrame,
    stage: str,
    mode: str,
    u: np.ndarray,
    m_grid: np.ndarray,
    kappa_grid: np.ndarray,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
) -> dict[str, float]:
    best = {"J": float("inf"), "m": float("nan"), "kappa": float("nan")}
    for m in m_grid:
        for kappa in kappa_grid:
            w = _beta_weights(u, float(m), float(kappa))
            J = float(
                _objective_m2_for_stage(
                    cache=cache,
                    rows=rows,
                    stage=stage,
                    mode=mode,
                    weights=w,
                    tc_sigma_rel=tc_sigma_rel,
                    vp_sigma_rel=vp_sigma_rel,
                    vs_sigma_rel=vs_sigma_rel,
                )
            )
            if np.isfinite(J) and J < best["J"]:
                best = {"J": J, "m": float(m), "kappa": float(kappa)}
    # derived summaries of alpha distribution
    w_best = _beta_weights(u, best["m"], best["kappa"])
    a_p50 = _weighted_quantile(cache.alphas, w_best, 0.50)
    a_p05 = _weighted_quantile(cache.alphas, w_best, 0.05)
    a_p95 = _weighted_quantile(cache.alphas, w_best, 0.95)
    return {
        "J": float(best["J"]),
        "m": float(best["m"]),
        "kappa": float(best["kappa"]),
        "alpha_p50": float(a_p50),
        "alpha_p05": float(a_p05),
        "alpha_p95": float(a_p95),
    }


def _plot_summary(
    out_png: Path,
    df: pd.DataFrame,
    *,
    mode: str,
) -> None:
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }

    tmp = df[df["mode"] == mode].copy()
    if tmp.empty:
        return

    # stable order: increasing phi_before
    tmp = tmp.sort_values(["phi_before_pct"]).reset_index(drop=True)
    x = np.arange(tmp.shape[0])

    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)

        # delta_z
        ax1.axhline(0.0, color="0.2", lw=1.0, ls="--", alpha=0.4)
        ax1.scatter(x, tmp["m1_delta_z"], s=70, label="M1", color="C0")
        ax1.scatter(x, tmp["m2_delta_z_p50"], s=70, label="M2 (p50)", color="C3")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(s) for s in tmp["lab_sample_id"]])
        ax1.set_ylabel(r"$\Delta z=\log_{10}\alpha_{after}-\log_{10}\alpha_{before}$")
        ax1.set_title("Crack-like shift summary")
        ax1.grid(True, alpha=0.25)
        ax1.legend(frameon=False, loc="best")

        # alpha before/after (p50 for M2)
        ax2.scatter(tmp["m1_alpha_before"], tmp["m1_alpha_after"], s=70, label="M1", color="C0")
        ax2.scatter(tmp["m2_alpha_before_p50"], tmp["m2_alpha_after_p50"], s=70, label="M2 (p50)", color="C3")
        lo = min(tmp["m1_alpha_before"].min(), tmp["m1_alpha_after"].min(), tmp["m2_alpha_before_p50"].min(), tmp["m2_alpha_after_p50"].min())
        hi = max(tmp["m1_alpha_before"].max(), tmp["m1_alpha_after"].max(), tmp["m2_alpha_before_p50"].max(), tmp["m2_alpha_after_p50"].max())
        lo = max(float(lo) / 1.5, 1e-6)
        hi = float(hi) * 1.5
        ax2.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, ls="--", alpha=0.4)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.set_xlabel(r"$\alpha_{before}$")
        ax2.set_ylabel(r"$\alpha_{after}$")
        ax2.set_title("Before vs after")
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend(frameon=False, loc="best")

        fig.suptitle(f"OSP/GSA inversion summary ({mode})")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_misfit_detail(
    out_png: Path,
    misfit_df: pd.DataFrame,
    *,
    mode: str,
) -> None:
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    tmp = misfit_df[misfit_df["mode"] == mode].copy()
    if tmp.empty:
        return

    if mode == "tc_only":
        categories = ["TC_dry", "TC_wet"]
    else:
        categories = ["TC_dry", "TC_wet", "VP_dry", "VP_wet", "VS_dry", "VS_wet"]

    # Stable row ordering: by increasing phi_before_pct (from M1_M2 rows), then stage
    order = (
        tmp[["lab_sample_id", "phi_before_pct"]]
        .drop_duplicates()
        .sort_values(["phi_before_pct", "lab_sample_id"])
        .reset_index(drop=True)
    )
    samples = [float(x) for x in order["lab_sample_id"].tolist()]
    row_keys: list[tuple[float, str]] = []
    row_labels: list[str] = []
    for sid in samples:
        for stage in ("before", "after"):
            row_keys.append((sid, stage))
            row_labels.append(f"{sid:g} {stage}")

    models = ["M1", "M2"]
    height = max(6.0, 0.34 * len(row_keys))
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(13.6, height), constrained_layout=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        vmax = float(np.nanpercentile(tmp["misfit_pct"].to_numpy(float), 95))
        vmax = max(vmax, 10.0)
        vmin = 0.0

        for ax, model in zip(axes, models, strict=True):
            sub = tmp[tmp["model"] == model]
            mat = np.full((len(row_keys), len(categories)), np.nan, dtype=float)
            for i, (sid, stage) in enumerate(row_keys):
                for j, cat in enumerate(categories):
                    v = sub[
                        (sub["lab_sample_id"] == sid)
                        & (sub["stage"] == stage)
                        & (sub["category"] == cat)
                    ]["misfit_pct"]
                    if not v.empty:
                        mat[i, j] = float(v.iloc[0])

            # Pastel-ish sequential palette
            im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="PuBuGn")
            ax.set_title(f"{model} misfit (%)")
            ax.set_xticks(np.arange(len(categories)))
            ax.set_xticklabels(categories, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.grid(False)

            # annotate
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if not np.isfinite(val):
                        continue
                    color = "white" if val > 0.62 * vmax else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9)

            # 5% target guide (visual cue via title note)
            ax.text(
                0.99,
                -0.10,
                "target: ≤ 5%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color="0.25",
            )

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
        cbar.set_label("misfit (%)")
        fig.suptitle(f"Per-property misfit vs experiments ({mode})")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="M1+M2 inversion for OSP/GSA (TC and TC+Vp+Vs), by sample.")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet-measurements", type=str, default="measurements_long")
    ap.add_argument("--sheet-stage", type=str, default="sample_stage")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    ap.add_argument("--samples", type=str, default="22.2,18.2,15.0", help="Comma-separated lab_sample_id list, or 'all'.")

    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=61)

    # Matrix uncertainty search (HS-feasible interval)
    ap.add_argument("--matrix-search", choices=("fixed", "grid"), default="grid")
    ap.add_argument("--matrix-lambda-min", type=float, default=2.72)
    ap.add_argument("--matrix-lambda-max", type=float, default=2.98)
    ap.add_argument("--matrix-vp-min", type=float, default=5506.0)
    ap.add_argument("--matrix-vp-max", type=float, default=7063.0)
    ap.add_argument("--matrix-vs-min", type=float, default=3153.0)
    ap.add_argument("--matrix-vs-max", type=float, default=3909.0)
    ap.add_argument("--matrix-grid-n", type=int, default=3, help="Grid points per matrix parameter for --matrix-search grid.")

    ap.add_argument("--m-step", type=float, default=0.05)
    ap.add_argument("--kappa-grid", type=str, default="2,5,10,20,50,100")

    # matrix (HS feasible set p50)
    ap.add_argument("--matrix-lambda", type=float, default=2.85)
    ap.add_argument("--matrix-vp-m-s", type=float, default=6282.0)
    ap.add_argument("--matrix-vs-m-s", type=float, default=3531.0)
    ap.add_argument("--matrix-rho-kg-m3", type=float, default=2720.0)

    # fluids for TC/elastic
    ap.add_argument("--air-lambda", type=float, default=0.03)
    ap.add_argument("--air-K-gpa", type=float, default=0.0001)
    ap.add_argument("--air-rho-kg-m3", type=float, default=1.2)

    ap.add_argument("--brine-lambda", type=float, default=0.60)
    ap.add_argument("--brine-K-gpa", type=float, default=2.20)
    ap.add_argument("--brine-rho-kg-m3", type=float, default=1030.0)

    # Relative measurement uncertainties (used as weights)
    ap.add_argument("--tc-sigma-rel", type=float, default=0.025, help="Relative sigma for TC (e.g. 0.025 = 2.5%).")
    ap.add_argument("--vp-sigma-rel", type=float, default=0.05, help="Relative sigma for Vp (e.g. 0.05 = 5%).")
    ap.add_argument("--vs-sigma-rel", type=float, default=0.05, help="Relative sigma for Vs (e.g. 0.05 = 5%).")

    # elastic backend
    ap.add_argument("--green-fortran", type=Path, default=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"))
    ap.add_argument("--elastic-backend-so", type=Path, default=Path("test_new/gsa/plots/libgsa_elastic_fortran.so"))

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    dfm = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet_measurements))
    dfm["lab_sample_id"] = pd.to_numeric(dfm["lab_sample_id"], errors="coerce")
    dfm["stage"] = dfm["stage"].astype(str).str.strip()
    dfm["fluid_state"] = dfm["fluid_state"].astype(str).str.strip()
    dfs = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet_stage))
    dfs["lab_sample_id"] = pd.to_numeric(dfs["lab_sample_id"], errors="coerce")
    if "stage" in dfs.columns:
        dfs["stage"] = dfs["stage"].astype(str).str.strip()

    samples = _parse_samples(args.samples)
    if not samples:
        samples = sorted([float(x) for x in dfm["lab_sample_id"].dropna().unique().tolist()])

    alphas = np.logspace(np.log10(float(args.alpha_min)), np.log10(float(args.alpha_max)), int(args.alpha_n))
    u = _u_from_alpha(alphas, float(args.alpha_min), float(args.alpha_max))
    # avoid exact 0/1 in beta weights
    eps = 0.5 / float(args.alpha_n)
    u = np.clip(u, eps, 1.0 - eps)

    m_step = float(args.m_step)
    m_grid = np.arange(m_step, 1.0, m_step)
    kappa_grid = np.array([float(x) for x in str(args.kappa_grid).split(",") if str(x).strip()], dtype=float)

    matrix_fixed = MatrixParams(
        lambda_w_mk=float(args.matrix_lambda),
        vp_m_s=float(args.matrix_vp_m_s),
        vs_m_s=float(args.matrix_vs_m_s),
        rho_kg_m3=float(args.matrix_rho_kg_m3),
    )
    fluid_dry = FluidParams(
        name="air",
        lambda_w_mk=float(args.air_lambda),
        K_gpa=float(args.air_K_gpa),
        rho_kg_m3=float(args.air_rho_kg_m3),
    )
    fluid_wet = FluidParams(
        name="brine",
        lambda_w_mk=float(args.brine_lambda),
        K_gpa=float(args.brine_K_gpa),
        rho_kg_m3=float(args.brine_rho_kg_m3),
    )

    backend = gsa_elastic.build_backend(
        green_fortran=args.green_fortran,
        output_library=args.elastic_backend_so,
        force_rebuild=False,
    )

    rows_out: list[dict[str, float | str]] = []
    misfit_rows: list[dict[str, float | str]] = []

    grid_n = int(args.matrix_grid_n)
    lam_grid = np.linspace(float(args.matrix_lambda_min), float(args.matrix_lambda_max), grid_n)
    vp_grid = np.linspace(float(args.matrix_vp_min), float(args.matrix_vp_max), grid_n)
    vs_grid = np.linspace(float(args.matrix_vs_min), float(args.matrix_vs_max), grid_n)

    for sid in samples:
        rows = dfm[dfm["lab_sample_id"] == float(sid)].copy()
        if rows.empty:
            print(f"WARNING: sample {sid:g} not found in data; skipping.")
            continue

        # permeability (before only)
        stage_row = dfs[(dfs["lab_sample_id"] == float(sid)) & (dfs["stage"] == "before")]
        perm_md = float(stage_row.iloc[0]["permeability_md_modeled"]) if (not stage_row.empty and pd.notna(stage_row.iloc[0].get("permeability_md_modeled"))) else float("nan")

        phi_before = float(rows[rows["stage"] == "before"].iloc[0]["phi_pct"])
        phi_after = float(rows[rows["stage"] == "after"].iloc[0]["phi_pct"])

        for mode in ("tc_only", "tc_vp_vs"):
            # -----------------------------------------------------------------
            # Optionally search matrix within HS interval (shared for before/after)
            # -----------------------------------------------------------------
            best_matrix = matrix_fixed
            best_cache_before: ForwardCache | None = None
            best_cache_after: ForwardCache | None = None
            best_m1 = {
                "J_total": float("inf"),
                "Jb": float("nan"),
                "Ja": float("nan"),
                "ib": None,
                "ia": None,
            }

            if str(args.matrix_search) == "fixed":
                cand_mats = [matrix_fixed]
            else:
                cand_mats = [
                    MatrixParams(lambda_w_mk=float(lam), vp_m_s=float(vp), vs_m_s=float(vs), rho_kg_m3=float(matrix_fixed.rho_kg_m3))
                    for lam in lam_grid
                    for vp in vp_grid
                    for vs in vs_grid
                ]

            for mat in cand_mats:
                # sanity: derived K must be positive
                K_m, G_m = _matrix_KG_from_vp_vs_rho(mat.vp_m_s, mat.vs_m_s, mat.rho_kg_m3)
                if not (np.isfinite(K_m) and np.isfinite(G_m) and K_m > 0 and G_m > 0):
                    continue

                cache_before = _build_forward_cache_for_stage(
                    rows=rows,
                    stage="before",
                    alphas=alphas,
                    matrix=mat,
                    fluid_dry=fluid_dry,
                    fluid_wet=fluid_wet,
                    backend=backend,
                )
                cache_after = _build_forward_cache_for_stage(
                    rows=rows,
                    stage="after",
                    alphas=alphas,
                    matrix=mat,
                    fluid_dry=fluid_dry,
                    fluid_wet=fluid_wet,
                    backend=backend,
                )
                if cache_before is None or cache_after is None:
                    continue

                Jb_arr = _objective_m1_for_stage(
                    cache=cache_before,
                    rows=rows,
                    stage="before",
                    mode=mode,
                    tc_sigma_rel=float(args.tc_sigma_rel),
                    vp_sigma_rel=float(args.vp_sigma_rel),
                    vs_sigma_rel=float(args.vs_sigma_rel),
                )
                Ja_arr = _objective_m1_for_stage(
                    cache=cache_after,
                    rows=rows,
                    stage="after",
                    mode=mode,
                    tc_sigma_rel=float(args.tc_sigma_rel),
                    vp_sigma_rel=float(args.vp_sigma_rel),
                    vs_sigma_rel=float(args.vs_sigma_rel),
                )
                ib = int(np.nanargmin(Jb_arr))
                ia = int(np.nanargmin(Ja_arr))
                Jb = float(Jb_arr[ib])
                Ja = float(Ja_arr[ia])
                J_total = _combine_stage_objectives(Jb, Ja)

                if np.isfinite(J_total) and J_total < float(best_m1["J_total"]):
                    best_m1 = {"J_total": J_total, "Jb": Jb, "Ja": Ja, "ib": ib, "ia": ia}
                    best_matrix = mat
                    best_cache_before = cache_before
                    best_cache_after = cache_after

            if best_cache_before is None or best_cache_after is None or best_m1["ib"] is None or best_m1["ia"] is None:
                print(f"WARNING: sample {sid:g} missing before/after or invalid matrix; skipping mode {mode}.")
                continue

            # M1
            ib = int(best_m1["ib"])
            ia = int(best_m1["ia"])
            a_b = float(alphas[ib])
            a_a = float(alphas[ia])
            z_b = float(np.log10(a_b))
            z_a = float(np.log10(a_a))

            # M2 (grid)
            m2b = _invert_m2_grid(
                cache=best_cache_before,
                rows=rows,
                stage="before",
                mode=mode,
                u=u,
                m_grid=m_grid,
                kappa_grid=kappa_grid,
                tc_sigma_rel=float(args.tc_sigma_rel),
                vp_sigma_rel=float(args.vp_sigma_rel),
                vs_sigma_rel=float(args.vs_sigma_rel),
            )
            m2a = _invert_m2_grid(
                cache=best_cache_after,
                rows=rows,
                stage="after",
                mode=mode,
                u=u,
                m_grid=m_grid,
                kappa_grid=kappa_grid,
                tc_sigma_rel=float(args.tc_sigma_rel),
                vp_sigma_rel=float(args.vp_sigma_rel),
                vs_sigma_rel=float(args.vs_sigma_rel),
            )

            # report derived matrix moduli for transparency
            K_m, G_m = _matrix_KG_from_vp_vs_rho(best_matrix.vp_m_s, best_matrix.vs_m_s, best_matrix.rho_kg_m3)

            rows_out.append(
                {
                    "lab_sample_id": float(sid),
                    "mode": mode,
                    "phi_before_pct": float(phi_before),
                    "phi_after_pct": float(phi_after),
                    "perm_before_md_modeled": float(perm_md),
                    "matrix_lambda_w_mk": float(best_matrix.lambda_w_mk),
                    "matrix_vp_m_s": float(best_matrix.vp_m_s),
                    "matrix_vs_m_s": float(best_matrix.vs_m_s),
                    "matrix_rho_kg_m3": float(best_matrix.rho_kg_m3),
                    "matrix_K_gpa": float(K_m) / 1e9,
                    "matrix_G_gpa": float(G_m) / 1e9,
                    "m1_alpha_before": float(a_b),
                    "m1_alpha_after": float(a_a),
                    "m1_z_before": float(z_b),
                    "m1_z_after": float(z_a),
                    "m1_delta_z": float(z_a - z_b),
                    "m1_J_before": float(best_m1["Jb"]),
                    "m1_J_after": float(best_m1["Ja"]),
                    "m1_J_total": float(best_m1["J_total"]),
                    "m2_m_before": float(m2b["m"]),
                    "m2_kappa_before": float(m2b["kappa"]),
                    "m2_alpha_before_p50": float(m2b["alpha_p50"]),
                    "m2_alpha_before_p05": float(m2b["alpha_p05"]),
                    "m2_alpha_before_p95": float(m2b["alpha_p95"]),
                    "m2_J_before": float(m2b["J"]),
                    "m2_m_after": float(m2a["m"]),
                    "m2_kappa_after": float(m2a["kappa"]),
                    "m2_alpha_after_p50": float(m2a["alpha_p50"]),
                    "m2_alpha_after_p05": float(m2a["alpha_p05"]),
                    "m2_alpha_after_p95": float(m2a["alpha_p95"]),
                    "m2_J_after": float(m2a["J"]),
                    "m2_z_before_p50": float(np.log10(float(m2b["alpha_p50"]))),
                    "m2_z_after_p50": float(np.log10(float(m2a["alpha_p50"]))),
                    "m2_delta_z_p50": float(np.log10(float(m2a["alpha_p50"])) - np.log10(float(m2b["alpha_p50"]))),
                }
            )

            # -----------------------------------------------------------------
            # Detailed per-property misfit tables (vs experiments)
            # -----------------------------------------------------------------
            def _add_misfit_rows_for_model(
                *,
                model: str,
                stage: str,
                obs_rows: pd.DataFrame,
                cache: ForwardCache,
                alpha_used: float,
                pred_tc_dry: float,
                pred_tc_wet: float,
                pred_vp_dry: float,
                pred_vs_dry: float,
                pred_vp_wet: float,
                pred_vs_wet: float,
            ) -> None:
                for fluid_state, pred_tc, pred_vp, pred_vs in (
                    ("dry", pred_tc_dry, pred_vp_dry, pred_vs_dry),
                    ("wet", pred_tc_wet, pred_vp_wet, pred_vs_wet),
                ):
                    obs = _collect_obs(obs_rows, stage=stage, fluid_state=fluid_state)
                    if obs is None:
                        continue

                    def add(category: str, obs_val: float | None, pred_val: float) -> None:
                        if obs_val is None:
                            return
                        mis = 100.0 * abs(float(pred_val) - float(obs_val)) / max(float(obs_val), 1e-12)
                        misfit_rows.append(
                            {
                                "lab_sample_id": float(sid),
                                "mode": str(mode),
                                "model": str(model),
                                "stage": str(stage),
                                "fluid_state": str(fluid_state),
                                "category": str(category),
                                "phi_before_pct": float(phi_before),
                                "phi_after_pct": float(phi_after),
                                "perm_before_md_modeled": float(perm_md),
                                "alpha_used": float(alpha_used),
                                "misfit_pct": float(mis),
                                "obs": float(obs_val),
                                "pred": float(pred_val),
                            }
                        )

                    add(f"TC_{fluid_state}", obs.tc_w_mk, float(pred_tc))
                    if mode == "tc_vp_vs":
                        add(f"VP_{fluid_state}", obs.vp_m_s, float(pred_vp))
                        add(f"VS_{fluid_state}", obs.vs_m_s, float(pred_vs))

            # M1 detailed (uses grid-point predictions at argmin indices)
            _add_misfit_rows_for_model(
                model="M1",
                stage="before",
                obs_rows=rows,
                cache=best_cache_before,
                alpha_used=float(a_b),
                pred_tc_dry=float(best_cache_before.tc_dry[ib]),
                pred_tc_wet=float(best_cache_before.tc_wet[ib]),
                pred_vp_dry=float(best_cache_before.vp_dry[ib]),
                pred_vs_dry=float(best_cache_before.vs_dry[ib]),
                pred_vp_wet=float(best_cache_before.vp_wet[ib]),
                pred_vs_wet=float(best_cache_before.vs_wet[ib]),
            )
            _add_misfit_rows_for_model(
                model="M1",
                stage="after",
                obs_rows=rows,
                cache=best_cache_after,
                alpha_used=float(a_a),
                pred_tc_dry=float(best_cache_after.tc_dry[ia]),
                pred_tc_wet=float(best_cache_after.tc_wet[ia]),
                pred_vp_dry=float(best_cache_after.vp_dry[ia]),
                pred_vs_dry=float(best_cache_after.vs_dry[ia]),
                pred_vp_wet=float(best_cache_after.vp_wet[ia]),
                pred_vs_wet=float(best_cache_after.vs_wet[ia]),
            )

            # M2 detailed (use weighted-average predictions from beta distribution)
            wb = _beta_weights(u, float(m2b["m"]), float(m2b["kappa"]))
            wa = _beta_weights(u, float(m2a["m"]), float(m2a["kappa"]))
            _add_misfit_rows_for_model(
                model="M2",
                stage="before",
                obs_rows=rows,
                cache=best_cache_before,
                alpha_used=float(m2b["alpha_p50"]),
                pred_tc_dry=float(np.sum(wb * best_cache_before.tc_dry)),
                pred_tc_wet=float(np.sum(wb * best_cache_before.tc_wet)),
                pred_vp_dry=float(np.sum(wb * best_cache_before.vp_dry)),
                pred_vs_dry=float(np.sum(wb * best_cache_before.vs_dry)),
                pred_vp_wet=float(np.sum(wb * best_cache_before.vp_wet)),
                pred_vs_wet=float(np.sum(wb * best_cache_before.vs_wet)),
            )
            _add_misfit_rows_for_model(
                model="M2",
                stage="after",
                obs_rows=rows,
                cache=best_cache_after,
                alpha_used=float(m2a["alpha_p50"]),
                pred_tc_dry=float(np.sum(wa * best_cache_after.tc_dry)),
                pred_tc_wet=float(np.sum(wa * best_cache_after.tc_wet)),
                pred_vp_dry=float(np.sum(wa * best_cache_after.vp_dry)),
                pred_vs_dry=float(np.sum(wa * best_cache_after.vs_dry)),
                pred_vp_wet=float(np.sum(wa * best_cache_after.vp_wet)),
                pred_vs_wet=float(np.sum(wa * best_cache_after.vs_wet)),
            )

    out_df = pd.DataFrame(rows_out)
    misfit_df = pd.DataFrame(misfit_rows)
    out_xlsx = out_dir / "gsa_tc_elastic_m1_m2_results.xlsx"
    # Use openpyxl by default (xlsxwriter may be missing in some environments).
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name="M1_M2", index=False)
        misfit_df.to_excel(w, sheet_name="misfit_detail", index=False)

    _plot_summary(out_dir / "gsa_m1_m2_summary_tc_only.png", out_df, mode="tc_only")
    _plot_summary(out_dir / "gsa_m1_m2_summary_tc_vp_vs.png", out_df, mode="tc_vp_vs")
    _plot_misfit_detail(out_dir / "gsa_misfit_detail_tc_only.png", misfit_df, mode="tc_only")
    _plot_misfit_detail(out_dir / "gsa_misfit_detail_tc_vp_vs.png", misfit_df, mode="tc_vp_vs")

    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_dir / 'gsa_m1_m2_summary_tc_only.png'}")
    print(f"Saved: {out_dir / 'gsa_m1_m2_summary_tc_vp_vs.png'}")
    print(f"Saved: {out_dir / 'gsa_misfit_detail_tc_only.png'}")
    print(f"Saved: {out_dir / 'gsa_misfit_detail_tc_vp_vs.png'}")


if __name__ == "__main__":
    main()
