from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa_elastic
from rockphysx.models.emt import gsa_transport


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num) / float(den)


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


def _collect_obs(rows: pd.DataFrame, *, stage: str, fluid_state: str) -> dict[str, float] | None:
    sub = rows[(rows["stage"] == stage) & (rows["fluid_state"] == fluid_state)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    out: dict[str, float] = {"phi": _phi_from_row(r)}
    if pd.notna(r.get("tc_w_mk")):
        out["tc"] = float(r["tc_w_mk"])
    if pd.notna(r.get("vp_m_s")):
        out["vp"] = float(r["vp_m_s"])
    if pd.notna(r.get("vs_m_s")):
        out["vs"] = float(r["vs_m_s"])
    if pd.notna(r.get("bulk_density_g_cm3")):
        out["rho"] = float(r["bulk_density_g_cm3"]) * 1e3
    return out


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


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.nanmax(a))
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.nansum(np.exp(a - m))))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    j = int(np.searchsorted(cw, q, side="left"))
    j = min(max(j, 0), v.size - 1)
    return float(v[j])


def _u_from_alpha(alpha: np.ndarray, alpha_min: float, alpha_max: float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    lo = np.log10(float(alpha_min))
    hi = np.log10(float(alpha_max))
    u = (np.log10(a) - lo) / (hi - lo)
    eps = 1e-8
    return np.clip(u, eps, 1.0 - eps)


def _beta_weights(u: np.ndarray, m: float, kappa: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    m = float(m)
    kappa = float(kappa)
    a = max(m * kappa, 1e-8)
    b = max((1.0 - m) * kappa, 1e-8)
    logw = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u)
    logw = logw - float(np.max(logw))
    w = np.exp(logw)
    return w / float(np.sum(w))


def _build_combo_grid(*, m_step: float, kappa_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m_grid = np.arange(float(m_step), 1.0, float(m_step))
    if m_grid.size < 1:
        raise ValueError("m_step produces empty grid.")
    combos: list[tuple[float, float]] = []
    for m in m_grid:
        for k in kappa_grid:
            combos.append((float(m), float(k)))
    return np.array([c[0] for c in combos], dtype=float), np.array([c[1] for c in combos], dtype=float)


@dataclass
class ForwardCache:
    alphas: np.ndarray
    phi: float
    tc_dry: np.ndarray
    tc_wet: np.ndarray
    vp_dry: np.ndarray | None
    vs_dry: np.ndarray | None
    vp_wet: np.ndarray | None
    vs_wet: np.ndarray | None


def _build_forward_cache_for_stage(
    *,
    rows: pd.DataFrame,
    stage: str,
    alphas: np.ndarray,
    matrix: MatrixParams,
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
    need_elastic: bool,
) -> ForwardCache | None:
    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")
    if obs_dry is None and obs_wet is None:
        return None

    phi = float(obs_dry["phi"] if obs_dry is not None else obs_wet["phi"])  # type: ignore[index]
    K_m, G_m = _matrix_KG_from_vp_vs_rho(matrix.vp_m_s, matrix.vs_m_s, matrix.rho_kg_m3)

    tc_dry = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_dry.lambda_w_mk, phi, float(a)) for a in alphas])
    tc_wet = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_wet.lambda_w_mk, phi, float(a)) for a in alphas])

    if not need_elastic:
        return ForwardCache(
            alphas=np.asarray(alphas, dtype=float),
            phi=float(phi),
            tc_dry=tc_dry,
            tc_wet=tc_wet,
            vp_dry=None,
            vs_dry=None,
            vp_wet=None,
            vs_wet=None,
        )

    rho_dry = float(obs_dry["rho"]) if (obs_dry is not None and "rho" in obs_dry) else float(matrix.rho_kg_m3)
    rho_wet = float(obs_wet["rho"]) if (obs_wet is not None and "rho" in obs_wet) else float(matrix.rho_kg_m3)

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
        phi=float(phi),
        tc_dry=tc_dry,
        tc_wet=tc_wet,
        vp_dry=vp_dry,
        vs_dry=vs_dry,
        vp_wet=vp_wet,
        vs_wet=vs_wet,
    )


def _stage_loglike_over_combos(
    *,
    stage: str,
    rows: pd.DataFrame,
    cache: ForwardCache,
    W: np.ndarray,  # (n_combo, n_alpha)
    mode: str,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
) -> np.ndarray:
    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")
    if obs_dry is None and obs_wet is None:
        return np.full((W.shape[0],), float("-inf"), dtype=float)

    # weighted predictions per combo
    pred_tc_dry = W @ cache.tc_dry
    pred_tc_wet = W @ cache.tc_wet

    ll = np.zeros((W.shape[0],), dtype=float)

    def add_rel_gauss(pred: np.ndarray, obs: float | None, sigma_rel: float) -> None:
        if obs is None or not np.isfinite(obs) or obs == 0:
            return
        e_rel = (pred - float(obs)) / float(obs)
        z = e_rel / max(float(sigma_rel), 1e-12)
        ll[:] += -0.5 * (z * z)

    if obs_dry is not None:
        add_rel_gauss(pred_tc_dry, float(obs_dry["tc"]) if "tc" in obs_dry else None, tc_sigma_rel)
    if obs_wet is not None:
        add_rel_gauss(pred_tc_wet, float(obs_wet["tc"]) if "tc" in obs_wet else None, tc_sigma_rel)

    if mode == "tc_only":
        return ll

    if cache.vp_dry is None or cache.vs_dry is None or cache.vp_wet is None or cache.vs_wet is None:
        return np.full((W.shape[0],), float("-inf"), dtype=float)

    pred_vp_dry = W @ np.asarray(cache.vp_dry, dtype=float)
    pred_vs_dry = W @ np.asarray(cache.vs_dry, dtype=float)
    pred_vp_wet = W @ np.asarray(cache.vp_wet, dtype=float)
    pred_vs_wet = W @ np.asarray(cache.vs_wet, dtype=float)

    if obs_dry is not None:
        add_rel_gauss(pred_vp_dry, float(obs_dry["vp"]) if "vp" in obs_dry else None, vp_sigma_rel)
        add_rel_gauss(pred_vs_dry, float(obs_dry["vs"]) if "vs" in obs_dry else None, vs_sigma_rel)
    if obs_wet is not None:
        add_rel_gauss(pred_vp_wet, float(obs_wet["vp"]) if "vp" in obs_wet else None, vp_sigma_rel)
        add_rel_gauss(pred_vs_wet, float(obs_wet["vs"]) if "vs" in obs_wet else None, vs_sigma_rel)
    return ll


def _auto_pick_diverse_samples(df: pd.DataFrame, *, mode: str, n: int = 4) -> list[float]:
    sub = df[df["mode"] == mode].copy()
    sub = sub[np.isfinite(pd.to_numeric(sub["phi_before_pct"], errors="coerce"))]
    if sub.empty:
        return []
    sub["phi_before_pct"] = pd.to_numeric(sub["phi_before_pct"], errors="coerce").astype(float)
    sub = sub.sort_values("phi_before_pct").reset_index(drop=True)
    idxs = {0, len(sub) - 1}
    if len(sub) >= 4:
        idxs.add(int(round(0.33 * (len(sub) - 1))))
        idxs.add(int(round(0.66 * (len(sub) - 1))))
    picked = sub.iloc[sorted(idxs)]["lab_sample_id"].tolist()
    out: list[float] = []
    for x in picked:
        v = float(x)
        if v not in out:
            out.append(v)
        if len(out) >= n:
            break
    return out


def _posterior_alpha_from_combo_marginal(W: np.ndarray, w_combo: np.ndarray) -> np.ndarray:
    w = np.asarray(w_combo, dtype=float)
    if w.ndim != 1:
        raise ValueError("w_combo must be 1D.")
    w = np.clip(w, 0.0, np.inf)
    if not np.isfinite(w).any() or float(np.sum(w)) <= 0.0:
        return np.full((W.shape[1],), float("nan"), dtype=float)
    w = w / float(np.sum(w))
    p = w @ W  # (n_alpha,)
    p = np.clip(p, 0.0, np.inf)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        return np.full((W.shape[1],), float("nan"), dtype=float)
    return p / s


def _plot_posterior_alpha_panels(
    out_png: Path,
    *,
    posterior_alpha: pd.DataFrame,
    sample_ids: list[float],
    mode: str,
) -> None:
    import matplotlib.pyplot as plt

    sub = posterior_alpha[(posterior_alpha["mode"] == mode) & (posterior_alpha["lab_sample_id"].isin(sample_ids))].copy()
    if sub.empty:
        return

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    with plt.rc_context(rc):
        fig, axs = plt.subplots(2, 2, figsize=(14.5, 8.0), constrained_layout=True)
        axs = axs.ravel()
        for ax, sid in zip(axs, sample_ids, strict=False):
            tmp = sub[sub["lab_sample_id"] == float(sid)].copy()
            if tmp.empty:
                ax.axis("off")
                continue
            tmp = tmp.sort_values("alpha")
            for stage, color in (("before", "C0"), ("after", "C3")):
                s2 = tmp[tmp["stage"] == stage]
                if s2.empty:
                    continue
                ax.plot(s2["alpha"], s2["prob"], color=color, lw=2.2, label=stage.capitalize())
                # mean line (in log space, consistent with Δz definition)
                z = np.log10(s2["alpha"].to_numpy(float))
                w = s2["prob"].to_numpy(float)
                if np.isfinite(z).all() and np.isfinite(w).all() and float(np.sum(w)) > 0:
                    z_mean = float(np.sum(z * w) / np.sum(w))
                    ax.axvline(10 ** z_mean, color=color, ls="--", lw=1.2, alpha=0.8)

            ax.set_xscale("log")
            ax.set_xlabel(r"Aspect ratio $\alpha$")
            ax.set_ylabel("Posterior density")
            ax.set_title(f"Sample {sid:g}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", frameon=True)

        for ax in axs[len(sample_ids) :]:
            ax.axis("off")

        fig.suptitle(f"Bayesian M2: posterior p(α) before vs after (mode={mode})", y=1.02, fontsize=16)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_measured_vs_predicted_panels(
    out_png: Path,
    *,
    summary: pd.DataFrame,
    misfit_detail: pd.DataFrame,
    prop: str,
    mode: str,
) -> None:
    """
    Creates a 2x2 figure (M2 Bayesian only):
      rows: M2 (single row replicated for style consistency)
      cols: dry / wet

    Uses the ±5/10/20% bands style from MT plots.
    """
    import matplotlib.pyplot as plt

    tmp = misfit_detail[(misfit_detail["mode"] == mode) & (misfit_detail["property"] == prop)].copy()
    if tmp.empty:
        return

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    def add_bands(ax: plt.Axes) -> None:  # type: ignore[name-defined]
        xs = np.linspace(0.0, 1.0, 100)
        ax.fill_between(xs, xs * 0.95, xs * 1.05, color="#d7f0d7", alpha=0.7, lw=0.0)
        ax.fill_between(xs, xs * 0.90, xs * 1.10, color="#f7efd0", alpha=0.55, lw=0.0)
        ax.fill_between(xs, xs * 0.80, xs * 1.20, color="#f5d7d7", alpha=0.45, lw=0.0)
        ax.plot(xs, xs, color="k", ls="--", lw=1.3, alpha=0.8)
        ax.plot(xs, xs * 1.05, color="0.25", ls=":", lw=1.0, alpha=0.7)
        ax.plot(xs, xs * 0.95, color="0.25", ls=":", lw=1.0, alpha=0.7)
        ax.plot(xs, xs * 1.10, color="0.25", ls="--", lw=1.0, alpha=0.5)
        ax.plot(xs, xs * 0.90, color="0.25", ls="--", lw=1.0, alpha=0.5)
        ax.plot(xs, xs * 1.20, color="0.25", ls="-.", lw=1.0, alpha=0.4)
        ax.plot(xs, xs * 0.80, color="0.25", ls="-.", lw=1.0, alpha=0.4)

    # rescale each property to [0..1] for shared band template (keeps look consistent)
    # We'll scale by global min/max across both stages/states.
    x_all = pd.to_numeric(tmp["obs"], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(tmp["pred"], errors="coerce").to_numpy(float)
    vmin = float(np.nanmin(np.concatenate([x_all, y_all])))
    vmax = float(np.nanmax(np.concatenate([x_all, y_all])))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return

    def scale(v: np.ndarray) -> np.ndarray:
        return (v - vmin) / (vmax - vmin)

    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 2, figsize=(13.0, 6.0), constrained_layout=True)
        for ax, state in zip(axs, ("dry", "wet"), strict=True):
            tt = tmp[tmp["fluid_state"] == state].copy()
            if tt.empty:
                ax.axis("off")
                continue
            add_bands(ax)
            for stage, color in (("before", "C1"), ("after", "C0")):
                t2 = tt[tt["stage"] == stage]
                if t2.empty:
                    continue
                x = scale(pd.to_numeric(t2["obs"], errors="coerce").to_numpy(float))
                y = scale(pd.to_numeric(t2["pred"], errors="coerce").to_numpy(float))
                ax.scatter(x, y, s=50, alpha=0.85, color=color, label=stage)
            ax.set_xlabel("Measured (scaled)")
            ax.set_ylabel("Predicted (scaled)")
            ax.set_title(f"M2 Bayesian — {state}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", frameon=True)

        fig.suptitle(f"Measured vs predicted — {prop.upper()} (mode={mode})", y=1.02, fontsize=16)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_bayes_delta_z(out_png: Path, df: pd.DataFrame, *, mode: str) -> None:
    import matplotlib.pyplot as plt

    tmp = df[df["mode"] == mode].copy()
    if tmp.empty:
        return
    tmp = tmp.sort_values(["phi_before_pct", "lab_sample_id"]).reset_index(drop=True)
    x = np.arange(tmp.shape[0])

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(14.8, 4.8), constrained_layout=True)
        ax.axhline(0.0, color="0.2", lw=1.0, ls="--", alpha=0.4)
        y = tmp["delta_z_p50"].to_numpy(float)
        yerr_lo = y - tmp["delta_z_p05"].to_numpy(float)
        yerr_hi = tmp["delta_z_p95"].to_numpy(float) - y
        ax.errorbar(x, y, yerr=np.vstack([yerr_lo, yerr_hi]), fmt="o", ms=6, color="C3", ecolor="0.3", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in tmp["lab_sample_id"]], rotation=45, ha="right")
        ax.set_ylabel(r"$\Delta z=\log_{10}\alpha_{after}-\log_{10}\alpha_{before}$")
        ax.set_title(f"Bayesian M2 summary: Δz (p05–p95), mode={mode}")
        ax.grid(True, alpha=0.25)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian M2 inversion (beta AR distribution) for OSP/GSA with matrix nuisance.")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet-measurements", type=str, default="measurements_long")
    ap.add_argument("--sheet-stage", type=str, default="sample_stage")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    ap.add_argument("--samples", type=str, default="all")

    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=61)

    ap.add_argument("--m-step", type=float, default=0.05)
    ap.add_argument("--kappa-grid", type=str, default="2,5,10,20,50,100")

    # matrix nuisance grid
    ap.add_argument("--matrix-lambda-min", type=float, default=2.72)
    ap.add_argument("--matrix-lambda-max", type=float, default=2.98)
    ap.add_argument("--matrix-vp-min", type=float, default=5506.0)
    ap.add_argument("--matrix-vp-max", type=float, default=7063.0)
    ap.add_argument("--matrix-vs-min", type=float, default=3153.0)
    ap.add_argument("--matrix-vs-max", type=float, default=3909.0)
    ap.add_argument("--matrix-grid-n", type=int, default=5)
    ap.add_argument("--matrix-rho-kg-m3", type=float, default=2720.0)

    # fluids
    ap.add_argument("--air-lambda", type=float, default=0.03)
    ap.add_argument("--air-K-gpa", type=float, default=0.0001)
    ap.add_argument("--air-rho-kg-m3", type=float, default=1.2)
    ap.add_argument("--brine-lambda", type=float, default=0.60)
    ap.add_argument("--brine-K-gpa", type=float, default=2.20)
    ap.add_argument("--brine-rho-kg-m3", type=float, default=1030.0)

    ap.add_argument("--tc-sigma-rel", type=float, default=0.025)
    ap.add_argument("--vp-sigma-rel", type=float, default=0.05)
    ap.add_argument("--vs-sigma-rel", type=float, default=0.05)
    ap.add_argument("--delta-thresholds", type=str, default="0.05,0.10,0.20")

    ap.add_argument("--green-fortran", type=Path, default=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"))
    ap.add_argument("--elastic-backend-so", type=Path, default=Path("test_new/gsa/plots/libgsa_elastic_fortran.so"))
    ap.add_argument("--force-rebuild", action="store_true")
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

    kappa_grid = np.array([float(x) for x in str(args.kappa_grid).split(",") if str(x).strip()], dtype=float)
    m_vals, k_vals = _build_combo_grid(m_step=float(args.m_step), kappa_grid=kappa_grid)
    n_combo = int(m_vals.size)

    # combo weights and alpha_p50 for each combo
    W = np.vstack([_beta_weights(u, float(m), float(k)) for m, k in zip(m_vals, k_vals, strict=True)])
    alpha_p50 = np.array([_weighted_quantile(alphas, W[i], 0.50) for i in range(n_combo)], dtype=float)
    z_p50 = np.log10(alpha_p50)

    grid_n = int(args.matrix_grid_n)
    lam_grid = np.linspace(float(args.matrix_lambda_min), float(args.matrix_lambda_max), grid_n)
    vp_grid = np.linspace(float(args.matrix_vp_min), float(args.matrix_vp_max), grid_n)
    vs_grid = np.linspace(float(args.matrix_vs_min), float(args.matrix_vs_max), grid_n)
    matrix_grid: list[MatrixParams] = []
    for lam in lam_grid:
        for vp in vp_grid:
            for vs in vs_grid:
                matrix_grid.append(MatrixParams(float(lam), float(vp), float(vs), float(args.matrix_rho_kg_m3)))

    fluid_dry = FluidParams("air", float(args.air_lambda), float(args.air_K_gpa), float(args.air_rho_kg_m3))
    fluid_wet = FluidParams("brine", float(args.brine_lambda), float(args.brine_K_gpa), float(args.brine_rho_kg_m3))

    backend = gsa_elastic.build_backend(
        green_fortran=args.green_fortran,
        output_library=args.elastic_backend_so,
        force_rebuild=bool(args.force_rebuild),
    )

    out_rows: list[dict[str, float | str]] = []
    for sid in samples:
        rows = dfm[dfm["lab_sample_id"] == float(sid)].copy()
        if rows.empty:
            continue

        stage_before = dfs[(dfs["lab_sample_id"] == float(sid)) & (dfs["stage"] == "before")]
        phi_before_pct = float(stage_before.iloc[0]["phi_pct"]) if (not stage_before.empty and pd.notna(stage_before.iloc[0].get("phi_pct"))) else float("nan")
        perm_md = float(stage_before.iloc[0]["permeability_md_modeled"]) if (not stage_before.empty and pd.notna(stage_before.iloc[0].get("permeability_md_modeled"))) else float("nan")

        # We compute posteriors for both branches in one pass over the matrix grid.
        logw_joint_tc = np.full((n_combo, n_combo), float("-inf"), dtype=float)
        logw_mat_tc = np.full((len(matrix_grid),), float("-inf"), dtype=float)
        logw_joint_el = np.full((n_combo, n_combo), float("-inf"), dtype=float)
        logw_mat_el = np.full((len(matrix_grid),), float("-inf"), dtype=float)

        for mi, mat in enumerate(matrix_grid):
            # TC-only caches (fast, no elastic)
            cache_b_tc = _build_forward_cache_for_stage(
                rows=rows,
                stage="before",
                alphas=alphas,
                matrix=mat,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                need_elastic=False,
            )
            cache_a_tc = _build_forward_cache_for_stage(
                rows=rows,
                stage="after",
                alphas=alphas,
                matrix=mat,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                need_elastic=False,
            )

            if cache_b_tc is not None and cache_a_tc is not None:
                ll_b_tc = _stage_loglike_over_combos(
                    stage="before",
                    rows=rows,
                    cache=cache_b_tc,
                    W=W,
                    mode="tc_only",
                    tc_sigma_rel=float(args.tc_sigma_rel),
                    vp_sigma_rel=float(args.vp_sigma_rel),
                    vs_sigma_rel=float(args.vs_sigma_rel),
                )
                ll_a_tc = _stage_loglike_over_combos(
                    stage="after",
                    rows=rows,
                    cache=cache_a_tc,
                    W=W,
                    mode="tc_only",
                    tc_sigma_rel=float(args.tc_sigma_rel),
                    vp_sigma_rel=float(args.vp_sigma_rel),
                    vs_sigma_rel=float(args.vs_sigma_rel),
                )
                joint_tc = ll_b_tc[:, None] + ll_a_tc[None, :]
                logw_mat_tc[mi] = _logsumexp(joint_tc)
                logw_joint_tc = np.logaddexp(logw_joint_tc, joint_tc)

            # Elastic branch caches (slow)
            cache_b_el = _build_forward_cache_for_stage(
                rows=rows,
                stage="before",
                alphas=alphas,
                matrix=mat,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                need_elastic=True,
            )
            cache_a_el = _build_forward_cache_for_stage(
                rows=rows,
                stage="after",
                alphas=alphas,
                matrix=mat,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                need_elastic=True,
            )
            if cache_b_el is not None and cache_a_el is not None:
                ll_b_el = _stage_loglike_over_combos(
                    stage="before",
                    rows=rows,
                    cache=cache_b_el,
                    W=W,
                    mode="tc_vp_vs",
                    tc_sigma_rel=float(args.tc_sigma_rel),
                    vp_sigma_rel=float(args.vp_sigma_rel),
                    vs_sigma_rel=float(args.vs_sigma_rel),
                )
                ll_a_el = _stage_loglike_over_combos(
                    stage="after",
                    rows=rows,
                    cache=cache_a_el,
                    W=W,
                    mode="tc_vp_vs",
                    tc_sigma_rel=float(args.tc_sigma_rel),
                    vp_sigma_rel=float(args.vp_sigma_rel),
                    vs_sigma_rel=float(args.vs_sigma_rel),
                )
                joint_el = ll_b_el[:, None] + ll_a_el[None, :]
                logw_mat_el[mi] = _logsumexp(joint_el)
                logw_joint_el = np.logaddexp(logw_joint_el, joint_el)

        delta_thresholds = [float(x) for x in str(args.delta_thresholds).split(",") if str(x).strip()]
        for mode, logw_joint, logw_matrix in (
            ("tc_only", logw_joint_tc, logw_mat_tc),
            ("tc_vp_vs", logw_joint_el, logw_mat_el),
        ):
            logZ_joint = _logsumexp(logw_joint)
            w_joint = np.exp(logw_joint - logZ_joint)

            # delta_z distribution from z_p50
            dz = z_p50[None, :] - z_p50[:, None]
            p_neg = float(np.sum(w_joint[dz < 0.0]))
            p_thresh: dict[float, float] = {}
            for d in delta_thresholds:
                dd = float(d)
                if not np.isfinite(dd) or dd <= 0:
                    continue
                p_thresh[dd] = float(np.sum(w_joint[dz < -dd]))
            dz_flat = dz.reshape(-1)
            w_flat = w_joint.reshape(-1)
            idx = np.argsort(dz_flat)
            dz_s = dz_flat[idx]
            w_s = w_flat[idx]
            cw = np.cumsum(w_s)
            cw = cw / cw[-1]

            def qval(q: float) -> float:
                j = int(np.searchsorted(cw, q, side="left"))
                j = min(max(j, 0), dz_s.size - 1)
                return float(dz_s[j])

            dz_p05 = qval(0.05)
            dz_p50 = qval(0.50)
            dz_p95 = qval(0.95)

            # marginals over combos
            w_b = np.sum(w_joint, axis=1)
            w_a = np.sum(w_joint, axis=0)
            # posterior over alpha (mixture of beta weights)
            p_alpha_b = _posterior_alpha_from_combo_marginal(W, w_b)
            p_alpha_a = _posterior_alpha_from_combo_marginal(W, w_a)

            # summaries on alpha_p50
            ab_p05 = _weighted_quantile(alpha_p50, w_b, 0.05)
            ab_p50 = _weighted_quantile(alpha_p50, w_b, 0.50)
            ab_p95 = _weighted_quantile(alpha_p50, w_b, 0.95)
            aa_p05 = _weighted_quantile(alpha_p50, w_a, 0.05)
            aa_p50 = _weighted_quantile(alpha_p50, w_a, 0.50)
            aa_p95 = _weighted_quantile(alpha_p50, w_a, 0.95)

            # MAP combos
            ib = int(np.nanargmax(w_b))
            ia = int(np.nanargmax(w_a))

            # matrix MAP
            logZ_m = _logsumexp(logw_matrix)
            wm = np.exp(logw_matrix - logZ_m)
            im_map = int(np.nanargmax(wm))
            mat_map = matrix_grid[im_map]
            K_m, G_m = _matrix_KG_from_vp_vs_rho(mat_map.vp_m_s, mat_map.vs_m_s, mat_map.rho_kg_m3)

            # Posterior predictive at MAP (use MAP before/after combos separately, same matrix MAP)
            cache_b_map = _build_forward_cache_for_stage(
                rows=rows,
                stage="before",
                alphas=alphas,
                matrix=mat_map,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                need_elastic=(mode != "tc_only"),
            )
            cache_a_map = _build_forward_cache_for_stage(
                rows=rows,
                stage="after",
                alphas=alphas,
                matrix=mat_map,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                need_elastic=(mode != "tc_only"),
            )

            # Weighted average using MAP beta weights for each stage
            wb_map = W[int(ib)]
            wa_map = W[int(ia)]

            def pred_stage(cache: ForwardCache | None, w: np.ndarray, key: str) -> float:
                if cache is None:
                    return float("nan")
                arr = getattr(cache, key)
                if arr is None:
                    return float("nan")
                return float(w @ np.asarray(arr, dtype=float))

            pred = {
                "before": {
                    "tc_dry": pred_stage(cache_b_map, wb_map, "tc_dry"),
                    "tc_wet": pred_stage(cache_b_map, wb_map, "tc_wet"),
                    "vp_dry": pred_stage(cache_b_map, wb_map, "vp_dry"),
                    "vs_dry": pred_stage(cache_b_map, wb_map, "vs_dry"),
                    "vp_wet": pred_stage(cache_b_map, wb_map, "vp_wet"),
                    "vs_wet": pred_stage(cache_b_map, wb_map, "vs_wet"),
                },
                "after": {
                    "tc_dry": pred_stage(cache_a_map, wa_map, "tc_dry"),
                    "tc_wet": pred_stage(cache_a_map, wa_map, "tc_wet"),
                    "vp_dry": pred_stage(cache_a_map, wa_map, "vp_dry"),
                    "vs_dry": pred_stage(cache_a_map, wa_map, "vs_dry"),
                    "vp_wet": pred_stage(cache_a_map, wa_map, "vp_wet"),
                    "vs_wet": pred_stage(cache_a_map, wa_map, "vs_wet"),
                },
            }

            # collect observed values (if present) and store misfit detail
            for stage in ("before", "after"):
                for fluid_state in ("dry", "wet"):
                    obs = _collect_obs(rows, stage=stage, fluid_state=fluid_state)
                    if obs is None:
                        continue
                    for prop, pred_key in (
                        ("tc", f"tc_{fluid_state}"),
                        ("vp", f"vp_{fluid_state}"),
                        ("vs", f"vs_{fluid_state}"),
                    ):
                        if prop not in obs or not np.isfinite(float(obs[prop])):
                            continue
                        if mode == "tc_only" and prop in {"vp", "vs"}:
                            continue
                        y = float(pred[stage][pred_key])
                        x = float(obs[prop])
                        out_rows.append(
                            {
                                "lab_sample_id": float(sid),
                                "mode": str(mode),
                                "stage": str(stage),
                                "fluid_state": str(fluid_state),
                                "property": str(prop),
                                "obs": float(x),
                                "pred": float(y),
                                "misfit_pct": 100.0 * abs(_safe_div(y - x, x)),
                            }
                        )

            out_rows.append(
                {
                    "lab_sample_id": float(sid),
                    "mode": str(mode),
                    "phi_before_pct": float(phi_before_pct),
                    "perm_before_md_modeled": float(perm_md),
                    "alpha_before_p50": float(ab_p50),
                    "alpha_after_p50": float(aa_p50),
                    "alpha_before_p05": float(ab_p05),
                    "alpha_before_p95": float(ab_p95),
                    "alpha_after_p05": float(aa_p05),
                    "alpha_after_p95": float(aa_p95),
                    "delta_z_p05": float(dz_p05),
                    "delta_z_p50": float(dz_p50),
                    "delta_z_p95": float(dz_p95),
                    "p_delta_z_lt_0": float(p_neg),
                    "m_before_map": float(m_vals[ib]),
                    "kappa_before_map": float(k_vals[ib]),
                    "m_after_map": float(m_vals[ia]),
                    "kappa_after_map": float(k_vals[ia]),
                    "matrix_lambda_map": float(mat_map.lambda_w_mk),
                    "matrix_vp_map": float(mat_map.vp_m_s),
                    "matrix_vs_map": float(mat_map.vs_m_s),
                    "matrix_K_map_gpa": float(K_m) / 1e9,
                    "matrix_G_map_gpa": float(G_m) / 1e9,
                }
            )
            # threshold probabilities
            for d, pv in sorted(p_thresh.items()):
                out_rows[-1][f"p_delta_z_lt_m{str(d).replace('.', 'p')}"] = float(pv)

            # store posterior alpha (long table)
            for stage, p in (("before", p_alpha_b), ("after", p_alpha_a)):
                if not np.isfinite(p).any():
                    continue
                for a, pa in zip(alphas, p, strict=True):
                    out_rows.append(
                        {
                            "lab_sample_id": float(sid),
                            "mode": str(mode),
                            "stage": str(stage),
                            "alpha": float(a),
                            "prob": float(pa),
                            "__kind": "posterior_alpha",
                        }
                    )

    out_df = pd.DataFrame(out_rows)
    # Split the long list into summary vs detail tables (keep file format similar to bayes M1)
    summary = out_df[(out_df.get("__kind").isna()) & (out_df.get("property").isna())].copy()
    misfit_detail = out_df[out_df.get("property").notna()].copy()
    posterior_alpha = out_df[out_df.get("__kind") == "posterior_alpha"].copy()
    for d in (posterior_alpha,):
        if not d.empty:
            d.drop(columns=["__kind"], inplace=True, errors="ignore")

    out_xlsx = out_dir / "gsa_bayes_m2_tc_elastic_results.xlsx"
    with pd.ExcelWriter(out_xlsx) as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        misfit_detail.to_excel(xw, sheet_name="misfit_detail", index=False)
        posterior_alpha.to_excel(xw, sheet_name="posterior_alpha", index=False)
    print(f"Saved: {out_xlsx}")

    _plot_bayes_delta_z(out_dir / "gsa_bayes_m2_delta_z_tc_only.png", summary, mode="tc_only")
    print(f"Saved: {out_dir / 'gsa_bayes_m2_delta_z_tc_only.png'}")
    _plot_bayes_delta_z(out_dir / "gsa_bayes_m2_delta_z_tc_vp_vs.png", summary, mode="tc_vp_vs")
    print(f"Saved: {out_dir / 'gsa_bayes_m2_delta_z_tc_vp_vs.png'}")

    # Posterior alpha plots for diverse samples (to match M1 handoff)
    sel = _auto_pick_diverse_samples(summary, mode="tc_only", n=4)
    if sel and not posterior_alpha.empty:
        _plot_posterior_alpha_panels(out_dir / "gsa_bayes_m2_posterior_alpha_tc_only.png", posterior_alpha=posterior_alpha, sample_ids=sel, mode="tc_only")
        print(f"Saved: {out_dir / 'gsa_bayes_m2_posterior_alpha_tc_only.png'}")
        _plot_posterior_alpha_panels(out_dir / "gsa_bayes_m2_posterior_alpha_tc_vp_vs.png", posterior_alpha=posterior_alpha, sample_ids=sel, mode="tc_vp_vs")
        print(f"Saved: {out_dir / 'gsa_bayes_m2_posterior_alpha_tc_vp_vs.png'}")
        print("Auto-picked diverse samples (by phi_before_pct):", ", ".join(f"{x:g}" for x in sel))

    # Measured vs predicted (scaled) panels for each property in each branch
    if not misfit_detail.empty:
        _save_measured_vs_predicted_panels(out_dir / "gsa_bayes_m2_measured_vs_predicted_tc_tc_only.png", summary=summary, misfit_detail=misfit_detail, prop="tc", mode="tc_only")
        print(f"Saved: {out_dir / 'gsa_bayes_m2_measured_vs_predicted_tc_tc_only.png'}")
        for prop in ("tc", "vp", "vs"):
            _save_measured_vs_predicted_panels(
                out_dir / f"gsa_bayes_m2_measured_vs_predicted_{prop}_tc_vp_vs.png",
                summary=summary,
                misfit_detail=misfit_detail,
                prop=prop,
                mode="tc_vp_vs",
            )
            print(f"Saved: {out_dir / f'gsa_bayes_m2_measured_vs_predicted_{prop}_tc_vp_vs.png'}")


if __name__ == "__main__":
    main()
