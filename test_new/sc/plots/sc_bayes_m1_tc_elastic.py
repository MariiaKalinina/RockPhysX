from __future__ import annotations

"""
Bayesian M1 inversion for the *self-consistent* (SC) EMT model.

This mirrors `test_new/gsa/plots/gsa_bayes_m1_tc_elastic.py`, but uses:
 - SC transport for thermal conductivity (self-consistent comparison body),
 - SC elastic (Berryman-style self-consistent / CPA) for Vp/Vs.

Outputs (PNG only)
------------------
- Excel summary table with per-sample posteriors (p05/p50/p95, P(Δz<0), etc.)
- Posterior curves for a few diverse samples
- Summary plots (Δz distribution, criterion heatmap, etc.)
"""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rockphysx.models.emt import gsa_transport
from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores


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
    K_gpa: float
    G_gpa: float
    rho_kg_m3: float


@dataclass(frozen=True)
class FluidParams:
    name: str
    lambda_w_mk: float
    K_gpa: float
    rho_kg_m3: float


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


def _predict_tc_isotropic_sc(matrix_lambda: float, fluid_lambda: float, phi: float, alpha: float) -> float:
    # For isotropic-random mixtures, gsa_transport has a special fast SC branch.
    return float(
        gsa_transport.two_phase_thermal_isotropic(
            float(matrix_lambda),
            float(fluid_lambda),
            float(phi),
            aspect_ratio=float(alpha),
            comparison="self_consistent",
            max_iter=400,
            tol=1e-12,
        )
    )


def _predict_sc_elastic_vp_vs(
    *,
    matrix_K_gpa: float,
    matrix_G_gpa: float,
    fluid_K_gpa: float,
    phi: float,
    alpha: float,
    rho_bulk_kg_m3: float,
    warm: tuple[float, float] | None,
) -> tuple[float, float, tuple[float, float]]:
    # Use the spheroidal Berryman SC implementation (Irina-style P/Q).
    # Relaxation is important for crack-like inclusions.
    relax = 0.6 if float(alpha) < 0.2 else 1.0
    K_eff, G_eff = berryman_self_consistent_spheroidal_pores(
        matrix_bulk_gpa=float(matrix_K_gpa),
        matrix_shear_gpa=float(matrix_G_gpa),
        porosity=float(phi),
        pore_bulk_gpa=float(fluid_K_gpa),
        aspect_ratio=float(alpha),
        relaxation=float(relax),
        max_iter=4000,
        tol=1e-10,
        initial_guess_gpa=warm,
    )
    rho = float(rho_bulk_kg_m3)
    C11 = (float(K_eff) + 4.0 * float(G_eff) / 3.0) * 1e9  # Pa
    mu = float(G_eff) * 1e9
    vp = float(np.sqrt(max(C11 / rho, 0.0)))
    vs = float(np.sqrt(max(mu / rho, 0.0)))
    return vp, vs, (float(K_eff), float(G_eff))


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


def _stage_loglike_over_alpha(
    *,
    stage: str,
    rows: pd.DataFrame,
    alphas: np.ndarray,
    matrix: MatrixParams,
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    mode: str,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
) -> np.ndarray:
    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")
    if obs_dry is None and obs_wet is None:
        return np.full_like(alphas, float("-inf"), dtype=float)

    phi = float(obs_dry["phi"] if obs_dry is not None else obs_wet["phi"])  # type: ignore[index]
    ll = np.zeros_like(alphas, dtype=float)

    def add_rel_gauss(pred: np.ndarray, obs: float | None, sigma_rel: float) -> None:
        if obs is None or not np.isfinite(obs) or obs == 0:
            return
        e_rel = (pred - float(obs)) / float(obs)
        z = e_rel / max(float(sigma_rel), 1e-12)
        ll[:] += -0.5 * (z * z)

    # TC
    tc_dry = np.array([_predict_tc_isotropic_sc(matrix.lambda_w_mk, fluid_dry.lambda_w_mk, phi, float(a)) for a in alphas])
    tc_wet = np.array([_predict_tc_isotropic_sc(matrix.lambda_w_mk, fluid_wet.lambda_w_mk, phi, float(a)) for a in alphas])
    if obs_dry is not None:
        add_rel_gauss(tc_dry, float(obs_dry["tc"]) if "tc" in obs_dry else None, tc_sigma_rel)
    if obs_wet is not None:
        add_rel_gauss(tc_wet, float(obs_wet["tc"]) if "tc" in obs_wet else None, tc_sigma_rel)

    if mode == "tc_only":
        return ll

    # Elastic
    rho_dry = float(obs_dry["rho"]) if (obs_dry is not None and "rho" in obs_dry) else float(matrix.rho_kg_m3)
    rho_wet = float(obs_wet["rho"]) if (obs_wet is not None and "rho" in obs_wet) else float(matrix.rho_kg_m3)

    vp_dry = np.full_like(alphas, np.nan, dtype=float)
    vs_dry = np.full_like(alphas, np.nan, dtype=float)
    vp_wet = np.full_like(alphas, np.nan, dtype=float)
    vs_wet = np.full_like(alphas, np.nan, dtype=float)

    warm_dry: tuple[float, float] | None = (float(matrix.K_gpa), float(matrix.G_gpa))
    warm_wet: tuple[float, float] | None = (float(matrix.K_gpa), float(matrix.G_gpa))
    for i, a in enumerate(alphas):
        vp_dry[i], vs_dry[i], warm_dry = _predict_sc_elastic_vp_vs(
            matrix_K_gpa=float(matrix.K_gpa),
            matrix_G_gpa=float(matrix.G_gpa),
            fluid_K_gpa=float(fluid_dry.K_gpa),
            phi=phi,
            alpha=float(a),
            rho_bulk_kg_m3=rho_dry,
            warm=warm_dry,
        )
        vp_wet[i], vs_wet[i], warm_wet = _predict_sc_elastic_vp_vs(
            matrix_K_gpa=float(matrix.K_gpa),
            matrix_G_gpa=float(matrix.G_gpa),
            fluid_K_gpa=float(fluid_wet.K_gpa),
            phi=phi,
            alpha=float(a),
            rho_bulk_kg_m3=rho_wet,
            warm=warm_wet,
        )

    if obs_dry is not None:
        add_rel_gauss(vp_dry, float(obs_dry["vp"]) if "vp" in obs_dry else None, vp_sigma_rel)
        add_rel_gauss(vs_dry, float(obs_dry["vs"]) if "vs" in obs_dry else None, vs_sigma_rel)
    if obs_wet is not None:
        add_rel_gauss(vp_wet, float(obs_wet["vp"]) if "vp" in obs_wet else None, vp_sigma_rel)
        add_rel_gauss(vs_wet, float(obs_wet["vs"]) if "vs" in obs_wet else None, vs_sigma_rel)

    return ll


def _bayes_m1_for_sample(
    *,
    rows: pd.DataFrame,
    alphas: np.ndarray,
    matrix_grid: list[MatrixParams],
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    mode: str,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
    delta_thresholds: list[float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    n_alpha = int(alphas.size)
    z = np.log10(alphas)

    logw_alpha_b = np.full(n_alpha, float("-inf"), dtype=float)
    logw_alpha_a = np.full(n_alpha, float("-inf"), dtype=float)
    logw_matrix = np.full(len(matrix_grid), float("-inf"), dtype=float)
    logw_dz = np.full((n_alpha, n_alpha), float("-inf"), dtype=float)

    for mi, mat in enumerate(matrix_grid):
        ll_b = _stage_loglike_over_alpha(
            stage="before",
            rows=rows,
            alphas=alphas,
            matrix=mat,
            fluid_dry=fluid_dry,
            fluid_wet=fluid_wet,
            mode=mode,
            tc_sigma_rel=tc_sigma_rel,
            vp_sigma_rel=vp_sigma_rel,
            vs_sigma_rel=vs_sigma_rel,
        )
        ll_a = _stage_loglike_over_alpha(
            stage="after",
            rows=rows,
            alphas=alphas,
            matrix=mat,
            fluid_dry=fluid_dry,
            fluid_wet=fluid_wet,
            mode=mode,
            tc_sigma_rel=tc_sigma_rel,
            vp_sigma_rel=vp_sigma_rel,
            vs_sigma_rel=vs_sigma_rel,
        )

        joint = ll_b[:, None] + ll_a[None, :]
        logZ_mat = _logsumexp(joint)
        logw_matrix[mi] = logZ_mat

        logp_b_given = np.array([_logsumexp(joint[i, :]) for i in range(n_alpha)], dtype=float)
        logp_a_given = np.array([_logsumexp(joint[:, j]) for j in range(n_alpha)], dtype=float)

        logw_alpha_b = np.logaddexp(logw_alpha_b, logp_b_given)
        logw_alpha_a = np.logaddexp(logw_alpha_a, logp_a_given)
        logw_dz = np.logaddexp(logw_dz, joint)

    wb = np.exp(logw_alpha_b - _logsumexp(logw_alpha_b))
    wa = np.exp(logw_alpha_a - _logsumexp(logw_alpha_a))

    wm = np.exp(logw_matrix - _logsumexp(logw_matrix))
    im_map = int(np.nanargmax(wm))
    mat_map = matrix_grid[im_map]

    logZ_joint = _logsumexp(logw_dz)
    w_joint = np.exp(logw_dz - logZ_joint)
    dz = z[None, :] - z[:, None]
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

    ab_p05 = _weighted_quantile(alphas, wb, 0.05)
    ab_p50 = _weighted_quantile(alphas, wb, 0.50)
    ab_p95 = _weighted_quantile(alphas, wb, 0.95)
    aa_p05 = _weighted_quantile(alphas, wa, 0.05)
    aa_p50 = _weighted_quantile(alphas, wa, 0.50)
    aa_p95 = _weighted_quantile(alphas, wa, 0.95)

    # MAP (max of marginal) for quick reporting
    ib_map = int(np.nanargmax(wb))
    ia_map = int(np.nanargmax(wa))

    summary: dict[str, float] = {
        "alpha_before_map": float(alphas[ib_map]),
        "alpha_after_map": float(alphas[ia_map]),
        "alpha_before_p05": float(ab_p05),
        "alpha_before_p50": float(ab_p50),
        "alpha_before_p95": float(ab_p95),
        "alpha_after_p05": float(aa_p05),
        "alpha_after_p50": float(aa_p50),
        "alpha_after_p95": float(aa_p95),
        "delta_z_p05": float(dz_p05),
        "delta_z_p50": float(dz_p50),
        "delta_z_p95": float(dz_p95),
        "p_delta_z_lt_0": float(p_neg),
        "matrix_lambda_map": float(mat_map.lambda_w_mk),
        "matrix_K_map_gpa": float(mat_map.K_gpa),
        "matrix_G_map_gpa": float(mat_map.G_gpa),
    }
    for d, pv in sorted(p_thresh.items()):
        key = f"p_delta_z_lt_m{str(d).replace('.', 'p')}"
        summary[key] = float(pv)
    return summary, wb, wa


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
    else:
        idxs |= set(np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int).tolist())
    picked = sub.iloc[sorted(idxs)]["lab_sample_id"].tolist()
    out: list[float] = []
    for x in picked:
        v = float(x)
        if v not in out:
            out.append(v)
        if len(out) >= n:
            break
    return out


def _plot_alpha_posteriors(
    out_png: Path,
    posterior_long: pd.DataFrame,
    *,
    mode: str,
    sample_ids: list[float],
    alpha_min: float,
    alpha_max: float,
) -> None:
    import matplotlib.pyplot as plt

    df = posterior_long[posterior_long["mode"] == mode].copy()
    if df.empty:
        return
    df = df[df["lab_sample_id"].isin(sample_ids)].copy()
    if df.empty:
        return

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(16.2, 10.8), constrained_layout=True, sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)

        for ax in axes:
            ax.axvspan(1e-4, 1e-2, color="#f8d7da", alpha=0.35)
            ax.axvspan(1e-2, 1e-1, color="#fff2cc", alpha=0.45)
            ax.axvspan(1e-1, 1e0, color="#d8f0d8", alpha=0.35)
            ax.grid(True, which="both", alpha=0.25)

        for ax, sid in zip(axes, sample_ids, strict=False):
            sub_s = df[df["lab_sample_id"] == float(sid)].copy()
            if sub_s.empty:
                ax.set_axis_off()
                continue

            for stage, color_line, color_fill in (
                ("before", "#0066ff", "#4aa3ff"),
                ("after", "#b00000", "#ff6b6b"),
            ):
                ss = sub_s[sub_s["stage"] == stage].copy()
                if ss.empty:
                    continue
                a = ss["alpha"].to_numpy(float)
                w = ss["weight"].to_numpy(float)
                m = np.isfinite(a) & np.isfinite(w)
                a = a[m]
                w = w[m]
                if a.size == 0:
                    continue
                if float(np.nanmax(w)) > 0:
                    w = w / float(np.nanmax(w))
                ax.fill_between(a, 0, w, color=color_fill, alpha=0.22)
                ax.plot(a, w, color=color_line, lw=2.0, label=stage.capitalize())

                med = float(ss["alpha_p50"].iloc[0]) if "alpha_p50" in ss.columns else float("nan")
                if np.isfinite(med):
                    ax.axvline(med, color=color_line, ls="--", lw=1.6, alpha=0.9)

            ax.set_title(f"Sample {float(sid):g}")
            ax.set_xscale("log")
            ax.set_xlim(alpha_min, alpha_max)
            ax.set_ylim(0.0, 1.02)

        for ax in axes[2:]:
            ax.set_xlabel(r"Aspect ratio ($\alpha$) — Log scale")
        for ax in axes[0::2]:
            ax.set_ylabel("Normalized posterior density")

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_handles = [
            Patch(facecolor="#4aa3ff", alpha=0.22, label="Before"),
            Patch(facecolor="#ff6b6b", alpha=0.22, label="After"),
            Line2D([0], [0], color="#0066ff", ls="--", lw=1.6, label="Median before"),
            Line2D([0], [0], color="#b00000", ls="--", lw=1.6, label="Median after"),
            Patch(facecolor="#f8d7da", alpha=0.35, label="Microcracks"),
            Patch(facecolor="#fff2cc", alpha=0.45, label="Crack-like pores"),
            Patch(facecolor="#d8f0d8", alpha=0.35, label="Interparticle pores"),
        ]
        fig.legend(handles=legend_handles, loc="center right", frameon=True, facecolor="white", edgecolor="0.8")
        fig.suptitle(f"SC Bayesian M1 posterior: before vs after ({mode})")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_bayes_summary(out_png: Path, df: pd.DataFrame, *, mode: str) -> None:
    import matplotlib.pyplot as plt

    tmp = df[df["mode"] == mode].copy()
    if tmp.empty:
        return

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.6), constrained_layout=True)

        x = tmp["phi_before_pct"].to_numpy(float)
        y = tmp["delta_z_p50"].to_numpy(float)
        ylo = tmp["delta_z_p05"].to_numpy(float)
        yhi = tmp["delta_z_p95"].to_numpy(float)
        p = tmp["p_delta_z_lt_0"].to_numpy(float)

        ax.errorbar(x, y, yerr=[y - ylo, yhi - y], fmt="o", ms=5, color="#222222", ecolor="#666666", alpha=0.85)
        sc = ax.scatter(x, y, c=p, cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$P(\Delta z < 0)$")

        ax.axhline(0.0, color="k", ls="--", lw=1.2, alpha=0.6)
        ax.set_xlabel("Porosity before (%)")
        ax.set_ylabel(r"$\Delta z$ (p50), where $z=\log_{10}(\alpha)$")
        ax.set_title(f"SC Bayesian M1: Δz summary ({mode})")
        ax.grid(True, alpha=0.25)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_threshold_heatmap(out_png: Path, df: pd.DataFrame, *, mode: str, thresholds: list[float]) -> None:
    import matplotlib.pyplot as plt

    tmp = df[df["mode"] == mode].copy()
    if tmp.empty:
        return
    tmp = tmp.sort_values("phi_before_pct").reset_index(drop=True)

    # Build matrix: rows=sample, cols=threshold
    cols: list[str] = []
    for d in thresholds:
        key = f"p_delta_z_lt_m{str(d).replace('.', 'p')}"
        if key in tmp.columns:
            cols.append(key)
    if not cols:
        return

    M = tmp[cols].to_numpy(float)
    ylab = tmp["lab_sample_id"].to_numpy(float)
    xlab = [c.replace("p_delta_z_lt_m", "δ*=") for c in cols]

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 10,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8), constrained_layout=True)
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="GnBu", vmin=0.0, vmax=1.0)
        ax.set_title(r"Threshold crack-like criterion: $P(\Delta z < -\delta_*)$")
        ax.set_xlabel("Threshold δ*")
        ax.set_ylabel("lab_sample_id (sorted by porosity)")
        ax.set_xticks(np.arange(len(xlab)))
        ax.set_xticklabels(xlab, rotation=0)
        ax.set_yticks(np.arange(len(ylab)))
        # show only a subset to keep readable
        if len(ylab) > 18:
            step = int(math.ceil(len(ylab) / 18))
            labels = [f"{v:g}" if (i % step == 0) else "" for i, v in enumerate(ylab)]
        else:
            labels = [f"{v:g}" for v in ylab]
        ax.set_yticklabels(labels)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("probability")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian M1 inversion for SC model (TC-only or TC+Vp+Vs).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet", type=str, default="measurements_long")
    ap.add_argument("--samples", type=str, default="all")
    ap.add_argument("--mode", type=str, choices=["tc_only", "tc_vp_vs"], default="tc_vp_vs")
    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--n-alpha", type=int, default=161)

    ap.add_argument("--tc-sigma-rel", type=float, default=0.025, help="Relative sigma for TC (e.g. 0.025 => 2.5%).")
    ap.add_argument("--vp-sigma-rel", type=float, default=0.05, help="Relative sigma for Vp.")
    ap.add_argument("--vs-sigma-rel", type=float, default=0.05, help="Relative sigma for Vs.")

    ap.add_argument("--matrix-grid-n", type=int, default=5)
    ap.add_argument("--matrix-lambda-min", type=float, default=2.72)
    ap.add_argument("--matrix-lambda-max", type=float, default=2.98)
    ap.add_argument("--matrix-K-min", type=float, default=46.43)
    ap.add_argument("--matrix-K-max", type=float, default=80.29)
    ap.add_argument("--matrix-G-min", type=float, default=27.03)
    ap.add_argument("--matrix-G-max", type=float, default=41.55)
    ap.add_argument("--matrix-rho-kg-m3", type=float, default=2720.0)

    ap.add_argument("--dry-name", type=str, default="gas")
    ap.add_argument("--dry-lambda", type=float, default=0.03)
    ap.add_argument("--dry-K-gpa", type=float, default=0.0001)
    ap.add_argument("--dry-rho-kg-m3", type=float, default=1.2)

    ap.add_argument("--wet-name", type=str, default="brine")
    ap.add_argument("--wet-lambda", type=float, default=0.60)
    ap.add_argument("--wet-K-gpa", type=float, default=2.20)
    ap.add_argument("--wet-rho-kg-m3", type=float, default=1030.0)

    ap.add_argument("--delta-thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/sc/plots/sc_bayes_m1_outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    df = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet))
    df = df.copy()
    for col in ["stage", "fluid_state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce")
    df = df[np.isfinite(df["lab_sample_id"])]

    want = _parse_samples(str(args.samples))
    if want:
        df = df[df["lab_sample_id"].isin(want)].copy()

    # alpha grid (log-spaced)
    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    alphas = np.logspace(math.log10(alpha_min), math.log10(alpha_max), int(args.n_alpha))

    # nuisance matrix grid (same for before/after by construction)
    grid_n = int(args.matrix_grid_n)
    lam_grid = np.linspace(float(args.matrix_lambda_min), float(args.matrix_lambda_max), grid_n)
    K_grid = np.linspace(float(args.matrix_K_min), float(args.matrix_K_max), grid_n)
    G_grid = np.linspace(float(args.matrix_G_min), float(args.matrix_G_max), grid_n)
    matrix_grid: list[MatrixParams] = []
    for lam in lam_grid:
        for K in K_grid:
            for G in G_grid:
                matrix_grid.append(MatrixParams(float(lam), float(K), float(G), float(args.matrix_rho_kg_m3)))

    fluid_dry = FluidParams(str(args.dry_name), float(args.dry_lambda), float(args.dry_K_gpa), float(args.dry_rho_kg_m3))
    fluid_wet = FluidParams(str(args.wet_name), float(args.wet_lambda), float(args.wet_K_gpa), float(args.wet_rho_kg_m3))

    mode = str(args.mode).strip().lower()
    tc_sig = float(args.tc_sigma_rel)
    vp_sig = float(args.vp_sigma_rel)
    vs_sig = float(args.vs_sigma_rel)
    thresholds = [float(x) for x in list(args.delta_thresholds)]

    summaries: list[dict[str, float]] = []
    posterior_long_rows: list[dict[str, float]] = []

    for sid, rows in df.groupby("lab_sample_id", sort=True):
        rows = rows.copy()
        if rows.empty:
            continue
        ssum, wb, wa = _bayes_m1_for_sample(
            rows=rows,
            alphas=alphas,
            matrix_grid=matrix_grid,
            fluid_dry=fluid_dry,
            fluid_wet=fluid_wet,
            mode=mode,
            tc_sigma_rel=tc_sig,
            vp_sigma_rel=vp_sig,
            vs_sigma_rel=vs_sig,
            delta_thresholds=thresholds,
        )

        # get phi per stage (for plotting / sorting)
        obs_b = _collect_obs(rows, stage="before", fluid_state="dry") or _collect_obs(rows, stage="before", fluid_state="wet")
        obs_a = _collect_obs(rows, stage="after", fluid_state="dry") or _collect_obs(rows, stage="after", fluid_state="wet")
        phi_b = float(obs_b["phi"]) if obs_b is not None else float("nan")  # type: ignore[index]
        phi_a = float(obs_a["phi"]) if obs_a is not None else float("nan")  # type: ignore[index]

        base = {
            "lab_sample_id": float(sid),
            "mode": mode,
            "phi_before_frac": float(phi_b),
            "phi_before_pct": float(phi_b) * 100.0,
            "phi_after_frac": float(phi_a),
            "phi_after_pct": float(phi_a) * 100.0,
        }
        base.update(ssum)
        summaries.append(base)

        # posterior curves (store weights for before/after)
        for stage, w in (("before", wb), ("after", wa)):
            p05 = float(base[f"alpha_{stage}_p05"])
            p50 = float(base[f"alpha_{stage}_p50"])
            p95 = float(base[f"alpha_{stage}_p95"])
            for a, ww in zip(alphas, w, strict=False):
                posterior_long_rows.append(
                    {
                        "lab_sample_id": float(sid),
                        "mode": mode,
                        "stage": stage,
                        "alpha": float(a),
                        "weight": float(ww),
                        "alpha_p05": p05,
                        "alpha_p50": p50,
                        "alpha_p95": p95,
                    }
                )

    out_df = pd.DataFrame(summaries).sort_values("phi_before_pct").reset_index(drop=True)
    post_long = pd.DataFrame(posterior_long_rows)

    out_xlsx = out_dir / "sc_bayes_m1_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name="summary", index=False)
        post_long.to_excel(w, sheet_name="posterior_long", index=False)

    # Plots
    diverse = _auto_pick_diverse_samples(out_df, mode=mode, n=4)
    if diverse:
        _plot_alpha_posteriors(out_dir / f"sc_bayes_m1_alpha_posteriors_{mode}.png", post_long, mode=mode, sample_ids=diverse, alpha_min=alpha_min, alpha_max=alpha_max)
    _plot_bayes_summary(out_dir / f"sc_bayes_m1_delta_z_summary_{mode}.png", out_df, mode=mode)
    _plot_threshold_heatmap(out_dir / f"sc_bayes_m1_threshold_heatmap_{mode}.png", out_df, mode=mode, thresholds=thresholds)

    print(f"Saved: {out_xlsx}")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()

