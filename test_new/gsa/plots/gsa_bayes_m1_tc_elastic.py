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


def _stage_loglike_over_alpha(
    *,
    stage: str,
    rows: pd.DataFrame,
    alphas: np.ndarray,
    matrix: MatrixParams,
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
    mode: str,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
) -> np.ndarray:
    obs_dry = _collect_obs(rows, stage=stage, fluid_state="dry")
    obs_wet = _collect_obs(rows, stage=stage, fluid_state="wet")
    if obs_dry is None and obs_wet is None:
        return np.full_like(alphas, float("-inf"), dtype=float)

    # phi for this stage (use any available obs row)
    phi = float(obs_dry["phi"] if obs_dry is not None else obs_wet["phi"])  # type: ignore[index]

    ll = np.zeros_like(alphas, dtype=float)

    # TC predictions
    tc_dry = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_dry.lambda_w_mk, phi, float(a)) for a in alphas])
    tc_wet = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_wet.lambda_w_mk, phi, float(a)) for a in alphas])

    def add_rel_gauss(pred: np.ndarray, obs: float | None, sigma_rel: float) -> None:
        if obs is None or not np.isfinite(obs) or obs == 0:
            return
        e_rel = (pred - float(obs)) / float(obs)
        z = e_rel / max(float(sigma_rel), 1e-12)
        ll[:] += -0.5 * (z * z)

    if obs_dry is not None:
        add_rel_gauss(tc_dry, float(obs_dry["tc"]) if "tc" in obs_dry else None, tc_sigma_rel)
    if obs_wet is not None:
        add_rel_gauss(tc_wet, float(obs_wet["tc"]) if "tc" in obs_wet else None, tc_sigma_rel)

    if mode == "tc_only":
        return ll

    # Elastic predictions
    K_m, G_m = _matrix_KG_from_vp_vs_rho(matrix.vp_m_s, matrix.vs_m_s, matrix.rho_kg_m3)

    # densities per state if observed
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
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
    mode: str,
    tc_sigma_rel: float,
    vp_sigma_rel: float,
    vs_sigma_rel: float,
    delta_thresholds: list[float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    # Work in log domain, and marginalize matrix as discrete nuisance.
    n_alpha = int(alphas.size)
    z = np.log10(alphas)

    # We'll accumulate log weights for: alpha_before, alpha_after, delta_z bins, and matrix.
    logw_alpha_b = np.full(n_alpha, float("-inf"), dtype=float)
    logw_alpha_a = np.full(n_alpha, float("-inf"), dtype=float)
    logw_matrix = np.full(len(matrix_grid), float("-inf"), dtype=float)
    logw_dz = np.full((n_alpha, n_alpha), float("-inf"), dtype=float)  # (i_before, j_after)

    for mi, mat in enumerate(matrix_grid):
        ll_b = _stage_loglike_over_alpha(
            stage="before",
            rows=rows,
            alphas=alphas,
            matrix=mat,
            fluid_dry=fluid_dry,
            fluid_wet=fluid_wet,
            backend=backend,
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
            backend=backend,
            mode=mode,
            tc_sigma_rel=tc_sigma_rel,
            vp_sigma_rel=vp_sigma_rel,
            vs_sigma_rel=vs_sigma_rel,
        )

        # joint over (i,j) is ll_b[i] + ll_a[j] (+ const priors); compute log evidence for this matrix
        joint = ll_b[:, None] + ll_a[None, :]
        logZ_mat = _logsumexp(joint)
        logw_matrix[mi] = logZ_mat

        # accumulate alpha marginals: logsumexp over j or i, then combine across matrices
        logp_b_given_mat = np.array([_logsumexp(joint[i, :]) for i in range(n_alpha)], dtype=float)
        logp_a_given_mat = np.array([_logsumexp(joint[:, j]) for j in range(n_alpha)], dtype=float)

        # combine across matrices in log domain: logw_alpha = logsumexp( logp_given_mat )
        logw_alpha_b = np.logaddexp(logw_alpha_b, logp_b_given_mat)
        logw_alpha_a = np.logaddexp(logw_alpha_a, logp_a_given_mat)

        # store joint for dz accumulation (logaddexp over matrices)
        logw_dz = np.logaddexp(logw_dz, joint)

    # normalize marginals
    logZ_b = _logsumexp(logw_alpha_b)
    logZ_a = _logsumexp(logw_alpha_a)
    wb = np.exp(logw_alpha_b - logZ_b)
    wa = np.exp(logw_alpha_a - logZ_a)

    # matrix posterior
    logZ_m = _logsumexp(logw_matrix)
    wm = np.exp(logw_matrix - logZ_m)
    im_map = int(np.nanargmax(wm))

    # delta_z posterior via joint weights
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

    # summarize delta_z (by sampling from joint CDF on grid)
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

    # alpha summaries
    ab_p05 = _weighted_quantile(alphas, wb, 0.05)
    ab_p50 = _weighted_quantile(alphas, wb, 0.50)
    ab_p95 = _weighted_quantile(alphas, wb, 0.95)
    aa_p05 = _weighted_quantile(alphas, wa, 0.05)
    aa_p50 = _weighted_quantile(alphas, wa, 0.50)
    aa_p95 = _weighted_quantile(alphas, wa, 0.95)

    # MAP
    ib_map = int(np.nanargmax(wb))
    ia_map = int(np.nanargmax(wa))

    mat_map = matrix_grid[im_map]
    K_m, G_m = _matrix_KG_from_vp_vs_rho(mat_map.vp_m_s, mat_map.vs_m_s, mat_map.rho_kg_m3)

    summary = {
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
        "matrix_vp_map": float(mat_map.vp_m_s),
        "matrix_vs_map": float(mat_map.vs_m_s),
        "matrix_K_map_gpa": float(K_m) / 1e9,
        "matrix_G_map_gpa": float(G_m) / 1e9,
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

    # Pick min, max, and two quantile-ish interior points.
    idxs = {0, len(sub) - 1}
    if len(sub) >= 4:
        idxs.add(int(round(0.33 * (len(sub) - 1))))
        idxs.add(int(round(0.66 * (len(sub) - 1))))
    else:
        # fallback: evenly spaced
        idxs |= set(np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int).tolist())
    picked = sub.iloc[sorted(idxs)]["lab_sample_id"].tolist()
    # truncate to n unique
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
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }

    # 2x2 layout like the reference
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.2), constrained_layout=True, sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)

        for ax in axes:
            ax.axvspan(1e-4, 1e-2, color="#f8d7da", alpha=0.35)  # Microcracks
            ax.axvspan(1e-2, 1e-1, color="#fff2cc", alpha=0.45)  # Crack-like pores
            ax.axvspan(1e-1, 1e0, color="#d8f0d8", alpha=0.35)  # Interparticle pores
            ax.grid(True, which="both", alpha=0.25)

        for ax, sid in zip(axes, sample_ids, strict=False):
            sub_s = df[df["lab_sample_id"] == float(sid)].copy()
            if sub_s.empty:
                ax.set_axis_off()
                continue

            # stage curves
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
                # normalize to peak=1 (visual)
                if float(np.nanmax(w)) > 0:
                    w = w / float(np.nanmax(w))
                ax.fill_between(a, 0, w, color=color_fill, alpha=0.22)
                ax.plot(a, w, color=color_line, lw=2.0, label=stage.capitalize())

                # median marker
                if "alpha_p50" in ss.columns:
                    med = float(ss["alpha_p50"].iloc[0])
                    ax.axvline(med, color=color_line, ls="--", lw=1.6, alpha=0.9)

            ax.set_title(f"Sample {float(sid):g}")
            ax.set_xscale("log")
            ax.set_xlim(alpha_min, alpha_max)
            ax.set_ylim(0.0, 1.02)

        for ax in axes[2:]:
            ax.set_xlabel(r"Aspect ratio ($\alpha$) — Log scale")
        for ax in axes[0::2]:
            ax.set_ylabel("Normalized Posterior Density")

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
        fig.suptitle(f"Bayesian M1 posterior over aspect ratio: before vs after ({mode})")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_bayes_summary(out_png: Path, df: pd.DataFrame, *, mode: str) -> None:
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
        ax.errorbar(x, y, yerr=np.vstack([yerr_lo, yerr_hi]), fmt="o", ms=6, color="C0", ecolor="0.3", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in tmp["lab_sample_id"]], rotation=45, ha="right")
        ax.set_ylabel(r"$\Delta z=\log_{10}\alpha_{after}-\log_{10}\alpha_{before}$")
        ax.set_title(f"Bayesian M1 summary: Δz (p05–p95), mode={mode}")
        ax.grid(True, alpha=0.25)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian M1 inversion for OSP/GSA (TC-only and TC+Vp+Vs) with matrix nuisance.")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet-measurements", type=str, default="measurements_long")
    ap.add_argument("--sheet-stage", type=str, default="sample_stage")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    ap.add_argument("--samples", type=str, default="all", help="Comma-separated lab_sample_id list, or 'all'.")

    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=61)
    ap.add_argument("--posterior-samples", type=str, default="auto", help="Comma-separated lab_sample_id list for posterior plots, or 'auto'.")

    # Matrix uncertainty search (HS-feasible interval)
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

    # Relative measurement uncertainties
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

    grid_n = int(args.matrix_grid_n)
    lam_grid = np.linspace(float(args.matrix_lambda_min), float(args.matrix_lambda_max), grid_n)
    vp_grid = np.linspace(float(args.matrix_vp_min), float(args.matrix_vp_max), grid_n)
    vs_grid = np.linspace(float(args.matrix_vs_min), float(args.matrix_vs_max), grid_n)
    matrix_grid: list[MatrixParams] = []
    for lam in lam_grid:
        for vp in vp_grid:
            for vs in vs_grid:
                matrix_grid.append(
                    MatrixParams(
                        lambda_w_mk=float(lam),
                        vp_m_s=float(vp),
                        vs_m_s=float(vs),
                        rho_kg_m3=float(args.matrix_rho_kg_m3),
                    )
                )

    fluid_dry = FluidParams("air", float(args.air_lambda), float(args.air_K_gpa), float(args.air_rho_kg_m3))
    fluid_wet = FluidParams("brine", float(args.brine_lambda), float(args.brine_K_gpa), float(args.brine_rho_kg_m3))

    backend = gsa_elastic.build_backend(
        green_fortran=args.green_fortran,
        output_library=args.elastic_backend_so,
        force_rebuild=bool(args.force_rebuild),
    )

    out_rows: list[dict[str, float | str]] = []
    posterior_rows: list[dict[str, float | str]] = []
    misfit_rows: list[dict[str, float | str]] = []
    for sid in samples:
        rows = dfm[dfm["lab_sample_id"] == float(sid)].copy()
        if rows.empty:
            continue

        # stage info for sorting/metadata
        stage_before = dfs[(dfs["lab_sample_id"] == float(sid)) & (dfs["stage"] == "before")]
        phi_before_pct = float(stage_before.iloc[0]["phi_pct"]) if (not stage_before.empty and pd.notna(stage_before.iloc[0].get("phi_pct"))) else float("nan")
        perm_md = float(stage_before.iloc[0]["permeability_md_modeled"]) if (not stage_before.empty and pd.notna(stage_before.iloc[0].get("permeability_md_modeled"))) else float("nan")

        for mode in ("tc_only", "tc_vp_vs"):
            delta_thresholds = [float(x) for x in str(args.delta_thresholds).split(",") if str(x).strip()]
            res, wb, wa = _bayes_m1_for_sample(
                rows=rows,
                alphas=alphas,
                matrix_grid=matrix_grid,
                fluid_dry=fluid_dry,
                fluid_wet=fluid_wet,
                backend=backend,
                mode=mode,
                tc_sigma_rel=float(args.tc_sigma_rel),
                vp_sigma_rel=float(args.vp_sigma_rel),
                vs_sigma_rel=float(args.vs_sigma_rel),
                delta_thresholds=delta_thresholds,
            )
            out_rows.append(
                {
                    "lab_sample_id": float(sid),
                    "mode": str(mode),
                    "phi_before_pct": float(phi_before_pct),
                    "perm_before_md_modeled": float(perm_md),
                    **res,
                }
            )

            # Posterior predictive at MAP for misfit diagnostics.
            mat_lambda = float(res["matrix_lambda_map"])
            mat_vp = float(res["matrix_vp_map"])
            mat_vs = float(res["matrix_vs_map"])
            mat_rho = float(args.matrix_rho_kg_m3)
            K_m, G_m = _matrix_KG_from_vp_vs_rho(mat_vp, mat_vs, mat_rho)

            for stage, alpha_map in (("before", float(res["alpha_before_map"])), ("after", float(res["alpha_after_map"]))):
                for fluid_state, fluid_lambda, fluid_K in (
                    ("dry", float(args.air_lambda), float(args.air_K_gpa)),
                    ("wet", float(args.brine_lambda), float(args.brine_K_gpa)),
                ):
                    obs = _collect_obs(rows, stage=stage, fluid_state=fluid_state)
                    if obs is None:
                        continue
                    phi = float(obs["phi"])
                    tc_pred = _predict_tc_isotropic(mat_lambda, fluid_lambda, phi, alpha_map)
                    if "tc" in obs and np.isfinite(float(obs["tc"])):
                        x = float(obs["tc"])
                        y = float(tc_pred)
                        misfit_rows.append(
                            {
                                "lab_sample_id": float(sid),
                                "mode": str(mode),
                                "stage": str(stage),
                                "fluid_state": str(fluid_state),
                                "property": "tc",
                                "obs": float(x),
                                "pred": float(y),
                                "misfit_pct": 100.0 * abs(_safe_div(y - x, x)),
                            }
                        )

                    if mode == "tc_only":
                        continue

                    rho_bulk = float(obs["rho"]) if ("rho" in obs and np.isfinite(float(obs["rho"]))) else float(mat_rho)
                    vp_pred, vs_pred = _predict_elastic_vp_vs(
                        backend=backend,
                        matrix_K_pa=float(K_m),
                        matrix_G_pa=float(G_m),
                        fluid_K_pa=float(fluid_K) * 1e9,
                        phi=phi,
                        alpha=alpha_map,
                        rho_bulk_kg_m3=rho_bulk,
                    )

                    if "vp" in obs and np.isfinite(float(obs["vp"])):
                        x = float(obs["vp"])
                        y = float(vp_pred)
                        misfit_rows.append(
                            {
                                "lab_sample_id": float(sid),
                                "mode": str(mode),
                                "stage": str(stage),
                                "fluid_state": str(fluid_state),
                                "property": "vp",
                                "obs": float(x),
                                "pred": float(y),
                                "misfit_pct": 100.0 * abs(_safe_div(y - x, x)),
                            }
                        )
                    if "vs" in obs and np.isfinite(float(obs["vs"])):
                        x = float(obs["vs"])
                        y = float(vs_pred)
                        misfit_rows.append(
                            {
                                "lab_sample_id": float(sid),
                                "mode": str(mode),
                                "stage": str(stage),
                                "fluid_state": str(fluid_state),
                                "property": "vs",
                                "obs": float(x),
                                "pred": float(y),
                                "misfit_pct": 100.0 * abs(_safe_div(y - x, x)),
                            }
                        )

            # store alpha posteriors as long table (discrete)
            for stage, w, med in (
                ("before", wb, float(res["alpha_before_p50"])),
                ("after", wa, float(res["alpha_after_p50"])),
            ):
                for a, ww in zip(alphas, w, strict=True):
                    posterior_rows.append(
                        {
                            "lab_sample_id": float(sid),
                            "mode": str(mode),
                            "stage": str(stage),
                            "alpha": float(a),
                            "weight": float(ww),
                            "alpha_p50": float(med),
                        }
                    )

    out_df = pd.DataFrame(out_rows)
    post_df = pd.DataFrame(posterior_rows)
    mis_df = pd.DataFrame(misfit_rows)
    out_xlsx = out_dir / "gsa_bayes_m1_tc_elastic_results.xlsx"
    with pd.ExcelWriter(out_xlsx) as w:
        out_df.to_excel(w, sheet_name="summary", index=False)
        post_df.to_excel(w, sheet_name="posterior_alpha", index=False)
        mis_df.to_excel(w, sheet_name="misfit_detail", index=False)
    print(f"Saved: {out_xlsx}")

    _plot_bayes_summary(out_dir / "gsa_bayes_m1_delta_z_tc_only.png", out_df, mode="tc_only")
    print(f"Saved: {out_dir / 'gsa_bayes_m1_delta_z_tc_only.png'}")
    _plot_bayes_summary(out_dir / "gsa_bayes_m1_delta_z_tc_vp_vs.png", out_df, mode="tc_vp_vs")
    print(f"Saved: {out_dir / 'gsa_bayes_m1_delta_z_tc_vp_vs.png'}")

    # posterior plots (choose samples with diverse porosity unless explicitly provided)
    if str(args.posterior_samples).strip().lower() in {"auto", "*", "all"}:
        sample_ids_tc_only = _auto_pick_diverse_samples(out_df, mode="tc_only", n=4)
        sample_ids_tc_vp_vs = _auto_pick_diverse_samples(out_df, mode="tc_vp_vs", n=4)
    else:
        sample_ids_tc_only = [float(x.strip()) for x in str(args.posterior_samples).split(",") if x.strip()]
        sample_ids_tc_vp_vs = list(sample_ids_tc_only)

    if sample_ids_tc_only:
        _plot_alpha_posteriors(
            out_dir / "gsa_bayes_m1_posterior_alpha_tc_only.png",
            post_df,
            mode="tc_only",
            sample_ids=sample_ids_tc_only,
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
        )
        print(f"Saved: {out_dir / 'gsa_bayes_m1_posterior_alpha_tc_only.png'}")
    if sample_ids_tc_vp_vs:
        _plot_alpha_posteriors(
            out_dir / "gsa_bayes_m1_posterior_alpha_tc_vp_vs.png",
            post_df,
            mode="tc_vp_vs",
            sample_ids=sample_ids_tc_vp_vs,
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
        )
        print(f"Saved: {out_dir / 'gsa_bayes_m1_posterior_alpha_tc_vp_vs.png'}")


if __name__ == "__main__":
    main()
