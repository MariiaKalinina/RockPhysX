from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing helpers from other test_new scripts when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_new.hs_bounds_tc_vp_vs_porosity import (  # noqa: E402
    hs_lower_conductivity,
    hs_lower_elastic,
    hs_upper_conductivity,
    hs_upper_elastic,
    velocities_from_KG_rho,
)


def _configure_matplotlib_env(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _matrix_moduli_from_velocities(vp_m_s: float, vs_m_s: float, rho_kg_m3: float) -> tuple[float, float]:
    rho = float(rho_kg_m3)
    vp = float(vp_m_s)
    vs = float(vs_m_s)
    G = rho * vs**2
    K = rho * vp**2 - 4.0 * G / 3.0
    return K, G


def _violation(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.maximum(0.0, np.maximum(lo - y, y - hi))


def _safe_sigma(rel: float, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = float(rel) * np.maximum(np.abs(y), 1e-12)
    return np.maximum(s, 1e-12)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF (no SciPy dependency).

    Uses a common rational approximation (Abramowitz & Stegun 7.1.26 style),
    accurate to ~1e-7 which is plenty for probability-weighting here.
    """
    x = np.asarray(x, dtype=float)
    # Φ(x) ≈ 1 - φ(x) * (b1 t + b2 t^2 + ... + b5 t^5),  t=1/(1+p x), x>=0
    p = 0.2316419
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    phi = np.exp(-0.5 * ax * ax) / np.sqrt(2.0 * np.pi)
    cdf_pos = 1.0 - phi * poly
    return np.where(x >= 0, cdf_pos, 1.0 - cdf_pos)


def _prob_inside_interval(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Under y_sim = y + eps, eps~N(0,sigma), return P(lo <= y_sim <= hi) for each point.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z_hi = (hi - y) / sigma
    z_lo = (lo - y) / sigma
    p = _norm_cdf(z_hi) - _norm_cdf(z_lo)
    return np.clip(p, 1e-300, 1.0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted quantile for 0<=q<=1.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[m]
    w = weights[m]
    if v.size == 0:
        return float("nan")
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return float(np.interp(q, cw, v))


def _latex_escape(s: object) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def df_to_latex_booktabs(df: pd.DataFrame, *, caption: str, label: str, out_path: Path) -> None:
    cols = list(df.columns)
    align = []
    for c in cols:
        align.append("r" if pd.api.types.is_numeric_dtype(df[c]) else "l")
    lines: list[str] = []
    lines.append("\\begin{table}[ht!]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append("\\begin{tabular}{%s}" % ("".join(align)))
    lines.append("\\toprule")
    lines.append(" & ".join([f"\\textbf{{{_latex_escape(c)}}}" for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PhaseProps:
    lambda_pore_w_mk: float
    K_pore_gpa: float
    rho_pore_kg_m3: float


@dataclass(frozen=True)
class ScanConfig:
    data_xlsx: Path
    out_dir: Path
    n_samples: int
    seed: int
    # parameter ranges (uniform sampling)
    lambda_min: float
    lambda_max: float
    vp_min: float
    vp_max: float
    vs_min: float
    vs_max: float
    rho_matrix_kg_m3: float
    # measurement uncertainty (relative)
    tc_rel_unc: float
    v_rel_unc: float
    # feasibility criterion: fraction of points with normalized violation <= tau
    tau: float
    frac_ok_min: float
    max_vn_max: float
    # probability outputs
    prob_mode: str  # analytic|mc
    mc_draws: int
    # pore phases
    phase_dry: PhaseProps
    phase_wet: PhaseProps
    # which properties to include
    use_tc: bool
    use_vp: bool
    use_vs: bool
    dpi: int = 300


def _load_arrays(cfg: ScanConfig) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """
    Returns dict keyed by (property, fluid_state) with arrays:
      phi, y, sigma, stage (as string array)
    property in {"tc","vp","vs"}
    """
    df = pd.read_excel(cfg.data_xlsx, sheet_name="measurements_long")
    df["stage"] = df["stage"].astype(str).str.strip()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip()
    df["phi_pct"] = pd.to_numeric(df["phi_pct"], errors="coerce")
    df["phi"] = df["phi_pct"] / 100.0
    df = df[df["phi"].notna()].copy()

    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for prop, col, rel in [
        ("tc", "tc_w_mk", cfg.tc_rel_unc),
        ("vp", "vp_m_s", cfg.v_rel_unc),
        ("vs", "vs_m_s", cfg.v_rel_unc),
    ]:
        if prop == "tc" and not cfg.use_tc:
            continue
        if prop == "vp" and not cfg.use_vp:
            continue
        if prop == "vs" and not cfg.use_vs:
            continue
        for state in ["dry", "wet"]:
            sub = df[(df["fluid_state"] == state) & df[col].notna()].copy()
            if sub.empty:
                continue
            phi = sub["phi"].to_numpy(dtype=float)
            y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(phi) & np.isfinite(y)
            phi = phi[m]
            y = y[m]
            stage = sub["stage"].to_numpy(dtype=str)[m]
            if prop in {"vp", "vs"}:
                y = y / 1000.0  # km/s
            if prop == "tc":
                m2 = y > 0
                phi, y, stage = phi[m2], y[m2], stage[m2]
            sigma = _safe_sigma(rel, y)
            out[(prop, state)] = {"phi": phi, "y": y, "sigma": sigma, "stage": stage}
    return out


def _bounds_tc(lambda_m: float, phase: PhaseProps, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = hs_upper_conductivity(lambda_m, phase.lambda_pore_w_mk, phi)
    lo = hs_lower_conductivity(lambda_m, phase.lambda_pore_w_mk, phi)
    return lo, hi


def _bounds_vpvs(
    vp_m_s: float,
    vs_m_s: float,
    rho_m: float,
    phase: PhaseProps,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Km, Gm = _matrix_moduli_from_velocities(vp_m_s, vs_m_s, rho_m)
    Kf = phase.K_pore_gpa * 1e9
    Gf = 0.0
    K_hi, G_hi = hs_upper_elastic(Km, Gm, Kf, Gf, phi)
    K_lo, G_lo = hs_lower_elastic(Km, Gm, Kf, Gf, phi)
    rho_mix = (1.0 - phi) * rho_m + phi * phase.rho_pore_kg_m3
    vp_hi, vs_hi = velocities_from_KG_rho(K_hi, G_hi, rho_mix)
    vp_lo, vs_lo = velocities_from_KG_rho(K_lo, G_lo, rho_mix)
    return (vp_lo / 1000.0, vp_hi / 1000.0, vs_lo / 1000.0, vs_hi / 1000.0)


def _prob_all_mc(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, sigma: np.ndarray, *, draws: int, rng: np.random.Generator) -> float:
    """
    Monte-Carlo estimate of P(all points inside interval) assuming independent Gaussian noise.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if y.size == 0:
        return float("nan")
    # Draw eps with shape (draws, n)
    eps = rng.normal(loc=0.0, scale=sigma, size=(int(draws), y.size))
    ys = y[None, :] + eps
    ok = (ys >= lo[None, :]) & (ys <= hi[None, :])
    return float(np.mean(np.all(ok, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan feasible matrix-property ranges using HS-bound penalty (no regularization).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/hs_matrix_feasible_set"))
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lambda-min", type=float, default=2.70)
    ap.add_argument("--lambda-max", type=float, default=3.20)
    ap.add_argument("--vp-min", type=float, default=6000.0)
    ap.add_argument("--vp-max", type=float, default=7600.0)
    ap.add_argument("--vs-min", type=float, default=3200.0)
    ap.add_argument("--vs-max", type=float, default=4400.0)
    ap.add_argument("--rho-matrix", type=float, default=2725.0)

    ap.add_argument("--tc-rel-unc", type=float, default=0.025)
    ap.add_argument("--v-rel-unc", type=float, default=0.05)

    ap.add_argument("--tau", type=float, default=1.0, help="Normalized violation threshold (in sigma units).")
    ap.add_argument("--frac-ok-min", type=float, default=0.95, help="Minimum fraction of points with vn<=tau.")
    ap.add_argument("--max-vn-max", type=float, default=3.0, help="Maximum allowed vn (worst point).")

    ap.add_argument(
        "--prob-mode",
        choices=["analytic", "mc"],
        default="analytic",
        help="Compute P(data inside HS | theta) either analytically (Gaussian CDF) or via Monte-Carlo.",
    )
    ap.add_argument("--mc-draws", type=int, default=500, help="Monte-Carlo draws per theta when --prob-mode mc.")

    ap.add_argument("--use-tc", action="store_true", default=True)
    ap.add_argument("--no-tc", dest="use_tc", action="store_false")
    ap.add_argument("--use-vp", action="store_true", default=True)
    ap.add_argument("--no-vp", dest="use_vp", action="store_false")
    ap.add_argument("--use-vs", action="store_true", default=True)
    ap.add_argument("--no-vs", dest="use_vs", action="store_false")

    ap.add_argument("--lambda-air", type=float, default=0.026)
    ap.add_argument("--lambda-brine", type=float, default=0.60)
    ap.add_argument("--K-air-gpa", type=float, default=1e-4)
    ap.add_argument("--K-brine-gpa", type=float, default=2.2)
    ap.add_argument("--rho-air", type=float, default=1.2)
    ap.add_argument("--rho-brine", type=float, default=1030.0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    cfg = ScanConfig(
        data_xlsx=Path(args.data_xlsx),
        out_dir=Path(args.out_dir),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        lambda_min=float(args.lambda_min),
        lambda_max=float(args.lambda_max),
        vp_min=float(args.vp_min),
        vp_max=float(args.vp_max),
        vs_min=float(args.vs_min),
        vs_max=float(args.vs_max),
        rho_matrix_kg_m3=float(args.rho_matrix),
        tc_rel_unc=float(args.tc_rel_unc),
        v_rel_unc=float(args.v_rel_unc),
        tau=float(args.tau),
        frac_ok_min=float(args.frac_ok_min),
        max_vn_max=float(args.max_vn_max),
        prob_mode=str(args.prob_mode),
        mc_draws=int(args.mc_draws),
        phase_dry=PhaseProps(lambda_pore_w_mk=float(args.lambda_air), K_pore_gpa=float(args.K_air_gpa), rho_pore_kg_m3=float(args.rho_air)),
        phase_wet=PhaseProps(
            lambda_pore_w_mk=float(args.lambda_brine), K_pore_gpa=float(args.K_brine_gpa), rho_pore_kg_m3=float(args.rho_brine)
        ),
        use_tc=bool(args.use_tc),
        use_vp=bool(args.use_vp),
        use_vs=bool(args.use_vs),
        dpi=int(args.dpi),
    )

    _configure_matplotlib_env(cfg.out_dir)
    arrays = _load_arrays(cfg)

    rng = np.random.default_rng(cfg.seed)
    lam = rng.uniform(cfg.lambda_min, cfg.lambda_max, size=cfg.n_samples)
    vp = rng.uniform(cfg.vp_min, cfg.vp_max, size=cfg.n_samples)
    vs = rng.uniform(cfg.vs_min, cfg.vs_max, size=cfg.n_samples)

    # Metrics
    J = np.zeros(cfg.n_samples, dtype=float)
    frac_ok = np.zeros(cfg.n_samples, dtype=float)
    max_vn = np.zeros(cfg.n_samples, dtype=float)
    J_before = np.zeros(cfg.n_samples, dtype=float)
    J_after = np.zeros(cfg.n_samples, dtype=float)
    frac_ok_before = np.zeros(cfg.n_samples, dtype=float)
    frac_ok_after = np.zeros(cfg.n_samples, dtype=float)
    logP_all = np.zeros(cfg.n_samples, dtype=float)
    logP_tc = np.zeros(cfg.n_samples, dtype=float)
    logP_vp = np.zeros(cfg.n_samples, dtype=float)
    logP_vs = np.zeros(cfg.n_samples, dtype=float)
    mean_p = np.zeros(cfg.n_samples, dtype=float)
    P_all_mc = np.full(cfg.n_samples, np.nan, dtype=float)

    # Pre-collect per-sample vn lists (as sums / counts) for speed.
    for i in range(cfg.n_samples):
        vn_all: list[np.ndarray] = []
        vn_before: list[np.ndarray] = []
        vn_after: list[np.ndarray] = []
        p_all_parts: list[np.ndarray] = []
        p_tc_parts: list[np.ndarray] = []
        p_vp_parts: list[np.ndarray] = []
        p_vs_parts: list[np.ndarray] = []

        for (prop, state), d in arrays.items():
            phi_i = d["phi"]
            y_i = d["y"]
            sigma_i = d["sigma"]
            stage_i = d["stage"]
            phase = cfg.phase_dry if state == "dry" else cfg.phase_wet

            if prop == "tc":
                lo, hi = _bounds_tc(lam[i], phase, phi_i)
                v = _violation(y_i, lo, hi)
                p = _prob_inside_interval(y_i, lo, hi, sigma_i)
                p_tc_parts.append(p)
            else:
                vp_lo, vp_hi, vs_lo, vs_hi = _bounds_vpvs(vp[i], vs[i], cfg.rho_matrix_kg_m3, phase, phi_i)
                if prop == "vp":
                    v = _violation(y_i, vp_lo, vp_hi)
                    p = _prob_inside_interval(y_i, vp_lo, vp_hi, sigma_i)
                    p_vp_parts.append(p)
                else:
                    v = _violation(y_i, vs_lo, vs_hi)
                    p = _prob_inside_interval(y_i, vs_lo, vs_hi, sigma_i)
                    p_vs_parts.append(p)

            vn = v / sigma_i
            vn_all.append(vn)
            p_all_parts.append(p)
            if stage_i.size:
                m_b = stage_i == "before"
                m_a = stage_i == "after"
                if np.any(m_b):
                    vn_before.append(vn[m_b])
                if np.any(m_a):
                    vn_after.append(vn[m_a])

        if not vn_all:
            J[i] = float("nan")
            frac_ok[i] = float("nan")
            max_vn[i] = float("nan")
            J_before[i] = float("nan")
            J_after[i] = float("nan")
            frac_ok_before[i] = float("nan")
            frac_ok_after[i] = float("nan")
            continue

        vn_all_cat = np.concatenate(vn_all) if vn_all else np.array([], dtype=float)
        J[i] = float(np.sum(vn_all_cat**2))
        max_vn[i] = float(np.max(vn_all_cat)) if vn_all_cat.size else float("nan")
        frac_ok[i] = float(np.mean(vn_all_cat <= cfg.tau)) if vn_all_cat.size else float("nan")

        p_cat = np.concatenate(p_all_parts) if p_all_parts else np.array([], dtype=float)
        p_tc = np.concatenate(p_tc_parts) if p_tc_parts else np.array([], dtype=float)
        p_vp = np.concatenate(p_vp_parts) if p_vp_parts else np.array([], dtype=float)
        p_vs = np.concatenate(p_vs_parts) if p_vs_parts else np.array([], dtype=float)
        if p_cat.size:
            logP_all[i] = float(np.sum(np.log(p_cat)))
            mean_p[i] = float(np.mean(p_cat))
        else:
            logP_all[i] = float("nan")
            mean_p[i] = float("nan")

        logP_tc[i] = float(np.sum(np.log(p_tc))) if p_tc.size else float("nan")
        logP_vp[i] = float(np.sum(np.log(p_vp))) if p_vp.size else float("nan")
        logP_vs[i] = float(np.sum(np.log(p_vs))) if p_vs.size else float("nan")

        if cfg.prob_mode == "mc" and p_cat.size:
            # Monte-Carlo estimate for the joint probability uses all observations as a single vector.
            # We need corresponding lo/hi per observation, so we recompute compact arrays.
            y_all: list[np.ndarray] = []
            lo_all: list[np.ndarray] = []
            hi_all: list[np.ndarray] = []
            sig_all: list[np.ndarray] = []
            for (prop, state), d in arrays.items():
                phi_i = d["phi"]
                y_i = d["y"]
                sigma_i = d["sigma"]
                phase = cfg.phase_dry if state == "dry" else cfg.phase_wet
                if prop == "tc":
                    lo, hi = _bounds_tc(lam[i], phase, phi_i)
                else:
                    vp_lo, vp_hi, vs_lo, vs_hi = _bounds_vpvs(vp[i], vs[i], cfg.rho_matrix_kg_m3, phase, phi_i)
                    if prop == "vp":
                        lo, hi = vp_lo, vp_hi
                    else:
                        lo, hi = vs_lo, vs_hi
                y_all.append(y_i)
                lo_all.append(lo)
                hi_all.append(hi)
                sig_all.append(sigma_i)
            yv = np.concatenate(y_all)
            lov = np.concatenate(lo_all)
            hiv = np.concatenate(hi_all)
            sv = np.concatenate(sig_all)
            P_all_mc[i] = _prob_all_mc(yv, lov, hiv, sv, draws=cfg.mc_draws, rng=rng)

        if vn_before:
            vn_b = np.concatenate(vn_before)
            J_before[i] = float(np.sum(vn_b**2))
            frac_ok_before[i] = float(np.mean(vn_b <= cfg.tau))
        else:
            J_before[i] = float("nan")
            frac_ok_before[i] = float("nan")

        if vn_after:
            vn_a = np.concatenate(vn_after)
            J_after[i] = float(np.sum(vn_a**2))
            frac_ok_after[i] = float(np.mean(vn_a <= cfg.tau))
        else:
            J_after[i] = float("nan")
            frac_ok_after[i] = float("nan")

    df = pd.DataFrame(
        {
            "lambda_M": lam,
            "vp_M_m_s": vp,
            "vs_M_m_s": vs,
            "J": J,
            "frac_ok": frac_ok,
            "max_vn": max_vn,
            "J_before": J_before,
            "J_after": J_after,
            "frac_ok_before": frac_ok_before,
            "frac_ok_after": frac_ok_after,
            "logP_all": logP_all,
            "logP_tc": logP_tc,
            "logP_vp": logP_vp,
            "logP_vs": logP_vs,
            "mean_p": mean_p,
            "P_all_mc": P_all_mc,
        }
    )

    feasible = (df["frac_ok"] >= cfg.frac_ok_min) & (df["max_vn"] <= cfg.max_vn_max)
    df["feasible"] = feasible

    out_csv = cfg.out_dir / "scan_points.csv"
    df.to_csv(out_csv, index=False)

    df_feas = df[df["feasible"]].copy()
    if df_feas.empty:
        print(f"Saved: {out_csv}")
        print("No feasible points found with current thresholds. Try widening ranges or relaxing --frac-ok-min / --max-vn-max.")
        return

    # Min/max and a few quantiles of feasible set.
    summary_rows = []
    for col in ["lambda_M", "vp_M_m_s", "vs_M_m_s"]:
        summary_rows.append(
            {
                "param": col,
                "min": float(df_feas[col].min()),
                "p05": float(df_feas[col].quantile(0.05)),
                "p50": float(df_feas[col].quantile(0.50)),
                "p95": float(df_feas[col].quantile(0.95)),
                "max": float(df_feas[col].max()),
                "n_feasible": int(df_feas.shape[0]),
                "n_total": int(df.shape[0]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    out_summary_csv = cfg.out_dir / "feasible_minmax_summary.csv"
    summary.to_csv(out_summary_csv, index=False)
    df_to_latex_booktabs(
        summary,
        caption="Feasible matrix-parameter ranges from HS-bound penalty scan (no regularization).",
        label="tab:hs_feasible_minmax",
        out_path=cfg.out_dir / "feasible_minmax_summary.tex",
    )

    # Plot: pairwise scatter (feasible vs non-feasible), colored by J for feasible
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None  # type: ignore[assignment]

    if plt is not None:
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
        with plt.rc_context(rc):
            fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
            xys = [
                ("lambda_M", "vp_M_m_s", r"$\lambda_M$ (W/m·K)", r"$V_{P,M}$ (m/s)"),
                ("lambda_M", "vs_M_m_s", r"$\lambda_M$ (W/m·K)", r"$V_{S,M}$ (m/s)"),
                ("vp_M_m_s", "vs_M_m_s", r"$V_{P,M}$ (m/s)", r"$V_{S,M}$ (m/s)"),
            ]
            for ax, (x, y, xl, yl) in zip(axes, xys, strict=True):
                # non-feasible background
                dn = df[~df["feasible"]]
                ax.scatter(dn[x], dn[y], s=6, color="0.80", alpha=0.25, rasterized=True)
                # feasible colored by J (lower is better)
                sc = ax.scatter(df_feas[x], df_feas[y], s=10, c=df_feas["J"], cmap="viridis", alpha=0.85, rasterized=True)
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.grid(True, alpha=0.25)
            fig.colorbar(sc, ax=ax, label=r"$J=\sum (v/\sigma)^2$")
            fig.suptitle("Feasible matrix-parameter set (HS penalty)", y=1.02)
            out_png = cfg.out_dir / "feasible_set_pairwise.png"
            fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
            plt.close(fig)

    # Probability-weighted 95% intervals (using analytic logP by default)
    logP = df["logP_all"].to_numpy(dtype=float)
    mP = np.isfinite(logP)
    if np.any(mP):
        logP0 = logP[mP]
        logP_shift = logP0 - float(np.max(logP0))
        w = np.zeros_like(logP, dtype=float)
        w[mP] = np.exp(logP_shift)
        prob_summary_rows = []
        for col in ["lambda_M", "vp_M_m_s", "vs_M_m_s"]:
            v = df[col].to_numpy(dtype=float)
            prob_summary_rows.append(
                {
                    "param": col,
                    "p2.5": _weighted_quantile(v, w, 0.025),
                    "p50": _weighted_quantile(v, w, 0.50),
                    "p97.5": _weighted_quantile(v, w, 0.975),
                }
            )
        prob_summary = pd.DataFrame(prob_summary_rows)
        out_prob_csv = cfg.out_dir / "probability_weighted_95ci.csv"
        prob_summary.to_csv(out_prob_csv, index=False)
        df_to_latex_booktabs(
            prob_summary,
            caption="Probability-weighted 95\\% intervals for matrix parameters using $P(\\text{data in HS}\\mid\\theta)$ (Gaussian measurement errors).",
            label="tab:hs_prob_weighted_95ci",
            out_path=cfg.out_dir / "probability_weighted_95ci.tex",
        )

        # 1D weighted histograms / KDE-like curves (hist only; no SciPy dependency)
        if plt is not None:
            with plt.rc_context(rc):
                fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
                specs = [
                    ("lambda_M", r"$\lambda_M$ (W/m·K)"),
                    ("vp_M_m_s", r"$V_{P,M}$ (km/s)"),
                    ("vs_M_m_s", r"$V_{S,M}$ (km/s)"),
                ]
                for ax, (col, lab) in zip(axes, specs, strict=True):
                    v = df[col].to_numpy(dtype=float)
                    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
                    v = v[m]
                    ww = w[m]
                    ax.hist(v, bins=40, weights=ww, density=True, color="#4c72b0", alpha=0.35, edgecolor="none")
                    q025 = float(prob_summary.loc[prob_summary["param"] == col, "p2.5"].iloc[0])
                    q50 = float(prob_summary.loc[prob_summary["param"] == col, "p50"].iloc[0])
                    q975 = float(prob_summary.loc[prob_summary["param"] == col, "p97.5"].iloc[0])
                    ax.axvline(q50, color="#4c72b0", lw=2.0)
                    ax.axvline(q025, color="#4c72b0", lw=1.6, ls="--")
                    ax.axvline(q975, color="#4c72b0", lw=1.6, ls="--")
                    ax.set_xlabel(lab)
                    ax.set_ylabel("Weighted density")
                    ax.grid(True, alpha=0.25)
                fig.suptitle(r"Probability-weighted 95% intervals ($P(\mathrm{data\ in\ HS}\mid\theta)$)", y=1.02)
                out_png = cfg.out_dir / "probability_weighted_1d.png"
                fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
                plt.close(fig)

            # Show how each property contributes (TC-only / Vp-only / Vs-only / All) on one figure
            def _weights_from_logp(logp: np.ndarray) -> np.ndarray:
                logp = np.asarray(logp, dtype=float)
                m = np.isfinite(logp)
                w_ = np.zeros_like(logp, dtype=float)
                if not np.any(m):
                    return w_
                lp = logp[m]
                lp = lp - float(np.max(lp))
                w_[m] = np.exp(lp)
                return w_

            w_all = _weights_from_logp(df["logP_all"].to_numpy(dtype=float))
            w_tc = _weights_from_logp(df["logP_tc"].to_numpy(dtype=float))
            w_vp = _weights_from_logp(df["logP_vp"].to_numpy(dtype=float))
            w_vs = _weights_from_logp(df["logP_vs"].to_numpy(dtype=float))

            with plt.rc_context(rc):
                fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
                specs = [
                    ("lambda_M", r"$\lambda_M$ (W/m·K)"),
                    ("vp_M_m_s", r"$V_{P,M}$ (m/s)"),
                    ("vs_M_m_s", r"$V_{S,M}$ (m/s)"),
                ]
                series = [
                    ("All", w_all, "#4c72b0"),
                    ("TC only", w_tc, "#55a868"),
                    ("Vp only", w_vp, "#c44e52"),
                    ("Vs only", w_vs, "#8172b3"),
                ]
                for ax, (col, lab) in zip(axes, specs, strict=True):
                    v = df[col].to_numpy(dtype=float)
                    if col in {"vp_M_m_s", "vs_M_m_s"}:
                        v = v / 1000.0
                    m0 = np.isfinite(v)
                    v0 = v[m0]
                    if v0.size == 0:
                        continue
                    # common bins
                    bins = np.linspace(float(np.min(v0)), float(np.max(v0)), 45)
                    for name, ww, color in series:
                        m = m0 & np.isfinite(ww) & (ww > 0)
                        if not np.any(m):
                            continue
                        ax.hist(v[m], bins=bins, weights=ww[m], density=True, histtype="step", lw=2.0, color=color, label=name)
                    ax.set_xlabel(lab)
                    ax.set_ylabel("Weighted density")
                    ax.grid(True, alpha=0.25)
                axes[0].legend(frameon=False, loc="upper left")
                fig.suptitle("Probability weights by property (P(data in HS | θ))", y=1.02)
                out_png = cfg.out_dir / "probability_weighted_by_property_1d.png"
                fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
                plt.close(fig)

            # 2D weighted contours via weighted 2D hist
            with plt.rc_context(rc):
                fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
                pairs = [
                    ("lambda_M", "vp_M_m_s", r"$\lambda_M$ (W/m·K)", r"$V_{P,M}$ (km/s)"),
                    ("lambda_M", "vs_M_m_s", r"$\lambda_M$ (W/m·K)", r"$V_{S,M}$ (km/s)"),
                    ("vp_M_m_s", "vs_M_m_s", r"$V_{P,M}$ (km/s)", r"$V_{S,M}$ (km/s)"),
                ]
                for ax, (xcol, ycol, xl, yl) in zip(axes, pairs, strict=True):
                    x = df[xcol].to_numpy(dtype=float)
                    y = df[ycol].to_numpy(dtype=float)
                    if xcol in {"vp_M_m_s", "vs_M_m_s"}:
                        x = x / 1000.0
                    if ycol in {"vp_M_m_s", "vs_M_m_s"}:
                        y = y / 1000.0
                    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
                    x = x[m]
                    y = y[m]
                    ww = w[m]
                    H, xe, ye = np.histogram2d(x, y, bins=70, weights=ww, density=False)
                    # Determine 95% highest-density region threshold
                    flat = H.ravel()
                    order = np.argsort(flat)[::-1]
                    csum = np.cumsum(flat[order])
                    total = csum[-1] if csum.size else 1.0
                    idx95 = int(np.searchsorted(csum, 0.95 * total, side="left"))
                    thr = float(flat[order[min(idx95, order.size - 1)]]) if order.size else 0.0
                    xc = 0.5 * (xe[:-1] + xe[1:])
                    yc = 0.5 * (ye[:-1] + ye[1:])
                    X, Y = np.meshgrid(xc, yc, indexing="ij")
                    # Pastel-ish sequential palette for thesis figures
                    pcm = ax.pcolormesh(xe, ye, H.T, cmap="YlGnBu", shading="auto")
                    ax.contour(X, Y, H, levels=[thr], colors=["white"], linewidths=2.0)
                    fig.colorbar(pcm, ax=ax, label=r"$\sum w(\theta)$")
                    ax.set_xlabel(xl)
                    ax.set_ylabel(yl)
                    ax.grid(True, alpha=0.18)
                out_png = cfg.out_dir / "probability_weighted_2d_contours.png"
                fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved: {out_prob_csv}")

    best = df.sort_values("J").iloc[0]
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_summary_csv}")
    print("Best (min J) point:")
    print(best[["lambda_M", "vp_M_m_s", "vs_M_m_s", "J", "frac_ok", "max_vn"]].to_string())  # noqa: T201
    print("Feasible min–max:")
    print(summary.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
