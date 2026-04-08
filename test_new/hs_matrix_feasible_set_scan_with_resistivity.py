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
    # parameter ranges
    lambda_min: float
    lambda_max: float
    vp_min: float
    vp_max: float
    vs_min: float
    vs_max: float
    rho_matrix_kg_m3: float
    # electrical conductivity of matrix (log-uniform sampling)
    sigmaM_min: float
    sigmaM_max: float
    # brine resistivity fixed per stage (Ohm·m); conductivity is sigma_f = 1/R_f
    brine_resistivity_before_ohm_m: float
    brine_resistivity_after_ohm_m: float
    # measurement uncertainty (relative)
    tc_rel_unc: float
    v_rel_unc: float
    ec_rel_unc: float
    # feasibility criterion
    tau: float
    frac_ok_min: float
    max_vn_max: float
    # probability mode
    prob_mode: str  # analytic|mc
    mc_draws: int
    # pore phases (TC + elastic)
    phase_dry: PhaseProps
    phase_wet: PhaseProps
    # which properties to include
    use_tc: bool
    use_vp: bool
    use_vs: bool
    use_ec: bool
    dpi: int = 300


def _load_arrays(cfg: ScanConfig) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """
    Returns dict keyed by (property, fluid_state) with arrays:
      phi, y, sigma, stage (as string array)
    property in {"tc","vp","vs","ec"}; ec is electrical conductivity sigma=1/R.
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
        ("ec", "resistivity_ohm_m", cfg.ec_rel_unc),
    ]:
        if prop == "tc" and not cfg.use_tc:
            continue
        if prop == "vp" and not cfg.use_vp:
            continue
        if prop == "vs" and not cfg.use_vs:
            continue
        if prop == "ec" and not cfg.use_ec:
            continue
        for state in ["dry", "wet"]:
            sub = df[(df["fluid_state"] == state) & df[col].notna()].copy()
            if sub.empty:
                continue
            phi = sub["phi"].to_numpy(dtype=float)
            y_raw = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(phi) & np.isfinite(y_raw)
            phi = phi[m]
            y_raw = y_raw[m]
            stage = sub["stage"].to_numpy(dtype=str)[m]

            if prop in {"vp", "vs"}:
                y = y_raw / 1000.0  # km/s
            elif prop == "tc":
                y = y_raw
            else:
                # Electrical conductivity sigma = 1/R
                # (use sigma for HS bounds, not resistivity itself)
                y = np.where(y_raw > 0, 1.0 / y_raw, np.nan)

            m2 = np.isfinite(phi) & np.isfinite(y)
            if prop in {"tc", "ec"}:
                m2 = m2 & (y > 0)
            phi, y, stage = phi[m2], y[m2], stage[m2]
            if phi.size == 0:
                continue

            sigma = _safe_sigma(rel, y)
            out[(prop, state)] = {"phi": phi, "y": y, "sigma": sigma, "stage": stage}
    return out


def _bounds_tc(lambda_m: float, phase: PhaseProps, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = hs_upper_conductivity(lambda_m, phase.lambda_pore_w_mk, phi)
    lo = hs_lower_conductivity(lambda_m, phase.lambda_pore_w_mk, phi)
    return lo, hi


def _bounds_ec(
    sigma_m: float,
    *,
    brine_resistivity_before_ohm_m: float,
    brine_resistivity_after_ohm_m: float,
    phi: np.ndarray,
    stage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Electrical conductivity HS bounds, with pore-phase conductivity fixed per stage:
      sigma_f = 1 / R_f
    """
    stage_s = stage.astype(str)
    lo = np.full_like(phi, np.nan, dtype=float)
    hi = np.full_like(phi, np.nan, dtype=float)
    # Convert fixed brine resistivity to conductivity (S/m).
    sigma_before = 1.0 / max(float(brine_resistivity_before_ohm_m), 1e-12)
    sigma_after = 1.0 / max(float(brine_resistivity_after_ohm_m), 1e-12)
    m_b = stage_s == "before"
    m_a = stage_s == "after"
    if np.any(m_b):
        b1 = hs_upper_conductivity(float(sigma_m), float(sigma_before), phi[m_b])
        b2 = hs_lower_conductivity(float(sigma_m), float(sigma_before), phi[m_b])
        lo[m_b] = np.minimum(b1, b2)
        hi[m_b] = np.maximum(b1, b2)
    if np.any(m_a):
        b1 = hs_upper_conductivity(float(sigma_m), float(sigma_after), phi[m_a])
        b2 = hs_lower_conductivity(float(sigma_m), float(sigma_after), phi[m_a])
        lo[m_a] = np.minimum(b1, b2)
        hi[m_a] = np.maximum(b1, b2)
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
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if y.size == 0:
        return float("nan")
    eps = rng.normal(loc=0.0, scale=sigma, size=(int(draws), y.size))
    ys = y[None, :] + eps
    ok = (ys >= lo[None, :]) & (ys <= hi[None, :])
    return float(np.mean(np.all(ok, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan feasible matrix-property ranges using HS-bound penalty (P2 weights), including electrical resistivity (via conductivity)."
    )
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/hs_matrix_feasible_set_with_resistivity"))
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lambda-min", type=float, default=2.70)
    ap.add_argument("--lambda-max", type=float, default=3.20)
    ap.add_argument("--vp-min", type=float, default=6000.0)
    ap.add_argument("--vp-max", type=float, default=7600.0)
    ap.add_argument("--vs-min", type=float, default=3200.0)
    ap.add_argument("--vs-max", type=float, default=4400.0)
    ap.add_argument("--rho-matrix", type=float, default=2725.0)

    ap.add_argument("--sigmaM-min", type=float, default=1e-6, help="Matrix electrical conductivity lower bound (S/m).")
    ap.add_argument("--sigmaM-max", type=float, default=1e-2, help="Matrix electrical conductivity upper bound (S/m).")

    # Fixed brine resistivity for 20 g/L NaCl (user-provided), per stage (Ohm·m).
    ap.add_argument("--brine-res-before", type=float, default=0.32, help="Brine resistivity R_f (Ohm·m) for 'before'.")
    ap.add_argument("--brine-res-after", type=float, default=0.26, help="Brine resistivity R_f (Ohm·m) for 'after'.")

    ap.add_argument("--tc-rel-unc", type=float, default=0.025)
    ap.add_argument("--v-rel-unc", type=float, default=0.05)
    ap.add_argument("--ec-rel-unc", type=float, default=0.05, help="Relative uncertainty for electrical conductivity (sigma=1/R).")

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
    ap.add_argument("--use-ec", action="store_true", default=True)
    ap.add_argument("--no-ec", dest="use_ec", action="store_false")

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
        sigmaM_min=float(args.sigmaM_min),
        sigmaM_max=float(args.sigmaM_max),
        brine_resistivity_before_ohm_m=float(args.brine_res_before),
        brine_resistivity_after_ohm_m=float(args.brine_res_after),
        tc_rel_unc=float(args.tc_rel_unc),
        v_rel_unc=float(args.v_rel_unc),
        ec_rel_unc=float(args.ec_rel_unc),
        tau=float(args.tau),
        frac_ok_min=float(args.frac_ok_min),
        max_vn_max=float(args.max_vn_max),
        prob_mode=str(args.prob_mode),
        mc_draws=int(args.mc_draws),
        phase_dry=PhaseProps(lambda_pore_w_mk=float(args.lambda_air), K_pore_gpa=float(args.K_air_gpa), rho_pore_kg_m3=float(args.rho_air)),
        phase_wet=PhaseProps(lambda_pore_w_mk=float(args.lambda_brine), K_pore_gpa=float(args.K_brine_gpa), rho_pore_kg_m3=float(args.rho_brine)),
        use_tc=bool(args.use_tc),
        use_vp=bool(args.use_vp),
        use_vs=bool(args.use_vs),
        use_ec=bool(args.use_ec),
        dpi=int(args.dpi),
    )

    _configure_matplotlib_env(cfg.out_dir)
    arrays = _load_arrays(cfg)
    rng = np.random.default_rng(cfg.seed)

    # Sample parameters
    lam = rng.uniform(cfg.lambda_min, cfg.lambda_max, size=cfg.n_samples)
    vp = rng.uniform(cfg.vp_min, cfg.vp_max, size=cfg.n_samples)
    vs = rng.uniform(cfg.vs_min, cfg.vs_max, size=cfg.n_samples)
    log10_sigmaM = rng.uniform(np.log10(cfg.sigmaM_min), np.log10(cfg.sigmaM_max), size=cfg.n_samples)
    sigmaM = 10.0 ** log10_sigmaM

    J = np.full(cfg.n_samples, np.nan, dtype=float)
    frac_ok = np.full(cfg.n_samples, np.nan, dtype=float)
    max_vn = np.full(cfg.n_samples, np.nan, dtype=float)
    J_before = np.full(cfg.n_samples, np.nan, dtype=float)
    J_after = np.full(cfg.n_samples, np.nan, dtype=float)
    frac_ok_before = np.full(cfg.n_samples, np.nan, dtype=float)
    frac_ok_after = np.full(cfg.n_samples, np.nan, dtype=float)

    logP_all = np.full(cfg.n_samples, np.nan, dtype=float)
    logP_tc = np.full(cfg.n_samples, np.nan, dtype=float)
    logP_vp = np.full(cfg.n_samples, np.nan, dtype=float)
    logP_vs = np.full(cfg.n_samples, np.nan, dtype=float)
    logP_ec = np.full(cfg.n_samples, np.nan, dtype=float)
    mean_p = np.full(cfg.n_samples, np.nan, dtype=float)
    P_all_mc = np.full(cfg.n_samples, np.nan, dtype=float)

    for i in range(cfg.n_samples):
        vn_all: list[np.ndarray] = []
        vn_before: list[np.ndarray] = []
        vn_after: list[np.ndarray] = []
        p_all_parts: list[np.ndarray] = []
        p_tc_parts: list[np.ndarray] = []
        p_vp_parts: list[np.ndarray] = []
        p_vs_parts: list[np.ndarray] = []
        p_ec_parts: list[np.ndarray] = []

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
            elif prop == "ec":
                if state != "wet":
                    continue
                lo, hi = _bounds_ec(
                    sigmaM[i],
                    brine_resistivity_before_ohm_m=cfg.brine_resistivity_before_ohm_m,
                    brine_resistivity_after_ohm_m=cfg.brine_resistivity_after_ohm_m,
                    phi=phi_i,
                    stage=stage_i,
                )
                v = _violation(y_i, lo, hi)
                p = _prob_inside_interval(y_i, lo, hi, sigma_i)
                p_ec_parts.append(p)
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
            continue

        vn_all_cat = np.concatenate(vn_all)
        J[i] = float(np.sum(vn_all_cat**2))
        max_vn[i] = float(np.max(vn_all_cat)) if vn_all_cat.size else float("nan")
        frac_ok[i] = float(np.mean(vn_all_cat <= cfg.tau)) if vn_all_cat.size else float("nan")

        p_cat = np.concatenate(p_all_parts) if p_all_parts else np.array([], dtype=float)
        p_tc = np.concatenate(p_tc_parts) if p_tc_parts else np.array([], dtype=float)
        p_vp = np.concatenate(p_vp_parts) if p_vp_parts else np.array([], dtype=float)
        p_vs = np.concatenate(p_vs_parts) if p_vs_parts else np.array([], dtype=float)
        p_ec = np.concatenate(p_ec_parts) if p_ec_parts else np.array([], dtype=float)

        if p_cat.size:
            logP_all[i] = float(np.sum(np.log(p_cat)))
            mean_p[i] = float(np.mean(p_cat))
        logP_tc[i] = float(np.sum(np.log(p_tc))) if p_tc.size else float("nan")
        logP_vp[i] = float(np.sum(np.log(p_vp))) if p_vp.size else float("nan")
        logP_vs[i] = float(np.sum(np.log(p_vs))) if p_vs.size else float("nan")
        logP_ec[i] = float(np.sum(np.log(p_ec))) if p_ec.size else float("nan")

        if cfg.prob_mode == "mc" and p_cat.size:
            y_all: list[np.ndarray] = []
            lo_all: list[np.ndarray] = []
            hi_all: list[np.ndarray] = []
            sig_all: list[np.ndarray] = []
            for (prop, state), d in arrays.items():
                phi_i = d["phi"]
                y_i = d["y"]
                sigma_i = d["sigma"]
                stage_i = d["stage"]
                phase = cfg.phase_dry if state == "dry" else cfg.phase_wet
                if prop == "tc":
                    lo, hi = _bounds_tc(lam[i], phase, phi_i)
                elif prop == "ec":
                    if state != "wet":
                        continue
                    lo, hi = _bounds_ec(
                        sigmaM[i],
                        brine_resistivity_before_ohm_m=cfg.brine_resistivity_before_ohm_m,
                        brine_resistivity_after_ohm_m=cfg.brine_resistivity_after_ohm_m,
                        phi=phi_i,
                        stage=stage_i,
                    )
                else:
                    vp_lo, vp_hi, vs_lo, vs_hi = _bounds_vpvs(vp[i], vs[i], cfg.rho_matrix_kg_m3, phase, phi_i)
                    lo, hi = (vp_lo, vp_hi) if prop == "vp" else (vs_lo, vs_hi)
                y_all.append(y_i)
                lo_all.append(lo)
                hi_all.append(hi)
                sig_all.append(sigma_i)
            yv = np.concatenate(y_all) if y_all else np.array([], dtype=float)
            lov = np.concatenate(lo_all) if lo_all else np.array([], dtype=float)
            hiv = np.concatenate(hi_all) if hi_all else np.array([], dtype=float)
            sv = np.concatenate(sig_all) if sig_all else np.array([], dtype=float)
            if yv.size:
                P_all_mc[i] = _prob_all_mc(yv, lov, hiv, sv, draws=cfg.mc_draws, rng=rng)

        if vn_before:
            vn_b = np.concatenate(vn_before)
            J_before[i] = float(np.sum(vn_b**2))
            frac_ok_before[i] = float(np.mean(vn_b <= cfg.tau))
        if vn_after:
            vn_a = np.concatenate(vn_after)
            J_after[i] = float(np.sum(vn_a**2))
            frac_ok_after[i] = float(np.mean(vn_a <= cfg.tau))

    df = pd.DataFrame(
        {
            "lambda_M": lam,
            "vp_M_m_s": vp,
            "vs_M_m_s": vs,
            "sigmaM_S_m": sigmaM,
            "log10_sigmaM": log10_sigmaM,
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
            "logP_ec": logP_ec,
            "mean_p": mean_p,
            "P_all_mc": P_all_mc,
        }
    )

    feasible = (df["frac_ok"] >= cfg.frac_ok_min) & (df["max_vn"] <= cfg.max_vn_max)
    df["feasible"] = feasible

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cfg.out_dir / "scan_points.csv"
    df.to_csv(out_csv, index=False)

    summary_rows = []
    for col in ["lambda_M", "vp_M_m_s", "vs_M_m_s", "sigmaM_S_m", "log10_sigmaM"]:
        v = df.loc[df["feasible"], col].to_numpy(dtype=float)
        summary_rows.append(
            {
                "param": col,
                "min": float(np.nanmin(v)) if v.size else float("nan"),
                "p05": float(np.nanpercentile(v, 5)) if v.size else float("nan"),
                "p50": float(np.nanpercentile(v, 50)) if v.size else float("nan"),
                "p95": float(np.nanpercentile(v, 95)) if v.size else float("nan"),
                "max": float(np.nanmax(v)) if v.size else float("nan"),
                "n_feasible": int(np.sum(df["feasible"])),
                "n_total": int(cfg.n_samples),
            }
        )
    out_summary_csv = cfg.out_dir / "feasible_minmax_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)

    # Probability-weighted 95% intervals using weights ~ exp(logP_all - max)
    logp = df["logP_all"].to_numpy(dtype=float)
    m = np.isfinite(logp)
    weights = np.zeros_like(logp, dtype=float)
    if np.any(m):
        lp = logp[m]
        lp = lp - float(np.max(lp))
        weights[m] = np.exp(lp)

    out_prob_csv = cfg.out_dir / "probability_weighted_95ci.csv"
    prob_rows = []
    for col in ["lambda_M", "vp_M_m_s", "vs_M_m_s", "sigmaM_S_m", "log10_sigmaM"]:
        vv = df[col].to_numpy(dtype=float)
        prob_rows.append(
            {
                "param": col,
                "p025": _weighted_quantile(vv, weights, 0.025),
                "p50": _weighted_quantile(vv, weights, 0.50),
                "p975": _weighted_quantile(vv, weights, 0.975),
            }
        )
    pd.DataFrame(prob_rows).to_csv(out_prob_csv, index=False)

    # Plots
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

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

    def _weights_from_logp(series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        mm = np.isfinite(series)
        w_ = np.zeros_like(series, dtype=float)
        if not np.any(mm):
            return w_
        lp_ = series[mm]
        lp_ = lp_ - float(np.max(lp_))
        w_[mm] = np.exp(lp_)
        return w_

    w_all = _weights_from_logp(df["logP_all"].to_numpy(dtype=float))
    w_tc = _weights_from_logp(df["logP_tc"].to_numpy(dtype=float))
    w_vp = _weights_from_logp(df["logP_vp"].to_numpy(dtype=float))
    w_vs = _weights_from_logp(df["logP_vs"].to_numpy(dtype=float))
    w_ec = _weights_from_logp(df["logP_ec"].to_numpy(dtype=float))

    with plt.rc_context(rc):
        # 1D weighted marginals (4 panels)
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.8), constrained_layout=True)
        specs = [
            ("lambda_M", r"$\lambda_M$ (W/m$\cdot$K)", None),
            ("vp_M_m_s", r"$V_{P,M}$ (km/s)", 1e-3),
            ("vs_M_m_s", r"$V_{S,M}$ (km/s)", 1e-3),
            ("log10_sigmaM", r"$\log_{10}\sigma_M$ (S/m)", None),
        ]
        for ax, (col, lab, scale) in zip(axes, specs, strict=True):
            v = df[col].to_numpy(dtype=float)
            if scale is not None:
                v = v * float(scale)
            m0 = np.isfinite(v) & np.isfinite(w_all) & (w_all > 0)
            if not np.any(m0):
                continue
            v0 = v[m0]
            bins = np.linspace(float(np.min(v0)), float(np.max(v0)), 55)
            ax.hist(v0, bins=bins, weights=w_all[m0], density=True, color="#4c72b0", alpha=0.55, edgecolor="none")
            ax.set_xlabel(lab)
            ax.set_ylabel("Weighted density")
            ax.grid(True, alpha=0.25)
        out_png = cfg.out_dir / "probability_weighted_1d.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context(rc):
        # 1D overlay by property (All/TC/Vp/Vs/EC)
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.8), constrained_layout=True)
        specs = [
            ("lambda_M", r"$\lambda_M$ (W/m$\cdot$K)", None),
            ("vp_M_m_s", r"$V_{P,M}$ (km/s)", 1e-3),
            ("vs_M_m_s", r"$V_{S,M}$ (km/s)", 1e-3),
            ("log10_sigmaM", r"$\log_{10}\sigma_M$ (S/m)", None),
        ]
        series = [
            ("All", w_all, "#4c72b0"),
            ("TC only", w_tc, "#55a868"),
            ("Vp only", w_vp, "#c44e52"),
            ("Vs only", w_vs, "#8172b3"),
            ("EC only", w_ec, "#ccb974"),
        ]
        for ax, (col, lab, scale) in zip(axes, specs, strict=True):
            v = df[col].to_numpy(dtype=float)
            if scale is not None:
                v = v * float(scale)
            m0 = np.isfinite(v)
            v0 = v[m0]
            if v0.size == 0:
                continue
            bins = np.linspace(float(np.min(v0)), float(np.max(v0)), 45)
            for name, ww, color in series:
                mm = m0 & np.isfinite(ww) & (ww > 0)
                if not np.any(mm):
                    continue
                vv = v[mm]
                ax.hist(vv, bins=bins, weights=ww[mm], density=True, histtype="step", lw=2.0, color=color, label=name)
            ax.set_xlabel(lab)
            ax.set_ylabel("Weighted density")
            ax.grid(True, alpha=0.25)
        axes[0].legend(frameon=False, loc="upper left")
        out_png = cfg.out_dir / "probability_weighted_by_property_1d.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context(rc):
        # 2D weighted contours (6 panels)
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)
        pairs = [
            ("lambda_M", "vp_M_m_s", r"$\lambda_M$ (W/m$\cdot$K)", r"$V_{P,M}$ (km/s)", (None, 1e-3)),
            ("lambda_M", "vs_M_m_s", r"$\lambda_M$ (W/m$\cdot$K)", r"$V_{S,M}$ (km/s)", (None, 1e-3)),
            ("vp_M_m_s", "vs_M_m_s", r"$V_{P,M}$ (km/s)", r"$V_{S,M}$ (km/s)", (1e-3, 1e-3)),
            ("lambda_M", "log10_sigmaM", r"$\lambda_M$ (W/m$\cdot$K)", r"$\log_{10}\sigma_M$ (S/m)", (None, None)),
            ("vp_M_m_s", "log10_sigmaM", r"$V_{P,M}$ (km/s)", r"$\log_{10}\sigma_M$ (S/m)", (1e-3, None)),
            ("vs_M_m_s", "log10_sigmaM", r"$V_{S,M}$ (km/s)", r"$\log_{10}\sigma_M$ (S/m)", (1e-3, None)),
        ]
        for ax, (xcol, ycol, xl, yl, scales) in zip(axes.ravel(), pairs, strict=True):
            xs = df[xcol].to_numpy(dtype=float)
            ys = df[ycol].to_numpy(dtype=float)
            sx, sy = scales
            if sx is not None:
                xs = xs * float(sx)
            if sy is not None:
                ys = ys * float(sy)
            m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(w_all) & (w_all > 0)
            x = xs[m]
            y = ys[m]
            ww = w_all[m]
            if x.size == 0:
                continue
            H, xe, ye = np.histogram2d(x, y, bins=70, weights=ww, density=False)
            flat = H.ravel()
            order = np.argsort(flat)[::-1]
            csum = np.cumsum(flat[order])
            total = csum[-1] if csum.size else 1.0
            idx95 = int(np.searchsorted(csum, 0.95 * total, side="left"))
            thr = float(flat[order[min(idx95, order.size - 1)]]) if order.size else 0.0
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            X, Y = np.meshgrid(xc, yc, indexing="ij")
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
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_summary_csv}")
    best = df.sort_values("J").iloc[0]
    print("Best (min J) point:")
    print(best[["lambda_M", "vp_M_m_s", "vs_M_m_s", "sigmaM_S_m", "log10_sigmaM", "J", "frac_ok", "max_vn"]].to_string())  # noqa: T201
    print("Feasible min–max:")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
