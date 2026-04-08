from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _configure_matplotlib_env(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def fit_linear(phi: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y = a * phi + b (phi is fraction).
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(phi) & np.isfinite(y)
    phi = phi[m]
    y = y[m]
    if phi.size < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(phi, y, deg=1)
    return float(a), float(b)


def fit_exp(phi: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y = A * exp(B * phi) (phi is fraction).
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
    phi = phi[m]
    y = y[m]
    if phi.size < 2:
        return float("nan"), float("nan")
    B, lnA = np.polyfit(phi, np.log(y), deg=1)
    return float(np.exp(lnA)), float(B)


def fit_slowness_linear(phi: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """
    Fit 1/v = a * phi + b (phi is fraction). Returns (a, b).
    """
    phi = np.asarray(phi, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(phi) & np.isfinite(v) & (v > 0)
    phi = phi[m]
    v = v[m]
    if phi.size < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(phi, 1.0 / v, deg=1)
    return float(a), float(b)


def choose_best_model(phi: np.ndarray, y: np.ndarray, *, prefer: str | None = None) -> dict[str, float | str]:
    """
    Compare linear vs exponential by R^2 and return a small dict with model + params.
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(phi) & np.isfinite(y)
    phi = phi[m]
    y = y[m]
    if phi.size < 2:
        return {"model": "none", "r2": float("nan"), "a": float("nan"), "b": float("nan")}

    a, b = fit_linear(phi, y)
    y_lin = a * phi + b
    r2_lin = r2_score(y, y_lin)

    A, B = fit_exp(phi, y)
    y_exp = A * np.exp(B * phi) if np.isfinite(A) and np.isfinite(B) else np.full_like(y, np.nan)
    r2_exp = r2_score(y, y_exp)

    if prefer == "linear" and np.isfinite(r2_lin):
        return {"model": "linear", "r2": float(r2_lin), "a": float(a), "b": float(b)}
    if prefer == "exp" and np.isfinite(r2_exp):
        return {"model": "exp", "r2": float(r2_exp), "A": float(A), "B": float(B)}

    if np.isfinite(r2_exp) and (not np.isfinite(r2_lin) or r2_exp >= r2_lin):
        return {"model": "exp", "r2": float(r2_exp), "A": float(A), "B": float(B)}
    return {"model": "linear", "r2": float(r2_lin), "a": float(a), "b": float(b)}


def choose_shared_vpvs_model(
    phi_vp: np.ndarray,
    vp: np.ndarray,
    phi_vs: np.ndarray,
    vs: np.ndarray,
    *,
    mode: str = "auto",
) -> str:
    """
    Choose a single regression *family* for both Vp and Vs (same functional form),
    using average R^2 over both properties.

    mode:
      - auto: pick best among linear / exp / slowness_linear by avg R^2
      - linear|exp|slowness_linear: force that family
    """
    if mode in {"linear", "exp", "slowness_linear"}:
        return mode

    candidates = ["linear", "exp", "slowness_linear"]
    scores: dict[str, float] = {}
    phi_vp = np.asarray(phi_vp, dtype=float)
    vp = np.asarray(vp, dtype=float)
    phi_vs = np.asarray(phi_vs, dtype=float)
    vs = np.asarray(vs, dtype=float)

    for cand in candidates:
        r2s: list[float] = []

        # Vp
        m = np.isfinite(phi_vp) & np.isfinite(vp) & (vp > 0)
        pv, yv = phi_vp[m], vp[m]
        if pv.size >= 2:
            if cand == "linear":
                a, b = fit_linear(pv, yv)
                r2s.append(r2_score(yv, a * pv + b))
            elif cand == "exp":
                A, B = fit_exp(pv, yv)
                r2s.append(r2_score(yv, A * np.exp(B * pv)))
            else:
                a, b = fit_slowness_linear(pv, yv)
                r2s.append(r2_score(1.0 / yv, a * pv + b))

        # Vs
        m = np.isfinite(phi_vs) & np.isfinite(vs) & (vs > 0)
        ps, ys = phi_vs[m], vs[m]
        if ps.size >= 2:
            if cand == "linear":
                a, b = fit_linear(ps, ys)
                r2s.append(r2_score(ys, a * ps + b))
            elif cand == "exp":
                A, B = fit_exp(ps, ys)
                r2s.append(r2_score(ys, A * np.exp(B * ps)))
            else:
                a, b = fit_slowness_linear(ps, ys)
                r2s.append(r2_score(1.0 / ys, a * ps + b))

        scores[cand] = float(np.nanmean(r2s)) if r2s else float("-inf")

    return max(scores.items(), key=lambda kv: kv[1])[0]


def hs_upper_conductivity(k_matrix: float, k_pore: float, phi: np.ndarray) -> np.ndarray:
    """
    Hashin–Shtrikman upper bound for scalar conductivity where the matrix is the host.
    """
    phi = np.asarray(phi, dtype=float)
    km = float(k_matrix)
    kf = float(k_pore)
    denom = (1.0 / (kf - km)) + (1.0 - phi) / (3.0 * km)
    return km + phi / denom


def hs_lower_conductivity(k_matrix: float, k_pore: float, phi: np.ndarray) -> np.ndarray:
    """
    Hashin–Shtrikman lower bound for scalar conductivity where the pore phase is the host.
    """
    phi = np.asarray(phi, dtype=float)
    km = float(k_matrix)
    kf = float(k_pore)
    denom = (1.0 / (km - kf)) + phi / (3.0 * kf)
    return kf + (1.0 - phi) / denom


def _zeta_shear(K: float, G: float) -> float:
    # HS helper for shear.
    return G * (9.0 * K + 8.0 * G) / (6.0 * (K + 2.0 * G))


def hs_upper_elastic(Km: float, Gm: float, Kf: float, Gf: float, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    HS upper bounds for (K, G) using matrix as host.
    """
    phi = np.asarray(phi, dtype=float)
    f2 = phi
    f1 = 1.0 - phi
    Km = float(Km)
    Gm = float(Gm)
    Kf = float(Kf)
    Gf = float(Gf)

    K = Km + f2 / ((1.0 / (Kf - Km)) + f1 / (Km + 4.0 * Gm / 3.0))

    zeta = _zeta_shear(Km, Gm)
    G = Gm + f2 / ((1.0 / (Gf - Gm)) + f1 / (Gm + zeta))
    return K, G


def hs_lower_elastic(Km: float, Gm: float, Kf: float, Gf: float, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    HS lower bounds for (K, G) using pore phase as host.

    Note: for fluids, Gf = 0 implies the shear lower bound is 0 (fluid-connected).
    """
    phi = np.asarray(phi, dtype=float)
    f2 = phi
    f1 = 1.0 - phi
    Km = float(Km)
    Gm = float(Gm)
    Kf = float(Kf)
    Gf = float(Gf)

    # Bulk modulus lower bound remains meaningful for Kf > 0.
    K = Kf + f1 / ((1.0 / (Km - Kf)) + f2 / (Kf + 4.0 * Gf / 3.0))

    if Gf <= 0.0:
        G = np.zeros_like(phi, dtype=float)
    else:
        zeta = _zeta_shear(Kf, Gf)
        G = Gf + f1 / ((1.0 / (Gm - Gf)) + f2 / (Gf + zeta))

    return K, G


def velocities_from_KG_rho(K: np.ndarray, G: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Vp, Vs) in m/s from (K, G) in Pa and rho in kg/m^3.
    """
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    rho = np.asarray(rho, dtype=float)
    vp = np.sqrt(np.maximum(K + 4.0 * G / 3.0, 0.0) / rho)
    vs = np.sqrt(np.maximum(G, 0.0) / rho)
    return vp, vs


def _select_close_to_bounds(
    y: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    *,
    frac: float,
) -> np.ndarray:
    """
    Flag points close to HS bounds (or outside) to draw uncertainty error bars.
    frac is the normalized distance to a bound within the HS range.
    """
    y = np.asarray(y, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    span = np.maximum(y_hi - y_lo, 1e-12)
    t = (y - y_lo) / span
    return (t <= frac) | (t >= (1.0 - frac)) | (y < y_lo) | (y > y_hi)


@dataclass(frozen=True)
class PhaseProps:
    # Transport
    lambda_pore_w_mk: float
    # Elastic
    K_pore_gpa: float
    rho_pore_kg_m3: float


@dataclass(frozen=True)
class Config:
    data_xlsx: Path
    out_dir: Path
    stage: str  # before|after
    close_frac: float
    tc_rel_unc: float
    v_rel_unc: float
    lambda_matrix_w_mk: float
    vp_matrix_m_s: float
    vs_matrix_m_s: float
    rho_matrix_kg_m3: float
    phase_dry: PhaseProps
    phase_wet: PhaseProps
    dpi: int = 300
    # Plot ranges (thesis-friendly)
    tc_ylim: tuple[float, float] = (1.2, 3.0)
    vp_ylim: tuple[float, float] = (3.0, 7.0)
    vs_ylim: tuple[float, float] = (1.5, 4.0)
    vpvs_reg_model: str = "linear"  # auto|linear|exp|slowness_linear


def load_measurements(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.data_xlsx, sheet_name="measurements_long")
    df["stage"] = df["stage"].astype(str).str.strip()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip()
    df["phi_pct"] = pd.to_numeric(df["phi_pct"], errors="coerce")
    df["phi"] = df["phi_pct"] / 100.0
    df = df[df["stage"] == cfg.stage].copy()
    df = df[df["phi"].notna()].copy()
    return df


def _matrix_moduli_from_velocities(vp_m_s: float, vs_m_s: float, rho_kg_m3: float) -> tuple[float, float]:
    rho = float(rho_kg_m3)
    vp = float(vp_m_s)
    vs = float(vs_m_s)
    G = rho * vs**2
    K = rho * vp**2 - 4.0 * G / 3.0
    return K, G


def _plot_stage(cfg: Config) -> None:
    _configure_matplotlib_env(cfg.out_dir)
    df = load_measurements(cfg)

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

    colors = {"dry": "#d62728", "wet": "#1f77b4"}
    stage_label = "Before MPCT" if cfg.stage == "before" else "After MPCT"

    # HS bounds grids
    phi_grid = np.linspace(0.0, max(0.25, float(np.nanmax(df["phi"])) * 1.05), 250)

    # Matrix moduli
    Km_pa, Gm_pa = _matrix_moduli_from_velocities(cfg.vp_matrix_m_s, cfg.vs_matrix_m_s, cfg.rho_matrix_kg_m3)

    def _phase(state: str) -> PhaseProps:
        return cfg.phase_dry if state == "dry" else cfg.phase_wet

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
        fig.subplots_adjust(bottom=0.30, wspace=0.25)

        # --- TC panel ---
        ax = axes[0]
        regression_rows: list[dict[str, object]] = []
        for state in ["dry", "wet"]:
            g = df[df["fluid_state"] == state].copy()
            if g.empty:
                continue
            phi = g["phi"].to_numpy(dtype=float)
            y = pd.to_numeric(g["tc_w_mk"], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
            phi, y = phi[m], y[m]
            if phi.size == 0:
                continue

            ph = _phase(state)
            y_hi = hs_upper_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi)
            y_lo = hs_lower_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi)
            close = _select_close_to_bounds(y, y_lo, y_hi, frac=cfg.close_frac)

            # Main scatter
            ax.scatter(phi, y, s=26, color=colors[state], alpha=0.9)

            # Error bars only for near-bound points
            if np.any(close):
                ax.errorbar(
                    phi[close],
                    y[close],
                    yerr=cfg.tc_rel_unc * y[close],
                    fmt="none",
                    ecolor=colors[state],
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.95,
                )

            # HS bounds curves
            hi = hs_upper_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi_grid)
            lo = hs_lower_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi_grid)
            ax.plot(phi_grid, hi, color=colors[state], lw=1.8)
            ax.plot(phi_grid, lo, color=colors[state], lw=1.8, ls="--")

            # Regression: use the same simple family as velocities (linear in porosity)
            a, b = fit_linear(phi, y)
            ax.plot(phi_grid, a * phi_grid + b, color=colors[state], lw=1.2, ls=":")
            r2 = r2_score(y, a * phi + b)
            regression_rows.append(
                {
                    "stage": cfg.stage,
                    "fluid_state": state,
                    "property": "tc_w_mk",
                    "model": "linear",
                    "b_matrix": b,
                    "a": a,
                    "r2": r2,
                    "hs_matrix": cfg.lambda_matrix_w_mk,
                }
            )

        ax.set_xlabel("Porosity (φ)")
        ax.set_ylabel(r"Thermal conductivity, W/(m·K)")
        ax.set_xlim(0.0, 0.25)
        ax.set_ylim(*cfg.tc_ylim)
        ax.grid(True, alpha=0.25)
        ax.set_title("a) TC vs porosity", pad=8)

        # Put regression equations under the panel
        eq_lines: list[str] = []
        for row in regression_rows:
            if row["property"] != "tc_w_mk":
                continue
            st_raw = str(row["fluid_state"])
            st = "brine-sat." if st_raw == "wet" else st_raw
            if row["model"] == "exp":
                eq_lines.append(rf"{st}: $\lambda={row['A_matrix']:.2f}\,e^{{{row['B']:.3f}\phi}}$, $R^2={row['r2']:.2f}$")
            else:
                eq_lines.append(rf"{st}: $\lambda={row['a']:.2f}\phi+{row['b_matrix']:.2f}$, $R^2={row['r2']:.2f}$")
        ax.text(0.0, -0.36, "\n".join(eq_lines), transform=ax.transAxes, ha="left", va="top", fontsize=10)

        # --- Vp panel ---
        ax = axes[1]
        regression_rows_v: list[dict[str, object]] = []
        for state in ["dry", "wet"]:
            g = df[df["fluid_state"] == state].copy()
            if g.empty:
                continue
            phi = g["phi"].to_numpy(dtype=float)
            y = pd.to_numeric(g["vp_m_s"], errors="coerce").to_numpy(dtype=float) / 1000.0
            m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
            phi, y = phi[m], y[m]
            if phi.size == 0:
                continue

            ph = _phase(state)
            K_hi, G_hi = hs_upper_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi)
            K_lo, G_lo = hs_lower_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi)
            rho = (1.0 - phi) * cfg.rho_matrix_kg_m3 + phi * ph.rho_pore_kg_m3
            vp_hi, _ = velocities_from_KG_rho(K_hi, G_hi, rho)
            vp_lo, _ = velocities_from_KG_rho(K_lo, G_lo, rho)
            vp_hi /= 1000.0
            vp_lo /= 1000.0
            close = _select_close_to_bounds(y, vp_lo, vp_hi, frac=cfg.close_frac)

            ax.scatter(phi, y, s=26, color=colors[state], alpha=0.9)
            if np.any(close):
                ax.errorbar(
                    phi[close],
                    y[close],
                    yerr=cfg.v_rel_unc * y[close],
                    fmt="none",
                    ecolor=colors[state],
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.95,
                )

            # Bounds curves on grid
            K_hi_g, G_hi_g = hs_upper_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
            K_lo_g, G_lo_g = hs_lower_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
            rho_g = (1.0 - phi_grid) * cfg.rho_matrix_kg_m3 + phi_grid * ph.rho_pore_kg_m3
            vp_hi_g, _ = velocities_from_KG_rho(K_hi_g, G_hi_g, rho_g)
            vp_lo_g, _ = velocities_from_KG_rho(K_lo_g, G_lo_g, rho_g)
            ax.plot(phi_grid, vp_hi_g / 1000.0, color=colors[state], lw=1.8)
            ax.plot(phi_grid, vp_lo_g / 1000.0, color=colors[state], lw=1.8, ls="--")

            # Regression: pick ONE model family shared with Vs (same functional form)
            vs_g = df[df["fluid_state"] == state].dropna(subset=["vs_m_s"]).copy()
            phi_vs = vs_g["phi"].to_numpy(dtype=float)
            ys = pd.to_numeric(vs_g["vs_m_s"], errors="coerce").to_numpy(dtype=float) / 1000.0
            shared = choose_shared_vpvs_model(phi, y, phi_vs, ys, mode=cfg.vpvs_reg_model)

            if shared == "exp":
                A, B = fit_exp(phi, y)
                yfit = A * np.exp(B * phi_grid)
                ax.plot(phi_grid, yfit, color=colors[state], lw=1.2, ls=":")
                r2 = r2_score(y, A * np.exp(B * phi))
                regression_rows_v.append(
                    {
                        "stage": cfg.stage,
                        "fluid_state": state,
                        "property": "vp_km_s",
                        "model": "exp",
                        "A_matrix": A,
                        "B": B,
                        "r2": r2,
                        "hs_matrix": cfg.vp_matrix_m_s / 1000.0,
                    }
                )
            elif shared == "slowness_linear":
                a, b = fit_slowness_linear(phi, y)
                yfit = 1.0 / np.maximum(a * phi_grid + b, 1e-12)
                ax.plot(phi_grid, yfit, color=colors[state], lw=1.2, ls=":")
                r2 = r2_score(1.0 / y, a * phi + b)
                regression_rows_v.append(
                    {
                        "stage": cfg.stage,
                        "fluid_state": state,
                        "property": "vp_km_s",
                        "model": "slowness_linear",
                        "a": a,
                        "b_matrix": b,
                        "r2": r2,
                        "hs_matrix": cfg.vp_matrix_m_s / 1000.0,
                    }
                )
            else:
                a, b = fit_linear(phi, y)
                yfit = a * phi_grid + b
                ax.plot(phi_grid, yfit, color=colors[state], lw=1.2, ls=":")
                r2 = r2_score(y, a * phi + b)
                regression_rows_v.append(
                    {
                        "stage": cfg.stage,
                        "fluid_state": state,
                        "property": "vp_km_s",
                        "model": "linear",
                        "b_matrix": b,
                        "a": a,
                        "r2": r2,
                        "hs_matrix": cfg.vp_matrix_m_s / 1000.0,
                    }
                )

        ax.set_xlabel("Porosity (φ)")
        ax.set_ylabel(r"$V_P$, km/s")
        ax.set_xlim(0.0, 0.25)
        ax.set_ylim(*cfg.vp_ylim)
        ax.grid(True, alpha=0.25)
        ax.set_title("b) $V_P$ vs porosity", pad=8)

        eq_lines = []
        for row in regression_rows_v:
            st_raw = str(row["fluid_state"])
            st = "brine-sat." if st_raw == "wet" else st_raw
            if row["model"] == "exp":
                eq_lines.append(rf"{st}: $V_P={row['A_matrix']:.2f}\,e^{{{row['B']:.3f}\phi}}$, $R^2={row['r2']:.2f}$")
            elif row["model"] == "slowness_linear":
                eq_lines.append(rf"{st}: $1/V_P={row['a']:.3f}\phi+{row['b_matrix']:.3f}$, $R^2={row['r2']:.2f}$")
            else:
                eq_lines.append(rf"{st}: $V_P={row['a']:.2f}\phi+{row['b_matrix']:.2f}$, $R^2={row['r2']:.2f}$")
        ax.text(0.0, -0.36, "\n".join(eq_lines), transform=ax.transAxes, ha="left", va="top", fontsize=10)

        # --- Vs panel ---
        ax = axes[2]
        regression_rows_s: list[dict[str, object]] = []
        for state in ["dry", "wet"]:
            g = df[df["fluid_state"] == state].copy()
            if g.empty:
                continue
            phi = g["phi"].to_numpy(dtype=float)
            y = pd.to_numeric(g["vs_m_s"], errors="coerce").to_numpy(dtype=float) / 1000.0
            m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
            phi, y = phi[m], y[m]
            if phi.size == 0:
                continue

            ph = _phase(state)
            K_hi, G_hi = hs_upper_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi)
            K_lo, G_lo = hs_lower_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi)
            rho = (1.0 - phi) * cfg.rho_matrix_kg_m3 + phi * ph.rho_pore_kg_m3
            _, vs_hi = velocities_from_KG_rho(K_hi, G_hi, rho)
            _, vs_lo = velocities_from_KG_rho(K_lo, G_lo, rho)
            vs_hi /= 1000.0
            vs_lo /= 1000.0
            close = _select_close_to_bounds(y, vs_lo, vs_hi, frac=cfg.close_frac)

            ax.scatter(phi, y, s=26, color=colors[state], alpha=0.9)
            if np.any(close):
                ax.errorbar(
                    phi[close],
                    y[close],
                    yerr=cfg.v_rel_unc * y[close],
                    fmt="none",
                    ecolor=colors[state],
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.95,
                )

            # Bounds curves on grid
            K_hi_g, G_hi_g = hs_upper_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
            K_lo_g, G_lo_g = hs_lower_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
            rho_g = (1.0 - phi_grid) * cfg.rho_matrix_kg_m3 + phi_grid * ph.rho_pore_kg_m3
            _, vs_hi_g = velocities_from_KG_rho(K_hi_g, G_hi_g, rho_g)
            _, vs_lo_g = velocities_from_KG_rho(K_lo_g, G_lo_g, rho_g)
            ax.plot(phi_grid, vs_hi_g / 1000.0, color=colors[state], lw=1.8)
            ax.plot(phi_grid, vs_lo_g / 1000.0, color=colors[state], lw=1.8, ls="--")

            # Use the same shared model family as Vp (chosen on both together)
            vp_g = df[df["fluid_state"] == state].dropna(subset=["vp_m_s"]).copy()
            phi_vp = vp_g["phi"].to_numpy(dtype=float)
            yv = pd.to_numeric(vp_g["vp_m_s"], errors="coerce").to_numpy(dtype=float) / 1000.0
            shared = choose_shared_vpvs_model(phi_vp, yv, phi, y, mode=cfg.vpvs_reg_model)

            if shared == "exp":
                A, B = fit_exp(phi, y)
                yfit = A * np.exp(B * phi_grid)
                ax.plot(phi_grid, yfit, color=colors[state], lw=1.2, ls=":")
                r2 = r2_score(y, A * np.exp(B * phi))
                regression_rows_s.append(
                    {
                        "stage": cfg.stage,
                        "fluid_state": state,
                        "property": "vs_km_s",
                        "model": "exp",
                        "A_matrix": A,
                        "B": B,
                        "r2": r2,
                        "hs_matrix": cfg.vs_matrix_m_s / 1000.0,
                    }
                )
            elif shared == "slowness_linear":
                a, b = fit_slowness_linear(phi, y)
                yfit = 1.0 / np.maximum(a * phi_grid + b, 1e-12)
                ax.plot(phi_grid, yfit, color=colors[state], lw=1.2, ls=":")
                r2 = r2_score(1.0 / y, a * phi + b)
                regression_rows_s.append(
                    {
                        "stage": cfg.stage,
                        "fluid_state": state,
                        "property": "vs_km_s",
                        "model": "slowness_linear",
                        "a": a,
                        "b_matrix": b,
                        "r2": r2,
                        "hs_matrix": cfg.vs_matrix_m_s / 1000.0,
                    }
                )
            else:
                a, b = fit_linear(phi, y)
                yfit = a * phi_grid + b
                ax.plot(phi_grid, yfit, color=colors[state], lw=1.2, ls=":")
                r2 = r2_score(y, a * phi + b)
                regression_rows_s.append(
                    {
                        "stage": cfg.stage,
                        "fluid_state": state,
                        "property": "vs_km_s",
                        "model": "linear",
                        "b_matrix": b,
                        "a": a,
                        "r2": r2,
                        "hs_matrix": cfg.vs_matrix_m_s / 1000.0,
                    }
                )

        ax.set_xlabel("Porosity (φ)")
        ax.set_ylabel(r"$V_S$, km/s")
        ax.set_xlim(0.0, 0.25)
        ax.set_ylim(*cfg.vs_ylim)
        ax.grid(True, alpha=0.25)
        ax.set_title("c) $V_S$ vs porosity", pad=8)

        # Figure-level title and legend (below)
        fig.suptitle(f"{stage_label}: HS bounds and measurement uncertainty", y=1.02)

        # Legend: points by saturation, and line styles for HS bounds
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", linestyle="none", color=colors["dry"], label=f"{stage_label} (Dried)"),
            Line2D([], [], marker="o", linestyle="none", color=colors["wet"], label=f"{stage_label} (Brine-saturated)"),
            Line2D([], [], color="0.15", lw=1.8, label="HS upper bound"),
            Line2D([], [], color="0.15", lw=1.8, ls="--", label="HS lower bound"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)

        eq_lines = []
        for row in regression_rows_s:
            st_raw = str(row["fluid_state"])
            st = "brine-sat." if st_raw == "wet" else st_raw
            if row["model"] == "exp":
                eq_lines.append(rf"{st}: $V_S={row['A_matrix']:.2f}\,e^{{{row['B']:.3f}\phi}}$, $R^2={row['r2']:.2f}$")
            elif row["model"] == "slowness_linear":
                eq_lines.append(rf"{st}: $1/V_S={row['a']:.3f}\phi+{row['b_matrix']:.3f}$, $R^2={row['r2']:.2f}$")
            else:
                eq_lines.append(rf"{st}: $V_S={row['a']:.2f}\phi+{row['b_matrix']:.2f}$, $R^2={row['r2']:.2f}$")
        ax.text(0.0, -0.36, "\n".join(eq_lines), transform=ax.transAxes, ha="left", va="top", fontsize=10)

        out_png = cfg.out_dir / f"hs_bounds_tc_vp_vs_{cfg.stage}.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # Terminal comparison table: matrix values from regression vs HS-config
    rows: list[dict[str, object]] = []
    for rset in [regression_rows, regression_rows_v, regression_rows_s]:
        for row in rset:
            model = str(row["model"])
            if model == "exp":
                reg_matrix = float(row["A_matrix"])
            elif model == "slowness_linear":
                b = float(row["b_matrix"])
                reg_matrix = float("nan") if (not np.isfinite(b) or b == 0.0) else 1.0 / b
            else:
                reg_matrix = float(row["b_matrix"])
            hs_matrix = float(row["hs_matrix"])
            rows.append(
                {
                    "stage": row["stage"],
                    "fluid_state": row["fluid_state"],
                    "property": row["property"],
                    "reg_model": model,
                    "reg_matrix": reg_matrix,
                    "hs_matrix": hs_matrix,
                    "diff": reg_matrix - hs_matrix,
                    "diff_%": 100.0 * (reg_matrix - hs_matrix) / hs_matrix if hs_matrix != 0 else float("nan"),
                    "r2": float(row["r2"]),
                }
            )
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        print("\nMatrix-property comparison (regression extrapolation vs HS matrix input):")
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))  # noqa: T201
        # summary stats per property
        grp = tbl.groupby(["stage", "property"], dropna=False)
        summary = grp.agg(
            n=("diff", "count"),
            mean_abs_diff=("diff", lambda x: float(np.mean(np.abs(x)))),
            mean_abs_diff_pct=("diff_%", lambda x: float(np.mean(np.abs(x)))),
        )
        print("\nSummary (mean absolute diff):")
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))  # noqa: T201


def main() -> None:
    ap = argparse.ArgumentParser(description="TC/Vp/Vs vs porosity with HS bounds + selective uncertainty bars.")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/porosity_relationships/hs_bounds"))
    ap.add_argument("--stage", choices=["before", "after", "both"], default="both")
    ap.add_argument("--close-frac", type=float, default=0.12, help="Show error bars if within this fraction of HS range.")
    ap.add_argument("--tc-rel-unc", type=float, default=0.025, help="Relative uncertainty for thermal conductivity.")
    ap.add_argument("--v-rel-unc", type=float, default=0.05, help="Relative uncertainty for velocities.")

    # Phase properties (simple defaults; can be overridden)
    ap.add_argument("--lambda-matrix", type=float, default=2.86, help="Matrix thermal conductivity (W/m/K).")
    ap.add_argument("--lambda-air", type=float, default=0.026, help="Pore thermal conductivity for dry (air).")
    ap.add_argument("--lambda-brine", type=float, default=0.60, help="Pore thermal conductivity for wet (brine).")
    ap.add_argument("--K-air-gpa", type=float, default=1e-4, help="Air bulk modulus (GPa).")
    ap.add_argument("--K-brine-gpa", type=float, default=2.2, help="Brine bulk modulus (GPa).")
    ap.add_argument("--rho-air", type=float, default=1.2, help="Air density (kg/m^3).")
    ap.add_argument("--rho-brine", type=float, default=1030.0, help="Brine density (kg/m^3).")

    # Matrix elastic reference (one set for all samples)
    ap.add_argument("--vp-matrix", type=float, default=6800.0, help="Matrix Vp (m/s) reference.")
    ap.add_argument("--vs-matrix", type=float, default=3800.0, help="Matrix Vs (m/s) reference.")
    ap.add_argument("--rho-matrix", type=float, default=2725.0, help="Matrix density (kg/m^3) reference.")

    ap.add_argument(
        "--vpvs-reg-model",
        choices=["auto", "linear", "exp", "slowness_linear"],
        default="linear",
        help="Force a shared regression family for both Vp and Vs (same functional form).",
    )

    ap.add_argument("--tc-ylim", type=float, nargs=2, default=(1.2, 3.0))
    ap.add_argument("--vp-ylim", type=float, nargs=2, default=(3.0, 7.0))
    ap.add_argument("--vs-ylim", type=float, nargs=2, default=(1.5, 4.0))

    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    cfg_base = dict(
        data_xlsx=Path(args.data_xlsx),
        out_dir=Path(args.out_dir),
        close_frac=float(args.close_frac),
        tc_rel_unc=float(args.tc_rel_unc),
        v_rel_unc=float(args.v_rel_unc),
        lambda_matrix_w_mk=float(args.lambda_matrix),
        vp_matrix_m_s=float(args.vp_matrix),
        vs_matrix_m_s=float(args.vs_matrix),
        rho_matrix_kg_m3=float(args.rho_matrix),
        phase_dry=PhaseProps(
            lambda_pore_w_mk=float(args.lambda_air),
            K_pore_gpa=float(args.K_air_gpa),
            rho_pore_kg_m3=float(args.rho_air),
        ),
        phase_wet=PhaseProps(
            lambda_pore_w_mk=float(args.lambda_brine),
            K_pore_gpa=float(args.K_brine_gpa),
            rho_pore_kg_m3=float(args.rho_brine),
        ),
        dpi=int(args.dpi),
        tc_ylim=(float(args.tc_ylim[0]), float(args.tc_ylim[1])),
        vp_ylim=(float(args.vp_ylim[0]), float(args.vp_ylim[1])),
        vs_ylim=(float(args.vs_ylim[0]), float(args.vs_ylim[1])),
        vpvs_reg_model=str(args.vpvs_reg_model),
    )

    stages = ["before", "after"] if args.stage == "both" else [str(args.stage)]
    for st in stages:
        cfg = Config(stage=st, **cfg_base)
        _plot_stage(cfg)


if __name__ == "__main__":
    main()
