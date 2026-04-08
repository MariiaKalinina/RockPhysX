from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_new.hs_bounds_tc_vp_vs_porosity import (  # noqa: E402
    PhaseProps,
    _configure_matplotlib_env,
    _matrix_moduli_from_velocities,
    _select_close_to_bounds,
    fit_exp,
    fit_linear,
    hs_lower_conductivity,
    hs_lower_elastic,
    hs_upper_conductivity,
    hs_upper_elastic,
    r2_score,
    velocities_from_KG_rho,
)


@dataclass(frozen=True)
class Config:
    data_xlsx: Path
    out_dir: Path
    close_frac: float
    tc_rel_unc: float
    v_rel_unc: float
    r_rel_unc: float
    lambda_matrix_w_mk: float
    vp_matrix_m_s: float
    vs_matrix_m_s: float
    rho_matrix_kg_m3: float
    sigma_matrix_s_m: float
    brine_res_before_ohm_m: float
    brine_res_after_ohm_m: float
    phase_dry: PhaseProps
    phase_wet: PhaseProps
    dpi: int
    tc_ylim: tuple[float, float]
    vp_ylim: tuple[float, float]
    vs_ylim: tuple[float, float]
    r_ylim: tuple[float, float]
    auto_expand_ylim: bool = True


def _expand_ylim_linear(ymin: float, ymax: float, data: np.ndarray, *, frac: float = 0.04) -> tuple[float, float]:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return ymin, ymax
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    span = max(ymax - ymin, 1e-9)
    new_min = ymin if dmin >= ymin else (dmin - frac * span)
    new_max = ymax if dmax <= ymax else (dmax + frac * span)
    return float(new_min), float(new_max)


def _expand_ylim_log(ymin: float, ymax: float, data: np.ndarray, *, frac: float = 0.10) -> tuple[float, float]:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return ymin, ymax
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    new_min = ymin if dmin >= ymin else dmin / (1.0 + frac)
    new_max = ymax if dmax <= ymax else dmax * (1.0 + frac)
    return float(new_min), float(new_max)


def _phase(cfg: Config, state: str) -> PhaseProps:
    return cfg.phase_dry if state == "dry" else cfg.phase_wet


def _sigma_from_resistivity(R_ohm_m: float) -> float:
    return 1.0 / max(float(R_ohm_m), 1e-12)


def _hs_bounds_resistivity(*, sigma_m: float, sigma_f: float, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute HS bounds for resistivity R by applying HS to conductivity sigma and inverting.
    Returns (R_lo, R_hi), lo<=hi.
    """
    phi = np.asarray(phi, dtype=float)
    b1 = hs_upper_conductivity(float(sigma_m), float(sigma_f), phi)
    b2 = hs_lower_conductivity(float(sigma_m), float(sigma_f), phi)
    sigma_lo = np.minimum(b1, b2)
    sigma_hi = np.maximum(b1, b2)
    R_lo = 1.0 / np.maximum(sigma_hi, 1e-300)
    R_hi = 1.0 / np.maximum(sigma_lo, 1e-300)
    return R_lo, R_hi


def _default_sigma_matrix_from_scan() -> float | None:
    p = Path("test_new/hs_matrix_feasible_set_with_resistivity/probability_weighted_95ci.csv")
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    row = df[df["param"] == "sigmaM_S_m"]
    if row.empty:
        return None
    try:
        v = float(row["p50"].iloc[0])
    except Exception:
        return None
    return v if np.isfinite(v) and v > 0 else None


def _plot_row(
    axes: np.ndarray,
    df_stage: pd.DataFrame,
    *,
    cfg: Config,
    stage_label: str,
    phi_grid: np.ndarray,
    Km_pa: float,
    Gm_pa: float,
    colors: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # (1) TC
    ax = axes[0]
    for state in ["dry", "wet"]:
        g = df_stage[df_stage["fluid_state"] == state].copy()
        if g.empty:
            continue
        phi = g["phi"].to_numpy(dtype=float)
        y = pd.to_numeric(g["tc_w_mk"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
        phi, y = phi[m], y[m]
        if phi.size == 0:
            continue

        ph = _phase(cfg, state)
        y_hi = hs_upper_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi)
        y_lo = hs_lower_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi)
        close = _select_close_to_bounds(y, y_lo, y_hi, frac=cfg.close_frac)

        ax.scatter(phi, y, s=26, color=colors[state], alpha=0.9)
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

        hi = hs_upper_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi_grid)
        lo = hs_lower_conductivity(cfg.lambda_matrix_w_mk, ph.lambda_pore_w_mk, phi_grid)
        ax.plot(phi_grid, hi, color=colors[state], lw=1.8)
        ax.plot(phi_grid, lo, color=colors[state], lw=1.8, ls="--")

        a, b = fit_linear(phi, y)
        ax.plot(phi_grid, a * phi_grid + b, color=colors[state], lw=1.2, ls=":")
        r2 = r2_score(y, a * phi + b)
        rows.append(
            {
                "stage": stage_label,
                "property": "tc_w_mk",
                "eq": rf"{'brine-sat.' if state=='wet' else state}: $\lambda={a:.2f}\phi+{b:.2f}$, $R^2={r2:.2f}$",
            }
        )

    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(*cfg.tc_ylim)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(r"$\lambda$, W/(m$\cdot$K)")

    # (2) Vp
    ax = axes[1]
    for state in ["dry", "wet"]:
        g = df_stage[df_stage["fluid_state"] == state].copy()
        if g.empty:
            continue
        phi = g["phi"].to_numpy(dtype=float)
        y = pd.to_numeric(g["vp_m_s"], errors="coerce").to_numpy(dtype=float) / 1000.0
        m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
        phi, y = phi[m], y[m]
        if phi.size == 0:
            continue

        ph = _phase(cfg, state)
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

        K_hi_g, G_hi_g = hs_upper_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
        K_lo_g, G_lo_g = hs_lower_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
        rho_g = (1.0 - phi_grid) * cfg.rho_matrix_kg_m3 + phi_grid * ph.rho_pore_kg_m3
        vp_hi_g, _ = velocities_from_KG_rho(K_hi_g, G_hi_g, rho_g)
        vp_lo_g, _ = velocities_from_KG_rho(K_lo_g, G_lo_g, rho_g)
        ax.plot(phi_grid, vp_hi_g / 1000.0, color=colors[state], lw=1.8)
        ax.plot(phi_grid, vp_lo_g / 1000.0, color=colors[state], lw=1.8, ls="--")

        a, b = fit_linear(phi, y)
        ax.plot(phi_grid, a * phi_grid + b, color=colors[state], lw=1.2, ls=":")
        r2 = r2_score(y, a * phi + b)
        rows.append(
            {
                "stage": stage_label,
                "property": "vp_km_s",
                "eq": rf"{'brine-sat.' if state=='wet' else state}: $V_P={a:.2f}\phi+{b:.2f}$, $R^2={r2:.2f}$",
            }
        )

    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(*cfg.vp_ylim)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(r"$V_P$, km/s")

    # (3) Vs
    ax = axes[2]
    for state in ["dry", "wet"]:
        g = df_stage[df_stage["fluid_state"] == state].copy()
        if g.empty:
            continue
        phi = g["phi"].to_numpy(dtype=float)
        y = pd.to_numeric(g["vs_m_s"], errors="coerce").to_numpy(dtype=float) / 1000.0
        m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
        phi, y = phi[m], y[m]
        if phi.size == 0:
            continue

        ph = _phase(cfg, state)
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

        K_hi_g, G_hi_g = hs_upper_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
        K_lo_g, G_lo_g = hs_lower_elastic(Km_pa, Gm_pa, ph.K_pore_gpa * 1e9, 0.0, phi_grid)
        rho_g = (1.0 - phi_grid) * cfg.rho_matrix_kg_m3 + phi_grid * ph.rho_pore_kg_m3
        _, vs_hi_g = velocities_from_KG_rho(K_hi_g, G_hi_g, rho_g)
        _, vs_lo_g = velocities_from_KG_rho(K_lo_g, G_lo_g, rho_g)
        ax.plot(phi_grid, vs_hi_g / 1000.0, color=colors[state], lw=1.8)
        ax.plot(phi_grid, vs_lo_g / 1000.0, color=colors[state], lw=1.8, ls="--")

        a, b = fit_linear(phi, y)
        ax.plot(phi_grid, a * phi_grid + b, color=colors[state], lw=1.2, ls=":")
        r2 = r2_score(y, a * phi + b)
        rows.append(
            {
                "stage": stage_label,
                "property": "vs_km_s",
                "eq": rf"{'brine-sat.' if state=='wet' else state}: $V_S={a:.2f}\phi+{b:.2f}$, $R^2={r2:.2f}$",
            }
        )

    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(*cfg.vs_ylim)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(r"$V_S$, km/s")

    # (4) Resistivity (brine-sat. only)
    ax = axes[3]
    g = df_stage[(df_stage["fluid_state"] == "wet") & df_stage["resistivity_ohm_m"].notna()].copy()
    if not g.empty:
        phi = g["phi"].to_numpy(dtype=float)
        R = pd.to_numeric(g["resistivity_ohm_m"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(phi) & np.isfinite(R) & (R > 0)
        phi, R = phi[m], R[m]
        if phi.size:
            sigma_f = _sigma_from_resistivity(cfg.brine_res_before_ohm_m if stage_label == "Before MPCT" else cfg.brine_res_after_ohm_m)
            R_lo, R_hi = _hs_bounds_resistivity(sigma_m=cfg.sigma_matrix_s_m, sigma_f=sigma_f, phi=phi)
            close = _select_close_to_bounds(R, R_lo, R_hi, frac=cfg.close_frac)

            ax.scatter(phi, R, s=26, color=colors["wet"], alpha=0.9)
            if np.any(close):
                ax.errorbar(
                    phi[close],
                    R[close],
                    yerr=cfg.r_rel_unc * R[close],
                    fmt="none",
                    ecolor=colors["wet"],
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.95,
                )

            R_lo_g, R_hi_g = _hs_bounds_resistivity(sigma_m=cfg.sigma_matrix_s_m, sigma_f=sigma_f, phi=phi_grid)
            # Use the same color-coding as other panels (brine-sat. = blue).
            ax.plot(phi_grid, R_hi_g, color=colors["wet"], lw=1.8)
            ax.plot(phi_grid, R_lo_g, color=colors["wet"], lw=1.8, ls="--")

            A, B = fit_exp(phi, R)
            ax.plot(phi_grid, A * np.exp(B * phi_grid), color=colors["wet"], lw=1.2, ls=":")
            r2 = r2_score(R, A * np.exp(B * phi))
            rows.append(
                {
                    "stage": stage_label,
                    "property": "R_ohm_m",
                    "eq": rf"brine-sat.: $R={A:.2f}e^{{{B:.2f}\phi}}$, $R^2={r2:.2f}$",
                }
            )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(*cfg.r_ylim)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylabel(r"$R$, $\Omega\cdot$m")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Before/after HS bounds + regression, including resistivity (same style as tc/vp/vs figure).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/porosity_relationships/hs_bounds"))
    ap.add_argument("--close-frac", type=float, default=0.12)
    ap.add_argument("--tc-rel-unc", type=float, default=0.025)
    ap.add_argument("--v-rel-unc", type=float, default=0.05)
    ap.add_argument("--r-rel-unc", type=float, default=0.05)

    ap.add_argument("--lambda-matrix", type=float, default=2.85)
    ap.add_argument("--lambda-air", type=float, default=0.025)
    ap.add_argument("--lambda-brine", type=float, default=0.60)

    ap.add_argument("--K-air-gpa", type=float, default=1e-4)
    ap.add_argument("--K-brine-gpa", type=float, default=2.2)
    ap.add_argument("--rho-air", type=float, default=1.2)
    ap.add_argument("--rho-brine", type=float, default=1030.0)

    ap.add_argument("--vp-matrix", type=float, default=6280.0)
    ap.add_argument("--vs-matrix", type=float, default=3520.0)
    ap.add_argument("--rho-matrix", type=float, default=2725.0)

    ap.add_argument("--brine-res-before", type=float, default=0.32)
    ap.add_argument("--brine-res-after", type=float, default=0.26)
    ap.add_argument(
        "--sigma-matrix",
        type=float,
        default=None,
        help="Matrix electrical conductivity sigma_M (S/m). If omitted, uses median from previous scan; else 1e-3.",
    )

    ap.add_argument("--tc-ylim", type=float, nargs=2, default=(1.2, 3.0))
    ap.add_argument("--vp-ylim", type=float, nargs=2, default=(3.0, 7.0))
    ap.add_argument("--vs-ylim", type=float, nargs=2, default=(1.5, 4.0))
    # Resistivity can span orders of magnitude; default range shows HS upper bound when sigma_M is very small.
    ap.add_argument("--r-ylim", type=float, nargs=2, default=(0.1, 1e6))
    ap.add_argument("--no-auto-expand-ylim", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    sigma_m = args.sigma_matrix
    if sigma_m is None:
        sigma_m = _default_sigma_matrix_from_scan()
    if sigma_m is None:
        sigma_m = 1e-3
    sigma_m = float(sigma_m)

    cfg = Config(
        data_xlsx=Path(args.data_xlsx),
        out_dir=Path(args.out_dir),
        close_frac=float(args.close_frac),
        tc_rel_unc=float(args.tc_rel_unc),
        v_rel_unc=float(args.v_rel_unc),
        r_rel_unc=float(args.r_rel_unc),
        lambda_matrix_w_mk=float(args.lambda_matrix),
        vp_matrix_m_s=float(args.vp_matrix),
        vs_matrix_m_s=float(args.vs_matrix),
        rho_matrix_kg_m3=float(args.rho_matrix),
        sigma_matrix_s_m=sigma_m,
        brine_res_before_ohm_m=float(args.brine_res_before),
        brine_res_after_ohm_m=float(args.brine_res_after),
        phase_dry=PhaseProps(lambda_pore_w_mk=float(args.lambda_air), K_pore_gpa=float(args.K_air_gpa), rho_pore_kg_m3=float(args.rho_air)),
        phase_wet=PhaseProps(lambda_pore_w_mk=float(args.lambda_brine), K_pore_gpa=float(args.K_brine_gpa), rho_pore_kg_m3=float(args.rho_brine)),
        dpi=int(args.dpi),
        tc_ylim=(float(args.tc_ylim[0]), float(args.tc_ylim[1])),
        vp_ylim=(float(args.vp_ylim[0]), float(args.vp_ylim[1])),
        vs_ylim=(float(args.vs_ylim[0]), float(args.vs_ylim[1])),
        r_ylim=(float(args.r_ylim[0]), float(args.r_ylim[1])),
        auto_expand_ylim=not bool(args.no_auto_expand_ylim),
    )

    _configure_matplotlib_env(cfg.out_dir)
    df = pd.read_excel(cfg.data_xlsx, sheet_name="measurements_long")
    df["stage"] = df["stage"].astype(str).str.strip()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip()
    df["phi_pct"] = pd.to_numeric(df["phi_pct"], errors="coerce")
    df["phi"] = df["phi_pct"] / 100.0
    df = df[df["phi"].notna()].copy()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
    colors = {"dry": "#d62728", "wet": "#1f77b4"}

    phi_grid = np.linspace(0.0, max(0.25, float(np.nanmax(df["phi"])) * 1.05), 250)
    Km_pa, Gm_pa = _matrix_moduli_from_velocities(cfg.vp_matrix_m_s, cfg.vs_matrix_m_s, cfg.rho_matrix_kg_m3)

    with plt.rc_context(rc):
        fig, axs = plt.subplots(2, 4, figsize=(17.3, 8.2), sharex=True)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.20, hspace=0.48, wspace=0.28)

        rows = []
        rows.extend(
            _plot_row(
                axs[0, :],
                df[df["stage"] == "before"].copy(),
                cfg=cfg,
                stage_label="Before MPCT",
                phi_grid=phi_grid,
                Km_pa=Km_pa,
                Gm_pa=Gm_pa,
                colors=colors,
            )
        )
        rows.extend(
            _plot_row(
                axs[1, :],
                df[df["stage"] == "after"].copy(),
                cfg=cfg,
                stage_label="After MPCT",
                phi_grid=phi_grid,
                Km_pa=Km_pa,
                Gm_pa=Gm_pa,
                colors=colors,
            )
        )

        if cfg.auto_expand_ylim:
            tc_vals = pd.to_numeric(df["tc_w_mk"], errors="coerce").to_numpy(dtype=float)
            vp_vals = (pd.to_numeric(df["vp_m_s"], errors="coerce") / 1000.0).to_numpy(dtype=float)
            vs_vals = (pd.to_numeric(df["vs_m_s"], errors="coerce") / 1000.0).to_numpy(dtype=float)
            R_vals = pd.to_numeric(df["resistivity_ohm_m"], errors="coerce").to_numpy(dtype=float)
            for r in range(2):
                lo, hi = axs[r, 0].get_ylim()
                axs[r, 0].set_ylim(*_expand_ylim_linear(lo, hi, tc_vals))
                lo, hi = axs[r, 1].get_ylim()
                axs[r, 1].set_ylim(*_expand_ylim_linear(lo, hi, vp_vals))
                lo, hi = axs[r, 2].get_ylim()
                axs[r, 2].set_ylim(*_expand_ylim_linear(lo, hi, vs_vals))
                lo, hi = axs[r, 3].get_ylim()
                axs[r, 3].set_ylim(*_expand_ylim_log(lo, hi, R_vals))

        # Put equations just below each subplot
        def _eq_under(ax, lines: list[str], *, y_axes: float) -> None:
            if not lines:
                return
            ax.text(0.5, y_axes, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=11, clip_on=False)

        for r in range(2):
            stage = "Before MPCT" if r == 0 else "After MPCT"
            for c, prop in enumerate(["tc_w_mk", "vp_km_s", "vs_km_s", "R_ohm_m"]):
                ax = axs[r, c]
                eq_lines = [row["eq"] for row in rows if row["stage"] == stage and row["property"] == prop]
                _eq_under(ax, eq_lines, y_axes=-0.18 if r == 0 else -0.26)

        # Row labels
        for r, label in enumerate(["Before MPCT", "After MPCT"]):
            box = axs[r, 0].get_position()
            y = 0.5 * (box.y0 + box.y1)
            x = box.x0 - 0.060
            fig.text(x, y, label, ha="center", va="center", fontsize=14, rotation=90)

        # Column titles only for first row
        axs[0, 0].set_title(r"$\lambda$", pad=8)
        axs[0, 1].set_title(r"$V_P$", pad=8)
        axs[0, 2].set_title(r"$V_S$", pad=8)
        axs[0, 3].set_title(r"$R$", pad=8)
        for ax in axs[1, :]:
            ax.set_title("")

        # X labels only bottom row
        for ax in axs[1, :]:
            ax.set_xlabel("Porosity (φ)")

        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", linestyle="none", color=colors["dry"], label="Dried"),
            Line2D([], [], marker="o", linestyle="none", color=colors["wet"], label="Brine-saturated"),
            Line2D([], [], color="0.15", lw=1.8, label="HS upper bound"),
            Line2D([], [], color="0.15", lw=1.8, ls="--", label="HS lower bound"),
            Line2D([], [], color="0.15", lw=1.2, ls=":", label="Regression"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5, frameon=False)

        out_png = cfg.out_dir / "hs_bounds_tc_vp_vs_R_before_after.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")  # noqa: T201
    print(f"Saved: {out_pdf}")  # noqa: T201


if __name__ == "__main__":
    main()
