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
    _configure_matplotlib_env,
    fit_linear,
    r2_score,
)


@dataclass(frozen=True)
class Config:
    data_xlsx: Path
    out_dir: Path
    dpi: int
    phi_xlim: tuple[float, float]
    logk_ylim: tuple[float, float]
    rho_ylim: tuple[float, float]


def _expand_ylim(ymin: float, ymax: float, data: np.ndarray, *, frac: float = 0.06) -> tuple[float, float]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Vertical figure: porosity–permeability (before only) and porosity–bulk density (before vs after).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/porosity_relationships/basic_props"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--phi-xlim", type=float, nargs=2, default=(0.0, 0.25))
    ap.add_argument("--logk-ylim", type=float, nargs=2, default=(-1.0, 2.2))
    ap.add_argument("--rho-ylim", type=float, nargs=2, default=(2.0, 2.75))
    ap.add_argument(
        "--density-fluid-state",
        choices=["dry", "wet", "all"],
        default="dry",
        help="Which measurements to use for density. Default 'dry' avoids mixing saturation effects.",
    )
    args = ap.parse_args()

    cfg = Config(
        data_xlsx=Path(args.data_xlsx),
        out_dir=Path(args.out_dir),
        dpi=int(args.dpi),
        phi_xlim=(float(args.phi_xlim[0]), float(args.phi_xlim[1])),
        logk_ylim=(float(args.logk_ylim[0]), float(args.logk_ylim[1])),
        rho_ylim=(float(args.rho_ylim[0]), float(args.rho_ylim[1])),
    )

    _configure_matplotlib_env(cfg.out_dir)

    df = pd.read_excel(cfg.data_xlsx, sheet_name="measurements_long")
    df["stage"] = df["stage"].astype(str).str.strip()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip()
    df["phi_pct"] = pd.to_numeric(df["phi_pct"], errors="coerce")
    df["phi"] = df["phi_pct"] / 100.0
    df = df[df["phi"].notna()].copy()

    df["bulk_density_g_cm3"] = pd.to_numeric(df["bulk_density_g_cm3"], errors="coerce")
    df["permeability_md_modeled"] = pd.to_numeric(df["permeability_md_modeled"], errors="coerce")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
    stage_colors = {"before": "#d62728", "after": "#1f77b4"}  # red / blue

    phi_grid = np.linspace(cfg.phi_xlim[0], cfg.phi_xlim[1], 250)

    rows: list[dict[str, object]] = []
    with plt.rc_context(rc):
        # Two panels side-by-side + dedicated rows for equations and legend (prevents overlap and wasted space).
        fig = plt.figure(figsize=(10.2, 6.3))
        gs = fig.add_gridspec(
            nrows=3,
            ncols=2,
            # Dedicated rows for: plots / xlabel+equations / legend (compact, no overlaps).
            height_ratios=[1.0, 0.26, 0.10],
            left=0.10,
            right=0.99,
            top=0.98,
            bottom=0.06,
            wspace=0.28,
            # Slightly larger gap so equation row never touches plots.
            hspace=0.12,
        )
        ax_k = fig.add_subplot(gs[0, 0])
        ax_rho = fig.add_subplot(gs[0, 1])
        ax_eq_k = fig.add_subplot(gs[1, 0])
        ax_eq_rho = fig.add_subplot(gs[1, 1])
        ax_leg = fig.add_subplot(gs[2, :])
        axs = [ax_k, ax_rho]

        # (1) Permeability (before only)
        ax = ax_k
        gk = df[(df["stage"] == "before") & df["permeability_md_modeled"].notna()].copy()
        if not gk.empty:
            if "lab_sample_id" in gk.columns:
                gk = gk.drop_duplicates(subset=["lab_sample_id"])
            phi = gk["phi"].to_numpy(dtype=float)
            k = gk["permeability_md_modeled"].to_numpy(dtype=float)
            m = np.isfinite(phi) & np.isfinite(k) & (k > 0)
            phi, k = phi[m], k[m]
            y = np.log10(k)
            ax.scatter(phi, y, s=28, color="0.15", alpha=0.9)
            a, b = fit_linear(phi, y)
            ax.plot(phi_grid, a * phi_grid + b, color="0.15", lw=1.4, ls=":")
            r2 = r2_score(y, a * phi + b)
            rows.append({"panel": "k", "eq": rf"$\log_{{10}}(k)={a:.2f}\phi+{b:.2f}$, $R^2={r2:.2f}$"})
        ax.set_xlim(*cfg.phi_xlim)
        ax.set_ylim(*cfg.logk_ylim)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(r"$\log_{10}(k)$, mD")
        ax.set_xlabel("Porosity (φ)", labelpad=6)

        # (2) Bulk density (before=red, after=blue)
        ax = ax_rho
        gr = df[df["bulk_density_g_cm3"].notna()].copy()
        if args.density_fluid_state != "all":
            gr = gr[gr["fluid_state"] == args.density_fluid_state].copy()
        for stage in ["before", "after"]:
            sub = gr[gr["stage"] == stage].copy()
            if sub.empty:
                continue
            phi = sub["phi"].to_numpy(dtype=float)
            rho = sub["bulk_density_g_cm3"].to_numpy(dtype=float)
            m = np.isfinite(phi) & np.isfinite(rho)
            phi, rho = phi[m], rho[m]
            if phi.size == 0:
                continue
            ax.scatter(phi, rho, s=28, color=stage_colors[stage], alpha=0.9)
            a, b = fit_linear(phi, rho)
            ax.plot(phi_grid, a * phi_grid + b, color=stage_colors[stage], lw=1.4, ls=":")
            r2 = r2_score(rho, a * phi + b)
            rows.append({"panel": f"rho_{stage}", "eq": rf"{stage}: $\rho={a:.3f}\phi+{b:.3f}$, $R^2={r2:.2f}$"})

        ax.set_xlim(*cfg.phi_xlim)
        ax.set_ylim(*cfg.rho_ylim)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(r"$\rho$, g/cm$^3$")
        ax.set_xlabel("Porosity (φ)", labelpad=6)

        # Expand ylims if needed
        logk_vals = np.log10(df["permeability_md_modeled"].to_numpy(dtype=float))
        if np.isfinite(logk_vals).any():
            lo, hi = ax_k.get_ylim()
            ax_k.set_ylim(*_expand_ylim(lo, hi, logk_vals))
        rho_vals = df["bulk_density_g_cm3"].to_numpy(dtype=float)
        if np.isfinite(rho_vals).any():
            lo, hi = ax_rho.get_ylim()
            ax_rho.set_ylim(*_expand_ylim(lo, hi, rho_vals))

        # Equations row (dedicated axes; no overlap with legend/plots)
        for ax_eq in [ax_eq_k, ax_eq_rho]:
            ax_eq.set_axis_off()
        eq_k = [r["eq"] for r in rows if r["panel"] == "k"]
        ax_eq_k.text(
            0.5,
            0.32,
            eq_k[0] if eq_k else "",
            ha="center",
            va="center",
            fontsize=13,
            clip_on=True,
        )
        eq_rho = [r["eq"] for r in rows if str(r["panel"]).startswith("rho_")]
        ax_eq_rho.text(
            0.5,
            0.32,
            "\n".join(eq_rho),
            ha="center",
            va="center",
            fontsize=13,
            clip_on=True,
        )

        # Legend below
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", linestyle="none", color="0.15", label="Permeability (before)"),
            Line2D([], [], marker="o", linestyle="none", color=stage_colors["before"], label="Density (before)"),
            Line2D([], [], marker="o", linestyle="none", color=stage_colors["after"], label="Density (after)"),
            Line2D([], [], color="0.15", lw=1.4, ls=":", label="Regression"),
        ]
        ax_leg.set_axis_off()
        ax_leg.legend(handles=handles, loc="center", ncol=4, frameon=False)

        out_png = cfg.out_dir / "porosity_perm_density_horizontal_tall.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")  # noqa: T201
    print(f"Saved: {out_pdf}")  # noqa: T201


if __name__ == "__main__":
    main()
