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
    _select_close_to_bounds,
    fit_exp,
    r2_score,
    hs_lower_conductivity,
    hs_upper_conductivity,
)


@dataclass(frozen=True)
class Config:
    data_xlsx: Path
    out_dir: Path
    close_frac: float
    r_rel_unc: float
    brine_res_before_ohm_m: float
    brine_res_after_ohm_m: float
    sigma_matrix_s_m: float
    dpi: int
    ylim: tuple[float, float]
    auto_expand_ylim: bool = True


def _sigma_from_resistivity(R_ohm_m: float) -> float:
    R = float(R_ohm_m)
    return 1.0 / max(R, 1e-12)


def _hs_bounds_resistivity(
    *,
    sigma_m: float,
    sigma_f: float,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute HS bounds for *resistivity* R (Ohm·m) by applying HS to conductivity sigma
    then inverting. Returns (R_lo, R_hi) where lo<=hi pointwise.
    """
    phi = np.asarray(phi, dtype=float)
    # HS bounds in conductivity:
    b1 = hs_upper_conductivity(float(sigma_m), float(sigma_f), phi)
    b2 = hs_lower_conductivity(float(sigma_m), float(sigma_f), phi)
    sigma_lo = np.minimum(b1, b2)
    sigma_hi = np.maximum(b1, b2)

    # Invert to resistivity: R = 1/sigma (monotone decreasing)
    R_lo = 1.0 / np.maximum(sigma_hi, 1e-300)
    R_hi = 1.0 / np.maximum(sigma_lo, 1e-300)
    return R_lo, R_hi


def _expand_ylim(ymin: float, ymax: float, data: np.ndarray, *, frac: float = 0.10) -> tuple[float, float]:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return ymin, ymax
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    # multiplicative padding for log axis
    new_min = ymin if dmin >= ymin else dmin / (1.0 + frac)
    new_max = ymax if dmax <= ymax else dmax * (1.0 + frac)
    return float(new_min), float(new_max)


def _default_sigma_matrix_from_previous_run() -> float | None:
    """
    If a previous resistivity-including scan exists, use its probability-weighted median
    as a reasonable default for sigma_M.
    """
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
    v = row["p50"].iloc[0]
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if np.isfinite(fv) and fv > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description="HS bounds + regression for electrical resistivity vs porosity (before/after).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/porosity_relationships/hs_bounds"))
    ap.add_argument("--close-frac", type=float, default=0.12)
    ap.add_argument("--r-rel-unc", type=float, default=0.05, help="Relative uncertainty for resistivity R (Ohm·m).")

    ap.add_argument("--brine-res-before", type=float, default=0.32, help="Brine resistivity R_f (Ohm·m) for 'before'.")
    ap.add_argument("--brine-res-after", type=float, default=0.26, help="Brine resistivity R_f (Ohm·m) for 'after'.")
    ap.add_argument(
        "--sigma-matrix",
        type=float,
        default=None,
        help="Matrix electrical conductivity sigma_M (S/m). If omitted, tries to read from the latest scan results; else falls back to 1e-3.",
    )

    ap.add_argument("--ylim", type=float, nargs=2, default=(1.0, 200.0), help="Y-limits for resistivity (log scale).")
    ap.add_argument("--no-auto-expand-ylim", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    sigma_m = args.sigma_matrix
    if sigma_m is None:
        sigma_m = _default_sigma_matrix_from_previous_run()
    if sigma_m is None:
        sigma_m = 1e-3
    sigma_m = float(sigma_m)
    if not (sigma_m > 0):
        raise ValueError(f"--sigma-matrix must be >0. Got {sigma_m}")

    cfg = Config(
        data_xlsx=Path(args.data_xlsx),
        out_dir=Path(args.out_dir),
        close_frac=float(args.close_frac),
        r_rel_unc=float(args.r_rel_unc),
        brine_res_before_ohm_m=float(args.brine_res_before),
        brine_res_after_ohm_m=float(args.brine_res_after),
        sigma_matrix_s_m=float(sigma_m),
        dpi=int(args.dpi),
        ylim=(float(args.ylim[0]), float(args.ylim[1])),
        auto_expand_ylim=not bool(args.no_auto_expand_ylim),
    )

    _configure_matplotlib_env(cfg.out_dir)
    df = pd.read_excel(cfg.data_xlsx, sheet_name="measurements_long")
    df["stage"] = df["stage"].astype(str).str.strip()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip()
    df["phi_pct"] = pd.to_numeric(df["phi_pct"], errors="coerce")
    df["phi"] = df["phi_pct"] / 100.0

    # Resistivity is only available for brine-sat. in this dataset.
    df = df[(df["fluid_state"] == "wet") & df["phi"].notna() & df["resistivity_ohm_m"].notna()].copy()
    if df.empty:
        raise RuntimeError("No resistivity data found in measurements_long for fluid_state='wet'.")

    df["R"] = pd.to_numeric(df["resistivity_ohm_m"], errors="coerce")
    df = df[np.isfinite(df["R"]) & (df["R"] > 0)].copy()
    if df.empty:
        raise RuntimeError("Resistivity column exists but contains no positive numeric values.")

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

    color = "#1f77b4"  # brine-sat points

    phi_grid = np.linspace(0.0, max(0.25, float(np.nanmax(df["phi"])) * 1.05), 250)

    rows: list[dict[str, object]] = []
    with plt.rc_context(rc):
        fig, axs = plt.subplots(2, 1, figsize=(7.4, 7.8), sharex=True)
        fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.16, hspace=0.50)

        for r, stage in enumerate(["before", "after"]):
            ax = axs[r]
            g = df[df["stage"] == stage].copy()
            if g.empty:
                ax.set_visible(False)
                continue

            phi = g["phi"].to_numpy(dtype=float)
            R = g["R"].to_numpy(dtype=float)

            sigma_f = _sigma_from_resistivity(cfg.brine_res_before_ohm_m if stage == "before" else cfg.brine_res_after_ohm_m)
            R_lo, R_hi = _hs_bounds_resistivity(sigma_m=cfg.sigma_matrix_s_m, sigma_f=sigma_f, phi=phi)
            close = _select_close_to_bounds(R, R_lo, R_hi, frac=cfg.close_frac)

            ax.scatter(phi, R, s=26, color=color, alpha=0.9)
            if np.any(close):
                ax.errorbar(
                    phi[close],
                    R[close],
                    yerr=cfg.r_rel_unc * R[close],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.95,
                )

            # Bounds on grid
            R_lo_g, R_hi_g = _hs_bounds_resistivity(sigma_m=cfg.sigma_matrix_s_m, sigma_f=sigma_f, phi=phi_grid)
            ax.plot(phi_grid, R_hi_g, color="0.15", lw=1.8)  # upper in R
            ax.plot(phi_grid, R_lo_g, color="0.15", lw=1.8, ls="--")

            # Regression: R = A exp(B phi) (straight line on semilog-y)
            A, B = fit_exp(phi, R)
            R_fit = A * np.exp(B * phi)
            r2 = r2_score(R, R_fit)
            ax.plot(phi_grid, A * np.exp(B * phi_grid), color="0.15", lw=1.2, ls=":")

            rows.append(
                {
                    "stage": "Before MPCT" if stage == "before" else "After MPCT",
                    "property": "R_ohm_m",
                    "reg_model": "exp",
                    "A": float(A),
                    "B": float(B),
                    "r2": float(r2),
                    "sigma_M_S_m": float(cfg.sigma_matrix_s_m),
                    "R_f_ohm_m": float(cfg.brine_res_before_ohm_m if stage == "before" else cfg.brine_res_after_ohm_m),
                }
            )

            ax.set_yscale("log")
            ax.set_xlim(0.0, 0.25)
            ax.set_ylim(*cfg.ylim)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_ylabel(r"$R$, $\Omega\cdot$m")

            eq = rf"$R = {A:.2f}\,\exp({B:.2f}\phi)$, $R^2={r2:.2f}$"
            ax.text(0.5, -0.22, eq, transform=ax.transAxes, ha="center", va="top", fontsize=12, clip_on=False)

        # Row labels
        for r, label in enumerate(["Before MPCT", "After MPCT"]):
            if not axs[r].get_visible():
                continue
            box = axs[r].get_position()
            y = 0.5 * (box.y0 + box.y1)
            x = box.x0 - 0.09
            fig.text(x, y, label, ha="center", va="center", fontsize=14, rotation=90)

        # Title only once
        axs[0].set_title("Electrical resistivity (brine-saturated)", pad=8)
        axs[1].set_xlabel("Porosity (φ)")

        if cfg.auto_expand_ylim:
            R_all = df["R"].to_numpy(dtype=float)
            for ax in axs:
                if not ax.get_visible():
                    continue
                lo, hi = ax.get_ylim()
                ax.set_ylim(*_expand_ylim(lo, hi, R_all))

        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", linestyle="none", color=color, label="Brine-saturated"),
            Line2D([], [], color="0.15", lw=1.8, label="HS upper bound"),
            Line2D([], [], color="0.15", lw=1.8, ls="--", label="HS lower bound"),
            Line2D([], [], color="0.15", lw=1.2, ls=":", label="Exponential regression"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False)

        out_png = cfg.out_dir / "hs_bounds_resistivity_before_after.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")  # noqa: T201
    print(f"Saved: {out_pdf}")  # noqa: T201
    if rows:
        tbl = pd.DataFrame(rows)
        print("\nResistivity regression summary:")  # noqa: T201
        def _fmt(v: float) -> str:
            v = float(v)
            if not np.isfinite(v):
                return "nan"
            if 0 < abs(v) < 1e-3:
                return f"{v:.2e}"
            return f"{v:.4f}"

        print(tbl.to_string(index=False, float_format=_fmt))  # noqa: T201


if __name__ == "__main__":
    main()
