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
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b), a * x + b


def fit_exp_vs_phi(phi_pct: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Fit y = A * exp(B * phi_pct) by linear regression on ln(y).
    Returns (A, B, y_fit).
    """
    phi = np.asarray(phi_pct, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(phi) & np.isfinite(y) & (y > 0)
    phi = phi[m]
    y = y[m]
    if phi.size < 2:
        return float("nan"), float("nan"), np.full_like(phi_pct, np.nan, dtype=float)
    B, lnA = np.polyfit(phi, np.log(y), deg=1)
    A = float(np.exp(lnA))
    y_fit = A * np.exp(B * np.asarray(phi_pct, dtype=float))
    return A, float(B), y_fit


def fit_logk_vs_lnphi(phi_pct: np.ndarray, k_md: np.ndarray) -> tuple[float, float]:
    """
    Fit log10(k) = a * ln(phi_pct) + b.
    Returns (a, b).
    """
    phi = np.asarray(phi_pct, dtype=float)
    k = np.asarray(k_md, dtype=float)
    m = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    phi = phi[m]
    k = k[m]
    if phi.size < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(np.log(phi), np.log10(k), deg=1)
    return float(a), float(b)


def _fmt_signed(val: float, *, fmt: str = "{:.2f}") -> str:
    if not np.isfinite(val):
        return "nan"
    s = fmt.format(abs(val))
    return f"+{s}" if val >= 0 else f"-{s}"


def _panel_label(ax, s: str) -> None:
    ax.text(
        0.02,
        0.98,
        s,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def _latex_escape(s: object) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def df_to_latex_booktabs(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    out_path: Path,
    float_fmt: str = "{:.4f}",
) -> None:
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
                vals.append(float_fmt.format(float(v)))
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Config:
    data_xlsx: Path
    out_dir: Path
    stage: str = "before"  # before|after|both
    dpi: int = 300


def load_measurements(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.data_xlsx, sheet_name="measurements_long")
    for c in ["stage", "fluid_state", "field", "well"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if cfg.stage in {"before", "after"}:
        df = df[df["stage"] == cfg.stage].copy()
    df["phi_pct"] = pd.to_numeric(df["phi_pct"], errors="coerce")
    df["phi_frac"] = pd.to_numeric(df["phi_frac"], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Thesis-style porosity relationship figure (multi-panel).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/porosity_relationships"))
    ap.add_argument("--stage", choices=["before", "after", "both"], default="before")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    cfg = Config(data_xlsx=args.data_xlsx, out_dir=args.out_dir, stage=args.stage, dpi=int(args.dpi))
    _configure_matplotlib_env(cfg.out_dir)

    df = load_measurements(cfg)

    # Use only rows with porosity
    df = df[df["phi_pct"].notna()].copy()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from e

    # Styling close to your reference figure
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

    fits_rows: list[dict[str, object]] = []

    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 2, figsize=(8.8, 9.8), constrained_layout=True)
        axes = axes.ravel()

        stage_markers = {"before": "o", "after": "^"}

        def _iter_stage_groups(d: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
            if cfg.stage == "both" and "stage" in d.columns:
                out: list[tuple[str, pd.DataFrame]] = []
                for st in ["before", "after"]:
                    g = d[d["stage"] == st]
                    if not g.empty:
                        out.append((st, g))
                return out
            return [(cfg.stage, d)]

        # (a) porosity vs permeability (modeled)
        ax = axes[0]
        sub = df.dropna(subset=["permeability_md_modeled"]).copy()
        for st, g in _iter_stage_groups(sub):
            x = g["phi_pct"].to_numpy(dtype=float)
            y = g["permeability_md_modeled"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            ax.scatter(
                x,
                np.log10(y),
                s=18,
                c="0.15",
                alpha=0.85,
                marker=stage_markers.get(st, "o"),
                edgecolors="none",
            )

        x = sub["phi_pct"].to_numpy(dtype=float)
        y = sub["permeability_md_modeled"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y = x[m], y[m]
        a, b = fit_logk_vs_lnphi(x, y)
        if np.isfinite(a) and np.isfinite(b):
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            yline = a * np.log(xx) + b
            ax.plot(xx, yline, ls="--", lw=1.4, c="0.25")
            yhat = a * np.log(x) + b
            r2 = r2_score(np.log10(y), yhat)
            ax.text(
                0.62,
                0.18,
                rf"$\log_{{10}}(k)={a:.2f}\ln(\phi)\,{_fmt_signed(b, fmt='{:.2f}')}$"
                + "\n"
                + rf"$R^2={r2:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
            )
            fits_rows.append({"panel": "a", "group": "all", "model": "log10(k)=a ln(phi)+b", "a": a, "b": b, "r2": r2, "n": int(x.size)})
        _panel_label(ax, "a)")
        ax.set_xlabel("Porosity (φ), %")
        ax.set_ylabel(r"$\log_{10}(k)$, mD")
        ax.grid(True, alpha=0.25)

        # (b) bulk density vs porosity
        ax = axes[1]
        sub = df.dropna(subset=["bulk_density_g_cm3"]).copy()
        for st, g in _iter_stage_groups(sub):
            x = g["phi_pct"].to_numpy(dtype=float)
            y = g["bulk_density_g_cm3"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            ax.scatter(x, y, s=18, c="0.15", alpha=0.85, marker=stage_markers.get(st, "o"), edgecolors="none")

        x = sub["phi_pct"].to_numpy(dtype=float)
        y = sub["bulk_density_g_cm3"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        a, b, yfit = fit_linear(x, y)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        ax.plot(xx, a * xx + b, ls="--", lw=1.4, c="0.25")
        r2 = r2_score(y, a * x + b)
        ax.text(
            0.56,
            0.70,
            rf"$\rho={a:.4f}\phi\,{_fmt_signed(b, fmt='{:.2f}')}$" + "\n" + rf"$R^2={r2:.2f}$",
            transform=ax.transAxes,
            fontsize=9,
        )
        fits_rows.append({"panel": "b", "group": "all", "model": "rho=a phi + b", "a": a, "b": b, "r2": r2, "n": int(x.size)})
        _panel_label(ax, "b)")
        ax.set_xlabel("Porosity (φ), %")
        ax.set_ylabel(r"Bulk density, g/cm$^3$")
        ax.grid(True, alpha=0.25)

        # Color maps
        sat_colors = {"dry": "#d62728", "wet": "#1f77b4", "oil": "#2ca02c"}
        sat_labels = {"dry": "Dried", "wet": "Brine-saturated", "oil": "Oil-saturated"}

        # (c) thermal conductivity vs porosity, color by fluid_state
        ax = axes[2]
        sub = df.dropna(subset=["tc_w_mk", "fluid_state"]).copy()
        handles_for_legend = []
        for state, g in sub.groupby("fluid_state"):
            x = g["phi_pct"].to_numpy(dtype=float)
            y = g["tc_w_mk"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            label = sat_labels.get(str(state), str(state))
            sc = ax.scatter(x, y, s=18, alpha=0.80, color=sat_colors.get(str(state), "0.5"), label=label)
            handles_for_legend.append(sc)
            A, B, yfit = fit_exp_vs_phi(x, y)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            yline = A * np.exp(B * xx)
            ax.plot(xx, yline, ls="--", lw=1.4, color=sat_colors.get(str(state), "0.5"))
            r2 = r2_score(np.log(y), np.log(A) + B * x) if np.isfinite(A) else float("nan")
            fits_rows.append({"panel": "c", "group": str(state), "model": "y=A exp(B phi)", "A": A, "B": B, "r2": r2, "n": int(x.size)})
        _panel_label(ax, "c)")
        ax.set_xlabel("Porosity (φ), %")
        ax.set_ylabel(r"Thermal conductivity, W/(m·K)")
        ax.grid(True, alpha=0.25)
        # Fit text (one line per saturation), colored
        y0 = 0.08
        for i, row in enumerate([r for r in fits_rows if r.get("panel") == "c"]):
            st = str(row.get("group"))
            A = float(row.get("A", float("nan")))
            B = float(row.get("B", float("nan")))
            r2 = float(row.get("r2", float("nan")))
            if not (np.isfinite(A) and np.isfinite(B)):
                continue
            ax.text(
                0.58,
                y0 + 0.10 * i,
                rf"$\lambda={A:.2f}\,e^{{{B:.3f}\phi}}$" + "\n" + rf"$R^2={r2:.2f}$",
                transform=ax.transAxes,
                fontsize=9,
                color=sat_colors.get(st, "0.25"),
            )

        # (d) electrical resistivity vs porosity (log y)
        ax = axes[3]
        sub = df.dropna(subset=["resistivity_ohm_m"]).copy()
        for st, g in _iter_stage_groups(sub):
            x = g["phi_pct"].to_numpy(dtype=float)
            y = g["resistivity_ohm_m"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            ax.scatter(x, y, s=18, c="#7a1fa2", alpha=0.75, marker=stage_markers.get(st, "o"), edgecolors="none")

        x = sub["phi_pct"].to_numpy(dtype=float)
        y = sub["resistivity_ohm_m"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x, y = x[m], y[m]
        A, B, yfit = fit_exp_vs_phi(x, y)
        if np.isfinite(A) and np.isfinite(B):
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            ax.plot(xx, A * np.exp(B * xx), ls="--", lw=1.4, c="#7a1fa2")
            r2 = r2_score(np.log(y), np.log(A) + B * x)
            ax.text(0.58, 0.70, rf"$R={A:.1f}e^{{{B:.3f}\phi}}$" + "\n" + rf"$R^2={r2:.2f}$", transform=ax.transAxes, fontsize=9, color="#7a1fa2")
            fits_rows.append({"panel": "d", "group": "wet (brine)", "model": "R=A exp(B phi)", "A": A, "B": B, "r2": r2, "n": int(x.size)})
        ax.set_yscale("log")
        _panel_label(ax, "d)")
        ax.set_xlabel("Porosity (φ), %")
        ax.set_ylabel(r"Resistivity, Ohm·m")
        ax.grid(True, alpha=0.25, which="both")
        ax.text(0.02, 0.96, "Brine-saturated only (no salinity column)", transform=ax.transAxes, va="top", fontsize=9, color="0.25")

        # (e) Vp vs porosity, color by fluid_state
        ax = axes[4]
        sub = df.dropna(subset=["vp_m_s", "fluid_state"]).copy()
        for state, g in sub.groupby("fluid_state"):
            x = g["phi_pct"].to_numpy(dtype=float)
            y = (g["vp_m_s"].to_numpy(dtype=float) / 1000.0)
            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            label = sat_labels.get(str(state), str(state))
            ax.scatter(x, y, s=18, alpha=0.80, color=sat_colors.get(str(state), "0.5"), label=label)
            A, B, yfit = fit_exp_vs_phi(x, y)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            ax.plot(xx, A * np.exp(B * xx), ls="--", lw=1.4, color=sat_colors.get(str(state), "0.5"))
            r2 = r2_score(np.log(y), np.log(A) + B * x) if np.isfinite(A) else float("nan")
            fits_rows.append({"panel": "e", "group": str(state), "model": "Vp=A exp(B phi)", "A": A, "B": B, "r2": r2, "n": int(x.size)})
        _panel_label(ax, "e)")
        ax.set_xlabel("Porosity (φ), %")
        ax.set_ylabel(r"$V_P$, km/s")
        ax.grid(True, alpha=0.25)
        y0 = 0.08
        for i, row in enumerate([r for r in fits_rows if r.get("panel") == "e"]):
            st = str(row.get("group"))
            A = float(row.get("A", float("nan")))
            B = float(row.get("B", float("nan")))
            r2 = float(row.get("r2", float("nan")))
            if not (np.isfinite(A) and np.isfinite(B)):
                continue
            ax.text(
                0.58,
                y0 + 0.10 * i,
                rf"$V_P={A:.2f}\,e^{{{B:.3f}\phi}}$" + "\n" + rf"$R^2={r2:.2f}$",
                transform=ax.transAxes,
                fontsize=9,
                color=sat_colors.get(st, "0.25"),
            )

        # (f) Vs vs porosity, color by fluid_state
        ax = axes[5]
        sub = df.dropna(subset=["vs_m_s", "fluid_state"]).copy()
        for state, g in sub.groupby("fluid_state"):
            x = g["phi_pct"].to_numpy(dtype=float)
            y = (g["vs_m_s"].to_numpy(dtype=float) / 1000.0)
            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            label = sat_labels.get(str(state), str(state))
            ax.scatter(x, y, s=18, alpha=0.80, color=sat_colors.get(str(state), "0.5"), label=label)
            A, B, yfit = fit_exp_vs_phi(x, y)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            ax.plot(xx, A * np.exp(B * xx), ls="--", lw=1.4, color=sat_colors.get(str(state), "0.5"))
            r2 = r2_score(np.log(y), np.log(A) + B * x) if np.isfinite(A) else float("nan")
            fits_rows.append({"panel": "f", "group": str(state), "model": "Vs=A exp(B phi)", "A": A, "B": B, "r2": r2, "n": int(x.size)})
        _panel_label(ax, "f)")
        ax.set_xlabel("Porosity (φ), %")
        ax.set_ylabel(r"$V_S$, km/s")
        ax.grid(True, alpha=0.25)
        y0 = 0.08
        for i, row in enumerate([r for r in fits_rows if r.get("panel") == "f"]):
            st = str(row.get("group"))
            A = float(row.get("A", float("nan")))
            B = float(row.get("B", float("nan")))
            r2 = float(row.get("r2", float("nan")))
            if not (np.isfinite(A) and np.isfinite(B)):
                continue
            ax.text(
                0.58,
                y0 + 0.10 * i,
                rf"$V_S={A:.2f}\,e^{{{B:.3f}\phi}}$" + "\n" + rf"$R^2={r2:.2f}$",
                transform=ax.transAxes,
                fontsize=9,
                color=sat_colors.get(st, "0.25"),
            )

        # Consistent x-limits (close to the reference figure)
        phi_max = float(np.nanmax(df["phi_pct"].to_numpy(dtype=float)))
        x_max = float(np.ceil(max(phi_max, 25.0) / 5.0) * 5.0)
        for ax in axes:
            ax.set_xlim(0.0, x_max)

        # Single legend below the whole figure (thesis-friendly)
        sat_handles, sat_texts = [], []
        for key in ["dry", "wet", "oil"]:
            if (df.get("fluid_state") == key).any():
                sat_handles.append(plt.Line2D([], [], linestyle="none", marker="o", color=sat_colors[key], markersize=6))
                sat_texts.append(sat_labels.get(key, key))
        stage_handles, stage_texts = [], []
        if cfg.stage == "both":
            stage_handles = [
                plt.Line2D([], [], linestyle="none", marker=stage_markers["before"], color="0.15", markersize=6),
                plt.Line2D([], [], linestyle="none", marker=stage_markers["after"], color="0.15", markersize=6),
            ]
            stage_texts = ["Before", "After"]
        handles = sat_handles + stage_handles
        labels = sat_texts + stage_texts
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=max(3, len(handles)),
                frameon=False,
            )

        out_png = cfg.out_dir / "figure_porosity_relationships.png"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    fits = pd.DataFrame(fits_rows)
    fits_out = cfg.out_dir / "porosity_relationships_fits.csv"
    fits.to_csv(fits_out, index=False)
    df_to_latex_booktabs(
        fits.fillna(""),
        caption="Regression fits used in the porosity-relationship figure.",
        label="tab:porosity_relationships_fits",
        out_path=cfg.out_dir / "porosity_relationships_fits.tex",
        float_fmt="{:.4f}",
    )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {fits_out}")


if __name__ == "__main__":
    main()
