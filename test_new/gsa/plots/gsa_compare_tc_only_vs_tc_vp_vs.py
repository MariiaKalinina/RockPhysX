from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _load_results(results_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1m2 = pd.read_excel(results_xlsx, sheet_name="M1_M2")
    mis = pd.read_excel(results_xlsx, sheet_name="misfit_detail")
    m1m2["lab_sample_id"] = pd.to_numeric(m1m2["lab_sample_id"], errors="coerce")
    mis["lab_sample_id"] = pd.to_numeric(mis["lab_sample_id"], errors="coerce")
    m1m2["mode"] = m1m2["mode"].astype(str)
    mis["mode"] = mis["mode"].astype(str)
    mis["model"] = mis["model"].astype(str)
    mis["stage"] = mis["stage"].astype(str)
    mis["fluid_state"] = mis["fluid_state"].astype(str)
    mis["category"] = mis["category"].astype(str)
    return m1m2, mis


@dataclass(frozen=True)
class HeatingZone:
    name: str
    distance_mm: float
    temperature_c: float


def _parse_heating_zones(experiment_metadata: pd.DataFrame) -> list[HeatingZone]:
    df = experiment_metadata[experiment_metadata["section"].astype(str) == "heating_zone"].copy()
    if df.empty:
        return []

    # The table is recorded as repeated blocks: zone_name + (temperature, duration, distance)
    # with the "note" column carrying the zone label for the numeric rows.
    zones: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        param = str(r.get("parameter"))
        if param == "zone_name":
            continue
        note = r.get("note")
        zone = str(note).strip() if pd.notna(note) else "unknown"
        if zone.lower() in {"nan", ""}:
            zone = "unknown"
        val = r.get("value")
        if pd.isna(val):
            continue
        zones.setdefault(zone, {})
        zones[zone][param] = float(val)

    out: list[HeatingZone] = []
    for zone, d in zones.items():
        if "distance" not in d or "temperature" not in d:
            continue
        out.append(HeatingZone(name=zone, distance_mm=float(d["distance"]), temperature_c=float(d["temperature"])))
    out.sort(key=lambda z: z.distance_mm)
    return out


def _assign_zone(tg_position_mm: float, zones: list[HeatingZone]) -> str:
    if not zones or not np.isfinite(tg_position_mm):
        return "unknown"
    pos = float(tg_position_mm)
    dists = np.array([z.distance_mm for z in zones], dtype=float)
    # Bin edges are midpoints between zone distances.
    edges = 0.5 * (dists[:-1] + dists[1:])
    idx = int(np.searchsorted(edges, pos, side="right"))
    idx = min(max(idx, 0), len(zones) - 1)
    return str(zones[idx].name)


def _build_branch_table(m1m2: pd.DataFrame) -> pd.DataFrame:
    df = m1m2.copy()
    needed = [
        "lab_sample_id",
        "mode",
        "phi_before_pct",
        "phi_after_pct",
        "perm_before_md_modeled",
        "m1_alpha_before",
        "m1_alpha_after",
        "m1_delta_z",
        "m2_alpha_before_p50",
        "m2_alpha_after_p50",
        "m2_delta_z_p50",
        "m1_J_total",
        "m2_J_before",
        "m2_J_after",
    ]
    keep = [c for c in needed if c in df.columns]
    df = df[keep].copy()
    wide = df.pivot(index="lab_sample_id", columns="mode")
    wide.columns = [f"{a}__{b}" for a, b in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    # Convenience differences: tc_vp_vs - tc_only in log10(alpha)
    def safe_log10(x: pd.Series) -> pd.Series:
        return np.log10(pd.to_numeric(x, errors="coerce").astype(float).clip(lower=1e-12))

    for model, a_before, a_after, dz in (
        ("m1", "m1_alpha_before", "m1_alpha_after", "m1_delta_z"),
        ("m2", "m2_alpha_before_p50", "m2_alpha_after_p50", "m2_delta_z_p50"),
    ):
        b = f"{a_before}__tc_only"
        a = f"{a_before}__tc_vp_vs"
        if b in wide.columns and a in wide.columns:
            wide[f"{model}_dlog10_alpha_before_(tc_vp_vs-tc_only)"] = safe_log10(wide[a]) - safe_log10(wide[b])

        b = f"{a_after}__tc_only"
        a = f"{a_after}__tc_vp_vs"
        if b in wide.columns and a in wide.columns:
            wide[f"{model}_dlog10_alpha_after_(tc_vp_vs-tc_only)"] = safe_log10(wide[a]) - safe_log10(wide[b])

        b = f"{dz}__tc_only"
        a = f"{dz}__tc_vp_vs"
        if b in wide.columns and a in wide.columns:
            wide[f"{model}_d_delta_z_(tc_vp_vs-tc_only)"] = pd.to_numeric(wide[a], errors="coerce") - pd.to_numeric(
                wide[b], errors="coerce"
            )

    return wide


def _plot_branch_alpha_comparison(out_png: Path, table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

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

    # Pick columns
    cols = {
        "M1_before_tc_only": "m1_alpha_before__tc_only",
        "M1_before_tc_vp_vs": "m1_alpha_before__tc_vp_vs",
        "M1_after_tc_only": "m1_alpha_after__tc_only",
        "M1_after_tc_vp_vs": "m1_alpha_after__tc_vp_vs",
        "M2_before_tc_only": "m2_alpha_before_p50__tc_only",
        "M2_before_tc_vp_vs": "m2_alpha_before_p50__tc_vp_vs",
        "M2_after_tc_only": "m2_alpha_after_p50__tc_only",
        "M2_after_tc_vp_vs": "m2_alpha_after_p50__tc_vp_vs",
    }
    if not all(c in table.columns for c in cols.values()):
        return

    df = table.copy()
    for c in cols.values():
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # common limits
    a_all = np.concatenate([df[c].to_numpy(float) for c in cols.values()])
    a_all = a_all[np.isfinite(a_all)]
    lo = max(float(np.nanmin(a_all)) / 1.8, 1e-6)
    hi = float(np.nanmax(a_all)) * 1.8

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2), constrained_layout=True, sharex=True, sharey=True)

        def panel(ax, xcol: str, ycol: str, title: str) -> None:
            x = df[xcol].to_numpy(float)
            y = df[ycol].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            ax.scatter(x, y, s=45, color="C0", alpha=0.85)
            ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, ls="--", alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(title)

        panel(axes[0, 0], cols["M1_before_tc_only"], cols["M1_before_tc_vp_vs"], "M1: before (TC-only vs TC+Vp+Vs)")
        panel(axes[0, 1], cols["M1_after_tc_only"], cols["M1_after_tc_vp_vs"], "M1: after (TC-only vs TC+Vp+Vs)")
        panel(axes[1, 0], cols["M2_before_tc_only"], cols["M2_before_tc_vp_vs"], "M2 (p50): before")
        panel(axes[1, 1], cols["M2_after_tc_only"], cols["M2_after_tc_vp_vs"], "M2 (p50): after")

        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\alpha^*_{\mathrm{TC+V_P+V_S}}$")
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\alpha^*_{\mathrm{TC-only}}$")

        fig.suptitle("Branch incompatibility diagnostic: best-fit aspect ratio")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_measured_vs_predicted_tc(out_png: Path, misfit_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

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

    df = misfit_df.copy()
    df = df[df["category"].isin({"TC_dry", "TC_wet"})].copy()
    if df.empty:
        return

    df["obs"] = pd.to_numeric(df["obs"], errors="coerce").astype(float)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce").astype(float)
    df["misfit_pct"] = pd.to_numeric(df["misfit_pct"], errors="coerce").astype(float)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(14.6, 11.6), constrained_layout=True, sharex=False, sharey=False)
        axes = np.asarray(axes)

        panels = [
            ("tc_only", "M1", "TC_dry", axes[0, 0]),
            ("tc_only", "M1", "TC_wet", axes[0, 1]),
            ("tc_only", "M2", "TC_dry", axes[1, 0]),
            ("tc_only", "M2", "TC_wet", axes[1, 1]),
        ]

        def add_bands(ax, x0, x1) -> None:
            xs = np.linspace(x0, x1, 200)
            # background bands (reference-like)
            for pct, alpha, color in (
                (0.20, 0.10, "#eeeeee"),
                (0.10, 0.14, "#f6dada"),
                (0.05, 0.18, "#d8f0d8"),
            ):
                ax.fill_between(xs, (1 - pct) * xs, (1 + pct) * xs, color=color, alpha=alpha, linewidth=0)

            # diagonal guides (match the feel of the reference)
            ax.plot(xs, xs, color="0.1", lw=1.6, ls="--", alpha=0.75)
            for pct, ls, lw, c in (
                (0.05, ":", 1.2, "0.25"),
                (0.10, "--", 1.1, "0.35"),
                (0.20, "-.", 1.0, "0.45"),
            ):
                ax.plot(xs, (1 + pct) * xs, color=c, lw=lw, ls=ls, alpha=0.8)
                ax.plot(xs, (1 - pct) * xs, color=c, lw=lw, ls=ls, alpha=0.8)

        for mode, model, cat, ax in panels:
            sub = df[(df["mode"] == mode) & (df["model"] == model) & (df["category"] == cat)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue
            x = sub["obs"].to_numpy(float)
            y = sub["pred"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            if x.size == 0:
                ax.set_axis_off()
                continue

            x0 = float(np.nanmin(x))
            x1 = float(np.nanmax(x))
            pad = 0.03 * (x1 - x0 if x1 > x0 else 1.0)
            x0 -= pad
            x1 += pad

            add_bands(ax, x0, x1)

            # match example: after = blue, before = orange
            for stage, color in (("after", "C0"), ("before", "C1")):
                ss = sub[sub["stage"] == stage]
                ax.scatter(ss["obs"], ss["pred"], s=55, alpha=0.85, label=stage, color=color, edgecolor="none")

            title_model = "M1_single_effective_AR" if model == "M1" else "M2_beta_AR_distribution"
            title_fluid = "dry" if cat == "TC_dry" else "wet"
            ax.set_title(f"{title_model} — {title_fluid}")
            ax.set_xlabel("Measured")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=True, facecolor="white", edgecolor="0.8", framealpha=1.0, loc="upper left")

        fig.suptitle("Measured vs predicted TC (split by fluid)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_measured_vs_predicted_property_tc_vp_vs(
    out_png: Path,
    misfit_df: pd.DataFrame,
    *,
    property_label: str,
    cat_dry: str,
    cat_wet: str,
) -> None:
    import matplotlib.pyplot as plt

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

    df = misfit_df.copy()
    df = df[(df["mode"] == "tc_vp_vs") & (df["category"].isin({cat_dry, cat_wet}))].copy()
    if df.empty:
        return

    df["obs"] = pd.to_numeric(df["obs"], errors="coerce").astype(float)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce").astype(float)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(14.6, 11.6), constrained_layout=True, sharex=False, sharey=False)
        axes = np.asarray(axes)

        panels = [
            ("M1", cat_dry, axes[0, 0]),
            ("M1", cat_wet, axes[0, 1]),
            ("M2", cat_dry, axes[1, 0]),
            ("M2", cat_wet, axes[1, 1]),
        ]

        def add_bands(ax, x0, x1) -> None:
            xs = np.linspace(x0, x1, 200)
            for pct, alpha, color in (
                (0.20, 0.10, "#eeeeee"),
                (0.10, 0.14, "#f6dada"),
                (0.05, 0.18, "#d8f0d8"),
            ):
                ax.fill_between(xs, (1 - pct) * xs, (1 + pct) * xs, color=color, alpha=alpha, linewidth=0)

            ax.plot(xs, xs, color="0.1", lw=1.6, ls="--", alpha=0.75)
            for pct, ls, lw, c in (
                (0.05, ":", 1.2, "0.25"),
                (0.10, "--", 1.1, "0.35"),
                (0.20, "-.", 1.0, "0.45"),
            ):
                ax.plot(xs, (1 + pct) * xs, color=c, lw=lw, ls=ls, alpha=0.8)
                ax.plot(xs, (1 - pct) * xs, color=c, lw=lw, ls=ls, alpha=0.8)

        for model, cat, ax in panels:
            sub = df[(df["model"] == model) & (df["category"] == cat)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            x = sub["obs"].to_numpy(float)
            y = sub["pred"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            if x.size == 0:
                ax.set_axis_off()
                continue

            x0 = float(np.nanmin(x))
            x1 = float(np.nanmax(x))
            pad = 0.03 * (x1 - x0 if x1 > x0 else 1.0)
            x0 -= pad
            x1 += pad

            add_bands(ax, x0, x1)

            # after = blue, before = orange
            for stage, color in (("after", "C0"), ("before", "C1")):
                ss = sub[sub["stage"] == stage]
                ax.scatter(ss["obs"], ss["pred"], s=55, alpha=0.85, label=stage, color=color, edgecolor="none")

            title_model = "M1_single_effective_AR" if model == "M1" else "M2_beta_AR_distribution"
            title_fluid = "dry" if cat.endswith("_dry") else "wet"
            ax.set_title(f"{title_model} — {title_fluid}")
            ax.set_xlabel("Measured")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=True, facecolor="white", edgecolor="0.8", framealpha=1.0, loc="upper left")

        fig.suptitle(f"Measured vs predicted {property_label} (TC+Vp+Vs inversion; split by fluid)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_alpha_change_by_heating_zone(
    out_png: Path,
    m1m2: pd.DataFrame,
    sample_stage: pd.DataFrame,
    zones: list[HeatingZone],
    *,
    mode: str,
) -> None:
    import matplotlib.pyplot as plt

    if not zones:
        return

    df = m1m2[m1m2["mode"].astype(str) == str(mode)].copy()
    if df.empty:
        return

    stage_before = sample_stage[sample_stage["stage"].astype(str) == "before"][["lab_sample_id", "tg_position_mm"]].copy()
    stage_before["lab_sample_id"] = pd.to_numeric(stage_before["lab_sample_id"], errors="coerce")
    stage_before["zone"] = stage_before["tg_position_mm"].apply(lambda x: _assign_zone(float(x) if pd.notna(x) else float("nan"), zones))

    df = df.merge(stage_before[["lab_sample_id", "zone"]], on="lab_sample_id", how="left")
    df["zone"] = df["zone"].fillna("unknown").astype(str)

    # percent change
    def pct_change(a_after: pd.Series, a_before: pd.Series) -> np.ndarray:
        aa = pd.to_numeric(a_after, errors="coerce").astype(float)
        bb = pd.to_numeric(a_before, errors="coerce").astype(float)
        return (aa / bb - 1.0).to_numpy(float) * 100.0

    df["m1_pct"] = pct_change(df["m1_alpha_after"], df["m1_alpha_before"])
    df["m2_pct"] = pct_change(df["m2_alpha_after_p50"], df["m2_alpha_before_p50"])

    zone_order = [z.name for z in zones]
    temps = [z.temperature_c for z in zones]

    data_m1 = [df[df["zone"] == zn]["m1_pct"].to_numpy(float) for zn in zone_order]
    data_m2 = [df[df["zone"] == zn]["m2_pct"].to_numpy(float) for zn in zone_order]

    # If too sparse, still plot; matplotlib handles empty arrays.
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(16.0, 7.6), constrained_layout=True)
        ax.set_facecolor("#f6e6d8")  # warm pastel background like the reference

        # background qualitative bands (pastel, same wording as reference)
        band_moderate = ax.axhspan(-20, -5, color="#f8edc6", alpha=0.55, zorder=0)
        band_strong = ax.axhspan(-45, -20, color="#f5c6cb", alpha=0.35, zorder=0)
        band_none = ax.axhspan(-5, 5, color="#d4edda", alpha=0.35, zorder=0)

        positions = np.arange(1, len(zone_order) + 1)
        width = 0.34

        bp1 = ax.boxplot(
            data_m1,
            positions=positions - width / 2,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ff8c00", "linewidth": 1.6},
            whiskerprops={"color": "0.15", "linewidth": 1.2},
            capprops={"color": "0.15", "linewidth": 1.2},
            boxprops={"edgecolor": "0.15", "linewidth": 1.2},
        )
        for b in bp1["boxes"]:
            b.set_facecolor("#9ad0f5")  # M1 blue
            b.set_alpha(0.90)

        bp2 = ax.boxplot(
            data_m2,
            positions=positions + width / 2,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ff8c00", "linewidth": 1.6},
            whiskerprops={"color": "0.15", "linewidth": 1.2},
            capprops={"color": "0.15", "linewidth": 1.2},
            boxprops={"edgecolor": "0.15", "linewidth": 1.2},
        )
        for b in bp2["boxes"]:
            b.set_facecolor("#f4a6a6")  # M2 red/pink
            b.set_alpha(0.90)

        # Translate zone labels to English if needed
        zone_labels: list[str] = []
        for zn in zone_order:
            s = str(zn)
            if "Зона" in s:
                s = s.replace("Зона", "Zone")
            zone_labels.append(s)
        ax.set_xticks(positions)
        ax.set_xticklabels(zone_labels)
        ax.set_ylabel(r"Aspect-ratio change  $100\cdot(\alpha_{after}/\alpha_{before}-1)$  (%)")
        ax.grid(True, axis="y", color="0.75", alpha=0.35, linewidth=1.0)
        ax.set_axisbelow(True)

        # y limits and ticks: match the reference style
        ax.set_ylim(-45.0, -9.0)
        ax.set_yticks(np.arange(-45, -9, 5))

        # Temperature profile on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            positions,
            temps,
            color="0.25",
            marker="o",
            markersize=8,
            lw=2.5,
            label="Zone temperature profile",
            zorder=5,
        )
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_ylim(300.0, 485.0)
        ax2.set_yticks([300, 325, 350, 375, 400, 425, 450, 475])
        ax2.grid(False)

        # Legend (custom) — match reference ordering and wording
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        handles = [
            Patch(facecolor="#9ad0f5", edgecolor="0.15", label="M1: % change"),
            Patch(facecolor="#f4a6a6", edgecolor="0.15", label="M2: % change"),
            Line2D([0], [0], color="0.25", marker="o", lw=2.5, label="Zone temperature profile"),
            Patch(facecolor="#f8edc6", edgecolor="none", label="−20% to −5%: moderate crack-like shift"),
            Patch(facecolor="#f5c6cb", edgecolor="none", label="< −20%: strong crack-like shift"),
            Patch(facecolor="#d4edda", edgecolor="none", label="−5% to +5%: ~no change"),
        ]
        ax.legend(
            handles=handles,
            frameon=True,
            facecolor="white",
            edgecolor="0.75",
            framealpha=1.0,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.26),
        )

        fig.suptitle("Aspect-ratio change by heating zone (with temperature profile)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _beta_pdf_alpha(alpha: np.ndarray, *, alpha_min: float, alpha_max: float, m: float, kappa: float) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    lo = float(np.log10(alpha_min))
    hi = float(np.log10(alpha_max))
    u = (np.log10(alpha) - lo) / (hi - lo)
    eps = 1e-8
    u = np.clip(u, eps, 1.0 - eps)

    a = max(float(m) * float(kappa), 1e-8)
    b = max((1.0 - float(m)) * float(kappa), 1e-8)

    import math

    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    logpdf_u = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u) - logB
    logpdf_u = logpdf_u - float(np.max(logpdf_u))
    pdf_u = np.exp(logpdf_u)

    jac = 1.0 / ((hi - lo) * alpha * np.log(10.0))
    pdf_a = pdf_u * jac

    area = float(np.trapz(pdf_a, alpha))
    if np.isfinite(area) and area > 0:
        pdf_a = pdf_a / area
    return pdf_a


def _plot_m2_beta_fit_before_after(
    out_png: Path,
    m1m2: pd.DataFrame,
    *,
    mode: str,
    alpha_min: float = 1e-4,
    alpha_max: float = 1.0,
    sample_ids: list[float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    df = m1m2[m1m2["mode"].astype(str) == str(mode)].copy()
    if df.empty:
        return

    if sample_ids is None:
        sample_ids = [12.0, 20.0, 24.2, 18.2]

    df = df[df["lab_sample_id"].isin(sample_ids)].copy()
    if df.empty:
        return

    df["__order"] = df["lab_sample_id"].apply(lambda x: sample_ids.index(float(x)) if float(x) in sample_ids else 999)
    df = df.sort_values("__order")

    alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 500)

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

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.2), constrained_layout=True, sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)

        for ax in axes:
            ax.axvspan(1e-4, 1e-2, color="#f8d7da", alpha=0.35)  # Microcracks
            ax.axvspan(1e-2, 1e-1, color="#fff2cc", alpha=0.45)  # Crack-like pores
            ax.axvspan(1e-1, 1e0, color="#d8f0d8", alpha=0.35)  # Interparticle pores
            ax.grid(True, which="both", alpha=0.25)

        for ax, (_, r) in zip(axes, df.iterrows(), strict=False):
            sid = float(r["lab_sample_id"])
            m_b = float(r["m2_m_before"])
            k_b = float(r["m2_kappa_before"])
            m_a = float(r["m2_m_after"])
            k_a = float(r["m2_kappa_after"])

            pdf_b = _beta_pdf_alpha(alpha, alpha_min=alpha_min, alpha_max=alpha_max, m=m_b, kappa=k_b)
            pdf_a = _beta_pdf_alpha(alpha, alpha_min=alpha_min, alpha_max=alpha_max, m=m_a, kappa=k_a)

            # Normalize to peak=1 for visual comparison
            if float(np.nanmax(pdf_b)) > 0:
                pdf_b = pdf_b / float(np.nanmax(pdf_b))
            if float(np.nanmax(pdf_a)) > 0:
                pdf_a = pdf_a / float(np.nanmax(pdf_a))

            ax.fill_between(alpha, 0, pdf_b, color="#4aa3ff", alpha=0.22)
            ax.fill_between(alpha, 0, pdf_a, color="#ff6b6b", alpha=0.22)
            ax.plot(alpha, pdf_b, color="#0066ff", lw=2.0)
            ax.plot(alpha, pdf_a, color="#b00000", lw=2.0)

            lo = np.log10(alpha_min)
            hi = np.log10(alpha_max)
            mean_before = 10 ** (lo + m_b * (hi - lo))
            mean_after = 10 ** (lo + m_a * (hi - lo))
            ax.axvline(mean_before, color="#0066ff", ls="--", lw=1.6, alpha=0.9)
            ax.axvline(mean_after, color="#b00000", ls="--", lw=1.6, alpha=0.9)

            ax.set_title(f"Sample {sid:g}")
            ax.set_xscale("log")
            ax.set_xlim(alpha_min, alpha_max)
            ax.set_ylim(0.0, 1.02)

        for ax in axes[2:]:
            ax.set_xlabel(r"Aspect ratio ($\alpha$) — Log scale")
        for ax in axes[0::2]:
            ax.set_ylabel("Normalized Probability Density")

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_handles = [
            Patch(facecolor="#4aa3ff", alpha=0.22, label="Before: Dominant interparticle pores"),
            Patch(facecolor="#ff6b6b", alpha=0.22, label="After: Increased microcracks"),
            Line2D([0], [0], color="#0066ff", ls="--", lw=1.6, label="Mean before"),
            Line2D([0], [0], color="#b00000", ls="--", lw=1.6, label="Mean after"),
            Patch(facecolor="#f8d7da", alpha=0.35, label="Microcracks"),
            Patch(facecolor="#fff2cc", alpha=0.45, label="Crack-like pores"),
            Patch(facecolor="#d8f0d8", alpha=0.35, label="Interparticle pores"),
        ]
        fig.legend(handles=legend_handles, loc="center right", frameon=True, facecolor="white", edgecolor="0.8")
        fig.suptitle("Aspect ratio (M2 beta fit): before vs after")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare TC-only vs TC+Vp+Vs branches and build diagnostics.")
    ap.add_argument("--results-xlsx", type=Path, default=Path("test_new/gsa/plots/gsa_tc_elastic_m1_m2_results.xlsx"))
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet-sample-stage", type=str, default="sample_stage")
    ap.add_argument("--sheet-experiment-metadata", type=str, default="experiment_metadata")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    m1m2, mis = _load_results(Path(args.results_xlsx))
    stage = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet_sample_stage))
    stage["lab_sample_id"] = pd.to_numeric(stage["lab_sample_id"], errors="coerce")
    meta = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet_experiment_metadata))

    zones = _parse_heating_zones(meta)

    table = _build_branch_table(m1m2)
    out_table = out_dir / "gsa_branch_comparison_table.xlsx"
    table.to_excel(out_table, index=False)
    print(f"Saved: {out_table}")

    _plot_branch_alpha_comparison(out_dir / "gsa_branch_alpha_comparison.png", table)
    print(f"Saved: {out_dir / 'gsa_branch_alpha_comparison.png'}")

    _plot_measured_vs_predicted_tc(out_dir / "gsa_measured_vs_predicted_tc.png", mis)
    print(f"Saved: {out_dir / 'gsa_measured_vs_predicted_tc.png'}")

    # temperature / heating-zone link: start with tc_only (more stable)
    _plot_alpha_change_by_heating_zone(out_dir / "gsa_alpha_change_by_heating_zone_tc_only.png", m1m2, stage, zones, mode="tc_only")
    print(f"Saved: {out_dir / 'gsa_alpha_change_by_heating_zone_tc_only.png'}")

    _plot_m2_beta_fit_before_after(out_dir / "gsa_m2_beta_fit_before_after_tc_only.png", m1m2, mode="tc_only")
    print(f"Saved: {out_dir / 'gsa_m2_beta_fit_before_after_tc_only.png'}")

    _plot_measured_vs_predicted_property_tc_vp_vs(
        out_dir / "gsa_measured_vs_predicted_tc_tc_vp_vs.png",
        mis,
        property_label="TC",
        cat_dry="TC_dry",
        cat_wet="TC_wet",
    )
    print(f"Saved: {out_dir / 'gsa_measured_vs_predicted_tc_tc_vp_vs.png'}")

    _plot_measured_vs_predicted_property_tc_vp_vs(
        out_dir / "gsa_measured_vs_predicted_vp_tc_vp_vs.png",
        mis,
        property_label=r"$V_P$",
        cat_dry="VP_dry",
        cat_wet="VP_wet",
    )
    print(f"Saved: {out_dir / 'gsa_measured_vs_predicted_vp_tc_vp_vs.png'}")

    _plot_measured_vs_predicted_property_tc_vp_vs(
        out_dir / "gsa_measured_vs_predicted_vs_tc_vp_vs.png",
        mis,
        property_label=r"$V_S$",
        cat_dry="VS_dry",
        cat_wet="VS_wet",
    )
    print(f"Saved: {out_dir / 'gsa_measured_vs_predicted_vs_tc_vp_vs.png'}")


if __name__ == "__main__":
    main()
