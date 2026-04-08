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


@dataclass(frozen=True)
class HeatingZone:
    name: str
    distance_mm: float
    temperature_c: float


def _parse_heating_zones(experiment_metadata: pd.DataFrame) -> list[HeatingZone]:
    df = experiment_metadata[experiment_metadata["section"].astype(str) == "heating_zone"].copy()
    if df.empty:
        return []

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
    edges = 0.5 * (dists[:-1] + dists[1:])
    idx = int(np.searchsorted(edges, pos, side="right"))
    idx = min(max(idx, 0), len(zones) - 1)
    return str(zones[idx].name)


def _load_bayes_results(results_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_excel(results_xlsx, sheet_name="summary")
    misfit = pd.read_excel(results_xlsx, sheet_name="misfit_detail")
    summary["lab_sample_id"] = pd.to_numeric(summary["lab_sample_id"], errors="coerce")
    misfit["lab_sample_id"] = pd.to_numeric(misfit["lab_sample_id"], errors="coerce")
    summary["mode"] = summary["mode"].astype(str)
    misfit["mode"] = misfit["mode"].astype(str)
    misfit["stage"] = misfit["stage"].astype(str)
    misfit["fluid_state"] = misfit["fluid_state"].astype(str)
    misfit["property"] = misfit["property"].astype(str)
    return summary, misfit


def _make_wide_misfit_table(misfit: pd.DataFrame) -> pd.DataFrame:
    df = misfit.copy()
    df["key"] = df["property"].str.upper() + "_" + df["fluid_state"].str.lower() + "_" + df["stage"].str.lower()
    df["misfit_pct"] = pd.to_numeric(df["misfit_pct"], errors="coerce").astype(float)
    wide = df.pivot_table(index=["lab_sample_id", "mode"], columns="key", values="misfit_pct", aggfunc="mean")
    wide = wide.reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _plot_misfit_heatmap(out_png: Path, summary: pd.DataFrame, misfit: pd.DataFrame, *, model: str, mode: str) -> None:
    import matplotlib.pyplot as plt

    m = misfit[misfit["mode"] == mode].copy()
    if m.empty:
        return

    wide = _make_wide_misfit_table(m)
    tmp = summary[summary["mode"] == mode].copy()
    tmp = tmp[["lab_sample_id", "phi_before_pct"]].copy()
    tmp["phi_before_pct"] = pd.to_numeric(tmp["phi_before_pct"], errors="coerce").astype(float)
    wide = wide.merge(tmp, on="lab_sample_id", how="left")
    wide = wide.sort_values(["phi_before_pct", "lab_sample_id"]).reset_index(drop=True)

    keys = [c for c in wide.columns if c not in {"lab_sample_id", "mode", "phi_before_pct"}]
    if not keys:
        return
    mat = wide[keys].to_numpy(float)
    vmax = float(np.nanpercentile(mat, 95)) if np.isfinite(mat).any() else 10.0
    vmax = max(vmax, 5.0)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(12.8, max(5.0, 0.35 * len(wide))), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=vmax)
        ax.set_yticks(np.arange(len(wide)))
        ax.set_yticklabels([str(s) for s in wide["lab_sample_id"]])
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_title(f"Bayesian {model}: misfit% (MAP predictive), mode={mode}")
        ax.set_xlabel("Property / fluid / stage")
        ax.set_ylabel("lab_sample_id (sorted by φ)")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("misfit (%)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_pdelta_heatmap(out_png: Path, m1: pd.DataFrame, m2: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # Build a compact wide table: rows = samples; cols = (model,mode)
    def prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
        tmp = df[["lab_sample_id", "mode", "p_delta_z_lt_0", "phi_before_pct"]].copy()
        tmp["lab_sample_id"] = pd.to_numeric(tmp["lab_sample_id"], errors="coerce")
        tmp["p_delta_z_lt_0"] = pd.to_numeric(tmp["p_delta_z_lt_0"], errors="coerce").astype(float)
        tmp["phi_before_pct"] = pd.to_numeric(tmp["phi_before_pct"], errors="coerce").astype(float)
        tmp["col"] = label + "_" + tmp["mode"].astype(str)
        wide = tmp.pivot_table(index=["lab_sample_id", "phi_before_pct"], columns="col", values="p_delta_z_lt_0", aggfunc="mean")
        return wide.reset_index()

    a = prep(m1, "M1")
    b = prep(m2, "M2")
    wide = a.merge(b, on=["lab_sample_id", "phi_before_pct"], how="outer")
    wide = wide.sort_values(["phi_before_pct", "lab_sample_id"]).reset_index(drop=True)
    cols = [c for c in wide.columns if c not in {"lab_sample_id", "phi_before_pct"}]
    if not cols:
        return
    mat = wide[cols].to_numpy(float)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(8.8, max(5.0, 0.35 * len(wide))), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(wide)))
        ax.set_yticklabels([str(s) for s in wide["lab_sample_id"]])
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_title(r"Posterior probability $P(\Delta z < 0)$")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("probability")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_pdelta_threshold_heatmaps(out_dir: Path, m1: pd.DataFrame, m2: pd.DataFrame, *, thresholds: list[float]) -> None:
    import matplotlib.pyplot as plt

    # collect columns present in summaries
    def cols_for(df: pd.DataFrame) -> dict[float, str]:
        out: dict[float, str] = {}
        for d in thresholds:
            key = f"p_delta_z_lt_m{str(d).replace('.', 'p')}"
            if key in df.columns:
                out[float(d)] = key
        return out

    c1 = cols_for(m1)
    c2 = cols_for(m2)
    if not c1 and not c2:
        return

    # merged wide matrix: rows=samples, cols=(model,mode,delta)
    tmp = []
    for label, df, cmap_cols in (("M1", m1, c1), ("M2", m2, c2)):
        if not cmap_cols:
            continue
        for d, col in cmap_cols.items():
            part = df[["lab_sample_id", "mode", "phi_before_pct", col]].copy()
            part["lab_sample_id"] = pd.to_numeric(part["lab_sample_id"], errors="coerce")
            part["phi_before_pct"] = pd.to_numeric(part["phi_before_pct"], errors="coerce").astype(float)
            part[col] = pd.to_numeric(part[col], errors="coerce").astype(float)
            # IMPORTANT: build label elementwise; do not embed a Series in an f-string.
            part["colname"] = (
                str(label)
                + "_"
                + part["mode"].astype(str)
                + f"_dz<-{float(d):g}"
            )
            part = part.rename(columns={col: "p"})
            tmp.append(part[["lab_sample_id", "phi_before_pct", "colname", "p"]])

    if not tmp:
        return
    long = pd.concat(tmp, ignore_index=True)
    wide = long.pivot_table(index=["lab_sample_id", "phi_before_pct"], columns="colname", values="p", aggfunc="mean").reset_index()
    wide = wide.sort_values(["phi_before_pct", "lab_sample_id"]).reset_index(drop=True)
    cols = [c for c in wide.columns if c not in {"lab_sample_id", "phi_before_pct"}]
    if not cols:
        return
    mat = wide[cols].to_numpy(float)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(12.8, max(5.2, 0.38 * len(wide))), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(wide)))
        ax.set_yticklabels([str(s) for s in wide["lab_sample_id"]])
        ax.set_xticks(np.arange(len(cols)))
        # Use compact, readable tick labels
        pretty = []
        for c in cols:
            # format: M1_tc_only_dz<-0.1
            parts = str(c).split("_")
            if len(parts) >= 4 and parts[0] in {"M1", "M2"}:
                model = parts[0]
                mode = parts[1]
                dz = parts[-1].replace("dz<-", "")
                mode_s = "TC-only" if mode == "tc" or mode == "tc_only" else "TC+Vp+Vs"
                pretty.append(f"{model}\n{mode_s}\nδ={dz}")
            else:
                pretty.append(str(c))
        ax.set_xticklabels(pretty, rotation=0, ha="center")
        ax.set_title(r"Threshold crack-like criterion: $P(\Delta z < -\delta_*)$")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("probability")
        ax.set_xlabel("Model / data branch / threshold")
        ax.set_ylabel("lab_sample_id (sorted by φ)")
        fig.savefig(out_dir / "gsa_bayes_p_delta_z_thresholds_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_delta_z_vs_zone_temperature(
    out_png: Path,
    *,
    m1: pd.DataFrame,
    m2: pd.DataFrame,
    sample_stage: pd.DataFrame,
    experiment_metadata: pd.DataFrame,
    mode: str,
) -> None:
    """
    Reference-like plot:
      - background bands for "strong/moderate/no change"
      - side-by-side boxplots (M1 vs M2) for each zone
      - temperature profile over zones on the right axis
    """
    import matplotlib.pyplot as plt

    zones = _parse_heating_zones(experiment_metadata)
    if not zones:
        return

    st = sample_stage.copy()
    st["lab_sample_id"] = pd.to_numeric(st["lab_sample_id"], errors="coerce")
    st["tg_position_mm"] = pd.to_numeric(st.get("tg_position_mm"), errors="coerce")
    st = st.sort_values(["lab_sample_id", "stage"]).drop_duplicates(subset=["lab_sample_id"], keep="first")
    st["zone"] = st["tg_position_mm"].apply(lambda x: _assign_zone(float(x), zones))

    def prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
        tmp = df[df["mode"] == mode].copy()
        tmp = tmp[["lab_sample_id", "alpha_before_p50", "alpha_after_p50", "delta_z_p50"]].copy()
        tmp["lab_sample_id"] = pd.to_numeric(tmp["lab_sample_id"], errors="coerce")
        tmp["alpha_before_p50"] = pd.to_numeric(tmp["alpha_before_p50"], errors="coerce").astype(float)
        tmp["alpha_after_p50"] = pd.to_numeric(tmp["alpha_after_p50"], errors="coerce").astype(float)
        tmp["pct_change"] = 100.0 * (tmp["alpha_after_p50"] / tmp["alpha_before_p50"] - 1.0)
        tmp["model"] = label
        return tmp.merge(st[["lab_sample_id", "zone"]], on="lab_sample_id", how="left")

    a = prep(m1, "M1")
    b = prep(m2, "M2")
    df = pd.concat([a, b], ignore_index=True)
    df = df[np.isfinite(pd.to_numeric(df["pct_change"], errors="coerce"))].copy()
    if df.empty:
        return

    zone_order = [z.name for z in zones]
    df["zone"] = df["zone"].astype(str)
    df["zone"] = pd.Categorical(df["zone"], categories=zone_order, ordered=True)
    df = df.sort_values(["zone", "model"]).reset_index(drop=True)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(16.5, 7.5), constrained_layout=True)

        # background bands (match reference feel)
        ax.axhspan(-45, -20, color="#f5d7d7", alpha=0.55, zorder=0, lw=0)
        ax.axhspan(-20, -5, color="#f7efd0", alpha=0.50, zorder=0, lw=0)
        ax.axhspan(-5, 5, color="#d7f0d7", alpha=0.55, zorder=0, lw=0)

        xs = np.arange(len(zone_order)) + 1
        width = 0.32
        positions_m1 = xs - width / 1.8
        positions_m2 = xs + width / 1.8

        def boxplot_for(model: str, pos: np.ndarray, color: str) -> None:
            data = [df[(df["zone"] == z) & (df["model"] == model)]["pct_change"].to_numpy(float) for z in zone_order]
            bp = ax.boxplot(
                data,
                positions=pos,
                widths=width,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.70)
                patch.set_edgecolor("0.25")
                patch.set_linewidth(1.2)
            for med in bp["medians"]:
                med.set_color("#ff7f0e")
                med.set_linewidth(2.0)

        boxplot_for("M1", positions_m1, "#9bd1ff")
        boxplot_for("M2", positions_m2, "#ffb3b3")

        ax.set_xticks(xs)
        ax.set_xticklabels([f"Zone {i+1}" if str(z).lower().startswith("zone") else str(z) for i, z in enumerate(zone_order)])
        ax.set_ylabel(r"Aspect-ratio change: $100\cdot(\alpha_{after}/\alpha_{before}-1)$ (%)")
        ax.set_title(f"Aspect-ratio change by heating zone (Bayesian, mode={mode})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(-45, 5)

        # Temperature profile (right axis)
        ax2 = ax.twinx()
        temps = [z.temperature_c for z in zones]
        ax2.plot(xs, temps, color="0.25", marker="o", lw=2.0, ms=7, label="Zone temperature profile")
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_ylim(min(temps) - 10, max(temps) + 10)

        # Legend (bottom, reference-like)
        from matplotlib.patches import Patch

        legend_items = [
            Patch(facecolor="#9bd1ff", edgecolor="0.25", alpha=0.70, label="M1: % change"),
            Patch(facecolor="#ffb3b3", edgecolor="0.25", alpha=0.70, label="M2: % change"),
            plt.Line2D([0], [0], color="0.25", lw=2.0, marker="o", label="Zone temperature profile"),
            Patch(facecolor="#f7efd0", edgecolor="none", alpha=0.50, label="−20% to −5%: moderate crack-like shift"),
            Patch(facecolor="#f5d7d7", edgecolor="none", alpha=0.55, label="< −20%: strong crack-like shift"),
            Patch(facecolor="#d7f0d7", edgecolor="none", alpha=0.55, label="−5% to +5%: ~no change"),
        ]
        ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=True)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collection-level artifacts from Bayesian M1/M2 inversions (TC-only, TC+Vp+Vs).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet-stage", type=str, default="sample_stage")
    ap.add_argument("--sheet-meta", type=str, default="experiment_metadata")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    ap.add_argument("--bayes-m1-xlsx", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m1_tc_elastic_results.xlsx"))
    ap.add_argument("--bayes-m2-xlsx", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m2_tc_elastic_results.xlsx"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    m1, m1_mis = _load_bayes_results(Path(args.bayes_m1_xlsx))
    m2, m2_mis = _load_bayes_results(Path(args.bayes_m2_xlsx))

    st = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet_stage))
    meta = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet_meta))

    # Heatmaps: misfit%
    for model, summ, mis in (("M1", m1, m1_mis), ("M2", m2, m2_mis)):
        for mode in ("tc_only", "tc_vp_vs"):
            _plot_misfit_heatmap(out_dir / f"gsa_bayes_{model.lower()}_misfit_heatmap_{mode}.png", summ, mis, model=model, mode=mode)

    # Heatmap: P(Δz<0)
    _plot_pdelta_heatmap(out_dir / "gsa_bayes_p_delta_z_lt_0_heatmap.png", m1, m2)
    _plot_pdelta_threshold_heatmaps(out_dir, m1, m2, thresholds=[0.05, 0.10, 0.20])

    # Δz vs heating zone (Bayesian p50) with temperature profile
    for mode in ("tc_only", "tc_vp_vs"):
        _plot_delta_z_vs_zone_temperature(
            out_dir / f"gsa_bayes_alpha_change_by_heating_zone_{mode}.png",
            m1=m1,
            m2=m2,
            sample_stage=st,
            experiment_metadata=meta,
            mode=mode,
        )

    # Table export
    out_table = m1.merge(m2, on=["lab_sample_id", "mode"], how="outer", suffixes=("_m1", "_m2"))
    out_xlsx = out_dir / "gsa_bayes_collection_table.xlsx"
    out_table.to_excel(out_xlsx, index=False)
    print(f"Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
