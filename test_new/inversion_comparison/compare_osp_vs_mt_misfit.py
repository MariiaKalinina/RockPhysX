from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _canonical_property(name: str) -> str:
    key = str(name).strip().lower()
    return {"tc_w_mk": "tc", "vp_m_s": "vp", "vs_m_s": "vs"}.get(key, key)


def _load_osp_m1_misfits(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="misfit_detail")
    df = df.rename(columns={"lab_sample_id": "sample_id"})
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype(float)
    df["mode"] = df["mode"].astype(str)
    df["stage"] = df["stage"].astype(str)
    df["fluid_state"] = df["fluid_state"].astype(str)
    df["property"] = df["property"].map(_canonical_property)
    df["misfit_pct"] = pd.to_numeric(df["misfit_pct"], errors="coerce").astype(float)
    return df[["sample_id", "mode", "stage", "fluid_state", "property", "misfit_pct"]].copy()


def _load_mt_tc_only_misfits(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="predictions")
    df = df.rename(columns={"sample_id": "sample_id", "tc_obs_w_mk": "obs", "tc_pred_w_mk": "pred"})
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].astype(str)
    df["fluid_state"] = df["fluid_state"].astype(str)
    df["property"] = "tc"
    df["misfit_pct"] = (pd.to_numeric(df["pred"], errors="coerce") - pd.to_numeric(df["obs"], errors="coerce")).abs()
    df["misfit_pct"] = df["misfit_pct"] / pd.to_numeric(df["obs"], errors="coerce").abs() * 100.0
    df["mode"] = "tc_only"
    return df[["sample_id", "mode", "stage", "fluid_state", "property", "misfit_pct"]].copy()


def _load_mt_joint_m1_misfits(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="predictions")
    df = df.rename(columns={"sample_id": "sample_id"})
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype(float)
    df["stage"] = df["stage"].astype(str)
    df["fluid_state"] = df["fluid_state"].astype(str)
    df["property"] = df["property"].map(_canonical_property)
    df["mode"] = "tc_vp_vs"

    df = df[df["model"].astype(str).str.upper() == "M1"].copy()

    obs = pd.to_numeric(df["obs"], errors="coerce").astype(float).to_numpy(float)
    pred = pd.to_numeric(df["pred"], errors="coerce").astype(float).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        misfit_pct = np.abs(pred - obs) / np.abs(obs) * 100.0
    df["misfit_pct"] = misfit_pct

    return df[["sample_id", "mode", "stage", "fluid_state", "property", "misfit_pct"]].copy()


def _summary_table(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    out = (
        df.groupby(["property", "stage", "fluid_state"])["misfit_pct"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    out["tag"] = tag
    return out


def _boxplot(out_png: Path, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # Keep only the properties we actually compare.
    df = df[df["property"].isin(["tc", "vp", "vs"])].copy()

    # Define panel order.
    properties = ["tc", "vp", "vs"]
    stages = ["before", "after"]
    fluids = ["dry", "wet"]
    tags = ["MT", "OSP"]

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(properties), 1, figsize=(10.5, 9.5), constrained_layout=True, sharex=True)
        if len(properties) == 1:
            axes = [axes]

        colors = {"MT": "#8da0cb", "OSP": "#fc8d62"}  # pastel blue/orange

        x = 0
        xticks = []
        xticklabels = []
        for prop_i, prop in enumerate(properties):
            ax = axes[prop_i]
            x = 0
            positions = []
            box_data = []
            box_colors = []
            for stage in stages:
                for fluid in fluids:
                    # Two methods side-by-side.
                    for tag in tags:
                        sel = df[
                            (df["tag"] == tag)
                            & (df["property"] == prop)
                            & (df["stage"] == stage)
                            & (df["fluid_state"] == fluid)
                        ]["misfit_pct"].dropna()
                        box_data.append(sel.to_numpy(float))
                        positions.append(x)
                        box_colors.append(colors[tag])
                        x += 1
                    # small gap between dry/wet
                    x += 0.6
                # bigger gap between before/after
                x += 0.8

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.7,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "0.15", "linewidth": 1.5},
                whiskerprops={"color": "0.35", "linewidth": 1.2},
                capprops={"color": "0.35", "linewidth": 1.2},
            )
            for patch, c in zip(bp["boxes"], box_colors, strict=False):
                patch.set_facecolor(c)
                patch.set_edgecolor("0.35")
                patch.set_alpha(0.85)

            ax.grid(True, axis="y", alpha=0.25)
            ax.set_ylabel("misfit (%)")
            ax.set_title(f"{prop.upper()} misfit")

            # X ticks: one per (stage,fluid) group, centered between MT/OSP.
            xt = []
            xl = []
            x_local = 0
            for stage in stages:
                for fluid in fluids:
                    center = x_local + 0.5  # between MT and OSP
                    xt.append(center)
                    xl.append(f"{stage}\\n{fluid}")
                    x_local += 2 + 0.6
                x_local += 0.8
            ax.set_xticks(xt)
            ax.set_xticklabels(xl)

            # Legend once.
            if prop_i == 0:
                handles = [
                    plt.Line2D([0], [0], color=colors["MT"], lw=10, alpha=0.85),
                    plt.Line2D([0], [0], color=colors["OSP"], lw=10, alpha=0.85),
                ]
                ax.legend(handles, ["MT", "OSP"], loc="upper right", frameon=True)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle("OSP vs Mori–Tanaka: misfit distributions (M1)", fontsize=16)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare misfit% distributions: OSP (GSA) vs Mori–Tanaka (M1).")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--osp-m1-xlsx", type=Path, default=Path("test_new/gsa/plots/gsa_bayes_m1_tc_elastic_results.xlsx"))
    ap.add_argument("--mt-tc-only-xlsx", type=Path, default=Path("test_new/mt_tc_m1_m2_results.xlsx"))
    ap.add_argument("--mt-joint-xlsx", type=Path, default=Path("test_new/strict_mt_tc_vp_vs_inversion_fast_results.xlsx"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    osp = _load_osp_m1_misfits(Path(args.osp_m1_xlsx))
    mt_tc = _load_mt_tc_only_misfits(Path(args.mt_tc_only_xlsx))
    mt_joint = _load_mt_joint_m1_misfits(Path(args.mt_joint_xlsx))

    # Normalize tags and merge in one long table.
    osp = osp.copy()
    osp["tag"] = "OSP"
    mt_tc = mt_tc.copy()
    mt_tc["tag"] = "MT"
    mt_joint = mt_joint.copy()
    mt_joint["tag"] = "MT"

    # OSP contains both modes. MT is split into two sources; merge them.
    long_df = pd.concat([osp, mt_tc, mt_joint], ignore_index=True)

    # Summary CSV
    summary = pd.concat(
        [
            _summary_table(long_df[(long_df["tag"] == "OSP") & (long_df["mode"] == "tc_only")], "OSP_M1_tc_only"),
            _summary_table(long_df[(long_df["tag"] == "OSP") & (long_df["mode"] == "tc_vp_vs")], "OSP_M1_tc_vp_vs"),
            _summary_table(long_df[(long_df["tag"] == "MT") & (long_df["mode"] == "tc_only")], "MT_tc_only"),
            _summary_table(long_df[(long_df["tag"] == "MT") & (long_df["mode"] == "tc_vp_vs")], "MT_M1_tc_vp_vs"),
        ],
        ignore_index=True,
    )
    summary = summary.sort_values(["property", "stage", "fluid_state", "tag"]).reset_index(drop=True)
    out_csv = out_dir / "osp_vs_mt_misfit_summary.csv"
    summary.to_csv(out_csv, index=False)

    # Boxplot figure (joint only, where all three properties exist).
    joint_long = long_df[long_df["mode"] == "tc_vp_vs"].copy()
    out_png = out_dir / "osp_vs_mt_misfit_boxplots.png"
    _boxplot(out_png, joint_long)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

