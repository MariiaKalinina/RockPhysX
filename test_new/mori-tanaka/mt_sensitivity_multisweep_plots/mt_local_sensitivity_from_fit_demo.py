
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd

if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd()
_BACKEND_DIR = _THIS_DIR.parent  # `strict_mt_elastic_pores.py` lives one level up
for _p in (str(_THIS_DIR), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from strict_mt_elastic_pores import ElasticPhase, strict_mt_elastic_random_spheroidal_pores  # noqa: E402


@dataclass(frozen=True)
class LocalConfig:
    dpi: int = 300
    objective: str = "rel_l1"
    sample_id: str = "25.0"

    # local ranges
    n_ar: int = 151
    ar_window_log10: float = 0.7
    phi_window_abs: float = 0.05
    n_param: int = 101

    Km_rel_window: float = 0.25
    Gm_rel_window: float = 0.25
    lambda_m_rel_window: float = 0.25
    Kf_rel_window: float = 0.50


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def depolarization_factors_spheroid(aspect_ratio: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(aspect_ratio, dtype=float)
    n3 = np.empty_like(r)
    sphere = np.isclose(r, 1.0)
    oblate = r < 1.0
    prolate = r > 1.0
    n3[sphere] = 1.0 / 3.0
    rr = r[oblate]
    if rr.size > 0:
        xi = np.sqrt(np.maximum(1.0 / (rr * rr) - 1.0, 0.0))
        n3[oblate] = ((1.0 + xi * xi) / (xi**3)) * (xi - np.arctan(xi))
    rr = r[prolate]
    if rr.size > 0:
        e = np.sqrt(np.maximum(1.0 - 1.0 / (rr * rr), 0.0))
        n3[prolate] = ((1.0 - e * e) / (2.0 * e**3)) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
    n1 = (1.0 - n3) / 2.0
    return n1, n3


def mt_transport_single_aspect_ratio(phi: float, aspect_ratio: float | np.ndarray, prop_matrix: float, prop_fluid: float) -> float | np.ndarray:
    phi = float(phi)
    r = np.asarray(aspect_ratio, dtype=float)
    n1, n3 = depolarization_factors_spheroid(r)
    dm = float(prop_matrix)
    df = float(prop_fluid)
    delta = df - dm
    a1 = dm / (dm + n1 * delta)
    a3 = dm / (dm + n3 * delta)
    a_bar = (2.0 * a1 + a3) / 3.0
    c_i = phi
    c_m = 1.0 - phi
    value = dm + c_i * delta * a_bar / (c_m + c_i * a_bar)
    if np.ndim(value) == 0:
        return float(value)
    return value


def property_tuple(phi: float, ar: float, Km_pa: float, Gm_pa: float, Kf_pa: float,
                   rho_m: float, rho_f: float, lambda_m: float, lambda_f: float) -> tuple[float, float, float]:
    tc = float(mt_transport_single_aspect_ratio(phi, ar, lambda_m, lambda_f))
    elastic = strict_mt_elastic_random_spheroidal_pores(
        phi=float(phi),
        a_ratio=float(ar),
        matrix=ElasticPhase(K=Km_pa, G=Gm_pa),
        inclusion=ElasticPhase(K=Kf_pa, G=1e-9),
        rho_matrix_kg_m3=float(rho_m),
        rho_inclusion_kg_m3=float(rho_f),
    )
    vp = elastic.vp_m_s / 1e3
    vs = elastic.vs_m_s / 1e3
    return tc, vp, vs


def load_data(base_dir: Path):
    inv = pd.read_excel(base_dir / "strict_mt_tc_vp_vs_inversion_fast_results.xlsx", sheet_name="M1_fits"), \
          pd.read_excel(base_dir / "strict_mt_tc_vp_vs_inversion_fast_results.xlsx", sheet_name="M2_fits")
    joint = pd.read_excel(base_dir / "strict_mt_tc_vp_vs_inversion_fast_results.xlsx", sheet_name="joint_dataset")
    constants = pd.read_excel(base_dir / "data_restructured_for_MT_v2.xlsx", sheet_name="sample_constants")
    return inv[0], inv[1], joint, constants


def scalar_constant(constants_df: pd.DataFrame, parameter: str, fallback: float) -> float:
    row = constants_df.loc[constants_df["parameter"] == parameter]
    return float(row["default_value"].iloc[0]) if not row.empty else float(fallback)


def get_base_params(constants_df: pd.DataFrame, fluid_state: str):
    is_dry = str(fluid_state).lower() == "dry"
    return dict(
        Km_pa=scalar_constant(constants_df, "k_matrix_gpa", 76.8) * 1e9,
        Gm_pa=scalar_constant(constants_df, "g_matrix_gpa", 32.0) * 1e9,
        Kf_pa=(scalar_constant(constants_df, "k_gas_gpa", 0.00014) if is_dry else scalar_constant(constants_df, "k_water_gpa", 2.2)) * 1e9,
        rho_m=scalar_constant(constants_df, "rho_matrix_kg_m3", 2710.0),
        rho_f=1.2 if is_dry else 1000.0,
        lambda_m=scalar_constant(constants_df, "lambda_matrix_w_mk", 2.86),
        lambda_f=scalar_constant(constants_df, "lambda_gas_w_mk", 0.025) if is_dry else scalar_constant(constants_df, "lambda_water_w_mk", 0.60),
    )


def sample_observations(joint_df: pd.DataFrame, sample_id: str):
    g = joint_df[joint_df["sample_id"].astype(str) == str(sample_id)].copy()
    g = g.sort_values(["stage", "fluid_state"]).reset_index(drop=True)
    return g


def save_local_ar_phi_png(sample_df, fit_row, model_name: str, cfg: LocalConfig, constants_df: pd.DataFrame, out_dir: Path) -> Path:
    _configure_matplotlib_env(out_dir)
    import matplotlib.pyplot as plt

    # Use wet rows for local illustrative plot (stronger fluid effect, cleaner comparison)
    rows = {stage: sample_df[(sample_df["stage"] == stage) & (sample_df["fluid_state"] == "wet")].iloc[0] for stage in ["before", "after"]}

    if model_name == "M1":
        ar_before = float(fit_row["ar_before"])
        ar_after = float(fit_row["ar_after"])
    else:
        # convert mean position m into representative AR on log-scale
        ar_before = 10.0 ** (-4.0 + 4.0 * float(fit_row["m_before"]))
        ar_after = 10.0 ** (-4.0 + 4.0 * float(fit_row["m_after"]))

    z_center = np.log10(np.sqrt(ar_before * ar_after))
    ar_grid = np.geomspace(10 ** max(-4.0, z_center - cfg.ar_window_log10),
                           10 ** min(0.0, z_center + cfg.ar_window_log10),
                           cfg.n_ar)

    phi_before = float(rows["before"]["phi_frac"])
    phi_after = float(rows["after"]["phi_frac"])
    phi_grid_before = np.linspace(max(0.001, phi_before - cfg.phi_window_abs), min(0.45, phi_before + cfg.phi_window_abs), cfg.n_param)
    phi_grid_after = np.linspace(max(0.001, phi_after - cfg.phi_window_abs), min(0.45, phi_after + cfg.phi_window_abs), cfg.n_param)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), constrained_layout=True)
    props = [("TC", "λeff, W/(m·K)"), ("Vp", "Vp, km/s"), ("Vs", "Vs, km/s")]

    # left column: AR sensitivity
    for i, (pname, ylabel) in enumerate(props):
        ax = axes[i, 0]
        for stage, ar_mark in [("before", ar_before), ("after", ar_after)]:
            row = rows[stage]
            pars = get_base_params(constants_df, row["fluid_state"])
            vals = []
            for ar in ar_grid:
                tc, vp, vs = property_tuple(float(row["phi_frac"]), float(ar), **pars)
                vals.append({"TC": tc, "Vp": vp, "Vs": vs}[pname])
            ax.plot(np.log10(ar_grid), vals, lw=2.0, label=stage)
            obs_val = {"TC": float(row["tc_w_mk"]), "Vp": float(row["vp_m_s"]) / 1e3, "Vs": float(row["vs_m_s"]) / 1e3}[pname]
            pred_val = property_tuple(float(row["phi_frac"]), float(ar_mark), **pars)
            pred_val = {"TC": pred_val[0], "Vp": pred_val[1], "Vs": pred_val[2]}[pname]
            ax.scatter([np.log10(ar_mark)], [pred_val], s=55, marker="o")
            ax.scatter([np.log10(ar_mark)], [obs_val], s=55, marker="x")
        ax.set_title(f"{pname} vs log10(α)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if i == 2:
            ax.set_xlabel("log10(α)")

    # right column: phi sensitivity
    for i, (pname, ylabel) in enumerate(props):
        ax = axes[i, 1]
        for stage, phi_grid, ar_fix in [("before", phi_grid_before, ar_before), ("after", phi_grid_after, ar_after)]:
            row = rows[stage]
            pars = get_base_params(constants_df, row["fluid_state"])
            vals = []
            for ph in phi_grid:
                tc, vp, vs = property_tuple(float(ph), float(ar_fix), **pars)
                vals.append({"TC": tc, "Vp": vp, "Vs": vs}[pname])
            ax.plot(phi_grid, vals, lw=2.0, label=stage)
            obs_val = {"TC": float(row["tc_w_mk"]), "Vp": float(row["vp_m_s"]) / 1e3, "Vs": float(row["vs_m_s"]) / 1e3}[pname]
            pred_val = property_tuple(float(row["phi_frac"]), float(ar_fix), **pars)
            pred_val = {"TC": pred_val[0], "Vp": pred_val[1], "Vs": pred_val[2]}[pname]
            ax.scatter([float(row["phi_frac"])], [pred_val], s=55, marker="o")
            ax.scatter([float(row["phi_frac"])], [obs_val], s=55, marker="x")
        ax.set_title(f"{pname} vs φ")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if i == 2:
            ax.set_xlabel("Porosity, φ")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Local sensitivity around fitted solution — sample {cfg.sample_id}, {model_name}", y=1.01, fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"local_ar_phi_sample_{cfg.sample_id}_{model_name}.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_local_nuisance_png(sample_df, fit_row, model_name: str, cfg: LocalConfig, constants_df: pd.DataFrame, out_dir: Path) -> Path:
    _configure_matplotlib_env(out_dir)
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    row = sample_df[(sample_df["stage"] == "before") & (sample_df["fluid_state"] == "wet")].iloc[0]

    if model_name == "M1":
        ar_fix = float(fit_row["ar_before"])
    else:
        ar_fix = 10.0 ** (-4.0 + 4.0 * float(fit_row["m_before"]))

    base = get_base_params(constants_df, row["fluid_state"])

    sweep_defs = {
        "Km": np.linspace(base["Km_pa"] * (1-cfg.Km_rel_window), base["Km_pa"] * (1+cfg.Km_rel_window), cfg.n_param),
        "Gm": np.linspace(base["Gm_pa"] * (1-cfg.Gm_rel_window), base["Gm_pa"] * (1+cfg.Gm_rel_window), cfg.n_param),
        "lambda_m": np.linspace(base["lambda_m"] * (1-cfg.lambda_m_rel_window), base["lambda_m"] * (1+cfg.lambda_m_rel_window), cfg.n_param),
        "Kf": np.linspace(max(1e7, base["Kf_pa"] * (1-cfg.Kf_rel_window)), base["Kf_pa"] * (1+cfg.Kf_rel_window), cfg.n_param),
    }

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), constrained_layout=True)
    panels = [("TC", "λeff, W/(m·K)"), ("Vp", "Vp, km/s"), ("Vs", "Vs, km/s"), ("TC", "λeff, W/(m·K)"), ("Vp", "Vp, km/s"), ("Vs", "Vs, km/s")]
    sweep_names = ["Km", "Km", "Km", "Gm", "lambda_m", "Kf"]

    cmap = mpl.colormaps["turbo"]
    color = cmap(mpl.colors.Normalize(vmin=-4.0, vmax=0.0)(np.log10(ar_fix)))

    for ax, (pname, ylabel), sweep_name in zip(axes.ravel(), panels, sweep_names, strict=True):
        xgrid = sweep_defs[sweep_name]
        vals = []
        for xv in xgrid:
            pars = base.copy()
            if sweep_name == "Km":
                pars["Km_pa"] = float(xv)
            elif sweep_name == "Gm":
                pars["Gm_pa"] = float(xv)
            elif sweep_name == "lambda_m":
                pars["lambda_m"] = float(xv)
            elif sweep_name == "Kf":
                pars["Kf_pa"] = float(xv)
            tc, vp, vs = property_tuple(float(row["phi_frac"]), float(ar_fix), **pars)
            vals.append({"TC": tc, "Vp": vp, "Vs": vs}[pname])
        xplot = xgrid / 1e9 if sweep_name in ["Km", "Gm", "Kf"] else xgrid
        ax.plot(xplot, vals, color=color, lw=2.0)
        obs_val = {"TC": float(row["tc_w_mk"]), "Vp": float(row["vp_m_s"]) / 1e3, "Vs": float(row["vs_m_s"]) / 1e3}[pname]
        base_pred = property_tuple(float(row["phi_frac"]), float(ar_fix), **base)
        base_pred = {"TC": base_pred[0], "Vp": base_pred[1], "Vs": base_pred[2]}[pname]
        x0 = {"Km": base["Km_pa"]/1e9, "Gm": base["Gm_pa"]/1e9, "lambda_m": base["lambda_m"], "Kf": base["Kf_pa"]/1e9}[sweep_name]
        ax.scatter([x0], [base_pred], s=55, marker="o", color=color)
        ax.scatter([x0], [obs_val], s=55, marker="x", color=color)
        ax.set_title(f"{pname} vs {sweep_name}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel({"Km":"Km (GPa)", "Gm":"Gm (GPa)", "lambda_m":"λm (W/(m·K))", "Kf":"Kf (GPa)"}[sweep_name])

    fig.suptitle(f"Local nuisance-parameter sensitivity — sample {cfg.sample_id}, {model_name}, wet/before", y=1.01, fontsize=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"local_nuisance_sample_{cfg.sample_id}_{model_name}.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Local sensitivity plots around fitted inversion results.")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "mt_local_sensitivity_from_fit_demo_plots")
    p.add_argument("--sample-id", type=str, default=LocalConfig.sample_id)
    p.add_argument("--dpi", type=int, default=LocalConfig.dpi)
    args = p.parse_args()

    cfg = LocalConfig(sample_id=str(args.sample_id), dpi=int(args.dpi))
    out_dir = args.out_dir

    base_dir = _THIS_DIR
    m1, m2, joint, constants = load_data(base_dir)
    sample_df = sample_observations(joint, cfg.sample_id)

    fit1 = m1[m1["sample_id"].astype(str) == cfg.sample_id].iloc[0]
    fit2 = m2[m2["sample_id"].astype(str) == cfg.sample_id].iloc[0]

    p1 = save_local_ar_phi_png(sample_df, fit1, "M1", cfg, constants, out_dir)
    p2 = save_local_ar_phi_png(sample_df, fit2, "M2", cfg, constants, out_dir)
    p3 = save_local_nuisance_png(sample_df, fit1, "M1", cfg, constants, out_dir)
    p4 = save_local_nuisance_png(sample_df, fit2, "M2", cfg, constants, out_dir)

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {p4}")


if __name__ == "__main__":
    main()
