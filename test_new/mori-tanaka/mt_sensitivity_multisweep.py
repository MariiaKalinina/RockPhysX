
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np

# local elastic MT backend
if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from strict_mt_elastic_pores import ElasticPhase, strict_mt_elastic_random_spheroidal_pores  # noqa: E402


@dataclass(frozen=True)
class SensitivityConfig:
    dpi: int = 300

    ar_min: float = 1e-4
    ar_max: float = 1.0
    n_ar: int = 201
    phi_list: tuple[float, ...] = (0.05, 0.10, 0.20)

    phi_ref: float = 0.15
    ar_color_list: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)

    phi_max: float = 0.35
    n_phi: int = 41

    Km_min_GPa: float = 50.0
    Km_max_GPa: float = 110.0
    Gm_min_GPa: float = 20.0
    Gm_max_GPa: float = 50.0
    lambda_m_min: float = 2.0
    lambda_m_max: float = 4.0
    Kf_min_GPa: float = 0.01
    Kf_max_GPa: float = 3.0
    n_param: int = 41

    Km_pa: float = 76.8e9
    Gm_pa: float = 32.0e9
    Kf_pa: float = 2.2e9
    rho_m_kg_m3: float = 2710.0
    rho_f_kg_m3: float = 1000.0

    tc_matrix: float = 2.86
    tc_fluid: float = 0.60


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def depolarization_factors_spheroid(aspect_ratio: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(aspect_ratio, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("Aspect ratio must be positive and finite.")

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


def mt_transport_single_aspect_ratio(
    phi: float,
    aspect_ratio: float | np.ndarray,
    prop_matrix: float,
    prop_fluid: float,
) -> float | np.ndarray:
    phi = float(phi)
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"Porosity must be in [0,1). Got {phi}")

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


def _property_tuple(phi: float, ar: float, Km_pa: float, Gm_pa: float, Kf_pa: float,
                    rho_m: float, rho_f: float, lambda_m: float, lambda_f: float) -> tuple[float, float, float, float, float, float]:
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
    keff = elastic.K_eff / 1e9
    geff = elastic.G_eff / 1e9
    vpratio = vp / vs if vs > 0 else np.nan
    return tc, keff, geff, vp, vs, vpratio


def compute_ar_sweep_fixed_phi(cfg: SensitivityConfig) -> dict[str, np.ndarray]:
    a_ratio = np.geomspace(cfg.ar_min, cfg.ar_max, cfg.n_ar)
    log10_ar = np.log10(a_ratio)
    phi = np.asarray(cfg.phi_list, dtype=float)

    out = {name: np.full((phi.size, a_ratio.size), np.nan, dtype=float)
           for name in ["tc", "Keff", "Geff", "Vp", "Vs", "VpVs"]}

    for i, ph in enumerate(phi):
        for j, ar in enumerate(a_ratio):
            vals = _property_tuple(
                phi=float(ph), ar=float(ar),
                Km_pa=cfg.Km_pa, Gm_pa=cfg.Gm_pa, Kf_pa=cfg.Kf_pa,
                rho_m=cfg.rho_m_kg_m3, rho_f=cfg.rho_f_kg_m3,
                lambda_m=cfg.tc_matrix, lambda_f=cfg.tc_fluid,
            )
            for key, val in zip(["tc", "Keff", "Geff", "Vp", "Vs", "VpVs"], vals, strict=True):
                out[key][i, j] = val

    out["a_ratio"] = a_ratio
    out["log10_ar"] = log10_ar
    out["phi"] = phi
    return out


def compute_phi_sweep_parametric(cfg: SensitivityConfig) -> dict[str, np.ndarray]:
    phi = np.linspace(0.0, cfg.phi_max, cfg.n_phi)
    a_ratio = np.asarray(cfg.ar_color_list, dtype=float)
    log10_ar = np.log10(a_ratio)

    out = {name: np.full((a_ratio.size, phi.size), np.nan, dtype=float)
           for name in ["tc", "Keff", "Geff", "Vp", "Vs", "VpVs"]}

    for i, ar in enumerate(a_ratio):
        for j, ph in enumerate(phi):
            vals = _property_tuple(
                phi=float(ph), ar=float(ar),
                Km_pa=cfg.Km_pa, Gm_pa=cfg.Gm_pa, Kf_pa=cfg.Kf_pa,
                rho_m=cfg.rho_m_kg_m3, rho_f=cfg.rho_f_kg_m3,
                lambda_m=cfg.tc_matrix, lambda_f=cfg.tc_fluid,
            )
            for key, val in zip(["tc", "Keff", "Geff", "Vp", "Vs", "VpVs"], vals, strict=True):
                out[key][i, j] = val

    out["phi"] = phi
    out["a_ratio"] = a_ratio
    out["log10_ar"] = log10_ar
    return out


def compute_single_param_sweep(cfg: SensitivityConfig, param_name: str) -> dict[str, np.ndarray]:
    a_ratio = np.asarray(cfg.ar_color_list, dtype=float)
    log10_ar = np.log10(a_ratio)
    phi = float(cfg.phi_ref)

    if param_name == "Km":
        x = np.linspace(cfg.Km_min_GPa, cfg.Km_max_GPa, cfg.n_param)
    elif param_name == "Gm":
        x = np.linspace(cfg.Gm_min_GPa, cfg.Gm_max_GPa, cfg.n_param)
    elif param_name == "lambda_m":
        x = np.linspace(cfg.lambda_m_min, cfg.lambda_m_max, cfg.n_param)
    elif param_name == "Kf":
        x = np.linspace(cfg.Kf_min_GPa, cfg.Kf_max_GPa, cfg.n_param)
    else:
        raise ValueError(f"Unknown param_name={param_name}")

    out = {name: np.full((a_ratio.size, x.size), np.nan, dtype=float)
           for name in ["tc", "Keff", "Geff", "Vp", "Vs", "VpVs"]}

    for i, ar in enumerate(a_ratio):
        for j, xv in enumerate(x):
            Km_pa = cfg.Km_pa
            Gm_pa = cfg.Gm_pa
            Kf_pa = cfg.Kf_pa
            lambda_m = cfg.tc_matrix

            if param_name == "Km":
                Km_pa = xv * 1e9
            elif param_name == "Gm":
                Gm_pa = xv * 1e9
            elif param_name == "lambda_m":
                lambda_m = xv
            elif param_name == "Kf":
                Kf_pa = xv * 1e9

            vals = _property_tuple(
                phi=phi, ar=float(ar),
                Km_pa=Km_pa, Gm_pa=Gm_pa, Kf_pa=Kf_pa,
                rho_m=cfg.rho_m_kg_m3, rho_f=cfg.rho_f_kg_m3,
                lambda_m=lambda_m, lambda_f=cfg.tc_fluid,
            )
            for key, val in zip(["tc", "Keff", "Geff", "Vp", "Vs", "VpVs"], vals, strict=True):
                out[key][i, j] = val

    out["x"] = x
    out["a_ratio"] = a_ratio
    out["log10_ar"] = log10_ar
    out["phi_ref"] = phi
    out["param_name"] = param_name
    return out


def _panel_layout():
    return [
        ("Thermal conductivity", "λeff, W/(m·K)", "tc", False),
        ("Bulk modulus", "Keff, GPa", "Keff", False),
        ("Shear modulus", "Geff, GPa", "Geff", False),
        ("P-wave velocity", "Vp, km/s", "Vp", False),
        ("S-wave velocity", "Vs, km/s", "Vs", False),
        ("Vp/Vs ratio", "Vp/Vs", "VpVs", False),
    ]


def save_fixed_phi_ar_png(curves: dict[str, np.ndarray], cfg: SensitivityConfig, out_dir: Path) -> Path:
    _configure_matplotlib_env(out_dir)
    import matplotlib.pyplot as plt

    x = curves["log10_ar"]
    phi = curves["phi"]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True, constrained_layout=True)
    panels = _panel_layout()

    for ax, (title, ylab, key, logy) in zip(axes.ravel(), panels, strict=True):
        Y = curves[key]
        for i, ph in enumerate(phi):
            ax.plot(x, Y[i, :], lw=2.0, label=f"φ={ph*100:.0f}%")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        if logy and np.nanmin(Y) > 0:
            ax.set_yscale("log")

    for ax in axes[-1, :]:
        ax.set_xlabel("log10(α)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(phi), frameon=True, bbox_to_anchor=(0.5, -0.01))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mt_sensitivity_ar_fixed_phi.png"
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_parametric_png(maps: dict[str, np.ndarray], cfg: SensitivityConfig, out_dir: Path, x_key: str, x_label: str, filename: str, subtitle: str) -> Path:
    _configure_matplotlib_env(out_dir)
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    x = maps[x_key]
    a_ratio = maps["a_ratio"]
    log10_ar = np.log10(a_ratio)

    norm = mpl.colors.Normalize(vmin=float(log10_ar.min()), vmax=float(log10_ar.max()))
    cmap = mpl.colormaps["turbo"]
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True, constrained_layout=True)
    panels = _panel_layout()

    for ax, (title, ylab, key, logy) in zip(axes.ravel(), panels, strict=True):
        Y = maps[key]
        for i, lg in enumerate(log10_ar):
            color = cmap(norm(float(lg)))
            ax.plot(x, Y[i, :], color=color, lw=1.2, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        if logy and np.nanmin(Y) > 0:
            ax.set_yscale("log")

    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01)
    cbar.set_label("log10(α)")
    fig.suptitle(subtitle, y=1.01, fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Unified sensitivity plots for strict MT rock-physics model.")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "mt_sensitivity_multisweep_plots")
    p.add_argument("--dpi", type=int, default=SensitivityConfig.dpi)
    args = p.parse_args()

    cfg = SensitivityConfig(dpi=int(args.dpi))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ar_curves = compute_ar_sweep_fixed_phi(cfg)
    path1 = save_fixed_phi_ar_png(ar_curves, cfg, out_dir)
    print(f"Saved: {path1}")

    phi_maps = compute_phi_sweep_parametric(cfg)
    path2 = save_parametric_png(
        phi_maps, cfg, out_dir,
        x_key="phi",
        x_label="Porosity, φ (fraction)",
        filename="mt_sensitivity_phi_parametric.png",
        subtitle=f"Km={cfg.Km_pa/1e9:g} GPa, Gm={cfg.Gm_pa/1e9:g} GPa, Kf={cfg.Kf_pa/1e9:g} GPa, λm={cfg.tc_matrix:g}, λf={cfg.tc_fluid:g}",
    )
    print(f"Saved: {path2}")

    for param_name, x_label, filename in [
        ("Km", "Matrix bulk modulus, Km (GPa)", "mt_sensitivity_Km_parametric.png"),
        ("Gm", "Matrix shear modulus, Gm (GPa)", "mt_sensitivity_Gm_parametric.png"),
        ("lambda_m", "Matrix thermal conductivity, λm (W/(m·K))", "mt_sensitivity_lambda_m_parametric.png"),
        ("Kf", "Fluid bulk modulus, Kf (GPa)", "mt_sensitivity_Kf_parametric.png"),
    ]:
        maps = compute_single_param_sweep(cfg, param_name=param_name)
        subtitle = f"φ={cfg.phi_ref:.2f}, λf={cfg.tc_fluid:g}, Km={cfg.Km_pa/1e9:g} GPa, Gm={cfg.Gm_pa/1e9:g} GPa, Kf={cfg.Kf_pa/1e9:g} GPa"
        path = save_parametric_png(
            maps, cfg, out_dir,
            x_key="x",
            x_label=x_label,
            filename=filename,
            subtitle=subtitle,
        )
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
