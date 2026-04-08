from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa_elastic
from rockphysx.models.emt import gsa_transport

# MT elastic is implemented as a standalone module under `test_new/mori-tanaka/`,
# which is not a valid Python package name (hyphen). Load it dynamically.
import importlib.util
from functools import lru_cache
import sys


@lru_cache(maxsize=1)
def _load_mt_elastic_module() -> object:
    path = Path("test_new/mori-tanaka/strict_mt_elastic_pores.py").resolve()
    spec = importlib.util.spec_from_file_location("strict_mt_elastic_pores", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load MT elastic module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[misc]
    return mod


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _read_constants(path: Path) -> dict[str, float]:
    df = pd.read_excel(path, sheet_name="sample_constants")
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        key = str(r["parameter"]).strip()
        if pd.isna(r["default_value"]):
            continue
        v = pd.to_numeric(r["default_value"], errors="coerce")
        if pd.isna(v):
            continue
        out[key] = float(v)
    return out


def _median_rho_by_stage_fluid(path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_excel(path, sheet_name="measurements_long")
    df = df.dropna(subset=["stage", "fluid_state", "bulk_density_g_cm3"])
    df["stage"] = df["stage"].astype(str).str.strip().str.lower()
    df["fluid_state"] = df["fluid_state"].astype(str).str.strip().str.lower()
    g = df.groupby(["stage", "fluid_state"])["bulk_density_g_cm3"].median()
    out: dict[tuple[str, str], float] = {}
    for (stage, fluid), rho_gcc in g.items():
        out[(str(stage), str(fluid))] = float(rho_gcc) * 1000.0
    return out


def _vp_vs_from_KG_rho(K_pa: float, G_pa: float, rho_kg_m3: float) -> tuple[float, float]:
    vp, vs = gsa_elastic.velocities_from_KG_rho(float(K_pa), float(G_pa), float(rho_kg_m3))
    return float(vp), float(vs)


def _predict_tc_osp(matrix_lambda: float, fluid_lambda: float, phi: float, alpha: float) -> float:
    return float(
        gsa_transport.two_phase_thermal_isotropic(
            float(matrix_lambda),
            float(fluid_lambda),
            float(phi),
            aspect_ratio=float(alpha),
            comparison="matrix",
            max_iter=1,
        )
    )


def _predict_elastic_vp_vs_osp(
    *,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
    matrix_K_pa: float,
    matrix_G_pa: float,
    fluid_K_pa: float,
    phi: float,
    alpha: float,
    rho_bulk_kg_m3: float,
) -> tuple[float, float]:
    matrix = gsa_elastic.ElasticPhase(K=float(matrix_K_pa), G=float(matrix_G_pa))
    inclusion = gsa_elastic.ElasticPhase(K=float(fluid_K_pa), G=1e-9)
    _, _, K_eff, G_eff = gsa_elastic.gsa_effective_stiffness_random_two_phase(
        phi=float(phi),
        matrix=matrix,
        inclusion=inclusion,
        pore_aspect_ratio=float(alpha),
        backend=backend,
        comparison_body="matrix",
        k_connectivity=None,
        sign=-1,
    )
    return _vp_vs_from_KG_rho(float(K_eff), float(G_eff), float(rho_bulk_kg_m3))


def _predict_tc_mt(matrix_lambda: float, fluid_lambda: float, phi: float, alpha: float) -> float:
    # Same scalar MT transport block used in MT inversion scripts.
    # Depolarization factors computed internally via spheroid formula.
    r = float(alpha)
    if abs(r - 1.0) < 1e-12:
        n3 = 1.0 / 3.0
    elif r < 1.0:
        xi = float(np.sqrt(max(1.0 / (r * r) - 1.0, 0.0)))
        n3 = ((1.0 + xi * xi) / (xi**3)) * (xi - float(np.arctan(xi)))
    else:
        e = float(np.sqrt(max(1.0 - 1.0 / (r * r), 0.0)))
        n3 = ((1.0 - e * e) / (2.0 * e**3)) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
    n1 = (1.0 - n3) / 2.0
    a1 = matrix_lambda / (matrix_lambda + n1 * (fluid_lambda - matrix_lambda))
    a3 = matrix_lambda / (matrix_lambda + n3 * (fluid_lambda - matrix_lambda))
    a_bar = (2.0 * a1 + a3) / 3.0
    c_i = float(phi)
    c_m = 1.0 - c_i
    delta = fluid_lambda - matrix_lambda
    return float(matrix_lambda + c_i * delta * a_bar / (c_m + c_i * a_bar))


def _predict_elastic_vp_vs_mt(
    *,
    matrix_K_pa: float,
    matrix_G_pa: float,
    fluid_K_pa: float,
    phi: float,
    alpha: float,
    rho_bulk_kg_m3: float,
) -> tuple[float, float]:
    mt = _load_mt_elastic_module()
    ElasticPhase = getattr(mt, "ElasticPhase")
    strict_mt_elastic_random_spheroidal_pores = getattr(mt, "strict_mt_elastic_random_spheroidal_pores")

    res = strict_mt_elastic_random_spheroidal_pores(
        phi=float(phi),
        a_ratio=float(alpha),
        matrix=ElasticPhase(K=float(matrix_K_pa), G=float(matrix_G_pa)),
        inclusion=ElasticPhase(K=float(fluid_K_pa), G=1e-9),
        rho_matrix_kg_m3=2700.0,
        rho_inclusion_kg_m3=1000.0,
    )
    return _vp_vs_from_KG_rho(float(res.K_eff), float(res.G_eff), float(rho_bulk_kg_m3))


def _style_rc() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }


def _make_phi_grid(phi_max: float, n: int) -> np.ndarray:
    return np.linspace(0.0, float(phi_max), int(n))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensitivity plots: TC/Vp/Vs vs porosity and vs aspect ratio (MT vs OSP).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/inversion_comparison"))
    ap.add_argument("--phi-max", type=float, default=0.30)
    ap.add_argument("--phi-n", type=int, default=121)
    ap.add_argument("--alpha-fixed", type=float, default=0.1)
    ap.add_argument("--phi-fixed-pct", type=str, default="5,15")
    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=81)
    ap.add_argument(
        "--osp-backend-so",
        type=Path,
        default=Path("test_new/gsa/plots/libgsa_elastic_fortran.so"),
        help="Prebuilt elastic GSA backend shared library (avoids gfortran).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    const = _read_constants(Path(args.data_xlsx))
    rho_med = _median_rho_by_stage_fluid(Path(args.data_xlsx))

    # Matrix (HS-feasible set p50 as provided by you earlier)
    matrix_lambda = 2.85
    rho_matrix = 2720.0
    vp_matrix = 6282.0
    vs_matrix = 3531.0
    # Derived moduli ranges come from HS; use p50 here.
    K_matrix = rho_matrix * (vp_matrix**2 - (4.0 / 3.0) * vs_matrix**2)
    G_matrix = rho_matrix * vs_matrix**2

    # Fluids (use the same values as in your modeling table)
    lambda_gas = float(const.get("lambda_gas_w_mk", 0.025))
    lambda_water = float(const.get("lambda_water_w_mk", 0.60))
    K_gas = float(const.get("k_gas_gpa", 0.00014)) * 1e9
    K_water = float(const.get("k_water_gpa", 2.2)) * 1e9

    # OSP backend (load .so, do not rebuild)
    backend = gsa_elastic.build_backend(
        green_fortran=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"),
        output_library=Path(args.osp_backend_so),
        force_rebuild=False,
    )

    # Parse fixed porosities for alpha-sensitivity
    phi_fixed = [float(x) / 100.0 for x in str(args.phi_fixed_pct).split(",") if str(x).strip()]
    if not phi_fixed:
        raise ValueError("--phi-fixed-pct must contain at least one value (e.g. '5,15').")

    stages = ["before", "after"]
    fluids = [("dry", "gas"), ("wet", "water")]

    # ------------------------------------------------------------------
    # Figure 1: TC/Vp/Vs vs porosity at fixed alpha
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    phi_grid = _make_phi_grid(float(args.phi_max), int(args.phi_n))
    alpha0 = float(args.alpha_fixed)

    with plt.rc_context(_style_rc()):
        fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True, sharex=True)
        for r, stage in enumerate(stages):
            for c, prop in enumerate(["tc", "vp", "vs"]):
                ax = axes[r, c]
                for fluid_state, fluid_kind in fluids:
                    if prop == "tc":
                        fl = lambda_gas if fluid_kind == "gas" else lambda_water
                        y_osp = np.array([_predict_tc_osp(matrix_lambda, fl, float(ph), alpha0) for ph in phi_grid])
                        y_mt = np.array([_predict_tc_mt(matrix_lambda, fl, float(ph), alpha0) for ph in phi_grid])
                    else:
                        rho = rho_med.get((stage, fluid_state), rho_matrix)
                        Kf = K_gas if fluid_kind == "gas" else K_water
                        y_osp = np.array(
                            [
                                _predict_elastic_vp_vs_osp(
                                    backend=backend,
                                    matrix_K_pa=K_matrix,
                                    matrix_G_pa=G_matrix,
                                    fluid_K_pa=Kf,
                                    phi=float(ph),
                                    alpha=alpha0,
                                    rho_bulk_kg_m3=rho,
                                )[0 if prop == "vp" else 1]
                                for ph in phi_grid
                            ]
                        )
                        y_mt = np.array(
                            [
                                _predict_elastic_vp_vs_mt(
                                    matrix_K_pa=K_matrix,
                                    matrix_G_pa=G_matrix,
                                    fluid_K_pa=Kf,
                                    phi=float(ph),
                                    alpha=alpha0,
                                    rho_bulk_kg_m3=rho,
                                )[0 if prop == "vp" else 1]
                                for ph in phi_grid
                            ]
                        )

                    color = "#4C78A8" if fluid_state == "wet" else "#F58518"  # pastel-ish blue/orange
                    ax.plot(phi_grid * 100.0, y_osp, color=color, lw=2.4, label=f"OSP {fluid_state}" if (r, c) == (0, 0) else None)
                    ax.plot(phi_grid * 100.0, y_mt, color=color, lw=2.0, ls="--", label=f"MT {fluid_state}" if (r, c) == (0, 0) else None)

                ax.grid(True, alpha=0.25)
                if r == 1:
                    ax.set_xlabel("Porosity, φ (%)")
                if c == 0:
                    ax.set_ylabel("W/(m·K)" if prop == "tc" else "m/s")
                title = {"tc": "Thermal conductivity", "vp": "P-wave velocity", "vs": "S-wave velocity"}[prop]
                ax.set_title(f"{title} — {stage}, α={alpha0:g}")

        axes[0, 0].legend(loc="upper right", frameon=True)
        out_png = out_dir / "sensitivity_tc_vp_vs_vs_porosity_alpha0p1_mt_vs_osp.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: TC/Vp/Vs vs alpha at fixed porosity (phi=5%,15%)
    # ------------------------------------------------------------------
    alphas = np.logspace(np.log10(float(args.alpha_min)), np.log10(float(args.alpha_max)), int(args.alpha_n))

    with plt.rc_context(_style_rc()):
        fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True, sharex=True)
        for r, stage in enumerate(stages):
            for c, prop in enumerate(["tc", "vp", "vs"]):
                ax = axes[r, c]
                for phi in phi_fixed:
                    for fluid_state, fluid_kind in fluids:
                        if prop == "tc":
                            fl = lambda_gas if fluid_kind == "gas" else lambda_water
                            y_osp = np.array([_predict_tc_osp(matrix_lambda, fl, float(phi), float(a)) for a in alphas])
                            y_mt = np.array([_predict_tc_mt(matrix_lambda, fl, float(phi), float(a)) for a in alphas])
                        else:
                            rho = rho_med.get((stage, fluid_state), rho_matrix)
                            Kf = K_gas if fluid_kind == "gas" else K_water
                            y_osp = np.array(
                                [
                                    _predict_elastic_vp_vs_osp(
                                        backend=backend,
                                        matrix_K_pa=K_matrix,
                                        matrix_G_pa=G_matrix,
                                        fluid_K_pa=Kf,
                                        phi=float(phi),
                                        alpha=float(a),
                                        rho_bulk_kg_m3=rho,
                                    )[0 if prop == "vp" else 1]
                                    for a in alphas
                                ]
                            )
                            y_mt = np.array(
                                [
                                    _predict_elastic_vp_vs_mt(
                                        matrix_K_pa=K_matrix,
                                        matrix_G_pa=G_matrix,
                                        fluid_K_pa=Kf,
                                        phi=float(phi),
                                        alpha=float(a),
                                        rho_bulk_kg_m3=rho,
                                    )[0 if prop == "vp" else 1]
                                    for a in alphas
                                ]
                            )

                        # encoding: color by phi, linestyle by fluid_state; method by solid vs dashed overlay
                        color = "#54A24B" if abs(phi - 0.05) < 1e-12 else "#B279A2"  # green / purple
                        ls = "-" if fluid_state == "wet" else "--"
                        lab_base = f"φ={phi*100:.0f}%, {fluid_state}"
                        ax.plot(alphas, y_osp, color=color, lw=2.3, ls=ls, label=f"OSP {lab_base}" if (r, c) == (0, 0) else None)
                        ax.plot(alphas, y_mt, color=color, lw=1.9, ls=ls, alpha=0.9, dashes=(4, 2), label=f"MT {lab_base}" if (r, c) == (0, 0) else None)

                ax.set_xscale("log")
                ax.grid(True, which="both", alpha=0.25)
                if r == 1:
                    ax.set_xlabel("Aspect ratio α (log scale)")
                if c == 0:
                    ax.set_ylabel("W/(m·K)" if prop == "tc" else "m/s")
                title = {"tc": "Thermal conductivity", "vp": "P-wave velocity", "vs": "S-wave velocity"}[prop]
                ax.set_title(f"{title} — {stage}")

        axes[0, 0].legend(loc="upper right", frameon=True, ncols=2)
        out_png = out_dir / "sensitivity_tc_vp_vs_vs_alpha_phi5_15_mt_vs_osp.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
