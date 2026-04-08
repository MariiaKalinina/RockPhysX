from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa_elastic
from rockphysx.models.emt import gsa_transport


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _parse_samples(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("--samples must contain at least one lab_sample_id")
    return out


def _phi_from_row(row: pd.Series) -> float:
    if pd.notna(row.get("phi_frac")):
        return float(row["phi_frac"])
    if pd.notna(row.get("phi_pct")):
        return float(row["phi_pct"]) / 100.0
    raise ValueError("Row has neither phi_frac nor phi_pct.")


@dataclass(frozen=True)
class MatrixParams:
    lambda_w_mk: float
    vp_m_s: float
    vs_m_s: float
    rho_kg_m3: float
    sigma_s_m: float


@dataclass(frozen=True)
class FluidParams:
    name: str
    lambda_w_mk: float
    K_gpa: float
    rho_kg_m3: float
    sigma_s_m: float | None = None


def _matrix_KG_from_vp_vs_rho(vp_m_s: float, vs_m_s: float, rho_kg_m3: float) -> tuple[float, float]:
    vp = float(vp_m_s)
    vs = float(vs_m_s)
    rho = float(rho_kg_m3)
    G = rho * vs * vs
    K = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
    return float(K), float(G)


def _predict_tc_isotropic(matrix_lambda: float, fluid_lambda: float, phi: float, alpha: float) -> float:
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


def _predict_sigma_isotropic(
    matrix_sigma: float,
    fluid_sigma: float,
    phi: float,
    alpha: float,
    *,
    comparison: str,
    solid_aspect_ratio: float | None = None,
) -> float:
    phi = float(phi)
    alpha = float(alpha)

    # Transport GSA uses the *comparison tensor* Tc to build the inclusion operator.
    # If Tc is exactly equal to a phase property, that phase has Δ=0, hence Mi=I and
    # becomes shape-independent. This is why "fluid comparison" kills α-sensitivity if
    # α is attached to the fluid phase.
    #
    # We support two practical topologies for sanity checks:
    #
    # 1) comparison="matrix":
    #    - Phase list: [solid matrix, pore fluid]
    #    - Tc = solid matrix
    #    - α is the pore shape (fluid inclusions in solid).
    #
    # 2) comparison="fluid_host":
    #    - Phase list: [pore fluid, solid matrix]
    #    - Tc = pore fluid (connected brine host proxy)
    #    - α is applied to the *solid* inclusion shape (solid grains in brine).
    #      This gives α-sensitivity, but note it is now the solid-inclusion aspect ratio.
    if comparison == "matrix":
        phases = [
            gsa_transport.make_phase("matrix", 1.0 - phi, float(matrix_sigma), aspect_ratio=1.0, orientation="random"),
            gsa_transport.make_phase("pore_fluid", phi, float(fluid_sigma), aspect_ratio=alpha, orientation="random"),
        ]
        body = gsa_transport.ComparisonBody(kind="matrix", matrix_index=0)
    elif comparison == "fluid_host":
        a_solid = alpha if solid_aspect_ratio is None else float(solid_aspect_ratio)
        phases = [
            gsa_transport.make_phase("pore_fluid", phi, float(fluid_sigma), aspect_ratio=1.0, orientation="random"),
            gsa_transport.make_phase("matrix", 1.0 - phi, float(matrix_sigma), aspect_ratio=a_solid, orientation="random"),
        ]
        body = gsa_transport.ComparisonBody(kind="matrix", matrix_index=0)
    else:
        raise ValueError("comparison must be 'matrix' or 'fluid_host'")

    return float(gsa_transport.gsa_transport_isotropic(phases, body, max_iter=1))


def _predict_elastic_vp_vs(
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
    # fluid shear is ~0 but must be non-zero for numerical stability in the closure
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
    vp, vs = gsa_elastic.velocities_from_KG_rho(K_eff, G_eff, float(rho_bulk_kg_m3))
    return float(vp), float(vs)


def _plot_one_sample(
    *,
    out_png: Path,
    sample_id: float,
    rows: pd.DataFrame,
    alphas: np.ndarray,
    matrix: MatrixParams,
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    brine_sigma_before: float,
    brine_sigma_after: float,
    sigma_comparison: str,
    sigma_solid_aspect_map: str,
    r_scale: str,
    r_ylim_min: float | None,
    r_ylim_max: float | None,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
) -> None:
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
    }

    # Ensure rows exist for before/after
    stages = ["before", "after"]
    fluids = ["dry", "wet"]

    # Compute matrix elastic moduli (fixed, from HS-feasible set)
    K_m, G_m = _matrix_KG_from_vp_vs_rho(matrix.vp_m_s, matrix.vs_m_s, matrix.rho_kg_m3)

    # Precompute predictions
    pred = {}
    for stage in stages:
        sub_stage = rows[rows["stage"] == stage]
        if sub_stage.empty:
            continue
        # fixed phi per stage (from data; assumed same for dry/wet)
        phi = _phi_from_row(sub_stage.iloc[0])
        rho_bulk_gcc = float(sub_stage.iloc[0]["bulk_density_g_cm3"])
        rho_bulk = rho_bulk_gcc * 1e3

        pred[(stage, "tc_dry")] = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_dry.lambda_w_mk, phi, a) for a in alphas])
        pred[(stage, "tc_wet")] = np.array([_predict_tc_isotropic(matrix.lambda_w_mk, fluid_wet.lambda_w_mk, phi, a) for a in alphas])

        # Resistivity (wet only): brine sigma depends on stage (before/after MPCT)
        sigma_f = float(brine_sigma_before if stage == "before" else brine_sigma_after)
        if sigma_comparison == "fluid_host":
            if sigma_solid_aspect_map == "direct":
                solid_ar = None  # use α directly
            elif sigma_solid_aspect_map == "inverse":
                solid_ar = 1.0 / alphas
            else:
                raise ValueError("sigma_solid_aspect_map must be 'direct' or 'inverse'")
        else:
            solid_ar = None
        sigma_eff = np.array(
            [
                _predict_sigma_isotropic(
                    matrix.sigma_s_m,
                    sigma_f,
                    phi,
                    float(a),
                    comparison=sigma_comparison,
                    solid_aspect_ratio=(None if solid_ar is None else float(solid_ar[i])),
                )
                for i, a in enumerate(alphas)
            ]
        )
        pred[(stage, "R_wet")] = 1.0 / sigma_eff

        # Elastic velocities
        vp_dry = np.full_like(alphas, np.nan, dtype=float)
        vs_dry = np.full_like(alphas, np.nan, dtype=float)
        vp_wet = np.full_like(alphas, np.nan, dtype=float)
        vs_wet = np.full_like(alphas, np.nan, dtype=float)
        for i, a in enumerate(alphas):
            vp_dry[i], vs_dry[i] = _predict_elastic_vp_vs(
                backend=backend,
                matrix_K_pa=K_m,
                matrix_G_pa=G_m,
                fluid_K_pa=float(fluid_dry.K_gpa) * 1e9,
                phi=phi,
                alpha=float(a),
                rho_bulk_kg_m3=rho_bulk,
            )
            vp_wet[i], vs_wet[i] = _predict_elastic_vp_vs(
                backend=backend,
                matrix_K_pa=K_m,
                matrix_G_pa=G_m,
                fluid_K_pa=float(fluid_wet.K_gpa) * 1e9,
                phi=phi,
                alpha=float(a),
                rho_bulk_kg_m3=rho_bulk,
            )
        pred[(stage, "vp_dry")] = vp_dry
        pred[(stage, "vs_dry")] = vs_dry
        pred[(stage, "vp_wet")] = vp_wet
        pred[(stage, "vs_wet")] = vs_wet

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.0), constrained_layout=True, sharex=True)
        ax_tc, ax_R, ax_vp, ax_vs = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # style: stage colors
        stage_color = {"before": "C0", "after": "C3"}
        ls_state = {"dry": "-", "wet": "--"}

        for stage in stages:
            sub_stage = rows[rows["stage"] == stage]
            if sub_stage.empty:
                continue
            phi_pct = float(sub_stage.iloc[0]["phi_pct"])
            c = stage_color[stage]

            ax_tc.plot(alphas, pred[(stage, "tc_dry")], color=c, lw=2.2, ls=ls_state["dry"], label=rf"{stage} TC dry, $\phi={phi_pct:.1f}\%$")
            ax_tc.plot(alphas, pred[(stage, "tc_wet")], color=c, lw=2.2, ls=ls_state["wet"], label=rf"{stage} TC wet, $\phi={phi_pct:.1f}\%$")

            # measured TC (horizontal lines)
            for fs, ls in [("dry", ":"), ("wet", "-.")]:
                sub = sub_stage[sub_stage["fluid_state"] == fs]
                if not sub.empty and pd.notna(sub.iloc[0]["tc_w_mk"]):
                    ax_tc.axhline(float(sub.iloc[0]["tc_w_mk"]), color=c, lw=1.1, ls=ls, alpha=0.45)

            # resistivity wet only
            ax_R.plot(alphas, pred[(stage, "R_wet")], color=c, lw=2.2, label=rf"{stage} $R$ wet, $\phi={phi_pct:.1f}\%$")
            sub_wet = sub_stage[sub_stage["fluid_state"] == "wet"]
            if not sub_wet.empty and pd.notna(sub_wet.iloc[0]["resistivity_ohm_m"]):
                ax_R.axhline(float(sub_wet.iloc[0]["resistivity_ohm_m"]), color=c, lw=1.1, ls=":", alpha=0.45)

            # velocities
            ax_vp.plot(alphas, pred[(stage, "vp_dry")] / 1e3, color=c, lw=2.2, ls=ls_state["dry"], label=rf"{stage} $V_P$ dry")
            ax_vp.plot(alphas, pred[(stage, "vp_wet")] / 1e3, color=c, lw=2.2, ls=ls_state["wet"], label=rf"{stage} $V_P$ wet")

            for fs, ls in [("dry", ":"), ("wet", "-.")]:
                sub = sub_stage[sub_stage["fluid_state"] == fs]
                if not sub.empty and pd.notna(sub.iloc[0]["vp_m_s"]):
                    ax_vp.axhline(float(sub.iloc[0]["vp_m_s"]) / 1e3, color=c, lw=1.1, ls=ls, alpha=0.45)

            ax_vs.plot(alphas, pred[(stage, "vs_dry")] / 1e3, color=c, lw=2.2, ls=ls_state["dry"], label=rf"{stage} $V_S$ dry")
            ax_vs.plot(alphas, pred[(stage, "vs_wet")] / 1e3, color=c, lw=2.2, ls=ls_state["wet"], label=rf"{stage} $V_S$ wet")

            for fs, ls in [("dry", ":"), ("wet", "-.")]:
                sub = sub_stage[sub_stage["fluid_state"] == fs]
                if not sub.empty and pd.notna(sub.iloc[0]["vs_m_s"]):
                    ax_vs.axhline(float(sub.iloc[0]["vs_m_s"]) / 1e3, color=c, lw=1.1, ls=ls, alpha=0.45)

        for ax in (ax_tc, ax_R, ax_vp, ax_vs):
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.25)

        ax_tc.set_title("Thermal conductivity vs aspect ratio")
        ax_tc.set_ylabel(r"$\lambda_{eff}$ (W/m/K)")
        ax_tc.legend(frameon=False, ncol=2, loc="best")

        ax_R.set_title("Resistivity (wet only) vs aspect ratio")
        ax_R.set_ylabel(r"$R_{eff}$ ($\Omega\cdot$m)")
        if r_scale == "log":
            ax_R.set_yscale("log")
        elif r_scale == "linear":
            ax_R.set_yscale("linear")
        else:
            raise ValueError("r_scale must be 'log' or 'linear'")

        # y-limits: either user-provided or auto based on model+measurements
        if r_ylim_min is not None or r_ylim_max is not None:
            ax_R.set_ylim(bottom=r_ylim_min, top=r_ylim_max)
        else:
            ys: list[float] = []
            for stage in stages:
                if (stage, "R_wet") in pred:
                    ys.extend([float(x) for x in pred[(stage, "R_wet")] if np.isfinite(x)])
                sub_stage = rows[rows["stage"] == stage]
                if not sub_stage.empty:
                    sub_wet = sub_stage[sub_stage["fluid_state"] == "wet"]
                    if not sub_wet.empty and pd.notna(sub_wet.iloc[0]["resistivity_ohm_m"]):
                        ys.append(float(sub_wet.iloc[0]["resistivity_ohm_m"]))
            if ys:
                y_min = float(min(ys))
                y_max = float(max(ys))
                if r_scale == "log":
                    ax_R.set_ylim(bottom=max(y_min / 1.8, 1e-12), top=y_max * 1.8)
                else:
                    pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.5
                    ax_R.set_ylim(bottom=max(y_min - pad, 0.0), top=y_max + pad)
        ax_R.legend(frameon=False, loc="best")

        ax_vp.set_title("P-wave velocity vs aspect ratio")
        ax_vp.set_ylabel(r"$V_P$ (km/s)")
        ax_vp.set_xlabel(r"Aspect ratio $\alpha$")
        ax_vp.legend(frameon=False, ncol=2, loc="best")

        ax_vs.set_title("S-wave velocity vs aspect ratio")
        ax_vs.set_ylabel(r"$V_S$ (km/s)")
        ax_vs.set_xlabel(r"Aspect ratio $\alpha$")
        ax_vs.legend(frameon=False, ncol=2, loc="best")

        sigma_tag = (
            "UES(wet): solid host (pore inclusions)"
            if sigma_comparison == "matrix"
            else f"UES(wet): brine host (solid inclusions, {sigma_solid_aspect_map})"
        )
        fig.suptitle(rf"Forward sanity (OSP/GSA M1 sweep), sample {sample_id:g} | {sigma_tag}")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward sanity plots (OSP/GSA, M1 sweep) for selected samples.")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet", type=str, default="measurements_long")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    ap.add_argument("--samples", type=str, default="11.2,20.0", help="Comma-separated lab_sample_id list.")

    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=61)

    # matrix (HS feasible set p50)
    ap.add_argument("--matrix-lambda", type=float, default=2.85)
    ap.add_argument("--matrix-vp-m-s", type=float, default=6282.0)
    ap.add_argument("--matrix-vs-m-s", type=float, default=3531.0)
    ap.add_argument("--matrix-rho-kg-m3", type=float, default=2720.0)
    ap.add_argument("--matrix-sigma", type=float, default=1.02e-6)

    # fluids for TC/elastic
    ap.add_argument("--air-lambda", type=float, default=0.03)
    ap.add_argument("--air-K-gpa", type=float, default=0.0001)
    ap.add_argument("--air-rho-kg-m3", type=float, default=1.2)

    ap.add_argument("--brine-lambda", type=float, default=0.60)
    ap.add_argument("--brine-K-gpa", type=float, default=2.20)
    ap.add_argument("--brine-rho-kg-m3", type=float, default=1030.0)

    # brine electrical before/after (20 g/L NaCl)
    ap.add_argument("--brine-sigma-before", type=float, default=3.1250)
    ap.add_argument("--brine-sigma-after", type=float, default=3.8462)
    ap.add_argument(
        "--sigma-comparison",
        choices=("matrix", "fluid_host"),
        default="fluid_host",
        help=(
            "Electrical transport topology for UES/resistivity (wet only): "
            "'matrix' = solid host, pore α affects fluid inclusions (isolated pores); "
            "'fluid_host' = brine host, α applied to solid inclusions (connected brine network proxy)."
        ),
    )
    ap.add_argument(
        "--sigma-solid-aspect-map",
        choices=("direct", "inverse"),
        default="direct",
        help="Only for --sigma-comparison fluid_host: map pore α -> solid inclusion aspect ratio (direct or 1/α).",
    )
    ap.add_argument("--R-scale", choices=("log", "linear"), default="log", help="Y-scale for resistivity subplot.")
    ap.add_argument("--R-ylim-min", type=float, default=None, help="Optional min y-limit for resistivity subplot.")
    ap.add_argument("--R-ylim-max", type=float, default=None, help="Optional max y-limit for resistivity subplot.")

    # elastic backend
    ap.add_argument("--green-fortran", type=Path, default=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"))
    ap.add_argument(
        "--elastic-backend-so",
        type=Path,
        default=Path("test_new/gsa/plots/libgsa_elastic_fortran.so"),
        help="Prebuilt shared library; avoids needing gfortran.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    df = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet))
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce")

    samples = _parse_samples(args.samples)

    # alpha sweep
    alphas = np.logspace(np.log10(float(args.alpha_min)), np.log10(float(args.alpha_max)), int(args.alpha_n))

    matrix = MatrixParams(
        lambda_w_mk=float(args.matrix_lambda),
        vp_m_s=float(args.matrix_vp_m_s),
        vs_m_s=float(args.matrix_vs_m_s),
        rho_kg_m3=float(args.matrix_rho_kg_m3),
        sigma_s_m=float(args.matrix_sigma),
    )
    fluid_dry = FluidParams(
        name="air",
        lambda_w_mk=float(args.air_lambda),
        K_gpa=float(args.air_K_gpa),
        rho_kg_m3=float(args.air_rho_kg_m3),
        sigma_s_m=None,
    )
    fluid_wet = FluidParams(
        name="brine",
        lambda_w_mk=float(args.brine_lambda),
        K_gpa=float(args.brine_K_gpa),
        rho_kg_m3=float(args.brine_rho_kg_m3),
        sigma_s_m=None,
    )

    backend = gsa_elastic.build_backend(
        green_fortran=args.green_fortran,
        output_library=args.elastic_backend_so,
        force_rebuild=False,
    )

    for sid in samples:
        sub = df[df["lab_sample_id"] == float(sid)].copy()
        if sub.empty:
            print(f"WARNING: sample {sid:g} not found in data; skipping.")
            continue
        tag = str(args.sigma_comparison)
        if tag == "fluid_host":
            tag = f"{tag}_{args.sigma_solid_aspect_map}"
        out_png = out_dir / f"forward_sanity_osp_m1_sample_{sid:g}_sigma_{tag}.png"
        _plot_one_sample(
            out_png=out_png,
            sample_id=float(sid),
            rows=sub,
            alphas=alphas,
            matrix=matrix,
            fluid_dry=fluid_dry,
            fluid_wet=fluid_wet,
            brine_sigma_before=float(args.brine_sigma_before),
            brine_sigma_after=float(args.brine_sigma_after),
            sigma_comparison=str(args.sigma_comparison),
            sigma_solid_aspect_map=str(args.sigma_solid_aspect_map),
            r_scale=str(args.R_scale),
            r_ylim_min=(None if args.R_ylim_min is None else float(args.R_ylim_min)),
            r_ylim_max=(None if args.R_ylim_max is None else float(args.R_ylim_max)),
            backend=backend,
        )
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
