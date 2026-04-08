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


@dataclass(frozen=True)
class FluidParams:
    name: str
    lambda_w_mk: float
    K_gpa: float
    rho_kg_m3: float


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


def _rel_l2(errors: list[float]) -> float:
    if not errors:
        return float("nan")
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.nanmean(e * e)))


def _collect_measurements(
    rows: pd.DataFrame,
    *,
    stage: str,
    fluid_state: str,
) -> dict[str, float] | None:
    sub = rows[(rows["stage"] == stage) & (rows["fluid_state"] == fluid_state)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    out: dict[str, float] = {}
    if pd.notna(r.get("tc_w_mk")):
        out["tc_w_mk"] = float(r["tc_w_mk"])
    if pd.notna(r.get("vp_m_s")):
        out["vp_m_s"] = float(r["vp_m_s"])
    if pd.notna(r.get("vs_m_s")):
        out["vs_m_s"] = float(r["vs_m_s"])
    if pd.notna(r.get("bulk_density_g_cm3")):
        out["bulk_density_kg_m3"] = float(r["bulk_density_g_cm3"]) * 1e3
    out["phi"] = _phi_from_row(r)
    return out


def _misfit_curves_for_sample(
    *,
    rows: pd.DataFrame,
    alphas: np.ndarray,
    matrix: MatrixParams,
    fluid_dry: FluidParams,
    fluid_wet: FluidParams,
    backend: gsa_elastic._Backend,  # type: ignore[attr-defined]
) -> dict[str, dict[str, np.ndarray]]:
    K_m, G_m = _matrix_KG_from_vp_vs_rho(matrix.vp_m_s, matrix.vs_m_s, matrix.rho_kg_m3)

    out: dict[str, dict[str, np.ndarray]] = {
        "tc_only": {},
        "tc_vp_vs": {},
    }

    for stage in ("before", "after"):
        # Gather available measurements for both fluid states
        meas = {
            "dry": _collect_measurements(rows, stage=stage, fluid_state="dry"),
            "wet": _collect_measurements(rows, stage=stage, fluid_state="wet"),
        }
        if meas["dry"] is None and meas["wet"] is None:
            continue

        tc_only = np.full_like(alphas, np.nan, dtype=float)
        tc_vp_vs = np.full_like(alphas, np.nan, dtype=float)

        for i, a in enumerate(alphas):
            rel_tc: list[float] = []
            rel_all: list[float] = []

            for state, fl in (("dry", fluid_dry), ("wet", fluid_wet)):
                m = meas[state]
                if m is None:
                    continue

                phi = float(m["phi"])

                # TC
                if "tc_w_mk" in m:
                    pred_tc = _predict_tc_isotropic(matrix.lambda_w_mk, fl.lambda_w_mk, phi, float(a))
                    rel = abs(pred_tc - float(m["tc_w_mk"])) / max(float(m["tc_w_mk"]), 1e-12)
                    rel_tc.append(rel)
                    rel_all.append(rel)

                # Elastic (Vp, Vs)
                if "bulk_density_kg_m3" in m and ("vp_m_s" in m or "vs_m_s" in m):
                    pred_vp, pred_vs = _predict_elastic_vp_vs(
                        backend=backend,
                        matrix_K_pa=float(K_m),
                        matrix_G_pa=float(G_m),
                        fluid_K_pa=float(fl.K_gpa) * 1e9,
                        phi=phi,
                        alpha=float(a),
                        rho_bulk_kg_m3=float(m["bulk_density_kg_m3"]),
                    )

                    if "vp_m_s" in m:
                        rel_all.append(abs(pred_vp - float(m["vp_m_s"])) / max(float(m["vp_m_s"]), 1e-12))
                    if "vs_m_s" in m:
                        rel_all.append(abs(pred_vs - float(m["vs_m_s"])) / max(float(m["vs_m_s"]), 1e-12))

            tc_only[i] = _rel_l2(rel_tc)
            tc_vp_vs[i] = _rel_l2(rel_all)

        out["tc_only"][stage] = tc_only
        out["tc_vp_vs"][stage] = tc_vp_vs

    return out


def _plot_misfit_curves(
    *,
    out_png: Path,
    sample_id: float,
    curves: dict[str, dict[str, np.ndarray]],
    alphas: np.ndarray,
    rows: pd.DataFrame,
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
        "legend.fontsize": 10,
    }

    stage_color = {"before": "C0", "after": "C3"}

    def _best_alpha(y: np.ndarray) -> float | None:
        yy = np.asarray(y, dtype=float)
        if not np.isfinite(yy).any():
            return None
        j = int(np.nanargmin(yy))
        return float(alphas[j])

    # phi summary
    phi_before = rows[rows["stage"] == "before"]
    phi_after = rows[rows["stage"] == "after"]
    phi_b = float(phi_before.iloc[0]["phi_pct"]) if not phi_before.empty and pd.notna(phi_before.iloc[0].get("phi_pct")) else float("nan")
    phi_a = float(phi_after.iloc[0]["phi_pct"]) if not phi_after.empty and pd.notna(phi_after.iloc[0].get("phi_pct")) else float("nan")

    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9), constrained_layout=True, sharex=True)

        for stage in ("before", "after"):
            c = stage_color[stage]
            y1 = curves.get("tc_only", {}).get(stage)
            y2 = curves.get("tc_vp_vs", {}).get(stage)
            if y1 is not None:
                ax1.plot(alphas, 100.0 * y1, color=c, lw=2.4, label=f"{stage}")
                ba = _best_alpha(y1)
                if ba is not None:
                    ax1.axvline(ba, color=c, lw=1.0, ls=":", alpha=0.35)
            if y2 is not None:
                ax2.plot(alphas, 100.0 * y2, color=c, lw=2.4, label=f"{stage}")
                ba = _best_alpha(y2)
                if ba is not None:
                    ax2.axvline(ba, color=c, lw=1.0, ls=":", alpha=0.35)

        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlabel(r"Aspect ratio $\alpha$")
            ax.set_ylabel("Relative misfit (%)")

        ax1.set_title("Objective: TC only (all states)")
        ax1.legend(frameon=False, loc="best")
        ax2.set_title(r"Objective: TC + $V_P$ + $V_S$ (all states)")
        ax2.legend(frameon=False, loc="best")

        fig.suptitle(rf"Pre-sanity misfit vs $\alpha$ (OSP/GSA), sample {sample_id:g} | $\phi$ before={phi_b:.2f}%, after={phi_a:.2f}%")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-sanity: misfit(α) curves for TC-only and TC+Vp+Vs (before+after, dry+wet).")
    ap.add_argument("--data-xlsx", type=Path, default=Path("test_new/data_restructured_for_MT_v2.xlsx"))
    ap.add_argument("--sheet", type=str, default="measurements_long")
    ap.add_argument("--out-dir", type=Path, default=Path("test_new/gsa/plots"))
    ap.add_argument("--samples", type=str, default="22.2,18.2,15.0")

    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--alpha-n", type=int, default=61)

    # matrix (HS feasible set p50)
    ap.add_argument("--matrix-lambda", type=float, default=2.85)
    ap.add_argument("--matrix-vp-m-s", type=float, default=6282.0)
    ap.add_argument("--matrix-vs-m-s", type=float, default=3531.0)
    ap.add_argument("--matrix-rho-kg-m3", type=float, default=2720.0)

    # fluids for TC/elastic
    ap.add_argument("--air-lambda", type=float, default=0.03)
    ap.add_argument("--air-K-gpa", type=float, default=0.0001)
    ap.add_argument("--air-rho-kg-m3", type=float, default=1.2)

    ap.add_argument("--brine-lambda", type=float, default=0.60)
    ap.add_argument("--brine-K-gpa", type=float, default=2.20)
    ap.add_argument("--brine-rho-kg-m3", type=float, default=1030.0)

    # elastic backend
    ap.add_argument("--green-fortran", type=Path, default=Path("src/rockphysx/models/emt/GREEN_ANAL_VTI.f90"))
    ap.add_argument("--elastic-backend-so", type=Path, default=Path("test_new/gsa/plots/libgsa_elastic_fortran.so"))

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    df = pd.read_excel(Path(args.data_xlsx), sheet_name=str(args.sheet))
    df["lab_sample_id"] = pd.to_numeric(df["lab_sample_id"], errors="coerce")

    samples = _parse_samples(args.samples)
    alphas = np.logspace(np.log10(float(args.alpha_min)), np.log10(float(args.alpha_max)), int(args.alpha_n))

    matrix = MatrixParams(
        lambda_w_mk=float(args.matrix_lambda),
        vp_m_s=float(args.matrix_vp_m_s),
        vs_m_s=float(args.matrix_vs_m_s),
        rho_kg_m3=float(args.matrix_rho_kg_m3),
    )
    fluid_dry = FluidParams(
        name="air",
        lambda_w_mk=float(args.air_lambda),
        K_gpa=float(args.air_K_gpa),
        rho_kg_m3=float(args.air_rho_kg_m3),
    )
    fluid_wet = FluidParams(
        name="brine",
        lambda_w_mk=float(args.brine_lambda),
        K_gpa=float(args.brine_K_gpa),
        rho_kg_m3=float(args.brine_rho_kg_m3),
    )

    backend = gsa_elastic.build_backend(
        green_fortran=args.green_fortran,
        output_library=args.elastic_backend_so,
        force_rebuild=False,
    )

    for sid in samples:
        rows = df[df["lab_sample_id"] == float(sid)].copy()
        if rows.empty:
            print(f"WARNING: sample {sid:g} not found in data; skipping.")
            continue

        curves = _misfit_curves_for_sample(
            rows=rows,
            alphas=alphas,
            matrix=matrix,
            fluid_dry=fluid_dry,
            fluid_wet=fluid_wet,
            backend=backend,
        )

        out_png = out_dir / f"pre_sanity_misfit_alpha_tc_elastic_sample_{sid:g}.png"
        _plot_misfit_curves(out_png=out_png, sample_id=float(sid), curves=curves, alphas=alphas, rows=rows)
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

