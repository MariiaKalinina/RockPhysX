from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _matrix_moduli_from_vp_vs_rho(vp_kms: float, vs_kms: float, rho_gcc: float) -> tuple[float, float]:
    vp = float(vp_kms) * 1e3
    vs = float(vs_kms) * 1e3
    rho = float(rho_gcc) * 1e3
    K = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
    G = rho * (vs * vs)
    return float(K), float(G)

def _vr_bounds(phi: np.ndarray, M1: float, M2: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Voigt and Reuss bounds for a scalar modulus.
    Phase 1 fraction is (1-phi), phase 2 fraction is phi.
    """
    phi = np.asarray(phi, dtype=float)
    f2 = phi
    f1 = 1.0 - phi
    M1 = float(M1)
    M2 = float(M2)
    Mv = f1 * M1 + f2 * M2
    # Reuss: guard zeros
    denom = np.zeros_like(phi, dtype=float)
    if M1 > 0:
        denom += f1 / M1
    else:
        denom += np.inf
    if M2 > 0:
        denom += f2 / M2
    else:
        denom += np.inf
    Mr = np.where(np.isfinite(denom) & (denom > 0), 1.0 / denom, 0.0)
    return Mv, Mr


def _hs_bounds_elastic(
    phi: np.ndarray,
    *,
    Km: float,
    Gm: float,
    Kf: float,
    Gf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hashin–Shtrikman bounds for isotropic two-phase elastic moduli (K, G).
    Matrix is assumed to be the stiffer phase (typical rock + fluid/pores).
    """
    phi = np.asarray(phi, dtype=float)
    f1 = 1.0 - phi  # matrix
    f2 = phi        # fluid/pores

    K1, G1 = float(Km), float(Gm)
    K2, G2 = float(Kf), float(Gf)

    # Bulk modulus bounds (HS+ / HS-)
    #
    # Assume phase 1 (matrix) is the stiffer phase and phase 2 (fluid/pores) the softer.
    # HS+ corresponds to stiff host, HS- to soft host.
    Kp = np.full_like(phi, np.nan, dtype=float)
    Km_ = np.full_like(phi, np.nan, dtype=float)
    if not np.isclose(K2, K1):
        denomKp = (1.0 / (K2 - K1)) + (f1 / (K1 + 4.0 * G1 / 3.0))
        Kp = K1 + f2 / denomKp
        denomKm = (1.0 / (K1 - K2)) + (f2 / (K2 + 4.0 * G2 / 3.0))
        Km_ = K2 + f1 / denomKm
    else:
        Kp[:] = K1
        Km_[:] = K1

    # Shear helper zeta
    def zeta(K: float, G: float) -> float:
        K = float(K)
        G = float(G)
        denom = 6.0 * (K + 2.0 * G)
        if denom <= 0.0 or not np.isfinite(denom):
            return np.inf
        return float(G * (9.0 * K + 8.0 * G) / denom)

    z1 = zeta(K1, G1)
    z2 = zeta(K2, G2) if G2 > 0 else 0.0

    # Shear modulus bounds (HS+ / HS-)
    Gp = np.full_like(phi, np.nan, dtype=float)
    Gm_ = np.full_like(phi, np.nan, dtype=float)
    if not np.isclose(G2, G1):
        denomGp = (1.0 / (G2 - G1)) + (f1 / (G1 + z1))
        Gp = G1 + f2 / denomGp
        # For fluids/pores, G2=0 and z2=0 => HS- tends to 0 (soft host connected).
        denomGm = (1.0 / (G1 - G2)) + (f2 / (G2 + z2) if (G2 + z2) > 0 else np.inf)
        Gm_ = G2 + f1 / denomGm
    else:
        Gp[:] = G1
        Gm_[:] = G1

    return Kp, Km_, Gp, Gm_


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Elastic GSA (isotropic random): Vp, Vs vs porosity for a fixed aspect ratio, gas vs water."
    )
    ap.add_argument("--alpha-fixed", type=float, default=1e-2, help="Fixed aspect ratio α, e.g. 1e-2")
    ap.add_argument("--phi-max", type=float, default=1.0, help="Maximum porosity (fraction), use 1.0 for 100%")
    ap.add_argument("--n-phi", type=int, default=201, help="Number of porosity points")
    ap.add_argument("--phi-units", choices=("fraction", "percent"), default="percent")

    ap.add_argument("--matrix-vp-kms", type=float, default=5.8, help="Matrix VP (km/s)")
    ap.add_argument("--matrix-vs-kms", type=float, default=3.2, help="Matrix VS (km/s)")
    ap.add_argument("--matrix-rho-gcc", type=float, default=2.71, help="Matrix density (g/cm^3)")

    ap.add_argument("--gas-K-gpa", type=float, default=0.02, help="Gas bulk modulus (GPa)")
    ap.add_argument("--gas-rho-gcc", type=float, default=0.0012, help="Gas density (g/cm^3)")
    ap.add_argument("--water-K-gpa", type=float, default=2.2, help="Water bulk modulus (GPa)")
    ap.add_argument("--water-rho-gcc", type=float, default=1.0, help="Water density (g/cm^3)")

    ap.add_argument("--comparison-body", choices=("matrix", "bayuk_linear_mix"), default="matrix")
    ap.add_argument("--k-connectivity", type=float, default=3.0, help="k for comparison_body='bayuk_linear_mix'.")
    ap.add_argument("--sign", choices=("+", "-"), default="-", help="Operator sign in A = (I ± G0·ΔC)^-1.")

    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    ap.add_argument(
        "--add-bounds",
        action="store_true",
        help="Also plot Voigt–Reuss and Hashin–Shtrikman bounds for K and G.",
    )
    args = ap.parse_args()

    alpha = float(args.alpha_fixed)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("--alpha-fixed must be positive and finite.")

    phi_max = float(args.phi_max)
    if not np.isfinite(phi_max) or phi_max <= 0.0:
        raise ValueError("--phi-max must be positive and finite.")
    if phi_max > 1.0:
        raise ValueError("--phi-max must be <= 1.0 (100%).")
    phi_end = float(np.nextafter(1.0, 0.0)) if np.isclose(phi_max, 1.0) else phi_max
    phi = np.linspace(0.0, phi_end, int(args.n_phi))
    phi_x = phi * 100.0 if args.phi_units == "percent" else phi
    xlab = "Porosity, φ (%)" if args.phi_units == "percent" else "Porosity, φ (fraction)"

    Km_pa, Gm_pa = _matrix_moduli_from_vp_vs_rho(args.matrix_vp_kms, args.matrix_vs_kms, args.matrix_rho_gcc)
    rho_m = float(args.matrix_rho_gcc) * 1e3
    matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)

    fluids = {
        "Water": {"K": float(args.water_K_gpa) * 1e9, "rho": float(args.water_rho_gcc) * 1e3, "color": "C0"},
        "Gas": {"K": float(args.gas_K_gpa) * 1e9, "rho": float(args.gas_rho_gcc) * 1e3, "color": "C3"},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )
    sign = (-1 if args.sign == "-" else +1)

    series: dict[str, dict[str, np.ndarray]] = {}
    for fluid_name, fd in fluids.items():
        pores = gsa.ElasticPhase(K=float(fd["K"]), G=1e-9)
        rho_f = float(fd["rho"])
        vp = np.full_like(phi, np.nan, dtype=float)
        vs = np.full_like(phi, np.nan, dtype=float)
        K_eff_gpa = np.full_like(phi, np.nan, dtype=float)
        G_eff_gpa = np.full_like(phi, np.nan, dtype=float)
        for i, ph in enumerate(phi):
            _, _, K_eff, G_eff = gsa.gsa_effective_stiffness_random_two_phase(
                phi=float(ph),
                matrix=matrix,
                inclusion=pores,
                pore_aspect_ratio=float(alpha),
                backend=backend,
                comparison_body=str(args.comparison_body),
                k_connectivity=(float(args.k_connectivity) if str(args.comparison_body) == "bayuk_linear_mix" else None),
                sign=sign,
            )
            rho_eff = (1.0 - float(ph)) * rho_m + float(ph) * rho_f
            K_eff_gpa[i] = float(K_eff / 1e9)
            G_eff_gpa[i] = float(G_eff / 1e9)
            vp[i] = float(np.sqrt(max(K_eff + 4.0 * G_eff / 3.0, 0.0) / rho_eff) / 1e3)
            vs[i] = float(np.sqrt(max(G_eff, 0.0) / rho_eff) / 1e3)
        series[fluid_name] = {
            "vp": vp,
            "vs": vs,
            "K_eff_gpa": K_eff_gpa,
            "G_eff_gpa": G_eff_gpa,
            "K_f_gpa": float(fd["K"]) / 1e9,
            "G_f_gpa": 0.0,
            "color": str(fd["color"]),
        }

    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }

    with plt.rc_context(rc):
        if args.add_bounds:
            fig, (axv, axk, axg) = plt.subplots(
                3,
                1,
                figsize=(7.8, 9.2),
                sharex=True,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
            )
        else:
            fig, axv = plt.subplots(1, 1, figsize=(7.8, 4.8), constrained_layout=True)
            axk = None
            axg = None

        # Velocities
        for fluid_name, d in series.items():
            color = d["color"]
            axv.plot(phi_x, d["vp"], color=color, lw=2.4, ls="-", label=f"{fluid_name} $V_P$")
            axv.plot(phi_x, d["vs"], color=color, lw=2.4, ls="--", label=f"{fluid_name} $V_S$")

        axv.set_ylabel("km/s")
        axv.grid(True, alpha=0.25)
        axv.set_title(rf"Isotropic random GSA, $\alpha={alpha:g}$")
        axv.legend(frameon=False, ncol=2, loc="best")

        # Bounds in moduli space (K, G)
        if args.add_bounds and axk is not None and axg is not None:
            Km_gpa = Km_pa / 1e9
            Gm_gpa = Gm_pa / 1e9

            for fluid_name, d in series.items():
                color = d["color"]
                Kf_gpa = float(d["K_f_gpa"])
                Gf_gpa = 0.0

                Kv, Kr = _vr_bounds(phi, Km_gpa, Kf_gpa)
                Gv, Gr = _vr_bounds(phi, Gm_gpa, Gf_gpa)
                Khs_u, Khs_l, Ghs_u, Ghs_l = _hs_bounds_elastic(phi, Km=Km_gpa, Gm=Gm_gpa, Kf=Kf_gpa, Gf=Gf_gpa)
                Khs_low = np.minimum(Khs_l, Khs_u)
                Khs_high = np.maximum(Khs_l, Khs_u)
                Ghs_low = np.minimum(Ghs_l, Ghs_u)
                Ghs_high = np.maximum(Ghs_l, Ghs_u)

                # K panel
                axk.plot(phi_x, Khs_high, color=color, lw=1.2, ls="-.", alpha=0.9, label=f"{fluid_name} HS upper")
                axk.plot(phi_x, Khs_low, color=color, lw=1.2, ls="-.", alpha=0.5, label=f"{fluid_name} HS lower")
                axk.plot(phi_x, Kv, color=color, lw=1.4, ls=":", alpha=0.9, label=f"{fluid_name} Voigt")
                axk.plot(phi_x, Kr, color=color, lw=1.4, ls="--", alpha=0.9, label=f"{fluid_name} Reuss")
                axk.plot(phi_x, d["K_eff_gpa"], color=color, lw=2.2, ls="-", label=f"{fluid_name} GSA")

                # G panel
                axg.plot(phi_x, Ghs_high, color=color, lw=1.2, ls="-.", alpha=0.9, label=f"{fluid_name} HS upper")
                axg.plot(phi_x, Ghs_low, color=color, lw=1.2, ls="-.", alpha=0.5, label=f"{fluid_name} HS lower")
                axg.plot(phi_x, Gv, color=color, lw=1.4, ls=":", alpha=0.9, label=f"{fluid_name} Voigt")
                axg.plot(phi_x, Gr, color=color, lw=1.4, ls="--", alpha=0.9, label=f"{fluid_name} Reuss")
                axg.plot(phi_x, d["G_eff_gpa"], color=color, lw=2.2, ls="-", label=f"{fluid_name} GSA")

            axk.set_ylabel("GPa")
            axk.set_title("Bulk modulus $K$: GSA vs Voigt–Reuss and HS bounds")
            axk.grid(True, alpha=0.25)
            axk.legend(frameon=False, ncol=2, loc="best")

            axg.set_ylabel("GPa")
            axg.set_title("Shear modulus $G$: GSA vs Voigt–Reuss and HS bounds")
            axg.grid(True, alpha=0.25)
            axg.legend(frameon=False, ncol=2, loc="best")

            axg.set_xlabel(xlab)
        else:
            axv.set_xlabel(xlab)

        out_png = out_dir / f"gsa_elastic_random_isotropic_velocities_vs_phi_alpha_{alpha:g}_gas_water.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
