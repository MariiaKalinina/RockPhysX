from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rockphysx.models.emt import gsa_transport


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _gsa_matrix_comparison_multi_phase(
    *,
    matrix_k: float,
    phases: list[tuple[float, float, np.ndarray]],
) -> np.ndarray:
    """
    Closed-form transport GSA for comparison body = matrix.

    Parameters
    ----------
    matrix_k
        Matrix conductivity (scalar), comparison tensor Tc = matrix_k * I.
    phases
        List of (fraction, phase_k, g_tensor) for non-matrix phases.
        Matrix is implicit with fraction = 1 - sum(fractions).

    Returns
    -------
    T_eff : (3,3)
        Effective conductivity tensor.
    """
    Tm = np.eye(3) * float(matrix_k)
    Tc = Tm
    I = np.eye(3)

    phi_incl = float(sum(f for (f, _, _) in phases))
    if phi_incl < 0.0 or phi_incl > 1.0:
        raise ValueError("Sum of inclusion fractions must be in [0,1].")
    f_m = 1.0 - phi_incl

    num = f_m * Tm
    den = f_m * I

    for f, k_i, g in phases:
        f = float(f)
        Ti = np.eye(3) * float(k_i)
        d = Ti - Tc
        M_i = np.linalg.inv(I + g @ d)
        A_i = Ti @ M_i
        num = num + f * A_i
        den = den + f * M_i

    return num @ np.linalg.inv(den)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Thermal conductivity (GSA transport): isotropic calcite matrix with a fixed isotropic pore system "
            "(random spheroids) plus a growing aligned crack system. Plots λ_bedding and λ_normal vs crack porosity."
        )
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out-png", type=Path, default=None)

    ap.add_argument("--matrix-k", type=float, default=3.3, help="Calcite thermal conductivity (W/m/K).")
    ap.add_argument("--oil-k", type=float, default=0.13, help="Oil thermal conductivity (W/m/K).")

    ap.add_argument("--phi-pores", type=float, default=0.08, help="Fixed porosity of isotropic pores (fraction).")
    ap.add_argument("--alpha-pores", type=float, default=0.1, help="Aspect ratio for isotropic pores.")

    ap.add_argument("--phi-cracks-max", type=float, default=0.03, help="Max crack porosity (fraction).")
    ap.add_argument("--phi-cracks-n", type=int, default=61, help="Number of crack porosity points.")
    ap.add_argument("--alpha-cracks", type=float, default=1e-4, help="Aspect ratio for aligned cracks.")

    ap.add_argument("--axis", type=str, default="0,0,1", help="Crack normal axis (x,y,z).")

    ap.add_argument("--n-orientation", type=int, default=90, help="Random ODF samples for pores.")
    ap.add_argument("--n-theta", type=int, default=None, help="Override theta quadrature (else auto).")
    ap.add_argument("--n-phi-ang", type=int, default=None, help="Override phi quadrature (else auto).")
    ap.add_argument("--theta-quadrature", choices=("uniform", "gauss"), default="uniform")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    matrix_k = float(args.matrix_k)
    oil_k = float(args.oil_k)

    phi_pores = float(args.phi_pores)
    if not (0.0 <= phi_pores < 1.0):
        raise ValueError("--phi-pores must be in [0,1).")

    phi_cracks_max = float(args.phi_cracks_max)
    if not (0.0 <= phi_cracks_max < 1.0):
        raise ValueError("--phi-cracks-max must be in [0,1).")

    if phi_pores + phi_cracks_max >= 1.0:
        raise ValueError("phi_pores + phi_cracks_max must be < 1.")

    phi_cracks = np.linspace(0.0, phi_cracks_max, int(args.phi_cracks_n))

    axis = tuple(float(x.strip()) for x in str(args.axis).split(","))
    if len(axis) != 3 or not np.isfinite(axis).all():  # type: ignore[attr-defined]
        raise ValueError("--axis must be 'x,y,z' numeric.")
    axis = (float(axis[0]), float(axis[1]), float(axis[2]))

    alpha_pores = float(args.alpha_pores)
    alpha_cracks = float(args.alpha_cracks)
    if alpha_pores <= 0 or alpha_cracks <= 0:
        raise ValueError("Aspect ratios must be positive.")

    n_theta = args.n_theta if args.n_theta is None else int(args.n_theta)
    n_phi_ang = args.n_phi_ang if args.n_phi_ang is None else int(args.n_phi_ang)
    theta_quad = str(args.theta_quadrature)

    Tc = np.eye(3) * matrix_k

    # Pores: random orientation -> isotropic effect, but we compute orientation-averaged g tensor once.
    odf_random = gsa_transport.OrientationDistribution("random")
    shape_pores = gsa_transport.Shape.sphere() if np.isclose(alpha_pores, 1.0) else gsa_transport.Shape.spheroid(alpha_pores)
    g_pores = gsa_transport.orientation_averaged_g_tensor(
        Tc,
        shape_pores,
        odf_random,
        n_orientation=int(args.n_orientation),
        n_theta=n_theta,
        n_phi=n_phi_ang,
        theta_quadrature=theta_quad,  # type: ignore[arg-type]
    )

    # Cracks: aligned with given axis (crack normal).
    shape_cracks = gsa_transport.Shape.sphere() if np.isclose(alpha_cracks, 1.0) else gsa_transport.Shape.spheroid(alpha_cracks)
    g_cracks = gsa_transport.g_tensor_transport_numeric(
        Tc,
        shape_cracks,
        orientation_axis=axis,
        n_theta=n_theta,
        n_phi=n_phi_ang,
        theta_quadrature=theta_quad,  # type: ignore[arg-type]
    )

    lam_normal = np.full_like(phi_cracks, np.nan, dtype=float)
    lam_bedding = np.full_like(phi_cracks, np.nan, dtype=float)
    ratio = np.full_like(phi_cracks, np.nan, dtype=float)

    for i, phi_c in enumerate(phi_cracks):
        # phases list: (fraction, k, g)
        # both pores and cracks are oil-filled here.
        phases = [
            (phi_pores, oil_k, g_pores),
            (float(phi_c), oil_k, g_cracks),
        ]
        T = _gsa_matrix_comparison_multi_phase(matrix_k=matrix_k, phases=phases)
        ln = float(T[2, 2])
        lb = float(0.5 * (T[0, 0] + T[1, 1]))
        lam_normal[i] = ln
        lam_bedding[i] = lb
        ratio[i] = lb / ln if ln > 0 else np.nan

    # Baseline (no cracks)
    T0 = _gsa_matrix_comparison_multi_phase(
        matrix_k=matrix_k,
        phases=[(phi_pores, oil_k, g_pores)],
    )
    ln0 = float(T0[2, 2])
    lb0 = float(0.5 * (T0[0, 0] + T0[1, 1]))

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

    out_png = args.out_png
    if out_png is None:
        out_png = out_dir / "gsa_tc_fracture_growth_fixed_pores_calcite_oil.png"
    out_png = Path(out_png)

    x = phi_cracks * 100.0
    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)

        ax1.plot(x, lam_normal, color="C0", lw=2.8, label=r"$\lambda_{normal}$")
        ax1.plot(x, lam_bedding, color="C1", lw=2.8, ls="--", label=r"$\lambda_{bedding}$")
        ax1.axhline(ln0, color="C0", lw=1.4, alpha=0.35)
        ax1.axhline(lb0, color="C1", lw=1.4, ls="--", alpha=0.35)
        ax1.set_xlabel(r"Crack porosity, $\phi_{cracks}$ (%)")
        ax1.set_ylabel(r"Thermal conductivity (W m$^{-1}$ K$^{-1}$)")
        ax1.set_title(r"(a) Components vs crack porosity")
        ax1.grid(True, alpha=0.35)
        ax1.legend(frameon=False, loc="best")

        ax2.plot(x, ratio, color="C2", lw=2.8)
        ax2.set_xlabel(r"Crack porosity, $\phi_{cracks}$ (%)")
        ax2.set_ylabel(r"$\lambda_{bedding}/\lambda_{normal}$")
        ax2.set_title(r"(b) Anisotropy ratio")
        ax2.grid(True, alpha=0.35)

        fig.suptitle(
            "Fracture growth sensitivity (GSA transport): calcite matrix, oil-filled pores + aligned oil-filled cracks\n"
            + rf"Fixed pores: $\phi_{{pores}}={phi_pores:.2f}$, $\alpha_{{pores}}={alpha_pores:g}$ (random); "
            + rf"Cracks: $\alpha_{{cracks}}={alpha_cracks:g}$ (aligned)"
        )

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

