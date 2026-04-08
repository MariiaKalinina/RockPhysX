from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _Inclusion:
    K_gpa: float
    G_gpa: float
    label: str


def _solve_moduli(
    *,
    method: str,
    Km: float,
    Gm: float,
    phi: float,
    alpha: float,
    inclusion: _Inclusion,
    warm: tuple[float, float] | None,
    backend_osp=None,
) -> tuple[float, float]:
    method_u = method.strip().upper()
    if method_u == "MT":
        from rockphysx.models.emt.mt_elastic import mt_elastic_pores

        return mt_elastic_pores(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=inclusion.K_gpa)

    if method_u == "KT":
        from rockphysx.models.emt.kt_elastic import kt_elastic_pores

        return kt_elastic_pores(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=inclusion.K_gpa)

    if method_u == "SC":
        from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores

        return berryman_self_consistent_spheroidal_pores(
            matrix_bulk_gpa=Km,
            matrix_shear_gpa=Gm,
            porosity=phi,
            pore_bulk_gpa=inclusion.K_gpa,
            aspect_ratio=alpha,
            relaxation=0.6 if alpha < 0.2 else 1.0,
            max_iter=4000,
            initial_guess_gpa=warm,
        )

    if method_u == "DEM":
        from rockphysx.models.emt.dem_transport_elastic import dem_elastic_moduli

        r = dem_elastic_moduli(
            matrix_bulk=Km,
            matrix_shear=Gm,
            inclusion_bulk=inclusion.K_gpa,
            inclusion_shear=inclusion.G_gpa,
            inclusion_fraction=phi,
            aspect_ratio=alpha,
            n_steps=800,
        )
        return float(r.bulk_modulus), float(r.shear_modulus)

    if method_u == "OSP":
        from rockphysx.models.emt.gsa_elastic_random_isotropic import ElasticPhase, gsa_elastic_random_spheroidal_pores

        if backend_osp is None:
            raise RuntimeError("OSP backend not initialized.")

        matrix = ElasticPhase(K=Km * 1e9, G=Gm * 1e9)
        # Avoid exact zeros (singular matrices) in the elastic kernel.
        inc_K = max(float(inclusion.K_gpa), 1e-6) * 1e9
        inc_G = max(float(inclusion.G_gpa), 1e-6) * 1e9
        inclusion_phase = ElasticPhase(K=inc_K, G=inc_G)

        r = gsa_elastic_random_spheroidal_pores(
            phi=phi,
            a_ratio=alpha,
            matrix=matrix,
            inclusion=inclusion_phase,
            rho_matrix_kg_m3=2700.0,
            rho_inclusion_kg_m3=1000.0,
            backend=backend_osp,
            comparison_body="matrix",
        )
        return float(r.K_eff / 1e9), float(r.G_eff / 1e9)

    raise ValueError(f"Unknown method {method!r}. Use MT, KT, SC, DEM, or OSP.")


def _compute_curves(
    *,
    methods: list[str],
    Km: float,
    Gm: float,
    phi: float,
    alphas: np.ndarray,
    inclusion: _Inclusion,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    backend_osp = None
    if any(m.strip().upper() == "OSP" for m in methods):
        from rockphysx.models.emt.gsa_elastic_random_isotropic import build_backend

        backend_osp = build_backend()

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        K = np.empty_like(alphas, dtype=float)
        G = np.empty_like(alphas, dtype=float)
        warm: tuple[float, float] | None = (Km, Gm)
        for i, a in enumerate(alphas):
            try:
                k_i, g_i = _solve_moduli(
                    method=method,
                    Km=Km,
                    Gm=Gm,
                    phi=phi,
                    alpha=float(a),
                    inclusion=inclusion,
                    warm=warm,
                    backend_osp=backend_osp,
                )
            except Exception:
                K[i:] = np.nan
                G[i:] = np.nan
                break
            K[i] = float(k_i)
            G[i] = float(g_i)
            warm = (float(k_i), float(g_i))
        out[method.strip().upper()] = (K, G)
    return out


def _plot(
    *,
    out_path: Path,
    title: str,
    Km: float,
    Gm: float,
    phi: float,
    alphas: np.ndarray,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }

    # Model colors (kept stable across plots).
    model_colors = {
        "MT": "#1f77b4",   # blue
        "KT": "#d62728",   # red
        "SC": "#2ca02c",   # green
        "DEM": "#9467bd",  # purple
        "OSP": "#ff7f0e",  # orange
    }
    model_labels = {
        "MT": "Mori-Tanaka",
        "KT": "Kuster-Toksöz",
        "SC": "Self-Consistent",
        "DEM": "DEM",
        "OSP": "OSP (GSA)",
    }

    with plt.rc_context(rc):
        fig, (axK, axG) = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True)

        for method_u, (K, G) in curves.items():
            c = model_colors.get(method_u, "#333333")
            lab = model_labels.get(method_u, method_u)
            axK.plot(alphas, K, lw=2.2, color=c, label=lab)
            axG.plot(alphas, G, lw=2.2, color=c, label=lab)

        axK.set_xscale("log")
        axG.set_xscale("log")
        axK.set_ylabel("Bulk modulus (GPa)")
        axG.set_ylabel("Shear modulus (GPa)")
        axG.set_xlabel("Aspect ratio $\\alpha$ (log scale)")

        axK.grid(True, which="both", alpha=0.35)
        axG.grid(True, which="both", alpha=0.35)
        axK.legend(loc="best", frameon=True, ncol=1)

        fig.suptitle(title, fontsize=14)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic sensitivity vs aspect ratio for fixed porosity (dry or wet).")
    ap.add_argument("--Km", type=float, default=77.78, help="Matrix bulk modulus (GPa).")
    ap.add_argument("--Gm", type=float, default=31.82, help="Matrix shear modulus (GPa).")
    ap.add_argument("--phi", type=float, default=0.10, help="Fixed porosity (fraction).")
    ap.add_argument("--alpha-min", type=float, default=1e-4, help="Min aspect ratio.")
    ap.add_argument("--alpha-max", type=float, default=1.001, help="Max aspect ratio.")
    ap.add_argument("--n-alpha", type=int, default=61, help="Number of alpha points (logspace).")
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["MT", "KT", "SC", "DEM", "OSP"],
        help="Methods to plot.",
    )
    ap.add_argument("--out-dry", type=Path, default=Path("test_new/emt/plots/elastic_vs_alpha_phi10_dry.png"))
    ap.add_argument("--out-wet", type=Path, default=Path("test_new/emt/plots/elastic_vs_alpha_phi10_water.png"))
    ap.add_argument("--water-K-gpa", type=float, default=2.2, help="Water bulk modulus (GPa) for wet plot.")
    args = ap.parse_args()

    Km = float(args.Km)
    Gm = float(args.Gm)
    phi = float(args.phi)

    a_min = float(args.alpha_min)
    a_max = float(args.alpha_max)
    if a_min <= 0 or a_max <= 0:
        raise ValueError("alpha-min and alpha-max must be positive.")
    if a_min >= a_max:
        raise ValueError("alpha-min must be < alpha-max.")

    alphas = np.logspace(np.log10(a_min), np.log10(a_max), int(args.n_alpha))

    methods = [m.strip().upper() for m in args.methods]

    # Dry: pores ~ vacuum (very small but invertible stiffness).
    dry_inc = _Inclusion(K_gpa=0.0, G_gpa=0.0, label="dry")
    curves_dry = _compute_curves(methods=methods, Km=Km, Gm=Gm, phi=phi, alphas=alphas, inclusion=dry_inc)
    _plot(
        out_path=Path(args.out_dry),
        title=rf"Elastic sensitivity vs aspect ratio (dry), $\phi={phi*100:.0f}\%$",
        Km=Km,
        Gm=Gm,
        phi=phi,
        alphas=alphas,
        curves=curves_dry,
    )

    # Wet: fluid-filled pores (water), shear ~ 0.
    wet_inc = _Inclusion(K_gpa=float(args.water_K_gpa), G_gpa=0.0, label="water")
    curves_wet = _compute_curves(methods=methods, Km=Km, Gm=Gm, phi=phi, alphas=alphas, inclusion=wet_inc)
    _plot(
        out_path=Path(args.out_wet),
        title=rf"Elastic sensitivity vs aspect ratio (water-saturated), $\phi={phi*100:.0f}\%$",
        Km=Km,
        Gm=Gm,
        phi=phi,
        alphas=alphas,
        curves=curves_wet,
    )

    print(f"Saved: {args.out_dry}")
    print(f"Saved: {args.out_wet}")


if __name__ == "__main__":
    main()

