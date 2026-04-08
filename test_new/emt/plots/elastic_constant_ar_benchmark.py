from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _compute_curves(
    *,
    method: str,
    Km: float,
    Gm: float,
    alphas: list[float],
    phis: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    method_u = method.strip().upper()
    if method_u == "MT":
        from rockphysx.models.emt.mt_elastic import mt_elastic_pores

        def solve(phi: float, alpha: float, _warm: tuple[float, float] | None) -> tuple[float, float]:
            return mt_elastic_pores(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=0.0)

    elif method_u == "KT":
        from rockphysx.models.emt.kt_elastic import kt_elastic_pores

        def solve(phi: float, alpha: float, _warm: tuple[float, float] | None) -> tuple[float, float]:
            return kt_elastic_pores(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=0.0)

    elif method_u == "SC":
        from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores

        def solve(phi: float, alpha: float, warm: tuple[float, float] | None) -> tuple[float, float]:
            return berryman_self_consistent_spheroidal_pores(
                matrix_bulk_gpa=Km,
                matrix_shear_gpa=Gm,
                porosity=phi,
                pore_bulk_gpa=0.0,
                aspect_ratio=alpha,
                relaxation=0.6 if alpha < 0.2 else 1.0,
                max_iter=4000,
                initial_guess_gpa=warm,
            )

    elif method_u == "DEM":
        from rockphysx.models.emt.dem_transport_elastic import dem_elastic_moduli

        def solve(phi: float, alpha: float, _warm: tuple[float, float] | None) -> tuple[float, float]:
            r = dem_elastic_moduli(
                matrix_bulk=Km,
                matrix_shear=Gm,
                inclusion_bulk=0.0,
                inclusion_shear=0.0,
                inclusion_fraction=phi,
                aspect_ratio=alpha,
                n_steps=800,
            )
            return float(r.bulk_modulus), float(r.shear_modulus)

    elif method_u == "OSP":
        # OSP / GSA (general singular approximation) implementation used elsewhere in this repo.
        # Notes:
        # - The underlying solver operates in SI (Pa), while this benchmark script uses GPa.
        # - For "dry pores" we cannot pass exact zeros (singular stiffness); we use a tiny value.
        from rockphysx.models.emt.gsa_elastic_random_isotropic import (
            ElasticPhase,
            build_backend,
            gsa_elastic_random_spheroidal_pores,
        )

        backend = build_backend()
        matrix = ElasticPhase(K=Km * 1e9, G=Gm * 1e9)
        inclusion = ElasticPhase(K=1e-6 * 1e9, G=1e-6 * 1e9)  # ~vacuum, but invertible

        def solve(phi: float, alpha: float, _warm: tuple[float, float] | None) -> tuple[float, float]:
            r = gsa_elastic_random_spheroidal_pores(
                phi=phi,
                a_ratio=alpha,
                matrix=matrix,
                inclusion=inclusion,
                rho_matrix_kg_m3=2700.0,
                rho_inclusion_kg_m3=1.0,
                backend=backend,
                comparison_body="matrix",
            )
            return float(r.K_eff / 1e9), float(r.G_eff / 1e9)

    else:
        raise ValueError(f"Unknown method {method!r}. Use MT, KT, SC, DEM, or OSP.")

    K_curves: dict[float, np.ndarray] = {}
    G_curves: dict[float, np.ndarray] = {}
    for a in alphas:
        Ks = np.empty_like(phis, dtype=float)
        Gs = np.empty_like(phis, dtype=float)
        warm: tuple[float, float] | None = (Km, Gm)
        for j, phi in enumerate(phis):
            try:
                k, g = solve(float(phi), float(a), warm)
            except Exception:
                # KT (and sometimes SC) can become non-physical outside its validity range.
                # For a paper-style comparison plot we stop the curve where it fails.
                Ks[j:] = np.nan
                Gs[j:] = np.nan
                break
            Ks[j] = k
            Gs[j] = g
            warm = (float(k), float(g))
        K_curves[float(a)] = Ks
        G_curves[float(a)] = Gs
    return K_curves, G_curves


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-style benchmark for elastic EMT models (constant aspect ratio).")
    ap.add_argument("--Km", type=float, default=77.78, help="Matrix bulk modulus (GPa).")
    ap.add_argument("--Gm", type=float, default=31.82, help="Matrix shear modulus (GPa).")
    ap.add_argument("--phi-max", type=float, default=0.30, help="Max porosity (fraction).")
    ap.add_argument("--n-phi", type=int, default=151, help="Number of porosity points.")
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 1.001],
        help="Aspect ratios (minor/major).",
    )
    ap.add_argument("--out", type=Path, default=Path("test_new/emt/plots/elastic_constant_ar_benchmark.png"))
    ap.add_argument(
        "--include-osp",
        action="store_true",
        help="Also save an extra figure with an additional OSP (GSA) column.",
    )
    args = ap.parse_args()

    Km = float(args.Km)
    Gm = float(args.Gm)
    alphas = [float(a) for a in args.alphas]
    phis = np.linspace(0.0, float(args.phi_max), int(args.n_phi))

    methods_main = ["MT", "KT", "SC", "DEM"]
    methods_with_osp = ["MT", "KT", "SC", "DEM", "OSP"]

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
    # Match the reference palette exactly:
    # α=0.1 blue, 0.2 red, 0.3 green, 0.4 magenta, 0.5 cyan, 1.001 yellow
    color_map = {
        0.1: "#0000ff",   # blue
        0.2: "#ff0000",   # red
        0.3: "#00aa00",   # green
        0.4: "#ff00ff",   # magenta
        0.5: "#00c8ff",   # cyan-ish
        1.001: "#d4c400", # yellow-ish
    }
    colors = [color_map.get(float(a), "#333333") for a in alphas]

    def plot_grid(methods: list[str], out_path: Path) -> None:
        ncol = len(methods)
        fig, axes = plt.subplots(2, ncol, figsize=(3.65 * ncol, 6.2), sharex=True, sharey="row", constrained_layout=True)
        if ncol == 1:
            axes = np.asarray([[axes[0]], [axes[1]]])

        for col, method in enumerate(methods):
            Kc, Gc = _compute_curves(method=method, Km=Km, Gm=Gm, alphas=alphas, phis=phis)

            axK = axes[0, col]
            axG = axes[1, col]
            for i, a in enumerate(alphas):
                lab = rf"$\alpha={a:g}$"
                axK.plot(phis, Kc[float(a)], color=colors[i], lw=2.0, label=lab)
                axG.plot(phis, Gc[float(a)], color=colors[i], lw=2.0, label=lab)

            title = {
                "MT": "Mori-Tanaka",
                "KT": "Kuster-Toksöz",
                "SC": "Self-Consistent",
                "DEM": "Differential Effective Medium",
                "OSP": "OSP (GSA)",
            }[method.strip().upper()]
            axK.set_title(title)
            if col == 0:
                axK.set_ylabel("Bulk modulus (GPa)")
                axG.set_ylabel("Shear modulus (GPa)")
            axG.set_xlabel("Porosity")
            axK.grid(True, alpha=0.35)
            axG.grid(True, alpha=0.35)

            # Match the reference y-ranges.
            axK.set_ylim(30.0, 80.0)
            axG.set_ylim(15.0, 35.0)
            axK.set_yticks(np.arange(30.0, 81.0, 10.0))
            axG.set_yticks(np.arange(15.0, 35.1, 2.5))
            axK.set_xlim(0.0, float(args.phi_max))

            # Legend in every panel (as in the reference).
            axK.legend(loc="upper right", frameon=True, ncol=1)

        fig.suptitle("Elastic EMT benchmark (dry pores, constant aspect ratio)", fontsize=14)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context(rc):
        plot_grid(methods_main, Path(args.out))
        if args.include_osp:
            out2 = Path(args.out).with_name(Path(args.out).stem + "_with_osp.png")
            plot_grid(methods_with_osp, out2)

    print(f"Saved: {args.out}")
    if args.include_osp:
        print(f"Saved: {Path(args.out).with_name(Path(args.out).stem + '_with_osp.png')}")


if __name__ == "__main__":
    main()
