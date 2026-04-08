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


def _parse_aspect_ratios(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("At least one aspect ratio is required.")
    return out


def _alpha_label(ar: float) -> str:
    ar_f = float(ar)
    if any(np.isclose(ar_f, v) for v in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
        exp = int(round(np.log10(ar_f))) if ar_f > 0 else 0
        return rf"$\alpha=10^{{{exp}}}$" if not np.isclose(ar_f, 1.0) else r"$\alpha=10^{0}$"
    return rf"$\alpha={ar_f:g}$"


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic GSA (aligned VTI): stiffness components vs porosity.")
    ap.add_argument(
        "--aspect-ratios",
        nargs="+",
        default=["1e-4", "1e-3", "1e-2", "1e-1", "1"],
        help="Pore aspect ratios α (space-separated), e.g. 1e-4 1e-3 1e-2 1e-1 1",
    )
    ap.add_argument("--phi-max", type=float, default=0.35, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")

    ap.add_argument("--matrix-K-gpa", type=float, default=76.0, help="Matrix bulk modulus (GPa)")
    ap.add_argument("--matrix-G-gpa", type=float, default=32.0, help="Matrix shear modulus (GPa)")
    ap.add_argument("--pore-K-gpa", type=float, default=0.02, help="Pore/fluid bulk modulus (GPa)")
    ap.add_argument("--green-fortran", type=Path, default=Path("GREEN_ANAL_VTI.f90"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force-rebuild-backend", action="store_true")
    args = ap.parse_args()

    aspect_ratios = _parse_aspect_ratios(list(args.aspect_ratios))
    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))

    Km_pa = float(args.matrix_K_gpa) * 1e9
    Gm_pa = float(args.matrix_G_gpa) * 1e9
    Kp_pa = max(float(args.pore_K_gpa) * 1e9, 1e-9)
    Gp_pa = 1e-9  # fluids/pores

    matrix = gsa.ElasticPhase(K=Km_pa, G=Gm_pa)
    pores = gsa.ElasticPhase(K=Kp_pa, G=Gp_pa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_env(out_dir)

    backend = gsa.build_backend(
        green_fortran=args.green_fortran,
        output_library=out_dir / "libgsa_elastic_fortran.so",
        force_rebuild=bool(args.force_rebuild_backend),
    )

    # curves per alpha
    curves: dict[float, dict[str, np.ndarray]] = {}
    for ar in aspect_ratios:
        C11 = np.full_like(phi, np.nan, dtype=float)
        C33 = np.full_like(phi, np.nan, dtype=float)
        C13 = np.full_like(phi, np.nan, dtype=float)
        C44 = np.full_like(phi, np.nan, dtype=float)
        C66 = np.full_like(phi, np.nan, dtype=float)

        for i, ph in enumerate(phi):
            C_eff_66, _ = gsa.gsa_effective_stiffness_aligned_vti_two_phase(
                phi=float(ph),
                matrix=matrix,
                inclusion=pores,
                pore_aspect_ratio=float(ar),
                backend=backend,
            )
            C = np.asarray(C_eff_66, dtype=float) / 1e9
            C11[i] = float(C[0, 0])
            C33[i] = float(C[2, 2])
            C13[i] = float(C[0, 2])
            C44[i] = float(C[3, 3])
            C66[i] = float(C[5, 5])

        curves[float(ar)] = {"C11": C11, "C33": C33, "C13": C13, "C44": C44, "C66": C66}

    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharex=True, constrained_layout=True)
        axs = axes.ravel()
        keys = ["C11", "C33", "C13", "C44", "C66"]
        titles = [r"$C_{11}$", r"$C_{33}$", r"$C_{13}$", r"$C_{44}$", r"$C_{66}$"]

        for ax, k, t in zip(axs, keys + ["_empty"], titles + [""]):
            if k == "_empty":
                ax.axis("off")
                continue
            for ar in aspect_ratios:
                ax.plot(phi, curves[float(ar)][k], lw=2.2, label=_alpha_label(float(ar)))
            ax.set_title(t)
            ax.set_ylabel("GPa")
            ax.grid(True, alpha=0.25)

        for ax in axes[1, :]:
            ax.set_xlabel("Porosity, φ (fraction)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, -0.02))

        out_png = out_dir / "gsa_elastic_aligned_vti_stiffness_vs_phi.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

