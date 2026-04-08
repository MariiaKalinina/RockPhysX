from __future__ import annotations

"""
Quick validation / sweep for the elastic self-consistent implementation.

This script can be run from the repo root and will:
- compute (Keff, Geff, Vp, Vs) vs porosity for a chosen pore aspect ratio
- optionally compare to the Berryman P/Q self-consistent implementation

Outputs (PNG) are saved next to this file by default.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import numpy as np

import sys

# Make this file runnable from repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# strict_mt_elastic_pores lives in test_new/mori-tanaka
_MT_DIR = REPO_ROOT / "test_new" / "mori-tanaka"
if _MT_DIR.exists() and str(_MT_DIR) not in sys.path:
    sys.path.insert(0, str(_MT_DIR))

from strict_mt_elastic_pores import (
    ArrayLike,
    TensorTools,
    bulk_shear_from_stiffness_tensor,
    eshelby_spheroid_isotropic_matrix,
    isotropic_stiffness_from_KG,
    poisson_from_KG,
)


@dataclass
class SCElasticResult:
    Ceff66: ArrayLike
    Ceff3333: ArrayLike
    K_eff: float
    G_eff: float
    rho_eff_kg_m3: float | None = None
    vp_m_s: float | None = None
    vs_m_s: float | None = None
    n_iter: int | None = None
    converged: bool = False


def _as_array(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _normalize_fractions(x: np.ndarray) -> np.ndarray:
    if np.any(x < 0.0):
        raise ValueError("Volume fractions must be non-negative.")
    s = float(np.sum(x))
    if s <= 0.0:
        raise ValueError("Volume fractions must sum to a positive number.")
    return x / s


def _vrh_initial_guess(K: np.ndarray, G: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    eps = 1e-12
    Kv = float(np.sum(X * K))
    Gv = float(np.sum(X * G))
    Kr = float(1.0 / np.sum(X / np.maximum(K, eps)))
    Gr = float(1.0 / np.sum(X / np.maximum(G, eps)))
    return max(0.5 * (Kv + Kr), eps), max(0.5 * (Gv + Gr), eps)


def _orientation_averaged_A_local(
    C_comp66: ArrayLike,
    C_phase66: ArrayLike,
    aspect_ratio: float,
) -> ArrayLike:
    Kc, Gc = bulk_shear_from_stiffness_tensor(TensorTools.mandel2tensor(C_comp66))
    nu_c = float(poisson_from_KG(Kc, Gc))
    # Keep Eshelby kernel stable even if intermediate iterates become slightly unphysical.
    nu_c = float(np.clip(nu_c, -0.999, 0.499))

    S66 = eshelby_spheroid_isotropic_matrix(
        a_ratio=float(aspect_ratio),
        nu_matrix=float(nu_c),
        return_dim="66",
    )
    I6 = np.eye(6)
    dC = C_phase66 - C_comp66
    try:
        Cc_inv = np.linalg.inv(C_comp66)
    except np.linalg.LinAlgError:
        Cc_inv = np.linalg.pinv(C_comp66, rcond=1e-12)

    M = I6 + TensorTools.tensor_product(S66, TensorTools.tensor_product(Cc_inv, dC))
    try:
        A_dil = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        A_dil = np.linalg.pinv(M, rcond=1e-12)

    weighted_A_local_66 = TensorTools.tensor_product(dC, A_dil)
    weighted_A_local_3333 = TensorTools.mandel2tensor(weighted_A_local_66)

    N2 = TensorTools.random_isotropic_N2()
    N4 = TensorTools.random_isotropic_N4()
    weighted_A_avg_3333 = TensorTools.orientation_average(weighted_A_local_3333, N2, N4)
    weighted_A_avg_66 = TensorTools.tensor2mandel(weighted_A_avg_3333)

    if np.linalg.norm(dC) < 1e-18:
        return np.eye(6)
    try:
        A_avg = np.linalg.solve(dC, weighted_A_avg_66)
    except np.linalg.LinAlgError:
        A_avg = np.linalg.pinv(dC, rcond=1e-12) @ weighted_A_avg_66
    return A_avg


def self_consistent_elastic_random_spheroids(
    K: Sequence[float],
    G: Sequence[float],
    X: Sequence[float],
    aspect_ratio: Sequence[float],
    *,
    rho_kg_m3: Sequence[float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    damping: float = 0.35,
) -> SCElasticResult:
    K = _as_array(K, "K")
    G = _as_array(G, "G")
    X = _normalize_fractions(_as_array(X, "X"))
    aspect_ratio = _as_array(aspect_ratio, "aspect_ratio")

    if not (K.size == G.size == X.size == aspect_ratio.size):
        raise ValueError("K, G, X, and aspect_ratio must have the same length.")
    if np.any(K <= 0.0) or np.any(G < 0.0):
        raise ValueError("Bulk moduli must be positive and shear moduli non-negative.")
    if np.any(aspect_ratio <= 0.0):
        raise ValueError("All aspect ratios must be positive.")

    phase_stiffness = [isotropic_stiffness_from_KG(float(k), float(g)) for k, g in zip(K, G)]

    K_eff, G_eff = _vrh_initial_guess(K, np.maximum(G, 1e-12), X)
    Ceff66 = isotropic_stiffness_from_KG(K_eff, G_eff)

    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        A_sum = np.zeros((6, 6), dtype=float)
        CA_sum = np.zeros((6, 6), dtype=float)

        for xi, Ci66, ar in zip(X, phase_stiffness, aspect_ratio):
            A_i = _orientation_averaged_A_local(Ceff66, Ci66, float(ar))
            A_sum += float(xi) * A_i
            CA_sum += float(xi) * TensorTools.tensor_product(Ci66, A_i)

        try:
            inv_A_sum = np.linalg.inv(A_sum)
        except np.linalg.LinAlgError:
            inv_A_sum = np.linalg.pinv(A_sum, rcond=1e-12)
        Cnew66 = TensorTools.tensor_product(CA_sum, inv_A_sum)
        Ceff66 = (1.0 - damping) * Ceff66 + damping * Cnew66

        K_new, G_new = bulk_shear_from_stiffness_tensor(TensorTools.mandel2tensor(Ceff66))
        K_new = float(max(K_new, 1e-12))
        G_new = float(max(G_new, 1e-12))
        # Project back to a physical isotropic stiffness tensor (keeps Poisson ratio in range).
        Ceff66 = isotropic_stiffness_from_KG(K_new, G_new)

        relK = abs(K_new - K_eff) / max(abs(K_eff), 1e-18)
        relG = abs(G_new - G_eff) / max(abs(G_eff), 1e-18)

        K_eff, G_eff = float(K_new), float(G_new)
        n_iter = it

        if max(relK, relG) < tol:
            converged = True
            break

    Ceff3333 = TensorTools.mandel2tensor(Ceff66)

    rho_eff = None
    vp = None
    vs = None
    if rho_kg_m3 is not None:
        rho = _as_array(rho_kg_m3, "rho_kg_m3")
        if rho.size != X.size:
            raise ValueError("rho_kg_m3 must have the same length as phase arrays.")
        if np.any(rho <= 0.0):
            raise ValueError("All densities must be positive.")
        rho_eff = float(np.sum(X * rho))
        vp = float(np.sqrt(max((K_eff + 4.0 * G_eff / 3.0) / rho_eff, 1e-30)))
        vs = float(np.sqrt(max(G_eff / rho_eff, 1e-30)))

    return SCElasticResult(
        Ceff66=Ceff66,
        Ceff3333=Ceff3333,
        K_eff=K_eff,
        G_eff=G_eff,
        rho_eff_kg_m3=rho_eff,
        vp_m_s=vp,
        vs_m_s=vs,
        n_iter=n_iter,
        converged=converged,
    )


def _configure_matplotlib_env(out_dir: Path) -> None:
    cache_dir = (out_dir / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_config_dir = (out_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def sweep_porosity(
    *,
    phi: np.ndarray,
    aspect_ratio_pore: float,
    Km_pa: float,
    Gm_pa: float,
    Kf_pa: float,
    rho_m_kg_m3: float,
    rho_f_kg_m3: float,
    max_iter: int,
    tol: float,
    damping: float,
    model: str = "berryman",
) -> dict[str, np.ndarray]:
    K_eff = np.full_like(phi, np.nan, dtype=float)
    G_eff = np.full_like(phi, np.nan, dtype=float)
    vp = np.full_like(phi, np.nan, dtype=float)
    vs = np.full_like(phi, np.nan, dtype=float)
    n_iter = np.full_like(phi, -1, dtype=int)
    ok = np.zeros_like(phi, dtype=bool)

    for i, ph in enumerate(phi):
        which = model.strip().lower()
        if which == "stiffness":
            res = self_consistent_elastic_random_spheroids(
                K=[Km_pa, Kf_pa],
                G=[Gm_pa, 1e-9],
                X=[1.0 - float(ph), float(ph)],
                aspect_ratio=[1.0, float(aspect_ratio_pore)],
                rho_kg_m3=[rho_m_kg_m3, rho_f_kg_m3],
                max_iter=max_iter,
                tol=tol,
                damping=damping,
            )
            K_eff[i] = res.K_eff
            G_eff[i] = res.G_eff
            vp[i] = float(res.vp_m_s) if res.vp_m_s is not None else np.nan
            vs[i] = float(res.vs_m_s) if res.vs_m_s is not None else np.nan
            n_iter[i] = int(res.n_iter or -1)
            ok[i] = bool(res.converged)
        elif which == "berryman":
            from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores

            try:
                warm_guess = None if i == 0 else (float(K_eff[i - 1]) / 1e9, float(G_eff[i - 1]) / 1e9)
                # Crack-like pores can drive Geff extremely close to zero; the fixed-point solver
                # becomes numerically stiff. We try a few fallbacks rather than returning NaNs.
                rel_list = [float(damping)]
                if aspect_ratio_pore <= 0.1:
                    rel_list += [0.1, 0.05, 0.02]
                guess_list: list[tuple[float, float] | None] = [
                    warm_guess,
                    (max(float(Kf_pa / 1e9), 1e-6), 1e-6),
                    (max((1.0 - float(ph)) * float(Km_pa / 1e9) + float(ph) * float(Kf_pa / 1e9), 1e-6), 1e-6),
                ]

                last_exc: Exception | None = None
                K_gpa = G_gpa = None  # type: ignore[assignment]
                for rel in rel_list:
                    for guess in guess_list:
                        try:
                            K_gpa, G_gpa = berryman_self_consistent_spheroidal_pores(
                                matrix_bulk_gpa=Km_pa / 1e9,
                                matrix_shear_gpa=Gm_pa / 1e9,
                                porosity=float(ph),
                                pore_bulk_gpa=Kf_pa / 1e9,
                                aspect_ratio=float(aspect_ratio_pore),
                                tol=tol,
                                max_iter=max_iter,
                                relaxation=float(rel),
                                initial_guess_gpa=guess,
                            )
                            raise StopIteration
                        except StopIteration:
                            break
                        except Exception as exc:
                            last_exc = exc
                    else:
                        continue
                    break
                else:
                    raise last_exc or RuntimeError("Berryman SC failed to converge.")

                K_eff[i] = float(K_gpa) * 1e9
                G_eff[i] = float(G_gpa) * 1e9
                rho_eff = (1.0 - float(ph)) * rho_m_kg_m3 + float(ph) * rho_f_kg_m3
                vp[i] = float(np.sqrt(max((K_eff[i] + 4.0 * G_eff[i] / 3.0) / rho_eff, 1e-30)))
                vs[i] = float(np.sqrt(max(G_eff[i] / rho_eff, 1e-30)))
                ok[i] = True
                n_iter[i] = -1
            except Exception:
                ok[i] = False
                n_iter[i] = -1
        else:
            raise ValueError("model must be 'berryman' or 'stiffness'.")

    return {"phi": phi, "K_eff": K_eff, "G_eff": G_eff, "vp": vp, "vs": vs, "n_iter": n_iter, "ok": ok}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Elastic self-consistent sweep vs porosity.")
    ap.add_argument("--aspect-ratio", type=float, default=1e-2, help="Pore aspect ratio α (e.g., 1e-2)")
    ap.add_argument("--pore-K-gpa", type=float, default=2.2, help="Pore/fluid bulk modulus (GPa). Use 0 for dry.")
    ap.add_argument("--phi-max", type=float, default=0.35, help="Maximum porosity (fraction)")
    ap.add_argument("--n-phi", type=int, default=71, help="Number of porosity points")
    ap.add_argument("--max-iter", type=int, default=4000, help="Max SC iterations")
    ap.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance (relative)")
    ap.add_argument("--damping", type=float, default=0.2, help="Relaxation/damping factor (0..1)")
    ap.add_argument(
        "--model",
        type=str,
        default="berryman",
        choices=["berryman", "stiffness"],
        help="Which SC implementation to use.",
    )
    ap.add_argument("--include-dem", action="store_true", help="Also compute DEM curve for comparison")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    Km_pa = 76.8e9
    Gm_pa = 32.0e9
    Kf_pa = float(args.pore_K_gpa) * 1e9
    rho_m = 2710.0
    rho_f = 1000.0

    phi = np.linspace(0.0, float(args.phi_max), int(args.n_phi))
    sweep = sweep_porosity(
        phi=phi,
        aspect_ratio_pore=float(args.aspect_ratio),
        Km_pa=Km_pa,
        Gm_pa=Gm_pa,
        Kf_pa=Kf_pa,
        rho_m_kg_m3=rho_m,
        rho_f_kg_m3=rho_f,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        damping=float(args.damping),
        model=str(args.model),
    )

    dem = None
    if bool(args.include_dem):
        from rockphysx.models.emt.dem_transport_elastic import dem_elastic_moduli

        Kd = np.full_like(phi, np.nan, dtype=float)
        Gd = np.full_like(phi, np.nan, dtype=float)
        vpd = np.full_like(phi, np.nan, dtype=float)
        vsd = np.full_like(phi, np.nan, dtype=float)
        for i, ph in enumerate(phi):
            try:
                r = dem_elastic_moduli(
                    matrix_bulk=Km_pa / 1e9,
                    matrix_shear=Gm_pa / 1e9,
                    inclusion_bulk=Kf_pa / 1e9,
                    inclusion_shear=0.0,
                    inclusion_fraction=float(ph),
                    aspect_ratio=float(args.aspect_ratio),
                    n_steps=800,
                )
                K_gpa = float(r.bulk_modulus)
                G_gpa = float(r.shear_modulus)
            except Exception:
                continue
            Kd[i] = K_gpa * 1e9
            Gd[i] = G_gpa * 1e9
            rho_eff = (1.0 - float(ph)) * rho_m + float(ph) * rho_f
            vpd[i] = float(np.sqrt(max((Kd[i] + 4.0 * Gd[i] / 3.0) / rho_eff, 1e-30)))
            vsd[i] = float(np.sqrt(max(Gd[i] / rho_eff, 1e-30)))
        dem = {"K_eff": Kd, "G_eff": Gd, "vp": vpd, "vs": vsd}

    # Terminal summary (a few points)
    idx = np.linspace(0, len(phi) - 1, 8).round().astype(int)
    print("Porosity sweep (elastic SC):")
    for i in idx:
        print(
            f"  phi={phi[i]:.3f}  Keff={sweep['K_eff'][i]/1e9:7.3f} GPa  Geff={sweep['G_eff'][i]/1e9:7.3f} GPa"
        )
        print(
            f"       Vp={sweep['vp'][i]/1e3:6.3f} km/s  Vs={sweep['vs'][i]/1e3:6.3f} km/s  it={sweep['n_iter'][i]:3d}  ok={bool(sweep['ok'][i])}"
        )

    # Plot
    _configure_matplotlib_env(Path(args.out_dir))
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
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6), sharex=True, constrained_layout=True)
        axK, axG, axVp, axVs = axes.ravel()

        label_sc = "SC (Berryman)" if str(args.model).strip().lower() == "berryman" else "SC (stiffness)"
        axK.plot(phi, sweep["K_eff"] / 1e9, lw=2.2, label=label_sc)
        axG.plot(phi, sweep["G_eff"] / 1e9, lw=2.2, label=label_sc)
        axVp.plot(phi, sweep["vp"] / 1e3, lw=2.2, label=label_sc)
        axVs.plot(phi, sweep["vs"] / 1e3, lw=2.2, label=label_sc)

        if dem is not None:
            axK.plot(phi, dem["K_eff"] / 1e9, lw=2.2, ls=":", label="DEM")
            axG.plot(phi, dem["G_eff"] / 1e9, lw=2.2, ls=":", label="DEM")
            axVp.plot(phi, dem["vp"] / 1e3, lw=2.2, ls=":", label="DEM")
            axVs.plot(phi, dem["vs"] / 1e3, lw=2.2, ls=":", label="DEM")

        axK.set_ylabel("Keff (GPa)")
        axG.set_ylabel("Geff (GPa)")
        axVp.set_ylabel("Vp (km/s)")
        axVs.set_ylabel("Vs (km/s)")
        for ax in (axK, axG, axVp, axVs):
            ax.grid(True, alpha=0.25)
        for ax in (axVp, axVs):
            ax.set_xlabel("Porosity, φ (fraction)")

        handles, labels = axK.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, -0.02))

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sca_elastic_self_consistent_vs_phi_ar_{args.aspect_ratio:g}".replace("-", "m").replace(".", "p")
        out_png = out_dir / f"{stem}.png"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_png}")
