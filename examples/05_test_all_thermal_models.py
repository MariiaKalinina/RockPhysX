from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.models.emt.bounds import (
    Hashin_Strikman_Average,
    Likhteneker,
    Lower_Hashin_Strikman,
    Upper_Hashin_Strikman,
    Wiener_Average,
    Wiener_Lower_Bound,
    Wiener_Upper_Bound,
)
from rockphysx.models.emt.gsa_thermal import gsa_effective_property
from rockphysx.models.emt.sca_thermal import sca_effective_conductivity
from rockphysx.models.emt.bruggeman import bruggeman_isotropic
from rockphysx.models.emt.maxwell import maxwell_garnett_isotropic
from rockphysx.models.emt.dem_thermal import dem_thermal_conductivity
from rockphysx.models.emt.gdem_thermal import generalized_dem_thermal_conductivity

def safe_run(func, *args, **kwargs) -> Optional[float]:
    try:
        return float(func(*args, **kwargs))
    except Exception as e:
        print(f"{func.__name__} failed: {e}")
        return None


def run_console_test() -> None:
    test_cases = [
        {
            "description": "Base case: Sediment matrix with air-filled pores",
            "phi": [0.7, 0.3],
            "lambda_i": [3.0, 0.025],
            "alpha_i": [1.0, 0.1],
        },
        {
            "description": "Three-phase composite: Sediment matrix with oil-filled pores and water-filled cracks",
            "phi": [0.8, 0.15, 0.05],
            "lambda_i": [3.0, 0.13, 0.60],
            "alpha_i": [1.0, 0.1, 0.001],
        },
    ]

    print("TESTING ALL THERMAL CONDUCTIVITY MODELS")
    print("═" * 90)

    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['description']}")
        print(f"   φ      = {case['phi']}")
        print(f"   λ_i    = {case['lambda_i']}")
        print(f"   α_i    = {case['alpha_i']}")

        phi = case["phi"]
        lam = case["lambda_i"]
        alpha = case["alpha_i"]

        wiener_lower = Wiener_Lower_Bound(phi, lam)
        wiener_upper = Wiener_Upper_Bound(phi, lam)
        wiener_avg = Wiener_Average(phi, lam)

        print("\nBOUNDS (Reference):")
        print(f"   Wiener Lower: {wiener_lower:8.4f} W/m·K")
        print(f"   Wiener Upper: {wiener_upper:8.4f} W/m·K")
        print(f"   Wiener Avg:   {wiener_avg:8.4f} W/m·K")

        results: Dict[str, Optional[float]] = {}
        results["Wiener_Upper"] = safe_run(Wiener_Upper_Bound, phi, lam)
        results["Wiener_Lower"] = safe_run(Wiener_Lower_Bound, phi, lam)
        results["Wiener_Avg"] = safe_run(Wiener_Average, phi, lam)
        results["Lichtenecker"] = safe_run(Likhteneker, phi, lam)

        results["HS_Lower"] = safe_run(Lower_Hashin_Strikman, phi, lam)
        results["HS_Upper"] = safe_run(Upper_Hashin_Strikman, phi, lam)
        results["HS_Avg"] = safe_run(Hashin_Strikman_Average, phi, lam)

        if len(phi) == 2:
            results["SCA"] = safe_run(
                sca_effective_conductivity,
                lam[0],
                lam[1],
                phi[1],
                aspect_ratio=alpha[1],
            )
        else:
            results["SCA"] = None

        results["GSA"] = safe_run(gsa_effective_property, phi, lam, alpha)

        if len(phi) == 2:
            results["Maxwell"] = safe_run(
                maxwell_garnett_isotropic,
                lam[0],
                lam[1],
                phi[1],
            )
        else:
            results["Maxwell"] = None

        results["Bruggeman"] = safe_run(bruggeman_isotropic, phi, lam)

        if len(phi) == 2:
            results["DEM"] = safe_run(
                dem_thermal_conductivity,
                lam[0],
                lam[1],
                phi[1],
                aspect_ratio=alpha[1],
            )
        else:
            results["DEM"] = None

        print("\nMODEL RESULTS:")
        for name, val in results.items():
            if val is not None:
                print(f"   {name:>15}: {val:8.4f} W/m·K")
            else:
                print(f"   {name:>15}: {'---':>8}")

        valid = True
        for name, val in results.items():
            if val is None:
                continue
            if not (wiener_lower - 1e-6 <= val <= wiener_upper + 1e-6):
                print(
                    f"{name}={val:.4f} outside Wiener bounds "
                    f"[{wiener_lower:.4f}, {wiener_upper:.4f}]"
                )
                valid = False

        if valid:
            print("All results are physically consistent (within Wiener bounds)")
        else:
            print("Some models produced unphysical results")

        hs_l = results["HS_Lower"]
        hs_u = results["HS_Upper"]
        hs_avg = results["HS_Avg"]
        if all(v is not None for v in [hs_l, hs_u, hs_avg]):
            if not (hs_l - 1e-6 <= hs_avg <= hs_u + 1e-6):
                print(f"HS Average {hs_avg:.4f} not in [{hs_l:.4f}, {hs_u:.4f}]")

    print("\nALL TESTS COMPLETED")
    print("Check output for physical consistency and model behavior")


def run_porosity_vs_tc_plot() -> None:
    lambda_matrix = 3.0
    lambda_void = 0.13
    porosity_values = np.linspace(0.0, 1.0, 120)

    results = {
        "Wiener_Upper": [],
        "Wiener_Lower": [],
        "Wiener_Avg": [],
        "Lichtenecker": [],
        "HS_Lower": [],
        "HS_Upper": [],
        "HS_Average": [],
        "SCA": [],
        "GSA": [],
        "Bruggeman": [],
        "Maxwell": [],
        "DEM": [],
        "GDEM": [],
    }

    alpha_i = [1.0, 0.1]

    for phi_void in porosity_values:
        phi_matrix = 1.0 - phi_void
        phi = [phi_matrix, phi_void]
        lam = [lambda_matrix, lambda_void]

        results["Wiener_Upper"].append(safe_run(Wiener_Upper_Bound, phi, lam))
        results["Wiener_Lower"].append(safe_run(Wiener_Lower_Bound, phi, lam))
        results["Wiener_Avg"].append(safe_run(Wiener_Average, phi, lam))
        results["Lichtenecker"].append(safe_run(Likhteneker, phi, lam))
        results["HS_Lower"].append(safe_run(Lower_Hashin_Strikman, phi, lam))
        results["HS_Upper"].append(safe_run(Upper_Hashin_Strikman, phi, lam))
        results["HS_Average"].append(safe_run(Hashin_Strikman_Average, phi, lam))
        results["SCA"].append(
            safe_run(
                sca_effective_conductivity,
                lambda_matrix,
                lambda_void,
                phi_void,
                aspect_ratio=alpha_i[1],
            )
        )
        results["GSA"].append(safe_run(gsa_effective_property, phi, lam, alpha_i))
        results["Bruggeman"].append(safe_run(bruggeman_isotropic, phi, lam))
        results["Maxwell"].append(safe_run(
                maxwell_garnett_isotropic,
                lambda_matrix,
                lambda_void,
                phi_void,
            ))
        
        results["DEM"].append(
            safe_run(
                dem_thermal_conductivity,
                lambda_matrix,
                lambda_void,
                phi_void,
                aspect_ratio=alpha_i[1],
            )
        )

        results["GDEM"].append(
            safe_run(
                generalized_dem_thermal_conductivity,
                [1.0 - phi_void, phi_void],
                [lambda_matrix, lambda_void],
                [1.0, alpha_i[1]],
                backbone_index=0,
            )
        )
                

    for key in results:
        results[key] = np.array(
            [np.nan if v is None else v for v in results[key]],
            dtype=float,
        )

    plt.figure(figsize=(9, 6))
    porosity_plot = porosity_values * 100.0

    styles = {
        "Wiener_Upper": ("tab:gray", "--", "Wiener Upper"),
        "Wiener_Lower": ("tab:gray", ":", "Wiener Lower"),
        "Wiener_Avg": ("tab:gray", "-", "Wiener Average"),
        "Lichtenecker": ("tab:brown", "-", "Lichtenecker"),
        "HS_Upper": ("tab:blue", "--", "HS Upper"),
        "HS_Lower": ("tab:blue", ":", "HS Lower"),
        "HS_Average": ("tab:blue", "-", "HS Average"),
        "SCA": ("tab:purple", "-", "SCA Random Inclusions"),
        "GSA": ("tab:red", "-", "GSA"),
        "Bruggeman": ("tab:green", "-", "Bruggeman EMA"),
        "Maxwell": ("tab:orange", "-", "Maxwell-Garnett"),
        "DEM": ("tab:pink", "-", "DEM (2-phase)"),
        "GDEM": ("tab:cyan", "-", "GDEM (backbone)"),
    }

    for key, (color, ls, label) in styles.items():
        data = results[key]
        valid = ~np.isnan(data)
        if np.any(valid):
            plt.plot(
                porosity_plot[valid],
                data[valid],
                color=color,
                linestyle=ls,
                linewidth=2,
                label=label,
            )

    plt.xlabel("Porosity (%)", fontsize=13)
    plt.ylabel("Effective Thermal Conductivity (W/m·K)", fontsize=13)
    plt.title(
        "Thermal conductivity vs porosity\n"
        f"Matrix: {lambda_matrix:.2f} W/m·K, Void: {lambda_void:.3f} W/m·K",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper right", ncol=2)
    plt.xlim(0, 96)
    plt.ylim(bottom=0)
    plt.tight_layout()

    outdir = Path("figures/thermal")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "05_tc_all_models_vs_porosity.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "05_tc_all_models_vs_porosity.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_console_test()
    run_porosity_vs_tc_plot()
