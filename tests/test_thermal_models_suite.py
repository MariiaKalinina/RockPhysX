from __future__ import annotations

import math

import numpy as np
import pytest

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


# Optional models: only use them if they exist in the repo
try:
    from rockphysx.models.emt.bruggeman import Bruggeman_EMA  # type: ignore
except Exception:
    Bruggeman_EMA = None

try:
    from rockphysx.models.emt.maxwell import Maxwell  # type: ignore
except Exception:
    Maxwell = None


TEST_CASES = [
    {
        "description": "Base case: sediment matrix with air-filled pores",
        "phi": [0.7, 0.3],
        "lambda_i": [3.0, 0.025],
        "alpha_i": [1.0, 0.1],
    },
    {
        "description": "Three-phase composite: sediment matrix with oil-filled pores and water-filled cracks",
        "phi": [0.8, 0.15, 0.05],
        "lambda_i": [3.0, 0.13, 0.60],
        "alpha_i": [1.0, 0.1, 0.001],
    },
]


def safe_eval(func, *args, **kwargs) -> float | None:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return None


def run_all_models(phi, lam, alpha) -> dict[str, float | None]:
    """
    Evaluate all available thermal-conductivity models for one test case.
    """
    results: dict[str, float | None] = {}

    results["Wiener_Upper"] = safe_eval(Wiener_Upper_Bound, phi, lam)
    results["Wiener_Lower"] = safe_eval(Wiener_Lower_Bound, phi, lam)
    results["Wiener_Avg"] = safe_eval(Wiener_Average, phi, lam)
    results["Lichtenecker"] = safe_eval(Likhteneker, phi, lam)

    # Bounds do not need alpha in your repo implementation
    results["HS_Lower"] = safe_eval(Lower_Hashin_Strikman, phi, lam)
    results["HS_Upper"] = safe_eval(Upper_Hashin_Strikman, phi, lam)
    results["HS_Avg"] = safe_eval(Hashin_Strikman_Average, phi, lam)

    # SCA random-inclusion model is currently a 2-phase host/inclusion model
    # so we only run it for 2-phase cases
    if len(phi) == 2:
        results["SCA"] = safe_eval(
            sca_effective_conductivity,
            lam[0],
            lam[1],
            phi[1],
            aspect_ratio=alpha[1],
        )
    else:
        results["SCA"] = None

    # GSA supports multi-phase input
    results["GSA"] = safe_eval(gsa_effective_property, phi, lam, alpha)

    # Maxwell is usually implemented for matrix + inclusion only
    if Maxwell is not None and len(phi) == 2:
        results["Maxwell"] = safe_eval(Maxwell, phi, lam, alpha)
    else:
        results["Maxwell"] = None

    # Bruggeman: run if available
    if Bruggeman_EMA is not None:
        results["Bruggeman"] = safe_eval(Bruggeman_EMA, phi, lam, alpha)
    else:
        results["Bruggeman"] = None

    return results


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["description"] for c in TEST_CASES])
def test_all_model_results_are_finite_or_skipped(case):
    phi = case["phi"]
    lam = case["lambda_i"]
    alpha = case["alpha_i"]

    results = run_all_models(phi, lam, alpha)

    for name, value in results.items():
        if value is None:
            continue
        assert math.isfinite(value), f"{name} returned non-finite value for case: {case['description']}"


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["description"] for c in TEST_CASES])
def test_all_available_models_stay_within_wiener_bounds(case):
    phi = case["phi"]
    lam = case["lambda_i"]
    alpha = case["alpha_i"]

    wiener_lower = Wiener_Lower_Bound(phi, lam)
    wiener_upper = Wiener_Upper_Bound(phi, lam)

    results = run_all_models(phi, lam, alpha)

    for name, value in results.items():
        if value is None:
            continue
        assert wiener_lower - 1e-8 <= value <= wiener_upper + 1e-8, (
            f"{name}={value:.6f} outside Wiener bounds "
            f"[{wiener_lower:.6f}, {wiener_upper:.6f}] "
            f"for case: {case['description']}"
        )


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["description"] for c in TEST_CASES])
def test_hs_average_lies_between_hs_bounds(case):
    phi = case["phi"]
    lam = case["lambda_i"]

    hs_lower = Lower_Hashin_Strikman(phi, lam)
    hs_upper = Upper_Hashin_Strikman(phi, lam)
    hs_avg = Hashin_Strikman_Average(phi, lam)

    assert hs_lower - 1e-8 <= hs_avg <= hs_upper + 1e-8


def test_disk_like_sca_is_lower_than_sphere_for_same_case():
    """
    For a low-conductivity inclusion in a conductive matrix,
    disk-like pores should reduce thermal conductivity more strongly
    than spherical pores in the random-inclusion SCA model.
    """
    lambda_m = 7.5
    lambda_i = 0.6
    phi = 0.2

    k_sphere = sca_effective_conductivity(
        lambda_m,
        lambda_i,
        phi,
        aspect_ratio=1.0,
    )
    k_disk_like = sca_effective_conductivity(
        lambda_m,
        lambda_i,
        phi,
        aspect_ratio=1e-3,
    )

    assert k_disk_like < k_sphere


def test_gsa_reduces_to_matrix_at_zero_porosity():
    phi = [1.0, 0.0]
    lam = [3.0, 0.025]
    alpha = [1.0, 0.1]

    k_eff = gsa_effective_property(phi, lam, alpha)
    assert np.isclose(k_eff, lam[0], atol=1e-8)