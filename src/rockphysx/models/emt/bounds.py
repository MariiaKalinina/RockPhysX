from __future__ import annotations

import numpy as np

from rockphysx.utils.validation import normalize_fractions


def Likhteneker(volume_fractions, thermal_conductivities):
    """
    Calculate the effective thermal conductivity using the Likhteneker model.
    """
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(thermal_conductivities, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("thermal_conductivities must be strictly positive.")
    return float(np.exp(np.sum(phi * np.log(lam))))


def Wiener_Upper_Bound(volume_fractions, thermal_conductivities):
    """
    Calculate the upper bound of thermal conductivity using the Wiener model.
    """
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(thermal_conductivities, dtype=float)
    return float(np.sum(phi * lam))


def Wiener_Lower_Bound(volume_fractions, thermal_conductivities):
    """
    Calculate the lower bound of thermal conductivity using the Wiener model.
    """
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(thermal_conductivities, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("thermal_conductivities must be strictly positive.")
    return float(1.0 / np.sum(phi / lam))


def Wiener_Average(volume_fractions, thermal_conductivities):
    """
    Calculate the average thermal conductivity using the Wiener bounds.
    """
    upper_bound = Wiener_Upper_Bound(volume_fractions, thermal_conductivities)
    lower_bound = Wiener_Lower_Bound(volume_fractions, thermal_conductivities)
    return float(np.mean([upper_bound, lower_bound]))


def Upper_Hashin_Strikman(phi, lam):
    """
    Calculate the upper bound of thermal conductivity using the Hashin-Strikman model.
    """
    phi = normalize_fractions(phi)
    lam = np.asarray(lam, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("thermal_conductivities must be strictly positive.")

    L0 = float(np.max(lam))
    f = 1.0 / 3.0
    hs_comp1, hs_comp2 = [], []

    for i, j in zip(phi, lam, strict=True):
        termc_3 = i * j / (L0 * (1.0 - f) + j * f)
        termz_3 = i / (L0 * (1.0 - f) + j * f)

        termc_12 = i * j / (L0 * (1.0 + f) / 2.0 + j * (1.0 - f) / 2.0)
        termz_12 = i / (L0 * (1.0 + f) / 2.0 + j * (1.0 - f) / 2.0)

        termc_all = (termc_3 + 2.0 * termc_12) / 3.0
        termz_all = (termz_3 + 2.0 * termz_12) / 3.0

        hs_comp1.append(termc_all)
        hs_comp2.append(termz_all)

    return float(np.sum(hs_comp1) / np.sum(hs_comp2))


def Lower_Hashin_Strikman(phi, lam):
    """
    Calculate the lower bound of thermal conductivity using the Hashin-Strikman model.
    """
    phi = normalize_fractions(phi)
    lam = np.asarray(lam, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("thermal_conductivities must be strictly positive.")

    L0 = float(np.min(lam))
    f = 1.0 / 3.0
    hs_comp1, hs_comp2 = [], []

    for i, j in zip(phi, lam, strict=True):
        termc_3 = i * j / (L0 * (1.0 - f) + j * f)
        termz_3 = i / (L0 * (1.0 - f) + j * f)

        termc_12 = i * j / (L0 * (1.0 + f) / 2.0 + j * (1.0 - f) / 2.0)
        termz_12 = i / (L0 * (1.0 + f) / 2.0 + j * (1.0 - f) / 2.0)

        termc_all = (termc_3 + 2.0 * termc_12) / 3.0
        termz_all = (termz_3 + 2.0 * termz_12) / 3.0

        hs_comp1.append(termc_all)
        hs_comp2.append(termz_all)

    return float(np.sum(hs_comp1) / np.sum(hs_comp2))


def Hashin_Strikman_Average(volume_fractions, thermal_conductivities):
    """
    Calculate the average thermal conductivity using the Hashin-Strikman bounds.
    """
    upper_bound = Upper_Hashin_Strikman(volume_fractions, thermal_conductivities)
    lower_bound = Lower_Hashin_Strikman(volume_fractions, thermal_conductivities)
    return float(np.mean([upper_bound, lower_bound]))


# Compatibility aliases
lichtenecker_average = Likhteneker
geometric_mean = Likhteneker
wiener_upper = Wiener_Upper_Bound
wiener_lower = Wiener_Lower_Bound
hashin_shtrikman_upper = Upper_Hashin_Strikman
hashin_shtrikman_lower = Lower_Hashin_Strikman