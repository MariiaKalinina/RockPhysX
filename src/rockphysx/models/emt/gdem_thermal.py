from __future__ import annotations

"""
Generalized multicomponent/backbone DEM model for thermal conductivity.

Source / attribution
--------------------
This implementation is inspired by:

Norris, A. N., Callegari, A. J., and Sheng, P. (1985)
"A generalized differential effective medium theory"

Key conceptual points from the paper abstract:
- ordinary DEM corresponds to incremental build-up with one phase dilute
  and another phase as a percolating backbone;
- generalized DEM assumes a third phase which acts as a backbone;
- the other two phases are progressively added to that backbone;
- for φ < 1 the result depends on the backbone and the mixture path;
- as φ -> 1 the effective-medium approximation (EMA) is obtained.

Important note
--------------
This file is an engineering implementation of that *backbone/path* idea for
thermal conductivity by repeatedly applying the 2-phase thermal DEM update
to a chosen backbone phase.

It is intentionally NOT presented as a verbatim transcription of all Norris et al.
equations. The goal is a practical, documented, path-dependent generalized DEM
for use in RockPhysX.
"""

from collections.abc import Sequence

import numpy as np

from rockphysx.models.emt.dem_thermal import dem_thermal_conductivity
from rockphysx.utils.validation import normalize_fractions


def generalized_dem_thermal_conductivity(
    volume_fractions: Sequence[float],
    thermal_conductivities: Sequence[float],
    aspect_ratios: Sequence[float],
    *,
    backbone_index: int = 0,
    addition_order: Sequence[int] | None = None,
) -> float:
    """
    Generalized multicomponent/backbone DEM for thermal conductivity.

    Parameters
    ----------
    volume_fractions
        Final phase volume fractions. Must sum to 1.
    thermal_conductivities
        Conductivity of each phase.
    aspect_ratios
        Aspect ratio assigned to each phase when it is added as an inclusion.
        The backbone phase aspect ratio is ignored.
    backbone_index
        Index of the initial backbone phase.
    addition_order
        Order in which non-backbone phases are added.
        If omitted, phases are added in their natural index order.

    Returns
    -------
    float
        Effective thermal conductivity.

    Notes
    -----
    The algorithm:
    1. Start from the backbone phase conductivity.
    2. Add each non-backbone phase in the chosen order.
    3. At each step, renormalize the added phase fraction relative to the
       current partially-built composite:
           dphi_eff = phi_j / (1 - accumulated_added_fraction)
    4. Apply the 2-phase thermal DEM update using the current composite as
       the host/background and phase j as the added inclusion phase.

    Because the method is path-dependent by construction, changing
    `backbone_index` or `addition_order` changes the result.
    """
    phi = normalize_fractions(volume_fractions)
    lam = np.asarray(thermal_conductivities, dtype=float)
    alpha = np.asarray(aspect_ratios, dtype=float)

    if not (len(phi) == len(lam) == len(alpha)):
        raise ValueError("volume_fractions, thermal_conductivities, and aspect_ratios must have the same length.")

    n = len(phi)
    if not 0 <= backbone_index < n:
        raise ValueError("backbone_index out of range.")

    if addition_order is None:
        addition_order = [i for i in range(n) if i != backbone_index]
    else:
        addition_order = list(addition_order)
        expected = {i for i in range(n) if i != backbone_index}
        if set(addition_order) != expected:
            raise ValueError("addition_order must contain each non-backbone phase exactly once.")

    k_eff = float(lam[backbone_index])
    added_fraction = 0.0

    for j in addition_order:
        remaining_host_fraction = 1.0 - added_fraction
        if remaining_host_fraction <= 0.0:
            raise ValueError("Invalid phase-addition path: no host fraction remains.")

        dphi_eff = float(phi[j] / remaining_host_fraction)

        k_eff = dem_thermal_conductivity(
            matrix_conductivity=k_eff,
            inclusion_conductivity=float(lam[j]),
            inclusion_fraction=dphi_eff,
            aspect_ratio=float(alpha[j]),
        )

        added_fraction += float(phi[j])

    return float(k_eff)

"""
Example usage:

from rockphysx.models.emt.gdem_thermal import generalized_dem_thermal_conductivity

k_gdem = generalized_dem_thermal_conductivity(
    volume_fractions=[0.80, 0.15, 0.05],
    thermal_conductivities=[3.0, 0.13, 0.60],
    aspect_ratios=[1.0, 0.1, 0.001],
    backbone_index=0,         # phase 0 is the backbone
    addition_order=[1, 2],    # add oil-filled pores, then water-filled cracks
)

"""