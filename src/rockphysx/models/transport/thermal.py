from __future__ import annotations

from rockphysx.core.parameters import MicrostructureParameters
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


def thermal_conductivity(
    matrix_value: float,
    fluid_value: float,
    porosity: float,
    microstructure: MicrostructureParameters,
    *,
    model: str = "gsa",
) -> float:
    """
    Thermal-conductivity forward model.

    Supported thermal models:
    - gsa
    - likhteneker / lichtenecker / geometric
    - wiener_upper
    - wiener_lower
    - wiener_average
    - hs_upper
    - hs_lower
    - hs_average
    """
    solid_fraction = 1.0 - porosity
    fractions = [solid_fraction, porosity]
    values = [matrix_value, fluid_value]

    model_name = model.lower()

    if model_name == "gsa":
        return gsa_effective_property(
            fractions,
            values,
            [1.0, microstructure.aspect_ratio],
        )
    if model_name in {"likhteneker", "lichtenecker", "geometric"}:
        return Likhteneker(fractions, values)
    if model_name == "wiener_upper":
        return Wiener_Upper_Bound(fractions, values)
    if model_name == "wiener_lower":
        return Wiener_Lower_Bound(fractions, values)
    if model_name == "wiener_average":
        return Wiener_Average(fractions, values)
    if model_name == "hs_upper":
        return Upper_Hashin_Strikman(fractions, values)
    if model_name == "hs_lower":
        return Lower_Hashin_Strikman(fractions, values)
    if model_name == "hs_average":
        return Hashin_Strikman_Average(fractions, values)

    raise ValueError(f"Unknown thermal model {model!r}.")