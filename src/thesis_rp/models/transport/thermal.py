from __future__ import annotations

from thesis_rp.core.parameters import MicrostructureParameters
from thesis_rp.models.emt.bounds import (
    geometric_mean,
    hashin_shtrikman_lower,
    hashin_shtrikman_upper,
    wiener_lower,
    wiener_upper,
)
from thesis_rp.models.emt.bruggeman import bruggeman_isotropic
from thesis_rp.models.emt.maxwell import maxwell_garnett_isotropic
from thesis_rp.models.emt.self_consistent import gsa_effective_property


def thermal_conductivity(
    matrix_value: float,
    fluid_value: float,
    porosity: float,
    microstructure: MicrostructureParameters,
    *,
    model: str = "gsa",
) -> float:
    """Predict effective thermal conductivity for a porous rock.

    Parameters
    ----------
    matrix_value
        Thermal conductivity of the solid matrix, W/(m·K).
    fluid_value
        Thermal conductivity of the pore fluid, W/(m·K).
    porosity
        Pore volume fraction.
    microstructure
        Shared microstructural parameter object.
    model
        Supported values:
        - "gsa"
        - "maxwell"
        - "bruggeman"
        - "geometric"
        - "wiener_upper"
        - "wiener_lower"
        - "hs_upper"
        - "hs_lower"
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
    if model_name == "maxwell":
        return maxwell_garnett_isotropic(matrix_value, fluid_value, porosity)
    if model_name == "bruggeman":
        return bruggeman_isotropic(fractions, values)
    if model_name == "geometric":
        return geometric_mean(fractions, values)
    if model_name == "wiener_upper":
        return wiener_upper(fractions, values)
    if model_name == "wiener_lower":
        return wiener_lower(fractions, values)
    if model_name == "hs_upper":
        return hashin_shtrikman_upper(fractions, values)
    if model_name == "hs_lower":
        return hashin_shtrikman_lower(fractions, values)

    raise ValueError(f"Unknown thermal model {model!r}.")