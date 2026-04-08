from __future__ import annotations

from rockphysx.core.parameters import MicrostructureParameters
from rockphysx.models.emt.bounds import geometric_mean
from rockphysx.models.emt.maxwell import maxwell_garnett_isotropic
from rockphysx.models.emt.gsa_thermal import gsa_effective_property
from rockphysx.models.emt.sca_thermal import sca_effective_conductivity, sca_sc_effective_conductivity


def electrical_conductivity(
    matrix_value: float,
    fluid_value: float,
    porosity: float,
    microstructure: MicrostructureParameters,
    *,
    model: str = "gsa",
) -> float:
    """Predict effective electrical conductivity in S/m.

    In the current implementation, the connectivity factor multiplies the pore-fluid
    conductivity contribution, allowing a first-order representation of pore-network
    disconnection during cross-property or A2-style calibration.
    """
    effective_fluid = microstructure.connectivity * fluid_value
    solid_fraction = 1.0 - porosity
    model_name = model.lower()
    if model_name == "gsa":
        return gsa_effective_property(
            [solid_fraction, porosity],
            [matrix_value, effective_fluid],
            [1.0, microstructure.aspect_ratio],
        )
    if model_name == "sca":
        return sca_effective_conductivity(
            matrix_value,
            effective_fluid,
            porosity,
            aspect_ratio=microstructure.aspect_ratio,
        )
    if model_name == "sca_sc":
        return sca_sc_effective_conductivity(
            matrix_value,
            effective_fluid,
            porosity,
            aspect_ratio=microstructure.aspect_ratio,
        )
    if model_name == "maxwell":
        return maxwell_garnett_isotropic(matrix_value, effective_fluid, porosity)
    if model_name == "geometric":
        return geometric_mean([solid_fraction, porosity], [matrix_value, effective_fluid])
    raise ValueError(f"Unknown electrical model {model!r}.")


def electrical_resistivity(
    matrix_value: float,
    fluid_value: float,
    porosity: float,
    microstructure: MicrostructureParameters,
    *,
    model: str = "gsa",
) -> float:
    """Predict effective electrical resistivity in ohm·m as the inverse of conductivity."""
    sigma = electrical_conductivity(matrix_value, fluid_value, porosity, microstructure, model=model)
    if sigma <= 0.0:
        raise ValueError("Predicted electrical conductivity must be positive to compute resistivity.")
    return 1.0 / sigma
