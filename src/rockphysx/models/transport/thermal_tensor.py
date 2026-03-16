from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rockphysx.models.emt.gsa_thermal import EffectiveConductivityModel


@dataclass(slots=True, frozen=True)
class AxisymmetricThermalConductivity:
    lambda_parallel: float
    lambda_perpendicular: float

    @property
    def ratio(self) -> float:
        return float(self.lambda_parallel / self.lambda_perpendicular)

    @property
    def tensor(self) -> np.ndarray:
        return np.diag(
            [
                self.lambda_perpendicular,
                self.lambda_perpendicular,
                self.lambda_parallel,
            ]
        )


def _run_tensor_gsa(
    matrix_value: float,
    fluid_value: float,
    porosity: float,
    *,
    aspect_ratio: float,
    fluid_orientation_flag: int,
    ind_friab: int = 11,
    friab: float = 0.0,
) -> np.ndarray:
    solid_fraction = 1.0 - porosity

    model = EffectiveConductivityModel(n_components=2)
    model.set_component_properties(
        0,
        conductivity=matrix_value,
        aspect_ratio=1.0,
        porosity=solid_fraction,
        orientation=1,
    )
    model.set_component_properties(
        1,
        conductivity=fluid_value,
        aspect_ratio=aspect_ratio,
        porosity=porosity,
        orientation=fluid_orientation_flag,
    )

    return model.calculate(ind_friab=ind_friab, friab=friab)


def thermal_conductivity_tensor_gsa(
    matrix_value: float,
    fluid_value: float,
    porosity: float,
    *,
    aspect_ratio: float,
    orientation_order: float = 1.0,
    ind_friab: int = 11,
    friab: float = 0.0,
) -> AxisymmetricThermalConductivity:
    """
    Tensor GSA thermal conductivity.

    orientation_order:
        0.0 -> random pore orientation
        1.0 -> perfectly aligned pore fabric
    """
    if not 0.0 <= orientation_order <= 1.0:
        raise ValueError("orientation_order must lie in [0, 1].")

    aligned = _run_tensor_gsa(
        matrix_value,
        fluid_value,
        porosity,
        aspect_ratio=aspect_ratio,
        fluid_orientation_flag=0,
        ind_friab=ind_friab,
        friab=friab,
    )

    random_ = _run_tensor_gsa(
        matrix_value,
        fluid_value,
        porosity,
        aspect_ratio=aspect_ratio,
        fluid_orientation_flag=1,
        ind_friab=ind_friab,
        friab=friab,
    )

    blended = random_ + orientation_order * (aligned - random_)

    return AxisymmetricThermalConductivity(
        lambda_parallel=float(blended[2, 2]),
        lambda_perpendicular=float(0.5 * (blended[0, 0] + blended[1, 1])),
    )