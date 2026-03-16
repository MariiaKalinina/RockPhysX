from __future__ import annotations

from dataclasses import replace

from thesis_rp.core.parameters import MicrostructureParameters
from thesis_rp.core.sample import SampleDescription
from thesis_rp.core.saturation import SaturationState
from thesis_rp.models.elastic.moduli import saturated_elastic_properties
from thesis_rp.models.transport.electrical import electrical_conductivity, electrical_resistivity
from thesis_rp.models.transport.permeability import kozeny_carman_permeability
from thesis_rp.models.transport.thermal import thermal_conductivity


class ForwardSolver:
    """Unified forward API for dissertation workflows."""

    def predict(
        self,
        property_name: str,
        sample: SampleDescription,
        saturation: SaturationState,
        microstructure: MicrostructureParameters | None = None,
        *,
        model: str | None = None,
        grain_size_m: float = 250e-6,
        critical_porosity: float = 0.40,
    ) -> float:
        """Predict a single effective property."""
        matrix = sample.matrix_properties
        fluid = sample.fluid_for(saturation)
        micro = microstructure or sample.microstructure
        property_key = property_name.lower()

        if property_key == "thermal_conductivity":
            return thermal_conductivity(
                matrix.thermal_conductivity_wmk,
                fluid.thermal_conductivity_wmk,
                sample.porosity,
                micro,
                model=model or "gsa",
            )

        if property_key == "electrical_conductivity":
            return electrical_conductivity(
                matrix.electrical_conductivity_sm,
                fluid.electrical_conductivity_sm,
                sample.porosity,
                micro,
                model=model or "gsa",
            )

        if property_key == "electrical_resistivity":
            return electrical_resistivity(
                matrix.electrical_conductivity_sm,
                fluid.electrical_conductivity_sm,
                sample.porosity,
                micro,
                model=model or "gsa",
            )

        if property_key == "permeability":
            return kozeny_carman_permeability(
                sample.porosity,
                grain_size_m,
                micro,
            )

        if property_key in {"bulk_modulus", "shear_modulus", "vp", "vs", "density"}:
            state = saturated_elastic_properties(
                matrix.bulk_modulus_gpa,
                matrix.shear_modulus_gpa,
                matrix.density_gcc,
                fluid.bulk_modulus_gpa,
                fluid.density_gcc,
                sample.porosity,
                critical_porosity=critical_porosity,
            )
            mapping = {
                "bulk_modulus": state.bulk_modulus_gpa,
                "shear_modulus": state.shear_modulus_gpa,
                "vp": state.vp_mps,
                "vs": state.vs_mps,
                "density": state.density_gcc,
            }
            return mapping[property_key]

        raise ValueError(f"Unsupported property_name {property_name!r}.")

    def with_aspect_ratio(self, sample: SampleDescription, aspect_ratio: float) -> SampleDescription:
        """Return a shallow copy of the sample with updated microstructure aspect ratio."""
        new_micro = replace(sample.microstructure, aspect_ratio=aspect_ratio)
        return replace(sample, microstructure=new_micro)


def predict_property(
    property_name: str,
    sample: SampleDescription,
    saturation: SaturationState,
    microstructure: MicrostructureParameters | None = None,
    **kwargs,
) -> float:
    """Functional wrapper around :class:`ForwardSolver`."""
    return ForwardSolver().predict(
        property_name,
        sample,
        saturation,
        microstructure=microstructure,
        **kwargs,
    )
