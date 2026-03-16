from rockphysx.core.parameters import FluidPhase, MatrixProperties, MicrostructureParameters, MineralPhase
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.cross_property.approach_a1 import thermal_only_calibration_then_predict
from rockphysx.forward.solver import ForwardSolver


def main() -> None:
    quartz = MineralPhase(
        name="quartz",
        volume_fraction=1.0,
        bulk_modulus_gpa=37.0,
        shear_modulus_gpa=44.0,
        density_gcc=2.65,
        thermal_conductivity_wmk=6.5,
        electrical_conductivity_sm=1e-8,
    )
    matrix = MatrixProperties.from_minerals([quartz])
    fluids = {
        SaturationState.DRY: FluidPhase.air(),
        SaturationState.BRINE: FluidPhase.brine(),
        SaturationState.OIL: FluidPhase.oil(),
    }
    sample = SampleDescription(
        name="synthetic_sandstone",
        porosity=0.18,
        minerals=[quartz],
        matrix=matrix,
        fluids=fluids,
        microstructure=MicrostructureParameters(aspect_ratio=0.12, connectivity=0.9),
    )

    solver = ForwardSolver()

    print("Forward predictions")
    for state in (SaturationState.DRY, SaturationState.BRINE, SaturationState.OIL):
        thermal = solver.predict("thermal_conductivity", sample, state)
        vp = solver.predict("vp", sample, state)
        sigma = solver.predict("electrical_conductivity", sample, state)
        print(f"  {state.value:>5s} | thermal={thermal:.4f} W/mK | vp={vp:.1f} m/s | sigma={sigma:.6e} S/m")

    measured_thermal = {
        SaturationState.DRY: solver.predict("thermal_conductivity", sample, SaturationState.DRY),
        SaturationState.BRINE: solver.predict("thermal_conductivity", sample, SaturationState.BRINE),
    }

    inverse_sample = SampleDescription(
        name="inverse_case",
        porosity=sample.porosity,
        minerals=sample.minerals,
        matrix=sample.matrix,
        fluids=sample.fluids,
        microstructure=MicrostructureParameters(aspect_ratio=0.50, connectivity=0.9),
    )

    result = thermal_only_calibration_then_predict(
        inverse_sample,
        measured_thermal,
        target_properties=("electrical_conductivity", "electrical_resistivity", "vp"),
        target_saturations=(SaturationState.BRINE,),
        solver=solver,
    )

    print("\nCross-property A1")
    print(f"  calibrated alpha = {result.calibration.alpha_hat:.4f}")
    for (property_name, saturation), value in result.predictions.items():
        print(f"  predicted {property_name} at {saturation.value}: {value:.6g}")


if __name__ == "__main__":
    main()
