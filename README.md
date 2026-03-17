# RockPhysX

`RockPhysX` is a Python package for forward modeling, inverse calibration, and cross-property prediction of sedimentary rocks. The repository is structured around the workflow logic of the dissertation:

**models -> forward problem -> inverse problem -> cross-property prediction**

This first implementation pass is intentionally small but working. It establishes the package architecture, typed scientific inputs, a reproducible forward API, inverse calibration for a constant effective aspect ratio, and a first cross-property workflow.

## Scientific purpose

The package targets effective-property modeling of porous sedimentary rocks with shared microstructural parameters:

- transport properties:
  - thermal conductivity,
  - electrical conductivity / resistivity,
  - permeability;
- elastic properties:
  - bulk modulus,
  - shear modulus,
  - P-wave velocity,
  - S-wave velocity.

The current implementation focuses on isotropic forward calculations while keeping the API and parameter containers compatible with future anisotropic extensions.

## Implemented in this milestone

### Core data model
- typed dataclasses for minerals, fluids, matrix properties, microstructure, and samples;
- explicit saturation states: `dry`, `brine`, `oil`;
- validation utilities for fractions and positive properties.

### Effective-medium and transport models
- Wiener bounds and geometric mean;
- Hashin-Shtrikman bounds for scalar transport properties;
- Maxwell-Garnett scalar mixing;
- Bruggeman scalar EMA;
- generalized self-consistent transport model (GSA-inspired, isotropic scalar implementation);
- thermal conductivity wrapper;
- electrical conductivity / resistivity wrapper;
- Kozeny-Carman permeability baseline.

### Elastic models
- Voigt-Reuss-Hill mineral mixing;
- critical-porosity dry moduli;
- Gassmann saturation;
- velocity calculation from saturated moduli and bulk density.

### Workflow layer
- unified `ForwardSolver.predict(...)` API;
- weighted least-squares inversion for constant aspect ratio `alpha`;
- cross-property workflow A1:
  1. calibrate `alpha` from measured thermal conductivity,
  2. reuse `alpha` to predict additional properties.

### Reproducibility
- tests for API consistency, sanity checks, and synthetic inverse round-trip;
- example script in `examples/basic_workflow.py`.

## What is planned next

- distribution-based aspect-ratio parameterization `(gamma, delta)`;
- joint inversion A2 for `(alpha, connectivity)`;
- richer anisotropic tensor workflows;
- additional elastic EMT models (DEM, Kuster-Toksoz, Hudson, Brown-Korringa);
- more permeability models and calibration utilities;
- notebooks reproducing dissertation figures.

## Design notes

Two sources informed this initial scaffold:

1. Your uploaded notebook (`MODELS_TEST.ipynb`), which contains thermal conductivity implementations and tests for Wiener, Hashin-Shtrikman, Bruggeman, Maxwell-Garnett, and a generalized self-consistent model.
2. The open-source `rockphypy` project, which organizes rock-physics models in reusable Python modules and includes baseline elastic relations such as critical porosity and Gassmann. The present repository reimplements selected formulas in a cleaner dissertation-specific architecture rather than copying the package layout wholesale.

Where formulas were adapted from `rockphypy`, the relevant functions include attribution in comments/docstrings.

## Quickstart

```python
from rockphysx.core.parameters import MineralPhase, FluidPhase, MatrixProperties, MicrostructureParameters
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver

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
thermal_brine = solver.predict("thermal_conductivity", sample, SaturationState.BRINE)
vp_brine = solver.predict("vp", sample, SaturationState.BRINE)
print(thermal_brine, vp_brine)
```

## Running tests

```bash
pytest
```

## Repository layout

```text
src/rockphysx/
  core/            typed scientific inputs and base abstractions
  models/          EMT, transport, elastic, fluid, and matrix models
  forward/         unified forward solver
  inverse/         objective functions and optimizers
  cross_property/  workflow-style prediction strategies
  utils/           validation, I/O, and plotting helpers
tests/
examples/
notebooks/
```
