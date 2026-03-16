from __future__ import annotations

import numpy as np

from rockphysx.core.parameters import FluidPhase
from rockphysx.utils.validation import normalize_fractions


def lichtenecker_average(volume_fractions, values) -> float:
    phi = normalize_fractions(volume_fractions)
    vals = np.asarray(values, dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("Lichtenecker average requires strictly positive values.")
    return float(np.exp(np.sum(phi * np.log(vals))))


def arithmetic_average(volume_fractions, values) -> float:
    phi = normalize_fractions(volume_fractions)
    vals = np.asarray(values, dtype=float)
    return float(np.sum(phi * vals))


def wood_bulk_modulus(volume_fractions, bulk_moduli_gpa) -> float:
    phi = normalize_fractions(volume_fractions)
    vals = np.asarray(bulk_moduli_gpa, dtype=float)
    return float(1.0 / np.sum(phi / vals))


def mix_fluid_phases(phases, volume_fractions, *, name: str | None = None) -> FluidPhase:
    phi = normalize_fractions(volume_fractions)

    return FluidPhase(
        name=name or " + ".join(f"{v:.2f} {p.name}" for v, p in zip(phi, phases, strict=True)),
        bulk_modulus_gpa=wood_bulk_modulus(phi, [p.bulk_modulus_gpa for p in phases]),
        density_gcc=arithmetic_average(phi, [p.density_gcc for p in phases]),
        thermal_conductivity_wmk=lichtenecker_average(phi, [p.thermal_conductivity_wmk for p in phases]),
        electrical_conductivity_sm=lichtenecker_average(phi, [p.electrical_conductivity_sm for p in phases]),
        viscosity_pas=lichtenecker_average(phi, [p.viscosity_pas for p in phases if p.viscosity_pas is not None])
        if all(p.viscosity_pas is not None for p in phases)
        else None,
    )