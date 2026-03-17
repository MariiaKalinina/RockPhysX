from __future__ import annotations

"""Sensitivity-analysis helpers for EMT-based thermal-conductivity workflows.

The tools in this module are written around the current RockPhysX forward solver
and inversion abstractions. They are intended for dissertation-style analyses of
parameter influence, uncertainty propagation, and inversion identifiability for
models of the form

    lambda_star = F(phi, lambda_M, lambda_f, alpha)

where the forward operator is evaluated through :class:`rockphysx.forward.solver.ForwardSolver`.
"""

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from rockphysx.core.parameters import MatrixProperties
from rockphysx.core.sample import SampleDescription
from rockphysx.core.saturation import SaturationState
from rockphysx.forward.solver import ForwardSolver
from rockphysx.inverse.objective import Observation
from rockphysx.utils.validation import ensure_fraction, ensure_positive

THERMAL_PARAMETER_NAMES = ("porosity", "matrix_conductivity", "fluid_conductivity", "aspect_ratio")


@dataclass(slots=True, frozen=True)
class LocalSensitivityResult:
    """Single local sensitivity estimate for one parameter and saturation state."""

    saturation: SaturationState
    parameter: str
    baseline_parameter_value: float
    baseline_prediction: float
    perturbed_minus_parameter_value: float
    perturbed_plus_parameter_value: float
    perturbed_minus_prediction: float
    perturbed_plus_prediction: float
    normalized_sensitivity: float
    absolute_sensitivity: float
    epsilon: float


@dataclass(slots=True, frozen=True)
class MisfitGridResult:
    """2D misfit grid in (alpha, lambda_M) space."""

    alpha_values: np.ndarray
    matrix_conductivity_values: np.ndarray
    misfit: np.ndarray
    normalized: bool
    power: int
    saturation_labels: tuple[str, ...]



def _replace_porosity(sample: SampleDescription, porosity: float) -> SampleDescription:
    return replace(sample, porosity=ensure_fraction(porosity, "porosity"))



def _replace_matrix_conductivity(sample: SampleDescription, matrix_conductivity: float) -> SampleDescription:
    matrix = sample.matrix_properties
    explicit_matrix = replace(
        matrix,
        thermal_conductivity_wmk=ensure_positive(matrix_conductivity, "matrix_conductivity"),
    )
    return replace(sample, matrix=explicit_matrix)



def _replace_fluid_conductivity(
    sample: SampleDescription,
    saturation: SaturationState,
    fluid_conductivity: float,
) -> SampleDescription:
    fluids = dict(sample.fluids)
    target = sample.fluid_for(saturation)
    fluids[saturation] = replace(
        target,
        thermal_conductivity_wmk=ensure_positive(fluid_conductivity, "fluid_conductivity"),
    )
    return replace(sample, fluids=fluids)



def _replace_aspect_ratio(sample: SampleDescription, aspect_ratio: float) -> SampleDescription:
    return replace(
        sample,
        microstructure=replace(
            sample.microstructure,
            aspect_ratio=ensure_positive(aspect_ratio, "aspect_ratio"),
        ),
    )



def _thermal_parameter_value(
    sample: SampleDescription,
    saturation: SaturationState,
    parameter: str,
) -> float:
    key = parameter.lower()
    if key == "porosity":
        return float(sample.porosity)
    if key == "matrix_conductivity":
        return float(sample.matrix_properties.thermal_conductivity_wmk)
    if key == "fluid_conductivity":
        return float(sample.fluid_for(saturation).thermal_conductivity_wmk)
    if key == "aspect_ratio":
        return float(sample.microstructure.aspect_ratio)
    raise ValueError(f"Unsupported thermal sensitivity parameter {parameter!r}.")



def _perturb_sample(
    sample: SampleDescription,
    saturation: SaturationState,
    parameter: str,
    value: float,
) -> SampleDescription:
    key = parameter.lower()
    if key == "porosity":
        return _replace_porosity(sample, value)
    if key == "matrix_conductivity":
        return _replace_matrix_conductivity(sample, value)
    if key == "fluid_conductivity":
        return _replace_fluid_conductivity(sample, saturation, value)
    if key == "aspect_ratio":
        return _replace_aspect_ratio(sample, value)
    raise ValueError(f"Unsupported thermal sensitivity parameter {parameter!r}.")



def build_thermal_observations(
    measured_by_saturation: Mapping[SaturationState, float],
    *,
    weight: float = 1.0,
    model: str = "gsa",
) -> list[Observation]:
    """Build thermal-conductivity observations for one or more saturation states."""
    return [
        Observation(
            property_name="thermal_conductivity",
            saturation=saturation,
            measured_value=float(value),
            weight=weight,
            model=model,
        )
        for saturation, value in measured_by_saturation.items()
    ]



def local_normalized_sensitivity(
    sample: SampleDescription,
    saturation: SaturationState,
    parameter: str,
    *,
    epsilon: float = 0.01,
    solver: ForwardSolver | None = None,
    model: str = "gsa",
) -> LocalSensitivityResult:
    """Estimate local normalized and absolute sensitivities by central differences.

    The normalized sensitivity is computed as

        S_p ≈ [ln F(p(1+ε)) - ln F(p(1-ε))] / (2 ε)

    which is a central finite-difference approximation to

        ∂ ln(lambda*) / ∂ ln(p).
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1).")

    forward = solver or ForwardSolver()
    base_parameter_value = _thermal_parameter_value(sample, saturation, parameter)
    if base_parameter_value <= 0.0:
        raise ValueError(
            f"Local normalized sensitivity for {parameter!r} requires a positive base value, "
            f"got {base_parameter_value!r}."
        )

    p_minus = base_parameter_value * (1.0 - epsilon)
    p_plus = base_parameter_value * (1.0 + epsilon)

    sample_minus = _perturb_sample(sample, saturation, parameter, p_minus)
    sample_plus = _perturb_sample(sample, saturation, parameter, p_plus)

    lambda_base = forward.predict("thermal_conductivity", sample, saturation, model=model)
    lambda_minus = forward.predict("thermal_conductivity", sample_minus, saturation, model=model)
    lambda_plus = forward.predict("thermal_conductivity", sample_plus, saturation, model=model)

    if lambda_minus <= 0.0 or lambda_plus <= 0.0 or lambda_base <= 0.0:
        raise RuntimeError("Thermal-conductivity sensitivity requires strictly positive predictions.")

    normalized = (np.log(lambda_plus) - np.log(lambda_minus)) / (2.0 * epsilon)
    absolute = (lambda_plus - lambda_minus) / (p_plus - p_minus)

    return LocalSensitivityResult(
        saturation=saturation,
        parameter=parameter,
        baseline_parameter_value=float(base_parameter_value),
        baseline_prediction=float(lambda_base),
        perturbed_minus_parameter_value=float(p_minus),
        perturbed_plus_parameter_value=float(p_plus),
        perturbed_minus_prediction=float(lambda_minus),
        perturbed_plus_prediction=float(lambda_plus),
        normalized_sensitivity=float(normalized),
        absolute_sensitivity=float(absolute),
        epsilon=float(epsilon),
    )



def compute_local_sensitivities(
    sample: SampleDescription,
    saturation_states: Iterable[SaturationState],
    *,
    epsilon: float = 0.01,
    parameters: Sequence[str] = THERMAL_PARAMETER_NAMES,
    solver: ForwardSolver | None = None,
    model: str = "gsa",
) -> list[LocalSensitivityResult]:
    """Compute local sensitivities for several parameters and saturation states."""
    forward = solver or ForwardSolver()
    results: list[LocalSensitivityResult] = []
    for saturation in saturation_states:
        for parameter in parameters:
            results.append(
                local_normalized_sensitivity(
                    sample,
                    saturation,
                    parameter,
                    epsilon=epsilon,
                    solver=forward,
                    model=model,
                )
            )
    return results



def sensitivity_row_vector(
    sample: SampleDescription,
    saturation: SaturationState,
    *,
    epsilon: float = 0.01,
    parameters: Sequence[str] = THERMAL_PARAMETER_NAMES,
    solver: ForwardSolver | None = None,
    model: str = "gsa",
) -> np.ndarray:
    """Return the row vector J = [S_phi, S_lambdaM, S_lambdaf, S_alpha]."""
    sensitivities = compute_local_sensitivities(
        sample,
        [saturation],
        epsilon=epsilon,
        parameters=parameters,
        solver=solver,
        model=model,
    )
    return np.asarray([entry.normalized_sensitivity for entry in sensitivities], dtype=float)



def propagate_relative_uncertainty_independent(
    sensitivity_map: Mapping[str, float],
    relative_uncertainty_map: Mapping[str, float],
) -> float:
    """Approximate relative uncertainty of thermal conductivity for independent inputs.

    Implements

        (δλ*/λ*)² ≈ Σ_p (S_p δp/p)².
    """
    total = 0.0
    for key, sensitivity in sensitivity_map.items():
        rel_unc = float(relative_uncertainty_map[key])
        total += (float(sensitivity) * rel_unc) ** 2
    return float(np.sqrt(total))



def propagate_log_variance(
    sensitivity_vector: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """Return Var[ln(lambda*)] ≈ J C Jᵀ for correlated input uncertainties."""
    J = np.asarray(sensitivity_vector, dtype=float).reshape(1, -1)
    C = np.asarray(covariance_matrix, dtype=float)
    if C.shape != (J.shape[1], J.shape[1]):
        raise ValueError("covariance_matrix must be square and consistent with sensitivity_vector length.")
    return float((J @ C @ J.T)[0, 0])



def thermal_misfit(
    sample: SampleDescription,
    observations: Sequence[Observation],
    *,
    aspect_ratio: float,
    matrix_conductivity: float,
    solver: ForwardSolver | None = None,
    normalized: bool = True,
    power: int = 2,
    model: str | None = None,
) -> float:
    """Evaluate misfit in (alpha, lambda_M) space for one or more saturation states.

    Parameters
    ----------
    normalized
        If True, use the dimensionless residual ((calc - exp) / exp).
        If False, use the dimensional residual (calc - exp).
    power
        Power in the aggregated objective. Use 1 for absolute-value-type profiles
        and 2 for least-squares-type maps.
    """
    if power not in {1, 2}:
        raise ValueError("power must be 1 or 2.")

    forward = solver or ForwardSolver()
    modified = _replace_matrix_conductivity(sample, matrix_conductivity)
    modified = _replace_aspect_ratio(modified, aspect_ratio)

    total = 0.0
    for obs in observations:
        predicted = forward.predict(
            obs.property_name,
            modified,
            obs.saturation,
            model=obs.model or model or "gsa",
        )
        residual = predicted - obs.measured_value
        if normalized:
            residual /= obs.measured_value
        if power == 1:
            contribution = abs(residual)
        else:
            contribution = residual**2
        total += obs.weight * contribution
    return float(total)



def compute_alpha_lambda_misfit_grid(
    sample: SampleDescription,
    observations: Sequence[Observation],
    alpha_values: Sequence[float],
    matrix_conductivity_values: Sequence[float],
    *,
    solver: ForwardSolver | None = None,
    normalized: bool = True,
    power: int = 2,
    model: str | None = None,
) -> MisfitGridResult:
    """Build a 2D misfit map in (alpha, lambda_M) space."""
    alpha_arr = np.asarray(alpha_values, dtype=float)
    lambda_m_arr = np.asarray(matrix_conductivity_values, dtype=float)
    grid = np.empty((len(lambda_m_arr), len(alpha_arr)), dtype=float)

    for i, lambda_m in enumerate(lambda_m_arr):
        for j, alpha in enumerate(alpha_arr):
            grid[i, j] = thermal_misfit(
                sample,
                observations,
                aspect_ratio=float(alpha),
                matrix_conductivity=float(lambda_m),
                solver=solver,
                normalized=normalized,
                power=power,
                model=model,
            )

    return MisfitGridResult(
        alpha_values=alpha_arr,
        matrix_conductivity_values=lambda_m_arr,
        misfit=grid,
        normalized=normalized,
        power=power,
        saturation_labels=tuple(obs.saturation.value for obs in observations),
    )



def profile_misfit_over_alpha(
    sample: SampleDescription,
    observations: Sequence[Observation],
    alpha_values: Sequence[float],
    matrix_conductivity_values: Sequence[float],
    *,
    solver: ForwardSolver | None = None,
    normalized: bool = True,
    power: int = 2,
    model: str | None = None,
) -> dict[float, np.ndarray]:
    """Return profile misfit curves versus alpha for fixed matrix conductivities."""
    alpha_arr = np.asarray(alpha_values, dtype=float)
    profiles: dict[float, np.ndarray] = {}

    for lambda_m in matrix_conductivity_values:
        profile = [
            thermal_misfit(
                sample,
                observations,
                aspect_ratio=float(alpha),
                matrix_conductivity=float(lambda_m),
                solver=solver,
                normalized=normalized,
                power=power,
                model=model,
            )
            for alpha in alpha_arr
        ]
        profiles[float(lambda_m)] = np.asarray(profile, dtype=float)
    return profiles
