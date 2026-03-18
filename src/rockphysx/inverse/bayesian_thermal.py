from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from rockphysx.core.sample import SampleDescription
from rockphysx.models.transport.thermal import thermal_conductivity


@dataclass(slots=True, frozen=True)
class ThermalDatum:
    """
    One thermal-conductivity observation for one sample and one fluid state.

    label
        Human-readable state label, e.g. 'dry', 'oil', 'brine_0_6', 'brine_6'.
    measured_tc
        Observed effective thermal conductivity.
    fluid_conductivity
        Thermal conductivity of the pore fluid for this state.
    relative_sigma
        Relative observational standard deviation, e.g. 0.02 for 2%.
    weight
        Optional extra weighting factor.
    """
    label: str
    measured_tc: float
    fluid_conductivity: float
    relative_sigma: float = 0.03
    weight: float = 1.0


@dataclass(slots=True, frozen=True)
class PosteriorGridResult:
    log10_alpha_values: np.ndarray
    alpha_values: np.ndarray
    matrix_conductivity_values: np.ndarray
    log_posterior: np.ndarray
    posterior: np.ndarray
    map_log10_alpha: float
    map_alpha: float
    map_matrix_conductivity: float


def predict_tc_for_datum(
    sample: SampleDescription,
    datum: ThermalDatum,
    *,
    aspect_ratio: float,
    matrix_conductivity: float,
    model: str = "gsa",
) -> float:
    return float(
        thermal_conductivity(
            matrix_value=matrix_conductivity,
            fluid_value=datum.fluid_conductivity,
            porosity=sample.porosity,
            microstructure=sample.microstructure.__class__(
                aspect_ratio=aspect_ratio,
                connectivity=sample.microstructure.connectivity,
                orientation=sample.microstructure.orientation,
                topology=sample.microstructure.topology,
            ),
            model=model,
        )
    )


def log_likelihood_relative_gaussian(
    sample: SampleDescription,
    data: Sequence[ThermalDatum],
    *,
    log10_alpha: float,
    matrix_conductivity: float,
    model: str = "gsa",
) -> float:
    alpha = 10.0 ** log10_alpha
    total = 0.0

    for datum in data:
        pred = predict_tc_for_datum(
            sample,
            datum,
            aspect_ratio=alpha,
            matrix_conductivity=matrix_conductivity,
            model=model,
        )
        resid = (pred - datum.measured_tc) / datum.measured_tc
        sigma = float(datum.relative_sigma)
        total += datum.weight * (-0.5 * ((resid / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)))

    return float(total)


def log_uniform_box_prior(
    log10_alpha: float,
    matrix_conductivity: float,
    *,
    log10_alpha_min: float,
    log10_alpha_max: float,
    lambda_m_min: float,
    lambda_m_max: float,
) -> float:
    inside = (
        log10_alpha_min <= log10_alpha <= log10_alpha_max
        and lambda_m_min <= matrix_conductivity <= lambda_m_max
    )
    return 0.0 if inside else -np.inf


def compute_bayesian_posterior_grid(
    sample: SampleDescription,
    data: Sequence[ThermalDatum],
    log10_alpha_values: Sequence[float],
    matrix_conductivity_values: Sequence[float],
    *,
    log10_alpha_min: float,
    log10_alpha_max: float,
    lambda_m_min: float,
    lambda_m_max: float,
    model: str = "gsa",
) -> PosteriorGridResult:
    u = np.asarray(log10_alpha_values, dtype=float)
    lam_m = np.asarray(matrix_conductivity_values, dtype=float)

    logpost = np.empty((len(lam_m), len(u)), dtype=float)

    for i, lm in enumerate(lam_m):
        for j, uj in enumerate(u):
            lp = log_uniform_box_prior(
                uj,
                lm,
                log10_alpha_min=log10_alpha_min,
                log10_alpha_max=log10_alpha_max,
                lambda_m_min=lambda_m_min,
                lambda_m_max=lambda_m_max,
            )
            if np.isneginf(lp):
                logpost[i, j] = -np.inf
            else:
                logpost[i, j] = lp + log_likelihood_relative_gaussian(
                    sample,
                    data,
                    log10_alpha=uj,
                    matrix_conductivity=lm,
                    model=model,
                )

    # stabilize
    finite = np.isfinite(logpost)
    maxlog = np.max(logpost[finite])
    post = np.zeros_like(logpost)
    post[finite] = np.exp(logpost[finite] - maxlog)
    post /= np.sum(post)

    idx = np.unravel_index(np.argmax(post), post.shape)

    return PosteriorGridResult(
        log10_alpha_values=u,
        alpha_values=10.0 ** u,
        matrix_conductivity_values=lam_m,
        log_posterior=logpost,
        posterior=post,
        map_log10_alpha=float(u[idx[1]]),
        map_alpha=float(10.0 ** u[idx[1]]),
        map_matrix_conductivity=float(lam_m[idx[0]]),
    )


def marginal_alpha(posterior: PosteriorGridResult) -> np.ndarray:
    return np.sum(posterior.posterior, axis=0)


def marginal_lambda_m(posterior: PosteriorGridResult) -> np.ndarray:
    return np.sum(posterior.posterior, axis=1)