from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_distribution

from rockphysx.core.parameters import AspectRatioBetaDistribution


def sample_aspect_ratio_distribution(
    params: AspectRatioBetaDistribution,
    *,
    n_samples: int = 200,
) -> np.ndarray:
    """Sample bounded aspect ratios from a beta distribution."""
    unit_samples = beta_distribution.rvs(params.gamma, params.delta, size=n_samples, random_state=1234)
    return params.lower + (params.upper - params.lower) * unit_samples
