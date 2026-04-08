import numpy as np

from rockphysx.models.emt.sca_thermal import sca_sc_effective_conductivity


def test_sca_sc_two_phase_limits():
    em, ei = 2.86, 0.6
    assert np.isclose(sca_sc_effective_conductivity(em, ei, 0.0, aspect_ratio=0.1), em)
    assert np.isclose(sca_sc_effective_conductivity(em, ei, 1.0, aspect_ratio=0.1), ei)


def test_sca_sc_is_bounded_and_positive_for_high_contrast():
    # Very high contrast, crack-like shape: should remain finite and non-negative.
    em, ei = 1e-6, 3.5
    for phi in [0.01, 0.05, 0.1, 0.2]:
        val = sca_sc_effective_conductivity(em, ei, phi, aspect_ratio=1e-4)
        assert np.isfinite(val)
        assert val >= 0.0
        assert val <= max(em, ei) * 1.000001

