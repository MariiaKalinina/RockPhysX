import numpy as np

from rockphysx.models.emt import gsa_transport as m


def test_phi_to_zero():
    val = m.two_phase_thermal_isotropic(3.0, 0.025, 1e-6, aspect_ratio=0.1, comparison='matrix', n_orientation=8, n_theta=8, n_phi=16)
    assert abs(val - 3.0) < 1e-3


def test_sphere_random_isotropic_tensor():
    T = m.homogenize_transport_gsa(
        [
            m.make_phase('matrix', 0.9, 3.0, aspect_ratio=1.0),
            m.make_phase('pores', 0.1, 0.025, aspect_ratio=1.0, orientation='random'),
        ],
        m.ComparisonBody(kind='matrix'),
        n_orientation=20, n_theta=10, n_phi=20,
    )
    assert np.allclose(T[0,0], T[1,1], atol=1e-3)
    assert np.allclose(T[1,1], T[2,2], atol=1e-3)


def test_aligned_ti_is_anisotropic():
    res = m.two_phase_thermal_ti(3.0, 0.025, 0.1, aspect_ratio=0.01, comparison='matrix', n_theta=12, n_phi=24)
    assert abs(res.lambda_parallel - res.lambda_perpendicular) > 1e-3
