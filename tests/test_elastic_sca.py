import numpy as np

from rockphysx.models.emt.sca_elastic import (
    berryman_self_consistent_spheroidal_pores,
    sca_elastic_crack_like_pores,
    sca_elastic_pores,
    sca_elastic_pores_unified,
)


def test_sca_elastic_pores_limits_are_sensible():
    Km, Gm = 37.0, 44.0

    K0, G0 = sca_elastic_pores(Km, Gm, 0.0, pore_bulk_gpa=0.0)
    assert K0 == Km
    assert np.isclose(G0, Gm)

    # With almost all pore phase, effective moduli approach pore phase (K->Kf, G->0).
    Kf = 2.2
    K1, G1 = sca_elastic_pores(Km, Gm, 0.999, pore_bulk_gpa=Kf)
    assert np.isfinite(K1) and np.isfinite(G1)
    assert K1 <= Km
    assert G1 <= Gm
    assert G1 >= 0.0


def test_sca_elastic_unified_matches_crack_like_branch_when_enabled():
    Km, Gm = 76.8, 32.0
    phi = 0.08
    alpha = 1e-3
    Kf = 2.2
    K0, G0 = sca_elastic_pores_unified(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=Kf, crack_like_threshold=1e-2)
    K1, G1 = sca_elastic_crack_like_pores(Km, Gm, phi, aspect_ratio=alpha, fluid_bulk_gpa=Kf)
    assert np.isclose(K0, K1)
    assert np.isclose(G0, G1)


def test_sca_elastic_unified_matches_spheroid_branch_for_larger_alpha():
    Km, Gm = 76.8, 32.0
    phi = 0.12
    alpha = 0.1
    Kf = 2.2
    K0, G0 = sca_elastic_pores_unified(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=Kf, crack_like_threshold=1e-3)
    K1, G1 = berryman_self_consistent_spheroidal_pores(Km, Gm, phi, pore_bulk_gpa=Kf, aspect_ratio=alpha)
    assert np.isfinite(K0) and np.isfinite(G0)
    assert np.isclose(K0, K1)
    assert np.isclose(G0, G1)
