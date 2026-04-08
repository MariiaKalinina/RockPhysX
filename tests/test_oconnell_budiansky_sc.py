import numpy as np

from rockphysx.models.emt.sca_elastic import (
    oc_budiansky_sc_penny_cracks_from_phi_alpha,
    penny_crack_density_from_porosity,
    sca_elastic_crack_like_pores,
)


def test_penny_crack_density_matches_phi_alpha_relation():
    phi = 0.12
    alpha = 1e-2
    eps = penny_crack_density_from_porosity(phi, alpha)
    phi_back = (4.0 / 3.0) * np.pi * alpha * eps
    assert np.isclose(phi_back, phi)


def test_ocb_sc_limits_are_sensible_dry():
    Km, Gm = 76.8, 32.0
    alpha = 1e-2

    K0, G0 = oc_budiansky_sc_penny_cracks_from_phi_alpha(Km, Gm, 0.0, alpha, fluid_bulk_gpa=0.0)
    assert np.isclose(K0, Km)
    assert np.isclose(G0, Gm)

    # At moderate porosity, moduli must decrease and remain non-negative.
    K1, G1 = oc_budiansky_sc_penny_cracks_from_phi_alpha(Km, Gm, 0.05, alpha, fluid_bulk_gpa=0.0)
    assert 0.0 <= K1 < Km
    assert 0.0 <= G1 < Gm


def test_ocb_sc_limits_are_sensible_fluid():
    Km, Gm = 76.8, 32.0
    Kf = 2.2
    alpha = 1e-2

    K0, G0 = oc_budiansky_sc_penny_cracks_from_phi_alpha(Km, Gm, 0.0, alpha, fluid_bulk_gpa=Kf)
    assert np.isclose(K0, Km)
    assert np.isclose(G0, Gm)

    K1, G1 = oc_budiansky_sc_penny_cracks_from_phi_alpha(Km, Gm, 0.05, alpha, fluid_bulk_gpa=Kf)
    assert 0.0 <= K1 < Km
    assert 0.0 <= G1 < Gm


def test_crack_like_wrapper_matches_ocb():
    Km, Gm = 76.8, 32.0
    alpha = 1e-3
    phi = 0.08

    K0, G0 = oc_budiansky_sc_penny_cracks_from_phi_alpha(Km, Gm, phi, alpha, fluid_bulk_gpa=0.0)
    K1, G1 = sca_elastic_crack_like_pores(Km, Gm, phi, aspect_ratio=alpha, fluid_bulk_gpa=0.0)
    assert np.isclose(K0, K1)
    assert np.isclose(G0, G1)
