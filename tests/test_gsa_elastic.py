import numpy as np

from rockphysx.models.emt import gsa_elastic as gsa
from rockphysx.models.emt import sca_elastic as sca


def test_phi_to_zero_matrix_comparison():
    K_eff, G_eff = gsa.two_phase_elastic_isotropic(
        matrix_bulk_gpa=76.0,
        matrix_shear_gpa=32.0,
        inclusion_bulk_gpa=0.0,
        inclusion_shear_gpa=0.0,
        inclusion_fraction=1e-6,
        aspect_ratio=0.01,
        comparison="matrix",
    )
    # For very crack-like aspect ratios the dilute sensitivity can be high;
    # this check just guards against large deviations at tiny porosity.
    assert abs(K_eff - 76.0) < 5e-2
    assert abs(G_eff - 32.0) < 5e-2


def test_spherical_sc_matches_sca_sphere():
    # For spheres, the isotropic GSA self-consistent reduces to the standard
    # spherical Berryman SC iteration implemented in sca_elastic.
    phi = 0.2
    K_eff_gsa, G_eff_gsa = gsa.two_phase_elastic_isotropic(
        matrix_bulk_gpa=76.0,
        matrix_shear_gpa=32.0,
        inclusion_bulk_gpa=0.0,
        inclusion_shear_gpa=0.0,
        inclusion_fraction=phi,
        aspect_ratio=1.0,
        comparison="self_consistent",
        tol=1e-12,
        max_iter=2000,
    )
    K_eff_sca, G_eff_sca = sca.sca_elastic_pores(76.0, 32.0, phi, pore_bulk_gpa=0.0, tol=1e-12, max_iter=2000)

    assert np.isclose(K_eff_gsa, K_eff_sca, rtol=0.0, atol=5e-6)
    assert np.isclose(G_eff_gsa, G_eff_sca, rtol=0.0, atol=5e-6)


def test_crack_like_decreases_with_porosity():
    K0, G0 = 76.0, 32.0
    # Avoid extremely thin cracks here to keep the fixed-point on a stable
    # branch in CI.
    ar = 1e-2

    K1, G1 = gsa.two_phase_elastic_isotropic(
        matrix_bulk_gpa=K0,
        matrix_shear_gpa=G0,
        inclusion_bulk_gpa=0.0,
        inclusion_shear_gpa=0.0,
        inclusion_fraction=0.05,
        aspect_ratio=ar,
        comparison="self_consistent",
        relaxation=0.5,
    )
    K2, G2 = gsa.two_phase_elastic_isotropic(
        matrix_bulk_gpa=K0,
        matrix_shear_gpa=G0,
        inclusion_bulk_gpa=0.0,
        inclusion_shear_gpa=0.0,
        inclusion_fraction=0.10,
        aspect_ratio=ar,
        comparison="self_consistent",
        relaxation=0.5,
    )
    assert K2 < K1 < K0
    assert G2 < G1 < G0
