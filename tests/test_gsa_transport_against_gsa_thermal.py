import numpy as np

from rockphysx.models.emt import gsa_transport
from rockphysx.models.emt.gsa_thermal import gsa_effective_property
from rockphysx.models.transport.thermal_tensor import thermal_conductivity_tensor_gsa


def test_gsa_transport_isotropic_random_matches_gsa_thermal_matrix_comparison() -> None:
    # Two-phase isotropic random case: scalar outputs should match exactly
    # (both use the same depolarization factor closed form for spheroids).
    matrix = 3.0
    fluid = 0.6
    for porosity in (0.05, 0.2, 0.35):
        for alpha in (1.0, 1e-3, 1e-2, 1e-1, 10.0):
            t1 = gsa_effective_property(
                [1.0 - porosity, porosity],
                [matrix, fluid],
                [1.0, alpha],
                ind_friab=11,
            )
            t2 = gsa_transport.two_phase_thermal_isotropic(
                matrix,
                fluid,
                porosity,
                aspect_ratio=alpha,
                comparison="matrix",
                max_iter=1,
            )
            assert np.isclose(t1, t2, rtol=0.0, atol=1e-12)


def test_gsa_transport_aligned_ti_close_to_legacy_tensor_gsa() -> None:
    # Legacy tensor GSA (EffectiveConductivityModel) uses a closed-form depolarization
    # approximation for aligned pores, while gsa_transport uses angular quadrature.
    # They should agree closely.
    matrix = 3.0
    fluid = 0.6
    porosity = 0.3
    for alpha in (1.0, 1e-2, 1e-1, 10.0):
        legacy = thermal_conductivity_tensor_gsa(
            matrix,
            fluid,
            porosity,
            aspect_ratio=alpha,
            orientation_order=1.0,
            ind_friab=11,
        )
        new = gsa_transport.two_phase_thermal_ti(
            matrix,
            fluid,
            porosity,
            aspect_ratio=alpha,
            comparison="matrix",
            max_iter=1,
            n_orientation=1,  # aligned => orientation loop is trivial
            # Need higher angular resolution for crack-like pores (small alpha)
            # to match the closed-form legacy approximation closely.
            n_theta=240,
            n_phi=480,
        )
        assert np.isclose(legacy.lambda_parallel, new.lambda_parallel, rtol=2e-3, atol=0.0)
        assert np.isclose(legacy.lambda_perpendicular, new.lambda_perpendicular, rtol=2e-3, atol=0.0)
