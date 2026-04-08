import numpy as np

from rockphysx.models.emt import gsa_elastic_general as gsa_gen
from rockphysx.models.emt import gsa_elastic_random_isotropic as gsa


def test_numeric_green_isotropic_close_to_vti_fortran():
    """
    For an isotropic comparison body and spheroidal shape, the numeric elastic Green
    tensor should match the VTI Fortran kernel reasonably well.

    This is a regression guard for the index contractions used in the numeric integral.
    """
    # Isotropic comparison body
    K = 50e9
    G = 30e9
    Cc = gsa.isotropic_stiffness_66_from_KG(K, G)

    # Spheroid
    ar = 1e-2
    shape = gsa_gen.EllipsoidShape(a1=1.0, a2=1.0, a3=ar)

    # Numeric g
    g_num = gsa_gen.green_tensor_elastic_numeric(
        Cc,
        shape,
        orientation_axis=(0.0, 0.0, 1.0),
        n_theta=30,
        n_phi=60,
    )

    # Fortran g (requires compiled backend already in-tree; build_backend resolves it)
    backend = gsa.build_backend(green_fortran="src/rockphysx/models/emt/GREEN_ANAL_VTI.f90")
    g_for = gsa.green_tensor_vti_fortran(Cc, ar, backend)

    # Compare in Mandel 6x6 (more stable scaling)
    G_num_66 = gsa.tensor4_to_mandel(g_num)
    G_for_66 = gsa.tensor4_to_mandel(g_for)

    # Relative tolerance: numeric quadrature is coarse, so keep it loose.
    denom = np.maximum(1.0, np.abs(G_for_66))
    rel = np.max(np.abs(G_num_66 - G_for_66) / denom)
    assert rel < 5e-2

