from __future__ import annotations

import numpy as np


def test_elastic_methods_phi_zero_and_positive() -> None:
    from rockphysx.models.emt.kt_elastic import kt_elastic_pores
    from rockphysx.models.emt.mt_elastic import mt_elastic_pores
    from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores
    from rockphysx.models.emt.dem_transport_elastic import dem_elastic_moduli

    Km = 77.78
    Gm = 31.82
    alpha = 0.2

    methods = {
        "MT": lambda phi: mt_elastic_pores(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=0.0),
        "KT": lambda phi: kt_elastic_pores(Km, Gm, phi, aspect_ratio=alpha, pore_bulk_gpa=0.0),
        "SC": lambda phi: berryman_self_consistent_spheroidal_pores(
            Km, Gm, phi, pore_bulk_gpa=0.0, aspect_ratio=alpha, relaxation=0.8, max_iter=4000
        ),
        "DEM": lambda phi: (
            dem_elastic_moduli(Km, Gm, 0.0, 0.0, phi, aspect_ratio=alpha, n_steps=400).bulk_modulus,
            dem_elastic_moduli(Km, Gm, 0.0, 0.0, phi, aspect_ratio=alpha, n_steps=400).shear_modulus,
        ),
    }

    for name, fn in methods.items():
        k0, g0 = fn(0.0)
        assert np.isfinite(k0) and np.isfinite(g0)
        assert abs(k0 - Km) / Km < 1e-10, name
        assert abs(g0 - Gm) / Gm < 1e-10, name

        k1, g1 = fn(0.1)
        assert np.isfinite(k1) and np.isfinite(g1)
        assert k1 > 0.0 and g1 > 0.0
        assert k1 < Km + 1e-9, name
        assert g1 < Gm + 1e-9, name


def test_elastic_methods_monotone_decreasing_vs_phi() -> None:
    from rockphysx.models.emt.kt_elastic import kt_elastic_pores
    from rockphysx.models.emt.mt_elastic import mt_elastic_pores
    from rockphysx.models.emt.sca_elastic import berryman_self_consistent_spheroidal_pores

    Km = 77.78
    Gm = 31.82
    alpha = 0.1
    # KT is a first-order scattering approximation and can become non-physical at
    # higher porosity / crack-like aspect ratios. For monotonicity checks we use
    # a conservative porosity range for KT, while keeping the full range for MT/SC.
    phis_full = np.linspace(0.0, 0.30, 31)
    phis_kt = np.linspace(0.0, 0.20, 21)

    def curve(method: str) -> tuple[np.ndarray, np.ndarray]:
        if method == "MT":
            ks, gs = zip(*(mt_elastic_pores(Km, Gm, float(p), aspect_ratio=alpha) for p in phis_full), strict=True)
        elif method == "KT":
            ks, gs = zip(*(kt_elastic_pores(Km, Gm, float(p), aspect_ratio=alpha) for p in phis_kt), strict=True)
        elif method == "SC":
            warm = None
            out = []
            for p in phis_full:
                warm = berryman_self_consistent_spheroidal_pores(
                    Km,
                    Gm,
                    float(p),
                    pore_bulk_gpa=0.0,
                    aspect_ratio=alpha,
                    # Use the library's fallback damping (omega=1.0 branch) for robustness
                    relaxation=1.0,
                    max_iter=8000,
                    initial_guess_gpa=warm,
                )
                out.append(warm)
            ks, gs = zip(*out, strict=True)
        else:
            raise ValueError(method)
        return np.asarray(ks, float), np.asarray(gs, float)

    for m in ["MT", "KT", "SC"]:
        ks, gs = curve(m)
        # allow tiny numerical noise, but overall must be non-increasing
        assert np.all(np.diff(ks) <= 1e-8), m
        assert np.all(np.diff(gs) <= 1e-8), m
