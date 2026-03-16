from __future__ import annotations

import numpy as np
from scipy.linalg import inv


class EffectiveConductivityModel:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.lambda_vals = [np.eye(3) for _ in range(n_components)]
        self.ar_dim = [1.0] * n_components
        self.porosities = [0.0] * n_components
        self.orientations = [0] * n_components

    def set_component_properties(self, component_idx, conductivity, aspect_ratio, porosity, orientation=0):
        if isinstance(conductivity, (int, float)):
            self.lambda_vals[component_idx] = np.eye(3) * conductivity
        else:
            self.lambda_vals[component_idx] = np.asarray(conductivity, dtype=float)

        if isinstance(aspect_ratio, (list, np.ndarray)):
            self.ar_dim[component_idx] = aspect_ratio
        else:
            self.ar_dim[component_idx] = [aspect_ratio] * 3

        self.porosities[component_idx] = porosity
        self.orientations[component_idx] = orientation

    def calculate(self, friab=0.0, ind_friab=11):
        total_porosity = sum(self.porosities[1:])
        lambda0 = np.zeros((3, 3), dtype=float)

        if ind_friab == 11:
            lambda0 = self.lambda_vals[0].copy()
        elif ind_friab == 12:
            lambda0 = (1 - total_porosity) * self.lambda_vals[0]
            for i in range(1, self.n_components):
                lambda0 += self.porosities[i] * self.lambda_vals[i]
        elif ind_friab == 17:
            lambda_R = self._reuss_average()
            lambda_V = self._voigt_average()
            lambda0 = lambda_R * (1 - friab * total_porosity) + lambda_V * friab * total_porosity
        else:
            raise ValueError(f"Unsupported ind_friab={ind_friab!r}.")

        F = [self._calculate_form_factor(ar) for ar in self.ar_dim]

        sumc = np.zeros((3, 3), dtype=float)
        sumz = np.zeros((3, 3), dtype=float)

        for i in range(self.n_components):
            sumc_i, sumz_i = self._parts(lambda0, self.lambda_vals[i], F[i])

            if self.orientations[i] == 1:
                trace_sumc = np.trace(sumc_i) / 3.0
                trace_sumz = np.trace(sumz_i) / 3.0
                sumc_i_rot = np.eye(3) * trace_sumc
                sumz_i_rot = np.eye(3) * trace_sumz
            else:
                sumc_i_rot = sumc_i
                sumz_i_rot = sumz_i

            sumc += self.porosities[i] * sumc_i_rot
            sumz += self.porosities[i] * sumz_i_rot

        return sumc @ inv(sumz)

    def _calculate_form_factor(self, ar):
        if isinstance(ar, (list, np.ndarray)):
            ar = ar[0]

        if ar < 1:
            t1 = ar**2
            t2 = 1 / t1
            t4 = np.sqrt(t2 - 1)
            t5 = np.arctan(t4)
            t8 = t4**2
            return t2 * (t4 - t5) / t8 / t4
        elif ar > 1:
            t1 = ar**2
            t2 = 1 / t1
            t4 = np.sqrt(1 - t2)
            t6 = np.log(1 + t4)
            t9 = np.log(1 - t4)
            t13 = t4**2
            return t2 * (t6 / 2 - t9 / 2 - t4) / t13 / t4
        else:
            return 1 / 3

    def _parts(self, lambda0, lambda_i, F_i):
        F_ten = np.diag([(1 - F_i) / 2, (1 - F_i) / 2, F_i])
        I = np.eye(3)
        skob = I - F_ten

        S1 = lambda0 @ skob
        S2 = lambda_i @ F_ten
        S3 = S1 + S2

        S3_inv = inv(S3)
        sumz_i = S3_inv
        sumc_i = lambda_i @ S3_inv
        return sumc_i, sumz_i

    def _reuss_average(self):
        inv_sum = np.zeros((3, 3), dtype=float)
        for i in range(self.n_components):
            inv_sum += self.porosities[i] * inv(self.lambda_vals[i])
        return inv(inv_sum)

    def _voigt_average(self):
        sum_ = np.zeros((3, 3), dtype=float)
        for i in range(self.n_components):
            sum_ += self.porosities[i] * self.lambda_vals[i]
        return sum_


def gsa_effective_property(
    volume_fractions,
    thermal_conductivities,
    aspect_ratios,
    *,
    ind_friab: int = 11,
    friab: float = 0.0,
) -> float:
    """
    Isotropic scalar wrapper around the tensor GSA model.

    For isotropic output, the result is taken as the mean diagonal value.
    """
    phi = np.asarray(volume_fractions, dtype=float)
    lam = np.asarray(thermal_conductivities, dtype=float)
    ar = np.asarray(aspect_ratios, dtype=float)

    if not (len(phi) == len(lam) == len(ar)):
        raise ValueError("volume_fractions, thermal_conductivities, and aspect_ratios must have the same length.")

    model = EffectiveConductivityModel(n_components=len(phi))
    for i in range(len(phi)):
        model.set_component_properties(
            i,
            lam[i],
            ar[i],
            phi[i],
            orientation=1,
        )

    result = model.calculate(friab=friab, ind_friab=ind_friab)
    return float(np.trace(result) / 3.0)