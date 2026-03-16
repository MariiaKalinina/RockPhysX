from __future__ import annotations

from thesis_rp.core.parameters import MicrostructureParameters


def kozeny_carman_permeability(
    porosity: float,
    grain_size_m: float,
    microstructure: MicrostructureParameters,
) -> float:
    """Kozeny-Carman permeability with a simple connectivity multiplier.

    Notes
    -----
    The baseline Kozeny-Carman term follows the same functional form used in
    `rockphypy` for unconsolidated spheres. The connectivity factor is introduced
    here as a dissertation-oriented extension to support future joint inversion.
    """
    base = grain_size_m**2 / 180.0 * porosity**3 / (1.0 - porosity) ** 2
    return microstructure.connectivity * base
