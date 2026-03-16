from __future__ import annotations

from enum import StrEnum


class SaturationState(StrEnum):
    """Supported saturation states for first-pass workflows."""

    DRY = "dry"
    BRINE = "brine"
    OIL = "oil"
