# strict_mt_elastic_pores.py

This is a forked strict-ish elastic Mori–Tanaka module for:

- isotropic matrix
- isotropic fluid-filled spheroidal pores/inclusions
- random isotropic orientation
- explicit support for:
  - oblate spheroids (`AR < 1`)
  - sphere (`AR = 1`)
  - prolate spheroids (`AR > 1`)

## Design choices
The module follows the **homopy / Benveniste tensor architecture**:
- stiffness tensors in Mandel notation
- Eshelby tensor
- dilute strain concentration tensor
- orientation averaging
- Mori–Tanaka concentration tensor
- effective stiffness -> K, G -> Vp, Vs

## Important note
This is meant to replace the earlier scalar proxy used for joint `TC + Vp + Vs`.
It is still a focused implementation for *your* rock-physics setting, not a general-purpose composite toolbox.

## Files
- `strict_mt_elastic_pores.py` — main module
- `strict_mt_elastic_pores_demo.py` — tiny usage example
- `strict_mt_elastic_pores_smoke_test.csv` — smoke-test outputs