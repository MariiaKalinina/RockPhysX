# strict_mt_tc_vp_vs_inversion_fast.py

Fast joint sample-by-sample inversion for:
- TC
- Vp
- Vs

using:
- M1: one effective aspect ratio before/after TG
- M2: beta-distributed aspect ratio before/after TG

## Physics
- Thermal conductivity: scalar Mori–Tanaka block
- Elastic properties: `strict_mt_elastic_pores.py`

## Speedup
All TC / Vp / Vs predictions are precomputed on a shared AR grid per observation.
The optimizer then uses:
- interpolation for M1
- weighted sums for M2

## Default objective
`rel_l1`