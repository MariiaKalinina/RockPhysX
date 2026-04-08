# mt_local_sensitivity_from_fit_demo.py

Local sensitivity plots around *real fitted inversion results*.

Unlike the global multisweep plots, this script uses:
- the fitted M1/M2 results from `strict_mt_tc_vp_vs_inversion_fast_results.xlsx`
- the real sample porosity / fluid state / observed TC, Vp, Vs
- local sweeps around the fitted solution

## Output
For one chosen sample (default: `11.2`) it creates:

1. `local_ar_phi_sample_<sample>_M1.png`
2. `local_ar_phi_sample_<sample>_M2.png`

These show:
- left column: sensitivity to `log10(α)` around the fitted before/after solution
- right column: sensitivity to `φ` around the measured before/after porosity
- markers:
  - circle = model prediction at fitted point
  - x = observed value

3. `local_nuisance_sample_<sample>_M1.png`
4. `local_nuisance_sample_<sample>_M2.png`

These show sensitivity (wet/before state) to:
- Km
- Gm
- λm
- Kf

using the fitted AR for that sample/model.