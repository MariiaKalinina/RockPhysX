# HS-feasible matrix-property scan (no regularization)

This folder contains a script that estimates **min–max feasible ranges** for matrix properties by scanning candidate matrix parameters and penalizing only **violations of the Hashin–Shtrikman (HS) bounds**, normalized by measurement uncertainty.

The model assumption is:

- The mineral matrix properties are **constant** (shared across `before/after` and `dry/wet`).
- Pore-phase properties differ by saturation (`dry` vs `wet`), and are treated as known constants.
- For each observation, HS bounds define a physically admissible interval. Any value outside the interval is penalized.

## Script

- `test_new/hs_matrix_feasible_set_scan.py`

## Output artifacts (in `--out-dir`)

- `scan_points.csv` — all sampled points with metrics (`J`, `frac_ok`, `max_vn`, plus before/after metrics).
- `feasible_minmax_summary.csv` — min / max and quantiles of the feasible subset.
- `feasible_minmax_summary.tex` — LaTeX (booktabs) table for Overleaf.
- `feasible_set_pairwise.png` / `.pdf` — pairwise scatter of feasible set (colored by `J`).

## Metrics

For each observation \(i\):

- HS interval: \([y_{lo,i}^{HS}(\theta),\;y_{hi,i}^{HS}(\theta)]\)
- violation: \(v_i(\theta)=\max(0,\;y_{lo,i}-y_i,\;y_i-y_{hi,i})\)
- normalized violation: \(vn_i=v_i/\sigma_i\)

Objective:

- \(J(\theta)=\sum_i vn_i^2\)

## Frequentist feasibility probability (P2)

Additionally, the script computes a frequentist-style feasibility probability under Gaussian measurement errors:

- \(y_i^{sim}=y_i+\epsilon_i\), \(\epsilon_i\sim \mathcal{N}(0,\sigma_i)\)
- \(p_i(\theta)=P(y_{lo,i}^{HS}(\theta)\le y_i^{sim}\le y_{hi,i}^{HS}(\theta))\)
- \(P_{all}(\theta)=\prod_i p_i(\theta)\) (independent errors)

By default this is computed **analytically** using a Normal CDF approximation (equivalent to infinite Monte-Carlo draws).
You can request Monte-Carlo via `--prob-mode mc --mc-draws ...`, but it is slower.

Probability-weighted outputs:

- `probability_weighted_95ci.csv` / `.tex` — 95% probability-weighted intervals for \(\lambda_M, V_{P,M}, V_{S,M}\)
- `probability_weighted_1d.png` / `.pdf` — 1D weighted histograms with 95% interval markers
- `probability_weighted_2d_contours.png` / `.pdf` — 2D weighted contours (95% mass)

Feasibility filter (default):

- `frac_ok >= frac_ok_min`, where `frac_ok = mean(vn <= tau)`
- `max_vn <= max_vn_max`

## Typical run (tight ranges)

```bash
python test_new/hs_matrix_feasible_set_scan.py \
  --n-samples 50000 \
  --lambda-min 2.80 --lambda-max 2.92 \
  --vp-min 6460 --vp-max 7140 \
  --vs-min 3610 --vs-max 3990
```

If you get “No feasible points found”, relax thresholds:

- decrease `--frac-ok-min` (e.g. `0.90`)
- increase `--max-vn-max` (e.g. `5.0`)
- widen parameter ranges

## Overleaf

The `.tex` tables use `\\toprule/\\midrule/\\bottomrule` so you need:

```tex
\\usepackage{booktabs}
```
