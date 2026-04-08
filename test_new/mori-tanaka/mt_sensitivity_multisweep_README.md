# mt_sensitivity_multisweep.py

Unified sensitivity-plot script in the same visual style as your uploaded scripts.

It keeps:
- `dataclass` config
- `_configure_matplotlib_env`
- `grid(alpha=0.25)`
- `turbo` + colorbar for parametric sweeps
- legend below the plot for fixed-phi AR sweep
- `constrained_layout=True`
- `bbox_inches="tight"` on save

## Output figures
- `mt_sensitivity_ar_fixed_phi.png`
- `mt_sensitivity_phi_parametric.png`
- `mt_sensitivity_Km_parametric.png`
- `mt_sensitivity_Gm_parametric.png`
- `mt_sensitivity_lambda_m_parametric.png`
- `mt_sensitivity_Kf_parametric.png`