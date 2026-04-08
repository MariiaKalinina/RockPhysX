# Mori–Tanaka TC models: M1 and M2

This script fits thermal conductivity only.

## M1
One effective aspect ratio before TG and one after TG.

Reported:
- z_before, z_after
- delta_z
- ar_before, ar_after
- ar_ratio_after_before

## M2
Beta-distributed aspect ratio on the transformed coordinate:
- u in (0,1)
- AR = 10^(-4 + 4u)

Reported:
- m_before, m_after, delta_m
- kappa_before, kappa_after, delta_kappa

## Run
```bash
python mt_tc_m1_m2.py
```

## Input
- /mnt/data/data_restructured_for_MT_v2.xlsx

## Output
- /mnt/data/mt_tc_m1_m2_results.xlsx
- /mnt/data/mt_tc_m1_m2_plots/