import numpy as np
import pandas as pd
from pathlib import Path
import os

# ============================================================
# 1) INPUT DATA: new carbonate collection
# ============================================================

DATA = [
    {"sample_id": "11.2", "ar_before": 0.084719193, "phi_frac_before": 20.23542135, "tc_dried_before": 1.37, "tc_brine_before": 2.08},
    {"sample_id": "12.0", "ar_before": 0.081302166, "phi_frac_before": 17.51736303, "tc_dried_before": 1.57, "tc_brine_before": 2.05},
    {"sample_id": "15.0", "ar_before": 0.056192285, "phi_frac_before": 17.43567379, "tc_dried_before": 1.47, "tc_brine_before": 2.25},
    {"sample_id": "16.2", "ar_before": 0.053403055, "phi_frac_before": 16.71596566, "tc_dried_before": 1.57, "tc_brine_before": 2.20},
    {"sample_id": "18.1", "ar_before": 0.060217558, "phi_frac_before": 15.91996355, "tc_dried_before": 1.74, "tc_brine_before": 2.28},
    {"sample_id": "18.2", "ar_before": 0.091183798, "phi_frac_before": 15.59544658, "tc_dried_before": 1.77, "tc_brine_before": 2.24},
    {"sample_id": "19.1", "ar_before": 0.061678097, "phi_frac_before": 15.38029014, "tc_dried_before": 1.81, "tc_brine_before": 2.31},
    {"sample_id": "19.2", "ar_before": 0.079700105, "phi_frac_before": 14.96686216, "tc_dried_before": 1.78, "tc_brine_before": 2.35},
    {"sample_id": "20.0", "ar_before": 0.064155593, "phi_frac_before": 14.63404327, "tc_dried_before": 1.86, "tc_brine_before": 2.37},
    {"sample_id": "20.1", "ar_before": 0.078651026, "phi_frac_before": 13.82772717, "tc_dried_before": 1.95, "tc_brine_before": 2.35},
    {"sample_id": "20.2", "ar_before": 0.085672945, "phi_frac_before": 13.61656461, "tc_dried_before": 1.91, "tc_brine_before": 2.38},
    {"sample_id": "21.0", "ar_before": 0.088556115, "phi_frac_before": 13.41438200, "tc_dried_before": 1.94, "tc_brine_before": 2.43},
    {"sample_id": "22.1", "ar_before": 0.095273697, "phi_frac_before": 13.41369814, "tc_dried_before": 1.76, "tc_brine_before": 2.32},
    {"sample_id": "22.2", "ar_before": 0.086578152, "phi_frac_before": 13.34436952, "tc_dried_before": 1.91, "tc_brine_before": 2.34},
    {"sample_id": "22.3", "ar_before": 0.101501924, "phi_frac_before": 13.33580981, "tc_dried_before": 1.83, "tc_brine_before": 2.45},
    {"sample_id": "23.0", "ar_before": 0.088273908, "phi_frac_before": 13.01172910, "tc_dried_before": 1.94, "tc_brine_before": 2.37},
    {"sample_id": "24.2", "ar_before": 0.089861396, "phi_frac_before": 12.91514143, "tc_dried_before": 1.93, "tc_brine_before": 2.34},
    {"sample_id": "25.0", "ar_before": 0.072601453, "phi_frac_before": 11.87056101, "tc_dried_before": 1.99, "tc_brine_before": 2.46},
]

df = pd.DataFrame(DATA)
df["phi"] = df["phi_frac_before"] / 100.0  # porosity: percent -> fraction

# ============================================================
# 2) OUTPUT FOLDER
# ============================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "la_test_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str((OUT_DIR / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((OUT_DIR / ".mplconfig").resolve()))

# ============================================================
# 3) USER-DEFINED CONSTANTS
# ============================================================

LAMBDA_AIR = 0.025   # W/(m K)
LAMBDA_BRINE = 0.60  # W/(m K)

F_MEAN_DRY = 0.63
F_MEAN_BRINE = 0.64

LAMBDA_M_MIN = 2.70
LAMBDA_M_MAX = 3.20
N_GRID = 1001

# ============================================================
# 4) MODEL FUNCTIONS
# ============================================================

def f_old_relation(alpha: np.ndarray, lambda_f: float) -> np.ndarray:
    """
    Previous empirical relation:
    f = [ (0.816 * lambda_f^0.16 - 0.145) * alpha + 0.09 * lambda_f + 0.034 ]
        / [ alpha + 0.114 * lambda_f ]
    """
    alpha = np.asarray(alpha, dtype=float)
    numerator = ((0.816 * lambda_f**0.16 - 0.145) * alpha +
                 0.09 * lambda_f + 0.034)
    denominator = alpha + 0.114 * lambda_f
    return numerator / denominator


def la_tc(phi: np.ndarray, lambda_m: float, lambda_f: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Lichtenecker-Asaad forward model:
    lambda_eff = lambda_m^(1 - f*phi) * lambda_f^(f*phi)
    """
    phi = np.asarray(phi, dtype=float)
    lambda_f = np.asarray(lambda_f, dtype=float)
    f = np.asarray(f, dtype=float)
    return (lambda_m ** (1.0 - f * phi)) * (lambda_f ** (f * phi))


def invert_f_from_la(lambda_eff: np.ndarray, phi: np.ndarray, lambda_m: float, lambda_f: float) -> np.ndarray:
    """
    Analytical inversion of f from LA model:
    f = ln(lambda_eff / lambda_m) / [ phi * (ln(lambda_f) - ln(lambda_m)) ]
    """
    lambda_eff = np.asarray(lambda_eff, dtype=float)
    phi = np.asarray(phi, dtype=float)

    denom = phi * (np.log(lambda_f) - np.log(lambda_m))
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Denominator is too close to zero while inverting f.")

    return np.log(lambda_eff / lambda_m) / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def collect_metrics(df_long: pd.DataFrame, pred_col: str) -> dict:
    obs_all = df_long["tc_obs"].values
    pred_all = df_long[pred_col].values

    dry = df_long["state"] == "dry"
    brine = df_long["state"] == "brine"

    obs_dry = df_long.loc[dry, "tc_obs"].values
    pred_dry = df_long.loc[dry, pred_col].values

    obs_brine = df_long.loc[brine, "tc_obs"].values
    pred_brine = df_long.loc[brine, pred_col].values

    return {
        "RMSE_all": rmse(obs_all, pred_all),
        "RMSE_dry": rmse(obs_dry, pred_dry),
        "RMSE_brine": rmse(obs_brine, pred_brine),
        "MAE_all": mae(obs_all, pred_all),
        "MAE_dry": mae(obs_dry, pred_dry),
        "MAE_brine": mae(obs_brine, pred_brine),
        "R2_all": r2_score(obs_all, pred_all),
        "R2_dry": r2_score(obs_dry, pred_dry),
        "R2_brine": r2_score(obs_brine, pred_brine),
    }


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def df_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    out_path: Path,
    float_fmt: str = "{:.4f}",
) -> None:
    cols = list(df.columns)

    col_align = []
    for c in cols:
        is_num = pd.api.types.is_numeric_dtype(df[c])
        col_align.append("r" if is_num else "l")

    lines = []
    lines.append("\\begin{table}[ht!]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append("\\begin{tabular}{%s}" % "".join(col_align))
    lines.append("\\toprule")
    lines.append(" & ".join([f"\\textbf{{{_latex_escape(c)}}}" for c in cols]) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating, int, np.integer)) and pd.notna(v):
                if isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                else:
                    vals.append(float_fmt.format(float(v)))
            else:
                vals.append(_latex_escape("" if pd.isna(v) else str(v)))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 5) FIT ONE COLLECTION-WIDE MATRIX TC USING OLD RELATION
# ============================================================

lambda_m_grid = np.linspace(LAMBDA_M_MIN, LAMBDA_M_MAX, N_GRID)

f_old_dry = f_old_relation(df["ar_before"].values, LAMBDA_AIR)
f_old_brine = f_old_relation(df["ar_before"].values, LAMBDA_BRINE)

obs_all = np.r_[df["tc_dried_before"].values, df["tc_brine_before"].values]

scores = []
for lambda_m in lambda_m_grid:
    pred_dry = la_tc(df["phi"].values, lambda_m, LAMBDA_AIR, f_old_dry)
    pred_brine = la_tc(df["phi"].values, lambda_m, LAMBDA_BRINE, f_old_brine)
    pred_all = np.r_[pred_dry, pred_brine]
    scores.append(rmse(obs_all, pred_all))

scores = np.array(scores)
best_idx = np.argmin(scores)
lambda_m_best = float(lambda_m_grid[best_idx])

print("=" * 72)
print("STEP 1: Fit matrix thermal conductivity using OLD transfer relation")
print(f"Best lambda_m = {lambda_m_best:.5f} W/(m K)")
print(f"RMSE(old transfer) = {scores[best_idx]:.5f} W/(m K)")
print("=" * 72)

# Save old-model predictions
df["f_old_dry"] = f_old_dry
df["f_old_brine"] = f_old_brine
df["tc_pred_old_dry"] = la_tc(df["phi"].values, lambda_m_best, LAMBDA_AIR, df["f_old_dry"].values)
df["tc_pred_old_brine"] = la_tc(df["phi"].values, lambda_m_best, LAMBDA_BRINE, df["f_old_brine"].values)

# ============================================================
# 6) INVERT SAMPLE-WISE f VALUES FROM OBSERVED TC
# ============================================================

df["f_inv_dry"] = invert_f_from_la(df["tc_dried_before"].values, df["phi"].values, lambda_m_best, LAMBDA_AIR)
df["f_inv_brine"] = invert_f_from_la(df["tc_brine_before"].values, df["phi"].values, lambda_m_best, LAMBDA_BRINE)

print("\nSTEP 2: Inverted sample-wise correction factors")
print(df[
    ["sample_id", "ar_before", "phi", "f_old_dry", "f_inv_dry", "f_old_brine", "f_inv_brine"]
].round(5).to_string(index=False))

# ============================================================
# 7) BUILD A LONG TABLE FOR POOLED ANALYSIS
# ============================================================

rows = []
for _, row in df.iterrows():
    rows.append({
        "sample_id": row["sample_id"],
        "alpha": row["ar_before"],
        "phi": row["phi"],
        "state": "dry",
        "lambda_f": LAMBDA_AIR,
        "tc_obs": row["tc_dried_before"],
        "f_old": row["f_old_dry"],
        "f_inv": row["f_inv_dry"],
    })
    rows.append({
        "sample_id": row["sample_id"],
        "alpha": row["ar_before"],
        "phi": row["phi"],
        "state": "brine",
        "lambda_f": LAMBDA_BRINE,
        "tc_obs": row["tc_brine_before"],
        "f_old": row["f_old_brine"],
        "f_inv": row["f_inv_brine"],
    })

long_df = pd.DataFrame(rows)

# ============================================================
# 8) MODEL A: SIMPLE LICHTENECKER (f = 1)
# ============================================================

long_df["f_plain"] = 1.0
long_df["tc_pred_plain"] = la_tc(
    long_df["phi"].values,
    lambda_m_best,
    long_df["lambda_f"].values,
    long_df["f_plain"].values,
)

# ============================================================
# 9) MODEL B: FIXED MEAN f
# ============================================================

long_df["f_fixed"] = np.where(long_df["state"] == "brine", F_MEAN_BRINE, F_MEAN_DRY)
long_df["tc_pred_fixed"] = la_tc(
    long_df["phi"].values,
    lambda_m_best,
    long_df["lambda_f"].values,
    long_df["f_fixed"].values,
)

# ============================================================
# 10) MODEL C: OLD TRANSFER RELATION
# ============================================================

long_df["tc_pred_old"] = la_tc(
    long_df["phi"].values,
    lambda_m_best,
    long_df["lambda_f"].values,
    long_df["f_old"].values,
)

# ============================================================
# 11) MODEL D: UNIFIED RELATION
#     f(alpha, lambda_f) = beta0 + beta1*alpha + beta2*lambda_f + beta3*alpha*lambda_f
# ============================================================

X = np.c_[
    np.ones(len(long_df)),
    long_df["alpha"].values,
    long_df["lambda_f"].values,
    long_df["alpha"].values * long_df["lambda_f"].values,
]
y = long_df["f_inv"].values

beta = np.linalg.lstsq(X, y, rcond=None)[0]
beta0, beta1, beta2, beta3 = beta

long_df["f_unified"] = (
    beta0
    + beta1 * long_df["alpha"].values
    + beta2 * long_df["lambda_f"].values
    + beta3 * long_df["alpha"].values * long_df["lambda_f"].values
)

long_df["tc_pred_unified"] = la_tc(
    long_df["phi"].values,
    lambda_m_best,
    long_df["lambda_f"].values,
    long_df["f_unified"].values,
)

print("\nSTEP 3: Unified collection-specific relation")
print(
    f"f(alpha, lambda_f) = "
    f"{beta0:.5f} + {beta1:.5f}*alpha + {beta2:.5f}*lambda_f + {beta3:.5f}*(alpha*lambda_f)"
)

# Explicit dry/brine lines derived from unified relation
dry_intercept = beta0 + beta2 * LAMBDA_AIR
dry_slope = beta1 + beta3 * LAMBDA_AIR

brine_intercept = beta0 + beta2 * LAMBDA_BRINE
brine_slope = beta1 + beta3 * LAMBDA_BRINE

print("\nSTEP 4: State-specific lines implied by the unified relation")
print(f"f_dry(alpha)   ≈ {dry_slope:.5f} * alpha + {dry_intercept:.5f}")
print(f"f_brine(alpha) ≈ {brine_slope:.5f} * alpha + {brine_intercept:.5f}")

# Optional direct independent state-wise linear fits
coef_dry = np.polyfit(df["ar_before"].values, df["f_inv_dry"].values, deg=1)
coef_brine = np.polyfit(df["ar_before"].values, df["f_inv_brine"].values, deg=1)

print("\nSTEP 5: Direct state-wise linear fits to inverted f")
print(f"Direct fit dry:   f_dry(alpha)   ≈ {coef_dry[0]:.5f} * alpha + {coef_dry[1]:.5f}")
print(f"Direct fit brine: f_brine(alpha) ≈ {coef_brine[0]:.5f} * alpha + {coef_brine[1]:.5f}")

# ============================================================
# 12) MODEL COMPARISON
# ============================================================

metrics_plain = collect_metrics(long_df, "tc_pred_plain")
metrics_fixed = collect_metrics(long_df, "tc_pred_fixed")
metrics_old = collect_metrics(long_df, "tc_pred_old")
metrics_unified = collect_metrics(long_df, "tc_pred_unified")

summary = pd.DataFrame([
    {"model": "A: simple Lichtenecker (f=1)", **metrics_plain},
    {"model": "B: fixed mean f", **metrics_fixed},
    {"model": "C: old transfer relation", **metrics_old},
    {"model": "D: unified f(alpha, lambda_f)", **metrics_unified},
])

print("\nSTEP 6: Model comparison")
print(summary.round(5).to_string(index=False))

# ============================================================
# 13) SAVE TABLES
# ============================================================

summary.to_csv(OUT_DIR / "la_model_comparison.csv", index=False)
df_to_latex_table(
    summary,
    caption="Model comparison for LA thermal conductivity for the analyzed carbonate collection.",
    label="tab:la_model_comparison",
    out_path=OUT_DIR / "la_model_comparison.tex",
    float_fmt="{:.5f}",
)

params_df = pd.DataFrame([
    {"parameter": "lambda_m_best", "value": lambda_m_best, "unit": "W/(m·K)"},
    {"parameter": "beta0", "value": beta0, "unit": "-"},
    {"parameter": "beta1", "value": beta1, "unit": "-"},
    {"parameter": "beta2", "value": beta2, "unit": "-"},
    {"parameter": "beta3", "value": beta3, "unit": "-"},
    {"parameter": "dry_intercept", "value": dry_intercept, "unit": "-"},
    {"parameter": "dry_slope", "value": dry_slope, "unit": "-"},
    {"parameter": "brine_intercept", "value": brine_intercept, "unit": "-"},
    {"parameter": "brine_slope", "value": brine_slope, "unit": "-"},
])

params_df.to_csv(OUT_DIR / "la_fitted_parameters.csv", index=False)
df_to_latex_table(
    params_df,
    caption="Fitted parameters for the unified relation $f(\\alpha, \\lambda_F)$.",
    label="tab:la_fitted_parameters",
    out_path=OUT_DIR / "la_fitted_parameters.tex",
    float_fmt="{:.5f}",
)

# ============================================================
# 14) DIAGNOSTIC TABLES
# ============================================================

df["delta_f_dry"] = df["f_inv_dry"] - df["f_old_dry"]
df["delta_f_brine"] = df["f_inv_brine"] - df["f_old_brine"]

print("\nSTEP 7: Departure from the old transfer relation")
print(df[
    ["sample_id", "ar_before", "f_old_dry", "f_inv_dry", "delta_f_dry", "f_old_brine", "f_inv_brine", "delta_f_brine"]
].round(5).to_string(index=False))

df.to_csv(OUT_DIR / "la_per_sample_wide.csv", index=False)
long_df.to_csv(OUT_DIR / "la_per_sample_long.csv", index=False)

delta_stats = pd.DataFrame([
    {
        "state": "dry",
        "mean": float(df["delta_f_dry"].mean()),
        "std": float(df["delta_f_dry"].std(ddof=1)),
        "min": float(df["delta_f_dry"].min()),
        "max": float(df["delta_f_dry"].max()),
    },
    {
        "state": "brine",
        "mean": float(df["delta_f_brine"].mean()),
        "std": float(df["delta_f_brine"].std(ddof=1)),
        "min": float(df["delta_f_brine"].min()),
        "max": float(df["delta_f_brine"].max()),
    },
])

delta_stats.to_csv(OUT_DIR / "la_delta_f_stats.csv", index=False)
df_to_latex_table(
    delta_stats,
    caption="Statistics of deviation from the old transfer relation, $\\Delta f = f_{inv} - f_{old}$.",
    label="tab:la_delta_f_stats",
    out_path=OUT_DIR / "la_delta_f_stats.tex",
    float_fmt="{:.5f}",
)

# Additional residuals
long_df["resid_plain"] = long_df["tc_pred_plain"] - long_df["tc_obs"]
long_df["resid_fixed"] = long_df["tc_pred_fixed"] - long_df["tc_obs"]
long_df["resid_old"] = long_df["tc_pred_old"] - long_df["tc_obs"]
long_df["resid_unified"] = long_df["tc_pred_unified"] - long_df["tc_obs"]

# ============================================================
# 15) OPTIONAL PLOTTING
# ============================================================

try:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    model_color = {
        "A: simple Lichtenecker (f=1)": "#1f77b4",
        "B: fixed mean f": "#2ca02c",
        "C: old transfer relation": "#ff7f0e",
        "D: unified f(alpha, lambda_f)": "#9467bd",
    }
    state_marker = {"dry": "o", "brine": "s"}

    # ------------------------------------------------------------
    # Plot 1: alpha vs f (old vs inverted)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(df["ar_before"], df["f_inv_dry"], label="Dry: inverted $f$", c="#d62728", s=35, alpha=0.85)
    ax.scatter(df["ar_before"], df["f_old_dry"], label="Dry: old $f$", c="#d62728", s=35, alpha=0.85, marker="x")
    ax.scatter(df["ar_before"], df["f_inv_brine"], label="Brine: inverted $f$", c="#1f77b4", s=35, alpha=0.85)
    ax.scatter(df["ar_before"], df["f_old_brine"], label="Brine: old $f$", c="#1f77b4", s=35, alpha=0.85, marker="x")

    alpha_grid = np.linspace(df["ar_before"].min(), df["ar_before"].max(), 200)
    f_dry_line = dry_intercept + dry_slope * alpha_grid
    f_brine_line = brine_intercept + brine_slope * alpha_grid
    ax.plot(alpha_grid, f_dry_line, c="#d62728", lw=1.8, alpha=0.9, label="Unified dry line")
    ax.plot(alpha_grid, f_brine_line, c="#1f77b4", lw=1.8, alpha=0.9, label="Unified brine line")

    ax.set_xlabel("Aspect ratio, $\\alpha$")
    ax.set_ylabel("Correction factor, $f$")
    ax.set_title("Old relation vs inverted $f$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.43), ncol=2, frameon=True)

    # ------------------------------------------------------------
    # Plot 2: measured vs predicted
    # ------------------------------------------------------------
    ax = axes[1]
    for model_name, col in [
        ("A: simple Lichtenecker (f=1)", "tc_pred_plain"),
        ("B: fixed mean f", "tc_pred_fixed"),
        ("C: old transfer relation", "tc_pred_old"),
        ("D: unified f(alpha, lambda_f)", "tc_pred_unified"),
    ]:
        for state in ["dry", "brine"]:
            g = long_df[long_df["state"] == state]
            ax.scatter(
                g["tc_obs"],
                g[col],
                label=f"{model_name} ({state})",
                s=32,
                alpha=0.80,
                c=model_color[model_name],
                marker=state_marker[state],
                edgecolors="none",
            )

    mn = min(
        long_df["tc_obs"].min(),
        long_df["tc_pred_plain"].min(),
        long_df["tc_pred_fixed"].min(),
        long_df["tc_pred_old"].min(),
        long_df["tc_pred_unified"].min(),
    )
    mx = max(
        long_df["tc_obs"].max(),
        long_df["tc_pred_plain"].max(),
        long_df["tc_pred_fixed"].max(),
        long_df["tc_pred_old"].max(),
        long_df["tc_pred_unified"].max(),
    )

    ax.plot([mn, mx], [mn, mx], linestyle="--", color="k", alpha=0.6)
    ax.set_xlabel("Measured $\\lambda$ (W/(m·K))")
    ax.set_ylabel("Predicted $\\lambda$ (W/(m·K))")
    ax.set_title("Measured vs predicted (all samples)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2, frameon=True)

    fig.savefig(OUT_DIR / "la_alpha_vs_f_and_parity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Plot 3: residual distributions
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)

    res = pd.DataFrame({
        "plain": long_df["resid_plain"],
        "fixed": long_df["resid_fixed"],
        "old": long_df["resid_old"],
        "unified": long_df["resid_unified"],
    })

    bins = np.linspace(float(res.min().min()), float(res.max().max()), 18)

    ax.hist(res["plain"], bins=bins, alpha=0.55, label="A: simple Lichtenecker")
    ax.hist(res["fixed"], bins=bins, alpha=0.55, label="B: fixed mean $f$")
    ax.hist(res["old"], bins=bins, alpha=0.55, label="C: old transfer")
    ax.hist(res["unified"], bins=bins, alpha=0.55, label="D: unified $f(\\alpha,\\lambda_f)$")

    ax.axvline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Residual $\\lambda_{pred} - \\lambda_{meas}$ (W/(m·K))")
    ax.set_ylabel("Count")
    ax.set_title("Residual distributions")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    fig.savefig(OUT_DIR / "la_residual_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Plot 4: inverted f vs unified predicted f
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)

    for state in ["dry", "brine"]:
        g = long_df[long_df["state"] == state]
        ax.scatter(
            g["f_inv"],
            g["f_unified"],
            label=state.capitalize(),
            s=38,
            alpha=0.85,
            marker=state_marker[state],
        )

    fmin = min(long_df["f_inv"].min(), long_df["f_unified"].min())
    fmax = max(long_df["f_inv"].max(), long_df["f_unified"].max())
    ax.plot([fmin, fmax], [fmin, fmax], "--", color="k", alpha=0.6)

    ax.set_xlabel("Inverted $f$")
    ax.set_ylabel("Unified-model $f$")
    ax.set_title("Inverted vs unified-predicted $f$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    fig.savefig(OUT_DIR / "la_finv_vs_funified.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

except ImportError:
    print("\nmatplotlib is not installed; skipping plots.")

# ============================================================
# 16) SAVE TEXT SUMMARY
# ============================================================

summary_txt = []
summary_txt.append("LA unified relation summary")
summary_txt.append("=" * 50)
summary_txt.append(f"Best matrix conductivity lambda_m = {lambda_m_best:.6f} W/(m K)")
summary_txt.append("")
summary_txt.append("Unified relation:")
summary_txt.append(
    f"f(alpha, lambda_f) = {beta0:.6f} + {beta1:.6f}*alpha + {beta2:.6f}*lambda_f + {beta3:.6f}*(alpha*lambda_f)"
)
summary_txt.append("")
summary_txt.append(f"Dry line   : f_dry(alpha)   = {dry_slope:.6f}*alpha + {dry_intercept:.6f}")
summary_txt.append(f"Brine line : f_brine(alpha) = {brine_slope:.6f}*alpha + {brine_intercept:.6f}")
summary_txt.append("")
summary_txt.append("Model comparison:")
summary_txt.append(summary.round(6).to_string(index=False))
summary_txt.append("")

(OUT_DIR / "la_summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

print(f"\nDone. Outputs saved to:\n{OUT_DIR}")