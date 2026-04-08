from __future__ import annotations

"""Bayesian inversion of effective aspect ratio change for one sample.

This script uses an existing GSA transport backend (the one you posted) as a
black-box forward model and builds a PyMC model for a *single sample* with:

- Beta prior on the baseline aspect ratio (mapped to [alpha_min, alpha_max])
- Gaussian change parameter on the logit scale after thermal treatment
- Log-normal observation model for dry / wet thermal conductivity
- Final visualization with three panels:
    1) observed vs posterior-predictive conductivity,
    2) posterior of alpha_before and alpha_after,
    3) posterior of ratio alpha_after / alpha_before and delta.

IMPORTANT
---------
1) Save your GSA backend code as a Python module, for example:
       gsa_transport_model.py
   in the same folder as this script.

2) The backend must export `two_phase_thermal_isotropic`, matching the code you
   shared in chat.

3) Because the GSA forward model is used as a black-box PyTensor Op, gradients
   are not available here. Therefore the script uses Slice sampling instead of
   NUTS.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import importlib
import sys

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class InversionConfig:
    # Data
    phi_before: float = 0.1294
    phi_after: float = 0.1394
    lambda_before_dry_obs: float = 1.93
    lambda_before_wet_obs: float = 2.34
    lambda_after_dry_obs: float = 1.82
    lambda_after_wet_obs: float = 2.34

    # Phase conductivities
    lambda_matrix: float = 3.0
    lambda_gas: float = 0.025
    lambda_water: float = 0.60

    # Prior bounds for effective aspect ratio
    alpha_min: float = 0.01
    alpha_max: float = 1.0

    # Prior hyperparameters
    beta_a: float = 2.0
    beta_b: float = 2.0
    delta_mu: float = 0.0
    delta_sigma: float = 1.0
    sigma_lambda_scale: float = 0.08

    # GSA closure
    comparison: str = "matrix"  # "matrix" | "self_consistent" | "bayuk_linear_mix"
    k_connectivity: float | None = None

    # Sampler settings
    draws: int = 3000
    tune: int = 2000
    chains: int = 2
    cores: int = 1
    target_accept: float = 0.90  # not used by Slice, kept for completeness
    random_seed: int = 42

    # Output
    output_dir: str = "bayes_gsa_output"
    figure_name: str = "bayes_gsa_single_sample.png"
    summary_name: str = "posterior_summary.csv"


CFG = InversionConfig()


# -----------------------------------------------------------------------------
# Backend loading
# -----------------------------------------------------------------------------


def load_gsa_backend() -> Callable[..., float]:
    """Try to import the two-phase isotropic GSA forward model.

    By default this script tries to load the in-repo implementation from
    `rockphysx.models.emt.gsa_transport`. If you have an external backend module,
    edit `candidate_modules`.
    """
    # Make the in-repo package importable when running as:
    #   python examples/bayes_gsa_aspect_ratio_single_sample.py
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if src_root.exists():
        src_str = str(src_root)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    candidate_modules = [
        "rockphysx.models.emt.gsa_transport",
        "gsa_transport_model",
        "gsa_thermal",
        "gsa_model",
        "transport_gsa",
    ]

    last_error: Exception | None = None
    for module_name in candidate_modules:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, "two_phase_thermal_isotropic")
            return fn
        except Exception as exc:  # pragma: no cover - import convenience
            last_error = exc

    raise ImportError(
        "Could not import `two_phase_thermal_isotropic` from any candidate module. "
        "Either use the in-repo backend (`rockphysx.models.emt.gsa_transport`) "
        "or save your backend as `gsa_transport_model.py` next to this script, "
        "or edit `candidate_modules` in this script."
    ) from last_error


GSA_FORWARD = load_gsa_backend()


# -----------------------------------------------------------------------------
# Deterministic helpers
# -----------------------------------------------------------------------------


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))



def alpha_from_z(z: np.ndarray | float, alpha_min: float, alpha_max: float) -> np.ndarray | float:
    return alpha_min + z * (alpha_max - alpha_min)



def run_forward_numpy(alpha_before: float, alpha_after: float, cfg: InversionConfig) -> np.ndarray:
    """Return the four predicted thermal conductivities.

    Output order:
        [before_dry, before_wet, after_dry, after_wet]
    """
    lam_bd = GSA_FORWARD(
        matrix_value=cfg.lambda_matrix,
        inclusion_value=cfg.lambda_gas,
        porosity=cfg.phi_before,
        aspect_ratio=float(alpha_before),
        comparison=cfg.comparison,
        k_connectivity=cfg.k_connectivity,
    )
    lam_bw = GSA_FORWARD(
        matrix_value=cfg.lambda_matrix,
        inclusion_value=cfg.lambda_water,
        porosity=cfg.phi_before,
        aspect_ratio=float(alpha_before),
        comparison=cfg.comparison,
        k_connectivity=cfg.k_connectivity,
    )
    lam_ad = GSA_FORWARD(
        matrix_value=cfg.lambda_matrix,
        inclusion_value=cfg.lambda_gas,
        porosity=cfg.phi_after,
        aspect_ratio=float(alpha_after),
        comparison=cfg.comparison,
        k_connectivity=cfg.k_connectivity,
    )
    lam_aw = GSA_FORWARD(
        matrix_value=cfg.lambda_matrix,
        inclusion_value=cfg.lambda_water,
        porosity=cfg.phi_after,
        aspect_ratio=float(alpha_after),
        comparison=cfg.comparison,
        k_connectivity=cfg.k_connectivity,
    )
    return np.asarray([lam_bd, lam_bw, lam_ad, lam_aw], dtype=float)


# -----------------------------------------------------------------------------
# PyMC model
# -----------------------------------------------------------------------------


def build_model(cfg: InversionConfig) -> pm.Model:
    try:
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.compile.ops import wrap_py
        from pytensor.tensor import dscalar, dvector
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "PyMC/PyTensor failed to import. This is usually caused by a NumPy binary "
            "mismatch (e.g. pip-installed NumPy 2.x inside a conda env with extensions "
            "built against NumPy 1.x). Fix by using a consistent environment, e.g.:\n"
            "  - conda install 'numpy<2' (recommended), or\n"
            "  - reinstall all compiled deps against your NumPy.\n"
            "Then rerun this script."
        ) from exc

    observed = np.asarray(
        [
            cfg.lambda_before_dry_obs,
            cfg.lambda_before_wet_obs,
            cfg.lambda_after_dry_obs,
            cfg.lambda_after_wet_obs,
        ],
        dtype=float,
    )

    @wrap_py(itypes=[dscalar, dscalar], otypes=[dvector])
    def gsa_forward_op(alpha_before: float, alpha_after: float) -> np.ndarray:
        return run_forward_numpy(alpha_before, alpha_after, cfg)

    with pm.Model() as model:
        # Baseline aspect ratio: z in (0, 1), alpha in [alpha_min, alpha_max]
        z_before = pm.Beta("z_before", alpha=cfg.beta_a, beta=cfg.beta_b)
        logit_z_before = pm.Deterministic("logit_z_before", pt.log(z_before) - pt.log1p(-z_before))

        # Change after treatment on logit scale
        delta = pm.Normal("delta", mu=cfg.delta_mu, sigma=cfg.delta_sigma)
        logit_z_after = pm.Deterministic("logit_z_after", logit_z_before + delta)
        z_after = pm.Deterministic("z_after", pm.math.sigmoid(logit_z_after))

        alpha_before = pm.Deterministic(
            "alpha_before",
            cfg.alpha_min + z_before * (cfg.alpha_max - cfg.alpha_min),
        )
        alpha_after = pm.Deterministic(
            "alpha_after",
            cfg.alpha_min + z_after * (cfg.alpha_max - cfg.alpha_min),
        )
        ratio_alpha = pm.Deterministic("ratio_alpha", alpha_after / alpha_before)

        # Black-box GSA forward model
        lambda_pred = pm.Deterministic(
            "lambda_pred",
            gsa_forward_op(alpha_before, alpha_after),
        )

        # Observation noise on log-scale
        sigma_lambda = pm.HalfNormal("sigma_lambda", sigma=cfg.sigma_lambda_scale)

        pm.Normal(
            "lambda_obs",
            mu=pt.log(lambda_pred),
            sigma=sigma_lambda,
            observed=np.log(observed),
        )

    return model


# -----------------------------------------------------------------------------
# Posterior utilities
# -----------------------------------------------------------------------------


def posterior_predictive_from_samples(idata: az.InferenceData, cfg: InversionConfig) -> np.ndarray:
    """Compute posterior-predictive mean curve for the four lambda values.

    Returns an array with shape (n_samples, 4), without observational noise.
    """
    posterior = idata.posterior
    alpha_before = posterior["alpha_before"].values.reshape(-1)
    alpha_after = posterior["alpha_after"].values.reshape(-1)

    pred = np.zeros((alpha_before.size, 4), dtype=float)
    for i, (ab, aa) in enumerate(zip(alpha_before, alpha_after)):
        pred[i] = run_forward_numpy(float(ab), float(aa), cfg)
    return pred



def summarize_intervals(samples: np.ndarray, probs: tuple[float, float] = (0.03, 0.97)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=0)
    low = np.quantile(samples, probs[0], axis=0)
    high = np.quantile(samples, probs[1], axis=0)
    return mean, low, high


def simple_posterior_summary(
    idata,
    *,
    var_names: list[str],
    hdi_prob: float = 0.94,
) -> list[dict[str, float | str]]:
    """Minimal ArviZ-free summary for environments with broken compiled deps."""
    alpha = (1.0 - hdi_prob) / 2.0
    low_q = alpha
    high_q = 1.0 - alpha

    rows: list[dict[str, float | str]] = []
    posterior = idata.posterior
    for name in var_names:
        values = np.asarray(posterior[name].values, dtype=float).reshape(-1)
        rows.append(
            {
                "var_name": name,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                f"hdi_{low_q:.3f}": float(np.quantile(values, low_q)),
                f"hdi_{high_q:.3f}": float(np.quantile(values, high_q)),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    import csv

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def make_final_figure(idata: az.InferenceData, pred_samples: np.ndarray, cfg: InversionConfig, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    observed = np.asarray(
        [
            cfg.lambda_before_dry_obs,
            cfg.lambda_before_wet_obs,
            cfg.lambda_after_dry_obs,
            cfg.lambda_after_wet_obs,
        ],
        dtype=float,
    )
    labels = ["dry before", "wet before", "dry after", "wet after"]
    x = np.arange(len(labels))

    pred_mean, pred_low, pred_high = summarize_intervals(pred_samples)

    alpha_before = idata.posterior["alpha_before"].values.reshape(-1)
    alpha_after = idata.posterior["alpha_after"].values.reshape(-1)
    ratio_alpha = idata.posterior["ratio_alpha"].values.reshape(-1)
    delta = idata.posterior["delta"].values.reshape(-1)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.0, 1.0])

    # Panel A: observed vs posterior-predictive
    ax0 = fig.add_subplot(gs[0, :])
    ax0.errorbar(
        x,
        pred_mean,
        yerr=[pred_mean - pred_low, pred_high - pred_mean],
        fmt="o",
        capsize=4,
        label="posterior predictive (94% CI)",
    )
    ax0.scatter(x, observed, marker="x", s=90, label="observed")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("Thermal conductivity")
    ax0.set_title("Observed vs posterior predictive conductivity")
    ax0.grid(alpha=0.25)
    ax0.legend()

    # Panel B: alpha_before and alpha_after
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(alpha_before, bins=40, density=True, alpha=0.6, label=r"$\alpha_{before}$")
    ax1.hist(alpha_after, bins=40, density=True, alpha=0.6, label=r"$\alpha_{after}$")
    ax1.set_xlabel("Effective aspect ratio")
    ax1.set_ylabel("Posterior density")
    ax1.set_title("Posterior of effective aspect ratio")
    ax1.grid(alpha=0.25)
    ax1.legend()

    # Panel C: ratio and delta
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist(ratio_alpha, bins=40, density=True, alpha=0.65, label=r"$\alpha_{after}/\alpha_{before}$")
    ax2.axvline(1.0, linestyle="--", linewidth=1.2, label="no change")
    ax2.set_xlabel("Aspect-ratio ratio")
    ax2.set_ylabel("Posterior density")
    ax2.set_title("Posterior of structural change")
    ax2.grid(alpha=0.25)

    # secondary axis for delta summary text
    p_decrease = float(np.mean(ratio_alpha < 1.0))
    q_ratio = np.quantile(ratio_alpha, [0.03, 0.5, 0.97])
    q_delta = np.quantile(delta, [0.03, 0.5, 0.97])
    text = (
        f"P(alpha_after < alpha_before) = {p_decrease:.3f}\n"
        f"ratio 94% CI: [{q_ratio[0]:.3f}, {q_ratio[2]:.3f}]\n"
        f"ratio median: {q_ratio[1]:.3f}\n"
        f"delta 94% CI: [{q_delta[0]:.3f}, {q_delta[2]:.3f}]"
    )
    ax2.text(0.03, 0.97, text, transform=ax2.transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax2.legend()

    fig.suptitle("Bayesian GSA inversion of aspect-ratio change for one sample", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    try:
        import arviz as az
        import pymc as pm
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Failed to import PyMC/ArviZ. Your Python environment has binary-incompatible "
            "packages (common when mixing pip NumPy 2.x with conda-compiled extensions). "
            "Recommended fix in conda:\n"
            "  1) pip uninstall -y numpy\n"
            "  2) conda install 'numpy<2'\n"
            "  3) conda install pymc arviz h5py numba pyarrow -c conda-forge\n"
            "Then rerun this script."
        ) from exc

    out_dir = Path(CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(CFG)

    with model:
        # Black-box Op -> use gradient-free sampler
        step = pm.Slice()
        idata = pm.sample(
            draws=CFG.draws,
            tune=CFG.tune,
            chains=CFG.chains,
            cores=CFG.cores,
            step=step,
            random_seed=CFG.random_seed,
            return_inferencedata=True,
            compute_convergence_checks=False,
            progressbar=True,
        )

    pred_samples = posterior_predictive_from_samples(idata, CFG)

    # Save summary
    summary_path = out_dir / CFG.summary_name
    summary_var_names = ["alpha_before", "alpha_after", "ratio_alpha", "delta", "sigma_lambda"]
    try:
        summary = az.summary(
            idata,
            var_names=summary_var_names,
            hdi_prob=0.94,
        )
        summary.to_csv(summary_path)
    except Exception:
        rows = simple_posterior_summary(idata, var_names=summary_var_names, hdi_prob=0.94)
        write_summary_csv(rows, summary_path)
        summary = None

    # Save figure
    figure_path = out_dir / CFG.figure_name
    make_final_figure(idata, pred_samples, CFG, figure_path)

    # Save a reproducible artifact.
    # `InferenceData.to_netcdf` may fail in environments where h5py/netcdf backends
    # are binary-incompatible with NumPy; fall back to a NumPy NPZ dump.
    idata_path = out_dir / "idata.nc"
    try:
        idata.to_netcdf(idata_path)
    except Exception:
        npz_path = out_dir / "idata_posterior.npz"
        np.savez_compressed(
            npz_path,
            alpha_before=idata.posterior["alpha_before"].values,
            alpha_after=idata.posterior["alpha_after"].values,
            ratio_alpha=idata.posterior["ratio_alpha"].values,
            delta=idata.posterior["delta"].values,
            sigma_lambda=idata.posterior["sigma_lambda"].values,
        )
        idata_path = npz_path

    # Console output
    p_decrease = float(np.mean(idata.posterior["ratio_alpha"].values.reshape(-1) < 1.0))
    print("=" * 80)
    print("CONFIG")
    print(asdict(CFG))
    print("-" * 80)
    if summary is not None:
        print(summary)
    else:
        print(f"Saved summary CSV to: {summary_path.resolve()}")
    print("-" * 80)
    print(f"P(alpha_after < alpha_before | data) = {p_decrease:.4f}")
    print(f"Saved figure to:   {figure_path.resolve()}")
    print(f"Saved summary to:  {summary_path.resolve()}")
    print(f"Saved idata to:    {idata_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
