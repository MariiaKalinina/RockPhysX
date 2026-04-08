from __future__ import annotations

"""
Example 08
----------
Thesis-ready dimensionless sensitivity atlas for the selected GSA
thermal-conductivity model.

This version improves the earlier draft by:
- keeping the upper row as 1D sensitivity curves;
- plotting the lower row in the native parameter space
  (phi, lambda_f / lambda_M), which is much more stable and readable
  than transformed triangulated coordinates;
- using filled contours with a small number of labeled isolines;
- overlaying representative dry / kerosene / brine points;
- restricting axes to realistic, publication-friendly domains.

Sensitivity definition
----------------------
For any parameter p, the local normalized sensitivity is

    K_p = d ln(lambda*) / d ln(p)

computed by central finite difference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rockphysx.core.parameters import MicrostructureParameters
from rockphysx.models.transport.thermal import thermal_conductivity


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL = "gsa"
EPSILON = 0.01

# Reference state used for the 1D panels
PHI_REF = 0.18
ALPHA_REF = 1e-2
LAMBDA_M = 1.0  # dimensionless normalization

# 1D ranges
RATIO_LINE = np.linspace(0.01, 0.95, 300)

# 2D ranges for native-parameter-space maps
PHI_GRID = np.linspace(0.02, 0.35, 160)
RATIO_GRID = np.linspace(0.005, 0.25, 160)

# Representative fluid-to-matrix conductivity ratios
# Replace these by your final Timan-Pechora values if needed.
STATE_RATIOS = {
    "dry": 0.025 / 3.0,
    "kerosene": 0.13 / 3.0,
    "brine": 0.60 / 3.0,
}


# ---------------------------------------------------------------------
# Forward model helper
# ---------------------------------------------------------------------
def tc_model(phi: float, lambda_m: float, lambda_f: float, alpha: float) -> float:
    """Evaluate the selected EMT thermal-conductivity model."""
    micro = MicrostructureParameters(
        aspect_ratio=alpha,
        connectivity=0.65,
        orientation="isotropic",
        topology="intergranular",
    )
    return float(
        thermal_conductivity(
            matrix_value=lambda_m,
            fluid_value=lambda_f,
            porosity=phi,
            microstructure=micro,
            model=MODEL,
        )
    )


# ---------------------------------------------------------------------
# Local normalized sensitivity
# ---------------------------------------------------------------------
def local_normalized_sensitivity(
    parameter: str,
    *,
    phi: float,
    lambda_m: float,
    lambda_f: float,
    alpha: float,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute K_p = d ln(lambda*) / d ln(p) by central finite difference.
    """
    if parameter == "phi":
        p_minus = max(1e-6, phi * (1.0 - epsilon))
        p_plus = min(1.0 - 1e-6, phi * (1.0 + epsilon))

        k_minus = tc_model(p_minus, lambda_m, lambda_f, alpha)
        k_plus = tc_model(p_plus, lambda_m, lambda_f, alpha)

    elif parameter == "lambda_m":
        p_minus = max(1e-8, lambda_m * (1.0 - epsilon))
        p_plus = lambda_m * (1.0 + epsilon)

        k_minus = tc_model(phi, p_minus, lambda_f, alpha)
        k_plus = tc_model(phi, p_plus, lambda_f, alpha)

    elif parameter == "lambda_f":
        p_minus = max(1e-8, lambda_f * (1.0 - epsilon))
        p_plus = lambda_f * (1.0 + epsilon)

        k_minus = tc_model(phi, lambda_m, p_minus, alpha)
        k_plus = tc_model(phi, lambda_m, p_plus, alpha)

    elif parameter == "alpha":
        p_minus = max(1e-8, alpha * (1.0 - epsilon))
        p_plus = alpha * (1.0 + epsilon)

        k_minus = tc_model(phi, lambda_m, lambda_f, p_minus)
        k_plus = tc_model(phi, lambda_m, lambda_f, p_plus)

    else:
        raise ValueError(f"Unknown parameter {parameter!r}")

    return float((np.log(k_plus) - np.log(k_minus)) / (np.log(p_plus) - np.log(p_minus)))


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def label_state_points(ax: plt.Axes, phi_ref: float) -> None:
    """Overlay representative dry / kerosene / brine states."""
    colors = {
        "dry": "tab:blue",
        "kerosene": "tab:green",
        "brine": "tab:orange",
    }

    for name, ratio in STATE_RATIOS.items():
        ax.plot(phi_ref, ratio, "o", color=colors[name], markersize=5, zorder=5)
        ax.text(
            phi_ref + 0.006,
            ratio,
            name,
            fontsize=10,
            va="center",
            ha="left",
            color=colors[name],
        )


def safe_levels(z: np.ndarray, n_fill: int = 17, n_lines: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Generate stable contour and contour-line levels."""
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))

    if np.isclose(zmin, zmax):
        zmin -= 1e-6
        zmax += 1e-6

    fill_levels = np.linspace(zmin, zmax, n_fill)
    line_levels = np.linspace(zmin, zmax, n_lines)
    return fill_levels, line_levels


# ---------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------
def main() -> None:
    # -------------------------
    # (a) K_lambda_M vs lambda*/lambda_M
    # -------------------------
    x_a = []
    y_a = []

    for ratio in RATIO_LINE:
        lambda_f = ratio * LAMBDA_M
        lambda_eff = tc_model(PHI_REF, LAMBDA_M, lambda_f, ALPHA_REF)

        x_a.append(lambda_eff / LAMBDA_M)
        y_a.append(
            local_normalized_sensitivity(
                "lambda_m",
                phi=PHI_REF,
                lambda_m=LAMBDA_M,
                lambda_f=lambda_f,
                alpha=ALPHA_REF,
            )
        )

    x_a = np.asarray(x_a)
    y_a = np.asarray(y_a)
    order_a = np.argsort(x_a)
    x_a = x_a[order_a]
    y_a = y_a[order_a]

    # -------------------------
    # (b) K_lambda_f vs lambda_f/lambda_M
    # -------------------------
    x_b = RATIO_LINE.copy()
    y_b = np.array(
        [
            local_normalized_sensitivity(
                "lambda_f",
                phi=PHI_REF,
                lambda_m=LAMBDA_M,
                lambda_f=ratio * LAMBDA_M,
                alpha=ALPHA_REF,
            )
            for ratio in RATIO_LINE
        ]
    )

    # -------------------------
    # (c) K_alpha in native space (phi, lambda_f/lambda_M)
    # -------------------------
    phi_mesh, ratio_mesh = np.meshgrid(PHI_GRID, RATIO_GRID, indexing="xy")
    k_alpha = np.empty_like(phi_mesh)
    k_phi = np.empty_like(phi_mesh)

    for i in range(ratio_mesh.shape[0]):
        for j in range(ratio_mesh.shape[1]):
            phi = float(phi_mesh[i, j])
            ratio = float(ratio_mesh[i, j])
            lambda_f = ratio * LAMBDA_M

            k_alpha[i, j] = local_normalized_sensitivity(
                "alpha",
                phi=phi,
                lambda_m=LAMBDA_M,
                lambda_f=lambda_f,
                alpha=ALPHA_REF,
            )

            k_phi[i, j] = local_normalized_sensitivity(
                "phi",
                phi=phi,
                lambda_m=LAMBDA_M,
                lambda_f=lambda_f,
                alpha=ALPHA_REF,
            )

    fill_levels_c, line_levels_c = safe_levels(k_alpha)
    fill_levels_d, line_levels_d = safe_levels(k_phi)

    # -------------------------
    # Plot layout
    # -------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Panel (a)
    ax = axs[0, 0]
    ax.plot(x_a, y_a, linewidth=2.0)
    ax.set_xlabel(r"$\lambda^*/\lambda_M$")
    ax.set_ylabel(r"$K_{\lambda_M}$")
    ax.set_title("(a)")
    ax.grid(True, alpha=0.35)

    # Panel (b)
    ax = axs[0, 1]
    ax.plot(x_b, y_b, linewidth=2.0)
    ax.set_xlabel(r"$\lambda_f/\lambda_M$")
    ax.set_ylabel(r"$K_{\lambda_f}$")
    ax.set_xscale("log")
    ax.set_title("(b)")
    ax.grid(True, alpha=0.35)

    # Panel (c): K_alpha
    ax = axs[1, 0]
    cf_c = ax.contourf(
        phi_mesh,
        ratio_mesh,
        k_alpha,
        levels=fill_levels_c,
        cmap="viridis",
        extend="both",
    )
    cs_c = ax.contour(
        phi_mesh,
        ratio_mesh,
        k_alpha,
        levels=line_levels_c,
        colors="k",
        linewidths=0.9,
    )
    ax.clabel(cs_c, inline=True, fontsize=9, fmt="%.2f")
    label_state_points(ax, PHI_REF)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\lambda_f/\lambda_M$")
    ax.set_yscale("log")
    ax.set_title("(c)")
    ax.grid(True, alpha=0.20)
    cbar_c = fig.colorbar(cf_c, ax=ax, fraction=0.046, pad=0.04)
    cbar_c.set_label(r"$K_{\alpha}$")

    # Panel (d): K_phi
    ax = axs[1, 1]
    cf_d = ax.contourf(
        phi_mesh,
        ratio_mesh,
        k_phi,
        levels=fill_levels_d,
        cmap="magma",
        extend="both",
    )
    cs_d = ax.contour(
        phi_mesh,
        ratio_mesh,
        k_phi,
        levels=line_levels_d,
        colors="k",
        linewidths=0.9,
    )
    ax.clabel(cs_d, inline=True, fontsize=9, fmt="%.2f")
    label_state_points(ax, PHI_REF)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\lambda_f/\lambda_M$")
    ax.set_title("(d)")
    ax.grid(True, alpha=0.20)
    cbar_d = fig.colorbar(cf_d, ax=ax, fraction=0.046, pad=0.04)
    cbar_d.set_label(r"$K_{\phi}$")

    fig.suptitle(
        "Dimensionless sensitivity atlas for the selected GSA thermal-conductivity model",
        y=0.98,
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    outdir = Path("figures/thermal")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "08_timan_pechora_sensitivity_atlas.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "08_timan_pechora_sensitivity_atlas.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
