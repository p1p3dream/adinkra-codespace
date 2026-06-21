#!/usr/bin/env python3
"""
Generate publication-quality heatmaps of N=16 adinkra gadget matrices.

Reads the full pipeline output from /tmp/n16_full_pipeline.json and produces:
  - One PNG per k-stratum (gadget_heatmap_k1.png .. gadget_heatmap_k8.png)
  - One combined figure with all 8 strata (gadget_heatmaps_combined.png)

All output is saved to the same directory as this script (viz/).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = "/tmp/n16_full_pipeline.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 300

# Matrices above this dimension get interpolation for readability
INTERPOLATION_THRESHOLD = 200


def load_data(path: str) -> dict:
    print(f"Loading data from {path} ...")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  n={data['n']}, {len(data['gadget_strata'])} strata loaded.")
    return data


def make_single_heatmap(
    mat: np.ndarray,
    k: int,
    d: int,
    num_reps: int,
    output_path: str,
) -> None:
    """Render one stratum's gadget matrix as a standalone heatmap."""

    n_dim = mat.shape[0]
    diag_val = mat[0, 0]

    # Diverging colormap centered on the diagonal value.
    # vmin/vmax are symmetric around diag_val so the diagonal sits at the
    # colormap center, making off-diagonal deviations visually obvious.
    off_diag_mask = ~np.eye(n_dim, dtype=bool)
    if off_diag_mask.any():
        off_min = mat[off_diag_mask].min()
        off_max = mat[off_diag_mask].max()
    else:
        off_min, off_max = diag_val, diag_val

    # Symmetric extent around diag_val
    extent = max(abs(diag_val - off_min), abs(diag_val - off_max)) * 1.05
    vmin = diag_val - extent
    vmax = diag_val + extent

    norm = mcolors.TwoSlopeNorm(vcenter=diag_val, vmin=vmin, vmax=vmax)

    # Choose interpolation for large matrices
    interp = "antialiased" if n_dim > INTERPOLATION_THRESHOLD else "nearest"

    # Figure sizing: keep small matrices from being tiny, large ones readable
    base = max(6, min(12, n_dim / 80))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(base + 1.5, base))

        im = ax.imshow(
            mat,
            cmap="RdBu_r",
            norm=norm,
            interpolation=interp,
            aspect="equal",
            origin="upper",
        )

        ax.set_title(
            f"N=16 Gadget Matrix, k={k} (d={d:,}, {num_reps:,} reps)",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Rep index")
        ax.set_ylabel("Rep index")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Gadget value", fontsize=10)

        # For small matrices, show tick labels; for large ones, reduce clutter
        if n_dim <= 50:
            ax.set_xticks(range(n_dim))
            ax.set_yticks(range(n_dim))
            ax.tick_params(labelsize=7)
        else:
            ax.tick_params(labelsize=8)

        fig.tight_layout()
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {output_path}  ({n_dim}x{n_dim})")


def make_combined_figure(strata: list[dict], output_path: str) -> None:
    """All 8 strata as subplots in a single figure."""

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes_flat = axes.flatten()

        for idx, stratum in enumerate(strata):
            ax = axes_flat[idx]
            mat = np.array(stratum["matrix"], dtype=np.float64)
            k = stratum["k"]
            d = stratum["d"]
            num_reps = stratum["num_reps"]
            n_dim = mat.shape[0]
            diag_val = mat[0, 0]

            off_diag_mask = ~np.eye(n_dim, dtype=bool)
            if off_diag_mask.any():
                off_min = mat[off_diag_mask].min()
                off_max = mat[off_diag_mask].max()
            else:
                off_min, off_max = diag_val, diag_val

            extent = max(abs(diag_val - off_min), abs(diag_val - off_max)) * 1.05
            vmin = diag_val - extent
            vmax = diag_val + extent
            norm = mcolors.TwoSlopeNorm(vcenter=diag_val, vmin=vmin, vmax=vmax)

            interp = "antialiased" if n_dim > INTERPOLATION_THRESHOLD else "nearest"

            im = ax.imshow(
                mat,
                cmap="RdBu_r",
                norm=norm,
                interpolation=interp,
                aspect="equal",
                origin="upper",
            )

            ax.set_title(
                f"k={k} (d={d:,}, {num_reps:,} reps)",
                fontsize=10,
                fontweight="bold",
            )

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Minimal tick labels for combined view
            ax.tick_params(labelsize=6)

        fig.suptitle(
            "N=16 Adinkra Gadget Matrices by k-Stratum",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {output_path}")


def main() -> None:
    data = load_data(INPUT_PATH)
    strata = sorted(data["gadget_strata"], key=lambda s: s["k"])

    print(f"\nGenerating individual heatmaps ...")
    for stratum in strata:
        mat = np.array(stratum["matrix"], dtype=np.float64)
        k = stratum["k"]
        d = stratum["d"]
        num_reps = stratum["num_reps"]
        out = os.path.join(OUTPUT_DIR, f"gadget_heatmap_k{k}.png")
        make_single_heatmap(mat, k, d, num_reps, out)

    print(f"\nGenerating combined figure ...")
    combined_out = os.path.join(OUTPUT_DIR, "gadget_heatmaps_combined.png")
    make_combined_figure(strata, combined_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
