#!/usr/bin/env python3
"""
Diffusion map analysis of N=16 adinkra gadget matrices.

The gadget matrix G[i,j] is a positive semi-definite kernel measuring similarity
between SUSY representations.  Diffusion maps use the eigenvectors of a
normalized version of this kernel to embed data points in a low-dimensional
space where Euclidean distance approximates diffusion distance.

Pipeline:
  1. For each k-stratum with >= 10 reps, normalize G into a Markov transition
     matrix P = D^{-1} G and extract the top eigenvectors of P.
  2. Embed using eigenvectors 2 and 3 (skipping the trivial constant
     eigenvector 1) for 2D, or eigenvectors 2-4 for 3D.
  3. Color each point by its source code_index to reveal clustering by code.

Outputs (saved to viz/):
  - diffusion_2d_by_stratum.png   : 2D embedding per qualifying stratum
  - diffusion_3d_k<K>.png         : 3D scatter for the largest stratum
  - diffusion_distance_k<K>.png   : pairwise diffusion distance heatmap
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = Path("/tmp/n16_full_pipeline.json")
OUTPUT_DIR = Path(__file__).resolve().parent  # viz/
DPI = 300
MIN_REPS = 10  # minimum stratum size for diffusion analysis

STRATUM_COLORS = {
    1: "#FF6B6B",
    2: "#FFD93D",
    3: "#6BCB77",
    4: "#4D96FF",
    5: "#C77DFF",
    6: "#FF922B",
    7: "#20C997",
    8: "#F06595",
}


# ---------------------------------------------------------------------------
# Data loading and code-to-rep mapping
# ---------------------------------------------------------------------------
def load_data(path: Path) -> dict:
    print(f"Loading data from {path} ...")
    with open(path) as f:
        data = json.load(f)
    print(f"  n={data['n']}, {len(data['gadget_strata'])} strata, "
          f"{len(data['results'])} codes")
    return data


def build_code_labels(results: list[dict]) -> dict[int, np.ndarray]:
    """
    Build a mapping from stratum k -> array of code_index labels for each rep.

    Within each stratum, representations appear in code_index order.  Each code
    contributes num_dashings consecutive reps.
    """
    strata_codes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in results:
        strata_codes[r["k"]].append((r["code_index"], r["num_dashings"]))

    labels = {}
    for k, code_list in strata_codes.items():
        # Sort by code_index to match the matrix ordering
        code_list.sort(key=lambda x: x[0])
        arr = []
        for code_idx, nd in code_list:
            arr.extend([code_idx] * nd)
        labels[k] = np.array(arr)
    return labels


# ---------------------------------------------------------------------------
# Diffusion map computation
# ---------------------------------------------------------------------------
def diffusion_map(G: np.ndarray, n_components: int = 4):
    """
    Compute diffusion map embedding from a PSD kernel matrix G.

    Returns:
      eigenvalues  : shape (n_components,), descending (excluding trivial)
      eigenvectors : shape (n, n_components), columns are embedding coords
    """
    n = G.shape[0]

    # Row sums for degree matrix
    d = G.sum(axis=1)

    # Guard against zero rows (shouldn't happen for PSD gadget, but be safe)
    d[d == 0] = 1.0

    # Normalize: P = D^{-1} G (right-stochastic Markov matrix)
    # For a symmetric kernel, the eigendecomposition of the symmetric
    # normalized matrix D^{-1/2} G D^{-1/2} gives the same eigenvalues
    # and the eigenvectors of P are recovered via D^{-1/2} v.
    d_inv_sqrt = 1.0 / np.sqrt(d)
    # Symmetric normalized matrix: M = D^{-1/2} G D^{-1/2}
    M = G * np.outer(d_inv_sqrt, d_inv_sqrt)

    # Compute top eigenpairs of the symmetric matrix
    # Request n_components + 1 to skip the trivial leading eigenvector
    n_eig = min(n_components + 1, n)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # eigh returns ascending order; reverse to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Convert back to eigenvectors of P: phi = D^{-1/2} v
    phi = eigenvectors * d_inv_sqrt[:, np.newaxis]

    # Skip the trivial (constant) leading eigenvector
    # Return components 1..n_components (0-indexed: columns 1 onward)
    return eigenvalues[1 : n_eig], phi[:, 1 : n_eig]


# ---------------------------------------------------------------------------
# Plot 1: 2D diffusion embedding per stratum
# ---------------------------------------------------------------------------
def plot_2d_by_stratum(strata: list[dict], code_labels: dict[int, np.ndarray]):
    """One subplot per qualifying stratum showing diffusion coordinates 2 vs 3."""
    qualifying = [s for s in strata if s["num_reps"] >= MIN_REPS]
    n_plots = len(qualifying)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle(
        "N=16 Adinkra: Diffusion Map Embeddings by Stratum",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    for idx, stratum in enumerate(qualifying):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        k = stratum["k"]
        G = np.array(stratum["matrix"], dtype=np.float64)
        n_reps = G.shape[0]
        labels = code_labels[k]

        print(f"  Computing diffusion map for k={k} ({n_reps} reps) ...")
        evals, evecs = diffusion_map(G, n_components=3)

        # Use diffusion coordinates: scale by eigenvalue for diffusion distance
        x = evecs[:, 0] * evals[0]
        y = evecs[:, 1] * evals[1]

        # Color by code_index
        unique_codes = np.unique(labels)
        cmap = matplotlib.colormaps.get_cmap("turbo").resampled(len(unique_codes))
        # Map code_index to a color index
        code_to_cidx = {c: i for i, c in enumerate(unique_codes)}
        colors = [cmap(code_to_cidx[c]) for c in labels]

        sc = ax.scatter(
            x, y,
            c=[code_to_cidx[c] for c in labels],
            cmap="turbo",
            vmin=0,
            vmax=len(unique_codes) - 1,
            s=max(1, 30 - n_reps // 100),
            alpha=0.7,
            edgecolors="none",
        )
        ax.set_xlabel(f"$\\psi_2$ ($\\lambda_2$={evals[0]:.4f})", fontsize=9)
        ax.set_ylabel(f"$\\psi_3$ ($\\lambda_3$={evals[1]:.4f})", fontsize=9)
        ax.set_title(
            f"k={k} ({n_reps} reps, {len(unique_codes)} codes)",
            fontsize=11,
            fontweight="bold",
            color=STRATUM_COLORS[k],
        )
        ax.tick_params(labelsize=7)

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("code index", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "diffusion_2d_by_stratum.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: 3D diffusion embedding for the largest stratum
# ---------------------------------------------------------------------------
def plot_3d_largest(strata: list[dict], code_labels: dict[int, np.ndarray]):
    """3D scatter plot of diffusion coordinates for the largest stratum."""
    # Find the stratum with the most reps
    largest = max(strata, key=lambda s: s["num_reps"])
    k = largest["k"]
    G = np.array(largest["matrix"], dtype=np.float64)
    n_reps = G.shape[0]
    labels = code_labels[k]

    print(f"  Computing 3D diffusion map for k={k} ({n_reps} reps) ...")
    evals, evecs = diffusion_map(G, n_components=4)

    # Diffusion coordinates scaled by eigenvalue
    x = evecs[:, 0] * evals[0]
    y = evecs[:, 1] * evals[1]
    z = evecs[:, 2] * evals[2]

    unique_codes = np.unique(labels)
    code_to_cidx = {c: i for i, c in enumerate(unique_codes)}
    c_vals = np.array([code_to_cidx[c] for c in labels])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=c_vals,
        cmap="turbo",
        vmin=0,
        vmax=len(unique_codes) - 1,
        s=6,
        alpha=0.65,
        edgecolors="none",
        depthshade=True,
    )

    ax.set_xlabel(f"$\\psi_2$ ($\\lambda$={evals[0]:.4f})", fontsize=10, labelpad=8)
    ax.set_ylabel(f"$\\psi_3$ ($\\lambda$={evals[1]:.4f})", fontsize=10, labelpad=8)
    ax.set_zlabel(f"$\\psi_4$ ($\\lambda$={evals[2]:.4f})", fontsize=10, labelpad=8)
    ax.set_title(
        f"N=16 Adinkra: 3D Diffusion Embedding, k={k} ({n_reps} reps, {len(unique_codes)} codes)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(labelsize=7)

    # Set a nice viewing angle
    ax.view_init(elev=25, azim=135)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.1, shrink=0.7)
    cbar.set_label("code index", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / f"diffusion_3d_k{k}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: Diffusion distance heatmap for the largest stratum
# ---------------------------------------------------------------------------
def plot_diffusion_distance_heatmap(
    strata: list[dict], code_labels: dict[int, np.ndarray]
):
    """
    Compute pairwise diffusion distances from the top eigenvectors and display
    as a heatmap sorted by code_index.
    """
    largest = max(strata, key=lambda s: s["num_reps"])
    k = largest["k"]
    G = np.array(largest["matrix"], dtype=np.float64)
    n_reps = G.shape[0]
    labels = code_labels[k]

    # Use more components for the distance computation to capture fine structure
    n_diff_components = min(20, n_reps - 1)
    print(f"  Computing diffusion distances for k={k} ({n_reps} reps, "
          f"{n_diff_components} components) ...")
    evals, evecs = diffusion_map(G, n_components=n_diff_components)

    # Diffusion coordinates: each row is a point, columns scaled by eigenvalue
    coords = evecs * evals[np.newaxis, :]

    # Sort by code_index so blocks are visible
    sort_order = np.argsort(labels)
    coords_sorted = coords[sort_order]
    labels_sorted = labels[sort_order]

    # Pairwise Euclidean distance in diffusion space
    # For large matrices, use the efficient formula:
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms_sq = np.sum(coords_sorted ** 2, axis=1)
    dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * coords_sorted @ coords_sorted.T
    dist_sq = np.maximum(dist_sq, 0.0)  # numerical guard
    dist = np.sqrt(dist_sq)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(
        dist,
        cmap="inferno",
        interpolation="antialiased",
        aspect="equal",
        origin="upper",
    )
    ax.set_title(
        f"N=16 Adinkra: Pairwise Diffusion Distance, k={k} ({n_reps} reps)\n"
        f"sorted by code_index, {n_diff_components} diffusion components",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Rep index (sorted by code)", fontsize=11)
    ax.set_ylabel("Rep index (sorted by code)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Diffusion distance", fontsize=10)

    # Add code boundary markers
    unique_codes = np.unique(labels_sorted)
    boundaries = []
    for code in unique_codes:
        mask = labels_sorted == code
        first = np.where(mask)[0][0]
        boundaries.append(first)
    # Draw subtle lines at code boundaries
    for b in boundaries[1:]:  # skip the first at 0
        ax.axhline(b - 0.5, color="white", linewidth=0.3, alpha=0.4)
        ax.axvline(b - 0.5, color="white", linewidth=0.3, alpha=0.4)

    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / f"diffusion_distance_k{k}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlepad": 10,
        "figure.dpi": 100,
    })

    data = load_data(INPUT_PATH)
    strata = sorted(data["gadget_strata"], key=lambda s: s["k"])
    code_labels = build_code_labels(data["results"])

    # Verify label counts match matrix dimensions
    for s in strata:
        k = s["k"]
        expected = s["num_reps"]
        got = len(code_labels.get(k, []))
        if got != expected:
            print(f"  WARNING: k={k} has {expected} reps in matrix but "
                  f"{got} from code labels")

    print("\n--- 2D Diffusion Embeddings ---")
    plot_2d_by_stratum(strata, code_labels)

    print("\n--- 3D Diffusion Embedding (largest stratum) ---")
    plot_3d_largest(strata, code_labels)

    print("\n--- Diffusion Distance Heatmap (largest stratum) ---")
    plot_diffusion_distance_heatmap(strata, code_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()
