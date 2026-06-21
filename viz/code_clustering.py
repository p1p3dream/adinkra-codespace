#!/usr/bin/env python3
"""
Visualize how the 145 doubly-even codes cluster based on gadget relationships.

Reads the full N=16 pipeline output from /tmp/n16_full_pipeline.json and produces:
  - code_network_by_stratum.png   : Force-directed network of code-code gadget similarity
  - code_dendrogram.png           : Hierarchical clustering dendrograms for k=3,4,5
  - code_similarity_k4.png        : Heatmap of code-code similarity for k=4 stratum
  - all_codes_landscape.png       : Bubble chart of all 145 codes by (code_index, k)

All output is saved to the viz/ directory alongside this script.
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = "/tmp/n16_full_pipeline.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 300

# Colors for k-strata (k=1..8)
STRATUM_COLORS = {
    1: "#e6194b",  # red
    2: "#f58231",  # orange
    3: "#ffe119",  # yellow
    4: "#3cb44b",  # green
    5: "#42d4f4",  # cyan
    6: "#4363d8",  # blue
    7: "#911eb4",  # purple
    8: "#f032e6",  # magenta
}


def load_data(path: str) -> dict:
    print(f"Loading data from {path} ...")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  n={data['n']}, {data['num_codes']} codes, {len(data['gadget_strata'])} strata.")
    return data


def build_code_groups(results: list[dict]) -> dict[int, list[dict]]:
    """Group code results by k-stratum, preserving order."""
    by_k = defaultdict(list)
    for r in results:
        by_k[r["k"]].append(r)
    return dict(by_k)


def compute_code_similarity_matrix(
    gadget_matrix: np.ndarray,
    codes_in_stratum: list[dict],
) -> tuple[np.ndarray, list[int]]:
    """
    Compute a code-code similarity matrix from the full rep-level gadget matrix.

    For each pair of codes (i, j), extract the sub-block of the gadget matrix
    corresponding to (reps of code i) x (reps of code j) and average all values.
    For the diagonal (i == i), average the off-diagonal elements within the
    code's self-block (excluding the rep-level diagonal).

    Returns:
        sim_matrix: (num_codes, num_codes) array of average gadget values
        code_indices: list of code_index values in order
    """
    num_codes = len(codes_in_stratum)
    sim_matrix = np.zeros((num_codes, num_codes))
    code_indices = [c["code_index"] for c in codes_in_stratum]

    # Build rep offset ranges for each code
    offsets = []
    pos = 0
    for c in codes_in_stratum:
        nd = c["num_dashings"]
        offsets.append((pos, pos + nd))
        pos += nd

    for i in range(num_codes):
        ri_start, ri_end = offsets[i]
        for j in range(i, num_codes):
            rj_start, rj_end = offsets[j]

            block = gadget_matrix[ri_start:ri_end, rj_start:rj_end]

            if i == j:
                # Self-block: average off-diagonal elements
                nd = ri_end - ri_start
                if nd > 1:
                    mask = ~np.eye(nd, dtype=bool)
                    avg = block[mask].mean()
                else:
                    avg = block[0, 0]
            else:
                # Cross-block: average all elements
                avg = block.mean()

            sim_matrix[i, j] = avg
            sim_matrix[j, i] = avg

    return sim_matrix, code_indices


def plot_code_network(
    code_groups: dict[int, list[dict]],
    strata_by_k: dict[int, dict],
    output_path: str,
) -> None:
    """
    Draw a force-directed network graph for strata with multiple codes (k=2..7).
    Nodes = codes, edges = gadget similarity, colored by k-stratum.
    """
    print("  Building code network ...")

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes_flat = axes.flatten()

        plot_k_values = [2, 3, 4, 5, 6, 7]

        for ax_idx, k in enumerate(plot_k_values):
            ax = axes_flat[ax_idx]
            codes = code_groups[k]
            stratum = strata_by_k[k]
            mat = np.array(stratum["matrix"], dtype=np.float64)

            sim_matrix, code_indices = compute_code_similarity_matrix(mat, codes)
            num_codes = len(codes)

            # Normalize similarity to [0, 1] for edge weights
            sim_min = sim_matrix.min()
            sim_max = sim_matrix.max()
            if sim_max > sim_min:
                sim_norm = (sim_matrix - sim_min) / (sim_max - sim_min)
            else:
                sim_norm = np.ones_like(sim_matrix)

            # Build networkx graph
            G = nx.Graph()
            for i, c in enumerate(codes):
                G.add_node(i, code_index=c["code_index"], num_dashings=c["num_dashings"])

            # Add edges with similarity weights (skip self-loops)
            for i in range(num_codes):
                for j in range(i + 1, num_codes):
                    G.add_edge(i, j, weight=sim_norm[i, j], raw_sim=sim_matrix[i, j])

            # Spring layout weighted by similarity (closer = more similar)
            pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5 / np.sqrt(num_codes))

            # Draw edges with opacity/width proportional to similarity
            edges = G.edges(data=True)
            edge_weights = [e[2]["weight"] for e in edges]

            if edge_weights:
                max_w = max(edge_weights)
                min_w = min(edge_weights)
                if max_w > min_w:
                    edge_alphas = [0.1 + 0.7 * (w - min_w) / (max_w - min_w) for w in edge_weights]
                    edge_widths = [0.3 + 2.5 * (w - min_w) / (max_w - min_w) for w in edge_weights]
                else:
                    edge_alphas = [0.5] * len(edge_weights)
                    edge_widths = [1.0] * len(edge_weights)

                for (u, v, d), alpha, width in zip(edges, edge_alphas, edge_widths):
                    x = [pos[u][0], pos[v][0]]
                    y = [pos[u][1], pos[v][1]]
                    ax.plot(x, y, color=STRATUM_COLORS[k], alpha=alpha, linewidth=width)

            # Draw nodes sized by num_dashings (log scale for visibility)
            node_sizes = [codes[i]["num_dashings"] for i in range(num_codes)]
            # Scale node sizes for visual appeal
            size_scale = 300 / max(node_sizes) if max(node_sizes) > 0 else 1
            scaled_sizes = [s * size_scale + 30 for s in node_sizes]

            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=scaled_sizes,
                node_color=STRATUM_COLORS[k],
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
            )

            # Labels: show code_index
            labels = {i: str(codes[i]["code_index"]) for i in range(num_codes)}
            label_fontsize = max(4, min(7, 100 // num_codes))
            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=label_fontsize,
                font_color="white",
                font_weight="bold",
            )

            ax.set_title(
                f"k={k} ({num_codes} codes, {codes[0]['num_dashings']} dashings each)",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
            ax.set_axis_off()

        fig.suptitle(
            "N=16 Adinkra Code Similarity Networks by k-Stratum",
            fontsize=18,
            fontweight="bold",
            color="white",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {output_path}")


def plot_dendrograms(
    code_groups: dict[int, list[dict]],
    strata_by_k: dict[int, dict],
    output_path: str,
) -> None:
    """
    Hierarchical clustering dendrograms for k=3, 4, 5 strata.
    Distance = 1 - normalized_similarity.
    """
    print("  Building dendrograms ...")

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        for ax_idx, k in enumerate([3, 4, 5]):
            ax = axes[ax_idx]
            codes = code_groups[k]
            stratum = strata_by_k[k]
            mat = np.array(stratum["matrix"], dtype=np.float64)

            sim_matrix, code_indices = compute_code_similarity_matrix(mat, codes)
            num_codes = len(codes)

            # Normalize similarity to [0, 1]
            sim_min = sim_matrix.min()
            sim_max = sim_matrix.max()
            if sim_max > sim_min:
                sim_norm = (sim_matrix - sim_min) / (sim_max - sim_min)
            else:
                sim_norm = np.ones_like(sim_matrix)

            # Distance = 1 - normalized_similarity
            dist_matrix = 1.0 - sim_norm
            np.fill_diagonal(dist_matrix, 0.0)

            # Convert to condensed form for scipy
            dist_condensed = squareform(dist_matrix, checks=False)

            # Hierarchical clustering
            Z = linkage(dist_condensed, method="ward")

            # Labels are code indices
            labels = [str(ci) for ci in code_indices]

            dendrogram(
                Z,
                labels=labels,
                ax=ax,
                leaf_rotation=90,
                leaf_font_size=max(5, min(8, 200 // num_codes)),
                color_threshold=0.7 * max(Z[:, 2]) if len(Z) > 0 else 0,
                above_threshold_color=STRATUM_COLORS[k],
            )

            ax.set_title(
                f"k={k} ({num_codes} codes)",
                fontsize=13,
                fontweight="bold",
                color="white",
            )
            ax.set_xlabel("Code index", fontsize=10)
            ax.set_ylabel("Distance (1 - norm. similarity)", fontsize=10)
            ax.tick_params(colors="white")

        fig.suptitle(
            "Hierarchical Clustering of Doubly-Even Codes by Gadget Similarity",
            fontsize=16,
            fontweight="bold",
            color="white",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {output_path}")


def plot_similarity_heatmap_k4(
    code_groups: dict[int, list[dict]],
    strata_by_k: dict[int, dict],
    output_path: str,
) -> None:
    """
    Code-code similarity heatmap for k=4 (38 codes), reordered by hierarchical clustering.
    """
    print("  Building k=4 similarity heatmap ...")

    k = 4
    codes = code_groups[k]
    stratum = strata_by_k[k]
    mat = np.array(stratum["matrix"], dtype=np.float64)

    sim_matrix, code_indices = compute_code_similarity_matrix(mat, codes)
    num_codes = len(codes)

    # Normalize for clustering
    sim_min = sim_matrix.min()
    sim_max = sim_matrix.max()
    if sim_max > sim_min:
        sim_norm = (sim_matrix - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.ones_like(sim_matrix)

    # Hierarchical clustering for reordering
    dist_matrix = 1.0 - sim_norm
    np.fill_diagonal(dist_matrix, 0.0)
    dist_condensed = squareform(dist_matrix, checks=False)
    Z = linkage(dist_condensed, method="ward")

    # Get leaf order from dendrogram
    dn = dendrogram(Z, no_plot=True)
    leaf_order = list(map(int, dn["leaves"]))

    # Reorder similarity matrix
    sim_reordered = sim_matrix[np.ix_(leaf_order, leaf_order)]
    labels_reordered = [str(code_indices[i]) for i in leaf_order]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(14, 12))

        im = ax.imshow(
            sim_reordered,
            cmap="inferno",
            aspect="equal",
            origin="upper",
        )

        ax.set_xticks(range(num_codes))
        ax.set_yticks(range(num_codes))
        ax.set_xticklabels(labels_reordered, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_reordered, fontsize=7)
        ax.set_xlabel("Code index (clustered order)", fontsize=11)
        ax.set_ylabel("Code index (clustered order)", fontsize=11)

        ax.set_title(
            f"Code-Code Average Gadget Similarity, k=4 ({num_codes} codes, hierarchically clustered)",
            fontsize=13,
            fontweight="bold",
            color="white",
            pad=12,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Average gadget value", fontsize=10)

        fig.tight_layout()
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {output_path}")


def plot_all_codes_landscape(
    results: list[dict],
    output_path: str,
) -> None:
    """
    Bubble chart: all 145 codes arranged by k (y-axis) and code_index (x-axis),
    with bubble size proportional to num_dashings.
    """
    print("  Building all-codes landscape ...")

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(20, 8))

        for r in results:
            ci = r["code_index"]
            k = r["k"]
            nd = r["num_dashings"]
            color = STRATUM_COLORS[k]

            # Size: scale log2(num_dashings) for visual balance
            # num_dashings ranges from 2 (k=1) to 256 (k=8)
            bubble_size = 20 + 8 * np.log2(nd) ** 2
            ax.scatter(
                ci, k,
                s=bubble_size,
                c=color,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

        # Annotate stratum statistics
        by_k = build_code_groups(results)
        for k_val in sorted(by_k.keys()):
            codes = by_k[k_val]
            nd = codes[0]["num_dashings"]
            count = len(codes)
            ax.annotate(
                f"{count} codes, 2^{int(np.log2(nd))} dashings",
                xy=(max(r["code_index"] for r in results) + 2, k_val),
                fontsize=8,
                color=STRATUM_COLORS[k_val],
                va="center",
                fontweight="bold",
            )

        ax.set_xlabel("Code index", fontsize=12)
        ax.set_ylabel("k (stratum)", fontsize=12)
        ax.set_yticks(range(1, 9))
        ax.set_yticklabels([f"k={k}" for k in range(1, 9)])
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        ax.set_title(
            "N=16 Doubly-Even Code Landscape (145 codes, bubble size ~ num dashings)",
            fontsize=15,
            fontweight="bold",
            color="white",
            pad=12,
        )

        ax.grid(True, alpha=0.15, linestyle="--")
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  Saved {output_path}")


def main() -> None:
    data = load_data(INPUT_PATH)
    results = data["results"]
    strata = data["gadget_strata"]

    # Build lookups
    code_groups = build_code_groups(results)
    strata_by_k = {s["k"]: s for s in strata}

    print(f"\nCode distribution:")
    for k in sorted(code_groups.keys()):
        print(f"  k={k}: {len(code_groups[k])} codes")

    print(f"\n--- Generating visualizations ---\n")

    # 1. Code similarity network (k=2..7)
    plot_code_network(
        code_groups, strata_by_k,
        os.path.join(OUTPUT_DIR, "code_network_by_stratum.png"),
    )

    # 2. Dendrograms (k=3, 4, 5)
    plot_dendrograms(
        code_groups, strata_by_k,
        os.path.join(OUTPUT_DIR, "code_dendrogram.png"),
    )

    # 3. k=4 similarity heatmap
    plot_similarity_heatmap_k4(
        code_groups, strata_by_k,
        os.path.join(OUTPUT_DIR, "code_similarity_k4.png"),
    )

    # 4. All-codes landscape
    plot_all_codes_landscape(
        results,
        os.path.join(OUTPUT_DIR, "all_codes_landscape.png"),
    )

    print("\nDone. All outputs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
