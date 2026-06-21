#!/usr/bin/env python3
"""
Visualize the distribution of gadget values across all N=16 adinkra strata.

Produces five figures:
  1. gadget_ridge_plot.png        - Ridge/density plots of off-diagonal gadget values per k-stratum
  2. gadget_discrete_values.png   - Stem plots of sorted unique gadget values per stratum
  3. gadget_cdf.png               - Cumulative distribution functions per stratum
  4. gadget_intra_vs_inter.png    - Intra-code vs inter-code gadget value histograms
  5. gadget_stats_table.png       - Summary statistics table
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PIPELINE_FILE = "/tmp/n16_full_pipeline.json"
OUTPUT_DIR = os.path.expanduser("~/code/adinkra-codespace/viz")
DPI = 300
N = 16
DMIN = 128
GADGET_QUANTUM = 1.0 / (N * (N - 1) * DMIN / 2)  # = 1/15360

# Consistent colour palette per stratum (k=1..8)
STRATUM_COLORS = [
    "#FF6B6B",  # k=1  coral red
    "#FFA94D",  # k=2  orange
    "#FFD93D",  # k=3  gold
    "#6BCB77",  # k=4  green
    "#4D96FF",  # k=5  blue
    "#9B59B6",  # k=6  purple
    "#E056A0",  # k=7  magenta-pink
    "#00CEC9",  # k=8  teal
]

plt.style.use("dark_background")


def load_data():
    """Load pipeline JSON and return strata list and per-code results."""
    print("Loading pipeline data ...")
    with open(PIPELINE_FILE) as f:
        data = json.load(f)
    strata = data["gadget_strata"]
    results = data["results"]
    return strata, results


def extract_off_diag(matrix_np):
    """Return array of upper-triangle off-diagonal values."""
    n = matrix_np.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix_np[idx]


def build_code_boundaries(results):
    """
    For each k-stratum, return a list of (start, end) index pairs
    delineating which matrix rows/cols belong to each code.
    Codes are ordered by code_index within each k.
    """
    by_k = defaultdict(list)
    for r in results:
        by_k[r["k"]].append(r)

    boundaries = {}
    for k_val in sorted(by_k):
        codes = sorted(by_k[k_val], key=lambda r: r["code_index"])
        bounds = []
        offset = 0
        for c in codes:
            nd = c["num_dashings"]
            bounds.append((offset, offset + nd))
            offset += nd
        boundaries[k_val] = bounds
    return boundaries


def classify_pairs(matrix_np, bounds):
    """
    Classify off-diagonal upper-triangle pairs into intra-code and inter-code.
    Returns (intra_values, inter_values) as numpy arrays.
    """
    n = matrix_np.shape[0]
    # Build a code-id array
    code_id = np.zeros(n, dtype=int)
    for ci, (s, e) in enumerate(bounds):
        code_id[s:e] = ci

    idx_i, idx_j = np.triu_indices(n, k=1)
    vals = matrix_np[idx_i, idx_j]
    same_code = code_id[idx_i] == code_id[idx_j]
    return vals[same_code], vals[~same_code]


# ---------------------------------------------------------------------------
# Figure 1: Ridge / density plot
# ---------------------------------------------------------------------------
def plot_ridge(strata):
    print("Generating ridge plot ...")
    fig, ax = plt.subplots(figsize=(14, 12))

    from scipy.stats import gaussian_kde

    RIDGE_HEIGHT = 1.0   # fixed visual height for each ridge
    RIDGE_GAP = 0.15     # gap between ridges

    y_offset = 0
    y_ticks = []
    y_labels = []

    for i, s in enumerate(strata):
        k = s["k"]
        mat = np.array(s["matrix"], dtype=np.float64)
        vals = extract_off_diag(mat)
        if len(vals) < 2:
            continue

        # KDE -- adaptive bandwidth: tighter for narrower distributions
        x_range = vals.max() - vals.min()
        bw = max(0.02, min(0.1, x_range / 50))
        kde = gaussian_kde(vals, bw_method=bw)
        x_min, x_max = vals.min(), vals.max()
        margin = x_range * 0.15 + 0.1
        xs = np.linspace(x_min - margin, x_max + margin, 2000)
        density = kde(xs)

        # Normalise density so peak = RIDGE_HEIGHT
        density = density / density.max() * RIDGE_HEIGHT

        color = STRATUM_COLORS[i]
        ax.fill_between(xs, y_offset, y_offset + density,
                         alpha=0.55, color=color, linewidth=0)
        ax.plot(xs, y_offset + density, color=color, linewidth=1.2)

        y_ticks.append(y_offset + RIDGE_HEIGHT * 0.25)
        d = s["d"]
        y_labels.append(f"k={k}  (d={d}, n={s['num_reps']})")
        y_offset += RIDGE_HEIGHT + RIDGE_GAP

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Gadget Value  G(R, R')", fontsize=12)
    ax.set_title("Distribution of Off-Diagonal Gadget Values by k-Stratum  (N=16)",
                 fontsize=14, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.1, y_offset)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "gadget_ridge_plot.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Gadget discreteness (stem plots)
# ---------------------------------------------------------------------------
def plot_discrete(strata):
    print("Generating discrete values plot ...")
    nk = len(strata)
    fig, axes = plt.subplots(nk, 1, figsize=(16, 3 * nk), sharex=False)
    if nk == 1:
        axes = [axes]

    for i, s in enumerate(strata):
        ax = axes[i]
        k = s["k"]
        mat = np.array(s["matrix"], dtype=np.float64)
        vals = extract_off_diag(mat)
        unique_vals = np.unique(np.round(vals, decimals=10))

        color = STRATUM_COLORS[i]

        # Stem plot: markerline, stemlines, baseline
        markerline, stemlines, baseline = ax.stem(
            np.arange(len(unique_vals)), unique_vals,
            linefmt="-", markerfmt="o", basefmt=" "
        )
        markerline.set_color(color)
        markerline.set_markersize(3)
        stemlines.set_color(color)
        stemlines.set_alpha(0.5)
        stemlines.set_linewidth(0.6)

        ax.set_ylabel("Gadget value", fontsize=9)
        ax.set_title(
            f"k={k}:  {len(unique_vals)} distinct values   "
            f"(quantum = 1/{int(1/GADGET_QUANTUM)} = {GADGET_QUANTUM:.2e})",
            fontsize=10, loc="left"
        )
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Sorted unique value index", fontsize=10)
    fig.suptitle("Discreteness of Gadget Values  (N=16, dmin=128)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "gadget_discrete_values.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: CDF of off-diagonal gadget values
# ---------------------------------------------------------------------------
def plot_cdf(strata):
    print("Generating CDF plot ...")
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, s in enumerate(strata):
        k = s["k"]
        mat = np.array(s["matrix"], dtype=np.float64)
        vals = extract_off_diag(mat)
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        color = STRATUM_COLORS[i]
        ax.plot(sorted_vals, cdf, color=color, linewidth=1.5,
                label=f"k={k}  (d={s['d']}, n={s['num_reps']})")

    ax.set_xlabel("Gadget Value  G(R, R')", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("CDF of Off-Diagonal Gadget Values by Stratum  (N=16)",
                 fontsize=14, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "gadget_cdf.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Intra-code vs inter-code gadget comparison
# ---------------------------------------------------------------------------
def plot_intra_inter(strata, results):
    print("Generating intra vs inter-code plot ...")
    boundaries = build_code_boundaries(results)

    # Only strata with at least 2 codes and meaningful intra-code pairs
    eligible = [s for s in strata if len(boundaries.get(s["k"], [])) >= 2
                and any(e - b >= 2 for b, e in boundaries.get(s["k"], []))]

    nk = len(eligible)
    fig, axes = plt.subplots(nk, 1, figsize=(14, 3.5 * nk), sharex=False)
    if nk == 1:
        axes = [axes]

    for idx, s in enumerate(eligible):
        ax = axes[idx]
        k = s["k"]
        mat = np.array(s["matrix"], dtype=np.float64)
        bounds = boundaries[k]
        intra, inter = classify_pairs(mat, bounds)

        # Choose bins that cover the full range
        all_vals = np.concatenate([intra, inter])
        lo, hi = all_vals.min(), all_vals.max()
        bins = np.linspace(lo, hi, min(80, max(30, len(np.unique(np.round(all_vals, 8))) // 2)))

        ax.hist(inter, bins=bins, alpha=0.5, color="#4D96FF",
                label=f"Inter-code ({len(inter):,} pairs)", density=True)
        ax.hist(intra, bins=bins, alpha=0.6, color="#FF6B6B",
                label=f"Intra-code ({len(intra):,} pairs)", density=True)

        ax.set_title(f"k={k}   (d={s['d']},  {len(bounds)} codes,  {s['num_reps']} reps)",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Gadget Value  G(R, R')", fontsize=11)
    fig.suptitle("Intra-Code vs Inter-Code Gadget Values  (N=16)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "gadget_intra_vs_inter.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Statistics summary table
# ---------------------------------------------------------------------------
def plot_stats_table(strata, results):
    print("Generating statistics table ...")
    boundaries = build_code_boundaries(results)

    rows = []
    for s in strata:
        k = s["k"]
        d = s["d"]
        nr = s["num_reps"]
        mat = np.array(s["matrix"], dtype=np.float64)
        vals = extract_off_diag(mat)
        unique_vals = np.unique(np.round(vals, decimals=10))

        # Check if row sums are constant
        row_sums = mat.sum(axis=1)
        row_sum_std = np.std(row_sums)
        row_sum_const = "Yes" if row_sum_std < 1e-6 else f"No (std={row_sum_std:.4f})"
        row_sum_mean = np.mean(row_sums)

        # Ratio of values to gadget quantum
        min_gap = np.min(np.diff(unique_vals)) if len(unique_vals) > 1 else 0
        quantum_ratio = min_gap / GADGET_QUANTUM if GADGET_QUANTUM > 0 and min_gap > 0 else 0

        rows.append([
            f"k={k}",
            f"{d}",
            f"{nr}",
            f"{vals.min():.4f}",
            f"{vals.max():.4f}",
            f"{vals.mean():.4f}",
            f"{vals.std():.4f}",
            f"{len(unique_vals)}",
            f"{min_gap:.6f}",
            f"{quantum_ratio:.1f}",
            row_sum_const,
        ])

    col_labels = [
        "Stratum", "d", "# Reps",
        "Min G", "Max G", "Mean G", "Std G",
        "# Distinct", "Min Gap", "Gap/Q",
        "Row Sum Const?"
    ]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            if i % 2 == 0:
                cell.set_facecolor("#1a1a2e")
            else:
                cell.set_facecolor("#16213e")
            cell.set_text_props(color="white")

    ax.set_title(
        f"Gadget Statistics Summary  (N={N}, dmin={DMIN}, quantum=1/{int(1/GADGET_QUANTUM)})",
        fontsize=14, pad=20, color="white"
    )

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "gadget_stats_table.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    strata, results = load_data()

    plot_ridge(strata)
    plot_discrete(strata)
    plot_cdf(strata)
    plot_intra_inter(strata, results)
    plot_stats_table(strata, results)

    print("\nAll figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
