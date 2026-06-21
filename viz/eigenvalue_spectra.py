#!/usr/bin/env python3
"""
Eigenvalue spectra visualizations for N=16 adinkra gadget matrices.

Reads the full pipeline output from /tmp/n16_full_pipeline.json and produces
four publication-quality figures characterizing the spectral structure of the
gadget matrices across all eight k-strata.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = Path("/tmp/n16_full_pipeline.json")
OUTPUT_DIR = Path(__file__).resolve().parent  # viz/
DPI = 300

# Palette: one color per stratum (k=1..8), chosen for contrast on dark bg
STRATUM_COLORS = {
    1: "#FF6B6B",  # coral
    2: "#FFD93D",  # gold
    3: "#6BCB77",  # green
    4: "#4D96FF",  # blue
    5: "#C77DFF",  # violet
    6: "#FF922B",  # orange
    7: "#20C997",  # teal
    8: "#F06595",  # pink
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_strata(path: Path) -> list[dict]:
    print(f"Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    strata = sorted(data["gadget_strata"], key=lambda s: s["k"])
    # Convert matrices to numpy immediately and print progress per stratum
    for s in strata:
        s["matrix"] = np.array(s["matrix"], dtype=np.float64)
        k, shape = s["k"], s["matrix"].shape
        print(f"  Stratum k={k}: loaded {shape[0]}x{shape[1]} matrix "
              f"(d={s['d']}, num_reps={s['num_reps']})")
    print(f"  All {len(strata)} strata loaded.")
    return strata


def compute_eigenvalues(strata: list[dict]) -> dict[int, np.ndarray]:
    """Return {k: sorted_eigenvalues} for every stratum."""
    eigs = {}
    for s in strata:
        k = s["k"]
        mat = s["matrix"]  # already np.float64 from load_strata
        vals = np.linalg.eigvalsh(mat)
        vals = np.sort(vals)[::-1]  # descending
        print(f"  k={k}: {mat.shape[0]}x{mat.shape[1]} -> {len(vals)} eigenvalues, "
              f"range [{vals[-1]:.4f}, {vals[0]:.4f}]")
        eigs[k] = vals
    return eigs


# ---------------------------------------------------------------------------
# Plot 1: Eigenvalue distribution per stratum (stem plots)
# ---------------------------------------------------------------------------
def plot_spectra_by_k(eigs: dict[int, np.ndarray]):
    """Horizontal stem plot of eigenvalues with multiplicity, one panel per k."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("N=16 Adinkra Gadget: Eigenvalue Spectra by Stratum",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, k in enumerate(sorted(eigs.keys())):
        ax = axes[idx // 4, idx % 4]
        vals = eigs[k]
        color = STRATUM_COLORS[k]

        # Compute multiplicities by rounding to avoid floating-point noise
        rounded = np.round(vals, decimals=8)
        counts = Counter(rounded)
        unique_vals = sorted(counts.keys(), reverse=True)
        mults = [counts[v] for v in unique_vals]

        # Stem plot: y = distinct eigenvalue, x = multiplicity
        y_pos = np.arange(len(unique_vals))
        ax.barh(y_pos, mults, color=color, alpha=0.85, height=0.7,
                edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{v:.4g}" for v in unique_vals],
                           fontsize=7 if len(unique_vals) > 30 else 8)
        ax.set_xlabel("Multiplicity", fontsize=10)
        ax.set_ylabel("Eigenvalue", fontsize=10)
        ax.set_title(f"k={k}  (d={2**(15-k+1)}, dim={len(vals)})",
                     fontsize=12, color=color, fontweight="bold")
        ax.invert_yaxis()

        # If too many distinct eigenvalues, thin out tick labels
        if len(unique_vals) > 40:
            for i, label in enumerate(ax.get_yticklabels()):
                if i % max(1, len(unique_vals) // 20) != 0:
                    label.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "eigenvalue_spectra_by_k.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Combined waterfall
# ---------------------------------------------------------------------------
def plot_waterfall(eigs: dict[int, np.ndarray]):
    """All strata eigenvalues in one figure, sorted descending per stratum."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("N=16 Adinkra Gadget: Eigenvalue Waterfall (All Strata)",
                 fontsize=15, fontweight="bold")

    for k in sorted(eigs.keys()):
        vals = eigs[k]
        indices = np.arange(len(vals))
        ax.plot(indices, vals, "o-", markersize=max(1, 5 - k // 2),
                linewidth=0.8, color=STRATUM_COLORS[k], alpha=0.85,
                label=f"k={k} (n={len(vals)})")

    ax.set_xlabel("Eigenvalue Index (sorted descending)", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.2)

    # Use symlog if there are negative eigenvalues
    all_vals = np.concatenate(list(eigs.values()))
    if np.any(all_vals < 0):
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel("Eigenvalue (symlog scale)", fontsize=12)

    plt.tight_layout()
    out = OUTPUT_DIR / "eigenvalue_waterfall.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: Eigenvalue histograms for large strata
# ---------------------------------------------------------------------------
def plot_histograms(eigs: dict[int, np.ndarray]):
    """Histogram of eigenvalue distribution for k=4,5,6."""
    target_ks = [k for k in [4, 5, 6] if k in eigs]
    fig, axes = plt.subplots(1, len(target_ks), figsize=(6 * len(target_ks), 5))
    if len(target_ks) == 1:
        axes = [axes]
    fig.suptitle("N=16 Adinkra Gadget: Eigenvalue Histograms (Large Strata)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, k in zip(axes, target_ks):
        vals = eigs[k]
        color = STRATUM_COLORS[k]
        nbins = min(80, max(20, len(set(np.round(vals, 6)))))
        ax.hist(vals, bins=nbins, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Eigenvalue", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"k={k}  (dim={len(vals)})", fontsize=12,
                     color=color, fontweight="bold")
        ax.axvline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.5)

        # Annotate basic stats
        ax.text(0.97, 0.95,
                f"min={vals[-1]:.2f}\nmax={vals[0]:.2f}\n"
                f"mean={np.mean(vals):.2f}\nmedian={np.median(vals):.2f}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    plt.tight_layout()
    out = OUTPUT_DIR / "eigenvalue_histograms.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 4: Rank vs nullity
# ---------------------------------------------------------------------------
def plot_rank_nullity(eigs: dict[int, np.ndarray]):
    """Stacked bar chart: rank and nullity for each stratum."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("N=16 Adinkra Gadget: Matrix Rank vs. Nullity by Stratum",
                 fontsize=14, fontweight="bold")

    ks = sorted(eigs.keys())
    dims = []
    ranks = []
    nullities = []

    for k in ks:
        vals = eigs[k]
        n = len(vals)
        # Count non-zero eigenvalues (eigenvalues > 1e-8 are "non-zero")
        r = int(np.sum(np.abs(vals) > 1e-8))
        dims.append(n)
        ranks.append(r)
        nullities.append(n - r)

    x = np.arange(len(ks))
    width = 0.55

    bars_rank = ax.bar(x, ranks, width, label="Rank",
                       color=[STRATUM_COLORS[k] for k in ks],
                       edgecolor="white", linewidth=0.5)
    bars_null = ax.bar(x, nullities, width, bottom=ranks, label="Nullity",
                       color=[STRATUM_COLORS[k] for k in ks], alpha=0.35,
                       edgecolor="white", linewidth=0.5, hatch="//")

    # Annotate bars
    for i, (k, r, nu, d) in enumerate(zip(ks, ranks, nullities, dims)):
        ax.text(i, d + d * 0.02, f"{d}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="white")
        if r > 0:
            ax.text(i, r / 2, f"r={r}", ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold")
        if nu > 0:
            ax.text(i, r + nu / 2, f"n={nu}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=11)
    ax.set_ylabel("Dimension", fontsize=12)
    ax.set_xlabel("Stratum", fontsize=12)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.7)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    out = OUTPUT_DIR / "rank_nullity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


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

    strata = load_strata(INPUT_PATH)
    eigs = compute_eigenvalues(strata)

    print("\nGenerating plots ...")
    plot_spectra_by_k(eigs)
    plot_waterfall(eigs)
    plot_histograms(eigs)
    plot_rank_nullity(eigs)
    print("\nDone.")


if __name__ == "__main__":
    main()
