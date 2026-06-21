#!/usr/bin/env python3
"""
N=16 Adinkra Representation Landscape Dashboard
================================================
Generates a 2x3 multi-panel figure from the complete N=16 pipeline output,
plus individual panel images.

Usage:
    python viz/strata_dashboard.py [path_to_json]

Default JSON path: /tmp/n16_full_pipeline.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/n16_full_pipeline.json"
OUT_DIR = Path(__file__).resolve().parent
DPI = 300

# Dark style
plt.style.use("dark_background")

# Colormap: one color per k-stratum (k=1..8), consistent across panels
CMAP = plt.cm.plasma
K_VALUES = list(range(1, 9))
K_COLORS = {k: CMAP((k - 1) / (len(K_VALUES) - 1)) for k in K_VALUES}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def aggregate(data: dict) -> dict:
    """Pre-compute per-stratum aggregates."""
    by_k = defaultdict(list)
    for r in data["results"]:
        by_k[r["k"]].append(r)

    strata_lookup = {s["k"]: s for s in data["gadget_strata"]}

    agg = {}
    for k in sorted(by_k):
        codes = by_k[k]
        d = 2 ** (15 - k)
        num_codes = len(codes)
        num_reps = sum(c["num_dashings"] for c in codes)
        gadget_vals = []
        for c in codes:
            gadget_vals.extend(c["gadget_self_values"])
        all_garden = all(c["garden_algebra_verified"] for c in codes)
        all_worldsheet = all(c.get("worldsheet_trivial", False) for c in codes)
        mat_size = strata_lookup[k]["num_reps"] if k in strata_lookup else 0
        agg[k] = {
            "d": d,
            "num_codes": num_codes,
            "num_reps": num_reps,
            "gadget_self_values": gadget_vals,
            "gadget_self_unique": sorted(set(gadget_vals)),
            "all_garden": all_garden,
            "all_worldsheet": all_worldsheet,
            "matrix_size": mat_size,
        }
    return agg


# ---------------------------------------------------------------------------
# Individual panel functions
# ---------------------------------------------------------------------------
def panel_codes_per_stratum(ax, agg):
    """Bar chart: number of doubly-even codes at each k."""
    ks = sorted(agg)
    counts = [agg[k]["num_codes"] for k in ks]
    colors = [K_COLORS[k] for k in ks]

    bars = ax.bar(ks, counts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(c),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="white",
        )
    ax.set_xlabel("k (code dimension)", fontsize=10)
    ax.set_ylabel("Number of codes", fontsize=10)
    ax.set_title("Doubly-Even Codes per Stratum", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, max(counts) * 1.2)


def panel_reps_per_stratum(ax, agg):
    """Bar chart: total representations (dashings) per k-stratum."""
    ks = sorted(agg)
    reps = [agg[k]["num_reps"] for k in ks]
    colors = [K_COLORS[k] for k in ks]

    bars = ax.bar(ks, reps, color=colors, edgecolor="white", linewidth=0.5)
    for bar, r in zip(bars, reps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(r),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color="white",
        )
    ax.set_xlabel("k (code dimension)", fontsize=10)
    ax.set_ylabel("Number of representations", fontsize=10)
    ax.set_title("Representations per Stratum", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, max(reps) * 1.2)


def panel_dimension_scaling(ax, agg):
    """Plot d = 2^(15-k) vs k with log-scale y-axis."""
    ks = sorted(agg)
    ds = [agg[k]["d"] for k in ks]
    colors = [K_COLORS[k] for k in ks]

    ax.plot(ks, ds, "w--", alpha=0.4, linewidth=1)
    ax.scatter(ks, ds, c=colors, s=80, edgecolors="white", linewidths=0.8, zorder=5)

    for k, d in zip(ks, ds):
        ax.annotate(
            f"{d:,}",
            (k, d),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
            color="white",
        )

    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("k (code dimension)", fontsize=10)
    ax.set_ylabel("d = 2^(15-k)  (codeword count)", fontsize=10)
    ax.set_title("Dimension Scaling (log₂ y-axis)", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])

    # Annotate dmin
    ax.axhline(y=128, color="cyan", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(
        ks[-1] + 0.15,
        128,
        "d_min = 128",
        fontsize=7,
        color="cyan",
        va="center",
    )


def panel_self_gadget(ax, agg):
    """Scatter: d/d_min = d/128 vs k for each stratum, showing the ratio."""
    ks = sorted(agg)
    dmin = 128
    ratios = [agg[k]["d"] / dmin for k in ks]
    colors = [K_COLORS[k] for k in ks]

    # Also show gadget self-value (constant per stratum here)
    gadget_vals = [agg[k]["gadget_self_unique"][0] for k in ks]

    ax.scatter(ks, ratios, c=colors, s=100, edgecolors="white", linewidths=0.8, zorder=5, marker="D")
    ax.plot(ks, ratios, "w--", alpha=0.3, linewidth=1)

    for k, ratio, gv in zip(ks, ratios, gadget_vals):
        label = f"d/d_min={ratio:.0f}\ngadget={gv:.0f}"
        ax.annotate(
            label,
            (k, ratio),
            textcoords="offset points",
            xytext=(12, -2),
            fontsize=6,
            color="white",
            ha="left",
        )

    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:.2f}"))
    ax.set_xlabel("k (code dimension)", fontsize=10)
    ax.set_ylabel("d / d_min", fontsize=10)
    ax.set_title("Self-Gadget Values & d/d_min Ratio", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])

    ax.axhline(y=1, color="lime", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(ks[0] - 0.3, 1, "d = d_min", fontsize=7, color="lime", va="bottom")


def panel_matrix_sizes(ax, agg):
    """Bar chart: gadget matrix dimensions (num_reps x num_reps) per stratum."""
    ks = sorted(agg)
    sizes = [agg[k]["matrix_size"] for k in ks]
    colors = [K_COLORS[k] for k in ks]

    bars = ax.bar(ks, sizes, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate the big ones (top 3)
    threshold = sorted(sizes, reverse=True)[2]
    for bar, s, k in zip(bars, sizes, ks):
        fontsize = 8 if s >= threshold else 7
        weight = "bold" if s >= threshold else "normal"
        color = "yellow" if s >= threshold else "white"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{s}x{s}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight=weight,
            color=color,
        )

    ax.set_xlabel("k (code dimension)", fontsize=10)
    ax.set_ylabel("Matrix dimension (n x n)", fontsize=10)
    ax.set_title("Gadget Matrix Sizes per Stratum", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, max(sizes) * 1.2)


def panel_summary(ax, data, agg, elapsed):
    """Text panel with key computation stats."""
    ax.axis("off")

    total_codes = data["num_codes"]
    total_reps = data["total_reps"]
    all_garden = all(a["all_garden"] for a in agg.values())
    all_worldsheet = all(a["all_worldsheet"] for a in agg.values())

    # Check all gadget self-values positive (PSD)
    all_psd = True
    for a in agg.values():
        for v in a["gadget_self_values"]:
            if v < 0:
                all_psd = False
                break

    num_strata = len(agg)
    max_matrix_k = max(agg, key=lambda k: agg[k]["matrix_size"])
    max_matrix_sz = agg[max_matrix_k]["matrix_size"]
    max_codes_k = max(agg, key=lambda k: agg[k]["num_codes"])
    max_reps_k = max(agg, key=lambda k: agg[k]["num_reps"])

    # Total matrix entries computed
    total_entries = sum(agg[k]["matrix_size"] ** 2 for k in agg)

    lines = [
        ("COMPUTATION SUMMARY", 14, "bold", "cyan"),
        ("", 10, "normal", "white"),
        (f"N = {data['n']}    (supercharge count)", 11, "normal", "white"),
        (f"Doubly-even codes:  {total_codes}", 11, "normal", "white"),
        (f"Total representations:  {total_reps:,}", 11, "normal", "white"),
        (f"Strata (distinct k):  {num_strata}", 11, "normal", "white"),
        (f"d_min(16) = 128    (k=8 stratum)", 11, "normal", "white"),
        ("", 10, "normal", "white"),
        ("VERIFICATION", 12, "bold", "lime"),
        (f"Garden algebra:  {'PASS (all {0})'.format(total_codes)}" if all_garden else "FAIL", 10, "normal", "lime" if all_garden else "red"),
        (f"Worldsheet trivial:  {'PASS (all {0})'.format(total_codes)}" if all_worldsheet else "FAIL", 10, "normal", "lime" if all_worldsheet else "red"),
        (f"Gadget PSD:  {'PASS (all self-values >= 0)' if all_psd else 'FAIL'}", 10, "normal", "lime" if all_psd else "red"),
        ("", 10, "normal", "white"),
        ("SCALE", 12, "bold", "orange"),
        (f"Largest gadget matrix:  {max_matrix_sz}x{max_matrix_sz}  (k={max_matrix_k})", 10, "normal", "white"),
        (f"Total matrix entries:  {total_entries:,}", 10, "normal", "white"),
        (f"Peak codes at k={max_codes_k}:  {agg[max_codes_k]['num_codes']}", 10, "normal", "white"),
        (f"Peak reps at k={max_reps_k}:  {agg[max_reps_k]['num_reps']:,}", 10, "normal", "white"),
        (f"Elapsed:  {elapsed:.1f}s", 10, "normal", "white"),
    ]

    y = 0.95
    for text, size, weight, color in lines:
        ax.text(
            0.08,
            y,
            text,
            transform=ax.transAxes,
            fontsize=size,
            fontweight=weight,
            color=color,
            verticalalignment="top",
            fontfamily="monospace",
        )
        y -= 0.052


# ---------------------------------------------------------------------------
# Save individual panels
# ---------------------------------------------------------------------------
PANEL_FUNCS = {
    "codes_per_stratum": panel_codes_per_stratum,
    "reps_per_stratum": panel_reps_per_stratum,
    "dimension_scaling": panel_dimension_scaling,
    "self_gadget_values": panel_self_gadget,
    "matrix_sizes": panel_matrix_sizes,
}


def save_individual_panels(agg, data, elapsed):
    """Save each panel as its own PNG."""
    for name, func in PANEL_FUNCS.items():
        fig_i, ax_i = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
        ax_i.set_facecolor("#0d1117")
        func(ax_i, agg)
        fig_i.tight_layout(pad=1.5)
        out = OUT_DIR / f"panel_{name}.png"
        fig_i.savefig(out, dpi=DPI, facecolor="#0d1117", bbox_inches="tight")
        plt.close(fig_i)
        print(f"  Saved {out}")

    # Summary panel
    fig_s, ax_s = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax_s.set_facecolor("#0d1117")
    panel_summary(ax_s, data, agg, elapsed)
    fig_s.tight_layout(pad=1.5)
    out = OUT_DIR / "panel_computation_summary.png"
    fig_s.savefig(out, dpi=DPI, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig_s)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {JSON_PATH} ...")
    data = load_data(JSON_PATH)
    elapsed = data.get("elapsed_secs", 0.0)
    agg = aggregate(data)

    print(f"  {data['num_codes']} codes, {data['total_reps']} reps, {len(agg)} strata")

    # ---- Composite dashboard (2x3) ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#0d1117")

    fig.suptitle(
        "N=16 Adinkra Representation Landscape: Complete Classification",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    # Row 0
    panel_codes_per_stratum(axes[0, 0], agg)
    panel_reps_per_stratum(axes[0, 1], agg)
    panel_dimension_scaling(axes[0, 2], agg)

    # Row 1
    panel_self_gadget(axes[1, 0], agg)
    panel_matrix_sizes(axes[1, 1], agg)
    panel_summary(axes[1, 2], data, agg, elapsed)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=3.0)

    dashboard_path = OUT_DIR / "strata_dashboard.png"
    fig.savefig(dashboard_path, dpi=DPI, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    print(f"\nDashboard saved: {dashboard_path}")

    # ---- Individual panels ----
    print("\nSaving individual panels:")
    save_individual_panels(agg, data, elapsed)

    print("\nDone.")


if __name__ == "__main__":
    main()
