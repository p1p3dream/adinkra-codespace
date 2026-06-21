#!/usr/bin/env python3
"""Analyze the full N=16 pipeline output."""

import json
import sys
from collections import Counter
from fractions import Fraction

import numpy as np


def to_fraction(g, n=16, dmin=128):
    denom = n * (n - 1) * dmin // 2  # 15360
    numerator = round(g * denom)
    return Fraction(numerator, denom)


def analyze_stratum(stratum, results_by_code):
    k = stratum["k"]
    d = stratum["d"]
    num_reps = stratum["num_reps"]
    matrix = np.array(stratum["matrix"])

    print(f"\n{'='*70}")
    print(f"  k={k}  |  d={d}  |  {num_reps} reps  |  matrix {num_reps}x{num_reps}")
    print(f"{'='*70}")

    # Diagonal check
    diag = np.diag(matrix)
    dmin_val = 128  # dmin(16) = 128
    expected_diag = d / dmin_val
    all_diag_ok = np.allclose(diag, expected_diag)
    print(f"  Diagonal = d/dmin = {d}/{dmin_val} = {expected_diag:.4f}: {'PASS' if all_diag_ok else 'FAIL'}")

    # Symmetry
    max_asym = np.max(np.abs(matrix - matrix.T))
    print(f"  Symmetry: max |G[i,j]-G[j,i]| = {max_asym:.1e}")

    # Eigenvalues
    eigs = np.linalg.eigvalsh(matrix)
    min_eig = eigs[0]
    max_eig = eigs[-1]
    neg_count = np.sum(eigs < -1e-8)
    zero_count = np.sum(np.abs(eigs) < 1e-8)
    print(f"  PSD: min eigenvalue = {min_eig:.6f} ({'PASS' if min_eig > -1e-8 else 'FAIL'})")
    print(f"  Eigenvalue range: [{min_eig:.4f}, {max_eig:.4f}]")
    print(f"  Rank: {num_reps - zero_count} (nullity {zero_count})")

    # Distinct eigenvalues
    eig_counts = Counter(round(e, 4) for e in eigs)
    num_distinct = len(eig_counts)
    print(f"  Distinct eigenvalues: {num_distinct}")
    if num_distinct <= 15:
        for val, count in sorted(eig_counts.items()):
            print(f"    lambda = {val:+.4f} (mult {count})")

    # Row sums
    row_sums = matrix.sum(axis=1)
    rs_min, rs_max = row_sums.min(), row_sums.max()
    rs_const = rs_max - rs_min < 1e-6
    print(f"  Row sums: {'constant' if rs_const else 'NOT constant'} = {rs_min:.6f}" +
          (f" to {rs_max:.6f}" if not rs_const else ""))

    # Off-diagonal distinct values
    off_diag = []
    for i in range(num_reps):
        for j in range(i + 1, num_reps):
            off_diag.append(matrix[i][j])
    vals = sorted(set(round(v, 6) for v in off_diag))
    print(f"  Distinct off-diagonal gadget values: {len(vals)}")
    print(f"  Gadget range: [{min(off_diag):.6f}, {max(off_diag):.6f}]")

    # Value distribution (top 10)
    val_counts = Counter(round(v, 6) for v in off_diag)
    print(f"  Top 10 most common values:")
    for val, count in val_counts.most_common(10):
        frac = to_fraction(val)
        print(f"    G = {val:+.6f} = {frac}  ({count} pairs)")

    # Per-code breakdown within this stratum
    codes_in_stratum = [r for r in results_by_code if r["k"] == k]
    if len(codes_in_stratum) > 1:
        print(f"\n  Per-code breakdown ({len(codes_in_stratum)} codes):")
        offset = 0
        for r in codes_in_stratum:
            nd = r["num_dashings"]
            print(f"    Code {r['code_index']}: {nd} dashings (reps {offset}..{offset+nd-1})")
            offset += nd

    return {
        "k": k,
        "d": d,
        "num_reps": num_reps,
        "psd": min_eig > -1e-8,
        "rank": num_reps - zero_count,
        "distinct_gadgets": len(vals),
        "distinct_eigenvalues": num_distinct,
        "row_sum": float(rs_min),
        "gadget_range": (float(min(off_diag)), float(max(off_diag))),
        "max_eigenvalue": float(max_eig),
    }


def main():
    data = json.load(sys.stdin)
    n = data["n"]
    num_codes = data["num_codes"]
    total_reps = data["total_reps"]
    elapsed = data["elapsed_secs"]

    print(f"N={n} Full Pipeline Results")
    print(f"{'='*70}")
    print(f"  Codes: {num_codes}")
    print(f"  Total representations: {total_reps}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # Garden algebra check
    all_garden = all(r["garden_algebra_verified"] for r in data["results"])
    print(f"  Garden algebra: {'ALL PASS' if all_garden else 'FAILURES DETECTED'}")

    # Self-gadget check
    all_self = all(
        all(abs(g - 1.0) < 1e-9 for g in r["gadget_self_values"])
        for r in data["results"]
        if r["d"] == 128  # only irreducible reps have self-gadget = 1
    )
    print(f"  Self-gadget (irreducible, d=dmin): {'ALL = 1.0' if all_self else 'FAILURES'}")

    dmin_val = 128
    all_self_general = all(
        all(abs(g - r["d"] / dmin_val) < 1e-9 for g in r["gadget_self_values"])
        for r in data["results"]
    )
    print(f"  Self-gadget (general, d/dmin): {'ALL CORRECT' if all_self_general else 'FAILURES'}")

    # Per-stratum analysis
    summaries = []
    for stratum in data["gadget_strata"]:
        s = analyze_stratum(stratum, data["results"])
        summaries.append(s)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'k':>2} | {'d':>5} | {'reps':>5} | {'rank':>5} | {'PSD':>4} | {'dist_G':>6} | {'dist_eig':>7} | {'row_sum':>10} | {'G_range'}")
    print(f"  {'-'*2}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*20}")
    for s in summaries:
        print(f"  {s['k']:>2} | {s['d']:>5} | {s['num_reps']:>5} | {s['rank']:>5} | {'Y' if s['psd'] else 'N':>4} | {s['distinct_gadgets']:>6} | {s['distinct_eigenvalues']:>7} | {s['row_sum']:>10.4f} | [{s['gadget_range'][0]:.4f}, {s['gadget_range'][1]:.4f}]")

    total_rank = sum(s["rank"] for s in summaries)
    total_distinct = sum(s["distinct_gadgets"] for s in summaries)
    all_psd = all(s["psd"] for s in summaries)
    print(f"\n  Total rank across all strata: {total_rank}")
    print(f"  Total distinct gadget values: {total_distinct}")
    print(f"  All strata PSD: {'YES' if all_psd else 'NO'}")


if __name__ == "__main__":
    main()
