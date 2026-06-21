#!/usr/bin/env python3
"""Fast analysis of the full N=16 pipeline output using vectorized numpy."""

import json
import sys
import numpy as np


def analyze_stratum(stratum, results_by_code):
    k = stratum["k"]
    d = stratum["d"]
    num_reps = stratum["num_reps"]
    matrix = np.array(stratum["matrix"], dtype=np.float64)
    dmin_val = 128

    print(f"\n{'='*70}")
    print(f"  k={k}  |  d={d}  |  {num_reps} reps  |  matrix {num_reps}x{num_reps}")
    print(f"{'='*70}")

    expected_diag = d / dmin_val
    diag = np.diag(matrix)
    all_diag_ok = np.allclose(diag, expected_diag)
    print(f"  Diagonal = d/dmin = {d}/{dmin_val} = {expected_diag:.4f}: {'PASS' if all_diag_ok else 'FAIL'}")

    max_asym = np.max(np.abs(matrix - matrix.T))
    print(f"  Symmetry: max |G[i,j]-G[j,i]| = {max_asym:.1e}")

    eigs = np.linalg.eigvalsh(matrix)
    min_eig = eigs[0]
    max_eig = eigs[-1]
    psd = min_eig > -1e-8
    zero_count = int(np.sum(np.abs(eigs) < 1e-8))
    rank = num_reps - zero_count
    print(f"  PSD: min eigenvalue = {min_eig:.6f} ({'PASS' if psd else 'FAIL'})")
    print(f"  Eigenvalue range: [{min_eig:.4f}, {max_eig:.4f}]")
    print(f"  Rank: {rank} (nullity {zero_count})")

    eig_rounded = np.round(eigs, 4)
    unique_eigs, eig_mults = np.unique(eig_rounded, return_counts=True)
    print(f"  Distinct eigenvalues: {len(unique_eigs)}")
    if len(unique_eigs) <= 12:
        for val, mult in zip(unique_eigs, eig_mults):
            print(f"    lambda = {val:+.4f} (mult {mult})")

    row_sums = matrix.sum(axis=1)
    rs_min, rs_max = row_sums.min(), row_sums.max()
    rs_const = rs_max - rs_min < 1e-6
    print(f"  Row sums: {'constant' if rs_const else 'VARYING'} = {rs_min:.6f}" +
          (f" to {rs_max:.6f}" if not rs_const else ""))

    # Vectorized off-diagonal extraction
    triu_idx = np.triu_indices(num_reps, k=1)
    off_diag = matrix[triu_idx]
    off_rounded = np.round(off_diag, 6)
    unique_vals = np.unique(off_rounded)
    print(f"  Distinct off-diagonal gadget values: {len(unique_vals)}")
    print(f"  Gadget range: [{off_diag.min():.6f}, {off_diag.max():.6f}]")

    # Top values by frequency (vectorized)
    vals, counts = np.unique(off_rounded, return_counts=True)
    top_idx = np.argsort(-counts)[:10]
    print(f"  Top 10 most common values:")
    for idx in top_idx:
        print(f"    G = {vals[idx]:+.6f}  ({counts[idx]} pairs)")

    # Per-code breakdown
    codes_in_stratum = [r for r in results_by_code if r["k"] == k]
    if len(codes_in_stratum) > 1 and len(codes_in_stratum) <= 40:
        print(f"\n  Per-code breakdown ({len(codes_in_stratum)} codes):")
        offset = 0
        for r in codes_in_stratum:
            nd = r["num_dashings"]
            print(f"    Code {r['code_index']}: {nd} dashings (reps {offset}..{offset+nd-1})")
            offset += nd

    return {
        "k": k, "d": d, "num_reps": num_reps,
        "psd": psd, "rank": rank,
        "distinct_gadgets": len(unique_vals),
        "distinct_eigenvalues": len(unique_eigs),
        "row_sum_min": float(rs_min),
        "row_sum_max": float(rs_max),
        "row_sum_const": rs_const,
        "gadget_min": float(off_diag.min()),
        "gadget_max": float(off_diag.max()),
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
    print(f"  Elapsed (Rust pipeline): {elapsed:.1f}s")

    all_garden = all(r["garden_algebra_verified"] for r in data["results"])
    print(f"  Garden algebra: {'ALL PASS' if all_garden else 'FAILURES DETECTED'}")

    dmin_val = 128
    all_self = all(
        all(abs(g - r["d"] / dmin_val) < 1e-9 for g in r["gadget_self_values"])
        for r in data["results"]
    )
    print(f"  Self-gadget (d/dmin): {'ALL CORRECT' if all_self else 'FAILURES'}")

    summaries = []
    for stratum in data["gadget_strata"]:
        s = analyze_stratum(stratum, data["results"])
        summaries.append(s)

    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    hdr = f"  {'k':>2} | {'d':>5} | {'reps':>5} | {'rank':>5} | {'PSD':>3} | {'distG':>5} | {'distE':>5} | {'rowsum_const':>12} | {'G_range':>24}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for s in summaries:
        rs = f"{s['row_sum_min']:.2f}" if s['row_sum_const'] else f"{s['row_sum_min']:.2f}-{s['row_sum_max']:.2f}"
        print(f"  {s['k']:>2} | {s['d']:>5} | {s['num_reps']:>5} | {s['rank']:>5} | {'Y' if s['psd'] else 'N':>3} | {s['distinct_gadgets']:>5} | {s['distinct_eigenvalues']:>5} | {rs:>12} | [{s['gadget_min']:.4f}, {s['gadget_max']:.4f}]")

    total_rank = sum(s["rank"] for s in summaries)
    total_distinct = sum(s["distinct_gadgets"] for s in summaries)
    all_psd = all(s["psd"] for s in summaries)
    all_const_rows = all(s["row_sum_const"] for s in summaries)
    print(f"\n  Total rank across all strata: {total_rank} / {total_reps}")
    print(f"  Total distinct gadget values: {total_distinct}")
    print(f"  All strata PSD: {'YES' if all_psd else 'NO'}")
    print(f"  All strata constant row sums: {'YES' if all_const_rows else 'NO'}")


if __name__ == "__main__":
    main()
