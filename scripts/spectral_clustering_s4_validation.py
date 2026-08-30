#!/usr/bin/env python3
"""
Spectral clustering on the S4 Cayley graph as validation for the hex decomposition pipeline.

Goal: verify that spectral clustering with k=6 recovers the V4 coset structure
(the 6 quartets P1-P6) from the graph Laplacian alone, without being told about V4.

The S4 Cayley graph:
  - 24 vertices (permutations of 1234)
  - 36 edges (adjacent transpositions: swap positions 1-2, 2-3, 3-4)
  - 3-regular, diameter 6
  - V4 right cosets partition into 6 quartets of 4
"""

import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from itertools import permutations

ATLAS_PATH = "data/permutahedron_s4_atlas.json"


def load_atlas():
    with open(ATLAS_PATH) as f:
        return json.load(f)


def build_adjacency(atlas):
    n = atlas["metadata"]["vertex_count"]
    row, col = [], []
    for src, tgt, _gen in atlas["edges"]:
        row.extend([src, tgt])
        col.extend([tgt, src])
    data = np.ones(len(row), dtype=np.float64)
    return csr_matrix((data, (row, col)), shape=(n, n))


def build_laplacian(A, degree):
    n = A.shape[0]
    from scipy.sparse import eye
    return degree * eye(n) - A


def spectral_embed(L, k):
    if k >= L.shape[0]:
        from scipy.linalg import eigh
        L_dense = L.toarray() if hasattr(L, "toarray") else L
        eigenvalues, eigenvectors = eigh(L_dense)
    else:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def cluster_from_eigenvectors(V, k):
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(V)
    return labels


def compare_partitions(discovered, ground_truth_slices, permutation_names):
    gt_labels = np.zeros(len(permutation_names), dtype=int)
    for i, sl in enumerate(ground_truth_slices):
        for rank in sl:
            gt_labels[rank] = i

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(gt_labels, discovered)
    nmi = normalized_mutual_info_score(gt_labels, discovered)
    return ari, nmi, gt_labels


def print_clusters(labels, permutation_names, title):
    clusters = {}
    for rank, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(permutation_names[rank])
    print(f"\n{title}")
    print("-" * 60)
    for cid in sorted(clusters):
        members = sorted(clusters[cid])
        print(f"  Cluster {cid}: {members}")


def main():
    atlas = load_atlas()
    perms = atlas["permutations"]
    n = len(perms)
    print(f"Loaded S4 atlas: {n} vertices, {len(atlas['edges'])} edges")

    A = build_adjacency(atlas)
    L = build_laplacian(A, degree=3)

    print("\n=== EIGENVALUE ANALYSIS ===")
    all_evals, all_evecs = spectral_embed(L, k=n)
    print(f"\nAll {n} eigenvalues of the graph Laplacian:")
    for i, ev in enumerate(all_evals):
        print(f"  lambda_{i:2d} = {ev:8.4f}")

    unique_evals = []
    prev = None
    for ev in all_evals:
        if prev is None or abs(ev - prev) > 1e-6:
            unique_evals.append((ev, 1))
        else:
            unique_evals[-1] = (unique_evals[-1][0], unique_evals[-1][1] + 1)
        prev = ev

    print(f"\nDistinct eigenvalues (value, multiplicity):")
    for ev, mult in unique_evals:
        print(f"  {ev:8.4f}  x{mult}")

    print("\n=== SPECTRAL CLUSTERING (k=6) ===")
    embed_6 = all_evecs[:, :6]
    labels_6 = cluster_from_eigenvectors(embed_6, k=6)

    gt_slices = atlas["right_slices"]
    ari, nmi, gt_labels = compare_partitions(labels_6, gt_slices, perms)

    print(f"\nAdjusted Rand Index:           {ari:.4f}  (1.0 = perfect match)")
    print(f"Normalized Mutual Information: {nmi:.4f}  (1.0 = perfect match)")

    print_clusters(labels_6, perms, "Discovered clusters (spectral, k=6)")
    print_clusters(gt_labels, perms, "Ground truth V4 cosets (P1-P6)")

    reps = atlas.get("representations", [])
    if reps:
        print("\nGround truth labeled:")
        for rep in reps:
            print(f"  {rep['id']:5s} ({rep['label']:30s}): {rep['member_addresses']}")

    exact = (ari == 1.0)
    print(f"\n{'PASS' if exact else 'FAIL'}: Spectral clustering "
          f"{'exactly recovers' if exact else 'does NOT recover'} V4 cosets")

    print("\n=== EIGENSPACE ANALYSIS ===")
    print("\nProjection of V4 coset indicator functions onto eigenvectors:")
    print(f"{'Coset':>8s}", end="")
    for i in range(min(8, n)):
        print(f"  ev_{i:d}", end="")
    print()
    for sid, sl in enumerate(gt_slices):
        indicator = np.zeros(n)
        for rank in sl:
            indicator[rank] = 1.0
        projections = all_evecs.T @ indicator
        print(f"  P{sid+1:d}   ", end="")
        for i in range(min(8, n)):
            print(f"  {projections[i]:+6.3f}", end="")
        print()

    print("\n=== SENSITIVITY: VARYING k ===")
    for k in [2, 3, 4, 5, 6, 7, 8, 12]:
        embed_k = all_evecs[:, :k]
        labels_k = cluster_from_eigenvectors(embed_k, k=6)
        ari_k, nmi_k, _ = compare_partitions(labels_k, gt_slices, perms)
        print(f"  k={k:2d} eigenvectors -> ARI={ari_k:.4f}, NMI={nmi_k:.4f}")

    print("\n=== SKLEARN SPECTRAL CLUSTERING (SANITY CHECK) ===")
    sc = SpectralClustering(n_clusters=6, affinity="precomputed",
                            assign_labels="kmeans", random_state=42, n_init=50)
    labels_sk = sc.fit_predict(A.toarray())
    ari_sk, nmi_sk, _ = compare_partitions(labels_sk, gt_slices, perms)
    print(f"  sklearn SpectralClustering: ARI={ari_sk:.4f}, NMI={nmi_sk:.4f}")

    print_clusters(labels_sk, perms, "sklearn clusters")

    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
