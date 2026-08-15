#!/usr/bin/env python3
"""
Deep spectral analysis of the S4 Cayley graph.

Key finding from the initial validation: naive spectral clustering (lowest
eigenvectors) does NOT recover V4 cosets. It recovers the Young subgroup
S_{1,2} x S_{3,4} cosets instead, because those are more geometrically
prominent in the Cayley graph.

This script investigates:
1. Which subgroup does the spectral partition correspond to?
2. Which eigenspace carries the V4 coset structure?
3. Can we recover V4 using targeted eigenspaces?
4. What does this mean for the S8 pipeline?
"""

import json
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import combinations

ATLAS_PATH = "data/permutahedron_s4_atlas.json"


def load_atlas():
    with open(ATLAS_PATH) as f:
        return json.load(f)


def perm_from_str(s):
    return [int(c) for c in s]


def compose(a, b):
    return [a[b[i] - 1] for i in range(len(a))]


def inverse(a):
    inv = [0] * len(a)
    for i, v in enumerate(a):
        inv[v - 1] = i + 1
    return inv


def perm_to_str(p):
    return "".join(str(x) for x in p)


def find_subgroup(cluster_members, all_perms):
    """Given a cluster containing the identity, find the subgroup it represents."""
    identity = [1, 2, 3, 4]
    id_str = "1234"

    id_cluster = None
    for cid, members in cluster_members.items():
        if id_str in members:
            id_cluster = cid
            break

    if id_cluster is None:
        return None, None

    subgroup_strs = cluster_members[id_cluster]
    subgroup_perms = [perm_from_str(s) for s in subgroup_strs]

    print(f"\nCluster containing identity: {sorted(subgroup_strs)}")

    closed = True
    for a in subgroup_perms:
        for b in subgroup_perms:
            prod = compose(a, b)
            if perm_to_str(prod) not in subgroup_strs:
                closed = False
                break

    print(f"Closed under composition: {closed}")

    if closed:
        print("Elements in cycle notation:")
        for p in subgroup_perms:
            cycles = []
            visited = set()
            for i in range(1, 5):
                if i not in visited:
                    cycle = []
                    j = i
                    while j not in visited:
                        visited.add(j)
                        cycle.append(j)
                        j = p[j - 1]
                    if len(cycle) > 1:
                        cycles.append(tuple(cycle))
            if not cycles:
                print(f"  {perm_to_str(p)} = e")
            else:
                print(f"  {perm_to_str(p)} = {''.join(str(c) for c in cycles)}")

    return id_cluster, subgroup_strs


def build_all_order4_subgroups():
    """Enumerate all subgroups of S4 with order 4."""
    from itertools import permutations as iter_perms

    all_s4 = [list(p) for p in iter_perms([1, 2, 3, 4])]
    identity = [1, 2, 3, 4]

    subgroups = []
    for combo in combinations(all_s4, 3):
        candidate = [identity] + list(combo)
        candidate_strs = {perm_to_str(p) for p in candidate}

        closed = True
        for a in candidate:
            for b in candidate:
                if perm_to_str(compose(a, b)) not in candidate_strs:
                    closed = False
                    break
            if not closed:
                break

        if closed:
            key = tuple(sorted(candidate_strs))
            subgroups.append(key)

    unique = list(set(subgroups))
    return unique


def coset_partition(subgroup_strs, all_perm_strs):
    """Compute right cosets of the subgroup."""
    subgroup = [perm_from_str(s) for s in subgroup_strs]
    assigned = set()
    cosets = []

    for sigma_str in all_perm_strs:
        if sigma_str in assigned:
            continue
        sigma = perm_from_str(sigma_str)
        coset = set()
        for h in subgroup:
            product = compose(h, sigma)
            coset.add(perm_to_str(product))
        cosets.append(sorted(coset))
        assigned.update(coset)

    return cosets


def main():
    atlas = load_atlas()
    perms = atlas["permutations"]
    n = len(perms)
    perm_to_rank = {p: i for i, p in enumerate(perms)}

    A = np.zeros((n, n))
    edge_colors = np.zeros((n, n), dtype=int)
    for src, tgt, gen in atlas["edges"]:
        A[src, tgt] = 1
        A[tgt, src] = 1
        edge_colors[src, tgt] = gen
        edge_colors[tgt, src] = gen

    L = 3 * np.eye(n) - A
    eigenvalues, eigenvectors = eigh(L)

    gt_slices = atlas["right_slices"]
    gt_labels = np.zeros(n, dtype=int)
    for i, sl in enumerate(gt_slices):
        for rank in sl:
            gt_labels[rank] = i

    print("=" * 70)
    print("DEEP SPECTRAL ANALYSIS OF S4 CAYLEY GRAPH")
    print("=" * 70)

    print("\n--- 1. IDENTIFY THE SPECTRAL PARTITION'S SUBGROUP ---")

    labels_spec = KMeans(n_clusters=6, n_init=50, random_state=42).fit_predict(
        eigenvectors[:, :6]
    )
    spec_clusters = {}
    for rank, label in enumerate(labels_spec):
        spec_clusters.setdefault(int(label), []).append(perms[rank])

    find_subgroup(spec_clusters, perms)

    print("\n--- 2. ALL ORDER-4 SUBGROUPS OF S4 ---")
    all_subgroups = build_all_order4_subgroups()
    print(f"Found {len(all_subgroups)} subgroups of order 4:")
    for sg in sorted(all_subgroups):
        elements = sorted(sg)
        cosets = coset_partition(elements, perms)
        coset_labels = np.zeros(n, dtype=int)
        for i, coset in enumerate(cosets):
            for p in coset:
                coset_labels[perm_to_rank[p]] = i

        ari_vs_v4 = adjusted_rand_score(gt_labels, coset_labels)
        ari_vs_spec = adjusted_rand_score(labels_spec, coset_labels)

        cycles_list = []
        for elem in elements:
            p = perm_from_str(elem)
            cycles = []
            visited = set()
            for i in range(1, 5):
                if i not in visited:
                    cycle = []
                    j = i
                    while j not in visited:
                        visited.add(j)
                        cycle.append(j)
                        j = p[j - 1]
                    if len(cycle) > 1:
                        cycles.append(tuple(cycle))
            if not cycles:
                cycles_list.append("e")
            else:
                cycles_list.append("".join(str(c) for c in cycles))

        print(f"  {elements}")
        print(f"    Cycles: {cycles_list}")
        print(f"    ARI vs V4 cosets: {ari_vs_v4:.4f}  |  ARI vs spectral: {ari_vs_spec:.4f}")

    print("\n--- 3. EIGENSPACE PROJECTION ANALYSIS ---")
    print("\nWhich eigenspace carries each subgroup's coset indicators?")

    v4_elements = ["1234", "2143", "3412", "4321"]
    v4_cosets = coset_partition(v4_elements, perms)
    v4_coset_labels = np.zeros(n, dtype=int)
    for i, coset in enumerate(v4_cosets):
        for p in coset:
            v4_coset_labels[perm_to_rank[p]] = i

    unique_evals = []
    eval_start = []
    prev = None
    for i, ev in enumerate(eigenvalues):
        if prev is None or abs(ev - prev) > 1e-6:
            unique_evals.append(ev)
            eval_start.append(i)
        prev = ev
    eval_start.append(n)

    print(f"\n{'Eigenvalue':>12s} {'Mult':>4s} {'V4 energy':>12s} {'Spectral energy':>16s}")
    print("-" * 50)

    for idx, (ev, start) in enumerate(zip(unique_evals, eval_start)):
        end = eval_start[idx + 1] if idx + 1 < len(eval_start) else n
        mult = end - start
        block = eigenvectors[:, start:end]

        v4_energy = 0
        for i in range(6):
            indicator = (v4_coset_labels == i).astype(float)
            indicator -= indicator.mean()
            proj = block.T @ indicator
            v4_energy += np.sum(proj ** 2)

        spec_energy = 0
        for i in range(6):
            indicator = (labels_spec == i).astype(float)
            indicator -= indicator.mean()
            proj = block.T @ indicator
            spec_energy += np.sum(proj ** 2)

        print(f"  {ev:10.4f}   x{mult}  {v4_energy:12.4f}    {spec_energy:16.4f}")

    print("\n--- 4. TARGETED EIGENSPACE CLUSTERING ---")
    print("\nTest: cluster using ONLY the eigenspace that carries V4 energy")

    for desc, indices in [
        ("ev 4-5 (lambda~1.27)", [4, 5]),
        ("ev 1-5", list(range(1, 6))),
        ("ev 4-8", list(range(4, 9))),
        ("ev 1-3 (lowest nontrivial)", [1, 2, 3]),
        ("ev 1-8", list(range(1, 9))),
        ("ev 4-5 + 9-11", [4, 5, 9, 10, 11]),
    ]:
        embed = eigenvectors[:, indices]
        labels_t = KMeans(n_clusters=6, n_init=50, random_state=42).fit_predict(embed)
        ari = adjusted_rand_score(gt_labels, labels_t)
        nmi = normalized_mutual_info_score(gt_labels, labels_t)
        print(f"  {desc:35s}  ARI={ari:.4f}  NMI={nmi:.4f}")

    print("\n--- 5. COLOR-AWARE LAPLACIAN ---")
    print("\nThe Cayley graph has 3 generator colors (adjacent transpositions).")
    print("Build per-color adjacency matrices and a color-weighted Laplacian.\n")

    for gen_id in [1, 2, 3]:
        A_color = np.zeros((n, n))
        for src, tgt, gen in atlas["edges"]:
            if gen == gen_id:
                A_color[src, tgt] = 1
                A_color[tgt, src] = 1
        L_color = np.eye(n) - A_color
        evals_c, evecs_c = eigh(L_color)

        labels_c = KMeans(n_clusters=6, n_init=50, random_state=42).fit_predict(
            evecs_c[:, :6]
        )
        ari_v4 = adjusted_rand_score(gt_labels, labels_c)
        ari_spec = adjusted_rand_score(labels_spec, labels_c)
        print(f"  Generator {gen_id} (swap pos {gen_id},{gen_id+1}):  "
              f"ARI vs V4={ari_v4:.4f}  ARI vs Young={ari_spec:.4f}")

    print("\n  Combined: sum of per-color Laplacians (= full Laplacian / normalization)")
    A_sum = np.zeros((n, n))
    for gen_id in [1, 2, 3]:
        for src, tgt, gen in atlas["edges"]:
            if gen == gen_id:
                A_sum[src, tgt] += 1
                A_sum[tgt, src] += 1

    for w1, w2, w3 in [(1, 0, 1), (0, 1, 0), (1, 1, 0), (0, 0, 1),
                        (1, 2, 1), (2, 1, 2), (0, 1, 1), (1, 0, 0)]:
        A_w = np.zeros((n, n))
        for src, tgt, gen in atlas["edges"]:
            w = [0, w1, w2, w3][gen]
            A_w[src, tgt] = w
            A_w[tgt, src] = w
        deg = A_w.sum(axis=1)
        if deg.max() == 0:
            continue
        L_w = np.diag(deg) - A_w
        evals_w, evecs_w = eigh(L_w)
        labels_w = KMeans(n_clusters=6, n_init=50, random_state=42).fit_predict(
            evecs_w[:, :6]
        )
        ari_v4 = adjusted_rand_score(gt_labels, labels_w)
        print(f"  Weights ({w1},{w2},{w3}): ARI vs V4={ari_v4:.4f}")

    print("\n--- 6. REPRESENTATION-THEORETIC DECOMPOSITION ---")
    print("\nS4 irreps (by partition of 4):")
    print("  (4)     dim=1   trivial")
    print("  (3,1)   dim=3   standard")
    print("  (2,2)   dim=2   two-dimensional")
    print("  (2,1,1) dim=3   standard tensor sign")
    print("  (1,1,1,1) dim=1 sign")
    print(f"  Sum of squares: 1+9+4+9+1 = 24 = 4!")

    print("\nEigenvalue multiplicities: 1, 3, 2, 3, 3, 3, 3, 2, 3, 1")
    print("These correspond to irreps appearing in the regular representation.")
    print("\nV4 coset indicators project onto eigenspace with lambda~1.27 (mult=2).")
    print("This is the (2,2) irrep eigenspace, the two-dimensional irrep of S4.")
    print("The (2,2) irrep is precisely the one that sees V4 structure,")
    print("because V4 is the kernel of the S4 -> S3 quotient and (2,2)")
    print("is the irrep that factors through this quotient.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key finding: Naive spectral clustering on the S4 Cayley graph does NOT
recover V4 cosets. It recovers the Young subgroup cosets instead, because
those align with the lowest nontrivial eigenspace (the standard irrep).

V4 coset structure lives in the (2,2) irrep eigenspace (lambda ~ 1.27),
which is NOT the lowest nontrivial eigenspace. To recover V4 spectrally,
you must know to look at the right eigenspace.

IMPLICATION FOR S8: Naive spectral clustering on the S8 Cayley graph
will likely NOT recover R8 cosets either. The R8 structure is encoded
in specific irreducible representations of S8 (related to the [8,4,4]
Hamming code), not in the lowest eigenspaces. ML approaches that work
must either:
  (a) Be told which eigenspace to look at (supervised)
  (b) Use additional features beyond graph structure (Garden signs,
      G-matrix data, chromotopology) to break the spectral degeneracy
  (c) Use equivariant architectures that respect the code structure
  (d) Search over all eigenspace combinations (expensive but feasible)

The color-aware Laplacian analysis tests whether weighting generators
differently can select for V4 structure.
""")


if __name__ == "__main__":
    main()
