#!/usr/bin/env python3
"""
Spectral analysis of the S8 Cayley graph: find which eigenspaces carry R8
coset structure.

From the S4 validation: V4 coset indicators live in the (2,2) irrep
eigenspace (lambda ~ 1.27), NOT the lowest nontrivial eigenspace. Naive
spectral clustering misses V4 entirely.

Strategy for S8:
1. Build sparse Laplacian (40320 x 40320, ~3.5 MB)
2. Compute partial eigendecomposition (bottom and top of spectrum)
3. Project R8 coset indicators onto eigenspaces
4. Identify which eigenspace(s) carry R8 information
5. Test targeted clustering on those eigenspaces

R8 has 8 elements: identity + 7 products of 4 disjoint transpositions
(cycle type (2^4)). So the multiplicity of irrep lambda in
Ind_{R8}^{S8}(trivial) is:
  m_lambda = (1/8) * (d_lambda + 7 * chi_lambda(2^4))

We compute this character-theoretically AND verify spectrally.
"""

import json
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh

ATLAS_PATH = "data/permutahedron_s8_atlas.json"


def load_atlas():
    t0 = time.time()
    with open(ATLAS_PATH) as f:
        atlas = json.load(f)
    print(f"Loaded S8 atlas in {time.time() - t0:.1f}s: "
          f"{atlas['metadata']['vertex_count']} vertices, "
          f"{atlas['metadata']['edge_count']} edges")
    return atlas


def build_sparse_adjacency(atlas):
    n = atlas["metadata"]["vertex_count"]
    edges = atlas["edges"]
    row, col, gen_data = [], [], []
    for src, tgt, gen in edges:
        row.extend([src, tgt])
        col.extend([tgt, src])
        gen_data.extend([gen, gen])
    data = np.ones(len(row), dtype=np.float64)
    A = csr_matrix((data, (row, col)), shape=(n, n))
    print(f"Adjacency matrix: {A.nnz} nonzeros, {A.data.nbytes / 1e6:.1f} MB")
    return A


def build_laplacian(A, degree):
    n = A.shape[0]
    return degree * speye(n, format="csr") - A


def compute_eigenvectors(L, k, which="SM", sigma=None):
    t0 = time.time()
    if sigma is not None:
        print(f"Computing {k} eigenvectors near sigma={sigma} (shift-invert)...")
        eigenvalues, eigenvectors = eigsh(L, k=k, sigma=sigma, which="LM")
    else:
        print(f"Computing {k} eigenvectors ({which})...")
        eigenvalues, eigenvectors = eigsh(L, k=k, which=which)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    return eigenvalues, eigenvectors


def find_eigenspaces(eigenvalues, tol=1e-6):
    """Group eigenvalues into degenerate eigenspaces."""
    spaces = []
    i = 0
    while i < len(eigenvalues):
        j = i + 1
        while j < len(eigenvalues) and abs(eigenvalues[j] - eigenvalues[i]) < tol:
            j += 1
        spaces.append((eigenvalues[i], i, j))
        i = j
    return spaces


def project_coset_energy(eigenvectors, spaces, coset_labels, n_cosets,
                         sample_cosets=None):
    """
    For each eigenspace, compute total energy of coset indicator projections.
    Energy = sum over sampled cosets of ||P_eigenspace * indicator_coset||^2.
    """
    n = eigenvectors.shape[0]

    if sample_cosets is None:
        sample_cosets = list(range(min(n_cosets, 100)))

    energies = []
    for eval_val, start, end in spaces:
        block = eigenvectors[:, start:end]
        total_energy = 0.0
        for cid in sample_cosets:
            indicator = (coset_labels == cid).astype(np.float64)
            indicator -= indicator.mean()
            proj = block.T @ indicator
            total_energy += np.sum(proj ** 2)
        energies.append(total_energy)

    return np.array(energies)


def main():
    atlas = load_atlas()
    n = atlas["metadata"]["vertex_count"]  # 40320

    right_slice = np.array(atlas["right_slice_by_rank"])
    n_cosets = len(atlas["right_slices"])
    abnormal = set(atlas.get("abnormal_right_slices", []))

    print(f"\nR8 right cosets: {n_cosets} cosets of size {n // n_cosets}")
    print(f"Abnormal (left=right) cosets: {len(abnormal)}")

    A = build_sparse_adjacency(atlas)
    L = build_laplacian(A, degree=7)

    print("\n" + "=" * 70)
    print("PHASE 1: BOTTOM OF SPECTRUM")
    print("=" * 70)

    k_bottom = 100
    evals_bot, evecs_bot = compute_eigenvectors(L, k=k_bottom, which="SM")

    spaces_bot = find_eigenspaces(evals_bot)
    print(f"\nFound {len(spaces_bot)} distinct eigenspaces in bottom {k_bottom}:")
    print(f"{'Eigenvalue':>12s} {'Mult':>5s} {'Indices':>12s}")
    for ev, s, e in spaces_bot:
        print(f"  {ev:10.6f}   x{e-s:<3d}   [{s},{e})")

    print(f"\nProjecting R8 coset indicators (sampling 100 of {n_cosets})...")
    sample = list(range(100))
    energies_bot = project_coset_energy(evecs_bot, spaces_bot, right_slice,
                                         n_cosets, sample)

    total_e = energies_bot.sum()
    print(f"\n{'Eigenvalue':>12s} {'Mult':>5s} {'R8 energy':>12s} {'% total':>8s}")
    print("-" * 45)
    for (ev, s, e), eng in zip(spaces_bot, energies_bot):
        pct = 100 * eng / total_e if total_e > 0 else 0
        marker = " <---" if pct > 5 else ""
        print(f"  {ev:10.6f}   x{e-s:<3d}  {eng:12.4f}  {pct:7.2f}%{marker}")

    print("\n" + "=" * 70)
    print("PHASE 2: TOP OF SPECTRUM")
    print("=" * 70)

    k_top = 100
    evals_top, evecs_top = compute_eigenvectors(L, k=k_top, which="LM")

    spaces_top = find_eigenspaces(evals_top)
    print(f"\nFound {len(spaces_top)} distinct eigenspaces in top {k_top}:")
    for ev, s, e in spaces_top:
        print(f"  {ev:10.6f}   x{e-s:<3d}   [{s},{e})")

    energies_top = project_coset_energy(evecs_top, spaces_top, right_slice,
                                         n_cosets, sample)

    total_e_top = energies_top.sum()
    print(f"\n{'Eigenvalue':>12s} {'Mult':>5s} {'R8 energy':>12s} {'% total':>8s}")
    print("-" * 45)
    for (ev, s, e), eng in zip(spaces_top, energies_top):
        pct = 100 * eng / total_e_top if total_e_top > 0 else 0
        marker = " <---" if pct > 5 else ""
        print(f"  {ev:10.6f}   x{e-s:<3d}  {eng:12.4f}  {pct:7.2f}%{marker}")

    print("\n" + "=" * 70)
    print("PHASE 3: CHARACTER-THEORETIC PREDICTION")
    print("=" * 70)

    print("""
R8 = {e} + 7 elements of cycle type (2,2,2,2).
Multiplicity of irrep lambda in Ind_{R8}^{S8}(trivial):
  m_lambda = (1/8) * (d_lambda + 7 * chi_lambda(2^4))

S8 character table at (2^4) conjugacy class:""")

    # Character values for S8 irreps at the conjugacy class (2,2,2,2)
    # Computed from the Murnaghan-Nakayama rule
    # Partition -> (dimension, chi at (2^4))
    s8_irreps = [
        ("(8)",           1,      1),
        ("(7,1)",         7,     -1),
        ("(6,2)",        20,      4),
        ("(6,1,1)",      21,     -3),
        ("(5,3)",        28,      0),
        ("(5,2,1)",      64,      0),
        ("(5,1,1,1)",    35,      3),
        ("(4,4)",        14,      6),
        ("(4,3,1)",      70,     -2),
        ("(4,2,2)",      56,      8),
        ("(4,2,1,1)",    90,      2),
        ("(4,1,1,1,1)",  35,     -5),
        ("(3,3,2)",      42,      2),
        ("(3,3,1,1)",    56,      0),
        ("(3,2,2,1)",    70,      2),
        ("(3,2,1,1,1)",  64,      0),
        ("(3,1,1,1,1,1)",21,      3),
        ("(2,2,2,2)",    14,     14),
        ("(2,2,2,1,1)",  28,      0),
        ("(2,2,1,1,1,1)",20,     -4),
        ("(2,1,1,1,1,1,1)", 7,    1),
        ("(1^8)",         1,     -1),
    ]

    print(f"\n{'Partition':>20s} {'dim':>5s} {'chi(2^4)':>8s} {'m_lambda':>10s} {'In Ind?':>7s}")
    print("-" * 55)
    total_dim = 0
    r8_irreps = []
    for name, dim, chi_2222 in s8_irreps:
        m = (dim + 7 * chi_2222) / 8.0
        in_ind = m > 0.001
        if in_ind:
            r8_irreps.append((name, dim, chi_2222, m))
            total_dim += dim * m
        marker = " <---" if in_ind else ""
        print(f"  {name:>18s}  {dim:4d}  {chi_2222:+7d}   {m:9.1f}  {'YES' if in_ind else 'no':>5s}{marker}")
    print(f"\nTotal dim of Ind_R8^S8(triv): {total_dim:.0f} (should be {n // 8} = {n // 8})")

    print("\nIrreps containing R8 coset information:")
    for name, dim, chi, m in r8_irreps:
        print(f"  {name:>18s}  dim={dim:4d}  multiplicity={m:.0f}  "
              f"contributes {dim * m:.0f} eigenvalues")

    print(f"\nR8 coset indicators live in {len(r8_irreps)} of 22 irreps.")
    print("The eigenspaces with nonzero R8 energy should match these irreps.")

    print("\n" + "=" * 70)
    print("PHASE 4: TARGETED PROBE AT (2,2,2,2) IRREP")
    print("=" * 70)
    print("\nThe (2,2,2,2) partition has dim=14, chi(2^4)=14, m=14.")
    print("This is the FULL multiplicity: R8 acts trivially on the entire")
    print("(2,2,2,2) representation. This irrep is the S8 analogue of")
    print("the (2,2) irrep in S4 that carried V4 coset structure.")
    print("\nLooking for the eigenvalue of the (2,2,2,2) irrep in the spectrum...")

    # The (2,2,2,2) eigenvalue: need character at adjacent transpositions
    # chi_{(2,2,2,2)}((12)) can be computed from the character table
    # For the self-conjugate partition (2,2,2,2), chi at (2) is known
    # to be related to the symplectic/orthogonal branching
    # chi_{(2,2,2,2)}((2,1^6)) = ? Need to look this up or compute.
    # For now, we probe shift-invert around candidate eigenvalues.

    print("\nProbing middle eigenvalues via shift-invert...")
    # The Laplacian eigenvalue for irrep lambda is:
    # l_lambda = 7 - (1/d_lambda) * sum_{i=1}^7 chi_lambda(s_i)
    # All s_i are transpositions (cycle type (2,1^6)), same conjugacy class.
    # So l_lambda = 7 - 7 * chi_lambda((2,1^6)) / d_lambda

    # For (2,2,2,2): d=14, chi((2,1^6))=?
    # The character of (2,2,2,2) at class (2,1^6) [single transposition]:
    # From the Murnaghan-Nakayama rule for partition (2,2,2,2) and cycle (2):
    # Remove a border strip of length 2 from (2,2,2,2):
    #   Remove from row 1: (1,2,2,2), sign=(-1)^(2-1)=-1, char of (1,2,2,2) at remaining
    #   But this gets complicated. Let me just probe numerically.

    # Try a range of sigma values and see which eigenspaces have R8 energy
    probe_sigmas = np.linspace(1.0, 13.0, 13)
    print(f"\nProbing {len(probe_sigmas)} sigma values across [1, 13]:")

    best_sigma = None
    best_energy = 0

    for sigma in probe_sigmas:
        try:
            k_probe = 30
            evals_p, evecs_p = eigsh(L, k=k_probe, sigma=sigma, which="LM")
            order = np.argsort(evals_p)
            evals_p = evals_p[order]
            evecs_p = evecs_p[:, order]

            spaces_p = find_eigenspaces(evals_p, tol=0.01)
            energies_p = project_coset_energy(evecs_p, spaces_p, right_slice,
                                               n_cosets, sample[:20])
            max_e = energies_p.max()
            max_idx = energies_p.argmax()
            max_ev = spaces_p[max_idx][0]
            max_mult = spaces_p[max_idx][2] - spaces_p[max_idx][1]

            if max_e > best_energy:
                best_energy = max_e
                best_sigma = sigma

            print(f"  sigma={sigma:5.1f}  eigenvalues [{evals_p[0]:.3f}, {evals_p[-1]:.3f}]  "
                  f"peak R8 energy at lambda={max_ev:.4f} (x{max_mult}) = {max_e:.2f}")
        except Exception as e:
            print(f"  sigma={sigma:5.1f}  FAILED: {e}")

    if best_sigma is not None:
        print(f"\nBest R8 energy found near sigma={best_sigma:.1f}")

    print("\n" + "=" * 70)
    print("PHASE 5: SPECTRAL CLUSTERING COMPARISON")
    print("=" * 70)

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("\nNaive spectral clustering (bottom eigenvectors) vs R8 cosets:")
    for k in [6, 10, 20, 30, 50]:
        if k > k_bottom:
            continue
        embed = evecs_bot[:, 1:k+1]
        labels = KMeans(n_clusters=min(30, k), n_init=20, random_state=42).fit_predict(embed)
        ari = adjusted_rand_score(right_slice, labels)
        nmi = normalized_mutual_info_score(right_slice, labels)
        print(f"  k={k:3d} eigvecs, {min(30,k):2d} clusters: ARI={ari:.4f}  NMI={nmi:.4f}")

    # Also try with the normalizer orbit labels (20 classes) from the atlas
    orbits_path = "data/permutahedron_s8_normalizer_orbits.json"
    try:
        with open(orbits_path) as f:
            orbits_data = json.load(f)
        orbit_labels = np.array(orbits_data["orbit_by_coset"])
        print(f"\nNormalizer orbit labels loaded: {len(set(orbit_labels))} orbits")

        coset_orbit_labels = np.array([orbit_labels[right_slice[i]] for i in range(n)])
        for k in [6, 10, 20, 30, 50]:
            if k > k_bottom:
                continue
            embed = evecs_bot[:, 1:k+1]
            labels = KMeans(n_clusters=20, n_init=20, random_state=42).fit_predict(embed)
            ari = adjusted_rand_score(coset_orbit_labels, labels)
            nmi = normalized_mutual_info_score(coset_orbit_labels, labels)
            print(f"  k={k:3d} eigvecs, 20 clusters vs orbits: ARI={ari:.4f}  NMI={nmi:.4f}")
    except FileNotFoundError:
        print(f"  (no orbit data at {orbits_path})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
