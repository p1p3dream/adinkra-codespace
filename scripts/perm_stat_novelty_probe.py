#!/usr/bin/env python3
"""Independent cross-check: build L-matrices as signed permutation ARRAYS for the
145 N=16 doubly-even code classes, compute per-color permutation statistics
(inversions / descents / major index) and a per-class signature, then test
whether that signature adds discriminating power BEYOND the invariants the
codebase already stores (k, automorphism_group_size, weight enumerator) and the
gadget data.

Fully independent of the Rust code; only the *construction rule* is mirrored
from src/chromotopology.rs (coset quotient I^N / C, even/odd weight bipartition).
"""
import json
from collections import Counter, defaultdict
from itertools import combinations

N = 16
JSON_PATH = "adinkra_codes_n16.json"


# ---------------------------------------------------------------------------
# 1. Code -> chromotopology color permutations (mirrors src/chromotopology.rs)
# ---------------------------------------------------------------------------
def codewords_from_generators(gens):
    """All 2^k codewords as the XOR-span of the generators (ints, N bits)."""
    words = [0]
    for g in gens:
        words = words + [w ^ g for w in words]
    return words


def build_color_perms(gens):
    """Return (d, color_perms) where color_perms[I][j] = fermion rank that
    boson rank j connects to via color I. Exactly the construction in
    chromotopology.rs::from_code."""
    total = 1 << N
    cws = codewords_from_generators(gens)
    k = len(cws).bit_length() - 1  # since len==2^k
    assert (1 << k) == len(cws)
    num_vertices = total >> k
    d = num_vertices >> 1  # even/odd split each get d

    # coset map: element -> coset index, canonical rep = first-seen (smallest scan order)
    coset_map = [-1] * total
    coset_reps = []
    for v in range(total):
        if coset_map[v] != -1:
            continue
        idx = len(coset_reps)
        coset_reps.append(v)
        for c in cws:
            coset_map[v ^ c] = idx
    assert len(coset_reps) == num_vertices

    # bipartition by weight parity of the representative
    boson_rank = [-1] * num_vertices
    fermion_rank = [-1] * num_vertices
    b = f = 0
    for idx, rep in enumerate(coset_reps):
        if bin(rep).count("1") % 2 == 0:
            boson_rank[idx] = b
            b += 1
        else:
            fermion_rank[idx] = f
            f += 1
    assert b == d and f == d, (b, f, d)

    # boson coset indices in rank order
    boson_coset_of_rank = [None] * d
    for idx in range(num_vertices):
        if boson_rank[idx] != -1:
            boson_coset_of_rank[boson_rank[idx]] = idx

    color_perms = []
    for color in range(N):
        basis = 1 << color
        perm = [0] * d
        for j in range(d):
            coset_idx = boson_coset_of_rank[j]
            rep = coset_reps[coset_idx]
            neighbor = rep ^ basis
            ncoset = coset_map[neighbor]
            fr = fermion_rank[ncoset]
            assert fr != -1, "neighbor not a fermion"
            perm[j] = fr
        color_perms.append(perm)
    return d, color_perms


# ---------------------------------------------------------------------------
# 2. Dashing + Garden algebra verification (signed permutations)
# ---------------------------------------------------------------------------
def garden_ok(d, color_perms, dashing):
    """Verify L_I R_J + L_J R_I = 2 d_IJ I with R = L^T (inverse of signed perm).
    Represent L_I as (perm, sign): row i nonzero at col perm[i] w/ value sign[i].
    Only need: L_I L_I^T = I (always true for a permutation w/ any signs) and
    L_I L_J^T = -(L_J L_I^T) for I != J."""
    def sp(I):
        return color_perms[I], [dashing[I][j] for j in range(d)]

    def compose_LT(I, J):
        # returns matrix M = L_I * (L_J)^T as (perm, sign) mapping.
        # L_J^T: row a nonzero at col permJ[a]? No: (L_J)^T has nonzero in
        # column permJ[b] row b -> transpose swaps. Build dense-free.
        pI, sI = sp(I)
        pJ, sJ = sp(J)
        # L_J^T maps: entry (permJ[b], b) = sJ[b]; as perm on columns:
        # (L_J^T)[r] nonzero at col c where permJ[c]=r, value sJ[c].
        invJ = [0] * d
        for b in range(d):
            invJ[pJ[b]] = b
        # M[i] = L_I row i picks col pI[i] with sI[i]; then L_J^T at row pI[i]:
        outp = [0] * d
        outs = [0] * d
        for i in range(d):
            col = pI[i]
            # L_J^T row=col nonzero at invJ[col] with value sJ[invJ[col]]
            c2 = invJ[col]
            outp[i] = c2
            outs[i] = sI[i] * sJ[c2]
        return outp, outs

    for I in range(N):
        p, s = compose_LT(I, I)
        if p != list(range(d)) or any(x != 1 for x in s):
            return False
    for I in range(N):
        for J in range(I + 1, N):
            p1, s1 = compose_LT(I, J)
            p2, s2 = compose_LT(J, I)
            if p1 != p2:
                return False
            if any(a != -b for a, b in zip(s1, s2)):
                return False
    return True


def vertex_dashing(d, color_perms):
    """A simple, code-independent valid dashing that satisfies the Garden
    algebra for these quotient chromotopologies: sign on edge (color I, boson j)
    = parity of (popcount of boson-rep restricted below color I). We instead use
    the standard 'binary vertex code' dashing derived from the boson index bits,
    which is known to work for these hypercube-quotient adinkras. If a closed
    form fails Garden, fall back to a search is out of scope; construction below
    is the Gray/binary dashing chi(I,j) = (-1)^(popcount(j & mask_I))."""
    dash = []
    for I in range(N):
        row = []
        mask = (1 << I) - 1  # bits below color I
        for j in range(d):
            row.append(-1 if bin(j & mask).count("1") % 2 else 1)
        dash.append(row)
    return dash


# ---------------------------------------------------------------------------
# 3. Permutation statistics (on the UNSIGNED color permutations)
# ---------------------------------------------------------------------------
def inversions(p):
    """O(d log d) inversion count via a Fenwick / BIT over the value range.
    p is a permutation of 0..n-1, so values are already the ranks."""
    n = len(p)
    tree = [0] * (n + 1)
    inv = 0
    seen = 0
    for x in p:
        # count of already-seen values <= x  (prefix sum up to x+1)
        i = x + 1
        cnt = 0
        while i > 0:
            cnt += tree[i]
            i -= i & (-i)
        # values already seen that are greater than x -> inversions with x
        inv += seen - cnt
        seen += 1
        i = x + 1
        while i <= n:
            tree[i] += 1
            i += i & (-i)
    return inv


def descents_and_maj(p):
    des = 0
    maj = 0
    for i in range(len(p) - 1):
        if p[i] > p[i + 1]:
            des += 1
            maj += i + 1  # 1-indexed descent position
    return des, maj


def perm_signature(d, color_perms):
    """Per-class signature: the multiset (sorted tuple over colors) of
    (inversions, descents, major_index) triples. Multiset because color
    labelling is not canonical, so we sort to make it label-invariant."""
    trips = []
    for I in range(N):
        p = color_perms[I]
        inv = inversions(p)
        des, maj = descents_and_maj(p)
        trips.append((inv, des, maj))
    return tuple(sorted(trips))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    data = json.load(open(JSON_PATH))
    codes = data["codes"]

    records = []
    garden_checked = 0
    garden_pass = 0
    for c in codes:
        gens = c["generators_raw"]
        d, cperms = build_color_perms(gens)
        # verify Garden algebra on a sample (first 12 classes + the two k=8)
        if c["index"] < 12 or c["k"] == 8:
            dash = vertex_dashing(d, cperms)
            ok = garden_ok(d, cperms, dash) if d <= 4096 else None
            garden_checked += 1 if ok is not None else 0
            garden_pass += 1 if ok else 0
        sig = perm_signature(d, cperms)
        records.append({
            "index": c["index"],
            "k": c["k"],
            "d": d,
            "aut": c["automorphism_group_size"],
            "we": tuple(c["weight_enumerator_full"]),
            "wd": tuple(map(tuple, c["weight_distribution"])),
            "sig": sig,
        })

    print(f"Garden algebra: {garden_pass}/{garden_checked} sampled classes PASS "
          f"(vertex/Gray dashing)")

    # ---- distinct signature count (compare vs sibling agent) ----
    sigs = [r["sig"] for r in records]
    distinct_sig = len(set(sigs))
    print(f"\n=== DISTINCT PERMUTATION SIGNATURES: {distinct_sig} / 145 ===")

    # ---- baseline invariant separating powers ----
    def npart(keyfn):
        return len(set(keyfn(r) for r in records))

    print("\nSeparating power (distinct values over 145 classes):")
    print(f"  k alone:                         {npart(lambda r: r['k'])}")
    print(f"  aut alone:                       {npart(lambda r: r['aut'])}")
    print(f"  weight_enumerator alone:         {npart(lambda r: r['we'])}")
    print(f"  (k, weight_enumerator):          {npart(lambda r: (r['k'], r['we']))}")
    print(f"  perm signature alone:            {npart(lambda r: r['sig'])}")
    print(f"  (k, we, perm sig):               {npart(lambda r: (r['k'], r['we'], r['sig']))}")

    # ---- (a) is perm sig a FUNCTION of (k, we)? i.e. redundant/derivable ----
    # It's a function of (k,we) iff no two classes share (k,we) but differ in sig,
    # AND it partitions no finer... function means: (k,we) -> sig is well defined.
    kwe_to_sigs = defaultdict(set)
    for r in records:
        kwe_to_sigs[(r["k"], r["we"])].add(r["sig"])
    sig_is_function_of_kwe = all(len(s) == 1 for s in kwe_to_sigs.values())
    print(f"\n(a) perm-sig is a FUNCTION of (k, weight_enum)? {sig_is_function_of_kwe}")
    if not sig_is_function_of_kwe:
        bad = [(kwe[0]) for kwe, s in kwe_to_sigs.items() if len(s) > 1]
        print(f"    -> NO: {sum(len(s)>1 for s in kwe_to_sigs.values())} (k,we) buckets "
              f"map to >1 signature (so sig carries info beyond (k,we))")

    # ---- (b) does perm sig make a GENUINELY NEW separation? ----
    # i.e. two classes identical in ALL known invariants (k, aut, we) but
    # separated by the perm signature.
    known_key = lambda r: (r["k"], r["aut"], r["we"])
    buckets = defaultdict(list)
    for r in records:
        buckets[known_key(r)].append(r)
    new_separations = []
    for key, rs in buckets.items():
        if len(rs) < 2:
            continue
        sigset = set(r["sig"] for r in rs)
        if len(sigset) > 1:
            new_separations.append((key, rs))
    print(f"\n(b) NEW separations (classes identical in k+aut+we, split by perm sig): "
          f"{len(new_separations)} bucket(s)")
    for key, rs in new_separations[:6]:
        idxs = [r["index"] for r in rs]
        nsig = len(set(r["sig"] for r in rs))
        print(f"    k={key[0]} we-collision bucket -> classes {idxs} "
              f"split into {nsig} signatures")

    # ---- (c) does perm sig FAIL to separate classes known invariants DO? ----
    # classes with equal perm sig but different (k, aut, we).
    sig_buckets = defaultdict(list)
    for r in records:
        sig_buckets[r["sig"]].append(r)
    weaker_failures = []
    for sig, rs in sig_buckets.items():
        if len(rs) < 2:
            continue
        knownset = set(known_key(r) for r in rs)
        if len(knownset) > 1:
            weaker_failures.append((sig, rs))
    print(f"\n(c) FAILURES (same perm sig but DIFFER in k/aut/we): "
          f"{len(weaker_failures)} bucket(s)")
    for sig, rs in weaker_failures[:8]:
        idxs = [r["index"] for r in rs]
        ks = sorted(set(r["k"] for r in rs))
        nwe = len(set(r["we"] for r in rs))
        print(f"    classes {idxs}: same perm-sig but k in {ks}, {nwe} distinct WE")

    # cross-k collisions specifically (dramatic weakness)
    crossk = [(sig, rs) for sig, rs in weaker_failures
              if len(set(r["k"] for r in rs)) > 1]
    print(f"\n    of those, {len(crossk)} collide ACROSS different k")

    # ---- summary combined power ----
    combined = npart(lambda r: (r["k"], r["we"], r["sig"]))
    kwe = npart(lambda r: (r["k"], r["we"]))
    print(f"\nCombined (k,we,sig) distinct = {combined} vs (k,we) distinct = {kwe} "
          f"-> perm sig adds {combined - kwe} extra separations on top of (k,we)")


if __name__ == "__main__":
    main()
