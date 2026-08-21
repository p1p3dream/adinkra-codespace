#!/usr/bin/env python3
"""group_n.py: the full signed-permutation normalizer group of the CLS A matrices.

Artifact
--------
results/four_color_cls_gmatrix.json in the adinkra-codespace repo stores two
12x12 integer matrices A_L and A_R.  Each is block diagonal with three 4x4
blocks B_0, B_1, B_2 (block b occupies rows/cols 4b..4b+3).  The blocks are
pairwise DISTINCT as matrices but pairwise conjugate by signed permutations.

Conventions
-----------
A signed permutation is a pair (perm, signs): perm a permutation of range(n)
as a tuple, signs a tuple of +1/-1 of length n.  It represents the monomial
matrix D with

    D[i][perm[i]] = signs[i]      (zeros elsewhere).

Conjugation then acts entrywise as

    (D G D^-1)[i][j] = signs[i] * signs[j] * G[perm[i]][perm[j]]

see conj_matrix (n x n) and conj_block4 (4x4 convenience).  A signed
permutation NORMALIZES A when D A D^-1 == A entrywise.  The covariant action
on a vector is D v, i.e. act_on_vector(v, perm, signs)[i] = signs[i]*v[perm[i]].

Group structure (all re-verified by the __main__ self-check)
------------------------------------------------------------
* Block normalizer: {(p,s) : s_i s_j B[p_i][p_j] == B[i][j]} has exactly 24
  elements; the permutation parts are exactly the 12 even permutations (A4),
  each with two sign patterns that are negatives of each other.
* Full normalizer of the m-block matrix (m = 1, 2, 3): the wreath-like
  product of a block shuffle tau in S_m with, for each target block i, a
  signed 4x4 conjugator (p_i, s_i) carrying block tau^-1(i) onto block i:
      perm[4i + r] = 4*tau^-1(i) + p_i[r],   signs[4i + r] = s_i[r]
  Order 24^m * m!  (m=1: 24, m=2: 1152, m=3: 82944).
* Block-0 stabilizer (tau(0) = 0): order 24^m * (m-1)!  (m=3: 27648).
* The global -I (identity perm, all signs -1) lies in the group and acts
  trivially, and every element's sign-negated twin is also in the group, so
  the effective action on matrices has order |N| / 2 (41472 at m=3).

Completeness proof carried by __main__
--------------------------------------
all_signed_normalizers(A) re-derives the whole normalizer by exhaustive
backtracking over signed permutations of range(4m), with no wreath
assumption, and __main__ confirms set equality with the constructed census
at m = 1, 2, 3.  So the census is the COMPLETE signed-permutation
normalizer group, not just a subgroup.  (Structural reason this can work:
every block entry is nonzero and off-diagonal blocks vanish, so the support
pattern of A is exactly the block partition, and any conjugating
permutation must carry blocks onto whole blocks.)

Caching: group_elements(m, side) writes /tmp/lever_a/group_m{m}_{side}.json
(plain JSON list of [perm, signs] pairs) on first build and reloads it when
the count and shape validate.  Tomorrow's orbit step can read
/tmp/lever_a/group_m3_L.json directly.

Stdlib only: itertools, json, math, os, sys, time.  No numpy needed.
"""

from itertools import permutations, product
import json
import math
import os
import sys
import time

DEFAULT_REPO_ROOT = "/Users/brandon/code/adinkra-codespace"
ARTIFACT_RELPATH = os.path.join("results", "four_color_cls_gmatrix.json")
CACHE_DIR = "/tmp/lever_a"

_SIGNS4 = tuple(product((1, -1), repeat=4))


# ------------------------------------------------------------------ loading

def load_A(side="L", repo_root=DEFAULT_REPO_ROOT):
    """Load A_L or A_R (12x12 ints) from the repo artifact as tuple of tuples."""
    if side not in ("L", "R"):
        raise ValueError("side must be 'L' or 'R'")
    path = os.path.join(repo_root, ARTIFACT_RELPATH)
    with open(path) as f:
        data = json.load(f)
    key = "A_" + side
    if key not in data:
        raise KeyError("artifact %s lacks field %s; fields: %s" % (path, key, sorted(data)))
    A = tuple(tuple(int(x) for x in row) for row in data[key])
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("%s is not square" % key)
    return A


def blocks(A, m=None):
    """The m 4x4 diagonal blocks of A (block b = rows/cols 4b..4b+3)."""
    n = len(A)
    if n % 4:
        raise ValueError("matrix size %d is not a multiple of 4" % n)
    if m is None:
        m = n // 4
    if not (1 <= m <= n // 4):
        raise ValueError("bad block count m=%d for %dx%d matrix" % (m, n, n))
    return [
        tuple(tuple(A[4 * b + r][4 * b + c] for c in range(4)) for r in range(4))
        for b in range(m)
    ]


def is_block_diagonal(A, m=None):
    """True iff every off-diagonal 4x4 block of A is identically zero."""
    n = len(A)
    m = n // 4 if m is None else m
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            for r in range(4):
                for c in range(4):
                    if A[4 * a + r][4 * b + c]:
                        return False
    return True


def top_left(A, k):
    """Top-left k x k submatrix of A."""
    return tuple(tuple(A[i][j] for j in range(k)) for i in range(k))


# -------------------------------------------------------- conjugation core

def conj_block4(V, perm4, signs4):
    """Conjugate one 4x4 block: entry (i,j) = signs4[i]*signs4[j]*V[perm4[i]][perm4[j]]."""
    return tuple(
        tuple(signs4[i] * signs4[j] * V[perm4[i]][perm4[j]] for j in range(4))
        for i in range(4)
    )


def conj_matrix(G, perm, signs):
    """D G D^-1 for n x n G: entry (i,j) = signs[i]*signs[j]*G[perm[i]][perm[j]].

    G may be tuple-of-tuples or list-of-lists; the result is a list of lists.
    """
    n = len(G)
    return [
        [signs[i] * signs[j] * G[perm[i]][perm[j]] for j in range(n)]
        for i in range(n)
    ]


def normalizes(A, perm, signs):
    """True iff D A D^-1 == A entrywise (full n x n check, early exit)."""
    n = len(A)
    for i in range(n):
        ai = A[i]
        pi = perm[i]
        si = signs[i]
        row = A[pi]
        for j in range(n):
            if si * signs[j] * row[perm[j]] != ai[j]:
                return False
    return True


def element_matrix(perm, signs):
    """Dense n x n monomial matrix D with D[i][perm[i]] = signs[i]."""
    n = len(perm)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        D[i][perm[i]] = signs[i]
    return D


def compose(ea, eb):
    """Group product D_ea @ D_eb returned as (perm, signs).

    (D_a D_b)[i][pb[pa[i]]] = sa[i] * sb[pa[i]].
    """
    pa, sa = ea
    pb, sb = eb
    return (
        tuple(pb[pa[i]] for i in range(len(pa))),
        tuple(sa[i] * sb[pa[i]] for i in range(len(pa))),
    )


def act_on_vector(v, perm, signs):
    """Covariant action D v on a value vector: out[i] = signs[i] * v[perm[i]]."""
    return tuple(signs[i] * v[perm[i]] for i in range(len(v)))


# ------------------------------------------------- 4x4 conjugators / stabilizers

def _signed_perm_conjugators(Bs, Bd):
    """All (perm4, signs4) with conj_block4(Bs, p, s) == Bd, sorted.

    Brute force over all 24 * 16 = 384 signed permutations of range(4).
    """
    out = []
    for p in permutations(range(4)):
        for s in _SIGNS4:
            if conj_block4(Bs, p, s) == Bd:
                out.append((p, s))
    return sorted(out)


def block_normalizers(B):
    """All signed permutations stabilizing B by conjugation (24 for CLS blocks)."""
    return _signed_perm_conjugators(B, B)


def conjugator_sets(Bs, Bd):
    """All signed permutations carrying Bs onto Bd (24 for any CLS block pair)."""
    return _signed_perm_conjugators(Bs, Bd)


# ------------------------------------------------------------ census build

def expected_order(m):
    """24^m * m!"""
    return (24 ** m) * math.factorial(m)


def _build_group(A_m, m, fix_block0=False):
    """Wreath census of the normalizer of the m-block 4m x 4m matrix A_m.

    Element = (tau in S_m shuffling block positions) together with, for each
    target position i, a signed 4x4 conjugator (p_i, s_i) with
    T_i B_{tau^-1(i)} T_i^-1 = B_i, embedded as
        perm[4i + r] = 4 * tau^-1(i) + p_i[r]
        signs[4i + r] = s_i[r]
    fix_block0=True keeps only tau with tau(0) = 0 (block-0 stabilizer).
    """
    n = 4 * m
    if len(A_m) != n:
        raise ValueError("A_m must be %dx%d" % (n, n))
    Bs = blocks(A_m, m)
    table = {
        (s, d): _signed_perm_conjugators(Bs[s], Bs[d])
        for s in range(m)
        for d in range(m)
    }
    elems = []
    for tau in permutations(range(m)):
        if fix_block0 and tau[0] != 0:
            continue
        inv = [0] * m  # inv[i] = tau^-1(i): the source block feeding target i
        for j in range(m):
            inv[tau[j]] = j
        choices = [table[(inv[i], i)] for i in range(m)]
        for combo in product(*choices):
            perm = [0] * n
            signs = [0] * n
            for i in range(m):
                p, s = combo[i]
                base = 4 * inv[i]
                for r in range(4):
                    perm[4 * i + r] = base + p[r]
                    signs[4 * i + r] = s[r]
            elems.append((tuple(perm), tuple(signs)))
    return elems


def _cache_path(m, side):
    return os.path.join(CACHE_DIR, "group_m%d_%s.json" % (m, side))


def _load_cache(m, side):
    """Load cached census if it exists and validates; else None."""
    path = _cache_path(m, side)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        n = 4 * m
        if not isinstance(raw, list) or len(raw) != expected_order(m):
            return None
        elems = []
        for pair in raw:
            p = tuple(int(x) for x in pair[0])
            s = tuple(int(x) for x in pair[1])
            if len(p) != n or len(s) != n:
                return None
            if sorted(p) != list(range(n)):
                return None
            if any(x not in (1, -1) for x in s):
                return None
            elems.append((p, s))
        return elems
    except Exception:
        return None


def _save_cache(m, side, elems):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(m, side)
    with open(path, "w") as f:
        json.dump([[list(p), list(s)] for (p, s) in elems], f)
    return path


def group_elements(m, side="L", A=None, use_cache=True, force_rebuild=False):
    """The m-block normalizer census as a list of (perm, signs), each length 4m.

    m=1 -> 24, m=2 -> 1152, m=3 -> 82944 elements, built from the wreath
    product over the blocks of A_side from the repo artifact (or over the
    blocks of a caller-supplied 12x12 A).  Cached under CACHE_DIR as
    group_m{m}_{side}.json and reloaded when the file validates.
    """
    if m not in (1, 2, 3):
        raise ValueError("m must be 1, 2 or 3")
    if A is None:
        A = load_A(side)
    elems = None
    if use_cache and not force_rebuild:
        elems = _load_cache(m, side)
    if elems is None:
        elems = _build_group(top_left(A, 4 * m), m)
        _save_cache(m, side, elems)
    return elems


def block0_stabilizer_elements(m=3, side="L", A=None):
    """Subgroup of the m-block census with tau(0) = 0.

    Order 24^m * (m-1)!  (m=3: 24^3 * 2 = 27648).  Never cached; builds fast.
    """
    if m not in (1, 2, 3):
        raise ValueError("m must be 1, 2 or 3")
    if A is None:
        A = load_A(side)
    return _build_group(top_left(A, 4 * m), m, fix_block0=True)


# --------------------------------------------- independent exhaustive search

def all_signed_normalizers(A):
    """Every (perm, signs) with D A D^-1 == A, by exhaustive backtracking.

    Sound and complete: the pair constraint for rows (i, j) is enforced as
    soon as both rows are placed (equations for entries (i,j), (j,i) and the
    diagonal (i,i), where the signs cancel), and at each depth every unused
    row index and both signs are tried.  No block structure is assumed, so
    set equality with the wreath census proves the census is the FULL
    signed-permutation normalizer.
    """
    n = len(A)
    rows = [tuple(r) for r in A]
    results = []
    perm = [-1] * n
    signs = [0] * n
    used = [False] * n

    def bt(i):
        if i == n:
            results.append((tuple(perm), tuple(signs)))
            return
        row_i = rows[i]
        for k in range(n):
            if used[k]:
                continue
            row_k = rows[k]
            if row_k[k] != row_i[i]:  # entry (i,i): signs cancel
                continue
            for sg in (1, -1):
                ok = True
                for j in range(i):
                    pj = perm[j]
                    ss = sg * signs[j]
                    if ss * row_k[pj] != row_i[j]:        # entry (i,j)
                        ok = False
                        break
                    if ss * rows[pj][k] != rows[j][i]:    # entry (j,i)
                        ok = False
                        break
                if ok:
                    used[k] = True
                    perm[i] = k
                    signs[i] = sg
                    bt(i + 1)
                    used[k] = False
                    perm[i] = -1

    bt(0)
    return results


# --------------------------------------------------------------- self check

def perm_parity(p):
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inv += 1
    return inv & 1


def _fmt(M):
    return " | ".join(" ".join("%2d" % x for x in row) for row in M)


def main():
    t_all = time.perf_counter()
    ok_all = True

    def check(label, ok, info=""):
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        suffix = "  (%s)" % info if info else ""
        print("[%s] %s%s" % ("PASS" if ok else "FAIL", label, suffix))
        return ok

    print("group_n.py self-verification")
    print("repo root : %s" % DEFAULT_REPO_ROOT)
    print("cache dir : %s" % CACHE_DIR)
    print()

    # ---- artifact and block structure
    print("== artifact and block structure ==")
    A_L = load_A("L")
    A_R = load_A("R")
    check("A_L and A_R load as 12x12 integer matrices",
          len(A_L) == 12 and len(A_R) == 12
          and all(len(r) == 12 for r in A_L + A_R))
    check("A_L is block diagonal (three 4x4 blocks)", is_block_diagonal(A_L))
    check("A_R is block diagonal (three 4x4 blocks)", is_block_diagonal(A_R))
    BL = blocks(A_L)
    BR = blocks(A_R)
    for b, M in enumerate(BL):
        print("  A_L block B%d: %s" % (b, _fmt(M)))
    for b, M in enumerate(BR):
        print("  A_R block C%d: %s" % (b, _fmt(M)))
    check("A_L blocks pairwise distinct as matrices", len(set(BL)) == 3)
    check("A_R blocks pairwise distinct as matrices", len(set(BR)) == 3)
    dense = all(x != 0 for M in BL + BR for row in M for x in row)
    check("every block entry nonzero (block-support certificate)", dense,
          "support of A is exactly the block partition, so any conjugating permutation carries blocks onto blocks")

    # ---- m = 1
    print()
    print("== m=1: 4x4 block normalizer ==")
    N0 = block_normalizers(BL[0])
    check("|N(B0)| = 24", len(N0) == 24,
          "exhaustive over all 24*16 = 384 signed perms of range(4)")
    even4 = sorted(p for p in permutations(range(4)) if perm_parity(p) == 0)
    check("perm parts of N(B0) are exactly A4 (the 12 even permutations)",
          sorted({p for (p, s) in N0}) == even4)
    by_perm = {}
    for p, s in N0:
        by_perm.setdefault(p, []).append(s)
    check("each even perm carries exactly 2 sign patterns, negatives of each other",
          len(by_perm) == 12 and all(
              len(ss) == 2 and ss[0] == tuple(-x for x in ss[1])
              for ss in by_perm.values()))
    check("|N(B1)| = |N(B2)| = 24",
          len(block_normalizers(BL[1])) == 24 and len(block_normalizers(BL[2])) == 24)
    E1 = group_elements(1, "L", A=A_L, use_cache=False)
    check("m=1 census equals block_normalizers(B0)",
          len(E1) == 24 and set(E1) == set(N0))

    # ---- m = 2
    print()
    print("== m=2: top-left 8x8 of A_L ==")
    A8 = top_left(A_L, 8)
    t = time.perf_counter()
    E2 = group_elements(2, "L", A=A_L, use_cache=False)
    t2 = time.perf_counter()
    check("|N(A8)| = 1152", len(E2) == 1152, "wreath build %.2fs" % (t2 - t))
    check("elements distinct", len(set(E2)) == len(E2))
    ok = all(normalizes(A8, p, s) for (p, s) in E2)
    t3 = time.perf_counter()
    check("every element satisfies P A8 P^-1 = A8", ok,
          "1152 x 64 entry checks, %.2fs" % (t3 - t2))
    X2 = all_signed_normalizers(A8)
    t4 = time.perf_counter()
    check("independent exhaustive search finds exactly the same 1152",
          len(X2) == 1152 and set(X2) == set(E2),
          "backtracking over all signed perms of range(8), %.2fs" % (t4 - t3))

    # ---- m = 3
    print()
    print("== m=3: full 12x12 A_L normalizer ==")
    A12 = A_L
    t = time.perf_counter()
    E3 = group_elements(3, "L", A=A_L, use_cache=False)
    t2 = time.perf_counter()
    check("|N(A12)| = 82944", len(E3) == 82944, "wreath build %.2fs" % (t2 - t))
    check("elements distinct", len(set(E3)) == len(E3))
    ok = all(normalizes(A12, p, s) for (p, s) in E3)
    t3 = time.perf_counter()
    check("every element satisfies P A12 P^-1 = A12", ok,
          "82944 x 144 entry checks, %.2fs" % (t3 - t2))
    X3 = all_signed_normalizers(A12)
    t4 = time.perf_counter()
    check("independent exhaustive search finds exactly the same 82944",
          len(X3) == 82944 and set(X3) == set(E3),
          "backtracking over all signed perms of range(12), %.2fs" % (t4 - t3))
    reloaded = _load_cache(3, "L")
    check("cache round-trip /tmp/lever_a/group_m3_L.json", reloaded == E3)

    # ---- global -I and sign pairing
    print()
    print("== global -I and sign pairing ==")
    S3 = set(E3)
    minus_I = (tuple(range(12)), (-1,) * 12)
    check("(identity perm, all signs -1) is in the group", minus_I in S3)
    check("conjugation by global -I fixes A_L entrywise",
          conj_matrix(A_L, *minus_I) == [list(r) for r in A_L])
    twins = all((p, tuple(-x for x in s)) in S3 for (p, s) in E3)
    check("every element's sign-negated twin is in the group", twins,
          "effective order = 82944 / 2 = 41472")

    # ---- block-0 stabilizer
    print()
    print("== block-0 stabilizer (tau(0)=0) at m=3 ==")
    ST = block0_stabilizer_elements(3, "L", A=A_L)
    check("|stab(block 0)| = 27648", len(ST) == 27648)
    check("stabilizer is a subset of the full group", set(ST) <= S3)
    check("every stabilizer element satisfies P A12 P^-1 = A12",
          all(normalizes(A12, p, s) for (p, s) in ST))
    check("stabilizer perms carry rows 0..3 onto block 0",
          all(sorted(p[:4]) == [0, 1, 2, 3] for (p, s) in ST))
    ST_indep = [e for e in X3 if sorted(e[0][:4]) == [0, 1, 2, 3]]
    check("independent search: same 27648 stabilizer",
          len(ST_indep) == 27648 and set(ST_indep) == set(ST))

    # ---- conjugator counts between blocks
    print()
    print("== conjugator counts between blocks ==")
    within = {(i, j): len(conjugator_sets(BL[i], BL[j]))
              for i in range(3) for j in range(3) if i != j}
    check("A_L block pairs: 6/6 ordered pairs have 24 conjugators",
          len(within) == 6 and all(v == 24 for v in within.values()),
          "counts %s" % sorted(within.values()))
    cross = {(i, j): len(conjugator_sets(BL[i], BR[j]))
             for i in range(3) for j in range(3)}
    check("A_L blocks -> A_R blocks: 9/9 ordered pairs have 24 conjugators",
          len(cross) == 9 and all(v == 24 for v in cross.values()),
          "counts %s" % sorted(cross.values()))

    # ---- algebra sanity on strided samples (for tomorrow's orbit join)
    print()
    print("== compose / act_on_vector identities (strided samples from m=3) ==")
    pairs = [(E3[i], E3[i + 1]) for i in range(17, len(E3) - 1, 401)]
    ok_c = all(
        conj_matrix(conj_matrix(A_L, *e1), *e2) == conj_matrix(A_L, *compose(e2, e1))
        for (e1, e2) in pairs
    )
    check("conj(conj(G,e1),e2) == conj(G, compose(e2,e1))", ok_c,
          "%d sampled pairs" % len(pairs))
    v = tuple(range(1, 13))
    ok_a = all(
        act_on_vector(act_on_vector(v, *e1), *e2) == act_on_vector(v, *compose(e2, e1))
        for (e1, e2) in pairs
    )
    check("act(act(v,e1),e2) == act(v, compose(e2,e1))", ok_a)
    ident = (tuple(range(12)), (1,) * 12)
    ok_i = all(compose(e, ident) == e and compose(ident, e) == e
               for (e, _) in pairs[:20])
    check("compose with identity is the identity", ok_i)

    print()
    t_end = time.perf_counter()
    if ok_all:
        print("ALL CHECKS PASSED  (total %.2fs)" % (t_end - t_all))
        return 0
    print("SOME CHECKS FAILED  (total %.2fs)" % (t_end - t_all))
    return 1


if __name__ == "__main__":
    sys.exit(main())
