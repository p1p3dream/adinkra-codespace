#!/usr/bin/env python3
"""Exact scan: which stored census representatives conjugate the CLS color
matrices L_I to signed-permutation (Garden presentation) form?

Motivation: an external exact-arithmetic review found that only 9 of the
4,077 distinct stored representative matrices G produce signed-permutation
conjugates G L_I G^-1 for all four colors, that the property is NOT constant
on the (nnz, support, ranks) taxonomy classes, and that a claimed population
of 33,696 qualifying matrices is therefore unsupported.  This script
re-derives all of those numbers independently from the committed shards.

Facts used (src/four_color/cls.rs, arXiv:2408.09342 Appendix C):
  - the four CLS L-matrices at dim 12, transcribed in signed-address form;
  - the Garden algebra L_I L_J^T + L_J L_I^T = 2 delta_IJ I (the arbiter of
    the transcription), which for skew monomial L_I is accompanied by the
    multiplicative relations L_I^2 = -I, L_I L_J = -L_J L_I;
  - A_L = (L_1 + L_2 + L_3 + L_4) L_1^-1  (src/four_color/gmatrix_verify.rs),
    tying the color matrices to the census target.

Two relation families are tracked separately because they behave differently
under similarity by a NON-orthogonal G:
  - multiplicative relations are preserved by any similarity (so checking
    them on conjugates carries no information beyond C_I^2 = G L_I^2 G^-1);
  - the transpose-Garden relation is NOT similarity-invariant in general;
    for monomial conjugates the diagonal part is automatic (orthogonality)
    and the off-diagonal part (C_I C_J^T skew) is a genuine extra condition.

Buckets (matching the review):
  1  all four C_I integer signed-permutation matrices
  2  all four C_I integer, at least one not signed-permutation
  3  at least one C_I has a non-integer entry

Run: python3 scripts/lever_a/garden_presentation_scan.py
"""
import glob
import json
import os
import sys
from fractions import Fraction
from itertools import permutations

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARD_DIRS = [
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_stonkbot_mirror"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_b300_mirror"),
]
A_PATH = os.path.join(ROOT, "results", "four_color_cls_gmatrix.json")

# CLS L-matrices, Appendix C signed-address form (row i -> sign*(col+1)).
CLS_L = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [2, -1, 4, -3, 6, -5, -8, 7, -10, 9, 12, -11],
    [3, -4, -1, 2, -7, -8, 5, 6, 11, 12, -9, -10],
    [4, 3, -2, -1, -8, 7, -6, 5, -12, 11, -10, 9],
]
N = 12


def address_to_matrix(addr):
    M = [[Fraction(0)] * N for _ in range(N)]
    for i, a in enumerate(addr):
        M[i][abs(a) - 1] = Fraction(1 if a > 0 else -1)
    return M


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(N)) for j in range(N)]
            for i in range(N)]


def transpose(A):
    return [[A[j][i] for j in range(N)] for i in range(N)]


def matadd(A, B):
    return [[A[i][j] + B[i][j] for j in range(N)] for i in range(N)]


def inverse(A):
    """Exact rational inverse by Gauss-Jordan."""
    M = [list(map(Fraction, A[i])) + [Fraction(1 if i == j else 0) for j in range(N)]
         for i in range(N)]
    for c in range(N):
        p = next(r for r in range(c, N) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(N):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[N:] for row in M]


def is_signed_perm(C):
    """Exactly one nonzero per row and per column, every nonzero exactly +-1."""
    for row in C:
        nz = [x for x in row if x != 0]
        if len(nz) != 1 or abs(nz[0]) != 1:
            return False
    for j in range(N):
        nz = [C[i][j] for i in range(N) if C[i][j] != 0]
        if len(nz) != 1 or abs(nz[0]) != 1:
            return False
    return True


def is_integer(C):
    return all(x.denominator == 1 for row in C for x in row)


checks = []


def check(label, ok, info=""):
    checks.append(ok)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", label,
                         ("  (%s)" % info) if info else ""))


# ---------------------------------------------------------------- sanity
L = [address_to_matrix(a) for a in CLS_L]
check("all four L_I are signed permutations", all(is_signed_perm(M) for M in L))
ident = [[Fraction(1 if i == j else 0) for j in range(N)] for i in range(N)]
garden_ok = True
for i in range(4):
    for j in range(4):
        P = matadd(matmul(L[i], transpose(L[j])), matmul(L[j], transpose(L[i])))
        if i == j:
            garden_ok = garden_ok and all(P[r][c] == (2 if r == c else 0)
                                          for r in range(N) for c in range(N))
        else:
            garden_ok = garden_ok and all(x == 0 for row in P for x in row)
check("L_I satisfy the transpose-Garden algebra L_i L_j^T + L_j L_i^T = 2 delta IJ", garden_ok)
# Which colors satisfy the multiplicative L_i^2 = -I at all?  (L_1 = I can never
# satisfy it; the review's "(C_I^2=-I)" phrasing is not the load-bearing check
# here.  Report the factual pattern; require only that the facts be as stated.)
sq_neg = []
for i in range(4):
    S = matmul(L[i], L[i])
    sq_neg.append(all(S[r][c] == (-1 if r == c else 0) for r in range(N) for c in range(N)))
check("multiplicative squares: L_1^2 = +I (it is the identity), "
      "L_2^2 = L_3^2 = L_4^2 = -I (they are skew)",
      sq_neg == [False, True, True, True], "pattern = %s" % sq_neg)
# Colors 2-4 anticommute multiplicatively; pairs involving color 1 cannot
# (L_1 = I gives L_1 L_j + L_j L_1 = 2 L_j).  These premises plus monomiality
# of the conjugates IMPLY the transpose-Garden relation for bucket 1, so the
# 9/9 entrywise Garden check below is a corollary, not an independent filter.
ac_234 = all(all(x == 0 for row in matadd(matmul(L[i], L[j]), matmul(L[j], L[i]))
                 for x in row)
             for i in range(1, 4) for j in range(i + 1, 4))
check("multiplicative anticommutation L_i L_j = -L_j L_i for colors 2,3,4", ac_234)
check("anticommutation with color 1 fails as forced (L_1 = I)",
      any(any(x != 0 for row in matadd(matmul(L[0], L[j]), matmul(L[j], L[0]))
              for x in row) for j in range(1, 4)))

A_artifact = json.load(open(A_PATH))["A_L"]
A_from_L = matmul(matadd(matadd(matadd(L[0], L[1]), L[2]), L[3]), inverse(L[0]))
check("A_L = (L_1+L_2+L_3+L_4) L_1^-1 matches the census artifact entrywise",
      all(A_from_L[r][c] == A_artifact[r][c] for r in range(N) for c in range(N)))

# ---------------------------------------------------------------- records
records = []            # (item, taxonomy key, matrix)
for d in SHARD_DIRS:
    for p in sorted(glob.glob(os.path.join(d, "shard_*.json"))):
        s = json.load(open(p))
        for c in s["classes"]:
            records.append((s["item"], (c["nnz"], c["support"], tuple(c["ranks"])),
                            tuple(tuple(r) for r in c["rep"])))
check("4,536 stored representative records across all shard files",
      len(records) == 4536, "%d" % len(records))

distinct = sorted({m for _, _, m in records})
check("4,077 distinct stored matrices", len(distinct) == 4077, "%d" % len(distinct))
keys = {k for _, k, _ in records}
check("1,076 taxonomy keys", len(keys) == 1076, "%d" % len(keys))

# ------------------------------------------------------- conjugation scan
def conjugates(G):
    """All four C_I = G L_I G^-1, always computed in full (no short-circuit:
    the per-color flag statistics below need every color even when the matrix
    lands in the noninteger bucket)."""
    Gf = [list(map(Fraction, row)) for row in G]
    Ginv = inverse(Gf)
    return [matmul(matmul(Gf, I), Ginv) for I in L]


def classify(G):
    """Return (bucket, Cs, integer flags, signed-perm flags).

    1 all four C_I integer signed-permutation matrices
    2 all four integer, at least one not signed-permutation
    3 at least one non-integer entry
    """
    Cs = conjugates(G)
    int_flags = tuple(is_integer(C) for C in Cs)
    sp_flags = tuple(is_signed_perm(C) for C in Cs)
    if not all(int_flags):
        b = 3
    elif all(sp_flags):
        b = 1
    else:
        b = 2
    return b, Cs, int_flags, sp_flags


bucket_of = {}
sp_of = {}
int_of = {}
for idx, G in enumerate(distinct):
    b, _, int_flags, sp_flags = classify(G)
    bucket_of[G] = b
    int_of[G] = int_flags
    sp_of[G] = sp_flags
    if idx % 500 == 0:
        print("  ... %d/%d classified" % (idx, len(distinct)), file=sys.stderr)

counts = [0, 0, 0]
for G in distinct:
    counts[bucket_of[G] - 1] += 1
print()
print("classification over all %d distinct stored matrices:" % len(distinct))
print("  all four colors signed-permutation:          %d" % counts[0])
print("  integer but only partially signed-perm:      %d" % counts[1])
print("  at least one noninteger conjugated color:    %d" % counts[2])

# Relation checks on the bucket-1 presentations.  COROLLARY, not an
# independent filter: if all four C_I are signed permutations (orthogonal,
# C^-1 = C^T) then the similarity-preserved squares (C_i^2 = -I for colors
# 2-4) give C_i^T = -C_i, and the preserved anticommutation gives
# C_i C_j^T + C_j C_i^T = -(C_i C_j + C_j C_i) = 0.  The check below confirms
# the implication holds entrywise on the data.  The equivalent centralizer
# statement (G^T G commutes with every L_i L_j^T) was independently
# confirmed by the external review.
g1 = [G for G in distinct if bucket_of[G] == 1]
tp_garden = 0
for G in g1:
    _, Cs, _, _ = classify(G)
    ok = True
    for i in range(4):
        for j in range(4):
            P = matadd(matmul(Cs[i], transpose(Cs[j])),
                       matmul(Cs[j], transpose(Cs[i])))
            want_zero_or_diag = (i != j)
            if want_zero_or_diag:
                ok = ok and all(x == 0 for row in P for x in row)
            else:
                ok = ok and all(P[r][c] == (2 if r == c else 0)
                                for r in range(N) for c in range(N))
    tp_garden += ok
print("  bucket-1 presentations also satisfying transpose-Garden entrywise "
      "(corollary, see comment): %d/%d" % (tp_garden, len(g1)))

# ------------------------------------------------ taxonomy-constancy stats
# Five explicit definitions of "the key's reps change classification", all
# over the full four-color flag vectors.  The external review's 55 is the
# combined (bucket, signed-perm flags) definition.  (An earlier version of
# this script short-circuited classify() at the first non-integer color and
# computed the flag statistics on truncated color lists; that produced wrong
# variant counts 119/126.  classify() now always returns all four colors.)
mats_per_key = {}
for item, k, m in records:
    mats_per_key.setdefault(k, set()).add(m)
multi_rep_keys = sum(1 for v in mats_per_key.values() if len(v) > 1)


def diversity(key_getter):
    per_key = {}
    for item, k, m in records:
        per_key.setdefault(k, set()).add(key_getter(m))
    return sum(1 for v in per_key.values() if len(v) > 1)


mixed_bucket = diversity(lambda m: bucket_of[m])
mixed_sp = diversity(lambda m: sp_of[m])
mixed_bucket_sp = diversity(lambda m: (bucket_of[m], sp_of[m]))
mixed_nint = diversity(lambda m: sum(int_of[m]))
mixed_int = diversity(lambda m: int_of[m])

print()
print("taxonomy keys with multiple distinct stored representatives: %d" % multi_rep_keys)
print("keys whose stored reps change, by definition of the classification:")
print("  coarse bucket (3-way):                          %d" % mixed_bucket)
print("  full four-color signed-permutation flags:       %d" % mixed_sp)
print("  combined (bucket, signed-perm flags):           %d" % mixed_bucket_sp)
print("  number of integer colors:                       %d" % mixed_nint)
print("  full four-color integer flags:                  %d" % mixed_int)

# ------------------------------------------------------------- the 9
print()
print("the bucket-1 matrices (item provenance, first record each):")
seen = set()
for item, k, m in records:
    if bucket_of[m] == 1 and m not in seen:
        seen.add(m)
        print("  item %3d  key nnz=%d support=%d" % (item, k[0], k[1]))

ok_all = (all(checks) and counts == [9, 228, 3840] and multi_rep_keys == 712
          and (mixed_bucket, mixed_sp, mixed_bucket_sp, mixed_nint, mixed_int)
          == (34, 50, 55, 182, 188))
print()
print("REVIEW NUMBERS REPRODUCED: buckets %s (expected [9, 228, 3840]), "
      "%d multi-rep keys (expected 712); mixed-key counts by definition: "
      "bucket %d, signed-perm flags %d, combined %d, #integer %d, "
      "integer flags %d (expected 34/50/55/182/188)"
      % (counts, multi_rep_keys, mixed_bucket, mixed_sp, mixed_bucket_sp,
         mixed_nint, mixed_int))
print("ALL CHECKS PASSED" if ok_all else "CHECK ABOVE FOR FAILURES")
sys.exit(0 if ok_all else 1)
