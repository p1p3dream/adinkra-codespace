#!/usr/bin/env python3
"""Equivalence analysis of the bucket-1 (Garden-presentation) conjugates.

Context: the census scan found 9 stored G-matrices whose conjugates
C_I = G L_I G^-1 are all signed permutations. A follow-up session computed a
holoraumy gadget matrix over {CLS, T1, T2, T3} and read the cross entries
(0s and 1s against a self-value of 3) as "CLS holoraumy-orthogonal to its
conjugates" and "targets sharing one irreducible component". That reading is
wrong: all targets are conjugates of CLS by construction, hence isomorphic,
so any isomorphism-invariant comparison must return the self-value on every
entry. This script demonstrates exactly where the invariance breaks and then
answers the question that actually matters (the external review's step 2):
are the conjugated presentations equivalent to the original under signed
node relabelings, color permutations, and color sign rescalings?

Method: exact rational/integer arithmetic only.

  1. Recompute the 9 bucket-1 conjugates from the shard files (full
     classification pass, same rules as garden_presentation_scan.py).
  2. Reproduce the gadget matrix from the definition in src/holoraumy.rs,
     then conjugate a target by a MONOMIAL relabeling (definitionally the
     same representation) and show the cross entries move: the gadget is a
     basis-alignment functional of the intertwiner, not an invariant.
  3. Verify the 9 conjugates collapse to 3 distinct ordered quadruples
     and that the collapse is commutant arithmetic (G_j^-1 G_i commutes
     with all four L_I).
  4. Exact signed-monomial equivalence search: all quadruples involved are
     block-diagonal 4+4+4, and each 4-node block is a connected component of
     the union of its colored node graph. A signed monomial intertwiner
     preserves colored adjacency, so it maps each whole connected block onto
     a whole connected block. Brute force over all 3! block assignments, all
     384 signed permutations inside each 4x4 block, and the allowed global
     color permutations and color signs therefore decides the stated
     signed-monomial equivalence relation exactly.

Run: python3 scripts/lever_a/garden_target_equivalence_check.py
"""
import glob
import json
import os
import sys
from fractions import Fraction
from itertools import permutations, product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHARD_DIRS = [
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_stonkbot_mirror"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_b300_mirror"),
]

CLS_L = [
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    (2, -1, 4, -3, 6, -5, -8, 7, -10, 9, 12, -11),
    (3, -4, -1, 2, -7, -8, 5, 6, 11, 12, -9, -10),
    (4, 3, -2, -1, -8, 7, -6, 5, -12, 11, -10, 9),
]
N = 12
NB = 4

checks = []


def check(label, ok, info=""):
    checks.append(ok)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", label,
                         ("  (%s)" % info) if info else ""))


# ------------------------------------------------- integer address algebra
def addr_to_mat(addr):
    n = len(addr)
    M = [[0] * n for _ in range(n)]
    for i, a in enumerate(addr):
        M[i][abs(a) - 1] = 1 if a > 0 else -1
    return M


def mat_to_addr(M):
    n = len(M)
    addr = []
    for i in range(n):
        nz = [(j + 1, M[i][j]) for j in range(n) if M[i][j] != 0]
        if len(nz) != 1:
            return None
        addr.append(int(nz[0][0]) if nz[0][1] == 1 else -int(nz[0][0]))
    a = tuple(addr)
    # column condition
    cols = [0] * n
    for i, x in enumerate(a):
        cols[abs(x) - 1] += 1
    return a if all(c == 1 for c in cols) else None


def comp_addr(a, b):
    """Product A.B of signed permutations in address form (A row i -> col
    |a_i| with sign; result row i = a_i-scaled row |a_i|-1 of B)."""
    return tuple(b[abs(x) - 1] if x > 0 else -b[abs(x) - 1] for x in a)


def inv_addr(a):
    inv = [0] * len(a)
    for i, x in enumerate(a):
        inv[abs(x) - 1] = (i + 1) if x > 0 else -(i + 1)
    return tuple(inv)


def trace_prod(a, b):
    """Tr(A.B) for signed permutations in address form: +1 for a +1 on the
    diagonal, -1 for a -1 (a signed permutation's diagonal entries are in
    {-1, 0, 1})."""
    ab = comp_addr(a, b)
    t = 0
    for i, x in enumerate(ab):
        if x == i + 1:
            t += 1
        elif x == -(i + 1):
            t -= 1
    return t


def matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def inverse(A):
    n = len(A)
    M = [[Fraction(A[i][j]) for j in range(n)] +
         [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


L = [addr_to_mat(a) for a in CLS_L]

# ------------------------------------------------------------- 1. the nine
records = {}
for d in SHARD_DIRS:
    for p in sorted(glob.glob(os.path.join(d, "shard_*.json"))):
        s = json.load(open(p))
        for c in s["classes"]:
            records[tuple(tuple(r) for r in c["rep"])] = s["item"]
check("4,077 distinct stored matrices", len(records) == 4077, "%d" % len(records))

bucket1 = []
for idx, G in enumerate(sorted(records)):
    Gf = [[Fraction(x) for x in row] for row in G]
    Ginv = inverse(Gf)
    Cs = [matmul(matmul(Gf, L[I]), Ginv) for I in range(4)]
    if all(all(x.denominator == 1 for row in C for x in row)
           and mat_to_addr([[int(x) for x in row] for row in C]) is not None
           for C in Cs):
        bucket1.append((G, tuple(mat_to_addr(
            [[int(x) for x in row] for row in C]) for C in Cs)))
    if idx % 1000 == 0:
        print("  ... %d/%d classified" % (idx, len(records)), file=sys.stderr)
check("exactly 9 bucket-1 conjugates", len(bucket1) == 9, "%d" % len(bucket1))

quad_count = {}
for G, quad in bucket1:
    quad_count.setdefault(quad, []).append(G)
check("the 9 collapse to 3 distinct ordered quadruples", len(quad_count) == 3,
      "group sizes %s" % sorted((len(v) for v in quad_count.values()),
                                reverse=True))

comm_ok = True
for quad, Gs in quad_count.items():
    G0inv = inverse([[Fraction(x) for x in row] for row in Gs[0]])
    for G in Gs[1:]:
        T = matmul(G0inv, [[Fraction(x) for x in row] for row in G])
        for I in range(4):
            if matmul(T, [[Fraction(x) for x in row] for row in L[I]]) != \
               matmul([[Fraction(x) for x in row] for row in L[I]], T):
                comm_ok = False
check("within each collapse group, G_j^-1 G_i commutes with all four L_I",
      comm_ok)

# ------------------------------------------------- 2. gadget non-invariance
def vtilde_addrs(quad):
    """Fermionic holoraumy L_I^-1 L_J, I>J, address form (matches
    src/holoraumy.rs HoloraumyData::from_rep)."""
    out = []
    for i in range(1, 4):
        for j in range(i):
            out.append(comp_addr(inv_addr(quad[i]), quad[j]))
    return out


def gadget(quad_a, quad_b):
    """Exact gadget per src/holoraumy.rs:
    -2/(N(N-1) dmin) * sum Tr(Vtilde_a Vtilde_b), dmin(4) = 4."""
    va = vtilde_addrs(quad_a)
    vb = vtilde_addrs(quad_b)
    s = sum(trace_prod(a, b) for a, b in zip(va, vb))
    return Fraction(-2 * s, 4 * 3 * 4)


orig_quad = tuple(tuple(a) for a in CLS_L)
targets = sorted(quad_count, key=lambda q: -len(quad_count[q]))
reps = [orig_quad] + list(targets)
labels = ["CLS"] + ["T%d(%dG)" % (i + 1, len(quad_count[q]))
                    for i, q in enumerate(targets)]
print()
print("gadget matrix, exact rational (reproduces the session's output):")
print("            " + "".join("%12s" % l for l in labels))
Gmat = [[gadget(a, b) for b in reps] for a in reps]
for i, row in enumerate(Gmat):
    print("%11s " % labels[i] + "".join("%12s" % str(x) for x in row))

check("isomorphic pairs already disagree with self-values in the matrix "
      "above (cross 0 or 1 vs self 3)",
      any(Gmat[i][j] != Gmat[i][i] for i in range(len(reps))
          for j in range(len(reps)) if i != j))

# ------------------------------------------------- 3. block structure
def blocks(M):
    return [[row[4 * j:4 * j + 4] for row in M[4 * j:4 * j + 4]]
            for j in range(3)]


def block_diag(quad):
    for a in quad:
        M = addr_to_mat(a)
        for i in range(N):
            for j in range(N):
                if i // 4 != j // 4 and M[i][j] != 0:
                    return False
    return True


def block_connected(quad, block_index):
    """Connectivity of one 4-node block in the unsigned union of color edges.

    Identity-color loops do not affect connectivity. Signs also do not affect
    the underlying colored adjacency that a signed monomial intertwiner must
    preserve.
    """
    lo = NB * block_index
    hi = lo + NB
    adj = {i: set() for i in range(lo, hi)}
    for addr in quad[1:]:
        for i in range(lo, hi):
            j = abs(addr[i]) - 1
            if not lo <= j < hi:
                return False
            adj[i].add(j)
            adj[j].add(i)
    seen = {lo}
    stack = [lo]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == NB


check("original L_I are block-diagonal 4+4+4", block_diag(orig_quad))
check("all 3 target quadruples are block-diagonal 4+4+4",
      all(block_diag(q) for q in targets))
check("every 4-node block is connected in the union colored graph",
      all(block_connected(q, k) for q in [orig_quad] + list(targets)
          for k in range(3)))

OB = [[blocks(addr_to_mat(a))[k] for a in orig_quad] for k in range(3)]
# OB[k][I] = 4x4 block k of color I.

COLOR_PERMS = [p for p in permutations(range(4)) if p[0] == 0]


def block_origin_candidates(quad):
    """Per target block j: ALL (k, sigma) with quad block j, color I equal to
    OB[k][sigma[I]] for every color."""
    Ms = [blocks(addr_to_mat(a)) for a in quad]
    out = []
    for j in range(3):
        cands = []
        for k in range(3):
            for sig in COLOR_PERMS:
                if all(Ms[I][j] == OB[k][sig[I]] for I in range(4)):
                    cands.append((k, sig))
        out.append(cands)
    return out


def cyc(sig):
    names = ["1", "2", "3", "4"]
    parts, seen = [], set()
    for s in range(1, 4):
        if s in seen:
            continue
        cyc_l = []
        x = s
        while x not in seen:
            seen.add(x)
            cyc_l.append(names[x])
            x = sig[x]
        if len(cyc_l) > 1:
            parts.append("".join(cyc_l))
    return "id" if not parts else "(" + ")(".join(parts) + ")"


for i, q in enumerate(targets):
    cands = block_origin_candidates(q)
    uniq = all(len(c) == 1 for c in cands)
    plan = " ".join("b%d<-b%d cols %s" % (
        j + 1, c[0][0] + 1 if len(c) == 1 else -1,
        cyc(c[0][1]) if len(c) == 1 else "?%d" % len(c))
        for j, c in enumerate(cands))
    print("T%d block plan: %s%s" % (i + 1, plan,
                                    "" if uniq else "  (AMBIGUOUS)"))
    check("T%d blocks are original blocks under color permutations" % (i + 1),
          all(len(c) >= 1 for c in cands),
          "unique" if uniq else "ambiguous, all candidates explored below")

# --------------------------------------------- 4. exact equivalence search
# P[(k1,k2)] = {(sigma, eps): exists a 4x4 signed perm S with
#   S OB[k1][I] S^-1 = eps[I] * OB[k2][sigma[I]] for I = 1,2,3}
# (color 1 is the identity block, so sigma fixes color 1 and eps[0] = +1).
def sp4(perm, signs):
    M = [[0] * NB for _ in range(NB)]
    for i in range(NB):
        M[i][perm[i]] = signs[i]
    addr = tuple((perm[i] + 1) * signs[i] for i in range(NB))
    return (M, [[Fraction(x) for x in r] for r in inverse(M)], addr)


SIGNED_PERMS_4 = [sp4(p, s) for p in permutations(range(NB))
                  for s in product([1, -1], repeat=NB)]

EPS_SETS = [e for e in product([1, -1], repeat=4) if e[0] == 1]
target_triples = {}
for k2 in range(3):
    for sig in COLOR_PERMS:
        for eps in EPS_SETS:
            key = tuple(tuple(tuple(eps[I] * x for x in row)
                              for row in OB[k2][sig[I]]) for I in (1, 2, 3))
            target_triples.setdefault(key, []).append((k2, sig, eps))

def conj4(S, A, Sinv):
    """S A S^-1 for 4x4 blocks; entries are provably in {-1,0,1}."""
    return [[int(sum(sum(S[i][t] * A[t][u] for t in range(NB)) * Sinv[u][j]
                     for u in range(NB))) for j in range(NB)] for i in range(NB)]


P = {(a, b): {} for a in range(3) for b in range(3)}
for k1 in range(3):
    A123 = [OB[k1][I] for I in (1, 2, 3)]
    for S, Sinv, saddr in SIGNED_PERMS_4:
        key = tuple(tuple(tuple(r) for r in conj4(S, A, Sinv)) for A in A123)
        for (k2, sig, eps) in target_triples.get(key, []):
            P[(k1, k2)].setdefault((sig, eps), saddr)

for a in range(3):
    for b in range(3):
        pure = sorted({cyc(s) for (s, e) in P[(a, b)] if e == (1, 1, 1, 1)})
        print("inducible (b%d -> b%d): %d (sigma,eps) combos, pure-color: %s"
              % (a + 1, b + 1, len(P[(a, b)]), " ".join(pure) or "none"))

# Global signed-monomial equivalence, exact by the connected-block argument:
#   S signed 12-perm of block-tau form, global sigma (fixes color 1),
#   global eps (eps[0] = +1), with S L^a_I S^-1 = eps[I] L^b_{sigma[I]}.
# Per target block j fed from source block tau(j) with plans
# (ka, sa) and (kb, sb): need (sb . sigma . sa^-1, eps . sa^-1) inducible
# on the block pair (ka, kb).
def equivalent(quad_a, quad_b):
    """Returns (sigma, eps, tau, block witnesses) or None."""
    ca = block_origin_candidates(quad_a)
    cb = block_origin_candidates(quad_b)
    for sigma in COLOR_PERMS:
        for eps in EPS_SETS:
            for tau in permutations(range(3)):
                wit = []
                for j in range(3):
                    kb, sb = cb[j][0]
                    hit = None
                    for (ka, sa) in ca[tau[j]]:
                        psi = tuple(sb[sigma[sa.index(J)]] for J in range(4))
                        eps_p = tuple(eps[sa.index(J)] for J in range(4))
                        if (psi, eps_p) in P[(ka, kb)]:
                            hit = P[(ka, kb)][(psi, eps_p)]
                            break
                    if hit is None:
                        break
                    wit.append(hit)
                else:
                    return (sigma, eps, tau, wit)
    return None


def full_witness(quad_a, quad_b):
    """Assemble the 12x12 block-tau signed permutation S (address form)
    conjugating quad_a to quad_b, from the per-block witnesses."""
    w = equivalent(quad_a, quad_b)
    if w is None:
        return None
    sigma, eps, tau, wit = w
    addr = []
    for j in range(3):        # target block j is fed by source block tau[j]
        for x in wit[j]:      # 4x4 witness, 1-indexed within the block
            addr.append(tau[j] * 4 + abs(x) if x > 0
                        else -(tau[j] * 4 + abs(x)))
    return tuple(addr)


# Explicit certificate: S with S L_I S^-1 = T_I for every color, verified
# entrywise in address arithmetic.
for i, q in enumerate(targets):
    S = full_witness(orig_quad, q)
    ok = S is not None
    if ok:
        Si = inv_addr(S)
        for I in range(4):
            if comp_addr(comp_addr(S, orig_quad[I]), Si) != q[I]:
                ok = False
    check("witness certificate: S L_I S^-1 = T%d_I entrywise for all I" % (i + 1),
          ok, "S = %s" % (S,))


check("CLS is equivalent to itself (sanity, S = I certificate expected)",
      equivalent(orig_quad, orig_quad) is not None)

# Monomial rebasis demonstration: T2 is signed-node-relabeling equivalent to T1
# (witness just verified), so rebasing T2 by that witness maps it EXACTLY
# onto T1, a pure relabeling of the same representation. A basis-independent
# functional cannot move under it; the gadget does.
w21 = equivalent(targets[1], targets[0])
sigma_w, eps_w, tau_w, wit_w = w21
S21 = full_witness(targets[1], targets[0])
Si = inv_addr(S21)
rebased_T2 = tuple(comp_addr(comp_addr(S21, a), Si) for a in targets[1])
check("rebased T2 is exactly T1 (signed node relabeling)",
      rebased_T2 == targets[0])
g_before = gadget(targets[0], targets[1])
g_after = gadget(targets[0], rebased_T2)
check("gadget is NOT an invariant: relabeling T2 onto T1 moves gadget(T1, .)",
      g_after != g_before, "%s -> %s" % (g_before, g_after))

all_quads = [orig_quad] + list(targets)
names = ["CLS"] + ["T%d" % (i + 1) for i in range(len(targets))]
print()
print("equivalence under signed-node + color-perm + color-sign operations:")
for i, q in enumerate(all_quads):
    for j in range(i):
        w = equivalent(all_quads[j], q)
        if w:
            print("  %s ~ %s  (sigma=%s eps=%s tau=%s)"
                  % (names[j], names[i], cyc(w[0]), w[1],
                     "".join(str(x + 1) for x in w[2])))
        else:
            print("  %s NOT equivalent to %s" % (names[j], names[i]))

classes = []
for i, q in enumerate(all_quads):
    for cls in classes:
        if equivalent(cls[0], q):
            cls.append(names[i])
            break
    else:
        classes.append([q, names[i]])
print()
print("classes: " + "; ".join("+".join(c[1:]) for c in classes))

print()
print("ALL CHECKS PASSED" if all(checks) else "SOME CHECKS FAILED")
sys.exit(0 if all(checks) else 1)
