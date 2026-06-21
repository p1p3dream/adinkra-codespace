#!/usr/bin/env python3
"""Separate exact (symbolic / rational) evaluator of the 10D Garden Algebra L/R matrices.

This is a from-scratch, EXACT-arithmetic re-implementation of the Garden Algebra
construction, written directly from the authoritative Mathematica source file
"Garden Algebra" (the generative source shipped in github.com/mcmulaz/Super-Sym).
It is intended as the strongest available substitute for re-running the authors'
Mathematica, since no headless Wolfram Engine is installable here.

SCOPE OF WHAT THIS PROVES: this is a SECOND exact implementation of the SAME Garden
formulas, so a match corroborates "our JSON equals the Garden Algebra source" (the
coefficient bookkeeping is re-derived, so a transcription slip would show up). It is
NOT an independent proof of the paper's intended matrix convention, and it does not
resolve the paper-vs-Garden worked-example discrepancy (see PROVENANCE.md section 6).

Every construction step below cites the LINE NUMBERS of the Mathematica source
("GA:Ln") it mirrors. Arithmetic is exact throughout: sigma2 carries sympy's
imaginary unit I, all coefficients are sympy Rational, and the only irrationals
(Sqrt[8] and 1/Sqrt[8]) are kept symbolic via sympy.sqrt(8). There are no floats
anywhere in the construction.

Independence note: this evaluator is written purely from the Mathematica DEFINITIONS,
not from scripts/gen_10d_data.py. The sigma-Kronecker table (GA:Ln 9-25) and the
gauge-fix/ordering choices (GA:Ln 67-92, 107-115) are FIXED by the Mathematica
source itself, so any faithful re-derivation MUST reproduce them; those are
"shared-convention" points, not independent corroboration. The genuinely
independent content is the algebraic machinery (Sig2/Sig3/MixedLeft contractions,
the row-assembly arithmetic for R and L, and the exact coefficient bookkeeping),
which we re-derived symbolically and which could disagree if a coefficient were
mis-transcribed. See the printed report for the full breakdown.

Run:  python3 scripts/eval_garden_exact.py            (uses data/tendim_10d_lr.json)
      python3 scripts/eval_garden_exact.py path/to.json
"""

import json
import os
import sys
from itertools import permutations

import sympy as sp
from sympy import I, Rational, sqrt, Matrix, zeros, eye

# ---------------------------------------------------------------------------
# Sigma / Pauli setup  --  GA:Ln 4-25
# Exact 2x2 Pauli matrices; sigma2 carries the exact imaginary unit I.
# ---------------------------------------------------------------------------
s1 = Matrix([[0, 1], [1, 0]])           # GA:Ln 4
s2 = Matrix([[0, -I], [I, 0]])          # GA:Ln 5
s3 = Matrix([[1, 0], [0, -1]])          # GA:Ln 6
I2 = eye(2)                              # GA:Ln 7


def kron(*mats):
    """Left-folded Kronecker product, matching Mathematica KroneckerProduct[a,b,c,d]."""
    out = mats[0]
    for m in mats[1:]:
        out = sp.Matrix(sp.kronecker_product(out, m))
    return out


# sigUp[mu] : the ten 16x16 sigma matrices  --  GA:Ln 9-25 (exact transliteration of
# the Kronecker structure FIXED by the Mathematica source; shared-convention).
sigUp = {
    0: kron(I2, I2, I2, I2),            # GA:Ln 9
    1: kron(s2, s2, s2, s2),            # GA:Ln 10-11
    2: kron(s2, s2, I2, s1),            # GA:Ln 12-13
    3: kron(s2, s2, I2, s3),            # GA:Ln 14-15
    4: kron(s2, s1, s2, I2),            # GA:Ln 16-17
    5: kron(s2, s3, s2, I2),            # GA:Ln 18-19
    6: kron(s2, I2, s1, s2),            # GA:Ln 20-21
    7: kron(s2, I2, s3, s2),            # GA:Ln 22-23
    8: kron(s1, I2, I2, I2),            # GA:Ln 24
    9: kron(s3, I2, I2, I2),            # GA:Ln 25
}


def etasign(mu):
    """eta-sign: -1 for the timelike index 0, +1 otherwise.  GA:Ln 27."""
    return -1 if mu == 0 else 1


def sigma(mu):
    """Sigma[mu] = etasign[mu] * sigUp[mu].  GA:Ln 28."""
    return etasign(mu) * sigUp[mu]


def sigmaTildeUp(mu):
    """SigmaTildeUp[mu] = etasign[mu] * sigUp[mu].  GA:Ln 30."""
    return etasign(mu) * sigUp[mu]


def sigmaTilde(mu):
    """SigmaTilde[mu] = etasign[mu] * SigmaTildeUp[mu].  GA:Ln 31."""
    return etasign(mu) * sigmaTildeUp(mu)


# ---------------------------------------------------------------------------
# Sigma-bilinear / trilinear contractions  --  GA:Ln 33-63
# Genuinely-independent algebra: re-derived directly from the symbolic defs.
# ---------------------------------------------------------------------------
def Sig2Up(mu, nu):
    """GA:Ln 33-35.  (1/2)(sigUp[mu].tildeUp[nu] - sigUp[nu].tildeUp[mu])."""
    return Rational(1, 2) * (sigUp[mu] * sigmaTildeUp(nu) - sigUp[nu] * sigmaTildeUp(mu))


def Sig2(mu, nu):
    """GA:Ln 37-38.  etasign[mu] etasign[nu] Sig2Up[mu,nu]."""
    return etasign(mu) * etasign(nu) * Sig2Up(mu, nu)


def TildeSig2Up(mu, nu):
    """GA:Ln 40-42.  (1/2)(tildeUp[mu].sigUp[nu] - tildeUp[nu].sigUp[mu])."""
    return Rational(1, 2) * (sigmaTildeUp(mu) * sigUp[nu] - sigmaTildeUp(nu) * sigUp[mu])


def _signature(perm):
    """Signature of a permutation given as a tuple of the actual values (GA uses
    Signature[p] on the permuted value-triple, GA:Ln 47-48 / 55-56)."""
    p = list(perm)
    n = len(p)
    s = 1
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                s = -s
    return s


def Sig3Up(mu, nu, rho):
    """GA:Ln 44-50.  (1/6) sum over perms p of {mu,nu,rho} of
    Signature[p] * sigUp[p1].tildeUp[p2].sigUp[p3]."""
    tot = zeros(16, 16)
    for p in permutations([mu, nu, rho]):
        tot += _signature(p) * (sigUp[p[0]] * sigmaTildeUp(p[1]) * sigUp[p[2]])
    return Rational(1, 6) * tot


def Sig3UpT(mu, nu, rho):
    """GA:Ln 52-58.  (1/6) sum over perms p of {mu,nu,rho} of
    Signature[p] * tildeUp[p1].sigUp[p2].tildeUp[p3]."""
    tot = zeros(16, 16)
    for p in permutations([mu, nu, rho]):
        tot += _signature(p) * (sigmaTildeUp(p[0]) * sigUp[p[1]] * sigmaTildeUp(p[2]))
    return Rational(1, 6) * tot


def MixedLeft(mu, nu, rho, xi):
    """GA:Ln 60-61.  SigmaTilde[mu] . Sig3Up[nu,rho,xi]."""
    return sigmaTilde(mu) * Sig3Up(nu, rho, xi)


def MixedRight(nu, rho, xi, mu):
    """GA:Ln 62-63.  Sig3UpT[nu,rho,xi] . Sigma[mu]  (unused by L/R but mirrored)."""
    return Sig3UpT(nu, rho, xi) * sigma(mu)


# ---------------------------------------------------------------------------
# Gauge-fix & bosonic ordering  --  GA:Ln 65-107
# These ordering choices are FIXED by the Mathematica source (shared-convention).
# ---------------------------------------------------------------------------
# HPairsAll: {mu,nu} with 0<=mu<=9, mu<=nu<=9  (55 symmetric pairs).  GA:Ln 67-69
HPairsAll = [(mu, nu) for mu in range(10) for nu in range(mu, 10)]
# BPairsAll: {mu,nu} with mu<nu  (45 antisymmetric pairs).  GA:Ln 70-72
BPairsAll = [(mu, nu) for mu in range(10) for nu in range(mu + 1, 10)]

# Gauge fix: drop every pair whose first index is 0.  GA:Ln 74-77
HPairsGF = [p for p in HPairsAll if p[0] != 0]    # -> 45
BPairsGF = [p for p in BPairsAll if p[0] != 0]    # -> 36

# Custom H order: {1,1} first, then the rest in their natural (lexicographic) order.
# GA:Ln 79-80   (Prepend[DeleteCases[HPairsGF,{1,1}],{1,1}])
HOrder = [(1, 1)] + [p for p in HPairsGF if p != (1, 1)]
# B order: default lexicographic.  GA:Ln 81-83
BOrder = list(BPairsGF)

assert len(HOrder) == 45, len(HOrder)
assert len(BOrder) == 36, len(BOrder)

# Index maps (1-based to mirror Mathematica Range).  GA:Ln 87-92
HIndex = {key: i + 1 for i, key in enumerate(HOrder)}
BIndex = {key: i + 1 for i, key in enumerate(BOrder)}

betaScale = 1                                       # GA:Ln 95
colPhi = len(HOrder) + len(BOrder) + 1              # = 45+36+1 = 82.  GA:Ln 107
assert colPhi == 82, colPhi


def addH(row, key, val):
    """GA:Ln 98-100.  row[[HIndex[key]]] += val if key in HIndex (1-based -> 0-based)."""
    if tuple(key) in HIndex:
        row[HIndex[tuple(key)] - 1] += val


def addB(row, key, val):
    """GA:Ln 102-105.  row[[Length[HOrder] + BIndex[key]]] += val if key in BIndex."""
    if tuple(key) in BIndex:
        row[len(HOrder) + BIndex[tuple(key)] - 1] += val


# ---------------------------------------------------------------------------
# Fermion indexing (psi then chi)  --  GA:Ln 109-115
# 1-based Mathematica indices; we store 0-based by subtracting 1 at use sites.
# ---------------------------------------------------------------------------
spinorCount = 16                                   # GA:Ln 110


def psiRow(mu, b):
    """GA:Ln 111.  psiRow[mu,b] = mu*16 + b   (1..160)."""
    return mu * spinorCount + b


def chiRow(b):
    """GA:Ln 112.  chiRow[b] = 160 + b   (161..176)."""
    return 10 * spinorCount + b


def colPsi(mu, b):
    return psiRow(mu, b)                            # GA:Ln 114


def colChi(b):
    return chiRow(b)                                # GA:Ln 115


# ---------------------------------------------------------------------------
# R[a]: bosons -> fermions  --  GA:Ln 117-167
# Genuinely-independent: re-derived from the three additive terms (h, B, A_[3]).
# ---------------------------------------------------------------------------
def rowForPsi(mu, dotb, a):
    """GA:Ln 118-145.  One psi-row of R[a] (length colPhi=82).
    dotb,a are 1-based spinor indices (matrix entry [[dotb,a]])."""
    row = [sp.Integer(0)] * colPhi
    # Term 1: h_{rho mu} term, -(1/2) tildeSig2Up[0,rho] d_0 h_{rho mu}.  GA:Ln 121-125
    for rho in range(10):
        coeff = Rational(-1, 2) * TildeSig2Up(0, rho)[dotb - 1, a - 1]
        keyH = tuple(sorted((rho, mu)))            # h symmetric -> sort.  GA:Ln 124
        addH(row, keyH, coeff)
    # Term 2: B_{rho mu} term, -(1/2) tildeSig2Up[0,rho] d_0 B_{rho mu}.  GA:Ln 126-135
    for rho in range(10):
        if rho != mu:
            coeffB = Rational(-1, 2) * TildeSig2Up(0, rho)[dotb - 1, a - 1]
            # B antisymmetric: canonical pair + sign.  GA:Ln 130-133
            if rho < mu:
                keyB, signB = (rho, mu), 1
            else:
                keyB, signB = (mu, rho), -1
            addB(row, keyB, betaScale * signB * coeffB)
    # Term 3: A_[3] term, +(1/16) MixedLeft[mu,0,rho,xi].  GA:Ln 136-144
    for rho in range(10):
        for xi in range(rho + 1, 10):
            keyB = (rho, xi)                        # already canonical (rho<xi)
            coeffA = Rational(1, 16) * MixedLeft(mu, 0, rho, xi)[dotb - 1, a - 1]
            addB(row, keyB, betaScale * coeffA)
    return row


def rowForChi(b, a):
    """GA:Ln 148-155.  One chi-row of R[a] (length colPhi=82)."""
    row = [sp.Integer(0)] * colPhi
    # Phi column gets (1/Sqrt[8]) sigUp[0][[b,a]].  GA:Ln 150
    row[colPhi - 1] += (1 / sqrt(8)) * sigUp[0][b - 1, a - 1]
    # B columns get -(1/8) Sig3Up[0,rho,xi][[b,a]].  GA:Ln 151-154
    for rho in range(10):
        for xi in range(rho + 1, 10):
            keyB = (rho, xi)
            addB(row, keyB, betaScale * Rational(-1, 8) * Sig3Up(0, rho, xi)[b - 1, a - 1])
    return row


def R(a):
    """GA:Ln 157-167.  R[a]: 176 x 82.  psi rows (160) then chi rows (16)."""
    rows = []
    for mu in range(10):                            # GA:Ln 161-163
        for dotb in range(1, 17):
            rows.append(rowForPsi(mu, dotb, a))
    for b in range(1, 17):                          # GA:Ln 164
        rows.append(rowForChi(b, a))
    return Matrix(rows)


# ---------------------------------------------------------------------------
# L[a]: fermions -> bosons  --  GA:Ln 169-214
# Genuinely-independent: re-derived from the coeffRow / Sig2 row assembly.
# ---------------------------------------------------------------------------
def coeffRow(mu, a):
    """GA:Ln 170.  Sigma[mu][[a, All]]  -> the a-th row of Sigma[mu]."""
    M = sigma(mu)
    return [M[a - 1, b] for b in range(16)]


def coeffRow2(mu, nu, a):
    """GA:Ln 171-172.  Sig2[mu,nu][[a, All]]."""
    M = Sig2(mu, nu)
    return [M[a - 1, b] for b in range(16)]


def rowForH(mu, nu, a):
    """GA:Ln 182-189.  One H-row of L[a] (length 176)."""
    row = [sp.Integer(0)] * 176
    cmu = coeffRow(mu, a)
    cnu = coeffRow(nu, a)
    for b in range(1, 17):                          # b is 1-based spinor index
        cmu_b = cmu[b - 1]
        cnu_b = cnu[b - 1]
        if cmu_b != 0:
            row[colPsi(nu, b) - 1] += cmu_b         # GA:Ln 186
        if cnu_b != 0:
            row[colPsi(mu, b) - 1] += cnu_b         # GA:Ln 187
    return row


def rowForB(mu, nu, a):
    """GA:Ln 191-201.  One B-row of L[a] (length 176)."""
    row = [sp.Integer(0)] * 176
    cmu = coeffRow(mu, a)
    cnu = coeffRow(nu, a)
    c2 = coeffRow2(mu, nu, a)
    for b in range(1, 17):
        cmu_b = cmu[b - 1]
        cnu_b = cnu[b - 1]
        if cmu_b != 0:
            row[colPsi(nu, b) - 1] += cmu_b         # GA:Ln 196
        if cnu_b != 0:
            row[colPsi(mu, b) - 1] += -cnu_b        # GA:Ln 197 (note the minus)
    for b in range(1, 17):
        coeff = c2[b - 1]
        if coeff != 0:
            row[colChi(b) - 1] += coeff             # GA:Ln 199-200
    return [betaScale * x for x in row]             # GA:Ln 201


def PhiRowVector(a):
    """GA:Ln 207-209.  Single Phi row of L[a]: Sqrt[8] in the chi[a] column."""
    row = [sp.Integer(0)] * 176
    row[colChi(a) - 1] = sqrt(8)
    return row


def L(a):
    """GA:Ln 211-214.  L[a]: 82 x 176.  H rows (45), B rows (36), Phi row (1)."""
    rows = []
    for (mu, nu) in HOrder:                         # GA:Ln 203 / 212
        rows.append(rowForH(mu, nu, a))
    for (mu, nu) in BOrder:                         # GA:Ln 205 / 213
        rows.append(rowForB(mu, nu, a))
    rows.append(PhiRowVector(a))                    # GA:Ln 213
    return Matrix(rows)


# ---------------------------------------------------------------------------
# Driver / verification
# ---------------------------------------------------------------------------
def to_float_matrix(M):
    """Exact sympy matrix -> nested python floats (for JSON comparison)."""
    return [[complex(sp.N(M[i, j])) for j in range(M.cols)] for i in range(M.rows)]


def nearest_exact_str(x):
    """Render a float as the canonical exact token it is closest to."""
    sqrt8 = 2 ** 1.5
    candidates = {
        "0": 0.0, "1": 1.0, "-1": -1.0, "2": 2.0, "-2": -2.0,
        "sqrt8": sqrt8, "-sqrt8": -sqrt8,
        "1/2": 0.5, "-1/2": -0.5, "7/16": 7 / 16, "-7/16": -7 / 16,
        "1/8": 1 / 8, "-1/8": -1 / 8, "1/16": 1 / 16, "-1/16": -1 / 16,
        "1/sqrt8": 1 / sqrt8, "-1/sqrt8": -1 / sqrt8,
    }
    best, bestd = None, None
    for name, v in candidates.items():
        d = abs(x - v)
        if bestd is None or d < bestd:
            best, bestd = name, d
    return best, bestd


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "data", "tendim_10d_lr.json")
    json_path = os.path.relpath(json_path) if not os.path.isabs(json_path) else json_path

    print("=" * 78)
    print("EXACT (symbolic) Garden Algebra L/R evaluator")
    print("Source: /tmp/super-sym/Garden Algebra  (line citations 'GA:Ln' in code)")
    print("=" * 78)

    with open(json_path) as f:
        data = json.load(f)
    Ljson = data["L"]   # 16 x 82 x 176
    Rjson = data["R"]   # 16 x 176 x 82
    print(f"\nLoaded JSON: {json_path}")
    print(f"  nb={data['nb']} nf={data['nf']} n={data['n']}")
    print(f"  L shape {len(Ljson)}x{len(Ljson[0])}x{len(Ljson[0][0])}, "
          f"R shape {len(Rjson)}x{len(Rjson[0])}x{len(Rjson[0][0])}")

    # ---- Build all 16 exact L and R ----
    print("\nBuilding 16 exact L[a] (82x176) and R[a] (176x82) ...")
    Lex = [L(a) for a in range(1, 17)]
    Rex = [R(a) for a in range(1, 17)]
    print("  done.")

    # ---- Collect exact value sets ----
    Lvals, Rvals = set(), set()
    for M in Lex:
        for v in M:
            Lvals.add(sp.nsimplify(v))
    for M in Rex:
        for v in M:
            Rvals.add(sp.nsimplify(v))

    def tok(v):
        return sp.sstr(sp.simplify(v))

    Ltokens = sorted({tok(v) for v in Lvals}, key=lambda s: (len(s), s))
    Rtokens = sorted({tok(v) for v in Rvals}, key=lambda s: (len(s), s))

    # ---- Entrywise comparison to JSON ----
    TOL = 1e-9
    max_disc = 0.0
    mismatches = 0
    where = None
    for a in range(16):
        Mf = to_float_matrix(Lex[a])
        ref = Ljson[a]
        for i in range(82):
            for j in range(176):
                d = abs(Mf[i][j] - ref[i][j])
                if d > max_disc:
                    max_disc = d; where = ("L", a, i, j, Mf[i][j], ref[i][j])
                if d > TOL:
                    mismatches += 1
    for a in range(16):
        Mf = to_float_matrix(Rex[a])
        ref = Rjson[a]
        for i in range(176):
            for j in range(82):
                d = abs(Mf[i][j] - ref[i][j])
                if d > max_disc:
                    max_disc = d; where = ("R", a, i, j, Mf[i][j], ref[i][j])
                if d > TOL:
                    mismatches += 1

    # ---- Exact bosonic Garden relation: L_I R_J + L_J R_I = 2 delta_IJ I_82 ----
    print("\nVerifying EXACT bosonic relation L_I R_J + L_J R_I = 2 delta_IJ I_82 ...")
    I82 = eye(82)
    boson_ok = True
    boson_max_nonzero_pairs = 0
    diag_pairs = []
    for i in range(16):
        for j in range(i, 16):
            lhs = Lex[i] * Rex[j] + Lex[j] * Rex[i]
            rhs = (2 * I82) if i == j else zeros(82, 82)
            diff = sp.simplify(lhs - rhs)
            if not diff.is_zero_matrix:
                boson_ok = False
                boson_max_nonzero_pairs += 1
                diag_pairs.append((i + 1, j + 1))
    print(f"  exact bosonic residual zero for all 136 (i<=j) pairs: {boson_ok}")
    if not boson_ok:
        print(f"  NONZERO pairs: {diag_pairs[:10]}{' ...' if len(diag_pairs)>10 else ''}")

    # ---- Exact fermionic remnant E_IJ for a sample pair ----
    # R_I L_J + R_J L_I = 2 delta_IJ I_176 + 2 E_IJ   (per JSON 'relations.fermionic')
    print("\nComputing EXACT fermionic remnant E_IJ = (R_I L_J + R_J L_I - 2 delta_IJ I_176)/2 ...")
    I176 = eye(176)
    # i=j=1 (diagonal) and i=1,j=2 (off-diagonal) as representative samples
    samples = [(0, 0), (0, 1)]
    fermi_report = []
    for (i, j) in samples:
        lhs = Rex[i] * Lex[j] + Rex[j] * Lex[i]
        delta = 2 if i == j else 0
        E = sp.simplify((lhs - delta * I176) / 2)
        nz = sum(1 for v in E if v != 0)
        Evals = sorted({sp.sstr(sp.simplify(v)) for v in E if v != 0}, key=lambda s: (len(s), s))
        fermi_report.append((i + 1, j + 1, E.is_zero_matrix, nz, Evals))

    # ===================== REPORT =====================
    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)

    print("\n[1] EXACT vs JSON entrywise comparison (tol = 1e-9):")
    print(f"    max entrywise discrepancy : {max_disc:.3e}")
    print(f"    mismatch count (> tol)    : {mismatches}")
    if where is not None:
        kind, a, i, j, got, ref = where
        print(f"    location of max disc      : {kind}[{a+1}][{i}][{j}] "
              f"exact={got!r} json={ref!r}")
    print(f"    => MATCH: {'YES' if mismatches == 0 else 'NO'}")

    print("\n[2] EXACT bosonic Garden relation L_I R_J + L_J R_I = 2 delta_IJ I_82:")
    print(f"    exactly satisfied (residual identically 0): {boson_ok}")
    print("    (this is EXACT sympy zero, not a float residual)")

    print("\n[3] EXACT fermionic remnant E_IJ (samples):")
    for (i, j, isz, nz, vals) in fermi_report:
        print(f"    E[{i},{j}]: zero-matrix={isz}  nonzero-entries={nz}  "
              f"distinct nonzero values={vals}")

    print("\n[4] Canonical exact value sets (from the symbolic matrices):")
    print(f"    L value-set ({len(Ltokens)}): {Ltokens}")
    print(f"    R value-set ({len(Rtokens)}): {Rtokens}")
    # cross-check each token maps to an expected canonical name via float distance
    print("\n    Token -> nearest canonical (float check):")
    for grp, toks in (("L", Ltokens), ("R", Rtokens)):
        for t in toks:
            fv = float(sp.N(sp.sympify(t)))
            name, dist = nearest_exact_str(fv)
            print(f"      {grp}: {t:>12}  ~  {name:>10}  (float={fv:+.6f}, d={dist:.1e})")

    print("\n[5] Independence breakdown (shared-convention vs independent):")
    print("    SHARED-CONVENTION (fixed by the Mathematica source; a faithful")
    print("    re-derivation MUST match these, so agreement here is NOT independent")
    print("    corroboration):")
    print("      - sigma Kronecker table sigUp[0..9]           GA:Ln 9-25")
    print("      - etasign / sigma / tilde sign conventions    GA:Ln 27-31")
    print("      - gauge-fix (drop first-index-0) + H/B order   GA:Ln 74-83")
    print("      - {1,1}-first H ordering, lexicographic B      GA:Ln 80-81")
    print("      - psi-then-chi fermion indexing                GA:Ln 110-115")
    print("      - colPhi=82 column layout (H,B,Phi)            GA:Ln 107")
    print("    GENUINELY INDEPENDENT (re-derived symbolically; a mis-transcribed")
    print("    coefficient WOULD surface as a mismatch):")
    print("      - Sig2/TildeSig2/Sig3/MixedLeft contractions  GA:Ln 33-61")
    print("      - R psi-row 3-term assembly (-1/2, -1/2, 1/16) GA:Ln 121-144")
    print("      - R chi-row (1/Sqrt8 Phi, -1/8 B)              GA:Ln 150-154")
    print("      - L H/B-row Sig2 assembly + minus sign         GA:Ln 186-200")
    print("      - L Phi row Sqrt8                               GA:Ln 208")
    print("    A clean match on [1] is strongest evidence precisely on the")
    print("    GENUINELY-INDEPENDENT pieces, since those are where the two")
    print("    derivations could have diverged.")

    print("\n" + "=" * 78)
    overall = (mismatches == 0) and boson_ok
    print(f"OVERALL: exact evaluation {'MATCHES' if mismatches==0 else 'DIFFERS FROM'} "
          f"the JSON; bosonic relation {'holds exactly' if boson_ok else 'FAILS'}.")
    print("=" * 78)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
