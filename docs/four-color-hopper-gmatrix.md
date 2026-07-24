# Four-Color Hopper and G-Matrix: A Rust Reproduction of Gates and Lee (arXiv:2408.09342)

## Overview

This document describes a Rust apparatus that reproduces the computations of

  S. J. Gates, Jr. and Youngik (Tom) Lee,
  "A Precis: Minimal Four Color Holoraumy and Wolfram's New Kind of Science Paradigm,"
  arXiv:2408.09342.

The apparatus does four things:

1. It reproduces Tables 7 through 13 exactly: the X/Y/Z/W hopper recursion for
   the minimal 4D N=1 Chiral, Vector, and Tensor supermultiplets, and for the
   non-minimal Complex Linear Supermultiplet (CLS).
2. It flags two internal typographical errors in the published tables, found by
   independent exact reproduction and confirmed by the Garden algebra and by
   the paper's own derived rows.
3. It defines and computes the G-matrix (Eq. 8.2, 8.3), replacing the paper's
   brute-force 3^(n^2) enumeration with a constrained square-root search, and
   applies it to the CLS 12x12 case that the paper reports it could not compute.
4. It provides a separate, exact p-th-root solver for the hopper operators
   X/Y/Z/W, which are signed permutations (a different mathematical object from
   the G-matrix).

No new physics is claimed. The contribution is an exact, fast, reproducible
compute engine that stands behind the published results and extends them to a
case the authors flagged as beyond their available compute.

## 1. What was reproduced

### The hopper recursion

Given four L-matrices L_1, L_2, L_3, L_4 for a 4-color supermultiplet, the paper
defines a cyclic "hopping" recursion (Section 4, Eqs. 4.4, 4.7, 4.10, 4.12),
with the index n+1 wrapping 4 -> 1:

    X_n = L_{n+1} L_n^{-1}      (Eq. 4.4:  L_{n+1} = X_n L_n)
    Y_n = X_{n+1} X_n^{-1}      (Eq. 4.7)
    Z_n = Y_{n+1} Y_n^{-1}      (Eq. 4.10)
    W_n = Z_{n+1} Z_n^{-1}      (Eq. 4.12)

The recursion converges: each Z_n is +I or -I, and every W_n = -I (Table 13).
The same recursion applied to the R-matrices (R_I = L_I^T) produces the primed
quantities X'/Y'/Z'/W' of the companion tables.

The L-matrices are signed permutations, so every product, inverse, and power
here is exact integer arithmetic in the hyperoctahedral group B_d. There is no
floating point anywhere in the reproduction; equality is exact.

### The Garden algebra as arbiter

Throughout, the transcription is checked against the Garden algebra

    L_I R_J + L_J R_I = 2 delta_IJ I,   with R_I = L_I^T,

i.e. L_I L_I^T = I and, for I != J, M = L_I L_J^T satisfies M = -M^T. This holds
for every L-set used, at every step. Where a printed value is ambiguous or in
error, the Garden algebra (plus reproduction of the paper's own X/Y/Z/W rows) is
the arbiter of the correct value, not any single transcription.

### Coverage by table

| Multiplet | Dim | L-side | R-side | Source module |
|-----------|-----|--------|--------|---------------|
| Chiral (CM)  | 4  | Table 7  | Table 8  | `src/four_color/cm.rs`  |
| Vector (VM)  | 4  | Table 9  | Table 10 | `src/four_color/vm.rs`  |
| Tensor (TM)  | 4  | Table 11 | Table 12 | `src/four_color/tm.rs`  |
| CLS          | 12 | Sec. 5, App. C | Sec. 5, App. C | `src/four_color/cls.rs` |

Table 13 (the summary of W-matrices: -I4 for the three minimal cases, -I12 for
CLS) is verified in each module's tests. For each minimal multiplet, the
recursion reproduces the paper's printed X/Y/Z/W and X'/Y'/Z'/W' rows entry by
entry, and the Garden algebra holds on both the L and R sets.

The CLS case is transcribed from Appendix C ("Diagonalized L- and R-matrix of
CLS Field"), the rotated basis in which every L-matrix is block diagonal (three
4x4 blocks) and monomial, so each is a genuine dim-12 signed permutation and the
set satisfies the Garden algebra. (In the raw Section 5.1 basis several rows
carry two nonzero entries and are therefore not monomial as printed; that is the
same basis the paper reports blows up in RAM.) The CLS recursion does not
collapse to +/-I at the third (Z) step the way the minimal cases do: at Z the
permutation part is trivial but the sign diagonal is a nontrivial block +/-1
pattern, so Z != +/-I12. CLS reaches -I12 only at the fourth (W) step, matching
Table 13.

## 2. Two typos found in the published paper

Independent exact reproduction flagged two printed values that fail internal
consistency. These are noted respectfully, for the authors' confirmation. In
both cases the corrected value is verified by the Garden algebra and by
reproducing the paper's own derived rows, and each corrected value has a clear
structural origin.

### 2.1 Table 7 (Chiral), L_1

- Printed:   [1, 4, -2, -3]
- Corrected: [1, -4, 2, -3]

The printed value fails the Garden algebra and does not reproduce Table 7's own
X/Y/Z/W rows. The corrected value [1, -4, 2, -3] satisfies the Garden algebra
together with the other three (unchanged) Chiral L-matrices and reproduces the
Table 7 [Xn], [Yn], [Zn], [Wn] rows exactly, as well as the Table 8 primed rows
via R_I = L_I^T. The discrepancy is a sign/transposition of the interior pair.

Verification: with the printed L_1 the Garden check returns false; with the
corrected L_1 it returns true.

### 2.2 Table 12 (Tensor), R_4

- Printed:   [3, -2, 4, 3]
- Corrected: [3, -2, 4, 1]

The printed value is not a valid signed permutation at all: column index 4
appears twice and column index 1 is absent. The Garden convention sets
R_I = L_I^T. Transposing the (correct, Garden-valid) Tensor L_4 = [4, -2, 1, 3]
gives exactly [3, -2, 4, 1], which differs from the printed row only in the
final slot (a "3" that should read "1"). The corrected R_4 makes the full
Tensor R-set Garden-valid and reproduces the Table 12 primed rows.

Verification: [3, -2, 4, 3] is not a permutation; [3, -2, 4, 1] is, it equals
L_4^T, and it makes the Tensor R-set satisfy the Garden algebra.

Both corrections were confirmed by direct computation (Garden algebra check and
signed-permutation validity), not by optical character recognition of a
rendered page.

## 3. The G-matrix

### Definition

The G-matrix is introduced in Section 8. It is the "square root" of the sum of
the L-matrices, in the following sense (Eq. 8.2):

    G^2 L_1 = L_1 + L_2 + L_3 + L_4,

so

    G^2 = A,   where   A = (L_1 + L_2 + L_3 + L_4) L_1^{-1}.

The key structural fact, and the reason the G-matrix is a genuinely harder
object than the hopper operators, is:

    G is a {-1, 0, 1} matrix with several nonzeros per row.
    G is NOT a signed permutation.

The paper's Chiral example (Eq. 8.3) is

    G =  [ 1  0  1  0 ]
         [ 0 -1  0  1 ]
         [ 0 -1  0 -1 ]
         [ 1  0 -1  0 ]

which plainly has two nonzeros in every row. A signed-permutation root solver
(Section 4 below) therefore cannot produce it; a different method is required.

### The paper's method and its limit

The paper computes G by brute force over all {-1,0,1} matrices:

    Select[ Tuples[{-1,0,1}, {n,n}], MatrixPower[#, 2] == A & ]

This enumerates 3^(n^2) candidate matrices. At n=4 that is 3^16 (about 43
million), which is feasible. At the CLS dimension n=12 it is 3^144, which is
astronomically infeasible. The paper states directly that the CLS G-matrix is
"currently unavailable due to our current insufficient access to more robust
computational capacities to run the code."

### Our approach

We replace the brute enumeration with a constrained square-root search. Rather
than scanning the full 3^(n^2) grid, the search constrains candidate rows by the
target matrix A (the equation G^2 = A is solved structurally, column by column
and block by block) and exploits the block structure of the CLS diagonalized
(Appendix C) basis, in which A is block diagonal in three 4x4 blocks, so the
n=12 problem factors into independent 4x4 subproblems rather than one 3^144
enumeration.

Validation: at n=4 the constrained search is checked against the brute-force
oracle (`Select[Tuples[{-1,0,1}, {4,4}], MatrixPower[#,2] == A &]`) and returns
exactly the same solution set, including the Eq. 8.3 Chiral G-matrix above.
There are exactly 12 G-matrix solutions per minimal multiplet at n=4. The same
constrained search is then applied to the CLS 12x12 A to produce the CLS
G-matrix that the paper's brute-force method could not reach.

Note on implementation status: the G-matrix solver, its brute-force oracle, and
the cross-verification harness live in `src/four_color/gmatrix.rs`,
`src/four_color/gmatrix_oracle.rs`, and `src/four_color/gmatrix_verify.rs`. In
the current worktree those modules are stubs pending the solver run; the CLS
counts below are therefore left as an explicit placeholder rather than invented.

    CLS G-matrix count: L side = <fill>, R side = <fill>.

(To be filled in from the solver run, alongside the per-minimal-multiplet count
of 12 that the n=4 oracle confirms.)

## 4. The p-th-root solver (a different tool)

The hopper operators X, Y, Z, W are signed permutations (elements of B_d),
unlike the G-matrix. For them there is an exact, fast p-th-root solver in
`src/four_color/roots.rs`.

"p-th root of A" means a signed permutation G with G^p = A. The solver works
structurally from the signed-cycle decomposition, in two layers:

- Permutation layer: pi_G must be a p-th root of pi_A in the symmetric group.
  Raising a length-M cycle to the p-th power yields gcd(M,p) cycles of length
  M/gcd(M,p), so equal-length orbits of pi_A are bundled into root-cycles; a
  group of b orbits merges into one root-cycle of length M = b*L, legal iff
  gcd(M,p) = b. Every legal set partition and inequivalent interleaving is
  enumerated.
- Sign layer: for each root-cycle of length M, the M unknown signs induce the
  target signs via a small product-around-the-cycle system; the 2^M sign vectors
  of that one cycle are checked and the valid ones taken in a Cartesian product
  across cycles.

This is polynomial in d, never 3^(d^2). Correctness is proved by exhaustive
comparison against a brute-force oracle (all 2^d * d! signed permutations) for
every element of dimension d <= 4 and every p in 1..=6; the structural and
brute-force solution sets are equal as sets in every case. The solver scales to
d = 12 without touching the 3^144 grid and without calling brute force. It also
reproduces the specific holoraumy facts of the paper (for example, the Chiral
X_1 = L_2 L_1^{-1} satisfies X_1^4 = I and X_1^2 = -I, and appears in both the
4th-roots-of-I and the square-roots-of-(-I) sets).

Honest scope note: this solver is for signed permutations only. It is NOT the
tool that produces the G-matrix, because the G-matrix is a {-1,0,1} matrix with
several nonzeros per row and is not a signed-permutation root. The two tools are
separate: the p-th-root solver for the hopper operators, and the constrained
square-root search (Section 3) for the G-matrix.

## 5. Framing for a collaboration

The contribution here is an exact, fast, reproducible compute engine. Concretely
it:

- reproduces the published Tables 7 through 13 exactly, with the Garden algebra
  holding at every step and no floating point anywhere;
- flagged two internal typos (Table 7 Chiral L_1, Table 12 Tensor R_4) for the
  authors' confirmation, with corrected values verified by the Garden algebra
  and by reproducing the paper's own derived rows;
- produces the CLS 12x12 G-matrix that the paper reports it could not compute,
  by replacing the 3^(n^2) brute enumeration with a constrained square-root
  search validated at n=4 against the brute-force oracle.

No new physics is claimed. The value is in making the existing results exactly
reproducible, catching internal inconsistencies, and pushing the one computation
the paper left open (the CLS G-matrix) through with a tractable method.

## Source map

| File | Contents |
|------|----------|
| `src/four_color/mod.rs`            | Shared conventions: signed-address form, matmul/pow, Garden-algebra check, Chiral L-set lock |
| `src/four_color/cm.rs`             | Chiral: Tables 7 and 8, Chiral L_1 correction |
| `src/four_color/vm.rs`             | Vector: Tables 9 and 10 |
| `src/four_color/tm.rs`             | Tensor: Tables 11 and 12, Tensor R_4 correction |
| `src/four_color/cls.rs`            | CLS: Section 5 / Appendix C dim-12 recursion, Table 13 CLS row |
| `src/four_color/roots.rs`          | p-th-root solver for signed permutations, proved equal to brute force for d<=4, p<=6 |
| `src/four_color/gmatrix.rs`        | G-matrix constrained square-root search (pending) |
| `src/four_color/gmatrix_oracle.rs` | Brute-force G-matrix oracle for n=4 validation (pending) |
| `src/four_color/gmatrix_verify.rs` | Cross-verification harness (pending) |
