# Four-Color Apparatus: Source Verification Against the arXiv LaTeX E-Print

## Scope

This records the re-verification of the four-color signed-permutation apparatus
that reproduces Gates and Lee, arXiv:2408.09342, Tables 7-13. The apparatus
covers three supermultiplets:

- CM (chiral), Tables 7 and 8
- VM (vector), Tables 9 and 10
- TM (tensor), Tables 11 and 12
- Table 13 (recursion convergence: Z_n in {+I, -I}, W_n = -I4)

The reference used is the authoritative arXiv LaTeX e-print for 2408.09342
(`https://arxiv.org/e-print/2408.09342`), file `AdnkWolfram.tex`. This is the
LaTeX source of the published paper, not a rendered PDF or an OCR pass, so the
values below are read directly from the author-supplied source.

## Result

Every L-matrix and R-matrix of the CM, VM, and TM supermultiplets, and every
derived X/Y/Z/W and primed X'/Y'/Z'/W' row generated from them by the hopping
recursion (paper Eqs 4.4, 4.7, 4.10, 4.12), reproduces the paper exactly, with
exactly two exceptions. Both exceptions are genuine typographical errors in the
published paper: each is present verbatim in the `AdnkWolfram.tex` source, so
neither is an OCR or transcription artifact introduced downstream.

For both, the printed value was rejected and the internally consistent value was
used, on the basis of two convention-free tests:

1. The corrected value satisfies the Garden algebra of the supermultiplet.
2. The corrected value reproduces the paper's own derived X/Y/Z/W rows for that
   table byte-for-byte under the recursion.

The printed values fail both tests. All arithmetic used to confirm this was
convention-free 4x4 signed-matrix arithmetic (no reliance on any sign or index
convention that could mask the discrepancy).

## Typo 1: Table 7, Chiral (CM) L1

- Paper prints (LaTeX `AdnkWolfram.tex`): `\vev{14\bar2\bar3}` = `[1, 4, -2, -3]`
- Corrected, used in code: `\vev{1\bar42\bar3}` = `[1, -4, 2, -3]`

The printed `[1, 4, -2, -3]` does not satisfy the CM Garden algebra and does not
reproduce Table 7's own `[Xn]`, `[Yn]`, `[Zn]`, `[Wn]` rows. The corrected
`[1, -4, 2, -3]` satisfies the Garden algebra and, under the recursion, produces
the Table 7 X/Y/Z/W rows exactly. It is the value in `super::cm_l_matrices()`.

The other three CM L entries (n = 2, 3, 4) match the paper as printed.

Confirmation: Garden-algebra check plus exact reproduction of the paper's own
derived rows, via convention-free 4x4 matrix arithmetic.

## Typo 2: Table 12, Tensor (TM) R4

- Paper prints (LaTeX `AdnkWolfram.tex`): `\vev{3\bar243}` = `[3, -2, 4, 3]`
- Corrected, used in code: `\vev{3\bar241}` = `[3, -2, 4, 1]` (= L4 transpose)

The printed `[3, -2, 4, 3]` is not even a valid signed permutation: column 3
appears twice and column 1 is absent. Under the module convention R_I = L_I^T,
the fourth R-matrix must be L4 transpose. Transposing L4 = `[4, -2, 1, 3]` yields
`[3, -2, 4, 1]`, which matches the paper's `[3, -2, 4, _]` in the first three
slots and fixes the mistyped final slot (`3` should read `1`). The corrected
`[3, -2, 4, 1]` satisfies the TM Garden algebra; the printed value does not.

Confirmation: Garden-algebra check plus the R_I = L_I^T transpose relation, via
convention-free 4x4 matrix arithmetic.

## Everything Else

All remaining CM, VM, and TM L and R matrices, and all of their derived X/Y/Z/W
and X'/Y'/Z'/W' rows, and the Table 13 convergence pattern, reproduce the
`AdnkWolfram.tex` source exactly with no changes. No other value in the apparatus
was adjusted.
