# Garden-presentation scan of the stored census representatives, 2026-08-22

An external exact-arithmetic review examined the claim that census G-matrices
conjugate the CLS color matrices L_I into signed-permutation (Garden
presentation) form, and rejected the extrapolated population count of 33,696
as unsupported. This document records the independent re-derivation of every
number in that review, the corrected headline, and what the review kills.

Grep-verified context: the overstated claim exists in no deliverable file.
docs/cls-gmatrix-census-precis-for-gates.tex and the 2026-08-21 computation
writeup mention holoraumy only as future work; "33,696" appears nowhere in
docs, results, or the writeup. Nothing needs retracting from the shipped
record; this scan exists so the corrected numbers are on record before
anything further is said about the property.

## Method

scripts/lever_a/garden_presentation_scan.py, exact Fraction arithmetic, no
floating point anywhere:

- L_I transcribed from src/four_color/cls.rs (arXiv:2408.09342 Appendix C
  signed-address form), L_1 = I_12.
- Sanity anchors, all PASS: the four L_I are signed permutations; the
  transpose-Garden algebra L_i L_j^T + L_j L_i^T = 2*delta_IJ I holds
  entrywise; A_L = (L_1+L_2+L_3+L_4) L_1^-1 matches the census artifact
  results/four_color_cls_gmatrix.json entrywise, tying the color matrices to
  the census target; multiplicative squares L_i^2 = -I hold for colors
  2, 3, 4 (they are skew) and never for color 1 (L_1 = I).
- Input: all 138 shard files across the three run dirs; every stored class
  representative, 4,536 records, 4,077 distinct matrices, 1,076 taxonomy keys
  (nnz, support, ranks). All three counts reproduce the review exactly.
- Classification: C_I = G L_I G^-1 by exact rational inverse; bucket 1 if all
  four C_I are integer signed-permutation matrices, bucket 2 if all integer
  but at least one not signed-permutation, bucket 3 if any entry non-integer.

Runs in ~50 s on one core.

## Results (exact)

Over all 4,077 distinct stored matrices:

| classification | count |
|---|---|
| all four colors signed-permutation | 9 |
| integer, only partially signed-permutation | 228 |
| at least one non-integer conjugated color | 3,840 |

These reproduce the review's numbers exactly. The review's other counts also
reproduce: 4,536 records, 4,077 distinct, 1,076 keys, 712 keys with multiple
distinct stored representatives.

The nine bucket-1 matrices, by first-record provenance: 7 first stored in
item 52's shard, 2 in item 204's shard (the two count-crown items). All nine
also satisfy the transpose-Garden relation entrywise (9/9). That part is a
genuine extra property, not similarity bookkeeping: similarity by a
non-orthogonal G does not preserve transpose relations, and these G are not
orthogonal (G^2 = A with multi-entry rows). Equivalently, for these nine,
G^T G commutes with every L_i L_j^T.

Taxonomy constancy is false, as the review asserted. Counts under explicit
definitions: 34 keys have stored representatives in different buckets (same
34 under per-(item,key) dedup before cross-item comparison); 119 keys change
the number of integer colors; 126 keys change the per-color monomial flags.
The review reported 55 mixed keys without a definition; no dedup variant I
constructed yields 55. The discrepancy does not affect the conclusion: every
variant is far above zero, so a taxonomy key does not determine the
conjugation property, and population extrapolation from the 9 is unjustified.

## What dies, what survives

Dies:
- The 33,696 population claim (extrapolated by assuming the property is
  constant on taxonomy classes; it is not).
- Any framing that the scan computed holoraumy. It tested conjugation and
  signed-permutation structure. Preservation of algebraic relations under
  similarity is tautological and carries no information.
- Any use of C_I^2 = -I on conjugates as evidence: it is similarity-forced.

Survives, as a narrow exact fact: nine of the 4,077 stored matrices
conjugate all four L_I into signed-permutation matrices that satisfy the
transpose-Garden algebra entrywise. Whether additional matrices in those
nine taxonomy classes do the same is unknown until class members beyond the
stored representatives are tested (which needs the solution-dump hook and a
targeted rerun, not a new census).

## Next steps (per the review, unchanged)

1. Compute genuine holoraumy for the nine presentations.
2. Equivalence classes of the nine under node permutations, color
   permutations, and sign flips (intertwiner S with
   S^-1 C^b_I S = C^a_{sigma(I)}).
3. Targeted class-member tests before attaching any population count.

Reproduce: `python3 scripts/lever_a/garden_presentation_scan.py`
