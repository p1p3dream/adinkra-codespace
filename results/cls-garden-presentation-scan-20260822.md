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
also satisfy the transpose-Garden relation entrywise (9/9), but this is a
COROLLARY of bucket 1, not an additional independent filter: once all four
conjugates are signed permutations they are orthogonal (C^-1 = C^T), the
similarity-preserved squares C_i^2 = -I (colors 2-4; verified skew on the
L's) force C_i^T = -C_i, and the preserved anticommutation among colors 2-4
(verified on the L's; pairs with color 1 cannot anticommute since L_1 = I)
gives C_i C_j^T + C_j C_i^T = -(C_i C_j + C_j C_i) = 0. The equivalent
centralizer statement, G^T G commutes with every L_i L_j^T for these nine,
was independently confirmed by the external review. Useful structural
characterization of bucket 1; carries no filtering power beyond signed-
permutation-ness itself.

Taxonomy constancy is false, as the review asserted. Counts of keys whose
stored representatives change classification, by explicit definition:

| definition of the classification | mixed keys |
|---|---|
| coarse bucket (3-way) | 34 |
| full four-color signed-permutation flags | 50 |
| combined (bucket, signed-perm flags) | 55 |
| number of integer colors | 182 |
| full four-color integer flags | 188 |

The review's original 55 is the combined definition. An earlier run of this
scan reported 119/126 for the two finer rows: that was a bug (classify()
short-circuited at the first non-integer color, so the flag statistics were
computed on truncated color lists); fixed in the correction commit, and all
five rows now reproduce the review's table exactly. Every definition is far
above zero, so a taxonomy key does not determine the conjugation property,
and population extrapolation from the 9 is unjustified.

## What dies, what survives

Dies:
- The 33,696 population claim (extrapolated by assuming the property is
  constant on taxonomy classes; it is not).
- Any framing that the scan computed holoraumy. It tested conjugation and
  signed-permutation structure. Preservation of algebraic relations under
  similarity is tautological and carries no information.
- Any use of C_I^2 = -I for colors 2-4 on conjugates as evidence: it is
  similarity-forced. Color 1 instead has C_1^2 = +I.

Survives, as a narrow exact fact: nine of the 4,077 stored matrices
conjugate all four L_I into signed-permutation matrices (which then
necessarily satisfy the transpose-Garden algebra, see the corollary above).
Whether additional matrices in those nine taxonomy classes do the same is
unknown until class members beyond the stored representatives are tested
(which needs the solution-dump hook and a targeted rerun, not a new census).

## Next steps (per the review, unchanged)

1. Compute genuine holoraumy for the three presentations produced by the nine
   qualifying matrices.
2. Equivalence classes of the three presentations under signed node
   relabelings, color
   permutations, and optional color sign rescalings (intertwiner S with
   S C^a_I S^-1 = epsilon_I C^b_{sigma(I)}).
3. Targeted class-member tests before attaching any population count.

Reproduce: `python3 scripts/lever_a/garden_presentation_scan.py`

## Holoraumy gadget cross-matrix is a basis artifact; one equivalence class (2026-08-22)

A follow-up session computed the holoraumy gadget matrix over
{CLS, T1, T2, T3}, where T1/T2/T3 are the 3 distinct conjugated quadruples
produced by the 9 bucket-1 G-matrices (verified: the 9 collapse 7/1/1 by
commutant collisions, G_j^-1 G_i commuting with all four L_I within each
group):

```
            CLS   T1(7G)  T2(1G)  T3(1G)
CLS           3       0       0       0
T1(7G)        0       3       1       1
T2(1G)        0       1       3       0
T3(1G)        0       1       0       3
```

The session read the cross entries as "CLS holoraumy-orthogonal to its
conjugates" and "targets sharing one irreducible component". Both readings
are wrong, and the matrix refutes itself: every target is a conjugate of
CLS by construction (C_I = G L_I G^-1), so all four rows present ONE
isomorphism class. An isomorphism-invariant comparison returns the
self-value (3) on every entry; the observed 0s and 1s therefore prove the
cross-gadget is not an invariant. The mechanism is in the formula
(src/holoraumy.rs): gadget(a,b) = -2/(N(N-1)dmin) sum Tr(Vtilde^a_IJ
Vtilde^b_IJ) multiplies a matrix from a's basis by one from b's basis, and
such traces move under independent conjugation of the two factors (only
products within a single rep conjugate through; same trap family as the
transpose-Garden discussion above).

Direct certificate (scripts/lever_a/garden_target_equivalence_check.py,
all checks PASS): a signed-node-relabeling witness S maps T2 exactly onto T1,
and under that relabeling gadget(T1, .) moves from 1 to 3.

Exact equivalence answer (the external review's step 2): CLS, T1, T2, T3 are
all equivalent under signed node relabeling alone (sigma = identity and
epsilon = (+1,+1,+1,+1), so no color permutation or color sign rescaling).
Witnesses in address form, each verified entrywise
(S L_I S^-1 = T_I for all I):

- CLS -> T1: (1, 4, 2, 3, 5, -7, 8, -6, 9, -11, -12, 10)
- CLS -> T2: (5, -7, 6, -8, 1, 4, -3, -2, 9, -11, -12, 10)
- CLS -> T3: (5, -8, -7, 6, 9, 11, 12, 10, 1, -2, 4, -3)

The one-class conclusion is constructive: the three witnesses above already
prove it. The accompanying signed-monomial search is exact, not heuristic.
All four quadruples are block-diagonal 4+4+4, and every 4-node block is a
connected component of the union of its colored node graph. A signed
monomial intertwiner preserves that colored adjacency, so it must map each
whole connected block onto a whole connected block. The search enumerates all
3! block assignments, all 4! 2^4 = 384 signed permutations inside each
matched block, all allowed global color permutations, and all color sign
choices. The script now gates block connectivity explicitly.

Consequences:

- Review step 1 closes at the equivalence-class level: the 9 qualifying
  matrices produce presentations related to CLS by signed node relabelings,
  so their holoraumy tuples are conjugate by the same relabelings, not
  generally entrywise identical in the displayed bases. Exactly one
  holoraumy equivalence class occurs among the stored Garden presentations.
- Block plans: T1 = per-block color permutations (234)/(243)/(243) of the
  original blocks; T2 = (23)/(24)/(243); T3 = (24)/(243)/(34). Within a
  block family the inducible pure color permutations are exactly the even
  ones (id and the two 3-cycles); the cross-family block conjugators
  induce exactly the transpositions. Every plan is therefore realizable by
  signed node relabeling, which is why everything collapses to one class.
- The raw cross-gadget is a basis-alignment functional for every pair of
  presentations, whether the underlying representations are equivalent or
  inequivalent. A representation-level comparison requires a common basis
  convention, an explicit alignment prescription, orbit optimization, or a
  separately proved invariant construction.
- What the 9 G-matrices actually are: integral, non-monomial basis changes
  connecting the CLS monomial presentation to signed node relabelings of
  itself.
  The nontrivial census fact stays as stated above (nine of 4,077 stored
  reps conjugate into signed-permutation form); no new adinkra invariant
  content.

## Correction log

2026-08-22, same day, prompted by the external review's second pass: (a) the
review's 55 mixed keys was resolved to the combined (bucket, four-color
signed-perm flags) definition; (b) classify() no longer short-circuits at
the first non-integer color, so all flag statistics run on full four-color
lists (the earlier 119/126 figures were truncation artifacts and are
superseded by 182/188); (c) the multiplicative-square sanity gate is no
longer fail-open (it now requires exactly L_1^2 = +I, L_2^2 = L_3^2 =
L_4^2 = -I, plus anticommutation among colors 2-4 and the forced failure of
anticommutation with color 1); (d) the transpose-Garden 9/9 result was
downgraded from "genuine extra property" to corollary of bucket 1, per the
derivation above. The 9/228/3,840 headline and all bookkeeping counts were
unaffected by the bug.

2026-08-22, second correction (session follow-up): the holoraumy gadget
cross-matrix over {CLS, T1, T2, T3} was correctly computed but
misinterpreted; cross entries are basis artifacts (witness certificate:
relabeling T2 onto T1 moves gadget(T1, .) from 1 to 3). Exact equivalence
search: all four quadruples form one class under signed node relabeling alone;
review steps 1 and 2 are thereby closed. Step 3 (population testing via
the solution-dump hook) remains open. Script:
scripts/lever_a/garden_target_equivalence_check.py.

2026-08-26, mathematical-language correction: removed the false claim that
real 4x4 intertwiner spaces are at most one-dimensional by Schur. Exact
linear solves show dimension four for several block pairs. The one-class
result is unaffected because it is constructively certified by the displayed
witnesses. Exhaustiveness of the signed-monomial search now rests on the
explicitly gated connectivity of each 4-node colored block, not on commutant
dimension. Also corrected "node permutation" to "signed node relabeling,"
holoraumy "identical" to "conjugate," and the scope of raw cross-gadget basis
dependence to all representation pairs.
