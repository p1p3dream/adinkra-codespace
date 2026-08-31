# D21 GPU pivot and exact replay audit

Date: 2026-08-31

## Verdict

The current GPU result is not a 52-map closure certificate. It is a test-only
rank canary. No production checkpoint or report artifact exists, no actual GPU
pivot minor has been replayed over the rationals, and the retained RREF stops at
the expected rank. That stop condition hid an excess scalar-channel pivot.

Three independent tensor-convention defects explain the invalid raw witness:

1. The four covariant target slots were evaluated in one labeled order instead
   of being antisymmetrized over all 24 permutations.
2. A metric edge joining the contravariant source `H^c` to a covariant target
   slot was treated as a Kronecker coefficient. It must lower `H^c` and
   therefore contributes `eta_aa`. Only the scalar contraction `p_c H^c` has
   coefficient one.
3. A lower momentum component inserted into a free covariant target slot was
   multiplied by `eta_aa`. This insertion is `delta_a^c p_c`, so its numeric
   coefficient is one. The final edge table is `p-H: 1`, `p-output: 1`, and
   `H-output: eta_aa`.

After both corrections, the exact CPU scalar `10001` rank falls from two to the
required one. The surviving pivot is global raw diagram 0 at target row 14.
The former excess pivot, global raw diagram 12 at target row 5,492, disappears.
The old H-output-delta variant recovers rank two and is a suitable negative
control. Identity-output-only is not detected by this one scalar rank, so its
negative gate must use exact stream identity or Lorentz equivariance.

The stale GPU `01001` excess witness selected raw diagrams 0 and 12 at rows
`(source 56320,target 14)` and `(source 991077,target 211)`. Exact CPU replay
of the corrected four-pass C4 numerators gives
`[-1032192,-3096576;-2580480,-7741440]`, whose determinant is exactly zero.
Thus the concrete two-pivot GPU witness collapses after both variance fixes.
The exact boost `(0,1)` commutator also has zero corrected residual on both
maps. Restoring the old H-output-delta convention produces 36 residual entries
for diagram 0 and 84 for diagram 12.

## Candidate inventory

The raw grammar contains 21, 209, and 170 labeled terms in the scalar,
Lambda3, and Lambda4 Fierz channels. These are not independent intertwiners.
An independent permutation-orbit enumeration, including target-wedge parity
and gamma-axis reorder signs, gives 4, 26, and 23 nonzero antisymmetrized orbit
sums. Each orbit sum is then projected separately into the five orthogonal
target sectors. Consequently the sum of sector ranks may exceed the number of
unprojected orbit labels. The required Lambda4 sector ranks `[3,5,6,7,6]` are
each at most 23, so this count alone neither proves nor disproves epsilon/Hodge
completeness.

Candidate order must be:

1. Fierz channel scalar, Lambda3, Lambda4.
2. Within a channel, lexicographic canonical orbit representative.
3. Within an orbit, lexicographic raw diagram ordinal with its exact signed
   multiplicity.

GPU local pivot columns must be translated through this manifest before they
are called seed ordinals. The old local raw-diagram indices are not the 52
abstract seed ordinals in the Hom inventory.

## Required rank gates

For each target sector in canonical order `00001`, `00011`, `00101`, `01001`,
`10001`:

1. Run separate RREF blocks for the three certified orthogonal source Fierz
   projectors, and bind their zero cross-products and source-channel purity.
   Their ranks may then be summed. Without those bindings, a combined RREF is
   required.
2. Do not stop at the expected multiplicity. Retain at least one extra pivot.
3. Require ranks `[7,7,11,14,13]` at each of the three pinned primes.
4. Require no excess pivot at any prime.
5. Record candidate pivots and actual canonical row ordinals for each prime.
   Equal ranks are mandatory. Equal pivot identities are useful but are not a
   mathematical requirement if an exact minor from one prime is replayed.

## Exact rational replay

For one prime's selected pivots in each sector, decode every GPU row ordinal
into its source coordinate and D G4 target coordinate. Recompute every selected
matrix coefficient on CPU from the corrected signed orbit sum, apply the same
four C4 numerator passes, and form the actual square minor. The common
denominator must be positive and coprime to all three primes. Exact Gaussian
elimination over `Q(i)` must produce a nonzero determinant.

A nonzero minor proves rank at least 52. The separately certified Hom dimension
and exhaustive Lorentz-equivariance gate then give rank exactly 52 and close
the Cartesian generator construction. The existing synthetic identity-minor
test does not satisfy this requirement.

## Dependency bindings

The final report must bind at least:

- corrected orbit-manifest semantic hash and binary hash;
- corrected grammar source hash;
- Hhat source basis hash;
- source Fierz projector hash;
- D G4 Cartesian basis, C4 operator, CSR, and projector-polynomial hashes;
- ordered proof primes and denominator audit;
- CUDA source hash and built-binary hash;
- canonical row encoding and normal-form hash;
- exact selected-minor matrix hash and determinant.

The current device manifest is stale. It binds grammar source
`9701f31ccb7f6b6db080aedb5b36af4da29375d696e415a0757a6a37749ad636`,
while the corrected local grammar source is presently
`9a8633cbf1d3d3b64e36124883346c48c29e4d8011fa9caaf8a4d2e85d90ede1`.
It also describes 400 raw labeled terms rather than the corrected
antisymmetrized orbit semantics.

## Publication and mutations

There is no production D21 report-last writer yet. Publish immutable payloads
and per-sector checkpoints first, fsync them, then atomically publish one final
report. The final report must reject absent, truncated, stale, or hash-mismatched
dependencies.

Required negative controls are:

- identity output permutation only;
- H-output coefficient one instead of `eta_aa`;
- wrong C4 shift;
- one flipped orbit-member sign;
- duplicate or reordered candidate representative;
- mutated pivot row or candidate ordinal;
- stale dependency hash;
- missing exact minor replay;
- expected-rank cap with a hidden sentinel pivot.

## Current status

The corrected scalar CPU gate is evidence that the variance diagnosis is
right. It is not evidence that the full grammar is exhaustive or that the
combined rank is 52. CUDA regeneration, all 55 Lorentz commutators, three-prime
sentinel RREF, actual-row exact replay, and report-last publication remain open.
