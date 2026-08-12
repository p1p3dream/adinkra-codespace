# Higher-dimensional gate for the eight-color recursion

## Purpose

The signed recursion produces 24 irreducible `8|8` Garden representations in
one fixed-color nodal class. Half are built from the named `CM`, `TM`, and `VM`
inputs. Half are built from `VM1`, `VM2`, and `VM3`, for which the source gives
no four-dimensional parent. One-dimensional closure, signed equivalence,
HYMN, commutants, and the first Gadget do not recover that distinction.

The next gate must retain the spatial derivative and gauge data discarded by
the Carroll reduction.

## Primary-source positive controls

arXiv:1405.0048 supplies the needed four-dimensional data for the two known
closing systems.

### Chiral-vector

- Eqs. (32)-(35): the two supersymmetry transformation laws;
- Eqs. (36)-(37): invariant Lagrangian and field strength;
- Eq. (38): closure, including the gauge term on the vector potential;
- Eq. (40): temporal gauge `A_0 = 0`;
- Eq. (41): the field-to-node reduction map; and
- Appendix B: the resulting `8 x 8` linkage matrices.

### Chiral-tensor

- Eqs. (44)-(47): the two supersymmetry transformation laws;
- Eqs. (48)-(49): invariant Lagrangian and tensor field strength;
- Eq. (50): closure, including the gauge term on the two-form potential;
- Eq. (52): temporal gauge `B_0i = 0`;
- Eq. (53): the field-to-node reduction map; and
- Appendix C: the resulting `8 x 8` linkage matrices.

These are the positive controls. Their four-dimensional algebra must be
reproduced first, then their gauge-fixed reductions must match the exact `CV`
and `CT` anchors already stored in
`data/permutahedron_s8_signed_equivalence.json`.

## Current status

The chiral-vector positive control is complete. Exact Rust and independent
JavaScript implementations verify all 612 component relations, including the
vector gauge residue, and recover all 512 entries of the committed `CV` anchor.
See `docs/chiral-vector-4d-positive-control.md`.

The chiral-tensor positive control is also complete. Exact Rust and independent
JavaScript implementations verify all 684 component relations, including the
two-form gauge residue, and recover all 512 entries of the committed `CT`
anchor. See `docs/chiral-tensor-4d-positive-control.md`.

## Exact implementation order

1. Pin the four-dimensional Majorana gamma-matrix, charge-conjugation, metric,
   and epsilon conventions used by arXiv:1405.0048. Verify the Clifford and
   charge-conjugation identities independently.
2. Represent fields and first derivatives as exact sparse jets. Retain the
   vector and two-form potentials so gauge terms remain visible.
3. Completed: transcribe the `CV` rules in Eqs. (32)-(35). Evaluate every supercharge pair
   on every component field and compare the residue with Eq. (38), including
   the vector gauge transformation.
4. Completed: apply Eqs. (40)-(41), reduce to one temporal coordinate, and verify the
   resulting matrices against the committed `CV` anchor byte for byte.
5. Completed: repeat the component closure and reduction checks for `CT` using
   Eqs. (44)-(53), including the two-form gauge transformation.
6. Completed: record the spatial linkage and gauge data that distinguish the
   two positive controls despite their common `8|8` Garden size. See
   `docs/cv-ct-higher-dimensional-fingerprints.md`.
7. Completed: define the minimum additional data that a `VM1`, `VM2`, or
   `VM3` candidate would need before the same test is meaningful. Do not infer
   missing spatial transformations from a valise matrix alone.

## Gauge and phantom continuation

The Maxwell magnetic phantom sector is now an additional positive control.
Exact Rust and independent JavaScript implementations extract the 12 nonzero
phantom entries and verify every spatial magnetic row in Eq. (5.8) of
arXiv:0907.3605. See `docs/maxwell-phantom-positive-control.md`.

The canonical Bianchi reshuffling in Eqs. (5.4)-(5.5) and the complete
gauge-enhancement condition in Eq. (5.11) are now implemented. The known
Maxwell source passes with 144 raw bosonic Omega entries reduced to zero, and
zero fermionic residual entries. The worldline-only search also recovers eight
signed-frame witnesses for Maxwell in both source and scrambled bases, while a
chiral negative control has none. See `docs/maxwell-worldline-recovery.md`.

The remaining gate before any eight-color application is a target
specification: Lorentz representations, gauge-potential and field-strength
degrees, phantom inventory, and Bianchi complex. Those data are not determined
by the eight-color valise matrices.

The Maxwell gate has also been run across all 96 published fiducial signed S4
quartets. Exactly the 48 signings with `chi0 = -1` pass; all 48 with
`chi0 = +1` fail. Thus the complete four-color gauge calculation supplies no
additional selector beyond `chi0` on this library. See
`docs/maxwell-s4-published-signing-scan.md`.

The next finite eight-color calculation is limited to the ordered pair of
embedded four-color subalgebra classes. It does not claim a complete
eight-supercharge gauge-enhancement test.

## Acceptance gates

The positive-control implementation passes only if all of the following hold:

- exact four-dimensional closure on every nongauge field;
- exact factorization of the vector and two-form residues into the published
  gauge transformations;
- exact agreement with the published temporal gauge choices;
- byte-for-byte agreement of the reduced signed matrices with the stored `CV`
  and `CT` anchors; and
- independent verification using a separately entered source fixture or a
  second exact implementation.

## What this can establish

This pass can identify which data distinguish known four-dimensional parents
after one-dimensional equivalence has collapsed them together. It can also
turn the phrase "higher-dimensional enhancement" into a concrete set of
verified equations and gauge residues.

It cannot manufacture four-dimensional transformation laws for `VM1`, `VM2`,
or `VM3`. A negative result for those candidates is meaningful only after a
candidate spatial linkage and gauge structure has been specified.

## Howard-talk boundary

`HowardTLK.v2.pdf`, pp. 61, 64, 73, 77, and 80-81, fixes the six-quartet
permutation geometry and hopping construction used as the base case. It does
not contain the four-dimensional component laws required here. Those laws
come from arXiv:1405.0048.
