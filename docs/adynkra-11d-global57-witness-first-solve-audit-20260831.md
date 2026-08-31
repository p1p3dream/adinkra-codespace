# Global-57 witness-first solve audit

Date: 2026-08-31

## Scope and verdict

This audit is narrower than the final physical-promotion contract. It asks
only whether the declared 56-map ansatz equals the corrected teleparallel
target on all 320 H-hat inputs and on both canonical PBW branches.

For that bounded equality question, the minimal scientific matrix is

```text
A = [ M | -t ],
```

where `M` has the 56 selected candidate maps and `t` is one shared target
column. The row family is the exact support union of `M` and `t` over:

```text
(2,1): Lambda2(spinor) x V-momentum x H-hat x D-G4
(0,2): Sym2(V-momentum) x H-hat x D-G4.
```

Dense zero rows need not be emitted. Candidate-only and target-only rows must
be emitted. Duplicate contributions must be exactly accumulated before a row
is reduced.

**Launch verdict:** the design is sound, but the current scoped coefficient
artifact is not this solve. If the global confirmation run proceeds, it still
needs the D02 target in its exact pivot fit and complete all-320 residual
traversal for the selected D21 and D02 maps. A 57-wide modular reducer is useful
independent parity, but is not required by the minimal exact witness-first
algorithm below.

## PBW integrability is not an extra equality row family

Let `V_PBW` be the published canonical normal-form row space and let `L` be
any linear PBW, Bianchi, or descendant identity satisfied by the corrected
target. If the all-row solve returns

```text
M c = s t,  with s != 0,
```

then

```text
L(M c) = s L(t) = 0.
```

More generally, if integrability means that `t` has a certified ancestor
under a linear descendant map, then `M c=s t` has the correspondingly scaled
ancestor. Therefore a separate D21-to-D02 PBW translation matrix is redundant
for this bounded equality decision, provided that:

1. both sides use the identical canonical PBW serializer;
2. every nonzero row in both declared branches is compared;
3. `s` is proved nonzero; and
4. the target's own PBW/descendant certificate is pinned.

PBW, Bianchi, source-gauge, and target-gauge audits remain valuable independent
integrity and physical-promotion gates. They are not prerequisites for
launching the bounded all-row equality screen. Equality cannot repair an
uncertified target convention, so those gates still control later wording.

## Exact rank and nullity inference

The selected candidate map has rank 56:

* D21 has exact rank 52 from the nonzero projected minors;
* D02 has rank four from the three distinct target sectors and the exact
  rank-two `10001` pair;
* the two PBW branches form a direct sum.

Consequently `rank(A)` has only two possible values.

| exact result | homogeneous nullity | inference |
|---|---:|---|
| `rank(A)=57` | 0 | `t` is not in the declared 56-map span |
| `rank(A)=56` | 1 | unique ray `(c,s)`, automatically with `s != 0` |

The second statement follows because `s=0` would imply `M c=0`, and rank 56
of `M` forces `c=0`.

For a negative result, rank 57 at one denominator-admissible pinned prime is
already a characteristic-zero proof. Preserve an actual-row nonzero 57 by 57
minor or an equivalent replayable modular witness.

For a positive result, modular rank 56 at three primes is not enough. Solve an
exact nonsingular 56-row subsystem over `Q(i)`, normalize its kernel vector to
`s=1`, and replay `M c-t=0` on the complete support union. The exact solution
gives `rank(A)<=56`; the existing exact rank-56 candidate witness gives
`rank(A)>=56`, closing equality and uniqueness.

If the implementation does not consume the existing exact rank-56 candidate
witness, it must separately report `rank(M)` and `rank(A)`. In the general
rank-deficient case, `rank(A)=rank(M)` means the target is in the span, while
`rank(A)=rank(M)+1` means it is not. Nullity alone does not distinguish these
cases because `s=0` candidate-kernel directions can remain.

## Minimal exact witness-first algorithm

1. Retain an exact invertible 56-functional witness `P M`. It may use the
   already certified source-Fierz and target-sector projections, because an
   invertible linear projection proves independence of the underlying raw
   columns. It must include the exact four-column D02 witness.
2. Normalize `s=1` and solve `(P M)c=P t` exactly over `Q(i)`. This loses no
   nonzero-scale solution because `(c,s)` rescales to `(c/s,1)`.
3. Stream the single exact residual `M c-t` over the complete canonical
   support union, accumulating all contributions before testing a row.
4. Stop on the first exact nonzero residual. Since `P M` is invertible, the
   pivot-fit `c` is the only possible coefficient vector, so one mismatch is
   a characteristic-zero no-solution proof.
5. If every row vanishes, the exact rank-56 witness plus the complete replay
   proves existence and uniqueness.
6. Run the three-prime augmented RREF as independent parity if desired, but
   do not make it a prerequisite for the exact proof.

The scale column must be shared across both PBW branches. Solving D21 and D02
with independent scales would answer a weaker and incorrect question.

## Actual current blockers

### 0. RHS target-basis defect found and corrected locally

The teleparallel operator numbers its four-form coordinate with
`lexicographic_combinations(4)`. The DG4 Casimir projectors and the D21/D02
candidate generators number it by increasing numeric four-form bitmask.
The first RHS adapter copied `FullChainRowKey.output_coordinate` without
applying this permutation. The first scoped target-sector fit likewise fed
lexicographic coordinates directly to a numeric-basis projector.

This is not a cosmetic ordering difference. Applying the numeric-basis
`p wedge` implementation to the unpermuted column-zero target produced
2,386,880 nonzero Bianchi rows. All scoped cross-pivot mismatches from that
unpermuted run are invalid and must not be published.

The working tree now converts

```text
spinor * 330 + lexicographic_four_form_ordinal
    -> spinor * 330 + numeric_mask_four_form_ordinal.
```

The exhaustive 330-form forward/inverse bijection passes. The corrected
column-zero target has 342,640 D21 rows, 1,080 D02 rows, zero Bianchi residual,
and SHA-256
`dfd7fc0ace00d202b83a7c3ae15aa2af666fd876bea4b7f5d59c3086aeeee997`.
The old identity join remains a rejecting mutation. All earlier target stream
and coefficient hashes must be regenerated.

### 0a. Corrected scoped replay already gives a bounded no-solution witness

After the target permutation, the complete outer-degree-three to `01001` Hom
block still fails. Its six selected candidate columns have exact nonzero pivot
minor

```text
46628102050115953563425165663010816000 / 3870720^6.
```

The exact pivot-fit combination evaluated at projected functional row
1,392,410,608 is

```text
-21707/184320 + 7 i/9216,
```

while the corrected teleparallel target is zero there. Appending this row to
the six pivot functionals gives a nonzero 7 by 7 augmented minor. Its Gaussian
residues are nonzero at all three pins:

```text
1073741783: (751787349, 1057120381)
1073741723: (197780636, 908340501)
1073741719: (253625624, 124401636)
```

Exact source-Fierz degree three and target-C4 `01001` projectors isolate this
block. Its Hom multiplicity is six, so no other D21 coefficient can repair the
failure. D02 is a different PBW branch. Once this corrected witness is
published report-last with the new basis-permutation and RHS hashes, it is a
complete bounded no-solution proof for the declared 56-map ansatz. The full
global traversal is then confirmation, not a logical requirement.

### 1. The existing coefficient artifact omits the physical D02 target

`results/adynkra_11d_four_form_56_physical_coefficient_solve.json` forces
columns 52 through 55 to zero and describes a D21 target. It is based on
sector witness rows, not the all-row support union.

The exact row adapter proves that the corrected target is not D21-only. On
H-hat source ordinal zero it contains:

```text
D21 rows: 342,640
D02 rows:   1,080
total:    343,720
SHA-256: 9f0d105f0c1bfe40395b5e42c01c21b61b1ca328f19a83b7e2545cdb097bab95
```

Thus the current artifact cannot be adopted or used as the exact starting
solution for global-57.

### 2. Complete selected D21 residual traversal is not yet bound

The D21 artifact binds selected diagrams and exact witness minors. The launch
needs their complete projected raw-coordinate action with immutable global
column identities. It may emit each full column, or directly accumulate the
coefficient-weighted residual. On-demand evaluation is acceptable if it is
deterministic, covers candidate-only rows as well as target support, and
publishes the complete streamed digest.

### 3. Optional augmented reducer is 56-wide

`StreamedCandidateRref` rejects widths above 56 and decodes keys modulo 56.
The CUDA reducer also hardcodes `kColumns=56`. Global-57 must use width 57 in
the key serializer, CUDA bounds, row grouping, checkpoint schema, and exact
pivot replay if the augmented modular path is used. This does not block the
minimal exact pivot-fit plus residual-replay path. In an augmented run, a
target-only row must encode as column 56, never as a separate side channel
that can be dropped from candidate support.

### 4. Aggregate all-320 RHS publication is missing

`teleparallel_rhs_column` correctly joins one source ordinal into both
branches. The launch needs all 320 columns streamed in source order, with an
aggregate digest, per-branch counts, denominator audit, and resume boundary at
an H-hat ordinal or smaller canonical row block.

### 5. Modular denominator encoding must be closed

The candidate projectors and target have different rational denominators.
Before launch, either prove one common denominator and scaled numerators fit
the declared integer widths, or reduce each exact rational using its own
denominator inverse modulo every pinned prime. Every denominator must be
nonzero at each prime. This is an arithmetic input gate, not a new scientific
row family.

## Not launch blockers for the bounded equality screen

The following remain required before broader claims but need not delay the
global-57 equality launch:

* a separate PBW translation matrix;
* an appended Bianchi matrix, if the target Bianchi certificate is pinned;
* physical K and the full source-gauge quotient;
* Euler/Noether rows;
* bidegree exhaustion beyond `(2,1)` and `(0,2)`;
* final normalization against an external component convention.

The report boundary must say that a positive result identifies the unique map
in the declared 56-dimensional two-branch ansatz relative to the corrected
teleparallel target. It does not by itself prove source-gauge descent,
bidegree exhaustion, complete physical F, or irreducibility.
