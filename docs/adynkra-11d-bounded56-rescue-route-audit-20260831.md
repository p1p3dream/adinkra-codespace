# Rescue-route audit after the corrected bounded-56 witness

**Date:** 2026-08-31
**Status:** witness-first route selection, no source constraint adopted
**Scope:** the corrected `(d_D,d_p)=(2,1)+(0,2)` comparison from the
`320`-component gamma-traceless `H_hat` source into `D G4`.

## 1. What is established, and what is not

The corrected target-basis join sends the teleparallel four-form ordinal to the
numeric-mask ordinal used by the `D G4` projectors. Its column-zero stream has
`342,640` D21 rows, `1,080` D02 rows, no `p wedge G4` Bianchi residual, and
stream SHA-256

```text
dfd7fc0ace00d202b83a7c3ae15aa2af666fd876bea4b7f5d59c3086aeeee997
```

The corrected scoped replay isolates outer Fierz degree three and target
sector `01001`. Six candidate maps have an exact nonzero six-by-six pivot
minor, but the corrected target adds a seventh independent functional. The
first augmented functional is canonical row `1,392,410,608`; the target is
zero and the pivot-fit candidate value is

```text
-21707/184320 + 7 i/9216.
```

The augmented determinant has nonzero residues at all three pins. This is a
bounded no-solution witness for the declared 56-map direct sum. It is not an
exhaustion of source constraints, source equations of motion, or all possible
filtered differential operators.

The production compact witness currently resides at
`results/adynkra_11d_four_form_57_global_augmented_witness.json`, SHA-256
`e8f0ff40a6ceece037dec822e176b0dd91da408ce2ffd4d967343ee74f1c4af2`.
The fuller corrected scoped replay used for the exact cross residual is
`/tmp/adynkra_global57_scoped_corrected.json`, SHA-256
`6cf02fd6bdb872118431c7e0c5e1887891737fa472e10150ee57e30f50759e75`.
The temporary file is not durable evidence and must not be cited as a sealed
certificate.

## 2. Mandatory consistency gate before any rescue

There is a representation-theoretic incompatibility that must be resolved
first.

For the failing source Fierz degree three and target `01001` block, the exact
Hom multiplicity is six. The six selected D21 maps have exact rank six. The
higher-bidegree inventory also states that the corrected teleparallel target
is a Lorentz-equivariant member of this same Hom space. These three statements
imply that the target lies in the six-dimensional candidate span. They cannot
coexist with augmented rank seven.

Therefore at least one of the following remains false:

1. the gauge-fixed teleparallel target is Lorentz equivariant as a map on the
   declared canonical `H_hat` jet;
2. its source and target variance, charge-duality, form-ordinal, or PBW actions
   match those used by the invariant-diagram basis;
3. the six-map block is a complete basis for the declared Hom space; or
4. the source-Fierz and target-Casimir projected functionals are being replayed
   with the same normalization on both sides.

This is the first launch gate, not optional cleanup. A source equation must not
be introduced to hide a convention failure.

### Gate E0: witness-local target equivariance

Decode row `1,392,410,608` as:

```text
source coordinate 131857
outer spinor pair (1,8)
momentum axis 5
H_hat ordinal 17
target coordinate 688 = spinor 2, numeric four-form ordinal 28
```

Compute

```text
rho_DG4(X) T - T rho_D2pH(X)
```

first for a boost touching the time convention and for a rotation touching the
witness indices. Recommended canaries are `M_01`, `M_18`, and `M_15`. Reuse the
exact source actions and `dg4_lorentz_generator_action_integer` already used by
`src/eleven_dimensional_d21_invariant_diagrams.rs`. If any canary is nonzero,
record the first source coordinate, target coordinate, exact coefficient, and
which of source action, target action, form permutation, or metric factor
created it. Then stop the rescue line and correct that adapter.

If the canaries vanish, replay all 55 Lorentz generators over the complete
canonical source basis. A zero result forces a re-audit of the six-dimensional
Hom completeness and projector normalization, because augmented rank seven is
then impossible under the stated representation data.

## 3. Known quotients cannot remove this witness

The corpus and repository already close the obvious quotient candidates.

### 3.1 Gamma trace

The local gamma-trace image has dimension 32 and is annihilated by `P_320`.
The bounded comparison begins on the 320-dimensional image of `P_320`.
Reintroducing the trace spinor as an independent source adds different
source-tagged rows and cannot cancel a residual on an existing `H_hat` row.
This is recorded in `docs/adynkra-11d-physical-k-target-quotient-20260824.md`
and `docs/adynkra-11d-enlarged-source-hom-inventory-20260831.md`.

### 3.2 Local Lorentz

The exact linearized flat-background orbit has

```text
delta H_hat = 0,
delta Psi_[2] = Lambda_[2],
delta Psi_[1,3,4,5] = 0.
```

The source-fixed audit records these facts at
`src/eleven_dimensional_lorentz_holonomy_compensator_audit.rs:520-526`.
`p wedge Psi_[2]` is target `A3` gauge and has zero `G4` image. Thus the known
local-Lorentz quotient has no nonzero `H_hat` direction capable of killing the
augmented witness.

### 3.3 Eq. (40) compensators and scale

`Psi_[1,3,4,5]` are already algebraically pulled back from `D H` by Eq. (40).
Adding independent copies double-counts those source directions unless a new
homogeneous solution is derived. The scalar scale is independently tagged and
cannot alter an `H_hat` row without a new proved differential pullback. The
post-quotient source currently is `320 H_hat + 1 scale`, as summarized in
`docs/adynkra-11d-enlarged-source-hom-inventory-20260831.md:218-243`.

## 4. Smallest source restriction that could work

No audited source prints an additional off-shell differential constraint on
`H_hat` in convention-fixed executable form. The nearest exact restriction
already implemented in the repository is the free Rarita-Schwinger equation
pulled back through the direct gravitino frame and curl.

Define

```text
C_RS = D_outer o Euler_RS o Curl o Eq25Frame o H_hat.
```

Here `Curl o Eq25Frame` is the direct gravitino-curvature stream.
`Euler_RS` is the exact gamma-three contraction of that curvature. One further
outer spinor derivative places the constraint on the same source-jet order as
the corrected `D G4` comparison.

This is executable with existing pieces:

* direct curl construction:
  `src/eleven_dimensional_complete_f.rs:1261-1364`;
* Rarita curvature and curvature-to-Euler matrices:
  `src/eleven_dimensional_complete_f.rs:559-587`;
* public exact Euler application:
  `src/eleven_dimensional_complete_f.rs:1916-1920`;
* target equation complex and its explicit on-shell boundary:
  `src/eleven_dimensional_target_equation_complex.rs:1-13` and
  `559-588`.

The additional derivative must use the same right-charge adapter and PBW
normal form as the corrected full-chain comparator. Anticommuting the two
spinor derivatives produces both D21 and D02 terms. A valid quotient test must
therefore retain both branches. Testing only the D21 projection can create a
false positive by ignoring a D02 obstruction.

This restriction is explicitly on-shell. If it removes the witness, the result
is an on-shell identification only. It does not certify the off-shell `F`, an
off-shell prepotential, or irreducibility.

## 5. Witness-first quotient test

Let `w` be the exact augmented residual functional defined by the failing
seven-by-seven minor. Let `C` be a sparse matrix whose rows are the complete
canonical PBW coefficients of `C_RS` on the same source coordinates.
Equality restricted to `ker C` requires

```text
w in row(C).
```

For the full operator residual `R=M c-s t`, the positive certificate is the
stronger factorization

```text
R = Y C
```

for an exact multiplier matrix `Y`.

### Gate Q0: one-witness negative screen

1. Emit the 11,264 raw one-derivative prolongations of the 352
   Rarita-Schwinger Euler coordinates.
2. PBW-normalize them into the common D21 plus D02 source key.
3. Deduplicate exact rows and compute `rank(C)` and `rank([C;w])` at one pinned
   prime.
4. If the rank increases, `C_RS` cannot kill the witness. Stop. This one rank
   increase is a valid negative certificate.
5. Repeat at all three pins only for provenance and bad-prime rejection.

A generic-momentum specialization is unnecessary if the canonical PBW
coefficient rows are retained symbolically. If an implementation uses a
momentum specialization as a fast canary, a rank increase disproves the
restriction, but equal rank does not prove it.

### Gate Q1: exact positive replay

If the modular rank does not increase:

1. solve an exact sparse row combination `w=y C` over `Q(i)`;
2. replay every source coordinate, including the D02 tail;
3. require a nonzero residual after deleting one PBW anticommutator term;
4. extend from the single witness to all rows and solve `M c-s t=Y C`;
5. require `s != 0`, all-row exact residual zero, and the target Bianchi gate.

Because `w` and the prolonged constraint are already represented at the same
finite PBW bidegrees, this first exact membership test is finite-dimensional
linear algebra. It does not require a general Groebner calculation.

### Gate Q2: only then add Einstein

If `C_RS` fails, the next smallest physically motivated restriction is

```text
C = [C_RS; C_Einstein],
```

where `C_Einstein` pulls the direct linearized Riemann stream through the exact
graviton curvature-to-Euler map. The required target matrices are cached at
`src/eleven_dimensional_complete_f.rs:589-607`. Do not build this larger block
unless the Rarita-only witness screen fails to remove `w` or a source theorem
requires the coupled equations.

## 6. Why a different unrestricted bidegree is not a repair

At the same engineering degree, the first omitted canonical PBW branch is
`(4,0)`, alongside the already built `(2,1)` and `(0,2)` branches. It matters
for a future completeness theorem, but it does not by itself repair the
corrected scoped witness.

Canonical PBW bidegrees are direct-sum row families. A new column supported
only on `(4,0)` is zero on the failing D21 functionals, so row monotonicity
leaves the augmented rank-seven minor unchanged. If a four-spinor construction
has lower anticommutator tails on D21 or D02, those tails are lower-filtered
Lorentz-equivariant maps. They must already lie in the complete D21 plus D02
Hom spaces if the present completeness premise is correct. Otherwise their
failure to do so is evidence that the premise or PBW adapter is incomplete,
not evidence for a new physical coefficient.

A different bidegree can affect this witness only when a proved differential
source relation identifies its rows with D21 or D02. That is precisely the
source-constraint factorization problem above. Randomly enumerating higher
bidegrees before Gate E0 and Gate Q0 cannot answer the rescue question.

The mass-shell relation `p^2=0` alone is also insufficient. The failing
functional is linear in momentum, while the ideal generated by `p^2` has no
momentum-degree-one component. It cannot contain `w` without an additional
spinorial or field-equation constraint.

## 7. Paper boundary

The audited corpus supplies the following, and no more:

* arXiv:2007.05097 fixes the gamma-traceless convention and local gamma-trace
  redundancy. Those are already implemented by `P_320`.
* hep-th/0101037 supplies the Eq. (25) frame, compensator quotient, and Eq. (40)
  algebraic solves. It describes the semi-prepotential as differentially
  constrained but does not print the missing fundamental constraint in an
  executable convention-fixed form.
* hep-th/0107155 supplies the first-descendant four-form/gravitino relation and
  on-shell superspace descendants. It is an on-shell oracle, not a new
  off-shell source quotient.
* the target free complexes reproduce the `44/84/128` physical quotients, but
  their module states explicitly that it is target-side and on-shell.

Therefore `C_RS` is the smallest honest executable restriction. It is a test
of whether the corrected bounded failure disappears on the known free
on-shell locus, not a derivation of the missing off-shell semi-prepotential
constraint.

## 8. Execution order and stop rules

```text
E0  target Lorentz equivariance at the augmented witness
    nonzero -> fix conventions; do not impose source constraints
    zero -> all 55 generators, then re-audit Hom completeness

K0  replay known gamma-trace, local-Lorentz, and Eq. (40) quotient facts
    expected result: none removes an H_hat witness row

Q0  build D-extended Rarita Euler constraint and test w in row(C_RS)
    rank increase -> Rarita restriction rejected
    equal rank -> exact lift and mutation

Q1  if exact witness lift passes, solve the full equality modulo C_RS
    success -> on-shell-only identification
    failure -> Rarita restriction does not rescue the operator

Q2  add Einstein only if physically required and repeat the same witness test

B4  inventory `(4,0)` for completeness only
    do not call it a rescue unless a proved source relation connects branches
```

The first new code should be Gate E0. The first possible physics rescue should
be Gate Q0. Both are witness-first and should complete without constructing a
full quotient basis.
