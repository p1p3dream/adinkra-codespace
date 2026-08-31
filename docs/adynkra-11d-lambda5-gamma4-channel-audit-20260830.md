# Exact Lambda5 and Gamma4 channel audit for the eleven-dimensional four-form gate

**Status date:** 2026-08-30
**Scope:** One-spinor-derivative maps from the gamma-traceless vector-spinor
`H_hat` into the three-form potential and four-form curvature
**Status:** Implementation-oriented audit and acceptance plan. No new physical
channel is promoted by this document.

## 1. Executive result

The repository contains exact pieces adjacent to the requested channel, but it
does not contain an exported exact Cartesian Gamma4 decomposition that can be
wired safely into the physical `H_hat -> A3/G4` map today.

The decisive representation fact is:

```text
(00001) tensor (10001)
  = (00002) + (00010) + (00100) + (01000) + (10000)
    + (10002) + (10010) + (10100) + (11000) + (20000)
```

This 10,240-dimensional product is multiplicity-free. In particular, it
contains exactly one `(00100)` three-form. Therefore the apparent Gamma2 and
Gamma4 routes into `A3` cannot supply two independent coefficients after the
source is restricted correctly to `H_hat`. They must either be proportional
or one of the implementations has a variance, charge-conjugation, Lorentzian
metric, or basis-join error.

The existing Eq. (40) `Psi_[3]` candidate is the Gamma2 exterior route. A
Gamma4 trace route is valuable as an exact cross-check of that ray, but it is
not a second one-derivative ansatz direction. The Gamma4 exterior `Lambda5`
and Gamma4 hook `(10010)` components certify completeness of the decomposition;
they do not map directly into `A3`.

An exploratory isolated implementation copied the current raised-gamma
convention into rank four and passed the raw tensor decomposition tests, but
failed the mandatory multiplicity-one proportionality check on the canonical
10,240-dimensional `D tensor H_hat` domain. Common nonzero rows had ratio 3,
while 29,667 exact rows remained. That implementation was removed rather than
promoted. The failure proves that extending the source-fixed rank-two/rank-five
formula mechanically is not yet a basis-safe construction.

## 2. Exact tensor inventory

Let `V` be the eleven-dimensional Lorentz vector and `S` the 32-dimensional
Majorana spinor. The source field is the gamma-traceless vector-spinor
`H_hat=(10001)` of dimension 320.

The ambient Clifford slice before gamma-trace restriction has dimension:

```text
D_beta H_alpha^c: 32 * 32 * 11 = 11,264
```

The physical one-derivative source is:

```text
S_D^* tensor H_hat: 32 * 320 = 10,240
```

The relevant form-vector decompositions are:

```text
Lambda2 V tensor V
  = Lambda1 V + Lambda3 V + (11000)
  = 11 + 165 + 429
  = 605

Lambda4 V tensor V
  = Lambda3 V + Lambda5 V + (10010)
  = 165 + 462 + 3,003
  = 3,630

Lambda5 V tensor V
  = Lambda4 V + Lambda6 V + (10002)
  = 330 + 462 + 4,290
  = 5,082
```

The Hodge dual identifies `Lambda6 V` with `Lambda5 V` in eleven dimensions,
but that identification includes the Lorentz metric and epsilon convention. It
must not be inferred from a matching dimension alone.

## 3. What exists exactly

### 3.1 Cartesian Majorana and `H_hat` basis

| Object | Exact type or function | Dimension | Status |
|---|---|---:|---|
| Lorentz gamma matrices | `eleven_dimensional_majorana::real_gamma_matrices` | 11 matrices of 32 by 32 | Exact signed permutations |
| Charge conjugation | `real_charge_conjugation` | 32 by 32 | Exact antisymmetric signed permutation |
| `H_hat` basis | `canonical_gamma_traceless_frame_basis` | 320 in ambient 352 | Exact, two signed entries per basis vector |
| Superderivative jet | `visit_linearized_frame_jet` | sparse canonical normal form | Exact ordered-superderivative stream |
| Ambient Gamma2 `D H` | `gamma_dh_operator(2)` | 605 by 11,264 | Exact, 19,360 nonzeros |
| Ambient Gamma5 `D H` | `gamma_dh_operator(5)` | 5,082 by 11,264 | Exact, 162,624 nonzeros |

The internal raised-spinor convention is `Gamma_[p] C^-1` with the signs
derived in `eleven_dimensional_physical_curvature::raised_gamma`. The exported
Gamma operator intentionally accepts only ranks two and five because those are
the source-fixed Eq. (39) channels.

### 3.2 Exact Eq. (40) projectors and solves

| Object | Function | Exact output |
|---|---|---|
| Gamma2 hook | `hook_projector_operator(2)` | rank-429 `(11000)` in ambient dimension 605 |
| Gamma5 hook | `hook_projector_operator(5)` | rank-4,290 `(10002)` in ambient dimension 5,082 |
| Conventional solve | `solve_conventional_compensators` | `Psi1[11]`, `Psi3[165]`, `Psi4[330]`, `Psi5[462]` |
| Differentiated solve | `solve_higher_jet_conventional_compensators` | `D Psi3[5,280]`, `D Psi4[10,560]`, `D Psi5[14,784]`, plus other sectors |
| Polynomial X image | `apply_polynomial_fx` | exact `(11000)+(10002)` with momentum and exterior-spinor keys |

The named `Psi3` is obtained from the normalized total antisymmetric part of
the Gamma2 form-vector. `Psi4` is obtained from the mixed trace of the Gamma5
form-vector. `Psi5` is obtained from the Gamma5 exterior six-form through the
pinned inverse-Hodge convention.

None of those Gamma5 objects is the missing Gamma4 exterior `Lambda5` map.
The representation label and dimension of `Psi5` match `Lambda5`, but the
domain, projector, and normalization path differ.

### 3.3 Exact target `A3/G4` complex

The target-side complex is unambiguous:

| Map | Type | Dimension |
|---|---|---:|
| Three-form curvature | `d: A3 -> G4` | 330 by 165 |
| Four-form Bianchi | `d: G4 -> Lambda5` | 462 by 330 |
| Gravitino descendant | `curl(psi) -> D G4` | 10,560 by 1,760 |

These are exposed by `target_sector_complex(TargetSector::FourForm)` and
`linearized_gravitino_curl_to_d_f_four_operator`. Their polynomial momentum
ordering is exact and already used by the normalization comparison.

### 3.4 Exact Gamma4 maps into raw W

Two rank-four Clifford contractions already target a 330-coordinate four-form:

| Function | Type |
|---|---|
| `t_alpha_e_gamma_to_w_operator` | `T_{alpha e}^gamma[11,264] -> W4[330]` |
| `d_j_to_w_operator` | `D J[1,024] -> W4[330]` |

These are the source-fixed raw-W construction. They do not construct a
Gamma4 `D H` form-vector, a Gamma4 hook projector, or an `A3` potential. Raw W
has a nonzero Bianchi residual and is already proven nonproportional to the
conditional closed candidate on the pinned nonzero-H canary. These operators
must remain separate from a physical `A3/G4` ansatz.

## 4. What the abstract and level-18 infrastructure proves

### 4.1 Exact representation oracles

The spinor-bridge inventory certifies the multiplicity-free decomposition of
`(00001) tensor (10001)`. It contains:

- one `(00100)` of dimension 165;
- one `(00002)` of dimension 462;
- one `(10010)` of dimension 3,003.

The stored level-14 and level-18 highest-weight fixtures include exact copies
of `(00002)` and `(10010)`, with zero raising residuals and verified lowering
strings. They are strong target-image oracles for a future Cartesian join.

### 4.2 What cannot be wired directly

`eleven_dimensional_abstract_clifford_join` implements the abstract Gamma2
contraction and rank-429 `(11000)` hook projector. It does not implement
Gamma4, `(00002)`, or `(10010)` Cartesian projectors.

`eleven_dimensional_level18_target_quotient` assembles certified incidence
blocks for source-gauge routing. Its 77 blocks concern the level-18 target
irreps reached from the existing hook lineage. The block API stores ranks,
offsets, and hashes rather than full Cartesian Clebsch-Gordan coefficients.
It cannot convert a Cartesian Gamma4 form-vector into `Lambda5` or `(10010)`.

The level-18 Hodge-lift fixtures live in exterior powers of the 32-dimensional
spinor basis. They are not the canonical Lorentz-mask bases used by
`FormVectorTensor`. Matching `(00002)` or `(10010)` labels does not supply the
missing intertwiner.

## 5. The one-derivative Hom-space consequence

Define the two apparent three-form contractions:

```text
C2(H) = Alt_3[(Gamma_[2])^(beta alpha) D_beta H_alpha^c]
C4(H) = Tr[(Gamma_[4])^(beta alpha) D_beta H_alpha^c]
```

Before imposing the gamma trace on `H`, these arise from different ambient
form-vector summands. After restricting to `S_D^* tensor (10001)`, the exact
representation inventory contains one `(00100)`. Therefore:

```text
C4 restricted to H_hat = q * C2 restricted to H_hat
```

for one exact Gaussian-rational scalar `q`, provided both maps use the same
spinor variance, charge conjugation, Lorentz metric, and basis conventions.

Consequences:

1. The current one-derivative `H_hat -> A3` ansatz has one coefficient, not two.
2. Gamma4 is a basis and convention cross-check, not a new physical degree of
   freedom.
3. The Gamma4 exterior and hook outputs are completeness witnesses. They do
   not enter `A3` at this bidegree.
4. If an implementation reports independent C2 and C4 images, it has failed a
   representation gate and must not be used in the physical solve.
5. If the correctly joined unique ray still fails the teleparallel descendant,
   the next search must enumerate different bidegrees or additional constrained
   source structures. It must not add a duplicate copy of `(00100)`.

## 6. Why no channel module was retained

The raw Lorentz-mask decomposition is unambiguous, but the source spinor
variance join is not yet exported for rank four. The current `raised_gamma`
helper is private and has been validated in source-fixed ranks two and five.
Mechanical extension to rank four is not enough.

An exploratory module performed:

1. exact rank-four `Gamma_[4] C^-1` contraction;
2. exact decomposition of `Lambda4 V tensor V` into dimensions
   `165 + 462 + 3,003`;
3. exhaustive hook idempotence, trace, exterior, and complement checks;
4. restriction to all `32 * 320 = 10,240` canonical `D tensor H_hat` basis
   states;
5. exact comparison of the Gamma4 trace with the existing Gamma2 exterior.

The tensor projector gates passed, but the multiplicity-one source gate did
not. The common rows all had exact ratio 3, yet the complete maps had different
support and 29,667 residual coordinates. This points to a missing or mismatched
spinor-dual, charge-conjugation, or Lorentzian basis join. Retaining the module
would have converted that mismatch into an apparently valid second channel.
It was therefore removed.

This is the required fail-closed outcome under the instruction to implement
only when the basis is unambiguous.

## 7. Exact channel-construction plan

### Phase C0: freeze the typed source and target bases

Create one basis contract containing:

```text
source derivative spinor: S_D^*, dimension 32
source field: H_hat=(10001), dimension 320
source product: canonical derivative-major, H_hat-minor order, dimension 10,240
ambient DH: derivative-major, H-spinor, vector-minor order, dimension 11,264
Gamma4 form-vector: four-form-mask-major, upper-vector-minor order, dimension 3,630
A3: increasing three-form masks, dimension 165
Lambda5: increasing five-form masks, dimension 462
hook: ambient coordinates in the exact projector image, rank 3,003
G4: increasing four-form masks, dimension 330
D G4: derivative-spinor-major, four-form-minor order, dimension 10,560
```

The contract must hash:

- Majorana gamma matrices;
- charge conjugation and its inverse;
- spinor-dual identification;
- mostly-plus Lorentz metric;
- epsilon orientation;
- `H_hat` basis;
- every form-mask list;
- source and target coordinate order.

### Phase C1: construct Gamma4 by equivariance, not extrapolation

Construct exact Lorentz generators on:

```text
S_D^* tensor H_hat
Lambda4 V tensor V
```

Build the candidate Clifford contraction in the declared variance convention.
Require, for every Lorentz-generator basis element `M_ab`:

```text
Y4 * rho_source(M_ab) = rho_target(M_ab) * Y4
```

over exact `Q(i)`. Also solve the intertwiner space independently. Its rank
must be one for the selected Gamma4 occurrence, and the constructed matrix
must span that kernel. Do not infer correctness only from gamma identities in
the ambient 32 by 32 matrices.

Expected operator shape after direct source restriction:

```text
Y4: 3,630 by 10,240
```

The ambient 3,630 by 11,264 operator may be emitted as a diagnostic, but the
authoritative map includes the exact `H_hat` injection.

### Phase C2: export the complete Gamma4 tensor decomposition

Build exact sparse operators:

```text
T3: Lambda4 V tensor V -> Lambda3 V       165 by 3,630
A5: Lambda4 V tensor V -> Lambda5 V       462 by 3,630
P10010: Lambda4 V tensor V -> same       3,630 by 3,630, rank 3,003
I3: Lambda3 V -> Lambda4 V tensor V      3,630 by 165
I5: Lambda5 V -> Lambda4 V tensor V      3,630 by 462
```

Use normalized total antisymmetrization and mixed trace in the existing
Lorentz-mask convention. For degree four, the mixed-trace injection has the
exact eigenvalue `-8`; the exterior normalization is `1/5`.

Require:

```text
T3 I3 = identity_165
A5 I5 = identity_462
P10010^2 = P10010
T3 P10010 = 0
A5 P10010 = 0
P10010 + I3 T3 + I5 A5 = identity_3630
trace(P10010) = 3003
```

### Phase C3: construct Cartesian-to-B5 joins

Build and export two exact intertwiners:

```text
J00002: Cartesian Lambda5[462] -> stored B5 `(00002)` orbit[462]
J10010: Cartesian hook image[3003] -> stored B5 `(10010)` orbit[3003]
```

Fix each map by a primitive highest-weight vector and all lowering strings.
Check every simple-root generator. Bind the result to the exact fixture hashes.
The joins must be invertible on their declared images.

This step converts the existing highest-weight fixtures from representation
oracles into usable Cartesian projector certificates. Level-18 incidence
blocks remain downstream consumers, not substitutes for these joins.

### Phase C4: close the multiplicity-one A3 gate

Construct:

```text
C2 = normalized Gamma2 exterior after H_hat restriction
C4 = T3 * Y4
```

Solve one exact scalar `q` from the first common canonical nonzero coordinate,
then verify all 10,240 source basis columns:

```text
C4 - q C2 = 0
```

Also compute exact ranks. Each nonzero map must have rank 165. The combined
stack must still have one-dimensional multiplicity space.

If this gate fails, stop. Do not send either map into the normalization solve.

### Phase C5: rebuild the one-ray physical descendant

Choose the primitive A3 ray only after Phase C4. Apply the exact target
curvature:

```text
H_hat -> A3[165] -> G4[330]
```

Differentiate in the existing ordered-superderivative normal form and compare
with:

```text
H_hat -> gravitino curl[1,760] -> D G4[10,560]
```

The comparison solves one overall physical normalization. It must use the
union of exact canonical polynomial rows across all 320 `H_hat` basis columns.

The current one-column nonproportionality witness must be replayed. A corrected
basis join may change the candidate stream. If it does not, the unique
one-derivative ray is ruled out on the unrestricted `H_hat` source domain.

### Phase C6: expand only by proved bidegrees

If the unique one-derivative ray fails after all basis gates pass, enumerate
new Hom spaces at different `(d_D,d_p)`. For each bidegree:

1. compute the full representation multiplicity of `(00010)` or the potential
   `(00100)`;
2. construct every exact Cartesian intertwiner;
3. quotient normal-form identities before counting coefficients;
4. impose closure and descendant constraints;
5. prove that the scanned bidegree inventory is exhaustive for the stated
   locality and degree bound.

Do not count `Lambda5`, `(10010)`, raw W, or repeated presentations of the
same `(00100)` ray as new coefficients.

## 8. Acceptance tests

### Gate G0: basis and variance

- Verify `C^T=-C`, `C^2=-I`, and the complete Clifford relation.
- Verify the derivative-spinor dual action explicitly.
- Verify the `H_hat` basis has zero gamma trace on all 320 states.
- Mutate `C^-1` placement, transpose, time-axis sign, and spinor factor order;
  every mutation must fail equivariance or the C2/C4 proportionality gate.

### Gate G1: Gamma4 raw operator

- Require exact shape 3,630 by 10,240 after source restriction.
- Compare sparse application with a direct index-loop oracle.
- Require exact Lorentz equivariance for all 55 generators.
- Verify the unique intertwiner-space dimension.

### Gate G2: tensor projectors

- Verify dimensions `165 + 462 + 3,003 = 3,630`.
- Verify every projector identity in Phase C2 exhaustively.
- Verify pairwise orthogonality and identity resolution.
- Mutate `1/5`, `-1/8`, insertion signs, and the time-axis metric sign.

### Gate G3: B5 joins

- Verify all five raising and lowering actions exactly.
- Verify ranks 462 and 3,003.
- Verify primitive highest-weight normalization and inverse composition.
- Mutate one fixture coefficient and require a nonzero exact residual.

### Gate G4: multiplicity-one source restriction

- Require the exact tensor inventory to contain one `(00100)`.
- Require both C2 and C4 to be nonzero and rank 165.
- Require one universal exact proportionality factor on all 10,240 columns.
- Reject any support mismatch, even if all common rows have the same ratio.

### Gate G5: target curvature

- Require `d: 165 -> 330` and Bianchi `d: 330 -> 462`.
- Require `d^2=0` as an exact polynomial matrix.
- Preserve momentum exponents and exterior-spinor masks exactly.
- Mutate one wedge sign and require a Bianchi residual.

### Gate G6: physical descendant

- Solve one exact normalization against the 10,560-row teleparallel target.
- Verify every row on all 320 source columns.
- Capture the minimum canonical exact mismatch if the solve fails.
- A modular match is insufficient; final residuals are over `Q(i)`.

### Gate G7: regression and scope

- Preserve the current Gamma2 Eq. (40) stream digest when no source bug is
  found.
- Preserve Gamma5/X5 hook results and their rank-4,290 certificate.
- Preserve raw W as a separate conditional stream.
- State explicitly that a one-ray failure rules out only the declared
  bidegree on the unrestricted source domain.

## 9. Performance decision

This is initially a CPU exact-algebra problem. The coefficient space at the
one-derivative gate has dimension one after the multiplicity proof. The sparse
operators have at most 10,240 source columns and small per-column support.
GPU rank or nullspace code would add complexity without changing wall time
materially.

GPU application becomes eligible only after:

- the CPU implementation passes all basis and mutation gates;
- profiling shows more than one minute in sparse application; and
- batches contain roughly one million or more exact sparse products.

The direct device-output and three-prime plan remains appropriate if later
bidegree enumeration produces hundreds of independent columns. It is not the
next step for the Gamma4 cross-check.

## 10. Smallest safe patch sequence

1. Add an exact dual-spinor and Lorentz-generator basis contract.
2. Add an isolated Gamma4 module with a direct 10,240-column source-restricted
   operator.
3. Add the exact `Lambda3 + Lambda5 + (10010)` projectors.
4. Pass G0 through G2 before exposing any channel to physical normalization.
5. Add the two Cartesian-to-B5 joins and pass G3.
6. Prove C2/C4 proportionality and exact rank in G4.
7. Only then add a second typed input path to the normalization runner for
   cross-checking the same one-dimensional ray.
8. Replay the teleparallel comparison and classify the result.
9. If the ray fails, begin an explicit higher-bidegree Hom inventory rather
   than adding ad hoc channels.

## 11. Bottom line

The exact Gamma5/X5 and level-18 machinery is useful evidence, but it cannot be
wired directly into a new physical `A3/G4` channel. The missing object is a
variance-correct, equivariant Cartesian Gamma4 decomposition with explicit
joins for `Lambda5` and `(10010)`. Once built, its A3 trace must collapse onto
the existing Gamma2 ray because `(00100)` occurs once. Its primary scientific
value is therefore to validate or falsify the present basis convention before
the one-ray teleparallel comparison is trusted.
