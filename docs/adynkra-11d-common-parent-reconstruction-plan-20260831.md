# 11D common-parent reconstruction plan

Date: 2026-08-31

## Objective

Derive a constrained map from an unconstrained parent prepotential into the
physical eleven-dimensional component complex. Do not search for another
unrestricted map from `Hhat` to `A3` at the same engineering weight.

The target component complex is already exact:

```text
(xi, Lambda2, epsilon)
        | K_component
        v
   (h, A3, psi)
        | F_component
        v
  (R, G4, curl)
```

Its formal compositions vanish, and its null-momentum cohomology is
`44 + 84 | 128`. The unresolved diagram is:

```text
Xi_target --> Hhat --> physical component complex
                  ^
                  |
             parent prepotential
```

The corrected Eq. 40 `Lambda3` ray cannot fill the second arrow on unrestricted
`Hhat`. The exact source-zero certificate has all 3,660 genuine `(D,p)=(2,1)`
PBW slices outside the physical Eq. 3.1g image, with first residual `-1/56`.

## Governing hypothesis

`Hhat` is a semi-prepotential, not an arbitrary 320-component source. The
common parent proposed in the audited literature is the unconstrained spinor
prepotential itself. Its scalar contraction is a distinguished subroute:

```text
Psi_alpha --> Hhat,
Psi_alpha --> V = D^alpha Psi_alpha --> Hhat.
```

The scalar-factorizing local bridge has already failed its complete
`p D_[13]` correction test and is retained only as a negative-control line.
It must not be substituted for the full parent space. The repository contains
twelve exact direct level-16 leading embeddings, seven level-17 hook
embeddings, forty-four first-momentum maps, the declared seventy-seven
second-momentum maps, PBW normal form, B5 character machinery, and the
higher-momentum response infrastructure needed to test the direct hypothesis.

## Phase 1: derive the allowed Hhat image

Construct the complete filtered equivariant family

```text
P_H: J(Psi) --> Hhat
```

in the existing Cartesian Majorana basis. Begin with the exact direct-spinor
`D^16 Psi` and `p D^14 Psi` bases and continue through every lower symbol at
the declared engineering weight. Completeness must come from exact B5
multiplicities and an exact independent generator rank, not from a selected
list of gamma diagrams. The scalar-factorizing direction is tagged inside the
direct basis and is never used as the whole candidate space.

Let `O` be the certified physical-image obstruction

```text
O = (1 - P_physical) D(d Psi3).
```

The obstruction `O` is a valid negative-control functional for any candidate
that claims the physical three-form factors through the rejected Eq. 40
`Hhat` ray. It is not a valid constraint on a direct parent map `P_A` that does
not factor through `Hhat`.

The first decisive common-parent calculation is the coupled kernel

```text
kernel C(P_H, P_h, P_A, P_psi),
```

where `C` contains the component-curvature, supersymmetry, and source-gauge
descent equations. Run the exact `-1/56` witness first only on the
`Hhat`-factorizing `P_A` subblock. Only coupled survivors advance to complete
physical-image replay.

### Phase 1 outcomes

1. `nullity = 0`: reject the declared direct-spinor common-parent filtration.
2. Nonzero nullity but zero physical rank: reject an overconstrained solution.
3. Nonzero nullity and nonzero physical rank: define the allowed source as
   `im(P_H)` and continue.

No constraint is accepted merely because it kills the recorded witness.

## Phase 2: construct all component descendants from the same parent

Do not require the physical three-form to factor through `Hhat`. Construct

```text
P_h:   J(Psi) --> h
P_A:   J(Psi) --> A3
P_psi: J(Psi) --> psi
P_H:   J(Psi) --> Hhat
```

at the engineering degrees fixed by the source inventory. Enumerate every
equivariant map at each declared bidegree. Each generated Cartesian basis must
pass exact Lorentz equivariance, expected multiplicity rank, PBW typing, and
mutation gates.

The first leading blocks are already fixed by the exact spinor-prepotential
inventory:

| map | parent level | target | leading multiplicity |
|---|---:|---|---:|
| `P_H` | 16 | `(10001)` gamma-traceless vector-spinor | 12 |
| `P_h` | 17 | `(20000)` conformal graviton | 2 |
| `P_A` | 17 | `(00100)` three-form | 8 |
| `P_psi` | 18 | `(10001)` conformal gravitino | 8 |

These are candidate multiplicities, not physical copies. The coupled chain
equations must select their relative combinations and all required lower
symbols.

The complete filtered representation inventory is now frozen. Its canonical
manifest contains 1,386 columns:

| block | total columns | counts by lower-symbol order `q` |
|---|---:|---|
| `P_H` | 386 | `12, 44, 77, 100, 81, 41, 21, 9, 1` |
| `P_h` | 132 | `2, 14, 24, 31, 30, 17, 8, 4, 2` |
| `P_A` | 268 | `8, 30, 49, 64, 58, 32, 16, 9, 2` |
| `P_psi` | 600 | `8, 57, 109, 136, 133, 85, 41, 21, 9, 1` |

Every multiplicity has two independent exact character checks. The canonical
manifest SHA-256 is
`38ba66b5f90a2938706c9f68b9cc1cd969b60b407655e0643e33dbeb411f36cb`.
This closes representation counting, not Cartesian construction.

The existing bounded calculations are now explicitly negative controls:

- the scalar-factorizing bridge has correction rank 2 and augmented rank 3;
- the `P_H` `q=0,1` family has rank 56 and nullity 0;
- the `P_H` `q=2` family has rank 77 and nullity 0.

They do not test the 1,386-column direct common-parent family. The immediate
construction gap is 253 `P_H` Cartesian emitters at `q >= 3` and 1,000 coupled
`P_h`, `P_A`, and `P_psi` emitters.

## Phase 3: solve the coupled chain-map equations

The frozen component `F` and `K` maps are acceptance targets, not adjustable
fit data.

### Curvature compatibility

The component descendants must land in the certified physical complex:

```text
F_component P_component.
```

### Supersymmetry compatibility

For every component sector require

```text
D P_i = Q_ij P_j + K_component S_i.
```

A discrepancy is permitted only in the exact component gauge image.

### Source-gauge descent

For the six exact source gauge maps `G_q`, solve

```text
P_component G_q = K_component R_q
```

with typed routing maps `R_q`. This determines which combinations of the six
independent source parameter domains become component diffeomorphisms,
two-form gauge transformations, and local supersymmetry. The six source maps
must never be treated as six scalar coefficients of target `K`.

The coupled coefficient system is solved over three admissible finite fields.
Any survivor requires exact reconstruction over `Q(i)` and complete all-row
replay.

## Phase 4: derive the prepotential K map

After the source routing is fixed, construct

```text
K_pre = P_H G_source R
```

as the map from the physical target gauge parameters into `Hhat`. Require a
bound exact identity

```text
F_pre K_pre = 0
```

over the full eleven-momentum polynomial ring. A sampled-momentum identity is
not sufficient.

## Phase 5: extract the Hhat differential constraints

Compute the relations defining `im(P_H)`:

```text
C_H Hhat = 0,
C_H P_H = 0.
```

These relations, rather than `O(Hhat)=0` imposed by definition, are the
derived semi-prepotential constraints. Publish:

- homogeneous generators and derivative degrees;
- Spin(1,10) representations;
- reducibility maps;
- Hilbert series and Hilbert polynomial;
- graded Betti table and regularity bound;
- generic, null-momentum, and torsion ranks;
- deterministic quotient normal forms.

## Final acceptance gate

The common-parent construction passes only if:

1. the full coupled compatibility operator annihilates the selected parent
   combination exactly;
2. every `Hhat`-factorizing three-form subblock satisfies `O P_H = 0` exactly;
3. the physical `G4` image remains nonzero;
4. the Riemann, four-form, and gravitino Bianchi identities vanish;
5. all source gauge maps descend through physical component `K`;
6. `F_pre K_pre = 0` over formal momentum;
7. the null-momentum cohomology is exactly `44 + 84 | 128`;
8. no additional physical states survive;
9. every modular survivor replays exactly over `Q(i)`;
10. source, basis, operator, artifact, and row-dictionary hashes reproduce.

## Rejected shortcuts

Do not:

- impose the `-1/56` obstruction as a source equation by definition;
- identify `Hhat` directly with the physical component gravitino;
- reuse unrestricted Eq. 25 as a physical component extractor;
- add the full-H gamma-trace ray after `P320`, where it vanishes;
- use higher-bidegree maps to cancel support-disjoint rows without a proved
  source differential relation;
- identify the six source `G_q` domains with target gauge parameters without
  exact routing;
- infer a polynomial identity from one momentum specialization.

## Execution order

1. Freeze the physical component `F` and `K` certificates.
2. Export the complete direct-spinor `P_H` coefficient basis, including all
   lower symbols at the declared engineering weight.
3. Build the parallel `P_h`, `P_A`, and `P_psi` bases from the same parent.
4. Apply the `-1/56` witness only to the `Hhat`-factorizing `P_A` subblock.
5. Assemble the coupled common-parent compatibility matrix on GPU.
6. Solve the coupled supersymmetry and gauge-descent system.
7. Derive `K_pre`, `C_H`, reducibility, and quotient normal forms.
8. Verify formal-momentum closure and `44 + 84 | 128` cohomology.

## Compute architecture

This project is GPU-first from the first implementation. Every operation whose
cost grows with Cartesian coordinates, jet order, candidate count, row count,
or finite-field width must be implemented on CUDA before production launch.

The GPU owns:

- parent-map Cartesian evaluation;
- PBW traversal and normal-form stream construction;
- obstruction and component-curvature composition;
- sparse emission, sorting, reduction, and cancellation;
- simultaneous three-prime rank, kernel, and retained-pivot calculations;
- coupled supersymmetry and source-gauge descent systems;
- formal-momentum closure batches;
- constraint, syzygy, and reducibility rank screens;
- complete residual replay over the support union.

Use persistent device contexts, compressed signed-permutation Clifford tables,
device-resident sparse operators, fused kernels, asynchronous double buffering,
bounded-memory batches, and retained witness pivots. Upload immutable bases and
operators once. Do not insert host round trips between PBW stages, projector
stages, or finite-field reductions. Never materialize a dense ambient
jet-to-component map.

CPU work is restricted to small proof-control tasks: exact character counts,
canonical grammar and manifest generation, immutable hashes, denominator
audits, exact `Q(i)` reconstruction of retained minors, and final certificate
publication. Every CPU oracle must have a CUDA parity canary before production
data are accepted.

Every production kernel must expose five-second heartbeats, batch progress,
throughput, nonzero counts, retained ranks, first failing witness, VRAM
resident and high-water bytes, checkpoint identity, and deterministic stream
hashes. No long opaque run is permitted.

## Current boundary

The physical component `F` and `K` complexes are complete. The unrestricted
same-weight `Hhat -> A3` source identification is exhausted and fails. This
plan tests whether a shared unconstrained parent restricts `Hhat` to a proper
subspace and simultaneously produces the physical component fields. Until the
coupled chain-map and source-gauge descent gates pass, no prepotential-level
`F`, prepotential-level `K`, or irreducibility theorem is claimed.

The GPU substrate is manifest-driven and variable-width. Its first acceptance
fixture is the 30-column leading family `12 + 2 + 8 + 8`; its production input
is the frozen 1,386-column manifest after every Cartesian/PBW emitter is
certified. The engine must fail closed while any component block lacks explicit
source kernels, target Clebsches, or exact equivariance gates.
