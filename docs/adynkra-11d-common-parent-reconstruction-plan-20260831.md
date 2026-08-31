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
common parent proposed in the audited literature is an unconstrained spinor
prepotential and its scalar contraction:

```text
Psi_alpha --> V = D^alpha Psi_alpha --> Hhat.
```

The repository already contains the exact level-15 scalar bridge, twelve
level-16 source embeddings, seven level-17 embeddings, PBW normal form, B5
character machinery, and the higher-momentum response infrastructure needed
to test this hypothesis.

## Phase 1: derive the allowed Hhat image

Construct the complete equivariant family

```text
P_H: J^15(V) --> Hhat
```

in the existing Cartesian Majorana basis. Completeness must come from exact
B5 multiplicities and an exact independent generator rank, not from a selected
list of gamma diagrams.

Let `O` be the certified physical-image obstruction

```text
O = (1 - P_physical) D(d Psi3).
```

The first decisive calculation is

```text
kernel(O P_H).
```

Run the exact `-1/56` witness functional first. Only surviving combinations
advance to the complete physical-image replay.

### Phase 1 outcomes

1. `nullity = 0`: reject the current scalar/spinor common-parent realization.
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

1. `O P_H = 0` exactly;
2. the physical `G4` image remains nonzero;
3. the Riemann, four-form, and gravitino Bianchi identities vanish;
4. all source gauge maps descend through physical component `K`;
5. `F_pre K_pre = 0` over formal momentum;
6. the null-momentum cohomology is exactly `44 + 84 | 128`;
7. no additional physical states survive;
8. every modular survivor replays exactly over `Q(i)`;
9. source, basis, operator, artifact, and row-dictionary hashes reproduce.

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
2. Export the complete level-15/16 `P_H` coefficient basis.
3. Run the `-1/56` witness matrix for `O P_H`.
4. Replay the complete obstruction only for witness survivors.
5. Build the parallel `P_h`, `P_A`, and `P_psi` bases.
6. Solve the coupled supersymmetry and gauge-descent system.
7. Derive `K_pre`, `C_H`, reducibility, and quotient normal forms.
8. Verify formal-momentum closure and `44 + 84 | 128` cohomology.

## Compute architecture

Use CPU exact character and highest-weight machinery for completeness and
source provenance. Use CUDA for Cartesian diagram evaluation, PBW stream
construction, sparse compaction, and simultaneous three-prime rank screens
when measured output volume warrants it. Keep canonical pivot minors and final
`Q(i)` replay on CPU. Never materialize a dense ambient jet-to-component map.

## Current boundary

The physical component `F` and `K` complexes are complete. The unrestricted
same-weight `Hhat -> A3` source identification is exhausted and fails. This
plan tests whether a shared unconstrained parent restricts `Hhat` to a proper
subspace and simultaneously produces the physical component fields. Until the
coupled chain-map and source-gauge descent gates pass, no prepotential-level
`F`, prepotential-level `K`, or irreducibility theorem is claimed.
