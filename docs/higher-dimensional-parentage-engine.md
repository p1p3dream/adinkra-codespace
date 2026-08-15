# Higher-dimensional parentage engine

## Purpose

An unsigned worldline support does not determine a four-dimensional parent. In
particular, the same S8 support can carry an ordinary Garden signing or a
one-generator central extension. Parentage therefore has to retain physical
data that the worldline graph forgets.

This implementation adds a bounded, evidence-aware classifier for the current
four-dimensional fixture catalog. It does not claim a complete classification
of all off-shell multiplets.

## Phase 1: physical fingerprints

The physical fingerprint retains:

- boson and fermion counts
- Lorentz representation and engineering level by field block
- physical, auxiliary, and gauge-potential roles
- differential-form degree, gauge parameter degree, and reducibility depth
- field-strength and Bianchi degrees
- temporal and spatial derivative linkage
- algebraic auxiliary multiplicity
- local Stueckelberg generators separately from physical central generators
- bosonic and fermionic central rank, centrality, and involutivity
- fixture certification and unresolved closure residues

The aggregate catalog key is invariant under arbitrary invertible basis changes
inside a fixed statistics, Lorentz, role, and engineering-level block. It is an
exact discriminant for the retained catalog, not a complete invariant for an
arbitrary tuple of derivative matrices.

`higher_dimensional_canonical` also provides an exact presentation-level
canonicalizer. It retains component linkage, gauge arrows, Bianchi identities,
and central actions and enumerates an explicitly declared finite group of
type-preserving signed permutations. It fails closed if the group orbit exceeds
the configured bound. Its canonical hash must only be described as canonical
under that declared finite group, not under unrestricted continuous field
redefinitions.

The older `higher_dimensional_fingerprint` hashes remain useful reproducibility
identifiers in their source bases. They are not used as basis-independent
parentage proofs.

## Phase 2: validated catalog

The catalog contains four entries with different evidence levels:

| Entry | Result | Evidence boundary |
| --- | --- | --- |
| Chiral-vector, CV | Exact fixture | 612 of 612 component closure relations, including one-form gauge residues |
| Chiral-tensor, CT | Exact fixture | 684 of 684 component closure relations, including reducible two-form gauge residues |
| Vector-tensor, VT-one-Z | Qualified fixture | 720 component relations and the worldline one-Z extension pass separately; the term-by-term source-normalization bridge remains partial |
| Scalar-tensor regular tangent | Structural preflight | Regular 8+8 tangent is CT-compatible, but full component closure and an exact four-dimensional intertwiner have not been solved |

The exact CV versus CT rejection is already visible in the gauge complex. CV
has an irreducible one-form gauge potential, while CT has a reducible two-form
gauge potential. VT is separated by simultaneous one-form and two-form content
plus its physical rank-one central facet. The scalar-tensor tangent shares the
CT linear aggregate key but has a distinct nonlinear completion key.

Mutation controls verify that the classifier rejects:

1. replacing the CV one-form complex with a two-form complex
2. erasing the VT physical central generator
3. lowering CT auxiliary scalars to the physical engineering level

## Phase 3: inference

Inference returns one of four decisions:

- `identified`: one exact catalog fixture is selected by complete supplied data
- `compatible`: one or more fixtures match, but the selected evidence is
  qualified or incomplete
- `insufficient`: required physical decorations are absent
- `unsupported`: no retained fixture matches, with a first mismatch witness for
  every rejected candidate

A worldline-only 8+8 query is intentionally insufficient. Complete CV and CT
queries identify their exact fixtures. VT returns qualified compatibility until
the central source-normalization bridge is complete. The scalar-tensor tangent
returns qualified compatibility, never a proof of CT equivalence.

These decisions are relative to the checked-in catalog. A unique result means
"unique among catalog entries," not "unique among all possible multiplets."

## Commands

Build and verify the checked-in audit artifact:

```bash
cargo run --release -- higher-dimensional-parentage-build
cargo run --release -- higher-dimensional-parentage-verify
```

Classify a JSON query:

```bash
cargo run --release -- higher-dimensional-parentage-query query.json
```

Minimal worldline-only query:

```json
{
  "worldline_size": [8, 8]
}
```

The generated audit is
`results/higher_dimensional_parentage.json`. It records the catalog keys,
certification levels, positive queries, ambiguity checks, rejection witnesses,
and mutation controls.

## Remaining research boundary

The next rigorous extension is not another unsigned spectral test. It is a
source-convention bridge that transports the exact four-dimensional VT central
action to the extracted worldline generator term by term. After that, the
presentation-level canonicalizer can be populated from exact Gaussian-rational
linkage adapters for CV, CT, and VT and compared under explicitly generated
finite basis groups. Arbitrary continuous equivalence of full derivative-matrix
tuples remains a separate algebraic problem.
