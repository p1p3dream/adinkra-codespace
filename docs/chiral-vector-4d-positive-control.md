# Four-dimensional chiral-vector positive control

## Result

The four-dimensional chiral-vector system in arXiv:1405.0048 has been
reproduced with exact arithmetic. The calculation transcribes Eqs. (32)-(35),
checks the algebra stated in Eq. (38), applies the temporal gauge and field map
in Eqs. (40)-(41), and recovers the published `CV` eight-color matrices exactly.

This is a reproduction of a known off-shell construction. It is not a new
four-dimensional representation.

## Source conventions

The component paper refers to the gamma-matrix conventions of
arXiv:0902.3830. Appendix A of that paper fixes:

- the mostly-plus metric;
- the four Majorana gamma matrices in Eq. (A.3);
- `gamma_5` in Eq. (A.4);
- the charge-conjugation matrix in Eq. (A.6); and
- the index-lowering identities in Eqs. (A.8)-(A.11).

The implementation constructs these matrices over the Gaussian rationals and
verifies the Clifford relation exactly. No floating-point tolerance is used.

Local source hashes:

| Source | SHA256 |
|---|---|
| arXiv:1405.0048 PDF | `8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20` |
| arXiv:0902.3830 PDF | `d55b80ef20e53f74ed8e4158ea6ce5db0ba4ff71f9bc276b6ca24bdf9174b385` |

## Component closure

The raw component system contains 17 fields:

- four chiral bosons and auxiliaries: `A`, `B`, `F`, and `G`;
- the four components of the vector potential `A_mu`;
- the vector auxiliary `d`;
- four components of `psi`; and
- four components of `lambda`.

There are eight real supercharges and 36 unordered charge pairs. The exact
calculation checks every pair on every field:

```text
17 fields x 36 charge pairs = 612 component relations
```

The result is:

| Quantity | Count |
|---|---:|
| nongauge-field relations | 468 |
| vector-potential relations | 144 |
| total relations | 612 |
| relations with a residual | 0 |
| residual terms | 0 |

For the vector potential, the verifier compares against the complete right
side of Eq. (38): the field-strength translation term plus the mixed
supersymmetry gauge term. The mixed gauge term is nonzero and required. It is
not discarded by gauge fixing during the four-dimensional check.

## Reduction to the eight-color matrices

Only after the component algebra passes does the calculation impose the source
reduction:

- all spatial derivatives are set to zero;
- `A_0 = 0`;
- the eight fermions are ordered as the four components of `psi`, followed by
  the four components of `lambda`; and
- the bosonic node map is the one printed in Eq. (41), including the raised
  auxiliary nodes.

The derived eight `8 x 8` matrices are compared entry by entry with the
published `CV` fixture from arXiv:2012.14015. All 512 entries agree exactly.

This supplies the first positive-control bridge between a complete
four-dimensional gauge multiplet and the signed S8 recursion artifact.

## Reproduction

```sh
cargo run --release -- chiral-vector-4d-build
cargo run --release -- chiral-vector-4d-verify
cargo test --bin adinkra-codespace chiral_vector_4d
node scripts/test_chiral_vector_4d.mjs
```

The JavaScript check separately constructs the source gamma matrices,
transcribes the component rules, verifies all 612 relations, performs the
worldline reduction, and compares all 512 reduced entries. It does not read
the Rust closure flags.

Artifacts:

- `data/chiral_vector_4d.json`
- `results/chiral_vector_4d_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `af4b949a5c2256aef8e6bc7a68c133f2d43e2a6c050d3ad83d714442ce0be22a` |
| validation | `7b9f7d6ca1cd7af92a6dbc975b0ba869db67a8f8ae0a332176a4c9d9c28f9a07` |

## Consequence

The known `CV` parent now supplies a validated higher-dimensional reference
object. Its one-dimensional matrix class is not being treated as evidence of
parentage. The verified spatial derivative terms, field strength, gauge
residue, temporal gauge, and field-to-node map are the parentage data.

The next pass is the chiral-tensor positive control from Eqs. (44)-(53) and
Appendix C of arXiv:1405.0048. Its two-form gauge residue must be retained and
its reduction must recover the committed `CT` matrices exactly.

## Boundary

This result establishes the published linear abelian chiral-vector algebra and
its reduction. It does not assign a higher-dimensional parent to `VM1`, `VM2`,
or `VM3`, and it does not show that the one-dimensional signed class determines
four-dimensional physics.
