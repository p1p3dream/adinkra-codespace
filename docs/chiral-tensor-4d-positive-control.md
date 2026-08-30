# Four-dimensional chiral-tensor positive control

## Result

The four-dimensional chiral-tensor system in arXiv:1405.0048 has been
reproduced with exact arithmetic. The calculation transcribes Eqs. (44)-(47),
checks the algebra stated in Eq. (50), applies the temporal gauge and field map
in Eqs. (52)-(53), and recovers the published `CT` eight-color matrices exactly.

This is a reproduction of a known off-shell construction. It is not a new
four-dimensional representation.

## Source conventions

The calculation uses the Majorana gamma-matrix conventions in Appendix A of
arXiv:0902.3830. It constructs the matrices over the Gaussian rationals and
verifies the Clifford relation exactly. The mostly-plus metric and
`epsilon_0123 = -1` are retained explicitly. No floating-point tolerance is
used.

Local source hashes:

| Source | SHA256 |
|---|---|
| arXiv:1405.0048 PDF | `8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20` |
| arXiv:0902.3830 PDF | `d55b80ef20e53f74ed8e4158ea6ce5db0ba4ff71f9bc276b6ca24bdf9174b385` |

## Component closure

The raw component system contains 19 fields:

- the chiral fields `A`, `B`, `F`, and `G`;
- the tensor scalar `phi`;
- the six independent components of `B_mu_nu`;
- four components of `psi`; and
- four components of `chi`.

There are eight real supercharges and 36 unordered charge pairs. The exact
calculation checks every pair on every field:

```text
19 fields x 36 charge pairs = 684 component relations
```

| Quantity | Count |
|---|---:|
| nongauge-field relations | 468 |
| two-form-potential relations | 216 |
| total relations | 684 |
| relations with a residual | 0 |
| residual terms | 0 |

For the two-form potential, the verifier compares against the complete right
side of Eq. (50): the tensor field-strength term plus the mixed-supersymmetry
gauge term. The gauge term is nonzero and required. It is not removed during
the four-dimensional check.

## Reduction to the eight-color matrices

Only after component closure passes does the calculation impose the source
reduction:

- all spatial derivatives are set to zero;
- `B_0i = 0`;
- the eight fermions are ordered as the four components of `psi`, followed by
  the four components of `chi`; and
- the bosonic node map is Eq. (53), including `2 B_12`, `2 B_23`, and
  `2 B_31`.

The derived eight `8 x 8` matrices are compared entry by entry with the
committed `CT` fixture from arXiv:2012.14015. All 512 entries agree exactly.

## Reproduction

```sh
cargo run --release -- chiral-tensor-4d-build
cargo run --release -- chiral-tensor-4d-verify
cargo test --bin adinkra-codespace chiral_tensor_4d
node scripts/test_chiral_tensor_4d.mjs
```

The JavaScript cross-check separately constructs the source gamma matrices,
transcribes the component rules, verifies all 684 relations, performs the
worldline reduction, and compares all 512 reduced entries. It does not read
the Rust closure flags.

Artifacts:

- `data/chiral_tensor_4d.json`
- `results/chiral_tensor_4d_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `58a4afc187acfd9b2dcac1822333b4bf0fb7d76c2d59fec9d8483cd794660fbe` |
| validation | `0017f109e3145a94a68c058cd1957b200a51494035df6ae68ab71fa158385879` |

## Consequence

The `CV` and `CT` positive controls now both have verified four-dimensional
parents and verified reductions. The next pass is to compare the spatial
linkage and gauge data that distinguish them before reduction.

## Boundary

This result establishes the published linear abelian chiral-tensor algebra
and its reduction. It does not assign a higher-dimensional parent to `VM1`,
`VM2`, or `VM3`, and it does not show that a one-dimensional signed class
determines four-dimensional physics.
