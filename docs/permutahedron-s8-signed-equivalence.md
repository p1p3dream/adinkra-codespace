# S8 relative-color scan and signed equivalence

## Question

The fixed-order signed recursion recovered the published `CT` system but not
the printed `CV` system. This pass tests whether the difference comes from the
relative order of the four colors in the second input, then classifies every
closing result under explicit signed equivalence relations.

## Source basis

- arXiv:2304.09830, Eqs. (2.17)-(2.25), supplies the block recursion,
  Boolean-factor rule, cyclic four-bit flips, and Garden acceptance test.
- arXiv:2012.13308 retains the color order as data and discusses the 24 color
  orderings.
- arXiv:1712.07826, Eq. (2.33), gives the fixed-color nodal relation
  `L'_I = X L_I Y`, with signed permutation matrices acting on the two node
  levels.
- `HowardTLK.v2.pdf`, pp. 61, 64, 73, 77, and 80-81, supplies the six-quartet
  geometric base case, the 24-node permutahedron, and the hopping construction
  that motivates the higher-color separation problem.

## Exhaustive scan

The Rust calculation tests:

```text
30 ordered distinct source pairs
x 24 relative color orders for the second source
x 8 cyclic four-bit masks
= 5,760 signed candidates
```

Exactly 24 candidates close. They occupy 12 pair and alignment
configurations, with two complementary masks in every configuration.

The fixed-order results remain present. Three additional relative alignments
close:

- `CM -> VM` and `VM -> CM`, second color order `2143`, mask starts 4 and 8;
- `VM1 -> VM3` and `VM3 -> VM1`, second color order `3412`, mask starts 4 and
  8.

The other closing configurations use order `1234` and mask starts 2 and 6.

## Published anchors

Two outputs reproduce printed signed systems exactly:

| system | ordered pair | second color order | mask start |
|---|---|---:|---:|
| `CT` | `CM -> TM` | `1234` | 6 |
| `CV` | `CM -> VM` | `2143` | 4 |

The `CV` discrepancy in the fixed-order pass was therefore a relative color
alignment issue, not a failure of the recursion.

Every closing candidate has commutant dimension one and antisymmetric
commutant dimension zero. Thus every result is an irreducible real `8|8`
Garden representation with scalar commutant.

## Exact signed equivalence

For two signed systems `A` and `B`, the verifier searches for explicit maps
of the eight boson nodes, eight fermion nodes, and, where allowed, the eight
colors. It then solves the 64 edge-sign equations over `GF(2)`. A serialized
witness is accepted only if it reproduces the target endpoint and sign on all
64 colored edges.

The root image and color map determine the unsigned node maps. This makes the
search complete for these connected minimal systems. The sign equations then
decide whether the unsigned map lifts to a signed one.

Four relations are kept separate:

1. fixed-color nodal `BC8` equivalence;
2. fixed-color nodal equivalence plus independent supercharge signs;
3. full unlabeled-color signed graph isomorphism;
4. the preceding relation plus interchange of the boson and fermion levels.

All 24 closing candidates already form one class under the first, narrowest
relation. The broader three relations therefore also produce one class.

This result has a direct interpretation. The 24 closing outputs are different
signed matrix presentations inside this finite scan, but they do not define
24 inequivalent fixed-color one-dimensional Adinkras under signed node flips
and flops. In particular, the `CT` and `CV` anchors are nodally equivalent
after reduction to the minimal one-dimensional system. Their distinct
higher-dimensional labels are not recovered by Garden closure and fixed-color
nodal classification alone.

This failure is not limited to the `CT` and `CV` names. Exactly 12 closing
candidates are built from pairs among `CM`, `TM`, and `VM`, whose sources are
Carroll reductions of named four-dimensional multiplets. The other 12 are
built from pairs among `VM1`, `VM2`, and `VM3`, for which the source states no
four-dimensional parent. All 24 occupy the same fixed-color nodal class.
Therefore no invariant of that one-dimensional equivalence class can recover
the source-parentage distinction. This does not show that the latter 12 lack
a higher-dimensional parent. It shows that parentage is not determined by the
one-dimensional signed class.

## Consequence for the Gadget test

The result narrows the selection-rule question. A raw cross-Gadget matrix can
change when representatives are independently changed inside the same nodal
class. Therefore an orthonormal six-tuple, if found, is first a distinguished
joint choice of representatives. It becomes evidence for a physical
selection rule only if it is rare, stable under the declared equivalences,
and predicts the named closing systems without being fitted to them.

That calculation is complete in
`docs/permutahedron-s4-gadget-frame-census.md`. Among the complete fixed-order
Boolean library, 28,862,180,229,120 of 281,474,976,710,656 six-frames are
orthonormal, a fraction of `105/1024`. Orthonormality is therefore a common
joint convention in this library rather than a unique or rare selector.

## Reproduction

```sh
cargo run --release -- perm-s8-equivalence-build
cargo run --release -- perm-s8-equivalence-verify
cargo test --bin adinkra-codespace permutahedron_s8_signed
node scripts/test_permutahedron_s8_signed_equivalence.mjs
```

Artifacts:

- `data/permutahedron_s8_signed_equivalence.json`
- `results/permutahedron_s8_signed_equivalence_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `a128d7e74669dfbba4f5f00b11f8744d8f132e27af7985eee594db161749bad7` |
| validation | `7dcd0987c7a9ab3df5fc36c04c4a77767f5e8921355a04b2d76759e46029f608` |

The independent JavaScript audit reconstructs all 5,760 candidates from the
six printed signed four-color words, recomputes all closure decisions, checks
the exact `CT` and `CV` anchors, and verifies every serialized equivalence
witness edge by edge.

## Boundary

The class IDs are canonical only within this finite scan. The result does not
identify the nodal quotient with complete physical equivalence, prove
four-dimensional enhancement, or enumerate arbitrary Garden signings outside
the paper's cyclic-flip construction.

## Remaining higher-dimensional gate

The next discriminator cannot be computed from the common valise class alone.
It requires data discarded by one-dimensional reduction:

1. attach the four-dimensional field content, engineering heights, and gauge
   identifications for the published `CT` and `CV` anchors;
2. reconstruct or source the spatial linkage matrices and any gauge or phantom
   corrections;
3. verify the higher-dimensional supersymmetry algebra on those two positive
   controls; and
4. apply the same criterion to the closing systems built from `VM1`, `VM2`,
   and `VM3` without inferring parentage from their one-dimensional names.

`HowardTLK.v2.pdf` fixes the permutation geometry and hopping construction. It
does not supply the spatial linkage or gauge data needed for this gate. Until
those data are specified, the exact stopping statement is that the signed
one-dimensional recursion is irreducible but does not determine
four-dimensional parentage.
