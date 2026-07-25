# Six supersymmetric sectors of the four-color permutahedron

## Result

The 24 vertices of the `S4` permutahedron separate into the six published
quartets `P1` through `P6`. Each quartet is one coset of the normal Klein
four-group `V4` in `S4`.

The Rust calculation:

1. enumerates all 24 permutations;
2. verifies that the six quartets cover them once;
3. checks that each quartet is a `V4` coset;
4. attaches all 16 Boolean-factor quartets printed for each sector in Appendix
   B of Ref. [1];
5. constructs the resulting 96 signed `L`-matrix quartets;
6. verifies Garden closure by signed-permutation algebra and by an independent
   dense-entry calculation;
7. constructs an explicit valise Adinkra for every signed quartet;
8. checks the odd-dashing condition on every two-color square; and
9. records exact and signed invariants for recognizing the sectors.

All checks pass.

| Check | Result |
|---|---:|
| permutation vertices | 24 |
| disjoint quartets | 6 |
| published Boolean-factor quartets | 96 |
| Eq. (5.10) source matrices | exact match |
| Garden-closing signed quartets | 96 |
| dense Garden entries checked | 24,576 |
| dense residual entries | 0 |
| explicit Adinkras | 96 |
| Adinkra edges checked | 1,536 |
| two-color squares checked | 1,152 |
| odd-dashing failures | 0 |
| intra-quartet spectrum | `(12, 0, -4, -8)` for all six |
| intra-quartet eigenvalue norm squared | 224 |
| largest inter-quartet eigenvalue norm squared | 208 |

## Matrix convention

A permutation is stored in one-line form

```text
p = [p(1), p(2), p(3), p(4)].
```

Its permutation matrix has its nonzero entry in row `r`, column `p(r)`. A
Boolean factor `b` is a four-bit integer. Bit zero controls row one:

```text
S(b) = diag((-1)^bit_0(b), ..., (-1)^bit_3(b)).
```

The signed matrices are

```text
L_I = S(b_I) P_I,
R_I = L_I^T.
```

This convention gives

```text
S(10) = diag(+1, -1, +1, -1),
S(12) = diag(+1, +1, -1, -1),
S(6)  = diag(+1, -1, -1, +1),
S(0)  = diag(+1, +1, +1, +1),
```

matching the matrix example in Eqs. (5.8)-(5.10) of Ref. [1].

## Garden verification

For each of the 96 signed quartets, the program checks

```text
L_I R_J + L_J R_I = 2 delta_IJ I_4
```

in two ways:

- sparse signed-permutation composition; and
- direct multiplication of every entry of every matrix relation.

The second calculation checks `96 x 4 x 4 x 4 x 4 = 24,576` entries. Every
residual is zero.

## Adinkras

Each stored representative is a valise graph with:

- four bosons;
- four fermions;
- four colors;
- one perfect matching per color;
- 16 colored edges; and
- 12 two-color squares.

A positive matrix entry produces a solid edge. A negative entry produces a
dashed edge. Every two-color square contains an odd number of dashed edges.

The JSON stores the matrices, nodes, edges, signs, and squares for all 96
published fiducial signings. The viewer can display every one.

## What distinguishes the six sectors

The exact unsigned sector is its `V4` coset. In the fixed one-line convention,
the quotient

```text
S4 / V4 isomorphic to S3.
```

Conjugation by any member of a quartet permutes the three nonidentity elements
of `V4`. Elements from the same quartet induce the same permutation, while the
six quartets induce all six elements of `S3`. Once those three elements are
ordered, this gives a distinct, convention-fixed coordinate for each unsigned
sector. Relabeling the underlying objects can conjugate the displayed `S3`
coordinate, so the table does not claim that the written cycle is independent
of every relabeling.

| Sector | `S3` quotient action | ordered Bruhat word | canonical `chi0` |
|---|---|---|---:|
| `P1` | `(132)` | `4,6,2,2,6,4` | `+1` |
| `P2` | `(123)` | `6,2,4,4,2,6` | `-1` |
| `P3` | `(12)` | `4,2,6,6,2,4` | `-1` |
| `P4` | `(13)` | `6,4,2,2,4,6` | `+1` |
| `P5` | `(23)` | `2,6,4,4,6,2` | `+1` |
| `P6` | `()` | `2,4,6,6,4,2` | `-1` |

The ordered Bruhat word is the upper triangle of the published intrasection
distance matrix in its fixed `L1,L2,L3,L4` order. It makes that published
ordering recognizable, but it is not invariant under an arbitrary reordering
of the four colors.

All six intra-quartet matrices have eigenvalues `(12, 0, -4, -8)`. Their
eigenvalue-vector norm squared is 224. The largest value among the 15 published
inter-quartet blocks is 208. This reproduces the criterion in Ref. [2] that the
six supersymmetry quartets attain the largest spectral length. It recognizes
the six valid quartets collectively; it does not distinguish one of the six
from another.

`chi0` and the gadget row describe the selected signed representative. They do
not provide the six-way unsigned partition. Boolean choices can change the
`chi0` sign inside a fixed unsigned sector.

## Full four-color library

The 96 Appendix-B quartets are fiducial signed seeds. Independent color
complements and color permutations generate the complete ordered library:

```text
6 sectors x 16 fiducial signings x 16 color complements x 24 color orders
= 36,864.
```

The artifact stores the 96 seeds and one graph per sector rather than
serializing 36,864 generated variants.

## Reproduction

```sh
cargo run --release -- perm-s4-susy-build
cargo run --release -- perm-s4-susy-verify
node scripts/test_permutahedron_s4_supersymmetry.mjs
```

Artifacts:

- `data/permutahedron_s4_supersymmetry.json`
- `results/permutahedron_s4_supersymmetry_validation.json`
- `visualizer/permutahedron_s4_supersymmetry.html`

| Generated artifact | SHA256 |
|---|---|
| `data/permutahedron_s4_supersymmetry.json` | `7bc8552824d4da794814dfd89ec314a7ceab77d5df343680a64f084e7c19c2cb` |
| `results/permutahedron_s4_supersymmetry_validation.json` | `5fca0190fe3f6601dcd2191fc3ee3bff31ea4660310103c821541434a65be59e` |

Serve the repository root over HTTP, then open the HTML file to inspect all six
graphs, signed matrices, Boolean factors, and sector labels.

## Boundary

This calculation verifies the six unsigned sectors and the 96 published
fiducial signed quartets. It does not establish that the six quotient elements
are six inequivalent four-dimensional supermultiplets. It also does not derive
a higher-dimensional field equation. It supplies a complete, reproducible
four-color base case for the larger permutahedron program.

## Primary references

1. S. J. Gates Jr., F. Guyton, S. Harmalkar, D. S. Kessler, V. Korotkikh, and
   V. A. Meszaros, "Adinkras From Ordered Quartets of BC4 Coxeter Group
   Elements and Regarding 1,358,954,496 Matrix Elements of the Gadget,"
   [arXiv:1701.00304](https://arxiv.org/abs/1701.00304).
2. A. J. Cianciara, S. J. Gates Jr., Y. Hu, and R. Kirk, "The 300
   'Correlators' Suggests 4D, N=1 SUSY Is a Solution to a Set of Sudoku
   Puzzles," [arXiv:2012.13308](https://arxiv.org/abs/2012.13308).
3. I. Chappell II, S. J. Gates Jr., and T. Hübsch, "Adinkra (In)equivalence
   From Coxeter Group Representations: A Case Study,"
   [arXiv:1210.0478](https://arxiv.org/abs/1210.0478).
4. S. J. Gates, K. Iga, L. Kang, V. Korotkikh, and K. Stiffler, "Generating
   All 36,864 Four-Color Adinkras via Signed Permutations and Organizing Into
   ell- and tilde-ell-Equivalence Classes,"
   [arXiv:1712.07826](https://arxiv.org/abs/1712.07826).
