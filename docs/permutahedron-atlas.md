# Permutahedron atlas

## Scope

This atlas implements the finite permutation objects used in the four-color and
eight-color supersymmetry program described in Refs. [1-3]. It contains:

- the complete `S4` permutahedron with 24 vertices and 36 edges;
- the six published `S4` quartets and all 300 independent two-point distances;
- the complete `S8` permutahedron with 40,320 vertices and 141,120 edges;
- the `R8` subgroup and its partition of `S8` into 5,040 right cosets of eight
  vertices each;
- the corresponding left-coset partition and the 168 slices for which the left
  and right cosets agree as sets;
- the six published `N=2` permutation octets `CC`, `CT`, `CV`, `TT`, `TV`, and
  `VV`, together with the `R8` or Diadem octet;
- the complete `56 x 56` Bruhat-distance matrix for those seven named octets.

The atlas treats a two-point correlator as a minimal weak-Bruhat graph distance.
It is not a holoraumy gadget. Signed matrix data and Garden closure are separate
calculations.

## Graph convention

A vertex is a permutation in one-line notation. Vertex identifiers are the
zero-based lexicographic Lehmer ranks:

- `S4-00000` through `S4-00023`;
- `S8-00000` through `S8-40319`.

An edge swaps adjacent positions in a one-line address. This is right
multiplication by one of the adjacent transpositions. The distance between
permutations is the minimum number of these swaps, equivalently their Kendall
distance.

This convention reproduces the `S4` correlator tables in Ref. [1]. Left and
right group multiplication remain separate operations in the implementation.

## Literature reproduction

Reference [1] supplies the complete `S4` vertex dictionary, the six quartets,
and the 21 oriented `4 x 4` blocks that determine the symmetric `24 x 24`
distance matrix. The implementation verifies every published block entry.

Reference [2] supplies the six `S8` representation octets and the Diadem octet.
It states that a complete arrangement of all 40,320 vertices and the requested
two-point matrices remained to be calculated.

Reference [3] supplies the `R8` hopper subgroup, the 5,040-octet partition, the
magic-number value 112, the 168 left-right coincident cosets, and the published
`S8` face counts. The atlas completes the finite graph and named distance-matrix
calculations from those definitions.

Reference [4] was reviewed because it was included in the four papers supplied
for this program. Its subject is unfolded Adinkras. It does not define the
finite `S8` atlas or change the edge and correlator conventions used here.

## Reproduction

The atlas is generated in Rust. From the repository root, run:

```sh
cargo run --release -- perm-atlas-build
cargo run --release -- perm-atlas-verify
```

Generated data:

- `data/permutahedron_s4_atlas.json`
- `data/permutahedron_s8_atlas.json`
- `results/permutahedron_validation.json`

Validated artifact hashes:

| Artifact | SHA256 |
|---|---|
| `data/permutahedron_s4_atlas.json` | `0738927f723ffd16b310ff7a7bcbafa8629b791583524a165a9c1635c008be2c` |
| `data/permutahedron_s8_atlas.json` | `b178d1642d05f1b7e0b9b5b5befdf3fb3c412717b80202c9836fe1bf061f4ba3` |
| `results/permutahedron_validation.json` | `d5328f76691b647f7ee04a06f13397c5fa1ea13fcd114c1af26ef4cdfbdbe74e` |

The validation report records zero mismatches in the 576 entries determined by
the published `S4` blocks. It checks all 40,320 `R8` intra-coset rows at magic
number 112 and all 392 within- and between-octet rows for the seven named `S8`
octets, covering 3,136 named correlator entries, also with zero failures.

The complete `S8` pairwise distance matrix is not stored. It would contain
812,871,360 upper-triangular entries including the diagonal. Distances are
computed from the permutation addresses when requested. The named `56 x 56`
matrix is stored because it is the calculation requested in Ref. [2].

Serve the repository root with a local static-file server and open
`visualizer/permutahedron_atlas.html` to use the browser atlas.

## Boundaries

- `CT`, `CV`, and the Diadem have published Garden-closing sign choices.
- `CC`, `TT`, `TV`, and `VV` have published nonclosure terms. Their unsigned
  permutation octets remain valid atlas records but are not labeled as off-shell
  Garden representations.
- The atlas does not solve the general sign-assignment problem.
- The interpretation of the permutahedron as a supersymmetry weight space is a
  research proposal, not an established representation theorem.
- The reason exactly 168 `S8` cosets have coincident left and right hopper sets
  remains an open structural question in Ref. [3].
- No conclusion about the sought field equation follows from the atlas alone.

## Primary references

1. A. J. Cianciara, S. J. Gates Jr., Y. Hu, and R. Kirk, "The 300
   'Correlators' Suggests 4D, N=1 SUSY Is a Solution to a Set of Sudoku
   Puzzles," [arXiv:2012.13308v6](https://arxiv.org/abs/2012.13308).
2. D. D. Bristow, J. H. Caporaletti, A. J. Cianciara, S. J. Gates Jr., D.
   Levine, and G. Yerger, "A Note On Exemplary Off-Shell Constructions Of 4D,
   N=2 Supersymmetry Representations,"
   [arXiv:2012.14015v7](https://arxiv.org/abs/2012.14015).
3. A. J. Cianciara, Z. Coleman, S. J. Gates Jr., Y. Lee, and Z. Zhang, "N=2
   SUSY & the Hexipentisteriruncicantitruncated 7-Simplex,"
   [arXiv:2304.09830v2](https://arxiv.org/abs/2304.09830).
4. A. J. Cianciara, S. J. Gates Jr., Y. Lee, E. T. Levy, T. O. Razzaz, and J.
   Richardson, "Unfolded Adinkra Properties of Supermultiplets (I),"
   [arXiv:2311.06842v1](https://arxiv.org/abs/2311.06842).
