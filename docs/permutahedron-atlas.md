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
- the six signed `S4` sectors, all 96 published fiducial Boolean-factor
  quartets, and all 96 corresponding Adinkras;
- the six published signed `S8` systems, all 51 height-sign branches, and their
  Garden, HYMN, graph, Bruhat, coset, chromocharacter, and gadget records;
- a complete scan of the seven `R8`-invariant four-plus-four partitions,
  identifying the 904 cosets that inherit paired four-color sector labels;
- a Garden-sign feasibility calculation for every one of the 5,040 right
  `R8` cosets.

The atlas treats a two-point correlator as a minimal weak-Bruhat graph distance.
It is not a holoraumy gadget. Signed matrix data and Garden closure are separate
calculations. The accompanying sign scan performs that separate finite
calculation for the `R8` cosets.

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

## Garden-sign scan and the 168 cosets

Each unsigned octet defines eight `8 x 8` permutation matrices. A sign bit is
assigned to each of their 64 nonzero entries. For each distinct color pair,
the program first checks that the two monomial products have the same support.
Their cancellation condition is then an affine system over `GF(2)` equivalent
to

```text
L_I L_J^T + L_J L_I^T = 2 delta_IJ I.
```

All 5,040 systems have rank 45 and nullity 19. Consequently, every right `R8`
coset admits 524,288 signings. The program constructs one signing for every
coset, verifies it with the independent sparse `AdinkraRep` implementation,
and checks 20,643,840 dense matrix entries. No residual entry remains.

The uniform result has a direct explanation. Every right coset has the form
`R8 g`, so all eight unsigned matrices receive the same permutation action.
Such a common basis action preserves Garden-sign feasibility and the affine
solution dimension. The full scan verifies that invariance for every labeled
coset rather than treating 5,040 equivalent unsigned systems as distinct
topological obstructions.

The 168 left-right coincident cosets also have a direct group-theoretic
description. A right coset `R8 g` equals `g R8` precisely when `g` normalizes
`R8`. Enumeration gives

```text
|N_S8(R8)| = 1,344,
|N_S8(R8)/R8| = 1,344 / 8 = 168.
```

`R8` is elementary abelian of rank three. Conjugation by its normalizer induces
168 distinct automorphisms, identifying the quotient with `GL(3,2)` and the
normalizer with `AGL(3,2)`.

This closes the unsigned feasibility question: left-right coset coincidence
does not distinguish whether an `R8` coset admits Garden signs. All 168
left-right coincident cosets and all 4,872 other cosets admit them.

## Six signed eight-color systems

The six signed systems in Ref. [2] are reconstructed from the printed
permutations and Boolean factors. All 16 residue classes
`(m mod 4, n mod 4)` are evaluated for `TT`, `TV`, and `VV`, producing 51
branches in total. Both Garden relations are checked on 417,792 dense matrix
entries.

Only `CT` and `CV` close. Their graphs satisfy all 112 two-color odd-dashing
conditions and are stored as valise Adinkras. The other 49 branches retain the
published nonclosure and are stored as signed colored graphs.

The HYMN calculation reproduces Eq. (3.3) on all 51 branches:
`sigma_3 tensor I_8` for `CT` and `CV`, and `I_16` for the other four systems.
Unsigned left-right coset coincidence does not reproduce this separation
because `TV` is not left-right coincident but does not close. Full method and
boundary details are in
[`permutahedron-s8-supersymmetry.md`](permutahedron-s8-supersymmetry.md).

## Paired four-color separation probe

The seven `R8`-invariant partitions of eight labels into two blocks of four
provide a direct test of the paired construction used for the published
eight-color systems. Each partition contains 144 complete `R8` cosets. Those
cosets divide into the 21 unordered pairs of the six four-color sectors.

Across all seven partitions, 904 distinct cosets admit at least one such
decomposition and 4,136 admit none. Every diagonal pair coincides with a
left-right coincident coset for that partition. The 168 diagonal incidences
therefore connect the paired-sector construction to the existing normalizer
result.

This is a partial separation of the hex, not a physical classification of all
5,040 octets. Full results and boundaries are in
[`permutahedron-s8-separation-probe.md`](permutahedron-s8-separation-probe.md).

## Reproduction

The atlas is generated in Rust. From the repository root, run:

```sh
cargo run --release -- perm-atlas-build
cargo run --release -- perm-atlas-verify
cargo run --release -- perm-garden-scan
cargo run --release -- perm-s4-susy-build
cargo run --release -- perm-s4-susy-verify
cargo run --release -- perm-s8-separation-build
cargo run --release -- perm-s8-separation-verify
cargo run --release -- perm-s8-susy-build
cargo run --release -- perm-s8-susy-verify
```

Generated data:

- `data/permutahedron_s4_atlas.json`
- `data/permutahedron_s8_atlas.json`
- `data/permutahedron_s8_garden.json`
- `data/permutahedron_s4_supersymmetry.json`
- `data/permutahedron_s8_separation_probe.json`
- `data/permutahedron_s8_supersymmetry.json`
- `results/permutahedron_validation.json`
- `results/permutahedron_s4_supersymmetry_validation.json`
- `results/permutahedron_s8_separation_probe_validation.json`
- `results/permutahedron_s8_supersymmetry_validation.json`

Validated artifact hashes:

| Artifact | SHA256 |
|---|---|
| `data/permutahedron_s4_atlas.json` | `0738927f723ffd16b310ff7a7bcbafa8629b791583524a165a9c1635c008be2c` |
| `data/permutahedron_s8_atlas.json` | `b178d1642d05f1b7e0b9b5b5befdf3fb3c412717b80202c9836fe1bf061f4ba3` |
| `data/permutahedron_s8_garden.json` | `9e0d983f678ee89704809008ea5fdeea993fbb5d535f53564477b9a313056a67` |
| `data/permutahedron_s8_separation_probe.json` | `a4ad22c1cefb1df44b474d59d7d7d2ec9fc928773d975223e870989efa714c4d` |
| `data/permutahedron_s8_supersymmetry.json` | `3df539d4d82472566cc4ab90dc5dbf5f8ab1035ee9708cd1d7c26a51a550d241` |
| `results/permutahedron_validation.json` | `feb3d95a19a525dd538e53baa2b56018988629fe8576ce6cbc29d39174973cc1` |
| `results/permutahedron_s8_separation_probe_validation.json` | `9411560dab54dfbd2cc3e5ee3dbd8d3c42cfb9074efe87d667470b3ab01c11b4` |
| `results/permutahedron_s8_supersymmetry_validation.json` | `a6fb4d1ec4e6571d55d9780db81906f589aa1fc354bc9c6b19d00048f41002b4` |

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

Open `visualizer/permutahedron_s4_supersymmetry.html` for the signed four-color
sector viewer. It displays all 16 published fiducial sign choices within each
of the six sectors.

Open `visualizer/permutahedron_s8_supersymmetry.html` for the six signed
eight-color systems. It displays all 51 height-sign branches and distinguishes
the two Garden-closing Adinkras from the 49 signed graphs with nonclosure.

Validate the browser datasets and their Garden-scan join with:

```sh
node scripts/test_permutahedron_atlas.mjs
node scripts/test_permutahedron_s4_supersymmetry.mjs
node scripts/test_permutahedron_s8_separation.mjs
node scripts/test_permutahedron_s8_supersymmetry.mjs
```

## Boundaries

- `CT`, `CV`, and the Diadem have published Garden-closing sign choices.
- `CC`, `TT`, `TV`, and `VV` have published nonclosure terms. Their unsigned
  permutation octets nevertheless admit other Garden signings. This does not
  change the closure status of the particular published Boolean factors or the
  corresponding four-dimensional constructions.
- The scan solves sign feasibility for right cosets of this `R8`. It does not
  solve sign assignment for arbitrary collections of eight permutations.
- The interpretation of the permutahedron as a supersymmetry weight space is a
  research proposal, not an established representation theorem.
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
