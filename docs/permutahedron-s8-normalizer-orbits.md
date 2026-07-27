# Normalizer-conjugacy classes in the fixed R8 atlas

## Question

The existing block-system probe divides the 5,040 cosets of one fixed `R8`
subgroup into:

- 904 cosets that preserve at least one of the seven `R8`-invariant
  four-plus-four block systems;
- 4,136 cosets that preserve none of them.

This calculation asks whether the 4,136 cosets admit a stronger exact
classification using only the finite permutation data already present in the
atlas.

## Independent reproduction

The new Rust calculation rederives block compatibility without using the pair
labels or implementation in `permutahedron_s8_separation.rs`.

It enumerates the 35 unordered partitions of eight labels into two sets of
four, finds the seven preserved or exchanged by every element of `R8`, and
tests one representative of every `R8` coset. This is sufficient because all
elements of `R8` preserve or exchange each of those seven block systems.

The result independently reproduces:

| Compatible block systems | Cosets |
|---:|---:|
| 0 | 4,136 |
| 1 | 854 |
| 3 | 49 |
| 7 | 1 |

There are 904 compatible cosets and 1,008 compatible incidences. The existing
independent JavaScript enumeration also reproduces these counts.

## Exact group action

The normalizer

```text
N_S8(R8) = R8 semidirect GL(3,2)
|N_S8(R8)| = 1,344
```

acts by conjugation on the cosets of the fixed subgroup:

```text
R8 g  ->  R8 (n g n^-1).
```

Exact enumeration gives 20 orbits covering all 5,040 cosets once. Their sizes
are:

```text
1, 21, 24, 24, 28, 42, 56, 56, 56, 84,
112, 168, 224, 336, 336, 336, 448, 672, 672, 1,344.
```

The 904 block-compatible cosets form ten complete orbits:

```text
1, 21, 28, 42, 56, 56, 84, 112, 168, 336.
```

The remaining 4,136 cosets form ten complete orbits:

```text
24, 24, 56, 224, 336, 336, 448, 672, 672, 1,344.
```

This classifies every fixed-`R8` coset under a stated finite symmetry. It does
not assign physical names to the classes.

## Exact signatures

For each orbit the calculation records:

1. the multiset of cycle types of its eight permutations;
2. the order of `R8` intersected with its conjugate by a representative;
3. for left-right coincident cosets, the induced element of `GL(3,2)`.

The first two data separate 19 of the 20 orbits. The remaining two are the two
order-seven conjugacy classes of `GL(3,2)`. Their traces over `GF(2)` are zero
and one, and their characteristic polynomials are:

```text
x^3 + x + 1
x^3 + x^2 + 1.
```

The combined signature separates all 20 orbits exactly.

The subgroup-intersection orders also give a coarser checked distribution:

| Intersection order | Orbits | Cosets |
|---:|---:|---:|
| 1 | 9 | 3,696 |
| 2 | 5 | 1,176 |
| 8 | 6 | 168 |

## Signed and graph diagnostics

The existing exact Garden scan supplies no further separation at this level:

- all 5,040 cosets admit Garden signs;
- every affine sign system has rank 45 and nullity 19;
- the canonical Garden signing has zero HYMN trace for every coset.

These are exact uniformity results. They do not imply that the printed Boolean
factors of every published system close. The printed signs and unrestricted
Garden sign feasibility are different questions.

Bruhat distance and adjacent-transposition graph data depend on the fixed
label ordering and are not invariant under the full normalizer-conjugacy
action. They are therefore not used to define these 20 classes.

## Relation to the published count of 30

The conclusion of arXiv:2304.09830 counts 30 ordered pairs of distinct
four-color sectors:

```text
6 x 5 = 30.
```

The same paper notes the separate count of 30 conjugate `R8` subgroups and
leaves a possible relationship open.

The 20 classes above are neither of those 30-element sets. They are orbits of
the normalizer acting by conjugation on the 5,040 cosets of one fixed `R8`.
Therefore this calculation does not establish a correspondence between the
30 ordered pairs and the 30 conjugate subgroups.

The original 904 count includes diagonal pairs and forgets the order of the
two four-color sectors. Applying the unsigned noncoincidence condition, which
the pair scan identifies with mixed pairs, leaves 784 unsigned noncoincident
distinct-sector candidates in six orbits. Its
complement contains 4,256 cosets in fourteen orbits. This still does not
retain the ordered-pair data required by the published 30-pair question.

## Boundary

The result is a complete unsigned classification for one specified group
action on one fixed-`R8` coset universe. It does not establish:

- 20 supermultiplet types;
- a physical interpretation of the ten classes containing the 4,136
  block-incompatible cosets;
- a map between the 30 ordered pairs and 30 conjugate subgroups;
- closure of any prescribed Boolean-factor construction;
- equivalence or inequivalence after all boson, fermion, color, switching,
  duality, and higher-dimensional identifications.

The full unsigned support theorem concerns 30 conjugate `R8` subgroups and
151,200 labeled supports. That larger calculation is separate.

## Reproduction

Implementation:

```text
src/permutahedron_s8_orbits.rs
```

Generated artifacts:

- `data/permutahedron_s8_normalizer_orbits.json`
- `results/permutahedron_s8_normalizer_orbits_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `f25f85cfe8f01ac4767817a9a020c798605c219231a1090adf777ef927ae4a85` |
| validation | `0d4981e9f7800aa5431c1fa1d287c32b09b86f45224522994524455e5528c055` |

Build and verify:

```sh
cargo run --release -- perm-s8-orbits-build
cargo run --release -- perm-s8-orbits-verify
cargo test --release --bin adinkra-codespace permutahedron_s8_orbits
node scripts/test_permutahedron_s8_separation.mjs
```

The Rust module contains four exact tests covering the independent block
count, complete orbit cover, 20 signature classes, the two order-seven
classes, and the failure of the 20-class count to reproduce either published
30-element set.

## References

1. S. J. Gates Jr., T. Hübsch, K. Iga, and S. Mendez-Diez, "N=4 and N=8
   SUSY Quantum Mechanics and Klein's Vierergruppe," arXiv:1608.07864v1.
2. A. J. Cianciara, Z. Coleman, S. J. Gates Jr., Y. Lee, and Z. Zhang,
   "N=2 SUSY & the Hexipentisteriruncicantitruncated 7-Simplex,"
   arXiv:2304.09830v2.
