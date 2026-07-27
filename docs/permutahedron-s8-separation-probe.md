# Single-R8 block-decomposition probe

## Question

The four-color permutahedron separates into six published `V4` cosets,
`P1` through `P6`. The paired eight-color constructions in Ref. [2] place two
of those four-color sectors into one `R8` octet. This calculation asks how far
that mechanism extends across the complete 5,040-octet right-coset partition
of `S8`.

This is a test of a specific unsigned decomposition criterion:

> An `R8` coset is four-plus-four decomposable when all eight of its
> permutations preserve or exchange the two blocks of an `R8`-invariant
> partition of the eight labels into sets of four.

The calculation does not assume that every resulting combinatorial class is a
distinct physical representation.

## Construction

The Rana subgroup is isomorphic to `(Z2)^3`. Its action on eight labels has
seven invariant unordered partitions into two blocks of four. For example,

```text
{1,2,3,4} | {5,6,7,8}.
```

For each invariant partition, its stabilizer in `S8` is the wreath product

```text
S4 wr S2
```

of order `2(4!)^2 = 1,152`. Since this stabilizer contains `R8`, it contains

```text
1,152 / 8 = 144
```

complete right `R8` cosets.

Each block restriction is an element of `S4` and therefore belongs to one of
the six published four-color sectors. Exchanging the two blocks does not change
the octet, so the eight-color label is an unordered pair

```text
Pi + Pj,  1 <= i <= j <= 6.
```

There are 21 such unordered pair classes. The later decomposability proposal
in Ref. [3] instead retains block order and requires distinct sectors, giving
30 ordered classes. Those are tested separately in the all-conjugate scan.

## Complete scan

The Rust calculation checks all 5,040 right `R8` cosets against all seven
invariant four-plus-four partitions.

| Quantity | Result |
|---|---:|
| right `R8` cosets | 5,040 |
| invariant four-plus-four partitions | 7 |
| compatible cosets per partition | 144 |
| pair classes per partition | 21 |
| total compatible incidences | 1,008 |
| distinct compatible cosets | 904 |
| cosets with no compatible partition | 4,136 |
| diagonal pair incidences | 168 |
| mixed pair incidences | 840 |

For each partition, every diagonal pair `Pi+Pi` contains four cosets and every
mixed pair `Pi+Pj` contains eight. Thus

```text
6(4) + 15(8) = 144.
```

The number of invariant decompositions carried by one coset has the following
distribution:

| Compatible partitions | Cosets |
|---:|---:|
| 0 | 4,136 |
| 1 | 854 |
| 3 | 49 |
| 7 | 1 |

This gives a complete result for this single-family block criterion:

- 904 cosets decompose through at least one `R8`-invariant four-plus-four
  split;
- 4,136 do not decompose by this criterion.

Applying the ordered-distinct construction proposed in Ref. [3] removes 120
cosets that carry only diagonal decompositions. It leaves 784 unsigned
noncoincident distinct-sector candidates and a complement of 4,256.

## Relation to left-right coset coincidence

For every compatible coset-partition incidence,

```text
Pi + Pi
```

occurs if and only if the right `R8` coset is equal, as a set, to the
corresponding left coset. Mixed pairs occur only outside that left-right
coincident set.

Across the seven partitions, the 168 diagonal incidences reproduce the number
168 obtained in Ref. [3]. The 168 left-right coincident cosets have this
decomposition-count distribution:

| Compatible partitions | Left-right coincident cosets |
|---:|---:|
| 0 | 48 |
| 1 | 98 |
| 3 | 21 |
| 7 | 1 |

The remaining cosets have distribution `4,088`, `756`, and `28` at zero, one,
and three compatible partitions.

This does not replace the normalizer explanation

```text
N_S8(R8) / R8 isomorphic to GL(3,2).
```

It connects that result to the paired four-color sector construction.

## Published systems

For the standard split `{1,2,3,4}|{5,6,7,8}`, the six systems in Ref. [2]
occupy:

| System | Four-color pair | Left-right coincident |
|---|---|---:|
| `CC` | `P1+P1` | yes |
| `CT` | `P1+P2` | no |
| `CV` | `P1+P3` | no |
| `TT` | `P2+P2` | yes |
| `TV` | `P2+P3` | no |
| `VV` | `P3+P3` | yes |

This recovers the expected chiral, tensor, and vector pair pattern directly
from the unsigned permutation octets. It also reproduces the observed
left-right coincidence of `CC`, `TT`, and `VV`.

The combinatorial decomposition does not reproduce the off-shell closure split.
The signed calculation still finds closure only for `CT` and `CV`. Therefore
the unsigned pair class and the signed HYMN/Garden calculation carry different
information.

## Reproduction

```sh
cargo run --release -- perm-s8-separation-build
cargo run --release -- perm-s8-separation-verify
node scripts/test_permutahedron_s8_separation.mjs
```

Generated artifacts:

- `data/permutahedron_s8_separation_probe.json`
- `results/permutahedron_s8_separation_probe_validation.json`

| Generated artifact | SHA256 |
|---|---|
| `data/permutahedron_s8_separation_probe.json` | `a4ad22c1cefb1df44b474d59d7d7d2ec9fc928773d975223e870989efa714c4d` |
| `results/permutahedron_s8_separation_probe_validation.json` | `9411560dab54dfbd2cc3e5ee3dbd8d3c42cfb9074efe87d667470b3ab01c11b4` |

The independent JavaScript check regenerates all 40,320 permutations, all
5,040 right cosets, the seven invariant partitions, every pair label, and the
left-right coincidence test.

## What this establishes

The scan supplies a complete answer for one fixed-`R8` question: which cosets
inherit an unordered paired four-color decomposition, and which pair they
inherit for each compatible partition.

The complete all-conjugate calculation shows that transporting the other 28
four-plus-four partitions recovers none of the 4,136 cosets. Other conjugate
`R8` subgroups define different octet families rather than new decompositions
of these fixed-family octets.

The block-incompatible cosets could still require:

- a different decomposition mechanism;
- a decomposition not built from two four-color sectors; or
- exclusion from the intended higher-dimensional representation class.

Choosing among those possibilities requires the physical separation criterion
Gates intended, not another unsigned statistic.

## Boundaries

- The result classifies unsigned permutation octets.
- Boolean factors, Garden closure, HYMN, and holoraumy remain separate signed
  calculations.
- The 21 pair classes are combinatorial classes in a fixed block convention.
  Their interpretation as inequivalent physical representations is not
  established.
- Ref. [3] proposes 30 ordered pairs of distinct sectors. The unordered
  21-class result here must not be substituted for that proposal.
- Failure of this four-plus-four criterion is not proof of irreducibility.
- This does not solve the full hex separation problem.

## Primary references

1. A. J. Cianciara, S. J. Gates Jr., Y. Hu, and R. Kirk, "The 300
   'Correlators' Suggests 4D, N=1 SUSY Is a Solution to a Set of Sudoku
   Puzzles," [arXiv:2012.13308v6](https://arxiv.org/abs/2012.13308).
2. D. D. Bristow, J. H. Caporaletti, A. J. Cianciara, S. J. Gates Jr.,
   D. Levine, and G. Yerger, "A Note On Exemplary Off-Shell Constructions Of
   4D, N=2 Supersymmetry Representations,"
   [arXiv:2012.14015v7](https://arxiv.org/abs/2012.14015).
3. A. J. Cianciara, Z. Coleman, S. J. Gates Jr., Y. Lee, and Z. Zhang,
   "N=2 SUSY & the Hexipentisteriruncicantitruncated 7-Simplex,"
   [arXiv:2304.09830v2](https://arxiv.org/abs/2304.09830).
