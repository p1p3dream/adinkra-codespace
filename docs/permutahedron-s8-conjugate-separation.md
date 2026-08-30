# S8 paired-sector scan across the 30 conjugate R8 subgroups

## Primary-source question

The conclusion of arXiv:2304.09830v2 states that a decomposable eight-color
basis built from the six four-color subsets must use an ordered pair of
distinct subsets. This gives

```text
6 x 5 = 30
```

ordered pairs. The paper notes that Ref. [2] also found 30 conjugate `R8`
subgroups and leaves the possible relation between these two counts as a future
question.

This calculation tests whether the unsigned subgroup and paired-block data
supply a canonical one-to-one correspondence. It also tests whether scanning
all conjugate subgroups recovers the 4,136 cosets outside the earlier fixed
`R8` four-plus-four scan.

## Canonical subgroup enumeration

For every `g` in lexicographic `S8` order, the Rust calculation forms

```text
g R8 g^-1
```

and canonicalizes the subgroup as the sorted eight-element vector of Lehmer
ranks. This produces 30 distinct subgroup keys. Every key has exactly 1,344
conjugators. Therefore

```text
40,320 / 1,344 = 30,
```

in agreement with the normalizer and orbit-stabilizer calculation.

The 435 unordered pairs of distinct conjugate subgroups have intersection-order
histogram

| Intersection order | Subgroup pairs |
|---:|---:|
| 1 | 330 |
| 2 | 105 |

The thirty subgroups collectively preserve all 35 unordered partitions of
eight labels into two blocks of four. Every block partition is preserved by
exactly six conjugate subgroups.

Their 5,040 right cosets per subgroup give

```text
30 x 5,040 = 151,200
```

distinct octet sets. This reproduces the full unsigned `GR(8,8)` permutation-
set count in Theorem 3 of Ref. [2], not merely the subgroup count. The theorem
is stated for left cosets, while the implementation uses right cosets. For a
conjugate subgroup `H`,

```text
(gH)^-1 = H g^-1
gH = (g H g^-1) g
Hg = g (g^-1 H g).
```

The first identity gives the inversion bridge from a theorem support to a
right coset of the same `H`. The next two identities prove that the unions of
left and right coset supports over all conjugates of `R8` are identical as
sets of octets.

## Broad unsigned support relation is complete, not bijective

Each canonical subgroup was scanned through all 5,040 of its right cosets and
all seven of its invariant four-plus-four partitions. Sector labels were
computed after pulling the scan back through the subgroup's least-rank
conjugator. This fixes a reproducible coordinate chart without treating that
lexicographic choice as intrinsic physics.

The result is not a bijection:

| Quantity | Result |
|---|---:|
| conjugate `R8` subgroups | 30 |
| ordered distinct `Pi -> Pj` labels | 30 |
| ordered labels realized by each subgroup | 30 |
| subgroups realizing each ordered label | 30 |
| occurrences of each ordered label in each subgroup scan | 28 |
| support-relation edges | 900 |
| edges required for a bijection | 30 |

Every conjugate subgroup realizes every ordered distinct pair. The exact
support relation is the complete `30 x 30` relation, not a matching.
Lexicographically pairing the two lists would create an arbitrary bijection;
it would not discover one.

There is also an intrinsic obstruction to selecting a pairing from the subgroup
alone. The normalizer has order 1,344 and induces 168 distinct actions on the
seven invariant four-plus-four partitions. This action is transitive and no
partition is fixed by the full normalizer. Thus a conjugate subgroup does not
distinguish one block split or an ordering of its two blocks.

This does not prove that no correspondence exists after adding signed Adinkra
or higher-dimensional data. It establishes that this broad unsigned support
relation does not supply one.

## Deterministic recursive construction

The broad incidence scan is not a substitute for the specific unsigned
recursion in arXiv:2304.09830v2, Eqs. (2.17)-(2.19). The calculation therefore
also applies that recursion to all 30 ordered distinct sector pairs.

For colors one through four it concatenates the first sector permutation with
the second sector permutation shifted to labels five through eight. For colors
five through eight it concatenates the first permutation shifted to labels
five through eight with the unshifted second permutation. This is the
one-line convention illustrated explicitly in Eqs. (2.15) and (2.16), using
the published color order retained by `S4_ORDERED_QUARTETS`.

Exact identification of the right-coset subgroup for every recursive octet
gives:

| Quantity | Result |
|---|---:|
| ordered distinct sector pairs tested | 30 |
| distinct recursive unsigned octets | 30 |
| conjugate `R8` families used | 1 |
| family used | standard `R8` |

Thus the deterministic recursive pair-to-family map is constant, not
bijective. This resolves the tested unsigned recursion, while the broader
relation left open by arXiv:2304.09830 can still depend on signed equivalence
or higher-dimensional structure not present here.

## Corrected distinct-pair count

The earlier block-compatibility probe treated the two four-color sectors as an
unordered pair and included diagonal pairs. The conclusion of
arXiv:2304.09830v2 requires ordered, distinct pairs. The two counts must be kept
separate.

For every conjugate subgroup:

| Quantity | Block criterion | Distinct-sector candidate restriction |
|---|---:|---:|
| unsigned block-compatible cosets | 904 | - |
| unsigned noncoincident distinct-sector candidates | - | 784 |
| complement | 4,136 | 4,256 |
| diagonal-only cosets removed | - | 120 |

The block-decomposition multiplicities are

```text
0: 4,136
1:   854
3:    49
7:     1
```

After excluding diagonal pairs, the unsigned distinct-sector candidate
multiplicities are

```text
0: 4,256
1:   756
3:    28
```

Each subgroup has 1,008 block-compatible coset-partition incidences: 168
diagonal and 840 mixed. The mixed incidences divide uniformly among the 30
ordered distinct labels, with 28 incidences per label.

## The 4,136 gap

The 30 conjugate subgroups define 30 different partitions of `S8` into octets.
They are not additional labels on the original 5,040 `R8` cosets. Across all
thirty families, the scan finds 151,200 distinct coset octets. No octet is a
coset of two different conjugate subgroups.

To test the fixed-universe gap without changing octet identity, the calculation
applies the union of all 35 transported four-plus-four partitions directly to
the original 5,040 `R8` cosets. The result remains 904 block-compatible cosets.
The 28 additional partitions recover zero of the 4,136 original gap cosets.
No original gap octet equals a compatible coset octet from another conjugate
subgroup.

Across all thirty alternative coset families:

| Quantity | Result |
|---|---:|
| distinct coset octets | 151,200 |
| block-compatible octets | 27,120 |
| block-incompatible octets | 124,080 |
| unsigned noncoincident distinct-sector candidate octets | 23,520 |
| candidate complement | 127,680 |
| block-compatible incidences | 30,240 |
| diagonal incidences | 5,040 |
| mixed incidences | 25,200 |

The conjugate scan transports the same `904 / 4,136` block split to thirty
alternative octet partitions. It does not fill the fixed `R8` gap.

## Reproduction artifacts

Rust module:

- `src/permutahedron_s8_conjugate_separation.rs`

Generated artifacts:

- `data/permutahedron_s8_conjugate_separation.json`
- `results/permutahedron_s8_conjugate_separation_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `300a4ef9934a641c39bda92cb63c3c9661ca30d68e189e323fc87e9447404245` |
| validation | `7a6aa57b204744b78e346676cb72165c3a0a00e455a8eb76003d656e949073bd` |

Run:

```sh
cargo run --release -- perm-s8-conjugates-build
cargo run --release -- perm-s8-conjugates-verify
```

The exact tests verify the 30 equal conjugator fibers, all subgroup scans, the
complete ordered-pair support relation, all 30 recursively constructed
unsigned octets and their single standard-`R8` family, normalizer transitivity,
octet-family uniqueness, and zero recovery of the original gap.

## Boundaries

- This is an unsigned permutation and block-decomposition calculation.
- The least-rank conjugator provides deterministic coordinates, not an
  invariant physical labeling.
- The result does not exclude a correspondence that uses Boolean factors,
  Garden signs, HYMN, holoraumy, or other signed representation data.
- The broad support relation and deterministic unsigned recursion are both
  non-bijective. This does not close the broader relation that
  arXiv:2304.09830 leaves for future work.
- Failure of the paired four-plus-four criterion does not prove
  irreducibility.
- The calculation does not classify physical supermultiplets or solve the full
  hex separation problem.

## Primary references

1. A. J. Cianciara, Z. Coleman, S. J. Gates Jr., Y. Lee, and Z. Zhang,
   "N=2 SUSY & the Hexipentisteriruncicantitruncated 7-Simplex,"
   arXiv:2304.09830v2, Sec. 3.3 and conclusion.
2. S. J. Gates Jr., T. Hubsch, K. Iga, and S. Mendez-Diez, "N=4 and N=8
   SUSY Quantum Mechanics and Klein's Vierergruppe," arXiv:1608.07864,
   Theorem 3 and Fig. 7.
3. A. J. Cianciara, S. J. Gates Jr., Y. Hu, and R. Kirk, "The 300
   'Correlators' Suggests 4D, N=1 SUSY Is a Solution to a Set of Sudoku
   Puzzles," arXiv:2012.13308v6.
