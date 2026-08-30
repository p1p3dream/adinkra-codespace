# Source audit of the S8 separation problem

## Finding

The papers do not define a technical object called the "hex separation."
They contain three related, but different, separation problems:

1. separate the six printed paired systems `CC`, `CT`, `CV`, `TT`, `TV`,
   and `VV` by their signed supersymmetry properties;
2. classify the unsigned supports of all minimal `GR(8,8)` systems inside
   `S8`;
3. determine which eight-color systems decompose into two distinct
   four-color sectors.

The repository has completed the first problem for the six printed systems,
the full unsigned support enumeration in the second, and the unsigned
subgroup-block test in the third. The latter does not produce a one-to-one
map: every conjugate subgroup realizes every ordered distinct-sector label.
The deterministic unsigned recursion in arXiv:2304.09830, Eqs.
(2.17)-(2.19), also does not produce such a map: its 30 distinct ordered-pair
octets all belong to the standard `R8` right-coset family.
The 2023 paper explicitly leaves the numerical coincidence between the two
30-element sets as a question for future work. Signed or higher-dimensional
data could still refine that question.

## Primary-source statements

### Complete unsigned `GR(8,8)` support criterion

Gates, Hübsch, Iga, and Mendez-Diez prove that the unsigned permutation set

```text
{|L1|, ..., |L8|}
```

of a minimal `GR(8,8)` system is a left coset of a conjugate of the
elementary abelian subgroup `A`, which is the subgroup later called `R8` or
the Diadem. This is a necessary and sufficient unsigned-support statement,
not a heuristic.

The normalizer is

```text
N_S8(A) = A semidirect GL(3,2)
|A| = 8
|GL(3,2)| = 168
|N_S8(A)| = 1,344
```

so orbit-stabilizer gives

```text
number of conjugates of A = 8! / 1,344 = 30.
```

Each conjugate has `8!/8 = 5,040` left cosets. Therefore the full labeled
unsigned `GR(8,8)` support set contains

```text
30 x 5,040 = 151,200
```

octets. The paper then gives `2^19 = 524,288` Garden signings for each
unsigned support.

Source: arXiv:1608.07864v1, PDF pp. 9-11, Theorem 3 and Lemma 4.

### The 30 conjugates are not the 168 ab-normal cosets

These numbers arise from different quotients:

```text
30  = |S8 / N_S8(R8)|
168 = |N_S8(R8) / R8|.
```

The 30 count distinct conjugate cube-translation subgroups. Each subgroup
defines a separate 5,040-coset partition of `S8`.

For one fixed `R8`, a right coset `R8 g` equals the corresponding left coset
`g R8` precisely when `g` lies in the normalizer. There are therefore 168
such cosets. These are the sets called ab-normal in arXiv:2304.09830.

Self-duality is a further restriction and must not be identified with
ab-normality. arXiv:1608.07864 finds 22 self-dual cosets per conjugate,
because the quotient element must have order one or two. This gives 660
self-dual unsigned classes over all 30 conjugates.

Sources:

- arXiv:1608.07864v1, PDF pp. 10-13;
- arXiv:2304.09830v2, PDF pp. 19 and 22-24, Secs. 3.3 and 4.

### Published decomposability restriction

The conclusion of arXiv:2304.09830 introduces a "restriction of
decomposability." Starting from the six four-color `S4` subsets, the two
members used to construct a decomposable `N=8` system are required to be
distinct. The paper therefore counts

```text
6 x 5 = 30
```

ordered pairings.

The paper notes that 30 also counts the conjugates of `R8`, but it does not
construct a bijection between the two sets. It states that deciding whether
the repeated number is accidental is future work.

This restriction is necessary in the stated construction, but it is not
sufficient for signed off-shell closure. In the six physical pair types
studied in arXiv:2012.14015, `CT`, `CV`, and `TV` use distinct constituent
types, but only `CT` and `CV` close with the printed Boolean factors.

Source: arXiv:2304.09830v2, PDF pp. 28-29, conclusion; its Refs. [6] and
[33] are arXiv:2012.14015 and arXiv:1608.07864, respectively.

### Ordering is counted, but physical inequivalence is not established

The `6 x 5` statement counts `(Pi,Pj)` and `(Pj,Pi)` separately. The
recursive unsigned construction also places the first and second factors in
different matrix blocks.

The papers do not show that reversing those blocks gives a physically
inequivalent one-dimensional supermultiplet. At the matrix level, a common
block-exchange permutation conjugates the recursive support for `(Pi,Pj)`
to that for `(Pj,Pi)`. arXiv:1608.07864 also explains that common boson and
fermion relabelings can change a labeled `L`-matrix set while leaving the
Adinkra or supermultiplet equivalent.

Consequently:

- 30 is the published count of ordered decomposable candidates;
- 15 is the count after forgetting order;
- whether order retains higher-dimensional physical information is not
  settled by these papers.

Any proposed map from the 30 ordered pairs to the 30 conjugate subgroups
must state the equivalence convention and demonstrate that the map is
well-defined. A numerical match alone does not supply that map.

## Published filters and their logical roles

| Filter | Signed? | Published role | What it does not establish |
|---|---:|---|---|
| Coset of a conjugate of `R8` | no | Necessary and sufficient support criterion for labeled minimal `GR(8,8)` matrices | Which higher-dimensional multiplet the support represents |
| `R8` magic number 112 | no | Position or validation rule for the published octets | Garden closure or the `CT/CV` split |
| Bruhat correlator spectrum | no | Separates intra- and inter-quartet geometry in `S4`; proposed as an `S8` sorting tool | A signed supersymmetry representation by itself |
| Graph vertex coloring and face structure | no | Proposed polytopic organization and lower-to-higher construction | Off-shell closure |
| Distinct four-color factors | no | Necessary restriction for the proposed decomposable construction | Sufficiency; `TV` is the counterexample among the printed systems |
| Complementary BK weights or digit sums | no | Property preserved by the recursive construction | Garden closure |
| Doubly-even four-bit Boolean flip | yes | Published recursive sign ansatz for `CT`, `CV`, and the Diadem | Exhaustion of all possible Garden dashings on that unsigned support |
| Garden relations | yes | Decisive one-dimensional off-shell closure test | Enhancement to a specified four-dimensional multiplet |
| HYMN | yes | Separates the six printed systems into the same two classes as their closure calculation | A proved classification of all 151,200 unsigned supports and all signings |
| Chromocharacter | yes | Established elsewhere as a signed Adinkra invariant; `chi_o` is treated for four-color and unfolded systems | A published `S8` separation rule in these permutahedron papers |

### Boolean factors and arbitrary Garden signs

The unsigned permutation and Boolean factors are separate data. In the
recursive construction of arXiv:2304.09830, the first four Boolean words are
built from the two constituent four-color systems. The last four are tested
using cyclic choices of a neighboring four-bit flip. The authors use Garden
closure to select the accepted choice. For `CT`, the accepted flip is
`{6,7,8,1}`.

The repository's affine `GF(2)` solver asks a broader question: does any
assignment of 64 edge signs close on this unsigned octet? It finds
`2^19` signings for each coset in the one standard `R8` partition, agreeing
with the count in arXiv:1608.07864.

This does not repair `CC`, `TT`, `TV`, or `VV`. Their printed Boolean factors
come from particular four-dimensional transformation laws and retain the
published nonclosure. A different one-dimensional dashing on the same
unsigned support is a different signed system.

### Garden closure and HYMN

For the six printed systems, arXiv:2012.14015 defines

```text
gamma_hat_I = [[0, L_I], [R_I, 0]]
C_hat = gamma_hat_8 ... gamma_hat_1.
```

The result is

```text
CT, CV:          C_hat = sigma_3 tensor I_8
CC, TT, TV, VV:  C_hat = I_16.
```

The first class has zero HYMN trace and satisfies the Clifford or Garden
relations. The second has trace 16 and has the printed nonclosure terms. The
Diadem lies in the first class.

This is the source-backed separation of the six named systems. It is stronger
than an unsigned partition because it uses the printed signs and checks the
algebra. It is narrower than a full `S8` classification because the paper
does not evaluate all signed equivalence classes arising from all 30
conjugates.

Source: arXiv:2012.14015v7, PDF pp. 3-6, Eqs. (2.1)-(2.7) and
(3.1)-(3.6).

### Chromocharacters

The four permutahedron papers do not publish a chromocharacter criterion that
solves the `S8` separation problem. arXiv:2311.06842 discusses `chi_o`,
including

```text
chi_o = Tr(L1 R2 L3 R4) / 4,
```

for four-color folded and unfolded systems.

The repository's generalized `S8` chromocharacter quantity is therefore an
additional diagnostic. Its separation of the six printed examples is worth
recording, but it must not be presented as Gates's published `S8` criterion
or as a classification theorem.

## Repository status

### Completed

The repository currently supplies:

- the complete `S8` graph with 40,320 vertices;
- the requested `56 x 56` Bruhat matrix for the six named octets and the
  Diadem;
- the 5,040 cosets of one standard `R8`;
- the normalizer order 1,344 and its 168 left-right coincident cosets;
- the magic-number check for all 5,040 cosets in that partition;
- an unrestricted Garden-sign feasibility scan for those 5,040 cosets;
- the printed Boolean-factor, Garden, and HYMN calculation for all six named
  systems and all 51 parameter branches;
- a scan of the seven block systems of the standard `R8`.
- all 30 conjugate `R8` subgroups and all 151,200 distinct unsigned supports;
- the seven block systems for every conjugate subgroup;
- the 30 ordered distinct-sector labels in deterministic coordinate charts;
- an exact test showing that the unsigned subgroup-block incidence relation is
  complete rather than one-to-one;
- the 20 normalizer-conjugacy classes of the fixed-`R8` coset family.

The original single-family scan classifies 21 unordered four-color pair
labels. Applying the published distinct-factor restriction leaves:

```text
840 mixed-pair incidences
784 unsigned noncoincident distinct-sector candidates
```

The 168 diagonal incidences, carried by 120 distinct cosets, do not satisfy
the published distinct-factor restriction. These are repository-derived
counts, not claims from the papers.

The all-conjugate scan uses right cosets, while the support theorem in
arXiv:1608.07864 is stated for left cosets. For each conjugate subgroup `H`,

```text
(gH)^-1 = H g^-1,
gH = (g H g^-1) g,
Hg = g (g^-1 H g).
```

The first identity is the inversion bridge. The next two prove that the
left-coset and right-coset unions over all 30 conjugates are the same 151,200
octet sets.

The repository also implements the deterministic unsigned recursion in
arXiv:2304.09830, Eqs. (2.17)-(2.19), for all 30 ordered distinct sector
pairs. It retains the published color order and uses the one-line block
conventions illustrated in Eqs. (2.15) and (2.16). The result is 30 distinct
octets in one conjugate family, the standard `R8` family. Therefore the
recursive pair-to-family map is constant, not bijective.

### Not completed

The present atlas does not yet contain:

1. the prescribed recursive Boolean-factor test on every decomposable
   ordered pair;
2. canonical signed equivalence classes with field relabeling, vertex
   switching, color ordering, and duality stated explicitly;
3. a higher-dimensional enhancement test assigning a four-dimensional
   multiplet interpretation to the resulting one-dimensional systems.

## What would close each gap

### Full unsigned `GR(8,8)` atlas: complete

The implementation now:

1. enumerates one representative of every conjugacy orbit
   `g R8 g^-1`;
2. verifies there are 30 distinct subgroups and one orbit under `S8`;
3. verifies each normalizer has order 1,344;
4. enumerates 5,040 cosets for each subgroup;
5. verifies 151,200 distinct labeled supports and the theorem's full cover;
6. records conjugate-subgroup, coset, and equivalence identifiers.

All six checks pass. The unsigned support-enumeration gap is closed.

### Broader published 30-to-30 question

The repository has now tested two exact unsigned relations:

1. The all-conjugate support relation is complete, with all 900
   subgroup-pair edges.
2. The deterministic recursive pair-to-family relation is constant, with all
   30 pairs mapping to the standard `R8` family.

Neither is a bijection. This settles those unsigned tests, but it does not
close the broader relation left open by arXiv:2304.09830. Closing that question
would require a stated signed or higher-dimensional equivalence convention
and a well-defined invariant map under that convention.

### Signed one-dimensional separation

This gap closes only after Boolean factors are included. There are two
different questions:

1. unrestricted Garden feasibility for every unsigned support;
2. closure of the Boolean factors produced by the published recursive
   construction.

The first is transferred across the full conjugacy orbit by relabeling and
has the published `2^19` count. The second carries the information that
separates the printed `CT/CV` systems from the printed
`CC/TT/TV/VV` systems.

### Physical separation

A complete physical classification requires more than the permutahedron:

- state the quotient by boson, fermion, and color relabelings;
- distinguish vertex switching from a genuinely different dashing;
- compute Garden closure and signed invariants only on valid signed systems;
- test enhancement or reconstruct the higher-dimensional transformation
  laws and action;
- verify off-shell closure in the intended higher dimension.

Without those steps, even a complete 151,200-support atlas remains an
unsigned `GR(8,8)` classification, not a classification of distinct
four-dimensional off-shell supermultiplets.

## References

1. S. J. Gates Jr., T. Hübsch, K. Iga, and S. Mendez-Diez, "N=4 and N=8
   SUSY Quantum Mechanics and Klein's Vierergruppe,"
   arXiv:1608.07864v1.
2. A. J. Cianciara, S. J. Gates Jr., Y. Hu, and R. Kirk, "The 300
   'Correlators' Suggests 4D, N=1 SUSY Is a Solution to a Set of Sudoku
   Puzzles," arXiv:2012.13308v6.
3. D. D. Bristow, J. H. Caporaletti, A. J. Cianciara, S. J. Gates Jr.,
   D. Levine, and G. Yerger, "A Note On Exemplary Off-Shell Constructions Of
   4D, N=2 Supersymmetry Representations," arXiv:2012.14015v7.
4. A. J. Cianciara, Z. Coleman, S. J. Gates Jr., Y. Lee, and Z. Zhang,
   "N=2 SUSY & the Hexipentisteriruncicantitruncated 7-Simplex,"
   arXiv:2304.09830v2.
5. A. J. Cianciara, S. J. Gates Jr., Y. Lee, E. T. Levy, T. O. Razzaz, and
   J. Richardson, "Unfolded Adinkra Properties of Supermultiplets (I),"
   arXiv:2311.06842v1.
