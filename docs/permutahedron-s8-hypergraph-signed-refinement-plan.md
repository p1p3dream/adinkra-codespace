# S8 Hypergraph Signed-Refinement Plan

## Question

The unsigned Garden compatibility condition discovers exactly 30 order-eight
identity blocks in `S8`. Their right translates form 30 complete partitions of
the 40,320 permutations into 151,200 distinct octets. The next question is
whether signed or higher-dimensional data distinguish one partition, one
octet family, or a smaller physically relevant subset.

## Established input

- 105 fixed-point-free involutions form a degree-12 compatibility graph.
- Its size-seven cliques give exactly 30 identity octets without loading `R8`
  labels.
- Right translation gives 5,040 octets per identity block and 151,200 octets
  in total.
- Each permutation belongs to 30 octets, one in every partition.
- No octet occurs in two partitions.
- Every octet intersects exactly 204 octets from other partitions.

These are exact unsigned statements. They do not select a physical
representation.

## Phase 1: Garden-space transport audit

For each of the 30 discovered identity octets:

1. construct its Garden equations over `GF(2)`;
2. record feasibility, equation rank, nullity, and solution count;
3. transport one certified signing through all 5,040 right translates;
4. verify the Garden algebra directly on all 151,200 transported systems; and
5. record whether any family differs.

Expected mathematical control: replacing every permutation factor `P_I` by
`P_I P_g` is a common fermion-node relabeling. Therefore it preserves the
Garden equations. This phase must verify that implementation-level ordering
and sign conventions realize that isomorphism exactly.

Stopping condition: if every family has rank 45, nullity 19, and all
transports close, unsigned support plus sign feasibility cannot break the
30-fold symmetry. Do not enumerate all
`151,200 * 2^19 = 79,272,345,600` labeled signings.

### Phase 1 result

Complete. All 30 identity octets have equation rank 45 and nullity 19. A
certified signing from each identity octet was transported, with the required
row-node sign reindexing, to all 5,040 members of its family. All 151,200
transported systems satisfy the Garden algebra. Garden feasibility and affine
solution-space dimension therefore do not break the 30-fold symmetry.

The first implementation attempt copied row signs without reindexing and
closed on only 135 translates. That failure fixed the convention audit:
with the repository's row-to-column permutation matrices,
`M_(h composed with g) = M_g M_h`, so a right-coset translate requires the
common row-node relabeling induced by `g`. The corrected transport closes on
all 151,200 octets.

## Phase 2: Quotient signed data before computing invariants

Integrate the completed signed-equivalence calculation from the
`s8-signed-recursion` workstream. Keep these relations separate:

1. fixed-color signed node equivalence;
2. the preceding relation plus supercharge signs;
3. unlabeled-color signed graph isomorphism; and
4. the preceding relation plus boson-fermion level exchange.

Do not use the solver's canonical sign mask as an invariant. It depends on
variable order and pivot choices.

Stopping condition: if all transported representatives lie in one declared
one-dimensional signed class, no invariant of that class can choose an `R8`
partition.

### Phase 2 status

Complete. The signed-equivalence ledger keeps four relations separate. The
class counts for one deterministic signing on each identity support are:

- fixed-color nodal equivalence: 30 classes;
- fixed-color nodal equivalence plus supercharge signs: 30 classes;
- unlabeled-color signed graph equivalence: 1 class; and
- the preceding relation plus boson-fermion level duality: 1 class.

The fixed-color distinction is entirely tied to retaining the solver's
rank-ordered color labels. Once color relabeling is admitted, all 30 supports
collapse to one signed graph class. Every serialized membership witness was
checked directly on all 64 colored edges, and a mutated witness was rejected.

The result extends from selected representatives to every signing without
enumeration. On each support, the `GF(2)` rank of the 24 boson-switch,
fermion-switch, and supercharge-sign generators is 19, exactly equal to the
Garden nullity. Direct checks show all 24 generators preserve closure. Their
span is therefore the complete homogeneous Garden solution space. Combined
with right-translation covariance, all 79,272,345,600 labeled signings across
the 151,200 hyperedges belong to one unlabeled-color signed graph class.

The existing signed-recursion work independently agrees with this loss of
one-dimensional discrimination: its 24 closing candidates form one
fixed-color nodal class, and the published `CV` and `CT` reductions are nodally
equivalent. Its Gadget six-frame and commutant tests also fail to provide a
rare one-dimensional selector. Solver-selected sign masks are recorded only
for reproducibility and are not invariants.

## Phase 3: Hypergraph resolution test

The 30 subgroup families are known exact covers. Determine whether the
151,200-edge hypergraph admits additional covers mixing octets from different
families.

1. formulate the vertex-octet incidence matrix as an exact-cover problem;
2. fix one octet by transitivity to remove the ambient symmetry;
3. search for a cover not equal to any known subgroup partition;
4. certify any result by checking disjointness and complete 40,320-vertex
   coverage; and
5. if no mixed cover is found, report only the bounded search or prove a
   structural obstruction before claiming uniqueness.

Stopping condition: a single mixed cover disproves uniqueness. Failure to
find one is not a proof unless the search is exhaustive with a checkable
certificate.

### Phase 3 result

Complete for the uniqueness question. A certified four-for-four trade replaces
four octets from discovered family 0 by four octets from family 1 on exactly
the same 32 vertices. Keeping the other 5,036 family-0 octets produces an
exact 5,040-edge cover that differs from all 30 subgroup partitions.

The trade is minimum size. Any two distinct hyperedges intersect in at most
two vertices. An added octet contained in the union of `t` removed octets can
therefore contain at most `2t` vertices, so `t` must be at least four. The
32-vertex witness attains that lower bound.

Across all 435 pairs of discovered identity subgroups, the exact
intersection/join census is:

- 105 pairs: intersection order 2, generated subgroup order 32;
- 210 pairs: intersection order 1, generated subgroup order 96; and
- 120 pairs: intersection order 1, generated subgroup order 20,160.

Each order-32 pair has 1,260 right translates, giving 132,300 elementary
four-trade certificates. Exact-cover uniqueness is therefore decisively
false and should not be used as an `R8` selector.

## Phase 4: Signed and higher-dimensional augmentation

Only after quotienting add features that are well defined under the selected
equivalence relation:

- HYMN data;
- holoraumy and Gadget data;
- engineering heights;
- gauge and phantom linkage data;
- four-dimensional field content; and
- verified enhancement or parentage results.

Use the published `O`, `CV`, and `CT` systems as positive controls. Hold them
out when testing any learned or optimized selection rule.

### Phase 4 bridge result

The seven published controls were projected into the complete 151,200-edge
hypergraph:

- positive Garden controls: `O`, `CT`, and `CV`;
- published nonclosing controls: `CC`, `TT`, `TV`, and `VV`.

All seven belong to discovered family 0, the standard `R8` partition. Their
slice IDs are `O=0`, `VV=722`, `TT=843`, `TV=850`, `CT=1443`, `CC=1444`, and
`CV=1450`. Every unsigned support admits `2^19` valid Garden signings. Direct
matrix checks reproduce closure for the certified `O` signing and the printed
`CT` and `CV` signs, while the printed `CC`, `TT`, `TV`, and `VV` assignments
do not close.

This supplies a particularly sharp negative control: positive and negative
published systems occupy the same subgroup family, and all their unsigned
supports are signable. Neither family membership nor existence of some
Garden signing can classify the printed systems. The distinguishing data are
the particular sign assignment and, for physical parentage, information not
contained in the unsigned hypergraph or its one-dimensional signed quotient.

### Phase 4 higher-dimensional gate result

The sourced four-dimensional positive controls have now been integrated and
reverified in this workstream.

- `CV`: 612 exact component relations pass, including the vector-potential
  gauge residue, and all 512 reduced matrix entries reproduce the committed
  `CV` anchor.
- `CT`: 684 exact component relations pass, including the two-form gauge
  residue, and all 512 reduced matrix entries reproduce the committed `CT`
  anchor.
- Their sourced spatial-linkage operators differ.
- Their sourced gauge-residue operators differ.
- Both nevertheless lie in family 0 and in the single valid unlabeled-color
  worldline signing class.

This proves directly that the spatial and gauge information distinguishing
the two known parents is discarded by the worldline quotient.

The source-fixture eligibility audit corrects the scope of this gate. `O` is
defined as an original one-dimensional diadem construction; the audited
sources do not assert a four-dimensional component parent for which a fixture
is merely missing. `VM1`, `VM2`, and `VM3` are likewise identified as
mathematical Garden solution sectors rather than sourced 0-brane reductions.
The printed `CC`, `TT`, `TV`, and `VV` assignments fail the Garden prerequisite,
but their unsigned supports admit valid re-signings and lack specified Lorentz,
spatial-linkage, gauge, Bianchi, and reduction data.

The applicable stated-parent positive-control gate is therefore complete:
`CV` and `CT` both pass. No independent physical holdout was found in the
audited corpus. A higher-dimensional parent cannot be assigned to or excluded
for the remaining constructions without a new target specification.

The bounded audit therefore passes, but a broader 151,200-edge enhancement
scan is not authorized. This is an input-bound stop condition, not a numerical
failure.

### Phase 4 Maxwell and recursion bridge result

The complete four-color Maxwell phantom and Bianchi gate from
arXiv:0907.3605 was integrated as an additional sourced control. It reconstructs
the known Maxwell field-strength multiplet from worldline linkage alone and
rejects a chiral negative control.

The exhaustive published S4 scan checks 14,155,776 signed frame pairs across
all 96 fiducial signings. Exactly 48 signings pass, and the result is equivalent
to `chi0 = -1` on this library. Every one of the six unsigned S4 quartets
contains eight passers and eight failures. Thus the gauge calculation confirms
that signs matter, but it does not refine the existing four-color `chi0`
classification.

The two retained four-color blocks were then extracted from every closing
candidate in the signed S8 recursion. The exact finite census is:

- 5,760 aligned recursion candidates checked;
- 24 exact S8 Garden closers;
- 48 embedded S4 blocks, all Garden-closing;
- every S8 closer pairs one `chi0=+1` block with one `chi0=-1` block;
- every closer contains exactly one Maxwell passer;
- the Maxwell result is exactly the ordered `chi0` pair; and
- CT and CV both have ordered Maxwell signature `(fail, pass)`.

Projecting those 24 closers into the complete hypergraph gives a new structural
bridge. All 24 occupy discovered family 0 and use 12 distinct unsigned
supports, with exactly two closing recursive signings per support. Both ordered
Maxwell signatures mix source pairs with named four-dimensional parents and
source pairs with no stated parent. CT and CV occupy different unsigned
supports but share the same embedded-Maxwell signature.

The 12 selected supports occupy four of the 20 normalizer-conjugacy orbits of
the fixed family-0 atlas. The support counts are orbit 1: four, orbit 5: two,
orbit 7: four, and orbit 17: two. In this finite recursion library, orbits 7
and 17 contain exactly the named-parent source pairs, while orbits 1 and 5
contain exactly the unstated-parent source pairs. Every occupied orbit contains
both ordered Maxwell signatures. CT lies in orbit 17 and CV lies in orbit 7.

The Boolean-mask restriction was then removed. Scanning all 256 masks, all 24
relative color orders, and all 36 ordered source pairs checks 221,184 exact
candidates. The unrestricted census finds 64 closers on 32 supports, exactly
two realizations per support. No same-source or mixed-source candidate closes.
All 32 supports remain in family 0. The published subset is recovered exactly,
while 40 noncyclic closers add 20 supports. The fixed-basis source-category
split survives and occupies orbits 1, 4, and 5 on the unstated-parent side and
orbits 7 and 17 on the named-parent side.

The required node-relabeling negative control then closes this correlation as
an intrinsic selector. From one closing support, all 40,320 common relabelings
of one node level reach every one of the 5,040 family-0 supports, each exactly
eight times, and reach all 20 normalizer orbits. Normalizer-orbit ID is not an
invariant of the unlabeled valise. The category split is exact only in the
fixed published component basis.

This exhausts the four-color Maxwell information retained by the stated S8
recursion. It is a strong consistency result and a negative selector result:
standard-family membership, the embedded Maxwell signature, and fixed-family
normalizer orbit do not recover intrinsic higher-dimensional parentage from the
unlabeled valise.
The block-level gate is not an eight-supercharge enhancement theorem because
the full eight-color representation is irreducible.

## Required negative controls

- Random relabelings within a signed equivalence class must not change a
  claimed invariant.
- Conjugating an `R8` family must not create a distinction from unsigned data
  alone.
- The `TV` counterexample must remain rejected by any rule presented as a
  physical selector.
- A criterion fitted to `CV` or `CT` must be evaluated on an independently
  sourced physical holdout before it is called predictive. `O` remains a
  one-dimensional Garden control.

## Deliverables

1. machine-readable Garden transport artifact;
2. exact-cover formulation and certificate format;
3. signed-equivalence ledger;
4. invariant table with provenance and equivalence scope; and
5. a final statement separating unsigned classification, one-dimensional
   signed classification, and higher-dimensional parentage.

## Next bounded computation

The control projection, sourced `CV` and `CT` component gates, embedded
four-color Maxwell census, unrestricted recursion scan, node-basis leakage
audit, and source-fixture eligibility audit are complete. Family membership,
unsigned support, Garden feasibility, unlabeled-color signed class, embedded
Maxwell signature, and basis-free normalizer orbit have all failed as physical
selectors.

The primary-source audit found no independent physical holdout. Further
discriminating computation requires new physical input: an independently
sourced higher-dimensional fixture for `O`, `VM1`, `VM2`, or `VM3`, or a
complete target specification for a valid signing on another support. Until
that exists, do not infer missing spatial transformations from valise matrices.
See `docs/permutahedron-s8-source-fixture-audit.md` for the eligibility ledger.

The remaining targeted-spectral question is also closed. All 30 R8 conjugates
define equitable 5,040-coset partitions, and exact left-regular graph
automorphisms transport the standard partition to every conjugate partition.
All quotient graphs are therefore isomorphic. Any adjacency-polynomial filter
preserves this 30-fold ambiguity, so interior-eigenspace targeting cannot
canonically select the standard R8 family without external symmetry breaking.
See `docs/permutahedron-s8-project-closeout.md` for the final synthesis.
