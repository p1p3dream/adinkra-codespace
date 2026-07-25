# N=16 Adinkra research through 2026-07-13: précis

## Cutoff and scope

This document records the state of the broader `adinkra-codespace` research
program at the end of 2026-07-13. It excludes the Siegel-Roček, Spin(9),
locality, gauge-complex, representation-decomposition, and direct
four-dimensional calculations created on 2026-07-14.

The cutoff has one timeline ambiguity. The last repository commit before the
cutoff is `82986a6` from 2026-07-02. Work on 2026-07-12 and 2026-07-13 was
recorded in session history, project memory, and documents under `~/Documents`,
but it did not produce another repository commit. The long-running session log
was modified again on 2026-07-14, so its filesystem timestamp alone is not a
reliable boundary. The account below uses message timestamps, the commit log,
file history, and the dated planning documents. Where July 13 work remained a
plan or specification, it is labeled as such.

## Background

The project studies one-dimensional $N=16$ supersymmetry representations
encoded by Adinkras. A chromotopology is obtained from a quotient of the
16-cube by a doubly-even binary code. A complete Adinkra contains more data than
the code: at minimum it also has a dashing and a hanging or ranking.

The checked-in catalog contains 145 permutation-equivalence classes of
positive-dimensional doubly-even codes of length 16 and dimensions
$k=1,...,8$:

```text
k:       1   2   3   4   5   6   7   8
classes: 4  10  23  38  36  23   9   2
```

Thus, "145" counts code classes, and therefore chromotopology classes. It does
not count complete Adinkras. The dimension-zero code, whose chromotopology is
the unquotiented 16-cube, is not included. Accounting for the $2^k$ dashing
cohomology classes associated with each catalogued code gives 5,128
code-and-dashing combinations before height assignments are included.

For a length-16 code of dimension $k$, the associated valise has

\[
d=2^{15-k}
\]

bosons and the same number of fermions. The two $k=8$ classes therefore give
the minimal $d=128$ case.

## Implementation established by the cutoff

### Catalog and Garden construction

The Rust code enumerates and canonicalizes doubly-even codes, constructs their
quotient chromotopologies, enumerates dashing cohomology classes, builds signed
permutation $L_I,R_I$ matrices, and verifies the Garden relations. The 145
class distribution matches the Miller reference counts. Integration tests
exercise the real construction path at low $N$, including odd face dashings,
Garden closure, holoraumy, and self-gadget checks.

The browser interfaces display the catalog without requiring direct inspection
of the JSON file. The main interface filters all 145 records. Additional views
display the $k=8$ Gadget values, four selected three-dimensional Adinkras, and
spectral, diffusion, clustering, and summary plots. These views do not
establish dimensional enhancement.

### Irreducible structure and gadget computation

For $k<8$, the raw valise dimension exceeds the minimal $d=128$, so the
representation is reducible. The project implemented symbolic commutant
construction, numerical isotypic splitting, dense and disk-backed Gram paths,
and a commutant-only structural path that reaches the large low-$k$ strata.

The $k=8$ computation covers 512 minimal representations and reports 336
distinct off-diagonal gadget values. The two topologies were independently
matched to the E8 x E8 and D16 cases and their published distance spectra.

Below $k=8$, arbitrary cross-summand
gadget values depend on the relative basis orientation. Equivalent summands can
be aligned by an orthogonal intertwiner, after which their canonical
cross-gadget is 1. Values obtained before that alignment depend on the chosen
bases and are not invariants. The basis-independent information is the irreducible
class, its multiplicity and Schur type, and the already validated gadget between
inequivalent minimal classes.

### Worldsheet lifting

The Gates-Hübsch worldline-to-worldsheet conditions were implemented for
Adinkras with assigned heights. Each reported result includes the height
assignment and the left- and right-moving chirality assignment. A separate
implementation checks the height assignment and spin-sum equation.

By 2026-07-01, every one of the 145 catalogued code classes had a verified
height and chirality assignment with $p,q>0$ satisfying these conditions. The
last unresolved case was the all-ones $[16,1]$ code. A parity-gradient height
assignment satisfies those conditions for an $(8,8)$ split.

Under Corollary 2.5 of Gates and Hübsch, these data meet the condition for an
adinkraic worldsheet $(p,q)$ extension without central charges. The calculation
does not independently construct the worldsheet transformation laws, does not
show that the reported split is maximal over all height assignments, and does
not imply a four- or ten-dimensional off-shell extension.

### Central commutant and chromocharacter calculations

The antisymmetric part of the commutant was implemented as the
worldline-central operator count. It reproduces the quaternionic $N=4$
control and vanishes for the real irreducible $k=8$ modules. More generally,
the answer is determined by real Schur type and multiplicity. It can also vanish
for reducible sums of distinct real irreducibles, so it is not an irreducibility
test and does not establish dimensional enhancement.

The proposed $N=4$-inspired rank-four chromocharacter quantity was also
tested over all 5,128 valise code-and-dashing combinations. Its (Q), support, and raw
four-distinct-color trace activity all vanish at $N=16$. It therefore has no
power to distinguish classes in this catalog.

### Non-gauge dimensional-enhancement condition

The non-gauge dimensional-enhancement condition of Faux, Iga, and Landweber was
first checked against the published $N=4$ result: 4 of 60 minimal Adinkras with
specified heights and dashings satisfy the condition, while the valise height
assignments do not.

The implementation was then extended to the $N=16$, ten-dimensional case using
nine spatial Clifford directions. A sparse linkage implementation reduced the
cost to $O(d)$ and was checked entrywise against the dense calculation on a
tractable $k=8$ case. Validation conditions included the worldline Garden anchor,
zero residual on the expected Lambda support, and responsiveness of the full
Frobenius residual.

For each of the 145 code classes, the calculation used dashing class 0 and
tested the valise, any valid parity-gradient height assignment, and structured
source-raised height assignments. No tested combination satisfied the
non-gauge condition. Its scope is:

- neither the dashing classes nor the height assignments were searched
  exhaustively;
- the gauge and phantom sector was not implemented;
- the calculation tests a necessary non-gauge enhancement condition, not the
  Siegel-Roček no-go theorem;
- the residual magnitude was used only as a consistency check, while zero versus
  nonzero determined whether the condition was satisfied.

### Ten-dimensional supergravity reference data

The repository also contains a regenerated non-square $82$ by $176$
ten-dimensional supergravity $L/R$ dataset derived from pinned upstream
generative Mathematica. Deterministic token snapping, a canonical content hash, and the
bosonic Garden relation were verified. The data were used to check conventions
and software behavior. They describe a supergravity system, not a
super-Yang--Mills supermultiplet, and therefore do not establish the desired
extension. The disagreement with worked examples printed in the accompanying
paper remained unresolved, so no claim of entrywise agreement with the printed
examples was made.

## Direction mapped on 2026-07-12 and 2026-07-13

Several Maxwell-inspired ideas were reviewed. Identifying the $N=16$
chromocharacter with a theta term and treating electromagnetic duality as an
Adinkra operation were rejected because the compared quantities have different
definitions. The remaining technically applicable case was the phantom-field
extension of the published dimensional-enhancement formalism, because Maxwell
field strengths lie outside the scope of the non-gauge condition.

On 2026-07-13 the next proposed reference calculation became the
Baulieu-Berkovits-Bossard-Martin partial off-shell construction for ten-dimensional
SYM. It retains 9 of 16 supercharges and introduces seven auxiliary scalars,
with residual symmetry $SO(1,1)$ by $Spin(7)$. The intended program was:

1. reproduce the published 9-of-16 closure before applying the method to new
   representations;
2. encode its polynomial differential operators, Spin(7) projector, gauge
   action, and full closure identities;
3. only then formulate a catalog-wide calculation with an independent check of
   every reported closure identity.

This was specification work. No implementation of the Baulieu-Berkovits-
Bossard-Martin construction or its 9-of-16 closure existed by the cutoff.
Review identified two risks. First, the
gauge-potential quotient over the time-derivative polynomial ring retains a
zero-momentum torsion sector unless a localization, boundary condition, or explicit
gauge-complex policy is chosen. Second, every $N=16$ worldline Garden
representation can be restricted to nine supercharges, so the selection of
nine colors alone does not reproduce the published field transformations or
their closure. The existing non-gauge `Sieve10D` calculation does not implement
this problem.

The July 13 endpoint was therefore a requirement, not a result: reproduce the
gauge quotient and the full field-level closure of the published construction
before searching the catalog.

## Status at the cutoff

The project had a tested combinatorial and representation-theoretic
implementation. For every catalogued code class, it also had explicit height
and chirality data satisfying the published worldsheet-extension conditions.
The sampled non-gauge ten-dimensional enhancement calculation was negative. It
did not have:

- a gauge or phantom enhancement implementation;
- an implementation of the Baulieu-Berkovits-Bossard-Martin 9-of-16
  construction;
- an exhaustive search over all hangings;
- full Lorentz-covariant ten-dimensional SYM closure;
- a finite local off-shell $N=4$ or ten-dimensional SYM multiplet;
- an identified error in the Siegel-Roček reasoning.

The next work was to reproduce a published gauge-aware closure,
settle its zero-momentum and quotient conventions, and record every field-level
closure identity in independently checkable form. Only after reproducing that
published result could a catalog-wide extension be interpreted physically.

## Reproduction commands available at the cutoff

The code baseline is commit `82986a6`. From that revision:

```sh
cargo test
cargo run --release -- validate-miller 16
cargo run --release -- pipeline-k 8 adinkra_codes_n16.json > /tmp/k8.json
```

Worldsheet-extension search and the constructive $k=1$ checks:

```sh
for k in 1 2 3 4 5 6 7 8; do
  ADINKRA_LIFT_CHAINS=128 ADINKRA_LIFT_MAXRANK=8000 \
    cargo run --release -- lift-scan "$k" adinkra_codes_n16.json \
    > "/tmp/lift-k${k}.json"
done
cargo run --release -- lift-construct 1 adinkra_codes_n16.json
cargo run --release -- lift-search 113 0 1 adinkra_codes_n16.json
cargo test pipeline::construct_tests
```

Full class-by-class non-gauge enhancement sweep:

```sh
for k in 1 2 3 4 5 6 7 8; do
  cargo run --release -- enhance-scan "$k" adinkra_codes_n16.json
done
```

Representation and invariant calculations:

```sh
for k in 1 2 3 4 5 6 7 8; do
  cargo run --release -- decompose-structure "$k" adinkra_codes_n16.json
  cargo run --release -- q-scan "$k" adinkra_codes_n16.json --no-struct \
    > "/tmp/q-k${k}.json"
  cargo run --release -- central-charge "$k" adinkra_codes_n16.json
done
```

Ten-dimensional reference-data verification and the browser interface:

```sh
python3 scripts/eval_garden_exact.py
python3 -m http.server 8000
```

The explorer is then available at
`http://localhost:8000/visualizer/`. The scientific Python visualizations do
not share a single pinned environment, and the repository did not yet provide
one command that regenerated every catalog, analysis, and browser artifact.
