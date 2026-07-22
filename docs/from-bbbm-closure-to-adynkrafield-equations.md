# From BBBM closure to an Adynkrafield equation

## Purpose

This document records the completed BBBM calculations and sets out the next
research steps. One problem is finite off-shell closure for all sixteen
supersymmetries. The broader and primary objective is to determine whether
Adynkrafields can express a supersymmetric field equation with the structural
role played by Einstein's equation in general relativity.

The second objective is not yet defined here as a unique mathematical problem.
The first task is therefore to define a candidate operator and state tests that
can disprove it. A larger graph catalog by itself will not supply a field
equation.

## Status update: 2026-07-21

The BBBM reproduction is complete through the full nonabelian nine-charge
algebra and the sixteen-charge on-shell closure calculation. A second validated
baseline is now available from Cigliano, Dahl, and Gates's ten-dimensional
linearized supergravity L/R construction [9]:

- the complete 16-color `82 x 176` and `176 x 82` matrices are generated in Rust;
- all 461,824 entries match the retained NumPy and SymPy cross-checks;
- all 136 bosonic Garden relations close exactly over `Q(sqrt(2))`;
- the complete fermionic nonclosure blocks are measured;
- the executable `1/16 MixedLeft` coefficient closes exactly;
- the alternative `1/8` coefficient stated in a source comment fails all 136
  bosonic charge-pair relations at 7,296 scalar entries;
- neither coefficient choice resolves the separate displayed-example
  discrepancies in Eqs. 6.0.5-6.0.6.

The 10D data baseline is documented in
[`2512-lr-reproduction.md`](2512-lr-reproduction.md). Its machine-readable
convention scan is `results/tendim_2512_convention_scan.json`.

These results change the work order. The direct finite sixteen-charge extension
remains a high-risk auxiliary-field problem. Its nonclosure decomposition is a
useful measurement program, but it is not assumed to yield the missing equation.
The primary equation route is the Adynkrafield differential program below:

1. reproduce the published four-dimensional `N=1` supergravity genomes;
2. implement supercovariant derivative maps and multiplicity intertwiners;
3. build the prepotential gauge, field-strength, and Bianchi complex;
4. reproduce a known linearized supergravity equation and quadratic action;
5. express that known equation as a tested Adynkrafield operator;
6. only then apply the same machinery to the eleven-dimensional scalar
   superfield and study its irreducible reduction;
7. treat nonlinear products and interaction terms only after the linearized
   operator and its Clebsch-Gordan data are validated.

In parallel, the validated 10D L/R matrices may be used for bounded embedding
searches. Every candidate must preserve the physical L/R block, cancel the
measured fermionic nonclosure, satisfy the full Garden relations, and state its
gauge, locality, zero-mode, and auxiliary-field assumptions. The permutahedron
atlas remains a separate supporting calculation.

## Established BBBM results

Baulieu, Berkovits, Bossard, and Martin give a ten-dimensional super-Yang-Mills
construction in which nine of sixteen supersymmetries close off shell after
seven auxiliary scalars are added [1]. The repository now contains the
following source-based reproductions.

### Nine-charge auxiliary system

The transformations in Eqs. (22)-(23) of [1] were implemented as graded
derivations of a noncommutative differential algebra. The calculation checks
all 45 symmetric charge pairs on all 33 component fields:

```text
45 charge pairs x 33 fields = 1,485 relations
```

Every relation reduces to the stated translation and gauge transformation.
The residual is zero without equations of motion, integration by parts, trace
cyclicity, numerical sampling, or a commutativity assumption. A separately
written free-word implementation and source-convention tests reproduce the
result.

The one-dimensional reduction has engineering heights `(9,16,7)`. Its linkage
matrices satisfy the nine-supercharge Garden relations coefficient by
coefficient over the polynomial differential ring. Formally lowering the seven
auxiliary nodes requires an inverse time derivative and does not identify zero
modes, so the resulting `(16|16)` valise is not a local equivalent of the
published system.

### Full sixteen-charge on-shell system

After the seven auxiliaries are eliminated, the ordinary ten-dimensional
super-Yang-Mills transformations act on ten gauge-potential components and
sixteen gaugino components. The implementation checks all 136 symmetric charge
pairs on all 26 fields:

```text
136 charge pairs x 26 fields = 3,536 relations
```

The result is:

- all 1,360 gauge-potential relations close as translations plus gauge
  transformations;
- all 2,176 gaugino relations factor through the gaugino Dirac equation;
- 1,120 gaugino component relations have a nonzero Dirac multiplier in the
  implemented sparse spinor basis;
- no term remains after the Dirac factors are subtracted;
- 2,366 checked relations involve at least one charge in the seven-dimensional
  tensor subspace.

This verifies the complete on-shell algebra. It is not a sixteen-charge
off-shell extension of the 33-field BBBM multiplet.

### Location of the obstruction

Under `Spin(7)`, the sixteen supersymmetry parameters split as

```text
16 = 1 + 8 + 7.
```

The scalar and vector sectors give the nine retained BBBM charges. The remaining
sector is an antiselfdual two-form representation of dimension seven. In the
covariant linear auxiliary-spinor solution of [1], the corresponding parameter
is set to zero. The paper does not provide a linear action of that sector on the
seven independent auxiliary fields.

The completed calculation sharpens the problem. The missing object is not a
table of seven transformations that can be transcribed from [1]. It is an
off-shell action of the tensor sector on independent auxiliaries that cancels
the measured Dirac-equation remainder while preserving gauge covariance,
locality, and all sixteen anticommutators.

The present implementation identifies the `1+8+7` subspaces in an octonionic
spinor basis. It does not yet provide the component intertwiner from each of the
seven spinor coordinates to BBBM's antiselfdual two-form notation.

## What the BBBM calculation establishes

1. The published nine-charge nonabelian algebra is reproduced on every
   component field.
2. The full sixteen-charge on-shell algebra is reproduced in the same
   computational framework.
3. The nonclosure of the gaugino sector is available as an explicit linear map
   into the Dirac equation.
4. BBBM's stated covariant linear auxiliary-spinor ansatz does not supply the
   missing tensor-sector action.

It does not establish that no finite off-shell extension exists. It excludes
neither different auxiliary representations nor nonvalise, constrained, or
rectangular complexes.

## Immediate calculation: the nonclosure representation

The next calculation should use the measured remainder rather than guess
another field count.

### Step 1: construct the tensor-charge intertwiner

Construct and verify the map between the implemented spinor-component charge
basis and BBBM's scalar, vector, and antiselfdual two-form basis. This will make
each of the seven excluded charges explicit and permit source-level comparison
of every tensor index.

**Acceptance test:** the map has ranks `1`, `8`, and `7`, preserves the
`Spin(7)` action, and reproduces the existing nine-charge sector.

### Step 2: extract the nonclosure module

For charge pairs in `9 x 7` and `7 x 7`, collect the Dirac multipliers as an
equivariant tensor rather than as 1,120 separate component identities. Compute
its rank, kernel, image, stabilizer, and irreducible `Spin(7)` decomposition.
Repeat the decomposition after restriction to the four-dimensional compact
subgroup used in the Siegel-Rocek study.

**Acceptance test:** reconstruction from the decomposed tensor reproduces every
gaugino remainder with zero residual.

### Step 3: derive the minimum auxiliary inventory

Treat cancellation of the nonclosure tensor as a representation problem. Solve
for the smallest bosonic and fermionic auxiliary modules that admit equivariant
maps into the measured image. Include engineering height, gauge-complex degree,
and boson-fermion balance as constraints.

**Acceptance test:** a proposed inventory contains every representation needed
to cancel the tensor, with no representation added solely to equalize a raw
dimension count.

### Step 4: solve the extension equations

Introduce the minimum inventory and solve the sixteen-charge linkage and gauge
compatibility equations. Begin at zero spatial momentum, then restore local
spatial derivatives. Do not assume BBBM's auxiliary-spinor ansatz.

**Acceptance test:** all sixteen charges close without the Maxwell equation,
the gaugino equation, an auxiliary equation of motion, inverse derivatives, or
zero-mode deletion.

This is the most direct current route to the unresolved auxiliary-field
question. It converts a general search into a finite representation and
polynomial-system problem.

## From representation data to a field equation

Adinkras and Adynkras organize supersymmetry representations. A field equation
requires more structure: differential operators, gauge equivalence, curvature
or field-strength objects, Bianchi identities, and an action or Euler-Lagrange
map.

The Adynkra literature identifies a specific missing step. The published
libraries determine representation content, but a complete account of the
component transformations requires supercovariant derivative operators acting
on Adynkrafields [5]. The 2024 supergravity-prepotential paper also states that
the representation-level method does not naturally provide Clebsch-Gordan
coefficients [6]. Those coefficients, or equivalent intertwiners, are needed to
recover local transformations and interactions when representations occur with
multiplicity.

The next program should therefore proceed in the following order.

### A. Implement the Adynkrafield representation algebra

Implement the level-by-level genome operations used in [4]-[6]:

- Lorentz representations in Dynkin-label and Young-tableau form;
- tensor products and irreducible decomposition;
- exterior powers and plethysm for superspace levels;
- multiplicities and conjugation;
- embeddings, projections, and candidate gauge complexes.

The initial benchmark should reproduce the published four-dimensional,
`N=1` supergravity-prepotential genome in [6].

### B. Add supercovariant differential operators

Construct representation-valued supercovariant derivative maps between genome
levels. Multiplicity spaces must carry explicit intertwiners rather than only
dimension labels. Verify the supersymmetry anticommutators and known component
transformations on the four-dimensional benchmark.

This addresses the technical gap stated in [5]. It is also the point at which
an Adynkrafield becomes more than a representation inventory.

### C. Build the gauge and curvature complex

For the supergravity prepotential, encode:

```text
gauge parameter -> prepotential -> gauge-invariant field strength
                -> Bianchi relations
```

Compute kernels, images, and cohomology at each representation level. Compare
the result with the known linearized supergravity multiplet.

### D. Define the first equation candidate

The first defensible target is a linearized, supersymmetry-covariant
Euler-Lagrange or curvature projection written entirely in Adynkrafield data.
It must not be named as the desired equation until it passes all of the
following tests:

1. supersymmetry covariance;
2. gauge invariance or a stated gauge-covariant transformation law;
3. compatible Bianchi identities;
4. the correct physical cohomology and on-shell degrees of freedom;
5. agreement with a known linearized four-dimensional supergravity equation;
6. derivation from, or compatibility with, a local invariant quadratic action.

A nonlinear extension would then require products of Adynkrafields and their
intertwiners. Representation multiplicities without Clebsch-Gordan data are not
enough for that step.

### E. Apply the validated operators to the eleven-dimensional scalar superfield

After the four-dimensional benchmark passes, apply the same derivative,
gauge-complex, and reducibility machinery to the proposed eleven-dimensional
scalar superfield `U`. Reference [6] identifies its complete reduction to an
irreducible supersymmetry representation as the next major problem. Locate the
graviton sector in the resulting quotient and determine whether the validated
curvature construction extends to it.

## Role of the permutahedron program

The `S_8` permutahedron has 40,320 vertices and is the natural finite object in
the four-dimensional `N=2` program [2,3,7,8]. The finite atlas is now complete:

1. all 40,320 `S_8` vertices and 141,120 weak-Bruhat edges are enumerated;
2. the published `S_4` atlas and all 300 independent correlators are reproduced;
3. `R_8` partitions `S_8` into 5,040 checked octets, with 168 left-right
   coincident slices and 30 conjugate `R_8` subgroups;
4. the six published `N=2` octets and the Diadem octet are mapped into the
   atlas;
5. the requested `56 x 56` two-point matrix for those seven octets is computed;
6. a searchable browser atlas covers the complete vertex, edge, and coset data.
7. all 5,040 right `R_8` cosets admit Garden signs; every affine sign system has
   rank 45 and nullity 19;
8. the 168 coincident slices are precisely the cosets in
   `N_{S_8}(R_8)/R_8`, which is `GL(3,2)`.

The implementation and validation report are documented in
[`permutahedron-atlas.md`](permutahedron-atlas.md).

This track can reveal representation embeddings and invariant structure. It
does not by itself provide a differential field equation. It should run as a
supporting atlas and validation program while the nonclosure and Adynkrafield
differential work proceeds. The scoped `R_8` sign-feasibility and 168-coset
questions are complete. The closure of particular Boolean-factor assignments,
their four-dimensional interpretation, and the relation of the atlas to
nonclosure remain separate questions.

## Status update: four-dimensional genome baseline

Track 2, Step 1 is complete. The Rust representation engine reproduces all six
four-dimensional `N=1` genomes in Eqs. (3.6)-(3.11) of [6]. A separate literal
source fixture checks all 47 terms, their left and right level degrees,
multiplicities, and factorial coefficients. The generated dimensions are 4, 8,
8, 16, 32, and 64 for the chiral, 2-form gauge, 1-form variant gauge, 1-form
gauge, matter-gravitino, and supergravity genomes, respectively.

The implementation is documented in
[`adynkra-4d-n1-genomes.md`](adynkra-4d-n1-genomes.md). This completes the
representation-inventory baseline only. Track 2, Step 2, supercovariant
derivatives and multiplicity intertwiners, remains open.

The source-convention supercovariant derivative algebra in Eq. (2.22) is also
implemented. All 160 anticommutator relations on the complete Grassmann basis
close exactly. This completes the derivative-algebra baseline for Step 2, but
not the irreducible multiplicity intertwiners. See
[`adynkra-4d-n1-derivatives.md`](adynkra-4d-n1-derivatives.md).

The first explicit irreducible intertwiners are also complete for the bosonic
rank-two sector in Eqs. (2.5) and (2.18). Four exact projectors split
`[1,1] tensor [1,1]` into ranks 1, 9, 3, and 3, commute with all six `so(4)`
generators, and reconstruct the full tensor space with zero residual. See
[`adynkra-4d-n1-intertwiners.md`](adynkra-4d-n1-intertwiners.md).

The vector-spinor decompositions in Eqs. (2.13), (2.14), and (2.19) are now
implemented as complementary exact projectors for both chiralities. Their
ranks are 2 and 6, and all completeness, orthogonality, idempotency, and
`sl(2)_L + sl(2)_R` equivariance checks pass.

The fundamental derivative intertwiners are now implemented in an exact
rational binary-form basis for every Lorentz representation in the six
published genomes. All 18 left- and right-handed tensor products pass
projection, reconstruction, cross-channel, and equivariance checks. The three
repeated representations at total level two are distinguished canonically by
their `(2,0)` and `(0,2)` bidegrees; the combined chirality selectors have full
rank. This validation does not by itself fix named-component normalizations;
the gauge map is implemented directly in superspace below. See
[`adynkra-4d-n1-derivative-intertwiners.md`](adynkra-4d-n1-derivative-intertwiners.md).

The first gauge-curvature sequence is also implemented. The complete
64-component prepotential gauge map in Eq. (2.21) contains 192 sparse
differential terms and obeys every Grassmann-bidegree selection rule. The
chiral super-Weyl curvature in Eq. (5.2.5) of *Superspace* is nonzero, chiral
on all 64 prepotential basis inputs, and annihilates the complete 64-component
gauge image. The calculation checks 512 chirality relations and 256 gauge-
invariance relations with zero residuals in exact arithmetic. The remaining
supergravity curvatures, Bianchi identities, compensator choice, and
cohomology are open. See
[`adynkra-4d-n1-gauge-curvature.md`](adynkra-4d-n1-gauge-curvature.md).

## Ordered research plan

### Track 0: validated 10D supergravity baseline and embedding scan

1. Generate and verify the complete L/R artifact in Rust. **Complete.**
2. Compare the executable `1/16` and source-comment `1/8` branches. **Complete:**
   only `1/16` satisfies the bosonic Garden relations.
3. Obtain author clarification of the remaining printed L-row and R-coefficient
   discrepancies, without blocking source-convention experiments.
4. Decompose the measured fermionic nonclosure under the relevant Lorentz and
   compact subgroups.
5. Define explicit auxiliary-block embedding ansatzes and solve them as bounded
   algebraic problems.
6. Reject any candidate requiring equations of motion, inverse derivatives,
   undeclared zero-mode deletion, or an unbounded auxiliary tower.

### Track 1: sixteen-charge off-shell closure

1. Construct the explicit `1+8+7` charge-basis intertwiner.
2. Build and decompose the measured nonclosure module.
3. Derive the minimum auxiliary representation inventory.
4. Solve the zero-momentum algebraic extension.
5. Restore local spatial derivatives and verify full closure.
6. Establish Lorentz and R-symmetry covariance.
7. Construct an invariant local action.
8. Identify which assumption in the counting argument the construction changes.
9. Extend a successful abelian result to the nonabelian theory.

### Track 2: Adynkrafield equation

1. Reproduce the four-dimensional `N=1` supergravity-prepotential genome.
   **Complete for the six published genomes in Eqs. (3.6)-(3.11).**
2. Implement supercovariant derivatives and multiplicity intertwiners.
   **Derivative algebra, rank-two bosonic intertwiners, vector-spinor
   intertwiners, fundamental Clebsch-Gordan maps, and all repeated sectors in
   the six published genomes complete.**
3. Construct the prepotential gauge and curvature complex. **Gauge map and
   chiral super-Weyl curvature complete; remaining curvatures, Bianchi
   identities, compensator choice, and cohomology open.**
4. Reproduce a known linearized supergravity equation.
5. Express that equation as an Adynkrafield operator.
6. Apply the validated operators to the eleven-dimensional scalar superfield
   and test its complete irreducible reduction.
7. Determine which representation products and intertwiners are required for a
   nonlinear extension.

### Track 3: finite combinatorial atlas

1. Complete and validate the 40,320-vertex `S_8` atlas. **Complete.**
2. Reproduce the published correlator calculations. **Complete for the full
   `S_4` matrix and the seven named `S_8` octets.**
3. Test the proposed lower-color embeddings and the organization of nonclosure
   terms. **All 5,040 unsigned octets admit Garden signs, so ab-normality is not
   a sign-feasibility discriminator. The nonclosure of particular published
   Boolean factors remains separate.**
4. Feed any invariant structures into Tracks 1 and 2.

The immediate equation priority is Track 2, Steps 1-5. Track 0 provides a
validated 10D measurement and bounded embedding program. Track 1, Steps 1-3,
remain useful for decomposing the BBBM obstruction, but a finite extension is
not presumed. Track 3 is bounded and valuable, but it should not displace the
differential-operator work.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

```bash
cargo run --release -- bbbm
cargo run --release -- bbbm-holoraumy
cargo run --release -- bbbm-closure
cargo run --release -- bbbm-nonabelian
cargo run --release -- bbbm-sixteen-onshell
cargo run --release -- tendim-generate
cargo run --release -- tendim-reproduce
cargo run --release -- tendim-convention-scan
cargo run --release -- perm-atlas-build
cargo run --release -- perm-atlas-verify
cargo run --release -- perm-garden-scan
cargo run --release -- adynkra-genome-build
cargo run --release -- adynkra-genome-verify
cargo run --release -- adynkra-derivative-verify
cargo run --release -- adynkra-intertwiner-verify
cargo run --release -- adynkra-vector-spinor-verify
cargo test
```

## References

1. L. Baulieu, N. Berkovits, G. Bossard, and A. Martin, "Ten-dimensional
   super-Yang-Mills with nine off-shell supersymmetries,"
   [arXiv:0705.2002](https://arxiv.org/abs/0705.2002).
2. D. D. Bristow, J. H. Caporaletti, A. J. Cianciara, S. J. Gates Jr., D.
   Levine, and G. Yerger, "A Note On Exemplary Off-Shell Constructions Of 4D,
   N = 2 Supersymmetry Representations,"
   [arXiv:2012.14015](https://arxiv.org/abs/2012.14015).
3. A. J. Cianciara, Z. Coleman, S. J. Gates Jr., Y. Lee, and Z. Zhang, "N = 2
   SUSY & the Hexipentisteriruncicantitruncated 7-Simplex,"
   [arXiv:2304.09830](https://arxiv.org/abs/2304.09830).
4. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Advening to Adynkrafields: Young
   Tableaux to Component Fields of the 10D, N = 1 Scalar Superfield,"
   [arXiv:2006.03609](https://arxiv.org/abs/2006.03609).
5. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Component Decompositions and
   Adynkra Libraries for Supermultiplets in Lower Dimensional Superspaces,"
   [arXiv:2007.07390](https://arxiv.org/abs/2007.07390).
6. S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D,
   N = 1 Supergravity Superfield Prepotential,"
   [arXiv:2407.09334](https://arxiv.org/abs/2407.09334).
7. A. J. Cianciara, S. J. Gates Jr., Y. Hu, and R. Kirk, "The 300
   'Correlators' Suggests 4D, N = 1 SUSY Is a Solution to a Set of Sudoku
   Puzzles," [arXiv:2012.13308](https://arxiv.org/abs/2012.13308).
8. A. J. Cianciara, S. J. Gates Jr., Y. Lee, E. T. Levy, T. O. Razzaz, and J.
   Richardson, "Unfolded Adinkra Properties of Supermultiplets (I),"
   [arXiv:2311.06842](https://arxiv.org/abs/2311.06842).
9. J. Cigliano, B. Dahl, and S. J. Gates Jr., "10D Supergravity Numerical Data
   Sets for L & R Matrices,"
   [arXiv:2512.12157](https://arxiv.org/abs/2512.12157).
