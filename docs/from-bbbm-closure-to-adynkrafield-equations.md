# From BBBM closure to an Adynkrafield equation

## Purpose

This document records the completed BBBM calculations and sets out the next
research steps. The immediate problem is finite off-shell closure for all
sixteen supersymmetries. The broader objective is to determine whether
Adynkrafields can express a supersymmetric field equation with the structural
role played by Einstein's equation in general relativity.

The second objective is not yet defined here as a unique mathematical problem.
The first task is therefore to define a candidate operator and state tests that
can disprove it. A larger graph catalog by itself will not supply a field
equation.

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
the four-dimensional `N=2` program [2,3,7,8]. Completing its atlas is feasible
and directly responsive to the work Gates identified:

1. enumerate all `S_8` vertices and the selected Bruhat or permutahedron edges;
2. reproduce the published lower-order correlators and representation fixtures;
3. compute the stated two-point correlator matrices for the six `N=2`
   representations;
4. test embeddings of lower-color structures and organize nonclosure terms.

This track can reveal representation embeddings and invariant structure. It
does not by itself provide a differential field equation. It should run as a
supporting atlas and validation program while the nonclosure and Adynkrafield
differential work proceeds.

## Ordered research plan

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
2. Implement supercovariant derivatives and multiplicity intertwiners.
3. Construct the prepotential gauge and curvature complex.
4. Reproduce a known linearized supergravity equation.
5. Express that equation as an Adynkrafield operator.
6. Apply the validated operators to the eleven-dimensional scalar superfield
   and test its complete irreducible reduction.
7. Determine which representation products and intertwiners are required for a
   nonlinear extension.

### Track 3: finite combinatorial atlas

1. Complete and validate the 40,320-vertex `S_8` atlas.
2. Reproduce the published correlator calculations.
3. Test the proposed lower-color embeddings and the organization of nonclosure
   terms.
4. Feed any invariant structures into Tracks 1 and 2.

The immediate priority is Track 1, Steps 1-3. They use the new full-closure
calculation and can either produce a minimal auxiliary target or rule out a
well-defined class of targets. Track 2 should begin with its published
four-dimensional benchmark in parallel. Track 3 is bounded and valuable, but
it should not displace the differential-operator work.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

```bash
cargo run --release -- bbbm
cargo run --release -- bbbm-holoraumy
cargo run --release -- bbbm-closure
cargo run --release -- bbbm-nonabelian
cargo run --release -- bbbm-sixteen-onshell
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
