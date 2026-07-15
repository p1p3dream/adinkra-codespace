# Testing a Minimal N=16 Adinkra Completion

**Précis prepared for S. James Gates Jr.**

## Summary

Following the question posed in the 2014 and 2015 adinkra papers, I tested the
first N=16 valise size suggested by counting auxiliary fermionic units without
pairing them. Within the minimal monomial Garden class, the literal-subblock
completion fails for both chromotopologies and every dashing class tested. A
formulation using linear injection and projection maps has a real solution of
the projected one-dimensional linkage equations for the E8 x E8 topology. A
Banach fixed-point calculation establishes the existence of that solution. The
full valise field
spaces have SO(9) quadratic
Casimir 18, while the physical vector and spinor have Casimirs 8 and 9. Hence
every SO(9)-equivariant linear map between these representations is zero.

The projected one-dimensional equations are therefore consistent, but the
injection and projection maps cannot be made SO(9)-equivariant on the tested
valise. Ordinary node raising and finite local spatial derivatives do not
produce a linear retraction on the same potential representation, even modulo
gauge and Bianchi corrections
that vanish at zero spatial momentum. This is not a general no-go for a
gauge-extended representation with different field content or maps that remain
nonzero at zero momentum, or
for a nonlocal construction that removes the spatial zero mode.

A character calculation identifies the Spin(9) representation carried by the valise:
its Spin(9) content is `44 + 84 | 128`, with symmetric-traceless tensor,
three-form, and gamma-traceless vector-spinor sectors. This is the
state content of the eleven-dimensional supergravity multiplet, not an SO(9)
module containing the vector `9` and spinor `16` required by the time-reduced
super-Yang--Mills transformations. The mismatch is independent of the
particular real injection and projection maps.

Under `Spin(3) x Spin(6)`, the physical real gaugino 16 occurs
once, but the bosonic `(3,1)` spatial vector and `(1,6)` six-scalar sectors both
have multiplicity zero. The tested `128|128` valise therefore has no ordinary
`Spin(3) x Spin(6)`-equivariant linear retraction onto the four-dimensional
physical fields. A gauge or auxiliary complex with maps that remain nonzero at
zero momentum has not been tested.

## Motivation and scope

For the 4D N=4 abelian vector multiplet, the reviewed count is

```text
16 + 32 n = 128 m,
```

where auxiliary fermions occur in pairs. Provisionally counting the
16-component units individually gives

```text
16 + 16 q = 128 m,
```

whose smallest solution is `m = 1`, `q = 7`. This makes dimension 128 the first
case to test. In the `16 + 4p` notation of the 2014 paper, it is `p = 28`.
This arithmetic selects a size. It does not establish that seven
unpaired auxiliary spinors are dynamically or Lorentz-covariantly admissible.

The recent *Adinkras & Genomics in Sixteen Color Systems (I)* distinguishes a
literal subgraph embedding from encoding by quotient operations. The linear
injection and projection maps tested below are not the graph quotient defined
in that paper.

The count is a four-dimensional statement. The target matrices used below are
obtained by reducing the free abelian ten-dimensional N=1 super-Yang--Mills
transformations to one time dimension. Extension back to ten dimensions is a
separate problem.

## Computation

Nine real symmetric 16 by 16 SO(9) gamma matrices define the projected
one-dimensional linkage matrices between nine spatial gauge components and
sixteen gauginos. I tested the two
self-dual length-16 chromotopologies at dimension `128|128`, using all 256
dashing cohomology classes for each.

Results for literal and mixed embeddings:

1. No set of nine bosonic coordinates has the required support, independently
   of the dashing.
2. Even with arbitrary real mixing of the nine physical bosons and sixteen
   physical fermions, the standard no-leakage embedding has only the zero
   solution for all 512 dashings.

The tested subblock-type completion has no solution.

The quotient test introduces injections and projections

```text
J_B : R^9  -> R^128       P_B : R^128 -> R^9
J_F : R^16 -> R^128       P_F : R^128 -> R^16
```

with `P_B J_B = I_9` and `P_F J_F = I_16`. For each supercharge, the two
projected full-Garden linkage matrices must reproduce the 9 by 16 SO(9) gamma
matrices:

```text
P_F Lhat_alpha^T J_B = G_alpha^T,
P_B Lhat_alpha   J_F = G_alpha.
```

For E8 x E8, catalog entry 75 with dashing 0, a coordinate `J_B` and
rank-16 `P_F` satisfy the first equation, including simultaneous
support-and-sign compatibility across all nine bosons and sixteen
supercharges. The projection has sixteen signed fibers of size eight. Its
kernel has dimension 112, necessarily, because `128 - 16 = 112`. Writing this
as `7 x 16` is bookkeeping, not independent evidence for seven auxiliary
spinors. For D16, the corresponding support condition has no solution within
this coordinate-boson ansatz, independently of the dashing.

For this E8 x E8 projection, no coordinate right inverse for `P_F` satisfies
the remaining linkage equations. The uniform right inverse
`J_F = P_F^T / 8` also fails: its rational coefficient matrix has rank 128,
while the augmented matrix has rank 129.

Removing those restrictions leaves 2,304 bilinear linkage equations in 2,863
affine variables. A search found a point with maximum residual `1.11e-15` and a
Jacobian of full row rank. A separate Banach fixed-point calculation establishes
the existence of a real zero near the numerical solution. The residual at the
center is
evaluated as a dyadic rational; the inverse defect uses standard
IEEE-754 error bounds; and the final contraction tests are rational
inequalities.

```text
||F(x0)||_infinity < 1e-14
||B||_infinity     < 1024
||I - B J(x0)||    < 1e-6
machine-checked Lipschitz bound: 430
adopted Lipschitz bound: 448
validated radius: 3e-11
contraction bound: 46133 / 3125000000 < 1
```

Both retraction identities hold by construction. Since the system has 559 more
variables than equations, this establishes consistency of the projected
linkage system, not uniqueness
or a higher-dimensional field interpretation.

## SO(9) equivariance condition

The sixteen Garden colors carry the same Spin(9) spinor action as the physical
supercharges. That color action induces skew generators on the full 128 bosons
and 128 fermions. Their covariance and so(9) commutators were verified symbolically.
The homogeneous covariant-pair space has dimension one and is spanned by the
identity, so the skew field-action lift is unique.

The quadratic Casimirs are

```text
physical vector:    8
physical spinor:    9
full boson space:  18
full fermion space:18
```

An intertwiner must commute with the quadratic Casimir. The unequal scalar
eigenvalues therefore force `J_B`, `P_B`, `J_F`, and `P_F` to vanish if they are
required to be SO(9)-equivariant. This excludes every choice of injection and
projection maps on this valise representation, not only the numerical solution
described above. Direct substitution of that solution also fails all 36
rotation-generator intertwining equations.

Ordinary node raising does not change this result. If `D = d/dt`, a local
nonvalise map has the form `T(D) = sum_k T_k D^k`. Spatial SO(9) commutes with
`D`, so Casimir intertwining gives `(c_target - c_source) T_k = 0` for every
coefficient. The bosonic difference is 10 and the fermionic difference is 9;
therefore every coefficient vanishes. The same argument holds over the Laurent
ring allowing `D^-1`. Node raising or lowering within the same `128|128` module
does not change the Casimir mismatch.

Finite local spatial derivatives also fail on the tested representation. Write such a
map as `T(p,D)` over
`R[D,p_1,...,p_9]`. Evaluating at `p=0` leaves an equivariant map over `R[D]`,
which the preceding result forces to zero. Consequently

```text
(P_B J_B)(0,D) = 0, not I_9,
(P_F J_F)(0,D) = 0, not I_16.
```

For the nine spatial potentials, the gauge shift `delta A_i = p_i epsilon`
also vanishes at `p=0`. The same is true of Bianchi corrections with positive
spatial derivative degree. Hence an identity `P J = I + K` with
`K` in the spatial ideal `(p_1,...,p_9)` would evaluate to `0 = I`. This closes
the local polynomial retraction on the same potential representation under the
stated class of gauge and Bianchi corrections.

This statement does not exclude
adding gauge/Bianchi or auxiliary representations, using a larger rectangular
module, admitting an algebraic correction that remains nonzero at zero spatial
momentum, or localizing away from `p=0` with inverse spatial operators or
boundary conditions. The next calculation separately tests the ordinary
exterior differential complex for potentials, field strengths, and Bianchi
identities.

## Representation and gauge-complex follow-up

The joint Cartan character was computed from `M_01`, `M_23`, `M_45`, and
`M_67`. Individual characteristic polynomials and a separating base-32 Cartan
combination agree with

```text
bosons:   Sym^2_0(9) + Lambda^3(9) = 44 + 84,
fermions: (9 tensor 16) - 16 = 128.
```

The ordinary abelian gauge complex was then built over
`R[D,p_1,...,p_9]`:

```text
gauge parameter -> potential -> field strength -> Bianchi
      1              1+9           9+36          36+84.
```

Its differential squares to zero symbolically and has ranks `1,9,36` at generic
nonzero momentum. Only the `84` occurs in the tested bosonic valise
representation. The
`1`, `9`, `36`, and physical fermion `16` do not.

A chain-homotopy retraction identity also fails with derivative terms alone. At
total momentum `(D,p)=0`, every exterior differential vanishes, so
`P J - I = d H + H d` reduces to the ordinary linear retraction already ruled
out by the Casimir mismatch. Changing from potentials to ordinary field
strengths is
therefore insufficient unless the field representation or its algebra at zero
momentum is
also changed.

The next conventional valise size retaining the current block is `256|256`.
Direct sums of the irreducible block and its parity reversal contain only the
Spin(9) sectors `44`, `84`, and `128`. None contains a bosonic `9` and a
fermionic `16`, so this smallest direct-sum valise enlargement also fails.

A remaining case is a nonvalise or gauge-extended representation that
contains `9|16` at zero momentum, or an algebraic auxiliary/gauge differential
that remains nonzero at zero momentum.

## Direct 4D subgroup test

Because the original counting problem is four-dimensional, I separately
restricted the full field action to

```text
Spin(3) x Spin(6) ~= SU(2) x SU(4)_R.
```

Restricted Cartan characters and joint-Casimir eigenspaces give

```text
44 -> (5,1) + (3,6) + (1,20') + (1,1),
84 -> (1,1) + (3,6) + (3,15) + (1,20),
128 -> (4,4) + (4,4bar) + (2,4) + (2,4bar)
       + (2,20) + (2,20bar).
```

The real physical gaugino `(2,4)+(2,4bar)` has multiplicity one. The required
bosons do not occur: the `(3,1)` spatial-vector eigenspace and `(1,6)`
six-scalar eigenspace both have dimension zero. Compact-group complete
reducibility therefore excludes an ordinary
`Spin(3) x Spin(6)`-equivariant linear retraction from the tested valise onto
the four-dimensional physical fields, separately from the Spin(9) result.

This does not exclude a nonvalise gauge or auxiliary complex whose algebraic
zero-momentum maps change the representation or place the physical fields in
cohomology rather than as an ordinary field-space subrepresentation or
quotient.

## Interpretation and questions

Results within the tested class:

* the minimal literal-subspace completion remains obstructed;
* the projected one-dimensional linkage equations have a real solution;
* the corresponding injection and projection maps have no SO(9)-equivariant
  realization in the same valise representation;
* local time or spatial derivatives do not fix the zero-momentum fiber of that
  module, even modulo positive-spatial-degree gauge/Bianchi corrections;
* the full valise is `44+84|128`, and the ordinary derivative gauge
  complex and the first direct-sum valise enlargement do not supply `9|16`;
* restriction to `Spin(3) x Spin(6)` supplies the physical gaugino 16 but not
  the spatial vector or six scalars, so an ordinary direct four-dimensional
  linear retraction also fails.

It does not establish or exclude a rectangular or gauge-extended
nonvalise construction with changed representation content. It also does not
establish Spin(1,9) covariance, full gauge and Bianchi closure, an invariant
local action, or nonabelian interactions.

Questions:

1. Is the Casimir mismatch the expected representation-theoretic reason that a
   valise quotient cannot implement the 2014 and 2015 embedding problem?
2. Is the appearance of the `44+84|128` eleven-dimensional-supergravity
   state content the expected consequence of restricting the irreducible N=16
   valise Clifford module along the Spin(9) spinor embedding?
3. The direct four-dimensional compact-subgroup calculation lacks the required
   bosonic representations. Does a viable construction require a
   four-dimensional gauge and auxiliary complex with maps that remain nonzero
   at zero momentum, and if so, which Lorentz and R-symmetry
   representations should replace or augment the valise before another
   linkage search?

## Reproducibility and context

The implementation records all 512 embedding tests, the E8 x E8 injection and
projection maps, the rational rank calculation, the numerical solution, and
the Banach fixed-point bounds. The SO(9) calculation records the induced field
action, Lie-algebra closure, all 36 failed intertwining equations, and the
Casimir mismatch. A separate calculation over the time-derivative ring records
the node-raising result. The zero-spatial-momentum calculation records the
local polynomial result and its stated gauge and Bianchi scope. Symbolic
Cartan and gauge-complex calculations record the `44+84|128` decomposition,
exterior complex, zero-momentum chain-homotopy result, and `256|256` direct-sum
result. A direct four-dimensional branching calculation records the subgroup decomposition,
the single physical gaugino sector, and the absent physical bosons. The Rust
suite reports 238 passed, 0 failed, and 6 ignored slow tests.

Primary context: Siegel and Roček, *Phys. Lett. B* 105 (1981) 275; Calkins,
D. E. A. Gates, S. J. Gates Jr., and McPeak, *JHEP* 05 (2014) 057,
arXiv:1402.5765; Calkins, D. E. A. Gates, S. J. Gates Jr., and Golding,
*JHEP* 04 (2015) 056, arXiv:1502.04164; Arunseangroj, Bedessem, Gates, and
Yerger, arXiv:2503.13797; Plefka and Waldron,
*Asymptotic Supergraviton States in Matrix Theory*, arXiv:hep-th/9801093.
