# Minimal Siegel-Rocek auxiliary-field investigation

## Question

What remains viable if the paired-auxiliary-spinor premise in the conventional
Siegel-Rocek counting argument is removed?

The paired count is

```text
16 + 32 n = 128 m
```

and has no nonnegative integer solution. Without that premise, the first case is

```text
16 + 16 n = 128 m,
m = 1,
n = 7.
```

The smallest target is therefore an N=16 Garden module with 128 bosons and 128
fermions.

## Time-reduced target transformations

`src/sr_hole.rs` constructs nine real symmetric 16 by 16 SO(9) gamma matrices
from imaginary-octonion left multiplication. It then forms the zero-brane 10D
SYM linkage matrices after reduction to one time dimension

```text
(L_alpha)[i,beta] = (gamma_i)[alpha,beta],
R_alpha = L_alpha^T.
```

The implementation verifies the identity:

```text
L_alpha R_beta + L_beta R_alpha = 2 delta_alpha,beta I_9.
```

The fermionic relation fails, as expected for the on-shell transformations. For every one of
the sixteen diagonal charge pairs, its remnant has rank 7.

## Minimal embedding test

The command tests both self-dual length-16 chromotopologies, catalog entries 75
and 76, and all 256 inequivalent dashings of each.

Two nested ansatzes are checked.

### Literal coordinate subblock

A literal embedding needs nine boson coordinate vertices whose complete sets of
sixteen colored fermion neighbors agree. The largest multiplicities are:

```text
catalog 75: 1
catalog 76: 2
required:   9
```

So the literal monomial-subblock ansatz is excluded independently of dashing.

### Arbitrary real field mixing

Let `b_i` be nine arbitrary vectors in the 128-dimensional boson space and let
`f_beta` be sixteen arbitrary vectors in the fermion space. Preserving the
standard gauge-field transformation requires

```text
L_alpha^T b_i = sum_beta (gamma_i)[alpha,beta] f_beta.
```

Because each gamma row is a signed coordinate vector, eliminating the `f_beta`
reduces this to homogeneous signed equations of the form `x = +/- y`. A signed
union-find solves the complete linear system without floating-point arithmetic.

Result:

```text
topologies:                         2
dashings per topology:            256
dashings tested:                  512
dashings with nonzero solution:     0
```

## Auxiliary-spinor quotient analysis

The next calculation relaxes item 3 without weakening the full Garden
algebra. Instead of requiring the sixteen physical gauginos to be a literal
fermion subspace, introduce a rank-16 on-shell projection

```text
P_F : R^128 -> R^16
```

and require only

```text
P_F L_alpha^T b_i = (gamma_i)[alpha,beta] e_beta.
```

Distinct full fermions may therefore project to the same physical gaugino. Their
signed differences lie in `ker P_F` and are the auxiliary-spinor directions that
the conventional transformation omitted.

This becomes a discrete constraint-satisfaction problem. Every candidate physical
boson assigns each of its sixteen colored fermion neighbors to a signed physical
gaugino component. A bitset constraint calculation selects nine mutually
compatible bosons and records the complete projection.

### Positive result for E8 x E8

For catalog entry 75 with dashing 0, the compatible bosonic coordinates are

```text
b_i = [48, 39, 22, 64, 106, 79, 25, 125, 83].
```

The independently checked projection has:

```text
full fermions reached:               128
physical gaugino components:          16
full coordinates per component:        8
projection-kernel directions:         112
112 / 16:                               7
```

The kernel dimension is necessarily 112 once a rank-16 quotient of a
128-dimensional space has been chosen. Therefore `112 = 7 x 16` is bookkeeping,
not independent evidence for seven Lorentz-covariant auxiliary spinors. The
calculated result is that the time-reduced SO(9) linkage matrices admit a simultaneous
support-and-sign-compatible quotient within this ansatz.

### D16 comparison

Catalog entry 76 has no support-compatible coordinate quotient. This obstruction
ignores dashing signs, so it excludes every dashing of that topology within the
coordinate-boson quotient ansatz.

### Q lambda consistency test

For this E8 x E8 projection, the search also tried to choose one coordinate
representative from each eight-element fermion fiber and a compatible boson
projection reproducing the on-shell `Q lambda` linkage. No such coordinate
right inverse exists for this projection. This does not exclude a different
one-sided projection or a non-coordinate linear right inverse. The next problem is
therefore a joint search for `P_F`, `J_F`, and `P_B`, followed by nonvalise gauge
closure.

### Non-coordinate right-inverse analysis

The uniform right inverse is the signed average over each eight-element
fiber:

```text
J_F = (1/8) P_F^T,
P_F J_F = I_16.
```

For this choice of `J_F`, the remaining `Q lambda` equations are linear in `P_B`.
The Rust command emits all 265 integer-scaled equations, and
`scripts/verify_sr_uniform_section.py` computes their ranks over the
rationals:

```text
rank(A):                    128
rank([A|all 9 targets]):    137
rank([A|one target]):       129  for every target component
```

The uniform section is excluded by the rank increase.

The search was then enlarged to completely general real `J_F` and `P_B`, while
enforcing the inverse constraints by construction:

```text
P_F J_F = I_16,
P_B J_B = I_9.
```

`scripts/search_sr_joint_section.py` alternates the two remaining linear solves
and finishes with Newton correction in 2,863 affine variables. It found:

```text
maximum one-dimensional linkage residual:    1.11e-15
Frobenius one-dimensional linkage residual:  6.95e-15
fermion inverse residual:             2.22e-16
boson inverse residual:               0
Jacobian rank:                        2304 / 2304 rows
```

This is a machine-precision numerical solution with a full-row-rank Jacobian.

### Banach fixed-point existence proof

`scripts/prove_sr_joint_root.py` closes the numerical gap. It treats every
affine coordinate of the numerical solution as a dyadic rational, selects a square set of
2,304 variables, freezes the remaining 559, and applies the Banach fixed-point
theorem to `T(x) = x - B F(x)`.

The center residual is evaluated with an integer denominator of
`2^2148`. The inverse-defect calculation uses standard IEEE-754 `gamma_n` rounding
bounds, while the final contraction inequalities are rational
comparisons. The deliberately coarse certified bounds are:

```text
||F(x0)||_infinity < 1e-14
||B||_infinity     < 1024
||I - B J(x0)||    < 1e-6
Jacobian Lipschitz constant <= 448
certified radius: 3e-11
contraction bound: 46133 / 3125000000 < 1
```

The closed ball maps strictly inside itself. Therefore a real zero of all 2,304
projected linkage equations exists in that ball, with `P_F J_F = I_16` and
`P_B J_B = I_9` holding by construction in the affine parameterization. This is an
existence proof for the projected one-dimensional linkage system. It does not
yet supply a compact symbolic form for the unrestricted real maps.

## Scope

This is not a no-go theorem for arbitrary off-shell 10D SYM. The negative
result applies to:

1. the minimal 128 by 128 adinkraic Garden module;
2. the standard time-reduced SO(9) on-shell transformations;
3. the standard gauge-field transformation with no extra auxiliary-fermion term;
4. a valise, orthonormal Garden metric.

Dropping the paired-spinor count is insufficient inside the
standard subspace embedding, but the projected one-dimensional system has a
real solution for E8 x E8 and no solution for D16 in the coordinate-boson
ansatz. This is a
consistency result for the proposed linear retraction, not evidence by itself
for admissible auxiliary spinors. It is not yet a 10D off-shell multiplet
because the derivative linkages, Lorentz representations, gauge quotient, and
action remain untested.

The coordinate and uniform right inverses fail. An unrestricted real right
inverse and compatible projection exist.

## SO(9) equivariance test

`scripts/check_sr_so9_equivariance.py` constructs the Spin(9) action on the
sixteen Garden colors and its induced skew action on the full 128 bosons and 128
fermions. It verifies the full covariance equations and so(9) commutators
symbolically. A signed-union-find calculation gives homogeneous covariant-pair
dimension one, spanned by the identity, so the skew field-action lift is unique.

The quadratic Casimirs are scalar:

```text
physical vector:    8
physical spinor:    9
full boson space:  18
full fermion space:18
```

Every equivariant map must intertwine these Casimirs. The eigenvalue mismatch
therefore forces every candidate `J_B`, `P_B`, `J_F`, and `P_F` to vanish. The
numerical solution also fails all 36 rotation-generator intertwining equations.
Thus the projected one-dimensional solution cannot define an
SO(9)-equivariant linear retraction on this valise representation. This
conclusion is independent of the choice of right inverse or projection maps.

Any remaining construction must change the representation problem through
nonvalise engineering, gauge and Bianchi sectors, or a larger module. Returning
another search for unrestricted real maps on the same valise does not change
the Casimir mismatch.

### Local node-raising corollary

`scripts/check_sr_nonvalise_locality.py` extends the Casimir argument over the
time-derivative rings `R[D]` and `R[D,D^-1]`. Because spatial SO(9) commutes with
`D`, an equivariant differential map `T(D) = sum_k T_k D^k` must satisfy the
Casimir equation coefficient by coefficient. The bosonic Casimir difference is
10 and the fermionic difference is 9, so every `T_k` vanishes. Ordinary node
raising, lowering, and inverse time derivatives do not change the
representation content.

Any remaining construction must therefore add new SO(9) representation content
or formulate closure modulo gauge transformations and Bianchi identities. A
rectangular gauge-extended construction is not excluded by this corollary.

### Local spatial derivatives and the zero-momentum fiber

`scripts/check_sr_spatial_gauge_locality.py` evaluates the next possibility over
the local differential-operator ring

```text
R[D,p_1,...,p_9].
```

Evaluation at zero spatial momentum is a ring homomorphism to `R[D]`. For an
SO(9)-equivariant polynomial map `T(p,D)`, the value `T(0,D)` is therefore an
SO(9)-equivariant time-derivative map. The preceding Casimir result forces all
four values `J_B(0,D)`, `P_B(0,D)`, `J_F(0,D)`, and `P_F(0,D)` to vanish.
Multiplicativity of evaluation then gives

```text
(P_B J_B)(0,D) = 0, not I_9,
(P_F J_F)(0,D) = 0, not I_16.
```

Thus finite local spatial derivatives do not restore an ordinary linear retraction on the
same field module. They can make higher-order momentum coefficients transform
in new SO(9) types, but they do not change the representation obtained by
setting the spatial momentum to zero.

The same contradiction persists for the stated class of gauge and Bianchi
corrections. For the nine
spatial potentials, a gauge shift has the form `delta A_i = p_i epsilon`.
It lies in the spatial ideal `m = (p_1,...,p_9)` and vanishes at `p=0`. Any
Bianchi correction of positive spatial degree does likewise. If
`P J = I + K` with every entry of `K` in `m`, evaluation gives `0 = I`.
There is no corresponding gauge correction for the fermion identity.

Taken alone, this is not a blanket exclusion of all gauge constructions. It
does not cover added gauge or Bianchi representations, a larger rectangular
module, algebraic corrections that remain nonzero at zero spatial momentum, or
localization away from `p=0` using inverse spatial operators or boundary
conditions. The following calculation separately tests the ordinary exterior
field-strength complex.

## Spin(9) identity of the valise module

`scripts/check_sr_spin9_decomposition.py` computes the joint Cartan
character of the full field action. The four commuting generators are
`M_01`, `M_23`, `M_45`, and `M_67`. Their individual characteristic
polynomials and the characteristic polynomial of a separating base-32 linear
combination agree with

```text
bosons:  Sym^2_0(9) + Lambda^3(9) = 44 + 84,
fermions: (9 tensor 16) - 16 = 128.
```

The base-32 combination is injective on the prescribed finite weight box, so this
determines the joint weight multiset rather than matching a subset of traces.
The fermionic representation is the gamma-traceless vector-spinor. Thus the
tested `128|128` valise has the Spin(9) state content of
the eleven-dimensional supergravity multiplet. It is not an SO(9) module that
contains the vector `9` and spinor `16` required by the time-reduced
ten-dimensional SYM transformations.

This explains the Casimir failure structurally. The result is not that the
projected one-dimensional equations were inconsistent. Their solution lies in a
different Spin(9) representation.

## Gauge complex and direct-sum enlargement

`scripts/check_sr_gauge_complex.py` constructs the abelian ten-dimensional
exterior complex over `R[D,p_1,...,p_9]`:

```text
Lambda^0 -> Lambda^1 -> Lambda^2 -> Lambda^3
    1          10          45          120
```

It verifies `d^2=0` symbolically. At generic nonzero momentum the first three ranks
are `1, 9, 36`, as expected. Under spatial SO(9), the cochains are

```text
gauge parameter:  1
potential:        1 + 9
field strength:   9 + 36
Bianchi:          36 + 84
```

The tested valise contains the `84`, but it lacks the `1`, `9`, `36`, and the
physical fermion `16` required by the SYM complex.

The chain-homotopy identity also fails on the tested representation. If

```text
P J - I = d H + H d,
```

then at total momentum `(D,p) = 0`, every derivative-built differential
vanishes. Thus `dH+Hd=0` even when `H` has a constant term. The identity
reduces to the ordinary linear retraction already excluded by the Casimir mismatch.
Consequently, switching from potentials to ordinary field-strength variables
does not by itself resolve the representation mismatch.

A conventional N=16 valise has `128m` bosons and `128m` fermions. The
next size retaining the current block is therefore `256|256`. The three direct
sum possibilities built from the valise block and its parity reversal contain
only combinations of `44`, `84`, and `128`; none contains a bosonic `9` and a
fermionic `16`. The smallest direct-sum enlargement therefore fails the
same representation test.

The remaining local cases are:

1. use a nonvalise or gauge-extended module whose zero-momentum
   representation directly contains `9|16`; or
2. introduce an algebraic differential or homotopy that remains nonzero at
   total momentum zero.

Derivative-only gauge complexes and larger direct sums of the same
valise irreducible cannot do either.

## Direct 4D compact-symmetry test

`scripts/check_sr_spin3_spin6_branching.py` also restricts the validated field
action to

```text
Spin(3) x Spin(6) ~= SU(2) x SU(4)_R.
```

Restricted Cartan characters and joint-Casimir nullities give

```text
44 -> (5,1) + (3,6) + (1,20') + (1,1),
84 -> (1,1) + (3,6) + (3,15) + (1,20),
128 -> (4,4) + (4,4bar) + (2,4) + (2,4bar)
       + (2,20) + (2,20bar)
```

after complexification where appropriate. The physical real gaugino
`(2,4)+(2,4bar)` has multiplicity one, with dimension 16. The required bosonic
sectors do not occur:

```text
(3,1) spatial vector eigenspace: 0, required 3,
(1,6) scalar eigenspace:         0, required 6.
```

Compact-group complete reducibility therefore forbids an ordinary
`Spin(3) x Spin(6)`-equivariant linear retraction from the tested valise onto
the four-dimensional physical representation. The missing representations are
bosonic. This excludes the tested `128|128` valise under direct
four-dimensional compact covariance as well as under Spin(9) covariance.

The conclusion still does not exclude a changed nonvalise gauge or auxiliary
complex. The next construction must add maps that remain nonzero at zero momentum or
change the representation at zero momentum so that `(3,1)+(1,6)|16` occurs in the
physical cohomology.

## Reproduction

```sh
cargo run --release -- sr-investigation
cargo test
python3 scripts/verify_sr_uniform_section.py
python3 scripts/search_sr_joint_section.py --starts 4 --iterations 100
python3 scripts/prove_sr_joint_root.py
python3 scripts/check_sr_so9_equivariance.py
python3 scripts/check_sr_nonvalise_locality.py
python3 scripts/check_sr_spatial_gauge_locality.py
python3 scripts/check_sr_spin9_decomposition.py
python3 scripts/check_sr_gauge_complex.py
python3 scripts/check_sr_spin3_spin6_branching.py
```

The command writes a JSON file containing the arithmetic, linkage residuals,
topology coverage, dashing coverage, and the stated limits of the ansatz.
