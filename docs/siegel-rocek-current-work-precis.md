# Minimal N=16 Adinkra completion: current calculations

**Status through 2026-07-14**

## Question and scope

The conventional auxiliary-field count for the 4D N=4 vector multiplet uses

```text
16 + 32 n = 128 m,
```

where auxiliary spinors are counted in pairs. Dropping that pairing premise
gives

```text
16 + 16 q = 128 m,
```

whose smallest case is `m=1`, `q=7`, suggesting a first test at `128|128`.
This arithmetic only selects a candidate size. It does not prove that seven
independent auxiliary spinors are Lorentz-covariant, local, or dynamically
admissible.

The investigation tests whether a minimal N=16 Garden representation contains
the physical vector and gaugino as a covariant subrepresentation or quotient,
and whether the resulting one-dimensional transformations extend to a local,
higher-dimensional representation with closure modulo gauge transformations.
No off-shell 4D N=4 or 10D N=1 supermultiplet is claimed.

## Projected one-dimensional linkage equations

The target linkage matrices are the nine real symmetric SO(9) gamma matrices
relating the nine spatial vector components to sixteen gauginos after reduction
to one time dimension. Both self-dual length-16
chromotopologies and all 256 dashings of each were tested.

- No set of nine bosonic coordinates has the required support.
- Arbitrary real field mixing with the standard no-leakage embedding has only
  the zero solution for all 512 dashings.
- For E8xE8, catalog entry 75 and dashing 0, nine coordinate bosons and a
  rank-16 fermion projection satisfy one set of projected linkage equations.
  D16 does not satisfy the corresponding support condition.
- A coordinate right inverse for the fermion projection fails. The uniform
  fiber-average right inverse also fails because its coefficient matrix has
  rank 128 and its augmented matrix has rank 129 over the rationals.
- With unrestricted real matrices `J_F` and `P_B`, a numerical solution
  satisfies 2,304 linkage equations while the injection and projection maps
  obey `P_B J_B=I_9` and `P_F J_F=I_16`.
- A separate Banach fixed-point calculation establishes a real solution
  within radius `3e-11` of the numerical solution. Its final contraction bound is
  `46133/3125000000 < 1`.

Thus the projected one-dimensional system has a real solution. The
`112=7x16` dimension of the projection kernel does not establish the presence
of seven Lorentz-covariant auxiliary spinors.

## SO(9) representation mismatch

The induced Spin(9) actions on the full `128|128` valise were constructed and
their covariance, skewness, and Lie-algebra closure were checked symbolically. The
quadratic Casimirs are

```text
physical vector:     8
physical spinor:     9
full boson space:   18
full fermion space: 18
```

Any equivariant injection or projection must intertwine the Casimir. The scalar
eigenvalue mismatch therefore forces all four maps in the proposed linear
retraction to vanish. The solution of the projected linkage equations is
consistent in one dimension, but it is not an SO(9)-equivariant linear
retraction of the tested valise representation.

## Locality tests

The Casimir obstruction persists for time-derivative maps over both `R[D]` and
`R[D,D^-1]`. Since spatial rotations commute with `D`, every coefficient is an
intertwiner and is forced to zero. Ordinary node raising, lowering, and inverse
time derivatives do not change this representation-theoretic result.

Finite local spatial derivatives were then allowed over
`R[D,p_1,...,p_9]`. Evaluating at zero spatial momentum reduces every candidate
to the already excluded time-derivative map. Consequently

```text
(P_B J_B)(0,D) = 0, not I_9,
(P_F J_F)(0,D) = 0, not I_16.
```

For spatial potentials, gauge shifts `delta A_i=p_i epsilon` vanish on this
fiber, as do Bianchi corrections of positive spatial degree. Such corrections
cannot change the contradiction. This result applies to local polynomial maps
on the tested representation. It does not exclude additional representations,
maps that remain nonzero at zero momentum, or nonlocal constructions that
remove the zero-momentum sector.

## Spin(9) content of the tested valise

A joint Cartan-character calculation identifies the full Spin(9) representation as

```text
bosons:   Sym^2_0(9) + Lambda^3(9) = 44 + 84
fermions: gamma-traceless vector-spinor = 128
```

Individual characteristic polynomials and an injective base-32 separating
combination agree with those representations. This is the
`44+84|128` Spin(9) content of the eleven-dimensional supergravity multiplet.
It does not contain the vector `9` and spinor `16` required by the time-reduced
10D SYM transformations. The Casimir mismatch is independent of the particular
right inverse used in the projected one-dimensional calculation.

## Exterior gauge complex and direct-sum enlargement

The ordinary abelian exterior complex was constructed over the local operator
ring:

```text
Lambda^0 -> Lambda^1 -> Lambda^2 -> Lambda^3
    1          10          45          120
```

The calculation verifies `d^2=0` symbolically and generic ranks `1,9,36`. Under SO(9), its
cochains contain

```text
gauge parameter:  1
potential:        1 + 9
field strength:   9 + 36
Bianchi:          36 + 84
```

The tested valise supplies the `84` but not the required bosonic `1`, `9`, and
`36`, or the physical fermionic `16`.

The chain-homotopy relaxation

```text
P J - I = d H + H d
```

also fails on the tested representation. At total momentum zero, every
derivative-generated differential vanishes, so the homotopy identity reduces
to the ordinary linear retraction already excluded by the Casimir calculation.
field-strength variables alone therefore do not solve the problem.

An ordinary N=16 valise has dimensions in multiples of `128|128`. The smallest
direct-sum enlargement containing the tested representation is `256|256`.
Direct sums of this representation
and its parity reversal contain only combinations of `44`, `84`, and `128`.
None contains both a bosonic `9` and a fermionic `16`, so all three
direct-sum candidates fail before a linkage search.

## Direct 4D branching result

The validated field action was also restricted to

```text
Spin(3) x Spin(6) ~= SU(2) x SU(4)_R
```

This tests the 4D physical target

```text
9 -> (3,1) + (1,6)
```

together with the corresponding real 16-dimensional gaugino representation.
The subgroup generators, their mutual commutation, and the restricted Cartan
characters were checked symbolically. The bosonic branchings are

```text
44 -> (5,1) + (3,6) + (1,20') + (1,1)
84 -> (1,1) + (3,6) + (3,15) + (1,20)
```

After complexification, the fermionic branching is

```text
128 -> (4,4) + (4,4bar)
     + (2,4) + (2,4bar)
     + (2,20) + (2,20bar).
```

The joint Spin(3) and Spin(6) Casimir eigenspaces give:

```text
required bosonic (3,1): eigenspace dimension 0, required 3
required bosonic (1,6): eigenspace dimension 0, required 6
physical real gaugino:  eigenspace dimension 16, required 16
```

Thus the physical gaugino occurs once, but neither the three spatial vector
components nor the six scalars occur in the bosonic `128`. Compact-group
complete reducibility therefore forbids an ordinary
`Spin(3) x Spin(6)`-equivariant linear retraction onto the full `9|16` physical
representation. It does not
exclude a nonvalise gauge or auxiliary complex
whose algebraic zero-momentum maps change the representation.

## Remaining cases

The tests exclude the tested valise under Spin(9) covariance and under an
ordinary direct four-dimensional linear retraction. A remaining local
construction must:

1. use a nonvalise or gauge-extended zero-momentum representation
   that contains the physical 4D fields directly;
2. introduce algebraic auxiliary, gauge, or homotopy maps that remain nonzero at
   total momentum zero;
3. use a new rectangular complex rather than a direct sum of the existing
   valise irreducible representation.

Next, compute the minimal Lorentz and `SU(4)_R` representation inventory at
zero momentum. Solve closure for that inventory before adding derivatives.
Inverse spatial derivatives and boundary conditions that remove zero modes
remain separate cases, but they do not address the finite local auxiliary
field problem in its usual form.

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

Each calculation writes its inputs, intermediate invariants, and results to a
JSON file under `results/`.
