# Scalar-tensor rigid tangent result

## Decision

The scalar-tensor multiplet of
[arXiv:2412.16527](https://arxiv.org/abs/2412.16527) has a regular rigid tangent,
but that tangent is not a credible independent S8 holdout. After the required
central-U(1) quotient, its field content and linear Q-coupling pattern are the
existing chiral-tensor, or `CT`, multiplet.

The nonlinear conformal multiplet remains distinct. Its new information is in
the composite connection, nonlinear gauge realization, conformal weights, and
the singular zero-norm locus. A regular linear worldline shadow forgets those
features.

## Exact regular background

Freeze the Weyl multiplet to flat space, take constant Q parameters, set the
S-supersymmetry parameter to zero, and expand about

```text
xi_i = (1, 0)
X = 0
B_mu_nu = 0
psi = theta = 0
```

The source composites contain `(xi^i xi_i)^-1`, so the origin is singular. A
nonzero background value of `X` also fails to preserve all eight Q charges. The
point above has unit scalar norm, preserves all rigid Q transformations, and
sets all three composite fields to zero at background order.

Writing

```text
xi_1 = 1 + a + i b
xi_2 = c + i d
```

gives

```text
xi^i xi_i = 1 + 2a + O(2)
(xi^i xi_i)^-1 = 1 - 2a + O(2)
```

The executable preflight verifies this inverse exactly through first order.

## Linearized composites

On the flat background, Eqs. (5.4), (5.11), and (5.15) give

```text
Omega_1 =  i slash(partial) psi_R
Omega_2 = -i slash(partial) theta_R

Y_11 = -2i box(c-i d)
Y_12 =  2i box(a)
Y_22 =  2i box(c+i d)
```

Define the real dual-strength convention used by the artifact as

```text
h_a = -(i/3!) epsilon_abcd H^bcd.
```

Then

```text
W_a = h_a - 2 partial_a b
D_a xi_1 = partial_a a + (i/2) h_a
D_a xi_2 = partial_a(c+i d).
```

The phase derivative cancels exactly from `D xi_1`. Under
`delta_z xi_i = -i z xi_i/2`, the tangent phase shifts by `delta b=-z/2`, so
the composite connection transforms as `delta W_a=partial_a z`.

## Gauge quotient and 8+8 count

The phase slice is

```text
Im(v^i q_i) = 0.
```

Raw Q transformations are returned to the slice with

```text
alpha_Q = (2/r) Im[v^i(-bar(epsilon_i) theta_R
                     + epsilon_ij bar(epsilon^j) psi_L)].
```

The count is exact:

```text
4 complex-doublet real components
+ 2 complex-X real components
+ 6 two-form components
- 3 reducible tensor-gauge directions
- 1 central-U(1) direction
= 8 bosons.
```

The two Majorana spinors contribute eight fermionic components.

## Why the tangent is CT

In the unitary gauge `b=0`, the source fields have the following roles:

| Scalar-tensor tangent | CT role |
|---|---|
| `Re xi_2`, `Im xi_2` | chiral scalars `A,B` |
| `Re X`, `Im X` | chiral auxiliaries `F,G` |
| `Re xi_1` | tensor scalar `phi` |
| `B_mu_nu` | tensor gauge field |
| realifications of `psi_L`, `theta_R` | crossed CT fermions |

This is more than component counting. The two supersymmetries cross-couple the
chiral and tensor fermions in the same pattern as Eqs. (44)-(47) of
[arXiv:1405.0048](https://arxiv.org/abs/1405.0048). The composite `W_mu` inserts
the dual three-form into the radial scalar derivative, while `xi_2` retains an
ordinary complex gradient and `X` transforms as an auxiliary pair.

An equivalent gauge-invariant statement uses the tangent Hopf map

```text
ell^A = v^dagger sigma^A q + q^dagger sigma^A v.
```

It maps the complex doublet onto three real scalar directions and has kernel
exactly equal to the central-U(1) phase direction.

The exact source-to-repository Majorana/Clifford intertwiner is not yet solved.
Therefore the bounded result is: reject this target as independent, while
retaining exact 4D equivalence as an optional convention-validation exercise.
Published CT signs were not used to fill any missing source signs.

## Source issue and closure boundary

The v3 TeX for Eq. (5.19) has a mismatched gravitino SU(2) index. Eq. (5.17)
contains the consistent contraction. This term vanishes in the rigid tangent,
so it does not alter the preflight.

The paper also states that composite `Y_ij` may enter `delta Omega_i` and may be
needed for closure. A full source-faithful component fixture still requires:

1. one exact source-to-repository Majorana, gamma, chirality, and epsilon bridge;
2. Q variation of every linearized composite, including `Y_ij`;
3. 4D closure modulo tensor and central-U(1) gauge transformations;
4. the phase compensator before the central quotient;
5. temporal tensor gauge only after 4D closure;
6. a blinded exact CT intertwiner solve.

Until those gates pass, no scalar-tensor S8 matrices should be added to the
atlas.

## Executable artifact

```bash
cargo run -- scalar-tensor-tangent-build
cargo run -- scalar-tensor-tangent-verify
```

The artifact at `results/scalar_tensor_tangent.json` certifies background
regularity, first-order denominator inversion, all three composite expansions,
central-gauge covariance, phase cancellation, and the 8+8 count. It explicitly
marks full component closure and the exact CT intertwiner as unfinished, so a
passing preflight cannot be mistaken for a completed 4D fixture.

The PDF and source archive used for the derivation have SHA-256 hashes recorded
in the artifact.
