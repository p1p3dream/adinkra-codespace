# Four-dimensional N=1 old-minimal Adynkrafield operator

## Result

The known linearized old-minimal supergravity equation is now expressed in
Adynkrafield coordinates. The potential is

```text
Phi_AF = (H_AF, chi_AF, chi_bar_AF)
dimension = 64 + 4 + 4 = 72
```

`H_AF` uses the 16 Lorentz blocks of the published supergravity genome in
Eq. (3.11) of Gates and Hu. `chi_AF` uses the four components of the chiral
genome in Eq. (3.6), and `chi_bar_AF` uses its conjugate. The coordinate maps
include the rational Clebsch-Gordan embeddings and the published
`1/(p!q!)` level coefficients.

For each momentum fiber, let `U_AF` embed Adynkrafield coordinates into the
superspace component basis and let `P_AF` be its left inverse. The translated
operator is

```text
E_AF = P_AF (G, R, Rbar) U_AF
E_AF Phi_AF = 0
```

The implementation verifies the reconstruction identity

```text
U_AF E_AF = (G, R, Rbar) U_AF
```

on seven exact rational momentum fibers.

## Coordinate certificate

- The supergravity genome map is a `64 x 64` exact change of basis of rank 64.
- The chiral and conjugate compensator maps each have rank 4.
- The combined map has rank 72 at every tested momentum.
- Every coordinate projection followed by embedding is the identity.
- All factorial coefficients in the 16 supergravity blocks agree with the
  published genome.

The calculation checks 48,384 operator-reconstruction entries with zero
residuals. The operator ranks are

| Momentum class | Fibers | Rank |
|---|---:|---:|
| zero | 1 | 6 |
| null | 3 | 20 |
| non-null | 3 | 24 |

These are the same ranks obtained before the coordinate change. All 448 gauge
columns are reconstructed in Adynkrafield coordinates and annihilated by the
translated operator with zero residuals.

## Interpretation

This completes the four-dimensional benchmark through the following chain:

```text
published genome
  -> exact Lorentz intertwiners
  -> Adynkrafield coordinates
  -> gauge and curvature operator
  -> quadratic action
  -> Euler-Lagrange equation
  -> physical null spectrum
```

The result is a coordinate translation and verification of the known
four-dimensional equation. It is not a new gravitational equation.

## Artifacts

- `src/adynkrafield_operator.rs`
- `results/adynkra_4d_n1_adynkrafield_operator_validation.json`
- `src/minimal_supergravity_curvatures.rs`
- `src/minimal_supergravity_action.rs`

## Reproduction

```bash
cargo run --release -- adynkrafield-operator-verify \
  > results/adynkra_4d_n1_adynkrafield_operator_validation.json
cargo test adynkrafield_operator
```

## Boundary

The equivalence is exact on one zero, three null, and three non-null rational
momentum fibers. A polynomial-module proof over the full momentum ring remains
open. Applying the construction in eleven dimensions also requires a specified
prepotential, gauge transformation, and physical constraint system.

## References

1. S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D,
   N=1 Supergravity Superfield Prepotential,"
   [arXiv:2407.09334v1](https://arxiv.org/abs/2407.09334), Eqs. (3.6), (3.11),
   (4.4), (4.6), and (4.13)-(4.14).
2. S. J. Gates Jr., M. T. Grisaru, M. Rocek, and W. Siegel, *Superspace, or
   One Thousand and One Lessons in Supersymmetry*,
   [arXiv:hep-th/0108200](https://arxiv.org/abs/hep-th/0108200), Eqs. (5.5.45),
   (5.5.48), and (7.5.19).
