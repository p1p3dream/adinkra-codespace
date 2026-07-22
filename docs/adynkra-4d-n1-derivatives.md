# Four-dimensional N=1 supercovariant derivative baseline

## Result

The Rust implementation reproduces the supercovariant derivatives in Eq.
(2.22) of Gates and Hu, arXiv:2407.09334v1:

```text
D_alpha     = partial_alpha     + (i/2) theta_bar^dot_alpha partial_alpha,dot_alpha
D_dot_alpha = partial_dot_alpha + (i/2) theta^alpha         partial_alpha,dot_alpha
```

The operators act on the complete 16-dimensional Grassmann monomial basis of
two left and two right spinor coordinates. Exact Gaussian-rational arithmetic
checks every unordered pair of the four derivative operators on every basis
monomial:

```text
10 derivative pairs x 16 monomials = 160 relations
```

All 160 relations have zero residual. The same-chirality anticommutators
vanish, and each mixed anticommutator equals the corresponding spacetime
derivative with coefficient `i`.

## Artifacts

- `src/supercovariant_derivative.rs`: exact Grassmann and derivative algebra
- `results/adynkra_4d_n1_derivative_validation.json`: validation report

## Reproduction

```bash
cargo run --release -- adynkra-derivative-verify \
  > results/adynkra_4d_n1_derivative_validation.json
cargo test supercovariant_derivative
```

## Boundary

This establishes the source-convention superspace derivative algebra. It does
not yet construct the irreducible Clebsch-Gordan intertwiners between repeated
Lorentz representations, the prepotential gauge complex, or its cohomology.

## Reference

S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D, N=1
Supergravity Superfield Prepotential," [arXiv:2407.09334v1](https://arxiv.org/abs/2407.09334).
