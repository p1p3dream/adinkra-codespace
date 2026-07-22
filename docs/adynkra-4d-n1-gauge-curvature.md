# Four-dimensional N=1 prepotential gauge map and curvature

## Gauge map

The Rust implementation encodes the linearized supergravity prepotential gauge
transformation in Eq. (2.21) of Gates and Hu, arXiv:2407.09334v1:

```text
delta H_(alpha dot-alpha)
  = D_alpha Lbar_dot-alpha - Dbar_dot-alpha L_alpha
```

The map acts on the complete 16-element Grassmann basis. The spinor gauge
parameter and its conjugate are treated independently over the complexified
component space, giving a 64-component domain and 64-component codomain.

The sparse differential operator contains 192 terms:

- 64 terms with no spacetime derivative;
- 128 terms with one spacetime derivative;
- no terms of higher derivative order.

All 192 terms obey the expected Grassmann-bidegree selection rule. Its exact
rank is 39 at zero momentum and 48 at each of four nonzero rational momentum
probes. These ranks are diagnostics of the displayed finite matrices, not a
claim about polynomial-module cohomology.

## Chiral super-Weyl curvature

The implementation also encodes Eq. (5.2.5) of *Superspace*:

```text
W_(alpha beta gamma)
  = -(i/3!) Dbar^2 D_(alpha partial_(beta dot-beta)
      H_(gamma))^dot-beta
```

The spinor-metric convention is `epsilon^(01) = 1`, with
`Dbar^2 = Dbar_1 Dbar_0` as in Eq. (3.4.10) of *Superspace*.

The calculation uses all 64 prepotential basis inputs and the four components
of the symmetric rank-three spinor. The operator is nonzero on 152 of the 256
input-output pairs.

Two complete operator identities pass in exact Gaussian-rational arithmetic:

- chirality: 512 relations, zero residuals;
- gauge invariance: 256 relations on the complete gauge-parameter basis, zero
  residuals.

A mutation test removes the spinor derivative from the gauge map and produces
nonzero curvature residuals.

Thus the composition

```text
gauge parameter -> prepotential -> chiral super-Weyl curvature
```

vanishes exactly in the implemented conventions.

## Old-minimal scalar curvature

The old-minimal chiral compensator pair and scalar-curvature pair from Eqs.
(7.4.2b) and (7.5.19) of *Superspace* are also implemented:

```text
delta chi_bar = (1/3) D^2 Dbar^dot-alpha Lbar_dot-alpha
R = Dbar^2 (chi_bar - (i/3) partial_a H^a)
Rbar = D^2 (chi + (i/3) partial_a H^a)
```

Here `phi = 1 + chi`, so the factor `1/3` follows by linearizing the source
transformation of `phi^3`. The validation checks:

- chirality of `R`: 160 relations, zero residuals;
- antichirality of `Rbar`: 160 relations, zero residuals;
- gauge invariance of both curvatures on all 64 gauge-parameter basis inputs:
  128 relations, zero residuals;
- removal of the two compensator contributions: 48 nonzero residuals.

The last check establishes that both compensator contributions are required.

## Artifacts

- `src/prepotential_gauge.rs`: gauge differential and rank diagnostics
- `src/prepotential_curvature.rs`: chiral curvature and operator identities
- `src/minimal_supergravity_curvatures.rs`: old-minimal compensator and scalar
  curvature
- `results/adynkra_4d_n1_prepotential_gauge_validation.json`
- `results/adynkra_4d_n1_prepotential_curvature_validation.json`
- `results/adynkra_4d_n1_minimal_scalar_curvature_validation.json`

## Reproduction

```bash
cargo run --release -- adynkra-prepotential-gauge-verify \
  > results/adynkra_4d_n1_prepotential_gauge_validation.json
cargo run --release -- adynkra-prepotential-curvature-verify \
  > results/adynkra_4d_n1_prepotential_curvature_validation.json
cargo run --release -- adynkra-minimal-curvature-verify \
  > results/adynkra_4d_n1_minimal_scalar_curvature_validation.json
cargo test prepotential
cargo test minimal_supergravity_curvatures
```

## Boundary

This completes the gauge map, the conformal super-Weyl curvature, and the
old-minimal scalar-curvature pair with its chiral compensator pair. It does not yet
implement the vector curvature `G_a`, the Bianchi identities,
polynomial-module cohomology, or an Euler-Lagrange equation.

## References

1. S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D, N=1
   Supergravity Superfield Prepotential," [arXiv:2407.09334v1](https://arxiv.org/abs/2407.09334).
2. S. J. Gates Jr., M. T. Grisaru, M. Rocek, and W. Siegel, *Superspace, or One
   Thousand and One Lessons in Supersymmetry*,
   [arXiv:hep-th/0108200](https://arxiv.org/abs/hep-th/0108200), Eq. (5.2.5).
