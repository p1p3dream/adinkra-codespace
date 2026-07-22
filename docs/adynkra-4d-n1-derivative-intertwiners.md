# Four-dimensional N=1 derivative intertwiners

## Result

The Rust implementation constructs the exact rational Clebsch-Gordan maps
needed when a left- or right-handed supercovariant derivative acts on every
Lorentz representation present in the six genomes of Gates and Hu,
arXiv:2407.09334v1.

For each `SL(2)` factor it realizes

```text
[n] tensor [1] = [n+1] + [n-1]
```

as explicit embeddings and projections in the binary-form basis. The
implementation verifies:

- projection followed by embedding is the identity on every summand;
- the two channels have zero cross-composition;
- the channel projectors sum to the identity on the tensor product;
- every embedding and projection intertwines the `E`, `F`, and `H`
  generators exactly.

The validation covers nine distinct Lorentz representations and all 18 left-
and right-handed fundamental products occurring in the published genomes.

## Repeated representations

Exactly three published genomes contain the same Lorentz representation twice
at the same total level:

| Genome | Level | Representation | Bidegrees |
|---|---:|---|---|
| one-form gauge | 2 | `[0,0]` | `(0,2)`, `(2,0)` |
| matter gravitino | 2 | `[1,0]` | `(0,2)`, `(2,0)` |
| supergravity | 2 | `[1,1]` | `(0,2)`, `(2,0)` |

These are not an unresolved multiplicity. Their left and right Grassmann
bidegrees provide a canonical two-dimensional multiplicity basis. The
left-handed derivative selects the `(2,0)` channel and the right-handed
derivative selects the `(0,2)` channel. Both selectors have rank one, and the
combined selector has rank two in all three sectors.

## Artifacts

- `src/adynkra_derivative_intertwiners.rs`: exact rational maps and checks
- `results/adynkra_4d_n1_derivative_intertwiner_validation.json`: validation
  report

## Reproduction

```bash
cargo run --release -- adynkra-derivative-intertwiner-verify \
  > results/adynkra_4d_n1_derivative_intertwiner_validation.json
cargo test adynkra_derivative_intertwiners
```

## Boundary

This supplies the fundamental Lorentz Clebsch-Gordan maps and distinguishes
every repeated representation in the six published genomes. It does not yet
assemble the source component normalizations in the prepotential gauge map or
compute gauge cohomology.

## Reference

S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D, N=1
Supergravity Superfield Prepotential," [arXiv:2407.09334v1](https://arxiv.org/abs/2407.09334).
