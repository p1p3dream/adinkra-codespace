# Four-dimensional N=1 Adynkra genome reproduction

## Result

The Rust implementation reproduces the six Adynkra genomes in Eqs. (3.6)-
(3.11) of Gates and Hu, arXiv:2407.09334v1:

| Equation | Supermultiplet | Terms | Total representation dimension |
|---|---|---:|---:|
| 3.6 | chiral | 3 | 4 |
| 3.7 | 2-form gauge field | 4 | 8 |
| 3.8 | 1-form variant gauge field | 3 | 8 |
| 3.9 | 1-form gauge field | 9 | 16 |
| 3.10 | matter gravitino | 12 | 32 |
| 3.11 | supergravity | 16 | 64 |

All 47 representation terms, their left and right level degrees, their
multiplicities, and their factorial coefficients agree with a separate literal
transcription of the six published equations.

## Method

Complexified four-dimensional Lorentz representations are recorded as
`SL(2)_L x SL(2)_R` Dynkin labels `[a,b]`. The two Grassmann level parameters
transform as `[1,0]` and `[0,1]`. Their exterior powers are

```text
wedge^0 [1,0] = [0,0]   wedge^1 [1,0] = [1,0]   wedge^2 [1,0] = [0,0]
wedge^0 [0,1] = [0,0]   wedge^1 [0,1] = [0,1]   wedge^2 [0,1] = [0,0]
```

Each seed representation is tensored with these exterior powers. The product
is decomposed independently in both `SL(2)` factors using the usual
Clebsch-Gordan range. A bidegree `(p,q)` carries the coefficient `1/(p!q!)`
from the genome exponential.

The six seeds and chirality restrictions are:

| Equation | Seed | Levels |
|---|---|---|
| 3.6 | `[0,0]` | left only |
| 3.7 | `[1,0]` | left only |
| 3.8 | `[0,1]` | left only |
| 3.9 | `[0,0]` | left and right |
| 3.10 | `[1,0]` | left and right |
| 3.11 | `[1,1]` | left and right |

## Artifacts

- `data/adynkra_4d_n1_genomes.json`: complete generated genomes
- `results/adynkra_4d_n1_genome_validation.json`: equation-by-equation check
- `src/adynkra_genome.rs`: representation algebra, source fixture, and tests

The source PDF SHA-256 is
`64f0ae888933a8a6ff7b768c73d21656baa557ef7b089aab9f6252129ee58f81`.

## Reproduction

```bash
cargo run --release -- adynkra-genome-build
cargo run --release -- adynkra-genome-verify
cargo test adynkra_genome
```

## Boundary

This calculation reproduces representation content and multiplicities. It does
not supply Clebsch-Gordan coefficients, supercovariant derivative maps,
component transformation laws, or a field equation. Those are the next parts
of the differential-operator program.

## Reference

S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D, N=1
Supergravity Superfield Prepotential," [arXiv:2407.09334v1](https://arxiv.org/abs/2407.09334).
