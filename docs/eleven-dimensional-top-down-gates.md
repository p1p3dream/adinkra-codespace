# Eleven-dimensional top-down gates

## Scope

This layer attacks the 11D problem from the target theory downward. It does not
enumerate N=32 worldline graphs. It builds exact sparse gauge, curvature,
Bianchi, and equation complexes, then asks which superfield constructions can
map into them.

The current result is a set of bounded gates, not an irreducible 11D
supermultiplet and not a supersymmetric extension of Einstein's equation.

## Gate 1: free component complex

`eleven_dimensional_free_complex` constructs exact sparse matrices over
Gaussian rationals at null momentum `p=(1,1,0,...,0)` in mostly-plus signature.

| Sector | Potential dimension | Gauge rank | Equation rank | Physical quotient |
| --- | ---: | ---: | ---: | ---: |
| graviton `h_ab` | 66 | 11 | 11 | 44 |
| three-form `A_abc` | 165 | 45 | 36 | 84 |
| gravitino `psi_a` | 352 | 32 | 192 | 128 |

Every implemented gauge-to-curvature, curvature-to-Bianchi,
gauge-to-Euler-Lagrange, and Euler-Lagrange-to-Noether composition vanishes
exactly at three checked momenta. The three-form sector includes its full
scalar-to-one-form-to-two-form reducibility chain and both `d^2=0` gates.

This certifies the complexified free target census `44 + 84 | 128`. The
gravitino calculation uses an exact `Q(i)` Clifford basis. A compatible 11D
Majorana real form and the supersymmetry maps between the three sectors have not
yet been constructed.

## Gate 2: hook continuation

The committed zero-momentum level-16 incidence differential has shape `7x12`,
rank 7, nullity 5, and left nullity zero. Consequently every relaxed rational
next-Bianchi row `B` satisfying

```text
B d16 = 0
```

is forced to vanish, and the bounded zero-momentum group
`H17=ker(d17)/im(d16)` is zero.

For the hook tensor `(11000)`, spinor tensoring produces four multiplicity-one
targets:

| Target | Dimension |
| --- | ---: |
| `(01001)` | 1,408 |
| `(10001)` | 320 |
| `(11001)` | 10,240 |
| `(20001)` | 1,760 |

Their total dimension is `13,728 = 32 x 429`. A complete level-18 incidence
requires 16 distinct source irreps, 42 source copies, and 77 embedded
source-target copies. The generated 161-item worklist records those missing
objects. This gate does not prove momentum-dependent or gauge-quotiented
superspace cohomology.

## Gate 3: prepotential gauge-curvature worklist

The source-side audit expands the six inequivalent gauge-parameter channels
into:

- 12 leading operator maps
- 44 first-momentum jobs
- 72 `D^17` jobs
- 336 `p D^15` jobs

The parameter domains are inequivalent, so coefficients from different
channels cannot cancel one another. Gauge invariance must hold separately for
each active channel.

The physical condition

```text
F A G_p = 0
```

is not executable yet. The minimal missing scientific object is a
convention-fixed target superfield complex

```text
P --K--> H_hat(10001) --F--> C
```

with `F K=0`. The minimal missing code interface is a target-resolved exact
`11x32` composition stream. Existing visitors project only one highest-weight
coefficient and therefore cannot apply a general target curvature.

## Aggregate status

The aggregate scan retains the physical B5 seed labels

```text
00000, 20000, 00100, 00010, 00001, 10001, 11000
```

and records their exact spinor incidences. The following gates are green:

1. the complexified free component target and `44+84|128` census
2. the bounded zero-momentum hook result
3. the complete source-side six-channel worklist
4. the bounded physical-seed inventory

The following claims remain false in the artifact:

1. exact superspace differentials for all seed transitions
2. the superspace gauge quotient and Bianchi kernel
3. pure-spinor or spinorial-cohomology comparison
4. isolation of the physical 11D supergravity multiplet
5. a source-to-target superfield map and gauge-invariant curvature operator
6. supersymmetry closure of a linearized 11D equation
7. agreement with the final component equations

## Next executable steps

1. Implement the target-resolved `11x32` sparse composition stream.
2. Fix conventions for `K` and a first candidate curvature `F` into a selected
   target irrep.
3. Run all six channel identities separately and reject any failed channel.
4. Construct the missing embedded level-18 kernels and rerun the hook gate with
   momentum and gauge quotient retained.
5. Add the 11D Majorana real form and exact supersymmetry maps to the free
   `44+84|128` complex.
6. Only after those gates pass, attempt a linearized superfield equation and its
   component reduction.

## Reproduction

```bash
cargo test --release eleven_dimensional_free_complex
cargo test --release eleven_dimensional_hook_bianchi
cargo test --release eleven_dimensional_prepotential_gate
cargo test --release eleven_dimensional_top_down

cargo run --release -- adynkra-11d-free-complex-build
cargo run --release -- adynkra-11d-hook-bianchi-build
cargo run --release -- adynkra-11d-prepotential-gate-build
cargo run --release -- adynkra-11d-top-down-build
```

Artifacts:

- `data/eleven_dimensional_free_complex.json`
- `results/adynkra_11d_free_complex_validation.json`
- `data/eleven_dimensional_hook_bianchi.json`
- `results/adynkra_11d_hook_bianchi_validation.json`
- `results/adynkra_11d_prepotential_gate.json`
- `results/adynkra_11d_top_down.json`
