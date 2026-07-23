# Eleven-dimensional first-momentum certification: build order

## Objective

Construct the 44 exact first-momentum intertwiners in the direct
spinor-prepotential calculation:

\[
p_a D^{14}\Psi
\longrightarrow
(10000)\otimes(10001).
\]

The target product contains

```text
(00001) + (01001) + (10001) + (20001).
```

Their level-14 multiplicities are 5, 18, 8, and 13. The total map space
therefore has dimension 44.

This work supplies the first spacetime-momentum correction to the completed
zero-momentum `7 x 12` exterior symbol. It does not select the gauge
coefficients or determine a field equation.

Status on 2026-07-23: phases 1 through 4 are complete. The joint compatibility
matrix and gauge intersection in phases 5 and 6 remain.

## Fixed source work list

The 44 maps use 23 source-target pairs and 28 distinct embedded source copies.
Each target occurs with multiplicity one in its source representation tensored
with the 32-dimensional spinor.

| Level-14 source | Embedded copies |
|---|---:|
| `(00000)` | 1 |
| `(00010)` | 2 |
| `(00100)` | 2 |
| `(01002)` | 5 |
| `(01010)` | 1 |
| `(01100)` | 2 |
| `(02000)` | 2 |
| `(10002)` | 2 |
| `(10010)` | 1 |
| `(10100)` | 1 |
| `(20002)` | 2 |
| `(20010)` | 4 |
| `(20100)` | 3 |
| **Total** | **28** |

The exact source-target pairing is recorded in
`results/adynkra_11d_first_momentum_source_precheck.json`.

## Phase 1: source embeddings

1. Build the 13 level-14 highest-weight raising systems.
2. Recover 28 primitive integer kernel candidates.
3. Store each candidate in the narrowest signed little-endian format that
   contains every primitive coefficient.
4. Rebuild every raising equation in Rust.
5. Accept a fixture only if every exact integer residual is zero.
6. Verify the expected first-lowering string for every simple root.

The floating-point sparse eigensolver proposes integer vectors. It is not the
certificate. Rust integer residual checks are the acceptance gate.

### Phase 1 result

Phase 1 is complete:

- 13 exact highest-weight systems;
- 28 primitive integer source vectors;
- 11,167,170 exact raising rows checked;
- zero nonzero raising residuals;
- all expected first-lowering strings verified.

Twenty-four files use signed 16-bit little-endian coefficients. The four
`(20010)` files use signed 32-bit little-endian coefficients because the fourth
primitive vector contains a coefficient of magnitude 485,880. The Rust verifier
decodes both formats into signed 64-bit integers before applying the raising
operators. Storage width does not enter the acceptance criterion.

The exact certificate is
`results/adynkra_11d_first_momentum_kernels.json`. The independent numerical
proposal reports are under `results/level14_first_momentum_crosschecks/`.

## Phase 2: 23 abstract couplings

For every source-target pair:

1. generate the required source weights with canonical PBW lowering words;
2. construct the tensor-product highest-weight equations;
3. solve the exact rational Gram system;
4. verify that the coupling multiplicity is one;
5. normalize the result to a primitive integer vector;
6. check all five raising residuals exactly.

Abstract couplings are shared by repeated exterior embeddings of the same
source representation.

## Phase 3: 44 embedded maps

Apply each abstract coupling to every corresponding level-14 source copy.
Stream one simple-root residual at a time and write an atomic result for each
map.

Acceptance requires:

- 23 exact abstract couplings;
- 44 exact embedded maps;
- zero residual under all five simple-root raising operators;
- a coefficient-mutation test that produces a nonzero residual.

### Phases 2 and 3 result

Phases 2 and 3 are complete:

- 23 source-target pairs certified;
- kernel dimension one for every abstract coupling;
- 44 embedded maps certified;
- embedded target counts `(00001): 5`, `(01001): 18`, `(10001): 8`,
  and `(20001): 13`;
- zero raising residuals for every embedded map;
- maximum primitive abstract coefficient magnitude 8,960;
- maximum checked embedded accumulator magnitude 3,999,195,200.

The four-worker stonkbot run completed in 9 minutes 54 seconds with a peak
resident set size of 45,530,608 KiB and no swap. The aggregate certificate is
`results/adynkra_11d_first_momentum_couplings_all.json`; the 23 abstract and 44
embedded certificates are stored beside it.

## Phase 4: momentum target couplings

Construct the four multiplicity-one target maps

\[
(10000)\otimes(10001)
\longrightarrow
(00001),(01001),(10001),(20001).
\]

Use the exact vector momentum basis and the established vector-spinor target
basis. Check every raising equation and fix a reproducible primitive
normalization.

### Phase 4 result

Phase 4 is complete. All four target products have one-dimensional
highest-weight kernels:

| Target | Highest-weight domain | Kernel dimension | Nonzero primitive coefficients |
|---|---:|---:|---:|
| `(00001)` | 35 | 1 | 35 |
| `(01001)` | 2 | 1 | 2 |
| `(10001)` | 10 | 1 | 10 |
| `(20001)` | 1 | 1 | 1 |

Every exact raising residual is zero. A coefficient mutation is detected in
each channel by a nonzero raising residual or, for the one-term channel, loss
of primitive normalization. The certificate is
`results/adynkra_11d_first_momentum_target_couplings.json`.

## Phase 5: joint compatibility matrix

Compose:

1. the 44 level-14 source intertwiners;
2. the four momentum target couplings;
3. the gamma-matrix anticommutator term in the superspace derivative algebra.

Combine this with the completed 12 leading maps and seven hook maps. Solve the
resulting exact linear system for the leading and first-momentum coefficients.

Report:

- exact rank;
- primitive integer kernel basis;
- whether the scalar-factorizing direction extends;
- whether any of the four direct-spinor kernel directions extends;
- all nonzero residual sectors if no extension exists.

## Phase 6: gauge intersection

Construct the six possible first-derivative gauge-parameter channels without
assuming that all six are symmetries. Compute their exact induced maps and
intersect the gauge-compatible space with the leading plus first-momentum
solution.

The gauge calculation must distinguish:

- a channel absent because its coefficient is zero;
- a channel annihilated by the candidate map;
- a channel removable by gauge-for-gauge reducibility;
- a channel that produces a genuine obstruction.

## Execution

The source systems range from 36,608 to 657,520 columns. Numerical kernel
proposal and the full exact batch belong on stonkbot. Coupling and residual
certification remain Rust computations. Worker count must be set by measured
resident memory rather than CPU count.

## Completion criterion

The first-momentum stage is complete only when the repository contains:

1. all 28 exact level-14 source fixtures;
2. a Rust certificate for every source fixture;
3. all 23 abstract coupling certificates;
4. all 44 embedded-map certificates;
5. the four exact momentum target couplings;
6. the joint leading plus first-momentum compatibility matrix;
7. explicit exact kernel vectors or an exact nonexistence certificate;
8. local and remote artifact hashes;
9. a statement separating the calculated symbol from any gauge, curvature,
   action, or field-equation claim.
