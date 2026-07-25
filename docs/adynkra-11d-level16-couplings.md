# 11D level-16 couplings into (10001)

## Scope

This computation certifies the level-16 couplings

`V_lambda tensor S -> (10001)`

for the eight source representations and twelve embedded copies already stored
in `data/eleven_dimensional_spinor_bridge/`. It addresses this channel under the
stated exterior-algebra conventions. It does not solve the full 11D
prepotential problem.

## Inputs

| source representation | embedded copies |
|---|---:|
| (10000) | 1 |
| (20000) | 1 |
| (00100) | 2 |
| (00010) | 2 |
| (00002) | 1 |
| (10100) | 1 |
| (10010) | 1 |
| (10002) | 3 |

The precheck confirms this 8-representation, 12-copy manifest and confirms that
the multiplicity of `(10001)` in each `V_lambda tensor S` is one.

## Certificate construction

The certificate path is implemented in Rust.

1. Exterior masks are sorted and assigned contiguous indices in each required
   weight space.
2. Source states are generated in lexicographic PBW lowering-word order.
3. Linear dependence among lowering words is resolved by the rational rank of
   the integer Gram matrix in the exterior realization. No modular rank result
   is accepted as a certificate.
4. The five exterior raising and lowering actions are stored as CSR operators.
   State coefficients are contiguous `Vec<i64>` arrays.
5. The tensor-product highest-weight equations are reduced to a small integer
   Gram matrix. Its one-dimensional rational nullspace is converted to a
   primitive integer vector.
6. Every accepted vector is checked against all five simple-root raising
   equations. Accumulation uses checked `i128`.
7. The abstract coupling is constructed once for each source representation.
   The same PBW basis and primitive coefficients are then applied to every
   embedded copy of that representation.

The `(20000)` computation is the golden gate. In the canonical domain order the
primitive coefficients are

`(1, -2, 2, -2, 2, -4)`.

This is the reversed, sign-normalized form of the previously committed
`(4, -2, 2, -2, 2, -1)` lowering-chain convention. A test mutates one
coefficient and confirms that the raising residual becomes nonzero.

## Commands

```bash
cargo run --release -- adynkra-11d-level16-coupling-precheck
cargo run --release -- adynkra-11d-level16-coupling-build --label 00100
cargo run --release -- adynkra-11d-level16-coupling-verify \
  --label 00100 --copy 2
cargo run --release -- adynkra-11d-level16-coupling-verify --all --resume
```

The all-copy command limits parallelism with:

```bash
ADINKRA_LEVEL16_WORKERS=4
ADINKRA_LEVEL16_RAM_GIB=48
ADINKRA_LEVEL16_WORKER_GIB=10
```

The worker count is the minimum of the requested worker count, the memory
budget divided by the per-worker estimate, and the number of source
representations.

## Checkpoints and reproduction

Each abstract coupling and embedded-copy certificate is written independently
under `results/`. The writer serializes and validates a temporary JSON file,
calls `fsync`, renames it atomically, and calls `fsync` on the parent
directory. `--resume` accepts only parseable artifacts with `passed: true`.
A later failure does not remove earlier certificates.

The primary artifacts are:

```text
results/adynkra_11d_level16_coupling_<label>_abstract.json
results/adynkra_11d_level16_coupling_<label>_copy<n>.json
results/adynkra_11d_level16_couplings_all.json
```

The certificate fields record the PBW domain basis, primitive coefficients,
Gram rank, kernel dimension, residual counts for all five simple roots,
coefficient storage, checked-accumulator maximum, and the distinction between
the shared abstract coupling and each exterior embedding.

## Result

All eight abstract couplings have a one-dimensional kernel. All twelve
embedded copies have zero residual under every simple-root raising operator.

| source | domain dimension | largest primitive coefficient | CSR actions | CSR entries |
|---|---:|---:|---:|---:|
| (00002) | 35 | 16 | 180 | 201,155,046 |
| (00010) | 26 | 8 | 155 | 181,614,148 |
| (00100) | 14 | 4 | 95 | 121,994,140 |
| (10000) | 1 | 1 | 5 | 8,274,008 |
| (10002) | 262 | 1,344 | 785 | 542,070,578 |
| (10010) | 192 | 1,280 | 660 | 489,656,832 |
| (10100) | 100 | 1,152 | 395 | 346,321,252 |
| (20000) | 6 | 4 | 45 | 61,359,872 |

| source copy | coupled nonzero terms | raising residuals |
|---|---:|---:|
| (00002), copy 1 | 5,010,137 | 0, 0, 0, 0, 0 |
| (00010), copy 1 | 6,062,694 | 0, 0, 0, 0, 0 |
| (00010), copy 2 | 126,984 | 0, 0, 0, 0, 0 |
| (00100), copy 1 | 4,812,342 | 0, 0, 0, 0, 0 |
| (00100), copy 2 | 113,256 | 0, 0, 0, 0, 0 |
| (10000), copy 1 | 594,896 | 0, 0, 0, 0, 0 |
| (10002), copy 1 | 10,562,130 | 0, 0, 0, 0, 0 |
| (10002), copy 2 | 9,108,349 | 0, 0, 0, 0, 0 |
| (10002), copy 3 | 1,062,708 | 0, 0, 0, 0, 0 |
| (10010), copy 1 | 10,036,388 | 0, 0, 0, 0, 0 |
| (10100), copy 1 | 9,229,122 | 0, 0, 0, 0, 0 |
| (20000), copy 1 | 2,735,880 | 0, 0, 0, 0, 0 |

The four-worker local run completed in 202.92 seconds with a maximum resident
set size of 16.23 GiB. The independent four-worker stonkbot run completed in
217.66 seconds with a maximum resident set size of 36.70 GiB under the 48 GiB
budget.

All 22 level-16 result files, including the precheck and aggregate report, had
matching SHA-256 hashes on the local machine and stonkbot. The independently
regenerated full aggregate had SHA-256
`19e15f19b45c21d3dafb7d7d08fd5a6c0039c99390a295390acf36020e872b41`.
The subsequent atomic-resume summary, using the conservative 10 GiB per-worker
estimate, matched at
`bada78574729dec6700dbd27979af87c444d5bdeb5a4ec9cddc9f5c2151a4547`.

The full Rust test suite passed with 398 tests passed, zero failed, and nine
ignored slow tests.

## Boundary

These artifacts certify the twelve source embeddings and their couplings into
the `(10001)` channel. The seven hook-source couplings and the exact
exterior-derivative matrix were completed in the subsequent calculation
documented in
[`adynkra-11d-level17-hook-derivative-matrix.md`](adynkra-11d-level17-hook-derivative-matrix.md).
Neither calculation determines the gauge quotient, momentum corrections, or
a nonlinear 11D supergravity equation.
