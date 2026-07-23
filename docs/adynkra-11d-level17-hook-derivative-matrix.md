# Eleven-dimensional level-17 hook couplings and derivative matrix

## Result

The direct spinor-prepotential calculation now includes:

1. all seven source couplings
   \[
   S\otimes\bigwedge^{17}S\longrightarrow(11000);
   \]
2. the multiplicity-one target coupling
   \[
   (00001)\otimes(10001)\longrightarrow(11000);
   \]
3. the exact exterior derivative from the twelve level-16 leading maps to
   the seven level-17 hook maps.

The resulting rational matrix has size \(7\times12\), rank 7, and nullity 5.
Every one of its twelve columns reconstructs exactly in the seven-dimensional
hook basis. All twelve reconstruction residual norms are zero.

This is the zero-spacetime-momentum exterior symbol. It does not include the
44 first-momentum correction directions, select a gauge transformation, or
determine a field equation.

## Bases

The column basis, in order, is

```text
10000#1, 20000#1,
00100#1, 00100#2,
00010#1, 00010#2,
00002#1,
10100#1,
10010#1,
10002#1, 10002#2, 10002#3
```

The row basis is

```text
10001#1,
01001#1, 01001#2,
20001#1,
11001#1, 11001#2, 11001#3
```

The complete rational matrix is stored in
`results/adynkra_11d_level17_derivative_matrix.json`.

## Exact kernel

In the column ordering above, a primitive integer basis for the five-dimensional
kernel is

```text
(0, 0, 1, -62, -1, 0, 0, 0, 0, 0, 0, 0)
(0, 0, 0,   1,  0, 1, 0, 0, 0, 0, 0, 0)
(6, 0, 0,   0,  0, 0, 1, 0, 0, 0, 0, 0)
(11520, 0, 540, 2160, 0, 0, 0, -5, 9, 0, 0, 0)
(168, 84, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0)
```

Rust verifies the product of the matrix with each vector to be exactly zero.
A one-unit mutation of a nonzero kernel coefficient produces a nonzero
residual.

The existence of this kernel means that five independent combinations of the
twelve leading maps have zero hook image under this exterior symbol. It does
not mean that those combinations are gauge invariant or extend through the
momentum-dependent superspace algebra.

## Scalar-factorizing direction

The map inherited from

\[
V=D^\alpha\Psi_\alpha
\]

was constructed directly from the committed level-15 scalar bridge and the
exact charge-conjugation bilinear. In the twelve-column basis its coordinates
are

```text
(-7/16, -1/16, 3/128, 3/32, -3/64, 99/32,
  1/32, 1/4608, -1/2560, 0, 1/1344, 0).
```

The twelve leading vectors have Gram rank 12. The scalar-factorizing vector
reconstructs in their span with zero residual norm, and its image under the
new derivative matrix is exactly zero in all seven hook coordinates. This is
an independent compatibility check between the scalar bridge and the direct
spinor calculation.

The scalar-factorizing direction therefore occupies one dimension of the
five-dimensional kernel. The remaining quotient has dimension four at this
exterior-symbol stage.

## Exactness and execution

Accepted calculations use:

- primitive integer highest-weight couplings;
- canonical PBW lowering words;
- exact rational Gram solves;
- checked `i128` accumulation;
- exact rational rank and nullspace calculations;
- atomic JSON checkpoints.

The seven hook couplings completed on stonkbot in 3 minutes 0.94 seconds with
a peak resident set of 17,942,172 KiB. The derivative matrix and scalar
factorization completed in 10 minutes 11.29 seconds with a peak resident set
of 17,521,840 KiB.

The copied and remote derivative artifacts have the same SHA-256:

```text
630fc8701ef7d93e9ce37cdd50be4863dba8b062b8c0112a5d88277d8ec4f5cd
```

The Rust suite passes 403 tests, with zero failures and nine ignored slow
tests.

## Reproduction

```bash
cargo run --release -- adynkra-11d-level17-hook-precheck

ADINKRA_LEVEL17_WORKERS=4 \
ADINKRA_LEVEL17_RAM_GIB=48 \
ADINKRA_LEVEL17_WORKER_GIB=10 \
cargo run --release -- adynkra-11d-level17-hook-verify --all

cargo run --release -- adynkra-11d-level17-derivative-matrix
```

Implementation:

- `src/eleven_dimensional_level16_couplings.rs`
- `src/eleven_dimensional_bridge.rs`
- `src/eleven_dimensional_spinor_bridge_kernels.rs`
- `src/eleven_dimensional_clifford.rs`

## Next gate

The next calculation is the gauge-compatible extension:

1. construct the six first-derivative gauge-parameter intertwiners;
2. leave their coefficients undetermined;
3. compute their induced action on the twelve leading directions;
4. intersect the exact gauge-compatible space with the five-dimensional hook
   kernel;
5. construct the 44 first-momentum correction intertwiners and repeat the
   calculation with the full first-momentum symbol.

Only combinations that survive those steps can enter a linearized curvature
or field-equation search.
