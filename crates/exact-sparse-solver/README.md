# Exact sparse solver

A standalone exact solver and accelerator reference for the large sparse systems.
It lives in a focused crate so solver tests and benchmarks do not rebuild the main binary.

Current scope:

- deterministic CSR construction from signed integer triplets
- deterministic CSR to CSC transpose-index construction
- arithmetic over the Mersenne prime `2^31 - 1`
- scalar `A*x` and `A^T*x`
- coordinate-major block `A*X` and `A^T*X` for future block Krylov methods
- Python-compatible level-12 highest-weight raising-matrix construction
- separate versioned numeric-CSR and ordered-source-labeled matrix digests
- a matrix inspection and release-mode SpMV benchmark CLI
- signed-unit CSR/transpose packing with the sign in bit 31
- a versioned semantic matrix digest and pinned nonzero diagonal PRNG
- an allocation-free CPU reference for 32-lane `A^T D A` blocks
- deterministic sparse echelon reduction with explicit fill and width budgets
- rational reconstruction, primitive integer normalization, and exact integer residual certificates

The implementation favors a compact, auditable representation: 32-bit offsets, indices,
field elements, and signed coefficients. It rejects matrices with dimensions or nonzero
counts that cannot be represented by that format. Parallel execution, structural
reduction, Krylov iteration, checkpointing, and accelerators are deliberately out of
scope for this first reference layer.

Run:

```sh
cargo test --manifest-path crates/exact-sparse-solver/Cargo.toml

cargo run --release \
  --manifest-path crates/exact-sparse-solver/Cargo.toml \
  --bin level12_matrix -- 30002 20

# Packed 32-lane normal-operator benchmark
cargo run --release \
  --manifest-path crates/exact-sparse-solver/Cargo.toml \
  --bin level12_atda_bench -- 30002 3

# Deterministic exact elimination with fill and pivot-width limits
cargo run --release \
  --manifest-path crates/exact-sparse-solver/Cargo.toml \
  --bin level12_eliminate -- 30002 10000000 10000
```

`level12_matrix` reports the checked CSR path, including its canonical-input
validation scan, after one warmup. `level12_atda_bench` reports the packed hot
path with no validation scan. Its memory figure is estimated logical buffer
payload, not peak RSS, and its JSON pins matrix, diagonal, input, PRNG, chain,
and output identities for reproducibility.
