# Exact Sparse Solver Architecture

## Objective

Build a reusable solver for the large, tall, extremely sparse integer matrices used by
the 11D level-12 highest-weight calculation. The immediate systems have up to 353,120
columns, 1,194,000 rows, and 4,378,928 nonzero entries. Every original entry is `+1`
or `-1`, and every row has at most seven entries.

The solver must prioritize correctness and reproducibility before throughput. No
kernel is publishable until it passes a full integer residual check against the
original raising operator.

## Execution layers

1. **Deterministic matrix construction**
   - Preserve the Python generator's sorted source-mask column ordering.
   - Build canonical CSR and CSC representations.
   - Combine duplicate coordinates over the integers and remove zeros.
   - Hash dimensions, offsets, indices, coefficients, and source masks.

2. **CPU reference backend**
   - Perform exact arithmetic over `GF(2^31 - 1)`.
   - Provide scalar and block `A X` and `A^T X` operations.
   - Reuse caller-owned buffers in performance paths.
   - Serve as the parity oracle for optimized CPU and CUDA implementations.

3. **Structural analysis**
   - Compute row and column degree distributions.
   - Use maximum matching and Dulmage-Mendelsohn or SCC structure only as ordering
     information.
   - Do not infer numerical rank from structural rank. The three missing matrices
     have full structural column rank but exact nullity one.

4. **Deterministic sparse elimination**
   - Use rectangular Markowitz pivot selection with explicit row and column
     adjacency.
   - Prefer unit pivots and low-fill pivots.
   - Track fill, memory, and pivot progress in restartable checkpoints.
   - Stop cleanly at configured fill or memory thresholds rather than exhausting
     machine memory.

5. **Black-box fallback**
   - Apply `B = A^T D A` without materializing `B`, where `D` is a reproducible
     nonzero random diagonal over the field.
   - Use block Lanczos or block Wiedemann for large residual cores.
   - Use independent seeds and, when practical, a second prime.
   - Record clearly whether the rank lower bound is deterministic or probabilistic.

6. **CUDA backend**
   - Keep Rust responsible for construction, hashes, checkpoints, reconstruction,
     and CPU verification.
   - Use custom CUDA kernels behind a small C ABI for exact block sparse
     multiplication.
   - Store both orientations with packed signed 31-bit indices.
   - Use coordinate-major block vectors with width 32 so one warp handles one sparse
     row and one lane handles one block component.
   - Avoid cuSPARSE because the operation requires exact Mersenne-field arithmetic.

## Publication contract

A generated kernel must satisfy all of the following:

1. Matrix hashes and source-mask ordering match the declared generator version.
2. Modular rank plus modular nullity equals the number of columns.
3. The recovered vector is rationally reconstructed and converted to primitive
   integer form with deterministic sign normalization.
4. Every original integer row has exactly zero dot product with every published
   vector.
5. Output byte width, byte count, SHA-256, copy order, and checkpoint metadata match
   the existing artifact schema.
6. A conflicting same-label certificate is rejected before any published file is
   replaced.

## Delivery sequence

1. Land compact CSR/CSC and field arithmetic with dense parity tests.
2. Port the level-12 matrix generator and prove byte-for-byte ordering parity.
3. Benchmark CPU scalar and block operators on a completed fixture.
4. Add deterministic rectangular sparse elimination and checkpointing.
5. Solve a completed hard system and reproduce its published kernel hash.
6. Attempt `00010`, `00100`, and `00000` in that order.
7. Add block Krylov and CUDA only where measured CPU profiles justify them.

This sequence provides a trustworthy CPU path quickly while preserving a stable
boundary for the maximum-performance GPU backend.

## Initial measurements

The first Rust reference implementation already provides a large deterministic
elimination speedup while preserving the original leftmost-column semantics:

| Completed system | Python elimination | Rust elimination | Speedup |
|---|---:|---:|---:|
| `30002` | 0.25 s | 0.018 s | 14x |
| `01002` | 74.51 s | 4.72 s | 16x |
| `01100` | 2,083.77 s | 109.48 s | 19x |

The Rust `01100` run reproduced rank 44,938, nullity 2, free columns 43,993 and
43,994, maximum pivot width 668, and zero modular residuals. Peak RSS was about
44 MB.

The first previously missing system, `00010`, completed through the same
deterministic path in 140.05 seconds of elimination time. It produced rank
144,677, nullity 1, free column 100,875, maximum pivot width 477, and 856,225
retained pivot entries. Rational reconstruction gave a primitive integer kernel
with maximum absolute coefficient 1 and SHA-256
`5f78511aa24ab9ab779f3eea217e72f58dbe7a1d09c96d3e094b6f538bc982b2`.
The full integer residual, kernel independence, and characteristic-zero rank
bounds passed. Peak RSS was about 100 MB. Publication into the shared fixture
manifest remains a separate locked step.

On the M4 Pro, the packed block-width-32 `A^T D A` CPU reference measured:

| Missing system | Milliseconds per operator | Estimated working payload |
|---|---:|---:|
| `00010` | 23.90 ms | 111.6 MB |
| `00100` | 30.64 ms | 140.7 MB |
| `00000` | 63.90 ms | 289.2 MB |

An RTX 4090 CUDA microbenchmark used the `00000` dimensions and nonzero count
with a synthetic matching degree profile. Its exact packed block-width-32
operator measured 0.361 ms for `D A X`, 0.517 ms for `A^T Y`, and 0.878 ms for
the complete `A^T D A` application. A lane-zero CPU modular comparison passed.
The feature-gated production backend then passed exact CPU parity and
repeatability on small matrices and the real `30002` and `01002` matrices under
CUDA 12.6 for `sm_89`. This remains a sparse-kernel floor, not an end-to-end
solve time. Recurrence updates, checkpointing, reconstruction, and independent
verification remain outside that number.
