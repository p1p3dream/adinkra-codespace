# Eleven-dimensional four-form GPU plan for a sub-30-minute exact gate

**Status date:** 2026-08-30
**Scope:** Exact four-form constraint construction, modular rank and nullspace,
and characteristic-zero reconstruction after the physical normalization is fixed
**Status:** Design and acceptance specification. No solver implementation is
authorized by this document.

## 1. Executive decision

The present normalization problem has only one or two coefficient columns.
That is far below the crossover at which GPU elimination is useful. The next
implementation should therefore accelerate exact constraint construction and
retain the existing CPU rank/nullspace path. A general GPU eliminator is a
contingency for a later Hom-space inventory that produces hundreds or thousands
of columns, not a prerequisite for the current gate.

The repository already contains the right building blocks, but not a reusable
four-form solver:

1. `cuda/complete_f_sparse_cuda.cu` and `ExactCudaSparseOperator` provide exact
   denominator-cleared sparse application, batching, composition, compaction,
   persistent device state, memory accounting, and CPU/GPU parity tests.
2. `cuda/second_momentum_fx_cuda.cu` provides proven three-prime scheduling,
   arithmetic over pinned primes near `2^30`, Gaussian-field arithmetic,
   overflow flags, bounded output protocols, and CUDA error handling.
3. The streamed CPU modular elimination used by the complete-F and p3 paths is
   already fast enough for small coefficient counts.
4. `crates/exact-sparse-solver` is not a general replacement for those paths.
   Its CUDA algorithm is specialized to packed signed-unit matrices, the fixed
   Mersenne prime `2^31 - 1`, products of the form `A^T D A`, and a bordered-CG
   recovery protocol intended for nullity at most one.

**Hard rule:** do not adapt the signed-unit exact-sparse CUDA path blindly.
Supporting arbitrary Gaussian coefficients, three independent primes, and
general nullity would replace its arithmetic, data model, certificates, and
recovery argument. That is a new solver and must be designed and validated as
such.

## 2. Claim boundary

The GPU can accelerate a convention-fixed linear-algebra calculation. It
cannot establish which physical normalization or equation convention is
authoritative. The following inputs must exist before an authoritative run:

- a pinned definition of the four-form equation or constraint family;
- a canonical source and target basis with digests;
- the relative normalization against the graviton and gravitino sectors;
- a declared local-Lorentz and target-gauge descent map;
- a complete inventory of coefficient columns;
- an ordered row-family schema that is stable across CPU and GPU execution.

Until these are fixed, GPU results are engineering canaries only. The
sub-30-minute objective begins after these inputs are available.

## 3. Measured facts and performance target

The design separates measurements from projections.

### 3.1 Existing measurements

- The current complete-F COL3 construction contains 321 columns and
  35,611,900 exact terms.
- Its existing streamed modular rank scan takes about 16.6 seconds on CPU.
- The complete-F sparse CUDA canary processed 7,493 products in about
  0.063 milliseconds, approximately 119 million sparse products per second.
- The exact-sparse RTX 4090 production measured a 353,120-column,
  1,194,000-row, 4,378,928-nonzero signed-unit matrix, but that result exercises
  its specialized `A^T D A` and bordered-CG algorithm. It does not validate a
  general four-form eliminator.

### 3.2 End-to-end target

For a convention-fixed ansatz with at most 4,096 columns, the desired budget is:

| Stage | Target wall time |
|---|---:|
| Input validation, hashes, denominator audit | 2 minutes |
| Exact GPU constraint construction | 15 minutes |
| Three-prime rank or nullspace | 2 minutes |
| CRT lifting and exact residual verification | 8 minutes |
| Final publication | 1 minute |
| Reserve | 2 minutes |
| **Total** | **30 minutes** |

These are acceptance targets, not measured promises. The first production run
must publish timings for every stage. Hom-space enumeration, interpretation of
the physical normalization, and derivation of the descent maps are outside this
wall-time claim.

## 4. Crossover policy

The number of coefficient columns controls the rank/nullspace implementation.

| Column count `n` | Required path |
|---:|---|
| `1 <= n <= 512` | Existing streamed CPU elimination |
| `513 <= n <= 8192` | GPU panel elimination is eligible after a measured CPU baseline |
| `n > 8192` | Stop and profile; consider generalized block Wiedemann only after review |

Additional gates:

1. GPU sparse evaluation is eligible when a batch contains roughly one million
   or more sparse products, or when profiling shows more than one minute in the
   CPU evaluation loop.
2. GPU elimination is not implemented merely because GPU evaluation is used.
3. For the current one- or two-column normalization system, the CPU must compute
   rank and kernel directly. Kernel extraction is a few exact scalar tests, not
   a GPU problem.
4. For a later `n <= 512` system, GPU construction may still feed the CPU
   streamed eliminator without materializing the full matrix.
5. For `513 <= n <= 8192`, implementation of GPU elimination requires a real
   row-count, sparsity, fill-in, and CPU timing report. If CPU rank remains below
   one minute, retain CPU elimination.

## 5. End-to-end data flow

```text
canonical exact operators and coefficient columns
                    |
                    v
       exact sparse CUDA construction
                    |
                    v
 denominator-cleared device COO or row batches
                    |
          +---------+---------+
          |                   |
          v                   v
  n <= 512 handoff      n >= 513 optional
  to CPU row stream     three-prime GPU panels
          |                   |
          +---------+---------+
                    v
        modular rank and kernel records
                    |
                    v
        CPU CRT and rational reconstruction
                    |
                    v
     exact Q(i) residual against every row
                    |
                    v
       immutable certificate and manifest
```

The exact row stream is authoritative. Modular reductions, pivots, and kernels
are derived views. Row ordering must be canonical and independent of thread
scheduling.

## 6. Constraint-construction architecture

### 6.1 Canonical row visitor

Add a dynamic row visitor only after the physical convention is fixed. It must
emit, for every row:

- row-family identifier;
- canonical multi-index or representation key;
- coefficient-column index;
- Gaussian rational numerator pair `(real, imaginary)`;
- positive denominator;
- source operator and basis identifiers.

The visitor must support both a CPU reference sink and a CUDA staging sink.
Neither sink may define semantic row order. The visitor does.

### 6.2 Device layout

Use structure-of-arrays layout:

- `row_offsets: u64[row_count + 1]`;
- `column_indices: u32[nnz]`;
- `exact_real: i64[nnz]`;
- `exact_imaginary: i64[nnz]`;
- optional `row_keys` in a separate canonical digest stream;
- three modular real planes and three modular imaginary planes when reduction
  is required.

For a dense row batch used by elimination, use prime-major storage:

```text
[prime][row_in_batch][column][real_or_imag]
```

This makes three-prime independence explicit and gives coalesced column access
inside a panel. Do not interleave exact coefficients with modular residues.

### 6.3 Kernel sequence

The minimum kernel sequence is:

1. existing exact sparse apply or composed-apply kernel;
2. exact cancellation and compaction;
3. denominator-cleared validation or modular reduction;
4. optional lane-major to prime-major transpose;
5. optional GPU panel reduction when the crossover gate is met;
6. bounded result emission with overflow and capacity flags.

Constraint generation should traverse the exact operator once and produce all
three prime reductions in one launch sequence. Do not repeat the expensive
operator traversal independently for each prime.

## 7. Direct device-output handoff

The current compact complete-F API returns COO data to Rust. Downloading a
large exact intermediate and uploading it again for modular reduction wastes
bandwidth and synchronization time. Add an opaque device-output handoff only
after the CPU reference path is stable.

### 7.1 Required handle

The handoff object owns or borrows:

- device row offsets;
- device column indices;
- device exact real and imaginary arrays;
- nonzero count and capacity;
- common denominator and conservative accumulation bounds;
- stream and completion event;
- context generation number;
- semantic input digest and row-range identifier.

The consumer must wait on the producer event in the same context or through an
explicit CUDA event dependency. Raw device pointers must not escape to general
Rust code.

### 7.2 Lifetime and failure rules

- The producing context remains alive until the consumer releases the handle.
- A generation number prevents reuse after context reset.
- Capacity overflow, arithmetic overflow, invalid indices, and asynchronous
  CUDA errors invalidate the whole batch.
- A failed batch publishes no pivot checkpoint and no final artifact.
- The fallback path downloads the exact COO batch and uses the CPU reducer.
- Direct handoff and fallback must produce the same canonical row digest.

### 7.3 Why this is the first optimization

The raw sparse CUDA kernel is already fast. Host construction, conversion,
allocation, and copies dominate the observed end-to-end time. Removing the
round trip can improve the whole pipeline without introducing a new rank
algorithm.

## 8. Exact arithmetic rules

### 8.1 Denominator clearing

Every exact entry is an element of `Q(i)`. Before modular reduction:

1. compute a declared common denominator per immutable batch or for the whole
   matrix;
2. prove every scaled real and imaginary numerator is integral;
3. record the denominator and its SHA-256-bound provenance;
4. compute `gcd(denominator, p) = 1` for every pinned prime;
5. reject the run if any denominator is zero, negative after canonicalization,
   nonintegral after scaling, or noninvertible modulo a pinned prime.

The pinned primes are:

```text
1073741783
1073741723
1073741719
```

All are below `2^30` and congruent to 3 modulo 4. Therefore `x^2 + 1` is
irreducible and the pair representation defines the field `F_p(i)`.

### 8.2 Gaussian arithmetic

Represent `a + b i` as two canonical residues in `[0, p)`. Multiplication is:

```text
real = ac - bd mod p
imaginary = ad + bc mod p
```

For nonzero `a + b i`, inversion is:

```text
(a + b i)^-1 = (a - b i) * (a^2 + b^2)^-1 mod p
```

The norm is nonzero because each pinned field has `p = 3 mod 4`. The scalar
inverse may use fixed-exponent Fermat exponentiation. Arithmetic code should be
copied from or factored from the proven second-momentum implementation only
with direct parity tests. Its hard-coded p3 row and plan kernels are not
reusable as four-form kernels.

### 8.3 Near-`2^30` reduction

For each pinned prime write `p = 2^30 - c`, with small positive `c`. A 64-bit
product can be reduced using exact 30-bit folds followed by bounded conditional
subtractions. The implementation must be proved against a wider CPU reference
for boundary values and random inputs. No floating-point arithmetic is allowed.

### 8.4 Accumulation and overflow

`complete_f_sparse_cuda.cu` uses signed 64-bit atomic accumulation. It is exact
only when the Rust caller has proved a conservative per-lane L1 bound below
`i64::MAX`. The four-form path must preserve that contract.

For every exact output lane, compute a bound of the form:

```text
sum over contributing products of
abs(operator_scaled_coefficient) * abs(input_scaled_coefficient)
```

Include every composition stage and multiplicity. Reject before launch when
the bound exceeds `i64::MAX`. Never rely on signed wraparound.

If a valid workload exceeds the bound, choose exactly one reviewed remedy:

1. accumulate with the existing wide-value and explicit overflow pattern; or
2. reduce each contribution modulo the three primes before accumulation.

The second remedy cannot emit an authoritative exact row stream by itself. It
must be paired with a separately validated exact CPU path or an exact wide
accumulator for characteristic-zero residuals.

## 9. Rank and nullspace strategy

### 9.1 Current one- or two-column system

Stream each canonical constraint row to the existing CPU reducer. For one
column, rank is one if any admissible modular entry is nonzero. For two columns,
ordinary exact or modular elimination is immediate. Always scan the complete
required constraint set before declaring a survivor. Early termination is
allowed only after full column rank, because additional rows cannot reduce
rank.

### 9.2 CPU path through 512 columns

Maintain one normalized echelon state per prime. Consume rows in canonical
order, reduce against existing pivots, select the leftmost nonzero pivot,
normalize, and append the row. If any prime reaches rank `n`, the
characteristic-zero matrix has full column rank after denominator
admissibility is certified.

### 9.3 Optional GPU panels for 513 through 8192 columns

If the crossover gates are met, keep three independent dense echelon states on
the GPU. Each row batch is reduced against resident normalized pivots. Pivot
selection must be deterministic: leftmost nonzero column and first surviving
canonical row. Normalize with Gaussian inversion, eliminate within the panel,
then append pivots in canonical order.

Approximate storage for three Gaussian prime planes is `24 n^2` bytes:

| `n` | Echelon storage |
|---:|---:|
| 1,024 | 24 MiB |
| 2,048 | 96 MiB |
| 4,096 | 384 MiB |
| 8,192 | 1.5 GiB |

A double-buffered 1,024-row dense batch adds about 192 MiB for `n = 4,096`
and 384 MiB for `n = 8,192`. Exact COO storage is approximately 24 bytes per
nonzero before allocator overhead. The run must report actual peak allocated
and high-water device memory.

### 9.4 Nullspace extraction

After all required rows are consumed, record pivot and free columns. Produce a
canonical reduced-row-echelon representation, then back-substitute one basis
vector per free column. A modular survivor is provisional. It is not a
physical survivor and not a characteristic-zero certificate.

## 10. Characteristic-zero reconstruction boundary

The CPU owns final reconstruction and verification.

1. Align pivot patterns and free-column normalizations across primes.
2. CRT-combine corresponding Gaussian coordinates. The three pinned primes
   give a combined modulus of approximately 90 bits.
3. Apply rational reconstruction to real and imaginary components.
4. Clear denominators and normalize each vector to a primitive Gaussian-
   integer representative with a fixed sign and phase convention.
5. Apply every original exact constraint over `Q(i)`.
6. Accept only if every residual is exactly zero.

If modular nullity persists but reconstruction fails, add admissible primes or
strengthen the reconstruction bound. Do not guess a rational vector. Do not
publish a survivor from modular nullity alone.

Proof directions must remain explicit:

- full rank at one admissible prime proves characteristic-zero full column
  rank;
- rank loss modulo a prime does not prove characteristic-zero rank loss;
- a reconstructed exact nonzero kernel vector proves a characteristic-zero
  survivor for the scanned constraint set;
- a survivor of a subset of bidegrees is not a survivor of the complete
  physical condition until bidegree exhaustion is proved.

## 11. Checkpoint and provenance gates

### 11.1 Immutable run manifest

Before computation, write a manifest containing:

- repository commit and dirty-state policy;
- executable SHA-256;
- CUDA, nvcc, driver, and runtime versions;
- compute target, expected `sm_89` for the RTX 4090;
- GPU model, UUID, and memory;
- hostname and start time;
- ordered prime list;
- source, target, coefficient, gauge, and Lorentz basis identifiers and digests;
- equation and constraint-family identifiers;
- row schema version and canonical family order;
- matrix dimensions and expected or bounded nonzero count;
- operator, normalization, and denominator-stream digests;
- denominator-admissibility results;
- requested tile, batch, and memory limits.

The run must refuse to resume if any identity field changes.

### 11.2 Checkpoint boundary

Checkpoint only after a complete constraint family, bidegree, or declared row
range. Never checkpoint a partially reduced canonical row without a transaction
record. Each checkpoint contains:

- rows generated and rows accepted per family;
- exact row-stream SHA-256 chain state;
- per-prime ranks;
- normalized pivot rows;
- pivot and free-column lists;
- invalid, capacity-overflow, and arithmetic-overflow counters;
- stage timings and candidate/product counts;
- current and peak device memory;
- prior-checkpoint digest and current-checkpoint digest.

Write to a new temporary path, flush, fsync, atomically rename, then fsync the
parent directory. Use an exclusive run lock. Never overwrite an authoritative
artifact during validation.

### 11.3 Heartbeat and publication

A heartbeat may be updated every five seconds with observational progress. It
is not proof evidence and is never used for resume. The final report is written
last, after all payload hashes and exact residual checks pass. It references
immutable payloads by SHA-256 rather than embedding mutable paths alone.

## 12. Exact validation ladder

No production claim is allowed until all applicable rungs pass.

### Gate V0: static inputs

- Independently verify each pinned prime is prime and 3 modulo 4.
- Verify denominator gcd is one at every prime.
- Verify all basis, operator, normalization, and row-schema digests.

### Gate V1: scalar arithmetic

- Compare GPU and wide CPU modular reduction at zero, one, `p - 1`, `p`,
  `2p - 1`, `2^30 - 1`, `u32::MAX`, and 64-bit product boundaries.
- Compare at least one million deterministic random products per prime.
- Test Gaussian addition, subtraction, multiplication, norm, and inversion.
- Verify `x * inverse(x) = 1` for every sampled nonzero Gaussian value.

### Gate V2: sparse construction

- Reuse current exact complete-F CPU/GPU parity fixtures.
- Include exact cancellation to zero, imaginary-only entries, duplicate COO
  entries, multi-stage composition, empty rows, and maximum safe L1 bounds.
- Confirm direct device handoff and host fallback have identical row digests.
- Intentionally exceed the accumulation and output-capacity bounds and require
  a hard failure.

### Gate V3: synthetic rank and kernel

- Test zero, full-rank, and every intermediate-rank matrix shape used by the
  implementation.
- Test nullity greater than 32 so no warp-width assumption survives.
- Test duplicate and dependent rows, zero columns, imaginary pivots, and
  different pivot patterns before canonical RREF.
- Include a matrix whose rank drops at one intentionally bad prime while the
  characteristic-zero rank remains full.
- Compare CPU and GPU rank, pivot sets, and canonical modular kernels.

### Gate V4: deterministic operations

- Repeat an uninterrupted run and require byte-identical proof payloads.
- Interrupt only at a safe checkpoint and require resumed output to match.
- Reject corrupted checkpoints, reordered row families, changed primes,
  changed basis digests, and changed binaries.

### Gate V5: repository oracles

- Replay the archived p3 77-column matrix and require rank 77 at all three
  primes when its complete production payload is mounted.
- Replay the COL3 four-form oracle and require rank 320, nullity 1, and the
  canonical kernel proportional to `e_320` when its archived shard payload is
  available. If the payload is absent locally, regenerate under a new output
  root rather than overwriting any authoritative path.

### Gate V6: physical four-form result

- Scan every required constraint family and bidegree.
- Verify every reconstructed vector against the original exact `Q(i)` rows.
- Verify local-Lorentz and target-gauge descent separately.
- Bind the result to the authoritative relative normalization.
- Record whether the result is a full-rank no-go or an exact survivor.
- State the completeness scope. Do not label a restricted ansatz result as an
  irreducibility theorem.

## 13. Smallest implementation sequence

No solver code should be written while the live normalization system remains
one or two columns unless profiling identifies a construction bottleneck.

1. Freeze the physical convention, row schema, and coefficient inventory.
2. Add a CPU canonical constraint-row visitor and exact one- or two-column
   reference certificate.
3. Profile construction, normalization, serialization, and CPU rank
   separately.
4. If sparse construction exceeds one minute, add the direct device-output
   handoff to the existing complete-F CUDA context.
5. Fuse three-prime reduction into that handoff, while retaining the CPU
   streamed eliminator.
6. Add V0 through V2 and the current small-system physical oracle.
7. Add manifest, checkpoint, heartbeat, and atomic publication support.
8. Recompute the column inventory. Implement GPU panel elimination only if
   `n >= 513` and the measured CPU rank stage exceeds one minute.
9. If GPU elimination is authorized, implement V3 through V5 before production.
10. Run V6 and publish exact residuals with the final certificate.

## 14. Stop rules

Stop without a scientific claim if any of the following occurs:

- the physical normalization or equation identity is not authoritative;
- the required bidegree set is not defined;
- a basis or operator digest changes after the manifest is sealed;
- a denominator is not invertible at a pinned prime;
- an exact accumulation bound exceeds `i64::MAX` without a reviewed wide or
  modular alternative;
- any CUDA invalid, capacity, or overflow flag is nonzero;
- CPU/GPU row digests differ;
- checkpoint provenance does not form a valid SHA-256 chain;
- modular kernels cannot be aligned or reconstructed;
- a reconstructed vector has any nonzero exact residual;
- the local-Lorentz or target-gauge descent certificate is incomplete.

The accepted output is either a fully scoped exact no-go or a set of exact
survivors for the declared ansatz. Neither output alone proves full
eleven-dimensional irreducibility.

## 15. Definition of done

The GPU-assisted four-form gate is complete only when:

1. all applicable validation gates pass;
2. the end-to-end wall time is measured on the named RTX 4090;
3. every input and payload has a stable digest;
4. denominator admissibility and arithmetic bounds are recorded;
5. every modular claim has the correct characteristic-zero interpretation;
6. every survivor is reconstructed and exactly verified;
7. resume and deterministic replay have been demonstrated;
8. the final report states the exact ansatz and bidegree scope;
9. no specialized signed-unit solver has been presented as a general
   four-form certificate; and
10. the authoritative artifacts were written to new paths and published only
    after validation.
