# Second-Momentum Multi-Column GPU Production Handoff

Date: 2026-08-19

## Assignment

Promote the proven multi-column CUDA contraction prototype into the production 11D second-momentum `F_X` pipeline. Group only columns whose source Dynkin label and PBW plan are identical, compute their independent coefficient lanes in one CUDA pass, preserve every existing per-column proof and artifact byte contract, and add resumable group execution.

Do not launch production columns as part of the implementation. Finish the code, tests, exact parity gates, and an isolated canary command first.

## Working-tree safety

The shared worktree contains unrelated uncommitted second-momentum and G-matrix work.

- Do not reset, clean, stash, checkout, or overwrite the shared worktree.
- Do not edit `src/four_color/gmatrix_csp.rs`, `scripts/lever_a/`, or G-matrix result files.
- Do not overwrite validated column artifacts.
- Work in an isolated copy or worktree.
- Stage specific files only. Never use `git add .` or `git add -A`.
- Do not launch or terminate production jobs without explicit approval.

## Proven baseline

The current production path is single-column:

1. `run_cuda_column` builds one static context and one accumulator.
2. One column's exact PBW descendants are lowered on the GPU.
3. Raw descendant-times-reciprocal terms stream through bounded host batches.
4. Each batch is sorted and reduced exactly by semantic source key.
5. The flat-plan CUDA kernel contracts the reduced keys into one 25,344-row Gaussian-residue column.
6. Per-column hashes, parity, rank, binary, and JSON are published independently.

Relevant production code:

- `src/eleven_dimensional_second_momentum_gpu.rs:1441`, single-batch `accumulate_terms`
- `src/eleven_dimensional_second_momentum_gpu.rs:1559`, `CudaStreamingColumnAccumulator`
- `src/eleven_dimensional_second_momentum_gpu.rs:2052`, persistent lowering visitor
- `src/eleven_dimensional_second_momentum_gpu.rs:3015`, `run_cuda_column`
- `cuda/second_momentum_fx_cuda.cu:585`, single-column flat-plan kernel
- `cuda/second_momentum_fx_cuda.cu:911`, reusable recoupling workspace
- `cuda/second_momentum_fx_cuda.cu:1567`, fused exact recoupling entry point
- `src/eleven_dimensional_second_momentum_20001_fx.rs:550`, `(20001)` preflight
- `src/eleven_dimensional_second_momentum_30001_fx.rs:555`, `(30001)` preflight
- `src/second_momentum_gpu_progress.rs`, live progress and batch telemetry
- `src/second_momentum_gpu_checkpoint.rs`, checkpoint identity, lock, rows, counters, and atomic persistence
- `src/second_momentum_gpu_word_hash.rs`, resumable word hash chain

## Measured bottleneck

Validated B300 column 74:

- 2,467,944,035 raw source terms
- 29,952,496,858,007 expanded contributions
- 51 minutes 7 seconds end to end
- 43 minutes 25 seconds in CUDA contraction
- 12.8 seconds in persistent PBW lowering kernels
- CPU parity passed
- single-column modular rank 1
- artifact SHA-256 `00e9c020063437caeee3928e7d5869ae8e397cbcb40d0f5d2bce9a3495094178`

Validated clean B300 column 76 spent 83.2 percent of wall time in contraction and only 0.5 percent in persistent lowering. Sharing traversal alone is not sufficient. The contraction must be vectorized across column lanes.

## Prototype and profile evidence

Exact prototype snapshots:

- `results/adynkra_11d_second_momentum_gpu_multicol_profile_20260819/prototype/second_momentum_fx_cuda.cu`
- `results/adynkra_11d_second_momentum_gpu_multicol_profile_20260819/prototype/eleven_dimensional_second_momentum_gpu.rs`

Machine-readable results:

- `results/adynkra_11d_second_momentum_gpu_multicol_profile_20260819/profile-summary.json`
- `results/adynkra_11d_second_momentum_gpu_multicol_profile_20260819/b300.log`
- `results/adynkra_11d_second_momentum_gpu_multicol_profile_20260819/rtx4090.log`

The profile used three real `(30002)` copy prefixes with 131,072 terms per copy:

- 168,066 union keys
- 130,107 keys shared by at least two copies
- 95,043 keys shared by all three copies
- every output row matched the current production kernel exactly

Measured B300 contraction:

| Width | Sequential | Multi-column | Speedup |
|---:|---:|---:|---:|
| 1 | 98.79 ms | 99.46 ms | 0.99x |
| 2 | 198.13 ms | 119.85 ms | 1.65x |
| 3 | 296.98 ms | 136.83 ms | 2.17x |
| 4 | 395.77 ms | 115.20 ms | 3.44x |
| 8 | 792.09 ms | 129.56 ms | 6.11x |
| 15 | 1,484.90 ms | 310.81 ms | 4.78x |
| 32 | 3,167.93 ms | 374.60 ms | 8.46x |

Widths above three cycle the three real coefficient vectors. They demonstrate hardware scaling but are not a claim that unrelated source labels can be grouped profitably.

Measured RTX 4090 contraction:

- width 2: 1.33x
- width 3: 1.46x
- width 15: 3.87x
- width 32: 6.88x

## Legal production groups

Group only copies of the same source Dynkin label. Require identical PBW plan, reciprocal map, prime, flat plan, row layout, and static semantic digest.

`(20001)` groups:

- globals 53-54: `10002`, width 2
- globals 55-56: `20100`, width 2
- globals 57-59: `20010`, width 3
- globals 60-61: `20002`, width 2

`(30001)` groups:

- global 62: `40000`, singleton fallback
- globals 63-64: `20100`, width 2
- global 65: `31000`, singleton fallback
- globals 66-68: `20010`, width 3
- globals 69-70: `20002`, width 2
- global 71: `30100`, singleton fallback
- globals 72-73: `30010`, width 2
- globals 74-76: `30002`, width 3

Do not combine different labels merely to fill lanes. Their union-key sparsity can erase the gain and complicate proof identities.

## Required architecture

### 1. Production multi-column CUDA context

Promote the prototype into a normal CUDA ABI, not a test-only entry point.

Suggested entry point:

```c
int adynkra_fx_cuda_accumulate_recoupled_multicol(
    void *context,
    const uint64_t *keys,
    const WideValue *key_major_values,
    uint32_t unique_count,
    uint32_t active_columns,
    uint32_t *row_major_output_real,
    uint32_t *row_major_output_imaginary,
    MultiColumnStats *stats,
    char *error,
    size_t error_capacity);
```

Contracts:

- `active_columns` is 1 through 3 in production, with tests through 32.
- Input layout is `[unique_key][column]`.
- Output layout is `[functional_row][column]`.
- Each column has independent exact coefficients and row accumulators.
- Use the prototype's dynamic power-of-two subwarp width: 1, 2, or 4 lanes for production widths 1, 2, or 3.
- Share semantic-key loads, plan-entry loads, wedge signs, functional routing, and hashes.
- Keep coefficient multiplication and row accumulation independent by column.
- Preserve the existing `u64` row-sum proof independently for every column.
- Context-owned buffers must be reusable. No `cudaMalloc` or `cudaFree` in the production batch loop.
- Enforce the declared total device cap before growth, including old plus new buffers and 64 MiB headroom.
- Preserve exact signed `i128` reduction or replace it only with a formally wider exact type.

### 2. Exact bounded union construction

The profile constructed a canonical union on the host. Production must make this boundary explicit and bounded.

For each source-copy group and word:

1. Produce each column's ordered raw terms exactly once.
2. Update each column's raw-term count and hash in its original order before unioning.
3. Merge semantic keys into one sorted union.
4. Store one exact coefficient lane per column, using zero for an absent key.
5. Drop a key only when all lanes are exactly zero.
6. Split the union into bounded batches without changing per-column additive semantics.

Preferred implementation is a block-valued persistent sparse handle:

- one canonical key array
- width-2 or width-3 coefficient vectors
- componentwise exact reduction for every simple-root lowering step
- immutable compact output handle after each root
- adjacent-PBW-prefix reuse as in the current opaque handle traversal

If the first milestone uses host merging, use a k-way merge over canonical sorted streams. Do not restore a global `BTreeMap`, do not retain a whole column, and prove the host/device caps from capacities rather than logical lengths.

### 3. Rust group runner

Add a group orchestrator instead of modifying single-column behavior in place.

Suggested API:

```rust
pub(crate) fn run_cuda_column_group(
    tranche: &str,
    local_ordinals: &[usize],
    prime: u32,
    device: i32,
    output_directory: &Path,
    cpu_parity_terms: usize,
    live_progress: Option<&LiveProgress>,
) -> Result<GpuFxColumnGroupReport, String>;
```

The runner must:

- preflight every member before allocating CUDA state
- reject mixed labels, PBW plans, reciprocal maps, primes, static digests, or flat-plan digests
- acquire one process-lifetime group lock covering every member output
- maintain independent per-column rows, counts, raw hashes, packed hashes, metadata, rank gates, and artifact paths
- flush all lanes atomically at a word boundary
- write the same individual binary and JSON artifact format as the existing single-column runner
- preserve singleton fallback through the current `run_cuda_column`
- refuse a conflict if any member already has different published bytes

Do not change column ordinals or binary row order.

### 4. Checkpoint and resume

Extend the existing checkpoint identity rather than inventing a second protocol.

A group checkpoint must bind:

- ordered tranche and local/global ordinals
- source label and source-copy list
- every fixture, abstract certificate, embedded-map, reciprocal-map, and PBW-plan digest
- ordered `(prime, static semantic digest)` pairs
- flat-plan digest
- active lane width and lane order
- next word ordinal
- independent row accumulator for every column
- independent per-column counts and v3 word-hash state
- group batch ordinal and rolling batch digest

Commit only after all pending terms for the word are flushed into every column accumulator. Write temp, fsync file, rename, and fsync parent. Resume must reject any identity or lane-order drift.

### 5. Observability

Extend the existing JSONL/status fields with:

- group ID and active lane count
- ordered local/global ordinals and source copies
- word ordinal and PBW root
- union-key count
- keys present in 1, 2, or 3 lanes
- raw terms per column
- batch terms and bytes
- upload, union/sort, reduce, contract, finalize, and download milliseconds
- per-column and aggregate terms per second
- device resident bytes, high-water bytes, and cap
- checkpoint generation, path, age, and digest
- per-column final row and artifact digests

Heartbeats must continue while a synchronous CUDA call is running.

## Correctness gates

All are mandatory.

### Unit and synthetic tests

- widths 1, 2, 3, 4, 8, 15, and 32
- zero lane, all-zero key, and mixed zero/nonzero lanes
- positive and negative `i128` coefficients
- `i128::MIN` rejection or formally safe wider handling
- duplicate keys within one lane
- same-key cancellation within and across batches
- cancellation in one lane while another remains nonzero
- disjoint key sets and fully overlapping key sets
- malformed mask, momentum pair, free spinor, and metadata rejection
- row-sum bound and allocation-cap failure before launch
- wrong lane order and wrong identity rejection

### Exact parity tests

- prototype real `(30002)` three-copy prefix, all three pinned primes
- real `(20010)` three-copy prefix, all three pinned primes
- one width-2 group from each tranche
- per-column CPU reference equality
- per-column current single-column CUDA equality
- identical expanded-contribution accounting per column
- identical final row bytes and semantic SHA-256

### Resume and publication tests

- crash before group checkpoint
- crash after checkpoint rename
- resume from a completed word
- reject truncated, reordered, wrong-prime, or wrong-copy checkpoint
- existing member artifact with identical bytes is idempotent
- existing member artifact with differing bytes is a hard conflict
- no member checkpoint advances unless every lane's word delta is durable

## Performance acceptance gates

On the B300 real 131,072-term prefix:

- width 1 regression no worse than 5 percent
- width 2 contraction speedup at least 1.50x
- width 3 contraction speedup at least 2.00x
- exact row parity required before timing is accepted
- no hot-loop allocation
- no cap or high-water underreporting

End-to-end canary targets:

- width 2 group at least 1.35x faster than two independent runs
- width 3 group at least 1.60x faster than three independent runs
- artifact bytes identical to independent single-column outputs

The expected tranche improvement is about 1.5x to 1.7x, not 10x.

## Implementation sequence

### Milestone A: Clean prototype promotion

1. Recreate the prototype changes in an isolated worktree.
2. Add a normal CUDA ABI and Rust wrapper.
3. Replace per-call allocations with context-owned reusable buffers.
4. Port the existing real-prefix parity benchmark unchanged.
5. Require B300 width-2 and width-3 acceptance gates.

Deliverable: production-quality multi-column contraction primitive with no runner changes.

### Milestone B: Bounded exact union stream

1. Add width-2/3 exact coefficient-vector type.
2. Add canonical k-way union merge and invariants.
3. Preserve original per-column hash order before merging.
4. Add cross-batch cancellation tests.
5. Measure union construction separately from contraction.

Deliverable: bounded group batches that reproduce independent column rows.

### Milestone C: Shared persistent lowering

1. Upload all highest-state copies for one label into a union-key vector handle.
2. Lower each PBW root once with componentwise exact coefficient reduction.
3. Compact all-zero vectors after each root.
4. Preserve adjacent-prefix opaque handles on device.
5. Download terminal ranges only if the contraction context cannot consume the handle directly.

Deliverable: one PBW traversal per copy group, with per-root parity against independent lowering.

### Milestone D: Group runner, checkpoint, and telemetry

1. Implement group preflight and locking.
2. Wire row accumulators, per-column hashes, word checkpoints, and resume.
3. Add group JSONL/status telemetry.
4. Publish the existing per-column artifacts unchanged.
5. Preserve singleton fallback.

Deliverable: resumable production group command with no production launch.

### Milestone E: Canary

After explicit approval only:

1. Run one `(30002)` width-3 group for prime `1073741783` in an isolated output directory.
2. Require all three per-column parity and rank gates.
3. Compare any already-known column bytes exactly.
4. Record B300 stage timings, memory, utilization, and artifact hashes.
5. Run one restart injection at a word boundary.
6. Promote remaining legal groups only after the canary passes.

## Suggested CLI

```text
adynkra-11d-second-momentum-gpu-fx-group \
  <tranche> <local-ordinals> <prime> <output-dir> \
  [cpu-parity-terms] [device] [status-file] [checkpoint-file]
```

Examples:

```text
adynkra-11d-second-momentum-gpu-fx-group \
  30001 12-14 1073741783 /isolated/output 128 0 status.json group.chk

adynkra-11d-second-momentum-gpu-fx-group \
  20001 4-6 1073741783 /isolated/output 128 0 status.json group.chk
```

Reject unsorted ordinals, duplicate ordinals, mixed-label groups, and widths above the group's certified multiplicity.

## Validation commands

Local type checks:

```bash
cargo check
DOCS_RS=1 cargo check --features cuda
```

CUDA build on RTX 4090:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export ADYNKRA_CUDA_ARCH=sm_89
cargo test --release --features cuda <focused-multicol-tests> -- --nocapture
```

CUDA build on B300:

```bash
export CUDA_HOME=/usr/local/cuda
export ADYNKRA_CUDA_ARCH=sm_100
cargo test --release --features cuda <focused-multicol-tests> -- --nocapture
```

Do not treat `cargo check` as proof of CUDA correctness. The B300 parity benchmark and exact per-column byte comparison are required.

## Final handoff deliverables

1. Changed-file inventory with ownership boundaries.
2. CUDA ABI and memory-layout documentation.
3. Exact proof argument for componentwise union, batching, and row folding.
4. Focused unit, parity, resume, and publication test results.
5. B300 width-2 and width-3 measurements.
6. Estimated complete `(20001)` and `(30001)` wall time from measured group runs.
7. Canary command, but no canary launch without approval.
8. No changes to G-matrix code or active census outputs.
