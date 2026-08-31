# Shared GPU proof observability contract for the eleven-dimensional physical gates

**Status date:** 2026-08-30
**Scope:** Four-form identification, relative normalization, local-Lorentz
descent, and target-gauge descent
**Status:** Schema and operations contract. It is not a physical certificate.

## 1. Purpose

The four physical-F work tracks must expose one operational and evidentiary
surface:

| Track ID | Work product |
|---|---|
| `four_form_identification` | Authoritative identification of the closed Eq. (40) `Psi_[3]` branch with physical `A_3/G_4` |
| `relative_normalization` | Relative coefficient and phase against the graviton and gravitino sectors |
| `local_lorentz_descent` | Exact representative-independence and local-Lorentz descent checks |
| `target_gauge_descent` | Exact `F K = 0`, quotient routing, and separately scoped completeness work |

Every track may have different kernels and proof payloads. All four must use
the same manifest, heartbeat, event, checkpoint, witness, error, and final
report schemas defined here. A job that does not implement the shared contract
is a development canary and cannot publish an authoritative certificate.

The contract serves three distinct purposes:

1. live operational visibility without reading partial proof artifacts;
2. safe interruption, transfer, and exact checkpoint adoption;
3. durable evidence that binds outputs to source semantics and execution
   identity.

Observability fields are not mathematical evidence unless the relevant field
is explicitly included in a semantic digest. Heartbeats are always
observational. Manifests, validated checkpoints, immutable payloads, and final
reports form the evidence chain.

## 2. Normative language and scalar conventions

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative.

- JSON integers MUST be nonnegative decimal integers unless a field explicitly
  permits a signed value.
- Byte counts MUST use bytes, not MiB, in JSON.
- Durations MUST use integer nanoseconds or milliseconds as named.
- Rates MUST be finite nonnegative JSON numbers. NaN and infinity are invalid.
- Timestamps MUST be UTC RFC 3339 strings with a `Z` suffix.
- SHA-256 values MUST be 64 lowercase hexadecimal characters.
- Enum strings and object keys are case-sensitive.
- Schemas MUST reject unknown fields in evidence-bearing objects.
- Arrays whose order affects semantics MUST state and validate that order.
- Paths in evidence MUST be relative to the immutable run root unless marked
  as observational.

Each serialized object has:

```json
{
  "schema_version": "adynkra-11d-gpu-proof-<object>-v1",
  "...": "object-specific fields"
}
```

Schema changes require a new version. Readers MUST NOT infer compatibility
from a similar field set.

## 3. Run-root layout and publication boundary

Each job owns one run root:

```text
<run-root>/
  manifest.json
  owner.lock
  status.json
  events.jsonl
  cancel.request.json
  checkpoints/
    checkpoint-00000001.json
    checkpoint-00000001.payload
    ...
  witnesses/
    witness-<kind>-<ordinal>.json
    witness-<kind>-<ordinal>.bin
  payloads/
    ... immutable proof payloads ...
  errors/
    error-<sequence>.json
  report.json
```

Rules:

1. `manifest.json` is written and fsynced before proof work starts.
2. `owner.lock` is held with a nonblocking process-lifetime advisory lock.
3. `status.json` is the latest atomic heartbeat snapshot.
4. `events.jsonl` is append-only observational history.
5. checkpoints and proof payloads are immutable after atomic publication.
6. `report.json` is the commit record and MUST be published last.
7. The existence of `report.json` does not imply validity. Adoption verifies
   the report, manifest, payload inventory, hashes, and semantic gates.
8. A failed or cancelled run MUST NOT publish a success report.

Temporary files MUST remain inside the run root and use a `.tmp-<pid>-<nonce>`
suffix. Atomic publication is:

1. create a new temporary file with exclusive creation;
2. write all bytes;
3. flush userspace buffers;
4. `fsync` the file;
5. rename to a previously nonexistent final path;
6. `fsync` the parent directory.

Conflicting existing bytes are an error. Authoritative paths are never
overwritten.

## 4. Shared identity and hash vocabulary

All evidence objects refer to a `RunIdentity`. Four different hash classes
must remain separate.

| Hash class | Meaning |
|---|---|
| `source_sha256` | Exact source fixture or input tensor bytes |
| `basis_sha256` | Ordered basis labels, dimensions, and encoding |
| `map_sha256` | Canonical sparse or dense map entries with domain/codomain basis IDs |
| `semantic_sha256` | Domain-separated digest of all mathematical choices relevant to the work unit |

The semantic hash MUST include, as applicable:

- track ID and work-unit ID;
- equation and convention IDs;
- source, target, coefficient, gauge, and Lorentz basis hashes;
- normal-form and row-key schema versions;
- map hashes and composition order;
- coefficient-column inventory and order;
- constraint-family and bidegree inventory and order;
- exact denominator policy and pinned primes;
- partition, block, tile, and canonical iteration definitions;
- expected output type and acceptance predicate.

Operational settings that cannot change mathematical output, such as heartbeat
frequency or GPU block size, belong in the manifest but not necessarily in the
semantic hash. If changing a setting can change row order, pivot selection,
capacity truncation, or arithmetic, it is semantic.

### 4.1 RunIdentity schema

```json
{
  "track_id": "four_form_identification",
  "run_id": "20260830T120000Z-four-form-identification-0001",
  "job_id": "four-form-eq40-blocks",
  "work_unit_id": "bidegree-p3-block-000042",
  "semantic_sha256": "<64 hex>",
  "manifest_sha256": "<64 hex>"
}
```

| Field | Type | Required | Invariant |
|---|---|---:|---|
| `track_id` | enum | yes | One of the four IDs in Section 1 |
| `run_id` | string | yes | Unique immutable run-root identity |
| `job_id` | string | yes | Stable logical job identity |
| `work_unit_id` | string | yes | Stable canonical shard or block identity |
| `semantic_sha256` | SHA-256 | yes | Matches manifest semantic hash |
| `manifest_sha256` | SHA-256 | yes | Hash of canonical manifest payload excluding this self-reference |

## 5. Immutable manifest schema

`manifest.json` has schema version
`adynkra-11d-gpu-proof-manifest-v1`.

### 5.1 Complete JSON shape

```json
{
  "schema_version": "adynkra-11d-gpu-proof-manifest-v1",
  "identity": {
    "track_id": "target_gauge_descent",
    "run_id": "20260830T120000Z-target-gauge-0001",
    "job_id": "fk-zero-all-families",
    "work_unit_id": "all",
    "semantic_sha256": "<64 hex>",
    "manifest_sha256": "<64 hex>"
  },
  "created_at": "2026-08-30T18:00:00Z",
  "repository": {
    "commit": "<40 hex>",
    "branch": "gpu-four-form-descent",
    "dirty": false,
    "dirty_diff_sha256": null
  },
  "executable": {
    "relative_path": "bin/adinkra-codespace",
    "sha256": "<64 hex>",
    "build_profile": "release",
    "rustc_version": "<string>"
  },
  "accelerator": {
    "backend": "cuda",
    "cuda_runtime_version": "<string>",
    "cuda_driver_version": "<string>",
    "nvcc_version": "<string>",
    "compute_target": "sm_89",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_uuid": "<string>",
    "total_vram_bytes": 25757220864
  },
  "host": {
    "hostname": "<string>",
    "os": "linux",
    "architecture": "x86_64",
    "logical_cpu_count": 32,
    "total_ram_bytes": 0
  },
  "semantics": {
    "equation_id": "<string>",
    "convention_id": "<string>",
    "normal_form_schema": "<string>",
    "row_key_schema": "<string>",
    "source_sha256": ["<64 hex>"],
    "basis_sha256": {"source": "<64 hex>", "target": "<64 hex>"},
    "map_sha256": {"F": "<64 hex>", "K": "<64 hex>"},
    "coefficient_inventory_sha256": "<64 hex>",
    "constraint_inventory_sha256": "<64 hex>",
    "ordered_primes": [1073741783, 1073741723, 1073741719],
    "common_denominator": "13440",
    "denominator_audit_sha256": "<64 hex>",
    "acceptance_predicate_id": "<string>"
  },
  "partition": {
    "phase_order": ["prepare", "generate", "reduce", "verify", "publish"],
    "block_count": 1,
    "block_schema": "<string>",
    "canonical_block_order_sha256": "<64 hex>",
    "checkpoint_boundary": "completed_block"
  },
  "limits": {
    "host_memory_cap_bytes": 0,
    "device_memory_cap_bytes": 0,
    "device_headroom_bytes": 0,
    "output_capacity_bytes": 0,
    "wall_time_limit_seconds": 1800,
    "heartbeat_interval_seconds": 5
  },
  "launch": {
    "argv": ["<string>"],
    "environment_allowlist": {"NAME": "value"},
    "requested_cpu_threads": 1,
    "requested_cuda_streams": 1,
    "tile_shape": [1, 1, 1],
    "batch_size": 1
  }
}
```

### 5.2 Field rules

- `repository.dirty=true` is permitted only for an explicitly nonauthoritative
  canary. `dirty_diff_sha256` is then required.
- `source_sha256` is an ordered nonempty array. The order is semantic.
- `basis_sha256` and `map_sha256` are typed dictionaries. Required keys depend
  on the track and MUST be enumerated by its adapter.
- Missing physical maps are represented by refusing the run, not by null
  hashes in an authoritative manifest.
- `common_denominator` is a decimal string to avoid JSON integer limits.
- Every prime requires a separately hashed denominator-admissibility record.
- A zero memory cap means "not configured" only for canaries. Authoritative
  runs require explicit nonzero host, device, headroom, and output caps.
- `environment_allowlist` contains only variables that can affect the run.
  Secrets and irrelevant ambient environment values MUST NOT be recorded.
- `manifest_sha256` is computed from canonical JSON with that field replaced
  by 64 zeroes, using the domain
  `adynkra-11d-gpu-proof-manifest-v1\0`.

## 6. Five-second heartbeat and status schema

The worker MUST publish `status.json` immediately after manifest validation,
then every five seconds until it reaches a terminal state. The interval is
measured by monotonic time. Wall-clock timestamps are labels, not duration
sources.

The heartbeat thread MUST be independent of the compute loop so a long CUDA
kernel or blocked CPU stage remains observable. If the monitor cannot publish
status, it appends a best-effort error to stderr and sets a shared observability
fault flag. An authoritative run MUST fail before final publication if that
fault prevented the required history or final snapshot.

### 6.1 Status JSON shape

```json
{
  "schema_version": "adynkra-11d-gpu-proof-status-v1",
  "identity": {"track_id": "local_lorentz_descent", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
  "event": "heartbeat",
  "sequence": 42,
  "timestamp": "2026-08-30T18:03:30Z",
  "monotonic_elapsed_ms": 210000,
  "pid": 12345,
  "hostname": "stonkbot",
  "state": "running",
  "phase": "generate",
  "work": {
    "unit": "canonical_blocks",
    "completed": 42,
    "total": 100,
    "remaining": 58,
    "fraction": 0.42
  },
  "rate": {
    "window_seconds": 30,
    "current_units_per_second": 0.2,
    "mean_units_per_second": 0.19,
    "current_products_per_second": 119000000.0,
    "current_rows_per_second": 25000.0
  },
  "eta": {
    "method": "bounded_recent_rate",
    "seconds": 290,
    "lower_seconds": 250,
    "upper_seconds": 420,
    "confidence": "medium"
  },
  "resources": {
    "cpu_utilization_percent": 320.0,
    "rss_bytes": 4294967296,
    "rss_high_water_bytes": 5368709120,
    "gpu_utilization_percent": 97.0,
    "vram_used_bytes": 8589934592,
    "vram_high_water_bytes": 9663676416,
    "device_allocator_bytes": 7516192768,
    "device_allocator_high_water_bytes": 8589934592
  },
  "blocks": {
    "active_block_ordinal": 42,
    "completed_blocks": 42,
    "total_blocks": 100,
    "counters": {
      "input_terms": 0,
      "expanded_products": 0,
      "reduced_keys": 0,
      "nonzero_outputs": 0,
      "rows_tested": 0,
      "pivots_found": 0,
      "exact_residuals_tested": 0,
      "witness_candidates": 0,
      "invalid_values": 0,
      "capacity_overflows": 0,
      "arithmetic_overflows": 0
    }
  },
  "checkpoint": {
    "generation": 7,
    "last_completed_block_ordinal": 41,
    "relative_path": "checkpoints/checkpoint-00000007.json",
    "semantic_sha256": "<64 hex>",
    "age_seconds": 12
  },
  "first_witness": {
    "observed": true,
    "kind": "nonzero_residual",
    "canonical_ordinal": 9001,
    "relative_path": "witnesses/witness-nonzero-residual-00009001.json",
    "semantic_sha256": "<64 hex>",
    "canonicalized": true
  },
  "cancellation": {
    "requested": false,
    "request_observed_at_block": null,
    "safe_stop_pending": false
  },
  "last_error": null
}
```

### 6.2 Status field invariants

| Field | Rule |
|---|---|
| `sequence` | Starts at zero and strictly increases |
| `state` | `starting`, `running`, `checkpointing`, `cancelling`, `cancelled`, `failed`, or `completed` |
| `phase` | One manifest-declared phase or `terminal` |
| `work.completed` | Monotone and no greater than `work.total` |
| `work.remaining` | Exactly `total - completed` |
| `work.fraction` | Exactly representable calculation within declared tolerance, in `[0,1]` |
| rate counters | Derived from monotonic counter differences, never wall clock |
| `eta.seconds` | Null until rate is estimable; never used as a proof gate |
| resource high-water fields | Monotone for the lifetime of the job |
| proof counters | Monotone; invalid/capacity/arithmetic counters must be zero for success |
| checkpoint generation | Monotone and names an existing validated immutable checkpoint |

CPU utilization may exceed 100 percent when multiple cores are used. GPU
utilization and VRAM usage SHOULD come from NVML or an equivalent stable API.
Allocator bytes and device-wide VRAM are separate measurements and MUST NOT be
conflated.

The final synchronous status snapshot is mandatory. It is written after the
compute loop stops and before a final report is attempted.

## 7. Event stream schema

`events.jsonl` contains one JSON object per line with schema
`adynkra-11d-gpu-proof-event-v1`. Each event repeats the identity, sequence,
timestamp, elapsed time, event name, phase, and a typed payload.

```json
{
  "schema_version": "adynkra-11d-gpu-proof-event-v1",
  "identity": {"track_id": "relative_normalization", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
  "sequence": 12,
  "timestamp": "2026-08-30T18:01:00Z",
  "monotonic_elapsed_ms": 60000,
  "event": "block_completed",
  "phase": "reduce",
  "payload": {"block_ordinal": 5, "rows": 1000, "semantic_sha256": "<64 hex>"}
}
```

Required event names:

```text
run_started
manifest_validated
phase_started
phase_completed
block_started
block_completed
heartbeat
first_witness_observed
first_witness_canonicalized
checkpoint_started
checkpoint_published
checkpoint_adopted
cancellation_requested
cancellation_observed
safe_stop_completed
error
payload_published
report_published
run_completed
```

Events are operational history. Missing event lines cannot be reconstructed
from proof payloads and MUST be disclosed, but an append interruption does not
corrupt an already atomically published checkpoint. Readers ignore a final
unterminated JSONL line.

## 8. Per-block counters

Each track maps its work to canonical blocks. A block report has schema
`adynkra-11d-gpu-proof-block-v1`.

```json
{
  "schema_version": "adynkra-11d-gpu-proof-block-v1",
  "block_ordinal": 42,
  "block_id": "constraint-family-03-bidegree-p3-row-range-000042",
  "input_sha256": "<64 hex>",
  "output_sha256": "<64 hex>",
  "started_elapsed_ms": 1000,
  "finished_elapsed_ms": 1400,
  "counters": {
    "input_terms": 100,
    "expanded_products": 7493,
    "reduced_keys": 3400,
    "nonzero_outputs": 900,
    "rows_tested": 32,
    "pivots_found": 1,
    "exact_residuals_tested": 0,
    "witness_candidates": 0,
    "invalid_values": 0,
    "capacity_overflows": 0,
    "arithmetic_overflows": 0
  },
  "timings_ns": {
    "host_prepare": 0,
    "host_to_device": 0,
    "kernel": 0,
    "device_to_host": 0,
    "host_reduce": 0,
    "exact_verify": 0,
    "checkpoint": 0
  },
  "prime_results": [
    {"prime": 1073741783, "rank_before": 0, "rank_after": 1, "rows_consumed": 32}
  ]
}
```

All counter keys are required even when zero. Track-specific extension counters
belong under a versioned `track_counters` object, never by silently adding keys
to `counters`. Aggregate counters in status, checkpoint, and report MUST equal
the exact sum of completed block counters, except high-water values which use
maximum rather than sum.

## 9. First-witness capture

A witness is the earliest canonical object that demonstrates a declared
predicate, such as:

- a nonzero exact residual;
- a failed `F K = 0` coordinate;
- representative dependence under a local-Lorentz shift;
- disagreement between two normalization channels;
- the first modular pivot proving full column rank;
- a provisional nonzero kernel or survivor candidate;
- an invalid arithmetic or capacity condition.

GPU race order is not canonical order. An atomic first writer is only a
provisional witness. The worker MUST either:

1. use an atomic minimum over canonical ordinals and then replay that ordinal;
   or
2. collect candidate minima per block, reduce them deterministically, and
   replay the global minimum.

The replay produces `adynkra-11d-gpu-proof-witness-v1`:

```json
{
  "schema_version": "adynkra-11d-gpu-proof-witness-v1",
  "identity": {"track_id": "target_gauge_descent", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
  "kind": "fk_nonzero_residual",
  "predicate_id": "exact-gaussian-residual-nonzero-v1",
  "canonical_ordinal": 9001,
  "canonical_key": {"constraint_family": "<string>", "bidegree": "p3", "row": 17, "column": 2},
  "source_block_ordinal": 42,
  "observed_prime": null,
  "modular_value": null,
  "exact_value": {"real_numerator": "1", "imaginary_numerator": "0", "denominator": "2"},
  "input_sha256": "<64 hex>",
  "payload_relative_path": null,
  "payload_sha256": null,
  "replay_backend": "cpu_exact",
  "replay_passed": true,
  "witness_semantic_sha256": "<64 hex>"
}
```

Rules:

- A failure witness MUST be replayed with exact CPU arithmetic before it can
  close a mathematical gate.
- A modular survivor witness is provisional until characteristic-zero lifting
  and exact residual verification pass.
- First witness means minimum canonical ordinal, not earliest timestamp.
- Once canonicalized, a lower ordinal discovered later is an internal error and
  indicates incomplete candidate reduction.
- Large tensor context belongs in an immutable binary payload referenced by
  hash. JSON contains only the typed key and minimal exact value.

## 10. Checkpoint schema

Checkpointing occurs only at boundaries declared by the manifest, normally a
completed canonical block, constraint family, bidegree, or PBW word. A
checkpoint is a complete commit unit. It has no partially applied row or block.

### 10.1 Envelope and payload

```json
{
  "schema_version": "adynkra-11d-gpu-proof-checkpoint-envelope-v1",
  "checkpoint_semantic_sha256": "<64 hex>",
  "checkpoint": {
    "schema_version": "adynkra-11d-gpu-proof-checkpoint-v1",
    "identity": {"track_id": "four_form_identification", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
    "generation": 7,
    "previous_checkpoint_sha256": "<64 hex or null for generation zero>",
    "next_block_ordinal": 42,
    "completed_block_count": 42,
    "completed_block_hash_chain_sha256": "<64 hex>",
    "source_row_hash_chain_sha256": "<64 hex>",
    "counters": {"input_terms": 0, "expanded_products": 0, "reduced_keys": 0, "nonzero_outputs": 0, "rows_tested": 0, "pivots_found": 0, "exact_residuals_tested": 0, "witness_candidates": 0, "invalid_values": 0, "capacity_overflows": 0, "arithmetic_overflows": 0},
    "prime_states": [
      {"prime": 1073741783, "rank": 0, "pivot_columns": [], "state_relative_path": "checkpoints/checkpoint-00000007-prime-1073741783.bin", "state_sha256": "<64 hex>"}
    ],
    "track_state": {"schema_version": "<track-specific version>", "payload_relative_path": "checkpoints/checkpoint-00000007.payload", "payload_sha256": "<64 hex>"},
    "first_witness": null,
    "resource_high_water": {"rss_bytes": 0, "vram_bytes": 0, "device_allocator_bytes": 0},
    "elapsed_compute_ms": 0,
    "created_at": "2026-08-30T18:03:30Z"
  }
}
```

`checkpoint_semantic_sha256` is a domain-separated hash of canonical checkpoint
JSON with the envelope digest replaced by 64 zeroes, plus the ordered payload
hashes. A checkpoint generation MUST link to the previous checkpoint digest.

### 10.2 Atomicity and ownership

- Acquire the run lock before reading or writing checkpoints.
- Publish all binary state payloads first and the checkpoint envelope last.
- Never modify a published checkpoint or payload.
- Checkpoint counters and high-water values are monotone.
- `next_block_ordinal` means every lower canonical block is included exactly
  once and no later block is included.
- CUDA work must be synchronized and checked for asynchronous errors before a
  checkpoint is eligible.
- A cancellation request does not make in-flight partial state adoptable.

## 11. Resume and adoption rules

Resume means continuing within the same run identity. Adoption means a new
process or run root accepts prior immutable work.

### 11.1 Required adoption checks

An adopter MUST verify all of the following before loading payload state:

1. schema versions are exactly supported;
2. manifest hash is valid;
3. track, job, work-unit, and semantic identities match;
4. source, basis, map, convention, normal-form, row-schema, inventory, prime,
   and denominator hashes match;
5. executable policy permits adoption from the producing commit and binary;
6. checkpoint SHA chain is complete from generation zero;
7. every referenced payload exists, has the declared size, and matches SHA-256;
8. block and row hash chains match the declared prefix;
9. counters are internally consistent and monotone;
10. invalid, capacity-overflow, and arithmetic-overflow counters are zero;
11. pivot or accumulator state validates structurally for every prime;
12. the checkpoint boundary is complete and `next_block_ordinal` is valid;
13. any witness reference resolves and replays;
14. no success report conflicts with the checkpoint state.

If any check fails, adoption fails closed. A worker MAY start a fresh run root,
but MUST NOT repair or overwrite the rejected evidence in place.

### 11.2 Allowed and forbidden differences

Allowed when semantics and arithmetic are unchanged:

- hostname, PID, wall-clock start time;
- GPU UUID within an explicitly equivalent `sm_89` backend policy;
- CPU thread count;
- heartbeat interval only if it remains at most five seconds;
- tile and batch sizes proven output-invariant by validation;
- output root path.

Forbidden:

- different source, basis, map, convention, or equation hashes;
- changed prime list or prime order;
- changed denominator or denominator policy;
- changed canonical block partition or ordering;
- changed normal-form or row-key schema;
- changed acceptance predicate;
- any missing or mutated checkpoint payload;
- adopting a dirty canary as authoritative evidence;
- adopting a report without independently validating its complete inventory.

## 12. Shared cancellation and fail-fast behavior

Cancellation is cooperative and evidence-preserving. It is never implemented
by deleting files or publishing a false completion record.

### 12.1 Cancellation request schema

`cancel.request.json` has schema
`adynkra-11d-gpu-proof-cancel-request-v1`:

```json
{
  "schema_version": "adynkra-11d-gpu-proof-cancel-request-v1",
  "identity": {"track_id": "relative_normalization", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
  "requested_at": "2026-08-30T18:05:00Z",
  "reason_code": "operator_request",
  "reason": "<short text>",
  "requester": "<string>"
}
```

The worker checks for cancellation at least once per heartbeat and at every
block boundary. On observation it:

1. changes state to `cancelling`;
2. stops scheduling new blocks;
3. allows an in-flight CUDA kernel to complete unless device safety requires
   process termination;
4. checks CUDA completion and errors;
5. discards any incomplete block;
6. publishes the last complete checkpoint if safe and not already published;
7. writes a terminal `cancelled` status and `safe_stop_completed` event;
8. exits with a distinct cancellation status;
9. does not publish `report.json`.

External process termination is a last resort. Before terminating a job, the
operator must inspect current progress, heartbeat age, phase, and checkpoint
age. SIGTERM requests the same cooperative path. SIGKILL produces no new
checkpoint and requires adoption from the last validated one.

### 12.2 Fail-fast conditions

The worker MUST stop scheduling new work immediately on:

- manifest or semantic identity mismatch;
- invalid denominator or prime;
- host or device memory-cap breach;
- output capacity overflow;
- arithmetic overflow or invalid residue;
- CUDA launch, synchronization, or device error;
- CPU/GPU parity or row-digest mismatch;
- nonmonotone counters or impossible block ordering;
- corrupted checkpoint or payload;
- exact replay failure for a claimed witness;
- an authoritative prerequisite marked conditional or missing.

Some mathematical witnesses also permit early success or failure:

- A nonzero exact residual is a decisive failure for an identity claim after
  CPU replay.
- Full column rank at one admissible prime is decisive for a scoped no-kernel
  claim after denominator validation.
- A modular nullity is never decisive success and must not stop the complete
  row scan unless the declared theorem permits it.
- Completeness claims never stop at the first passing sample.

## 13. Error taxonomy

Every terminal or recoverable error uses
`adynkra-11d-gpu-proof-error-v1`:

```json
{
  "schema_version": "adynkra-11d-gpu-proof-error-v1",
  "identity": {"track_id": "local_lorentz_descent", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
  "sequence": 1,
  "timestamp": "2026-08-30T18:04:00Z",
  "phase": "verify",
  "block_ordinal": 42,
  "class": "proof_mismatch",
  "code": "exact_residual_nonzero",
  "severity": "fatal",
  "retryability": "requires_input_change",
  "message": "<bounded text>",
  "witness_relative_path": "witnesses/witness-nonzero-residual-00009001.json",
  "checkpoint_generation": 7
}
```

| Class | Examples | Default retryability |
|---|---|---|
| `configuration` | missing map, unsupported schema, bad limits | `requires_input_change` |
| `provenance` | hash mismatch, dirty authoritative run, inventory conflict | `requires_input_change` |
| `arithmetic` | noninvertible denominator, invalid residue, integer overflow | `requires_code_or_input_change` |
| `capacity` | host cap, VRAM cap, output bound | `retry_with_limits` only if semantics are unchanged |
| `cuda` | launch, sync, illegal access, device lost | `retry_fresh_or_checkpoint` |
| `io` | disk full, fsync failure, short write | `retry_fresh_or_checkpoint` |
| `checkpoint` | broken chain, corrupt payload, invalid next block | `reject_checkpoint` |
| `proof_mismatch` | CPU/GPU mismatch, exact residual nonzero | `requires_investigation` |
| `cancellation` | operator request, scheduler stop | `resume_from_checkpoint` |
| `internal` | impossible state, nonmonotone counters | `requires_code_change` |

`message` is diagnostic only. Automated decisions use `class`, `code`,
`severity`, and `retryability`. Stack traces and logs may be separate payloads
but cannot replace typed errors.

## 14. Final report and report-last publication

`report.json` has schema `adynkra-11d-gpu-proof-report-v1` and is the only
commit record for a completed job.

```json
{
  "schema_version": "adynkra-11d-gpu-proof-report-v1",
  "identity": {"track_id": "four_form_identification", "run_id": "<string>", "job_id": "<string>", "work_unit_id": "<string>", "semantic_sha256": "<64 hex>", "manifest_sha256": "<64 hex>"},
  "outcome": "pass",
  "claim_scope": {
    "theorem_id": "<string>",
    "ansatz_id": "<string>",
    "constraint_inventory_sha256": "<64 hex>",
    "bidegrees_exhaustive": false,
    "completeness_claimed": false,
    "conditional_prerequisites": []
  },
  "started_at": "2026-08-30T18:00:00Z",
  "completed_at": "2026-08-30T18:10:00Z",
  "elapsed_ms": 600000,
  "completed_blocks": 100,
  "aggregate_counters": {"input_terms": 0, "expanded_products": 0, "reduced_keys": 0, "nonzero_outputs": 0, "rows_tested": 0, "pivots_found": 0, "exact_residuals_tested": 0, "witness_candidates": 0, "invalid_values": 0, "capacity_overflows": 0, "arithmetic_overflows": 0},
  "resource_high_water": {"rss_bytes": 0, "vram_bytes": 0, "device_allocator_bytes": 0},
  "prime_results": [{"prime": 1073741783, "rank": 0, "nullity": 0, "result_sha256": "<64 hex>"}],
  "first_witness": null,
  "payloads": [{"kind": "exact-certificate", "relative_path": "payloads/certificate.json", "bytes": 0, "sha256": "<64 hex>", "semantic_sha256": "<64 hex>"}],
  "final_checkpoint": {"generation": 10, "relative_path": "checkpoints/checkpoint-00000010.json", "sha256": "<64 hex>"},
  "validation_gates": [{"gate_id": "V0", "outcome": "pass", "evidence_sha256": "<64 hex>"}],
  "error_count": 0,
  "report_semantic_sha256": "<64 hex>"
}
```

Publication order:

1. finish all blocks;
2. synchronize CUDA and inspect all error counters;
3. publish final checkpoint;
4. publish exact witness and proof payloads;
5. validate every payload by reopening and hashing it;
6. run the final CPU exact checks;
7. write terminal `completed` status;
8. build the report from reopened immutable evidence;
9. atomically publish `report.json`;
10. append `report_published` and `run_completed` events.

If step 10 is interrupted, the report can still be valid because it is the
commit record. If any earlier step fails, no report is published.

Allowed `outcome` values are `pass`, `scoped_no_go`, and `exact_survivors`.
Failure and cancellation are terminal status states, not successful report
outcomes. A track adapter MAY define more precise outcomes in a separately
versioned payload, but the shared outcome remains one of these values.

## 15. Track adapter requirements

Every track implements a small adapter that declares:

- required source, basis, and map hash keys;
- canonical block key and order;
- phase list;
- track-specific counters;
- witness kinds and replay functions;
- checkpoint payload schema;
- proof payload schema;
- acceptance predicate;
- legal early-stop rules;
- completeness and conditional-prerequisite fields.

### 15.1 Four-form identification

Required semantic bindings include Eq. (40), gamma and epsilon conventions,
closed-form projection, physical `A_3/G_4` target basis, Bianchi identities,
and all row families used to distinguish auxiliary from physical branches.
Its decisive failure witness is the minimum exact coordinate violating the
declared physical identities.

### 15.2 Relative normalization

Required bindings include the authorized graviton, gravitino, and four-form
maps, their target equations, phase convention, and all normalization test
channels. A ratio inferred from one channel is provisional until every required
channel agrees. The report records the primitive exact ratio, not a floating-
point approximation.

### 15.3 Local-Lorentz descent

Required bindings include raw and gamma-traceless frame bases, quotient map,
canonical representative map, local-Lorentz injection, and the final physical
F map. Witnesses identify the minimum exact coordinate in `q L`, `q s - 1`, or
`F(raw + L lambda) - F(raw)` that fails.

### 15.4 Target-gauge descent

Required bindings include the physical F and K maps, polynomial ring and
monomial order, target-gauge source basis, quotient routing, and scanned degree
or bidegree inventory. `F K = 0` and completeness of K are separate predicates,
payloads, and report fields. Passing the former cannot close the latter.

## 16. Validation and mutation gates

### Gate O0: schema strictness

- Round-trip every object through strict readers with unknown-field rejection.
- Reject missing required fields, duplicate JSON keys, invalid enums, negative
  counters, NaN rates, malformed hashes, and unsorted semantic arrays.
- Freeze canonical JSON and domain-separation test vectors.

### Gate O1: manifest identity

- Recompute manifest and semantic hashes independently.
- Mutate one source byte, basis label, map entry, prime, denominator,
  coefficient order, row order, equation ID, and convention ID.
- Require every mutation to change semantic identity and prevent adoption.

### Gate O2: heartbeat behavior

- Block the compute thread for at least 20 seconds and require heartbeats at
  five-second intervals within scheduler tolerance.
- Verify sequence, monotone elapsed time, work, counters, and high-water marks.
- Simulate unavailable GPU metrics and require explicit nullability or a typed
  observability error, never fabricated zero utilization.
- Require a synchronous terminal snapshot.

### Gate O3: block accounting

- Verify aggregate counters from an independent sum of immutable block records.
- Mutate each counter and require report validation failure.
- Reorder, duplicate, omit, and truncate blocks and require hash-chain failure.
- Verify current and mean rates against monotonic timestamps.

### Gate O4: witness determinism

- Inject multiple simultaneous GPU witnesses in different launch orders.
- Require the same minimum canonical witness and byte-identical replay record.
- Mutate the key, exact value, source block, or payload and require rejection.
- Force a modular-only survivor and require the report to remain provisional.

### Gate O5: checkpoint atomicity

- Kill the worker during payload write, fsync, envelope write, rename, and
  parent fsync.
- Require adoption of only the last complete generation.
- Mutate prior digest, generation, next-block ordinal, counters, pivot state,
  payload length, and payload bytes and require rejection.
- Verify exclusive locks reject overlapping live owners.

### Gate O6: cancellation

- Request cancellation during CPU preparation, CUDA execution, download,
  exact verification, and checkpoint publication.
- Require no partial block adoption and no final report.
- Resume from the last checkpoint and require byte-identical final proof
  payloads relative to an uninterrupted run.

### Gate O7: fail-fast taxonomy

- Trigger one fixture for every error class.
- Require the correct class, stable code, severity, retryability, phase, and
  checkpoint reference.
- Verify fatal capacity and arithmetic flags stop new scheduling.
- Verify exact mathematical witnesses are CPU-replayed before gate closure.

### Gate O8: report-last publication

- Interrupt every publication step and verify that a report exists only when
  its complete immutable inventory is present and valid.
- Mutate or delete each referenced payload and require report adoption failure.
- Place a plausible uncommitted report in a run root and require rejection.

### Gate O9: cross-track consistency

- Require the four-form hash in normalization to equal the accepted
  identification output hash.
- Require the physical F hash in both descent tracks to equal the normalized F
  output hash.
- Require target-gauge quotient basis hashes to match the basis joined by the
  local-Lorentz descent.
- Mutate every cross-track link independently and require downstream adoption
  failure.

## 17. Adoption decision procedure

The shared reader follows this total decision procedure:

```text
read manifest with strict schema
  -> invalid: reject
verify manifest and semantic hashes
  -> invalid: reject
acquire exclusive run or import lock
  -> unavailable: report live owner, do not race
if report exists:
  verify report-last inventory and every payload
    -> valid and scope acceptable: adopt completed result
    -> invalid: quarantine by new external record, do not edit source root
else find highest complete checkpoint generation:
  verify identity, chain, payloads, counters, boundary, and track state
    -> valid: adopt exact prefix and resume at next_block_ordinal
    -> none valid: begin fresh run root
on any semantic mismatch:
  reject, never partially reuse
```

Adoption MUST return a machine-readable decision:

```json
{
  "schema_version": "adynkra-11d-gpu-proof-adoption-v1",
  "decision": "adopt_completed",
  "reason_code": "validated_report",
  "source_run_id": "<string>",
  "source_manifest_sha256": "<64 hex>",
  "source_report_sha256": "<64 hex or null>",
  "checkpoint_generation": null,
  "next_block_ordinal": null,
  "validated_payload_count": 12,
  "validated_payload_bytes": 123456,
  "adoption_record_sha256": "<64 hex>"
}
```

`decision` is one of `adopt_completed`, `resume_checkpoint`, `start_fresh`, or
`reject`. `start_fresh` always uses a new run root. It never clears the old
root.

## 18. Minimum implementation sequence

1. Implement strict shared Rust structs with `deny_unknown_fields` and
   canonical hash functions.
2. Implement immutable manifest creation and validation.
3. Extract the existing five-second monitor into a track-neutral progress
   object with shared monotone counters and resource sampling.
4. Implement block records, SHA chains, and deterministic witness capture.
5. Implement atomic checkpoint envelopes, locks, and adoption validation by
   adapting the proven second-momentum patterns.
6. Implement cooperative cancellation and the typed error taxonomy.
7. Implement immutable payload inventory and report-last publication.
8. Add track adapters without weakening shared field requirements.
9. Run gates O0 through O9 before any track publishes an authoritative report.

## 19. Definition of done

The observability contract is implemented only when:

1. all four tracks emit the same versioned shared objects;
2. five-second heartbeats continue while compute threads are blocked;
3. phase, work, rates, ETA, per-block counters, RSS, VRAM, and high-water values
   are visible without opening proof payloads;
4. the minimum canonical first witness is captured and exactly replayed;
5. cancellation preserves only complete atomic checkpoints;
6. checkpoint adoption proves exact semantic and byte identity;
7. every fatal condition has a typed error and fail-fast behavior;
8. a final report can be adopted only after its complete payload inventory
   validates;
9. cross-track source, basis, and map hashes form an unbroken chain; and
10. mutation tests demonstrate that no stale, partial, reordered, truncated,
    or semantically incompatible work can be mistaken for proof.
