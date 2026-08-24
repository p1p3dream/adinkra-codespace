# Second-Momentum GPU Fleet Operations

## Inventory

The canonical manifest contains 36 jobs: 12 legal source-copy groups for each
of three primes.

Per prime:

- 3 width-3 groups
- 6 width-2 groups
- 3 singleton fallbacks

Generate and optionally persist the manifest:

```bash
target/release/adinkra-codespace \
  adynkra-11d-second-momentum-gpu-fx-plan results/second_momentum_gpu_fx
```

## Portable job lists

Workers accept one comma-separated positional job list. Supported selectors:

- `all@0`: every group at prime index 0
- `20001@0`: all `(20001)` groups at prime index 0
- `30001@0`: all `(30001)` groups at prime index 0
- `30001-g1-p0,30001-g3-p0`: explicit groups

Completed jobs are validated and skipped. A nonblocking process-lifetime lock
rejects overlapping live ownership, so the same list may be safely retried.

Example two-machine split:

```bash
scripts/run_second_momentum_gpu_worker.sh 20001@0 /data/second-momentum 0
scripts/run_second_momentum_gpu_worker.sh 30001@0 /data/second-momentum 0
```

Example handoff of unfinished work:

```bash
scripts/run_second_momentum_gpu_worker.sh \
  30001-g3-p0,30001-g4-p0,30001-g6-p0,30001-g7-p0 \
  /data/second-momentum 0
```

Workers may use a shared output root or separate roots. To consolidate separate
roots, rsync each remote root into a separate local staging directory, then use
the verified importer:

```bash
target/release/adinkra-codespace \
  adynkra-11d-second-momentum-gpu-fx-import \
  30001@0 /staging/b300-run /data/second-momentum
```

Job commit records use contained relative artifact paths, so copied completed
jobs validate on the receiving machine. The importer verifies the source
manifest, every binary digest, every portable job identity, and publishes the
job commit record last. Conflicting bytes fail rather than overwrite.

## Live status

```bash
target/release/adinkra-codespace \
  adynkra-11d-second-momentum-gpu-fx-status all@0 /data/second-momentum
```

The status output reports each selected job, hostname, PID, heartbeat age,
phase, word progress, raw terms per lane, union-key counts, global batch
ordinal, per-stage CUDA timing, throughput, RSS, GPU utilization and memory,
configured memory caps, device high-water, checkpoint generation and paths.
It also emits `pending_job_list` and `stale_job_list` as paste-ready selectors
for another worker.

Each job writes:

```text
jobs/<job-id>/status.json
jobs/<job-id>/events.jsonl
jobs/<job-id>/checkpoint.json
jobs/<job-id>/job-report.json
```

Heartbeats and status snapshots are updated every five seconds. The supervisor
repairs a still-running status after SIGKILL or OOM. Width-2 and width-3 jobs
resume from their last atomically published word checkpoint. Singleton jobs use
the validated single-column fallback and are intentionally nonresumable.

## Memory and batch controls

```text
ADYNKRA_GPU_GROUP_RAW_BATCH_TERMS
ADYNKRA_GPU_GROUP_UNION_KEYS
ADYNKRA_GPU_GROUP_HOST_CAP_BYTES
ADYNKRA_GPU_GROUP_DEVICE_CAP_BYTES
ADYNKRA_GPU_GROUP_CONTRACTION_CAP_BYTES
ADYNKRA_GPU_GROUP_LANE_HOST_CAP_BYTES
ADYNKRA_GPU_GROUP_DOWNLOAD_TERMS
```

Defaults are encoded in the job report. Capacity checks are fail-closed, and
the declared aggregate device budget includes the contraction context,
persistent lane contexts, and mandatory CUDA headroom.

## Publication boundary

Individual binary artifacts are written and verified first. Individual column
reports follow. `job-report.json` is published last and is the commit record.
Consumers must adopt only jobs whose commit record and artifact digests validate.
