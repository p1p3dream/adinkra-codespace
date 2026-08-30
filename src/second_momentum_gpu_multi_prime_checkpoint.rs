//! Durable whole-word checkpoints for grouped multi-prime GPU execution.
//!
//! The exact source traversal is shared by every active prime. A checkpoint
//! therefore stores one prime-independent transcript and one row/timing state
//! per prime. The complete payload is replaced atomically only after every
//! prime has folded every exact union batch for a PBW word.

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::eleven_dimensional_second_momentum_gpu::{
    FUNCTIONAL_ROW_COUNT, GPU_FX_PRIMES, GPU_FX_SCHEMA, GaussianResidue,
};
use crate::second_momentum_gpu_group::MAX_PRODUCTION_GROUP_WIDTH;

pub(crate) const MULTI_PRIME_CHECKPOINT_SCHEMA: &str =
    "adynkra-11d-second-momentum-gpu-multi-prime-checkpoint-v1";

const SOURCE_HASH_DOMAIN: &[u8] = b"\0streamed-source-terms-v1\0";
const PACKED_HASH_DOMAIN: &[u8] = b"\0streamed-packed-terms-v1\0";
const SOURCE_RECORD_BYTES: u64 = 23;
const PACKED_RECORD_BYTES: u64 = 24;
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Serializable state compatible with the grouped runner's incremental
/// SHA-256 implementation. The runner can copy these three fields without
/// exposing its compression implementation to this persistence module.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializableSha256CheckpointState {
    pub state: [u32; 8],
    pub total_bytes: u64,
    pub buffer: Vec<u8>,
}

impl SerializableSha256CheckpointState {
    pub(crate) fn validate_for_total_bytes(&self, expected_total: u64) -> Result<(), String> {
        if self.total_bytes != expected_total
            || self.buffer.len() >= 64
            || self.buffer.len() as u64 != self.total_bytes % 64
        {
            return Err("serialized SHA-256 checkpoint state is invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MultiPrimeRuntimeIdentity {
    pub prime_index: usize,
    pub prime: u32,
    pub single_job_id: String,
    pub group_id: String,
    pub plan_sha256: String,
    pub static_semantic_sha256: String,
    pub flat_plan_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SharedLaneLoweringTotals {
    pub enabled: bool,
    pub roots_lowered: u64,
    pub input_entry_visits: u64,
    pub expanded_entry_visits: u64,
    pub output_entry_visits: u64,
    pub gpu_milliseconds: f64,
    pub scratch_high_water_bytes: u64,
    pub peak_immutable_handle_bytes: u64,
    pub maximum_absolute_coefficient: u64,
    pub device_hard_cap_bytes: u64,
    pub download_chunk_terms: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SharedUnionTotals {
    pub union_batches: u64,
    pub union_milliseconds: u128,
    pub union_keys: u64,
    pub peak_union_keys: usize,
    pub reduced_key_visits_per_column: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SharedExactStreamState {
    pub raw_terms_per_column: Vec<u64>,
    pub source_hashers: Vec<SerializableSha256CheckpointState>,
    pub packed_hashers: Vec<SerializableSha256CheckpointState>,
    pub union: SharedUnionTotals,
    pub lowering_per_column: Vec<SharedLaneLoweringTotals>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PrimeCudaTimingTotals {
    pub upload_milliseconds: f64,
    pub contract_milliseconds: f64,
    pub finalize_milliseconds: f64,
    pub download_milliseconds: f64,
    pub total_cuda_milliseconds: f64,
    pub expanded_contributions_per_column: Vec<u64>,
    pub nonzero_reduced_term_visits_per_column: Vec<u64>,
    pub device_high_water_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PrimeCheckpointState {
    pub prime_index: usize,
    pub batches_folded: u64,
    /// Lane-major rows: `[lane][functional_row]`.
    pub rows: Vec<Vec<GaussianResidue>>,
    pub timing: PrimeCudaTimingTotals,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MultiPrimeCheckpoint {
    pub schema_version: String,
    pub bundle_id: String,
    pub run_schema_version: String,
    pub work_manifest_sha256: String,
    pub tranche: String,
    pub group_index: usize,
    pub source_group_sha256: String,
    pub multi_prime_group_sha256: String,
    pub ordered_local_ordinals: Vec<usize>,
    pub ordered_global_ordinals: Vec<usize>,
    pub ordered_source_copies: Vec<usize>,
    pub pbw_word_count: usize,
    pub active_columns: usize,
    pub prime_runtimes: Vec<MultiPrimeRuntimeIdentity>,
    pub next_word_ordinal: usize,
    pub next_global_batch_ordinal: u64,
    pub checkpoint_generation: u64,
    pub shared: SharedExactStreamState,
    pub primes: Vec<PrimeCheckpointState>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct MultiPrimeCheckpointWriteEnvelope<'a> {
    schema_version: &'a str,
    payload_sha256: &'a str,
    checkpoint: &'a RawValue,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MultiPrimeCheckpointReadEnvelope {
    schema_version: String,
    payload_sha256: String,
    checkpoint: Box<RawValue>,
}

pub(crate) fn write_multi_prime_checkpoint(
    path: &Path,
    checkpoint: &MultiPrimeCheckpoint,
) -> Result<String, String> {
    validate_multi_prime_checkpoint(checkpoint)?;
    let payload = serde_json::to_string(checkpoint).map_err(|error| error.to_string())?;
    let payload_sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
    let raw_payload = RawValue::from_string(payload).map_err(|error| error.to_string())?;
    let envelope = MultiPrimeCheckpointWriteEnvelope {
        schema_version: MULTI_PRIME_CHECKPOINT_SCHEMA,
        payload_sha256: &payload_sha256,
        checkpoint: raw_payload.as_ref(),
    };
    let mut bytes = serde_json::to_vec(&envelope).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    write_atomic_durable(path, &bytes)?;
    Ok(payload_sha256)
}

pub(crate) fn read_multi_prime_checkpoint(path: &Path) -> Result<MultiPrimeCheckpoint, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let envelope: MultiPrimeCheckpointReadEnvelope =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if envelope.schema_version != MULTI_PRIME_CHECKPOINT_SCHEMA {
        return Err("unsupported multi-prime checkpoint envelope schema".to_string());
    }
    validate_sha256("payload", &envelope.payload_sha256)?;
    let observed = format!("{:x}", Sha256::digest(envelope.checkpoint.get().as_bytes()));
    if envelope.payload_sha256 != observed {
        return Err(format!(
            "multi-prime checkpoint payload digest mismatch: stored {}, observed {}",
            envelope.payload_sha256, observed
        ));
    }
    let checkpoint: MultiPrimeCheckpoint =
        serde_json::from_str(envelope.checkpoint.get()).map_err(|error| error.to_string())?;
    validate_multi_prime_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}

pub(crate) fn validate_multi_prime_checkpoint(
    checkpoint: &MultiPrimeCheckpoint,
) -> Result<(), String> {
    if checkpoint.schema_version != MULTI_PRIME_CHECKPOINT_SCHEMA
        || checkpoint.bundle_id.is_empty()
        || checkpoint.run_schema_version.is_empty()
        || checkpoint.tranche.is_empty()
        || checkpoint.active_columns == 0
        || checkpoint.active_columns > MAX_PRODUCTION_GROUP_WIDTH
        || checkpoint.active_columns != checkpoint.ordered_local_ordinals.len()
        || checkpoint.active_columns != checkpoint.ordered_global_ordinals.len()
        || checkpoint.active_columns != checkpoint.ordered_source_copies.len()
        || checkpoint.pbw_word_count == 0
        || checkpoint.next_word_ordinal > checkpoint.pbw_word_count
    {
        return Err("multi-prime checkpoint identity or word range is invalid".to_string());
    }
    validate_sha256("work manifest", &checkpoint.work_manifest_sha256)?;
    validate_sha256("source group", &checkpoint.source_group_sha256)?;
    validate_sha256("multi-prime group", &checkpoint.multi_prime_group_sha256)?;
    validate_strictly_increasing("local ordinals", &checkpoint.ordered_local_ordinals)?;
    validate_strictly_increasing("global ordinals", &checkpoint.ordered_global_ordinals)?;
    validate_strictly_increasing("source copies", &checkpoint.ordered_source_copies)?;

    let expected_generation = u64::try_from(checkpoint.next_word_ordinal)
        .map_err(|_| "next word ordinal does not fit u64".to_string())?;
    if checkpoint.checkpoint_generation != expected_generation {
        return Err("checkpoint generation is not the committed word boundary".to_string());
    }

    validate_prime_runtimes(&checkpoint.prime_runtimes)?;
    if checkpoint.primes.len() != checkpoint.prime_runtimes.len() {
        return Err("multi-prime runtime and row-state counts differ".to_string());
    }
    validate_shared_state(checkpoint)?;

    for (runtime, prime_state) in checkpoint.prime_runtimes.iter().zip(&checkpoint.primes) {
        if prime_state.prime_index != runtime.prime_index
            || prime_state.batches_folded != checkpoint.next_global_batch_ordinal
            || prime_state.rows.len() != checkpoint.active_columns
            || prime_state
                .rows
                .iter()
                .any(|rows| rows.len() != FUNCTIONAL_ROW_COUNT)
            || prime_state
                .rows
                .iter()
                .flatten()
                .any(|residue| residue.real >= runtime.prime || residue.imaginary >= runtime.prime)
            || prime_state.timing.expanded_contributions_per_column.len()
                != checkpoint.active_columns
            || prime_state
                .timing
                .nonzero_reduced_term_visits_per_column
                .len()
                != checkpoint.active_columns
        {
            return Err("multi-prime row or timing state is torn or malformed".to_string());
        }
        validate_nonnegative_finite(
            "prime CUDA timing",
            &[
                prime_state.timing.upload_milliseconds,
                prime_state.timing.contract_milliseconds,
                prime_state.timing.finalize_milliseconds,
                prime_state.timing.download_milliseconds,
                prime_state.timing.total_cuda_milliseconds,
            ],
        )?;
    }
    Ok(())
}

fn validate_shared_state(checkpoint: &MultiPrimeCheckpoint) -> Result<(), String> {
    let shared = &checkpoint.shared;
    let width = checkpoint.active_columns;
    if shared.raw_terms_per_column.len() != width
        || shared.source_hashers.len() != width
        || shared.packed_hashers.len() != width
        || shared.lowering_per_column.len() != width
        || shared.union.reduced_key_visits_per_column.len() != width
        || shared.union.union_batches != checkpoint.next_global_batch_ordinal
    {
        return Err("shared exact stream state is torn or malformed".to_string());
    }
    let source_prefix = checked_prefix_length(SOURCE_HASH_DOMAIN)?;
    let packed_prefix = checked_prefix_length(PACKED_HASH_DOMAIN)?;
    for lane in 0..width {
        let source_bytes = shared.raw_terms_per_column[lane]
            .checked_mul(SOURCE_RECORD_BYTES)
            .and_then(|bytes| bytes.checked_add(source_prefix))
            .ok_or_else(|| "source transcript byte count overflow".to_string())?;
        let packed_bytes = shared.raw_terms_per_column[lane]
            .checked_mul(PACKED_RECORD_BYTES)
            .and_then(|bytes| bytes.checked_add(packed_prefix))
            .ok_or_else(|| "packed transcript byte count overflow".to_string())?;
        shared.source_hashers[lane].validate_for_total_bytes(source_bytes)?;
        shared.packed_hashers[lane].validate_for_total_bytes(packed_bytes)?;
        if shared.lowering_per_column[lane].download_chunk_terms == 0 {
            return Err("shared lowering download chunk is zero".to_string());
        }
        validate_nonnegative_finite(
            "shared lowering timing",
            &[shared.lowering_per_column[lane].gpu_milliseconds],
        )?;
    }
    Ok(())
}

fn validate_prime_runtimes(runtimes: &[MultiPrimeRuntimeIdentity]) -> Result<(), String> {
    if runtimes.is_empty() || runtimes.len() > GPU_FX_PRIMES.len() {
        return Err("multi-prime runtime list is empty or too large".to_string());
    }
    let mut previous_index = None;
    for runtime in runtimes {
        if runtime.prime_index >= GPU_FX_PRIMES.len()
            || previous_index.is_some_and(|previous| previous >= runtime.prime_index)
            || runtime.prime != GPU_FX_PRIMES[runtime.prime_index]
            || runtime.single_job_id.is_empty()
        {
            return Err("multi-prime runtime order or pinned prime is invalid".to_string());
        }
        for (label, digest) in [
            ("prime group", runtime.group_id.as_str()),
            ("prime plan", runtime.plan_sha256.as_str()),
            ("static semantic", runtime.static_semantic_sha256.as_str()),
            ("flat plan", runtime.flat_plan_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        previous_index = Some(runtime.prime_index);
    }
    Ok(())
}

fn validate_strictly_increasing(label: &str, values: &[usize]) -> Result<(), String> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "multi-prime checkpoint {label} are not strictly increasing"
        ));
    }
    Ok(())
}

fn validate_sha256(label: &str, digest: &str) -> Result<(), String> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "multi-prime checkpoint {label} SHA-256 is malformed"
        ));
    }
    Ok(())
}

fn validate_nonnegative_finite(label: &str, values: &[f64]) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!("{label} is negative or non-finite"));
    }
    Ok(())
}

fn checked_prefix_length(domain: &[u8]) -> Result<u64, String> {
    u64::try_from(GPU_FX_SCHEMA.len())
        .ok()
        .and_then(|schema| {
            u64::try_from(domain.len())
                .ok()
                .and_then(|domain| schema.checked_add(domain))
        })
        .ok_or_else(|| "transcript prefix byte count overflow".to_string())
}

fn write_atomic_durable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "checkpoint path has no UTF-8 file name".to_string())?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{name}.{}.{}.{}.tmp",
        std::process::id(),
        nanos,
        sequence
    ));
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        fs::rename(&temporary, path).map_err(|error| error.to_string())?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> String {
        format!("{value:02x}").repeat(32)
    }

    fn sha_state(total_bytes: u64) -> SerializableSha256CheckpointState {
        SerializableSha256CheckpointState {
            state: [0x1234_5678; 8],
            total_bytes,
            buffer: vec![7; (total_bytes % 64) as usize],
        }
    }

    fn timing(width: usize) -> PrimeCudaTimingTotals {
        PrimeCudaTimingTotals {
            upload_milliseconds: 1.0,
            contract_milliseconds: 2.0,
            finalize_milliseconds: 3.0,
            download_milliseconds: 4.0,
            total_cuda_milliseconds: 10.0,
            expanded_contributions_per_column: vec![11; width],
            nonzero_reduced_term_visits_per_column: vec![9; width],
            device_high_water_bytes: 4096,
        }
    }

    fn checkpoint() -> MultiPrimeCheckpoint {
        let width = 2;
        let counts = [4_u64, 5_u64];
        let source_prefix = checked_prefix_length(SOURCE_HASH_DOMAIN).unwrap();
        let packed_prefix = checked_prefix_length(PACKED_HASH_DOMAIN).unwrap();
        let prime_runtimes = [0_usize, 2_usize]
            .into_iter()
            .map(|prime_index| MultiPrimeRuntimeIdentity {
                prime_index,
                prime: GPU_FX_PRIMES[prime_index],
                single_job_id: format!("30001-g7-p{prime_index}"),
                group_id: digest(10 + prime_index as u8),
                plan_sha256: digest(20 + prime_index as u8),
                static_semantic_sha256: digest(30 + prime_index as u8),
                flat_plan_sha256: digest(40 + prime_index as u8),
            })
            .collect::<Vec<_>>();
        let primes = prime_runtimes
            .iter()
            .map(|runtime| PrimeCheckpointState {
                prime_index: runtime.prime_index,
                batches_folded: 7,
                rows: vec![vec![GaussianResidue::zero(); FUNCTIONAL_ROW_COUNT]; width],
                timing: timing(width),
            })
            .collect();
        MultiPrimeCheckpoint {
            schema_version: MULTI_PRIME_CHECKPOINT_SCHEMA.to_string(),
            bundle_id: "30001-g7-mp02".to_string(),
            run_schema_version: "adynkra-11d-second-momentum-gpu-group-run-v1".to_string(),
            work_manifest_sha256: digest(1),
            tranche: "30001".to_string(),
            group_index: 7,
            source_group_sha256: digest(2),
            multi_prime_group_sha256: digest(3),
            ordered_local_ordinals: vec![13, 14],
            ordered_global_ordinals: vec![75, 76],
            ordered_source_copies: vec![14, 15],
            pbw_word_count: 483,
            active_columns: width,
            prime_runtimes,
            next_word_ordinal: 1,
            next_global_batch_ordinal: 7,
            checkpoint_generation: 1,
            shared: SharedExactStreamState {
                raw_terms_per_column: counts.to_vec(),
                source_hashers: counts
                    .iter()
                    .map(|count| sha_state(source_prefix + SOURCE_RECORD_BYTES * count))
                    .collect(),
                packed_hashers: counts
                    .iter()
                    .map(|count| sha_state(packed_prefix + PACKED_RECORD_BYTES * count))
                    .collect(),
                union: SharedUnionTotals {
                    union_batches: 7,
                    union_milliseconds: 2,
                    union_keys: 100,
                    peak_union_keys: 20,
                    reduced_key_visits_per_column: vec![30, 40],
                },
                lowering_per_column: vec![
                    SharedLaneLoweringTotals {
                        enabled: true,
                        roots_lowered: 5,
                        input_entry_visits: 10,
                        expanded_entry_visits: 20,
                        output_entry_visits: 15,
                        gpu_milliseconds: 1.5,
                        scratch_high_water_bytes: 1024,
                        peak_immutable_handle_bytes: 512,
                        maximum_absolute_coefficient: 7,
                        device_hard_cap_bytes: 1 << 30,
                        download_chunk_terms: 1024,
                    };
                    width
                ],
            },
            primes,
        }
    }

    fn temporary_directory(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "adynkra-multi-prime-checkpoint-{label}-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn checkpoint_round_trip_hashes_exact_embedded_payload_bytes() {
        let directory = temporary_directory("round-trip");
        let path = directory.join("checkpoint.json");
        let expected = checkpoint();
        let digest = write_multi_prime_checkpoint(&path, &expected).unwrap();
        let bytes = fs::read(&path).unwrap();
        let envelope: MultiPrimeCheckpointReadEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            digest,
            format!("{:x}", Sha256::digest(envelope.checkpoint.get().as_bytes()))
        );
        assert_eq!(read_multi_prime_checkpoint(&path).unwrap(), expected);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_accepts_a_digest_over_a_legacy_float_spelling() {
        let directory = temporary_directory("float-spelling");
        let path = directory.join("checkpoint.json");
        let expected = checkpoint();
        write_multi_prime_checkpoint(&path, &expected).unwrap();
        let canonical = fs::read_to_string(&path).unwrap();
        let legacy = canonical.replacen(
            "\"upload_milliseconds\":1.0",
            "\"upload_milliseconds\":1.00000000000000000",
            1,
        );
        assert_ne!(legacy, canonical);
        let envelope: MultiPrimeCheckpointReadEnvelope = serde_json::from_str(&legacy).unwrap();
        let legacy_digest = format!("{:x}", Sha256::digest(envelope.checkpoint.get().as_bytes()));
        let legacy = legacy.replacen(
            &format!("\"payload_sha256\":\"{}\"", envelope.payload_sha256),
            &format!("\"payload_sha256\":\"{legacy_digest}\""),
            1,
        );
        fs::write(&path, legacy).unwrap();
        assert_eq!(read_multi_prime_checkpoint(&path).unwrap(), expected);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_rejects_payload_corruption_and_noncanonical_prime_order() {
        let directory = temporary_directory("corruption");
        let path = directory.join("checkpoint.json");
        let expected = checkpoint();
        write_multi_prime_checkpoint(&path, &expected).unwrap();
        let original = fs::read_to_string(&path).unwrap();
        let corrupted = original.replacen("\"union_keys\":100", "\"union_keys\":101", 1);
        assert_ne!(corrupted, original);
        fs::write(&path, corrupted).unwrap();
        assert!(
            read_multi_prime_checkpoint(&path)
                .unwrap_err()
                .contains("payload digest mismatch")
        );

        let mut reordered = expected;
        reordered.prime_runtimes.swap(0, 1);
        reordered.primes.swap(0, 1);
        assert!(validate_multi_prime_checkpoint(&reordered).is_err());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_rejects_torn_word_prime_and_hash_boundaries() {
        let mut torn_batches = checkpoint();
        torn_batches.primes[1].batches_folded -= 1;
        assert!(validate_multi_prime_checkpoint(&torn_batches).is_err());

        let mut torn_word = checkpoint();
        torn_word.checkpoint_generation = 0;
        assert!(validate_multi_prime_checkpoint(&torn_word).is_err());

        let mut torn_hash = checkpoint();
        torn_hash.shared.source_hashers[0].total_bytes += SOURCE_RECORD_BYTES;
        assert!(validate_multi_prime_checkpoint(&torn_hash).is_err());

        let mut bad_residue = checkpoint();
        let prime = bad_residue.prime_runtimes[0].prime;
        bad_residue.primes[0].rows[0][0].real = prime;
        assert!(validate_multi_prime_checkpoint(&bad_residue).is_err());
    }

    #[test]
    fn interrupted_or_invalid_write_preserves_last_whole_word() {
        let directory = temporary_directory("atomic-boundary");
        let path = directory.join("checkpoint.json");
        let committed = checkpoint();
        write_multi_prime_checkpoint(&path, &committed).unwrap();

        let abandoned = directory.join(".checkpoint.json.interrupted.tmp");
        fs::write(&abandoned, b"{\"partial\":true").unwrap();
        assert_eq!(read_multi_prime_checkpoint(&path).unwrap(), committed);

        let mut invalid_next = committed.clone();
        invalid_next.next_word_ordinal = 2;
        invalid_next.checkpoint_generation = 2;
        invalid_next.next_global_batch_ordinal = 8;
        invalid_next.shared.union.union_batches = 8;
        invalid_next.primes[0].batches_folded = 8;
        assert!(write_multi_prime_checkpoint(&path, &invalid_next).is_err());
        assert_eq!(read_multi_prime_checkpoint(&path).unwrap(), committed);

        let mut next = committed.clone();
        next.next_word_ordinal = 2;
        next.checkpoint_generation = 2;
        write_multi_prime_checkpoint(&path, &next).unwrap();
        assert_eq!(read_multi_prime_checkpoint(&path).unwrap(), next);
        let _ = fs::remove_dir_all(directory);
    }
}
