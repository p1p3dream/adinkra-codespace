//! Distributed production jobs for grouped second-momentum CUDA columns.
//!
//! The manifest is deterministic and machine independent. Workers accept a
//! compact list of job IDs, publish one commit record per job, and can be
//! reassigned freely because completed reports are validated before adoption.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::checkpointable_sha256::CheckpointableSha256 as SerializableSha256;
use crate::eleven_dimensional_second_momentum_gpu::{GPU_FX_PRIMES, GaussianResidue};
use crate::second_momentum_gpu_group::{GpuFxTranche, discover_legal_cuda_column_groups};

pub(crate) const GPU_GROUP_JOB_SCHEMA: &str = "adynkra-11d-second-momentum-gpu-group-jobs-v1";
pub(crate) const GPU_GROUP_CHECKPOINT_SCHEMA: &str =
    "adynkra-11d-second-momentum-gpu-group-checkpoint-v1";
pub(crate) const GPU_GROUP_RUN_SCHEMA: &str = "adynkra-11d-second-momentum-gpu-group-run-v1";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GpuGroupJobKey {
    pub tranche: String,
    pub group_index: usize,
    pub prime_index: usize,
}

impl GpuGroupJobKey {
    pub(crate) fn new(
        tranche: GpuFxTranche,
        group_index: usize,
        prime_index: usize,
    ) -> Result<Self, String> {
        let groups = discover_legal_cuda_column_groups(tranche);
        if group_index >= groups.len() || prime_index >= GPU_FX_PRIMES.len() {
            return Err("GPU group job index is out of range".to_string());
        }
        Ok(Self {
            tranche: tranche.as_str().to_string(),
            group_index,
            prime_index,
        })
    }

    pub(crate) fn tranche(&self) -> Result<GpuFxTranche, String> {
        GpuFxTranche::parse(&self.tranche)
    }

    pub(crate) fn prime(&self) -> Result<u32, String> {
        GPU_FX_PRIMES
            .get(self.prime_index)
            .copied()
            .ok_or_else(|| "GPU group job prime index is out of range".to_string())
    }

    pub(crate) fn local_ordinals(&self) -> Result<Vec<usize>, String> {
        discover_legal_cuda_column_groups(self.tranche()?)
            .get(self.group_index)
            .cloned()
            .ok_or_else(|| "GPU group job group index is out of range".to_string())
    }

    pub(crate) fn id(&self) -> String {
        format!(
            "{}-g{}-p{}",
            self.tranche, self.group_index, self.prime_index
        )
    }

    pub(crate) fn parse_id(value: &str) -> Result<Self, String> {
        let mut fields = value.split('-');
        let tranche = fields
            .next()
            .ok_or_else(|| "job ID has no tranche".to_string())?;
        let group = fields
            .next()
            .and_then(|field| field.strip_prefix('g'))
            .ok_or_else(|| "job ID has no g<index> field".to_string())?;
        let prime = fields
            .next()
            .and_then(|field| field.strip_prefix('p'))
            .ok_or_else(|| "job ID has no p<index> field".to_string())?;
        if fields.next().is_some() {
            return Err("job ID has extra fields".to_string());
        }
        let tranche = GpuFxTranche::parse(tranche)?;
        let group_index = group
            .parse::<usize>()
            .map_err(|_| "job group index is not an integer".to_string())?;
        let prime_index = prime
            .parse::<usize>()
            .map_err(|_| "job prime index is not an integer".to_string())?;
        Self::new(tranche, group_index, prime_index)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GpuGroupJobManifestEntry {
    pub job_id: String,
    pub tranche: String,
    pub group_index: usize,
    pub prime_index: usize,
    pub prime: u32,
    pub local_ordinals: Vec<usize>,
    pub global_ordinals: Vec<usize>,
    pub width: usize,
    pub singleton_fallback: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GpuGroupJobManifest {
    pub schema_version: String,
    pub jobs: Vec<GpuGroupJobManifestEntry>,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishedColumnArtifact {
    global_ordinal: usize,
    binary_relative_path: String,
    report_relative_path: String,
    binary_sha256: String,
    column_semantic_sha256: String,
}

pub(crate) fn build_job_manifest() -> Result<GpuGroupJobManifest, String> {
    let mut jobs = Vec::new();
    for tranche in [GpuFxTranche::Two0001, GpuFxTranche::Three0001] {
        let first_global = match tranche {
            GpuFxTranche::Two0001 => 53,
            GpuFxTranche::Three0001 => 62,
        };
        for (group_index, local_ordinals) in discover_legal_cuda_column_groups(tranche)
            .into_iter()
            .enumerate()
        {
            for prime_index in 0..GPU_FX_PRIMES.len() {
                let key = GpuGroupJobKey::new(tranche, group_index, prime_index)?;
                jobs.push(GpuGroupJobManifestEntry {
                    job_id: key.id(),
                    tranche: key.tranche,
                    group_index,
                    prime_index,
                    prime: GPU_FX_PRIMES[prime_index],
                    global_ordinals: local_ordinals
                        .iter()
                        .map(|ordinal| first_global + ordinal)
                        .collect(),
                    width: local_ordinals.len(),
                    singleton_fallback: local_ordinals.len() == 1,
                    local_ordinals: local_ordinals.clone(),
                });
            }
        }
    }
    let mut manifest = GpuGroupJobManifest {
        schema_version: GPU_GROUP_JOB_SCHEMA.to_string(),
        jobs,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn manifest_digest(manifest: &GpuGroupJobManifest) -> Result<String, String> {
    let mut copy = manifest.clone();
    copy.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&copy).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(crate) fn validate_manifest(manifest: &GpuGroupJobManifest) -> Result<(), String> {
    if manifest.schema_version != GPU_GROUP_JOB_SCHEMA
        || manifest.manifest_sha256 != manifest_digest(manifest)?
    {
        return Err("GPU group job manifest schema or digest is invalid".to_string());
    }
    let expected = build_manifest_entries_without_validation()?;
    if manifest.jobs != expected {
        return Err("GPU group job manifest does not equal the canonical inventory".to_string());
    }
    Ok(())
}

fn build_manifest_entries_without_validation() -> Result<Vec<GpuGroupJobManifestEntry>, String> {
    let mut entries = Vec::new();
    for tranche in [GpuFxTranche::Two0001, GpuFxTranche::Three0001] {
        let first_global = match tranche {
            GpuFxTranche::Two0001 => 53,
            GpuFxTranche::Three0001 => 62,
        };
        for (group_index, locals) in discover_legal_cuda_column_groups(tranche)
            .into_iter()
            .enumerate()
        {
            for prime_index in 0..GPU_FX_PRIMES.len() {
                let key = GpuGroupJobKey::new(tranche, group_index, prime_index)?;
                entries.push(GpuGroupJobManifestEntry {
                    job_id: key.id(),
                    tranche: key.tranche,
                    group_index,
                    prime_index,
                    prime: GPU_FX_PRIMES[prime_index],
                    global_ordinals: locals
                        .iter()
                        .map(|ordinal| first_global + ordinal)
                        .collect(),
                    width: locals.len(),
                    singleton_fallback: locals.len() == 1,
                    local_ordinals: locals.clone(),
                });
            }
        }
    }
    Ok(entries)
}

/// Parse canonical job IDs and convenient inventory selectors.
///
/// Supported selectors are `all@P`, `20001@P`, `30001@P`, and canonical
/// individual IDs such as `30001-g7-p0`. Commas separate selectors.
pub(crate) fn parse_job_list(value: &str) -> Result<Vec<GpuGroupJobKey>, String> {
    if value.trim().is_empty() {
        return Err("GPU group job list is empty".to_string());
    }
    let manifest = build_job_manifest()?;
    let mut selected = BTreeSet::new();
    for raw in value.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            return Err("GPU group job list contains an empty selector".to_string());
        }
        if let Some((scope, prime)) = token.split_once('@') {
            let prime_index = prime
                .parse::<usize>()
                .map_err(|_| format!("invalid prime index in selector {token}"))?;
            if prime_index >= GPU_FX_PRIMES.len() {
                return Err(format!(
                    "prime index in selector {token} must be 0, 1, or 2"
                ));
            }
            if !matches!(scope, "all" | "20001" | "30001") {
                return Err(format!("invalid tranche selector {scope}"));
            }
            for entry in &manifest.jobs {
                if entry.prime_index == prime_index && (scope == "all" || entry.tranche == scope) {
                    selected.insert(GpuGroupJobKey::new(
                        GpuFxTranche::parse(&entry.tranche)?,
                        entry.group_index,
                        entry.prime_index,
                    )?);
                }
            }
        } else {
            selected.insert(GpuGroupJobKey::parse_id(token)?);
        }
    }
    if selected.is_empty() {
        return Err("GPU group job selector matched no jobs".to_string());
    }
    Ok(selected.into_iter().collect())
}

pub(crate) fn write_or_validate_manifest(output_directory: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(output_directory).map_err(|error| error.to_string())?;
    let path = output_directory.join("work-manifest.json");
    let manifest = build_job_manifest()?;
    let mut bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    if path.exists() {
        let existing = fs::read(&path).map_err(|error| error.to_string())?;
        if existing != bytes {
            return Err(format!(
                "{} already contains a different work manifest",
                path.display()
            ));
        }
    } else {
        write_atomic_durable(&path, &bytes)?;
    }
    Ok(path)
}

fn serializable_checkpoint_state(
    value: &SerializableSha256,
) -> crate::second_momentum_gpu_multi_prime_checkpoint::SerializableSha256CheckpointState {
    let (state, total_bytes, buffer) = value.continuation_parts();
    crate::second_momentum_gpu_multi_prime_checkpoint::SerializableSha256CheckpointState {
        state,
        total_bytes,
        buffer,
    }
}

fn serializable_from_checkpoint_state(
    value: &crate::second_momentum_gpu_multi_prime_checkpoint::SerializableSha256CheckpointState,
) -> Result<SerializableSha256, String> {
    SerializableSha256::from_continuation_parts(
        value.state,
        value.total_bytes,
        value.buffer.clone(),
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupTimingTotals {
    union_milliseconds: u128,
    upload_milliseconds: f64,
    contract_milliseconds: f64,
    finalize_milliseconds: f64,
    download_milliseconds: f64,
    total_cuda_milliseconds: f64,
    union_batches: u64,
    union_keys: u64,
    peak_union_keys: usize,
    device_high_water_bytes: u64,
    expanded_contributions_per_column: Vec<u64>,
    reduced_key_visits_per_column: Vec<u64>,
    nonzero_reduced_term_visits_per_column: Vec<u64>,
}

impl GroupTimingTotals {
    fn new(width: usize) -> Self {
        Self {
            union_milliseconds: 0,
            upload_milliseconds: 0.0,
            contract_milliseconds: 0.0,
            finalize_milliseconds: 0.0,
            download_milliseconds: 0.0,
            total_cuda_milliseconds: 0.0,
            union_batches: 0,
            union_keys: 0,
            peak_union_keys: 0,
            device_high_water_bytes: 0,
            expanded_contributions_per_column: vec![0; width],
            reduced_key_visits_per_column: vec![0; width],
            nonzero_reduced_term_visits_per_column: vec![0; width],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupJobCheckpoint {
    schema_version: String,
    job_id: String,
    group_id: String,
    plan_sha256: String,
    next_word_ordinal: usize,
    next_global_batch_ordinal: u64,
    batches_folded: u64,
    raw_terms_per_column: Vec<u64>,
    source_hashers: Vec<SerializableSha256>,
    packed_hashers: Vec<SerializableSha256>,
    rows: Vec<Vec<GaussianResidue>>,
    timing: GroupTimingTotals,
    #[serde(default)]
    lowering_summaries:
        Option<Vec<crate::second_momentum_gpu_multi_prime_checkpoint::SharedLaneLoweringTotals>>,
    checkpoint_generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupJobCheckpointEnvelope {
    schema_version: String,
    payload_sha256: String,
    checkpoint: GroupJobCheckpoint,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGroupJobCheckpointEnvelope {
    schema_version: String,
    payload_sha256: String,
    checkpoint: Box<serde_json::value::RawValue>,
}

fn write_checkpoint(path: &Path, checkpoint: &GroupJobCheckpoint) -> Result<String, String> {
    validate_checkpoint_shape(checkpoint)?;
    let payload = serde_json::to_vec(checkpoint).map_err(|error| error.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&payload));
    let envelope = GroupJobCheckpointEnvelope {
        schema_version: GPU_GROUP_CHECKPOINT_SCHEMA.to_string(),
        payload_sha256: digest.clone(),
        checkpoint: checkpoint.clone(),
    };
    let mut bytes = serde_json::to_vec(&envelope).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    write_atomic_durable(path, &bytes)?;
    Ok(digest)
}

fn read_checkpoint(path: &Path) -> Result<GroupJobCheckpoint, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let envelope: RawGroupJobCheckpointEnvelope =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if envelope.schema_version != GPU_GROUP_CHECKPOINT_SCHEMA {
        return Err("unsupported GPU group checkpoint schema".to_string());
    }
    // Hash the exact checkpoint bytes embedded in the envelope. Parsing and
    // reserializing floating-point timing fields can change their shortest
    // decimal spelling across serde_json versions even though the f64 value is
    // identical, which made valid durable checkpoints spuriously unreadable.
    let observed_payload_sha256 =
        format!("{:x}", Sha256::digest(envelope.checkpoint.get().as_bytes()));
    if envelope.payload_sha256 != observed_payload_sha256 {
        return Err(format!(
            "GPU group checkpoint payload digest mismatch: stored {}, observed {}",
            envelope.payload_sha256, observed_payload_sha256
        ));
    }
    let checkpoint: GroupJobCheckpoint =
        serde_json::from_str(envelope.checkpoint.get()).map_err(|error| error.to_string())?;
    validate_checkpoint_shape(&checkpoint)?;
    Ok(checkpoint)
}

fn validate_checkpoint_shape(checkpoint: &GroupJobCheckpoint) -> Result<(), String> {
    let width = checkpoint.raw_terms_per_column.len();
    if checkpoint.schema_version != GPU_GROUP_CHECKPOINT_SCHEMA
        || width == 0
        || checkpoint.source_hashers.len() != width
        || checkpoint.packed_hashers.len() != width
        || checkpoint.rows.len() != width
        || checkpoint.rows.iter().any(|rows| {
            rows.len() != crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
        })
        || checkpoint.timing.expanded_contributions_per_column.len() != width
        || checkpoint.timing.reduced_key_visits_per_column.len() != width
        || checkpoint
            .timing
            .nonzero_reduced_term_visits_per_column
            .len()
            != width
        || checkpoint
            .lowering_summaries
            .as_ref()
            .is_some_and(|summaries| summaries.len() != width)
    {
        return Err("GPU group checkpoint shape is invalid".to_string());
    }
    for hasher in checkpoint
        .source_hashers
        .iter()
        .chain(&checkpoint.packed_hashers)
    {
        hasher.validate()?;
    }
    Ok(())
}

fn write_atomic_durable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let name = path
        .file_name()
        .ok_or_else(|| "output path has no file name".to_string())?
        .to_string_lossy();
    let temporary = parent.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        unix_milliseconds()
    ));
    let result = (|| -> Result<(), String> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        let mut writer = BufWriter::new(file);
        writer.write_all(bytes).map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
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

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupExecutionConfig {
    pub raw_batch_terms_per_lane: usize,
    pub max_union_keys_per_batch: usize,
    pub aggregate_host_payload_cap_bytes: u64,
    pub aggregate_device_cap_bytes: u64,
    pub contraction_device_cap_bytes: u64,
    pub per_lane_host_staging_cap_bytes: u64,
    pub download_chunk_terms: usize,
}

#[cfg(feature = "cuda")]
impl GroupExecutionConfig {
    pub(crate) fn from_environment() -> Result<Self, String> {
        let config = Self {
            raw_batch_terms_per_lane: env_usize("ADYNKRA_GPU_GROUP_RAW_BATCH_TERMS", 131_072)?,
            max_union_keys_per_batch: env_usize("ADYNKRA_GPU_GROUP_UNION_KEYS", 262_144)?,
            aggregate_host_payload_cap_bytes: env_u64(
                "ADYNKRA_GPU_GROUP_HOST_CAP_BYTES",
                1024 * 1024 * 1024,
            )?,
            aggregate_device_cap_bytes: env_u64(
                "ADYNKRA_GPU_GROUP_DEVICE_CAP_BYTES",
                16 * 1024 * 1024 * 1024,
            )?,
            contraction_device_cap_bytes: env_u64(
                "ADYNKRA_GPU_GROUP_CONTRACTION_CAP_BYTES",
                2 * 1024 * 1024 * 1024,
            )?,
            per_lane_host_staging_cap_bytes: env_u64(
                "ADYNKRA_GPU_GROUP_LANE_HOST_CAP_BYTES",
                256 * 1024 * 1024,
            )?,
            download_chunk_terms: env_usize("ADYNKRA_GPU_GROUP_DOWNLOAD_TERMS", 262_144)?,
        };
        if config.raw_batch_terms_per_lane == 0
            || config.max_union_keys_per_batch == 0
            || config.aggregate_host_payload_cap_bytes == 0
            || config.aggregate_device_cap_bytes == 0
            || config.contraction_device_cap_bytes == 0
            || config.per_lane_host_staging_cap_bytes == 0
            || config.download_chunk_terms == 0
        {
            return Err("GPU group execution caps must all be nonzero".to_string());
        }
        Ok(config)
    }
}

#[cfg(feature = "cuda")]
fn env_u64(name: &str, default: u64) -> Result<u64, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{name} must be an unsigned integer")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("cannot read {name}: {error}")),
    }
}

#[cfg(feature = "cuda")]
fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    let value = env_u64(name, default as u64)?;
    usize::try_from(value).map_err(|_| format!("{name} does not fit usize"))
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GpuGroupJobRunReport {
    pub schema_version: String,
    pub job_id: String,
    pub work_manifest_sha256: String,
    pub group_id: String,
    pub plan_sha256: String,
    pub tranche: String,
    pub group_index: usize,
    pub prime_index: usize,
    pub prime: u32,
    pub device: i32,
    pub device_name: String,
    pub resumed: bool,
    pub resume_word_ordinal: usize,
    pub completed_words: usize,
    pub total_words: usize,
    pub next_global_batch_ordinal: u64,
    pub checkpoint_path: String,
    pub checkpoint_sha256: String,
    pub event_log_path: String,
    pub execution_config: GroupExecutionConfig,
    pub device_budget: crate::eleven_dimensional_second_momentum_gpu::PersistentGroupDeviceBudget,
    pub timing: GroupTimingTotals,
    pub raw_terms_per_column: Vec<u64>,
    pub lowering_summaries: serde_json::Value,
    pub column_reports: Vec<crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnReport>,
    artifact_inventory: Vec<PublishedColumnArtifact>,
    pub end_to_end_milliseconds: u128,
    pub passed: bool,
    pub proof_boundary: String,
}

#[cfg(feature = "cuda")]
struct GroupJobLock {
    file: File,
}

#[cfg(all(feature = "cuda", unix))]
const LOCK_EX: std::ffi::c_int = 2;
#[cfg(all(feature = "cuda", unix))]
const LOCK_NB: std::ffi::c_int = 4;
#[cfg(all(feature = "cuda", unix))]
const LOCK_UN: std::ffi::c_int = 8;

#[cfg(all(feature = "cuda", unix))]
unsafe extern "C" {
    #[link_name = "flock"]
    fn group_job_flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
}

#[cfg(all(feature = "cuda", unix))]
impl Drop for GroupJobLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe {
            group_job_flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}

#[cfg(all(feature = "cuda", not(unix)))]
impl Drop for GroupJobLock {
    fn drop(&mut self) {}
}

#[cfg(feature = "cuda")]
fn acquire_group_job_lock(output_directory: &Path, job_id: &str) -> Result<GroupJobLock, String> {
    let directory = output_directory.join(".locks");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(format!("{job_id}.lock"));
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::io::{Seek, SeekFrom};
        use std::os::fd::AsRawFd;
        let status = unsafe { group_job_flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if status != 0 {
            return Err(format!(
                "another live worker owns job {job_id}: {}",
                io::Error::last_os_error()
            ));
        }
        file.set_len(0).map_err(|error| error.to_string())?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| error.to_string())?;
        writeln!(
            file,
            "{{\"job_id\":\"{job_id}\",\"pid\":{},\"started_unix_ms\":{}}}",
            std::process::id(),
            unix_milliseconds()
        )
        .map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
    }
    Ok(GroupJobLock { file })
}

#[cfg(feature = "cuda")]
fn plan_sha256(
    plan: &crate::second_momentum_gpu_group::PreparedColumnGroup,
) -> Result<String, String> {
    serde_json::to_vec(plan)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| error.to_string())
}

#[cfg(feature = "cuda")]
fn new_stream_hashers(
    width: usize,
) -> Result<(Vec<SerializableSha256>, Vec<SerializableSha256>), String> {
    let mut source = Vec::with_capacity(width);
    let mut packed = Vec::with_capacity(width);
    for _ in 0..width {
        let mut source_hasher = SerializableSha256::new();
        source_hasher
            .update(crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA.as_bytes())?;
        source_hasher.update(b"\0streamed-source-terms-v1\0")?;
        let mut packed_hasher = SerializableSha256::new();
        packed_hasher
            .update(crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA.as_bytes())?;
        packed_hasher.update(b"\0streamed-packed-terms-v1\0")?;
        source.push(source_hasher);
        packed.push(packed_hasher);
    }
    Ok((source, packed))
}

#[cfg(feature = "cuda")]
fn update_stream_hashers(
    source: &mut SerializableSha256,
    packed: &mut SerializableSha256,
    term: &crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm,
) -> Result<(), String> {
    let key = pack_job_term(term)?;
    // Preserve the byte contract exactly, but feed one contiguous record to
    // each SHA state instead of seven tiny updates per raw term. Production
    // columns contain billions of terms, so call overhead here is material.
    let mut source_record = [0_u8; 23];
    source_record[..2].copy_from_slice(&term.momentum_pair);
    source_record[2] = term.free_spinor;
    source_record[3..7].copy_from_slice(&term.exterior_mask.to_le_bytes());
    source_record[7..].copy_from_slice(&term.coefficient.to_le_bytes());
    source.update(&source_record)?;

    let mut packed_record = [0_u8; 24];
    packed_record[..8].copy_from_slice(&key.to_le_bytes());
    packed_record[8..16].copy_from_slice(&(term.coefficient as u128 as u64).to_le_bytes());
    packed_record[16..].copy_from_slice(&((term.coefficient >> 64) as i64).to_le_bytes());
    packed.update(&packed_record)
}

#[cfg(feature = "cuda")]
fn update_stream_hashers_batch(
    source: &mut SerializableSha256,
    packed: &mut SerializableSha256,
    terms: &[crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm],
    source_scratch: &mut Vec<u8>,
    packed_scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let source_bytes = terms
        .len()
        .checked_mul(23)
        .ok_or_else(|| "source hash batch byte count overflow".to_string())?;
    let packed_bytes = terms
        .len()
        .checked_mul(24)
        .ok_or_else(|| "packed hash batch byte count overflow".to_string())?;
    source_scratch.clear();
    packed_scratch.clear();
    source_scratch
        .try_reserve(source_bytes.saturating_sub(source_scratch.capacity()))
        .map_err(|error| format!("reserve source hash batch: {error}"))?;
    packed_scratch
        .try_reserve(packed_bytes.saturating_sub(packed_scratch.capacity()))
        .map_err(|error| format!("reserve packed hash batch: {error}"))?;
    for term in terms {
        let key = pack_job_term(term)?;
        source_scratch.extend_from_slice(&term.momentum_pair);
        source_scratch.push(term.free_spinor);
        source_scratch.extend_from_slice(&term.exterior_mask.to_le_bytes());
        source_scratch.extend_from_slice(&term.coefficient.to_le_bytes());
        packed_scratch.extend_from_slice(&key.to_le_bytes());
        packed_scratch.extend_from_slice(&(term.coefficient as u128 as u64).to_le_bytes());
        packed_scratch.extend_from_slice(&((term.coefficient >> 64) as i64).to_le_bytes());
    }
    debug_assert_eq!(source_scratch.len(), source_bytes);
    debug_assert_eq!(packed_scratch.len(), packed_bytes);
    source.update(source_scratch)?;
    packed.update(packed_scratch)
}

#[cfg(feature = "cuda")]
fn orchestration_host_cap_after_hash_scratch(
    aggregate_host_cap: u64,
    raw_batch_terms_per_lane: usize,
) -> Result<u64, String> {
    let hash_scratch_bytes = u64::try_from(raw_batch_terms_per_lane)
        .ok()
        .and_then(|terms| terms.checked_mul(23 + 24))
        .ok_or_else(|| "hash scratch byte count overflow".to_string())?;
    aggregate_host_cap
        .checked_sub(hash_scratch_bytes)
        .ok_or_else(|| {
            "aggregate host cap cannot cover the batched source and packed hash scratch".to_string()
        })
}

#[cfg(feature = "cuda")]
fn pack_job_term(
    term: &crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm,
) -> Result<u64, String> {
    if term.momentum_pair[0] > term.momentum_pair[1]
        || term.momentum_pair[1] >= 11
        || term.free_spinor >= 32
        || term.exterior_mask.count_ones() != 12
        || term.coefficient == 0
        || term.coefficient == i128::MIN
    {
        return Err("invalid raw term at grouped production hash boundary".to_string());
    }
    let metadata = u32::from(term.momentum_pair[0])
        | (u32::from(term.momentum_pair[1]) << 4)
        | (u32::from(term.free_spinor) << 8);
    Ok((u64::from(metadata) << 32) | u64::from(term.exterior_mask))
}

#[cfg(feature = "cuda")]
fn finish_source_digest(
    hasher: &SerializableSha256,
    global_ordinal: usize,
    source_label: &str,
    source_copy: usize,
    source_terms: u64,
) -> Result<String, String> {
    let mut outer = Sha256::new();
    outer.update(crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA.as_bytes());
    outer.update(b"\0bounded-streamed-source-v1\0");
    outer.update((global_ordinal as u64).to_le_bytes());
    outer.update(source_label.as_bytes());
    outer.update((source_copy as u64).to_le_bytes());
    outer.update(source_terms.to_le_bytes());
    outer.update(hasher.finalize_bytes()?);
    Ok(format!("{:x}", outer.finalize()))
}

#[cfg(feature = "cuda")]
fn finish_packed_digest(
    hasher: &SerializableSha256,
    global_ordinal: usize,
    source_terms: u64,
) -> Result<String, String> {
    let mut outer = Sha256::new();
    outer.update(crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA.as_bytes());
    outer.update(b"\0bounded-streamed-packed-v1\0");
    outer.update((global_ordinal as u64).to_le_bytes());
    outer.update(source_terms.to_le_bytes());
    outer.update(hasher.finalize_bytes()?);
    Ok(format!("{:x}", outer.finalize()))
}

#[cfg(feature = "cuda")]
fn update_timing(
    totals: &mut GroupTimingTotals,
    observation: &crate::second_momentum_gpu_group::GroupBatchObservation,
) -> Result<(), String> {
    totals.union_batches = totals
        .union_batches
        .checked_add(1)
        .ok_or_else(|| "group union batch count overflow".to_string())?;
    totals.union_milliseconds = totals
        .union_milliseconds
        .checked_add(observation.union.union_milliseconds)
        .ok_or_else(|| "group union timing overflow".to_string())?;
    totals.union_keys = totals
        .union_keys
        .checked_add(observation.union.union_key_count as u64)
        .ok_or_else(|| "group union key count overflow".to_string())?;
    totals.peak_union_keys = totals
        .peak_union_keys
        .max(observation.union.union_key_count);
    for (total, value) in totals
        .reduced_key_visits_per_column
        .iter_mut()
        .zip(&observation.union.reduced_terms_per_column)
    {
        *total = total
            .checked_add(*value as u64)
            .ok_or_else(|| "group reduced-key count overflow".to_string())?;
    }
    let cuda = observation
        .cuda
        .as_ref()
        .ok_or_else(|| "group batch has no CUDA telemetry".to_string())?;
    totals.upload_milliseconds += cuda.upload_milliseconds;
    totals.contract_milliseconds += cuda.contract_milliseconds;
    totals.finalize_milliseconds += cuda.finalize_milliseconds;
    totals.download_milliseconds += cuda.download_milliseconds;
    totals.total_cuda_milliseconds += cuda.total_milliseconds;
    totals.device_high_water_bytes = totals
        .device_high_water_bytes
        .max(cuda.device_high_water_bytes);
    for (total, value) in totals
        .expanded_contributions_per_column
        .iter_mut()
        .zip(&cuda.expanded_contributions_per_column)
    {
        *total = total
            .checked_add(*value)
            .ok_or_else(|| "group expanded-contribution count overflow".to_string())?;
    }
    for (total, value) in totals
        .nonzero_reduced_term_visits_per_column
        .iter_mut()
        .zip(&cuda.nonzero_terms_per_column)
    {
        *total = total
            .checked_add(*value)
            .ok_or_else(|| "group nonzero reduced-term count overflow".to_string())?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn update_shared_union_timing(
    totals: &mut crate::second_momentum_gpu_multi_prime_checkpoint::SharedUnionTotals,
    observation: &crate::second_momentum_gpu_group::GroupBatchObservation,
) -> Result<(), String> {
    totals.union_batches = totals
        .union_batches
        .checked_add(1)
        .ok_or_else(|| "multi-prime union batch count overflow".to_string())?;
    totals.union_milliseconds = totals
        .union_milliseconds
        .checked_add(observation.union.union_milliseconds)
        .ok_or_else(|| "multi-prime union timing overflow".to_string())?;
    totals.union_keys = totals
        .union_keys
        .checked_add(observation.union.union_key_count as u64)
        .ok_or_else(|| "multi-prime union key count overflow".to_string())?;
    totals.peak_union_keys = totals
        .peak_union_keys
        .max(observation.union.union_key_count);
    for (total, value) in totals
        .reduced_key_visits_per_column
        .iter_mut()
        .zip(&observation.union.reduced_terms_per_column)
    {
        *total = total
            .checked_add(*value as u64)
            .ok_or_else(|| "multi-prime reduced-key count overflow".to_string())?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn update_prime_cuda_timing(
    totals: &mut crate::second_momentum_gpu_multi_prime_checkpoint::PrimeCudaTimingTotals,
    observation: &crate::second_momentum_gpu_group::GroupBatchObservation,
) -> Result<(), String> {
    let cuda = observation
        .cuda
        .as_ref()
        .ok_or_else(|| "multi-prime batch has no CUDA telemetry".to_string())?;
    totals.upload_milliseconds += cuda.upload_milliseconds;
    totals.contract_milliseconds += cuda.contract_milliseconds;
    totals.finalize_milliseconds += cuda.finalize_milliseconds;
    totals.download_milliseconds += cuda.download_milliseconds;
    totals.total_cuda_milliseconds += cuda.total_milliseconds;
    totals.device_high_water_bytes = totals
        .device_high_water_bytes
        .max(cuda.device_high_water_bytes);
    for (total, value) in totals
        .expanded_contributions_per_column
        .iter_mut()
        .zip(&cuda.expanded_contributions_per_column)
    {
        *total = total
            .checked_add(*value)
            .ok_or_else(|| "multi-prime expanded-contribution count overflow".to_string())?;
    }
    for (total, value) in totals
        .nonzero_reduced_term_visits_per_column
        .iter_mut()
        .zip(&cuda.nonzero_terms_per_column)
    {
        *total = total
            .checked_add(*value)
            .ok_or_else(|| "multi-prime nonzero reduced-term count overflow".to_string())?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn compose_group_timing(
    shared: &crate::second_momentum_gpu_multi_prime_checkpoint::SharedUnionTotals,
    prime: &crate::second_momentum_gpu_multi_prime_checkpoint::PrimeCudaTimingTotals,
) -> GroupTimingTotals {
    GroupTimingTotals {
        union_milliseconds: shared.union_milliseconds,
        upload_milliseconds: prime.upload_milliseconds,
        contract_milliseconds: prime.contract_milliseconds,
        finalize_milliseconds: prime.finalize_milliseconds,
        download_milliseconds: prime.download_milliseconds,
        total_cuda_milliseconds: prime.total_cuda_milliseconds,
        union_batches: shared.union_batches,
        union_keys: shared.union_keys,
        peak_union_keys: shared.peak_union_keys,
        device_high_water_bytes: prime.device_high_water_bytes,
        expanded_contributions_per_column: prime.expanded_contributions_per_column.clone(),
        reduced_key_visits_per_column: shared.reduced_key_visits_per_column.clone(),
        nonzero_reduced_term_visits_per_column: prime
            .nonzero_reduced_term_visits_per_column
            .clone(),
    }
}

#[cfg(feature = "cuda")]
fn empty_prime_cuda_timing(
    width: usize,
) -> crate::second_momentum_gpu_multi_prime_checkpoint::PrimeCudaTimingTotals {
    crate::second_momentum_gpu_multi_prime_checkpoint::PrimeCudaTimingTotals {
        upload_milliseconds: 0.0,
        contract_milliseconds: 0.0,
        finalize_milliseconds: 0.0,
        download_milliseconds: 0.0,
        total_cuda_milliseconds: 0.0,
        expanded_contributions_per_column: vec![0; width],
        nonzero_reduced_term_visits_per_column: vec![0; width],
        device_high_water_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
fn group_event_log_write(
    writer: &mut BufWriter<File>,
    value: &impl Serialize,
) -> Result<(), String> {
    let mut bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    writer
        .write_all(&bytes)
        .map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    let line = serde_json::to_string(value).map_err(|error| error.to_string())?;
    println!("{line}");
    Ok(())
}

fn publish_conflict_safe(path: &Path, bytes: &[u8]) -> Result<(), String> {
    match fs::read(path) {
        Ok(existing) if existing == bytes => Ok(()),
        Ok(_) => Err(format!(
            "refusing to replace differing published artifact {}",
            path.display()
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => write_atomic_durable(path, bytes),
        Err(error) => Err(error.to_string()),
    }
}

pub(crate) fn import_completed_jobs(
    source_directory: &Path,
    destination_directory: &Path,
    jobs: &[GpuGroupJobKey],
) -> Result<serde_json::Value, String> {
    if source_directory == destination_directory {
        return Err("source and destination work roots must differ".to_string());
    }
    let source_manifest: GpuGroupJobManifest = serde_json::from_slice(
        &fs::read(source_directory.join("work-manifest.json"))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    validate_manifest(&source_manifest)?;
    write_or_validate_manifest(destination_directory)?;
    let mut imported = Vec::new();
    let mut adopted = Vec::new();
    for job in jobs {
        if !validate_completed_job(source_directory, job)? {
            return Err(format!("source job {} is not complete", job.id()));
        }
        let source_report_path = completed_job_report_path(source_directory, job);
        let source_report_bytes =
            fs::read(&source_report_path).map_err(|error| error.to_string())?;
        let source_report: serde_json::Value =
            serde_json::from_slice(&source_report_bytes).map_err(|error| error.to_string())?;
        let inventory = source_report
            .get("artifact_inventory")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("source job {} has no portable artifact inventory", job.id()))?;
        for artifact in inventory {
            let binary_relative = artifact
                .get("binary_relative_path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "artifact inventory has no binary relative path".to_string())?;
            let report_relative = artifact
                .get("report_relative_path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "artifact inventory has no report relative path".to_string())?;
            let source_binary = safe_relative_artifact_path(source_directory, binary_relative)?;
            let destination_binary =
                safe_relative_artifact_path(destination_directory, binary_relative)?;
            publish_conflict_safe(
                &destination_binary,
                &fs::read(&source_binary).map_err(|error| error.to_string())?,
            )?;
            let source_column_report =
                safe_relative_artifact_path(source_directory, report_relative)?;
            let destination_column_report =
                safe_relative_artifact_path(destination_directory, report_relative)?;
            publish_compatible_column_report(&source_column_report, &destination_column_report)?;
        }
        let destination_report_path = completed_job_report_path(destination_directory, job);
        if destination_report_path.exists() {
            if !validate_completed_job(destination_directory, job)? {
                return Err(format!(
                    "destination job {} exists but does not validate",
                    job.id()
                ));
            }
            let destination: serde_json::Value = serde_json::from_slice(
                &fs::read(&destination_report_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            if portable_job_identity(&destination) != portable_job_identity(&source_report) {
                return Err(format!(
                    "destination job {} has a different portable identity",
                    job.id()
                ));
            }
            adopted.push(job.id());
        } else {
            write_atomic_durable(&destination_report_path, &source_report_bytes)?;
            if !validate_completed_job(destination_directory, job)? {
                return Err(format!("imported job {} failed validation", job.id()));
            }
            imported.push(job.id());
        }
    }
    Ok(serde_json::json!({
        "schema_version": GPU_GROUP_JOB_SCHEMA,
        "source_directory": source_directory,
        "destination_directory": destination_directory,
        "selected_jobs": jobs.len(),
        "imported": imported,
        "adopted_existing": adopted,
        "passed": true,
    }))
}

fn portable_job_identity(value: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "schema_version": value.get("schema_version"),
        "job_id": value.get("job_id"),
        "work_manifest_sha256": value.get("work_manifest_sha256"),
        "group_id": value.get("group_id"),
        "plan_sha256": value.get("plan_sha256"),
        "tranche": value.get("tranche"),
        "group_index": value.get("group_index"),
        "prime_index": value.get("prime_index"),
        "prime": value.get("prime"),
        "artifact_inventory": value.get("artifact_inventory"),
        "passed": value.get("passed"),
    })
}

fn publish_compatible_column_report(source: &Path, destination: &Path) -> Result<(), String> {
    let source_bytes = fs::read(source).map_err(|error| error.to_string())?;
    match fs::read(destination) {
        Ok(existing) if existing == source_bytes => Ok(()),
        Ok(existing) => {
            let source_value: serde_json::Value =
                serde_json::from_slice(&source_bytes).map_err(|error| error.to_string())?;
            let destination_value: serde_json::Value =
                serde_json::from_slice(&existing).map_err(|error| error.to_string())?;
            let fields = [
                "schema_version",
                "global_ordinal",
                "prime",
                "binary_sha256",
                "column_semantic_sha256",
                "passed",
            ];
            if fields
                .iter()
                .all(|field| source_value.get(field) == destination_value.get(field))
            {
                Ok(())
            } else {
                Err(format!(
                    "refusing incompatible column report {}",
                    destination.display()
                ))
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            write_atomic_durable(destination, &source_bytes)
        }
        Err(error) => Err(error.to_string()),
    }
}

pub(crate) fn completed_job_report_path(output_directory: &Path, job: &GpuGroupJobKey) -> PathBuf {
    output_directory
        .join("jobs")
        .join(job.id())
        .join("job-report.json")
}

pub(crate) fn validate_completed_job(
    output_directory: &Path,
    job: &GpuGroupJobKey,
) -> Result<bool, String> {
    let path = completed_job_report_path(output_directory, job);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.to_string()),
    };
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some(GPU_GROUP_RUN_SCHEMA)
        || value.get("job_id").and_then(serde_json::Value::as_str) != Some(&job.id())
        || value.get("passed").and_then(serde_json::Value::as_bool) != Some(true)
    {
        return Err(format!(
            "{} is not a valid completed job report",
            path.display()
        ));
    }
    if let Some(inventory) = value
        .get("artifact_inventory")
        .and_then(serde_json::Value::as_array)
    {
        for artifact in inventory {
            let relative = artifact
                .get("binary_relative_path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "artifact inventory entry has no relative binary path".to_string()
                })?;
            let binary_path = safe_relative_artifact_path(output_directory, relative)?;
            let expected = artifact
                .get("binary_sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "artifact inventory entry has no binary digest".to_string())?;
            verify_binary_digest(&binary_path, expected)?;
        }
        return Ok(true);
    }
    let reports = value
        .get("column_reports")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "completed group report has no column reports".to_string())?;
    for report in reports {
        let stored_path = report
            .get("binary_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "column report has no binary path".to_string())?;
        let stored = Path::new(stored_path);
        let binary_path = if stored.is_file() {
            stored.to_path_buf()
        } else {
            let name = stored
                .file_name()
                .ok_or_else(|| "column report binary path has no file name".to_string())?;
            output_directory.join(name)
        };
        let expected = report
            .get("binary_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "column report has no binary digest".to_string())?;
        verify_binary_digest(&binary_path, expected)?;
    }
    Ok(true)
}

fn safe_relative_artifact_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err("artifact inventory path is not a contained relative path".to_string());
    }
    Ok(root.join(path))
}

fn verify_binary_digest(path: &Path, expected: &str) -> Result<(), String> {
    let observed = fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| error.to_string())?;
    if observed != expected {
        return Err(format!(
            "completed binary digest mismatch for {}",
            path.display()
        ));
    }
    Ok(())
}

pub(crate) fn summarize_jobs(
    output_directory: &Path,
    jobs: &[GpuGroupJobKey],
) -> serde_json::Value {
    const LIVE_HEARTBEAT_MAX_AGE_MS: u128 = 30_000;
    let now = unix_milliseconds();
    let local_hostname = machine_hostname();
    let mut completed = Vec::new();
    let mut running = Vec::new();
    let mut failed = Vec::new();
    let mut stale = Vec::new();
    let mut pending = Vec::new();
    let mut job_details = Vec::new();
    for job in jobs {
        let id = job.id();
        match validate_completed_job(output_directory, job) {
            Ok(true) => {
                completed.push(id.clone());
                job_details.push(serde_json::json!({
                    "job_id": id,
                    "state": "completed",
                    "report_path": completed_job_report_path(output_directory, job),
                }));
                continue;
            }
            Err(error) => {
                failed.push(serde_json::json!({"job_id": id, "error": error}));
                job_details.push(serde_json::json!({
                    "job_id": job.id(),
                    "state": "invalid_completed_artifact",
                    "error": error,
                    "report_path": completed_job_report_path(output_directory, job),
                }));
                continue;
            }
            Ok(false) => {}
        }
        let status_path = output_directory.join("jobs").join(&id).join("status.json");
        let status = fs::read(&status_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        match status {
            Some(value) => {
                let state = value
                    .get("state")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("invalid");
                let pid = value
                    .get("pid")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok());
                let hostname = value
                    .get("hostname")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                let timestamp = value
                    .get("timestamp_unix_ms")
                    .and_then(serde_json::Value::as_u64)
                    .map(u128::from);
                let heartbeat_age_ms = timestamp.map(|value| now.saturating_sub(value));
                let heartbeat_recent =
                    heartbeat_age_ms.is_some_and(|age| age <= LIVE_HEARTBEAT_MAX_AGE_MS);
                let local_process_live =
                    hostname == local_hostname && pid.is_some_and(process_is_running);
                let remote_process_live =
                    hostname != "unknown" && hostname != local_hostname && heartbeat_recent;
                let legacy_process_live = hostname == "unknown"
                    && (pid.is_some_and(process_is_running) || heartbeat_recent);
                let observed_live =
                    local_process_live || remote_process_live || legacy_process_live;
                let effective_state = if state == "running" && observed_live {
                    running.push(id.clone());
                    "running"
                } else if state == "running" {
                    stale.push(id.clone());
                    "stale"
                } else if state == "failed" || state == "terminated" {
                    failed.push(serde_json::json!({"job_id": id, "state": state}));
                    "failed"
                } else {
                    pending.push(id.clone());
                    "pending"
                };
                job_details.push(serde_json::json!({
                    "job_id": job.id(),
                    "state": effective_state,
                    "reported_state": state,
                    "hostname": hostname,
                    "pid": pid,
                    "heartbeat_age_milliseconds": heartbeat_age_ms,
                    "phase": value.get("phase"),
                    "elapsed_seconds": value.get("elapsed_seconds"),
                    "message": value.get("message"),
                    "error": value.get("error"),
                    "group_progress": value.get("group_progress"),
                    "streaming": value.get("streaming"),
                    "gpu_batches": value.get("gpu_batches"),
                    "throughput": value.get("throughput"),
                    "resources": value.get("resources"),
                    "paths": value.get("paths"),
                    "status_path": status_path,
                }));
            }
            None => {
                pending.push(id.clone());
                job_details.push(serde_json::json!({
                    "job_id": id,
                    "state": "pending",
                    "status_path": status_path,
                }));
            }
        }
    }
    let pending_job_list = pending.join(",");
    let stale_job_list = stale.join(",");
    serde_json::json!({
        "schema_version": GPU_GROUP_JOB_SCHEMA,
        "timestamp_unix_ms": unix_milliseconds(),
        "output_directory": output_directory,
        "selected_jobs": jobs.len(),
        "completed_count": completed.len(),
        "running_count": running.len(),
        "failed_count": failed.len(),
        "stale_count": stale.len(),
        "pending_count": pending.len(),
        "completed": completed,
        "running": running,
        "failed": failed,
        "stale": stale,
        "pending": pending,
        "pending_job_list": pending_job_list,
        "stale_job_list": stale_job_list,
        "job_details": job_details,
        "complete": completed.len() == jobs.len(),
    })
}

fn machine_hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn process_is_running(pid: u32) -> bool {
    std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid="])
        .output()
        .is_ok_and(|output| {
            output.status.success() && !String::from_utf8_lossy(&output.stdout).trim().is_empty()
        })
}

#[cfg(feature = "cuda")]
pub(crate) fn commit_singleton_job(
    output_directory: &Path,
    job: &GpuGroupJobKey,
    report: &crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnReport,
) -> Result<serde_json::Value, String> {
    let manifest = build_job_manifest()?;
    let binary_name = Path::new(&report.binary_path)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "singleton binary path has no portable file name".to_string())?;
    let report_name = format!(
        "second_momentum_{}_column_{:02}_p{}.json",
        report.tranche, report.global_ordinal, report.prime
    );
    let value = serde_json::json!({
        "schema_version": GPU_GROUP_RUN_SCHEMA,
        "job_id": job.id(),
        "work_manifest_sha256": manifest.manifest_sha256,
        "tranche": &job.tranche,
        "group_index": job.group_index,
        "prime_index": job.prime_index,
        "prime": job.prime()?,
        "singleton_fallback": true,
        "resumable": false,
        "column_reports": [report],
        "artifact_inventory": [{
            "global_ordinal": report.global_ordinal,
            "binary_relative_path": binary_name,
            "report_relative_path": report_name,
            "binary_sha256": report.binary_sha256,
            "column_semantic_sha256": report.column_semantic_sha256,
        }],
        "passed": report.passed,
        "proof_boundary": "Singleton inventory entries use the validated single-column CUDA fallback and publish this job commit record last."
    });
    if !report.passed {
        return Err("singleton column report failed its gates".to_string());
    }
    let path = completed_job_report_path(output_directory, job);
    let mut bytes = serde_json::to_vec_pretty(&value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    publish_conflict_safe(&path, &bytes)?;
    Ok(value)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_singleton_job(
    output_directory: &Path,
    job: &GpuGroupJobKey,
    device: i32,
    cpu_parity_terms: usize,
    live_progress: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<serde_json::Value, String> {
    let locals = job.local_ordinals()?;
    if locals.len() != 1 {
        return Err("singleton fallback received a multi-column job".to_string());
    }
    let _lock = acquire_group_job_lock(output_directory, &job.id())?;
    if validate_completed_job(output_directory, job)? {
        return fs::read(completed_job_report_path(output_directory, job))
            .map_err(|error| error.to_string())
            .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()));
    }
    let report = crate::eleven_dimensional_second_momentum_gpu::run_cuda_column(
        job.tranche()?.as_str(),
        locals[0],
        job.prime()?,
        device,
        output_directory,
        cpu_parity_terms,
        Some(live_progress),
    )?;
    commit_singleton_job(output_directory, job, &report)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_group_job(
    job: &GpuGroupJobKey,
    output_directory: &Path,
    device: i32,
    cpu_parity_terms: usize,
    live_progress: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<GpuGroupJobRunReport, String> {
    run_group_job_request(
        GroupJobRequest::Legacy(job.clone()),
        output_directory,
        device,
        cpu_parity_terms,
        live_progress,
    )
}

/// Execute two or three prime jobs for one legacy width-2/3 source group in a
/// single exact PBW traversal. Raw generation, transcript hashing, reduction,
/// and key union happen once. Each prime retains an independent CUDA context,
/// row accumulator, checkpoint state, and standard publication artifact.
#[cfg(feature = "cuda")]
pub(crate) fn run_multi_prime_group_jobs(
    requested_jobs: &[GpuGroupJobKey],
    output_directory: &Path,
    device: i32,
    cpu_parity_terms: usize,
    live_progress: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<Vec<GpuGroupJobRunReport>, String> {
    use crate::eleven_dimensional_second_momentum_gpu::{
        CudaModularFx, ModularFxStaticData, PersistentCudaMultiPrimeGroupExecutor,
    };
    use crate::second_momentum_gpu_group::{
        GroupRuntimeIdentity, GroupWordOrchestrationConfig, multi_prime_group_identity_sha256,
        prepare_cuda_column_group, source_group_identity_sha256,
    };
    use crate::second_momentum_gpu_multi_prime_checkpoint::{
        MULTI_PRIME_CHECKPOINT_SCHEMA, MultiPrimeCheckpoint, MultiPrimeRuntimeIdentity,
        PrimeCheckpointState, SharedExactStreamState, SharedLaneLoweringTotals, SharedUnionTotals,
        read_multi_prime_checkpoint, write_multi_prime_checkpoint,
    };
    use crate::second_momentum_gpu_progress::{
        GpuBatchProgress, GroupLiveProgress, SourceVisitorProgress,
    };

    if device < 0
        || cpu_parity_terms == 0
        || !(2..=GPU_FX_PRIMES.len()).contains(&requested_jobs.len())
    {
        return Err(
            "multi-prime runner requires 2-3 jobs, a nonnegative device, and nonzero parity"
                .to_string(),
        );
    }
    let mut jobs = requested_jobs.to_vec();
    jobs.sort_by_key(|job| job.prime_index);
    let first = jobs
        .first()
        .ok_or_else(|| "multi-prime job list is empty".to_string())?;
    if jobs.iter().enumerate().any(|(slot, job)| {
        job.tranche != first.tranche
            || job.group_index != first.group_index
            || slot > 0 && jobs[slot - 1].prime_index >= job.prime_index
    }) {
        return Err(
            "multi-prime jobs must be one group in canonical prime-index order".to_string(),
        );
    }
    let local_ordinals = first.local_ordinals()?;
    if local_ordinals.len() < 2 {
        return Err("multi-prime grouped runner currently requires a width-2/3 group".to_string());
    }
    let manifest = build_job_manifest()?;
    write_or_validate_manifest(output_directory)?;
    for job in &jobs {
        if validate_completed_job(output_directory, job)? {
            return Err(format!(
                "job {} is already complete and validated",
                job.id()
            ));
        }
        let legacy_checkpoint = output_directory
            .join("jobs")
            .join(job.id())
            .join("checkpoint.json");
        if legacy_checkpoint.exists() {
            return Err(format!(
                "refusing to merge partial single-prime checkpoint {} into a multi-prime run",
                legacy_checkpoint.display()
            ));
        }
    }

    let bundle_id = format!(
        "{}-g{}-mp{}",
        first.tranche,
        first.group_index,
        jobs.iter()
            .map(|job| job.prime_index.to_string())
            .collect::<String>()
    );
    let bundle_directory = output_directory.join("jobs").join(&bundle_id);
    fs::create_dir_all(&bundle_directory).map_err(|error| error.to_string())?;
    let checkpoint_path = bundle_directory.join("checkpoint.json");
    let event_log_path = bundle_directory.join("events.jsonl");
    let event_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&event_log_path)
        .map_err(|error| error.to_string())?;
    let mut event_log = BufWriter::new(event_file);

    // Lock every standard job in canonical order for the complete shared
    // traversal. This excludes single-prime workers without inventing a second
    // ownership protocol.
    let locks = jobs
        .iter()
        .map(|job| acquire_group_job_lock(output_directory, &job.id()))
        .collect::<Result<Vec<_>, _>>()?;
    let execution_config = GroupExecutionConfig::from_environment()?;
    let orchestration_host_cap = orchestration_host_cap_after_hash_scratch(
        execution_config.aggregate_host_payload_cap_bytes,
        execution_config.raw_batch_terms_per_lane,
    )?;
    let started = std::time::Instant::now();
    let mut static_data = Vec::with_capacity(jobs.len());
    let mut plans = Vec::with_capacity(jobs.len());
    let mut plan_digests = Vec::with_capacity(jobs.len());
    let mut device_name = None;
    for job in &jobs {
        let prime = job.prime()?;
        let data = ModularFxStaticData::build(prime)?;
        let probe = CudaModularFx::new(&data, device)?;
        if let Some(expected) = device_name.as_deref() {
            if expected != probe.device_name() {
                return Err("CUDA device identity changed between prime contexts".to_string());
            }
        } else {
            device_name = Some(probe.device_name().to_string());
        }
        let runtime = GroupRuntimeIdentity {
            prime,
            static_semantic_sha256: data.semantic_sha256().to_string(),
            flat_plan_sha256: probe.flat_plan_sha256().to_string(),
        };
        drop(probe);
        let plan = prepare_cuda_column_group(job.tranche()?, &local_ordinals, runtime)?;
        plan_digests.push(plan_sha256(&plan)?);
        plans.push(plan);
        static_data.push(data);
    }
    let source_group_sha256 = source_group_identity_sha256(&plans[0]);
    let multi_prime_group_sha256 = multi_prime_group_identity_sha256(&plans)?;
    let width = plans[0].active_columns;
    let prime_runtimes = jobs
        .iter()
        .zip(&plans)
        .zip(&plan_digests)
        .map(|((job, plan), plan_sha256)| MultiPrimeRuntimeIdentity {
            prime_index: job.prime_index,
            prime: plan.runtime.prime,
            single_job_id: job.id(),
            group_id: plan.group_id.clone(),
            plan_sha256: plan_sha256.clone(),
            static_semantic_sha256: plan.runtime.static_semantic_sha256.clone(),
            flat_plan_sha256: plan.runtime.flat_plan_sha256.clone(),
        })
        .collect::<Vec<_>>();

    let mut checkpoint = if checkpoint_path.exists() {
        let existing = read_multi_prime_checkpoint(&checkpoint_path)?;
        if existing.bundle_id != bundle_id
            || existing.work_manifest_sha256 != manifest.manifest_sha256
            || existing.source_group_sha256 != source_group_sha256
            || existing.multi_prime_group_sha256 != multi_prime_group_sha256
            || existing.prime_runtimes != prime_runtimes
            || existing.ordered_local_ordinals != plans[0].ordered_local_ordinals
            || existing.ordered_global_ordinals != plans[0].ordered_global_ordinals
            || existing.ordered_source_copies != plans[0].ordered_source_copies
            || existing.pbw_word_count != plans[0].pbw_word_count
        {
            return Err("multi-prime checkpoint identity does not match this run".to_string());
        }
        existing
    } else {
        let (source_hashers, packed_hashers) = new_stream_hashers(width)?;
        MultiPrimeCheckpoint {
            schema_version: MULTI_PRIME_CHECKPOINT_SCHEMA.to_string(),
            bundle_id: bundle_id.clone(),
            run_schema_version: GPU_GROUP_RUN_SCHEMA.to_string(),
            work_manifest_sha256: manifest.manifest_sha256.clone(),
            tranche: first.tranche.clone(),
            group_index: first.group_index,
            source_group_sha256: source_group_sha256.clone(),
            multi_prime_group_sha256: multi_prime_group_sha256.clone(),
            ordered_local_ordinals: plans[0].ordered_local_ordinals.clone(),
            ordered_global_ordinals: plans[0].ordered_global_ordinals.clone(),
            ordered_source_copies: plans[0].ordered_source_copies.clone(),
            pbw_word_count: plans[0].pbw_word_count,
            active_columns: width,
            prime_runtimes: prime_runtimes.clone(),
            next_word_ordinal: 0,
            next_global_batch_ordinal: 0,
            checkpoint_generation: 0,
            shared: SharedExactStreamState {
                raw_terms_per_column: vec![0; width],
                source_hashers: source_hashers
                    .iter()
                    .map(serializable_checkpoint_state)
                    .collect(),
                packed_hashers: packed_hashers
                    .iter()
                    .map(serializable_checkpoint_state)
                    .collect(),
                union: SharedUnionTotals {
                    union_batches: 0,
                    union_milliseconds: 0,
                    union_keys: 0,
                    peak_union_keys: 0,
                    reduced_key_visits_per_column: vec![0; width],
                },
                lowering_per_column: vec![
                    SharedLaneLoweringTotals {
                        enabled: true,
                        roots_lowered: 0,
                        input_entry_visits: 0,
                        expanded_entry_visits: 0,
                        output_entry_visits: 0,
                        gpu_milliseconds: 0.0,
                        scratch_high_water_bytes: 0,
                        peak_immutable_handle_bytes: 0,
                        maximum_absolute_coefficient: 0,
                        device_hard_cap_bytes: 0,
                        download_chunk_terms: execution_config.download_chunk_terms,
                    };
                    width
                ],
            },
            primes: jobs
                .iter()
                .map(|job| PrimeCheckpointState {
                    prime_index: job.prime_index,
                    batches_folded: 0,
                    rows: vec![
                        vec![
                            GaussianResidue::zero();
                            crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
                        ];
                        width
                    ],
                    timing: empty_prime_cuda_timing(width),
                })
                .collect(),
        }
    };
    let resumed = checkpoint.next_word_ordinal != 0;
    let mut source_hashers = checkpoint
        .shared
        .source_hashers
        .iter()
        .map(serializable_from_checkpoint_state)
        .collect::<Result<Vec<_>, _>>()?;
    let mut packed_hashers = checkpoint
        .shared
        .packed_hashers
        .iter()
        .map(serializable_from_checkpoint_state)
        .collect::<Result<Vec<_>, _>>()?;
    let mut parity_terms = (0..width)
        .map(|_| Vec::with_capacity(cpu_parity_terms))
        .collect::<Vec<Vec<crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm>>>();
    let mut source_hash_scratch = Vec::new();
    let mut packed_hash_scratch = Vec::new();
    let mut executor = PersistentCudaMultiPrimeGroupExecutor::new(
        plans.clone(),
        &static_data,
        device,
        execution_config.max_union_keys_per_batch,
        execution_config.aggregate_device_cap_bytes,
        execution_config.contraction_device_cap_bytes,
        execution_config.per_lane_host_staging_cap_bytes,
        execution_config.download_chunk_terms,
        None,
    )?;
    executor.restore_columns(
        checkpoint
            .primes
            .iter()
            .map(|prime| prime.rows.clone())
            .collect(),
        checkpoint.next_global_batch_ordinal,
    )?;
    if resumed {
        parity_terms = executor.collect_parity_prefix(cpu_parity_terms)?;
    }
    group_event_log_write(
        &mut event_log,
        &serde_json::json!({
            "schema_version": GPU_GROUP_RUN_SCHEMA,
            "event": "multi_prime_job_start",
            "timestamp_unix_ms": unix_milliseconds(),
            "bundle_id": bundle_id,
            "prime_indices": jobs.iter().map(|job| job.prime_index).collect::<Vec<_>>(),
            "primes": jobs.iter().map(GpuGroupJobKey::prime).collect::<Result<Vec<_>, _>>()?,
            "resume_word_ordinal": checkpoint.next_word_ordinal,
            "total_words": plans[0].pbw_word_count,
            "execution_config": execution_config,
            "device_budget": executor.device_budget(),
        }),
    )?;

    for word_ordinal in checkpoint.next_word_ordinal..plans[0].pbw_word_count {
        let before_raw = checkpoint.shared.raw_terms_per_column.clone();
        let first_global_batch_ordinal = checkpoint.next_global_batch_ordinal;
        let checkpoint_cell = std::cell::RefCell::new(&mut checkpoint);
        let orchestration = executor.run_word_synchronous_batched(
            GroupWordOrchestrationConfig {
                start_word_ordinal: word_ordinal,
                end_word_ordinal_exclusive: word_ordinal + 1,
                first_global_batch_ordinal,
                raw_batch_term_cap_per_lane: execution_config.raw_batch_terms_per_lane,
                max_union_keys_per_batch: execution_config.max_union_keys_per_batch,
                aggregate_host_payload_cap_bytes: orchestration_host_cap,
            },
            |lane, _, terms| {
                let mut borrowed = checkpoint_cell.borrow_mut();
                let checkpoint = &mut **borrowed;
                update_stream_hashers_batch(
                    &mut source_hashers[lane],
                    &mut packed_hashers[lane],
                    terms,
                    &mut source_hash_scratch,
                    &mut packed_hash_scratch,
                )?;
                checkpoint.shared.raw_terms_per_column[lane] =
                    checkpoint.shared.raw_terms_per_column[lane]
                        .checked_add(terms.len() as u64)
                        .ok_or_else(|| "multi-prime source-term count overflow".to_string())?;
                if parity_terms[lane].len() < cpu_parity_terms {
                    let remaining = cpu_parity_terms - parity_terms[lane].len();
                    parity_terms[lane].extend(terms.iter().take(remaining).cloned());
                }
                Ok(())
            },
            |prime_slot, prime, observation| {
                let mut borrowed = checkpoint_cell.borrow_mut();
                let checkpoint = &mut **borrowed;
                if prime_slot == 0 {
                    update_shared_union_timing(&mut checkpoint.shared.union, observation)?;
                } else if observation.union.batch_ordinal + 1
                    != checkpoint.shared.union.union_batches
                {
                    return Err("multi-prime union batch identity diverged".to_string());
                }
                update_prime_cuda_timing(&mut checkpoint.primes[prime_slot].timing, observation)?;
                group_event_log_write(
                    &mut event_log,
                    &serde_json::json!({
                        "schema_version": GPU_GROUP_RUN_SCHEMA,
                        "event": "multi_prime_batch",
                        "bundle_id": bundle_id,
                        "prime_slot": prime_slot,
                        "prime": prime,
                        "observation": observation,
                    }),
                )?;
                if prime_slot + 1 == jobs.len() {
                    let cuda = observation.cuda.as_ref().ok_or_else(|| {
                        "multi-prime observation is missing CUDA timing".to_string()
                    })?;
                    live_progress.record_gpu_batch(GpuBatchProgress {
                        batches_completed: checkpoint.shared.union.union_batches,
                        last_batch_ms: observation.union.union_milliseconds as f64
                            + cuda.total_milliseconds,
                        total_batch_ms: checkpoint.shared.union.union_milliseconds as f64
                            + checkpoint
                                .primes
                                .iter()
                                .map(|state| state.timing.total_cuda_milliseconds)
                                .sum::<f64>(),
                        last_upload_ms: cuda.upload_milliseconds,
                        total_upload_ms: checkpoint.primes[prime_slot].timing.upload_milliseconds,
                        last_sort_ms: observation.union.union_milliseconds as f64,
                        total_sort_ms: checkpoint.shared.union.union_milliseconds as f64,
                        last_reduce_ms: 0.0,
                        total_reduce_ms: 0.0,
                        last_contract_ms: cuda.contract_milliseconds,
                        total_contract_ms: checkpoint.primes[prime_slot]
                            .timing
                            .contract_milliseconds,
                        last_download_ms: cuda.download_milliseconds,
                        total_download_ms: checkpoint.primes[prime_slot]
                            .timing
                            .download_milliseconds,
                    });
                }
                Ok(())
            },
            |completed_word, completions| {
                if completed_word != word_ordinal || completions.len() != width {
                    return Err("multi-prime word completion identity changed".to_string());
                }
                Ok(())
            },
        )?;
        drop(checkpoint_cell);
        for lane in 0..width {
            let observed = checkpoint.shared.raw_terms_per_column[lane] - before_raw[lane];
            if observed != orchestration.raw_terms_per_column[lane] {
                return Err("multi-prime word raw-term accounting mismatch".to_string());
            }
        }
        checkpoint.next_word_ordinal = orchestration.next_word_ordinal;
        checkpoint.next_global_batch_ordinal = orchestration.next_global_batch_ordinal;
        checkpoint.checkpoint_generation = checkpoint.next_word_ordinal as u64;
        checkpoint.shared.source_hashers = source_hashers
            .iter()
            .map(serializable_checkpoint_state)
            .collect();
        checkpoint.shared.packed_hashers = packed_hashers
            .iter()
            .map(serializable_checkpoint_state)
            .collect();
        checkpoint.shared.lowering_per_column = executor
            .lowering_summaries()?
            .into_iter()
            .map(|summary| SharedLaneLoweringTotals {
                enabled: summary.enabled,
                roots_lowered: summary.roots_lowered,
                input_entry_visits: summary.input_entry_visits,
                expanded_entry_visits: summary.expanded_entry_visits,
                output_entry_visits: summary.output_entry_visits,
                gpu_milliseconds: summary.gpu_milliseconds,
                scratch_high_water_bytes: summary.scratch_high_water_bytes,
                peak_immutable_handle_bytes: summary.peak_immutable_handle_bytes,
                maximum_absolute_coefficient: summary.maximum_absolute_coefficient,
                device_hard_cap_bytes: summary.device_hard_cap_bytes,
                download_chunk_terms: summary.download_chunk_terms,
            })
            .collect();
        let batches_folded = executor.batches_folded()?;
        if batches_folded != checkpoint.next_global_batch_ordinal {
            return Err("multi-prime checkpoint batch boundary diverged".to_string());
        }
        for prime_slot in 0..jobs.len() {
            checkpoint.primes[prime_slot].batches_folded = batches_folded;
            checkpoint.primes[prime_slot].rows = executor.final_columns(prime_slot)?.to_vec();
        }
        let checkpoint_sha256 = write_multi_prime_checkpoint(&checkpoint_path, &checkpoint)?;
        live_progress.update_source(SourceVisitorProgress {
            word: Some(word_ordinal as u64),
            root: None,
            raw_terms_emitted: checkpoint.shared.raw_terms_per_column.iter().sum(),
            batches_flushed: checkpoint.shared.union.union_batches,
            current_batch_terms: 0,
            current_batch_bytes: source_hash_scratch.capacity() as u64
                + packed_hash_scratch.capacity() as u64,
            hard_memory_cap_bytes: execution_config.aggregate_host_payload_cap_bytes,
            eta_sample_count: checkpoint.shared.union.union_batches,
        });
        live_progress.update_group(GroupLiveProgress {
            group_id: Some(bundle_id.clone()),
            words_completed: checkpoint.next_word_ordinal,
            words_total: checkpoint.pbw_word_count,
            global_batch_ordinal: checkpoint.next_global_batch_ordinal,
            raw_terms_per_column: checkpoint.shared.raw_terms_per_column.clone(),
            last_union_key_count: 0,
            cumulative_union_keys: checkpoint.shared.union.union_keys,
            keys_by_present_lane_count: Vec::new(),
            host_capacity_bytes: source_hash_scratch.capacity() as u64
                + packed_hash_scratch.capacity() as u64,
            aggregate_host_cap_bytes: execution_config.aggregate_host_payload_cap_bytes,
            device_resident_bytes: executor.device_budget().total_contraction_resident_bytes,
            device_high_water_bytes: checkpoint
                .primes
                .iter()
                .map(|state| state.timing.device_high_water_bytes)
                .max()
                .unwrap_or(0),
            aggregate_device_cap_bytes: execution_config.aggregate_device_cap_bytes,
            checkpoint_generation: checkpoint.checkpoint_generation,
            checkpoint_sha256: Some(checkpoint_sha256.clone()),
            checkpoint_written_unix_ms: Some(unix_milliseconds()),
        });
        group_event_log_write(
            &mut event_log,
            &serde_json::json!({
                "schema_version": GPU_GROUP_RUN_SCHEMA,
                "event": "multi_prime_word_checkpoint",
                "timestamp_unix_ms": unix_milliseconds(),
                "bundle_id": bundle_id,
                "completed_word_ordinal": word_ordinal,
                "next_word_ordinal": checkpoint.next_word_ordinal,
                "word_total": checkpoint.pbw_word_count,
                "raw_terms_per_column": checkpoint.shared.raw_terms_per_column,
                "batches_folded": checkpoint.next_global_batch_ordinal,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_path": checkpoint_path,
            }),
        )?;
    }
    if checkpoint.next_word_ordinal != checkpoint.pbw_word_count
        || parity_terms.iter().any(Vec::is_empty)
    {
        return Err(
            "multi-prime execution did not complete every word or parity prefix".to_string(),
        );
    }
    write_multi_prime_checkpoint(&checkpoint_path, &checkpoint)?;
    drop(executor);

    // Materialize standard complete per-prime checkpoints. The existing,
    // heavily validated publication path then emits the exact same ADFXGPU3
    // binaries and per-prime reports without repeating any PBW word.
    for (slot, job) in jobs.iter().enumerate() {
        let job_directory = output_directory.join("jobs").join(job.id());
        fs::create_dir_all(&job_directory).map_err(|error| error.to_string())?;
        let legacy_checkpoint_path = job_directory.join("checkpoint.json");
        let timing =
            compose_group_timing(&checkpoint.shared.union, &checkpoint.primes[slot].timing);
        let legacy_checkpoint = GroupJobCheckpoint {
            schema_version: GPU_GROUP_CHECKPOINT_SCHEMA.to_string(),
            job_id: job.id(),
            group_id: plans[slot].group_id.clone(),
            plan_sha256: plan_digests[slot].clone(),
            next_word_ordinal: checkpoint.next_word_ordinal,
            next_global_batch_ordinal: checkpoint.next_global_batch_ordinal,
            batches_folded: checkpoint.next_global_batch_ordinal,
            raw_terms_per_column: checkpoint.shared.raw_terms_per_column.clone(),
            source_hashers: source_hashers.clone(),
            packed_hashers: packed_hashers.clone(),
            rows: checkpoint.primes[slot].rows.clone(),
            timing,
            lowering_summaries: Some(checkpoint.shared.lowering_per_column.clone()),
            checkpoint_generation: checkpoint.checkpoint_generation,
        };
        write_checkpoint(&legacy_checkpoint_path, &legacy_checkpoint)?;
    }
    drop(locks);
    let mut reports = Vec::with_capacity(jobs.len());
    for job in &jobs {
        reports.push(run_group_job(
            job,
            output_directory,
            device,
            cpu_parity_terms,
            live_progress,
        )?);
    }
    group_event_log_write(
        &mut event_log,
        &serde_json::json!({
            "schema_version": GPU_GROUP_RUN_SCHEMA,
            "event": "multi_prime_job_complete",
            "timestamp_unix_ms": unix_milliseconds(),
            "bundle_id": bundle_id,
            "job_ids": jobs.iter().map(GpuGroupJobKey::id).collect::<Vec<_>>(),
            "completed_words": checkpoint.next_word_ordinal,
            "raw_terms_per_column": checkpoint.shared.raw_terms_per_column,
            "end_to_end_milliseconds": started.elapsed().as_millis(),
            "passed": reports.iter().all(|report| report.passed),
        }),
    )?;
    Ok(reports)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_full_group_job(
    job: &crate::second_momentum_full_gpu_jobs::FullGpuJobKey,
    map_directory: &Path,
    output_directory: &Path,
    device: i32,
    cpu_parity_terms: usize,
    live_progress: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<GpuGroupJobRunReport, String> {
    run_group_job_request(
        GroupJobRequest::Full {
            job: job.clone(),
            map_directory: map_directory.to_path_buf(),
        },
        output_directory,
        device,
        cpu_parity_terms,
        live_progress,
    )
}

#[cfg(feature = "cuda")]
enum GroupJobRequest {
    Legacy(GpuGroupJobKey),
    Full {
        job: crate::second_momentum_full_gpu_jobs::FullGpuJobKey,
        map_directory: PathBuf,
    },
}

#[cfg(feature = "cuda")]
fn run_group_job_request(
    request: GroupJobRequest,
    output_directory: &Path,
    device: i32,
    cpu_parity_terms: usize,
    live_progress: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<GpuGroupJobRunReport, String> {
    use crate::eleven_dimensional_second_momentum_gpu::{
        CudaModularFx, GpuFxColumnInput, GpuFxColumnReport, ModularFunctionalColumn,
        ModularFxStaticData, PersistentCudaGroupExecutor, encode_modular_column, rank_columns,
    };
    use crate::second_momentum_gpu_group::{
        GroupRuntimeIdentity, GroupWordOrchestrationConfig, prepare_cuda_column_group,
        prepare_full_cuda_column_group,
    };
    use crate::second_momentum_gpu_progress::{
        GpuBatchProgress, GroupLiveProgress, SourceVisitorProgress,
    };

    if device < 0 || cpu_parity_terms == 0 {
        return Err(
            "group job requires a nonnegative device and nonzero parity prefix".to_string(),
        );
    }
    let started = std::time::Instant::now();
    let (job_id, work_manifest_sha256, group_index, prime_index, prime, legacy, map_directory) =
        match &request {
            GroupJobRequest::Legacy(job) => {
                let manifest = build_job_manifest()?;
                let _manifest_path = write_or_validate_manifest(output_directory)?;
                if validate_completed_job(output_directory, job)? {
                    return Err(format!(
                        "job {} is already complete and validated",
                        job.id()
                    ));
                }
                (
                    job.id(),
                    manifest.manifest_sha256,
                    job.group_index,
                    job.prime_index,
                    job.prime()?,
                    true,
                    None,
                )
            }
            GroupJobRequest::Full { job, map_directory } => {
                let manifest = crate::second_momentum_full_gpu_jobs::build_manifest()?;
                let _manifest_path =
                    crate::second_momentum_full_gpu_jobs::write_or_validate_manifest(
                        output_directory,
                    )?;
                if crate::second_momentum_full_gpu_jobs::validate_completed_job(
                    output_directory,
                    job,
                )? {
                    return Err(format!(
                        "job {} is already complete and validated",
                        job.id()
                    ));
                }
                (
                    job.id(),
                    manifest.manifest_sha256,
                    job.group_index,
                    job.prime_index,
                    job.prime(),
                    false,
                    Some(map_directory.clone()),
                )
            }
        };
    let job_directory = output_directory.join("jobs").join(&job_id);
    let run_schema = if legacy {
        GPU_GROUP_RUN_SCHEMA
    } else {
        crate::second_momentum_full_gpu_jobs::FULL_GPU_RUN_SCHEMA
    };
    fs::create_dir_all(&job_directory).map_err(|error| error.to_string())?;
    let _lock = acquire_group_job_lock(output_directory, &job_id)?;
    let event_log_path = job_directory.join("events.jsonl");
    let checkpoint_path = job_directory.join("checkpoint.json");
    let report_path = job_directory.join("job-report.json");
    let event_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&event_log_path)
        .map_err(|error| error.to_string())?;
    let mut event_log = BufWriter::new(event_file);

    let (legacy_tranche, local_ordinals, full_global_ordinals) = match &request {
        GroupJobRequest::Legacy(job) => (Some(job.tranche()?), job.local_ordinals()?, None),
        GroupJobRequest::Full { job, .. } => {
            let globals = job.global_ordinals();
            (None, globals.clone(), Some(globals))
        }
    };
    if legacy && local_ordinals.len() == 1 {
        return Err("singleton job must use the existing single-column fallback".to_string());
    }
    let static_started = std::time::Instant::now();
    let static_data = ModularFxStaticData::build(prime)?;
    let static_build_milliseconds = static_started.elapsed().as_millis();
    let probe = CudaModularFx::new(&static_data, device)?;
    let device_name = probe.device_name().to_string();
    let flat_plan_sha256 = probe.flat_plan_sha256().to_string();
    drop(probe);
    let runtime = GroupRuntimeIdentity {
        prime,
        static_semantic_sha256: static_data.semantic_sha256().to_string(),
        flat_plan_sha256: flat_plan_sha256.clone(),
    };
    let plan = if let Some(tranche) = legacy_tranche {
        prepare_cuda_column_group(tranche, &local_ordinals, runtime)?
    } else {
        prepare_full_cuda_column_group(
            full_global_ordinals
                .as_deref()
                .ok_or_else(|| "full GPU ordinals are missing".to_string())?,
            map_directory
                .as_deref()
                .ok_or_else(|| "full GPU map directory is missing".to_string())?,
            runtime,
        )?
    };
    let plan_sha256 = plan_sha256(&plan)?;
    let execution_config = GroupExecutionConfig::from_environment()?;
    let orchestration_host_cap = orchestration_host_cap_after_hash_scratch(
        execution_config.aggregate_host_payload_cap_bytes,
        execution_config.raw_batch_terms_per_lane,
    )?;

    let initial = if checkpoint_path.exists() {
        let checkpoint = read_checkpoint(&checkpoint_path)?;
        if checkpoint.job_id != job_id
            || checkpoint.group_id != plan.group_id
            || checkpoint.plan_sha256 != plan_sha256
            || checkpoint.next_word_ordinal > plan.pbw_word_count
        {
            return Err("GPU group checkpoint identity does not match this job".to_string());
        }
        checkpoint
    } else {
        let (source_hashers, packed_hashers) = new_stream_hashers(plan.active_columns)?;
        GroupJobCheckpoint {
            schema_version: GPU_GROUP_CHECKPOINT_SCHEMA.to_string(),
            job_id: job_id.clone(),
            group_id: plan.group_id.clone(),
            plan_sha256: plan_sha256.clone(),
            next_word_ordinal: 0,
            next_global_batch_ordinal: 0,
            batches_folded: 0,
            raw_terms_per_column: vec![0; plan.active_columns],
            source_hashers,
            packed_hashers,
            rows: vec![
                vec![
                    GaussianResidue::zero();
                    crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
                ];
                plan.active_columns
            ],
            timing: GroupTimingTotals::new(plan.active_columns),
            lowering_summaries: None,
            checkpoint_generation: 0,
        }
    };
    let resumed = initial.next_word_ordinal != 0;
    let resume_word_ordinal = initial.next_word_ordinal;
    let mut checkpoint = initial;
    let mut parity_terms = (0..plan.active_columns)
        .map(|_| Vec::with_capacity(cpu_parity_terms))
        .collect::<Vec<Vec<crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm>>>();
    let mut source_hash_scratch = Vec::new();
    let mut packed_hash_scratch = Vec::new();

    let mut executor = if let Some(map_directory) = map_directory.as_deref() {
        PersistentCudaGroupExecutor::new_full(
            plan.clone(),
            &static_data,
            device,
            execution_config.max_union_keys_per_batch,
            execution_config.aggregate_device_cap_bytes,
            execution_config.contraction_device_cap_bytes,
            execution_config.per_lane_host_staging_cap_bytes,
            execution_config.download_chunk_terms,
            map_directory,
        )?
    } else {
        PersistentCudaGroupExecutor::new(
            plan.clone(),
            &static_data,
            device,
            execution_config.max_union_keys_per_batch,
            execution_config.aggregate_device_cap_bytes,
            execution_config.contraction_device_cap_bytes,
            execution_config.per_lane_host_staging_cap_bytes,
            execution_config.download_chunk_terms,
        )?
    };
    executor.restore_columns(checkpoint.rows.clone(), checkpoint.batches_folded)?;
    let device_budget = executor.device_budget();
    if resumed {
        parity_terms = executor.collect_parity_prefix(cpu_parity_terms)?;
    }

    group_event_log_write(
        &mut event_log,
        &serde_json::json!({
            "schema_version": run_schema,
            "event": "job_start",
            "timestamp_unix_ms": unix_milliseconds(),
            "job_id": job_id,
            "group_id": plan.group_id,
            "tranche": plan.tranche,
            "local_ordinals": plan.ordered_local_ordinals,
            "global_ordinals": plan.ordered_global_ordinals,
            "source_copies": plan.ordered_source_copies,
            "prime": prime,
            "device": device,
            "device_name": device_name,
            "resume_word_ordinal": resume_word_ordinal,
            "total_words": plan.pbw_word_count,
            "execution_config": execution_config,
            "device_budget": device_budget,
        }),
    )?;

    for word_ordinal in checkpoint.next_word_ordinal..plan.pbw_word_count {
        group_event_log_write(
            &mut event_log,
            &serde_json::json!({
                "schema_version": run_schema,
                "event": "word_start",
                "timestamp_unix_ms": unix_milliseconds(),
                "job_id": job_id,
                "group_id": plan.group_id,
                "word_ordinal": word_ordinal,
                "word_total": plan.pbw_word_count,
            }),
        )?;
        let before_raw = checkpoint.raw_terms_per_column.clone();
        let first_global_batch_ordinal = checkpoint.next_global_batch_ordinal;
        let checkpoint_cell = std::cell::RefCell::new(&mut checkpoint);
        let orchestration = executor.run_word_synchronous_batched(
            GroupWordOrchestrationConfig {
                start_word_ordinal: word_ordinal,
                end_word_ordinal_exclusive: word_ordinal + 1,
                first_global_batch_ordinal,
                raw_batch_term_cap_per_lane: execution_config.raw_batch_terms_per_lane,
                max_union_keys_per_batch: execution_config.max_union_keys_per_batch,
                aggregate_host_payload_cap_bytes: orchestration_host_cap,
            },
            |lane, _, terms| {
                let mut borrowed = checkpoint_cell.borrow_mut();
                let checkpoint = &mut **borrowed;
                update_stream_hashers_batch(
                    &mut checkpoint.source_hashers[lane],
                    &mut checkpoint.packed_hashers[lane],
                    terms,
                    &mut source_hash_scratch,
                    &mut packed_hash_scratch,
                )?;
                let term_count = u64::try_from(terms.len())
                    .map_err(|_| "group source-term batch count exceeds u64".to_string())?;
                checkpoint.raw_terms_per_column[lane] = checkpoint.raw_terms_per_column[lane]
                    .checked_add(term_count)
                    .ok_or_else(|| "group source-term count overflow".to_string())?;
                if parity_terms[lane].len() < cpu_parity_terms {
                    let remaining = cpu_parity_terms - parity_terms[lane].len();
                    parity_terms[lane].extend(terms.iter().take(remaining).cloned());
                }
                Ok(())
            },
            |observation| {
                let mut borrowed = checkpoint_cell.borrow_mut();
                let checkpoint = &mut **borrowed;
                update_timing(&mut checkpoint.timing, observation)?;
                group_event_log_write(&mut event_log, observation)?;
                let cuda = observation
                    .cuda
                    .as_ref()
                    .ok_or_else(|| "group observation is missing CUDA timing".to_string())?;
                let batch_ms =
                    observation.union.union_milliseconds as f64 + cuda.total_milliseconds;
                live_progress.record_gpu_batch(GpuBatchProgress {
                    batches_completed: checkpoint.timing.union_batches,
                    last_batch_ms: batch_ms,
                    total_batch_ms: checkpoint.timing.union_milliseconds as f64
                        + checkpoint.timing.total_cuda_milliseconds,
                    last_upload_ms: cuda.upload_milliseconds,
                    total_upload_ms: checkpoint.timing.upload_milliseconds,
                    last_sort_ms: observation.union.union_milliseconds as f64,
                    total_sort_ms: checkpoint.timing.union_milliseconds as f64,
                    last_reduce_ms: 0.0,
                    total_reduce_ms: 0.0,
                    last_contract_ms: cuda.contract_milliseconds,
                    total_contract_ms: checkpoint.timing.contract_milliseconds,
                    last_download_ms: cuda.download_milliseconds,
                    total_download_ms: checkpoint.timing.download_milliseconds,
                });
                live_progress.update_source(SourceVisitorProgress {
                    word: Some(word_ordinal as u64),
                    root: observation.pbw_root.map(u64::from),
                    raw_terms_emitted: checkpoint.raw_terms_per_column.iter().sum(),
                    batches_flushed: checkpoint.timing.union_batches,
                    current_batch_terms: observation.raw_terms_per_column.iter().sum(),
                    current_batch_bytes: observation.union.host_capacity_bytes,
                    hard_memory_cap_bytes: execution_config.aggregate_host_payload_cap_bytes,
                    eta_sample_count: checkpoint.timing.union_batches,
                });
                live_progress.update_group(GroupLiveProgress {
                    group_id: Some(plan.group_id.clone()),
                    words_completed: word_ordinal,
                    words_total: plan.pbw_word_count,
                    global_batch_ordinal: observation.union.batch_ordinal,
                    raw_terms_per_column: checkpoint.raw_terms_per_column.clone(),
                    last_union_key_count: observation.union.union_key_count,
                    cumulative_union_keys: checkpoint.timing.union_keys,
                    keys_by_present_lane_count: observation
                        .union
                        .keys_by_present_lane_count
                        .clone(),
                    host_capacity_bytes: observation.union.host_capacity_bytes,
                    aggregate_host_cap_bytes: execution_config.aggregate_host_payload_cap_bytes,
                    device_resident_bytes: cuda.device_resident_bytes,
                    device_high_water_bytes: checkpoint.timing.device_high_water_bytes,
                    aggregate_device_cap_bytes: execution_config.aggregate_device_cap_bytes,
                    checkpoint_generation: checkpoint.checkpoint_generation,
                    checkpoint_sha256: None,
                    checkpoint_written_unix_ms: None,
                });
                Ok(())
            },
            |completed_word, completions| {
                if completed_word != word_ordinal || completions.len() != plan.active_columns {
                    return Err("group word completion identity changed".to_string());
                }
                Ok(())
            },
        )?;
        drop(checkpoint_cell);
        for lane in 0..plan.active_columns {
            let observed = checkpoint.raw_terms_per_column[lane] - before_raw[lane];
            if observed != orchestration.raw_terms_per_column[lane] {
                return Err("group word raw-term accounting mismatch".to_string());
            }
        }
        checkpoint.next_word_ordinal = orchestration.next_word_ordinal;
        checkpoint.next_global_batch_ordinal = orchestration.next_global_batch_ordinal;
        checkpoint.batches_folded = executor.batches_folded();
        checkpoint.rows = executor.final_columns().to_vec();
        checkpoint.checkpoint_generation = checkpoint
            .checkpoint_generation
            .checked_add(1)
            .ok_or_else(|| "checkpoint generation overflow".to_string())?;
        let checkpoint_sha256 = write_checkpoint(&checkpoint_path, &checkpoint)?;
        let written = unix_milliseconds();
        live_progress.update_group(GroupLiveProgress {
            group_id: Some(plan.group_id.clone()),
            words_completed: checkpoint.next_word_ordinal,
            words_total: plan.pbw_word_count,
            global_batch_ordinal: checkpoint.next_global_batch_ordinal,
            raw_terms_per_column: checkpoint.raw_terms_per_column.clone(),
            last_union_key_count: 0,
            cumulative_union_keys: checkpoint.timing.union_keys,
            keys_by_present_lane_count: Vec::new(),
            host_capacity_bytes: 0,
            aggregate_host_cap_bytes: execution_config.aggregate_host_payload_cap_bytes,
            device_resident_bytes: 0,
            device_high_water_bytes: checkpoint.timing.device_high_water_bytes,
            aggregate_device_cap_bytes: execution_config.aggregate_device_cap_bytes,
            checkpoint_generation: checkpoint.checkpoint_generation,
            checkpoint_sha256: Some(checkpoint_sha256.clone()),
            checkpoint_written_unix_ms: Some(written),
        });
        group_event_log_write(
            &mut event_log,
            &serde_json::json!({
                "schema_version": run_schema,
                "event": "word_checkpoint",
                "timestamp_unix_ms": written,
                "job_id": job_id,
                "group_id": plan.group_id,
                "completed_word_ordinal": word_ordinal,
                "next_word_ordinal": checkpoint.next_word_ordinal,
                "word_total": plan.pbw_word_count,
                "raw_terms_per_column": checkpoint.raw_terms_per_column,
                "batches_folded": checkpoint.batches_folded,
                "checkpoint_generation": checkpoint.checkpoint_generation,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_path": checkpoint_path,
            }),
        )?;
    }

    if checkpoint.next_word_ordinal != plan.pbw_word_count
        || parity_terms.iter().any(|terms| terms.is_empty())
    {
        return Err("group execution did not complete every word or parity prefix".to_string());
    }
    let mut parity_passed = vec![false; plan.active_columns];
    for lane in 0..plan.active_columns {
        let member = &plan.members[lane];
        let parity = GpuFxColumnInput {
            global_ordinal: member.global_ordinal,
            source_label: plan.source_dynkin_label.clone(),
            source_copy: member.source_copy,
            terms: parity_terms[lane].clone(),
            raising_residuals: [0; 5],
        };
        let cpu = crate::eleven_dimensional_second_momentum_gpu::accumulate_column_cpu(
            &static_data,
            &parity,
        )?;
        let mut cuda = CudaModularFx::new(&static_data, device)?;
        let gpu = cuda.accumulate(&parity)?.0;
        if cpu.rows != gpu.rows || cpu.semantic_sha256 != gpu.semantic_sha256 {
            return Err(format!(
                "CPU/CUDA parity failed for grouped lane {} on {} terms",
                member.global_ordinal,
                parity.terms.len()
            ));
        }
        parity_passed[lane] = true;
    }

    let lowering_summaries = executor.lowering_summaries()?;
    let rows = executor.final_columns().to_vec();
    let semantic_digests = executor.final_column_semantic_sha256();
    let checkpoint_sha256 = write_checkpoint(&checkpoint_path, &checkpoint)?;
    let mut column_reports = Vec::with_capacity(plan.active_columns);
    for lane in 0..plan.active_columns {
        let member = &plan.members[lane];
        let source_terms = checkpoint.raw_terms_per_column[lane];
        let source_terms_sha256 = finish_source_digest(
            &checkpoint.source_hashers[lane],
            member.global_ordinal,
            &plan.source_dynkin_label,
            member.source_copy,
            source_terms,
        )?;
        let packed_input_sha256 = finish_packed_digest(
            &checkpoint.packed_hashers[lane],
            member.global_ordinal,
            source_terms,
        )?;
        let modular_column = ModularFunctionalColumn {
            prime,
            global_ordinal: member.global_ordinal,
            rows: rows[lane].clone(),
            expanded_contributions: checkpoint.timing.expanded_contributions_per_column[lane],
            semantic_sha256: semantic_digests[lane].clone(),
        };
        let rank = rank_columns(std::slice::from_ref(&modular_column))?;
        let nonzero_functional_rows = modular_column
            .rows
            .iter()
            .filter(|value| !value.is_zero())
            .count();
        let stem = format!(
            "second_momentum_{}_column_{:02}_p{}",
            plan.tranche, member.global_ordinal, prime
        );
        let binary_path = output_directory.join(format!("{stem}.bin"));
        let binary = encode_modular_column(
            &modular_column,
            static_data.semantic_sha256(),
            &source_terms_sha256,
            source_terms,
        );
        publish_conflict_safe(&binary_path, &binary)?;
        let binary_sha256 = format!("{:x}", Sha256::digest(&binary));
        let summary = lowering_summaries[lane];
        let report = GpuFxColumnReport {
            schema_version: crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA,
            tranche: plan.tranche.clone(),
            local_ordinal: member.local_ordinal,
            global_ordinal: member.global_ordinal,
            source_label: plan.source_dynkin_label.clone(),
            source_copy: member.source_copy,
            prime,
            functional_seeds: crate::eleven_dimensional_second_momentum_gpu::GPU_FX_FUNCTIONAL_SEEDS,
            functional_row_count: crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT,
            device_name: device_name.clone(),
            static_semantic_sha256: static_data.semantic_sha256().to_string(),
            flat_plan_sha256: flat_plan_sha256.clone(),
            source_terms: usize::try_from(source_terms)
                .map_err(|_| "source term count does not fit usize".to_string())?,
            source_terms_sha256,
            expanded_contributions: checkpoint.timing.expanded_contributions_per_column[lane],
            nonzero_functional_rows,
            column_semantic_sha256: modular_column.semantic_sha256,
            binary_path: binary_path.display().to_string(),
            binary_sha256,
            binary_bytes: binary.len() as u64,
            source_build_milliseconds: started.elapsed().as_millis()
                .saturating_sub(static_build_milliseconds),
            static_build_milliseconds,
            cuda_kernel_milliseconds: checkpoint.timing.total_cuda_milliseconds as f32,
            cuda_upload_milliseconds: checkpoint.timing.upload_milliseconds as f32,
            cuda_sort_milliseconds: checkpoint.timing.union_milliseconds as f32,
            cuda_reduce_milliseconds: 0.0,
            cuda_contract_milliseconds: checkpoint.timing.contract_milliseconds as f32,
            cuda_download_milliseconds: checkpoint.timing.download_milliseconds as f32,
            batch_reduced_key_visits: checkpoint.timing.reduced_key_visits_per_column[lane],
            batch_nonzero_reduced_term_visits: checkpoint
                .timing
                .nonzero_reduced_term_visits_per_column[lane],
            cuda_buffer_high_water_bytes: checkpoint.timing.device_high_water_bytes,
            packed_recoupling_input_sha256: packed_input_sha256,
            cuda_input_terms_per_second: if checkpoint.timing.total_cuda_milliseconds > 0.0 {
                source_terms as f64 * 1_000.0 / checkpoint.timing.total_cuda_milliseconds
            } else {
                0.0
            },
            cuda_batches: checkpoint.timing.union_batches,
            cuda_peak_batch_terms: checkpoint.timing.peak_union_keys,
            cuda_batch_term_cap: execution_config.max_union_keys_per_batch,
            cuda_host_hard_cap_bytes: execution_config.aggregate_host_payload_cap_bytes,
            cuda_device_hard_cap_bytes: execution_config.contraction_device_cap_bytes,
            cuda_total_device_hard_cap_bytes: execution_config.aggregate_device_cap_bytes,
            persistent_lowering_enabled: summary.enabled,
            persistent_lowering_roots: summary.roots_lowered,
            persistent_lowering_input_entry_visits: summary.input_entry_visits,
            persistent_lowering_expanded_entry_visits: summary.expanded_entry_visits,
            persistent_lowering_output_entry_visits: summary.output_entry_visits,
            persistent_lowering_gpu_milliseconds: summary.gpu_milliseconds,
            persistent_lowering_high_water_bytes: summary.scratch_high_water_bytes,
            persistent_lowering_peak_output_handle_bytes: summary.peak_immutable_handle_bytes,
            persistent_lowering_maximum_absolute_coefficient: summary.maximum_absolute_coefficient,
            persistent_lowering_device_hard_cap_bytes: summary.device_hard_cap_bytes,
            persistent_lowering_download_chunk_terms: summary.download_chunk_terms,
            cpu_parity_terms: parity_terms[lane].len(),
            cpu_parity_passed: parity_passed[lane],
            end_to_end_milliseconds: started.elapsed().as_millis(),
            raising_residuals: [0; 5],
            highest_weight_certification: "Exact source, embedded-map, reciprocal-map, PBW-plan, and grouped lane identities were preflighted before execution.".to_string(),
            direct_composed_raising_residuals_materialized: false,
            single_column_rank: rank.rank_over_gaussian_extension,
            passed: rank.rank_over_gaussian_extension == 1
                && nonzero_functional_rows != 0
                && parity_passed[lane],
            proof_boundary: "Every lane preserves its original exact term order for hashing. Bounded exact per-lane reduction and canonical key union are additive, CUDA coefficients and rows remain lane separated, and every word is checkpointed only after all lane deltas are folded. Full modular rank gives the declared characteristic-zero lower bound when denominators are invertible.".to_string(),
        };
        if !report.passed {
            return Err(format!(
                "grouped column {} failed rank or parity gates",
                member.global_ordinal
            ));
        }
        let json_path = output_directory.join(format!("{stem}.json"));
        let mut json = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
        json.push(b'\n');
        if json_path.exists() {
            let existing: serde_json::Value =
                serde_json::from_slice(&fs::read(&json_path).map_err(|error| error.to_string())?)
                    .map_err(|error| error.to_string())?;
            let matches = existing.get("passed").and_then(serde_json::Value::as_bool) == Some(true)
                && existing
                    .get("binary_sha256")
                    .and_then(serde_json::Value::as_str)
                    == Some(report.binary_sha256.as_str())
                && existing
                    .get("column_semantic_sha256")
                    .and_then(serde_json::Value::as_str)
                    == Some(report.column_semantic_sha256.as_str());
            if !matches {
                return Err(format!(
                    "refusing to replace differing published report {}",
                    json_path.display()
                ));
            }
        } else {
            write_atomic_durable(&json_path, &json)?;
        }
        column_reports.push(report);
    }

    let passed = column_reports.iter().all(|report| report.passed)
        && checkpoint.next_word_ordinal == plan.pbw_word_count;
    let artifact_inventory = column_reports
        .iter()
        .map(|column| {
            let binary_relative_path = Path::new(&column.binary_path)
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| "group binary path has no portable file name".to_string())?
                .to_owned();
            Ok(PublishedColumnArtifact {
                global_ordinal: column.global_ordinal,
                report_relative_path: format!(
                    "second_momentum_{}_column_{:02}_p{}.json",
                    column.tranche, column.global_ordinal, column.prime
                ),
                binary_relative_path,
                binary_sha256: column.binary_sha256.clone(),
                column_semantic_sha256: column.column_semantic_sha256.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let report = GpuGroupJobRunReport {
        schema_version: if legacy {
            GPU_GROUP_RUN_SCHEMA.to_string()
        } else {
            crate::second_momentum_full_gpu_jobs::FULL_GPU_RUN_SCHEMA.to_string()
        },
        job_id: job_id.clone(),
        work_manifest_sha256,
        group_id: plan.group_id.clone(),
        plan_sha256,
        tranche: plan.tranche.clone(),
        group_index,
        prime_index,
        prime,
        device,
        device_name,
        resumed,
        resume_word_ordinal,
        completed_words: checkpoint.next_word_ordinal,
        total_words: plan.pbw_word_count,
        next_global_batch_ordinal: checkpoint.next_global_batch_ordinal,
        checkpoint_path: checkpoint_path.display().to_string(),
        checkpoint_sha256,
        event_log_path: event_log_path.display().to_string(),
        execution_config,
        device_budget,
        timing: checkpoint.timing.clone(),
        raw_terms_per_column: checkpoint.raw_terms_per_column.clone(),
        lowering_summaries: serde_json::to_value(&lowering_summaries)
            .map_err(|error| error.to_string())?,
        column_reports,
        artifact_inventory,
        end_to_end_milliseconds: started.elapsed().as_millis(),
        passed,
        proof_boundary: "The group report is the commit record. Individual binaries are written first with conflict checks, individual reports follow, and this job report is published last. A validated report can be adopted by any worker without machine ownership state.".to_string(),
    };
    if !report.passed {
        return Err("GPU group job failed final publication gates".to_string());
    }
    let mut bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    publish_conflict_safe(&report_path, &bytes)?;
    group_event_log_write(
        &mut event_log,
        &serde_json::json!({
            "schema_version": run_schema,
            "event": "job_complete",
            "timestamp_unix_ms": unix_milliseconds(),
            "job_id": job_id,
            "group_id": plan.group_id,
            "report_path": report_path,
            "passed": true,
            "end_to_end_milliseconds": report.end_to_end_milliseconds,
        }),
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_manifest_has_twelve_groups_and_thirty_six_prime_jobs() {
        let manifest = build_job_manifest().unwrap();
        assert_eq!(manifest.jobs.len(), 36);
        let p0 = manifest
            .jobs
            .iter()
            .filter(|job| job.prime_index == 0)
            .collect::<Vec<_>>();
        assert_eq!(p0.len(), 12);
        assert_eq!(p0.iter().filter(|job| job.width == 3).count(), 3);
        assert_eq!(p0.iter().filter(|job| job.width == 2).count(), 6);
        assert_eq!(p0.iter().filter(|job| job.width == 1).count(), 3);
        validate_manifest(&manifest).unwrap();
    }

    #[test]
    fn list_parser_supports_machine_sized_slices() {
        assert_eq!(parse_job_list("20001@0").unwrap().len(), 4);
        assert_eq!(parse_job_list("30001@0").unwrap().len(), 8);
        assert_eq!(parse_job_list("all@0").unwrap().len(), 12);
        let selected = parse_job_list("20001-g2-p0,30001-g7-p0").unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].id(), "20001-g2-p0");
        assert_eq!(selected[1].id(), "30001-g7-p0");
        assert!(parse_job_list("30001-g99-p0").is_err());
    }

    #[test]
    fn serializable_sha256_matches_sha2_at_every_chunk_boundary() {
        let bytes = (0..4097)
            .map(|index| (index * 73 + 19) as u8)
            .collect::<Vec<_>>();
        let expected = format!("{:x}", Sha256::digest(&bytes));
        for chunk in [1, 2, 3, 7, 31, 63, 64, 65, 127, 1024] {
            let mut observed = SerializableSha256::new();
            for part in bytes.chunks(chunk) {
                observed.update(part).unwrap();
                let encoded = serde_json::to_vec(&observed).unwrap();
                observed = serde_json::from_slice(&encoded).unwrap();
            }
            assert_eq!(observed.finalize_hex().unwrap(), expected, "chunk {chunk}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn contiguous_term_hash_records_preserve_the_legacy_byte_contract() {
        let term = crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm {
            momentum_pair: [3, 9],
            free_spinor: 17,
            exterior_mask: 0x0000_0fff,
            coefficient: -(1_i128 << 99) + 0x1234_5678,
        };
        let (mut source, mut packed) = new_stream_hashers(1).unwrap();
        update_stream_hashers(&mut source[0], &mut packed[0], &term).unwrap();

        let (mut legacy_source, mut legacy_packed) = new_stream_hashers(1).unwrap();
        let key = pack_job_term(&term).unwrap();
        legacy_source[0].update(&term.momentum_pair).unwrap();
        legacy_source[0].update(&[term.free_spinor]).unwrap();
        legacy_source[0]
            .update(&term.exterior_mask.to_le_bytes())
            .unwrap();
        legacy_source[0]
            .update(&term.coefficient.to_le_bytes())
            .unwrap();
        legacy_packed[0].update(&key.to_le_bytes()).unwrap();
        legacy_packed[0]
            .update(&(term.coefficient as u128 as u64).to_le_bytes())
            .unwrap();
        legacy_packed[0]
            .update(&((term.coefficient >> 64) as i64).to_le_bytes())
            .unwrap();

        assert_eq!(
            source[0].finalize_hex().unwrap(),
            legacy_source[0].finalize_hex().unwrap()
        );
        assert_eq!(
            packed[0].finalize_hex().unwrap(),
            legacy_packed[0].finalize_hex().unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn batched_term_hashing_is_byte_identical_at_irregular_boundaries() {
        let terms = (0..257)
            .map(|index| {
                let left = (index % 10) as u8;
                crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm {
                    momentum_pair: [left, left + 1],
                    free_spinor: (index % 32) as u8,
                    exterior_mask: ((1_u32 << 12) - 1) << (index % 8),
                    coefficient: i128::from(index + 1) * if index % 2 == 0 { 1 } else { -1 },
                }
            })
            .collect::<Vec<_>>();
        let (mut scalar_source, mut scalar_packed) = new_stream_hashers(1).unwrap();
        for term in &terms {
            update_stream_hashers(&mut scalar_source[0], &mut scalar_packed[0], term).unwrap();
        }
        for chunk in [1, 2, 3, 7, 31, 64, 127, 257] {
            let (mut source, mut packed) = new_stream_hashers(1).unwrap();
            let mut source_scratch = Vec::new();
            let mut packed_scratch = Vec::new();
            for batch in terms.chunks(chunk) {
                update_stream_hashers_batch(
                    &mut source[0],
                    &mut packed[0],
                    batch,
                    &mut source_scratch,
                    &mut packed_scratch,
                )
                .unwrap();
            }
            assert_eq!(
                source[0].finalize_hex().unwrap(),
                scalar_source[0].finalize_hex().unwrap()
            );
            assert_eq!(
                packed[0].finalize_hex().unwrap(),
                scalar_packed[0].finalize_hex().unwrap()
            );
        }
    }

    #[test]
    fn checkpoint_detects_corruption_and_restores_hash_state() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-gpu-group-checkpoint-test-{}-{}",
            std::process::id(),
            unix_milliseconds()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("job.chk");
        let mut source = SerializableSha256::new();
        source.update(b"prefix").unwrap();
        let checkpoint = GroupJobCheckpoint {
            schema_version: GPU_GROUP_CHECKPOINT_SCHEMA.to_string(),
            job_id: "20001-g0-p0".to_string(),
            group_id: "a".repeat(64),
            plan_sha256: "b".repeat(64),
            next_word_ordinal: 1,
            next_global_batch_ordinal: 7,
            batches_folded: 7,
            raw_terms_per_column: vec![4, 5],
            source_hashers: vec![source.clone(), source.clone()],
            packed_hashers: vec![source.clone(), source],
            rows: vec![
                vec![
                    GaussianResidue::zero();
                    crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
                ],
                vec![
                    GaussianResidue::zero();
                    crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
                ],
            ],
            timing: GroupTimingTotals::new(2),
            lowering_summaries: None,
            checkpoint_generation: 1,
        };
        write_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_checkpoint(&path).unwrap();
        assert_eq!(restored, checkpoint);

        // A valid legacy writer may use a different exact decimal spelling
        // for an equivalent f64. The digest binds those original bytes, so the
        // reader must not parse and reserialize before checking it.
        let original = fs::read_to_string(&path).unwrap();
        let legacy_spelling = original.replacen(
            "\"upload_milliseconds\":0.0",
            "\"upload_milliseconds\":0.00000000000000000",
            1,
        );
        assert_ne!(legacy_spelling, original);
        let raw_envelope: RawGroupJobCheckpointEnvelope =
            serde_json::from_str(&legacy_spelling).unwrap();
        let legacy_digest = format!(
            "{:x}",
            Sha256::digest(raw_envelope.checkpoint.get().as_bytes())
        );
        let legacy_spelling = legacy_spelling.replacen(
            &format!("\"payload_sha256\":\"{}\"", raw_envelope.payload_sha256),
            &format!("\"payload_sha256\":\"{legacy_digest}\""),
            1,
        );
        fs::write(&path, &legacy_spelling).unwrap();
        assert_eq!(read_checkpoint(&path).unwrap(), checkpoint);

        let undigested_spelling_change = legacy_spelling.replacen(
            "\"upload_milliseconds\":0.00000000000000000",
            "\"upload_milliseconds\":0.000000000000000000",
            1,
        );
        fs::write(&path, undigested_spelling_change).unwrap();
        assert!(
            read_checkpoint(&path)
                .unwrap_err()
                .contains("payload digest mismatch")
        );

        fs::write(&path, legacy_spelling).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let index = bytes.len() / 2;
        bytes[index] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(read_checkpoint(&path).is_err());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn completed_job_inventory_survives_directory_transfer() {
        let root = std::env::temp_dir().join(format!(
            "adynkra-gpu-group-transfer-test-{}-{}",
            std::process::id(),
            unix_milliseconds()
        ));
        let directory = root.join("source");
        let destination = root.join("destination");
        let job = GpuGroupJobKey::parse_id("20001-g0-p0").unwrap();
        let binary_name = "portable-column.bin";
        let column_report_name = "portable-column.json";
        let binary = b"portable exact artifact";
        fs::create_dir_all(
            completed_job_report_path(&directory, &job)
                .parent()
                .unwrap(),
        )
        .unwrap();
        fs::write(directory.join(binary_name), binary).unwrap();
        write_or_validate_manifest(&directory).unwrap();
        let digest = format!("{:x}", Sha256::digest(binary));
        fs::write(
            directory.join(column_report_name),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": "column-v1",
                "global_ordinal": 53,
                "prime": GPU_FX_PRIMES[0],
                "binary_sha256": digest,
                "column_semantic_sha256": "c".repeat(64),
                "passed": true,
            }))
            .unwrap(),
        )
        .unwrap();
        let report = serde_json::json!({
            "schema_version": GPU_GROUP_RUN_SCHEMA,
            "job_id": job.id(),
            "work_manifest_sha256": build_job_manifest().unwrap().manifest_sha256,
            "passed": true,
            "column_reports": [],
            "artifact_inventory": [{
                "global_ordinal": 53,
                "binary_relative_path": binary_name,
                "report_relative_path": column_report_name,
                "binary_sha256": digest,
                "column_semantic_sha256": "c".repeat(64),
            }]
        });
        fs::write(
            completed_job_report_path(&directory, &job),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        assert_eq!(validate_completed_job(&directory, &job), Ok(true));
        let imported =
            import_completed_jobs(&directory, &destination, std::slice::from_ref(&job)).unwrap();
        assert_eq!(imported["imported"], serde_json::json!([job.id()]));
        assert_eq!(validate_completed_job(&destination, &job), Ok(true));
        assert!(safe_relative_artifact_path(&directory, "../escape.bin").is_err());
        let _ = fs::remove_dir_all(root);
    }
}
