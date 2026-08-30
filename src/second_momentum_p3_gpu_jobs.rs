//! Portable production inventory and fail-closed publication for p^3 D^11.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "cuda")]
use crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput;
use crate::eleven_dimensional_second_momentum_gpu::{
    GPU_FX_PRIMES, GPU_P3_FX_SCHEMA, ModularP3FunctionalColumn, ModularRankCertificate,
    P3_FUNCTIONAL_ROW_COUNT, decode_p3_column_artifact, encode_p3_column_artifact, rank_p3_columns,
};

pub(crate) const P3_JOB_SCHEMA: &str = "adynkra-11d-second-momentum-p3-jobs-v1";
pub(crate) const P3_CHECKPOINT_SCHEMA: &str = "adynkra-11d-second-momentum-p3-checkpoint-v1";
pub(crate) const P3_RUN_SCHEMA: &str = "adynkra-11d-second-momentum-p3-run-v1";
pub(crate) const P3_STATUS_SCHEMA: &str = "adynkra-11d-second-momentum-p3-status-v1";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3JobKey {
    pub global_ordinal: usize,
    pub prime_index: usize,
}

impl P3JobKey {
    pub(crate) fn new(global_ordinal: usize, prime_index: usize) -> Result<Self, String> {
        if global_ordinal >= 77 || prime_index >= GPU_FX_PRIMES.len() {
            return Err("p3 job index is out of range".to_string());
        }
        Ok(Self {
            global_ordinal,
            prime_index,
        })
    }

    pub(crate) fn id(&self) -> String {
        format!("p3-c{}-p{}", self.global_ordinal, self.prime_index)
    }

    pub(crate) fn prime(&self) -> u32 {
        GPU_FX_PRIMES[self.prime_index]
    }

    pub(crate) fn parse_id(value: &str) -> Result<Self, String> {
        let suffix = value
            .strip_prefix("p3-c")
            .ok_or_else(|| "p3 job ID must begin with p3-c".to_string())?;
        let (column, prime) = suffix
            .split_once("-p")
            .ok_or_else(|| "p3 job ID requires -p<index>".to_string())?;
        Self::new(
            column
                .parse()
                .map_err(|_| "p3 column index is not an integer".to_string())?,
            prime
                .parse()
                .map_err(|_| "p3 prime index is not an integer".to_string())?,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3JobManifest {
    pub schema_version: String,
    pub row_schema_version: String,
    pub physical_columns: usize,
    pub primes: Vec<u32>,
    pub jobs: Vec<P3JobKey>,
    pub manifest_sha256: String,
}

fn manifest_digest(manifest: &P3JobManifest) -> Result<String, String> {
    let mut copy = manifest.clone();
    copy.manifest_sha256.clear();
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&copy).map_err(|error| error.to_string())?)
    ))
}

pub(crate) fn build_manifest() -> Result<P3JobManifest, String> {
    let mut jobs = Vec::with_capacity(77 * GPU_FX_PRIMES.len());
    for global_ordinal in 0..77 {
        for prime_index in 0..GPU_FX_PRIMES.len() {
            jobs.push(P3JobKey::new(global_ordinal, prime_index)?);
        }
    }
    let mut manifest = P3JobManifest {
        schema_version: P3_JOB_SCHEMA.to_string(),
        row_schema_version: GPU_P3_FX_SCHEMA.to_string(),
        physical_columns: 77,
        primes: GPU_FX_PRIMES.to_vec(),
        jobs,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

pub(crate) fn validate_manifest(manifest: &P3JobManifest) -> Result<(), String> {
    if manifest.schema_version != P3_JOB_SCHEMA
        || manifest.row_schema_version != GPU_P3_FX_SCHEMA
        || manifest.physical_columns != 77
        || manifest.primes != GPU_FX_PRIMES
        || manifest.jobs.len() != 77 * GPU_FX_PRIMES.len()
        || manifest.manifest_sha256 != manifest_digest(manifest)?
    {
        return Err("p3 manifest schema, inventory, or digest is invalid".to_string());
    }
    let expected = (0..77)
        .flat_map(|column| {
            (0..GPU_FX_PRIMES.len()).map(move |prime| P3JobKey {
                global_ordinal: column,
                prime_index: prime,
            })
        })
        .collect::<Vec<_>>();
    if manifest.jobs != expected {
        return Err("p3 manifest differs from the canonical all-77 inventory".to_string());
    }
    Ok(())
}

fn parse_range(value: &str) -> Result<Vec<usize>, String> {
    if value == "all" {
        return Ok((0..77).collect());
    }
    let mut output = Vec::new();
    for part in value.split(',') {
        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start
                .parse()
                .map_err(|_| "invalid p3 range start".to_string())?;
            let end: usize = end
                .parse()
                .map_err(|_| "invalid p3 range end".to_string())?;
            if start > end || end >= 77 {
                return Err("p3 column range is out of bounds".to_string());
            }
            output.extend(start..=end);
        } else {
            let ordinal: usize = part
                .parse()
                .map_err(|_| "invalid p3 column selector".to_string())?;
            if ordinal >= 77 {
                return Err("p3 column selector is out of bounds".to_string());
            }
            output.push(ordinal);
        }
    }
    output.sort_unstable();
    output.dedup();
    Ok(output)
}

/// Accept explicit IDs, `all`, `all@0`, or a portable column range such as
/// `0-18,23,40-76@2`.
pub(crate) fn parse_job_list(value: &str) -> Result<Vec<P3JobKey>, String> {
    let mut jobs = if value.starts_with("p3-c") {
        value
            .split(',')
            .map(P3JobKey::parse_id)
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some((columns, prime)) = value.rsplit_once('@') {
        let prime_index: usize = prime
            .parse()
            .map_err(|_| "p3 @prime selector is not an integer".to_string())?;
        parse_range(columns)?
            .into_iter()
            .map(|column| P3JobKey::new(column, prime_index))
            .collect::<Result<Vec<_>, _>>()?
    } else if value == "all" {
        build_manifest()?.jobs
    } else {
        return Err(
            "p3 selection requires explicit IDs or a column selector with @prime".to_string(),
        );
    };
    jobs.sort();
    jobs.dedup();
    if jobs.is_empty() {
        return Err("p3 job selection is empty".to_string());
    }
    Ok(jobs)
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
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
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    atomic_bytes(path, &bytes)
}

fn append_event(path: &Path, value: serde_json::Value) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    writer
        .get_ref()
        .sync_data()
        .map_err(|error| error.to_string())
}

pub(crate) fn write_or_validate_manifest(output: &Path) -> Result<PathBuf, String> {
    let expected = build_manifest()?;
    let path = output.join("p3-work-manifest.json");
    if path.exists() {
        let observed: P3JobManifest =
            serde_json::from_reader(File::open(&path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        validate_manifest(&observed)?;
        if observed != expected {
            return Err("existing p3 manifest differs from this build".to_string());
        }
    } else {
        atomic_json(&path, &expected)?;
    }
    Ok(path)
}

fn job_directory(output: &Path, job: &P3JobKey) -> PathBuf {
    output.join("jobs").join(job.id())
}

pub(crate) fn artifact_path(output: &Path, job: &P3JobKey) -> PathBuf {
    job_directory(output, job).join("column.adfxp3")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3JobCheckpoint {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub job: P3JobKey,
    pub prime: u32,
    pub input_sha256: String,
    pub flat_plan_sha256: String,
    pub state: String,
    pub generation: u64,
    pub artifact_sha256: Option<String>,
    pub column_semantic_sha256: Option<String>,
    pub expanded_contributions: Option<u64>,
    pub source_count: Option<usize>,
    pub plan_entry_count: Option<u32>,
    pub kernel_milliseconds: Option<f32>,
    pub resident_bytes: Option<u64>,
    pub buffer_high_water_bytes: Option<u64>,
    pub device_hard_cap_bytes: Option<u64>,
    pub updated_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3JobReport {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub job: P3JobKey,
    pub prime: u32,
    pub input_sha256: String,
    pub flat_plan_sha256: String,
    pub artifact_relative_path: String,
    pub artifact_sha256: String,
    pub column_semantic_sha256: String,
    pub expanded_contributions: u64,
    pub source_count: usize,
    pub plan_entry_count: u32,
    pub kernel_milliseconds: f32,
    pub resident_bytes: u64,
    pub buffer_high_water_bytes: u64,
    pub device_hard_cap_bytes: u64,
    pub completed_unix_ms: u128,
    pub passed: bool,
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn relative_artifact_path(job: &P3JobKey) -> String {
    format!("jobs/{}/column.adfxp3", job.id())
}

#[cfg(feature = "cuda")]
struct P3JobLock {
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
    fn p3_job_flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
}

#[cfg(all(feature = "cuda", unix))]
impl Drop for P3JobLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe {
            p3_job_flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}

#[cfg(all(feature = "cuda", not(unix)))]
impl Drop for P3JobLock {
    fn drop(&mut self) {}
}

#[cfg(feature = "cuda")]
fn acquire_job_lock(output: &Path, job: &P3JobKey) -> Result<P3JobLock, String> {
    let directory = output.join(".locks");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(format!("{}.lock", job.id()));
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::io::{Seek, SeekFrom};
        use std::os::fd::AsRawFd;
        if unsafe { p3_job_flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) } != 0 {
            return Err(format!(
                "another live worker owns p3 job {}: {}",
                job.id(),
                std::io::Error::last_os_error()
            ));
        }
        file.set_len(0).map_err(|error| error.to_string())?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| error.to_string())?;
        writeln!(
            file,
            "{{\"job_id\":\"{}\",\"pid\":{},\"started_unix_ms\":{}}}",
            job.id(),
            std::process::id(),
            unix_ms()
        )
        .map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
    }
    Ok(P3JobLock { file })
}

fn validate_checkpoint_identity(
    checkpoint: &P3JobCheckpoint,
    expected: &P3JobCheckpoint,
) -> Result<(), String> {
    if checkpoint.schema_version != P3_CHECKPOINT_SCHEMA
        || checkpoint.manifest_sha256 != expected.manifest_sha256
        || checkpoint.job != expected.job
        || checkpoint.prime != expected.prime
        || checkpoint.input_sha256 != expected.input_sha256
        || checkpoint.flat_plan_sha256 != expected.flat_plan_sha256
    {
        return Err("existing p3 checkpoint is stale or incompatible".to_string());
    }
    Ok(())
}

fn checkpoint_report(
    checkpoint: &P3JobCheckpoint,
    completed_unix_ms: u128,
) -> Result<P3JobReport, String> {
    if checkpoint.state != "artifact_published" {
        return Err("p3 checkpoint is not publish-complete".to_string());
    }
    Ok(P3JobReport {
        schema_version: P3_RUN_SCHEMA.to_string(),
        manifest_sha256: checkpoint.manifest_sha256.clone(),
        job: checkpoint.job.clone(),
        prime: checkpoint.prime,
        input_sha256: checkpoint.input_sha256.clone(),
        flat_plan_sha256: checkpoint.flat_plan_sha256.clone(),
        artifact_relative_path: relative_artifact_path(&checkpoint.job),
        artifact_sha256: checkpoint
            .artifact_sha256
            .clone()
            .ok_or_else(|| "published p3 checkpoint lacks artifact digest".to_string())?,
        column_semantic_sha256: checkpoint
            .column_semantic_sha256
            .clone()
            .ok_or_else(|| "published p3 checkpoint lacks column digest".to_string())?,
        expanded_contributions: checkpoint
            .expanded_contributions
            .ok_or_else(|| "published p3 checkpoint lacks expanded count".to_string())?,
        source_count: checkpoint
            .source_count
            .ok_or_else(|| "published p3 checkpoint lacks source count".to_string())?,
        plan_entry_count: checkpoint
            .plan_entry_count
            .ok_or_else(|| "published p3 checkpoint lacks plan entry count".to_string())?,
        kernel_milliseconds: checkpoint
            .kernel_milliseconds
            .ok_or_else(|| "published p3 checkpoint lacks kernel timing".to_string())?,
        resident_bytes: checkpoint
            .resident_bytes
            .ok_or_else(|| "published p3 checkpoint lacks resident bytes".to_string())?,
        buffer_high_water_bytes: checkpoint
            .buffer_high_water_bytes
            .ok_or_else(|| "published p3 checkpoint lacks high-water bytes".to_string())?,
        device_hard_cap_bytes: checkpoint
            .device_hard_cap_bytes
            .ok_or_else(|| "published p3 checkpoint lacks device cap".to_string())?,
        completed_unix_ms,
        passed: true,
    })
}

fn publish_report_from_checkpoint(
    output: &Path,
    job: &P3JobKey,
    checkpoint: &P3JobCheckpoint,
    events: &Path,
) -> Result<P3JobReport, String> {
    let report = checkpoint_report(checkpoint, unix_ms())?;
    validate_report_artifact(output, job, &report)?;
    atomic_json(&job_directory(output, job).join("job-report.json"), &report)?;
    append_event(
        events,
        serde_json::json!({
            "schema_version": P3_RUN_SCHEMA,
            "event": "job_adopted_from_checkpoint",
            "timestamp_unix_ms": unix_ms(),
            "job_id": job.id(),
            "checkpoint_generation": checkpoint.generation,
            "artifact_sha256": report.artifact_sha256,
        }),
    )?;
    Ok(report)
}

fn validate_report_artifact(
    output: &Path,
    job: &P3JobKey,
    report: &P3JobReport,
) -> Result<(), String> {
    let manifest = build_manifest()?;
    if report.schema_version != P3_RUN_SCHEMA
        || report.manifest_sha256 != manifest.manifest_sha256
        || report.job != *job
        || report.prime != job.prime()
        || report.artifact_relative_path != relative_artifact_path(job)
        || !report.passed
    {
        return Err("p3 completed report identity is invalid".to_string());
    }
    let bytes = fs::read(artifact_path(output, job)).map_err(|error| error.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != report.artifact_sha256 {
        return Err("p3 completed artifact SHA-256 mismatch".to_string());
    }
    let (plan, column) = decode_p3_column_artifact(&bytes)?;
    if plan != report.flat_plan_sha256
        || column.prime != job.prime()
        || column.global_ordinal != job.global_ordinal
        || column.semantic_sha256 != report.column_semantic_sha256
        || column.expanded_contributions != report.expanded_contributions
    {
        return Err("p3 completed artifact semantic identity is invalid".to_string());
    }
    Ok(())
}

pub(crate) fn validate_completed_job(output: &Path, job: &P3JobKey) -> Result<bool, String> {
    let report_path = job_directory(output, job).join("job-report.json");
    if !report_path.exists() {
        return Ok(false);
    }
    let report: P3JobReport =
        serde_json::from_reader(File::open(&report_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    validate_report_artifact(output, job, &report)?;
    Ok(true)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_job(
    job: &P3JobKey,
    input: &GpuFxColumnInput,
    output: &Path,
    device: i32,
    device_hard_cap_bytes: u64,
    live: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<P3JobReport, String> {
    use crate::eleven_dimensional_second_momentum_gpu::{
        CudaModularP3, ModularFxStaticData, build_p3_modular_flat_plan,
    };
    use crate::second_momentum_gpu_progress::{
        GpuBatchProgress, GroupLiveProgress, SourceVisitorProgress,
    };

    if input.global_ordinal != job.global_ordinal || input.raising_residuals != [0; 5] {
        return Err("p3 job input identity is invalid".to_string());
    }
    write_or_validate_manifest(output)?;
    let _lock = acquire_job_lock(output, job)?;
    if validate_completed_job(output, job)? {
        return serde_json::from_reader(
            File::open(job_directory(output, job).join("job-report.json"))
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string());
    }
    let manifest = build_manifest()?;
    let directory = job_directory(output, job);
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let events = directory.join("events.jsonl");
    let checkpoint_path = directory.join("checkpoint.json");
    let input_bytes = serde_json::to_vec(input).map_err(|error| error.to_string())?;
    let input_sha256 = format!("{:x}", Sha256::digest(&input_bytes));
    append_event(
        &events,
        serde_json::json!({
            "schema_version": P3_RUN_SCHEMA,
            "event": "job_start",
            "timestamp_unix_ms": unix_ms(),
            "job_id": job.id(),
            "pid": std::process::id(),
            "source_count": input.terms.len(),
        }),
    )?;
    live.update_source(SourceVisitorProgress {
        raw_terms_emitted: input.terms.len() as u64,
        current_batch_terms: input.terms.len() as u64,
        current_batch_bytes: input_bytes.len() as u64,
        hard_memory_cap_bytes: device_hard_cap_bytes,
        eta_sample_count: 1,
        ..SourceVisitorProgress::default()
    });

    let static_data = ModularFxStaticData::build(job.prime())?;
    let plan = build_p3_modular_flat_plan(&static_data)?;
    let mut checkpoint = P3JobCheckpoint {
        schema_version: P3_CHECKPOINT_SCHEMA.to_string(),
        manifest_sha256: manifest.manifest_sha256.clone(),
        job: job.clone(),
        prime: job.prime(),
        input_sha256: input_sha256.clone(),
        flat_plan_sha256: plan.semantic_sha256().to_string(),
        state: "prepared".to_string(),
        generation: 1,
        artifact_sha256: None,
        column_semantic_sha256: None,
        expanded_contributions: None,
        source_count: None,
        plan_entry_count: None,
        kernel_milliseconds: None,
        resident_bytes: None,
        buffer_high_water_bytes: None,
        device_hard_cap_bytes: None,
        updated_unix_ms: unix_ms(),
    };
    if checkpoint_path.exists() {
        let prior: P3JobCheckpoint = serde_json::from_reader(
            File::open(&checkpoint_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        validate_checkpoint_identity(&prior, &checkpoint)?;
        if prior.state == "artifact_published" {
            let report = publish_report_from_checkpoint(output, job, &prior, &events)?;
            live.update_group(GroupLiveProgress {
                group_id: Some(job.id()),
                words_completed: 1,
                words_total: 1,
                raw_terms_per_column: vec![report.source_count as u64],
                device_resident_bytes: report.resident_bytes,
                device_high_water_bytes: report.buffer_high_water_bytes,
                aggregate_device_cap_bytes: report.device_hard_cap_bytes,
                checkpoint_generation: prior.generation,
                checkpoint_written_unix_ms: Some(prior.updated_unix_ms),
                ..GroupLiveProgress::default()
            });
            return Ok(report);
        }
        if prior.state != "prepared" {
            return Err("existing p3 checkpoint has an unknown state".to_string());
        }
        checkpoint.generation = prior.generation + 1;
    }
    atomic_json(&checkpoint_path, &checkpoint)?;
    live.update_group(GroupLiveProgress {
        group_id: Some(job.id()),
        words_completed: 0,
        words_total: 1,
        raw_terms_per_column: vec![input.terms.len() as u64],
        aggregate_device_cap_bytes: device_hard_cap_bytes,
        checkpoint_generation: checkpoint.generation,
        checkpoint_written_unix_ms: Some(checkpoint.updated_unix_ms),
        ..GroupLiveProgress::default()
    });

    let mut cuda =
        CudaModularP3::new_with_device_cap(&static_data, &plan, device, device_hard_cap_bytes)?;
    let (column, timing) = cuda.accumulate(input)?;
    live.record_gpu_batch(GpuBatchProgress {
        batches_completed: 1,
        last_batch_ms: f64::from(timing.kernel_milliseconds),
        total_batch_ms: f64::from(timing.kernel_milliseconds),
        last_contract_ms: f64::from(timing.kernel_milliseconds),
        total_contract_ms: f64::from(timing.kernel_milliseconds),
        ..GpuBatchProgress::default()
    });
    let artifact = encode_p3_column_artifact(plan.semantic_sha256(), &column)?;
    let artifact_sha256 = format!("{:x}", Sha256::digest(&artifact));
    atomic_bytes(&artifact_path(output, job), &artifact)?;
    checkpoint.state = "artifact_published".to_string();
    checkpoint.generation += 1;
    checkpoint.artifact_sha256 = Some(artifact_sha256.clone());
    checkpoint.column_semantic_sha256 = Some(column.semantic_sha256.clone());
    checkpoint.expanded_contributions = Some(column.expanded_contributions);
    checkpoint.source_count = Some(timing.source_count);
    checkpoint.plan_entry_count = Some(timing.plan_entry_count);
    checkpoint.kernel_milliseconds = Some(timing.kernel_milliseconds);
    checkpoint.resident_bytes = Some(timing.resident_bytes);
    checkpoint.buffer_high_water_bytes = Some(timing.buffer_high_water_bytes);
    checkpoint.device_hard_cap_bytes = Some(timing.device_hard_cap_bytes);
    checkpoint.updated_unix_ms = unix_ms();
    atomic_json(&checkpoint_path, &checkpoint)?;
    live.update_group(GroupLiveProgress {
        group_id: Some(job.id()),
        words_completed: 1,
        words_total: 1,
        raw_terms_per_column: vec![input.terms.len() as u64],
        device_resident_bytes: timing.resident_bytes,
        device_high_water_bytes: timing.buffer_high_water_bytes,
        aggregate_device_cap_bytes: timing.device_hard_cap_bytes,
        checkpoint_generation: checkpoint.generation,
        checkpoint_written_unix_ms: Some(checkpoint.updated_unix_ms),
        ..GroupLiveProgress::default()
    });
    let report = checkpoint_report(&checkpoint, unix_ms())?;
    atomic_json(&directory.join("job-report.json"), &report)?;
    append_event(
        &events,
        serde_json::json!({
            "schema_version": P3_RUN_SCHEMA,
            "event": "job_complete",
            "timestamp_unix_ms": unix_ms(),
            "job_id": job.id(),
            "source_count": timing.source_count,
            "plan_entry_count": timing.plan_entry_count,
            "expanded_contributions": timing.expanded_contributions,
            "kernel_milliseconds": timing.kernel_milliseconds,
            "resident_bytes": timing.resident_bytes,
            "buffer_high_water_bytes": timing.buffer_high_water_bytes,
            "device_hard_cap_bytes": timing.device_hard_cap_bytes,
            "artifact_sha256": report.artifact_sha256,
        }),
    )?;
    Ok(report)
}

pub(crate) fn summarize(output: &Path, jobs: &[P3JobKey]) -> serde_json::Value {
    let mut completed = Vec::new();
    let mut pending = Vec::new();
    let mut invalid = BTreeMap::new();
    for job in jobs {
        match validate_completed_job(output, job) {
            Ok(true) => completed.push(job.id()),
            Ok(false) => pending.push(job.id()),
            Err(error) => {
                invalid.insert(job.id(), error);
            }
        }
    }
    serde_json::json!({
        "schema_version": P3_STATUS_SCHEMA,
        "timestamp_unix_ms": unix_ms(),
        "selected_jobs": jobs.len(),
        "completed_count": completed.len(),
        "pending_count": pending.len(),
        "invalid_count": invalid.len(),
        "completed": completed,
        "pending": pending,
        "pending_job_list": pending.join(","),
        "invalid": invalid,
        "complete": completed.len() == jobs.len(),
    })
}

pub(crate) fn join_all_77(
    prime: u32,
    input_directories: &[PathBuf],
) -> Result<ModularRankCertificate, String> {
    let prime_index = GPU_FX_PRIMES
        .iter()
        .position(|candidate| *candidate == prime)
        .ok_or_else(|| "p3 join prime is not pinned".to_string())?;
    if input_directories.is_empty() {
        return Err("p3 join requires at least one artifact directory".to_string());
    }
    let mut columns = BTreeMap::<usize, ModularP3FunctionalColumn>::new();
    let mut common_plan_sha256 = None::<String>;
    for directory in input_directories {
        for ordinal in 0..77 {
            let job = P3JobKey::new(ordinal, prime_index)?;
            let path = artifact_path(directory, &job);
            let bytes = match fs::read(&path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.to_string()),
            };
            if !validate_completed_job(directory, &job)? {
                return Err(format!(
                    "p3 join found unpublished artifact for {}",
                    job.id()
                ));
            }
            let (plan_sha256, column) = decode_p3_column_artifact(&bytes)?;
            if let Some(expected) = &common_plan_sha256 {
                if expected != &plan_sha256 {
                    return Err("p3 join artifacts disagree on flat-plan identity".to_string());
                }
            } else {
                common_plan_sha256 = Some(plan_sha256);
            }
            if column.prime != prime || column.global_ordinal != ordinal {
                return Err("p3 join artifact identity mismatch".to_string());
            }
            if let Some(existing) = columns.get(&ordinal) {
                if existing.rows != column.rows
                    || existing.semantic_sha256 != column.semantic_sha256
                    || existing.expanded_contributions != column.expanded_contributions
                {
                    return Err("p3 duplicate artifact disagreement".to_string());
                }
            } else {
                columns.insert(ordinal, column);
            }
        }
    }
    let observed = columns.keys().copied().collect::<BTreeSet<_>>();
    let expected = (0..77).collect::<BTreeSet<_>>();
    if observed != expected {
        let missing = expected.difference(&observed).copied().collect::<Vec<_>>();
        return Err(format!(
            "p3 all-77 coverage gate is incomplete; missing {missing:?}"
        ));
    }
    rank_p3_columns(&columns.into_values().collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eleven_dimensional_second_momentum_gpu::{
        GaussianResidue, p3_column_semantic_sha256,
    };

    fn temporary_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("adynkra-{label}-{}", std::process::id()))
    }

    #[test]
    fn p3_manifest_and_portable_selectors_cover_all_77_by_three_primes() {
        let manifest = build_manifest().unwrap();
        assert_eq!(manifest.jobs.len(), 231);
        assert_eq!(parse_job_list("all").unwrap(), manifest.jobs);
        let selected = parse_job_list("0-2,76@1").unwrap();
        assert_eq!(selected.len(), 4);
        assert!(selected.iter().all(|job| job.prime_index == 1));
        assert!(parse_job_list("77@0").is_err());
        assert!(parse_job_list("all@3").is_err());
    }

    #[test]
    fn p3_checkpoint_publication_adopts_and_detects_mutation() {
        let output = temporary_directory("p3-job-canary");
        let _ = fs::remove_dir_all(&output);
        let job = P3JobKey::new(3, 0).unwrap();
        let manifest = build_manifest().unwrap();
        write_or_validate_manifest(&output).unwrap();
        let mut rows = vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
        rows[11].real = 7;
        let column = ModularP3FunctionalColumn {
            prime: job.prime(),
            global_ordinal: job.global_ordinal,
            expanded_contributions: 13,
            semantic_sha256: p3_column_semantic_sha256(job.prime(), job.global_ordinal, &rows),
            rows,
        };
        let plan = "a".repeat(64);
        let artifact = encode_p3_column_artifact(&plan, &column).unwrap();
        atomic_bytes(&artifact_path(&output, &job), &artifact).unwrap();
        let artifact_sha256 = format!("{:x}", Sha256::digest(&artifact));
        let checkpoint = P3JobCheckpoint {
            schema_version: P3_CHECKPOINT_SCHEMA.to_string(),
            manifest_sha256: manifest.manifest_sha256,
            job: job.clone(),
            prime: job.prime(),
            input_sha256: "b".repeat(64),
            flat_plan_sha256: plan,
            state: "artifact_published".to_string(),
            generation: 2,
            artifact_sha256: Some(artifact_sha256),
            column_semantic_sha256: Some(column.semantic_sha256),
            expanded_contributions: Some(column.expanded_contributions),
            source_count: Some(2),
            plan_entry_count: Some(3),
            kernel_milliseconds: Some(1.0),
            resident_bytes: Some(4),
            buffer_high_water_bytes: Some(5),
            device_hard_cap_bytes: Some(6),
            updated_unix_ms: unix_ms(),
        };
        atomic_json(
            &job_directory(&output, &job).join("checkpoint.json"),
            &checkpoint,
        )
        .unwrap();
        let report = publish_report_from_checkpoint(
            &output,
            &job,
            &checkpoint,
            &job_directory(&output, &job).join("events.jsonl"),
        )
        .unwrap();
        assert_eq!(report.source_count, 2);
        assert!(validate_completed_job(&output, &job).unwrap());
        assert!(
            join_all_77(job.prime(), &[output.clone()])
                .unwrap_err()
                .contains("missing")
        );
        let mut mutated = fs::read(artifact_path(&output, &job)).unwrap();
        *mutated.last_mut().unwrap() ^= 1;
        atomic_bytes(&artifact_path(&output, &job), &mutated).unwrap();
        assert!(validate_completed_job(&output, &job).is_err());
        fs::remove_dir_all(output).unwrap();
    }
}
