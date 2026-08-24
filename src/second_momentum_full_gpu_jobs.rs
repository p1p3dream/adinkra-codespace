//! Portable work inventory for the 53 non-large-tranche GPU columns.
//!
//! This is intentionally versioned separately from the live large-tranche job
//! manifest so completed and in-flight artifacts remain adoptable.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::eleven_dimensional_second_momentum_full_inventory::{
    full_column_specs, layout_sha256, missing_gpu_groups,
};
use crate::eleven_dimensional_second_momentum_gpu::GPU_FX_PRIMES;

pub(crate) const FULL_GPU_JOB_SCHEMA: &str = "adynkra-11d-second-momentum-full-gpu-jobs-v2";
pub(crate) const FULL_GPU_RUN_SCHEMA: &str = "adynkra-11d-second-momentum-full-gpu-run-v1";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FullGpuJobKey {
    pub group_index: usize,
    pub prime_index: usize,
}

impl FullGpuJobKey {
    pub(crate) fn new(group_index: usize, prime_index: usize) -> Result<Self, String> {
        if group_index >= missing_gpu_groups().len() || prime_index >= GPU_FX_PRIMES.len() {
            return Err("full GPU job index is out of range".to_string());
        }
        Ok(Self {
            group_index,
            prime_index,
        })
    }

    pub(crate) fn id(&self) -> String {
        format!("full-g{}-p{}", self.group_index, self.prime_index)
    }

    pub(crate) fn prime(&self) -> u32 {
        GPU_FX_PRIMES[self.prime_index]
    }

    pub(crate) fn global_ordinals(&self) -> Vec<usize> {
        missing_gpu_groups()[self.group_index].clone()
    }

    pub(crate) fn tranche(&self) -> String {
        let columns = full_column_specs();
        columns[self.global_ordinals()[0]]
            .intermediate_dynkin_label
            .clone()
    }

    pub(crate) fn parse_id(value: &str) -> Result<Self, String> {
        let suffix = value
            .strip_prefix("full-g")
            .ok_or_else(|| "full GPU job ID must begin with full-g".to_string())?;
        let (group, prime) = suffix
            .split_once("-p")
            .ok_or_else(|| "full GPU job ID requires -p<index>".to_string())?;
        Self::new(
            group
                .parse::<usize>()
                .map_err(|_| "full GPU group index is not an integer".to_string())?,
            prime
                .parse::<usize>()
                .map_err(|_| "full GPU prime index is not an integer".to_string())?,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FullGpuJobManifestEntry {
    pub job_id: String,
    pub tranche: String,
    pub group_index: usize,
    pub prime_index: usize,
    pub prime: u32,
    pub global_ordinals: Vec<usize>,
    pub width: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FullGpuJobManifest {
    pub schema_version: String,
    pub full_column_layout_sha256: String,
    pub physical_columns: usize,
    pub groups: usize,
    pub jobs: Vec<FullGpuJobManifestEntry>,
    pub manifest_sha256: String,
}

fn manifest_digest(manifest: &FullGpuJobManifest) -> Result<String, String> {
    let mut copy = manifest.clone();
    copy.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&copy).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(crate) fn build_manifest() -> Result<FullGpuJobManifest, String> {
    let groups = missing_gpu_groups();
    let columns = full_column_specs();
    let mut jobs = Vec::with_capacity(groups.len() * GPU_FX_PRIMES.len());
    for (group_index, ordinals) in groups.iter().enumerate() {
        let tranche = columns[ordinals[0]].intermediate_dynkin_label.clone();
        if ordinals.iter().any(|ordinal| {
            columns[*ordinal].intermediate_dynkin_label != tranche || !matches!(*ordinal, 0..=52)
        }) {
            return Err("full GPU group crosses a channel or inventory boundary".to_string());
        }
        for prime_index in 0..GPU_FX_PRIMES.len() {
            let key = FullGpuJobKey::new(group_index, prime_index)?;
            jobs.push(FullGpuJobManifestEntry {
                job_id: key.id(),
                tranche: tranche.clone(),
                group_index,
                prime_index,
                prime: key.prime(),
                global_ordinals: ordinals.clone(),
                width: ordinals.len(),
            });
        }
    }
    let mut manifest = FullGpuJobManifest {
        schema_version: FULL_GPU_JOB_SCHEMA.to_string(),
        full_column_layout_sha256: layout_sha256(),
        physical_columns: groups.iter().map(Vec::len).sum(),
        groups: groups.len(),
        jobs,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

pub(crate) fn validate_manifest(manifest: &FullGpuJobManifest) -> Result<(), String> {
    if manifest.schema_version != FULL_GPU_JOB_SCHEMA
        || manifest.full_column_layout_sha256 != layout_sha256()
        || manifest.physical_columns != 53
        || manifest.groups != missing_gpu_groups().len()
        || manifest.manifest_sha256 != manifest_digest(manifest)?
    {
        return Err("full GPU manifest schema, layout, counts, or digest is invalid".to_string());
    }
    let rebuilt = build_entries()?;
    if manifest.jobs != rebuilt {
        return Err("full GPU manifest differs from the canonical inventory".to_string());
    }
    Ok(())
}

fn build_entries() -> Result<Vec<FullGpuJobManifestEntry>, String> {
    let groups = missing_gpu_groups();
    let columns = full_column_specs();
    let mut entries = Vec::new();
    for (group_index, ordinals) in groups.iter().enumerate() {
        for prime_index in 0..GPU_FX_PRIMES.len() {
            let key = FullGpuJobKey::new(group_index, prime_index)?;
            entries.push(FullGpuJobManifestEntry {
                job_id: key.id(),
                tranche: columns[ordinals[0]].intermediate_dynkin_label.clone(),
                group_index,
                prime_index,
                prime: key.prime(),
                global_ordinals: ordinals.clone(),
                width: ordinals.len(),
            });
        }
    }
    Ok(entries)
}

pub(crate) fn parse_job_list(value: &str) -> Result<Vec<FullGpuJobKey>, String> {
    let manifest = build_manifest()?;
    let selected: Vec<String> = if value == "all" {
        manifest
            .jobs
            .iter()
            .map(|entry| entry.job_id.clone())
            .collect()
    } else if let Some((selector, prime)) = value.split_once('@') {
        let prime_index = prime
            .parse::<usize>()
            .map_err(|_| "full GPU @prime selector is not an integer".to_string())?;
        if prime_index >= GPU_FX_PRIMES.len() {
            return Err("full GPU @prime selector is out of range".to_string());
        }
        manifest
            .jobs
            .iter()
            .filter(|entry| {
                entry.prime_index == prime_index && (selector == "all" || entry.tranche == selector)
            })
            .map(|entry| entry.job_id.clone())
            .collect()
    } else {
        value.split(',').map(str::to_string).collect()
    };
    if selected.is_empty() {
        return Err("full GPU job selection is empty".to_string());
    }
    let canonical = manifest
        .jobs
        .iter()
        .map(|entry| entry.job_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut jobs = Vec::with_capacity(selected.len());
    for id in selected {
        if !canonical.contains(id.as_str()) {
            return Err(format!("unknown full GPU job ID {id}"));
        }
        jobs.push(FullGpuJobKey::parse_id(&id)?);
    }
    jobs.sort();
    jobs.dedup();
    Ok(jobs)
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    let file = writer.into_inner().map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    fs::rename(&temporary, path).map_err(|error| error.to_string())?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(crate) fn write_or_validate_manifest(output_directory: &Path) -> Result<PathBuf, String> {
    let manifest = build_manifest()?;
    let path = output_directory.join("full-gpu-work-manifest.json");
    if path.exists() {
        let existing: FullGpuJobManifest =
            serde_json::from_reader(File::open(&path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        validate_manifest(&existing)?;
        if existing != manifest {
            return Err("existing full GPU manifest differs from this build".to_string());
        }
    } else {
        atomic_json(&path, &manifest)?;
    }
    Ok(path)
}

pub(crate) fn report_path(output_directory: &Path, job: &FullGpuJobKey) -> PathBuf {
    output_directory
        .join("jobs")
        .join(job.id())
        .join("job-report.json")
}

pub(crate) fn validate_completed_job(
    output_directory: &Path,
    job: &FullGpuJobKey,
) -> Result<bool, String> {
    let path = report_path(output_directory, job);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.to_string()),
    };
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let manifest = build_manifest()?;
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some(FULL_GPU_RUN_SCHEMA)
        || value.get("job_id").and_then(serde_json::Value::as_str) != Some(job.id().as_str())
        || value
            .get("work_manifest_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(manifest.manifest_sha256.as_str())
        || value.get("prime").and_then(serde_json::Value::as_u64) != Some(u64::from(job.prime()))
        || value.get("passed").and_then(serde_json::Value::as_bool) != Some(true)
    {
        return Err(format!(
            "{} is not a valid full GPU job report",
            path.display()
        ));
    }
    let inventory = value
        .get("artifact_inventory")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "full GPU report has no artifact inventory".to_string())?;
    if inventory.len() != job.global_ordinals().len() {
        return Err("full GPU artifact inventory width changed".to_string());
    }
    let expected_ordinals = job.global_ordinals().into_iter().collect::<BTreeSet<_>>();
    let mut observed_ordinals = BTreeSet::new();
    let mut static_digests = BTreeSet::new();
    for artifact in inventory {
        let ordinal = artifact
            .get("global_ordinal")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| "full GPU artifact has no valid global ordinal".to_string())?;
        if !observed_ordinals.insert(ordinal) {
            return Err("full GPU artifact inventory repeats a global ordinal".to_string());
        }
        let relative = artifact
            .get("binary_relative_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "full GPU artifact has no relative path".to_string())?;
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err("full GPU artifact path is not contained".to_string());
        }
        let expected = artifact
            .get("binary_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "full GPU artifact has no SHA-256".to_string())?;
        let binary_path = output_directory.join(relative_path);
        let observed = fs::read(&binary_path)
            .map(|binary| format!("{:x}", Sha256::digest(binary)))
            .map_err(|error| error.to_string())?;
        if observed != expected {
            return Err("full GPU artifact SHA-256 mismatch".to_string());
        }
        let decoded = decode_column_artifact(&binary_path)?
            .ok_or_else(|| "full GPU artifact does not have ADFXGPU3 encoding".to_string())?;
        if decoded.column.prime != job.prime()
            || decoded.column.global_ordinal != ordinal
            || decoded.binary_sha256 != expected
        {
            return Err("full GPU artifact header identity mismatch".to_string());
        }
        static_digests.insert(decoded.static_semantic_sha256);
    }
    if observed_ordinals != expected_ordinals || static_digests.len() != 1 {
        return Err("full GPU artifact inventory identity or static digest changed".to_string());
    }
    let checkpoint_path = output_directory
        .join("jobs")
        .join(job.id())
        .join("checkpoint.json");
    if checkpoint_path.exists() {
        let envelope: serde_json::Value = serde_json::from_reader(
            File::open(&checkpoint_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if envelope
            .get("payload_sha256")
            .and_then(serde_json::Value::as_str)
            != value
                .get("checkpoint_sha256")
                .and_then(serde_json::Value::as_str)
        {
            return Err("full GPU checkpoint digest does not match its job report".to_string());
        }
    }
    Ok(true)
}

pub(crate) fn summarize(output_directory: &Path, jobs: &[FullGpuJobKey]) -> serde_json::Value {
    const LIVE_HEARTBEAT_MAX_AGE_MS: u128 = 30_000;
    let now = unix_milliseconds();
    let local_hostname = machine_hostname();
    let mut completed = Vec::new();
    let mut running = Vec::new();
    let mut pending = Vec::new();
    let mut failed = Vec::new();
    let mut stale = Vec::new();
    let mut job_details = Vec::new();
    for job in jobs {
        let report = report_path(output_directory, job);
        let status = output_directory
            .join("jobs")
            .join(job.id())
            .join("status.json");
        match validate_completed_job(output_directory, job) {
            Ok(true) => {
                completed.push(job.id());
                job_details.push(serde_json::json!({
                    "job_id": job.id(),
                    "state": "completed",
                    "report_path": report,
                }));
                continue;
            }
            Err(error) => {
                failed.push(job.id());
                job_details.push(serde_json::json!({
                    "job_id": job.id(),
                    "state": "invalid_completed_artifact",
                    "error": error,
                    "report_path": report,
                }));
                continue;
            }
            Ok(false) => {}
        }
        let value = fs::read(&status)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        let Some(value) = value else {
            pending.push(job.id());
            job_details.push(serde_json::json!({
                "job_id": job.id(),
                "state": "pending",
                "status_path": status,
            }));
            continue;
        };
        let reported_state = value
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
        let heartbeat_age_ms = value
            .get("timestamp_unix_ms")
            .and_then(serde_json::Value::as_u64)
            .map(u128::from)
            .map(|timestamp| now.saturating_sub(timestamp));
        let heartbeat_recent = heartbeat_age_ms.is_some_and(|age| age <= LIVE_HEARTBEAT_MAX_AGE_MS);
        let observed_live = if hostname == local_hostname {
            pid.is_some_and(process_is_running)
        } else if hostname == "unknown" {
            pid.is_some_and(process_is_running) || heartbeat_recent
        } else {
            heartbeat_recent
        };
        let state = if reported_state == "running" && observed_live {
            running.push(job.id());
            "running"
        } else if reported_state == "running" {
            stale.push(job.id());
            "stale"
        } else if matches!(reported_state, "failed" | "terminated") {
            failed.push(job.id());
            "failed"
        } else {
            pending.push(job.id());
            "pending"
        };
        job_details.push(serde_json::json!({
            "job_id": job.id(),
            "state": state,
            "reported_state": reported_state,
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
            "status_path": status,
        }));
    }
    serde_json::json!({
        "schema_version": "adynkra-11d-second-momentum-full-gpu-status-v1",
        "timestamp_unix_ms": now,
        "output_directory": output_directory,
        "selected_jobs": jobs.len(),
        "completed_count": completed.len(),
        "running_count": running.len(),
        "pending_count": pending.len(),
        "failed_count": failed.len(),
        "stale_count": stale.len(),
        "completed": completed,
        "running": running,
        "pending": pending,
        "failed": failed,
        "stale": stale,
        "pending_job_list": pending.join(","),
        "stale_job_list": stale.join(","),
        "job_details": job_details,
        "complete": completed.len() == jobs.len(),
    })
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
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
        .unwrap_or_else(|| "unknown".to_string())
}

fn process_is_running(pid: u32) -> bool {
    std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid="])
        .output()
        .is_ok_and(|output| {
            output.status.success() && !String::from_utf8_lossy(&output.stdout).trim().is_empty()
        })
}

#[derive(Clone, Debug)]
struct DecodedColumnArtifact {
    column: crate::eleven_dimensional_second_momentum_gpu::ModularFunctionalColumn,
    binary_sha256: String,
    static_semantic_sha256: String,
    source_terms_sha256: String,
    source_terms: u64,
    path: PathBuf,
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated modular column header".to_string())?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "truncated modular column header".to_string())?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn decode_hex_digest(bytes: &[u8], field: &str) -> Result<String, String> {
    if bytes.len() != 64 || !bytes.iter().all(u8::is_ascii_hexdigit) {
        return Err(format!(
            "modular column {field} is not a hexadecimal SHA-256"
        ));
    }
    String::from_utf8(bytes.to_vec()).map_err(|error| error.to_string())
}

fn decode_column_artifact(path: &Path) -> Result<Option<DecodedColumnArtifact>, String> {
    use crate::eleven_dimensional_second_momentum_gpu::{
        FUNCTIONAL_ROW_COUNT, GPU_FX_SCHEMA, GaussianResidue, ModularFunctionalColumn,
    };

    const HEADER_BYTES: usize = 156;
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if bytes.get(..8) != Some(b"ADFXGPU3") {
        return Ok(None);
    }
    let prime = read_u32(&bytes, 8)?;
    let global_ordinal = usize::try_from(read_u32(&bytes, 12)?).unwrap();
    let row_count = usize::try_from(read_u32(&bytes, 16)?).unwrap();
    let source_terms = read_u64(&bytes, 20)?;
    let static_semantic_sha256 = decode_hex_digest(
        bytes
            .get(28..92)
            .ok_or_else(|| "truncated static digest".to_string())?,
        "static digest",
    )?;
    let source_terms_sha256 = decode_hex_digest(
        bytes
            .get(92..156)
            .ok_or_else(|| "truncated source digest".to_string())?,
        "source digest",
    )?;
    if global_ordinal >= 77 || row_count != FUNCTIONAL_ROW_COUNT {
        return Err(format!(
            "{} has invalid ordinal or row count",
            path.display()
        ));
    }
    let expected_bytes = HEADER_BYTES
        .checked_add(
            row_count
                .checked_mul(8)
                .ok_or_else(|| "modular row byte count overflow".to_string())?,
        )
        .ok_or_else(|| "modular artifact byte count overflow".to_string())?;
    if bytes.len() != expected_bytes {
        return Err(format!("{} has a noncanonical byte count", path.display()));
    }
    let mut rows = Vec::with_capacity(row_count);
    for chunk in bytes[HEADER_BYTES..].chunks_exact(8) {
        let real = u32::from_le_bytes(chunk[..4].try_into().unwrap());
        let imaginary = u32::from_le_bytes(chunk[4..].try_into().unwrap());
        if real >= prime || imaginary >= prime {
            return Err(format!(
                "{} contains a noncanonical residue",
                path.display()
            ));
        }
        rows.push(GaussianResidue { real, imaginary });
    }
    let mut semantic = Sha256::new();
    semantic.update(GPU_FX_SCHEMA.as_bytes());
    semantic.update(prime.to_le_bytes());
    semantic.update((global_ordinal as u64).to_le_bytes());
    semantic.update(static_semantic_sha256.as_bytes());
    // The artifact payload is the canonical little-endian row encoding.
    // Hash it in one contiguous update instead of two updates per residue.
    semantic.update(&bytes[HEADER_BYTES..]);
    let semantic_sha256 = format!("{:x}", semantic.finalize());
    let binary_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let report_path = path.with_extension("json");
    let report: serde_json::Value =
        serde_json::from_reader(File::open(&report_path).map_err(|error| {
            format!(
                "cannot read companion report {}: {error}",
                report_path.display()
            )
        })?)
        .map_err(|error| error.to_string())?;
    if report
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some(GPU_FX_SCHEMA)
        || report.get("passed").and_then(serde_json::Value::as_bool) != Some(true)
        || report.get("prime").and_then(serde_json::Value::as_u64) != Some(u64::from(prime))
        || report
            .get("global_ordinal")
            .and_then(serde_json::Value::as_u64)
            != Some(global_ordinal as u64)
        || report
            .get("binary_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(binary_sha256.as_str())
        || report
            .get("column_semantic_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(semantic_sha256.as_str())
        || report
            .get("static_semantic_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(static_semantic_sha256.as_str())
        || report
            .get("source_terms_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(source_terms_sha256.as_str())
        || report
            .get("source_terms")
            .and_then(serde_json::Value::as_u64)
            != Some(source_terms)
    {
        return Err(format!(
            "companion report {} does not bind the modular column",
            report_path.display()
        ));
    }
    Ok(Some(DecodedColumnArtifact {
        column: ModularFunctionalColumn {
            prime,
            global_ordinal,
            rows,
            expanded_contributions: report
                .get("expanded_contributions")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            semantic_sha256,
        },
        binary_sha256,
        static_semantic_sha256,
        source_terms_sha256,
        source_terms,
        path: path.to_path_buf(),
    }))
}

pub(crate) fn aggregate_full_rank(
    prime: u32,
    input_directories: &[PathBuf],
) -> Result<serde_json::Value, String> {
    if !crate::eleven_dimensional_second_momentum_gpu::GPU_FX_PRIMES.contains(&prime) {
        return Err("full rank aggregation requires one of the pinned primes".to_string());
    }
    if input_directories.is_empty() {
        return Err("full rank aggregation requires at least one input directory".to_string());
    }
    let mut artifacts = BTreeMap::<usize, DecodedColumnArtifact>::new();
    for directory in input_directories {
        for entry in fs::read_dir(directory).map_err(|error| {
            format!(
                "cannot read artifact directory {}: {error}",
                directory.display()
            )
        })? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("bin") {
                continue;
            }
            let Some(decoded) = decode_column_artifact(&path)? else {
                continue;
            };
            if decoded.column.prime != prime {
                continue;
            }
            match artifacts.get(&decoded.column.global_ordinal) {
                Some(existing)
                    if existing.binary_sha256 == decoded.binary_sha256
                        && existing.column.semantic_sha256 == decoded.column.semantic_sha256 => {}
                Some(_) => {
                    return Err(format!(
                        "conflicting artifacts for global column {}",
                        decoded.column.global_ordinal
                    ));
                }
                None => {
                    artifacts.insert(decoded.column.global_ordinal, decoded);
                }
            }
        }
    }
    let missing = (0..77)
        .filter(|ordinal| !artifacts.contains_key(ordinal))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "full rank aggregation is missing global columns {missing:?}"
        ));
    }
    let static_digests = artifacts
        .values()
        .map(|artifact| artifact.static_semantic_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if static_digests.len() != 1 {
        return Err("full rank artifacts do not share one static semantic digest".to_string());
    }
    let columns = artifacts
        .values()
        .map(|artifact| artifact.column.clone())
        .collect::<Vec<_>>();
    let rank = crate::eleven_dimensional_second_momentum_gpu::rank_columns(&columns)?;
    let inventory = artifacts
        .values()
        .map(|artifact| {
            serde_json::json!({
                "global_ordinal": artifact.column.global_ordinal,
                "path": artifact.path,
                "binary_sha256": artifact.binary_sha256,
                "column_semantic_sha256": artifact.column.semantic_sha256,
                "source_terms": artifact.source_terms,
                "source_terms_sha256": artifact.source_terms_sha256,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "schema_version": "adynkra-11d-second-momentum-full-rank-v1",
        "prime": prime,
        "input_directories": input_directories,
        "static_semantic_sha256": static_digests.into_iter().next().unwrap(),
        "physical_columns": 77,
        "rank_over_gaussian_extension": rank.rank_over_gaussian_extension,
        "nullity_upper_bound": rank.nullity_upper_bound,
        "full_column_rank": rank.full_column_rank,
        "matrix_sha256": rank.matrix_sha256,
        "column_ordinals": rank.column_ordinals,
        "artifact_inventory": inventory,
        "proof_boundary": "All 77 columns were loaded at one pinned prime, every binary and companion report was verified, and exact Gaussian finite-field elimination supplied the characteristic-zero rank lower bound.",
        "passed": true,
    }))
}

pub(crate) fn publish_full_rank(
    prime: u32,
    input_directories: &[PathBuf],
    output_path: &Path,
) -> Result<serde_json::Value, String> {
    let report = aggregate_full_rank(prime, input_directories)?;
    if output_path.exists() {
        let existing: serde_json::Value =
            serde_json::from_reader(File::open(output_path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        if existing != report {
            return Err(format!(
                "refusing to replace differing full-rank report {}",
                output_path.display()
            ));
        }
    } else {
        atomic_json(output_path, &report)?;
    }
    Ok(report)
}

pub(crate) fn publish_declared_28_rank(
    prime: u32,
    input_directories: &[PathBuf],
    output_path: &Path,
) -> Result<serde_json::Value, String> {
    if !crate::eleven_dimensional_second_momentum_gpu::GPU_FX_PRIMES.contains(&prime)
        || input_directories.is_empty()
    {
        return Err("declared-28 aggregation requires a pinned prime and input directories".into());
    }
    let expected = (19..=22).chain(53..=76).collect::<BTreeSet<_>>();
    let mut artifacts = BTreeMap::<usize, DecodedColumnArtifact>::new();
    for directory in input_directories {
        for entry in fs::read_dir(directory).map_err(|error| {
            format!("cannot read artifact directory {}: {error}", directory.display())
        })? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("bin") {
                continue;
            }
            let Some(decoded) = decode_column_artifact(&path)? else {
                continue;
            };
            let ordinal = decoded.column.global_ordinal;
            if decoded.column.prime != prime || !expected.contains(&ordinal) {
                continue;
            }
            match artifacts.get(&ordinal) {
                Some(existing)
                    if existing.binary_sha256 == decoded.binary_sha256
                        && existing.column.semantic_sha256 == decoded.column.semantic_sha256 => {}
                Some(_) => return Err(format!("conflicting artifacts for global column {ordinal}")),
                None => {
                    artifacts.insert(ordinal, decoded);
                }
            }
        }
    }
    let missing = expected
        .iter()
        .filter(|ordinal| !artifacts.contains_key(ordinal))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("declared-28 aggregation is missing columns {missing:?}"));
    }
    let static_digests = artifacts
        .values()
        .map(|artifact| artifact.static_semantic_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if static_digests.len() != 1 {
        return Err("declared-28 artifacts do not share one static semantic digest".into());
    }
    let columns = artifacts
        .values()
        .map(|artifact| artifact.column.clone())
        .collect::<Vec<_>>();
    let rank = crate::eleven_dimensional_second_momentum_gpu::rank_columns(&columns)?;
    let report = serde_json::json!({
        "schema_version": "adynkra-11d-second-momentum-declared-28-rank-v1",
        "prime": prime,
        "input_directories": input_directories,
        "static_semantic_sha256": static_digests.into_iter().next().unwrap(),
        "physical_columns": 28,
        "rank_over_gaussian_extension": rank.rank_over_gaussian_extension,
        "nullity_upper_bound": rank.nullity_upper_bound,
        "full_column_rank": rank.full_column_rank,
        "matrix_sha256": rank.matrix_sha256,
        "column_ordinals": rank.column_ordinals,
        "artifact_inventory": artifacts.values().map(|artifact| serde_json::json!({
            "global_ordinal": artifact.column.global_ordinal,
            "path": artifact.path,
            "binary_sha256": artifact.binary_sha256,
            "column_semantic_sha256": artifact.column.semantic_sha256,
            "source_terms": artifact.source_terms,
            "source_terms_sha256": artifact.source_terms_sha256,
        })).collect::<Vec<_>>(),
        "proof_boundary": "All 28 declared-slice columns were decoded from verified ADFXGPU3 artifacts at one pinned prime, and exact Gaussian finite-field elimination supplied the characteristic-zero rank lower bound.",
        "passed": rank.full_column_rank,
    });
    if output_path.exists() {
        let existing: serde_json::Value = serde_json::from_reader(
            File::open(output_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if existing != report {
            return Err(format!("refusing to replace differing declared-28 report {}", output_path.display()));
        }
    } else {
        atomic_json(output_path, &report)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eleven_dimensional_second_momentum_gpu::{FUNCTIONAL_ROW_COUNT, GPU_FX_SCHEMA};

    #[test]
    fn manifest_covers_53_columns_at_three_primes() {
        let manifest = build_manifest().unwrap();
        assert_eq!(manifest.physical_columns, 53);
        assert_eq!(manifest.groups, 32);
        assert_eq!(manifest.jobs.len(), 96);
        assert_eq!(
            manifest
                .jobs
                .iter()
                .filter(|job| job.prime_index == 0)
                .count(),
            32
        );
        validate_manifest(&manifest).unwrap();
    }

    #[test]
    fn selectors_are_portable_and_channel_aware() {
        assert_eq!(parse_job_list("all@0").unwrap().len(), 32);
        assert_eq!(parse_job_list("00001@0").unwrap().len(), 3);
        assert_eq!(parse_job_list("10001@0").unwrap().len(), 8);
        assert_eq!(parse_job_list("full-g0-p0").unwrap()[0].id(), "full-g0-p0");
        assert!(parse_job_list("full-g99-p0").is_err());
    }

    #[test]
    fn full_rank_aggregator_verifies_all_artifacts_and_finds_rank_77() {
        let root = std::env::temp_dir().join(format!(
            "adynkra-full-rank-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let prime = GPU_FX_PRIMES[0];
        let static_digest = "a".repeat(64);
        let source_digest = "b".repeat(64);
        for ordinal in 0..77_usize {
            let mut rows = vec![
                crate::eleven_dimensional_second_momentum_gpu::GaussianResidue::zero(
                );
                FUNCTIONAL_ROW_COUNT
            ];
            rows[ordinal].real = 1;
            let mut binary = Vec::with_capacity(156 + FUNCTIONAL_ROW_COUNT * 8);
            binary.extend_from_slice(b"ADFXGPU3");
            binary.extend_from_slice(&prime.to_le_bytes());
            binary.extend_from_slice(&(ordinal as u32).to_le_bytes());
            binary.extend_from_slice(&(FUNCTIONAL_ROW_COUNT as u32).to_le_bytes());
            binary.extend_from_slice(&1_u64.to_le_bytes());
            binary.extend_from_slice(static_digest.as_bytes());
            binary.extend_from_slice(source_digest.as_bytes());
            for value in &rows {
                binary.extend_from_slice(&value.real.to_le_bytes());
                binary.extend_from_slice(&value.imaginary.to_le_bytes());
            }
            let binary_sha256 = format!("{:x}", Sha256::digest(&binary));
            let mut semantic = Sha256::new();
            semantic.update(GPU_FX_SCHEMA.as_bytes());
            semantic.update(prime.to_le_bytes());
            semantic.update((ordinal as u64).to_le_bytes());
            semantic.update(static_digest.as_bytes());
            for value in &rows {
                semantic.update(value.real.to_le_bytes());
                semantic.update(value.imaginary.to_le_bytes());
            }
            let column_semantic_sha256 = format!("{:x}", semantic.finalize());
            let stem = format!("second_momentum_test_column_{ordinal:02}_p{prime}");
            fs::write(root.join(format!("{stem}.bin")), binary).unwrap();
            fs::write(
                root.join(format!("{stem}.json")),
                serde_json::to_vec(&serde_json::json!({
                    "schema_version": GPU_FX_SCHEMA,
                    "passed": true,
                    "prime": prime,
                    "global_ordinal": ordinal,
                    "binary_sha256": binary_sha256,
                    "column_semantic_sha256": column_semantic_sha256,
                    "static_semantic_sha256": static_digest,
                    "source_terms_sha256": source_digest,
                    "source_terms": 1,
                    "expanded_contributions": 1,
                }))
                .unwrap(),
            )
            .unwrap();
        }
        let report = aggregate_full_rank(prime, std::slice::from_ref(&root)).unwrap();
        assert_eq!(report["rank_over_gaussian_extension"], 77);
        assert_eq!(report["nullity_upper_bound"], 0);
        assert_eq!(report["full_column_rank"], true);
        assert_eq!(report["artifact_inventory"].as_array().unwrap().len(), 77);
        fs::remove_dir_all(root).unwrap();
    }
}
