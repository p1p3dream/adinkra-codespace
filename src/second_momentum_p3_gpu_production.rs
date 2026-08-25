//! Bounded, word-resumable production runner for the p^3 D^11 functional.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::checkpointable_sha256::CheckpointableSha256;
#[cfg(feature = "cuda")]
use crate::eleven_dimensional_second_momentum_gpu::p3_recoupling_key;
use crate::eleven_dimensional_second_momentum_gpu::{
    GPU_FX_PRIMES, GPU_P3_FX_SCHEMA, GaussianResidue, ModularP3FunctionalColumn,
    ModularRankCertificate, P3_FUNCTIONAL_ROW_COUNT, decode_p3_column_artifact,
    encode_p3_column_artifact, p3_column_semantic_sha256, rank_p3_columns,
};
use crate::second_momentum_gpu_group::{GpuFxTranche, discover_legal_cuda_column_groups};

pub(crate) const P3_PRODUCTION_JOB_SCHEMA: &str =
    "adynkra-11d-second-momentum-p3-production-jobs-v1";
pub(crate) const P3_PRODUCTION_CHECKPOINT_SCHEMA: &str =
    "adynkra-11d-second-momentum-p3-production-checkpoint-v2";
pub(crate) const P3_PRODUCTION_RUN_SCHEMA: &str =
    "adynkra-11d-second-momentum-p3-production-run-v2";

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(crate) fn canonical_p3_groups() -> Result<Vec<Vec<usize>>, String> {
    let mut groups = crate::eleven_dimensional_second_momentum_full_inventory::missing_gpu_groups();
    groups.extend(
        discover_legal_cuda_column_groups(GpuFxTranche::Two0001)
            .into_iter()
            .map(|group| group.into_iter().map(|ordinal| ordinal + 53).collect()),
    );
    groups.extend(
        discover_legal_cuda_column_groups(GpuFxTranche::Three0001)
            .into_iter()
            .map(|group| group.into_iter().map(|ordinal| ordinal + 62).collect()),
    );
    let flattened = groups.iter().flatten().copied().collect::<Vec<_>>();
    if flattened != (0..77).collect::<Vec<_>>()
        || groups
            .iter()
            .any(|group| group.is_empty() || group.len() > 3)
    {
        return Err("p3 production groups do not partition all 77 columns".to_string());
    }
    Ok(groups)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3ProductionJobKey {
    pub group_index: usize,
    pub prime_index: usize,
}

impl P3ProductionJobKey {
    pub(crate) fn new(group_index: usize, prime_index: usize) -> Result<Self, String> {
        if group_index >= canonical_p3_groups()?.len() || prime_index >= GPU_FX_PRIMES.len() {
            return Err("p3 production job index is out of range".to_string());
        }
        Ok(Self {
            group_index,
            prime_index,
        })
    }

    pub(crate) fn id(&self) -> String {
        format!("p3-g{}-p{}", self.group_index, self.prime_index)
    }

    pub(crate) fn prime(&self) -> u32 {
        GPU_FX_PRIMES[self.prime_index]
    }

    pub(crate) fn global_ordinals(&self) -> Result<Vec<usize>, String> {
        canonical_p3_groups()?
            .get(self.group_index)
            .cloned()
            .ok_or_else(|| "p3 production group index is out of range".to_string())
    }

    pub(crate) fn parse_id(value: &str) -> Result<Self, String> {
        let suffix = value
            .strip_prefix("p3-g")
            .ok_or_else(|| "p3 production job ID must begin with p3-g".to_string())?;
        let (group, prime) = suffix
            .split_once("-p")
            .ok_or_else(|| "p3 production job ID requires -p<index>".to_string())?;
        Self::new(
            group
                .parse()
                .map_err(|_| "p3 group index is not an integer".to_string())?,
            prime
                .parse()
                .map_err(|_| "p3 prime index is not an integer".to_string())?,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3ProductionManifestEntry {
    pub job_id: String,
    pub group_index: usize,
    pub prime_index: usize,
    pub prime: u32,
    pub global_ordinals: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3ProductionManifest {
    pub schema_version: String,
    pub row_schema_version: String,
    pub full_column_layout_sha256: String,
    pub physical_columns: usize,
    pub groups: usize,
    pub jobs: Vec<P3ProductionManifestEntry>,
    pub manifest_sha256: String,
}

fn manifest_digest(manifest: &P3ProductionManifest) -> Result<String, String> {
    let mut copy = manifest.clone();
    copy.manifest_sha256.clear();
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&copy).map_err(|error| error.to_string())?)
    ))
}

pub(crate) fn build_manifest() -> Result<P3ProductionManifest, String> {
    let groups = canonical_p3_groups()?;
    let mut jobs = Vec::with_capacity(groups.len() * GPU_FX_PRIMES.len());
    for (group_index, global_ordinals) in groups.iter().enumerate() {
        for prime_index in 0..GPU_FX_PRIMES.len() {
            let key = P3ProductionJobKey::new(group_index, prime_index)?;
            jobs.push(P3ProductionManifestEntry {
                job_id: key.id(),
                group_index,
                prime_index,
                prime: key.prime(),
                global_ordinals: global_ordinals.clone(),
            });
        }
    }
    let mut manifest = P3ProductionManifest {
        schema_version: P3_PRODUCTION_JOB_SCHEMA.to_string(),
        row_schema_version: GPU_P3_FX_SCHEMA.to_string(),
        full_column_layout_sha256:
            crate::eleven_dimensional_second_momentum_full_inventory::layout_sha256(),
        physical_columns: 77,
        groups: groups.len(),
        jobs,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

pub(crate) fn validate_manifest(manifest: &P3ProductionManifest) -> Result<(), String> {
    if manifest.schema_version != P3_PRODUCTION_JOB_SCHEMA
        || manifest.row_schema_version != GPU_P3_FX_SCHEMA
        || manifest.full_column_layout_sha256
            != crate::eleven_dimensional_second_momentum_full_inventory::layout_sha256()
        || manifest.physical_columns != 77
        || manifest.groups != canonical_p3_groups()?.len()
        || manifest.manifest_sha256 != manifest_digest(manifest)?
    {
        return Err("p3 production manifest identity is invalid".to_string());
    }
    let rebuilt = build_manifest_entries()?;
    if manifest.jobs != rebuilt {
        return Err("p3 production manifest differs from canonical inventory".to_string());
    }
    Ok(())
}

fn build_manifest_entries() -> Result<Vec<P3ProductionManifestEntry>, String> {
    let groups = canonical_p3_groups()?;
    let mut jobs = Vec::new();
    for (group_index, global_ordinals) in groups.into_iter().enumerate() {
        for prime_index in 0..GPU_FX_PRIMES.len() {
            let key = P3ProductionJobKey::new(group_index, prime_index)?;
            jobs.push(P3ProductionManifestEntry {
                job_id: key.id(),
                group_index,
                prime_index,
                prime: key.prime(),
                global_ordinals: global_ordinals.clone(),
            });
        }
    }
    Ok(jobs)
}

pub(crate) fn parse_job_list(value: &str) -> Result<Vec<P3ProductionJobKey>, String> {
    let manifest = build_manifest()?;
    let selected = if value == "all" {
        manifest
            .jobs
            .iter()
            .map(|entry| entry.job_id.clone())
            .collect::<Vec<_>>()
    } else if let Some((selector, prime)) = value.rsplit_once('@') {
        let prime_index: usize = prime
            .parse()
            .map_err(|_| "p3 @prime selector is not an integer".to_string())?;
        if prime_index >= GPU_FX_PRIMES.len() {
            return Err("p3 @prime selector is out of range".to_string());
        }
        let groups = if selector == "all" {
            (0..manifest.groups).collect::<BTreeSet<_>>()
        } else {
            let mut groups = BTreeSet::new();
            for part in selector.split(',') {
                if let Some((start, end)) = part.split_once('-') {
                    let start: usize = start
                        .parse()
                        .map_err(|_| "invalid p3 group range start".to_string())?;
                    let end: usize = end
                        .parse()
                        .map_err(|_| "invalid p3 group range end".to_string())?;
                    if start > end || end >= manifest.groups {
                        return Err("p3 group range is out of bounds".to_string());
                    }
                    groups.extend(start..=end);
                } else {
                    let group: usize = part
                        .parse()
                        .map_err(|_| "invalid p3 group selector".to_string())?;
                    if group >= manifest.groups {
                        return Err("p3 group selector is out of bounds".to_string());
                    }
                    groups.insert(group);
                }
            }
            groups
        };
        groups
            .into_iter()
            .map(|group| P3ProductionJobKey::new(group, prime_index).map(|job| job.id()))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        value.split(',').map(str::to_string).collect::<Vec<_>>()
    };
    let canonical = manifest
        .jobs
        .iter()
        .map(|entry| entry.job_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut jobs = Vec::new();
    for id in selected {
        if !canonical.contains(id.as_str()) {
            return Err(format!("unknown p3 production job {id}"));
        }
        jobs.push(P3ProductionJobKey::parse_id(&id)?);
    }
    jobs.sort();
    jobs.dedup();
    if jobs.is_empty() {
        return Err("p3 production job selection is empty".to_string());
    }
    Ok(jobs)
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension(format!("tmp.{}.{}", std::process::id(), unix_ms()));
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

fn append_event(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    writer
        .get_ref()
        .sync_data()
        .map_err(|error| error.to_string())
}

pub(crate) fn write_or_validate_manifest(output: &Path) -> Result<PathBuf, String> {
    let expected = build_manifest()?;
    let path = output.join("p3-production-manifest.json");
    if path.exists() {
        let observed: P3ProductionManifest =
            serde_json::from_reader(File::open(&path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        validate_manifest(&observed)?;
        if observed != expected {
            return Err("existing p3 production manifest differs from this build".to_string());
        }
    } else {
        atomic_json(&path, &expected)?;
    }
    Ok(path)
}

fn job_directory(output: &Path, job: &P3ProductionJobKey) -> PathBuf {
    output.join("jobs").join(job.id())
}

fn final_artifact_path(output: &Path, ordinal: usize, prime_index: usize) -> PathBuf {
    output
        .join("columns")
        .join(format!("p3-c{ordinal}-p{prime_index}.adfxp3"))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3PublishedArtifact {
    pub global_ordinal: usize,
    pub relative_path: String,
    pub artifact_sha256: String,
    pub column_semantic_sha256: String,
    pub expanded_contributions: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3ProductionCheckpoint {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub job: P3ProductionJobKey,
    pub prime: u32,
    pub group_plan_sha256: String,
    pub source_group_sha256: String,
    pub pbw_plan_sha256: String,
    pub source_label: String,
    pub source_copies: Vec<usize>,
    pub flat_plan_sha256: String,
    pub next_word_ordinal: usize,
    pub total_words: usize,
    pub next_batch_ordinal: u64,
    pub p3_batches_completed: u64,
    pub raw_terms_per_column: Vec<u64>,
    /// Published semantic count over the unreduced raw source schedule.
    pub expanded_contributions_per_column: Vec<u64>,
    /// Diagnostic CUDA work count after exact union reduction.
    pub reduced_expanded_contributions_per_column: Vec<u64>,
    pub source_hashers: Vec<CheckpointableSha256>,
    pub p2_rows: Vec<Vec<GaussianResidue>>,
    pub p2_batches_folded: u64,
    pub kernel_milliseconds: f64,
    pub device_resident_bytes: u64,
    pub device_high_water_bytes: u64,
    pub device_hard_cap_bytes: u64,
    pub checkpoint_generation: u64,
    pub state: String,
    pub artifacts: Vec<P3PublishedArtifact>,
    pub updated_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct P3ProductionReport {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub job: P3ProductionJobKey,
    pub prime: u32,
    pub group_plan_sha256: String,
    pub source_group_sha256: String,
    pub pbw_plan_sha256: String,
    pub source_label: String,
    pub source_copies: Vec<usize>,
    pub flat_plan_sha256: String,
    pub completed_words: usize,
    pub total_words: usize,
    pub batches_completed: u64,
    pub raw_terms_per_column: Vec<u64>,
    pub source_terms_sha256: Vec<String>,
    /// Published semantic count over the unreduced raw source schedule.
    pub expanded_contributions_per_column: Vec<u64>,
    /// Diagnostic CUDA work count after exact union reduction.
    pub reduced_expanded_contributions_per_column: Vec<u64>,
    pub p2_batches_folded: u64,
    pub p2_rows_sha256: Vec<String>,
    pub kernel_milliseconds: f64,
    pub artifacts: Vec<P3PublishedArtifact>,
    pub completed_unix_ms: u128,
    pub passed: bool,
}

pub(crate) fn checkpoint_path(output: &Path, job: &P3ProductionJobKey) -> PathBuf {
    job_directory(output, job).join("checkpoint.json")
}

pub(crate) fn report_path(output: &Path, job: &P3ProductionJobKey) -> PathBuf {
    job_directory(output, job).join("job-report.json")
}

fn validate_artifact(
    output: &Path,
    job: &P3ProductionJobKey,
    artifact: &P3PublishedArtifact,
    expected_flat_plan_sha256: &str,
) -> Result<ModularP3FunctionalColumn, String> {
    let relative = Path::new(&artifact.relative_path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("p3 production artifact path is not confined to the output root".to_string());
    }
    let bytes =
        fs::read(output.join(&artifact.relative_path)).map_err(|error| error.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != artifact.artifact_sha256 {
        return Err("p3 production artifact SHA-256 mismatch".to_string());
    }
    let (plan, column) = decode_p3_column_artifact(&bytes)?;
    if plan != expected_flat_plan_sha256
        || column.prime != job.prime()
        || column.global_ordinal != artifact.global_ordinal
        || column.semantic_sha256 != artifact.column_semantic_sha256
        || column.expanded_contributions != artifact.expanded_contributions
    {
        return Err("p3 production artifact semantic identity mismatch".to_string());
    }
    Ok(column)
}

pub(crate) fn validate_completed_job(
    output: &Path,
    job: &P3ProductionJobKey,
) -> Result<bool, String> {
    let path = report_path(output, job);
    if !path.exists() {
        return Ok(false);
    }
    let report: P3ProductionReport =
        serde_json::from_reader(File::open(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let manifest = build_manifest()?;
    let ordinals = job.global_ordinals()?;
    if report.schema_version != P3_PRODUCTION_RUN_SCHEMA
        || report.manifest_sha256 != manifest.manifest_sha256
        || report.job != *job
        || report.prime != job.prime()
        || report.completed_words != report.total_words
        || report.raw_terms_per_column.len() != ordinals.len()
        || report.source_terms_sha256.len() != ordinals.len()
        || report.source_copies.len() != ordinals.len()
        || report.source_label.is_empty()
        || report.pbw_plan_sha256.len() != 64
        || report.expanded_contributions_per_column.len() != ordinals.len()
        || report.reduced_expanded_contributions_per_column.len() != ordinals.len()
        || report.p2_rows_sha256.len() != ordinals.len()
        || report.artifacts.len() != ordinals.len()
        || report.group_plan_sha256.len() != 64
        || report.source_group_sha256.len() != 64
        || report.flat_plan_sha256.len() != 64
        || report.source_terms_sha256.iter().any(|digest| {
            digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        || report.p2_rows_sha256.iter().any(|digest| {
            digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        || report
            .artifacts
            .iter()
            .map(|artifact| artifact.global_ordinal)
            .collect::<Vec<_>>()
            != ordinals
        || report.artifacts.iter().any(|artifact| {
            artifact.relative_path
                != final_artifact_path(output, artifact.global_ordinal, job.prime_index)
                    .strip_prefix(output)
                    .map(|path| path.to_string_lossy().to_string())
                    .unwrap_or_default()
        })
        || !report.passed
    {
        return Err("p3 production completed report identity is invalid".to_string());
    }
    for artifact in &report.artifacts {
        let column = validate_artifact(output, job, artifact, &report.flat_plan_sha256)?;
        if column.expanded_contributions
            != report.expanded_contributions_per_column[ordinals
                .iter()
                .position(|ordinal| *ordinal == column.global_ordinal)
                .unwrap()]
        {
            return Err("p3 production report expanded count mismatch".to_string());
        }
    }
    Ok(true)
}

pub(crate) fn summarize(output: &Path, jobs: &[P3ProductionJobKey]) -> serde_json::Value {
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
        "schema_version": P3_PRODUCTION_RUN_SCHEMA,
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
        .ok_or_else(|| "p3 production join prime is not pinned".to_string())?;
    let manifest = build_manifest()?;
    let jobs = manifest
        .jobs
        .iter()
        .filter(|entry| entry.prime_index == prime_index)
        .map(|entry| P3ProductionJobKey::new(entry.group_index, entry.prime_index))
        .collect::<Result<Vec<_>, _>>()?;
    let mut columns = BTreeMap::<usize, ModularP3FunctionalColumn>::new();
    let mut common_plan = None::<String>;
    for directory in input_directories {
        for job in &jobs {
            match validate_completed_job(directory, job) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(error) => {
                    return Err(format!("invalid p3 production job {}: {error}", job.id()));
                }
            }
            let report: P3ProductionReport = serde_json::from_reader(
                File::open(report_path(directory, job)).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            if let Some(expected) = &common_plan {
                if expected != &report.flat_plan_sha256 {
                    return Err("p3 production artifacts disagree on flat plan".to_string());
                }
            } else {
                common_plan = Some(report.flat_plan_sha256.clone());
            }
            for artifact in &report.artifacts {
                let column = validate_artifact(directory, job, artifact, &report.flat_plan_sha256)?;
                if let Some(existing) = columns.get(&column.global_ordinal) {
                    if existing.rows != column.rows
                        || existing.semantic_sha256 != column.semantic_sha256
                        || existing.expanded_contributions != column.expanded_contributions
                    {
                        return Err("p3 production duplicate column disagreement".to_string());
                    }
                } else {
                    columns.insert(column.global_ordinal, column);
                }
            }
        }
    }
    let observed = columns.keys().copied().collect::<BTreeSet<_>>();
    let expected = (0..77).collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(format!(
            "p3 production all-77 coverage incomplete; missing {:?}",
            expected.difference(&observed).copied().collect::<Vec<_>>()
        ));
    }
    rank_p3_columns(&columns.into_values().collect::<Vec<_>>())
}

pub(crate) fn publish_all_77_rank(
    prime: u32,
    input_directories: &[PathBuf],
    output_path: &Path,
) -> Result<ModularRankCertificate, String> {
    let certificate = join_all_77(prime, input_directories)?;
    atomic_json(output_path, &certificate)?;
    Ok(certificate)
}

#[cfg(feature = "cuda")]
struct P3ProductionLock {
    file: File,
}

#[cfg(all(feature = "cuda", unix))]
unsafe extern "C" {
    #[link_name = "flock"]
    fn p3_production_flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
}

#[cfg(all(feature = "cuda", unix))]
impl Drop for P3ProductionLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe {
            p3_production_flock(self.file.as_raw_fd(), 8);
        }
    }
}

#[cfg(all(feature = "cuda", not(unix)))]
impl Drop for P3ProductionLock {
    fn drop(&mut self) {}
}

#[cfg(feature = "cuda")]
fn acquire_lock(output: &Path, job: &P3ProductionJobKey) -> Result<P3ProductionLock, String> {
    let directory = output.join(".locks");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(directory.join(format!("{}.lock", job.id())))
        .map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::io::{Seek, SeekFrom};
        use std::os::fd::AsRawFd;
        if unsafe { p3_production_flock(file.as_raw_fd(), 2 | 4) } != 0 {
            return Err(format!(
                "another live worker owns {}: {}",
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
    Ok(P3ProductionLock { file })
}

#[cfg(feature = "cuda")]
fn seed_source_hashers(
    plan: &crate::second_momentum_gpu_group::PreparedColumnGroup,
) -> Result<Vec<CheckpointableSha256>, String> {
    plan.members
        .iter()
        .map(|_| {
            let mut hash = CheckpointableSha256::new();
            hash.update(crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA.as_bytes())?;
            hash.update(b"\0streamed-source-terms-v1\0")?;
            Ok(hash)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn update_source_hash(
    hash: &mut CheckpointableSha256,
    _word_ordinal: usize,
    terms: &[crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm],
) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(terms.len().saturating_mul(23));
    for term in terms {
        p3_recoupling_key(term)?;
        bytes.extend_from_slice(&term.momentum_pair);
        bytes.push(term.free_spinor);
        bytes.extend_from_slice(&term.exterior_mask.to_le_bytes());
        bytes.extend_from_slice(&term.coefficient.to_le_bytes());
    }
    hash.update(&bytes)
}

fn finish_source_hash(
    hash: &CheckpointableSha256,
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
    outer.update(hash.finalize_bytes()?);
    Ok(format!("{:x}", outer.finalize()))
}

fn union_lane_has_terms(
    union: &crate::second_momentum_gpu_group::ExactUnionBatch,
    active_columns: usize,
    lane: usize,
) -> Result<bool, String> {
    if active_columns == 0
        || lane >= active_columns
        || union.key_major_values.len() != union.keys.len().saturating_mul(active_columns)
    {
        return Err("p3 union lane shape is invalid".to_string());
    }
    Ok(union
        .key_major_values
        .chunks_exact(active_columns)
        .any(|values| values[lane] != 0))
}

fn write_generation_artifacts(
    output: &Path,
    job: &P3ProductionJobKey,
    plan_sha256: &str,
    rows: &[Vec<GaussianResidue>],
    expanded: &[u64],
    generation: u64,
    final_publication: bool,
) -> Result<Vec<P3PublishedArtifact>, String> {
    let ordinals = job.global_ordinals()?;
    if rows.len() != ordinals.len() || expanded.len() != ordinals.len() {
        return Err("p3 production artifact lane shape changed".to_string());
    }
    let mut inventory = Vec::with_capacity(ordinals.len());
    for (lane, ordinal) in ordinals.iter().copied().enumerate() {
        let semantic = p3_column_semantic_sha256(job.prime(), ordinal, &rows[lane]);
        let column = ModularP3FunctionalColumn {
            prime: job.prime(),
            global_ordinal: ordinal,
            rows: rows[lane].clone(),
            expanded_contributions: expanded[lane],
            semantic_sha256: semantic.clone(),
        };
        let bytes = encode_p3_column_artifact(plan_sha256, &column)?;
        let path = if final_publication {
            final_artifact_path(output, ordinal, job.prime_index)
        } else {
            job_directory(output, job)
                .join("partial")
                .join(format!("generation-{generation:08}-c{ordinal}.adfxp3"))
        };
        atomic_bytes(&path, &bytes)?;
        let relative_path = path
            .strip_prefix(output)
            .map_err(|_| "p3 artifact escaped output directory".to_string())?
            .to_string_lossy()
            .to_string();
        inventory.push(P3PublishedArtifact {
            global_ordinal: ordinal,
            relative_path,
            artifact_sha256: format!("{:x}", Sha256::digest(&bytes)),
            column_semantic_sha256: semantic,
            expanded_contributions: expanded[lane],
        });
    }
    Ok(inventory)
}

fn restore_p3_rows(
    output: &Path,
    job: &P3ProductionJobKey,
    checkpoint: &P3ProductionCheckpoint,
) -> Result<Vec<Vec<GaussianResidue>>, String> {
    if checkpoint.next_word_ordinal == 0 {
        if !checkpoint.artifacts.is_empty() {
            return Err("initial p3 production checkpoint contains partial artifacts".to_string());
        }
        return Ok(vec![
            vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
            job.global_ordinals()?.len()
        ]);
    }
    let ordinals = job.global_ordinals()?;
    if checkpoint.artifacts.len() != ordinals.len()
        || checkpoint
            .artifacts
            .iter()
            .map(|artifact| artifact.global_ordinal)
            .collect::<Vec<_>>()
            != ordinals
    {
        return Err("resumed p3 production checkpoint lacks lane artifacts".to_string());
    }
    checkpoint
        .artifacts
        .iter()
        .map(|artifact| {
            validate_artifact(output, job, artifact, &checkpoint.flat_plan_sha256)
                .map(|column| column.rows)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn remove_superseded_partials(output: &Path, artifacts: &[P3PublishedArtifact]) {
    for artifact in artifacts {
        let relative = Path::new(&artifact.relative_path);
        if relative
            .components()
            .any(|component| component.as_os_str() == "partial")
        {
            let _ = fs::remove_file(output.join(relative));
        }
    }
}

fn report_from_checkpoint(
    checkpoint: &P3ProductionCheckpoint,
) -> Result<P3ProductionReport, String> {
    if checkpoint.state != "artifact_published"
        || checkpoint.next_word_ordinal != checkpoint.total_words
    {
        return Err("p3 production checkpoint is not publication-complete".to_string());
    }
    let ordinals = checkpoint.job.global_ordinals()?;
    Ok(P3ProductionReport {
        schema_version: P3_PRODUCTION_RUN_SCHEMA.to_string(),
        manifest_sha256: checkpoint.manifest_sha256.clone(),
        job: checkpoint.job.clone(),
        prime: checkpoint.prime,
        group_plan_sha256: checkpoint.group_plan_sha256.clone(),
        source_group_sha256: checkpoint.source_group_sha256.clone(),
        pbw_plan_sha256: checkpoint.pbw_plan_sha256.clone(),
        source_label: checkpoint.source_label.clone(),
        source_copies: checkpoint.source_copies.clone(),
        flat_plan_sha256: checkpoint.flat_plan_sha256.clone(),
        completed_words: checkpoint.next_word_ordinal,
        total_words: checkpoint.total_words,
        batches_completed: checkpoint.p3_batches_completed,
        raw_terms_per_column: checkpoint.raw_terms_per_column.clone(),
        source_terms_sha256: checkpoint
            .source_hashers
            .iter()
            .enumerate()
            .map(|(lane, hash)| {
                finish_source_hash(
                    hash,
                    ordinals[lane],
                    &checkpoint.source_label,
                    checkpoint.source_copies[lane],
                    checkpoint.raw_terms_per_column[lane],
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        expanded_contributions_per_column: checkpoint.expanded_contributions_per_column.clone(),
        reduced_expanded_contributions_per_column: checkpoint
            .reduced_expanded_contributions_per_column
            .clone(),
        p2_batches_folded: checkpoint.p2_batches_folded,
        p2_rows_sha256: checkpoint
            .p2_rows
            .iter()
            .enumerate()
            .map(|(lane, rows)| p2_rows_sha256(checkpoint.prime, ordinals[lane], rows))
            .collect(),
        kernel_milliseconds: checkpoint.kernel_milliseconds,
        artifacts: checkpoint.artifacts.clone(),
        completed_unix_ms: unix_ms(),
        passed: true,
    })
}

fn p2_rows_sha256(prime: u32, global_ordinal: usize, rows: &[GaussianResidue]) -> String {
    let mut hash = Sha256::new();
    hash.update(P3_PRODUCTION_RUN_SCHEMA.as_bytes());
    hash.update(b"\0embedded-p2-rows-v1\0");
    hash.update(prime.to_le_bytes());
    hash.update((global_ordinal as u64).to_le_bytes());
    hash.update((rows.len() as u64).to_le_bytes());
    for value in rows {
        hash.update(value.real.to_le_bytes());
        hash.update(value.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// Production execution follows the established full-inventory source path.
/// Each exact raw batch is consumed by both the existing p2 executor and the
/// p3 accumulator. Durable checkpoints are committed only after a whole PBW
/// word and all lane batches have completed.
#[cfg(feature = "cuda")]
pub(crate) fn run_production_job(
    job: &P3ProductionJobKey,
    map_directory: &Path,
    output: &Path,
    device: i32,
    total_device_hard_cap_bytes: u64,
    live: &crate::second_momentum_gpu_progress::LiveProgress,
) -> Result<P3ProductionReport, String> {
    use crate::eleven_dimensional_second_momentum_gpu::{
        CudaModularFx, CudaModularP3, ModularFxStaticData, PersistentCudaGroupExecutor,
        build_p3_modular_flat_plan,
    };
    use crate::second_momentum_gpu_group::{
        GroupRuntimeIdentity, GroupWordOrchestrationConfig, prepare_cuda_column_group,
        prepare_full_cuda_column_group, source_group_identity_sha256,
    };
    use crate::second_momentum_gpu_jobs::GroupExecutionConfig;
    use crate::second_momentum_gpu_progress::{
        GpuBatchProgress, GroupLiveProgress, SourceVisitorProgress,
    };

    if device < 0 || total_device_hard_cap_bytes < 512 * 1024 * 1024 {
        return Err("p3 production device or hard cap is invalid".to_string());
    }
    write_or_validate_manifest(output)?;
    let _lock = acquire_lock(output, job)?;
    let manifest = build_manifest()?;
    let directory = job_directory(output, job);
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let events = directory.join("events.jsonl");
    let static_data = ModularFxStaticData::build(job.prime())?;
    let p3_flat = build_p3_modular_flat_plan(&static_data)?;
    let p3_raw_fanout = p3_flat.raw_fanout_table()?;
    let p2_probe = CudaModularFx::new(&static_data, device)?;
    let runtime = GroupRuntimeIdentity {
        prime: job.prime(),
        static_semantic_sha256: static_data.semantic_sha256().to_string(),
        flat_plan_sha256: p2_probe.flat_plan_sha256().to_string(),
    };
    drop(p2_probe);
    let ordinals = job.global_ordinals()?;
    let plan = if ordinals.iter().all(|ordinal| *ordinal <= 52) {
        prepare_full_cuda_column_group(&ordinals, map_directory, runtime)?
    } else if ordinals.iter().all(|ordinal| (53..=61).contains(ordinal)) {
        prepare_cuda_column_group(
            GpuFxTranche::Two0001,
            &ordinals
                .iter()
                .map(|ordinal| ordinal - 53)
                .collect::<Vec<_>>(),
            runtime,
        )?
    } else if ordinals.iter().all(|ordinal| (62..=76).contains(ordinal)) {
        prepare_cuda_column_group(
            GpuFxTranche::Three0001,
            &ordinals
                .iter()
                .map(|ordinal| ordinal - 62)
                .collect::<Vec<_>>(),
            runtime,
        )?
    } else {
        return Err("p3 production group crosses a certified tranche boundary".to_string());
    };
    if plan.ordered_global_ordinals != ordinals {
        return Err("p3 production group preflight changed column order".to_string());
    }
    let group_plan_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&plan).map_err(|error| error.to_string())?)
    );
    let source_group_sha256 = source_group_identity_sha256(&plan);
    if validate_completed_job(output, job)? {
        let report: P3ProductionReport = serde_json::from_reader(
            File::open(report_path(output, job)).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if report.group_plan_sha256 != group_plan_sha256
            || report.source_group_sha256 != source_group_sha256
            || report.pbw_plan_sha256 != plan.pbw_plan_sha256
            || report.source_label != plan.source_dynkin_label
            || report.source_copies != plan.ordered_source_copies
            || report.flat_plan_sha256 != p3_flat.semantic_sha256()
            || report.total_words != plan.pbw_word_count
        {
            return Err(
                "completed p3 job does not match the canonical rebuilt source plan".to_string(),
            );
        }
        return Ok(report);
    }
    let mut execution = GroupExecutionConfig::from_environment()?;
    let requested_p3_cap = std::env::var("ADYNKRA_P3_CONTRACTION_CAP_BYTES")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()
        .map_err(|_| "ADYNKRA_P3_CONTRACTION_CAP_BYTES must be unsigned".to_string())?
        .unwrap_or(2 * 1024 * 1024 * 1024);
    if requested_p3_cap >= total_device_hard_cap_bytes {
        return Err("p3 contraction cap leaves no budget for exact source traversal".to_string());
    }
    execution.aggregate_device_cap_bytes = execution
        .aggregate_device_cap_bytes
        .min(total_device_hard_cap_bytes - requested_p3_cap);
    if execution.contraction_device_cap_bytes >= execution.aggregate_device_cap_bytes {
        return Err("p3 total cap is too small for the configured p2 traversal budget".to_string());
    }
    let p3_cap = requested_p3_cap;

    let mut checkpoint = if checkpoint_path(output, job).exists() {
        let observed: P3ProductionCheckpoint = serde_json::from_reader(
            File::open(checkpoint_path(output, job)).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if observed.schema_version != P3_PRODUCTION_CHECKPOINT_SCHEMA
            || observed.manifest_sha256 != manifest.manifest_sha256
            || observed.job != *job
            || observed.prime != job.prime()
            || observed.group_plan_sha256 != group_plan_sha256
            || observed.source_group_sha256 != source_group_sha256
            || observed.pbw_plan_sha256 != plan.pbw_plan_sha256
            || observed.source_label != plan.source_dynkin_label
            || observed.source_copies != plan.ordered_source_copies
            || observed.flat_plan_sha256 != p3_flat.semantic_sha256()
            || observed.total_words != plan.pbw_word_count
            || observed.next_word_ordinal > observed.total_words
            || (observed.state == "running"
                && observed.checkpoint_generation != observed.next_word_ordinal as u64)
            || (observed.state == "artifact_published"
                && (observed.next_word_ordinal != observed.total_words
                    || observed.checkpoint_generation != observed.total_words as u64 + 1))
            || observed.raw_terms_per_column.len() != plan.active_columns
            || observed.expanded_contributions_per_column.len() != plan.active_columns
            || observed.reduced_expanded_contributions_per_column.len() != plan.active_columns
            || observed.source_hashers.len() != plan.active_columns
            || observed.p2_rows.len() != plan.active_columns
            || observed.p2_rows.iter().any(|rows| {
                rows.len() != crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
                    || rows
                        .iter()
                        .any(|value| value.real >= job.prime() || value.imaginary >= job.prime())
            })
            || observed.artifacts.len()
                != if observed.state == "running" && observed.next_word_ordinal == 0 {
                    0
                } else {
                    plan.active_columns
                }
            || observed
                .artifacts
                .iter()
                .map(|artifact| artifact.global_ordinal)
                .collect::<Vec<_>>()
                != if observed.state == "running" && observed.next_word_ordinal == 0 {
                    Vec::new()
                } else {
                    ordinals.clone()
                }
            || observed
                .artifacts
                .iter()
                .enumerate()
                .any(|(lane, artifact)| {
                    artifact.expanded_contributions
                        != observed
                            .expanded_contributions_per_column
                            .get(lane)
                            .copied()
                            .unwrap_or(u64::MAX)
                        || if observed.state == "artifact_published" {
                            PathBuf::from(&artifact.relative_path)
                                != final_artifact_path(
                                    output,
                                    artifact.global_ordinal,
                                    job.prime_index,
                                )
                                .strip_prefix(output)
                                .unwrap_or_else(|_| Path::new("__invalid__"))
                        } else if observed.next_word_ordinal == 0 {
                            true
                        } else {
                            PathBuf::from(&artifact.relative_path)
                                != PathBuf::from("jobs").join(job.id()).join("partial").join(
                                    format!(
                                        "generation-{:08}-c{}.adfxp3",
                                        observed.checkpoint_generation, artifact.global_ordinal
                                    ),
                                )
                        }
                })
        {
            return Err("p3 production checkpoint identity is stale or invalid".to_string());
        }
        if observed.state == "artifact_published" {
            let report = report_from_checkpoint(&observed)?;
            for artifact in &report.artifacts {
                validate_artifact(output, job, artifact, &report.flat_plan_sha256)?;
            }
            atomic_json(&report_path(output, job), &report)?;
            if !validate_completed_job(output, job)? {
                return Err(
                    "adopted p3 checkpoint did not produce a valid completed report".to_string(),
                );
            }
            append_event(
                &events,
                &serde_json::json!({
                    "schema_version": P3_PRODUCTION_RUN_SCHEMA,
                    "event": "job_adopted_from_checkpoint",
                    "timestamp_unix_ms": unix_ms(),
                    "job_id": job.id(),
                    "checkpoint_generation": observed.checkpoint_generation,
                }),
            )?;
            return Ok(report);
        }
        if observed.state != "running" {
            return Err("p3 production checkpoint state is invalid".to_string());
        }
        observed
    } else {
        P3ProductionCheckpoint {
            schema_version: P3_PRODUCTION_CHECKPOINT_SCHEMA.to_string(),
            manifest_sha256: manifest.manifest_sha256.clone(),
            job: job.clone(),
            prime: job.prime(),
            group_plan_sha256: group_plan_sha256.clone(),
            source_group_sha256: source_group_sha256.clone(),
            pbw_plan_sha256: plan.pbw_plan_sha256.clone(),
            source_label: plan.source_dynkin_label.clone(),
            source_copies: plan.ordered_source_copies.clone(),
            flat_plan_sha256: p3_flat.semantic_sha256().to_string(),
            next_word_ordinal: 0,
            total_words: plan.pbw_word_count,
            next_batch_ordinal: 0,
            p3_batches_completed: 0,
            raw_terms_per_column: vec![0; plan.active_columns],
            expanded_contributions_per_column: vec![0; plan.active_columns],
            reduced_expanded_contributions_per_column: vec![0; plan.active_columns],
            source_hashers: seed_source_hashers(&plan)?,
            p2_rows: vec![
                vec![
                    GaussianResidue::zero();
                    crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT
                ];
                plan.active_columns
            ],
            p2_batches_folded: 0,
            kernel_milliseconds: 0.0,
            device_resident_bytes: 0,
            device_high_water_bytes: 0,
            device_hard_cap_bytes: total_device_hard_cap_bytes,
            checkpoint_generation: 0,
            state: "running".to_string(),
            artifacts: Vec::new(),
            updated_unix_ms: unix_ms(),
        }
    };
    let mut p3_rows = restore_p3_rows(output, job, &checkpoint)?;
    let mut p2 = if ordinals.iter().all(|ordinal| *ordinal <= 52) {
        PersistentCudaGroupExecutor::new_full(
            plan.clone(),
            &static_data,
            device,
            execution.max_union_keys_per_batch,
            execution.aggregate_device_cap_bytes,
            execution.contraction_device_cap_bytes,
            execution.per_lane_host_staging_cap_bytes,
            execution.download_chunk_terms,
            map_directory,
        )?
    } else {
        PersistentCudaGroupExecutor::new(
            plan.clone(),
            &static_data,
            device,
            execution.max_union_keys_per_batch,
            execution.aggregate_device_cap_bytes,
            execution.contraction_device_cap_bytes,
            execution.per_lane_host_staging_cap_bytes,
            execution.download_chunk_terms,
        )?
    };
    p2.restore_columns(checkpoint.p2_rows.clone(), checkpoint.p2_batches_folded)?;
    let mut p3 = CudaModularP3::new_with_device_cap(&static_data, &p3_flat, device, p3_cap)?;
    p3.reset_persistent_columns(&p3_rows)?;
    if checkpoint.checkpoint_generation == 0 {
        atomic_json(&checkpoint_path(output, job), &checkpoint)?;
    }
    append_event(
        &events,
        &serde_json::json!({
            "schema_version": P3_PRODUCTION_RUN_SCHEMA,
            "event": "job_start",
            "timestamp_unix_ms": unix_ms(),
            "job_id": job.id(),
            "global_ordinals": ordinals,
            "prime": job.prime(),
            "resume_word_ordinal": checkpoint.next_word_ordinal,
            "total_words": checkpoint.total_words,
            "p2_traversal_cap_bytes": execution.aggregate_device_cap_bytes,
            "p3_contraction_cap_bytes": p3_cap,
            "total_device_hard_cap_bytes": total_device_hard_cap_bytes,
        }),
    )?;

    let canary_max_words = std::env::var("ADYNKRA_P3_CANARY_MAX_WORDS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| "ADYNKRA_P3_CANARY_MAX_WORDS must be a positive integer".to_string())?;
    if canary_max_words == Some(0) {
        return Err("ADYNKRA_P3_CANARY_MAX_WORDS must be positive".to_string());
    }
    let canary_start_word = checkpoint.next_word_ordinal;
    for word in checkpoint.next_word_ordinal..checkpoint.total_words {
        let first_batch = checkpoint.next_batch_ordinal;
        let fanout_cache_limit = execution.max_union_keys_per_batch.max(1);
        let mut raw_fanout_by_key = BTreeMap::<u64, u64>::new();
        let mut word_source_hashers = checkpoint.source_hashers.clone();
        let mut word_raw_terms = checkpoint.raw_terms_per_column.clone();
        let mut word_raw_expanded = checkpoint.expanded_contributions_per_column.clone();
        let orchestration = p2.run_word_synchronous_batched_with_union(
            GroupWordOrchestrationConfig {
                start_word_ordinal: word,
                end_word_ordinal_exclusive: word + 1,
                first_global_batch_ordinal: first_batch,
                raw_batch_term_cap_per_lane: execution.raw_batch_terms_per_lane,
                max_union_keys_per_batch: execution.max_union_keys_per_batch,
                aggregate_host_payload_cap_bytes: execution.aggregate_host_payload_cap_bytes,
            },
            |lane, _, terms| {
                update_source_hash(&mut word_source_hashers[lane], word, terms)?;
                word_raw_terms[lane] = word_raw_terms[lane]
                    .checked_add(terms.len() as u64)
                    .ok_or_else(|| "p3 production raw term count overflow".to_string())?;
                for term in terms {
                    if term.coefficient.rem_euclid(i128::from(job.prime())) == 0 {
                        continue;
                    }
                    let packed_key = p3_recoupling_key(term)?;
                    // Fanout depends on free spinor and exterior mask, not on
                    // the momentum pair. Normalize pair bits for cache reuse.
                    let key = packed_key & !(0xff_u64 << 32);
                    let fanout = if let Some(&fanout) = raw_fanout_by_key.get(&key) {
                        fanout
                    } else {
                        let mut normalized = *term;
                        normalized.coefficient = 1;
                        let fanout = p3_raw_fanout.fanout(&normalized)?;
                        if raw_fanout_by_key.len() >= fanout_cache_limit {
                            raw_fanout_by_key.clear();
                        }
                        raw_fanout_by_key.insert(key, fanout);
                        fanout
                    };
                    word_raw_expanded[lane] = word_raw_expanded[lane]
                        .checked_add(fanout)
                        .ok_or_else(|| "p3 production raw expanded count overflow".to_string())?;
                }
                Ok(())
            },
            |_, union| {
                let timing = p3.accumulate_reduced_union_multilane_persistent(
                    &union.keys,
                    &union.key_major_values,
                    plan.active_columns,
                )?;
                let mut active_lanes = 0_u64;
                for lane in 0..plan.active_columns {
                    if timing.source_counts[lane] == 0 {
                        continue;
                    }
                    active_lanes += 1;
                    checkpoint.reduced_expanded_contributions_per_column[lane] = checkpoint
                        .reduced_expanded_contributions_per_column[lane]
                        .checked_add(timing.expanded_contributions[lane])
                        .ok_or_else(|| "p3 reduced expanded count overflow".to_string())?;
                }
                checkpoint.kernel_milliseconds += f64::from(timing.kernel_milliseconds);
                checkpoint.p3_batches_completed = checkpoint
                    .p3_batches_completed
                    .checked_add(active_lanes)
                    .ok_or_else(|| "p3 production batch count overflow".to_string())?;
                checkpoint.device_resident_bytes = timing.resident_bytes;
                checkpoint.device_high_water_bytes = checkpoint
                    .device_high_water_bytes
                    .max(timing.buffer_high_water_bytes);
                live.record_gpu_batch(GpuBatchProgress {
                    batches_completed: checkpoint.p3_batches_completed,
                    last_batch_ms: f64::from(timing.kernel_milliseconds),
                    total_batch_ms: checkpoint.kernel_milliseconds,
                    last_contract_ms: f64::from(timing.kernel_milliseconds),
                    total_contract_ms: checkpoint.kernel_milliseconds,
                    ..GpuBatchProgress::default()
                });
                Ok(())
            },
            |_| Ok(()),
            |completed_word, completions| {
                if completed_word != word || completions.len() != plan.active_columns {
                    return Err("p3 production word completion identity changed".to_string());
                }
                Ok(())
            },
        )?;
        checkpoint.source_hashers = word_source_hashers;
        checkpoint.raw_terms_per_column = word_raw_terms;
        checkpoint.expanded_contributions_per_column = word_raw_expanded;
        p3_rows = p3.download_persistent_columns(plan.active_columns)?;
        checkpoint.next_word_ordinal = orchestration.next_word_ordinal;
        checkpoint.next_batch_ordinal = orchestration.next_global_batch_ordinal;
        checkpoint.p2_rows = p2.final_columns().to_vec();
        checkpoint.p2_batches_folded = p2.batches_folded();
        checkpoint.checkpoint_generation = checkpoint
            .checkpoint_generation
            .checked_add(1)
            .ok_or_else(|| "p3 checkpoint generation overflow".to_string())?;
        let superseded_artifacts = std::mem::take(&mut checkpoint.artifacts);
        checkpoint.artifacts = write_generation_artifacts(
            output,
            job,
            p3_flat.semantic_sha256(),
            &p3_rows,
            &checkpoint.expanded_contributions_per_column,
            checkpoint.checkpoint_generation,
            false,
        )?;
        checkpoint.updated_unix_ms = unix_ms();
        atomic_json(&checkpoint_path(output, job), &checkpoint)?;
        remove_superseded_partials(output, &superseded_artifacts);
        live.update_source(SourceVisitorProgress {
            word: Some(word as u64),
            raw_terms_emitted: checkpoint.raw_terms_per_column.iter().sum(),
            batches_flushed: checkpoint.next_batch_ordinal,
            hard_memory_cap_bytes: execution.aggregate_host_payload_cap_bytes,
            eta_sample_count: checkpoint.next_batch_ordinal,
            ..SourceVisitorProgress::default()
        });
        live.update_group(GroupLiveProgress {
            group_id: Some(plan.group_id.clone()),
            words_completed: checkpoint.next_word_ordinal,
            words_total: checkpoint.total_words,
            global_batch_ordinal: checkpoint.next_batch_ordinal,
            raw_terms_per_column: checkpoint.raw_terms_per_column.clone(),
            aggregate_host_cap_bytes: execution.aggregate_host_payload_cap_bytes,
            device_resident_bytes: checkpoint.device_resident_bytes,
            device_high_water_bytes: checkpoint.device_high_water_bytes,
            aggregate_device_cap_bytes: total_device_hard_cap_bytes,
            checkpoint_generation: checkpoint.checkpoint_generation,
            checkpoint_written_unix_ms: Some(checkpoint.updated_unix_ms),
            ..GroupLiveProgress::default()
        });
        append_event(
            &events,
            &serde_json::json!({
                "schema_version": P3_PRODUCTION_RUN_SCHEMA,
                "event": "word_checkpoint",
                "timestamp_unix_ms": checkpoint.updated_unix_ms,
                "job_id": job.id(),
                "completed_word_ordinal": word,
                "next_word_ordinal": checkpoint.next_word_ordinal,
                "total_words": checkpoint.total_words,
                "next_batch_ordinal": checkpoint.next_batch_ordinal,
                "raw_terms_per_column": checkpoint.raw_terms_per_column,
                "expanded_contributions_per_column": checkpoint.expanded_contributions_per_column,
                "reduced_expanded_contributions_per_column": checkpoint.reduced_expanded_contributions_per_column,
                "checkpoint_generation": checkpoint.checkpoint_generation,
            }),
        )?;
        if canary_max_words.is_some_and(|limit| {
            checkpoint.next_word_ordinal < checkpoint.total_words
                && checkpoint.next_word_ordinal - canary_start_word >= limit
        }) {
            append_event(
                &events,
                &serde_json::json!({
                    "schema_version": P3_PRODUCTION_RUN_SCHEMA,
                    "event": "canary_stopped_after_durable_checkpoint",
                    "timestamp_unix_ms": unix_ms(),
                    "job_id": job.id(),
                    "next_word_ordinal": checkpoint.next_word_ordinal,
                    "checkpoint_generation": checkpoint.checkpoint_generation,
                }),
            )?;
            return Err("intentional p3 canary stop after durable word checkpoint".to_string());
        }
    }

    let superseded_artifacts = std::mem::take(&mut checkpoint.artifacts);
    checkpoint.artifacts = write_generation_artifacts(
        output,
        job,
        p3_flat.semantic_sha256(),
        &p3_rows,
        &checkpoint.expanded_contributions_per_column,
        checkpoint.checkpoint_generation + 1,
        true,
    )?;
    checkpoint.state = "artifact_published".to_string();
    checkpoint.checkpoint_generation += 1;
    checkpoint.updated_unix_ms = unix_ms();
    atomic_json(&checkpoint_path(output, job), &checkpoint)?;
    let report = report_from_checkpoint(&checkpoint)?;
    atomic_json(&report_path(output, job), &report)?;
    remove_superseded_partials(output, &superseded_artifacts);
    append_event(
        &events,
        &serde_json::json!({
            "schema_version": P3_PRODUCTION_RUN_SCHEMA,
            "event": "job_complete",
            "timestamp_unix_ms": unix_ms(),
            "job_id": job.id(),
            "raw_terms_per_column": report.raw_terms_per_column,
            "expanded_contributions_per_column": report.expanded_contributions_per_column,
            "reduced_expanded_contributions_per_column": report.reduced_expanded_contributions_per_column,
            "kernel_milliseconds": report.kernel_milliseconds,
            "artifacts": report.artifacts,
        }),
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "adynkra-{label}-{}-{}",
            std::process::id(),
            unix_ms()
        ))
    }

    #[test]
    fn p3_production_manifest_partitions_all_77_and_selects_portably() {
        let groups = canonical_p3_groups().unwrap();
        assert_eq!(
            groups.iter().flatten().copied().collect::<Vec<_>>(),
            (0..77).collect::<Vec<_>>()
        );
        assert!(groups.iter().all(|group| (1..=3).contains(&group.len())));
        let manifest = build_manifest().unwrap();
        assert_eq!(manifest.jobs.len(), groups.len() * 3);
        assert_eq!(parse_job_list("all@0").unwrap().len(), groups.len());
        assert_eq!(parse_job_list("0-2@1").unwrap().len(), 3);
        assert!(parse_job_list(&format!("{}@0", groups.len())).is_err());
    }

    #[test]
    fn p3_production_union_lane_detection_handles_unequal_empty_tail() {
        use crate::second_momentum_gpu_group::{ExactUnionBatch, UnionBatchTelemetry};
        let union = ExactUnionBatch {
            keys: vec![11, 22],
            key_major_values: vec![3, 5, 0, 7, 9, 0],
            telemetry: UnionBatchTelemetry {
                schema_version: "test".to_string(),
                group_id: "test".to_string(),
                batch_ordinal: 0,
                active_columns: 3,
                union_key_count: 2,
                keys_by_present_lane_count: vec![0; 4],
                reduced_terms_per_column: vec![2, 2, 0],
                key_capacity: 2,
                value_capacity: 6,
                host_capacity_bytes: 0,
                union_milliseconds: 0,
                union_keys_per_second: 0,
                deterministic_batch_sha256: "0".repeat(64),
            },
        };
        assert!(union_lane_has_terms(&union, 3, 0).unwrap());
        assert!(union_lane_has_terms(&union, 3, 1).unwrap());
        assert!(!union_lane_has_terms(&union, 3, 2).unwrap());
        assert!(union_lane_has_terms(&union, 2, 0).is_err());
        assert!(union_lane_has_terms(&union, 3, 3).is_err());
    }

    #[test]
    fn p3_production_checkpoint_canary_adopts_final_artifacts_and_refuses_mutation() {
        let output = temporary_directory("p3-production-checkpoint");
        let _ = fs::remove_dir_all(&output);
        write_or_validate_manifest(&output).unwrap();
        let job = P3ProductionJobKey::new(0, 0).unwrap();
        let ordinals = job.global_ordinals().unwrap();
        let rows = ordinals
            .iter()
            .enumerate()
            .map(|(lane, _)| {
                let mut rows = vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
                rows[lane + 1].real = (lane + 3) as u32;
                rows
            })
            .collect::<Vec<_>>();
        let expanded = vec![7; ordinals.len()];
        let plan = "a".repeat(64);
        let artifacts =
            write_generation_artifacts(&output, &job, &plan, &rows, &expanded, 1, true).unwrap();
        let manifest = build_manifest().unwrap();
        let mut hashers = Vec::new();
        for ordinal in &ordinals {
            let mut hash = CheckpointableSha256::new();
            hash.update(&ordinal.to_le_bytes()).unwrap();
            hashers.push(hash);
        }
        let checkpoint = P3ProductionCheckpoint {
            schema_version: P3_PRODUCTION_CHECKPOINT_SCHEMA.to_string(),
            manifest_sha256: manifest.manifest_sha256,
            job: job.clone(),
            prime: job.prime(),
            group_plan_sha256: "b".repeat(64),
            source_group_sha256: "c".repeat(64),
            pbw_plan_sha256: "d".repeat(64),
            source_label: "(10001)".to_string(),
            source_copies: vec![1; ordinals.len()],
            flat_plan_sha256: plan,
            next_word_ordinal: 1,
            total_words: 1,
            next_batch_ordinal: 2,
            p3_batches_completed: 2,
            raw_terms_per_column: vec![11; ordinals.len()],
            expanded_contributions_per_column: expanded,
            reduced_expanded_contributions_per_column: vec![5; ordinals.len()],
            source_hashers: hashers,
            p2_rows: vec![Vec::new(); ordinals.len()],
            p2_batches_folded: 2,
            kernel_milliseconds: 1.25,
            device_resident_bytes: 10,
            device_high_water_bytes: 20,
            device_hard_cap_bytes: 30,
            checkpoint_generation: 2,
            state: "artifact_published".to_string(),
            artifacts,
            updated_unix_ms: unix_ms(),
        };
        atomic_json(&checkpoint_path(&output, &job), &checkpoint).unwrap();
        let report = report_from_checkpoint(&checkpoint).unwrap();
        atomic_json(&report_path(&output, &job), &report).unwrap();
        assert!(validate_completed_job(&output, &job).unwrap());
        fs::remove_file(report_path(&output, &job)).unwrap();
        let adopted = report_from_checkpoint(&checkpoint).unwrap();
        atomic_json(&report_path(&output, &job), &adopted).unwrap();
        assert!(validate_completed_job(&output, &job).unwrap());

        // An unreferenced next-generation partial models a crash between lane
        // writes. It must never be adopted over the committed inventory.
        let stray = job_directory(&output, &job)
            .join("partial")
            .join("generation-00000003-c999.adfxp3");
        atomic_bytes(&stray, b"uncommitted partial").unwrap();
        assert!(validate_completed_job(&output, &job).unwrap());

        let mut missing_lane = adopted.clone();
        missing_lane.artifacts.clear();
        atomic_json(&report_path(&output, &job), &missing_lane).unwrap();
        assert!(validate_completed_job(&output, &job).is_err());

        let mut wrong_counter = adopted.clone();
        wrong_counter.expanded_contributions_per_column[0] += 1;
        atomic_json(&report_path(&output, &job), &wrong_counter).unwrap();
        assert!(validate_completed_job(&output, &job).is_err());

        let mut partial_path = adopted.clone();
        partial_path.artifacts[0].relative_path =
            format!("jobs/{}/partial/generation-00000001-c0.adfxp3", job.id());
        atomic_json(&report_path(&output, &job), &partial_path).unwrap();
        assert!(validate_completed_job(&output, &job).is_err());

        atomic_json(&report_path(&output, &job), &adopted).unwrap();
        let artifact_path = output.join(&checkpoint.artifacts[0].relative_path);
        let mut bytes = fs::read(&artifact_path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        atomic_bytes(&artifact_path, &bytes).unwrap();
        assert!(validate_completed_job(&output, &job).is_err());
        fs::remove_dir_all(output).unwrap();
    }
}
