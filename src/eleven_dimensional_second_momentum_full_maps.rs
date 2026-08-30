//! Durable exact source-to-intermediate maps for the 49 columns outside the
//! original 28-column production slice.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use crate::eleven_dimensional_second_momentum_full_inventory::{
    Level12FixtureRef, level12_fixtures, missing_49_column_specs,
};

const SCHEMA_VERSION: &str = "adynkra-11d-second-momentum-missing-maps-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MissingMapJob {
    pub target_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
}

impl MissingMapJob {
    pub(crate) fn key(&self) -> String {
        format!(
            "{}_from_{}_copy{}",
            self.target_dynkin_label, self.source_dynkin_label, self.source_copy
        )
    }
}

pub(crate) fn worklist() -> Vec<MissingMapJob> {
    missing_49_column_specs()
        .into_iter()
        .map(|column| MissingMapJob {
            target_dynkin_label: column.intermediate_dynkin_label,
            source_dynkin_label: column.source_dynkin_label,
            source_copy: column.source_copy,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn parse_job_list(spec: &str) -> Result<Vec<usize>, String> {
    let total = worklist().len();
    if spec == "all" {
        return Ok((0..total).collect());
    }
    let mut ordinals = Vec::new();
    for token in spec.split(',') {
        if token.is_empty() {
            return Err("empty map job-list token".to_string());
        }
        if let Some((start, end)) = token.split_once('-') {
            let start = start
                .parse::<usize>()
                .map_err(|_| format!("invalid map job ordinal {start}"))?;
            let end = end
                .parse::<usize>()
                .map_err(|_| format!("invalid map job ordinal {end}"))?;
            if start > end {
                return Err(format!("reversed map job range {token}"));
            }
            ordinals.extend(start..=end);
        } else {
            ordinals.push(
                token
                    .parse::<usize>()
                    .map_err(|_| format!("invalid map job ordinal {token}"))?,
            );
        }
    }
    if ordinals.is_empty()
        || ordinals.windows(2).any(|pair| pair[0] >= pair[1])
        || ordinals.iter().any(|ordinal| *ordinal >= total)
    {
        return Err(format!(
            "map job list must be strictly increasing and lie in 0..{total}"
        ));
    }
    Ok(ordinals)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MissingAbstractMapCheckpoint {
    pub schema_version: String,
    pub target_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coefficient_width_bytes: usize,
    pub certificate_sha256: String,
    pub elapsed_milliseconds: u128,
    pub observed_process_rss_bytes: Option<u64>,
    pub certificate: crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MissingEmbeddedMapCheckpoint {
    pub schema_version: String,
    pub job: MissingMapJob,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coefficient_width_bytes: usize,
    pub abstract_certificate_sha256: String,
    pub coupled_map_sha256: String,
    pub target_dimension: u64,
    pub elapsed_milliseconds: u128,
    pub observed_process_rss_bytes: Option<u64>,
    pub certificate: crate::eleven_dimensional_level16_couplings::EmbeddedCouplingCertificate,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MissingMapProgress {
    pub schema_version: String,
    pub event: String,
    pub state: String,
    pub phase: String,
    pub job_ordinal: usize,
    pub jobs_total: usize,
    pub selected_job_position: usize,
    pub selected_jobs_total: usize,
    pub completed_jobs: usize,
    pub target_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub elapsed_seconds: f64,
    pub job_elapsed_seconds: f64,
    pub estimated_remaining_seconds: Option<f64>,
    pub observed_process_rss_bytes: Option<u64>,
    pub abstract_checkpoint_path: String,
    pub embedded_checkpoint_path: String,
    pub message: String,
}

pub(crate) struct MissingMapProgressReporter {
    shared: Arc<Mutex<MissingMapProgress>>,
    stop: Arc<AtomicBool>,
    heartbeat: Option<thread::JoinHandle<()>>,
    status_path: PathBuf,
}

impl MissingMapProgressReporter {
    pub(crate) fn start(status_path: PathBuf, jobs_total: usize) -> io::Result<Self> {
        let initial = MissingMapProgress {
            schema_version: format!("{SCHEMA_VERSION}-progress"),
            event: "worker_start".to_string(),
            state: "running".to_string(),
            phase: "initializing".to_string(),
            job_ordinal: 0,
            jobs_total,
            selected_job_position: 0,
            selected_jobs_total: jobs_total,
            completed_jobs: 0,
            target_dynkin_label: String::new(),
            source_dynkin_label: String::new(),
            source_copy: 0,
            elapsed_seconds: 0.0,
            job_elapsed_seconds: 0.0,
            estimated_remaining_seconds: None,
            observed_process_rss_bytes: observed_process_rss_bytes(),
            abstract_checkpoint_path: String::new(),
            embedded_checkpoint_path: String::new(),
            message: "full-map worker initialized".to_string(),
        };
        atomic_json(&status_path, &initial)?;
        println!(
            "{}",
            serde_json::to_string(&initial).expect("serialize full-map worker start")
        );
        let shared = Arc::new(Mutex::new(initial));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_shared = Arc::clone(&shared);
        let thread_stop = Arc::clone(&stop);
        let thread_status_path = status_path.clone();
        let started = Instant::now();
        let heartbeat = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                thread::park_timeout(Duration::from_secs(5));
                if thread_stop.load(Ordering::Acquire) {
                    break;
                }
                let mut state = thread_shared
                    .lock()
                    .expect("full-map progress lock poisoned");
                state.event = "heartbeat".to_string();
                state.elapsed_seconds = started.elapsed().as_secs_f64();
                state.observed_process_rss_bytes = observed_process_rss_bytes();
                if let Err(error) = atomic_json(&thread_status_path, &*state) {
                    eprintln!("full-map heartbeat status write failed: {error}");
                }
                println!(
                    "{}",
                    serde_json::to_string(&*state).expect("serialize full-map heartbeat")
                );
            }
        });
        Ok(Self {
            shared,
            stop,
            heartbeat: Some(heartbeat),
            status_path,
        })
    }

    pub(crate) fn observe(&self, event: &MissingMapProgress) -> io::Result<()> {
        let mut state = self.shared.lock().expect("full-map progress lock poisoned");
        *state = event.clone();
        state.state = "running".to_string();
        atomic_json(&self.status_path, &*state)?;
        println!(
            "{}",
            serde_json::to_string(&*state).expect("serialize full-map progress")
        );
        Ok(())
    }

    pub(crate) fn finish(mut self, summary: &MissingMapSummary) -> io::Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.heartbeat.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
        let terminal = serde_json::json!({
            "schema_version": format!("{SCHEMA_VERSION}-progress"),
            "event": "worker_complete",
            "state": "succeeded",
            "status_path": self.status_path.display().to_string(),
            "summary": summary
        });
        atomic_json(&self.status_path, &terminal)?;
        println!(
            "{}",
            serde_json::to_string(&terminal).expect("serialize full-map terminal status")
        );
        Ok(())
    }

    pub(crate) fn fail(mut self, error: &io::Error) -> io::Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.heartbeat.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
        let current = self
            .shared
            .lock()
            .expect("full-map progress lock poisoned")
            .clone();
        let terminal = serde_json::json!({
            "schema_version": format!("{SCHEMA_VERSION}-progress"),
            "event": "worker_failed",
            "state": "failed",
            "status_path": self.status_path.display().to_string(),
            "last_progress": current,
            "error": error.to_string()
        });
        atomic_json(&self.status_path, &terminal)?;
        println!(
            "{}",
            serde_json::to_string(&terminal).expect("serialize full-map failure status")
        );
        Ok(())
    }
}

impl Drop for MissingMapProgressReporter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.heartbeat.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MissingMapSummary {
    pub schema_version: String,
    pub role: String,
    pub expected_jobs: usize,
    pub completed_jobs: usize,
    pub expected_source_target_pairs: usize,
    pub completed_source_target_pairs: usize,
    pub completed_columns_enabled: usize,
    pub remaining_jobs: Vec<MissingMapJob>,
    pub maximum_observed_process_rss_bytes: Option<u64>,
    pub every_exact_raising_residual_is_zero: bool,
    pub every_checkpoint_hash_is_bound: bool,
    pub passed: bool,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixture_for(source: &str, copy: usize) -> io::Result<Level12FixtureRef> {
    level12_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == source && fixture.copy == copy)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing level-12 source fixture {source} copy {copy}"),
            )
        })
}

fn first_fixture(source: &str) -> io::Result<Level12FixtureRef> {
    fixture_for(source, 1)
}

fn target_dimension(source: &str, target: &str) -> io::Result<u64> {
    crate::eleven_dimensional_prepotential::spinor_tensor_channels(source)
        .into_iter()
        .find_map(|(label, dimension)| (label == target).then_some(dimension))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source {source} does not couple to target {target}"),
            )
        })
}

fn abstract_path(directory: &Path, target: &str, source: &str) -> PathBuf {
    directory.join(format!("abstract_{target}_from_{source}.json"))
}

fn embedded_path(directory: &Path, job: &MissingMapJob) -> PathBuf {
    directory.join(format!("embedded_{}.json", job.key()))
}

fn observed_process_rss_bytes() -> Option<u64> {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        if let Some(bytes) = status.lines().find_map(|line| {
            line.strip_prefix("VmRSS:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
                .and_then(|kib| kib.checked_mul(1_024))
        }) {
            return Some(bytes);
        }
    }
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1_024))
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        let file = writer.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn valid_abstract(checkpoint: &MissingAbstractMapCheckpoint) -> bool {
    let Ok(fixture) = first_fixture(&checkpoint.source_dynkin_label) else {
        return false;
    };
    let Ok(payload) = serde_json::to_vec(&checkpoint.certificate) else {
        return false;
    };
    checkpoint.schema_version == format!("{SCHEMA_VERSION}-abstract")
        && checkpoint.target_dynkin_label == checkpoint.certificate.target_dynkin_label
        && checkpoint.source_dynkin_label == checkpoint.certificate.source_dynkin_label
        && checkpoint.source_fixture == fixture.artifact
        && checkpoint.source_fixture_sha256 == sha256(fixture.bytes)
        && checkpoint.coefficient_width_bytes == fixture.coefficient_width_bytes
        && checkpoint.certificate_sha256 == sha256(&payload)
        && checkpoint.certificate.kernel_dimension == 1
        && checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
        && checkpoint.certificate.passed
        && checkpoint.passed
}

fn valid_embedded(
    checkpoint: &MissingEmbeddedMapCheckpoint,
    abstract_checkpoint: &MissingAbstractMapCheckpoint,
) -> bool {
    let Ok(fixture) = fixture_for(
        &checkpoint.job.source_dynkin_label,
        checkpoint.job.source_copy,
    ) else {
        return false;
    };
    checkpoint.schema_version == format!("{SCHEMA_VERSION}-embedded")
        && checkpoint.job.target_dynkin_label == abstract_checkpoint.target_dynkin_label
        && checkpoint.job.source_dynkin_label == abstract_checkpoint.source_dynkin_label
        && checkpoint.source_fixture == fixture.artifact
        && checkpoint.source_fixture_sha256 == sha256(fixture.bytes)
        && checkpoint.coefficient_width_bytes == fixture.coefficient_width_bytes
        && checkpoint.abstract_certificate_sha256 == abstract_checkpoint.certificate_sha256
        && checkpoint.coupled_map_sha256.len() == 64
        && checkpoint.target_dimension
            == target_dimension(
                &checkpoint.job.source_dynkin_label,
                &checkpoint.job.target_dynkin_label,
            )
            .unwrap_or_default()
        && checkpoint.certificate.source_dynkin_label == checkpoint.job.source_dynkin_label
        && checkpoint.certificate.source_copy == checkpoint.job.source_copy
        && checkpoint.certificate.target_dynkin_label == checkpoint.job.target_dynkin_label
        && checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
        && checkpoint.certificate.passed
        && checkpoint.passed
}

pub(crate) fn load_verified_checkpoints(
    directory: &Path,
    job: &MissingMapJob,
) -> io::Result<(MissingAbstractMapCheckpoint, MissingEmbeddedMapCheckpoint)> {
    let abstract_checkpoint: MissingAbstractMapCheckpoint =
        serde_json::from_reader(File::open(abstract_path(
            directory,
            &job.target_dynkin_label,
            &job.source_dynkin_label,
        ))?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let embedded_checkpoint: MissingEmbeddedMapCheckpoint =
        serde_json::from_reader(File::open(embedded_path(directory, job))?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !valid_abstract(&abstract_checkpoint)
        || !valid_embedded(&embedded_checkpoint, &abstract_checkpoint)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid exact full-inventory map checkpoint {}", job.key()),
        ));
    }
    Ok((abstract_checkpoint, embedded_checkpoint))
}

pub(crate) fn construct_abstract_checkpoint(
    directory: &Path,
    target: &str,
    source: &str,
) -> io::Result<MissingAbstractMapCheckpoint> {
    let path = abstract_path(directory, target, source);
    if path.exists() {
        let checkpoint: MissingAbstractMapCheckpoint = serde_json::from_reader(File::open(&path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        return valid_abstract(&checkpoint)
            .then_some(checkpoint)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid abstract map checkpoint",
                )
            });
    }
    let fixture = first_fixture(source)?;
    let started = Instant::now();
    let certificate = crate::eleven_dimensional_level16_couplings::build_second_momentum_abstract(
        target,
        source,
        fixture.copy,
        fixture.coefficient_width_bytes,
        fixture.bytes,
    );
    let certificate_sha256 = sha256(
        &serde_json::to_vec(&certificate)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
    );
    let checkpoint = MissingAbstractMapCheckpoint {
        schema_version: format!("{SCHEMA_VERSION}-abstract"),
        target_dynkin_label: target.to_string(),
        source_dynkin_label: source.to_string(),
        source_fixture: fixture.artifact.to_string(),
        source_fixture_sha256: sha256(fixture.bytes),
        coefficient_width_bytes: fixture.coefficient_width_bytes,
        certificate_sha256,
        elapsed_milliseconds: started.elapsed().as_millis(),
        observed_process_rss_bytes: observed_process_rss_bytes(),
        passed: certificate.passed
            && certificate.kernel_dimension == 1
            && certificate.exact_raising_residual_terms_by_simple_root == [0; 5],
        certificate,
    };
    if !valid_abstract(&checkpoint) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "constructed abstract map failed its proof gate",
        ));
    }
    atomic_json(&path, &checkpoint)?;
    Ok(checkpoint)
}

pub(crate) fn construct_embedded_checkpoint(
    directory: &Path,
    job: &MissingMapJob,
) -> io::Result<MissingEmbeddedMapCheckpoint> {
    let abstract_checkpoint = construct_abstract_checkpoint(
        directory,
        &job.target_dynkin_label,
        &job.source_dynkin_label,
    )?;
    let path = embedded_path(directory, job);
    if path.exists() {
        let checkpoint: MissingEmbeddedMapCheckpoint = serde_json::from_reader(File::open(&path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        return valid_embedded(&checkpoint, &abstract_checkpoint)
            .then_some(checkpoint)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid embedded map checkpoint",
                )
            });
    }
    let fixture = fixture_for(&job.source_dynkin_label, job.source_copy)?;
    let started = Instant::now();
    let (certificate, coupled_map_sha256) =
        crate::eleven_dimensional_level16_couplings::verify_second_momentum_embedding_with_hash(
            &job.target_dynkin_label,
            &abstract_checkpoint.certificate,
            fixture.copy,
            fixture.artifact,
            fixture.coefficient_width_bytes,
            fixture.bytes,
        );
    let checkpoint = MissingEmbeddedMapCheckpoint {
        schema_version: format!("{SCHEMA_VERSION}-embedded"),
        job: job.clone(),
        source_fixture: fixture.artifact.to_string(),
        source_fixture_sha256: sha256(fixture.bytes),
        coefficient_width_bytes: fixture.coefficient_width_bytes,
        abstract_certificate_sha256: abstract_checkpoint.certificate_sha256.clone(),
        coupled_map_sha256,
        target_dimension: target_dimension(&job.source_dynkin_label, &job.target_dynkin_label)?,
        elapsed_milliseconds: started.elapsed().as_millis(),
        observed_process_rss_bytes: observed_process_rss_bytes(),
        passed: certificate.passed
            && certificate.exact_raising_residual_terms_by_simple_root == [0; 5],
        certificate,
    };
    if !valid_embedded(&checkpoint, &abstract_checkpoint) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "constructed embedded map failed its proof gate",
        ));
    }
    atomic_json(&path, &checkpoint)?;
    Ok(checkpoint)
}

pub(crate) fn run_jobs<F>(
    directory: &Path,
    selected_job_ordinals: &[usize],
    mut progress: F,
) -> io::Result<MissingMapSummary>
where
    F: FnMut(&MissingMapProgress) -> io::Result<()>,
{
    let jobs = worklist();
    if selected_job_ordinals.is_empty()
        || selected_job_ordinals
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || selected_job_ordinals
            .iter()
            .any(|ordinal| *ordinal >= jobs.len())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "map job ordinals must be a nonempty increasing in-range list",
        ));
    }
    let run_started = Instant::now();
    let mut completed = 0_usize;
    for (selected_job_position, &job_ordinal) in selected_job_ordinals.iter().enumerate() {
        let job = &jobs[job_ordinal];
        let job_started = Instant::now();
        let abstract_checkpoint_path = abstract_path(
            directory,
            &job.target_dynkin_label,
            &job.source_dynkin_label,
        );
        let embedded_checkpoint_path = embedded_path(directory, job);
        progress(&MissingMapProgress {
            schema_version: format!("{SCHEMA_VERSION}-progress"),
            event: "job_start".to_string(),
            state: "running".to_string(),
            phase: "abstract_or_resume".to_string(),
            job_ordinal,
            jobs_total: jobs.len(),
            selected_job_position,
            selected_jobs_total: selected_job_ordinals.len(),
            completed_jobs: completed,
            target_dynkin_label: job.target_dynkin_label.clone(),
            source_dynkin_label: job.source_dynkin_label.clone(),
            source_copy: job.source_copy,
            elapsed_seconds: run_started.elapsed().as_secs_f64(),
            job_elapsed_seconds: 0.0,
            estimated_remaining_seconds: (completed > 0).then(|| {
                run_started.elapsed().as_secs_f64() / completed as f64
                    * (selected_job_ordinals.len() - completed) as f64
            }),
            observed_process_rss_bytes: observed_process_rss_bytes(),
            abstract_checkpoint_path: abstract_checkpoint_path.display().to_string(),
            embedded_checkpoint_path: embedded_checkpoint_path.display().to_string(),
            message: format!("starting exact map job {}", job.key()),
        })?;
        construct_abstract_checkpoint(
            directory,
            &job.target_dynkin_label,
            &job.source_dynkin_label,
        )?;
        progress(&MissingMapProgress {
            schema_version: format!("{SCHEMA_VERSION}-progress"),
            event: "job_phase".to_string(),
            state: "running".to_string(),
            phase: "embedded_map_or_resume".to_string(),
            job_ordinal,
            jobs_total: jobs.len(),
            selected_job_position,
            selected_jobs_total: selected_job_ordinals.len(),
            completed_jobs: completed,
            target_dynkin_label: job.target_dynkin_label.clone(),
            source_dynkin_label: job.source_dynkin_label.clone(),
            source_copy: job.source_copy,
            elapsed_seconds: run_started.elapsed().as_secs_f64(),
            job_elapsed_seconds: job_started.elapsed().as_secs_f64(),
            estimated_remaining_seconds: (completed > 0).then(|| {
                run_started.elapsed().as_secs_f64() / completed as f64
                    * (selected_job_ordinals.len() - completed) as f64
            }),
            observed_process_rss_bytes: observed_process_rss_bytes(),
            abstract_checkpoint_path: abstract_checkpoint_path.display().to_string(),
            embedded_checkpoint_path: embedded_checkpoint_path.display().to_string(),
            message: format!("abstract checkpoint ready for {}", job.key()),
        })?;
        construct_embedded_checkpoint(directory, job)?;
        completed += 1;
        progress(&MissingMapProgress {
            schema_version: format!("{SCHEMA_VERSION}-progress"),
            event: "job_complete".to_string(),
            state: "running".to_string(),
            phase: "durable_checkpoint".to_string(),
            job_ordinal,
            jobs_total: jobs.len(),
            selected_job_position,
            selected_jobs_total: selected_job_ordinals.len(),
            completed_jobs: completed,
            target_dynkin_label: job.target_dynkin_label.clone(),
            source_dynkin_label: job.source_dynkin_label.clone(),
            source_copy: job.source_copy,
            elapsed_seconds: run_started.elapsed().as_secs_f64(),
            job_elapsed_seconds: job_started.elapsed().as_secs_f64(),
            estimated_remaining_seconds: Some(
                run_started.elapsed().as_secs_f64() / completed as f64
                    * (selected_job_ordinals.len() - completed) as f64,
            ),
            observed_process_rss_bytes: observed_process_rss_bytes(),
            abstract_checkpoint_path: abstract_checkpoint_path.display().to_string(),
            embedded_checkpoint_path: embedded_checkpoint_path.display().to_string(),
            message: format!("completed exact map job {}", job.key()),
        })?;
    }
    Ok(summarize(directory))
}

pub(crate) fn summarize(directory: &Path) -> MissingMapSummary {
    let jobs = worklist();
    let completed = jobs
        .iter()
        .filter_map(|job| {
            let abstract_checkpoint: MissingAbstractMapCheckpoint = serde_json::from_reader(
                File::open(abstract_path(
                    directory,
                    &job.target_dynkin_label,
                    &job.source_dynkin_label,
                ))
                .ok()?,
            )
            .ok()?;
            let embedded_checkpoint: MissingEmbeddedMapCheckpoint =
                serde_json::from_reader(File::open(embedded_path(directory, job)).ok()?).ok()?;
            (valid_abstract(&abstract_checkpoint)
                && valid_embedded(&embedded_checkpoint, &abstract_checkpoint))
            .then_some((abstract_checkpoint, embedded_checkpoint))
        })
        .collect::<Vec<_>>();
    let completed_keys = completed
        .iter()
        .map(|(_, checkpoint)| checkpoint.job.clone())
        .collect::<BTreeSet<_>>();
    let remaining_jobs = jobs
        .iter()
        .filter(|job| !completed_keys.contains(*job))
        .cloned()
        .collect::<Vec<_>>();
    let completed_source_target_pairs = completed
        .iter()
        .map(|(_, checkpoint)| {
            (
                checkpoint.job.source_dynkin_label.as_str(),
                checkpoint.job.target_dynkin_label.as_str(),
            )
        })
        .collect::<BTreeSet<_>>()
        .len();
    let maximum_observed_process_rss_bytes = completed
        .iter()
        .flat_map(|(abstract_checkpoint, embedded_checkpoint)| {
            [
                abstract_checkpoint.observed_process_rss_bytes,
                embedded_checkpoint.observed_process_rss_bytes,
            ]
        })
        .flatten()
        .max();
    let every_exact_raising_residual_is_zero =
        completed
            .iter()
            .all(|(abstract_checkpoint, embedded_checkpoint)| {
                abstract_checkpoint
                    .certificate
                    .exact_raising_residual_terms_by_simple_root
                    == [0; 5]
                    && embedded_checkpoint
                        .certificate
                        .exact_raising_residual_terms_by_simple_root
                        == [0; 5]
            });
    let every_checkpoint_hash_is_bound =
        completed
            .iter()
            .all(|(abstract_checkpoint, embedded_checkpoint)| {
                valid_abstract(abstract_checkpoint)
                    && valid_embedded(embedded_checkpoint, abstract_checkpoint)
            });
    MissingMapSummary {
        schema_version: format!("{SCHEMA_VERSION}-summary"),
        role: "durable exact level-12 source-to-intermediate maps for the missing 49 physical second-momentum columns".to_string(),
        expected_jobs: jobs.len(),
        completed_jobs: completed.len(),
        expected_source_target_pairs: jobs
            .iter()
            .map(|job| (&job.source_dynkin_label, &job.target_dynkin_label))
            .collect::<BTreeSet<_>>()
            .len(),
        completed_source_target_pairs,
        completed_columns_enabled: missing_49_column_specs()
            .iter()
            .filter(|column| {
                completed_keys.contains(&MissingMapJob {
                    target_dynkin_label: column.intermediate_dynkin_label.clone(),
                    source_dynkin_label: column.source_dynkin_label.clone(),
                    source_copy: column.source_copy,
                })
            })
            .count(),
        remaining_jobs,
        maximum_observed_process_rss_bytes,
        every_exact_raising_residual_is_zero,
        every_checkpoint_hash_is_bound,
        passed: completed.len() == jobs.len()
            && every_exact_raising_residual_is_zero
            && every_checkpoint_hash_is_bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_map_worklist_enables_exactly_49_columns() {
        let jobs = worklist();
        assert_eq!(jobs.len(), 47);
        assert_eq!(
            jobs.iter()
                .map(|job| (&job.source_dynkin_label, &job.target_dynkin_label))
                .collect::<BTreeSet<_>>()
                .len(),
            22
        );
        let enabled = missing_49_column_specs()
            .iter()
            .filter(|column| {
                jobs.contains(&MissingMapJob {
                    target_dynkin_label: column.intermediate_dynkin_label.clone(),
                    source_dynkin_label: column.source_dynkin_label.clone(),
                    source_copy: column.source_copy,
                })
            })
            .count();
        assert_eq!(enabled, 49);
    }

    #[test]
    fn empty_summary_is_honest_and_resume_safe() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-missing-map-empty-summary-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let summary = summarize(&directory);
        assert_eq!(summary.expected_jobs, 47);
        assert_eq!(summary.completed_jobs, 0);
        assert_eq!(summary.completed_columns_enabled, 0);
        assert_eq!(summary.remaining_jobs.len(), 47);
        assert!(!summary.passed);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn selected_job_ordinals_must_be_canonical() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-missing-map-invalid-selection-{}",
            std::process::id()
        ));
        let result = run_jobs(&directory, &[1, 1], |_| Ok(()));
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn job_lists_are_portable_and_fail_closed() {
        assert_eq!(parse_job_list("0-2,4,6-7").unwrap(), vec![0, 1, 2, 4, 6, 7]);
        assert_eq!(parse_job_list("all").unwrap().len(), 47);
        assert!(parse_job_list("2,1").is_err());
        assert!(parse_job_list("0,0").is_err());
        assert!(parse_job_list("47").is_err());
    }
}
