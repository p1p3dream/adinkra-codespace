//! Production progress reporting for the second-momentum GPU command.
//!
//! A dedicated monitor thread keeps JSONL heartbeats and an atomic observational
//! status snapshot alive while the command is inside long CPU or CUDA phases.
//! The snapshot is explicitly not a resumable computation checkpoint.

use serde::Serialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, Once};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) const PROGRESS_SCHEMA: &str = "adynkra-11d-second-momentum-gpu-progress-v2";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const ROLLING_RATE_WINDOW: Duration = Duration::from_secs(60);
const MAX_ROLLING_RATE_SAMPLES: usize = 128;
const OPTIONAL_COUNTER_MISSING: u64 = u64::MAX;
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static TERMINATION_SIGNAL: AtomicI32 = AtomicI32::new(0);
static INSTALL_SIGNAL_HANDLERS: Once = Once::new();

#[derive(Clone, Debug)]
pub(crate) struct ProgressConfig {
    pub command: String,
    pub tranche: String,
    pub local_ordinal: usize,
    pub global_ordinal: usize,
    pub tranche_columns_total: usize,
    pub prime: u32,
    pub device: i32,
    pub cpu_parity_terms: usize,
    pub output_directory: PathBuf,
    pub binary_output_path: PathBuf,
    pub report_output_path: PathBuf,
    pub status_snapshot_path: PathBuf,
    pub group: Option<GroupProgressConfig>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct GroupProgressConfig {
    pub job_id: String,
    pub group_id: String,
    pub active_columns: usize,
    pub ordered_local_ordinals: Vec<usize>,
    pub ordered_global_ordinals: Vec<usize>,
    pub ordered_source_copies: Vec<usize>,
    pub checkpoint_path: PathBuf,
    pub event_log_path: PathBuf,
    pub resumable: bool,
}

/// Absolute source-visitor counters. Callers should update this once per batch,
/// not once per term. Absolute values make retries and repeated callbacks safe.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SourceVisitorProgress {
    pub word: Option<u64>,
    pub root: Option<u64>,
    pub raw_terms_emitted: u64,
    pub batches_flushed: u64,
    pub current_batch_terms: u64,
    pub current_batch_bytes: u64,
    pub hard_memory_cap_bytes: u64,
    pub eta_sample_count: u64,
}

/// Absolute GPU batch counters and cumulative timing supplied by the caller.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuBatchProgress {
    pub batches_completed: u64,
    pub last_batch_ms: f64,
    pub total_batch_ms: f64,
    pub last_upload_ms: f64,
    pub total_upload_ms: f64,
    pub last_sort_ms: f64,
    pub total_sort_ms: f64,
    pub last_reduce_ms: f64,
    pub total_reduce_ms: f64,
    pub last_contract_ms: f64,
    pub total_contract_ms: f64,
    pub last_download_ms: f64,
    pub total_download_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct GroupLiveProgress {
    pub group_id: Option<String>,
    pub words_completed: usize,
    pub words_total: usize,
    pub global_batch_ordinal: u64,
    pub raw_terms_per_column: Vec<u64>,
    pub last_union_key_count: usize,
    pub cumulative_union_keys: u64,
    pub keys_by_present_lane_count: Vec<u64>,
    pub host_capacity_bytes: u64,
    pub aggregate_host_cap_bytes: u64,
    pub device_resident_bytes: u64,
    pub device_high_water_bytes: u64,
    pub aggregate_device_cap_bytes: u64,
    pub checkpoint_generation: u64,
    pub checkpoint_sha256: Option<String>,
    pub checkpoint_written_unix_ms: Option<u128>,
}

#[derive(Debug)]
struct LiveMetrics {
    word: AtomicU64,
    root: AtomicU64,
    raw_terms_emitted: AtomicU64,
    batches_flushed: AtomicU64,
    current_batch_terms: AtomicU64,
    current_batch_bytes: AtomicU64,
    hard_memory_cap_bytes: AtomicU64,
    eta_sample_count: AtomicU64,
    gpu_batches_completed: AtomicU64,
    last_gpu_batch_micros: AtomicU64,
    total_gpu_batch_micros: AtomicU64,
    last_gpu_upload_micros: AtomicU64,
    total_gpu_upload_micros: AtomicU64,
    last_gpu_sort_micros: AtomicU64,
    total_gpu_sort_micros: AtomicU64,
    last_gpu_reduce_micros: AtomicU64,
    total_gpu_reduce_micros: AtomicU64,
    last_gpu_contract_micros: AtomicU64,
    total_gpu_contract_micros: AtomicU64,
    last_gpu_download_micros: AtomicU64,
    total_gpu_download_micros: AtomicU64,
    rolling_rate: Mutex<RollingRate>,
    group: Mutex<GroupLiveProgress>,
}

#[derive(Debug, Default)]
struct RollingRate {
    samples: VecDeque<(Instant, u64)>,
}

#[derive(Clone, Debug)]
pub(crate) struct LiveProgress {
    metrics: Arc<LiveMetrics>,
}

impl LiveProgress {
    /// Store one coherent absolute source-progress sample. This is `Send + Sync`
    /// and only locks the bounded rolling-rate window.
    pub(crate) fn update_source(&self, progress: SourceVisitorProgress) {
        let metrics = &self.metrics;
        if let Some(word) = progress.word {
            metrics.word.store(word, Ordering::Relaxed);
        }
        if let Some(root) = progress.root {
            metrics.root.store(root, Ordering::Relaxed);
        }
        metrics
            .raw_terms_emitted
            .store(progress.raw_terms_emitted, Ordering::Relaxed);
        metrics
            .batches_flushed
            .store(progress.batches_flushed, Ordering::Relaxed);
        metrics
            .current_batch_terms
            .store(progress.current_batch_terms, Ordering::Relaxed);
        metrics
            .current_batch_bytes
            .store(progress.current_batch_bytes, Ordering::Relaxed);
        metrics
            .hard_memory_cap_bytes
            .store(progress.hard_memory_cap_bytes, Ordering::Relaxed);
        metrics
            .eta_sample_count
            .store(progress.eta_sample_count, Ordering::Relaxed);

        let now = Instant::now();
        let mut rate = lock(&metrics.rolling_rate);
        if rate
            .samples
            .back()
            .is_some_and(|(_, previous)| progress.raw_terms_emitted < *previous)
        {
            rate.samples.clear();
        }
        rate.samples.push_back((now, progress.raw_terms_emitted));
        while rate.samples.len() > MAX_ROLLING_RATE_SAMPLES {
            rate.samples.pop_front();
        }
        while rate.samples.len() > 2
            && rate
                .samples
                .front()
                .is_some_and(|(time, _)| now.duration_since(*time) > ROLLING_RATE_WINDOW)
        {
            rate.samples.pop_front();
        }
    }

    pub(crate) fn record_gpu_batch(&self, progress: GpuBatchProgress) {
        self.metrics
            .gpu_batches_completed
            .store(progress.batches_completed, Ordering::Relaxed);
        self.metrics.last_gpu_batch_micros.store(
            milliseconds_to_micros(progress.last_batch_ms),
            Ordering::Relaxed,
        );
        self.metrics.total_gpu_batch_micros.store(
            milliseconds_to_micros(progress.total_batch_ms),
            Ordering::Relaxed,
        );
        for (metric, milliseconds) in [
            (
                &self.metrics.last_gpu_upload_micros,
                progress.last_upload_ms,
            ),
            (
                &self.metrics.total_gpu_upload_micros,
                progress.total_upload_ms,
            ),
            (&self.metrics.last_gpu_sort_micros, progress.last_sort_ms),
            (&self.metrics.total_gpu_sort_micros, progress.total_sort_ms),
            (
                &self.metrics.last_gpu_reduce_micros,
                progress.last_reduce_ms,
            ),
            (
                &self.metrics.total_gpu_reduce_micros,
                progress.total_reduce_ms,
            ),
            (
                &self.metrics.last_gpu_contract_micros,
                progress.last_contract_ms,
            ),
            (
                &self.metrics.total_gpu_contract_micros,
                progress.total_contract_ms,
            ),
            (
                &self.metrics.last_gpu_download_micros,
                progress.last_download_ms,
            ),
            (
                &self.metrics.total_gpu_download_micros,
                progress.total_download_ms,
            ),
        ] {
            metric.store(milliseconds_to_micros(milliseconds), Ordering::Relaxed);
        }
    }

    pub(crate) fn update_group(&self, mut progress: GroupLiveProgress) {
        let mut current = lock(&self.metrics.group);
        if progress.checkpoint_sha256.is_none() {
            progress
                .checkpoint_sha256
                .clone_from(&current.checkpoint_sha256);
        }
        if progress.checkpoint_written_unix_ms.is_none() {
            progress.checkpoint_written_unix_ms = current.checkpoint_written_unix_ms;
        }
        *current = progress;
    }
}

impl Default for LiveMetrics {
    fn default() -> Self {
        Self {
            word: AtomicU64::new(OPTIONAL_COUNTER_MISSING),
            root: AtomicU64::new(OPTIONAL_COUNTER_MISSING),
            raw_terms_emitted: AtomicU64::new(0),
            batches_flushed: AtomicU64::new(0),
            current_batch_terms: AtomicU64::new(0),
            current_batch_bytes: AtomicU64::new(0),
            hard_memory_cap_bytes: AtomicU64::new(0),
            eta_sample_count: AtomicU64::new(0),
            gpu_batches_completed: AtomicU64::new(0),
            last_gpu_batch_micros: AtomicU64::new(0),
            total_gpu_batch_micros: AtomicU64::new(0),
            last_gpu_upload_micros: AtomicU64::new(0),
            total_gpu_upload_micros: AtomicU64::new(0),
            last_gpu_sort_micros: AtomicU64::new(0),
            total_gpu_sort_micros: AtomicU64::new(0),
            last_gpu_reduce_micros: AtomicU64::new(0),
            total_gpu_reduce_micros: AtomicU64::new(0),
            last_gpu_contract_micros: AtomicU64::new(0),
            total_gpu_contract_micros: AtomicU64::new(0),
            last_gpu_download_micros: AtomicU64::new(0),
            total_gpu_download_micros: AtomicU64::new(0),
            rolling_rate: Mutex::new(RollingRate::default()),
            group: Mutex::new(GroupLiveProgress::default()),
        }
    }
}

#[derive(Clone, Debug)]
struct ProgressState {
    state: &'static str,
    phase: &'static str,
    phase_started: Instant,
    columns_completed: usize,
    primes_completed: usize,
    message: Option<String>,
    error: Option<String>,
    result: Option<Value>,
    termination_signal: Option<i32>,
}

struct Shared {
    config: ProgressConfig,
    hostname: String,
    started: Instant,
    state: Mutex<ProgressState>,
    output_lock: Mutex<()>,
    status_lock: Mutex<()>,
    live_metrics: Arc<LiveMetrics>,
}

pub(crate) struct ProgressReporter {
    shared: Arc<Shared>,
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
    terminal: bool,
}

impl ProgressReporter {
    pub(crate) fn start(config: ProgressConfig) -> io::Result<Self> {
        install_termination_handlers();
        TERMINATION_SIGNAL.store(0, Ordering::SeqCst);
        fs::create_dir_all(&config.output_directory)?;
        fs::create_dir_all(status_parent(&config.status_snapshot_path)?)?;
        let now = Instant::now();
        let shared = Arc::new(Shared {
            config,
            hostname: machine_hostname(),
            started: now,
            state: Mutex::new(ProgressState {
                state: "running",
                phase: "startup",
                phase_started: now,
                columns_completed: 0,
                primes_completed: 0,
                message: Some("GPU column command initialized".to_owned()),
                error: None,
                result: None,
                termination_signal: None,
            }),
            output_lock: Mutex::new(()),
            status_lock: Mutex::new(()),
            live_metrics: Arc::new(LiveMetrics::default()),
        });
        emit_and_snapshot(&shared, "run_start")?;
        let (stop, receiver) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let worker = match thread::Builder::new()
            .name("second-momentum-gpu-monitor".to_owned())
            .spawn(move || {
                loop {
                    match receiver.recv_timeout(HEARTBEAT_INTERVAL) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                    let signal = TERMINATION_SIGNAL.load(Ordering::SeqCst);
                    if signal != 0 {
                        set_terminal_state(
                            &worker_shared,
                            "terminated",
                            Some(format!("received termination signal {signal}")),
                            None,
                            Some(signal),
                            None,
                        );
                        let _ = emit_and_snapshot(&worker_shared, "terminal");
                        std::process::exit(128 + signal);
                    }
                    if let Err(error) = emit_and_snapshot(&worker_shared, "heartbeat") {
                        emit_fallback_error("heartbeat_status_snapshot_error", &error.to_string());
                    }
                }
            }) {
            Ok(worker) => worker,
            Err(error) => {
                set_terminal_state(
                    &shared,
                    "failed",
                    Some("failed to start GPU progress monitor".to_owned()),
                    Some(error.to_string()),
                    None,
                    None,
                );
                let _ = emit_and_snapshot(&shared, "terminal");
                return Err(error);
            }
        };
        Ok(Self {
            shared,
            stop: Some(stop),
            worker: Some(worker),
            terminal: false,
        })
    }

    pub(crate) fn live_progress(&self) -> LiveProgress {
        LiveProgress {
            metrics: Arc::clone(&self.shared.live_metrics),
        }
    }

    pub(crate) fn phase_start(&self, phase: &'static str) -> io::Result<()> {
        {
            let mut state = lock(&self.shared.state);
            state.phase = phase;
            state.phase_started = Instant::now();
            state.message = Some(format!("phase {phase} started"));
        }
        emit_and_snapshot(&self.shared, "phase_start")
    }

    pub(crate) fn phase_end(&self, message: impl Into<String>) -> io::Result<()> {
        {
            let mut state = lock(&self.shared.state);
            state.message = Some(message.into());
        }
        emit_and_snapshot(&self.shared, "phase_end")
    }

    pub(crate) fn observed_termination_signal(&self) -> Option<i32> {
        let signal = TERMINATION_SIGNAL.load(Ordering::SeqCst);
        (signal != 0).then_some(signal)
    }

    pub(crate) fn finish_success(mut self, result: Value) -> io::Result<()> {
        self.stop_worker();
        set_terminal_state(
            &self.shared,
            "succeeded",
            Some("GPU column completed and artifacts were published".to_owned()),
            None,
            None,
            Some(result),
        );
        self.terminal = true;
        emit_and_snapshot(&self.shared, "terminal")
    }

    pub(crate) fn finish_failure(mut self, error: impl Into<String>) -> io::Result<()> {
        self.stop_worker();
        set_terminal_state(
            &self.shared,
            "failed",
            Some("GPU column command failed".to_owned()),
            Some(error.into()),
            None,
            None,
        );
        self.terminal = true;
        emit_and_snapshot(&self.shared, "terminal")
    }

    pub(crate) fn finish_terminated(mut self, signal: i32) -> io::Result<()> {
        self.stop_worker();
        set_terminal_state(
            &self.shared,
            "terminated",
            Some(format!("received termination signal {signal}")),
            None,
            Some(signal),
            None,
        );
        self.terminal = true;
        emit_and_snapshot(&self.shared, "terminal")
    }

    fn stop_worker(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        self.stop_worker();
        if self.terminal {
            return;
        }
        set_terminal_state(
            &self.shared,
            "failed",
            Some("progress reporter dropped before a terminal result".to_owned()),
            Some("command ended without recording success, failure, or termination".to_owned()),
            None,
            None,
        );
        let _ = emit_and_snapshot(&self.shared, "terminal");
    }
}

fn set_terminal_state(
    shared: &Shared,
    terminal_state: &'static str,
    message: Option<String>,
    error: Option<String>,
    termination_signal: Option<i32>,
    result: Option<Value>,
) {
    let mut state = lock(&shared.state);
    state.state = terminal_state;
    state.phase = "terminal";
    state.phase_started = Instant::now();
    state.columns_completed = usize::from(terminal_state == "succeeded");
    state.primes_completed = usize::from(terminal_state == "succeeded");
    state.message = message;
    state.error = error;
    state.termination_signal = termination_signal;
    state.result = result;
}

fn emit_and_snapshot(shared: &Shared, event: &'static str) -> io::Result<()> {
    let value = event_value(shared, event);
    let mut line = serde_json::to_vec(&value).map_err(io::Error::other)?;
    line.push(b'\n');
    {
        let _guard = lock(&shared.output_lock);
        let stdout = io::stdout();
        let mut output = stdout.lock();
        output.write_all(&line)?;
        output.flush()?;
        if let Some(group) = &shared.config.group {
            if let Some(parent) = group.event_log_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut log = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&group.event_log_path)?;
            log.write_all(&line)?;
            log.flush()?;
        }
    }
    let _guard = lock(&shared.status_lock);
    write_json_atomic(&shared.config.status_snapshot_path, &value)
}

fn event_value(shared: &Shared, event: &'static str) -> Value {
    let state = lock(&shared.state).clone();
    let elapsed = shared.started.elapsed().as_secs_f64();
    let phase_elapsed = state.phase_started.elapsed().as_secs_f64();
    let columns_per_second = state.columns_completed as f64 / elapsed.max(f64::EPSILON);
    let primes_per_second = state.primes_completed as f64 / elapsed.max(f64::EPSILON);
    let live = live_snapshot(&shared.live_metrics);
    let group_live = lock(&shared.live_metrics.group).clone();
    let batches_per_second = live.gpu_batches_completed as f64 / elapsed.max(f64::EPSILON);
    let source_terms = state
        .result
        .as_ref()
        .and_then(|result| result.get("source_terms"))
        .and_then(Value::as_u64);
    let expanded_contributions = state
        .result
        .as_ref()
        .and_then(|result| result.get("expanded_contributions"))
        .and_then(Value::as_u64);
    let observed_source_terms = (live.raw_terms_emitted != 0)
        .then_some(live.raw_terms_emitted)
        .or(source_terms);
    json!({
        "schema_version": PROGRESS_SCHEMA,
        "event": event,
        "state": state.state,
        "phase": state.phase,
        "timestamp_unix_ms": unix_milliseconds(),
        "pid": std::process::id(),
        "hostname": shared.hostname,
        "command": shared.config.command,
        "tranche": shared.config.tranche,
        "local_column_ordinal": shared.config.local_ordinal,
        "global_column_ordinal": shared.config.global_ordinal,
        "tranche_columns_total": shared.config.tranche_columns_total,
        "prime": shared.config.prime,
        "device": shared.config.device,
        "cpu_parity_terms": shared.config.cpu_parity_terms,
        "group": &shared.config.group,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL.as_secs(),
        "elapsed_seconds": elapsed,
        "phase_elapsed_seconds": phase_elapsed,
        "progress": {
            "columns_completed": state.columns_completed,
            "columns_total": 1,
            "primes_completed": state.primes_completed,
            "primes_total": 1,
            "prime_index": 0,
            "gpu_batches_completed": live.gpu_batches_completed,
            "gpu_batches_total": Value::Null,
            "batches_flushed": live.batches_flushed
        },
        "streaming": {
            "word": live.word,
            "root": live.root,
            "raw_terms_emitted": live.raw_terms_emitted,
            "batches_flushed": live.batches_flushed,
            "current_batch_terms": live.current_batch_terms,
            "current_batch_bytes": live.current_batch_bytes,
            "hard_memory_cap_bytes": live.hard_memory_cap_bytes,
            "memory_cap_utilization": live.memory_cap_utilization,
            "memory_cap_exceeded": live.memory_cap_exceeded,
            "eta_sample_count": live.eta_sample_count
        },
        "gpu_batches": {
            "completed": live.gpu_batches_completed,
            "last_batch_milliseconds": live.last_gpu_batch_ms,
            "total_batch_milliseconds": live.total_gpu_batch_ms,
            "average_batch_milliseconds": live.average_gpu_batch_ms,
            "last_stage_milliseconds": {
                "upload": live.last_gpu_upload_ms,
                "sort": live.last_gpu_sort_ms,
                "reduce": live.last_gpu_reduce_ms,
                "contract": live.last_gpu_contract_ms,
                "download": live.last_gpu_download_ms
            },
            "total_stage_milliseconds": {
                "upload": live.total_gpu_upload_ms,
                "sort": live.total_gpu_sort_ms,
                "reduce": live.total_gpu_reduce_ms,
                "contract": live.total_gpu_contract_ms,
                "download": live.total_gpu_download_ms
            }
        },
        "group_progress": group_live,
        "throughput": {
            "columns_per_second": columns_per_second,
            "primes_per_second": primes_per_second,
            "batches_per_second": batches_per_second,
            "source_terms_per_second": observed_source_terms.map(|count| count as f64 / elapsed.max(f64::EPSILON)),
            "rolling_raw_terms_per_second": live.rolling_raw_terms_per_second,
            "batches_flushed_per_second": live.batches_flushed as f64 / elapsed.max(f64::EPSILON),
            "expanded_contributions_per_second": expanded_contributions.map(|count| count as f64 / elapsed.max(f64::EPSILON))
        },
        "resources": {
            "process": process_memory_metrics(),
            "gpu": gpu_metrics(shared.config.device)
        },
        "paths": {
            "output_directory": shared.config.output_directory.display().to_string(),
            "binary_output_path": shared.config.binary_output_path.display().to_string(),
            "report_output_path": shared.config.report_output_path.display().to_string(),
            "status_snapshot_path": shared.config.status_snapshot_path.display().to_string(),
            "checkpoint_path": shared.config.group.as_ref().map(|group| group.checkpoint_path.display().to_string()),
            "event_log_path": shared.config.group.as_ref().map(|group| group.event_log_path.display().to_string())
        },
        "status_snapshot": {
            "path": shared.config.status_snapshot_path.display().to_string(),
            "resumable": shared.config.group.as_ref().is_some_and(|group| group.resumable),
            "semantics": if shared.config.group.as_ref().is_some_and(|group| group.resumable) {
                "observational snapshot plus separately durable word-boundary checkpoint"
            } else {
                "atomic observational status only; it cannot resume computation"
            }
        },
        "message": state.message,
        "error": state.error,
        "termination_signal": state.termination_signal,
        "result": state.result
    })
}

#[derive(Debug)]
struct LiveSnapshot {
    word: Option<u64>,
    root: Option<u64>,
    raw_terms_emitted: u64,
    batches_flushed: u64,
    current_batch_terms: u64,
    current_batch_bytes: u64,
    hard_memory_cap_bytes: u64,
    memory_cap_utilization: Option<f64>,
    memory_cap_exceeded: bool,
    eta_sample_count: u64,
    gpu_batches_completed: u64,
    last_gpu_batch_ms: f64,
    total_gpu_batch_ms: f64,
    average_gpu_batch_ms: Option<f64>,
    last_gpu_upload_ms: f64,
    total_gpu_upload_ms: f64,
    last_gpu_sort_ms: f64,
    total_gpu_sort_ms: f64,
    last_gpu_reduce_ms: f64,
    total_gpu_reduce_ms: f64,
    last_gpu_contract_ms: f64,
    total_gpu_contract_ms: f64,
    last_gpu_download_ms: f64,
    total_gpu_download_ms: f64,
    rolling_raw_terms_per_second: Option<f64>,
}

fn live_snapshot(metrics: &LiveMetrics) -> LiveSnapshot {
    let value_or_none = |value: u64| (value != OPTIONAL_COUNTER_MISSING).then_some(value);
    let hard_memory_cap_bytes = metrics.hard_memory_cap_bytes.load(Ordering::Relaxed);
    let current_batch_bytes = metrics.current_batch_bytes.load(Ordering::Relaxed);
    let gpu_batches_completed = metrics.gpu_batches_completed.load(Ordering::Relaxed);
    let last_gpu_batch_ms =
        micros_to_milliseconds(metrics.last_gpu_batch_micros.load(Ordering::Relaxed));
    let total_gpu_batch_ms =
        micros_to_milliseconds(metrics.total_gpu_batch_micros.load(Ordering::Relaxed));
    LiveSnapshot {
        word: value_or_none(metrics.word.load(Ordering::Relaxed)),
        root: value_or_none(metrics.root.load(Ordering::Relaxed)),
        raw_terms_emitted: metrics.raw_terms_emitted.load(Ordering::Relaxed),
        batches_flushed: metrics.batches_flushed.load(Ordering::Relaxed),
        current_batch_terms: metrics.current_batch_terms.load(Ordering::Relaxed),
        current_batch_bytes,
        hard_memory_cap_bytes,
        memory_cap_utilization: (hard_memory_cap_bytes != 0)
            .then_some(current_batch_bytes as f64 / hard_memory_cap_bytes as f64),
        memory_cap_exceeded: hard_memory_cap_bytes != 0
            && current_batch_bytes > hard_memory_cap_bytes,
        eta_sample_count: metrics.eta_sample_count.load(Ordering::Relaxed),
        gpu_batches_completed,
        last_gpu_batch_ms,
        total_gpu_batch_ms,
        average_gpu_batch_ms: (gpu_batches_completed != 0)
            .then_some(total_gpu_batch_ms / gpu_batches_completed as f64),
        last_gpu_upload_ms: micros_to_milliseconds(
            metrics.last_gpu_upload_micros.load(Ordering::Relaxed),
        ),
        total_gpu_upload_ms: micros_to_milliseconds(
            metrics.total_gpu_upload_micros.load(Ordering::Relaxed),
        ),
        last_gpu_sort_ms: micros_to_milliseconds(
            metrics.last_gpu_sort_micros.load(Ordering::Relaxed),
        ),
        total_gpu_sort_ms: micros_to_milliseconds(
            metrics.total_gpu_sort_micros.load(Ordering::Relaxed),
        ),
        last_gpu_reduce_ms: micros_to_milliseconds(
            metrics.last_gpu_reduce_micros.load(Ordering::Relaxed),
        ),
        total_gpu_reduce_ms: micros_to_milliseconds(
            metrics.total_gpu_reduce_micros.load(Ordering::Relaxed),
        ),
        last_gpu_contract_ms: micros_to_milliseconds(
            metrics.last_gpu_contract_micros.load(Ordering::Relaxed),
        ),
        total_gpu_contract_ms: micros_to_milliseconds(
            metrics.total_gpu_contract_micros.load(Ordering::Relaxed),
        ),
        last_gpu_download_ms: micros_to_milliseconds(
            metrics.last_gpu_download_micros.load(Ordering::Relaxed),
        ),
        total_gpu_download_ms: micros_to_milliseconds(
            metrics.total_gpu_download_micros.load(Ordering::Relaxed),
        ),
        rolling_raw_terms_per_second: rolling_terms_per_second(&metrics.rolling_rate),
    }
}

fn rolling_terms_per_second(rate: &Mutex<RollingRate>) -> Option<f64> {
    let rate = lock(rate);
    let (first_time, first_count) = rate.samples.front()?;
    let (last_time, last_count) = rate.samples.back()?;
    let seconds = last_time.duration_since(*first_time).as_secs_f64();
    (seconds > 0.0 && last_count >= first_count)
        .then_some((*last_count - *first_count) as f64 / seconds)
}

fn milliseconds_to_micros(milliseconds: f64) -> u64 {
    if !milliseconds.is_finite() || milliseconds <= 0.0 {
        0
    } else {
        (milliseconds * 1_000.0).round().min(u64::MAX as f64) as u64
    }
}

fn micros_to_milliseconds(microseconds: u64) -> f64 {
    microseconds as f64 / 1_000.0
}

fn process_memory_metrics() -> Value {
    match fs::read_to_string("/proc/self/status") {
        Ok(status) => {
            let (rss, peak) = parse_proc_status_memory(&status);
            json!({
                "available": rss.is_some() || peak.is_some(),
                "rss_bytes": rss,
                "peak_rss_bytes": peak,
                "source": "/proc/self/status",
                "reason": Value::Null
            })
        }
        Err(proc_error) => match current_rss_from_ps() {
            Ok(rss) => json!({
                "available": true,
                "rss_bytes": rss,
                "peak_rss_bytes": Value::Null,
                "source": "ps rss",
                "reason": "peak RSS is unavailable on this platform"
            }),
            Err(ps_error) => json!({
                "available": false,
                "rss_bytes": Value::Null,
                "peak_rss_bytes": Value::Null,
                "source": Value::Null,
                "reason": format!("/proc unavailable ({proc_error}); ps unavailable ({ps_error})")
            }),
        },
    }
}

fn parse_proc_status_memory(status: &str) -> (Option<u64>, Option<u64>) {
    let mut rss = None;
    let mut peak = None;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            rss = parse_kib(value);
        } else if let Some(value) = line.strip_prefix("VmHWM:") {
            peak = parse_kib(value);
        }
    }
    (rss, peak)
}

fn parse_kib(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

fn current_rss_from_ps() -> Result<u64, String> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!("ps exited with {}", output.status));
    }
    let kib = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .map_err(|error| error.to_string())?;
    kib.checked_mul(1024)
        .ok_or_else(|| "RSS byte count overflow".to_owned())
}

fn gpu_metrics(device: i32) -> Value {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            &device.to_string(),
        ])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            return json!({
                "available": false,
                "reason": format!("nvidia-smi unavailable: {error}")
            });
        }
    };
    if !output.status.success() {
        return json!({
            "available": false,
            "reason": format!(
                "nvidia-smi exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let fields = text.trim().split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 7 {
        return json!({
            "available": false,
            "reason": format!("unexpected nvidia-smi metric row: {}", text.trim())
        });
    }
    let parsed = (
        fields[1].parse::<u32>(),
        fields[2].parse::<u32>(),
        fields[3].parse::<u64>(),
        fields[4].parse::<u64>(),
        fields[5].parse::<u32>(),
        fields[6].parse::<f64>(),
    );
    match parsed {
        (Ok(gpu), Ok(memory), Ok(used), Ok(free), Ok(temperature), Ok(power)) => json!({
            "available": true,
            "device_name": fields[0],
            "gpu_utilization_percent": gpu,
            "memory_utilization_percent": memory,
            "memory_used_bytes": used * 1024 * 1024,
            "memory_free_bytes": free * 1024 * 1024,
            "temperature_celsius": temperature,
            "power_watts": power,
            "source": "nvidia-smi"
        }),
        _ => json!({
            "available": false,
            "reason": format!("failed to parse nvidia-smi metric row: {}", text.trim())
        }),
    }
}

fn write_json_atomic(path: &Path, value: &Value) -> io::Result<()> {
    let parent = status_parent(path)?;
    fs::create_dir_all(parent)?;
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "status path has no file name")
    })?;
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{}.{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        unix_milliseconds(),
        sequence
    ));
    let mut bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.flush()?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        sync_directory(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

fn status_parent(path: &Path) -> io::Result<&Path> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "status path has no parent"))?;
    Ok(if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    })
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
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
        .unwrap_or_else(|| "unknown".to_owned())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReconciliationResult {
    pub reconciled: bool,
    pub state: String,
}

/// Finalize a still-running observational snapshot after a supervisor has
/// waited for the owning child. This covers uncatchable SIGKILL and OOM kills.
/// A terminal snapshot is preserved byte for byte. Group snapshots may point
/// at a separate resumable checkpoint; reconciliation never edits it.
///
/// A shell supervisor should `wait` for the child, then invoke the matching
/// status-reconcile CLI with the child's PID and observed exit or signal.
pub(crate) fn reconcile_status_snapshot(
    path: &Path,
    child_pid: u32,
    exit_code: Option<i32>,
    signal: Option<i32>,
) -> io::Result<ReconciliationResult> {
    if exit_code.is_some() && signal.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "supervisor observation cannot contain both exit code and signal",
        ));
    }
    let bytes = fs::read(path)?;
    let mut status: Value = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    if status.get("schema_version").and_then(Value::as_str) != Some(PROGRESS_SCHEMA) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "status snapshot schema does not match the GPU progress schema",
        ));
    }
    if status
        .get("status_snapshot")
        .and_then(|value| value.get("resumable"))
        .and_then(Value::as_bool)
        .is_none()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file is not a GPU status snapshot",
        ));
    }
    let recorded_pid = status.get("pid").and_then(Value::as_u64);
    if recorded_pid != Some(u64::from(child_pid)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "status snapshot PID {:?} does not match supervised child {child_pid}",
                recorded_pid
            ),
        ));
    }
    let previous_state = status
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "status has no state"))?;
    if previous_state != "running" {
        return Ok(ReconciliationResult {
            reconciled: false,
            state: previous_state.to_owned(),
        });
    }

    let terminal_state = if signal.is_some() {
        "terminated"
    } else {
        "failed"
    };
    let observation = match (exit_code, signal) {
        (_, Some(number)) => format!("supervisor observed child termination by signal {number}"),
        (Some(code), _) => {
            format!("supervisor observed child exit code {code} before a terminal snapshot")
        }
        _ => "supervisor observed child exit before a terminal snapshot".to_owned(),
    };
    let object = status.as_object_mut().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "status snapshot is not an object",
        )
    })?;
    object.insert("event".to_owned(), Value::String("terminal".to_owned()));
    object.insert("state".to_owned(), Value::String(terminal_state.to_owned()));
    object.insert("phase".to_owned(), Value::String("terminal".to_owned()));
    object.insert("timestamp_unix_ms".to_owned(), json!(unix_milliseconds()));
    object.insert("message".to_owned(), Value::String(observation.clone()));
    object.insert("error".to_owned(), Value::String(observation));
    object.insert("termination_signal".to_owned(), json!(signal));
    object.insert(
        "supervisor_reconciliation".to_owned(),
        json!({
            "reconciled": true,
            "observed_child_pid": child_pid,
            "exit_code": exit_code,
            "signal": signal,
            "timestamp_unix_ms": unix_milliseconds()
        }),
    );
    if let Some(snapshot) = object
        .get_mut("status_snapshot")
        .and_then(Value::as_object_mut)
    {
        snapshot.insert("reconciled_by_supervisor".to_owned(), Value::Bool(true));
    }
    write_json_atomic(path, &status)?;
    Ok(ReconciliationResult {
        reconciled: true,
        state: terminal_state.to_owned(),
    })
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn emit_fallback_error(event: &str, error: &str) {
    let value = json!({
        "schema_version": PROGRESS_SCHEMA,
        "event": "terminal",
        "error_kind": event,
        "state": "failed",
        "phase": "terminal",
        "timestamp_unix_ms": unix_milliseconds(),
        "pid": std::process::id(),
        "error": error
    });
    let line = serde_json::to_string(&value).unwrap_or_else(|_| {
        format!(
            "{{\"schema_version\":\"{PROGRESS_SCHEMA}\",\"event\":\"serialization_failure\",\"state\":\"failed\"}}"
        )
    });
    let _ = writeln!(io::stdout().lock(), "{line}");
}

#[cfg(unix)]
fn install_termination_handlers() {
    INSTALL_SIGNAL_HANDLERS.call_once(|| unsafe {
        for signal_number in [1, 2, 15] {
            signal(
                signal_number,
                capture_termination_signal as *const () as usize,
            );
        }
    });
}

#[cfg(not(unix))]
fn install_termination_handlers() {
    INSTALL_SIGNAL_HANDLERS.call_once(|| {});
}

#[cfg(unix)]
extern "C" fn capture_termination_signal(signal_number: i32) {
    TERMINATION_SIGNAL.store(signal_number, Ordering::SeqCst);
}

#[cfg(unix)]
unsafe extern "C" {
    fn signal(signal_number: i32, handler: usize) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_updates_retain_the_last_durable_checkpoint_identity() {
        let live = LiveProgress {
            metrics: Arc::new(LiveMetrics::default()),
        };
        live.update_group(GroupLiveProgress {
            checkpoint_generation: 3,
            checkpoint_sha256: Some("a".repeat(64)),
            checkpoint_written_unix_ms: Some(42),
            ..GroupLiveProgress::default()
        });
        live.update_group(GroupLiveProgress {
            checkpoint_generation: 3,
            global_batch_ordinal: 9,
            ..GroupLiveProgress::default()
        });
        let observed = lock(&live.metrics.group).clone();
        assert_eq!(observed.checkpoint_generation, 3);
        assert_eq!(observed.global_batch_ordinal, 9);
        assert_eq!(observed.checkpoint_sha256, Some("a".repeat(64)));
        assert_eq!(observed.checkpoint_written_unix_ms, Some(42));
    }
    use std::io::Read;

    #[test]
    fn parses_linux_current_and_peak_rss() {
        let status = "Name:\ttest\nVmHWM:\t  8192 kB\nVmRSS:\t4096 kB\n";
        assert_eq!(
            parse_proc_status_memory(status),
            (Some(4 * 1024 * 1024), Some(8 * 1024 * 1024))
        );
    }

    #[test]
    fn atomic_status_write_leaves_only_complete_json() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("status.json");
        let value = json!({"state": "running", "columns_completed": 0});
        write_json_atomic(&path, &value).unwrap();
        let mut bytes = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), value);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn reporter_records_a_clean_terminal_failure() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-reporter-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let status_snapshot_path = directory.join("column.status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "test-command".to_owned(),
            tranche: "20001".to_owned(),
            local_ordinal: 0,
            global_ordinal: 53,
            tranche_columns_total: 9,
            prime: 1_073_741_783,
            device: 0,
            cpu_parity_terms: 128,
            output_directory: directory.clone(),
            binary_output_path: directory.join("column.bin"),
            report_output_path: directory.join("column.json"),
            status_snapshot_path: status_snapshot_path.clone(),
            group: None,
        })
        .unwrap();
        reporter.phase_start("column_execution").unwrap();
        reporter.phase_end("test failure").unwrap();
        reporter.finish_failure("injected failure").unwrap();

        let status: Value =
            serde_json::from_slice(&fs::read(status_snapshot_path).unwrap()).unwrap();
        assert_eq!(status["event"], "terminal");
        assert_eq!(status["state"], "failed");
        assert_eq!(status["phase"], "terminal");
        assert_eq!(status["error"], "injected failure");
        assert_eq!(status["progress"]["columns_completed"], 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn reporter_records_success_counters_and_work_throughput() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-success-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let status_snapshot_path = directory.join("column.status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "test-command".to_owned(),
            tranche: "30001".to_owned(),
            local_ordinal: 0,
            global_ordinal: 62,
            tranche_columns_total: 15,
            prime: 1_073_741_783,
            device: 0,
            cpu_parity_terms: 128,
            output_directory: directory.clone(),
            binary_output_path: directory.join("column.bin"),
            report_output_path: directory.join("column.json"),
            status_snapshot_path: status_snapshot_path.clone(),
            group: None,
        })
        .unwrap();
        reporter.phase_start("column_execution").unwrap();
        reporter.phase_end("test success").unwrap();
        reporter
            .finish_success(json!({
                "source_terms": 256,
                "expanded_contributions": 4096
            }))
            .unwrap();

        let status: Value =
            serde_json::from_slice(&fs::read(status_snapshot_path).unwrap()).unwrap();
        assert_eq!(status["event"], "terminal");
        assert_eq!(status["state"], "succeeded");
        assert_eq!(status["progress"]["columns_completed"], 1);
        assert_eq!(status["progress"]["primes_completed"], 1);
        assert_eq!(status["progress"]["gpu_batches_completed"], 0);
        assert_eq!(status["status_snapshot"]["resumable"], false);
        assert!(status["paths"].get("progress_checkpoint_path").is_none());
        assert!(
            status["throughput"]["expanded_contributions_per_second"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn reporter_records_a_clean_terminated_state() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-terminated-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let status_snapshot_path = directory.join("column.status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "test-command".to_owned(),
            tranche: "20001".to_owned(),
            local_ordinal: 0,
            global_ordinal: 53,
            tranche_columns_total: 9,
            prime: 1_073_741_783,
            device: 0,
            cpu_parity_terms: 128,
            output_directory: directory.clone(),
            binary_output_path: directory.join("column.bin"),
            report_output_path: directory.join("column.json"),
            status_snapshot_path: status_snapshot_path.clone(),
            group: None,
        })
        .unwrap();
        reporter.phase_start("column_execution").unwrap();
        reporter.finish_terminated(15).unwrap();

        let status: Value =
            serde_json::from_slice(&fs::read(status_snapshot_path).unwrap()).unwrap();
        assert_eq!(status["event"], "terminal");
        assert_eq!(status["state"], "terminated");
        assert_eq!(status["termination_signal"], 15);
        assert_eq!(status["progress"]["columns_completed"], 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn background_monitor_heartbeats_during_blocked_work() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-heartbeat-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let status_snapshot_path = directory.join("column.status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "test-command".to_owned(),
            tranche: "30001".to_owned(),
            local_ordinal: 0,
            global_ordinal: 62,
            tranche_columns_total: 15,
            prime: 1_073_741_783,
            device: 0,
            cpu_parity_terms: 128,
            output_directory: directory.clone(),
            binary_output_path: directory.join("column.bin"),
            report_output_path: directory.join("column.json"),
            status_snapshot_path: status_snapshot_path.clone(),
            group: None,
        })
        .unwrap();
        reporter.phase_start("blocked_cpu_work").unwrap();
        thread::sleep(HEARTBEAT_INTERVAL + Duration::from_millis(750));

        let status: Value =
            serde_json::from_slice(&fs::read(&status_snapshot_path).unwrap()).unwrap();
        assert_eq!(status["event"], "heartbeat");
        assert_eq!(status["state"], "running");
        assert_eq!(status["phase"], "blocked_cpu_work");
        assert!(status["elapsed_seconds"].as_f64().unwrap() >= 5.0);
        reporter.finish_failure("test complete").unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn live_progress_is_send_sync_and_reports_streaming_and_gpu_batches() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LiveProgress>();

        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-live-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let status_snapshot_path = directory.join("column.status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "test-command".to_owned(),
            tranche: "30001".to_owned(),
            local_ordinal: 2,
            global_ordinal: 64,
            tranche_columns_total: 15,
            prime: 1_073_741_783,
            device: 0,
            cpu_parity_terms: 128,
            output_directory: directory.clone(),
            binary_output_path: directory.join("column.bin"),
            report_output_path: directory.join("column.json"),
            status_snapshot_path: status_snapshot_path.clone(),
            group: None,
        })
        .unwrap();
        let live = reporter.live_progress();
        live.update_source(SourceVisitorProgress {
            word: Some(7),
            root: Some(11),
            raw_terms_emitted: 1_000,
            batches_flushed: 2,
            current_batch_terms: 100,
            current_batch_bytes: 4_096,
            hard_memory_cap_bytes: 8_192,
            eta_sample_count: 1,
        });
        thread::sleep(Duration::from_millis(10));
        live.update_source(SourceVisitorProgress {
            word: None,
            root: None,
            raw_terms_emitted: 2_000,
            batches_flushed: 3,
            current_batch_terms: 150,
            current_batch_bytes: 9_000,
            hard_memory_cap_bytes: 8_192,
            eta_sample_count: 2,
        });
        live.record_gpu_batch(GpuBatchProgress {
            batches_completed: 3,
            last_batch_ms: 1.25,
            total_batch_ms: 4.5,
            last_upload_ms: 0.1,
            total_upload_ms: 0.3,
            last_sort_ms: 0.2,
            total_sort_ms: 0.6,
            last_reduce_ms: 0.3,
            total_reduce_ms: 0.9,
            last_contract_ms: 0.4,
            total_contract_ms: 1.2,
            last_download_ms: 0.05,
            total_download_ms: 0.15,
        });
        reporter.phase_end("sample live state").unwrap();

        let status: Value =
            serde_json::from_slice(&fs::read(&status_snapshot_path).unwrap()).unwrap();
        assert_eq!(status["streaming"]["word"], 7);
        assert_eq!(status["streaming"]["root"], 11);
        assert_eq!(status["streaming"]["raw_terms_emitted"], 2_000);
        assert_eq!(status["streaming"]["batches_flushed"], 3);
        assert_eq!(status["streaming"]["current_batch_terms"], 150);
        assert_eq!(status["streaming"]["current_batch_bytes"], 9_000);
        assert_eq!(status["streaming"]["hard_memory_cap_bytes"], 8_192);
        assert_eq!(status["streaming"]["memory_cap_exceeded"], true);
        assert_eq!(status["streaming"]["eta_sample_count"], 2);
        assert_eq!(status["gpu_batches"]["completed"], 3);
        assert_eq!(status["gpu_batches"]["last_batch_milliseconds"], 1.25);
        assert_eq!(status["gpu_batches"]["total_batch_milliseconds"], 4.5);
        assert_eq!(
            status["gpu_batches"]["last_stage_milliseconds"]["contract"],
            0.4
        );
        assert_eq!(
            status["gpu_batches"]["total_stage_milliseconds"]["contract"],
            1.2
        );
        assert!(
            status["throughput"]["rolling_raw_terms_per_second"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        reporter.finish_failure("test complete").unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn supervisor_reconciles_running_snapshot_after_uncatchable_signal() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-reconcile-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("column.status.json");
        write_json_atomic(
            &path,
            &json!({
                "schema_version": PROGRESS_SCHEMA,
                "event": "heartbeat",
                "state": "running",
                "phase": "source_streaming",
                "pid": std::process::id(),
                "unknown_field": {"preserved": true},
                "status_snapshot": {
                    "path": path.display().to_string(),
                    "resumable": false
                }
            }),
        )
        .unwrap();
        let result = reconcile_status_snapshot(&path, std::process::id(), None, Some(9)).unwrap();
        assert_eq!(
            result,
            ReconciliationResult {
                reconciled: true,
                state: "terminated".to_owned()
            }
        );
        let status: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(status["event"], "terminal");
        assert_eq!(status["state"], "terminated");
        assert_eq!(status["termination_signal"], 9);
        assert_eq!(status["status_snapshot"]["resumable"], false);
        assert_eq!(status["status_snapshot"]["reconciled_by_supervisor"], true);
        assert_eq!(status["unknown_field"]["preserved"], true);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn supervisor_preserves_existing_terminal_snapshot_byte_for_byte() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-progress-reconcile-terminal-test-{}-{}",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("column.status.json");
        let bytes = format!(
            "{{\"schema_version\":\"{PROGRESS_SCHEMA}\",\"state\":\"succeeded\",\"pid\":{},\"status_snapshot\":{{\"resumable\":false}}}}\n",
            std::process::id()
        )
        .into_bytes();
        fs::write(&path, &bytes).unwrap();
        let result = reconcile_status_snapshot(&path, std::process::id(), Some(0), None).unwrap();
        assert!(!result.reconciled);
        assert_eq!(result.state, "succeeded");
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(directory).unwrap();
    }
}
