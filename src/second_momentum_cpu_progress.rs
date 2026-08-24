//! Live, machine-readable progress for exact CPU second-momentum tranche runs.

use serde::Serialize;
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) const PROGRESS_SCHEMA: &str = "adynkra-11d-second-momentum-cpu-progress-v1";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug)]
pub(crate) enum CpuTrancheEvent {
    ColumnStarted {
        local_ordinal: usize,
        global_ordinal: usize,
    },
    ColumnCompleted {
        local_ordinal: usize,
        global_ordinal: usize,
        elapsed_milliseconds: u128,
        gauge_residual_terms: usize,
        projected_terms: u64,
        observed_rss_bytes: Option<u64>,
    },
    Finalizing,
}

#[derive(Clone, Debug)]
pub(crate) struct CpuProgressConfig {
    pub tranche: String,
    pub columns_total: usize,
    pub output_path: PathBuf,
    pub status_path: PathBuf,
}

#[derive(Clone, Debug)]
struct State {
    state: &'static str,
    phase: &'static str,
    phase_started: Instant,
    current_local_ordinal: Option<usize>,
    current_global_ordinal: Option<usize>,
    columns_completed: usize,
    completed_column_seconds: Vec<f64>,
    gauge_residual_terms: u64,
    projected_terms: u64,
    maximum_observed_rss_bytes: Option<u64>,
    message: String,
    error: Option<String>,
    result: Option<Value>,
}

struct Shared {
    config: CpuProgressConfig,
    started: Instant,
    state: Mutex<State>,
    output_lock: Mutex<()>,
    status_lock: Mutex<()>,
}

pub(crate) struct CpuProgressReporter {
    shared: Arc<Shared>,
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
    terminal: bool,
}

impl CpuProgressReporter {
    pub(crate) fn start(config: CpuProgressConfig) -> io::Result<Self> {
        if let Some(parent) = config.status_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(parent) = config.output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let now = Instant::now();
        let shared = Arc::new(Shared {
            config,
            started: now,
            state: Mutex::new(State {
                state: "running",
                phase: "startup",
                phase_started: now,
                current_local_ordinal: None,
                current_global_ordinal: None,
                columns_completed: 0,
                completed_column_seconds: Vec::new(),
                gauge_residual_terms: 0,
                projected_terms: 0,
                maximum_observed_rss_bytes: None,
                message: "CPU tranche command initialized".to_owned(),
                error: None,
                result: None,
            }),
            output_lock: Mutex::new(()),
            status_lock: Mutex::new(()),
        });
        emit_and_snapshot(&shared, "run_start")?;
        let (stop, receiver) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("second-momentum-cpu-monitor".to_owned())
            .spawn(move || {
                loop {
                    match receiver.recv_timeout(HEARTBEAT_INTERVAL) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                        Err(RecvTimeoutError::Timeout) => {
                            let _ = emit_and_snapshot(&worker_shared, "heartbeat");
                        }
                    }
                }
            })?;
        Ok(Self {
            shared,
            stop: Some(stop),
            worker: Some(worker),
            terminal: false,
        })
    }

    pub(crate) fn observe(&self, event: CpuTrancheEvent) -> io::Result<()> {
        {
            let mut state = lock(&self.shared.state);
            match event {
                CpuTrancheEvent::ColumnStarted {
                    local_ordinal,
                    global_ordinal,
                } => {
                    state.phase = "column_execution";
                    state.phase_started = Instant::now();
                    state.current_local_ordinal = Some(local_ordinal);
                    state.current_global_ordinal = Some(global_ordinal);
                    state.message = format!("column {global_ordinal} started");
                }
                CpuTrancheEvent::ColumnCompleted {
                    local_ordinal,
                    global_ordinal,
                    elapsed_milliseconds,
                    gauge_residual_terms,
                    projected_terms,
                    observed_rss_bytes,
                } => {
                    state.columns_completed = local_ordinal + 1;
                    state
                        .completed_column_seconds
                        .push(elapsed_milliseconds as f64 / 1_000.0);
                    state.gauge_residual_terms = state
                        .gauge_residual_terms
                        .saturating_add(gauge_residual_terms as u64);
                    state.projected_terms = state.projected_terms.saturating_add(projected_terms);
                    state.maximum_observed_rss_bytes = state
                        .maximum_observed_rss_bytes
                        .into_iter()
                        .chain(observed_rss_bytes)
                        .max();
                    state.message = format!("column {global_ordinal} completed");
                }
                CpuTrancheEvent::Finalizing => {
                    state.phase = "finalization";
                    state.phase_started = Instant::now();
                    state.current_local_ordinal = None;
                    state.current_global_ordinal = None;
                    state.message =
                        "rank solve, validation, and atomic publication started".to_owned();
                }
            }
        }
        emit_and_snapshot(&self.shared, "progress")
    }

    pub(crate) fn finish_success<T: Serialize>(mut self, result: &T) -> io::Result<()> {
        self.stop_worker();
        {
            let mut state = lock(&self.shared.state);
            state.state = "succeeded";
            state.phase = "terminal";
            state.phase_started = Instant::now();
            state.current_local_ordinal = None;
            state.current_global_ordinal = None;
            state.message = "tranche completed, validated, and published".to_owned();
            state.result = Some(serde_json::to_value(result).map_err(io::Error::other)?);
        }
        self.terminal = true;
        emit_and_snapshot(&self.shared, "terminal")
    }

    pub(crate) fn finish_failure(mut self, error: impl Into<String>) -> io::Result<()> {
        self.stop_worker();
        {
            let mut state = lock(&self.shared.state);
            state.state = "failed";
            state.phase = "terminal";
            state.phase_started = Instant::now();
            state.message = "tranche failed without publishing a final artifact".to_owned();
            state.error = Some(error.into());
        }
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

impl Drop for CpuProgressReporter {
    fn drop(&mut self) {
        self.stop_worker();
        if self.terminal {
            return;
        }
        {
            let mut state = lock(&self.shared.state);
            state.state = "failed";
            state.phase = "terminal";
            state.message = "reporter dropped before a terminal result".to_owned();
            state.error = Some("command exited unexpectedly".to_owned());
        }
        let _ = emit_and_snapshot(&self.shared, "terminal");
    }
}

fn emit_and_snapshot(shared: &Shared, event: &'static str) -> io::Result<()> {
    let value = event_value(shared, event);
    let bytes = serde_json::to_vec(&value).map_err(io::Error::other)?;
    {
        let _guard = lock(&shared.output_lock);
        let stdout = io::stdout();
        let mut output = stdout.lock();
        output.write_all(&bytes)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    let _guard = lock(&shared.status_lock);
    write_json_atomic(&shared.config.status_path, &value)
}

fn event_value(shared: &Shared, event: &'static str) -> Value {
    let state = lock(&shared.state).clone();
    let elapsed = shared.started.elapsed().as_secs_f64();
    let mean_column_seconds = (!state.completed_column_seconds.is_empty()).then(|| {
        state.completed_column_seconds.iter().sum::<f64>()
            / state.completed_column_seconds.len() as f64
    });
    let remaining = shared
        .config
        .columns_total
        .saturating_sub(state.columns_completed);
    let eta_seconds = mean_column_seconds.map(|mean| mean * remaining as f64);
    let process = process_memory_metrics();
    json!({
        "schema_version": PROGRESS_SCHEMA,
        "event": event,
        "state": state.state,
        "phase": state.phase,
        "timestamp_unix_ms": unix_milliseconds(),
        "pid": std::process::id(),
        "command": "adynkra-11d-second-momentum-cpu-fx",
        "tranche": shared.config.tranche,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL.as_secs(),
        "elapsed_seconds": elapsed,
        "phase_elapsed_seconds": state.phase_started.elapsed().as_secs_f64(),
        "progress": {
            "columns_completed": state.columns_completed,
            "columns_total": shared.config.columns_total,
            "current_local_ordinal": state.current_local_ordinal,
            "current_global_ordinal": state.current_global_ordinal,
            "completion_fraction": state.columns_completed as f64 / shared.config.columns_total as f64,
            "mean_completed_column_seconds": mean_column_seconds,
            "estimated_remaining_seconds": eta_seconds,
            "gauge_residual_terms": state.gauge_residual_terms,
            "projected_terms": state.projected_terms
        },
        "resources": {
            "process": process,
            "maximum_observed_column_boundary_rss_bytes": state.maximum_observed_rss_bytes
        },
        "paths": {
            "output_path": shared.config.output_path.display().to_string(),
            "status_path": shared.config.status_path.display().to_string()
        },
        "checkpoint": {
            "resumable": false,
            "semantics": "atomic observational status; final artifact is published atomically only after all columns and rank validation complete"
        },
        "message": state.message,
        "error": state.error,
        "result": state.result
    })
}

fn process_memory_metrics() -> Value {
    match fs::read_to_string("/proc/self/status") {
        Ok(status) => {
            let mut rss = None;
            let mut peak = None;
            for line in status.lines() {
                if let Some(value) = line.strip_prefix("VmRSS:") {
                    rss = parse_kib(value);
                } else if let Some(value) = line.strip_prefix("VmHWM:") {
                    peak = parse_kib(value);
                }
            }
            json!({"available": true, "rss_bytes": rss, "peak_rss_bytes": peak, "source": "/proc/self/status"})
        }
        Err(error) => {
            json!({"available": false, "rss_bytes": Value::Null, "peak_rss_bytes": Value::Null, "reason": error.to_string()})
        }
    }
}

fn parse_kib(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

fn write_json_atomic(path: &Path, value: &Value) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("status"),
        std::process::id()
    ));
    {
        let mut file = File::create(&temporary)?;
        serde_json::to_writer(&mut file, value).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
