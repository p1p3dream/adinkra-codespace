//! Crash-safe publication of exact level-12 kernel fixtures and checkpoints.
//!
//! All candidate certificates and files are validated under one exclusive
//! advisory lock before any final kernel is replaced. Kernel files are synced
//! and atomically renamed first. The checkpoint is synced, renamed last, and
//! followed by a parent-directory sync.
//!
//! Unlike the Python publisher, an existing checkpoint with a missing or
//! corrupt kernel may be repaired by supplying a staged same-label candidate
//! whose output certificate is identical. The existing system metadata and
//! checkpoint bytes are preserved. This intentional difference permits safe,
//! deterministic recovery after a binary is lost without weakening conflict
//! checks.

use crate::PRIME;
use fs4::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHECKPOINT_SCHEMA_VERSION: &str =
    "adynkra-11d-level12-second-momentum-kernel-generation-v1";
pub const CHECKPOINT_ROLE: &str =
    "exact level-12 source fixtures for p^2 D^12 second-momentum operators";
pub const CHECKPOINT_METHOD: &str = "deterministic sparse echelon over 2^31-1, rational reconstruction, and full integer residual verification";
pub const EXPECTED_SYSTEMS: usize = 19;
pub const EXPECTED_KERNEL_COPIES: usize = 41;

const SYSTEM_ORDER: [&str; EXPECTED_SYSTEMS] = [
    "00000", "00010", "00100", "01002", "01100", "02000", "10002", "11002", "11010", "11100",
    "12000", "20002", "20010", "20100", "30002", "30010", "30100", "31000", "40000",
];
const SYSTEM_MULTIPLICITIES: [usize; EXPECTED_SYSTEMS] =
    [1, 1, 1, 4, 2, 2, 2, 5, 4, 3, 1, 2, 3, 2, 3, 2, 1, 1, 1];
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KernelOutputMetadata {
    pub copy: usize,
    pub path: String,
    pub sha256: String,
    pub bytes: usize,
    pub nonzero_coefficients: usize,
    pub maximum_absolute_coefficient: u64,
    #[serde(default, flatten)]
    pub extra_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SystemSeconds {
    pub matrix: f64,
    pub echelon: f64,
    pub reconstruct_and_integer_verify: f64,
    pub total: f64,
    #[serde(default, flatten)]
    pub extra_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PublishedSystem {
    pub dynkin_label: String,
    pub exterior_degree: usize,
    pub source_columns: usize,
    pub raising_rows: usize,
    pub nonzero_entries: usize,
    pub prime: u32,
    pub exact_modular_rank: usize,
    pub exact_nullity: usize,
    pub free_columns: Vec<usize>,
    pub maximum_pivot_width: usize,
    pub coefficient_width_bytes: usize,
    pub outputs: Vec<KernelOutputMetadata>,
    pub seconds: SystemSeconds,
    pub passed: bool,
    #[serde(default, flatten)]
    pub extra_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PublicationArtifact {
    pub schema_version: String,
    pub role: String,
    pub method: String,
    pub systems: Vec<PublishedSystem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_systems: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_kernel_copies: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_systems: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_kernel_copies: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inventory_complete: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    #[serde(default, flatten)]
    pub extra_fields: BTreeMap<String, Value>,
}

impl PublicationArtifact {
    pub fn empty() -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
            role: CHECKPOINT_ROLE.to_owned(),
            method: CHECKPOINT_METHOD.to_owned(),
            systems: Vec::new(),
            completed_systems: None,
            completed_kernel_copies: None,
            expected_systems: None,
            expected_kernel_copies: None,
            inventory_complete: None,
            passed: None,
            extra_fields: BTreeMap::new(),
        }
    }

    fn finalize_summary(&mut self) {
        self.completed_systems = Some(self.systems.len());
        self.completed_kernel_copies =
            Some(self.systems.iter().map(|system| system.exact_nullity).sum());
        self.expected_systems = Some(EXPECTED_SYSTEMS);
        self.expected_kernel_copies = Some(EXPECTED_KERNEL_COPIES);
        self.inventory_complete = Some(self.systems.len() == EXPECTED_SYSTEMS);
        self.passed = Some(self.systems.iter().all(|system| system.passed));
    }
}

#[derive(Clone, Debug)]
pub struct StagedOutput {
    pub copy: usize,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct StagedSystem {
    pub system: PublishedSystem,
    pub outputs: Vec<StagedOutput>,
}

#[derive(Debug)]
pub enum PublicationError {
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Json(serde_json::Error),
    Validation(String),
    Conflict(String),
    PathEscape(PathBuf),
    InjectedPreCheckpointFailure,
}

impl Display for PublicationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io {
                action,
                path,
                source,
            } => write!(formatter, "failed to {action} {}: {source}", path.display()),
            Self::Json(source) => {
                write!(formatter, "invalid publication checkpoint JSON: {source}")
            }
            Self::Validation(message) => {
                write!(formatter, "publication validation failed: {message}")
            }
            Self::Conflict(message) => write!(formatter, "publication conflict: {message}"),
            Self::PathEscape(path) => write!(
                formatter,
                "publication path escapes the repository root: {}",
                path.display()
            ),
            Self::InjectedPreCheckpointFailure => {
                write!(formatter, "injected failure before checkpoint publication")
            }
        }
    }
}

impl Error for PublicationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for PublicationError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub type PublicationResult<T> = Result<T, PublicationError>;

struct PreparedSystem {
    system: PublishedSystem,
    staged_by_copy: BTreeMap<usize, PathBuf>,
}

struct PublishAction {
    staged: PathBuf,
    final_path: PathBuf,
}

struct StageCleanup {
    paths: Vec<PathBuf>,
}

impl Drop for StageCleanup {
    fn drop(&mut self) {
        for path in &self.paths {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(_) => {}
            }
        }
    }
}

/// Publish staged kernel systems under an exclusive checkpoint lock.
pub fn publish_staged_systems(
    root: impl AsRef<Path>,
    checkpoint_relative: impl AsRef<Path>,
    candidates: Vec<StagedSystem>,
) -> PublicationResult<PublicationArtifact> {
    publish_staged_systems_impl(
        root.as_ref(),
        checkpoint_relative.as_ref(),
        candidates,
        false,
    )
}

fn publish_staged_systems_impl(
    root: &Path,
    checkpoint_relative: &Path,
    candidates: Vec<StagedSystem>,
    inject_pre_checkpoint_failure: bool,
) -> PublicationResult<PublicationArtifact> {
    let root = canonicalize_path(root, "canonicalize repository root")?;
    let checkpoint = contained_relative_path(&root, checkpoint_relative)?;
    validate_existing_ancestor(&root, &checkpoint)?;
    let (prepared, cleanup) = prepare_staged_systems(&root, candidates)?;

    let checkpoint_parent = checkpoint.parent().ok_or_else(|| {
        PublicationError::Validation("checkpoint has no parent directory".to_owned())
    })?;
    create_contained_directory(&root, checkpoint_parent)?;
    let lock_path = sibling_with_suffix(&checkpoint, ".lock")?;
    reject_symlink_if_present(&lock_path, "publication lock")?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|source| PublicationError::Io {
            action: "open publication lock",
            path: lock_path.clone(),
            source,
        })?;
    FileExt::lock(&lock).map_err(|source| PublicationError::Io {
        action: "acquire exclusive publication lock",
        path: lock_path.clone(),
        source,
    })?;

    let result = publish_locked(&root, &checkpoint, &prepared, inject_pre_checkpoint_failure);
    let unlock = FileExt::unlock(&lock).map_err(|source| PublicationError::Io {
        action: "release publication lock",
        path: lock_path,
        source,
    });
    drop(cleanup);
    match (result, unlock) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(artifact), Ok(())) => Ok(artifact),
    }
}

fn publish_locked(
    root: &Path,
    checkpoint: &Path,
    candidates: &[PreparedSystem],
    inject_pre_checkpoint_failure: bool,
) -> PublicationResult<PublicationArtifact> {
    reject_symlink_if_present(checkpoint, "publication checkpoint")?;
    let existing_bytes = match fs::read(checkpoint) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(source) => {
            return Err(PublicationError::Io {
                action: "read publication checkpoint",
                path: checkpoint.to_owned(),
                source,
            });
        }
    };
    let existing = match &existing_bytes {
        Some(bytes) => serde_json::from_slice::<PublicationArtifact>(bytes)?,
        None => PublicationArtifact::empty(),
    };
    validate_artifact_metadata(&existing)?;
    validate_prepared_candidates(candidates)?;

    let existing_by_label: BTreeMap<_, _> = existing
        .systems
        .iter()
        .map(|system| (system.dynkin_label.as_str(), system))
        .collect();
    let candidate_by_label: BTreeMap<_, _> = candidates
        .iter()
        .map(|candidate| (candidate.system.dynkin_label.as_str(), candidate))
        .collect();
    if candidate_by_label.len() != candidates.len() {
        return Err(PublicationError::Conflict(
            "duplicate staged system label".to_owned(),
        ));
    }

    for candidate in candidates {
        if let Some(old) = existing_by_label.get(candidate.system.dynkin_label.as_str())
            && output_identity(old) != output_identity(&candidate.system)
        {
            return Err(PublicationError::Conflict(format!(
                "conflicting exact kernels for {}",
                candidate.system.dynkin_label
            )));
        }
    }

    // Every existing checkpoint output must already be valid or have an
    // identical same-label staged replacement in this transaction.
    for system in &existing.systems {
        let repair = candidate_by_label
            .get(system.dynkin_label.as_str())
            .copied();
        for output in &system.outputs {
            let final_path = contained_relative_path(root, Path::new(&output.path))?;
            validate_existing_ancestor(root, &final_path)?;
            if file_matches(&final_path, output)? {
                continue;
            }
            let Some(repair) = repair else {
                return Err(PublicationError::Validation(format!(
                    "existing kernel is missing or corrupt: {}",
                    final_path.display()
                )));
            };
            let staged = repair.staged_by_copy.get(&output.copy).ok_or_else(|| {
                PublicationError::Validation(format!(
                    "missing staged repair for {} copy {}",
                    system.dynkin_label, output.copy
                ))
            })?;
            validate_staged_file(staged, output)?;
        }
    }

    let mut actions = Vec::new();
    let mut final_paths = BTreeSet::new();
    let mut added_new_system = false;
    for candidate in candidates {
        let is_existing = existing_by_label.contains_key(candidate.system.dynkin_label.as_str());
        added_new_system |= !is_existing;
        for output in &candidate.system.outputs {
            let staged = candidate.staged_by_copy.get(&output.copy).ok_or_else(|| {
                PublicationError::Validation(format!(
                    "missing staged output for {} copy {}",
                    candidate.system.dynkin_label, output.copy
                ))
            })?;
            validate_staged_file(staged, output)?;
            let final_path = contained_relative_path(root, Path::new(&output.path))?;
            validate_existing_ancestor(root, &final_path)?;
            if staged == &final_path {
                return Err(PublicationError::Validation(format!(
                    "staged and final paths are identical: {}",
                    staged.display()
                )));
            }
            if !final_paths.insert(final_path.clone()) {
                return Err(PublicationError::Conflict(format!(
                    "duplicate final kernel path {}",
                    final_path.display()
                )));
            }
            if file_matches(&final_path, output)? {
                continue;
            }
            if !is_existing && path_entry_exists(&final_path)? {
                return Err(PublicationError::Conflict(format!(
                    "conflicting untracked kernel {}",
                    final_path.display()
                )));
            }
            actions.push(PublishAction {
                staged: staged.clone(),
                final_path,
            });
        }
    }

    // Complete preflight succeeded. Sync each staged file before the first
    // final-path mutation.
    for action in &actions {
        sync_file(&action.staged, "sync staged kernel")?;
    }
    for action in &actions {
        let parent = action.final_path.parent().ok_or_else(|| {
            PublicationError::Validation("kernel output has no parent directory".to_owned())
        })?;
        create_contained_directory(root, parent)?;
        fs::rename(&action.staged, &action.final_path).map_err(|source| PublicationError::Io {
            action: "atomically publish staged kernel",
            path: action.final_path.clone(),
            source,
        })?;
        sync_directory(parent)?;
    }

    let mut updated = existing.clone();
    if added_new_system {
        let mut systems: BTreeMap<usize, PublishedSystem> = updated
            .systems
            .into_iter()
            .map(|system| Ok((system_order(&system.dynkin_label)?, system)))
            .collect::<PublicationResult<_>>()?;
        for candidate in candidates {
            if !existing_by_label.contains_key(candidate.system.dynkin_label.as_str()) {
                systems.insert(
                    system_order(&candidate.system.dynkin_label)?,
                    candidate.system.clone(),
                );
            }
        }
        updated.systems = systems.into_values().collect();
        updated.finalize_summary();
    }
    verify_all_published_outputs(root, &updated)?;

    if inject_pre_checkpoint_failure {
        return Err(PublicationError::InjectedPreCheckpointFailure);
    }
    if !added_new_system {
        return Ok(existing);
    }

    write_checkpoint_last(checkpoint, &updated)?;
    Ok(updated)
}

fn prepare_staged_systems(
    root: &Path,
    candidates: Vec<StagedSystem>,
) -> PublicationResult<(Vec<PreparedSystem>, StageCleanup)> {
    let mut prepared = Vec::with_capacity(candidates.len());
    let mut all_staged = BTreeSet::new();
    let mut cleanup = StageCleanup { paths: Vec::new() };
    for candidate in candidates {
        let mut staged_by_copy = BTreeMap::new();
        for output in candidate.outputs {
            let path = contained_existing_file(root, &output.path)?;
            if !all_staged.insert(path.clone()) {
                return Err(PublicationError::Conflict(format!(
                    "duplicate staged kernel path {}",
                    path.display()
                )));
            }
            cleanup.paths.push(path.clone());
            if staged_by_copy.insert(output.copy, path).is_some() {
                return Err(PublicationError::Conflict(format!(
                    "duplicate staged copy {} for {}",
                    output.copy, candidate.system.dynkin_label
                )));
            }
        }
        prepared.push(PreparedSystem {
            system: candidate.system,
            staged_by_copy,
        });
    }
    Ok((prepared, cleanup))
}

fn validate_prepared_candidates(candidates: &[PreparedSystem]) -> PublicationResult<()> {
    for candidate in candidates {
        validate_system(&candidate.system)?;
        if candidate.staged_by_copy.len() != candidate.system.outputs.len() {
            return Err(PublicationError::Validation(format!(
                "incomplete staged output set for {}",
                candidate.system.dynkin_label
            )));
        }
    }
    Ok(())
}

fn validate_artifact_metadata(artifact: &PublicationArtifact) -> PublicationResult<()> {
    if artifact.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(PublicationError::Validation(
            "level-12 kernel checkpoint schema mismatch".to_owned(),
        ));
    }
    if artifact.role != CHECKPOINT_ROLE || artifact.method != CHECKPOINT_METHOD {
        return Err(PublicationError::Validation(
            "level-12 kernel checkpoint role or method mismatch".to_owned(),
        ));
    }
    let mut labels = BTreeSet::new();
    let mut output_paths = BTreeSet::new();
    let mut previous_order = None;
    for system in &artifact.systems {
        validate_system(system)?;
        if !labels.insert(system.dynkin_label.as_str()) {
            return Err(PublicationError::Validation(format!(
                "duplicate checkpoint system {}",
                system.dynkin_label
            )));
        }
        let order = system_order(&system.dynkin_label)?;
        if previous_order.is_some_and(|previous| previous >= order) {
            return Err(PublicationError::Validation(
                "checkpoint systems are not in deterministic inventory order".to_owned(),
            ));
        }
        previous_order = Some(order);
        for output in &system.outputs {
            if !output_paths.insert(output.path.as_str()) {
                return Err(PublicationError::Validation(format!(
                    "duplicate checkpoint kernel path {}",
                    output.path
                )));
            }
        }
    }
    let completed_copies: usize = artifact
        .systems
        .iter()
        .map(|system| system.exact_nullity)
        .sum();
    validate_optional_summary(
        "completed_systems",
        artifact.completed_systems,
        artifact.systems.len(),
    )?;
    validate_optional_summary(
        "completed_kernel_copies",
        artifact.completed_kernel_copies,
        completed_copies,
    )?;
    validate_optional_summary(
        "expected_systems",
        artifact.expected_systems,
        EXPECTED_SYSTEMS,
    )?;
    validate_optional_summary(
        "expected_kernel_copies",
        artifact.expected_kernel_copies,
        EXPECTED_KERNEL_COPIES,
    )?;
    if artifact
        .inventory_complete
        .is_some_and(|value| value != (artifact.systems.len() == EXPECTED_SYSTEMS))
    {
        return Err(PublicationError::Validation(
            "checkpoint inventory_complete summary mismatch".to_owned(),
        ));
    }
    if artifact
        .passed
        .is_some_and(|value| value != artifact.systems.iter().all(|system| system.passed))
    {
        return Err(PublicationError::Validation(
            "checkpoint passed summary mismatch".to_owned(),
        ));
    }
    Ok(())
}

fn validate_optional_summary(
    name: &str,
    actual: Option<usize>,
    expected: usize,
) -> PublicationResult<()> {
    if actual.is_some_and(|value| value != expected) {
        return Err(PublicationError::Validation(format!(
            "checkpoint {name} summary mismatch"
        )));
    }
    Ok(())
}

fn validate_system(system: &PublishedSystem) -> PublicationResult<()> {
    let order = system_order(&system.dynkin_label)?;
    let multiplicity = SYSTEM_MULTIPLICITIES[order];
    if system.exterior_degree != 12 || !system.passed || system.prime != PRIME {
        return Err(PublicationError::Validation(format!(
            "invalid certificate metadata for {}",
            system.dynkin_label
        )));
    }
    if system.exact_modular_rank.checked_add(system.exact_nullity) != Some(system.source_columns) {
        return Err(PublicationError::Validation(format!(
            "rank-nullity mismatch for {}",
            system.dynkin_label
        )));
    }
    if system.exact_nullity != multiplicity || system.outputs.len() != multiplicity {
        return Err(PublicationError::Validation(format!(
            "published multiplicity mismatch for {}",
            system.dynkin_label
        )));
    }
    if !matches!(system.coefficient_width_bytes, 2 | 4) {
        return Err(PublicationError::Validation(format!(
            "unsupported coefficient width for {}",
            system.dynkin_label
        )));
    }
    if system.free_columns.len() != multiplicity
        || system
            .free_columns
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != multiplicity
        || system
            .free_columns
            .iter()
            .any(|&column| column >= system.source_columns)
    {
        return Err(PublicationError::Validation(format!(
            "invalid free-column certificate for {}",
            system.dynkin_label
        )));
    }
    let expected_bytes = system
        .source_columns
        .checked_mul(system.coefficient_width_bytes)
        .ok_or_else(|| {
            PublicationError::Validation(format!(
                "kernel byte count overflows for {}",
                system.dynkin_label
            ))
        })?;
    for (index, output) in system.outputs.iter().enumerate() {
        if output.copy != index + 1 {
            return Err(PublicationError::Validation(format!(
                "copy order mismatch for {}",
                system.dynkin_label
            )));
        }
        let suffix = if multiplicity == 1 {
            String::new()
        } else {
            format!("_{}", output.copy)
        };
        let expected_path = format!(
            "data/eleven_dimensional_spinor_bridge/level12_{}_highest_weight_kernel{}.i{}le",
            system.dynkin_label,
            suffix,
            system.coefficient_width_bytes * 8
        );
        if output.path != expected_path
            || output.bytes != expected_bytes
            || !valid_sha256(&output.sha256)
            || output.nonzero_coefficients > system.source_columns
        {
            return Err(PublicationError::Validation(format!(
                "invalid output metadata for {} copy {}",
                system.dynkin_label, output.copy
            )));
        }
    }
    let seconds = [
        system.seconds.matrix,
        system.seconds.echelon,
        system.seconds.reconstruct_and_integer_verify,
        system.seconds.total,
    ];
    if seconds
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PublicationError::Validation(format!(
            "invalid timing metadata for {}",
            system.dynkin_label
        )));
    }
    Ok(())
}

fn output_identity(system: &PublishedSystem) -> Vec<(usize, &str, &str, usize)> {
    system
        .outputs
        .iter()
        .map(|output| {
            (
                output.copy,
                output.path.as_str(),
                output.sha256.as_str(),
                output.bytes,
            )
        })
        .collect()
}

fn verify_all_published_outputs(
    root: &Path,
    artifact: &PublicationArtifact,
) -> PublicationResult<()> {
    for system in &artifact.systems {
        for output in &system.outputs {
            let path = contained_relative_path(root, Path::new(&output.path))?;
            if !file_matches(&path, output)? {
                return Err(PublicationError::Validation(format!(
                    "published kernel is missing or corrupt: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn validate_staged_file(path: &Path, output: &KernelOutputMetadata) -> PublicationResult<()> {
    if !file_matches(path, output)? {
        return Err(PublicationError::Validation(format!(
            "staged kernel does not match certificate: {}",
            path.display()
        )));
    }
    Ok(())
}

fn file_matches(path: &Path, output: &KernelOutputMetadata) -> PublicationResult<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(source) => {
            return Err(PublicationError::Io {
                action: "inspect kernel file",
                path: path.to_owned(),
                source,
            });
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Ok(false);
    }
    if metadata.len() != output.bytes as u64 {
        return Ok(false);
    }
    Ok(sha256_file(path)? == output.sha256)
}

fn sha256_file(path: &Path) -> PublicationResult<String> {
    let mut file = File::open(path).map_err(|source| PublicationError::Io {
        action: "open kernel for hashing",
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| PublicationError::Io {
                action: "read kernel for hashing",
                path: path.to_owned(),
                source,
            })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn system_order(label: &str) -> PublicationResult<usize> {
    SYSTEM_ORDER
        .iter()
        .position(|candidate| *candidate == label)
        .ok_or_else(|| PublicationError::Validation(format!("unexpected system label {label}")))
}

fn contained_relative_path(root: &Path, relative: &Path) -> PublicationResult<PathBuf> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PublicationError::PathEscape(relative.to_owned()));
    }
    Ok(root.join(relative))
}

fn contained_existing_file(root: &Path, path: &Path) -> PublicationResult<PathBuf> {
    let joined = if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    };
    let metadata = fs::symlink_metadata(&joined).map_err(|source| PublicationError::Io {
        action: "inspect staged kernel",
        path: joined.clone(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PublicationError::Validation(format!(
            "staged kernel is not a regular file: {}",
            joined.display()
        )));
    }
    let canonical = canonicalize_path(&joined, "canonicalize staged kernel")?;
    if !canonical.starts_with(root) {
        return Err(PublicationError::PathEscape(joined));
    }
    Ok(canonical)
}

fn validate_existing_ancestor(root: &Path, path: &Path) -> PublicationResult<()> {
    let mut ancestor = path
        .parent()
        .ok_or_else(|| PublicationError::PathEscape(path.to_owned()))?;
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .ok_or_else(|| PublicationError::PathEscape(path.to_owned()))?;
    }
    let canonical = canonicalize_path(ancestor, "canonicalize path ancestor")?;
    if !canonical.starts_with(root) {
        return Err(PublicationError::PathEscape(path.to_owned()));
    }
    Ok(())
}

fn create_contained_directory(root: &Path, directory: &Path) -> PublicationResult<()> {
    validate_existing_ancestor(root, directory)?;
    let relative = directory
        .strip_prefix(root)
        .map_err(|_| PublicationError::PathEscape(directory.to_owned()))?;
    let mut current = root.to_owned();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(PublicationError::PathEscape(directory.to_owned()));
        };
        let parent = current.clone();
        current.push(name);
        match fs::create_dir(&current) {
            Ok(()) => sync_directory(&parent)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(PublicationError::Io {
                    action: "create publication directory",
                    path: current,
                    source,
                });
            }
        }
    }
    let canonical = canonicalize_path(directory, "canonicalize publication directory")?;
    if !canonical.starts_with(root) {
        return Err(PublicationError::PathEscape(directory.to_owned()));
    }
    Ok(())
}

fn reject_symlink_if_present(path: &Path, role: &str) -> PublicationResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(PublicationError::Validation(
            format!("{role} must not be a symbolic link: {}", path.display()),
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PublicationError::Io {
            action: "inspect publication path",
            path: path.to_owned(),
            source,
        }),
    }
}

fn path_entry_exists(path: &Path) -> PublicationResult<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(PublicationError::Io {
            action: "inspect publication path",
            path: path.to_owned(),
            source,
        }),
    }
}

fn canonicalize_path(path: &Path, action: &'static str) -> PublicationResult<PathBuf> {
    fs::canonicalize(path).map_err(|source| PublicationError::Io {
        action,
        path: path.to_owned(),
        source,
    })
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> PublicationResult<PathBuf> {
    let name = path.file_name().ok_or_else(|| {
        PublicationError::Validation(format!("path has no file name: {}", path.display()))
    })?;
    let mut sibling = name.to_os_string();
    sibling.push(suffix);
    Ok(path.with_file_name(sibling))
}

fn sync_file(path: &Path, action: &'static str) -> PublicationResult<()> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| PublicationError::Io {
            action,
            path: path.to_owned(),
            source,
        })
}

fn sync_directory(path: &Path) -> PublicationResult<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| PublicationError::Io {
            action: "sync parent directory",
            path: path.to_owned(),
            source,
        })
}

fn write_checkpoint_last(
    checkpoint: &Path,
    artifact: &PublicationArtifact,
) -> PublicationResult<()> {
    let parent = checkpoint.parent().ok_or_else(|| {
        PublicationError::Validation("checkpoint has no parent directory".to_owned())
    })?;
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let suffix = format!(".{}.{}.tmp", std::process::id(), sequence);
    let temporary = sibling_with_suffix(checkpoint, &suffix)?;
    let mut encoded = serde_json::to_vec_pretty(artifact)?;
    encoded.push(b'\n');
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| PublicationError::Io {
                action: "create staged checkpoint",
                path: temporary.clone(),
                source,
            })?;
        file.write_all(&encoded)
            .and_then(|()| file.flush())
            .and_then(|()| file.sync_all())
            .map_err(|source| PublicationError::Io {
                action: "write and sync staged checkpoint",
                path: temporary.clone(),
                source,
            })?;
        fs::rename(&temporary, checkpoint).map_err(|source| PublicationError::Io {
            action: "atomically publish checkpoint",
            path: checkpoint.to_owned(),
            source,
        })?;
        sync_directory(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "adynkra-exact-publish-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn candidate(root: &Path, label: &str, payload: &[u8], tag: &str) -> StagedSystem {
        let multiplicity = SYSTEM_MULTIPLICITIES[system_order(label).unwrap()];
        assert_eq!(multiplicity, 1, "test helper supports nullity-one labels");
        let relative = format!(
            "data/eleven_dimensional_spinor_bridge/level12_{label}_highest_weight_kernel.i16le"
        );
        let final_path = root.join(&relative);
        fs::create_dir_all(final_path.parent().unwrap()).unwrap();
        let staged = final_path.with_file_name(format!(
            "{}.{}.staged",
            final_path.file_name().unwrap().to_string_lossy(),
            tag
        ));
        fs::write(&staged, payload).unwrap();
        StagedSystem {
            system: PublishedSystem {
                dynkin_label: label.to_owned(),
                exterior_degree: 12,
                source_columns: 1,
                raising_rows: 0,
                nonzero_entries: 0,
                prime: PRIME,
                exact_modular_rank: 0,
                exact_nullity: 1,
                free_columns: vec![0],
                maximum_pivot_width: 0,
                coefficient_width_bytes: 2,
                outputs: vec![KernelOutputMetadata {
                    copy: 1,
                    path: relative,
                    sha256: sha256(payload),
                    bytes: payload.len(),
                    nonzero_coefficients: usize::from(payload != [0, 0]),
                    maximum_absolute_coefficient: 1,
                    extra_fields: BTreeMap::new(),
                }],
                seconds: SystemSeconds {
                    matrix: 0.0,
                    echelon: 0.0,
                    reconstruct_and_integer_verify: 0.0,
                    total: 0.0,
                    extra_fields: BTreeMap::new(),
                },
                passed: true,
                extra_fields: BTreeMap::new(),
            },
            outputs: vec![StagedOutput {
                copy: 1,
                path: staged,
            }],
        }
    }

    fn checkpoint() -> PathBuf {
        PathBuf::from("results/checkpoint.json")
    }

    fn final_path(root: &Path, label: &str) -> PathBuf {
        root.join(format!(
            "data/eleven_dimensional_spinor_bridge/level12_{label}_highest_weight_kernel.i16le"
        ))
    }

    #[test]
    fn initial_publish_syncs_binary_and_checkpoint() {
        let root = TestDirectory::new();
        let artifact = publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "initial")],
        )
        .unwrap();
        assert_eq!(fs::read(final_path(&root.path, "00000")).unwrap(), [1, 0]);
        assert_eq!(artifact.completed_systems, Some(1));
        assert_eq!(artifact.completed_kernel_copies, Some(1));
        let saved: PublicationArtifact =
            serde_json::from_slice(&fs::read(root.path.join(checkpoint())).unwrap()).unwrap();
        assert_eq!(saved, artifact);
    }

    #[test]
    fn identical_repeat_is_idempotent_and_preserves_checkpoint_bytes() {
        let root = TestDirectory::new();
        publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "first")],
        )
        .unwrap();
        let checkpoint_path = root.path.join(checkpoint());
        let before_checkpoint = fs::read(&checkpoint_path).unwrap();
        let before_binary = fs::read(final_path(&root.path, "00000")).unwrap();
        publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "repeat")],
        )
        .unwrap();
        assert_eq!(fs::read(checkpoint_path).unwrap(), before_checkpoint);
        assert_eq!(
            fs::read(final_path(&root.path, "00000")).unwrap(),
            before_binary
        );
    }

    #[test]
    fn idempotent_repeat_preserves_unknown_fields_and_checkpoint_bytes() {
        let root = TestDirectory::new();
        let mut extended = publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "first")],
        )
        .unwrap();
        extended
            .extra_fields
            .insert("future_top".to_owned(), serde_json::json!({"revision": 2}));
        let system = &mut extended.systems[0];
        system
            .extra_fields
            .insert("future_system".to_owned(), serde_json::json!([1, 2, 3]));
        system.outputs[0].extra_fields.insert(
            "future_output".to_owned(),
            serde_json::json!({"encoding": "signed"}),
        );
        system.seconds.extra_fields.insert(
            "future_seconds".to_owned(),
            serde_json::json!({"cpu": 0.25}),
        );

        let checkpoint_path = root.path.join(checkpoint());
        let mut extended_bytes = vec![b'\n'];
        extended_bytes.extend(serde_json::to_vec_pretty(&extended).unwrap());
        extended_bytes.extend(b"\n\n");
        fs::write(&checkpoint_path, &extended_bytes).unwrap();

        let returned = publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "repeat-extra")],
        )
        .unwrap();

        assert_eq!(fs::read(checkpoint_path).unwrap(), extended_bytes);
        assert_eq!(returned, extended);
        assert_eq!(returned.extra_fields["future_top"]["revision"], 2);
        assert_eq!(
            returned.systems[0].extra_fields["future_system"],
            serde_json::json!([1, 2, 3])
        );
        assert_eq!(
            returned.systems[0].outputs[0].extra_fields["future_output"]["encoding"],
            "signed"
        );
        assert_eq!(
            returned.systems[0].seconds.extra_fields["future_seconds"]["cpu"],
            0.25
        );
    }

    #[test]
    fn conflicting_same_label_rejects_without_mutating_pinned_state() {
        let root = TestDirectory::new();
        publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "first")],
        )
        .unwrap();
        let checkpoint_path = root.path.join(checkpoint());
        let before_checkpoint = fs::read(&checkpoint_path).unwrap();
        let before_binary = fs::read(final_path(&root.path, "00000")).unwrap();
        let result = publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[2, 0], "conflict")],
        );
        assert!(matches!(result, Err(PublicationError::Conflict(_))));
        assert_eq!(fs::read(checkpoint_path).unwrap(), before_checkpoint);
        assert_eq!(
            fs::read(final_path(&root.path, "00000")).unwrap(),
            before_binary
        );
    }

    #[test]
    fn identical_candidate_repairs_missing_published_binary() {
        let root = TestDirectory::new();
        publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "first")],
        )
        .unwrap();
        let checkpoint_path = root.path.join(checkpoint());
        let checkpoint_bytes = fs::read(&checkpoint_path).unwrap();
        fs::remove_file(final_path(&root.path, "00000")).unwrap();
        publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "repair")],
        )
        .unwrap();
        assert_eq!(fs::read(checkpoint_path).unwrap(), checkpoint_bytes);
        assert_eq!(fs::read(final_path(&root.path, "00000")).unwrap(), [1, 0]);
    }

    #[test]
    fn staged_path_escape_is_rejected() {
        let root = TestDirectory::new();
        let outside = root.path.parent().unwrap().join(format!(
            "adynkra-outside-stage-{}",
            TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&outside, [1, 0]).unwrap();
        let mut candidate = candidate(&root.path, "00000", &[1, 0], "inside");
        candidate.outputs[0].path = outside.clone();
        let result = publish_staged_systems(&root.path, checkpoint(), vec![candidate]);
        assert!(matches!(result, Err(PublicationError::PathEscape(_))));
        assert!(!root.path.join(checkpoint()).exists());
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn untracked_final_conflict_is_rejected() {
        let root = TestDirectory::new();
        let candidate = candidate(&root.path, "00000", &[1, 0], "candidate");
        fs::write(final_path(&root.path, "00000"), [9, 0]).unwrap();
        let result = publish_staged_systems(&root.path, checkpoint(), vec![candidate]);
        assert!(matches!(result, Err(PublicationError::Conflict(_))));
        assert_eq!(fs::read(final_path(&root.path, "00000")).unwrap(), [9, 0]);
        assert!(!root.path.join(checkpoint()).exists());
    }

    #[test]
    fn pre_checkpoint_failure_never_creates_a_dangling_checkpoint() {
        let root = TestDirectory::new();
        let result = publish_staged_systems_impl(
            &root.path,
            &checkpoint(),
            vec![candidate(&root.path, "00000", &[1, 0], "injected")],
            true,
        );
        assert!(matches!(
            result,
            Err(PublicationError::InjectedPreCheckpointFailure)
        ));
        assert_eq!(fs::read(final_path(&root.path, "00000")).unwrap(), [1, 0]);
        assert!(!root.path.join(checkpoint()).exists());
    }

    #[test]
    fn systems_are_published_in_inventory_order_with_deterministic_counts() {
        let root = TestDirectory::new();
        let artifact = publish_staged_systems(
            &root.path,
            checkpoint(),
            vec![
                candidate(&root.path, "00010", &[2, 0], "second"),
                candidate(&root.path, "00000", &[1, 0], "first"),
            ],
        )
        .unwrap();
        assert_eq!(
            artifact
                .systems
                .iter()
                .map(|system| system.dynkin_label.as_str())
                .collect::<Vec<_>>(),
            ["00000", "00010"]
        );
        assert_eq!(artifact.completed_systems, Some(2));
        assert_eq!(artifact.completed_kernel_copies, Some(2));
        assert_eq!(artifact.expected_systems, Some(EXPECTED_SYSTEMS));
        assert_eq!(
            artifact.expected_kernel_copies,
            Some(EXPECTED_KERNEL_COPIES)
        );
        assert_eq!(artifact.inventory_complete, Some(false));
        assert_eq!(artifact.passed, Some(true));
    }
}
