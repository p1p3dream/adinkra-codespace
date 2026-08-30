//! Exact level-12 source maps into the `(30001)` second-momentum channel.
//!
//! The eight source irreps below account for fifteen level-12 exterior
//! embeddings. Each source tensored with one spinor contains `(30001)` once.
//! This module constructs that abstract intertwiner, applies it to every exact
//! source copy, checks all five raising residuals over the integers, and pins
//! the resulting component-map hashes in resumable checkpoints.
//!
//! These are source-to-intermediate maps. The separate
//! `(30001) -> Sym^2(V) tensor (10001)` momentum recoupling is not constructed
//! here, so no physical `F A G_p = 0` claim follows from this tranche alone.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SCHEMA_VERSION: &str = "adynkra-11d-second-momentum-30001-maps-v1";
const TARGET_DYNKIN_LABEL: &str = "30001";
const EXPECTED_TARGET_DIMENSION: u64 = 7_040;

#[derive(Clone, Copy, Debug)]
struct SourceFixture {
    dynkin_label: &'static str,
    copy: usize,
    artifact: &'static str,
    bytes: &'static [u8],
}

macro_rules! fixture {
    ($label:literal, $copy:literal, $artifact:literal) => {
        SourceFixture {
            dynkin_label: $label,
            copy: $copy,
            artifact: $artifact,
            bytes: include_bytes!(concat!(
                "../data/eleven_dimensional_spinor_bridge/",
                $artifact
            )),
        }
    };
}

fn source_fixtures() -> Vec<SourceFixture> {
    vec![
        fixture!("40000", 1, "level12_40000_highest_weight_kernel.i16le"),
        fixture!("20100", 1, "level12_20100_highest_weight_kernel_1.i16le"),
        fixture!("20100", 2, "level12_20100_highest_weight_kernel_2.i16le"),
        fixture!("31000", 1, "level12_31000_highest_weight_kernel.i16le"),
        fixture!("20010", 1, "level12_20010_highest_weight_kernel_1.i16le"),
        fixture!("20010", 2, "level12_20010_highest_weight_kernel_2.i16le"),
        fixture!("20010", 3, "level12_20010_highest_weight_kernel_3.i16le"),
        fixture!("20002", 1, "level12_20002_highest_weight_kernel_1.i16le"),
        fixture!("20002", 2, "level12_20002_highest_weight_kernel_2.i16le"),
        fixture!("30100", 1, "level12_30100_highest_weight_kernel.i16le"),
        fixture!("30010", 1, "level12_30010_highest_weight_kernel_1.i16le"),
        fixture!("30010", 2, "level12_30010_highest_weight_kernel_2.i16le"),
        fixture!("30002", 1, "level12_30002_highest_weight_kernel_1.i16le"),
        fixture!("30002", 2, "level12_30002_highest_weight_kernel_2.i16le"),
        fixture!("30002", 3, "level12_30002_highest_weight_kernel_3.i16le"),
    ]
}

fn source_labels() -> Vec<&'static str> {
    let mut labels = source_fixtures()
        .into_iter()
        .map(|fixture| fixture.dynkin_label)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    labels.sort_unstable();
    labels
}

fn first_fixture(label: &str) -> SourceFixture {
    source_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("missing first level-12 source fixture for {label}"))
}

fn fixture_for(label: &str, copy: usize) -> SourceFixture {
    source_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == label && fixture.copy == copy)
        .unwrap_or_else(|| panic!("missing level-12 source fixture {label} copy {copy}"))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct SecondMomentum30001MapJob {
    pub source_dynkin_label: String,
    pub source_copy: usize,
}

impl SecondMomentum30001MapJob {
    pub fn key(&self) -> String {
        format!(
            "30001_from_{}_copy{}",
            self.source_dynkin_label, self.source_copy
        )
    }
}

pub fn worklist() -> Vec<SecondMomentum30001MapJob> {
    source_fixtures()
        .into_iter()
        .map(|fixture| SecondMomentum30001MapJob {
            source_dynkin_label: fixture.dynkin_label.to_string(),
            source_copy: fixture.copy,
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecondMomentum30001AbstractCheckpoint {
    pub schema_version: String,
    pub source_dynkin_label: String,
    pub target_dynkin_label: String,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub certificate_sha256: String,
    pub elapsed_milliseconds: u128,
    pub observed_process_rss_bytes: Option<u64>,
    pub csr_storage_lower_bound_bytes: u64,
    pub certificate: crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecondMomentum30001EmbeddedCheckpoint {
    pub schema_version: String,
    pub job: SecondMomentum30001MapJob,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub abstract_certificate_sha256: String,
    pub coupled_map_sha256: String,
    pub target_dimension: u64,
    pub certified_irrep_image_rank: u64,
    pub rank_derivation: String,
    pub elapsed_milliseconds: u128,
    pub observed_process_rss_bytes: Option<u64>,
    pub certificate: crate::eleven_dimensional_level16_couplings::EmbeddedCouplingCertificate,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecondMomentum30001MutationControl {
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub mutated_primitive_coefficient: usize,
    pub exact_raising_residual_detected_mutation: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecondMomentum30001Report {
    pub schema_version: String,
    pub exterior_degree: usize,
    pub intermediate_dynkin_label: String,
    pub expected_source_irreps: usize,
    pub exact_abstract_intertwiners: usize,
    pub expected_source_copies: usize,
    pub exact_embedded_component_maps: usize,
    pub exact_embedded_maps_by_source: BTreeMap<String, usize>,
    pub certified_intermediate_target_rank: u64,
    pub every_exact_raising_residual_is_zero: bool,
    pub every_source_copy_map_hash_is_distinct: bool,
    pub checkpointed_runtime_milliseconds: u128,
    pub maximum_observed_process_rss_bytes: Option<u64>,
    pub maximum_csr_storage_lower_bound_bytes: u64,
    pub mutation_control: Option<SecondMomentum30001MutationControl>,
    pub remaining_jobs: Vec<SecondMomentum30001MapJob>,
    pub source_to_30001_component_maps_complete: bool,
    pub momentum_recoupling_30001_to_10001_complete: bool,
    pub physical_target_gauge_quotient_complete: bool,
    pub full_physical_f_complete: bool,
    pub full_f_a_g_p_zero_proved: bool,
    pub passed: bool,
    pub boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn abstract_path(directory: &Path, source: &str) -> PathBuf {
    directory.join(format!("abstract_30001_from_{source}.json"))
}

fn embedded_path(directory: &Path, job: &SecondMomentum30001MapJob) -> PathBuf {
    directory.join(format!("embedded_{}.json", job.key()))
}

fn mutation_path(directory: &Path) -> PathBuf {
    directory.join("mutation_control.json")
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    serde_json::to_writer_pretty(BufWriter::new(File::create(&temporary)?), value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    fs::rename(temporary, path)
}

fn observed_process_rss_bytes() -> Option<u64> {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        if let Some(kib) = status.lines().find_map(|line| {
            line.strip_prefix("VmRSS:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
        }) {
            return kib.checked_mul(1_024);
        }
    }
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?
        .checked_mul(1_024)
}

fn csr_storage_lower_bound_bytes(
    certificate: &crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    fixture_bytes: usize,
) -> u64 {
    let entry_bytes = 2 * std::mem::size_of::<usize>() + std::mem::size_of::<i8>();
    u64::try_from(
        fixture_bytes
            + certificate.csr_nonzero_entries * entry_bytes
            + certificate.primitive_domain_coefficients.len() * std::mem::size_of::<i64>(),
    )
    .expect("audited CSR storage lower bound fits u64")
}

fn read_abstract(
    directory: &Path,
    source: &str,
) -> io::Result<SecondMomentum30001AbstractCheckpoint> {
    serde_json::from_reader(File::open(abstract_path(directory, source))?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn abstract_is_valid(checkpoint: &SecondMomentum30001AbstractCheckpoint) -> bool {
    let fixture = first_fixture(&checkpoint.source_dynkin_label);
    let payload = serde_json::to_vec(&checkpoint.certificate).expect("serialize certificate");
    checkpoint.schema_version == format!("{SCHEMA_VERSION}-abstract")
        && checkpoint.target_dynkin_label == TARGET_DYNKIN_LABEL
        && checkpoint.source_fixture == fixture.artifact
        && checkpoint.source_fixture_sha256 == sha256(fixture.bytes)
        && valid_sha256(&checkpoint.certificate_sha256)
        && checkpoint.certificate_sha256 == sha256(&payload)
        && checkpoint.certificate.source_dynkin_label == checkpoint.source_dynkin_label
        && checkpoint.certificate.target_dynkin_label == TARGET_DYNKIN_LABEL
        && checkpoint.certificate.kernel_dimension == 1
        && checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
        && checkpoint.certificate.passed
        && checkpoint.passed
}

pub fn construct_abstract_checkpoint(
    directory: &Path,
    source: &str,
) -> io::Result<SecondMomentum30001AbstractCheckpoint> {
    if abstract_path(directory, source).exists() {
        let checkpoint = read_abstract(directory, source)?;
        if abstract_is_valid(&checkpoint) {
            return Ok(checkpoint);
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid existing second-momentum abstract checkpoint",
        ));
    }
    let fixture = first_fixture(source);
    let started = Instant::now();
    let certificate =
        crate::eleven_dimensional_level16_couplings::build_second_momentum_30001_abstract(
            source,
            fixture.copy,
            2,
            fixture.bytes,
        );
    let elapsed_milliseconds = started.elapsed().as_millis();
    let payload = serde_json::to_vec(&certificate).expect("serialize certificate");
    let checkpoint = SecondMomentum30001AbstractCheckpoint {
        schema_version: format!("{SCHEMA_VERSION}-abstract"),
        source_dynkin_label: source.to_string(),
        target_dynkin_label: TARGET_DYNKIN_LABEL.to_string(),
        source_fixture: fixture.artifact.to_string(),
        source_fixture_sha256: sha256(fixture.bytes),
        certificate_sha256: sha256(&payload),
        elapsed_milliseconds,
        observed_process_rss_bytes: observed_process_rss_bytes(),
        csr_storage_lower_bound_bytes: csr_storage_lower_bound_bytes(
            &certificate,
            fixture.bytes.len(),
        ),
        passed: certificate.passed
            && certificate.kernel_dimension == 1
            && certificate.exact_raising_residual_terms_by_simple_root == [0; 5],
        certificate,
    };
    if !checkpoint.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact level-12 abstract intertwiner failed",
        ));
    }
    atomic_json(&abstract_path(directory, source), &checkpoint)?;
    Ok(checkpoint)
}

fn embedded_is_valid(
    checkpoint: &SecondMomentum30001EmbeddedCheckpoint,
    abstract_checkpoint: &SecondMomentum30001AbstractCheckpoint,
) -> bool {
    let fixture = fixture_for(
        &checkpoint.job.source_dynkin_label,
        checkpoint.job.source_copy,
    );
    checkpoint.schema_version == format!("{SCHEMA_VERSION}-embedded")
        && abstract_checkpoint.source_dynkin_label == checkpoint.job.source_dynkin_label
        && checkpoint.source_fixture == fixture.artifact
        && checkpoint.source_fixture_sha256 == sha256(fixture.bytes)
        && checkpoint.abstract_certificate_sha256 == abstract_checkpoint.certificate_sha256
        && valid_sha256(&checkpoint.coupled_map_sha256)
        && checkpoint.target_dimension == EXPECTED_TARGET_DIMENSION
        && checkpoint.certified_irrep_image_rank == EXPECTED_TARGET_DIMENSION
        && checkpoint.certificate.source_dynkin_label == checkpoint.job.source_dynkin_label
        && checkpoint.certificate.source_copy == checkpoint.job.source_copy
        && checkpoint.certificate.target_dynkin_label == TARGET_DYNKIN_LABEL
        && checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
        && checkpoint.certificate.passed
        && checkpoint.passed
}

pub fn construct_embedded_checkpoint(
    directory: &Path,
    job: &SecondMomentum30001MapJob,
) -> io::Result<SecondMomentum30001EmbeddedCheckpoint> {
    let abstract_checkpoint = construct_abstract_checkpoint(directory, &job.source_dynkin_label)?;
    let path = embedded_path(directory, job);
    if path.exists() {
        let checkpoint: SecondMomentum30001EmbeddedCheckpoint =
            serde_json::from_reader(File::open(&path)?)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if checkpoint.job == *job && embedded_is_valid(&checkpoint, &abstract_checkpoint) {
            return Ok(checkpoint);
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid existing second-momentum embedded checkpoint",
        ));
    }
    let fixture = fixture_for(&job.source_dynkin_label, job.source_copy);
    let started = Instant::now();
    let (certificate, coupled_map_sha256) = crate::eleven_dimensional_level16_couplings::
        verify_second_momentum_30001_embedding_with_hash(
            &abstract_checkpoint.certificate,
            fixture.copy,
            fixture.artifact,
            2,
            fixture.bytes,
        );
    let checkpoint = SecondMomentum30001EmbeddedCheckpoint {
        schema_version: format!("{SCHEMA_VERSION}-embedded"),
        job: job.clone(),
        source_fixture: fixture.artifact.to_string(),
        source_fixture_sha256: sha256(fixture.bytes),
        abstract_certificate_sha256: abstract_checkpoint.certificate_sha256,
        coupled_map_sha256,
        target_dimension: EXPECTED_TARGET_DIMENSION,
        certified_irrep_image_rank: EXPECTED_TARGET_DIMENSION,
        rank_derivation: "a nonzero B5-equivariant map onto the multiplicity-one irreducible (30001) target is surjective".to_string(),
        elapsed_milliseconds: started.elapsed().as_millis(),
        observed_process_rss_bytes: observed_process_rss_bytes(),
        passed: certificate.passed
            && certificate.exact_raising_residual_terms_by_simple_root == [0; 5],
        certificate,
    };
    if !checkpoint.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact level-12 embedded intertwiner failed",
        ));
    }
    atomic_json(&path, &checkpoint)?;
    Ok(checkpoint)
}

pub fn construct_mutation_control(
    directory: &Path,
) -> io::Result<SecondMomentum30001MutationControl> {
    let source = "40000";
    let job = SecondMomentum30001MapJob {
        source_dynkin_label: source.to_string(),
        source_copy: 1,
    };
    let abstract_checkpoint = construct_abstract_checkpoint(directory, source)?;
    let mut mutated = abstract_checkpoint.certificate;
    let mutated_primitive_coefficient = mutated
        .primitive_domain_coefficients
        .iter()
        .position(|coefficient| *coefficient != 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty abstract map"))?;
    mutated.primitive_domain_coefficients[mutated_primitive_coefficient] += 1;
    let fixture = fixture_for(source, 1);
    let (certificate, _) = crate::eleven_dimensional_level16_couplings::
        verify_second_momentum_30001_embedding_with_hash(
            &mutated,
            fixture.copy,
            fixture.artifact,
            2,
            fixture.bytes,
        );
    let exact_raising_residual_detected_mutation =
        !certificate.passed && certificate.exact_raising_residual_terms_by_simple_root != [0; 5];
    let control = SecondMomentum30001MutationControl {
        source_dynkin_label: job.source_dynkin_label,
        source_copy: job.source_copy,
        mutated_primitive_coefficient,
        exact_raising_residual_detected_mutation,
        passed: exact_raising_residual_detected_mutation,
    };
    if !control.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "coefficient mutation escaped exact raising residual",
        ));
    }
    atomic_json(&mutation_path(directory), &control)?;
    Ok(control)
}

fn completed_abstracts(
    directory: &Path,
) -> BTreeMap<String, SecondMomentum30001AbstractCheckpoint> {
    source_labels()
        .into_iter()
        .filter_map(|source| {
            let checkpoint = read_abstract(directory, source).ok()?;
            (checkpoint.source_dynkin_label == source && abstract_is_valid(&checkpoint))
                .then_some((source.to_string(), checkpoint))
        })
        .collect()
}

fn completed_embedded(directory: &Path) -> Vec<SecondMomentum30001EmbeddedCheckpoint> {
    let abstracts = completed_abstracts(directory);
    worklist()
        .into_iter()
        .filter_map(|job| {
            let checkpoint: SecondMomentum30001EmbeddedCheckpoint =
                serde_json::from_reader(File::open(embedded_path(directory, &job)).ok()?).ok()?;
            let abstract_checkpoint = abstracts.get(&job.source_dynkin_label)?;
            (checkpoint.job == job && embedded_is_valid(&checkpoint, abstract_checkpoint))
                .then_some(checkpoint)
        })
        .collect()
}

pub fn summarize(directory: &Path) -> SecondMomentum30001Report {
    let abstracts = completed_abstracts(directory);
    let embedded = completed_embedded(directory);
    let completed_jobs = embedded
        .iter()
        .map(|checkpoint| checkpoint.job.clone())
        .collect::<BTreeSet<_>>();
    let remaining_jobs = worklist()
        .into_iter()
        .filter(|job| !completed_jobs.contains(job))
        .collect::<Vec<_>>();
    let mut exact_embedded_maps_by_source = BTreeMap::new();
    for checkpoint in &embedded {
        *exact_embedded_maps_by_source
            .entry(checkpoint.job.source_dynkin_label.clone())
            .or_insert(0) += 1;
    }
    let hashes = embedded
        .iter()
        .map(|checkpoint| checkpoint.coupled_map_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let mutation_control: Option<SecondMomentum30001MutationControl> =
        File::open(mutation_path(directory))
            .ok()
            .and_then(|file| serde_json::from_reader(file).ok())
            .filter(|control: &SecondMomentum30001MutationControl| {
                control.source_dynkin_label == "40000"
                    && control.source_copy == 1
                    && control.exact_raising_residual_detected_mutation
                    && control.passed
            });
    let every_exact_raising_residual_is_zero = abstracts.values().all(|checkpoint| {
        checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
    }) && embedded.iter().all(|checkpoint| {
        checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
    });
    let every_source_copy_map_hash_is_distinct = hashes.len() == embedded.len();
    let checkpointed_runtime_milliseconds = abstracts
        .values()
        .map(|checkpoint| checkpoint.elapsed_milliseconds)
        .chain(
            embedded
                .iter()
                .map(|checkpoint| checkpoint.elapsed_milliseconds),
        )
        .sum();
    let maximum_observed_process_rss_bytes = abstracts
        .values()
        .filter_map(|checkpoint| checkpoint.observed_process_rss_bytes)
        .chain(
            embedded
                .iter()
                .filter_map(|checkpoint| checkpoint.observed_process_rss_bytes),
        )
        .max();
    let maximum_csr_storage_lower_bound_bytes = abstracts
        .values()
        .map(|checkpoint| checkpoint.csr_storage_lower_bound_bytes)
        .max()
        .unwrap_or(0);
    let source_to_30001_component_maps_complete = abstracts.len() == 8
        && embedded.len() == 15
        && remaining_jobs.is_empty()
        && every_exact_raising_residual_is_zero
        && every_source_copy_map_hash_is_distinct
        && mutation_control
            .as_ref()
            .is_some_and(|control| control.passed);
    SecondMomentum30001Report {
        schema_version: SCHEMA_VERSION.to_string(),
        exterior_degree: 12,
        intermediate_dynkin_label: TARGET_DYNKIN_LABEL.to_string(),
        expected_source_irreps: 8,
        exact_abstract_intertwiners: abstracts.len(),
        expected_source_copies: 15,
        exact_embedded_component_maps: embedded.len(),
        exact_embedded_maps_by_source,
        certified_intermediate_target_rank: if embedded.is_empty() {
            0
        } else {
            EXPECTED_TARGET_DIMENSION
        },
        every_exact_raising_residual_is_zero,
        every_source_copy_map_hash_is_distinct,
        checkpointed_runtime_milliseconds,
        maximum_observed_process_rss_bytes,
        maximum_csr_storage_lower_bound_bytes,
        mutation_control,
        remaining_jobs,
        source_to_30001_component_maps_complete,
        momentum_recoupling_30001_to_10001_complete: false,
        physical_target_gauge_quotient_complete: false,
        full_physical_f_complete: false,
        full_f_a_g_p_zero_proved: false,
        passed: source_to_30001_component_maps_complete,
        boundary: "This certifies all fifteen exact level-12 source-times-spinor component maps into the (30001) intermediate channel when passed. The momentum recoupling from (30001) into Sym^2(V) tensor (10001), the physical target gauge quotient, complete F, and full F A G_p = 0 remain unproved.".to_string(),
    }
}

pub fn construct_all(directory: &Path) -> io::Result<SecondMomentum30001Report> {
    for source in source_labels() {
        construct_abstract_checkpoint(directory, source)?;
    }
    for job in worklist() {
        construct_embedded_checkpoint(directory, &job)?;
    }
    construct_mutation_control(directory)?;
    let report = summarize(directory);
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incomplete exact (30001) second-momentum map tranche",
        ));
    }
    Ok(report)
}

/// Build or resume every `(30001)` checkpoint and atomically publish the
/// aggregate validation report.
pub fn write_artifact(
    checkpoint_directory: &Path,
    output_path: &Path,
) -> io::Result<SecondMomentum30001Report> {
    let report = construct_all(checkpoint_directory)?;
    atomic_json(output_path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(role: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "adynkra-11d-30001-{role}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn worklist_has_eight_source_irreps_and_fifteen_copies() {
        let jobs = worklist();
        assert_eq!(jobs.len(), 15);
        assert_eq!(source_labels().len(), 8);
        assert_eq!(
            crate::eleven_dimensional_prepotential::b5_dimension(TARGET_DYNKIN_LABEL),
            EXPECTED_TARGET_DIMENSION
        );
        assert_eq!(
            jobs.iter()
                .fold(BTreeMap::<String, usize>::new(), |mut counts, job| {
                    *counts.entry(job.source_dynkin_label.clone()).or_default() += 1;
                    counts
                }),
            BTreeMap::from([
                ("20002".to_string(), 2),
                ("20010".to_string(), 3),
                ("20100".to_string(), 2),
                ("30002".to_string(), 3),
                ("30010".to_string(), 2),
                ("30100".to_string(), 1),
                ("31000".to_string(), 1),
                ("40000".to_string(), 1),
            ])
        );
    }

    #[test]
    fn every_binary_fixture_is_hash_distinct_and_an_exact_highest_weight() {
        let mut hashes = BTreeSet::new();
        for fixture in source_fixtures() {
            assert!(hashes.insert(sha256(fixture.bytes)));
            let masks =
                crate::eleven_dimensional_level16_couplings::exterior_highest_weight_basis_masks(
                    12,
                    fixture.dynkin_label,
                );
            let coefficients = fixture
                .bytes
                .chunks_exact(2)
                .map(|pair| i64::from(i16::from_le_bytes([pair[0], pair[1]])))
                .collect::<Vec<_>>();
            assert_eq!(masks.len(), coefficients.len());
            let terms = masks
                .into_iter()
                .zip(coefficients)
                .filter(|(_, coefficient)| *coefficient != 0)
                .collect::<Vec<_>>();
            assert!(crate::eleven_dimensional_level16_couplings::
                exterior_highest_weight_raising_residuals_are_zero(&terms));
        }
        assert_eq!(hashes.len(), 15);
    }

    #[test]
    fn summary_is_fail_closed_without_exact_checkpoints() {
        let temporary = temporary_directory("empty");
        let report = summarize(&temporary);
        assert!(!report.passed);
        assert!(!report.source_to_30001_component_maps_complete);
        assert!(!report.momentum_recoupling_30001_to_10001_complete);
        assert!(!report.physical_target_gauge_quotient_complete);
        assert!(!report.full_physical_f_complete);
        assert!(!report.full_f_a_g_p_zero_proved);
        assert_eq!(report.remaining_jobs.len(), 15);
        fs::remove_dir_all(temporary).unwrap();
    }

    #[test]
    #[ignore = "constructs and checkpoints the real level-12 (40000) to (30001) map"]
    fn representative_40000_map_and_mutation_control_pass() {
        let temporary = temporary_directory("representative");
        let abstract_checkpoint = construct_abstract_checkpoint(&temporary, "40000").unwrap();
        assert!(abstract_checkpoint.passed);
        let job = SecondMomentum30001MapJob {
            source_dynkin_label: "40000".to_string(),
            source_copy: 1,
        };
        let embedded = construct_embedded_checkpoint(&temporary, &job).unwrap();
        assert!(embedded.passed);
        assert_eq!(
            embedded.certified_irrep_image_rank,
            EXPECTED_TARGET_DIMENSION
        );
        assert!(construct_mutation_control(&temporary).unwrap().passed);
        fs::remove_dir_all(temporary).unwrap();
    }

    #[test]
    #[ignore = "writes all fifteen exact (30001) checkpoints and their aggregate report"]
    fn write_complete_30001_map_tranche() {
        let report = write_artifact(
            Path::new("results/adynkra_11d_second_momentum_30001_checkpoints"),
            Path::new("results/adynkra_11d_second_momentum_30001_maps.json"),
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.exact_abstract_intertwiners, 8);
        assert_eq!(report.exact_embedded_component_maps, 15);
        assert!(report.remaining_jobs.is_empty());
    }
}
