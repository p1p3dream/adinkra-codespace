//! Checkpointed exact construction of the seventy-seven level-18 source maps.
//!
//! Appendix F of arXiv:2002.08502 fixes the scalar-superfield inventory.  The
//! four targets are the multiplicity-one summands of `(11000) tensor (00001)`.
//! Twenty abstract intertwiners into `(01001)`, `(10001)`, and `(20001)` were
//! already certified at level fourteen.  Abstract B5 intertwiners do not
//! depend on the exterior realization, so those certificates can be applied
//! to the exact level-18 source embeddings.  The fourteen `(11001)` abstract
//! intertwiners are constructed directly and checkpointed separately.
//!
//! This module certifies representation maps.  It does not supply the physical
//! curvature operator, physical target gauge maps, or a superspace quotient.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: &str = "adynkra-11d-level18-embedded-maps-v1";
const EXISTING_ABSTRACTS: &str =
    include_str!("../results/adynkra_11d_first_momentum_couplings_all.json");

#[derive(Clone, Debug, Deserialize)]
struct ExistingAbstractReport {
    abstract_couplings:
        Vec<crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Level18EmbeddedJob {
    pub target_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
}

impl Level18EmbeddedJob {
    pub fn key(&self) -> String {
        format!(
            "{}_from_{}_copy{}",
            self.target_dynkin_label, self.source_dynkin_label, self.source_copy
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Level18AbstractCheckpoint {
    pub schema_version: String,
    pub target_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_fixture_sha256: String,
    pub reused_level14_abstract_certificate: bool,
    pub certificate_sha256: String,
    pub certificate: crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Level18EmbeddedCheckpoint {
    pub schema_version: String,
    pub job: Level18EmbeddedJob,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub abstract_certificate_sha256: String,
    pub coupled_map_sha256: String,
    pub target_dimension: u64,
    pub certified_irrep_image_rank: u64,
    pub rank_derivation: String,
    pub certificate: crate::eleven_dimensional_level16_couplings::EmbeddedCouplingCertificate,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddedMutationControl {
    pub source_dynkin_label: String,
    pub target_dynkin_label: String,
    pub source_copy: usize,
    pub mutated_primitive_coefficient: usize,
    pub mutation_detected_by_exact_raising_residual: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactTargetGaugeQuotientApiAudit {
    pub exact_sparse_rank_api_available: bool,
    pub six_channel_positive_control_passed: bool,
    pub six_channel_negative_control_rejected: bool,
    pub provenance_fields_required: bool,
    pub physical_curvature_maps_supplied: bool,
    pub physical_target_gauge_maps_supplied: bool,
    pub actual_target_gauge_quotient_computed: bool,
    pub passed: bool,
    pub boundary: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Level18EmbeddedReport {
    pub schema_version: String,
    pub inventory_source: String,
    pub source_scope: String,
    pub target_labels: Vec<String>,
    pub expected_abstract_source_target_pairs: usize,
    pub exact_abstract_source_target_pairs: usize,
    pub reused_level14_abstract_source_target_pairs: usize,
    pub direct_level18_abstract_source_target_pairs: usize,
    pub expected_embedded_maps: usize,
    pub exact_embedded_maps: usize,
    pub exact_embedded_maps_by_target: BTreeMap<String, usize>,
    pub remaining_jobs: Vec<Level18EmbeddedJob>,
    pub every_completed_residual_is_zero: bool,
    pub every_completed_map_hash_is_unique_within_source_copy: bool,
    pub mutation_control: Option<EmbeddedMutationControl>,
    pub quotient_api: ExactTargetGaugeQuotientApiAudit,
    pub all_77_embedded_maps_complete: bool,
    pub physical_target_gauge_quotient_complete: bool,
    pub passed: bool,
    pub boundary: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Level18EmbeddedArtifact {
    pub schema_version: String,
    pub title: String,
    pub report: Level18EmbeddedReport,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn target_labels() -> Vec<String> {
    crate::eleven_dimensional_prepotential::spinor_tensor_channels("11000")
        .into_iter()
        .map(|(label, _)| label)
        .collect()
}

pub fn worklist() -> Vec<Level18EmbeddedJob> {
    let mut jobs = target_labels()
        .into_iter()
        .flat_map(|target| {
            crate::eleven_dimensional_prepotential::spinor_level_channel_sources(18, &target)
                .into_iter()
                .flat_map(move |(source, _, multiplicity)| {
                    let target = target.clone();
                    (1..=multiplicity).map(move |source_copy| Level18EmbeddedJob {
                        target_dynkin_label: target.clone(),
                        source_dynkin_label: source.clone(),
                        source_copy,
                    })
                })
        })
        .collect::<Vec<_>>();
    jobs.sort();
    assert_eq!(jobs.len(), 77);
    jobs
}

fn source_fixtures()
-> BTreeMap<(String, usize), crate::eleven_dimensional_level18_momentum::Level18SourceFixture> {
    crate::eleven_dimensional_level18_momentum::level18_source_fixtures()
        .into_iter()
        .map(|fixture| ((fixture.dynkin_label.clone(), fixture.copy), fixture))
        .collect()
}

fn existing_abstracts() -> BTreeMap<(String, String), Level18AbstractCheckpoint> {
    let report: ExistingAbstractReport =
        serde_json::from_str(EXISTING_ABSTRACTS).expect("parse committed abstract couplings");
    let fixtures = source_fixtures();
    report
        .abstract_couplings
        .into_iter()
        .filter(|certificate| {
            matches!(
                certificate.target_dynkin_label.as_str(),
                "01001" | "10001" | "20001"
            ) && fixtures.contains_key(&(certificate.source_dynkin_label.clone(), 1))
        })
        .map(|certificate| {
            let source = certificate.source_dynkin_label.clone();
            let target = certificate.target_dynkin_label.clone();
            let fixture = &fixtures[&(source.clone(), 1)];
            let payload = serde_json::to_vec(&certificate).expect("serialize abstract certificate");
            (
                (target.clone(), source.clone()),
                Level18AbstractCheckpoint {
                    schema_version: format!("{SCHEMA_VERSION}-abstract-v1"),
                    target_dynkin_label: target,
                    source_dynkin_label: source,
                    source_fixture_sha256: sha256(&fixture.bytes),
                    reused_level14_abstract_certificate: true,
                    certificate_sha256: sha256(&payload),
                    passed: certificate.passed,
                    certificate,
                },
            )
        })
        .collect()
}

pub fn abstract_checkpoint_path(directory: &Path, target: &str, source: &str) -> PathBuf {
    directory.join(format!("abstract_{target}_from_{source}.json"))
}

pub fn embedded_checkpoint_path(directory: &Path, job: &Level18EmbeddedJob) -> PathBuf {
    directory.join(format!("embedded_{}.json", job.key()))
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let file = File::create(&temporary)?;
    serde_json::to_writer_pretty(BufWriter::new(file), value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    fs::rename(temporary, path)
}

/// Write the twenty exterior-degree-independent abstract certificates already
/// proved by the level-14 computation.
pub fn checkpoint_reused_abstracts(directory: &Path) -> io::Result<usize> {
    let checkpoints = existing_abstracts();
    assert_eq!(checkpoints.len(), 20);
    for ((target, source), checkpoint) in &checkpoints {
        atomic_json(
            &abstract_checkpoint_path(directory, target, source),
            checkpoint,
        )?;
    }
    Ok(checkpoints.len())
}

pub fn construct_abstract_checkpoint(
    directory: &Path,
    target: &str,
    source: &str,
) -> io::Result<Level18AbstractCheckpoint> {
    if let Some(checkpoint) = existing_abstracts().remove(&(target.to_string(), source.to_string()))
    {
        atomic_json(
            &abstract_checkpoint_path(directory, target, source),
            &checkpoint,
        )?;
        return Ok(checkpoint);
    }
    assert_eq!(target, "11001", "only (11001) lacks a prior abstract map");
    let fixtures = source_fixtures();
    let fixture = &fixtures[&(source.to_string(), 1)];
    let certificate = if source == "11002" {
        crate::eleven_dimensional_level16_couplings::build_level18_abstract_low_memory(
            source,
            target,
            fixture.copy,
            fixture.coefficient_width_bytes,
            &fixture.bytes,
        )
    } else {
        crate::eleven_dimensional_level16_couplings::build_level18_abstract(
            source,
            target,
            fixture.copy,
            fixture.coefficient_width_bytes,
            &fixture.bytes,
        )
    };
    let payload = serde_json::to_vec(&certificate).expect("serialize abstract certificate");
    let checkpoint = Level18AbstractCheckpoint {
        schema_version: format!("{SCHEMA_VERSION}-abstract-v1"),
        target_dynkin_label: target.to_string(),
        source_dynkin_label: source.to_string(),
        source_fixture_sha256: sha256(&fixture.bytes),
        reused_level14_abstract_certificate: false,
        certificate_sha256: sha256(&payload),
        passed: certificate.passed,
        certificate,
    };
    if !checkpoint.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact abstract coupling failed",
        ));
    }
    atomic_json(
        &abstract_checkpoint_path(directory, target, source),
        &checkpoint,
    )?;
    Ok(checkpoint)
}

fn read_abstract_checkpoint(
    directory: &Path,
    target: &str,
    source: &str,
) -> io::Result<Level18AbstractCheckpoint> {
    serde_json::from_reader(File::open(abstract_checkpoint_path(
        directory, target, source,
    ))?)
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn construct_embedded_checkpoint(
    directory: &Path,
    job: &Level18EmbeddedJob,
) -> io::Result<Level18EmbeddedCheckpoint> {
    let abstract_checkpoint = read_abstract_checkpoint(
        directory,
        &job.target_dynkin_label,
        &job.source_dynkin_label,
    )?;
    if !abstract_checkpoint.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "abstract checkpoint did not pass",
        ));
    }
    let fixtures = source_fixtures();
    let fixture = &fixtures[&(job.source_dynkin_label.clone(), job.source_copy)];
    let (certificate, coupled_map_sha256) =
        crate::eleven_dimensional_level16_couplings::verify_level18_embedding_with_hash(
            &abstract_checkpoint.certificate,
            fixture.copy,
            &fixture.artifact,
            fixture.coefficient_width_bytes,
            &fixture.bytes,
        );
    let target_dimension =
        crate::eleven_dimensional_prepotential::b5_dimension(&job.target_dynkin_label);
    let checkpoint = Level18EmbeddedCheckpoint {
        schema_version: format!("{SCHEMA_VERSION}-embedded-v1"),
        job: job.clone(),
        source_fixture: fixture.artifact.clone(),
        source_fixture_sha256: sha256(&fixture.bytes),
        abstract_certificate_sha256: abstract_checkpoint.certificate_sha256,
        coupled_map_sha256,
        target_dimension,
        certified_irrep_image_rank: target_dimension,
        rank_derivation:
            "a nonzero B5-equivariant map onto a multiplicity-one irreducible target is surjective"
                .to_string(),
        passed: certificate.passed,
        certificate,
    };
    if !checkpoint.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact embedded coupling failed",
        ));
    }
    atomic_json(&embedded_checkpoint_path(directory, job), &checkpoint)?;
    Ok(checkpoint)
}

fn read_completed_abstracts(directory: &Path) -> Vec<Level18AbstractCheckpoint> {
    let fixtures = source_fixtures();
    let pairs = worklist()
        .into_iter()
        .map(|job| (job.target_dynkin_label, job.source_dynkin_label))
        .collect::<BTreeSet<_>>();
    pairs
        .into_iter()
        .filter_map(|(target, source)| {
            let checkpoint = read_abstract_checkpoint(directory, &target, &source).ok()?;
            Some(((target, source), checkpoint))
        })
        .filter(|((target, source), checkpoint)| {
            let certificate_payload =
                serde_json::to_vec(&checkpoint.certificate).expect("serialize certificate");
            let fixture = fixtures.get(&(checkpoint.source_dynkin_label.clone(), 1));
            checkpoint.target_dynkin_label == *target
                && checkpoint.source_dynkin_label == *source
                && checkpoint.schema_version == format!("{SCHEMA_VERSION}-abstract-v1")
                && checkpoint.passed
                && checkpoint.certificate.passed
                && checkpoint.certificate.source_dynkin_label == checkpoint.source_dynkin_label
                && checkpoint.certificate.target_dynkin_label == checkpoint.target_dynkin_label
                && is_sha256(&checkpoint.certificate_sha256)
                && is_sha256(&checkpoint.source_fixture_sha256)
                && checkpoint.certificate_sha256 == sha256(&certificate_payload)
                && fixture.is_some_and(|fixture| {
                    checkpoint.source_fixture_sha256 == sha256(&fixture.bytes)
                })
        })
        .map(|(_, checkpoint)| checkpoint)
        .collect()
}

fn read_completed_embedded(directory: &Path) -> Vec<Level18EmbeddedCheckpoint> {
    let fixtures = source_fixtures();
    let abstract_hashes = read_completed_abstracts(directory)
        .into_iter()
        .map(|checkpoint| {
            (
                (
                    checkpoint.target_dynkin_label,
                    checkpoint.source_dynkin_label,
                ),
                checkpoint.certificate_sha256,
            )
        })
        .collect::<BTreeMap<_, _>>();
    worklist()
        .into_iter()
        .filter_map(|job| {
            let checkpoint = serde_json::from_reader(
                File::open(embedded_checkpoint_path(directory, &job)).ok()?,
            )
            .ok()?;
            Some((job, checkpoint))
        })
        .filter(
            |(expected_job, checkpoint): &(Level18EmbeddedJob, Level18EmbeddedCheckpoint)| {
                let fixture = fixtures.get(&(
                    checkpoint.job.source_dynkin_label.clone(),
                    checkpoint.job.source_copy,
                ));
                let abstract_hash = abstract_hashes.get(&(
                    checkpoint.job.target_dynkin_label.clone(),
                    checkpoint.job.source_dynkin_label.clone(),
                ));
                checkpoint.job == *expected_job
                    && checkpoint.schema_version == format!("{SCHEMA_VERSION}-embedded-v1")
                    && checkpoint.passed
                    && checkpoint.certificate.passed
                    && checkpoint.certificate.source_dynkin_label
                        == checkpoint.job.source_dynkin_label
                    && checkpoint.certificate.target_dynkin_label
                        == checkpoint.job.target_dynkin_label
                    && checkpoint
                        .certificate
                        .exact_raising_residual_terms_by_simple_root
                        == [0; 5]
                    && is_sha256(&checkpoint.coupled_map_sha256)
                    && abstract_hash
                        .is_some_and(|hash| *hash == checkpoint.abstract_certificate_sha256)
                    && checkpoint.target_dimension
                        == crate::eleven_dimensional_prepotential::b5_dimension(
                            &checkpoint.job.target_dynkin_label,
                        )
                    && checkpoint.certified_irrep_image_rank == checkpoint.target_dimension
                    && fixture.is_some_and(|fixture| {
                        checkpoint.source_fixture == fixture.artifact
                            && checkpoint.source_fixture_sha256 == sha256(&fixture.bytes)
                    })
            },
        )
        .map(|(_, checkpoint)| checkpoint)
        .collect()
}

fn mutation_control_path(directory: &Path) -> PathBuf {
    directory.join("mutation_control.json")
}

pub fn construct_mutation_control(directory: &Path) -> io::Result<EmbeddedMutationControl> {
    let source = "12000";
    let target = "11001";
    let abstract_checkpoint = read_abstract_checkpoint(directory, target, source)?;
    let mut mutated = abstract_checkpoint.certificate;
    let coefficient = mutated
        .primitive_domain_coefficients
        .iter()
        .position(|value| *value != 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty abstract map"))?;
    mutated.primitive_domain_coefficients[coefficient] += 1;
    let fixtures = source_fixtures();
    let fixture = &fixtures[&(source.to_string(), 1)];
    let (certificate, _) =
        crate::eleven_dimensional_level16_couplings::verify_level18_embedding_with_hash(
            &mutated,
            fixture.copy,
            &fixture.artifact,
            fixture.coefficient_width_bytes,
            &fixture.bytes,
        );
    let mutation_detected_by_exact_raising_residual =
        certificate.exact_raising_residual_terms_by_simple_root != [0; 5] && !certificate.passed;
    let control = EmbeddedMutationControl {
        source_dynkin_label: source.to_string(),
        target_dynkin_label: target.to_string(),
        source_copy: 1,
        mutated_primitive_coefficient: coefficient,
        mutation_detected_by_exact_raising_residual,
        passed: mutation_detected_by_exact_raising_residual,
    };
    if !control.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "embedded map mutation was not detected",
        ));
    }
    atomic_json(&mutation_control_path(directory), &control)?;
    Ok(control)
}

fn read_mutation_control(directory: &Path) -> Option<EmbeddedMutationControl> {
    serde_json::from_reader(File::open(mutation_control_path(directory)).ok()?).ok()
}

fn sparse_entry(
    row: usize,
    column: usize,
) -> crate::eleven_dimensional_level18_momentum::ExactSparseMapEntry {
    crate::eleven_dimensional_level18_momentum::ExactSparseMapEntry {
        row,
        column,
        numerator: "1".to_string(),
        denominator: "1".to_string(),
    }
}

fn quotient_control_input(
    negative: bool,
) -> crate::eleven_dimensional_level18_momentum::TargetGaugeQuotientInput {
    use crate::eleven_dimensional_level18_momentum::{
        ExactSparseMapInput, TargetGaugeChannelQuotientInput, TargetGaugeQuotientInput,
    };
    let parameter_components = [1, 11, 55, 165, 330, 462];
    let (target_stream_content_sha256, source_fixed_curvature_content_sha256) =
        crate::eleven_dimensional_level18_momentum::target_gauge_quotient_provenance_hashes();
    TargetGaugeQuotientInput {
        target_stream_schema_version: "adynkra-11d-target-resolved-composition-stream-v2"
            .to_string(),
        target_stream_content_sha256,
        source_fixed_curvature_schema_version: "adynkra-11d-source-fixed-curvature-scaffold-v1"
            .to_string(),
        source_fixed_curvature_content_sha256,
        channels: parameter_components
            .into_iter()
            .enumerate()
            .map(|(form_degree, columns)| TargetGaugeChannelQuotientInput {
                form_degree,
                parameter_components: columns,
                target_gauge_map: ExactSparseMapInput {
                    rows: 2,
                    columns: 1,
                    entries: vec![sparse_entry(0, 0)],
                },
                curvature_variation: ExactSparseMapInput {
                    rows: 2,
                    columns,
                    entries: if negative {
                        vec![sparse_entry(1, 0)]
                    } else {
                        vec![sparse_entry(0, 0)]
                    },
                },
            })
            .collect(),
    }
}

fn quotient_api_audit() -> ExactTargetGaugeQuotientApiAudit {
    let positive = crate::eleven_dimensional_level18_momentum::evaluate_target_gauge_quotient(
        &quotient_control_input(false),
    )
    .expect("positive quotient control input is valid");
    let negative = crate::eleven_dimensional_level18_momentum::evaluate_target_gauge_quotient(
        &quotient_control_input(true),
    )
    .expect("negative quotient control input is valid");
    let six_channel_positive_control_passed = positive.passed
        && positive.channels.len() == 6
        && positive
            .channels
            .iter()
            .all(|channel| channel.curvature_variation_lies_in_target_gauge_image);
    let six_channel_negative_control_rejected = !negative.passed
        && negative.channels.len() == 6
        && negative
            .channels
            .iter()
            .all(|channel| !channel.curvature_variation_lies_in_target_gauge_image);
    ExactTargetGaugeQuotientApiAudit {
        exact_sparse_rank_api_available: true,
        six_channel_positive_control_passed,
        six_channel_negative_control_rejected,
        provenance_fields_required: true,
        physical_curvature_maps_supplied: false,
        physical_target_gauge_maps_supplied: false,
        actual_target_gauge_quotient_computed: false,
        passed: six_channel_positive_control_passed && six_channel_negative_control_rejected,
        boundary: "The controls certify exact sparse image containment and rejection under content hashes of the committed target-stream and source-fixed-curvature validation artifacts. No physical map is supplied or claimed."
            .to_string(),
    }
}

pub fn verify_in(directory: &Path) -> Level18EmbeddedReport {
    let jobs = worklist();
    let abstracts = read_completed_abstracts(directory);
    let embedded = read_completed_embedded(directory);
    let completed_keys = embedded
        .iter()
        .map(|checkpoint| checkpoint.job.clone())
        .collect::<BTreeSet<_>>();
    let remaining_jobs = jobs
        .iter()
        .filter(|job| !completed_keys.contains(*job))
        .cloned()
        .collect::<Vec<_>>();
    let mut exact_embedded_maps_by_target = BTreeMap::new();
    for checkpoint in &embedded {
        *exact_embedded_maps_by_target
            .entry(checkpoint.job.target_dynkin_label.clone())
            .or_insert(0) += 1;
    }
    let every_completed_residual_is_zero = embedded.iter().all(|checkpoint| {
        checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            == [0; 5]
    });
    let hashes_by_source_copy = embedded.iter().fold(
        BTreeMap::<(String, usize), Vec<String>>::new(),
        |mut hashes, checkpoint| {
            hashes
                .entry((
                    checkpoint.job.source_dynkin_label.clone(),
                    checkpoint.job.source_copy,
                ))
                .or_default()
                .push(checkpoint.coupled_map_sha256.clone());
            hashes
        },
    );
    let every_completed_map_hash_is_unique_within_source_copy = hashes_by_source_copy
        .values()
        .all(|hashes| hashes.iter().collect::<BTreeSet<_>>().len() == hashes.len());
    let quotient_api = quotient_api_audit();
    let mutation_control = read_mutation_control(directory).filter(|control| control.passed);
    let all_77_embedded_maps_complete = embedded.len() == 77
        && remaining_jobs.is_empty()
        && every_completed_residual_is_zero
        && every_completed_map_hash_is_unique_within_source_copy;
    Level18EmbeddedReport {
        schema_version: SCHEMA_VERSION.to_string(),
        inventory_source: "S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, arXiv:2002.08502, Appendix F"
            .to_string(),
        source_scope: "The paper fixes the representation inventory. The exact Clebsch-Gordan coefficients here are computationally derived under the committed B5 exterior-basis conventions."
            .to_string(),
        target_labels: target_labels(),
        expected_abstract_source_target_pairs: 34,
        exact_abstract_source_target_pairs: abstracts.len(),
        reused_level14_abstract_source_target_pairs: abstracts
            .iter()
            .filter(|checkpoint| checkpoint.reused_level14_abstract_certificate)
            .count(),
        direct_level18_abstract_source_target_pairs: abstracts
            .iter()
            .filter(|checkpoint| !checkpoint.reused_level14_abstract_certificate)
            .count(),
        expected_embedded_maps: 77,
        exact_embedded_maps: embedded.len(),
        exact_embedded_maps_by_target,
        remaining_jobs,
        every_completed_residual_is_zero,
        every_completed_map_hash_is_unique_within_source_copy,
        mutation_control,
        quotient_api: quotient_api.clone(),
        all_77_embedded_maps_complete,
        physical_target_gauge_quotient_complete: false,
        passed: quotient_api.passed
            && every_completed_residual_is_zero
            && every_completed_map_hash_is_unique_within_source_copy,
        boundary: "Representation-map completion is independent of the physical K/F problem. Even after all seventy-seven maps pass, the physical quotient remains false until convention-fixed curvature variations and target gauge images are supplied."
            .to_string(),
    }
}

pub fn verify() -> Level18EmbeddedReport {
    verify_in(Path::new("results/eleven_dimensional_level18_embedded"))
}

pub fn write_artifact(
    checkpoint_directory: &Path,
    output: &Path,
) -> io::Result<Level18EmbeddedReport> {
    let report = verify_in(checkpoint_directory);
    let artifact = Level18EmbeddedArtifact {
        schema_version: format!("{SCHEMA_VERSION}-artifact-v1"),
        title: "Exact level-18 embedded source-target map census".to_string(),
        report: report.clone(),
    };
    atomic_json(output, &artifact)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_worklist_has_the_published_77_incidence_count() {
        let jobs = worklist();
        assert_eq!(jobs.len(), 77);
        let by_target = jobs.into_iter().fold(BTreeMap::new(), |mut counts, job| {
            *counts.entry(job.target_dynkin_label).or_insert(0) += 1;
            counts
        });
        assert_eq!(
            by_target,
            BTreeMap::from([
                ("01001".to_string(), 18),
                ("10001".to_string(), 8),
                ("11001".to_string(), 38),
                ("20001".to_string(), 13),
            ])
        );
    }

    #[test]
    fn prior_abstract_maps_cover_exactly_twenty_pairs() {
        let abstracts = existing_abstracts();
        assert_eq!(abstracts.len(), 20);
        assert!(abstracts.values().all(|checkpoint| checkpoint.passed));
    }

    #[test]
    fn quotient_api_accepts_image_and_rejects_transverse_variation() {
        let audit = quotient_api_audit();
        assert!(audit.passed);
        assert!(!audit.actual_target_gauge_quotient_computed);
    }

    #[test]
    fn complete_checkpoint_report_has_exact_provenance() {
        let report = verify();
        assert_eq!(report.exact_abstract_source_target_pairs, 34);
        assert_eq!(report.exact_embedded_maps, 77);
        assert!(report.remaining_jobs.is_empty());
        assert!(report.every_completed_residual_is_zero);
        assert!(report.every_completed_map_hash_is_unique_within_source_copy);
        assert!(report.all_77_embedded_maps_complete);
        assert!(report.passed);
        assert!(!report.physical_target_gauge_quotient_complete);
    }

    #[test]
    fn partial_checkpoint_report_is_fail_closed_about_physics() {
        let temporary = std::env::temp_dir().join(format!(
            "adynkra-level18-embedded-empty-{}",
            std::process::id()
        ));
        let report = verify_in(&temporary);
        assert_eq!(report.exact_embedded_maps, 0);
        assert_eq!(report.remaining_jobs.len(), 77);
        assert!(!report.all_77_embedded_maps_complete);
        assert!(!report.physical_target_gauge_quotient_complete);
        assert!(report.passed);
    }

    #[test]
    #[ignore = "writes the reusable abstracts and thirty-nine non-(11001) embedded maps"]
    fn write_reused_embedded_checkpoints() {
        let directory = Path::new("results/eleven_dimensional_level18_embedded");
        assert_eq!(checkpoint_reused_abstracts(directory).unwrap(), 20);
        let jobs = worklist()
            .into_iter()
            .filter(|job| job.target_dynkin_label != "11001")
            .collect::<Vec<_>>();
        assert_eq!(jobs.len(), 39);
        for (ordinal, job) in jobs.iter().enumerate() {
            eprintln!("embedded map {}/39: {}", ordinal + 1, job.key());
            construct_embedded_checkpoint(directory, job).unwrap();
        }
        let report = write_artifact(
            directory,
            Path::new("results/adynkra_11d_level18_embedded_maps.json"),
        )
        .unwrap();
        assert_eq!(report.exact_abstract_source_target_pairs, 20);
        assert_eq!(report.exact_embedded_maps, 39);
    }

    #[test]
    #[ignore = "constructs one expensive direct (11001) abstract and all its source copies"]
    fn write_one_11001_source_from_environment() {
        let source = std::env::var("ADYNKRA_LEVEL18_SOURCE")
            .expect("set ADYNKRA_LEVEL18_SOURCE to one (11001) source label");
        let directory = Path::new("results/eleven_dimensional_level18_embedded");
        construct_abstract_checkpoint(directory, "11001", &source).unwrap();
        let jobs = worklist()
            .into_iter()
            .filter(|job| job.target_dynkin_label == "11001" && job.source_dynkin_label == source)
            .collect::<Vec<_>>();
        assert!(!jobs.is_empty());
        for job in &jobs {
            eprintln!("embedded map: {}", job.key());
            construct_embedded_checkpoint(directory, job).unwrap();
        }
        write_artifact(
            directory,
            Path::new("results/adynkra_11d_level18_embedded_maps.json"),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "writes the exact coefficient-mutation negative control"]
    fn write_mutation_control() {
        let directory = Path::new("results/eleven_dimensional_level18_embedded");
        let control = construct_mutation_control(directory).unwrap();
        assert!(control.passed);
        let report = write_artifact(
            directory,
            Path::new("results/adynkra_11d_level18_embedded_maps.json"),
        )
        .unwrap();
        assert!(
            report
                .mutation_control
                .is_some_and(|control| control.passed)
        );
    }
}
