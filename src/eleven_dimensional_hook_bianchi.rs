//! Bounded representation-level continuation of the 11D `(11000)` hook.
//!
//! The committed level-16 to level-17 exterior-symbol matrix is exact and
//! surjective onto its seven-dimensional hook-coupling space. This module
//! audits that result, enumerates every B5 channel in
//! `S tensor (11000)`, and applies the strongest next-Bianchi screen possible
//! without level-18 embedded highest-weight kernels. The screen deliberately
//! relaxes each possible next map to an arbitrary rational row on the seven
//! hook copies. Since the committed matrix has full row rank, exact
//! composition forces every such row to vanish.
//!
//! This is a zero-spacetime-momentum representation-level result. It does not
//! replace the missing embedded level-18 coupling calculation, momentum
//! corrections, gauge quotient, or polynomial-module cohomology.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "adynkra-11d-hook-bianchi-v1";
const HOOK_LABEL: &str = "11000";
const LEVEL17_MATRIX_JSON: &str =
    include_str!("../results/adynkra_11d_level17_derivative_matrix.json");

const EXPECTED_LEVEL16_BASIS: [&str; 12] = [
    "10000#1", "20000#1", "00100#1", "00100#2", "00010#1", "00010#2", "00002#1", "10100#1",
    "10010#1", "10002#1", "10002#2", "10002#3",
];

const EXPECTED_LEVEL17_HOOK_BASIS: [&str; 7] = [
    "10001#1", "01001#1", "01001#2", "20001#1", "11001#1", "11001#2", "11001#3",
];

type Rational = Ratio<BigInt>;

#[derive(Clone, Debug, Serialize)]
pub struct ExactLevel17MapAudit {
    pub input_artifact: &'static str,
    pub input_sha256: String,
    pub input_schema_version: String,
    pub source_columns: usize,
    pub hook_rows: usize,
    pub exact_rank: usize,
    pub exact_nullity: usize,
    pub recomputed_kernel_dimension: usize,
    pub committed_kernel_vectors: usize,
    pub committed_kernel_vectors_annihilated: usize,
    pub left_kernel_dimension: usize,
    pub source_basis_matches: bool,
    pub hook_basis_matches: bool,
    pub scalar_factorizing_direction_is_closed: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Level18SourceChannel {
    pub source_dynkin_label: String,
    pub source_dimension: u64,
    pub multiplicity: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Level18TargetCandidate {
    pub target_dynkin_label: String,
    pub target_dimension: u64,
    pub multiplicity_in_hook_tensor_spinor: usize,
    pub level18_source_channels: Vec<Level18SourceChannel>,
    pub level18_distinct_source_irreps: usize,
    pub level18_embedded_source_copies: usize,
    pub relaxed_composition_row_dimension: usize,
    pub exact_composition_compatible_row_dimension: usize,
    pub nonzero_representation_level_bianchi_row_survives: bool,
    pub abstract_target_coupling_available: bool,
    pub embedded_level18_source_kernels_available: bool,
    pub embedded_composition_computed: bool,
    pub status: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratedWorkItem {
    pub ordinal: usize,
    pub stage: &'static str,
    pub target_dynkin_label: Option<String>,
    pub source_dynkin_label: Option<String>,
    pub source_copy: Option<usize>,
    pub expected_artifact: String,
    pub blocked_by: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct BoundedCohomology {
    pub field: &'static str,
    pub complex: &'static str,
    pub level16_dimension: usize,
    pub level16_to_level17_rank: usize,
    pub level16_kernel_dimension: usize,
    pub level16_cohomology_lower_bound: usize,
    pub level16_cohomology_upper_bound: usize,
    pub level16_incoming_image_known: bool,
    pub level17_dimension: usize,
    pub forced_level17_to_level18_rank: usize,
    pub level17_kernel_dimension: usize,
    pub level17_image_dimension: usize,
    pub level17_bounded_cohomology_dimension: usize,
    pub level18_candidate_irreps: usize,
    pub level18_candidate_embedded_copies: usize,
    pub level18_cohomology_computed: bool,
    pub interpretation: &'static str,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompletenessFlags {
    pub committed_level17_matrix_reverified_exactly: bool,
    pub hook_tensor_spinor_decomposition_complete: bool,
    pub level18_inventory_incidence_complete: bool,
    pub level14_hodge_dual_analogue_manifest_audited: bool,
    pub hodge_duality_kernel_lift_verified: bool,
    pub relaxed_representation_level_composition_solved_exactly: bool,
    pub bounded_zero_momentum_level17_cohomology_computed: bool,
    pub abstract_target_couplings_constructed: bool,
    pub embedded_level18_highest_weight_kernels_present: bool,
    pub embedded_next_bianchi_compositions_computed: bool,
    pub momentum_corrected_complex_computed: bool,
    pub gauge_quotient_computed: bool,
    pub polynomial_module_cohomology_computed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct HookBianchiReport {
    pub schema_version: &'static str,
    pub source_hook_dynkin_label: &'static str,
    pub source_hook_dimension: u64,
    pub spinor_dimension: usize,
    pub tensor_product_dimension: u64,
    pub target_dimension_sum: u64,
    pub target_dimension_sum_matches: bool,
    pub target_candidates: Vec<Level18TargetCandidate>,
    pub target_candidate_count: usize,
    pub level18_distinct_source_fixture_count: usize,
    pub level18_source_fixture_copy_count: usize,
    pub level14_hodge_dual_analogue_fixture_count: usize,
    pub level18_source_fixtures_present: usize,
    pub level18_source_target_embedded_copy_count: usize,
    pub level17_map: ExactLevel17MapAudit,
    pub cohomology: BoundedCohomology,
    pub generated_work_items: usize,
    pub completeness: CompletenessFlags,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct HookBianchiArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub report: HookBianchiReport,
    pub worklist: Vec<GeneratedWorkItem>,
}

fn parse_rational(
    entry: &crate::eleven_dimensional_level16_couplings::RationalMatrixEntry,
) -> Rational {
    let numerator = entry
        .numerator
        .parse::<BigInt>()
        .expect("committed matrix numerator is an integer");
    let denominator = entry
        .denominator
        .parse::<BigInt>()
        .expect("committed matrix denominator is an integer");
    assert!(!denominator.is_zero(), "committed denominator is nonzero");
    Ratio::new(numerator, denominator)
}

fn multiply(matrix: &[Vec<Rational>], vector: &[Rational]) -> Vec<Rational> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .fold(Rational::zero(), |sum, (left, right)| sum + left * right)
        })
        .collect()
}

fn transpose(matrix: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    (0..matrix[0].len())
        .map(|column| matrix.iter().map(|row| row[column].clone()).collect())
        .collect()
}

fn exact_level17_audit() -> (ExactLevel17MapAudit, Vec<Vec<Rational>>) {
    let committed: crate::eleven_dimensional_level16_couplings::Level17DerivativeMatrixReport =
        serde_json::from_str(LEVEL17_MATRIX_JSON).expect("parse committed level-17 matrix");
    let matrix = committed
        .matrix_rows_by_hook_columns_by_source
        .iter()
        .map(|row| row.iter().map(parse_rational).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert!(matrix.iter().all(|row| row.len() == 12));
    let exact_rank = crate::eleven_dimensional_level16_couplings::rational_matrix_rank(&matrix);
    let recomputed_kernel =
        crate::eleven_dimensional_level16_couplings::rational_nullspace(&matrix);
    let left_kernel =
        crate::eleven_dimensional_level16_couplings::rational_nullspace(&transpose(&matrix));
    let committed_kernel = committed
        .primitive_integer_kernel_basis
        .iter()
        .map(|vector| {
            vector
                .iter()
                .map(|value| {
                    Ratio::from_integer(
                        value
                            .parse::<BigInt>()
                            .expect("committed kernel coefficient is an integer"),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let committed_kernel_vectors_annihilated = committed_kernel
        .iter()
        .filter(|vector| multiply(&matrix, vector).iter().all(Ratio::is_zero))
        .count();
    let source_basis_matches = committed.source_basis
        == EXPECTED_LEVEL16_BASIS
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
    let hook_basis_matches = committed.hook_basis
        == EXPECTED_LEVEL17_HOOK_BASIS
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
    let input_sha256 = format!("{:x}", Sha256::digest(LEVEL17_MATRIX_JSON.as_bytes()));
    let passed = committed.passed
        && matrix.len() == 7
        && exact_rank == 7
        && recomputed_kernel.len() == 5
        && committed_kernel.len() == 5
        && committed_kernel_vectors_annihilated == committed_kernel.len()
        && left_kernel.is_empty()
        && source_basis_matches
        && hook_basis_matches
        && committed.scalar_factorizing_hook_image_is_zero;
    (
        ExactLevel17MapAudit {
            input_artifact: "results/adynkra_11d_level17_derivative_matrix.json",
            input_sha256,
            input_schema_version: committed.schema_version,
            source_columns: matrix.first().map_or(0, Vec::len),
            hook_rows: matrix.len(),
            exact_rank,
            exact_nullity: matrix.first().map_or(0, Vec::len) - exact_rank,
            recomputed_kernel_dimension: recomputed_kernel.len(),
            committed_kernel_vectors: committed_kernel.len(),
            committed_kernel_vectors_annihilated,
            left_kernel_dimension: left_kernel.len(),
            source_basis_matches,
            hook_basis_matches,
            scalar_factorizing_direction_is_closed: committed.scalar_factorizing_hook_image_is_zero,
            passed,
        },
        matrix,
    )
}

fn level18_candidates(left_kernel_dimension: usize) -> Vec<Level18TargetCandidate> {
    crate::eleven_dimensional_prepotential::spinor_tensor_channels(HOOK_LABEL)
        .into_iter()
        .map(|(target, target_dimension)| {
            let multiplicity_in_hook_tensor_spinor =
                crate::eleven_dimensional_prepotential::spinor_tensor_channels(HOOK_LABEL)
                    .iter()
                    .filter(|(label, _)| label == &target)
                    .count();
            let level18_source_channels =
                crate::eleven_dimensional_prepotential::spinor_level_channel_sources(18, &target)
                    .into_iter()
                    .map(
                        |(source_dynkin_label, source_dimension, multiplicity)| {
                            Level18SourceChannel {
                                source_dynkin_label,
                                source_dimension,
                                multiplicity,
                            }
                        },
                    )
                    .collect::<Vec<_>>();
            let level18_embedded_source_copies = level18_source_channels
                .iter()
                .map(|channel| channel.multiplicity)
                .sum();
            Level18TargetCandidate {
                target_dynkin_label: target,
                target_dimension,
                multiplicity_in_hook_tensor_spinor,
                level18_distinct_source_irreps: level18_source_channels.len(),
                level18_embedded_source_copies,
                level18_source_channels,
                relaxed_composition_row_dimension: 7,
                exact_composition_compatible_row_dimension: left_kernel_dimension,
                nonzero_representation_level_bianchi_row_survives: left_kernel_dimension != 0,
                abstract_target_coupling_available: false,
                embedded_level18_source_kernels_available: false,
                embedded_composition_computed: false,
                status: "excluded as a nonzero nilpotent row by the exact relaxed multiplicity-space screen; embedded construction remains a falsification check",
            }
        })
        .collect()
}

fn generated_worklist(candidates: &[Level18TargetCandidate]) -> Vec<GeneratedWorkItem> {
    let mut work = Vec::new();
    let mut ordinal = 1usize;
    let mut unique_source_copies = BTreeSet::new();
    let level14_analogues = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
        .into_iter()
        .map(|fixture| {
            (
                (fixture.dynkin_label.to_string(), fixture.copy),
                fixture.artifact,
            )
        })
        .collect::<BTreeMap<_, _>>();
    for candidate in candidates {
        for channel in &candidate.level18_source_channels {
            for copy in 1..=channel.multiplicity {
                unique_source_copies.insert((channel.source_dynkin_label.clone(), copy));
            }
        }
    }
    for (source, copy) in unique_source_copies {
        let analogue = level14_analogues.get(&(source.clone(), copy));
        let scalar_encoding = match analogue {
            Some(path) if path.ends_with(".i32le") => "i32le",
            Some(_) => "i16le",
            None => "integer-le",
        };
        work.push(GeneratedWorkItem {
            ordinal,
            stage: "level18_highest_weight_kernel",
            target_dynkin_label: None,
            source_dynkin_label: Some(source.clone()),
            source_copy: Some(copy),
            expected_artifact: format!(
                "data/eleven_dimensional_spinor_bridge/level18_{source}_highest_weight_kernel_{copy}.{scalar_encoding}"
            ),
            blocked_by: if analogue.is_some() {
                "a level14 analogue exists, but the exact Hodge lift and level18 raising verification are not implemented"
            } else {
                "no level14 analogue exists and the level18 kernel generator is not exposed by the current shared machinery"
            },
        });
        ordinal += 1;
    }
    for candidate in candidates {
        let target = &candidate.target_dynkin_label;
        work.push(GeneratedWorkItem {
            ordinal,
            stage: "abstract_hook_target_coupling",
            target_dynkin_label: Some(target.clone()),
            source_dynkin_label: Some(HOOK_LABEL.to_string()),
            source_copy: None,
            expected_artifact: format!(
                "results/adynkra_11d_hook_bianchi_target_{target}_abstract.json"
            ),
            blocked_by: "the generic S tensor (11000) coupling builder is not public",
        });
        ordinal += 1;
        for channel in &candidate.level18_source_channels {
            work.push(GeneratedWorkItem {
                ordinal,
                stage: "abstract_level18_source_target_coupling",
                target_dynkin_label: Some(target.clone()),
                source_dynkin_label: Some(channel.source_dynkin_label.clone()),
                source_copy: None,
                expected_artifact: format!(
                    "results/adynkra_11d_hook_bianchi_{target}_from_{}_abstract.json",
                    channel.source_dynkin_label
                ),
                blocked_by: "level18 embedded highest-weight kernels are unavailable",
            });
            ordinal += 1;
            for copy in 1..=channel.multiplicity {
                work.push(GeneratedWorkItem {
                    ordinal,
                    stage: "embedded_level18_source_target_coupling",
                    target_dynkin_label: Some(target.clone()),
                    source_dynkin_label: Some(channel.source_dynkin_label.clone()),
                    source_copy: Some(copy),
                    expected_artifact: format!(
                        "results/adynkra_11d_hook_bianchi_{target}_from_{}_copy{copy}.json",
                        channel.source_dynkin_label
                    ),
                    blocked_by: "level18 embedded highest-weight kernels are unavailable",
                });
                ordinal += 1;
            }
        }
        work.push(GeneratedWorkItem {
            ordinal,
            stage: "exact_embedded_composition",
            target_dynkin_label: Some(target.clone()),
            source_dynkin_label: None,
            source_copy: None,
            expected_artifact: format!(
                "results/adynkra_11d_hook_bianchi_{target}_composition.json"
            ),
            blocked_by: "both target and level18 source couplings must be constructed first",
        });
        ordinal += 1;
    }
    work
}

pub fn verify() -> HookBianchiReport {
    let (level17_map, _matrix) = exact_level17_audit();
    let candidates = level18_candidates(level17_map.left_kernel_dimension);
    let source_hook_dimension = crate::eleven_dimensional_prepotential::b5_dimension(HOOK_LABEL);
    let tensor_product_dimension = source_hook_dimension * 32;
    let target_dimension_sum = candidates
        .iter()
        .map(|candidate| {
            candidate.target_dimension
                * u64::try_from(candidate.multiplicity_in_hook_tensor_spinor).unwrap()
        })
        .sum();
    let target_dimension_sum_matches = target_dimension_sum == tensor_product_dimension;
    let level18_source_target_embedded_copy_count = candidates
        .iter()
        .map(|candidate| candidate.level18_embedded_source_copies)
        .sum();
    let mut unique_source_multiplicities = BTreeMap::<String, usize>::new();
    for candidate in &candidates {
        for source in &candidate.level18_source_channels {
            unique_source_multiplicities
                .entry(source.source_dynkin_label.clone())
                .and_modify(|multiplicity| *multiplicity = (*multiplicity).max(source.multiplicity))
                .or_insert(source.multiplicity);
        }
    }
    let level18_distinct_source_fixture_count = unique_source_multiplicities.len();
    let level18_source_fixture_copy_count = unique_source_multiplicities.values().sum();
    let required_source_copies = unique_source_multiplicities
        .iter()
        .flat_map(|(label, multiplicity)| {
            (1..=*multiplicity).map(move |copy| (label.as_str(), copy))
        })
        .collect::<BTreeSet<_>>();
    let level14_hodge_dual_analogue_fixture_count =
        crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
            .into_iter()
            .filter(|fixture| {
                required_source_copies.contains(&(fixture.dynkin_label, fixture.copy))
            })
            .count();
    let level18_source_fixtures_present = 0usize;
    let worklist = generated_worklist(&candidates);
    let cohomology = BoundedCohomology {
        field: "Q",
        complex: "Q^12 --d16--> Q^7 --d17--> 0 at the zero-momentum relaxed representation-symbol level",
        level16_dimension: 12,
        level16_to_level17_rank: level17_map.exact_rank,
        level16_kernel_dimension: level17_map.exact_nullity,
        level16_cohomology_lower_bound: 0,
        level16_cohomology_upper_bound: level17_map.exact_nullity,
        level16_incoming_image_known: false,
        level17_dimension: 7,
        forced_level17_to_level18_rank: 0,
        level17_kernel_dimension: 7,
        level17_image_dimension: level17_map.exact_rank,
        level17_bounded_cohomology_dimension: 7 - level17_map.exact_rank,
        level18_candidate_irreps: candidates.len(),
        level18_candidate_embedded_copies: level18_source_target_embedded_copy_count,
        level18_cohomology_computed: false,
        interpretation: "The committed d16 is surjective. Therefore any rational row B satisfying B d16 = 0 is zero, even before restricting B to rows produced by B5 intertwiners. The bounded hook cohomology is consequently zero.",
        boundary: "The level16 kernel has dimension five, but its cohomology is only bounded between zero and five because the incoming map is not part of this artifact. No level18 or momentum-dependent cohomology is claimed.",
    };
    let completeness = CompletenessFlags {
        committed_level17_matrix_reverified_exactly: level17_map.passed,
        hook_tensor_spinor_decomposition_complete: target_dimension_sum_matches
            && candidates.len() == 4
            && candidates
                .iter()
                .all(|candidate| candidate.multiplicity_in_hook_tensor_spinor == 1),
        level18_inventory_incidence_complete: candidates.iter().all(|candidate| {
            candidate.level18_embedded_source_copies
                == crate::eleven_dimensional_prepotential::spinor_level_multiplicity(
                    18,
                    &candidate.target_dynkin_label,
                )
        }),
        level14_hodge_dual_analogue_manifest_audited: level14_hodge_dual_analogue_fixture_count
            == 27,
        hodge_duality_kernel_lift_verified: false,
        relaxed_representation_level_composition_solved_exactly: level17_map.left_kernel_dimension
            == 0,
        bounded_zero_momentum_level17_cohomology_computed: cohomology
            .level17_bounded_cohomology_dimension
            == 0,
        abstract_target_couplings_constructed: false,
        embedded_level18_highest_weight_kernels_present: false,
        embedded_next_bianchi_compositions_computed: false,
        momentum_corrected_complex_computed: false,
        gauge_quotient_computed: false,
        polynomial_module_cohomology_computed: false,
    };
    let passed = level17_map.passed
        && target_dimension_sum_matches
        && candidates.len() == 4
        && level18_distinct_source_fixture_count == 16
        && level18_source_fixture_copy_count == 42
        && level14_hodge_dual_analogue_fixture_count == 27
        && level18_source_fixtures_present == 0
        && level18_source_target_embedded_copy_count == 77
        && candidates
            .iter()
            .all(|candidate| !candidate.nonzero_representation_level_bianchi_row_survives)
        && cohomology.level17_bounded_cohomology_dimension == 0
        && completeness.level18_inventory_incidence_complete;
    HookBianchiReport {
        schema_version: SCHEMA_VERSION,
        source_hook_dynkin_label: HOOK_LABEL,
        source_hook_dimension,
        spinor_dimension: 32,
        tensor_product_dimension,
        target_dimension_sum,
        target_dimension_sum_matches,
        target_candidate_count: candidates.len(),
        target_candidates: candidates,
        level18_distinct_source_fixture_count,
        level18_source_fixture_copy_count,
        level14_hodge_dual_analogue_fixture_count,
        level18_source_fixtures_present,
        level18_source_target_embedded_copy_count,
        level17_map,
        cohomology,
        generated_work_items: worklist.len(),
        completeness,
        passed,
        result: "No nonzero next-Bianchi row survives the exact relaxed zero-momentum representation-level composition gate.",
        boundary: "This excludes a nonzero nilpotent outgoing row on the committed seven-copy hook multiplicity space. It does not assert that embedded level18 kernels exist, construct target Clebsch-Gordan coefficients, include the superspace momentum term, quotient gauge transformations, or prove physical exactness.",
    }
}

pub fn build() -> HookBianchiArtifact {
    let report = verify();
    let worklist = generated_worklist(&report.target_candidates);
    HookBianchiArtifact {
        schema_version: "adynkra-11d-hook-bianchi-artifact-v1",
        title: "Bounded representation-level continuation of the 11D level-17 (11000) hook",
        report,
        worklist,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> HookBianchiReport {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create hook Bianchi data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create hook Bianchi validation directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create hook Bianchi artifact")),
        &artifact,
    )
    .expect("write hook Bianchi artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create hook Bianchi validation")),
        &artifact.report,
    )
    .expect("write hook Bianchi validation");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_level17_map_is_reverified_and_has_no_left_kernel() {
        let report = verify();
        assert!(report.level17_map.passed);
        assert_eq!(report.level17_map.exact_rank, 7);
        assert_eq!(report.level17_map.exact_nullity, 5);
        assert_eq!(report.level17_map.left_kernel_dimension, 0);
        assert_eq!(report.level17_map.committed_kernel_vectors_annihilated, 5);
    }

    #[test]
    fn hook_tensor_spinor_has_four_complete_b5_targets() {
        let report = verify();
        let labels = report
            .target_candidates
            .iter()
            .map(|candidate| candidate.target_dynkin_label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["01001", "10001", "11001", "20001"]);
        assert!(report.target_dimension_sum_matches);
        assert!(
            report
                .target_candidates
                .iter()
                .all(|candidate| candidate.multiplicity_in_hook_tensor_spinor == 1)
        );
    }

    #[test]
    fn level18_incidence_and_worklist_are_finite_and_exact() {
        let artifact = build();
        let counts = artifact
            .report
            .target_candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.target_dynkin_label.as_str(),
                    candidate.level18_embedded_source_copies,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            counts,
            vec![("01001", 18), ("10001", 8), ("11001", 38), ("20001", 13)]
        );
        assert_eq!(artifact.report.level18_distinct_source_fixture_count, 16);
        assert_eq!(artifact.report.level18_source_fixture_copy_count, 42);
        assert_eq!(
            artifact.report.level14_hodge_dual_analogue_fixture_count,
            27
        );
        assert_eq!(artifact.report.level18_source_fixtures_present, 0);
        assert_eq!(
            artifact.report.level18_source_target_embedded_copy_count,
            77
        );
        assert_eq!(
            artifact.report.generated_work_items,
            artifact.worklist.len()
        );
    }

    #[test]
    fn bounded_hook_cohomology_is_zero_but_completeness_stays_false() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.cohomology.level17_bounded_cohomology_dimension, 0);
        assert_eq!(report.cohomology.level16_cohomology_upper_bound, 5);
        assert!(!report.cohomology.level16_incoming_image_known);
        assert!(!report.completeness.abstract_target_couplings_constructed);
        assert!(
            !report
                .completeness
                .embedded_level18_highest_weight_kernels_present
        );
        assert!(!report.completeness.hodge_duality_kernel_lift_verified);
        assert!(
            !report
                .completeness
                .embedded_next_bianchi_compositions_computed
        );
        assert!(!report.completeness.gauge_quotient_computed);
        assert!(!report.completeness.polynomial_module_cohomology_computed);
    }
}
