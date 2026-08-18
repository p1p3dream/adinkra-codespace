//! Aggregated, fail-closed status for the top-down 11D program.
//!
//! This module separates completed exact gates from the open prepotential,
//! superspace-cohomology, and field-equation claims.  Representation incidence
//! is used only to generate a sparse physical-seed work list.  It is never
//! promoted to a cohomology result without explicit differentials.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct SeedOccurrence {
    pub role: &'static str,
    pub dynkin_label: &'static str,
    pub dimension: u64,
    pub levels_and_multiplicities: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SuperspaceCohomologyGate {
    pub seed_occurrences: Vec<SeedOccurrence>,
    pub all_seeds_present: bool,
    pub exact_differentials_constructed: bool,
    pub gauge_image_quotiented: bool,
    pub bianchi_kernel_computed: bool,
    pub pure_spinor_oracle_compared: bool,
    pub physical_supermultiplet_isolated: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct LinearizedEquationGate {
    pub target_free_equations_constructed: bool,
    pub lorentzian_majorana_real_form_constructed: bool,
    pub physical_light_cone_susy_maps_constructed: bool,
    pub physical_light_cone_susy_closure_checked: bool,
    pub source_to_target_superfield_map_constructed: bool,
    pub gauge_invariant_superfield_operator_constructed: bool,
    pub superfield_supersymmetry_closure_checked: bool,
    pub component_einstein_rarita_schwinger_three_form_match: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct LeadingX2ArtifactGate {
    pub artifact_path: &'static str,
    pub artifact_present: bool,
    pub artifact_sha256: Option<String>,
    pub schema_version: Option<String>,
    pub report_passed: bool,
    pub leading_symbol_f0_a_g_established_by_job: bool,
    pub exact_cross_operator_column_ranks_established: bool,
    pub joint_six_channel_exact_rank: Option<usize>,
    pub joint_six_channel_exact_nullity: Option<usize>,
    pub joint_source_stream_kernel_relations_checked: usize,
    pub joint_source_stream_kernel_residual_entries: usize,
    pub physical_operator_combination_selected: bool,
    pub full_f_a_g_p_established: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstMomentumFunctionalScreenArtifact {
    pub artifact_path: String,
    pub artifact_sha256: Option<String>,
    pub gauge_form_degree: usize,
    pub parameter_components: usize,
    pub parameter_projection_is_complete: bool,
    pub exact_functional_rank: usize,
    pub exact_functional_nullity: usize,
    pub leading_projection_rank: usize,
    pub nonzero_leading_extension_excluded: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstMomentumFunctionalScreenGate {
    pub screens: Vec<FirstMomentumFunctionalScreenArtifact>,
    pub all_expected_screens_present: bool,
    pub expected_parameter_counts_matched: bool,
    pub all_parameter_projections_complete: bool,
    pub old_leading_extension_excluded_in_all_three_channels: bool,
    pub generic_polynomial_f_a_g_p_established: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstMomentumPhysicalFxArtifactGate {
    pub artifact_path: &'static str,
    pub artifact_present: bool,
    pub artifact_sha256: Option<String>,
    pub artifact_hash_matches: bool,
    pub schema_version: Option<String>,
    pub promotion_manifest_path: &'static str,
    pub promotion_manifest_present: bool,
    pub promotion_manifest_sha256: Option<String>,
    pub promotion_manifest_schema_version: Option<String>,
    pub promotion_manifest_hash_matches: bool,
    pub promotion_manifest_checkpoint_hashes: usize,
    pub promotion_manifest_complete_key_set: bool,
    pub promotion_manifest_passed: bool,
    pub promotion_manifest_integrity_passed: bool,
    pub curvature_artifact_sha256: Option<String>,
    pub fx_input_snapshot_path: &'static str,
    pub fx_input_snapshot_artifact_sha256: Option<String>,
    pub fx_input_snapshot_schema_version: Option<String>,
    pub fx_input_snapshot_hash_matches: bool,
    pub current_physical_curvature_artifact_sha256: Option<String>,
    pub current_physical_curvature_schema_version: Option<String>,
    pub current_physical_curvature_envelope_hash_matches: bool,
    pub curvature_artifact_hash_matches: bool,
    pub channel_degrees: Vec<usize>,
    pub every_channel_has_56_operator_columns: bool,
    pub every_channel_uses_declared_parameter_slice: bool,
    pub every_channel_uses_declared_target_slice: bool,
    pub emitted_target_terms: u64,
    pub global_x2_rank: usize,
    pub global_x2_nullity: usize,
    pub global_x5_rank: usize,
    pub global_x5_nullity: usize,
    pub global_joint_rank: usize,
    pub global_joint_nullity: usize,
    pub global_joint_rank_exact_by_dimension_saturation: bool,
    pub surviving_leading_projection_rank_upper_bound: usize,
    pub all_six_channels_composed_on_declared_slice: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub mutation_detected: bool,
    pub partial_fx_only: bool,
    pub recorded_49_dimensional_ansatz_excluded_on_declared_slice: bool,
    pub full_f_a_g_p_established: bool,
    pub report_integrity_passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct KAndFagArtifactGate {
    pub artifact_path: &'static str,
    pub artifact_present: bool,
    pub artifact_sha256: Option<String>,
    pub artifact_hash_matches: bool,
    pub schema_version: Option<String>,
    pub current_physical_curvature_envelope_sha256: Option<String>,
    pub current_physical_curvature_envelope_hash_matches: bool,
    pub fx_input_snapshot_sha256: Option<String>,
    pub fx_input_snapshot_hash_matches: bool,
    pub report_passed: bool,
    pub generic_k_solved: bool,
    pub physical_fag_established: bool,
    pub report_integrity_passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug)]
struct FxPromotionManifestStatus {
    present: bool,
    sha256: Option<String>,
    schema_version: Option<String>,
    hash_matches: bool,
    checkpoint_hashes: usize,
    complete_key_set: bool,
    passed: bool,
    integrity_passed: bool,
}

fn fx_promotion_manifest_status(path: &str, expected_sha256: &str) -> FxPromotionManifestStatus {
    const SCHEMA: &str = "adynkra-11d-fx-shared-promotion-report-v1";
    let Ok(bytes) = fs::read(path) else {
        return FxPromotionManifestStatus {
            present: false,
            sha256: None,
            schema_version: None,
            hash_matches: false,
            checkpoint_hashes: 0,
            complete_key_set: false,
            passed: false,
            integrity_passed: false,
        };
    };
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return FxPromotionManifestStatus {
            present: true,
            sha256: Some(sha256),
            schema_version: None,
            hash_matches: false,
            checkpoint_hashes: 0,
            complete_key_set: false,
            passed: false,
            integrity_passed: false,
        };
    };
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let hash_matches = sha256 == expected_sha256;
    let checkpoint_hashes = value
        .get("candidate_sha256")
        .and_then(serde_json::Value::as_object)
        .map_or(0, serde_json::Map::len);
    let complete_key_set = value
        .get("candidate_sha256")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|hashes| {
            hashes.len() == 336
                && (0..6).all(|degree| {
                    (0..56).all(|operator| {
                        let key = format!("form-{degree}/operator-{operator:02}");
                        hashes
                            .get(&key)
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|digest| {
                                digest.len() == 64
                                    && digest.bytes().all(|byte| {
                                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                                    })
                            })
                    })
                })
        });
    let passed = value
        .get("passed")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let accounted_checkpoints = value
        .get("copied_missing")
        .and_then(serde_json::Value::as_u64)
        .zip(
            value
                .get("verified_existing")
                .and_then(serde_json::Value::as_u64),
        )
        .is_some_and(|(copied, verified)| copied + verified == 336);
    let no_partial_replacements = value
        .get("replaced_partial")
        .and_then(serde_json::Value::as_u64)
        == Some(0);
    let integrity_passed = schema_version.as_deref() == Some(SCHEMA)
        && hash_matches
        && complete_key_set
        && accounted_checkpoints
        && no_partial_replacements
        && passed;
    FxPromotionManifestStatus {
        present: true,
        sha256: Some(sha256),
        schema_version,
        hash_matches,
        checkpoint_hashes,
        complete_key_set,
        passed,
        integrity_passed,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ElevenDimensionalTopDownReport {
    pub schema_version: &'static str,
    pub free_complex: crate::eleven_dimensional_free_complex::ElevenDimensionalFreeComplexReport,
    pub target_stream: crate::eleven_dimensional_target_stream::ElevenDimensionalTargetStreamReport,
    pub source_fixed_curvature:
        crate::eleven_dimensional_source_fixed_curvature::SourceFixedCurvatureReport,
    pub abstract_clifford_join:
        crate::eleven_dimensional_abstract_clifford_join::AbstractCliffordJoinReport,
    pub b5_majorana_target_join:
        crate::eleven_dimensional_b5_majorana_target_join::B5MajoranaTargetJoinReport,
    pub leading_x2_gauge: LeadingX2ArtifactGate,
    pub first_momentum_functional_screens: FirstMomentumFunctionalScreenGate,
    pub first_momentum_physical_fx: FirstMomentumPhysicalFxArtifactGate,
    pub physical_curvature:
        crate::eleven_dimensional_physical_curvature::PhysicalCurvatureOperatorReport,
    pub physical_adapter_audit:
        crate::eleven_dimensional_physical_adapter_audit::PhysicalAdapterAuditReport,
    pub j1_lorentz_residual:
        crate::eleven_dimensional_j1_lorentz_residual::JOneLorentzResidualReport,
    pub direct_local_lorentz:
        crate::eleven_dimensional_direct_local_lorentz::LocalLorentzDiagnosticReport,
    pub lorentz_holonomy_compensator_audit: crate::eleven_dimensional_lorentz_holonomy_compensator_audit::LorentzHolonomyCompensatorAuditReport,
    pub k_fag_polynomial_harness:
        crate::eleven_dimensional_k_fag_solver::KAndFagHarnessReport,
    pub k_fag_artifact: KAndFagArtifactGate,
    pub level18_embedded: crate::eleven_dimensional_level18_embedded::Level18EmbeddedReport,
    pub level18_target_quotient:
        crate::eleven_dimensional_level18_target_quotient::Level18TargetQuotientReport,
    pub covariant_cohomology_entry:
        crate::eleven_dimensional_covariant_cohomology_gate::ElevenDimensionalCovariantCohomologyGateReport,
    pub lowest_spinorial_differential:
        crate::eleven_dimensional_spinorial_differential::ElevenDimensionalSpinorialDifferentialReport,
    pub relaxed_spinorial_cohomology:
        crate::eleven_dimensional_relaxed_spinorial_cohomology::RelaxedSpinorialCohomologyReport,
    pub target_equation_complex:
        crate::eleven_dimensional_target_equation_complex::TargetEquationComplexReport,
    pub first_superspace_jet:
        crate::eleven_dimensional_first_superspace_jet::FirstSuperspaceJetReport,
    pub prepotential_gate:
        crate::eleven_dimensional_prepotential_gate::ElevenDimensionalPrepotentialGateReport,
    pub hook_bianchi: crate::eleven_dimensional_hook_bianchi::HookBianchiReport,
    pub level18_momentum: crate::eleven_dimensional_level18_momentum::Level18MomentumReport,
    pub majorana: crate::eleven_dimensional_majorana::ElevenDimensionalMajoranaReport,
    pub linear_susy: crate::eleven_dimensional_linear_susy::ElevenDimensionalLinearSusyReport,
    pub superspace_cohomology: SuperspaceCohomologyGate,
    pub linearized_equation: LinearizedEquationGate,
    pub bounded_gates_passed: bool,
    pub full_program_complete: bool,
    pub next_exact_steps: Vec<&'static str>,
    pub boundary: &'static str,
}

fn first_momentum_functional_screen_gate() -> FirstMomentumFunctionalScreenGate {
    let screens = [1_usize, 2, 5]
        .into_iter()
        .map(|degree| {
            let path =
                format!("results/adynkra_11d_first_momentum_gauge_functional_p{degree}.json");
            let Ok(bytes) = fs::read(&path) else {
                return FirstMomentumFunctionalScreenArtifact {
                    artifact_path: path,
                    artifact_sha256: None,
                    gauge_form_degree: degree,
                    parameter_components: 0,
                    parameter_projection_is_complete: false,
                    exact_functional_rank: 0,
                    exact_functional_nullity: 0,
                    leading_projection_rank: 0,
                    nonzero_leading_extension_excluded: false,
                    passed: false,
                };
            };
            let digest = format!("{:x}", Sha256::digest(&bytes));
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
            let integer = |name: &str| {
                value
                    .get(name)
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|number| usize::try_from(number).ok())
                    .unwrap_or(0)
            };
            let boolean = |name: &str| {
                value
                    .get(name)
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            };
            FirstMomentumFunctionalScreenArtifact {
                artifact_path: path,
                artifact_sha256: Some(digest),
                gauge_form_degree: integer("gauge_form_degree"),
                parameter_components: integer("parameter_components"),
                parameter_projection_is_complete: boolean("parameter_projection_is_complete"),
                exact_functional_rank: integer("exact_functional_rank"),
                exact_functional_nullity: integer("exact_functional_nullity"),
                leading_projection_rank: integer("functional_kernel_leading_projection_rank"),
                nonzero_leading_extension_excluded: boolean(
                    "nonzero_leading_extension_excluded_by_functionals",
                ),
                passed: boolean("passed") && boolean("functional_kernel_residuals_exactly_zero"),
            }
        })
        .collect::<Vec<_>>();
    let all_expected_screens_present = screens
        .iter()
        .map(|screen| screen.gauge_form_degree)
        .eq([1, 2, 5]);
    let expected_parameter_counts_matched = screens
        .iter()
        .map(|screen| screen.parameter_components)
        .eq([11, 55, 462]);
    let all_parameter_projections_complete = screens
        .iter()
        .all(|screen| screen.parameter_projection_is_complete);
    let old_leading_extension_excluded_in_all_three_channels = screens.iter().all(|screen| {
        screen.passed
            && screen.nonzero_leading_extension_excluded
            && screen.leading_projection_rank == 0
    });
    let passed = all_expected_screens_present
        && expected_parameter_counts_matched
        && all_parameter_projections_complete
        && old_leading_extension_excluded_in_all_three_channels;
    FirstMomentumFunctionalScreenGate {
        screens,
        all_expected_screens_present,
        expected_parameter_counts_matched,
        all_parameter_projections_complete,
        old_leading_extension_excluded_in_all_three_channels,
        generic_polynomial_f_a_g_p_established: false,
        passed,
        boundary: "These complete exact parameter-component screens exclude a first-momentum extension of the old 12-leading plus 44-correction ansatz in gauge degrees 1, 2, and 5. They are negative controls, not a generic-polynomial test of the source-fixed physical F A G_p composition.",
    }
}

fn first_momentum_physical_fx_artifact_gate() -> FirstMomentumPhysicalFxArtifactGate {
    const PATH: &str = "results/adynkra_11d_first_momentum_physical_fx_functional.json";
    const ARTIFACT_SHA256: &str =
        "5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9";
    const PHYSICAL_PATH: &str = "results/adynkra_11d_physical_curvature_validation.json";
    const PHYSICAL_SHA256: &str =
        "3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13";
    const FX_INPUT_PATH: &str = "results/adynkra_11d_physical_curvature_fx_input_v10.json";
    const FX_INPUT_SHA256: &str =
        "c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f";
    const PHYSICAL_SCHEMA: &str = "adynkra-11d-physical-curvature-operator-v10";
    const PROMOTION_PATH: &str =
        "results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json";
    const PROMOTION_SHA256: &str =
        "98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4";
    const SCHEMA: &str = "adynkra-11d-first-momentum-partial-fx-functional-v3";

    let promotion = fx_promotion_manifest_status(PROMOTION_PATH, PROMOTION_SHA256);
    let current_physical_bytes = fs::read(PHYSICAL_PATH).ok();
    let current_physical_curvature_artifact_sha256 = current_physical_bytes
        .as_deref()
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    let current_physical_curvature_schema_version = current_physical_bytes
        .as_deref()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok())
        .and_then(|value| value.get("schema_version")?.as_str().map(str::to_owned));
    let current_physical_curvature_envelope_hash_matches =
        current_physical_curvature_artifact_sha256.as_deref() == Some(PHYSICAL_SHA256)
            && current_physical_curvature_schema_version.as_deref() == Some(PHYSICAL_SCHEMA);
    let fx_input_bytes = fs::read(FX_INPUT_PATH).ok();
    let fx_input_snapshot_artifact_sha256 = fx_input_bytes
        .as_deref()
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    let fx_input_snapshot_schema_version = fx_input_bytes
        .as_deref()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok())
        .and_then(|value| value.get("schema_version")?.as_str().map(str::to_owned));
    let fx_input_snapshot_hash_matches = fx_input_snapshot_artifact_sha256.as_deref()
        == Some(FX_INPUT_SHA256)
        && fx_input_snapshot_schema_version.as_deref() == Some(PHYSICAL_SCHEMA);
    let Ok(bytes) = fs::read(PATH) else {
        return FirstMomentumPhysicalFxArtifactGate {
            artifact_path: PATH,
            artifact_present: false,
            artifact_sha256: None,
            artifact_hash_matches: false,
            schema_version: None,
            promotion_manifest_path: PROMOTION_PATH,
            promotion_manifest_present: promotion.present,
            promotion_manifest_sha256: promotion.sha256,
            promotion_manifest_schema_version: promotion.schema_version,
            promotion_manifest_hash_matches: promotion.hash_matches,
            promotion_manifest_checkpoint_hashes: promotion.checkpoint_hashes,
            promotion_manifest_complete_key_set: promotion.complete_key_set,
            promotion_manifest_passed: promotion.passed,
            promotion_manifest_integrity_passed: promotion.integrity_passed,
            curvature_artifact_sha256: None,
            fx_input_snapshot_path: FX_INPUT_PATH,
            fx_input_snapshot_artifact_sha256,
            fx_input_snapshot_schema_version,
            fx_input_snapshot_hash_matches,
            current_physical_curvature_artifact_sha256,
            current_physical_curvature_schema_version,
            current_physical_curvature_envelope_hash_matches,
            curvature_artifact_hash_matches: false,
            channel_degrees: Vec::new(),
            every_channel_has_56_operator_columns: false,
            every_channel_uses_declared_parameter_slice: false,
            every_channel_uses_declared_target_slice: false,
            emitted_target_terms: 0,
            global_x2_rank: 0,
            global_x2_nullity: 49,
            global_x5_rank: 0,
            global_x5_nullity: 49,
            global_joint_rank: 0,
            global_joint_nullity: 49,
            global_joint_rank_exact_by_dimension_saturation: false,
            surviving_leading_projection_rank_upper_bound: 5,
            all_six_channels_composed_on_declared_slice: false,
            full_parameter_projection_complete: false,
            full_target_projection_complete: false,
            mutation_detected: false,
            partial_fx_only: true,
            recorded_49_dimensional_ansatz_excluded_on_declared_slice: false,
            full_f_a_g_p_established: false,
            report_integrity_passed: false,
            boundary: "The standalone first-momentum physical F_X artifact is absent. The aggregate fails closed and never recomputes the 336 exact operator jobs.",
        };
    };
    let artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let artifact_hash_matches = artifact_sha256 == ARTIFACT_SHA256;
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return FirstMomentumPhysicalFxArtifactGate {
            artifact_path: PATH,
            artifact_present: true,
            artifact_sha256: Some(artifact_sha256),
            artifact_hash_matches,
            schema_version: None,
            promotion_manifest_path: PROMOTION_PATH,
            promotion_manifest_present: promotion.present,
            promotion_manifest_sha256: promotion.sha256,
            promotion_manifest_schema_version: promotion.schema_version,
            promotion_manifest_hash_matches: promotion.hash_matches,
            promotion_manifest_checkpoint_hashes: promotion.checkpoint_hashes,
            promotion_manifest_complete_key_set: promotion.complete_key_set,
            promotion_manifest_passed: promotion.passed,
            promotion_manifest_integrity_passed: promotion.integrity_passed,
            curvature_artifact_sha256: None,
            fx_input_snapshot_path: FX_INPUT_PATH,
            fx_input_snapshot_artifact_sha256,
            fx_input_snapshot_schema_version,
            fx_input_snapshot_hash_matches,
            current_physical_curvature_artifact_sha256,
            current_physical_curvature_schema_version,
            current_physical_curvature_envelope_hash_matches,
            curvature_artifact_hash_matches: false,
            channel_degrees: Vec::new(),
            every_channel_has_56_operator_columns: false,
            every_channel_uses_declared_parameter_slice: false,
            every_channel_uses_declared_target_slice: false,
            emitted_target_terms: 0,
            global_x2_rank: 0,
            global_x2_nullity: 49,
            global_x5_rank: 0,
            global_x5_nullity: 49,
            global_joint_rank: 0,
            global_joint_nullity: 49,
            global_joint_rank_exact_by_dimension_saturation: false,
            surviving_leading_projection_rank_upper_bound: 5,
            all_six_channels_composed_on_declared_slice: false,
            full_parameter_projection_complete: false,
            full_target_projection_complete: false,
            mutation_detected: false,
            partial_fx_only: true,
            recorded_49_dimensional_ansatz_excluded_on_declared_slice: false,
            full_f_a_g_p_established: false,
            report_integrity_passed: false,
            boundary: "The standalone first-momentum physical F_X artifact is not valid JSON. The aggregate fails closed and never recomputes the 336 exact operator jobs.",
        };
    };

    let integer = |name: &str| {
        value
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .unwrap_or(0)
    };
    let boolean = |name: &str| {
        value
            .get(name)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let curvature_artifact_sha256 = value
        .get("curvature_artifact_sha256")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let curvature_artifact_hash_matches = curvature_artifact_sha256.as_deref()
        == Some(FX_INPUT_SHA256)
        && fx_input_snapshot_hash_matches;
    let channels = value
        .get("channel_reports")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let channel_integer = |channel: &serde_json::Value, name: &str| {
        channel
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .unwrap_or(0)
    };
    let channel_degrees = channels
        .iter()
        .map(|channel| channel_integer(channel, "gauge_form_degree"))
        .collect::<Vec<_>>();
    let every_channel_has_56_operator_columns = channels.len() == 6
        && channels
            .iter()
            .all(|channel| channel_integer(channel, "operator_columns_composed") == 56);
    let every_channel_uses_declared_parameter_slice = channels.len() == 6
        && channels.iter().all(|channel| {
            channel
                .get("parameter_components_selected")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|selected| selected.len() == 1 && selected[0].as_u64() == Some(0))
        });
    let every_channel_uses_declared_target_slice = channels.len() == 6
        && channels.iter().all(|channel| {
            channel
                .get("target_basis_ordinals_selected")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|selected| selected.len() == 1 && selected[0].as_u64() == Some(319))
        });
    let expected_parameter_components = [1_usize, 11, 55, 165, 330, 462];
    let channel_shapes_match = channel_degrees == (0..6).collect::<Vec<_>>()
        && channels
            .iter()
            .zip(expected_parameter_components)
            .all(|(channel, expected)| {
                channel_integer(channel, "parameter_components_total") == expected
                    && channel_integer(channel, "joint_functional_rank_lower_bound")
                        + channel_integer(channel, "joint_functional_nullity_upper_bound")
                        == 49
            });
    let emitted_target_terms = channels.iter().fold(0_u64, |sum, channel| {
        sum.saturating_add(
            channel
                .get("emitted_target_terms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
    });
    let global_x2_rank = integer("global_x2_rank_lower_bound");
    let global_x2_nullity = integer("global_x2_nullity_upper_bound");
    let global_x5_rank = integer("global_x5_rank_lower_bound");
    let global_x5_nullity = integer("global_x5_nullity_upper_bound");
    let global_joint_rank = integer("global_joint_rank_lower_bound");
    let global_joint_nullity = integer("global_joint_nullity_upper_bound");
    let global_joint_rank_exact_by_dimension_saturation =
        boolean("global_joint_rank_exact_by_dimension_saturation");
    let surviving_leading_projection_rank_upper_bound =
        integer("surviving_leading_projection_rank_upper_bound");
    let all_six_channels_composed_on_declared_slice =
        boolean("all_six_channels_composed_on_declared_slice");
    let full_parameter_projection_complete = boolean("full_parameter_projection_complete");
    let full_target_projection_complete = boolean("full_target_projection_complete");
    let mutation_detected = boolean("mutation_detected");
    let partial_fx_only = boolean("partial_fx_only");
    let full_f_a_g_p_established = boolean("full_f_a_g_p_established");
    let recorded_49_dimensional_ansatz_excluded_on_declared_slice = global_x2_rank == 49
        && global_x2_nullity == 0
        && global_x5_rank == 49
        && global_x5_nullity == 0
        && global_joint_rank == 49
        && global_joint_nullity == 0
        && global_joint_rank_exact_by_dimension_saturation
        && surviving_leading_projection_rank_upper_bound == 0;
    let report_integrity_passed = artifact_hash_matches
        && schema_version.as_deref() == Some(SCHEMA)
        && promotion.integrity_passed
        && current_physical_curvature_envelope_hash_matches
        && curvature_artifact_hash_matches
        && channel_shapes_match
        && every_channel_has_56_operator_columns
        && every_channel_uses_declared_parameter_slice
        && every_channel_uses_declared_target_slice
        && all_six_channels_composed_on_declared_slice
        && !full_parameter_projection_complete
        && !full_target_projection_complete
        && mutation_detected
        && partial_fx_only
        && recorded_49_dimensional_ansatz_excluded_on_declared_slice
        && !full_f_a_g_p_established;

    FirstMomentumPhysicalFxArtifactGate {
        artifact_path: PATH,
        artifact_present: true,
        artifact_sha256: Some(artifact_sha256),
        artifact_hash_matches,
        schema_version,
        promotion_manifest_path: PROMOTION_PATH,
        promotion_manifest_present: promotion.present,
        promotion_manifest_sha256: promotion.sha256,
        promotion_manifest_schema_version: promotion.schema_version,
        promotion_manifest_hash_matches: promotion.hash_matches,
        promotion_manifest_checkpoint_hashes: promotion.checkpoint_hashes,
        promotion_manifest_complete_key_set: promotion.complete_key_set,
        promotion_manifest_passed: promotion.passed,
        promotion_manifest_integrity_passed: promotion.integrity_passed,
        curvature_artifact_sha256,
        fx_input_snapshot_path: FX_INPUT_PATH,
        fx_input_snapshot_artifact_sha256,
        fx_input_snapshot_schema_version,
        fx_input_snapshot_hash_matches,
        current_physical_curvature_artifact_sha256,
        current_physical_curvature_schema_version,
        current_physical_curvature_envelope_hash_matches,
        curvature_artifact_hash_matches,
        channel_degrees,
        every_channel_has_56_operator_columns,
        every_channel_uses_declared_parameter_slice,
        every_channel_uses_declared_target_slice,
        emitted_target_terms,
        global_x2_rank,
        global_x2_nullity,
        global_x5_rank,
        global_x5_nullity,
        global_joint_rank,
        global_joint_nullity,
        global_joint_rank_exact_by_dimension_saturation,
        surviving_leading_projection_rank_upper_bound,
        all_six_channels_composed_on_declared_slice,
        full_parameter_projection_complete,
        full_target_projection_complete,
        mutation_detected,
        partial_fx_only,
        recorded_49_dimensional_ansatz_excluded_on_declared_slice,
        full_f_a_g_p_established,
        report_integrity_passed,
        boundary: "This SHA-256-addressed artifact composes partial F_X=(X_[2],X_[5]) with all 56 recorded operators in all six gauge channels on parameter component zero and target highest-weight state 319. Its c308 input is validated against the immutable physical-curvature v10 snapshot, while the enriched current physical-curvature envelope is validated independently. Exact rank 49 and nullity zero exclude the recorded five-leading-kernel plus 44-correction coefficient space. This one declared slice is sufficient for that bounded exclusion, but it is not a complete parameter or target projection, omits J and W, and does not establish full F A G_p.",
    }
}

fn k_fag_artifact_gate() -> KAndFagArtifactGate {
    const PATH: &str = "results/adynkra_11d_k_fag_polynomial_harness.json";
    const SHA256: &str = "11ec33c36d9536e17e617839cc8dbabc885b9d30bf13ff05a4d0dc5e6b9fe562";
    const SCHEMA: &str = "adynkra-11d-k-fag-polynomial-harness-v1";
    const PHYSICAL_SHA256: &str =
        "3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13";
    const FX_INPUT_SHA256: &str =
        "c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f";

    let Ok(bytes) = fs::read(PATH) else {
        return KAndFagArtifactGate {
            artifact_path: PATH,
            artifact_present: false,
            artifact_sha256: None,
            artifact_hash_matches: false,
            schema_version: None,
            current_physical_curvature_envelope_sha256: None,
            current_physical_curvature_envelope_hash_matches: false,
            fx_input_snapshot_sha256: None,
            fx_input_snapshot_hash_matches: false,
            report_passed: false,
            generic_k_solved: false,
            physical_fag_established: false,
            report_integrity_passed: false,
            boundary: "The K/FAG harness artifact is absent, so its provenance gate fails closed.",
        };
    };
    let artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let artifact_hash_matches = artifact_sha256 == SHA256;
    let value = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_default();
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let current_physical_curvature_envelope_sha256 = value
        .get("current_physical_curvature_envelope_sha256")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let current_physical_curvature_envelope_hash_matches =
        current_physical_curvature_envelope_sha256.as_deref() == Some(PHYSICAL_SHA256)
            && value
                .get("current_physical_curvature_envelope_provenance_validated")
                .and_then(serde_json::Value::as_bool)
                == Some(true);
    let fx_control = value.get("final_physical_fx_bounded_negative_control");
    let fx_input_snapshot_sha256 = fx_control
        .and_then(|control| control.get("fx_input_snapshot"))
        .and_then(|snapshot| snapshot.get("artifact_sha256"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let fx_input_snapshot_hash_matches = fx_input_snapshot_sha256.as_deref()
        == Some(FX_INPUT_SHA256)
        && fx_control
            .and_then(|control| control.get("fx_input_snapshot"))
            .and_then(|snapshot| snapshot.get("strict_contract_validated"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        && fx_control
            .and_then(|control| control.get("fx_report_curvature_input_sha256"))
            .and_then(serde_json::Value::as_str)
            == Some(FX_INPUT_SHA256);
    let report_passed = value
        .get("passed")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let generic_k_solved = value
        .get("generic_k_solved")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let physical_fag_established = value
        .get("physical_fag_established")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let report_integrity_passed = artifact_hash_matches
        && schema_version.as_deref() == Some(SCHEMA)
        && current_physical_curvature_envelope_hash_matches
        && fx_input_snapshot_hash_matches
        && report_passed
        && !generic_k_solved
        && !physical_fag_established;
    KAndFagArtifactGate {
        artifact_path: PATH,
        artifact_present: true,
        artifact_sha256: Some(artifact_sha256),
        artifact_hash_matches,
        schema_version,
        current_physical_curvature_envelope_sha256,
        current_physical_curvature_envelope_hash_matches,
        fx_input_snapshot_sha256,
        fx_input_snapshot_hash_matches,
        report_passed,
        generic_k_solved,
        physical_fag_established,
        report_integrity_passed,
        boundary: "This gate pins the current K/FAG harness artifact, its enriched physical-curvature envelope, and the distinct immutable c308 F_X input snapshot. Passing it preserves generic K and complete physical F A G_p as false.",
    }
}

fn leading_x2_artifact_gate() -> LeadingX2ArtifactGate {
    const PATH: &str = "results/adynkra_11d_leading_x2_gauge_validation.json";
    let Ok(bytes) = fs::read(PATH) else {
        return LeadingX2ArtifactGate {
            artifact_path: PATH,
            artifact_present: false,
            artifact_sha256: None,
            schema_version: None,
            report_passed: false,
            leading_symbol_f0_a_g_established_by_job: false,
            exact_cross_operator_column_ranks_established: false,
            joint_six_channel_exact_rank: None,
            joint_six_channel_exact_nullity: None,
            joint_source_stream_kernel_relations_checked: 0,
            joint_source_stream_kernel_residual_entries: 0,
            physical_operator_combination_selected: false,
            full_f_a_g_p_established: false,
            boundary: "The standalone leading X_[2] artifact is absent. The aggregate never recomputes this expensive exact gate.",
        };
    };
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => {
            return LeadingX2ArtifactGate {
                artifact_path: PATH,
                artifact_present: true,
                artifact_sha256: Some(digest),
                schema_version: None,
                report_passed: false,
                leading_symbol_f0_a_g_established_by_job: false,
                exact_cross_operator_column_ranks_established: false,
                joint_six_channel_exact_rank: None,
                joint_six_channel_exact_nullity: None,
                joint_source_stream_kernel_relations_checked: 0,
                joint_source_stream_kernel_residual_entries: 0,
                physical_operator_combination_selected: false,
                full_f_a_g_p_established: false,
                boundary: "The standalone leading X_[2] artifact is not valid JSON. The aggregate fails closed and never recomputes this expensive exact gate.",
            };
        }
    };
    LeadingX2ArtifactGate {
        artifact_path: PATH,
        artifact_present: true,
        artifact_sha256: Some(digest),
        schema_version: value
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        report_passed: value
            .get("passed")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        leading_symbol_f0_a_g_established_by_job: value
            .get("leading_symbol_f0_a_g_established_by_job")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        exact_cross_operator_column_ranks_established: value
            .get("exact_cross_operator_column_ranks_established")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        joint_six_channel_exact_rank: value
            .get("joint_six_channel_exact_rank")
            .and_then(serde_json::Value::as_u64)
            .and_then(|number| usize::try_from(number).ok()),
        joint_six_channel_exact_nullity: value
            .get("joint_six_channel_exact_nullity")
            .and_then(serde_json::Value::as_u64)
            .and_then(|number| usize::try_from(number).ok()),
        joint_source_stream_kernel_relations_checked: value
            .get("joint_source_stream_kernel_relations_checked")
            .and_then(serde_json::Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .unwrap_or(0),
        joint_source_stream_kernel_residual_entries: value
            .get("joint_source_stream_kernel_residual_entries")
            .and_then(serde_json::Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .unwrap_or(0),
        physical_operator_combination_selected: value
            .get("physical_operator_combination_selected")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        full_f_a_g_p_established: value
            .get("full_f_a_g_p_established")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        boundary: "This is a SHA-256-addressed summary of the standalone exact leading X_[2] artifact. The aggregate does not recompute it. The six-channel direct sum has an exact joint rank and kernel, while individual channel ranks can remain lower bounds. This does not select the physical operator or establish the momentum branch or full F A G_p identity.",
    }
}

fn seed(role: &'static str, dynkin_label: &'static str) -> SeedOccurrence {
    let levels_and_multiplicities = (0..=32)
        .filter_map(|level| {
            let multiplicity = crate::eleven_dimensional_prepotential::spinor_level_multiplicity(
                level,
                dynkin_label,
            );
            (multiplicity > 0).then_some((level, multiplicity))
        })
        .collect();
    SeedOccurrence {
        role,
        dynkin_label,
        dimension: crate::eleven_dimensional_prepotential::b5_dimension(dynkin_label),
        levels_and_multiplicities,
    }
}

pub fn build() -> ElevenDimensionalTopDownReport {
    let free = crate::eleven_dimensional_free_complex::build().report;
    let target_stream = crate::eleven_dimensional_target_stream::verify();
    let source_fixed_curvature = crate::eleven_dimensional_source_fixed_curvature::verify();
    let abstract_clifford_join = crate::eleven_dimensional_abstract_clifford_join::verify();
    let b5_majorana_target_join = crate::eleven_dimensional_b5_majorana_target_join::verify();
    let leading_x2_gauge = leading_x2_artifact_gate();
    let first_momentum_functional_screens = first_momentum_functional_screen_gate();
    let first_momentum_physical_fx = first_momentum_physical_fx_artifact_gate();
    let physical_curvature = crate::eleven_dimensional_physical_curvature::verify();
    let physical_adapter_audit = crate::eleven_dimensional_physical_adapter_audit::verify();
    let j1_lorentz_residual = crate::eleven_dimensional_j1_lorentz_residual::verify();
    let direct_local_lorentz = crate::eleven_dimensional_direct_local_lorentz::verify();
    let lorentz_holonomy_compensator_audit =
        crate::eleven_dimensional_lorentz_holonomy_compensator_audit::verify();
    let k_fag_polynomial_harness = crate::eleven_dimensional_k_fag_solver::verify();
    let k_fag_artifact = k_fag_artifact_gate();
    let level18_embedded = crate::eleven_dimensional_level18_embedded::verify();
    let level18_target_quotient = crate::eleven_dimensional_level18_target_quotient::verify()
        .expect("verify level-18 target quotient scaffold");
    let covariant_cohomology_entry = crate::eleven_dimensional_covariant_cohomology_gate::verify();
    let lowest_spinorial_differential = crate::eleven_dimensional_spinorial_differential::verify();
    let relaxed_spinorial_cohomology =
        crate::eleven_dimensional_relaxed_spinorial_cohomology::verify();
    let target_equation_complex = crate::eleven_dimensional_target_equation_complex::verify();
    let first_superspace_jet = crate::eleven_dimensional_first_superspace_jet::verify();
    let prepotential_gate = crate::eleven_dimensional_prepotential_gate::verify();
    let hook_bianchi = crate::eleven_dimensional_hook_bianchi::verify();
    let level18_momentum = crate::eleven_dimensional_level18_momentum::verify();
    let majorana = crate::eleven_dimensional_majorana::verify();
    let linear_susy = crate::eleven_dimensional_linear_susy::verify();
    let seed_occurrences = vec![
        seed("metric trace", "00000"),
        seed("symmetric traceless graviton potential", "20000"),
        seed("three-form potential", "00100"),
        seed("four-form curvature", "00010"),
        seed("spinor gauge parameter", "00001"),
        seed("gamma-traceless vector-spinor", "10001"),
        seed("retained torsion hook", "11000"),
    ];
    let all_seeds_present = seed_occurrences
        .iter()
        .all(|record| !record.levels_and_multiplicities.is_empty());
    let superspace_cohomology = SuperspaceCohomologyGate {
        seed_occurrences,
        all_seeds_present,
        exact_differentials_constructed: false,
        gauge_image_quotiented: false,
        bianchi_kernel_computed: false,
        pure_spinor_oracle_compared: false,
        physical_supermultiplet_isolated: false,
        boundary: "Inventory incidence is a work-list generator, not a differential, kernel, image, quotient, or superspace cohomology calculation.",
    };
    let linearized_equation = LinearizedEquationGate {
        target_free_equations_constructed: target_equation_complex.passed,
        lorentzian_majorana_real_form_constructed: majorana.majorana_real_form_constructed,
        physical_light_cone_susy_maps_constructed: linear_susy.linearized_susy_maps_constructed,
        physical_light_cone_susy_closure_checked: linear_susy.passed,
        source_to_target_superfield_map_constructed: false,
        gauge_invariant_superfield_operator_constructed: false,
        superfield_supersymmetry_closure_checked: false,
        component_einstein_rarita_schwinger_three_form_match: target_equation_complex.passed,
        passed: false,
        boundary: "The component target equations are exact, but no gauge-invariant operator from the conjectured spinor prepotential to those equations has been constructed.",
    };
    let bounded_gates_passed = free.passed
        && target_stream.passed
        && source_fixed_curvature.passed
        && abstract_clifford_join.passed
        && b5_majorana_target_join.passed
        && leading_x2_gauge.report_passed
        && first_momentum_functional_screens.passed
        && first_momentum_physical_fx.report_integrity_passed
        && physical_curvature.bounded_slice_passed
        && physical_adapter_audit.passed
        && j1_lorentz_residual.passed
        && direct_local_lorentz.passed
        && lorentz_holonomy_compensator_audit.passed
        && k_fag_polynomial_harness.passed
        && k_fag_artifact.report_integrity_passed
        && level18_embedded.passed
        && level18_target_quotient.passed
        && covariant_cohomology_entry.passed
        && lowest_spinorial_differential.passed
        && relaxed_spinorial_cohomology.passed
        && target_equation_complex.passed
        && first_superspace_jet.passed
        && prepotential_gate.worklist_consistent_with_current_exact_engine
        && hook_bianchi.passed
        && level18_momentum.bounded_program_passed
        && majorana.passed
        && linear_susy.passed
        && all_seeds_present;
    ElevenDimensionalTopDownReport {
        schema_version: "adynkra-11d-top-down-v5",
        free_complex: free,
        target_stream,
        source_fixed_curvature,
        abstract_clifford_join,
        b5_majorana_target_join,
        leading_x2_gauge,
        first_momentum_functional_screens,
        first_momentum_physical_fx,
        physical_curvature,
        physical_adapter_audit,
        j1_lorentz_residual,
        direct_local_lorentz,
        lorentz_holonomy_compensator_audit,
        k_fag_polynomial_harness,
        k_fag_artifact,
        level18_embedded,
        level18_target_quotient,
        covariant_cohomology_entry,
        lowest_spinorial_differential,
        relaxed_spinorial_cohomology,
        target_equation_complex,
        first_superspace_jet,
        prepotential_gate,
        hook_bianchi,
        level18_momentum,
        majorana,
        linear_susy,
        superspace_cohomology,
        linearized_equation,
        bounded_gates_passed,
        full_program_complete: false,
        next_exact_steps: vec![
            "complete the convention-fixed H_hat to torsion and curvature operator F",
            "solve the physical Psi_alpha to H_hat map K and its six channel coefficients",
            "specialize the 77-block target quotient using the physical K routing and coefficients",
            "extend the bounded partial-F_X obstruction to complete F, complete parameter and target projections, and generic polynomial momentum",
            "build the curvature, Bianchi, and field-equation complex and compare its physical quotient with 44+84|128",
            "extend the exact spinorial differential to the relaxed X_[2] plus X_[5] torsion complex",
            "join the exact physical light-cone supersymmetry maps to a covariant source-to-equation superfield operator",
        ],
        boundary: "Passing this aggregate means the completed bounded gates agree, including the exact rank-49 partial-F_X obstruction on its declared first-momentum slice, the first geometry-jet prolongation, the Lorentzian Majorana form, and the on-shell light-cone 44+84|128 supersymmetry maps. It does not complete F, establish full F A G_p, select a physical spinor-prepotential gauge symmetry, compute the full superspace cohomology, derive an Adynkrafield equation, construct an off-shell multiplet, or address nonlinear eleven-dimensional supergravity.",
    }
}

pub fn write_artifact(path: &Path) -> ElevenDimensionalTopDownReport {
    let report = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create 11D top-down artifact directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create 11D top-down artifact")),
        &report,
    )
    .expect("write 11D top-down artifact");
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fx_promotion_manifest_gate_fails_closed() {
        let missing = std::env::temp_dir().join(format!(
            "adynkra-missing-fx-promotion-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_file(&missing);
        let status = fx_promotion_manifest_status(
            missing.to_str().expect("temporary path is UTF-8"),
            "not-a-real-hash",
        );
        assert!(!status.present);
        assert!(!status.integrity_passed);

        let status = fx_promotion_manifest_status(
            "results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json",
            "not-a-real-hash",
        );
        assert!(status.present);
        assert!(!status.hash_matches);
        assert!(!status.integrity_passed);
    }

    #[test]
    fn aggregate_passes_bounded_gates_but_keeps_full_program_open() {
        let report = build();
        assert_eq!(report.schema_version, "adynkra-11d-top-down-v5");
        assert!(report.bounded_gates_passed);
        assert!(!report.full_program_complete);
        assert!(report.free_complex.passed);
        assert!(report.target_stream.passed);
        assert!(report.source_fixed_curvature.passed);
        assert!(report.abstract_clifford_join.passed);
        assert!(report.b5_majorana_target_join.passed);
        assert!(report.leading_x2_gauge.artifact_present);
        assert!(report.leading_x2_gauge.report_passed);
        assert_eq!(
            report.leading_x2_gauge.joint_six_channel_exact_rank,
            Some(7)
        );
        assert_eq!(
            report.leading_x2_gauge.joint_six_channel_exact_nullity,
            Some(5)
        );
        assert_eq!(
            report
                .leading_x2_gauge
                .joint_source_stream_kernel_relations_checked,
            5
        );
        assert_eq!(
            report
                .leading_x2_gauge
                .joint_source_stream_kernel_residual_entries,
            0
        );
        assert!(!report.leading_x2_gauge.full_f_a_g_p_established);
        assert!(report.first_momentum_functional_screens.passed);
        assert!(
            report
                .first_momentum_functional_screens
                .old_leading_extension_excluded_in_all_three_channels
        );
        assert!(
            !report
                .first_momentum_functional_screens
                .generic_polynomial_f_a_g_p_established
        );
        assert!(report.first_momentum_physical_fx.artifact_present);
        assert!(report.first_momentum_physical_fx.artifact_hash_matches);
        assert_eq!(
            report.first_momentum_physical_fx.artifact_sha256.as_deref(),
            Some("5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9")
        );
        assert!(report.first_momentum_physical_fx.report_integrity_passed);
        assert!(
            report
                .first_momentum_physical_fx
                .promotion_manifest_integrity_passed
        );
        assert_eq!(
            report
                .first_momentum_physical_fx
                .promotion_manifest_sha256
                .as_deref(),
            Some("98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4")
        );
        assert_eq!(
            report
                .first_momentum_physical_fx
                .promotion_manifest_checkpoint_hashes,
            336
        );
        assert_eq!(
            report.first_momentum_physical_fx.channel_degrees,
            vec![0, 1, 2, 3, 4, 5]
        );
        assert!(
            report
                .first_momentum_physical_fx
                .every_channel_has_56_operator_columns
        );
        assert!(
            report
                .first_momentum_physical_fx
                .every_channel_uses_declared_parameter_slice
        );
        assert!(
            report
                .first_momentum_physical_fx
                .every_channel_uses_declared_target_slice
        );
        assert_eq!(
            report.first_momentum_physical_fx.emitted_target_terms,
            1_014_543_703
        );
        assert_eq!(report.first_momentum_physical_fx.global_x2_rank, 49);
        assert_eq!(report.first_momentum_physical_fx.global_x2_nullity, 0);
        assert_eq!(report.first_momentum_physical_fx.global_x5_rank, 49);
        assert_eq!(report.first_momentum_physical_fx.global_x5_nullity, 0);
        assert_eq!(report.first_momentum_physical_fx.global_joint_rank, 49);
        assert_eq!(report.first_momentum_physical_fx.global_joint_nullity, 0);
        assert!(
            report
                .first_momentum_physical_fx
                .recorded_49_dimensional_ansatz_excluded_on_declared_slice
        );
        assert!(
            !report
                .first_momentum_physical_fx
                .full_parameter_projection_complete
        );
        assert!(
            !report
                .first_momentum_physical_fx
                .full_target_projection_complete
        );
        assert!(report.first_momentum_physical_fx.partial_fx_only);
        assert!(!report.first_momentum_physical_fx.full_f_a_g_p_established);
        assert!(
            report
                .first_momentum_physical_fx
                .fx_input_snapshot_hash_matches
        );
        assert_eq!(
            report
                .first_momentum_physical_fx
                .fx_input_snapshot_artifact_sha256
                .as_deref(),
            Some("c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f")
        );
        assert!(
            report
                .first_momentum_physical_fx
                .current_physical_curvature_envelope_hash_matches
        );
        assert_eq!(
            report
                .first_momentum_physical_fx
                .current_physical_curvature_artifact_sha256
                .as_deref(),
            Some("3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13")
        );
        assert!(report.physical_curvature.bounded_slice_passed);
        assert!(report.physical_adapter_audit.passed);
        assert!(report.j1_lorentz_residual.passed);
        assert!(report.direct_local_lorentz.passed);
        assert!(report.lorentz_holonomy_compensator_audit.passed);
        assert!(!report.physical_curvature.complete_f_from_h_hat_implemented);
        assert!(report.k_fag_polynomial_harness.passed);
        assert!(!report.k_fag_polynomial_harness.generic_k_solved);
        assert!(!report.k_fag_polynomial_harness.physical_fag_established);
        assert!(report.k_fag_artifact.report_integrity_passed);
        assert_eq!(
            report.k_fag_artifact.artifact_sha256.as_deref(),
            Some("11ec33c36d9536e17e617839cc8dbabc885b9d30bf13ff05a4d0dc5e6b9fe562")
        );
        assert!(
            report
                .k_fag_artifact
                .current_physical_curvature_envelope_hash_matches
        );
        assert!(report.k_fag_artifact.fx_input_snapshot_hash_matches);
        assert!(!report.k_fag_artifact.generic_k_solved);
        assert!(!report.k_fag_artifact.physical_fag_established);
        assert!(report.level18_embedded.passed);
        assert!(report.level18_embedded.all_77_embedded_maps_complete);
        assert!(report.level18_target_quotient.passed);
        assert!(
            !report
                .level18_target_quotient
                .physical_target_gauge_quotient_complete
        );
        assert!(
            !report
                .level18_embedded
                .physical_target_gauge_quotient_complete
        );
        assert!(report.covariant_cohomology_entry.passed);
        assert!(report.lowest_spinorial_differential.passed);
        assert!(report.relaxed_spinorial_cohomology.passed);
        assert!(report.target_equation_complex.passed);
        assert!(report.first_superspace_jet.passed);
        assert!(
            !report
                .first_superspace_jet
                .h_hat_to_first_geometry_jet_implemented
        );
        assert!(!report.first_superspace_jet.complete_physical_f_implemented);
        assert!(!report.first_superspace_jet.full_fag_established);
        assert!(
            !report
                .lowest_spinorial_differential
                .full_relaxed_torsion_differential_computed
        );
        assert!(report.level18_momentum.bounded_program_passed);
        assert!(!report.level18_momentum.full_requested_step_complete);
        assert!(report.majorana.passed);
        assert!(report.linear_susy.passed);
        assert!(report.superspace_cohomology.all_seeds_present);
        assert!(!report.superspace_cohomology.exact_differentials_constructed);
        assert!(!report.prepotential_gate.physical_kill_gate_executed);
        assert!(
            report
                .linearized_equation
                .physical_light_cone_susy_closure_checked
        );
        assert!(
            report
                .linearized_equation
                .component_einstein_rarita_schwinger_three_form_match
        );
        assert!(
            !report
                .linearized_equation
                .superfield_supersymmetry_closure_checked
        );
        assert!(!report.linearized_equation.passed);
    }
}
