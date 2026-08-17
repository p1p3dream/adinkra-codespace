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
    pub physical_operator_combination_selected: bool,
    pub full_f_a_g_p_established: bool,
    pub boundary: &'static str,
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
    pub leading_x2_gauge: LeadingX2ArtifactGate,
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
        physical_operator_combination_selected: value
            .get("physical_operator_combination_selected")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        full_f_a_g_p_established: value
            .get("full_f_a_g_p_established")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        boundary: "This is a SHA-256-addressed summary of the standalone exact leading X_[2] artifact. The aggregate does not recompute it. A passing bounded leading-symbol computation does not select the physical operator or establish the leading or full first-momentum F A G_p identity.",
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
    let leading_x2_gauge = leading_x2_artifact_gate();
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
        target_free_equations_constructed: free.passed,
        lorentzian_majorana_real_form_constructed: majorana.majorana_real_form_constructed,
        physical_light_cone_susy_maps_constructed: linear_susy.linearized_susy_maps_constructed,
        physical_light_cone_susy_closure_checked: linear_susy.passed,
        source_to_target_superfield_map_constructed: false,
        gauge_invariant_superfield_operator_constructed: false,
        superfield_supersymmetry_closure_checked: false,
        component_einstein_rarita_schwinger_three_form_match: false,
        passed: false,
        boundary: "The component target equations are exact, but no gauge-invariant operator from the conjectured spinor prepotential to those equations has been constructed.",
    };
    let bounded_gates_passed = free.passed
        && target_stream.passed
        && source_fixed_curvature.passed
        && abstract_clifford_join.passed
        && leading_x2_gauge.report_passed
        && prepotential_gate.worklist_consistent_with_current_exact_engine
        && hook_bianchi.passed
        && level18_momentum.bounded_program_passed
        && majorana.passed
        && linear_susy.passed
        && all_seeds_present;
    ElevenDimensionalTopDownReport {
        schema_version: "adynkra-11d-top-down-v3",
        free_complex: free,
        target_stream,
        source_fixed_curvature,
        abstract_clifford_join,
        leading_x2_gauge,
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
            "complete the source-fixed H_hat to torsion differential join and certify F composed with K equals zero",
            "apply F to the target-resolved D^17 Lambda and p D^15 Lambda streams independently for all six source channels",
            "construct the 77 embedded level-18 source-target maps and the momentum-dependent target gauge quotient",
            "compute gauge-quotiented superspace cohomology and compare it with independent pure-spinor cohomology",
            "join the exact physical light-cone supersymmetry maps to a covariant source-to-equation superfield operator",
        ],
        boundary: "Passing this aggregate means the completed bounded gates agree, including the Lorentzian Majorana form and on-shell light-cone 44+84|128 supersymmetry maps. It does not select a physical spinor-prepotential gauge symmetry, compute the full superspace cohomology, derive an Adynkrafield equation, construct an off-shell multiplet, or address nonlinear eleven-dimensional supergravity.",
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
    fn aggregate_fails_closed_while_completed_subgates_pass() {
        let report = build();
        assert!(!report.bounded_gates_passed);
        assert!(!report.full_program_complete);
        assert!(report.free_complex.passed);
        assert!(report.target_stream.passed);
        assert!(report.source_fixed_curvature.passed);
        assert!(report.abstract_clifford_join.passed);
        assert!(report.leading_x2_gauge.artifact_present);
        assert!(report.leading_x2_gauge.report_passed);
        assert!(!report.leading_x2_gauge.full_f_a_g_p_established);
        assert!(!report.level18_momentum.bounded_program_passed);
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
            !report
                .linearized_equation
                .superfield_supersymmetry_closure_checked
        );
        assert!(!report.linearized_equation.passed);
    }
}
