//! Aggregated, fail-closed status for the top-down 11D program.
//!
//! This module separates completed exact gates from the open prepotential,
//! superspace-cohomology, and field-equation claims.  Representation incidence
//! is used only to generate a sparse physical-seed work list.  It is never
//! promoted to a cohomology result without explicit differentials.

use serde::Serialize;
use std::fs::File;
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
    pub source_to_target_superfield_map_constructed: bool,
    pub gauge_invariant_superfield_operator_constructed: bool,
    pub supersymmetry_closure_checked: bool,
    pub component_einstein_rarita_schwinger_three_form_match: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ElevenDimensionalTopDownReport {
    pub schema_version: &'static str,
    pub free_complex: crate::eleven_dimensional_free_complex::ElevenDimensionalFreeComplexReport,
    pub prepotential_gate:
        crate::eleven_dimensional_prepotential_gate::ElevenDimensionalPrepotentialGateReport,
    pub hook_bianchi: crate::eleven_dimensional_hook_bianchi::HookBianchiReport,
    pub superspace_cohomology: SuperspaceCohomologyGate,
    pub linearized_equation: LinearizedEquationGate,
    pub bounded_gates_passed: bool,
    pub full_program_complete: bool,
    pub next_exact_steps: Vec<&'static str>,
    pub boundary: &'static str,
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
    let prepotential_gate = crate::eleven_dimensional_prepotential_gate::verify();
    let hook_bianchi = crate::eleven_dimensional_hook_bianchi::verify();
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
        source_to_target_superfield_map_constructed: false,
        gauge_invariant_superfield_operator_constructed: false,
        supersymmetry_closure_checked: false,
        component_einstein_rarita_schwinger_three_form_match: false,
        passed: false,
        boundary: "The component target equations are exact, but no gauge-invariant operator from the conjectured spinor prepotential to those equations has been constructed.",
    };
    let bounded_gates_passed = free.passed
        && prepotential_gate.worklist_consistent_with_current_exact_engine
        && hook_bianchi.passed
        && all_seeds_present;
    ElevenDimensionalTopDownReport {
        schema_version: "adynkra-11d-top-down-v1",
        free_complex: free,
        prepotential_gate,
        hook_bianchi,
        superspace_cohomology,
        linearized_equation,
        bounded_gates_passed,
        full_program_complete: false,
        next_exact_steps: vec![
            "construct a target-resolved 11x32 composition stream for every A composed with G_p job",
            "supply the target superfield gauge map K and curvature F with F composed with K equal to zero",
            "construct embedded level-18 hook-target kernels and the momentum-corrected next differential",
            "compute gauge-quotiented superspace cohomology and compare it with independent pure-spinor cohomology",
            "solve for a gauge-invariant source-to-equation operator and verify all 32 supersymmetries",
        ],
        boundary: "Passing this aggregate means the completed bounded gates agree. It does not select a physical spinor-prepotential gauge symmetry, compute the full superspace cohomology, derive an Adynkrafield equation, or address nonlinear eleven-dimensional supergravity.",
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
    fn bounded_gates_pass_without_promoting_open_claims() {
        let report = build();
        assert!(report.bounded_gates_passed);
        assert!(!report.full_program_complete);
        assert!(report.free_complex.passed);
        assert!(report.superspace_cohomology.all_seeds_present);
        assert!(!report.superspace_cohomology.exact_differentials_constructed);
        assert!(!report.prepotential_gate.physical_kill_gate_executed);
        assert!(!report.linearized_equation.passed);
    }
}
