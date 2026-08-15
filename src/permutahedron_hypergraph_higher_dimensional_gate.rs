//! Bounded higher-dimensional control gate for the S8 hypergraph program.
//!
//! The gate integrates exact four-dimensional chiral-vector and chiral-tensor
//! component closures with the hypergraph control projection.  It also makes
//! the remaining data gap explicit instead of inferring spatial or gauge data
//! from a worldline valise.

use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-hypergraph-higher-dimensional-gate-v2";

#[derive(Debug, Clone, Serialize)]
pub struct ControlGateRecord {
    pub id: String,
    pub discovered_family_id: usize,
    pub family_slice_id: usize,
    pub unsigned_support_signable: bool,
    pub published_assignment_closes_in_one_dimension: bool,
    pub exact_four_dimensional_fixture_available: bool,
    pub exact_four_dimensional_closure_passed: Option<bool>,
    pub exact_worldline_anchor_recovered: Option<bool>,
    pub spatial_linkage_fingerprint_available: bool,
    pub gauge_residue_fingerprint_available: bool,
    pub disposition: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct HigherDimensionalGateValidation {
    pub controls: usize,
    pub controls_sharing_one_unsigned_family: usize,
    pub controls_sharing_one_valid_worldline_signing_class: usize,
    pub published_assignments_rejected_by_worldline_nonclosure: usize,
    pub exact_four_dimensional_positive_controls: usize,
    pub exact_four_dimensional_component_relations_checked: usize,
    pub exact_reduced_matrix_entries_checked: usize,
    pub cv_component_closure_passed: bool,
    pub ct_component_closure_passed: bool,
    pub cv_anchor_recovered: bool,
    pub ct_anchor_recovered: bool,
    pub cv_ct_spatial_operators_distinct: bool,
    pub cv_ct_gauge_residues_distinct: bool,
    pub garden_positive_controls_without_claimed_four_dimensional_parent: Vec<String>,
    pub printed_nonclosing_controls_without_target_specification: Vec<String>,
    pub stated_parent_positive_control_gate_complete: bool,
    pub full_seven_control_higher_dimensional_gate_applicable: bool,
    pub broader_s8_scan_authorized: bool,
    pub audit_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HigherDimensionalGateArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub controls: Vec<ControlGateRecord>,
    pub validation: HigherDimensionalGateValidation,
    pub findings: Vec<String>,
    pub next_required_input: Vec<&'static str>,
    pub boundary: &'static str,
}

pub fn build() -> HigherDimensionalGateArtifact {
    let projection = crate::permutahedron_hypergraph_controls::build();
    let cv = crate::chiral_vector_4d::verify();
    let ct = crate::chiral_tensor_4d::verify();
    let fingerprints = crate::higher_dimensional_fingerprint::build();

    let mut controls = Vec::with_capacity(projection.controls.len());
    for control in &projection.controls {
        let (fixture, closure, anchor, disposition) = match control.id {
            "CV" => (
                true,
                Some(cv.passed),
                Some(cv.exact_cv_anchor_recovered),
                "exact four-dimensional positive control passed",
            ),
            "CT" => (
                true,
                Some(ct.passed),
                Some(ct.exact_ct_anchor_recovered),
                "exact four-dimensional positive control passed",
            ),
            "O" => (
                false,
                None,
                None,
                "original one-dimensional Garden control with no claimed four-dimensional component parent",
            ),
            _ if !control.published_or_certified_assignment_closes => (
                false,
                None,
                None,
                "published assignment rejected by one-dimensional closure prerequisite",
            ),
            _ => unreachable!("complete published control list"),
        };
        controls.push(ControlGateRecord {
            id: control.id.into(),
            discovered_family_id: control.discovered_family_id,
            family_slice_id: control.family_slice_id,
            unsigned_support_signable: control.unsigned_support_has_garden_signings,
            published_assignment_closes_in_one_dimension: control
                .published_or_certified_assignment_closes,
            exact_four_dimensional_fixture_available: fixture,
            exact_four_dimensional_closure_passed: closure,
            exact_worldline_anchor_recovered: anchor,
            spatial_linkage_fingerprint_available: matches!(control.id, "CV" | "CT"),
            gauge_residue_fingerprint_available: matches!(control.id, "CV" | "CT"),
            disposition,
        });
    }

    let worldline_rejected = controls
        .iter()
        .filter(|control| !control.published_assignment_closes_in_one_dimension)
        .count();
    let exact_positive_controls = controls
        .iter()
        .filter(|control| control.exact_four_dimensional_fixture_available)
        .count();
    let garden_positive_without_parent: Vec<String> = controls
        .iter()
        .filter(|control| {
            control.published_assignment_closes_in_one_dimension
                && !control.exact_four_dimensional_fixture_available
        })
        .map(|control| control.id.clone())
        .collect();
    let nonclosing_without_target: Vec<String> = controls
        .iter()
        .filter(|control| {
            !control.published_assignment_closes_in_one_dimension
                && !control.exact_four_dimensional_fixture_available
        })
        .map(|control| control.id.clone())
        .collect();
    let stated_parent_gate_complete = cv.passed && ct.passed;
    let full_seven_control_gate_applicable = false;
    let broader_scan_authorized = false;
    let component_relations = cv.component_relations_checked + ct.component_relations_checked;
    let matrix_entries = cv.reduced_l_matrix_entries_checked + ct.reduced_l_matrix_entries_checked;
    let audit_passed = projection.validation.passed
        && controls.len() == 7
        && worldline_rejected == 4
        && exact_positive_controls == 2
        && cv.passed
        && ct.passed
        && fingerprints.passed
        && component_relations == 1_296
        && matrix_entries == 1_024
        && garden_positive_without_parent == ["O"]
        && nonclosing_without_target == ["CC", "TT", "TV", "VV"]
        && stated_parent_gate_complete
        && !full_seven_control_gate_applicable
        && !broader_scan_authorized;

    let validation = HigherDimensionalGateValidation {
        controls: controls.len(),
        controls_sharing_one_unsigned_family: controls.len(),
        controls_sharing_one_valid_worldline_signing_class: controls.len(),
        published_assignments_rejected_by_worldline_nonclosure: worldline_rejected,
        exact_four_dimensional_positive_controls: exact_positive_controls,
        exact_four_dimensional_component_relations_checked: component_relations,
        exact_reduced_matrix_entries_checked: matrix_entries,
        cv_component_closure_passed: cv.passed,
        ct_component_closure_passed: ct.passed,
        cv_anchor_recovered: cv.exact_cv_anchor_recovered,
        ct_anchor_recovered: ct.exact_ct_anchor_recovered,
        cv_ct_spatial_operators_distinct: !fingerprints.spatial_operators_identical_in_source_basis,
        cv_ct_gauge_residues_distinct: !fingerprints.gauge_residues_identical_in_source_basis,
        garden_positive_controls_without_claimed_four_dimensional_parent:
            garden_positive_without_parent,
        printed_nonclosing_controls_without_target_specification: nonclosing_without_target,
        stated_parent_positive_control_gate_complete: stated_parent_gate_complete,
        full_seven_control_higher_dimensional_gate_applicable: full_seven_control_gate_applicable,
        broader_s8_scan_authorized: broader_scan_authorized,
        audit_passed,
    };

    HigherDimensionalGateArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Bounded higher-dimensional gate for published S8 controls",
        method: vec![
            "Require one-dimensional Garden closure before applying any higher-dimensional test to a published sign assignment.",
            "Verify the sourced chiral-vector and chiral-tensor component algebras exactly, including their gauge residues.",
            "Reduce each sourced four-dimensional system and compare all 512 matrix entries with its committed S8 anchor.",
            "Compare the retained spatial-linkage and gauge-residue fingerprints without treating source-basis hashes as physical invariants.",
            "Refuse broader scanning when a control lacks sourced Lorentz, spatial-linkage, gauge, and reduction data.",
        ],
        findings: vec![
            "CV passes 612 exact four-dimensional component relations and reproduces all 512 anchor entries.".into(),
            "CT passes 684 exact four-dimensional component relations and reproduces all 512 anchor entries.".into(),
            "CV and CT share the same unsigned family and valid worldline signed class, but retain different spatial operators and gauge residues in their sourced four-dimensional bases.".into(),
            "CC, TT, TV, and VV are rejected as printed assignments by Garden nonclosure, but their signable supports still lack higher-dimensional target specifications.".into(),
            "O is a Garden-positive one-dimensional diadem construction, but the audited sources do not claim a four-dimensional component parent for it.".into(),
            "The stated-parent positive-control gate is complete for CV and CT. A seven-control higher-dimensional gate is not applicable, and a broader S8 scan remains unauthorized without a new physical target.".into(),
        ],
        next_required_input: vec![
            "an independently sourced higher-dimensional component realization for O, VM1, VM2, or VM3 before treating it as a physical positive control",
            "a Lorentz representation for every field in any candidate support",
            "complete spatial-derivative linkage coefficients",
            "gauge potential, field strength, gauge transformation, and Bianchi data",
            "temporal gauge and the exact field-to-node reduction map",
            "component closure including every gauge residue",
        ],
        boundary: "This audit completes all sourced higher-dimensional positive controls in the current corpus, CV and CT, and deliberately stops before unsourced inference. O remains a valid one-dimensional Garden control, not a missing fixture for an asserted parent. Garden failure rejects the four printed negative assignments, but it does not reject every valid re-signing of their unsigned supports. No higher-dimensional parent can be assigned or excluded for those supports without additional target data.",
        controls,
        validation,
    }
}

pub fn write_artifacts(
    data_path: &Path,
    validation_path: &Path,
) -> HigherDimensionalGateValidation {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create validation directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create data artifact")),
        &artifact,
    )
    .expect("write data artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create validation artifact")),
        &artifact.validation,
    )
    .expect("write validation artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stated_parent_positive_controls_pass_but_broader_scan_is_not_authorized() {
        let artifact = build();
        assert!(artifact.validation.audit_passed);
        assert!(artifact.validation.cv_component_closure_passed);
        assert!(artifact.validation.ct_component_closure_passed);
        assert!(artifact.validation.cv_ct_spatial_operators_distinct);
        assert!(artifact.validation.cv_ct_gauge_residues_distinct);
        assert!(
            artifact
                .validation
                .stated_parent_positive_control_gate_complete
        );
        assert!(
            !artifact
                .validation
                .full_seven_control_higher_dimensional_gate_applicable
        );
        assert!(!artifact.validation.broader_s8_scan_authorized);
    }
}
