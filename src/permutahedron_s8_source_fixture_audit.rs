//! Primary-source eligibility audit for higher-dimensional S8 controls.
//!
//! This ledger separates representations with published higher-dimensional
//! component transformations from Garden-algebra constructions that have no
//! stated four-dimensional parent. It prevents the latter from being treated
//! as missing fixtures for a parent that the sources never claim.

use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-source-fixture-audit-v1";

#[derive(Debug, Clone, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub title: &'static str,
    pub locator: &'static str,
    pub pdf_sha256: &'static str,
    pub role: &'static str,
    pub finding: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureClassRecord {
    pub id: &'static str,
    pub source_class: &'static str,
    pub garden_status: &'static str,
    pub stated_higher_dimensional_parent: bool,
    pub exact_higher_dimensional_fixture_available: bool,
    pub eligible_as_physical_positive_control: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceFixtureAuditValidation {
    pub primary_sources_reviewed: usize,
    pub source_hashes_are_distinct: bool,
    pub source_eligible_higher_dimensional_positive_controls: Vec<&'static str>,
    pub garden_positive_controls_without_claimed_higher_dimensional_parent: Vec<&'static str>,
    pub mathematical_s4_sectors_without_stated_4d_parent: Vec<&'static str>,
    pub printed_nonclosing_s8_controls: Vec<&'static str>,
    pub exact_cv_component_gate_passed: bool,
    pub exact_ct_component_gate_passed: bool,
    pub garden_positive_o_control_reproduced: bool,
    pub unrestricted_recursion_audit_passed: bool,
    pub node_basis_leakage_audit_passed: bool,
    pub stated_parent_positive_control_gate_complete: bool,
    pub new_independent_holdout_fixture_found: bool,
    pub external_physical_input_required: bool,
    pub broader_s8_scan_authorized: bool,
    pub audit_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceFixtureAuditArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub sources: Vec<SourceRecord>,
    pub fixture_classes: Vec<FixtureClassRecord>,
    pub validation: SourceFixtureAuditValidation,
    pub findings: Vec<&'static str>,
    pub next_required_input: Vec<&'static str>,
    pub boundary: &'static str,
}

fn source_records() -> Vec<SourceRecord> {
    vec![
        SourceRecord {
            arxiv_id: "1210.0478",
            title: "Adinkra (in)equivalence from Coxeter group representations: A case study",
            locator: "Secs. 2 and 5, especially pp. 6 and 11",
            pdf_sha256: "510a7f588546821003b85e5908668e81285a49e33d89245198db2fd65ff85af8",
            role: "classification of the six GR(4,4) permutation quartets",
            finding: "CM, VM, and TM are tied to familiar reduced multiplets; the remaining quartets are algebraic solution classes whose higher-dimensional equivalence is explicitly posed as a separate problem.",
        },
        SourceRecord {
            arxiv_id: "1608.07864",
            title: "N=4 and N=8 SUSY Quantum Mechanics and Klein's Vierergruppe",
            locator: "Sec. 2.1, Eq. (7), p. 6",
            pdf_sha256: "aec4378c3bdd22401dcaa32c96f909a6dedd81bf0e0ec466373afcde3c9b49de",
            role: "explicit distinction between physical reductions and other S4 types",
            finding: "CM, VM, and TM are identified as reductions of 4D N=1 multiplets, while VM1, VM2, and VM3 are called the other types.",
        },
        SourceRecord {
            arxiv_id: "2012.14015",
            title: "A note on exemplary off-shell constructions of 4D, N=2 supersymmetry representations",
            locator: "Sec. 3, Eq. (3.6), p. 6",
            pdf_sha256: "67606499dda53ff78e3d80ca405498d8d1e2fd060936605fac04804ca7cd9b47",
            role: "definition of the DO(8) or O octet",
            finding: "The O octet is constructed by an algorithm independent of combining N=1 multiplets; no four-dimensional component transformations, Lorentz assignment, or gauge target are supplied for O.",
        },
        SourceRecord {
            arxiv_id: "2304.09830",
            title: "N=2 SUSY and the Hexipentisteriruncicantitruncated 7-Simplex",
            locator: "Secs. 1 and 4.1, especially p. 21",
            pdf_sha256: "d68c14ad31e824b52076bd4be078c6aec0167f0a622c55c4a9d4d2a6d2ef2a34",
            role: "recursive one-dimensional construction and diadem terminology",
            finding: "The work studies recursive N-extended 1D adinkra matrices and identifies the diadem O with the Rana subgroup; it does not provide an independent higher-dimensional component fixture for O or VM1, VM2, or VM3.",
        },
        SourceRecord {
            arxiv_id: "2408.09342",
            title: "A Precis: Minimal Four Color Holoraumy and Wolfram's New Kind of Science Paradigm",
            locator: "Sec. 2, Table 3 and accompanying paragraph, p. 8",
            pdf_sha256: "8344013301257c6efc51c7b016ec7c1faf6bb7468804d3626b4892eeb770cca0",
            role: "direct provenance statement for VM1, VM2, and VM3",
            finding: "The paper contrasts 0-brane reductions of 4D N=1 multiplets with VM1, VM2, and VM3, which were derived solely as mathematical Garden-equation solutions.",
        },
    ]
}

fn fixture_classes() -> Vec<FixtureClassRecord> {
    vec![
        FixtureClassRecord {
            id: "CV",
            source_class: "sourced four-dimensional positive control",
            garden_status: "published assignment closes",
            stated_higher_dimensional_parent: true,
            exact_higher_dimensional_fixture_available: true,
            eligible_as_physical_positive_control: true,
        },
        FixtureClassRecord {
            id: "CT",
            source_class: "sourced four-dimensional positive control",
            garden_status: "published assignment closes",
            stated_higher_dimensional_parent: true,
            exact_higher_dimensional_fixture_available: true,
            eligible_as_physical_positive_control: true,
        },
        FixtureClassRecord {
            id: "O",
            source_class: "original one-dimensional diadem construction",
            garden_status: "certified assignment closes",
            stated_higher_dimensional_parent: false,
            exact_higher_dimensional_fixture_available: false,
            eligible_as_physical_positive_control: false,
        },
        FixtureClassRecord {
            id: "VM1/VM2/VM3",
            source_class: "mathematical S4 Garden solution sectors",
            garden_status: "Garden solution classes",
            stated_higher_dimensional_parent: false,
            exact_higher_dimensional_fixture_available: false,
            eligible_as_physical_positive_control: false,
        },
        FixtureClassRecord {
            id: "CC/TT/TV/VV",
            source_class: "printed S8 recursion controls",
            garden_status: "printed assignments do not close",
            stated_higher_dimensional_parent: false,
            exact_higher_dimensional_fixture_available: false,
            eligible_as_physical_positive_control: false,
        },
    ]
}

pub fn build() -> SourceFixtureAuditArtifact {
    let sources = source_records();
    let fixtures = fixture_classes();
    let cv = crate::chiral_vector_4d::verify();
    let ct = crate::chiral_tensor_4d::verify();
    let controls = crate::permutahedron_hypergraph_controls::build();
    let unrestricted = crate::permutahedron_s8_unrestricted_recursion::build();
    let leakage = crate::permutahedron_s8_orbit_leakage::build();

    let source_hashes_are_distinct = {
        let hashes: std::collections::BTreeSet<_> =
            sources.iter().map(|source| source.pdf_sha256).collect();
        hashes.len() == sources.len()
    };
    let garden_positive_o_control_reproduced = controls
        .controls
        .iter()
        .any(|control| control.id == "O" && control.published_or_certified_assignment_closes);
    let stated_parent_positive_control_gate_complete = cv.passed && ct.passed;
    let new_independent_holdout_fixture_found = false;
    let external_physical_input_required = !new_independent_holdout_fixture_found;
    let broader_s8_scan_authorized = false;
    let audit_passed = sources.len() == 5
        && source_hashes_are_distinct
        && fixtures.len() == 5
        && cv.passed
        && ct.passed
        && garden_positive_o_control_reproduced
        && unrestricted.validation.audit_passed
        && leakage.validation.audit_passed
        && stated_parent_positive_control_gate_complete
        && !new_independent_holdout_fixture_found
        && external_physical_input_required
        && !broader_s8_scan_authorized;

    SourceFixtureAuditArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Primary-source eligibility audit for higher-dimensional S8 controls",
        method: vec![
            "Review the local primary-source corpus for an asserted higher-dimensional parent and complete component transformations for O, VM1, VM2, or VM3.",
            "Separate exact higher-dimensional positive controls from one-dimensional Garden constructions and printed nonclosing controls.",
            "Re-run the exact CV and CT component gates, the O Garden check, unrestricted recursion census, and node-basis leakage audit.",
            "Authorize no broader S8 physical scan without an independently sourced Lorentz, spatial-linkage, gauge, and reduction target.",
        ],
        sources,
        fixture_classes: fixtures,
        validation: SourceFixtureAuditValidation {
            primary_sources_reviewed: 5,
            source_hashes_are_distinct,
            source_eligible_higher_dimensional_positive_controls: vec!["CT", "CV"],
            garden_positive_controls_without_claimed_higher_dimensional_parent: vec!["O"],
            mathematical_s4_sectors_without_stated_4d_parent: vec!["VM1", "VM2", "VM3"],
            printed_nonclosing_s8_controls: vec!["CC", "TT", "TV", "VV"],
            exact_cv_component_gate_passed: cv.passed,
            exact_ct_component_gate_passed: ct.passed,
            garden_positive_o_control_reproduced,
            unrestricted_recursion_audit_passed: unrestricted.validation.audit_passed,
            node_basis_leakage_audit_passed: leakage.validation.audit_passed,
            stated_parent_positive_control_gate_complete,
            new_independent_holdout_fixture_found,
            external_physical_input_required,
            broader_s8_scan_authorized,
            audit_passed,
        },
        findings: vec![
            "The exact stated-parent positive-control gate is complete: both CT and CV pass their sourced four-dimensional component and reduction checks.",
            "O is a valid one-dimensional Garden control, but the audited sources define it as an original diadem construction rather than supplying an asserted four-dimensional component parent.",
            "VM1, VM2, and VM3 are explicitly identified as mathematical Garden solution sectors, not 0-brane reductions with published higher-dimensional component laws.",
            "No independent physical holdout fixture was found in the audited corpus.",
            "The next physical discrimination step is input-bound, not compute-bound.",
        ],
        next_required_input: vec![
            "an independently sourced higher-dimensional component realization not used to construct the selector",
            "complete Lorentz representations and spatial-derivative linkage coefficients",
            "all gauge transformations, field strengths, Bianchi identities, and closure residues",
            "the temporal-gauge field-to-node reduction map",
        ],
        boundary: "This is a provenance and eligibility audit of the available corpus, not a proof that O, VM1, VM2, or VM3 can never admit a higher-dimensional realization. It establishes only that the reviewed sources do not supply the physical target data required for an independent holdout test.",
    }
}

pub fn write_artifact(path: &Path) -> SourceFixtureAuditValidation {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create source fixture audit artifact")),
        &artifact,
    )
    .expect("write source fixture audit artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stated_parent_gate_is_complete_but_no_independent_holdout_exists() {
        let artifact = build();
        assert!(artifact.validation.audit_passed);
        assert!(
            artifact
                .validation
                .stated_parent_positive_control_gate_complete
        );
        assert!(!artifact.validation.new_independent_holdout_fixture_found);
        assert!(artifact.validation.external_physical_input_required);
        assert!(!artifact.validation.broader_s8_scan_authorized);
    }
}
