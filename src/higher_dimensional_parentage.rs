//! Basis-independent physical fingerprints and bounded 4D parentage inference.
//!
//! The invariants here deliberately forget individual component labels.  They
//! retain multiplicities of Lorentz irreducibles, engineering levels,
//! differential-form gauge complexes, covariant derivative couplings, and
//! physical central operators.  Those data are invariant under arbitrary
//! invertible basis changes inside each retained block.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "higher-dimensional-parentage-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Statistics {
    Boson,
    Fermion,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LorentzType {
    Scalar,
    MajoranaSpinor,
    FormPotential { degree: u8 },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldRole {
    Physical,
    Auxiliary,
    GaugePotential,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct FieldBlock {
    pub statistics: Statistics,
    pub lorentz_type: LorentzType,
    pub role: FieldRole,
    pub irrep_multiplicity: u8,
    pub real_component_count: u8,
    /// Twice the engineering level relative to the lowest physical boson.
    pub engineering_level_twice: i8,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct GaugeComplex {
    pub potential_degree: u8,
    pub parameter_degree: u8,
    pub reducibility_depth: u8,
    pub field_strength_degree: u8,
    pub bianchi_degree: u8,
    pub potential_components: u8,
    pub independent_gauge_directions: u8,
    pub quotient_components: u8,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DerivativeContent {
    pub maximum_spacetime_order: u8,
    pub scalar_gradient_multiplicity: u8,
    pub form_strength_coupling_degrees: Vec<u8>,
    pub algebraic_auxiliary_multiplicity: u8,
    pub temporal_linkage_present: bool,
    pub spatial_linkage_present: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CentralStructure {
    pub local_stueckelberg_generators: u8,
    pub physical_worldline_generators: u8,
    pub bosonic_rank: u8,
    pub fermionic_rank: u8,
    pub commutes_with_all_supercharges: bool,
    pub involutive_when_nonzero: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Certification {
    ExactFourDimensionalClosure,
    ExactFourDimensionalClosureWithoutCentralBridge,
    ExactTangentPreflight,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ClosureCertificate {
    pub certification: Certification,
    pub component_relations_checked: usize,
    pub unexplained_residual_relations: usize,
    pub gauge_residues_retained: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFingerprint {
    pub parent_id: &'static str,
    pub linear_class: &'static str,
    pub source: &'static str,
    pub worldline_bosons: u8,
    pub worldline_fermions: u8,
    pub field_blocks: Vec<FieldBlock>,
    pub gauge_complexes: Vec<GaugeComplex>,
    pub derivative_content: DerivativeContent,
    pub central: CentralStructure,
    pub closure: ClosureCertificate,
    pub nonlinear_composite_connection: bool,
    pub requires_regular_background_patch: bool,
    pub canonical_linear_key: String,
    pub canonical_completion_key: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ParentageQuery {
    pub worldline_size: Option<[u8; 2]>,
    pub field_blocks: Option<Vec<FieldBlock>>,
    pub gauge_complexes: Option<Vec<GaugeComplex>>,
    pub derivative_content: Option<DerivativeContent>,
    pub central: Option<CentralStructure>,
    pub require_exact_four_dimensional_closure: bool,
    pub nonlinear_composite_connection: Option<bool>,
    pub requires_regular_background_patch: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RejectionWitness {
    pub candidate: &'static str,
    pub invariant: &'static str,
    pub expected: String,
    pub observed: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CandidateResult {
    pub parent_id: &'static str,
    pub linear_class: &'static str,
    pub exact_four_dimensional_fixture: bool,
    pub qualification: Option<&'static str>,
    pub missing_information: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceDecision {
    Identified,
    Compatible,
    Insufficient,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InferenceResult {
    pub decision: InferenceDecision,
    pub compatible: Vec<CandidateResult>,
    pub rejected: Vec<RejectionWitness>,
    pub unique_parent: Option<&'static str>,
    pub unique_linear_class: Option<&'static str>,
    pub interpretation: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct MutationControl {
    pub mutation: &'static str,
    pub rejected_parent: &'static str,
    pub witness_invariant: &'static str,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParentageArtifact {
    pub schema_version: &'static str,
    pub equivalence_scope: Vec<&'static str>,
    pub completeness_boundary: &'static str,
    pub catalog: Vec<PhysicalFingerprint>,
    pub linear_classes: BTreeMap<&'static str, Vec<&'static str>>,
    pub worldline_only_query: InferenceResult,
    pub chiral_vector_query: InferenceResult,
    pub chiral_tensor_query: InferenceResult,
    pub ct_structural_query: InferenceResult,
    pub vector_tensor_query: InferenceResult,
    pub scalar_tensor_completion_query: InferenceResult,
    pub mutation_controls: Vec<MutationControl>,
    pub passed: bool,
}

fn scalar_block(role: FieldRole, multiplicity: u8, level: i8) -> FieldBlock {
    FieldBlock {
        statistics: Statistics::Boson,
        lorentz_type: LorentzType::Scalar,
        role,
        irrep_multiplicity: multiplicity,
        real_component_count: multiplicity,
        engineering_level_twice: level,
    }
}

fn form_block(degree: u8, components: u8) -> FieldBlock {
    FieldBlock {
        statistics: Statistics::Boson,
        lorentz_type: LorentzType::FormPotential { degree },
        role: FieldRole::GaugePotential,
        irrep_multiplicity: 1,
        real_component_count: components,
        engineering_level_twice: 0,
    }
}

fn fermion_block() -> FieldBlock {
    FieldBlock {
        statistics: Statistics::Fermion,
        lorentz_type: LorentzType::MajoranaSpinor,
        role: FieldRole::Physical,
        irrep_multiplicity: 2,
        real_component_count: 8,
        engineering_level_twice: 1,
    }
}

fn gauge_complex(degree: u8) -> GaugeComplex {
    match degree {
        1 => GaugeComplex {
            potential_degree: 1,
            parameter_degree: 0,
            reducibility_depth: 0,
            field_strength_degree: 2,
            bianchi_degree: 3,
            potential_components: 4,
            independent_gauge_directions: 1,
            quotient_components: 3,
        },
        2 => GaugeComplex {
            potential_degree: 2,
            parameter_degree: 1,
            reducibility_depth: 1,
            field_strength_degree: 3,
            bianchi_degree: 4,
            potential_components: 6,
            independent_gauge_directions: 3,
            quotient_components: 3,
        },
        _ => panic!("unsupported fixture form degree"),
    }
}

fn central_none(local_stueckelberg_generators: u8) -> CentralStructure {
    CentralStructure {
        local_stueckelberg_generators,
        physical_worldline_generators: 0,
        bosonic_rank: 0,
        fermionic_rank: 0,
        commutes_with_all_supercharges: true,
        involutive_when_nonzero: false,
    }
}

fn central_one_z() -> CentralStructure {
    CentralStructure {
        local_stueckelberg_generators: 0,
        physical_worldline_generators: 1,
        bosonic_rank: 8,
        fermionic_rank: 8,
        commutes_with_all_supercharges: true,
        involutive_when_nonzero: true,
    }
}

fn canonical_lines(fingerprint: &PhysicalFingerprint, completion: bool) -> Vec<String> {
    let mut fields = fingerprint.field_blocks.clone();
    fields.sort();
    let mut gauges = fingerprint.gauge_complexes.clone();
    gauges.sort();
    let mut strengths = fingerprint
        .derivative_content
        .form_strength_coupling_degrees
        .clone();
    strengths.sort();
    let mut lines = vec![
        format!("schema={SCHEMA_VERSION}"),
        format!(
            "worldline={}:{}",
            fingerprint.worldline_bosons, fingerprint.worldline_fermions
        ),
        format!(
            "fields={}",
            serde_json::to_string(&fields).expect("serialize canonical field blocks")
        ),
        format!(
            "gauges={}",
            serde_json::to_string(&gauges).expect("serialize canonical gauge complexes")
        ),
        format!(
            "derivatives={}:{}:{}:{}:{}:{}",
            fingerprint.derivative_content.maximum_spacetime_order,
            fingerprint.derivative_content.scalar_gradient_multiplicity,
            serde_json::to_string(&strengths).expect("serialize canonical strength degrees"),
            fingerprint
                .derivative_content
                .algebraic_auxiliary_multiplicity,
            fingerprint.derivative_content.temporal_linkage_present,
            fingerprint.derivative_content.spatial_linkage_present,
        ),
        format!(
            "central={}:{}:{}:{}:{}:{}",
            fingerprint.central.physical_worldline_generators,
            fingerprint.central.bosonic_rank,
            fingerprint.central.fermionic_rank,
            fingerprint.central.commutes_with_all_supercharges,
            fingerprint.central.involutive_when_nonzero,
            if completion {
                fingerprint.central.local_stueckelberg_generators
            } else {
                0
            },
        ),
    ];
    if completion {
        lines.push(format!(
            "completion={}:{}",
            fingerprint.nonlinear_composite_connection,
            fingerprint.requires_regular_background_patch
        ));
    }
    lines
}

fn sha256(lines: &[String]) -> String {
    let mut digest = Sha256::new();
    for line in lines {
        digest.update(line.as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

fn finish(mut fingerprint: PhysicalFingerprint) -> PhysicalFingerprint {
    fingerprint.field_blocks.sort();
    fingerprint.gauge_complexes.sort();
    fingerprint
        .derivative_content
        .form_strength_coupling_degrees
        .sort();
    fingerprint.canonical_linear_key = sha256(&canonical_lines(&fingerprint, false));
    fingerprint.canonical_completion_key = sha256(&canonical_lines(&fingerprint, true));
    fingerprint
}

pub fn known_catalog() -> Vec<PhysicalFingerprint> {
    let cv = finish(PhysicalFingerprint {
        parent_id: "chiral-vector",
        linear_class: "CV",
        source: "arXiv:1405.0048 Eqs. (32)-(41)",
        worldline_bosons: 8,
        worldline_fermions: 8,
        field_blocks: vec![
            scalar_block(FieldRole::Physical, 2, 0),
            scalar_block(FieldRole::Auxiliary, 3, 2),
            form_block(1, 4),
            fermion_block(),
        ],
        gauge_complexes: vec![gauge_complex(1)],
        derivative_content: DerivativeContent {
            maximum_spacetime_order: 1,
            scalar_gradient_multiplicity: 2,
            form_strength_coupling_degrees: vec![2],
            algebraic_auxiliary_multiplicity: 3,
            temporal_linkage_present: true,
            spatial_linkage_present: true,
        },
        central: central_none(0),
        closure: ClosureCertificate {
            certification: Certification::ExactFourDimensionalClosure,
            component_relations_checked: 612,
            unexplained_residual_relations: 0,
            gauge_residues_retained: true,
        },
        nonlinear_composite_connection: false,
        requires_regular_background_patch: false,
        canonical_linear_key: String::new(),
        canonical_completion_key: String::new(),
    });
    let ct = finish(PhysicalFingerprint {
        parent_id: "chiral-tensor",
        linear_class: "CT",
        source: "arXiv:1405.0048 Eqs. (44)-(53)",
        worldline_bosons: 8,
        worldline_fermions: 8,
        field_blocks: vec![
            scalar_block(FieldRole::Physical, 3, 0),
            scalar_block(FieldRole::Auxiliary, 2, 2),
            form_block(2, 6),
            fermion_block(),
        ],
        gauge_complexes: vec![gauge_complex(2)],
        derivative_content: DerivativeContent {
            maximum_spacetime_order: 1,
            scalar_gradient_multiplicity: 3,
            form_strength_coupling_degrees: vec![3],
            algebraic_auxiliary_multiplicity: 2,
            temporal_linkage_present: true,
            spatial_linkage_present: true,
        },
        central: central_none(0),
        closure: ClosureCertificate {
            certification: Certification::ExactFourDimensionalClosure,
            component_relations_checked: 684,
            unexplained_residual_relations: 0,
            gauge_residues_retained: true,
        },
        nonlinear_composite_connection: false,
        requires_regular_background_patch: false,
        canonical_linear_key: String::new(),
        canonical_completion_key: String::new(),
    });
    let vt = finish(PhysicalFingerprint {
        parent_id: "vector-tensor-one-z",
        linear_class: "VT-one-Z",
        source: "arXiv:1405.0048 Eqs. (59)-(61), (76)-(84); hep-th/9609016 Eqs. (4.5)-(4.6)",
        worldline_bosons: 8,
        worldline_fermions: 8,
        field_blocks: vec![
            scalar_block(FieldRole::Physical, 1, 0),
            scalar_block(FieldRole::Auxiliary, 1, 2),
            form_block(1, 4),
            form_block(2, 6),
            fermion_block(),
        ],
        gauge_complexes: vec![gauge_complex(1), gauge_complex(2)],
        derivative_content: DerivativeContent {
            maximum_spacetime_order: 1,
            scalar_gradient_multiplicity: 1,
            form_strength_coupling_degrees: vec![2, 3],
            algebraic_auxiliary_multiplicity: 1,
            temporal_linkage_present: true,
            spatial_linkage_present: true,
        },
        central: central_one_z(),
        closure: ClosureCertificate {
            certification: Certification::ExactFourDimensionalClosureWithoutCentralBridge,
            component_relations_checked: 720,
            unexplained_residual_relations: 0,
            gauge_residues_retained: true,
        },
        nonlinear_composite_connection: false,
        requires_regular_background_patch: false,
        canonical_linear_key: String::new(),
        canonical_completion_key: String::new(),
    });
    let st = finish(PhysicalFingerprint {
        parent_id: "scalar-tensor-regular-tangent",
        linear_class: "CT-compatible",
        source: "arXiv:2412.16527v3 Eqs. (5.4), (5.11), (5.15), (5.19)",
        worldline_bosons: 8,
        worldline_fermions: 8,
        field_blocks: ct.field_blocks.clone(),
        gauge_complexes: ct.gauge_complexes.clone(),
        derivative_content: ct.derivative_content.clone(),
        central: central_none(1),
        closure: ClosureCertificate {
            certification: Certification::ExactTangentPreflight,
            component_relations_checked: 0,
            unexplained_residual_relations: 0,
            gauge_residues_retained: false,
        },
        nonlinear_composite_connection: true,
        requires_regular_background_patch: true,
        canonical_linear_key: String::new(),
        canonical_completion_key: String::new(),
    });
    vec![cv, ct, vt, st]
}

fn exact_4d(fingerprint: &PhysicalFingerprint) -> bool {
    fingerprint.closure.certification == Certification::ExactFourDimensionalClosure
        && fingerprint.closure.unexplained_residual_relations == 0
}

fn qualification(fingerprint: &PhysicalFingerprint) -> Option<&'static str> {
    match fingerprint.closure.certification {
        Certification::ExactFourDimensionalClosure => None,
        Certification::ExactFourDimensionalClosureWithoutCentralBridge => Some(
            "4D closure and worldline central closure are exact separately; the source-normalization bridge remains partial",
        ),
        Certification::ExactTangentPreflight => Some(
            "structural tangent only; full 4D closure and an exact source-convention intertwiner remain unresolved",
        ),
    }
}

fn missing(query: &ParentageQuery) -> Vec<&'static str> {
    let mut output = Vec::new();
    if query.field_blocks.is_none() {
        output.push("Lorentz field blocks and engineering levels");
    }
    if query.gauge_complexes.is_none() {
        output.push("gauge complex and reducibility");
    }
    if query.derivative_content.is_none() {
        output.push("covariant spatial and temporal derivative content");
    }
    if query.central.is_none() {
        output.push("physical central-generator rank and local gauge distinction");
    }
    output
}

fn mismatch(candidate: &PhysicalFingerprint, query: &ParentageQuery) -> Option<RejectionWitness> {
    let witness = |invariant, expected: String, observed: String| RejectionWitness {
        candidate: candidate.parent_id,
        invariant,
        expected,
        observed,
    };
    if let Some(observed) = query.worldline_size {
        let expected = [candidate.worldline_bosons, candidate.worldline_fermions];
        if observed != expected {
            return Some(witness(
                "worldline_size",
                format!("{expected:?}"),
                format!("{observed:?}"),
            ));
        }
    }
    if let Some(observed) = &query.field_blocks {
        let mut observed = observed.clone();
        observed.sort();
        if observed != candidate.field_blocks {
            return Some(witness(
                "lorentz_field_blocks",
                format!("{:?}", candidate.field_blocks),
                format!("{observed:?}"),
            ));
        }
    }
    if let Some(observed) = &query.gauge_complexes {
        let mut observed = observed.clone();
        observed.sort();
        if observed != candidate.gauge_complexes {
            return Some(witness(
                "gauge_complex",
                format!("{:?}", candidate.gauge_complexes),
                format!("{observed:?}"),
            ));
        }
    }
    if let Some(observed) = &query.derivative_content {
        let mut observed = observed.clone();
        observed.form_strength_coupling_degrees.sort();
        if observed != candidate.derivative_content {
            return Some(witness(
                "derivative_content",
                format!("{:?}", candidate.derivative_content),
                format!("{observed:?}"),
            ));
        }
    }
    if let Some(observed) = &query.central {
        if observed != &candidate.central {
            return Some(witness(
                "central_structure",
                format!("{:?}", candidate.central),
                format!("{observed:?}"),
            ));
        }
    }
    if query.require_exact_four_dimensional_closure && !exact_4d(candidate) {
        return Some(witness(
            "certification",
            "exact_four_dimensional_closure".into(),
            format!("{:?}", candidate.closure.certification),
        ));
    }
    if let Some(observed) = query.nonlinear_composite_connection {
        if observed != candidate.nonlinear_composite_connection {
            return Some(witness(
                "nonlinear_composite_connection",
                candidate.nonlinear_composite_connection.to_string(),
                observed.to_string(),
            ));
        }
    }
    if let Some(observed) = query.requires_regular_background_patch {
        if observed != candidate.requires_regular_background_patch {
            return Some(witness(
                "requires_regular_background_patch",
                candidate.requires_regular_background_patch.to_string(),
                observed.to_string(),
            ));
        }
    }
    None
}

pub fn infer(query: &ParentageQuery, catalog: &[PhysicalFingerprint]) -> InferenceResult {
    let mut compatible = Vec::new();
    let mut rejected = Vec::new();
    let missing_information = missing(query);
    for candidate in catalog {
        if let Some(witness) = mismatch(candidate, query) {
            rejected.push(witness);
        } else {
            compatible.push(CandidateResult {
                parent_id: candidate.parent_id,
                linear_class: candidate.linear_class,
                exact_four_dimensional_fixture: exact_4d(candidate),
                qualification: qualification(candidate),
                missing_information: missing_information.clone(),
            });
        }
    }
    let unique_parent = (compatible.len() == 1).then(|| compatible[0].parent_id);
    let classes: BTreeSet<_> = compatible
        .iter()
        .map(|candidate| candidate.linear_class)
        .collect();
    let unique_linear_class = (classes.len() == 1)
        .then(|| classes.iter().next().copied())
        .flatten();
    let all_required_supplied = missing_information.is_empty();
    let decision = if compatible.is_empty() {
        InferenceDecision::Unsupported
    } else if !all_required_supplied {
        InferenceDecision::Insufficient
    } else if compatible.len() == 1
        && compatible[0].exact_four_dimensional_fixture
        && compatible[0].qualification.is_none()
    {
        InferenceDecision::Identified
    } else {
        InferenceDecision::Compatible
    };
    let interpretation = if compatible.is_empty() {
        "No catalog fixture matches; inspect the exact rejection witnesses."
    } else if decision == InferenceDecision::Identified {
        "The supplied physical invariants identify one catalog parent."
    } else if compatible.len() == 1 {
        "One catalog candidate matches, subject to the candidate's stated qualification."
    } else if unique_linear_class.is_some() {
        "Several completions share one linear class; nonlinear completion data are required."
    } else {
        "The supplied data are insufficient; missing physical invariants are listed per candidate."
    };
    InferenceResult {
        decision,
        compatible,
        rejected,
        unique_parent,
        unique_linear_class,
        interpretation,
    }
}

fn complete_query(fingerprint: &PhysicalFingerprint) -> ParentageQuery {
    ParentageQuery {
        worldline_size: Some([fingerprint.worldline_bosons, fingerprint.worldline_fermions]),
        field_blocks: Some(fingerprint.field_blocks.clone()),
        gauge_complexes: Some(fingerprint.gauge_complexes.clone()),
        derivative_content: Some(fingerprint.derivative_content.clone()),
        central: Some(fingerprint.central.clone()),
        require_exact_four_dimensional_closure: false,
        nonlinear_composite_connection: None,
        requires_regular_background_patch: None,
    }
}

pub fn build() -> ParentageArtifact {
    let catalog = known_catalog();
    let cv = &catalog[0];
    let ct = &catalog[1];
    let vt = &catalog[2];
    let st = &catalog[3];

    let worldline_only_query = infer(
        &ParentageQuery {
            worldline_size: Some([8, 8]),
            ..ParentageQuery::default()
        },
        &catalog,
    );
    let chiral_vector_query = infer(&complete_query(cv), &catalog);
    let chiral_tensor_query = infer(&complete_query(ct), &catalog);
    let mut ct_structural = complete_query(ct);
    ct_structural.central = None;
    let ct_structural_query = infer(&ct_structural, &catalog);
    let vector_tensor_query = infer(&complete_query(vt), &catalog);
    let mut scalar_completion = complete_query(st);
    scalar_completion.nonlinear_composite_connection = Some(true);
    scalar_completion.requires_regular_background_patch = Some(true);
    let scalar_tensor_completion_query = infer(&scalar_completion, &catalog);

    let mut wrong_cv_gauge = complete_query(cv);
    wrong_cv_gauge.gauge_complexes = Some(vec![gauge_complex(2)]);
    let wrong_cv = infer(&wrong_cv_gauge, &catalog);
    let mut missing_vt_central = complete_query(vt);
    missing_vt_central.central = Some(central_none(0));
    let wrong_vt = infer(&missing_vt_central, &catalog);
    let mut wrong_ct_heights = complete_query(ct);
    let mut blocks = ct.field_blocks.clone();
    blocks
        .iter_mut()
        .filter(|block| block.role == FieldRole::Auxiliary)
        .for_each(|block| block.engineering_level_twice = 0);
    wrong_ct_heights.field_blocks = Some(blocks);
    let wrong_ct = infer(&wrong_ct_heights, &catalog);
    let mutation_controls = vec![
        MutationControl {
            mutation: "replace CV one-form gauge complex with a two-form complex",
            rejected_parent: "chiral-vector",
            witness_invariant: "gauge_complex",
            passed: wrong_cv
                .rejected
                .iter()
                .any(|item| item.candidate == "chiral-vector" && item.invariant == "gauge_complex"),
        },
        MutationControl {
            mutation: "erase the vector-tensor physical central generator",
            rejected_parent: "vector-tensor-one-z",
            witness_invariant: "central_structure",
            passed: wrong_vt.rejected.iter().any(|item| {
                item.candidate == "vector-tensor-one-z" && item.invariant == "central_structure"
            }),
        },
        MutationControl {
            mutation: "lower CT auxiliary scalars to the physical engineering level",
            rejected_parent: "chiral-tensor",
            witness_invariant: "lorentz_field_blocks",
            passed: wrong_ct.rejected.iter().any(|item| {
                item.candidate == "chiral-tensor" && item.invariant == "lorentz_field_blocks"
            }),
        },
    ];

    let mut linear_classes: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
    for fingerprint in &catalog {
        linear_classes
            .entry(fingerprint.linear_class)
            .or_default()
            .push(fingerprint.parent_id);
    }
    let control_fixtures_pass = crate::chiral_vector_4d::verify().passed
        && crate::chiral_tensor_4d::verify().passed
        && crate::vector_tensor_4d::verify().passed
        && crate::vector_tensor_central_charge::build()
            .validation
            .passed
        && crate::scalar_tensor_tangent::build().validation.passed;
    let passed = control_fixtures_pass
        && catalog
            .iter()
            .all(|item| !item.canonical_linear_key.is_empty())
        && cv.canonical_linear_key != ct.canonical_linear_key
        && ct.canonical_linear_key == st.canonical_linear_key
        && ct.canonical_completion_key != st.canonical_completion_key
        && vt.central.physical_worldline_generators == 1
        && worldline_only_query.compatible.len() == 4
        && chiral_vector_query.unique_parent == Some("chiral-vector")
        && chiral_tensor_query.unique_parent == Some("chiral-tensor")
        && chiral_tensor_query.decision == InferenceDecision::Identified
        && ct_structural_query.decision == InferenceDecision::Insufficient
        && ct_structural_query.compatible.len() == 2
        && vector_tensor_query.unique_parent == Some("vector-tensor-one-z")
        && vector_tensor_query.decision == InferenceDecision::Compatible
        && scalar_tensor_completion_query.unique_parent == Some("scalar-tensor-regular-tangent")
        && scalar_tensor_completion_query.decision == InferenceDecision::Compatible
        && mutation_controls.iter().all(|control| control.passed);

    ParentageArtifact {
        schema_version: SCHEMA_VERSION,
        equivalence_scope: vec![
            "arbitrary invertible changes of basis within a fixed statistics, Lorentz, role, and engineering-level block",
            "permutations and sign changes of component labels and supercharge labels",
            "gauge-potential presentations with the same differential-form complex and reducibility depth",
            "spatial orientation changes that preserve differential-form degree",
        ],
        completeness_boundary: "The catalog invariants are exact discriminants for the four retained fixtures. They are not a complete canonical form for arbitrary tuples of derivative matrices, and no parent is inferred outside the catalog.",
        catalog,
        linear_classes,
        worldline_only_query,
        chiral_vector_query,
        chiral_tensor_query,
        ct_structural_query,
        vector_tensor_query,
        scalar_tensor_completion_query,
        mutation_controls,
        passed,
    }
}

pub fn write_artifact(path: &Path) -> ParentageArtifact {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parentage artifact directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create parentage artifact")),
        &artifact,
    )
    .expect("write parentage artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_catalog_separates_linear_classes_and_completions() {
        let artifact = build();
        assert!(artifact.passed);
        assert_eq!(artifact.worldline_only_query.compatible.len(), 4);
        assert_eq!(
            artifact.chiral_vector_query.unique_parent,
            Some("chiral-vector")
        );
        assert_eq!(
            artifact.chiral_tensor_query.unique_parent,
            Some("chiral-tensor")
        );
        assert_eq!(artifact.ct_structural_query.compatible.len(), 2);
        assert_eq!(
            artifact.vector_tensor_query.unique_parent,
            Some("vector-tensor-one-z")
        );
        assert_eq!(
            artifact.scalar_tensor_completion_query.unique_parent,
            Some("scalar-tensor-regular-tangent")
        );
        assert!(artifact.mutation_controls.iter().all(|item| item.passed));
    }

    #[test]
    fn canonical_keys_ignore_field_order_but_retain_completion_data() {
        let artifact = build();
        let ct = artifact
            .catalog
            .iter()
            .find(|item| item.parent_id == "chiral-tensor")
            .unwrap();
        let st = artifact
            .catalog
            .iter()
            .find(|item| item.parent_id == "scalar-tensor-regular-tangent")
            .unwrap();
        let mut permuted = ct.clone();
        permuted.field_blocks.reverse();
        permuted.gauge_complexes.reverse();
        let permuted = finish(permuted);
        assert_eq!(ct.canonical_linear_key, permuted.canonical_linear_key);
        assert_eq!(ct.canonical_linear_key, st.canonical_linear_key);
        assert_ne!(ct.canonical_completion_key, st.canonical_completion_key);
    }
}
