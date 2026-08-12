//! Source-fixed four-dimensional fingerprints retained before worldline reduction.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DerivativeOperatorFingerprint {
    pub transformation_relations: usize,
    pub transformation_terms: usize,
    pub algebraic_terms: usize,
    pub temporal_derivative_terms: usize,
    pub spatial_derivative_terms: usize,
    pub relations_with_spatial_derivatives: usize,
    pub canonical_spatial_operator_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GaugeFingerprint {
    pub potential_form_degree: usize,
    pub potential_components_before_gauge_fixing: usize,
    pub temporal_gauge_components_removed: usize,
    pub potential_components_after_temporal_gauge: usize,
    pub field_strength_form_degree: usize,
    pub gauge_parameter_form_degree: usize,
    pub potential_closure_relations: usize,
    pub nonzero_gauge_residue_relations: usize,
    pub gauge_residue_terms: usize,
    pub temporal_derivative_terms: usize,
    pub spatial_derivative_terms: usize,
    pub residue_source_fields: Vec<String>,
    pub canonical_gauge_residue_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MultipletFingerprint {
    pub name: &'static str,
    pub source: &'static str,
    pub raw_component_fields: usize,
    pub worldline_bosons: usize,
    pub worldline_fermions: usize,
    pub derivative_operator: DerivativeOperatorFingerprint,
    pub gauge: GaugeFingerprint,
}

#[derive(Clone, Debug, Serialize)]
pub struct HigherDimensionalComparison {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub chiral_vector: MultipletFingerprint,
    pub chiral_tensor: MultipletFingerprint,
    pub same_worldline_size: bool,
    pub reduced_matrices_identical_in_source_basis: bool,
    pub spatial_operators_identical_in_source_basis: bool,
    pub gauge_residues_identical_in_source_basis: bool,
    pub minimum_candidate_data: Vec<&'static str>,
    pub passed: bool,
    pub boundary: &'static str,
}

pub(crate) fn sha256_lines(lines: &[String]) -> String {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

pub fn build() -> HigherDimensionalComparison {
    let chiral_vector = crate::chiral_vector_4d::higher_dimensional_fingerprint();
    let chiral_tensor = crate::chiral_tensor_4d::higher_dimensional_fingerprint();
    let cv_matrices = crate::chiral_vector_4d::build().reduced_l_matrices;
    let ct_matrices = crate::chiral_tensor_4d::build().reduced_l_matrices;
    let same_worldline_size = chiral_vector.worldline_bosons == chiral_tensor.worldline_bosons
        && chiral_vector.worldline_fermions == chiral_tensor.worldline_fermions;
    let reduced_matrices_identical_in_source_basis = cv_matrices == ct_matrices;
    let spatial_operators_identical_in_source_basis = chiral_vector
        .derivative_operator
        .canonical_spatial_operator_sha256
        == chiral_tensor
            .derivative_operator
            .canonical_spatial_operator_sha256;
    let gauge_residues_identical_in_source_basis =
        chiral_vector.gauge.canonical_gauge_residue_sha256
            == chiral_tensor.gauge.canonical_gauge_residue_sha256;
    let minimum_candidate_data = vec![
        "Lorentz representation assigned to every bosonic and fermionic component",
        "complete spatial-derivative linkage coefficients in the source basis",
        "gauge-potential form degree and complete gauge transformation",
        "gauge-invariant field strength and any Bianchi identity",
        "temporal gauge choice, surviving zero modes, and field-to-node reduction map",
        "closure of every supercharge pair on every component, including gauge residues",
    ];
    let passed = crate::chiral_vector_4d::verify().passed
        && crate::chiral_tensor_4d::verify().passed
        && same_worldline_size
        && !reduced_matrices_identical_in_source_basis
        && !spatial_operators_identical_in_source_basis
        && !gauge_residues_identical_in_source_basis
        && chiral_vector.gauge.potential_form_degree == 1
        && chiral_tensor.gauge.potential_form_degree == 2;
    HigherDimensionalComparison {
        schema_version: "higher-dimensional-fingerprint-v1",
        title: "Four-dimensional fingerprints of the chiral-vector and chiral-tensor positive controls",
        chiral_vector,
        chiral_tensor,
        same_worldline_size,
        reduced_matrices_identical_in_source_basis,
        spatial_operators_identical_in_source_basis,
        gauge_residues_identical_in_source_basis,
        minimum_candidate_data,
        passed,
        boundary: "These fingerprints compare two published source-basis realizations. Their hashes are reproducibility identifiers, not basis-independent physical invariants. No four-dimensional parent is inferred for VM1, VM2, or VM3.",
    }
}

pub fn write_artifact(path: &Path) -> HigherDimensionalComparison {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create higher-dimensional fingerprint artifact")),
        &artifact,
    )
    .expect("write higher-dimensional fingerprint artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_controls_retain_different_four_dimensional_data() {
        let artifact = build();
        assert!(artifact.same_worldline_size);
        assert!(!artifact.reduced_matrices_identical_in_source_basis);
        assert!(!artifact.spatial_operators_identical_in_source_basis);
        assert!(!artifact.gauge_residues_identical_in_source_basis);
        assert_eq!(artifact.chiral_vector.gauge.potential_form_degree, 1);
        assert_eq!(artifact.chiral_tensor.gauge.potential_form_degree, 2);
        assert_eq!(
            artifact
                .chiral_vector
                .derivative_operator
                .spatial_derivative_terms,
            192
        );
        assert_eq!(
            artifact
                .chiral_tensor
                .derivative_operator
                .spatial_derivative_terms,
            192
        );
        assert_eq!(artifact.chiral_vector.gauge.gauge_residue_terms, 128);
        assert_eq!(artifact.chiral_tensor.gauge.gauge_residue_terms, 384);
        assert_eq!(
            artifact
                .chiral_vector
                .derivative_operator
                .canonical_spatial_operator_sha256,
            "0d477476173478eaa1597aa71651177142c2809a9993dcc4f3934cbbe48c76e0"
        );
        assert_eq!(
            artifact
                .chiral_tensor
                .derivative_operator
                .canonical_spatial_operator_sha256,
            "2670ba5f78f83126be375c8799ab9ff0f812a6d016f658b659f8974b1b9bc383"
        );
        assert_eq!(
            artifact.chiral_vector.gauge.canonical_gauge_residue_sha256,
            "5ca5a1a3bdb62f0752d2e1b3d59ca9475bc68aa0f8855a92edc7565df765a7aa"
        );
        assert_eq!(
            artifact.chiral_tensor.gauge.canonical_gauge_residue_sha256,
            "6a3086cbf284517a5fe9ab70c9070dbfd7f384f77bec48628d90dc17f43b0504"
        );
        assert!(artifact.passed);
    }
}
