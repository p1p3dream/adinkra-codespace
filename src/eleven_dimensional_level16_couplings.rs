//! Fixed work list and representation-level gates for the 11D level-16
//! source-to-vector-spinor coupling certificates.

use serde::Serialize;
use std::collections::BTreeMap;

const TARGET_DYNKIN_LABEL: &str = "10001";
const GOLDEN_COMMIT: &str = "89f20fc";

#[derive(Debug, Clone, Serialize)]
pub struct Level16FixtureManifestEntry {
    pub source_dynkin_label: &'static str,
    pub copy: usize,
    pub artifact: &'static str,
    pub byte_length: usize,
    pub coefficient_count: usize,
    pub signed_little_endian_bits: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorMultiplicityAudit {
    pub source_dynkin_label: &'static str,
    pub target_dynkin_label: &'static str,
    pub target_multiplicity_in_source_tensor_spinor: usize,
    pub multiplicity_one: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Level16CouplingPrecheckReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub exterior_degree: usize,
    pub spinor_dimension: usize,
    pub target_dynkin_label: &'static str,
    pub distinct_source_irreps: usize,
    pub expected_distinct_source_irreps: usize,
    pub embedded_source_copies: usize,
    pub expected_embedded_source_copies: usize,
    pub fixtures: Vec<Level16FixtureManifestEntry>,
    pub copy_counts_by_irrep: BTreeMap<&'static str, usize>,
    pub tensor_multiplicities: Vec<TensorMultiplicityAudit>,
    pub every_target_multiplicity_is_one: bool,
    pub golden_source_dynkin_label: &'static str,
    pub golden_source_copy: usize,
    pub golden_commit: &'static str,
    pub experimentally_validated_source_dynkin_label: &'static str,
    pub experimentally_validated_source_copy: usize,
    pub experimentally_validated_checkpoint_required: bool,
    pub passed: bool,
}

pub fn verify() -> Level16CouplingPrecheckReport {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut copy_counts_by_irrep = BTreeMap::new();
    for fixture in &fixtures {
        *copy_counts_by_irrep
            .entry(fixture.dynkin_label)
            .or_insert(0) += 1;
    }
    let tensor_multiplicities = copy_counts_by_irrep
        .keys()
        .copied()
        .map(|source_dynkin_label| {
            let target_multiplicity_in_source_tensor_spinor =
                crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
                    .iter()
                    .filter(|(target, _)| target == TARGET_DYNKIN_LABEL)
                    .count();
            TensorMultiplicityAudit {
                source_dynkin_label,
                target_dynkin_label: TARGET_DYNKIN_LABEL,
                target_multiplicity_in_source_tensor_spinor,
                multiplicity_one: target_multiplicity_in_source_tensor_spinor == 1,
            }
        })
        .collect::<Vec<_>>();
    let every_target_multiplicity_is_one = tensor_multiplicities
        .iter()
        .all(|audit| audit.multiplicity_one);
    let manifest = fixtures
        .iter()
        .map(|fixture| Level16FixtureManifestEntry {
            source_dynkin_label: fixture.dynkin_label,
            copy: fixture.copy,
            artifact: fixture.artifact,
            byte_length: fixture.bytes.len(),
            coefficient_count: fixture.bytes.len() / 2,
            signed_little_endian_bits: 16,
        })
        .collect::<Vec<_>>();
    let expected_counts = BTreeMap::from([
        ("00002", 1),
        ("00010", 2),
        ("00100", 2),
        ("10000", 1),
        ("10002", 3),
        ("10010", 1),
        ("10100", 1),
        ("20000", 1),
    ]);
    let fixture_encoding_valid = fixtures
        .iter()
        .all(|fixture| !fixture.bytes.is_empty() && fixture.bytes.len() % 2 == 0);
    let distinct_source_irreps = copy_counts_by_irrep.len();
    let embedded_source_copies = fixtures.len();
    let passed = distinct_source_irreps == 8
        && embedded_source_copies == 12
        && copy_counts_by_irrep == expected_counts
        && fixture_encoding_valid
        && every_target_multiplicity_is_one;
    Level16CouplingPrecheckReport {
        schema_version: "adynkra-11d-level16-coupling-precheck-v1",
        role: "fixed source manifest and multiplicity-one gate for level-16 couplings into (10001)",
        exterior_degree: 16,
        spinor_dimension: 32,
        target_dynkin_label: TARGET_DYNKIN_LABEL,
        distinct_source_irreps,
        expected_distinct_source_irreps: 8,
        embedded_source_copies,
        expected_embedded_source_copies: 12,
        fixtures: manifest,
        copy_counts_by_irrep,
        tensor_multiplicities,
        every_target_multiplicity_is_one,
        golden_source_dynkin_label: "20000",
        golden_source_copy: 1,
        golden_commit: GOLDEN_COMMIT,
        experimentally_validated_source_dynkin_label: "00100",
        experimentally_validated_source_copy: 1,
        experimentally_validated_checkpoint_required: true,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_work_list_and_multiplicity_gate_pass() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.distinct_source_irreps, 8);
        assert_eq!(report.embedded_source_copies, 12);
        assert!(report.every_target_multiplicity_is_one);
        assert_eq!(
            report
                .tensor_multiplicities
                .iter()
                .filter(|audit| audit.multiplicity_one)
                .count(),
            8
        );
    }
}
