//! Exact integer highest-weight kernels for the direct 11D spinor bridge.
//!
//! Numerical sparse eigensolves propose the stored integer vectors.  This
//! module is the primary verifier: it reconstructs every simple-root raising
//! equation in Rust and checks every stored coefficient with integer
//! arithmetic.

use crate::eleven_dimensional_bridge::{
    verify_exterior_highest_weight_kernel_fixtures, ExteriorHighestWeightKernelFixture,
    HighestWeightSystemReport,
};
use serde::Serialize;

macro_rules! kernel {
    ($name:ident, $path:literal) => {
        const $name: &[u8] =
            include_bytes!(concat!("../data/eleven_dimensional_spinor_bridge/", $path));
    };
}

kernel!(L16_10000, "level16_10000_highest_weight_kernel.i16le");
kernel!(L16_20000, "level16_20000_highest_weight_kernel.i16le");
kernel!(L16_00100_1, "level16_00100_highest_weight_kernel_1.i16le");
kernel!(L16_00100_2, "level16_00100_highest_weight_kernel_2.i16le");
kernel!(L16_00010_1, "level16_00010_highest_weight_kernel_1.i16le");
kernel!(L16_00010_2, "level16_00010_highest_weight_kernel_2.i16le");
kernel!(L16_00002, "level16_00002_highest_weight_kernel.i16le");
kernel!(L16_10100, "level16_10100_highest_weight_kernel.i16le");
kernel!(L16_10010, "level16_10010_highest_weight_kernel.i16le");
kernel!(L16_10002_1, "level16_10002_highest_weight_kernel_1.i16le");
kernel!(L16_10002_2, "level16_10002_highest_weight_kernel_2.i16le");
kernel!(L16_10002_3, "level16_10002_highest_weight_kernel_3.i16le");
kernel!(L17_10001, "level17_10001_highest_weight_kernel.i16le");
kernel!(L17_01001_1, "level17_01001_highest_weight_kernel_1.i16le");
kernel!(L17_01001_2, "level17_01001_highest_weight_kernel_2.i16le");
kernel!(L17_20001, "level17_20001_highest_weight_kernel.i16le");
kernel!(L17_11001_1, "level17_11001_highest_weight_kernel_1.i16le");
kernel!(L17_11001_2, "level17_11001_highest_weight_kernel_2.i16le");
kernel!(L17_11001_3, "level17_11001_highest_weight_kernel_3.i16le");

#[derive(Debug, Clone, Serialize)]
pub struct DirectSpinorKernelReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub leading_systems: Vec<HighestWeightSystemReport>,
    pub hook_systems: Vec<HighestWeightSystemReport>,
    pub systems_verified: usize,
    pub expected_systems: usize,
    pub integer_kernel_vectors_verified: usize,
    pub expected_integer_kernel_vectors: usize,
    pub raising_rows_checked: usize,
    pub nonzero_raising_residual_rows: usize,
    pub every_first_lowering_string_verified: bool,
    pub hook_target_coupling: crate::eleven_dimensional_bridge::DirectHookTargetCouplingAudit,
    pub derivative_matrix_constructed: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

const L16_10000_FIXTURES: [(&str, &[u8]); 1] =
    [("level16_10000_highest_weight_kernel.i16le", L16_10000)];
const L16_20000_FIXTURES: [(&str, &[u8]); 1] =
    [("level16_20000_highest_weight_kernel.i16le", L16_20000)];
const L16_00100_FIXTURES: [(&str, &[u8]); 2] = [
    ("level16_00100_highest_weight_kernel_1.i16le", L16_00100_1),
    ("level16_00100_highest_weight_kernel_2.i16le", L16_00100_2),
];
const L16_00010_FIXTURES: [(&str, &[u8]); 2] = [
    ("level16_00010_highest_weight_kernel_1.i16le", L16_00010_1),
    ("level16_00010_highest_weight_kernel_2.i16le", L16_00010_2),
];
const L16_00002_FIXTURES: [(&str, &[u8]); 1] =
    [("level16_00002_highest_weight_kernel.i16le", L16_00002)];
const L16_10100_FIXTURES: [(&str, &[u8]); 1] =
    [("level16_10100_highest_weight_kernel.i16le", L16_10100)];
const L16_10010_FIXTURES: [(&str, &[u8]); 1] =
    [("level16_10010_highest_weight_kernel.i16le", L16_10010)];
const L16_10002_FIXTURES: [(&str, &[u8]); 3] = [
    ("level16_10002_highest_weight_kernel_1.i16le", L16_10002_1),
    ("level16_10002_highest_weight_kernel_2.i16le", L16_10002_2),
    ("level16_10002_highest_weight_kernel_3.i16le", L16_10002_3),
];
const L17_10001_FIXTURES: [(&str, &[u8]); 1] =
    [("level17_10001_highest_weight_kernel.i16le", L17_10001)];
const L17_01001_FIXTURES: [(&str, &[u8]); 2] = [
    ("level17_01001_highest_weight_kernel_1.i16le", L17_01001_1),
    ("level17_01001_highest_weight_kernel_2.i16le", L17_01001_2),
];
const L17_20001_FIXTURES: [(&str, &[u8]); 1] =
    [("level17_20001_highest_weight_kernel.i16le", L17_20001)];
const L17_11001_FIXTURES: [(&str, &[u8]); 3] = [
    ("level17_11001_highest_weight_kernel_1.i16le", L17_11001_1),
    ("level17_11001_highest_weight_kernel_2.i16le", L17_11001_2),
    ("level17_11001_highest_weight_kernel_3.i16le", L17_11001_3),
];

fn leading_fixtures() -> Vec<ExteriorHighestWeightKernelFixture> {
    vec![
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10000",
            kernel_artifacts: &L16_10000_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "20000",
            kernel_artifacts: &L16_20000_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "00100",
            kernel_artifacts: &L16_00100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "00010",
            kernel_artifacts: &L16_00010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "00002",
            kernel_artifacts: &L16_00002_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10100",
            kernel_artifacts: &L16_10100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10010",
            kernel_artifacts: &L16_10010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10002",
            kernel_artifacts: &L16_10002_FIXTURES,
        },
    ]
}

fn hook_fixtures() -> Vec<ExteriorHighestWeightKernelFixture> {
    vec![
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "10001",
            kernel_artifacts: &L17_10001_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "01001",
            kernel_artifacts: &L17_01001_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "20001",
            kernel_artifacts: &L17_20001_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "11001",
            kernel_artifacts: &L17_11001_FIXTURES,
        },
    ]
}

pub fn verify() -> DirectSpinorKernelReport {
    let leading_systems = verify_exterior_highest_weight_kernel_fixtures(&leading_fixtures());
    let hook_systems = verify_exterior_highest_weight_kernel_fixtures(&hook_fixtures());
    let systems_verified = leading_systems
        .iter()
        .chain(&hook_systems)
        .filter(|system| {
            system.exact_sparse_system_constructed
                && system.exact_kernel_vectors.len() == system.expected_kernel_dimension
                && system
                    .exact_kernel_vectors
                    .iter()
                    .all(|kernel| kernel.exact_kernel_verified)
        })
        .count();
    let integer_kernel_vectors_verified = leading_systems
        .iter()
        .chain(&hook_systems)
        .flat_map(|system| &system.exact_kernel_vectors)
        .filter(|kernel| kernel.exact_kernel_verified)
        .count();
    let raising_rows_checked = leading_systems
        .iter()
        .chain(&hook_systems)
        .map(|system| system.total_rows * system.exact_kernel_vectors.len())
        .sum();
    let nonzero_raising_residual_rows = leading_systems
        .iter()
        .chain(&hook_systems)
        .flat_map(|system| &system.exact_kernel_vectors)
        .map(|kernel| kernel.nonzero_residual_rows)
        .sum();
    let every_first_lowering_string_verified = leading_systems
        .iter()
        .chain(&hook_systems)
        .flat_map(|system| &system.exact_kernel_vectors)
        .flat_map(|kernel| &kernel.first_lowering_descendants)
        .all(|check| check.matches_highest_weight_string);
    let hook_target_coupling =
        crate::eleven_dimensional_bridge::audit_direct_hook_target_coupling();
    let expected_systems = 12;
    let expected_integer_kernel_vectors = 19;
    let passed = systems_verified == expected_systems
        && integer_kernel_vectors_verified == expected_integer_kernel_vectors
        && nonzero_raising_residual_rows == 0
        && every_first_lowering_string_verified
        && hook_target_coupling.passed;
    DirectSpinorKernelReport {
        schema_version: "adynkra-11d-spinor-bridge-kernels-v1",
        role: "exact Rust verification of the decomposed direct-spinor source embeddings",
        leading_systems,
        hook_systems,
        systems_verified,
        expected_systems,
        integer_kernel_vectors_verified,
        expected_integer_kernel_vectors,
        raising_rows_checked,
        nonzero_raising_residual_rows,
        every_first_lowering_string_verified,
        hook_target_coupling,
        derivative_matrix_constructed: false,
        boundary:
            "this verifies the nineteen source embeddings; the Clebsch-Gordan coupling to the target vector-spinor and the 12-by-7 exterior-derivative matrix remain separate",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "constructs and checks 10.8 million sparse raising rows"]
    fn all_nineteen_integer_kernels_are_exact() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.systems_verified, 12);
        assert_eq!(report.integer_kernel_vectors_verified, 19);
        assert_eq!(report.nonzero_raising_residual_rows, 0);
    }
}
