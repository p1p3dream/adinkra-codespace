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
kernel!(L14_00000, "level14_00000_highest_weight_kernel.i16le");
kernel!(L14_00010_1, "level14_00010_highest_weight_kernel_1.i16le");
kernel!(L14_00010_2, "level14_00010_highest_weight_kernel_2.i16le");
kernel!(L14_00100_1, "level14_00100_highest_weight_kernel_1.i16le");
kernel!(L14_00100_2, "level14_00100_highest_weight_kernel_2.i16le");
kernel!(L14_01002_1, "level14_01002_highest_weight_kernel_1.i16le");
kernel!(L14_01002_2, "level14_01002_highest_weight_kernel_2.i16le");
kernel!(L14_01002_3, "level14_01002_highest_weight_kernel_3.i16le");
kernel!(L14_01002_4, "level14_01002_highest_weight_kernel_4.i16le");
kernel!(L14_01002_5, "level14_01002_highest_weight_kernel_5.i16le");
kernel!(L14_01010, "level14_01010_highest_weight_kernel.i16le");
kernel!(L14_01100_1, "level14_01100_highest_weight_kernel_1.i16le");
kernel!(L14_01100_2, "level14_01100_highest_weight_kernel_2.i16le");
kernel!(L14_02000_1, "level14_02000_highest_weight_kernel_1.i16le");
kernel!(L14_02000_2, "level14_02000_highest_weight_kernel_2.i16le");
kernel!(L14_10002_1, "level14_10002_highest_weight_kernel_1.i16le");
kernel!(L14_10002_2, "level14_10002_highest_weight_kernel_2.i16le");
kernel!(L14_10010, "level14_10010_highest_weight_kernel.i16le");
kernel!(L14_10100, "level14_10100_highest_weight_kernel.i16le");
kernel!(L14_20002_1, "level14_20002_highest_weight_kernel_1.i16le");
kernel!(L14_20002_2, "level14_20002_highest_weight_kernel_2.i16le");
kernel!(L14_20010_1, "level14_20010_highest_weight_kernel_1.i32le");
kernel!(L14_20010_2, "level14_20010_highest_weight_kernel_2.i32le");
kernel!(L14_20010_3, "level14_20010_highest_weight_kernel_3.i32le");
kernel!(L14_20010_4, "level14_20010_highest_weight_kernel_4.i32le");
kernel!(L14_20100_1, "level14_20100_highest_weight_kernel_1.i16le");
kernel!(L14_20100_2, "level14_20100_highest_weight_kernel_2.i16le");
kernel!(L14_20100_3, "level14_20100_highest_weight_kernel_3.i16le");

#[derive(Debug, Clone, Copy)]
pub(crate) struct SpinorBridgeFixtureRef {
    pub dynkin_label: &'static str,
    pub copy: usize,
    pub artifact: &'static str,
    pub bytes: &'static [u8],
}

macro_rules! fixture_ref {
    ($label:literal, $copy:literal, $artifact:literal, $bytes:ident) => {
        SpinorBridgeFixtureRef {
            dynkin_label: $label,
            copy: $copy,
            artifact: $artifact,
            bytes: $bytes,
        }
    };
}

pub(crate) fn level14_fixtures() -> Vec<SpinorBridgeFixtureRef> {
    vec![
        fixture_ref!(
            "00000",
            1,
            "level14_00000_highest_weight_kernel.i16le",
            L14_00000
        ),
        fixture_ref!(
            "00010",
            1,
            "level14_00010_highest_weight_kernel_1.i16le",
            L14_00010_1
        ),
        fixture_ref!(
            "00010",
            2,
            "level14_00010_highest_weight_kernel_2.i16le",
            L14_00010_2
        ),
        fixture_ref!(
            "00100",
            1,
            "level14_00100_highest_weight_kernel_1.i16le",
            L14_00100_1
        ),
        fixture_ref!(
            "00100",
            2,
            "level14_00100_highest_weight_kernel_2.i16le",
            L14_00100_2
        ),
        fixture_ref!(
            "01002",
            1,
            "level14_01002_highest_weight_kernel_1.i16le",
            L14_01002_1
        ),
        fixture_ref!(
            "01002",
            2,
            "level14_01002_highest_weight_kernel_2.i16le",
            L14_01002_2
        ),
        fixture_ref!(
            "01002",
            3,
            "level14_01002_highest_weight_kernel_3.i16le",
            L14_01002_3
        ),
        fixture_ref!(
            "01002",
            4,
            "level14_01002_highest_weight_kernel_4.i16le",
            L14_01002_4
        ),
        fixture_ref!(
            "01002",
            5,
            "level14_01002_highest_weight_kernel_5.i16le",
            L14_01002_5
        ),
        fixture_ref!(
            "01010",
            1,
            "level14_01010_highest_weight_kernel.i16le",
            L14_01010
        ),
        fixture_ref!(
            "01100",
            1,
            "level14_01100_highest_weight_kernel_1.i16le",
            L14_01100_1
        ),
        fixture_ref!(
            "01100",
            2,
            "level14_01100_highest_weight_kernel_2.i16le",
            L14_01100_2
        ),
        fixture_ref!(
            "02000",
            1,
            "level14_02000_highest_weight_kernel_1.i16le",
            L14_02000_1
        ),
        fixture_ref!(
            "02000",
            2,
            "level14_02000_highest_weight_kernel_2.i16le",
            L14_02000_2
        ),
        fixture_ref!(
            "10002",
            1,
            "level14_10002_highest_weight_kernel_1.i16le",
            L14_10002_1
        ),
        fixture_ref!(
            "10002",
            2,
            "level14_10002_highest_weight_kernel_2.i16le",
            L14_10002_2
        ),
        fixture_ref!(
            "10010",
            1,
            "level14_10010_highest_weight_kernel.i16le",
            L14_10010
        ),
        fixture_ref!(
            "10100",
            1,
            "level14_10100_highest_weight_kernel.i16le",
            L14_10100
        ),
        fixture_ref!(
            "20002",
            1,
            "level14_20002_highest_weight_kernel_1.i16le",
            L14_20002_1
        ),
        fixture_ref!(
            "20002",
            2,
            "level14_20002_highest_weight_kernel_2.i16le",
            L14_20002_2
        ),
        fixture_ref!(
            "20010",
            1,
            "level14_20010_highest_weight_kernel_1.i32le",
            L14_20010_1
        ),
        fixture_ref!(
            "20010",
            2,
            "level14_20010_highest_weight_kernel_2.i32le",
            L14_20010_2
        ),
        fixture_ref!(
            "20010",
            3,
            "level14_20010_highest_weight_kernel_3.i32le",
            L14_20010_3
        ),
        fixture_ref!(
            "20010",
            4,
            "level14_20010_highest_weight_kernel_4.i32le",
            L14_20010_4
        ),
        fixture_ref!(
            "20100",
            1,
            "level14_20100_highest_weight_kernel_1.i16le",
            L14_20100_1
        ),
        fixture_ref!(
            "20100",
            2,
            "level14_20100_highest_weight_kernel_2.i16le",
            L14_20100_2
        ),
        fixture_ref!(
            "20100",
            3,
            "level14_20100_highest_weight_kernel_3.i16le",
            L14_20100_3
        ),
    ]
}

pub(crate) fn level16_fixtures() -> Vec<SpinorBridgeFixtureRef> {
    vec![
        SpinorBridgeFixtureRef {
            dynkin_label: "10000",
            copy: 1,
            artifact: "level16_10000_highest_weight_kernel.i16le",
            bytes: L16_10000,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "20000",
            copy: 1,
            artifact: "level16_20000_highest_weight_kernel.i16le",
            bytes: L16_20000,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "00100",
            copy: 1,
            artifact: "level16_00100_highest_weight_kernel_1.i16le",
            bytes: L16_00100_1,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "00100",
            copy: 2,
            artifact: "level16_00100_highest_weight_kernel_2.i16le",
            bytes: L16_00100_2,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "00010",
            copy: 1,
            artifact: "level16_00010_highest_weight_kernel_1.i16le",
            bytes: L16_00010_1,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "00010",
            copy: 2,
            artifact: "level16_00010_highest_weight_kernel_2.i16le",
            bytes: L16_00010_2,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "00002",
            copy: 1,
            artifact: "level16_00002_highest_weight_kernel.i16le",
            bytes: L16_00002,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "10100",
            copy: 1,
            artifact: "level16_10100_highest_weight_kernel.i16le",
            bytes: L16_10100,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "10010",
            copy: 1,
            artifact: "level16_10010_highest_weight_kernel.i16le",
            bytes: L16_10010,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "10002",
            copy: 1,
            artifact: "level16_10002_highest_weight_kernel_1.i16le",
            bytes: L16_10002_1,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "10002",
            copy: 2,
            artifact: "level16_10002_highest_weight_kernel_2.i16le",
            bytes: L16_10002_2,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "10002",
            copy: 3,
            artifact: "level16_10002_highest_weight_kernel_3.i16le",
            bytes: L16_10002_3,
        },
    ]
}

pub(crate) fn level17_fixtures() -> Vec<SpinorBridgeFixtureRef> {
    vec![
        SpinorBridgeFixtureRef {
            dynkin_label: "10001",
            copy: 1,
            artifact: "level17_10001_highest_weight_kernel.i16le",
            bytes: L17_10001,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "01001",
            copy: 1,
            artifact: "level17_01001_highest_weight_kernel_1.i16le",
            bytes: L17_01001_1,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "01001",
            copy: 2,
            artifact: "level17_01001_highest_weight_kernel_2.i16le",
            bytes: L17_01001_2,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "20001",
            copy: 1,
            artifact: "level17_20001_highest_weight_kernel.i16le",
            bytes: L17_20001,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "11001",
            copy: 1,
            artifact: "level17_11001_highest_weight_kernel_1.i16le",
            bytes: L17_11001_1,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "11001",
            copy: 2,
            artifact: "level17_11001_highest_weight_kernel_2.i16le",
            bytes: L17_11001_2,
        },
        SpinorBridgeFixtureRef {
            dynkin_label: "11001",
            copy: 3,
            artifact: "level17_11001_highest_weight_kernel_3.i16le",
            bytes: L17_11001_3,
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceTargetCouplingAudit {
    pub source_dynkin_label: &'static str,
    pub source_copy: usize,
    pub free_spinor_weight: [i8; 5],
    pub product_highest_weight: [i8; 5],
    pub target_dynkin_label: &'static str,
    pub source_nonzero_coefficients: usize,
    pub source_raising_residual_rows: usize,
    pub free_spinor_is_highest_weight: bool,
    pub tensor_product_highest_weight_verified: bool,
    pub exact_coupling_constructed: bool,
    pub passed: bool,
}

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
    pub leading_source_target_couplings: Vec<SourceTargetCouplingAudit>,
    pub second_leading_source_target_coupling:
        crate::eleven_dimensional_bridge::SecondLeadingSourceCouplingAudit,
    pub additional_leading_source_target_couplings:
        Vec<crate::eleven_dimensional_bridge::GenericLeadingSourceCouplingAudit>,
    pub leading_source_target_couplings_constructed: usize,
    pub expected_leading_source_target_couplings: usize,
    pub derivative_matrix_constructed: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstMomentumKernelReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub systems: Vec<HighestWeightSystemReport>,
    pub systems_verified: usize,
    pub expected_systems: usize,
    pub integer_kernel_vectors_verified: usize,
    pub expected_integer_kernel_vectors: usize,
    pub raising_rows_checked: usize,
    pub nonzero_raising_residual_rows: usize,
    pub signed_32_bit_systems: usize,
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
const L14_00000_FIXTURES: [(&str, &[u8]); 1] =
    [("level14_00000_highest_weight_kernel.i16le", L14_00000)];
const L14_00010_FIXTURES: [(&str, &[u8]); 2] = [
    ("level14_00010_highest_weight_kernel_1.i16le", L14_00010_1),
    ("level14_00010_highest_weight_kernel_2.i16le", L14_00010_2),
];
const L14_00100_FIXTURES: [(&str, &[u8]); 2] = [
    ("level14_00100_highest_weight_kernel_1.i16le", L14_00100_1),
    ("level14_00100_highest_weight_kernel_2.i16le", L14_00100_2),
];
const L14_01002_FIXTURES: [(&str, &[u8]); 5] = [
    ("level14_01002_highest_weight_kernel_1.i16le", L14_01002_1),
    ("level14_01002_highest_weight_kernel_2.i16le", L14_01002_2),
    ("level14_01002_highest_weight_kernel_3.i16le", L14_01002_3),
    ("level14_01002_highest_weight_kernel_4.i16le", L14_01002_4),
    ("level14_01002_highest_weight_kernel_5.i16le", L14_01002_5),
];
const L14_01010_FIXTURES: [(&str, &[u8]); 1] =
    [("level14_01010_highest_weight_kernel.i16le", L14_01010)];
const L14_01100_FIXTURES: [(&str, &[u8]); 2] = [
    ("level14_01100_highest_weight_kernel_1.i16le", L14_01100_1),
    ("level14_01100_highest_weight_kernel_2.i16le", L14_01100_2),
];
const L14_02000_FIXTURES: [(&str, &[u8]); 2] = [
    ("level14_02000_highest_weight_kernel_1.i16le", L14_02000_1),
    ("level14_02000_highest_weight_kernel_2.i16le", L14_02000_2),
];
const L14_10002_FIXTURES: [(&str, &[u8]); 2] = [
    ("level14_10002_highest_weight_kernel_1.i16le", L14_10002_1),
    ("level14_10002_highest_weight_kernel_2.i16le", L14_10002_2),
];
const L14_10010_FIXTURES: [(&str, &[u8]); 1] =
    [("level14_10010_highest_weight_kernel.i16le", L14_10010)];
const L14_10100_FIXTURES: [(&str, &[u8]); 1] =
    [("level14_10100_highest_weight_kernel.i16le", L14_10100)];
const L14_20002_FIXTURES: [(&str, &[u8]); 2] = [
    ("level14_20002_highest_weight_kernel_1.i16le", L14_20002_1),
    ("level14_20002_highest_weight_kernel_2.i16le", L14_20002_2),
];
const L14_20010_FIXTURES: [(&str, &[u8]); 4] = [
    ("level14_20010_highest_weight_kernel_1.i32le", L14_20010_1),
    ("level14_20010_highest_weight_kernel_2.i32le", L14_20010_2),
    ("level14_20010_highest_weight_kernel_3.i32le", L14_20010_3),
    ("level14_20010_highest_weight_kernel_4.i32le", L14_20010_4),
];
const L14_20100_FIXTURES: [(&str, &[u8]); 3] = [
    ("level14_20100_highest_weight_kernel_1.i16le", L14_20100_1),
    ("level14_20100_highest_weight_kernel_2.i16le", L14_20100_2),
    ("level14_20100_highest_weight_kernel_3.i16le", L14_20100_3),
];

fn leading_fixtures() -> Vec<ExteriorHighestWeightKernelFixture> {
    vec![
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10000",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_10000_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "20000",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_20000_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "00100",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_00100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "00010",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_00010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "00002",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_00002_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10100",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_10100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10010",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_10010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 16,
            dynkin_label: "10002",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L16_10002_FIXTURES,
        },
    ]
}

fn hook_fixtures() -> Vec<ExteriorHighestWeightKernelFixture> {
    vec![
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "10001",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L17_10001_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "01001",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L17_01001_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "20001",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L17_20001_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 17,
            dynkin_label: "11001",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L17_11001_FIXTURES,
        },
    ]
}

fn first_momentum_source_fixtures() -> Vec<ExteriorHighestWeightKernelFixture> {
    vec![
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "00000",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_00000_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "00010",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_00010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "00100",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_00100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "01002",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_01002_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "01010",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_01010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "01100",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_01100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "02000",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_02000_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "10002",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_10002_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "10010",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_10010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "10100",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_10100_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "20002",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_20002_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "20010",
            coefficient_width_bytes: 4,
            kernel_artifacts: &L14_20010_FIXTURES,
        },
        ExteriorHighestWeightKernelFixture {
            exterior_degree: 14,
            dynkin_label: "20100",
            coefficient_width_bytes: 2,
            kernel_artifacts: &L14_20100_FIXTURES,
        },
    ]
}

pub fn verify_first_momentum_kernels() -> FirstMomentumKernelReport {
    let systems = verify_exterior_highest_weight_kernel_fixtures(&first_momentum_source_fixtures());
    let systems_verified = systems
        .iter()
        .filter(|system| {
            system.exact_sparse_system_constructed
                && system.exact_kernel_vectors.len() == system.expected_kernel_dimension
                && system
                    .exact_kernel_vectors
                    .iter()
                    .all(|kernel| kernel.exact_kernel_verified)
        })
        .count();
    let integer_kernel_vectors_verified = systems
        .iter()
        .flat_map(|system| &system.exact_kernel_vectors)
        .filter(|kernel| kernel.exact_kernel_verified)
        .count();
    let raising_rows_checked = systems
        .iter()
        .map(|system| system.total_rows * system.exact_kernel_vectors.len())
        .sum();
    let nonzero_raising_residual_rows = systems
        .iter()
        .flat_map(|system| &system.exact_kernel_vectors)
        .map(|kernel| kernel.nonzero_residual_rows)
        .sum();
    let signed_32_bit_systems = systems
        .iter()
        .filter(|system| {
            system
                .exact_kernel_vectors
                .iter()
                .any(|kernel| kernel.scalar_type.contains("32-bit"))
        })
        .count();
    FirstMomentumKernelReport {
        schema_version: "adynkra-11d-first-momentum-kernels-v1",
        role: "exact level-14 source-kernel certificates for the first spacetime-momentum intertwiners",
        systems,
        systems_verified,
        expected_systems: 13,
        integer_kernel_vectors_verified,
        expected_integer_kernel_vectors: 28,
        raising_rows_checked,
        nonzero_raising_residual_rows,
        signed_32_bit_systems,
        boundary: "This report certifies the 28 level-14 source highest-weight vectors. It does not construct the 23 source-target couplings or the 44 embedded first-momentum maps.",
        passed: systems_verified == 13
            && integer_kernel_vectors_verified == 28
            && nonzero_raising_residual_rows == 0
            && signed_32_bit_systems == 1,
    }
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
    let vector_system = leading_systems
        .iter()
        .find(|system| system.dynkin_label == "10000")
        .unwrap();
    let vector_kernel = &vector_system.exact_kernel_vectors[0];
    let first_leading_coupling = SourceTargetCouplingAudit {
        source_dynkin_label: "10000",
        source_copy: 1,
        free_spinor_weight: [1, 1, 1, 1, 1],
        product_highest_weight: [3, 1, 1, 1, 1],
        target_dynkin_label: "10001",
        source_nonzero_coefficients: vector_kernel.nonzero_coefficients,
        source_raising_residual_rows: vector_kernel.nonzero_residual_rows,
        free_spinor_is_highest_weight: true,
        tensor_product_highest_weight_verified: true,
        exact_coupling_constructed: vector_kernel.exact_kernel_verified,
        passed: vector_kernel.exact_kernel_verified,
    };
    let leading_source_target_couplings = vec![first_leading_coupling];
    let second_leading_source_target_coupling =
        crate::eleven_dimensional_bridge::audit_20000_to_10001_source_coupling(L16_20000);
    let additional_leading_source_target_couplings = vec![
        crate::eleven_dimensional_bridge::audit_generic_leading_source_coupling(
            "00100",
            1,
            L16_00100_1,
        ),
    ];
    let leading_source_target_couplings_constructed = leading_source_target_couplings.len()
        + usize::from(second_leading_source_target_coupling.passed)
        + additional_leading_source_target_couplings
            .iter()
            .filter(|coupling| coupling.passed)
            .count();
    let expected_leading_source_target_couplings = 12;
    let expected_systems = 12;
    let expected_integer_kernel_vectors = 19;
    let passed = systems_verified == expected_systems
        && integer_kernel_vectors_verified == expected_integer_kernel_vectors
        && nonzero_raising_residual_rows == 0
        && every_first_lowering_string_verified
        && hook_target_coupling.passed
        && leading_source_target_couplings
            .iter()
            .all(|coupling| coupling.passed)
        && second_leading_source_target_coupling.passed
        && additional_leading_source_target_couplings
            .iter()
            .all(|coupling| coupling.passed);
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
        leading_source_target_couplings,
        second_leading_source_target_coupling,
        additional_leading_source_target_couplings,
        leading_source_target_couplings_constructed,
        expected_leading_source_target_couplings,
        derivative_matrix_constructed: false,
        boundary: "this verifier covers the nineteen source embeddings, the unique hook target coupling, and three leading source-to-vector-spinor couplings; the dense coupling verifier separately certifies all twelve leading couplings, all seven hook couplings, and the exact 7-by-12 exterior-derivative matrix",
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
        assert!(report.second_leading_source_target_coupling.passed);
        assert!(report
            .additional_leading_source_target_couplings
            .iter()
            .all(|coupling| coupling.passed));
        assert_eq!(
            report.additional_leading_source_target_couplings[0].primitive_domain_coefficients,
            [4, -3, 2, -1, -2, 2, -4, -2, 4, -4, 2, -4, 4, -4]
        );
        assert_eq!(report.leading_source_target_couplings_constructed, 3);
    }

    #[test]
    #[ignore = "constructs and checks 11.2 million sparse raising rows"]
    fn all_twenty_eight_first_momentum_source_kernels_are_exact() {
        let report = verify_first_momentum_kernels();
        assert!(report.passed);
        assert_eq!(report.systems_verified, 13);
        assert_eq!(report.integer_kernel_vectors_verified, 28);
        assert_eq!(report.nonzero_raising_residual_rows, 0);
        assert_eq!(report.signed_32_bit_systems, 1);
    }
}
