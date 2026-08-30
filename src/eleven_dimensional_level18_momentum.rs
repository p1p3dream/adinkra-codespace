//! Exact level-18 source kernels and the first-momentum source-gauge screen.
//!
//! Appendix F of arXiv:2002.08502 supplies levels zero through sixteen and
//! states the complementary-level inventory used here.  The complex
//! 32-dimensional B5 spinor is self-dual and has trivial
//! determinant.  Consequently `Lambda^18 S` is equivariantly isomorphic to
//! `Lambda^14 S`.  This module constructs that isomorphism in the committed
//! weight basis, applies it to every relevant level-14 kernel, and sends the
//! resulting vectors back through the exact level-18 raising verifier.  The
//! four irreps without level-14 analogues are supplied by direct exact sparse
//! solves with modular rank certificates and full integer residual checks.
//!
//! The same report audits the committed first-momentum gauge artifacts.  That
//! audit proves an obstruction for the strict source-invariance condition
//! `A G_p = 0`.  It is not a target gauge quotient: a quotient allowing
//! `A G_p = K_p T_p` still requires independently specified target maps.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SCHEMA_VERSION: &str = "adynkra-11d-level18-momentum-v1";
const FULL_MASK: u32 = u32::MAX;

type Weight = [i8; 5];

const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];

const FIRST_MOMENTUM_SUMMARY: &str =
    include_str!("../results/adynkra_11d_gauge_first_momentum_summary.json");
const DIRECT_GENERATION_REPORT: &str =
    include_str!("../results/adynkra_11d_level18_direct_kernel_generation.json");
const TARGET_STREAM_VALIDATION: &str =
    include_str!("../results/adynkra_11d_target_stream_validation.json");
const SOURCE_FIXED_CURVATURE_VALIDATION: &str =
    include_str!("../results/eleven_dimensional_source_fixed_curvature_validation.json");
const ZERO_MOMENTUM_REPORTS: [&str; 6] = [
    include_str!("../results/adynkra_11d_gauge_zero_momentum_form_0.json"),
    include_str!("../results/adynkra_11d_gauge_zero_momentum_form_1.json"),
    include_str!("../results/adynkra_11d_gauge_zero_momentum_form_2.json"),
    include_str!("../results/adynkra_11d_gauge_zero_momentum_form_3.json"),
    include_str!("../results/adynkra_11d_gauge_zero_momentum_form_4.json"),
    include_str!("../results/adynkra_11d_gauge_zero_momentum_form_5.json"),
];
const FIRST_MOMENTUM_REPORTS: [(usize, &str, &str); 4] = [
    (
        0,
        include_str!("../results/adynkra_11d_gauge_first_momentum_functional_form_0.json"),
        "4b86b7803fdfe82e930b742ff85f3b6c2c050409e723f820869269e958c21718",
    ),
    (
        1,
        include_str!("../results/adynkra_11d_first_momentum_gauge_functional_p1.json"),
        "f183def003a71cd08b7516ad5a666e589eff20629706bdda64bb5d0eb4e3b62c",
    ),
    (
        2,
        include_str!("../results/adynkra_11d_first_momentum_gauge_functional_p2.json"),
        "9177fe087728bced2df21a984020a1d7d5c485a59e01f9ac1094673ccc32a7cd",
    ),
    (
        5,
        include_str!("../results/adynkra_11d_first_momentum_gauge_functional_p5.json"),
        "281999a56b85ab59b7fa50a40c4b2f6afa645f4c2cb24fc6563d60c621b272c2",
    ),
];

macro_rules! direct_kernel {
    ($name:ident, $path:literal) => {
        const $name: &[u8] =
            include_bytes!(concat!("../data/eleven_dimensional_spinor_bridge/", $path));
    };
}

direct_kernel!(D18_12000, "level18_12000_highest_weight_kernel.i16le");
direct_kernel!(D18_11100_1, "level18_11100_highest_weight_kernel_1.i16le");
direct_kernel!(D18_11100_2, "level18_11100_highest_weight_kernel_2.i16le");
direct_kernel!(D18_11100_3, "level18_11100_highest_weight_kernel_3.i16le");
direct_kernel!(D18_11010_1, "level18_11010_highest_weight_kernel_1.i32le");
direct_kernel!(D18_11010_2, "level18_11010_highest_weight_kernel_2.i32le");
direct_kernel!(D18_11010_3, "level18_11010_highest_weight_kernel_3.i32le");
direct_kernel!(D18_11010_4, "level18_11010_highest_weight_kernel_4.i32le");
direct_kernel!(D18_11010_5, "level18_11010_highest_weight_kernel_5.i32le");
direct_kernel!(D18_11002_1, "level18_11002_highest_weight_kernel_1.i16le");
direct_kernel!(D18_11002_2, "level18_11002_highest_weight_kernel_2.i16le");
direct_kernel!(D18_11002_3, "level18_11002_highest_weight_kernel_3.i16le");
direct_kernel!(D18_11002_4, "level18_11002_highest_weight_kernel_4.i16le");
direct_kernel!(D18_11002_5, "level18_11002_highest_weight_kernel_5.i16le");
direct_kernel!(D18_11002_6, "level18_11002_highest_weight_kernel_6.i16le");

#[derive(Clone, Debug, Serialize)]
pub struct HodgeConventionAudit {
    pub spinor_dimension: usize,
    pub source_exterior_degree: usize,
    pub target_exterior_degree: usize,
    pub opposite_weight_pairing_is_complete: bool,
    pub invariant_pairing_edge_checks: usize,
    pub invariant_pairing_edge_residuals: usize,
    pub hodge_square_sign: i8,
    pub expected_hodge_square_sign: i8,
    pub convention: &'static str,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LiftedKernelAudit {
    pub dynkin_label: String,
    pub copy: usize,
    pub source_artifact: String,
    pub output_artifact: String,
    pub construction: &'static str,
    pub coefficient_width_bytes: usize,
    pub source_coefficients: usize,
    pub target_coefficients: usize,
    pub nonzero_coefficients: usize,
    pub output_sha256: String,
    pub exact_level18_raising_verified: bool,
    pub lowering_strings_verified: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct MissingKernelWorkItem {
    pub dynkin_label: String,
    pub copy_count: usize,
    pub source_weight_space_columns: usize,
    pub raising_block_rows: [usize; 5],
    pub total_raising_rows: usize,
    pub expected_artifacts: Vec<String>,
    pub blocked_by: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Level18KernelReport {
    pub required_distinct_irreps: usize,
    pub required_kernel_copies: usize,
    pub required_source_target_embedded_copies: usize,
    pub hodge_lifted_distinct_irreps: usize,
    pub hodge_lifted_kernel_copies: usize,
    pub exactly_verified_kernel_copies: usize,
    pub direct_solved_distinct_irreps: usize,
    pub direct_solved_kernel_copies: usize,
    pub direct_generation_report_sha256: String,
    pub direct_generation_rank_nullity_verified: bool,
    pub missing_distinct_irreps: usize,
    pub missing_kernel_copies: usize,
    pub source_ready_embedded_copies: usize,
    pub source_ready_embedded_fraction: String,
    pub embedded_couplings_computed: usize,
    pub hodge_convention: HodgeConventionAudit,
    pub kernels: Vec<LiftedKernelAudit>,
    pub missing_work: Vec<MissingKernelWorkItem>,
    pub hodge_subset_complete: bool,
    pub full_level18_kernel_inventory_complete: bool,
    pub all_embedded_compositions_complete: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Deserialize)]
struct ZeroMomentumArtifact {
    schema_version: String,
    passed: bool,
    gauge_form_degree: usize,
    parameter_dynkin_label: String,
    primitive_integer_kernel_basis: Vec<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
struct CompleteFirstMomentumArtifact {
    schema_version: String,
    passed: bool,
    gauge_form_degree: usize,
    parameter_dynkin_label: String,
    parameter_components: usize,
    evaluated_parameter_components: Vec<usize>,
    parameter_projection_is_complete: bool,
    zero_momentum_kernel_dimension: usize,
    parameterized_columns: usize,
    functional_rows: usize,
    exact_functional_rank: usize,
    exact_functional_nullity: usize,
    functional_kernel_leading_projection_rank: usize,
    nonzero_leading_extension_excluded_by_functionals: bool,
    functional_kernel_residuals_exactly_zero: bool,
    source_artifact_sha256: Vec<String>,
    zero_momentum_kernel_report_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct FirstMomentumSummary {
    schema_version: String,
    zero_momentum_kernel_dimensions_by_form_degree: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GaugeChannelMomentumAudit {
    pub form_degree: usize,
    pub parameter_dynkin_label: String,
    pub zero_momentum_kernel_dimension: usize,
    pub zero_momentum_artifact_sha256: String,
    pub first_momentum_screen_present: bool,
    pub evaluated_parameter_components: Vec<usize>,
    pub parameter_projection_complete: bool,
    pub parameterized_columns: usize,
    pub functional_rows: usize,
    pub functional_rank: usize,
    pub functional_nullity: usize,
    pub leading_projection_rank: usize,
    pub partial_parameter_screen_excludes_leading_projection: bool,
    pub nonzero_leading_extension_excluded: bool,
    pub first_momentum_artifact_sha256: Option<String>,
    pub artifact_hash_matches_pinned_complete_projection: bool,
    pub complete_projection_artifact_provenance_verified: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct MomentumGaugeReport {
    pub summary_schema_version: String,
    pub summary_sha256: String,
    pub channels: Vec<GaugeChannelMomentumAudit>,
    pub all_six_zero_momentum_channels_exact: bool,
    pub zero_momentum_kernel_dimensions: Vec<usize>,
    pub form_3_and_4_excluded_at_zero_momentum: bool,
    pub remaining_channels_excluded_at_first_momentum: bool,
    pub every_nonempty_channel_subset_excluded_under_strict_source_invariance: bool,
    pub momentum_corrected_strict_source_quotient_computed: bool,
    pub target_gauge_maps_supplied: bool,
    pub momentum_dependent_target_gauge_quotient_computed: bool,
    pub polynomial_module_cohomology_computed: bool,
    pub generic_momentum_quotient_computed: bool,
    pub target_quotient_api: TargetGaugeQuotientReadiness,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetGaugeQuotientReadiness {
    pub target_stream_schema_version: String,
    pub source_fixed_curvature_schema_version: String,
    pub target_stream_contract_passed: bool,
    pub source_fixed_curvature_scaffold_passed: bool,
    pub typed_target_stream_join_available: bool,
    pub full_curvature_map_available: bool,
    pub exact_sparse_image_containment_api_available: bool,
    pub exact_level18_embedded_maps_available: bool,
    pub exact_level18_embedded_map_count: usize,
    pub actual_target_maps_supplied: bool,
    pub quotient_computed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactSparseMapEntry {
    pub row: usize,
    pub column: usize,
    pub numerator: String,
    pub denominator: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactSparseMapInput {
    pub rows: usize,
    pub columns: usize,
    pub entries: Vec<ExactSparseMapEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetGaugeChannelQuotientInput {
    pub form_degree: usize,
    pub parameter_components: usize,
    pub curvature_variation: ExactSparseMapInput,
    pub target_gauge_map: ExactSparseMapInput,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetGaugeQuotientInput {
    pub target_stream_schema_version: String,
    pub source_fixed_curvature_schema_version: String,
    pub target_stream_content_sha256: String,
    pub source_fixed_curvature_content_sha256: String,
    pub channels: Vec<TargetGaugeChannelQuotientInput>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetGaugeChannelQuotientResult {
    pub form_degree: usize,
    pub target_rows: usize,
    pub curvature_variation_columns: usize,
    pub target_gauge_columns: usize,
    pub target_gauge_rank: usize,
    pub augmented_rank: usize,
    pub curvature_variation_lies_in_target_gauge_image: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetGaugeQuotientResult {
    pub input_valid: bool,
    pub channels: Vec<TargetGaugeChannelQuotientResult>,
    pub every_channel_variation_lies_in_target_gauge_image: bool,
    pub quotient_computed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Level18MomentumReport {
    pub schema_version: &'static str,
    pub published_inventory_source: &'static str,
    pub gauge_ansatz_source: &'static str,
    pub source_scope: &'static str,
    pub level18: Level18KernelReport,
    pub momentum_gauge: MomentumGaugeReport,
    pub bounded_program_passed: bool,
    pub full_requested_step_complete: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Level18MomentumArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub report: Level18MomentumReport,
}

#[derive(Clone, Debug)]
struct LiftedKernel {
    label: String,
    copy: usize,
    source_artifact: String,
    output_artifact: String,
    construction: &'static str,
    width: usize,
    bytes: Vec<u8>,
    source_coefficients: usize,
}

/// One exact highest-weight source embedding in `Lambda^18 S`.
///
/// The owned byte vector is intentional: twenty-seven entries are produced by
/// the verified Hodge lift at runtime, while fifteen are committed direct
/// solves.  Consumers therefore receive one uniform representation.
#[derive(Clone, Debug)]
pub struct Level18SourceFixture {
    pub dynkin_label: String,
    pub copy: usize,
    pub artifact: String,
    pub coefficient_width_bytes: usize,
    pub bytes: Vec<u8>,
}

/// Return all forty-two exact level-18 source embeddings.
pub fn level18_source_fixtures() -> Vec<Level18SourceFixture> {
    static FIXTURES: OnceLock<Vec<Level18SourceFixture>> = OnceLock::new();
    FIXTURES
        .get_or_init(|| {
            let (_, kernels) = verify_level18();
            kernels
                .into_iter()
                .map(|kernel| Level18SourceFixture {
                    dynkin_label: kernel.label,
                    copy: kernel.copy,
                    artifact: kernel.output_artifact,
                    coefficient_width_bytes: kernel.width,
                    bytes: kernel.bytes,
                })
                .collect()
        })
        .clone()
}

#[derive(Clone, Debug, Deserialize)]
struct DirectGenerationOutput {
    copy: usize,
    path: String,
    sha256: String,
    bytes: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct DirectGenerationSystem {
    dynkin_label: String,
    exterior_degree: usize,
    source_columns: usize,
    prime: u64,
    exact_modular_rank: usize,
    exact_nullity: usize,
    coefficient_width_bytes: usize,
    outputs: Vec<DirectGenerationOutput>,
    passed: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct DirectGenerationArtifact {
    schema_version: String,
    method: String,
    completed_systems: usize,
    completed_kernel_copies: usize,
    systems: Vec<DirectGenerationSystem>,
    passed: bool,
}

fn spinor_weights() -> [Weight; 32] {
    std::array::from_fn(|index| {
        std::array::from_fn(|axis| {
            if (index >> (4 - axis)) & 1 == 0 {
                1
            } else {
                -1
            }
        })
    })
}

fn add(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn raised_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = add(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn invariant_pairing_signs(weights: &[Weight; 32]) -> ([i8; 32], usize, usize) {
    let mut signs = [0_i8; 32];
    signs[0] = 1;
    let mut queue = VecDeque::from([0_usize]);
    let mut edge_checks = 0;
    let mut edge_residuals = 0;
    while let Some(index) = queue.pop_front() {
        for root in 0..5 {
            let adjacent = raised_spinor_index(index, root, weights).or_else(|| {
                (0..32)
                    .find(|candidate| raised_spinor_index(*candidate, root, weights) == Some(index))
            });
            let Some(adjacent) = adjacent else { continue };
            let expected = -signs[index];
            if signs[adjacent] == 0 {
                signs[adjacent] = expected;
                queue.push_back(adjacent);
            } else if signs[adjacent] != expected {
                edge_residuals += 1;
            }
        }
    }
    for lower in 0..32 {
        for root in 0..5 {
            if let Some(upper) = raised_spinor_index(lower, root, weights) {
                edge_checks += 1;
                if signs[upper] + signs[lower] != 0 {
                    edge_residuals += 1;
                }
            }
        }
    }
    assert!(signs.iter().all(|sign| *sign != 0));
    (signs, edge_checks, edge_residuals)
}

fn mask_weight(mask: u32, weights: &[Weight; 32]) -> Weight {
    let mut weight = [0_i8; 5];
    for (index, spinor_weight) in weights.iter().enumerate() {
        if mask & (1_u32 << index) != 0 {
            for axis in 0..5 {
                weight[axis] += spinor_weight[axis];
            }
        }
    }
    weight
}

fn half_mask_weight(mask: u16, offset: usize, weights: &[Weight; 32]) -> Weight {
    let mut weight = [0_i8; 5];
    for local in 0..16 {
        if mask & (1_u16 << local) != 0 {
            for axis in 0..5 {
                weight[axis] += weights[offset + local][axis];
            }
        }
    }
    weight
}

fn half_groups(offset: usize, weights: &[Weight; 32]) -> HashMap<(u8, Weight), Vec<u16>> {
    let mut groups = HashMap::<(u8, Weight), Vec<u16>>::new();
    for mask in 0_u32..=u32::from(u16::MAX) {
        let mask = mask as u16;
        groups
            .entry((
                mask.count_ones() as u8,
                half_mask_weight(mask, offset, weights),
            ))
            .or_default()
            .push(mask);
    }
    groups
}

fn dynkin_highest_weight(label: &str) -> Weight {
    let labels = label
        .bytes()
        .map(|byte| i8::try_from(byte - b'0').unwrap())
        .collect::<Vec<_>>();
    std::array::from_fn(|index| 2 * labels[index..4].iter().sum::<i8>() + labels[4])
}

fn masks_of_weight(degree: u8, target: Weight, weights: &[Weight; 32]) -> Vec<u32> {
    let left = half_groups(0, weights);
    let right = half_groups(16, weights);
    let mut masks = Vec::new();
    for left_degree in 0_u8..=degree.min(16) {
        let right_degree = degree - left_degree;
        if right_degree > 16 {
            continue;
        }
        for ((candidate_degree, left_weight), left_masks) in &left {
            if *candidate_degree != left_degree {
                continue;
            }
            let needed = std::array::from_fn(|axis| target[axis] - left_weight[axis]);
            if let Some(right_masks) = right.get(&(right_degree, needed)) {
                for left_mask in left_masks {
                    for right_mask in right_masks {
                        masks.push(u32::from(*left_mask) | (u32::from(*right_mask) << 16));
                    }
                }
            }
        }
    }
    masks.sort_unstable();
    masks
}

fn decode_coefficients(bytes: &[u8], width: usize) -> Vec<i64> {
    match width {
        2 => bytes
            .chunks_exact(2)
            .map(|word| i64::from(i16::from_le_bytes([word[0], word[1]])))
            .collect(),
        4 => bytes
            .chunks_exact(4)
            .map(|word| i64::from(i32::from_le_bytes(word.try_into().unwrap())))
            .collect(),
        _ => panic!("unsupported signed integer width {width}"),
    }
}

fn encode_coefficients(coefficients: &[i64], width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(coefficients.len() * width);
    for coefficient in coefficients {
        match width {
            2 => bytes.extend_from_slice(&i16::try_from(*coefficient).unwrap().to_le_bytes()),
            4 => bytes.extend_from_slice(&i32::try_from(*coefficient).unwrap().to_le_bytes()),
            _ => panic!("unsupported signed integer width {width}"),
        }
    }
    bytes
}

fn inversion_sign_between(mask: u32, complement: u32) -> i8 {
    let mut inversions = 0_u32;
    for index in 0..32 {
        if mask & (1_u32 << index) != 0 {
            inversions += (complement & ((1_u32 << index) - 1)).count_ones();
        }
    }
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn hodge_mask_and_sign(mask: u32, pairing_signs: &[i8; 32]) -> (u32, i8) {
    assert_eq!(mask.count_ones(), 14);
    let mut paired = 0_u32;
    let mut sign = if (14 * 13 / 2) % 2 == 0 { 1 } else { -1 };
    for index in 0..32 {
        if mask & (1_u32 << index) != 0 {
            paired |= 1_u32 << (31 - index);
            sign *= pairing_signs[index];
        }
    }
    let output = FULL_MASK ^ paired;
    sign *= inversion_sign_between(paired, output);
    (output, sign)
}

fn output_name(label: &str, copy: usize, copies: usize, width: usize) -> String {
    let suffix = if copies == 1 {
        String::new()
    } else {
        format!("_{copy}")
    };
    format!(
        "level18_{label}_highest_weight_kernel{suffix}.i{}le",
        width * 8
    )
}

fn construct_hodge_lifts() -> (HodgeConventionAudit, Vec<LiftedKernel>) {
    let weights = spinor_weights();
    let opposite_weight_pairing_is_complete = (0..32)
        .all(|index| weights[31 - index] == std::array::from_fn(|axis| -weights[index][axis]));
    let (pairing_signs, edge_checks, edge_residuals) = invariant_pairing_signs(&weights);
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
        .into_iter()
        .filter(|fixture| fixture.dynkin_label != "00000")
        .collect::<Vec<_>>();
    let counts = fixtures
        .iter()
        .fold(BTreeMap::new(), |mut counts, fixture| {
            *counts.entry(fixture.dynkin_label).or_insert(0_usize) += 1;
            counts
        });
    let mut basis_cache = BTreeMap::<String, (Vec<u32>, Vec<u32>)>::new();
    let mut lifted = Vec::new();
    for fixture in fixtures {
        let (source_basis, target_basis) = basis_cache
            .entry(fixture.dynkin_label.to_string())
            .or_insert_with(|| {
                let highest = dynkin_highest_weight(fixture.dynkin_label);
                (
                    masks_of_weight(14, highest, &weights),
                    masks_of_weight(18, highest, &weights),
                )
            });
        let width = if fixture.artifact.ends_with(".i32le") {
            4
        } else {
            2
        };
        let source = decode_coefficients(fixture.bytes, width);
        assert_eq!(source.len(), source_basis.len());
        assert_eq!(source_basis.len(), target_basis.len());
        let target_index = target_basis
            .iter()
            .copied()
            .enumerate()
            .map(|(index, mask)| (mask, index))
            .collect::<BTreeMap<_, _>>();
        let mut target = vec![0_i64; target_basis.len()];
        for (mask, coefficient) in source_basis.iter().copied().zip(source) {
            let (output_mask, sign) = hodge_mask_and_sign(mask, &pairing_signs);
            assert_eq!(
                mask_weight(output_mask, &weights),
                dynkin_highest_weight(fixture.dynkin_label)
            );
            target[target_index[&output_mask]] = coefficient * i64::from(sign);
        }
        lifted.push(LiftedKernel {
            label: fixture.dynkin_label.to_string(),
            copy: fixture.copy,
            source_artifact: fixture.artifact.to_string(),
            output_artifact: output_name(
                fixture.dynkin_label,
                fixture.copy,
                counts[fixture.dynkin_label],
                width,
            ),
            construction: "exact B5-equivariant Hodge lift from the verified level-14 kernel",
            width,
            bytes: encode_coefficients(&target, width),
            source_coefficients: source_basis.len(),
        });
    }

    // The chosen pairing and orientation have star^2 = (-1)^(14*18) = +1.
    let hodge_square_sign = 1;
    let expected_hodge_square_sign = if (14 * 18) % 2 == 0 { 1 } else { -1 };
    let passed = opposite_weight_pairing_is_complete
        && edge_checks == 48
        && edge_residuals == 0
        && hodge_square_sign == expected_hodge_square_sign;
    (
        HodgeConventionAudit {
            spinor_dimension: 32,
            source_exterior_degree: 14,
            target_exterior_degree: 18,
            opposite_weight_pairing_is_complete,
            invariant_pairing_edge_checks: edge_checks,
            invariant_pairing_edge_residuals: edge_residuals,
            hodge_square_sign,
            expected_hodge_square_sign,
            convention: "B(e_i,e_(31-i)) has signs propagated by B(E_r u,v)+B(u,E_r v)=0; exterior indices are ascending and alpha wedge star(alpha) uses the ascending 32-spinor volume orientation",
            passed,
        },
        lifted,
    )
}

fn construct_direct_kernels() -> (Vec<LiftedKernel>, bool) {
    let report: DirectGenerationArtifact = serde_json::from_str(DIRECT_GENERATION_REPORT)
        .expect("parse direct level-18 kernel generation report");
    let embedded: [(&str, usize, usize, &[u8]); 15] = [
        ("12000", 1, 2, D18_12000),
        ("11100", 1, 2, D18_11100_1),
        ("11100", 2, 2, D18_11100_2),
        ("11100", 3, 2, D18_11100_3),
        ("11010", 1, 4, D18_11010_1),
        ("11010", 2, 4, D18_11010_2),
        ("11010", 3, 4, D18_11010_3),
        ("11010", 4, 4, D18_11010_4),
        ("11010", 5, 4, D18_11010_5),
        ("11002", 1, 2, D18_11002_1),
        ("11002", 2, 2, D18_11002_2),
        ("11002", 3, 2, D18_11002_3),
        ("11002", 4, 2, D18_11002_4),
        ("11002", 5, 2, D18_11002_5),
        ("11002", 6, 2, D18_11002_6),
    ];
    let expected_counts = BTreeMap::from([
        ("12000", 1_usize),
        ("11100", 3_usize),
        ("11010", 5_usize),
        ("11002", 6_usize),
    ]);
    let mut rank_nullity_verified = report.schema_version
        == "adynkra-11d-level18-direct-kernel-generation-v1"
        && report.method
            == "deterministic exact sparse echelon over 2^31-1, rational reconstruction, and full integer residual verification"
        && report.passed
        && report.completed_systems == expected_counts.len()
        && report.completed_kernel_copies == embedded.len()
        && report.systems.len() == expected_counts.len();
    let mut kernels = Vec::new();
    for (label, copy, width, bytes) in embedded {
        let copies = expected_counts[label];
        let output_artifact = output_name(label, copy, copies, width);
        let system = report
            .systems
            .iter()
            .find(|system| system.dynkin_label == label);
        let output = system.and_then(|system| {
            system.outputs.iter().find(|output| {
                output.copy == copy
                    && Path::new(&output.path)
                        .file_name()
                        .is_some_and(|name| name == output_artifact.as_str())
            })
        });
        rank_nullity_verified &= system.is_some_and(|system| {
            system.passed
                && system.exterior_degree == 18
                && system.prime == 2_147_483_647
                && system.exact_modular_rank + system.exact_nullity == system.source_columns
                && system.exact_nullity == expected_counts[label]
                && system.outputs.len() == expected_counts[label]
                && system.coefficient_width_bytes == width
                && bytes.len() == system.source_columns * width
        }) && output.is_some_and(|output| {
            output.bytes == bytes.len() && output.sha256 == format!("{:x}", Sha256::digest(bytes))
        });
        kernels.push(LiftedKernel {
            label: label.to_string(),
            copy,
            source_artifact:
                "results/adynkra_11d_level18_direct_kernel_generation.json".to_string(),
            output_artifact,
            construction: "direct exact sparse degree-18 raising-kernel solve with modular rank certificate and full integer residual verification",
            width,
            bytes: bytes.to_vec(),
            source_coefficients: system.map_or(0, |system| system.source_columns),
        });
    }
    (kernels, rank_nullity_verified)
}

fn verify_level18() -> (Level18KernelReport, Vec<LiftedKernel>) {
    let (hodge_convention, mut lifted) = construct_hodge_lifts();
    let hodge_lifted_kernel_copies = lifted.len();
    let hodge_lifted_distinct_irreps = lifted
        .iter()
        .map(|kernel| kernel.label.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let (direct, direct_generation_rank_nullity_verified) = construct_direct_kernels();
    let direct_solved_kernel_copies = direct.len();
    let direct_solved_distinct_irreps = direct
        .iter()
        .map(|kernel| kernel.label.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    lifted.extend(direct);
    let mut by_label = BTreeMap::<String, Vec<&LiftedKernel>>::new();
    for kernel in &lifted {
        by_label
            .entry(kernel.label.clone())
            .or_default()
            .push(kernel);
    }
    let mut fixtures = Vec::new();
    for (label, kernels) in &by_label {
        let label: &'static str = Box::leak(label.clone().into_boxed_str());
        let artifacts = kernels
            .iter()
            .map(|kernel| {
                let name: &'static str = Box::leak(kernel.output_artifact.clone().into_boxed_str());
                let bytes: &'static [u8] = Box::leak(kernel.bytes.clone().into_boxed_slice());
                (name, bytes)
            })
            .collect::<Vec<_>>();
        let artifacts: &'static [(&'static str, &'static [u8])] =
            Box::leak(artifacts.into_boxed_slice());
        fixtures.push(
            crate::eleven_dimensional_bridge::ExteriorHighestWeightKernelFixture {
                exterior_degree: 18,
                dynkin_label: label,
                coefficient_width_bytes: kernels[0].width,
                kernel_artifacts: artifacts,
            },
        );
    }
    let systems =
        crate::eleven_dimensional_bridge::verify_exterior_highest_weight_kernel_fixtures(&fixtures);
    let verification = systems
        .iter()
        .flat_map(|system| {
            system.exact_kernel_vectors.iter().map(move |kernel| {
                (
                    (system.dynkin_label.to_string(), kernel.artifact.to_string()),
                    (
                        kernel.exact_kernel_verified,
                        kernel
                            .first_lowering_descendants
                            .iter()
                            .all(|audit| audit.matches_highest_weight_string),
                    ),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let kernels = lifted
        .iter()
        .map(|kernel| {
            let coefficients = decode_coefficients(&kernel.bytes, kernel.width);
            let key = (kernel.label.clone(), kernel.output_artifact.clone());
            let (exact_level18_raising_verified, lowering_strings_verified) = verification[&key];
            LiftedKernelAudit {
                dynkin_label: kernel.label.clone(),
                copy: kernel.copy,
                source_artifact: kernel.source_artifact.clone(),
                output_artifact: kernel.output_artifact.clone(),
                construction: kernel.construction,
                coefficient_width_bytes: kernel.width,
                source_coefficients: kernel.source_coefficients,
                target_coefficients: coefficients.len(),
                nonzero_coefficients: coefficients.iter().filter(|value| **value != 0).count(),
                output_sha256: format!("{:x}", Sha256::digest(&kernel.bytes)),
                exact_level18_raising_verified,
                lowering_strings_verified,
                passed: exact_level18_raising_verified && lowering_strings_verified,
            }
        })
        .collect::<Vec<_>>();

    let missing_work = Vec::<MissingKernelWorkItem>::new();
    let exactly_verified_kernel_copies = kernels.iter().filter(|kernel| kernel.passed).count();
    let hodge_verified_kernel_copies = kernels
        .iter()
        .filter(|kernel| {
            kernel.construction
                == "exact B5-equivariant Hodge lift from the verified level-14 kernel"
                && kernel.passed
        })
        .count();
    let missing_kernel_copies = missing_work.iter().map(|work| work.copy_count).sum();
    let ready_labels = by_label.keys().cloned().collect::<BTreeSet<_>>();
    let target_labels = crate::eleven_dimensional_prepotential::spinor_tensor_channels("11000")
        .into_iter()
        .map(|(label, _)| label)
        .collect::<Vec<_>>();
    let source_ready_embedded_copies = target_labels
        .iter()
        .flat_map(|target| {
            crate::eleven_dimensional_prepotential::spinor_level_channel_sources(18, target)
        })
        .filter(|(source, _, _)| ready_labels.contains(source))
        .map(|(_, _, multiplicity)| multiplicity)
        .sum::<usize>();
    let hodge_subset_complete = hodge_convention.passed
        && hodge_lifted_distinct_irreps == 12
        && hodge_lifted_kernel_copies == 27
        && hodge_verified_kernel_copies == 27;
    let full_level18_kernel_inventory_complete = hodge_subset_complete
        && direct_generation_rank_nullity_verified
        && direct_solved_distinct_irreps == 4
        && direct_solved_kernel_copies == 15
        && exactly_verified_kernel_copies == 42
        && by_label.len() == 16
        && source_ready_embedded_copies == 77;
    let report = Level18KernelReport {
        required_distinct_irreps: 16,
        required_kernel_copies: 42,
        required_source_target_embedded_copies: 77,
        hodge_lifted_distinct_irreps,
        hodge_lifted_kernel_copies,
        exactly_verified_kernel_copies,
        direct_solved_distinct_irreps,
        direct_solved_kernel_copies,
        direct_generation_report_sha256: format!("{:x}", Sha256::digest(DIRECT_GENERATION_REPORT)),
        direct_generation_rank_nullity_verified,
        missing_distinct_irreps: missing_work.len(),
        missing_kernel_copies,
        source_ready_embedded_copies,
        source_ready_embedded_fraction: format!("{source_ready_embedded_copies}/77"),
        embedded_couplings_computed: 0,
        hodge_convention,
        kernels,
        missing_work,
        hodge_subset_complete,
        full_level18_kernel_inventory_complete,
        all_embedded_compositions_complete: false,
        passed: full_level18_kernel_inventory_complete,
        boundary: "All forty-two required level-18 source kernels are exact across sixteen irreps. The source inventory covers all seventy-seven source incidences in the target-resolved census, but it does not construct the seventy-seven embedded source-target Clebsch-Gordan maps.",
    };
    (report, lifted)
}

fn exact_dense_map(input: &ExactSparseMapInput) -> Result<Vec<Vec<Ratio<BigInt>>>, String> {
    if input.rows == 0 || input.columns == 0 {
        return Err("exact sparse maps must have nonzero dimensions".to_string());
    }
    if input.rows.checked_mul(input.columns).is_none() || input.rows * input.columns > 16_777_216 {
        return Err("exact sparse map exceeds the bounded dense rank gate".to_string());
    }
    let mut matrix = vec![vec![Ratio::from_integer(BigInt::from(0)); input.columns]; input.rows];
    for entry in &input.entries {
        if entry.row >= input.rows || entry.column >= input.columns {
            return Err("exact sparse map entry is out of bounds".to_string());
        }
        let numerator = entry
            .numerator
            .parse::<BigInt>()
            .map_err(|error| format!("invalid exact numerator: {error}"))?;
        let denominator = entry
            .denominator
            .parse::<BigInt>()
            .map_err(|error| format!("invalid exact denominator: {error}"))?;
        if denominator.is_zero() {
            return Err("exact sparse map denominator is zero".to_string());
        }
        matrix[entry.row][entry.column] += Ratio::new(numerator, denominator);
    }
    Ok(matrix)
}

pub fn evaluate_target_gauge_quotient(
    input: &TargetGaugeQuotientInput,
) -> Result<TargetGaugeQuotientResult, String> {
    let (target_stream_sha256, source_fixed_curvature_sha256) =
        target_gauge_quotient_provenance_hashes();
    if input.target_stream_schema_version != "adynkra-11d-target-resolved-composition-stream-v2"
        || input.source_fixed_curvature_schema_version
            != "adynkra-11d-source-fixed-curvature-scaffold-v1"
        || input.target_stream_content_sha256 != target_stream_sha256
        || input.source_fixed_curvature_content_sha256 != source_fixed_curvature_sha256
        || input.channels.len() != 6
    {
        return Err("target quotient input provenance or channel census is incomplete".to_string());
    }
    let mut seen = BTreeSet::new();
    let mut channels = Vec::new();
    for channel in &input.channels {
        if channel.form_degree > 5
            || !seen.insert(channel.form_degree)
            || channel.parameter_components != [1, 11, 55, 165, 330, 462][channel.form_degree]
            || channel.curvature_variation.columns != channel.parameter_components
            || channel.curvature_variation.rows != channel.target_gauge_map.rows
        {
            return Err("target quotient channel dimensions are inconsistent".to_string());
        }
        let target = exact_dense_map(&channel.target_gauge_map)?;
        let variation = exact_dense_map(&channel.curvature_variation)?;
        let target_gauge_rank =
            crate::eleven_dimensional_level16_couplings::rational_matrix_rank(&target);
        let augmented = target
            .iter()
            .zip(&variation)
            .map(|(target_row, variation_row)| {
                target_row
                    .iter()
                    .chain(variation_row)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let augmented_rank =
            crate::eleven_dimensional_level16_couplings::rational_matrix_rank(&augmented);
        channels.push(TargetGaugeChannelQuotientResult {
            form_degree: channel.form_degree,
            target_rows: channel.curvature_variation.rows,
            curvature_variation_columns: channel.curvature_variation.columns,
            target_gauge_columns: channel.target_gauge_map.columns,
            target_gauge_rank,
            augmented_rank,
            curvature_variation_lies_in_target_gauge_image: augmented_rank == target_gauge_rank,
        });
    }
    channels.sort_by_key(|channel| channel.form_degree);
    let every_channel_variation_lies_in_target_gauge_image = channels
        .iter()
        .all(|channel| channel.curvature_variation_lies_in_target_gauge_image);
    Ok(TargetGaugeQuotientResult {
        input_valid: true,
        quotient_computed: true,
        passed: every_channel_variation_lies_in_target_gauge_image,
        every_channel_variation_lies_in_target_gauge_image,
        channels,
        boundary: "This exact rank gate tests whether each supplied F A G_p image lies inside the supplied target gauge image. It does not establish that the supplied K, F, or target gauge maps are the physical eleven-dimensional operators; provenance and convention fixing remain caller obligations.",
    })
}

pub fn target_gauge_quotient_provenance_hashes() -> (String, String) {
    (
        format!("{:x}", Sha256::digest(TARGET_STREAM_VALIDATION)),
        format!("{:x}", Sha256::digest(SOURCE_FIXED_CURVATURE_VALIDATION)),
    )
}

fn verify_momentum_gauge() -> MomentumGaugeReport {
    let summary: FirstMomentumSummary =
        serde_json::from_str(FIRST_MOMENTUM_SUMMARY).expect("parse first-momentum summary");
    let report_by_degree = FIRST_MOMENTUM_REPORTS
        .iter()
        .map(|(degree, bytes, pinned_sha256)| (*degree, (*bytes, *pinned_sha256)))
        .collect::<BTreeMap<_, _>>();
    let is_sha256 = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    let mut channels = Vec::new();
    for (degree, bytes) in ZERO_MOMENTUM_REPORTS.iter().enumerate() {
        let zero: ZeroMomentumArtifact =
            serde_json::from_str(bytes).expect("parse zero-momentum gauge artifact");
        assert_eq!(degree, zero.gauge_form_degree);
        let zero_momentum_kernel_dimension = zero.primitive_integer_kernel_basis.len();
        let zero_hash = format!("{:x}", Sha256::digest(bytes));
        let complete = report_by_degree
            .get(&degree)
            .map(|(artifact, pinned_sha256)| {
                let parsed: CompleteFirstMomentumArtifact = serde_json::from_str(artifact)
                    .expect("parse complete first-momentum projection artifact");
                let actual_sha256 = format!("{:x}", Sha256::digest(artifact));
                let expected_components = [1, 11, 55, 165, 330, 462][degree];
                let provenance_verified = parsed.schema_version
                    == "adynkra-11d-first-momentum-gauge-functional-merge-v1"
                    && parsed.passed
                    && actual_sha256 == *pinned_sha256
                    && parsed.gauge_form_degree == degree
                    && parsed.parameter_dynkin_label == zero.parameter_dynkin_label
                    && parsed.parameter_components == expected_components
                    && parsed.evaluated_parameter_components
                        == (0..expected_components).collect::<Vec<_>>()
                    && parsed.parameter_projection_is_complete
                    && parsed.zero_momentum_kernel_dimension == zero_momentum_kernel_dimension
                    && parsed.zero_momentum_kernel_report_sha256 == zero_hash
                    && parsed.exact_functional_rank + parsed.exact_functional_nullity
                        == parsed.parameterized_columns
                    && parsed.functional_kernel_leading_projection_rank == 0
                    && parsed.nonzero_leading_extension_excluded_by_functionals
                    && parsed.functional_kernel_residuals_exactly_zero
                    && parsed.source_artifact_sha256.len() == 56
                    && parsed
                        .source_artifact_sha256
                        .iter()
                        .all(|hash| is_sha256(hash));
                (parsed, actual_sha256, provenance_verified)
            });
        let parameter_projection_complete =
            complete.as_ref().is_some_and(|(artifact, _, valid)| {
                *valid && artifact.parameter_projection_is_complete
            });
        let partial_parameter_screen_excludes_leading_projection =
            complete.as_ref().is_some_and(|(artifact, _, valid)| {
                *valid && artifact.nonzero_leading_extension_excluded_by_functionals
            });
        let excluded = zero_momentum_kernel_dimension == 0
            || (parameter_projection_complete
                && partial_parameter_screen_excludes_leading_projection);
        let complete_projection_artifact_provenance_verified = match complete.as_ref() {
            Some((_, _, provenance_verified)) => *provenance_verified,
            None => true,
        };
        channels.push(GaugeChannelMomentumAudit {
            form_degree: degree,
            parameter_dynkin_label: zero.parameter_dynkin_label,
            zero_momentum_kernel_dimension,
            zero_momentum_artifact_sha256: zero_hash,
            first_momentum_screen_present: complete.is_some(),
            evaluated_parameter_components: complete
                .as_ref()
                .map(|(artifact, _, _)| artifact.evaluated_parameter_components.clone())
                .unwrap_or_default(),
            parameter_projection_complete,
            parameterized_columns: complete
                .as_ref()
                .map_or(0, |(artifact, _, _)| artifact.parameterized_columns),
            functional_rows: complete
                .as_ref()
                .map_or(0, |(artifact, _, _)| artifact.functional_rows),
            functional_rank: complete
                .as_ref()
                .map_or(0, |(artifact, _, _)| artifact.exact_functional_rank),
            functional_nullity: complete
                .as_ref()
                .map_or(0, |(artifact, _, _)| artifact.exact_functional_nullity),
            leading_projection_rank: complete.as_ref().map_or(0, |(artifact, _, _)| {
                artifact.functional_kernel_leading_projection_rank
            }),
            partial_parameter_screen_excludes_leading_projection,
            nonzero_leading_extension_excluded: excluded,
            first_momentum_artifact_sha256: complete.as_ref().map(|(_, hash, _)| hash.clone()),
            artifact_hash_matches_pinned_complete_projection:
                complete_projection_artifact_provenance_verified,
            complete_projection_artifact_provenance_verified,
            passed: zero.schema_version == "adynkra-11d-zero-momentum-gauge-kernel-v1"
                && zero.passed
                && complete_projection_artifact_provenance_verified
                && excluded,
        });
    }
    let dimensions = channels
        .iter()
        .map(|channel| channel.zero_momentum_kernel_dimension)
        .collect::<Vec<_>>();
    let all_six_zero_momentum_channels_exact =
        channels.len() == 6 && dimensions == summary.zero_momentum_kernel_dimensions_by_form_degree;
    let form_3_and_4_excluded_at_zero_momentum = dimensions[3] == 0 && dimensions[4] == 0;
    let remaining_channels_excluded_at_first_momentum = [0, 1, 2, 5].iter().all(|degree| {
        channels[*degree].first_momentum_screen_present
            && channels[*degree].parameter_projection_complete
            && channels[*degree].leading_projection_rank == 0
            && channels[*degree].nonzero_leading_extension_excluded
            && channels[*degree].complete_projection_artifact_provenance_verified
    });
    let every_nonempty_channel_subset_excluded_under_strict_source_invariance =
        form_3_and_4_excluded_at_zero_momentum && remaining_channels_excluded_at_first_momentum;
    let target_stream = crate::eleven_dimensional_target_stream::verify();
    let source_fixed_curvature = crate::eleven_dimensional_source_fixed_curvature::verify();
    let embedded = crate::eleven_dimensional_level18_embedded::verify();
    let target_quotient_api = TargetGaugeQuotientReadiness {
        target_stream_schema_version: target_stream.schema_version.to_string(),
        source_fixed_curvature_schema_version: source_fixed_curvature.schema_version.to_string(),
        target_stream_contract_passed: target_stream.passed,
        source_fixed_curvature_scaffold_passed: source_fixed_curvature.passed,
        typed_target_stream_join_available: source_fixed_curvature
            .typed_target_stream_join_available,
        full_curvature_map_available: source_fixed_curvature.full_f_a_g_p_test_ready,
        exact_sparse_image_containment_api_available: true,
        exact_level18_embedded_maps_available: embedded.all_77_embedded_maps_complete,
        exact_level18_embedded_map_count: embedded.exact_embedded_maps,
        actual_target_maps_supplied: false,
        quotient_computed: false,
        boundary: "All seventy-seven exact level-18 source-target representation maps and the exact image-containment API are available. The physical gate remains false because convention-fixed F A G_p variations and physical target gauge-image maps have not both been supplied.",
    };
    MomentumGaugeReport {
        summary_schema_version: summary.schema_version,
        summary_sha256: format!("{:x}", Sha256::digest(FIRST_MOMENTUM_SUMMARY)),
        channels,
        all_six_zero_momentum_channels_exact,
        zero_momentum_kernel_dimensions: dimensions,
        form_3_and_4_excluded_at_zero_momentum,
        remaining_channels_excluded_at_first_momentum,
        every_nonempty_channel_subset_excluded_under_strict_source_invariance,
        momentum_corrected_strict_source_quotient_computed:
            every_nonempty_channel_subset_excluded_under_strict_source_invariance,
        target_gauge_maps_supplied: false,
        momentum_dependent_target_gauge_quotient_computed: false,
        polynomial_module_cohomology_computed: false,
        generic_momentum_quotient_computed: false,
        target_quotient_api,
        passed: all_six_zero_momentum_channels_exact
            && every_nonempty_channel_subset_excluded_under_strict_source_invariance,
        result: "The exact bounded first-momentum screen excludes every nonempty subset of the six candidate source-gauge channels under strict source invariance A G_p = 0.",
        boundary: "The complete coordinate projections prove a bounded first-momentum negative result for strict source invariance. They do not compute a generic polynomial-momentum quotient, do not test F A G_p modulo physical target gauge images, and do not establish superspace cohomology.",
    }
}

fn compute_report() -> Level18MomentumReport {
    let (mut level18, _) = verify_level18();
    let embedded = crate::eleven_dimensional_level18_embedded::verify();
    level18.embedded_couplings_computed = embedded.exact_embedded_maps;
    level18.all_embedded_compositions_complete = embedded.all_77_embedded_maps_complete;
    level18.boundary = "All forty-two level-18 source kernels and all seventy-seven exact source-target representation maps are complete. The physical target gauge quotient remains separate and is not claimed here.";
    let momentum_gauge = verify_momentum_gauge();
    Level18MomentumReport {
        schema_version: SCHEMA_VERSION,
        published_inventory_source: "S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, arXiv:2002.08502, Appendix F",
        gauge_ansatz_source: "S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, arXiv:2007.05097, a 10D Weyl/prepotential paper whose introduction states a conjectural high-dimensional one-spinor-derivative rule; not an 11D cohomology source",
        source_scope: "The cited papers supply the representation inventory and motivate the Lorentz-compatible gauge ansatz. They do not provide the level-18 embedded Clebsch-Gordan maps or a target gauge quotient.",
        bounded_program_passed: level18.passed && momentum_gauge.passed,
        full_requested_step_complete: level18.full_level18_kernel_inventory_complete
            && level18.all_embedded_compositions_complete
            && momentum_gauge.momentum_dependent_target_gauge_quotient_computed
            && momentum_gauge.polynomial_module_cohomology_computed,
        level18,
        momentum_gauge,
    }
}

pub fn verify() -> Level18MomentumReport {
    static REPORT: OnceLock<Level18MomentumReport> = OnceLock::new();
    REPORT.get_or_init(compute_report).clone()
}

pub fn build() -> Level18MomentumArtifact {
    Level18MomentumArtifact {
        schema_version: "adynkra-11d-level18-momentum-artifact-v1",
        title: "Exact level-18 source kernels and first-momentum source-gauge screen",
        report: verify(),
    }
}

pub fn write_artifacts(
    kernel_directory: &Path,
    data_path: &Path,
    validation_path: &Path,
) -> Level18MomentumReport {
    let (mut level18, lifted) = verify_level18();
    let embedded = crate::eleven_dimensional_level18_embedded::verify();
    level18.embedded_couplings_computed = embedded.exact_embedded_maps;
    level18.all_embedded_compositions_complete = embedded.all_77_embedded_maps_complete;
    level18.boundary = "All forty-two level-18 source kernels and all seventy-seven exact source-target representation maps are complete. The physical target gauge quotient remains separate and is not claimed here.";
    fs::create_dir_all(kernel_directory).expect("create level-18 kernel directory");
    for kernel in lifted {
        let path = kernel_directory.join(&kernel.output_artifact);
        let temporary = temporary_path(&path);
        File::create(&temporary)
            .and_then(|mut file| file.write_all(&kernel.bytes))
            .expect("write level-18 kernel");
        fs::rename(temporary, path).expect("publish level-18 Hodge kernel");
    }
    let report = Level18MomentumReport {
        schema_version: SCHEMA_VERSION,
        published_inventory_source: "S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, arXiv:2002.08502, Appendix F",
        gauge_ansatz_source: "S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, arXiv:2007.05097, a 10D Weyl/prepotential paper whose introduction states a conjectural high-dimensional one-spinor-derivative rule; not an 11D cohomology source",
        source_scope: "The cited papers supply the representation inventory and motivate the Lorentz-compatible gauge ansatz. They do not provide the level-18 embedded Clebsch-Gordan maps or a target gauge quotient.",
        bounded_program_passed: level18.passed && verify_momentum_gauge().passed,
        full_requested_step_complete: false,
        level18,
        momentum_gauge: verify_momentum_gauge(),
    };
    let artifact = Level18MomentumArtifact {
        schema_version: "adynkra-11d-level18-momentum-artifact-v1",
        title: "Exact level-18 source kernels and first-momentum source-gauge screen",
        report: report.clone(),
    };
    for path in [data_path, validation_path] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create level-18 report directory");
        }
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create level-18 data report")),
        &artifact,
    )
    .expect("write level-18 data report");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create level-18 validation report")),
        &report,
    )
    .expect("write level-18 validation report");
    report
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}.tmp", std::process::id()));
    PathBuf::from(temporary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hodge_lifts_twenty_seven_level18_kernels_exactly() {
        let report = verify();
        assert!(report.level18.hodge_convention.passed);
        assert_eq!(report.level18.hodge_lifted_distinct_irreps, 12);
        assert_eq!(report.level18.hodge_lifted_kernel_copies, 27);
        assert_eq!(
            report
                .level18
                .kernels
                .iter()
                .filter(|kernel| kernel.construction.contains("Hodge lift") && kernel.passed)
                .count(),
            27
        );
        assert!(report.level18.kernels.iter().all(|kernel| kernel.passed));
    }

    #[test]
    fn direct_solves_complete_the_level18_kernel_inventory() {
        let report = verify();
        assert_eq!(report.level18.direct_solved_distinct_irreps, 4);
        assert_eq!(report.level18.direct_solved_kernel_copies, 15);
        assert!(report.level18.direct_generation_rank_nullity_verified);
        assert_eq!(report.level18.exactly_verified_kernel_copies, 42);
        assert!(report.level18.missing_work.is_empty());
        assert_eq!(report.level18.missing_kernel_copies, 0);
        assert_eq!(report.level18.source_ready_embedded_copies, 77);
        assert_eq!(report.level18.source_ready_embedded_fraction, "77/77");
        assert!(report.level18.full_level18_kernel_inventory_complete);
        assert_eq!(report.level18.embedded_couplings_computed, 77);
        assert!(report.level18.all_embedded_compositions_complete);
        assert!(report.level18.passed);
    }

    #[test]
    fn complete_first_momentum_screens_close_the_bounded_strict_source_gate() {
        let report = verify();
        assert!(report.momentum_gauge.passed);
        assert_eq!(
            report.momentum_gauge.zero_momentum_kernel_dimensions,
            vec![11, 1, 1, 0, 0, 1]
        );
        for degree in [0, 1, 2, 5] {
            assert!(report.momentum_gauge.channels[degree].parameter_projection_complete);
            assert!(
                report.momentum_gauge.channels[degree]
                    .complete_projection_artifact_provenance_verified
            );
        }
        assert!(
            report
                .momentum_gauge
                .every_nonempty_channel_subset_excluded_under_strict_source_invariance
        );
        assert!(
            !report
                .momentum_gauge
                .momentum_dependent_target_gauge_quotient_computed
        );
        assert!(
            report
                .momentum_gauge
                .target_quotient_api
                .exact_sparse_image_containment_api_available
        );
        assert!(
            !report
                .momentum_gauge
                .target_quotient_api
                .actual_target_maps_supplied
        );
        assert!(
            report
                .momentum_gauge
                .target_quotient_api
                .exact_level18_embedded_maps_available
        );
        assert_eq!(
            report
                .momentum_gauge
                .target_quotient_api
                .exact_level18_embedded_map_count,
            77
        );
        assert!(!report.momentum_gauge.generic_momentum_quotient_computed);
        assert!(report.bounded_program_passed);
        assert!(!report.full_requested_step_complete);
    }

    fn map(rows: usize, columns: usize, entries: &[(usize, usize, i64)]) -> ExactSparseMapInput {
        ExactSparseMapInput {
            rows,
            columns,
            entries: entries
                .iter()
                .map(|(row, column, numerator)| ExactSparseMapEntry {
                    row: *row,
                    column: *column,
                    numerator: numerator.to_string(),
                    denominator: "1".to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn target_gauge_quotient_api_checks_exact_image_containment() {
        let channels = [1, 11, 55, 165, 330, 462]
            .into_iter()
            .enumerate()
            .map(
                |(form_degree, parameter_components)| TargetGaugeChannelQuotientInput {
                    form_degree,
                    parameter_components,
                    curvature_variation: map(2, parameter_components, &[(0, 0, 2)]),
                    target_gauge_map: map(2, 1, &[(0, 0, 1)]),
                },
            )
            .collect::<Vec<_>>();
        let (target_stream_content_sha256, source_fixed_curvature_content_sha256) =
            target_gauge_quotient_provenance_hashes();
        let mut input = TargetGaugeQuotientInput {
            target_stream_schema_version: "adynkra-11d-target-resolved-composition-stream-v2"
                .to_string(),
            source_fixed_curvature_schema_version: "adynkra-11d-source-fixed-curvature-scaffold-v1"
                .to_string(),
            target_stream_content_sha256,
            source_fixed_curvature_content_sha256,
            channels,
        };
        let contained = evaluate_target_gauge_quotient(&input).unwrap();
        assert!(contained.quotient_computed);
        assert!(contained.passed);
        input.channels[5].curvature_variation = map(2, 462, &[(1, 0, 1)]);
        let escaped = evaluate_target_gauge_quotient(&input).unwrap();
        assert!(!escaped.passed);
        assert!(!escaped.channels[5].curvature_variation_lies_in_target_gauge_image);
        input.target_stream_content_sha256.replace_range(0..1, "f");
        assert!(evaluate_target_gauge_quotient(&input).is_err());
    }

    #[test]
    #[ignore = "writes the reproducible level-18 kernel and JSON artifacts"]
    fn write_committed_artifacts() {
        let report = write_artifacts(
            Path::new("data/eleven_dimensional_spinor_bridge"),
            Path::new("data/eleven_dimensional_level18_momentum.json"),
            Path::new("results/adynkra_11d_level18_momentum_validation.json"),
        );
        assert!(report.bounded_program_passed);
        assert!(!report.full_requested_step_complete);
    }
}
