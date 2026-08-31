//! Fail-closed manifest for the direct-spinor common-parent program.
//!
//! The scalar-factorizing V = D.Psi route and the already exhausted Hhat-only
//! coefficient spaces are negative controls. The viable Phase 1 object is a
//! common unconstrained spinor parent with simultaneous leading maps into
//! Hhat, h, A3, and the component gravitino. Only the Hhat leading block has a
//! complete explicit map basis in the current durable repository.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const LEVEL16: &[u8] = include_bytes!("../results/adynkra_11d_level16_couplings_all.json");
const LEVEL17: &[u8] = include_bytes!("../results/adynkra_11d_level17_derivative_matrix.json");
const FIRST_MOMENTUM: &[u8] =
    include_bytes!("../results/adynkra_11d_first_momentum_couplings_all.json");
const FIRST_TARGET: &[u8] =
    include_bytes!("../results/adynkra_11d_first_momentum_target_couplings.json");
const LEVEL15_SCALAR: &[u8] =
    include_bytes!("../results/adynkra_11d_level15_bridge_validation.json");
const SECOND_RANK: &[u8] =
    include_bytes!("../results/adynkra_11d_second_momentum_full_77_rank_p0.json");

const LEVEL16_SHA: &str = "bada78574729dec6700dbd27979af87c444d5bdeb5a4ec9cddc9f5c2151a4547";
const LEVEL17_SHA: &str = "630fc8701ef7d93e9ce37cdd50be4863dba8b062b8c0112a5d88277d8ec4f5cd";
const FIRST_MOMENTUM_SHA: &str = "fbd64623af146cf4e23f145efc45349885b17e2399e6cc2ec0e85379e6f3a634";
const FIRST_TARGET_SHA: &str = "2b04a743eaf6d02b7ed05ed554748f1286a5dbd57cb760d1d3d101dcde08a8d5";
const LEVEL15_SCALAR_SHA: &str = "a60f45da974b999beb9f66efadd84ad7e821243f4b7ee8ef0edb461cee128387";
const SECOND_RANK_SHA: &str = "d2d59a078bba548df55b89d66ae500666d07a099e47225d6b8d914a8436c9153";

const LEADING_ORDER: [&str; 12] = [
    "10000#1", "20000#1", "00100#1", "00100#2", "00010#1", "00010#2", "00002#1", "10100#1",
    "10010#1", "10002#1", "10002#2", "10002#3",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ImmutableBinding {
    pub role: String,
    pub actual_sha256: String,
    pub expected_sha256: String,
    pub matches: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct PackedLeadingHhatCandidate {
    pub ordinal: usize,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coupled_nonzero_terms: u64,
    pub descriptor_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct CommonParentLeadingBlock {
    pub target_name: String,
    pub target_dynkin_label: String,
    pub source_level: usize,
    pub exact_multiplicity: usize,
    pub explicit_source_kernels_complete: bool,
    pub explicit_clebsches_complete: bool,
    pub candidate_manifest_sha256: Option<String>,
    pub ready: bool,
    pub blocker: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct NegativeControl {
    pub name: String,
    pub coefficient_dimension: usize,
    pub certified_rank: usize,
    pub certified_nullity: usize,
    pub rejected: bool,
    pub certificate: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CommonParentPhManifest {
    pub schema_version: String,
    pub role: String,
    pub immutable_bindings: Vec<ImmutableBinding>,
    pub hhat_leading_candidates: Vec<PackedLeadingHhatCandidate>,
    pub hhat_leading_candidates_sha256: String,
    pub leading_component_blocks: Vec<CommonParentLeadingBlock>,
    pub level17_exterior_rows: usize,
    pub level17_exterior_columns: usize,
    pub level17_exterior_rank: usize,
    pub level17_exterior_nullity: usize,
    pub level17_kernel_basis_dimension: usize,
    pub level17_kernel_residuals_zero: bool,
    pub scalar_factorizing_coordinates: Vec<String>,
    pub scalar_factorizing_exterior_image_zero: bool,
    pub scalar_completion_correction_rank: usize,
    pub scalar_completion_augmented_rank: usize,
    pub scalar_completion_exists: bool,
    pub negative_controls: Vec<NegativeControl>,
    pub bounded_inventory_passed: bool,
    pub complete_common_parent_leading_family: bool,
    pub publication_ready: bool,
    pub boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn binding(role: &str, bytes: &[u8], expected: &str) -> ImmutableBinding {
    let actual = sha256(bytes);
    ImmutableBinding {
        role: role.to_string(),
        matches: actual == expected,
        actual_sha256: actual,
        expected_sha256: expected.to_string(),
    }
}

fn json(bytes: &[u8]) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| error.to_string())
}

fn integer(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| format!("missing integer field {key}"))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field {key}"))
}

fn flag(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing boolean field {key}"))
}

fn repo_path(relative: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn fixture_hash(name: &str) -> Result<String, String> {
    fs::read(repo_path("data/eleven_dimensional_spinor_bridge").join(name))
        .map(|bytes| sha256(&bytes))
        .map_err(|error| format!("cannot hash fixture {name}: {error}"))
}

fn rational(value: &Value) -> Result<BigRational, String> {
    let numerator = text(value, "numerator")?
        .parse::<BigInt>()
        .map_err(|error| error.to_string())?;
    let denominator = text(value, "denominator")?
        .parse::<BigInt>()
        .map_err(|error| error.to_string())?;
    if denominator.is_zero() {
        return Err("zero denominator".to_string());
    }
    Ok(BigRational::new(numerator, denominator))
}

fn matrix(value: &Value) -> Result<Vec<Vec<BigRational>>, String> {
    value
        .as_array()
        .ok_or_else(|| "matrix is not an array".to_string())?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| "matrix row is not an array".to_string())?
                .iter()
                .map(rational)
                .collect()
        })
        .collect()
}

fn rank(matrix: &[Vec<BigRational>]) -> usize {
    if matrix.is_empty() {
        return 0;
    }
    let mut work = matrix.to_vec();
    let columns = work[0].len();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..work.len()).find(|&row| !work[row][column].is_zero()) else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..work.len() {
            if row == pivot_row || work[row][column].is_zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next in column..columns {
                let product = factor.clone() * work[pivot_row][next].clone();
                work[row][next] -= product;
            }
        }
        pivot_row += 1;
        if pivot_row == work.len() {
            break;
        }
    }
    pivot_row
}

fn annihilates(matrix: &[Vec<BigRational>], vector: &[BigRational]) -> bool {
    matrix.iter().all(|row| {
        row.iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .fold(BigRational::zero(), |sum, value| sum + value)
            .is_zero()
    })
}

fn split_key(key: &str) -> Result<(&str, usize), String> {
    let (label, copy) = key
        .split_once('#')
        .ok_or_else(|| format!("invalid source key {key}"))?;
    Ok((
        label,
        copy.parse::<usize>().map_err(|error| error.to_string())?,
    ))
}

fn leading_candidates(level16: &Value) -> Result<Vec<PackedLeadingHhatCandidate>, String> {
    if !flag(level16, "passed")?
        || integer(level16, "embedded_source_copies_certified")? != 12
        || !flag(level16, "every_residual_is_exactly_zero")?
    {
        return Err("level-16 source embedding certificate is incomplete".to_string());
    }
    let entries = level16["embedded_copies"]
        .as_array()
        .ok_or_else(|| "missing embedded_copies".to_string())?;
    let by_key = entries
        .iter()
        .map(|entry| {
            Ok((
                format!(
                    "{}#{}",
                    text(entry, "source_dynkin_label")?,
                    integer(entry, "source_copy")?
                ),
                entry,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;

    LEADING_ORDER
        .iter()
        .enumerate()
        .map(|(ordinal, key)| {
            let entry = by_key
                .get(*key)
                .ok_or_else(|| format!("missing leading source {key}"))?;
            let residuals_zero = entry["exact_raising_residual_terms_by_simple_root"]
                .as_array()
                .is_some_and(|rows| rows.iter().all(|row| row.as_u64() == Some(0)));
            if !flag(entry, "passed")? || !residuals_zero {
                return Err(format!("leading source {key} failed exact parity"));
            }
            let (label, copy) = split_key(key)?;
            let fixture = text(entry, "source_fixture")?;
            let fixture_sha256 = fixture_hash(fixture)?;
            let coupled_nonzero_terms = integer(entry, "coupled_nonzero_terms")? as u64;
            let payload = serde_json::to_vec(&(
                ordinal,
                label,
                copy,
                fixture,
                &fixture_sha256,
                coupled_nonzero_terms,
                LEVEL16_SHA,
            ))
            .map_err(|error| error.to_string())?;
            Ok(PackedLeadingHhatCandidate {
                ordinal,
                source_dynkin_label: label.to_string(),
                source_copy: copy,
                source_fixture: fixture.to_string(),
                source_fixture_sha256: fixture_sha256,
                coupled_nonzero_terms,
                descriptor_sha256: sha256(&payload),
            })
        })
        .collect()
}

fn component_blocks(hhat_sha: &str) -> Vec<CommonParentLeadingBlock> {
    vec![
        CommonParentLeadingBlock {
            target_name: "Hhat".to_string(),
            target_dynkin_label: "10001".to_string(),
            source_level: 16,
            exact_multiplicity: 12,
            explicit_source_kernels_complete: true,
            explicit_clebsches_complete: true,
            candidate_manifest_sha256: Some(hhat_sha.to_string()),
            ready: true,
            blocker: String::new(),
        },
        CommonParentLeadingBlock {
            target_name: "h".to_string(),
            target_dynkin_label: "20000".to_string(),
            source_level: 17,
            exact_multiplicity: 2,
            explicit_source_kernels_complete: false,
            explicit_clebsches_complete: false,
            candidate_manifest_sha256: None,
            ready: false,
            blocker: "two explicit level-17 source kernels and source-to-graviton Clebsches are not durably certified".to_string(),
        },
        CommonParentLeadingBlock {
            target_name: "A3".to_string(),
            target_dynkin_label: "00100".to_string(),
            source_level: 17,
            exact_multiplicity: 8,
            explicit_source_kernels_complete: false,
            explicit_clebsches_complete: false,
            candidate_manifest_sha256: None,
            ready: false,
            blocker: "eight explicit level-17 source kernels and source-to-three-form Clebsches are not durably certified".to_string(),
        },
        CommonParentLeadingBlock {
            target_name: "component_gravitino".to_string(),
            target_dynkin_label: "10001".to_string(),
            source_level: 18,
            exact_multiplicity: 8,
            explicit_source_kernels_complete: false,
            explicit_clebsches_complete: false,
            candidate_manifest_sha256: None,
            ready: false,
            blocker: "eight explicit level-18 source kernels and source-to-component-gravitino Clebsches are not durably certified".to_string(),
        },
    ]
}

/// Build the packed manifest and run the CPU exact parity oracle.
///
/// The caller supplies the already-computed direct Hhat joint rank certificate.
/// This avoids silently rerunning the multi-billion-coordinate producer. A
/// report is accepted only if it binds all 56 columns and certifies rank 56,
/// nullity zero by an exact functional minor.
pub(crate) fn build_common_parent_ph_manifest(
    joint: &crate::eleven_dimensional_level16_couplings::JointCompatibilityMatrixReport,
) -> Result<CommonParentPhManifest, String> {
    let level16 = json(LEVEL16)?;
    let level17 = json(LEVEL17)?;
    let first = json(FIRST_MOMENTUM)?;
    let first_target = json(FIRST_TARGET)?;
    let scalar = json(LEVEL15_SCALAR)?;
    let second = json(SECOND_RANK)?;

    let immutable_bindings = vec![
        binding("level16_Hhat_embeddings", LEVEL16, LEVEL16_SHA),
        binding("level17_exterior_matrix", LEVEL17, LEVEL17_SHA),
        binding(
            "first_momentum_source_maps",
            FIRST_MOMENTUM,
            FIRST_MOMENTUM_SHA,
        ),
        binding("first_momentum_target_maps", FIRST_TARGET, FIRST_TARGET_SHA),
        binding(
            "scalar_negative_control",
            LEVEL15_SCALAR,
            LEVEL15_SCALAR_SHA,
        ),
        binding("second_momentum_rank", SECOND_RANK, SECOND_RANK_SHA),
    ];

    let candidates = leading_candidates(&level16)?;
    let candidate_bytes = serde_json::to_vec(&candidates).map_err(|error| error.to_string())?;
    let hhat_sha = sha256(&candidate_bytes);
    let leading_component_blocks = component_blocks(&hhat_sha);

    let exterior = matrix(&level17["matrix_rows_by_hook_columns_by_source"])?;
    if exterior.len() != 7 || exterior.iter().any(|row| row.len() != 12) {
        return Err("level-17 exterior matrix is not 7 by 12".to_string());
    }
    let exterior_rank = rank(&exterior);
    let kernel = level17["primitive_integer_kernel_basis"]
        .as_array()
        .ok_or_else(|| "missing level-17 kernel basis".to_string())?
        .iter()
        .map(|vector| {
            vector
                .as_array()
                .ok_or_else(|| "kernel vector is not an array".to_string())?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .ok_or_else(|| "kernel coefficient is not a string".to_string())?
                        .parse::<BigInt>()
                        .map(BigRational::from_integer)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    let kernel_residuals_zero =
        kernel.len() == 5 && kernel.iter().all(|vector| annihilates(&exterior, vector));
    let scalar_coordinates = level17["scalar_factorizing_coordinates"]
        .as_array()
        .ok_or_else(|| "missing scalar coordinates".to_string())?
        .iter()
        .map(rational)
        .collect::<Result<Vec<_>, String>>()?;
    let scalar_image_zero =
        scalar_coordinates.len() == 12 && annihilates(&exterior, &scalar_coordinates);
    let completion = &scalar["first_momentum_completion_audit"];
    let scalar_cancellation = flag(completion, "cancellation_exists")?;

    let first_inventory = integer(&first, "embedded_maps_certified")? == 44
        && flag(&first, "passed")?
        && flag(&first, "every_residual_is_exactly_zero")?
        && integer(&first_target, "couplings_verified")? == 4
        && flag(&first_target, "passed")?
        && flag(&first_target, "every_residual_is_exactly_zero")?
        && flag(&first_target, "every_mutation_is_detected")?;
    let joint_negative = joint.coefficient_columns == 56
        && joint.leading_columns == 12
        && joint.first_momentum_columns == 44
        && joint.exact_functional_matrix_rank == 56
        && joint.exact_functional_nullity == 0
        && joint.full_rank_certified_by_functional_minor
        && joint.exact_joint_nullity == Some(0)
        && joint.passed;
    let second_negative = integer(&second, "physical_columns")? == 77
        && integer(&second, "rank_over_gaussian_extension")? == 77
        && flag(&second, "full_column_rank")?
        && flag(&second, "passed")?;

    let negative_controls = vec![
        NegativeControl {
            name: "scalar_factorizing_V_route".to_string(),
            coefficient_dimension: 4,
            certified_rank: 3,
            certified_nullity: 1,
            rejected: !scalar_cancellation,
            certificate: format!(
                "complete pD13 correction system has correction rank 2 and augmented rank 3; {} exact functional rows",
                integer(completion, "exact_functional_rows")?
            ),
        },
        NegativeControl {
            name: "Hhat_only_D16_plus_pD14".to_string(),
            coefficient_dimension: 56,
            certified_rank: joint.exact_functional_matrix_rank,
            certified_nullity: joint.exact_functional_nullity,
            rejected: joint_negative,
            certificate: "exact functional minor of the complete 12 plus 44 compatibility matrix"
                .to_string(),
        },
        NegativeControl {
            name: "Hhat_only_p2D12".to_string(),
            coefficient_dimension: 77,
            certified_rank: integer(&second, "rank_over_gaussian_extension")?,
            certified_nullity: integer(&second, "nullity_upper_bound")?,
            rejected: second_negative,
            certificate: format!(
                "pinned F_p2 full-rank certificate, matrix {}",
                text(&second, "matrix_sha256")?
            ),
        },
    ];

    let unique_descriptors = candidates
        .iter()
        .map(|candidate| candidate.descriptor_sha256.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        == 12;
    let bounded_inventory_passed = immutable_bindings.iter().all(|item| item.matches)
        && candidates.len() == 12
        && unique_descriptors
        && exterior_rank == 7
        && kernel_residuals_zero
        && scalar_image_zero
        && !scalar_cancellation
        && first_inventory
        && joint_negative
        && second_negative;
    let complete_common_parent_leading_family =
        leading_component_blocks.iter().all(|block| block.ready);

    Ok(CommonParentPhManifest {
        schema_version: "adynkra.11d.common-parent-ph-manifest.v1".to_string(),
        role: "packed direct-spinor common-parent leading inventory with exhausted Hhat-only routes retained as negative controls".to_string(),
        immutable_bindings,
        hhat_leading_candidates: candidates,
        hhat_leading_candidates_sha256: hhat_sha,
        leading_component_blocks,
        level17_exterior_rows: 7,
        level17_exterior_columns: 12,
        level17_exterior_rank: exterior_rank,
        level17_exterior_nullity: 12 - exterior_rank,
        level17_kernel_basis_dimension: kernel.len(),
        level17_kernel_residuals_zero: kernel_residuals_zero,
        scalar_factorizing_coordinates: scalar_coordinates
            .iter()
            .map(ToString::to_string)
            .collect(),
        scalar_factorizing_exterior_image_zero: scalar_image_zero,
        scalar_completion_correction_rank: 2,
        scalar_completion_augmented_rank: 3,
        scalar_completion_exists: scalar_cancellation,
        negative_controls,
        bounded_inventory_passed,
        complete_common_parent_leading_family,
        publication_ready: bounded_inventory_passed && complete_common_parent_leading_family,
        boundary: "The Hhat leading block has 12 exact maps and its exterior matrix has rank 7 and nullity 5. The scalar line is ruled out. The complete Hhat-only 12 plus 44 system has rank 56 and nullity 0, and the p2D12 system has rank 77 and nullity 0. These are negative controls, not a common-parent no-go. Exact leading multiplicities for h, A3, and component gravitino are 2, 8, and 8, but their explicit source kernels and Clebsches are not durably certified. Publication therefore fails closed. The next computation is the simultaneous direct component family P_h, P_A, and P_psi, not another Hhat-only solve.".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_hhat_leading_inventory_and_level17_rank_are_stable() {
        let level16 = json(LEVEL16).unwrap();
        let level17 = json(LEVEL17).unwrap();
        let candidates = leading_candidates(&level16).unwrap();
        assert_eq!(candidates.len(), 12);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.descriptor_sha256.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            12
        );
        let exterior = matrix(&level17["matrix_rows_by_hook_columns_by_source"]).unwrap();
        assert_eq!(rank(&exterior), 7);
        assert_eq!(12 - rank(&exterior), 5);
    }

    #[test]
    fn incomplete_component_blocks_fail_closed() {
        let blocks = component_blocks("fixture");
        assert_eq!(
            blocks
                .iter()
                .map(|block| (block.target_name.as_str(), block.exact_multiplicity))
                .collect::<Vec<_>>(),
            vec![
                ("Hhat", 12),
                ("h", 2),
                ("A3", 8),
                ("component_gravitino", 8)
            ]
        );
        assert!(blocks[0].ready);
        assert!(blocks[1..].iter().all(|block| !block.ready));
        assert!(!blocks.iter().all(|block| block.ready));
    }
}
