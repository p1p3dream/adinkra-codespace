//! Fixed work list and representation-level gates for the 11D level-16
//! source-to-vector-spinor coupling certificates.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const TARGET_DYNKIN_LABEL: &str = "10001";
const GOLDEN_COMMIT: &str = "89f20fc";
const SCALAR_BRIDGE_VECTOR_SPINOR_FIXTURE: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/10001_highest_weight_kernel.i16le");
type Weight = [i8; 5];

const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];
const TARGET_WEIGHT: Weight = [3, 1, 1, 1, 1];

#[derive(Debug, Clone, Copy)]
struct CouplingProblem {
    exterior_degree: u8,
    target_dynkin_label: &'static str,
    target_weight: Weight,
    schema_prefix: &'static str,
}

const LEVEL16_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 16,
    target_dynkin_label: TARGET_DYNKIN_LABEL,
    target_weight: TARGET_WEIGHT,
    schema_prefix: "adynkra-11d-level16",
};

const LEVEL17_HOOK_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 17,
    target_dynkin_label: "11000",
    target_weight: [4, 2, 0, 0, 0],
    schema_prefix: "adynkra-11d-level17-hook",
};

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
    pub experimentally_validated_checkpoint_present: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Level17HookCouplingPrecheckReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub exterior_degree: usize,
    pub spinor_dimension: usize,
    pub target_dynkin_label: &'static str,
    pub distinct_source_irreps: usize,
    pub embedded_source_copies: usize,
    pub copy_counts_by_irrep: BTreeMap<&'static str, usize>,
    pub tensor_multiplicities: Vec<TensorMultiplicityAudit>,
    pub every_target_multiplicity_is_one: bool,
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
        experimentally_validated_checkpoint_present: true,
        passed,
    }
}

pub fn verify_hook_precheck() -> Level17HookCouplingPrecheckReport {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures();
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
                    .filter(|(target, _)| target == "11000")
                    .count();
            TensorMultiplicityAudit {
                source_dynkin_label,
                target_dynkin_label: "11000",
                target_multiplicity_in_source_tensor_spinor,
                multiplicity_one: target_multiplicity_in_source_tensor_spinor == 1,
            }
        })
        .collect::<Vec<_>>();
    let every_target_multiplicity_is_one = tensor_multiplicities
        .iter()
        .all(|audit| audit.multiplicity_one);
    let expected = BTreeMap::from([("01001", 2), ("10001", 1), ("11001", 3), ("20001", 1)]);
    let passed =
        copy_counts_by_irrep == expected && fixtures.len() == 7 && every_target_multiplicity_is_one;
    Level17HookCouplingPrecheckReport {
        schema_version: "adynkra-11d-level17-hook-coupling-precheck-v1",
        role: "fixed source manifest and multiplicity-one gate for level-17 couplings into (11000)",
        exterior_degree: 17,
        spinor_dimension: 32,
        target_dynkin_label: "11000",
        distinct_source_irreps: copy_counts_by_irrep.len(),
        embedded_source_copies: fixtures.len(),
        copy_counts_by_irrep,
        tensor_multiplicities,
        every_target_multiplicity_is_one,
        passed,
    }
}

#[derive(Debug, Clone)]
struct DenseState {
    weight: Weight,
    pbw_word: Vec<u8>,
    coefficients: Vec<i64>,
}

#[derive(Debug)]
struct WeightSpace {
    masks: Vec<u32>,
    index: HashMap<u32, usize>,
}

#[derive(Debug)]
struct CsrExteriorAction {
    target_weight: Weight,
    target_dimension: usize,
    source_offsets: Vec<usize>,
    destination_indices: Vec<usize>,
    signs: Vec<i8>,
}

#[derive(Debug)]
struct ExteriorModel {
    exterior_degree: u8,
    spinors: [Weight; 32],
    left: HashMap<(u8, Weight), Vec<u16>>,
    right: HashMap<(u8, Weight), Vec<u16>>,
    spaces: BTreeMap<Weight, WeightSpace>,
    actions: BTreeMap<(Weight, usize, bool), Arc<CsrExteriorAction>>,
    maximum_absolute_accumulator: i128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalDomainBasisEntry {
    pub free_spinor_index: usize,
    pub free_spinor_weight: Weight,
    pub source_weight: Weight,
    pub pbw_word_simple_roots: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractCouplingCertificate {
    pub schema_version: String,
    pub role: String,
    pub source_dynkin_label: String,
    pub source_fixture_copy: usize,
    pub target_dynkin_label: String,
    pub basis_method: String,
    pub dependency_test: String,
    pub product_weight_domain_dimension: usize,
    pub source_weight_spaces_used: usize,
    pub source_weight_multiplicities: Vec<(Weight, usize)>,
    pub domain_basis: Vec<CanonicalDomainBasisEntry>,
    pub gram_matrix_rank: usize,
    pub kernel_dimension: usize,
    pub primitive_domain_coefficients: Vec<i64>,
    pub primitive_coefficient_gcd: i64,
    pub maximum_absolute_primitive_coefficient: i64,
    pub exact_raising_residual_terms_by_simple_root: [usize; 5],
    pub maximum_absolute_checked_accumulator: i128,
    pub storage_type: String,
    pub exterior_action_storage: String,
    pub csr_actions_built: usize,
    pub csr_nonzero_entries: usize,
    pub multiplicity_one: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedCouplingCertificate {
    pub schema_version: String,
    pub role: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub target_dynkin_label: String,
    pub abstract_coupling_source_copy: usize,
    pub product_weight_domain_dimension: usize,
    pub primitive_domain_coefficients: Vec<i64>,
    pub coupled_nonzero_terms: usize,
    pub exact_raising_residual_terms_by_simple_root: [usize; 5],
    pub maximum_absolute_checked_accumulator: i128,
    pub shared_abstract_coupling_applied: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllCouplingCertificateReport {
    pub schema_version: String,
    pub role: String,
    pub abstract_couplings: Vec<AbstractCouplingCertificate>,
    pub embedded_copies: Vec<EmbeddedCouplingCertificate>,
    pub distinct_source_irreps_certified: usize,
    pub embedded_source_copies_certified: usize,
    pub expected_distinct_source_irreps: usize,
    pub expected_embedded_source_copies: usize,
    pub every_residual_is_exactly_zero: bool,
    pub execution_workers: usize,
    pub memory_budget_gib: usize,
    pub estimated_memory_gib_per_worker: usize,
    pub resumed_from_atomic_checkpoints: bool,
    pub boundary: String,
    pub passed: bool,
}

#[derive(Debug, Clone)]
struct CoupledDenseState {
    total_weight: Weight,
    components: BTreeMap<usize, DenseState>,
}

#[derive(Debug, Clone)]
struct CoupledSparseState {
    components: BTreeMap<usize, Vec<(u32, i128)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RationalMatrixEntry {
    pub numerator: String,
    pub denominator: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level17DerivativeMatrixReport {
    pub schema_version: String,
    pub role: String,
    pub source_basis: Vec<String>,
    pub hook_basis: Vec<String>,
    pub target_hook_dynkin_label: String,
    pub target_coupling_terms: usize,
    pub target_coupling_primitive_coefficients: Vec<i64>,
    pub hook_gram_rank: usize,
    pub derivative_matrix_rank: usize,
    pub derivative_matrix_nullity: usize,
    pub matrix_rows_by_hook_columns_by_source: Vec<Vec<RationalMatrixEntry>>,
    pub primitive_integer_kernel_basis: Vec<Vec<String>>,
    pub kernel_residuals_exactly_zero: bool,
    pub kernel_coefficient_mutation_detected: bool,
    pub leading_gram_rank: usize,
    pub scalar_factorizing_coordinates: Vec<RationalMatrixEntry>,
    pub scalar_factorizing_reconstruction_residual_norm: RationalMatrixEntry,
    pub scalar_factorizing_direction_is_in_leading_span: bool,
    pub scalar_factorizing_hook_image: Vec<RationalMatrixEntry>,
    pub scalar_factorizing_hook_image_is_zero: bool,
    pub exact_reconstruction_residual_norms: Vec<RationalMatrixEntry>,
    pub every_derivative_column_is_in_hook_span: bool,
    pub maximum_absolute_checked_accumulator: i128,
    pub convention: String,
    pub interpretation: String,
    pub boundary: String,
    pub passed: bool,
}

impl ExteriorModel {
    fn new(exterior_degree: u8) -> Self {
        let spinors = spinor_weights();
        Self {
            exterior_degree,
            spinors,
            left: half_groups(0, &spinors),
            right: half_groups(16, &spinors),
            spaces: BTreeMap::new(),
            actions: BTreeMap::new(),
            maximum_absolute_accumulator: 0,
        }
    }

    fn space(&mut self, weight: Weight) -> &WeightSpace {
        self.spaces.entry(weight).or_insert_with(|| {
            let masks = weight_basis(self.exterior_degree, weight, &self.left, &self.right);
            let index = masks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, mask)| (mask, index))
                .collect();
            WeightSpace { masks, index }
        })
    }

    fn action(
        &mut self,
        source_weight: Weight,
        root: usize,
        raising: bool,
    ) -> Arc<CsrExteriorAction> {
        let key = (source_weight, root, raising);
        if let Some(action) = self.actions.get(&key) {
            return Arc::clone(action);
        }
        let target_weight = if raising {
            add(source_weight, SIMPLE_ROOTS[root])
        } else {
            subtract(source_weight, SIMPLE_ROOTS[root])
        };
        let source_masks = self.space(source_weight).masks.clone();
        let target_index = self.space(target_weight).index.clone();
        let mut source_offsets = Vec::with_capacity(source_masks.len() + 1);
        let mut destination_indices = Vec::new();
        let mut signs = Vec::new();
        source_offsets.push(0);
        for source_mask in source_masks {
            for occupied_index in 0..32 {
                if source_mask & (1_u32 << occupied_index) == 0 {
                    continue;
                }
                let replacement_index = if raising {
                    raised_spinor_index(occupied_index, root, &self.spinors)
                } else {
                    lowered_spinor_index(occupied_index, root, &self.spinors)
                };
                let Some(replacement_index) = replacement_index else {
                    continue;
                };
                if source_mask & (1_u32 << replacement_index) != 0 {
                    continue;
                }
                let output_mask =
                    (source_mask ^ (1_u32 << occupied_index)) | (1_u32 << replacement_index);
                destination_indices.push(
                    *target_index
                        .get(&output_mask)
                        .expect("exterior action left its target weight space"),
                );
                signs.push(
                    i8::try_from(exterior_replacement_sign(
                        source_mask,
                        occupied_index,
                        replacement_index,
                    ))
                    .unwrap(),
                );
            }
            source_offsets.push(destination_indices.len());
        }
        let action = Arc::new(CsrExteriorAction {
            target_weight,
            target_dimension: target_index.len(),
            source_offsets,
            destination_indices,
            signs,
        });
        self.actions.insert(key, Arc::clone(&action));
        action
    }

    fn apply_action(&mut self, source: &DenseState, root: usize, raising: bool) -> DenseState {
        let action = self.action(source.weight, root, raising);
        let mut accumulator = vec![0_i128; action.target_dimension];
        for (source_index, coefficient) in source.coefficients.iter().copied().enumerate() {
            if coefficient == 0 {
                continue;
            }
            for edge in action.source_offsets[source_index]..action.source_offsets[source_index + 1]
            {
                let destination = action.destination_indices[edge];
                accumulator[destination] = accumulator[destination]
                    .checked_add(i128::from(action.signs[edge]) * i128::from(coefficient))
                    .expect("i128 overflow in exact CSR exterior action");
                self.maximum_absolute_accumulator = self
                    .maximum_absolute_accumulator
                    .max(accumulator[destination].abs());
            }
        }
        let coefficients = accumulator
            .into_iter()
            .map(|value| {
                i64::try_from(value).expect("CSR exterior coefficient exceeds i64 storage")
            })
            .collect();
        let mut pbw_word = source.pbw_word.clone();
        if !raising {
            pbw_word.push(u8::try_from(root + 1).unwrap());
        }
        DenseState {
            weight: action.target_weight,
            pbw_word,
            coefficients,
        }
    }

    fn fixture_state(&mut self, dynkin_label: &str, fixture_bytes: &[u8]) -> DenseState {
        let weight = dynkin_highest_weight(dynkin_label);
        let expected = self.space(weight).masks.len();
        let coefficients = decode_kernel(fixture_bytes);
        assert_eq!(
            coefficients.len(),
            expected,
            "fixture length does not match canonical mask basis for {dynkin_label}"
        );
        DenseState {
            weight,
            pbw_word: Vec::new(),
            coefficients,
        }
    }

    fn lower(&mut self, source: &DenseState, root: usize) -> DenseState {
        self.apply_action(source, root, false)
    }

    fn raise_coefficients(&mut self, source: &DenseState, root: usize) -> (Weight, Vec<i64>) {
        let raised = self.apply_action(source, root, true);
        (raised.weight, raised.coefficients)
    }
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

fn subtract(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dynkin_highest_weight(label: &str) -> Weight {
    assert_eq!(label.len(), 5);
    let labels = label
        .bytes()
        .map(|byte| {
            assert!(byte.is_ascii_digit());
            i8::try_from(byte - b'0').unwrap()
        })
        .collect::<Vec<_>>();
    std::array::from_fn(|index| 2 * labels[index..4].iter().sum::<i8>() + labels[4])
}

fn mask_weight(mask: u16, offset: usize, weights: &[Weight; 32]) -> Weight {
    let mut sum = [0_i8; 5];
    for local in 0..16 {
        if mask & (1 << local) != 0 {
            for axis in 0..5 {
                sum[axis] += weights[offset + local][axis];
            }
        }
    }
    sum
}

fn half_groups(offset: usize, weights: &[Weight; 32]) -> HashMap<(u8, Weight), Vec<u16>> {
    let mut groups = HashMap::<(u8, Weight), Vec<u16>>::new();
    for mask in 0_u32..=u32::from(u16::MAX) {
        let mask = mask as u16;
        groups
            .entry((mask.count_ones() as u8, mask_weight(mask, offset, weights)))
            .or_default()
            .push(mask);
    }
    groups
}

fn weight_basis(
    exterior_degree: u8,
    target: Weight,
    left: &HashMap<(u8, Weight), Vec<u16>>,
    right: &HashMap<(u8, Weight), Vec<u16>>,
) -> Vec<u32> {
    let mut basis = Vec::new();
    for left_degree in 0_u8..=exterior_degree.min(16) {
        let right_degree = exterior_degree - left_degree;
        if right_degree > 16 {
            continue;
        }
        for ((degree, left_weight), left_masks) in left {
            if *degree != left_degree {
                continue;
            }
            if let Some(right_masks) = right.get(&(right_degree, subtract(target, *left_weight))) {
                for &left_mask in left_masks {
                    for &right_mask in right_masks {
                        basis.push(u32::from(left_mask) | (u32::from(right_mask) << 16));
                    }
                }
            }
        }
    }
    basis.sort_unstable();
    basis
}

fn decode_kernel(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(2)
        .map(|pair| i64::from(i16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

fn raised_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = add(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn lowered_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = subtract(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn exterior_replacement_sign(mask: u32, first: usize, second: usize) -> i64 {
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    let interval = if high == low + 1 {
        0
    } else {
        ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
    };
    if (mask & interval).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn lowering_coordinates(upper: Weight, lower: Weight) -> Option<[i16; 5]> {
    let mut difference = [0_i16; 5];
    for index in 0..5 {
        let value = i16::from(upper[index]) - i16::from(lower[index]);
        if value % 2 != 0 {
            return None;
        }
        difference[index] = value / 2;
    }
    let coordinates = [
        difference[0],
        difference[0] + difference[1],
        difference[0] + difference[1] + difference[2],
        difference[0] + difference[1] + difference[2] + difference[3],
        difference.iter().sum(),
    ];
    coordinates
        .iter()
        .all(|coordinate| *coordinate >= 0)
        .then_some(coordinates)
}

fn dot_i128(left: &[i64], right: &[i64]) -> i128 {
    assert_eq!(left.len(), right.len());
    left.iter().zip(right).fold(0_i128, |sum, (a, b)| {
        sum.checked_add(i128::from(*a) * i128::from(*b))
            .expect("i128 overflow in exact Gram entry")
    })
}

fn rational_rank_i128(rows: &[Vec<i128>]) -> usize {
    let zero = Ratio::from_integer(BigInt::zero());
    let mut matrix = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| Ratio::from_integer(BigInt::from(*value)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if matrix.is_empty() {
        return 0;
    }
    let columns = matrix[0].len();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..matrix.len()).find(|row| matrix[*row][column] != zero) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let normalization = matrix[rank][column].clone();
        for value in &mut matrix[rank][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = matrix[rank].clone();
        for row in (rank + 1)..matrix.len() {
            let factor = matrix[row][column].clone();
            if factor == zero {
                continue;
            }
            for index in column..columns {
                matrix[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
        rank += 1;
        if rank == matrix.len() {
            break;
        }
    }
    rank
}

fn exact_gram(states: &[DenseState]) -> Vec<Vec<i128>> {
    states
        .iter()
        .map(|left| {
            states
                .iter()
                .map(|right| dot_i128(&left.coefficients, &right.coefficients))
                .collect()
        })
        .collect()
}

fn select_canonical_independent_states(mut candidates: Vec<DenseState>) -> Vec<DenseState> {
    candidates.sort_by(|left, right| left.pbw_word.cmp(&right.pbw_word));
    candidates.dedup_by(|left, right| left.pbw_word == right.pbw_word);
    let mut selected = Vec::new();
    let mut rank = 0;
    for candidate in candidates {
        if candidate.coefficients.iter().all(|value| *value == 0) {
            continue;
        }
        let mut trial = selected.clone();
        trial.push(candidate.clone());
        let next_rank = rational_rank_i128(&exact_gram(&trial));
        if next_rank > rank {
            selected.push(candidate);
            rank = next_rank;
        }
    }
    selected
}

fn relevant_source_bases(
    model: &mut ExteriorModel,
    highest: DenseState,
    target_weight: Weight,
) -> BTreeMap<Weight, Vec<DenseState>> {
    let source_highest_weight = highest.weight;
    let needed_weights = model
        .spinors
        .iter()
        .map(|spinor| subtract(target_weight, *spinor))
        .filter_map(|weight| {
            lowering_coordinates(source_highest_weight, weight).map(|coordinates| {
                let depth = coordinates.iter().map(|value| *value as usize).sum();
                (weight, depth)
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_depth = needed_weights.values().copied().max().unwrap_or(0);
    let mut current = BTreeMap::from([(source_highest_weight, vec![highest])]);
    let mut required = BTreeMap::new();
    for depth in 0..=maximum_depth {
        let mut next_candidates = BTreeMap::<Weight, Vec<DenseState>>::new();
        for (weight, basis) in current {
            if needed_weights.get(&weight) == Some(&depth) {
                required.insert(weight, basis.clone());
            }
            if depth == maximum_depth {
                continue;
            }
            for root in 0..5 {
                let next_weight = subtract(weight, SIMPLE_ROOTS[root]);
                let remains_relevant = needed_weights.keys().any(|needed| {
                    lowering_coordinates(next_weight, *needed).is_some()
                        && lowering_coordinates(source_highest_weight, next_weight).is_some_and(
                            |coordinates| {
                                coordinates
                                    .iter()
                                    .map(|value| *value as usize)
                                    .sum::<usize>()
                                    == depth + 1
                            },
                        )
                });
                if !remains_relevant {
                    continue;
                }
                for vector in &basis {
                    let lowered = model.lower(vector, root);
                    if lowered.coefficients.iter().any(|value| *value != 0) {
                        next_candidates
                            .entry(next_weight)
                            .or_default()
                            .push(lowered);
                    }
                }
            }
        }
        current = next_candidates
            .into_iter()
            .map(|(weight, candidates)| (weight, select_canonical_independent_states(candidates)))
            .filter(|(_, basis)| !basis.is_empty())
            .collect();
    }
    required
}

#[derive(Debug, Clone)]
struct TensorOutput {
    components: BTreeMap<usize, Vec<i64>>,
}

fn add_component(target: &mut Vec<i64>, source: &[i64]) {
    assert_eq!(target.len(), source.len());
    for (target, source) in target.iter_mut().zip(source) {
        *target = i64::try_from(
            i128::from(*target)
                .checked_add(i128::from(*source))
                .expect("i128 overflow while combining tensor output"),
        )
        .expect("tensor output coefficient exceeds i64 storage");
    }
}

fn tensor_output(
    model: &mut ExteriorModel,
    source: &DenseState,
    spinor_index: usize,
    root: usize,
) -> TensorOutput {
    let (_, raised_source) = model.raise_coefficients(source, root);
    let mut components = BTreeMap::new();
    components.insert(spinor_index, raised_source);
    if let Some(next_spinor) = raised_spinor_index(spinor_index, root, &model.spinors) {
        let component = components
            .entry(next_spinor)
            .or_insert_with(|| vec![0; source.coefficients.len()]);
        add_component(component, &source.coefficients);
    }
    TensorOutput { components }
}

fn tensor_output_dot(left: &TensorOutput, right: &TensorOutput) -> i128 {
    left.components
        .iter()
        .filter_map(|(spinor, coefficients)| {
            right
                .components
                .get(spinor)
                .map(|other| dot_i128(coefficients, other))
        })
        .try_fold(0_i128, |sum, value| sum.checked_add(value))
        .expect("i128 overflow in tensor-output Gram entry")
}

fn bigint_nullspace(rows: &[Vec<BigInt>], columns: usize) -> Vec<Vec<Ratio<BigInt>>> {
    let zero = Ratio::from_integer(BigInt::zero());
    let mut reduced = rows
        .iter()
        .filter(|row| row.iter().any(|value| !value.is_zero()))
        .map(|row| {
            row.iter()
                .cloned()
                .map(Ratio::from_integer)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut pivot_columns = Vec::new();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot_row) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero)
        else {
            continue;
        };
        reduced.swap(rank, pivot_row);
        let normalization = reduced[rank][column].clone();
        for value in &mut reduced[rank] {
            *value /= normalization.clone();
        }
        let pivot = reduced[rank].clone();
        for row in 0..reduced.len() {
            if row == rank || reduced[row][column] == zero {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot[index].clone();
            }
        }
        pivot_columns.push(column);
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    (0..columns)
        .filter(|column| !pivot_columns.contains(column))
        .map(|free| {
            let mut vector = vec![zero.clone(); columns];
            vector[free] = Ratio::from_integer(BigInt::one());
            for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free].clone();
            }
            vector
        })
        .collect()
}

fn bigint_gcd(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = left.abs();
    right = right.abs();
    while !right.is_zero() {
        let remainder = left % &right;
        left = right;
        right = remainder;
    }
    left
}

fn bigint_lcm(left: BigInt, right: BigInt) -> BigInt {
    if left.is_zero() || right.is_zero() {
        BigInt::zero()
    } else {
        (&left / bigint_gcd(left.clone(), right.clone())) * right
    }
}

fn primitive_i64(vector: &[Ratio<BigInt>]) -> Vec<i64> {
    let denominator = vector.iter().fold(BigInt::one(), |common, coefficient| {
        bigint_lcm(common, coefficient.denom().clone())
    });
    let mut integers = vector
        .iter()
        .map(|coefficient| coefficient.numer() * (&denominator / coefficient.denom()))
        .collect::<Vec<_>>();
    let gcd = integers
        .iter()
        .fold(BigInt::zero(), |gcd, value| bigint_gcd(gcd, value.clone()));
    assert!(!gcd.is_zero());
    for value in &mut integers {
        *value /= &gcd;
    }
    if integers
        .iter()
        .find(|value| !value.is_zero())
        .unwrap()
        .is_negative()
    {
        for value in &mut integers {
            *value = -value.clone();
        }
    }
    integers
        .into_iter()
        .map(|value| {
            value
                .to_i64()
                .expect("primitive coupling coefficient exceeds i64")
        })
        .collect()
}

fn tensor_gram(outputs_by_root: &[Vec<TensorOutput>]) -> Vec<Vec<BigInt>> {
    let columns = outputs_by_root[0].len();
    (0..columns)
        .map(|left| {
            (0..columns)
                .map(|right| {
                    let value = outputs_by_root.iter().fold(0_i128, |sum, outputs| {
                        sum.checked_add(tensor_output_dot(&outputs[left], &outputs[right]))
                            .expect("i128 overflow in total tensor Gram")
                    });
                    BigInt::from(value)
                })
                .collect()
        })
        .collect()
}

fn exact_residual_counts(
    outputs_by_root: &[Vec<TensorOutput>],
    primitive: &[i64],
) -> ([usize; 5], i128) {
    let mut maximum = 0_i128;
    let counts = std::array::from_fn(|root| {
        let mut combined = BTreeMap::<usize, Vec<i128>>::new();
        for (output, coefficient) in outputs_by_root[root].iter().zip(primitive) {
            for (spinor, values) in &output.components {
                let destination = combined
                    .entry(*spinor)
                    .or_insert_with(|| vec![0; values.len()]);
                for (slot, value) in destination.iter_mut().zip(values) {
                    *slot = slot
                        .checked_add(i128::from(*coefficient) * i128::from(*value))
                        .expect("i128 overflow in exact raising residual");
                    maximum = maximum.max(slot.abs());
                }
            }
        }
        combined
            .values()
            .map(|values| values.iter().filter(|value| **value != 0).count())
            .sum()
    });
    (counts, maximum)
}

fn build_abstract_from_fixture(
    problem: CouplingProblem,
    dynkin_label: &str,
    fixture_copy: usize,
    fixture_bytes: &[u8],
) -> (AbstractCouplingCertificate, Vec<DenseState>) {
    let mut model = ExteriorModel::new(problem.exterior_degree);
    let highest = model.fixture_state(dynkin_label, fixture_bytes);
    let bases = relevant_source_bases(&mut model, highest, problem.target_weight);
    let mut domain = Vec::<(usize, DenseState)>::new();
    for (spinor_index, spinor_weight) in model.spinors.iter().copied().enumerate() {
        let source_weight = subtract(problem.target_weight, spinor_weight);
        if let Some(states) = bases.get(&source_weight) {
            domain.extend(states.iter().cloned().map(|state| (spinor_index, state)));
        }
    }
    let outputs_by_root = (0..5)
        .map(|root| {
            domain
                .iter()
                .map(|(spinor_index, state)| tensor_output(&mut model, state, *spinor_index, root))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let gram = tensor_gram(&outputs_by_root);
    let nullspace = bigint_nullspace(&gram, domain.len());
    let primitive = if nullspace.len() == 1 {
        primitive_i64(&nullspace[0])
    } else {
        Vec::new()
    };
    let (residuals, residual_maximum) = if primitive.len() == domain.len() {
        exact_residual_counts(&outputs_by_root, &primitive)
    } else {
        ([usize::MAX; 5], 0)
    };
    let gram_matrix_rank = domain.len().saturating_sub(nullspace.len());
    let domain_basis = domain
        .iter()
        .map(|(spinor_index, state)| CanonicalDomainBasisEntry {
            free_spinor_index: *spinor_index,
            free_spinor_weight: model.spinors[*spinor_index],
            source_weight: state.weight,
            pbw_word_simple_roots: state.pbw_word.clone(),
        })
        .collect::<Vec<_>>();
    let source_weight_multiplicities = bases
        .iter()
        .map(|(weight, states)| (*weight, states.len()))
        .collect::<Vec<_>>();
    let primitive_coefficient_gcd = primitive
        .iter()
        .fold(0_i64, |gcd, value| gcd_i64(gcd, *value));
    let maximum_absolute_primitive_coefficient =
        primitive.iter().map(|value| value.abs()).max().unwrap_or(0);
    let multiplicity_one = nullspace.len() == 1;
    let passed = multiplicity_one
        && primitive.len() == domain.len()
        && residuals == [0; 5]
        && primitive_coefficient_gcd == 1;
    let csr_actions_built = model.actions.len();
    let csr_nonzero_entries = model
        .actions
        .values()
        .map(|action| action.destination_indices.len())
        .sum();
    let states = domain.into_iter().map(|(_, state)| state).collect();
    (
        AbstractCouplingCertificate {
            schema_version: format!("{}-abstract-coupling-v1", problem.schema_prefix),
            role: format!(
                "exact canonical abstract coupling into ({})",
                problem.target_dynkin_label
            ),
            source_dynkin_label: dynkin_label.to_string(),
            source_fixture_copy: fixture_copy,
            target_dynkin_label: problem.target_dynkin_label.to_string(),
            basis_method:
                "lexicographic PBW lowering words with exact exterior-realization Gram rank"
                    .to_string(),
            dependency_test:
                "exact rational rank of the integer Gram matrix; no modular acceptance".to_string(),
            product_weight_domain_dimension: domain_basis.len(),
            source_weight_spaces_used: bases.len(),
            source_weight_multiplicities,
            domain_basis,
            gram_matrix_rank,
            kernel_dimension: nullspace.len(),
            primitive_domain_coefficients: primitive,
            primitive_coefficient_gcd,
            maximum_absolute_primitive_coefficient,
            exact_raising_residual_terms_by_simple_root: residuals,
            maximum_absolute_checked_accumulator: model
                .maximum_absolute_accumulator
                .max(residual_maximum),
            storage_type: "i64 coefficients with checked i128 accumulation".to_string(),
            exterior_action_storage: "precomputed CSR over sorted exterior-mask weight spaces"
                .to_string(),
            csr_actions_built,
            csr_nonzero_entries,
            multiplicity_one,
            passed,
        },
        states,
    )
}

fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn state_for_word(
    model: &mut ExteriorModel,
    highest: &DenseState,
    word: &[u8],
    cache: &mut BTreeMap<Vec<u8>, DenseState>,
) -> DenseState {
    if let Some(state) = cache.get(word) {
        return state.clone();
    }
    let prefix = &word[..word.len() - 1];
    let parent = state_for_word(model, highest, prefix, cache);
    let root = usize::from(word[word.len() - 1] - 1);
    let state = model.lower(&parent, root);
    cache.insert(word.to_vec(), state.clone());
    state
}

fn add_scaled_dense_component(
    components: &mut BTreeMap<usize, DenseState>,
    spinor_index: usize,
    source: &DenseState,
    scale: i64,
    maximum: &mut i128,
) {
    if scale == 0 {
        return;
    }
    let destination = components
        .entry(spinor_index)
        .or_insert_with(|| DenseState {
            weight: source.weight,
            pbw_word: Vec::new(),
            coefficients: vec![0; source.coefficients.len()],
        });
    assert_eq!(destination.weight, source.weight);
    assert_eq!(destination.coefficients.len(), source.coefficients.len());
    for (output, input) in destination
        .coefficients
        .iter_mut()
        .zip(&source.coefficients)
    {
        let value = i128::from(*output)
            .checked_add(i128::from(scale) * i128::from(*input))
            .expect("i128 overflow while assembling a coupled state");
        *maximum = (*maximum).max(value.abs());
        *output = i64::try_from(value).expect("coupled-state coefficient exceeds i64 storage");
    }
}

fn materialize_coupled_highest(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
) -> (ExteriorModel, CoupledDenseState, i128) {
    assert_eq!(
        abstract_certificate.target_dynkin_label,
        problem.target_dynkin_label
    );
    assert!(abstract_certificate.passed);
    let mut model = ExteriorModel::new(problem.exterior_degree);
    let highest = model.fixture_state(&abstract_certificate.source_dynkin_label, fixture_bytes);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut components = BTreeMap::new();
    let mut maximum = 0_i128;
    for (entry, coefficient) in abstract_certificate
        .domain_basis
        .iter()
        .zip(&abstract_certificate.primitive_domain_coefficients)
    {
        let state = state_for_word(
            &mut model,
            &highest,
            &entry.pbw_word_simple_roots,
            &mut cache,
        );
        assert_eq!(state.weight, entry.source_weight);
        add_scaled_dense_component(
            &mut components,
            entry.free_spinor_index,
            &state,
            *coefficient,
            &mut maximum,
        );
    }
    (
        model,
        CoupledDenseState {
            total_weight: problem.target_weight,
            components,
        },
        maximum,
    )
}

fn lower_coupled_state(
    model: &mut ExteriorModel,
    source: &CoupledDenseState,
    root: usize,
    maximum: &mut i128,
) -> CoupledDenseState {
    let spinors = model.spinors;
    let mut components = BTreeMap::new();
    for (&free_spinor, exterior) in &source.components {
        if let Some(lowered_free_spinor) = lowered_spinor_index(free_spinor, root, &spinors) {
            add_scaled_dense_component(&mut components, lowered_free_spinor, exterior, 1, maximum);
        }
        let lowered_exterior = model.lower(exterior, root);
        if lowered_exterior
            .coefficients
            .iter()
            .any(|coefficient| *coefficient != 0)
        {
            add_scaled_dense_component(&mut components, free_spinor, &lowered_exterior, 1, maximum);
        }
    }
    CoupledDenseState {
        total_weight: subtract(source.total_weight, SIMPLE_ROOTS[root]),
        components,
    }
}

fn coupled_state_for_word(
    model: &mut ExteriorModel,
    highest: &CoupledDenseState,
    word: &[u8],
    cache: &mut BTreeMap<Vec<u8>, CoupledDenseState>,
    maximum: &mut i128,
) -> CoupledDenseState {
    if let Some(state) = cache.get(word) {
        return state.clone();
    }
    let prefix = &word[..word.len() - 1];
    let parent = coupled_state_for_word(model, highest, prefix, cache, maximum);
    let root = usize::from(word[word.len() - 1] - 1);
    let state = lower_coupled_state(model, &parent, root, maximum);
    cache.insert(word.to_vec(), state.clone());
    state
}

fn dense_coupled_to_sparse(
    model: &mut ExteriorModel,
    source: &CoupledDenseState,
) -> CoupledSparseState {
    let components = source
        .components
        .iter()
        .map(|(spinor, state)| {
            let masks = model.space(state.weight).masks.clone();
            let values = masks
                .into_iter()
                .zip(&state.coefficients)
                .filter_map(|(mask, coefficient)| {
                    (*coefficient != 0).then_some((mask, i128::from(*coefficient)))
                })
                .collect();
            (*spinor, values)
        })
        .collect();
    CoupledSparseState { components }
}

fn build_derivative_candidate(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
    target_terms: &[crate::eleven_dimensional_bridge::DirectHookTargetCouplingTerm],
) -> (CoupledSparseState, CoupledSparseState, i128) {
    let (mut level16, highest, mut maximum) =
        materialize_coupled_highest(LEVEL16_PROBLEM, abstract_certificate, fixture_bytes);
    let leading = dense_coupled_to_sparse(&mut level16, &highest);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut level17 = ExteriorModel::new(17);
    let mut accumulated = BTreeMap::<usize, (Weight, Vec<i128>)>::new();
    for term in target_terms {
        let target_state = coupled_state_for_word(
            &mut level16,
            &highest,
            &term.pbw_word_simple_roots,
            &mut cache,
            &mut maximum,
        );
        assert_eq!(target_state.total_weight, term.vector_spinor_weight);
        let outer_bit = 1_u32 << term.outer_spinor_index;
        let lower_bits = outer_bit - 1;
        for (&free_spinor, source) in &target_state.components {
            let destination_weight = add(source.weight, level16.spinors[term.outer_spinor_index]);
            assert_eq!(
                add(destination_weight, level16.spinors[free_spinor]),
                LEVEL17_HOOK_PROBLEM.target_weight
            );
            let source_masks = level16.space(source.weight).masks.clone();
            let destination_space = level17.space(destination_weight);
            let destination = accumulated
                .entry(free_spinor)
                .or_insert_with(|| (destination_weight, vec![0; destination_space.masks.len()]));
            assert_eq!(destination.0, destination_weight);
            for (mask, coefficient) in source_masks.into_iter().zip(&source.coefficients) {
                if *coefficient == 0 || mask & outer_bit != 0 {
                    continue;
                }
                let sign = if (mask & lower_bits).count_ones() % 2 == 0 {
                    1_i128
                } else {
                    -1_i128
                };
                let output_mask = mask | outer_bit;
                let output_index = destination_space.index[&output_mask];
                let value = destination.1[output_index]
                    .checked_add(
                        i128::from(term.primitive_coefficient) * i128::from(*coefficient) * sign,
                    )
                    .expect("i128 overflow in exterior derivative candidate");
                maximum = maximum.max(value.abs());
                destination.1[output_index] = value;
            }
        }
    }
    let components = accumulated
        .into_iter()
        .map(|(spinor, (weight, coefficients))| {
            let masks = level17.space(weight).masks.clone();
            let values = masks
                .into_iter()
                .zip(coefficients)
                .filter(|(_, coefficient)| *coefficient != 0)
                .collect();
            (spinor, values)
        })
        .collect();
    (CoupledSparseState { components }, leading, maximum)
}

fn build_scalar_factorizing_candidate() -> (CoupledSparseState, i128) {
    let mut level15 = ExteriorModel::new(15);
    let highest = level15.fixture_state(TARGET_DYNKIN_LABEL, SCALAR_BRIDGE_VECTOR_SPINOR_FIXTURE);
    assert_eq!(highest.weight, TARGET_WEIGHT);
    let source_masks = level15.space(highest.weight).masks.clone();
    let charge = crate::eleven_dimensional_clifford::spinor_charge_bilinear();
    let zero = Ratio::from_integer(0);
    let phase = charge
        .iter()
        .flat_map(|row| row.iter())
        .find(|value| value.re != zero || value.im != zero)
        .unwrap()
        .clone();
    let spinors = level15.spinors;
    let mut level16 = ExteriorModel::new(16);
    let mut accumulated = BTreeMap::<usize, (Weight, Vec<i128>)>::new();
    let mut maximum = 0_i128;
    for derivative_spinor in 0..32 {
        let derivative_bit = 1_u32 << derivative_spinor;
        let lower_bits = derivative_bit - 1;
        for free_spinor in 0..32 {
            let normalized = charge[derivative_spinor][free_spinor].clone() / phase.clone();
            assert_eq!(normalized.im, zero);
            assert_eq!(*normalized.re.denom(), 1);
            let contraction = *normalized.re.numer();
            if contraction == 0 {
                continue;
            }
            let destination_weight = add(highest.weight, spinors[derivative_spinor]);
            assert_eq!(add(destination_weight, spinors[free_spinor]), TARGET_WEIGHT);
            let destination_space = level16.space(destination_weight);
            let destination = accumulated
                .entry(free_spinor)
                .or_insert_with(|| (destination_weight, vec![0; destination_space.masks.len()]));
            for (mask, coefficient) in source_masks.iter().zip(&highest.coefficients) {
                if *coefficient == 0 || mask & derivative_bit != 0 {
                    continue;
                }
                let sign = if (mask & lower_bits).count_ones() % 2 == 0 {
                    1_i128
                } else {
                    -1_i128
                };
                let output_index = destination_space.index[&(mask | derivative_bit)];
                let value = destination.1[output_index]
                    .checked_add(i128::from(contraction) * i128::from(*coefficient) * sign)
                    .expect("i128 overflow in scalar-factorizing candidate");
                maximum = maximum.max(value.abs());
                destination.1[output_index] = value;
            }
        }
    }
    let components = accumulated
        .into_iter()
        .map(|(spinor, (weight, coefficients))| {
            let masks = level16.space(weight).masks.clone();
            (
                spinor,
                masks
                    .into_iter()
                    .zip(coefficients)
                    .filter(|(_, coefficient)| *coefficient != 0)
                    .collect(),
            )
        })
        .collect();
    (CoupledSparseState { components }, maximum)
}

fn sparse_coupled_dot(left: &CoupledSparseState, right: &CoupledSparseState) -> i128 {
    let mut total = 0_i128;
    for (spinor, left_values) in &left.components {
        let Some(right_values) = right.components.get(spinor) else {
            continue;
        };
        let mut left_index = 0;
        let mut right_index = 0;
        while left_index < left_values.len() && right_index < right_values.len() {
            let (left_mask, left_value) = left_values[left_index];
            let (right_mask, right_value) = right_values[right_index];
            match left_mask.cmp(&right_mask) {
                std::cmp::Ordering::Less => left_index += 1,
                std::cmp::Ordering::Greater => right_index += 1,
                std::cmp::Ordering::Equal => {
                    total = total
                        .checked_add(
                            left_value
                                .checked_mul(right_value)
                                .expect("i128 overflow in sparse coupled product"),
                        )
                        .expect("i128 overflow in sparse coupled dot product");
                    left_index += 1;
                    right_index += 1;
                }
            }
        }
    }
    total
}

fn solve_bigint_system(
    matrix: &[Vec<BigInt>],
    right_hand_side: &[BigInt],
) -> Option<Vec<Ratio<BigInt>>> {
    let dimension = matrix.len();
    assert_eq!(right_hand_side.len(), dimension);
    assert!(matrix.iter().all(|row| row.len() == dimension));
    let zero = Ratio::from_integer(BigInt::zero());
    let mut augmented = matrix
        .iter()
        .zip(right_hand_side)
        .map(|(row, right)| {
            let mut values = row
                .iter()
                .cloned()
                .map(Ratio::from_integer)
                .collect::<Vec<_>>();
            values.push(Ratio::from_integer(right.clone()));
            values
        })
        .collect::<Vec<_>>();
    for column in 0..dimension {
        let pivot = (column..dimension).find(|row| augmented[*row][column] != zero)?;
        augmented.swap(column, pivot);
        let normalization = augmented[column][column].clone();
        for value in &mut augmented[column][column..=dimension] {
            *value /= normalization.clone();
        }
        let pivot_row = augmented[column].clone();
        for row in 0..dimension {
            if row == column || augmented[row][column] == zero {
                continue;
            }
            let factor = augmented[row][column].clone();
            for index in column..=dimension {
                augmented[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
    }
    Some(
        augmented
            .into_iter()
            .map(|row| row[dimension].clone())
            .collect(),
    )
}

fn rational_matrix_rank(matrix: &[Vec<Ratio<BigInt>>]) -> usize {
    if matrix.is_empty() {
        return 0;
    }
    let zero = Ratio::from_integer(BigInt::zero());
    let mut reduced = matrix.to_vec();
    let columns = reduced[0].len();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero) else {
            continue;
        };
        reduced.swap(rank, pivot);
        let normalization = reduced[rank][column].clone();
        for value in &mut reduced[rank][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = reduced[rank].clone();
        for row in (rank + 1)..reduced.len() {
            let factor = reduced[row][column].clone();
            if factor == zero {
                continue;
            }
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    rank
}

fn rational_nullspace(matrix: &[Vec<Ratio<BigInt>>]) -> Vec<Vec<Ratio<BigInt>>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let columns = matrix[0].len();
    let zero = Ratio::from_integer(BigInt::zero());
    let mut reduced = matrix.to_vec();
    let mut pivot_columns = Vec::new();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero) else {
            continue;
        };
        reduced.swap(rank, pivot);
        let normalization = reduced[rank][column].clone();
        for value in &mut reduced[rank][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = reduced[rank].clone();
        for row in 0..reduced.len() {
            if row == rank || reduced[row][column] == zero {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
        pivot_columns.push(column);
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    (0..columns)
        .filter(|column| !pivot_columns.contains(column))
        .map(|free| {
            let mut vector = vec![zero.clone(); columns];
            vector[free] = Ratio::from_integer(BigInt::one());
            for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free].clone();
            }
            vector
        })
        .collect()
}

fn primitive_bigint_vector(vector: &[Ratio<BigInt>]) -> Vec<BigInt> {
    let denominator = vector.iter().fold(BigInt::one(), |common, coefficient| {
        bigint_lcm(common, coefficient.denom().clone())
    });
    let mut integers = vector
        .iter()
        .map(|coefficient| coefficient.numer() * (&denominator / coefficient.denom()))
        .collect::<Vec<_>>();
    let gcd = integers.iter().fold(BigInt::zero(), |common, value| {
        bigint_gcd(common, value.clone())
    });
    assert!(!gcd.is_zero());
    for value in &mut integers {
        *value /= &gcd;
    }
    if integers
        .iter()
        .find(|value| !value.is_zero())
        .is_some_and(BigInt::is_negative)
    {
        for value in &mut integers {
            *value = -value.clone();
        }
    }
    integers
}

fn matrix_times_integer_vector_is_zero(matrix: &[Vec<Ratio<BigInt>>], vector: &[BigInt]) -> bool {
    matrix.iter().all(|row| {
        row.iter()
            .zip(vector)
            .fold(
                Ratio::from_integer(BigInt::zero()),
                |sum, (coefficient, value)| {
                    sum + coefficient.clone() * Ratio::from_integer(value.clone())
                },
            )
            .is_zero()
    })
}

fn rational_entry(value: &Ratio<BigInt>) -> RationalMatrixEntry {
    RationalMatrixEntry {
        numerator: value.numer().to_string(),
        denominator: value.denom().to_string(),
    }
}

pub fn build_level17_derivative_matrix() -> Level17DerivativeMatrixReport {
    let target_terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
    assert_eq!(target_terms.len(), 8);
    eprintln!("constructed the eight-term target coupling into (11000)");

    let hook_fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures();
    let mut hook_abstract = BTreeMap::new();
    for fixture in &hook_fixtures {
        if fixture.copy == 1 {
            hook_abstract.insert(
                fixture.dynkin_label,
                build_abstract_from_fixture(
                    LEVEL17_HOOK_PROBLEM,
                    fixture.dynkin_label,
                    fixture.copy,
                    fixture.bytes,
                )
                .0,
            );
            eprintln!(
                "constructed abstract hook coupling {}",
                fixture.dynkin_label
            );
        }
    }
    let mut maximum = 0_i128;
    let mut hook_basis = Vec::new();
    let mut hook_labels = Vec::new();
    for fixture in &hook_fixtures {
        let certificate = &hook_abstract[fixture.dynkin_label];
        let (mut model, dense, local_maximum) =
            materialize_coupled_highest(LEVEL17_HOOK_PROBLEM, certificate, fixture.bytes);
        maximum = maximum.max(local_maximum);
        hook_basis.push(dense_coupled_to_sparse(&mut model, &dense));
        hook_labels.push(format!("{}#{}", fixture.dynkin_label, fixture.copy));
        eprintln!(
            "materialized hook basis vector {}#{} ({}/7)",
            fixture.dynkin_label,
            fixture.copy,
            hook_basis.len()
        );
    }
    let hook_gram_i128 = hook_basis
        .iter()
        .map(|left| {
            hook_basis
                .iter()
                .map(|right| sparse_coupled_dot(left, right))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let hook_gram_rank = rational_rank_i128(&hook_gram_i128);
    assert_eq!(hook_gram_rank, hook_basis.len());
    eprintln!("hook Gram matrix has exact rank {hook_gram_rank}");
    let hook_gram = hook_gram_i128
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| BigInt::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let level16_fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut level16_abstract = BTreeMap::new();
    for fixture in &level16_fixtures {
        if fixture.copy == 1 {
            level16_abstract.insert(
                fixture.dynkin_label,
                build_abstract_from_fixture(
                    LEVEL16_PROBLEM,
                    fixture.dynkin_label,
                    fixture.copy,
                    fixture.bytes,
                )
                .0,
            );
            eprintln!(
                "constructed abstract level-16 coupling {}",
                fixture.dynkin_label
            );
        }
    }
    let mut source_labels = Vec::new();
    let mut columns = Vec::<Vec<Ratio<BigInt>>>::new();
    let mut residual_norms = Vec::new();
    let mut leading_basis = Vec::new();
    let (scalar_factorizing_candidate, scalar_maximum) = build_scalar_factorizing_candidate();
    maximum = maximum.max(scalar_maximum);
    eprintln!("constructed the scalar-factorizing leading candidate");
    for fixture in &level16_fixtures {
        let (candidate, leading, local_maximum) = build_derivative_candidate(
            &level16_abstract[fixture.dynkin_label],
            fixture.bytes,
            &target_terms,
        );
        maximum = maximum.max(local_maximum);
        leading_basis.push(leading);
        let overlaps = hook_basis
            .iter()
            .map(|hook| BigInt::from(sparse_coupled_dot(hook, &candidate)))
            .collect::<Vec<_>>();
        let coordinates = solve_bigint_system(&hook_gram, &overlaps)
            .expect("hook Gram matrix must be invertible");
        let candidate_norm =
            Ratio::from_integer(BigInt::from(sparse_coupled_dot(&candidate, &candidate)));
        let projected_norm = coordinates.iter().zip(&overlaps).fold(
            Ratio::from_integer(BigInt::zero()),
            |sum, (coordinate, overlap)| {
                sum + coordinate.clone() * Ratio::from_integer(overlap.clone())
            },
        );
        residual_norms.push(candidate_norm - projected_norm);
        columns.push(coordinates);
        source_labels.push(format!("{}#{}", fixture.dynkin_label, fixture.copy));
        eprintln!(
            "projected derivative column {}#{} ({}/12)",
            fixture.dynkin_label,
            fixture.copy,
            columns.len()
        );
    }
    let matrix = (0..hook_basis.len())
        .map(|row| {
            columns
                .iter()
                .map(|column| column[row].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let derivative_matrix_rank = rational_matrix_rank(&matrix);
    let derivative_matrix_nullity = source_labels.len() - derivative_matrix_rank;
    let primitive_integer_kernel_basis = rational_nullspace(&matrix)
        .iter()
        .map(|vector| primitive_bigint_vector(vector))
        .collect::<Vec<_>>();
    let kernel_residuals_exactly_zero = primitive_integer_kernel_basis
        .iter()
        .all(|vector| matrix_times_integer_vector_is_zero(&matrix, vector));
    let mut mutated_kernel_vector = primitive_integer_kernel_basis[0].clone();
    let mutated_index = mutated_kernel_vector
        .iter()
        .position(|value| !value.is_zero())
        .unwrap();
    mutated_kernel_vector[mutated_index] += BigInt::one();
    let kernel_coefficient_mutation_detected =
        !matrix_times_integer_vector_is_zero(&matrix, &mutated_kernel_vector);
    let leading_gram_i128 = leading_basis
        .iter()
        .map(|left| {
            leading_basis
                .iter()
                .map(|right| sparse_coupled_dot(left, right))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let leading_gram_rank = rational_rank_i128(&leading_gram_i128);
    let leading_gram = leading_gram_i128
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| BigInt::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scalar_overlaps = leading_basis
        .iter()
        .map(|leading| BigInt::from(sparse_coupled_dot(leading, &scalar_factorizing_candidate)))
        .collect::<Vec<_>>();
    let scalar_factorizing_coordinates = solve_bigint_system(&leading_gram, &scalar_overlaps)
        .expect("the twelve leading vectors must be independent");
    let scalar_candidate_norm = Ratio::from_integer(BigInt::from(sparse_coupled_dot(
        &scalar_factorizing_candidate,
        &scalar_factorizing_candidate,
    )));
    let scalar_projected_norm = scalar_factorizing_coordinates
        .iter()
        .zip(&scalar_overlaps)
        .fold(
            Ratio::from_integer(BigInt::zero()),
            |sum, (coordinate, overlap)| {
                sum + coordinate.clone() * Ratio::from_integer(overlap.clone())
            },
        );
    let scalar_factorizing_reconstruction_residual_norm =
        scalar_candidate_norm - scalar_projected_norm;
    let scalar_factorizing_direction_is_in_leading_span =
        scalar_factorizing_reconstruction_residual_norm.is_zero();
    let scalar_factorizing_hook_image = matrix
        .iter()
        .map(|row| {
            row.iter().zip(&scalar_factorizing_coordinates).fold(
                Ratio::from_integer(BigInt::zero()),
                |sum, (entry, coordinate)| sum + entry.clone() * coordinate.clone(),
            )
        })
        .collect::<Vec<_>>();
    let scalar_factorizing_hook_image_is_zero =
        scalar_factorizing_hook_image.iter().all(Ratio::is_zero);
    let every_derivative_column_is_in_hook_span = residual_norms.iter().all(Ratio::is_zero);
    let passed = hook_gram_rank == 7
        && source_labels.len() == 12
        && every_derivative_column_is_in_hook_span
        && primitive_integer_kernel_basis.len() == derivative_matrix_nullity
        && kernel_residuals_exactly_zero
        && kernel_coefficient_mutation_detected
        && leading_gram_rank == 12
        && scalar_factorizing_direction_is_in_leading_span
        && scalar_factorizing_hook_image_is_zero;
    Level17DerivativeMatrixReport {
        schema_version: "adynkra-11d-level17-derivative-matrix-v1".to_string(),
        role: "exact exterior-derivative map from twelve level-16 vector-spinor couplings to seven level-17 hook couplings".to_string(),
        source_basis: source_labels,
        hook_basis: hook_labels,
        target_hook_dynkin_label: "11000".to_string(),
        target_coupling_terms: target_terms.len(),
        target_coupling_primitive_coefficients: target_terms
            .iter()
            .map(|term| term.primitive_coefficient)
            .collect(),
        hook_gram_rank,
        derivative_matrix_rank,
        derivative_matrix_nullity,
        matrix_rows_by_hook_columns_by_source: matrix
            .iter()
            .map(|row| row.iter().map(rational_entry).collect())
            .collect(),
        primitive_integer_kernel_basis: primitive_integer_kernel_basis
            .iter()
            .map(|vector| vector.iter().map(ToString::to_string).collect())
            .collect(),
        kernel_residuals_exactly_zero,
        kernel_coefficient_mutation_detected,
        leading_gram_rank,
        scalar_factorizing_coordinates: scalar_factorizing_coordinates
            .iter()
            .map(rational_entry)
            .collect(),
        scalar_factorizing_reconstruction_residual_norm: rational_entry(
            &scalar_factorizing_reconstruction_residual_norm,
        ),
        scalar_factorizing_direction_is_in_leading_span,
        scalar_factorizing_hook_image: scalar_factorizing_hook_image
            .iter()
            .map(rational_entry)
            .collect(),
        scalar_factorizing_hook_image_is_zero,
        exact_reconstruction_residual_norms: residual_norms.iter().map(rational_entry).collect(),
        every_derivative_column_is_in_hook_span,
        maximum_absolute_checked_accumulator: maximum,
        convention: "canonical sorted spinor-mask exterior basis; left exterior multiplication by the seventeenth derivative; primitive integer source and target couplings".to_string(),
        interpretation: format!(
            "the exact map has rank {derivative_matrix_rank} and a {derivative_matrix_nullity}-dimensional kernel in the twelve-dimensional leading-map coefficient space"
        ),
        boundary: "this is the zero-spacetime-momentum exterior symbol in the direct spinor-prepotential representation complex; it does not select physical coefficients, define the gauge quotient, include momentum corrections, or derive a field equation".to_string(),
        passed,
    }
}

fn verify_embedded_with_abstract(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    fixture_bytes: &[u8],
) -> EmbeddedCouplingCertificate {
    let mut model = ExteriorModel::new(problem.exterior_degree);
    let highest = model.fixture_state(&abstract_certificate.source_dynkin_label, fixture_bytes);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut domain = Vec::new();
    for entry in &abstract_certificate.domain_basis {
        let state = state_for_word(
            &mut model,
            &highest,
            &entry.pbw_word_simple_roots,
            &mut cache,
        );
        assert_eq!(state.weight, entry.source_weight);
        domain.push((entry.free_spinor_index, state));
    }
    let outputs_by_root = (0..5)
        .map(|root| {
            domain
                .iter()
                .map(|(spinor_index, state)| tensor_output(&mut model, state, *spinor_index, root))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (residuals, residual_maximum) = exact_residual_counts(
        &outputs_by_root,
        &abstract_certificate.primitive_domain_coefficients,
    );
    let mut coupled = BTreeMap::<usize, Vec<i128>>::new();
    for ((spinor_index, state), coefficient) in domain
        .iter()
        .zip(&abstract_certificate.primitive_domain_coefficients)
    {
        let destination = coupled
            .entry(*spinor_index)
            .or_insert_with(|| vec![0; state.coefficients.len()]);
        for (slot, value) in destination.iter_mut().zip(&state.coefficients) {
            *slot = slot
                .checked_add(i128::from(*coefficient) * i128::from(*value))
                .expect("i128 overflow while applying abstract coupling");
        }
    }
    let coupled_nonzero_terms = coupled
        .values()
        .map(|values| values.iter().filter(|value| **value != 0).count())
        .sum();
    let passed = abstract_certificate.passed
        && domain.len() == abstract_certificate.product_weight_domain_dimension
        && residuals == [0; 5];
    EmbeddedCouplingCertificate {
        schema_version: format!("{}-embedded-coupling-v1", problem.schema_prefix),
        role: "exact application of the shared abstract coupling to one exterior embedding"
            .to_string(),
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        target_dynkin_label: problem.target_dynkin_label.to_string(),
        abstract_coupling_source_copy: abstract_certificate.source_fixture_copy,
        product_weight_domain_dimension: domain.len(),
        primitive_domain_coefficients: abstract_certificate.primitive_domain_coefficients.clone(),
        coupled_nonzero_terms,
        exact_raising_residual_terms_by_simple_root: residuals,
        maximum_absolute_checked_accumulator: model
            .maximum_absolute_accumulator
            .max(residual_maximum),
        shared_abstract_coupling_applied: true,
        passed,
    }
}

pub fn build_abstract(dynkin_label: &str) -> AbstractCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-16 source irrep {dynkin_label}"));
    build_abstract_from_fixture(LEVEL16_PROBLEM, dynkin_label, fixture.copy, fixture.bytes).0
}

pub fn verify_copy(dynkin_label: &str, copy: usize) -> EmbeddedCouplingCertificate {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let abstract_fixture = fixtures
        .iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-16 source irrep {dynkin_label}"));
    let fixture = fixtures
        .iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == copy)
        .unwrap_or_else(|| panic!("unknown copy {copy} for level-16 source irrep {dynkin_label}"));
    let (abstract_certificate, _) = build_abstract_from_fixture(
        LEVEL16_PROBLEM,
        dynkin_label,
        abstract_fixture.copy,
        abstract_fixture.bytes,
    );
    verify_embedded_with_abstract(
        LEVEL16_PROBLEM,
        &abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn verify_copy_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    copy: usize,
) -> EmbeddedCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
        .into_iter()
        .find(|fixture| {
            fixture.dynkin_label == abstract_certificate.source_dynkin_label && fixture.copy == copy
        })
        .unwrap_or_else(|| {
            panic!(
                "unknown copy {copy} for level-16 source irrep {}",
                abstract_certificate.source_dynkin_label
            )
        });
    verify_embedded_with_abstract(
        LEVEL16_PROBLEM,
        abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn build_hook_abstract(dynkin_label: &str) -> AbstractCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-17 hook source irrep {dynkin_label}"));
    build_abstract_from_fixture(
        LEVEL17_HOOK_PROBLEM,
        dynkin_label,
        fixture.copy,
        fixture.bytes,
    )
    .0
}

pub fn verify_hook_copy_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    copy: usize,
) -> EmbeddedCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures()
        .into_iter()
        .find(|fixture| {
            fixture.dynkin_label == abstract_certificate.source_dynkin_label && fixture.copy == copy
        })
        .unwrap_or_else(|| {
            panic!(
                "unknown copy {copy} for level-17 hook source irrep {}",
                abstract_certificate.source_dynkin_label
            )
        });
    verify_embedded_with_abstract(
        LEVEL17_HOOK_PROBLEM,
        abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn hook_copy_manifest() -> BTreeMap<&'static str, Vec<usize>> {
    let mut copies = BTreeMap::<&str, Vec<usize>>::new();
    for fixture in crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures() {
        copies
            .entry(fixture.dynkin_label)
            .or_default()
            .push(fixture.copy);
    }
    copies
}

fn summarize_problem(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
    expected_distinct_source_irreps: usize,
    expected_embedded_source_copies: usize,
    schema_version: &str,
    role: &str,
    boundary: &str,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> AllCouplingCertificateReport {
    let distinct_source_irreps_certified = abstract_couplings
        .iter()
        .filter(|report| report.passed)
        .count();
    let embedded_source_copies_certified = embedded_copies
        .iter()
        .filter(|report| report.passed)
        .count();
    let every_residual_is_exactly_zero = embedded_copies
        .iter()
        .all(|report| report.exact_raising_residual_terms_by_simple_root == [0; 5]);
    let passed = distinct_source_irreps_certified == expected_distinct_source_irreps
        && embedded_source_copies_certified == expected_embedded_source_copies
        && every_residual_is_exactly_zero;
    AllCouplingCertificateReport {
        schema_version: schema_version.to_string(),
        role: role.to_string(),
        abstract_couplings,
        embedded_copies,
        distinct_source_irreps_certified,
        embedded_source_copies_certified,
        expected_distinct_source_irreps,
        expected_embedded_source_copies,
        every_residual_is_exactly_zero,
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
        boundary: boundary.to_string(),
        passed,
    }
}

pub fn summarize_all(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> AllCouplingCertificateReport {
    summarize_problem(
        abstract_couplings,
        embedded_copies,
        8,
        12,
        "adynkra-11d-level16-all-couplings-v1",
        "exact dense certification of all level-16 source couplings into (10001)",
        "this certifies the twelve source embeddings and their couplings into the (10001) channel under the stated exterior-algebra conventions; it does not solve the full Gates-Hu prepotential problem",
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
    )
}

pub fn summarize_hooks(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> AllCouplingCertificateReport {
    summarize_problem(
        abstract_couplings,
        embedded_copies,
        4,
        7,
        "adynkra-11d-level17-hook-all-couplings-v1",
        "exact dense certification of all level-17 source couplings into (11000)",
        "this certifies the seven source embeddings and their couplings into the (11000) hook channel; the derivative matrix is a separate calculation",
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
    )
}

#[allow(dead_code)]
pub fn verify_all() -> AllCouplingCertificateReport {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut grouped = BTreeMap::<&str, Vec<_>>::new();
    for fixture in &fixtures {
        grouped
            .entry(fixture.dynkin_label)
            .or_default()
            .push(*fixture);
    }
    let mut abstract_couplings = Vec::new();
    let mut embedded_copies = Vec::new();
    for (label, copies) in grouped {
        let first = copies
            .iter()
            .find(|fixture| fixture.copy == 1)
            .expect("each irrep must have copy 1");
        let (abstract_certificate, _) =
            build_abstract_from_fixture(LEVEL16_PROBLEM, label, first.copy, first.bytes);
        for fixture in copies {
            embedded_copies.push(verify_embedded_with_abstract(
                LEVEL16_PROBLEM,
                &abstract_certificate,
                fixture.copy,
                fixture.artifact,
                fixture.bytes,
            ));
        }
        abstract_couplings.push(abstract_certificate);
    }
    summarize_all(abstract_couplings, embedded_copies, 1, 0, 0, false)
}

pub fn write_atomic_json<T: Serialize>(output: &Path, report: &T, passed: bool) -> io::Result<()> {
    if !passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "refusing to checkpoint a failed certificate",
        ));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = temporary_path(output);
    let payload = serde_json::to_vec_pretty(report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let parsed: serde_json::Value = serde_json::from_slice(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if parsed.get("passed").and_then(|value| value.as_bool()) != Some(true) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "serialized certificate does not contain passed=true",
        ));
    }
    {
        let mut file = File::create(&temporary)?;
        file.write_all(&payload)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temporary, output)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn temporary_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("certificate.json");
    output.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
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

    #[test]
    fn dense_engine_reproduces_the_committed_20000_golden_coupling() {
        let report = build_abstract("20000");
        assert!(report.passed);
        assert_eq!(report.product_weight_domain_dimension, 6);
        assert_eq!(report.kernel_dimension, 1);
        assert_eq!(report.primitive_domain_coefficients, [1, -2, 2, -2, 2, -4]);
        assert_eq!(report.exact_raising_residual_terms_by_simple_root, [0; 5]);
        assert_eq!(
            report
                .domain_basis
                .iter()
                .map(|entry| entry.pbw_word_simple_roots.clone())
                .collect::<Vec<_>>(),
            vec![
                vec![1, 2, 3, 4, 5],
                vec![1, 2, 3, 4],
                vec![1, 2, 3],
                vec![1, 2],
                vec![1],
                vec![],
            ]
        );
    }

    #[test]
    fn golden_gate_detects_a_primitive_coefficient_mutation() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "20000")
            .unwrap();
        let mut abstract_certificate = build_abstract("20000");
        abstract_certificate.primitive_domain_coefficients[0] += 1;
        let report = verify_embedded_with_abstract(
            LEVEL16_PROBLEM,
            &abstract_certificate,
            fixture.copy,
            fixture.artifact,
            fixture.bytes,
        );
        assert!(!report.passed);
        assert!(report
            .exact_raising_residual_terms_by_simple_root
            .iter()
            .any(|terms| *terms != 0));
    }

    #[test]
    fn atomic_checkpoint_requires_and_preserves_a_passing_report() {
        let path = std::env::temp_dir().join(format!(
            "adinkra-level16-atomic-checkpoint-{}.json",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let passing = serde_json::json!({"schema_version": "test", "passed": true});
        write_atomic_json(&path, &passing, true).unwrap();
        let restored: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(restored["passed"], true);
        fs::remove_file(&path).unwrap();

        let failing = serde_json::json!({"schema_version": "test", "passed": false});
        assert!(write_atomic_json(&path, &failing, false).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn level17_hook_manifest_and_10001_golden_coupling_pass() {
        let precheck = verify_hook_precheck();
        assert!(precheck.passed);
        assert_eq!(precheck.distinct_source_irreps, 4);
        assert_eq!(precheck.embedded_source_copies, 7);
        let report = build_hook_abstract("10001");
        assert!(report.passed);
        assert_eq!(report.target_dynkin_label, "11000");
        assert_eq!(report.product_weight_domain_dimension, 8);
        assert_eq!(report.kernel_dimension, 1);
        assert_eq!(
            report.primitive_domain_coefficients,
            [1, -1, 1, -1, -1, 1, -1, 1]
        );
        assert_eq!(report.exact_raising_residual_terms_by_simple_root, [0; 5]);
    }

    #[test]
    fn level17_hook_golden_gate_detects_a_coefficient_mutation() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "10001")
            .unwrap();
        let mut abstract_certificate = build_hook_abstract("10001");
        abstract_certificate.primitive_domain_coefficients[0] += 1;
        let report = verify_embedded_with_abstract(
            LEVEL17_HOOK_PROBLEM,
            &abstract_certificate,
            fixture.copy,
            fixture.artifact,
            fixture.bytes,
        );
        assert!(!report.passed);
        assert!(report
            .exact_raising_residual_terms_by_simple_root
            .iter()
            .any(|terms| *terms != 0));
    }

    #[test]
    fn direct_hook_target_blueprint_has_eight_exact_terms() {
        let terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
        assert_eq!(terms.len(), 8);
        assert_eq!(
            terms
                .iter()
                .map(|term| term.primitive_coefficient)
                .collect::<Vec<_>>(),
            [1, -1, 1, -1, -1, 1, -1, 1]
        );
        assert!(terms.iter().all(|term| add(
            term.vector_spinor_weight,
            spinor_weights()[term.outer_spinor_index]
        ) == LEVEL17_HOOK_PROBLEM.target_weight));
    }

    #[test]
    fn scalar_factorizing_candidate_is_nonzero_in_the_direct_leading_space() {
        let (candidate, maximum) = build_scalar_factorizing_candidate();
        assert!(maximum > 0);
        assert_eq!(candidate.components.len(), 32);
        assert!(candidate
            .components
            .values()
            .all(|component| !component.is_empty()));
    }

    #[test]
    fn rational_coordinate_solver_and_rank_are_exact() {
        let gram = vec![
            vec![BigInt::from(2), BigInt::from(1)],
            vec![BigInt::from(1), BigInt::from(2)],
        ];
        let coordinates = solve_bigint_system(&gram, &[BigInt::from(1), BigInt::from(0)]).unwrap();
        assert_eq!(
            coordinates,
            [
                Ratio::new(BigInt::from(2), BigInt::from(3)),
                Ratio::new(BigInt::from(-1), BigInt::from(3))
            ]
        );
        let matrix = vec![
            vec![
                Ratio::from_integer(BigInt::from(1)),
                Ratio::from_integer(BigInt::from(2)),
                Ratio::from_integer(BigInt::from(3)),
            ],
            vec![
                Ratio::from_integer(BigInt::from(0)),
                Ratio::from_integer(BigInt::from(1)),
                Ratio::from_integer(BigInt::from(1)),
            ],
        ];
        assert_eq!(rational_matrix_rank(&matrix), 2);
        let kernel = rational_nullspace(&matrix);
        assert_eq!(kernel.len(), 1);
        let primitive = primitive_bigint_vector(&kernel[0]);
        assert_eq!(
            primitive,
            [BigInt::from(1), BigInt::from(1), BigInt::from(-1)]
        );
        assert!(matrix_times_integer_vector_is_zero(&matrix, &primitive));
        let mut mutated = primitive;
        mutated[0] += BigInt::one();
        assert!(!matrix_times_integer_vector_is_zero(&matrix, &mutated));
    }
}
