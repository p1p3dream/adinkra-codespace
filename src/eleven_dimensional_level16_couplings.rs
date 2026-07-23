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
type Weight = [i8; 5];

const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];
const TARGET_WEIGHT: Weight = [3, 1, 1, 1, 1];

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

impl ExteriorModel {
    fn new() -> Self {
        let spinors = spinor_weights();
        Self {
            spinors,
            left: half_groups(0, &spinors),
            right: half_groups(16, &spinors),
            spaces: BTreeMap::new(),
            actions: BTreeMap::new(),
            maximum_absolute_accumulator: 0,
        }
    }

    fn space(&mut self, exterior_degree: u8, weight: Weight) -> &WeightSpace {
        self.spaces.entry(weight).or_insert_with(|| {
            let masks = weight_basis(exterior_degree, weight, &self.left, &self.right);
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
        let source_masks = self.space(16, source_weight).masks.clone();
        let target_index = self.space(16, target_weight).index.clone();
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
        let expected = self.space(16, weight).masks.len();
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
    dynkin_label: &str,
    fixture_copy: usize,
    fixture_bytes: &[u8],
) -> (AbstractCouplingCertificate, Vec<DenseState>) {
    let mut model = ExteriorModel::new();
    let highest = model.fixture_state(dynkin_label, fixture_bytes);
    let bases = relevant_source_bases(&mut model, highest, TARGET_WEIGHT);
    let mut domain = Vec::<(usize, DenseState)>::new();
    for (spinor_index, spinor_weight) in model.spinors.iter().copied().enumerate() {
        let source_weight = subtract(TARGET_WEIGHT, spinor_weight);
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
            schema_version: "adynkra-11d-level16-abstract-coupling-v1".to_string(),
            role: "exact canonical abstract coupling into (10001)".to_string(),
            source_dynkin_label: dynkin_label.to_string(),
            source_fixture_copy: fixture_copy,
            target_dynkin_label: TARGET_DYNKIN_LABEL.to_string(),
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

fn verify_embedded_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    fixture_bytes: &[u8],
) -> EmbeddedCouplingCertificate {
    let mut model = ExteriorModel::new();
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
        schema_version: "adynkra-11d-level16-embedded-coupling-v1".to_string(),
        role: "exact application of the shared abstract coupling to one exterior embedding"
            .to_string(),
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        target_dynkin_label: TARGET_DYNKIN_LABEL.to_string(),
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
    build_abstract_from_fixture(dynkin_label, fixture.copy, fixture.bytes).0
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
    let (abstract_certificate, _) =
        build_abstract_from_fixture(dynkin_label, abstract_fixture.copy, abstract_fixture.bytes);
    verify_embedded_with_abstract(
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
        abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn summarize_all(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
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
    let passed = distinct_source_irreps_certified == 8
        && embedded_source_copies_certified == 12
        && every_residual_is_exactly_zero;
    AllCouplingCertificateReport {
        schema_version: "adynkra-11d-level16-all-couplings-v1".to_string(),
        role: "exact dense certification of all level-16 source couplings into (10001)"
            .to_string(),
        abstract_couplings,
        embedded_copies,
        distinct_source_irreps_certified,
        embedded_source_copies_certified,
        expected_distinct_source_irreps: 8,
        expected_embedded_source_copies: 12,
        every_residual_is_exactly_zero,
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
        boundary: "this certifies the twelve source embeddings and their couplings into the (10001) channel under the stated exterior-algebra conventions; it does not solve the full Gates-Hu prepotential problem".to_string(),
        passed,
    }
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
        let (abstract_certificate, _) = build_abstract_from_fixture(label, first.copy, first.bytes);
        for fixture in copies {
            embedded_copies.push(verify_embedded_with_abstract(
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
}
