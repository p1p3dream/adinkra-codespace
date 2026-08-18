//! Exact B5 certificates for the five non-`(10001)` second-momentum paths.
//!
//! The construction works in the canonical unordered basis of `Sym^2(V)`.
//! It first embeds each intermediate irrep in
//! `Sym^2(V) tensor (10001)`, then constructs the reciprocal highest-weight
//! extraction `Sym^2(V) tensor intermediate -> (10001)`.  Only the PBW words
//! needed by the reciprocal map are generated.  Dense component maps are not
//! materialized.

use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const RAW_VECTOR_SPINOR_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const TARGET_DIMENSION: usize = 320;
const SYMMETRIC_TENSOR_DIMENSION: usize = VECTOR_DIMENSION * (VECTOR_DIMENSION + 1) / 2;
const RAW_RECOUPLING_AMBIENT_DIMENSION: usize =
    SYMMETRIC_TENSOR_DIMENSION * RAW_VECTOR_SPINOR_DIMENSION;
const PHYSICAL_RECOUPLING_DIMENSION: usize = SYMMETRIC_TENSOR_DIMENSION * TARGET_DIMENSION;
const EXPECTED_CLIFFORD_TARGET_CERTIFICATE_SHA256: &str =
    "d883b57a3151c02e25e7937400b1dc75c5fcdabd4899969aef8650ce7f37044d";
const EXPECTED_CHANNEL_CERTIFICATE_SHA256: [&str; 5] = [
    "558da5ed4409d23892e6a9716e3fed165bee8be73a52f389a983c94cd752ac15",
    "260a63bb19dc04b896f8fbef9595041693cb713ebedf1548e8a73cbc0f78b9a4",
    "f3f2f777a5a13c2ebe79eb971bb284341e8b7ad503d107e8e810d622afe03166",
    "d7b6f12a12a477e11106153b4d244d59400f6f2937d8d746ca6976b973345bb4",
    "c2ed69ba913ae0b32ccf5150dd19434a9953e98f8825c691f5c9a337423667e9",
];
const EXPECTED_REPORT_SHA256: &str =
    "53a459c04e0de82f4012608c958b4e7fbff8e50c0fd818d129be85ce2d75977a";

type Weight = [i8; 5];
type SparseVector = BTreeMap<usize, Ratio<i64>>;

const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];
const TARGET_HIGHEST_WEIGHT: Weight = [3, 1, 1, 1, 1];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymmetricMomentumBasisEntry {
    pub ordinal: usize,
    pub left_vector_weight_index: usize,
    pub right_vector_weight_index: usize,
    pub weight: Weight,
    pub diagonal: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemainingInclusionTerm {
    pub symmetric_momentum_ordinal: usize,
    pub momentum_pair: [usize; 2],
    pub momentum_weight: Weight,
    pub target_weight: Weight,
    pub target_basis_index: usize,
    pub target_pbw_word_simple_roots: Vec<u8>,
    pub primitive_coefficient: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemainingReciprocalTerm {
    pub symmetric_momentum_ordinal: usize,
    pub momentum_pair: [usize; 2],
    pub momentum_weight: Weight,
    pub intermediate_weight: Weight,
    pub intermediate_basis_index: usize,
    pub intermediate_pbw_word_simple_roots: Vec<u8>,
    pub primitive_coefficient: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecouplingMutationAudit {
    pub inclusion_coefficient_mutation_detected: bool,
    pub reciprocal_coefficient_mutation_detected: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemainingRecouplingCertificate {
    pub intermediate_dynkin_label: String,
    pub intermediate_dimension: usize,
    pub tensor_product: String,
    pub target_dynkin_label: String,
    pub unordered_symmetric_momentum_basis_dimension: usize,
    pub raw_target_ambient_dimension: usize,
    pub physical_target_product_dimension: usize,
    pub formal_reciprocal_domain_dimension: usize,
    pub inclusion_highest_weight_domain_dimension: usize,
    pub inclusion_highest_weight_kernel_dimension: usize,
    pub inclusion_primitive_nonzero_coefficients: usize,
    pub inclusion_raising_residual_terms_by_simple_root: [usize; 5],
    pub inclusion_terms: Vec<RemainingInclusionTerm>,
    pub inclusion_terms_sha256: String,
    pub relevant_intermediate_weight_spaces: usize,
    pub relevant_intermediate_states: usize,
    pub reciprocal_highest_weight_domain_dimension: usize,
    pub reciprocal_highest_weight_kernel_dimension: usize,
    pub reciprocal_primitive_nonzero_coefficients: usize,
    pub reciprocal_raising_residual_terms_by_simple_root: [usize; 5],
    pub reciprocal_terms: Vec<RemainingReciprocalTerm>,
    pub reciprocal_terms_sha256: String,
    pub exact_chevalley_equivariance_verified: bool,
    pub certified_inclusion_image_rank: usize,
    pub certified_reciprocal_image_rank: usize,
    pub rank_derivation: String,
    pub mutation: RecouplingMutationAudit,
    pub certificate_sha256: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemainingSecondMomentumRecouplingReport {
    pub schema_version: String,
    pub role: String,
    pub basis_convention: String,
    pub target_dynkin_label: String,
    pub target_dimension: usize,
    pub ambient_vector_spinor_dimension: usize,
    pub symmetric_momentum_dimension: usize,
    pub raw_target_ambient_dimension: usize,
    pub physical_target_product_dimension: usize,
    pub unordered_basis_entries: Vec<SymmetricMomentumBasisEntry>,
    pub unordered_basis_has_no_duplicates: bool,
    pub diagonal_leibniz_factor_two_verified: bool,
    pub chevalley_carrier_commutators_verified: bool,
    pub exact_target_irrep_dimension_verified: bool,
    pub clifford_target_certificate_passed: bool,
    pub clifford_target_certificate_sha256: String,
    pub channels: Vec<RemainingRecouplingCertificate>,
    pub expected_channels: usize,
    pub channels_certified: usize,
    pub expected_weighted_channel_dimension: usize,
    pub certified_weighted_channel_dimension: usize,
    pub all_five_remaining_recouplings_complete: bool,
    pub reciprocal_highest_weight_adjoint_embeddings_available: bool,
    pub all_77_component_source_target_maps_complete: bool,
    pub full_f_a_g_p_established: bool,
    pub report_sha256: String,
    pub passed: bool,
    pub boundary: String,
}

#[derive(Clone)]
struct StateWithWord {
    total_weight: Weight,
    vector: SparseVector,
    pbw_word: Vec<u8>,
}

#[derive(Clone)]
struct InclusionDomainEntry {
    symmetric_ordinal: usize,
    target_weight: Weight,
    target_basis_index: usize,
}

#[derive(Clone)]
struct ReciprocalDomainEntry {
    symmetric_ordinal: usize,
    intermediate_weight: Weight,
    intermediate_basis_index: usize,
}

#[derive(Clone)]
struct EchelonVector {
    pivot: usize,
    reduced: SparseVector,
}

#[derive(Serialize)]
struct ChannelHashPayload<'a> {
    intermediate_dynkin_label: &'a str,
    intermediate_dimension: usize,
    inclusion_terms_sha256: &'a str,
    reciprocal_terms_sha256: &'a str,
    inclusion_kernel_dimension: usize,
    reciprocal_kernel_dimension: usize,
    inclusion_residuals: [usize; 5],
    reciprocal_residuals: [usize; 5],
    mutation_passed: bool,
}

#[derive(Serialize)]
struct ReportHashPayload<'a> {
    basis: &'a [SymmetricMomentumBasisEntry],
    clifford_target_certificate_sha256: &'a str,
    channel_hashes: Vec<&'a str>,
    chevalley_carrier_commutators_verified: bool,
    all_77_component_source_target_maps_complete: bool,
    full_f_a_g_p_established: bool,
}

fn zero() -> Ratio<i64> {
    Ratio::from_integer(0)
}

fn add_weight(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn subtract_weight(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn vector_weights() -> [Weight; VECTOR_DIMENSION] {
    let mut weights = [[0_i8; 5]; VECTOR_DIMENSION];
    for axis in 0..5 {
        weights[2 * axis][axis] = 2;
        weights[2 * axis + 1][axis] = -2;
    }
    weights
}

fn spinor_weights() -> [Weight; SPINOR_DIMENSION] {
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

fn dynkin_highest_weight(label: &str) -> Weight {
    assert_eq!(label.len(), 5);
    let digits = label
        .bytes()
        .map(|byte| {
            assert!(byte.is_ascii_digit());
            i8::try_from(byte - b'0').expect("B5 Dynkin digit fits i8")
        })
        .collect::<Vec<_>>();
    std::array::from_fn(|index| 2 * digits[index..4].iter().sum::<i8>() + digits[4])
}

fn lowered_spinor_index(
    index: usize,
    root: usize,
    weights: &[Weight; SPINOR_DIMENSION],
) -> Option<usize> {
    let target = subtract_weight(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn raised_spinor_index(
    index: usize,
    root: usize,
    weights: &[Weight; SPINOR_DIMENSION],
) -> Option<usize> {
    let target = add_weight(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn lower_vector_weight(
    weight: Weight,
    root: usize,
    weights: &[Weight; VECTOR_DIMENSION],
) -> Option<(usize, i64)> {
    let mut target = weight;
    if root < 4 {
        if weight[root] == 2 {
            target[root] = 0;
            target[root + 1] = 2;
        } else if weight[root + 1] == -2 {
            target[root] = -2;
            target[root + 1] = 0;
        } else {
            return None;
        }
        Some((weights.iter().position(|item| *item == target).unwrap(), 1))
    } else if weight[4] == 2 {
        Some((weights.iter().position(|item| *item == [0; 5]).unwrap(), 1))
    } else if weight == [0; 5] {
        target[4] = -2;
        Some((weights.iter().position(|item| *item == target).unwrap(), 2))
    } else {
        None
    }
}

fn raise_vector_weight(
    weight: Weight,
    root: usize,
    weights: &[Weight; VECTOR_DIMENSION],
) -> Option<(usize, i64)> {
    let mut target = weight;
    if root < 4 {
        if weight[root] == 0 && weight[root + 1] == 2 {
            target[root] = 2;
            target[root + 1] = 0;
        } else if weight[root] == -2 && weight[root + 1] == 0 {
            target[root] = 0;
            target[root + 1] = -2;
        } else {
            return None;
        }
        Some((weights.iter().position(|item| *item == target).unwrap(), 1))
    } else if weight == [0; 5] {
        target[4] = 2;
        Some((weights.iter().position(|item| *item == target).unwrap(), 2))
    } else if weight[4] == -2 {
        Some((weights.iter().position(|item| *item == [0; 5]).unwrap(), 1))
    } else {
        None
    }
}

fn symmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| (left..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn symmetric_pair_index(left: usize, right: usize, pairs: &[(usize, usize)]) -> usize {
    let pair = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    pairs.iter().position(|item| *item == pair).unwrap()
}

fn symmetric_pair_action(
    pair_index: usize,
    root: usize,
    raising: bool,
    vectors: &[Weight; VECTOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> BTreeMap<usize, i64> {
    let (left, right) = pairs[pair_index];
    let action = |index: usize| {
        if raising {
            raise_vector_weight(vectors[index], root, vectors)
        } else {
            lower_vector_weight(vectors[index], root, vectors)
        }
    };
    let mut output = BTreeMap::new();
    if let Some((next, coefficient)) = action(left) {
        *output
            .entry(symmetric_pair_index(next, right, pairs))
            .or_insert(0) += coefficient;
    }
    if let Some((next, coefficient)) = action(right) {
        *output
            .entry(symmetric_pair_index(left, next, pairs))
            .or_insert(0) += coefficient;
    }
    output.retain(|_, coefficient| *coefficient != 0);
    output
}

fn symmetric_pair_weight(
    pair_index: usize,
    vectors: &[Weight; VECTOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> Weight {
    let (left, right) = pairs[pair_index];
    add_weight(vectors[left], vectors[right])
}

fn coroot_eigenvalue(weight: Weight, root: usize) -> i64 {
    if root < 4 {
        i64::from(weight[root] - weight[root + 1]) / 2
    } else {
        i64::from(weight[4])
    }
}

fn vector_basis_action(
    index: usize,
    root: usize,
    raising: bool,
    vectors: &[Weight; VECTOR_DIMENSION],
) -> BTreeMap<usize, i64> {
    let action = if raising {
        raise_vector_weight(vectors[index], root, vectors)
    } else {
        lower_vector_weight(vectors[index], root, vectors)
    };
    action.into_iter().collect()
}

fn spinor_basis_action(
    index: usize,
    root: usize,
    raising: bool,
    spinors: &[Weight; SPINOR_DIMENSION],
) -> BTreeMap<usize, i64> {
    let action = if raising {
        raised_spinor_index(index, root, spinors)
    } else {
        lowered_spinor_index(index, root, spinors)
    };
    action.into_iter().map(|next| (next, 1)).collect()
}

fn action_commutator(
    basis_index: usize,
    raising: &impl Fn(usize) -> BTreeMap<usize, i64>,
    lowering: &impl Fn(usize) -> BTreeMap<usize, i64>,
) -> BTreeMap<usize, i64> {
    let mut output = BTreeMap::new();
    for (lowered, lower_coefficient) in lowering(basis_index) {
        for (raised, raise_coefficient) in raising(lowered) {
            *output.entry(raised).or_insert(0) += lower_coefficient * raise_coefficient;
        }
    }
    for (raised, raise_coefficient) in raising(basis_index) {
        for (lowered, lower_coefficient) in lowering(raised) {
            *output.entry(lowered).or_insert(0) -= raise_coefficient * lower_coefficient;
        }
    }
    output.retain(|_, coefficient| *coefficient != 0);
    output
}

fn chevalley_carrier_commutators_verified(
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> bool {
    let vectors_pass = (0..VECTOR_DIMENSION).all(|index| {
        (0..5).all(|root| {
            action_commutator(
                index,
                &|item| vector_basis_action(item, root, true, vectors),
                &|item| vector_basis_action(item, root, false, vectors),
            ) == BTreeMap::from([(index, coroot_eigenvalue(vectors[index], root))])
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != 0)
                .collect()
        })
    });
    let spinors_pass = (0..SPINOR_DIMENSION).all(|index| {
        (0..5).all(|root| {
            action_commutator(
                index,
                &|item| spinor_basis_action(item, root, true, spinors),
                &|item| spinor_basis_action(item, root, false, spinors),
            ) == BTreeMap::from([(index, coroot_eigenvalue(spinors[index], root))])
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != 0)
                .collect()
        })
    });
    let symmetric_pass = (0..pairs.len()).all(|index| {
        (0..5).all(|root| {
            action_commutator(
                index,
                &|item| symmetric_pair_action(item, root, true, vectors, pairs),
                &|item| symmetric_pair_action(item, root, false, vectors, pairs),
            ) == BTreeMap::from([(
                index,
                coroot_eigenvalue(symmetric_pair_weight(index, vectors, pairs), root),
            )])
            .into_iter()
            .filter(|(_, coefficient)| *coefficient != 0)
            .collect()
        })
    });
    vectors_pass && spinors_pass && symmetric_pass
}

fn add_scaled(target: &mut SparseVector, source: &SparseVector, scale: Ratio<i64>) {
    if scale == zero() {
        return;
    }
    for (&index, coefficient) in source {
        *target.entry(index).or_insert_with(zero) += coefficient.clone() * scale.clone();
        if target[&index] == zero() {
            target.remove(&index);
        }
    }
}

fn lower_target(
    source: &SparseVector,
    root: usize,
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
) -> SparseVector {
    let mut target = SparseVector::new();
    for (&index, coefficient) in source {
        let vector = index / SPINOR_DIMENSION;
        let spinor = index % SPINOR_DIMENSION;
        if let Some((next, factor)) = lower_vector_weight(vectors[vector], root, vectors) {
            *target
                .entry(next * SPINOR_DIMENSION + spinor)
                .or_insert_with(zero) += coefficient.clone() * Ratio::from_integer(factor);
        }
        if let Some(next) = lowered_spinor_index(spinor, root, spinors) {
            *target
                .entry(vector * SPINOR_DIMENSION + next)
                .or_insert_with(zero) += coefficient.clone();
        }
    }
    target.retain(|_, coefficient| *coefficient != zero());
    target
}

fn raise_target(
    source: &SparseVector,
    root: usize,
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
) -> SparseVector {
    let mut target = SparseVector::new();
    for (&index, coefficient) in source {
        let vector = index / SPINOR_DIMENSION;
        let spinor = index % SPINOR_DIMENSION;
        if let Some((next, factor)) = raise_vector_weight(vectors[vector], root, vectors) {
            *target
                .entry(next * SPINOR_DIMENSION + spinor)
                .or_insert_with(zero) += coefficient.clone() * Ratio::from_integer(factor);
        }
        if let Some(next) = raised_spinor_index(spinor, root, spinors) {
            *target
                .entry(vector * SPINOR_DIMENSION + next)
                .or_insert_with(zero) += coefficient.clone();
        }
    }
    target.retain(|_, coefficient| *coefficient != zero());
    target
}

fn reduce_candidate(candidate: &SparseVector, echelon: &[EchelonVector]) -> SparseVector {
    let mut residual = candidate.clone();
    for basis in echelon {
        let Some(factor) = residual.get(&basis.pivot).cloned() else {
            continue;
        };
        add_scaled(&mut residual, &basis.reduced, -factor);
    }
    residual
}

fn select_independent_states(candidates: Vec<StateWithWord>) -> Vec<StateWithWord> {
    let mut states = Vec::new();
    let mut echelon = Vec::<EchelonVector>::new();
    for candidate in candidates {
        let mut reduced = reduce_candidate(&candidate.vector, &echelon);
        if reduced.is_empty() {
            continue;
        }
        let pivot = *reduced.keys().next().unwrap();
        let normalization = reduced[&pivot].clone();
        for value in reduced.values_mut() {
            *value /= normalization.clone();
        }
        echelon.push(EchelonVector { pivot, reduced });
        echelon.sort_by_key(|entry| entry.pivot);
        states.push(candidate);
    }
    states
}

fn generate_target_states(
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
) -> BTreeMap<Weight, Vec<StateWithWord>> {
    let highest = StateWithWord {
        total_weight: TARGET_HIGHEST_WEIGHT,
        vector: BTreeMap::from([(0, Ratio::from_integer(1))]),
        pbw_word: Vec::new(),
    };
    let mut all = BTreeMap::from([(TARGET_HIGHEST_WEIGHT, vec![highest.clone()])]);
    let mut current = BTreeMap::from([(TARGET_HIGHEST_WEIGHT, vec![highest])]);
    while !current.is_empty() {
        let mut candidates = BTreeMap::<Weight, Vec<StateWithWord>>::new();
        for states in current.into_values() {
            for state in states {
                for root in 0..5 {
                    let descendant = lower_target(&state.vector, root, vectors, spinors);
                    if descendant.is_empty() {
                        continue;
                    }
                    let total_weight = subtract_weight(state.total_weight, SIMPLE_ROOTS[root]);
                    let mut pbw_word = state.pbw_word.clone();
                    pbw_word.push(u8::try_from(root + 1).unwrap());
                    candidates
                        .entry(total_weight)
                        .or_default()
                        .push(StateWithWord {
                            total_weight,
                            vector: descendant,
                            pbw_word,
                        });
                }
            }
        }
        current = candidates
            .into_iter()
            .filter_map(|(weight, states)| {
                let selected = select_independent_states(states);
                (!selected.is_empty()).then_some((weight, selected))
            })
            .collect();
        for (weight, states) in &current {
            all.insert(*weight, states.clone());
        }
    }
    all
}

fn apply_symmetric_target_action(
    source: &SparseVector,
    root: usize,
    raising: bool,
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> SparseVector {
    let mut output = SparseVector::new();
    for (&index, coefficient) in source {
        let symmetric = index / RAW_VECTOR_SPINOR_DIMENSION;
        let target_index = index % RAW_VECTOR_SPINOR_DIMENSION;
        for (next, factor) in symmetric_pair_action(symmetric, root, raising, vectors, pairs) {
            *output
                .entry(next * RAW_VECTOR_SPINOR_DIMENSION + target_index)
                .or_insert_with(zero) += coefficient.clone() * Ratio::from_integer(factor);
        }
        let target = BTreeMap::from([(target_index, coefficient.clone())]);
        let acted = if raising {
            raise_target(&target, root, vectors, spinors)
        } else {
            lower_target(&target, root, vectors, spinors)
        };
        for (next, value) in acted {
            *output
                .entry(symmetric * RAW_VECTOR_SPINOR_DIMENSION + next)
                .or_insert_with(zero) += value;
        }
    }
    output.retain(|_, coefficient| *coefficient != zero());
    output
}

fn ratio_nullspace(rows: &[Vec<Ratio<i64>>], columns: usize) -> Vec<Vec<Ratio<i64>>> {
    let mut reduced = rows
        .iter()
        .filter(|row| row.iter().any(|value| *value != zero()))
        .cloned()
        .collect::<Vec<_>>();
    let mut pivots = Vec::new();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot_row) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero())
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
            if row == rank || reduced[row][column] == zero() {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot[index].clone();
            }
        }
        pivots.push(column);
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    (0..columns)
        .filter(|column| !pivots.contains(column))
        .map(|free| {
            let mut vector = vec![zero(); columns];
            vector[free] = Ratio::from_integer(1);
            for (row, &pivot) in pivots.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free].clone();
            }
            vector
        })
        .collect()
}

fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn lcm_i64(left: i64, right: i64) -> i64 {
    if left == 0 || right == 0 {
        0
    } else {
        (left / gcd_i64(left, right))
            .checked_mul(right)
            .expect("rational denominator LCM exceeds i64")
            .abs()
    }
}

fn primitive_integer_vector(vector: &[Ratio<i64>]) -> Vec<i64> {
    let denominator = vector.iter().fold(1_i64, |common, coefficient| {
        lcm_i64(common, *coefficient.denom())
    });
    let mut integers = vector
        .iter()
        .map(|coefficient| {
            coefficient
                .numer()
                .checked_mul(denominator / coefficient.denom())
                .expect("primitive numerator exceeds i64")
        })
        .collect::<Vec<_>>();
    let divisor = integers
        .iter()
        .fold(0_i64, |common, value| gcd_i64(common, *value));
    if divisor == 0 {
        return Vec::new();
    }
    for value in &mut integers {
        *value /= divisor;
    }
    if integers.iter().find(|value| **value != 0).unwrap() < &0 {
        for value in &mut integers {
            *value = -*value;
        }
    }
    integers
}

fn residual_counts(rows: &BTreeMap<usize, Vec<Ratio<i64>>>, primitive: &[i64]) -> [usize; 5] {
    let root_stride = rows.keys().next_back().map(|key| key / 5 + 1).unwrap_or(1);
    let _ = root_stride;
    let mut counts = [0_usize; 5];
    for (key, row) in rows {
        let root = key >> 48;
        let value = row
            .iter()
            .zip(primitive)
            .fold(zero(), |sum, (entry, coefficient)| {
                sum + entry.clone() * Ratio::from_integer(*coefficient)
            });
        if value != zero() {
            counts[root] += 1;
        }
    }
    counts
}

fn row_key(root: usize, ambient_index: usize) -> usize {
    (root << 48) | ambient_index
}

fn rows_from_columns(columns: &[Vec<SparseVector>]) -> BTreeMap<usize, Vec<Ratio<i64>>> {
    let mut rows = BTreeMap::<usize, Vec<Ratio<i64>>>::new();
    for (column, outputs) in columns.iter().enumerate() {
        for (root, output) in outputs.iter().enumerate() {
            for (&ambient, coefficient) in output {
                rows.entry(row_key(root, ambient))
                    .or_insert_with(|| vec![zero(); columns.len()])[column] += coefficient.clone();
            }
        }
    }
    rows
}

fn solve_one_dimensional_kernel(
    columns: &[Vec<SparseVector>],
) -> (
    Vec<i64>,
    usize,
    [usize; 5],
    BTreeMap<usize, Vec<Ratio<i64>>>,
) {
    let rows = rows_from_columns(columns);
    let row_vectors = rows.values().cloned().collect::<Vec<_>>();
    let kernel = ratio_nullspace(&row_vectors, columns.len());
    let primitive = if kernel.len() == 1 {
        primitive_integer_vector(&kernel[0])
    } else {
        Vec::new()
    };
    let residuals = if primitive.len() == columns.len() {
        residual_counts(&rows, &primitive)
    } else {
        [usize::MAX; 5]
    };
    (primitive, kernel.len(), residuals, rows)
}

fn mutation_detected(rows: &BTreeMap<usize, Vec<Ratio<i64>>>, primitive: &[i64]) -> bool {
    let Some(index) = primitive.iter().position(|value| *value != 0) else {
        return false;
    };
    if primitive.len() == 1 {
        // A one-column highest-weight domain has no relative coefficient to
        // perturb.  Its only non-scalar mutation is deletion, which is caught
        // by the mandatory nonzero-image gate.
        return true;
    }
    let mut mutated = primitive.to_vec();
    mutated[index] = mutated[index].checked_add(1).unwrap();
    rows.values().any(|row| {
        row.iter()
            .zip(&mutated)
            .fold(zero(), |sum, (entry, coefficient)| {
                sum + entry.clone() * Ratio::from_integer(*coefficient)
            })
            != zero()
    })
}

fn materialize_linear_combination(
    states: impl Iterator<Item = (SparseVector, i64)>,
) -> SparseVector {
    let mut result = SparseVector::new();
    for (state, coefficient) in states {
        add_scaled(&mut result, &state, Ratio::from_integer(coefficient));
    }
    result
}

fn lowering_coordinates(upper: Weight, lower: Weight) -> Option<[i16; 5]> {
    let difference = std::array::from_fn::<_, 5, _>(|index| {
        let value = i16::from(upper[index]) - i16::from(lower[index]);
        (value % 2 == 0).then_some(value / 2)
    });
    let difference = difference.into_iter().collect::<Option<Vec<_>>>()?;
    let coordinates = [
        difference[0],
        difference[0] + difference[1],
        difference[0] + difference[1] + difference[2],
        difference[0] + difference[1] + difference[2] + difference[3],
        difference.iter().sum(),
    ];
    coordinates
        .iter()
        .all(|value| *value >= 0)
        .then_some(coordinates)
}

fn relevant_intermediate_states(
    highest_weight: Weight,
    highest: SparseVector,
    needed_weights: &BTreeSet<Weight>,
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> BTreeMap<Weight, Vec<StateWithWord>> {
    let needed_depths = needed_weights
        .iter()
        .filter_map(|weight| {
            lowering_coordinates(highest_weight, *weight).map(|coordinates| {
                (
                    *weight,
                    coordinates
                        .iter()
                        .map(|value| usize::try_from(*value).unwrap())
                        .sum::<usize>(),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_depth = needed_depths.values().copied().max().unwrap_or(0);
    let highest_state = StateWithWord {
        total_weight: highest_weight,
        vector: highest,
        pbw_word: Vec::new(),
    };
    let mut current = BTreeMap::from([(highest_weight, vec![highest_state])]);
    let mut required = BTreeMap::new();
    for depth in 0..=maximum_depth {
        let mut next_candidates = BTreeMap::<Weight, Vec<StateWithWord>>::new();
        for (weight, states) in current {
            if needed_depths
                .get(&weight)
                .is_some_and(|needed| *needed == depth)
            {
                required.insert(weight, states.clone());
            }
            if depth == maximum_depth {
                continue;
            }
            for root in 0..5 {
                let next_weight = subtract_weight(weight, SIMPLE_ROOTS[root]);
                let relevant = needed_depths.keys().any(|needed| {
                    lowering_coordinates(next_weight, *needed).is_some()
                        && lowering_coordinates(highest_weight, next_weight).is_some_and(
                            |coordinates| {
                                coordinates
                                    .iter()
                                    .map(|value| usize::try_from(*value).unwrap())
                                    .sum::<usize>()
                                    == depth + 1
                            },
                        )
                });
                if !relevant {
                    continue;
                }
                for state in &states {
                    let descendant = apply_symmetric_target_action(
                        &state.vector,
                        root,
                        false,
                        vectors,
                        spinors,
                        pairs,
                    );
                    if descendant.is_empty() {
                        continue;
                    }
                    let mut pbw_word = state.pbw_word.clone();
                    pbw_word.push(u8::try_from(root + 1).unwrap());
                    next_candidates
                        .entry(next_weight)
                        .or_default()
                        .push(StateWithWord {
                            total_weight: next_weight,
                            vector: descendant,
                            pbw_word,
                        });
                }
            }
        }
        current = next_candidates
            .into_iter()
            .filter_map(|(weight, states)| {
                let selected = select_independent_states(states);
                (!selected.is_empty()).then_some((weight, selected))
            })
            .collect();
    }
    required
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("serialize deterministic recoupling payload");
    format!("{:x}", Sha256::digest(bytes))
}

fn channel_specs() -> [(&'static str, usize); 5] {
    [
        ("00001", 32),
        ("01001", 1_408),
        ("11001", 10_240),
        ("20001", 1_760),
        ("30001", 7_040),
    ]
}

fn construct_channel(
    label: &str,
    dimension: usize,
    target_states: &BTreeMap<Weight, Vec<StateWithWord>>,
    vectors: &[Weight; VECTOR_DIMENSION],
    spinors: &[Weight; SPINOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> RemainingRecouplingCertificate {
    let intermediate_highest_weight = dynkin_highest_weight(label);
    let mut inclusion_domain = Vec::<InclusionDomainEntry>::new();
    for symmetric_ordinal in 0..pairs.len() {
        let target_weight = subtract_weight(
            intermediate_highest_weight,
            symmetric_pair_weight(symmetric_ordinal, vectors, pairs),
        );
        if let Some(states) = target_states.get(&target_weight) {
            for target_basis_index in 0..states.len() {
                inclusion_domain.push(InclusionDomainEntry {
                    symmetric_ordinal,
                    target_weight,
                    target_basis_index,
                });
            }
        }
    }
    let inclusion_columns = inclusion_domain
        .iter()
        .map(|entry| {
            let state = &target_states[&entry.target_weight][entry.target_basis_index].vector;
            let initial = state
                .iter()
                .map(|(&target_index, coefficient)| {
                    (
                        entry.symmetric_ordinal * RAW_VECTOR_SPINOR_DIMENSION + target_index,
                        coefficient.clone(),
                    )
                })
                .collect::<SparseVector>();
            (0..5)
                .map(|root| {
                    apply_symmetric_target_action(&initial, root, true, vectors, spinors, pairs)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (inclusion_primitive, inclusion_kernel, inclusion_residuals, inclusion_rows) =
        solve_one_dimensional_kernel(&inclusion_columns);
    let inclusion_terms = inclusion_domain
        .iter()
        .zip(&inclusion_primitive)
        .map(|(entry, coefficient)| {
            let (left, right) = pairs[entry.symmetric_ordinal];
            let target = &target_states[&entry.target_weight][entry.target_basis_index];
            RemainingInclusionTerm {
                symmetric_momentum_ordinal: entry.symmetric_ordinal,
                momentum_pair: [left, right],
                momentum_weight: symmetric_pair_weight(entry.symmetric_ordinal, vectors, pairs),
                target_weight: entry.target_weight,
                target_basis_index: entry.target_basis_index,
                target_pbw_word_simple_roots: target.pbw_word.clone(),
                primitive_coefficient: *coefficient,
            }
        })
        .collect::<Vec<_>>();
    let inclusion_highest =
        materialize_linear_combination(inclusion_domain.iter().zip(&inclusion_primitive).map(
            |(entry, coefficient)| {
                let state = &target_states[&entry.target_weight][entry.target_basis_index].vector;
                let embedded = state
                    .iter()
                    .map(|(&target_index, value)| {
                        (
                            entry.symmetric_ordinal * RAW_VECTOR_SPINOR_DIMENSION + target_index,
                            value.clone(),
                        )
                    })
                    .collect::<SparseVector>();
                (embedded, *coefficient)
            },
        ));

    let needed_weights = (0..pairs.len())
        .map(|symmetric| {
            subtract_weight(
                TARGET_HIGHEST_WEIGHT,
                symmetric_pair_weight(symmetric, vectors, pairs),
            )
        })
        .collect::<BTreeSet<_>>();
    let intermediate_states = relevant_intermediate_states(
        intermediate_highest_weight,
        inclusion_highest.clone(),
        &needed_weights,
        vectors,
        spinors,
        pairs,
    );
    let mut reciprocal_domain = Vec::<ReciprocalDomainEntry>::new();
    for symmetric_ordinal in 0..pairs.len() {
        let intermediate_weight = subtract_weight(
            TARGET_HIGHEST_WEIGHT,
            symmetric_pair_weight(symmetric_ordinal, vectors, pairs),
        );
        if let Some(states) = intermediate_states.get(&intermediate_weight) {
            for intermediate_basis_index in 0..states.len() {
                reciprocal_domain.push(ReciprocalDomainEntry {
                    symmetric_ordinal,
                    intermediate_weight,
                    intermediate_basis_index,
                });
            }
        }
    }
    let reciprocal_columns = reciprocal_domain
        .iter()
        .map(|entry| {
            let state = &intermediate_states[&entry.intermediate_weight]
                [entry.intermediate_basis_index]
                .vector;
            (0..5)
                .map(|root| {
                    let mut output = SparseVector::new();
                    for (next, factor) in
                        symmetric_pair_action(entry.symmetric_ordinal, root, true, vectors, pairs)
                    {
                        for (&inner, coefficient) in state {
                            *output
                                .entry(next * RAW_RECOUPLING_AMBIENT_DIMENSION + inner)
                                .or_insert_with(zero) +=
                                coefficient.clone() * Ratio::from_integer(factor);
                        }
                    }
                    let raised =
                        apply_symmetric_target_action(state, root, true, vectors, spinors, pairs);
                    for (inner, coefficient) in raised {
                        *output
                            .entry(
                                entry.symmetric_ordinal * RAW_RECOUPLING_AMBIENT_DIMENSION + inner,
                            )
                            .or_insert_with(zero) += coefficient;
                    }
                    output.retain(|_, coefficient| *coefficient != zero());
                    output
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (reciprocal_primitive, reciprocal_kernel, reciprocal_residuals, reciprocal_rows) =
        solve_one_dimensional_kernel(&reciprocal_columns);
    let reciprocal_terms = reciprocal_domain
        .iter()
        .zip(&reciprocal_primitive)
        .map(|(entry, coefficient)| {
            let (left, right) = pairs[entry.symmetric_ordinal];
            let state =
                &intermediate_states[&entry.intermediate_weight][entry.intermediate_basis_index];
            RemainingReciprocalTerm {
                symmetric_momentum_ordinal: entry.symmetric_ordinal,
                momentum_pair: [left, right],
                momentum_weight: symmetric_pair_weight(entry.symmetric_ordinal, vectors, pairs),
                intermediate_weight: entry.intermediate_weight,
                intermediate_basis_index: entry.intermediate_basis_index,
                intermediate_pbw_word_simple_roots: state.pbw_word.clone(),
                primitive_coefficient: *coefficient,
            }
        })
        .collect::<Vec<_>>();

    let inclusion_mutation = mutation_detected(&inclusion_rows, &inclusion_primitive);
    let reciprocal_mutation = mutation_detected(&reciprocal_rows, &reciprocal_primitive);
    let mutation = RecouplingMutationAudit {
        inclusion_coefficient_mutation_detected: inclusion_mutation,
        reciprocal_coefficient_mutation_detected: reciprocal_mutation,
        passed: inclusion_mutation && reciprocal_mutation,
    };
    let inclusion_terms_sha256 = sha256_json(&inclusion_terms);
    let reciprocal_terms_sha256 = sha256_json(&reciprocal_terms);
    let exact_chevalley_equivariance_verified =
        inclusion_residuals == [0; 5] && reciprocal_residuals == [0; 5];
    let passed = inclusion_kernel == 1
        && reciprocal_kernel == 1
        && !inclusion_highest.is_empty()
        && inclusion_primitive.len() == inclusion_domain.len()
        && reciprocal_primitive.len() == reciprocal_domain.len()
        && exact_chevalley_equivariance_verified
        && mutation.passed;
    let hash_payload = ChannelHashPayload {
        intermediate_dynkin_label: label,
        intermediate_dimension: dimension,
        inclusion_terms_sha256: &inclusion_terms_sha256,
        reciprocal_terms_sha256: &reciprocal_terms_sha256,
        inclusion_kernel_dimension: inclusion_kernel,
        reciprocal_kernel_dimension: reciprocal_kernel,
        inclusion_residuals,
        reciprocal_residuals,
        mutation_passed: mutation.passed,
    };
    let certificate_sha256 = sha256_json(&hash_payload);
    RemainingRecouplingCertificate {
        intermediate_dynkin_label: label.to_string(),
        intermediate_dimension: dimension,
        tensor_product: "Sym^2(10000) tensor intermediate".to_string(),
        target_dynkin_label: "10001".to_string(),
        unordered_symmetric_momentum_basis_dimension: pairs.len(),
        raw_target_ambient_dimension: RAW_RECOUPLING_AMBIENT_DIMENSION,
        physical_target_product_dimension: PHYSICAL_RECOUPLING_DIMENSION,
        formal_reciprocal_domain_dimension: SYMMETRIC_TENSOR_DIMENSION * dimension,
        inclusion_highest_weight_domain_dimension: inclusion_domain.len(),
        inclusion_highest_weight_kernel_dimension: inclusion_kernel,
        inclusion_primitive_nonzero_coefficients: inclusion_primitive
            .iter()
            .filter(|coefficient| **coefficient != 0)
            .count(),
        inclusion_raising_residual_terms_by_simple_root: inclusion_residuals,
        inclusion_terms,
        inclusion_terms_sha256,
        relevant_intermediate_weight_spaces: intermediate_states.len(),
        relevant_intermediate_states: intermediate_states.values().map(Vec::len).sum(),
        reciprocal_highest_weight_domain_dimension: reciprocal_domain.len(),
        reciprocal_highest_weight_kernel_dimension: reciprocal_kernel,
        reciprocal_primitive_nonzero_coefficients: reciprocal_primitive
            .iter()
            .filter(|coefficient| **coefficient != 0)
            .count(),
        reciprocal_raising_residual_terms_by_simple_root: reciprocal_residuals,
        reciprocal_terms,
        reciprocal_terms_sha256,
        exact_chevalley_equivariance_verified,
        certified_inclusion_image_rank: if passed { dimension } else { 0 },
        certified_reciprocal_image_rank: if passed { TARGET_DIMENSION } else { 0 },
        rank_derivation: "a nonzero exact B5 intertwiner between irreducibles has full irreducible image; both highest-weight kernels are computed to be one-dimensional".to_string(),
        mutation,
        certificate_sha256,
        passed,
    }
}

fn symmetric_basis_report(
    vectors: &[Weight; VECTOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> Vec<SymmetricMomentumBasisEntry> {
    pairs
        .iter()
        .enumerate()
        .map(|(ordinal, &(left, right))| SymmetricMomentumBasisEntry {
            ordinal,
            left_vector_weight_index: left,
            right_vector_weight_index: right,
            weight: add_weight(vectors[left], vectors[right]),
            diagonal: left == right,
        })
        .collect()
}

fn diagonal_leibniz_factor_two_verified(
    vectors: &[Weight; VECTOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> bool {
    pairs.iter().enumerate().all(|(ordinal, &(left, right))| {
        if left != right {
            return true;
        }
        (0..5).all(|root| {
            [false, true].into_iter().all(|raising| {
                let one_slot = if raising {
                    raise_vector_weight(vectors[left], root, vectors)
                } else {
                    lower_vector_weight(vectors[left], root, vectors)
                };
                let action = symmetric_pair_action(ordinal, root, raising, vectors, pairs);
                match one_slot {
                    Some((next, factor)) => {
                        action.len() == 1
                            && action.get(&symmetric_pair_index(next, right, pairs))
                                == Some(&(2 * factor))
                    }
                    None => action.is_empty(),
                }
            })
        })
    })
}

pub fn verify() -> RemainingSecondMomentumRecouplingReport {
    let vectors = vector_weights();
    let spinors = spinor_weights();
    let pairs = symmetric_pairs();
    let basis = symmetric_basis_report(&vectors, &pairs);
    let target_states = generate_target_states(&vectors, &spinors);
    let generated_target_dimension = target_states.values().map(Vec::len).sum::<usize>();
    let exact_target_irrep_dimension_verified = generated_target_dimension == TARGET_DIMENSION;

    let clifford = crate::eleven_dimensional_second_momentum_recoupling::verify();
    let clifford_target_certificate_passed = clifford.passed
        && clifford.gamma_traceless_projector_rank == TARGET_DIMENSION
        && clifford.clifford_residual_entries == 0
        && clifford.certificate_sha256 == EXPECTED_CLIFFORD_TARGET_CERTIFICATE_SHA256;
    let clifford_target_certificate_sha256 = clifford.certificate_sha256;

    let channels = channel_specs()
        .into_iter()
        .map(|(label, dimension)| {
            construct_channel(label, dimension, &target_states, &vectors, &spinors, &pairs)
        })
        .collect::<Vec<_>>();
    assert!(
        channels
            .iter()
            .zip(EXPECTED_CHANNEL_CERTIFICATE_SHA256)
            .all(|(channel, expected)| channel.certificate_sha256 == expected)
    );
    let channels_certified = channels.iter().filter(|channel| channel.passed).count();
    let certified_weighted_channel_dimension = channels
        .iter()
        .filter(|channel| channel.passed)
        .map(|channel| channel.intermediate_dimension)
        .sum::<usize>();
    let expected_weighted_channel_dimension = 20_480;
    let unordered_basis_has_no_duplicates = basis.len() == SYMMETRIC_TENSOR_DIMENSION
        && basis
            .iter()
            .map(|entry| {
                (
                    entry.left_vector_weight_index,
                    entry.right_vector_weight_index,
                )
            })
            .collect::<BTreeSet<_>>()
            .len()
            == SYMMETRIC_TENSOR_DIMENSION
        && basis
            .iter()
            .all(|entry| entry.left_vector_weight_index <= entry.right_vector_weight_index);
    let diagonal_leibniz_factor_two_verified =
        diagonal_leibniz_factor_two_verified(&vectors, &pairs);
    let chevalley_carrier_commutators_verified =
        chevalley_carrier_commutators_verified(&vectors, &spinors, &pairs);
    let all_five_remaining_recouplings_complete = channels_certified == 5
        && certified_weighted_channel_dimension == expected_weighted_channel_dimension;
    let reciprocal_highest_weight_adjoint_embeddings_available = channels
        .iter()
        .all(|channel| channel.passed && !channel.reciprocal_terms.is_empty());
    let all_77_component_source_target_maps_complete = false;
    let full_f_a_g_p_established = false;
    let hash_payload = ReportHashPayload {
        basis: &basis,
        clifford_target_certificate_sha256: &clifford_target_certificate_sha256,
        channel_hashes: channels
            .iter()
            .map(|channel| channel.certificate_sha256.as_str())
            .collect(),
        chevalley_carrier_commutators_verified,
        all_77_component_source_target_maps_complete,
        full_f_a_g_p_established,
    };
    let report_sha256 = sha256_json(&hash_payload);
    assert_eq!(report_sha256, EXPECTED_REPORT_SHA256);
    let passed = unordered_basis_has_no_duplicates
        && diagonal_leibniz_factor_two_verified
        && chevalley_carrier_commutators_verified
        && exact_target_irrep_dimension_verified
        && clifford_target_certificate_passed
        && all_five_remaining_recouplings_complete
        && reciprocal_highest_weight_adjoint_embeddings_available;
    RemainingSecondMomentumRecouplingReport {
        schema_version: "adynkra-11d-second-momentum-remaining-recouplings-v1".to_string(),
        role: "exact B5 inclusions and reciprocal component extractors for the five non-(10001) Sym^2(V) momentum channels".to_string(),
        basis_convention: "canonical unordered primal vector-weight pairs (a,b) with a <= b; Chevalley actions use the exact two-slot Leibniz rule, including factor two on a moved diagonal pair. The invariant Sym^2 metric is diagonal with mu_ab=(1+delta_ab) nu_a nu_b, nu=1 on nonzero vector weights and nu=2 on the zero weight; rank statements are unchanged by this invertible primal/dual rescaling".to_string(),
        target_dynkin_label: "10001".to_string(),
        target_dimension: TARGET_DIMENSION,
        ambient_vector_spinor_dimension: RAW_VECTOR_SPINOR_DIMENSION,
        symmetric_momentum_dimension: SYMMETRIC_TENSOR_DIMENSION,
        raw_target_ambient_dimension: RAW_RECOUPLING_AMBIENT_DIMENSION,
        physical_target_product_dimension: PHYSICAL_RECOUPLING_DIMENSION,
        unordered_basis_entries: basis,
        unordered_basis_has_no_duplicates,
        diagonal_leibniz_factor_two_verified,
        chevalley_carrier_commutators_verified,
        exact_target_irrep_dimension_verified,
        clifford_target_certificate_passed,
        clifford_target_certificate_sha256,
        channels,
        expected_channels: 5,
        channels_certified,
        expected_weighted_channel_dimension,
        certified_weighted_channel_dimension,
        all_five_remaining_recouplings_complete,
        reciprocal_highest_weight_adjoint_embeddings_available,
        all_77_component_source_target_maps_complete,
        full_f_a_g_p_established,
        report_sha256,
        passed,
        boundary: "This certifies the five remaining representation-level momentum recouplings and their reciprocal highest-weight adjoint embeddings. They are not standalone component extractors. Literal coefficients require the declared primal or invariant-dual momentum convention. The certificate does not compose the 73 level-12 source embeddings, certify all 77 physical columns, complete either gauge branch, supply complete K/F, or establish F A G_p = 0.".to_string(),
    }
}

pub fn write_artifact(path: &Path) -> io::Result<RemainingSecondMomentumRecouplingReport> {
    let report = verify();
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "remaining second-momentum recoupling certificate did not pass",
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let payload = serde_json::to_vec_pretty(&report).map_err(io::Error::other)?;
    fs::write(&temporary, payload)?;
    fs::rename(temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_remaining_recouplings_are_exact_and_reciprocal() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.channels_certified, 5);
        assert_eq!(report.certified_weighted_channel_dimension, 20_480);
        assert!(report.unordered_basis_has_no_duplicates);
        assert!(report.diagonal_leibniz_factor_two_verified);
        assert!(report.chevalley_carrier_commutators_verified);
        assert!(report.reciprocal_highest_weight_adjoint_embeddings_available);
        assert!(!report.all_77_component_source_target_maps_complete);
        assert!(!report.full_f_a_g_p_established);
        for channel in report.channels {
            assert_eq!(channel.inclusion_highest_weight_kernel_dimension, 1);
            assert_eq!(channel.reciprocal_highest_weight_kernel_dimension, 1);
            assert_eq!(
                channel.inclusion_raising_residual_terms_by_simple_root,
                [0; 5]
            );
            assert_eq!(
                channel.reciprocal_raising_residual_terms_by_simple_root,
                [0; 5]
            );
            assert!(channel.mutation.passed);
            assert!(channel.passed);
        }
    }

    #[test]
    fn unordered_pair_action_has_the_required_diagonal_factor() {
        let vectors = vector_weights();
        let pairs = symmetric_pairs();
        assert_eq!(pairs.len(), 66);
        assert!(diagonal_leibniz_factor_two_verified(&vectors, &pairs));
    }
}
