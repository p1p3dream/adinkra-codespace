//! Exact highest-weight checkpoints for the `(10002) -> (10001)` part of
//! the 11D second-momentum operator inventory.
//!
//! Two independent level-12 `(10002)` embeddings are coupled to the unique
//! `(10001)` summand in `(10002) tensor S`.  Each embedded highest-weight
//! state is then paired with the independently certified trace and
//! symmetric-traceless momentum paths.  This gives four exact map seeds.
//! The remaining `(00100)` and `(00010)` source labels, descendant component
//! maps, and the physical `F A G_p` calculation remain fail-closed.

use num_bigint::BigInt;
use num_traits::Zero;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

type Weight = [i8; 5];

const EXTERIOR_DEGREE: u8 = 12;
const SOURCE_DYNKIN_LABEL: &str = "10002";
const INTERMEDIATE_DYNKIN_LABEL: &str = "10001";
const TARGET_WEIGHT: Weight = [3, 1, 1, 1, 1];
const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];

const ABSTRACT_COUPLING_JSON: &str =
    include_str!("../results/adynkra_11d_first_momentum_10001_from_10002_abstract.json");
const ABSTRACT_COUPLING_SHA256: &str =
    "d151f0a22dd9738148b7b28650a97e3b8687d27691ff9e7dacc5180a65a7aca8";
const SOURCE_KERNELS: [&[u8]; 2] = [
    include_bytes!(
        "../data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_1.i16le"
    ),
    include_bytes!(
        "../data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_2.i16le"
    ),
];
const SOURCE_KERNEL_SHA256: [&str; 2] = [
    "c3eb687d6d868cd08fcd90a0c741815681b3c68ca4cf3157ddd929df0aa42e28",
    "ec51e06970fcff1b7b719d7ee5c3d9a69775fdf6558334c400d1540dbd81c7a1",
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecondMomentum10001Path {
    Trace,
    SymmetricTraceless,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentum10001MapSpec {
    pub variable_ordinal: usize,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub intermediate_dynkin_label: String,
    pub momentum_path: SecondMomentum10001Path,
    pub momentum_degree: usize,
    pub exterior_derivative_order: usize,
    pub gauge_composition_wedge_order: usize,
    pub highest_weight_checkpoint_available: bool,
    pub all_descendant_component_maps_complete: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentum10001SourceTerm {
    pub source_copy: usize,
    pub intermediate_spinor_weight_index: usize,
    pub exterior_mask: u32,
    pub coefficient: i128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentum10001RecouplingPath {
    pub path: SecondMomentum10001Path,
    pub image_rank: usize,
    pub trace_delta_coefficient: i64,
    pub delta_embedding_coefficient: i64,
    pub gamma_embedding_coefficient: i64,
    pub momentum_trace_subtraction_coefficient: i64,
    pub target_gamma_residual_entries: usize,
    pub momentum_trace_residual_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentum10001EmbeddedSourceAudit {
    pub source_copy: usize,
    pub fixture_sha256: String,
    pub abstract_coupling_sha256: String,
    pub abstract_domain_dimension: usize,
    pub coupled_nonzero_terms: usize,
    pub coupled_map_sha256: String,
    pub exact_raising_residual_terms_by_simple_root: [usize; 5],
    pub maximum_absolute_coefficient: i128,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentum10001MapReport {
    pub schema_version: String,
    pub role: String,
    pub source_dynkin_label: String,
    pub intermediate_dynkin_label: String,
    pub abstract_coupling_transfer_justification: String,
    pub source_kernel_copies: usize,
    pub momentum_paths: usize,
    pub raw_operator_variables: usize,
    pub map_specs: Vec<SecondMomentum10001MapSpec>,
    pub recoupling_paths: Vec<SecondMomentum10001RecouplingPath>,
    pub embedded_sources: Vec<SecondMomentum10001EmbeddedSourceAudit>,
    pub source_map_gram: [[String; 2]; 2],
    pub source_map_gram_determinant: String,
    pub source_map_rank: usize,
    pub recoupling_path_rank: usize,
    pub combined_variable_rank: usize,
    pub recoupling_certificate_sha256: String,
    pub recoupling_extraction_matrix: [[i64; 2]; 2],
    pub recoupling_extraction_determinant: i64,
    pub source_mutation_detected: bool,
    pub recoupling_mutation_detected: bool,
    pub highest_target_basis_ordinal: usize,
    pub p2_d13_highest_weight_seeds_ready: bool,
    pub all_descendant_component_maps_complete: bool,
    pub remaining_10001_source_labels: Vec<String>,
    pub full_10001_source_tranche_complete: bool,
    pub full_physical_fag_established: bool,
    pub passed: bool,
    pub boundary: String,
}

#[derive(Debug, Clone)]
struct DenseState {
    weight: Weight,
    coefficients: Vec<i64>,
}

#[derive(Debug)]
struct CsrAction {
    target_weight: Weight,
    target_dimension: usize,
    source_offsets: Vec<u32>,
    destination_indices: Vec<u32>,
    signs: Vec<i8>,
}

#[derive(Debug)]
struct ExteriorReplayModel {
    spinors: [Weight; 32],
    left: FxHashMap<(u8, Weight), Vec<u16>>,
    right: FxHashMap<(u8, Weight), Vec<u16>>,
    spaces: BTreeMap<Weight, Arc<Vec<u32>>>,
    actions: BTreeMap<(Weight, usize), Arc<CsrAction>>,
}

#[derive(Debug, Clone)]
pub(crate) struct CoupledComponent {
    weight: Weight,
    pub(crate) masks: Arc<Vec<u32>>,
    pub(crate) coefficients: Vec<i128>,
}

#[derive(Debug, Clone)]
pub(crate) struct EmbeddedSourceMap {
    pub(crate) copy: usize,
    pub(crate) components: BTreeMap<usize, CoupledComponent>,
}

impl ExteriorReplayModel {
    fn new() -> Self {
        let spinors = spinor_weights();
        Self {
            spinors,
            left: half_groups(0, &spinors),
            right: half_groups(16, &spinors),
            spaces: BTreeMap::new(),
            actions: BTreeMap::new(),
        }
    }

    fn space(&mut self, weight: Weight) -> Arc<Vec<u32>> {
        Arc::clone(self.spaces.entry(weight).or_insert_with(|| {
            Arc::new(weight_basis(
                EXTERIOR_DEGREE,
                weight,
                &self.left,
                &self.right,
            ))
        }))
    }

    fn action(&mut self, source_weight: Weight, root: usize) -> Arc<CsrAction> {
        let key = (source_weight, root);
        if let Some(action) = self.actions.get(&key) {
            return Arc::clone(action);
        }
        let target_weight = subtract(source_weight, SIMPLE_ROOTS[root]);
        let source_masks = self.space(source_weight);
        let target_masks = self.space(target_weight);
        let target_index = target_masks
            .iter()
            .copied()
            .enumerate()
            .map(|(index, mask)| (mask, u32::try_from(index).unwrap()))
            .collect::<FxHashMap<_, _>>();
        let lowering_pairs = (0..32)
            .filter_map(|occupied| {
                lowered_spinor_index(occupied, root, &self.spinors)
                    .map(|replacement| (occupied, replacement))
            })
            .collect::<Vec<_>>();
        let mut source_offsets = Vec::with_capacity(source_masks.len() + 1);
        let mut destination_indices = Vec::new();
        let mut signs = Vec::new();
        source_offsets.push(0);
        for &source_mask in source_masks.iter() {
            for &(occupied, replacement) in &lowering_pairs {
                if source_mask & (1_u32 << occupied) == 0
                    || source_mask & (1_u32 << replacement) != 0
                {
                    continue;
                }
                let output_mask = (source_mask ^ (1_u32 << occupied)) | (1_u32 << replacement);
                destination_indices.push(target_index[&output_mask]);
                signs.push(
                    i8::try_from(exterior_replacement_sign(
                        source_mask,
                        occupied,
                        replacement,
                    ))
                    .unwrap(),
                );
            }
            source_offsets.push(u32::try_from(destination_indices.len()).unwrap());
        }
        let action = Arc::new(CsrAction {
            target_weight,
            target_dimension: target_masks.len(),
            source_offsets,
            destination_indices,
            signs,
        });
        self.actions.insert(key, Arc::clone(&action));
        action
    }

    fn lower(&mut self, source: &DenseState, root: usize) -> DenseState {
        let action = self.action(source.weight, root);
        let mut accumulator = vec![0_i64; action.target_dimension];
        for (source_index, coefficient) in source.coefficients.iter().copied().enumerate() {
            if coefficient == 0 {
                continue;
            }
            let first = usize::try_from(action.source_offsets[source_index]).unwrap();
            let last = usize::try_from(action.source_offsets[source_index + 1]).unwrap();
            for edge in first..last {
                let destination = usize::try_from(action.destination_indices[edge]).unwrap();
                accumulator[destination] = accumulator[destination]
                    .checked_add(
                        coefficient
                            .checked_mul(i64::from(action.signs[edge]))
                            .expect("level-12 lowering product exceeds i64"),
                    )
                    .expect("level-12 descendant coefficient exceeds i64");
            }
        }
        DenseState {
            weight: action.target_weight,
            coefficients: accumulator,
        }
    }

    fn fixture_state(&mut self, bytes: &[u8]) -> DenseState {
        let weight = dynkin_highest_weight(SOURCE_DYNKIN_LABEL);
        let masks = self.space(weight);
        assert_eq!(bytes.len(), 2 * masks.len());
        DenseState {
            weight,
            coefficients: bytes
                .chunks_exact(2)
                .map(|pair| i64::from(i16::from_le_bytes([pair[0], pair[1]])))
                .collect(),
        }
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
    let labels = label
        .bytes()
        .map(|byte| i8::try_from(byte - b'0').unwrap())
        .collect::<Vec<_>>();
    assert_eq!(labels.len(), 5);
    std::array::from_fn(|index| 2 * labels[index..4].iter().sum::<i8>() + labels[4])
}

fn half_groups(offset: usize, weights: &[Weight; 32]) -> FxHashMap<(u8, Weight), Vec<u16>> {
    let mut groups = FxHashMap::<(u8, Weight), Vec<u16>>::default();
    let mut degrees = vec![0_u8; usize::from(u16::MAX) + 1];
    let mut mask_weights = vec![[0_i8; 5]; usize::from(u16::MAX) + 1];
    for raw in 0_u32..=u32::from(u16::MAX) {
        let mask = raw as u16;
        if mask != 0 {
            let local = mask.trailing_zeros() as usize;
            let previous = mask & (mask - 1);
            degrees[usize::from(mask)] = degrees[usize::from(previous)] + 1;
            mask_weights[usize::from(mask)] = std::array::from_fn(|axis| {
                mask_weights[usize::from(previous)][axis] + weights[offset + local][axis]
            });
        }
        groups
            .entry((degrees[usize::from(mask)], mask_weights[usize::from(mask)]))
            .or_default()
            .push(mask);
    }
    groups
}

fn weight_basis(
    degree: u8,
    target: Weight,
    left: &FxHashMap<(u8, Weight), Vec<u16>>,
    right: &FxHashMap<(u8, Weight), Vec<u16>>,
) -> Vec<u32> {
    let mut basis = Vec::new();
    for left_degree in 0..=degree.min(16) {
        let right_degree = degree - left_degree;
        if right_degree > 16 {
            continue;
        }
        for ((candidate_degree, left_weight), left_masks) in left {
            if *candidate_degree != left_degree {
                continue;
            }
            let needed = subtract(target, *left_weight);
            let Some(right_masks) = right.get(&(right_degree, needed)) else {
                continue;
            };
            for &left_mask in left_masks {
                for &right_mask in right_masks {
                    basis.push(u32::from(left_mask) | (u32::from(right_mask) << 16));
                }
            }
        }
    }
    basis.sort_unstable();
    basis
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

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn abstract_coupling() -> crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate {
    assert_eq!(
        sha256(ABSTRACT_COUPLING_JSON.as_bytes()),
        ABSTRACT_COUPLING_SHA256
    );
    let certificate = serde_json::from_str::<
        crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    >(ABSTRACT_COUPLING_JSON)
    .expect("parse pinned abstract 10002 to 10001 coupling");
    assert_eq!(certificate.source_dynkin_label, SOURCE_DYNKIN_LABEL);
    assert_eq!(certificate.target_dynkin_label, INTERMEDIATE_DYNKIN_LABEL);
    assert_eq!(certificate.product_weight_domain_dimension, 262);
    assert_eq!(certificate.gram_matrix_rank, 261);
    assert_eq!(certificate.kernel_dimension, 1);
    assert_eq!(
        certificate.exact_raising_residual_terms_by_simple_root,
        [0; 5]
    );
    assert!(certificate.passed);
    certificate
}

fn prefix_children(
    certificate: &crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
) -> BTreeMap<Vec<u8>, BTreeSet<u8>> {
    let mut children = BTreeMap::<Vec<u8>, BTreeSet<u8>>::new();
    for entry in &certificate.domain_basis {
        for depth in 0..entry.pbw_word_simple_roots.len() {
            children
                .entry(entry.pbw_word_simple_roots[..depth].to_vec())
                .or_default()
                .insert(entry.pbw_word_simple_roots[depth]);
        }
    }
    children
}

fn terminal_coefficients(
    certificate: &crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
) -> BTreeMap<Vec<u8>, Vec<(usize, Weight, i64)>> {
    let mut terminals = BTreeMap::<Vec<u8>, Vec<(usize, Weight, i64)>>::new();
    for (entry, &coefficient) in certificate
        .domain_basis
        .iter()
        .zip(&certificate.primitive_domain_coefficients)
    {
        terminals
            .entry(entry.pbw_word_simple_roots.clone())
            .or_default()
            .push((entry.free_spinor_index, entry.source_weight, coefficient));
    }
    terminals
}

fn add_terminal(
    model: &mut ExteriorReplayModel,
    coupled: &mut BTreeMap<usize, CoupledComponent>,
    state: &DenseState,
    spinor: usize,
    expected_weight: Weight,
    scale: i64,
) {
    assert_eq!(state.weight, expected_weight);
    let component = coupled.entry(spinor).or_insert_with(|| CoupledComponent {
        weight: state.weight,
        masks: model.space(state.weight),
        coefficients: vec![0; state.coefficients.len()],
    });
    assert_eq!(component.weight, state.weight);
    for (target, &source) in component.coefficients.iter_mut().zip(&state.coefficients) {
        *target = target
            .checked_add(i128::from(scale) * i128::from(source))
            .expect("embedded highest-weight coefficient exceeds i128");
    }
}

fn materialize_source(
    model: &mut ExteriorReplayModel,
    certificate: &crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    copy: usize,
    bytes: &[u8],
) -> EmbeddedSourceMap {
    let children = prefix_children(certificate);
    let terminals = terminal_coefficients(certificate);
    let maximum_depth = certificate
        .domain_basis
        .iter()
        .map(|entry| entry.pbw_word_simple_roots.len())
        .max()
        .unwrap_or(0);
    let mut current = BTreeMap::from([(Vec::<u8>::new(), model.fixture_state(bytes))]);
    let mut coupled = BTreeMap::<usize, CoupledComponent>::new();
    for depth in 0..=maximum_depth {
        let mut next = BTreeMap::<Vec<u8>, DenseState>::new();
        for (word, state) in current {
            if let Some(entries) = terminals.get(&word) {
                for &(spinor, expected_weight, scale) in entries {
                    add_terminal(model, &mut coupled, &state, spinor, expected_weight, scale);
                }
            }
            if depth == maximum_depth {
                continue;
            }
            if let Some(roots) = children.get(&word) {
                for &one_based_root in roots {
                    let mut child = word.clone();
                    child.push(one_based_root);
                    next.insert(child, model.lower(&state, usize::from(one_based_root - 1)));
                }
            }
        }
        current = next;
    }
    assert_eq!(coupled.len(), 32);
    EmbeddedSourceMap {
        copy,
        components: coupled,
    }
}

fn for_each_term(map: &EmbeddedSourceMap, mut visitor: impl FnMut(SecondMomentum10001SourceTerm)) {
    for (&spinor, component) in &map.components {
        for (&mask, &coefficient) in component.masks.iter().zip(&component.coefficients) {
            if coefficient != 0 {
                visitor(SecondMomentum10001SourceTerm {
                    source_copy: map.copy,
                    intermediate_spinor_weight_index: spinor,
                    exterior_mask: mask,
                    coefficient,
                });
            }
        }
    }
}

fn any_term(
    map: &EmbeddedSourceMap,
    mut predicate: impl FnMut(SecondMomentum10001SourceTerm) -> bool,
) -> bool {
    for (&spinor, component) in &map.components {
        for (&mask, &coefficient) in component.masks.iter().zip(&component.coefficients) {
            if coefficient != 0
                && predicate(SecondMomentum10001SourceTerm {
                    source_copy: map.copy,
                    intermediate_spinor_weight_index: spinor,
                    exterior_mask: mask,
                    coefficient,
                })
            {
                return true;
            }
        }
    }
    false
}

fn raising_residuals(map: &EmbeddedSourceMap) -> [usize; 5] {
    let spinors = spinor_weights();
    let raised: [[Option<usize>; 32]; 5] = std::array::from_fn(|root| {
        std::array::from_fn(|index| raised_spinor_index(index, root, &spinors))
    });
    (0..5)
        .into_par_iter()
        .map(|root| {
            let mut residual = FxHashMap::<u64, i128>::default();
            for_each_term(map, |term| {
                let mut occupied_mask = term.exterior_mask;
                while occupied_mask != 0 {
                    let occupied = occupied_mask.trailing_zeros() as usize;
                    occupied_mask &= occupied_mask - 1;
                    let Some(replacement) = raised[root][occupied] else {
                        continue;
                    };
                    if term.exterior_mask & (1_u32 << replacement) != 0 {
                        continue;
                    }
                    let output_mask =
                        (term.exterior_mask ^ (1_u32 << occupied)) | (1_u32 << replacement);
                    let key = ((term.intermediate_spinor_weight_index as u64) << 32)
                        | u64::from(output_mask);
                    add_residual(
                        &mut residual,
                        key,
                        term.coefficient
                            .checked_mul(i128::from(exterior_replacement_sign(
                                term.exterior_mask,
                                occupied,
                                replacement,
                            )))
                            .expect("raising residual product exceeds i128"),
                    );
                }
                if let Some(next_spinor) = raised[root][term.intermediate_spinor_weight_index] {
                    let key = ((next_spinor as u64) << 32) | u64::from(term.exterior_mask);
                    add_residual(&mut residual, key, term.coefficient);
                }
            });
            residual.len()
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("five simple-root residual counts")
}

fn add_residual(residual: &mut FxHashMap<u64, i128>, key: u64, delta: i128) {
    if delta == 0 {
        return;
    }
    match residual.entry(key) {
        Entry::Occupied(mut entry) => {
            let sum = entry
                .get()
                .checked_add(delta)
                .expect("raising residual exceeds i128");
            if sum == 0 {
                entry.remove();
            } else {
                *entry.get_mut() = sum;
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(delta);
        }
    }
}

fn single_term_raising_entries(term: SecondMomentum10001SourceTerm) -> usize {
    let spinors = spinor_weights();
    (0..5)
        .map(|root| {
            let exterior = (0..32)
                .filter(|occupied| term.exterior_mask & (1_u32 << occupied) != 0)
                .filter_map(|occupied| {
                    raised_spinor_index(occupied, root, &spinors)
                        .filter(|replacement| term.exterior_mask & (1_u32 << replacement) == 0)
                })
                .count();
            exterior
                + usize::from(
                    raised_spinor_index(term.intermediate_spinor_weight_index, root, &spinors)
                        .is_some(),
                )
        })
        .sum()
}

fn coupled_map_sha256(map: &EmbeddedSourceMap) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-11d-second-momentum-10001-source-map-v1\0");
    for_each_term(map, |term| {
        hasher.update((term.source_copy as u64).to_le_bytes());
        hasher.update((term.intermediate_spinor_weight_index as u64).to_le_bytes());
        hasher.update(term.exterior_mask.to_le_bytes());
        hasher.update(term.coefficient.to_le_bytes());
    });
    format!("{:x}", hasher.finalize())
}

fn map_audit(
    map: &EmbeddedSourceMap,
    fixture_sha256: &str,
    abstract_domain_dimension: usize,
) -> SecondMomentum10001EmbeddedSourceAudit {
    let mut nonzero = 0;
    let mut maximum = 0_i128;
    for_each_term(map, |term| {
        nonzero += 1;
        maximum = maximum.max(term.coefficient.abs());
    });
    let residuals = raising_residuals(map);
    SecondMomentum10001EmbeddedSourceAudit {
        source_copy: map.copy,
        fixture_sha256: fixture_sha256.to_string(),
        abstract_coupling_sha256: ABSTRACT_COUPLING_SHA256.to_string(),
        abstract_domain_dimension,
        coupled_nonzero_terms: nonzero,
        coupled_map_sha256: coupled_map_sha256(map),
        exact_raising_residual_terms_by_simple_root: residuals,
        maximum_absolute_coefficient: maximum,
        passed: residuals == [0; 5] && nonzero > 0,
    }
}

struct CheckedBigAccumulator {
    total: BigInt,
    pending: i128,
}

impl CheckedBigAccumulator {
    fn new() -> Self {
        Self {
            total: BigInt::zero(),
            pending: 0,
        }
    }

    fn add_product(&mut self, left: i128, right: i128) {
        let Some(product) = left.checked_mul(right) else {
            self.flush();
            self.total += BigInt::from(left) * BigInt::from(right);
            return;
        };
        if let Some(sum) = self.pending.checked_add(product) {
            self.pending = sum;
        } else {
            self.flush();
            self.pending = product;
        }
    }

    fn flush(&mut self) {
        if self.pending != 0 {
            self.total += self.pending;
            self.pending = 0;
        }
    }

    fn finish(mut self) -> BigInt {
        self.flush();
        self.total
    }
}

fn map_gram(first: &EmbeddedSourceMap, second: &EmbeddedSourceMap) -> (BigInt, BigInt, BigInt) {
    let mut gram00 = CheckedBigAccumulator::new();
    let mut gram01 = CheckedBigAccumulator::new();
    let mut gram11 = CheckedBigAccumulator::new();
    for (spinor, first_component) in &first.components {
        let second_component = second
            .components
            .get(spinor)
            .expect("source maps have identical spinor components");
        assert_eq!(first_component.weight, second_component.weight);
        assert_eq!(first_component.masks, second_component.masks);
        for (&left, &right) in first_component
            .coefficients
            .iter()
            .zip(&second_component.coefficients)
        {
            gram00.add_product(left, left);
            gram01.add_product(left, right);
            gram11.add_product(right, right);
        }
    }
    (gram00.finish(), gram01.finish(), gram11.finish())
}

fn map_specs() -> Vec<SecondMomentum10001MapSpec> {
    let mut specs = Vec::new();
    for source_copy in 1..=2 {
        for path in [
            SecondMomentum10001Path::Trace,
            SecondMomentum10001Path::SymmetricTraceless,
        ] {
            specs.push(SecondMomentum10001MapSpec {
                variable_ordinal: specs.len(),
                source_dynkin_label: SOURCE_DYNKIN_LABEL.to_string(),
                source_copy,
                intermediate_dynkin_label: INTERMEDIATE_DYNKIN_LABEL.to_string(),
                momentum_path: path,
                momentum_degree: 2,
                exterior_derivative_order: 12,
                gauge_composition_wedge_order: 13,
                highest_weight_checkpoint_available: true,
                all_descendant_component_maps_complete: false,
            });
        }
    }
    specs
}

fn recoupling_paths(
    recoupling: &crate::eleven_dimensional_second_momentum_recoupling::SecondMomentumRecouplingReport,
) -> Vec<SecondMomentum10001RecouplingPath> {
    vec![
        SecondMomentum10001RecouplingPath {
            path: SecondMomentum10001Path::Trace,
            image_rank: recoupling.trace_path_image_rank,
            trace_delta_coefficient: 1,
            delta_embedding_coefficient: 0,
            gamma_embedding_coefficient: 0,
            momentum_trace_subtraction_coefficient: 0,
            target_gamma_residual_entries: recoupling.trace_path_target_gamma_residual_entries,
            momentum_trace_residual_entries: recoupling.trace_extraction_residual_entries,
        },
        SecondMomentum10001RecouplingPath {
            path: SecondMomentum10001Path::SymmetricTraceless,
            image_rank: recoupling.symmetric_traceless_path_image_rank,
            trace_delta_coefficient: 0,
            delta_embedding_coefficient: recoupling.symmetric_traceless_embedding_coefficients[0],
            gamma_embedding_coefficient: recoupling.symmetric_traceless_embedding_coefficients[1],
            momentum_trace_subtraction_coefficient: recoupling
                .symmetric_traceless_embedding_coefficients[2],
            target_gamma_residual_entries: recoupling
                .symmetric_traceless_target_gamma_residual_entries,
            momentum_trace_residual_entries: recoupling
                .symmetric_traceless_momentum_trace_residual_entries,
        },
    ]
}

/// Visit the exact source side of one highest-weight checkpoint.
///
/// The terms are in canonical `(spinor-weight index, exterior mask)` order.
/// The momentum path is selected separately by `second_momentum_10001_map_specs`.
pub fn visit_second_momentum_10001_highest_weight_source_terms(
    source_copy: usize,
    mut visitor: impl FnMut(SecondMomentum10001SourceTerm),
) {
    assert!((1..=2).contains(&source_copy));
    let certificate = abstract_coupling();
    let mut model = ExteriorReplayModel::new();
    let map = materialize_source(
        &mut model,
        &certificate,
        source_copy,
        SOURCE_KERNELS[source_copy - 1],
    );
    for_each_term(&map, &mut visitor);
}

pub fn second_momentum_10001_map_specs() -> Vec<SecondMomentum10001MapSpec> {
    map_specs()
}

fn verify_second_momentum_10001_maps_internal(
    retain_source_maps: bool,
) -> (SecondMomentum10001MapReport, Option<[EmbeddedSourceMap; 2]>) {
    for (bytes, expected) in SOURCE_KERNELS.into_iter().zip(SOURCE_KERNEL_SHA256) {
        assert_eq!(sha256(bytes), expected);
    }
    let certificate = abstract_coupling();
    let (first, second) = rayon::join(
        || {
            let mut model = ExteriorReplayModel::new();
            materialize_source(&mut model, &certificate, 1, SOURCE_KERNELS[0])
        },
        || {
            let mut model = ExteriorReplayModel::new();
            materialize_source(&mut model, &certificate, 2, SOURCE_KERNELS[1])
        },
    );
    let ((first_audit, second_audit), ((gram00, gram01, gram11), recoupling)) = rayon::join(
        || {
            rayon::join(
                || {
                    map_audit(
                        &first,
                        SOURCE_KERNEL_SHA256[0],
                        certificate.product_weight_domain_dimension,
                    )
                },
                || {
                    map_audit(
                        &second,
                        SOURCE_KERNEL_SHA256[1],
                        certificate.product_weight_domain_dimension,
                    )
                },
            )
        },
        || {
            rayon::join(
                || map_gram(&first, &second),
                crate::eleven_dimensional_second_momentum_recoupling::verify,
            )
        },
    );
    let embedded_sources = vec![first_audit, second_audit];

    let determinant = &gram00 * &gram11 - &gram01 * &gram01;
    let source_map_rank = if determinant != BigInt::zero() { 2 } else { 1 };

    assert!(recoupling.passed);
    let paths = recoupling_paths(&recoupling);
    let recoupling_path_rank = recoupling.multiplicity_space_rank;
    let combined_variable_rank = source_map_rank * recoupling_path_rank;
    let source_mutation_detected = any_term(&first, |term| single_term_raising_entries(term) != 0);
    let [delta, gamma, trace] = recoupling.symmetric_traceless_embedding_coefficients;
    let recoupling_mutation_detected =
        delta + 11 * gamma == 0 && 2 * delta + 11 * trace == 0 && 2 * delta + 11 * (trace + 1) != 0;
    let highest_target_basis_ordinal =
        crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
            .into_iter()
            .find(|state| {
                state.doubled_weight == TARGET_WEIGHT && state.pbw_word_simple_roots.is_empty()
            })
            .expect("canonical 10001 highest target state is absent")
            .ordinal;
    let specs = map_specs();
    let passed = embedded_sources.iter().all(|audit| audit.passed)
        && source_map_rank == 2
        && recoupling_path_rank == 2
        && combined_variable_rank == 4
        && recoupling.extraction_determinant != 0
        && paths.iter().all(|path| {
            path.image_rank == 320
                && path.target_gamma_residual_entries == 0
                && path.momentum_trace_residual_entries == 0
        })
        && source_mutation_detected
        && recoupling_mutation_detected
        && specs.len() == 4;

    let report = SecondMomentum10001MapReport {
        schema_version: "adynkra-11d-second-momentum-10001-highest-weight-maps-v1".to_string(),
        role: "exact highest-weight map checkpoints for two level-12 10002 source copies times the trace and symmetric-traceless 10001 momentum paths".to_string(),
        source_dynkin_label: SOURCE_DYNKIN_LABEL.to_string(),
        intermediate_dynkin_label: INTERMEDIATE_DYNKIN_LABEL.to_string(),
        abstract_coupling_transfer_justification: "The pinned coupling is an abstract Chevalley-PBW map between the 10002 and 10001 irreps, so it is independent of the exterior degree used to realize 10002. Transfer to each level-12 embedding is accepted only after every ambient characteristic-zero raising coordinate vanishes.".to_string(),
        source_kernel_copies: 2,
        momentum_paths: 2,
        raw_operator_variables: 4,
        map_specs: specs,
        recoupling_paths: paths,
        embedded_sources,
        source_map_gram: [
            [gram00.to_string(), gram01.to_string()],
            [gram01.to_string(), gram11.to_string()],
        ],
        source_map_gram_determinant: determinant.to_string(),
        source_map_rank,
        recoupling_path_rank,
        combined_variable_rank,
        recoupling_certificate_sha256: recoupling.certificate_sha256,
        recoupling_extraction_matrix: recoupling.extraction_matrix,
        recoupling_extraction_determinant: recoupling.extraction_determinant,
        source_mutation_detected,
        recoupling_mutation_detected,
        highest_target_basis_ordinal,
        p2_d13_highest_weight_seeds_ready: passed,
        all_descendant_component_maps_complete: false,
        remaining_10001_source_labels: vec!["00100".to_string(), "00010".to_string()],
        full_10001_source_tranche_complete: false,
        full_physical_fag_established: false,
        passed,
        boundary: "This certifies four independent highest-weight seeds from the two exact level-12 10002 embeddings and the two exact 10001 momentum paths. It does not yet descend those seeds into complete component Clebsch-Gordan maps. The 00100 and 00010 sources remain, and full physical F A G_p is false.".to_string(),
    };
    let source_maps = retain_source_maps.then_some([first, second]);
    (report, source_maps)
}

pub fn verify_second_momentum_10001_maps() -> SecondMomentum10001MapReport {
    verify_second_momentum_10001_maps_internal(false).0
}

pub(crate) fn verify_second_momentum_10001_maps_with_embedded_sources()
-> (SecondMomentum10001MapReport, [EmbeddedSourceMap; 2]) {
    let (report, source_maps) = verify_second_momentum_10001_maps_internal(true);
    (
        report,
        source_maps.expect("retained 10001 source maps were not returned"),
    )
}

pub fn write_second_momentum_10001_map_artifact(
    path: &Path,
) -> io::Result<SecondMomentum10001MapReport> {
    let report = verify_second_momentum_10001_maps();
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "second-momentum 10001 highest-weight map certificate did not pass",
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&report).map_err(io::Error::other)?,
    )?;
    fs::rename(temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sources_times_two_paths_give_four_exact_map_seeds() {
        let report = verify_second_momentum_10001_maps();
        assert!(report.passed);
        assert_eq!(report.source_map_rank, 2);
        assert_eq!(report.recoupling_path_rank, 2);
        assert_eq!(report.combined_variable_rank, 4);
        assert_eq!(report.raw_operator_variables, 4);
        assert_ne!(report.source_map_gram_determinant, "0");
        assert!(
            report
                .embedded_sources
                .iter()
                .all(|audit| audit.exact_raising_residual_terms_by_simple_root == [0; 5])
        );
        assert!(!report.all_descendant_component_maps_complete);
        assert!(!report.full_10001_source_tranche_complete);
        assert!(!report.full_physical_fag_established);
    }

    #[test]
    fn source_and_stt_mutations_are_detected() {
        let report = verify_second_momentum_10001_maps();
        assert!(report.source_mutation_detected);
        assert!(report.recoupling_mutation_detected);
    }
}
