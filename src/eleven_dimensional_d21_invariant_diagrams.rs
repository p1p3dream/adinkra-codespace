//! Fierz-complete invariant-diagram grammar for the `(d_D,d_p)=(2,1)` source.
//!
//! Four spinor indices are recoupled in the `(outer,outer)|(H,output)`
//! channel. Completeness of the 11D Clifford basis reduces the first factor
//! to the antisymmetric `C Gamma_[r]` degrees and the second to
//! `Gamma_[s]`, with `0 <= r,s <= 5`. Vector indices are then exhausted by
//! external attachments, cross-gamma contractions, or metric pairings.

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::eleven_dimensional_dg4_casimir_projectors::{
    dg4_lorentz_generator_action_integer, dg4_projector_numerator_oracle, project_dg4_target,
};
use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
use crate::eleven_dimensional_majorana::{real_charge_conjugation, real_gamma_matrices};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalIndex {
    Momentum,
    HVector,
    Output0,
    Output1,
    Output2,
    Output3,
}

const EXTERNALS: [ExternalIndex; 6] = [
    ExternalIndex::Momentum,
    ExternalIndex::HVector,
    ExternalIndex::Output0,
    ExternalIndex::Output1,
    ExternalIndex::Output2,
    ExternalIndex::Output3,
];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct D21InvariantDiagram {
    pub outer_bilinear_degree: usize,
    pub inner_to_output_degree: usize,
    pub cross_gamma_contractions: usize,
    pub outer_gamma_external_indices: Vec<ExternalIndex>,
    pub inner_gamma_external_indices: Vec<ExternalIndex>,
    pub metric_pairs: Vec<[ExternalIndex; 2]>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedD21Diagram {
    pub outer_degree: u8,
    pub inner_degree: u8,
    pub cross: u8,
    pub outer_count: u8,
    pub inner_count: u8,
    pub metric_count: u8,
    pub reserved0: u8,
    pub reserved1: u8,
    pub outer_external: [u8; 6],
    pub inner_external: [u8; 6],
    pub metric_pairs: [u8; 12],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct D21CoefficientQuery {
    pub diagram: u16,
    pub outer_left: u8,
    pub outer_right: u8,
    pub momentum: u8,
    pub h_vector: u8,
    pub output_axes: [u8; 4],
    pub input_spinor: u8,
    pub output_spinor: u8,
    pub h_coefficient: i16,
    pub reserved: u16,
}

/// Canonical compact raw-diagram COO emitted before target projection.
/// Ordering is `(source_coordinate, target_coordinate, diagram_ordinal)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct D21CompactRawCooEntry {
    pub source_coordinate: u32,
    pub target_coordinate: u16,
    pub diagram_ordinal: u16,
    pub coefficient: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D21SectorPivotReplayRequestV2 {
    pub source_coordinate: u32,
    pub target_coordinate: u16,
    pub diagram_ordinal: u16,
    pub target_sector: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D21SectorPivotReplayResultV2 {
    pub request: D21SectorPivotReplayRequestV2,
    pub raw_coefficient: i64,
    pub ordered_shift_eigenvalues: [i64; 4],
    pub pass_nonzero_counts: [usize; 4],
    pub projector_denominator: i64,
    pub projected_numerator: i128,
    pub passed_nonzero: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D21InvariantDiagramReport {
    pub schema_version: &'static str,
    pub fierz_pairing: &'static str,
    pub clifford_degrees_checked: usize,
    pub antisymmetric_outer_bilinear_degrees: Vec<usize>,
    pub symmetric_outer_bilinear_degrees: Vec<usize>,
    pub mixed_symmetry_residual_masks: usize,
    pub raw_diagrams: usize,
    pub diagrams_by_outer_degree: BTreeMap<usize, usize>,
    pub diagrams_by_target_gamma_degree: BTreeMap<usize, usize>,
    pub duplicate_signatures: usize,
    pub packed_diagrams_sha256: String,
    pub gamma_mask_table_sha256: String,
    pub charge_gamma_mask_table_sha256: String,
    pub volume_element_scalar: i16,
    pub volume_element_residual_entries: usize,
    pub hodge_duality_masks_checked: usize,
    pub hodge_duality_residual_masks: usize,
    pub epsilon_hodge_redundancy_proved: bool,
    pub outer_spinor_antisymmetrization_required: bool,
    pub target_projection_required: bool,
    pub passed_grammar: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D21PivotReplayRequest {
    pub diagram_ordinal: usize,
    pub outer_pair_ordinal: usize,
    pub momentum_axis: usize,
    pub h_hat_ordinal: usize,
    pub target_coordinate: usize,
    pub target_sector: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D21PivotReplayResult {
    pub request: D21PivotReplayRequest,
    pub outer_pair: [usize; 2],
    pub raw_coefficient: String,
    pub projected_coefficient: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D21OuterZeroCanaryReport {
    pub schema_version: &'static str,
    pub diagrams_evaluated: usize,
    pub outer_pair: [usize; 2],
    pub momentum_axis: usize,
    pub h_hat_ordinal: usize,
    pub nonzero_raw_diagrams: usize,
    pub exact_sector_ranks: BTreeMap<&'static str, usize>,
    pub expected_full_sector_ranks: BTreeMap<&'static str, usize>,
    pub is_lower_bound_only: bool,
    pub passed_canary: bool,
    pub boundary: &'static str,
}

type Matrix = Vec<Vec<i16>>;
type SparseIntegerVector = BTreeMap<usize, i64>;

fn identity() -> Matrix {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for index in 0..SPINOR_DIMENSION {
        output[index][index] = 1;
    }
    output
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            if left[row][pivot] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += left[row][pivot] * right[pivot][column];
            }
        }
    }
    output
}

fn gamma_product(mask: u16, gammas: &[Matrix]) -> Matrix {
    let mut output = identity();
    for axis in 0..VECTOR_DIMENSION {
        if mask & (1_u16 << axis) != 0 {
            output = multiply(&output, &gammas[axis]);
        }
    }
    output
}

fn gamma_mask_matrices() -> &'static Vec<Matrix> {
    static MATRICES: OnceLock<Vec<Matrix>> = OnceLock::new();
    MATRICES.get_or_init(|| {
        let gammas = real_gamma_matrices()
            .into_iter()
            .map(|matrix| {
                matrix
                    .into_iter()
                    .map(|row| row.into_iter().map(i16::from).collect())
                    .collect()
            })
            .collect::<Vec<Matrix>>();
        (0_u16..(1_u16 << VECTOR_DIMENSION))
            .map(|mask| gamma_product(mask, &gammas))
            .collect()
    })
}

fn charge_gamma_mask_matrices() -> &'static Vec<Matrix> {
    static MATRICES: OnceLock<Vec<Matrix>> = OnceLock::new();
    MATRICES.get_or_init(|| {
        let charge = real_charge_conjugation()
            .into_iter()
            .map(|row| row.into_iter().map(i16::from).collect())
            .collect::<Matrix>();
        gamma_mask_matrices()
            .iter()
            .map(|gamma| multiply(&charge, gamma))
            .collect()
    })
}

/// Certify, in the repository's real Majorana convention, that the odd
/// dimensional Clifford volume is scalar and that every rank-six-through-
/// eleven Clifford form is Hodge-dual to a rank-zero-through-five form.
/// This is the executable reason an explicit epsilon vertex adds no new
/// spinor intertwiner to the declared diagram grammar: one epsilon is moved
/// onto a Clifford form and removed by this identity, while two epsilons
/// reduce to a determinant of metrics.
fn epsilon_hodge_certificate() -> (i16, usize, usize, usize) {
    let matrices = gamma_mask_matrices();
    let volume = &matrices[(1_usize << VECTOR_DIMENSION) - 1];
    let scalar = volume[0][0];
    let mut volume_residual_entries = 0;
    for row in 0..SPINOR_DIMENSION {
        for column in 0..SPINOR_DIMENSION {
            let expected = if row == column { scalar } else { 0 };
            volume_residual_entries += usize::from(volume[row][column] != expected);
        }
    }

    let full_mask = (1_u16 << VECTOR_DIMENSION) - 1;
    let mut checked = 0;
    let mut residual_masks = 0;
    for mask in 0_u16..=full_mask {
        if mask.count_ones() <= 5 {
            continue;
        }
        checked += 1;
        let left = &matrices[usize::from(mask)];
        let right = &matrices[usize::from(full_mask ^ mask)];
        let sign = (0..SPINOR_DIMENSION)
            .find_map(|row| {
                (0..SPINOR_DIMENSION).find_map(|column| {
                    let l = left[row][column];
                    let r = right[row][column];
                    (r != 0).then_some(if l == r {
                        1_i16
                    } else if l == -r {
                        -1_i16
                    } else {
                        0_i16
                    })
                })
            })
            .unwrap_or(0);
        let residual = sign == 0
            || (0..SPINOR_DIMENSION).any(|row| {
                (0..SPINOR_DIMENSION).any(|column| left[row][column] != sign * right[row][column])
            });
        residual_masks += usize::from(residual);
    }
    (scalar, volume_residual_entries, checked, residual_masks)
}

fn ordered_gamma_product(axes: &[usize], gammas: &[Matrix]) -> Matrix {
    let mut output = identity();
    for &axis in axes {
        output = multiply(&output, &gammas[axis]);
    }
    output
}

fn lorentz_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn form_masks(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn add_integer(output: &mut SparseIntegerVector, row: usize, value: i64) {
    if value == 0 {
        return;
    }
    *output.entry(row).or_default() += value;
    if output[&row] == 0 {
        output.remove(&row);
    }
}

fn transpose_sign(matrix: &Matrix) -> Option<i8> {
    let symmetric = (0..SPINOR_DIMENSION)
        .all(|row| (0..SPINOR_DIMENSION).all(|column| matrix[row][column] == matrix[column][row]));
    let antisymmetric = (0..SPINOR_DIMENSION)
        .all(|row| (0..SPINOR_DIMENSION).all(|column| matrix[row][column] == -matrix[column][row]));
    match (symmetric, antisymmetric) {
        (true, false) => Some(1),
        (false, true) => Some(-1),
        _ => None,
    }
}

fn outer_bilinear_symmetries() -> (Vec<usize>, Vec<usize>, usize) {
    let gammas = real_gamma_matrices()
        .into_iter()
        .map(|matrix| {
            matrix
                .into_iter()
                .map(|row| row.into_iter().map(i16::from).collect())
                .collect()
        })
        .collect::<Vec<Matrix>>();
    let charge = real_charge_conjugation()
        .into_iter()
        .map(|row| row.into_iter().map(i16::from).collect())
        .collect::<Matrix>();
    let mut antisymmetric = Vec::new();
    let mut symmetric = Vec::new();
    let mut residuals = 0;
    for degree in 0..=5 {
        let mut observed = None;
        for mask in 0_u16..(1_u16 << VECTOR_DIMENSION) {
            if mask.count_ones() as usize != degree {
                continue;
            }
            let sign = transpose_sign(&multiply(&charge, &gamma_product(mask, &gammas)));
            if sign.is_none() || observed.is_some_and(|prior| Some(prior) != sign) {
                residuals += 1;
            }
            observed = observed.or(sign);
        }
        match observed {
            Some(-1) => antisymmetric.push(degree),
            Some(1) => symmetric.push(degree),
            _ => {}
        }
    }
    (antisymmetric, symmetric, residuals)
}

fn valid_metric_pair(left: ExternalIndex, right: ExternalIndex) -> bool {
    !matches!(
        (left, right),
        (
            ExternalIndex::Output0
                | ExternalIndex::Output1
                | ExternalIndex::Output2
                | ExternalIndex::Output3,
            ExternalIndex::Output0
                | ExternalIndex::Output1
                | ExternalIndex::Output2
                | ExternalIndex::Output3
        )
    )
}

fn metric_matchings(indices: &[ExternalIndex]) -> Vec<Vec<[ExternalIndex; 2]>> {
    if indices.is_empty() {
        return vec![Vec::new()];
    }
    let first = indices[0];
    let mut output = Vec::new();
    for next in 1..indices.len() {
        if !valid_metric_pair(first, indices[next]) {
            continue;
        }
        let mut remaining = indices[1..].to_vec();
        let second = remaining.remove(next - 1);
        for mut matching in metric_matchings(&remaining) {
            let mut pair = [first, second];
            pair.sort();
            matching.push(pair);
            matching.sort();
            output.push(matching);
        }
    }
    output
}

fn choose(indices: &[ExternalIndex], count: usize) -> Vec<Vec<ExternalIndex>> {
    fn visit(
        indices: &[ExternalIndex],
        count: usize,
        next: usize,
        current: &mut Vec<ExternalIndex>,
        output: &mut Vec<Vec<ExternalIndex>>,
    ) {
        if current.len() == count {
            output.push(current.clone());
            return;
        }
        for index in next..indices.len() {
            current.push(indices[index]);
            visit(indices, count, index + 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    visit(indices, count, 0, &mut Vec::new(), &mut output);
    output
}

fn external_code(index: ExternalIndex) -> u8 {
    index as u8
}

pub fn packed_diagrams() -> Vec<PackedD21Diagram> {
    enumerate_diagrams()
        .into_iter()
        .map(|diagram| {
            let mut packed = PackedD21Diagram {
                outer_degree: u8::try_from(diagram.outer_bilinear_degree).unwrap(),
                inner_degree: u8::try_from(diagram.inner_to_output_degree).unwrap(),
                cross: u8::try_from(diagram.cross_gamma_contractions).unwrap(),
                outer_count: u8::try_from(diagram.outer_gamma_external_indices.len()).unwrap(),
                inner_count: u8::try_from(diagram.inner_gamma_external_indices.len()).unwrap(),
                metric_count: u8::try_from(diagram.metric_pairs.len()).unwrap(),
                ..PackedD21Diagram::default()
            };
            for (slot, &index) in diagram.outer_gamma_external_indices.iter().enumerate() {
                packed.outer_external[slot] = external_code(index);
            }
            for (slot, &index) in diagram.inner_gamma_external_indices.iter().enumerate() {
                packed.inner_external[slot] = external_code(index);
            }
            for (slot, pair) in diagram.metric_pairs.iter().enumerate() {
                packed.metric_pairs[2 * slot] = external_code(pair[0]);
                packed.metric_pairs[2 * slot + 1] = external_code(pair[1]);
            }
            packed
        })
        .collect()
}

pub fn flattened_gamma_mask_tables() -> (Vec<i16>, Vec<i16>) {
    let flatten = |matrices: &[Matrix]| {
        matrices
            .iter()
            .flat_map(|matrix| matrix.iter().flat_map(|row| row.iter().copied()))
            .collect::<Vec<_>>()
    };
    (
        flatten(gamma_mask_matrices()),
        flatten(charge_gamma_mask_matrices()),
    )
}

fn packed_diagrams_sha256(diagrams: &[PackedD21Diagram]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-d21-packed-diagrams-v1\0");
    for diagram in diagrams {
        hash.update([
            diagram.outer_degree,
            diagram.inner_degree,
            diagram.cross,
            diagram.outer_count,
            diagram.inner_count,
            diagram.metric_count,
            diagram.reserved0,
            diagram.reserved1,
        ]);
        hash.update(diagram.outer_external);
        hash.update(diagram.inner_external);
        hash.update(diagram.metric_pairs);
    }
    format!("{:x}", hash.finalize())
}

fn i16_table_sha256(domain: &[u8], table: &[i16]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    for value in table {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub fn packed_contract_sha256() -> (String, String, String) {
    let diagrams = packed_diagrams();
    let (gamma, charge_gamma) = flattened_gamma_mask_tables();
    (
        packed_diagrams_sha256(&diagrams),
        i16_table_sha256(b"adynkra-11d-d21-gamma-mask-table-v1\0", &gamma),
        i16_table_sha256(
            b"adynkra-11d-d21-charge-gamma-mask-table-v1\0",
            &charge_gamma,
        ),
    )
}

pub fn enumerate_diagrams() -> Vec<D21InvariantDiagram> {
    let (outer_degrees, _, residuals) = outer_bilinear_symmetries();
    assert_eq!(residuals, 0);
    let mut output = Vec::new();
    for outer_degree in outer_degrees {
        for inner_degree in 0..=5 {
            for cross in 0..=outer_degree.min(inner_degree) {
                let outer_external_count = outer_degree - cross;
                let inner_external_count = inner_degree - cross;
                if outer_external_count + inner_external_count > EXTERNALS.len() {
                    continue;
                }
                for outer_external in choose(&EXTERNALS, outer_external_count) {
                    let remaining = EXTERNALS
                        .iter()
                        .copied()
                        .filter(|index| !outer_external.contains(index))
                        .collect::<Vec<_>>();
                    for inner_external in choose(&remaining, inner_external_count) {
                        let unmatched = remaining
                            .iter()
                            .copied()
                            .filter(|index| !inner_external.contains(index))
                            .collect::<Vec<_>>();
                        if unmatched.len() % 2 != 0 {
                            continue;
                        }
                        for metric_pairs in metric_matchings(&unmatched) {
                            output.push(D21InvariantDiagram {
                                outer_bilinear_degree: outer_degree,
                                inner_to_output_degree: inner_degree,
                                cross_gamma_contractions: cross,
                                outer_gamma_external_indices: outer_external.clone(),
                                inner_gamma_external_indices: inner_external.clone(),
                                metric_pairs,
                            });
                        }
                    }
                }
            }
        }
    }
    output.sort();
    output
}

fn external_axis(
    index: ExternalIndex,
    momentum: usize,
    h_vector: usize,
    output_axes: &[usize; 4],
) -> usize {
    match index {
        ExternalIndex::Momentum => momentum,
        ExternalIndex::HVector => h_vector,
        ExternalIndex::Output0 => output_axes[0],
        ExternalIndex::Output1 => output_axes[1],
        ExternalIndex::Output2 => output_axes[2],
        ExternalIndex::Output3 => output_axes[3],
    }
}

fn attachment_metric(index: ExternalIndex, axis: usize) -> i64 {
    match index {
        ExternalIndex::Momentum => 1,
        ExternalIndex::HVector
        | ExternalIndex::Output0
        | ExternalIndex::Output1
        | ExternalIndex::Output2
        | ExternalIndex::Output3 => lorentz_sign(axis),
    }
}

fn metric_pair_factor(pair: [ExternalIndex; 2], left_axis: usize, right_axis: usize) -> i64 {
    if left_axis != right_axis {
        return 0;
    }
    match pair {
        [ExternalIndex::Momentum, ExternalIndex::HVector]
        | [ExternalIndex::HVector, ExternalIndex::Momentum]
        | [
            ExternalIndex::Momentum,
            ExternalIndex::Output0
            | ExternalIndex::Output1
            | ExternalIndex::Output2
            | ExternalIndex::Output3,
        ]
        | [
            ExternalIndex::Output0
            | ExternalIndex::Output1
            | ExternalIndex::Output2
            | ExternalIndex::Output3,
            ExternalIndex::Momentum,
        ] => 1,
        _ => lorentz_sign(left_axis),
    }
}

fn metric_pair_factor_h_output_delta_mutation(
    pair: [ExternalIndex; 2],
    left_axis: usize,
    right_axis: usize,
) -> i64 {
    if left_axis != right_axis {
        return 0;
    }
    match pair {
        [ExternalIndex::HVector, ExternalIndex::Output0]
        | [ExternalIndex::HVector, ExternalIndex::Output1]
        | [ExternalIndex::HVector, ExternalIndex::Output2]
        | [ExternalIndex::HVector, ExternalIndex::Output3]
        | [ExternalIndex::Output0, ExternalIndex::HVector]
        | [ExternalIndex::Output1, ExternalIndex::HVector]
        | [ExternalIndex::Output2, ExternalIndex::HVector]
        | [ExternalIndex::Output3, ExternalIndex::HVector] => 1,
        _ => metric_pair_factor(pair, left_axis, right_axis),
    }
}

fn h_hat_integer_basis() -> &'static Vec<SparseIntegerVector> {
    static BASIS: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    BASIS.get_or_init(|| {
        canonical_gamma_traceless_frame_basis()
            .into_iter()
            .map(|column| {
                column
                    .into_iter()
                    .map(|(coordinate, value)| {
                        assert_eq!(*value.real.denom(), 1);
                        assert_eq!(value.imaginary, Ratio::from_integer(0));
                        (coordinate, *value.real.numer())
                    })
                    .collect()
            })
            .collect()
    })
}

fn axis_combinations(count: usize) -> Vec<Vec<usize>> {
    fn visit(count: usize, next: usize, current: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
        if current.len() == count {
            output.push(current.clone());
            return;
        }
        for axis in next..VECTOR_DIMENSION {
            current.push(axis);
            visit(count, axis + 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    visit(count, 0, &mut Vec::new(), &mut output);
    output
}

fn gamma_axis_sequence(axes: &[usize]) -> Option<(u16, i64)> {
    let mut mask = 0_u16;
    let mut inversions = 0;
    for (position, &axis) in axes.iter().enumerate() {
        if mask & (1_u16 << axis) != 0 {
            return None;
        }
        inversions += axes[..position]
            .iter()
            .filter(|&&prior| prior > axis)
            .count();
        mask |= 1_u16 << axis;
    }
    Some((mask, if inversions % 2 == 0 { 1 } else { -1 }))
}

fn signed_output_axis_permutations(
    sorted_axes: [usize; 4],
    antisymmetrize: bool,
) -> Vec<([usize; 4], i64)> {
    if !antisymmetrize {
        return vec![(sorted_axes, 1)];
    }
    fn visit(position: usize, axes: &mut [usize; 4], output: &mut Vec<([usize; 4], i64)>) {
        if position == axes.len() {
            let inversions = (0..4)
                .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
                .filter(|&(left, right)| axes[left] > axes[right])
                .count();
            output.push((*axes, if inversions % 2 == 0 { 1 } else { -1 }));
            return;
        }
        for next in position..axes.len() {
            axes.swap(position, next);
            visit(position + 1, axes, output);
            axes.swap(position, next);
        }
    }
    let mut axes = sorted_axes;
    let mut output = Vec::with_capacity(24);
    visit(0, &mut axes, &mut output);
    output.sort_unstable_by_key(|(axes, _)| *axes);
    output
}

fn query_axis(label: u8, query: &D21CoefficientQuery) -> usize {
    match label {
        0 => usize::from(query.momentum),
        1 => usize::from(query.h_vector),
        2..=5 => usize::from(query.output_axes[usize::from(label - 2)]),
        _ => panic!("invalid packed external label {label}"),
    }
}

pub fn evaluate_packed_query_cpu(diagrams: &[PackedD21Diagram], query: D21CoefficientQuery) -> i64 {
    let diagram = diagrams[usize::from(query.diagram)];
    let mut base = i64::from(query.h_coefficient);
    for pair in 0..usize::from(diagram.metric_count) {
        let left = diagram.metric_pairs[2 * pair];
        let right = diagram.metric_pairs[2 * pair + 1];
        let left_axis = query_axis(left, &query);
        let right_axis = query_axis(right, &query);
        if left_axis != right_axis {
            return 0;
        }
        base *= if (left == 0 && right == 1)
            || (left == 1 && right == 0)
            || (left == 1 && right >= 2)
            || (right == 1 && left >= 2)
        {
            1
        } else {
            lorentz_sign(left_axis)
        };
    }
    let mut outer_external = Vec::new();
    for slot in 0..usize::from(diagram.outer_count) {
        let label = diagram.outer_external[slot];
        let axis = query_axis(label, &query);
        base *= if label == 0 { 1 } else { lorentz_sign(axis) };
        outer_external.push(axis);
    }
    let mut inner_external = Vec::new();
    for slot in 0..usize::from(diagram.inner_count) {
        let label = diagram.inner_external[slot];
        let axis = query_axis(label, &query);
        base *= if label == 0 { 1 } else { lorentz_sign(axis) };
        inner_external.push(axis);
    }
    if base == 0 {
        return 0;
    }
    let mut sum = 0_i64;
    for internal in axis_combinations(usize::from(diagram.cross)) {
        let mut outer_axes = outer_external.clone();
        outer_axes.extend(&internal);
        let Some((outer_mask, outer_sign)) = gamma_axis_sequence(&outer_axes) else {
            continue;
        };
        let mut inner_axes = inner_external.clone();
        inner_axes.extend(&internal);
        let Some((inner_mask, inner_sign)) = gamma_axis_sequence(&inner_axes) else {
            continue;
        };
        let cross_metric = internal
            .iter()
            .map(|&axis| lorentz_sign(axis))
            .product::<i64>();
        let left = i64::from(
            charge_gamma_mask_matrices()[usize::from(outer_mask)][usize::from(query.outer_left)]
                [usize::from(query.outer_right)],
        );
        let right = i64::from(
            gamma_mask_matrices()[usize::from(inner_mask)][usize::from(query.output_spinor)]
                [usize::from(query.input_spinor)],
        );
        sum += base * outer_sign * inner_sign * cross_metric * left * right;
    }
    sum
}

fn evaluate_diagram_with_output_mode(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
    antisymmetrize_output: bool,
    h_output_delta_mutation: bool,
) -> SparseIntegerVector {
    let h_basis = h_hat_integer_basis();
    let forms4 = form_masks(4);
    let internal_choices = axis_combinations(diagram.cross_gamma_contractions);
    let mut output = SparseIntegerVector::new();
    for (&h_coordinate, &h_coefficient) in &h_basis[h_hat_ordinal] {
        let input_spinor = h_coordinate / VECTOR_DIMENSION;
        let h_vector = h_coordinate % VECTOR_DIMENSION;
        for (form_ordinal, &mask) in forms4.iter().enumerate() {
            let axes = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            let sorted_output_axes = [axes[0], axes[1], axes[2], axes[3]];
            for (output_axes, permutation_sign) in
                signed_output_axis_permutations(sorted_output_axes, antisymmetrize_output)
            {
                let mut scalar = h_coefficient * permutation_sign;
                for &pair in &diagram.metric_pairs {
                    let left_axis = external_axis(pair[0], momentum, h_vector, &output_axes);
                    let right_axis = external_axis(pair[1], momentum, h_vector, &output_axes);
                    scalar *= if h_output_delta_mutation {
                        metric_pair_factor_h_output_delta_mutation(pair, left_axis, right_axis)
                    } else {
                        metric_pair_factor(pair, left_axis, right_axis)
                    };
                }
                if scalar == 0 {
                    continue;
                }
                let mut outer_external_axes = Vec::new();
                for &index in &diagram.outer_gamma_external_indices {
                    let axis = external_axis(index, momentum, h_vector, &output_axes);
                    scalar *= attachment_metric(index, axis);
                    outer_external_axes.push(axis);
                }
                let mut inner_external_axes = Vec::new();
                for &index in &diagram.inner_gamma_external_indices {
                    let axis = external_axis(index, momentum, h_vector, &output_axes);
                    scalar *= attachment_metric(index, axis);
                    inner_external_axes.push(axis);
                }
                if scalar == 0 {
                    continue;
                }
                for internal in &internal_choices {
                    let mut outer_axes = outer_external_axes.clone();
                    outer_axes.extend(internal);
                    let Some((outer_mask, outer_sign)) = gamma_axis_sequence(&outer_axes) else {
                        continue;
                    };
                    let mut inner_axes = inner_external_axes.clone();
                    inner_axes.extend(internal);
                    let Some((inner_mask, inner_sign)) = gamma_axis_sequence(&inner_axes) else {
                        continue;
                    };
                    let cross_metric = internal
                        .iter()
                        .map(|&axis| lorentz_sign(axis))
                        .product::<i64>();
                    let outer_value = i64::from(
                        charge_gamma_mask_matrices()[usize::from(outer_mask)][outer_pair[0]]
                            [outer_pair[1]],
                    );
                    if outer_value == 0 {
                        continue;
                    }
                    let coefficient = scalar * outer_sign * inner_sign * cross_metric * outer_value;
                    let gamma = &gamma_mask_matrices()[usize::from(inner_mask)];
                    for (output_spinor, row) in gamma.iter().enumerate() {
                        add_integer(
                            &mut output,
                            output_spinor * forms4.len() + form_ordinal,
                            coefficient * i64::from(row[input_spinor]),
                        );
                    }
                }
            }
        }
    }
    output
}

fn evaluate_diagram(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
) -> SparseIntegerVector {
    evaluate_diagram_with_output_mode(diagram, outer_pair, momentum, h_hat_ordinal, true, false)
}

fn evaluate_diagram_identity_output_mutation(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
) -> SparseIntegerVector {
    evaluate_diagram_with_output_mode(diagram, outer_pair, momentum, h_hat_ordinal, false, false)
}

fn evaluate_diagram_h_output_delta_mutation(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
) -> SparseIntegerVector {
    evaluate_diagram_with_output_mode(diagram, outer_pair, momentum, h_hat_ordinal, true, true)
}

fn covector_boost_action(axis: usize, left: usize, right: usize) -> Vec<(usize, i64)> {
    let mut output = Vec::new();
    if axis == left {
        output.push((right, -lorentz_sign(right)));
    }
    if axis == right {
        output.push((left, lorentz_sign(left)));
    }
    output
}

fn vector_boost_action(axis: usize, left: usize, right: usize) -> Vec<(usize, i64)> {
    let mut output = Vec::new();
    if axis == right {
        output.push((left, lorentz_sign(right)));
    }
    if axis == left {
        output.push((right, -lorentz_sign(left)));
    }
    output
}

fn h_hat_generator_coefficients(
    h_hat_ordinal: usize,
    left: usize,
    right: usize,
) -> SparseIntegerVector {
    let h = &h_hat_integer_basis()[h_hat_ordinal];
    let mask = (1_u16 << left) | (1_u16 << right);
    let metric = lorentz_sign(left) * lorentz_sign(right);
    let spin = &gamma_mask_matrices()[usize::from(mask)];
    let mut acted = SparseIntegerVector::new();
    for (&coordinate, &coefficient) in h {
        let input_spinor = coordinate / VECTOR_DIMENSION;
        let input_vector = coordinate % VECTOR_DIMENSION;
        for (next_spinor, row) in spin.iter().enumerate() {
            add_integer(
                &mut acted,
                next_spinor * VECTOR_DIMENSION + input_vector,
                coefficient * metric * i64::from(row[input_spinor]),
            );
        }
        for (next_vector, value) in vector_boost_action(input_vector, left, right) {
            add_integer(
                &mut acted,
                input_spinor * VECTOR_DIMENSION + next_vector,
                2 * coefficient * value,
            );
        }
    }
    let mut coefficients = SparseIntegerVector::new();
    for spatial_vector in 1..VECTOR_DIMENSION {
        for spinor in 0..SPINOR_DIMENSION {
            add_integer(
                &mut coefficients,
                (spatial_vector - 1) * SPINOR_DIMENSION + spinor,
                acted
                    .get(&(spinor * VECTOR_DIMENSION + spatial_vector))
                    .copied()
                    .unwrap_or(0),
            );
        }
    }
    coefficients
}

fn scalar_diagram_boost_residual_entries(
    diagram: &D21InvariantDiagram,
    momentum: usize,
    h_hat_ordinal: usize,
    left: usize,
    right: usize,
    evaluate: fn(&D21InvariantDiagram, [usize; 2], usize, usize) -> SparseIntegerVector,
) -> Result<usize, String> {
    let outer_pair = [0, 17];
    let column = evaluate(diagram, outer_pair, momentum, h_hat_ordinal);
    let lhs = dg4_lorentz_generator_action_integer(left, right, &column)?;
    let mut rhs = SparseIntegerVector::new();
    for (next_momentum, coefficient) in covector_boost_action(momentum, left, right) {
        for (row, value) in evaluate(diagram, outer_pair, next_momentum, h_hat_ordinal) {
            add_integer(&mut rhs, row, 2 * coefficient * value);
        }
    }
    for (next_h, coefficient) in h_hat_generator_coefficients(h_hat_ordinal, left, right) {
        for (row, value) in evaluate(diagram, outer_pair, momentum, next_h) {
            add_integer(&mut rhs, row, coefficient * value);
        }
    }
    Ok(lhs
        .keys()
        .chain(rhs.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|row| lhs.get(row) != rhs.get(row))
        .count())
}

fn full_diagram_lorentz_residual_entries(
    diagram: &D21InvariantDiagram,
    source_coordinate: u32,
    left: usize,
    right: usize,
) -> Result<usize, String> {
    let (outer_pair, momentum, h_hat) = decode_source_coordinate(source_coordinate)?;
    let column = evaluate_diagram(diagram, outer_pair, momentum, h_hat);
    let lhs = dg4_lorentz_generator_action_integer(left, right, &column)?;
    let mut rhs = SparseIntegerVector::new();
    for term in d21_source_lorentz_generator_terms(source_coordinate, left, right)? {
        let (next_pair, next_momentum, next_h) = decode_source_coordinate(term.source_coordinate)?;
        for (row, value) in evaluate_diagram(diagram, next_pair, next_momentum, next_h) {
            add_integer(&mut rhs, row, term.coefficient * value);
        }
    }
    Ok(lhs
        .keys()
        .chain(rhs.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|row| lhs.get(row) != rhs.get(row))
        .count())
}

fn signed_permutation_image(matrix: &Matrix, column: usize) -> Result<(usize, i64), String> {
    let entries = matrix
        .iter()
        .enumerate()
        .filter_map(|(row, values)| (values[column] != 0).then_some((row, values[column])))
        .collect::<Vec<_>>();
    match entries.as_slice() {
        &[(row, value @ (-1 | 1))] => Ok((row, i64::from(value))),
        _ => Err("D21 inner gamma is not a signed-permutation column".to_string()),
    }
}

/// Exact sparse inverse-emission oracle. It is algebraically identical to
/// `evaluate_diagram`, but uses the unique nonzero row of each signed-
/// permutation gamma column instead of scanning all 32 output spinors.
fn evaluate_diagram_inverse_sparse_with_output_mode(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
    antisymmetrize_output: bool,
) -> Result<SparseIntegerVector, String> {
    let h_basis = h_hat_integer_basis();
    let forms4 = form_masks(4);
    let internal_choices = axis_combinations(diagram.cross_gamma_contractions);
    let mut output = SparseIntegerVector::new();
    for (&h_coordinate, &h_coefficient) in &h_basis[h_hat_ordinal] {
        let input_spinor = h_coordinate / VECTOR_DIMENSION;
        let h_vector = h_coordinate % VECTOR_DIMENSION;
        for (form_ordinal, &mask) in forms4.iter().enumerate() {
            let axes = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            let sorted_output_axes = [axes[0], axes[1], axes[2], axes[3]];
            for (output_axes, permutation_sign) in
                signed_output_axis_permutations(sorted_output_axes, antisymmetrize_output)
            {
                let mut scalar = h_coefficient * permutation_sign;
                for &pair in &diagram.metric_pairs {
                    scalar *= metric_pair_factor(
                        pair,
                        external_axis(pair[0], momentum, h_vector, &output_axes),
                        external_axis(pair[1], momentum, h_vector, &output_axes),
                    );
                }
                if scalar == 0 {
                    continue;
                }
                let mut outer_external_axes = Vec::new();
                for &index in &diagram.outer_gamma_external_indices {
                    let axis = external_axis(index, momentum, h_vector, &output_axes);
                    scalar *= attachment_metric(index, axis);
                    outer_external_axes.push(axis);
                }
                let mut inner_external_axes = Vec::new();
                for &index in &diagram.inner_gamma_external_indices {
                    let axis = external_axis(index, momentum, h_vector, &output_axes);
                    scalar *= attachment_metric(index, axis);
                    inner_external_axes.push(axis);
                }
                if scalar == 0 {
                    continue;
                }
                for internal in &internal_choices {
                    let mut outer_axes = outer_external_axes.clone();
                    outer_axes.extend(internal);
                    let Some((outer_mask, outer_sign)) = gamma_axis_sequence(&outer_axes) else {
                        continue;
                    };
                    let mut inner_axes = inner_external_axes.clone();
                    inner_axes.extend(internal);
                    let Some((inner_mask, inner_sign)) = gamma_axis_sequence(&inner_axes) else {
                        continue;
                    };
                    let outer_value = i64::from(
                        charge_gamma_mask_matrices()[usize::from(outer_mask)][outer_pair[0]]
                            [outer_pair[1]],
                    );
                    if outer_value == 0 {
                        continue;
                    }
                    let cross_metric = internal
                        .iter()
                        .map(|&axis| lorentz_sign(axis))
                        .product::<i64>();
                    let coefficient = scalar * outer_sign * inner_sign * cross_metric * outer_value;
                    let (output_spinor, gamma_value) = signed_permutation_image(
                        &gamma_mask_matrices()[usize::from(inner_mask)],
                        input_spinor,
                    )?;
                    add_integer(
                        &mut output,
                        output_spinor * forms4.len() + form_ordinal,
                        coefficient * gamma_value,
                    );
                }
            }
        }
    }
    Ok(output)
}

fn evaluate_diagram_inverse_sparse(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
) -> Result<SparseIntegerVector, String> {
    evaluate_diagram_inverse_sparse_with_output_mode(
        diagram,
        outer_pair,
        momentum,
        h_hat_ordinal,
        true,
    )
}

pub(crate) const D21_SOURCE_COORDINATES: usize = 496 * VECTOR_DIMENSION * 320;

pub(crate) fn decode_source_coordinate(source: u32) -> Result<([usize; 2], usize, usize), String> {
    let source = usize::try_from(source).unwrap();
    if source >= D21_SOURCE_COORDINATES {
        return Err("D21 inverse-emission source coordinate is out of range".to_string());
    }
    let h_hat = source % 320;
    let quotient = source / 320;
    let momentum = quotient % VECTOR_DIMENSION;
    let pair_ordinal = quotient / VECTOR_DIMENSION;
    Ok((spinor_pairs()[pair_ordinal], momentum, h_hat))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct D21SourceLorentzTerm {
    pub source_coordinate: u32,
    pub coefficient: i64,
}

fn encode_source_coordinate(
    pair: [usize; 2],
    momentum: usize,
    h_hat: usize,
) -> Result<u32, String> {
    if pair[0] >= pair[1]
        || pair[1] >= SPINOR_DIMENSION
        || momentum >= VECTOR_DIMENSION
        || h_hat >= 320
    {
        return Err("D21 source coordinate parts are out of range".to_string());
    }
    let pair_ordinal = spinor_pairs()
        .iter()
        .position(|candidate| *candidate == pair)
        .ok_or_else(|| "D21 spinor pair is absent".to_string())?;
    u32::try_from((pair_ordinal * VECTOR_DIMENSION + momentum) * 320 + h_hat)
        .map_err(|_| "D21 source coordinate overflowed u32".to_string())
}

/// Apply twice the Lorentz generator to the canonical
/// `Lambda^2 S tensor V* tensor Hhat` source basis. The normalization matches
/// `dg4_lorentz_generator_action_integer`: `Gamma_ab` on each spinor slot and
/// twice the tensor generator on vector slots.
pub(crate) fn d21_source_lorentz_generator_terms(
    source_coordinate: u32,
    left: usize,
    right: usize,
) -> Result<Vec<D21SourceLorentzTerm>, String> {
    if left >= right || right >= VECTOR_DIMENSION {
        return Err("D21 Lorentz generator indices are invalid".to_string());
    }
    let (pair, momentum, h_hat) = decode_source_coordinate(source_coordinate)?;
    let mut output = BTreeMap::<u32, i64>::new();
    let mut add = |next_pair: [usize; 2], next_momentum: usize, next_h: usize, value: i64| {
        if value == 0 {
            return Ok(());
        }
        let next = encode_source_coordinate(next_pair, next_momentum, next_h)?;
        *output.entry(next).or_default() += value;
        if output[&next] == 0 {
            output.remove(&next);
        }
        Ok::<(), String>(())
    };

    let mask = (1_u16 << left) | (1_u16 << right);
    let spin = &gamma_mask_matrices()[usize::from(mask)];
    let spin_metric = lorentz_sign(left) * lorentz_sign(right);
    for slot in 0..2 {
        let old_spinor = pair[slot];
        let other = pair[1 - slot];
        for (next_spinor, row) in spin.iter().enumerate() {
            let mut coefficient = spin_metric * i64::from(row[old_spinor]);
            if coefficient == 0 || next_spinor == other {
                continue;
            }
            let next_pair = if slot == 0 && next_spinor < other {
                [next_spinor, other]
            } else if slot == 0 {
                coefficient = -coefficient;
                [other, next_spinor]
            } else if other < next_spinor {
                [other, next_spinor]
            } else {
                coefficient = -coefficient;
                [next_spinor, other]
            };
            add(next_pair, momentum, h_hat, coefficient)?;
        }
    }
    for (next_momentum, coefficient) in covector_boost_action(momentum, left, right) {
        add(pair, next_momentum, h_hat, 2 * coefficient)?;
    }
    for (next_h, coefficient) in h_hat_generator_coefficients(h_hat, left, right) {
        add(pair, momentum, next_h, coefficient)?;
    }
    Ok(output
        .into_iter()
        .map(|(source_coordinate, coefficient)| D21SourceLorentzTerm {
            source_coordinate,
            coefficient,
        })
        .collect())
}

pub fn validate_compact_raw_coo(entries: &[D21CompactRawCooEntry]) -> Result<(), String> {
    let mut previous = None;
    for entry in entries {
        if usize::try_from(entry.source_coordinate).unwrap() >= D21_SOURCE_COORDINATES
            || usize::from(entry.target_coordinate) >= SPINOR_DIMENSION * 330
            || usize::from(entry.diagram_ordinal) >= 400
            || entry.coefficient == 0
        {
            return Err("D21 compact raw COO entry is invalid".to_string());
        }
        let key = (
            entry.source_coordinate,
            entry.target_coordinate,
            entry.diagram_ordinal,
        );
        if previous.is_some_and(|prior| prior >= key) {
            return Err("D21 compact raw COO is not strictly canonical".to_string());
        }
        previous = Some(key);
    }
    Ok(())
}

pub fn compact_raw_coo_sha256(entries: &[D21CompactRawCooEntry]) -> Result<String, String> {
    validate_compact_raw_coo(entries)?;
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-d21-compact-raw-coo-v1\0");
    for entry in entries {
        hash.update(entry.source_coordinate.to_le_bytes());
        hash.update(entry.target_coordinate.to_le_bytes());
        hash.update(entry.diagram_ordinal.to_le_bytes());
        hash.update(entry.coefficient.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Emit exact raw coefficients for one source coordinate and an ordered
/// diagram subset. Output is compact and canonically sorted by source,
/// target, then diagram, matching the device compaction contract.
pub fn inverse_sparse_emission_oracle(
    source_coordinate: u32,
    diagram_ordinals: &[u16],
) -> Result<Vec<D21CompactRawCooEntry>, String> {
    if diagram_ordinals.is_empty()
        || diagram_ordinals.windows(2).any(|pair| pair[0] >= pair[1])
        || diagram_ordinals.iter().any(|&diagram| diagram >= 400)
    {
        return Err("D21 inverse-emission diagram subset is not canonical".to_string());
    }
    let (outer_pair, momentum, h_hat) = decode_source_coordinate(source_coordinate)?;
    let diagrams = enumerate_diagrams();
    let mut output = Vec::new();
    for &diagram_ordinal in diagram_ordinals {
        let column = evaluate_diagram_inverse_sparse(
            &diagrams[usize::from(diagram_ordinal)],
            outer_pair,
            momentum,
            h_hat,
        )?;
        output.extend(
            column
                .into_iter()
                .map(|(target, coefficient)| D21CompactRawCooEntry {
                    source_coordinate,
                    target_coordinate: u16::try_from(target).unwrap(),
                    diagram_ordinal,
                    coefficient,
                }),
        );
    }
    output.sort_unstable();
    validate_compact_raw_coo(&output)?;
    Ok(output)
}

/// Replay one device-selected sector pivot through raw inverse emission and
/// the exact four-pass C4 projector numerator. The numerator/denominator pair
/// compares directly with modular device output without rational ambiguity.
pub fn replay_sector_pivot_v2(
    request: D21SectorPivotReplayRequestV2,
) -> Result<D21SectorPivotReplayResultV2, String> {
    if usize::from(request.target_coordinate) >= SPINOR_DIMENSION * 330 {
        return Err("D21 sector pivot target coordinate is out of range".to_string());
    }
    let raw =
        inverse_sparse_emission_oracle(request.source_coordinate, &[request.diagram_ordinal])?;
    let compact = raw
        .iter()
        .map(|entry| (entry.target_coordinate, entry.coefficient))
        .collect::<Vec<_>>();
    let raw_coefficient = raw
        .iter()
        .find(|entry| entry.target_coordinate == request.target_coordinate)
        .map(|entry| entry.coefficient)
        .unwrap_or(0);
    let projector = dg4_projector_numerator_oracle(&request.target_sector, &compact)?;
    let pass_nonzero_counts: [usize; 4] = projector
        .passes
        .iter()
        .map(|pass| pass.entries.len())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "D21 sector pivot replay did not execute four C4 passes".to_string())?;
    let projected_numerator = projector
        .numerator
        .iter()
        .find_map(|&(row, value)| (row == request.target_coordinate).then_some(value))
        .unwrap_or(0);
    Ok(D21SectorPivotReplayResultV2 {
        request,
        raw_coefficient,
        ordered_shift_eigenvalues: projector.ordered_shift_eigenvalues,
        pass_nonzero_counts,
        projector_denominator: projector.denominator,
        projected_numerator,
        passed_nonzero: projected_numerator != 0,
    })
}

fn evaluate_outer_zero_diagram(
    diagram: &D21InvariantDiagram,
    outer_pair: [usize; 2],
    momentum: usize,
    h_hat_ordinal: usize,
) -> SparseIntegerVector {
    assert_eq!(diagram.outer_bilinear_degree, 0);
    evaluate_diagram(diagram, outer_pair, momentum, h_hat_ordinal)
}

fn exact_sparse_column_rank(columns: &[BTreeMap<usize, Ratio<i64>>]) -> usize {
    let mut rows = BTreeMap::<usize, Vec<Ratio<i64>>>::new();
    for (column, entries) in columns.iter().enumerate() {
        for (&row, value) in entries {
            rows.entry(row)
                .or_insert_with(|| vec![Ratio::from_integer(0); columns.len()])[column] =
                value.clone();
        }
    }
    let mut basis = Vec::<(usize, Vec<Ratio<i64>>)>::new();
    for mut row in rows.into_values() {
        for (pivot, existing) in &basis {
            let scale = row[*pivot].clone();
            if scale == Ratio::from_integer(0) {
                continue;
            }
            for column in *pivot..row.len() {
                row[column] -= scale.clone() * existing[column].clone();
            }
        }
        let Some(pivot) = (0..row.len()).find(|&column| row[column] != Ratio::from_integer(0))
        else {
            continue;
        };
        let scale = row[pivot].clone();
        for value in &mut row {
            *value /= scale.clone();
        }
        basis.push((pivot, row));
    }
    basis.len()
}

fn spinor_pairs() -> Vec<[usize; 2]> {
    (0..SPINOR_DIMENSION)
        .flat_map(|left| ((left + 1)..SPINOR_DIMENSION).map(move |right| [left, right]))
        .collect()
}

fn rational_string(value: &Ratio<i64>) -> String {
    if *value.denom() == 1 {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

pub fn replay_projected_pivot(
    request: D21PivotReplayRequest,
) -> Result<D21PivotReplayResult, String> {
    let diagrams = enumerate_diagrams();
    let pairs = spinor_pairs();
    if request.diagram_ordinal >= diagrams.len()
        || request.outer_pair_ordinal >= pairs.len()
        || request.momentum_axis >= VECTOR_DIMENSION
        || request.h_hat_ordinal >= 320
        || request.target_coordinate >= 32 * 330
    {
        return Err("D21 pivot replay coordinate is out of range".to_string());
    }
    if !["00001", "00011", "00101", "01001", "10001"].contains(&request.target_sector.as_str()) {
        return Err("D21 pivot replay target sector is invalid".to_string());
    }
    let outer_pair = pairs[request.outer_pair_ordinal];
    let raw = evaluate_diagram(
        &diagrams[request.diagram_ordinal],
        outer_pair,
        request.momentum_axis,
        request.h_hat_ordinal,
    );
    let raw_coefficient =
        Ratio::from_integer(raw.get(&request.target_coordinate).copied().unwrap_or(0));
    let rational = raw
        .into_iter()
        .map(|(row, value)| (row, Ratio::from_integer(value)))
        .collect::<BTreeMap<_, _>>();
    let projected = project_dg4_target(&request.target_sector, &rational)?;
    let projected_coefficient = projected
        .get(&request.target_coordinate)
        .cloned()
        .unwrap_or_else(|| Ratio::from_integer(0));
    Ok(D21PivotReplayResult {
        request,
        outer_pair,
        raw_coefficient: rational_string(&raw_coefficient),
        projected_coefficient: rational_string(&projected_coefficient),
        passed: projected_coefficient != Ratio::from_integer(0),
    })
}

pub fn build_outer_zero_canary() -> Result<D21OuterZeroCanaryReport, String> {
    let diagrams = enumerate_diagrams()
        .into_iter()
        .filter(|diagram| diagram.outer_bilinear_degree == 0)
        .collect::<Vec<_>>();
    let charge = real_charge_conjugation();
    let outer_pair = (0..SPINOR_DIMENSION)
        .flat_map(|left| ((left + 1)..SPINOR_DIMENSION).map(move |right| [left, right]))
        .find(|pair| charge[pair[0]][pair[1]] != 0)
        .ok_or_else(|| "antisymmetric charge bilinear has no nonzero outer pair".to_string())?;
    let raw = diagrams
        .iter()
        .map(|diagram| evaluate_outer_zero_diagram(diagram, outer_pair, 0, 0))
        .collect::<Vec<_>>();
    let nonzero_raw_diagrams = raw.iter().filter(|column| !column.is_empty()).count();
    let sectors = ["00001", "00011", "00101", "01001", "10001"];
    let mut ranks = BTreeMap::new();
    for sector in sectors {
        let projected = raw
            .iter()
            .map(|column| {
                let rational = column
                    .iter()
                    .map(|(&row, &value)| (row, Ratio::from_integer(value)))
                    .collect::<BTreeMap<_, _>>();
                project_dg4_target(sector, &rational)
            })
            .collect::<Result<Vec<_>, _>>()?;
        ranks.insert(sector, exact_sparse_column_rank(&projected));
    }
    let expected = BTreeMap::from([
        ("00001", 7),
        ("00011", 7),
        ("00101", 11),
        ("01001", 14),
        ("10001", 13),
    ]);
    Ok(D21OuterZeroCanaryReport {
        schema_version: "adynkra-11d-d21-outer-zero-canary-v1",
        diagrams_evaluated: diagrams.len(),
        outer_pair,
        momentum_axis: 0,
        h_hat_ordinal: 0,
        nonzero_raw_diagrams,
        exact_sector_ranks: ranks,
        expected_full_sector_ranks: expected,
        is_lower_bound_only: true,
        passed_canary: nonzero_raw_diagrams > 0,
        boundary: "Ranks use one fixed source column and only the 21 outer-degree-zero diagrams. They are certified lower bounds, not sector completion claims.",
    })
}

pub fn build_report() -> D21InvariantDiagramReport {
    let (antisymmetric, symmetric, residuals) = outer_bilinear_symmetries();
    let diagrams = enumerate_diagrams();
    let unique = diagrams
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut by_outer = BTreeMap::new();
    let mut by_inner = BTreeMap::new();
    for diagram in &diagrams {
        *by_outer.entry(diagram.outer_bilinear_degree).or_default() += 1;
        *by_inner.entry(diagram.inner_to_output_degree).or_default() += 1;
    }
    let duplicate_signatures = diagrams.len() - unique.len();
    let (packed_diagrams_sha256, gamma_mask_table_sha256, charge_gamma_mask_table_sha256) =
        packed_contract_sha256();
    let (
        volume_element_scalar,
        volume_element_residual_entries,
        hodge_duality_masks_checked,
        hodge_duality_residual_masks,
    ) = epsilon_hodge_certificate();
    let epsilon_hodge_redundancy_proved = volume_element_scalar.abs() == 1
        && volume_element_residual_entries == 0
        && hodge_duality_masks_checked == 1024
        && hodge_duality_residual_masks == 0;
    D21InvariantDiagramReport {
        schema_version: "adynkra-11d-d21-invariant-diagram-grammar-v1",
        fierz_pairing: "(outer spinor wedge outer spinor) | (H spinor to output spinor)",
        clifford_degrees_checked: 6,
        antisymmetric_outer_bilinear_degrees: antisymmetric,
        symmetric_outer_bilinear_degrees: symmetric,
        mixed_symmetry_residual_masks: residuals,
        raw_diagrams: diagrams.len(),
        diagrams_by_outer_degree: by_outer,
        diagrams_by_target_gamma_degree: by_inner,
        duplicate_signatures,
        packed_diagrams_sha256,
        gamma_mask_table_sha256,
        charge_gamma_mask_table_sha256,
        volume_element_scalar,
        volume_element_residual_entries,
        hodge_duality_masks_checked,
        hodge_duality_residual_masks,
        epsilon_hodge_redundancy_proved,
        outer_spinor_antisymmetrization_required: true,
        target_projection_required: true,
        passed_grammar: residuals == 0
            && duplicate_signatures == 0
            && !diagrams.is_empty()
            && epsilon_hodge_redundancy_proved,
        boundary: "The grammar is Fierz-complete before tensor-identity reduction. Explicit epsilon opcodes are redundant only because the exact 11D volume-element and rank-6-through-11 Hodge gates pass; any convention drift fails the grammar closed. It does not claim diagram independence, PBW rank, target-sector rank, or a complete 52-map basis until Cartesian evaluation and streamed RREF close those gates.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d21_fierz_diagram_grammar_is_exact_and_deduplicated() {
        let report = build_report();
        eprintln!("D21_DIAGRAM_GRAMMAR {report:?}");
        assert!(report.passed_grammar);
        assert_eq!(report.mixed_symmetry_residual_masks, 0);
        assert_eq!(report.duplicate_signatures, 0);
        assert_eq!(report.volume_element_scalar.abs(), 1);
        assert_eq!(report.volume_element_residual_entries, 0);
        assert_eq!(report.hodge_duality_masks_checked, 1024);
        assert_eq!(report.hodge_duality_residual_masks, 0);
        assert!(report.epsilon_hodge_redundancy_proved);
        assert_eq!(
            report.antisymmetric_outer_bilinear_degrees.len()
                + report.symmetric_outer_bilinear_degrees.len(),
            6
        );
    }

    #[test]
    fn d21_epsilon_hodge_gate_detects_mutation() {
        let matrices = gamma_mask_matrices();
        let full_mask = (1_usize << VECTOR_DIMENSION) - 1;
        let mut mutated_volume = matrices[full_mask].clone();
        mutated_volume[0][0] += 1;
        let scalar = mutated_volume[0][0];
        let residuals = (0..SPINOR_DIMENSION)
            .flat_map(|row| {
                let value = mutated_volume.clone();
                (0..SPINOR_DIMENSION)
                    .map(move |column| value[row][column] != if row == column { scalar } else { 0 })
            })
            .filter(|residual| *residual)
            .count();
        assert!(residuals > 0);

        let mask = form_masks(6)[0];
        let complement = u16::try_from(full_mask).unwrap() ^ mask;
        let mut mutated = matrices[usize::from(complement)].clone();
        mutated[0][0] += 1;
        assert_ne!(matrices[usize::from(mask)], mutated);
    }

    #[test]
    fn d21_outer_zero_cartesian_canary_is_nonzero() {
        let report = build_outer_zero_canary().unwrap();
        eprintln!("D21_OUTER_ZERO_CANARY {report:?}");
        assert!(report.passed_canary);
        assert_eq!(report.diagrams_evaluated, 21);
        assert!(report.is_lower_bound_only);
    }

    #[test]
    fn d21_scalar_10001_excess_is_localized() {
        let all = enumerate_diagrams();
        let selected = all
            .iter()
            .enumerate()
            .filter(|(_, diagram)| diagram.outer_bilinear_degree == 0)
            .collect::<Vec<_>>();
        let outer_pair = [0, 17];
        let projected = selected
            .iter()
            .map(|(_, diagram)| {
                let raw = evaluate_diagram(diagram, outer_pair, 0, 0)
                    .into_iter()
                    .map(|(row, value)| (row, Ratio::from_integer(value)))
                    .collect::<BTreeMap<_, _>>();
                project_dg4_target("10001", &raw).unwrap()
            })
            .collect::<Vec<_>>();
        let mut rows = BTreeMap::<usize, Vec<Ratio<i64>>>::new();
        for (column, entries) in projected.iter().enumerate() {
            for (&row, value) in entries {
                rows.entry(row)
                    .or_insert_with(|| vec![Ratio::from_integer(0); projected.len()])[column] =
                    value.clone();
            }
        }
        let mut basis = Vec::<(usize, usize, Vec<Ratio<i64>>)>::new();
        for (witness_row, mut row) in rows {
            for (pivot, _, existing) in &basis {
                let scale = row[*pivot].clone();
                if scale == Ratio::from_integer(0) {
                    continue;
                }
                for column in *pivot..row.len() {
                    row[column] -= scale.clone() * existing[column].clone();
                }
            }
            let Some(pivot) = row
                .iter()
                .position(|value| *value != Ratio::from_integer(0))
            else {
                continue;
            };
            let scale = row[pivot].clone();
            for value in &mut row {
                *value /= scale.clone();
            }
            basis.push((pivot, witness_row, row));
        }
        eprintln!("D21_SCALAR_10001_EXCESS rank={}", basis.len());
        for (pivot, row, _) in &basis {
            let (global, diagram) = selected[*pivot];
            eprintln!("D21_SCALAR_10001_PIVOT diagram={global} row={row} signature={diagram:?}");
        }
        assert_eq!(
            basis.len(),
            1,
            "scalar 10001 must match exact Hom multiplicity"
        );
        let mutation_rank = |evaluate: fn(
            &D21InvariantDiagram,
            [usize; 2],
            usize,
            usize,
        ) -> SparseIntegerVector| {
            let columns = selected
                .iter()
                .map(|(_, diagram)| {
                    let raw = evaluate(diagram, outer_pair, 0, 0)
                        .into_iter()
                        .map(|(row, value)| (row, Ratio::from_integer(value)))
                        .collect::<BTreeMap<_, _>>();
                    project_dg4_target("10001", &raw).unwrap()
                })
                .collect::<Vec<_>>();
            exact_sparse_column_rank(&columns)
        };
        let identity_only_rank = mutation_rank(evaluate_diagram_identity_output_mutation);
        let h_output_delta_rank = mutation_rank(evaluate_diagram_h_output_delta_mutation);
        eprintln!(
            "D21_SCALAR_10001_MUTATIONS identity_only_rank={identity_only_rank} h_output_delta_rank={h_output_delta_rank}"
        );
        assert_eq!(identity_only_rank, 1);
        assert_eq!(h_output_delta_rank, 2);
    }

    #[test]
    fn d21_stale_scalar_01001_gpu_minor_replays_exactly() {
        let witnesses = [(56_320_u32, 14_u16), (991_077_u32, 211_u16)];
        let diagrams = [0_u16, 12_u16];
        let coefficient = |source, target, diagram| {
            replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                source_coordinate: source,
                target_coordinate: target,
                diagram_ordinal: diagram,
                target_sector: "01001".to_string(),
            })
            .unwrap()
            .projected_numerator
        };
        let corrected = [
            coefficient(witnesses[0].0, witnesses[0].1, diagrams[0]),
            coefficient(witnesses[0].0, witnesses[0].1, diagrams[1]),
            coefficient(witnesses[1].0, witnesses[1].1, diagrams[0]),
            coefficient(witnesses[1].0, witnesses[1].1, diagrams[1]),
        ];
        let corrected_determinant = corrected[0] * corrected[3] - corrected[1] * corrected[2];

        eprintln!(
            "D21_STALE_01001_MINOR corrected={corrected:?} corrected_det={corrected_determinant}"
        );
        assert_ne!(corrected[0], 0);
        assert_ne!(corrected[3], 0);
        assert_eq!(corrected_determinant, 0);
    }

    #[test]
    fn d21_gpu_packed_contract_matches_cpu_cartesian_coefficient() {
        assert_eq!(std::mem::size_of::<PackedD21Diagram>(), 32);
        assert_eq!(std::mem::size_of::<D21CoefficientQuery>(), 16);
        let diagrams = enumerate_diagrams();
        let packed = packed_diagrams();
        assert_eq!(packed.len(), 400);
        let (gamma, charge_gamma) = flattened_gamma_mask_tables();
        assert_eq!(gamma.len(), 2048 * 32 * 32);
        assert_eq!(charge_gamma.len(), gamma.len());
        let outer_pair = [0, 17];
        let h_ordinal = 0;
        let (diagram_ordinal, full) = diagrams
            .iter()
            .enumerate()
            .find_map(|(ordinal, diagram)| {
                let column = evaluate_diagram(diagram, outer_pair, 0, h_ordinal);
                (!column.is_empty()).then_some((ordinal, column))
            })
            .unwrap();
        let (&target, &expected) = full.iter().next().unwrap();
        let forms = form_masks(4);
        let axes = (0..VECTOR_DIMENSION)
            .filter(|axis| forms[target % 330] & (1_u16 << axis) != 0)
            .map(|axis| u8::try_from(axis).unwrap())
            .collect::<Vec<_>>();
        let mut actual = 0_i64;
        for (&coordinate, &coefficient) in &h_hat_integer_basis()[h_ordinal] {
            actual += evaluate_packed_query_cpu(
                &packed,
                D21CoefficientQuery {
                    diagram: u16::try_from(diagram_ordinal).unwrap(),
                    outer_left: outer_pair[0] as u8,
                    outer_right: outer_pair[1] as u8,
                    momentum: 0,
                    h_vector: (coordinate % VECTOR_DIMENSION) as u8,
                    output_axes: [axes[0], axes[1], axes[2], axes[3]],
                    input_spinor: (coordinate / VECTOR_DIMENSION) as u8,
                    output_spinor: (target / 330) as u8,
                    h_coefficient: i16::try_from(coefficient).unwrap(),
                    reserved: 0,
                },
            );
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn d21_inverse_sparse_emission_matches_dense_oracle_and_is_canonical() {
        assert_eq!(std::mem::size_of::<D21CompactRawCooEntry>(), 16);
        let selected = [16_u16, 61, 69, 159, 218, 238, 287];
        let entries = inverse_sparse_emission_oracle(0, &selected).unwrap();
        validate_compact_raw_coo(&entries).unwrap();
        let (outer_pair, momentum, h_hat) = decode_source_coordinate(0).unwrap();
        let diagrams = enumerate_diagrams();
        for diagram in selected {
            let dense =
                evaluate_diagram(&diagrams[usize::from(diagram)], outer_pair, momentum, h_hat);
            let inverse = entries
                .iter()
                .filter(|entry| entry.diagram_ordinal == diagram)
                .map(|entry| (usize::from(entry.target_coordinate), entry.coefficient))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(
                inverse, dense,
                "inverse emission mismatch for diagram {diagram}"
            );
        }
        eprintln!(
            "D21_INVERSE_SPARSE source=0 diagrams={} entries={} sha256={}",
            selected.len(),
            entries.len(),
            compact_raw_coo_sha256(&entries).unwrap(),
        );
    }

    #[test]
    fn d21_inverse_sparse_ordering_and_sector_replay_fail_closed() {
        assert!(inverse_sparse_emission_oracle(0, &[2, 1]).is_err());
        assert!(inverse_sparse_emission_oracle(0, &[1, 1]).is_err());
        assert!(inverse_sparse_emission_oracle(D21_SOURCE_COORDINATES as u32, &[1]).is_err());
        assert!(inverse_sparse_emission_oracle(0, &[400]).is_err());

        let selected = [16_u16, 61, 69, 159, 218, 238, 287];
        let entries = inverse_sparse_emission_oracle(0, &selected).unwrap();
        let mut reordered = entries.clone();
        if reordered.len() >= 2 {
            reordered.swap(0, 1);
            assert!(validate_compact_raw_coo(&reordered).is_err());
        }
        let mut zero = entries[0];
        zero.coefficient = 0;
        assert!(validate_compact_raw_coo(&[zero]).is_err());

        let mut witness = None;
        for diagram in selected {
            let compact = entries
                .iter()
                .filter(|entry| entry.diagram_ordinal == diagram)
                .map(|entry| (entry.target_coordinate, entry.coefficient))
                .collect::<Vec<_>>();
            if compact.is_empty() {
                continue;
            }
            for sector in ["00001", "00011", "00101", "01001", "10001"] {
                let projected = dg4_projector_numerator_oracle(sector, &compact).unwrap();
                if !projected.numerator.is_empty() {
                    witness = Some((diagram, sector, projected));
                    break;
                }
            }
            if witness.is_some() {
                break;
            }
        }
        let (diagram, sector, projected) =
            witness.expect("selected raw diagrams have a nonzero target projection");
        let &(target_coordinate, expected_numerator) = projected.numerator.first().unwrap();
        let replay = replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
            source_coordinate: 0,
            target_coordinate,
            diagram_ordinal: diagram,
            target_sector: sector.to_string(),
        })
        .unwrap();
        assert_eq!(replay.projected_numerator, expected_numerator);
        assert_eq!(replay.pass_nonzero_counts.len(), 4);
        assert!(replay.passed_nonzero);
        eprintln!("D21_SECTOR_PIVOT_REPLAY {replay:?}");

        assert!(
            replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                source_coordinate: 0,
                target_coordinate: u16::MAX,
                diagram_ordinal: diagram,
                target_sector: sector.to_string(),
            })
            .is_err()
        );
    }

    #[test]
    fn d21_gpu_excess_witnesses_replay_after_variance_fix() {
        let rows = [(56_320_u32, 14_u16), (991_077_u32, 211_u16)];
        let diagrams = [0_u16, 12_u16];
        let mut matrix = [[0_i128; 2]; 2];
        for (row_slot, (source_coordinate, target_coordinate)) in rows.into_iter().enumerate() {
            for (column_slot, diagram_ordinal) in diagrams.into_iter().enumerate() {
                let raw =
                    inverse_sparse_emission_oracle(source_coordinate, &[diagram_ordinal]).unwrap();
                let raw_coefficient = raw
                    .iter()
                    .find(|entry| entry.target_coordinate == target_coordinate)
                    .map(|entry| entry.coefficient)
                    .unwrap_or(0);
                let replay = replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                    source_coordinate,
                    target_coordinate,
                    diagram_ordinal,
                    target_sector: "01001".to_string(),
                })
                .unwrap();
                matrix[row_slot][column_slot] = replay.projected_numerator;
                eprintln!(
                    "D21_GPU_EXCESS_REPLAY source={source_coordinate} diagram={diagram_ordinal} target={target_coordinate} raw={raw_coefficient} numerator={} denominator={} pass_nnz={:?}",
                    replay.projected_numerator,
                    replay.projector_denominator,
                    replay.pass_nonzero_counts,
                );
            }
        }
        let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        eprintln!("D21_GPU_EXCESS_MINOR matrix={matrix:?} determinant={determinant}");
        assert_eq!(determinant, 0);
    }

    #[test]
    fn d21_scalar_boost_commutator_passes_and_variance_mutation_fails() {
        let diagrams = enumerate_diagrams();
        let canaries = [(0_usize, 0_usize, 0_usize), (12, 6, 37)];
        let mut mutation_residuals = 0;
        for (diagram, momentum, h_hat) in canaries {
            let corrected = scalar_diagram_boost_residual_entries(
                &diagrams[diagram],
                momentum,
                h_hat,
                0,
                1,
                evaluate_diagram,
            )
            .unwrap();
            let mutated = scalar_diagram_boost_residual_entries(
                &diagrams[diagram],
                momentum,
                h_hat,
                0,
                1,
                evaluate_diagram_h_output_delta_mutation,
            )
            .unwrap();
            eprintln!(
                "D21_SCALAR_BOOST diagram={diagram} p={momentum} h={h_hat} corrected={corrected} mutated={mutated}"
            );
            assert_eq!(corrected, 0);
            mutation_residuals += mutated;
        }
        assert!(mutation_residuals > 0);
    }

    #[test]
    fn d21_full_source_generator_matches_invariant_diagrams() {
        let diagrams = enumerate_diagrams();
        let source = 131_857_u32;
        for diagram in [0_usize, 21, 238] {
            for left in 0..VECTOR_DIMENSION {
                for right in (left + 1)..VECTOR_DIMENSION {
                    let residual = full_diagram_lorentz_residual_entries(
                        &diagrams[diagram],
                        source,
                        left,
                        right,
                    )
                    .unwrap();
                    assert_eq!(
                        residual, 0,
                        "diagram {diagram} failed generator ({left},{right})"
                    );
                }
            }
        }
    }
}
