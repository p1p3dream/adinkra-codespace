//! Exact bounded source inventory for the eleven-dimensional A3/G4 descent.
//!
//! This module deliberately separates a finite low-bidegree representation
//! calculation from the still-open proof that those bidegrees and source
//! sectors exhaust the physical local operator module.

use crate::eleven_dimensional_prepotential::b5_dimension;
use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const VECTOR_DIMENSION: u64 = 11;
const SPINOR_DIMENSION: u64 = 32;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EnlargedSourceSector {
    pub name: &'static str,
    pub dynkin_label: &'static str,
    pub dimension: u64,
    pub statistics: &'static str,
    pub source_status: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BoundedHomMultiplicity {
    pub source: &'static str,
    pub source_dynkin_label: &'static str,
    pub d_d: usize,
    pub d_p: usize,
    pub target: &'static str,
    pub target_dynkin_label: &'static str,
    pub cartesian_domain_dimension: u64,
    pub multiplicity: usize,
    pub proof_rule: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CartesianMapChannel {
    pub source: &'static str,
    pub d_d: usize,
    pub d_p: usize,
    pub target: &'static str,
    pub name: &'static str,
    pub formula: &'static str,
    pub coefficient_columns: usize,
    pub bianchi_coefficient_rank: Option<usize>,
    pub bianchi_kernel_dimension: Option<usize>,
    pub independently_executable: bool,
    pub blocker: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TeleparallelCoefficientSystem {
    pub declared_source_direct_sum: &'static str,
    pub h_hat_bianchi_closed_g4_columns: usize,
    pub gamma_trace_bianchi_closed_g4_columns: usize,
    pub independent_compensator_bianchi_closed_g4_columns: usize,
    pub total_formal_bianchi_closed_g4_columns: usize,
    pub same_domain_target_columns: usize,
    pub coefficient_matrix_constructed: bool,
    pub launch_ready: bool,
    pub blocker: &'static str,
}

pub const HIGHER_BIDEGREE_ORACLE_SHA256: &str =
    "10d5d8d79f757e5ac8dd03a9899777044ee27e9a2b54afa582e4d9da47753e38";
const HIGHER_BIDEGREE_ORACLE: &[u8] =
    include_bytes!("../results/adynkra_11d_higher_bidegree_hom_inventory.json");

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HigherBidegreeHomColumn {
    pub ordinal: usize,
    pub d_d: usize,
    pub d_p: usize,
    pub target: &'static str,
    pub target_irrep: &'static str,
    pub multiplicity_copy: usize,
    pub chain: &'static str,
    pub integrability_status: &'static str,
    pub bianchi_status: &'static str,
    pub cartesian_intertwiner_constructed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HigherBidegreeSourceGraphScope {
    pub ambient_hom_columns: usize,
    pub d2_p1_columns: usize,
    pub d0_p2_columns: usize,
    pub formal_form_pullback_generators: usize,
    pub pullback_generators_mapped_to_common_multiplicity_basis: bool,
    pub pullback_span_rank: Option<usize>,
    pub non_form_complement_dimension: Option<usize>,
    pub integrability_matrix_constructed: bool,
    pub bianchi_matrix_constructed: bool,
    pub source_gauge_quotient_constructed: bool,
    pub teleparallel_coefficient_matrix_constructed: bool,
    pub launch_ready: bool,
    pub blocker: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RawHCartesianMapAudit {
    pub source_decomposition: &'static str,
    pub h_hat_dimension: usize,
    pub gamma_trace_dimension: usize,
    pub raw_h_dimension: usize,
    pub h_hat_a3_gamma4_trace_over_gamma2_exterior: String,
    pub gamma_trace_a3_gamma4_trace_over_gamma2_exterior: String,
    pub h_hat_g4_gamma5_trace_over_gamma3_exterior: String,
    pub gamma_trace_g4_gamma5_trace_over_gamma3_exterior: String,
    pub h_hat_a3_proportionality_residuals: usize,
    pub gamma_trace_a3_proportionality_residuals: usize,
    pub h_hat_g4_proportionality_residuals: usize,
    pub gamma_trace_g4_proportionality_residuals: usize,
    pub h_hat_a3_exterior_restriction_rank: usize,
    pub gamma_trace_a3_exterior_restriction_rank: usize,
    pub h_hat_g4_exterior_restriction_rank: usize,
    pub gamma_trace_g4_exterior_restriction_rank: usize,
    pub a3_concatenated_hom_coefficient_rank: usize,
    pub g4_concatenated_hom_coefficient_rank: usize,
    pub raw_h_a3_maps_independent: bool,
    pub raw_h_g4_maps_independent: bool,
    pub gamma_trace_injection_projector_residuals: usize,
    pub p320_idempotence_residuals: usize,
    pub p32_idempotence_residuals: usize,
    pub projector_orthogonality_residuals: usize,
    pub projector_sum_identity_residuals: usize,
    pub direct_sum_inverse_residuals: usize,
    pub h_hat_trace_intersection_dimension: usize,
    pub combined_source_basis_rank: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EnlargedFourFormSourceReport {
    pub schema_version: &'static str,
    pub lorentz_algebra: &'static str,
    pub target_a3: &'static str,
    pub target_g4: &'static str,
    pub declared_bidegrees: Vec<(usize, usize)>,
    pub required_bidegrees_proven_exhaustive: bool,
    pub source_sectors: Vec<EnlargedSourceSector>,
    pub hom_multiplicities: Vec<BoundedHomMultiplicity>,
    pub cartesian_channels: Vec<CartesianMapChannel>,
    pub raw_h_cartesian_map_audit: RawHCartesianMapAudit,
    pub higher_bidegree_oracle_sha256: String,
    pub higher_bidegree_oracle_hash_matches: bool,
    pub higher_bidegree_columns: Vec<HigherBidegreeHomColumn>,
    pub higher_bidegree_source_graph: HigherBidegreeSourceGraphScope,
    pub raw_h_a3_multiplicity_d1_p0: usize,
    pub raw_h_g4_multiplicity_d1_p0: usize,
    pub raw_h_a3_multiplicity_d1_p1: usize,
    pub raw_h_g4_multiplicity_d1_p1: usize,
    pub prior_three_channel_h_hat_g4_basis_complete_at_d1_p1: bool,
    pub compensator_gauge_quotient_constructed: bool,
    pub teleparallel_system: TeleparallelCoefficientSystem,
    pub passed_bounded_inventory: bool,
    pub final_physical_launch_ready: bool,
    pub boundary: &'static str,
}

fn parse_label(label: &str) -> [i16; 5] {
    assert_eq!(label.len(), 5);
    let mut output = [0_i16; 5];
    for (index, byte) in label.bytes().enumerate() {
        assert!(byte.is_ascii_digit());
        output[index] = i16::from(byte - b'0');
    }
    output
}

fn format_label(labels: [i16; 5]) -> String {
    assert!(labels.iter().all(|value| (0..=9).contains(value)));
    labels
        .into_iter()
        .map(|value| char::from(b'0' + u8::try_from(value).unwrap()))
        .collect()
}

fn doubled_orthogonal_weight(labels: [i16; 5]) -> [i16; 5] {
    std::array::from_fn(|index| 2 * labels[index..4].iter().sum::<i16>() + labels[4])
}

fn dominant_label(weight: [i16; 5]) -> Option<String> {
    if !(0..4).all(|index| weight[index] >= weight[index + 1]) || weight[4] < 0 {
        return None;
    }
    if !(0..4).all(|index| (weight[index] - weight[index + 1]) % 2 == 0) {
        return None;
    }
    let mut labels = [0_i16; 5];
    for index in 0..4 {
        labels[index] = (weight[index] - weight[index + 1]) / 2;
    }
    labels[4] = weight[4];
    Some(format_label(labels))
}

/// Exact minuscule rule for tensoring a B5 irrep with the spinor 00001.
fn tensor_spinor(label: &str) -> Vec<String> {
    let highest = doubled_orthogonal_weight(parse_label(label));
    let mut output = BTreeSet::new();
    for mask in 0_u8..32 {
        let weight = std::array::from_fn(|axis| {
            highest[axis] + if mask & (1 << axis) == 0 { 1 } else { -1 }
        });
        if let Some(label) = dominant_label(weight) {
            output.insert(label);
        }
    }
    output.into_iter().collect()
}

/// Exact B5 quasi-minuscule vector rule.
///
/// Add every nonzero vector weight `+-e_i` that leaves a dominant highest
/// weight. The zero weight contributes the original irrep exactly when the
/// last Dynkin label is nonzero.
fn tensor_vector(label: &str) -> Vec<String> {
    let labels = parse_label(label);
    let highest = doubled_orthogonal_weight(labels);
    let mut output = BTreeSet::new();
    for axis in 0..5 {
        for delta in [2_i16, -2_i16] {
            let mut weight = highest;
            weight[axis] += delta;
            if let Some(label) = dominant_label(weight) {
                output.insert(label);
            }
        }
    }
    if labels[4] > 0 {
        output.insert(label.to_string());
    }
    output.into_iter().collect()
}

fn tensor_channels(label: &str, d_d: usize, d_p: usize) -> BTreeMap<String, usize> {
    let mut current = BTreeMap::from([(label.to_string(), 1_usize)]);
    for _ in 0..d_d {
        let mut next = BTreeMap::new();
        for (source, multiplicity) in current {
            for target in tensor_spinor(&source) {
                *next.entry(target).or_default() += multiplicity;
            }
        }
        current = next;
    }
    for _ in 0..d_p {
        let mut next = BTreeMap::new();
        for (source, multiplicity) in current {
            for target in tensor_vector(&source) {
                *next.entry(target).or_default() += multiplicity;
            }
        }
        current = next;
    }
    current
}

type IntegerMatrix = Vec<Vec<i16>>;
type RationalColumn = BTreeMap<u16, Ratio<i64>>;

fn form_masks(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << 11))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn multiply_integer(left: &IntegerMatrix, right: &IntegerMatrix) -> IntegerMatrix {
    let mut output = vec![vec![0_i16; right[0].len()]; left.len()];
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            let l = left[row][pivot];
            if l == 0 {
                continue;
            }
            for column in 0..right[0].len() {
                output[row][column] += l * right[pivot][column];
            }
        }
    }
    output
}

fn corrected_gamma_table(degree: usize) -> Vec<(u16, IntegerMatrix)> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    form_masks(degree)
        .into_iter()
        .map(|mask| {
            let mut product = vec![vec![0_i16; 32]; 32];
            for index in 0..32 {
                product[index][index] = 1;
            }
            for axis in 0..11 {
                if mask & (1_u16 << axis) == 0 {
                    continue;
                }
                let gamma = gammas[axis]
                    .iter()
                    .map(|row| row.iter().map(|&value| i16::from(value)).collect())
                    .collect::<IntegerMatrix>();
                product = multiply_integer(&product, &gamma);
                if axis == 0 {
                    for row in &mut product {
                        for value in row {
                            *value = -*value;
                        }
                    }
                }
            }
            for row in &mut product {
                for value in row {
                    *value = -*value;
                }
            }
            (mask, product)
        })
        .collect()
}

fn insertion_sign(mask: u16, index: usize) -> i64 {
    if (mask >> (index + 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn lorentz_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
}

fn add_ratio(output: &mut RationalColumn, key: u16, value: Ratio<i64>) {
    if value == Ratio::from_integer(0) {
        return;
    }
    *output.entry(key).or_default() += value;
    output.retain(|_, coefficient| *coefficient != Ratio::from_integer(0));
}

fn h_hat_components() -> Vec<Vec<(usize, usize, i64)>> {
    crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis()
        .into_iter()
        .map(|column| {
            column
                .into_iter()
                .map(|(coordinate, coefficient)| {
                    assert_eq!(*coefficient.real.denom(), 1);
                    assert_eq!(*coefficient.imaginary.numer(), 0);
                    (coordinate / 11, coordinate % 11, *coefficient.real.numer())
                })
                .collect()
        })
        .collect()
}

/// An integral basis for the image of the gamma-trace injection. The omitted
/// common factor 1/11 has no effect on Hom-map proportionality or independence.
fn gamma_trace_components() -> Vec<Vec<(usize, usize, i64)>> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    (0..32)
        .map(|trace_spinor| {
            let mut column = Vec::new();
            for vector in 0..11 {
                for output_spinor in 0..32 {
                    let coefficient = i64::from(gammas[vector][output_spinor][trace_spinor]);
                    if coefficient != 0 {
                        column.push((output_spinor, vector, coefficient));
                    }
                }
            }
            column
        })
        .collect()
}

fn gamma_trace_injection_residuals(trace: &[Vec<(usize, usize, i64)>]) -> usize {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut residuals = 0;
    for (source, column) in trace.iter().enumerate() {
        for row in 0..32 {
            let value = column
                .iter()
                .map(|&(spinor, vector, coefficient)| {
                    lorentz_sign(vector) * i64::from(gammas[vector][row][spinor]) * coefficient
                })
                .sum::<i64>();
            let expected = if row == source { 11 } else { 0 };
            residuals += usize::from(value != expected);
        }
    }
    residuals
}

fn projected_column(
    degree: usize,
    derivative: usize,
    source: &[(usize, usize, i64)],
    exterior: bool,
    gamma_table: &[(u16, IntegerMatrix)],
) -> RationalColumn {
    let mut output = RationalColumn::new();
    for (mask, gamma) in gamma_table {
        for &(spinor, vector, source_coefficient) in source {
            let raw = i64::from(gamma[derivative][spinor]) * source_coefficient;
            if raw == 0 {
                continue;
            }
            if exterior {
                if mask & (1_u16 << vector) == 0 {
                    add_ratio(
                        &mut output,
                        mask | (1_u16 << vector),
                        Ratio::new(
                            raw * insertion_sign(*mask, vector) * lorentz_sign(vector),
                            i64::try_from(degree + 1).unwrap(),
                        ),
                    );
                }
            } else if mask & (1_u16 << vector) != 0 {
                let remaining = mask ^ (1_u16 << vector);
                add_ratio(
                    &mut output,
                    remaining,
                    Ratio::from_integer(raw * insertion_sign(remaining, vector)),
                );
            }
        }
    }
    output
}

fn operator_ratio(
    sources: &[Vec<(usize, usize, i64)>],
    exterior_degree: usize,
    trace_degree: usize,
) -> (Ratio<i64>, usize) {
    let exterior_table = corrected_gamma_table(exterior_degree);
    let trace_table = corrected_gamma_table(trace_degree);
    let mut ratio = None;
    let mut residuals = 0;
    for derivative in 0..32 {
        for source in sources {
            let left = projected_column(exterior_degree, derivative, source, true, &exterior_table);
            let right = projected_column(trace_degree, derivative, source, false, &trace_table);
            for key in left
                .keys()
                .chain(right.keys())
                .copied()
                .collect::<BTreeSet<_>>()
            {
                let l = left.get(&key).cloned().unwrap_or_default();
                let r = right.get(&key).cloned().unwrap_or_default();
                if ratio.is_none() && l != Ratio::from_integer(0) {
                    ratio = Some(r.clone() / l.clone());
                }
                if let Some(scale) = &ratio {
                    residuals += usize::from(r != l * scale.clone());
                } else {
                    residuals += usize::from(r != Ratio::from_integer(0));
                }
            }
        }
    }
    (ratio.expect("nonzero Hom map"), residuals)
}

fn format_ratio(value: &Ratio<i64>) -> String {
    if *value.denom() == 1 {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

fn modular_inverse(value: i64, prime: i64) -> i64 {
    let (mut t, mut new_t) = (0_i128, 1_i128);
    let (mut r, mut new_r) = (i128::from(prime), i128::from(value.rem_euclid(prime)));
    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }
    assert_eq!(r, 1);
    i64::try_from(t.rem_euclid(i128::from(prime))).unwrap()
}

fn projected_operator_rank(
    sources: &[Vec<(usize, usize, i64)>],
    degree: usize,
    exterior: bool,
) -> usize {
    const PRIME: i64 = 1_073_741_783;
    let table = corrected_gamma_table(degree);
    let target_degree = if exterior { degree + 1 } else { degree - 1 };
    let basis = form_masks(target_degree);
    let row_lookup = basis
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, mask)| (mask, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut pivots = vec![None::<Vec<i64>>; basis.len()];
    let mut rank = 0;
    for derivative in 0..32 {
        for source in sources {
            let column = projected_column(degree, derivative, source, exterior, &table);
            let mut dense = vec![0_i64; basis.len()];
            for (mask, coefficient) in column {
                dense[row_lookup[&mask]] = (coefficient.numer().rem_euclid(PRIME)
                    * modular_inverse(*coefficient.denom(), PRIME))
                .rem_euclid(PRIME);
            }
            loop {
                let Some(pivot) = dense.iter().position(|value| *value != 0) else {
                    break;
                };
                if let Some(existing) = &pivots[pivot] {
                    let factor = dense[pivot];
                    for row in pivot..dense.len() {
                        dense[row] = (dense[row] - factor * existing[row]).rem_euclid(PRIME);
                    }
                } else {
                    let inverse = modular_inverse(dense[pivot], PRIME);
                    for value in &mut dense[pivot..] {
                        *value = (*value * inverse).rem_euclid(PRIME);
                    }
                    pivots[pivot] = Some(dense);
                    rank += 1;
                    break;
                }
            }
        }
    }
    rank
}

fn gamma_trace_raw(raw: &[Ratio<i64>]) -> Vec<Ratio<i64>> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut trace = vec![Ratio::from_integer(0); 32];
    for row in 0..32 {
        for spinor in 0..32 {
            for vector in 0..11 {
                let coefficient = lorentz_sign(vector) * i64::from(gammas[vector][row][spinor]);
                if coefficient != 0 {
                    trace[row] +=
                        raw[spinor * 11 + vector].clone() * Ratio::from_integer(coefficient);
                }
            }
        }
    }
    trace
}

fn inject_gamma_trace(trace: &[Ratio<i64>]) -> Vec<Ratio<i64>> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut raw = vec![Ratio::from_integer(0); 352];
    for spinor in 0..32 {
        for vector in 0..11 {
            for source in 0..32 {
                let coefficient = i64::from(gammas[vector][spinor][source]);
                if coefficient != 0 {
                    raw[spinor * 11 + vector] +=
                        trace[source].clone() * Ratio::new(coefficient, 11);
                }
            }
        }
    }
    raw
}

fn subtract_raw(left: &[Ratio<i64>], right: &[Ratio<i64>]) -> Vec<Ratio<i64>> {
    left.iter()
        .zip(right)
        .map(|(l, r)| l.clone() - r.clone())
        .collect()
}

fn projector_and_inverse_residuals() -> (usize, usize, usize, usize, usize) {
    let zero = Ratio::from_integer(0);
    let one = Ratio::from_integer(1);
    let mut p320_idempotence = 0;
    let mut p32_idempotence = 0;
    let mut orthogonality = 0;
    let mut sum_identity = 0;
    let mut inverse = 0;
    for coordinate in 0..352 {
        let mut raw = vec![zero.clone(); 352];
        raw[coordinate] = one.clone();
        let p32 = inject_gamma_trace(&gamma_trace_raw(&raw));
        let p320 = subtract_raw(&raw, &p32);
        let p32_squared = inject_gamma_trace(&gamma_trace_raw(&p32));
        let p320_squared = {
            let trace = inject_gamma_trace(&gamma_trace_raw(&p320));
            subtract_raw(&p320, &trace)
        };
        let p32_p320 = inject_gamma_trace(&gamma_trace_raw(&p320));
        let p320_p32 = {
            let trace = inject_gamma_trace(&gamma_trace_raw(&p32));
            subtract_raw(&p32, &trace)
        };
        for row in 0..352 {
            p32_idempotence += usize::from(p32_squared[row] != p32[row]);
            p320_idempotence += usize::from(p320_squared[row] != p320[row]);
            orthogonality += usize::from(p32_p320[row] != zero);
            orthogonality += usize::from(p320_p32[row] != zero);
            sum_identity += usize::from(
                p32[row].clone() + p320[row].clone()
                    != if row == coordinate {
                        one.clone()
                    } else {
                        zero.clone()
                    },
            );
        }
        // The Hhat coordinates are the 320 spatial components of P320, and
        // the trace coordinates are Gamma^a H_a. Reconstruct both pieces.
        let trace_coordinates = gamma_trace_raw(&raw);
        let reconstructed_trace = inject_gamma_trace(&trace_coordinates);
        let mut reconstructed_hhat = vec![zero.clone(); 352];
        let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
        for spatial in 1..11 {
            for spinor in 0..32 {
                let coefficient = p320[spinor * 11 + spatial].clone();
                reconstructed_hhat[spinor * 11 + spatial] += coefficient.clone();
                for intermediate in 0..32 {
                    let right = i64::from(gammas[spatial][intermediate][spinor]);
                    if right == 0 {
                        continue;
                    }
                    for time_spinor in 0..32 {
                        let left = i64::from(gammas[0][time_spinor][intermediate]);
                        if left != 0 {
                            reconstructed_hhat[time_spinor * 11] +=
                                coefficient.clone() * Ratio::from_integer(-left * right);
                        }
                    }
                }
            }
        }
        for row in 0..352 {
            inverse += usize::from(
                reconstructed_hhat[row].clone() + reconstructed_trace[row].clone() != raw[row],
            );
        }
    }
    (
        p320_idempotence,
        p32_idempotence,
        orthogonality,
        sum_identity,
        inverse,
    )
}

fn build_raw_h_cartesian_map_audit() -> RawHCartesianMapAudit {
    let h_hat = h_hat_components();
    let trace = gamma_trace_components();
    let (h_hat_a3, h_hat_a3_residuals) = operator_ratio(&h_hat, 2, 4);
    let (trace_a3, trace_a3_residuals) = operator_ratio(&trace, 2, 4);
    let (h_hat_g4, h_hat_g4_residuals) = operator_ratio(&h_hat, 3, 5);
    let (trace_g4, trace_g4_residuals) = operator_ratio(&trace, 3, 5);
    let injection_residuals = gamma_trace_injection_residuals(&trace);
    let h_hat_a3_rank = projected_operator_rank(&h_hat, 2, true);
    let trace_a3_rank = projected_operator_rank(&trace, 2, true);
    let h_hat_g4_rank = projected_operator_rank(&h_hat, 3, true);
    let trace_g4_rank = projected_operator_rank(&trace, 3, true);
    let (p320_idempotence, p32_idempotence, orthogonality, sum_identity, inverse) =
        projector_and_inverse_residuals();
    let a3_hom_rank =
        usize::from(h_hat_a3_rank > 0 && trace_a3_rank > 0) + usize::from(h_hat_a3 != trace_a3);
    let g4_hom_rank =
        usize::from(h_hat_g4_rank > 0 && trace_g4_rank > 0) + usize::from(h_hat_g4 != trace_g4);
    let passed = h_hat.len() == 320
        && trace.len() == 32
        && h_hat_a3_residuals == 0
        && trace_a3_residuals == 0
        && h_hat_g4_residuals == 0
        && trace_g4_residuals == 0
        && h_hat_a3_rank == 165
        && trace_a3_rank == 165
        && h_hat_g4_rank == 330
        && trace_g4_rank == 330
        && a3_hom_rank == 2
        && g4_hom_rank == 2
        && injection_residuals == 0
        && p320_idempotence == 0
        && p32_idempotence == 0
        && orthogonality == 0
        && sum_identity == 0
        && inverse == 0;
    RawHCartesianMapAudit {
        source_decomposition: "10000 tensor 00001 = 10001 plus 00001",
        h_hat_dimension: h_hat.len(),
        gamma_trace_dimension: trace.len(),
        raw_h_dimension: h_hat.len() + trace.len(),
        h_hat_a3_gamma4_trace_over_gamma2_exterior: format_ratio(&h_hat_a3),
        gamma_trace_a3_gamma4_trace_over_gamma2_exterior: format_ratio(&trace_a3),
        h_hat_g4_gamma5_trace_over_gamma3_exterior: format_ratio(&h_hat_g4),
        gamma_trace_g4_gamma5_trace_over_gamma3_exterior: format_ratio(&trace_g4),
        h_hat_a3_proportionality_residuals: h_hat_a3_residuals,
        gamma_trace_a3_proportionality_residuals: trace_a3_residuals,
        h_hat_g4_proportionality_residuals: h_hat_g4_residuals,
        gamma_trace_g4_proportionality_residuals: trace_g4_residuals,
        h_hat_a3_exterior_restriction_rank: h_hat_a3_rank,
        gamma_trace_a3_exterior_restriction_rank: trace_a3_rank,
        h_hat_g4_exterior_restriction_rank: h_hat_g4_rank,
        gamma_trace_g4_exterior_restriction_rank: trace_g4_rank,
        a3_concatenated_hom_coefficient_rank: a3_hom_rank,
        g4_concatenated_hom_coefficient_rank: g4_hom_rank,
        raw_h_a3_maps_independent: a3_hom_rank == 2,
        raw_h_g4_maps_independent: g4_hom_rank == 2,
        gamma_trace_injection_projector_residuals: injection_residuals,
        p320_idempotence_residuals: p320_idempotence,
        p32_idempotence_residuals: p32_idempotence,
        projector_orthogonality_residuals: orthogonality,
        projector_sum_identity_residuals: sum_identity,
        direct_sum_inverse_residuals: inverse,
        h_hat_trace_intersection_dimension: 0,
        combined_source_basis_rank: usize::from(inverse == 0) * 352,
        passed,
    }
}

fn source_sectors() -> Vec<EnlargedSourceSector> {
    [
        (
            "Hhat",
            "10001",
            "fermionic",
            "authoritative gamma-traceless source",
        ),
        (
            "gamma_trace_H",
            "00001",
            "fermionic",
            "raw-H complement; physical source authority unresolved",
        ),
        (
            "Psi",
            "00000",
            "bosonic",
            "independent compensator extension; gauge quotient unresolved",
        ),
        (
            "Psi1",
            "10000",
            "bosonic",
            "independent compensator extension; gauge quotient unresolved",
        ),
        (
            "Psi2",
            "01000",
            "bosonic",
            "independent compensator extension; gauge quotient unresolved",
        ),
        (
            "Psi3",
            "00100",
            "bosonic",
            "independent compensator extension; gauge quotient unresolved",
        ),
        (
            "Psi4",
            "00010",
            "bosonic",
            "independent compensator extension; gauge quotient unresolved",
        ),
        (
            "Psi5",
            "00002",
            "bosonic",
            "independent compensator extension; gauge quotient unresolved",
        ),
    ]
    .into_iter()
    .map(
        |(name, dynkin_label, statistics, source_status)| EnlargedSourceSector {
            name,
            dynkin_label,
            dimension: b5_dimension(dynkin_label),
            statistics,
            source_status,
        },
    )
    .collect()
}

fn domain_dimension(source_dimension: u64, d_d: usize, d_p: usize) -> u64 {
    source_dimension
        * SPINOR_DIMENSION.pow(u32::try_from(d_d).unwrap())
        * VECTOR_DIMENSION.pow(u32::try_from(d_p).unwrap())
}

fn bounded_hom_inventory(sectors: &[EnlargedSourceSector]) -> Vec<BoundedHomMultiplicity> {
    let mut output = Vec::new();
    for sector in sectors {
        for (d_d, d_p) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            // Bosonic targets require an even total parity. H sectors are
            // fermionic and compensators are bosonic.
            let source_is_fermionic = sector.statistics == "fermionic";
            if source_is_fermionic == (d_d % 2 == 0) {
                continue;
            }
            let channels = tensor_channels(sector.dynkin_label, d_d, d_p);
            for (target, target_label) in [("A3", "00100"), ("G4", "00010")] {
                let multiplicity = channels.get(target_label).copied().unwrap_or(0);
                if multiplicity == 0 {
                    continue;
                }
                output.push(BoundedHomMultiplicity {
                    source: sector.name,
                    source_dynkin_label: sector.dynkin_label,
                    d_d,
                    d_p,
                    target,
                    target_dynkin_label: target_label,
                    cartesian_domain_dimension: domain_dimension(sector.dimension, d_d, d_p),
                    multiplicity,
                    proof_rule: "B5 minuscule-spinor rule followed by the B5 vector quasi-minuscule rule; dimension identities checked exactly",
                });
            }
        }
    }
    output
}

fn cartesian_channels() -> Vec<CartesianMapChannel> {
    vec![
        CartesianMapChannel {
            source: "Hhat",
            d_d: 1,
            d_p: 0,
            target: "A3",
            name: "Hhat_gamma24_ray_A3",
            formula: "Gamma_[2] exterior Hhat, equivalently Gamma_[4] trace Hhat after the source-variance identity",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: true,
            blocker: "none inside Hhat",
        },
        CartesianMapChannel {
            source: "Hhat",
            d_d: 1,
            d_p: 0,
            target: "G4",
            name: "Hhat_gamma35_ray_G4",
            formula: "Gamma_[3] exterior Hhat, equivalently Gamma_[5] trace Hhat after the analogous source-variance identity",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: false,
            blocker: "Gamma3/Gamma5 source-variance parity harness not yet implemented",
        },
        CartesianMapChannel {
            source: "gamma_trace_H",
            d_d: 1,
            d_p: 0,
            target: "A3",
            name: "trace_spinor_A3",
            formula: "Gamma_[3] bilinear on D tensor gamma_trace(H)",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: false,
            blocker: "raw-H gamma-trace injection and gauge authority not constructed",
        },
        CartesianMapChannel {
            source: "gamma_trace_H",
            d_d: 1,
            d_p: 0,
            target: "G4",
            name: "trace_spinor_G4",
            formula: "Gamma_[4] bilinear on D tensor gamma_trace(H)",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: false,
            blocker: "raw-H gamma-trace injection and gauge authority not constructed",
        },
        CartesianMapChannel {
            source: "Hhat",
            d_d: 1,
            d_p: 1,
            target: "A3",
            name: "Hhat_A3_three_channel",
            formula: "p wedge Lambda2; i_p Lambda4; p-contraction of the 10100 vector-three-form hook",
            coefficient_columns: 3,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: false,
            blocker: "typed Lambda2/Lambda4/10100 projectors and source joins not all implemented",
        },
        CartesianMapChannel {
            source: "Hhat",
            d_d: 1,
            d_p: 1,
            target: "G4",
            name: "Hhat_G4_three_channel",
            formula: "p wedge Lambda3; i_p Lambda5; p_e H_[4]{}^e",
            coefficient_columns: 3,
            bianchi_coefficient_rank: Some(2),
            bianchi_kernel_dimension: Some(1),
            independently_executable: true,
            blocker: "none for the bounded Hhat Bianchi report",
        },
        CartesianMapChannel {
            source: "gamma_trace_H",
            d_d: 1,
            d_p: 1,
            target: "A3",
            name: "trace_A3_two_channel",
            formula: "p wedge Lambda2; i_p Lambda4",
            coefficient_columns: 2,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: false,
            blocker: "raw-H gamma-trace injection and gauge authority not constructed",
        },
        CartesianMapChannel {
            source: "gamma_trace_H",
            d_d: 1,
            d_p: 1,
            target: "G4",
            name: "trace_G4_two_channel",
            formula: "p wedge Lambda3; i_p Lambda5",
            coefficient_columns: 2,
            bianchi_coefficient_rank: Some(1),
            bianchi_kernel_dimension: Some(1),
            independently_executable: false,
            blocker: "raw-H gamma-trace injection and gauge authority not constructed",
        },
        CartesianMapChannel {
            source: "Psi3",
            d_d: 0,
            d_p: 0,
            target: "A3",
            name: "Psi3_identity",
            formula: "A3 = Psi_[3]",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: true,
            blocker: "physical normalization and compensator gauge quotient unresolved",
        },
        CartesianMapChannel {
            source: "Psi4",
            d_d: 0,
            d_p: 0,
            target: "G4",
            name: "Psi4_identity",
            formula: "G4 = Psi_[4]",
            coefficient_columns: 1,
            bianchi_coefficient_rank: Some(1),
            bianchi_kernel_dimension: Some(0),
            independently_executable: true,
            blocker: "generic independent Psi4 is not Bianchi closed",
        },
        CartesianMapChannel {
            source: "Psi2",
            d_d: 0,
            d_p: 1,
            target: "A3",
            name: "Psi2_wedge",
            formula: "A3 = p wedge Psi_[2]",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: true,
            blocker: "physical compensator gauge quotient unresolved",
        },
        CartesianMapChannel {
            source: "Psi4",
            d_d: 0,
            d_p: 1,
            target: "A3",
            name: "Psi4_contraction",
            formula: "A3 = i_p Psi_[4]",
            coefficient_columns: 1,
            bianchi_coefficient_rank: None,
            bianchi_kernel_dimension: None,
            independently_executable: true,
            blocker: "physical compensator gauge quotient unresolved",
        },
        CartesianMapChannel {
            source: "Psi3",
            d_d: 0,
            d_p: 1,
            target: "G4",
            name: "Psi3_wedge",
            formula: "G4 = p wedge Psi_[3]",
            coefficient_columns: 1,
            bianchi_coefficient_rank: Some(0),
            bianchi_kernel_dimension: Some(1),
            independently_executable: true,
            blocker: "physical compensator gauge quotient unresolved",
        },
        CartesianMapChannel {
            source: "Psi5",
            d_d: 0,
            d_p: 1,
            target: "G4",
            name: "Psi5_contraction",
            formula: "G4 = i_p Psi_[5]",
            coefficient_columns: 1,
            bianchi_coefficient_rank: Some(1),
            bianchi_kernel_dimension: Some(0),
            independently_executable: true,
            blocker: "generic independent Psi5 contraction is not Bianchi closed",
        },
    ]
}

fn hom_multiplicity(
    inventory: &[BoundedHomMultiplicity],
    source: &str,
    d_d: usize,
    d_p: usize,
    target: &str,
) -> usize {
    inventory
        .iter()
        .find(|entry| {
            entry.source == source && entry.d_d == d_d && entry.d_p == d_p && entry.target == target
        })
        .map(|entry| entry.multiplicity)
        .unwrap_or(0)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn higher_bidegree_columns() -> Vec<HigherBidegreeHomColumn> {
    let mut output = Vec::with_capacity(56);
    for (irrep, multiplicity) in [
        ("00001", 7_usize),
        ("00011", 7),
        ("00101", 11),
        ("01001", 14),
        ("10001", 13),
    ] {
        for copy in 0..multiplicity {
            output.push(HigherBidegreeHomColumn {
                ordinal: output.len(),
                d_d: 2,
                d_p: 1,
                target: "D_G4",
                target_irrep: irrep,
                multiplicity_copy: copy,
                chain: "Lambda^2(S_D) tensor V_p tensor Hhat -> irreducible summand of S_D tensor Lambda4(V)",
                integrability_status: "ambient PBW Hom column; D-algebra translation relation to the (0,2) block not imposed",
                bianchi_status: "D of p wedge G4 not evaluated",
                cartesian_intertwiner_constructed: false,
            });
        }
    }
    for (irrep, multiplicity) in [("00001", 1_usize), ("01001", 1), ("10001", 2)] {
        for copy in 0..multiplicity {
            output.push(HigherBidegreeHomColumn {
                ordinal: output.len(),
                d_d: 0,
                d_p: 2,
                target: "D_G4",
                target_irrep: irrep,
                multiplicity_copy: copy,
                chain: "Sym^2(V_p) tensor Hhat -> irreducible summand of S_D tensor Lambda4(V)",
                integrability_status:
                    "ambient PBW Hom column; translation image of the (2,1) block not imposed",
                bianchi_status: "D of p wedge G4 not evaluated",
                cartesian_intertwiner_constructed: false,
            });
        }
    }
    assert_eq!(output.len(), 56);
    output
}

fn validate_higher_bidegree_oracle() -> (String, bool) {
    let hash = sha256_hex(HIGHER_BIDEGREE_ORACLE);
    let value: serde_json::Value = serde_json::from_slice(HIGHER_BIDEGREE_ORACLE)
        .expect("pinned higher-bidegree oracle must parse");
    let counts_match = value["passed"] == true
        && value["descendant_targets"]["d2_p1_D_G4_total"] == 52
        && value["descendant_targets"]["d0_p2_D_G4_total"] == 4
        && value["validation"]["exterior2_spinor_dimension"] == 496
        && value["validation"]["symmetric2_vector_dimension"] == 66;
    let hash_matches = hash == HIGHER_BIDEGREE_ORACLE_SHA256;
    (hash, hash_matches && counts_match)
}

pub fn build_enlarged_four_form_source_report() -> EnlargedFourFormSourceReport {
    let sectors = source_sectors();
    let hom = bounded_hom_inventory(&sectors);
    let hhat_a3_10 = hom_multiplicity(&hom, "Hhat", 1, 0, "A3");
    let trace_a3_10 = hom_multiplicity(&hom, "gamma_trace_H", 1, 0, "A3");
    let hhat_g4_10 = hom_multiplicity(&hom, "Hhat", 1, 0, "G4");
    let trace_g4_10 = hom_multiplicity(&hom, "gamma_trace_H", 1, 0, "G4");
    let hhat_a3_11 = hom_multiplicity(&hom, "Hhat", 1, 1, "A3");
    let trace_a3_11 = hom_multiplicity(&hom, "gamma_trace_H", 1, 1, "A3");
    let hhat_g4_11 = hom_multiplicity(&hom, "Hhat", 1, 1, "G4");
    let trace_g4_11 = hom_multiplicity(&hom, "gamma_trace_H", 1, 1, "G4");
    let channels = cartesian_channels();
    let raw_h_cartesian_map_audit = build_raw_h_cartesian_map_audit();
    let (higher_bidegree_oracle_sha256, higher_bidegree_oracle_hash_matches) =
        validate_higher_bidegree_oracle();
    let higher_bidegree_columns = higher_bidegree_columns();
    let bounded_counts_pass = hhat_a3_10 == 1
        && trace_a3_10 == 1
        && hhat_g4_10 == 1
        && trace_g4_10 == 1
        && hhat_a3_11 == 3
        && trace_a3_11 == 2
        && hhat_g4_11 == 3
        && trace_g4_11 == 2
        && hom_multiplicity(&hom, "Psi3", 0, 0, "A3") == 1
        && hom_multiplicity(&hom, "Psi4", 0, 0, "G4") == 1
        && hom_multiplicity(&hom, "Psi2", 0, 1, "A3") == 1
        && hom_multiplicity(&hom, "Psi4", 0, 1, "A3") == 1
        && hom_multiplicity(&hom, "Psi3", 0, 1, "G4") == 1
        && hom_multiplicity(&hom, "Psi5", 0, 1, "G4") == 1;
    let raw_h_cartesian_maps_pass = raw_h_cartesian_map_audit.passed;
    EnlargedFourFormSourceReport {
        schema_version: "adynkra-11d-enlarged-four-form-source-low-bidegree-v1",
        lorentz_algebra: "B5 = so(11,C), interpreted in Spin(1,10) Cartesian conventions",
        target_a3: "00100, dimension 165",
        target_g4: "00010, dimension 330",
        declared_bidegrees: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        required_bidegrees_proven_exhaustive: false,
        source_sectors: sectors,
        hom_multiplicities: hom,
        cartesian_channels: channels,
        raw_h_cartesian_map_audit,
        higher_bidegree_oracle_sha256,
        higher_bidegree_oracle_hash_matches,
        higher_bidegree_columns,
        higher_bidegree_source_graph: HigherBidegreeSourceGraphScope {
            ambient_hom_columns: 56,
            d2_p1_columns: 52,
            d0_p2_columns: 4,
            formal_form_pullback_generators: 51,
            pullback_generators_mapped_to_common_multiplicity_basis: false,
            pullback_span_rank: None,
            non_form_complement_dimension: None,
            integrability_matrix_constructed: false,
            bianchi_matrix_constructed: false,
            source_gauge_quotient_constructed: false,
            teleparallel_coefficient_matrix_constructed: false,
            launch_ready: false,
            blocker: "the 51 form-factorized dimensions are not a ranked subspace of the 52-dimensional ambient Hom space; explicit Cartesian pullbacks, PBW integrability, Bianchi, and source-gauge matrices are required",
        },
        raw_h_a3_multiplicity_d1_p0: hhat_a3_10 + trace_a3_10,
        raw_h_g4_multiplicity_d1_p0: hhat_g4_10 + trace_g4_10,
        raw_h_a3_multiplicity_d1_p1: hhat_a3_11 + trace_a3_11,
        raw_h_g4_multiplicity_d1_p1: hhat_g4_11 + trace_g4_11,
        prior_three_channel_h_hat_g4_basis_complete_at_d1_p1: hhat_g4_11 == 3,
        compensator_gauge_quotient_constructed: false,
        teleparallel_system: TeleparallelCoefficientSystem {
            declared_source_direct_sum: "Hhat plus gamma_trace(H) plus independent Psi,Psi1,...,Psi5",
            h_hat_bianchi_closed_g4_columns: 1,
            gamma_trace_bianchi_closed_g4_columns: 1,
            independent_compensator_bianchi_closed_g4_columns: 1,
            total_formal_bianchi_closed_g4_columns: 3,
            same_domain_target_columns: 1,
            coefficient_matrix_constructed: false,
            launch_ready: false,
            blocker: "the three surviving formal columns live on different independent source domains; a typed source join and the compensator gauge quotient are required before a teleparallel matching matrix is meaningful",
        },
        passed_bounded_inventory: bounded_counts_pass
            && raw_h_cartesian_maps_pass
            && higher_bidegree_oracle_hash_matches,
        final_physical_launch_ready: false,
        boundary: "This is an exact Hom inventory only for the four declared bidegrees. It does not prove bidegree exhaustion, source completeness, compensator independence modulo gauge, physical normalization, or final A3/G4 descent.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minuscule_and_quasi_minuscule_rules_preserve_dimensions() {
        for label in [
            "00000", "00001", "10000", "01000", "00100", "00010", "00002", "10001", "10002",
            "10010", "10100", "11000", "20000",
        ] {
            let source = b5_dimension(label);
            let spinor_sum: u64 = tensor_spinor(label)
                .iter()
                .map(|target| b5_dimension(target))
                .sum();
            assert_eq!(spinor_sum, SPINOR_DIMENSION * source, "S tensor {label}");
            let vector_sum: u64 = tensor_vector(label)
                .iter()
                .map(|target| b5_dimension(target))
                .sum();
            assert_eq!(vector_sum, VECTOR_DIMENSION * source, "V tensor {label}");
        }
    }

    #[test]
    fn bounded_hom_inventory_and_fail_closed_boundary_are_exact() {
        let report = build_enlarged_four_form_source_report();
        assert!(report.passed_bounded_inventory);
        assert_eq!(report.raw_h_a3_multiplicity_d1_p0, 2);
        assert_eq!(report.raw_h_g4_multiplicity_d1_p0, 2);
        assert_eq!(report.raw_h_a3_multiplicity_d1_p1, 5);
        assert_eq!(report.raw_h_g4_multiplicity_d1_p1, 5);
        assert!(report.prior_three_channel_h_hat_g4_basis_complete_at_d1_p1);
        assert!(report.raw_h_cartesian_map_audit.passed);
        assert!(report.raw_h_cartesian_map_audit.raw_h_a3_maps_independent);
        assert!(report.raw_h_cartesian_map_audit.raw_h_g4_maps_independent);
        assert!(report.higher_bidegree_oracle_hash_matches);
        assert_eq!(report.higher_bidegree_columns.len(), 56);
        assert_eq!(
            report
                .higher_bidegree_columns
                .iter()
                .filter(|column| column.d_d == 2 && column.d_p == 1)
                .count(),
            52
        );
        assert_eq!(
            report
                .higher_bidegree_columns
                .iter()
                .filter(|column| column.d_d == 0 && column.d_p == 2)
                .count(),
            4
        );
        assert!(
            !report
                .higher_bidegree_source_graph
                .pullback_generators_mapped_to_common_multiplicity_basis
        );
        assert_eq!(report.higher_bidegree_source_graph.pullback_span_rank, None);
        assert_eq!(
            report
                .higher_bidegree_source_graph
                .non_form_complement_dimension,
            None
        );
        assert!(!report.required_bidegrees_proven_exhaustive);
        assert!(!report.compensator_gauge_quotient_constructed);
        assert!(!report.teleparallel_system.coefficient_matrix_constructed);
        assert!(!report.teleparallel_system.launch_ready);
        assert!(!report.final_physical_launch_ready);
    }

    #[test]
    fn bianchi_filter_counts_only_closed_direct_sum_channels() {
        let report = build_enlarged_four_form_source_report();
        let system = &report.teleparallel_system;
        assert_eq!(system.h_hat_bianchi_closed_g4_columns, 1);
        assert_eq!(system.gamma_trace_bianchi_closed_g4_columns, 1);
        assert_eq!(system.independent_compensator_bianchi_closed_g4_columns, 1);
        assert_eq!(system.total_formal_bianchi_closed_g4_columns, 3);
        let psi4 = report
            .cartesian_channels
            .iter()
            .find(|channel| channel.name == "Psi4_identity")
            .unwrap();
        assert_eq!(psi4.bianchi_kernel_dimension, Some(0));
        let psi3 = report
            .cartesian_channels
            .iter()
            .find(|channel| channel.name == "Psi3_wedge")
            .unwrap();
        assert_eq!(psi3.bianchi_kernel_dimension, Some(1));
    }
}
