//! Exact source-domain constraint inventory for the corrected four-form no-go.
//!
//! This module separates three different operations that must not be conflated:
//!
//! * the algebraic gamma-trace quotient defining `H_hat`;
//! * PBW normal form for its `DH` and `DDH` descendants;
//! * the still-missing physical target-gauge module `K`.
//!
//! The first two are executable source constructions. Neither is target gauge
//! descent. The corrected seven-row augmented witness is replayed after the
//! gamma-trace restriction to determine whether an already-proved source
//! constraint removes it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::{BigRational, Ratio};
use num_traits::Zero;
use serde::Serialize;

use crate::eleven_dimensional_corrected_full_chain_oracle::{
    FullChainRowKey, corrected_full_chain_streams,
};
use crate::eleven_dimensional_d21_invariant_diagrams::{
    D21SectorPivotReplayRequestV2, decode_source_coordinate, flattened_gamma_mask_tables,
    replay_sector_pivot_v2,
};
use crate::eleven_dimensional_dg4_casimir_projectors::project_dg4_target;
use crate::eleven_dimensional_four_form_56_physics_rows::lexicographic_four_form_to_numeric;
use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
use crate::eleven_dimensional_physical_curvature::ExactQi;

pub const SCHEMA_VERSION: &str = "adynkra-11d-four-form-source-constraint-quotient-v1";
const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const RAW_H_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const H_HAT_DIMENSION: usize = (VECTOR_DIMENSION - 1) * SPINOR_DIMENSION;
const DH_HAT_DIMENSION: usize = SPINOR_DIMENSION * H_HAT_DIMENSION;
const DDH_HAT_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION * H_HAT_DIMENSION;
const DH_GAMMA_TRACE_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION;
const DDH_GAMMA_TRACE_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION;
const PBW_D2_H_HAT_DIMENSION: usize =
    (SPINOR_DIMENSION * (SPINOR_DIMENSION - 1) / 2 + VECTOR_DIMENSION) * H_HAT_DIMENSION;
const DG4_TARGET_DIMENSION: u64 = 10_560;

const WITNESS_PIVOT_ROWS: [u64; 6] = [
    594_739_214,
    594_739_530,
    1_392_410_462,
    1_392_410_476,
    2_784_820_051,
    2_784_820_068,
];
const WITNESS_CROSS_ROW: u64 = 1_392_410_608;
const WITNESS_DIAGRAMS: [u16; 6] = [21, 79, 33, 45, 47, 57];

type BigQi = Complex<BigRational>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceConstraintQuotientReport {
    pub schema_version: &'static str,
    pub raw_h_dimension: usize,
    pub gamma_trace_rank: usize,
    pub gamma_trace_kernel_dimension: usize,
    pub p320_rank: usize,
    pub p32_rank: usize,
    pub trace_lift_identity_residual_entries: usize,
    pub h_hat_trace_residual_entries: usize,
    pub p320_h_hat_section_residual_entries: usize,
    pub p320_gamma_trace_kernel_residual_entries: usize,
    pub p320_mutation_detected: bool,
    pub dh_ambient_dimension: usize,
    pub dh_restricted_rank: usize,
    pub dh_gamma_trace_kernel_dimension: usize,
    pub ddh_ambient_dimension: usize,
    pub ddh_restricted_rank: usize,
    pub ddh_gamma_trace_kernel_dimension: usize,
    pub pbw_d2_h_hat_dimension: usize,
    pub pbw_constant_anticommutator_residual_pairs: usize,
    pub pbw_degree_one_overlap_residual_triples: usize,
    pub witness_source_h_hat_ordinals: Vec<usize>,
    pub witness_rows: Vec<u64>,
    pub witness_rows_survive_p320: bool,
    pub witness_candidate_rank_after_source_restriction: usize,
    pub witness_augmented_rank_after_source_restriction: usize,
    pub witness_candidate_determinant_real: String,
    pub witness_candidate_determinant_denominator: String,
    pub witness_first_residual_real: String,
    pub witness_first_residual_imaginary: String,
    pub existing_source_restrictions_remove_witness: bool,
    pub physical_target_k_constructed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn raw_h_index(spinor: usize, vector: usize) -> usize {
    spinor * VECTOR_DIMENSION + vector
}

fn gamma_trace_and_lift() -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let gamma = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut trace = vec![vec![0_i64; RAW_H_DIMENSION]; SPINOR_DIMENSION];
    let mut lift = vec![vec![0_i64; SPINOR_DIMENSION]; RAW_H_DIMENSION];
    for vector in 0..VECTOR_DIMENSION {
        let metric = if vector == 0 { -1 } else { 1 };
        for output in 0..SPINOR_DIMENSION {
            for input in 0..SPINOR_DIMENSION {
                trace[output][raw_h_index(input, vector)] =
                    metric * i64::from(gamma[vector][output][input]);
                lift[raw_h_index(output, vector)][input] = i64::from(gamma[vector][output][input]);
            }
        }
    }
    (trace, lift)
}

fn trace_lift_residuals(trace: &[Vec<i64>], lift: &[Vec<i64>]) -> usize {
    (0..SPINOR_DIMENSION)
        .flat_map(|row| {
            (0..SPINOR_DIMENSION).map(move |column| {
                let value = (0..RAW_H_DIMENSION)
                    .map(|middle| trace[row][middle] * lift[middle][column])
                    .sum::<i64>();
                usize::from(
                    value
                        != if row == column {
                            VECTOR_DIMENSION as i64
                        } else {
                            0
                        },
                )
            })
        })
        .sum()
}

fn trace_sparse(trace: &[Vec<i64>], vector: &BTreeMap<usize, ExactQi>) -> Vec<ExactQi> {
    (0..SPINOR_DIMENSION)
        .map(|row| {
            let mut value = ExactQi::zero();
            for (&coordinate, coefficient) in vector {
                let integer = trace[row][coordinate];
                if integer != 0 {
                    value.add_assign(&coefficient.scaled(&Ratio::from_integer(integer)));
                }
            }
            value
        })
        .collect()
}

fn p320_sparse(
    trace: &[Vec<i64>],
    lift: &[Vec<i64>],
    vector: &BTreeMap<usize, ExactQi>,
    denominator: i64,
) -> BTreeMap<usize, ExactQi> {
    let trace_value = trace_sparse(trace, vector);
    let mut output = vector.clone();
    for coordinate in 0..RAW_H_DIMENSION {
        let mut correction = ExactQi::zero();
        for spinor in 0..SPINOR_DIMENSION {
            let integer = lift[coordinate][spinor];
            if integer != 0 {
                correction
                    .add_assign(&trace_value[spinor].scaled(&Ratio::new(integer, denominator)));
            }
        }
        if !correction.is_zero() {
            let entry = output.entry(coordinate).or_insert_with(ExactQi::zero);
            entry.add_assign(&correction.scaled(&Ratio::from_integer(-1)));
            if entry.is_zero() {
                output.remove(&coordinate);
            }
        }
    }
    output
}

fn p320_identities() -> (usize, usize, usize, bool) {
    let (trace, lift) = gamma_trace_and_lift();
    let basis = canonical_gamma_traceless_frame_basis();
    let h_hat_trace_residuals = basis
        .iter()
        .map(|column| {
            trace_sparse(&trace, column)
                .into_iter()
                .filter(|x| !x.is_zero())
                .count()
        })
        .sum();
    let p320_section_residuals = basis
        .iter()
        .filter(|column| p320_sparse(&trace, &lift, column, 11) != **column)
        .count();
    let mut lift_kernel_residuals = 0;
    let mut mutation_detected = false;
    for spinor in 0..SPINOR_DIMENSION {
        let column = (0..RAW_H_DIMENSION)
            .filter_map(|coordinate| {
                let value = lift[coordinate][spinor];
                (value != 0).then_some((coordinate, ExactQi::from_integer(value)))
            })
            .collect::<BTreeMap<_, _>>();
        lift_kernel_residuals += p320_sparse(&trace, &lift, &column, 11).len();
        mutation_detected |= !p320_sparse(&trace, &lift, &column, 10).is_empty();
    }
    (
        h_hat_trace_residuals,
        p320_section_residuals,
        lift_kernel_residuals,
        mutation_detected,
    )
}

fn big_ratio(value: &Ratio<i64>) -> BigRational {
    BigRational::new(BigInt::from(*value.numer()), BigInt::from(*value.denom()))
}

fn big_qi(value: &ExactQi) -> BigQi {
    Complex::new(big_ratio(&value.real), big_ratio(&value.imaginary))
}

fn pair_from_mask(mask: u32) -> Result<[usize; 2], String> {
    if mask.count_ones() != 2 {
        return Err("witness exterior spinor mask is not degree two".to_string());
    }
    let axes = (0..SPINOR_DIMENSION)
        .filter(|axis| mask & (1_u32 << axis) != 0)
        .collect::<Vec<_>>();
    Ok([axes[0], axes[1]])
}

fn one_momentum_axis(exponents: &[u16; VECTOR_DIMENSION]) -> Option<usize> {
    (exponents.iter().copied().sum::<u16>() == 1)
        .then(|| exponents.iter().position(|&value| value == 1).unwrap())
}

fn target_to_numeric(target: usize) -> Result<usize, String> {
    if target >= DG4_TARGET_DIMENSION as usize {
        return Err("teleparallel target coordinate is out of range".to_string());
    }
    Ok((target / 330) * 330 + lexicographic_four_form_to_numeric(target % 330)?)
}

fn source_fierz_target_slice(
    teleparallel: &BTreeMap<FullChainRowKey, ExactQi>,
    source_coordinate: u32,
) -> Result<BTreeMap<usize, BigQi>, String> {
    let (query_pair, momentum, _) = decode_source_coordinate(source_coordinate)?;
    let (_, charge_gamma) = flattened_gamma_mask_tables();
    let table = |mask: usize, left: usize, right: usize| -> i64 {
        i64::from(
            charge_gamma
                [mask * SPINOR_DIMENSION * SPINOR_DIMENSION + left * SPINOR_DIMENSION + right],
        )
    };
    let masks = (0_usize..(1_usize << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 3)
        .filter_map(|mask| {
            let value = table(mask, query_pair[0], query_pair[1]);
            (value != 0).then_some((mask, value))
        })
        .collect::<Vec<_>>();
    if masks.iter().any(|&(mask, _)| {
        (0..SPINOR_DIMENSION)
            .flat_map(|left| ((left + 1)..SPINOR_DIMENSION).map(move |right| (left, right)))
            .map(|(left, right)| {
                let value = table(mask, left, right);
                value * value
            })
            .sum::<i64>()
            != 16
    }) {
        return Err("outer-degree-three Fierz norm is not 16".to_string());
    }
    let mut output = BTreeMap::<usize, BigQi>::new();
    for (key, value) in teleparallel {
        if one_momentum_axis(&key.momentum_exponents) != Some(momentum) {
            continue;
        }
        let pair = pair_from_mask(key.exterior_spinor_mask)?;
        let numerator = masks
            .iter()
            .map(|&(mask, query)| query * table(mask, pair[0], pair[1]))
            .sum::<i64>();
        if numerator == 0 {
            continue;
        }
        let coordinate = target_to_numeric(key.output_coordinate)?;
        let entry = output
            .entry(coordinate)
            .or_insert_with(|| Complex::new(BigRational::zero(), BigRational::zero()));
        *entry += big_qi(value) * BigRational::new(BigInt::from(numerator), BigInt::from(16));
        if entry.re.is_zero() && entry.im.is_zero() {
            output.remove(&coordinate);
        }
    }
    Ok(output)
}

fn target_sector_slice(input: &BTreeMap<usize, BigQi>) -> Result<BTreeMap<usize, BigQi>, String> {
    let to_small = |value: &BigRational| -> Result<Ratio<i64>, String> {
        Ok(Ratio::new(
            i64::try_from(value.numer().clone())
                .map_err(|_| "projector numerator exceeds i64".to_string())?,
            i64::try_from(value.denom().clone())
                .map_err(|_| "projector denominator exceeds i64".to_string())?,
        ))
    };
    let real = input
        .iter()
        .filter(|(_, value)| !value.re.is_zero())
        .map(|(&row, value)| Ok((row, to_small(&value.re)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let imaginary = input
        .iter()
        .filter(|(_, value)| !value.im.is_zero())
        .map(|(&row, value)| Ok((row, to_small(&value.im)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let real = project_dg4_target("01001", &real)?;
    let imaginary = project_dg4_target("01001", &imaginary)?;
    let rows = real
        .keys()
        .chain(imaginary.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    Ok(rows
        .into_iter()
        .map(|row| {
            (
                row,
                Complex::new(
                    real.get(&row)
                        .map(big_ratio)
                        .unwrap_or_else(BigRational::zero),
                    imaginary
                        .get(&row)
                        .map(big_ratio)
                        .unwrap_or_else(BigRational::zero),
                ),
            )
        })
        .collect())
}

fn solve_real(
    mut matrix: Vec<Vec<BigRational>>,
    mut right: Vec<BigQi>,
) -> Result<Vec<BigQi>, String> {
    let n = matrix.len();
    if n == 0 || right.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err("witness pivot system is not square".to_string());
    }
    for column in 0..n {
        let pivot = (column..n)
            .find(|&row| !matrix[row][column].is_zero())
            .ok_or_else(|| format!("witness pivot system is singular at column {column}"))?;
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        let scale = matrix[column][column].clone();
        for value in &mut matrix[column][column..] {
            *value /= scale.clone();
        }
        right[column] /= scale;
        let pivot_row = matrix[column].clone();
        let pivot_right = right[column].clone();
        for row in 0..n {
            if row == column || matrix[row][column].is_zero() {
                continue;
            }
            let factor = matrix[row][column].clone();
            for next in column..n {
                matrix[row][next] -= factor.clone() * pivot_row[next].clone();
            }
            right[row] -= pivot_right.clone() * factor;
        }
    }
    Ok(right)
}

fn determinant(mut matrix: Vec<Vec<BigRational>>) -> Result<BigRational, String> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err("determinant input is not square".to_string());
    }
    let mut value = BigRational::from_integer(BigInt::from(1));
    for column in 0..n {
        let pivot = (column..n)
            .find(|&row| !matrix[row][column].is_zero())
            .ok_or_else(|| "candidate witness determinant vanished".to_string())?;
        if pivot != column {
            matrix.swap(column, pivot);
            value = -value;
        }
        let diagonal = matrix[column][column].clone();
        value *= diagonal.clone();
        for row in (column + 1)..n {
            if matrix[row][column].is_zero() {
                continue;
            }
            let factor = matrix[row][column].clone() / diagonal.clone();
            for next in column..n {
                let correction = factor.clone() * matrix[column][next].clone();
                matrix[row][next] -= correction;
            }
        }
    }
    Ok(value)
}

fn restricted_witness() -> Result<(Vec<usize>, BigRational, BigQi), String> {
    let mut rows = WITNESS_PIVOT_ROWS.to_vec();
    rows.push(WITNESS_CROSS_ROW);
    let mut source_ordinals = BTreeSet::new();
    let mut target_streams = BTreeMap::new();
    for &row in &rows {
        let source = u32::try_from(row / DG4_TARGET_DIMENSION)
            .map_err(|_| "witness source exceeds u32".to_string())?;
        let (_, _, h) = decode_source_coordinate(source)?;
        source_ordinals.insert(h);
        if let std::collections::btree_map::Entry::Vacant(entry) = target_streams.entry(h) {
            let (_, target) = corrected_full_chain_streams(h)?;
            entry.insert(target);
        }
    }
    let candidate = rows
        .iter()
        .map(|&row| {
            let source_coordinate = u32::try_from(row / DG4_TARGET_DIMENSION).unwrap();
            let target_coordinate = u16::try_from(row % DG4_TARGET_DIMENSION).unwrap();
            WITNESS_DIAGRAMS
                .iter()
                .map(|&diagram_ordinal| {
                    let replay = replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                        source_coordinate,
                        target_coordinate,
                        diagram_ordinal,
                        target_sector: "01001".to_string(),
                    })?;
                    Ok(BigRational::new(
                        BigInt::from(replay.projected_numerator),
                        BigInt::from(replay.projector_denominator),
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut source_cache = BTreeMap::new();
    let mut target_cache = BTreeMap::new();
    let mut right = Vec::with_capacity(rows.len());
    for &row in &rows {
        let source = u32::try_from(row / DG4_TARGET_DIMENSION).unwrap();
        let target = usize::try_from(row % DG4_TARGET_DIMENSION).unwrap();
        let (_, _, h) = decode_source_coordinate(source)?;
        if let std::collections::btree_map::Entry::Vacant(entry) = source_cache.entry(source) {
            entry.insert(source_fierz_target_slice(&target_streams[&h], source)?);
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = target_cache.entry(source) {
            entry.insert(target_sector_slice(&source_cache[&source])?);
        }
        right.push(
            target_cache[&source]
                .get(&target)
                .cloned()
                .unwrap_or_else(|| Complex::new(BigRational::zero(), BigRational::zero())),
        );
    }
    let pivot_matrix = candidate[..6].to_vec();
    let det = determinant(pivot_matrix.clone())?;
    let solution = solve_real(pivot_matrix, right[..6].to_vec())?;
    let actual = candidate[6].iter().zip(solution).fold(
        Complex::new(BigRational::zero(), BigRational::zero()),
        |sum, (entry, coefficient)| sum + coefficient * entry.clone(),
    );
    let residual = actual - right[6].clone();
    Ok((source_ordinals.into_iter().collect(), det, residual))
}

fn rational_string(value: &BigRational) -> String {
    if value.denom() == &BigInt::from(1) {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

fn build_report() -> Result<SourceConstraintQuotientReport, String> {
    let (trace, lift) = gamma_trace_and_lift();
    let trace_lift_identity_residual_entries = trace_lift_residuals(&trace, &lift);
    let (
        h_hat_trace_residual_entries,
        p320_h_hat_section_residual_entries,
        p320_gamma_trace_kernel_residual_entries,
        p320_mutation_detected,
    ) = p320_identities();
    let normal_form = crate::eleven_dimensional_superderivative_normal_form::verify();
    let (witness_source_h_hat_ordinals, determinant, witness_residual) = restricted_witness()?;
    let witness_rows_survive_p320 = witness_source_h_hat_ordinals.iter().all(|&ordinal| {
        let basis = canonical_gamma_traceless_frame_basis();
        ordinal < basis.len() && p320_sparse(&trace, &lift, &basis[ordinal], 11) == basis[ordinal]
    });
    let witness_augmented_rank =
        usize::from(!witness_residual.re.is_zero() || !witness_residual.im.is_zero()) + 6;
    let passed = trace_lift_identity_residual_entries == 0
        && h_hat_trace_residual_entries == 0
        && p320_h_hat_section_residual_entries == 0
        && p320_gamma_trace_kernel_residual_entries == 0
        && p320_mutation_detected
        && normal_form.constant_anticommutator_residual_pairs == 0
        && normal_form.degree_one_overlap_residual_triples == 0
        && witness_rows_survive_p320
        && !determinant.is_zero()
        && witness_augmented_rank == 7;
    Ok(SourceConstraintQuotientReport {
        schema_version: SCHEMA_VERSION,
        raw_h_dimension: RAW_H_DIMENSION,
        gamma_trace_rank: SPINOR_DIMENSION,
        gamma_trace_kernel_dimension: H_HAT_DIMENSION,
        p320_rank: H_HAT_DIMENSION,
        p32_rank: SPINOR_DIMENSION,
        trace_lift_identity_residual_entries,
        h_hat_trace_residual_entries,
        p320_h_hat_section_residual_entries,
        p320_gamma_trace_kernel_residual_entries,
        p320_mutation_detected,
        dh_ambient_dimension: SPINOR_DIMENSION * RAW_H_DIMENSION,
        dh_restricted_rank: DH_HAT_DIMENSION,
        dh_gamma_trace_kernel_dimension: DH_GAMMA_TRACE_DIMENSION,
        ddh_ambient_dimension: SPINOR_DIMENSION * SPINOR_DIMENSION * RAW_H_DIMENSION,
        ddh_restricted_rank: DDH_HAT_DIMENSION,
        ddh_gamma_trace_kernel_dimension: DDH_GAMMA_TRACE_DIMENSION,
        pbw_d2_h_hat_dimension: PBW_D2_H_HAT_DIMENSION,
        pbw_constant_anticommutator_residual_pairs: normal_form
            .constant_anticommutator_residual_pairs,
        pbw_degree_one_overlap_residual_triples: normal_form.degree_one_overlap_residual_triples,
        witness_source_h_hat_ordinals,
        witness_rows: WITNESS_PIVOT_ROWS
            .into_iter()
            .chain([WITNESS_CROSS_ROW])
            .collect(),
        witness_rows_survive_p320,
        witness_candidate_rank_after_source_restriction: 6,
        witness_augmented_rank_after_source_restriction: witness_augmented_rank,
        witness_candidate_determinant_real: determinant.numer().to_string(),
        witness_candidate_determinant_denominator: determinant.denom().to_string(),
        witness_first_residual_real: rational_string(&witness_residual.re),
        witness_first_residual_imaginary: rational_string(&witness_residual.im),
        existing_source_restrictions_remove_witness: false,
        physical_target_k_constructed: false,
        passed,
        boundary: "The exact P320 gamma-trace quotient and PBW normal form are source-domain constructions. The corrected seven-row outer-degree-three 01001 augmented witness lies entirely in Hhat and remains rank seven after those restrictions. Eq. (40) and the conventional connection solves define or eliminate compensators and do not impose an additional kernel on Hhat. This result is not physical target-gauge K, does not prove source-constraint completeness, and does not prove irreducibility.",
    })
}

pub fn verify() -> Result<SourceConstraintQuotientReport, String> {
    static REPORT: OnceLock<Result<SourceConstraintQuotientReport, String>> = OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_p320_kernel_image_and_tensor_lifts_close() {
        let report = verify().unwrap();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.gamma_trace_rank, 32);
        assert_eq!(report.gamma_trace_kernel_dimension, 320);
        assert_eq!(report.dh_restricted_rank, 10_240);
        assert_eq!(report.ddh_restricted_rank, 327_680);
        assert_eq!(report.pbw_d2_h_hat_dimension, 162_240);
    }

    #[test]
    fn corrected_seven_row_witness_survives_existing_source_constraints() {
        let report = verify().unwrap();
        assert_eq!(report.witness_source_h_hat_ordinals, vec![0, 17, 34]);
        assert!(report.witness_rows_survive_p320);
        assert_eq!(report.witness_candidate_rank_after_source_restriction, 6);
        assert_eq!(report.witness_augmented_rank_after_source_restriction, 7);
        assert_eq!(report.witness_first_residual_real, "-21707/184320");
        assert_eq!(report.witness_first_residual_imaginary, "7/9216");
        assert!(!report.existing_source_restrictions_remove_witness);
        assert!(!report.physical_target_k_constructed);
    }

    #[test]
    fn denominator_mutation_is_rejected() {
        assert!(verify().unwrap().p320_mutation_detected);
    }
}
