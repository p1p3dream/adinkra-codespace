//! Exact Lorentz-equivariance canary for the corrected teleparallel D21 map.
//!
//! The source action is the twice-generator action on
//! `Lambda^2 S tensor V* tensor Hhat`. The target action is the matching
//! twice-generator on `S tensor Lambda^4 V`. A map is equivariant exactly when
//! `rho_target(X) T(e_s) = T(rho_source(X)e_s)` for every checked source basis
//! vector and all 55 Lorentz generators.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use num_rational::Ratio;
use rayon::prelude::*;
use serde::Serialize;

use crate::eleven_dimensional_corrected_full_chain_oracle::{
    CorrectedFullChainStageStreams, FullChainRowKey, corrected_full_chain_stage_streams,
    corrected_full_chain_streams,
};
use crate::eleven_dimensional_d21_invariant_diagrams::{
    d21_source_lorentz_generator_terms, decode_source_coordinate,
};
use crate::eleven_dimensional_dg4_casimir_projectors::dg4_lorentz_generator_action_integer;
use crate::eleven_dimensional_four_form_56_physics_rows::lexicographic_four_form_to_numeric;
use crate::eleven_dimensional_physical_curvature::ExactQi;
use crate::eleven_dimensional_physical_curvature::{
    Eq25FermionicFrameInput, apply_eq25_fermionic_frame,
    cached_linearized_gravitino_curl_to_d_f_four_operator,
    inject_d_lorentz_compensator_into_d_delta,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-corrected-teleparallel-d21-equivariance-v1";
const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const TARGET_DIMENSION: usize = SPINOR_DIMENSION * 330;
const FIRST_WITNESS_ROW: u64 = 1_392_410_608;
const WITNESS_ROWS: [u64; 7] = [
    594_739_214,
    594_739_530,
    1_392_410_462,
    1_392_410_476,
    2_784_820_051,
    2_784_820_068,
    FIRST_WITNESS_ROW,
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExactQiPublic {
    pub real_numerator: i64,
    pub real_denominator: i64,
    pub imaginary_numerator: i64,
    pub imaginary_denominator: i64,
}

impl From<&ExactQi> for ExactQiPublic {
    fn from(value: &ExactQi) -> Self {
        Self {
            real_numerator: *value.real.numer(),
            real_denominator: *value.real.denom(),
            imaginary_numerator: *value.imaginary.numer(),
            imaginary_denominator: *value.imaginary.denom(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EquivarianceResidualWitness {
    pub source_coordinate: u32,
    pub source_pair: [usize; 2],
    pub source_momentum: usize,
    pub h_hat_ordinal: usize,
    pub generator_left: usize,
    pub generator_right: usize,
    pub target_coordinate: usize,
    pub target_action_value: ExactQiPublic,
    pub source_action_value: ExactQiPublic,
    pub residual: ExactQiPublic,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedTeleparallelEquivarianceReport {
    pub schema_version: &'static str,
    pub source_representation: &'static str,
    pub target_representation: &'static str,
    pub normalization: &'static str,
    pub first_witness_row: u64,
    pub first_witness_source_coordinate: u32,
    pub distinct_witness_source_coordinates: Vec<u32>,
    pub generators_checked_per_source: usize,
    pub source_columns_checked: usize,
    pub commutators_checked: usize,
    pub residual_entries: usize,
    pub first_residual: Option<EquivarianceResidualWitness>,
    pub output_charge_adapted_residual_entries: usize,
    pub output_charge_adapted_first_residual: Option<EquivarianceResidualWitness>,
    pub output_charge_adapter_restores_equivariance: bool,
    pub witness_source_canary_equivariant: bool,
    pub exhaustive_all_source_columns_complete: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn scale(value: &ExactQi, factor: i64) -> ExactQi {
    value.scaled(&Ratio::from_integer(factor))
}

fn add(output: &mut BTreeMap<usize, ExactQi>, row: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(row).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&row);
    }
}

fn subtract_maps(
    left: &BTreeMap<usize, ExactQi>,
    right: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = left.clone();
    for (&row, value) in right {
        add(&mut output, row, scale(value, -1));
    }
    output
}

fn multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn divide(left: &ExactQi, right: &ExactQi) -> Result<ExactQi, String> {
    if right.is_zero() {
        return Err("division by zero in local-Lorentz target-image reducer".to_string());
    }
    let norm =
        right.real.clone() * right.real.clone() + right.imaginary.clone() * right.imaginary.clone();
    Ok(ExactQi {
        real: (left.real.clone() * right.real.clone()
            + left.imaginary.clone() * right.imaginary.clone())
            / norm.clone(),
        imaginary: (left.imaginary.clone() * right.real.clone()
            - left.real.clone() * right.imaginary.clone())
            / norm,
    })
}

fn subtract_multiple(
    output: &mut BTreeMap<usize, ExactQi>,
    input: &BTreeMap<usize, ExactQi>,
    factor: &ExactQi,
) {
    for (&row, value) in input {
        add(output, row, scale(&multiply(value, factor), -1));
    }
}

#[derive(Clone, Default)]
struct ExactSparseImageReducer {
    pivots: BTreeMap<usize, BTreeMap<usize, ExactQi>>,
    pivot_original_columns: BTreeMap<usize, usize>,
    pivot_combinations: BTreeMap<usize, BTreeMap<usize, ExactQi>>,
}

impl ExactSparseImageReducer {
    fn reduce(
        &self,
        mut vector: BTreeMap<usize, ExactQi>,
    ) -> Result<BTreeMap<usize, ExactQi>, String> {
        loop {
            let Some((&row, value)) = vector.first_key_value() else {
                return Ok(vector);
            };
            let Some(pivot) = self.pivots.get(&row) else {
                return Ok(vector);
            };
            let factor = divide(value, pivot.get(&row).unwrap())?;
            subtract_multiple(&mut vector, pivot, &factor);
        }
    }

    fn insert_with_origin(
        &mut self,
        vector: BTreeMap<usize, ExactQi>,
        original_column: usize,
    ) -> Result<bool, String> {
        let mut reduced = vector;
        let mut combination = BTreeMap::from([(original_column, ExactQi::one())]);
        loop {
            let Some((&row, value)) = reduced.first_key_value() else {
                return Ok(false);
            };
            let Some(pivot) = self.pivots.get(&row) else {
                break;
            };
            let factor = divide(value, pivot.get(&row).unwrap())?;
            subtract_multiple(&mut reduced, pivot, &factor);
            subtract_multiple(
                &mut combination,
                self.pivot_combinations.get(&row).unwrap(),
                &factor,
            );
        }
        let Some((&row, value)) = reduced.first_key_value() else {
            return Ok(false);
        };
        let inverse = divide(&ExactQi::one(), value)?;
        let normalized = reduced
            .into_iter()
            .map(|(coordinate, coefficient)| (coordinate, multiply(&coefficient, &inverse)))
            .collect();
        let normalized_combination = combination
            .into_iter()
            .map(|(coordinate, coefficient)| (coordinate, multiply(&coefficient, &inverse)))
            .collect();
        self.pivots.insert(row, normalized);
        self.pivot_original_columns.insert(row, original_column);
        self.pivot_combinations.insert(row, normalized_combination);
        Ok(true)
    }

    fn solve_coordinates(
        &self,
        mut vector: BTreeMap<usize, ExactQi>,
    ) -> Result<(BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>), String> {
        let mut coordinates = BTreeMap::new();
        loop {
            let Some((&row, value)) = vector.first_key_value() else {
                return Ok((coordinates, vector));
            };
            let Some(pivot) = self.pivots.get(&row) else {
                return Ok((coordinates, vector));
            };
            let factor = divide(value, pivot.get(&row).unwrap())?;
            subtract_multiple(&mut vector, pivot, &factor);
            for (&column, coefficient) in self.pivot_combinations.get(&row).unwrap() {
                add(&mut coordinates, column, multiply(coefficient, &factor));
            }
        }
    }
}

fn local_lorentz_target_image_reducer(
    momentum_axis: usize,
) -> Result<ExactSparseImageReducer, String> {
    let mut reducer = ExactSparseImageReducer::default();
    for column in 0..SPINOR_DIMENSION * 55 {
        reducer.insert_with_origin(
            local_lorentz_target_response_column(column, momentum_axis)?,
            column,
        )?;
    }
    Ok(reducer)
}

fn cached_local_lorentz_target_image_reducer(
    momentum_axis: usize,
) -> Result<&'static ExactSparseImageReducer, String> {
    static REDUCERS: [OnceLock<Result<ExactSparseImageReducer, String>>; VECTOR_DIMENSION] =
        [const { OnceLock::new() }; VECTOR_DIMENSION];
    if momentum_axis >= VECTOR_DIMENSION {
        return Err("local-Lorentz target-image momentum is outside 0..11".to_string());
    }
    match REDUCERS[momentum_axis].get_or_init(|| local_lorentz_target_image_reducer(momentum_axis))
    {
        Ok(reducer) => Ok(reducer),
        Err(error) => Err(error.clone()),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LocalLorentzTargetImageColumn {
    pub original_d_psi_two_coordinate: usize,
    pub entries: BTreeMap<usize, ExactQi>,
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalPsi2ImageBasis {
    pub momentum_axis: usize,
    pub exact_rank: usize,
    pub pivot_rows: Vec<usize>,
    pub independent_original_columns: Vec<LocalLorentzTargetImageColumn>,
}

/// Canonical rank-320 basis of the vertical `D Psi_[2] -> D G4` image at one
/// formal momentum axis, without constructing any teleparallel commutator.
/// This is the small handoff used by quotient RREF and coboundary kernels.
pub(crate) fn canonical_psi2_image_basis(
    momentum_axis: usize,
) -> Result<CanonicalPsi2ImageBasis, String> {
    let reducer = cached_local_lorentz_target_image_reducer(momentum_axis)?;
    let original_ordinals = reducer
        .pivot_original_columns
        .values()
        .copied()
        .collect::<Vec<_>>();
    let pivot_rows = reducer
        .pivot_original_columns
        .keys()
        .copied()
        .collect::<Vec<_>>();
    let independent_original_columns = original_ordinals
        .into_iter()
        .map(|original_d_psi_two_coordinate| {
            Ok(LocalLorentzTargetImageColumn {
                original_d_psi_two_coordinate,
                entries: local_lorentz_target_response_column(
                    original_d_psi_two_coordinate,
                    momentum_axis,
                )?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(CanonicalPsi2ImageBasis {
        momentum_axis,
        exact_rank: reducer.pivots.len(),
        pivot_rows,
        independent_original_columns,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct LocalLorentzCommutatorColumn {
    pub generator_left: usize,
    pub generator_right: usize,
    pub entries: BTreeMap<usize, ExactQi>,
    pub image_coordinates: BTreeMap<usize, ExactQi>,
    pub exact_image_residual_entries: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LocalLorentzTargetImageHandoff {
    pub source_coordinate: u32,
    pub momentum_axis: usize,
    pub ambient_d_psi_two_columns: usize,
    pub exact_image_rank: usize,
    pub pivot_rows: Vec<usize>,
    pub independent_original_columns: Vec<LocalLorentzTargetImageColumn>,
    pub raw_commutators: Vec<LocalLorentzCommutatorColumn>,
}

/// Canonical exact handoff for a device quotient/cocycle solver. Independent
/// source columns are selected by ascending original `D_alpha Psi_[2]`
/// ordinal under ascending target-row sparse elimination. The exported maps
/// are the unnormalized original columns, not the normalized echelon pivots.
pub(crate) fn local_lorentz_target_image_handoff(
    source_coordinate: u32,
) -> Result<LocalLorentzTargetImageHandoff, String> {
    let (_, momentum_axis, _) = decode_source_coordinate(source_coordinate)?;
    let reducer = cached_local_lorentz_target_image_reducer(momentum_axis)?;
    let original_ordinals = reducer
        .pivot_original_columns
        .values()
        .copied()
        .collect::<Vec<_>>();
    let pivot_rows = reducer
        .pivot_original_columns
        .keys()
        .copied()
        .collect::<Vec<_>>();
    let mut independent_original_columns = Vec::with_capacity(original_ordinals.len());
    for original_d_psi_two_coordinate in original_ordinals {
        independent_original_columns.push(LocalLorentzTargetImageColumn {
            original_d_psi_two_coordinate,
            entries: local_lorentz_target_response_column(
                original_d_psi_two_coordinate,
                momentum_axis,
            )?,
        });
    }
    let mut cache = TeleparallelD21Cache::default();
    let mut raw_commutators = Vec::with_capacity(55);
    for generator_left in 0..VECTOR_DIMENSION {
        for generator_right in (generator_left + 1)..VECTOR_DIMENSION {
            let entries = commutator_residual(
                &mut cache,
                source_coordinate,
                generator_left,
                generator_right,
            )?;
            let (image_coordinates, image_residual) = reducer.solve_coordinates(entries.clone())?;
            raw_commutators.push(LocalLorentzCommutatorColumn {
                generator_left,
                generator_right,
                entries,
                image_coordinates,
                exact_image_residual_entries: image_residual.len(),
            });
        }
    }
    Ok(LocalLorentzTargetImageHandoff {
        source_coordinate,
        momentum_axis,
        ambient_d_psi_two_columns: SPINOR_DIMENSION * 55,
        exact_image_rank: reducer.pivots.len(),
        pivot_rows,
        independent_original_columns,
        raw_commutators,
    })
}

fn public_coefficient(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    }
}

/// Exact raw target-gauge response of one `D_alpha Psi_[2]` basis direction
/// after Eq. (25), the Rarita-Schwinger curl, and the teleparallel `D G_4`
/// operator. The returned four-form coordinate uses the D21 numeric-mask
/// convention. This is the candidate quotient image, not a claim that an
/// arbitrary fitted direction is the section cocycle.
fn local_lorentz_target_response_column(
    d_psi_two_coordinate: usize,
    momentum_axis: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    if d_psi_two_coordinate >= SPINOR_DIMENSION * 55 {
        return Err("D Psi_[2] coordinate is outside 32*55".to_string());
    }
    if momentum_axis >= VECTOR_DIMENSION {
        return Err("local-Lorentz response momentum axis is outside 0..11".to_string());
    }
    let frame = local_lorentz_eq25_frame_column(d_psi_two_coordinate)?;
    frame_to_target_response(&frame, momentum_axis)
}

fn local_lorentz_eq25_frame_column(
    d_psi_two_coordinate: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    if d_psi_two_coordinate >= SPINOR_DIMENSION * 55 {
        return Err("D Psi_[2] coordinate is outside 32*55".to_string());
    }
    let d_delta = inject_d_lorentz_compensator_into_d_delta(&BTreeMap::from([(
        d_psi_two_coordinate,
        ExactQi::one(),
    )]));
    apply_eq25_fermionic_frame(&Eq25FermionicFrameInput {
        d_delta,
        d_scale: BTreeMap::new(),
    })
}

/// Paper-derived horizontal Eq. (25) correction on one stored independent
/// lower two-form coordinate. The displayed Einstein contraction
/// `D Psi_de Gamma^{de}` contains both ordered `(d,e)` and `(e,d)` terms, so
/// it is twice the repository's increasing-pair coordinate. Consequently
/// `Q=-(i/64) Gamma_a D Psi_de Gamma^{de}` is exactly the negative of the
/// raw `(i/32) Gamma_a DDelta` response in this stored basis.
fn horizontal_local_lorentz_eq25_frame_column(
    d_psi_two_coordinate: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    let raw = local_lorentz_eq25_frame_column(d_psi_two_coordinate)?;
    let correction = explicit_horizontal_q_column(d_psi_two_coordinate, 2, -1, 64, true, true)?;
    let mut output = raw.clone();
    for (row, value) in correction {
        add(&mut output, row, value);
    }
    Ok(output)
}

fn multiply_integer_i8(left: &[Vec<i16>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            let factor = left[row][pivot];
            if factor == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += factor * i16::from(right[pivot][column]);
            }
        }
    }
    output
}

fn multiply_integer(left: &[Vec<i16>], right: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            let factor = left[row][pivot];
            if factor == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += factor * right[pivot][column];
            }
        }
    }
    output
}

/// Independent Cartesian implementation of
/// `Q_a^gamma=-(i/64)(gamma_a)^{beta delta}
/// (D_beta Psi_de)(gamma^{de})_delta{}^gamma`.
///
/// No Eq. (25), `DDelta`, or Lorentz-injection helper is called here. The
/// configurable arguments are used only by mutation gates.
fn explicit_horizontal_q_column(
    d_psi_two_coordinate: usize,
    ordered_pair_factor: i64,
    overall_sign: i64,
    denominator: i64,
    lower_output_vector: bool,
    raised_spinor_bilinear: bool,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    if d_psi_two_coordinate >= SPINOR_DIMENSION * 55 {
        return Err("D Psi_[2] coordinate is outside 32*55".to_string());
    }
    let derivative = d_psi_two_coordinate / 55;
    let pair = d_psi_two_coordinate % 55;
    let pair_mask = (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 2)
        .nth(pair)
        .unwrap();
    let axes = (0..VECTOR_DIMENSION)
        .filter(|axis| pair_mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    let gammas_i8 = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let gammas = gammas_i8
        .iter()
        .map(|matrix| {
            matrix
                .iter()
                .map(|row| row.iter().map(|&value| i16::from(value)).collect())
                .collect::<Vec<Vec<_>>>()
        })
        .collect::<Vec<_>>();
    let pair_gamma = multiply_integer(&gammas[axes[0]], &gammas[axes[1]]);
    let charge = crate::eleven_dimensional_majorana::real_charge_conjugation();
    let mut output = BTreeMap::new();
    for vector in 0..VECTOR_DIMENSION {
        let mut gamma_lower = gammas[vector].clone();
        if lower_output_vector && vector == 0 {
            for value in gamma_lower.iter_mut().flatten() {
                *value = -*value;
            }
        }
        let left = if raised_spinor_bilinear {
            multiply_integer_i8(&gamma_lower, &charge)
        } else {
            gamma_lower
        };
        let contraction = multiply_integer(&left, &pair_gamma);
        for output_spinor in 0..SPINOR_DIMENSION {
            let integer = i64::from(contraction[derivative][output_spinor]);
            if integer != 0 {
                add(
                    &mut output,
                    vector * SPINOR_DIMENSION + output_spinor,
                    ExactQi {
                        real: Ratio::from_integer(0),
                        imaginary: Ratio::new(
                            overall_sign * ordered_pair_factor * integer,
                            denominator,
                        ),
                    },
                );
            }
        }
    }
    Ok(output)
}

fn frame_to_target_response(
    frame: &BTreeMap<usize, ExactQi>,
    momentum_axis: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    let curl = frame_to_curl_response(frame, momentum_axis)?;
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    let mut output = BTreeMap::new();
    for (lexicographic_target, value) in operator.apply_sparse(&curl) {
        add(&mut output, numeric_target(lexicographic_target)?, value);
    }
    Ok(output)
}

fn frame_to_curl_response(
    frame: &BTreeMap<usize, ExactQi>,
    momentum_axis: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    if momentum_axis >= VECTOR_DIMENSION {
        return Err("local-Lorentz response momentum axis is outside 0..11".to_string());
    }
    let curvature = &target_sector_complex(TargetSector::RaritaSchwinger).curvature;
    let mut curl = BTreeMap::new();
    for (&frame_coordinate, frame_value) in frame {
        for (curl_coordinate, term) in curvature.column_terms(frame_coordinate) {
            let is_requested_momentum = term.monomial.exponents[momentum_axis] == 1
                && term
                    .monomial
                    .exponents
                    .iter()
                    .enumerate()
                    .all(|(axis, &exponent)| axis == momentum_axis || exponent == 0);
            if is_requested_momentum {
                add(
                    &mut curl,
                    curl_coordinate,
                    multiply(&frame_value, &public_coefficient(&term)),
                );
            }
        }
    }
    Ok(curl)
}

fn covector_spinor_gamma_trace(frame: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut trace = BTreeMap::new();
    for (&coordinate, value) in frame {
        let vector = coordinate / SPINOR_DIMENSION;
        let spinor = coordinate % SPINOR_DIMENSION;
        for row in 0..SPINOR_DIMENSION {
            let integer = i64::from(gammas[vector][row][spinor]);
            if integer != 0 {
                add(&mut trace, row, scale(value, integer));
            }
        }
    }
    trace
}

/// `S tensor V* = (10001) + (00001)` for the lower-vector Eq. (25)
/// coefficient. The trace is `Gamma^a psi_a`; reinjection is
/// `(1/11) Gamma_a trace`, so the time-axis metric belongs on reinjection.
fn split_covector_spinor(
    frame: &BTreeMap<usize, ExactQi>,
) -> (BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>) {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let trace = covector_spinor_gamma_trace(frame);
    let mut gamma_trace = BTreeMap::new();
    for vector in 0..VECTOR_DIMENSION {
        let metric = if vector == 0 { -1 } else { 1 };
        for row in 0..SPINOR_DIMENSION {
            for (&column, value) in &trace {
                let integer = i64::from(gammas[vector][row][column]) * metric;
                if integer != 0 {
                    add(
                        &mut gamma_trace,
                        vector * SPINOR_DIMENSION + row,
                        value.scaled(&Ratio::new(integer, 11)),
                    );
                }
            }
        }
    }
    let gamma_traceless = subtract_maps(frame, &gamma_trace);
    (gamma_traceless, gamma_trace)
}

/// Variance mutation: applies the upper-vector projector to the lower-vector
/// Eq. (25) coefficient by placing the time metric on the trace instead of
/// the reinjection.
fn split_covector_spinor_wrong_variance(
    frame: &BTreeMap<usize, ExactQi>,
) -> (BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>) {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut trace = BTreeMap::new();
    for (&coordinate, value) in frame {
        let vector = coordinate / SPINOR_DIMENSION;
        let spinor = coordinate % SPINOR_DIMENSION;
        let metric = if vector == 0 { -1 } else { 1 };
        for row in 0..SPINOR_DIMENSION {
            let integer = i64::from(gammas[vector][row][spinor]) * metric;
            if integer != 0 {
                add(&mut trace, row, scale(value, integer));
            }
        }
    }
    let mut gamma_trace = BTreeMap::new();
    for vector in 0..VECTOR_DIMENSION {
        for row in 0..SPINOR_DIMENSION {
            for (&column, value) in &trace {
                let integer = i64::from(gammas[vector][row][column]);
                if integer != 0 {
                    add(
                        &mut gamma_trace,
                        vector * SPINOR_DIMENSION + row,
                        value.scaled(&Ratio::new(integer, 11)),
                    );
                }
            }
        }
    }
    (subtract_maps(frame, &gamma_trace), gamma_trace)
}

fn numeric_target(lexicographic_target: usize) -> Result<usize, String> {
    if lexicographic_target >= TARGET_DIMENSION {
        return Err("corrected teleparallel target coordinate is out of range".to_string());
    }
    Ok((lexicographic_target / 330) * 330
        + lexicographic_four_form_to_numeric(lexicographic_target % 330)?)
}

fn pair_mask(pair: [usize; 2]) -> u32 {
    (1_u32 << pair[0]) | (1_u32 << pair[1])
}

fn is_one_momentum(exponents: &[u16; VECTOR_DIMENSION], axis: usize) -> bool {
    exponents[axis] == 1
        && exponents
            .iter()
            .enumerate()
            .all(|(other, &value)| other == axis || value == 0)
}

#[derive(Default)]
struct TeleparallelD21Cache {
    by_h: BTreeMap<usize, BTreeMap<FullChainRowKey, ExactQi>>,
    by_source: BTreeMap<u32, BTreeMap<usize, ExactQi>>,
}

#[derive(Default)]
struct StageD21Cache {
    by_h: BTreeMap<usize, CorrectedFullChainStageStreams>,
    d_delta_by_source: BTreeMap<u32, BTreeMap<usize, ExactQi>>,
    frame_by_source: BTreeMap<u32, BTreeMap<usize, ExactQi>>,
    curl_by_source: BTreeMap<u32, BTreeMap<usize, ExactQi>>,
    dg4_by_source: BTreeMap<u32, BTreeMap<usize, ExactQi>>,
}

impl StageD21Cache {
    fn ensure_h(&mut self, h: usize) -> Result<(), String> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.by_h.entry(h) {
            entry.insert(corrected_full_chain_stage_streams(h)?);
        }
        Ok(())
    }

    fn d_delta_slice(
        &mut self,
        source_coordinate: u32,
    ) -> Result<BTreeMap<usize, ExactQi>, String> {
        if let Some(cached) = self.d_delta_by_source.get(&source_coordinate) {
            return Ok(cached.clone());
        }
        let (pair, momentum, h) = decode_source_coordinate(source_coordinate)?;
        self.ensure_h(h)?;
        let mut output = BTreeMap::new();
        for (key, value) in &self.by_h[&h].d_delta {
            if key.exterior_spinor_mask == pair_mask(pair)
                && key.momentum_exponents.iter().all(|&exponent| exponent == 0)
            {
                add(
                    &mut output,
                    momentum * (SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION)
                        + key.output_coordinate,
                    value.clone(),
                );
            }
        }
        self.d_delta_by_source
            .insert(source_coordinate, output.clone());
        Ok(output)
    }

    fn frame_slice(&mut self, source_coordinate: u32) -> Result<BTreeMap<usize, ExactQi>, String> {
        if let Some(cached) = self.frame_by_source.get(&source_coordinate) {
            return Ok(cached.clone());
        }
        let (pair, momentum, h) = decode_source_coordinate(source_coordinate)?;
        self.ensure_h(h)?;
        let mut output = BTreeMap::new();
        for (key, value) in &self.by_h[&h].eq25_frame {
            if key.exterior_spinor_mask == pair_mask(pair)
                && key.momentum_exponents.iter().all(|&exponent| exponent == 0)
            {
                add(
                    &mut output,
                    momentum * 352 + key.output_coordinate,
                    value.clone(),
                );
            }
        }
        self.frame_by_source
            .insert(source_coordinate, output.clone());
        Ok(output)
    }

    fn curl_slice(&mut self, source_coordinate: u32) -> Result<BTreeMap<usize, ExactQi>, String> {
        if let Some(cached) = self.curl_by_source.get(&source_coordinate) {
            return Ok(cached.clone());
        }
        let (pair, momentum, h) = decode_source_coordinate(source_coordinate)?;
        self.ensure_h(h)?;
        let mut output = BTreeMap::new();
        for (key, value) in &self.by_h[&h].gravitino_curl {
            if key.exterior_spinor_mask == pair_mask(pair)
                && is_one_momentum(&key.momentum_exponents, momentum)
            {
                add(&mut output, key.output_coordinate, value.clone());
            }
        }
        self.curl_by_source
            .insert(source_coordinate, output.clone());
        Ok(output)
    }

    fn dg4_slice(&mut self, source_coordinate: u32) -> Result<BTreeMap<usize, ExactQi>, String> {
        if let Some(cached) = self.dg4_by_source.get(&source_coordinate) {
            return Ok(cached.clone());
        }
        let (pair, momentum, h) = decode_source_coordinate(source_coordinate)?;
        self.ensure_h(h)?;
        let mut output = BTreeMap::new();
        for (key, value) in &self.by_h[&h].teleparallel_dg4 {
            if key.exterior_spinor_mask == pair_mask(pair)
                && is_one_momentum(&key.momentum_exponents, momentum)
            {
                add(
                    &mut output,
                    numeric_target(key.output_coordinate)?,
                    value.clone(),
                );
            }
        }
        self.dg4_by_source.insert(source_coordinate, output.clone());
        Ok(output)
    }
}

fn lorentz_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn covector_generator(axis: usize, left: usize, right: usize) -> Vec<(usize, i64)> {
    let mut output = Vec::new();
    if axis == left {
        output.push((right, -lorentz_sign(right)));
    }
    if axis == right {
        output.push((left, lorentz_sign(left)));
    }
    output
}

fn vector_generator(axis: usize, left: usize, right: usize) -> Vec<(usize, i64)> {
    let mut output = Vec::new();
    if axis == right {
        output.push((left, lorentz_sign(right)));
    }
    if axis == left {
        output.push((right, -lorentz_sign(left)));
    }
    output
}

fn generator_ordinal(left: usize, right: usize) -> usize {
    (0..left)
        .map(|axis| VECTOR_DIMENSION - axis - 1)
        .sum::<usize>()
        + right
        - left
        - 1
}

fn spin_generator(left: usize, right: usize) -> &'static Vec<Vec<i16>> {
    static GENERATORS: OnceLock<Vec<Vec<Vec<i16>>>> = OnceLock::new();
    &GENERATORS.get_or_init(|| {
        let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
        let mut generators = Vec::with_capacity(55);
        for left in 0..VECTOR_DIMENSION {
            for right in (left + 1)..VECTOR_DIMENSION {
                let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
                let metric = i16::try_from(lorentz_sign(left) * lorentz_sign(right)).unwrap();
                for row in 0..SPINOR_DIMENSION {
                    for pivot in 0..SPINOR_DIMENSION {
                        let l = i16::from(gammas[left][row][pivot]);
                        if l == 0 {
                            continue;
                        }
                        for column in 0..SPINOR_DIMENSION {
                            output[row][column] +=
                                metric * l * i16::from(gammas[right][pivot][column]);
                        }
                    }
                }
                generators.push(output);
            }
        }
        generators
    })[generator_ordinal(left, right)]
}

fn frame_target_action(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
    dual_output_spinor: bool,
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        let spectator = coordinate / 352;
        let frame_coordinate = coordinate % 352;
        for (next, coefficient) in covector_generator(spectator, left, right) {
            add(
                &mut output,
                next * 352 + frame_coordinate,
                scale(value, 2 * coefficient),
            );
        }
        for (next_frame, coefficient) in
            frame_intrinsic_action_basis(frame_coordinate, left, right, dual_output_spinor)
        {
            add(
                &mut output,
                spectator * 352 + next_frame,
                scale(value, coefficient),
            );
        }
    }
    output
}

fn frame_intrinsic_action_basis(
    frame_coordinate: usize,
    left: usize,
    right: usize,
    dual_output_spinor: bool,
) -> Vec<(usize, i64)> {
    frame_intrinsic_action_basis_with_variance(
        frame_coordinate,
        left,
        right,
        dual_output_spinor,
        false,
    )
}

fn frame_intrinsic_action_basis_with_variance(
    frame_coordinate: usize,
    left: usize,
    right: usize,
    dual_output_spinor: bool,
    contravariant_output_vector: bool,
) -> Vec<(usize, i64)> {
    let spin = spin_generator(left, right);
    let vector = frame_coordinate / SPINOR_DIMENSION;
    let spinor = frame_coordinate % SPINOR_DIMENSION;
    let mut output = BTreeMap::new();
    let vector_terms = if contravariant_output_vector {
        vector_generator(vector, left, right)
    } else {
        covector_generator(vector, left, right)
    };
    for (next, coefficient) in vector_terms {
        *output.entry(next * SPINOR_DIMENSION + spinor).or_default() += 2 * coefficient;
    }
    if dual_output_spinor {
        for next in 0..SPINOR_DIMENSION {
            let coefficient = -i64::from(spin[spinor][next]);
            if coefficient != 0 {
                *output.entry(vector * SPINOR_DIMENSION + next).or_default() += coefficient;
            }
        }
    } else {
        for (next, row) in spin.iter().enumerate() {
            let coefficient = i64::from(row[spinor]);
            if coefficient != 0 {
                *output.entry(vector * SPINOR_DIMENSION + next).or_default() += coefficient;
            }
        }
    }
    output
        .into_iter()
        .filter(|(_, value)| *value != 0)
        .collect()
}

fn frame_intrinsic_action(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
    dual_output_spinor: bool,
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        for (next, coefficient) in
            frame_intrinsic_action_basis(coordinate, left, right, dual_output_spinor)
        {
            add(&mut output, next, scale(value, coefficient));
        }
    }
    output
}

fn frame_intrinsic_action_with_variance(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
    dual_output_spinor: bool,
    contravariant_output_vector: bool,
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        for (next, coefficient) in frame_intrinsic_action_basis_with_variance(
            coordinate,
            left,
            right,
            dual_output_spinor,
            contravariant_output_vector,
        ) {
            add(&mut output, next, scale(value, coefficient));
        }
    }
    output
}

fn frame_source_action(
    cache: &mut StageD21Cache,
    source_coordinate: u32,
    left: usize,
    right: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    let mut output = BTreeMap::new();
    for term in d21_source_lorentz_generator_terms(source_coordinate, left, right)? {
        for (row, value) in cache.frame_slice(term.source_coordinate)? {
            add(&mut output, row, scale(&value, term.coefficient));
        }
    }
    Ok(output)
}

fn stage_source_action<F>(
    cache: &mut StageD21Cache,
    source_coordinate: u32,
    left: usize,
    right: usize,
    mut slice: F,
) -> Result<BTreeMap<usize, ExactQi>, String>
where
    F: FnMut(&mut StageD21Cache, u32) -> Result<BTreeMap<usize, ExactQi>, String>,
{
    let mut output = BTreeMap::new();
    for term in d21_source_lorentz_generator_terms(source_coordinate, left, right)? {
        for (row, value) in slice(cache, term.source_coordinate)? {
            add(&mut output, row, scale(&value, term.coefficient));
        }
    }
    Ok(output)
}

fn two_form_masks() -> &'static Vec<u16> {
    static MASKS: OnceLock<Vec<u16>> = OnceLock::new();
    MASKS.get_or_init(|| {
        (0_u16..(1_u16 << VECTOR_DIMENSION))
            .filter(|mask| mask.count_ones() == 2)
            .collect()
    })
}

fn two_form_mask_lookup() -> &'static BTreeMap<u16, usize> {
    static LOOKUP: OnceLock<BTreeMap<u16, usize>> = OnceLock::new();
    LOOKUP.get_or_init(|| {
        two_form_masks()
            .iter()
            .enumerate()
            .map(|(ordinal, &mask)| (mask, ordinal))
            .collect()
    })
}

fn d_psi_two_generator_terms(
    coordinate: usize,
    left: usize,
    right: usize,
    dual_derivative_spinor: bool,
    contravariant_form_slots: [bool; 2],
) -> Vec<(usize, i64)> {
    let derivative = coordinate / 55;
    let pair = coordinate % 55;
    let masks = two_form_masks();
    let lookup = two_form_mask_lookup();
    let spin = spin_generator(left, right);
    let mut output = BTreeMap::new();
    if dual_derivative_spinor {
        for next in 0..SPINOR_DIMENSION {
            let coefficient = -i64::from(spin[derivative][next]);
            if coefficient != 0 {
                *output.entry(next * 55 + pair).or_default() += coefficient;
            }
        }
    } else {
        for (next, row) in spin.iter().enumerate() {
            let coefficient = i64::from(row[derivative]);
            if coefficient != 0 {
                *output.entry(next * 55 + pair).or_default() += coefficient;
            }
        }
    }
    let mask = masks[pair];
    let mut form_slot = 0;
    for axis in 0..VECTOR_DIMENSION {
        if mask & (1_u16 << axis) == 0 {
            continue;
        }
        let remaining = mask ^ (1_u16 << axis);
        let removal = if (remaining & ((1_u16 << axis) - 1)).count_ones() % 2 == 0 {
            1
        } else {
            -1
        };
        let vector_terms = if contravariant_form_slots[form_slot] {
            vector_generator(axis, left, right)
        } else {
            covector_generator(axis, left, right)
        };
        for (replacement, coefficient) in vector_terms {
            if remaining & (1_u16 << replacement) != 0 {
                continue;
            }
            let insertion = if (remaining & ((1_u16 << replacement) - 1)).count_ones() % 2 == 0 {
                1
            } else {
                -1
            };
            *output
                .entry(derivative * 55 + lookup[&(remaining | (1_u16 << replacement))])
                .or_default() += 2 * removal * insertion * coefficient;
        }
        form_slot += 1;
    }
    output
        .into_iter()
        .filter(|(_, value)| *value != 0)
        .collect()
}

/// Apply the source-fixed Lorentz action to one canonical `D_beta Psi^[de]`
/// coordinate. The stored derivative spinor is covariant, while the two form
/// slots are the raised slots used by the Eq. (24) Clifford injection. This is
/// the unique typed convention selected by the exhaustive Eq. (24)/Eq. (25)
/// intertwiner gates below.
pub(crate) fn authoritative_d_psi_two_generator_terms(
    coordinate: usize,
    left: usize,
    right: usize,
) -> Result<Vec<(usize, i64)>, String> {
    if coordinate >= SPINOR_DIMENSION * 55 || left >= right || right >= VECTOR_DIMENSION {
        return Err("authoritative D Psi_[2] Lorentz-action input is invalid".to_string());
    }
    Ok(d_psi_two_generator_terms(
        coordinate,
        left,
        right,
        true,
        [true, true],
    ))
}

/// Apply the Lorentz generator induced on the canonical rank-320 vertical
/// quotient at one formal momentum axis. Input and output keys are original
/// `D Psi_[2]` ordinals from the canonical pivot complement exported by
/// [`local_lorentz_target_image_handoff`]. The residual is returned
/// explicitly and must vanish before the coordinates are consumed by a
/// cocycle or coboundary solver.
pub(crate) fn canonical_psi2_quotient_action(
    momentum_axis: usize,
    coordinates: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
) -> Result<(BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>), String> {
    if momentum_axis >= VECTOR_DIMENSION
        || left >= right
        || right >= VECTOR_DIMENSION
        || coordinates
            .keys()
            .any(|&coordinate| coordinate >= SPINOR_DIMENSION * 55)
    {
        return Err("canonical Psi_[2] quotient-action input is invalid".to_string());
    }

    let mut acted_ambient = BTreeMap::new();
    for (&coordinate, value) in coordinates {
        for (next, coefficient) in authoritative_d_psi_two_generator_terms(coordinate, left, right)?
        {
            add(&mut acted_ambient, next, scale(value, coefficient));
        }
    }

    let mut acted_target = BTreeMap::new();
    for (coordinate, coefficient) in acted_ambient {
        for (row, value) in local_lorentz_target_response_column(coordinate, momentum_axis)? {
            add(&mut acted_target, row, multiply(&value, &coefficient));
        }
    }
    cached_local_lorentz_target_image_reducer(momentum_axis)?.solve_coordinates(acted_target)
}

/// Push an ambient 1,760-coordinate `D Psi_[2]` vector through the exact
/// vertical response and return its coordinates in the canonical rank-320
/// complement. A nonempty residual means the supplied vector did not land in
/// the declared vertical target image and is a hard failure.
pub(crate) fn canonical_psi2_quotient_reduce(
    momentum_axis: usize,
    ambient: &BTreeMap<usize, ExactQi>,
) -> Result<(BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>), String> {
    if momentum_axis >= VECTOR_DIMENSION
        || ambient
            .keys()
            .any(|&coordinate| coordinate >= SPINOR_DIMENSION * 55)
    {
        return Err("canonical Psi_[2] quotient-reduction input is invalid".to_string());
    }
    let mut target = BTreeMap::new();
    for (&coordinate, coefficient) in ambient {
        for (row, value) in local_lorentz_target_response_column(coordinate, momentum_axis)? {
            add(&mut target, row, multiply(&value, coefficient));
        }
    }
    cached_local_lorentz_target_image_reducer(momentum_axis)?.solve_coordinates(target)
}

/// Full twice-generator action on `V* tensor (D Psi_[2]/ker Q)`. Canonical
/// keys are `momentum_axis * 1760 + original_D_Psi2_ordinal`. The momentum
/// covector action is included before each destination-axis block is pushed
/// through `Q` and re-expanded in that axis's canonical rank-320 complement.
pub(crate) fn canonical_momentum_psi2_quotient_action(
    coordinates: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
) -> Result<(BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>), String> {
    const AMBIENT: usize = SPINOR_DIMENSION * 55;
    if left >= right
        || right >= VECTOR_DIMENSION
        || coordinates
            .keys()
            .any(|&coordinate| coordinate >= VECTOR_DIMENSION * AMBIENT)
    {
        return Err("momentum-Psi_[2] quotient-action input is invalid".to_string());
    }
    let mut ambient_by_momentum = vec![BTreeMap::new(); VECTOR_DIMENSION];
    for (&coordinate, value) in coordinates {
        let momentum = coordinate / AMBIENT;
        let psi = coordinate % AMBIENT;
        for (next, coefficient) in authoritative_d_psi_two_generator_terms(psi, left, right)? {
            add(
                &mut ambient_by_momentum[momentum],
                next,
                scale(value, coefficient),
            );
        }
        for (next_momentum, coefficient) in covector_generator(momentum, left, right) {
            add(
                &mut ambient_by_momentum[next_momentum],
                psi,
                scale(value, 2 * coefficient),
            );
        }
    }

    let mut canonical = BTreeMap::new();
    let mut residual = BTreeMap::new();
    for (momentum, ambient) in ambient_by_momentum.iter().enumerate() {
        if ambient.is_empty() {
            continue;
        }
        let (block, block_residual) = canonical_psi2_quotient_reduce(momentum, ambient)?;
        for (coordinate, value) in block {
            add(&mut canonical, momentum * AMBIENT + coordinate, value);
        }
        for (row, value) in block_residual {
            add(&mut residual, momentum * TARGET_DIMENSION + row, value);
        }
    }
    Ok((canonical, residual))
}

fn spinor_index_generator_terms(
    index: usize,
    left: usize,
    right: usize,
    dual: bool,
) -> Vec<(usize, i64)> {
    let spin = spin_generator(left, right);
    if dual {
        (0..SPINOR_DIMENSION)
            .filter_map(|next| {
                let coefficient = -i64::from(spin[index][next]);
                (coefficient != 0).then_some((next, coefficient))
            })
            .collect()
    } else {
        spin.iter()
            .enumerate()
            .filter_map(|(next, row)| {
                let coefficient = i64::from(row[index]);
                (coefficient != 0).then_some((next, coefficient))
            })
            .collect()
    }
}

fn d_delta_generator_action(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
    derivative_dual: bool,
    delta_dual: bool,
    epsilon_dual: bool,
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        let epsilon = coordinate % SPINOR_DIMENSION;
        let rest = coordinate / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let derivative = rest / SPINOR_DIMENSION;
        for (next, coefficient) in
            spinor_index_generator_terms(derivative, left, right, derivative_dual)
        {
            add(
                &mut output,
                (next * SPINOR_DIMENSION + delta) * SPINOR_DIMENSION + epsilon,
                scale(value, coefficient),
            );
        }
        for (next, coefficient) in spinor_index_generator_terms(delta, left, right, delta_dual) {
            add(
                &mut output,
                (derivative * SPINOR_DIMENSION + next) * SPINOR_DIMENSION + epsilon,
                scale(value, coefficient),
            );
        }
        for (next, coefficient) in spinor_index_generator_terms(epsilon, left, right, epsilon_dual)
        {
            add(
                &mut output,
                (derivative * SPINOR_DIMENSION + delta) * SPINOR_DIMENSION + next,
                scale(value, coefficient),
            );
        }
    }
    output
}

fn d_delta_stage_target_action(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
) -> BTreeMap<usize, ExactQi> {
    d_delta_stage_target_action_with_variance(input, left, right, true, true, false, false)
}

fn d_delta_stage_target_action_with_variance(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
    derivative_dual: bool,
    delta_dual: bool,
    epsilon_dual: bool,
    spectator_contravariant: bool,
) -> BTreeMap<usize, ExactQi> {
    const D_DELTA_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION;
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        let spectator = coordinate / D_DELTA_DIMENSION;
        let d_delta = coordinate % D_DELTA_DIMENSION;
        let spectator_terms = if spectator_contravariant {
            vector_generator(spectator, left, right)
        } else {
            covector_generator(spectator, left, right)
        };
        for (next, coefficient) in spectator_terms {
            add(
                &mut output,
                next * D_DELTA_DIMENSION + d_delta,
                scale(value, 2 * coefficient),
            );
        }
        let intrinsic = d_delta_generator_action(
            &BTreeMap::from([(d_delta, value.clone())]),
            left,
            right,
            derivative_dual,
            delta_dual,
            epsilon_dual,
        );
        for (next, coefficient) in intrinsic {
            add(
                &mut output,
                spectator * D_DELTA_DIMENSION + next,
                coefficient,
            );
        }
    }
    output
}

fn lexicographic_two_form_pairs() -> &'static Vec<[usize; 2]> {
    static PAIRS: OnceLock<Vec<[usize; 2]>> = OnceLock::new();
    PAIRS.get_or_init(|| {
        (0..VECTOR_DIMENSION)
            .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| [left, right]))
            .collect()
    })
}

fn curl_target_action(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
) -> BTreeMap<usize, ExactQi> {
    let pairs = lexicographic_two_form_pairs();
    let lookup = pairs
        .iter()
        .enumerate()
        .map(|(ordinal, &pair)| (pair, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        let pair_ordinal = coordinate / SPINOR_DIMENSION;
        let spinor = coordinate % SPINOR_DIMENSION;
        let pair = pairs[pair_ordinal];
        for slot in 0..2 {
            let axis = pair[slot];
            let other = pair[1 - slot];
            for (next, mut coefficient) in covector_generator(axis, left, right) {
                if next == other {
                    continue;
                }
                let next_pair = if slot == 0 && next < other {
                    [next, other]
                } else if slot == 0 {
                    coefficient = -coefficient;
                    [other, next]
                } else if other < next {
                    [other, next]
                } else {
                    coefficient = -coefficient;
                    [next, other]
                };
                add(
                    &mut output,
                    lookup[&next_pair] * SPINOR_DIMENSION + spinor,
                    scale(value, 2 * coefficient),
                );
            }
        }
        for (next, coefficient) in spinor_index_generator_terms(spinor, left, right, false) {
            add(
                &mut output,
                pair_ordinal * SPINOR_DIMENSION + next,
                scale(value, coefficient),
            );
        }
    }
    output
}

impl TeleparallelD21Cache {
    fn slice(&mut self, source_coordinate: u32) -> Result<BTreeMap<usize, ExactQi>, String> {
        if let Some(cached) = self.by_source.get(&source_coordinate) {
            return Ok(cached.clone());
        }
        let (pair, momentum, h) = decode_source_coordinate(source_coordinate)?;
        if let std::collections::btree_map::Entry::Vacant(entry) = self.by_h.entry(h) {
            let (_, target) = corrected_full_chain_streams(h)?;
            entry.insert(target);
        }
        let mut output = BTreeMap::new();
        for (key, value) in &self.by_h[&h] {
            if key.exterior_spinor_mask == pair_mask(pair)
                && is_one_momentum(&key.momentum_exponents, momentum)
            {
                add(
                    &mut output,
                    numeric_target(key.output_coordinate)?,
                    value.clone(),
                );
            }
        }
        self.by_source.insert(source_coordinate, output.clone());
        Ok(output)
    }
}

fn adapt_target_spinor(input: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
    let charge = crate::eleven_dimensional_majorana::real_charge_conjugation();
    let mut output = BTreeMap::new();
    for (&coordinate, value) in input {
        let source_spinor = coordinate / 330;
        let form = coordinate % 330;
        for (adapted_spinor, row) in charge.iter().enumerate() {
            let integer = i64::from(row[source_spinor]);
            if integer != 0 {
                add(
                    &mut output,
                    adapted_spinor * 330 + form,
                    scale(value, integer),
                );
            }
        }
    }
    output
}

fn target_action(
    input: &BTreeMap<usize, ExactQi>,
    left: usize,
    right: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    let mut output = BTreeMap::new();
    for (&coordinate, coefficient) in input {
        let action = dg4_lorentz_generator_action_integer(
            left,
            right,
            &BTreeMap::from([(coordinate, 1_i64)]),
        )?;
        for (next, integer) in action {
            add(&mut output, next, scale(coefficient, integer));
        }
    }
    Ok(output)
}

fn source_action(
    cache: &mut TeleparallelD21Cache,
    source_coordinate: u32,
    left: usize,
    right: usize,
    adapt_output_spinor: bool,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    let mut output = BTreeMap::new();
    for term in d21_source_lorentz_generator_terms(source_coordinate, left, right)? {
        let slice = cache.slice(term.source_coordinate)?;
        let slice = if adapt_output_spinor {
            adapt_target_spinor(&slice)
        } else {
            slice
        };
        for (target, value) in slice {
            add(&mut output, target, scale(&value, term.coefficient));
        }
    }
    Ok(output)
}

fn commutator_residual(
    cache: &mut TeleparallelD21Cache,
    source_coordinate: u32,
    left: usize,
    right: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    let input = cache.slice(source_coordinate)?;
    Ok(subtract_maps(
        &target_action(&input, left, right)?,
        &source_action(cache, source_coordinate, left, right, false)?,
    ))
}

fn witness_source_coordinates() -> Vec<u32> {
    let mut sources = WITNESS_ROWS
        .into_iter()
        .map(|row| u32::try_from(row / TARGET_DIMENSION as u64).unwrap())
        .collect::<Vec<_>>();
    sources.sort_unstable();
    sources.dedup();
    sources
}

fn build_report() -> Result<CorrectedTeleparallelEquivarianceReport, String> {
    let first_witness_source_coordinate =
        u32::try_from(FIRST_WITNESS_ROW / TARGET_DIMENSION as u64).unwrap();
    let sources = vec![first_witness_source_coordinate];
    let mut cache = TeleparallelD21Cache::default();
    let mut residual_entries = 0;
    let mut first_residual = None;
    let mut output_charge_adapted_residual_entries = 0;
    let mut output_charge_adapted_first_residual = None;
    let mut commutators_checked = 0;
    for &source in &sources {
        for left in 0..VECTOR_DIMENSION {
            for right in (left + 1)..VECTOR_DIMENSION {
                let input = cache.slice(source)?;
                let target = target_action(&input, left, right)?;
                let source_image = source_action(&mut cache, source, left, right, false)?;
                let residual = subtract_maps(&target, &source_image);
                residual_entries += residual.len();

                let adapted_input = adapt_target_spinor(&input);
                let adapted_target = target_action(&adapted_input, left, right)?;
                let adapted_source_image = source_action(&mut cache, source, left, right, true)?;
                let adapted_residual = subtract_maps(&adapted_target, &adapted_source_image);
                output_charge_adapted_residual_entries += adapted_residual.len();
                commutators_checked += 1;
                if first_residual.is_none() {
                    if let Some((&target_coordinate, value)) = residual.first_key_value() {
                        let (pair, momentum, h) = decode_source_coordinate(source)?;
                        first_residual = Some(EquivarianceResidualWitness {
                            source_coordinate: source,
                            source_pair: pair,
                            source_momentum: momentum,
                            h_hat_ordinal: h,
                            generator_left: left,
                            generator_right: right,
                            target_coordinate,
                            target_action_value: ExactQiPublic::from(
                                target.get(&target_coordinate).unwrap_or(&ExactQi::zero()),
                            ),
                            source_action_value: ExactQiPublic::from(
                                source_image
                                    .get(&target_coordinate)
                                    .unwrap_or(&ExactQi::zero()),
                            ),
                            residual: ExactQiPublic::from(value),
                        });
                    }
                }
                if output_charge_adapted_first_residual.is_none() {
                    if let Some((&target_coordinate, value)) = adapted_residual.first_key_value() {
                        let (pair, momentum, h) = decode_source_coordinate(source)?;
                        output_charge_adapted_first_residual = Some(EquivarianceResidualWitness {
                            source_coordinate: source,
                            source_pair: pair,
                            source_momentum: momentum,
                            h_hat_ordinal: h,
                            generator_left: left,
                            generator_right: right,
                            target_coordinate,
                            target_action_value: ExactQiPublic::from(
                                adapted_target
                                    .get(&target_coordinate)
                                    .unwrap_or(&ExactQi::zero()),
                            ),
                            source_action_value: ExactQiPublic::from(
                                adapted_source_image
                                    .get(&target_coordinate)
                                    .unwrap_or(&ExactQi::zero()),
                            ),
                            residual: ExactQiPublic::from(value),
                        });
                    }
                }
            }
        }
    }
    let output_charge_adapter_restores_equivariance = output_charge_adapted_residual_entries == 0;
    let witness_source_canary_equivariant = residual_entries == 0;
    Ok(CorrectedTeleparallelEquivarianceReport {
        schema_version: SCHEMA_VERSION,
        source_representation: "Lambda^2 S tensor V* tensor Hhat",
        target_representation: "S tensor Lambda^4 V",
        normalization: "twice-generator: Gamma_ab on each spinor slot and 2 M_ab on every vector slot",
        first_witness_row: FIRST_WITNESS_ROW,
        first_witness_source_coordinate,
        distinct_witness_source_coordinates: sources,
        generators_checked_per_source: 55,
        source_columns_checked: 1,
        commutators_checked,
        residual_entries,
        first_residual,
        output_charge_adapted_residual_entries,
        output_charge_adapted_first_residual,
        output_charge_adapter_restores_equivariance,
        witness_source_canary_equivariant,
        exhaustive_all_source_columns_complete: false,
        passed: output_charge_adapter_restores_equivariance,
        boundary: "This is an exact Lorentz-commutator canary on the first D21 source coordinate used by the corrected seven-row witness. A nonzero residual refutes equivariance of the corrected teleparallel serialization or one of its declared source/target actions. Zero residual on these sources does not prove all 1,745,920 source columns; full promotion requires an exhaustive orbit or all-column certificate.",
    })
}

pub fn verify() -> Result<CorrectedTeleparallelEquivarianceReport, String> {
    static REPORT: OnceLock<Result<CorrectedTeleparallelEquivarianceReport, String>> =
        OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrected_teleparallel_witness_sources_obey_or_localize_lorentz_commutator() {
        let report = verify().unwrap();
        eprintln!("TELEPARALLEL_EQUIVARIANCE {report:#?}");
        assert_eq!(report.first_witness_source_coordinate, 131_857);
        assert_eq!(report.distinct_witness_source_coordinates, vec![131_857]);
        assert_eq!(report.commutators_checked, 55);
        assert!(report.residual_entries > 0);
        assert!(report.first_residual.is_some());
        if report.output_charge_adapter_restores_equivariance {
            assert_eq!(report.output_charge_adapted_residual_entries, 0);
            assert!(report.output_charge_adapted_first_residual.is_none());
        } else {
            assert!(report.output_charge_adapted_residual_entries > 0);
            assert!(report.output_charge_adapted_first_residual.is_some());
        }
        assert_eq!(
            report.passed,
            report.output_charge_adapter_restores_equivariance
        );
        assert!(!report.exhaustive_all_source_columns_complete);
    }

    #[test]
    fn invalid_source_generator_indices_fail_closed() {
        assert!(d21_source_lorentz_generator_terms(131_857, 4, 4).is_err());
        assert!(dg4_lorentz_generator_action_integer(7, 7, &BTreeMap::new()).is_err());
        assert!(local_lorentz_target_response_column(1_760, 0).is_err());
        assert!(local_lorentz_target_response_column(0, 11).is_err());
    }

    #[test]
    fn first_residual_has_exact_local_lorentz_image_support_diagnostic() {
        let report = verify().unwrap();
        let witness = report.first_residual.unwrap();
        let mut supporting = Vec::new();
        for column in 0..SPINOR_DIMENSION * 55 {
            let image =
                local_lorentz_target_response_column(column, witness.source_momentum).unwrap();
            if let Some(value) = image.get(&witness.target_coordinate) {
                supporting.push((column, ExactQiPublic::from(value)));
            }
        }
        eprintln!(
            "LOCAL_LORENTZ_FIRST_ROW_SUPPORT target={} columns={supporting:?}",
            witness.target_coordinate
        );
        assert!(!supporting.is_empty());
    }

    #[test]
    fn first_commutator_reduces_exactly_or_localizes_outside_lorentz_image() {
        let source = 131_857;
        let (_, momentum, _) = decode_source_coordinate(source).unwrap();
        let reducer = local_lorentz_target_image_reducer(momentum).unwrap();
        eprintln!("LOCAL_LORENTZ_TARGET_IMAGE rank={}", reducer.pivots.len());
        let mut cache = TeleparallelD21Cache::default();
        let residual = commutator_residual(&mut cache, source, 0, 2).unwrap();
        let reduced = reducer.reduce(residual).unwrap();
        eprintln!(
            "LOCAL_LORENTZ_FIRST_COMMUTATOR_QUOTIENT residual_rows={} first={:?}",
            reduced.len(),
            reduced
                .first_key_value()
                .map(|(&row, value)| (row, ExactQiPublic::from(value)))
        );
        // Diagnostic gate: either exact membership or an exact first witness
        // outside the currently executable local-Lorentz image is required.
        if let Some((_, value)) = reduced.first_key_value() {
            assert!(!value.is_zero());
        }
    }

    #[test]
    fn all_55_commutators_reduce_against_fixed_momentum_lorentz_image() {
        let source = 131_857;
        let (_, momentum, _) = decode_source_coordinate(source).unwrap();
        let reducer = local_lorentz_target_image_reducer(momentum).unwrap();
        let mut cache = TeleparallelD21Cache::default();
        let mut raw_rows = 0;
        let mut quotient_rows = 0;
        let mut first = None;
        for left in 0..VECTOR_DIMENSION {
            for right in (left + 1)..VECTOR_DIMENSION {
                let residual = commutator_residual(&mut cache, source, left, right).unwrap();
                raw_rows += residual.len();
                let reduced = reducer.reduce(residual).unwrap();
                quotient_rows += reduced.len();
                if first.is_none() {
                    if let Some((&row, value)) = reduced.first_key_value() {
                        first = Some((left, right, row, ExactQiPublic::from(value)));
                    }
                }
            }
        }
        eprintln!(
            "LOCAL_LORENTZ_ALL55_QUOTIENT image_rank={} raw_rows={raw_rows} quotient_rows={quotient_rows} first={first:?}",
            reducer.pivots.len()
        );
        assert_eq!(raw_rows, 1_032);
        if quotient_rows == 0 {
            assert!(first.is_none());
        } else {
            assert!(first.is_some());
        }
    }

    #[test]
    fn raw_d0_psi01_direct_gravitino_canary_bypasses_hhat_section() {
        let mut counts = Vec::new();
        let mut first = None;
        for momentum_axis in 0..VECTOR_DIMENSION {
            let response = local_lorentz_target_response_column(0, momentum_axis).unwrap();
            counts.push(response.len());
            if first.is_none() {
                first = response
                    .first_key_value()
                    .map(|(&row, value)| (momentum_axis, row, ExactQiPublic::from(value)));
            }
        }
        eprintln!("RAW_D0_PSI01_DG4 counts={counts:?} first={first:?}");
        assert!(counts.iter().sum::<usize>() > 0);
        assert!(first.is_some());
    }

    #[test]
    fn psi2_eq25_frame_irrep_correction_kills_visible_vertical_dg4() {
        let momentum_axis = 5;
        let mut full_rank = ExactSparseImageReducer::default();
        let mut gamma_traceless_rank = ExactSparseImageReducer::default();
        let mut gamma_trace_rank = ExactSparseImageReducer::default();
        let mut full_target_rank = ExactSparseImageReducer::default();
        let mut gamma_traceless_target_rank = ExactSparseImageReducer::default();
        let mut gamma_trace_target_rank = ExactSparseImageReducer::default();
        let mut gamma_trace_target_rows = 0;
        let mut corrected_target_rows = 0;
        let mut gamma_trace_residual_rows = 0;
        for column in 0..SPINOR_DIMENSION * 55 {
            let frame = local_lorentz_eq25_frame_column(column).unwrap();
            let (gamma_traceless, gamma_trace) = split_covector_spinor(&frame);
            gamma_trace_residual_rows += covector_spinor_gamma_trace(&gamma_traceless).len();
            let reconstructed = {
                let mut value = gamma_traceless.clone();
                for (row, coefficient) in &gamma_trace {
                    add(&mut value, *row, coefficient.clone());
                }
                value
            };
            assert_eq!(reconstructed, frame);
            full_rank.insert_with_origin(frame.clone(), column).unwrap();
            gamma_traceless_rank
                .insert_with_origin(gamma_traceless.clone(), column)
                .unwrap();
            gamma_trace_rank
                .insert_with_origin(gamma_trace.clone(), column)
                .unwrap();
            let full_target = frame_to_target_response(&frame, momentum_axis).unwrap();
            let traceless_target =
                frame_to_target_response(&gamma_traceless, momentum_axis).unwrap();
            let trace_target = frame_to_target_response(&gamma_trace, momentum_axis).unwrap();
            full_target_rank
                .insert_with_origin(full_target, column)
                .unwrap();
            gamma_traceless_target_rank
                .insert_with_origin(traceless_target, column)
                .unwrap();
            gamma_trace_target_rank
                .insert_with_origin(trace_target.clone(), column)
                .unwrap();
            gamma_trace_target_rows += trace_target.len();
            // Q=-P_320 psi leaves P_32 psi, whose physical DG4 must vanish.
            corrected_target_rows += trace_target.len();
        }
        let frame0 = local_lorentz_eq25_frame_column(0).unwrap();
        let (_, wrong_trace) = split_covector_spinor_wrong_variance(&frame0);
        let wrong_variance_target_rows = frame_to_target_response(&wrong_trace, momentum_axis)
            .unwrap()
            .len();
        eprintln!(
            "PSI2_FRAME_IRREPS full_rank={} p320_rank={} p32_rank={} full_target_rank={} p320_target_rank={} p32_target_rank={} p320_trace_residual={} p32_target_rows={} corrected_target_rows={} wrong_variance_target_rows={}",
            full_rank.pivots.len(),
            gamma_traceless_rank.pivots.len(),
            gamma_trace_rank.pivots.len(),
            full_target_rank.pivots.len(),
            gamma_traceless_target_rank.pivots.len(),
            gamma_trace_target_rank.pivots.len(),
            gamma_trace_residual_rows,
            gamma_trace_target_rows,
            corrected_target_rows,
            wrong_variance_target_rows,
        );
        assert_eq!(
            gamma_traceless_rank.pivots.len() + gamma_trace_rank.pivots.len(),
            full_rank.pivots.len()
        );
        assert_eq!(gamma_trace_residual_rows, 0);
        assert!(wrong_variance_target_rows > 0);
    }

    #[test]
    fn paper_horizontal_eq25_correction_kills_all_pure_psi2_directions() {
        let mut raw_potential_rows = 0;
        let mut horizontal_potential_rows = 0;
        let mut horizontal_curl_dg4_rows = 0;
        let mut half_ordered_pair_mutation_rows = 0;
        let mut sign_mutation_rows = 0;
        let mut normalization_mutation_rows = 0;
        let mut time_metric_mutation_rows = 0;
        let mut spinor_variance_mutation_rows = 0;
        for column in 0..SPINOR_DIMENSION * 55 {
            let raw = local_lorentz_eq25_frame_column(column).unwrap();
            raw_potential_rows += raw.len();
            let horizontal = horizontal_local_lorentz_eq25_frame_column(column).unwrap();
            horizontal_potential_rows += horizontal.len();
            for momentum_axis in 0..VECTOR_DIMENSION {
                horizontal_curl_dg4_rows += frame_to_target_response(&horizontal, momentum_axis)
                    .unwrap()
                    .len();
            }
            // Mutation: forget the ordered-pair factor two in
            // `D Psi_de Gamma^{de}`, subtracting only one half of Eq. (25).
            let half = raw
                .iter()
                .map(|(&row, value)| (row, value.scaled(&Ratio::new(1, 2))))
                .collect::<BTreeMap<_, _>>();
            half_ordered_pair_mutation_rows += frame_to_target_response(&half, 5).unwrap().len();
            for (counter, mutation) in [
                (
                    &mut sign_mutation_rows,
                    explicit_horizontal_q_column(column, 2, 1, 64, true, true).unwrap(),
                ),
                (
                    &mut normalization_mutation_rows,
                    explicit_horizontal_q_column(column, 2, -1, 32, true, true).unwrap(),
                ),
                (
                    &mut time_metric_mutation_rows,
                    explicit_horizontal_q_column(column, 2, -1, 64, false, true).unwrap(),
                ),
                (
                    &mut spinor_variance_mutation_rows,
                    explicit_horizontal_q_column(column, 2, -1, 64, true, false).unwrap(),
                ),
            ] {
                let mut residual = raw.clone();
                for (row, value) in mutation {
                    add(&mut residual, row, value);
                }
                *counter += residual.len();
            }
        }
        eprintln!(
            "HORIZONTAL_EQ25_PSI2 columns={} raw_potential_rows={raw_potential_rows} horizontal_potential_rows={horizontal_potential_rows} horizontal_curl_dg4_rows={horizontal_curl_dg4_rows} half_pair_mutation_rows={half_ordered_pair_mutation_rows} sign_mutation_rows={sign_mutation_rows} normalization_mutation_rows={normalization_mutation_rows} time_metric_mutation_rows={time_metric_mutation_rows} spinor_variance_mutation_rows={spinor_variance_mutation_rows}",
            SPINOR_DIMENSION * 55,
        );
        assert!(raw_potential_rows > 0);
        assert_eq!(horizontal_potential_rows, 0);
        assert_eq!(horizontal_curl_dg4_rows, 0);
        assert!(half_ordered_pair_mutation_rows > 0);
        assert!(sign_mutation_rows > 0);
        assert!(normalization_mutation_rows > 0);
        assert!(time_metric_mutation_rows > 0);
        assert!(spinor_variance_mutation_rows > 0);
    }

    #[test]
    fn all55_canonical_original_column_coordinates_replay_exactly() {
        let handoff = local_lorentz_target_image_handoff(131_857).unwrap();
        assert_eq!(handoff.exact_image_rank, 320);
        assert_eq!(handoff.independent_original_columns.len(), 320);
        assert_eq!(handoff.raw_commutators.len(), 55);
        let columns = handoff
            .independent_original_columns
            .iter()
            .map(|column| (column.original_d_psi_two_coordinate, &column.entries))
            .collect::<BTreeMap<_, _>>();
        let mut nonzero_coordinate_terms = 0;
        for commutator in &handoff.raw_commutators {
            assert_eq!(commutator.exact_image_residual_entries, 0);
            let mut replay = BTreeMap::new();
            for (&column, coefficient) in &commutator.image_coordinates {
                nonzero_coordinate_terms += 1;
                for (&row, value) in columns[&column] {
                    add(&mut replay, row, multiply(value, coefficient));
                }
            }
            assert_eq!(replay, commutator.entries);
        }
        eprintln!(
            "LOCAL_LORENTZ_ALL55_COORDINATES rank={} nonzero_coordinate_terms={nonzero_coordinate_terms}",
            handoff.exact_image_rank
        );
        assert!(nonzero_coordinate_terms > 0);
    }

    #[test]
    fn stagewise_eq25_frame_commutator_localizes_variance_and_vertical_image() {
        let started = Instant::now();
        let source = 131_857;
        let (_, momentum, _) = decode_source_coordinate(source).unwrap();
        const D_DELTA_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION;
        let mut d_delta_image = ExactSparseImageReducer::default();
        let mut frame_image = ExactSparseImageReducer::default();
        let mut curl_image = ExactSparseImageReducer::default();
        let mut dg4_image = ExactSparseImageReducer::default();
        let mut q_frames = Vec::with_capacity(SPINOR_DIMENSION * 55);
        let mut q_curls = Vec::with_capacity(SPINOR_DIMENSION * 55);
        let mut q_dg4s = Vec::with_capacity(SPINOR_DIMENSION * 55);
        for column in 0..SPINOR_DIMENSION * 55 {
            let d_delta = inject_d_lorentz_compensator_into_d_delta(&BTreeMap::from([(
                column,
                ExactQi::one(),
            )]));
            d_delta_image
                .insert_with_origin(
                    d_delta
                        .into_iter()
                        .map(|(row, value)| (momentum * D_DELTA_DIMENSION + row, value))
                        .collect(),
                    column,
                )
                .unwrap();
            let frame = local_lorentz_eq25_frame_column(column).unwrap();
            frame_image
                .insert_with_origin(
                    frame
                        .iter()
                        .map(|(&row, value)| (momentum * 352 + row, value.clone()))
                        .collect(),
                    column,
                )
                .unwrap();
            curl_image
                .insert_with_origin(frame_to_curl_response(&frame, momentum).unwrap(), column)
                .unwrap();
            dg4_image
                .insert_with_origin(frame_to_target_response(&frame, momentum).unwrap(), column)
                .unwrap();
            let q = explicit_horizontal_q_column(column, 2, -1, 64, true, true).unwrap();
            q_curls.push(frame_to_curl_response(&q, momentum).unwrap());
            q_dg4s.push(frame_to_target_response(&q, momentum).unwrap());
            q_frames.push(
                q.into_iter()
                    .map(|(row, value)| (momentum * 352 + row, value))
                    .collect::<BTreeMap<_, _>>(),
            );
        }
        let mut cache = StageD21Cache::default();
        let mut stage_rows = [0_usize; 5];
        let mut stage_outside = [0_usize; 4];
        let mut horizontal_curl_rows = 0_usize;
        let mut horizontal_dg4_rows = 0_usize;
        let mut horizontal_first = None;
        let mut first_stage = None;
        for left in 0..VECTOR_DIMENSION {
            for right in (left + 1)..VECTOR_DIMENSION {
                let d_delta = cache.d_delta_slice(source).unwrap();
                let d_delta_rhs =
                    stage_source_action(&mut cache, source, left, right, |cache, next| {
                        cache.d_delta_slice(next)
                    })
                    .unwrap();
                let d_delta_residual = subtract_maps(
                    &d_delta_stage_target_action(&d_delta, left, right),
                    &d_delta_rhs,
                );
                stage_rows[0] += d_delta_residual.len();
                if first_stage.is_none() {
                    if let Some((&row, value)) = d_delta_residual.first_key_value() {
                        first_stage =
                            Some(("d_delta", left, right, row, ExactQiPublic::from(value)));
                    }
                }
                let (_d_delta_coordinates, d_delta_remainder) =
                    d_delta_image.solve_coordinates(d_delta_residual).unwrap();
                stage_outside[0] += d_delta_remainder.len();

                let frame = cache.frame_slice(source).unwrap();
                let frame_rhs = frame_source_action(&mut cache, source, left, right).unwrap();
                let frame_residual =
                    subtract_maps(&frame_target_action(&frame, left, right, false), &frame_rhs);
                stage_rows[1] += frame_residual.len();
                let (frame_coordinates, frame_remainder) = frame_image
                    .solve_coordinates(frame_residual.clone())
                    .unwrap();
                stage_outside[1] += frame_remainder.len();
                let mut horizontal_frame = frame_residual;
                for (&column, coefficient) in &frame_coordinates {
                    for (row, value) in &q_frames[column] {
                        add(&mut horizontal_frame, *row, multiply(value, coefficient));
                    }
                }
                stage_rows[2] += horizontal_frame.len();

                let curl = cache.curl_slice(source).unwrap();
                let curl_rhs =
                    stage_source_action(&mut cache, source, left, right, |cache, next| {
                        cache.curl_slice(next)
                    })
                    .unwrap();
                let curl_residual =
                    subtract_maps(&curl_target_action(&curl, left, right), &curl_rhs);
                stage_rows[3] += curl_residual.len();
                stage_outside[2] += curl_image.reduce(curl_residual.clone()).unwrap().len();
                let mut horizontal_curl = curl_residual;
                for (&column, coefficient) in &frame_coordinates {
                    for (row, value) in &q_curls[column] {
                        add(&mut horizontal_curl, *row, multiply(value, coefficient));
                    }
                }
                horizontal_curl_rows += horizontal_curl.len();
                if horizontal_first.is_none() {
                    if let Some((&row, value)) = horizontal_curl.first_key_value() {
                        horizontal_first =
                            Some(("curl", left, right, row, ExactQiPublic::from(value)));
                    }
                }

                let dg4 = cache.dg4_slice(source).unwrap();
                let dg4_rhs =
                    stage_source_action(&mut cache, source, left, right, |cache, next| {
                        cache.dg4_slice(next)
                    })
                    .unwrap();
                let dg4_residual =
                    subtract_maps(&target_action(&dg4, left, right).unwrap(), &dg4_rhs);
                stage_rows[4] += dg4_residual.len();
                stage_outside[3] += dg4_image.reduce(dg4_residual.clone()).unwrap().len();
                let mut horizontal_dg4 = dg4_residual;
                for (&column, coefficient) in &frame_coordinates {
                    for (row, value) in &q_dg4s[column] {
                        add(&mut horizontal_dg4, *row, multiply(value, coefficient));
                    }
                }
                horizontal_dg4_rows += horizontal_dg4.len();
                if horizontal_first.is_none() {
                    if let Some((&row, value)) = horizontal_dg4.first_key_value() {
                        horizontal_first =
                            Some(("dg4", left, right, row, ExactQiPublic::from(value)));
                    }
                }
            }
        }
        eprintln!(
            "HORIZONTAL_STAGEWISE d_delta_rows={} d_delta_outside={} frame_rows={} frame_outside={} horizontal_frame_rows={} curl_rows={} curl_outside={} horizontal_curl_rows={horizontal_curl_rows} dg4_rows={} dg4_outside={} horizontal_dg4_rows={horizontal_dg4_rows} ranks=[{},{},{},{}] first_stage={first_stage:?} horizontal_first={horizontal_first:?} elapsed_s={:.3}",
            stage_rows[0],
            stage_outside[0],
            stage_rows[1],
            stage_outside[1],
            stage_rows[2],
            stage_rows[3],
            stage_outside[2],
            stage_rows[4],
            stage_outside[3],
            d_delta_image.pivots.len(),
            frame_image.pivots.len(),
            curl_image.pivots.len(),
            dg4_image.pivots.len(),
            started.elapsed().as_secs_f64()
        );
        assert_eq!(d_delta_image.pivots.len(), SPINOR_DIMENSION * 55);
        assert_eq!(frame_image.pivots.len(), 352);
        assert!(stage_outside[0] > 0);
        assert_eq!(&stage_outside[1..], &[0, 0, 0]);
        assert_eq!(stage_rows[2], 0);
        assert_eq!(horizontal_curl_rows, 0);
        assert_eq!(horizontal_dg4_rows, 0);
    }

    #[test]
    fn hhat_d_delta_stage_selects_stored_variance_against_eq24_image() {
        let started = Instant::now();
        let source = 131_857;
        let (_, momentum, _) = decode_source_coordinate(source).unwrap();
        const D_DELTA_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION;
        let mut image = ExactSparseImageReducer::default();
        for column in 0..SPINOR_DIMENSION * 55 {
            let d_delta = inject_d_lorentz_compensator_into_d_delta(&BTreeMap::from([(
                column,
                ExactQi::one(),
            )]));
            image
                .insert_with_origin(
                    d_delta
                        .into_iter()
                        .map(|(row, value)| (momentum * D_DELTA_DIMENSION + row, value))
                        .collect(),
                    column,
                )
                .unwrap();
        }
        let mut cache = StageD21Cache::default();
        let input = cache.d_delta_slice(source).unwrap();
        let generators = (0..VECTOR_DIMENSION)
            .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let source_images = generators
            .iter()
            .map(|&(left, right)| {
                stage_source_action(&mut cache, source, left, right, |cache, next| {
                    cache.d_delta_slice(next)
                })
                .unwrap()
            })
            .collect::<Vec<_>>();
        let conventions = [false, true]
            .into_iter()
            .flat_map(|derivative_dual| {
                [false, true].into_iter().flat_map(move |delta_dual| {
                    [false, true].into_iter().flat_map(move |epsilon_dual| {
                        [false, true].into_iter().map(move |spectator_vector| {
                            (derivative_dual, delta_dual, epsilon_dual, spectator_vector)
                        })
                    })
                })
            })
            .collect::<Vec<_>>();
        let results = conventions
            .par_iter()
            .map(
                |&(derivative_dual, delta_dual, epsilon_dual, spectator_vector)| {
                    let mut rows = 0;
                    let mut outside = 0;
                    let mut first = None;
                    for (generator, &(left, right)) in generators.iter().enumerate() {
                        let residual = subtract_maps(
                            &d_delta_stage_target_action_with_variance(
                                &input,
                                left,
                                right,
                                derivative_dual,
                                delta_dual,
                                epsilon_dual,
                                spectator_vector,
                            ),
                            &source_images[generator],
                        );
                        rows += residual.len();
                        let reduced = image.reduce(residual).unwrap();
                        outside += reduced.len();
                        if first.is_none() {
                            first = reduced.first_key_value().map(|(&row, value)| {
                                (left, right, row, ExactQiPublic::from(value))
                            });
                        }
                    }
                    (
                        (derivative_dual, delta_dual, epsilon_dual, spectator_vector),
                        rows,
                        outside,
                        first,
                    )
                },
            )
            .collect::<Vec<_>>();
        for (convention, rows, outside, first) in &results {
            eprintln!(
                "DDELTA_VARIANCE convention={convention:?} rows={rows} outside={outside} first={first:?}"
            );
        }
        let zero = results
            .iter()
            .filter(|result| result.2 == 0)
            .map(|result| result.0)
            .collect::<Vec<_>>();
        eprintln!(
            "DDELTA_VARIANCE_ZERO {zero:?} elapsed_s={:.3}",
            started.elapsed().as_secs_f64()
        );
        assert!(zero.is_empty());
        assert!(results.iter().all(|result| result.2 == 226));
    }

    #[test]
    fn eq24_raised_two_form_injection_is_exact_typed_intertwiner() {
        let started = Instant::now();
        let injections = (0..SPINOR_DIMENSION * 55)
            .map(|column| {
                inject_d_lorentz_compensator_into_d_delta(&BTreeMap::from([(
                    column,
                    ExactQi::one(),
                )]))
            })
            .collect::<Vec<_>>();
        let generators = (0..VECTOR_DIMENSION)
            .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let source_terms = (0..SPINOR_DIMENSION * 55)
            .into_par_iter()
            .map(|column| {
                generators
                    .iter()
                    .map(|&(left, right)| {
                        d_psi_two_generator_terms(column, left, right, true, [true, true])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let observations = (0..SPINOR_DIMENSION * 55 * generators.len())
            .into_par_iter()
            .map(|job| {
                let column = job / generators.len();
                let generator = job % generators.len();
                let (left, right) = generators[generator];
                let mut source_image = BTreeMap::new();
                for &(next, coefficient) in &source_terms[column][generator] {
                    for (row, value) in &injections[next] {
                        add(&mut source_image, *row, scale(value, coefficient));
                    }
                }
                let residual = |derivative_dual, delta_dual, epsilon_dual| {
                    subtract_maps(
                        &d_delta_generator_action(
                            &injections[column],
                            left,
                            right,
                            derivative_dual,
                            delta_dual,
                            epsilon_dual,
                        ),
                        &source_image,
                    )
                    .len()
                };
                (
                    residual(true, true, false),
                    residual(false, true, false),
                    residual(true, false, false),
                    residual(true, true, true),
                )
            })
            .collect::<Vec<_>>();
        let totals = [0, 1, 2, 3].map(|slot| {
            observations
                .iter()
                .map(|observation| match slot {
                    0 => observation.0,
                    1 => observation.1,
                    2 => observation.2,
                    _ => observation.3,
                })
                .sum::<usize>()
        });
        eprintln!(
            "EQ24_TYPED_INTERTWINER residual={} derivative_variance_mutation={} delta_variance_mutation={} epsilon_variance_mutation={} elapsed_s={:.3}",
            totals[0],
            totals[1],
            totals[2],
            totals[3],
            started.elapsed().as_secs_f64()
        );
        assert_eq!(totals[0], 0);
        assert!(totals[1] > 0);
        assert!(totals[2] > 0);
        assert!(totals[3] > 0);
    }

    #[test]
    fn eq25_pure_psi2_injection_selects_frame_spinor_action() {
        let started = Instant::now();
        let frames = (0..SPINOR_DIMENSION * 55)
            .map(|column| local_lorentz_eq25_frame_column(column).unwrap())
            .collect::<Vec<_>>();
        eprintln!(
            "EQ25_PSI2_SELECTOR phase=frame_cache completed={} total={} elapsed_s={:.3}",
            frames.len(),
            SPINOR_DIMENSION * 55,
            started.elapsed().as_secs_f64()
        );
        let generators = (0..VECTOR_DIMENSION)
            .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let source_conventions = [false, true]
            .into_iter()
            .flat_map(|dual_derivative_spinor| {
                [false, true].into_iter().flat_map(move |first_vector| {
                    [false, true].into_iter().map(move |second_vector| {
                        (dual_derivative_spinor, [first_vector, second_vector])
                    })
                })
            })
            .collect::<Vec<_>>();
        let source_terms = source_conventions
            .par_iter()
            .copied()
            .map(|(dual_derivative_spinor, contravariant_form_slots)| {
                (0..SPINOR_DIMENSION * 55)
                    .map(|column| {
                        generators
                            .iter()
                            .map(|&(left, right)| {
                                d_psi_two_generator_terms(
                                    column,
                                    left,
                                    right,
                                    dual_derivative_spinor,
                                    contravariant_form_slots,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let total_jobs = SPINOR_DIMENSION * 55 * generators.len();
        eprintln!(
            "EQ25_PSI2_SELECTOR phase=source_action_cache completed={} total={} elapsed_s={:.3}",
            source_terms.len() * total_jobs,
            source_conventions.len() * total_jobs,
            started.elapsed().as_secs_f64()
        );
        let observation_started = Instant::now();
        let completed = AtomicUsize::new(0);
        let observations = (0..total_jobs)
            .into_par_iter()
            .map(|job| {
                let column = job / generators.len();
                let generator = job % generators.len();
                let (left, right) = generators[generator];
                let frame = &frames[column];
                let mut candidates = Vec::with_capacity(32);
                for (source_ordinal, &(source_dual, form_slots_vector)) in
                    source_conventions.iter().enumerate()
                {
                    let mut source_image = BTreeMap::new();
                    for &(next, coefficient) in &source_terms[source_ordinal][column][generator] {
                        for (row, value) in &frames[next] {
                            add(&mut source_image, *row, scale(value, coefficient));
                        }
                    }
                    for output_vector in [false, true] {
                        for output_dual in [false, true] {
                            let residual = subtract_maps(
                                &frame_intrinsic_action_with_variance(
                                    frame,
                                    left,
                                    right,
                                    output_dual,
                                    output_vector,
                                ),
                                &source_image,
                            );
                            candidates.push((
                                source_dual,
                                form_slots_vector,
                                output_dual,
                                output_vector,
                                residual.len(),
                                residual.first_key_value().map(|(&row, value)| {
                                    (column, left, right, row, ExactQiPublic::from(value))
                                }),
                            ));
                        }
                    }
                }
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 4_096 == 0 || done == total_jobs {
                    let elapsed = observation_started.elapsed().as_secs_f64();
                    let rate = done as f64 / elapsed.max(f64::MIN_POSITIVE);
                    let eta = (total_jobs - done) as f64 / rate.max(f64::MIN_POSITIVE);
                    eprintln!(
                        "EQ25_PSI2_SELECTOR phase=commutator completed={done} total={total_jobs} rate_per_s={rate:.1} eta_s={eta:.1}"
                    );
                }
                candidates
            })
            .collect::<Vec<_>>();
        let mut zero_candidates = Vec::new();
        for source_dual in [false, true] {
            for first_form_vector in [false, true] {
                for second_form_vector in [false, true] {
                    let form_slots_vector = [first_form_vector, second_form_vector];
                    for output_dual in [false, true] {
                        for output_vector in [false, true] {
                            let residual_rows = observations
                                .iter()
                                .flat_map(|job| job.iter())
                                .filter(|candidate| {
                                    candidate.0 == source_dual
                                        && candidate.1 == form_slots_vector
                                        && candidate.2 == output_dual
                                        && candidate.3 == output_vector
                                })
                                .map(|candidate| candidate.4)
                                .sum::<usize>();
                            let first = observations
                                .iter()
                                .flat_map(|job| job.iter())
                                .filter(|candidate| {
                                    candidate.0 == source_dual
                                        && candidate.1 == form_slots_vector
                                        && candidate.2 == output_dual
                                        && candidate.3 == output_vector
                                })
                                .find_map(|candidate| candidate.5.clone());
                            eprintln!(
                                "EQ25_PSI2_INJECTION source_dual={source_dual} form_slots_vector={form_slots_vector:?} output_dual={output_dual} output_vector={output_vector} residual_rows={residual_rows} first={first:?} elapsed_s={:.3}",
                                started.elapsed().as_secs_f64()
                            );
                            if residual_rows == 0 {
                                zero_candidates.push((
                                    source_dual,
                                    form_slots_vector,
                                    output_dual,
                                    output_vector,
                                ));
                            }
                        }
                    }
                }
            }
        }
        let authoritative = (true, [true, true], false, false);
        let globally_dual = (false, [false, false], true, true);
        assert_eq!(zero_candidates, vec![globally_dual, authoritative]);
        assert_ne!(authoritative, globally_dual);
    }
}
