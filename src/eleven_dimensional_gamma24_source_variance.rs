//! Exact source-variance join for the rank-two and rank-four `D H_hat` maps.
//!
//! The canonical `H_hat` basis stores a column spinor.  The Eq. (39) raised
//! bilinear is `B_[p] = Gamma_[p] C`, while lowering the canonical source
//! spinor contributes a second right-hand `C`.  Since the primitive Majorana
//! charge matrix obeys `C^2=-I`, the typed matrix is therefore
//! `B_[p] C = -Gamma_[p]`.  Applying `B_[p]` directly to the canonical basis is
//! retained only as a convention-mismatch witness.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
use crate::eleven_dimensional_majorana::{real_charge_conjugation, real_gamma_matrices};
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, SPINOR_DIMENSION, SparseQiEntry, SparseQiOperator, VECTOR_DIMENSION,
};

pub const H_HAT_DIMENSION: usize = 320;
pub const SOURCE_DIMENSION: usize = SPINOR_DIMENSION * H_HAT_DIMENSION;
pub const THREE_FORM_DIMENSION: usize = 165;
pub const FOUR_FORM_VECTOR_DIMENSION: usize = 330 * VECTOR_DIMENSION;
pub const FIVE_FORM_DIMENSION: usize = 462;
pub const GAMMA_FOUR_HOOK_RANK: usize = 3_003;
pub const PROOF_PRIME: i64 = 1_073_741_783;

pub const LEGACY_NORMALIZATION_REPORT_SHA256: &str =
    "1efa2a40eafef1a9a7c6c1ae40ca2988da8154230948e87b31fb45749b3018b8";
pub const LEGACY_CANDIDATE_STREAM_SHA256: &str =
    "1c90694c58edac95c3448c5a28417aa86cdc89c05b506d8dee6e143157558b17";
pub const LEGACY_TELEPARALLEL_STREAM_SHA256: &str =
    "4508051083d064f67bebb7471662f087a16beea6bb2d5ab293a95421c36417fe";
pub const LEGACY_RESIDUAL_STREAM_SHA256: &str =
    "51f950fa8b5ecf48421841d73b85e03c50b7e3521a1f6c677b525ddf152f236a";

type IntegerMatrix = Vec<Vec<i16>>;

#[derive(Clone, Debug)]
pub struct CorrectedGammaFourDecomposition {
    pub raw_form_vector: SparseQiOperator,
    pub trace_lambda_three: SparseQiOperator,
    pub exterior_lambda_five: SparseQiOperator,
    pub hook_10010: SparseQiOperator,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConventionMismatchWitness {
    pub derivative_spinor: usize,
    pub h_hat_ordinal: usize,
    pub target_three_form_mask: u16,
    pub three_times_gamma_two_exterior: i64,
    pub gamma_four_trace: i64,
    pub residual_against_legacy_common_ratio_three: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Gamma24SourceVarianceReport {
    pub schema_version: &'static str,
    pub source_type: &'static str,
    pub charge_join: &'static str,
    pub exact_identity: &'static str,
    pub source_dimension: usize,
    pub target_dimension: usize,
    pub coordinates_checked: usize,
    pub gamma_two_exterior_nonzero_entries: usize,
    pub gamma_four_trace_nonzero_entries: usize,
    pub exact_identity_residual_entries: usize,
    pub gamma_two_rank_mod_prime: usize,
    pub gamma_four_rank_mod_prime: usize,
    pub proof_prime: i64,
    pub corrected_gamma_two_sha256: String,
    pub corrected_gamma_four_trace_sha256: String,
    pub corrected_gamma_four_raw_sha256: String,
    pub corrected_gamma_four_exterior_sha256: String,
    pub corrected_gamma_four_hook_sha256: String,
    pub charge_antisymmetry_residual_entries: usize,
    pub charge_square_residual_entries: usize,
    pub gamma_trace_residual_entries: usize,
    pub legacy_common_ratio: i64,
    pub legacy_identity_residual_entries: usize,
    pub legacy_first_mismatch: Option<ConventionMismatchWitness>,
    pub legacy_normalization_report_sha256: &'static str,
    pub legacy_candidate_stream_sha256: &'static str,
    pub legacy_teleparallel_stream_sha256: &'static str,
    pub legacy_residual_stream_sha256: &'static str,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RawThreeChannelG4BianchiLaunchPlan {
    pub schema_version: &'static str,
    pub source_dimension: usize,
    pub trace_lambda_three_dimension: usize,
    pub exterior_lambda_five_dimension: usize,
    pub hook_10010_ambient_dimension: usize,
    pub hook_10010_rank: usize,
    pub measured_raw_rank: usize,
    pub measured_trace_rank: usize,
    pub measured_exterior_rank: usize,
    pub measured_hook_rank: usize,
    pub projector_reconstruction_residuals: usize,
    pub hook_trace_residuals: usize,
    pub hook_exterior_residuals: usize,
    pub trace_to_g4_adapter_ready: bool,
    pub exterior_to_g4_adapter_ready: bool,
    pub hook_to_g4_adapter_ready: bool,
    pub target_bianchi_ready: bool,
    pub launch_ready: bool,
    pub blockers: Vec<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct MomentumFourFormKey {
    four_form_mask: u16,
    momentum_axis: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct QuadraticFiveFormKey {
    five_form_mask: u16,
    momentum_axes: (usize, usize),
}

#[derive(Clone, Debug)]
struct PolynomialG4Operator {
    columns: Vec<BTreeMap<MomentumFourFormKey, ExactQi>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RawThreeChannelG4BianchiReport {
    pub schema_version: &'static str,
    pub source_dimension: usize,
    pub coefficient_columns: [&'static str; 3],
    pub g4_nonzero_rows_per_channel: [usize; 3],
    pub bianchi_nonzero_rows_per_channel: [usize; 3],
    pub g4_coefficient_rank_mod_prime: usize,
    pub bianchi_coefficient_rank_mod_prime: usize,
    pub measured_raw_rank: usize,
    pub measured_projection_ranks: [usize; 3],
    pub projector_reconstruction_residuals: usize,
    pub hook_trace_residuals: usize,
    pub hook_exterior_residuals: usize,
    pub trace_bianchi_residual_rows: usize,
    pub lambda_five_bianchi_residual_rows: usize,
    pub hook_bianchi_residual_rows: usize,
    pub adapter_equivariance_checks: usize,
    pub adapter_equivariance_residuals: usize,
    pub proof_prime: i64,
    pub all_g4_columns_independent: bool,
    pub bianchi_kernel_dimension: usize,
    pub exact_kernel_basis: Vec<[i64; 3]>,
    pub symbolic_rows_sha256: String,
    pub module_source_sha256: String,
    pub legacy_normalization_report_sha256: &'static str,
    pub legacy_candidate_stream_sha256: &'static str,
    pub legacy_teleparallel_stream_sha256: &'static str,
    pub legacy_residual_stream_sha256: &'static str,
    pub diagnostic_completed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn masks_of_degree(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn insertion_sign(mask: u16, index: usize) -> i64 {
    if (mask >> (index + 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn wedge_sign(mask: u16, index: usize) -> i64 {
    if (mask & ((1_u16 << index) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn lorentz_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
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

fn covariant_form_generator(mask: u16, left: usize, right: usize) -> Vec<(u16, i64)> {
    let mut output = Vec::new();
    for axis in 0..VECTOR_DIMENSION {
        if mask & (1_u16 << axis) == 0 {
            continue;
        }
        let remaining = mask ^ (1_u16 << axis);
        let removal = wedge_sign(remaining, axis);
        for (replacement, coefficient) in covector_generator(axis, left, right) {
            if remaining & (1_u16 << replacement) != 0 {
                continue;
            }
            let insertion = wedge_sign(remaining, replacement);
            output.push((
                remaining | (1_u16 << replacement),
                removal * insertion * coefficient,
            ));
        }
    }
    output
}

fn add_integer<K: Ord + Clone>(output: &mut BTreeMap<K, i64>, key: K, value: i64) {
    if value == 0 {
        return;
    }
    let entry = output.entry(key.clone()).or_default();
    *entry += value;
    if *entry == 0 {
        output.remove(&key);
    }
}

fn trace_adapter_basis(momentum: usize, form: u16) -> BTreeMap<u16, i64> {
    let mut output = BTreeMap::new();
    if form & (1_u16 << momentum) == 0 {
        output.insert(form | (1_u16 << momentum), wedge_sign(form, momentum));
    }
    output
}

fn lambda_five_adapter_basis(
    momentum: usize,
    form: u16,
    include_metric: bool,
) -> BTreeMap<u16, i64> {
    let mut output = BTreeMap::new();
    if form & (1_u16 << momentum) != 0 {
        let remaining = form ^ (1_u16 << momentum);
        let metric = if include_metric {
            lorentz_sign(momentum)
        } else {
            1
        };
        output.insert(remaining, wedge_sign(remaining, momentum) * metric);
    }
    output
}

fn hook_adapter_basis(
    momentum: usize,
    form: u16,
    vector: usize,
    include_metric: bool,
) -> BTreeMap<u16, i64> {
    let mut output = BTreeMap::new();
    if momentum == vector {
        output.insert(
            form,
            if include_metric {
                lorentz_sign(momentum)
            } else {
                1
            },
        );
    }
    output
}

fn adapter_equivariance_residuals(
    lambda_include_metric: bool,
    hook_include_metric: bool,
) -> (usize, usize) {
    let three_forms = masks_of_degree(3);
    let four_forms = masks_of_degree(4);
    let five_forms = masks_of_degree(5);
    let mut checks = 0;
    let mut residuals = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in (left + 1)..VECTOR_DIMENSION {
            for momentum in 0..VECTOR_DIMENSION {
                for &form in &three_forms {
                    let mut lhs = BTreeMap::new();
                    for (next, coefficient) in covector_generator(momentum, left, right) {
                        for (target, adapter) in trace_adapter_basis(next, form) {
                            add_integer(&mut lhs, target, coefficient * adapter);
                        }
                    }
                    for (next, coefficient) in covariant_form_generator(form, left, right) {
                        for (target, adapter) in trace_adapter_basis(momentum, next) {
                            add_integer(&mut lhs, target, coefficient * adapter);
                        }
                    }
                    let mut rhs = BTreeMap::new();
                    for (target, adapter) in trace_adapter_basis(momentum, form) {
                        for (next, coefficient) in covariant_form_generator(target, left, right) {
                            add_integer(&mut rhs, next, adapter * coefficient);
                        }
                    }
                    checks += 1;
                    residuals += usize::from(lhs != rhs);
                }
                for &form in &five_forms {
                    let mut lhs = BTreeMap::new();
                    for (next, coefficient) in covector_generator(momentum, left, right) {
                        for (target, adapter) in
                            lambda_five_adapter_basis(next, form, lambda_include_metric)
                        {
                            add_integer(&mut lhs, target, coefficient * adapter);
                        }
                    }
                    for (next, coefficient) in covariant_form_generator(form, left, right) {
                        for (target, adapter) in
                            lambda_five_adapter_basis(momentum, next, lambda_include_metric)
                        {
                            add_integer(&mut lhs, target, coefficient * adapter);
                        }
                    }
                    let mut rhs = BTreeMap::new();
                    for (target, adapter) in
                        lambda_five_adapter_basis(momentum, form, lambda_include_metric)
                    {
                        for (next, coefficient) in covariant_form_generator(target, left, right) {
                            add_integer(&mut rhs, next, adapter * coefficient);
                        }
                    }
                    checks += 1;
                    residuals += usize::from(lhs != rhs);
                }
                for &form in &four_forms {
                    for vector in 0..VECTOR_DIMENSION {
                        let mut lhs = BTreeMap::new();
                        for (next, coefficient) in covector_generator(momentum, left, right) {
                            for (target, adapter) in
                                hook_adapter_basis(next, form, vector, hook_include_metric)
                            {
                                add_integer(&mut lhs, target, coefficient * adapter);
                            }
                        }
                        for (next, coefficient) in covariant_form_generator(form, left, right) {
                            for (target, adapter) in
                                hook_adapter_basis(momentum, next, vector, hook_include_metric)
                            {
                                add_integer(&mut lhs, target, coefficient * adapter);
                            }
                        }
                        for (next, coefficient) in vector_generator(vector, left, right) {
                            for (target, adapter) in
                                hook_adapter_basis(momentum, form, next, hook_include_metric)
                            {
                                add_integer(&mut lhs, target, coefficient * adapter);
                            }
                        }
                        let mut rhs = BTreeMap::new();
                        for (target, adapter) in
                            hook_adapter_basis(momentum, form, vector, hook_include_metric)
                        {
                            for (next, coefficient) in covariant_form_generator(target, left, right)
                            {
                                add_integer(&mut rhs, next, adapter * coefficient);
                            }
                        }
                        checks += 1;
                        residuals += usize::from(lhs != rhs);
                    }
                }
            }
        }
    }
    (checks, residuals)
}

fn multiply(left: &IntegerMatrix, right: &IntegerMatrix) -> IntegerMatrix {
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

fn gamma_product_from(gammas: &[Vec<Vec<i8>>], indices: &[usize]) -> IntegerMatrix {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for index in 0..SPINOR_DIMENSION {
        output[index][index] = 1;
    }
    for &axis in indices {
        let gamma = gammas[axis]
            .iter()
            .map(|row| row.iter().map(|&value| i16::from(value)).collect())
            .collect::<IntegerMatrix>();
        output = multiply(&output, &gamma);
        if axis == 0 {
            for row in &mut output {
                for value in row {
                    *value = -*value;
                }
            }
        }
    }
    output
}

fn build_gamma_table(degree: usize, corrected: bool) -> Vec<(u16, IntegerMatrix)> {
    let gammas = real_gamma_matrices();
    let charge = real_charge_conjugation()
        .into_iter()
        .map(|row| row.into_iter().map(i16::from).collect())
        .collect::<IntegerMatrix>();
    masks_of_degree(degree)
        .into_iter()
        .map(|mask| {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            let product = gamma_product_from(&gammas, &indices);
            let matrix = if corrected {
                product
                    .into_iter()
                    .map(|row| row.into_iter().map(|value| -value).collect())
                    .collect()
            } else {
                multiply(&product, &charge)
            };
            (mask, matrix)
        })
        .collect()
}

fn gamma_table(degree: usize, corrected: bool) -> &'static Vec<(u16, IntegerMatrix)> {
    static CORRECTED_TWO: OnceLock<Vec<(u16, IntegerMatrix)>> = OnceLock::new();
    static CORRECTED_FOUR: OnceLock<Vec<(u16, IntegerMatrix)>> = OnceLock::new();
    static LEGACY_TWO: OnceLock<Vec<(u16, IntegerMatrix)>> = OnceLock::new();
    static LEGACY_FOUR: OnceLock<Vec<(u16, IntegerMatrix)>> = OnceLock::new();
    match (degree, corrected) {
        (2, true) => CORRECTED_TWO.get_or_init(|| build_gamma_table(2, true)),
        (4, true) => CORRECTED_FOUR.get_or_init(|| build_gamma_table(4, true)),
        (2, false) => LEGACY_TWO.get_or_init(|| build_gamma_table(2, false)),
        (4, false) => LEGACY_FOUR.get_or_init(|| build_gamma_table(4, false)),
        _ => panic!("source-variance gamma table is defined only for ranks two and four"),
    }
}

fn add_value<K: Ord + Clone>(output: &mut BTreeMap<K, ExactQi>, key: K, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(key.clone()).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&key);
    }
}

fn canonical_h_components(ordinal: usize, flip_time_component: bool) -> Vec<(usize, usize, i64)> {
    static COMPONENTS: OnceLock<Vec<Vec<(usize, usize, i64)>>> = OnceLock::new();
    let components = COMPONENTS.get_or_init(|| {
        canonical_gamma_traceless_frame_basis()
            .into_iter()
            .map(|column| {
                column
                    .into_iter()
                    .map(|(coordinate, value)| {
                        assert_eq!(*value.real.denom(), 1);
                        assert_eq!(*value.imaginary.numer(), 0);
                        (
                            coordinate / VECTOR_DIMENSION,
                            coordinate % VECTOR_DIMENSION,
                            *value.real.numer(),
                        )
                    })
                    .collect()
            })
            .collect()
    });
    components[ordinal]
        .iter()
        .map(|&(spinor, vector, source_coefficient)| {
            let mut coefficient = source_coefficient;
            if flip_time_component && vector == 0 {
                coefficient = -coefficient;
            }
            (spinor, vector, coefficient)
        })
        .collect()
}

fn raw_form_vector_column(
    degree: usize,
    derivative: usize,
    h_hat_ordinal: usize,
    corrected: bool,
    flip_time_component: bool,
) -> BTreeMap<(u16, usize), ExactQi> {
    let mut output = BTreeMap::new();
    for (mask, gamma) in gamma_table(degree, corrected) {
        for (spinor, vector, h_coefficient) in
            canonical_h_components(h_hat_ordinal, flip_time_component)
        {
            let coefficient = i64::from(gamma[derivative][spinor]) * h_coefficient;
            add_value(
                &mut output,
                (*mask, vector),
                ExactQi::from_integer(coefficient),
            );
        }
    }
    output
}

fn total_antisymmetric_part(
    degree: usize,
    input: &BTreeMap<(u16, usize), ExactQi>,
    include_lorentz_metric: bool,
) -> BTreeMap<u16, ExactQi> {
    let mut output = BTreeMap::new();
    for (&(mask, vector), value) in input {
        if mask & (1_u16 << vector) != 0 {
            continue;
        }
        let metric = if include_lorentz_metric {
            lorentz_sign(vector)
        } else {
            1
        };
        add_value(
            &mut output,
            mask | (1_u16 << vector),
            value.scaled(&Ratio::new(
                insertion_sign(mask, vector) * metric,
                (degree + 1) as i64,
            )),
        );
    }
    output
}

fn mixed_trace(input: &BTreeMap<(u16, usize), ExactQi>) -> BTreeMap<u16, ExactQi> {
    let mut output = BTreeMap::new();
    for (&(mask, vector), value) in input {
        if mask & (1_u16 << vector) == 0 {
            continue;
        }
        let remaining = mask ^ (1_u16 << vector);
        add_value(
            &mut output,
            remaining,
            value.scaled(&Ratio::from_integer(insertion_sign(remaining, vector))),
        );
    }
    output
}

fn inject_total_antisymmetric(
    degree: usize,
    input: &BTreeMap<u16, ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let mut output = BTreeMap::new();
    for (&mask, value) in input {
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) == 0 {
                continue;
            }
            let remaining = mask ^ (1_u16 << vector);
            output.insert(
                (remaining, vector),
                value.scaled(&Ratio::from_integer(
                    insertion_sign(remaining, vector) * lorentz_sign(vector),
                )),
            );
        }
    }
    assert!(
        output
            .keys()
            .all(|(mask, _)| mask.count_ones() as usize == degree)
    );
    output
}

fn inject_mixed_trace(
    degree: usize,
    input: &BTreeMap<u16, ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let sign = if (degree - 1) % 2 == 0 { 1 } else { -1 };
    let eigenvalue = sign * (VECTOR_DIMENSION - degree + 1) as i64;
    let mut output = BTreeMap::new();
    for (&mask, value) in input {
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) != 0 {
                continue;
            }
            let output_mask = mask | (1_u16 << vector);
            let less = (mask & ((1_u16 << vector) - 1)).count_ones();
            let insertion = if less % 2 == 0 { 1 } else { -1 };
            output.insert(
                (output_mask, vector),
                value.scaled(&Ratio::new(insertion, eigenvalue)),
            );
        }
    }
    output
}

fn hook_projection(
    degree: usize,
    input: &BTreeMap<(u16, usize), ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let mut output = input.clone();
    for projected in [
        inject_total_antisymmetric(degree, &total_antisymmetric_part(degree, input, true)),
        inject_mixed_trace(degree, &mixed_trace(input)),
    ] {
        for (key, value) in projected {
            add_value(&mut output, key, value.scaled(&Ratio::from_integer(-1)));
        }
    }
    output
}

fn sparse_operator_from_columns<K: Ord + Copy>(
    output_basis: &[K],
    columns: Vec<BTreeMap<K, ExactQi>>,
) -> SparseQiOperator {
    let lookup = output_basis
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, key)| (key, ordinal))
        .collect::<BTreeMap<_, _>>();
    SparseQiOperator {
        input_dimension: columns.len(),
        output_dimension: output_basis.len(),
        columns: columns
            .into_iter()
            .map(|column| {
                column
                    .into_iter()
                    .map(|(key, coefficient)| SparseQiEntry {
                        row: lookup[&key],
                        coefficient,
                    })
                    .collect()
            })
            .collect(),
    }
}

fn build_corrected_maps() -> (SparseQiOperator, CorrectedGammaFourDecomposition) {
    let mut gamma_two_columns = Vec::with_capacity(SOURCE_DIMENSION);
    let mut gamma_four_raw_columns = Vec::with_capacity(SOURCE_DIMENSION);
    let mut gamma_four_trace_columns = Vec::with_capacity(SOURCE_DIMENSION);
    let mut gamma_four_exterior_columns = Vec::with_capacity(SOURCE_DIMENSION);
    let mut gamma_four_hook_columns = Vec::with_capacity(SOURCE_DIMENSION);
    for derivative in 0..SPINOR_DIMENSION {
        for h_hat in 0..H_HAT_DIMENSION {
            let raw_two = raw_form_vector_column(2, derivative, h_hat, true, false);
            let raw_four = raw_form_vector_column(4, derivative, h_hat, true, false);
            gamma_two_columns.push(total_antisymmetric_part(2, &raw_two, true));
            gamma_four_trace_columns.push(mixed_trace(&raw_four));
            gamma_four_exterior_columns.push(total_antisymmetric_part(4, &raw_four, true));
            gamma_four_hook_columns.push(hook_projection(4, &raw_four));
            gamma_four_raw_columns.push(raw_four);
        }
    }
    let three_forms = masks_of_degree(3);
    let five_forms = masks_of_degree(5);
    let form_vector = masks_of_degree(4)
        .into_iter()
        .flat_map(|mask| (0..VECTOR_DIMENSION).map(move |vector| (mask, vector)))
        .collect::<Vec<_>>();
    (
        sparse_operator_from_columns(&three_forms, gamma_two_columns),
        CorrectedGammaFourDecomposition {
            raw_form_vector: sparse_operator_from_columns(&form_vector, gamma_four_raw_columns),
            trace_lambda_three: sparse_operator_from_columns(
                &three_forms,
                gamma_four_trace_columns,
            ),
            exterior_lambda_five: sparse_operator_from_columns(
                &five_forms,
                gamma_four_exterior_columns,
            ),
            hook_10010: sparse_operator_from_columns(&form_vector, gamma_four_hook_columns),
        },
    )
}

fn corrected_maps() -> &'static (SparseQiOperator, CorrectedGammaFourDecomposition) {
    static MAPS: OnceLock<(SparseQiOperator, CorrectedGammaFourDecomposition)> = OnceLock::new();
    MAPS.get_or_init(build_corrected_maps)
}

pub fn corrected_gamma_two_exterior_operator() -> SparseQiOperator {
    corrected_maps().0.clone()
}

/// Borrow the immutable corrected Gamma2 exterior operator without cloning
/// its 10,240 sparse columns. Descendant scans use this device-style shared
/// resident view across all 320 canonical source columns.
pub fn corrected_gamma_two_exterior_operator_ref() -> &'static SparseQiOperator {
    &corrected_maps().0
}

pub fn corrected_gamma_four_decomposition() -> CorrectedGammaFourDecomposition {
    corrected_maps().1.clone()
}

#[derive(Clone, Debug)]
struct GammaFourProjectionGate {
    raw_rank: usize,
    trace_rank: usize,
    exterior_rank: usize,
    hook_rank: usize,
    reconstruction_residuals: usize,
    hook_trace_residuals: usize,
    hook_exterior_residuals: usize,
}

fn form_vector_basis() -> Vec<(u16, usize)> {
    masks_of_degree(4)
        .into_iter()
        .flat_map(|mask| (0..VECTOR_DIMENSION).map(move |vector| (mask, vector)))
        .collect()
}

fn form_vector_column_as_map(
    operator: &SparseQiOperator,
    column: usize,
    basis: &[(u16, usize)],
) -> BTreeMap<(u16, usize), ExactQi> {
    operator.columns[column]
        .iter()
        .map(|entry| (basis[entry.row], entry.coefficient.clone()))
        .collect()
}

fn count_map_residuals<K: Ord + Clone>(
    left: &BTreeMap<K, ExactQi>,
    right: &BTreeMap<K, ExactQi>,
) -> usize {
    left.keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| left.get(key) != right.get(key))
        .count()
}

fn build_projection_gate() -> GammaFourProjectionGate {
    let maps = &corrected_maps().1;
    let basis = form_vector_basis();
    let mut reconstruction_residuals = 0;
    let mut hook_trace_residuals = 0;
    let mut hook_exterior_residuals = 0;
    for column in 0..SOURCE_DIMENSION {
        let raw = form_vector_column_as_map(&maps.raw_form_vector, column, &basis);
        let hook = form_vector_column_as_map(&maps.hook_10010, column, &basis);
        let mut reconstructed = hook.clone();
        for projected in [
            inject_total_antisymmetric(4, &total_antisymmetric_part(4, &raw, true)),
            inject_mixed_trace(4, &mixed_trace(&raw)),
        ] {
            for (key, value) in projected {
                add_value(&mut reconstructed, key, value);
            }
        }
        reconstruction_residuals += count_map_residuals(&raw, &reconstructed);
        hook_trace_residuals += mixed_trace(&hook).len();
        hook_exterior_residuals += total_antisymmetric_part(4, &hook, true).len();
    }
    GammaFourProjectionGate {
        raw_rank: modular_rank(&maps.raw_form_vector, PROOF_PRIME),
        trace_rank: modular_rank(&maps.trace_lambda_three, PROOF_PRIME),
        exterior_rank: modular_rank(&maps.exterior_lambda_five, PROOF_PRIME),
        hook_rank: modular_rank(&maps.hook_10010, PROOF_PRIME),
        reconstruction_residuals,
        hook_trace_residuals,
        hook_exterior_residuals,
    }
}

fn projection_gate() -> &'static GammaFourProjectionGate {
    static GATE: OnceLock<GammaFourProjectionGate> = OnceLock::new();
    GATE.get_or_init(build_projection_gate)
}

fn operator_column_as_map(
    operator: &SparseQiOperator,
    column: usize,
    basis: &[u16],
) -> BTreeMap<u16, ExactQi> {
    operator.columns[column]
        .iter()
        .map(|entry| (basis[entry.row], entry.coefficient.clone()))
        .collect()
}

fn build_trace_g4_operator(trace: &SparseQiOperator) -> PolynomialG4Operator {
    let three_forms = masks_of_degree(3);
    let columns = (0..SOURCE_DIMENSION)
        .map(|column| {
            let mut output = BTreeMap::new();
            for (form, value) in operator_column_as_map(trace, column, &three_forms) {
                for momentum in 0..VECTOR_DIMENSION {
                    for (four_form_mask, coefficient) in trace_adapter_basis(momentum, form) {
                        add_value(
                            &mut output,
                            MomentumFourFormKey {
                                four_form_mask,
                                momentum_axis: momentum,
                            },
                            value.scaled(&Ratio::from_integer(coefficient)),
                        );
                    }
                }
            }
            output
        })
        .collect();
    PolynomialG4Operator { columns }
}

fn build_lambda_five_g4_operator(exterior: &SparseQiOperator) -> PolynomialG4Operator {
    let five_forms = masks_of_degree(5);
    let columns = (0..SOURCE_DIMENSION)
        .map(|column| {
            let mut output = BTreeMap::new();
            for (form, value) in operator_column_as_map(exterior, column, &five_forms) {
                for momentum in 0..VECTOR_DIMENSION {
                    for (four_form_mask, coefficient) in
                        lambda_five_adapter_basis(momentum, form, true)
                    {
                        add_value(
                            &mut output,
                            MomentumFourFormKey {
                                four_form_mask,
                                momentum_axis: momentum,
                            },
                            value.scaled(&Ratio::from_integer(coefficient)),
                        );
                    }
                }
            }
            output
        })
        .collect();
    PolynomialG4Operator { columns }
}

fn build_hook_g4_operator(hook: &SparseQiOperator) -> PolynomialG4Operator {
    let form_vector = masks_of_degree(4)
        .into_iter()
        .flat_map(|mask| (0..VECTOR_DIMENSION).map(move |vector| (mask, vector)))
        .collect::<Vec<_>>();
    let columns = (0..SOURCE_DIMENSION)
        .map(|column| {
            let mut output = BTreeMap::new();
            for entry in &hook.columns[column] {
                let (form, vector) = form_vector[entry.row];
                for momentum in 0..VECTOR_DIMENSION {
                    for (four_form_mask, coefficient) in
                        hook_adapter_basis(momentum, form, vector, false)
                    {
                        add_value(
                            &mut output,
                            MomentumFourFormKey {
                                four_form_mask,
                                momentum_axis: momentum,
                            },
                            entry.coefficient.scaled(&Ratio::from_integer(coefficient)),
                        );
                    }
                }
            }
            output
        })
        .collect();
    PolynomialG4Operator { columns }
}

fn apply_bianchi(operator: &PolynomialG4Operator) -> Vec<BTreeMap<QuadraticFiveFormKey, ExactQi>> {
    operator
        .columns
        .iter()
        .map(|column| {
            let mut output = BTreeMap::new();
            for (key, value) in column {
                for derivative_axis in 0..VECTOR_DIMENSION {
                    if key.four_form_mask & (1_u16 << derivative_axis) != 0 {
                        continue;
                    }
                    let momentum_axes = if derivative_axis <= key.momentum_axis {
                        (derivative_axis, key.momentum_axis)
                    } else {
                        (key.momentum_axis, derivative_axis)
                    };
                    add_value(
                        &mut output,
                        QuadraticFiveFormKey {
                            five_form_mask: key.four_form_mask | (1_u16 << derivative_axis),
                            momentum_axes,
                        },
                        value.scaled(&Ratio::from_integer(wedge_sign(
                            key.four_form_mask,
                            derivative_axis,
                        ))),
                    );
                }
            }
            output
        })
        .collect()
}

fn aligned_coefficient_rank<K: Ord + Clone>(
    channels: &[Vec<BTreeMap<K, ExactQi>>; 3],
    prime: i64,
) -> usize {
    let mut pivots = Vec::<Vec<i64>>::new();
    for source in 0..SOURCE_DIMENSION {
        let keys = channels
            .iter()
            .flat_map(|channel| channel[source].keys().cloned())
            .collect::<BTreeSet<_>>();
        for key in keys {
            let mut row = channels
                .iter()
                .map(|channel| {
                    let value = channel[source]
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(ExactQi::zero);
                    assert_eq!(*value.imaginary.numer(), 0);
                    ((*value.real.numer()).rem_euclid(prime)
                        * mod_inverse((*value.real.denom()).rem_euclid(prime), prime))
                    .rem_euclid(prime)
                })
                .collect::<Vec<_>>();
            for pivot in &pivots {
                let Some(index) = pivot.iter().position(|&value| value != 0) else {
                    continue;
                };
                let factor = row[index];
                for column in index..3 {
                    row[column] = (row[column] - factor * pivot[column]).rem_euclid(prime);
                }
            }
            let Some(index) = row.iter().position(|&value| value != 0) else {
                continue;
            };
            let inverse = mod_inverse(row[index], prime);
            for value in &mut row[index..] {
                *value = (*value * inverse).rem_euclid(prime);
            }
            pivots.push(row);
            pivots.sort_by_key(|pivot| pivot.iter().position(|&value| value != 0));
            if pivots.len() == 3 {
                return 3;
            }
        }
    }
    pivots.len()
}

fn polynomial_rows(operator: &PolynomialG4Operator) -> Vec<BTreeMap<MomentumFourFormKey, ExactQi>> {
    operator.columns.clone()
}

fn hash_qi(hasher: &mut Sha256, value: &ExactQi) {
    hasher.update(value.real.numer().to_le_bytes());
    hasher.update(value.real.denom().to_le_bytes());
    hasher.update(value.imaginary.numer().to_le_bytes());
    hasher.update(value.imaginary.denom().to_le_bytes());
}

fn bianchi_rows_sha256(channels: &[Vec<BTreeMap<QuadraticFiveFormKey, ExactQi>>; 3]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-11d-g4-adapter-formal-bianchi-oracle-v1");
    for source in 0..SOURCE_DIMENSION {
        let keys = channels
            .iter()
            .flat_map(|channel| channel[source].keys().cloned())
            .collect::<BTreeSet<_>>();
        for key in keys {
            hasher.update((source as u64).to_le_bytes());
            hasher.update(key.five_form_mask.to_le_bytes());
            hasher.update([key.momentum_axes.0 as u8, key.momentum_axes.1 as u8]);
            for channel in channels {
                hash_qi(
                    &mut hasher,
                    &channel[source]
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(ExactQi::zero),
                );
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

fn operator_sha256(domain: &[u8], operator: &SparseQiOperator) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((operator.input_dimension as u64).to_le_bytes());
    hasher.update((operator.output_dimension as u64).to_le_bytes());
    for (column, entries) in operator.columns.iter().enumerate() {
        for entry in entries {
            hasher.update((column as u64).to_le_bytes());
            hasher.update((entry.row as u64).to_le_bytes());
            hasher.update(entry.coefficient.real.numer().to_le_bytes());
            hasher.update(entry.coefficient.real.denom().to_le_bytes());
            hasher.update(entry.coefficient.imaginary.numer().to_le_bytes());
            hasher.update(entry.coefficient.imaginary.denom().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn mod_inverse(value: i64, prime: i64) -> i64 {
    let mut base = value.rem_euclid(prime);
    let mut exponent = prime - 2;
    let mut output = 1_i64;
    while exponent > 0 {
        if exponent & 1 == 1 {
            output = (output * base).rem_euclid(prime);
        }
        base = (base * base).rem_euclid(prime);
        exponent >>= 1;
    }
    output
}

fn modular_rank(operator: &SparseQiOperator, prime: i64) -> usize {
    let mut pivots = vec![None::<Vec<i64>>; operator.output_dimension];
    let mut rank = 0;
    for entries in &operator.columns {
        let mut column = vec![0_i64; operator.output_dimension];
        for entry in entries {
            let denominator = (*entry.coefficient.real.denom()).rem_euclid(prime);
            assert_ne!(denominator, 0);
            column[entry.row] = ((*entry.coefficient.real.numer()).rem_euclid(prime)
                * mod_inverse(denominator, prime))
            .rem_euclid(prime);
        }
        loop {
            let Some(pivot) = column.iter().position(|&value| value != 0) else {
                break;
            };
            if let Some(existing) = &pivots[pivot] {
                let factor = column[pivot];
                for row in pivot..operator.output_dimension {
                    column[row] = (column[row] - factor * existing[row]).rem_euclid(prime);
                }
            } else {
                let inverse = mod_inverse(column[pivot], prime);
                for value in &mut column[pivot..] {
                    *value = (*value * inverse).rem_euclid(prime);
                }
                pivots[pivot] = Some(column);
                rank += 1;
                break;
            }
        }
    }
    rank
}

fn charge_residuals() -> (usize, usize) {
    let charge = real_charge_conjugation();
    let mut antisymmetry = 0;
    let mut square = 0;
    for row in 0..SPINOR_DIMENSION {
        for column in 0..SPINOR_DIMENSION {
            antisymmetry += usize::from(charge[row][column] != -charge[column][row]);
            let product = (0..SPINOR_DIMENSION)
                .map(|pivot| i16::from(charge[row][pivot]) * i16::from(charge[pivot][column]))
                .sum::<i16>();
            let expected = if row == column { -1 } else { 0 };
            square += usize::from(product != expected);
        }
    }
    (antisymmetry, square)
}

fn gamma_trace_residuals() -> usize {
    let gammas = real_gamma_matrices();
    let mut residuals = 0;
    for ordinal in 0..H_HAT_DIMENSION {
        let components = canonical_h_components(ordinal, false);
        for row in 0..SPINOR_DIMENSION {
            let value = components
                .iter()
                .map(|&(spinor, vector, coefficient)| {
                    lorentz_sign(vector) * i64::from(gammas[vector][row][spinor]) * coefficient
                })
                .sum::<i64>();
            residuals += usize::from(value != 0);
        }
    }
    residuals
}

fn scaled_identity_residuals(
    gamma_two: &SparseQiOperator,
    gamma_four: &SparseQiOperator,
    gamma_four_scale: i64,
) -> usize {
    assert_eq!(gamma_two.input_dimension, gamma_four.input_dimension);
    assert_eq!(gamma_two.output_dimension, gamma_four.output_dimension);
    let mut residuals = 0;
    for column in 0..gamma_two.input_dimension {
        let left = gamma_two.columns[column]
            .iter()
            .map(|entry| (entry.row, entry.coefficient.clone()))
            .collect::<BTreeMap<_, _>>();
        let right = gamma_four.columns[column]
            .iter()
            .map(|entry| (entry.row, entry.coefficient.clone()))
            .collect::<BTreeMap<_, _>>();
        for row in left
            .keys()
            .chain(right.keys())
            .copied()
            .collect::<BTreeSet<_>>()
        {
            let mut residual = left
                .get(&row)
                .cloned()
                .unwrap_or_else(ExactQi::zero)
                .scaled(&Ratio::from_integer(3));
            residual.add_assign(
                &right
                    .get(&row)
                    .cloned()
                    .unwrap_or_else(ExactQi::zero)
                    .scaled(&Ratio::from_integer(gamma_four_scale)),
            );
            residuals += usize::from(!residual.is_zero());
        }
    }
    residuals
}

fn build_legacy_trace_maps() -> (SparseQiOperator, SparseQiOperator) {
    let mut gamma_two_columns = Vec::with_capacity(SOURCE_DIMENSION);
    let mut gamma_four_columns = Vec::with_capacity(SOURCE_DIMENSION);
    for derivative in 0..SPINOR_DIMENSION {
        for h_hat in 0..H_HAT_DIMENSION {
            gamma_two_columns.push(total_antisymmetric_part(
                2,
                &raw_form_vector_column(2, derivative, h_hat, false, false),
                true,
            ));
            gamma_four_columns.push(mixed_trace(&raw_form_vector_column(
                4, derivative, h_hat, false, false,
            )));
        }
    }
    let basis = masks_of_degree(3);
    (
        sparse_operator_from_columns(&basis, gamma_two_columns),
        sparse_operator_from_columns(&basis, gamma_four_columns),
    )
}

fn legacy_mismatch(
    gamma_two: &SparseQiOperator,
    gamma_four: &SparseQiOperator,
) -> (usize, Option<ConventionMismatchWitness>) {
    let masks = masks_of_degree(3);
    let mut residuals = 0;
    let mut first = None;
    for column in 0..SOURCE_DIMENSION {
        let two = gamma_two.columns[column]
            .iter()
            .map(|entry| (entry.row, entry.coefficient.clone()))
            .collect::<BTreeMap<_, _>>();
        let four = gamma_four.columns[column]
            .iter()
            .map(|entry| (entry.row, entry.coefficient.clone()))
            .collect::<BTreeMap<_, _>>();
        for row in two
            .keys()
            .chain(four.keys())
            .copied()
            .collect::<BTreeSet<_>>()
        {
            let three_two = two
                .get(&row)
                .cloned()
                .unwrap_or_else(ExactQi::zero)
                .scaled(&Ratio::from_integer(3));
            let four_value = four.get(&row).cloned().unwrap_or_else(ExactQi::zero);
            let mut residual = four_value.clone();
            residual.add_assign(&three_two.scaled(&Ratio::from_integer(-1)));
            if !residual.is_zero() {
                residuals += 1;
                if first.is_none() {
                    assert_eq!(*three_two.real.denom(), 1);
                    assert_eq!(*four_value.real.denom(), 1);
                    assert_eq!(*residual.real.denom(), 1);
                    first = Some(ConventionMismatchWitness {
                        derivative_spinor: column / H_HAT_DIMENSION,
                        h_hat_ordinal: column % H_HAT_DIMENSION,
                        target_three_form_mask: masks[row],
                        three_times_gamma_two_exterior: *three_two.real.numer(),
                        gamma_four_trace: *four_value.real.numer(),
                        residual_against_legacy_common_ratio_three: *residual.real.numer(),
                    });
                }
            }
        }
    }
    (residuals, first)
}

pub fn verify_source_variance() -> Gamma24SourceVarianceReport {
    let gamma_two = corrected_gamma_two_exterior_operator();
    let gamma_four = corrected_gamma_four_decomposition();
    let identity_residuals =
        scaled_identity_residuals(&gamma_two, &gamma_four.trace_lambda_three, 1);
    let (charge_antisymmetry_residual_entries, charge_square_residual_entries) = charge_residuals();
    let gamma_trace_residual_entries = gamma_trace_residuals();
    let (legacy_two, legacy_four) = build_legacy_trace_maps();
    let (legacy_identity_residual_entries, legacy_first_mismatch) =
        legacy_mismatch(&legacy_two, &legacy_four);
    let gamma_two_rank_mod_prime = modular_rank(&gamma_two, PROOF_PRIME);
    let gamma_four_rank_mod_prime = modular_rank(&gamma_four.trace_lambda_three, PROOF_PRIME);
    let passed = identity_residuals == 0
        && charge_antisymmetry_residual_entries == 0
        && charge_square_residual_entries == 0
        && gamma_trace_residual_entries == 0
        && gamma_two_rank_mod_prime == THREE_FORM_DIMENSION
        && gamma_four_rank_mod_prime == THREE_FORM_DIMENSION
        && legacy_identity_residual_entries > 0
        && legacy_first_mismatch.is_some();
    Gamma24SourceVarianceReport {
        schema_version: "adynkra-11d-gamma24-source-variance-v1",
        source_type: "S_D^* tensor H_hat, derivative-major and canonical-H-hat-minor",
        charge_join: "B_[p] C = (Gamma_[p] C) C = -Gamma_[p], because C^2=-I",
        exact_identity: "Gamma4_trace + 3*Gamma2_exterior = 0",
        source_dimension: SOURCE_DIMENSION,
        target_dimension: THREE_FORM_DIMENSION,
        coordinates_checked: SOURCE_DIMENSION * THREE_FORM_DIMENSION,
        gamma_two_exterior_nonzero_entries: gamma_two.columns.iter().map(Vec::len).sum(),
        gamma_four_trace_nonzero_entries: gamma_four
            .trace_lambda_three
            .columns
            .iter()
            .map(Vec::len)
            .sum(),
        exact_identity_residual_entries: identity_residuals,
        gamma_two_rank_mod_prime,
        gamma_four_rank_mod_prime,
        proof_prime: PROOF_PRIME,
        corrected_gamma_two_sha256: operator_sha256(b"corrected-gamma2-exterior-v1", &gamma_two),
        corrected_gamma_four_trace_sha256: operator_sha256(
            b"corrected-gamma4-trace-v1",
            &gamma_four.trace_lambda_three,
        ),
        corrected_gamma_four_raw_sha256: operator_sha256(
            b"corrected-gamma4-raw-v1",
            &gamma_four.raw_form_vector,
        ),
        corrected_gamma_four_exterior_sha256: operator_sha256(
            b"corrected-gamma4-exterior-v1",
            &gamma_four.exterior_lambda_five,
        ),
        corrected_gamma_four_hook_sha256: operator_sha256(
            b"corrected-gamma4-hook-v1",
            &gamma_four.hook_10010,
        ),
        charge_antisymmetry_residual_entries,
        charge_square_residual_entries,
        gamma_trace_residual_entries,
        legacy_common_ratio: 3,
        legacy_identity_residual_entries,
        legacy_first_mismatch,
        legacy_normalization_report_sha256: LEGACY_NORMALIZATION_REPORT_SHA256,
        legacy_candidate_stream_sha256: LEGACY_CANDIDATE_STREAM_SHA256,
        legacy_teleparallel_stream_sha256: LEGACY_TELEPARALLEL_STREAM_SHA256,
        legacy_residual_stream_sha256: LEGACY_RESIDUAL_STREAM_SHA256,
        passed,
        boundary: "Passing proves the exact source-variance join, the Gamma2/Gamma4 multiplicity-one identity, and full rank of their common Lambda3 image. It does not identify the Lambda5 or (10010) momentum-to-G4 adapters, authorize a three-channel Bianchi launch, fix the physical four-form normalization, construct target gauge K, or prove irreducibility.",
    }
}

pub fn raw_three_channel_g4_bianchi_launch_plan() -> RawThreeChannelG4BianchiLaunchPlan {
    let gate = projection_gate();
    let projection_ready = gate.trace_rank == THREE_FORM_DIMENSION
        && gate.exterior_rank == FIVE_FORM_DIMENSION
        && gate.hook_rank == GAMMA_FOUR_HOOK_RANK
        && gate.raw_rank == FOUR_FORM_VECTOR_DIMENSION
        && gate.reconstruction_residuals == 0
        && gate.hook_trace_residuals == 0
        && gate.hook_exterior_residuals == 0;
    let blockers = if projection_ready {
        Vec::new()
    } else {
        vec!["Gamma4 projection ranks, reconstruction, or hook-annihilation gates failed"]
    };
    RawThreeChannelG4BianchiLaunchPlan {
        schema_version: "adynkra-11d-raw-three-channel-g4-bianchi-plan-v1",
        source_dimension: SOURCE_DIMENSION,
        trace_lambda_three_dimension: THREE_FORM_DIMENSION,
        exterior_lambda_five_dimension: FIVE_FORM_DIMENSION,
        hook_10010_ambient_dimension: FOUR_FORM_VECTOR_DIMENSION,
        hook_10010_rank: GAMMA_FOUR_HOOK_RANK,
        measured_raw_rank: gate.raw_rank,
        measured_trace_rank: gate.trace_rank,
        measured_exterior_rank: gate.exterior_rank,
        measured_hook_rank: gate.hook_rank,
        projector_reconstruction_residuals: gate.reconstruction_residuals,
        hook_trace_residuals: gate.hook_trace_residuals,
        hook_exterior_residuals: gate.hook_exterior_residuals,
        trace_to_g4_adapter_ready: projection_ready,
        exterior_to_g4_adapter_ready: projection_ready,
        hook_to_g4_adapter_ready: projection_ready,
        target_bianchi_ready: true,
        launch_ready: projection_ready,
        blockers,
    }
}

pub fn launch_raw_three_channel_g4_bianchi() -> Result<RawThreeChannelG4BianchiReport, String> {
    let plan = raw_three_channel_g4_bianchi_launch_plan();
    if !plan.launch_ready {
        return Err(format!(
            "raw three-channel G4/Bianchi launch is blocked: {}",
            plan.blockers.join("; ")
        ));
    }
    let decomposition = corrected_gamma_four_decomposition();
    let trace = build_trace_g4_operator(&decomposition.trace_lambda_three);
    let lambda_five = build_lambda_five_g4_operator(&decomposition.exterior_lambda_five);
    let hook = build_hook_g4_operator(&decomposition.hook_10010);
    let g4_channels = [
        polynomial_rows(&trace),
        polynomial_rows(&lambda_five),
        polynomial_rows(&hook),
    ];
    let trace_bianchi = apply_bianchi(&trace);
    let lambda_five_bianchi = apply_bianchi(&lambda_five);
    let hook_bianchi = apply_bianchi(&hook);
    let bianchi_channels = [
        trace_bianchi.clone(),
        lambda_five_bianchi.clone(),
        hook_bianchi.clone(),
    ];
    let g4_coefficient_rank_mod_prime = aligned_coefficient_rank(&g4_channels, PROOF_PRIME);
    let bianchi_coefficient_rank_mod_prime =
        aligned_coefficient_rank(&bianchi_channels, PROOF_PRIME);
    let symbolic_rows_sha256 = bianchi_rows_sha256(&bianchi_channels);
    let g4_nonzero_rows_per_channel =
        std::array::from_fn(|channel| g4_channels[channel].iter().map(BTreeMap::len).sum());
    let bianchi_nonzero_rows_per_channel =
        std::array::from_fn(|channel| bianchi_channels[channel].iter().map(BTreeMap::len).sum());
    let (adapter_equivariance_checks, adapter_equivariance_residuals) =
        adapter_equivariance_residuals(true, false);
    let trace_bianchi_residual_rows = bianchi_nonzero_rows_per_channel[0];
    let lambda_five_bianchi_residual_rows = bianchi_nonzero_rows_per_channel[1];
    let hook_bianchi_residual_rows = bianchi_nonzero_rows_per_channel[2];
    let all_g4_columns_independent = g4_coefficient_rank_mod_prime == 3;
    let bianchi_kernel_dimension = 3 - bianchi_coefficient_rank_mod_prime;
    let exact_kernel_basis =
        if trace_bianchi_residual_rows == 0 && bianchi_coefficient_rank_mod_prime == 2 {
            vec![[1, 0, 0]]
        } else {
            Vec::new()
        };
    let diagnostic_completed = true;
    let passed = adapter_equivariance_residuals == 0
        && all_g4_columns_independent
        && trace_bianchi_residual_rows == 0
        && lambda_five_bianchi_residual_rows > 0
        && hook_bianchi_residual_rows > 0
        && bianchi_coefficient_rank_mod_prime == 2
        && bianchi_kernel_dimension == 1;
    Ok(RawThreeChannelG4BianchiReport {
        schema_version: "adynkra-11d-raw-three-channel-g4-bianchi-v1",
        source_dimension: SOURCE_DIMENSION,
        coefficient_columns: [
            "p wedge Gamma4-trace Lambda3",
            "i_p Gamma4-exterior Lambda5",
            "p_e Gamma4-hook H_[4]{}^e",
        ],
        g4_nonzero_rows_per_channel,
        bianchi_nonzero_rows_per_channel,
        g4_coefficient_rank_mod_prime,
        bianchi_coefficient_rank_mod_prime,
        measured_raw_rank: plan.measured_raw_rank,
        measured_projection_ranks: [
            plan.measured_trace_rank,
            plan.measured_exterior_rank,
            plan.measured_hook_rank,
        ],
        projector_reconstruction_residuals: plan.projector_reconstruction_residuals,
        hook_trace_residuals: plan.hook_trace_residuals,
        hook_exterior_residuals: plan.hook_exterior_residuals,
        trace_bianchi_residual_rows,
        lambda_five_bianchi_residual_rows,
        hook_bianchi_residual_rows,
        adapter_equivariance_checks,
        adapter_equivariance_residuals,
        proof_prime: PROOF_PRIME,
        all_g4_columns_independent,
        bianchi_kernel_dimension,
        exact_kernel_basis,
        symbolic_rows_sha256,
        module_source_sha256: format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "eleven_dimensional_gamma24_source_variance.rs"
            ))
        ),
        legacy_normalization_report_sha256: LEGACY_NORMALIZATION_REPORT_SHA256,
        legacy_candidate_stream_sha256: LEGACY_CANDIDATE_STREAM_SHA256,
        legacy_teleparallel_stream_sha256: LEGACY_TELEPARALLEL_STREAM_SHA256,
        legacy_residual_stream_sha256: LEGACY_RESIDUAL_STREAM_SHA256,
        diagnostic_completed,
        passed,
        boundary: "This diagnostic proves exact tensor equivariance of the three raw momentum adapters, their coefficient rank on the complete canonical source basis, and the exact target Bianchi rank. It does not identify a raw channel with the physical four-form, fix normalization against the gravitino, impose target gauge K, or prove irreducibility.",
    })
}

fn atomic_report_last(path: &Path, report: &RawThreeChannelG4BianchiReport) -> io::Result<()> {
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("immutable report already exists: {}", path.display()),
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(temporary, path)
}

pub fn write_raw_three_channel_g4_bianchi_artifact(
    path: &Path,
) -> io::Result<RawThreeChannelG4BianchiReport> {
    let report = launch_raw_three_channel_g4_bianchi()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw three-channel G4/Bianchi report failed its acceptance gates",
        ));
    }
    atomic_report_last(path, &report)?;
    Ok(report)
}

pub fn validate_raw_three_channel_g4_bianchi_artifact(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let observed: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let replay = launch_raw_three_channel_g4_bianchi()?;
    let expected = serde_json::to_value(&replay)
        .map_err(|error| format!("serialize replay report: {error}"))?;
    if observed != expected {
        return Err("durable raw three-channel report differs from exact replay".to_string());
    }
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrected_gamma_two_and_four_are_exactly_the_same_lambda_three_ray() {
        let report = verify_source_variance();
        assert_eq!(report.coordinates_checked, 1_689_600);
        assert_eq!(report.gamma_two_exterior_nonzero_entries, 23_040);
        assert_eq!(report.gamma_four_trace_nonzero_entries, 23_040);
        assert_eq!(report.exact_identity_residual_entries, 0);
        assert_eq!(report.gamma_two_rank_mod_prime, 165);
        assert_eq!(report.gamma_four_rank_mod_prime, 165);
        assert!(report.passed);
    }

    #[test]
    fn charge_and_h_hat_variance_gates_are_exact() {
        let report = verify_source_variance();
        assert_eq!(report.charge_antisymmetry_residual_entries, 0);
        assert_eq!(report.charge_square_residual_entries, 0);
        assert_eq!(report.gamma_trace_residual_entries, 0);
    }

    #[test]
    fn legacy_raised_gamma_is_preserved_only_as_a_mismatch_witness() {
        let report = verify_source_variance();
        assert!(report.legacy_identity_residual_entries > 0);
        assert_eq!(
            report.legacy_first_mismatch,
            Some(ConventionMismatchWitness {
                derivative_spinor: 0,
                h_hat_ordinal: 0,
                target_three_form_mask: (1 << 0) | (1 << 1) | (1 << 10),
                three_times_gamma_two_exterior: -2,
                gamma_four_trace: 0,
                residual_against_legacy_common_ratio_three: 2,
            })
        );
        assert_eq!(
            report.legacy_normalization_report_sha256,
            LEGACY_NORMALIZATION_REPORT_SHA256
        );
    }

    #[test]
    fn missing_lorentz_metric_and_mutated_time_component_fail_identity() {
        let basis = masks_of_degree(3);
        let mut no_metric_two = Vec::with_capacity(SOURCE_DIMENSION);
        let mut flipped_two = Vec::with_capacity(SOURCE_DIMENSION);
        let mut flipped_four = Vec::with_capacity(SOURCE_DIMENSION);
        for derivative in 0..SPINOR_DIMENSION {
            for h_hat in 0..H_HAT_DIMENSION {
                no_metric_two.push(total_antisymmetric_part(
                    2,
                    &raw_form_vector_column(2, derivative, h_hat, true, false),
                    false,
                ));
                flipped_two.push(total_antisymmetric_part(
                    2,
                    &raw_form_vector_column(2, derivative, h_hat, true, true),
                    true,
                ));
                flipped_four.push(mixed_trace(&raw_form_vector_column(
                    4, derivative, h_hat, true, true,
                )));
            }
        }
        let no_metric_two = sparse_operator_from_columns(&basis, no_metric_two);
        let flipped_two = sparse_operator_from_columns(&basis, flipped_two);
        let flipped_four = sparse_operator_from_columns(&basis, flipped_four);
        let corrected_four = corrected_gamma_four_decomposition().trace_lambda_three;
        assert_eq!(
            scaled_identity_residuals(&no_metric_two, &corrected_four, 1),
            14_400
        );
        assert_eq!(
            scaled_identity_residuals(&flipped_two, &flipped_four, 1),
            52_800
        );
    }

    #[test]
    fn gamma_four_decomposition_is_typed_and_three_channel_diagnostic_launches() {
        let maps = corrected_gamma_four_decomposition();
        assert_eq!(maps.raw_form_vector.input_dimension, SOURCE_DIMENSION);
        assert_eq!(maps.raw_form_vector.output_dimension, 3_630);
        assert_eq!(maps.trace_lambda_three.output_dimension, 165);
        assert_eq!(maps.exterior_lambda_five.output_dimension, 462);
        assert_eq!(maps.hook_10010.output_dimension, 3_630);
        let plan = raw_three_channel_g4_bianchi_launch_plan();
        assert!(plan.trace_to_g4_adapter_ready);
        assert!(plan.exterior_to_g4_adapter_ready);
        assert!(plan.hook_to_g4_adapter_ready);
        assert!(plan.launch_ready);
        assert_eq!(plan.measured_trace_rank, 165);
        assert_eq!(plan.measured_exterior_rank, 462);
        assert_eq!(plan.measured_hook_rank, 3_003);
        assert_eq!(plan.measured_raw_rank, 3_630);
        assert_eq!(plan.projector_reconstruction_residuals, 0);
        assert_eq!(plan.hook_trace_residuals, 0);
        assert_eq!(plan.hook_exterior_residuals, 0);
        let report = launch_raw_three_channel_g4_bianchi().unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.adapter_equivariance_residuals, 0);
        assert_eq!(report.g4_coefficient_rank_mod_prime, 3);
        assert_eq!(report.bianchi_coefficient_rank_mod_prime, 2);
        assert_eq!(report.bianchi_kernel_dimension, 1);
        assert_eq!(report.exact_kernel_basis, vec![[1, 0, 0]]);
        assert_eq!(report.trace_bianchi_residual_rows, 0);
        assert!(report.lambda_five_bianchi_residual_rows > 0);
        assert!(report.hook_bianchi_residual_rows > 0);
        assert!(report.diagnostic_completed);
        assert!(report.passed);
    }

    #[test]
    fn momentum_adapter_metric_mutations_fail_exact_equivariance() {
        let (_, correct) = adapter_equivariance_residuals(true, false);
        let (_, lambda_without_metric) = adapter_equivariance_residuals(false, false);
        let (_, hook_with_metric) = adapter_equivariance_residuals(true, true);
        assert_eq!(correct, 0);
        assert!(lambda_without_metric > 0);
        assert!(hook_with_metric > 0);
    }

    #[test]
    fn one_coefficient_mutation_breaks_the_trace_bianchi_identity() {
        let decomposition = corrected_gamma_four_decomposition();
        let mut trace = build_trace_g4_operator(&decomposition.trace_lambda_three);
        let column = trace
            .columns
            .iter_mut()
            .find(|column| !column.is_empty())
            .expect("nonzero trace channel");
        let key = column.keys().next().cloned().expect("nonzero trace row");
        column
            .get_mut(&key)
            .unwrap()
            .add_assign(&ExactQi::from_rational(1, 7));
        let residuals = apply_bianchi(&trace)
            .iter()
            .map(BTreeMap::len)
            .sum::<usize>();
        assert!(residuals > 0);
    }

    #[test]
    fn durable_report_is_published_last_and_replays_exactly() {
        let root =
            std::env::temp_dir().join(format!("gamma24-three-channel-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let path = root.join("report.json");
        let report = write_raw_three_channel_g4_bianchi_artifact(&path).unwrap();
        assert!(report.passed);
        assert!(validate_raw_three_channel_g4_bianchi_artifact(&path).is_ok());
        assert!(write_raw_three_channel_g4_bianchi_artifact(&path).is_err());
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["bianchi_coefficient_rank_mod_prime"] = serde_json::json!(1);
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(validate_raw_three_channel_g4_bianchi_artifact(&path).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
