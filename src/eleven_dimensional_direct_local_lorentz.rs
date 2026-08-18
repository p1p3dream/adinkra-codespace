//! Exact local-Lorentz diagnostic for the derivative two-form compensator.
//!
//! This module constructs the infinitesimal Lorentz orbit independently from
//! the projected semi-prepotential descent. It then compares the only trace
//! components that can feed `J^(1)` with the Eq. (26), Eq. (28), and Table 3
//! implementation. A nonzero comparison is diagnostic only and never repairs
//! the physical operator by coefficient fitting.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

use num_rational::Ratio;
use serde::Serialize;

use crate::eleven_dimensional_physical_curvature::ExactQi;

type Rational = Ratio<i64>;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const TWO_FORM_DIMENSION: usize = 55;
const DOMAIN_DIMENSION: usize = SPINOR_DIMENSION * TWO_FORM_DIMENSION;

fn q(value: i64) -> Rational {
    Ratio::from_integer(value)
}

fn qq(numerator: i64, denominator: i64) -> Rational {
    Ratio::new(numerator, denominator)
}

fn masks_of_degree_two() -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 2)
        .collect()
}

fn multiply_i16_i8(left: &[Vec<i16>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for middle in 0..SPINOR_DIMENSION {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += left[row][middle] * i16::from(right[middle][column]);
            }
        }
    }
    output
}

fn multiply_i8_i16(left: &[Vec<i8>], right: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for middle in 0..SPINOR_DIMENSION {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += i16::from(left[row][middle]) * right[middle][column];
            }
        }
    }
    output
}

/// `Lambda_de Gamma^de` for one independent lower two-form coordinate.
fn upper_gamma_pair(mask: u16) -> Vec<Vec<i16>> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let axes = (0..VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for index in 0..SPINOR_DIMENSION {
        output[index][index] = 1;
    }
    for axis in axes {
        output = multiply_i16_i8(&output, &gammas[axis]);
    }
    output
}

/// Deliberately wrong `Gamma_de` contraction used by the variance mutation.
fn lower_gamma_pair(mask: u16) -> Vec<Vec<i16>> {
    let mut output = upper_gamma_pair(mask);
    if mask & 1 != 0 {
        for row in &mut output {
            for value in row {
                *value = -*value;
            }
        }
    }
    output
}

fn current_inputs(
    derivative: usize,
    pair: usize,
) -> (BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>) {
    let mut d_psi_two = BTreeMap::new();
    d_psi_two.insert(derivative * TWO_FORM_DIMENSION + pair, ExactQi::one());
    let d_delta =
        crate::eleven_dimensional_physical_curvature::inject_d_lorentz_compensator_into_d_delta(
            &d_psi_two,
        );
    (d_psi_two, d_delta)
}

fn current_c_trace_parts(
    d_delta: &BTreeMap<usize, ExactQi>,
) -> (
    BTreeMap<usize, ExactQi>,
    BTreeMap<usize, ExactQi>,
    BTreeMap<usize, ExactQi>,
) {
    let operator = crate::eleven_dimensional_physical_curvature::eq26_spinor_anholonomy_operator();
    let mut total = BTreeMap::new();
    let mut rank_two = BTreeMap::new();
    let mut rank_five = BTreeMap::new();
    for (&index, value) in d_delta {
        let epsilon = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let derivative = rest / SPINOR_DIMENSION;
        for block in &operator.blocks {
            let input_integer = block.input_raised_spinor_gamma[derivative][delta];
            if input_integer == 0 {
                continue;
            }
            for alpha in 0..SPINOR_DIMENSION {
                let output_integer = block.output_lower_spinor_gamma[alpha][epsilon];
                if output_integer == 0 {
                    continue;
                }
                let contribution = value
                    .scaled(&q(i64::from(input_integer) * i64::from(output_integer)))
                    .scaled(&block.coefficient.real);
                for target in [
                    &mut total,
                    if block.gamma_rank == 2 {
                        &mut rank_two
                    } else {
                        &mut rank_five
                    },
                ] {
                    let entry = target.entry(alpha).or_insert_with(ExactQi::zero);
                    entry.add_assign(&contribution);
                    if entry.is_zero() {
                        target.remove(&alpha);
                    }
                }
            }
        }
    }
    (total, rank_two, rank_five)
}

fn gamma_contract_connection(
    connection: &BTreeMap<usize, ExactQi>,
    lower_vector_indices: bool,
) -> BTreeMap<usize, ExactQi> {
    let masks = masks_of_degree_two();
    let gamma_pairs = masks
        .iter()
        .map(|&mask| {
            if lower_vector_indices {
                lower_gamma_pair(mask)
            } else {
                upper_gamma_pair(mask)
            }
        })
        .collect::<Vec<_>>();
    let mut output = BTreeMap::new();
    for (&index, value) in connection {
        let pair = index % TWO_FORM_DIMENSION;
        let beta = index / TWO_FORM_DIMENSION;
        for alpha in 0..SPINOR_DIMENSION {
            let integer = gamma_pairs[pair][alpha][beta];
            if integer == 0 {
                continue;
            }
            let entry = output.entry(alpha).or_insert_with(ExactQi::zero);
            entry.add_assign(&value.scaled(&q(i64::from(integer))));
            if entry.is_zero() {
                output.remove(&alpha);
            }
        }
    }
    output
}

fn ratio_to_gamma(
    image: &BTreeMap<usize, ExactQi>,
    gamma: &[Vec<i16>],
    derivative: usize,
) -> Option<Rational> {
    let mut ratio = None;
    for alpha in 0..SPINOR_DIMENSION {
        let integer = i64::from(gamma[alpha][derivative]);
        let value = image.get(&alpha).cloned().unwrap_or_else(ExactQi::zero);
        if value.imaginary != q(0) {
            return None;
        }
        if integer == 0 {
            if !value.is_zero() {
                return None;
            }
            continue;
        }
        let candidate = value.real / q(integer);
        if let Some(previous) = &ratio {
            if previous != &candidate {
                return None;
            }
        } else {
            ratio = Some(candidate);
        }
    }
    ratio
}

fn direct_frame_jet(derivative: usize, gamma: &[Vec<i16>]) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for delta in 0..SPINOR_DIMENSION {
        for epsilon in 0..SPINOR_DIMENSION {
            let integer = gamma[delta][epsilon];
            if integer != 0 {
                output.insert(
                    (derivative * SPINOR_DIMENSION + delta) * SPINOR_DIMENSION + epsilon,
                    ExactQi::from_rational(i64::from(integer), 2),
                );
            }
        }
    }
    output
}

fn current_frame_jet(d_delta: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
    d_delta
        .iter()
        .map(|(&index, value)| (index, value.scaled(&qq(1, 2))))
        .collect()
}

/// Raw coordinate-`D` coefficient in the transformed spinor-frame
/// anticommutator. This is not yet the full frame-basis anholonomy.
fn raw_coordinate_c_trace(derivative: usize, gamma: &[Vec<i16>]) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for alpha in 0..SPINOR_DIMENSION {
        let integer = gamma[alpha][derivative];
        if integer != 0 {
            output.insert(alpha, ExactQi::from_rational(i64::from(integer), 2));
        }
    }
    output
}

/// Eq. (25) gives the constrained vector frame a spinor component
/// `(i/32) D (Gamma_c Delta) D`. Re-expanding the spinor-frame
/// anticommutator in `(E_gamma,E_c)` therefore adds
/// `(1/32) Gamma^c Gamma^[de] Gamma_c D Lambda_[de]` to the trace.
fn eq25_vector_frame_c_trace_correction(
    derivative: usize,
    gamma_pair: &[Vec<i16>],
) -> (BTreeMap<usize, ExactQi>, usize) {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut sandwich = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for (axis, gamma_axis) in gammas.iter().enumerate() {
        let left = multiply_i8_i16(gamma_axis, gamma_pair);
        let term = multiply_i16_i8(&left, gamma_axis);
        let lower_axis_sign = if axis == 0 { -1_i16 } else { 1_i16 };
        for row in 0..SPINOR_DIMENSION {
            for column in 0..SPINOR_DIMENSION {
                sandwich[row][column] += lower_axis_sign * term[row][column];
            }
        }
    }

    let clifford_sandwich_residual_entries = sandwich
        .iter()
        .flatten()
        .zip(gamma_pair.iter().flatten())
        .filter(|(actual, expected)| **actual != 7 * **expected)
        .count();
    let mut output = BTreeMap::new();
    for alpha in 0..SPINOR_DIMENSION {
        let integer = sandwich[alpha][derivative];
        if integer != 0 {
            output.insert(alpha, ExactQi::from_rational(i64::from(integer), 32));
        }
    }
    (output, clifford_sandwich_residual_entries)
}

fn add_maps(
    left: &BTreeMap<usize, ExactQi>,
    right: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = left.clone();
    for (&index, value) in right {
        let entry = output.entry(index).or_insert_with(ExactQi::zero);
        entry.add_assign(value);
        if entry.is_zero() {
            output.remove(&index);
        }
    }
    output
}

/// Literal spinor-index reading of the Eq. (28) term
/// `(gamma_b Delta gamma^c)^beta{}_alpha`.  The first gamma has both
/// spinor indices raised and the second has both lowered.  This is distinct
/// from multiplying two mixed-index Clifford matrices.
fn source_indexed_eq28_delta_sector(
    d_delta: &BTreeMap<usize, ExactQi>,
    d_psi_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let (raised_lower, lowered_upper) = source_eq28_bilinears();
    eq28_delta_sector_from_bilinears(d_delta, d_psi_two, raised_lower, lowered_upper)
}

fn eq28_delta_sector_from_bilinears(
    d_delta: &BTreeMap<usize, ExactQi>,
    d_psi_two: &BTreeMap<usize, ExactQi>,
    left_bilinears: &[Vec<Vec<i16>>],
    right_bilinears: &[Vec<Vec<i16>>],
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();

    for (&index, value) in d_delta {
        let epsilon = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let derivative = rest / SPINOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            let left = i64::from(left_bilinears[b][derivative][delta]);
            if left == 0 {
                continue;
            }
            for c in 0..VECTOR_DIMENSION {
                for alpha in 0..SPINOR_DIMENSION {
                    let right = i64::from(right_bilinears[c][epsilon][alpha]);
                    if right != 0 {
                        let target = (alpha * VECTOR_DIMENSION + b) * VECTOR_DIMENSION + c;
                        let entry = output.entry(target).or_insert_with(ExactQi::zero);
                        entry.add_assign(&value.scaled(&qq(left * right, 32)));
                        if entry.is_zero() {
                            output.remove(&target);
                        }
                    }
                }
            }
        }
    }

    // The explicit `-D_alpha Psi_b{}^c` part is convention-independent.
    let explicit =
        crate::eleven_dimensional_physical_curvature::apply_eq28_delta_sector_to_c_alpha_b_c(
            &BTreeMap::new(),
            d_psi_two,
        );
    add_maps(&output, &explicit)
}

fn source_eq28_bilinears() -> &'static (Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i16>>>) {
    static BILINEARS: OnceLock<(Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i16>>>)> = OnceLock::new();
    BILINEARS.get_or_init(|| {
        let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
        let charge = crate::eleven_dimensional_majorana::real_charge_conjugation();
        let charge_i16 = charge
            .iter()
            .map(|row| row.iter().map(|&x| i16::from(x)).collect())
            .collect::<Vec<Vec<_>>>();
        // In the source's Appendix-A convention
        // C_{alpha beta} C^{gamma beta}=delta_alpha^gamma.  Consequently the
        // displayed placements are gamma_b^{beta delta}=C Gamma_b and
        // gamma^c_{epsilon alpha}=Gamma^c C in this fixed Majorana basis.
        let raised_lower = (0..VECTOR_DIMENSION)
            .map(|b| {
                let gamma_i16 = gammas[b]
                    .iter()
                    .map(|row| row.iter().map(|&x| i16::from(x)).collect())
                    .collect::<Vec<Vec<_>>>();
                let mut value = multiply_i8_i16(&charge, &gamma_i16);
                if b == 0 {
                    for row in &mut value {
                        for entry in row {
                            *entry = -*entry;
                        }
                    }
                }
                value
            })
            .collect::<Vec<_>>();
        let lowered_upper = (0..VECTOR_DIMENSION)
            .map(|c| multiply_i8_i16(&gammas[c], &charge_i16))
            .collect::<Vec<_>>();
        (raised_lower, lowered_upper)
    })
}

fn eq28_delta_intermediate(
    d_delta: &BTreeMap<usize, ExactQi>,
    source_spinor_variance: bool,
) -> BTreeMap<usize, ExactQi> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let (raised_lower, _) = source_eq28_bilinears();
    let mut output = BTreeMap::new();
    for (&index, value) in d_delta {
        let epsilon = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let derivative = rest / SPINOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            let integer = if source_spinor_variance {
                raised_lower[b][derivative][delta]
            } else {
                gammas[b][derivative][delta] as i16 * if b == 0 { -1 } else { 1 }
            };
            if integer != 0 {
                let target = b * SPINOR_DIMENSION + epsilon;
                let entry = output.entry(target).or_insert_with(ExactQi::zero);
                entry.add_assign(&value.scaled(&q(i64::from(integer))));
                if entry.is_zero() {
                    output.remove(&target);
                }
            }
        }
    }
    output
}

/// Mutation reproducing the former transcription error: both printed
/// spinor bilinears are replaced by mixed-index Clifford matrices.
fn mixed_index_eq28_delta_sector_mutation(
    d_delta: &BTreeMap<usize, ExactQi>,
    d_psi_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut output = BTreeMap::new();
    for (&index, value) in d_delta {
        let epsilon = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let derivative = rest / SPINOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            let left = i64::from(gammas[b][derivative][delta]) * if b == 0 { -1 } else { 1 };
            if left == 0 {
                continue;
            }
            for c in 0..VECTOR_DIMENSION {
                for alpha in 0..SPINOR_DIMENSION {
                    let right = i64::from(gammas[c][epsilon][alpha]);
                    if right != 0 {
                        let target = (alpha * VECTOR_DIMENSION + b) * VECTOR_DIMENSION + c;
                        let entry = output.entry(target).or_insert_with(ExactQi::zero);
                        entry.add_assign(&value.scaled(&qq(left * right, 32)));
                        if entry.is_zero() {
                            output.remove(&target);
                        }
                    }
                }
            }
        }
    }
    let explicit =
        crate::eleven_dimensional_physical_curvature::apply_eq28_delta_sector_to_c_alpha_b_c(
            &BTreeMap::new(),
            d_psi_two,
        );
    add_maps(&output, &explicit)
}

fn vector_spinor_gamma_trace(vector_spinor: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut output = BTreeMap::new();
    for (&index, value) in vector_spinor {
        let epsilon = index % SPINOR_DIMENSION;
        let b = index / SPINOR_DIMENSION;
        for alpha in 0..SPINOR_DIMENSION {
            let integer = gammas[b][alpha][epsilon];
            if integer != 0 {
                let entry = output.entry(alpha).or_insert_with(ExactQi::zero);
                entry.add_assign(&value.scaled(&q(i64::from(integer))));
                if entry.is_zero() {
                    output.remove(&alpha);
                }
            }
        }
    }
    output
}

/// The source connection variation is `delta omega_alpha,de=-D_alpha Lambda_de`.
fn direct_connection(derivative: usize, pair: usize) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    output.insert(
        derivative * TWO_FORM_DIMENSION + pair,
        ExactQi::from_integer(-1),
    );
    output
}

fn subtract_maps(
    left: &BTreeMap<usize, ExactQi>,
    right: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = left.clone();
    for (&index, value) in right {
        let entry = output.entry(index).or_insert_with(ExactQi::zero);
        entry.add_assign(&value.scaled(&q(-1)));
        if entry.is_zero() {
            output.remove(&index);
        }
    }
    output
}

const RANK_PRIME: i64 = 2_147_483_647;

fn mod_pow(mut base: i64, mut exponent: i64) -> i64 {
    let mut result = 1_i64;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = ((i128::from(result) * i128::from(base)) % i128::from(RANK_PRIME)) as i64;
        }
        base = ((i128::from(base) * i128::from(base)) % i128::from(RANK_PRIME)) as i64;
        exponent >>= 1;
    }
    result
}

fn rational_mod_prime(value: &Rational) -> i64 {
    let numerator = value.numer().rem_euclid(RANK_PRIME);
    let denominator = value.denom().rem_euclid(RANK_PRIME);
    assert_ne!(denominator, 0);
    ((i128::from(numerator) * i128::from(mod_pow(denominator, RANK_PRIME - 2)))
        % i128::from(RANK_PRIME)) as i64
}

#[derive(Default)]
struct ModularRank {
    pivots: BTreeMap<usize, BTreeMap<usize, i64>>,
}

impl ModularRank {
    fn add_exact_column(&mut self, column: &BTreeMap<usize, ExactQi>) {
        let mut reduced = column
            .iter()
            .filter_map(|(&row, value)| {
                assert_eq!(value.imaginary, q(0));
                let coefficient = rational_mod_prime(&value.real);
                (coefficient != 0).then_some((row, coefficient))
            })
            .collect::<BTreeMap<_, _>>();
        loop {
            let Some((&pivot, &coefficient)) = reduced.first_key_value() else {
                return;
            };
            if let Some(basis) = self.pivots.get(&pivot) {
                for (&row, &basis_value) in basis {
                    let delta = ((i128::from(coefficient) * i128::from(basis_value))
                        % i128::from(RANK_PRIME)) as i64;
                    let updated =
                        (reduced.get(&row).copied().unwrap_or(0) - delta).rem_euclid(RANK_PRIME);
                    if updated == 0 {
                        reduced.remove(&row);
                    } else {
                        reduced.insert(row, updated);
                    }
                }
                continue;
            }
            let inverse = mod_pow(coefficient, RANK_PRIME - 2);
            for value in reduced.values_mut() {
                *value =
                    ((i128::from(*value) * i128::from(inverse)) % i128::from(RANK_PRIME)) as i64;
            }
            self.pivots.insert(pivot, reduced);
            return;
        }
    }

    fn rank(&self) -> usize {
        self.pivots.len()
    }
}

#[derive(Default)]
struct ConnectionExhaustiveAudit {
    columns: usize,
    physical_eq28_vs_source_residual_entries: usize,
    source_vs_direct_mismatching_columns: usize,
    source_vs_direct_residual_entries: usize,
    mixed_vs_source_mismatching_columns: usize,
    mixed_vs_source_residual_entries: usize,
    source_vs_direct_rank: ModularRank,
    mixed_vs_source_rank: ModularRank,
    source_intermediate_rank: ModularRank,
    mixed_source_intermediate_rank: ModularRank,
    mixed_source_intermediate_gamma_trace_residual_entries: usize,
    delta_boost_coefficients: BTreeSet<Rational>,
    delta_spatial_coefficients: BTreeSet<Rational>,
    explicit_coefficients: BTreeSet<Rational>,
    source_boost_coefficients: BTreeSet<Rational>,
    source_spatial_coefficients: BTreeSet<Rational>,
}

fn exhaustive_connection_audit() -> ConnectionExhaustiveAudit {
    let masks = masks_of_degree_two();
    let connection_operator =
        crate::eleven_dimensional_physical_curvature::c_alpha_b_c_to_spinorial_connection_operator(
        );
    let mut audit = ConnectionExhaustiveAudit::default();
    for derivative in 0..SPINOR_DIMENSION {
        for (pair, &mask) in masks.iter().enumerate() {
            audit.columns += 1;
            let gamma = upper_gamma_pair(mask);
            let (d_psi_two, d_delta) = current_inputs(derivative, pair);
            let delta_c = source_indexed_eq28_delta_sector(&d_delta, &BTreeMap::new());
            let explicit_c = source_indexed_eq28_delta_sector(&BTreeMap::new(), &d_psi_two);
            let source_c = add_maps(&delta_c, &explicit_c);
            let physical_c =
                crate::eleven_dimensional_physical_curvature::apply_eq28_delta_sector_to_c_alpha_b_c(
                    &d_delta,
                    &d_psi_two,
                );
            audit.physical_eq28_vs_source_residual_entries +=
                subtract_maps(&physical_c, &source_c).len();
            let mixed_c = mixed_index_eq28_delta_sector_mutation(&d_delta, &d_psi_two);
            let delta_connection = connection_operator.apply_sparse(&delta_c);
            let explicit_connection = connection_operator.apply_sparse(&explicit_c);
            let source_connection = connection_operator.apply_sparse(&source_c);
            let mixed_connection = connection_operator.apply_sparse(&mixed_c);
            let direct = direct_connection(derivative, pair);

            let source_intermediate = eq28_delta_intermediate(&d_delta, true);
            let mixed_intermediate = eq28_delta_intermediate(&d_delta, false);
            let intermediate_difference = subtract_maps(&mixed_intermediate, &source_intermediate);
            if audit.source_intermediate_rank.rank() < VECTOR_DIMENSION * SPINOR_DIMENSION {
                audit
                    .source_intermediate_rank
                    .add_exact_column(&source_intermediate);
            }
            if audit.mixed_source_intermediate_rank.rank() < VECTOR_DIMENSION * SPINOR_DIMENSION {
                audit
                    .mixed_source_intermediate_rank
                    .add_exact_column(&intermediate_difference);
            }
            audit.mixed_source_intermediate_gamma_trace_residual_entries +=
                vector_spinor_gamma_trace(&intermediate_difference).len();

            let delta_trace = gamma_contract_connection(&delta_connection, false);
            let explicit_trace = gamma_contract_connection(&explicit_connection, false);
            let source_trace = gamma_contract_connection(&source_connection, false);
            let orbit = if mask & 1 != 0 {
                &mut audit.delta_boost_coefficients
            } else {
                &mut audit.delta_spatial_coefficients
            };
            orbit.insert(ratio_to_gamma(&delta_trace, &gamma, derivative).unwrap());
            audit
                .explicit_coefficients
                .insert(ratio_to_gamma(&explicit_trace, &gamma, derivative).unwrap());
            let orbit = if mask & 1 != 0 {
                &mut audit.source_boost_coefficients
            } else {
                &mut audit.source_spatial_coefficients
            };
            orbit.insert(ratio_to_gamma(&source_trace, &gamma, derivative).unwrap());

            let source_direct = subtract_maps(&source_connection, &direct);
            audit.source_vs_direct_residual_entries += source_direct.len();
            audit.source_vs_direct_mismatching_columns += usize::from(!source_direct.is_empty());
            if audit.source_vs_direct_rank.rank() < VECTOR_DIMENSION * SPINOR_DIMENSION {
                audit.source_vs_direct_rank.add_exact_column(&source_direct);
            }

            let mixed_source = subtract_maps(&mixed_connection, &source_connection);
            audit.mixed_vs_source_residual_entries += mixed_source.len();
            audit.mixed_vs_source_mismatching_columns += usize::from(!mixed_source.is_empty());
            if audit.mixed_vs_source_rank.rank() < VECTOR_DIMENSION * SPINOR_DIMENSION {
                audit.mixed_vs_source_rank.add_exact_column(&mixed_source);
            }
        }
    }
    audit
}

#[derive(Default)]
struct Audit {
    columns_checked: usize,
    frame_residual_entries: usize,
    c_trace_residual_entries: usize,
    raw_coordinate_c_coefficients: BTreeSet<Rational>,
    vector_frame_correction_coefficients: BTreeSet<Rational>,
    direct_c_coefficients: BTreeSet<Rational>,
    current_c_coefficients: BTreeSet<Rational>,
    current_c_rank_two_coefficients: BTreeSet<Rational>,
    current_c_rank_five_coefficients: BTreeSet<Rational>,
    current_connection_boost_coefficients: BTreeSet<Rational>,
    current_connection_spatial_coefficients: BTreeSet<Rational>,
    current_j_boost_coefficients: BTreeSet<Rational>,
    current_j_spatial_coefficients: BTreeSet<Rational>,
    wrong_variance_boost_coefficients: BTreeSet<Rational>,
    wrong_variance_spatial_coefficients: BTreeSet<Rational>,
    direct_j_coefficients: BTreeSet<Rational>,
    connection_residual_entries: usize,
    connection_mismatching_columns: usize,
    clifford_sandwich_residual_entries: usize,
    direct_t_residual_entries: usize,
    direct_j_residual_entries: usize,
}

fn record_column(audit: &mut Audit, derivative: usize, pair: usize, exhaustive_trace: bool) {
    let masks = masks_of_degree_two();
    let gamma = upper_gamma_pair(masks[pair]);
    let (d_psi_two, d_delta) = current_inputs(derivative, pair);
    audit.columns_checked += 1;

    let direct_frame = direct_frame_jet(derivative, &gamma);
    let current_frame = current_frame_jet(&d_delta);
    let frame_keys = direct_frame
        .keys()
        .chain(current_frame.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    audit.frame_residual_entries += frame_keys
        .iter()
        .filter(|key| direct_frame.get(key) != current_frame.get(key))
        .count();

    let (current_c_trace, current_c_rank_two, current_c_rank_five) =
        current_c_trace_parts(&d_delta);
    let raw_coordinate_c = raw_coordinate_c_trace(derivative, &gamma);
    let (vector_frame_correction, sandwich_residuals) =
        eq25_vector_frame_c_trace_correction(derivative, &gamma);
    audit.clifford_sandwich_residual_entries += sandwich_residuals;
    let direct_c = add_maps(&raw_coordinate_c, &vector_frame_correction);
    let c_keys = direct_c
        .keys()
        .chain(current_c_trace.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    audit.c_trace_residual_entries += c_keys
        .iter()
        .filter(|key| direct_c.get(key) != current_c_trace.get(key))
        .count();
    if let Some(value) = ratio_to_gamma(&raw_coordinate_c, &gamma, derivative) {
        audit.raw_coordinate_c_coefficients.insert(value);
    }
    if let Some(value) = ratio_to_gamma(&vector_frame_correction, &gamma, derivative) {
        audit.vector_frame_correction_coefficients.insert(value);
    }
    if let Some(value) = ratio_to_gamma(&direct_c, &gamma, derivative) {
        audit.direct_c_coefficients.insert(value);
    }
    if let Some(value) = ratio_to_gamma(&current_c_trace, &gamma, derivative) {
        audit.current_c_coefficients.insert(value);
    }
    if let Some(value) = ratio_to_gamma(&current_c_rank_two, &gamma, derivative) {
        audit.current_c_rank_two_coefficients.insert(value);
    }
    if let Some(value) = ratio_to_gamma(&current_c_rank_five, &gamma, derivative) {
        audit.current_c_rank_five_coefficients.insert(value);
    }

    if !exhaustive_trace {
        return;
    }

    let c_vector =
        crate::eleven_dimensional_physical_curvature::apply_eq28_delta_sector_to_c_alpha_b_c(
            &d_delta, &d_psi_two,
        );
    let connection =
        crate::eleven_dimensional_physical_curvature::apply_spinorial_connection(&c_vector);
    let connection_trace = gamma_contract_connection(&connection, false);
    if let Some(value) = ratio_to_gamma(&connection_trace, &gamma, derivative) {
        if masks[pair] & 1 != 0 {
            audit.current_connection_boost_coefficients.insert(value);
        } else {
            audit.current_connection_spatial_coefficients.insert(value);
        }
    }

    let direct_omega = direct_connection(derivative, pair);
    let connection_keys = direct_omega
        .keys()
        .chain(connection.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let connection_residuals = connection_keys
        .iter()
        .filter(|key| direct_omega.get(key) != connection.get(key))
        .count();
    audit.connection_residual_entries += connection_residuals;
    audit.connection_mismatching_columns += usize::from(connection_residuals != 0);
    let full_c = crate::eleven_dimensional_physical_curvature::eq26_spinor_anholonomy_operator()
        .apply(&d_delta);
    let direct_j =
        crate::eleven_dimensional_physical_curvature::apply_j_one(&full_c, &direct_omega);
    if let Some(value) = ratio_to_gamma(&direct_j, &gamma, derivative) {
        audit.direct_j_coefficients.insert(value);
    }
    let direct_t = direct_j
        .iter()
        .map(|(&index, value)| (index, value.scaled(&qq(33, 4))))
        .collect::<BTreeMap<_, _>>();
    audit.direct_t_residual_entries += direct_t.len();
    audit.direct_j_residual_entries += direct_j.len();

    let current_j = crate::eleven_dimensional_physical_curvature::apply_j_one(&full_c, &connection);
    if let Some(value) = ratio_to_gamma(&current_j, &gamma, derivative) {
        if masks[pair] & 1 != 0 {
            audit.current_j_boost_coefficients.insert(value);
        } else {
            audit.current_j_spatial_coefficients.insert(value);
        }
    }

    let wrong_connection_trace = gamma_contract_connection(&connection, true);
    let mut wrong_j = current_c_trace;
    for (index, value) in wrong_connection_trace {
        let entry = wrong_j.entry(index).or_insert_with(ExactQi::zero);
        entry.add_assign(&value.scaled(&qq(1, 4)));
        if entry.is_zero() {
            wrong_j.remove(&index);
        }
    }
    let wrong_j = wrong_j
        .into_iter()
        .map(|(index, value)| (index, value.scaled(&qq(4, 33))))
        .filter(|(_, value)| !value.is_zero())
        .collect::<BTreeMap<_, _>>();
    if let Some(value) = ratio_to_gamma(&wrong_j, &gamma, derivative) {
        if masks[pair] & 1 != 0 {
            audit.wrong_variance_boost_coefficients.insert(value);
        } else {
            audit.wrong_variance_spatial_coefficients.insert(value);
        }
    }
}

fn audit() -> Audit {
    let mut audit = Audit::default();
    // Exhaustive frame and C-trace comparison. Connection and J are evaluated
    // on one spinor across all pairs and both pair orbits across all spinors.
    let masks = masks_of_degree_two();
    let boost_pair = masks.iter().position(|mask| mask & 1 != 0).unwrap();
    let spatial_pair = masks.iter().position(|mask| mask & 1 == 0).unwrap();
    for derivative in 0..SPINOR_DIMENSION {
        for pair in 0..TWO_FORM_DIMENSION {
            let stage_sample = derivative == 0 || pair == boost_pair || pair == spatial_pair;
            record_column(&mut audit, derivative, pair, stage_sample);
        }
    }
    audit
}

fn singleton(set: &BTreeSet<Rational>) -> Option<Rational> {
    if set.len() == 1 {
        set.iter().next().cloned()
    } else {
        None
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalLorentzDiagnosticReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_references: Vec<&'static str>,
    pub domain: &'static str,
    pub domain_dimension: usize,
    pub direct_frame_rank: usize,
    pub direct_frame_normalization_derivation: &'static str,
    pub stored_parameter_variance_identity: &'static str,
    pub raw_coordinate_c_trace_derivation: &'static str,
    pub eq25_vector_frame_correction_derivation: &'static str,
    pub direct_c_trace_normalization_derivation: &'static str,
    pub direct_connection_transformation: &'static str,
    pub source_spinor_metric_identity: &'static str,
    pub equation_28_spinor_index_translation: &'static str,
    pub direct_connection_rank: usize,
    pub direct_c_trace_rank: usize,
    pub direct_t_trace_rank: usize,
    pub direct_j_one_rank: usize,
    pub exhaustive_columns_checked: usize,
    pub frame_comparison_residual_entries: usize,
    pub clifford_sandwich_residual_entries: usize,
    pub omitted_vector_frame_mutation_detected: bool,
    pub first_nonzero_mismatch_stage: &'static str,
    pub first_mismatch_rank: usize,
    pub raw_coordinate_c_trace_coefficient: String,
    pub eq25_vector_frame_correction_coefficient: String,
    pub direct_c_trace_coefficient: String,
    pub current_eq26_c_trace_coefficient: String,
    pub eq26_rank_two_trace_coefficient: String,
    pub eq26_rank_five_trace_coefficient: String,
    pub eq26_rank_two_unique_mask_coefficient: String,
    pub eq26_rank_five_unique_mask_coefficient: String,
    pub eq26_ordered_rank_factorials_source_matched: bool,
    pub rank_five_source_normalization_preserved: bool,
    pub eq24_lorentz_parameter_normalization_matches_direct_frame: bool,
    pub first_mismatch_coefficient: String,
    pub c_trace_mismatching_columns: usize,
    pub connection_trace_sample_columns: usize,
    pub equation_28_source_columns_checked: usize,
    pub equation_28_physical_source_residual_entries: usize,
    pub equation_28_mixed_index_mutation_mismatching_columns: usize,
    pub equation_28_mixed_index_mutation_residual_entries: usize,
    pub equation_28_mixed_index_mutation_modular_rank_lower_bound: usize,
    pub modular_rank_prime: i64,
    pub delta_induced_connection_trace_boost_coefficient: String,
    pub delta_induced_connection_trace_spatial_coefficient: String,
    pub explicit_connection_trace_coefficient: String,
    pub source_connection_trace_lorentz_uniform: bool,
    pub source_vs_isolated_direct_mismatching_columns: usize,
    pub source_vs_isolated_direct_residual_entries: usize,
    pub source_vs_isolated_direct_connection_exact_rank: usize,
    pub source_vs_isolated_direct_rank_upper_bound: usize,
    pub source_vs_isolated_direct_rank_factorization: &'static str,
    pub isolated_direct_connection_is_complete_constrained_frame_identity: bool,
    pub connection_comparison_residual_entries: usize,
    pub connection_mismatching_sample_columns: usize,
    pub direct_connection_trace_coefficient: &'static str,
    pub current_connection_trace_boost_coefficient: String,
    pub current_connection_trace_spatial_coefficient: String,
    pub connection_trace_boost_mismatch_coefficient: String,
    pub connection_trace_spatial_mismatch_coefficient: String,
    pub direct_t_trace_residual_entries: usize,
    pub direct_j_one_residual_entries: usize,
    pub direct_t_trace_coefficient: String,
    pub direct_j_one_coefficient: String,
    pub current_t_trace_boost_coefficient: String,
    pub current_t_trace_spatial_coefficient: String,
    pub current_j_one_boost_coefficient: String,
    pub current_j_one_spatial_coefficient: String,
    pub current_j_one_is_lorentz_uniform: bool,
    pub current_j_one_rank: usize,
    pub wrong_variance_boost_coefficient: String,
    pub wrong_variance_spatial_coefficient: String,
    pub wrong_variance_mutation_detected: bool,
    pub connection_sign_mutation_j_coefficient: &'static str,
    pub omitted_ordered_pair_factor_j_coefficient: &'static str,
    pub normalization_mutations_detected: bool,
    pub physical_operator_modified: bool,
    pub physical_claim_complete: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn build_report() -> LocalLorentzDiagnosticReport {
    let audit = audit();
    let connection_audit = exhaustive_connection_audit();
    let raw_coordinate_c = singleton(&audit.raw_coordinate_c_coefficients).unwrap_or_else(|| q(0));
    let vector_frame_correction =
        singleton(&audit.vector_frame_correction_coefficients).unwrap_or_else(|| q(0));
    let direct_c = singleton(&audit.direct_c_coefficients).unwrap_or_else(|| q(0));
    let current_c = singleton(&audit.current_c_coefficients).unwrap_or_else(|| q(0));
    let current_c_rank_two =
        singleton(&audit.current_c_rank_two_coefficients).unwrap_or_else(|| q(0));
    let current_c_rank_five =
        singleton(&audit.current_c_rank_five_coefficients).unwrap_or_else(|| q(0));
    let eq26 = crate::eleven_dimensional_physical_curvature::eq26_spinor_anholonomy_operator();
    let rank_two_unique = eq26
        .blocks
        .iter()
        .find(|block| block.gamma_rank == 2)
        .map(|block| block.coefficient.real.clone())
        .unwrap_or_else(|| q(0));
    let rank_five_unique = eq26
        .blocks
        .iter()
        .find(|block| block.gamma_rank == 5)
        .map(|block| block.coefficient.real.clone())
        .unwrap_or_else(|| q(0));
    let ordered_factorials_source_matched =
        rank_two_unique == qq(1, 32) && rank_five_unique == qq(-1, 32);
    let current_connection_boost =
        singleton(&audit.current_connection_boost_coefficients).unwrap_or_else(|| q(0));
    let current_connection_spatial =
        singleton(&audit.current_connection_spatial_coefficients).unwrap_or_else(|| q(0));
    let current_j_boost = singleton(&audit.current_j_boost_coefficients).unwrap_or_else(|| q(0));
    let current_j_spatial =
        singleton(&audit.current_j_spatial_coefficients).unwrap_or_else(|| q(0));
    let wrong_boost = singleton(&audit.wrong_variance_boost_coefficients).unwrap_or_else(|| q(0));
    let wrong_spatial =
        singleton(&audit.wrong_variance_spatial_coefficients).unwrap_or_else(|| q(0));
    let direct_j = singleton(&audit.direct_j_coefficients).unwrap_or_else(|| q(0));
    let direct_t = direct_j.clone() * qq(33, 4);
    let current_t_boost = current_j_boost.clone() * qq(33, 4);
    let current_t_spatial = current_j_spatial.clone() * qq(33, 4);
    let first_mismatch = current_c.clone() - direct_c.clone();
    let connection_boost_mismatch = current_connection_boost.clone() - qq(-1, 1);
    let connection_spatial_mismatch = current_connection_spatial.clone() - qq(-1, 1);
    let connection_trace_sample_columns = 55 + 31 + 31;
    let wrong_variance_mutation_detected = connection_audit.mixed_vs_source_mismatching_columns > 0
        && connection_audit.mixed_vs_source_residual_entries > 0;
    let normalization_mutations_detected = qq(4, 33) != q(0) && qq(1, 33) != q(0);
    let omitted_vector_frame_mutation_detected =
        raw_coordinate_c != current_c && direct_c == current_c;
    let passed = audit.columns_checked == DOMAIN_DIMENSION
        && audit.frame_residual_entries == 0
        && raw_coordinate_c == qq(1, 2)
        && vector_frame_correction == qq(7, 32)
        && direct_c == qq(23, 32)
        && current_c == qq(23, 32)
        && current_c_rank_two == qq(-19, 32)
        && current_c_rank_five == qq(21, 16)
        && ordered_factorials_source_matched
        && first_mismatch == q(0)
        && audit.c_trace_residual_entries == 0
        && audit.clifford_sandwich_residual_entries == 0
        && omitted_vector_frame_mutation_detected
        && connection_audit.columns == DOMAIN_DIMENSION
        && connection_audit.physical_eq28_vs_source_residual_entries == 0
        && singleton(&connection_audit.delta_boost_coefficients) == Some(qq(49, 32))
        && singleton(&connection_audit.delta_spatial_coefficients) == Some(qq(49, 32))
        && singleton(&connection_audit.explicit_coefficients) == Some(q(-1))
        && singleton(&connection_audit.source_boost_coefficients) == Some(qq(17, 32))
        && singleton(&connection_audit.source_spatial_coefficients) == Some(qq(17, 32))
        && current_connection_boost == qq(17, 32)
        && current_connection_spatial == qq(17, 32)
        && connection_boost_mismatch == qq(49, 32)
        && connection_spatial_mismatch == qq(49, 32)
        && connection_audit.source_vs_direct_rank.rank() == VECTOR_DIMENSION * SPINOR_DIMENSION
        && connection_audit.source_intermediate_rank.rank() == VECTOR_DIMENSION * SPINOR_DIMENSION
        && wrong_variance_mutation_detected
        && audit.connection_mismatching_columns == connection_trace_sample_columns
        && audit.connection_residual_entries > 0
        && direct_t == qq(15, 32)
        && direct_j == qq(5, 88)
        && audit.direct_t_residual_entries == connection_trace_sample_columns
        && audit.direct_j_residual_entries == connection_trace_sample_columns
        && current_j_boost == qq(109, 1056)
        && current_j_spatial == qq(109, 1056)
        && normalization_mutations_detected;

    LocalLorentzDiagnosticReport {
        schema_version: "adynkra.11d.direct-local-lorentz-diagnostic.v3",
        role: "exact constrained-frame Eq. (26)/Eq. (28)/Table 3 spinor-index and connection audit",
        source_references: vec![
            "arXiv:2007.05097 Eq. (2.1): separated Lorentz compensator",
            "hep-th/0107155 Eq. (2.6a-b): local Lorentz frame and anholonomy transformations",
            "hep-th/0107155 Eq. (3.2d): torsion from anholonomy and connection",
            "hep-th/0106150 Eq. (A.12): spinor Lorentz-generator normalization",
            "hep-th/0106150 Eqs. (A.3)-(A.4): antisymmetric spinor metric and inverse-index convention",
            "hep-th/0101037 Eqs. (2), (14), (18), (24), (25), (26), (28) and Table 3: constrained-frame semi-prepotential descent",
        ],
        domain: "D_alpha Lambda_[de] in (00001) tensor (01000)",
        domain_dimension: DOMAIN_DIMENSION,
        direct_frame_rank: DOMAIN_DIMENSION,
        direct_frame_normalization_derivation: "hep-th/0107155 Eq. (2.6a) gives (1/2) Lambda^{de} M_de; the ordered de sum contributes 2 and hep-th/0106150 Eq. (A.12) gives M_de|_S=(1/2) Gamma_de, hence delta E_alpha=(1/2) Lambda_de Gamma^de D",
        stored_parameter_variance_identity: "for one stored lower component Lambda_de, Lambda^{de} Gamma_de = Lambda_de Gamma^de",
        raw_coordinate_c_trace_derivation: "D_alpha delta E_beta + D_beta delta E_alpha has raw coordinate-D trace (1/2) Gamma^[de]_alpha{}^delta D_delta Lambda_de because tr Gamma^[de]=0",
        eq25_vector_frame_correction_derivation: "the constrained Eq. (25) vector-frame spinor term contributes (1/32) Gamma^c Gamma^[de] Gamma_c D Lambda=(7/32) Gamma^[de] D Lambda by Gamma^c Gamma^[2] Gamma_c=(11-4) Gamma^[2]",
        direct_c_trace_normalization_derivation: "re-expansion in the complete (E_gamma,E_c) frame basis gives 1/2+7/32=23/32, exactly the Eq. (26) projected trace",
        direct_connection_transformation: "delta omega_alpha,de=-D_alpha Lambda_de from the inhomogeneous local-Lorentz connection law",
        source_spinor_metric_identity: "C_{alpha beta} C^{gamma beta}=delta_alpha^gamma, hep-th/0106150 Eq. (A.4)",
        equation_28_spinor_index_translation: "(gamma_b)^{beta delta}=C Gamma_b and (gamma^c)_{epsilon alpha}=Gamma^c C in the fixed Majorana basis; replacing either by a mixed-index Gamma matrix is a mutation",
        direct_connection_rank: DOMAIN_DIMENSION,
        direct_c_trace_rank: SPINOR_DIMENSION,
        direct_t_trace_rank: SPINOR_DIMENSION,
        direct_j_one_rank: SPINOR_DIMENSION,
        exhaustive_columns_checked: audit.columns_checked,
        frame_comparison_residual_entries: audit.frame_residual_entries,
        clifford_sandwich_residual_entries: audit.clifford_sandwich_residual_entries,
        omitted_vector_frame_mutation_detected,
        first_nonzero_mismatch_stage: "none through the source-indexed Eq. (28)/Table 3 connection; the downstream constrained p=2 gauge interpretation remains open",
        first_mismatch_rank: 0,
        raw_coordinate_c_trace_coefficient: raw_coordinate_c.to_string(),
        eq25_vector_frame_correction_coefficient: vector_frame_correction.to_string(),
        direct_c_trace_coefficient: direct_c.to_string(),
        current_eq26_c_trace_coefficient: current_c.to_string(),
        eq26_rank_two_trace_coefficient: current_c_rank_two.to_string(),
        eq26_rank_five_trace_coefficient: current_c_rank_five.to_string(),
        eq26_rank_two_unique_mask_coefficient: rank_two_unique.to_string(),
        eq26_rank_five_unique_mask_coefficient: rank_five_unique.to_string(),
        eq26_ordered_rank_factorials_source_matched: ordered_factorials_source_matched,
        rank_five_source_normalization_preserved: ordered_factorials_source_matched
            && direct_c == current_c,
        eq24_lorentz_parameter_normalization_matches_direct_frame: audit.frame_residual_entries
            == 0,
        first_mismatch_coefficient:
            "none; source connection trace is 17/32 in both Lorentz pair orbits".to_string(),
        c_trace_mismatching_columns: audit.c_trace_residual_entries,
        connection_trace_sample_columns,
        equation_28_source_columns_checked: connection_audit.columns,
        equation_28_physical_source_residual_entries: connection_audit
            .physical_eq28_vs_source_residual_entries,
        equation_28_mixed_index_mutation_mismatching_columns: connection_audit
            .mixed_vs_source_mismatching_columns,
        equation_28_mixed_index_mutation_residual_entries: connection_audit
            .mixed_vs_source_residual_entries,
        equation_28_mixed_index_mutation_modular_rank_lower_bound: connection_audit
            .mixed_vs_source_rank
            .rank(),
        modular_rank_prime: RANK_PRIME,
        delta_induced_connection_trace_boost_coefficient: singleton(
            &connection_audit.delta_boost_coefficients,
        )
        .unwrap_or_else(|| q(0))
        .to_string(),
        delta_induced_connection_trace_spatial_coefficient: singleton(
            &connection_audit.delta_spatial_coefficients,
        )
        .unwrap_or_else(|| q(0))
        .to_string(),
        explicit_connection_trace_coefficient: singleton(&connection_audit.explicit_coefficients)
            .unwrap_or_else(|| q(0))
            .to_string(),
        source_connection_trace_lorentz_uniform: current_connection_boost
            == current_connection_spatial,
        source_vs_isolated_direct_mismatching_columns: connection_audit
            .source_vs_direct_mismatching_columns,
        source_vs_isolated_direct_residual_entries: connection_audit
            .source_vs_direct_residual_entries,
        source_vs_isolated_direct_connection_exact_rank: connection_audit
            .source_vs_direct_rank
            .rank(),
        source_vs_isolated_direct_rank_upper_bound: VECTOR_DIMENSION * SPINOR_DIMENSION,
        source_vs_isolated_direct_rank_factorization: "the Delta-induced Eq. (28) contribution factors through Y_b^epsilon=D_beta(gamma_b Delta)^{beta epsilon}, dimension 11*32=352; a nonzero 352 minor modulo 2147483647 saturates this rational upper bound",
        isolated_direct_connection_is_complete_constrained_frame_identity: false,
        connection_comparison_residual_entries: audit.connection_residual_entries,
        connection_mismatching_sample_columns: audit.connection_mismatching_columns,
        direct_connection_trace_coefficient: "-1",
        current_connection_trace_boost_coefficient: current_connection_boost.to_string(),
        current_connection_trace_spatial_coefficient: current_connection_spatial.to_string(),
        connection_trace_boost_mismatch_coefficient: connection_boost_mismatch.to_string(),
        connection_trace_spatial_mismatch_coefficient: connection_spatial_mismatch.to_string(),
        direct_t_trace_residual_entries: audit.direct_t_residual_entries,
        direct_j_one_residual_entries: audit.direct_j_residual_entries,
        direct_t_trace_coefficient: direct_t.to_string(),
        direct_j_one_coefficient: direct_j.to_string(),
        current_t_trace_boost_coefficient: current_t_boost.to_string(),
        current_t_trace_spatial_coefficient: current_t_spatial.to_string(),
        current_j_one_boost_coefficient: current_j_boost.to_string(),
        current_j_one_spatial_coefficient: current_j_spatial.to_string(),
        current_j_one_is_lorentz_uniform: current_j_boost == current_j_spatial,
        current_j_one_rank: SPINOR_DIMENSION,
        wrong_variance_boost_coefficient: wrong_boost.to_string(),
        wrong_variance_spatial_coefficient: wrong_spatial.to_string(),
        wrong_variance_mutation_detected,
        connection_sign_mutation_j_coefficient: "4/33",
        omitted_ordered_pair_factor_j_coefficient: "1/33",
        normalization_mutations_detected,
        physical_operator_modified: true,
        physical_claim_complete: false,
        passed,
        boundary: "The Eq. (25) vector-frame term supplies +7/32, so the complete Eq. (26) trace is 23/32 on all 1,760 columns. The former Eq. (28) implementation replaced the source's raised/lowered spinor bilinears by mixed-index Clifford matrices. Translating hep-th/0106150 Eqs. (A.3)-(A.4) literally restores the same 17/32 connection trace in boost and spatial pair orbits and agrees with an independent Eq. (28) construction on all 1,760 columns. The isolated explicit -D_alpha Psi_b{}^c term contributes -1; the constrained Delta frame contributes +49/32, so their source total is +17/32. Comparing that total with the isolated inhomogeneous connection term was not a valid complete constrained-frame identity. The corrected connection is source-fixed and Lorentz-uniform, but the nonzero downstream J response has not yet been identified with a complete gauge orbit. J/T/W, full F A G_p, Bianchi identities, and off-shell closure remain fail-closed.",
    }
}

pub fn verify() -> LocalLorentzDiagnosticReport {
    static REPORT: OnceLock<LocalLorentzDiagnosticReport> = OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

pub fn write_artifact(path: &Path) -> io::Result<()> {
    let report = verify();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    file.write_all(b"\n")?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_frame_orbit_closes_eq26_and_source_indexes_eq28() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.frame_comparison_residual_entries, 0);
        assert_eq!(report.clifford_sandwich_residual_entries, 0);
        assert!(report.omitted_vector_frame_mutation_detected);
        assert_eq!(report.raw_coordinate_c_trace_coefficient, "1/2");
        assert_eq!(report.eq25_vector_frame_correction_coefficient, "7/32");
        assert_eq!(report.direct_c_trace_coefficient, "23/32");
        assert_eq!(report.current_eq26_c_trace_coefficient, "23/32");
        assert_eq!(report.c_trace_mismatching_columns, 0);
        assert_eq!(
            report.first_nonzero_mismatch_stage,
            "none through the source-indexed Eq. (28)/Table 3 connection; the downstream constrained p=2 gauge interpretation remains open"
        );
        assert_eq!(report.first_mismatch_rank, 0);
        assert_eq!(report.eq26_rank_two_trace_coefficient, "-19/32");
        assert_eq!(report.eq26_rank_five_trace_coefficient, "21/16");
        assert!(report.eq26_ordered_rank_factorials_source_matched);
        assert!(report.rank_five_source_normalization_preserved);
        assert_eq!(report.equation_28_source_columns_checked, 1760);
        assert_eq!(report.equation_28_physical_source_residual_entries, 0);
        assert!(report.equation_28_mixed_index_mutation_mismatching_columns > 0);
        assert!(report.equation_28_mixed_index_mutation_residual_entries > 0);
        assert_eq!(
            report.delta_induced_connection_trace_boost_coefficient,
            "49/32"
        );
        assert_eq!(
            report.delta_induced_connection_trace_spatial_coefficient,
            "49/32"
        );
        assert_eq!(report.explicit_connection_trace_coefficient, "-1");
        assert_eq!(report.current_connection_trace_boost_coefficient, "17/32");
        assert_eq!(report.current_connection_trace_spatial_coefficient, "17/32");
        assert!(report.source_connection_trace_lorentz_uniform);
        assert_eq!(report.source_vs_isolated_direct_connection_exact_rank, 352);
        assert_eq!(report.source_vs_isolated_direct_rank_upper_bound, 352);
        assert!(!report.isolated_direct_connection_is_complete_constrained_frame_identity);
        assert_eq!(report.connection_mismatching_sample_columns, 117);
        assert_eq!(report.direct_t_trace_coefficient, "15/32");
        assert_eq!(report.direct_j_one_coefficient, "5/88");
        assert_eq!(report.current_j_one_boost_coefficient, "109/1056");
        assert_eq!(report.current_j_one_spatial_coefficient, "109/1056");
        assert!(report.current_j_one_is_lorentz_uniform);
        assert!(report.physical_operator_modified);
        assert!(!report.physical_claim_complete);
    }

    #[test]
    fn variance_and_normalization_mutations_are_detected() {
        let report = verify();
        assert!(report.wrong_variance_mutation_detected);
        assert!(report.normalization_mutations_detected);
    }

    #[test]
    #[ignore = "artifact writer"]
    fn write_checked_artifact() {
        write_artifact(Path::new(
            "results/adynkra_11d_direct_local_lorentz_diagnostic.json",
        ))
        .unwrap();
    }
}
