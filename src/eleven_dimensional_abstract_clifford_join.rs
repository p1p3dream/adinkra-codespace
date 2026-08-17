//! Exact Clifford intertwiners in the abstract B5 Chevalley weight bases.
//!
//! This module avoids a Cartesian change of basis.  It solves the intertwining
//! equations for the five simple-root raising and lowering operators directly
//! on `V tensor S`, then checks the resulting gamma trace against every state
//! in the repository's deterministic `(10001)` target basis.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::Ratio;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

type Rational = Ratio<i64>;
type BigRational = Ratio<BigInt>;
type SmallGaussian = Complex<Rational>;
type Weight = [i8; 5];
type SparseRow = BTreeMap<usize, Rational>;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const AMBIENT_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const LEVEL16_ALL_COUPLINGS_CERTIFICATE: &str =
    include_str!("../results/adynkra_11d_level16_couplings_all.json");
const LEVEL16_ALL_COUPLINGS_CERTIFICATE_SHA256: &str =
    "bada78574729dec6700dbd27979af87c444d5bdeb5a4ec9cddc9f5c2151a4547";
const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];

fn q(value: i64) -> Rational {
    Ratio::from_integer(value)
}

fn bq(value: i64) -> BigRational {
    Ratio::from_integer(BigInt::from(value))
}

fn add(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|index| left[index] + right[index])
}

fn subtract(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|index| left[index] - right[index])
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

fn vector_weights() -> [Weight; VECTOR_DIMENSION] {
    let mut weights = [[0_i8; 5]; VECTOR_DIMENSION];
    for axis in 0..5 {
        weights[2 * axis][axis] = 2;
        weights[2 * axis + 1][axis] = -2;
    }
    weights
}

#[derive(Clone, Copy)]
enum Direction {
    Raising,
    Lowering,
}

fn spinor_action(
    index: usize,
    root: usize,
    direction: Direction,
    weights: &[Weight; SPINOR_DIMENSION],
) -> Option<(usize, i64)> {
    let target = match direction {
        Direction::Raising => add(weights[index], SIMPLE_ROOTS[root]),
        Direction::Lowering => subtract(weights[index], SIMPLE_ROOTS[root]),
    };
    weights
        .iter()
        .position(|weight| *weight == target)
        .map(|target| (target, 1))
}

fn vector_action(
    index: usize,
    root: usize,
    direction: Direction,
    weights: &[Weight; VECTOR_DIMENSION],
) -> Option<(usize, i64)> {
    let weight = weights[index];
    let mut target = weight;
    match direction {
        Direction::Lowering if root < 4 => {
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
        }
        Direction::Raising if root < 4 => {
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
        }
        Direction::Lowering if weight[4] == 2 => Some((10, 1)),
        Direction::Lowering if weight == [0; 5] => Some((9, 2)),
        Direction::Raising if weight == [0; 5] => Some((8, 2)),
        Direction::Raising if weight[4] == -2 => Some((10, 1)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct GammaVariable {
    output_spinor: usize,
    vector: usize,
    input_spinor: usize,
}

fn gamma_variables() -> Vec<GammaVariable> {
    let vectors = vector_weights();
    let spinors = spinor_weights();
    let mut variables = Vec::new();
    for output_spinor in 0..SPINOR_DIMENSION {
        for vector in 0..VECTOR_DIMENSION {
            for input_spinor in 0..SPINOR_DIMENSION {
                if add(vectors[vector], spinors[input_spinor]) == spinors[output_spinor] {
                    variables.push(GammaVariable {
                        output_spinor,
                        vector,
                        input_spinor,
                    });
                }
            }
        }
    }
    variables
}

fn add_row_entry(row: &mut SparseRow, column: usize, value: i64) {
    *row.entry(column).or_insert_with(|| q(0)) += q(value);
    if row[&column] == q(0) {
        row.remove(&column);
    }
}

fn gamma_intertwining_rows(variables: &[GammaVariable]) -> (Vec<SparseRow>, [[usize; 5]; 2]) {
    let vectors = vector_weights();
    let spinors = spinor_weights();
    let lookup = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            (
                (
                    variable.output_spinor,
                    variable.vector,
                    variable.input_spinor,
                ),
                index,
            )
        })
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::new();
    let mut counts = [[0_usize; 5]; 2];
    for (direction_index, direction) in [Direction::Raising, Direction::Lowering]
        .into_iter()
        .enumerate()
    {
        for root in 0..5 {
            let before = rows.len();
            for vector in 0..VECTOR_DIMENSION {
                for input_spinor in 0..SPINOR_DIMENSION {
                    for output_spinor in 0..SPINOR_DIMENSION {
                        let mut row = SparseRow::new();
                        if let Some((next_vector, coefficient)) =
                            vector_action(vector, root, direction, &vectors)
                        {
                            if let Some(&column) =
                                lookup.get(&(output_spinor, next_vector, input_spinor))
                            {
                                add_row_entry(&mut row, column, coefficient);
                            }
                        }
                        if let Some((next_spinor, coefficient)) =
                            spinor_action(input_spinor, root, direction, &spinors)
                        {
                            if let Some(&column) = lookup.get(&(output_spinor, vector, next_spinor))
                            {
                                add_row_entry(&mut row, column, coefficient);
                            }
                        }
                        for source_output in 0..SPINOR_DIMENSION {
                            if spinor_action(source_output, root, direction, &spinors)
                                == Some((output_spinor, 1))
                            {
                                if let Some(&column) =
                                    lookup.get(&(source_output, vector, input_spinor))
                                {
                                    add_row_entry(&mut row, column, -1);
                                }
                            }
                        }
                        if !row.is_empty() {
                            rows.push(row);
                        }
                    }
                }
            }
            counts[direction_index][root] = rows.len() - before;
        }
    }
    (rows, counts)
}

fn echelon(rows: &[SparseRow]) -> BTreeMap<usize, SparseRow> {
    let mut pivots = BTreeMap::<usize, SparseRow>::new();
    for source in rows {
        let mut row = source.clone();
        while let Some((&pivot, coefficient)) = row.first_key_value() {
            let coefficient = coefficient.clone();
            if let Some(existing) = pivots.get(&pivot) {
                let existing = existing.clone();
                for (column, value) in existing {
                    let entry = row.entry(column).or_insert_with(|| q(0));
                    *entry -= coefficient.clone() * value;
                    if *entry == q(0) {
                        row.remove(&column);
                    }
                }
            } else {
                for value in row.values_mut() {
                    *value /= coefficient.clone();
                }
                pivots.insert(pivot, row);
                break;
            }
        }
    }
    pivots
}

fn one_dimensional_kernel(rows: &[SparseRow], width: usize) -> (usize, Vec<Rational>) {
    let pivots = echelon(rows);
    let free = (0..width)
        .filter(|column| !pivots.contains_key(column))
        .collect::<Vec<_>>();
    assert_eq!(free.len(), 1, "expected a unique intertwiner up to scale");
    let mut solution = vec![q(0); width];
    solution[free[0]] = q(1);
    for (&pivot, row) in pivots.iter().rev() {
        solution[pivot] = -row
            .iter()
            .filter(|(column, _)| **column != pivot)
            .map(|(column, value)| value.clone() * solution[*column].clone())
            .sum::<Rational>();
    }
    if solution.iter().find(|value| **value != q(0)).unwrap() < &q(0) {
        for value in &mut solution {
            *value = -value.clone();
        }
    }
    (pivots.len(), solution)
}

fn residual_rows(rows: &[SparseRow], solution: &[Rational]) -> usize {
    rows.iter()
        .filter(|row| {
            row.iter()
                .map(|(column, coefficient)| coefficient.clone() * solution[*column].clone())
                .sum::<Rational>()
                != q(0)
        })
        .count()
}

#[derive(Clone)]
struct AbstractGammaMap {
    variables: Vec<GammaVariable>,
    coefficients: Vec<Rational>,
    matrices: Vec<Vec<Vec<Rational>>>,
}

fn solve_gamma_trace() -> (AbstractGammaMap, usize, Vec<SparseRow>, [[usize; 5]; 2]) {
    let variables = gamma_variables();
    let (rows, row_counts) = gamma_intertwining_rows(&variables);
    let (rank, coefficients) = one_dimensional_kernel(&rows, variables.len());
    let mut matrices = vec![vec![vec![q(0); SPINOR_DIMENSION]; SPINOR_DIMENSION]; VECTOR_DIMENSION];
    for (variable, coefficient) in variables.iter().zip(&coefficients) {
        matrices[variable.vector][variable.output_spinor][variable.input_spinor] =
            coefficient.clone();
    }
    (
        AbstractGammaMap {
            variables,
            coefficients,
            matrices,
        },
        rank,
        rows,
        row_counts,
    )
}

fn sparse_matrix_rank(matrix: &[Vec<Rational>]) -> usize {
    let rows = matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, value)| **value != q(0))
                .map(|(column, value)| (column, value.clone()))
                .collect::<SparseRow>()
        })
        .collect::<Vec<_>>();
    echelon(&rows).len()
}

fn target_kernel_residuals(gamma: &AbstractGammaMap) -> (usize, usize) {
    let target = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let mut terms_checked = 0;
    let mut residual_entries = 0;
    for state in target {
        let mut trace = vec![q(0); SPINOR_DIMENSION];
        for term in state.raw_terms {
            terms_checked += 1;
            let coefficient = Ratio::new(term.numerator, term.denominator);
            for (output, value) in trace.iter_mut().enumerate() {
                *value += gamma.matrices[term.vector_weight_index][output]
                    [term.spinor_weight_index]
                    .clone()
                    * coefficient.clone();
            }
        }
        residual_entries += trace.iter().filter(|value| **value != q(0)).count();
    }
    (terms_checked, residual_entries)
}

fn multiply(left: &[Vec<Rational>], right: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    let mut product = vec![vec![q(0); right[0].len()]; left.len()];
    for (row, left_row) in left.iter().enumerate() {
        for (pivot, left_value) in left_row.iter().enumerate() {
            if *left_value == q(0) {
                continue;
            }
            for (column, right_value) in right[pivot].iter().enumerate() {
                if *right_value != q(0) {
                    product[row][column] += left_value.clone() * right_value.clone();
                }
            }
        }
    }
    product
}

fn matrix_add(left: &[Vec<Rational>], right: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(left, right)| left.clone() + right.clone())
                .collect()
        })
        .collect()
}

fn matrix_subtract(left: &[Vec<Rational>], right: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(left, right)| left.clone() - right.clone())
                .collect()
        })
        .collect()
}

fn scaled(matrix: &[Vec<Rational>], scalar: Rational) -> Vec<Vec<Rational>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.clone() * scalar.clone())
                .collect()
        })
        .collect()
}

fn identity(dimension: usize) -> Vec<Vec<Rational>> {
    let mut result = vec![vec![q(0); dimension]; dimension];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = q(1);
    }
    result
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricEntry {
    pub left_vector_weight_index: usize,
    pub right_vector_weight_index: usize,
    pub numerator: i64,
    pub denominator: i64,
}

fn clifford_certificate(gamma: &AbstractGammaMap) -> (usize, usize, Vec<MetricEntry>) {
    let mut metric = vec![vec![q(0); VECTOR_DIMENSION]; VECTOR_DIMENSION];
    let mut residual_entries = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in left..VECTOR_DIMENSION {
            let anticommutator = matrix_add(
                &multiply(&gamma.matrices[left], &gamma.matrices[right]),
                &multiply(&gamma.matrices[right], &gamma.matrices[left]),
            );
            let scalar = anticommutator[0][0].clone() / q(2);
            metric[left][right] = scalar.clone();
            metric[right][left] = scalar.clone();
            let expected = scaled(&identity(SPINOR_DIMENSION), scalar * q(2));
            residual_entries += anticommutator
                .iter()
                .zip(expected)
                .map(|(observed, expected)| {
                    observed
                        .iter()
                        .zip(expected)
                        .filter(|(observed, expected)| **observed != *expected)
                        .count()
                })
                .sum::<usize>();
        }
    }
    let entries = (0..VECTOR_DIMENSION)
        .flat_map(|left| {
            let metric = &metric;
            (left..VECTOR_DIMENSION).filter_map(move |right| {
                let value = &metric[left][right];
                (value != &q(0)).then(|| MetricEntry {
                    left_vector_weight_index: left,
                    right_vector_weight_index: right,
                    numerator: *value.numer(),
                    denominator: *value.denom(),
                })
            })
        })
        .collect();
    (sparse_matrix_rank(&metric), residual_entries, entries)
}

fn invariant_spinor_bilinear() -> (Vec<Vec<Rational>>, usize, usize, usize) {
    let spinors = spinor_weights();
    let variables = (0..SPINOR_DIMENSION)
        .flat_map(|left| {
            (0..SPINOR_DIMENSION)
                .filter(move |right| add(spinors[left], spinors[*right]) == [0; 5])
                .map(move |right| (left, right))
        })
        .collect::<Vec<_>>();
    let lookup = variables
        .iter()
        .enumerate()
        .map(|(index, pair)| (*pair, index))
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::new();
    for direction in [Direction::Raising, Direction::Lowering] {
        for root in 0..5 {
            for left in 0..SPINOR_DIMENSION {
                for right in 0..SPINOR_DIMENSION {
                    let mut row = SparseRow::new();
                    if let Some((next, coefficient)) =
                        spinor_action(left, root, direction, &spinors)
                    {
                        if let Some(&column) = lookup.get(&(next, right)) {
                            add_row_entry(&mut row, column, coefficient);
                        }
                    }
                    if let Some((next, coefficient)) =
                        spinor_action(right, root, direction, &spinors)
                    {
                        if let Some(&column) = lookup.get(&(left, next)) {
                            add_row_entry(&mut row, column, coefficient);
                        }
                    }
                    if !row.is_empty() {
                        rows.push(row);
                    }
                }
            }
        }
    }
    let (rank, solution) = one_dimensional_kernel(&rows, variables.len());
    let mut bilinear = vec![vec![q(0); SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for ((left, right), value) in variables.into_iter().zip(solution) {
        bilinear[left][right] = value;
    }
    let antisymmetry_residuals = (0..SPINOR_DIMENSION)
        .flat_map(|left| (0..SPINOR_DIMENSION).map(move |right| (left, right)))
        .filter(|(left, right)| {
            bilinear[*left][*right].clone() + bilinear[*right][*left].clone() != q(0)
        })
        .count();
    (bilinear, rank, rows.len(), antisymmetry_residuals)
}

fn two_form_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn wedge_pair_index(left: usize, right: usize, pairs: &[(usize, usize)]) -> Option<(usize, i64)> {
    if left == right {
        return None;
    }
    let (pair, sign) = if left < right {
        ((left, right), 1)
    } else {
        ((right, left), -1)
    };
    Some((
        pairs
            .iter()
            .position(|candidate| *candidate == pair)
            .unwrap(),
        sign,
    ))
}

fn wedge_action(
    pair_index: usize,
    root: usize,
    direction: Direction,
    vectors: &[Weight; VECTOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> Vec<(usize, i64)> {
    let (left, right) = pairs[pair_index];
    let mut output = BTreeMap::<usize, i64>::new();
    if let Some((next, coefficient)) = vector_action(left, root, direction, vectors) {
        if let Some((pair, sign)) = wedge_pair_index(next, right, pairs) {
            *output.entry(pair).or_default() += coefficient * sign;
        }
    }
    if let Some((next, coefficient)) = vector_action(right, root, direction, vectors) {
        if let Some((pair, sign)) = wedge_pair_index(left, next, pairs) {
            *output.entry(pair).or_default() += coefficient * sign;
        }
    }
    output
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .collect()
}

fn gamma_two_matrices(gamma: &AbstractGammaMap) -> Vec<Vec<Vec<Rational>>> {
    two_form_pairs()
        .into_iter()
        .map(|(left, right)| {
            scaled(
                &matrix_subtract(
                    &multiply(&gamma.matrices[left], &gamma.matrices[right]),
                    &multiply(&gamma.matrices[right], &gamma.matrices[left]),
                ),
                Ratio::new(1, 2),
            )
        })
        .collect()
}

fn gamma_two_intertwining_residuals(gamma_two: &[Vec<Vec<Rational>>]) -> (usize, usize) {
    let vectors = vector_weights();
    let spinors = spinor_weights();
    let pairs = two_form_pairs();
    let mut entries_checked = 0;
    let mut residual_entries = 0;
    for direction in [Direction::Raising, Direction::Lowering] {
        for root in 0..5 {
            for pair in 0..pairs.len() {
                for input_spinor in 0..SPINOR_DIMENSION {
                    for output_spinor in 0..SPINOR_DIMENSION {
                        entries_checked += 1;
                        let mut residual = q(0);
                        for (next_pair, coefficient) in
                            wedge_action(pair, root, direction, &vectors, &pairs)
                        {
                            residual += gamma_two[next_pair][output_spinor][input_spinor].clone()
                                * q(coefficient);
                        }
                        if let Some((next_spinor, coefficient)) =
                            spinor_action(input_spinor, root, direction, &spinors)
                        {
                            residual += gamma_two[pair][output_spinor][next_spinor].clone()
                                * q(coefficient);
                        }
                        for source_output in 0..SPINOR_DIMENSION {
                            if spinor_action(source_output, root, direction, &spinors)
                                == Some((output_spinor, 1))
                            {
                                residual -= gamma_two[pair][source_output][input_spinor].clone();
                            }
                        }
                        residual_entries += usize::from(residual != q(0));
                    }
                }
            }
        }
    }
    (entries_checked, residual_entries)
}

fn x2_equivariance_mutation_detected() -> bool {
    let (mut gamma, _, _, _) = solve_gamma_trace();
    let variable = gamma.variables[0];
    gamma.matrices[variable.vector][variable.output_spinor][variable.input_spinor] =
        -gamma.matrices[variable.vector][variable.output_spinor][variable.input_spinor].clone();
    gamma_two_intertwining_residuals(&gamma_two_matrices(&gamma)).1 > 0
}

fn gamma_two_raised(
    gamma_two: &[Vec<Vec<Rational>>],
    charge: &[Vec<Rational>],
) -> (Vec<Vec<Vec<Rational>>>, usize, usize) {
    let charge_inverse = scaled(charge, q(-1));
    let inverse_residuals = matrix_add(
        &multiply(charge, &charge_inverse),
        &scaled(&identity(SPINOR_DIMENSION), q(-1)),
    )
    .iter()
    .flatten()
    .filter(|value| **value != q(0))
    .count();
    let raised = gamma_two
        .iter()
        .map(|matrix| multiply(matrix, &charge_inverse))
        .collect::<Vec<_>>();
    let symmetry_residuals = raised
        .iter()
        .map(|matrix| {
            (0..SPINOR_DIMENSION)
                .flat_map(|left| (0..SPINOR_DIMENSION).map(move |right| (left, right)))
                .filter(|(left, right)| matrix[*left][*right] != matrix[*right][*left])
                .count()
        })
        .sum();
    (raised, inverse_residuals, symmetry_residuals)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AbstractDhTerm {
    pub derivative_spinor_weight_index: usize,
    pub target_vector_weight_index: usize,
    pub target_spinor_weight_index: usize,
    pub coefficient: Rational,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AbstractX2LeadingTerm {
    pub left_vector_weight_index: usize,
    pub right_vector_weight_index: usize,
    pub target_vector_weight_index: usize,
    pub coefficient: Rational,
}

/// Apply the source-normalized `(1/16) Gamma_[2] D H` contraction from
/// hep-th/0101037 Eq. (39), retaining abstract B5 vector-weight indices.
///
/// This is the leading gamma contraction only.  It does not add or eliminate
/// the compensator fields, and it does not apply the 429-dimensional hook
/// projector in a Cartesian Lorentz basis.
pub fn contract_gamma_two_d_h(input: &[AbstractDhTerm]) -> Vec<AbstractX2LeadingTerm> {
    let (gamma, _, _, _) = solve_gamma_trace();
    let (charge, _, _, _) = invariant_spinor_bilinear();
    let gamma_two = gamma_two_matrices(&gamma);
    let (raised, inverse_residuals, symmetry_residuals) = gamma_two_raised(&gamma_two, &charge);
    assert_eq!(inverse_residuals, 0);
    assert_eq!(symmetry_residuals, 0);
    let pairs = two_form_pairs();
    let mut output = BTreeMap::<(usize, usize), Rational>::new();
    for term in input {
        assert!(term.derivative_spinor_weight_index < SPINOR_DIMENSION);
        assert!(term.target_vector_weight_index < VECTOR_DIMENSION);
        assert!(term.target_spinor_weight_index < SPINOR_DIMENSION);
        for pair in 0..pairs.len() {
            let coefficient = raised[pair][term.derivative_spinor_weight_index]
                [term.target_spinor_weight_index]
                .clone()
                * term.coefficient.clone()
                / q(16);
            if coefficient != q(0) {
                *output
                    .entry((pair, term.target_vector_weight_index))
                    .or_insert_with(|| q(0)) += coefficient;
            }
        }
    }
    output
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != q(0))
        .map(
            |((pair, target_vector_weight_index), coefficient)| AbstractX2LeadingTerm {
                left_vector_weight_index: pairs[pair].0,
                right_vector_weight_index: pairs[pair].1,
                target_vector_weight_index,
                coefficient,
            },
        )
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetResolvedGammaTwoDhEntry {
    pub left_vector_weight_index: usize,
    pub right_vector_weight_index: usize,
    pub target_vector_weight_index: usize,
    pub parameter_component_index: usize,
    pub exterior_mask: u32,
    pub real: BigRational,
    pub imaginary: BigRational,
}

fn right_wedge_sign(mask: u32, spinor_index: usize) -> Option<i64> {
    let bit = 1_u32 << spinor_index;
    if mask & bit != 0 {
        return None;
    }
    let greater_bits = if spinor_index == 31 {
        0
    } else {
        !((1_u32 << (spinor_index + 1)) - 1)
    };
    Some(if (mask & greater_bits).count_ones() % 2 == 0 {
        1
    } else {
        -1
    })
}

/// Consume one target-resolved `D^17 Lambda` term and append the exterior
/// part of one more right-acting superspace derivative.  Terms in which the
/// derivative spinor already occurs in the exterior mask belong to the
/// momentum part of the normal form and are deliberately omitted here.
pub fn visit_gamma_two_d_h_stream_entry<F>(
    entry: &crate::eleven_dimensional_level16_couplings::TargetResolvedGaugeCompositionEntry,
    mut visit: F,
) -> io::Result<u64>
where
    F: FnMut(TargetResolvedGammaTwoDhEntry) -> io::Result<()>,
{
    assert!(entry.momentum_vector_weight_index.is_none());
    assert_eq!(entry.exterior_mask.count_ones(), 17);
    let (gamma, _, _, _) = solve_gamma_trace();
    let (charge, _, _, _) = invariant_spinor_bilinear();
    let gamma_two = gamma_two_matrices(&gamma);
    let (raised, inverse_residuals, symmetry_residuals) = gamma_two_raised(&gamma_two, &charge);
    assert_eq!(inverse_residuals, 0);
    assert_eq!(symmetry_residuals, 0);
    let pairs = two_form_pairs();
    let mut emitted = 0_u64;
    for derivative_spinor in 0..SPINOR_DIMENSION {
        let Some(wedge_sign) = right_wedge_sign(entry.exterior_mask, derivative_spinor) else {
            continue;
        };
        for (pair, (left, right)) in pairs.iter().copied().enumerate() {
            let gamma_coefficient =
                &raised[pair][derivative_spinor][entry.target_spinor_weight_index];
            if *gamma_coefficient == q(0) {
                continue;
            }
            let coefficient = Ratio::new(
                BigInt::from(*gamma_coefficient.numer() * wedge_sign),
                BigInt::from(*gamma_coefficient.denom() * 16),
            );
            let real = entry.real.clone() * coefficient.clone();
            let imaginary = entry.imaginary.clone() * coefficient;
            if real == bq(0) && imaginary == bq(0) {
                continue;
            }
            visit(TargetResolvedGammaTwoDhEntry {
                left_vector_weight_index: left,
                right_vector_weight_index: right,
                target_vector_weight_index: entry.target_vector_weight_index,
                parameter_component_index: entry.parameter_component_index,
                exterior_mask: entry.exterior_mask | (1_u32 << derivative_spinor),
                real,
                imaginary,
            })?;
            emitted += 1;
        }
    }
    Ok(emitted)
}

type HookTensor = BTreeMap<(usize, usize), Rational>;

fn gamma_metric(gamma: &AbstractGammaMap) -> Vec<Vec<Rational>> {
    let mut metric = vec![vec![q(0); VECTOR_DIMENSION]; VECTOR_DIMENSION];
    for left in 0..VECTOR_DIMENSION {
        for right in left..VECTOR_DIMENSION {
            let anticommutator = matrix_add(
                &multiply(&gamma.matrices[left], &gamma.matrices[right]),
                &multiply(&gamma.matrices[right], &gamma.matrices[left]),
            );
            let value = anticommutator[0][0].clone() / q(2);
            metric[left][right] = value.clone();
            metric[right][left] = value;
        }
    }
    metric
}

fn inverse_square_matrix(matrix: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    let dimension = matrix.len();
    let mut augmented = matrix
        .iter()
        .enumerate()
        .map(|(row, values)| {
            let mut values = values.clone();
            values.extend((0..dimension).map(|column| q(i64::from(row == column))));
            values
        })
        .collect::<Vec<_>>();
    for column in 0..dimension {
        let pivot = (column..dimension)
            .find(|row| augmented[*row][column] != q(0))
            .expect("invariant vector metric is singular");
        augmented.swap(column, pivot);
        let normalization = augmented[column][column].clone();
        for value in &mut augmented[column][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = augmented[column].clone();
        for row in 0..dimension {
            if row == column || augmented[row][column] == q(0) {
                continue;
            }
            let factor = augmented[row][column].clone();
            for entry in column..(2 * dimension) {
                augmented[row][entry] -= factor.clone() * pivot_row[entry].clone();
            }
        }
    }
    augmented
        .into_iter()
        .map(|row| row[dimension..].to_vec())
        .collect()
}

fn add_hook_component(tensor: &mut HookTensor, pair: usize, vector: usize, value: Rational) {
    if value == q(0) {
        return;
    }
    *tensor.entry((pair, vector)).or_insert_with(|| q(0)) += value;
    if tensor[&(pair, vector)] == q(0) {
        tensor.remove(&(pair, vector));
    }
}

fn triple_index_and_sign(left: usize, middle: usize, right: usize) -> Option<([usize; 3], i64)> {
    if left == middle || left == right || middle == right {
        return None;
    }
    let mut values = [left, middle, right];
    let mut sign = 1;
    for first in 0..3 {
        for second in (first + 1)..3 {
            if values[first] > values[second] {
                values.swap(first, second);
                sign = -sign;
            }
        }
    }
    Some((values, sign))
}

fn hook_wedge(tensor: &HookTensor, pairs: &[(usize, usize)]) -> BTreeMap<[usize; 3], Rational> {
    let mut output = BTreeMap::<[usize; 3], Rational>::new();
    for (&(pair, vector), coefficient) in tensor {
        let (left, right) = pairs[pair];
        let Some((triple, sign)) = triple_index_and_sign(left, right, vector) else {
            continue;
        };
        *output.entry(triple).or_insert_with(|| q(0)) += coefficient.clone() * q(sign);
        if output[&triple] == q(0) {
            output.remove(&triple);
        }
    }
    output
}

fn inject_hook_wedge(
    form: &BTreeMap<[usize; 3], Rational>,
    pairs: &[(usize, usize)],
    normalization: Rational,
) -> HookTensor {
    let mut output = HookTensor::new();
    for (&[left, middle, right], coefficient) in form {
        let terms = [
            ((left, middle), right, 1),
            ((left, right), middle, -1),
            ((middle, right), left, 1),
        ];
        for (pair, vector, sign) in terms {
            let pair = pairs
                .iter()
                .position(|candidate| *candidate == pair)
                .unwrap();
            add_hook_component(
                &mut output,
                pair,
                vector,
                coefficient.clone() * normalization.clone() * q(sign),
            );
        }
    }
    output
}

fn hook_trace(
    tensor: &HookTensor,
    pairs: &[(usize, usize)],
    metric: &[Vec<Rational>],
) -> Vec<Rational> {
    let mut output = vec![q(0); VECTOR_DIMENSION];
    for (&(pair, vector), coefficient) in tensor {
        let (left, right) = pairs[pair];
        output[left] += coefficient.clone() * metric[right][vector].clone();
        output[right] -= coefficient.clone() * metric[left][vector].clone();
    }
    output
}

fn inject_hook_trace(
    trace: &[Rational],
    metric_inverse: &[Vec<Rational>],
    pairs: &[(usize, usize)],
    normalization: Rational,
) -> HookTensor {
    let mut output = HookTensor::new();
    for (trace_vector, trace_coefficient) in trace.iter().enumerate() {
        if *trace_coefficient == q(0) {
            continue;
        }
        for basis_vector in 0..VECTOR_DIMENSION {
            if basis_vector == trace_vector {
                continue;
            }
            let (pair, wedge_sign) = wedge_pair_index(trace_vector, basis_vector, pairs).unwrap();
            for output_vector in 0..VECTOR_DIMENSION {
                let inverse_coefficient = &metric_inverse[basis_vector][output_vector];
                if *inverse_coefficient != q(0) {
                    add_hook_component(
                        &mut output,
                        pair,
                        output_vector,
                        trace_coefficient.clone()
                            * inverse_coefficient.clone()
                            * normalization.clone()
                            * q(wedge_sign),
                    );
                }
            }
        }
    }
    output
}

fn subtract_hook(left: &mut HookTensor, right: &HookTensor) {
    for (&(pair, vector), coefficient) in right {
        add_hook_component(left, pair, vector, -coefficient.clone());
    }
}

fn project_abstract_x2_hook_with_normalizations(
    tensor: &HookTensor,
    metric: &[Vec<Rational>],
    metric_inverse: &[Vec<Rational>],
    exterior_normalization: Rational,
    trace_normalization: Rational,
) -> HookTensor {
    let pairs = two_form_pairs();
    let exterior = inject_hook_wedge(&hook_wedge(tensor, &pairs), &pairs, exterior_normalization);
    let trace = inject_hook_trace(
        &hook_trace(tensor, &pairs, metric),
        metric_inverse,
        &pairs,
        trace_normalization,
    );
    let mut result = tensor.clone();
    subtract_hook(&mut result, &exterior);
    subtract_hook(&mut result, &trace);
    result
}

fn project_abstract_x2_hook(
    tensor: &HookTensor,
    metric: &[Vec<Rational>],
    metric_inverse: &[Vec<Rational>],
) -> HookTensor {
    project_abstract_x2_hook_with_normalizations(
        tensor,
        metric,
        metric_inverse,
        Ratio::new(1, 3),
        Ratio::new(1, 10),
    )
}

#[derive(Clone, Debug, Serialize)]
pub struct AbstractX2HookProjectorCertificate {
    pub ambient_dimension: usize,
    pub trace_dimension: usize,
    pub exterior_dimension: usize,
    pub hook_dimension: usize,
    pub operator_trace: String,
    pub projector_rank_from_idempotent_trace: usize,
    pub unit_columns_checked: usize,
    pub idempotence_residual_entries: usize,
    pub trace_residual_entries: usize,
    pub exterior_residual_entries: usize,
    pub chevalley_commutators_checked: usize,
    pub chevalley_commutator_residual_entries: usize,
    pub maximum_projected_column_support: usize,
    pub passed: bool,
}

fn act_hook_tensor(
    tensor: &HookTensor,
    root: usize,
    direction: Direction,
    vectors: &[Weight; VECTOR_DIMENSION],
    pairs: &[(usize, usize)],
) -> HookTensor {
    let mut output = HookTensor::new();
    for (&(pair, vector), coefficient) in tensor {
        for (next_pair, action_coefficient) in wedge_action(pair, root, direction, vectors, pairs) {
            add_hook_component(
                &mut output,
                next_pair,
                vector,
                coefficient.clone() * q(action_coefficient),
            );
        }
        if let Some((next_vector, action_coefficient)) =
            vector_action(vector, root, direction, vectors)
        {
            add_hook_component(
                &mut output,
                pair,
                next_vector,
                coefficient.clone() * q(action_coefficient),
            );
        }
    }
    output
}

fn certify_abstract_x2_hook(
    metric: &[Vec<Rational>],
    metric_inverse: &[Vec<Rational>],
) -> AbstractX2HookProjectorCertificate {
    let pairs = two_form_pairs();
    let ambient_dimension = pairs.len() * VECTOR_DIMENSION;
    let mut operator_trace = q(0);
    let mut idempotence_residual_entries = 0;
    let mut trace_residual_entries = 0;
    let mut exterior_residual_entries = 0;
    let mut maximum_projected_column_support = 0;
    for input in 0..ambient_dimension {
        let pair = input / VECTOR_DIMENSION;
        let vector = input % VECTOR_DIMENSION;
        let unit = HookTensor::from([((pair, vector), q(1))]);
        let projected = project_abstract_x2_hook(&unit, metric, metric_inverse);
        let twice = project_abstract_x2_hook(&projected, metric, metric_inverse);
        maximum_projected_column_support = maximum_projected_column_support.max(projected.len());
        let keys = projected
            .keys()
            .chain(twice.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        idempotence_residual_entries += keys
            .iter()
            .filter(|key| projected.get(key) != twice.get(key))
            .count();
        trace_residual_entries += hook_trace(&projected, &pairs, metric)
            .iter()
            .filter(|value| **value != q(0))
            .count();
        exterior_residual_entries += hook_wedge(&projected, &pairs).len();
        operator_trace += projected
            .get(&(pair, vector))
            .cloned()
            .unwrap_or_else(|| q(0));
    }
    let vectors = vector_weights();
    let mut chevalley_commutators_checked = 0;
    let mut chevalley_commutator_residual_entries = 0;
    for direction in [Direction::Raising, Direction::Lowering] {
        for root in 0..5 {
            for input in 0..ambient_dimension {
                chevalley_commutators_checked += 1;
                let unit = HookTensor::from([(
                    (input / VECTOR_DIMENSION, input % VECTOR_DIMENSION),
                    q(1),
                )]);
                let projected_then_acted = act_hook_tensor(
                    &project_abstract_x2_hook(&unit, metric, metric_inverse),
                    root,
                    direction,
                    &vectors,
                    &pairs,
                );
                let acted_then_projected = project_abstract_x2_hook(
                    &act_hook_tensor(&unit, root, direction, &vectors, &pairs),
                    metric,
                    metric_inverse,
                );
                let keys = projected_then_acted
                    .keys()
                    .chain(acted_then_projected.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                chevalley_commutator_residual_entries += keys
                    .iter()
                    .filter(|key| projected_then_acted.get(key) != acted_then_projected.get(key))
                    .count();
            }
        }
    }
    let hook_dimension = 429;
    let projector_rank_from_idempotent_trace =
        if idempotence_residual_entries == 0 && *operator_trace.denom() == 1 {
            usize::try_from(*operator_trace.numer()).unwrap_or(0)
        } else {
            0
        };
    let passed = ambient_dimension == 605
        && projector_rank_from_idempotent_trace == hook_dimension
        && idempotence_residual_entries == 0
        && trace_residual_entries == 0
        && exterior_residual_entries == 0
        && chevalley_commutator_residual_entries == 0;
    AbstractX2HookProjectorCertificate {
        ambient_dimension,
        trace_dimension: 11,
        exterior_dimension: 165,
        hook_dimension,
        operator_trace: operator_trace.to_string(),
        projector_rank_from_idempotent_trace,
        unit_columns_checked: ambient_dimension,
        idempotence_residual_entries,
        trace_residual_entries,
        exterior_residual_entries,
        chevalley_commutators_checked,
        chevalley_commutator_residual_entries,
        maximum_projected_column_support,
        passed,
    }
}

fn hook_projector_columns(
    metric: &[Vec<Rational>],
    metric_inverse: &[Vec<Rational>],
) -> Vec<Vec<(usize, Rational)>> {
    let ambient_dimension = two_form_pairs().len() * VECTOR_DIMENSION;
    (0..ambient_dimension)
        .map(|input| {
            let unit =
                HookTensor::from([((input / VECTOR_DIMENSION, input % VECTOR_DIMENSION), q(1))]);
            project_abstract_x2_hook(&unit, metric, metric_inverse)
                .into_iter()
                .map(|((pair, vector), coefficient)| {
                    (pair * VECTOR_DIMENSION + vector, coefficient)
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct LeadingX2GaugeJobReport {
    pub gauge_form_degree: usize,
    pub leading_operator_ordinal: usize,
    pub leading_operator_label: String,
    pub parameter_components_expected: usize,
    pub parameter_components_visited: usize,
    pub parameter_components_directly_combined: usize,
    pub cyclic_parameter_is_highest_weight: bool,
    pub target_basis_states_directly_sampled: usize,
    pub raw_target_terms_emitted: u64,
    pub gamma_two_derivative_terms_accumulated: u64,
    pub exact_functional_buckets: usize,
    pub nonzero_exact_functional_values: usize,
    pub nonzero_parameter_components: usize,
    pub hook_leading_symbol_proved_nonzero: bool,
    pub canonical_functional_output_sha256: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LeadingX2ChannelRankReport {
    pub gauge_form_degree: usize,
    pub operator_columns: usize,
    pub nonzero_operator_columns: usize,
    pub exact_functional_projection_rank: usize,
    pub exact_full_column_rank: Option<usize>,
    pub exact_full_kernel_dimension: Option<usize>,
    pub functionally_independent_operator_ordinals: Vec<usize>,
    pub full_kernel_relations_by_operator_ordinal: Vec<Vec<String>>,
    pub all_parameter_components_covered: bool,
    pub parameter_irrep_generated_dimension: usize,
    pub parameter_irrep_expected_dimension: usize,
    pub every_column_certified_spin11_equivariant: bool,
    pub relations_propagated_by_equivariance: bool,
    pub dependent_relations_descendant_spot_checked: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LeadingX2GaugeReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_reference: &'static str,
    pub bidegree: &'static str,
    pub hook_projector: AbstractX2HookProjectorCertificate,
    pub jobs: Vec<LeadingX2GaugeJobReport>,
    pub channel_ranks: Vec<LeadingX2ChannelRankReport>,
    pub gauge_channels_checked: usize,
    pub leading_operators_checked_per_channel: usize,
    pub total_jobs: usize,
    pub total_parameter_components_expected: usize,
    pub total_parameter_components_visited: usize,
    pub total_parameter_components_directly_combined: usize,
    pub jobs_proved_nonzero: usize,
    pub jobs_not_separated_by_functionals: usize,
    pub all_parameter_components_covered: bool,
    pub leading_symbol_f0_a_g_established_by_job: bool,
    pub exact_cross_operator_column_ranks_established: bool,
    pub cyclic_vector_reduction_certified: bool,
    pub equivariance_mutation_test_present: bool,
    pub physical_operator_combination_selected: bool,
    pub full_f_a_g_p_established: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1_usize, |value, index| value * (n - index) / (index + 1))
}

fn rational_to_big(value: &Rational) -> BigRational {
    Ratio::new(BigInt::from(*value.numer()), BigInt::from(*value.denom()))
}

fn small_gaussian(real: i64, imaginary: i64) -> SmallGaussian {
    Complex::new(q(real), q(imaginary))
}

fn cartesian_combinations(n: usize, degree: usize) -> Vec<Vec<usize>> {
    fn visit(
        start: usize,
        n: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(current.clone());
            return;
        }
        for value in start..=n - remaining {
            current.push(value);
            visit(value + 1, n, remaining - 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    visit(0, n, degree, &mut Vec::new(), &mut output);
    output
}

fn abstract_vector_to_cartesian_coefficients(gamma: &AbstractGammaMap) -> Vec<Vec<SmallGaussian>> {
    let cartesian = crate::eleven_dimensional_clifford::gamma_matrices();
    assert_eq!(cartesian.len(), VECTOR_DIMENSION);
    let mut change = vec![vec![small_gaussian(0, 0); VECTOR_DIMENSION]; VECTOR_DIMENSION];
    let gram = (0..VECTOR_DIMENSION)
        .map(|left| {
            (0..VECTOR_DIMENSION)
                .map(|right| {
                    (0..SPINOR_DIMENSION)
                        .flat_map(|row| {
                            let cartesian = &cartesian;
                            (0..SPINOR_DIMENSION).map(move |column| {
                                cartesian[left][row][column].conj()
                                    * cartesian[right][row][column].clone()
                            })
                        })
                        .sum::<SmallGaussian>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for abstract_vector in 0..VECTOR_DIMENSION {
        let right_hand_side = (0..VECTOR_DIMENSION)
            .map(|basis| {
                (0..SPINOR_DIMENSION)
                    .flat_map(|row| {
                        let cartesian = &cartesian;
                        (0..SPINOR_DIMENSION).map(move |column| {
                            cartesian[basis][row][column].conj()
                                * Complex::new(
                                    gamma.matrices[abstract_vector][row][column].clone(),
                                    q(0),
                                )
                        })
                    })
                    .sum::<SmallGaussian>()
            })
            .collect::<Vec<_>>();
        let mut augmented = gram
            .iter()
            .zip(right_hand_side)
            .map(|(row, right)| {
                let mut row = row.clone();
                row.push(right);
                row
            })
            .collect::<Vec<_>>();
        for cartesian_vector in 0..VECTOR_DIMENSION {
            let pivot = (cartesian_vector..VECTOR_DIMENSION)
                .find(|row| augmented[*row][cartesian_vector] != small_gaussian(0, 0))
                .unwrap();
            augmented.swap(cartesian_vector, pivot);
            let normalization = augmented[cartesian_vector][cartesian_vector].clone();
            for value in &mut augmented[cartesian_vector][cartesian_vector..] {
                *value /= normalization.clone();
            }
            let pivot_row = augmented[cartesian_vector].clone();
            for row in 0..VECTOR_DIMENSION {
                if row == cartesian_vector
                    || augmented[row][cartesian_vector] == small_gaussian(0, 0)
                {
                    continue;
                }
                let factor = augmented[row][cartesian_vector].clone();
                for column in cartesian_vector..=VECTOR_DIMENSION {
                    augmented[row][column] -= factor.clone() * pivot_row[column].clone();
                }
            }
        }
        change[abstract_vector] = augmented
            .into_iter()
            .map(|row| row[VECTOR_DIMENSION].clone())
            .collect();
        for row in 0..SPINOR_DIMENSION {
            for column in 0..SPINOR_DIMENSION {
                let reconstructed = (0..VECTOR_DIMENSION)
                    .map(|cartesian_vector| {
                        change[abstract_vector][cartesian_vector].clone()
                            * cartesian[cartesian_vector][row][column].clone()
                    })
                    .sum::<SmallGaussian>();
                assert_eq!(
                    reconstructed,
                    Complex::new(gamma.matrices[abstract_vector][row][column].clone(), q(0)),
                    "abstract vector {abstract_vector}, matrix entry ({row},{column})"
                );
            }
        }
    }
    change
}

fn highest_parameter_cartesian_combination(
    gamma: &AbstractGammaMap,
    degree: usize,
) -> BTreeMap<usize, SmallGaussian> {
    assert!(degree <= 5);
    if degree == 0 {
        return BTreeMap::from([(0, small_gaussian(1, 0))]);
    }
    let change = abstract_vector_to_cartesian_coefficients(gamma);
    let mut exterior = BTreeMap::<u16, SmallGaussian>::from([(0, small_gaussian(1, 0))]);
    for abstract_vector in (0..degree).map(|axis| 2 * axis) {
        let mut next = BTreeMap::<u16, SmallGaussian>::new();
        for (&mask, coefficient) in &exterior {
            for cartesian_vector in 0..VECTOR_DIMENSION {
                let vector_coefficient = &change[abstract_vector][cartesian_vector];
                let bit = 1_u16 << cartesian_vector;
                if mask & bit != 0 || *vector_coefficient == small_gaussian(0, 0) {
                    continue;
                }
                let greater = (mask >> (cartesian_vector + 1)).count_ones();
                let sign = if greater % 2 == 0 { 1 } else { -1 };
                *next
                    .entry(mask | bit)
                    .or_insert_with(|| small_gaussian(0, 0)) +=
                    coefficient.clone() * vector_coefficient.clone() * small_gaussian(sign, 0);
            }
        }
        next.retain(|_, coefficient| *coefficient != small_gaussian(0, 0));
        exterior = next;
    }
    let combinations = cartesian_combinations(VECTOR_DIMENSION, degree);
    exterior
        .into_iter()
        .map(|(mask, coefficient)| {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|index| mask & (1_u16 << index) != 0)
                .collect::<Vec<_>>();
            (
                combinations
                    .iter()
                    .position(|candidate| *candidate == indices)
                    .unwrap(),
                coefficient,
            )
        })
        .collect()
}

fn lower_abstract_form(
    source: &BTreeMap<u16, Rational>,
    root: usize,
    vectors: &[Weight; VECTOR_DIMENSION],
) -> BTreeMap<u16, Rational> {
    let mut output = BTreeMap::<u16, Rational>::new();
    for (&mask, coefficient) in source {
        for vector in 0..VECTOR_DIMENSION {
            let bit = 1_u16 << vector;
            if mask & bit == 0 {
                continue;
            }
            let Some((next, action_coefficient)) =
                vector_action(vector, root, Direction::Lowering, vectors)
            else {
                continue;
            };
            let next_bit = 1_u16 << next;
            if mask & next_bit != 0 {
                continue;
            }
            let without = mask ^ bit;
            let between = if vector < next {
                (without >> (vector + 1)) & ((1_u16 << (next - vector - 1)) - 1)
            } else {
                (without >> (next + 1)) & ((1_u16 << (vector - next - 1)) - 1)
            };
            let sign = if between.count_ones() % 2 == 0 { 1 } else { -1 };
            *output.entry(without | next_bit).or_insert_with(|| q(0)) +=
                coefficient.clone() * q(action_coefficient * sign);
        }
    }
    output.retain(|_, coefficient| *coefficient != q(0));
    output
}

fn add_form_orbit_basis(
    mut state: BTreeMap<u16, Rational>,
    basis: &mut Vec<(u16, BTreeMap<u16, Rational>)>,
) -> bool {
    basis.sort_by_key(|(pivot, _)| *pivot);
    for (pivot, existing) in basis.iter() {
        let Some(factor) = state.get(pivot).cloned() else {
            continue;
        };
        for (mask, value) in existing {
            *state.entry(*mask).or_insert_with(|| q(0)) -= factor.clone() * value.clone();
            if state[mask] == q(0) {
                state.remove(mask);
            }
        }
    }
    if state.is_empty() {
        return false;
    }
    let pivot = *state.first_key_value().unwrap().0;
    let normalization = state[&pivot].clone();
    for value in state.values_mut() {
        *value /= normalization.clone();
    }
    basis.push((pivot, state));
    true
}

fn parameter_lowering_orbit_dimension(degree: usize) -> usize {
    let vectors = vector_weights();
    let highest_mask = (0..degree).fold(0_u16, |mask, axis| mask | (1_u16 << (2 * axis)));
    let highest = BTreeMap::from([(highest_mask, q(1))]);
    let mut basis = Vec::new();
    add_form_orbit_basis(highest.clone(), &mut basis);
    let mut queue = std::collections::VecDeque::from([highest]);
    while let Some(state) = queue.pop_front() {
        for root in 0..5 {
            let descendant = lower_abstract_form(&state, root, &vectors);
            if !descendant.is_empty() && add_form_orbit_basis(descendant.clone(), &mut basis) {
                queue.push_back(descendant);
            }
        }
    }
    basis.len()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BigGaussian {
    real: BigRational,
    imaginary: BigRational,
}

impl BigGaussian {
    fn zero() -> Self {
        Self {
            real: bq(0),
            imaginary: bq(0),
        }
    }

    fn one() -> Self {
        Self {
            real: bq(1),
            imaginary: bq(0),
        }
    }

    fn is_zero(&self) -> bool {
        self.real == bq(0) && self.imaginary == bq(0)
    }

    fn multiply(&self, other: &Self) -> Self {
        Self {
            real: self.real.clone() * other.real.clone()
                - self.imaginary.clone() * other.imaginary.clone(),
            imaginary: self.real.clone() * other.imaginary.clone()
                + self.imaginary.clone() * other.real.clone(),
        }
    }

    fn divide(&self, other: &Self) -> Self {
        assert!(!other.is_zero());
        let norm = other.real.clone() * other.real.clone()
            + other.imaginary.clone() * other.imaginary.clone();
        Self {
            real: (self.real.clone() * other.real.clone()
                + self.imaginary.clone() * other.imaginary.clone())
                / norm.clone(),
            imaginary: (self.imaginary.clone() * other.real.clone()
                - self.real.clone() * other.imaginary.clone())
                / norm,
        }
    }

    fn subtract_assign_product(&mut self, left: &Self, right: &Self) {
        let product = left.multiply(right);
        self.real -= product.real;
        self.imaginary -= product.imaginary;
    }
}

type LeadingOutputKey = (usize, u32, usize);
type ComplexColumn = BTreeMap<LeadingOutputKey, BigGaussian>;
const LEADING_FUNCTIONAL_BUCKETS: usize = 1024;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn leading_functional_bucket_and_sign(mask: u32, raw_output: usize) -> (usize, i64) {
    let hash = splitmix64(
        u64::from(mask) ^ (u64::try_from(raw_output).unwrap() << 32) ^ 0x243f_6a88_85a3_08d3,
    );
    (
        (hash as usize) % LEADING_FUNCTIONAL_BUCKETS,
        if hash >> 63 == 0 { 1 } else { -1 },
    )
}

fn canonical_hook_hash(accumulated: &ComplexColumn) -> String {
    let mut hasher = Sha256::new();
    for ((parameter, mask, raw), value) in accumulated {
        hasher.update(format!(
            "{parameter}:{mask:08x}:{raw}:{}:{}\n",
            value.real, value.imaginary
        ));
    }
    format!("{:x}", hasher.finalize())
}

fn verify_leading_x2_operator(
    leading_operator_ordinal: usize,
    raised_gamma_two: &[Vec<Vec<Rational>>],
    projector_columns: &[Vec<(usize, Rational)>],
    cyclic_parameters: &[BTreeMap<usize, SmallGaussian>; 6],
    selected_target_basis_ordinals: &[usize],
) -> io::Result<Vec<(LeadingX2GaugeJobReport, ComplexColumn)>> {
    let pairs = two_form_pairs();
    let mut accumulated_by_degree = (0..6).map(|_| ComplexColumn::new()).collect::<Vec<_>>();
    let mut gamma_two_derivative_terms_accumulated = [0_u64; 6];
    let (spec, parameter_basis_by_degree, _, _, raw_target_terms_emitted_by_degree) =
        crate::eleven_dimensional_level16_couplings::visit_target_resolved_zero_momentum_gauge_composition_terms_all_degrees(
            leading_operator_ordinal,
            cyclic_parameters,
            Some(selected_target_basis_ordinals),
            |gauge_form_degree, entry| {
                for derivative_spinor in 0..SPINOR_DIMENSION {
                    let Some(wedge_sign) = right_wedge_sign(entry.exterior_mask, derivative_spinor) else {
                        continue;
                    };
                    let output_mask = entry.exterior_mask | (1_u32 << derivative_spinor);
                    for pair in 0..pairs.len() {
                        let gamma_coefficient = &raised_gamma_two[pair][derivative_spinor]
                            [entry.target_spinor_weight_index];
                        if *gamma_coefficient == q(0) {
                            continue;
                        }
                        gamma_two_derivative_terms_accumulated[gauge_form_degree] += 1;
                        let raw_input = pair * VECTOR_DIMENSION + entry.target_vector_weight_index;
                        let gamma_factor = rational_to_big(gamma_coefficient)
                            * bq(wedge_sign)
                            / bq(16);
                        for (raw_output, hook_coefficient) in &projector_columns[raw_input] {
                            let (bucket, functional_sign) =
                                leading_functional_bucket_and_sign(output_mask, *raw_output);
                            let factor = gamma_factor.clone()
                                * rational_to_big(hook_coefficient)
                                * bq(functional_sign);
                            let key = (0, 0, bucket);
                            let value = accumulated_by_degree[gauge_form_degree]
                                .entry(key)
                                .or_insert_with(BigGaussian::zero);
                            let source = BigGaussian {
                                real: entry.real.clone() * factor.clone(),
                                imaginary: entry.imaginary.clone() * factor,
                            };
                            value.real += source.real;
                            value.imaginary += source.imaginary;
                            if value.is_zero() {
                                accumulated_by_degree[gauge_form_degree].remove(&key);
                            }
                        }
                    }
                }
                Ok(())
            },
        )?;
    Ok(accumulated_by_degree
        .into_iter()
        .enumerate()
        .map(|(gauge_form_degree, accumulated)| {
            let parameter_components_expected = binomial(VECTOR_DIMENSION, gauge_form_degree);
            let parameter_components_visited = parameter_basis_by_degree[gauge_form_degree].len();
            let hook_leading_symbol_proved_nonzero = !accumulated.is_empty();
            let report = LeadingX2GaugeJobReport {
                gauge_form_degree,
                leading_operator_ordinal,
                leading_operator_label: spec.label.clone(),
                parameter_components_expected,
                parameter_components_visited,
                parameter_components_directly_combined: cyclic_parameters[gauge_form_degree].len(),
                cyclic_parameter_is_highest_weight: true,
                target_basis_states_directly_sampled: selected_target_basis_ordinals.len(),
                raw_target_terms_emitted: raw_target_terms_emitted_by_degree[gauge_form_degree],
                gamma_two_derivative_terms_accumulated: gamma_two_derivative_terms_accumulated
                    [gauge_form_degree],
                exact_functional_buckets: LEADING_FUNCTIONAL_BUCKETS,
                nonzero_exact_functional_values: accumulated.len(),
                nonzero_parameter_components: usize::from(hook_leading_symbol_proved_nonzero),
                hook_leading_symbol_proved_nonzero,
                canonical_functional_output_sha256: canonical_hook_hash(&accumulated),
                passed: parameter_components_visited == parameter_components_expected,
            };
            (report, accumulated)
        })
        .collect())
}

#[derive(Clone)]
struct ComplexColumnBasisVector {
    pivot: LeadingOutputKey,
    values: ComplexColumn,
    combination: Vec<BigGaussian>,
    source_ordinal: usize,
}

fn add_complex_column_to_echelon(
    mut column: ComplexColumn,
    ordinal: usize,
    width: usize,
    basis: &mut Vec<ComplexColumnBasisVector>,
) -> Option<Vec<BigGaussian>> {
    let mut combination = vec![BigGaussian::zero(); width];
    combination[ordinal] = BigGaussian::one();
    basis.sort_by_key(|entry| entry.pivot);
    for existing in basis.iter() {
        let Some(factor) = column.get(&existing.pivot).cloned() else {
            continue;
        };
        for (key, value) in &existing.values {
            let entry = column.entry(*key).or_insert_with(BigGaussian::zero);
            entry.subtract_assign_product(&factor, value);
            if entry.is_zero() {
                column.remove(key);
            }
        }
        for (entry, value) in combination.iter_mut().zip(&existing.combination) {
            entry.subtract_assign_product(&factor, value);
        }
    }
    if column.is_empty() {
        return Some(combination);
    }
    let pivot = *column.first_key_value().unwrap().0;
    let normalization = column[&pivot].clone();
    for value in column.values_mut() {
        *value = value.divide(&normalization);
    }
    for value in &mut combination {
        *value = value.divide(&normalization);
    }
    basis.push(ComplexColumnBasisVector {
        pivot,
        values: column,
        combination,
        source_ordinal: ordinal,
    });
    None
}

fn channel_rank_report(
    gauge_form_degree: usize,
    columns: Vec<ComplexColumn>,
    parameter_inventory_matches: bool,
    parameter_irrep_generated_dimension: usize,
    every_column_certified_spin11_equivariant: bool,
) -> LeadingX2ChannelRankReport {
    assert_eq!(columns.len(), 12);
    let nonzero_operator_columns = columns.iter().filter(|column| !column.is_empty()).count();
    let mut basis = Vec::<ComplexColumnBasisVector>::new();
    let mut projected_kernel_dimension = 0;
    for (ordinal, column) in columns.into_iter().enumerate() {
        if add_complex_column_to_echelon(column, ordinal, 12, &mut basis).is_some() {
            projected_kernel_dimension += 1;
        }
    }
    assert_eq!(basis.len() + projected_kernel_dimension, 12);
    let exact_functional_projection_rank = basis.len();
    let exact_full_column_rank = (exact_functional_projection_rank == 12).then_some(12);
    let exact_full_kernel_dimension = exact_full_column_rank.map(|rank| 12 - rank);
    let functionally_independent_operator_ordinals =
        basis.iter().map(|entry| entry.source_ordinal).collect();
    let full_kernel_relations_by_operator_ordinal = Vec::new();
    let parameter_irrep_expected_dimension = binomial(VECTOR_DIMENSION, gauge_form_degree);
    let relations_propagated_by_equivariance = exact_full_column_rank == Some(12)
        && every_column_certified_spin11_equivariant
        && parameter_irrep_generated_dimension == parameter_irrep_expected_dimension;
    let all_parameter_components_covered = parameter_inventory_matches
        && parameter_irrep_generated_dimension == parameter_irrep_expected_dimension
        && every_column_certified_spin11_equivariant;
    let dependent_relations_descendant_spot_checked = 0;
    let passed = all_parameter_components_covered;
    LeadingX2ChannelRankReport {
        gauge_form_degree,
        operator_columns: 12,
        nonzero_operator_columns,
        exact_functional_projection_rank,
        exact_full_column_rank,
        exact_full_kernel_dimension,
        functionally_independent_operator_ordinals,
        full_kernel_relations_by_operator_ordinal,
        all_parameter_components_covered,
        parameter_irrep_generated_dimension,
        parameter_irrep_expected_dimension,
        every_column_certified_spin11_equivariant,
        relations_propagated_by_equivariance,
        dependent_relations_descendant_spot_checked,
        passed,
    }
}

pub fn verify_leading_zero_momentum_x2_gauge() -> io::Result<LeadingX2GaugeReport> {
    let (gamma, _, _, _) = solve_gamma_trace();
    let metric = gamma_metric(&gamma);
    let metric_inverse = inverse_square_matrix(&metric);
    let hook_projector = certify_abstract_x2_hook(&metric, &metric_inverse);
    assert!(hook_projector.passed);
    let projector_columns = hook_projector_columns(&metric, &metric_inverse);
    let (charge, _, _, _) = invariant_spinor_bilinear();
    let gamma_two = gamma_two_matrices(&gamma);
    let (raised_gamma_two, inverse_residuals, symmetry_residuals) =
        gamma_two_raised(&gamma_two, &charge);
    assert_eq!(inverse_residuals, 0);
    assert_eq!(symmetry_residuals, 0);
    let (_, gamma_two_equivariance_residuals) = gamma_two_intertwining_residuals(&gamma_two);
    let source_column_certificates: serde_json::Value =
        serde_json::from_str(LEVEL16_ALL_COUPLINGS_CERTIFICATE).unwrap();
    let source_column_certificate_hash = format!(
        "{:x}",
        Sha256::digest(LEVEL16_ALL_COUPLINGS_CERTIFICATE.as_bytes())
    );
    let source_columns_certified = source_column_certificate_hash
        == LEVEL16_ALL_COUPLINGS_CERTIFICATE_SHA256
        && source_column_certificates["passed"].as_bool() == Some(true)
        && source_column_certificates["every_residual_is_exactly_zero"].as_bool() == Some(true)
        && source_column_certificates["embedded_source_copies_certified"].as_u64() == Some(12);
    let target_stream_certificate = crate::eleven_dimensional_target_stream::verify();
    let every_column_certified_spin11_equivariant = gamma_two_equivariance_residuals == 0
        && hook_projector.chevalley_commutator_residual_entries == 0
        && source_columns_certified
        && target_stream_certificate.passed;
    let target_basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let selected_target_basis_ordinals = target_basis
        .iter()
        .filter(|state| state.pbw_word_simple_roots.is_empty())
        .map(|state| state.ordinal)
        .collect::<Vec<_>>();
    assert_eq!(selected_target_basis_ordinals.len(), 1);
    let cyclic_parameters: [BTreeMap<usize, SmallGaussian>; 6] =
        std::array::from_fn(|degree| highest_parameter_cartesian_combination(&gamma, degree));
    let parameter_orbit_dimensions = (0..=5)
        .map(parameter_lowering_orbit_dimension)
        .collect::<Vec<_>>();
    let results = (0..12)
        .into_par_iter()
        .map(|leading_operator_ordinal| {
            let result = verify_leading_x2_operator(
                leading_operator_ordinal,
                &raised_gamma_two,
                &projector_columns,
                &cyclic_parameters,
                &selected_target_basis_ordinals,
            );
            #[cfg(test)]
            if let Ok(degrees) = &result {
                eprintln!(
                    "leading X2 gauge progress: all degrees, column {}/12, nonzero={:?}",
                    leading_operator_ordinal + 1,
                    degrees
                        .iter()
                        .map(|(report, _)| report.hook_leading_symbol_proved_nonzero)
                        .collect::<Vec<_>>()
                );
            }
            result
        })
        .collect::<Vec<_>>();
    let mut jobs = Vec::with_capacity(72);
    let mut columns_by_degree = (0..6).map(|_| Vec::with_capacity(12)).collect::<Vec<_>>();
    for result in results {
        for (job, column) in result? {
            columns_by_degree[job.gauge_form_degree].push(column);
            jobs.push(job);
        }
    }
    jobs.sort_by_key(|job| (job.gauge_form_degree, job.leading_operator_ordinal));
    let mut channel_ranks = Vec::with_capacity(6);
    for gauge_form_degree in 0..=5 {
        let coverage = jobs
            .iter()
            .filter(|job| job.gauge_form_degree == gauge_form_degree)
            .all(|job| job.passed);
        channel_ranks.push(channel_rank_report(
            gauge_form_degree,
            std::mem::take(&mut columns_by_degree[gauge_form_degree]),
            coverage,
            parameter_orbit_dimensions[gauge_form_degree],
            every_column_certified_spin11_equivariant,
        ));
    }
    let total_parameter_components_expected = jobs
        .iter()
        .map(|job| job.parameter_components_expected)
        .sum();
    let total_parameter_components_visited = jobs
        .iter()
        .map(|job| job.parameter_components_visited)
        .sum();
    let total_parameter_components_directly_combined = jobs
        .iter()
        .map(|job| job.parameter_components_directly_combined)
        .sum();
    let jobs_proved_nonzero = jobs
        .iter()
        .filter(|job| job.hook_leading_symbol_proved_nonzero)
        .count();
    let jobs_not_separated_by_functionals = jobs.len() - jobs_proved_nonzero;
    let all_parameter_components_covered = channel_ranks
        .iter()
        .all(|channel| channel.all_parameter_components_covered);
    let leading_symbol_f0_a_g_established_by_job = false;
    let exact_cross_operator_column_ranks_established = channel_ranks.iter().all(|channel| {
        channel.exact_full_column_rank.is_some() && channel.exact_full_kernel_dimension.is_some()
    });
    let cyclic_vector_reduction_certified = channel_ranks.iter().all(|channel| {
        channel.every_column_certified_spin11_equivariant
            && channel.parameter_irrep_generated_dimension
                == channel.parameter_irrep_expected_dimension
    });
    let equivariance_mutation_test_present = x2_equivariance_mutation_detected();
    let passed = hook_projector.passed
        && jobs.len() == 72
        && jobs.iter().all(|job| job.passed)
        && channel_ranks.iter().all(|channel| channel.passed)
        && all_parameter_components_covered
        && cyclic_vector_reduction_certified
        && equivariance_mutation_test_present;
    Ok(LeadingX2GaugeReport {
        schema_version: "adynkra-11d-leading-x2-gauge-v1",
        role: "exact projected-rank witness for the leading-symbol X_[2] hook on the target-resolved zero-momentum gauge-composition stream",
        source_reference: "hep-th/0101037 Eqs. (39)-(40): (1/16) Gamma_[2] D H followed by trace and total-antisymmetry removal",
        bidegree: "D^18 Lambda at zero momentum",
        hook_projector,
        jobs,
        channel_ranks,
        gauge_channels_checked: 6,
        leading_operators_checked_per_channel: 12,
        total_jobs: 72,
        total_parameter_components_expected,
        total_parameter_components_visited,
        total_parameter_components_directly_combined,
        jobs_proved_nonzero,
        jobs_not_separated_by_functionals,
        all_parameter_components_covered,
        leading_symbol_f0_a_g_established_by_job,
        exact_cross_operator_column_ranks_established,
        cyclic_vector_reduction_certified,
        equivariance_mutation_test_present,
        physical_operator_combination_selected: false,
        full_f_a_g_p_established: false,
        passed,
        boundary: "This is the exact zero-momentum D^18 Lambda leading-symbol test for the X_[2] hook term only. It evaluates the exact highest-weight gauge parameter and highest target state, then applies 1,024 deterministic exact output functionals. The rigorous projected-rank lower bounds for gauge degrees zero through five are [1,4,3,5,2,2]. Because every projected rank is below 12, these lower bounds do not determine any full 12-column rank or kernel, and all exact_full_* fields remain null. No cross-degree cancellation is allowed and no physical combination of the 12 bridge columns is selected. The test does not include the momentum anticommutator branch, lower-symbol corrections, compensator elimination, W, X_[5], J, or a complete physical curvature F, so it does not establish leading-symbol or full F A G_p = 0.",
    })
}

pub fn write_leading_x2_artifacts(data_path: &Path, results_path: &Path) -> io::Result<()> {
    let report = verify_leading_zero_momentum_x2_gauge()?;
    for path in [data_path, results_path] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        serde_json::to_writer_pretty(&mut file, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct AbstractCliffordJoinReport {
    pub schema_version: &'static str,
    pub source_references: Vec<&'static str>,
    pub basis_convention: &'static str,
    pub gamma_weight_allowed_variables: usize,
    pub gamma_intertwining_equations: usize,
    pub raising_equations_by_simple_root: [usize; 5],
    pub lowering_equations_by_simple_root: [usize; 5],
    pub gamma_intertwining_rank: usize,
    pub gamma_intertwiner_dimension: usize,
    pub gamma_intertwining_residual_equations: usize,
    pub gamma_nonzero_coefficients: usize,
    pub gamma_nonintegral_coefficients: usize,
    pub gamma_trace_rank: usize,
    pub gamma_traceless_kernel_dimension: usize,
    pub target_basis_states_checked: usize,
    pub target_basis_terms_checked: usize,
    pub target_gamma_trace_residual_entries: usize,
    pub clifford_metric_rank: usize,
    pub clifford_residual_entries: usize,
    pub clifford_metric_nonzero_upper_triangle: Vec<MetricEntry>,
    pub invariant_spinor_bilinear_variables: usize,
    pub invariant_spinor_bilinear_equations: usize,
    pub invariant_spinor_bilinear_rank: usize,
    pub invariant_spinor_bilinear_dimension: usize,
    pub invariant_spinor_bilinear_antisymmetry_residuals: usize,
    pub gamma_two_matrices: usize,
    pub gamma_two_independent_matrices: usize,
    pub gamma_two_intertwining_entries_checked: usize,
    pub gamma_two_intertwining_residual_entries: usize,
    pub gamma_two_raised_symmetry_residual_entries: usize,
    pub typed_gamma_two_d_h_contraction_available: bool,
    pub typed_target_stream_derivative_visitor_available: bool,
    pub abstract_x2_hook_projector_complete: bool,
    pub abstract_x2_hook_projector_rank: usize,
    pub abstract_x2_hook_chevalley_commutator_residual_entries: usize,
    pub cartesian_hook_projector_join_complete: bool,
    pub complete_typed_target_stream_application: bool,
    pub physical_f_a_g_p_established: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn verify() -> AbstractCliffordJoinReport {
    let (gamma, gamma_rank, gamma_rows, row_counts) = solve_gamma_trace();
    let gamma_residuals = residual_rows(&gamma_rows, &gamma.coefficients);
    let gamma_nonzero_coefficients = gamma
        .coefficients
        .iter()
        .filter(|coefficient| **coefficient != q(0))
        .count();
    let gamma_nonintegral_coefficients = gamma
        .coefficients
        .iter()
        .filter(|coefficient| *coefficient.denom() != 1)
        .count();
    let gamma_matrix = (0..SPINOR_DIMENSION)
        .map(|output| {
            (0..AMBIENT_DIMENSION)
                .map(|input| {
                    gamma.matrices[input / SPINOR_DIMENSION][output][input % SPINOR_DIMENSION]
                        .clone()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let gamma_trace_rank = sparse_matrix_rank(&gamma_matrix);
    let (target_basis_terms_checked, target_gamma_trace_residual_entries) =
        target_kernel_residuals(&gamma);
    let (clifford_metric_rank, clifford_residual_entries, metric_entries) =
        clifford_certificate(&gamma);
    let (charge, charge_rank, charge_equations, charge_antisymmetry_residuals) =
        invariant_spinor_bilinear();
    let gamma_two = gamma_two_matrices(&gamma);
    let gamma_two_flat = gamma_two
        .iter()
        .map(|matrix| matrix.iter().flatten().cloned().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let gamma_two_independent_matrices = sparse_matrix_rank(&gamma_two_flat);
    let (gamma_two_entries_checked, gamma_two_residuals) =
        gamma_two_intertwining_residuals(&gamma_two);
    let (_, charge_inverse_residuals, gamma_two_symmetry_residuals) =
        gamma_two_raised(&gamma_two, &charge);
    let metric = gamma_metric(&gamma);
    let metric_inverse = inverse_square_matrix(&metric);
    let hook_projector = certify_abstract_x2_hook(&metric, &metric_inverse);
    let passed = gamma.variables.len() == 192
        && gamma_rows.len() == 736
        && gamma_rank == 191
        && gamma_residuals == 0
        && gamma_nonzero_coefficients == 192
        && gamma_nonintegral_coefficients == 0
        && gamma_trace_rank == 32
        && target_gamma_trace_residual_entries == 0
        && clifford_metric_rank == 11
        && clifford_residual_entries == 0
        && charge_rank == 31
        && charge_antisymmetry_residuals == 0
        && charge_inverse_residuals == 0
        && gamma_two.len() == 55
        && gamma_two_independent_matrices == 55
        && gamma_two_residuals == 0
        && gamma_two_symmetry_residuals == 0
        && hook_projector.passed;
    AbstractCliffordJoinReport {
        schema_version: "adynkra-11d-abstract-b5-clifford-join-v1",
        source_references: vec![
            "arXiv:2007.05097 Eqs. (2.2)-(2.3): gamma-traceless (10001) semi-prepotential and gamma-trace representative redundancy",
            "hep-th/0101037 Eq. (39): the (1/16) Gamma_[2] D H leading term in X_[2]",
        ],
        basis_convention: "B5 Chevalley weight bases: V=(+e1,-e1,...,+e5,-e5,0), S lexicographic sign weights, with simple-root string coefficients fixed intrinsically",
        gamma_weight_allowed_variables: gamma.variables.len(),
        gamma_intertwining_equations: gamma_rows.len(),
        raising_equations_by_simple_root: row_counts[0],
        lowering_equations_by_simple_root: row_counts[1],
        gamma_intertwining_rank: gamma_rank,
        gamma_intertwiner_dimension: gamma.variables.len() - gamma_rank,
        gamma_intertwining_residual_equations: gamma_residuals,
        gamma_nonzero_coefficients,
        gamma_nonintegral_coefficients,
        gamma_trace_rank,
        gamma_traceless_kernel_dimension: AMBIENT_DIMENSION - gamma_trace_rank,
        target_basis_states_checked: 320,
        target_basis_terms_checked,
        target_gamma_trace_residual_entries,
        clifford_metric_rank,
        clifford_residual_entries,
        clifford_metric_nonzero_upper_triangle: metric_entries,
        invariant_spinor_bilinear_variables: 32,
        invariant_spinor_bilinear_equations: charge_equations,
        invariant_spinor_bilinear_rank: charge_rank,
        invariant_spinor_bilinear_dimension: 32 - charge_rank,
        invariant_spinor_bilinear_antisymmetry_residuals: charge_antisymmetry_residuals,
        gamma_two_matrices: gamma_two.len(),
        gamma_two_independent_matrices,
        gamma_two_intertwining_entries_checked: gamma_two_entries_checked,
        gamma_two_intertwining_residual_entries: gamma_two_residuals,
        gamma_two_raised_symmetry_residual_entries: gamma_two_symmetry_residuals,
        typed_gamma_two_d_h_contraction_available: true,
        typed_target_stream_derivative_visitor_available: true,
        abstract_x2_hook_projector_complete: hook_projector.passed,
        abstract_x2_hook_projector_rank: hook_projector.projector_rank_from_idempotent_trace,
        abstract_x2_hook_chevalley_commutator_residual_entries: hook_projector
            .chevalley_commutator_residual_entries,
        cartesian_hook_projector_join_complete: false,
        complete_typed_target_stream_application: false,
        physical_f_a_g_p_established: false,
        passed,
        boundary: "The unique abstract B5 gamma trace and invariant spinor bilinear are solved without a Cartesian basis conversion. All 320 deterministic (10001) target states are in the exact kernel. A typed visitor consumes target-resolved stream entries, appends one exterior derivative, applies the exact Gamma_[2] contraction, and joins the result to the exact Spin(11)-equivariant rank-429 abstract hook projector. The separate leading-X2 artifact applies this abstract pipeline to a bounded zero-momentum stream. The abstract weight output is not converted into the repository's Cartesian curvature and compensator pipeline, momentum and lower-symbol terms are absent, and no complete F A G_p identity is claimed.",
    }
}

pub fn write_artifacts(data_path: &Path, results_path: &Path) -> io::Result<()> {
    let report = verify();
    for path in [data_path, results_path] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        serde_json::to_writer_pretty(&mut file, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_gamma_trace_has_the_exact_320_target_kernel() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.gamma_intertwiner_dimension, 1);
        assert_eq!(report.gamma_trace_rank, 32);
        assert_eq!(report.gamma_traceless_kernel_dimension, 320);
        assert_eq!(report.target_basis_states_checked, 320);
        assert_eq!(report.target_gamma_trace_residual_entries, 0);
        assert!(!report.complete_typed_target_stream_application);
        assert!(!report.physical_f_a_g_p_established);
        let expected = serde_json::to_value(&report).unwrap();
        for path in [
            "data/eleven_dimensional_abstract_clifford_join.json",
            "results/adynkra_11d_abstract_clifford_join_validation.json",
        ] {
            let actual: serde_json::Value =
                serde_json::from_reader(File::open(path).unwrap()).unwrap();
            assert_eq!(
                actual, expected,
                "stale abstract Clifford artifact at {path}"
            );
        }
    }

    #[test]
    fn rank_two_clifford_contraction_is_exact_and_symmetric_when_raised() {
        let report = verify();
        assert_eq!(report.clifford_metric_rank, 11);
        assert_eq!(report.clifford_residual_entries, 0);
        assert_eq!(report.gamma_two_matrices, 55);
        assert_eq!(report.gamma_two_independent_matrices, 55);
        assert_eq!(report.gamma_two_intertwining_residual_entries, 0);
        assert_eq!(report.gamma_two_raised_symmetry_residual_entries, 0);

        let (gamma, _, _, _) = solve_gamma_trace();
        let (charge, _, _, _) = invariant_spinor_bilinear();
        let (raised, _, _) = gamma_two_raised(&gamma_two_matrices(&gamma), &charge);
        let (derivative_spinor_weight_index, target_spinor_weight_index) = (0..SPINOR_DIMENSION)
            .flat_map(|derivative| (0..SPINOR_DIMENSION).map(move |target| (derivative, target)))
            .find(|(derivative, target)| {
                raised
                    .iter()
                    .any(|matrix| matrix[*derivative][*target] != q(0))
            })
            .unwrap();
        let terms = contract_gamma_two_d_h(&[AbstractDhTerm {
            derivative_spinor_weight_index,
            target_vector_weight_index: 0,
            target_spinor_weight_index,
            coefficient: q(1),
        }]);
        assert!(!terms.is_empty());
        assert!(terms.iter().all(|term| term.coefficient != q(0)));
    }

    #[test]
    fn coefficient_mutation_breaks_both_certificates() {
        let (mut gamma, _, rows, _) = solve_gamma_trace();
        gamma.coefficients[0] = -gamma.coefficients[0].clone();
        let variable = gamma.variables[0];
        gamma.matrices[variable.vector][variable.output_spinor][variable.input_spinor] =
            gamma.coefficients[0].clone();
        assert!(residual_rows(&rows, &gamma.coefficients) > 0);
        assert!(target_kernel_residuals(&gamma).1 > 0);
        assert!(x2_equivariance_mutation_detected());
    }

    #[test]
    fn abstract_x2_hook_projector_has_exact_rank_429() {
        let (gamma, _, _, _) = solve_gamma_trace();
        let metric = gamma_metric(&gamma);
        let metric_inverse = inverse_square_matrix(&metric);
        let certificate = certify_abstract_x2_hook(&metric, &metric_inverse);
        assert!(certificate.passed);
        assert_eq!(certificate.ambient_dimension, 605);
        assert_eq!(certificate.projector_rank_from_idempotent_trace, 429);
        assert_eq!(certificate.idempotence_residual_entries, 0);
        assert_eq!(certificate.trace_residual_entries, 0);
        assert_eq!(certificate.exterior_residual_entries, 0);

        let unit = HookTensor::from([((0, 2), q(1))]);
        let mutated = project_abstract_x2_hook_with_normalizations(
            &unit,
            &metric,
            &metric_inverse,
            Ratio::new(1, 2),
            Ratio::new(1, 10),
        );
        let mutated_twice = project_abstract_x2_hook_with_normalizations(
            &mutated,
            &metric,
            &metric_inverse,
            Ratio::new(1, 2),
            Ratio::new(1, 10),
        );
        assert_ne!(mutated, mutated_twice);
    }

    #[test]
    fn highest_parameter_vectors_generate_all_six_irreducibles() {
        let (gamma, _, _, _) = solve_gamma_trace();
        for degree in 0..=5 {
            let combination = highest_parameter_cartesian_combination(&gamma, degree);
            assert!(!combination.is_empty());
            assert_eq!(
                parameter_lowering_orbit_dimension(degree),
                binomial(VECTOR_DIMENSION, degree)
            );
        }
    }

    #[test]
    fn typed_stream_visitor_appends_one_exact_exterior_derivative() {
        let (gamma, _, _, _) = solve_gamma_trace();
        let (charge, _, _, _) = invariant_spinor_bilinear();
        let (raised, _, _) = gamma_two_raised(&gamma_two_matrices(&gamma), &charge);
        let (derivative, target_spinor) = (0..SPINOR_DIMENSION)
            .flat_map(|derivative| (0..SPINOR_DIMENSION).map(move |target| (derivative, target)))
            .find(|(derivative, target)| {
                raised
                    .iter()
                    .any(|matrix| matrix[*derivative][*target] != q(0))
            })
            .unwrap();
        let exterior_mask = (0..SPINOR_DIMENSION)
            .filter(|index| *index != derivative)
            .take(17)
            .fold(0_u32, |mask, index| mask | (1_u32 << index));
        let entry =
            crate::eleven_dimensional_level16_couplings::TargetResolvedGaugeCompositionEntry {
                target_basis_ordinal: 0,
                target_vector_weight_index: 0,
                target_spinor_weight_index: target_spinor,
                parameter_component_index: 0,
                momentum_vector_weight_index: None,
                exterior_mask,
                real: bq(1),
                imaginary: bq(0),
            };
        let mut outputs = Vec::new();
        let emitted = visit_gamma_two_d_h_stream_entry(&entry, |output| {
            assert_eq!(output.exterior_mask.count_ones(), 18);
            outputs.push(output);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted as usize, outputs.len());
        assert!(emitted > 0);
    }

    #[test]
    #[ignore = "materializes one exact source column twice to compare weighted visitors"]
    fn combined_gauge_scan_matches_componentwise_scan() {
        type Key = (usize, usize, usize, u32);
        type Value = (BigRational, BigRational);
        fn add(map: &mut BTreeMap<Key, Value>, key: Key, real: BigRational, imag: BigRational) {
            let value = map.entry(key).or_insert_with(|| (bq(0), bq(0)));
            value.0 += real;
            value.1 += imag;
            if value.0 == bq(0) && value.1 == bq(0) {
                map.remove(&key);
            }
        }

        let (gamma, _, _, _) = solve_gamma_trace();
        let selected_degree = 0;
        let parameter = highest_parameter_cartesian_combination(&gamma, selected_degree);
        assert_eq!(parameter.len(), 1);
        let selected = parameter.keys().copied().collect::<Vec<_>>();
        let target_basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
        let highest_target = target_basis
            .iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .unwrap()
            .ordinal;
        let mut componentwise = BTreeMap::new();
        crate::eleven_dimensional_level16_couplings::visit_target_resolved_zero_momentum_gauge_composition_terms(
            selected_degree,
            0,
            Some(&selected),
            Some(&[highest_target]),
            |entry| {
                let weight = &parameter[&entry.parameter_component_index];
                let real = entry.real.clone() * rational_to_big(&weight.re)
                    - entry.imaginary.clone() * rational_to_big(&weight.im);
                let imaginary = entry.real.clone() * rational_to_big(&weight.im)
                    + entry.imaginary.clone() * rational_to_big(&weight.re);
                add(
                    &mut componentwise,
                    (
                        entry.target_basis_ordinal,
                        entry.target_vector_weight_index,
                        entry.target_spinor_weight_index,
                        entry.exterior_mask,
                    ),
                    real,
                    imaginary,
                );
                Ok(())
            },
        )
        .unwrap();

        let weights: [BTreeMap<usize, SmallGaussian>; 6] = std::array::from_fn(|degree| {
            if degree == selected_degree {
                parameter.clone()
            } else {
                BTreeMap::new()
            }
        });
        let mut combined = BTreeMap::new();
        crate::eleven_dimensional_level16_couplings::visit_target_resolved_zero_momentum_gauge_composition_terms_all_degrees(
            0,
            &weights,
            Some(&[highest_target]),
            |degree, entry| {
                assert_eq!(degree, selected_degree);
                add(
                    &mut combined,
                    (
                        entry.target_basis_ordinal,
                        entry.target_vector_weight_index,
                        entry.target_spinor_weight_index,
                        entry.exterior_mask,
                    ),
                    entry.real,
                    entry.imaginary,
                );
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(combined, componentwise);
        assert!(!combined.is_empty());
    }

    #[test]
    fn leading_x2_artifacts_record_the_complete_bounded_result() {
        let data: serde_json::Value = serde_json::from_reader(
            File::open("data/eleven_dimensional_leading_x2_gauge.json").unwrap(),
        )
        .unwrap();
        let validation: serde_json::Value = serde_json::from_reader(
            File::open("results/adynkra_11d_leading_x2_gauge_validation.json").unwrap(),
        )
        .unwrap();
        assert_eq!(data, validation);
        assert_eq!(validation["passed"].as_bool(), Some(true));
        assert_eq!(
            validation["exact_cross_operator_column_ranks_established"].as_bool(),
            Some(false)
        );
        assert_eq!(
            validation["physical_operator_combination_selected"].as_bool(),
            Some(false)
        );
        assert_eq!(
            validation["full_f_a_g_p_established"].as_bool(),
            Some(false)
        );
        let projected_ranks = validation["channel_ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|channel| {
                assert_eq!(channel["passed"].as_bool(), Some(true));
                assert!(channel["exact_full_column_rank"].is_null());
                assert!(channel["exact_full_kernel_dimension"].is_null());
                channel["exact_functional_projection_rank"]
                    .as_u64()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(projected_ranks, vec![1, 4, 3, 5, 2, 2]);
    }

    #[test]
    #[ignore = "writes checked-in exact certification artifacts"]
    fn write_exact_artifacts() {
        write_artifacts(
            Path::new("data/eleven_dimensional_abstract_clifford_join.json"),
            Path::new("results/adynkra_11d_abstract_clifford_join_validation.json"),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "executes and writes all 72 exact leading X2 gauge jobs"]
    fn write_leading_x2_gauge_artifacts() {
        write_leading_x2_artifacts(
            Path::new("data/eleven_dimensional_leading_x2_gauge.json"),
            Path::new("results/adynkra_11d_leading_x2_gauge_validation.json"),
        )
        .unwrap();
    }
}
