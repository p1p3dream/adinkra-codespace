//! Old-minimal 4D N=1 field equation in Adynkrafield coordinates.
//!
//! The supergravity prepotential is changed from its 64 monomial components
//! to the 16 Lorentz blocks in the published supergravity genome. The chiral
//! compensator and its conjugate are changed to their four-component genomes.
//! The validated `(G, R, Rbar)` operator is then transported through those
//! source-defined coordinate maps and reconstructed in the superspace basis.

use crate::adynkra_derivative_intertwiners::{lower_channel, upper_channel};
use crate::adynkra_genome::{self, GenomeTerm, LorentzIrrep};
use crate::minimal_supergravity_curvatures::{
    conjugate_scalar_curvature, d_bar_squared, d_squared, gauge_image_l, gauge_image_l_bar,
    scalar_curvature, vector_curvature, zero_h,
};
use crate::supercovariant_derivative::{GaussianRational, Polynomial};
use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

type Matrix = Vec<Vec<GaussianRational>>;
const GRASSMANN_DIMENSION: usize = 16;
const AMBIENT_DIMENSION: usize = 96;
const ADYNKRAFIELD_DIMENSION: usize = 72;

#[derive(Debug, Clone, Serialize)]
pub struct GenomeBlock {
    pub left_degree: u8,
    pub right_degree: u8,
    pub irrep: LorentzIrrep,
    pub dimension: usize,
    pub coordinate_offset: usize,
    pub factorial_denominator: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdynkrafieldMomentumCheck {
    pub momentum: [i64; 4],
    pub momentum_class: &'static str,
    pub domain_embedding_rank: usize,
    pub coordinate_roundtrip_residuals: usize,
    pub superspace_operator_rank_on_domain: usize,
    pub adynkrafield_operator_rank: usize,
    pub adynkrafield_operator_nonzero_entries: usize,
    pub operator_reconstruction_entries_checked: usize,
    pub operator_reconstruction_residuals: usize,
    pub gauge_columns_checked: usize,
    pub gauge_coordinate_reconstruction_residuals: usize,
    pub gauge_noether_residuals: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdynkrafieldOperatorReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_equations: &'static str,
    pub operator_source: &'static str,
    pub operator_equation: &'static str,
    pub coordinate_definition: &'static str,
    pub superspace_ambient_dimension: usize,
    pub adynkrafield_domain_dimension: usize,
    pub supergravity_genome_dimension: usize,
    pub chiral_compensator_genome_dimension: usize,
    pub conjugate_compensator_genome_dimension: usize,
    pub supergravity_genome_blocks: Vec<GenomeBlock>,
    pub supergravity_change_of_basis_rank: usize,
    pub factorial_coefficients_enforced: bool,
    pub momentum_checks: Vec<AdynkrafieldMomentumCheck>,
    pub total_operator_reconstruction_entries_checked: usize,
    pub total_operator_reconstruction_residuals: usize,
    pub total_gauge_coordinate_reconstruction_residuals: usize,
    pub total_gauge_noether_residuals: usize,
    pub expected_operator_ranks: Vec<usize>,
    pub observed_operator_ranks: Vec<usize>,
    pub superspace_and_adynkrafield_operators_equivalent_on_tested_fibers: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> GaussianRational {
    Complex::new(
        Ratio::new(real, denominator),
        Ratio::new(imaginary, denominator),
    )
}

fn zero() -> GaussianRational {
    gaussian(0, 0, 1)
}

fn zeros(rows: usize, columns: usize) -> Matrix {
    vec![vec![zero(); columns]; rows]
}

fn identity(dimension: usize) -> Matrix {
    let mut result = zeros(dimension, dimension);
    for index in 0..dimension {
        result[index][index] = gaussian(1, 0, 1);
    }
    result
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    assert!(!left.is_empty() && !right.is_empty());
    assert_eq!(left[0].len(), right.len());
    let mut result = zeros(left.len(), right[0].len());
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot] == zero() {
                continue;
            }
            for column in 0..right[0].len() {
                result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
            }
        }
    }
    result
}

fn matrix_residuals(left: &Matrix, right: &Matrix) -> usize {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            assert_eq!(left_row.len(), right_row.len());
            left_row
                .iter()
                .zip(right_row)
                .filter(|(left_value, right_value)| left_value != right_value)
                .count()
        })
        .sum()
}

fn rank(matrix: &Matrix) -> usize {
    let mut work = matrix.clone();
    let rows = work.len();
    let columns = work[0].len();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| work[row][column] != zero()) else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..rows {
            if row == pivot_row || work[row][column] == zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next in column..columns {
                let subtraction = factor.clone() * work[pivot_row][next].clone();
                work[row][next] -= subtraction;
            }
        }
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    pivot_row
}

fn inverse(matrix: &Matrix) -> Matrix {
    assert_eq!(matrix.len(), matrix[0].len());
    let dimension = matrix.len();
    let mut work: Vec<Vec<_>> = matrix
        .iter()
        .enumerate()
        .map(|(row, values)| {
            let mut augmented = values.clone();
            augmented.extend((0..dimension).map(|column| {
                if row == column {
                    gaussian(1, 0, 1)
                } else {
                    zero()
                }
            }));
            augmented
        })
        .collect();
    for column in 0..dimension {
        let pivot = (column..dimension)
            .find(|&row| work[row][column] != zero())
            .expect("genome coordinate map must be invertible");
        work.swap(column, pivot);
        let pivot_value = work[column][column].clone();
        for value in &mut work[column] {
            *value /= pivot_value.clone();
        }
        for row in 0..dimension {
            if row == column || work[row][column] == zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next in 0..2 * dimension {
                let subtraction = factor.clone() * work[column][next].clone();
                work[row][next] -= subtraction;
            }
        }
    }
    work.into_iter()
        .map(|row| row[dimension..].to_vec())
        .collect()
}

fn factorial(degree: u8) -> usize {
    match degree {
        0 | 1 => 1,
        2 => 2,
        _ => panic!("4D N=1 exterior degree exceeds two"),
    }
}

fn exterior_masks(left: bool, degree: u8) -> Vec<u8> {
    match (left, degree) {
        (_, 0) => vec![0],
        (true, 1) => vec![1, 2],
        (false, 1) => vec![4, 8],
        (true, 2) => vec![3],
        (false, 2) => vec![12],
        _ => panic!("invalid exterior degree"),
    }
}

fn factor_embedding(degree: u8, target_label: u8) -> Vec<Vec<Ratio<i64>>> {
    match degree {
        0 | 2 => {
            assert_eq!(target_label, 1);
            vec![
                vec![Ratio::from_integer(1), Ratio::from_integer(0)],
                vec![Ratio::from_integer(0), Ratio::from_integer(1)],
            ]
        }
        1 if target_label == 2 => upper_channel(1).0,
        1 if target_label == 0 => lower_channel(1).expect("lower channel at n=1").0,
        _ => panic!("target label is absent from [1] tensor exterior degree"),
    }
}

fn supergravity_genome() -> Vec<GenomeTerm> {
    adynkra_genome::artifact()
        .genomes
        .into_iter()
        .find(|genome| genome.id == "supergravity")
        .expect("published supergravity genome")
        .terms
}

fn supergravity_coordinate_maps() -> (Matrix, Matrix, Vec<GenomeBlock>) {
    let terms = supergravity_genome();
    let mut embedding = zeros(64, 64);
    let mut blocks = Vec::with_capacity(terms.len());
    let mut offset = 0;
    for term in terms {
        assert_eq!(term.multiplicity, 1);
        let left_embedding = factor_embedding(term.left_degree, term.irrep.left);
        let right_embedding = factor_embedding(term.right_degree, term.irrep.right);
        let left_masks = exterior_masks(true, term.left_degree);
        let right_masks = exterior_masks(false, term.right_degree);
        let denominator = factorial(term.left_degree) * factorial(term.right_degree);
        assert_eq!(term.coefficient_denominator, denominator);
        for target_left in 0..=term.irrep.left as usize {
            for target_right in 0..=term.irrep.right as usize {
                let column = offset + target_left * (term.irrep.right as usize + 1) + target_right;
                for alpha in 0..2 {
                    for (left_state, &left_mask) in left_masks.iter().enumerate() {
                        let left_row = left_masks.len() * alpha + left_state;
                        let left_coefficient = left_embedding[left_row][target_left];
                        if left_coefficient == Ratio::from_integer(0) {
                            continue;
                        }
                        for dotted in 0..2 {
                            for (right_state, &right_mask) in right_masks.iter().enumerate() {
                                let right_row = right_masks.len() * dotted + right_state;
                                let right_coefficient = right_embedding[right_row][target_right];
                                if right_coefficient == Ratio::from_integer(0) {
                                    continue;
                                }
                                let component = 2 * alpha + dotted;
                                let mask = left_mask | right_mask;
                                let row = component * GRASSMANN_DIMENSION + mask as usize;
                                embedding[row][column] += Complex::new(
                                    left_coefficient * right_coefficient
                                        / Ratio::from_integer(denominator as i64),
                                    Ratio::from_integer(0),
                                );
                            }
                        }
                    }
                }
            }
        }
        blocks.push(GenomeBlock {
            left_degree: term.left_degree,
            right_degree: term.right_degree,
            irrep: term.irrep,
            dimension: term.irrep.dimension(),
            coordinate_offset: offset,
            factorial_denominator: denominator,
        });
        offset += term.irrep.dimension();
    }
    assert_eq!(offset, 64);
    let projection = inverse(&embedding);
    (embedding, projection, blocks)
}

fn evaluate(polynomial: &Polynomial, momentum: [i64; 4]) -> Vec<GaussianRational> {
    let mut result = vec![zero(); GRASSMANN_DIMENSION];
    for (monomial, source_coefficient) in &polynomial.0 {
        let mut coefficient = source_coefficient.clone();
        for (index, &power) in monomial.spacetime_derivatives.iter().enumerate() {
            for _ in 0..power {
                coefficient *= Ratio::from_integer(momentum[index]);
            }
        }
        result[monomial.grassmann_mask as usize] += coefficient;
    }
    result
}

fn columns_to_matrix(columns: &[Vec<GaussianRational>]) -> Matrix {
    let mut result = zeros(columns[0].len(), columns.len());
    for (column, values) in columns.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            result[row][column] = value.clone();
        }
    }
    result
}

fn pivot_columns(matrix: &Matrix) -> Vec<usize> {
    let mut work = matrix.clone();
    let rows = work.len();
    let columns = work[0].len();
    let mut pivots = Vec::new();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| work[row][column] != zero()) else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..rows {
            if row == pivot_row || work[row][column] == zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next in column..columns {
                let subtraction = factor.clone() * work[pivot_row][next].clone();
                work[row][next] -= subtraction;
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    pivots
}

fn chiral_coordinate_maps(momentum: [i64; 4], conjugate: bool) -> (Matrix, Matrix) {
    let columns: Vec<_> = (0..GRASSMANN_DIMENSION as u8)
        .map(|mask| {
            let polynomial = if conjugate {
                d_squared(&Polynomial::basis(mask))
            } else {
                d_bar_squared(&Polynomial::basis(mask))
            };
            evaluate(&polynomial, momentum)
        })
        .collect();
    let spanning = columns_to_matrix(&columns);
    let leading_rows = if conjugate {
        [0usize, 4, 8, 12]
    } else {
        [0usize, 1, 2, 3]
    };
    let leading: Matrix = leading_rows
        .iter()
        .map(|&row| spanning[row].clone())
        .collect();
    let pivots = pivot_columns(&leading);
    assert_eq!(pivots.len(), 4);
    let selected: Matrix = (0..16)
        .map(|row| {
            pivots
                .iter()
                .map(|&column| spanning[row][column].clone())
                .collect()
        })
        .collect();
    let leading_square: Matrix = leading_rows
        .iter()
        .map(|&row| {
            pivots
                .iter()
                .map(|&column| spanning[row][column].clone())
                .collect()
        })
        .collect();
    let normalized = multiply(&selected, &inverse(&leading_square));
    let mut embedding = normalized;
    for row in &mut embedding {
        row[3] *= Ratio::new(1, 2);
    }
    let mut projection = zeros(4, 16);
    for (coordinate, &row) in leading_rows.iter().enumerate() {
        projection[coordinate][row] = if coordinate == 3 {
            gaussian(2, 0, 1)
        } else {
            gaussian(1, 0, 1)
        };
    }
    assert_eq!(multiply(&projection, &embedding), identity(4));
    (embedding, projection)
}

fn total_coordinate_maps(
    momentum: [i64; 4],
    h_embedding: &Matrix,
    h_projection: &Matrix,
) -> (Matrix, Matrix) {
    let (chi_embedding, chi_projection) = chiral_coordinate_maps(momentum, false);
    let (chi_bar_embedding, chi_bar_projection) = chiral_coordinate_maps(momentum, true);
    let mut embedding = zeros(AMBIENT_DIMENSION, ADYNKRAFIELD_DIMENSION);
    let mut projection = zeros(ADYNKRAFIELD_DIMENSION, AMBIENT_DIMENSION);
    for row in 0..64 {
        for column in 0..64 {
            embedding[row][column] = h_embedding[row][column].clone();
            projection[row][column] = h_projection[row][column].clone();
        }
    }
    for row in 0..16 {
        for column in 0..4 {
            embedding[64 + row][64 + column] = chi_embedding[row][column].clone();
            embedding[80 + row][68 + column] = chi_bar_embedding[row][column].clone();
            projection[64 + column][64 + row] = chi_projection[column][row].clone();
            projection[68 + column][80 + row] = chi_bar_projection[column][row].clone();
        }
    }
    (embedding, projection)
}

fn append(output: &mut Vec<GaussianRational>, polynomial: &Polynomial, momentum: [i64; 4]) {
    output.extend(evaluate(polynomial, momentum));
}

fn superspace_operator(momentum: [i64; 4]) -> Matrix {
    let mut columns = Vec::with_capacity(AMBIENT_DIMENSION);
    for input in 0..AMBIENT_DIMENSION {
        let mut h = zero_h();
        let mut chi = Polynomial::default();
        let mut chi_bar = Polynomial::default();
        if input < 64 {
            let component = input / 16;
            h[component / 2][component % 2] = Polynomial::basis((input % 16) as u8);
        } else if input < 80 {
            chi = Polynomial::basis((input - 64) as u8);
        } else {
            chi_bar = Polynomial::basis((input - 80) as u8);
        }
        let mut output = Vec::with_capacity(AMBIENT_DIMENSION);
        for component in 0..4 {
            append(
                &mut output,
                &vector_curvature(&h, &chi, &chi_bar, component),
                momentum,
            );
        }
        append(&mut output, &scalar_curvature(&h, &chi_bar), momentum);
        append(&mut output, &conjugate_scalar_curvature(&h, &chi), momentum);
        columns.push(output);
    }
    columns_to_matrix(&columns)
}

fn ambient_potential(
    h: &[Vec<Polynomial>],
    chi: &Polynomial,
    chi_bar: &Polynomial,
    momentum: [i64; 4],
) -> Vec<GaussianRational> {
    let mut result = Vec::with_capacity(AMBIENT_DIMENSION);
    for row in h {
        for component in row {
            append(&mut result, component, momentum);
        }
    }
    append(&mut result, chi, momentum);
    append(&mut result, chi_bar, momentum);
    result
}

fn gauge_matrix(momentum: [i64; 4]) -> Matrix {
    let mut columns = Vec::with_capacity(64);
    for chirality in 0..2 {
        for spinor in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                let (h, chi, chi_bar) = if chirality == 0 {
                    let (h, chi) = gauge_image_l(spinor, mask);
                    (h, chi, Polynomial::default())
                } else {
                    let (h, chi_bar) = gauge_image_l_bar(spinor, mask, true);
                    (h, Polynomial::default(), chi_bar)
                };
                columns.push(ambient_potential(&h, &chi, &chi_bar, momentum));
            }
        }
    }
    columns_to_matrix(&columns)
}

fn momentum_class(momentum: [i64; 4]) -> &'static str {
    if momentum == [0; 4] {
        "zero"
    } else if momentum[0] * momentum[3] - momentum[1] * momentum[2] == 0 {
        "null"
    } else {
        "non-null"
    }
}

fn momentum_check(
    momentum: [i64; 4],
    h_embedding: &Matrix,
    h_projection: &Matrix,
) -> AdynkrafieldMomentumCheck {
    let (embedding, projection) = total_coordinate_maps(momentum, h_embedding, h_projection);
    let coordinate_roundtrip_residuals =
        matrix_residuals(&multiply(&projection, &embedding), &identity(72));
    let superspace = superspace_operator(momentum);
    let restricted_superspace = multiply(&superspace, &embedding);
    let adynkrafield = multiply(&projection, &restricted_superspace);
    let reconstructed = multiply(&embedding, &adynkrafield);
    let operator_reconstruction_residuals =
        matrix_residuals(&reconstructed, &restricted_superspace);
    let gauge = gauge_matrix(momentum);
    let gauge_coordinates = multiply(&projection, &gauge);
    let gauge_reconstructed = multiply(&embedding, &gauge_coordinates);
    let gauge_coordinate_reconstruction_residuals = matrix_residuals(&gauge_reconstructed, &gauge);
    let gauge_noether = multiply(&adynkrafield, &gauge_coordinates);
    let gauge_noether_residuals = gauge_noether
        .iter()
        .flatten()
        .filter(|value| **value != zero())
        .count();
    AdynkrafieldMomentumCheck {
        momentum,
        momentum_class: momentum_class(momentum),
        domain_embedding_rank: rank(&embedding),
        coordinate_roundtrip_residuals,
        superspace_operator_rank_on_domain: rank(&restricted_superspace),
        adynkrafield_operator_rank: rank(&adynkrafield),
        adynkrafield_operator_nonzero_entries: adynkrafield
            .iter()
            .flatten()
            .filter(|value| **value != zero())
            .count(),
        operator_reconstruction_entries_checked: AMBIENT_DIMENSION * ADYNKRAFIELD_DIMENSION,
        operator_reconstruction_residuals,
        gauge_columns_checked: 64,
        gauge_coordinate_reconstruction_residuals,
        gauge_noether_residuals,
    }
}

pub fn verify() -> AdynkrafieldOperatorReport {
    let (h_embedding, h_projection, supergravity_genome_blocks) = supergravity_coordinate_maps();
    let supergravity_change_of_basis_rank = rank(&h_embedding);
    let momenta = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 2, 3, 6],
        [0, 1, 1, 0],
        [1, 2, 3, 5],
        [2, -1, 4, 3],
    ];
    let momentum_checks: Vec<_> = momenta
        .into_iter()
        .map(|momentum| momentum_check(momentum, &h_embedding, &h_projection))
        .collect();
    let total_operator_reconstruction_entries_checked = momentum_checks
        .iter()
        .map(|check| check.operator_reconstruction_entries_checked)
        .sum();
    let total_operator_reconstruction_residuals = momentum_checks
        .iter()
        .map(|check| check.operator_reconstruction_residuals)
        .sum();
    let total_gauge_coordinate_reconstruction_residuals = momentum_checks
        .iter()
        .map(|check| check.gauge_coordinate_reconstruction_residuals)
        .sum();
    let total_gauge_noether_residuals = momentum_checks
        .iter()
        .map(|check| check.gauge_noether_residuals)
        .sum();
    let observed_operator_ranks: Vec<_> = momentum_checks
        .iter()
        .map(|check| check.adynkrafield_operator_rank)
        .collect();
    let expected_operator_ranks = vec![6, 20, 20, 20, 24, 24, 24];
    let equivalent = total_operator_reconstruction_residuals == 0
        && total_gauge_coordinate_reconstruction_residuals == 0
        && total_gauge_noether_residuals == 0
        && momentum_checks.iter().all(|check| {
            check.domain_embedding_rank == ADYNKRAFIELD_DIMENSION
                && check.coordinate_roundtrip_residuals == 0
                && check.superspace_operator_rank_on_domain == check.adynkrafield_operator_rank
        });
    let factorial_coefficients_enforced = supergravity_genome_blocks.iter().all(|block| {
        block.factorial_denominator == factorial(block.left_degree) * factorial(block.right_degree)
    });
    let passed = supergravity_change_of_basis_rank == 64
        && supergravity_genome_blocks.len() == 16
        && factorial_coefficients_enforced
        && observed_operator_ranks == expected_operator_ranks
        && equivalent;
    AdynkrafieldOperatorReport {
        schema_version: "adynkra-4d-n1-old-minimal-operator-v1",
        source_arxiv: "2407.09334v1 and hep-th/0108200",
        source_equations: "Adynkra genomes Eqs. (3.6) and (3.11), Adynkrafield definition Eq. (4.4), and examples Eqs. (4.6) and (4.13)-(4.14)",
        operator_source: "Superspace Eqs. (5.5.45), (5.5.48), and (7.5.19)",
        operator_equation: "E_AF = P_AF (G, R, Rbar) U_AF; E_AF Phi_AF = 0",
        coordinate_definition: "published Lorentz-genome blocks with rational Clebsch-Gordan embeddings and the 1/(p! q!) Adynkrafield coefficients",
        superspace_ambient_dimension: AMBIENT_DIMENSION,
        adynkrafield_domain_dimension: ADYNKRAFIELD_DIMENSION,
        supergravity_genome_dimension: 64,
        chiral_compensator_genome_dimension: 4,
        conjugate_compensator_genome_dimension: 4,
        supergravity_genome_blocks,
        supergravity_change_of_basis_rank,
        factorial_coefficients_enforced,
        momentum_checks,
        total_operator_reconstruction_entries_checked,
        total_operator_reconstruction_residuals,
        total_gauge_coordinate_reconstruction_residuals,
        total_gauge_noether_residuals,
        expected_operator_ranks,
        observed_operator_ranks,
        superspace_and_adynkrafield_operators_equivalent_on_tested_fibers: equivalent,
        boundary: "exact coordinate translation of the known linearized 4D N=1 old-minimal equation on seven momentum fibers; it does not derive a new equation, prove polynomial-module equivalence, or choose eleven-dimensional constraints",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_supergravity_genome_is_an_exact_coordinate_basis() {
        let (embedding, projection, blocks) = supergravity_coordinate_maps();
        assert_eq!(blocks.len(), 16);
        assert_eq!(rank(&embedding), 64);
        assert_eq!(multiply(&projection, &embedding), identity(64));
        assert_eq!(
            blocks.iter().map(|block| block.dimension).sum::<usize>(),
            64
        );
    }

    #[test]
    fn chiral_genome_maps_round_trip_at_zero_null_and_non_null_momentum() {
        for momentum in [[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 3, 5]] {
            for conjugate in [false, true] {
                let (embedding, projection) = chiral_coordinate_maps(momentum, conjugate);
                assert_eq!(rank(&embedding), 4);
                assert_eq!(multiply(&projection, &embedding), identity(4));
            }
        }
    }

    #[test]
    fn adynkrafield_operator_reconstructs_the_superspace_equation() {
        let report = verify();
        assert_eq!(report.total_operator_reconstruction_residuals, 0);
        assert_eq!(report.total_gauge_coordinate_reconstruction_residuals, 0);
        assert_eq!(report.total_gauge_noether_residuals, 0);
        assert_eq!(
            report.observed_operator_ranks,
            report.expected_operator_ranks
        );
        assert!(report.superspace_and_adynkrafield_operators_equivalent_on_tested_fibers);
        assert!(report.passed);
    }
}
