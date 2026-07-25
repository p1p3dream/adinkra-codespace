//! Exact superspace gauge map for the 4D N=1 supergravity prepotential in
//! Gates and Hu, arXiv:2407.09334v1, Eq. (2.21).

use crate::supercovariant_derivative::{Derivative, GaussianRational, Polynomial, apply};
use num_complex::Complex;
use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;

const GRASSMANN_DIMENSION: usize = 16;
const SPINOR_DIMENSION: usize = 2;
const DOMAIN_DIMENSION: usize = 64;
const CODOMAIN_DIMENSION: usize = 64;

type Matrix = Vec<Vec<GaussianRational>>;

#[derive(Debug, Clone, Serialize)]
pub struct MomentumRankCheck {
    pub momentum: [i64; 4],
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrepotentialGaugeReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_equations: &'static str,
    pub gauge_map: &'static str,
    pub convention: &'static str,
    pub parameter_components: usize,
    pub prepotential_components: usize,
    pub sparse_operator_terms: usize,
    pub zero_derivative_terms: usize,
    pub first_derivative_terms: usize,
    pub maximum_spacetime_derivative_order: usize,
    pub level_selection_terms_checked: usize,
    pub level_selection_violations: usize,
    pub zero_momentum_rank: usize,
    pub momentum_rank_checks: Vec<MomentumRankCheck>,
    pub boundary: &'static str,
    pub passed: bool,
}

fn zero() -> GaussianRational {
    Complex::new(Ratio::from_integer(0), Ratio::from_integer(0))
}

fn one() -> GaussianRational {
    Complex::new(Ratio::from_integer(1), Ratio::from_integer(0))
}

fn minus_one() -> GaussianRational {
    Complex::new(Ratio::from_integer(-1), Ratio::from_integer(0))
}

fn domain_l(alpha: usize, mask: u8) -> usize {
    GRASSMANN_DIMENSION * alpha + mask as usize
}

fn domain_l_bar(dotted: usize, mask: u8) -> usize {
    2 * GRASSMANN_DIMENSION + GRASSMANN_DIMENSION * dotted + mask as usize
}

fn codomain_h(alpha: usize, dotted: usize, mask: u8) -> usize {
    GRASSMANN_DIMENSION * (2 * alpha + dotted) + mask as usize
}

#[derive(Debug, Clone)]
struct OperatorTerm {
    row: usize,
    column: usize,
    input_mask: u8,
    output_mask: u8,
    spacetime_derivatives: [u8; 4],
    coefficient: GaussianRational,
}

fn append_application(
    terms: &mut Vec<OperatorTerm>,
    row_spinor: (usize, usize),
    column: usize,
    input_mask: u8,
    derivative: Derivative,
    sign: GaussianRational,
) {
    let output = apply(derivative, &Polynomial::basis(input_mask));
    for (monomial, coefficient) in output.0 {
        terms.push(OperatorTerm {
            row: codomain_h(row_spinor.0, row_spinor.1, monomial.grassmann_mask),
            column,
            input_mask,
            output_mask: monomial.grassmann_mask,
            spacetime_derivatives: monomial.spacetime_derivatives,
            coefficient: coefficient * sign.clone(),
        });
    }
}

fn operator_terms() -> Vec<OperatorTerm> {
    let mut terms = Vec::new();
    for alpha in 0..SPINOR_DIMENSION {
        for dotted in 0..SPINOR_DIMENSION {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                // delta H_{alpha dot-alpha} = D_alpha Lbar_dot-alpha
                //                               - Dbar_dot-alpha L_alpha.
                append_application(
                    &mut terms,
                    (alpha, dotted),
                    domain_l_bar(dotted, mask),
                    mask,
                    Derivative::Left(alpha),
                    one(),
                );
                append_application(
                    &mut terms,
                    (alpha, dotted),
                    domain_l(alpha, mask),
                    mask,
                    Derivative::Right(dotted),
                    minus_one(),
                );
            }
        }
    }
    terms
}

fn evaluate(terms: &[OperatorTerm], momentum: [i64; 4]) -> Matrix {
    let mut matrix = vec![vec![zero(); DOMAIN_DIMENSION]; CODOMAIN_DIMENSION];
    for term in terms {
        let mut coefficient = term.coefficient.clone();
        for (index, &power) in term.spacetime_derivatives.iter().enumerate() {
            for _ in 0..power {
                coefficient *= Ratio::from_integer(momentum[index]);
            }
        }
        matrix[term.row][term.column] += coefficient;
    }
    matrix
}

fn rank(matrix: &Matrix) -> usize {
    let mut work = matrix.clone();
    let rows = work.len();
    let columns = work[0].len();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| !work[row][column].is_zero()) else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..rows {
            if row == pivot_row || work[row][column].is_zero() {
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

fn grassmann_bidegree(mask: u8) -> [i8; 2] {
    [
        (mask & 0b0011).count_ones() as i8,
        ((mask >> 2) & 0b0011).count_ones() as i8,
    ]
}

fn respects_level_selection(term: &OperatorTerm) -> bool {
    let input = grassmann_bidegree(term.input_mask);
    let output = grassmann_bidegree(term.output_mask);
    let derivative_order: u8 = term.spacetime_derivatives.iter().sum();
    match derivative_order {
        0 => {
            (output[0] == input[0] - 1 && output[1] == input[1])
                || (output[0] == input[0] && output[1] == input[1] - 1)
        }
        1 => {
            (output[0] == input[0] + 1 && output[1] == input[1])
                || (output[0] == input[0] && output[1] == input[1] + 1)
        }
        _ => false,
    }
}

pub fn verify() -> PrepotentialGaugeReport {
    let terms = operator_terms();
    let zero_derivative_terms = terms
        .iter()
        .filter(|term| term.spacetime_derivatives.iter().all(|&power| power == 0))
        .count();
    let first_derivative_terms = terms.len() - zero_derivative_terms;
    let maximum_spacetime_derivative_order = terms
        .iter()
        .map(|term| {
            term.spacetime_derivatives
                .iter()
                .map(|&power| power as usize)
                .sum()
        })
        .max()
        .unwrap_or(0);
    let level_selection_violations = terms
        .iter()
        .filter(|term| !respects_level_selection(term))
        .count();
    let zero_momentum_rank = rank(&evaluate(&terms, [0; 4]));
    let momenta = [[1, 0, 0, 0], [0, 1, 1, 0], [1, 2, 3, 5], [2, -1, 4, 3]];
    let momentum_rank_checks: Vec<_> = momenta
        .into_iter()
        .map(|momentum| MomentumRankCheck {
            momentum,
            rank: rank(&evaluate(&terms, momentum)),
        })
        .collect();
    let passed = terms.len() == 192
        && zero_derivative_terms == 64
        && first_derivative_terms == 128
        && maximum_spacetime_derivative_order == 1
        && level_selection_violations == 0;
    PrepotentialGaugeReport {
        schema_version: "adynkra-4d-n1-prepotential-gauge-v1",
        source_arxiv: "2407.09334",
        source_equations: "2.21 and 2.22",
        gauge_map: "delta H_{alpha dot-alpha} = D_alpha Lbar_dot-alpha - Dbar_dot-alpha L_alpha",
        convention: "L_alpha and its conjugate are independent over the complexified component space",
        parameter_components: DOMAIN_DIMENSION,
        prepotential_components: CODOMAIN_DIMENSION,
        sparse_operator_terms: terms.len(),
        zero_derivative_terms,
        first_derivative_terms,
        maximum_spacetime_derivative_order,
        level_selection_terms_checked: terms.len(),
        level_selection_violations,
        zero_momentum_rank,
        momentum_rank_checks,
        boundary: "this report covers the exact superspace gauge map; the old-minimal curvature complex is validated in separate artifacts; named-component matching, polynomial-module cohomology, and the Euler-Lagrange equation are separate",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equation_2_21_has_the_expected_sparse_differential_structure() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.sparse_operator_terms, 192);
        assert_eq!(report.zero_derivative_terms, 64);
        assert_eq!(report.first_derivative_terms, 128);
        assert_eq!(report.level_selection_violations, 0);
    }

    #[test]
    fn gauge_map_ranks_are_exact_at_zero_and_sampled_momenta() {
        let report = verify();
        assert_eq!(report.zero_momentum_rank, 39);
        assert_eq!(
            report
                .momentum_rank_checks
                .iter()
                .map(|check| check.rank)
                .collect::<Vec<_>>(),
            vec![48, 48, 48, 48]
        );
    }
}
