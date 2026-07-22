//! Exact vector-spinor projectors for arXiv:2407.09334v1,
//! Eqs. (2.13), (2.14), and (2.19).

use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;

type Rat = Ratio<i64>;
type Matrix = Vec<Vec<Rat>>;

#[derive(Debug, Clone, Serialize)]
pub struct VectorSpinorProjectorCheck {
    pub chirality: &'static str,
    pub sector: &'static str,
    pub dynkin_label: [u8; 2],
    pub expected_rank: usize,
    pub computed_rank: usize,
    pub idempotent: bool,
    pub equivariant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorSpinorIntertwinerReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_equations: &'static str,
    pub decompositions: [&'static str; 2],
    pub vector_spinor_dimension: usize,
    pub projectors: Vec<VectorSpinorProjectorCheck>,
    pub completeness_checks: usize,
    pub completeness_failures: usize,
    pub orthogonality_checks: usize,
    pub orthogonality_failures: usize,
    pub sl2_generators_per_chirality: usize,
    pub equivariance_commutators_checked: usize,
    pub boundary: &'static str,
    pub passed: bool,
}

fn zeros(dimension: usize) -> Matrix {
    vec![vec![Rat::zero(); dimension]; dimension]
}

fn identity(dimension: usize) -> Matrix {
    let mut result = zeros(dimension);
    for index in 0..dimension {
        result[index][index] = Rat::from_integer(1);
    }
    result
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.clone() + y.clone())
                .collect()
        })
        .collect()
}

fn subtract(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.clone() - y.clone())
                .collect()
        })
        .collect()
}

fn scale(matrix: &Matrix, scalar: Rat) -> Matrix {
    matrix
        .iter()
        .map(|row| row.iter().map(|x| x.clone() * scalar.clone()).collect())
        .collect()
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    let mut result = zeros(left.len());
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot].is_zero() {
                continue;
            }
            for column in 0..right.len() {
                result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
            }
        }
    }
    result
}

fn rank(matrix: &Matrix) -> usize {
    let mut work = matrix.clone();
    let mut pivot_row = 0;
    for column in 0..work.len() {
        let Some(found) = (pivot_row..work.len()).find(|&row| !work[row][column].is_zero()) else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..work.len() {
            if row == pivot_row || work[row][column].is_zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next in column..work.len() {
                let value = factor.clone() * work[pivot_row][next].clone();
                work[row][next] -= value;
            }
        }
        pivot_row += 1;
    }
    pivot_row
}

fn index(first: usize, spectator: usize, third: usize) -> usize {
    4 * first + 2 * spectator + third
}

fn exchange_projectors() -> [Matrix; 2] {
    let mut exchange = zeros(8);
    for first in 0..2 {
        for spectator in 0..2 {
            for third in 0..2 {
                exchange[index(first, spectator, third)][index(third, spectator, first)] =
                    Rat::from_integer(1);
            }
        }
    }
    let unit = identity(8);
    [
        scale(&subtract(&unit, &exchange), Rat::new(1, 2)),
        scale(&add(&unit, &exchange), Rat::new(1, 2)),
    ]
}

fn fundamental_generators() -> [Matrix; 3] {
    let mut e = zeros(2);
    e[0][1] = Rat::from_integer(1);
    let mut f = zeros(2);
    f[1][0] = Rat::from_integer(1);
    let mut h = zeros(2);
    h[0][0] = Rat::from_integer(1);
    h[1][1] = Rat::from_integer(-1);
    [e, f, h]
}

fn induced_generators(active_pair_is_left: bool) -> Vec<Matrix> {
    let fundamental = fundamental_generators();
    let mut result = Vec::new();
    for factor_is_left in [true, false] {
        for generator in &fundamental {
            let mut induced = zeros(8);
            for first in 0..2 {
                for spectator in 0..2 {
                    for third in 0..2 {
                        let source = index(first, spectator, third);
                        if factor_is_left == active_pair_is_left {
                            for output in 0..2 {
                                induced[index(output, spectator, third)][source] +=
                                    generator[output][first].clone();
                                induced[index(first, spectator, output)][source] +=
                                    generator[output][third].clone();
                            }
                        } else {
                            for output in 0..2 {
                                induced[index(first, output, third)][source] +=
                                    generator[output][spectator].clone();
                            }
                        }
                    }
                }
            }
            result.push(induced);
        }
    }
    result
}

pub fn verify() -> VectorSpinorIntertwinerReport {
    let projectors = exchange_projectors();
    let cases = [
        (
            "left",
            true,
            [
                ("spinor_trace", [0, 1], 2),
                ("spin_three_halves", [2, 1], 6),
            ],
        ),
        (
            "right",
            false,
            [
                ("spinor_trace", [1, 0], 2),
                ("spin_three_halves", [1, 2], 6),
            ],
        ),
    ];
    let mut checks = Vec::new();
    let mut completeness_failures = 0;
    let mut orthogonality_failures = 0;
    let mut equivariance_commutators_checked = 0;
    for (chirality, active_pair_is_left, labels) in cases {
        let generators = induced_generators(active_pair_is_left);
        if add(&projectors[0], &projectors[1]) != identity(8) {
            completeness_failures += 1;
        }
        for (projector, (sector, dynkin_label, expected_rank)) in projectors.iter().zip(labels) {
            let equivariant = generators.iter().all(|generator| {
                equivariance_commutators_checked += 1;
                multiply(generator, projector) == multiply(projector, generator)
            });
            checks.push(VectorSpinorProjectorCheck {
                chirality,
                sector,
                dynkin_label,
                expected_rank,
                computed_rank: rank(projector),
                idempotent: multiply(projector, projector) == *projector,
                equivariant,
            });
        }
        if multiply(&projectors[0], &projectors[1]) != zeros(8) {
            orthogonality_failures += 1;
        }
        if multiply(&projectors[1], &projectors[0]) != zeros(8) {
            orthogonality_failures += 1;
        }
    }
    let passed = completeness_failures == 0
        && orthogonality_failures == 0
        && checks.iter().all(|check| {
            check.expected_rank == check.computed_rank && check.idempotent && check.equivariant
        });
    VectorSpinorIntertwinerReport {
        schema_version: "adynkra-4d-n1-vector-spinor-intertwiners-v1",
        source_arxiv: "2407.09334",
        source_equations: "2.13, 2.14, 2.19",
        decompositions: [
            "[1,1] tensor [1,0] = [0,1] + [2,1]",
            "[1,1] tensor [0,1] = [1,0] + [1,2]",
        ],
        vector_spinor_dimension: 8,
        projectors: checks,
        completeness_checks: 2,
        completeness_failures,
        orthogonality_checks: 4,
        orthogonality_failures,
        sl2_generators_per_chirality: 6,
        equivariance_commutators_checked,
        boundary: "vector-spinor irreducible projectors only; repeated-irrep derivative maps and gauge cohomology remain open",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_spinor_projectors_reproduce_both_source_decompositions() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.projectors.len(), 4);
        assert_eq!(report.equivariance_commutators_checked, 24);
        assert_eq!(report.completeness_failures, 0);
        assert_eq!(report.orthogonality_failures, 0);
    }
}
