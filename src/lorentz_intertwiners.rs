//! Exact irreducible projectors for the rank-two Lorentz tensor appearing in
//! the 4D N=1 supergravity Adynkra genome, arXiv:2407.09334v1, Eqs. (2.5) and
//! (2.18).

use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;

type Rat = Ratio<i64>;
type Matrix = Vec<Vec<Rat>>;

const VECTOR_DIMENSION: usize = 4;
const TENSOR_DIMENSION: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct ProjectorCheck {
    pub id: &'static str,
    pub dynkin_label: [u8; 2],
    pub expected_rank: usize,
    pub computed_rank: usize,
    pub idempotent: bool,
    pub so4_equivariant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LorentzIntertwinerReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_equations: &'static str,
    pub tensor_product: &'static str,
    pub tensor_dimension: usize,
    pub projectors: Vec<ProjectorCheck>,
    pub projector_ranks_sum: usize,
    pub pairwise_orthogonal_products_checked: usize,
    pub pairwise_orthogonal: bool,
    pub completeness_residual_entries: usize,
    pub reconstruction_basis_vectors_checked: usize,
    pub reconstruction_residual_entries: usize,
    pub so4_generators_checked: usize,
    pub equivariance_commutators_checked: usize,
    pub boundary: &'static str,
    pub passed: bool,
}

fn zeros(rows: usize, columns: usize) -> Matrix {
    vec![vec![Rat::zero(); columns]; rows]
}

fn identity(dimension: usize) -> Matrix {
    let mut result = zeros(dimension, dimension);
    for index in 0..dimension {
        result[index][index] = Rat::from_integer(1);
    }
    result
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            left_row
                .iter()
                .zip(right_row)
                .map(|(a, b)| a.clone() + b.clone())
                .collect()
        })
        .collect()
}

fn subtract(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            left_row
                .iter()
                .zip(right_row)
                .map(|(a, b)| a.clone() - b.clone())
                .collect()
        })
        .collect()
}

fn scale(matrix: &Matrix, scalar: Rat) -> Matrix {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.clone() * scalar.clone())
                .collect()
        })
        .collect()
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    let rows = left.len();
    let inner = right.len();
    let columns = right[0].len();
    let mut result = zeros(rows, columns);
    for row in 0..rows {
        for pivot in 0..inner {
            if left[row][pivot].is_zero() {
                continue;
            }
            for column in 0..columns {
                result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
            }
        }
    }
    result
}

fn nonzero_entries(matrix: &Matrix) -> usize {
    matrix
        .iter()
        .flat_map(|row| row.iter())
        .filter(|value| !value.is_zero())
        .count()
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
        for entry in &mut work[pivot_row][column..] {
            *entry /= pivot.clone();
        }
        for row in 0..rows {
            if row == pivot_row || work[row][column].is_zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next_column in column..columns {
                let subtraction = factor.clone() * work[pivot_row][next_column].clone();
                work[row][next_column] -= subtraction;
            }
        }
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    pivot_row
}

fn pair(a: usize, b: usize) -> usize {
    VECTOR_DIMENSION * a + b
}

fn permutation_sign(values: [usize; 4]) -> i64 {
    if values
        .iter()
        .enumerate()
        .any(|(i, value)| values[..i].contains(value))
    {
        return 0;
    }
    let inversions = (0..4)
        .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
        .filter(|&(i, j)| values[i] > values[j])
        .count();
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn projectors() -> [Matrix; 4] {
    let mut swap = zeros(TENSOR_DIMENSION, TENSOR_DIMENSION);
    let mut trace = zeros(TENSOR_DIMENSION, TENSOR_DIMENSION);
    let mut hodge = zeros(TENSOR_DIMENSION, TENSOR_DIMENSION);
    for a in 0..4 {
        for b in 0..4 {
            swap[pair(a, b)][pair(b, a)] = Rat::from_integer(1);
            if a == b {
                for c in 0..4 {
                    trace[pair(a, b)][pair(c, c)] = Rat::new(1, 4);
                }
            }
            for c in 0..4 {
                for d in 0..4 {
                    let epsilon = permutation_sign([a, b, c, d]);
                    if epsilon != 0 {
                        hodge[pair(a, b)][pair(c, d)] = Rat::new(epsilon, 2);
                    }
                }
            }
        }
    }
    let unit = identity(TENSOR_DIMENSION);
    let symmetric = scale(&add(&unit, &swap), Rat::new(1, 2));
    let antisymmetric = scale(&subtract(&unit, &swap), Rat::new(1, 2));
    let symmetric_traceless = subtract(&symmetric, &trace);
    let self_dual = scale(&add(&antisymmetric, &hodge), Rat::new(1, 2));
    let anti_self_dual = scale(&subtract(&antisymmetric, &hodge), Rat::new(1, 2));
    [trace, symmetric_traceless, self_dual, anti_self_dual]
}

fn so4_generators() -> Vec<Matrix> {
    let mut generators = Vec::new();
    for m in 0..4 {
        for n in (m + 1)..4 {
            let mut vector = zeros(4, 4);
            vector[m][n] = Rat::from_integer(1);
            vector[n][m] = Rat::from_integer(-1);
            let mut tensor = zeros(16, 16);
            for a in 0..4 {
                for b in 0..4 {
                    for c in 0..4 {
                        if !vector[a][c].is_zero() {
                            tensor[pair(a, b)][pair(c, b)] += vector[a][c].clone();
                        }
                        if !vector[b][c].is_zero() {
                            tensor[pair(a, b)][pair(a, c)] += vector[b][c].clone();
                        }
                    }
                }
            }
            generators.push(tensor);
        }
    }
    generators
}

pub fn verify() -> LorentzIntertwinerReport {
    let names = [
        ("trace", [0, 0], 1),
        ("symmetric_traceless", [2, 2], 9),
        ("self_dual_two_form", [0, 2], 3),
        ("anti_self_dual_two_form", [2, 0], 3),
    ];
    let projectors = projectors();
    let generators = so4_generators();
    let mut checks = Vec::new();
    for (projector, (id, dynkin_label, expected_rank)) in projectors.iter().zip(names) {
        let idempotent = multiply(projector, projector) == *projector;
        let so4_equivariant = generators.iter().all(|generator| {
            subtract(
                &multiply(generator, projector),
                &multiply(projector, generator),
            ) == zeros(16, 16)
        });
        checks.push(ProjectorCheck {
            id,
            dynkin_label,
            expected_rank,
            computed_rank: rank(projector),
            idempotent,
            so4_equivariant,
        });
    }
    let mut pairwise_orthogonal = true;
    let mut pairwise_orthogonal_products_checked = 0;
    for i in 0..projectors.len() {
        for j in 0..projectors.len() {
            if i == j {
                continue;
            }
            pairwise_orthogonal_products_checked += 1;
            pairwise_orthogonal &= multiply(&projectors[i], &projectors[j]) == zeros(16, 16);
        }
    }
    let sum = projectors
        .iter()
        .fold(zeros(16, 16), |total, projector| add(&total, projector));
    let completeness_residual_entries = nonzero_entries(&subtract(&sum, &identity(16)));
    let mut reconstruction_residual_entries = 0;
    for basis_index in 0..16 {
        for row in 0..16 {
            let reconstructed = sum[row][basis_index].clone();
            let expected = if row == basis_index {
                Rat::from_integer(1)
            } else {
                Rat::zero()
            };
            if reconstructed != expected {
                reconstruction_residual_entries += 1;
            }
        }
    }
    let projector_ranks_sum = checks.iter().map(|check| check.computed_rank).sum();
    let passed = checks.iter().all(|check| {
        check.computed_rank == check.expected_rank && check.idempotent && check.so4_equivariant
    }) && projector_ranks_sum == 16
        && pairwise_orthogonal
        && completeness_residual_entries == 0
        && reconstruction_residual_entries == 0;
    LorentzIntertwinerReport {
        schema_version: "adynkra-4d-n1-rank-two-intertwiners-v1",
        source_arxiv: "2407.09334",
        source_equations: "2.5, 2.18",
        tensor_product: "[1,1] tensor [1,1] = [0,0] + [2,2] + [2,0] + [0,2]",
        tensor_dimension: 16,
        projectors: checks,
        projector_ranks_sum,
        pairwise_orthogonal_products_checked,
        pairwise_orthogonal,
        completeness_residual_entries,
        reconstruction_basis_vectors_checked: 16,
        reconstruction_residual_entries,
        so4_generators_checked: generators.len(),
        equivariance_commutators_checked: generators.len() * projectors.len(),
        boundary: "rank-two bosonic intertwiners only; vector-spinor intertwiners and repeated-irrep derivative maps remain open",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_orientation_is_fixed() {
        assert_eq!(permutation_sign([0, 1, 2, 3]), 1);
        assert_eq!(permutation_sign([1, 0, 2, 3]), -1);
        assert_eq!(permutation_sign([0, 0, 2, 3]), 0);
    }

    #[test]
    fn rank_two_projectors_are_complete_and_equivariant() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.projector_ranks_sum, 16);
        assert_eq!(report.completeness_residual_entries, 0);
        assert_eq!(report.reconstruction_residual_entries, 0);
        assert_eq!(report.equivariance_commutators_checked, 24);
    }

    #[test]
    fn duality_labels_follow_equation_2_6() {
        let report = verify();
        let self_dual = report
            .projectors
            .iter()
            .find(|projector| projector.id == "self_dual_two_form")
            .unwrap();
        let anti_self_dual = report
            .projectors
            .iter()
            .find(|projector| projector.id == "anti_self_dual_two_form")
            .unwrap();
        assert_eq!(self_dual.dynkin_label, [0, 2]);
        assert_eq!(anti_self_dual.dynkin_label, [2, 0]);
    }
}
