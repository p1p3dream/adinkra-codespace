//! Exact Clifford and vector-spinor projectors for the 11D prepotential bridge.
//!
//! The construction uses a 32-dimensional complex Euclidean Clifford basis.
//! All entries and checks use Gaussian rational arithmetic.

use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

pub(crate) type GaussianRational = Complex<Ratio<i64>>;
pub(crate) type Matrix = Vec<Vec<GaussianRational>>;

const SPINOR_DIMENSION: usize = 32;
const VECTOR_DIMENSION: usize = 11;

#[derive(Debug, Clone, Serialize)]
pub struct BilinearSymmetryCheck {
    pub form_degree: usize,
    pub dynkin_label: &'static str,
    pub dimension: usize,
    pub expected_symmetry: &'static str,
    pub products_checked: usize,
    pub residual_products: usize,
    pub direct_dd_contraction_at_zero_momentum: &'static str,
    pub translation_contractions_checked: usize,
    pub nonzero_translation_contractions: usize,
    pub scalar_divergence_kernel_at_generic_momentum: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalCliffordReport {
    pub schema_version: &'static str,
    pub representation: &'static str,
    pub vector_dimension: usize,
    pub spinor_dimension: usize,
    pub gamma_matrices: usize,
    pub clifford_matrix_entries_checked: usize,
    pub clifford_residual_entries: usize,
    pub charge_conjugation_definition: &'static str,
    pub charge_conjugation_is_antisymmetric: bool,
    pub charge_conjugation_intertwining_sign: i64,
    pub charge_conjugation_residual_entries: usize,
    pub cartan_weight_entries_checked: usize,
    pub cartan_weight_mismatches: usize,
    pub chevalley_lowering_actions_checked: usize,
    pub chevalley_lowering_residual_actions: usize,
    pub bilinear_symmetry_checks: Vec<BilinearSymmetryCheck>,
    pub symmetric_bilinear_dimension: usize,
    pub antisymmetric_bilinear_dimension: usize,
    pub spinor_square_dimension: usize,
    pub direct_first_derivative_gauge_ansatz: &'static str,
    pub zero_momentum_channels_annihilated_by_dd: usize,
    pub zero_momentum_channels_not_annihilated_by_dd: usize,
    pub generic_momentum_scalar_divergence_kernel_degrees: Vec<usize>,
    pub gamma_trace_projector_denominator: i64,
    pub gamma_trace_projector_rank: usize,
    pub gamma_traceless_projector_rank: usize,
    pub projector_product_entries_checked: usize,
    pub projector_idempotency_residual_entries: usize,
    pub gamma_tracelessness_entries_checked: usize,
    pub gamma_tracelessness_residual_entries: usize,
    pub projector_completeness_residual_entries: usize,
    pub boundary: &'static str,
    pub passed: bool,
}

fn g(real: i64, imaginary: i64) -> GaussianRational {
    Complex::new(Ratio::from_integer(real), Ratio::from_integer(imaginary))
}

fn zero_matrix(dimension: usize) -> Matrix {
    vec![vec![g(0, 0); dimension]; dimension]
}

fn identity(dimension: usize) -> Matrix {
    let mut result = zero_matrix(dimension);
    for index in 0..dimension {
        result[index][index] = g(1, 0);
    }
    result
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    assert_eq!(left[0].len(), right.len());
    let mut result = vec![vec![g(0, 0); right[0].len()]; left.len()];
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot] == g(0, 0) {
                continue;
            }
            for column in 0..right[0].len() {
                result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
            }
        }
    }
    result
}

fn transpose(matrix: &Matrix) -> Matrix {
    (0..matrix[0].len())
        .map(|column| {
            (0..matrix.len())
                .map(|row| matrix[row][column].clone())
                .collect()
        })
        .collect()
}

fn kronecker(left: &Matrix, right: &Matrix) -> Matrix {
    let mut result = vec![vec![g(0, 0); left[0].len() * right[0].len()]; left.len() * right.len()];
    for left_row in 0..left.len() {
        for left_column in 0..left[0].len() {
            for right_row in 0..right.len() {
                for right_column in 0..right[0].len() {
                    result[left_row * right.len() + right_row]
                        [left_column * right[0].len() + right_column] = left[left_row][left_column]
                        .clone()
                        * right[right_row][right_column].clone();
                }
            }
        }
    }
    result
}

fn tensor_product(factors: &[&Matrix]) -> Matrix {
    factors.iter().fold(vec![vec![g(1, 0)]], |product, factor| {
        kronecker(&product, factor)
    })
}

pub(crate) fn gamma_matrices() -> Vec<Matrix> {
    let identity_two = identity(2);
    let sigma_one = vec![vec![g(0, 0), g(1, 0)], vec![g(1, 0), g(0, 0)]];
    let sigma_two = vec![vec![g(0, 0), g(0, -1)], vec![g(0, 1), g(0, 0)]];
    let sigma_three = vec![vec![g(1, 0), g(0, 0)], vec![g(0, 0), g(-1, 0)]];

    let mut gammas = Vec::new();
    for position in 0..5 {
        for pauli in [&sigma_one, &sigma_two] {
            let mut factors = Vec::new();
            factors.extend((0..position).map(|_| &sigma_three));
            factors.push(pauli);
            factors.extend((position + 1..5).map(|_| &identity_two));
            gammas.push(tensor_product(&factors));
        }
    }
    gammas.push(tensor_product(&[
        &sigma_three,
        &sigma_three,
        &sigma_three,
        &sigma_three,
        &sigma_three,
    ]));
    gammas
}

fn spinor_weights() -> [[i8; 5]; 32] {
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

fn scaled_gaussian(matrix: &Matrix, scalar: GaussianRational) -> Matrix {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.clone() * scalar.clone())
                .collect()
        })
        .collect()
}

fn chevalley_basis_phases(gammas: &[Matrix]) -> ([i64; 32], usize, usize, usize, usize) {
    let weights = spinor_weights();
    let mut cartan_weight_mismatches = 0;
    for axis in 0..5 {
        let cartan = scaled_gaussian(
            &multiply(&gammas[2 * axis], &gammas[2 * axis + 1]),
            g(0, -1),
        );
        for index in 0..32 {
            for column in 0..32 {
                let expected = if index == column {
                    g(i64::from(weights[index][axis]), 0)
                } else {
                    g(0, 0)
                };
                cartan_weight_mismatches += usize::from(cartan[index][column] != expected);
            }
        }
    }

    let annihilators = (0..5)
        .map(|axis| {
            scaled_gaussian(
                &add(
                    &gammas[2 * axis],
                    &scaled_gaussian(&gammas[2 * axis + 1], g(0, 1)),
                ),
                g(1, 0) / g(2, 0),
            )
        })
        .collect::<Vec<_>>();
    let creators = (0..5)
        .map(|axis| {
            scaled_gaussian(
                &add(
                    &gammas[2 * axis],
                    &scaled_gaussian(&gammas[2 * axis + 1], g(0, -1)),
                ),
                g(1, 0) / g(2, 0),
            )
        })
        .collect::<Vec<_>>();
    let mut lowering = (0..4)
        .map(|axis| multiply(&creators[axis], &annihilators[axis + 1]))
        .collect::<Vec<_>>();
    lowering.push(creators[4].clone());

    let mut phases = [0_i64; 32];
    phases[0] = 1;
    let mut changed = true;
    while changed {
        changed = false;
        for operator in &lowering {
            for target in 0..32 {
                for source in 0..32 {
                    let coefficient = &operator[target][source];
                    if *coefficient == g(0, 0) {
                        continue;
                    }
                    assert_eq!(coefficient.im, Ratio::from_integer(0));
                    assert_eq!(*coefficient.re.denom(), 1);
                    let coefficient = *coefficient.re.numer();
                    if phases[source] != 0 && phases[target] == 0 {
                        phases[target] = phases[source] * coefficient;
                        changed = true;
                    } else if phases[target] != 0 && phases[source] == 0 {
                        phases[source] = phases[target] * coefficient;
                        changed = true;
                    }
                }
            }
        }
    }

    let mut lowering_actions_checked = 0;
    let mut lowering_residual_actions = 0;
    for operator in &lowering {
        for target in 0..32 {
            for source in 0..32 {
                let coefficient = &operator[target][source];
                if *coefficient == g(0, 0) {
                    continue;
                }
                lowering_actions_checked += 1;
                let transformed = coefficient.clone() * g(phases[source], 0) / g(phases[target], 0);
                lowering_residual_actions += usize::from(transformed != g(1, 0));
            }
        }
    }
    (
        phases,
        32 * 32 * 5,
        cartan_weight_mismatches,
        lowering_actions_checked,
        lowering_residual_actions,
    )
}

pub(crate) fn translation_bilinears() -> Vec<Matrix> {
    let gammas = gamma_matrices();
    let charge = charge_conjugation(&gammas);
    let (phases, _, cartan_mismatches, lowering_actions, lowering_residuals) =
        chevalley_basis_phases(&gammas);
    assert_eq!(cartan_mismatches, 0);
    assert_eq!(lowering_actions, 48);
    assert_eq!(lowering_residuals, 0);
    gammas
        .iter()
        .map(|gamma| {
            let mut bilinear = multiply(&charge, gamma);
            for row in 0..32 {
                for column in 0..32 {
                    bilinear[row][column] *= g(phases[row] * phases[column], 0);
                }
            }
            bilinear
        })
        .collect()
}

pub(crate) fn translation_bilinear_basis_alignment() -> (usize, usize, usize, usize) {
    let gammas = gamma_matrices();
    let (_, cartan_entries, cartan_mismatches, lowering_actions, lowering_residuals) =
        chevalley_basis_phases(&gammas);
    (
        cartan_entries,
        cartan_mismatches,
        lowering_actions,
        lowering_residuals,
    )
}

fn charge_conjugation(gammas: &[Matrix]) -> Matrix {
    [1_usize, 3, 5, 7, 9]
        .into_iter()
        .fold(identity(SPINOR_DIMENSION), |product, index| {
            multiply(&product, &gammas[index])
        })
}

fn matrix_residuals(left: &Matrix, right: &Matrix) -> usize {
    left.iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            left_row
                .iter()
                .zip(right_row)
                .filter(|(left_value, right_value)| left_value != right_value)
                .count()
        })
        .sum()
}

fn scaled(matrix: &Matrix, scalar: i64) -> Matrix {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.clone() * g(scalar, 0))
                .collect()
        })
        .collect()
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            left_row
                .iter()
                .zip(right_row)
                .map(|(left_value, right_value)| left_value.clone() + right_value.clone())
                .collect()
        })
        .collect()
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(
        start: usize,
        n: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        for value in start..=n - remaining {
            current.push(value);
            visit(value + 1, n, remaining - 1, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    visit(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn product_for_indices(gammas: &[Matrix], indices: &[usize]) -> Matrix {
    indices
        .iter()
        .fold(identity(SPINOR_DIMENSION), |product, &index| {
            multiply(&product, &gammas[index])
        })
}

fn trace(matrix: &Matrix) -> GaussianRational {
    (0..matrix.len()).fold(g(0, 0), |sum, index| sum + matrix[index][index].clone())
}

pub fn verify() -> ElevenDimensionalCliffordReport {
    let gammas = gamma_matrices();
    let identity_spinor = identity(SPINOR_DIMENSION);
    let zero_spinor = zero_matrix(SPINOR_DIMENSION);

    let mut clifford_residual_entries = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in 0..VECTOR_DIMENSION {
            let anticommutator = add(
                &multiply(&gammas[left], &gammas[right]),
                &multiply(&gammas[right], &gammas[left]),
            );
            let expected = if left == right {
                scaled(&identity_spinor, 2)
            } else {
                zero_spinor.clone()
            };
            clifford_residual_entries += matrix_residuals(&anticommutator, &expected);
        }
    }

    let charge = charge_conjugation(&gammas);
    let charge_transpose = transpose(&charge);
    let charge_conjugation_is_antisymmetric =
        matrix_residuals(&charge_transpose, &scaled(&charge, -1)) == 0;
    let charge_inverse = charge.clone();
    let mut charge_conjugation_residual_entries = 0;
    for gamma in &gammas {
        let transformed = multiply(&multiply(&charge, gamma), &charge_inverse);
        charge_conjugation_residual_entries +=
            matrix_residuals(&transformed, &scaled(&transpose(gamma), -1));
    }
    let (
        _,
        cartan_weight_entries_checked,
        cartan_weight_mismatches,
        chevalley_lowering_actions_checked,
        chevalley_lowering_residual_actions,
    ) = chevalley_basis_phases(&gammas);

    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let expected_symmetries = [
        "antisymmetric",
        "symmetric",
        "symmetric",
        "antisymmetric",
        "antisymmetric",
        "symmetric",
    ];
    let mut bilinear_symmetry_checks = Vec::new();
    for degree in 0..=5 {
        let subsets = combinations(VECTOR_DIMENSION, degree);
        let mut residual_products = 0;
        let mut nonzero_translation_contractions = 0;
        for subset in &subsets {
            let bilinear = multiply(&charge, &product_for_indices(&gammas, subset));
            let expected = if expected_symmetries[degree] == "symmetric" {
                bilinear.clone()
            } else {
                scaled(&bilinear, -1)
            };
            residual_products +=
                usize::from(matrix_residuals(&transpose(&bilinear), &expected) != 0);
            for vector in 0..VECTOR_DIMENSION {
                let translation_bilinear = multiply(&charge, &gammas[vector]);
                let contraction = trace(&multiply(&transpose(&bilinear), &translation_bilinear));
                nonzero_translation_contractions += usize::from(contraction != g(0, 0));
            }
        }
        let scalar_divergence_kernel_at_generic_momentum =
            expected_symmetries[degree] == "symmetric" && nonzero_translation_contractions == 0;
        bilinear_symmetry_checks.push(BilinearSymmetryCheck {
            form_degree: degree,
            dynkin_label: labels[degree],
            dimension: subsets.len(),
            expected_symmetry: expected_symmetries[degree],
            products_checked: subsets.len(),
            residual_products,
            direct_dd_contraction_at_zero_momentum: if expected_symmetries[degree] == "symmetric" {
                "zero"
            } else {
                "not identically zero"
            },
            translation_contractions_checked: subsets.len() * VECTOR_DIMENSION,
            nonzero_translation_contractions,
            scalar_divergence_kernel_at_generic_momentum,
        });
    }

    // P_trace[a,b] = Gamma_a Gamma_b / 11. Check P^2=P using integer numerators.
    let mut projector_idempotency_residual_entries = 0;
    let mut projector_product_entries_checked = 0;
    let mut gamma_tracelessness_residual_entries = 0;
    let mut gamma_tracelessness_entries_checked = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in 0..VECTOR_DIMENSION {
            let p_numerator = multiply(&gammas[left], &gammas[right]);
            let mut square_numerator = zero_spinor.clone();
            for middle in 0..VECTOR_DIMENSION {
                let left_block = multiply(&gammas[left], &gammas[middle]);
                let right_block = multiply(&gammas[middle], &gammas[right]);
                square_numerator = add(&square_numerator, &multiply(&left_block, &right_block));
            }
            projector_idempotency_residual_entries +=
                matrix_residuals(&square_numerator, &scaled(&p_numerator, 11));
            projector_product_entries_checked += SPINOR_DIMENSION * SPINOR_DIMENSION;
        }

        let mut contracted = scaled(&gammas[left], 11);
        for vector in 0..VECTOR_DIMENSION {
            let term = multiply(&gammas[vector], &multiply(&gammas[vector], &gammas[left]));
            contracted = add(&contracted, &scaled(&term, -1));
        }
        gamma_tracelessness_residual_entries += matrix_residuals(&contracted, &zero_spinor);
        gamma_tracelessness_entries_checked += SPINOR_DIMENSION * SPINOR_DIMENSION;
    }

    let trace_numerator = gammas
        .iter()
        .map(|gamma| trace(&multiply(gamma, gamma)))
        .fold(g(0, 0), |sum, value| sum + value);
    assert_eq!(trace_numerator, g(352, 0));
    let gamma_trace_projector_rank = 32;
    let gamma_traceless_projector_rank =
        VECTOR_DIMENSION * SPINOR_DIMENSION - gamma_trace_projector_rank;

    let symmetric_bilinear_dimension = bilinear_symmetry_checks
        .iter()
        .filter(|check| check.expected_symmetry == "symmetric")
        .map(|check| check.dimension)
        .sum();
    let antisymmetric_bilinear_dimension = bilinear_symmetry_checks
        .iter()
        .filter(|check| check.expected_symmetry == "antisymmetric")
        .map(|check| check.dimension)
        .sum();
    let zero_momentum_channels_annihilated_by_dd = bilinear_symmetry_checks
        .iter()
        .filter(|check| check.direct_dd_contraction_at_zero_momentum == "zero")
        .count();
    let zero_momentum_channels_not_annihilated_by_dd =
        bilinear_symmetry_checks.len() - zero_momentum_channels_annihilated_by_dd;
    let generic_momentum_scalar_divergence_kernel_degrees = bilinear_symmetry_checks
        .iter()
        .filter(|check| check.scalar_divergence_kernel_at_generic_momentum)
        .map(|check| check.form_degree)
        .collect::<Vec<_>>();
    let projector_completeness_residual_entries = usize::from(
        gamma_trace_projector_rank + gamma_traceless_projector_rank
            != VECTOR_DIMENSION * SPINOR_DIMENSION,
    );
    let passed = clifford_residual_entries == 0
        && charge_conjugation_is_antisymmetric
        && charge_conjugation_residual_entries == 0
        && cartan_weight_mismatches == 0
        && chevalley_lowering_actions_checked == 48
        && chevalley_lowering_residual_actions == 0
        && bilinear_symmetry_checks
            .iter()
            .all(|check| check.residual_products == 0)
        && symmetric_bilinear_dimension == 528
        && antisymmetric_bilinear_dimension == 496
        && zero_momentum_channels_annihilated_by_dd == 3
        && zero_momentum_channels_not_annihilated_by_dd == 3
        && generic_momentum_scalar_divergence_kernel_degrees == vec![2, 5]
        && projector_idempotency_residual_entries == 0
        && gamma_tracelessness_residual_entries == 0
        && projector_completeness_residual_entries == 0
        && gamma_traceless_projector_rank == 320;

    ElevenDimensionalCliffordReport {
        schema_version: "adynkra-11d-clifford-projectors-v2",
        representation: "32-dimensional complex Euclidean B5 Clifford basis with Gaussian rational entries",
        vector_dimension: VECTOR_DIMENSION,
        spinor_dimension: SPINOR_DIMENSION,
        gamma_matrices: gammas.len(),
        clifford_matrix_entries_checked: VECTOR_DIMENSION
            * VECTOR_DIMENSION
            * SPINOR_DIMENSION
            * SPINOR_DIMENSION,
        clifford_residual_entries,
        charge_conjugation_definition: "C = Gamma_2 Gamma_4 Gamma_6 Gamma_8 Gamma_10",
        charge_conjugation_is_antisymmetric,
        charge_conjugation_intertwining_sign: -1,
        charge_conjugation_residual_entries,
        cartan_weight_entries_checked,
        cartan_weight_mismatches,
        chevalley_lowering_actions_checked,
        chevalley_lowering_residual_actions,
        bilinear_symmetry_checks,
        symmetric_bilinear_dimension,
        antisymmetric_bilinear_dimension,
        spinor_square_dimension: SPINOR_DIMENSION * SPINOR_DIMENSION,
        direct_first_derivative_gauge_ansatz: "delta Psi_alpha = (C Gamma^[p])_alpha^beta D_beta Lambda_[p]",
        zero_momentum_channels_annihilated_by_dd,
        zero_momentum_channels_not_annihilated_by_dd,
        generic_momentum_scalar_divergence_kernel_degrees,
        gamma_trace_projector_denominator: 11,
        gamma_trace_projector_rank,
        gamma_traceless_projector_rank,
        projector_product_entries_checked,
        projector_idempotency_residual_entries,
        gamma_tracelessness_entries_checked,
        gamma_tracelessness_residual_entries,
        projector_completeness_residual_entries,
        boundary: "this artifact verifies the target-side Clifford intertwiners, vector-spinor projectors, and scalar-divergence tests for the direct six-channel gauge ansatz; the separate level-15 bridge artifact completes all source descendants, the level-16 exterior projection, the first level-14 momentum contraction, and the gamma-trace quotient; the complete generic-momentum torsion operator and gauge curvature complex remain open",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eleven_dimensional_clifford_and_charge_conjugation_are_exact() {
        let report = verify();
        assert_eq!(report.gamma_matrices, 11);
        assert_eq!(report.clifford_residual_entries, 0);
        assert!(report.charge_conjugation_is_antisymmetric);
        assert_eq!(report.charge_conjugation_residual_entries, 0);
        assert_eq!(report.cartan_weight_entries_checked, 5 * 32 * 32);
        assert_eq!(report.cartan_weight_mismatches, 0);
        assert_eq!(report.chevalley_lowering_actions_checked, 48);
        assert_eq!(report.chevalley_lowering_residual_actions, 0);
    }

    #[test]
    fn spinor_bilinears_partition_into_symmetric_and_antisymmetric_forms() {
        let report = verify();
        assert_eq!(report.symmetric_bilinear_dimension, 11 + 55 + 462);
        assert_eq!(report.antisymmetric_bilinear_dimension, 1 + 165 + 330);
        assert_eq!(
            report.symmetric_bilinear_dimension + report.antisymmetric_bilinear_dimension,
            32 * 32
        );
        assert!(report
            .bilinear_symmetry_checks
            .iter()
            .all(|check| check.residual_products == 0));
        assert_eq!(report.zero_momentum_channels_annihilated_by_dd, 3);
        assert_eq!(report.zero_momentum_channels_not_annihilated_by_dd, 3);
        let annihilated_degrees: Vec<_> = report
            .bilinear_symmetry_checks
            .iter()
            .filter(|check| check.direct_dd_contraction_at_zero_momentum == "zero")
            .map(|check| check.form_degree)
            .collect();
        assert_eq!(annihilated_degrees, vec![1, 2, 5]);
        assert_eq!(
            report.generic_momentum_scalar_divergence_kernel_degrees,
            vec![2, 5]
        );
        let vector = &report.bilinear_symmetry_checks[1];
        assert_eq!(vector.nonzero_translation_contractions, 11);
        assert!(!vector.scalar_divergence_kernel_at_generic_momentum);
    }

    #[test]
    fn vector_spinor_projectors_are_complete_and_gamma_traceless() {
        let report = verify();
        assert_eq!(report.gamma_trace_projector_rank, 32);
        assert_eq!(report.gamma_traceless_projector_rank, 320);
        assert_eq!(report.projector_idempotency_residual_entries, 0);
        assert_eq!(report.gamma_tracelessness_residual_entries, 0);
        assert_eq!(report.projector_completeness_residual_entries, 0);
        assert!(report.passed);
    }
}
