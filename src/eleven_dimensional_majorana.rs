//! Exact Majorana real form for the Lorentzian 11D Clifford representation.
//!
//! The existing B5 implementation is a complex Euclidean weight basis.  This
//! module performs the Lorentz continuation, constructs the antilinear
//! Majorana involution, and supplies an explicit fixed real basis.  In that
//! basis all eleven gamma matrices are real signed permutations.

use num_complex::Complex;
use num_rational::Ratio;
use num_traits::Signed;
use serde::Serialize;

type Gaussian = Complex<Ratio<i64>>;
type Matrix = Vec<Vec<Gaussian>>;

pub type ExactGaussian = Complex<Ratio<i64>>;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const MAJORANA_PRODUCT_INDICES: [usize; 5] = [2, 4, 6, 8, 10];
const CHARGE_CONJUGATION_PRODUCT_INDICES: [usize; 5] = [1, 3, 5, 7, 9];

fn q(real: i64, imaginary: i64) -> Gaussian {
    Complex::new(Ratio::from_integer(real), Ratio::from_integer(imaginary))
}

fn zero(rows: usize, columns: usize) -> Matrix {
    vec![vec![q(0, 0); columns]; rows]
}

fn identity(dimension: usize) -> Matrix {
    let mut result = zero(dimension, dimension);
    for index in 0..dimension {
        result[index][index] = q(1, 0);
    }
    result
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    assert_eq!(left[0].len(), right.len());
    let mut result = zero(left.len(), right[0].len());
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot] == q(0, 0) {
                continue;
            }
            for column in 0..right[0].len() {
                if right[pivot][column] != q(0, 0) {
                    result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
                }
            }
        }
    }
    result
}

fn conjugate(matrix: &Matrix) -> Matrix {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| Complex::new(value.re.clone(), -value.im.clone()))
                .collect()
        })
        .collect()
}

fn lorentz_gammas() -> Vec<Matrix> {
    crate::eleven_dimensional_clifford::gamma_matrices()
        .into_iter()
        .enumerate()
        .map(|(axis, matrix)| {
            if axis == 0 {
                matrix
                    .into_iter()
                    .map(|row| row.into_iter().map(|value| q(0, 1) * value).collect())
                    .collect()
            } else {
                matrix
            }
        })
        .collect()
}

fn majorana_intertwiner(gammas: &[Matrix]) -> Matrix {
    MAJORANA_PRODUCT_INDICES
        .into_iter()
        .fold(identity(SPINOR_DIMENSION), |product, axis| {
            multiply(&product, &gammas[axis])
        })
}

fn signed_permutation(matrix: &Matrix) -> Option<Vec<(usize, i64)>> {
    let mut result = Vec::with_capacity(matrix.len());
    let mut used_rows = vec![false; matrix.len()];
    for column in 0..matrix[0].len() {
        let entries = (0..matrix.len())
            .filter(|&row| matrix[row][column] != q(0, 0))
            .collect::<Vec<_>>();
        if entries.len() != 1 {
            return None;
        }
        let row = entries[0];
        let value = &matrix[row][column];
        if value.im != Ratio::from_integer(0)
            || *value.re.denom() != 1
            || value.re.abs() != Ratio::from_integer(1)
            || used_rows[row]
        {
            return None;
        }
        used_rows[row] = true;
        result.push((row, *value.re.numer()));
    }
    Some(result)
}

/// Columns of `S` obey `B conjugate(S) = S`.  They therefore identify real
/// coordinate vectors with Majorana spinors in the original complex basis.
fn fixed_basis(intertwiner: &Matrix) -> (Matrix, Matrix, usize) {
    let permutation = signed_permutation(intertwiner).expect("Majorana B is monomial");
    let mut basis = zero(SPINOR_DIMENSION, SPINOR_DIMENSION);
    let mut inverse = zero(SPINOR_DIMENSION, SPINOR_DIMENSION);
    let mut visited = [false; SPINOR_DIMENSION];
    let mut column = 0;
    let mut pairs = 0;
    for first in 0..SPINOR_DIMENSION {
        if visited[first] {
            continue;
        }
        let (second, sign) = permutation[first];
        assert_ne!(first, second);
        assert_eq!(permutation[second], (first, sign));
        assert!(first < second);
        visited[first] = true;
        visited[second] = true;

        basis[first][column] = q(1, 0);
        basis[second][column] = q(sign, 0);
        inverse[column][first] = q(1, 0) / q(2, 0);
        inverse[column][second] = q(sign, 0) / q(2, 0);
        column += 1;

        basis[first][column] = q(0, 1);
        basis[second][column] = q(0, -sign);
        inverse[column][first] = q(0, -1) / q(2, 0);
        inverse[column][second] = q(0, sign) / q(2, 0);
        column += 1;
        pairs += 1;
    }
    assert_eq!(column, SPINOR_DIMENSION);
    (basis, inverse, pairs)
}

/// Exact change of spinor coordinates from the original complex Euclidean
/// Clifford basis to the real Lorentzian Majorana basis.
///
/// If `psi_complex = S psi_majorana`, this returns `(S, S^{-1})`.  The columns
/// of `S` are fixed by the antilinear Majorana involution, so this is an
/// intertwiner derived from the Clifford representation rather than a phase
/// convention guessed state by state.
pub fn majorana_basis_change() -> (Vec<Vec<ExactGaussian>>, Vec<Vec<ExactGaussian>>) {
    let gammas = lorentz_gammas();
    let intertwiner = majorana_intertwiner(&gammas);
    let (basis, inverse, _) = fixed_basis(&intertwiner);
    (basis, inverse)
}

fn transform(inverse: &Matrix, matrix: &Matrix, basis: &Matrix) -> Matrix {
    multiply(&multiply(inverse, matrix), basis)
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

/// Real Lorentzian gamma matrices in the explicit Majorana basis.
pub fn real_gamma_matrices() -> Vec<Vec<Vec<i8>>> {
    let gammas = lorentz_gammas();
    let intertwiner = majorana_intertwiner(&gammas);
    let (basis, inverse, _) = fixed_basis(&intertwiner);
    gammas
        .iter()
        .map(|gamma| {
            transform(&inverse, gamma, &basis)
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| {
                            assert_eq!(value.im, Ratio::from_integer(0));
                            assert_eq!(*value.re.denom(), 1);
                            i8::try_from(*value.re.numer()).expect("real gamma entry fits i8")
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Charge-conjugation bilinear in the same real basis as
/// [`real_gamma_matrices`].  Its normalization is primitive integral, so the
/// matrix is an antisymmetric signed permutation and `C gamma^a` is
/// symmetric for every Lorentz axis.
pub fn real_charge_conjugation() -> Vec<Vec<i8>> {
    let euclidean = crate::eleven_dimensional_clifford::gamma_matrices();
    let lorentz = lorentz_gammas();
    let intertwiner = majorana_intertwiner(&lorentz);
    let (basis, _, _) = fixed_basis(&intertwiner);
    let charge_conjugation = CHARGE_CONJUGATION_PRODUCT_INDICES
        .into_iter()
        .fold(identity(SPINOR_DIMENSION), |product, axis| {
            multiply(&product, &euclidean[axis])
        });
    let transformed = multiply(&multiply(&transpose(&basis), &charge_conjugation), &basis);
    transformed
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| {
                    let primitive = value / q(2, 0);
                    assert_eq!(primitive.im, Ratio::from_integer(0));
                    assert_eq!(*primitive.re.denom(), 1);
                    i8::try_from(*primitive.re.numer())
                        .expect("real charge-conjugation entry fits i8")
                })
                .collect()
        })
        .collect()
}

/// Exact coordinate map from the Chevalley-aligned complex B5 spinor-weight
/// basis to the Lorentzian Majorana real coordinates used by this module.
/// If `x` is aligned, the original Euclidean coordinates are `D_phase x` and
/// the returned matrix acts as `S^{-1} D_phase x`.
pub fn aligned_weight_to_majorana_coordinates() -> Vec<Vec<ExactGaussian>> {
    let lorentz = lorentz_gammas();
    let intertwiner = majorana_intertwiner(&lorentz);
    let (_, inverse, _) = fixed_basis(&intertwiner);
    let phases = crate::eleven_dimensional_clifford::spinor_chevalley_basis_phases();
    let mut coordinates = inverse;
    for row in &mut coordinates {
        for column in 0..SPINOR_DIMENSION {
            row[column] *= q(phases[column], 0);
        }
    }
    coordinates
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ElevenDimensionalMajoranaReport {
    pub schema_version: &'static str,
    pub starting_basis: &'static str,
    pub lorentz_signature: &'static str,
    pub majorana_involution: &'static str,
    pub product_indices: [usize; 5],
    pub intertwiner_nonzero_entries: usize,
    pub intertwiner_is_real_signed_permutation: bool,
    pub involution_residual_entries: usize,
    pub gamma_intertwining_entries_checked: usize,
    pub gamma_intertwining_residual_entries: usize,
    pub fixed_real_basis_dimension: usize,
    pub fixed_basis_pairs: usize,
    pub fixed_basis_residual_entries: usize,
    pub basis_inverse_residual_entries: usize,
    pub transformed_gamma_nonzero_entries: usize,
    pub transformed_gamma_nonreal_entries: usize,
    pub transformed_gamma_nonintegral_entries: usize,
    pub transformed_gamma_not_signed_permutation: usize,
    pub real_clifford_entries_checked: usize,
    pub real_clifford_residual_entries: usize,
    pub majorana_real_form_constructed: bool,
    pub linearized_susy_maps_constructed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn verify() -> ElevenDimensionalMajoranaReport {
    let gammas = lorentz_gammas();
    let intertwiner = majorana_intertwiner(&gammas);
    let permutation = signed_permutation(&intertwiner);
    let involution = multiply(&intertwiner, &conjugate(&intertwiner));
    let involution_residual_entries = involution
        .iter()
        .flatten()
        .zip(identity(SPINOR_DIMENSION).into_iter().flatten())
        .filter(|(left, right)| **left != *right)
        .count();

    let mut gamma_intertwining_residual_entries = 0;
    for gamma in &gammas {
        let transformed = multiply(&multiply(&intertwiner, &conjugate(gamma)), &intertwiner);
        gamma_intertwining_residual_entries += transformed
            .iter()
            .flatten()
            .zip(gamma.iter().flatten())
            .filter(|(left, right)| *left != *right)
            .count();
    }

    let (basis, inverse, fixed_basis_pairs) = fixed_basis(&intertwiner);
    let fixed_basis_residual_entries = multiply(&intertwiner, &conjugate(&basis))
        .iter()
        .flatten()
        .zip(basis.iter().flatten())
        .filter(|(left, right)| *left != *right)
        .count();
    let basis_inverse_residual_entries = multiply(&inverse, &basis)
        .iter()
        .flatten()
        .zip(identity(SPINOR_DIMENSION).into_iter().flatten())
        .filter(|(left, right)| **left != *right)
        .count();

    let transformed = gammas
        .iter()
        .map(|gamma| transform(&inverse, gamma, &basis))
        .collect::<Vec<_>>();
    let transformed_gamma_nonzero_entries = transformed
        .iter()
        .flatten()
        .flatten()
        .filter(|value| **value != q(0, 0))
        .count();
    let transformed_gamma_nonreal_entries = transformed
        .iter()
        .flatten()
        .flatten()
        .filter(|value| value.im != Ratio::from_integer(0))
        .count();
    let transformed_gamma_nonintegral_entries = transformed
        .iter()
        .flatten()
        .flatten()
        .filter(|value| *value.re.denom() != 1)
        .count();
    let transformed_gamma_not_signed_permutation = transformed
        .iter()
        .filter(|gamma| signed_permutation(gamma).is_none())
        .count();

    let mut real_clifford_residual_entries = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in 0..VECTOR_DIMENSION {
            let anticommutator = {
                let lr = multiply(&transformed[left], &transformed[right]);
                let rl = multiply(&transformed[right], &transformed[left]);
                lr.into_iter()
                    .zip(rl)
                    .map(|(left_row, right_row)| {
                        left_row
                            .into_iter()
                            .zip(right_row)
                            .map(|(a, b)| a + b)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            let metric = if left == right {
                if left == 0 { -2 } else { 2 }
            } else {
                0
            };
            for row in 0..SPINOR_DIMENSION {
                for column in 0..SPINOR_DIMENSION {
                    let expected = if row == column { q(metric, 0) } else { q(0, 0) };
                    real_clifford_residual_entries +=
                        usize::from(anticommutator[row][column] != expected);
                }
            }
        }
    }

    let majorana_real_form_constructed = permutation.is_some()
        && involution_residual_entries == 0
        && gamma_intertwining_residual_entries == 0
        && fixed_basis_pairs == 16
        && fixed_basis_residual_entries == 0
        && basis_inverse_residual_entries == 0
        && transformed_gamma_nonreal_entries == 0
        && transformed_gamma_nonintegral_entries == 0
        && transformed_gamma_not_signed_permutation == 0
        && real_clifford_residual_entries == 0;

    ElevenDimensionalMajoranaReport {
        schema_version: "adynkra-11d-majorana-real-form-v1",
        starting_basis: "32-dimensional complex Euclidean B5 weight basis over Q(i)",
        lorentz_signature: "mostly plus (-,+,+,+,+,+,+,+,+,+,+)",
        majorana_involution: "psi = B conjugate(psi), B = gamma^2 gamma^4 gamma^6 gamma^8 gamma^10",
        product_indices: MAJORANA_PRODUCT_INDICES,
        intertwiner_nonzero_entries: intertwiner
            .iter()
            .flatten()
            .filter(|value| **value != q(0, 0))
            .count(),
        intertwiner_is_real_signed_permutation: permutation.is_some(),
        involution_residual_entries,
        gamma_intertwining_entries_checked: VECTOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION,
        gamma_intertwining_residual_entries,
        fixed_real_basis_dimension: SPINOR_DIMENSION,
        fixed_basis_pairs,
        fixed_basis_residual_entries,
        basis_inverse_residual_entries,
        transformed_gamma_nonzero_entries,
        transformed_gamma_nonreal_entries,
        transformed_gamma_nonintegral_entries,
        transformed_gamma_not_signed_permutation,
        real_clifford_entries_checked: VECTOR_DIMENSION
            * VECTOR_DIMENSION
            * SPINOR_DIMENSION
            * SPINOR_DIMENSION,
        real_clifford_residual_entries,
        majorana_real_form_constructed,
        linearized_susy_maps_constructed: false,
        passed: majorana_real_form_constructed,
        boundary: "this constructs the exact Lorentzian Majorana spinor real form. It does not by itself supply or certify the linearized 11D supergravity supersymmetry maps between h, A3, and psi",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majorana_involution_and_fixed_basis_are_exact() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.intertwiner_nonzero_entries, 32);
        assert!(report.intertwiner_is_real_signed_permutation);
        assert_eq!(report.involution_residual_entries, 0);
        assert_eq!(report.gamma_intertwining_residual_entries, 0);
        assert_eq!(report.fixed_real_basis_dimension, 32);
        assert_eq!(report.fixed_basis_pairs, 16);
        assert_eq!(report.fixed_basis_residual_entries, 0);
        assert_eq!(report.basis_inverse_residual_entries, 0);
    }

    #[test]
    fn all_lorentz_gammas_become_real_signed_permutations() {
        let report = verify();
        assert_eq!(report.transformed_gamma_nonzero_entries, 11 * 32);
        assert_eq!(report.transformed_gamma_nonreal_entries, 0);
        assert_eq!(report.transformed_gamma_nonintegral_entries, 0);
        assert_eq!(report.transformed_gamma_not_signed_permutation, 0);
        assert_eq!(report.real_clifford_residual_entries, 0);
        let gammas = real_gamma_matrices();
        assert_eq!(gammas.len(), 11);
        assert!(gammas.iter().all(|gamma| gamma.len() == 32));
    }

    #[test]
    fn report_does_not_claim_unbuilt_susy_maps() {
        let report = verify();
        assert!(report.majorana_real_form_constructed);
        assert!(!report.linearized_susy_maps_constructed);
    }

    #[test]
    fn charge_conjugation_has_the_required_real_symmetries() {
        let charge_conjugation = real_charge_conjugation();
        let gammas = real_gamma_matrices();
        for row in 0..SPINOR_DIMENSION {
            for column in 0..SPINOR_DIMENSION {
                assert_eq!(
                    charge_conjugation[row][column],
                    -charge_conjugation[column][row]
                );
            }
        }
        for gamma in gammas {
            let mut c_gamma = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
            for row in 0..SPINOR_DIMENSION {
                for pivot in 0..SPINOR_DIMENSION {
                    for column in 0..SPINOR_DIMENSION {
                        c_gamma[row][column] += i16::from(charge_conjugation[row][pivot])
                            * i16::from(gamma[pivot][column]);
                    }
                }
            }
            for row in 0..SPINOR_DIMENSION {
                for column in 0..SPINOR_DIMENSION {
                    assert_eq!(c_gamma[row][column], c_gamma[column][row]);
                }
            }
        }
    }

    #[test]
    fn aligned_weight_to_majorana_map_has_the_documented_direction() {
        let lorentz = lorentz_gammas();
        let intertwiner = majorana_intertwiner(&lorentz);
        let (basis, _, _) = fixed_basis(&intertwiner);
        let phases = crate::eleven_dimensional_clifford::spinor_chevalley_basis_phases();
        let map = aligned_weight_to_majorana_coordinates();
        let reconstructed = multiply(&basis, &map);
        for row in 0..SPINOR_DIMENSION {
            for column in 0..SPINOR_DIMENSION {
                let expected = if row == column {
                    q(phases[row], 0)
                } else {
                    q(0, 0)
                };
                assert_eq!(reconstructed[row][column], expected);
            }
        }
    }
}
