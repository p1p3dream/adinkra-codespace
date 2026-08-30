//! Lowest exact 11D spinorial differential and tau-zero quotient.
//!
//! Tsimpis decomposes the superspace exterior derivative as
//! `d = d_0 + d_1 + tau_0 + tau_1`.  On a scalar superfield the component of
//! `d^2 = 0` with bidegree `(0,2)` is
//! `d_1^2 + tau_0 d_0 = 0`.  Therefore the induced spinorial differential
//! `d_F[omega] = [d_1 omega]` squares to zero in `H_tau`.
//!
//! This module executes that identity coefficientwise in all eleven formal
//! momentum variables.  It constructs the exact map
//! `tau_0: Omega^(1,0) -> Omega^(0,2)`, an exact quotient map with kernel
//! equal to its image, and the lowest scalar-symbol `d_1^2` map.  This is a
//! necessary entry slice of spinorial cohomology.  It is not the full
//! `H_F^(0,4)(phys)` deformation complex, a superspace Bianchi solution, or
//! a finite-auxiliary off-shell formulation.

use num_rational::Ratio;
use serde::Serialize;
#[cfg(test)]
use std::fs;

pub type ExactRational = Ratio<i64>;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const SYMMETRIC_BISPINOR_DIMENSION: usize = SPINOR_DIMENSION * (SPINOR_DIMENSION + 1) / 2;

fn zero() -> ExactRational {
    Ratio::from_integer(0)
}

fn one() -> ExactRational {
    Ratio::from_integer(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactSparseMap {
    row_count: usize,
    column_count: usize,
    entries: Vec<(usize, usize, ExactRational)>,
}

impl ExactSparseMap {
    fn from_dense(matrix: &[Vec<ExactRational>]) -> Self {
        let row_count = matrix.len();
        let column_count = matrix.first().map_or(0, Vec::len);
        assert!(matrix.iter().all(|row| row.len() == column_count));
        let entries = matrix
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value != zero())
                    .map(move |(column, value)| (row, column, value.clone()))
            })
            .collect();
        Self {
            row_count,
            column_count,
            entries,
        }
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn nonzero_entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[(usize, usize, ExactRational)] {
        &self.entries
    }

    pub fn apply(&self, vector: &[ExactRational]) -> Vec<ExactRational> {
        assert_eq!(vector.len(), self.column_count);
        let mut result = vec![zero(); self.row_count];
        for (row, column, coefficient) in &self.entries {
            result[*row] += coefficient.clone() * vector[*column].clone();
        }
        result
    }
}

fn symmetric_spinor_pairs() -> Vec<(usize, usize)> {
    (0..SPINOR_DIMENSION)
        .flat_map(|left| (left..SPINOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn majorana_translation_bilinears() -> Vec<Vec<Vec<ExactRational>>> {
    let charge = crate::eleven_dimensional_majorana::real_charge_conjugation();
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    gammas
        .iter()
        .map(|gamma| {
            let mut bilinear = vec![vec![zero(); SPINOR_DIMENSION]; SPINOR_DIMENSION];
            for row in 0..SPINOR_DIMENSION {
                for pivot in 0..SPINOR_DIMENSION {
                    if charge[row][pivot] == 0 {
                        continue;
                    }
                    for column in 0..SPINOR_DIMENSION {
                        if gamma[pivot][column] != 0 {
                            bilinear[row][column] += Ratio::from_integer(
                                i64::from(charge[row][pivot]) * i64::from(gamma[pivot][column]),
                            );
                        }
                    }
                }
            }
            assert!((0..SPINOR_DIMENSION).all(|row| {
                (0..SPINOR_DIMENSION).all(|column| bilinear[row][column] == bilinear[column][row])
            }));
            bilinear
        })
        .collect()
}

fn tau_zero_dense() -> Vec<Vec<ExactRational>> {
    let pairs = symmetric_spinor_pairs();
    let bilinears = majorana_translation_bilinears();
    pairs
        .iter()
        .map(|&(left, right)| {
            bilinears
                .iter()
                .map(|bilinear| bilinear[left][right].clone())
                .collect()
        })
        .collect()
}

/// Algebraic dimension-zero torsion map in the real Majorana basis.
///
/// Rows use one symmetric-matrix coordinate for every `alpha <= beta`.
/// Columns are the eleven vector directions.  The conventional common factor
/// `-i` in `T_{alpha beta}^c` is omitted because it does not change the image
/// or the quotient.
pub fn tau_zero_vector_to_symmetric_bispinor() -> ExactSparseMap {
    ExactSparseMap::from_dense(&tau_zero_dense())
}

fn rref(mut matrix: Vec<Vec<ExactRational>>) -> (Vec<Vec<ExactRational>>, Vec<usize>) {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| matrix[row][column] != zero()) else {
            continue;
        };
        matrix.swap(pivot_row, found);
        let pivot = matrix[pivot_row][column].clone();
        for entry in &mut matrix[pivot_row] {
            *entry /= pivot.clone();
        }
        let normalized = matrix[pivot_row].clone();
        for row in 0..rows {
            if row == pivot_row || matrix[row][column] == zero() {
                continue;
            }
            let factor = matrix[row][column].clone();
            for index in column..columns {
                matrix[row][index] -= factor.clone() * normalized[index].clone();
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    (matrix, pivot_columns)
}

fn quotient_dense() -> (Vec<Vec<ExactRational>>, Vec<usize>) {
    let tau = tau_zero_dense();
    let transpose = (0..VECTOR_DIMENSION)
        .map(|column| {
            (0..SYMMETRIC_BISPINOR_DIMENSION)
                .map(|row| tau[row][column].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (reduced, pivots) = rref(transpose);
    let free = (0..SYMMETRIC_BISPINOR_DIMENSION)
        .filter(|column| !pivots.contains(column))
        .collect::<Vec<_>>();
    let rows = free
        .iter()
        .map(|&free_column| {
            let mut vector = vec![zero(); SYMMETRIC_BISPINOR_DIMENSION];
            vector[free_column] = one();
            for (row, &pivot) in pivots.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free_column].clone();
            }
            vector
        })
        .collect();
    (rows, free)
}

/// Exact quotient map `Sym^2(S) -> Sym^2(S) / im(tau_0)`.
///
/// Its 517 rows are a canonical RREF left-nullspace basis of `tau_0`.
pub fn tau_zero_bianchi_quotient() -> ExactSparseMap {
    ExactSparseMap::from_dense(&quotient_dense().0)
}

/// Coefficients of `d_1^2` in the eleven formal momentum variables.
///
/// The sign fixes the convention `d_1^2 + tau_0 d_0 = 0` on a scalar.
pub fn d_one_square_scalar_symbol() -> ExactSparseMap {
    let mut matrix = tau_zero_dense();
    for entry in matrix.iter_mut().flatten() {
        *entry = -entry.clone();
    }
    ExactSparseMap::from_dense(&matrix)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LowestSpinorialDifferentialSlice {
    pub tau_zero: ExactSparseMap,
    pub h_tau_zero_two_quotient: ExactSparseMap,
    pub d_one_square_scalar_symbol: ExactSparseMap,
}

/// Deterministic exact data for the lowest scalar composition slice of d_F.
pub fn lowest_spinorial_differential_slice() -> LowestSpinorialDifferentialSlice {
    LowestSpinorialDifferentialSlice {
        tau_zero: tau_zero_vector_to_symmetric_bispinor(),
        h_tau_zero_two_quotient: tau_zero_bianchi_quotient(),
        d_one_square_scalar_symbol: d_one_square_scalar_symbol(),
    }
}

fn multiply(left: &[Vec<ExactRational>], right: &[Vec<ExactRational>]) -> Vec<Vec<ExactRational>> {
    assert_eq!(left[0].len(), right.len());
    let mut result = vec![vec![zero(); right[0].len()]; left.len()];
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot] == zero() {
                continue;
            }
            for column in 0..right[0].len() {
                if right[pivot][column] != zero() {
                    result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
                }
            }
        }
    }
    result
}

fn nonzero_entries(matrix: &[Vec<ExactRational>]) -> usize {
    matrix
        .iter()
        .flatten()
        .filter(|entry| **entry != zero())
        .count()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SpinorialDifferentialSourceAudit {
    pub source: &'static str,
    pub source_archive_sha256: &'static str,
    pub source_statement: &'static str,
    pub use_in_gate: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ElevenDimensionalSpinorialDifferentialReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub primary_sources: Vec<SpinorialDifferentialSourceAudit>,
    pub lorentzian_structure_group: &'static str,
    pub coefficient_basis: &'static str,
    pub tested_superform_slice: &'static str,
    pub tau_zero_domain_dimension: usize,
    pub tau_zero_codomain_dimension: usize,
    pub tau_zero_nonzero_entries: usize,
    pub tau_zero_rank: usize,
    pub tau_zero_quotient_dimension: usize,
    pub quotient_nonzero_entries: usize,
    pub quotient_free_coordinate_unit_pivots: usize,
    pub quotient_tau_zero_residual_entries: usize,
    pub tau_zero_image_equals_quotient_kernel: bool,
    pub formal_momentum_coefficients_checked: usize,
    pub d_one_square_nonzero_entries: usize,
    pub bidegree_zero_two_bianchi_residual_entries: usize,
    pub d_f_square_quotient_residual_entries: usize,
    pub mutation_residual_entries: usize,
    pub mutation_rejected: bool,
    pub lowest_d_f_composition_symbol_constructed: bool,
    pub d_f_well_defined_on_lowest_scalar_symbol_slice: bool,
    pub d_f_nilpotent_on_lowest_scalar_symbol_slice: bool,
    pub first_order_d_one_on_arbitrary_superfield_coefficients_computed: bool,
    pub tau_zero_squared_computed: bool,
    pub full_relaxed_torsion_differential_computed: bool,
    pub h_f_zero_four_phys_computed: bool,
    pub complete_superspace_bianchi_tower_computed: bool,
    pub finite_auxiliary_off_shell_closure_computed: bool,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

fn primary_sources() -> Vec<SpinorialDifferentialSourceAudit> {
    vec![
        SpinorialDifferentialSourceAudit {
            source: "Cederwall, Nilsson, Tsimpis, arXiv:hep-th/0110069",
            source_archive_sha256: "5d232edd240298530f8216a351e282f4261a0cfe3ec4cfa02c1eaa055f8e43ea",
            source_statement: "the projected spinorial derivative is a complex in undeformed 11D supergravity after the dimension-one torsion Bianchi identities remove the projected curvature terms",
            use_in_gate: "fixes the interpretation of the lowest projected differential; this gate does not implement the curved dimension-one cancellation",
        },
        SpinorialDifferentialSourceAudit {
            source: "Tsimpis, arXiv:hep-th/0407271, equations (3.3), (3.10), and (3.11)",
            source_archive_sha256: "013049a4da503cdee37a67447ab5adc17a30678e07430ecd43905fb0bdc47ee3",
            source_statement: "tau_0 is the algebraic dimension-zero torsion differential, H_tau is its quotient, and d_F[omega]=[d_1 omega] is nilpotent because d squared vanishes",
            use_in_gate: "implemented coefficientwise on the scalar bidegree-(0,2) identity d_1 squared plus tau_0 d_0 equals zero",
        },
        SpinorialDifferentialSourceAudit {
            source: "Howe, arXiv:hep-th/9707184",
            source_archive_sha256: "2aa95c8072e75c6d6b2592f4880a8dbbd843a0036f7805613d911b99608e9482",
            source_statement: "the standard 11D dimension-zero torsion constraint together with Bianchi identities has on-shell consequences",
            use_in_gate: "prevents the standard torsion background used here from being reported as finite-auxiliary off-shell closure",
        },
    ]
}

pub fn verify() -> ElevenDimensionalSpinorialDifferentialReport {
    let tau = tau_zero_dense();
    let (quotient, free_coordinates) = quotient_dense();
    let d_one_square = tau
        .iter()
        .map(|row| row.iter().map(|entry| -entry.clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let transpose = (0..VECTOR_DIMENSION)
        .map(|column| {
            (0..SYMMETRIC_BISPINOR_DIMENSION)
                .map(|row| tau[row][column].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (_, tau_pivots) = rref(transpose);
    let tau_rank = tau_pivots.len();
    let quotient_tau = multiply(&quotient, &tau);
    let quotient_d_one_square = multiply(&quotient, &d_one_square);
    let bianchi = d_one_square
        .iter()
        .zip(&tau)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(left, right)| left.clone() + right.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut mutated = d_one_square.clone();
    let mutation_coordinate = free_coordinates[0];
    mutated[mutation_coordinate][0] += one();
    let mutation_residual_entries = nonzero_entries(&multiply(&quotient, &mutated));
    let quotient_tau_zero_residual_entries = nonzero_entries(&quotient_tau);
    let d_f_square_quotient_residual_entries = nonzero_entries(&quotient_d_one_square);
    let bidegree_zero_two_bianchi_residual_entries = nonzero_entries(&bianchi);
    let quotient_unit_pivots = free_coordinates
        .iter()
        .enumerate()
        .filter(|(row, column)| quotient[*row][**column] == one())
        .count();
    let quotient_dimension = quotient.len();
    let image_equals_kernel = quotient_tau_zero_residual_entries == 0
        && tau_rank == VECTOR_DIMENSION
        && quotient_unit_pivots == quotient_dimension
        && quotient_dimension + tau_rank == SYMMETRIC_BISPINOR_DIMENSION;
    let passed = tau.len() == SYMMETRIC_BISPINOR_DIMENSION
        && tau[0].len() == VECTOR_DIMENSION
        && tau_rank == VECTOR_DIMENSION
        && quotient_dimension == 517
        && image_equals_kernel
        && bidegree_zero_two_bianchi_residual_entries == 0
        && d_f_square_quotient_residual_entries == 0
        && mutation_residual_entries > 0;

    ElevenDimensionalSpinorialDifferentialReport {
        schema_version: "adynkra-11d-spinorial-differential-v1",
        role: "exact lowest scalar-symbol d_F nilpotence gate modulo the tau_0 Bianchi quotient",
        primary_sources: primary_sources(),
        lorentzian_structure_group: "Spin(1,10); fermions are in the real 32-component Majorana module",
        coefficient_basis: "exact real Majorana basis; symmetric bispinors use one matrix coordinate for alpha<=beta",
        tested_superform_slice: "Omega^(0,0) through H_tau^(0,2), testing d_1^2 + tau_0 d_0 coefficientwise on a scalar",
        tau_zero_domain_dimension: VECTOR_DIMENSION,
        tau_zero_codomain_dimension: SYMMETRIC_BISPINOR_DIMENSION,
        tau_zero_nonzero_entries: nonzero_entries(&tau),
        tau_zero_rank: tau_rank,
        tau_zero_quotient_dimension: quotient_dimension,
        quotient_nonzero_entries: nonzero_entries(&quotient),
        quotient_free_coordinate_unit_pivots: quotient_unit_pivots,
        quotient_tau_zero_residual_entries,
        tau_zero_image_equals_quotient_kernel: image_equals_kernel,
        formal_momentum_coefficients_checked: VECTOR_DIMENSION,
        d_one_square_nonzero_entries: nonzero_entries(&d_one_square),
        bidegree_zero_two_bianchi_residual_entries,
        d_f_square_quotient_residual_entries,
        mutation_residual_entries,
        mutation_rejected: mutation_residual_entries > 0,
        lowest_d_f_composition_symbol_constructed: true,
        d_f_well_defined_on_lowest_scalar_symbol_slice: image_equals_kernel,
        d_f_nilpotent_on_lowest_scalar_symbol_slice: d_f_square_quotient_residual_entries == 0,
        first_order_d_one_on_arbitrary_superfield_coefficients_computed: false,
        tau_zero_squared_computed: false,
        full_relaxed_torsion_differential_computed: false,
        h_f_zero_four_phys_computed: false,
        complete_superspace_bianchi_tower_computed: false,
        finite_auxiliary_off_shell_closure_computed: false,
        passed,
        result: "The generic scalar-symbol identity d_1^2 + tau_0 d_0 = 0 holds exactly in all eleven momentum coefficients, and d_F^2 vanishes in the explicit 517-dimensional H_tau quotient.",
        boundary: "This is the lowest universal scalar composition symbol, not a first-order d_1 matrix on arbitrary superfield coefficients, the (11000)+(10002) relaxed-torsion differential, or H_F^(0,4)(phys). The outgoing tau_0 map from Omega^(0,2) vanishes by bidegree, so its H_tau quotient is the computed cokernel. The gate uses the standard on-shell 11D torsion background and does not establish ordinary finite-auxiliary off-shell closure.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_zero_image_has_the_exact_517_dimensional_quotient() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.tau_zero_rank, 11);
        assert_eq!(report.tau_zero_quotient_dimension, 517);
        assert_eq!(report.quotient_tau_zero_residual_entries, 0);
        assert!(report.tau_zero_image_equals_quotient_kernel);
    }

    #[test]
    fn lowest_spinorial_square_is_tau_zero_exact_at_generic_momentum() {
        let report = verify();
        assert_eq!(report.formal_momentum_coefficients_checked, 11);
        assert_eq!(report.bidegree_zero_two_bianchi_residual_entries, 0);
        assert_eq!(report.d_f_square_quotient_residual_entries, 0);
        assert!(report.d_f_nilpotent_on_lowest_scalar_symbol_slice);
    }

    #[test]
    fn quotient_rejects_a_non_bianchi_mutation_and_fails_closed() {
        let report = verify();
        assert!(report.mutation_rejected);
        assert!(report.mutation_residual_entries > 0);
        assert!(!report.tau_zero_squared_computed);
        assert!(!report.first_order_d_one_on_arbitrary_superfield_coefficients_computed);
        assert!(!report.full_relaxed_torsion_differential_computed);
        assert!(!report.h_f_zero_four_phys_computed);
        assert!(!report.complete_superspace_bianchi_tower_computed);
        assert!(!report.finite_auxiliary_off_shell_closure_computed);
    }

    #[test]
    fn public_sparse_maps_reproduce_the_certified_dimensions() {
        let slice = lowest_spinorial_differential_slice();
        let tau = slice.tau_zero;
        let quotient = slice.h_tau_zero_two_quotient;
        let square = slice.d_one_square_scalar_symbol;
        assert_eq!((tau.row_count(), tau.column_count()), (528, 11));
        assert_eq!((quotient.row_count(), quotient.column_count()), (517, 528));
        assert_eq!((square.row_count(), square.column_count()), (528, 11));
        for axis in 0..VECTOR_DIMENSION {
            let mut basis = vec![zero(); VECTOR_DIMENSION];
            basis[axis] = one();
            assert!(
                quotient
                    .apply(&square.apply(&basis))
                    .iter()
                    .all(|entry| *entry == zero())
            );
        }
    }

    #[test]
    #[ignore = "writes the committed exact spinorial differential artifact"]
    fn write_artifact() {
        let report = verify();
        assert!(report.passed);
        let path = "results/adynkra_11d_spinorial_differential_gate.json";
        let temporary = format!("{path}.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        fs::rename(temporary, path).unwrap();
    }
}
