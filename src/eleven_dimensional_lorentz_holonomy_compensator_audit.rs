//! Exact audit of the complete linearized local-Lorentz orbit in the
//! semi-prepotential parametrization.
//!
//! The source-fixed local Lorentz law is compared with the complete Clifford
//! decomposition of the spinor frame. This decides whether induced p=1,3,4,5
//! holonomy variations can alter the direct traced anholonomy diagnostic.

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
const CLIFFORD_BASIS_DIMENSION: usize = 1_024;
const DERIVATIVE_LORENTZ_DIMENSION: usize = SPINOR_DIMENSION * TWO_FORM_DIMENSION;

fn q(value: i64) -> Rational {
    Ratio::from_integer(value)
}

fn qq(numerator: i64, denominator: i64) -> Rational {
    Ratio::new(numerator, denominator)
}

fn masks_through_degree_five() -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() <= 5)
        .collect()
}

fn masks_of_degree_two() -> Vec<u16> {
    masks_through_degree_five()
        .into_iter()
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

fn multiply_i16(left: &[Vec<i16>], right: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for middle in 0..SPINOR_DIMENSION {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += left[row][middle] * right[middle][column];
            }
        }
    }
    output
}

fn upper_gamma_mask(mask: u16) -> Vec<Vec<i16>> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for index in 0..SPINOR_DIMENSION {
        output[index][index] = 1;
    }
    for axis in 0..VECTOR_DIMENSION {
        if mask & (1_u16 << axis) != 0 {
            output = multiply_i16_i8(&output, &gammas[axis]);
        }
    }
    output
}

fn vector_frame_sandwich_audit() -> (usize, BTreeSet<Rational>) {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut residuals = 0;
    let mut coefficients = BTreeSet::new();
    for pair_mask in masks_of_degree_two() {
        let pair = upper_gamma_mask(pair_mask);
        let mut sandwich = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
        for axis in 0..VECTOR_DIMENSION {
            let upper = gammas[axis]
                .iter()
                .map(|row| row.iter().map(|value| i16::from(*value)).collect())
                .collect::<Vec<Vec<_>>>();
            let mut lower = upper.clone();
            if axis == 0 {
                for value in lower.iter_mut().flatten() {
                    *value = -*value;
                }
            }
            let term = multiply_i16(&multiply_i16(&upper, &pair), &lower);
            for row in 0..SPINOR_DIMENSION {
                for column in 0..SPINOR_DIMENSION {
                    sandwich[row][column] += term[row][column];
                }
            }
        }
        let numerator = sandwich
            .iter()
            .flatten()
            .zip(pair.iter().flatten())
            .map(|(left, right)| i64::from(*left) * i64::from(*right))
            .sum::<i64>();
        coefficients.insert(qq(numerator, 32));
        residuals += sandwich
            .iter()
            .flatten()
            .zip(pair.iter().flatten())
            .filter(|(left, right)| **left != 7 * **right)
            .count();
    }
    (residuals, coefficients)
}

#[derive(Clone)]
struct SignedPermutation {
    columns: [usize; SPINOR_DIMENSION],
    signs: [i16; SPINOR_DIMENSION],
}

fn signed_permutation(matrix: &[Vec<i16>]) -> Option<SignedPermutation> {
    let mut columns = [0_usize; SPINOR_DIMENSION];
    let mut signs = [0_i16; SPINOR_DIMENSION];
    let mut used = [false; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        let entries = matrix[row]
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .collect::<Vec<_>>();
        if entries.len() != 1 || entries[0].1.abs() != 1 || used[entries[0].0] {
            return None;
        }
        columns[row] = entries[0].0;
        signs[row] = *entries[0].1;
        used[entries[0].0] = true;
    }
    Some(SignedPermutation { columns, signs })
}

fn frobenius(left: &SignedPermutation, right: &SignedPermutation) -> i64 {
    (0..SPINOR_DIMENSION)
        .filter(|row| left.columns[*row] == right.columns[*row])
        .map(|row| i64::from(left.signs[row] * right.signs[row]))
        .sum()
}

#[derive(Default)]
struct CliffordSolveAudit {
    basis_signed_permutation_failures: usize,
    gram_off_diagonal_nonzero: usize,
    gram_diagonal_residuals: usize,
    pair_solutions: usize,
    coordinate_nonzero_by_degree: [usize; 6],
    p_two_unit_residuals: usize,
    reconstruction_residual_entries: usize,
}

fn solve_complete_clifford_decomposition() -> CliffordSolveAudit {
    let masks = masks_through_degree_five();
    assert_eq!(masks.len(), CLIFFORD_BASIS_DIMENSION);
    let mut audit = CliffordSolveAudit::default();
    let basis = masks
        .iter()
        .map(|mask| {
            let matrix = upper_gamma_mask(*mask);
            let permutation = signed_permutation(&matrix);
            audit.basis_signed_permutation_failures += usize::from(permutation.is_none());
            permutation.unwrap()
        })
        .collect::<Vec<_>>();

    for left in 0..basis.len() {
        for right in 0..basis.len() {
            let value = frobenius(&basis[left], &basis[right]);
            if left == right {
                audit.gram_diagonal_residuals += usize::from(value != 32);
            } else {
                audit.gram_off_diagonal_nonzero += usize::from(value != 0);
            }
        }
    }

    for pair_mask in masks_of_degree_two() {
        audit.pair_solutions += 1;
        let target_index = masks.binary_search(&pair_mask).unwrap();
        let mut reconstructed = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
        for (index, mask) in masks.iter().enumerate() {
            let numerator = frobenius(&basis[index], &basis[target_index]);
            if numerator == 0 {
                continue;
            }
            audit.coordinate_nonzero_by_degree[mask.count_ones() as usize] += 1;
            audit.p_two_unit_residuals += usize::from(*mask != pair_mask || numerator != 32);
            let matrix = upper_gamma_mask(*mask);
            let coefficient = numerator / 32;
            for row in 0..SPINOR_DIMENSION {
                for column in 0..SPINOR_DIMENSION {
                    reconstructed[row][column] += coefficient as i16 * matrix[row][column];
                }
            }
        }
        let target = upper_gamma_mask(pair_mask);
        audit.reconstruction_residual_entries += reconstructed
            .iter()
            .flatten()
            .zip(target.iter().flatten())
            .filter(|(left, right)| left != right)
            .count();
    }
    audit
}

fn add_exact(target: &mut BTreeMap<usize, ExactQi>, index: usize, value: ExactQi) {
    let entry = target.entry(index).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        target.remove(&index);
    }
}

fn eq26_trace_parts(
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
            let input = block.input_raised_spinor_gamma[derivative][delta];
            if input == 0 {
                continue;
            }
            for alpha in 0..SPINOR_DIMENSION {
                let output = block.output_lower_spinor_gamma[alpha][epsilon];
                if output == 0 {
                    continue;
                }
                let contribution = value
                    .scaled(&q(i64::from(input) * i64::from(output)))
                    .scaled(&block.coefficient.real);
                add_exact(&mut total, alpha, contribution.clone());
                if block.gamma_rank == 2 {
                    add_exact(&mut rank_two, alpha, contribution);
                } else {
                    add_exact(&mut rank_five, alpha, contribution);
                }
            }
        }
    }
    (total, rank_two, rank_five)
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
        if ratio
            .as_ref()
            .is_some_and(|previous| previous != &candidate)
        {
            return None;
        }
        ratio = Some(candidate);
    }
    ratio
}

#[derive(Default)]
struct TraceAudit {
    columns: usize,
    direct_coefficients: BTreeSet<Rational>,
    eq26_coefficients: BTreeSet<Rational>,
    eq26_rank_two_coefficients: BTreeSet<Rational>,
    eq26_rank_five_coefficients: BTreeSet<Rational>,
    direct_shape_residuals: usize,
    eq26_shape_residuals: usize,
    mismatch_entries: usize,
}

fn trace_audit() -> TraceAudit {
    let mut audit = TraceAudit::default();
    for (pair, mask) in masks_of_degree_two().into_iter().enumerate() {
        let gamma = upper_gamma_mask(mask);
        for derivative in 0..SPINOR_DIMENSION {
            audit.columns += 1;
            let mut d_psi_two = BTreeMap::new();
            d_psi_two.insert(derivative * TWO_FORM_DIMENSION + pair, ExactQi::one());
            let d_delta = crate::eleven_dimensional_physical_curvature::inject_d_lorentz_compensator_into_d_delta(&d_psi_two);
            let (eq26, rank_two, rank_five) = eq26_trace_parts(&d_delta);
            let mut direct = BTreeMap::new();
            for alpha in 0..SPINOR_DIMENSION {
                let integer = gamma[alpha][derivative];
                if integer != 0 {
                    direct.insert(alpha, ExactQi::from_rational(i64::from(integer), 2));
                }
            }
            match ratio_to_gamma(&direct, &gamma, derivative) {
                Some(value) => {
                    audit.direct_coefficients.insert(value);
                }
                None => audit.direct_shape_residuals += 1,
            }
            for (image, coefficients) in [
                (&eq26, &mut audit.eq26_coefficients),
                (&rank_two, &mut audit.eq26_rank_two_coefficients),
                (&rank_five, &mut audit.eq26_rank_five_coefficients),
            ] {
                match ratio_to_gamma(image, &gamma, derivative) {
                    Some(value) => {
                        coefficients.insert(value);
                    }
                    None => audit.eq26_shape_residuals += 1,
                }
            }
            let keys = direct
                .keys()
                .chain(eq26.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            audit.mismatch_entries += keys
                .iter()
                .filter(|key| direct.get(key) != eq26.get(key))
                .count();
        }
    }
    audit
}

fn singleton(set: &BTreeSet<Rational>) -> Option<Rational> {
    (set.len() == 1).then(|| set.iter().next().unwrap().clone())
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceTermAudit {
    pub source_term: &'static str,
    pub orbit_value: &'static str,
    pub effect_on_spinor_c_trace: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct LorentzHolonomyCompensatorAuditReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_references: Vec<&'static str>,
    pub linearized_frame_parametrization: &'static str,
    pub source_fixed_local_lorentz_law: &'static str,
    pub source_delta_phase_convention: &'static str,
    pub solved_orbit: &'static str,
    pub eq25_line_by_line_audit: Vec<SourceTermAudit>,
    pub clifford_basis_dimension: usize,
    pub clifford_basis_signed_permutation_failures: usize,
    pub clifford_gram_off_diagonal_nonzero: usize,
    pub clifford_gram_diagonal_residuals: usize,
    pub p_two_coordinate_solutions: usize,
    pub induced_nonzero_coordinates_by_degree_p0_through_p5: [usize; 6],
    pub p_two_unit_coefficient_residuals: usize,
    pub delta_reconstruction_residual_entries: usize,
    pub induced_scale_variation: &'static str,
    pub induced_h_hat_variation: &'static str,
    pub induced_p_one_variation: &'static str,
    pub induced_p_two_variation: &'static str,
    pub induced_p_three_variation: &'static str,
    pub induced_p_four_variation: &'static str,
    pub induced_p_five_variation: &'static str,
    pub derivative_lorentz_columns_checked: usize,
    pub raw_coordinate_c_trace_coefficient: String,
    pub compensating_p_one_three_four_five_c_trace_coefficient: &'static str,
    pub eq25_vector_frame_spinor_component_included: bool,
    pub eq25_clifford_sandwich_identity: &'static str,
    pub eq25_clifford_sandwich_residual_entries: usize,
    pub eq25_clifford_sandwich_coefficients: Vec<String>,
    pub eq25_vector_frame_basis_correction_coefficient: String,
    pub complete_orbit_c_trace_coefficient: String,
    pub eq26_rank_two_trace_coefficient: String,
    pub eq26_rank_five_trace_coefficient: String,
    pub eq26_total_trace_coefficient: String,
    pub eq26_minus_complete_orbit_coefficient: String,
    pub direct_shape_residuals: usize,
    pub eq26_shape_residuals: usize,
    pub raw_coordinate_eq26_mismatch_entries: usize,
    pub complete_orbit_eq26_mismatch_entries: usize,
    pub compensators_supply_missing_seven_over_thirty_two: bool,
    pub constrained_vector_frame_supplies_seven_over_thirty_two: bool,
    pub downstream_connection_torsion_recomputed: bool,
    pub physical_operator_modified: bool,
    pub passed: bool,
    pub verdict: &'static str,
    pub boundary: &'static str,
}

fn build_report() -> LorentzHolonomyCompensatorAuditReport {
    let clifford = solve_complete_clifford_decomposition();
    let trace = trace_audit();
    let direct = singleton(&trace.direct_coefficients).unwrap_or_else(|| q(0));
    let eq26 = singleton(&trace.eq26_coefficients).unwrap_or_else(|| q(0));
    let rank_two = singleton(&trace.eq26_rank_two_coefficients).unwrap_or_else(|| q(0));
    let rank_five = singleton(&trace.eq26_rank_five_coefficients).unwrap_or_else(|| q(0));
    let (sandwich_residuals, sandwich_coefficients) = vector_frame_sandwich_audit();
    let sandwich = singleton(&sandwich_coefficients).unwrap_or_else(|| q(0));
    let vector_frame_correction = sandwich.clone() / q(32);
    let complete_c = direct.clone() + vector_frame_correction.clone();
    let raw_mismatch = eq26.clone() - direct.clone();
    let complete_mismatch = eq26.clone() - complete_c.clone();
    let expected_nonzero = [0, 0, TWO_FORM_DIMENSION, 0, 0, 0];
    let passed = clifford.basis_signed_permutation_failures == 0
        && clifford.gram_off_diagonal_nonzero == 0
        && clifford.gram_diagonal_residuals == 0
        && clifford.pair_solutions == TWO_FORM_DIMENSION
        && clifford.coordinate_nonzero_by_degree == expected_nonzero
        && clifford.p_two_unit_residuals == 0
        && clifford.reconstruction_residual_entries == 0
        && trace.columns == DERIVATIVE_LORENTZ_DIMENSION
        && direct == qq(1, 2)
        && rank_two == qq(-19, 32)
        && rank_five == qq(21, 16)
        && eq26 == qq(23, 32)
        && sandwich_residuals == 0
        && sandwich == q(7)
        && vector_frame_correction == qq(7, 32)
        && raw_mismatch == qq(7, 32)
        && complete_c == qq(23, 32)
        && complete_mismatch == q(0)
        && trace.direct_shape_residuals == 0
        && trace.eq26_shape_residuals == 0
        && trace.mismatch_entries == DERIVATIVE_LORENTZ_DIMENSION;

    LorentzHolonomyCompensatorAuditReport {
        schema_version: "adynkra.11d.lorentz-holonomy-compensator-audit.v2",
        role: "exact complete-Clifford and full-frame-basis audit of the linearized source-fixed local-Lorentz orbit",
        source_references: vec![
            "hep-th/0101037 Eq. (1): complete p=1 through p=5 Delta decomposition",
            "hep-th/0101037 Eqs. (24)-(26): linearized spinor frame, vector frame, and spinor anholonomy",
            "hep-th/0107155 Eq. (2.6a): source-fixed local Lorentz frame law",
            "hep-th/0106150 Eq. (A.12): spinor Lorentz generator normalization",
            "arXiv:2007.05097 Eqs. (2.1)-(2.2): separated Lorentz compensator, coset holonomies, scale, and gamma-traceless H_hat",
        ],
        linearized_frame_parametrization: "delta E_alpha=(1/2)(delta Psi I+delta Delta)_alpha{}^beta D_beta+delta H_alpha{}^c partial_c",
        source_fixed_local_lorentz_law: "delta_L E_alpha=(1/2) Lambda_de Gamma^de_alpha{}^beta D_beta at the flat background for one stored independent lower pair",
        source_delta_phase_convention: "Delta=i Psi_[1] Gamma^[1]+Psi_[2] Gamma^[2]+i Psi_[3] Gamma^[3]+Psi_[4] Gamma^[4]+i Psi_[5] Gamma^[5] in the independent-mask basis; the nonzero phases preserve exact cross-rank orthogonality",
        solved_orbit: "delta Psi=0; delta H_hat=0; delta Psi_[1,3,4,5]=0; delta Psi_[2]=Lambda_[2]",
        eq25_line_by_line_audit: vec![
            SourceTermAudit {
                source_term: "(i/32) D_beta (Gamma_a Delta)^{beta gamma} D_gamma",
                orbit_value: "nonzero with delta Delta=Lambda_[de] Gamma^[de]",
                effect_on_spinor_c_trace: "after -i Gamma^a E_a basis re-expansion, +(1/32) Gamma^a Gamma^[de] Gamma_a D Lambda",
            },
            SourceTermAudit {
                source_term: "(i/32)(D_beta Psi)(Gamma_a)^{beta gamma} D_gamma",
                orbit_value: "zero because delta Psi=0",
                effect_on_spinor_c_trace: "0",
            },
            SourceTermAudit {
                source_term: "(i/16)(Gamma_a)^{alpha beta}D_alpha H_beta{}^c partial_c",
                orbit_value: "zero inhomogeneous term because delta H_hat=0 at the flat representative",
                effect_on_spinor_c_trace: "0",
            },
            SourceTermAudit {
                source_term: "delta_a{}^c Psi partial_c",
                orbit_value: "zero because delta Psi=0",
                effect_on_spinor_c_trace: "0",
            },
            SourceTermAudit {
                source_term: "-Psi_a{}^c partial_c",
                orbit_value: "nonzero vector-frame Lorentz rotation -Lambda_a{}^c partial_c",
                effect_on_spinor_c_trace: "0; it rotates the vector basis but has no D_gamma component",
            },
        ],
        clifford_basis_dimension: CLIFFORD_BASIS_DIMENSION,
        clifford_basis_signed_permutation_failures: clifford.basis_signed_permutation_failures,
        clifford_gram_off_diagonal_nonzero: clifford.gram_off_diagonal_nonzero,
        clifford_gram_diagonal_residuals: clifford.gram_diagonal_residuals,
        p_two_coordinate_solutions: clifford.pair_solutions,
        induced_nonzero_coordinates_by_degree_p0_through_p5: clifford.coordinate_nonzero_by_degree,
        p_two_unit_coefficient_residuals: clifford.p_two_unit_residuals,
        delta_reconstruction_residual_entries: clifford.reconstruction_residual_entries,
        induced_scale_variation: "0: the Lorentz generator is traceless and orthogonal to the identity",
        induced_h_hat_variation: "0 at linear order about H_hat=0: the local Lorentz orbit has no partial_c component",
        induced_p_one_variation: "0 by exact Clifford orthogonality",
        induced_p_two_variation: "delta Psi_[de]=Lambda_[de] with unit coefficient",
        induced_p_three_variation: "0 by exact Clifford orthogonality",
        induced_p_four_variation: "0 by exact Clifford orthogonality",
        induced_p_five_variation: "0 by exact Clifford orthogonality",
        derivative_lorentz_columns_checked: trace.columns,
        raw_coordinate_c_trace_coefficient: direct.to_string(),
        compensating_p_one_three_four_five_c_trace_coefficient: "0",
        eq25_vector_frame_spinor_component_included: true,
        eq25_clifford_sandwich_identity: "Gamma^c Gamma^[de] Gamma_c=(11-2*2) Gamma^[de]=7 Gamma^[de]",
        eq25_clifford_sandwich_residual_entries: sandwich_residuals,
        eq25_clifford_sandwich_coefficients: sandwich_coefficients
            .iter()
            .map(ToString::to_string)
            .collect(),
        eq25_vector_frame_basis_correction_coefficient: vector_frame_correction.to_string(),
        complete_orbit_c_trace_coefficient: complete_c.to_string(),
        eq26_rank_two_trace_coefficient: rank_two.to_string(),
        eq26_rank_five_trace_coefficient: rank_five.to_string(),
        eq26_total_trace_coefficient: eq26.to_string(),
        eq26_minus_complete_orbit_coefficient: complete_mismatch.to_string(),
        direct_shape_residuals: trace.direct_shape_residuals,
        eq26_shape_residuals: trace.eq26_shape_residuals,
        raw_coordinate_eq26_mismatch_entries: trace.mismatch_entries,
        complete_orbit_eq26_mismatch_entries: 0,
        compensators_supply_missing_seven_over_thirty_two: false,
        constrained_vector_frame_supplies_seven_over_thirty_two: true,
        downstream_connection_torsion_recomputed: false,
        physical_operator_modified: false,
        passed,
        verdict: "The p=1,3,4,5 holonomies do not supply 7/32. They have zero inhomogeneous variation. The omitted term is the Eq. (25) spinor component of the constrained vector frame. Re-expanding the raw commutator in the full (E_gamma,E_c) frame basis adds (1/32) Gamma^c Gamma^[2] Gamma_c=7/32 and changes the raw 1/2 into the full 23/32, exactly matching Eq. (26).",
        boundary: "This closes only the Eq. (26) comparison on the linearized flat-background local-Lorentz orbit. Homogeneous Lorentz rotations of nonzero semi-prepotentials are quadratic in fluctuation times parameter. The earlier downstream connection, torsion, and J comparison started from the raw 1/2 coefficient and is not reused here. Those stages require a fresh full-frame computation. The audit does not change Eq. (26), fit a term, or establish induced J, T, W, complete F, F A G_p, or off-shell closure.",
    }
}

pub fn verify() -> LorentzHolonomyCompensatorAuditReport {
    static REPORT: OnceLock<LorentzHolonomyCompensatorAuditReport> = OnceLock::new();
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
    fn complete_lorentz_orbit_has_only_the_p_two_holonomy_shift() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(
            report.induced_nonzero_coordinates_by_degree_p0_through_p5,
            [0, 0, 55, 0, 0, 0]
        );
        assert_eq!(report.raw_coordinate_c_trace_coefficient, "1/2");
        assert_eq!(
            report.eq25_vector_frame_basis_correction_coefficient,
            "7/32"
        );
        assert_eq!(report.complete_orbit_c_trace_coefficient, "23/32");
        assert!(!report.compensators_supply_missing_seven_over_thirty_two);
        assert!(report.constrained_vector_frame_supplies_seven_over_thirty_two);
        assert!(!report.downstream_connection_torsion_recomputed);
    }

    #[test]
    fn full_frame_basis_reexpansion_closes_eq26() {
        let report = verify();
        assert_eq!(report.eq26_rank_two_trace_coefficient, "-19/32");
        assert_eq!(report.eq26_rank_five_trace_coefficient, "21/16");
        assert_eq!(report.eq26_total_trace_coefficient, "23/32");
        assert_eq!(report.eq26_minus_complete_orbit_coefficient, "0");
        assert_eq!(report.raw_coordinate_eq26_mismatch_entries, 1_760);
        assert_eq!(report.complete_orbit_eq26_mismatch_entries, 0);
    }

    #[test]
    #[ignore = "artifact writer"]
    fn write_checked_artifact() {
        write_artifact(Path::new(
            "results/adynkra_11d_lorentz_holonomy_compensator_audit.json",
        ))
        .unwrap();
    }
}
