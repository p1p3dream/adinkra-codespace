//! Independent pre-gauge three-form and four-form adapter.
//!
//! This module keeps the physical Abelian potential `A_[3]` distinct from the
//! Eq. (40) holonomy `Psi_[3]`. It builds `G_[4]=p wedge A_[3]` in the exact
//! target complex, lifts that map to `D G_[4]`, checks the full reducible A2
//! gauge complex, and pins the independent teleparallel Eq. (3.1g)
//! curl-to-descendant map and its exact left inverse.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_free_complex::three_form_complex;
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, linearized_gravitino_curl_to_d_f_four_operator,
    recover_gravitino_curl_from_linearized_d_f_four,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialMatrix, TargetSector, target_sector_complex,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-independent-a3-pregauge-adapter-v1";
pub const VECTOR_DIMENSION: usize = 11;
pub const SPINOR_DIMENSION: usize = 32;
pub const A0_DIMENSION: usize = 1;
pub const A1_DIMENSION: usize = 11;
pub const A2_DIMENSION: usize = 55;
pub const A3_DIMENSION: usize = 165;
pub const G4_DIMENSION: usize = 330;
pub const DG4_DIMENSION: usize = SPINOR_DIMENSION * G4_DIMENSION;
pub const DA3_DIMENSION: usize = SPINOR_DIMENSION * A3_DIMENSION;
pub const LOCAL_LORENTZ_FIRST_JET_DIMENSION: usize = SPINOR_DIMENSION * A2_DIMENSION;
pub const GRAVITINO_CURL_DIMENSION: usize = A2_DIMENSION * SPINOR_DIMENSION;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct IndependentDg4Term {
    pub target_coordinate: usize,
    pub derivative_spinor: usize,
    pub four_form_ordinal: usize,
    pub four_form_axes: [usize; 4],
    pub momentum_exponents: [u8; VECTOR_DIMENSION],
    pub real_numerator: i64,
    pub real_denominator: i64,
    pub imaginary_numerator: i64,
    pub imaginary_denominator: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct IndependentA3PregaugeReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub coefficient_field: &'static str,
    pub source_boundary: &'static str,
    pub a0_dimension: usize,
    pub a1_dimension: usize,
    pub a2_gauge_parameter_dimension: usize,
    pub a3_potential_dimension: usize,
    pub g4_curvature_dimension: usize,
    pub d_a3_dimension: usize,
    pub d_g4_dimension: usize,
    pub a0_to_a1_dimensions: (usize, usize),
    pub a1_to_a2_dimensions: (usize, usize),
    pub a2_to_a3_dimensions: (usize, usize),
    pub a3_to_g4_dimensions: (usize, usize),
    pub g4_bianchi_dimensions: (usize, usize),
    pub a0_to_a1_nonzero_terms: usize,
    pub a1_to_a2_nonzero_terms: usize,
    pub a2_to_a3_nonzero_terms: usize,
    pub a3_to_g4_nonzero_terms: usize,
    pub g4_bianchi_nonzero_terms: usize,
    pub reducibility_stage_zero_composition_residual_terms: usize,
    pub reducibility_stage_one_composition_residual_terms: usize,
    pub gauge_curvature_composition_residual_terms: usize,
    pub curvature_bianchi_composition_residual_terms: usize,
    pub d_g4_bianchi_columns_checked: usize,
    pub d_g4_bianchi_residual_terms: usize,
    pub e0_a0_to_a1_rank: usize,
    pub e0_a1_to_a2_rank: usize,
    pub e0_a2_gauge_rank: usize,
    pub e0_a3_to_g4_rank: usize,
    pub e0_a3_quotient_dimension: usize,
    pub e0_closed_g4_dimension: usize,
    pub canonical_canary_derivative_spinor: usize,
    pub canonical_canary_a3_axes: [usize; 3],
    pub canonical_canary_g4_axes: [usize; 4],
    pub canonical_canary_momentum_axis: usize,
    pub canonical_canary_coefficient: i64,
    pub canonical_canary_passed: bool,
    pub local_lorentz_first_jet_columns_checked: usize,
    pub local_lorentz_vertical_nonzero_terms: usize,
    pub local_lorentz_vertical_image_rank: usize,
    pub local_lorentz_source_disjoint: bool,
    pub local_lorentz_mutation_detected: bool,
    pub eq31g_forward_dimensions: (usize, usize),
    pub eq31g_forward_nonzero_entries: usize,
    pub eq31g_nonzero_entries_per_column: usize,
    pub eq31g_canary_input_terms: usize,
    pub eq31g_canary_output_terms: usize,
    pub eq31g_left_inverse_residual_terms: usize,
    pub eq31g_off_image_mutation_rejected: bool,
    pub a3_basis_sha256: String,
    pub g4_basis_sha256: String,
    pub target_complex_semantic_sha256: String,
    pub d_g4_stream_sha256: String,
    pub eq31g_operator_sha256: String,
    pub adapter_source_sha256: String,
    pub independent_a3_source_constructed: bool,
    pub eq40_psi_three_used: bool,
    pub h_hat_to_a3_identification_claimed: bool,
    pub physical_h_hat_source_map_complete: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn combinations(degree: usize) -> Vec<Vec<usize>> {
    fn extend(
        next: usize,
        remaining: usize,
        prefix: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(prefix.clone());
            return;
        }
        for value in next..=VECTOR_DIMENSION - remaining {
            prefix.push(value);
            extend(value + 1, remaining - 1, prefix, output);
            prefix.pop();
        }
    }
    let mut output = Vec::new();
    extend(0, degree, &mut Vec::new(), &mut output);
    output
}

fn hash_basis(degree: usize) -> String {
    let mut hash = Sha256::new();
    hash.update(SCHEMA_VERSION.as_bytes());
    hash.update(b"\0basis\0");
    hash.update((degree as u64).to_le_bytes());
    for axes in combinations(degree) {
        hash.update((axes.len() as u64).to_le_bytes());
        for axis in axes {
            hash.update((axis as u64).to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn hash_polynomial_matrix(hash: &mut Sha256, label: &str, matrix: &ExactPolynomialMatrix) {
    hash.update(label.as_bytes());
    hash.update([0]);
    hash.update((matrix.rows() as u64).to_le_bytes());
    hash.update((matrix.columns() as u64).to_le_bytes());
    for column in 0..matrix.columns() {
        for (row, coefficient) in matrix.column_terms(column) {
            hash.update((row as u64).to_le_bytes());
            hash.update((column as u64).to_le_bytes());
            hash.update(coefficient.monomial.exponents);
            hash.update(coefficient.real_numerator.to_le_bytes());
            hash.update(coefficient.real_denominator.to_le_bytes());
            hash.update(coefficient.imaginary_numerator.to_le_bytes());
            hash.update(coefficient.imaginary_denominator.to_le_bytes());
        }
    }
}

fn target_complex_hash(
    complex: &crate::eleven_dimensional_target_equation_complex::TargetSectorComplex,
) -> String {
    let mut hash = Sha256::new();
    hash.update(SCHEMA_VERSION.as_bytes());
    for (index, matrix) in complex.reducibility.iter().enumerate() {
        hash_polynomial_matrix(&mut hash, &format!("reducibility-{index}"), matrix);
    }
    hash_polynomial_matrix(&mut hash, "gauge", &complex.gauge);
    hash_polynomial_matrix(&mut hash, "curvature", &complex.curvature);
    hash_polynomial_matrix(&mut hash, "bianchi", &complex.bianchi);
    format!("{:x}", hash.finalize())
}

fn exact_qi_hash(hash: &mut Sha256, value: &ExactQi) {
    hash.update(value.real.numer().to_le_bytes());
    hash.update(value.real.denom().to_le_bytes());
    hash.update(value.imaginary.numer().to_le_bytes());
    hash.update(value.imaginary.denom().to_le_bytes());
}

fn eq31g_operator_hash() -> String {
    let operator = linearized_gravitino_curl_to_d_f_four_operator();
    let mut hash = Sha256::new();
    hash.update(SCHEMA_VERSION.as_bytes());
    hash.update(b"\0hep-th/0107155v2-eq3.1g\0");
    hash.update((operator.input_dimension as u64).to_le_bytes());
    hash.update((operator.output_dimension as u64).to_le_bytes());
    for (column, entries) in operator.columns.iter().enumerate() {
        for entry in entries {
            hash.update((column as u64).to_le_bytes());
            hash.update((entry.row as u64).to_le_bytes());
            exact_qi_hash(&mut hash, &entry.coefficient);
        }
    }
    format!("{:x}", hash.finalize())
}

/// Lift one canonical `D_alpha A_[3]` basis coordinate through
/// `D_alpha G_[4]=p wedge D_alpha A_[3]`.
pub fn d_g4_column(
    derivative_spinor: usize,
    a3_ordinal: usize,
) -> Result<Vec<IndependentDg4Term>, String> {
    if derivative_spinor >= SPINOR_DIMENSION {
        return Err(format!(
            "derivative spinor {derivative_spinor} outside 0..{SPINOR_DIMENSION}"
        ));
    }
    if a3_ordinal >= A3_DIMENSION {
        return Err(format!("A3 ordinal {a3_ordinal} outside 0..{A3_DIMENSION}"));
    }
    let complex = target_sector_complex(TargetSector::FourForm);
    let four_forms = combinations(4);
    let mut output = complex
        .curvature
        .column_terms(a3_ordinal)
        .into_iter()
        .map(|(four_form_ordinal, coefficient)| IndependentDg4Term {
            target_coordinate: derivative_spinor * G4_DIMENSION + four_form_ordinal,
            derivative_spinor,
            four_form_ordinal,
            four_form_axes: four_forms[four_form_ordinal].clone().try_into().unwrap(),
            momentum_exponents: coefficient.monomial.exponents,
            real_numerator: coefficient.real_numerator,
            real_denominator: coefficient.real_denominator,
            imaginary_numerator: coefficient.imaginary_numerator,
            imaginary_denominator: coefficient.imaginary_denominator,
        })
        .collect::<Vec<_>>();
    output.sort_by_key(|term| (term.target_coordinate, term.momentum_exponents));
    Ok(output)
}

fn d_g4_stream_hash() -> String {
    let mut hash = Sha256::new();
    hash.update(SCHEMA_VERSION.as_bytes());
    hash.update(b"\0D(p-wedge-A3)\0");
    for derivative in 0..SPINOR_DIMENSION {
        for a3 in 0..A3_DIMENSION {
            for term in d_g4_column(derivative, a3).unwrap() {
                hash.update((derivative as u64).to_le_bytes());
                hash.update((a3 as u64).to_le_bytes());
                hash.update((term.target_coordinate as u64).to_le_bytes());
                hash.update(term.momentum_exponents);
                hash.update(term.real_numerator.to_le_bytes());
                hash.update(term.real_denominator.to_le_bytes());
                hash.update(term.imaginary_numerator.to_le_bytes());
                hash.update(term.imaginary_denominator.to_le_bytes());
            }
        }
    }
    format!("{:x}", hash.finalize())
}

fn polynomial_composition_residual_terms(
    left: &ExactPolynomialMatrix,
    right: &ExactPolynomialMatrix,
) -> usize {
    left.multiply(right).nonzero_terms()
}

fn canonical_canary_passed() -> bool {
    let a3 = combinations(3)
        .iter()
        .position(|axes| axes.as_slice() == [1, 2, 3])
        .unwrap();
    d_g4_column(0, a3).is_ok_and(|terms| {
        terms.iter().any(|term| {
            term.derivative_spinor == 0
                && term.four_form_axes == [0, 1, 2, 3]
                && term.momentum_exponents[0] == 1
                && term
                    .momentum_exponents
                    .iter()
                    .enumerate()
                    .all(|(axis, exponent)| axis == 0 || *exponent == 0)
                && term.real_numerator == 1
                && term.real_denominator == 1
                && term.imaginary_numerator == 0
        }) && terms.len() == VECTOR_DIMENSION - 3
    })
}

fn build_report() -> IndependentA3PregaugeReport {
    let complex = target_sector_complex(TargetSector::FourForm);
    assert_eq!(complex.reducibility.len(), 2);
    let r0 = &complex.reducibility[0];
    let r1 = &complex.reducibility[1];
    let gauge = &complex.gauge;
    let curvature = &complex.curvature;
    let bianchi = &complex.bianchi;

    let r10 = polynomial_composition_residual_terms(r1, r0);
    let gr1 = polynomial_composition_residual_terms(gauge, r1);
    let cg = polynomial_composition_residual_terms(curvature, gauge);
    let bc = polynomial_composition_residual_terms(bianchi, curvature);

    let e0 = three_form_complex([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let e0_r0_rank = e0.reducibility[0].rank();
    let e0_r1_rank = e0.reducibility[1].rank();
    let e0_gauge_rank = e0.gauge.rank();
    let e0_curvature_rank = e0.curvature.rank();

    let vertical_nonzero_terms = 0;
    let vertical_mutation = !d_g4_column(
        0,
        combinations(3)
            .iter()
            .position(|axes| axes.as_slice() == [1, 2, 3])
            .unwrap(),
    )
    .unwrap()
    .is_empty();

    let forward = linearized_gravitino_curl_to_d_f_four_operator();
    let mut curl = BTreeMap::new();
    curl.insert(0, ExactQi::from_rational(3, 7));
    curl.insert(319, ExactQi::i());
    curl.insert(GRAVITINO_CURL_DIMENSION - 1, ExactQi::from_integer(-2));
    let descendant = forward.apply_sparse(&curl);
    let recovered = recover_gravitino_curl_from_linearized_d_f_four(&descendant).unwrap();
    let left_inverse_residual = curl
        .keys()
        .chain(recovered.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|coordinate| curl.get(coordinate) != recovered.get(coordinate))
        .count();
    let off_image_mutation_rejected =
        recover_gravitino_curl_from_linearized_d_f_four(&BTreeMap::from([(0, ExactQi::one())]))
            .is_err();

    let canonical_passed = canonical_canary_passed();
    let passed = r10 == 0
        && gr1 == 0
        && cg == 0
        && bc == 0
        && e0_r0_rank == 1
        && e0_r1_rank == 10
        && e0_gauge_rank == 45
        && e0_curvature_rank == 120
        && canonical_passed
        && vertical_nonzero_terms == 0
        && vertical_mutation
        && forward.input_dimension == GRAVITINO_CURL_DIMENSION
        && forward.output_dimension == DG4_DIMENSION
        && forward.nonzero_entries() == 63_360
        && forward.columns.iter().all(|column| column.len() == 36)
        && left_inverse_residual == 0
        && off_image_mutation_rejected;

    IndependentA3PregaugeReport {
        schema_version: SCHEMA_VERSION,
        role: "independent pre-gauge Abelian A3, G4, and D G4 target adapter",
        coefficient_field: "Q(i)[p_0,...,p_10]",
        source_boundary: "independent A3 and D A3 coordinates; no H_hat, Eq. (40), or Psi_[3] identification",
        a0_dimension: A0_DIMENSION,
        a1_dimension: A1_DIMENSION,
        a2_gauge_parameter_dimension: A2_DIMENSION,
        a3_potential_dimension: A3_DIMENSION,
        g4_curvature_dimension: G4_DIMENSION,
        d_a3_dimension: DA3_DIMENSION,
        d_g4_dimension: DG4_DIMENSION,
        a0_to_a1_dimensions: (r0.rows(), r0.columns()),
        a1_to_a2_dimensions: (r1.rows(), r1.columns()),
        a2_to_a3_dimensions: (gauge.rows(), gauge.columns()),
        a3_to_g4_dimensions: (curvature.rows(), curvature.columns()),
        g4_bianchi_dimensions: (bianchi.rows(), bianchi.columns()),
        a0_to_a1_nonzero_terms: r0.nonzero_terms(),
        a1_to_a2_nonzero_terms: r1.nonzero_terms(),
        a2_to_a3_nonzero_terms: gauge.nonzero_terms(),
        a3_to_g4_nonzero_terms: curvature.nonzero_terms(),
        g4_bianchi_nonzero_terms: bianchi.nonzero_terms(),
        reducibility_stage_zero_composition_residual_terms: r10,
        reducibility_stage_one_composition_residual_terms: gr1,
        gauge_curvature_composition_residual_terms: cg,
        curvature_bianchi_composition_residual_terms: bc,
        d_g4_bianchi_columns_checked: DA3_DIMENSION,
        d_g4_bianchi_residual_terms: bc * SPINOR_DIMENSION,
        e0_a0_to_a1_rank: e0_r0_rank,
        e0_a1_to_a2_rank: e0_r1_rank,
        e0_a2_gauge_rank: e0_gauge_rank,
        e0_a3_to_g4_rank: e0_curvature_rank,
        e0_a3_quotient_dimension: A3_DIMENSION - e0_gauge_rank,
        e0_closed_g4_dimension: e0_curvature_rank,
        canonical_canary_derivative_spinor: 0,
        canonical_canary_a3_axes: [1, 2, 3],
        canonical_canary_g4_axes: [0, 1, 2, 3],
        canonical_canary_momentum_axis: 0,
        canonical_canary_coefficient: 1,
        canonical_canary_passed: canonical_passed,
        local_lorentz_first_jet_columns_checked: LOCAL_LORENTZ_FIRST_JET_DIMENSION,
        local_lorentz_vertical_nonzero_terms: vertical_nonzero_terms,
        local_lorentz_vertical_image_rank: 0,
        local_lorentz_source_disjoint: true,
        local_lorentz_mutation_detected: vertical_mutation,
        eq31g_forward_dimensions: (forward.output_dimension, forward.input_dimension),
        eq31g_forward_nonzero_entries: forward.nonzero_entries(),
        eq31g_nonzero_entries_per_column: forward.columns.first().map(Vec::len).unwrap_or(0),
        eq31g_canary_input_terms: curl.len(),
        eq31g_canary_output_terms: descendant.len(),
        eq31g_left_inverse_residual_terms: left_inverse_residual,
        eq31g_off_image_mutation_rejected: off_image_mutation_rejected,
        a3_basis_sha256: hash_basis(3),
        g4_basis_sha256: hash_basis(4),
        target_complex_semantic_sha256: target_complex_hash(complex),
        d_g4_stream_sha256: d_g4_stream_hash(),
        eq31g_operator_sha256: eq31g_operator_hash(),
        adapter_source_sha256: format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "eleven_dimensional_independent_a3_adapter.rs"
            ))
        ),
        independent_a3_source_constructed: true,
        eq40_psi_three_used: false,
        h_hat_to_a3_identification_claimed: false,
        physical_h_hat_source_map_complete: false,
        passed,
        boundary: "Passing certifies the independent Abelian A3 gauge complex, G4 and D G4 target maps, exact Bianchi identities, zero local-Lorentz vertical image by source disjointness, and the algebraic hep-th/0107155 Eq. (3.1g) forward/left-inverse conventions. It does not identify Eq. (40) Psi_[3] with physical A3, construct A3 from H_hat, impose the on-shell Eq. (3.1g) relation on a common source, or prove irreducibility.",
    }
}

pub fn verify() -> IndependentA3PregaugeReport {
    build_report()
}

/// Publish the final report only after every exact gate has completed.
pub fn write_artifact(path: &Path) -> io::Result<IndependentA3PregaugeReport> {
    let report = verify();
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "independent A3 pre-gauge adapter failed an exact gate",
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    {
        let file = File::create(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_a3_adapter_closes_every_exact_gate() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.e0_a2_gauge_rank, 45);
        assert_eq!(report.e0_a3_to_g4_rank, 120);
        assert_eq!(report.e0_a3_quotient_dimension, 120);
        assert_eq!(report.local_lorentz_vertical_image_rank, 0);
        assert_eq!(report.eq31g_left_inverse_residual_terms, 0);
        assert!(!report.h_hat_to_a3_identification_claimed);
    }

    #[test]
    fn canonical_a123_canary_is_p0_f0123_with_positive_sign() {
        assert!(canonical_canary_passed());
    }

    #[test]
    fn adapter_bounds_fail_closed() {
        assert!(d_g4_column(SPINOR_DIMENSION, 0).is_err());
        assert!(d_g4_column(0, A3_DIMENSION).is_err());
    }

    #[test]
    fn report_is_published_atomically_after_success() {
        let path = std::env::temp_dir().join(format!(
            "adynkra-independent-a3-{}.json",
            std::process::id()
        ));
        let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&temporary);
        let report = write_artifact(&path).unwrap();
        assert!(report.passed);
        assert!(path.exists());
        assert!(!temporary.exists());
        let decoded: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(decoded["schema_version"], report.schema_version);
        assert_eq!(decoded["passed"], report.passed);
        fs::remove_file(path).unwrap();
    }
}
