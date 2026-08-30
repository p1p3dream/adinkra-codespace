//! Exact target-resolved stream contract for the 11D gauge-curvature gate.
//!
//! Gates, Hu, and Mak define the gamma-traceless semi-prepotential by Eq.
//! (2.2) of arXiv:2007.05097 and its local gamma-trace symmetry by Eq. (2.3).
//! The spinor-prepotential proposal itself remains a conjecture in the Added
//! Note in Proof of arXiv:2002.08502.  This module therefore certifies only
//! the representation-theoretic composition stream.  It does not claim that
//! any of the six source maps is the physical supergravity gauge law.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalTargetStreamReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_references: Vec<&'static str>,
    pub target_dynkin_label: &'static str,
    pub target_dimension: usize,
    pub ambient_vector_spinor_dimension: usize,
    pub vector_weight_components: usize,
    pub spinor_weight_components: usize,
    pub target_basis_states: usize,
    pub target_distinct_weights: usize,
    pub raw_target_coordinates_touched: usize,
    pub maximum_weight_multiplicity: usize,
    pub target_dual_basis_states: usize,
    pub target_dual_has_only_exact_rational_coefficients: bool,
    pub target_dual_kronecker_pairings_checked: usize,
    pub target_dual_kronecker_pairing_residuals: usize,
    pub target_metric_chevalley_entries_checked: usize,
    pub target_metric_chevalley_invariance_residuals: usize,
    pub zero_momentum_stream_api_available: bool,
    pub first_momentum_stream_api_available: bool,
    pub zero_momentum_bidegree: &'static str,
    pub first_momentum_bidegree: &'static str,
    pub target_vector_basis_convention: &'static str,
    pub target_projection_convention: &'static str,
    pub stream_record_contract: &'static str,
    pub source_operator_columns: usize,
    pub gauge_parameter_channels: usize,
    pub zero_momentum_job_count: usize,
    pub first_momentum_job_count: usize,
    pub full_job_count: usize,
    pub all_jobs_executed_and_hashed: bool,
    pub physical_target_curvature_supplied: bool,
    pub physical_gauge_law_selected: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn verify() -> ElevenDimensionalTargetStreamReport {
    let basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let dual = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let dual_certificate =
        crate::eleven_dimensional_bridge::vector_spinor_target_dual_certificate();
    let distinct_weights = basis
        .iter()
        .map(|state| state.doubled_weight)
        .collect::<BTreeSet<_>>();
    let raw_coordinates = basis
        .iter()
        .flat_map(|state| {
            state
                .raw_terms
                .iter()
                .map(|term| (term.vector_weight_index, term.spinor_weight_index))
        })
        .collect::<BTreeSet<_>>();
    let maximum_weight_multiplicity = distinct_weights
        .iter()
        .map(|weight| {
            basis
                .iter()
                .filter(|state| state.doubled_weight == *weight)
                .count()
        })
        .max()
        .unwrap_or(0);
    let dual_exact = dual.iter().flat_map(|state| &state.raw_terms).all(|term| {
        term.denominator > 0 && term.vector_weight_index < 11 && term.spinor_weight_index < 32
    });
    let operator_columns = crate::eleven_dimensional_level16_couplings::joint_column_specs().len();
    let passed = basis.len() == 320
        && dual.len() == 320
        && distinct_weights.len() == 192
        && raw_coordinates.len() == 352
        && maximum_weight_multiplicity == 5
        && dual_exact
        && dual_certificate.passed
        && operator_columns == 56;

    ElevenDimensionalTargetStreamReport {
        schema_version: "adynkra-11d-target-resolved-composition-stream-v2",
        role: "exact 11 by 32 target-index API contract for the adjoint level-16 gauge-composition stream",
        source_references: vec![
            "arXiv:2007.05097 Eq. (2.2): gamma-traceless conformal-graviton semi-prepotential",
            "arXiv:2007.05097 Eq. (2.3): local gamma-trace symmetry",
            "arXiv:2002.08502 Added Note in Proof, Eqs. (6.2)-(6.3): 320+32 split and conjectured spinor prepotential",
        ],
        target_dynkin_label: "10001",
        target_dimension: 320,
        ambient_vector_spinor_dimension: 352,
        vector_weight_components: 11,
        spinor_weight_components: 32,
        target_basis_states: basis.len(),
        target_distinct_weights: distinct_weights.len(),
        raw_target_coordinates_touched: raw_coordinates.len(),
        maximum_weight_multiplicity,
        target_dual_basis_states: dual.len(),
        target_dual_has_only_exact_rational_coefficients: dual_exact,
        target_dual_kronecker_pairings_checked: dual_certificate.kronecker_pairings_checked,
        target_dual_kronecker_pairing_residuals: dual_certificate.kronecker_pairing_residuals,
        target_metric_chevalley_entries_checked: dual_certificate
            .chevalley_invariance_entries_checked,
        target_metric_chevalley_invariance_residuals: dual_certificate
            .chevalley_invariance_residuals,
        zero_momentum_stream_api_available: true,
        first_momentum_stream_api_available: true,
        zero_momentum_bidegree: "D^17 Lambda",
        first_momentum_bidegree: "p D^15 Lambda",
        target_vector_basis_convention: "B5 vector weight basis (+e_1,-e_1,...,+e_5,-e_5,0); the Cartesian Clifford change of basis is not implicit",
        target_projection_convention: "invariant-metric dual of the deterministic 320-state PBW basis; vector zero-weight norm 2, nonzero vector-weight norms 1, spinor-weight norms 1",
        stream_record_contract: "(target basis ordinal, target vector-weight index, target spinor-weight index, parameter component, optional momentum vector-weight index, exterior mask, exact Gaussian-rational coefficient); consumers sum identical physical keys",
        source_operator_columns: operator_columns,
        gauge_parameter_channels: 6,
        zero_momentum_job_count: 6 * 12,
        first_momentum_job_count: 6 * operator_columns,
        full_job_count: 6 * (12 + operator_columns),
        all_jobs_executed_and_hashed: false,
        physical_target_curvature_supplied: false,
        physical_gauge_law_selected: false,
        passed,
        boundary: "The exact APIs resolve requested source compositions into the full target vector-spinor weight basis and preserve exact rational arithmetic. Smoke tests cover a leading and a correction branch, but the 408 complete jobs are not materialized or content-hashed. The Gates-Hu sources do not supply a convention-fixed physical target gauge map K or curvature F, and they do not select one of the six source parameter channels. F A G_p = 0 remains a separate calculation.",
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
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use std::collections::BTreeMap;

    #[test]
    fn target_stream_contract_is_complete_and_fail_closed() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.target_basis_states, 320);
        assert_eq!(report.raw_target_coordinates_touched, 352);
        assert!(report.zero_momentum_stream_api_available);
        assert!(report.first_momentum_stream_api_available);
        assert_eq!(report.zero_momentum_job_count, 72);
        assert_eq!(report.first_momentum_job_count, 336);
        assert_eq!(report.full_job_count, 408);
        assert_eq!(report.target_dual_kronecker_pairings_checked, 320 * 320);
        assert_eq!(report.target_dual_kronecker_pairing_residuals, 0);
        assert_eq!(
            report.target_metric_chevalley_entries_checked,
            11 * 32 * 5 * 11 * 32
        );
        assert_eq!(report.target_metric_chevalley_invariance_residuals, 0);
        assert!(!report.all_jobs_executed_and_hashed);
        assert!(!report.physical_target_curvature_supplied);
        assert!(!report.physical_gauge_law_selected);
        let expected = serde_json::to_value(&report).unwrap();
        for path in [
            "data/eleven_dimensional_target_stream.json",
            "results/adynkra_11d_target_stream_validation.json",
        ] {
            let actual: serde_json::Value =
                serde_json::from_reader(File::open(path).unwrap()).unwrap();
            assert_eq!(actual, expected, "stale target-stream artifact at {path}");
        }
    }

    #[test]
    fn highest_weight_smoke_stream_has_explicit_target_indices() {
        let highest_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .unwrap()
            .ordinal;
        let highest_dual =
            crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
        assert_eq!(highest_dual[highest_ordinal].raw_terms.len(), 1);
        let mut zero_terms = 0_u64;
        let mut target_zero = BTreeMap::new();
        let (_, parameters, _, _, emitted) = crate::eleven_dimensional_level16_couplings::visit_target_resolved_zero_momentum_gauge_composition_terms(
            0,
            0,
            Some(&[0]),
            Some(&[highest_ordinal]),
            |entry| {
                assert!(entry.target_vector_weight_index < 11);
                assert!(entry.target_spinor_weight_index < 32);
                assert!(entry.momentum_vector_weight_index.is_none());
                let value = target_zero
                    .entry((entry.parameter_component_index, entry.exterior_mask))
                    .or_insert_with(|| {
                        (
                            Ratio::from_integer(BigInt::from(0)),
                            Ratio::from_integer(BigInt::from(0)),
                        )
                    });
                value.0 += entry.real;
                value.1 += entry.imaginary;
                zero_terms += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(parameters.len(), 1);
        assert_eq!(zero_terms, emitted);
        assert!(
            emitted > 0,
            "highest target stream unexpectedly emitted no terms"
        );
        target_zero.retain(|_, (real, imaginary)| {
            *real != Ratio::from_integer(BigInt::from(0))
                || *imaginary != Ratio::from_integer(BigInt::from(0))
        });
        let mut legacy_zero = BTreeMap::new();
        crate::eleven_dimensional_level16_couplings::visit_zero_momentum_gauge_composition_components(
            0,
            0,
            |parameter_component, _, residual| {
                for entry in residual {
                    legacy_zero.insert(
                        (parameter_component, entry.exterior_mask),
                        (
                            Ratio::from_integer(BigInt::from(entry.real)),
                            Ratio::from_integer(BigInt::from(entry.imaginary)),
                        ),
                    );
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(target_zero, legacy_zero);

        let mut first_terms = 0_u64;
        let (_, parameters, _, _, emitted) = crate::eleven_dimensional_level16_couplings::visit_target_resolved_first_momentum_gauge_composition_terms(
            0,
            0,
            Some(&[0]),
            Some(&[highest_ordinal]),
            |entry| {
                assert!(entry.target_vector_weight_index < 11);
                assert!(entry.target_spinor_weight_index < 32);
                assert!(entry.momentum_vector_weight_index.is_some_and(|index| index < 11));
                first_terms += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(parameters.len(), 1);
        assert_eq!(first_terms, emitted);
        assert!(emitted > 0);

        let mut correction_terms = 0_u64;
        let (spec, parameters, _, _, emitted) = crate::eleven_dimensional_level16_couplings::visit_target_resolved_first_momentum_gauge_composition_terms(
            0,
            12,
            Some(&[0]),
            None,
            |entry| {
                assert!(entry.target_vector_weight_index < 11);
                assert!(entry.target_spinor_weight_index < 32);
                assert!(entry.momentum_vector_weight_index.is_some_and(|index| index < 11));
                correction_terms += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(spec.kind, "first-momentum");
        assert_eq!(parameters.len(), 1);
        assert_eq!(correction_terms, emitted);
        assert!(emitted > 0);
    }
}
