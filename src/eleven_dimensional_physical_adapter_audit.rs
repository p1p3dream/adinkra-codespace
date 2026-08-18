//! Independent audit of the B5-to-Cartesian partial physical-curvature API.
//!
//! This module does not alter the curvature implementation. It reconstructs
//! one target-stream record directly in Cartesian coordinates, applies F_X,
//! and compares that result to the trait adapter term by term.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_k_fag_solver::{
    CurvatureVariationKey, ExactGaussian as BigGaussian, MomentumMonomial,
    PhysicalCurvaturePolynomialApi, TargetVariationKey,
};
use crate::eleven_dimensional_physical_curvature::{
    CartesianPolynomialFxApi, ExactQi, FxMomentumMonomial, PolynomialFxDhTerm,
    PolynomialFxOutputKey, apply_polynomial_fx,
};

type SmallGaussian = Complex<Ratio<i64>>;
const PHYSICAL_ARTIFACT: &str =
    include_str!("../results/adynkra_11d_physical_curvature_validation.json");

fn sq(value: i64) -> Ratio<i64> {
    Ratio::from_integer(value)
}

fn bq(value: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(value))
}

fn small_zero() -> SmallGaussian {
    Complex::new(sq(0), sq(0))
}

fn to_big(value: &Ratio<i64>) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(*value.numer()), BigInt::from(*value.denom()))
}

fn multiply_big(left: &BigGaussian, right: &BigGaussian) -> BigGaussian {
    BigGaussian {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn add_big(target: &mut BigGaussian, value: &BigGaussian) {
    target.real += value.real.clone();
    target.imaginary += value.imaginary.clone();
}

fn exact_qi_to_big(value: &ExactQi) -> BigGaussian {
    BigGaussian {
        real: to_big(&value.real),
        imaginary: to_big(&value.imaginary),
    }
}

fn exact_join_roundtrip_residuals() -> (usize, usize, usize) {
    let join = crate::eleven_dimensional_b5_majorana_target_join::exact_target_join();
    let states = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let mut terms = 0;
    let mut residuals = 0;
    for state in &states {
        terms += state.raw_terms.len();
        let mut forward = vec![small_zero(); 11 * 32];
        let mut source = vec![small_zero(); 11 * 32];
        for term in &state.raw_terms {
            let coefficient = Complex::new(Ratio::new(term.numerator, term.denominator), sq(0));
            source[term.vector_weight_index * 32 + term.spinor_weight_index] += coefficient.clone();
            for vector in 0..11 {
                for spinor in 0..32 {
                    forward[vector * 32 + spinor] += coefficient.clone()
                        * join.upper_vector_to_lorentz[vector][term.vector_weight_index].clone()
                        * join.spinor_to_majorana[spinor][term.spinor_weight_index].clone();
                }
            }
        }
        let mut back = vec![small_zero(); 11 * 32];
        for vector_weight in 0..11 {
            for spinor_weight in 0..32 {
                for vector in 0..11 {
                    for spinor in 0..32 {
                        if forward[vector * 32 + spinor] == small_zero() {
                            continue;
                        }
                        back[vector_weight * 32 + spinor_weight] +=
                            join.lorentz_to_upper_vector[vector_weight][vector].clone()
                                * join.majorana_to_spinor[spinor_weight][spinor].clone()
                                * forward[vector * 32 + spinor].clone();
                    }
                }
            }
        }
        residuals += back.iter().zip(&source).filter(|(a, b)| a != b).count();
    }
    (states.len(), terms, residuals)
}

fn direct_cartesian_application(
    input: &TargetVariationKey,
    coefficient: &BigGaussian,
    use_upper_vector: bool,
) -> BTreeMap<CurvatureVariationKey, BigGaussian> {
    let vector_weight = input.target_vector_weight_index.unwrap();
    let target_spinor = input.target_spinor_weight_index.unwrap();
    let join = crate::eleven_dimensional_b5_majorana_target_join::exact_target_join();
    let momentum = FxMomentumMonomial {
        exponents: input.momentum_monomial.exponents,
    };
    let mut terms = Vec::new();
    for derivative_weight in 0..32 {
        if input.spinor_derivative_mask & (1_u32 << derivative_weight) != 0 {
            continue;
        }
        let greater = (input.spinor_derivative_mask >> (derivative_weight + 1)).count_ones();
        let sign = if greater % 2 == 0 { 1 } else { -1 };
        let output_mask = input.spinor_derivative_mask | (1_u32 << derivative_weight);
        for derivative_majorana in 0..32 {
            for h_majorana in 0..32 {
                for output_vector in 0..11 {
                    let vector_factor = if use_upper_vector {
                        join.upper_vector_to_lorentz[output_vector][vector_weight].clone()
                    } else {
                        join.lower_vector_to_lorentz[output_vector][vector_weight].clone()
                    };
                    let factor = join.spinor_to_majorana[derivative_majorana][derivative_weight]
                        .clone()
                        * join.spinor_to_majorana[h_majorana][target_spinor].clone()
                        * vector_factor
                        * Complex::new(sq(sign), sq(0));
                    if factor == small_zero() {
                        continue;
                    }
                    terms.push(PolynomialFxDhTerm {
                        derivative_spinor: derivative_majorana,
                        h_spinor: h_majorana,
                        output_vector,
                        exterior_spinor_mask: output_mask,
                        momentum: momentum.clone(),
                        coefficient: ExactQi {
                            real: factor.re,
                            imaginary: factor.im,
                        },
                    });
                }
            }
        }
    }
    let image = apply_polynomial_fx(&terms);
    let mut output = BTreeMap::new();
    let mut append = |sector: &str, key: PolynomialFxOutputKey, value: ExactQi| {
        let value = multiply_big(coefficient, &exact_qi_to_big(&value));
        let output_key = CurvatureVariationKey {
            parameter_component: input.parameter_component,
            output_sector: sector.to_string(),
            output_coordinate: key.quotient_coordinate,
            spinor_derivative_mask: key.exterior_spinor_mask,
            spinor_derivative_order: key.exterior_spinor_mask.count_ones() as usize,
            momentum_monomial: MomentumMonomial {
                exponents: key.momentum.exponents,
            },
        };
        let entry = output.entry(output_key).or_insert_with(BigGaussian::zero);
        add_big(entry, &value);
    };
    for (key, value) in image.x_two_11000 {
        append("X2_11000", key, value);
    }
    for (key, value) in image.x_five_10002 {
        append("X5_10002", key, value);
    }
    output.retain(|_, value| !value.is_zero());
    output
}

fn one_record_control() -> (usize, usize, usize, usize, usize, bool, bool, bool, bool) {
    let state = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
        .into_iter()
        .next()
        .unwrap();
    let term = state.raw_terms.first().unwrap();
    let mut exponents = [0_u16; 11];
    for (axis, exponent) in exponents.iter_mut().enumerate() {
        *exponent = axis as u16 + 1;
    }
    let input = TargetVariationKey {
        parameter_component: 17,
        target_coordinate: state.ordinal,
        target_vector_weight_index: Some(term.vector_weight_index),
        target_spinor_weight_index: Some(term.spinor_weight_index),
        // Exactly one missing derivative makes the direct control small while
        // still checking the exterior insertion and its sign.
        spinor_derivative_mask: u32::MAX ^ 1,
        spinor_derivative_order: 31,
        momentum_monomial: MomentumMonomial { exponents },
    };
    let coefficient = BigGaussian {
        real: bq(2),
        imaginary: bq(3),
    };
    let adapter = CartesianPolynomialFxApi
        .apply_term(&input, &coefficient)
        .unwrap()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut changed_ordinal = input.clone();
    changed_ordinal.target_coordinate = 319;
    let raw_coordinates_override_ordinal = CartesianPolynomialFxApi
        .apply_term(&changed_ordinal, &coefficient)
        .unwrap()
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        == adapter;
    let direct = direct_cartesian_application(&input, &coefficient, true);
    let wrong_variance = direct_cartesian_application(&input, &coefficient, false);
    let residuals = adapter
        .iter()
        .filter(|(key, value)| direct.get(*key) != Some(*value))
        .count()
        + direct
            .iter()
            .filter(|(key, value)| adapter.get(*key) != Some(*value))
            .count();
    let x2 = adapter
        .keys()
        .filter(|key| key.output_sector == "X2_11000")
        .count();
    let x5 = adapter
        .keys()
        .filter(|key| key.output_sector == "X5_10002")
        .count();
    let wrong_variance_detected = wrong_variance != adapter;
    let masks_preserved = adapter
        .keys()
        .all(|key| key.spinor_derivative_mask == u32::MAX && key.spinor_derivative_order == 32);
    let momenta_preserved = adapter
        .keys()
        .all(|key| key.momentum_monomial.exponents == exponents);
    (
        adapter.len(),
        direct.len(),
        residuals,
        x2,
        x5,
        masks_preserved,
        momenta_preserved,
        wrong_variance_detected,
        raw_coordinates_override_ordinal,
    )
}

fn ambient_tensor_roundtrip_residuals() -> usize {
    let join = crate::eleven_dimensional_b5_majorana_target_join::exact_target_join();
    let mut vector_identity = vec![vec![small_zero(); 11]; 11];
    let mut spinor_identity = vec![vec![small_zero(); 32]; 32];
    for left in 0..11 {
        for right in 0..11 {
            vector_identity[left][right] = (0..11)
                .map(|middle| {
                    join.lorentz_to_upper_vector[left][middle].clone()
                        * join.upper_vector_to_lorentz[middle][right].clone()
                })
                .sum();
        }
    }
    for left in 0..32 {
        for right in 0..32 {
            spinor_identity[left][right] = (0..32)
                .map(|middle| {
                    join.majorana_to_spinor[left][middle].clone()
                        * join.spinor_to_majorana[middle][right].clone()
                })
                .sum();
        }
    }
    let mut residuals = 0;
    for output_vector in 0..11 {
        for output_spinor in 0..32 {
            for input_vector in 0..11 {
                for input_spinor in 0..32 {
                    let actual = vector_identity[output_vector][input_vector].clone()
                        * spinor_identity[output_spinor][input_spinor].clone();
                    let expected = if output_vector == input_vector && output_spinor == input_spinor
                    {
                        Complex::new(sq(1), sq(0))
                    } else {
                        small_zero()
                    };
                    residuals += usize::from(actual != expected);
                }
            }
        }
    }
    residuals
}

fn legacy_and_range_rejection() -> (bool, bool, bool) {
    let legacy = TargetVariationKey {
        parameter_component: 0,
        target_coordinate: 319,
        target_vector_weight_index: None,
        target_spinor_weight_index: None,
        spinor_derivative_mask: u32::MAX ^ 1,
        spinor_derivative_order: 31,
        momentum_monomial: MomentumMonomial::constant(),
    };
    let legacy_rejected = CartesianPolynomialFxApi
        .apply_term(&legacy, &BigGaussian::one())
        .is_err();
    let mut vector_out_of_range = legacy.clone();
    vector_out_of_range.target_vector_weight_index = Some(11);
    vector_out_of_range.target_spinor_weight_index = Some(0);
    let vector_range_rejected = CartesianPolynomialFxApi
        .apply_term(&vector_out_of_range, &BigGaussian::one())
        .is_err();
    let mut spinor_out_of_range = legacy;
    spinor_out_of_range.target_vector_weight_index = Some(0);
    spinor_out_of_range.target_spinor_weight_index = Some(32);
    let spinor_range_rejected = CartesianPolynomialFxApi
        .apply_term(&spinor_out_of_range, &BigGaussian::one())
        .is_err();
    (
        legacy_rejected,
        vector_range_rejected,
        spinor_range_rejected,
    )
}

#[derive(Clone, Debug, Serialize)]
pub struct PhysicalAdapterAuditReport {
    pub schema_version: &'static str,
    pub physical_schema_audited: String,
    pub physical_artifact_sha256: String,
    pub role: &'static str,
    pub raw_ambient_coordinates_checked: usize,
    pub raw_ambient_roundtrip_entries_checked: usize,
    pub raw_ambient_roundtrip_residual_entries: usize,
    pub target_basis_states_checked: usize,
    pub target_basis_terms_checked: usize,
    pub target_basis_roundtrip_residual_entries: usize,
    pub upper_vector_variance_used: bool,
    pub wrong_lower_vector_variance_detected: bool,
    pub exterior_mask_preserved: bool,
    pub all_eleven_momentum_exponents_preserved: bool,
    pub x2_target_dimension: usize,
    pub x5_target_dimension: usize,
    pub x2_embedded_ambient_dimension: usize,
    pub x5_embedded_ambient_dimension: usize,
    pub adapter_x2_embedded_coordinates_in_ambient_range: bool,
    pub adapter_x5_embedded_coordinates_in_ambient_range: bool,
    pub raw_coordinates_override_legacy_ordinal: bool,
    pub legacy_ordinal_only_key_rejected: bool,
    pub vector_index_out_of_range_rejected: bool,
    pub spinor_index_out_of_range_rejected: bool,
    pub end_to_end_target_records_checked: usize,
    pub end_to_end_adapter_terms: usize,
    pub end_to_end_direct_terms: usize,
    pub end_to_end_x2_terms: usize,
    pub end_to_end_x5_terms: usize,
    pub end_to_end_residual_terms: usize,
    pub direct_cartesian_agrees_exactly: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn verify() -> PhysicalAdapterAuditReport {
    let physical: serde_json::Value =
        serde_json::from_str(PHYSICAL_ARTIFACT).expect("checked physical artifact JSON");
    let physical_schema = physical["schema_version"].as_str().unwrap();
    let physical_sha256 = format!("{:x}", Sha256::digest(PHYSICAL_ARTIFACT.as_bytes()));
    let x2_dimension = physical["x2_hook_dimension"].as_u64().unwrap() as usize;
    let x5_dimension = physical["x5_hook_dimension"].as_u64().unwrap() as usize;
    let x2_ambient = physical["x2_ambient_dimension"].as_u64().unwrap() as usize;
    let x5_ambient = physical["x5_ambient_dimension"].as_u64().unwrap() as usize;
    let ambient_residuals = ambient_tensor_roundtrip_residuals();
    let (target_states, target_terms, target_residuals) = exact_join_roundtrip_residuals();
    let (
        adapter_terms,
        direct_terms,
        direct_residuals,
        x2_terms,
        x5_terms,
        masks,
        momenta,
        variance,
        ordinal_override,
    ) = one_record_control();
    let (legacy, vector_range, spinor_range) = legacy_and_range_rejection();
    let input_state = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
        .into_iter()
        .next()
        .unwrap();
    let term = input_state.raw_terms.first().unwrap();
    let input = TargetVariationKey {
        parameter_component: 17,
        target_coordinate: input_state.ordinal,
        target_vector_weight_index: Some(term.vector_weight_index),
        target_spinor_weight_index: Some(term.spinor_weight_index),
        spinor_derivative_mask: u32::MAX ^ 1,
        spinor_derivative_order: 31,
        momentum_monomial: MomentumMonomial {
            exponents: std::array::from_fn(|axis| axis as u16 + 1),
        },
    };
    let output = CartesianPolynomialFxApi
        .apply_term(
            &input,
            &BigGaussian {
                real: bq(2),
                imaginary: bq(3),
            },
        )
        .unwrap();
    let x2_in_range = output
        .iter()
        .filter(|(key, _)| key.output_sector == "X2_11000")
        .all(|(key, _)| key.output_coordinate < x2_ambient);
    let x5_in_range = output
        .iter()
        .filter(|(key, _)| key.output_sector == "X5_10002")
        .all(|(key, _)| key.output_coordinate < x5_ambient);
    let passed = physical_schema == "adynkra-11d-physical-curvature-operator-v10"
        && ambient_residuals == 0
        && target_states == 320
        && target_residuals == 0
        && masks
        && momenta
        && variance
        && ordinal_override
        && x2_dimension == 429
        && x5_dimension == 4_290
        && x2_ambient == 605
        && x5_ambient == 5_082
        && x2_terms > 0
        && x5_terms > 0
        && x2_in_range
        && x5_in_range
        && legacy
        && vector_range
        && spinor_range
        && direct_residuals == 0;
    PhysicalAdapterAuditReport {
        physical_schema_audited: physical_schema.to_string(),
        physical_artifact_sha256: physical_sha256,
        schema_version: "adynkra.11d.physical-b5-adapter-audit.v1",
        role: "independent raw-coordinate, variance, bookkeeping, quotient-dimension, and end-to-end F_X adapter audit",
        raw_ambient_coordinates_checked: 11 * 32,
        raw_ambient_roundtrip_entries_checked: (11 * 32) * (11 * 32),
        raw_ambient_roundtrip_residual_entries: ambient_residuals,
        target_basis_states_checked: target_states,
        target_basis_terms_checked: target_terms,
        target_basis_roundtrip_residual_entries: target_residuals,
        upper_vector_variance_used: true,
        wrong_lower_vector_variance_detected: variance,
        exterior_mask_preserved: masks,
        all_eleven_momentum_exponents_preserved: momenta,
        x2_target_dimension: x2_dimension,
        x5_target_dimension: x5_dimension,
        x2_embedded_ambient_dimension: x2_ambient,
        x5_embedded_ambient_dimension: x5_ambient,
        adapter_x2_embedded_coordinates_in_ambient_range: x2_in_range,
        adapter_x5_embedded_coordinates_in_ambient_range: x5_in_range,
        raw_coordinates_override_legacy_ordinal: ordinal_override,
        legacy_ordinal_only_key_rejected: legacy,
        vector_index_out_of_range_rejected: vector_range,
        spinor_index_out_of_range_rejected: spinor_range,
        end_to_end_target_records_checked: 1,
        end_to_end_adapter_terms: adapter_terms,
        end_to_end_direct_terms: direct_terms,
        end_to_end_x2_terms: x2_terms,
        end_to_end_x5_terms: x5_terms,
        end_to_end_residual_terms: direct_residuals,
        direct_cartesian_agrees_exactly: direct_residuals == 0,
        passed,
        boundary: "This audits only the exact B5-to-Cartesian adapter for the partial conventional-quotient F_X=(X_[2],X_[5]) map. It does not complete J or W, select physical K, prove F A G_p, or establish off-shell closure.",
    }
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
    fn current_physical_b5_adapter_matches_direct_cartesian_fx() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.raw_ambient_roundtrip_residual_entries, 0);
        assert_eq!(report.target_basis_states_checked, 320);
        assert_eq!(report.target_basis_roundtrip_residual_entries, 0);
        assert_eq!(report.end_to_end_residual_terms, 0);
    }

    #[test]
    fn adapter_rejects_lossy_and_out_of_range_keys() {
        let report = verify();
        assert!(report.legacy_ordinal_only_key_rejected);
        assert!(report.vector_index_out_of_range_rejected);
        assert!(report.spinor_index_out_of_range_rejected);
        assert!(report.wrong_lower_vector_variance_detected);
    }

    #[test]
    #[ignore = "artifact writer"]
    fn write_checked_artifact() {
        write_artifact(Path::new(
            "results/adynkra_11d_physical_b5_adapter_audit.json",
        ))
        .unwrap();
    }
}
