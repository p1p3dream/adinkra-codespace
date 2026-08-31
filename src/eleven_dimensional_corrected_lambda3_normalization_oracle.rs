//! Minimal exact comparison of the corrected Gamma4-trace Lambda3 descendant
//! with the independent teleparallel gravitino-curl descendant.

use std::collections::{BTreeMap, BTreeSet};

use num_rational::Ratio;
use serde::Serialize;

use crate::eleven_dimensional_complete_f::visit_gauge_fixed_linearized_gravitino_curl;
use crate::eleven_dimensional_gamma24_source_variance::{
    H_HAT_DIMENSION, corrected_gamma_two_exterior_operator,
};
use crate::eleven_dimensional_h_hat_jet::{
    LinearizedFrameSuperfields, canonical_gamma_traceless_frame_basis,
};
use crate::eleven_dimensional_physical_curvature::{
    D_F_FOUR_FORM_DIMENSION, ExactQi, GRAVITINO_CURL_DIMENSION, SPINOR_DIMENSION, VECTOR_DIMENSION,
    W_FOUR_FORM_DIMENSION, cached_linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial, left_multiply_d,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

const THREE_FORM_DIMENSION: usize = 165;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct CorrectedDescendantRowKey {
    pub output_coordinate: usize,
    pub exterior_spinor_mask: u32,
    pub momentum_exponents: [u16; VECTOR_DIMENSION],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PublicExactQi {
    pub real: [i64; 2],
    pub imaginary: [i64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedLambdaThreeMismatch {
    pub row: CorrectedDescendantRowKey,
    pub candidate: PublicExactQi,
    pub teleparallel: PublicExactQi,
    pub residual: PublicExactQi,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedLambdaThreeScalePivot {
    pub row: CorrectedDescendantRowKey,
    pub candidate: PublicExactQi,
    pub teleparallel: PublicExactQi,
    pub teleparallel_over_candidate: PublicExactQi,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedLambdaThreeColumnReport {
    pub schema_version: &'static str,
    pub candidate_normalization: &'static str,
    pub target_basis_join: &'static str,
    pub canonical_row_key: &'static str,
    pub source_ordinal: usize,
    pub candidate_rows: usize,
    pub teleparallel_rows: usize,
    pub common_rows: usize,
    pub candidate_only_rows: usize,
    pub teleparallel_only_rows: usize,
    pub scale_pivot: Option<CorrectedLambdaThreeScalePivot>,
    pub exact_scale: Option<PublicExactQi>,
    pub residual_rows: usize,
    pub first_mismatch: Option<CorrectedLambdaThreeMismatch>,
    pub passed: bool,
}

fn public(value: &ExactQi) -> PublicExactQi {
    PublicExactQi {
        real: [*value.real.numer(), *value.real.denom()],
        imaginary: [*value.imaginary.numer(), *value.imaginary.denom()],
    }
}

fn multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn divide(left: &ExactQi, right: &ExactQi) -> Option<ExactQi> {
    if right.is_zero() {
        return None;
    }
    let denominator =
        right.real.clone() * right.real.clone() + right.imaginary.clone() * right.imaginary.clone();
    Some(ExactQi {
        real: (left.real.clone() * right.real.clone()
            + left.imaginary.clone() * right.imaginary.clone())
            / denominator.clone(),
        imaginary: (left.imaginary.clone() * right.real.clone()
            - left.real.clone() * right.imaginary.clone())
            / denominator,
    })
}

fn coefficient(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    }
}

fn add_value(
    output: &mut BTreeMap<CorrectedDescendantRowKey, ExactQi>,
    key: CorrectedDescendantRowKey,
    value: ExactQi,
) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(key.clone()).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&key);
    }
}

fn multiply_momentum(
    monomial: &OrderedSuperderivativeMonomial,
    factor: &ExactPolynomialCoefficient,
) -> Result<[u16; VECTOR_DIMENSION], String> {
    let mut output = monomial.momentum.exponents;
    for (axis, exponent) in factor.monomial.exponents.into_iter().enumerate() {
        output[axis] = output[axis]
            .checked_add(u16::from(exponent))
            .ok_or_else(|| format!("momentum overflow on axis {axis}"))?;
    }
    Ok(output)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1, |value, index| value * (n - index) / (index + 1))
}

fn target_three_form_ordinal(mask: u16) -> usize {
    assert_eq!(mask.count_ones(), 3);
    let indices = (0..VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    let mut ordinal = 0;
    let mut next = 0;
    for (position, value) in indices.into_iter().enumerate() {
        for candidate in next..value {
            ordinal += binomial(VECTOR_DIMENSION - candidate - 1, 3 - position - 1);
        }
        next = value + 1;
    }
    ordinal
}

fn numeric_three_form_masks() -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 3)
        .collect()
}

fn basis_input(ordinal: usize) -> LinearizedFrameSuperfields {
    let vector = &canonical_gamma_traceless_frame_basis()[ordinal];
    LinearizedFrameSuperfields {
        h: vector
            .iter()
            .map(|(&coordinate, value)| {
                (coordinate, CanonicalSuperPolynomial::scalar(value.clone()))
            })
            .collect(),
        scale: CanonicalSuperPolynomial::default(),
        lorentz_two_form: BTreeMap::new(),
    }
}

fn corrected_candidate(
    source_ordinal: usize,
) -> Result<BTreeMap<CorrectedDescendantRowKey, ExactQi>, String> {
    let gamma_two = corrected_gamma_two_exterior_operator();
    assert_eq!(
        gamma_two.input_dimension,
        SPINOR_DIMENSION * H_HAT_DIMENSION
    );
    assert_eq!(gamma_two.output_dimension, THREE_FORM_DIMENSION);
    let masks = numeric_three_form_masks();
    let curvature = &target_sector_complex(TargetSector::FourForm).curvature;
    let scalar = CanonicalSuperPolynomial::scalar(ExactQi::one());
    let inner = (0..SPINOR_DIMENSION)
        .map(|spinor| left_multiply_d(spinor, &scalar))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = BTreeMap::new();
    for outer_spinor in 0..SPINOR_DIMENSION {
        for inner_spinor in 0..SPINOR_DIMENSION {
            let pbw = left_multiply_d(outer_spinor, &inner[inner_spinor])?;
            let source_column = inner_spinor * H_HAT_DIMENSION + source_ordinal;
            for gamma_entry in &gamma_two.columns[source_column] {
                let potential = target_three_form_ordinal(masks[gamma_entry.row]);
                for (four_form, curvature_term) in curvature.column_terms(potential) {
                    let psi_coefficient = gamma_entry.coefficient.scaled(&Ratio::new(1, 16));
                    let map_coefficient = multiply(&psi_coefficient, &coefficient(&curvature_term));
                    for (monomial, pbw_coefficient) in &pbw.terms {
                        add_value(
                            &mut output,
                            CorrectedDescendantRowKey {
                                output_coordinate: outer_spinor * W_FOUR_FORM_DIMENSION + four_form,
                                exterior_spinor_mask: monomial.exterior_spinor_mask,
                                momentum_exponents: multiply_momentum(monomial, &curvature_term)?,
                            },
                            multiply(pbw_coefficient, &map_coefficient),
                        );
                    }
                }
            }
        }
    }
    Ok(output)
}

fn teleparallel(
    source_ordinal: usize,
) -> Result<BTreeMap<CorrectedDescendantRowKey, ExactQi>, String> {
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    assert_eq!(operator.input_dimension, GRAVITINO_CURL_DIMENSION);
    assert_eq!(operator.output_dimension, D_F_FOUR_FORM_DIMENSION);
    let mut slices = BTreeMap::<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>::new();
    visit_gauge_fixed_linearized_gravitino_curl(&basis_input(source_ordinal), |entry| {
        let slice = slices.entry(entry.monomial).or_default();
        let value = slice.entry(entry.component).or_insert_with(ExactQi::zero);
        value.add_assign(&entry.coefficient);
        if value.is_zero() {
            slice.remove(&entry.component);
        }
        Ok(())
    })?;
    let mut output = BTreeMap::new();
    for (monomial, curl) in slices {
        for (coordinate, value) in operator.apply_sparse(&curl) {
            add_value(
                &mut output,
                CorrectedDescendantRowKey {
                    output_coordinate: coordinate,
                    exterior_spinor_mask: monomial.exterior_spinor_mask,
                    momentum_exponents: monomial.momentum.exponents,
                },
                value,
            );
        }
    }
    Ok(output)
}

pub fn compare_corrected_lambda_three_column(
    source_ordinal: usize,
) -> Result<CorrectedLambdaThreeColumnReport, String> {
    if source_ordinal >= H_HAT_DIMENSION {
        return Err(format!(
            "source ordinal {source_ordinal} is outside 0..{H_HAT_DIMENSION}"
        ));
    }
    let candidate = corrected_candidate(source_ordinal)?;
    let target = teleparallel(source_ordinal)?;
    let keys = candidate
        .keys()
        .chain(target.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let common_rows = keys
        .iter()
        .filter(|key| candidate.contains_key(*key) && target.contains_key(*key))
        .count();
    let pivot = keys.iter().find_map(|key| {
        let candidate_value = candidate.get(key)?;
        let target_value = target.get(key)?;
        let scale = divide(target_value, candidate_value)?;
        Some((
            key.clone(),
            candidate_value.clone(),
            target_value.clone(),
            scale,
        ))
    });
    let scale = pivot.as_ref().map(|(_, _, _, scale)| scale.clone());
    let mut residual_rows = 0;
    let mut first_mismatch = None;
    for key in &keys {
        let candidate_value = candidate.get(key).cloned().unwrap_or_else(ExactQi::zero);
        let target_value = target.get(key).cloned().unwrap_or_else(ExactQi::zero);
        let mut residual = scale
            .as_ref()
            .map(|value| multiply(value, &candidate_value))
            .unwrap_or_else(ExactQi::zero);
        residual.add_assign(&target_value.scaled(&Ratio::from_integer(-1)));
        if !residual.is_zero() {
            residual_rows += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(CorrectedLambdaThreeMismatch {
                    row: key.clone(),
                    candidate: public(&candidate_value),
                    teleparallel: public(&target_value),
                    residual: public(&residual),
                });
            }
        }
    }
    Ok(CorrectedLambdaThreeColumnReport {
        schema_version: "adynkra-11d-corrected-lambda3-normalization-oracle-v1",
        candidate_normalization: "Psi_[3] = (1/16) corrected Gamma2 exterior = -(1/48) corrected Gamma4 trace",
        target_basis_join: "increasing numeric three-form mask row -> lexicographic target A3 ordinal",
        canonical_row_key: "(source ordinal, explicit outer derivative spinor * 330 + lexicographic G4 ordinal, ordered exterior-D mask, p_0..p_10 exponents)",
        source_ordinal,
        candidate_rows: candidate.len(),
        teleparallel_rows: target.len(),
        common_rows,
        candidate_only_rows: candidate
            .keys()
            .filter(|key| !target.contains_key(*key))
            .count(),
        teleparallel_only_rows: target
            .keys()
            .filter(|key| !candidate.contains_key(*key))
            .count(),
        scale_pivot: pivot
            .as_ref()
            .map(|(row, candidate_value, target_value, pivot_scale)| {
                CorrectedLambdaThreeScalePivot {
                    row: row.clone(),
                    candidate: public(candidate_value),
                    teleparallel: public(target_value),
                    teleparallel_over_candidate: public(pivot_scale),
                }
            }),
        exact_scale: scale.as_ref().map(public),
        residual_rows,
        first_mismatch,
        passed: scale.is_some() && residual_rows == 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrected_lambda_three_column_zero_has_an_exact_decision() {
        let report = compare_corrected_lambda_three_column(0).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.candidate_rows, 18_972);
        assert_eq!(report.teleparallel_rows, 541_740);
        assert_eq!(report.common_rows, 13_596);
        assert_eq!(report.candidate_only_rows, 5_376);
        assert_eq!(report.teleparallel_only_rows, 528_144);
        assert_eq!(
            report.exact_scale,
            Some(PublicExactQi {
                real: [18_049, 560],
                imaginary: [-5, 32],
            })
        );
        let pivot = report.scale_pivot.as_ref().unwrap();
        assert_eq!(pivot.row.output_coordinate, 0);
        assert_eq!(pivot.row.exterior_spinor_mask, 3);
        assert_eq!(pivot.row.momentum_exponents[1], 1);
        assert_eq!(report.residual_rows, 546_892);
        let mismatch = report.first_mismatch.as_ref().unwrap();
        assert_eq!(mismatch.row.output_coordinate, 0);
        assert_eq!(mismatch.row.exterior_spinor_mask, 3);
        assert_eq!(mismatch.row.momentum_exponents[0], 1);
        assert_eq!(mismatch.candidate.real, [1, 48]);
        assert_eq!(mismatch.teleparallel.real, [-18_049, 26_880]);
        assert!(!report.passed);
    }
}
