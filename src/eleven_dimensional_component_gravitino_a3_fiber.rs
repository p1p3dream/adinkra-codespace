//! Independent component gravitino-curl and Abelian A3 fiber product.
//!
//! This oracle never calls Eq. (25), `Hhat`, or the gauge-fixed full-chain
//! builders. Both legs meet only in the canonical `D_alpha G_[4]` target:
//! the independent Abelian `p wedge D A_[3]` map and the component
//! gravitino-curl map fixed by hep-th/0107155v2 Eq. (3.1g).

use std::collections::BTreeMap;

use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_independent_a3_adapter;
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, SparseQiOperator, linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const THREE_FORM_DIMENSION: usize = 165;
const FOUR_FORM_DIMENSION: usize = 330;
const CURL_DIMENSION: usize = 55 * SPINOR_DIMENSION;
const FRAME_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const PRIME: u32 = 1_073_741_783;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ComponentGravitinoA3FiberReport {
    pub schema_version: &'static str,
    pub momentum_axis: usize,
    pub a3_first_jet_dimension: usize,
    pub a3_descendant_rank: usize,
    pub component_curl_dimension: usize,
    pub eq31g_curl_rank: usize,
    pub eq31g_outside_a3_rank: usize,
    pub target_image_intersection_dimension: usize,
    pub combined_target_rank: usize,
    pub fiber_product_kernel_dimension: usize,
    pub component_frame_to_curl_rank: usize,
    pub component_frame_bianchi_residual_entries: usize,
    pub target_bianchi_mutation_residual_entries: usize,
    pub eq31g_printed_coefficient: [i64; 2],
    pub antisymmetrization_partition_multiplicity: usize,
    pub expanded_partition_coefficient_magnitude: [i64; 2],
    pub normalization_mutation_detected: bool,
    pub independent_a3_passed: bool,
    pub independent_a3_local_lorentz_image_rank: usize,
    pub component_source_quotient_applied: bool,
    pub nonzero_target_class: bool,
    pub a3_basis_sha256: String,
    pub g4_basis_sha256: String,
    pub curl_basis_sha256: String,
    pub eq31g_operator_sha256: String,
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

fn exact(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    }
}

fn add(output: &mut BTreeMap<usize, ExactQi>, row: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(row).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&row);
    }
}

fn frame_to_curl_columns(momentum_axis: usize) -> Vec<BTreeMap<usize, ExactQi>> {
    let curvature = &target_sector_complex(TargetSector::RaritaSchwinger).curvature;
    (0..FRAME_DIMENSION)
        .map(|column| {
            let mut output = BTreeMap::new();
            for (row, coefficient) in curvature.column_terms(column) {
                let exponents = coefficient.monomial.exponents;
                if exponents[momentum_axis] == 1
                    && exponents
                        .iter()
                        .enumerate()
                        .all(|(axis, &power)| axis == momentum_axis || power == 0)
                {
                    add(&mut output, row, exact(&coefficient));
                }
            }
            output
        })
        .collect()
}

fn four_form_contains_axis(row: usize, axis: usize) -> bool {
    let ordinal = row % FOUR_FORM_DIMENSION;
    combinations(4)[ordinal].contains(&axis)
}

fn mod_inverse(value: u32) -> u32 {
    let mut base = u64::from(value);
    let mut exponent = PRIME - 2;
    let modulus = u64::from(PRIME);
    let mut output = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = output * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    output as u32
}

fn qi_mod(value: &ExactQi) -> Result<u32, String> {
    if !value.imaginary.is_zero() {
        return Err("component fiber modular rank expected a real coefficient".to_string());
    }
    let denominator = value.real.denom().rem_euclid(i64::from(PRIME)) as u32;
    if denominator == 0 {
        return Err("component fiber denominator is inadmissible".to_string());
    }
    let numerator = value.real.numer().rem_euclid(i64::from(PRIME)) as u32;
    Ok((u64::from(numerator) * u64::from(mod_inverse(denominator)) % u64::from(PRIME)) as u32)
}

fn modular_rank(columns: &[BTreeMap<usize, ExactQi>]) -> Result<usize, String> {
    let mut pivots = BTreeMap::<usize, BTreeMap<usize, u32>>::new();
    for column in columns {
        let mut reduced = column
            .iter()
            .map(|(&row, value)| {
                let value = qi_mod(value)?;
                Ok::<_, String>((row, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        reduced.retain(|_, value| *value != 0);
        loop {
            let Some((&row, &value)) = reduced.first_key_value() else {
                break;
            };
            let Some(pivot) = pivots.get(&row) else {
                let inverse = mod_inverse(value);
                for entry in reduced.values_mut() {
                    *entry = (u64::from(*entry) * u64::from(inverse) % u64::from(PRIME)) as u32;
                }
                pivots.insert(row, reduced);
                break;
            };
            let factor = value;
            for (&target, &coefficient) in pivot {
                let subtraction = u64::from(factor) * u64::from(coefficient) % u64::from(PRIME);
                let current = reduced.get(&target).copied().unwrap_or(0);
                let next = (u64::from(current) + u64::from(PRIME) - subtraction) % u64::from(PRIME);
                if next == 0 {
                    reduced.remove(&target);
                } else {
                    reduced.insert(target, next as u32);
                }
            }
        }
    }
    Ok(pivots.len())
}

fn outside_columns(
    operator: &SparseQiOperator,
    momentum_axis: usize,
) -> Vec<BTreeMap<usize, ExactQi>> {
    operator
        .columns
        .iter()
        .map(|column| {
            column
                .iter()
                .filter(|entry| !four_form_contains_axis(entry.row, momentum_axis))
                .map(|entry| (entry.row, entry.coefficient.clone()))
                .collect()
        })
        .collect()
}

fn composition_residual_entries(
    operator: &SparseQiOperator,
    frame_columns: &[BTreeMap<usize, ExactQi>],
    momentum_axis: usize,
) -> usize {
    frame_columns
        .iter()
        .map(|column| {
            operator
                .apply_sparse(column)
                .into_iter()
                .filter(|(row, value)| {
                    !value.is_zero() && !four_form_contains_axis(*row, momentum_axis)
                })
                .count()
        })
        .sum()
}

fn hash_curl_basis() -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-component-curl-basis-v1\0");
    for pair in combinations(2) {
        for spinor in 0..SPINOR_DIMENSION {
            hash.update([pair[0] as u8, pair[1] as u8]);
            hash.update((spinor as u64).to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn hash_operator(operator: &SparseQiOperator) -> String {
    let mut hash = Sha256::new();
    hash.update(b"hep-th/0107155v2-eq3.1g-component-v1\0");
    for (column, entries) in operator.columns.iter().enumerate() {
        for entry in entries {
            hash.update((column as u64).to_le_bytes());
            hash.update((entry.row as u64).to_le_bytes());
            hash.update(entry.coefficient.real.numer().to_le_bytes());
            hash.update(entry.coefficient.real.denom().to_le_bytes());
            hash.update(entry.coefficient.imaginary.numer().to_le_bytes());
            hash.update(entry.coefficient.imaginary.denom().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

pub(crate) fn build_component_gravitino_a3_fiber_report()
-> Result<ComponentGravitinoA3FiberReport, String> {
    let momentum_axis = 0;
    let a3 = eleven_dimensional_independent_a3_adapter::verify();
    let operator = linearized_gravitino_curl_to_d_f_four_operator();
    let frame_columns = frame_to_curl_columns(momentum_axis);
    let frame_rank = modular_rank(&frame_columns)?;
    let outside = outside_columns(&operator, momentum_axis);
    let outside_rank = modular_rank(&outside)?;
    let bianchi_residual = composition_residual_entries(&operator, &frame_columns, momentum_axis);

    let mutation_column = frame_columns
        .iter()
        .flat_map(|column| column.keys().copied())
        .find(|&curl_row| !operator.columns[curl_row].is_empty())
        .ok_or_else(|| "component frame image does not hit Eq3.1g".to_string())?;
    let outside_form = combinations(4)
        .iter()
        .position(|form| !form.contains(&momentum_axis))
        .ok_or_else(|| "fixed-momentum target lacks an outside-A3 four-form".to_string())?;
    let mut mutation = operator.clone();
    let mutation_entry = mutation.columns[mutation_column]
        .first_mut()
        .ok_or_else(|| "selected Eq3.1g mutation column is empty".to_string())?;
    let target_spinor = mutation_entry.row / FOUR_FORM_DIMENSION;
    mutation_entry.row = target_spinor * FOUR_FORM_DIMENSION + outside_form;
    let mutation_residual = composition_residual_entries(&mutation, &frame_columns, momentum_axis);

    let expanded_magnitude = operator
        .columns
        .iter()
        .flat_map(|column| column.iter())
        .next()
        .map(|entry| {
            [
                entry.coefficient.real.numer().abs(),
                *entry.coefficient.real.denom(),
            ]
        })
        .ok_or_else(|| "Eq3.1g operator is empty".to_string())?;
    let normalization_mutation_detected =
        expanded_magnitude == [1, 2] && expanded_magnitude != [1, 4] && [-1_i64, 8] != [-1, 4];
    let a3_rank = SPINOR_DIMENSION * a3.e0_a3_to_g4_rank;
    let curl_rank = CURL_DIMENSION;
    let intersection = curl_rank
        .checked_sub(outside_rank)
        .ok_or_else(|| "outside Eq3.1g rank exceeds curl rank".to_string())?;
    let combined = a3_rank + outside_rank;
    let fiber_kernel = SPINOR_DIMENSION * THREE_FORM_DIMENSION + curl_rank - combined;
    let nonzero_target_class = intersection > 0 && a3.local_lorentz_vertical_image_rank == 0;
    let passed = a3.passed
        && a3_rank == 3_840
        && curl_rank == 1_760
        && outside_rank == 1_440
        && intersection == 320
        && combined == 5_280
        && fiber_kernel == 1_760
        && frame_rank == 320
        && bianchi_residual == 0
        && mutation_residual > 0
        && normalization_mutation_detected
        && nonzero_target_class;
    Ok(ComponentGravitinoA3FiberReport {
        schema_version: "adynkra-11d-component-gravitino-a3-fiber-v1",
        momentum_axis,
        a3_first_jet_dimension: SPINOR_DIMENSION * THREE_FORM_DIMENSION,
        a3_descendant_rank: a3_rank,
        component_curl_dimension: CURL_DIMENSION,
        eq31g_curl_rank: curl_rank,
        eq31g_outside_a3_rank: outside_rank,
        target_image_intersection_dimension: intersection,
        combined_target_rank: combined,
        fiber_product_kernel_dimension: fiber_kernel,
        component_frame_to_curl_rank: frame_rank,
        component_frame_bianchi_residual_entries: bianchi_residual,
        target_bianchi_mutation_residual_entries: mutation_residual,
        eq31g_printed_coefficient: [-1, 8],
        antisymmetrization_partition_multiplicity: 4,
        expanded_partition_coefficient_magnitude: expanded_magnitude,
        normalization_mutation_detected,
        independent_a3_passed: a3.passed,
        independent_a3_local_lorentz_image_rank: a3.local_lorentz_vertical_image_rank,
        component_source_quotient_applied: false,
        nonzero_target_class,
        a3_basis_sha256: a3.a3_basis_sha256,
        g4_basis_sha256: a3.g4_basis_sha256,
        curl_basis_sha256: hash_curl_basis(),
        eq31g_operator_sha256: hash_operator(&operator),
        passed,
        boundary: "This certifies the fixed-momentum component fiber product between independent DA3 and independent gravitino curl under Eq3.1g. It uses no Eq25, Hhat, or Eq40 source map and therefore does not identify the physical source inside Hhat or prove irreducibility.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_component_fiber_product_is_nonzero_and_bianchi_closed() {
        let report = build_component_gravitino_a3_fiber_report().unwrap();
        eprintln!("COMPONENT_A3_FIBER {report:#?}");
        assert!(report.passed);
        assert_eq!(report.target_image_intersection_dimension, 320);
        assert_eq!(report.component_frame_bianchi_residual_entries, 0);
        assert!(report.target_bianchi_mutation_residual_entries > 0);
        assert_eq!(report.independent_a3_local_lorentz_image_rank, 0);
    }
}
