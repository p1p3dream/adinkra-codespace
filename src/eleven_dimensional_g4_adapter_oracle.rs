//! Standalone exact oracle for the two missing momentum-to-G4 adapters.
//!
//! This module deliberately consumes, but does not modify, the corrected
//! Gamma4 decomposition. It works directly in the repository's increasing-mask
//! Cartesian bases and mostly-plus convention.

use std::collections::BTreeMap;

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_gamma24_source_variance::{
    SOURCE_DIMENSION, corrected_gamma_four_decomposition,
};
use crate::eleven_dimensional_physical_curvature::{ExactQi, SparseQiOperator, VECTOR_DIMENSION};

const PROOF_PRIME: i64 = 1_073_741_783;

type MomentumPair = (u8, u8);
type SymbolRow = (u16, MomentumPair);

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OraclePivotRow {
    pub source_column: usize,
    pub target_five_form_mask: u16,
    pub momentum_axes: [u8; 2],
    pub exterior_value: [i64; 2],
    pub hook_value: [i64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OracleIndependenceWitness {
    pub first: OraclePivotRow,
    pub second: OraclePivotRow,
    pub exact_determinant: [i64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct G4AdapterOracleReport {
    pub schema_version: &'static str,
    pub convention: &'static str,
    pub lambda_five_formula: &'static str,
    pub hook_formula: &'static str,
    pub trace_formula: &'static str,
    pub source_columns: usize,
    pub formal_bianchi_rows: u64,
    pub trace_nonzero_bianchi_rows: u64,
    pub lambda_five_nonzero_bianchi_rows: u64,
    pub hook_nonzero_bianchi_rows: u64,
    pub bianchi_coefficient_matrix_dimensions: [u64; 2],
    pub bianchi_coefficient_rank_mod_prime: usize,
    pub proof_prime: i64,
    pub exact_kernel_basis: Vec<[i64; 3]>,
    pub first_independence_witness: Option<OracleIndependenceWitness>,
    pub symbolic_rows_sha256: String,
    pub lambda_five_time_component: &'static str,
    pub lambda_five_spatial_component: &'static str,
    pub hook_time_component: &'static str,
    pub hook_spatial_component: &'static str,
    pub passed: bool,
    pub boundary: &'static str,
}

fn masks(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn metric(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

/// Sign of inserting a covector at its sorted position in an increasing form.
fn wedge_sign(existing_mask: u16, axis: usize) -> i64 {
    if (existing_mask & ((1_u16 << axis) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Sign of moving the contracted index from its increasing position to front.
fn contraction_sign(form_mask: u16, axis: usize) -> i64 {
    debug_assert_ne!(form_mask & (1_u16 << axis), 0);
    wedge_sign(form_mask ^ (1_u16 << axis), axis)
}

fn pair(left: usize, right: usize) -> MomentumPair {
    (left.min(right) as u8, left.max(right) as u8)
}

fn add(
    output: &mut BTreeMap<SymbolRow, [ExactQi; 3]>,
    key: SymbolRow,
    lane: usize,
    value: ExactQi,
) {
    if value.is_zero() {
        return;
    }
    let row = output
        .entry(key)
        .or_insert_with(|| std::array::from_fn(|_| ExactQi::zero()));
    row[lane].add_assign(&value);
    if row.iter().all(ExactQi::is_zero) {
        output.remove(&key);
    }
}

fn scaled(value: &ExactQi, factor: i64) -> ExactQi {
    value.scaled(&Ratio::from_integer(factor))
}

fn add_trace_bianchi(
    rows: &mut BTreeMap<SymbolRow, [ExactQi; 3]>,
    source: &SparseQiOperator,
    column: usize,
) {
    let three = masks(3);
    for entry in &source.columns[column] {
        let three_mask = three[entry.row];
        for first in 0..VECTOR_DIMENSION {
            if three_mask & (1_u16 << first) != 0 {
                continue;
            }
            let four_mask = three_mask | (1_u16 << first);
            let first_sign = wedge_sign(three_mask, first);
            for second in 0..VECTOR_DIMENSION {
                if four_mask & (1_u16 << second) != 0 {
                    continue;
                }
                let five_mask = four_mask | (1_u16 << second);
                let factor = first_sign * wedge_sign(four_mask, second);
                add(
                    rows,
                    (five_mask, pair(first, second)),
                    0,
                    scaled(&entry.coefficient, factor),
                );
            }
        }
    }
}

fn add_lambda_five_bianchi(
    rows: &mut BTreeMap<SymbolRow, [ExactQi; 3]>,
    source: &SparseQiOperator,
    column: usize,
) {
    let five = masks(5);
    for entry in &source.columns[column] {
        let five_mask = five[entry.row];
        for contracted in 0..VECTOR_DIMENSION {
            if five_mask & (1_u16 << contracted) == 0 {
                continue;
            }
            let four_mask = five_mask ^ (1_u16 << contracted);
            let contraction = contraction_sign(five_mask, contracted) * metric(contracted);
            for exterior in 0..VECTOR_DIMENSION {
                if four_mask & (1_u16 << exterior) != 0 {
                    continue;
                }
                let output_mask = four_mask | (1_u16 << exterior);
                let factor = contraction * wedge_sign(four_mask, exterior);
                add(
                    rows,
                    (output_mask, pair(contracted, exterior)),
                    1,
                    scaled(&entry.coefficient, factor),
                );
            }
        }
    }
}

fn add_hook_bianchi(
    rows: &mut BTreeMap<SymbolRow, [ExactQi; 3]>,
    source: &SparseQiOperator,
    column: usize,
) {
    let four = masks(4);
    for entry in &source.columns[column] {
        let four_mask = four[entry.row / VECTOR_DIMENSION];
        let contracted = entry.row % VECTOR_DIMENSION;
        for exterior in 0..VECTOR_DIMENSION {
            if four_mask & (1_u16 << exterior) != 0 {
                continue;
            }
            let output_mask = four_mask | (1_u16 << exterior);
            let factor = wedge_sign(four_mask, exterior);
            add(
                rows,
                (output_mask, pair(contracted, exterior)),
                2,
                scaled(&entry.coefficient, factor),
            );
        }
    }
}

fn inverse_mod(value: i64) -> i64 {
    let mut base = value.rem_euclid(PROOF_PRIME);
    let mut exponent = PROOF_PRIME - 2;
    let mut result = 1_i64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = (result * base).rem_euclid(PROOF_PRIME);
        }
        base = (base * base).rem_euclid(PROOF_PRIME);
        exponent >>= 1;
    }
    result
}

fn residue(value: &ExactQi) -> [i64; 2] {
    let real_denominator = (*value.real.denom()).rem_euclid(PROOF_PRIME);
    let imaginary_denominator = (*value.imaginary.denom()).rem_euclid(PROOF_PRIME);
    assert_ne!(real_denominator, 0, "inadmissible real denominator");
    assert_ne!(
        imaginary_denominator, 0,
        "inadmissible imaginary denominator"
    );
    [
        (*value.real.numer()).rem_euclid(PROOF_PRIME) * inverse_mod(real_denominator) % PROOF_PRIME,
        (*value.imaginary.numer()).rem_euclid(PROOF_PRIME) * inverse_mod(imaginary_denominator)
            % PROOF_PRIME,
    ]
}

#[derive(Default)]
struct RankThree {
    pivots: Vec<([i64; 3], usize)>,
}

impl RankThree {
    fn consume(&mut self, source: [i64; 3]) {
        let mut row = source;
        self.pivots.sort_by_key(|(_, position)| *position);
        for (pivot, position) in &self.pivots {
            if row[*position] == 0 {
                continue;
            }
            let factor = row[*position];
            for column in *position..3 {
                row[column] = (row[column] - factor * pivot[column]).rem_euclid(PROOF_PRIME);
            }
        }
        let Some(position) = row.iter().position(|value| *value != 0) else {
            return;
        };
        let inverse = inverse_mod(row[position]);
        for value in &mut row[position..] {
            *value = (*value * inverse).rem_euclid(PROOF_PRIME);
        }
        self.pivots.push((row, position));
    }

    fn rank(&self) -> usize {
        self.pivots.len()
    }
}

fn hash_qi(hasher: &mut Sha256, value: &ExactQi) {
    hasher.update(value.real.numer().to_le_bytes());
    hasher.update(value.real.denom().to_le_bytes());
    hasher.update(value.imaginary.numer().to_le_bytes());
    hasher.update(value.imaginary.denom().to_le_bytes());
}

fn real_pair(value: &ExactQi) -> [i64; 2] {
    assert_eq!(*value.imaginary.numer(), 0);
    [*value.real.numer(), *value.real.denom()]
}

fn pivot_row(
    source_column: usize,
    mask: u16,
    momentum: MomentumPair,
    exterior: &ExactQi,
    hook: &ExactQi,
) -> OraclePivotRow {
    OraclePivotRow {
        source_column,
        target_five_form_mask: mask,
        momentum_axes: [momentum.0, momentum.1],
        exterior_value: real_pair(exterior),
        hook_value: real_pair(hook),
    }
}

fn determinant(
    first_e: &ExactQi,
    first_h: &ExactQi,
    second_e: &ExactQi,
    second_h: &ExactQi,
) -> ExactQi {
    let multiply = |left: &ExactQi, right: &ExactQi| ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    };
    let mut result = multiply(first_e, second_h);
    result.add_assign(&multiply(first_h, second_e).scaled(&Ratio::from_integer(-1)));
    result
}

pub fn verify() -> G4AdapterOracleReport {
    let maps = corrected_gamma_four_decomposition();
    assert_eq!(maps.trace_lambda_three.input_dimension, SOURCE_DIMENSION);
    assert_eq!(maps.exterior_lambda_five.input_dimension, SOURCE_DIMENSION);
    assert_eq!(maps.hook_10010.input_dimension, SOURCE_DIMENSION);

    let mut formal_bianchi_rows = 0_u64;
    let mut nonzero = [0_u64; 3];
    let mut modular_rank = RankThree::default();
    let mut first_pivot = None::<(OraclePivotRow, ExactQi, ExactQi)>;
    let mut first_independence_witness = None;
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-11d-g4-adapter-formal-bianchi-oracle-v1");

    for column in 0..SOURCE_DIMENSION {
        let mut rows = BTreeMap::new();
        add_trace_bianchi(&mut rows, &maps.trace_lambda_three, column);
        add_lambda_five_bianchi(&mut rows, &maps.exterior_lambda_five, column);
        add_hook_bianchi(&mut rows, &maps.hook_10010, column);
        for ((mask, momentum), values) in rows {
            formal_bianchi_rows += 1;
            hasher.update((column as u64).to_le_bytes());
            hasher.update(mask.to_le_bytes());
            hasher.update([momentum.0, momentum.1]);
            for lane in 0..3 {
                if !values[lane].is_zero() {
                    nonzero[lane] += 1;
                }
                hash_qi(&mut hasher, &values[lane]);
            }
            let residues: [[i64; 2]; 3] = std::array::from_fn(|lane| residue(&values[lane]));
            modular_rank.consume([residues[0][0], residues[1][0], residues[2][0]]);
            modular_rank.consume([residues[0][1], residues[1][1], residues[2][1]]);
            if first_independence_witness.is_none()
                && (!values[1].is_zero() || !values[2].is_zero())
            {
                let current = pivot_row(column, mask, momentum, &values[1], &values[2]);
                if let Some((first, first_e, first_h)) = &first_pivot {
                    let exact_determinant = determinant(first_e, first_h, &values[1], &values[2]);
                    if !exact_determinant.is_zero() {
                        first_independence_witness = Some(OracleIndependenceWitness {
                            first: first.clone(),
                            second: current,
                            exact_determinant: real_pair(&exact_determinant),
                        });
                    }
                } else {
                    first_pivot = Some((current, values[1].clone(), values[2].clone()));
                }
            }
        }
    }
    let rank = modular_rank.rank();
    let passed = nonzero[0] == 0
        && nonzero[1] > 0
        && nonzero[2] > 0
        && rank == 2
        && first_independence_witness.is_some();
    G4AdapterOracleReport {
        schema_version: "adynkra-11d-g4-adapter-oracle-v1",
        convention: "mostly-plus eta=(-,+,...,+), covariant p_a, increasing form masks, unnormalized exterior derivative",
        lambda_five_formula: "G_abcd = p^e E_eabcd, with p^0=-p_0 and p^i=p_i",
        hook_formula: "G_abcd = p_e K_abcd{}^e; the separate hook index is contravariant, so no metric factor is inserted",
        trace_formula: "G_abcd = (p wedge T)_abcd = sum_j (-1)^j p_a_j T_a_0...omit(a_j)...a_3",
        source_columns: SOURCE_DIMENSION,
        formal_bianchi_rows,
        trace_nonzero_bianchi_rows: nonzero[0],
        lambda_five_nonzero_bianchi_rows: nonzero[1],
        hook_nonzero_bianchi_rows: nonzero[2],
        bianchi_coefficient_matrix_dimensions: [formal_bianchi_rows, 3],
        bianchi_coefficient_rank_mod_prime: rank,
        proof_prime: PROOF_PRIME,
        exact_kernel_basis: if rank == 2 {
            vec![[1, 0, 0]]
        } else {
            Vec::new()
        },
        first_independence_witness,
        symbolic_rows_sha256: format!("{:x}", hasher.finalize()),
        lambda_five_time_component: "E_01234 contributes -p_0 E_01234 to G_1234",
        lambda_five_spatial_component: "E_01234 contributes +p_4 E_01234 to G_0123",
        hook_time_component: "K_0123{}^0 contributes +p_0 K_0123{}^0 to G_0123",
        hook_spatial_component: "K_0123{}^4 contributes +p_4 K_0123{}^4 to G_0123",
        passed,
        boundary: "This fixes the two Cartesian momentum adapters and proves the formal target-Bianchi coefficient kernel on the corrected Gamma4 source stream. It does not identify a surviving normalization with the physical component A3/G4 or impose additional source equations.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_index_signs_are_mostly_plus() {
        let five = (1_u16 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        assert_eq!(contraction_sign(five, 0) * metric(0), -1);
        assert_eq!(contraction_sign(five, 4) * metric(4), 1);
        assert_eq!(metric(0), -1);
        assert_eq!(metric(4), 1);
        // The hook's separate index is already upper: p_e K^e.
        assert_eq!(wedge_sign((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3), 4), 1);
    }

    #[test]
    fn formal_bianchi_kernel_selects_only_the_trace_channel() {
        let report = verify();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.trace_nonzero_bianchi_rows, 0);
        assert!(report.lambda_five_nonzero_bianchi_rows > 0);
        assert!(report.hook_nonzero_bianchi_rows > 0);
        assert_eq!(report.bianchi_coefficient_rank_mod_prime, 2);
        assert_eq!(report.exact_kernel_basis, vec![[1, 0, 0]]);
        let witness = report.first_independence_witness.as_ref().unwrap();
        assert_ne!(witness.exact_determinant[0], 0);
        assert_ne!(witness.exact_determinant[1], 0);
        assert!(report.passed);
    }
}
