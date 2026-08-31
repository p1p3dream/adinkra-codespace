//! Exact Cartesian Casimir projectors for `S tensor Lambda^4 V`.
//!
//! The five B5 summands occur with multiplicity one and have distinct
//! quadratic-Casimir eigenvalues. This gives canonical projectors without a
//! phase choice in separate highest-weight Clebsch systems.

use num_rational::Ratio;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

use crate::eleven_dimensional_majorana::real_gamma_matrices;
use crate::eleven_dimensional_prepotential::b5_dimension;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const FORM_DEGREE: usize = 4;
const FOUR_FORM_DIMENSION: usize = 330;
const TARGET_DIMENSION: usize = SPINOR_DIMENSION * FOUR_FORM_DIMENSION;
const MODULE_SOURCE: &[u8] = include_bytes!("eleven_dimensional_dg4_casimir_projectors.rs");

type SparseIntegerVector = BTreeMap<usize, i64>;
type SparseRationalVector = BTreeMap<usize, Ratio<i64>>;

pub(crate) const DG4_CASIMIR_ROW_SUPPORT: usize = 29;
pub(crate) const DG4_PROJECTOR_PROOF_PRIMES: [u32; 3] =
    [1_073_741_783, 1_073_741_723, 1_073_741_719];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dg4CasimirCsr {
    pub row_offsets: Vec<u32>,
    pub column_indices: Vec<u32>,
    pub exact_values: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Dg4CsrRowStatistics {
    pub rows: usize,
    pub nonzeros: usize,
    pub minimum_row_nonzeros: usize,
    pub maximum_row_nonzeros: usize,
    pub row_nonzero_histogram: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Dg4DeviceProjectorSpec {
    pub dynkin_label: String,
    pub target_eigenvalue: i64,
    pub ordered_shift_eigenvalues: [i64; 4],
    pub exact_denominator: i64,
    pub denominator_residues: [u32; 3],
    pub inverse_denominator_residues: [u32; 3],
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Dg4CsrParityCanary {
    pub dynkin_label: String,
    pub prime: u32,
    pub input_ordinal: usize,
    pub four_stages_checked: usize,
    pub expected_nonzeros: usize,
    pub residual_entries: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Dg4DeviceCsrReport {
    pub schema_version: &'static str,
    pub target_dimension: usize,
    pub proof_primes: [u32; 3],
    pub module_source_sha256: String,
    pub cartesian_basis_sha256: String,
    pub casimir_operator_sha256: String,
    pub projector_polynomials_sha256: String,
    pub row_statistics: Dg4CsrRowStatistics,
    pub exact_coefficient_minimum: i64,
    pub exact_coefficient_maximum: i64,
    pub csr_structure_sha256: String,
    pub exact_coefficients_sha256: String,
    pub modular_coefficients_sha256: String,
    pub projector_specs_sha256: String,
    pub binary_file: String,
    pub binary_sha256: String,
    pub binary_bytes: usize,
    pub projectors: Vec<Dg4DeviceProjectorSpec>,
    pub parity_canaries: Vec<Dg4CsrParityCanary>,
    pub parity_canary_sha256: String,
    pub wrong_shift_mutation_residual_entries: usize,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct Dg4ProjectorIntegerPass {
    pub shift_eigenvalue: i64,
    /// Strictly increasing Cartesian target ordinals with nonzero values.
    pub entries: Vec<(u16, i128)>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct Dg4ProjectorNumeratorOracle {
    pub schema_version: &'static str,
    pub dynkin_label: String,
    pub target_eigenvalue: i64,
    pub ordered_shift_eigenvalues: [i64; 4],
    pub denominator: i64,
    pub canonical_input: Vec<(u16, i128)>,
    pub passes: Vec<Dg4ProjectorIntegerPass>,
    pub numerator: Vec<(u16, i128)>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Dg4CasimirSector {
    pub dynkin_label: &'static str,
    pub expected_dimension: u64,
    pub casimir4_eigenvalue: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Dg4CasimirProjectorCanary {
    pub schema_version: &'static str,
    pub target: &'static str,
    pub target_dimension: usize,
    pub module_source_sha256: String,
    pub cartesian_basis_sha256: String,
    pub casimir_operator_sha256: String,
    pub projector_polynomials_sha256: String,
    pub sectors: Vec<Dg4CasimirSector>,
    pub sector_dimensions_sum: u64,
    pub casimir_columns_constructed: usize,
    pub minimum_casimir_column_support: usize,
    pub maximum_casimir_column_support: usize,
    pub sample_basis_ordinals: Vec<usize>,
    pub minimal_polynomial_sample_residuals: usize,
    pub projector_sum_sample_residuals: usize,
    pub projector_eigen_sample_residuals: usize,
    pub wrong_eigenvalue_mutation_residuals: usize,
    pub exhaustive_minimal_polynomial_columns_checked: usize,
    pub exhaustive_minimal_polynomial_residuals: usize,
    pub exhaustive_projector_traces: Vec<String>,
    pub exhaustive_projector_ranks: Vec<usize>,
    pub exhaustive_projector_ranks_constructed: bool,
    pub passed_canary: bool,
    pub boundary: &'static str,
}

fn masks_of_degree(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn lorentz_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn wedge_sign(mask: u16, index: usize) -> i64 {
    if (mask & ((1_u16 << index) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn covector_generator(axis: usize, left: usize, right: usize) -> Vec<(usize, i64)> {
    let mut output = Vec::new();
    if axis == left {
        output.push((right, -lorentz_sign(right)));
    }
    if axis == right {
        output.push((left, lorentz_sign(left)));
    }
    output
}

fn form_generator(mask: u16, left: usize, right: usize) -> Vec<(u16, i64)> {
    let mut output = BTreeMap::new();
    for axis in 0..VECTOR_DIMENSION {
        if mask & (1_u16 << axis) == 0 {
            continue;
        }
        let remaining = mask ^ (1_u16 << axis);
        let removal = wedge_sign(remaining, axis);
        for (replacement, coefficient) in covector_generator(axis, left, right) {
            if remaining & (1_u16 << replacement) != 0 {
                continue;
            }
            let insertion = wedge_sign(remaining, replacement);
            *output
                .entry(remaining | (1_u16 << replacement))
                .or_default() += removal * insertion * coefficient;
        }
    }
    output
        .into_iter()
        .filter(|(_, value)| *value != 0)
        .collect()
}

fn lower_gamma_product(left: usize, right: usize) -> Vec<Vec<i16>> {
    let gammas = real_gamma_matrices();
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    let metric = lorentz_sign(left) * lorentz_sign(right);
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            let l = i16::from(gammas[left][row][pivot]);
            if l == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] +=
                    i16::try_from(metric).unwrap() * l * i16::from(gammas[right][pivot][column]);
            }
        }
    }
    output
}

fn add_integer(output: &mut SparseIntegerVector, row: usize, value: i64) {
    if value == 0 {
        return;
    }
    *output.entry(row).or_default() += value;
    if output[&row] == 0 {
        output.remove(&row);
    }
}

fn basis_parts(ordinal: usize) -> (usize, usize) {
    (ordinal / FOUR_FORM_DIMENSION, ordinal % FOUR_FORM_DIMENSION)
}

fn basis_ordinal(spinor: usize, form_ordinal: usize) -> usize {
    spinor * FOUR_FORM_DIMENSION + form_ordinal
}

#[derive(Clone)]
struct GeneratorData {
    metric_pair: i64,
    gamma: Vec<Vec<i16>>,
    form_actions: Vec<Vec<(usize, i64)>>,
}

fn generator_data() -> &'static Vec<GeneratorData> {
    static DATA: OnceLock<Vec<GeneratorData>> = OnceLock::new();
    DATA.get_or_init(|| {
        let forms = masks_of_degree(FORM_DEGREE);
        let lookup = forms
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, mask)| (mask, ordinal))
            .collect::<BTreeMap<_, _>>();
        let mut output = Vec::with_capacity(55);
        for left in 0..VECTOR_DIMENSION {
            for right in (left + 1)..VECTOR_DIMENSION {
                output.push(GeneratorData {
                    metric_pair: lorentz_sign(left) * lorentz_sign(right),
                    gamma: lower_gamma_product(left, right),
                    form_actions: forms
                        .iter()
                        .map(|&mask| {
                            form_generator(mask, left, right)
                                .into_iter()
                                .map(|(next, coefficient)| (lookup[&next], coefficient))
                                .collect()
                        })
                        .collect(),
                });
            }
        }
        output
    })
}

/// Apply `K_ab = Gamma_ab + 2 M_ab`, twice the standard total generator.
fn apply_k_cached(input: &SparseIntegerVector, generator: &GeneratorData) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    for (&ordinal, &coefficient) in input {
        let (spinor, form_ordinal) = basis_parts(ordinal);
        for (next_spinor, row) in generator.gamma.iter().enumerate() {
            let value = i64::from(row[spinor]);
            if value != 0 {
                add_integer(
                    &mut output,
                    basis_ordinal(next_spinor, form_ordinal),
                    coefficient * value,
                );
            }
        }
        for &(next_form, value) in &generator.form_actions[form_ordinal] {
            add_integer(
                &mut output,
                basis_ordinal(spinor, next_form),
                2 * coefficient * value,
            );
        }
    }
    output
}

pub(crate) fn dg4_lorentz_generator_action_integer(
    left: usize,
    right: usize,
    input: &BTreeMap<usize, i64>,
) -> Result<BTreeMap<usize, i64>, String> {
    if left >= right
        || right >= VECTOR_DIMENSION
        || input.keys().any(|&row| row >= TARGET_DIMENSION)
    {
        return Err("D G4 Lorentz-generator input is invalid".to_string());
    }
    let ordinal = (0..left)
        .map(|axis| VECTOR_DIMENSION - axis - 1)
        .sum::<usize>()
        + right
        - left
        - 1;
    Ok(apply_k_cached(input, &generator_data()[ordinal]))
}

fn casimir_column(ordinal: usize) -> SparseIntegerVector {
    let input = BTreeMap::from([(ordinal, 1_i64)]);
    let mut output = SparseIntegerVector::new();
    for generator in generator_data() {
        let once = apply_k_cached(&input, generator);
        let twice = apply_k_cached(&once, generator);
        for (row, value) in twice {
            add_integer(&mut output, row, -generator.metric_pair * value);
        }
    }
    output
}

fn casimir_columns() -> &'static Vec<Vec<(usize, i64)>> {
    static COLUMNS: OnceLock<Vec<Vec<(usize, i64)>>> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        assert_eq!(masks_of_degree(FORM_DEGREE).len(), FOUR_FORM_DIMENSION);
        (0..TARGET_DIMENSION)
            .map(|ordinal| casimir_column(ordinal).into_iter().collect())
            .collect()
    })
}

/// Canonical row-major sparse matrix for the exact doubled-generator C4.
/// Rows and entries are both numeric Cartesian ordinals. Every row has the
/// representation-theoretically fixed support 29, so CUDA may use one fixed
/// stencil per output coordinate without offsets or atomics.
pub(crate) fn dg4_casimir_row_major() -> Result<Vec<[(u16, i16); DG4_CASIMIR_ROW_SUPPORT]>, String>
{
    let mut rows = vec![Vec::<(u16, i16)>::new(); TARGET_DIMENSION];
    for (column, entries) in casimir_columns().iter().enumerate() {
        let column = u16::try_from(column)
            .map_err(|_| "D G4 Casimir column does not fit u16".to_string())?;
        for &(row, value) in entries {
            let value = i16::try_from(value)
                .map_err(|_| "D G4 Casimir coefficient does not fit i16".to_string())?;
            rows[row].push((column, value));
        }
    }
    rows.into_iter()
        .enumerate()
        .map(|(row, entries)| {
            if entries.len() != DG4_CASIMIR_ROW_SUPPORT
                || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
                || entries.iter().any(|(_, value)| *value == 0)
            {
                return Err(format!(
                    "D G4 Casimir row {row} is not a canonical support-{DG4_CASIMIR_ROW_SUPPORT} stencil"
                ));
            }
            entries.try_into().map_err(|entries: Vec<_>| {
                format!(
                    "D G4 Casimir row {row} has support {}, expected {DG4_CASIMIR_ROW_SUPPORT}",
                    entries.len()
                )
            })
        })
        .collect()
}

/// Exact row-major CSR export of the doubled-generator C4 operator.
/// The arrays are canonical: rows are numeric target ordinals and columns
/// are strictly increasing inside every row.
pub(crate) fn dg4_casimir_csr_exact() -> Result<Dg4CasimirCsr, String> {
    let rows = dg4_casimir_row_major()?;
    let mut row_offsets = Vec::with_capacity(TARGET_DIMENSION + 1);
    let mut column_indices = Vec::with_capacity(TARGET_DIMENSION * DG4_CASIMIR_ROW_SUPPORT);
    let mut exact_values = Vec::with_capacity(TARGET_DIMENSION * DG4_CASIMIR_ROW_SUPPORT);
    row_offsets.push(0);
    for row in rows {
        for (column, value) in row {
            column_indices.push(u32::from(column));
            exact_values.push(i64::from(value));
        }
        row_offsets.push(
            u32::try_from(column_indices.len())
                .map_err(|_| "D G4 Casimir CSR nonzero count does not fit u32".to_string())?,
        );
    }
    Ok(Dg4CasimirCsr {
        row_offsets,
        column_indices,
        exact_values,
    })
}

fn residue_i128(value: i128, prime: u32) -> u32 {
    let prime_i128 = i128::from(prime);
    let residue = ((value % prime_i128) + prime_i128) % prime_i128;
    u32::try_from(residue).expect("canonical residue fits u32")
}

fn modular_power(mut base: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut output = 1_u64;
    let mut square = u64::from(base);
    let modulus = u64::from(prime);
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = output * square % modulus;
        }
        square = square * square % modulus;
        exponent >>= 1;
    }
    base = u32::try_from(output).expect("modular power fits u32");
    base
}

fn modular_inverse(value: i64, prime: u32) -> Result<u32, String> {
    let residue = residue_i128(i128::from(value), prime);
    if residue == 0 {
        return Err(format!(
            "D G4 projector denominator vanishes modulo {prime}"
        ));
    }
    Ok(modular_power(residue, prime - 2, prime))
}

fn apply_csr_shift_modular(
    csr: &Dg4CasimirCsr,
    input: &[u32],
    shift: i64,
    prime: u32,
) -> Result<Vec<u32>, String> {
    if input.len() != TARGET_DIMENSION || csr.row_offsets.len() != TARGET_DIMENSION + 1 {
        return Err("D G4 modular CSR vector dimension mismatch".to_string());
    }
    let modulus = u64::from(prime);
    let shift_residue = u64::from(residue_i128(i128::from(shift), prime));
    let mut output = vec![0_u32; TARGET_DIMENSION];
    for row in 0..TARGET_DIMENSION {
        let begin = usize::try_from(csr.row_offsets[row]).unwrap();
        let end = usize::try_from(csr.row_offsets[row + 1]).unwrap();
        let mut accumulator = 0_u64;
        for entry in begin..end {
            let column = usize::try_from(csr.column_indices[entry]).unwrap();
            let coefficient = u64::from(residue_i128(i128::from(csr.exact_values[entry]), prime));
            accumulator = (accumulator + coefficient * u64::from(input[column])) % modulus;
        }
        let diagonal = shift_residue * u64::from(input[row]) % modulus;
        output[row] = u32::try_from((accumulator + modulus - diagonal) % modulus).unwrap();
    }
    Ok(output)
}

/// Apply the complete four-stage spectral projector with CSR SpMV modulo a
/// pinned proof prime. This is the CPU parity oracle for the device kernel.
pub(crate) fn dg4_apply_projector_modular_csr(
    dynkin_label: &str,
    prime: u32,
    input: &[(u16, u32)],
) -> Result<Vec<u32>, String> {
    if !DG4_PROJECTOR_PROOF_PRIMES.contains(&prime) {
        return Err(format!("unpinned D G4 projector prime {prime}"));
    }
    if input.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || input.iter().any(|&(row, value)| {
            usize::from(row) >= TARGET_DIMENSION || value == 0 || value >= prime
        })
    {
        return Err("D G4 modular projector input is not canonical compact COO".to_string());
    }
    let (sector, eigenvalues) = checked_sector(dynkin_label)?;
    let mut state = vec![0_u32; TARGET_DIMENSION];
    for &(row, value) in input {
        state[usize::from(row)] = value;
    }
    let csr = dg4_casimir_csr_exact()?;
    for (other, &shift) in eigenvalues.iter().enumerate() {
        if other != sector {
            state = apply_csr_shift_modular(&csr, &state, shift, prime)?;
        }
    }
    let denominator = eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(other, &value)| (other != sector).then_some(eigenvalues[sector] - value))
        .product::<i64>();
    let inverse = u64::from(modular_inverse(denominator, prime)?);
    let modulus = u64::from(prime);
    for value in &mut state {
        *value = u32::try_from(u64::from(*value) * inverse % modulus).unwrap();
    }
    Ok(state)
}

fn canonical_i128(entries: &BTreeMap<usize, i128>) -> Vec<(u16, i128)> {
    entries
        .iter()
        .filter_map(|(&row, &value)| {
            (value != 0).then(|| (u16::try_from(row).expect("D G4 row fits u16"), value))
        })
        .collect()
}

fn apply_casimir_row_major_i128(
    input: &BTreeMap<usize, i128>,
    rows: &[[(u16, i16); DG4_CASIMIR_ROW_SUPPORT]],
) -> Result<BTreeMap<usize, i128>, String> {
    if rows.len() != TARGET_DIMENSION {
        return Err("D G4 row-major Casimir has the wrong row count".to_string());
    }
    let mut output = BTreeMap::new();
    for (row, stencil) in rows.iter().enumerate() {
        let mut value = 0_i128;
        for &(column, coefficient) in stencil {
            let Some(&input_value) = input.get(&usize::from(column)) else {
                continue;
            };
            value = value
                .checked_add(
                    input_value
                        .checked_mul(i128::from(coefficient))
                        .ok_or_else(|| "D G4 exact Casimir product overflowed i128".to_string())?,
                )
                .ok_or_else(|| "D G4 exact Casimir sum overflowed i128".to_string())?;
        }
        if value != 0 {
            output.insert(row, value);
        }
    }
    Ok(output)
}

fn apply_shift_row_major_i128(
    input: &BTreeMap<usize, i128>,
    shift: i64,
    rows: &[[(u16, i16); DG4_CASIMIR_ROW_SUPPORT]],
) -> Result<BTreeMap<usize, i128>, String> {
    let mut output = apply_casimir_row_major_i128(input, rows)?;
    for (&row, &value) in input {
        let shifted = value
            .checked_mul(i128::from(shift))
            .ok_or_else(|| "D G4 exact shift product overflowed i128".to_string())?;
        let next = output
            .get(&row)
            .copied()
            .unwrap_or(0)
            .checked_sub(shifted)
            .ok_or_else(|| "D G4 exact shift sum overflowed i128".to_string())?;
        if next == 0 {
            output.remove(&row);
        } else {
            output.insert(row, next);
        }
    }
    Ok(output)
}

/// Exact CPU oracle for the four sparse `(C4-lambda)` passes used by one
/// spectral projector. Input must be compact COO in strictly increasing
/// numeric target order. The returned numerator is not divided, which makes
/// it directly comparable with three-prime device residues.
pub(crate) fn dg4_projector_numerator_oracle(
    dynkin_label: &str,
    input: &[(u16, i64)],
) -> Result<Dg4ProjectorNumeratorOracle, String> {
    if input.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || input
            .iter()
            .any(|&(row, value)| usize::from(row) >= TARGET_DIMENSION || value == 0)
    {
        return Err("D G4 projector oracle input is not canonical compact COO".to_string());
    }
    let (sector, eigenvalues) = checked_sector(dynkin_label)?;
    let roots = eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(other, &value)| (other != sector).then_some(value))
        .collect::<Vec<_>>();
    let ordered_shift_eigenvalues: [i64; 4] = roots
        .clone()
        .try_into()
        .map_err(|_| "D G4 projector does not have four complementary roots".to_string())?;
    let denominator = ordered_shift_eigenvalues
        .iter()
        .map(|&root| eigenvalues[sector] - root)
        .try_fold(1_i64, |product, factor| {
            product
                .checked_mul(factor)
                .ok_or_else(|| "D G4 projector denominator overflowed i64".to_string())
        })?;
    let row_major = dg4_casimir_row_major()?;
    let mut state = input
        .iter()
        .map(|&(row, value)| (usize::from(row), i128::from(value)))
        .collect::<BTreeMap<_, _>>();
    let canonical_input = canonical_i128(&state);
    let mut passes = Vec::with_capacity(4);
    for shift in ordered_shift_eigenvalues {
        state = apply_shift_row_major_i128(&state, shift, &row_major)?;
        passes.push(Dg4ProjectorIntegerPass {
            shift_eigenvalue: shift,
            entries: canonical_i128(&state),
        });
    }
    let numerator = canonical_i128(&state);
    Ok(Dg4ProjectorNumeratorOracle {
        schema_version: "adynkra-11d-dg4-projector-numerator-oracle-v1",
        dynkin_label: dynkin_label.to_string(),
        target_eigenvalue: eigenvalues[sector],
        ordered_shift_eigenvalues,
        denominator,
        canonical_input,
        passes,
        numerator,
    })
}

fn apply_casimir_integer(input: &SparseIntegerVector) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    for (&column, &coefficient) in input {
        for &(row, value) in &casimir_columns()[column] {
            add_integer(&mut output, row, coefficient * value);
        }
    }
    output
}

fn apply_shift_integer(input: &SparseIntegerVector, eigenvalue: i64) -> SparseIntegerVector {
    let mut output = apply_casimir_integer(input);
    for (&row, &value) in input {
        add_integer(&mut output, row, -eigenvalue * value);
    }
    output
}

fn add_rational(output: &mut SparseRationalVector, row: usize, value: Ratio<i64>) {
    if value == Ratio::from_integer(0) {
        return;
    }
    *output.entry(row).or_default() += value;
    if output[&row] == Ratio::from_integer(0) {
        output.remove(&row);
    }
}

fn apply_casimir_rational(input: &SparseRationalVector) -> SparseRationalVector {
    let mut output = SparseRationalVector::new();
    for (&column, coefficient) in input {
        for &(row, value) in &casimir_columns()[column] {
            add_rational(
                &mut output,
                row,
                coefficient.clone() * Ratio::from_integer(value),
            );
        }
    }
    output
}

fn apply_projector(
    input: &SparseRationalVector,
    sector: usize,
    eigenvalues: &[i64],
) -> SparseRationalVector {
    let mut output = input.clone();
    let mut denominator = 1_i64;
    for (other, &eigenvalue) in eigenvalues.iter().enumerate() {
        if other == sector {
            continue;
        }
        let mut next = apply_casimir_rational(&output);
        for (&row, value) in &output {
            add_rational(
                &mut next,
                row,
                -value.clone() * Ratio::from_integer(eigenvalue),
            );
        }
        output = next;
        denominator *= eigenvalues[sector] - eigenvalue;
    }
    for value in output.values_mut() {
        *value /= Ratio::from_integer(denominator);
    }
    output.retain(|_, value| *value != Ratio::from_integer(0));
    output
}

fn checked_sector(label: &str) -> Result<(usize, Vec<i64>), String> {
    let sectors = sectors();
    let sector = sectors
        .iter()
        .position(|candidate| candidate.dynkin_label == label)
        .ok_or_else(|| format!("unknown D G4 target sector {label}"))?;
    let eigenvalues = sectors
        .iter()
        .map(|candidate| candidate.casimir4_eigenvalue)
        .collect();
    Ok((sector, eigenvalues))
}

fn validate_sparse_target(input: &BTreeMap<usize, Ratio<i64>>) -> Result<(), String> {
    if let Some(&ordinal) = input.keys().find(|&&ordinal| ordinal >= TARGET_DIMENSION) {
        return Err(format!(
            "D G4 Cartesian target ordinal {ordinal} exceeds {TARGET_DIMENSION}"
        ));
    }
    if input.values().any(|value| value.denom() <= &0) {
        return Err("D G4 sparse target has a nonpositive denominator".to_string());
    }
    Ok(())
}

/// Count nonzero Cartesian entries in `(C4 - eigenvalue(label)) input`.
pub(crate) fn casimir_eigen_residual_entries(
    label: &str,
    input: &BTreeMap<usize, Ratio<i64>>,
) -> Result<usize, String> {
    validate_sparse_target(input)?;
    let (sector, eigenvalues) = checked_sector(label)?;
    let mut residual = apply_casimir_rational(input);
    for (&row, value) in input {
        add_rational(
            &mut residual,
            row,
            -value.clone() * Ratio::from_integer(eigenvalues[sector]),
        );
    }
    Ok(residual.len())
}

fn polynomial_from_roots(roots: &[i64]) -> Vec<i64> {
    let mut coefficients = vec![1_i64];
    for &root in roots {
        let mut next = vec![0_i64; coefficients.len() + 1];
        for (degree, &coefficient) in coefficients.iter().enumerate() {
            next[degree] -= root * coefficient;
            next[degree + 1] += coefficient;
        }
        coefficients = next;
    }
    coefficients
}

fn exhaustive_spectral_certificate(eigenvalues: &[i64]) -> (usize, Vec<Ratio<i64>>, Vec<usize>) {
    let minimal = polynomial_from_roots(eigenvalues);
    let projector_polynomials = (0..eigenvalues.len())
        .map(|sector| {
            let roots = eigenvalues
                .iter()
                .enumerate()
                .filter_map(|(other, &value)| (other != sector).then_some(value))
                .collect::<Vec<_>>();
            let denominator = roots
                .iter()
                .map(|&other| eigenvalues[sector] - other)
                .product::<i64>();
            (polynomial_from_roots(&roots), denominator)
        })
        .collect::<Vec<_>>();
    let (residuals, traces) = (0..TARGET_DIMENSION)
        .into_par_iter()
        .map(|ordinal| {
            let mut powers = Vec::with_capacity(6);
            powers.push(BTreeMap::from([(ordinal, 1_i64)]));
            for degree in 1..=5 {
                let next = apply_casimir_integer(&powers[degree - 1]);
                powers.push(next);
            }
            let mut residual = SparseIntegerVector::new();
            for (degree, &coefficient) in minimal.iter().enumerate() {
                for (&row, &value) in &powers[degree] {
                    add_integer(&mut residual, row, coefficient * value);
                }
            }
            let diagonal_powers = powers
                .iter()
                .take(5)
                .map(|power| power.get(&ordinal).copied().unwrap_or(0))
                .collect::<Vec<_>>();
            let diagonal = projector_polynomials
                .iter()
                .map(|(polynomial, denominator)| {
                    let numerator = polynomial
                        .iter()
                        .zip(&diagonal_powers)
                        .map(|(&coefficient, &value)| coefficient * value)
                        .sum::<i64>();
                    Ratio::new(numerator, *denominator)
                })
                .collect::<Vec<_>>();
            (residual.len(), diagonal)
        })
        .reduce(
            || (0_usize, vec![Ratio::from_integer(0); eigenvalues.len()]),
            |(left_bad, mut left_trace), (right_bad, right_trace)| {
                for (left, right) in left_trace.iter_mut().zip(right_trace) {
                    *left += right;
                }
                (left_bad + right_bad, left_trace)
            },
        );
    let ranks = traces
        .iter()
        .map(|trace| {
            assert_eq!(*trace.denom(), 1, "projector trace must be integral");
            usize::try_from(*trace.numer()).unwrap()
        })
        .collect();
    (residuals, traces, ranks)
}

fn sectors() -> Vec<Dg4CasimirSector> {
    [
        ("00001", 55_i64),
        ("00011", 183),
        ("00101", 163),
        ("01001", 135),
        ("10001", 99),
    ]
    .into_iter()
    .map(|(dynkin_label, casimir4_eigenvalue)| Dg4CasimirSector {
        dynkin_label,
        expected_dimension: b5_dimension(dynkin_label),
        casimir4_eigenvalue,
    })
    .collect()
}

/// Apply one exact irreducible projector to a Cartesian `D G4` vector.
/// Target ordinals are `spinor * 330 + numeric_four_form_ordinal`.
pub fn project_dg4_target(
    dynkin_label: &str,
    input: &BTreeMap<usize, Ratio<i64>>,
) -> Result<BTreeMap<usize, Ratio<i64>>, String> {
    validate_sparse_target(input)?;
    let (sector, eigenvalues) = checked_sector(dynkin_label)?;
    Ok(apply_projector(input, sector, &eigenvalues))
}

pub fn dg4_target_dimension() -> usize {
    TARGET_DIMENSION
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn cartesian_basis_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-cartesian-basis-v1\0");
    for spinor in 0..SPINOR_DIMENSION {
        for mask in masks_of_degree(FORM_DEGREE) {
            hash.update(u32::try_from(spinor).unwrap().to_le_bytes());
            hash.update(mask.to_le_bytes());
        }
    }
    for gamma in real_gamma_matrices() {
        for row in gamma {
            for value in row {
                hash.update(value.to_le_bytes());
            }
        }
    }
    format!("{:x}", hash.finalize())
}

fn casimir_operator_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-casimir4-sparse-columns-v1\0");
    for (column, entries) in casimir_columns().iter().enumerate() {
        hash.update(u32::try_from(column).unwrap().to_le_bytes());
        hash.update(u32::try_from(entries.len()).unwrap().to_le_bytes());
        for &(row, value) in entries {
            hash.update(u32::try_from(row).unwrap().to_le_bytes());
            hash.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn projector_polynomials_sha256(eigenvalues: &[i64]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-casimir-projector-polynomials-v1\0");
    for (sector, &eigenvalue) in eigenvalues.iter().enumerate() {
        hash.update(u32::try_from(sector).unwrap().to_le_bytes());
        hash.update(eigenvalue.to_le_bytes());
        let roots = eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(other, &value)| (other != sector).then_some(value))
            .collect::<Vec<_>>();
        let denominator = roots
            .iter()
            .map(|&other| eigenvalue - other)
            .product::<i64>();
        hash.update(denominator.to_le_bytes());
        for coefficient in polynomial_from_roots(&roots) {
            hash.update(coefficient.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn device_projector_specs() -> Result<Vec<Dg4DeviceProjectorSpec>, String> {
    let sectors = sectors();
    let eigenvalues = sectors
        .iter()
        .map(|sector| sector.casimir4_eigenvalue)
        .collect::<Vec<_>>();
    sectors
        .iter()
        .enumerate()
        .map(|(sector, descriptor)| {
            let ordered_shift_eigenvalues: [i64; 4] = eigenvalues
                .iter()
                .enumerate()
                .filter_map(|(other, &value)| (other != sector).then_some(value))
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| "D G4 projector does not have four shifts".to_string())?;
            let exact_denominator = ordered_shift_eigenvalues
                .iter()
                .map(|&shift| descriptor.casimir4_eigenvalue - shift)
                .product::<i64>();
            let mut denominator_residues = [0_u32; 3];
            let mut inverse_denominator_residues = [0_u32; 3];
            for (prime_ordinal, &prime) in DG4_PROJECTOR_PROOF_PRIMES.iter().enumerate() {
                denominator_residues[prime_ordinal] =
                    residue_i128(i128::from(exact_denominator), prime);
                inverse_denominator_residues[prime_ordinal] =
                    modular_inverse(exact_denominator, prime)?;
            }
            Ok(Dg4DeviceProjectorSpec {
                dynkin_label: descriptor.dynkin_label.to_string(),
                target_eigenvalue: descriptor.casimir4_eigenvalue,
                ordered_shift_eigenvalues,
                exact_denominator,
                denominator_residues,
                inverse_denominator_residues,
            })
        })
        .collect()
}

fn hash_csr_structure(csr: &Dg4CasimirCsr) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-c4-csr-structure-v1\0");
    for value in &csr.row_offsets {
        hash.update(value.to_le_bytes());
    }
    for value in &csr.column_indices {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn hash_exact_coefficients(csr: &Dg4CasimirCsr) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-c4-csr-exact-i64-v1\0");
    for value in &csr.exact_values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn hash_modular_coefficients(csr: &Dg4CasimirCsr) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-c4-csr-modular-prime-major-u32-v1\0");
    for prime in DG4_PROJECTOR_PROOF_PRIMES {
        hash.update(prime.to_le_bytes());
        for &value in &csr.exact_values {
            hash.update(residue_i128(i128::from(value), prime).to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn hash_projector_specs(specs: &[Dg4DeviceProjectorSpec]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-dg4-c4-device-projector-specs-v1\0");
    for spec in specs {
        hash.update(spec.dynkin_label.as_bytes());
        hash.update([0]);
        hash.update(spec.target_eigenvalue.to_le_bytes());
        for shift in spec.ordered_shift_eigenvalues {
            hash.update(shift.to_le_bytes());
        }
        hash.update(spec.exact_denominator.to_le_bytes());
        for value in spec.denominator_residues {
            hash.update(value.to_le_bytes());
        }
        for value in spec.inverse_denominator_residues {
            hash.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn device_csr_blob(
    csr: &Dg4CasimirCsr,
    specs: &[Dg4DeviceProjectorSpec],
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ADG4CSR1");
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(TARGET_DIMENSION).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(csr.exact_values.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(DG4_CASIMIR_ROW_SUPPORT)
            .unwrap()
            .to_le_bytes(),
    );
    bytes.extend_from_slice(
        &u32::try_from(DG4_PROJECTOR_PROOF_PRIMES.len())
            .unwrap()
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&u32::try_from(specs.len()).unwrap().to_le_bytes());
    for prime in DG4_PROJECTOR_PROOF_PRIMES {
        bytes.extend_from_slice(&prime.to_le_bytes());
    }
    for value in &csr.row_offsets {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &csr.column_indices {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &csr.exact_values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for prime in DG4_PROJECTOR_PROOF_PRIMES {
        for &value in &csr.exact_values {
            bytes.extend_from_slice(&residue_i128(i128::from(value), prime).to_le_bytes());
        }
    }
    for spec in specs {
        bytes.extend_from_slice(&spec.target_eigenvalue.to_le_bytes());
        for shift in spec.ordered_shift_eigenvalues {
            bytes.extend_from_slice(&shift.to_le_bytes());
        }
        bytes.extend_from_slice(&spec.exact_denominator.to_le_bytes());
        for value in spec.denominator_residues {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in spec.inverse_denominator_residues {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn modular_stage_residuals(expected: &[(u16, i128)], actual: &[u32], prime: u32) -> usize {
    let expected = expected
        .iter()
        .map(|&(row, value)| (usize::from(row), residue_i128(value, prime)))
        .collect::<BTreeMap<_, _>>();
    actual
        .iter()
        .enumerate()
        .filter(|&(row, &value)| value != expected.get(&row).copied().unwrap_or(0))
        .count()
}

fn build_device_parity_canaries(
    csr: &Dg4CasimirCsr,
    specs: &[Dg4DeviceProjectorSpec],
) -> Result<(Vec<Dg4CsrParityCanary>, usize), String> {
    let input_ordinals = [0_usize, 5279, 10_559];
    let mut canaries = Vec::new();
    for spec in specs {
        for &prime in &DG4_PROJECTOR_PROOF_PRIMES {
            for &input_ordinal in &input_ordinals {
                let input_row = u16::try_from(input_ordinal).unwrap();
                let oracle = dg4_projector_numerator_oracle(&spec.dynkin_label, &[(input_row, 1)])?;
                let mut state = vec![0_u32; TARGET_DIMENSION];
                state[input_ordinal] = 1;
                let mut residual_entries = 0;
                for pass in &oracle.passes {
                    state = apply_csr_shift_modular(csr, &state, pass.shift_eigenvalue, prime)?;
                    residual_entries += modular_stage_residuals(&pass.entries, &state, prime);
                }
                let inverse = u64::from(modular_inverse(oracle.denominator, prime)?);
                let modulus = u64::from(prime);
                for value in &mut state {
                    *value = u32::try_from(u64::from(*value) * inverse % modulus).unwrap();
                }
                let expected_nonzeros = oracle
                    .numerator
                    .iter()
                    .filter(|(_, value)| residue_i128(*value, prime) != 0)
                    .count();
                let public =
                    dg4_apply_projector_modular_csr(&spec.dynkin_label, prime, &[(input_row, 1)])?;
                residual_entries += state.iter().zip(public).filter(|(a, b)| **a != *b).count();
                canaries.push(Dg4CsrParityCanary {
                    dynkin_label: spec.dynkin_label.clone(),
                    prime,
                    input_ordinal,
                    four_stages_checked: oracle.passes.len(),
                    expected_nonzeros,
                    residual_entries,
                });
            }
        }
    }

    let spec = &specs[0];
    let prime = DG4_PROJECTOR_PROOF_PRIMES[0];
    let oracle = dg4_projector_numerator_oracle(&spec.dynkin_label, &[(0, 1)])?;
    let mut mutated = vec![0_u32; TARGET_DIMENSION];
    mutated[0] = 1;
    for (stage, &shift) in spec.ordered_shift_eigenvalues.iter().enumerate() {
        mutated =
            apply_csr_shift_modular(csr, &mutated, shift + if stage == 0 { 1 } else { 0 }, prime)?;
    }
    let wrong_shift_mutation_residual_entries =
        modular_stage_residuals(&oracle.numerator, &mutated, prime);
    Ok((canaries, wrong_shift_mutation_residual_entries))
}

fn build_device_csr_report(
    binary_file: String,
    binary_sha256: String,
    binary_bytes: usize,
) -> Result<Dg4DeviceCsrReport, String> {
    let csr = dg4_casimir_csr_exact()?;
    let projectors = device_projector_specs()?;
    let eigenvalues = sectors()
        .iter()
        .map(|sector| sector.casimir4_eigenvalue)
        .collect::<Vec<_>>();
    let row_counts = csr
        .row_offsets
        .windows(2)
        .map(|pair| usize::try_from(pair[1] - pair[0]).unwrap())
        .collect::<Vec<_>>();
    let mut row_nonzero_histogram = BTreeMap::new();
    for &count in &row_counts {
        *row_nonzero_histogram.entry(count).or_default() += 1;
    }
    let (parity_canaries, wrong_shift_mutation_residual_entries) =
        build_device_parity_canaries(&csr, &projectors)?;
    let parity_bytes = serde_json::to_vec(&parity_canaries).map_err(|error| error.to_string())?;
    let passed = csr.row_offsets.len() == TARGET_DIMENSION + 1
        && csr.column_indices.len() == csr.exact_values.len()
        && row_counts
            .iter()
            .all(|&count| count == DG4_CASIMIR_ROW_SUPPORT)
        && parity_canaries
            .iter()
            .all(|canary| canary.four_stages_checked == 4 && canary.residual_entries == 0)
        && wrong_shift_mutation_residual_entries > 0;
    Ok(Dg4DeviceCsrReport {
        schema_version: "adynkra-11d-dg4-c4-device-csr-v1",
        target_dimension: TARGET_DIMENSION,
        proof_primes: DG4_PROJECTOR_PROOF_PRIMES,
        module_source_sha256: sha256_hex(MODULE_SOURCE),
        cartesian_basis_sha256: cartesian_basis_sha256(),
        casimir_operator_sha256: casimir_operator_sha256(),
        projector_polynomials_sha256: projector_polynomials_sha256(&eigenvalues),
        row_statistics: Dg4CsrRowStatistics {
            rows: TARGET_DIMENSION,
            nonzeros: csr.exact_values.len(),
            minimum_row_nonzeros: row_counts.iter().copied().min().unwrap_or(0),
            maximum_row_nonzeros: row_counts.iter().copied().max().unwrap_or(0),
            row_nonzero_histogram,
        },
        exact_coefficient_minimum: csr.exact_values.iter().copied().min().unwrap_or(0),
        exact_coefficient_maximum: csr.exact_values.iter().copied().max().unwrap_or(0),
        csr_structure_sha256: hash_csr_structure(&csr),
        exact_coefficients_sha256: hash_exact_coefficients(&csr),
        modular_coefficients_sha256: hash_modular_coefficients(&csr),
        projector_specs_sha256: hash_projector_specs(&projectors),
        binary_file,
        binary_sha256,
        binary_bytes,
        projectors,
        parity_canary_sha256: sha256_hex(&parity_bytes),
        parity_canaries,
        wrong_shift_mutation_residual_entries,
        passed,
        boundary: "The binary freezes exact C4 CSR, prime-major modular coefficients, and five four-shift projector specifications. CPU canaries replay every stage at three primes. Device execution and device-produced pivot certificates remain separate acceptance gates.",
    })
}

pub fn write_device_csr_artifact(path: &Path) -> io::Result<Dg4DeviceCsrReport> {
    let csr = dg4_casimir_csr_exact().map_err(io::Error::other)?;
    let specs = device_projector_specs().map_err(io::Error::other)?;
    let blob = device_csr_blob(&csr, &specs).map_err(io::Error::other)?;
    let binary_path = path.with_extension("csr.bin");
    let binary_file = binary_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid CSR binary path"))?
        .to_string();
    let report = build_device_csr_report(binary_file, sha256_hex(&blob), blob.len())
        .map_err(io::Error::other)?;
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "D G4 device CSR parity certificate did not pass",
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let binary_temporary = binary_path.with_extension(format!("bin.tmp-{}", std::process::id()));
    std::fs::write(&binary_temporary, &blob)?;
    std::fs::rename(&binary_temporary, &binary_path)?;
    let report_temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    std::fs::write(
        &report_temporary,
        serde_json::to_vec_pretty(&report).map_err(io::Error::other)?,
    )?;
    std::fs::rename(&report_temporary, path)?;
    Ok(report)
}

pub fn build_canary() -> Dg4CasimirProjectorCanary {
    let sectors = sectors();
    let eigenvalues = sectors
        .iter()
        .map(|sector| sector.casimir4_eigenvalue)
        .collect::<Vec<_>>();
    let samples = vec![0, 1, 329, 330, 1649, 5279, 10559];
    let mut minimal_residuals = 0;
    let mut sum_residuals = 0;
    let mut eigen_residuals = 0;
    for &ordinal in &samples {
        let basis_integer = BTreeMap::from([(ordinal, 1_i64)]);
        let mut polynomial = basis_integer;
        for &eigenvalue in &eigenvalues {
            polynomial = apply_shift_integer(&polynomial, eigenvalue);
        }
        minimal_residuals += polynomial.len();

        let basis = BTreeMap::from([(ordinal, Ratio::from_integer(1))]);
        let mut sum = SparseRationalVector::new();
        for sector in 0..sectors.len() {
            let projected = apply_projector(&basis, sector, &eigenvalues);
            let c_projected = apply_casimir_rational(&projected);
            for (&row, value) in &projected {
                let residual = c_projected.get(&row).cloned().unwrap_or_default()
                    - value.clone() * Ratio::from_integer(eigenvalues[sector]);
                eigen_residuals += usize::from(residual != Ratio::from_integer(0));
            }
            for (row, value) in projected {
                add_rational(&mut sum, row, value);
            }
        }
        for row in sum
            .keys()
            .chain(std::iter::once(&ordinal))
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
        {
            let expected = if row == ordinal {
                Ratio::from_integer(1)
            } else {
                Ratio::from_integer(0)
            };
            sum_residuals += usize::from(sum.get(&row).cloned().unwrap_or_default() != expected);
        }
    }
    let mut mutated_eigenvalues = eigenvalues.clone();
    mutated_eigenvalues[0] += 1;
    let mut mutation = BTreeMap::from([(0_usize, 1_i64)]);
    for eigenvalue in mutated_eigenvalues {
        mutation = apply_shift_integer(&mutation, eigenvalue);
    }
    let wrong_eigenvalue_mutation_residuals = mutation.len();
    let (exhaustive_minimal_residuals, exhaustive_traces, exhaustive_ranks) =
        exhaustive_spectral_certificate(&eigenvalues);
    let expected_ranks = sectors
        .iter()
        .map(|sector| usize::try_from(sector.expected_dimension).unwrap())
        .collect::<Vec<_>>();
    let columns = casimir_columns();
    let minimum_support = columns.iter().map(Vec::len).min().unwrap_or(0);
    let maximum_support = columns.iter().map(Vec::len).max().unwrap_or(0);
    let dimension_sum = sectors.iter().map(|sector| sector.expected_dimension).sum();
    let passed = dimension_sum == TARGET_DIMENSION as u64
        && minimal_residuals == 0
        && sum_residuals == 0
        && eigen_residuals == 0
        && wrong_eigenvalue_mutation_residuals > 0
        && exhaustive_minimal_residuals == 0
        && exhaustive_ranks == expected_ranks;
    Dg4CasimirProjectorCanary {
        schema_version: "adynkra-11d-dg4-casimir-projector-canary-v1",
        target: "S tensor Lambda4(V) = 00001 + 00011 + 00101 + 01001 + 10001",
        target_dimension: TARGET_DIMENSION,
        module_source_sha256: sha256_hex(MODULE_SOURCE),
        cartesian_basis_sha256: cartesian_basis_sha256(),
        casimir_operator_sha256: casimir_operator_sha256(),
        projector_polynomials_sha256: projector_polynomials_sha256(&eigenvalues),
        sectors,
        sector_dimensions_sum: dimension_sum,
        casimir_columns_constructed: columns.len(),
        minimum_casimir_column_support: minimum_support,
        maximum_casimir_column_support: maximum_support,
        sample_basis_ordinals: samples,
        minimal_polynomial_sample_residuals: minimal_residuals,
        projector_sum_sample_residuals: sum_residuals,
        projector_eigen_sample_residuals: eigen_residuals,
        wrong_eigenvalue_mutation_residuals,
        exhaustive_minimal_polynomial_columns_checked: TARGET_DIMENSION,
        exhaustive_minimal_polynomial_residuals: exhaustive_minimal_residuals,
        exhaustive_projector_traces: exhaustive_traces.iter().map(ToString::to_string).collect(),
        exhaustive_projector_ranks: exhaustive_ranks,
        exhaustive_projector_ranks_constructed: true,
        passed_canary: passed,
        boundary: "The canary checks the Casimir convention and exact spectral projectors on deterministic Cartesian basis samples. The minimal polynomial and projector ranks are exhaustive over all 10,560 Cartesian columns. Explicit serialized projector rows, source-side Hom intertwiners, PBW integrability, Bianchi descent, and GPU row emission remain open.",
    }
}

pub fn write_artifact(path: &Path) -> io::Result<Dg4CasimirProjectorCanary> {
    let report = build_canary();
    if !report.passed_canary || !report.exhaustive_projector_ranks_constructed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "D G4 Casimir projector certificate did not pass",
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(&report).map_err(io::Error::other)?;
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_projectors_reconstruct_and_are_idempotent() {
        let input = BTreeMap::from([
            (0_usize, Ratio::from_integer(1)),
            (329, Ratio::new(-3, 5)),
            (10_559, Ratio::new(7, 11)),
        ]);
        let mut reconstructed = SparseRationalVector::new();
        for label in ["00001", "00011", "00101", "01001", "10001"] {
            let projected = project_dg4_target(label, &input).unwrap();
            let projected_twice = project_dg4_target(label, &projected).unwrap();
            assert_eq!(projected_twice, projected, "idempotence in {label}");
            for (row, value) in projected {
                add_rational(&mut reconstructed, row, value);
            }
        }
        assert_eq!(reconstructed, input);
        assert!(project_dg4_target("bad", &input).is_err());
        assert!(
            project_dg4_target("00001", &BTreeMap::from([(10_560, Ratio::from_integer(1))]))
                .is_err()
        );
    }

    #[test]
    fn exact_casimir_projector_canary_passes() {
        let report = build_canary();
        assert_eq!(report.target_dimension, 10_560);
        assert_eq!(report.sector_dimensions_sum, 10_560);
        assert!(report.passed_canary, "{report:?}");
        assert!(report.exhaustive_projector_ranks_constructed);
        assert_eq!(report.exhaustive_minimal_polynomial_residuals, 0);
        assert!(report.wrong_eigenvalue_mutation_residuals > 0);
        assert_eq!(
            report.exhaustive_projector_ranks,
            vec![32, 5280, 3520, 1408, 320]
        );
    }

    #[test]
    fn narrow_sparse_projector_api_is_exact_and_fail_closed() {
        let seed = BTreeMap::from([(0_usize, Ratio::from_integer(1))]);
        let projected = project_dg4_target("00001", &seed).unwrap();
        assert!(!projected.is_empty());
        assert_eq!(
            casimir_eigen_residual_entries("00001", &projected).unwrap(),
            0
        );
        assert!(project_dg4_target("not-a-sector", &seed).is_err());
        assert!(
            project_dg4_target(
                "00001",
                &BTreeMap::from([(TARGET_DIMENSION, Ratio::from_integer(1))]),
            )
            .is_err()
        );
    }

    #[test]
    fn row_major_casimir_and_four_pass_numerator_match_exact_projector() {
        let rows = dg4_casimir_row_major().unwrap();
        assert_eq!(rows.len(), TARGET_DIMENSION);
        assert!(rows.iter().all(|row| row.len() == DG4_CASIMIR_ROW_SUPPORT));
        let max_coefficient = rows
            .iter()
            .flatten()
            .map(|(_, value)| value.unsigned_abs())
            .max()
            .unwrap();
        eprintln!(
            "DG4_C4_ROW_MAJOR rows={} support={} max_abs={max_coefficient}",
            rows.len(),
            DG4_CASIMIR_ROW_SUPPORT
        );

        let compact = [(0_u16, 2_i64), (329, -3), (10_559, 7)];
        for label in ["00001", "00011", "00101", "01001", "10001"] {
            let oracle = dg4_projector_numerator_oracle(label, &compact).unwrap();
            assert_eq!(oracle.passes.len(), 4);
            assert_eq!(
                oracle
                    .passes
                    .iter()
                    .map(|pass| pass.shift_eigenvalue)
                    .collect::<Vec<_>>(),
                oracle.ordered_shift_eigenvalues
            );
            let input = compact
                .iter()
                .map(|&(row, value)| (usize::from(row), Ratio::from_integer(value)))
                .collect::<BTreeMap<_, _>>();
            let expected = project_dg4_target(label, &input).unwrap();
            let actual = oracle
                .numerator
                .iter()
                .map(|&(row, value)| {
                    let value = i64::try_from(value).expect("canary numerator fits i64");
                    (usize::from(row), Ratio::new(value, oracle.denominator))
                })
                .collect::<BTreeMap<_, _>>();
            assert_eq!(actual, expected, "four-pass mismatch in {label}");
        }
    }

    #[test]
    fn row_major_and_compact_coo_mutations_fail_closed() {
        assert!(dg4_projector_numerator_oracle("00001", &[(2, 1), (1, 1)]).is_err());
        assert!(dg4_projector_numerator_oracle("00001", &[(1, 1), (1, 2)]).is_err());
        assert!(dg4_projector_numerator_oracle("00001", &[(1, 0)]).is_err());
        assert!(dg4_projector_numerator_oracle("bad", &[(1, 1)]).is_err());

        let rows = dg4_casimir_row_major().unwrap();
        let column = usize::from(rows[0][0].0);
        let input = BTreeMap::from([(column, 1_i128)]);
        let expected = apply_casimir_row_major_i128(&input, &rows).unwrap();
        let mut mutated = rows.clone();
        mutated[0][0].1 = mutated[0][0].1.checked_add(1).unwrap();
        let actual = apply_casimir_row_major_i128(&input, &mutated).unwrap();
        assert_ne!(actual, expected, "C4 coefficient mutation was not detected");

        let oracle = dg4_projector_numerator_oracle("00001", &[(0, 1)]).unwrap();
        let mut state = BTreeMap::from([(0_usize, 1_i128)]);
        for (ordinal, &shift) in oracle.ordered_shift_eigenvalues.iter().enumerate() {
            state = apply_shift_row_major_i128(
                &state,
                if ordinal == 0 { shift + 1 } else { shift },
                &rows,
            )
            .unwrap();
        }
        assert_ne!(canonical_i128(&state), oracle.numerator);
    }

    #[test]
    fn device_csr_replays_all_four_projector_stages_at_three_primes() {
        let csr = dg4_casimir_csr_exact().unwrap();
        assert_eq!(csr.row_offsets.len(), TARGET_DIMENSION + 1);
        assert_eq!(
            csr.exact_values.len(),
            TARGET_DIMENSION * DG4_CASIMIR_ROW_SUPPORT
        );
        let specs = device_projector_specs().unwrap();
        let blob = device_csr_blob(&csr, &specs).unwrap();
        let report =
            build_device_csr_report("canary.csr.bin".to_string(), sha256_hex(&blob), blob.len())
                .unwrap();
        assert!(report.passed, "{report:?}");
        assert_eq!(report.row_statistics.nonzeros, 306_240);
        assert_eq!(report.row_statistics.minimum_row_nonzeros, 29);
        assert_eq!(report.row_statistics.maximum_row_nonzeros, 29);
        assert_eq!(report.parity_canaries.len(), 45);
        assert!(
            report
                .parity_canaries
                .iter()
                .all(|canary| canary.residual_entries == 0)
        );
        assert!(report.wrong_shift_mutation_residual_entries > 0);
    }
}
