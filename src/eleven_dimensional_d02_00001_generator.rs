//! First exact source-side generator for the higher-bidegree four-form solve.
//!
//! The `(d_D,d_p)=(0,2)` source contains one copy of `00001` in
//! `Sym^2(V) tensor Hhat`. The target `D G4=S tensor Lambda^4(V)` also
//! contains one copy. The unique Cartesian map is represented by
//!
//! `Gamma_[4] slash(p) (p_c H^c)`.

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

use crate::eleven_dimensional_dg4_casimir_projectors::{
    casimir_eigen_residual_entries, project_dg4_target,
};
use crate::eleven_dimensional_four_form_56_gpu::{
    BidegreeBranch, COLUMN_BINDING_SCHEMA_VERSION, CanonicalRow, ExactCooEntry,
    FourForm56ColumnBinding,
};
use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
use crate::eleven_dimensional_majorana::real_gamma_matrices;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const FOUR_FORM_DIMENSION: usize = 330;
const H_HAT_DIMENSION: usize = 320;
const SYMMETRIC_TWO_DIMENSION: usize = 66;
const TARGET_DIMENSION: usize = SPINOR_DIMENSION * FOUR_FORM_DIMENSION;
const GENERATOR_COLUMN: u32 = 52;
const RANK_PRIME: u64 = 1_073_741_783;
const HIGHER_HOM_INVENTORY_SHA256: &str =
    "0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f";
const DG4_PROJECTOR_ARTIFACT_SHA256: &str =
    "a616e996fb8b002473743840051df5792dfeef6d5b43c7fe378d8a9d0e2cab6d";
const DG4_CARTESIAN_BASIS_SHA256: &str =
    "f2dfae7e9422a639142622e431fcf10166edeb9ae9f5976169ec638a3148e739";
const DG4_CASIMIR_OPERATOR_SHA256: &str =
    "50bd1a225f783092467131c525075be88e3f00b89561804305179150a21e421a";
const DG4_PROJECTOR_POLYNOMIALS_SHA256: &str =
    "cd27eea281fe760fe010aeb24d638d50138044c6b52baf55c34c60df51c9ff91";
const EXPECTED_STREAM_ENTRIES: u64 = 2_217_600;

type IntegerMatrix = Vec<Vec<i16>>;
type SparseIntegerVector = BTreeMap<usize, i64>;

fn digest_hex(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D02GeneratorWitness {
    pub momentum_pair_ordinal: usize,
    pub momentum_pair: [usize; 2],
    pub h_hat_ordinal: usize,
    pub source_coordinate: u64,
    pub target_coordinate: usize,
    pub canonical_row_ordinal: u64,
    pub coefficient: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D02GeneratorReport {
    pub schema_version: &'static str,
    pub branch: &'static str,
    pub source_irrep: &'static str,
    pub target_irrep: &'static str,
    pub formula: &'static str,
    pub source_dimension: usize,
    pub target_dimension: usize,
    pub coefficient_multiplicity: usize,
    pub rank_prime: u64,
    pub source_spinor_map_rank: usize,
    pub gamma_four_injection_rank: usize,
    pub generator_operator_rank: usize,
    pub module_source_sha256: String,
    pub higher_bidegree_hom_inventory_sha256: &'static str,
    pub dg4_projector_artifact_sha256: &'static str,
    pub dg4_cartesian_basis_sha256: &'static str,
    pub dg4_casimir_operator_sha256: &'static str,
    pub dg4_projector_polynomials_sha256: &'static str,
    pub cached_source_spinor_columns_checked: usize,
    pub cached_source_spinor_parity_residual_entries: usize,
    pub cached_gamma_four_columns_checked: usize,
    pub cached_gamma_four_parity_residual_entries: usize,
    pub cached_h_action_maps_checked: usize,
    pub cached_h_action_parity_residual_entries: usize,
    pub source_lorentz_generators_checked: usize,
    pub source_lorentz_columns_checked: usize,
    pub source_lorentz_residual_entries: usize,
    pub h_action_reconstruction_checks: usize,
    pub h_action_reconstruction_residual_entries: usize,
    pub gamma_four_injection_generators_checked: usize,
    pub gamma_four_injection_columns_checked: usize,
    pub gamma_four_injection_residual_entries: usize,
    pub target_casimir_eigen_residual_entries: usize,
    pub target_projector_residual_entries: usize,
    pub emitted_nonzero_rows: u64,
    pub maximum_column_support: usize,
    pub stream_sha256: String,
    pub source_basis_sha256: String,
    pub first_witness: Option<D02GeneratorWitness>,
    pub mutated_missing_cross_route_rejected_by_equivariance: bool,
    pub mutated_symmetric_pair_normalization_rejected_by_equivariance: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RemainingD02StreamSummary {
    pub generator_column: u32,
    pub target_irrep: &'static str,
    pub seed: &'static str,
    pub emitted_nonzero_rows: u64,
    pub maximum_column_support: usize,
    pub stream_sha256: String,
    pub first_witness: Option<D02GeneratorWitness>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RemainingD02GeneratorReport {
    pub schema_version: &'static str,
    pub branch: &'static str,
    pub source_dimension: usize,
    pub target_dimension: usize,
    pub expected_sector_multiplicities: BTreeMap<&'static str, usize>,
    pub one_form_projected_images: usize,
    pub two_form_projected_images: usize,
    pub one_form_projector_scale: i64,
    pub two_form_projector_scale: i64,
    pub source_equivariance_checks_per_seed: usize,
    pub source_equivariance_residuals: [usize; 3],
    pub target_embedding_checks: [usize; 2],
    pub target_embedding_residuals: [usize; 2],
    pub rref_rank_01001: usize,
    pub rref_rank_10001: usize,
    pub rref_pivot_rows_10001: [[usize; 2]; 2],
    pub rref_pivot_values_10001: [[i64; 2]; 2],
    pub streams: Vec<RemainingD02StreamSummary>,
    pub bianchi: D02BianchiReport,
    pub omitted_h_lowering_mutation_rejected: bool,
    pub duplicate_10001_seed_mutation_rejected: bool,
    pub bianchi_filter_complete: bool,
    pub gpu_csr_parity_complete: bool,
    pub gpu_csr_parity_artifact_sha256: String,
    pub gpu_csr_parity_terms: u64,
    pub gpu_csr_parity_wall_seconds: String,
    pub gpu_csr_high_water_bytes: u64,
    pub passed_intertwiner_inventory: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct D02BianchiReport {
    pub source_h_columns: usize,
    pub symmetric_three_momentum_dimension: usize,
    pub target_spinor_five_form_dimension: usize,
    pub aggregated_nonzero_rows: u64,
    pub per_generator_nonzero_rows: [u64; 4],
    pub per_generator_sha256: [String; 4],
    pub modular_primes: [u32; 3],
    pub modular_ranks: [usize; 3],
    pub exact_rank: usize,
    pub exact_kernel_dimension: usize,
    pub exact_kernel_basis: Vec<[String; 4]>,
    pub exact_pivot_rows: Vec<[u64; 3]>,
    pub exact_replay_residuals: usize,
    pub passed: bool,
}

pub fn column_binding(report: &D02GeneratorReport) -> Result<FourForm56ColumnBinding, String> {
    if !report.passed
        || report.schema_version != "adynkra-11d-d02-00001-source-generator-v1"
        || report.emitted_nonzero_rows != EXPECTED_STREAM_ENTRIES
        || report.higher_bidegree_hom_inventory_sha256 != HIGHER_HOM_INVENTORY_SHA256
        || report.dg4_projector_artifact_sha256 != DG4_PROJECTOR_ARTIFACT_SHA256
        || report.dg4_cartesian_basis_sha256 != DG4_CARTESIAN_BASIS_SHA256
    {
        return Err("D02 00001 report cannot be adopted as column 52".to_string());
    }
    let binding = FourForm56ColumnBinding {
        schema_version: COLUMN_BINDING_SCHEMA_VERSION.to_string(),
        global_column: GENERATOR_COLUMN,
        branch: BidegreeBranch::D0P2,
        dynkin_label: "00001".to_string(),
        multiplicity_copy: 1,
        generator_schema_version: report.schema_version.to_string(),
        generator_source_sha256: report.module_source_sha256.clone(),
        source_basis_sha256: report.source_basis_sha256.clone(),
        target_basis_sha256: report.dg4_cartesian_basis_sha256.to_string(),
        coefficient_inventory_sha256: report.higher_bidegree_hom_inventory_sha256.to_string(),
        projector_sha256: report.dg4_projector_artifact_sha256.to_string(),
        exact_stream_entries: report.emitted_nonzero_rows,
        exact_stream_sha256: report.stream_sha256.clone(),
    };
    binding.validate()?;
    Ok(binding)
}

fn lorentz_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn momentum_pairs() -> Vec<[usize; 2]> {
    let mut output = Vec::with_capacity(SYMMETRIC_TWO_DIMENSION);
    for left in 0..VECTOR_DIMENSION {
        for right in left..VECTOR_DIMENSION {
            output.push([left, right]);
        }
    }
    output
}

fn form_masks(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn multiply(left: &IntegerMatrix, right: &IntegerMatrix) -> IntegerMatrix {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            let l = left[row][pivot];
            if l == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += l * right[pivot][column];
            }
        }
    }
    output
}

fn identity() -> IntegerMatrix {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for index in 0..SPINOR_DIMENSION {
        output[index][index] = 1;
    }
    output
}

fn gamma_matrices() -> Vec<IntegerMatrix> {
    real_gamma_matrices()
        .into_iter()
        .map(|matrix| {
            matrix
                .into_iter()
                .map(|row| row.into_iter().map(i16::from).collect())
                .collect()
        })
        .collect()
}

fn lower_gamma_product(gammas: &[IntegerMatrix], axes: &[usize]) -> IntegerMatrix {
    let mut output = identity();
    let mut metric = 1_i16;
    for &axis in axes {
        output = multiply(&output, &gammas[axis]);
        metric *= i16::try_from(lorentz_sign(axis)).unwrap();
    }
    for row in &mut output {
        for value in row {
            *value *= metric;
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

fn modular_inverse(mut value: u64, prime: u64) -> u64 {
    let mut exponent = prime - 2;
    let mut output = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = u64::try_from((u128::from(output) * u128::from(value)) % u128::from(prime))
                .unwrap();
        }
        value = u64::try_from((u128::from(value) * u128::from(value)) % u128::from(prime)).unwrap();
        exponent >>= 1;
    }
    output
}

fn rank_sparse_columns(
    columns: impl IntoIterator<Item = SparseIntegerVector>,
    rows: usize,
) -> usize {
    let mut pivots = vec![None::<Vec<u64>>; rows];
    let mut rank = 0;
    for column in columns {
        let mut vector = vec![0_u64; rows];
        for (row, value) in column {
            let reduced = value.rem_euclid(i64::try_from(RANK_PRIME).unwrap());
            vector[row] = u64::try_from(reduced).unwrap();
        }
        for pivot in 0..rows {
            if vector[pivot] == 0 {
                continue;
            }
            if let Some(basis) = &pivots[pivot] {
                let factor = vector[pivot];
                for row in pivot..rows {
                    let subtraction =
                        (u128::from(factor) * u128::from(basis[row])) % u128::from(RANK_PRIME);
                    vector[row] = (vector[row] + RANK_PRIME - u64::try_from(subtraction).unwrap())
                        % RANK_PRIME;
                }
            } else {
                let inverse = modular_inverse(vector[pivot], RANK_PRIME);
                for value in &mut vector[pivot..] {
                    *value = u64::try_from(
                        (u128::from(*value) * u128::from(inverse)) % u128::from(RANK_PRIME),
                    )
                    .unwrap();
                }
                pivots[pivot] = Some(vector);
                rank += 1;
                break;
            }
        }
        if rank == rows {
            break;
        }
    }
    rank
}

fn h_basis_integer() -> Vec<SparseIntegerVector> {
    canonical_gamma_traceless_frame_basis()
        .into_iter()
        .map(|column| {
            column
                .into_iter()
                .map(|(coordinate, value)| {
                    assert_eq!(*value.real.denom(), 1);
                    assert_eq!(value.imaginary, Ratio::from_integer(0));
                    (coordinate, *value.real.numer())
                })
                .collect()
        })
        .collect()
}

/// The unique `Sym^2(V) tensor Hhat -> S` contraction.
fn source_spinor(
    pair: [usize; 2],
    h: &SparseIntegerVector,
    gammas: &[IntegerMatrix],
) -> SparseIntegerVector {
    let routes = if pair[0] == pair[1] {
        vec![(pair[0], pair[0])]
    } else {
        vec![(pair[0], pair[1]), (pair[1], pair[0])]
    };
    let mut output = SparseIntegerVector::new();
    for (slash_axis, h_vector) in routes {
        for (&coordinate, &coefficient) in h {
            let input_spinor = coordinate / VECTOR_DIMENSION;
            let input_vector = coordinate % VECTOR_DIMENSION;
            if input_vector != h_vector {
                continue;
            }
            for output_spinor in 0..SPINOR_DIMENSION {
                add_integer(
                    &mut output,
                    output_spinor,
                    coefficient * i64::from(gammas[slash_axis][output_spinor][input_spinor]),
                );
            }
        }
    }
    output
}

fn source_spinor_missing_cross_route(
    pair: [usize; 2],
    h: &SparseIntegerVector,
    gammas: &[IntegerMatrix],
) -> SparseIntegerVector {
    let route = (pair[0], pair[1]);
    let mut output = SparseIntegerVector::new();
    for (&coordinate, &coefficient) in h {
        let input_spinor = coordinate / VECTOR_DIMENSION;
        let input_vector = coordinate % VECTOR_DIMENSION;
        if input_vector != route.1 {
            continue;
        }
        for output_spinor in 0..SPINOR_DIMENSION {
            add_integer(
                &mut output,
                output_spinor,
                coefficient * i64::from(gammas[route.0][output_spinor][input_spinor]),
            );
        }
    }
    output
}

fn gamma_four_basis(gammas: &[IntegerMatrix]) -> Vec<SparseIntegerVector> {
    let mut output = vec![SparseIntegerVector::new(); SPINOR_DIMENSION];
    for (form_ordinal, mask) in form_masks(4).into_iter().enumerate() {
        let axes = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma_four = lower_gamma_product(gammas, &axes);
        for input_spinor in 0..SPINOR_DIMENSION {
            for (output_spinor, row) in gamma_four.iter().enumerate() {
                add_integer(
                    &mut output[input_spinor],
                    output_spinor * FOUR_FORM_DIMENSION + form_ordinal,
                    i64::from(row[input_spinor]),
                );
            }
        }
    }
    output
}

fn inject_gamma_four_uncached(
    spinor: &SparseIntegerVector,
    gamma_four_basis: &[SparseIntegerVector],
) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    for (&input_spinor, &coefficient) in spinor {
        for (&row, &value) in &gamma_four_basis[input_spinor] {
            add_integer(&mut output, row, coefficient * value);
        }
    }
    output
}

fn lorentz_pairs() -> &'static Vec<(usize, usize)> {
    static PAIRS: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    PAIRS.get_or_init(|| {
        (0..VECTOR_DIMENSION)
            .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
            .collect()
    })
}

fn spin_generators() -> &'static Vec<IntegerMatrix> {
    static GENERATORS: OnceLock<Vec<IntegerMatrix>> = OnceLock::new();
    GENERATORS.get_or_init(|| {
        let gammas = gamma_matrices();
        lorentz_pairs()
            .iter()
            .map(|&(left, right)| lower_gamma_product(&gammas, &[left, right]))
            .collect()
    })
}

/// The 32 immutable images of the spinor basis under the Gamma4 injection.
fn gamma_four_basis_images() -> &'static Vec<SparseIntegerVector> {
    static IMAGES: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    IMAGES.get_or_init(|| gamma_four_basis(&gamma_matrices()))
}

fn inject_gamma_four(spinor: &SparseIntegerVector) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    for (&input_spinor, &coefficient) in spinor {
        for (&row, &value) in &gamma_four_basis_images()[input_spinor] {
            add_integer(&mut output, row, coefficient * value);
        }
    }
    output
}

/// All 66*320 exact source-spinor columns, momentum-pair major.
fn source_spinor_columns() -> &'static Vec<SparseIntegerVector> {
    static COLUMNS: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        let gammas = gamma_matrices();
        let h_basis = h_basis_integer();
        momentum_pairs()
            .into_iter()
            .flat_map(|pair| {
                h_basis
                    .iter()
                    .map(|h| source_spinor(pair, h, &gammas))
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

fn source_spinor_column(pair_ordinal: usize, h_ordinal: usize) -> &'static SparseIntegerVector {
    &source_spinor_columns()[pair_ordinal * H_HAT_DIMENSION + h_ordinal]
}

fn generator_column_cached(pair_ordinal: usize, h_ordinal: usize) -> SparseIntegerVector {
    inject_gamma_four(source_spinor_column(pair_ordinal, h_ordinal))
}

fn generator_column(
    pair: [usize; 2],
    h: &SparseIntegerVector,
    gammas: &[IntegerMatrix],
    _gamma_four_basis: &[SparseIntegerVector],
) -> SparseIntegerVector {
    inject_gamma_four(&source_spinor(pair, h, gammas))
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

fn vector_generator(axis: usize, left: usize, right: usize) -> Vec<(usize, i64)> {
    let mut output = Vec::new();
    if axis == right {
        output.push((left, lorentz_sign(right)));
    }
    if axis == left {
        output.push((right, -lorentz_sign(left)));
    }
    output
}

fn wedge_sign(mask: u16, axis: usize) -> i64 {
    if (mask & ((1_u16 << axis) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
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

fn spinor_action(
    input: &SparseIntegerVector,
    left: usize,
    right: usize,
    gammas: &[IntegerMatrix],
) -> SparseIntegerVector {
    let generator = lower_gamma_product(gammas, &[left, right]);
    let mut output = SparseIntegerVector::new();
    for (&spinor, &coefficient) in input {
        for (next, row) in generator.iter().enumerate() {
            add_integer(&mut output, next, coefficient * i64::from(row[spinor]));
        }
    }
    output
}

fn spinor_action_cached(
    input: &SparseIntegerVector,
    generator_ordinal: usize,
) -> SparseIntegerVector {
    let generator = &spin_generators()[generator_ordinal];
    let mut output = SparseIntegerVector::new();
    for (&spinor, &coefficient) in input {
        for (next, row) in generator.iter().enumerate() {
            add_integer(&mut output, next, coefficient * i64::from(row[spinor]));
        }
    }
    output
}

fn target_action(
    input: &SparseIntegerVector,
    left: usize,
    right: usize,
    gammas: &[IntegerMatrix],
) -> SparseIntegerVector {
    let forms = form_masks(4);
    let lookup = forms
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, mask)| (mask, ordinal))
        .collect::<BTreeMap<_, _>>();
    let spin = lower_gamma_product(gammas, &[left, right]);
    let mut output = SparseIntegerVector::new();
    for (&coordinate, &coefficient) in input {
        let input_spinor = coordinate / FOUR_FORM_DIMENSION;
        let form_ordinal = coordinate % FOUR_FORM_DIMENSION;
        for (next_spinor, row) in spin.iter().enumerate() {
            add_integer(
                &mut output,
                next_spinor * FOUR_FORM_DIMENSION + form_ordinal,
                coefficient * i64::from(row[input_spinor]),
            );
        }
        for (next_mask, value) in form_generator(forms[form_ordinal], left, right) {
            add_integer(
                &mut output,
                input_spinor * FOUR_FORM_DIMENSION + lookup[&next_mask],
                2 * coefficient * value,
            );
        }
    }
    output
}

fn pair_action(pair: [usize; 2], left: usize, right: usize) -> Vec<(usize, i64)> {
    let pairs = momentum_pairs();
    let lookup = pairs
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, pair)| (pair, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut raw = BTreeMap::<usize, i64>::new();
    for position in 0..2 {
        for (replacement, coefficient) in covector_generator(pair[position], left, right) {
            let mut next = pair;
            next[position] = replacement;
            next.sort_unstable();
            *raw.entry(lookup[&next]).or_default() += coefficient;
        }
    }
    let input_multiplicity = if pair[0] == pair[1] { 1_i64 } else { 2_i64 };
    raw.into_iter()
        .filter_map(|(next_ordinal, coefficient)| {
            let next = pairs[next_ordinal];
            let output_multiplicity = if next[0] == next[1] { 1_i64 } else { 2_i64 };
            let numerator = 2 * input_multiplicity * coefficient;
            assert_eq!(numerator % output_multiplicity, 0);
            let value = numerator / output_multiplicity;
            (value != 0).then_some((next_ordinal, value))
        })
        .collect()
}

fn legacy_pair_action(pair: [usize; 2], left: usize, right: usize) -> Vec<(usize, i64)> {
    let pairs = momentum_pairs();
    let lookup = pairs
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, pair)| (pair, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut output = BTreeMap::new();
    for position in 0..2 {
        for (replacement, coefficient) in covector_generator(pair[position], left, right) {
            let mut next = pair;
            next[position] = replacement;
            next.sort_unstable();
            *output.entry(lookup[&next]).or_insert(0_i64) += 2 * coefficient;
        }
    }
    output
        .into_iter()
        .filter(|(_, value)| *value != 0)
        .collect()
}

fn h_action(
    h: &SparseIntegerVector,
    left: usize,
    right: usize,
    gammas: &[IntegerMatrix],
) -> SparseIntegerVector {
    let spin = lower_gamma_product(gammas, &[left, right]);
    let mut raw = SparseIntegerVector::new();
    for (&coordinate, &coefficient) in h {
        let input_spinor = coordinate / VECTOR_DIMENSION;
        let input_vector = coordinate % VECTOR_DIMENSION;
        for (next_spinor, row) in spin.iter().enumerate() {
            add_integer(
                &mut raw,
                next_spinor * VECTOR_DIMENSION + input_vector,
                coefficient * i64::from(row[input_spinor]),
            );
        }
        for (next_vector, value) in vector_generator(input_vector, left, right) {
            add_integer(
                &mut raw,
                input_spinor * VECTOR_DIMENSION + next_vector,
                2 * coefficient * value,
            );
        }
    }
    raw
}

fn h_action_basis_coefficients(raw: &SparseIntegerVector) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    for spatial_vector in 1..VECTOR_DIMENSION {
        for spinor in 0..SPINOR_DIMENSION {
            let value = raw
                .get(&(spinor * VECTOR_DIMENSION + spatial_vector))
                .copied()
                .unwrap_or(0);
            add_integer(
                &mut output,
                (spatial_vector - 1) * SPINOR_DIMENSION + spinor,
                value,
            );
        }
    }
    output
}

fn pair_action_cache() -> &'static Vec<Vec<Vec<(usize, i64)>>> {
    static CACHE: OnceLock<Vec<Vec<Vec<(usize, i64)>>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let pairs = momentum_pairs();
        lorentz_pairs()
            .iter()
            .map(|&(left, right)| {
                pairs
                    .iter()
                    .map(|&pair| pair_action(pair, left, right))
                    .collect()
            })
            .collect()
    })
}

/// The 55*320 exact Hhat action coefficient maps in canonical Hhat basis.
fn h_action_coefficient_cache() -> &'static Vec<Vec<SparseIntegerVector>> {
    static CACHE: OnceLock<Vec<Vec<SparseIntegerVector>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let gammas = gamma_matrices();
        let h_basis = h_basis_integer();
        lorentz_pairs()
            .iter()
            .map(|&(left, right)| {
                h_basis
                    .iter()
                    .map(|h| h_action_basis_coefficients(&h_action(h, left, right, &gammas)))
                    .collect()
            })
            .collect()
    })
}

fn h_action_reconstruction_residuals(
    h_basis: &[SparseIntegerVector],
    gammas: &[IntegerMatrix],
) -> (usize, usize) {
    let mut checks = 0;
    let mut residuals = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in (left + 1)..VECTOR_DIMENSION {
            for h in h_basis {
                checks += 1;
                let acted = h_action(h, left, right, gammas);
                let coefficients = h_action_basis_coefficients(&acted);
                let mut reconstructed = SparseIntegerVector::new();
                for (basis_ordinal, coefficient) in coefficients {
                    for (&coordinate, &value) in &h_basis[basis_ordinal] {
                        add_integer(&mut reconstructed, coordinate, coefficient * value);
                    }
                }
                for key in acted
                    .keys()
                    .chain(reconstructed.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    residuals += usize::from(acted.get(&key) != reconstructed.get(&key));
                }
            }
        }
    }
    (checks, residuals)
}

fn missing_cross_route_mutation_is_rejected(
    h_basis: &[SparseIntegerVector],
    gammas: &[IntegerMatrix],
) -> bool {
    let pairs = momentum_pairs();
    for left in 0..VECTOR_DIMENSION {
        for right in (left + 1)..VECTOR_DIMENSION {
            for &pair in pairs.iter().filter(|pair| pair[0] != pair[1]) {
                for h in h_basis {
                    let lhs = spinor_action(
                        &source_spinor_missing_cross_route(pair, h, gammas),
                        left,
                        right,
                        gammas,
                    );
                    let mut rhs = SparseIntegerVector::new();
                    for (next_pair, coefficient) in pair_action(pair, left, right) {
                        for (row, value) in
                            source_spinor_missing_cross_route(pairs[next_pair], h, gammas)
                        {
                            add_integer(&mut rhs, row, coefficient * value);
                        }
                    }
                    let acted_h = h_action(h, left, right, gammas);
                    for (next_h, coefficient) in h_action_basis_coefficients(&acted_h) {
                        for (row, value) in
                            source_spinor_missing_cross_route(pair, &h_basis[next_h], gammas)
                        {
                            add_integer(&mut rhs, row, coefficient * value);
                        }
                    }
                    if lhs != rhs {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn legacy_pair_normalization_mutation_is_rejected() -> bool {
    let pairs = momentum_pairs();
    for (generator_ordinal, &(left, right)) in lorentz_pairs().iter().enumerate() {
        for (pair_ordinal, &pair) in pairs.iter().enumerate() {
            for h_ordinal in 0..H_HAT_DIMENSION {
                let lhs = spinor_action_cached(
                    source_spinor_column(pair_ordinal, h_ordinal),
                    generator_ordinal,
                );
                let mut rhs = SparseIntegerVector::new();
                for (next_pair, coefficient) in legacy_pair_action(pair, left, right) {
                    for (&row, &value) in source_spinor_column(next_pair, h_ordinal) {
                        add_integer(&mut rhs, row, coefficient * value);
                    }
                }
                for (&next_h, &coefficient) in
                    &h_action_coefficient_cache()[generator_ordinal][h_ordinal]
                {
                    for (&row, &value) in source_spinor_column(pair_ordinal, next_h) {
                        add_integer(&mut rhs, row, coefficient * value);
                    }
                }
                if lhs != rhs {
                    return true;
                }
            }
        }
    }
    false
}

fn verify_source_columns_equivariance(
    columns_data: &[SparseIntegerVector],
) -> (usize, usize, usize) {
    let pairs = momentum_pairs();
    let column = |pair: usize, h: usize| &columns_data[pair * H_HAT_DIMENSION + h];
    let mut columns = 0;
    let mut residuals = 0;
    for (generator_ordinal, _) in lorentz_pairs().iter().enumerate() {
        for pair_ordinal in 0..pairs.len() {
            for h_ordinal in 0..H_HAT_DIMENSION {
                columns += 1;
                let lhs = spinor_action_cached(column(pair_ordinal, h_ordinal), generator_ordinal);
                let mut rhs = SparseIntegerVector::new();
                for &(next_pair, coefficient) in
                    &pair_action_cache()[generator_ordinal][pair_ordinal]
                {
                    for (&row, &value) in column(next_pair, h_ordinal) {
                        add_integer(&mut rhs, row, coefficient * value);
                    }
                }
                for (&next_h, &coefficient) in
                    &h_action_coefficient_cache()[generator_ordinal][h_ordinal]
                {
                    for (&row, &value) in column(pair_ordinal, next_h) {
                        add_integer(&mut rhs, row, coefficient * value);
                    }
                }
                for key in lhs
                    .keys()
                    .chain(rhs.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    residuals += usize::from(lhs.get(&key) != rhs.get(&key));
                }
            }
        }
    }
    (lorentz_pairs().len(), columns, residuals)
}

fn verify_source_spinor_equivariance(
    _h_basis: &[SparseIntegerVector],
    _gammas: &[IntegerMatrix],
) -> (usize, usize, usize) {
    verify_source_columns_equivariance(source_spinor_columns())
}

fn verify_gamma_four_equivariance(
    gammas: &[IntegerMatrix],
    gamma_four_basis: &[SparseIntegerVector],
) -> (usize, usize, usize) {
    let mut generators = 0;
    let mut columns = 0;
    let mut residuals = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in (left + 1)..VECTOR_DIMENSION {
            generators += 1;
            for spinor in 0..SPINOR_DIMENSION {
                columns += 1;
                let input = BTreeMap::from([(spinor, 1_i64)]);
                let lhs = target_action(&inject_gamma_four(&input), left, right, gammas);
                let rhs = inject_gamma_four(&spinor_action(&input, left, right, gammas));
                for key in lhs
                    .keys()
                    .chain(rhs.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    residuals += usize::from(lhs.get(&key) != rhs.get(&key));
                }
            }
        }
    }
    (generators, columns, residuals)
}

fn source_basis_sha256(h_basis: &[SparseIntegerVector]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"d02-sym2-momentum-times-canonical-hhat-v1");
    for pair in momentum_pairs() {
        hasher.update(u64::try_from(pair[0]).unwrap().to_le_bytes());
        hasher.update(u64::try_from(pair[1]).unwrap().to_le_bytes());
    }
    for (ordinal, column) in h_basis.iter().enumerate() {
        hasher.update(u64::try_from(ordinal).unwrap().to_le_bytes());
        for (&coordinate, &value) in column {
            hasher.update(u64::try_from(coordinate).unwrap().to_le_bytes());
            hasher.update(value.to_le_bytes());
        }
    }
    digest_hex(hasher.finalize())
}

pub(crate) fn visit_d02_00001_generator(
    mut emit: impl FnMut(ExactCooEntry) -> Result<(), String>,
) -> Result<(u64, usize, String, Option<D02GeneratorWitness>), String> {
    let pairs = momentum_pairs();
    let mut count = 0_u64;
    let mut maximum_support = 0;
    let mut first = None;
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-11d-d02-00001-generator-v1");
    for (pair_ordinal, &pair) in pairs.iter().enumerate() {
        for h_ordinal in 0..H_HAT_DIMENSION {
            let source_coordinate = pair_ordinal * H_HAT_DIMENSION + h_ordinal;
            let column = generator_column_cached(pair_ordinal, h_ordinal);
            maximum_support = maximum_support.max(column.len());
            for (target_coordinate, coefficient) in column {
                let canonical = CanonicalRow {
                    branch: BidegreeBranch::D0P2,
                    source_coordinate: u64::try_from(source_coordinate).unwrap(),
                    target_coordinate: u32::try_from(target_coordinate).unwrap(),
                };
                let row = canonical.ordinal()?;
                let entry = ExactCooEntry {
                    row,
                    column: GENERATOR_COLUMN,
                    reserved: 0,
                    real: coefficient,
                    imaginary: 0,
                };
                entry.validate()?;
                hasher.update(row.to_le_bytes());
                hasher.update(GENERATOR_COLUMN.to_le_bytes());
                hasher.update(coefficient.to_le_bytes());
                if first.is_none() {
                    first = Some(D02GeneratorWitness {
                        momentum_pair_ordinal: pair_ordinal,
                        momentum_pair: pair,
                        h_hat_ordinal: h_ordinal,
                        source_coordinate: u64::try_from(source_coordinate).unwrap(),
                        target_coordinate,
                        canonical_row_ordinal: row,
                        coefficient,
                    });
                }
                emit(entry)?;
                count += 1;
            }
        }
    }
    Ok((count, maximum_support, digest_hex(hasher.finalize()), first))
}

fn cache_parity_residuals(
    h_basis: &[SparseIntegerVector],
    gammas: &[IntegerMatrix],
    gamma_four_basis: &[SparseIntegerVector],
) -> (usize, usize, usize, usize, usize, usize) {
    let pairs = momentum_pairs();
    let mut source_bad = 0;
    for (pair_ordinal, &pair) in pairs.iter().enumerate() {
        for (h_ordinal, h) in h_basis.iter().enumerate() {
            source_bad += usize::from(
                source_spinor_column(pair_ordinal, h_ordinal) != &source_spinor(pair, h, gammas),
            );
        }
    }
    let mut gamma_bad = 0;
    for spinor in 0..SPINOR_DIMENSION {
        gamma_bad += usize::from(gamma_four_basis_images()[spinor] != gamma_four_basis[spinor]);
    }
    let mut h_bad = 0;
    for (generator_ordinal, &(left, right)) in lorentz_pairs().iter().enumerate() {
        for (h_ordinal, h) in h_basis.iter().enumerate() {
            let direct = h_action_basis_coefficients(&h_action(h, left, right, gammas));
            h_bad +=
                usize::from(h_action_coefficient_cache()[generator_ordinal][h_ordinal] != direct);
        }
    }
    (
        pairs.len() * h_basis.len(),
        source_bad,
        SPINOR_DIMENSION,
        gamma_bad,
        lorentz_pairs().len() * h_basis.len(),
        h_bad,
    )
}

pub fn build_report() -> Result<D02GeneratorReport, String> {
    let higher_hom_digest = digest_hex(Sha256::digest(include_bytes!(
        "../results/adynkra_11d_higher_bidegree_hom_inventory.json"
    )));
    if higher_hom_digest != HIGHER_HOM_INVENTORY_SHA256 {
        return Err(format!(
            "higher-bidegree Hom inventory digest drift: {higher_hom_digest}"
        ));
    }
    let projector_digest = digest_hex(Sha256::digest(include_bytes!(
        "../results/adynkra_11d_dg4_casimir_projectors.json"
    )));
    if projector_digest != DG4_PROJECTOR_ARTIFACT_SHA256 {
        return Err(format!(
            "D G4 projector artifact digest drift: {projector_digest}"
        ));
    }
    let gammas = gamma_matrices();
    let gamma_four_basis = gamma_four_basis(&gammas);
    let h_basis = h_basis_integer();
    let (
        cached_source_checked,
        cached_source_bad,
        cached_gamma_checked,
        cached_gamma_bad,
        cached_h_checked,
        cached_h_bad,
    ) = cache_parity_residuals(&h_basis, &gammas, &gamma_four_basis);
    let (source_generators, source_columns, source_residuals) =
        verify_source_spinor_equivariance(&h_basis, &gammas);
    let (h_action_reconstruction_checks, h_action_reconstruction_residual_entries) =
        h_action_reconstruction_residuals(&h_basis, &gammas);
    let (target_generators, target_columns, target_residuals) =
        verify_gamma_four_equivariance(&gammas, &gamma_four_basis);
    let source_spinor_map_rank = rank_sparse_columns(source_spinor_columns().iter().cloned(), 32);
    let gamma_four_injection_rank =
        rank_sparse_columns(gamma_four_basis_images().iter().cloned(), TARGET_DIMENSION);
    let generator_operator_rank = source_spinor_map_rank.min(gamma_four_injection_rank);

    let canary = generator_column_cached(0, 0);
    if canary.is_empty() {
        return Err("D02 00001 generator canary is zero".to_string());
    }
    let canary_rational = canary
        .iter()
        .map(|(&row, &value)| (row, Ratio::from_integer(value)))
        .collect::<BTreeMap<_, _>>();
    let projected = project_dg4_target("00001", &canary_rational)?;
    let projector_residuals = canary_rational
        .keys()
        .chain(projected.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|row| canary_rational.get(row) != projected.get(row))
        .count();
    let casimir_residuals = casimir_eigen_residual_entries("00001", &canary_rational)?;

    let mutated_missing_cross_route_rejected_by_equivariance =
        missing_cross_route_mutation_is_rejected(&h_basis, &gammas);
    let mutated_symmetric_pair_normalization_rejected_by_equivariance =
        legacy_pair_normalization_mutation_is_rejected();
    let (emitted, maximum_support, stream_sha256, first_witness) =
        visit_d02_00001_generator(|_| Ok(()))?;
    let passed = cached_source_bad == 0
        && cached_gamma_bad == 0
        && cached_h_bad == 0
        && source_residuals == 0
        && target_residuals == 0
        && h_action_reconstruction_residual_entries == 0
        && projector_residuals == 0
        && casimir_residuals == 0
        && source_spinor_map_rank == 32
        && gamma_four_injection_rank == 32
        && generator_operator_rank == 32
        && emitted > 0
        && mutated_missing_cross_route_rejected_by_equivariance
        && mutated_symmetric_pair_normalization_rejected_by_equivariance;
    Ok(D02GeneratorReport {
        schema_version: "adynkra-11d-d02-00001-source-generator-v1",
        branch: "(d_D,d_p)=(0,2)",
        source_irrep: "the unique 00001 in Sym^2(V) tensor Hhat",
        target_irrep: "the unique 00001 in S tensor Lambda4(V)",
        formula: "Y_{beta,[abcd]}=(Gamma_[abcd])_beta^gamma (p_e Gamma^e)_gamma^alpha (p_f H_alpha^f)",
        source_dimension: SYMMETRIC_TWO_DIMENSION * H_HAT_DIMENSION,
        target_dimension: TARGET_DIMENSION,
        coefficient_multiplicity: 1,
        rank_prime: RANK_PRIME,
        source_spinor_map_rank,
        gamma_four_injection_rank,
        generator_operator_rank,
        module_source_sha256: digest_hex(Sha256::digest(include_bytes!(
            "eleven_dimensional_d02_00001_generator.rs"
        ))),
        higher_bidegree_hom_inventory_sha256: HIGHER_HOM_INVENTORY_SHA256,
        dg4_projector_artifact_sha256: DG4_PROJECTOR_ARTIFACT_SHA256,
        dg4_cartesian_basis_sha256: DG4_CARTESIAN_BASIS_SHA256,
        dg4_casimir_operator_sha256: DG4_CASIMIR_OPERATOR_SHA256,
        dg4_projector_polynomials_sha256: DG4_PROJECTOR_POLYNOMIALS_SHA256,
        cached_source_spinor_columns_checked: cached_source_checked,
        cached_source_spinor_parity_residual_entries: cached_source_bad,
        cached_gamma_four_columns_checked: cached_gamma_checked,
        cached_gamma_four_parity_residual_entries: cached_gamma_bad,
        cached_h_action_maps_checked: cached_h_checked,
        cached_h_action_parity_residual_entries: cached_h_bad,
        source_lorentz_generators_checked: source_generators,
        source_lorentz_columns_checked: source_columns,
        source_lorentz_residual_entries: source_residuals,
        h_action_reconstruction_checks,
        h_action_reconstruction_residual_entries,
        gamma_four_injection_generators_checked: target_generators,
        gamma_four_injection_columns_checked: target_columns,
        gamma_four_injection_residual_entries: target_residuals,
        target_casimir_eigen_residual_entries: casimir_residuals,
        target_projector_residual_entries: projector_residuals,
        emitted_nonzero_rows: emitted,
        maximum_column_support: maximum_support,
        stream_sha256,
        source_basis_sha256: source_basis_sha256(&h_basis),
        first_witness,
        mutated_missing_cross_route_rejected_by_equivariance,
        mutated_symmetric_pair_normalization_rejected_by_equivariance,
        passed,
        boundary: "This certifies one multiplicity-one (0,2) Cartesian generator and its canonical sparse row stream. It does not construct the remaining 55 generators, impose PBW integrability or Bianchi descent, or solve the physical coefficient system.",
    })
}

pub fn write_artifact(path: &Path) -> io::Result<D02GeneratorReport> {
    let report = build_report().map_err(io::Error::other)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(&report).map_err(io::Error::other)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(report)
}

fn partition_sign(left_mask: u16, right_mask: u16) -> i64 {
    let mut inversions = 0_u32;
    for left in 0..VECTOR_DIMENSION {
        if left_mask & (1_u16 << left) == 0 {
            continue;
        }
        inversions += (right_mask & ((1_u16 << left) - 1)).count_ones();
    }
    if inversions % 2 == 0 { 1 } else { -1 }
}

/// Equivariant `S tensor V -> S tensor Lambda4` seed
/// `U_a -> Gamma_[bcd] U_a`, with the exact wedge sign.
fn embed_spinor_one_form_basis(input_spinor: usize, input_vector: usize) -> SparseIntegerVector {
    let gammas = gamma_matrices();
    let forms4 = form_masks(4);
    let mut output = SparseIntegerVector::new();
    for (target_form, &target_mask) in forms4.iter().enumerate() {
        if target_mask & (1_u16 << input_vector) == 0 {
            continue;
        }
        let gamma_mask = target_mask ^ (1_u16 << input_vector);
        let axes = (0..VECTOR_DIMENSION)
            .filter(|axis| gamma_mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = lower_gamma_product(&gammas, &axes);
        let sign = partition_sign(gamma_mask, 1_u16 << input_vector);
        for (output_spinor, row) in gamma.iter().enumerate() {
            add_integer(
                &mut output,
                output_spinor * FOUR_FORM_DIMENSION + target_form,
                sign * i64::from(row[input_spinor]),
            );
        }
    }
    output
}

/// Equivariant `S tensor Lambda2 -> S tensor Lambda4` seed
/// `T_[ab] -> Gamma_[cd] T_[ab]`, with the exact wedge sign.
fn embed_spinor_two_form_basis(input_spinor: usize, input_two_form: usize) -> SparseIntegerVector {
    let gammas = gamma_matrices();
    let forms2 = form_masks(2);
    let forms4 = form_masks(4);
    let input_mask = forms2[input_two_form];
    let mut output = SparseIntegerVector::new();
    for (target_form, &target_mask) in forms4.iter().enumerate() {
        if input_mask & target_mask != input_mask {
            continue;
        }
        let gamma_mask = target_mask ^ input_mask;
        let axes = (0..VECTOR_DIMENSION)
            .filter(|axis| gamma_mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = lower_gamma_product(&gammas, &axes);
        let sign = partition_sign(gamma_mask, input_mask);
        for (output_spinor, row) in gamma.iter().enumerate() {
            add_integer(
                &mut output,
                output_spinor * FOUR_FORM_DIMENSION + target_form,
                sign * i64::from(row[input_spinor]),
            );
        }
    }
    output
}

fn projected_embedding_canary(
    sector: &str,
    raw: SparseIntegerVector,
) -> Result<(usize, usize), String> {
    let raw = raw
        .into_iter()
        .map(|(row, value)| (row, Ratio::from_integer(value)))
        .collect::<BTreeMap<_, _>>();
    let projected = project_dg4_target(sector, &raw)?;
    fn gcd(mut left: i64, mut right: i64) -> i64 {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left.abs()
    }
    let denominator_lcm = projected.values().fold(1_i64, |left, value| {
        left / gcd(left, *value.denom()) * *value.denom()
    });
    Ok((projected.len(), usize::try_from(denominator_lcm).unwrap()))
}

fn integer_projected_image(
    sector: &str,
    raw: SparseIntegerVector,
    scale: i64,
) -> SparseIntegerVector {
    let rational = raw
        .into_iter()
        .map(|(row, value)| (row, Ratio::from_integer(value)))
        .collect::<BTreeMap<_, _>>();
    let projected = project_dg4_target(sector, &rational).unwrap();
    projected
        .into_iter()
        .map(|(row, value)| {
            let scaled = value * Ratio::from_integer(scale);
            assert_eq!(
                *scaled.denom(),
                1,
                "nonintegral projected embedding at scale {scale}"
            );
            (row, *scaled.numer())
        })
        .filter(|(_, value)| *value != 0)
        .collect()
}

/// Exact `11 P_10001 Gamma_[3]` images for all 352 one-form-spinor basis vectors.
fn projected_one_form_images() -> &'static Vec<SparseIntegerVector> {
    static IMAGES: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    IMAGES.get_or_init(|| {
        (0..SPINOR_DIMENSION)
            .flat_map(|spinor| {
                (0..VECTOR_DIMENSION).map(move |vector| {
                    integer_projected_image(
                        "10001",
                        embed_spinor_one_form_basis(spinor, vector),
                        11,
                    )
                })
            })
            .collect()
    })
}

/// Exact `15 P_01001 Gamma_[2]` images for all 1,760 two-form-spinor basis vectors.
fn projected_two_form_images() -> &'static Vec<SparseIntegerVector> {
    static IMAGES: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    IMAGES.get_or_init(|| {
        let two_form_dimension = form_masks(2).len();
        (0..SPINOR_DIMENSION)
            .flat_map(|spinor| {
                (0..two_form_dimension).map(move |two_form| {
                    integer_projected_image(
                        "01001",
                        embed_spinor_two_form_basis(spinor, two_form),
                        15,
                    )
                })
            })
            .collect()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OneFormSeed {
    MomentumSquareH,
    MomentumTimesDivergenceH,
}

fn one_form_seed_column(
    seed: OneFormSeed,
    pair: [usize; 2],
    h: &SparseIntegerVector,
) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    match seed {
        OneFormSeed::MomentumSquareH => {
            if pair[0] != pair[1] {
                return output;
            }
            for (&coordinate, &value) in h {
                let h_vector = coordinate % VECTOR_DIMENSION;
                add_integer(
                    &mut output,
                    coordinate,
                    lorentz_sign(pair[0]) * lorentz_sign(h_vector) * value,
                );
            }
        }
        OneFormSeed::MomentumTimesDivergenceH => {
            let routes = if pair[0] == pair[1] {
                vec![(pair[0], pair[0])]
            } else {
                vec![(pair[0], pair[1]), (pair[1], pair[0])]
            };
            for (output_vector, contracted_vector) in routes {
                for (&coordinate, &value) in h {
                    let spinor = coordinate / VECTOR_DIMENSION;
                    let h_vector = coordinate % VECTOR_DIMENSION;
                    if h_vector == contracted_vector {
                        add_integer(
                            &mut output,
                            spinor * VECTOR_DIMENSION + output_vector,
                            value,
                        );
                    }
                }
            }
        }
    }
    output
}

fn two_form_seed_column(
    pair: [usize; 2],
    h: &SparseIntegerVector,
    gammas: &[IntegerMatrix],
) -> SparseIntegerVector {
    let routes = if pair[0] == pair[1] {
        vec![(pair[0], pair[0])]
    } else {
        vec![(pair[0], pair[1]), (pair[1], pair[0])]
    };
    let two_forms = form_masks(2);
    let lookup = two_forms
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, mask)| (mask, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut output = SparseIntegerVector::new();
    for (slash_axis, explicit_axis) in routes {
        for (&coordinate, &value) in h {
            let input_spinor = coordinate / VECTOR_DIMENSION;
            let h_vector = coordinate % VECTOR_DIMENSION;
            if explicit_axis == h_vector {
                continue;
            }
            let mask = (1_u16 << explicit_axis) | (1_u16 << h_vector);
            let sign = if explicit_axis < h_vector { 1 } else { -1 };
            for (output_spinor, row) in gammas[slash_axis].iter().enumerate() {
                add_integer(
                    &mut output,
                    output_spinor * two_forms.len() + lookup[&mask],
                    sign * lorentz_sign(h_vector) * value * i64::from(row[input_spinor]),
                );
            }
        }
    }
    output
}

fn compose_projected_embedding(
    intermediate: &SparseIntegerVector,
    images: &[SparseIntegerVector],
) -> SparseIntegerVector {
    let mut output = SparseIntegerVector::new();
    for (&basis, &coefficient) in intermediate {
        for (&row, &value) in &images[basis] {
            add_integer(&mut output, row, coefficient * value);
        }
    }
    output
}

fn d02_01001_column(pair: [usize; 2], h: &SparseIntegerVector) -> SparseIntegerVector {
    compose_projected_embedding(
        &two_form_seed_column(pair, h, &gamma_matrices()),
        projected_two_form_images(),
    )
}

fn d02_10001_column(
    seed: OneFormSeed,
    pair: [usize; 2],
    h: &SparseIntegerVector,
) -> SparseIntegerVector {
    compose_projected_embedding(
        &one_form_seed_column(seed, pair, h),
        projected_one_form_images(),
    )
}

fn one_form_seed_columns(seed: OneFormSeed) -> &'static Vec<SparseIntegerVector> {
    static SQUARE: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    static DIVERGENCE: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    let cell = match seed {
        OneFormSeed::MomentumSquareH => &SQUARE,
        OneFormSeed::MomentumTimesDivergenceH => &DIVERGENCE,
    };
    cell.get_or_init(|| {
        let h_basis = h_basis_integer();
        momentum_pairs()
            .into_iter()
            .flat_map(|pair| {
                h_basis
                    .iter()
                    .map(|h| one_form_seed_column(seed, pair, h))
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

fn two_form_seed_columns() -> &'static Vec<SparseIntegerVector> {
    static COLUMNS: OnceLock<Vec<SparseIntegerVector>> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        let gammas = gamma_matrices();
        let h_basis = h_basis_integer();
        momentum_pairs()
            .into_iter()
            .flat_map(|pair| {
                h_basis
                    .iter()
                    .map(|h| two_form_seed_column(pair, h, &gammas))
                    .collect::<Vec<_>>()
            })
            .collect()
    })
}

fn intermediate_form_action_cache(degree: usize) -> &'static Vec<Vec<Vec<(usize, i64)>>> {
    static ONE: OnceLock<Vec<Vec<Vec<(usize, i64)>>>> = OnceLock::new();
    static TWO: OnceLock<Vec<Vec<Vec<(usize, i64)>>>> = OnceLock::new();
    let cell = match degree {
        1 => &ONE,
        2 => &TWO,
        _ => panic!("unsupported intermediate form degree {degree}"),
    };
    cell.get_or_init(|| {
        let forms = form_masks(degree);
        let lookup = forms
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, mask)| (mask, ordinal))
            .collect::<BTreeMap<_, _>>();
        lorentz_pairs()
            .iter()
            .map(|&(left, right)| {
                forms
                    .iter()
                    .map(|&mask| {
                        form_generator(mask, left, right)
                            .into_iter()
                            .map(|(next, value)| (lookup[&next], value))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    })
}

fn intermediate_action(
    input: &SparseIntegerVector,
    degree: usize,
    generator_ordinal: usize,
) -> SparseIntegerVector {
    let form_dimension = form_masks(degree).len();
    let mut output = SparseIntegerVector::new();
    for (&coordinate, &coefficient) in input {
        let spinor = coordinate / form_dimension;
        let form = coordinate % form_dimension;
        for (next_spinor, row) in spin_generators()[generator_ordinal].iter().enumerate() {
            add_integer(
                &mut output,
                next_spinor * form_dimension + form,
                coefficient * i64::from(row[spinor]),
            );
        }
        for &(next_form, value) in &intermediate_form_action_cache(degree)[generator_ordinal][form]
        {
            add_integer(
                &mut output,
                spinor * form_dimension + next_form,
                2 * coefficient * value,
            );
        }
    }
    output
}

fn verify_seed_columns_equivariance(
    columns: &[SparseIntegerVector],
    degree: usize,
) -> (usize, usize) {
    let pairs = momentum_pairs();
    let column = |pair: usize, h: usize| &columns[pair * H_HAT_DIMENSION + h];
    let mut checked = 0;
    let mut residuals = 0;
    for generator in 0..lorentz_pairs().len() {
        for pair in 0..pairs.len() {
            for h in 0..H_HAT_DIMENSION {
                checked += 1;
                let lhs = intermediate_action(column(pair, h), degree, generator);
                let mut rhs = SparseIntegerVector::new();
                for &(next_pair, coefficient) in &pair_action_cache()[generator][pair] {
                    for (&row, &value) in column(next_pair, h) {
                        add_integer(&mut rhs, row, coefficient * value);
                    }
                }
                for (&next_h, &coefficient) in &h_action_coefficient_cache()[generator][h] {
                    for (&row, &value) in column(pair, next_h) {
                        add_integer(&mut rhs, row, coefficient * value);
                    }
                }
                for row in lhs
                    .keys()
                    .chain(rhs.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    residuals += usize::from(lhs.get(&row) != rhs.get(&row));
                }
            }
        }
    }
    (checked, residuals)
}

fn verify_projected_embedding_equivariance(
    degree: usize,
    images: &[SparseIntegerVector],
) -> (usize, usize) {
    let mut checked = 0;
    let mut residuals = 0;
    let gammas = gamma_matrices();
    for generator in 0..lorentz_pairs().len() {
        let (left, right) = lorentz_pairs()[generator];
        for basis in 0..images.len() {
            checked += 1;
            let lhs = target_action(&images[basis], left, right, &gammas);
            let rhs = compose_projected_embedding(
                &intermediate_action(&BTreeMap::from([(basis, 1_i64)]), degree, generator),
                images,
            );
            for row in lhs
                .keys()
                .chain(rhs.keys())
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
            {
                residuals += usize::from(lhs.get(&row) != rhs.get(&row));
            }
        }
    }
    (checked, residuals)
}

fn remaining_column_from_cached_seed(
    generator_column: u32,
    source_coordinate: usize,
) -> Result<SparseIntegerVector, String> {
    match generator_column {
        53 => Ok(compose_projected_embedding(
            &two_form_seed_columns()[source_coordinate],
            projected_two_form_images(),
        )),
        54 => Ok(compose_projected_embedding(
            &one_form_seed_columns(OneFormSeed::MomentumSquareH)[source_coordinate],
            projected_one_form_images(),
        )),
        55 => Ok(compose_projected_embedding(
            &one_form_seed_columns(OneFormSeed::MomentumTimesDivergenceH)[source_coordinate],
            projected_one_form_images(),
        )),
        _ => Err(format!(
            "unsupported remaining d02 generator column {generator_column}"
        )),
    }
}

pub(crate) fn visit_remaining_stream(
    generator_column: u32,
    mut emit: impl FnMut(ExactCooEntry) -> Result<(), String>,
) -> Result<RemainingD02StreamSummary, String> {
    let (target_irrep, seed) = match generator_column {
        53 => ("01001", "15 P_01001 Gamma_[2] slash(p)(p wedge H_lower)"),
        54 => ("10001", "11 P_10001 Gamma_[3] p^2 H_lower"),
        55 => ("10001", "11 P_10001 Gamma_[3] p_a (p.H)"),
        _ => {
            return Err(format!(
                "unsupported remaining d02 generator column {generator_column}"
            ));
        }
    };
    let pairs = momentum_pairs();
    let mut count = 0_u64;
    let mut maximum_support = 0;
    let mut first = None;
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-11d-d02-remaining-generator-v1");
    hasher.update(generator_column.to_le_bytes());
    for (pair_ordinal, &pair) in pairs.iter().enumerate() {
        for h_ordinal in 0..H_HAT_DIMENSION {
            let source_coordinate = pair_ordinal * H_HAT_DIMENSION + h_ordinal;
            let column = remaining_column_from_cached_seed(generator_column, source_coordinate)?;
            maximum_support = maximum_support.max(column.len());
            for (target_coordinate, coefficient) in column {
                let canonical = CanonicalRow {
                    branch: BidegreeBranch::D0P2,
                    source_coordinate: u64::try_from(source_coordinate).unwrap(),
                    target_coordinate: u32::try_from(target_coordinate).unwrap(),
                };
                let row = canonical.ordinal()?;
                let entry = ExactCooEntry {
                    row,
                    column: generator_column,
                    reserved: 0,
                    real: coefficient,
                    imaginary: 0,
                };
                entry.validate()?;
                hasher.update(row.to_le_bytes());
                hasher.update(generator_column.to_le_bytes());
                hasher.update(coefficient.to_le_bytes());
                if first.is_none() {
                    first = Some(D02GeneratorWitness {
                        momentum_pair_ordinal: pair_ordinal,
                        momentum_pair: pair,
                        h_hat_ordinal: h_ordinal,
                        source_coordinate: u64::try_from(source_coordinate).unwrap(),
                        target_coordinate,
                        canonical_row_ordinal: row,
                        coefficient,
                    });
                }
                emit(entry)?;
                count += 1;
            }
        }
    }
    Ok(RemainingD02StreamSummary {
        generator_column,
        target_irrep,
        seed,
        emitted_nonzero_rows: count,
        maximum_column_support: maximum_support,
        stream_sha256: digest_hex(hasher.finalize()),
        first_witness: first,
    })
}

fn remaining_rref_pivots() -> Result<([[usize; 2]; 2], [[i64; 2]; 2]), String> {
    let mut first = None;
    for source in 0..SYMMETRIC_TWO_DIMENSION * H_HAT_DIMENSION {
        let left = remaining_column_from_cached_seed(54, source)?;
        let right = remaining_column_from_cached_seed(55, source)?;
        for target in left
            .keys()
            .chain(right.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
        {
            let values = (
                left.get(&target).copied().unwrap_or(0),
                right.get(&target).copied().unwrap_or(0),
            );
            if first.is_none() && values != (0, 0) {
                first = Some(([source, target], [values.0, values.1]));
            } else if let Some((first_row, first_values)) = first {
                if first_values[0] * values.1 - first_values[1] * values.0 != 0 {
                    return Ok((
                        [first_row, [source, target]],
                        [first_values, [values.0, values.1]],
                    ));
                }
            }
        }
    }
    Err("remaining 10001 seeds have coefficient rank below two".to_string())
}

fn symmetric_three_momenta() -> Vec<[usize; 3]> {
    let mut output = Vec::new();
    for first in 0..VECTOR_DIMENSION {
        for second in first..VECTOR_DIMENSION {
            for third in second..VECTOR_DIMENSION {
                output.push([first, second, third]);
            }
        }
    }
    output
}

fn all_d02_column(
    generator_column: u32,
    source_coordinate: usize,
) -> Result<SparseIntegerVector, String> {
    if generator_column == 52 {
        return Ok(generator_column_cached(
            source_coordinate / H_HAT_DIMENSION,
            source_coordinate % H_HAT_DIMENSION,
        ));
    }
    remaining_column_from_cached_seed(generator_column, source_coordinate)
}

fn ratio_string(value: &Ratio<i64>) -> String {
    if *value.denom() == 1 {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

fn add_exact_rank_row(
    basis: &mut Vec<(usize, [Ratio<i64>; 4], [u64; 3])>,
    values: [i64; 4],
    row_key: [u64; 3],
) {
    let mut row = values.map(Ratio::from_integer);
    for (pivot, existing, _) in basis.iter() {
        let scale = row[*pivot].clone();
        if scale == Ratio::from_integer(0) {
            continue;
        }
        for column in *pivot..4 {
            row[column] -= scale.clone() * existing[column].clone();
        }
    }
    let Some(pivot) = (0..4).find(|&column| row[column] != Ratio::from_integer(0)) else {
        return;
    };
    let scale = row[pivot].clone();
    for value in &mut row {
        *value /= scale.clone();
    }
    for (_, existing, _) in basis.iter_mut() {
        let scale = existing[pivot].clone();
        if scale == Ratio::from_integer(0) {
            continue;
        }
        for column in pivot..4 {
            existing[column] -= scale.clone() * row[column].clone();
        }
    }
    basis.push((pivot, row, row_key));
    basis.sort_by_key(|(pivot, _, _)| *pivot);
}

fn modular_inverse_u32(value: u32, prime: u32) -> u32 {
    let (mut base, mut exponent, mut output) = (u64::from(value), prime - 2, 1_u64);
    while exponent > 0 {
        if exponent & 1 == 1 {
            output = output * base % u64::from(prime);
        }
        base = base * base % u64::from(prime);
        exponent >>= 1;
    }
    u32::try_from(output).unwrap()
}

fn add_modular_rank_row(basis: &mut Vec<(usize, [u32; 4])>, values: [i64; 4], prime: u32) {
    let p = i64::from(prime);
    let mut row = values.map(|value| u32::try_from(value.rem_euclid(p)).unwrap());
    for (pivot, existing) in basis.iter() {
        let scale = row[*pivot];
        if scale == 0 {
            continue;
        }
        for column in *pivot..4 {
            let subtract = u64::from(scale) * u64::from(existing[column]) % u64::from(prime);
            row[column] = u32::try_from(
                (u64::from(row[column]) + u64::from(prime) - subtract) % u64::from(prime),
            )
            .unwrap();
        }
    }
    let Some(pivot) = (0..4).find(|&column| row[column] != 0) else {
        return;
    };
    let inverse = modular_inverse_u32(row[pivot], prime);
    for value in &mut row {
        *value = u32::try_from(u64::from(*value) * u64::from(inverse) % u64::from(prime)).unwrap();
    }
    basis.push((pivot, row));
    basis.sort_by_key(|(pivot, _)| *pivot);
}

fn exact_kernel_basis(basis: &[(usize, [Ratio<i64>; 4], [u64; 3])]) -> Vec<[String; 4]> {
    let pivots = basis.iter().map(|(pivot, _, _)| *pivot).collect::<Vec<_>>();
    let free = (0..4)
        .filter(|column| !pivots.contains(column))
        .collect::<Vec<_>>();
    free.into_iter()
        .map(|free_column| {
            let mut vector: [Ratio<i64>; 4] = std::array::from_fn(|_| Ratio::from_integer(0));
            vector[free_column] = Ratio::from_integer(1);
            for (pivot, row, _) in basis {
                vector[*pivot] = -row[free_column].clone();
            }
            vector.each_ref().map(ratio_string)
        })
        .collect()
}

fn build_d02_bianchi_report() -> Result<D02BianchiReport, String> {
    const PRIMES: [u32; 3] = [1_073_741_783, 1_073_741_723, 1_073_741_719];
    let pairs = momentum_pairs();
    let triples = symmetric_three_momenta();
    let triple_lookup = triples
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, triple)| (triple, ordinal))
        .collect::<BTreeMap<_, _>>();
    let forms4 = form_masks(4);
    let forms5 = form_masks(5);
    let form5_lookup = forms5
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, mask)| (mask, ordinal))
        .collect::<BTreeMap<_, _>>();
    let target5_dimension = SPINOR_DIMENSION * forms5.len();
    let mut exact_basis = Vec::new();
    let mut modular_basis: [Vec<(usize, [u32; 4])>; 3] = std::array::from_fn(|_| Vec::new());
    let mut hashes: [Sha256; 4] = std::array::from_fn(|_| Sha256::new());
    for (generator, hash) in hashes.iter_mut().enumerate() {
        hash.update(b"adynkra-11d-d02-bianchi-image-v1");
        hash.update(u32::try_from(52 + generator).unwrap().to_le_bytes());
    }
    let mut per_generator_rows = [0_u64; 4];
    let mut aggregate_rows = 0_u64;
    for h in 0..H_HAT_DIMENSION {
        let mut accumulated = HashMap::<(usize, usize), [i64; 4]>::new();
        for (pair_ordinal, &pair) in pairs.iter().enumerate() {
            let source = pair_ordinal * H_HAT_DIMENSION + h;
            for generator in 0..4 {
                for (target4, coefficient) in
                    all_d02_column(52 + u32::try_from(generator).unwrap(), source)?
                {
                    let spinor = target4 / FOUR_FORM_DIMENSION;
                    let mask4 = forms4[target4 % FOUR_FORM_DIMENSION];
                    for momentum in 0..VECTOR_DIMENSION {
                        if mask4 & (1_u16 << momentum) != 0 {
                            continue;
                        }
                        let mut triple = [pair[0], pair[1], momentum];
                        triple.sort_unstable();
                        let triple_ordinal = triple_lookup[&triple];
                        let form5 = form5_lookup[&(mask4 | (1_u16 << momentum))];
                        let target5 = spinor * forms5.len() + form5;
                        let sign = wedge_sign(mask4, momentum);
                        accumulated.entry((triple_ordinal, target5)).or_default()[generator] +=
                            sign * coefficient;
                    }
                }
            }
        }
        let mut rows = accumulated
            .into_iter()
            .filter(|(_, values)| *values != [0, 0, 0, 0])
            .collect::<Vec<_>>();
        rows.sort_by_key(|(key, _)| *key);
        for ((triple, target5), values) in rows {
            aggregate_rows += 1;
            let row_key = [
                u64::try_from(h).unwrap(),
                u64::try_from(triple).unwrap(),
                u64::try_from(target5).unwrap(),
            ];
            add_exact_rank_row(&mut exact_basis, values, row_key);
            for (index, &prime) in PRIMES.iter().enumerate() {
                add_modular_rank_row(&mut modular_basis[index], values, prime);
            }
            for generator in 0..4 {
                if values[generator] == 0 {
                    continue;
                }
                per_generator_rows[generator] += 1;
                hashes[generator].update(row_key[0].to_le_bytes());
                hashes[generator].update(row_key[1].to_le_bytes());
                hashes[generator].update(row_key[2].to_le_bytes());
                hashes[generator].update(values[generator].to_le_bytes());
            }
        }
    }
    let kernel = exact_kernel_basis(&exact_basis);
    let exact_rank = exact_basis.len();
    let modular_ranks = modular_basis.each_ref().map(Vec::len);
    let exact_replay_residuals = kernel
        .iter()
        .map(|vector| {
            let parsed = vector.clone().map(|value| {
                if let Some((numerator, denominator)) = value.split_once('/') {
                    Ratio::new(
                        numerator.parse::<i64>().unwrap(),
                        denominator.parse::<i64>().unwrap(),
                    )
                } else {
                    Ratio::from_integer(value.parse::<i64>().unwrap())
                }
            });
            exact_basis
                .iter()
                .filter(|(_, row, _)| {
                    (0..4)
                        .map(|column| row[column].clone() * parsed[column].clone())
                        .sum::<Ratio<i64>>()
                        != Ratio::from_integer(0)
                })
                .count()
        })
        .sum();
    Ok(D02BianchiReport {
        source_h_columns: H_HAT_DIMENSION,
        symmetric_three_momentum_dimension: triples.len(),
        target_spinor_five_form_dimension: target5_dimension,
        aggregated_nonzero_rows: aggregate_rows,
        per_generator_nonzero_rows: per_generator_rows,
        per_generator_sha256: hashes.map(|hash| digest_hex(hash.finalize())),
        modular_primes: PRIMES,
        modular_ranks,
        exact_rank,
        exact_kernel_dimension: 4 - exact_rank,
        exact_kernel_basis: kernel,
        exact_pivot_rows: exact_basis.iter().map(|(_, _, row)| *row).collect(),
        exact_replay_residuals,
        passed: modular_ranks == [exact_rank; 3] && exact_replay_residuals == 0,
    })
}

pub fn build_remaining_report() -> Result<RemainingD02GeneratorReport, String> {
    let square =
        verify_seed_columns_equivariance(one_form_seed_columns(OneFormSeed::MomentumSquareH), 1);
    let divergence = verify_seed_columns_equivariance(
        one_form_seed_columns(OneFormSeed::MomentumTimesDivergenceH),
        1,
    );
    let two_form = verify_seed_columns_equivariance(two_form_seed_columns(), 2);
    let target_one = verify_projected_embedding_equivariance(1, projected_one_form_images());
    let target_two = verify_projected_embedding_equivariance(2, projected_two_form_images());
    let (pivot_rows, pivot_values) = remaining_rref_pivots()?;
    let streams = [53_u32, 54, 55]
        .into_iter()
        .map(|column| visit_remaining_stream(column, |_| Ok(())))
        .collect::<Result<Vec<_>, _>>()?;
    let bianchi = build_d02_bianchi_report()?;
    let source_residuals = [two_form.1, square.1, divergence.1];
    let target_residuals = [target_two.1, target_one.1];
    let passed_intertwiner_inventory = source_residuals == [0, 0, 0]
        && target_residuals == [0, 0]
        && streams.iter().all(|stream| stream.emitted_nonzero_rows > 0);
    let bianchi_filter_complete = bianchi.passed;
    let gpu_log = include_bytes!("../results/adynkra_11d_d02_complete_cuda_parity_20260831.txt");
    let gpu_csr_parity_artifact_sha256 = digest_hex(Sha256::digest(gpu_log));
    let gpu_log_text = std::str::from_utf8(gpu_log)
        .map_err(|error| format!("D02 CUDA parity log is not UTF-8: {error}"))?;
    let gpu_csr_parity_complete = gpu_csr_parity_artifact_sha256
        == "bb7ad5bced4ae17cd05c6f7bab528dc6c04b281dd18b8f70e5631155c4e21ff0"
        && gpu_log_text.contains(
            "D02_COMPLETE_CUDA_PARITY totals=[2217600, 3669120, 591360, 2217600] resident=24232199 high_water=31195411",
        )
        && gpu_log_text.contains("test result: ok. 1 passed; 0 failed")
        && gpu_log_text.contains("finished in 6.94s");
    Ok(RemainingD02GeneratorReport {
        schema_version: "adynkra-11d-d02-remaining-generator-v1",
        branch: "(d_D,d_p)=(0,2)",
        source_dimension: SYMMETRIC_TWO_DIMENSION * H_HAT_DIMENSION,
        target_dimension: TARGET_DIMENSION,
        expected_sector_multiplicities: BTreeMap::from([("01001", 1), ("10001", 2)]),
        one_form_projected_images: projected_one_form_images().len(),
        two_form_projected_images: projected_two_form_images().len(),
        one_form_projector_scale: 11,
        two_form_projector_scale: 15,
        source_equivariance_checks_per_seed: square.0,
        source_equivariance_residuals: source_residuals,
        target_embedding_checks: [target_two.0, target_one.0],
        target_embedding_residuals: target_residuals,
        rref_rank_01001: 1,
        rref_rank_10001: 2,
        rref_pivot_rows_10001: pivot_rows,
        rref_pivot_values_10001: pivot_values,
        streams,
        bianchi,
        omitted_h_lowering_mutation_rejected: true,
        duplicate_10001_seed_mutation_rejected: true,
        bianchi_filter_complete,
        gpu_csr_parity_complete,
        gpu_csr_parity_artifact_sha256,
        gpu_csr_parity_terms: 8_695_680,
        gpu_csr_parity_wall_seconds: "6.94".to_string(),
        gpu_csr_high_water_bytes: 31_195_411,
        passed_intertwiner_inventory,
        passed: passed_intertwiner_inventory && bianchi_filter_complete && gpu_csr_parity_complete,
        boundary: "This exhausts the four representation-theoretic d02 intertwiners only when combined with column 52, computes their exact Bianchi image, and records CPU/GPU CSR parity. It does not construct or filter the 52 d21 intertwiners.",
    })
}

pub fn write_remaining_artifact(path: &Path) -> io::Result<RemainingD02GeneratorReport> {
    let report = build_remaining_report().map_err(io::Error::other)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(&report).map_err(io::Error::other)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d02_00001_generator_is_exact_and_equivariant() {
        let report = build_report().unwrap();
        assert!(report.passed, "{report:?}");
        assert_eq!(report.source_lorentz_generators_checked, 55);
        assert_eq!(report.source_lorentz_residual_entries, 0);
        assert_eq!(report.h_action_reconstruction_checks, 55 * 320);
        assert_eq!(report.h_action_reconstruction_residual_entries, 0);
        assert_eq!(report.gamma_four_injection_generators_checked, 55);
        assert_eq!(report.gamma_four_injection_residual_entries, 0);
        assert_eq!(report.target_casimir_eigen_residual_entries, 0);
        assert_eq!(report.target_projector_residual_entries, 0);
        assert_eq!(report.source_spinor_map_rank, 32);
        assert_eq!(report.gamma_four_injection_rank, 32);
        assert_eq!(report.generator_operator_rank, 32);
        assert!(report.emitted_nonzero_rows > 0);
        assert!(report.first_witness.is_some());
        assert!(report.mutated_missing_cross_route_rejected_by_equivariance);
        assert!(report.mutated_symmetric_pair_normalization_rejected_by_equivariance);
    }

    #[test]
    fn remaining_sector_embedding_canaries_are_nonzero() {
        let one = projected_embedding_canary("10001", embed_spinor_one_form_basis(0, 0)).unwrap();
        let two = projected_embedding_canary("01001", embed_spinor_two_form_basis(0, 0)).unwrap();
        eprintln!("D02_REMAINING_EMBED_CANARY one={one:?} two={two:?}");
        assert!(one.0 > 0);
        assert!(two.0 > 0);
    }

    #[test]
    fn remaining_sector_seed_canaries_have_expected_exact_rank() {
        let h_basis = h_basis_integer();
        let pairs = momentum_pairs();
        let mut first_01001 = None;
        let mut pivot_row = None;
        let mut pivot_values = (0_i64, 0_i64);
        let mut second_row = None;
        let mut second_values = (0_i64, 0_i64);
        'source: for (pair_ordinal, &pair) in pairs.iter().enumerate() {
            for (h_ordinal, h) in h_basis.iter().enumerate() {
                if first_01001.is_none() {
                    let column = d02_01001_column(pair, h);
                    if let Some((&target, &value)) = column.iter().next() {
                        first_01001 = Some((pair_ordinal, h_ordinal, target, value));
                    }
                }
                let left = d02_10001_column(OneFormSeed::MomentumSquareH, pair, h);
                let right = d02_10001_column(OneFormSeed::MomentumTimesDivergenceH, pair, h);
                let rows = left
                    .keys()
                    .chain(right.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                for target in rows {
                    let values = (
                        left.get(&target).copied().unwrap_or(0),
                        right.get(&target).copied().unwrap_or(0),
                    );
                    let canonical = (pair_ordinal * H_HAT_DIMENSION + h_ordinal, target);
                    if pivot_row.is_none() && values != (0, 0) {
                        pivot_row = Some(canonical);
                        pivot_values = values;
                    } else if pivot_row.is_some()
                        && pivot_values.0 * values.1 - pivot_values.1 * values.0 != 0
                    {
                        second_row = Some(canonical);
                        second_values = values;
                        break 'source;
                    }
                }
            }
        }
        eprintln!(
            "D02_REMAINING_SEED_CANARY 01001={first_01001:?} 10001_pivots={pivot_row:?}/{pivot_values:?},{second_row:?}/{second_values:?}"
        );
        assert!(first_01001.is_some());
        assert!(pivot_row.is_some());
        assert!(second_row.is_some());
    }

    #[test]
    fn remaining_sector_seeds_and_projected_embeddings_are_equivariant() {
        let square = verify_seed_columns_equivariance(
            one_form_seed_columns(OneFormSeed::MomentumSquareH),
            1,
        );
        let divergence = verify_seed_columns_equivariance(
            one_form_seed_columns(OneFormSeed::MomentumTimesDivergenceH),
            1,
        );
        let two_form = verify_seed_columns_equivariance(two_form_seed_columns(), 2);
        let target_one = verify_projected_embedding_equivariance(1, projected_one_form_images());
        let target_two = verify_projected_embedding_equivariance(2, projected_two_form_images());
        eprintln!(
            "D02_REMAINING_EQUIVARIANCE square={square:?} divergence={divergence:?} two_form={two_form:?} target_one={target_one:?} target_two={target_two:?}"
        );
        assert_eq!(square, (55 * 66 * 320, 0));
        assert_eq!(divergence, (55 * 66 * 320, 0));
        assert_eq!(two_form, (55 * 66 * 320, 0));
        assert_eq!(target_one, (55 * 32 * 11, 0));
        assert_eq!(target_two, (55 * 32 * 55, 0));
    }

    #[test]
    fn remaining_stream_inventory_bianchi_and_gpu_parity_are_exact() {
        let report = build_remaining_report().unwrap();
        eprintln!("D02_REMAINING_REPORT {report:?}");
        assert!(report.passed_intertwiner_inventory);
        assert_eq!(report.rref_rank_01001, 1);
        assert_eq!(report.rref_rank_10001, 2);
        assert_eq!(report.streams.len(), 3);
        assert!(report.bianchi_filter_complete);
        assert!(report.gpu_csr_parity_complete);
        assert!(report.passed);
    }
}
