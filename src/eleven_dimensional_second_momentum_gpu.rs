//! Packed exact finite-field backend for the bounded second-momentum `F_X` screen.
//!
//! The physical component construction is linear.  This module keeps that
//! linearity explicit so the large rational `BTreeMap` intermediates can be
//! replaced by a fused modular contraction on an accelerator.  The final
//! matrix has only 77 columns, so rank certification remains a small exact
//! host operation after the accelerator has produced its functional rows.

#[cfg(feature = "cuda")]
use std::fs::{self, File};
#[cfg(feature = "cuda")]
use std::io::{BufWriter, Write};
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::Instant;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_k_fag_solver::ExactGaussian;
use crate::eleven_dimensional_second_momentum_fx::{
    DegreeTwoMomentumMonomial, P3D11ExactStaticData, SECOND_MOMENTUM_FX_BUCKETS_PER_SEED,
    SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS, SecondMomentumFxSector, SecondMomentumGaugeBranch,
    SecondMomentumGaugeChannel, produce_p3_d11_contractions, project_p3_d11_to_physical_fx,
    second_momentum_fx_functional_assignments,
};

pub(crate) const GPU_FX_SCHEMA: &str = "adynkra-11d-second-momentum-gpu-fx-v3-one-seed";
pub(crate) const GPU_FX_PRIMES: [u32; 3] = [1_073_741_783, 1_073_741_723, 1_073_741_719];
pub(crate) const MOMENTUM_PAIR_COUNT: usize = 66;
pub(crate) const SECTOR_COUNT: usize = 2;
pub(crate) const GAUGE_DEGREE_COUNT: usize = 6;
pub(crate) const GPU_FX_FUNCTIONAL_SEEDS: [u64; 1] = [SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS[0]];
pub(crate) const FUNCTIONAL_ROW_COUNT: usize = GAUGE_DEGREE_COUNT
    * MOMENTUM_PAIR_COUNT
    * SECTOR_COUNT
    * GPU_FX_FUNCTIONAL_SEEDS.len()
    * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;
pub(crate) const GPU_P3_FX_SCHEMA: &str =
    "adynkra-11d-second-momentum-gpu-p3-fx-v1-one-seed-axis-retained";
pub(crate) const P3_CONTRACTION_AXIS_COUNT: usize = 11;
pub(crate) const P3_X2_OUTPUT_COORDINATES: usize = 55 * 11;
pub(crate) const P3_X5_OUTPUT_COORDINATES: usize = 462 * 11;
pub(crate) const P3_FUNCTIONAL_ROW_COUNT: usize = GAUGE_DEGREE_COUNT
    * MOMENTUM_PAIR_COUNT
    * SECTOR_COUNT
    * P3_CONTRACTION_AXIS_COUNT
    * GPU_FX_FUNCTIONAL_SEEDS.len()
    * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct P3FunctionalRowCoordinates {
    pub gauge_degree: usize,
    pub momentum_pair_ordinal: usize,
    pub sector: usize,
    pub contraction_axis: usize,
    pub seed: usize,
    pub bucket: usize,
}

pub(crate) fn p3_functional_row(coordinates: P3FunctionalRowCoordinates) -> Result<usize, String> {
    if coordinates.gauge_degree >= GAUGE_DEGREE_COUNT
        || coordinates.momentum_pair_ordinal >= MOMENTUM_PAIR_COUNT
        || coordinates.sector >= SECTOR_COUNT
        || coordinates.contraction_axis >= P3_CONTRACTION_AXIS_COUNT
        || coordinates.seed >= GPU_FX_FUNCTIONAL_SEEDS.len()
        || coordinates.bucket >= SECOND_MOMENTUM_FX_BUCKETS_PER_SEED
    {
        return Err("p3 functional row coordinate is out of bounds".to_string());
    }
    Ok(
        (((((coordinates.gauge_degree * MOMENTUM_PAIR_COUNT
            + coordinates.momentum_pair_ordinal)
            * SECTOR_COUNT
            + coordinates.sector)
            * P3_CONTRACTION_AXIS_COUNT
            + coordinates.contraction_axis)
            * GPU_FX_FUNCTIONAL_SEEDS.len()
            + coordinates.seed)
            * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED)
            + coordinates.bucket,
    )
}

pub(crate) fn decode_p3_functional_row(row: usize) -> Result<P3FunctionalRowCoordinates, String> {
    if row >= P3_FUNCTIONAL_ROW_COUNT {
        return Err("p3 functional row ordinal is out of bounds".to_string());
    }
    let bucket = row % SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;
    let quotient = row / SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;
    let seed = quotient % GPU_FX_FUNCTIONAL_SEEDS.len();
    let quotient = quotient / GPU_FX_FUNCTIONAL_SEEDS.len();
    let contraction_axis = quotient % P3_CONTRACTION_AXIS_COUNT;
    let quotient = quotient / P3_CONTRACTION_AXIS_COUNT;
    let sector = quotient % SECTOR_COUNT;
    let quotient = quotient / SECTOR_COUNT;
    let momentum_pair_ordinal = quotient % MOMENTUM_PAIR_COUNT;
    let gauge_degree = quotient / MOMENTUM_PAIR_COUNT;
    Ok(P3FunctionalRowCoordinates {
        gauge_degree,
        momentum_pair_ordinal,
        sector,
        contraction_axis,
        seed,
        bucket,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub(crate) struct GaussianResidue {
    pub real: u32,
    pub imaginary: u32,
}

impl GaussianResidue {
    pub(crate) const fn zero() -> Self {
        Self {
            real: 0,
            imaginary: 0,
        }
    }

    pub(crate) fn add(self, other: Self, prime: u32) -> Self {
        Self {
            real: add_mod(self.real, other.real, prime),
            imaginary: add_mod(self.imaginary, other.imaginary, prime),
        }
    }

    pub(crate) fn negate(self, prime: u32) -> Self {
        Self {
            real: negate_mod(self.real, prime),
            imaginary: negate_mod(self.imaginary, prime),
        }
    }

    pub(crate) fn multiply(self, other: Self, prime: u32) -> Self {
        let ac = multiply_mod(self.real, other.real, prime);
        let bd = multiply_mod(self.imaginary, other.imaginary, prime);
        let ad = multiply_mod(self.real, other.imaginary, prime);
        let bc = multiply_mod(self.imaginary, other.real, prime);
        Self {
            real: subtract_mod(ac, bd, prime),
            imaginary: add_mod(ad, bc, prime),
        }
    }

    pub(crate) fn scale(self, scalar: u32, prime: u32) -> Self {
        Self {
            real: multiply_mod(self.real, scalar, prime),
            imaginary: multiply_mod(self.imaginary, scalar, prime),
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        self.real == 0 && self.imaginary == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecoupledSourceTerm {
    pub momentum_pair: [u8; 2],
    pub free_spinor: u8,
    pub exterior_mask: u32,
    pub coefficient: i128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GpuFxColumnInput {
    pub global_ordinal: usize,
    pub source_label: String,
    pub source_copy: usize,
    pub terms: Vec<RecoupledSourceTerm>,
    pub raising_residuals: [usize; 5],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct GpuFxColumnMetadata {
    pub global_ordinal: usize,
    pub source_label: String,
    pub source_copy: usize,
    pub raising_residuals: [usize; 5],
}

impl From<&GpuFxColumnInput> for GpuFxColumnMetadata {
    fn from(column: &GpuFxColumnInput) -> Self {
        Self {
            global_ordinal: column.global_ordinal,
            source_label: column.source_label.clone(),
            source_copy: column.source_copy,
            raising_residuals: column.raising_residuals,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GaugeEntry {
    free_spinor: u8,
    derivative_spinor: u8,
    coefficient: GaussianResidue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TargetEntry {
    vector_weight: u8,
    spinor_weight: u8,
    coefficient: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FxTemplateEntry {
    derivative_spinor: u8,
    sector: u8,
    output_coordinate: u16,
    coefficient: GaussianResidue,
}

#[derive(Clone, Debug)]
pub(crate) struct ModularFxStaticData {
    prime: u32,
    gauge_by_degree_and_free_spinor: Vec<Vec<GaugeEntry>>,
    target: Vec<TargetEntry>,
    template_offsets: Vec<u32>,
    templates: Vec<FxTemplateEntry>,
    semantic_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct P3ModularPlanKey {
    contracted_spinor: u8,
    template_spinor: u8,
    contraction_axis: u8,
    sector: u8,
    output_coordinate: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct P3ModularPlanEntry {
    key: P3ModularPlanKey,
    coefficient: GaussianResidue,
}

#[derive(Clone, Debug)]
pub(crate) struct P3ModularFlatPlan {
    prime: u32,
    offsets: Vec<u32>,
    entries: Vec<P3ModularPlanEntry>,
    semantic_sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct P3RawFanoutTable {
    prime: u32,
    counts: Vec<u64>,
    within_byte: Vec<u32>,
    cross_byte: Vec<u32>,
}

const P3_FANOUT_BYTE_COUNT: usize = 4;
const P3_FANOUT_BYTE_VALUES: usize = 256;
const P3_FANOUT_BYTE_PAIRS: usize = 6;

fn p3_fanout_byte_pair_index(left: usize, right: usize) -> usize {
    debug_assert!(left < right && right < P3_FANOUT_BYTE_COUNT);
    left * (2 * P3_FANOUT_BYTE_COUNT - left - 1) / 2 + right - left - 1
}

impl P3ModularFlatPlan {
    pub(crate) fn semantic_sha256(&self) -> &str {
        &self.semantic_sha256
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of nonzero flattened schedule visits made by one canonical raw
    /// source term. This is deliberately evaluated before any union reduction,
    /// so duplicate or cancelling raw terms retain their published visit count.
    pub(crate) fn raw_expanded_fanout(&self, source: &RecoupledSourceTerm) -> Result<u64, String> {
        self.raw_fanout_table()?.fanout(source)
    }

    pub(crate) fn raw_fanout_table(&self) -> Result<P3RawFanoutTable, String> {
        validate_p3_modular_flat_plan(self)?;
        let mut counts = vec![0_u64; 32 * 32 * 32];
        for degree in 0..GAUGE_DEGREE_COUNT {
            for free_spinor in 0..32 {
                let schedule = degree * 32 + free_spinor;
                for entry in &self.entries
                    [self.offsets[schedule] as usize..self.offsets[schedule + 1] as usize]
                {
                    let index = free_spinor * 32 * 32
                        + usize::from(entry.key.contracted_spinor) * 32
                        + usize::from(entry.key.template_spinor);
                    counts[index] = counts[index]
                        .checked_add(1)
                        .ok_or_else(|| "p3 raw fanout table overflow".to_string())?;
                }
            }
        }
        let row_totals = counts
            .chunks_exact(32)
            .map(|row| {
                row.iter().try_fold(0_u64, |total, &count| {
                    total
                        .checked_add(count)
                        .ok_or_else(|| "p3 raw fanout row total overflow".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut within_byte = vec![0_u32; 32 * P3_FANOUT_BYTE_COUNT * P3_FANOUT_BYTE_VALUES];
        for free in 0..32 {
            for byte in 0..P3_FANOUT_BYTE_COUNT {
                for mask in 0..P3_FANOUT_BYTE_VALUES {
                    let mut total = 0_u64;
                    for contracted_bit in 0..8 {
                        if mask & (1 << contracted_bit) == 0 {
                            continue;
                        }
                        let contracted = byte * 8 + contracted_bit;
                        total = total
                            .checked_add(row_totals[free * 32 + contracted])
                            .ok_or_else(|| "p3 raw fanout byte total overflow".to_string())?;
                        for template_bit in 0..8 {
                            if template_bit == contracted_bit || mask & (1 << template_bit) == 0 {
                                continue;
                            }
                            let template = byte * 8 + template_bit;
                            total = total
                                .checked_sub(counts[free * 32 * 32 + contracted * 32 + template])
                                .ok_or_else(|| "p3 raw fanout within-byte underflow".to_string())?;
                        }
                    }
                    within_byte
                        [(free * P3_FANOUT_BYTE_COUNT + byte) * P3_FANOUT_BYTE_VALUES + mask] =
                        u32::try_from(total)
                            .map_err(|_| "p3 raw fanout within-byte exceeds u32".to_string())?;
                }
            }
        }
        let mut cross_byte =
            vec![0_u32; 32 * P3_FANOUT_BYTE_PAIRS * P3_FANOUT_BYTE_VALUES * P3_FANOUT_BYTE_VALUES];
        for free in 0..32 {
            for left_byte in 0..P3_FANOUT_BYTE_COUNT {
                for right_byte in left_byte + 1..P3_FANOUT_BYTE_COUNT {
                    let pair = p3_fanout_byte_pair_index(left_byte, right_byte);
                    let pair_base = (free * P3_FANOUT_BYTE_PAIRS + pair)
                        * P3_FANOUT_BYTE_VALUES
                        * P3_FANOUT_BYTE_VALUES;
                    for left_mask in 0..P3_FANOUT_BYTE_VALUES {
                        let mut contribution_by_right_bit = [0_u64; 8];
                        for left_bit in 0..8 {
                            if left_mask & (1 << left_bit) == 0 {
                                continue;
                            }
                            let left = left_byte * 8 + left_bit;
                            for (right_bit, contribution) in
                                contribution_by_right_bit.iter_mut().enumerate()
                            {
                                let right = right_byte * 8 + right_bit;
                                let bidirectional = counts[free * 32 * 32 + left * 32 + right]
                                    .checked_add(counts[free * 32 * 32 + right * 32 + left])
                                    .ok_or_else(|| {
                                        "p3 raw fanout cross-byte pair overflow".to_string()
                                    })?;
                                *contribution =
                                    contribution.checked_add(bidirectional).ok_or_else(|| {
                                        "p3 raw fanout cross-byte total overflow".to_string()
                                    })?;
                            }
                        }
                        for right_mask in 1..P3_FANOUT_BYTE_VALUES {
                            let right_bit = right_mask.trailing_zeros() as usize;
                            let previous = right_mask & (right_mask - 1);
                            let previous_value = u64::from(
                                cross_byte
                                    [pair_base + left_mask * P3_FANOUT_BYTE_VALUES + previous],
                            );
                            let value = previous_value
                                .checked_add(contribution_by_right_bit[right_bit])
                                .ok_or_else(|| {
                                    "p3 raw fanout cross-byte total overflow".to_string()
                                })?;
                            cross_byte
                                [pair_base + left_mask * P3_FANOUT_BYTE_VALUES + right_mask] =
                                u32::try_from(value).map_err(|_| {
                                    "p3 raw fanout cross-byte exceeds u32".to_string()
                                })?;
                        }
                    }
                }
            }
        }
        Ok(P3RawFanoutTable {
            prime: self.prime,
            counts,
            within_byte,
            cross_byte,
        })
    }
}

impl P3RawFanoutTable {
    pub(crate) fn fanout(&self, source: &RecoupledSourceTerm) -> Result<u64, String> {
        validate_p3_source_term(source, self.prime)?;
        if i128_mod(source.coefficient, self.prime) == 0 {
            return Ok(0);
        }
        let free = usize::from(source.free_spinor);
        let bytes = source.exterior_mask.to_le_bytes();
        let mut fanout = 0_u64;
        for (byte, &mask) in bytes.iter().enumerate() {
            fanout = fanout
                .checked_add(u64::from(
                    self.within_byte[(free * P3_FANOUT_BYTE_COUNT + byte) * P3_FANOUT_BYTE_VALUES
                        + usize::from(mask)],
                ))
                .ok_or_else(|| "p3 raw expanded fanout overflow".to_string())?;
        }
        for left_byte in 0..P3_FANOUT_BYTE_COUNT {
            for right_byte in left_byte + 1..P3_FANOUT_BYTE_COUNT {
                let pair = p3_fanout_byte_pair_index(left_byte, right_byte);
                let cross = self.cross_byte[(free * P3_FANOUT_BYTE_PAIRS + pair)
                    * P3_FANOUT_BYTE_VALUES
                    * P3_FANOUT_BYTE_VALUES
                    + usize::from(bytes[left_byte]) * P3_FANOUT_BYTE_VALUES
                    + usize::from(bytes[right_byte])];
                fanout = fanout
                    .checked_sub(u64::from(cross))
                    .ok_or_else(|| "p3 raw expanded fanout underflow".to_string())?;
            }
        }
        Ok(fanout)
    }
}

pub(crate) fn p3_recoupling_key(source: &RecoupledSourceTerm) -> Result<u64, String> {
    validate_p3_source_term(source, 2)?;
    let metadata = u32::from(source.momentum_pair[0])
        | (u32::from(source.momentum_pair[1]) << 4)
        | (u32::from(source.free_spinor) << 8);
    Ok((u64::from(metadata) << 32) | u64::from(source.exterior_mask))
}

fn validate_p3_source_term(source: &RecoupledSourceTerm, prime: u32) -> Result<(), String> {
    if prime < 2
        || source.momentum_pair[0] > source.momentum_pair[1]
        || source.momentum_pair[1] >= 11
        || source.free_spinor >= 32
        || source.exterior_mask.count_ones() != 12
        || source.coefficient == 0
        || source.coefficient == i128::MIN
    {
        return Err("p3 source term is not canonical".to_string());
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct ModularFunctionalColumn {
    pub prime: u32,
    pub global_ordinal: usize,
    pub rows: Vec<GaussianResidue>,
    pub expanded_contributions: u64,
    pub semantic_sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ModularP3FunctionalColumn {
    pub prime: u32,
    pub global_ordinal: usize,
    pub rows: Vec<GaussianResidue>,
    pub expanded_contributions: u64,
    pub semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ModularRankCertificate {
    pub schema_version: &'static str,
    pub prime: u32,
    pub row_count: usize,
    pub column_ordinals: Vec<usize>,
    pub rank_over_gaussian_extension: usize,
    pub nullity_upper_bound: usize,
    pub full_column_rank: bool,
    pub matrix_sha256: String,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Serialize)]
pub(crate) struct GpuFxColumnReport {
    pub schema_version: &'static str,
    pub tranche: String,
    pub local_ordinal: usize,
    pub global_ordinal: usize,
    pub source_label: String,
    pub source_copy: usize,
    pub prime: u32,
    pub functional_seeds: [u64; 1],
    pub functional_row_count: usize,
    pub device_name: String,
    pub static_semantic_sha256: String,
    pub flat_plan_sha256: String,
    pub source_terms: usize,
    pub source_terms_sha256: String,
    pub expanded_contributions: u64,
    pub nonzero_functional_rows: usize,
    pub column_semantic_sha256: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub binary_bytes: u64,
    pub source_build_milliseconds: u128,
    pub static_build_milliseconds: u128,
    pub cuda_kernel_milliseconds: f32,
    pub cuda_upload_milliseconds: f32,
    pub cuda_sort_milliseconds: f32,
    pub cuda_reduce_milliseconds: f32,
    pub cuda_contract_milliseconds: f32,
    pub cuda_download_milliseconds: f32,
    pub batch_reduced_key_visits: u64,
    pub batch_nonzero_reduced_term_visits: u64,
    pub cuda_buffer_high_water_bytes: u64,
    pub packed_recoupling_input_sha256: String,
    pub cuda_input_terms_per_second: f64,
    pub cuda_batches: u64,
    pub cuda_peak_batch_terms: usize,
    pub cuda_batch_term_cap: usize,
    pub cuda_host_hard_cap_bytes: u64,
    pub cuda_device_hard_cap_bytes: u64,
    pub cuda_total_device_hard_cap_bytes: u64,
    pub persistent_lowering_enabled: bool,
    pub persistent_lowering_roots: u64,
    pub persistent_lowering_input_entry_visits: u64,
    pub persistent_lowering_expanded_entry_visits: u64,
    pub persistent_lowering_output_entry_visits: u64,
    pub persistent_lowering_gpu_milliseconds: f64,
    pub persistent_lowering_high_water_bytes: u64,
    pub persistent_lowering_peak_output_handle_bytes: u64,
    pub persistent_lowering_maximum_absolute_coefficient: u64,
    pub persistent_lowering_device_hard_cap_bytes: u64,
    pub persistent_lowering_download_chunk_terms: usize,
    pub cpu_parity_terms: usize,
    pub cpu_parity_passed: bool,
    pub end_to_end_milliseconds: u128,
    pub raising_residuals: [usize; 5],
    pub highest_weight_certification: String,
    pub direct_composed_raising_residuals_materialized: bool,
    pub single_column_rank: usize,
    pub passed: bool,
    pub proof_boundary: String,
}

fn validate_prime(prime: u32) -> Result<(), String> {
    if prime % 4 != 3 {
        return Err(format!("finite-field prime {prime} must be 3 modulo 4"));
    }
    if !is_prime_u32(prime) {
        return Err(format!("finite-field modulus {prime} is not prime"));
    }
    Ok(())
}

fn is_prime_u32(value: u32) -> bool {
    if value < 2 {
        return false;
    }
    for divisor in 2..=((value as f64).sqrt() as u32) {
        if value % divisor == 0 {
            return value == divisor;
        }
    }
    true
}

fn add_mod(left: u32, right: u32, prime: u32) -> u32 {
    let sum = u64::from(left) + u64::from(right);
    (sum % u64::from(prime)) as u32
}

fn subtract_mod(left: u32, right: u32, prime: u32) -> u32 {
    if left >= right {
        left - right
    } else {
        prime - (right - left)
    }
}

fn negate_mod(value: u32, prime: u32) -> u32 {
    if value == 0 { 0 } else { prime - value }
}

fn multiply_mod(left: u32, right: u32, prime: u32) -> u32 {
    (u64::from(left) * u64::from(right) % u64::from(prime)) as u32
}

fn power_mod(mut base: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut output = 1_u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = multiply_mod(output, base, prime);
        }
        base = multiply_mod(base, base, prime);
        exponent >>= 1;
    }
    output
}

fn inverse_mod(value: u32, prime: u32) -> u32 {
    assert_ne!(value, 0);
    power_mod(value, prime - 2, prime)
}

fn bigint_mod(value: &BigInt, prime: u32) -> u32 {
    let modulus = BigInt::from(prime);
    let mut residue = (value % &modulus)
        .to_i64()
        .expect("residue modulo a 30-bit prime fits i64");
    if residue < 0 {
        residue += i64::from(prime);
    }
    residue as u32
}

fn rational_mod(value: &Ratio<BigInt>, prime: u32) -> Result<u32, String> {
    let numerator = bigint_mod(value.numer(), prime);
    let denominator = bigint_mod(value.denom(), prime);
    if denominator == 0 {
        return Err(format!(
            "rational denominator is zero modulo finite-field prime {prime}"
        ));
    }
    Ok(multiply_mod(
        numerator,
        inverse_mod(denominator, prime),
        prime,
    ))
}

fn gaussian_mod(value: &ExactGaussian, prime: u32) -> Result<GaussianResidue, String> {
    Ok(GaussianResidue {
        real: rational_mod(&value.real, prime)?,
        imaginary: rational_mod(&value.imaginary, prime)?,
    })
}

fn i128_mod(value: i128, prime: u32) -> u32 {
    let prime = i128::from(prime);
    let residue = value % prime;
    if residue < 0 {
        (residue + prime) as u32
    } else {
        residue as u32
    }
}

fn pair_ordinal(left: usize, right: usize) -> usize {
    assert!(left <= right && right < 11);
    left * 11 - left.saturating_sub(1) * left / 2 + (right - left)
}

fn functional_row(
    degree: usize,
    pair: [usize; 2],
    sector: SecondMomentumFxSector,
    seed: usize,
    bucket: usize,
) -> usize {
    let sector = match sector {
        SecondMomentumFxSector::X2 => 0,
        SecondMomentumFxSector::X5 => 1,
    };
    ((((degree * MOMENTUM_PAIR_COUNT + pair_ordinal(pair[0], pair[1])) * SECTOR_COUNT + sector)
        * GPU_FX_FUNCTIONAL_SEEDS.len()
        + seed)
        * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED)
        + bucket
}

fn right_wedge_sign(mask: u32, spinor: usize) -> Option<i8> {
    let bit = 1_u32 << spinor;
    if mask & bit != 0 {
        return None;
    }
    let greater = if spinor + 1 == 32 {
        0
    } else {
        (mask >> (spinor + 1)).count_ones()
    };
    Some(if greater % 2 == 0 { 1 } else { -1 })
}

fn degree14_to_degree13_mask(mask: u32) -> u32 {
    debug_assert_eq!(mask.count_ones(), 14);
    mask ^ (1_u32 << (31 - mask.leading_zeros()))
}

impl ModularFxStaticData {
    pub(crate) fn build(prime: u32) -> Result<Self, String> {
        validate_prime(prime)?;
        let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis();
        let mut gauge_by_degree_and_free_spinor = vec![Vec::new(); 6 * 32];
        for degree in 0..6 {
            let (_, _, matrix) = gauge_basis
                .iter()
                .find(|(candidate, _, _)| *candidate == degree)
                .ok_or_else(|| format!("missing gauge-form basis degree {degree}"))?;
            for free_spinor in 0..32 {
                for derivative_spinor in 0..32 {
                    let value = &matrix[free_spinor][derivative_spinor];
                    if *value.re.numer() == 0 && *value.im.numer() == 0 {
                        continue;
                    }
                    let exact = ExactGaussian {
                        real: Ratio::new(
                            BigInt::from(*value.re.numer()),
                            BigInt::from(*value.re.denom()),
                        ),
                        imaginary: Ratio::new(
                            BigInt::from(*value.im.numer()),
                            BigInt::from(*value.im.denom()),
                        ),
                    };
                    gauge_by_degree_and_free_spinor[degree * 32 + free_spinor].push(GaugeEntry {
                        free_spinor: free_spinor as u8,
                        derivative_spinor: derivative_spinor as u8,
                        coefficient: gaussian_mod(&exact, prime)?,
                    });
                }
            }
        }

        let highest = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .ok_or_else(|| "missing highest vector-spinor target state".to_string())?;
        let target = highest
            .raw_terms
            .iter()
            .map(|entry| {
                let exact = Ratio::new(
                    BigInt::from(entry.numerator),
                    BigInt::from(entry.denominator),
                );
                Ok(TargetEntry {
                    vector_weight: entry.vector_weight_index as u8,
                    spinor_weight: entry.spinor_weight_index as u8,
                    coefficient: rational_mod(&exact, prime)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut template_offsets = Vec::with_capacity(11 * 32 + 1);
        let mut templates = Vec::new();
        template_offsets.push(0);
        for vector_weight in 0..11 {
            for spinor_weight in 0..32 {
                crate::eleven_dimensional_physical_curvature::visit_exact_fx_derivative_templates(
                    vector_weight,
                    spinor_weight,
                    |entry| {
                        templates.push((
                            entry.derivative_spinor_weight_index,
                            entry.x_two_sector,
                            entry.output_coordinate,
                            entry.coefficient,
                        ));
                    },
                )?;
                template_offsets.push(templates.len() as u32);
            }
        }
        let templates = templates
            .into_iter()
            .map(
                |(derivative_spinor, x_two_sector, output_coordinate, coefficient)| {
                    Ok(FxTemplateEntry {
                        derivative_spinor: derivative_spinor as u8,
                        sector: if x_two_sector { 0 } else { 1 },
                        output_coordinate: u16::try_from(output_coordinate)
                            .map_err(|_| "F_X output coordinate exceeds u16".to_string())?,
                        coefficient: gaussian_mod(&coefficient, prime)?,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let semantic_sha256 = static_semantic_sha256(
            prime,
            &gauge_by_degree_and_free_spinor,
            &target,
            &template_offsets,
            &templates,
        );
        Ok(Self {
            prime,
            gauge_by_degree_and_free_spinor,
            target,
            template_offsets,
            templates,
            semantic_sha256,
        })
    }

    pub(crate) fn prime(&self) -> u32 {
        self.prime
    }

    pub(crate) fn semantic_sha256(&self) -> &str {
        &self.semantic_sha256
    }
}

fn static_semantic_sha256(
    prime: u32,
    gauge: &[Vec<GaugeEntry>],
    target: &[TargetEntry],
    offsets: &[u32],
    templates: &[FxTemplateEntry],
) -> String {
    let mut hash = Sha256::new();
    hash.update(GPU_FX_SCHEMA.as_bytes());
    hash.update(prime.to_le_bytes());
    hash.update((GPU_FX_FUNCTIONAL_SEEDS.len() as u64).to_le_bytes());
    for seed in GPU_FX_FUNCTIONAL_SEEDS {
        hash.update(seed.to_le_bytes());
    }
    for entries in gauge {
        hash.update((entries.len() as u64).to_le_bytes());
        for entry in entries {
            hash.update([entry.free_spinor, entry.derivative_spinor]);
            hash.update(entry.coefficient.real.to_le_bytes());
            hash.update(entry.coefficient.imaginary.to_le_bytes());
        }
    }
    for entry in target {
        hash.update([entry.vector_weight, entry.spinor_weight]);
        hash.update(entry.coefficient.to_le_bytes());
    }
    for offset in offsets {
        hash.update(offset.to_le_bytes());
    }
    for entry in templates {
        hash.update([entry.derivative_spinor, entry.sector]);
        hash.update(entry.output_coordinate.to_le_bytes());
        hash.update(entry.coefficient.real.to_le_bytes());
        hash.update(entry.coefficient.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub(crate) fn build_p3_modular_flat_plan(
    static_data: &ModularFxStaticData,
) -> Result<P3ModularFlatPlan, String> {
    let translation =
        crate::eleven_dimensional_level16_couplings::translation_weight_basis_coefficients();
    let mut offsets = Vec::with_capacity(GAUGE_DEGREE_COUNT * 32 + 1);
    let mut entries = Vec::new();
    offsets.push(0);
    for degree in 0..GAUGE_DEGREE_COUNT {
        for free_spinor in 0..32 {
            let mut schedule =
                std::collections::BTreeMap::<P3ModularPlanKey, GaussianResidue>::new();
            for gauge in &static_data.gauge_by_degree_and_free_spinor[degree * 32 + free_spinor] {
                for contracted_spinor in 0..32 {
                    for contraction_axis in 0..P3_CONTRACTION_AXIS_COUNT {
                        let (translation_real, translation_imaginary) = translation
                            [contracted_spinor][usize::from(gauge.derivative_spinor)]
                            [contraction_axis];
                        if translation_real == 0 && translation_imaginary == 0 {
                            continue;
                        }
                        let translated = gauge.coefficient.multiply(
                            GaussianResidue {
                                real: i128_mod(i128::from(translation_real), static_data.prime),
                                imaginary: i128_mod(
                                    i128::from(translation_imaginary),
                                    static_data.prime,
                                ),
                            },
                            static_data.prime,
                        );
                        for target in &static_data.target {
                            let raw = usize::from(target.vector_weight) * 32
                                + usize::from(target.spinor_weight);
                            let begin = static_data.template_offsets[raw] as usize;
                            let end = static_data.template_offsets[raw + 1] as usize;
                            for template in &static_data.templates[begin..end] {
                                let output_bound = if template.sector == 0 {
                                    P3_X2_OUTPUT_COORDINATES
                                } else {
                                    P3_X5_OUTPUT_COORDINATES
                                };
                                if usize::from(template.output_coordinate) >= output_bound {
                                    return Err(
                                        "p3 F_X ambient output coordinate is out of bounds"
                                            .to_string(),
                                    );
                                }
                                let coefficient = translated
                                    .scale(target.coefficient, static_data.prime)
                                    .multiply(template.coefficient, static_data.prime);
                                if coefficient.is_zero() {
                                    continue;
                                }
                                let key = P3ModularPlanKey {
                                    contracted_spinor: contracted_spinor as u8,
                                    template_spinor: template.derivative_spinor,
                                    contraction_axis: contraction_axis as u8,
                                    sector: template.sector,
                                    output_coordinate: template.output_coordinate,
                                };
                                let value =
                                    schedule.entry(key).or_insert_with(GaussianResidue::zero);
                                *value = value.add(coefficient, static_data.prime);
                                if value.is_zero() {
                                    schedule.remove(&key);
                                }
                            }
                        }
                    }
                }
            }
            entries.extend(
                schedule
                    .into_iter()
                    .map(|(key, coefficient)| P3ModularPlanEntry { key, coefficient }),
            );
            offsets.push(
                u32::try_from(entries.len())
                    .map_err(|_| "p3 modular flat plan exceeds u32".to_string())?,
            );
        }
    }
    if entries.is_empty() || offsets.len() != GAUGE_DEGREE_COUNT * 32 + 1 {
        return Err("p3 modular flat plan is empty or malformed".to_string());
    }
    let mut hash = Sha256::new();
    hash.update(GPU_P3_FX_SCHEMA.as_bytes());
    hash.update(b"\0flat-plan-v1\0");
    hash.update(static_data.prime.to_le_bytes());
    hash.update((P3_FUNCTIONAL_ROW_COUNT as u64).to_le_bytes());
    for offset in &offsets {
        hash.update(offset.to_le_bytes());
    }
    for entry in &entries {
        hash.update([
            entry.key.contracted_spinor,
            entry.key.template_spinor,
            entry.key.contraction_axis,
            entry.key.sector,
        ]);
        hash.update(entry.key.output_coordinate.to_le_bytes());
        hash.update(entry.coefficient.real.to_le_bytes());
        hash.update(entry.coefficient.imaginary.to_le_bytes());
    }
    Ok(P3ModularFlatPlan {
        prime: static_data.prime,
        offsets,
        entries,
        semantic_sha256: format!("{:x}", hash.finalize()),
    })
}

fn validate_p3_modular_flat_plan(plan: &P3ModularFlatPlan) -> Result<(), String> {
    if plan.offsets.len() != GAUGE_DEGREE_COUNT * 32 + 1
        || plan.offsets.first() != Some(&0)
        || plan.offsets.last().copied() != Some(plan.entries.len() as u32)
        || plan.offsets.windows(2).any(|pair| pair[0] > pair[1])
        || plan.entries.is_empty()
    {
        return Err("p3 modular flat plan shape is invalid".to_string());
    }
    for entry in &plan.entries {
        let output_bound = match entry.key.sector {
            0 => P3_X2_OUTPUT_COORDINATES,
            1 => P3_X5_OUTPUT_COORDINATES,
            _ => return Err("p3 modular plan sector is invalid".to_string()),
        };
        if entry.key.contracted_spinor >= 32
            || entry.key.template_spinor >= 32
            || usize::from(entry.key.contraction_axis) >= P3_CONTRACTION_AXIS_COUNT
            || usize::from(entry.key.output_coordinate) >= output_bound
            || entry.coefficient.is_zero()
            || entry.coefficient.real >= plan.prime
            || entry.coefficient.imaginary >= plan.prime
        {
            return Err("p3 modular flat plan entry is invalid".to_string());
        }
    }
    Ok(())
}

pub(crate) fn accumulate_column_cpu(
    static_data: &ModularFxStaticData,
    column: &GpuFxColumnInput,
) -> Result<ModularFunctionalColumn, String> {
    if column.global_ordinal >= 77 || column.raising_residuals != [0; 5] {
        return Err("GPU F_X input column is not certified".to_string());
    }
    let prime = static_data.prime;
    let mut rows = vec![GaussianResidue::zero(); FUNCTIONAL_ROW_COUNT];
    let mut expanded_contributions = 0_u64;
    for source in &column.terms {
        let pair = [
            usize::from(source.momentum_pair[0]),
            usize::from(source.momentum_pair[1]),
        ];
        if pair[0] > pair[1] || pair[1] >= 11 || source.free_spinor >= 32 {
            return Err("invalid packed recoupled source term".to_string());
        }
        for degree in 0..6 {
            for gauge in &static_data.gauge_by_degree_and_free_spinor
                [degree * 32 + usize::from(source.free_spinor)]
            {
                let Some(first_sign) =
                    right_wedge_sign(source.exterior_mask, usize::from(gauge.derivative_spinor))
                else {
                    continue;
                };
                let degree13_mask = source.exterior_mask | (1_u32 << gauge.derivative_spinor);
                let source_value = if first_sign > 0 {
                    GaussianResidue {
                        real: i128_mod(source.coefficient, prime),
                        imaginary: 0,
                    }
                } else {
                    GaussianResidue {
                        real: i128_mod(-source.coefficient, prime),
                        imaginary: 0,
                    }
                };
                let gauged = source_value.multiply(gauge.coefficient, prime);
                for target in &static_data.target {
                    let targeted = gauged.scale(target.coefficient, prime);
                    let raw =
                        usize::from(target.vector_weight) * 32 + usize::from(target.spinor_weight);
                    let begin = static_data.template_offsets[raw] as usize;
                    let end = static_data.template_offsets[raw + 1] as usize;
                    for template in &static_data.templates[begin..end] {
                        let Some(second_sign) = right_wedge_sign(
                            degree13_mask,
                            usize::from(template.derivative_spinor),
                        ) else {
                            continue;
                        };
                        let output_mask = degree13_mask | (1_u32 << template.derivative_spinor);
                        let functional_mask = degree14_to_degree13_mask(output_mask);
                        let sector = if template.sector == 0 {
                            SecondMomentumFxSector::X2
                        } else {
                            SecondMomentumFxSector::X5
                        };
                        let mut value = targeted.multiply(template.coefficient, prime);
                        if second_sign < 0 {
                            value = value.negate(prime);
                        }
                        if value.is_zero() {
                            continue;
                        }
                        let momentum = DegreeTwoMomentumMonomial::from_pair(pair[0], pair[1])?;
                        let assignments = crate::eleven_dimensional_second_momentum_fx::
                            second_momentum_fx_functional_assignments(
                                SecondMomentumGaugeChannel::new(degree)?,
                                SecondMomentumGaugeBranch::P2D13Wedge,
                                momentum,
                                0,
                                usize::from(template.output_coordinate),
                                functional_mask,
                                sector,
                            );
                        for (seed, (bucket, sign)) in assignments
                            .into_iter()
                            .take(GPU_FX_FUNCTIONAL_SEEDS.len())
                            .enumerate()
                        {
                            let row = functional_row(degree, pair, sector, seed, bucket);
                            let contribution = if sign > 0 { value } else { value.negate(prime) };
                            rows[row] = rows[row].add(contribution, prime);
                        }
                        expanded_contributions += 1;
                    }
                }
            }
        }
    }
    let semantic_sha256 = column_semantic_sha256(
        prime,
        column.global_ordinal,
        &static_data.semantic_sha256,
        &rows,
    );
    Ok(ModularFunctionalColumn {
        prime,
        global_ordinal: column.global_ordinal,
        rows,
        expanded_contributions,
        semantic_sha256,
    })
}

/// Proof-safe CPU reference for the complete contraction branch. This is the
/// parity oracle for the CUDA p3 plan, not the intended full-run execution
/// path.
pub(crate) fn accumulate_p3_column_cpu(
    exact_static_data: &P3D11ExactStaticData,
    modular_static_data: &ModularFxStaticData,
    column: &GpuFxColumnInput,
) -> Result<ModularP3FunctionalColumn, String> {
    if column.global_ordinal >= 77 || column.raising_residuals != [0; 5] {
        return Err("p3 F_X input column is not certified".to_string());
    }
    let prime = modular_static_data.prime;
    let mut rows = vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
    let mut expanded_contributions = 0_u64;
    for degree in 0..GAUGE_DEGREE_COUNT {
        let contractions = produce_p3_d11_contractions(
            exact_static_data,
            column.global_ordinal,
            degree,
            0,
            &column.terms,
        )?;
        let projected = project_p3_d11_to_physical_fx(exact_static_data, &contractions)?;
        for term in projected {
            let pair = term.source_momentum.canonical_pair()?;
            let pair = pair_ordinal(pair[0], pair[1]);
            let contraction_axis = match term.gauge_branch {
                SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis } => {
                    usize::from(momentum_axis)
                }
                SecondMomentumGaugeBranch::P2D13Wedge => {
                    return Err("p3 projection emitted a wedge branch".to_string());
                }
            };
            let sector = match term.sector {
                SecondMomentumFxSector::X2 => 0,
                SecondMomentumFxSector::X5 => 1,
            };
            let value = gaussian_mod(&term.coefficient, prime)?;
            for (seed, (bucket, sign)) in second_momentum_fx_functional_assignments(
                term.gauge_channel,
                term.gauge_branch,
                term.source_momentum,
                term.parameter_component,
                term.target_coordinate,
                term.spinor_derivative_mask,
                term.sector,
            )
            .into_iter()
            .take(GPU_FX_FUNCTIONAL_SEEDS.len())
            .enumerate()
            {
                let row = p3_functional_row(P3FunctionalRowCoordinates {
                    gauge_degree: degree,
                    momentum_pair_ordinal: pair,
                    sector,
                    contraction_axis,
                    seed,
                    bucket,
                })?;
                let contribution = if sign > 0 { value } else { value.negate(prime) };
                rows[row] = rows[row].add(contribution, prime);
                expanded_contributions = expanded_contributions
                    .checked_add(1)
                    .ok_or_else(|| "p3 expanded contribution count overflow".to_string())?;
            }
        }
    }
    let semantic_sha256 = p3_column_semantic_sha256(prime, column.global_ordinal, &rows);
    Ok(ModularP3FunctionalColumn {
        prime,
        global_ordinal: column.global_ordinal,
        rows,
        expanded_contributions,
        semantic_sha256,
    })
}

pub(crate) fn accumulate_p3_column_cpu_flat(
    plan: &P3ModularFlatPlan,
    column: &GpuFxColumnInput,
) -> Result<ModularP3FunctionalColumn, String> {
    validate_p3_modular_flat_plan(plan)?;
    if column.global_ordinal >= 77 || column.raising_residuals != [0; 5] {
        return Err("p3 flat F_X input column is not certified".to_string());
    }
    let prime = plan.prime;
    let mut rows = vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
    let mut expanded_contributions = 0_u64;
    for source in &column.terms {
        let left = usize::from(source.momentum_pair[0]);
        let right = usize::from(source.momentum_pair[1]);
        if left > right
            || right >= 11
            || source.free_spinor >= 32
            || source.exterior_mask.count_ones() != 12
            || source.coefficient == 0
        {
            return Err("p3 flat source term is not canonical".to_string());
        }
        let pair_ordinal = pair_ordinal(left, right);
        let source_momentum = DegreeTwoMomentumMonomial::from_pair(left, right)?;
        let source_value = GaussianResidue {
            real: i128_mod(source.coefficient, prime),
            imaginary: 0,
        };
        for degree in 0..GAUGE_DEGREE_COUNT {
            let schedule = degree * 32 + usize::from(source.free_spinor);
            let begin = plan.offsets[schedule] as usize;
            let end = plan.offsets[schedule + 1] as usize;
            for entry in &plan.entries[begin..end] {
                let contracted = usize::from(entry.key.contracted_spinor);
                let Some(contraction_sign) =
                    crate::eleven_dimensional_level16_couplings::right_contraction_sign(
                        source.exterior_mask,
                        contracted,
                    )
                else {
                    continue;
                };
                let degree_eleven_mask = source.exterior_mask ^ (1_u32 << contracted);
                let Some(wedge_sign) =
                    right_wedge_sign(degree_eleven_mask, usize::from(entry.key.template_spinor))
                else {
                    continue;
                };
                let degree_twelve_mask = degree_eleven_mask | (1_u32 << entry.key.template_spinor);
                if degree_twelve_mask.count_ones() != 12 {
                    return Err("p3 flat projection lost derivative degree".to_string());
                }
                let highest = 31 - degree_twelve_mask.leading_zeros();
                let functional_mask = degree_twelve_mask ^ (1_u32 << highest);
                let mut value = source_value.multiply(entry.coefficient, prime);
                if (contraction_sign < 0) != (wedge_sign < 0) {
                    value = value.negate(prime);
                }
                if value.is_zero() {
                    continue;
                }
                let sector = usize::from(entry.key.sector);
                let gauge_channel = SecondMomentumGaugeChannel::new(degree)?;
                let gauge_branch = SecondMomentumGaugeBranch::P3D11Contraction {
                    momentum_axis: entry.key.contraction_axis,
                };
                let fx_sector = if sector == 0 {
                    SecondMomentumFxSector::X2
                } else {
                    SecondMomentumFxSector::X5
                };
                for (seed, (bucket, sign)) in second_momentum_fx_functional_assignments(
                    gauge_channel,
                    gauge_branch,
                    source_momentum,
                    0,
                    usize::from(entry.key.output_coordinate),
                    functional_mask,
                    fx_sector,
                )
                .into_iter()
                .take(GPU_FX_FUNCTIONAL_SEEDS.len())
                .enumerate()
                {
                    let row = p3_functional_row(P3FunctionalRowCoordinates {
                        gauge_degree: degree,
                        momentum_pair_ordinal: pair_ordinal,
                        sector,
                        contraction_axis: usize::from(entry.key.contraction_axis),
                        seed,
                        bucket,
                    })?;
                    let contribution = if sign > 0 { value } else { value.negate(prime) };
                    rows[row] = rows[row].add(contribution, prime);
                }
                expanded_contributions = expanded_contributions
                    .checked_add(1)
                    .ok_or_else(|| "p3 flat expanded contribution count overflow".to_string())?;
            }
        }
    }
    let semantic_sha256 = p3_column_semantic_sha256(prime, column.global_ordinal, &rows);
    Ok(ModularP3FunctionalColumn {
        prime,
        global_ordinal: column.global_ordinal,
        rows,
        expanded_contributions,
        semantic_sha256,
    })
}

pub(crate) fn p3_column_semantic_sha256(
    prime: u32,
    global_ordinal: usize,
    rows: &[GaussianResidue],
) -> String {
    let mut hash = Sha256::new();
    hash.update(GPU_P3_FX_SCHEMA.as_bytes());
    hash.update(prime.to_le_bytes());
    hash.update((global_ordinal as u64).to_le_bytes());
    hash.update((P3_FUNCTIONAL_ROW_COUNT as u64).to_le_bytes());
    for value in rows {
        hash.update(value.real.to_le_bytes());
        hash.update(value.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn p3_artifact_semantic_sha256(
    flat_plan_sha256: &str,
    column: &ModularP3FunctionalColumn,
) -> String {
    let mut hash = Sha256::new();
    hash.update(GPU_P3_FX_SCHEMA.as_bytes());
    hash.update(column.prime.to_le_bytes());
    hash.update((column.global_ordinal as u64).to_le_bytes());
    hash.update((P3_FUNCTIONAL_ROW_COUNT as u64).to_le_bytes());
    hash.update(column.expanded_contributions.to_le_bytes());
    hash.update(flat_plan_sha256.as_bytes());
    for value in &column.rows {
        hash.update(value.real.to_le_bytes());
        hash.update(value.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

const GPU_P3_ARTIFACT_MAGIC: &[u8; 8] = b"ADFXP3V1";
const GPU_P3_ARTIFACT_HEADER_BYTES: usize = 156;

pub(crate) fn encode_p3_column_artifact(
    flat_plan_sha256: &str,
    column: &ModularP3FunctionalColumn,
) -> Result<Vec<u8>, String> {
    if column.global_ordinal >= 77
        || column.rows.len() != P3_FUNCTIONAL_ROW_COUNT
        || flat_plan_sha256.len() != 64
        || !flat_plan_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || column.semantic_sha256.len() != 64
        || column.semantic_sha256
            != p3_column_semantic_sha256(column.prime, column.global_ordinal, &column.rows)
        || column
            .rows
            .iter()
            .any(|value| value.real >= column.prime || value.imaginary >= column.prime)
    {
        return Err("p3 column artifact input is not canonical".to_string());
    }
    let mut bytes = Vec::with_capacity(GPU_P3_ARTIFACT_HEADER_BYTES + column.rows.len() * 8);
    bytes.extend_from_slice(GPU_P3_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&column.prime.to_le_bytes());
    bytes.extend_from_slice(&(column.global_ordinal as u32).to_le_bytes());
    bytes.extend_from_slice(&(P3_FUNCTIONAL_ROW_COUNT as u32).to_le_bytes());
    bytes.extend_from_slice(&column.expanded_contributions.to_le_bytes());
    bytes.extend_from_slice(flat_plan_sha256.as_bytes());
    bytes.extend_from_slice(p3_artifact_semantic_sha256(flat_plan_sha256, column).as_bytes());
    debug_assert_eq!(bytes.len(), GPU_P3_ARTIFACT_HEADER_BYTES);
    for value in &column.rows {
        bytes.extend_from_slice(&value.real.to_le_bytes());
        bytes.extend_from_slice(&value.imaginary.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) fn decode_p3_column_artifact(
    bytes: &[u8],
) -> Result<(String, ModularP3FunctionalColumn), String> {
    let expected = GPU_P3_ARTIFACT_HEADER_BYTES
        .checked_add(P3_FUNCTIONAL_ROW_COUNT * 8)
        .ok_or_else(|| "p3 artifact size overflow".to_string())?;
    if bytes.len() != expected || &bytes[..8] != GPU_P3_ARTIFACT_MAGIC {
        return Err("p3 column artifact magic or byte length is invalid".to_string());
    }
    let read_u32 =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let prime = read_u32(8);
    validate_prime(prime)?;
    let global_ordinal = read_u32(12) as usize;
    let row_count = read_u32(16) as usize;
    let expanded_contributions = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    if global_ordinal >= 77 || row_count != P3_FUNCTIONAL_ROW_COUNT {
        return Err("p3 column artifact header is invalid".to_string());
    }
    let flat_plan_sha256 = std::str::from_utf8(&bytes[28..92])
        .map_err(|_| "p3 flat plan digest is not UTF-8".to_string())?
        .to_string();
    let stored_semantic = std::str::from_utf8(&bytes[92..156])
        .map_err(|_| "p3 column semantic digest is not UTF-8".to_string())?
        .to_string();
    if !flat_plan_sha256
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit())
        || !stored_semantic.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("p3 artifact digest encoding is invalid".to_string());
    }
    let mut rows = Vec::with_capacity(P3_FUNCTIONAL_ROW_COUNT);
    for chunk in bytes[GPU_P3_ARTIFACT_HEADER_BYTES..].chunks_exact(8) {
        let value = GaussianResidue {
            real: u32::from_le_bytes(chunk[..4].try_into().unwrap()),
            imaginary: u32::from_le_bytes(chunk[4..].try_into().unwrap()),
        };
        if value.real >= prime || value.imaginary >= prime {
            return Err("p3 artifact contains a noncanonical residue".to_string());
        }
        rows.push(value);
    }
    let semantic_sha256 = p3_column_semantic_sha256(prime, global_ordinal, &rows);
    let column = ModularP3FunctionalColumn {
        prime,
        global_ordinal,
        rows,
        expanded_contributions,
        semantic_sha256,
    };
    if p3_artifact_semantic_sha256(&flat_plan_sha256, &column) != stored_semantic {
        return Err("p3 artifact semantic digest mismatch".to_string());
    }
    Ok((flat_plan_sha256, column))
}

fn column_semantic_sha256(
    prime: u32,
    global_ordinal: usize,
    static_digest: &str,
    rows: &[GaussianResidue],
) -> String {
    let mut hash = Sha256::new();
    hash.update(GPU_FX_SCHEMA.as_bytes());
    hash.update(prime.to_le_bytes());
    hash.update((global_ordinal as u64).to_le_bytes());
    hash.update(static_digest.as_bytes());
    for value in rows {
        hash.update(value.real.to_le_bytes());
        hash.update(value.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub(crate) fn rank_columns(
    columns: &[ModularFunctionalColumn],
) -> Result<ModularRankCertificate, String> {
    if columns.is_empty() {
        return Err("cannot rank an empty modular F_X column set".to_string());
    }
    let prime = columns[0].prime;
    let mut ordinals = Vec::with_capacity(columns.len());
    for column in columns {
        if column.prime != prime || column.rows.len() != FUNCTIONAL_ROW_COUNT {
            return Err("incompatible modular F_X columns".to_string());
        }
        ordinals.push(column.global_ordinal);
    }
    if ordinals
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != ordinals.len()
    {
        return Err("duplicate modular F_X column ordinal".to_string());
    }

    let width = columns.len();
    // Pivot indices are dense and bounded by the small physical column
    // inventory. A direct table avoids tree lookups, and one reusable row
    // avoids an allocation for every functional coordinate.
    let mut basis = vec![None::<Vec<GaussianResidue>>; width];
    let mut rank = 0_usize;
    let mut row = vec![GaussianResidue::zero(); width];
    for row_index in 0..FUNCTIONAL_ROW_COUNT {
        for (value, column) in row.iter_mut().zip(columns) {
            *value = column.rows[row_index];
        }
        loop {
            let Some(pivot) = row.iter().position(|value| !value.is_zero()) else {
                break;
            };
            if let Some(existing) = &basis[pivot] {
                // Stored pivots are normalized, so the elimination factor is
                // already row[pivot]. Avoid a finite-field inversion here.
                let factor = row[pivot];
                for column in pivot..width {
                    row[column] = row[column].add(
                        factor.multiply(existing[column], prime).negate(prime),
                        prime,
                    );
                }
            } else {
                let inverse = gaussian_inverse(row[pivot], prime);
                for value in &mut row[pivot..] {
                    *value = value.multiply(inverse, prime);
                }
                basis[pivot] = Some(row.clone());
                rank += 1;
                break;
            }
        }
        if rank == width {
            break;
        }
    }
    let mut hash = Sha256::new();
    hash.update(GPU_FX_SCHEMA.as_bytes());
    hash.update(prime.to_le_bytes());
    for ordinal in &ordinals {
        hash.update((*ordinal as u64).to_le_bytes());
    }
    let mut hash_row = Vec::with_capacity(width * 8);
    for row in 0..FUNCTIONAL_ROW_COUNT {
        hash_row.clear();
        for column in columns {
            hash_row.extend_from_slice(&column.rows[row].real.to_le_bytes());
            hash_row.extend_from_slice(&column.rows[row].imaginary.to_le_bytes());
        }
        hash.update(&hash_row);
    }
    Ok(ModularRankCertificate {
        schema_version: GPU_FX_SCHEMA,
        prime,
        row_count: FUNCTIONAL_ROW_COUNT,
        column_ordinals: ordinals,
        rank_over_gaussian_extension: rank,
        nullity_upper_bound: width - rank,
        full_column_rank: rank == width,
        matrix_sha256: format!("{:x}", hash.finalize()),
    })
}

pub(crate) fn rank_p3_columns(
    columns: &[ModularP3FunctionalColumn],
) -> Result<ModularRankCertificate, String> {
    if columns.is_empty() {
        return Err("cannot rank an empty modular p3 F_X column set".to_string());
    }
    let prime = columns[0].prime;
    if !GPU_FX_PRIMES.contains(&prime) {
        return Err("modular p3 F_X rank prime is not pinned".to_string());
    }
    let mut ordinals = Vec::with_capacity(columns.len());
    for column in columns {
        if column.prime != prime
            || column.global_ordinal >= 77
            || column.rows.len() != P3_FUNCTIONAL_ROW_COUNT
            || column
                .rows
                .iter()
                .any(|value| value.real >= prime || value.imaginary >= prime)
            || column.semantic_sha256
                != p3_column_semantic_sha256(prime, column.global_ordinal, &column.rows)
        {
            return Err("incompatible modular p3 F_X columns".to_string());
        }
        ordinals.push(column.global_ordinal);
    }
    if ordinals
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != ordinals.len()
    {
        return Err("duplicate modular p3 F_X column ordinal".to_string());
    }
    let width = columns.len();
    let mut basis = vec![None::<Vec<GaussianResidue>>; width];
    let mut rank = 0;
    let mut row = vec![GaussianResidue::zero(); width];
    for row_index in 0..P3_FUNCTIONAL_ROW_COUNT {
        for (value, column) in row.iter_mut().zip(columns) {
            *value = column.rows[row_index];
        }
        loop {
            let Some(pivot) = row.iter().position(|value| !value.is_zero()) else {
                break;
            };
            if let Some(existing) = &basis[pivot] {
                let factor = row[pivot];
                for column in pivot..width {
                    row[column] = row[column].add(
                        factor.multiply(existing[column], prime).negate(prime),
                        prime,
                    );
                }
            } else {
                let inverse = gaussian_inverse(row[pivot], prime);
                for value in &mut row[pivot..] {
                    *value = value.multiply(inverse, prime);
                }
                basis[pivot] = Some(row.clone());
                rank += 1;
                break;
            }
        }
        if rank == width {
            break;
        }
    }
    let mut hash = Sha256::new();
    hash.update(GPU_P3_FX_SCHEMA.as_bytes());
    hash.update(prime.to_le_bytes());
    for ordinal in &ordinals {
        hash.update((*ordinal as u64).to_le_bytes());
    }
    for row_index in 0..P3_FUNCTIONAL_ROW_COUNT {
        for column in columns {
            hash.update(column.rows[row_index].real.to_le_bytes());
            hash.update(column.rows[row_index].imaginary.to_le_bytes());
        }
    }
    Ok(ModularRankCertificate {
        schema_version: GPU_P3_FX_SCHEMA,
        prime,
        row_count: P3_FUNCTIONAL_ROW_COUNT,
        column_ordinals: ordinals,
        rank_over_gaussian_extension: rank,
        nullity_upper_bound: width - rank,
        full_column_rank: rank == width,
        matrix_sha256: format!("{:x}", hash.finalize()),
    })
}

fn gaussian_inverse(value: GaussianResidue, prime: u32) -> GaussianResidue {
    let norm = add_mod(
        multiply_mod(value.real, value.real, prime),
        multiply_mod(value.imaginary, value.imaginary, prime),
        prime,
    );
    let inverse_norm = inverse_mod(norm, prime);
    GaussianResidue {
        real: multiply_mod(value.real, inverse_norm, prime),
        imaginary: multiply_mod(negate_mod(value.imaginary, prime), inverse_norm, prime),
    }
}

#[cfg(feature = "cuda")]
mod cuda_backend {
    use std::cell::{Cell, RefCell};
    #[cfg(test)]
    use std::collections::BTreeMap;
    use std::ffi::{CStr, c_char, c_void};
    use std::marker::PhantomData;
    use std::ptr::NonNull;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use super::*;

    const ERROR_CAPACITY: usize = 1024;
    const RECOUPLING_METADATA_BITS: u32 = 13;

    fn pack_recoupling_key(term: &RecoupledSourceTerm) -> Result<u64, String> {
        p3_recoupling_key(term)
    }

    fn unpack_recoupling_key(key: u64) -> Result<([u8; 2], u8, u32), String> {
        let metadata = (key >> 32) as u32;
        let pair = [(metadata & 15) as u8, ((metadata >> 4) & 15) as u8];
        let free_spinor = ((metadata >> 8) & 31) as u8;
        let exterior_mask = key as u32;
        if metadata >> RECOUPLING_METADATA_BITS != 0
            || pair[0] > pair[1]
            || pair[1] >= 11
            || exterior_mask.count_ones() != 12
        {
            return Err("invalid canonical CUDA recoupling key".to_string());
        }
        Ok((pair, free_spinor, exterior_mask))
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CudaGaugeEntry {
        derivative_spinor: u32,
        coefficient: GaussianResidue,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CudaTargetEntry {
        vector_weight: u32,
        spinor_weight: u32,
        coefficient: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CudaTemplateEntry {
        derivative_spinor: u32,
        sector: u32,
        output_coordinate: u32,
        coefficient: GaussianResidue,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(C)]
    struct CudaPlanEntry {
        gauge_spinor: u32,
        template_spinor: u32,
        sector: u32,
        output_coordinate: u32,
        coefficient: GaussianResidue,
        functional_salt: u64,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CudaP3PlanEntry {
        contracted_spinor: u32,
        template_spinor: u32,
        contraction_axis: u32,
        sector: u32,
        output_coordinate: u32,
        coefficient: GaussianResidue,
        functional_salt: u64,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CudaP3ThreePrimePlanEntry {
        contracted_spinor: u32,
        template_spinor: u32,
        contraction_axis: u32,
        sector: u32,
        output_coordinate: u32,
        scaled_real: i16,
        scaled_imaginary: i16,
        functional_salt: u64,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CudaSourceEntry {
        exterior_mask: u32,
        coefficient: u32,
        metadata: u32,
    }

    #[derive(Clone, Copy, Debug, Default)]
    #[repr(C)]
    struct CudaWideValue {
        low: u64,
        high: i64,
        overflow: u32,
        reserved: u32,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(C)]
    struct CudaSparseEntry {
        key: u64,
        value: i64,
    }

    #[derive(Clone, Copy, Debug, Default, Serialize)]
    #[repr(C)]
    pub(crate) struct CudaSparseLoweringStats {
        pub input_count: u64,
        pub expanded_count: u64,
        pub reduced_count: u64,
        pub output_count: u64,
        pub scratch_high_water_bytes: u64,
        pub immutable_handle_bytes: u64,
        pub count_milliseconds: f32,
        pub scan_milliseconds: f32,
        pub emit_milliseconds: f32,
        pub sort_milliseconds: f32,
        pub reduce_milliseconds: f32,
        pub select_milliseconds: f32,
        pub total_milliseconds: f32,
    }

    #[derive(Clone, Copy, Debug, Default, Serialize)]
    pub(crate) struct PersistentLoweringSummary {
        pub enabled: bool,
        pub roots_lowered: u64,
        pub input_entry_visits: u64,
        pub expanded_entry_visits: u64,
        pub output_entry_visits: u64,
        pub gpu_milliseconds: f64,
        pub scratch_high_water_bytes: u64,
        pub peak_immutable_handle_bytes: u64,
        pub maximum_absolute_coefficient: u64,
        pub device_hard_cap_bytes: u64,
        pub download_chunk_terms: usize,
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct PersistentRootProgress {
        pub phase: &'static str,
        pub word_ordinal: usize,
        pub root: usize,
        pub stats: CudaSparseLoweringStats,
        pub resident_bytes: u64,
    }

    #[derive(Clone, Copy, Debug, Default)]
    #[repr(C)]
    struct CudaRecouplingStats {
        terms_before_reduce: u64,
        keys_after_reduce: u64,
        nonzero_terms_after_reduce: u64,
        expanded_contributions: u64,
        buffer_high_water_bytes: u64,
        upload_milliseconds: f32,
        sort_milliseconds: f32,
        reduce_milliseconds: f32,
        contract_milliseconds: f32,
        download_milliseconds: f32,
        total_milliseconds: f32,
    }

    #[derive(Clone, Copy, Debug, Default, Serialize)]
    #[repr(C)]
    pub(crate) struct CudaMultiColumnStats {
        pub unique_count: u64,
        pub active_columns: u32,
        reserved: u32,
        pub nonzero_terms: [u64; 32],
        pub expanded_contributions: [u64; 32],
        pub resident_bytes: u64,
        pub buffer_high_water_bytes: u64,
        pub device_hard_cap_bytes: u64,
        pub upload_milliseconds: f32,
        pub contract_milliseconds: f32,
        pub finalize_milliseconds: f32,
        pub download_milliseconds: f32,
        pub total_milliseconds: f32,
    }

    #[derive(Clone, Debug)]
    pub(crate) struct CudaMultiColumnBatch {
        pub columns: Vec<Vec<GaussianResidue>>,
        pub stats: CudaMultiColumnStats,
    }

    unsafe extern "C" {
        fn adynkra_fx_cuda_create(
            device: i32,
            prime: u32,
            gauge_offsets: *const u32,
            gauges: *const CudaGaugeEntry,
            gauge_count: u32,
            targets: *const CudaTargetEntry,
            target_count: u32,
            template_offsets: *const u32,
            templates: *const CudaTemplateEntry,
            template_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn adynkra_fx_cuda_create_v2(
            device: i32,
            prime: u32,
            gauge_offsets: *const u32,
            gauges: *const CudaGaugeEntry,
            gauge_count: u32,
            targets: *const CudaTargetEntry,
            target_count: u32,
            template_offsets: *const u32,
            templates: *const CudaTemplateEntry,
            template_count: u32,
            plan_offsets: *const u32,
            plan_entries: *const CudaPlanEntry,
            plan_entry_count: u32,
            pair_salts: *const u64,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn adynkra_fx_cuda_configure_p3(
            context: *mut c_void,
            plan_offsets: *const u32,
            plan_entries: *const CudaP3PlanEntry,
            plan_entry_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_configure_p3_three_prime(
            context: *mut c_void,
            primes: *const u32,
            inverse_scales: *const u32,
            plan_offsets: *const u32,
            plan_entries: *const CudaP3ThreePrimePlanEntry,
            plan_entry_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_accumulate_p3(
            context: *mut c_void,
            sources: *const CudaSourceEntry,
            source_count: u32,
            output_real: *mut u32,
            output_imaginary: *mut u32,
            expanded_contributions: *mut u64,
            kernel_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_reset_p3_columns(
            context: *mut c_void,
            input_real: *const u32,
            input_imaginary: *const u32,
            active_columns: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_accumulate_p3_column(
            context: *mut c_void,
            column: u32,
            sources: *const CudaSourceEntry,
            source_count: u32,
            expanded_contributions: *mut u64,
            kernel_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_accumulate_p3_multicol(
            context: *mut c_void,
            keys: *const u64,
            key_major_coefficients: *const u32,
            unique_count: u32,
            active_columns: u32,
            expanded_contributions: *mut u64,
            kernel_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_download_p3_columns(
            context: *mut c_void,
            active_columns: u32,
            output_real: *mut u32,
            output_imaginary: *mut u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_reset_p3_three_prime_columns(
            context: *mut c_void,
            input_real: *const u32,
            input_imaginary: *const u32,
            active_columns: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_accumulate_p3_three_prime_multicol(
            context: *mut c_void,
            keys: *const u64,
            key_prime_column_coefficients: *const u32,
            unique_count: u32,
            active_columns: u32,
            expanded_contributions: *mut u64,
            kernel_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_download_p3_three_prime_columns(
            context: *mut c_void,
            active_columns: u32,
            output_real: *mut u32,
            output_imaginary: *mut u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_accumulate_recoupled(
            context: *mut c_void,
            keys: *const u64,
            values: *const CudaWideValue,
            source_count: u32,
            output_real: *mut u32,
            output_imaginary: *mut u32,
            stats: *mut CudaRecouplingStats,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_reserve_multicol(
            context: *mut c_void,
            unique_capacity: u32,
            active_columns: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_accumulate_recoupled_multicol(
            context: *mut c_void,
            keys: *const u64,
            key_major_values: *const CudaWideValue,
            unique_count: u32,
            active_columns: u32,
            row_major_output_real: *mut u32,
            row_major_output_imaginary: *mut u32,
            stats: *mut CudaMultiColumnStats,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_set_recoupling_hard_cap(
            context: *mut c_void,
            hard_cap_bytes: u64,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_resident_bytes(context: *const c_void) -> u64;
        fn adynkra_fx_cuda_buffer_high_water_bytes(context: *const c_void) -> u64;
        fn adynkra_fx_cuda_hard_cap_bytes(context: *const c_void) -> u64;
        fn adynkra_fx_cuda_p3_plan_entry_count(context: *const c_void) -> u32;
        fn adynkra_fx_cuda_reserve_recoupling(
            context: *mut c_void,
            source_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_set_legacy_contraction(
            context: *mut c_void,
            enabled: i32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_destroy(context: *mut c_void);
        fn adynkra_fx_cuda_device_name(
            device: i32,
            name: *mut c_char,
            capacity: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_lower_sparse(
            device: i32,
            source_keys: *const u64,
            source_values: *const i64,
            source_count: u32,
            root: u32,
            output_keys: *mut u64,
            output_values: *mut i64,
            output_capacity: u32,
            output_count: *mut u32,
            kernel_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_sparse_context_create(
            device: i32,
            hard_cap_bytes: u64,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn adynkra_fx_cuda_sparse_context_destroy(context: *mut c_void);
        fn adynkra_fx_cuda_sparse_resident_bytes(context: *const c_void) -> u64;
        fn adynkra_fx_cuda_sparse_handle_upload(
            context: *mut c_void,
            entries: *const CudaSparseEntry,
            count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn adynkra_fx_cuda_sparse_handle_lower(
            context: *mut c_void,
            handle: *const c_void,
            root: u32,
            stats: *mut CudaSparseLoweringStats,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn adynkra_fx_cuda_sparse_handle_count(handle: *const c_void) -> u32;
        fn adynkra_fx_cuda_sparse_handle_max_abs(handle: *const c_void) -> u64;
        fn adynkra_fx_cuda_sparse_handle_download_range(
            context: *mut c_void,
            handle: *const c_void,
            start: u32,
            entries: *mut CudaSparseEntry,
            capacity: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_sparse_handle_download(
            context: *mut c_void,
            handle: *const c_void,
            entries: *mut CudaSparseEntry,
            capacity: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_fx_cuda_sparse_handle_destroy(handle: *mut c_void);
    }

    #[derive(Debug)]
    pub(crate) struct CudaModularFx {
        context: NonNull<c_void>,
        prime: u32,
        static_semantic_sha256: String,
        flat_plan_sha256: String,
        device_name: String,
        multicol_host_values: Vec<CudaWideValue>,
        multicol_host_real: Vec<u32>,
        multicol_host_imaginary: Vec<u32>,
        _not_send_or_sync: PhantomData<Rc<()>>,
    }

    #[derive(Clone, Debug, Serialize)]
    pub(crate) struct CudaColumnTiming {
        pub source_terms: usize,
        pub keys_after_reduce: u64,
        pub nonzero_terms_after_reduce: u64,
        pub expanded_contributions: u64,
        pub kernel_milliseconds: f32,
        pub upload_milliseconds: f32,
        pub sort_milliseconds: f32,
        pub reduce_milliseconds: f32,
        pub contract_milliseconds: f32,
        pub download_milliseconds: f32,
        pub buffer_high_water_bytes: u64,
        pub packed_input_sha256: String,
        pub input_terms_per_second: f64,
        pub batches: u64,
        pub peak_batch_terms: usize,
        pub batch_term_cap: usize,
        pub host_hard_cap_bytes: u64,
        pub device_hard_cap_bytes: u64,
    }

    #[derive(Clone, Debug)]
    pub(crate) struct CudaBatchCompletion {
        pub logical_batch_ordinal: u64,
        pub source_terms: usize,
        pub row_delta: Vec<GaussianResidue>,
        pub row_delta_sha256: String,
        pub timing: CudaColumnTiming,
    }

    const DEFAULT_STREAM_BATCH_TERMS: usize = 262_144;
    const DEFAULT_STREAM_HOST_HARD_CAP_BYTES: u64 = 256 * 1024 * 1024;
    // The persistent PBW workspace can exceed 2 GiB on a valid level-12
    // word. CUDA performs an independent free-memory plus 64 MiB headroom
    // check before every growth, so this is a logical combined budget rather
    // than an eager allocation.
    const DEFAULT_STREAM_DEVICE_HARD_CAP_BYTES: u64 = 16 * 1024 * 1024 * 1024;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct CudaStreamingConfig {
        pub batch_terms: usize,
        pub host_hard_cap_bytes: u64,
        pub device_hard_cap_bytes: u64,
    }

    impl Default for CudaStreamingConfig {
        fn default() -> Self {
            Self {
                batch_terms: DEFAULT_STREAM_BATCH_TERMS,
                host_hard_cap_bytes: DEFAULT_STREAM_HOST_HARD_CAP_BYTES,
                device_hard_cap_bytes: DEFAULT_STREAM_DEVICE_HARD_CAP_BYTES,
            }
        }
    }

    impl CudaStreamingConfig {
        pub(crate) fn from_environment() -> Result<Self, String> {
            let defaults = Self::default();
            Ok(Self {
                batch_terms: environment_usize("ADYNKRA_GPU_FX_BATCH_TERMS", defaults.batch_terms)?,
                host_hard_cap_bytes: environment_u64(
                    "ADYNKRA_GPU_FX_HOST_CAP_BYTES",
                    defaults.host_hard_cap_bytes,
                )?,
                device_hard_cap_bytes: environment_u64(
                    "ADYNKRA_GPU_FX_DEVICE_CAP_BYTES",
                    defaults.device_hard_cap_bytes,
                )?,
            })
        }

        fn validate(self) -> Result<Self, String> {
            if self.batch_terms == 0 || self.batch_terms > u32::MAX as usize {
                return Err("CUDA streaming batch term cap must lie in 1..=u32::MAX".to_string());
            }
            let required = self.required_host_bytes(0)?;
            if required > self.host_hard_cap_bytes {
                return Err(format!(
                    "CUDA streaming batch requires {required} host bytes, above hard cap {}",
                    self.host_hard_cap_bytes
                ));
            }
            Ok(self)
        }

        pub(crate) fn required_host_bytes(
            self,
            retained_prefix_terms: usize,
        ) -> Result<u64, String> {
            let per_term = std::mem::size_of::<RecoupledSourceTerm>()
                + std::mem::size_of::<u64>()
                + std::mem::size_of::<CudaWideValue>();
            let fixed = FUNCTIONAL_ROW_COUNT
                .checked_mul(3 * std::mem::size_of::<GaussianResidue>())
                .ok_or_else(|| "CUDA streaming fixed host size overflow".to_string())?;
            let required = self
                .batch_terms
                .checked_mul(per_term)
                .and_then(|bytes| bytes.checked_add(fixed))
                .and_then(|bytes| {
                    retained_prefix_terms
                        .checked_mul(std::mem::size_of::<RecoupledSourceTerm>())
                        .and_then(|prefix| bytes.checked_add(prefix))
                })
                .ok_or_else(|| "CUDA streaming host size overflow".to_string())?;
            u64::try_from(required).map_err(|_| "CUDA streaming host size exceeds u64".to_string())
        }
    }

    fn environment_u64(name: &str, default: u64) -> Result<u64, String> {
        match std::env::var(name) {
            Ok(value) => value
                .parse::<u64>()
                .map_err(|error| format!("invalid {name}={value}: {error}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(format!("cannot read {name}: {error}")),
        }
    }

    fn environment_usize(name: &str, default: usize) -> Result<usize, String> {
        let value = environment_u64(name, default as u64)?;
        usize::try_from(value).map_err(|_| format!("{name} exceeds usize"))
    }

    fn flat_functional_salt(degree: usize, output_coordinate: u32, sector: u32) -> u64 {
        (degree as u64).rotate_left(9)
            ^ u64::from(output_coordinate).rotate_left(31)
            ^ 0x02d1_3000_0000_0001
            ^ if sector == 0 {
                0x1100_0000_0000_0002
            } else {
                0x1000_2000_0000_0005
            }
    }

    fn flat_pair_salts() -> [u64; MOMENTUM_PAIR_COUNT] {
        let mut salts = [0_u64; MOMENTUM_PAIR_COUNT];
        for left in 0..11 {
            for right in left..11 {
                let mut salt = 0_u64;
                for axis in 0..11 {
                    let exponent = usize::from(axis == left) + usize::from(axis == right);
                    salt ^= (exponent as u64 + 1)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15_u64.rotate_left(axis as u32));
                }
                salts[pair_ordinal(left, right)] = salt;
            }
        }
        salts
    }

    fn build_flat_plan(
        static_data: &ModularFxStaticData,
    ) -> Result<(Vec<u32>, Vec<CudaPlanEntry>, [u64; MOMENTUM_PAIR_COUNT]), String> {
        let mut offsets = Vec::with_capacity(GAUGE_DEGREE_COUNT * 32 + 1);
        let mut entries = Vec::new();
        offsets.push(0);
        for degree in 0..GAUGE_DEGREE_COUNT {
            for free_spinor in 0..32 {
                for gauge in &static_data.gauge_by_degree_and_free_spinor[degree * 32 + free_spinor]
                {
                    for target in &static_data.target {
                        let raw = usize::from(target.vector_weight) * 32
                            + usize::from(target.spinor_weight);
                        let begin = static_data.template_offsets[raw] as usize;
                        let end = static_data.template_offsets[raw + 1] as usize;
                        for template in &static_data.templates[begin..end] {
                            // The ordered second wedge always rejects equal
                            // derivative spinors, independently of the source mask.
                            if gauge.derivative_spinor == template.derivative_spinor {
                                continue;
                            }
                            let coefficient = gauge
                                .coefficient
                                .scale(target.coefficient, static_data.prime)
                                .multiply(template.coefficient, static_data.prime);
                            if coefficient.is_zero() {
                                continue;
                            }
                            let sector = u32::from(template.sector);
                            let output_coordinate = u32::from(template.output_coordinate);
                            entries.push(CudaPlanEntry {
                                gauge_spinor: u32::from(gauge.derivative_spinor),
                                template_spinor: u32::from(template.derivative_spinor),
                                sector,
                                output_coordinate,
                                coefficient,
                                functional_salt: flat_functional_salt(
                                    degree,
                                    output_coordinate,
                                    sector,
                                ),
                            });
                        }
                    }
                }
                offsets.push(
                    u32::try_from(entries.len())
                        .map_err(|_| "flat CUDA F_X plan exceeds u32".to_string())?,
                );
            }
        }
        Ok((offsets, entries, flat_pair_salts()))
    }

    fn flat_plan_sha256(
        offsets: &[u32],
        entries: &[CudaPlanEntry],
        pair_salts: &[u64; MOMENTUM_PAIR_COUNT],
    ) -> String {
        let mut hash = Sha256::new();
        hash.update(GPU_FX_SCHEMA.as_bytes());
        hash.update(b"\0flat-plan-v2\0");
        for offset in offsets {
            hash.update(offset.to_le_bytes());
        }
        for entry in entries {
            hash.update(entry.gauge_spinor.to_le_bytes());
            hash.update(entry.template_spinor.to_le_bytes());
            hash.update(entry.sector.to_le_bytes());
            hash.update(entry.output_coordinate.to_le_bytes());
            hash.update(entry.coefficient.real.to_le_bytes());
            hash.update(entry.coefficient.imaginary.to_le_bytes());
            hash.update(entry.functional_salt.to_le_bytes());
        }
        for salt in pair_salts {
            hash.update(salt.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    impl CudaModularFx {
        pub(crate) fn new(static_data: &ModularFxStaticData, device: i32) -> Result<Self, String> {
            if static_data.prime > i32::MAX as u32 {
                return Err("CUDA F_X requires a prime no larger than 2^31-1".to_string());
            }
            let mut gauge_offsets = Vec::with_capacity(6 * 32 + 1);
            let mut gauges = Vec::new();
            gauge_offsets.push(0);
            for entries in &static_data.gauge_by_degree_and_free_spinor {
                gauges.extend(entries.iter().map(|entry| CudaGaugeEntry {
                    derivative_spinor: u32::from(entry.derivative_spinor),
                    coefficient: entry.coefficient,
                }));
                gauge_offsets.push(gauges.len() as u32);
            }
            let targets = static_data
                .target
                .iter()
                .map(|entry| CudaTargetEntry {
                    vector_weight: u32::from(entry.vector_weight),
                    spinor_weight: u32::from(entry.spinor_weight),
                    coefficient: entry.coefficient,
                })
                .collect::<Vec<_>>();
            let templates = static_data
                .templates
                .iter()
                .map(|entry| CudaTemplateEntry {
                    derivative_spinor: u32::from(entry.derivative_spinor),
                    sector: u32::from(entry.sector),
                    output_coordinate: u32::from(entry.output_coordinate),
                    coefficient: entry.coefficient,
                })
                .collect::<Vec<_>>();
            let (plan_offsets, plan_entries, pair_salts) = build_flat_plan(static_data)?;
            let flat_plan_sha256 = flat_plan_sha256(&plan_offsets, &plan_entries, &pair_salts);
            let mut error = [0_i8; ERROR_CAPACITY];
            let context = unsafe {
                adynkra_fx_cuda_create_v2(
                    device,
                    static_data.prime,
                    gauge_offsets.as_ptr(),
                    gauges.as_ptr(),
                    u32::try_from(gauges.len())
                        .map_err(|_| "too many packed gauge entries".to_string())?,
                    targets.as_ptr(),
                    u32::try_from(targets.len())
                        .map_err(|_| "too many packed target entries".to_string())?,
                    static_data.template_offsets.as_ptr(),
                    templates.as_ptr(),
                    u32::try_from(templates.len())
                        .map_err(|_| "too many packed F_X templates".to_string())?,
                    plan_offsets.as_ptr(),
                    plan_entries.as_ptr(),
                    u32::try_from(plan_entries.len())
                        .map_err(|_| "too many flat CUDA F_X plan entries".to_string())?,
                    pair_salts.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let context = NonNull::new(context).ok_or_else(|| error_string(&error))?;
            let device_name = device_name(device)?;
            Ok(Self {
                context,
                prime: static_data.prime,
                static_semantic_sha256: static_data.semantic_sha256.clone(),
                flat_plan_sha256,
                device_name,
                multicol_host_values: Vec::new(),
                multicol_host_real: Vec::new(),
                multicol_host_imaginary: Vec::new(),
                _not_send_or_sync: PhantomData,
            })
        }

        pub(crate) fn device_name(&self) -> &str {
            &self.device_name
        }

        pub(crate) fn flat_plan_sha256(&self) -> &str {
            &self.flat_plan_sha256
        }

        pub(crate) fn resident_bytes(&self) -> u64 {
            unsafe { adynkra_fx_cuda_resident_bytes(self.context.as_ptr()) }
        }

        pub(crate) fn reserve_recoupling_terms(&mut self, terms: usize) -> Result<(), String> {
            let terms = u32::try_from(terms)
                .map_err(|_| "CUDA recoupling reservation exceeds u32".to_string())?;
            if terms == 0 {
                return Err("CUDA recoupling reservation is empty".to_string());
            }
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_reserve_recoupling(
                    self.context.as_ptr(),
                    terms,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        pub(crate) fn reserve_multicol(
            &mut self,
            unique_capacity: usize,
            active_columns: usize,
        ) -> Result<(), String> {
            if unique_capacity == 0 || !(1..=32).contains(&active_columns) {
                return Err("invalid CUDA multi-column reservation dimensions".to_string());
            }
            let value_capacity = unique_capacity
                .checked_mul(active_columns)
                .ok_or_else(|| "CUDA multi-column host value capacity overflow".to_string())?;
            let row_capacity = FUNCTIONAL_ROW_COUNT
                .checked_mul(active_columns)
                .ok_or_else(|| "CUDA multi-column host row capacity overflow".to_string())?;
            self.multicol_host_values
                .try_reserve_exact(
                    value_capacity.saturating_sub(self.multicol_host_values.capacity()),
                )
                .map_err(|error| format!("reserve multi-column host values: {error}"))?;
            self.multicol_host_real
                .try_reserve_exact(row_capacity.saturating_sub(self.multicol_host_real.capacity()))
                .map_err(|error| format!("reserve multi-column host real rows: {error}"))?;
            self.multicol_host_imaginary
                .try_reserve_exact(
                    row_capacity.saturating_sub(self.multicol_host_imaginary.capacity()),
                )
                .map_err(|error| format!("reserve multi-column host imaginary rows: {error}"))?;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_reserve_multicol(
                    self.context.as_ptr(),
                    u32::try_from(unique_capacity)
                        .map_err(|_| "CUDA multi-column capacity exceeds u32".to_string())?,
                    active_columns as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        pub(crate) fn accumulate_reduced_multicol(
            &mut self,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
        ) -> Result<CudaMultiColumnBatch, String> {
            let mut columns = (0..active_columns)
                .map(|_| Vec::with_capacity(FUNCTIONAL_ROW_COUNT))
                .collect::<Vec<_>>();
            let stats = self.accumulate_reduced_multicol_into(
                keys,
                key_major_values,
                active_columns,
                &mut columns,
            )?;
            Ok(CudaMultiColumnBatch { columns, stats })
        }

        pub(crate) fn accumulate_reduced_multicol_into(
            &mut self,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
            columns: &mut [Vec<GaussianResidue>],
        ) -> Result<CudaMultiColumnStats, String> {
            if keys.is_empty() || !(1..=32).contains(&active_columns) {
                return Err("invalid CUDA multi-column batch dimensions".to_string());
            }
            let expected_values = keys
                .len()
                .checked_mul(active_columns)
                .ok_or_else(|| "CUDA multi-column value shape overflow".to_string())?;
            if key_major_values.len() != expected_values {
                return Err("CUDA multi-column key-major value shape mismatch".to_string());
            }
            let output_len = FUNCTIONAL_ROW_COUNT
                .checked_mul(active_columns)
                .ok_or_else(|| "CUDA multi-column row shape overflow".to_string())?;
            if columns.len() != active_columns
                || columns
                    .iter()
                    .any(|column| column.capacity() < FUNCTIONAL_ROW_COUNT)
                || self.multicol_host_values.capacity() < expected_values
                || self.multicol_host_real.capacity() < output_len
                || self.multicol_host_imaginary.capacity() < output_len
            {
                return Err(
                    "CUDA multi-column host buffers were not reserved before accumulation"
                        .to_string(),
                );
            }
            self.multicol_host_values.clear();
            self.multicol_host_values
                .extend(key_major_values.iter().map(|&value| {
                    let bits = value as u128;
                    CudaWideValue {
                        low: bits as u64,
                        high: (value >> 64) as i64,
                        overflow: 0,
                        reserved: 0,
                    }
                }));
            self.multicol_host_real.resize(output_len, 0);
            self.multicol_host_imaginary.resize(output_len, 0);
            let mut stats = CudaMultiColumnStats::default();
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_accumulate_recoupled_multicol(
                    self.context.as_ptr(),
                    keys.as_ptr(),
                    self.multicol_host_values.as_ptr(),
                    u32::try_from(keys.len())
                        .map_err(|_| "CUDA multi-column key count exceeds u32".to_string())?,
                    active_columns as u32,
                    self.multicol_host_real.as_mut_ptr(),
                    self.multicol_host_imaginary.as_mut_ptr(),
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            if stats.unique_count != keys.len() as u64
                || stats.active_columns != active_columns as u32
            {
                return Err("CUDA multi-column result shape invariant failed".to_string());
            }
            for column in columns.iter_mut() {
                column.resize(FUNCTIONAL_ROW_COUNT, GaussianResidue::zero());
            }
            for row in 0..FUNCTIONAL_ROW_COUNT {
                for column in 0..active_columns {
                    let index = row * active_columns + column;
                    columns[column][row] = GaussianResidue {
                        real: self.multicol_host_real[index],
                        imaginary: self.multicol_host_imaginary[index],
                    };
                }
            }
            Ok(stats)
        }

        #[cfg(test)]
        fn set_legacy_contraction(&mut self, enabled: bool) -> Result<(), String> {
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_set_legacy_contraction(
                    self.context.as_ptr(),
                    i32::from(enabled),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        pub(crate) fn set_recoupling_hard_cap(
            &mut self,
            hard_cap_bytes: u64,
        ) -> Result<(), String> {
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_set_recoupling_hard_cap(
                    self.context.as_ptr(),
                    hard_cap_bytes,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        fn accumulate_terms(
            &mut self,
            terms: &[RecoupledSourceTerm],
            digest_ordinal: usize,
        ) -> Result<(Vec<GaussianResidue>, CudaColumnTiming), String> {
            if terms.is_empty() {
                return Err("CUDA F_X source batch is empty".to_string());
            }
            let mut keys = Vec::with_capacity(terms.len());
            let mut values = Vec::with_capacity(terms.len());
            let mut packed_hash = Sha256::new();
            packed_hash.update(GPU_FX_SCHEMA.as_bytes());
            packed_hash.update(b"\0packed-recoupling-input-v1\0");
            packed_hash.update((digest_ordinal as u64).to_le_bytes());
            for term in terms {
                let key = pack_recoupling_key(term)?;
                let bits = term.coefficient as u128;
                let value = CudaWideValue {
                    low: bits as u64,
                    high: (term.coefficient >> 64) as i64,
                    overflow: 0,
                    reserved: 0,
                };
                packed_hash.update(key.to_le_bytes());
                packed_hash.update(value.low.to_le_bytes());
                packed_hash.update(value.high.to_le_bytes());
                keys.push(key);
                values.push(value);
            }
            let packed_input_sha256 = format!("{:x}", packed_hash.finalize());
            let mut real = vec![0_u32; FUNCTIONAL_ROW_COUNT];
            let mut imaginary = vec![0_u32; FUNCTIONAL_ROW_COUNT];
            let mut stats = CudaRecouplingStats::default();
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_accumulate_recoupled(
                    self.context.as_ptr(),
                    keys.as_ptr(),
                    values.as_ptr(),
                    u32::try_from(terms.len())
                        .map_err(|_| "too many CUDA source terms".to_string())?,
                    real.as_mut_ptr(),
                    imaginary.as_mut_ptr(),
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            if stats.terms_before_reduce != terms.len() as u64 {
                return Err("CUDA recoupling term-count invariant failed".to_string());
            }
            let rows = real
                .into_iter()
                .zip(imaginary)
                .map(|(real, imaginary)| GaussianResidue { real, imaginary })
                .collect::<Vec<_>>();
            Ok((
                rows,
                CudaColumnTiming {
                    source_terms: terms.len(),
                    keys_after_reduce: stats.keys_after_reduce,
                    nonzero_terms_after_reduce: stats.nonzero_terms_after_reduce,
                    expanded_contributions: stats.expanded_contributions,
                    kernel_milliseconds: stats.total_milliseconds,
                    upload_milliseconds: stats.upload_milliseconds,
                    sort_milliseconds: stats.sort_milliseconds,
                    reduce_milliseconds: stats.reduce_milliseconds,
                    contract_milliseconds: stats.contract_milliseconds,
                    download_milliseconds: stats.download_milliseconds,
                    buffer_high_water_bytes: stats.buffer_high_water_bytes,
                    packed_input_sha256,
                    input_terms_per_second: if stats.total_milliseconds > 0.0 {
                        terms.len() as f64 * 1_000.0 / f64::from(stats.total_milliseconds)
                    } else {
                        0.0
                    },
                    batches: 1,
                    peak_batch_terms: terms.len(),
                    batch_term_cap: terms.len(),
                    host_hard_cap_bytes: u64::MAX,
                    device_hard_cap_bytes: u64::MAX,
                },
            ))
        }

        pub(crate) fn accumulate(
            &mut self,
            column: &GpuFxColumnInput,
        ) -> Result<(ModularFunctionalColumn, CudaColumnTiming), String> {
            if column.global_ordinal >= 77 || column.raising_residuals != [0; 5] {
                return Err("CUDA F_X input column is not certified".to_string());
            }
            if column.terms.is_empty() {
                return Err("CUDA F_X source column is empty".to_string());
            }
            let (rows, timing) = self.accumulate_terms(&column.terms, column.global_ordinal)?;
            let semantic_sha256 = column_semantic_sha256(
                self.prime,
                column.global_ordinal,
                &self.static_semantic_sha256,
                &rows,
            );
            Ok((
                ModularFunctionalColumn {
                    prime: self.prime,
                    global_ordinal: column.global_ordinal,
                    rows,
                    expanded_contributions: timing.expanded_contributions,
                    semantic_sha256,
                },
                timing,
            ))
        }
    }

    pub(crate) struct CudaStreamingColumnAccumulator {
        cuda: CudaModularFx,
        config: CudaStreamingConfig,
        batch: Vec<RecoupledSourceTerm>,
        rows: Vec<GaussianResidue>,
        source_term_hash: Sha256,
        packed_term_hash: Sha256,
        timing: CudaColumnTiming,
        last_batch_timing: Option<CudaColumnTiming>,
        last_batch_completion: Option<CudaBatchCompletion>,
    }

    impl CudaStreamingColumnAccumulator {
        pub(crate) fn new(
            mut cuda: CudaModularFx,
            config: CudaStreamingConfig,
        ) -> Result<Self, String> {
            let config = config.validate()?;
            cuda.set_recoupling_hard_cap(config.device_hard_cap_bytes)?;
            let mut source_term_hash = Sha256::new();
            source_term_hash.update(GPU_FX_SCHEMA.as_bytes());
            source_term_hash.update(b"\0streamed-source-terms-v1\0");
            let mut packed_term_hash = Sha256::new();
            packed_term_hash.update(GPU_FX_SCHEMA.as_bytes());
            packed_term_hash.update(b"\0streamed-packed-terms-v1\0");
            Ok(Self {
                cuda,
                config,
                batch: Vec::with_capacity(config.batch_terms),
                rows: vec![GaussianResidue::zero(); FUNCTIONAL_ROW_COUNT],
                source_term_hash,
                packed_term_hash,
                timing: CudaColumnTiming {
                    source_terms: 0,
                    keys_after_reduce: 0,
                    nonzero_terms_after_reduce: 0,
                    expanded_contributions: 0,
                    kernel_milliseconds: 0.0,
                    upload_milliseconds: 0.0,
                    sort_milliseconds: 0.0,
                    reduce_milliseconds: 0.0,
                    contract_milliseconds: 0.0,
                    download_milliseconds: 0.0,
                    buffer_high_water_bytes: 0,
                    packed_input_sha256: String::new(),
                    input_terms_per_second: 0.0,
                    batches: 0,
                    peak_batch_terms: 0,
                    batch_term_cap: config.batch_terms,
                    host_hard_cap_bytes: config.host_hard_cap_bytes,
                    device_hard_cap_bytes: config.device_hard_cap_bytes,
                },
                last_batch_timing: None,
                last_batch_completion: None,
            })
        }

        pub(crate) fn device_name(&self) -> &str {
            self.cuda.device_name()
        }

        pub(crate) fn flat_plan_sha256(&self) -> &str {
            self.cuda.flat_plan_sha256()
        }

        pub(crate) fn accumulate_for_parity(
            &mut self,
            column: &GpuFxColumnInput,
        ) -> Result<ModularFunctionalColumn, String> {
            self.cuda.accumulate(column).map(|result| result.0)
        }

        pub(crate) fn take_last_batch_timing(&mut self) -> Option<CudaColumnTiming> {
            self.last_batch_timing.take()
        }

        #[allow(dead_code)]
        pub(crate) fn take_last_batch_completion(&mut self) -> Option<CudaBatchCompletion> {
            self.last_batch_completion.take()
        }

        pub(crate) fn flush_pending(&mut self) -> Result<(), String> {
            self.flush()
        }

        pub(crate) fn push(&mut self, term: RecoupledSourceTerm) -> Result<(), String> {
            let key = pack_recoupling_key(&term)?;
            self.source_term_hash.update(term.momentum_pair);
            self.source_term_hash.update([term.free_spinor]);
            self.source_term_hash
                .update(term.exterior_mask.to_le_bytes());
            self.source_term_hash.update(term.coefficient.to_le_bytes());
            self.packed_term_hash.update(key.to_le_bytes());
            self.packed_term_hash
                .update((term.coefficient as u128 as u64).to_le_bytes());
            self.packed_term_hash
                .update(((term.coefficient >> 64) as i64).to_le_bytes());
            self.batch.push(term);
            if self.batch.len() == self.config.batch_terms {
                self.flush()?;
            }
            Ok(())
        }

        fn flush(&mut self) -> Result<(), String> {
            if self.batch.is_empty() {
                return Ok(());
            }
            let batch_terms = self.batch.len();
            let batch_ordinal = self.timing.batches as usize;
            let (batch_rows, batch_timing) =
                self.cuda.accumulate_terms(&self.batch, batch_ordinal)?;
            if batch_rows.len() != self.rows.len() {
                return Err("CUDA streaming functional row-count invariant failed".to_string());
            }
            // Every batch is independently canonical modulo p. Folding the
            // fixed row vector after each successful batch gives a bounded
            // exact accumulator and processes every raw term exactly once.
            for (accumulated, value) in self.rows.iter_mut().zip(&batch_rows) {
                *accumulated = accumulated.add(*value, self.cuda.prime);
            }
            self.timing.source_terms = self
                .timing
                .source_terms
                .checked_add(batch_timing.source_terms)
                .ok_or_else(|| "CUDA streaming source-term count overflow".to_string())?;
            self.timing.keys_after_reduce = self
                .timing
                .keys_after_reduce
                .checked_add(batch_timing.keys_after_reduce)
                .ok_or_else(|| "CUDA streaming reduced-key count overflow".to_string())?;
            self.timing.nonzero_terms_after_reduce = self
                .timing
                .nonzero_terms_after_reduce
                .checked_add(batch_timing.nonzero_terms_after_reduce)
                .ok_or_else(|| "CUDA streaming nonzero-key count overflow".to_string())?;
            self.timing.expanded_contributions = self
                .timing
                .expanded_contributions
                .checked_add(batch_timing.expanded_contributions)
                .ok_or_else(|| "CUDA streaming expanded count overflow".to_string())?;
            self.timing.kernel_milliseconds += batch_timing.kernel_milliseconds;
            self.timing.upload_milliseconds += batch_timing.upload_milliseconds;
            self.timing.sort_milliseconds += batch_timing.sort_milliseconds;
            self.timing.reduce_milliseconds += batch_timing.reduce_milliseconds;
            self.timing.contract_milliseconds += batch_timing.contract_milliseconds;
            self.timing.download_milliseconds += batch_timing.download_milliseconds;
            self.timing.buffer_high_water_bytes = self
                .timing
                .buffer_high_water_bytes
                .max(batch_timing.buffer_high_water_bytes);
            self.timing.batches = self
                .timing
                .batches
                .checked_add(1)
                .ok_or_else(|| "CUDA streaming batch count overflow".to_string())?;
            self.timing.peak_batch_terms = self.timing.peak_batch_terms.max(batch_terms);
            let mut delta_hash = Sha256::new();
            delta_hash.update(GPU_FX_SCHEMA.as_bytes());
            delta_hash.update(b"\0canonical-modular-batch-row-delta-v1\0");
            delta_hash.update(self.cuda.prime.to_le_bytes());
            delta_hash.update((batch_ordinal as u64).to_le_bytes());
            delta_hash.update((batch_terms as u64).to_le_bytes());
            for value in &batch_rows {
                delta_hash.update(value.real.to_le_bytes());
                delta_hash.update(value.imaginary.to_le_bytes());
            }
            let completion = CudaBatchCompletion {
                logical_batch_ordinal: batch_ordinal as u64,
                source_terms: batch_terms,
                row_delta: batch_rows,
                row_delta_sha256: format!("{:x}", delta_hash.finalize()),
                timing: batch_timing.clone(),
            };
            self.last_batch_timing = Some(batch_timing);
            self.last_batch_completion = Some(completion);
            self.batch.clear();
            Ok(())
        }

        pub(crate) fn finalize(
            mut self,
            metadata: &GpuFxColumnMetadata,
        ) -> Result<(ModularFunctionalColumn, CudaColumnTiming, String), String> {
            if metadata.global_ordinal >= 77 || metadata.raising_residuals != [0; 5] {
                return Err("CUDA F_X streamed column is not certified".to_string());
            }
            self.flush()?;
            if self.timing.source_terms == 0 {
                return Err("CUDA F_X streamed source column is empty".to_string());
            }
            self.timing.input_terms_per_second = if self.timing.kernel_milliseconds > 0.0 {
                self.timing.source_terms as f64 * 1_000.0
                    / f64::from(self.timing.kernel_milliseconds)
            } else {
                0.0
            };
            let raw_source_digest = self.source_term_hash.finalize();
            let mut source_hash = Sha256::new();
            source_hash.update(GPU_FX_SCHEMA.as_bytes());
            source_hash.update(b"\0bounded-streamed-source-v1\0");
            source_hash.update((metadata.global_ordinal as u64).to_le_bytes());
            source_hash.update(metadata.source_label.as_bytes());
            source_hash.update((metadata.source_copy as u64).to_le_bytes());
            source_hash.update((self.timing.source_terms as u64).to_le_bytes());
            source_hash.update(raw_source_digest);
            let source_terms_sha256 = format!("{:x}", source_hash.finalize());

            let raw_packed_digest = self.packed_term_hash.finalize();
            let mut packed_hash = Sha256::new();
            packed_hash.update(GPU_FX_SCHEMA.as_bytes());
            packed_hash.update(b"\0bounded-streamed-packed-v1\0");
            packed_hash.update((metadata.global_ordinal as u64).to_le_bytes());
            packed_hash.update((self.timing.source_terms as u64).to_le_bytes());
            packed_hash.update(raw_packed_digest);
            self.timing.packed_input_sha256 = format!("{:x}", packed_hash.finalize());

            let semantic_sha256 = column_semantic_sha256(
                self.cuda.prime,
                metadata.global_ordinal,
                &self.cuda.static_semantic_sha256,
                &self.rows,
            );
            Ok((
                ModularFunctionalColumn {
                    prime: self.cuda.prime,
                    global_ordinal: metadata.global_ordinal,
                    rows: self.rows,
                    expanded_contributions: self.timing.expanded_contributions,
                    semantic_sha256,
                },
                self.timing,
                source_terms_sha256,
            ))
        }
    }

    pub(crate) struct CudaModularP3 {
        cuda: CudaModularFx,
        flat_plan_sha256: String,
        multicol_coefficients: Vec<u32>,
    }

    #[derive(Clone, Copy, Debug, Serialize)]
    pub(crate) struct CudaP3Timing {
        pub source_count: usize,
        pub plan_entry_count: u32,
        /// Number of nonzero flattened static-plan visits after contraction
        /// and wedge admissibility, before row aggregation.
        pub expanded_contributions: u64,
        pub kernel_milliseconds: f32,
        pub resident_bytes: u64,
        pub buffer_high_water_bytes: u64,
        pub device_hard_cap_bytes: u64,
    }

    #[derive(Clone, Debug, Serialize)]
    pub(crate) struct CudaP3MultiLaneTiming {
        pub source_counts: Vec<usize>,
        pub plan_entry_count: u32,
        pub expanded_contributions: Vec<u64>,
        pub kernel_milliseconds: f32,
        pub resident_bytes: u64,
        pub buffer_high_water_bytes: u64,
        pub device_hard_cap_bytes: u64,
    }

    fn cuda_p3_plan_entries(plan: &P3ModularFlatPlan) -> Vec<CudaP3PlanEntry> {
        let mut entries = plan
            .entries
            .iter()
            .map(|entry| CudaP3PlanEntry {
                contracted_spinor: u32::from(entry.key.contracted_spinor),
                template_spinor: u32::from(entry.key.template_spinor),
                contraction_axis: u32::from(entry.key.contraction_axis),
                sector: u32::from(entry.key.sector),
                output_coordinate: u32::from(entry.key.output_coordinate),
                coefficient: entry.coefficient,
                functional_salt: 0,
            })
            .collect::<Vec<_>>();
        for schedule in 0..GAUGE_DEGREE_COUNT * 32 {
            let degree = schedule / 32;
            for entry in
                &mut entries[plan.offsets[schedule] as usize..plan.offsets[schedule + 1] as usize]
            {
                entry.functional_salt = (degree as u64).rotate_left(9)
                    ^ u64::from(entry.output_coordinate).rotate_left(31)
                    ^ 0x03d1_1000_0000_0002
                    ^ u64::from(entry.contraction_axis).rotate_left(53)
                    ^ if entry.sector == 0 {
                        0x1100_0000_0000_0002
                    } else {
                        0x1000_2000_0000_0005
                    };
            }
        }
        entries
    }

    impl CudaModularP3 {
        pub(crate) fn new_with_device_cap(
            static_data: &ModularFxStaticData,
            plan: &P3ModularFlatPlan,
            device: i32,
            device_hard_cap_bytes: u64,
        ) -> Result<Self, String> {
            if plan.prime != static_data.prime {
                return Err("CUDA p3 plan prime mismatch".to_string());
            }
            validate_p3_modular_flat_plan(plan)?;
            let mut cuda = CudaModularFx::new(static_data, device)?;
            cuda.set_recoupling_hard_cap(device_hard_cap_bytes)?;
            let entries = cuda_p3_plan_entries(plan);
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_configure_p3(
                    cuda.context.as_ptr(),
                    plan.offsets.as_ptr(),
                    entries.as_ptr(),
                    u32::try_from(entries.len())
                        .map_err(|_| "CUDA p3 plan exceeds u32".to_string())?,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(Self {
                cuda,
                flat_plan_sha256: plan.semantic_sha256.clone(),
                multicol_coefficients: Vec::new(),
            })
        }

        pub(crate) fn flat_plan_sha256(&self) -> &str {
            &self.flat_plan_sha256
        }

        pub(crate) fn resident_bytes(&self) -> u64 {
            self.cuda.resident_bytes()
        }

        pub(crate) fn buffer_high_water_bytes(&self) -> u64 {
            unsafe { adynkra_fx_cuda_buffer_high_water_bytes(self.cuda.context.as_ptr()) }
        }

        pub(crate) fn device_hard_cap_bytes(&self) -> u64 {
            unsafe { adynkra_fx_cuda_hard_cap_bytes(self.cuda.context.as_ptr()) }
        }

        pub(crate) fn plan_entry_count(&self) -> u32 {
            unsafe { adynkra_fx_cuda_p3_plan_entry_count(self.cuda.context.as_ptr()) }
        }

        pub(crate) fn set_device_hard_cap(&mut self, hard_cap_bytes: u64) -> Result<(), String> {
            self.cuda.set_recoupling_hard_cap(hard_cap_bytes)
        }

        #[cfg(test)]
        fn reconfigure_for_test(&mut self, plan: &P3ModularFlatPlan) -> Result<(), String> {
            let entries = cuda_p3_plan_entries(plan);
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_configure_p3(
                    self.cuda.context.as_ptr(),
                    plan.offsets.as_ptr(),
                    entries.as_ptr(),
                    u32::try_from(entries.len())
                        .map_err(|_| "CUDA p3 plan exceeds u32".to_string())?,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        pub(crate) fn accumulate(
            &mut self,
            column: &GpuFxColumnInput,
        ) -> Result<(ModularP3FunctionalColumn, CudaP3Timing), String> {
            if column.global_ordinal >= 77
                || column.raising_residuals != [0; 5]
                || column.terms.is_empty()
            {
                return Err("CUDA p3 input column is not certified".to_string());
            }
            let mut sources = Vec::with_capacity(column.terms.len());
            for term in &column.terms {
                if term.coefficient == 0 {
                    return Err("CUDA p3 source coefficient is zero".to_string());
                }
                pack_recoupling_key(term)?;
                sources.push(CudaSourceEntry {
                    exterior_mask: term.exterior_mask,
                    coefficient: i128_mod(term.coefficient, self.cuda.prime),
                    metadata: u32::from(term.momentum_pair[0])
                        | (u32::from(term.momentum_pair[1]) << 4)
                        | (u32::from(term.free_spinor) << 8),
                });
            }
            self.accumulate_sources(column.global_ordinal, sources)
        }

        pub(crate) fn accumulate_reduced_union_lane(
            &mut self,
            global_ordinal: usize,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
            lane: usize,
        ) -> Result<(ModularP3FunctionalColumn, CudaP3Timing), String> {
            if global_ordinal >= 77
                || active_columns == 0
                || lane >= active_columns
                || key_major_values.len() != keys.len().saturating_mul(active_columns)
                || keys.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err("CUDA p3 reduced-union shape or identity is invalid".to_string());
            }
            let sources =
                self.reduced_union_sources(keys, key_major_values, active_columns, lane)?;
            if sources.is_empty() {
                return Err("CUDA p3 reduced-union lane is empty".to_string());
            }
            self.accumulate_sources(global_ordinal, sources)
        }

        fn reduced_union_sources(
            &self,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
            lane: usize,
        ) -> Result<Vec<CudaSourceEntry>, String> {
            let mut sources = Vec::with_capacity(keys.len());
            for (key_index, &key) in keys.iter().enumerate() {
                let coefficient = key_major_values[key_index * active_columns + lane];
                if coefficient == 0 {
                    continue;
                }
                let (momentum_pair, free_spinor, exterior_mask) = unpack_recoupling_key(key)?;
                sources.push(CudaSourceEntry {
                    exterior_mask,
                    coefficient: i128_mod(coefficient, self.cuda.prime),
                    metadata: u32::from(momentum_pair[0])
                        | (u32::from(momentum_pair[1]) << 4)
                        | (u32::from(free_spinor) << 8),
                });
            }
            Ok(sources)
        }

        pub(crate) fn reset_persistent_columns(
            &mut self,
            columns: &[Vec<GaussianResidue>],
        ) -> Result<(), String> {
            if columns.is_empty()
                || columns.len() > 32
                || columns.iter().any(|rows| {
                    rows.len() != P3_FUNCTIONAL_ROW_COUNT
                        || rows.iter().any(|value| {
                            value.real >= self.cuda.prime || value.imaginary >= self.cuda.prime
                        })
                })
            {
                return Err("CUDA p3 persistent column shape is invalid".to_string());
            }
            let mut real = Vec::with_capacity(columns.len() * P3_FUNCTIONAL_ROW_COUNT);
            let mut imaginary = Vec::with_capacity(columns.len() * P3_FUNCTIONAL_ROW_COUNT);
            for column in columns {
                real.extend(column.iter().map(|value| value.real));
                imaginary.extend(column.iter().map(|value| value.imaginary));
            }
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_reset_p3_columns(
                    self.cuda.context.as_ptr(),
                    real.as_ptr(),
                    imaginary.as_ptr(),
                    columns.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        pub(crate) fn accumulate_reduced_union_lane_persistent(
            &mut self,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
            lane: usize,
        ) -> Result<CudaP3Timing, String> {
            if active_columns == 0
                || active_columns > 32
                || lane >= active_columns
                || key_major_values.len() != keys.len().saturating_mul(active_columns)
                || keys.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err("CUDA persistent p3 reduced-union identity is invalid".to_string());
            }
            let sources =
                self.reduced_union_sources(keys, key_major_values, active_columns, lane)?;
            if sources.is_empty() {
                return Err("CUDA persistent p3 reduced-union lane is empty".to_string());
            }
            let source_count = u32::try_from(sources.len())
                .map_err(|_| "CUDA p3 source count exceeds u32".to_string())?;
            let mut expanded_contributions = 0;
            let mut kernel_milliseconds = 0.0;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_accumulate_p3_column(
                    self.cuda.context.as_ptr(),
                    lane as u32,
                    sources.as_ptr(),
                    source_count,
                    &mut expanded_contributions,
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(CudaP3Timing {
                source_count: sources.len(),
                plan_entry_count: self.plan_entry_count(),
                expanded_contributions,
                kernel_milliseconds,
                resident_bytes: self.resident_bytes(),
                buffer_high_water_bytes: self.buffer_high_water_bytes(),
                device_hard_cap_bytes: self.device_hard_cap_bytes(),
            })
        }

        pub(crate) fn accumulate_reduced_union_multilane_persistent(
            &mut self,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
        ) -> Result<CudaP3MultiLaneTiming, String> {
            if keys.is_empty()
                || active_columns == 0
                || active_columns > 32
                || key_major_values.len() != keys.len().saturating_mul(active_columns)
                || keys.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err("CUDA persistent p3 multi-lane identity is invalid".to_string());
            }
            self.cuda.reserve_multicol(keys.len(), active_columns)?;
            let value_count = keys
                .len()
                .checked_mul(active_columns)
                .ok_or_else(|| "CUDA persistent p3 multi-lane value count overflow".to_string())?;
            self.multicol_coefficients.clear();
            self.multicol_coefficients
                .try_reserve_exact(
                    value_count.saturating_sub(self.multicol_coefficients.capacity()),
                )
                .map_err(|error| format!("reserve CUDA p3 multi-lane coefficients: {error}"))?;
            let mut source_counts = vec![0_usize; active_columns];
            for (key_index, &key) in keys.iter().enumerate() {
                unpack_recoupling_key(key)?;
                for lane in 0..active_columns {
                    let coefficient = key_major_values[key_index * active_columns + lane];
                    let residue = i128_mod(coefficient, self.cuda.prime);
                    self.multicol_coefficients.push(residue);
                    if coefficient != 0 {
                        source_counts[lane] += 1;
                    }
                }
            }
            if source_counts.iter().all(|&count| count == 0) {
                return Err("CUDA persistent p3 multi-lane input is zero".to_string());
            }
            let mut expanded_contributions = vec![0_u64; active_columns];
            let mut kernel_milliseconds = 0.0;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_accumulate_p3_multicol(
                    self.cuda.context.as_ptr(),
                    keys.as_ptr(),
                    self.multicol_coefficients.as_ptr(),
                    u32::try_from(keys.len())
                        .map_err(|_| "CUDA p3 multi-lane key count exceeds u32".to_string())?,
                    active_columns as u32,
                    expanded_contributions.as_mut_ptr(),
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(CudaP3MultiLaneTiming {
                source_counts,
                plan_entry_count: self.plan_entry_count(),
                expanded_contributions,
                kernel_milliseconds,
                resident_bytes: self.resident_bytes(),
                buffer_high_water_bytes: self.buffer_high_water_bytes(),
                device_hard_cap_bytes: self.device_hard_cap_bytes(),
            })
        }

        pub(crate) fn download_persistent_columns(
            &mut self,
            active_columns: usize,
        ) -> Result<Vec<Vec<GaussianResidue>>, String> {
            if active_columns == 0 || active_columns > 32 {
                return Err("CUDA p3 persistent download width is invalid".to_string());
            }
            let values = active_columns
                .checked_mul(P3_FUNCTIONAL_ROW_COUNT)
                .ok_or_else(|| "CUDA p3 persistent download size overflow".to_string())?;
            let mut real = vec![0; values];
            let mut imaginary = vec![0; values];
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_download_p3_columns(
                    self.cuda.context.as_ptr(),
                    active_columns as u32,
                    real.as_mut_ptr(),
                    imaginary.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(real
                .chunks_exact(P3_FUNCTIONAL_ROW_COUNT)
                .zip(imaginary.chunks_exact(P3_FUNCTIONAL_ROW_COUNT))
                .map(|(real, imaginary)| {
                    real.iter()
                        .copied()
                        .zip(imaginary.iter().copied())
                        .map(|(real, imaginary)| GaussianResidue { real, imaginary })
                        .collect()
                })
                .collect())
        }

        fn accumulate_sources(
            &mut self,
            global_ordinal: usize,
            sources: Vec<CudaSourceEntry>,
        ) -> Result<(ModularP3FunctionalColumn, CudaP3Timing), String> {
            let mut real = vec![0; P3_FUNCTIONAL_ROW_COUNT];
            let mut imaginary = vec![0; P3_FUNCTIONAL_ROW_COUNT];
            let mut expanded_contributions = 0;
            let mut kernel_milliseconds = 0.0;
            let mut error = [0_i8; ERROR_CAPACITY];
            let source_count = u32::try_from(sources.len())
                .map_err(|_| "CUDA p3 source count exceeds u32".to_string())?;
            let status = unsafe {
                adynkra_fx_cuda_accumulate_p3(
                    self.cuda.context.as_ptr(),
                    sources.as_ptr(),
                    source_count,
                    real.as_mut_ptr(),
                    imaginary.as_mut_ptr(),
                    &mut expanded_contributions,
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            let rows = real
                .into_iter()
                .zip(imaginary)
                .map(|(real, imaginary)| GaussianResidue { real, imaginary })
                .collect::<Vec<_>>();
            let semantic_sha256 = p3_column_semantic_sha256(self.cuda.prime, global_ordinal, &rows);
            let timing = CudaP3Timing {
                source_count: sources.len(),
                plan_entry_count: self.plan_entry_count(),
                expanded_contributions,
                kernel_milliseconds,
                resident_bytes: self.resident_bytes(),
                buffer_high_water_bytes: self.buffer_high_water_bytes(),
                device_hard_cap_bytes: self.device_hard_cap_bytes(),
            };
            Ok((
                ModularP3FunctionalColumn {
                    prime: self.cuda.prime,
                    global_ordinal,
                    rows,
                    expanded_contributions,
                    semantic_sha256,
                },
                timing,
            ))
        }
    }

    pub(crate) struct CudaModularP3ThreePrime {
        cuda: CudaModularFx,
        flat_plan_sha256: [String; 3],
        coefficients: Vec<u32>,
    }

    #[derive(Clone, Debug, Serialize)]
    pub(crate) struct CudaP3ThreePrimeTiming {
        pub source_counts: Vec<Vec<usize>>,
        pub plan_entry_count: u32,
        pub expanded_contributions: Vec<Vec<u64>>,
        pub kernel_milliseconds: f32,
        pub resident_bytes: u64,
        pub buffer_high_water_bytes: u64,
        pub device_hard_cap_bytes: u64,
    }

    fn cuda_p3_three_prime_plan_entries(
        plans: &[P3ModularFlatPlan; 3],
    ) -> Result<Vec<CudaP3ThreePrimePlanEntry>, String> {
        const EXACT_SCALE: u64 = 13_440;
        for (slot, plan) in plans.iter().enumerate() {
            validate_p3_modular_flat_plan(plan)?;
            if plan.prime != GPU_FX_PRIMES[slot]
                || plan.offsets != plans[0].offsets
                || plan.entries.len() != plans[0].entries.len()
                || plan
                    .entries
                    .iter()
                    .zip(&plans[0].entries)
                    .any(|(entry, canonical)| entry.key != canonical.key)
            {
                return Err(
                    "CUDA p3 three-prime plans do not share canonical structure".to_string()
                );
            }
        }
        let mut entries = Vec::with_capacity(plans[0].entries.len());
        for index in 0..plans[0].entries.len() {
            let entry = plans[0].entries[index];
            let scaled = [0_usize, 1_usize]
                .map(|component| {
                    let values = (0..3)
                        .map(|slot| {
                            let prime = u64::from(GPU_FX_PRIMES[slot]);
                            let coefficient = plans[slot].entries[index].coefficient;
                            let residue = if component == 0 {
                                coefficient.real
                            } else {
                                coefficient.imaginary
                            };
                            let raw = u64::from(residue) * EXACT_SCALE % prime;
                            if raw <= prime / 2 {
                                raw as i64
                            } else {
                                raw as i64 - prime as i64
                            }
                        })
                        .collect::<Vec<_>>();
                    if values.windows(2).any(|pair| pair[0] != pair[1]) {
                        return Err(
                            "CUDA p3 three-prime coefficient has no common exact lift".to_string()
                        );
                    }
                    i16::try_from(values[0]).map_err(|_| {
                        "CUDA p3 three-prime exact coefficient exceeds i16".to_string()
                    })
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            if scaled == [0, 0] {
                return Err("CUDA p3 three-prime exact coefficient is zero".to_string());
            }
            entries.push(CudaP3ThreePrimePlanEntry {
                contracted_spinor: u32::from(entry.key.contracted_spinor),
                template_spinor: u32::from(entry.key.template_spinor),
                contraction_axis: u32::from(entry.key.contraction_axis),
                sector: u32::from(entry.key.sector),
                output_coordinate: u32::from(entry.key.output_coordinate),
                scaled_real: scaled[0],
                scaled_imaginary: scaled[1],
                functional_salt: 0,
            });
        }
        for schedule in 0..GAUGE_DEGREE_COUNT * 32 {
            let degree = schedule / 32;
            for entry in &mut entries
                [plans[0].offsets[schedule] as usize..plans[0].offsets[schedule + 1] as usize]
            {
                entry.functional_salt = (degree as u64).rotate_left(9)
                    ^ u64::from(entry.output_coordinate).rotate_left(31)
                    ^ 0x03d1_1000_0000_0002
                    ^ u64::from(entry.contraction_axis).rotate_left(53)
                    ^ if entry.sector == 0 {
                        0x1100_0000_0000_0002
                    } else {
                        0x1000_2000_0000_0005
                    };
            }
        }
        Ok(entries)
    }

    impl CudaModularP3ThreePrime {
        pub(crate) fn new_with_device_cap(
            static_data: &[ModularFxStaticData; 3],
            plans: &[P3ModularFlatPlan; 3],
            device: i32,
            device_hard_cap_bytes: u64,
        ) -> Result<Self, String> {
            for slot in 0..3 {
                if static_data[slot].prime != GPU_FX_PRIMES[slot]
                    || plans[slot].prime != GPU_FX_PRIMES[slot]
                {
                    return Err("CUDA p3 three-prime pinned-prime order is invalid".to_string());
                }
            }
            let entries = cuda_p3_three_prime_plan_entries(plans)?;
            let inverse_scales = GPU_FX_PRIMES.map(|prime| inverse_mod(13_440_u32 % prime, prime));
            let mut cuda = CudaModularFx::new(&static_data[0], device)?;
            cuda.set_recoupling_hard_cap(device_hard_cap_bytes)?;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_configure_p3_three_prime(
                    cuda.context.as_ptr(),
                    GPU_FX_PRIMES.as_ptr(),
                    inverse_scales.as_ptr(),
                    plans[0].offsets.as_ptr(),
                    entries.as_ptr(),
                    u32::try_from(entries.len())
                        .map_err(|_| "CUDA p3 three-prime plan exceeds u32".to_string())?,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(Self {
                cuda,
                flat_plan_sha256: std::array::from_fn(|slot| plans[slot].semantic_sha256.clone()),
                coefficients: Vec::new(),
            })
        }

        pub(crate) fn flat_plan_sha256(&self) -> &[String; 3] {
            &self.flat_plan_sha256
        }

        pub(crate) fn resident_bytes(&self) -> u64 {
            self.cuda.resident_bytes()
        }

        pub(crate) fn buffer_high_water_bytes(&self) -> u64 {
            unsafe { adynkra_fx_cuda_buffer_high_water_bytes(self.cuda.context.as_ptr()) }
        }

        pub(crate) fn device_hard_cap_bytes(&self) -> u64 {
            unsafe { adynkra_fx_cuda_hard_cap_bytes(self.cuda.context.as_ptr()) }
        }

        pub(crate) fn plan_entry_count(&self) -> u32 {
            unsafe { adynkra_fx_cuda_p3_plan_entry_count(self.cuda.context.as_ptr()) }
        }

        pub(crate) fn reset_persistent_columns(
            &mut self,
            columns: &[Vec<Vec<GaussianResidue>>],
        ) -> Result<(), String> {
            let active_columns = columns.first().map_or(0, Vec::len);
            if columns.len() != 3
                || active_columns == 0
                || active_columns > 32
                || columns.iter().any(|prime_columns| {
                    prime_columns.len() != active_columns
                        || prime_columns
                            .iter()
                            .any(|rows| rows.len() != P3_FUNCTIONAL_ROW_COUNT)
                })
            {
                return Err("CUDA p3 three-prime persistent column shape is invalid".to_string());
            }
            let value_count = 3_usize
                .checked_mul(active_columns)
                .and_then(|value| value.checked_mul(P3_FUNCTIONAL_ROW_COUNT))
                .ok_or_else(|| "CUDA p3 three-prime reset size overflow".to_string())?;
            let mut real = Vec::with_capacity(value_count);
            let mut imaginary = Vec::with_capacity(value_count);
            for prime_slot in 0..3 {
                let prime = GPU_FX_PRIMES[prime_slot];
                for rows in &columns[prime_slot] {
                    if rows
                        .iter()
                        .any(|value| value.real >= prime || value.imaginary >= prime)
                    {
                        return Err("CUDA p3 three-prime persistent residue is invalid".to_string());
                    }
                    real.extend(rows.iter().map(|value| value.real));
                    imaginary.extend(rows.iter().map(|value| value.imaginary));
                }
            }
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_reset_p3_three_prime_columns(
                    self.cuda.context.as_ptr(),
                    real.as_ptr(),
                    imaginary.as_ptr(),
                    active_columns as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }

        pub(crate) fn accumulate_reduced_union_multilane_persistent(
            &mut self,
            keys: &[u64],
            key_major_values: &[i128],
            active_columns: usize,
        ) -> Result<CudaP3ThreePrimeTiming, String> {
            if keys.is_empty()
                || active_columns == 0
                || active_columns > 32
                || key_major_values.len() != keys.len().saturating_mul(active_columns)
                || keys.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err("CUDA persistent p3 three-prime identity is invalid".to_string());
            }
            self.cuda.reserve_multicol(keys.len(), active_columns)?;
            let coefficient_count = keys
                .len()
                .checked_mul(3)
                .and_then(|value| value.checked_mul(active_columns))
                .ok_or_else(|| "CUDA p3 three-prime coefficient count overflow".to_string())?;
            self.coefficients.clear();
            self.coefficients
                .try_reserve_exact(coefficient_count.saturating_sub(self.coefficients.capacity()))
                .map_err(|error| format!("reserve CUDA p3 three-prime coefficients: {error}"))?;
            let mut source_counts = vec![vec![0_usize; active_columns]; 3];
            for (key_index, &key) in keys.iter().enumerate() {
                unpack_recoupling_key(key)?;
                for prime_slot in 0..3 {
                    let prime = GPU_FX_PRIMES[prime_slot];
                    for lane in 0..active_columns {
                        let coefficient = key_major_values[key_index * active_columns + lane];
                        let residue = i128_mod(coefficient, prime);
                        self.coefficients.push(residue);
                        if residue != 0 {
                            source_counts[prime_slot][lane] += 1;
                        }
                    }
                }
            }
            if source_counts.iter().flatten().all(|&count| count == 0) {
                return Err("CUDA persistent p3 three-prime input is zero".to_string());
            }
            let mut expanded_flat = vec![0_u64; 3 * active_columns];
            let mut kernel_milliseconds = 0.0;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_accumulate_p3_three_prime_multicol(
                    self.cuda.context.as_ptr(),
                    keys.as_ptr(),
                    self.coefficients.as_ptr(),
                    u32::try_from(keys.len())
                        .map_err(|_| "CUDA p3 three-prime key count exceeds u32".to_string())?,
                    active_columns as u32,
                    expanded_flat.as_mut_ptr(),
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(CudaP3ThreePrimeTiming {
                source_counts,
                plan_entry_count: self.plan_entry_count(),
                expanded_contributions: expanded_flat
                    .chunks_exact(active_columns)
                    .map(<[u64]>::to_vec)
                    .collect(),
                kernel_milliseconds,
                resident_bytes: self.resident_bytes(),
                buffer_high_water_bytes: self.buffer_high_water_bytes(),
                device_hard_cap_bytes: self.device_hard_cap_bytes(),
            })
        }

        pub(crate) fn download_persistent_columns(
            &mut self,
            active_columns: usize,
        ) -> Result<Vec<Vec<Vec<GaussianResidue>>>, String> {
            if active_columns == 0 || active_columns > 32 {
                return Err("CUDA p3 three-prime download width is invalid".to_string());
            }
            let value_count = 3_usize
                .checked_mul(active_columns)
                .and_then(|value| value.checked_mul(P3_FUNCTIONAL_ROW_COUNT))
                .ok_or_else(|| "CUDA p3 three-prime download size overflow".to_string())?;
            let mut real = vec![0; value_count];
            let mut imaginary = vec![0; value_count];
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_download_p3_three_prime_columns(
                    self.cuda.context.as_ptr(),
                    active_columns as u32,
                    real.as_mut_ptr(),
                    imaginary.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            let rows_per_prime = active_columns * P3_FUNCTIONAL_ROW_COUNT;
            Ok((0..3)
                .map(|prime_slot| {
                    let begin = prime_slot * rows_per_prime;
                    let end = begin + rows_per_prime;
                    real[begin..end]
                        .chunks_exact(P3_FUNCTIONAL_ROW_COUNT)
                        .zip(imaginary[begin..end].chunks_exact(P3_FUNCTIONAL_ROW_COUNT))
                        .map(|(real, imaginary)| {
                            real.iter()
                                .copied()
                                .zip(imaginary.iter().copied())
                                .map(|(real, imaginary)| GaussianResidue { real, imaginary })
                                .collect()
                        })
                        .collect()
                })
                .collect())
        }
    }

    impl Drop for CudaModularFx {
        fn drop(&mut self) {
            unsafe { adynkra_fx_cuda_destroy(self.context.as_ptr()) };
        }
    }

    fn device_name(device: i32) -> Result<String, String> {
        let mut name = [0_i8; 256];
        let mut error = [0_i8; ERROR_CAPACITY];
        let status = unsafe {
            adynkra_fx_cuda_device_name(
                device,
                name.as_mut_ptr(),
                name.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(error_string(&error));
        }
        unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_str()
            .map(str::to_owned)
            .map_err(|error| format!("CUDA device name is not UTF-8: {error}"))
    }

    fn error_string(buffer: &[c_char]) -> String {
        let pointer = buffer.as_ptr();
        let message = unsafe { CStr::from_ptr(pointer) }.to_string_lossy();
        if message.is_empty() {
            "CUDA F_X backend failed without an error message".to_string()
        } else {
            message.into_owned()
        }
    }

    struct PersistentSparseOwner {
        raw: NonNull<c_void>,
        operation_lock: Mutex<()>,
    }

    // The C context and its accounting are protected by operation_lock. Every
    // handle retains this owner, so the context cannot be destroyed first.
    unsafe impl Send for PersistentSparseOwner {}
    unsafe impl Sync for PersistentSparseOwner {}

    impl PersistentSparseOwner {
        fn lock(&self) -> Result<std::sync::MutexGuard<'_, ()>, String> {
            self.operation_lock
                .lock()
                .map_err(|_| "persistent sparse operation lock is poisoned".to_string())
        }
    }

    impl Drop for PersistentSparseOwner {
        fn drop(&mut self) {
            let _guard = self
                .operation_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { adynkra_fx_cuda_sparse_context_destroy(self.raw.as_ptr()) };
        }
    }

    #[derive(Clone)]
    struct PersistentSparseContext {
        owner: Arc<PersistentSparseOwner>,
    }

    impl PersistentSparseContext {
        fn new(device: i32, hard_cap_bytes: u64) -> Result<Self, String> {
            let mut error = [0_i8; ERROR_CAPACITY];
            let raw = unsafe {
                adynkra_fx_cuda_sparse_context_create(
                    device,
                    hard_cap_bytes,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            NonNull::new(raw)
                .map(|raw| Self {
                    owner: Arc::new(PersistentSparseOwner {
                        raw,
                        operation_lock: Mutex::new(()),
                    }),
                })
                .ok_or_else(|| error_string(&error))
        }

        fn upload(&self, entries: &[CudaSparseEntry]) -> Result<PersistentSparseHandle, String> {
            let _guard = self.owner.lock()?;
            let mut error = [0_i8; ERROR_CAPACITY];
            let raw = unsafe {
                adynkra_fx_cuda_sparse_handle_upload(
                    self.owner.raw.as_ptr(),
                    entries.as_ptr(),
                    u32::try_from(entries.len())
                        .map_err(|_| "persistent sparse input exceeds u32".to_string())?,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            NonNull::new(raw)
                .map(|raw| PersistentSparseHandle {
                    raw,
                    owner: Arc::clone(&self.owner),
                })
                .ok_or_else(|| error_string(&error))
        }

        fn resident_bytes(&self) -> u64 {
            let _guard = self
                .owner
                .operation_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { adynkra_fx_cuda_sparse_resident_bytes(self.owner.raw.as_ptr()) }
        }

        fn lower(
            &self,
            handle: &PersistentSparseHandle,
            root: usize,
        ) -> Result<(PersistentSparseHandle, CudaSparseLoweringStats), String> {
            let _guard = self.owner.lock()?;
            let mut error = [0_i8; ERROR_CAPACITY];
            let mut stats = CudaSparseLoweringStats::default();
            let raw = unsafe {
                adynkra_fx_cuda_sparse_handle_lower(
                    self.owner.raw.as_ptr(),
                    handle.raw.as_ptr(),
                    u32::try_from(root).map_err(|_| "persistent root exceeds u32".to_string())?,
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            NonNull::new(raw)
                .map(|raw| {
                    (
                        PersistentSparseHandle {
                            raw,
                            owner: Arc::clone(&self.owner),
                        },
                        stats,
                    )
                })
                .ok_or_else(|| error_string(&error))
        }

        fn download(&self, handle: &PersistentSparseHandle) -> Result<Vec<(u64, i64)>, String> {
            let _guard = self.owner.lock()?;
            let count = unsafe { adynkra_fx_cuda_sparse_handle_count(handle.raw.as_ptr()) };
            let mut entries = vec![CudaSparseEntry { key: 0, value: 0 }; count as usize];
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_sparse_handle_download(
                    self.owner.raw.as_ptr(),
                    handle.raw.as_ptr(),
                    entries.as_mut_ptr(),
                    count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            Ok(entries
                .into_iter()
                .map(|entry| (entry.key, entry.value))
                .collect())
        }

        fn visit_download<F>(
            &self,
            handle: &PersistentSparseHandle,
            chunk_terms: usize,
            mut visit: F,
        ) -> Result<u64, String>
        where
            F: FnMut(u64, i64) -> Result<(), String>,
        {
            if chunk_terms == 0 || chunk_terms > u32::MAX as usize {
                return Err("persistent CUDA download chunk is invalid".to_string());
            }
            let count = unsafe { adynkra_fx_cuda_sparse_handle_count(handle.raw.as_ptr()) };
            let mut buffer = vec![CudaSparseEntry { key: 0, value: 0 }; chunk_terms];
            let mut start = 0_u32;
            while start < count {
                let _guard = self.owner.lock()?;
                let take = (count - start).min(chunk_terms as u32);
                let mut error = [0_i8; ERROR_CAPACITY];
                let status = unsafe {
                    adynkra_fx_cuda_sparse_handle_download_range(
                        self.owner.raw.as_ptr(),
                        handle.raw.as_ptr(),
                        start,
                        buffer.as_mut_ptr(),
                        take,
                        error.as_mut_ptr(),
                        error.len(),
                    )
                };
                if status != 0 {
                    return Err(error_string(&error));
                }
                for entry in &buffer[..take as usize] {
                    visit(entry.key, entry.value)?;
                }
                start += take;
            }
            Ok(u64::from(count))
        }

        fn download_range_into(
            &self,
            handle: &PersistentSparseHandle,
            start: u32,
            entries: &mut [CudaSparseEntry],
        ) -> Result<(), String> {
            let _guard = self.owner.lock()?;
            let take = u32::try_from(entries.len())
                .map_err(|_| "persistent sparse download range exceeds u32".to_string())?;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_fx_cuda_sparse_handle_download_range(
                    self.owner.raw.as_ptr(),
                    handle.raw.as_ptr(),
                    start,
                    entries.as_mut_ptr(),
                    take,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(error_string(&error))
            }
        }
    }

    struct PersistentSparseHandle {
        raw: NonNull<c_void>,
        owner: Arc<PersistentSparseOwner>,
    }

    // A handle is immutable after construction. Its owning context is kept
    // alive by PersistentGroupLaneState and all access is serialized there.
    unsafe impl Send for PersistentSparseHandle {}

    impl PersistentSparseHandle {
        fn term_count(&self) -> u32 {
            unsafe { adynkra_fx_cuda_sparse_handle_count(self.raw.as_ptr()) }
        }

        fn maximum_absolute_coefficient(&self) -> u64 {
            unsafe { adynkra_fx_cuda_sparse_handle_max_abs(self.raw.as_ptr()) }
        }
    }

    impl Drop for PersistentSparseHandle {
        fn drop(&mut self) {
            let _guard = self
                .owner
                .operation_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { adynkra_fx_cuda_sparse_handle_destroy(self.raw.as_ptr()) };
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct PersistentHighestIdentity {
        term_count: usize,
        maximum_absolute_coefficient: u64,
        semantic_sha256: String,
    }

    fn persistent_highest_identity(
        highest: &crate::eleven_dimensional_level16_couplings::CanonicalSparseHighest64,
    ) -> std::io::Result<PersistentHighestIdentity> {
        let mut hash = Sha256::new();
        hash.update(b"adynkra-persistent-canonical-highest64-v1");
        hash.update((highest.term_count() as u64).to_le_bytes());
        hash.update(highest.maximum_absolute_coefficient().to_le_bytes());
        highest.visit_terms(|key, coefficient| {
            hash.update(key.to_le_bytes());
            hash.update(coefficient.to_le_bytes());
            Ok(())
        })?;
        Ok(PersistentHighestIdentity {
            term_count: highest.term_count(),
            maximum_absolute_coefficient: highest.maximum_absolute_coefficient(),
            semantic_sha256: format!("{:x}", hash.finalize()),
        })
    }

    #[derive(Clone, Debug)]
    enum PersistentLanePreflight {
        Two(crate::eleven_dimensional_second_momentum_20001_fx::SecondMomentum20001GpuColumnPreflight),
        Three(crate::eleven_dimensional_second_momentum_30001_fx::SecondMomentum30001GpuColumnPreflight),
        Full {
            value: crate::eleven_dimensional_second_momentum_full_fx::FullFxColumnPreflight,
            map_directory: std::path::PathBuf,
        },
    }

    impl PersistentLanePreflight {
        fn local_ordinal(&self) -> usize {
            match self {
                Self::Two(value) => value.local_column_ordinal,
                Self::Three(value) => value.local_column_ordinal,
                Self::Full { value, .. } => value.global_column_ordinal,
            }
        }

        fn global_ordinal(&self) -> usize {
            match self {
                Self::Two(value) => value.global_column_ordinal,
                Self::Three(value) => value.global_column_ordinal,
                Self::Full { value, .. } => value.global_column_ordinal,
            }
        }

        fn source_copy(&self) -> usize {
            match self {
                Self::Two(value) => value.source_copy,
                Self::Three(value) => value.source_copy,
                Self::Full { value, .. } => value.source_copy,
            }
        }

        fn word_count(&self) -> usize {
            match self {
                Self::Two(value) => value.pbw_word_count,
                Self::Three(value) => value.pbw_word_count,
                Self::Full { value, .. } => value.pbw_word_count,
            }
        }
    }

    enum PersistentWordHandle {
        Highest,
        Owned(PersistentSparseHandle),
    }

    pub(crate) struct PersistentGroupLaneState {
        // Drop the cached immutable handle before its owner. Drop::drop also
        // takes it explicitly so this invariant does not depend on field order.
        highest: Option<PersistentSparseHandle>,
        highest_identity: Option<PersistentHighestIdentity>,
        // All lanes share one internally serialized context so the large
        // lowering scratch allocation is reused instead of split three ways.
        context: PersistentSparseContext,
        preflight: PersistentLanePreflight,
        summary: PersistentLoweringSummary,
        host_staging_cap_bytes: u64,
        download_chunk_terms: usize,
    }

    impl Drop for PersistentGroupLaneState {
        fn drop(&mut self) {
            drop(self.highest.take());
        }
    }

    impl PersistentGroupLaneState {
        fn resolve<'a>(
            &'a self,
            handle: &'a PersistentWordHandle,
        ) -> std::io::Result<&'a PersistentSparseHandle> {
            match handle {
                PersistentWordHandle::Highest => self.highest.as_ref().ok_or_else(|| {
                    std::io::Error::other("persistent lane highest handle is not initialized")
                }),
                PersistentWordHandle::Owned(handle) => Ok(handle),
            }
        }

        fn ensure_highest(
            &mut self,
            highest: &crate::eleven_dimensional_level16_couplings::CanonicalSparseHighest64,
        ) -> std::io::Result<PersistentWordHandle> {
            let identity = persistent_highest_identity(highest)?;
            if let Some(expected) = &self.highest_identity {
                if expected != &identity {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "persistent lane canonical highest identity changed between words",
                    ));
                }
            } else {
                let handle =
                    upload_canonical_highest(&self.context, highest, self.host_staging_cap_bytes)?;
                if handle.term_count() as usize != identity.term_count {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "persistent lane highest term count changed during upload",
                    ));
                }
                self.summary.maximum_absolute_coefficient = identity.maximum_absolute_coefficient;
                self.summary.peak_immutable_handle_bytes =
                    self.summary.peak_immutable_handle_bytes.max(
                        u64::from(handle.term_count())
                            * std::mem::size_of::<CudaSparseEntry>() as u64,
                    );
                self.highest = Some(handle);
                self.highest_identity = Some(identity);
            }
            Ok(PersistentWordHandle::Highest)
        }

        fn lower_word(
            &mut self,
            source: &PersistentWordHandle,
            roots: &[u8],
            maximum: &mut i128,
        ) -> std::io::Result<PersistentWordHandle> {
            if roots.is_empty() || roots.iter().any(|root| !(1..=5).contains(root)) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "persistent lane PBW word contains an invalid simple root",
                ));
            }
            let mut owned = None;
            for &simple_root in roots {
                let base = match owned.as_ref() {
                    Some(handle) => handle,
                    None => self.resolve(source)?,
                };
                let (next, stats) = self
                    .context
                    .lower(base, usize::from(simple_root - 1))
                    .map_err(std::io::Error::other)?;
                let next_maximum = next.maximum_absolute_coefficient();
                *maximum = (*maximum).max(i128::from(next_maximum));
                self.summary.maximum_absolute_coefficient =
                    self.summary.maximum_absolute_coefficient.max(next_maximum);
                self.summary.roots_lowered =
                    self.summary.roots_lowered.checked_add(1).ok_or_else(|| {
                        std::io::Error::other("persistent lane root count overflow")
                    })?;
                self.summary.input_entry_visits = self
                    .summary
                    .input_entry_visits
                    .checked_add(stats.input_count)
                    .ok_or_else(|| std::io::Error::other("persistent lane input count overflow"))?;
                self.summary.expanded_entry_visits = self
                    .summary
                    .expanded_entry_visits
                    .checked_add(stats.expanded_count)
                    .ok_or_else(|| {
                        std::io::Error::other("persistent lane expansion count overflow")
                    })?;
                self.summary.output_entry_visits = self
                    .summary
                    .output_entry_visits
                    .checked_add(stats.output_count)
                    .ok_or_else(|| {
                        std::io::Error::other("persistent lane output count overflow")
                    })?;
                self.summary.gpu_milliseconds += f64::from(stats.total_milliseconds);
                self.summary.scratch_high_water_bytes = self
                    .summary
                    .scratch_high_water_bytes
                    .max(stats.scratch_high_water_bytes);
                self.summary.peak_immutable_handle_bytes = self
                    .summary
                    .peak_immutable_handle_bytes
                    .max(stats.immutable_handle_bytes);
                owned = Some(next);
            }
            owned
                .map(PersistentWordHandle::Owned)
                .ok_or_else(|| std::io::Error::other("persistent lane lowering produced no handle"))
        }

        fn download_terms(
            &self,
            handle: &PersistentWordHandle,
            visit: &mut dyn FnMut(u64, i64) -> std::io::Result<()>,
        ) -> std::io::Result<u64> {
            let handle = self.resolve(handle)?;
            let count = handle.term_count();
            let mut buffer = vec![
                CudaSparseEntry { key: 0, value: 0 };
                self.download_chunk_terms.min(count as usize)
            ];
            let mut start = 0_u32;
            while start < count {
                let take = (count - start).min(self.download_chunk_terms as u32) as usize;
                self.context
                    .download_range_into(handle, start, &mut buffer[..take])
                    .map_err(std::io::Error::other)?;
                // Do exact host recoupling outside the shared CUDA lock. Other
                // lanes can lower or download their next chunk concurrently.
                for entry in &buffer[..take] {
                    visit(entry.key, entry.value)?;
                }
                start += take as u32;
            }
            Ok(u64::from(count))
        }
    }

    pub(crate) struct PersistentGroupLaneAdapter {
        lanes: Vec<Mutex<PersistentGroupLaneState>>,
    }

    impl PersistentGroupLaneAdapter {
        pub(crate) fn new(
            plan: &crate::second_momentum_gpu_group::PreparedColumnGroup,
            device: i32,
            shared_device_hard_cap_bytes: u64,
            host_staging_cap_bytes: u64,
            download_chunk_terms: usize,
            full_map_directory: Option<&std::path::Path>,
        ) -> Result<Self, String> {
            if plan.active_columns == 0
                || plan.active_columns != plan.members.len()
                || download_chunk_terms == 0
                || download_chunk_terms > u32::MAX as usize
                || host_staging_cap_bytes == 0
            {
                return Err("invalid persistent group lane adapter configuration".to_string());
            }
            let context = PersistentSparseContext::new(device, shared_device_hard_cap_bytes)?;
            let mut lanes = Vec::with_capacity(plan.active_columns);
            for (lane_index, member) in plan.members.iter().enumerate() {
                if member.local_ordinal != plan.ordered_local_ordinals[lane_index]
                    || member.global_ordinal != plan.ordered_global_ordinals[lane_index]
                    || member.source_copy != plan.ordered_source_copies[lane_index]
                {
                    return Err("persistent group lane identity order changed".to_string());
                }
                let preflight = match plan.tranche.as_str() {
                    "20001" => PersistentLanePreflight::Two(
                        crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(
                            member.local_ordinal,
                        )
                        .map_err(|error| error.to_string())?,
                    ),
                    "30001" => PersistentLanePreflight::Three(
                        crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
                            member.local_ordinal,
                        )
                        .map_err(|error| error.to_string())?,
                    ),
                    "00001" | "01001" | "10001" | "11001" => {
                        let map_directory = full_map_directory.ok_or_else(|| {
                            "full-inventory persistent group requires its exact map directory"
                                .to_string()
                        })?;
                        PersistentLanePreflight::Full {
                            value: crate::eleven_dimensional_second_momentum_full_fx::
                                gpu_column_preflight(member.global_ordinal, map_directory)
                                .map_err(|error| error.to_string())?,
                            map_directory: map_directory.to_path_buf(),
                        }
                    }
                    _ => {
                        return Err(
                            "persistent group tranche must be 00001, 01001, 10001, 11001, 20001, or 30001"
                                .to_string(),
                        );
                    }
                };
                if preflight.local_ordinal() != member.local_ordinal
                    || preflight.global_ordinal() != member.global_ordinal
                    || preflight.source_copy() != member.source_copy
                    || preflight.word_count() != plan.pbw_word_count
                {
                    return Err("persistent group preflight identity changed".to_string());
                }
                lanes.push(Mutex::new(PersistentGroupLaneState {
                    highest: None,
                    highest_identity: None,
                    context: context.clone(),
                    preflight,
                    summary: PersistentLoweringSummary {
                        enabled: true,
                        device_hard_cap_bytes: shared_device_hard_cap_bytes,
                        download_chunk_terms,
                        ..PersistentLoweringSummary::default()
                    },
                    host_staging_cap_bytes,
                    download_chunk_terms,
                }));
            }
            Ok(Self { lanes })
        }

        pub(crate) fn run_lane_word(
            &self,
            lane_index: usize,
            expected_local_ordinal: usize,
            expected_global_ordinal: usize,
            expected_source_copy: usize,
            word_ordinal: usize,
            emit_term: &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
        ) -> Result<crate::second_momentum_gpu_group::LaneWordCompletion, String> {
            let lane = self
                .lanes
                .get(lane_index)
                .ok_or_else(|| "persistent group lane index is out of range".to_string())?;
            let mut state = lane
                .lock()
                .map_err(|_| "persistent group lane lock is poisoned".to_string())?;
            if state.preflight.local_ordinal() != expected_local_ordinal
                || state.preflight.global_ordinal() != expected_global_ordinal
                || state.preflight.source_copy() != expected_source_copy
                || word_ordinal >= state.preflight.word_count()
            {
                return Err("persistent group run-lane identity changed".to_string());
            }

            let state = RefCell::new(&mut *state);
            let mut emitted_terms = 0_u64;
            let mut observed_end = None;
            let preflight = state.borrow().preflight.clone();
            let metadata = match preflight {
                PersistentLanePreflight::Two(preflight) => {
                    crate::eleven_dimensional_second_momentum_20001_fx::
                        visit_gpu_column_word_contribution_events_from_handles(
                            &preflight,
                            word_ordinal,
                            |highest| state.borrow_mut().ensure_highest(highest),
                            |source, roots, maximum| {
                                state.borrow_mut().lower_word(source, roots, maximum)
                            },
                            |handle, visit| state.borrow().download_terms(handle, visit),
                            |event| {
                                use crate::eleven_dimensional_second_momentum_20001_fx::
                                    SecondMomentum20001GpuColumnEvent;
                                match event {
                                    SecondMomentum20001GpuColumnEvent::Term {
                                        requested_word_ordinal,
                                        term,
                                    } => {
                                        if requested_word_ordinal != word_ordinal {
                                            return Err(std::io::Error::other(
                                                "persistent 20001 lane term word changed",
                                            ));
                                        }
                                        emit_term(term).map_err(std::io::Error::other)?;
                                        emitted_terms = emitted_terms.checked_add(1).ok_or_else(|| {
                                            std::io::Error::other(
                                                "persistent lane raw-term count overflow",
                                            )
                                        })?;
                                    }
                                    SecondMomentum20001GpuColumnEvent::WordEnd {
                                        requested_word_ordinal,
                                        raw_terms_emitted,
                                    } => {
                                        if requested_word_ordinal != word_ordinal
                                            || raw_terms_emitted != emitted_terms
                                            || observed_end.replace(raw_terms_emitted).is_some()
                                        {
                                            return Err(std::io::Error::other(
                                                "persistent 20001 lane word-end accounting changed",
                                            ));
                                        }
                                    }
                                    SecondMomentum20001GpuColumnEvent::WordLoweringStart {
                                        requested_word_ordinal,
                                        ..
                                    }
                                    | SecondMomentum20001GpuColumnEvent::WordStart {
                                        requested_word_ordinal,
                                        ..
                                    } if requested_word_ordinal != word_ordinal => {
                                        return Err(std::io::Error::other(
                                            "persistent 20001 lane boundary word changed",
                                        ));
                                    }
                                    SecondMomentum20001GpuColumnEvent::WordLoweringStart { .. }
                                    | SecondMomentum20001GpuColumnEvent::WordStart { .. } => {}
                                }
                                Ok(())
                            },
                        )
                }
                PersistentLanePreflight::Three(preflight) => {
                    crate::eleven_dimensional_second_momentum_30001_fx::
                        visit_gpu_column_word_contribution_events_from_handles(
                            &preflight,
                            word_ordinal,
                            |highest| state.borrow_mut().ensure_highest(highest),
                            |source, roots, maximum| {
                                state.borrow_mut().lower_word(source, roots, maximum)
                            },
                            |handle, visit| state.borrow().download_terms(handle, visit),
                            |event| {
                                use crate::eleven_dimensional_second_momentum_30001_fx::
                                    SecondMomentum30001GpuColumnEvent;
                                match event {
                                    SecondMomentum30001GpuColumnEvent::Term {
                                        requested_word_ordinal,
                                        term,
                                    } => {
                                        if requested_word_ordinal != word_ordinal {
                                            return Err(std::io::Error::other(
                                                "persistent 30001 lane term word changed",
                                            ));
                                        }
                                        emit_term(term).map_err(std::io::Error::other)?;
                                        emitted_terms = emitted_terms.checked_add(1).ok_or_else(|| {
                                            std::io::Error::other(
                                                "persistent lane raw-term count overflow",
                                            )
                                        })?;
                                    }
                                    SecondMomentum30001GpuColumnEvent::WordEnd {
                                        requested_word_ordinal,
                                        raw_terms_emitted,
                                    } => {
                                        if requested_word_ordinal != word_ordinal
                                            || raw_terms_emitted != emitted_terms
                                            || observed_end.replace(raw_terms_emitted).is_some()
                                        {
                                            return Err(std::io::Error::other(
                                                "persistent 30001 lane word-end accounting changed",
                                            ));
                                        }
                                    }
                                    SecondMomentum30001GpuColumnEvent::WordLoweringStart {
                                        requested_word_ordinal,
                                        ..
                                    }
                                    | SecondMomentum30001GpuColumnEvent::WordStart {
                                        requested_word_ordinal,
                                        ..
                                    } if requested_word_ordinal != word_ordinal => {
                                        return Err(std::io::Error::other(
                                            "persistent 30001 lane boundary word changed",
                                        ));
                                    }
                                    SecondMomentum30001GpuColumnEvent::WordLoweringStart { .. }
                                    | SecondMomentum30001GpuColumnEvent::WordStart { .. } => {}
                                }
                                Ok(())
                            },
                        )
                }
                PersistentLanePreflight::Full {
                    value: preflight,
                    map_directory,
                } => crate::eleven_dimensional_second_momentum_full_fx::
                    visit_gpu_column_contribution_events_range_from_handles(
                        &preflight,
                        &map_directory,
                        word_ordinal,
                        word_ordinal + 1,
                        |highest| state.borrow_mut().ensure_highest(highest),
                        |source, roots, maximum| {
                            state.borrow_mut().lower_word(source, roots, maximum)
                        },
                        |handle, visit| state.borrow().download_terms(handle, visit),
                        |event| {
                            use crate::eleven_dimensional_second_momentum_full_fx::
                                FullFxColumnEvent;
                            match event {
                                FullFxColumnEvent::Term {
                                    requested_word_ordinal,
                                    term,
                                } => {
                                    if requested_word_ordinal != word_ordinal {
                                        return Err(std::io::Error::other(
                                            "persistent full-inventory lane term word changed",
                                        ));
                                    }
                                    emit_term(term).map_err(std::io::Error::other)?;
                                    emitted_terms = emitted_terms.checked_add(1).ok_or_else(|| {
                                        std::io::Error::other(
                                            "persistent full-inventory raw-term count overflow",
                                        )
                                    })?;
                                }
                                FullFxColumnEvent::WordEnd {
                                    requested_word_ordinal,
                                    raw_terms_emitted,
                                } => {
                                    if requested_word_ordinal != word_ordinal
                                        || raw_terms_emitted != emitted_terms
                                        || observed_end.replace(raw_terms_emitted).is_some()
                                    {
                                        return Err(std::io::Error::other(
                                            "persistent full-inventory word-end accounting changed",
                                        ));
                                    }
                                }
                                FullFxColumnEvent::WordLoweringStart {
                                    requested_word_ordinal,
                                    ..
                                }
                                | FullFxColumnEvent::WordStart {
                                    requested_word_ordinal,
                                    ..
                                } if requested_word_ordinal != word_ordinal => {
                                    return Err(std::io::Error::other(
                                        "persistent full-inventory boundary word changed",
                                    ));
                                }
                                FullFxColumnEvent::WordLoweringStart { .. }
                                | FullFxColumnEvent::WordStart { .. } => {}
                            }
                            Ok(())
                        },
                    ),
            }
            .map_err(|error| error.to_string())?;
            if metadata.global_ordinal != expected_global_ordinal
                || metadata.source_copy != expected_source_copy
                || metadata.raising_residuals != [0; 5]
                || observed_end != Some(emitted_terms)
            {
                return Err("persistent group lane completion identity changed".to_string());
            }
            Ok(crate::second_momentum_gpu_group::LaneWordCompletion {
                lane_index,
                local_ordinal: expected_local_ordinal,
                global_ordinal: expected_global_ordinal,
                source_copy: expected_source_copy,
                word_ordinal,
                raw_terms: emitted_terms,
            })
        }

        pub(crate) fn summaries(&self) -> Result<Vec<PersistentLoweringSummary>, String> {
            self.lanes
                .iter()
                .map(|lane| {
                    let state = lane
                        .lock()
                        .map_err(|_| "persistent group lane lock is poisoned".to_string())?;
                    let mut summary = state.summary;
                    summary.scratch_high_water_bytes = summary
                        .scratch_high_water_bytes
                        .max(state.context.resident_bytes());
                    Ok(summary)
                })
                .collect()
        }

        pub(crate) fn collect_parity_prefix(
            &self,
            maximum_terms_per_lane: usize,
        ) -> Result<Vec<Vec<RecoupledSourceTerm>>, String> {
            if maximum_terms_per_lane == 0 {
                return Err("persistent parity prefix must be nonzero".to_string());
            }
            let mut output = (0..self.lanes.len())
                .map(|_| Vec::with_capacity(maximum_terms_per_lane))
                .collect::<Vec<_>>();
            let word_count = self
                .lanes
                .first()
                .ok_or_else(|| "persistent group has no lanes".to_string())?
                .lock()
                .map_err(|_| "persistent group lane lock is poisoned".to_string())?
                .preflight
                .word_count();
            for word_ordinal in 0..word_count {
                for (lane_index, terms) in output.iter_mut().enumerate() {
                    if terms.len() == maximum_terms_per_lane {
                        continue;
                    }
                    let (local, global, copy) = {
                        let state = self.lanes[lane_index]
                            .lock()
                            .map_err(|_| "persistent group lane lock is poisoned".to_string())?;
                        (
                            state.preflight.local_ordinal(),
                            state.preflight.global_ordinal(),
                            state.preflight.source_copy(),
                        )
                    };
                    self.run_lane_word(
                        lane_index,
                        local,
                        global,
                        copy,
                        word_ordinal,
                        &mut |term| {
                            if terms.len() < maximum_terms_per_lane {
                                terms.push(term);
                            }
                            Ok(())
                        },
                    )?;
                }
                if output
                    .iter()
                    .all(|terms| terms.len() == maximum_terms_per_lane)
                {
                    break;
                }
            }
            if output.iter().any(Vec::is_empty) {
                return Err("persistent parity replay produced an empty lane".to_string());
            }
            Ok(output)
        }
    }

    fn validate_sparse_entries(entries: &[(u64, i64)]) -> Result<(), String> {
        if entries.is_empty() {
            return Err("exact CUDA sparse input is empty".to_string());
        }
        if entries.len() > (u32::MAX / 13) as usize {
            return Err("exact CUDA sparse input exceeds the u32 expansion bound".to_string());
        }
        for (index, &(key, coefficient)) in entries.iter().enumerate() {
            let free_spinor = (key >> 32) as u32;
            let mask = key as u32;
            if free_spinor >= 32
                || mask.count_ones() != 12
                || coefficient == 0
                || coefficient == i64::MIN
                || index != 0 && entries[index - 1].0 >= key
            {
                return Err(
                    "CUDA sparse input must be sorted unique nonzero degree-12 data".to_string(),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn persistent_sparse_enabled() -> Result<bool, String> {
        match std::env::var("ADYNKRA_GPU_FX_PERSISTENT_LOWERING") {
            Ok(value) => match value.as_str() {
                "1" | "true" | "yes" => Ok(true),
                "0" | "false" | "no" => Ok(false),
                _ => Err(format!(
                    "invalid ADYNKRA_GPU_FX_PERSISTENT_LOWERING={value}"
                )),
            },
            Err(std::env::VarError::NotPresent) => Ok(false),
            Err(error) => Err(format!(
                "cannot read ADYNKRA_GPU_FX_PERSISTENT_LOWERING: {error}"
            )),
        }
    }

    fn upload_canonical_highest(
        context: &PersistentSparseContext,
        highest: &crate::eleven_dimensional_level16_couplings::CanonicalSparseHighest64,
        host_staging_cap_bytes: u64,
    ) -> std::io::Result<PersistentSparseHandle> {
        let required_bytes = highest
            .term_count()
            .checked_mul(std::mem::size_of::<CudaSparseEntry>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| std::io::Error::other("persistent highest staging size overflow"))?;
        if required_bytes > host_staging_cap_bytes {
            return Err(std::io::Error::other(format!(
                "persistent highest staging requires {required_bytes} host bytes, above bounded allowance {host_staging_cap_bytes}"
            )));
        }
        let mut entries = Vec::with_capacity(highest.term_count());
        highest.visit_terms(|key, value| {
            entries.push(CudaSparseEntry { key, value });
            Ok(())
        })?;
        let handle = context.upload(&entries).map_err(std::io::Error::other)?;
        if handle.maximum_absolute_coefficient() != highest.maximum_absolute_coefficient() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent CUDA highest-state maximum coefficient changed during upload",
            ));
        }
        Ok(handle)
    }

    pub(crate) fn visit_persistent_column_contributions<F, P>(
        tranche: &str,
        local_ordinal: usize,
        device: i32,
        device_hard_cap_bytes: u64,
        host_staging_cap_bytes: u64,
        download_chunk_terms: usize,
        mut visit_term: F,
        mut root_progress: P,
    ) -> Result<(GpuFxColumnInput, PersistentLoweringSummary), String>
    where
        F: FnMut(RecoupledSourceTerm) -> std::io::Result<()>,
        P: FnMut(PersistentRootProgress),
    {
        if download_chunk_terms == 0 {
            return Err("persistent CUDA download chunk is empty".to_string());
        }
        let context = PersistentSparseContext::new(device, device_hard_cap_bytes)?;
        let current_word = Cell::new(None::<usize>);
        let maximum_observed = Cell::new(0_u64);
        let mut summary = PersistentLoweringSummary {
            enabled: true,
            device_hard_cap_bytes,
            download_chunk_terms,
            ..PersistentLoweringSummary::default()
        };

        let mut lower_word = |source: &PersistentSparseHandle,
                              roots: &[u8],
                              maximum: &mut i128|
         -> std::io::Result<PersistentSparseHandle> {
            if roots.is_empty() || roots.iter().any(|root| !(1..=5).contains(root)) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "persistent CUDA PBW segment contains an invalid simple root",
                ));
            }
            let word_ordinal = current_word.get().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "persistent CUDA lowering started outside a word boundary",
                )
            })?;
            let mut owned = None;
            for &simple_root in roots {
                let base = owned.as_ref().unwrap_or(source);
                root_progress(PersistentRootProgress {
                    phase: "started",
                    word_ordinal,
                    root: usize::from(simple_root),
                    stats: CudaSparseLoweringStats::default(),
                    resident_bytes: context.resident_bytes(),
                });
                let (next, stats) = context
                    .lower(base, usize::from(simple_root - 1))
                    .map_err(std::io::Error::other)?;
                let next_maximum = next.maximum_absolute_coefficient();
                maximum_observed.set(maximum_observed.get().max(next_maximum));
                *maximum = (*maximum).max(i128::from(next_maximum));
                summary.roots_lowered = summary
                    .roots_lowered
                    .checked_add(1)
                    .ok_or_else(|| std::io::Error::other("persistent root count overflow"))?;
                summary.input_entry_visits = summary
                    .input_entry_visits
                    .checked_add(stats.input_count)
                    .ok_or_else(|| std::io::Error::other("persistent input count overflow"))?;
                summary.expanded_entry_visits = summary
                    .expanded_entry_visits
                    .checked_add(stats.expanded_count)
                    .ok_or_else(|| std::io::Error::other("persistent expansion count overflow"))?;
                summary.output_entry_visits = summary
                    .output_entry_visits
                    .checked_add(stats.output_count)
                    .ok_or_else(|| std::io::Error::other("persistent output count overflow"))?;
                summary.gpu_milliseconds += f64::from(stats.total_milliseconds);
                summary.scratch_high_water_bytes = summary
                    .scratch_high_water_bytes
                    .max(stats.scratch_high_water_bytes);
                summary.peak_immutable_handle_bytes = summary
                    .peak_immutable_handle_bytes
                    .max(stats.immutable_handle_bytes);
                root_progress(PersistentRootProgress {
                    phase: "completed",
                    word_ordinal,
                    root: usize::from(simple_root),
                    stats,
                    resident_bytes: context.resident_bytes(),
                });
                owned = Some(next);
            }
            owned
                .ok_or_else(|| std::io::Error::other("persistent CUDA lowering produced no handle"))
        };

        let mut download_terms = |handle: &PersistentSparseHandle,
                                  visit: &mut dyn FnMut(u64, i64) -> std::io::Result<()>|
         -> std::io::Result<u64> {
            context
                .visit_download(handle, download_chunk_terms, |key, value| {
                    visit(key, value).map_err(|error| error.to_string())
                })
                .map_err(std::io::Error::other)
        };

        let metadata = match tranche {
            "20001" => {
                let preflight =
                    crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(
                        local_ordinal,
                    )
                    .map_err(|error| error.to_string())?;
                crate::eleven_dimensional_second_momentum_20001_fx::
                    visit_gpu_column_contribution_events_from_handles(
                        &preflight,
                        0,
                        |highest| {
                            let handle = upload_canonical_highest(
                                &context,
                                highest,
                                host_staging_cap_bytes,
                            )?;
                            maximum_observed.set(
                                maximum_observed
                                    .get()
                                    .max(handle.maximum_absolute_coefficient()),
                            );
                            Ok(handle)
                        },
                        &mut lower_word,
                        &mut download_terms,
                        |event| {
                            use crate::eleven_dimensional_second_momentum_20001_fx::
                                SecondMomentum20001GpuColumnEvent;
                            match event {
                                SecondMomentum20001GpuColumnEvent::WordLoweringStart {
                                    requested_word_ordinal,
                                    ..
                                } => current_word.set(Some(requested_word_ordinal)),
                                SecondMomentum20001GpuColumnEvent::Term { term, .. } => {
                                    visit_term(term)?;
                                }
                                SecondMomentum20001GpuColumnEvent::WordStart { .. }
                                | SecondMomentum20001GpuColumnEvent::WordEnd { .. } => {}
                            }
                            Ok(())
                        },
                    )
            }
            "30001" => {
                let preflight =
                    crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
                        local_ordinal,
                    )
                    .map_err(|error| error.to_string())?;
                crate::eleven_dimensional_second_momentum_30001_fx::
                    visit_gpu_column_contribution_events_from_handles(
                        &preflight,
                        0,
                        |highest| {
                            let handle = upload_canonical_highest(
                                &context,
                                highest,
                                host_staging_cap_bytes,
                            )?;
                            maximum_observed.set(
                                maximum_observed
                                    .get()
                                    .max(handle.maximum_absolute_coefficient()),
                            );
                            Ok(handle)
                        },
                        &mut lower_word,
                        &mut download_terms,
                        |event| {
                            use crate::eleven_dimensional_second_momentum_30001_fx::
                                SecondMomentum30001GpuColumnEvent;
                            match event {
                                SecondMomentum30001GpuColumnEvent::WordLoweringStart {
                                    requested_word_ordinal,
                                    ..
                                } => current_word.set(Some(requested_word_ordinal)),
                                SecondMomentum30001GpuColumnEvent::Term { term, .. } => {
                                    visit_term(term)?;
                                }
                                SecondMomentum30001GpuColumnEvent::WordStart { .. }
                                | SecondMomentum30001GpuColumnEvent::WordEnd { .. } => {}
                            }
                            Ok(())
                        },
                    )
            }
            _ => {
                return Err("persistent CUDA tranche must be 20001 or 30001".to_string());
            }
        }
        .map_err(|error| error.to_string())?;
        summary.maximum_absolute_coefficient = maximum_observed.get();
        Ok((metadata, summary))
    }

    pub(crate) fn lower_sparse_word_exact(
        entries: &[(u64, i64)],
        roots: &[usize],
        device: i32,
    ) -> Result<(Vec<(u64, i64)>, Vec<CudaSparseLoweringStats>), String> {
        let hard_cap_bytes = environment_u64(
            "ADYNKRA_GPU_FX_DEVICE_CAP_BYTES",
            DEFAULT_STREAM_DEVICE_HARD_CAP_BYTES,
        )?;
        lower_sparse_word_exact_with_cap(entries, roots, device, hard_cap_bytes)
    }

    fn lower_sparse_word_exact_with_cap(
        entries: &[(u64, i64)],
        roots: &[usize],
        device: i32,
        hard_cap_bytes: u64,
    ) -> Result<(Vec<(u64, i64)>, Vec<CudaSparseLoweringStats>), String> {
        validate_sparse_entries(entries)?;
        if roots.is_empty() || roots.iter().any(|root| *root >= 5) {
            return Err("persistent CUDA sparse root word is invalid".to_string());
        }
        let context = PersistentSparseContext::new(device, hard_cap_bytes)?;
        let packed = entries
            .iter()
            .map(|&(key, value)| CudaSparseEntry { key, value })
            .collect::<Vec<_>>();
        let mut handle = context.upload(&packed)?;
        let mut telemetry = Vec::with_capacity(roots.len());
        for &root in roots {
            let (next, stats) = context.lower(&handle, root)?;
            telemetry.push(stats);
            handle = next;
        }
        let output = context.download(&handle)?;
        Ok((output, telemetry))
    }

    pub(crate) fn lower_sparse_exact(
        entries: &[(u64, i64)],
        root: usize,
        device: i32,
    ) -> Result<(Vec<(u64, i64)>, f32), String> {
        if entries.is_empty() || root >= 5 {
            return Err("invalid exact CUDA sparse-lowering input".to_string());
        }
        let host_hard_cap_bytes = environment_u64(
            "ADYNKRA_GPU_FX_HOST_CAP_BYTES",
            DEFAULT_STREAM_HOST_HARD_CAP_BYTES,
        )?;
        let required_host_bytes = sparse_lowering_host_bytes(entries.len())?;
        if required_host_bytes > host_hard_cap_bytes {
            return Err(format!(
                "CUDA sparse lowering requires {required_host_bytes} host bytes, above hard cap {host_hard_cap_bytes}"
            ));
        }
        validate_sparse_entries(entries)?;
        for &(_, coefficient) in entries {
            if coefficient.unsigned_abs() > i64::MAX as u64 / 13 {
                return Err("CUDA sparse-lowering coefficient exceeds exact bound".to_string());
            }
        }
        let capacity = entries
            .len()
            .checked_mul(13)
            .ok_or_else(|| "CUDA sparse-lowering capacity overflow".to_string())?;
        let source_keys = entries.iter().map(|entry| entry.0).collect::<Vec<_>>();
        let source_values = entries.iter().map(|entry| entry.1).collect::<Vec<_>>();
        let mut output_keys = vec![0_u64; capacity];
        let mut output_values = vec![0_i64; capacity];
        let mut output_count = 0_u32;
        let mut kernel_milliseconds = 0_f32;
        let mut error = [0_i8; ERROR_CAPACITY];
        let status = unsafe {
            adynkra_fx_cuda_lower_sparse(
                device,
                source_keys.as_ptr(),
                source_values.as_ptr(),
                u32::try_from(entries.len())
                    .map_err(|_| "too many CUDA sparse source entries".to_string())?,
                root as u32,
                output_keys.as_mut_ptr(),
                output_values.as_mut_ptr(),
                u32::try_from(capacity)
                    .map_err(|_| "CUDA sparse output exceeds u32".to_string())?,
                &mut output_count,
                &mut kernel_milliseconds,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(error_string(&error));
        }
        let count = output_count as usize;
        if count > capacity {
            return Err("CUDA sparse-lowering returned an invalid count".to_string());
        }
        let output = output_keys
            .into_iter()
            .zip(output_values)
            .take(count)
            .filter(|(key, value)| *key != u64::MAX && *value != 0)
            .collect();
        Ok((output, kernel_milliseconds))
    }

    fn sparse_lowering_host_bytes(entry_count: usize) -> Result<u64, String> {
        // Caller entries, packed source key/value copies, and the two 13N
        // output arrays are simultaneously live at peak.
        let bytes_per_entry = std::mem::size_of::<(u64, i64)>()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<i64>()
            + 13 * (std::mem::size_of::<u64>() + std::mem::size_of::<i64>());
        entry_count
            .checked_mul(bytes_per_entry)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| "CUDA sparse-lowering host size overflow".to_string())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn p3_cuda_matches_exact_and_flat_for_three_primes_with_transactional_caps() {
            assert_eq!(std::mem::size_of::<GaussianResidue>(), 8);
            assert_eq!(std::mem::align_of::<GaussianResidue>(), 4);
            assert_eq!(std::mem::size_of::<CudaSourceEntry>(), 12);
            assert_eq!(std::mem::align_of::<CudaSourceEntry>(), 4);
            assert_eq!(std::mem::offset_of!(CudaSourceEntry, metadata), 8);
            assert_eq!(std::mem::size_of::<CudaP3PlanEntry>(), 40);
            assert_eq!(std::mem::align_of::<CudaP3PlanEntry>(), 8);
            assert_eq!(std::mem::offset_of!(CudaP3PlanEntry, coefficient), 20);
            assert_eq!(std::mem::offset_of!(CudaP3PlanEntry, functional_salt), 32);
            let exact_static = P3D11ExactStaticData::build().unwrap();
            let cancelling_terms = vec![
                RecoupledSourceTerm {
                    momentum_pair: [2, 7],
                    free_spinor: 15,
                    exterior_mask: 0x0000_0fff,
                    coefficient: 7,
                },
                RecoupledSourceTerm {
                    momentum_pair: [2, 7],
                    free_spinor: 15,
                    exterior_mask: 0x0000_0fff,
                    coefficient: -7,
                },
            ];
            let column = GpuFxColumnInput {
                global_ordinal: 0,
                source_label: "p3-cuda-three-prime-canary".to_string(),
                source_copy: 1,
                terms: cancelling_terms
                    .iter()
                    .cloned()
                    .chain([
                        RecoupledSourceTerm {
                            momentum_pair: [1, 8],
                            free_spinor: 14,
                            exterior_mask: 0x0000_1ffe,
                            coefficient: -3,
                        },
                        RecoupledSourceTerm {
                            momentum_pair: [0, 10],
                            free_spinor: 13,
                            exterior_mask: 0x0000_0fff,
                            coefficient: 5,
                        },
                    ])
                    .collect(),
                raising_residuals: [0; 5],
            };
            for prime in GPU_FX_PRIMES {
                let static_data = ModularFxStaticData::build(prime).unwrap();
                let plan = build_p3_modular_flat_plan(&static_data).unwrap();

                let cancelling_column = GpuFxColumnInput {
                    global_ordinal: column.global_ordinal,
                    source_label: "p3-cancellation".to_string(),
                    source_copy: 1,
                    terms: cancelling_terms.clone(),
                    raising_residuals: [0; 5],
                };
                let cancellation =
                    accumulate_p3_column_cpu_flat(&plan, &cancelling_column).unwrap();
                assert!(cancellation.rows.iter().all(|value| value.is_zero()));
                assert!(cancellation.expanded_contributions > 0);

                let base_resident = {
                    let base = CudaModularFx::new(&static_data, 0).unwrap();
                    base.resident_bytes()
                };
                assert!(
                    CudaModularP3::new_with_device_cap(&static_data, &plan, 0, base_resident,)
                        .is_err()
                );
                let mut cuda = CudaModularP3::new_with_device_cap(
                    &static_data,
                    &plan,
                    0,
                    10 * 1024 * 1024 * 1024,
                )
                .unwrap();
                assert_eq!(cuda.plan_entry_count() as usize, plan.entry_count());
                let configured_resident = cuda.resident_bytes();
                let configured_high_water = cuda.buffer_high_water_bytes();
                const P3_PAIR_OFFSET_BYTES: u64 =
                    ((GAUGE_DEGREE_COUNT * 32 * 32 * 32 + 1) * std::mem::size_of::<u32>()) as u64;
                assert_eq!(P3_PAIR_OFFSET_BYTES, 786_436);
                let expected_p3_resident = (GAUGE_DEGREE_COUNT * 32 + 1)
                    * std::mem::size_of::<u32>()
                    + plan.entry_count() * std::mem::size_of::<CudaP3PlanEntry>()
                    + 2 * P3_FUNCTIONAL_ROW_COUNT * 32 * std::mem::size_of::<u32>()
                    + P3_PAIR_OFFSET_BYTES as usize;
                assert_eq!(
                    configured_resident - base_resident,
                    expected_p3_resident as u64
                );

                let mut unordered_plan = plan.clone();
                let mut swapped = false;
                for schedule in 0..GAUGE_DEGREE_COUNT * 32 {
                    let begin = unordered_plan.offsets[schedule] as usize;
                    let end = unordered_plan.offsets[schedule + 1] as usize;
                    for index in begin..end.saturating_sub(1) {
                        let left = unordered_plan.entries[index].key;
                        let right = unordered_plan.entries[index + 1].key;
                        if (left.contracted_spinor, left.template_spinor)
                            != (right.contracted_spinor, right.template_spinor)
                        {
                            unordered_plan.entries.swap(index, index + 1);
                            swapped = true;
                            break;
                        }
                    }
                    if swapped {
                        break;
                    }
                }
                assert!(swapped, "p3 plan has no distinct adjacent spinor pairs");
                let mutation_error = CudaModularP3::new_with_device_cap(
                    &static_data,
                    &unordered_plan,
                    0,
                    10 * 1024 * 1024 * 1024,
                )
                .err()
                .expect("unordered p3 pair plan was accepted");
                assert!(mutation_error.contains("unordered CUDA p3 contracted/template"));
                assert_eq!(cuda.resident_bytes(), configured_resident);
                assert_eq!(cuda.buffer_high_water_bytes(), configured_high_water);
                assert!(cuda.reconfigure_for_test(&plan).is_err());
                assert_eq!(cuda.resident_bytes(), configured_resident);
                assert_eq!(cuda.buffer_high_water_bytes(), configured_high_water);

                cuda.set_device_hard_cap(configured_resident).unwrap();
                assert!(cuda.accumulate(&column).is_err());
                assert_eq!(cuda.resident_bytes(), configured_resident);
                assert_eq!(cuda.buffer_high_water_bytes(), configured_high_water);

                let source_bytes =
                    u64::try_from(column.terms.len() * std::mem::size_of::<CudaSourceEntry>())
                        .unwrap();
                cuda.set_device_hard_cap(configured_resident + source_bytes)
                    .unwrap();
                let exact = accumulate_p3_column_cpu(&exact_static, &static_data, &column).unwrap();
                let flat = accumulate_p3_column_cpu_flat(&plan, &column).unwrap();
                let raw_fanout = column
                    .terms
                    .iter()
                    .map(|term| plan.raw_expanded_fanout(term).unwrap())
                    .sum::<u64>();
                assert_eq!(raw_fanout, flat.expanded_contributions);
                let (gpu, timing) = cuda.accumulate(&column).unwrap();
                assert_eq!(gpu.rows, flat.rows, "CUDA/flat rows at prime {prime}");
                assert_eq!(gpu.semantic_sha256, flat.semantic_sha256);
                assert_eq!(
                    gpu.expanded_contributions, flat.expanded_contributions,
                    "expanded schedule visits at prime {prime}"
                );
                assert_eq!(flat.rows, exact.rows, "flat/exact rows at prime {prime}");
                assert_eq!(flat.semantic_sha256, exact.semantic_sha256);
                assert_eq!(timing.source_count, column.terms.len());
                assert_eq!(timing.plan_entry_count as usize, plan.entry_count());
                assert_eq!(timing.expanded_contributions, gpu.expanded_contributions);
                assert_eq!(timing.resident_bytes, configured_resident + source_bytes);
                assert_eq!(
                    timing.device_hard_cap_bytes,
                    configured_resident + source_bytes
                );
                assert!(timing.buffer_high_water_bytes >= timing.resident_bytes);
                assert!(timing.kernel_milliseconds >= 0.0);
                let populated_axes = gpu
                    .rows
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| !value.is_zero())
                    .map(|(row, _)| decode_p3_functional_row(row).unwrap().contraction_axis)
                    .collect::<std::collections::BTreeSet<_>>();
                assert!(populated_axes.len() >= 2);

                let mut reduced = BTreeMap::<u64, i128>::new();
                for term in &column.terms {
                    *reduced
                        .entry(pack_recoupling_key(term).unwrap())
                        .or_default() += term.coefficient;
                }
                reduced.retain(|_, coefficient| *coefficient != 0);
                let keys = reduced.keys().copied().collect::<Vec<_>>();
                let values = reduced.values().copied().collect::<Vec<_>>();
                cuda.reset_persistent_columns(&vec![vec![
                    GaussianResidue::zero();
                    P3_FUNCTIONAL_ROW_COUNT
                ]])
                .unwrap();
                let persistent_timing = cuda
                    .accumulate_reduced_union_lane_persistent(&keys, &values, 1, 0)
                    .unwrap();
                let persistent = cuda.download_persistent_columns(1).unwrap();
                assert_eq!(persistent[0], flat.rows);
                assert!(persistent_timing.expanded_contributions < raw_fanout);

                let mut lane_columns = vec![column.clone(); 3];
                lane_columns[1].global_ordinal = 1;
                lane_columns[1].terms = column.terms[2..].to_vec();
                lane_columns[2].global_ordinal = 2;
                lane_columns[2].terms = vec![column.terms[3]];
                let expected_lanes = lane_columns
                    .iter()
                    .map(|column| accumulate_p3_column_cpu_flat(&plan, column).unwrap().rows)
                    .collect::<Vec<_>>();
                let mut union = BTreeMap::<u64, [i128; 3]>::new();
                for (lane, column) in lane_columns.iter().enumerate() {
                    for term in &column.terms {
                        union.entry(pack_recoupling_key(term).unwrap()).or_default()[lane] +=
                            term.coefficient;
                    }
                }
                union.retain(|_, values| values.iter().any(|value| *value != 0));
                let union_keys = union.keys().copied().collect::<Vec<_>>();
                let union_values = union
                    .values()
                    .flat_map(|values| values.iter().copied())
                    .collect::<Vec<_>>();
                let zero_lanes = vec![vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT]; 3];
                cuda.reset_persistent_columns(&zero_lanes).unwrap();
                let mut sequential_expanded = Vec::new();
                let mut sequential_kernel_milliseconds = 0.0_f32;
                for lane in 0..3 {
                    let timing = cuda
                        .accumulate_reduced_union_lane_persistent(
                            &union_keys,
                            &union_values,
                            3,
                            lane,
                        )
                        .unwrap();
                    sequential_expanded.push(timing.expanded_contributions);
                    sequential_kernel_milliseconds += timing.kernel_milliseconds;
                }
                assert_eq!(
                    cuda.download_persistent_columns(3).unwrap(),
                    expected_lanes,
                    "persistent width-3 rows at prime {prime}"
                );

                // The earlier transactional-cap canary intentionally pins the
                // context to its current resident bytes. Multi-lane fusion
                // owns one additional bounded union workspace, so permit that
                // first reservation before exercising exact parity.
                cuda.set_device_hard_cap(u64::MAX).unwrap();
                cuda.cuda.reserve_multicol(union_keys.len(), 3).unwrap();
                let fused_resident_bytes = cuda.resident_bytes();
                cuda.set_device_hard_cap(fused_resident_bytes).unwrap();
                cuda.reset_persistent_columns(&zero_lanes).unwrap();
                let fused_timing = cuda
                    .accumulate_reduced_union_multilane_persistent(&union_keys, &union_values, 3)
                    .unwrap();
                assert_eq!(fused_timing.source_counts.len(), 3);
                assert_eq!(fused_timing.expanded_contributions, sequential_expanded);
                assert_eq!(fused_timing.resident_bytes, fused_resident_bytes);
                assert_eq!(fused_timing.device_hard_cap_bytes, fused_resident_bytes);
                eprintln!(
                    "p3 width3 prime={prime} sequential_kernel_ms={sequential_kernel_milliseconds:.6} fused_kernel_ms={:.6}",
                    fused_timing.kernel_milliseconds
                );
                assert_eq!(
                    cuda.download_persistent_columns(3).unwrap(),
                    expected_lanes,
                    "fused persistent width-3 rows at prime {prime}"
                );

                let split = union_keys.len() / 2;
                cuda.reset_persistent_columns(&zero_lanes).unwrap();
                for (keys, values) in [
                    (&union_keys[..split], &union_values[..split * 3]),
                    (&union_keys[split..], &union_values[split * 3..]),
                ] {
                    for lane in 0..3 {
                        if values.chunks_exact(3).any(|values| values[lane] != 0) {
                            cuda.accumulate_reduced_union_lane_persistent(keys, values, 3, lane)
                                .unwrap();
                        }
                    }
                    let checkpoint_rows = cuda.download_persistent_columns(3).unwrap();
                    cuda.reset_persistent_columns(&checkpoint_rows).unwrap();
                }
                assert_eq!(
                    cuda.download_persistent_columns(3).unwrap(),
                    expected_lanes,
                    "persistent split-resume width-3 rows at prime {prime}"
                );

                let resident_after_first = cuda.resident_bytes();
                let high_water_after_first = cuda.buffer_high_water_bytes();
                let (_, repeated_timing) = cuda.accumulate(&column).unwrap();
                assert_eq!(repeated_timing.resident_bytes, resident_after_first);
                assert_eq!(
                    repeated_timing.buffer_high_water_bytes,
                    high_water_after_first
                );

                let mut grown = column.clone();
                grown.terms.push(RecoupledSourceTerm {
                    momentum_pair: [3, 9],
                    free_spinor: 12,
                    exterior_mask: 0x0000_3ffc,
                    coefficient: 11,
                });
                cuda.set_device_hard_cap(resident_after_first).unwrap();
                assert!(cuda.accumulate(&grown).is_err());
                assert_eq!(cuda.resident_bytes(), resident_after_first);
                assert_eq!(cuda.buffer_high_water_bytes(), high_water_after_first);
                let grown_source_bytes =
                    u64::try_from(grown.terms.len() * std::mem::size_of::<CudaSourceEntry>())
                        .unwrap();
                let growth_cap = resident_after_first + grown_source_bytes;
                cuda.set_device_hard_cap(growth_cap).unwrap();
                let (_, grown_timing) = cuda.accumulate(&grown).unwrap();
                assert_eq!(
                    grown_timing.resident_bytes,
                    resident_after_first - source_bytes + grown_source_bytes
                );
                assert!(
                    grown_timing.buffer_high_water_bytes
                        >= resident_after_first + grown_source_bytes
                );
                assert_eq!(grown_timing.device_hard_cap_bytes, growth_cap);
            }
        }

        #[test]
        fn p3_three_prime_cuda_fusion_matches_serial_rows_counts_and_artifacts() {
            assert_eq!(std::mem::size_of::<CudaP3ThreePrimePlanEntry>(), 32);
            assert_eq!(std::mem::align_of::<CudaP3ThreePrimePlanEntry>(), 8);
            assert_eq!(
                std::mem::offset_of!(CudaP3ThreePrimePlanEntry, scaled_real),
                20
            );
            assert_eq!(
                std::mem::offset_of!(CudaP3ThreePrimePlanEntry, functional_salt),
                24
            );
            let static_data: [ModularFxStaticData; 3] =
                GPU_FX_PRIMES.map(|prime| ModularFxStaticData::build(prime).unwrap());
            let plans: [P3ModularFlatPlan; 3] =
                std::array::from_fn(|slot| build_p3_modular_flat_plan(&static_data[slot]).unwrap());
            let base_resident = CudaModularFx::new(&static_data[0], 0)
                .unwrap()
                .resident_bytes();
            assert!(
                CudaModularP3ThreePrime::new_with_device_cap(
                    &static_data,
                    &plans,
                    0,
                    base_resident
                )
                .is_err()
            );
            let mut mutated = plans.clone();
            mutated[1].entries[0].key.output_coordinate ^= 1;
            assert!(
                CudaModularP3ThreePrime::new_with_device_cap(
                    &static_data,
                    &mutated,
                    0,
                    10 * 1024 * 1024 * 1024
                )
                .is_err()
            );

            let mut fused = CudaModularP3ThreePrime::new_with_device_cap(
                &static_data,
                &plans,
                0,
                10 * 1024 * 1024 * 1024,
            )
            .unwrap();
            assert_eq!(fused.plan_entry_count() as usize, plans[0].entry_count());
            assert_eq!(
                fused.flat_plan_sha256(),
                &std::array::from_fn::<_, 3, _>(|slot| plans[slot].semantic_sha256.clone())
            );
            const PAIR_OFFSET_BYTES: usize =
                (GAUGE_DEGREE_COUNT * 32 * 32 * 32 + 1) * std::mem::size_of::<u32>();
            // The authenticated host ABI is 32 bytes, while configure_p3_three_prime
            // projects it into a validated 16-byte device-only hot entry.
            const COMPACT_DEVICE_PLAN_ENTRY_BYTES: usize = 16;
            let configured_expected = base_resident
                + ((GAUGE_DEGREE_COUNT * 32 + 1) * std::mem::size_of::<u32>()) as u64
                + PAIR_OFFSET_BYTES as u64
                + (plans[0].entry_count() * COMPACT_DEVICE_PLAN_ENTRY_BYTES) as u64
                + (3 * 2 * P3_FUNCTIONAL_ROW_COUNT * 32 * std::mem::size_of::<u32>()) as u64
                + (3 * 32 * std::mem::size_of::<u64>() + std::mem::size_of::<u32>()) as u64;
            assert_eq!(fused.resident_bytes(), configured_expected);

            let sources = [
                RecoupledSourceTerm {
                    momentum_pair: [0, 1],
                    free_spinor: 7,
                    exterior_mask: 0x0000_0fff,
                    coefficient: 1,
                },
                RecoupledSourceTerm {
                    momentum_pair: [2, 5],
                    free_spinor: 8,
                    exterior_mask: 0x0000_1ffe,
                    coefficient: 1,
                },
                RecoupledSourceTerm {
                    momentum_pair: [4, 9],
                    free_spinor: 9,
                    exterior_mask: 0x0000_3ffc,
                    coefficient: 1,
                },
            ];
            let lane_values = [
                [5_i128, -3, 0],
                [i128::from(GPU_FX_PRIMES[0]), 11, -13],
                [2, 0, -7],
            ];
            let mut union = sources
                .iter()
                .zip(lane_values)
                .map(|(source, values)| (pack_recoupling_key(source).unwrap(), values))
                .collect::<Vec<_>>();
            union.sort_by_key(|(key, _)| *key);
            let keys = union.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            let values = union
                .iter()
                .flat_map(|(_, values)| values.iter().copied())
                .collect::<Vec<_>>();
            let resident_before_failed_reservation = fused.resident_bytes();
            let high_water_before_failed_reservation = fused.buffer_high_water_bytes();
            fused
                .cuda
                .set_recoupling_hard_cap(resident_before_failed_reservation)
                .unwrap();
            assert!(
                fused
                    .accumulate_reduced_union_multilane_persistent(&keys, &values, 3)
                    .is_err()
            );
            assert_eq!(fused.resident_bytes(), resident_before_failed_reservation);
            assert_eq!(
                fused.buffer_high_water_bytes(),
                high_water_before_failed_reservation
            );
            fused
                .cuda
                .set_recoupling_hard_cap(10 * 1024 * 1024 * 1024)
                .unwrap();
            let zero_rows =
                vec![vec![vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT]; 3]; 3];
            fused.reset_persistent_columns(&zero_rows).unwrap();
            let fused_timing = fused
                .accumulate_reduced_union_multilane_persistent(&keys, &values, 3)
                .unwrap();
            assert_eq!(fused_timing.source_counts[0], vec![2, 2, 2]);
            assert_eq!(fused_timing.source_counts[1], vec![3, 2, 2]);
            assert_eq!(fused_timing.source_counts[2], vec![3, 2, 2]);
            let fused_rows = fused.download_persistent_columns(3).unwrap();

            // A durable boundary stores all three prime-major row planes. A
            // fresh context must restore that state and produce the exact
            // uninterrupted result after the remaining union keys.
            let split = 1;
            let mut before_restart = CudaModularP3ThreePrime::new_with_device_cap(
                &static_data,
                &plans,
                0,
                10 * 1024 * 1024 * 1024,
            )
            .unwrap();
            before_restart.reset_persistent_columns(&zero_rows).unwrap();
            let first_timing = before_restart
                .accumulate_reduced_union_multilane_persistent(
                    &keys[..split],
                    &values[..split * 3],
                    3,
                )
                .unwrap();
            let durable_rows = before_restart.download_persistent_columns(3).unwrap();
            drop(before_restart);
            let mut after_restart = CudaModularP3ThreePrime::new_with_device_cap(
                &static_data,
                &plans,
                0,
                10 * 1024 * 1024 * 1024,
            )
            .unwrap();
            after_restart
                .reset_persistent_columns(&durable_rows)
                .unwrap();
            let second_timing = after_restart
                .accumulate_reduced_union_multilane_persistent(
                    &keys[split..],
                    &values[split * 3..],
                    3,
                )
                .unwrap();
            assert_eq!(
                after_restart.download_persistent_columns(3).unwrap(),
                fused_rows
            );
            for prime_slot in 0..3 {
                for lane in 0..3 {
                    assert_eq!(
                        first_timing.expanded_contributions[prime_slot][lane]
                            + second_timing.expanded_contributions[prime_slot][lane],
                        fused_timing.expanded_contributions[prime_slot][lane]
                    );
                }
            }

            let mut semantic_hashes = std::collections::BTreeSet::new();
            for prime_slot in 0..3 {
                let mut serial = CudaModularP3::new_with_device_cap(
                    &static_data[prime_slot],
                    &plans[prime_slot],
                    0,
                    10 * 1024 * 1024 * 1024,
                )
                .unwrap();
                serial
                    .reset_persistent_columns(&vec![
                        vec![
                            GaussianResidue::zero();
                            P3_FUNCTIONAL_ROW_COUNT
                        ];
                        3
                    ])
                    .unwrap();
                let serial_timing = serial
                    .accumulate_reduced_union_multilane_persistent(&keys, &values, 3)
                    .unwrap();
                let serial_rows = serial.download_persistent_columns(3).unwrap();
                assert_eq!(fused_rows[prime_slot], serial_rows);
                assert_eq!(
                    fused_timing.expanded_contributions[prime_slot],
                    serial_timing.expanded_contributions
                );
                for lane in 0..3 {
                    let column = ModularP3FunctionalColumn {
                        prime: GPU_FX_PRIMES[prime_slot],
                        global_ordinal: lane,
                        semantic_sha256: p3_column_semantic_sha256(
                            GPU_FX_PRIMES[prime_slot],
                            lane,
                            &fused_rows[prime_slot][lane],
                        ),
                        rows: fused_rows[prime_slot][lane].clone(),
                        expanded_contributions: fused_timing.expanded_contributions[prime_slot]
                            [lane],
                    };
                    let bytes =
                        encode_p3_column_artifact(&plans[prime_slot].semantic_sha256, &column)
                            .unwrap();
                    let (decoded_plan, decoded) = decode_p3_column_artifact(&bytes).unwrap();
                    assert_eq!(decoded_plan, plans[prime_slot].semantic_sha256);
                    assert_eq!(decoded.prime, column.prime);
                    assert_eq!(decoded.global_ordinal, column.global_ordinal);
                    assert_eq!(decoded.rows, column.rows);
                    assert_eq!(
                        decoded.expanded_contributions,
                        column.expanded_contributions
                    );
                    assert_eq!(decoded.semantic_sha256, column.semantic_sha256);
                    semantic_hashes.insert(decoded.semantic_sha256);
                }
            }
            assert_eq!(semantic_hashes.len(), 9);
        }

        #[test]
        #[ignore = "runs one real width-3 PBW word through serial and fused three-prime CUDA paths"]
        fn p3_three_prime_real_word_benchmark_matches_three_serial_runs() {
            use crate::second_momentum_gpu_group::{
                GpuFxTranche, GroupRuntimeIdentity, GroupWordOrchestrationConfig,
                prepare_cuda_column_group,
            };
            const WIDTH: usize = 3;
            const LOCAL_ORDINALS: [usize; WIDTH] = [4, 5, 6];
            const RAW_BATCH_TERMS: usize = 65_536;
            const MAX_UNION_KEYS: usize = 131_072;
            const MIB: u64 = 1024 * 1024;
            const GIB: u64 = 1024 * MIB;

            let static_data: [ModularFxStaticData; 3] =
                GPU_FX_PRIMES.map(|prime| ModularFxStaticData::build(prime).unwrap());
            let p3_plans: [P3ModularFlatPlan; 3] =
                std::array::from_fn(|slot| build_p3_modular_flat_plan(&static_data[slot]).unwrap());
            let plans: Vec<crate::second_momentum_gpu_group::PreparedColumnGroup> = (0..3)
                .map(|slot| {
                    let probe = CudaModularFx::new(&static_data[slot], 0).unwrap();
                    let runtime = GroupRuntimeIdentity {
                        prime: GPU_FX_PRIMES[slot],
                        static_semantic_sha256: static_data[slot].semantic_sha256().to_string(),
                        flat_plan_sha256: probe.flat_plan_sha256().to_string(),
                    };
                    drop(probe);
                    prepare_cuda_column_group(GpuFxTranche::Two0001, &LOCAL_ORDINALS, runtime)
                        .unwrap()
                })
                .collect();
            assert!(plans.windows(2).all(|pair| {
                pair[0].ordered_global_ordinals == pair[1].ordered_global_ordinals
                    && pair[0].pbw_word_count == pair[1].pbw_word_count
            }));
            let config = GroupWordOrchestrationConfig {
                start_word_ordinal: 0,
                end_word_ordinal_exclusive: 1,
                first_global_batch_ordinal: 0,
                raw_batch_term_cap_per_lane: RAW_BATCH_TERMS,
                max_union_keys_per_batch: MAX_UNION_KEYS,
                aggregate_host_payload_cap_bytes: 4 * GIB,
            };
            let mut serial_p2_rows = Vec::new();
            let mut serial_p3_rows = Vec::new();
            let mut serial_expanded = Vec::new();
            let mut serial_raw_counts = Vec::new();
            let mut serial_p3_kernel_ms = Vec::new();
            let mut serial_wall_seconds = Vec::new();
            for slot in 0..3 {
                let mut executor = super::super::PersistentCudaGroupExecutor::new(
                    plans[slot].clone(),
                    &static_data[slot],
                    0,
                    MAX_UNION_KEYS,
                    6 * GIB,
                    2 * GIB,
                    GIB,
                    1_048_576,
                )
                .unwrap();
                let mut p3 = CudaModularP3::new_with_device_cap(
                    &static_data[slot],
                    &p3_plans[slot],
                    0,
                    2 * GIB,
                )
                .unwrap();
                p3.reset_persistent_columns(&vec![
                    vec![
                        GaussianResidue::zero();
                        P3_FUNCTIONAL_ROW_COUNT
                    ];
                    WIDTH
                ])
                .unwrap();
                let mut raw_counts = vec![0_u64; WIDTH];
                let mut expanded = vec![0_u64; WIDTH];
                let mut kernel_ms = 0.0_f64;
                let started = std::time::Instant::now();
                executor
                    .run_word_synchronous_batched_with_union(
                        config.clone(),
                        |lane, _, terms| {
                            raw_counts[lane] += terms.len() as u64;
                            Ok(())
                        },
                        |_, union| {
                            let timing = p3.accumulate_reduced_union_multilane_persistent(
                                &union.keys,
                                &union.key_major_values,
                                WIDTH,
                            )?;
                            for lane in 0..WIDTH {
                                expanded[lane] += timing.expanded_contributions[lane];
                            }
                            kernel_ms += f64::from(timing.kernel_milliseconds);
                            Ok(())
                        },
                        |_| Ok(()),
                        |_, _| Ok(()),
                    )
                    .unwrap();
                serial_wall_seconds.push(started.elapsed().as_secs_f64());
                serial_raw_counts.push(raw_counts);
                serial_expanded.push(expanded);
                serial_p3_kernel_ms.push(kernel_ms);
                serial_p2_rows.push(executor.final_columns().to_vec());
                serial_p3_rows.push(p3.download_persistent_columns(WIDTH).unwrap());
            }

            let mut fused_executor = super::super::PersistentCudaMultiPrimeGroupExecutor::new(
                plans.clone(),
                &static_data,
                0,
                MAX_UNION_KEYS,
                10 * GIB,
                2 * GIB,
                GIB,
                1_048_576,
                None,
            )
            .unwrap();
            let mut fused_p3 =
                CudaModularP3ThreePrime::new_with_device_cap(&static_data, &p3_plans, 0, 2 * GIB)
                    .unwrap();
            fused_p3
                .reset_persistent_columns(&vec![
                    vec![
                        vec![
                            GaussianResidue::zero();
                            P3_FUNCTIONAL_ROW_COUNT
                        ];
                        WIDTH
                    ];
                    3
                ])
                .unwrap();
            let mut fused_raw_counts = vec![0_u64; WIDTH];
            let mut fused_expanded = vec![vec![0_u64; WIDTH]; 3];
            let mut fused_kernel_ms = 0.0_f64;
            let started = std::time::Instant::now();
            fused_executor
                .run_word_synchronous_batched_with_union(
                    config,
                    |lane, _, terms| {
                        fused_raw_counts[lane] += terms.len() as u64;
                        Ok(())
                    },
                    |_, union| {
                        let timing = fused_p3.accumulate_reduced_union_multilane_persistent(
                            &union.keys,
                            &union.key_major_values,
                            WIDTH,
                        )?;
                        for prime_slot in 0..3 {
                            for lane in 0..WIDTH {
                                fused_expanded[prime_slot][lane] +=
                                    timing.expanded_contributions[prime_slot][lane];
                            }
                        }
                        fused_kernel_ms += f64::from(timing.kernel_milliseconds);
                        Ok(())
                    },
                    |_, _, _| Ok(()),
                    |_, _| Ok(()),
                )
                .unwrap();
            let fused_wall_seconds = started.elapsed().as_secs_f64();
            let fused_p3_rows = fused_p3.download_persistent_columns(WIDTH).unwrap();
            for slot in 0..3 {
                assert_eq!(fused_raw_counts, serial_raw_counts[slot]);
                assert_eq!(fused_expanded[slot], serial_expanded[slot]);
                assert_eq!(fused_p3_rows[slot], serial_p3_rows[slot]);
                assert_eq!(
                    fused_executor.final_columns(slot).unwrap(),
                    serial_p2_rows[slot]
                );
                for lane in 0..WIDTH {
                    let ordinal = plans[slot].ordered_global_ordinals[lane];
                    let column = ModularP3FunctionalColumn {
                        prime: GPU_FX_PRIMES[slot],
                        global_ordinal: ordinal,
                        rows: fused_p3_rows[slot][lane].clone(),
                        expanded_contributions: fused_expanded[slot][lane],
                        semantic_sha256: p3_column_semantic_sha256(
                            GPU_FX_PRIMES[slot],
                            ordinal,
                            &fused_p3_rows[slot][lane],
                        ),
                    };
                    let bytes =
                        encode_p3_column_artifact(p3_plans[slot].semantic_sha256(), &column)
                            .unwrap();
                    let (digest, decoded) = decode_p3_column_artifact(&bytes).unwrap();
                    assert_eq!(digest, p3_plans[slot].semantic_sha256());
                    assert_eq!(decoded.semantic_sha256, column.semantic_sha256);
                }
            }
            let serial_wall_total: f64 = serial_wall_seconds.iter().sum();
            println!(
                "{}",
                serde_json::json!({
                    "event": "p3_three_prime_real_word_benchmark",
                    "serial_wall_seconds": serial_wall_seconds,
                    "serial_wall_total_seconds": serial_wall_total,
                    "serial_p3_kernel_milliseconds": serial_p3_kernel_ms,
                    "fused_wall_seconds": fused_wall_seconds,
                    "fused_p3_kernel_milliseconds": fused_kernel_ms,
                    "wall_speedup": serial_wall_total / fused_wall_seconds,
                    "raw_terms_per_column": fused_raw_counts,
                    "union_batches": fused_executor.batches_folded().unwrap(),
                })
            );
        }

        fn sample_column(term_count: usize) -> GpuFxColumnInput {
            let base_mask = (0..12).fold(0_u32, |mask, bit| mask | (1_u32 << bit));
            GpuFxColumnInput {
                global_ordinal: 53,
                source_label: "synthetic".to_string(),
                source_copy: 1,
                terms: (0..term_count)
                    .map(|ordinal| RecoupledSourceTerm {
                        momentum_pair: [
                            (ordinal % 11) as u8,
                            ((ordinal % 11) + (ordinal / 11) % (11 - ordinal % 11)) as u8,
                        ],
                        free_spinor: (ordinal % 32) as u8,
                        exterior_mask: base_mask.rotate_left((ordinal % 32) as u32),
                        coefficient: ordinal as i128 * 17 - 31,
                    })
                    .collect(),
                raising_residuals: [0; 5],
            }
        }

        fn representative_unique_column(term_count: usize) -> GpuFxColumnInput {
            let mut mask = (1_u32 << 12) - 1;
            let mut terms = Vec::with_capacity(term_count);
            for ordinal in 0..term_count {
                terms.push(RecoupledSourceTerm {
                    momentum_pair: [((ordinal / 11) % 11) as u8, 10],
                    free_spinor: (ordinal % 32) as u8,
                    exterior_mask: mask,
                    coefficient: ordinal as i128 * 17 - 31,
                });
                let low = mask & mask.wrapping_neg();
                let ripple = mask.wrapping_add(low);
                mask = ripple | (((mask ^ ripple) >> 2) / low);
            }
            GpuFxColumnInput {
                global_ordinal: 74,
                source_label: "representative-unique-131072".to_string(),
                source_copy: 12,
                terms,
                raising_residuals: [0; 5],
            }
        }

        fn capture_cpu_20001_word(
            local_ordinal: usize,
            word_ordinal: usize,
        ) -> (
            crate::eleven_dimensional_second_momentum_20001_fx::SecondMomentum20001GpuColumnPreflight,
            Vec<RecoupledSourceTerm>,
        ){
            const COMPLETE: &str = "CPU one-word canary complete";
            use crate::eleven_dimensional_second_momentum_20001_fx::SecondMomentum20001GpuColumnEvent;
            let preflight =
                crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(
                    local_ordinal,
                )
                .unwrap();
            let mut terms = Vec::new();
            let result = crate::eleven_dimensional_second_momentum_20001_fx::
                visit_gpu_column_contribution_events_from(
                    &preflight,
                    word_ordinal,
                    |event| match event {
                        SecondMomentum20001GpuColumnEvent::Term {
                            requested_word_ordinal,
                            term,
                        } => {
                            assert_eq!(requested_word_ordinal, word_ordinal);
                            terms.push(term);
                            Ok(())
                        }
                        SecondMomentum20001GpuColumnEvent::WordEnd {
                            requested_word_ordinal,
                            raw_terms_emitted,
                        } if requested_word_ordinal == word_ordinal => {
                            assert_eq!(raw_terms_emitted, terms.len() as u64);
                            Err(std::io::Error::other(COMPLETE))
                        }
                        _ => Ok(()),
                    },
                );
            match result {
                Err(error) if error.to_string() == COMPLETE => {}
                Err(error) => panic!("CPU one-word capture failed: {error}"),
                Ok(_) => panic!("CPU one-word capture did not stop at its word boundary"),
            }
            (preflight, terms)
        }

        fn contract_streamed_expected(
            static_data: &ModularFxStaticData,
            global_ordinal: usize,
            terms: &[RecoupledSourceTerm],
        ) -> Vec<GaussianResidue> {
            let mut cuda = CudaModularFx::new(static_data, 0).unwrap();
            let mut rows = vec![GaussianResidue::zero(); FUNCTIONAL_ROW_COUNT];
            for chunk in terms.chunks(131_072) {
                let (delta, _) = cuda.accumulate_terms(chunk, global_ordinal).unwrap();
                for (row, value) in rows.iter_mut().zip(delta) {
                    *row = row.add(value, static_data.prime());
                }
            }
            rows
        }

        fn run_real_persistent_group_word_canary(local_ordinals: &[usize], word_ordinal: usize) {
            use crate::second_momentum_gpu_group::{
                GpuFxTranche, GroupRuntimeIdentity, GroupWordOrchestrationConfig,
                prepare_cuda_column_group,
            };
            let prime = GPU_FX_PRIMES[0];
            let static_data = ModularFxStaticData::build(prime).unwrap();
            let flat_plan_sha256 = {
                let cuda = CudaModularFx::new(&static_data, 0).unwrap();
                cuda.flat_plan_sha256().to_string()
            };
            let plan = prepare_cuda_column_group(
                GpuFxTranche::Two0001,
                local_ordinals,
                GroupRuntimeIdentity {
                    prime,
                    static_semantic_sha256: static_data.semantic_sha256().to_string(),
                    flat_plan_sha256,
                },
            )
            .unwrap();
            let captured = local_ordinals
                .iter()
                .map(|&local| capture_cpu_20001_word(local, word_ordinal))
                .collect::<Vec<_>>();
            let expected_rows = captured
                .iter()
                .map(|(preflight, terms)| {
                    contract_streamed_expected(&static_data, preflight.global_column_ordinal, terms)
                })
                .collect::<Vec<_>>();

            let mut executor = super::super::PersistentCudaGroupExecutor::new(
                plan,
                &static_data,
                0,
                262_144,
                12 * 1024 * 1024 * 1024,
                512 * 1024 * 1024,
                512 * 1024 * 1024,
                131_072,
            )
            .unwrap();
            let mut observed = vec![Vec::new(); local_ordinals.len()];
            let report = executor
                .run_word_synchronous(
                    GroupWordOrchestrationConfig {
                        start_word_ordinal: word_ordinal,
                        end_word_ordinal_exclusive: word_ordinal + 1,
                        first_global_batch_ordinal: 0,
                        raw_batch_term_cap_per_lane: 131_072,
                        max_union_keys_per_batch: 262_144,
                        aggregate_host_payload_cap_bytes: 512 * 1024 * 1024,
                    },
                    |lane, observed_word, term| {
                        assert_eq!(observed_word, word_ordinal);
                        observed[lane].push(term.clone());
                        Ok(())
                    },
                    |_| Ok(()),
                    |completed_word, completions| {
                        assert_eq!(completed_word, word_ordinal);
                        assert_eq!(completions.len(), local_ordinals.len());
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(report.completed_words, 1);
            for lane in 0..local_ordinals.len() {
                assert_eq!(observed[lane], captured[lane].1, "raw lane {lane}");
                assert_eq!(executor.final_columns()[lane], expected_rows[lane]);
            }
            let summaries = executor.lowering_summaries().unwrap();
            assert_eq!(summaries.len(), local_ordinals.len());
            assert!(summaries.iter().all(|summary| {
                summary.enabled
                    && summary.roots_lowered != 0
                    && summary.maximum_absolute_coefficient != 0
            }));
            let budget = executor.device_budget();
            let reserved = budget.contraction_hard_cap_bytes
                + budget.shared_lowering_hard_cap_bytes
                + budget.reserved_headroom_bytes;
            assert!(reserved <= budget.aggregate_hard_cap_bytes);
            eprintln!(
                "{}",
                serde_json::json!({
                    "event": "persistent_group_one_word_canary",
                    "width": local_ordinals.len(),
                    "word_ordinal": word_ordinal,
                    "raw_terms_per_lane": captured.iter().map(|entry| entry.1.len()).collect::<Vec<_>>(),
                    "union_batches": report.union_batches,
                    "aggregate_device_cap_bytes": budget.aggregate_hard_cap_bytes,
                    "shared_lowering_cap_bytes": budget.shared_lowering_hard_cap_bytes,
                })
            );
        }

        #[test]
        fn persistent_group_device_budget_is_aggregate() {
            let budget = super::super::PersistentGroupDeviceBudget::partition(
                1024 * 1024 * 1024,
                256 * 1024 * 1024,
                3,
            )
            .unwrap();
            assert_eq!(budget.active_lanes, 3);
            assert!(
                budget.contraction_hard_cap_bytes
                    + budget.shared_lowering_hard_cap_bytes
                    + budget.reserved_headroom_bytes
                    <= budget.aggregate_hard_cap_bytes
            );
            assert!(
                super::super::PersistentGroupDeviceBudget::partition(64 * 1024 * 1024, 1, 2,)
                    .is_err()
            );
        }

        #[test]
        #[ignore = "runs exact real width-2 and width-3 persistent one-word group canaries"]
        fn persistent_group_real_width_2_and_3_one_word_parity() {
            run_real_persistent_group_word_canary(&[0, 1], 1);
            run_real_persistent_group_word_canary(&[4, 5, 6], 1);
        }

        #[test]
        #[ignore = "runs exact real width-2 one-word parity through two prime contexts"]
        fn persistent_multi_prime_real_width_2_one_word_parity() {
            use crate::second_momentum_gpu_group::{
                GpuFxTranche, GroupRuntimeIdentity, GroupWordOrchestrationConfig,
                prepare_cuda_column_group,
            };
            let local_ordinals = [0, 1];
            let word_ordinal = 1;
            let captured = local_ordinals
                .iter()
                .map(|&local| capture_cpu_20001_word(local, word_ordinal))
                .collect::<Vec<_>>();
            let mut static_data = Vec::new();
            let mut plans = Vec::new();
            let mut expected = Vec::new();
            for &prime in &GPU_FX_PRIMES[1..] {
                let data = ModularFxStaticData::build(prime).unwrap();
                let flat_plan_sha256 = {
                    let cuda = CudaModularFx::new(&data, 0).unwrap();
                    cuda.flat_plan_sha256().to_string()
                };
                plans.push(
                    prepare_cuda_column_group(
                        GpuFxTranche::Two0001,
                        &local_ordinals,
                        GroupRuntimeIdentity {
                            prime,
                            static_semantic_sha256: data.semantic_sha256().to_string(),
                            flat_plan_sha256,
                        },
                    )
                    .unwrap(),
                );
                expected.push(
                    captured
                        .iter()
                        .map(|(preflight, terms)| {
                            contract_streamed_expected(
                                &data,
                                preflight.global_column_ordinal,
                                terms,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                static_data.push(data);
            }
            let mut executor = super::super::PersistentCudaMultiPrimeGroupExecutor::new(
                plans,
                &static_data,
                0,
                262_144,
                12 * 1024 * 1024 * 1024,
                512 * 1024 * 1024,
                512 * 1024 * 1024,
                131_072,
                None,
            )
            .unwrap();
            let mut observed = vec![Vec::new(); local_ordinals.len()];
            let mut batch_counts = vec![0_u64; static_data.len()];
            let report = executor
                .run_word_synchronous_batched(
                    GroupWordOrchestrationConfig {
                        start_word_ordinal: word_ordinal,
                        end_word_ordinal_exclusive: word_ordinal + 1,
                        first_global_batch_ordinal: 0,
                        raw_batch_term_cap_per_lane: 131_072,
                        max_union_keys_per_batch: 262_144,
                        aggregate_host_payload_cap_bytes: 512 * 1024 * 1024,
                    },
                    |lane, observed_word, terms| {
                        assert_eq!(observed_word, word_ordinal);
                        observed[lane].extend_from_slice(terms);
                        Ok(())
                    },
                    |prime_slot, prime, _| {
                        assert_eq!(prime, GPU_FX_PRIMES[prime_slot + 1]);
                        batch_counts[prime_slot] += 1;
                        Ok(())
                    },
                    |completed_word, completions| {
                        assert_eq!(completed_word, word_ordinal);
                        assert_eq!(completions.len(), local_ordinals.len());
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(report.completed_words, 1);
            assert!(
                batch_counts
                    .iter()
                    .all(|count| *count == report.union_batches)
            );
            for lane in 0..local_ordinals.len() {
                assert_eq!(observed[lane], captured[lane].1, "raw lane {lane}");
            }
            for (prime_slot, expected_columns) in expected.iter().enumerate() {
                assert_eq!(
                    executor.final_columns(prime_slot).unwrap(),
                    expected_columns
                );
            }
            assert_eq!(executor.batches_folded().unwrap(), report.union_batches);
            eprintln!(
                "{}",
                serde_json::json!({
                    "event": "persistent_multi_prime_one_word_canary",
                    "prime_indices": [1, 2],
                    "width": local_ordinals.len(),
                    "word_ordinal": word_ordinal,
                    "raw_terms_per_lane": captured.iter().map(|entry| entry.1.len()).collect::<Vec<_>>(),
                    "union_batches": report.union_batches,
                    "device_budget": executor.device_budget(),
                })
            );
        }

        #[test]
        #[ignore = "runs the unified full-inventory adapter against established column 62"]
        fn persistent_full_adapter_matches_established_30001_word() {
            use crate::second_momentum_gpu_group::{
                GpuFxTranche, GroupRuntimeIdentity, GroupWordOrchestrationConfig,
                prepare_cuda_column_group,
            };
            let prime = GPU_FX_PRIMES[0];
            let static_data = ModularFxStaticData::build(prime).unwrap();
            let flat_plan_sha256 = {
                let cuda = CudaModularFx::new(&static_data, 0).unwrap();
                cuda.flat_plan_sha256().to_string()
            };
            let plan = prepare_cuda_column_group(
                GpuFxTranche::Three0001,
                &[0],
                GroupRuntimeIdentity {
                    prime,
                    static_semantic_sha256: static_data.semantic_sha256().to_string(),
                    flat_plan_sha256,
                },
            )
            .unwrap();
            let config = GroupWordOrchestrationConfig {
                start_word_ordinal: 1,
                end_word_ordinal_exclusive: 2,
                first_global_batch_ordinal: 0,
                raw_batch_term_cap_per_lane: 131_072,
                max_union_keys_per_batch: 262_144,
                aggregate_host_payload_cap_bytes: 512 * 1024 * 1024,
            };
            let mut established = super::super::PersistentCudaGroupExecutor::new(
                plan.clone(),
                &static_data,
                0,
                262_144,
                12 * 1024 * 1024 * 1024,
                512 * 1024 * 1024,
                512 * 1024 * 1024,
                131_072,
            )
            .unwrap();
            let mut established_terms = Vec::new();
            established
                .run_word_synchronous(
                    config.clone(),
                    |lane, word, term| {
                        assert_eq!(lane, 0);
                        assert_eq!(word, 1);
                        established_terms.push(term.clone());
                        Ok(())
                    },
                    |_| Ok(()),
                    |_, _| Ok(()),
                )
                .unwrap();
            let established_rows = established.final_columns()[0].clone();
            drop(established);

            let mut unified = super::super::PersistentCudaGroupExecutor::new_full(
                plan,
                &static_data,
                0,
                262_144,
                12 * 1024 * 1024 * 1024,
                512 * 1024 * 1024,
                512 * 1024 * 1024,
                131_072,
                std::path::Path::new("unused-for-established-column-parity"),
            )
            .unwrap();
            let mut unified_terms = Vec::new();
            unified
                .run_word_synchronous(
                    config,
                    |lane, word, term| {
                        assert_eq!(lane, 0);
                        assert_eq!(word, 1);
                        unified_terms.push(term.clone());
                        Ok(())
                    },
                    |_| Ok(()),
                    |_, _| Ok(()),
                )
                .unwrap();
            assert_eq!(unified_terms, established_terms);
            assert_eq!(unified.final_columns()[0], established_rows);
            assert_eq!(
                unified.final_column_semantic_sha256(),
                vec![column_semantic_sha256(
                    prime,
                    62,
                    static_data.semantic_sha256(),
                    &established_rows,
                )]
            );
        }

        #[test]
        fn cuda_matches_fused_cpu_reference() {
            let static_data = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
            let input = sample_column(97);
            let cpu = accumulate_column_cpu(&static_data, &input).unwrap();
            let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
            assert!(cuda.resident_bytes() > 0);
            let (gpu, timing) = cuda.accumulate(&input).unwrap();
            assert_eq!(gpu.rows, cpu.rows);
            assert_eq!(gpu.semantic_sha256, cpu.semantic_sha256);
            assert_eq!(gpu.expanded_contributions, cpu.expanded_contributions);
            assert!(timing.kernel_milliseconds > 0.0);
            assert_eq!(timing.keys_after_reduce, input.terms.len() as u64);
            assert_eq!(timing.nonzero_terms_after_reduce, input.terms.len() as u64);
            assert!(timing.buffer_high_water_bytes > 0);
            assert!(!timing.packed_input_sha256.is_empty());
        }

        #[test]
        fn flat_plan_matches_legacy_and_cpu_for_all_pinned_primes() {
            let input = sample_column(257);
            for prime in GPU_FX_PRIMES {
                let static_data = ModularFxStaticData::build(prime).unwrap();
                let cpu = accumulate_column_cpu(&static_data, &input).unwrap();
                let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
                assert_eq!(cuda.flat_plan_sha256().len(), 64);
                let (planned, planned_timing) = cuda.accumulate(&input).unwrap();
                cuda.set_legacy_contraction(true).unwrap();
                let (legacy, legacy_timing) = cuda.accumulate(&input).unwrap();
                assert_eq!(planned.rows, legacy.rows, "prime {prime}");
                assert_eq!(planned.rows, cpu.rows, "prime {prime}");
                assert_eq!(
                    planned.expanded_contributions, legacy.expanded_contributions,
                    "prime {prime}"
                );
                assert_eq!(
                    planned_timing.nonzero_terms_after_reduce,
                    legacy_timing.nonzero_terms_after_reduce,
                    "prime {prime}"
                );
            }
        }

        fn assert_multicol_exact_width(width: usize, prime: u32) {
            let static_data = ModularFxStaticData::build(prime).unwrap();
            let base = sample_column(257);
            let mut distinct_lanes = Vec::new();
            for lane in 0..3 {
                let mut reduced = BTreeMap::<u64, i128>::new();
                for (ordinal, term) in base.terms.iter().enumerate() {
                    if lane == 1 && ordinal % 7 == 0 {
                        continue;
                    }
                    let coefficient = match lane {
                        0 => term.coefficient,
                        1 => term.coefficient * 3 + 1,
                        _ => -term.coefficient * 5 - 2,
                    };
                    *reduced
                        .entry(pack_recoupling_key(term).unwrap())
                        .or_default() += coefficient;
                }
                reduced.retain(|_, coefficient| *coefficient != 0);
                distinct_lanes.push(reduced);
            }
            let lanes = (0..width)
                .map(|lane| distinct_lanes[lane % distinct_lanes.len()].clone())
                .collect::<Vec<_>>();
            let mut union = BTreeMap::<u64, Vec<i128>>::new();
            for (lane, reduced) in lanes.iter().enumerate() {
                for (&key, &coefficient) in reduced {
                    union.entry(key).or_insert_with(|| vec![0; width])[lane] = coefficient;
                }
            }
            union.retain(|_, values| values.iter().any(|value| *value != 0));
            let keys = union.keys().copied().collect::<Vec<_>>();
            let values = union
                .values()
                .flat_map(|values| values.iter().copied())
                .collect::<Vec<_>>();

            let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
            let mut expected_rows = Vec::new();
            let mut expected_expanded = Vec::new();
            for (lane, reduced) in lanes.iter().enumerate() {
                let terms = reduced
                    .iter()
                    .map(|(&key, &coefficient)| {
                        let (momentum_pair, free_spinor, exterior_mask) =
                            unpack_recoupling_key(key).unwrap();
                        RecoupledSourceTerm {
                            momentum_pair,
                            free_spinor,
                            exterior_mask,
                            coefficient,
                        }
                    })
                    .collect::<Vec<_>>();
                let (rows, timing) = cuda.accumulate_terms(&terms, 10_000 + lane).unwrap();
                expected_rows.push(rows);
                expected_expanded.push(timing.expanded_contributions);
            }
            cuda.reserve_multicol(keys.len(), width).unwrap();
            let reserved_bytes = cuda.resident_bytes();
            let observed = cuda
                .accumulate_reduced_multicol(&keys, &values, width)
                .unwrap();
            assert_eq!(observed.columns, expected_rows, "width {width}");
            assert_eq!(observed.stats.unique_count, keys.len() as u64);
            assert_eq!(observed.stats.active_columns, width as u32);
            assert_eq!(observed.stats.resident_bytes, reserved_bytes);
            assert!(observed.stats.buffer_high_water_bytes >= reserved_bytes);
            for lane in 0..width {
                assert_eq!(
                    observed.stats.nonzero_terms[lane],
                    lanes[lane].len() as u64,
                    "width {width} lane {lane}"
                );
                assert_eq!(
                    observed.stats.expanded_contributions[lane], expected_expanded[lane],
                    "width {width} lane {lane}"
                );
            }
        }

        #[test]
        fn multicol_cuda_exact_parity_for_production_widths() {
            for prime in GPU_FX_PRIMES {
                for width in [1, 2, 3] {
                    assert_multicol_exact_width(width, prime);
                }
            }
        }

        #[test]
        fn multicol_cuda_exact_parity_through_width_32() {
            for width in [4, 8, 15, 32] {
                assert_multicol_exact_width(width, GPU_FX_PRIMES[0]);
            }
        }

        #[test]
        fn multicol_cuda_rejects_unreserved_malformed_and_all_zero_inputs() {
            let static_data = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
            let key = pack_recoupling_key(&sample_column(1).terms[0]).unwrap();
            let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
            cuda.reserve_multicol(2, 2).unwrap();
            assert!(cuda.accumulate_reduced_multicol(&[key], &[1], 2).is_err());
            assert!(
                cuda.accumulate_reduced_multicol(&[key], &[1, 2, 3], 3)
                    .is_err()
            );
            assert!(
                cuda.accumulate_reduced_multicol(&[key], &[0, 0], 2)
                    .unwrap_err()
                    .contains("all-zero")
            );
            assert!(
                cuda.accumulate_reduced_multicol(&[key | (1_u64 << 63)], &[1, 0], 2)
                    .unwrap_err()
                    .contains("canonical")
            );
            assert!(
                cuda.accumulate_reduced_multicol(&[key, key], &[1, 0, 0, 1], 2)
                    .unwrap_err()
                    .contains("canonical")
            );

            let mut capped = CudaModularFx::new(&static_data, 0).unwrap();
            capped
                .set_recoupling_hard_cap(capped.resident_bytes())
                .unwrap();
            assert!(
                capped
                    .reserve_multicol(1, 2)
                    .unwrap_err()
                    .contains("device cap")
            );
        }

        #[test]
        #[ignore = "benchmarks exact production multi-column contraction on real 30002 prefixes"]
        fn benchmark_real_30002_multicol_production() {
            const TERMS: usize = 131_072;
            const LOCALS: [usize; 3] = [12, 13, 14];
            const STOP: &str = "real multi-column prefix complete";
            let captured = LOCALS
                .into_iter()
                .map(|local| {
                    let mut terms = Vec::with_capacity(TERMS);
                    let result = crate::eleven_dimensional_second_momentum_30001_fx::
                        visit_gpu_column_contributions(local, |term| {
                            if terms.len() == TERMS {
                                return Err(std::io::Error::other(STOP));
                            }
                            terms.push(term);
                            Ok(())
                        });
                    match result {
                        Err(error) if error.to_string() == STOP => {}
                        Err(error) => panic!("real lane {local} capture failed: {error}"),
                        Ok(_) => assert_eq!(terms.len(), TERMS),
                    }
                    terms
                })
                .collect::<Vec<_>>();
            let reduced = captured
                .iter()
                .map(|terms| {
                    let mut lane = BTreeMap::<u64, i128>::new();
                    for term in terms {
                        let entry = lane.entry(pack_recoupling_key(term).unwrap()).or_default();
                        *entry = entry.checked_add(term.coefficient).unwrap();
                    }
                    lane.retain(|_, coefficient| *coefficient != 0);
                    lane
                })
                .collect::<Vec<_>>();

            for prime in GPU_FX_PRIMES {
                let static_data = ModularFxStaticData::build(prime).unwrap();
                for width in [1_usize, 2, 3] {
                    let mut union = BTreeMap::<u64, Vec<i128>>::new();
                    for (lane, coefficients) in reduced[..width].iter().enumerate() {
                        for (&key, &coefficient) in coefficients {
                            union.entry(key).or_insert_with(|| vec![0; width])[lane] = coefficient;
                        }
                    }
                    let keys = union.keys().copied().collect::<Vec<_>>();
                    let values = union
                        .values()
                        .flat_map(|values| values.iter().copied())
                        .collect::<Vec<_>>();
                    let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
                    let mut expected = Vec::new();
                    let mut expected_expanded = Vec::new();
                    let mut sequential_contract_ms = 0_f64;
                    for (lane, terms) in captured[..width].iter().enumerate() {
                        let (rows, timing) = cuda.accumulate_terms(terms, 74 + lane).unwrap();
                        expected.push(rows);
                        expected_expanded.push(timing.expanded_contributions);
                        sequential_contract_ms += f64::from(timing.contract_milliseconds);
                    }
                    cuda.reserve_multicol(keys.len(), width).unwrap();
                    let observed = cuda
                        .accumulate_reduced_multicol(&keys, &values, width)
                        .unwrap();
                    assert_eq!(observed.columns, expected, "prime {prime} width {width}");
                    assert_eq!(
                        &observed.stats.expanded_contributions[..width],
                        expected_expanded,
                        "prime {prime} width {width}"
                    );
                    eprintln!(
                        "{}",
                        serde_json::json!({
                            "event": "multicol_production_prefix",
                            "prime": prime,
                            "active_columns": width,
                            "terms_per_column": TERMS,
                            "union_keys": keys.len(),
                            "sequential_contract_ms": sequential_contract_ms,
                            "multicol_contract_ms": observed.stats.contract_milliseconds,
                            "contract_speedup": sequential_contract_ms
                                / f64::from(observed.stats.contract_milliseconds),
                            "resident_bytes": observed.stats.resident_bytes,
                            "high_water_bytes": observed.stats.buffer_high_water_bytes,
                        })
                    );
                }
            }
        }

        #[test]
        #[ignore = "captures and benchmarks one real 131072-term 30001 profile batch"]
        fn benchmark_real_30001_batch_flat_plan_against_legacy() {
            const TERMS: usize = 131_072;
            const STOP: &str = "real CUDA F_X benchmark batch complete";
            let mut captured = Vec::with_capacity(TERMS);
            let result =
                crate::eleven_dimensional_second_momentum_30001_fx::visit_gpu_column_contributions(
                    12,
                    |term| {
                        if captured.len() == TERMS {
                            return Err(std::io::Error::other(STOP));
                        }
                        captured.push(term);
                        Ok(())
                    },
                );
            match result {
                Err(error) if error.to_string() == STOP => {}
                Err(error) => panic!("real batch capture failed: {error}"),
                Ok(_) => assert_eq!(captured.len(), TERMS),
            }
            assert_eq!(captured.len(), TERMS);
            for prime in GPU_FX_PRIMES {
                let static_data = ModularFxStaticData::build(prime).unwrap();
                let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
                cuda.set_legacy_contraction(true).unwrap();
                let (legacy_rows, legacy) = cuda.accumulate_terms(&captured, 74).unwrap();
                cuda.set_legacy_contraction(false).unwrap();
                let (planned_rows, planned) = cuda.accumulate_terms(&captured, 74).unwrap();
                assert_eq!(planned_rows, legacy_rows, "prime {prime}");
                assert_eq!(
                    planned.expanded_contributions, legacy.expanded_contributions,
                    "prime {prime}"
                );
                eprintln!(
                    "{{\"prime\":{prime},\"terms\":{TERMS},\"legacy_contract_ms\":{},\"flat_contract_ms\":{},\"speedup\":{}}}",
                    legacy.contract_milliseconds,
                    planned.contract_milliseconds,
                    f64::from(legacy.contract_milliseconds)
                        / f64::from(planned.contract_milliseconds)
                );
            }
        }

        #[test]
        #[ignore = "benchmarks a representative 131072-unique-key profile batch"]
        fn benchmark_representative_131072_flat_plan_against_legacy() {
            let input = representative_unique_column(131_072);
            let prime = GPU_FX_PRIMES[0];
            let static_data = ModularFxStaticData::build(prime).unwrap();
            let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
            cuda.set_legacy_contraction(true).unwrap();
            let (legacy, legacy_timing) = cuda.accumulate(&input).unwrap();
            cuda.set_legacy_contraction(false).unwrap();
            let (planned, planned_timing) = cuda.accumulate(&input).unwrap();
            assert_eq!(planned.rows, legacy.rows);
            assert_eq!(
                planned.expanded_contributions,
                legacy.expanded_contributions
            );
            eprintln!(
                "{{\"prime\":{prime},\"terms\":131072,\"unique_keys\":{},\"legacy_contract_ms\":{},\"flat_contract_ms\":{},\"speedup\":{}}}",
                planned_timing.keys_after_reduce,
                legacy_timing.contract_milliseconds,
                planned_timing.contract_milliseconds,
                f64::from(legacy_timing.contract_milliseconds)
                    / f64::from(planned_timing.contract_milliseconds)
            );
        }

        #[test]
        fn packed_recoupling_key_is_canonical_and_round_trips() {
            let term = &sample_column(1).terms[0];
            let key = pack_recoupling_key(term).unwrap();
            let (pair, free_spinor, exterior_mask) = unpack_recoupling_key(key).unwrap();
            assert_eq!(pair, term.momentum_pair);
            assert_eq!(free_spinor, term.free_spinor);
            assert_eq!(exterior_mask, term.exterior_mask);
            assert!(unpack_recoupling_key(key | (1_u64 << (32 + 13))).is_err());
            assert!(unpack_recoupling_key(u64::MAX).is_err());
        }

        #[test]
        fn cuda_exactly_reduces_duplicates_cancellation_and_i128_values() {
            let static_data = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
            let canonical = sample_column(23);
            let mut duplicate = canonical.clone();
            duplicate.terms.clear();
            for term in &canonical.terms {
                let mut positive = *term;
                positive.coefficient = positive.coefficient.checked_add(1_i128 << 100).unwrap();
                let mut negative = *term;
                negative.coefficient = -(1_i128 << 100);
                duplicate.terms.extend([positive, negative]);
            }
            // Add an exactly cancelling semantic key that should disappear
            // before the fused contraction.
            let mut cancel = RecoupledSourceTerm {
                momentum_pair: [10, 10],
                free_spinor: 31,
                exterior_mask: 0xfff0_0000,
                coefficient: 0,
            };
            assert!(
                canonical
                    .terms
                    .iter()
                    .all(|term| pack_recoupling_key(term).unwrap()
                        != pack_recoupling_key(&cancel).unwrap())
            );
            cancel.coefficient = 123_456_789;
            let mut cancel_negative = cancel;
            cancel_negative.coefficient = -cancel.coefficient;
            duplicate.terms.extend([cancel, cancel_negative]);

            let expected = accumulate_column_cpu(&static_data, &canonical).unwrap();
            let mut cuda = CudaModularFx::new(&static_data, 0).unwrap();
            let (observed, timing) = cuda.accumulate(&duplicate).unwrap();
            assert_eq!(observed.rows, expected.rows);
            assert_eq!(timing.source_terms, duplicate.terms.len());
            assert_eq!(
                timing.nonzero_terms_after_reduce,
                canonical.terms.len() as u64
            );
            assert!(timing.keys_after_reduce >= timing.nonzero_terms_after_reduce);

            let mut overflow = canonical.clone();
            overflow.terms = vec![canonical.terms[0], canonical.terms[0]];
            overflow.terms[0].coefficient = i128::MAX;
            overflow.terms[1].coefficient = 1;
            assert!(cuda.accumulate(&overflow).is_err());
        }

        #[test]
        fn cuda_streaming_batches_match_cpu_across_boundary_duplicates() {
            let static_data = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
            let mut input = sample_column(8);
            let mut positive = input.terms[0];
            positive.coefficient = 91_337;
            let mut negative = positive;
            negative.coefficient = -positive.coefficient;
            input.terms.insert(2, positive);
            input.terms.insert(3, negative);
            let expected = accumulate_column_cpu(&static_data, &input).unwrap();
            let metadata = GpuFxColumnMetadata::from(&input);

            let run = |batch_terms| {
                let cuda = CudaModularFx::new(&static_data, 0).unwrap();
                let mut stream = CudaStreamingColumnAccumulator::new(
                    cuda,
                    CudaStreamingConfig {
                        batch_terms,
                        host_hard_cap_bytes: 64 * 1024 * 1024,
                        device_hard_cap_bytes: 256 * 1024 * 1024,
                    },
                )
                .unwrap();
                for term in &input.terms {
                    stream.push(*term).unwrap();
                }
                stream.finalize(&metadata).unwrap()
            };
            let (observed3, timing3, source3) = run(3);
            let (observed4, timing4, source4) = run(4);
            assert_eq!(observed3.rows, expected.rows);
            assert_eq!(observed4.rows, expected.rows);
            assert_eq!(observed3.semantic_sha256, expected.semantic_sha256);
            assert_eq!(observed4.semantic_sha256, expected.semantic_sha256);
            assert_eq!(timing3.source_terms, input.terms.len());
            assert_eq!(timing3.batches, 4);
            assert_eq!(timing4.batches, 3);
            assert_eq!(source3, source4);
            assert_eq!(timing3.packed_input_sha256, timing4.packed_input_sha256);
            assert_ne!(
                timing3.expanded_contributions, timing4.expanded_contributions,
                "fixture must exercise a batch-dependent execution counter"
            );
            let binary3 = encode_modular_column(
                &observed3,
                static_data.semantic_sha256(),
                &source3,
                timing3.source_terms as u64,
            );
            let binary4 = encode_modular_column(
                &observed4,
                static_data.semantic_sha256(),
                &source4,
                timing4.source_terms as u64,
            );
            assert_eq!(binary3, binary4);
        }

        #[test]
        fn cuda_streaming_caps_fail_before_batch_allocation() {
            let static_data = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
            let cuda = CudaModularFx::new(&static_data, 0).unwrap();
            assert!(
                CudaStreamingColumnAccumulator::new(
                    cuda,
                    CudaStreamingConfig {
                        batch_terms: 1_000_000,
                        host_hard_cap_bytes: 1,
                        device_hard_cap_bytes: 256 * 1024 * 1024,
                    },
                )
                .is_err()
            );

            let required = sparse_lowering_host_bytes(250_000_000).unwrap();
            assert!(required > DEFAULT_STREAM_HOST_HARD_CAP_BYTES);
        }

        #[test]
        fn cuda_sparse_lowering_matches_exact_cpu_reference() {
            let entries = vec![
                (((3_u64) << 32) | 0x0000_0fff, -17),
                (((3_u64) << 32) | 0x0000_1ffe, 9),
                (((9_u64) << 32) | 0x0000_0fff, 4),
            ];
            for root in 0..5 {
                let expected = cpu_lower_for_test(&entries, root);
                let (observed, milliseconds) = lower_sparse_exact(&entries, root, 0).unwrap();
                assert_eq!(observed, expected, "root {root}");
                assert!(milliseconds > 0.0);
            }
            assert!(lower_sparse_exact(&[(3_u64 << 32 | 3, 1)], 0, 0).is_err());
            assert!(lower_sparse_exact(&[(3_u64 << 32 | 0xfff, i64::MIN)], 0, 0).is_err());
            assert!(lower_sparse_exact(&[entries[1], entries[0]], 0, 0).is_err());
        }

        #[test]
        fn persistent_cuda_sparse_word_matches_exact_cpu_prefixes() {
            let entries = vec![
                (((3_u64) << 32) | 0x0000_0fff, -17),
                (((3_u64) << 32) | 0x0000_1ffe, 9),
                (((9_u64) << 32) | 0x0000_0fff, 4),
            ];
            for roots in [&[0_usize][..], &[4][..], &[0, 1][..], &[4, 3][..]] {
                let mut expected = entries.clone();
                for &root in roots {
                    expected = cpu_lower_for_test(&expected, root);
                }
                assert!(!expected.is_empty(), "fixture root word {roots:?}");
                let (observed, telemetry) = lower_sparse_word_exact(&entries, roots, 0).unwrap();
                assert_eq!(observed, expected, "root word {roots:?}");
                assert_eq!(telemetry.len(), roots.len());
                for (step, stats) in telemetry.iter().enumerate() {
                    assert!(stats.input_count > 0, "step {step}");
                    assert!(stats.expanded_count >= stats.output_count, "step {step}");
                    assert!(stats.reduced_count >= stats.output_count, "step {step}");
                    assert!(stats.scratch_high_water_bytes > 0, "step {step}");
                    assert_eq!(
                        stats.immutable_handle_bytes,
                        stats.output_count * std::mem::size_of::<CudaSparseEntry>() as u64,
                        "step {step}"
                    );
                    assert!(stats.total_milliseconds > 0.0, "step {step}");
                }
            }
        }

        #[test]
        fn persistent_cuda_sparse_caps_and_deferred_context_destroy_are_safe() {
            assert!(PersistentSparseContext::new(0, 1).is_err());
            let entries = vec![CudaSparseEntry {
                key: (3_u64 << 32) | 0x0000_0fff,
                value: 7,
            }];
            let tiny = PersistentSparseContext::new(0, 24).unwrap();
            assert!(tiny.resident_bytes() <= 24);
            assert!(tiny.upload(&entries).is_err());
            drop(tiny);

            let context = PersistentSparseContext::new(0, 1024 * 1024).unwrap();
            let handle = context.upload(&entries).unwrap();
            assert_eq!(handle.maximum_absolute_coefficient(), 7);
            let mut ranged = Vec::new();
            assert_eq!(
                context
                    .visit_download(&handle, 1, |key, value| {
                        ranged.push((key, value));
                        Ok(())
                    })
                    .unwrap(),
                1
            );
            assert_eq!(ranged, vec![(entries[0].key, entries[0].value)]);
            let other = PersistentSparseContext::new(0, 1024 * 1024).unwrap();
            let mut wrong_owner_stats = CudaSparseLoweringStats::default();
            let mut error = [0_i8; ERROR_CAPACITY];
            let _other_guard = other.owner.lock().unwrap();
            let wrong_owner = unsafe {
                adynkra_fx_cuda_sparse_handle_lower(
                    other.owner.raw.as_ptr(),
                    handle.raw.as_ptr(),
                    0,
                    &mut wrong_owner_stats,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            assert!(wrong_owner.is_null());
            assert!(error_string(&error).contains("invalid persistent sparse lowering input"));
            drop(_other_guard);
            drop(other);
            // Arc ownership keeps the context alive until its final immutable
            // handle is released, preventing a dangling raw owner pointer.
            drop(context);
            drop(handle);
        }

        #[test]
        fn persistent_cuda_sparse_zero_state_and_coefficient_bound_are_exact() {
            let odd_mask = (0..12).fold(0_u32, |mask, bit| mask | (1_u32 << (2 * bit + 1)));
            let zero_input = vec![((1_u64 << 32) | u64::from(odd_mask), 7)];
            let (zero, telemetry) = lower_sparse_word_exact(&zero_input, &[4, 3], 0).unwrap();
            assert!(zero.is_empty());
            assert_eq!(telemetry.len(), 2);
            assert_eq!(telemetry[0].output_count, 0);
            assert_eq!(telemetry[1].input_count, 0);

            let too_large = vec![((3_u64 << 32) | 0x0000_0fff, i64::MAX / 13 + 1)];
            assert!(
                lower_sparse_word_exact(&too_large, &[0], 0)
                    .unwrap_err()
                    .contains("coefficient bound")
            );
        }

        #[test]
        #[ignore = "benchmarks one million canonical degree-12 entries"]
        fn benchmark_persistent_cuda_sparse_million_entry_root() {
            const COUNT: usize = 1_000_000;
            let mut entries = Vec::with_capacity(COUNT);
            let mut mask = (1_u32 << 12) - 1;
            for ordinal in 0..COUNT {
                entries.push((
                    (3_u64 << 32) | u64::from(mask),
                    ordinal as i64 % 1_000_003 + 1,
                ));
                let low = mask & mask.wrapping_neg();
                let ripple = mask.wrapping_add(low);
                mask = ripple | (((mask ^ ripple) >> 2) / low);
            }
            let legacy_started = Instant::now();
            let (legacy, legacy_kernel_ms) = lower_sparse_exact(&entries, 4, 0).unwrap();
            let legacy_wall_ms = legacy_started.elapsed().as_secs_f64() * 1_000.0;
            let persistent_started = Instant::now();
            let (persistent, telemetry) = lower_sparse_word_exact(&entries, &[4], 0).unwrap();
            let persistent_wall_ms = persistent_started.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(persistent, legacy);
            let stats = telemetry[0];
            eprintln!(
                "{{\"entries\":{COUNT},\"output_entries\":{},\"legacy_kernel_ms\":{legacy_kernel_ms},\"legacy_wall_ms\":{legacy_wall_ms},\"persistent_gpu_ms\":{},\"persistent_wall_ms\":{persistent_wall_ms},\"scratch_high_water_bytes\":{},\"handle_bytes\":{}}}",
                persistent.len(),
                stats.total_milliseconds,
                stats.scratch_high_water_bytes,
                stats.immutable_handle_bytes
            );
        }

        #[test]
        #[ignore = "runs one real 30001 word through CPU and persistent CUDA lowering"]
        fn persistent_cuda_real_30001_selected_word_matches_cpu() {
            use crate::eleven_dimensional_second_momentum_30001_fx::SecondMomentum30001GpuColumnEvent;

            const LOCAL_ORDINAL: usize = 12;
            const WORD_ORDINAL: usize = 1;
            const STOP: &str = "real persistent prefix complete";
            fn hash_term(hash: &mut Sha256, term: RecoupledSourceTerm) {
                hash.update(term.momentum_pair);
                hash.update([term.free_spinor]);
                hash.update(term.exterior_mask.to_le_bytes());
                hash.update(term.coefficient.to_le_bytes());
            }

            let preflight =
                crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
                    LOCAL_ORDINAL,
                )
                .unwrap();
            let cpu_started = Instant::now();
            let mut cpu_hash = Sha256::new();
            let mut cpu_terms = 0_u64;
            let cpu_result = crate::eleven_dimensional_second_momentum_30001_fx::
                visit_gpu_column_contribution_events_from(&preflight, WORD_ORDINAL, |event| match event {
                    SecondMomentum30001GpuColumnEvent::Term { term, .. } => {
                        hash_term(&mut cpu_hash, term);
                        cpu_terms += 1;
                        Ok(())
                    }
                    SecondMomentum30001GpuColumnEvent::WordEnd {
                        requested_word_ordinal: WORD_ORDINAL,
                        ..
                    } => Err(std::io::Error::other(STOP)),
                    _ => Ok(()),
                });
            assert_eq!(cpu_result.unwrap_err().to_string(), STOP);
            let cpu_milliseconds = cpu_started.elapsed().as_secs_f64() * 1_000.0;
            let cpu_digest = format!("{:x}", cpu_hash.finalize());

            let context = PersistentSparseContext::new(0, 2 * 1024 * 1024 * 1024).unwrap();
            let current_word = Cell::new(None::<usize>);
            let mut telemetry = Vec::new();
            let mut gpu_hash = Sha256::new();
            let mut gpu_terms = 0_u64;
            let gpu_started = Instant::now();
            let gpu_result = crate::eleven_dimensional_second_momentum_30001_fx::
                visit_gpu_column_contribution_events_from_handles(
                    &preflight,
                    WORD_ORDINAL,
                    |highest| upload_canonical_highest(&context, highest, 256 * 1024 * 1024),
                    |source, roots, observed_maximum| {
                        let mut owned = None;
                        for &simple_root in roots {
                            let base = owned.as_ref().unwrap_or(source);
                            let (next, stats) = context
                                .lower(base, usize::from(simple_root - 1))
                                .map_err(std::io::Error::other)?;
                            *observed_maximum = (*observed_maximum)
                                .max(i128::from(next.maximum_absolute_coefficient()));
                            telemetry.push((
                                current_word.get().unwrap(),
                                usize::from(simple_root),
                                stats,
                                next.maximum_absolute_coefficient(),
                            ));
                            owned = Some(next);
                        }
                        owned.ok_or_else(|| std::io::Error::other("empty real PBW segment"))
                    },
                    |handle, visit| {
                        context
                            .visit_download(handle, 65_536, |key, value| {
                                visit(key, value).map_err(|error| error.to_string())
                            })
                            .map_err(std::io::Error::other)
                    },
                    |event| match event {
                        SecondMomentum30001GpuColumnEvent::WordLoweringStart {
                            requested_word_ordinal,
                            ..
                        } => {
                            current_word.set(Some(requested_word_ordinal));
                            Ok(())
                        }
                        SecondMomentum30001GpuColumnEvent::Term { term, .. } => {
                            hash_term(&mut gpu_hash, term);
                            gpu_terms += 1;
                            Ok(())
                        }
                        SecondMomentum30001GpuColumnEvent::WordEnd {
                            requested_word_ordinal: WORD_ORDINAL,
                            ..
                        } => Err(std::io::Error::other(STOP)),
                        _ => Ok(()),
                    },
                );
            assert_eq!(gpu_result.unwrap_err().to_string(), STOP);
            let gpu_wall_milliseconds = gpu_started.elapsed().as_secs_f64() * 1_000.0;
            let gpu_digest = format!("{:x}", gpu_hash.finalize());
            assert_eq!(gpu_terms, cpu_terms);
            assert_eq!(gpu_digest, cpu_digest);
            assert!(!telemetry.is_empty());
            let gpu_stage_milliseconds = telemetry
                .iter()
                .map(|(_, _, stats, _)| f64::from(stats.total_milliseconds))
                .sum::<f64>();
            let high_water_bytes = telemetry
                .iter()
                .map(|(_, _, stats, _)| stats.scratch_high_water_bytes)
                .max()
                .unwrap();
            let maximum = telemetry
                .iter()
                .map(|(_, _, _, maximum)| *maximum)
                .max()
                .unwrap();
            let input_entries = telemetry[0].2.input_count;
            let output_entries = telemetry.last().unwrap().2.output_count;
            eprintln!(
                "{{\"tranche\":\"30001\",\"local_ordinal\":{LOCAL_ORDINAL},\"word_ordinal\":{WORD_ORDINAL},\"terms\":{gpu_terms},\"roots\":{},\"input_entries\":{input_entries},\"output_entries\":{output_entries},\"cpu_ms\":{cpu_milliseconds},\"persistent_gpu_ms\":{gpu_stage_milliseconds},\"persistent_wall_ms\":{gpu_wall_milliseconds},\"high_water_bytes\":{high_water_bytes},\"maximum_absolute_coefficient\":\"{maximum}\",\"digest\":\"{gpu_digest}\"}}",
                telemetry.len()
            );
        }

        #[test]
        #[ignore = "loads the real 30001 reciprocal certificate and PBW plan"]
        fn real_30001_zero_reciprocal_words_are_pruned_from_gpu_plan() {
            const LOCAL_ORDINAL: usize = 0;
            const PREVIOUS_UNFILTERED_WORD_COUNT: usize = 484;
            const EXPECTED_CANONICAL_WORD_COUNT: usize = 483;

            let preflight =
                crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
                    LOCAL_ORDINAL,
                )
                .unwrap();
            assert_eq!(preflight.pbw_word_count, EXPECTED_CANONICAL_WORD_COUNT);
            assert!(preflight.pbw_word_count < PREVIOUS_UNFILTERED_WORD_COUNT);
        }

        fn cpu_lower_for_test(entries: &[(u64, i64)], root: usize) -> Vec<(u64, i64)> {
            fn lower(index: u32, root: usize) -> Option<u32> {
                if root < 4 {
                    let left = 1_u32 << (4 - root);
                    let right = 1_u32 << (3 - root);
                    (index & left == 0 && index & right != 0).then_some(index ^ left ^ right)
                } else {
                    (index & 1 == 0).then_some(index | 1)
                }
            }
            let mut output = BTreeMap::<u64, i64>::new();
            for &(key, coefficient) in entries {
                let free = (key >> 32) as u32;
                let mask = key as u32;
                if let Some(next) = lower(free, root) {
                    *output
                        .entry((u64::from(next) << 32) | u64::from(mask))
                        .or_default() += coefficient;
                }
                let mut occupied = mask;
                while occupied != 0 {
                    let upper = occupied.trailing_zeros();
                    occupied &= occupied - 1;
                    let Some(next) = lower(upper, root) else {
                        continue;
                    };
                    if mask & (1_u32 << next) != 0 {
                        continue;
                    }
                    let low = upper.min(next);
                    let high = upper.max(next);
                    let interval = if high == low + 1 {
                        0
                    } else {
                        ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
                    };
                    let sign = if (mask & interval).count_ones() % 2 == 0 {
                        1
                    } else {
                        -1
                    };
                    let next_mask = mask ^ (1_u32 << upper) ^ (1_u32 << next);
                    *output
                        .entry((u64::from(free) << 32) | u64::from(next_mask))
                        .or_default() += coefficient * sign;
                }
            }
            output.into_iter().filter(|entry| entry.1 != 0).collect()
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda_backend::{
    CudaModularFx, CudaModularP3, CudaModularP3ThreePrime, CudaMultiColumnBatch,
    CudaMultiColumnStats, CudaP3ThreePrimeTiming, CudaP3Timing, lower_sparse_exact,
};

#[cfg(feature = "cuda")]
const PERSISTENT_GROUP_DEVICE_HEADROOM_BYTES: u64 = 64 * 1024 * 1024;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct PersistentGroupDeviceBudget {
    pub aggregate_hard_cap_bytes: u64,
    pub contraction_hard_cap_bytes: u64,
    pub shared_lowering_hard_cap_bytes: u64,
    pub active_lanes: usize,
    pub reserved_headroom_bytes: u64,
}

#[cfg(feature = "cuda")]
impl PersistentGroupDeviceBudget {
    fn partition(
        aggregate_hard_cap_bytes: u64,
        contraction_hard_cap_bytes: u64,
        active_lanes: usize,
    ) -> Result<Self, String> {
        if active_lanes == 0 || active_lanes > 3 || contraction_hard_cap_bytes == 0 {
            return Err("invalid persistent group device budget shape".to_string());
        }
        let lowering_total = aggregate_hard_cap_bytes
            .checked_sub(PERSISTENT_GROUP_DEVICE_HEADROOM_BYTES)
            .and_then(|bytes| bytes.checked_sub(contraction_hard_cap_bytes))
            .ok_or_else(|| {
                "aggregate CUDA cap cannot cover contraction plus 64 MiB headroom".to_string()
            })?;
        if lowering_total < 4 * std::mem::size_of::<u32>() as u64 + 8 {
            return Err("aggregate CUDA cap leaves no usable shared lowering budget".to_string());
        }
        let reserved = contraction_hard_cap_bytes
            .checked_add(lowering_total)
            .and_then(|bytes| bytes.checked_add(PERSISTENT_GROUP_DEVICE_HEADROOM_BYTES))
            .ok_or_else(|| "persistent aggregate device budget overflow".to_string())?;
        if reserved > aggregate_hard_cap_bytes {
            return Err("persistent aggregate device budget exceeds its hard cap".to_string());
        }
        Ok(Self {
            aggregate_hard_cap_bytes,
            contraction_hard_cap_bytes,
            shared_lowering_hard_cap_bytes: lowering_total,
            active_lanes,
            reserved_headroom_bytes: PERSISTENT_GROUP_DEVICE_HEADROOM_BYTES,
        })
    }
}

/// Owns the contraction context and one serialized persistent lowering
/// context shared by all group lanes under one checked device allocation budget.
#[cfg(feature = "cuda")]
pub(crate) struct PersistentCudaGroupExecutor {
    contraction: crate::second_momentum_gpu_group::CudaGroupBatchExecutor,
    lanes: cuda_backend::PersistentGroupLaneAdapter,
    budget: PersistentGroupDeviceBudget,
}

#[cfg(feature = "cuda")]
impl PersistentCudaGroupExecutor {
    pub(crate) fn new(
        plan: crate::second_momentum_gpu_group::PreparedColumnGroup,
        static_data: &ModularFxStaticData,
        device: i32,
        max_union_keys: usize,
        aggregate_device_hard_cap_bytes: u64,
        contraction_device_hard_cap_bytes: u64,
        per_lane_host_staging_cap_bytes: u64,
        download_chunk_terms: usize,
    ) -> Result<Self, String> {
        Self::new_with_full_map_directory(
            plan,
            static_data,
            device,
            max_union_keys,
            aggregate_device_hard_cap_bytes,
            contraction_device_hard_cap_bytes,
            per_lane_host_staging_cap_bytes,
            download_chunk_terms,
            None,
        )
    }

    pub(crate) fn new_full(
        plan: crate::second_momentum_gpu_group::PreparedColumnGroup,
        static_data: &ModularFxStaticData,
        device: i32,
        max_union_keys: usize,
        aggregate_device_hard_cap_bytes: u64,
        contraction_device_hard_cap_bytes: u64,
        per_lane_host_staging_cap_bytes: u64,
        download_chunk_terms: usize,
        map_directory: &std::path::Path,
    ) -> Result<Self, String> {
        Self::new_with_full_map_directory(
            plan,
            static_data,
            device,
            max_union_keys,
            aggregate_device_hard_cap_bytes,
            contraction_device_hard_cap_bytes,
            per_lane_host_staging_cap_bytes,
            download_chunk_terms,
            Some(map_directory),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_full_map_directory(
        plan: crate::second_momentum_gpu_group::PreparedColumnGroup,
        static_data: &ModularFxStaticData,
        device: i32,
        max_union_keys: usize,
        aggregate_device_hard_cap_bytes: u64,
        contraction_device_hard_cap_bytes: u64,
        per_lane_host_staging_cap_bytes: u64,
        download_chunk_terms: usize,
        map_directory: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let budget = PersistentGroupDeviceBudget::partition(
            aggregate_device_hard_cap_bytes,
            contraction_device_hard_cap_bytes,
            plan.active_columns,
        )?;
        let contraction = crate::second_momentum_gpu_group::CudaGroupBatchExecutor::new(
            plan.clone(),
            static_data,
            device,
            max_union_keys,
            budget.contraction_hard_cap_bytes,
        )?;
        let lanes = cuda_backend::PersistentGroupLaneAdapter::new(
            &plan,
            device,
            budget.shared_lowering_hard_cap_bytes,
            per_lane_host_staging_cap_bytes,
            download_chunk_terms,
            map_directory,
        )?;
        Ok(Self {
            contraction,
            lanes,
            budget,
        })
    }

    pub(crate) fn run_word_synchronous<O, B, W>(
        &mut self,
        config: crate::second_momentum_gpu_group::GroupWordOrchestrationConfig,
        observe_raw_term: O,
        observe_batch: B,
        complete_word: W,
    ) -> Result<crate::second_momentum_gpu_group::GroupWordOrchestrationReport, String>
    where
        O: FnMut(usize, usize, &RecoupledSourceTerm) -> Result<(), String>,
        B: FnMut(&crate::second_momentum_gpu_group::GroupBatchObservation) -> Result<(), String>,
        W: FnMut(
            usize,
            &[crate::second_momentum_gpu_group::LaneWordCompletion],
        ) -> Result<(), String>,
    {
        let lanes = &self.lanes;
        self.contraction.run_word_synchronous(
            config,
            &|lane_index,
              expected_local_ordinal,
              expected_global_ordinal,
              expected_source_copy,
              word_ordinal,
              emit_term| {
                lanes.run_lane_word(
                    lane_index,
                    expected_local_ordinal,
                    expected_global_ordinal,
                    expected_source_copy,
                    word_ordinal,
                    emit_term,
                )
            },
            observe_raw_term,
            observe_batch,
            complete_word,
        )
    }

    pub(crate) fn run_word_synchronous_batched<O, B, W>(
        &mut self,
        config: crate::second_momentum_gpu_group::GroupWordOrchestrationConfig,
        observe_raw_batch: O,
        observe_batch: B,
        complete_word: W,
    ) -> Result<crate::second_momentum_gpu_group::GroupWordOrchestrationReport, String>
    where
        O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
        B: FnMut(&crate::second_momentum_gpu_group::GroupBatchObservation) -> Result<(), String>,
        W: FnMut(
            usize,
            &[crate::second_momentum_gpu_group::LaneWordCompletion],
        ) -> Result<(), String>,
    {
        let lanes = &self.lanes;
        self.contraction.run_word_synchronous_batched(
            config,
            &|lane_index,
              expected_local_ordinal,
              expected_global_ordinal,
              expected_source_copy,
              word_ordinal,
              emit_term| {
                lanes.run_lane_word(
                    lane_index,
                    expected_local_ordinal,
                    expected_global_ordinal,
                    expected_source_copy,
                    word_ordinal,
                    emit_term,
                )
            },
            observe_raw_batch,
            observe_batch,
            complete_word,
        )
    }

    pub(crate) fn run_word_synchronous_batched_with_union<O, U, B, W>(
        &mut self,
        config: crate::second_momentum_gpu_group::GroupWordOrchestrationConfig,
        observe_raw_batch: O,
        observe_union: U,
        observe_batch: B,
        complete_word: W,
    ) -> Result<crate::second_momentum_gpu_group::GroupWordOrchestrationReport, String>
    where
        O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
        U: FnMut(usize, &crate::second_momentum_gpu_group::ExactUnionBatch) -> Result<(), String>,
        B: FnMut(&crate::second_momentum_gpu_group::GroupBatchObservation) -> Result<(), String>,
        W: FnMut(
            usize,
            &[crate::second_momentum_gpu_group::LaneWordCompletion],
        ) -> Result<(), String>,
    {
        let lanes = &self.lanes;
        self.contraction.run_word_synchronous_batched_with_union(
            config,
            &|lane_index,
              expected_local_ordinal,
              expected_global_ordinal,
              expected_source_copy,
              word_ordinal,
              emit_term| {
                lanes.run_lane_word(
                    lane_index,
                    expected_local_ordinal,
                    expected_global_ordinal,
                    expected_source_copy,
                    word_ordinal,
                    emit_term,
                )
            },
            observe_raw_batch,
            observe_union,
            observe_batch,
            complete_word,
        )
    }

    pub(crate) const fn device_budget(&self) -> PersistentGroupDeviceBudget {
        self.budget
    }

    pub(crate) fn lowering_summaries(
        &self,
    ) -> Result<Vec<cuda_backend::PersistentLoweringSummary>, String> {
        self.lanes.summaries()
    }

    pub(crate) fn final_columns(&self) -> &[Vec<GaussianResidue>] {
        self.contraction.final_columns()
    }

    pub(crate) fn final_column_semantic_sha256(&self) -> Vec<String> {
        self.contraction.final_column_semantic_sha256()
    }

    pub(crate) fn restore_columns(
        &mut self,
        columns: Vec<Vec<GaussianResidue>>,
        batches_folded: u64,
    ) -> Result<(), String> {
        self.contraction.restore_columns(columns, batches_folded)
    }

    pub(crate) const fn batches_folded(&self) -> u64 {
        self.contraction.batches_folded()
    }

    pub(crate) fn collect_parity_prefix(
        &self,
        maximum_terms_per_lane: usize,
    ) -> Result<Vec<Vec<RecoupledSourceTerm>>, String> {
        self.lanes.collect_parity_prefix(maximum_terms_per_lane)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PersistentMultiPrimeDeviceBudget {
    pub aggregate_hard_cap_bytes: u64,
    pub initial_per_prime_contraction_hard_cap_bytes: u64,
    pub contraction_resident_bytes_by_prime: Vec<u64>,
    pub total_contraction_resident_bytes: u64,
    pub shared_lowering_hard_cap_bytes: u64,
    pub active_lanes: usize,
    pub active_primes: usize,
    pub reserved_headroom_bytes: u64,
}

#[cfg(feature = "cuda")]
impl PersistentMultiPrimeDeviceBudget {
    fn partition_after_reserve(
        aggregate_hard_cap_bytes: u64,
        initial_per_prime_contraction_hard_cap_bytes: u64,
        contraction_resident_bytes_by_prime: Vec<u64>,
        active_lanes: usize,
        active_primes: usize,
    ) -> Result<Self, String> {
        if active_lanes == 0
            || active_lanes > 3
            || active_primes == 0
            || active_primes > GPU_FX_PRIMES.len()
            || initial_per_prime_contraction_hard_cap_bytes == 0
            || contraction_resident_bytes_by_prime.len() != active_primes
            || contraction_resident_bytes_by_prime.contains(&0)
        {
            return Err("invalid persistent multi-prime device budget shape".to_string());
        }
        let total_contraction_resident_bytes = contraction_resident_bytes_by_prime
            .iter()
            .try_fold(0_u64, |sum, bytes| sum.checked_add(*bytes))
            .ok_or_else(|| "multi-prime contraction resident-byte overflow".to_string())?;
        let shared_lowering_hard_cap_bytes = aggregate_hard_cap_bytes
            .checked_sub(PERSISTENT_GROUP_DEVICE_HEADROOM_BYTES)
            .and_then(|bytes| bytes.checked_sub(total_contraction_resident_bytes))
            .ok_or_else(|| {
                "aggregate CUDA cap cannot cover every prime plus 64 MiB headroom".to_string()
            })?;
        if shared_lowering_hard_cap_bytes < 4 * std::mem::size_of::<u32>() as u64 + 8 {
            return Err("aggregate CUDA cap leaves no usable shared lowering budget".to_string());
        }
        Ok(Self {
            aggregate_hard_cap_bytes,
            initial_per_prime_contraction_hard_cap_bytes,
            contraction_resident_bytes_by_prime,
            total_contraction_resident_bytes,
            shared_lowering_hard_cap_bytes,
            active_lanes,
            active_primes,
            reserved_headroom_bytes: PERSISTENT_GROUP_DEVICE_HEADROOM_BYTES,
        })
    }
}

/// Shares one exact PBW traversal, raw hash stream, lane reduction, and union
/// batch across several independent prime-specific CUDA contraction contexts.
/// No exact union payload is cloned between primes.
#[cfg(feature = "cuda")]
pub(crate) struct PersistentCudaMultiPrimeGroupExecutor {
    source_plan: crate::second_momentum_gpu_group::PreparedColumnGroup,
    bundle_group_id: String,
    contractions: Vec<crate::second_momentum_gpu_group::CudaGroupBatchExecutor>,
    lanes: cuda_backend::PersistentGroupLaneAdapter,
    budget: PersistentMultiPrimeDeviceBudget,
}

#[cfg(feature = "cuda")]
impl PersistentCudaMultiPrimeGroupExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plans: Vec<crate::second_momentum_gpu_group::PreparedColumnGroup>,
        static_data: &[ModularFxStaticData],
        device: i32,
        max_union_keys: usize,
        aggregate_device_hard_cap_bytes: u64,
        per_prime_contraction_hard_cap_bytes: u64,
        per_lane_host_staging_cap_bytes: u64,
        download_chunk_terms: usize,
        map_directory: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        if plans.len() != static_data.len() || plans.is_empty() {
            return Err("multi-prime plans and static data have different shapes".to_string());
        }
        let bundle_group_id =
            crate::second_momentum_gpu_group::multi_prime_group_identity_sha256(&plans)?;
        let active_lanes = plans[0].active_columns;
        let mut contractions = Vec::with_capacity(plans.len());
        for (plan, data) in plans.iter().cloned().zip(static_data) {
            contractions.push(
                crate::second_momentum_gpu_group::CudaGroupBatchExecutor::new_for_batch_group(
                    plan,
                    bundle_group_id.clone(),
                    data,
                    device,
                    max_union_keys,
                    per_prime_contraction_hard_cap_bytes,
                )?,
            );
        }
        let contraction_resident_bytes_by_prime = contractions
            .iter_mut()
            .map(|contraction| contraction.tighten_device_hard_cap_to_resident())
            .collect::<Result<Vec<_>, _>>()?;
        let budget = PersistentMultiPrimeDeviceBudget::partition_after_reserve(
            aggregate_device_hard_cap_bytes,
            per_prime_contraction_hard_cap_bytes,
            contraction_resident_bytes_by_prime,
            active_lanes,
            plans.len(),
        )?;
        let source_plan = plans[0].clone();
        let lanes = cuda_backend::PersistentGroupLaneAdapter::new(
            &source_plan,
            device,
            budget.shared_lowering_hard_cap_bytes,
            per_lane_host_staging_cap_bytes,
            download_chunk_terms,
            map_directory,
        )?;
        Ok(Self {
            source_plan,
            bundle_group_id,
            contractions,
            lanes,
            budget,
        })
    }

    pub(crate) fn run_word_synchronous<O, B, W>(
        &mut self,
        config: crate::second_momentum_gpu_group::GroupWordOrchestrationConfig,
        observe_raw_term: O,
        mut observe_batch: B,
        complete_word: W,
    ) -> Result<crate::second_momentum_gpu_group::GroupWordOrchestrationReport, String>
    where
        O: FnMut(usize, usize, &RecoupledSourceTerm) -> Result<(), String>,
        B: FnMut(
            usize,
            u32,
            &crate::second_momentum_gpu_group::GroupBatchObservation,
        ) -> Result<(), String>,
        W: FnMut(
            usize,
            &[crate::second_momentum_gpu_group::LaneWordCompletion],
        ) -> Result<(), String>,
    {
        let lanes = &self.lanes;
        let source_plan = self.source_plan.clone();
        let contractions = &mut self.contractions;
        crate::second_momentum_gpu_group::orchestrate_group_words_with_batch_group_id(
            &source_plan,
            &self.bundle_group_id,
            config,
            &|lane_index,
              expected_local_ordinal,
              expected_global_ordinal,
              expected_source_copy,
              word_ordinal,
              emit_term| {
                lanes.run_lane_word(
                    lane_index,
                    expected_local_ordinal,
                    expected_global_ordinal,
                    expected_source_copy,
                    word_ordinal,
                    emit_term,
                )
            },
            observe_raw_term,
            |word_ordinal, raw_counts, batch| {
                for (prime_slot, contraction) in contractions.iter_mut().enumerate() {
                    let prime = contraction.prime();
                    let observation = contraction.accumulate_batch(
                        &batch,
                        word_ordinal,
                        None,
                        raw_counts.clone(),
                    )?;
                    observe_batch(prime_slot, prime, &observation)?;
                }
                Ok(())
            },
            complete_word,
        )
    }

    pub(crate) fn run_word_synchronous_batched<O, B, W>(
        &mut self,
        config: crate::second_momentum_gpu_group::GroupWordOrchestrationConfig,
        observe_raw_batch: O,
        mut observe_batch: B,
        complete_word: W,
    ) -> Result<crate::second_momentum_gpu_group::GroupWordOrchestrationReport, String>
    where
        O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
        B: FnMut(
            usize,
            u32,
            &crate::second_momentum_gpu_group::GroupBatchObservation,
        ) -> Result<(), String>,
        W: FnMut(
            usize,
            &[crate::second_momentum_gpu_group::LaneWordCompletion],
        ) -> Result<(), String>,
    {
        let lanes = &self.lanes;
        let source_plan = self.source_plan.clone();
        let contractions = &mut self.contractions;
        crate::second_momentum_gpu_group::orchestrate_group_words_with_batch_observer(
            &source_plan,
            &self.bundle_group_id,
            config,
            &|lane_index,
              expected_local_ordinal,
              expected_global_ordinal,
              expected_source_copy,
              word_ordinal,
              emit_term| {
                lanes.run_lane_word(
                    lane_index,
                    expected_local_ordinal,
                    expected_global_ordinal,
                    expected_source_copy,
                    word_ordinal,
                    emit_term,
                )
            },
            observe_raw_batch,
            |word_ordinal, raw_counts, batch| {
                for (prime_slot, contraction) in contractions.iter_mut().enumerate() {
                    let prime = contraction.prime();
                    let observation = contraction.accumulate_batch(
                        &batch,
                        word_ordinal,
                        None,
                        raw_counts.clone(),
                    )?;
                    observe_batch(prime_slot, prime, &observation)?;
                }
                Ok(())
            },
            complete_word,
        )
    }

    pub(crate) fn run_word_synchronous_batched_with_union<O, U, B, W>(
        &mut self,
        config: crate::second_momentum_gpu_group::GroupWordOrchestrationConfig,
        observe_raw_batch: O,
        mut observe_union: U,
        mut observe_batch: B,
        complete_word: W,
    ) -> Result<crate::second_momentum_gpu_group::GroupWordOrchestrationReport, String>
    where
        O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
        U: FnMut(usize, &crate::second_momentum_gpu_group::ExactUnionBatch) -> Result<(), String>,
        B: FnMut(
            usize,
            u32,
            &crate::second_momentum_gpu_group::GroupBatchObservation,
        ) -> Result<(), String>,
        W: FnMut(
            usize,
            &[crate::second_momentum_gpu_group::LaneWordCompletion],
        ) -> Result<(), String>,
    {
        let lanes = &self.lanes;
        let source_plan = self.source_plan.clone();
        let contractions = &mut self.contractions;
        crate::second_momentum_gpu_group::orchestrate_group_words_with_batch_observer(
            &source_plan,
            &self.bundle_group_id,
            config,
            &|lane_index,
              expected_local_ordinal,
              expected_global_ordinal,
              expected_source_copy,
              word_ordinal,
              emit_term| {
                lanes.run_lane_word(
                    lane_index,
                    expected_local_ordinal,
                    expected_global_ordinal,
                    expected_source_copy,
                    word_ordinal,
                    emit_term,
                )
            },
            observe_raw_batch,
            |word_ordinal, raw_counts, batch| {
                observe_union(word_ordinal, &batch)?;
                for (prime_slot, contraction) in contractions.iter_mut().enumerate() {
                    let prime = contraction.prime();
                    let observation = contraction.accumulate_batch(
                        &batch,
                        word_ordinal,
                        None,
                        raw_counts.clone(),
                    )?;
                    observe_batch(prime_slot, prime, &observation)?;
                }
                Ok(())
            },
            complete_word,
        )
    }

    pub(crate) fn device_budget(&self) -> &PersistentMultiPrimeDeviceBudget {
        &self.budget
    }

    pub(crate) fn lowering_summaries(
        &self,
    ) -> Result<Vec<cuda_backend::PersistentLoweringSummary>, String> {
        self.lanes.summaries()
    }

    pub(crate) fn final_columns(
        &self,
        prime_slot: usize,
    ) -> Result<&[Vec<GaussianResidue>], String> {
        self.contractions
            .get(prime_slot)
            .map(crate::second_momentum_gpu_group::CudaGroupBatchExecutor::final_columns)
            .ok_or_else(|| "multi-prime column slot is out of range".to_string())
    }

    pub(crate) fn final_column_semantic_sha256(
        &self,
        prime_slot: usize,
    ) -> Result<Vec<String>, String> {
        self.contractions
            .get(prime_slot)
            .map(crate::second_momentum_gpu_group::CudaGroupBatchExecutor::final_column_semantic_sha256)
            .ok_or_else(|| "multi-prime digest slot is out of range".to_string())
    }

    pub(crate) fn restore_columns(
        &mut self,
        rows_by_prime: Vec<Vec<Vec<GaussianResidue>>>,
        batches_folded: u64,
    ) -> Result<(), String> {
        if rows_by_prime.len() != self.contractions.len() {
            return Err("multi-prime restore shape changed".to_string());
        }
        for (contraction, rows) in self.contractions.iter_mut().zip(rows_by_prime) {
            contraction.restore_columns(rows, batches_folded)?;
        }
        Ok(())
    }

    pub(crate) fn batches_folded(&self) -> Result<u64, String> {
        let first = self
            .contractions
            .first()
            .ok_or_else(|| "multi-prime executor has no contractions".to_string())?
            .batches_folded();
        if self
            .contractions
            .iter()
            .any(|contraction| contraction.batches_folded() != first)
        {
            return Err("multi-prime contraction batch counts diverged".to_string());
        }
        Ok(first)
    }

    pub(crate) fn collect_parity_prefix(
        &self,
        maximum_terms_per_lane: usize,
    ) -> Result<Vec<Vec<RecoupledSourceTerm>>, String> {
        self.lanes.collect_parity_prefix(maximum_terms_per_lane)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn run_cuda_column(
    tranche: &str,
    local_ordinal: usize,
    prime: u32,
    device: i32,
    output_directory: &Path,
    cpu_parity_terms: usize,
    live_progress: Option<&crate::second_momentum_gpu_progress::LiveProgress>,
) -> Result<GpuFxColumnReport, String> {
    if cpu_parity_terms == 0 {
        return Err(
            "CUDA F_X column certification requires a nonzero CPU parity prefix".to_string(),
        );
    }
    let started = Instant::now();
    let static_started = Instant::now();
    let static_data = ModularFxStaticData::build(prime)?;
    let static_build_milliseconds = static_started.elapsed().as_millis();
    let config = cuda_backend::CudaStreamingConfig::from_environment()?;
    if cpu_parity_terms > config.batch_terms {
        return Err(format!(
            "CPU parity prefix {cpu_parity_terms} exceeds bounded batch term cap {}",
            config.batch_terms
        ));
    }
    let host_peak_with_parity = config.required_host_bytes(cpu_parity_terms)?;
    if host_peak_with_parity > config.host_hard_cap_bytes {
        return Err(format!(
            "CUDA stream plus CPU parity prefix requires {host_peak_with_parity} host bytes, above hard cap {}",
            config.host_hard_cap_bytes
        ));
    }
    const PERSISTENT_DEVICE_HEADROOM_BYTES: u64 = 64 * 1024 * 1024;
    const MAX_PERSISTENT_DOWNLOAD_CHUNK_TERMS: usize = 65_536;
    let total_device_hard_cap_bytes = config.device_hard_cap_bytes;
    let persistent_enabled = cuda_backend::persistent_sparse_enabled()?;
    let mut persistent_device_hard_cap_bytes = 0_u64;
    let mut persistent_host_staging_cap_bytes = 0_u64;
    let mut persistent_download_chunk_terms = 0_usize;
    let mut stream_config = config;
    let mut cuda = CudaModularFx::new(&static_data, device)?;
    if persistent_enabled {
        cuda.reserve_recoupling_terms(config.batch_terms)?;
        let recoupling_resident_bytes = cuda.resident_bytes();
        persistent_device_hard_cap_bytes = total_device_hard_cap_bytes
            .checked_sub(recoupling_resident_bytes)
            .and_then(|remaining| remaining.checked_sub(PERSISTENT_DEVICE_HEADROOM_BYTES))
            .ok_or_else(|| {
                format!(
                    "combined CUDA cap {total_device_hard_cap_bytes} cannot hold reserved F_X workspace {recoupling_resident_bytes} plus {PERSISTENT_DEVICE_HEADROOM_BYTES} bytes headroom"
                )
            })?;
        stream_config.device_hard_cap_bytes = recoupling_resident_bytes;
        persistent_host_staging_cap_bytes = config
            .host_hard_cap_bytes
            .checked_sub(host_peak_with_parity)
            .ok_or_else(|| "persistent CUDA host staging budget underflow".to_string())?;
        persistent_download_chunk_terms = usize::try_from(
            persistent_host_staging_cap_bytes / std::mem::size_of::<(u64, i64)>() as u64,
        )
        .unwrap_or(usize::MAX)
        .min(MAX_PERSISTENT_DOWNLOAD_CHUNK_TERMS);
        if persistent_download_chunk_terms == 0 {
            return Err(
                "CUDA host cap leaves no bounded persistent download staging space".to_string(),
            );
        }
    }
    let mut stream = cuda_backend::CudaStreamingColumnAccumulator::new(cuda, stream_config)?;
    let device_name = stream.device_name().to_string();
    let flat_plan_sha256 = stream.flat_plan_sha256().to_string();
    let mut parity_terms = Vec::with_capacity(cpu_parity_terms);
    let mut raw_terms_emitted = 0_u64;
    let mut batches_flushed = 0_u64;
    let mut cumulative_batch_milliseconds = 0_f64;
    let mut cumulative_upload_milliseconds = 0_f64;
    let mut cumulative_sort_milliseconds = 0_f64;
    let mut cumulative_reduce_milliseconds = 0_f64;
    let mut cumulative_contract_milliseconds = 0_f64;
    let mut cumulative_download_milliseconds = 0_f64;
    let progress_raw_terms = std::cell::Cell::new(0_u64);
    let progress_batches = std::cell::Cell::new(0_u64);
    let progress_current_batch_terms = std::cell::Cell::new(0_u64);
    if let Some(progress) = live_progress {
        progress.update_source(crate::second_momentum_gpu_progress::SourceVisitorProgress {
            word: None,
            root: None,
            raw_terms_emitted: 0,
            batches_flushed: 0,
            current_batch_terms: 0,
            current_batch_bytes: 0,
            hard_memory_cap_bytes: config.host_hard_cap_bytes,
            eta_sample_count: 0,
        });
    }
    let mut consume = |term: RecoupledSourceTerm| -> std::io::Result<()> {
        if parity_terms.len() < cpu_parity_terms {
            parity_terms.push(term);
        }
        stream
            .push(term)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        raw_terms_emitted = raw_terms_emitted
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("raw GPU contribution count overflow"))?;
        let current_batch_terms = raw_terms_emitted % config.batch_terms as u64;
        progress_raw_terms.set(raw_terms_emitted);
        progress_current_batch_terms.set(current_batch_terms);
        if let Some(batch_timing) = stream.take_last_batch_timing() {
            batches_flushed = batches_flushed
                .checked_add(1)
                .ok_or_else(|| std::io::Error::other("GPU batch count overflow"))?;
            let batch_milliseconds = f64::from(batch_timing.kernel_milliseconds);
            cumulative_batch_milliseconds += batch_milliseconds;
            cumulative_upload_milliseconds += f64::from(batch_timing.upload_milliseconds);
            cumulative_sort_milliseconds += f64::from(batch_timing.sort_milliseconds);
            cumulative_reduce_milliseconds += f64::from(batch_timing.reduce_milliseconds);
            cumulative_contract_milliseconds += f64::from(batch_timing.contract_milliseconds);
            cumulative_download_milliseconds += f64::from(batch_timing.download_milliseconds);
            progress_batches.set(batches_flushed);
            if let Some(progress) = live_progress {
                progress.record_gpu_batch(crate::second_momentum_gpu_progress::GpuBatchProgress {
                    batches_completed: batches_flushed,
                    last_batch_ms: batch_milliseconds,
                    total_batch_ms: cumulative_batch_milliseconds,
                    last_upload_ms: f64::from(batch_timing.upload_milliseconds),
                    total_upload_ms: cumulative_upload_milliseconds,
                    last_sort_ms: f64::from(batch_timing.sort_milliseconds),
                    total_sort_ms: cumulative_sort_milliseconds,
                    last_reduce_ms: f64::from(batch_timing.reduce_milliseconds),
                    total_reduce_ms: cumulative_reduce_milliseconds,
                    last_contract_ms: f64::from(batch_timing.contract_milliseconds),
                    total_contract_ms: cumulative_contract_milliseconds,
                    last_download_ms: f64::from(batch_timing.download_milliseconds),
                    total_download_ms: cumulative_download_milliseconds,
                });
            }
        }
        if current_batch_terms == 0 || raw_terms_emitted == 1 {
            if let Some(progress) = live_progress {
                progress.update_source(
                    crate::second_momentum_gpu_progress::SourceVisitorProgress {
                        word: None,
                        root: None,
                        raw_terms_emitted,
                        batches_flushed,
                        current_batch_terms,
                        current_batch_bytes: current_batch_terms
                            .saturating_mul(std::mem::size_of::<RecoupledSourceTerm>() as u64),
                        hard_memory_cap_bytes: config.host_hard_cap_bytes,
                        eta_sample_count: batches_flushed,
                    },
                );
            }
        }
        Ok(())
    };
    let source_started = Instant::now();
    let (column_metadata, persistent_summary) = if persistent_enabled {
        cuda_backend::visit_persistent_column_contributions(
            tranche,
            local_ordinal,
            device,
            persistent_device_hard_cap_bytes,
            persistent_host_staging_cap_bytes,
            persistent_download_chunk_terms,
            &mut consume,
            |root_progress| {
                if let Some(progress) = live_progress {
                    let current_batch_terms = progress_current_batch_terms.get();
                    eprintln!(
                        "{}",
                        serde_json::json!({
                            "event": "gpu_persistent_root",
                            "phase": root_progress.phase,
                            "word_ordinal": root_progress.word_ordinal,
                            "simple_root": root_progress.root,
                            "input_entries": root_progress.stats.input_count,
                            "expanded_entries": root_progress.stats.expanded_count,
                            "reduced_entries": root_progress.stats.reduced_count,
                            "output_entries": root_progress.stats.output_count,
                            "gpu_milliseconds": root_progress.stats.total_milliseconds,
                            "resident_bytes": root_progress.resident_bytes,
                            "high_water_bytes": root_progress.stats.scratch_high_water_bytes,
                        })
                    );
                    progress.update_source(
                        crate::second_momentum_gpu_progress::SourceVisitorProgress {
                            word: Some(root_progress.word_ordinal as u64),
                            root: Some(root_progress.root as u64),
                            raw_terms_emitted: progress_raw_terms.get(),
                            batches_flushed: progress_batches.get(),
                            current_batch_terms,
                            current_batch_bytes: current_batch_terms
                                .saturating_mul(std::mem::size_of::<RecoupledSourceTerm>() as u64),
                            hard_memory_cap_bytes: config.host_hard_cap_bytes,
                            eta_sample_count: progress_batches.get(),
                        },
                    );
                }
            },
        )?
    } else {
        let metadata = match tranche {
            "20001" => crate::eleven_dimensional_second_momentum_20001_fx::
                visit_gpu_column_contributions(local_ordinal, &mut consume),
            "30001" => crate::eleven_dimensional_second_momentum_30001_fx::
                visit_gpu_column_contributions(local_ordinal, &mut consume),
            _ => {
                return Err(
                    "GPU F_X tranche must be 20001 or 30001; the four-column 10001 path uses its smaller exact direct evaluator"
                        .to_string(),
                );
            }
        }
        .map_err(|error| error.to_string())?;
        (metadata, cuda_backend::PersistentLoweringSummary::default())
    };
    drop(consume);
    let source_build_milliseconds = source_started.elapsed().as_millis();
    let metadata = GpuFxColumnMetadata::from(&column_metadata);

    let cpu_parity_terms = parity_terms.len();
    let cpu_parity_passed = if parity_terms.is_empty() {
        false
    } else {
        let parity = GpuFxColumnInput {
            global_ordinal: metadata.global_ordinal,
            source_label: metadata.source_label.clone(),
            source_copy: metadata.source_copy,
            terms: parity_terms,
            raising_residuals: metadata.raising_residuals,
        };
        let cpu = accumulate_column_cpu(&static_data, &parity)?;
        let gpu = stream.accumulate_for_parity(&parity)?;
        if cpu.rows != gpu.rows || cpu.semantic_sha256 != gpu.semantic_sha256 {
            return Err(format!(
                "CPU/CUDA parity failed on {} real source terms",
                parity.terms.len()
            ));
        }
        true
    };
    stream.flush_pending()?;
    if let Some(batch_timing) = stream.take_last_batch_timing() {
        batches_flushed = batches_flushed
            .checked_add(1)
            .ok_or_else(|| "GPU batch count overflow".to_string())?;
        let batch_milliseconds = f64::from(batch_timing.kernel_milliseconds);
        cumulative_batch_milliseconds += batch_milliseconds;
        cumulative_upload_milliseconds += f64::from(batch_timing.upload_milliseconds);
        cumulative_sort_milliseconds += f64::from(batch_timing.sort_milliseconds);
        cumulative_reduce_milliseconds += f64::from(batch_timing.reduce_milliseconds);
        cumulative_contract_milliseconds += f64::from(batch_timing.contract_milliseconds);
        cumulative_download_milliseconds += f64::from(batch_timing.download_milliseconds);
        if let Some(progress) = live_progress {
            progress.record_gpu_batch(crate::second_momentum_gpu_progress::GpuBatchProgress {
                batches_completed: batches_flushed,
                last_batch_ms: batch_milliseconds,
                total_batch_ms: cumulative_batch_milliseconds,
                last_upload_ms: f64::from(batch_timing.upload_milliseconds),
                total_upload_ms: cumulative_upload_milliseconds,
                last_sort_ms: f64::from(batch_timing.sort_milliseconds),
                total_sort_ms: cumulative_sort_milliseconds,
                last_reduce_ms: f64::from(batch_timing.reduce_milliseconds),
                total_reduce_ms: cumulative_reduce_milliseconds,
                last_contract_ms: f64::from(batch_timing.contract_milliseconds),
                total_contract_ms: cumulative_contract_milliseconds,
                last_download_ms: f64::from(batch_timing.download_milliseconds),
                total_download_ms: cumulative_download_milliseconds,
            });
        }
    }
    let (modular_column, timing, source_terms_sha256) = stream.finalize(&metadata)?;
    debug_assert_eq!(timing.batches, batches_flushed);
    if let Some(progress) = live_progress {
        progress.update_source(crate::second_momentum_gpu_progress::SourceVisitorProgress {
            word: None,
            root: None,
            raw_terms_emitted: timing.source_terms as u64,
            batches_flushed: timing.batches,
            current_batch_terms: 0,
            current_batch_bytes: 0,
            hard_memory_cap_bytes: timing.host_hard_cap_bytes,
            eta_sample_count: timing.batches,
        });
    }
    let rank = rank_columns(std::slice::from_ref(&modular_column))?;
    let nonzero_functional_rows = modular_column
        .rows
        .iter()
        .filter(|value| !value.is_zero())
        .count();
    fs::create_dir_all(output_directory).map_err(|error| error.to_string())?;
    let stem = format!(
        "second_momentum_{tranche}_column_{:02}_p{prime}",
        metadata.global_ordinal
    );
    let binary_path = output_directory.join(format!("{stem}.bin"));
    let binary = encode_modular_column(
        &modular_column,
        static_data.semantic_sha256(),
        &source_terms_sha256,
        timing.source_terms as u64,
    );
    write_atomic(&binary_path, &binary)?;
    let binary_sha256 = format!("{:x}", Sha256::digest(&binary));
    let report = GpuFxColumnReport {
        schema_version: GPU_FX_SCHEMA,
        tranche: tranche.to_string(),
        local_ordinal,
        global_ordinal: metadata.global_ordinal,
        source_label: metadata.source_label,
        source_copy: metadata.source_copy,
        prime,
        functional_seeds: GPU_FX_FUNCTIONAL_SEEDS,
        functional_row_count: FUNCTIONAL_ROW_COUNT,
        device_name,
        static_semantic_sha256: static_data.semantic_sha256().to_string(),
        flat_plan_sha256,
        source_terms: timing.source_terms,
        source_terms_sha256,
        expanded_contributions: timing.expanded_contributions,
        nonzero_functional_rows,
        column_semantic_sha256: modular_column.semantic_sha256,
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        binary_bytes: binary.len() as u64,
        source_build_milliseconds,
        static_build_milliseconds,
        cuda_kernel_milliseconds: timing.kernel_milliseconds,
        cuda_upload_milliseconds: timing.upload_milliseconds,
        cuda_sort_milliseconds: timing.sort_milliseconds,
        cuda_reduce_milliseconds: timing.reduce_milliseconds,
        cuda_contract_milliseconds: timing.contract_milliseconds,
        cuda_download_milliseconds: timing.download_milliseconds,
        batch_reduced_key_visits: timing.keys_after_reduce,
        batch_nonzero_reduced_term_visits: timing.nonzero_terms_after_reduce,
        cuda_buffer_high_water_bytes: timing.buffer_high_water_bytes,
        packed_recoupling_input_sha256: timing.packed_input_sha256,
        cuda_input_terms_per_second: timing.input_terms_per_second,
        cuda_batches: timing.batches,
        cuda_peak_batch_terms: timing.peak_batch_terms,
        cuda_batch_term_cap: timing.batch_term_cap,
        cuda_host_hard_cap_bytes: timing.host_hard_cap_bytes,
        cuda_device_hard_cap_bytes: timing.device_hard_cap_bytes,
        cuda_total_device_hard_cap_bytes: total_device_hard_cap_bytes,
        persistent_lowering_enabled: persistent_summary.enabled,
        persistent_lowering_roots: persistent_summary.roots_lowered,
        persistent_lowering_input_entry_visits: persistent_summary.input_entry_visits,
        persistent_lowering_expanded_entry_visits: persistent_summary.expanded_entry_visits,
        persistent_lowering_output_entry_visits: persistent_summary.output_entry_visits,
        persistent_lowering_gpu_milliseconds: persistent_summary.gpu_milliseconds,
        persistent_lowering_high_water_bytes: persistent_summary.scratch_high_water_bytes,
        persistent_lowering_peak_output_handle_bytes: persistent_summary
            .peak_immutable_handle_bytes,
        persistent_lowering_maximum_absolute_coefficient: persistent_summary
            .maximum_absolute_coefficient,
        persistent_lowering_device_hard_cap_bytes: persistent_summary.device_hard_cap_bytes,
        persistent_lowering_download_chunk_terms: persistent_summary.download_chunk_terms,
        cpu_parity_terms,
        cpu_parity_passed,
        end_to_end_milliseconds: started.elapsed().as_millis(),
        raising_residuals: metadata.raising_residuals,
        highest_weight_certification: "Exact abstract highest-weight source, exact equivariant embedded source map, and exact highest-weight reciprocal map. The bounded streamed GPU path does not materialize the globally recoupled exact map.".to_string(),
        direct_composed_raising_residuals_materialized: false,
        single_column_rank: rank.rank_over_gaussian_extension,
        passed: rank.rank_over_gaussian_extension == 1
            && nonzero_functional_rows != 0
            && metadata.raising_residuals == [0; 5]
            && cpu_parity_terms != 0
            && cpu_parity_passed,
        proof_boundary: "Each raw descendant-times-reciprocal contribution is contracted exactly once. Every bounded batch is reduced and contracted modulo the declared prime, and its canonical row residues are folded into a fixed modular accumulator. Batch reduced-key and expanded-contribution counts are execution counters and depend on batch boundaries. Highest-weight certification follows from the exact abstract highest-weight source plus exact equivariance of the embedded and reciprocal maps; this path does not directly materialize the global exact recoupled residual. Full modular column rank proves the corresponding characteristic-zero lower bound when all rational denominators are invertible. This artifact covers only the declared p2D13 slice.".to_string(),
    };
    let json_path = output_directory.join(format!("{stem}.json"));
    let mut json = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    json.push(b'\n');
    write_atomic(&json_path, &json)?;
    if !report.passed {
        return Err("CUDA F_X column failed its rank or provenance gates".to_string());
    }
    Ok(report)
}

#[cfg(feature = "cuda")]
pub(crate) fn encode_modular_column(
    column: &ModularFunctionalColumn,
    static_digest: &str,
    source_digest: &str,
    source_terms: u64,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(160 + column.rows.len() * 8);
    output.extend_from_slice(b"ADFXGPU3");
    output.extend_from_slice(&column.prime.to_le_bytes());
    output.extend_from_slice(&(column.global_ordinal as u32).to_le_bytes());
    output.extend_from_slice(&(column.rows.len() as u32).to_le_bytes());
    output.extend_from_slice(&source_terms.to_le_bytes());
    output.extend_from_slice(static_digest.as_bytes());
    output.extend_from_slice(source_digest.as_bytes());
    for value in &column.rows {
        output.extend_from_slice(&value.real.to_le_bytes());
        output.extend_from_slice(&value.imaginary.to_le_bytes());
    }
    output
}

#[cfg(feature = "cuda")]
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    {
        let file = File::create(&temporary).map_err(|error| error.to_string())?;
        let mut writer = BufWriter::new(file);
        writer.write_all(bytes).map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
    }
    fs::rename(&temporary, path).map_err(|error| error.to_string())?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p3_three_prime_plans_have_identical_structure() {
        let plans = GPU_FX_PRIMES
            .iter()
            .map(|&prime| {
                let static_data = ModularFxStaticData::build(prime).unwrap();
                build_p3_modular_flat_plan(&static_data).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.entries.len())
                .collect::<Vec<_>>(),
            vec![5_390_208; 3]
        );
        for slot in 1..plans.len() {
            let left = &plans[0];
            let right = &plans[slot];
            let same_offsets = left.offsets == right.offsets;
            let same_keys = left.entries.len() == right.entries.len()
                && left
                    .entries
                    .iter()
                    .zip(&right.entries)
                    .all(|(left, right)| left.key == right.key);
            let differing_coefficients = left
                .entries
                .iter()
                .zip(&right.entries)
                .filter(|(left, right)| left.coefficient != right.coefficient)
                .count();
            assert!(same_offsets);
            assert!(same_keys);
            assert_eq!(differing_coefficients, plans[0].entries.len());
        }
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.semantic_sha256.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        for index in 0..plans[0].entries.len() {
            let scaled = [0_usize, 1_usize].map(|component| {
                (0..3)
                    .map(|slot| {
                        let prime = u64::from(GPU_FX_PRIMES[slot]);
                        let coefficient = plans[slot].entries[index].coefficient;
                        let residue = if component == 0 {
                            coefficient.real
                        } else {
                            coefficient.imaginary
                        };
                        let raw = u64::from(residue) * 13_440 % prime;
                        if raw <= prime / 2 {
                            raw as i64
                        } else {
                            raw as i64 - prime as i64
                        }
                    })
                    .collect::<Vec<_>>()
            });
            assert!(
                scaled
                    .iter()
                    .all(|values| values.windows(2).all(|pair| pair[0] == pair[1]))
            );
            assert!(scaled.iter().flatten().all(|value| value.abs() <= 3_024));
            assert!(scaled[0][0] != 0 || scaled[1][0] != 0);
        }
    }

    #[test]
    fn pinned_primes_are_valid_gaussian_extension_fields() {
        for prime in GPU_FX_PRIMES {
            validate_prime(prime).unwrap();
            assert_eq!(prime % 4, 3);
        }
    }

    #[test]
    fn gaussian_field_operations_are_exact() {
        let prime = GPU_FX_PRIMES[0];
        let left = GaussianResidue {
            real: 2,
            imaginary: 3,
        };
        let right = GaussianResidue {
            real: 5,
            imaginary: 7,
        };
        assert_eq!(
            left.multiply(right, prime),
            GaussianResidue {
                real: prime - 11,
                imaginary: 29,
            }
        );
        assert_eq!(
            left.multiply(gaussian_inverse(left, prime), prime),
            GaussianResidue {
                real: 1,
                imaginary: 0
            }
        );
    }

    #[test]
    fn pair_ordinals_cover_all_symmetric_monomials() {
        let mut observed = Vec::new();
        for left in 0..11 {
            for right in left..11 {
                observed.push(pair_ordinal(left, right));
            }
        }
        assert_eq!(observed, (0..66).collect::<Vec<_>>());
    }

    #[test]
    fn p3_functional_layout_is_a_278784_row_bijection_with_axis_retained() {
        assert_eq!(P3_FUNCTIONAL_ROW_COUNT, 278_784);
        for row in 0..P3_FUNCTIONAL_ROW_COUNT {
            let coordinates = decode_p3_functional_row(row).unwrap();
            assert_eq!(p3_functional_row(coordinates).unwrap(), row);
        }
        assert!(decode_p3_functional_row(P3_FUNCTIONAL_ROW_COUNT).is_err());
        assert!(
            p3_functional_row(P3FunctionalRowCoordinates {
                gauge_degree: 0,
                momentum_pair_ordinal: 0,
                sector: 0,
                contraction_axis: 11,
                seed: 0,
                bucket: 0,
            })
            .is_err()
        );
    }

    #[test]
    fn p3_cpu_reference_production_path_emits_valid_nonzero_rows() {
        let prime = GPU_FX_PRIMES[0];
        let exact = P3D11ExactStaticData::build().unwrap();
        let modular = ModularFxStaticData::build(prime).unwrap();
        let column = GpuFxColumnInput {
            global_ordinal: 0,
            source_label: "p3-tiny-canary".to_string(),
            source_copy: 1,
            terms: vec![RecoupledSourceTerm {
                momentum_pair: [2, 7],
                free_spinor: 15,
                exterior_mask: 0x0000_0fff,
                coefficient: 7,
            }],
            raising_residuals: [0; 5],
        };
        let result = accumulate_p3_column_cpu(&exact, &modular, &column).unwrap();
        let plan = build_p3_modular_flat_plan(&modular).unwrap();
        assert!(plan.entry_count() > 0);
        assert_eq!(plan.semantic_sha256().len(), 64);
        let fanout_table = plan.raw_fanout_table().unwrap();
        assert_eq!(fanout_table.within_byte.len(), 32 * 4 * 256);
        assert_eq!(fanout_table.cross_byte.len(), 32 * 6 * 256 * 256);
        assert_eq!(
            (0..4)
                .flat_map(
                    |left| (left + 1..4).map(move |right| p3_fanout_byte_pair_index(left, right))
                )
                .collect::<Vec<_>>(),
            (0..6).collect::<Vec<_>>()
        );
        for (free_spinor, exterior_mask) in [
            (0, 0x0000_0fff),
            (7, 0x0707_0707),
            (15, 0xff00_000f),
            (23, 0xf000_00ff),
            (31, 0x003f_003f),
        ] {
            let source = RecoupledSourceTerm {
                momentum_pair: [0, 10],
                free_spinor,
                exterior_mask,
                coefficient: 1,
            };
            assert_eq!(source.exterior_mask.count_ones(), 12);
            let free = usize::from(source.free_spinor);
            let mut direct_fanout = 0_u64;
            for contracted in 0..32 {
                if source.exterior_mask & (1_u32 << contracted) == 0 {
                    continue;
                }
                let degree_eleven_mask = source.exterior_mask ^ (1_u32 << contracted);
                for template in 0..32 {
                    if degree_eleven_mask & (1_u32 << template) == 0 {
                        direct_fanout +=
                            fanout_table.counts[free * 32 * 32 + contracted * 32 + template];
                    }
                }
            }
            assert_eq!(fanout_table.fanout(&source).unwrap(), direct_fanout);
        }
        let flat = accumulate_p3_column_cpu_flat(&plan, &column).unwrap();
        assert_eq!(flat.rows, result.rows);
        assert_eq!(flat.semantic_sha256, result.semantic_sha256);
        for (sector, invalid_coordinate) in
            [(0, P3_X2_OUTPUT_COORDINATES), (1, P3_X5_OUTPUT_COORDINATES)]
        {
            let mut mutated = plan.clone();
            let entry = mutated
                .entries
                .iter_mut()
                .find(|entry| usize::from(entry.key.sector) == sector)
                .unwrap();
            entry.key.output_coordinate = invalid_coordinate as u16;
            assert!(validate_p3_modular_flat_plan(&mutated).is_err());
            assert!(accumulate_p3_column_cpu_flat(&mutated, &column).is_err());
        }
        assert_eq!(result.prime, prime);
        assert_eq!(result.global_ordinal, 0);
        assert_eq!(result.rows.len(), P3_FUNCTIONAL_ROW_COUNT);
        assert!(result.expanded_contributions > 0);
        assert_eq!(result.semantic_sha256.len(), 64);
        let nonzero = result
            .rows
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_zero())
            .collect::<Vec<_>>();
        assert!(!nonzero.is_empty());
        assert!(nonzero.iter().all(|(row, _)| {
            decode_p3_functional_row(*row)
                .is_ok_and(|coordinates| coordinates.contraction_axis < 11)
        }));
    }

    #[test]
    fn p3_rank_certificate_covers_full_layout_and_rejects_bad_shape() {
        let prime = GPU_FX_PRIMES[0];
        let mut first = vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
        let mut second = first.clone();
        first[0].real = 1;
        second[P3_FUNCTIONAL_ROW_COUNT - 1].imaginary = 1;
        let columns = [
            ModularP3FunctionalColumn {
                prime,
                global_ordinal: 0,
                semantic_sha256: p3_column_semantic_sha256(prime, 0, &first),
                rows: first,
                expanded_contributions: 1,
            },
            ModularP3FunctionalColumn {
                prime,
                global_ordinal: 1,
                semantic_sha256: p3_column_semantic_sha256(prime, 1, &second),
                rows: second,
                expanded_contributions: 1,
            },
        ];
        let certificate = rank_p3_columns(&columns).unwrap();
        assert_eq!(certificate.schema_version, GPU_P3_FX_SCHEMA);
        assert_eq!(certificate.row_count, P3_FUNCTIONAL_ROW_COUNT);
        assert_eq!(certificate.rank_over_gaussian_extension, 2);
        assert!(certificate.full_column_rank);
        assert!(
            rank_p3_columns(&[ModularP3FunctionalColumn {
                prime,
                global_ordinal: 0,
                rows: vec![GaussianResidue::zero(); FUNCTIONAL_ROW_COUNT],
                expanded_contributions: 0,
                semantic_sha256: "c".repeat(64),
            }])
            .is_err()
        );
        let mut bad_ordinal = columns.clone();
        bad_ordinal[0].global_ordinal = 77;
        assert!(rank_p3_columns(&bad_ordinal).is_err());
        let mut bad_prime = columns.clone();
        bad_prime[0].prime = 19;
        bad_prime[1].prime = 19;
        assert!(rank_p3_columns(&bad_prime).is_err());
        let mut bad_residue = columns.clone();
        bad_residue[0].rows[0].real = prime;
        assert!(rank_p3_columns(&bad_residue).is_err());
        let mut bad_semantic = columns;
        bad_semantic[0].semantic_sha256.replace_range(0..1, "z");
        assert!(rank_p3_columns(&bad_semantic).is_err());
    }

    #[test]
    fn p3_artifact_decoder_validates_shape_residues_and_semantic_digest() {
        let prime = GPU_FX_PRIMES[0];
        let mut rows = vec![GaussianResidue::zero(); P3_FUNCTIONAL_ROW_COUNT];
        rows[17] = GaussianResidue {
            real: 3,
            imaginary: 5,
        };
        let column = ModularP3FunctionalColumn {
            prime,
            global_ordinal: 4,
            semantic_sha256: p3_column_semantic_sha256(prime, 4, &rows),
            rows,
            expanded_contributions: 9,
        };
        let plan = "a".repeat(64);
        let encoded = encode_p3_column_artifact(&plan, &column).unwrap();
        let (decoded_plan, decoded) = decode_p3_column_artifact(&encoded).unwrap();
        assert_eq!(decoded_plan, plan);
        assert_eq!(decoded.rows, column.rows);
        assert_eq!(decoded.semantic_sha256, column.semantic_sha256);
        let mut mutated = encoded.clone();
        mutated[GPU_P3_ARTIFACT_HEADER_BYTES + 17 * 8] ^= 1;
        assert!(decode_p3_column_artifact(&mutated).is_err());
        let mut mutated_count = encoded.clone();
        mutated_count[20] ^= 1;
        assert!(decode_p3_column_artifact(&mutated_count).is_err());
        let mut mutated_plan = encoded.clone();
        mutated_plan[28] = if mutated_plan[28] == b'a' { b'b' } else { b'a' };
        assert!(decode_p3_column_artifact(&mutated_plan).is_err());
        let mut bad_residue = encoded;
        bad_residue[GPU_P3_ARTIFACT_HEADER_BYTES..GPU_P3_ARTIFACT_HEADER_BYTES + 4]
            .copy_from_slice(&prime.to_le_bytes());
        assert!(decode_p3_column_artifact(&bad_residue).is_err());
    }

    #[test]
    fn static_modular_schedule_is_deterministic() {
        let left = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
        let right = ModularFxStaticData::build(GPU_FX_PRIMES[0]).unwrap();
        assert_eq!(left.semantic_sha256(), right.semantic_sha256());
        assert_eq!(left.target, right.target);
        assert_eq!(left.templates, right.templates);
        assert!(!left.templates.is_empty());
    }

    #[test]
    fn modular_rank_detects_independent_columns() {
        let prime = GPU_FX_PRIMES[0];
        let mut first = vec![GaussianResidue::zero(); FUNCTIONAL_ROW_COUNT];
        let mut second = first.clone();
        first[0].real = 1;
        second[1].imaginary = 1;
        let certificate = rank_columns(&[
            ModularFunctionalColumn {
                prime,
                global_ordinal: 53,
                rows: first,
                expanded_contributions: 1,
                semantic_sha256: "a".repeat(64),
            },
            ModularFunctionalColumn {
                prime,
                global_ordinal: 54,
                rows: second,
                expanded_contributions: 1,
                semantic_sha256: "b".repeat(64),
            },
        ])
        .unwrap();
        assert_eq!(certificate.rank_over_gaussian_extension, 2);
        assert!(certificate.full_column_rank);
    }
}
