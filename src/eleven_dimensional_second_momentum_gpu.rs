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
    DegreeTwoMomentumMonomial, SECOND_MOMENTUM_FX_BUCKETS_PER_SEED,
    SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS, SecondMomentumFxSector, SecondMomentumGaugeBranch,
    SecondMomentumGaugeChannel,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecoupledSourceTerm {
    pub momentum_pair: [u8; 2],
    pub free_spinor: u8,
    pub exterior_mask: u32,
    pub coefficient: i128,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub(crate) struct ModularFunctionalColumn {
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
        if term.momentum_pair[0] > term.momentum_pair[1]
            || term.momentum_pair[1] >= 11
            || term.free_spinor >= 32
            || term.exterior_mask.count_ones() != 12
        {
            return Err("invalid CUDA recoupled source term".to_string());
        }
        // Low 32 bits are the degree-12 exterior mask. High metadata bits
        // 0..3 are pair_left, 4..7 pair_right, and 8..12 free_spinor. All
        // remaining bits are zero, making semantic keys canonical.
        let metadata = u32::from(term.momentum_pair[0])
            | (u32::from(term.momentum_pair[1]) << 4)
            | (u32::from(term.free_spinor) << 8);
        debug_assert_eq!(metadata >> RECOUPLING_METADATA_BITS, 0);
        Ok((u64::from(metadata) << 32) | u64::from(term.exterior_mask))
    }

    #[cfg(test)]
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
    CudaModularFx, CudaMultiColumnBatch, CudaMultiColumnStats, lower_sparse_exact,
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
