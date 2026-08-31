//! Exact CPU contract for the final 56-column four-form coefficient system.
//!
//! The executable pieces are deliberately separated from the still-missing
//! PBW source-graph and physical target-gauge quotient matrices.  A caller can
//! build the convention-correct teleparallel right-hand side and the complete
//! formal Bianchi image today.  Final launch readiness remains fail closed
//! until independently bound integrability and supplied-K descent artifacts
//! are present.

use std::collections::{BTreeMap, BTreeSet};

use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_corrected_full_chain_oracle::{
    FullChainRowKey, horizontal_corrected_full_chain_streams,
};
use crate::eleven_dimensional_four_form_56_gpu::{
    BidegreeBranch, CanonicalRow, D_G4_COORDINATES, D02_COLUMN_COUNT, D02_ROW_COUNT,
    D02_SOURCE_COORDINATES, D21_COLUMN_COUNT, D21_ROW_COUNT, D21_SECTORS, D21_SOURCE_COORDINATES,
    ExactCooEntry, PINNED_PRIMES, ThreePrimeGaussian,
};
use crate::eleven_dimensional_physical_curvature::ExactQi;

pub const PHYSICS_ROW_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-physics-rows-v1";
pub const TELEPARALLEL_RHS_SCHEMA_VERSION: &str =
    "adynkra-11d-four-form-56-right-c-teleparallel-rhs-v1";
pub const BIANCHI_ROW_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-bianchi-row-v1";
pub const PBW_BINDING_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-pbw-binding-v1";
pub const SOURCE_GAUGE_BINDING_SCHEMA_VERSION: &str =
    "adynkra-11d-four-form-56-source-gauge-binding-v1";
pub const EQUALITY_LAUNCH_BINDING_SCHEMA_VERSION: &str =
    "adynkra-11d-four-form-global57-equality-launch-v1";
pub const SELECTED_COLUMN_AUDIT_SCHEMA_VERSION: &str =
    "adynkra-11d-four-form-56-selected-column-audit-v1";
pub const GLOBAL57_RHS_SCHEMA_VERSION: &str = "adynkra-11d-four-form-global57-right-c-rhs-v1";
pub const AUGMENTED_TARGET_COLUMN: u32 = 56;
pub const AUGMENTED_COLUMN_COUNT: u32 = 57;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const H_HAT_DIMENSION: usize = 320;
const FOUR_FORM_DIMENSION: usize = 330;
const FIVE_FORM_DIMENSION: usize = 462;
const D_BIANCHI_TARGET_DIMENSION: u64 = (SPINOR_DIMENSION * FIVE_FORM_DIMENSION) as u64;
const EXTERIOR_TWO_DIMENSION: u64 = 496;
const SYMMETRIC_TWO_DIMENSION: u64 = 66;
const SYMMETRIC_THREE_DIMENSION: u64 = 286;
const D21_BIANCHI_SOURCE_DIMENSION: u64 =
    EXTERIOR_TWO_DIMENSION * SYMMETRIC_TWO_DIMENSION * H_HAT_DIMENSION as u64;
const D02_BIANCHI_SOURCE_DIMENSION: u64 = SYMMETRIC_THREE_DIMENSION * H_HAT_DIMENSION as u64;
const D21_BIANCHI_ROW_COUNT: u64 = D21_BIANCHI_SOURCE_DIMENSION * D_BIANCHI_TARGET_DIMENSION;
const TOTAL_BIANCHI_ROW_COUNT: u64 =
    D21_BIANCHI_ROW_COUNT + D02_BIANCHI_SOURCE_DIMENSION * D_BIANCHI_TARGET_DIMENSION;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactGaussianRational {
    pub real_numerator: i64,
    pub real_denominator: i64,
    pub imaginary_numerator: i64,
    pub imaginary_denominator: i64,
}

impl From<&ExactQi> for ExactGaussianRational {
    fn from(value: &ExactQi) -> Self {
        Self {
            real_numerator: *value.real.numer(),
            real_denominator: *value.real.denom(),
            imaginary_numerator: *value.imaginary.numer(),
            imaginary_denominator: *value.imaginary.denom(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeleparallelRhsEntry {
    pub row: u64,
    pub coefficient: ExactGaussianRational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeleparallelRhsColumn {
    pub schema_version: String,
    pub h_hat_ordinal: u32,
    pub d21_entries: u64,
    pub d02_entries: u64,
    pub entries: Vec<TeleparallelRhsEntry>,
    pub stream_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Global57RhsEntry {
    pub row: u64,
    pub column: u32,
    pub coefficient: ExactGaussianRational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Global57RhsBatch {
    pub schema_version: String,
    pub batch_ordinal: u32,
    pub first_h_hat_ordinal: u32,
    pub last_h_hat_ordinal_exclusive: u32,
    pub d21_entries: u64,
    pub d02_entries: u64,
    pub entries: Vec<Global57RhsEntry>,
    pub batch_sha256: String,
}

impl Global57RhsBatch {
    pub fn exact_value(&self, row: u64) -> Option<&ExactGaussianRational> {
        self.entries
            .binary_search_by_key(&row, |entry| entry.row)
            .ok()
            .map(|index| &self.entries[index].coefficient)
    }
}

fn pow_mod(mut base: u32, mut exponent: u32, modulus: u32) -> u32 {
    let mut output = 1_u64;
    let mut base64 = u64::from(base);
    let modulus64 = u64::from(modulus);
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = output * base64 % modulus64;
        }
        base64 = base64 * base64 % modulus64;
        exponent >>= 1;
    }
    base = output as u32;
    base
}

/// Exact Q(i) to the pinned three-prime Gaussian lanes used by the 57-column
/// reducer. Denominators divisible by a pinned prime fail closed.
pub fn global57_rhs_three_prime(entry: &Global57RhsEntry) -> Result<ThreePrimeGaussian, String> {
    if entry.column != AUGMENTED_TARGET_COLUMN {
        return Err("global57 modular RHS entry is not column 56".to_string());
    }
    let values = [
        (
            entry.coefficient.real_numerator,
            entry.coefficient.real_denominator,
        ),
        (
            entry.coefficient.imaginary_numerator,
            entry.coefficient.imaginary_denominator,
        ),
    ];
    let mut output = ThreePrimeGaussian::default();
    for (slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
        for (component, (numerator, denominator)) in values.into_iter().enumerate() {
            let denominator = denominator.rem_euclid(i64::from(prime)) as u32;
            if denominator == 0 {
                return Err(format!(
                    "global57 RHS denominator is inadmissible at prime {prime}"
                ));
            }
            let numerator = numerator.rem_euclid(i64::from(prime)) as u32;
            let inverse = pow_mod(denominator, prime - 2, prime);
            output.value[2 * slot + component] =
                (u64::from(numerator) * u64::from(inverse) % u64::from(prime)) as u32;
        }
    }
    Ok(output)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Global57RhsManifest {
    pub schema_version: String,
    pub h_hat_columns: u32,
    pub batches: u32,
    pub d21_entries: u64,
    pub d02_entries: u64,
    pub total_entries: u64,
    pub stream_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BianchiRowKey {
    pub branch: BidegreeBranch,
    pub source_coordinate: u64,
    pub target_coordinate: u32,
}

impl BianchiRowKey {
    pub fn ordinal(self) -> Result<u64, String> {
        if u64::from(self.target_coordinate) >= D_BIANCHI_TARGET_DIMENSION {
            return Err("Bianchi target coordinate is out of range".to_string());
        }
        let (base, bound) = match self.branch {
            BidegreeBranch::D2P1 => (0, D21_BIANCHI_SOURCE_DIMENSION),
            BidegreeBranch::D0P2 => (D21_BIANCHI_ROW_COUNT, D02_BIANCHI_SOURCE_DIMENSION),
        };
        if self.source_coordinate >= bound {
            return Err("Bianchi source coordinate is out of range".to_string());
        }
        Ok(base
            + self.source_coordinate * D_BIANCHI_TARGET_DIMENSION
            + u64::from(self.target_coordinate))
    }

    pub fn from_ordinal(ordinal: u64) -> Result<Self, String> {
        if ordinal >= TOTAL_BIANCHI_ROW_COUNT {
            return Err("Bianchi row ordinal is out of range".to_string());
        }
        let (branch, relative) = if ordinal < D21_BIANCHI_ROW_COUNT {
            (BidegreeBranch::D2P1, ordinal)
        } else {
            (BidegreeBranch::D0P2, ordinal - D21_BIANCHI_ROW_COUNT)
        };
        Ok(Self {
            branch,
            source_coordinate: relative / D_BIANCHI_TARGET_DIMENSION,
            target_coordinate: (relative % D_BIANCHI_TARGET_DIMENSION) as u32,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactBianchiCooEntry {
    pub row: u64,
    pub column: u32,
    pub reserved: u32,
    pub real: i64,
    pub imaginary: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBianchiRhsEntry {
    pub row: u64,
    pub column: u32,
    pub coefficient: ExactGaussianRational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PbwIntegrabilityBinding {
    pub schema_version: String,
    pub generator_map_sha256: String,
    pub source_graph_sha256: String,
    pub exact_matrix_sha256: String,
    pub exact_rows: u64,
    pub d21_to_d02_translation_complete: bool,
    pub all_56_columns_bound: bool,
    pub unrestricted_target_bianchi_residual_routed: bool,
    pub mutation_rejected: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EqualityLaunchBinding {
    pub schema_version: String,
    pub ordered_generator_map_sha256: String,
    pub immutable_candidate_matrix_sha256: String,
    pub all_320_rhs_manifest_sha256: String,
    pub both_branch_join_sha256: String,
    pub arithmetic_parity_sha256: String,
    pub global57_reducer_sha256: String,
    pub immutable_candidate_matrix: bool,
    pub all_320_rhs_complete: bool,
    pub d21_and_d02_join_complete: bool,
    pub exact_cpu_three_prime_parity: bool,
    pub global57_reducer_ready: bool,
    pub mutation_rejected: bool,
}

impl EqualityLaunchBinding {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EQUALITY_LAUNCH_BINDING_SCHEMA_VERSION
            || !self.immutable_candidate_matrix
            || !self.all_320_rhs_complete
            || !self.d21_and_d02_join_complete
            || !self.exact_cpu_three_prime_parity
            || !self.global57_reducer_ready
            || !self.mutation_rejected
        {
            return Err("bounded global57 equality launch binding is incomplete".to_string());
        }
        validate_digests([
            self.ordered_generator_map_sha256.as_str(),
            self.immutable_candidate_matrix_sha256.as_str(),
            self.all_320_rhs_manifest_sha256.as_str(),
            self.both_branch_join_sha256.as_str(),
            self.arithmetic_parity_sha256.as_str(),
            self.global57_reducer_sha256.as_str(),
        ])
    }
}

impl PbwIntegrabilityBinding {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != PBW_BINDING_SCHEMA_VERSION
            || self.exact_rows == 0
            || !self.d21_to_d02_translation_complete
            || !self.all_56_columns_bound
            || !self.unrestricted_target_bianchi_residual_routed
            || !self.mutation_rejected
        {
            return Err("PBW integrability binding is incomplete".to_string());
        }
        validate_digests([
            self.generator_map_sha256.as_str(),
            self.source_graph_sha256.as_str(),
            self.exact_matrix_sha256.as_str(),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceGaugeDescentBinding {
    pub schema_version: String,
    pub generator_map_sha256: String,
    pub complete_f_sha256: String,
    pub physical_k_sha256: String,
    pub quotient_normal_form_sha256: String,
    pub exact_matrix_sha256: String,
    pub exact_rows: u64,
    pub fk_zero_exact: bool,
    pub all_six_source_channels: bool,
    pub polynomial_witnesses_complete: bool,
    pub quotient_normal_forms_replayed: bool,
    pub mutation_rejected: bool,
}

impl SourceGaugeDescentBinding {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SOURCE_GAUGE_BINDING_SCHEMA_VERSION
            || self.exact_rows == 0
            || !self.fk_zero_exact
            || !self.all_six_source_channels
            || !self.polynomial_witnesses_complete
            || !self.quotient_normal_forms_replayed
            || !self.mutation_rejected
        {
            return Err("source-gauge descent binding is incomplete".to_string());
        }
        validate_digests([
            self.generator_map_sha256.as_str(),
            self.complete_f_sha256.as_str(),
            self.physical_k_sha256.as_str(),
            self.quotient_normal_form_sha256.as_str(),
            self.exact_matrix_sha256.as_str(),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicsAssemblyReadiness {
    pub schema_version: String,
    pub ordered_generator_map_sha256: String,
    pub immutable_candidate_matrix_sha256: String,
    pub all_320_rhs_manifest_sha256: String,
    pub teleparallel_match_ready: bool,
    pub bianchi_transform_ready: bool,
    pub pbw_integrability_ready: bool,
    pub source_gauge_descent_ready: bool,
    pub raw_equality_solve_ready: bool,
    pub physical_descent_ready: bool,
    pub launch_ready: bool,
    pub equality_blockers: Vec<String>,
    pub physical_promotion_blockers: Vec<String>,
}

pub fn physics_assembly_readiness(
    equality: Option<&EqualityLaunchBinding>,
    pbw: Option<&PbwIntegrabilityBinding>,
    source_gauge: Option<&SourceGaugeDescentBinding>,
) -> PhysicsAssemblyReadiness {
    let equality_ready = equality.is_some_and(|binding| binding.validate().is_ok());
    let ordered_generator_map_sha256 = equality
        .map(|binding| binding.ordered_generator_map_sha256.clone())
        .unwrap_or_default();
    let immutable_candidate_matrix_sha256 = equality
        .map(|binding| binding.immutable_candidate_matrix_sha256.clone())
        .unwrap_or_default();
    let all_320_rhs_manifest_sha256 = equality
        .map(|binding| binding.all_320_rhs_manifest_sha256.clone())
        .unwrap_or_default();
    let pbw_ready = pbw.is_some_and(|binding| {
        binding.validate().is_ok() && binding.generator_map_sha256 == ordered_generator_map_sha256
    });
    let source_gauge_ready = source_gauge.is_some_and(|binding| {
        binding.validate().is_ok() && binding.generator_map_sha256 == ordered_generator_map_sha256
    });
    let mut equality_blockers = Vec::new();
    let mut physical_promotion_blockers = Vec::new();
    if !equality_ready {
        equality_blockers.push(
            "immutable M, all-320 t, both-branch join, arithmetic parity, or global57 reducer is absent"
                .to_string(),
        );
    }
    if !pbw_ready {
        physical_promotion_blockers.push(
            "exact PBW translation matrix tying the d21 and d02 branches is absent".to_string(),
        );
    }
    if !source_gauge_ready {
        physical_promotion_blockers.push(
            "physical K and quotient-normal-form source-gauge descent matrix are absent"
                .to_string(),
        );
    }
    PhysicsAssemblyReadiness {
        schema_version: PHYSICS_ROW_SCHEMA_VERSION.to_string(),
        ordered_generator_map_sha256: ordered_generator_map_sha256.to_string(),
        immutable_candidate_matrix_sha256,
        all_320_rhs_manifest_sha256,
        teleparallel_match_ready: equality_ready,
        bianchi_transform_ready: true,
        pbw_integrability_ready: pbw_ready,
        source_gauge_descent_ready: source_gauge_ready,
        raw_equality_solve_ready: equality_ready,
        physical_descent_ready: equality_ready && pbw_ready && source_gauge_ready,
        launch_ready: equality_ready,
        equality_blockers,
        physical_promotion_blockers,
    }
}

fn validate_digest(value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid SHA-256 digest".to_string());
    }
    Ok(())
}

fn validate_digests<'a>(values: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    values.into_iter().try_for_each(validate_digest)
}

fn exterior_two_ordinal(mask: u32) -> Result<usize, String> {
    if mask.count_ones() != 2 {
        return Err("exterior-spinor mask is not a canonical degree-two mask".to_string());
    }
    let pair = (0..SPINOR_DIMENSION)
        .filter(|axis| mask & (1_u32 << axis) != 0)
        .collect::<Vec<_>>();
    let mut ordinal = 0;
    for left in 0..SPINOR_DIMENSION {
        for right in (left + 1)..SPINOR_DIMENSION {
            if pair == [left, right] {
                return Ok(ordinal);
            }
            ordinal += 1;
        }
    }
    Err("degree-two mask did not map to Lambda2(S)".to_string())
}

fn symmetric_two_ordinal(left: usize, right: usize) -> Result<usize, String> {
    if left >= VECTOR_DIMENSION || right >= VECTOR_DIMENSION {
        return Err("symmetric momentum pair is out of range".to_string());
    }
    let pair = if left <= right {
        [left, right]
    } else {
        [right, left]
    };
    let mut ordinal = 0;
    for first in 0..VECTOR_DIMENSION {
        for second in first..VECTOR_DIMENSION {
            if pair == [first, second] {
                return Ok(ordinal);
            }
            ordinal += 1;
        }
    }
    unreachable!()
}

fn symmetric_three_ordinal(mut axes: [usize; 3]) -> Result<usize, String> {
    if axes.iter().any(|&axis| axis >= VECTOR_DIMENSION) {
        return Err("symmetric momentum triple is out of range".to_string());
    }
    axes.sort_unstable();
    let mut ordinal = 0;
    for first in 0..VECTOR_DIMENSION {
        for second in first..VECTOR_DIMENSION {
            for third in second..VECTOR_DIMENSION {
                if axes == [first, second, third] {
                    return Ok(ordinal);
                }
                ordinal += 1;
            }
        }
    }
    unreachable!()
}

fn degree_and_axes(exponents: &[u16; VECTOR_DIMENSION]) -> (u32, Vec<usize>) {
    let degree = exponents.iter().map(|&value| u32::from(value)).sum();
    let mut axes = Vec::new();
    for (axis, &exponent) in exponents.iter().enumerate() {
        axes.extend(std::iter::repeat_n(axis, usize::from(exponent)));
    }
    (degree, axes)
}

fn lexicographic_four_form_masks() -> Vec<u16> {
    let mut output = Vec::with_capacity(FOUR_FORM_DIMENSION);
    for first in 0..VECTOR_DIMENSION {
        for second in (first + 1)..VECTOR_DIMENSION {
            for third in (second + 1)..VECTOR_DIMENSION {
                for fourth in (third + 1)..VECTOR_DIMENSION {
                    output.push(
                        (1_u16 << first) | (1_u16 << second) | (1_u16 << third) | (1_u16 << fourth),
                    );
                }
            }
        }
    }
    output
}

/// Convert the teleparallel operator's lexicographic four-form ordinal to the
/// ascending numeric-mask ordinal used by D21, D02, and the DG4 projectors.
pub(crate) fn lexicographic_four_form_to_numeric(ordinal: usize) -> Result<usize, String> {
    let mask = *lexicographic_four_form_masks()
        .get(ordinal)
        .ok_or_else(|| "lexicographic four-form ordinal is out of range".to_string())?;
    form_masks(4)
        .iter()
        .position(|&candidate| candidate == mask)
        .ok_or_else(|| "lexicographic four-form mask is absent from numeric basis".to_string())
}

pub(crate) fn numeric_four_form_to_lexicographic(ordinal: usize) -> Result<usize, String> {
    let mask = *form_masks(4)
        .get(ordinal)
        .ok_or_else(|| "numeric four-form ordinal is out of range".to_string())?;
    lexicographic_four_form_masks()
        .iter()
        .position(|&candidate| candidate == mask)
        .ok_or_else(|| "numeric four-form mask is absent from lexicographic basis".to_string())
}

fn teleparallel_target_to_numeric(output_coordinate: usize) -> Result<usize, String> {
    if output_coordinate >= D_G4_COORDINATES as usize {
        return Err("teleparallel target coordinate is out of range".to_string());
    }
    let spinor = output_coordinate / FOUR_FORM_DIMENSION;
    let lexicographic_form = output_coordinate % FOUR_FORM_DIMENSION;
    Ok(spinor * FOUR_FORM_DIMENSION + lexicographic_four_form_to_numeric(lexicographic_form)?)
}

fn canonical_rhs_row(h_hat_ordinal: usize, key: &FullChainRowKey) -> Result<CanonicalRow, String> {
    if h_hat_ordinal >= H_HAT_DIMENSION {
        return Err("teleparallel coordinate is out of range".to_string());
    }
    let target_coordinate = teleparallel_target_to_numeric(key.output_coordinate)?;
    let (degree, axes) = degree_and_axes(&key.momentum_exponents);
    if key.exterior_spinor_mask.count_ones() == 2 && degree == 1 {
        let pair = exterior_two_ordinal(key.exterior_spinor_mask)?;
        let source = ((pair * VECTOR_DIMENSION + axes[0]) * H_HAT_DIMENSION) + h_hat_ordinal;
        return Ok(CanonicalRow {
            branch: BidegreeBranch::D2P1,
            source_coordinate: source as u64,
            target_coordinate: target_coordinate as u32,
        });
    }
    if key.exterior_spinor_mask == 0 && degree == 2 {
        let pair = symmetric_two_ordinal(axes[0], axes[1])?;
        let source = pair * H_HAT_DIMENSION + h_hat_ordinal;
        return Ok(CanonicalRow {
            branch: BidegreeBranch::D0P2,
            source_coordinate: source as u64,
            target_coordinate: target_coordinate as u32,
        });
    }
    Err(format!(
        "teleparallel PBW row lies outside declared branches: D-degree {}, p-degree {degree}",
        key.exterior_spinor_mask.count_ones()
    ))
}

pub fn teleparallel_rhs_column(h_hat_ordinal: usize) -> Result<TeleparallelRhsColumn, String> {
    if h_hat_ordinal >= H_HAT_DIMENSION {
        return Err("Hhat ordinal is out of range".to_string());
    }
    let horizontal = horizontal_corrected_full_chain_streams(h_hat_ordinal)?;
    if !horizontal.section_psi_two_is_zero
        || !horizontal.section_values_unchanged_by_horizontalization
    {
        return Err("horizontal section contract is not closed".to_string());
    }
    let teleparallel = horizontal.section_target;
    let mut entries = Vec::with_capacity(teleparallel.len());
    let mut branch_counts = [0_u64; 2];
    for (key, value) in teleparallel {
        let row = canonical_rhs_row(h_hat_ordinal, &key)?;
        branch_counts[match row.branch {
            BidegreeBranch::D2P1 => 0,
            BidegreeBranch::D0P2 => 1,
        }] += 1;
        entries.push(TeleparallelRhsEntry {
            row: row.ordinal()?,
            coefficient: ExactGaussianRational::from(&value),
        });
    }
    entries.sort_by_key(|entry| entry.row);
    if entries.windows(2).any(|pair| pair[0].row >= pair[1].row) {
        return Err("teleparallel RHS rows are duplicated or noncanonical".to_string());
    }
    let stream_sha256 = teleparallel_rhs_sha256(h_hat_ordinal, &entries);
    Ok(TeleparallelRhsColumn {
        schema_version: TELEPARALLEL_RHS_SCHEMA_VERSION.to_string(),
        h_hat_ordinal: h_hat_ordinal as u32,
        d21_entries: branch_counts[0],
        d02_entries: branch_counts[1],
        entries,
        stream_sha256,
    })
}

fn teleparallel_rhs_sha256(h_hat_ordinal: usize, entries: &[TeleparallelRhsEntry]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-four-form-56-right-c-teleparallel-rhs-v1\0");
    hash.update((h_hat_ordinal as u64).to_le_bytes());
    for entry in entries {
        hash.update(entry.row.to_le_bytes());
        hash.update(entry.coefficient.real_numerator.to_le_bytes());
        hash.update(entry.coefficient.real_denominator.to_le_bytes());
        hash.update(entry.coefficient.imaginary_numerator.to_le_bytes());
        hash.update(entry.coefficient.imaginary_denominator.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// Visit the exact augmented target column in bounded Hhat batches. Each
/// batch is canonically sorted by row and supports logarithmic exact lookup.
/// Batches partition Hhat ordinals, so a consumer may process them in any
/// order while binding `batch_sha256` and the aggregate manifest digest.
pub fn visit_global57_rhs_batches(
    h_hat_columns_per_batch: usize,
    mut emit: impl FnMut(&Global57RhsBatch) -> Result<(), String>,
) -> Result<Global57RhsManifest, String> {
    if h_hat_columns_per_batch == 0 || h_hat_columns_per_batch > H_HAT_DIMENSION {
        return Err("global57 Hhat batch width is out of range".to_string());
    }
    let mut aggregate = Sha256::new();
    aggregate.update(b"adynkra-11d-four-form-global57-right-c-rhs-v1\0");
    let mut total_d21 = 0_u64;
    let mut total_d02 = 0_u64;
    let mut batches = 0_u32;
    for first in (0..H_HAT_DIMENSION).step_by(h_hat_columns_per_batch) {
        let last = (first + h_hat_columns_per_batch).min(H_HAT_DIMENSION);
        let batch = global57_rhs_batch(batches, first, last)?;
        aggregate.update((batches as u64).to_le_bytes());
        aggregate.update((first as u64).to_le_bytes());
        aggregate.update((last as u64).to_le_bytes());
        aggregate.update(batch.batch_sha256.as_bytes());
        emit(&batch)?;
        total_d21 += batch.d21_entries;
        total_d02 += batch.d02_entries;
        batches += 1;
    }
    Ok(Global57RhsManifest {
        schema_version: GLOBAL57_RHS_SCHEMA_VERSION.to_string(),
        h_hat_columns: H_HAT_DIMENSION as u32,
        batches,
        d21_entries: total_d21,
        d02_entries: total_d02,
        total_entries: total_d21 + total_d02,
        stream_sha256: format!("{:x}", aggregate.finalize()),
    })
}

pub fn global57_rhs_batch(
    batch_ordinal: u32,
    first_h_hat_ordinal: usize,
    last_h_hat_ordinal_exclusive: usize,
) -> Result<Global57RhsBatch, String> {
    if first_h_hat_ordinal >= last_h_hat_ordinal_exclusive
        || last_h_hat_ordinal_exclusive > H_HAT_DIMENSION
    {
        return Err("global57 RHS Hhat interval is invalid".to_string());
    }
    let mut entries = Vec::new();
    let mut d21 = 0_u64;
    let mut d02 = 0_u64;
    for h in first_h_hat_ordinal..last_h_hat_ordinal_exclusive {
        let column = teleparallel_rhs_column(h)?;
        d21 += column.d21_entries;
        d02 += column.d02_entries;
        entries.extend(column.entries.into_iter().map(|entry| Global57RhsEntry {
            row: entry.row,
            column: AUGMENTED_TARGET_COLUMN,
            coefficient: entry.coefficient,
        }));
    }
    entries.sort_by_key(|entry| entry.row);
    if entries.windows(2).any(|pair| pair[0].row >= pair[1].row) {
        return Err("global57 RHS batch contains duplicate or noncanonical rows".to_string());
    }
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-four-form-global57-right-c-rhs-batch-v1\0");
    hash.update((first_h_hat_ordinal as u64).to_le_bytes());
    hash.update((last_h_hat_ordinal_exclusive as u64).to_le_bytes());
    for entry in &entries {
        hash.update(entry.row.to_le_bytes());
        hash.update(entry.column.to_le_bytes());
        hash.update(entry.coefficient.real_numerator.to_le_bytes());
        hash.update(entry.coefficient.real_denominator.to_le_bytes());
        hash.update(entry.coefficient.imaginary_numerator.to_le_bytes());
        hash.update(entry.coefficient.imaginary_denominator.to_le_bytes());
    }
    Ok(Global57RhsBatch {
        schema_version: GLOBAL57_RHS_SCHEMA_VERSION.to_string(),
        batch_ordinal,
        first_h_hat_ordinal: first_h_hat_ordinal as u32,
        last_h_hat_ordinal_exclusive: last_h_hat_ordinal_exclusive as u32,
        d21_entries: d21,
        d02_entries: d02,
        entries,
        batch_sha256: format!("{:x}", hash.finalize()),
    })
}

fn form_masks(degree: u32) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == degree)
        .collect()
}

fn wedge_sign(mask: u16, axis: usize) -> i64 {
    if (mask & ((1_u16 << axis) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn decode_d21_source(source: u64) -> Result<(usize, usize, usize), String> {
    if source >= D21_SOURCE_COORDINATES {
        return Err("d21 source coordinate is out of range".to_string());
    }
    let h = source as usize % H_HAT_DIMENSION;
    let quotient = source as usize / H_HAT_DIMENSION;
    Ok((quotient / VECTOR_DIMENSION, quotient % VECTOR_DIMENSION, h))
}

fn decode_d02_source(source: u64) -> Result<(usize, usize), String> {
    if source >= D02_SOURCE_COORDINATES {
        return Err("d02 source coordinate is out of range".to_string());
    }
    Ok((
        source as usize / H_HAT_DIMENSION,
        source as usize % H_HAT_DIMENSION,
    ))
}

fn symmetric_two_pair(ordinal: usize) -> Result<[usize; 2], String> {
    let mut current = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in left..VECTOR_DIMENSION {
            if current == ordinal {
                return Ok([left, right]);
            }
            current += 1;
        }
    }
    Err("symmetric pair ordinal is out of range".to_string())
}

/// Apply the formal target Bianchi operator `p wedge` to an exact candidate
/// COO stream. The input and output retain the same common denominator.
/// Cancellations are accumulated before canonical publication.
fn bianchi_expansions(row: CanonicalRow) -> Result<Vec<(u64, i64)>, String> {
    let forms4 = form_masks(4);
    let forms5 = form_masks(5);
    let form5_ordinals = forms5
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, mask)| (mask, ordinal))
        .collect::<BTreeMap<_, _>>();
    let target = usize::try_from(row.target_coordinate).unwrap();
    let spinor = target / FOUR_FORM_DIMENSION;
    let mask4 = forms4[target % FOUR_FORM_DIMENSION];
    let mut output = Vec::with_capacity(7);
    for derivative in 0..VECTOR_DIMENSION {
        if mask4 & (1_u16 << derivative) != 0 {
            continue;
        }
        let form5 = form5_ordinals[&(mask4 | (1_u16 << derivative))];
        let target5 = spinor * FIVE_FORM_DIMENSION + form5;
        let source5 = match row.branch {
            BidegreeBranch::D2P1 => {
                let (spinor_pair, momentum, h) = decode_d21_source(row.source_coordinate)?;
                let pair = symmetric_two_ordinal(momentum, derivative)?;
                (spinor_pair * SYMMETRIC_TWO_DIMENSION as usize + pair) * H_HAT_DIMENSION + h
            }
            BidegreeBranch::D0P2 => {
                let (pair, h) = decode_d02_source(row.source_coordinate)?;
                let [left, right] = symmetric_two_pair(pair)?;
                let triple = symmetric_three_ordinal([left, right, derivative])?;
                triple * H_HAT_DIMENSION + h
            }
        };
        output.push((
            BianchiRowKey {
                branch: row.branch,
                source_coordinate: source5 as u64,
                target_coordinate: target5 as u32,
            }
            .ordinal()?,
            wedge_sign(mask4, derivative),
        ));
    }
    Ok(output)
}

pub fn bianchi_image(
    entries: &[ExactCooEntry],
    common_denominator: u64,
) -> Result<Vec<ExactBianchiCooEntry>, String> {
    if common_denominator == 0 {
        return Err("Bianchi common denominator is zero".to_string());
    }
    let mut accumulated = BTreeMap::<(u64, u32), (i128, i128)>::new();
    for entry in entries {
        entry.validate()?;
        let row = CanonicalRow::from_ordinal(entry.row)?;
        for (bianchi_row, sign) in bianchi_expansions(row)? {
            let value = accumulated.entry((bianchi_row, entry.column)).or_default();
            value.0 += i128::from(sign) * i128::from(entry.real);
            value.1 += i128::from(sign) * i128::from(entry.imaginary);
        }
    }
    let mut output = Vec::new();
    for ((row, column), (real, imaginary)) in accumulated {
        if real == 0 && imaginary == 0 {
            continue;
        }
        output.push(ExactBianchiCooEntry {
            row,
            column,
            reserved: 0,
            real: i64::try_from(real)
                .map_err(|_| "Bianchi real numerator overflowed i64".to_string())?,
            imaginary: i64::try_from(imaginary)
                .map_err(|_| "Bianchi imaginary numerator overflowed i64".to_string())?,
        });
    }
    Ok(output)
}

/// Apply the formal target Bianchi operator to augmented target column 56.
/// The result is exact over Q(i), with cancellations completed before output.
pub fn bianchi_rhs_image(
    entries: &[Global57RhsEntry],
) -> Result<Vec<ExactBianchiRhsEntry>, String> {
    let mut accumulated = BTreeMap::<u64, ExactQi>::new();
    for entry in entries {
        if entry.column != AUGMENTED_TARGET_COLUMN {
            return Err("Bianchi RHS entry is not in augmented column 56".to_string());
        }
        let row = CanonicalRow::from_ordinal(entry.row)?;
        let value = ExactQi {
            real: Ratio::new(
                entry.coefficient.real_numerator,
                entry.coefficient.real_denominator,
            ),
            imaginary: Ratio::new(
                entry.coefficient.imaginary_numerator,
                entry.coefficient.imaginary_denominator,
            ),
        };
        for (bianchi_row, sign) in bianchi_expansions(row)? {
            let scaled = value.scaled(&Ratio::from_integer(sign));
            let target = accumulated.entry(bianchi_row).or_insert_with(ExactQi::zero);
            target.add_assign(&scaled);
            if target.is_zero() {
                accumulated.remove(&bianchi_row);
            }
        }
    }
    Ok(accumulated
        .into_iter()
        .map(|(row, coefficient)| ExactBianchiRhsEntry {
            row,
            column: AUGMENTED_TARGET_COLUMN,
            coefficient: ExactGaussianRational::from(&coefficient),
        })
        .collect())
}

pub fn bianchi_image_sha256(entries: &[ExactBianchiCooEntry], common_denominator: u64) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-four-form-56-bianchi-image-v1\0");
    hash.update(common_denominator.to_le_bytes());
    for entry in entries {
        hash.update(entry.row.to_le_bytes());
        hash.update(entry.column.to_le_bytes());
        hash.update(entry.real.to_le_bytes());
        hash.update(entry.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[derive(Debug, Deserialize)]
struct WitnessArtifact {
    schema_version: String,
    passed: bool,
    channels: Vec<WitnessChannel>,
}

#[derive(Debug, Deserialize)]
struct WitnessChannel {
    outer_fierz_degree: u8,
    sectors: Vec<WitnessSector>,
}

#[derive(Debug, Deserialize)]
struct WitnessSector {
    sector: String,
    expected_rank: usize,
    per_prime_ranks: [usize; 3],
    selected_diagram_ordinals: Vec<u16>,
    global_columns: Option<Vec<u32>>,
    exact_determinant_nonzero: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedColumnBinding {
    pub global_column: u32,
    pub target_sector: String,
    pub multiplicity_copy: u32,
    pub outer_fierz_degree: u8,
    pub diagram_ordinal: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedColumnAudit {
    pub schema_version: String,
    pub witness_artifact_sha256: String,
    pub d21_bindings: Vec<SelectedColumnBinding>,
    pub d02_order: Vec<String>,
    pub passed: bool,
}

fn expected_d21_identity(column: u32) -> Option<(&'static str, u32)> {
    D21_SECTORS.iter().find_map(|sector| {
        (sector.first_global_column..sector.first_global_column + sector.multiplicity)
            .contains(&column)
            .then_some((sector.dynkin_label, column - sector.first_global_column + 1))
    })
}

/// Audit the device-selected 52 d21 pivots and append the fixed d02 order.
/// The expected within-sector order is outer Fierz degree 0, then 3, then 4,
/// preserving each exact-RREF pivot order from the witness artifact.
pub fn audit_selected_column_order(bytes: &[u8]) -> Result<SelectedColumnAudit, String> {
    let artifact: WitnessArtifact =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid witness JSON: {error}"))?;
    if artifact.schema_version != "adynkra-11d-d21-gpu-witness-rank-v1" || !artifact.passed {
        return Err("D21 witness artifact is not authoritative".to_string());
    }
    if artifact
        .channels
        .iter()
        .map(|channel| channel.outer_fierz_degree)
        .collect::<Vec<_>>()
        != [0, 3, 4]
    {
        return Err("D21 witness channels are not in canonical Fierz order".to_string());
    }
    let mut bindings = Vec::with_capacity(D21_COLUMN_COUNT as usize);
    let mut seen = BTreeSet::new();
    for channel in artifact.channels {
        for sector in channel.sectors {
            if sector.per_prime_ranks != [sector.expected_rank; 3]
                || sector.selected_diagram_ordinals.len() != sector.expected_rank
            {
                return Err("D21 sector rank or selected diagram count drifted".to_string());
            }
            if sector.expected_rank == 0 {
                if sector
                    .global_columns
                    .as_deref()
                    .is_some_and(|columns| !columns.is_empty())
                {
                    return Err("rank-zero D21 sector has global columns".to_string());
                }
                continue;
            }
            if sector.exact_determinant_nonzero != Some(true) {
                return Err("D21 selected minor lacks exact nonzero determinant".to_string());
            }
            let columns = sector
                .global_columns
                .ok_or_else(|| "D21 selected sector lacks global columns".to_string())?;
            if columns.len() != sector.expected_rank {
                return Err("D21 global column count differs from exact rank".to_string());
            }
            for (column, diagram) in columns
                .into_iter()
                .zip(sector.selected_diagram_ordinals.into_iter())
            {
                let (expected_sector, copy) = expected_d21_identity(column)
                    .ok_or_else(|| "D21 global column is outside 0..52".to_string())?;
                if expected_sector != sector.sector || !seen.insert(column) {
                    return Err("D21 global column sector is wrong or duplicated".to_string());
                }
                bindings.push(SelectedColumnBinding {
                    global_column: column,
                    target_sector: sector.sector.clone(),
                    multiplicity_copy: copy,
                    outer_fierz_degree: channel.outer_fierz_degree,
                    diagram_ordinal: diagram,
                });
            }
        }
    }
    bindings.sort_by_key(|binding| binding.global_column);
    if bindings.len() != D21_COLUMN_COUNT as usize
        || bindings
            .iter()
            .enumerate()
            .any(|(expected, binding)| binding.global_column != expected as u32)
    {
        return Err("D21 selected bindings do not exhaust global columns 0..51".to_string());
    }
    for sector in D21_SECTORS {
        let degrees = bindings
            .iter()
            .filter(|binding| binding.target_sector == sector.dynkin_label)
            .map(|binding| binding.outer_fierz_degree)
            .collect::<Vec<_>>();
        if degrees.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("D21 copies are not ordered by Fierz degree within sector".to_string());
        }
    }
    Ok(SelectedColumnAudit {
        schema_version: SELECTED_COLUMN_AUDIT_SCHEMA_VERSION.to_string(),
        witness_artifact_sha256: format!("{:x}", Sha256::digest(bytes)),
        d21_bindings: bindings,
        d02_order: vec![
            "52:00001:copy1".to_string(),
            "53:01001:copy1".to_string(),
            "54:10001:copy1".to_string(),
            "55:10001:copy2".to_string(),
        ],
        passed: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn equality_binding(map: &str) -> EqualityLaunchBinding {
        EqualityLaunchBinding {
            schema_version: EQUALITY_LAUNCH_BINDING_SCHEMA_VERSION.to_string(),
            ordered_generator_map_sha256: map.to_string(),
            immutable_candidate_matrix_sha256: digest('1'),
            all_320_rhs_manifest_sha256: digest('2'),
            both_branch_join_sha256: digest('3'),
            arithmetic_parity_sha256: digest('4'),
            global57_reducer_sha256: digest('5'),
            immutable_candidate_matrix: true,
            all_320_rhs_complete: true,
            d21_and_d02_join_complete: true,
            exact_cpu_three_prime_parity: true,
            global57_reducer_ready: true,
            mutation_rejected: true,
        }
    }

    #[test]
    fn rhs_row_join_uses_lexicographic_spinor_pairs_and_symmetric_momenta() {
        let mut p = [0_u16; VECTOR_DIMENSION];
        p[3] = 1;
        let row = canonical_rhs_row(
            7,
            &FullChainRowKey {
                output_coordinate: 19,
                exterior_spinor_mask: (1_u32 << 0) | (1_u32 << 17),
                momentum_exponents: p,
            },
        )
        .unwrap();
        assert_eq!(row.branch, BidegreeBranch::D2P1);
        assert_eq!(row.source_coordinate, ((16 * 11 + 3) * 320 + 7) as u64);

        let mut p2 = [0_u16; VECTOR_DIMENSION];
        p2[2] = 1;
        p2[5] = 1;
        let row2 = canonical_rhs_row(
            7,
            &FullChainRowKey {
                output_coordinate: 19,
                exterior_spinor_mask: 0,
                momentum_exponents: p2,
            },
        )
        .unwrap();
        assert_eq!(row2.branch, BidegreeBranch::D0P2);
        assert_eq!(
            row2.source_coordinate,
            (symmetric_two_ordinal(2, 5).unwrap() * 320 + 7) as u64
        );
    }

    #[test]
    fn four_form_lexicographic_numeric_join_is_an_exhaustive_bijection() {
        let numeric = (0..FOUR_FORM_DIMENSION)
            .map(|lexicographic| {
                let numeric = lexicographic_four_form_to_numeric(lexicographic).unwrap();
                assert_eq!(
                    numeric_four_form_to_lexicographic(numeric).unwrap(),
                    lexicographic
                );
                numeric
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(numeric.len(), FOUR_FORM_DIMENSION);
        assert_eq!(numeric.iter().next().copied(), Some(0));
        assert_eq!(numeric.iter().next_back().copied(), Some(329));
        assert!(lexicographic_four_form_to_numeric(FOUR_FORM_DIMENSION).is_err());
        assert!(numeric_four_form_to_lexicographic(FOUR_FORM_DIMENSION).is_err());
    }

    #[test]
    fn corrected_right_c_teleparallel_column0_lands_entirely_in_declared_branches() {
        let report = teleparallel_rhs_column(0).unwrap();
        assert_eq!(report.schema_version, TELEPARALLEL_RHS_SCHEMA_VERSION);
        assert_eq!(report.entries.len(), 343_720);
        assert!(report.d21_entries > 0);
        assert!(report.d02_entries > 0);
        assert_eq!(report.d21_entries + report.d02_entries, 343_720);
        assert_eq!(report.stream_sha256.len(), 64);
        eprintln!(
            "teleparallel column0: d21={} d02={} sha256={}",
            report.d21_entries, report.d02_entries, report.stream_sha256
        );
        let batch = global57_rhs_batch(0, 0, 1).unwrap();
        assert_eq!(batch.entries.len(), 343_720);
        assert_eq!(batch.d21_entries, 342_640);
        assert_eq!(batch.d02_entries, 1_080);
        assert_eq!(batch.entries[0].column, AUGMENTED_TARGET_COLUMN);
        assert_eq!(
            batch.exact_value(batch.entries[0].row),
            Some(&batch.entries[0].coefficient)
        );
        assert!(batch.exact_value(u64::MAX).is_none());
        let modular = global57_rhs_three_prime(&batch.entries[0]).unwrap();
        assert!(!modular.is_zero_at_every_prime());
        let mut inadmissible = batch.entries[0].clone();
        inadmissible.coefficient.real_denominator = i64::from(PINNED_PRIMES[0]);
        assert!(global57_rhs_three_prime(&inadmissible).is_err());
        let bianchi = bianchi_rhs_image(&batch.entries).unwrap();
        eprintln!(
            "teleparallel column0 Bianchi residual rows={}",
            bianchi.len()
        );
        assert!(bianchi.is_empty());
        let wrong_basis = batch
            .entries
            .iter()
            .map(|entry| {
                let mut row = CanonicalRow::from_ordinal(entry.row).unwrap();
                let spinor = row.target_coordinate as usize / FOUR_FORM_DIMENSION;
                let numeric = row.target_coordinate as usize % FOUR_FORM_DIMENSION;
                let lexicographic = numeric_four_form_to_lexicographic(numeric).unwrap();
                row.target_coordinate = (spinor * FOUR_FORM_DIMENSION + lexicographic) as u32;
                let mut mutated = entry.clone();
                mutated.row = row.ordinal().unwrap();
                mutated
            })
            .collect::<Vec<_>>();
        assert_eq!(bianchi_rhs_image(&wrong_basis).unwrap().len(), 2_386_880);
        let mut wrong_column = batch.entries[..1].to_vec();
        wrong_column[0].column = 0;
        assert!(bianchi_rhs_image(&wrong_column).is_err());
    }

    #[test]
    fn bianchi_rows_round_trip_at_branch_boundaries() {
        for ordinal in [
            0,
            D21_BIANCHI_ROW_COUNT - 1,
            D21_BIANCHI_ROW_COUNT,
            TOTAL_BIANCHI_ROW_COUNT - 1,
        ] {
            let row = BianchiRowKey::from_ordinal(ordinal).unwrap();
            assert_eq!(row.ordinal().unwrap(), ordinal);
        }
        assert!(BianchiRowKey::from_ordinal(TOTAL_BIANCHI_ROW_COUNT).is_err());
    }

    #[test]
    fn exact_bianchi_image_cancels_and_mutations_survive() {
        let target = 0_u32;
        let d21 = CanonicalRow {
            branch: BidegreeBranch::D2P1,
            source_coordinate: 0,
            target_coordinate: target,
        }
        .ordinal()
        .unwrap();
        let cancelling = [
            ExactCooEntry {
                row: d21,
                column: 0,
                reserved: 0,
                real: 3,
                imaginary: -2,
            },
            ExactCooEntry {
                row: d21,
                column: 0,
                reserved: 0,
                real: -3,
                imaginary: 2,
            },
        ];
        assert!(bianchi_image(&cancelling, 5).unwrap().is_empty());
        let mutated = bianchi_image(&cancelling[..1], 5).unwrap();
        assert_eq!(mutated.len(), 7);
        assert!(
            mutated
                .windows(2)
                .all(|pair| (pair[0].row, pair[0].column) < (pair[1].row, pair[1].column))
        );
    }

    #[test]
    fn selected_global_column_artifact_is_exhaustive_and_mutation_fails() {
        let bytes = include_bytes!("../results/adynkra_11d_d21_gpu_witness_ranks.json");
        let report = audit_selected_column_order(bytes).unwrap();
        assert!(report.passed);
        assert_eq!(report.d21_bindings.len(), 52);
        assert_eq!(report.d21_bindings[0].global_column, 0);
        assert_eq!(report.d21_bindings[51].global_column, 51);
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["channels"][0]["sectors"][0]["global_columns"][0] = serde_json::json!(1);
        assert!(audit_selected_column_order(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn readiness_fails_closed_without_pbw_and_physical_k() {
        let map = digest('a');
        let equality = equality_binding(&map);
        let report = physics_assembly_readiness(Some(&equality), None, None);
        assert!(report.teleparallel_match_ready);
        assert!(report.bianchi_transform_ready);
        assert!(!report.pbw_integrability_ready);
        assert!(!report.source_gauge_descent_ready);
        assert!(report.raw_equality_solve_ready);
        assert!(!report.physical_descent_ready);
        assert!(report.launch_ready);
        assert!(report.equality_blockers.is_empty());
        assert_eq!(report.physical_promotion_blockers.len(), 2);

        let missing = physics_assembly_readiness(None, None, None);
        assert!(!missing.raw_equality_solve_ready);
        assert!(!missing.launch_ready);

        let mut partial = equality;
        partial.all_320_rhs_complete = false;
        let partial_report = physics_assembly_readiness(Some(&partial), None, None);
        assert!(!partial_report.raw_equality_solve_ready);
        assert!(!partial_report.launch_ready);
    }

    #[test]
    fn supplied_bindings_require_exact_gates_and_same_generator_map() {
        let map = digest('a');
        let pbw = PbwIntegrabilityBinding {
            schema_version: PBW_BINDING_SCHEMA_VERSION.to_string(),
            generator_map_sha256: map.clone(),
            source_graph_sha256: digest('b'),
            exact_matrix_sha256: digest('c'),
            exact_rows: 1,
            d21_to_d02_translation_complete: true,
            all_56_columns_bound: true,
            unrestricted_target_bianchi_residual_routed: true,
            mutation_rejected: true,
        };
        let gauge = SourceGaugeDescentBinding {
            schema_version: SOURCE_GAUGE_BINDING_SCHEMA_VERSION.to_string(),
            generator_map_sha256: map.clone(),
            complete_f_sha256: digest('d'),
            physical_k_sha256: digest('e'),
            quotient_normal_form_sha256: digest('f'),
            exact_matrix_sha256: digest('0'),
            exact_rows: 1,
            fk_zero_exact: true,
            all_six_source_channels: true,
            polynomial_witnesses_complete: true,
            quotient_normal_forms_replayed: true,
            mutation_rejected: true,
        };
        let equality = equality_binding(&map);
        let report = physics_assembly_readiness(Some(&equality), Some(&pbw), Some(&gauge));
        assert!(report.launch_ready);
        assert!(report.physical_descent_ready);
        let mut bad = gauge;
        bad.fk_zero_exact = false;
        let bad_report = physics_assembly_readiness(Some(&equality), Some(&pbw), Some(&bad));
        assert!(bad_report.launch_ready);
        assert!(!bad_report.physical_descent_ready);
    }

    #[test]
    fn constants_bind_the_existing_candidate_grid() {
        assert_eq!(D21_COLUMN_COUNT + D02_COLUMN_COUNT, 56);
        assert_eq!(D21_ROW_COUNT + D02_ROW_COUNT, 18_659_942_400);
        assert_eq!(D_G4_COORDINATES, 10_560);
        assert_eq!(D21_SOURCE_COORDINATES, 1_745_920);
        assert_eq!(D02_SOURCE_COORDINATES, 21_120);
        assert_eq!(D21_BIANCHI_SOURCE_DIMENSION, 10_475_520);
        assert_eq!(D02_BIANCHI_SOURCE_DIMENSION, 91_520);
    }
}
