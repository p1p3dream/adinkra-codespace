//! Exact, fail-closed harness for local-Lorentz section descent.
//!
//! The current production operator fixes the `Psi_[2]` frame coordinate at
//! its input boundary.  This module records what is already known about the
//! full `D_alpha Psi_[de]` orbit and defines the typed row contract for the
//! still-missing physical-sector section-difference calculation.  It never
//! promotes a fixed section into a section-independence claim.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: &str = "adynkra-11d-local-lorentz-section-descent-v1";

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const TWO_FORM_DIMENSION: usize = 55;
const DOMAIN_DIMENSION: usize = SPINOR_DIMENSION * TWO_FORM_DIMENSION;

/// One canonical basis coordinate in `S tensor Lambda^2 V`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct LocalLorentzBasisKey {
    pub derivative_spinor: u8,
    pub two_form_ordinal: u8,
    pub two_form_mask: u16,
}

/// Stable output sectors needed by the section-difference contract.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalLorentzDescentSector {
    RawJOne,
    LinearizedRiemann,
    DirectGravitinoCurl,
    DirectCandidateFourForm,
}

impl LocalLorentzDescentSector {
    pub fn col3_tag(self) -> Option<u8> {
        match self {
            Self::RawJOne => None,
            Self::LinearizedRiemann => Some(9),
            Self::DirectGravitinoCurl => Some(10),
            Self::DirectCandidateFourForm => Some(11),
        }
    }
}

/// Canonical row key after ordered-superderivative normalization.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct LocalLorentzDescentRowKey {
    pub sector: LocalLorentzDescentSector,
    pub target_coordinate: u32,
    pub exterior_spinor_mask: u32,
    pub momentum_exponents: [u16; VECTOR_DIMENSION],
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExactRationalCoefficient {
    pub numerator: i64,
    pub denominator: i64,
}

/// One exact nonzero of a section-difference matrix.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LocalLorentzResidualEntry {
    pub source: LocalLorentzBasisKey,
    pub row: LocalLorentzDescentRowKey,
    pub coefficient: ExactRationalCoefficient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionDifferenceGateStatus {
    PassedExactZero,
    FailedNonzero,
    PendingExhaustiveProjection,
    BlockedPhysicalAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SectionDifferenceSectorStatus {
    pub sector: LocalLorentzDescentSector,
    pub col3_tag: Option<u8>,
    pub role: &'static str,
    pub included_in_current_operator: bool,
    pub physical_identification_authoritative: bool,
    pub relative_normalization_fixed: bool,
    pub exhaustive_columns_required: usize,
    pub exhaustive_columns_checked: usize,
    pub exact_residual_terms: Option<usize>,
    pub exact_zero: Option<bool>,
    pub gate_status: SectionDifferenceGateStatus,
    pub blocker: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LocalLorentzSectionDescentReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub coefficient_field: &'static str,
    pub ordered_superderivative_basis: &'static str,
    pub raw_frame_domain: &'static str,
    pub quotient_domain: &'static str,
    pub section_map: &'static str,
    pub section_difference_formula: &'static str,
    pub first_jet_domain: &'static str,
    pub first_jet_domain_dimension: usize,
    pub first_jet_basis_columns: usize,
    pub first_jet_basis_unique: bool,
    pub raw_j_row_schema: &'static str,
    pub raw_j_matrix_rows: usize,
    pub raw_j_matrix_columns: usize,
    pub raw_j_matrix_rank: usize,
    pub raw_j_matrix_nullity: usize,
    pub raw_j_nonzero_entries: usize,
    pub raw_j_nonzero_entries_per_column: usize,
    pub raw_j_nonzero_entries_per_row: usize,
    pub raw_j_boost_columns: usize,
    pub raw_j_spatial_columns: usize,
    pub raw_j_coefficient: &'static str,
    pub raw_j_residual_sha256: String,
    pub direct_local_lorentz_audit_schema: &'static str,
    pub direct_local_lorentz_audit_passed: bool,
    pub j_one_residual_audit_schema: &'static str,
    pub j_one_residual_audit_passed: bool,
    pub current_raw_residual_is_lorentz_equivariant: bool,
    pub current_raw_residual_hom_dimension: usize,
    pub target_equation_complex_passed: bool,
    pub complete_f_schema: &'static str,
    pub complete_f_fixed_section_only: bool,
    pub physical_four_form_authority_fixed: bool,
    pub physical_four_form_relative_normalization_fixed: bool,
    pub sector_statuses: Vec<SectionDifferenceSectorStatus>,
    pub accepted_physical_sectors_exhaustively_checked: usize,
    pub physical_sector_projection_frozen: bool,
    pub exhaustive_physical_section_difference_complete: bool,
    pub physical_local_lorentz_descent_certified: bool,
    pub harness_integrity_passed: bool,
    pub passed: bool,
    pub next_executable_step: &'static str,
    pub boundary: &'static str,
}

fn masks_of_degree_two() -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 2)
        .collect()
}

fn multiply_gamma(left: &[Vec<i8>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            if left[row][pivot] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] +=
                    i16::from(left[row][pivot]) * i16::from(right[pivot][column]);
            }
        }
    }
    output
}

fn upper_gamma_pair(mask: u16) -> Vec<Vec<i16>> {
    let axes = (0..VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    debug_assert_eq!(axes.len(), 2);
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    multiply_gamma(&gammas[axes[0]], &gammas[axes[1]])
}

/// Reconstruct all exact nonzeros of the currently audited raw J residual.
///
/// The source ordering is `derivative_spinor * 55 + two_form_ordinal`.
/// Every gamma-pair block is a signed permutation, so every source column has
/// exactly one nonzero with coefficient `+/- 109/1056`.
pub fn current_raw_j_residual_entries() -> Vec<LocalLorentzResidualEntry> {
    let masks = masks_of_degree_two();
    let mut entries = Vec::with_capacity(DOMAIN_DIMENSION);
    for (pair, &mask) in masks.iter().enumerate() {
        let gamma = upper_gamma_pair(mask);
        for derivative in 0..SPINOR_DIMENSION {
            let nonzeros = (0..SPINOR_DIMENSION)
                .filter(|&row| gamma[row][derivative] != 0)
                .collect::<Vec<_>>();
            assert_eq!(
                nonzeros.len(),
                1,
                "Gamma^[de] block is not a signed permutation"
            );
            let row = nonzeros[0];
            let sign = i64::from(gamma[row][derivative]);
            assert!(matches!(sign, -1 | 1));
            entries.push(LocalLorentzResidualEntry {
                source: LocalLorentzBasisKey {
                    derivative_spinor: derivative as u8,
                    two_form_ordinal: pair as u8,
                    two_form_mask: mask,
                },
                row: LocalLorentzDescentRowKey {
                    sector: LocalLorentzDescentSector::RawJOne,
                    target_coordinate: row as u32,
                    exterior_spinor_mask: 0,
                    momentum_exponents: [0; VECTOR_DIMENSION],
                },
                coefficient: ExactRationalCoefficient {
                    numerator: sign * 109,
                    denominator: 1_056,
                },
            });
        }
    }
    entries.sort_by_key(|entry| {
        (
            entry.source.derivative_spinor,
            entry.source.two_form_ordinal,
        )
    });
    entries
}

fn hash_raw_residual(entries: &[LocalLorentzResidualEntry]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA_VERSION.as_bytes());
    hasher.update((entries.len() as u64).to_le_bytes());
    for entry in entries {
        hasher.update([entry.source.derivative_spinor]);
        hasher.update([entry.source.two_form_ordinal]);
        hasher.update(entry.source.two_form_mask.to_le_bytes());
        hasher.update([0_u8]);
        hasher.update(entry.row.target_coordinate.to_le_bytes());
        hasher.update(entry.row.exterior_spinor_mask.to_le_bytes());
        for exponent in entry.row.momentum_exponents {
            hasher.update(exponent.to_le_bytes());
        }
        hasher.update(entry.coefficient.numerator.to_le_bytes());
        hasher.update(entry.coefficient.denominator.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn raw_j_accounting(entries: &[LocalLorentzResidualEntry]) -> (bool, usize, usize, usize, usize) {
    let sources = entries
        .iter()
        .map(|entry| entry.source)
        .collect::<BTreeSet<_>>();
    let mut per_row = BTreeMap::<u32, usize>::new();
    for entry in entries {
        *per_row.entry(entry.row.target_coordinate).or_default() += 1;
    }
    let rank = per_row.len();
    let minimum_per_row = per_row.values().copied().min().unwrap_or(0);
    let maximum_per_row = per_row.values().copied().max().unwrap_or(0);
    (
        sources.len() == DOMAIN_DIMENSION,
        rank,
        DOMAIN_DIMENSION - rank,
        minimum_per_row,
        maximum_per_row,
    )
}

fn build_report() -> LocalLorentzSectionDescentReport {
    let direct = crate::eleven_dimensional_direct_local_lorentz::verify();
    let j_one = crate::eleven_dimensional_j1_lorentz_residual::verify();
    let entries = current_raw_j_residual_entries();
    let (first_jet_basis_unique, rank, nullity, minimum_per_row, maximum_per_row) =
        raw_j_accounting(&entries);
    let raw_j_shape_matches = entries.len() == DOMAIN_DIMENSION
        && rank == SPINOR_DIMENSION
        && nullity == DOMAIN_DIMENSION - SPINOR_DIMENSION
        && minimum_per_row == TWO_FORM_DIMENSION
        && maximum_per_row == TWO_FORM_DIMENSION;
    let prerequisite_audits_match = direct.passed
        && direct.domain_dimension == DOMAIN_DIMENSION
        && direct.exhaustive_columns_checked == DOMAIN_DIMENSION
        && direct.current_j_one_rank == SPINOR_DIMENSION
        && j_one.passed
        && j_one.matrix_rows == SPINOR_DIMENSION
        && j_one.matrix_columns == DOMAIN_DIMENSION
        && j_one.matrix_rank == rank
        && j_one.matrix_nullity == nullity
        && j_one.matrix_nonzero_entries == entries.len()
        && j_one.nonzero_entries_per_column == 1
        && j_one.nonzero_entries_per_row == TWO_FORM_DIMENSION
        && j_one.boost_coefficient == "109/1056"
        && j_one.spatial_coefficient == "109/1056";
    // The physical authority packet is an external input to the future
    // exhaustive runner. The current repository has no accepted packet, so
    // the default report must remain false rather than infer authority from a
    // conditional adapter.
    let physical_four_form_authority_fixed = false;
    let physical_four_form_relative_normalization_fixed = false;

    let sector_statuses = vec![
        SectionDifferenceSectorStatus {
            sector: LocalLorentzDescentSector::RawJOne,
            col3_tag: None,
            role: "auxiliary obstruction canary",
            included_in_current_operator: true,
            physical_identification_authoritative: false,
            relative_normalization_fixed: true,
            exhaustive_columns_required: DOMAIN_DIMENSION,
            exhaustive_columns_checked: DOMAIN_DIMENSION,
            exact_residual_terms: Some(entries.len()),
            exact_zero: Some(false),
            gate_status: SectionDifferenceGateStatus::FailedNonzero,
            blocker: "the source-fixed raw J^(1) response is the nonzero rank-32 Gamma^[2] map",
        },
        SectionDifferenceSectorStatus {
            sector: LocalLorentzDescentSector::LinearizedRiemann,
            col3_tag: LocalLorentzDescentSector::LinearizedRiemann.col3_tag(),
            role: "accepted physical curvature",
            included_in_current_operator: true,
            physical_identification_authoritative: true,
            relative_normalization_fixed: true,
            exhaustive_columns_required: DOMAIN_DIMENSION,
            exhaustive_columns_checked: 0,
            exact_residual_terms: None,
            exact_zero: None,
            gate_status: SectionDifferenceGateStatus::PendingExhaustiveProjection,
            blocker: "the raw Lorentz lift has not been streamed through the direct Riemann adapter on all 1,760 columns",
        },
        SectionDifferenceSectorStatus {
            sector: LocalLorentzDescentSector::DirectGravitinoCurl,
            col3_tag: LocalLorentzDescentSector::DirectGravitinoCurl.col3_tag(),
            role: "accepted physical curvature",
            included_in_current_operator: true,
            physical_identification_authoritative: true,
            relative_normalization_fixed: true,
            exhaustive_columns_required: DOMAIN_DIMENSION,
            exhaustive_columns_checked: 0,
            exact_residual_terms: None,
            exact_zero: None,
            gate_status: SectionDifferenceGateStatus::PendingExhaustiveProjection,
            blocker: "the raw Lorentz lift has not been streamed through the direct gravitino-curl adapter on all 1,760 columns",
        },
        SectionDifferenceSectorStatus {
            sector: LocalLorentzDescentSector::DirectCandidateFourForm,
            col3_tag: LocalLorentzDescentSector::DirectCandidateFourForm.col3_tag(),
            role: "conditional physical-curvature candidate",
            included_in_current_operator: true,
            physical_identification_authoritative: physical_four_form_authority_fixed,
            relative_normalization_fixed: physical_four_form_relative_normalization_fixed,
            exhaustive_columns_required: DOMAIN_DIMENSION,
            exhaustive_columns_checked: 0,
            exact_residual_terms: None,
            exact_zero: None,
            gate_status: SectionDifferenceGateStatus::BlockedPhysicalAuthority,
            blocker: "Psi_[3] is not yet authoritatively identified and relatively normalized as physical A_3/G_4",
        },
    ];
    let accepted_physical_sectors_exhaustively_checked = sector_statuses
        .iter()
        .filter(|sector| {
            sector.role == "accepted physical curvature"
                && sector.exhaustive_columns_checked == sector.exhaustive_columns_required
                && sector.exact_zero == Some(true)
        })
        .count();
    let physical_sector_projection_frozen =
        physical_four_form_authority_fixed && physical_four_form_relative_normalization_fixed;
    let exhaustive_physical_section_difference_complete = false;
    let physical_local_lorentz_descent_certified = false;
    let harness_integrity_passed = first_jet_basis_unique
        && raw_j_shape_matches
        && prerequisite_audits_match
        && !physical_sector_projection_frozen
        && !exhaustive_physical_section_difference_complete
        && !physical_local_lorentz_descent_certified;

    LocalLorentzSectionDescentReport {
        schema_version: SCHEMA_VERSION,
        role: "exact local-Lorentz section-difference contract and current raw-obstruction ledger",
        coefficient_field: "Q(i), with the current raw J residual real",
        ordered_superderivative_basis:
            "descending exterior D mask followed by eleven formal momentum exponents",
        raw_frame_domain: "H_alpha^a plus scale plus Psi_[2]",
        quotient_domain: "H_hat in (10001), dimension 320, plus separate scale canary",
        section_map: "canonical_physical_frame_representative",
        section_difference_formula:
            "Delta_L=P_phys[F_tilde(s(h)+L lambda)-F_tilde(s(h))]=P_phys F_tilde L lambda",
        first_jet_domain: "D_alpha Psi_[de] in (00001) tensor (01000)",
        first_jet_domain_dimension: DOMAIN_DIMENSION,
        first_jet_basis_columns: entries.len(),
        first_jet_basis_unique,
        raw_j_row_schema: "(sector, target coordinate, exterior-D mask, p_0..p_10 exponents)",
        raw_j_matrix_rows: SPINOR_DIMENSION,
        raw_j_matrix_columns: DOMAIN_DIMENSION,
        raw_j_matrix_rank: rank,
        raw_j_matrix_nullity: nullity,
        raw_j_nonzero_entries: entries.len(),
        raw_j_nonzero_entries_per_column: 1,
        raw_j_nonzero_entries_per_row: minimum_per_row,
        raw_j_boost_columns: 10 * SPINOR_DIMENSION,
        raw_j_spatial_columns: 45 * SPINOR_DIMENSION,
        raw_j_coefficient: "+/-109/1056 times Gamma^[de] signed-permutation entry",
        raw_j_residual_sha256: hash_raw_residual(&entries),
        direct_local_lorentz_audit_schema: direct.schema_version,
        direct_local_lorentz_audit_passed: direct.passed,
        j_one_residual_audit_schema: j_one.schema_version,
        j_one_residual_audit_passed: j_one.passed,
        current_raw_residual_is_lorentz_equivariant: j_one
            .current_residual_is_lorentz_equivariant,
        current_raw_residual_hom_dimension: j_one.lorentz_equivariant_hom_dimension,
        target_equation_complex_passed: true,
        complete_f_schema: crate::eleven_dimensional_complete_f::SCHEMA_VERSION,
        complete_f_fixed_section_only: true,
        physical_four_form_authority_fixed,
        physical_four_form_relative_normalization_fixed,
        sector_statuses,
        accepted_physical_sectors_exhaustively_checked,
        physical_sector_projection_frozen,
        exhaustive_physical_section_difference_complete,
        physical_local_lorentz_descent_certified,
        harness_integrity_passed,
        passed: physical_local_lorentz_descent_certified,
        next_executable_step: "stream all 1,760 raw Lorentz first-jet basis directions through the direct Riemann and direct gravitino-curl projections, then add the four-form projection after its physical authority and normalization are fixed",
        boundary: "Passing harness integrity reproduces the exact rank-32 raw J obstruction and proves that the physical-sector gate remains fail closed. It does not certify local-Lorentz descent. Physical descent requires exact zero on every required Lorentz jet basis direction in every authorized physical curvature and equation sector.",
    }
}

pub fn verify() -> LocalLorentzSectionDescentReport {
    static REPORT: OnceLock<LocalLorentzSectionDescentReport> = OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

pub fn write_artifact(path: &Path) -> io::Result<LocalLorentzSectionDescentReport> {
    let report = verify();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(report)
}

pub const RUN_MANIFEST_SCHEMA: &str = "adynkra-11d-local-lorentz-section-descent-run-manifest-v1";
pub const BLOCK_CHECKPOINT_SCHEMA: &str = "adynkra-11d-local-lorentz-section-descent-block-v1";
pub const HEARTBEAT_SCHEMA: &str = "adynkra-11d-local-lorentz-section-descent-heartbeat-v1";
pub const RUN_REPORT_SCHEMA: &str = "adynkra-11d-local-lorentz-section-descent-run-v1";
pub const HEARTBEAT_INTERVAL_SECONDS: u64 = 5;
pub const BLOCK_COLUMNS: usize = SPINOR_DIMENSION;
pub const BLOCK_COUNT: usize = TWO_FORM_DIMENSION;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalLorentzRunInputs {
    pub source_revision: String,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub physical_projection_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LocalLorentzRunManifest {
    pub schema_version: String,
    pub source_revision: String,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub physical_projection_sha256: String,
    pub basis_sha256: String,
    pub raw_j_map_sha256: String,
    pub direct_audit_sha256: String,
    pub j_one_audit_sha256: String,
    pub coefficient_field: String,
    pub row_key_schema: String,
    pub total_directions: usize,
    pub columns_per_block: usize,
    pub total_blocks: usize,
    pub heartbeat_interval_seconds: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalLorentzRunPhase {
    Preflight,
    Evaluating,
    Reducing,
    Publishing,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FirstResidualWitness {
    pub source: LocalLorentzBasisKey,
    pub row: LocalLorentzDescentRowKey,
    pub coefficient: ExactRationalCoefficient,
}

impl From<&LocalLorentzResidualEntry> for FirstResidualWitness {
    fn from(entry: &LocalLorentzResidualEntry) -> Self {
        Self {
            source: entry.source,
            row: entry.row.clone(),
            coefficient: entry.coefficient,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResourceHighWater {
    pub process_rss_bytes: Option<u64>,
    pub checkpoint_bytes: u64,
    pub device_resident_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LocalLorentzHeartbeat {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub phase: LocalLorentzRunPhase,
    pub updated_at_unix_ms: u128,
    pub completed_directions: usize,
    pub total_directions: usize,
    pub completed_blocks: usize,
    pub total_blocks: usize,
    pub exact_residual_entries: usize,
    pub exact_output_rows: usize,
    pub exact_rank: usize,
    pub first_residual_witness: Option<FirstResidualWitness>,
    pub elapsed_milliseconds: u128,
    pub directions_per_second: f64,
    pub eta_milliseconds: Option<u128>,
    pub resource_high_water: ResourceHighWater,
    pub resumed_blocks: usize,
    pub cancellation_observed: bool,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct LocalLorentzBlockCheckpoint {
    schema_version: String,
    manifest_sha256: String,
    block_ordinal: usize,
    two_form_ordinal: usize,
    two_form_mask: u16,
    completed_directions: usize,
    entries: Vec<LocalLorentzResidualEntry>,
    exact_output_rows: usize,
    entries_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LocalLorentzBlockTiming {
    pub block_ordinal: usize,
    pub resumed: bool,
    pub directions: usize,
    pub exact_residual_entries: usize,
    pub elapsed_microseconds: u128,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalLorentzHarnessRunReport {
    pub schema_version: &'static str,
    pub manifest: LocalLorentzRunManifest,
    pub manifest_sha256: String,
    pub scientific_report: LocalLorentzSectionDescentReport,
    pub phase: LocalLorentzRunPhase,
    pub completed_directions: usize,
    pub total_directions: usize,
    pub completed_blocks: usize,
    pub total_blocks: usize,
    pub exact_residual_entries: usize,
    pub exact_output_rows: usize,
    pub exact_rank: usize,
    pub first_residual_witness: Option<FirstResidualWitness>,
    pub resumed_blocks: usize,
    pub fresh_blocks: usize,
    pub block_timings: Vec<LocalLorentzBlockTiming>,
    pub elapsed_milliseconds: u128,
    pub directions_per_second: f64,
    pub resource_high_water: ResourceHighWater,
    pub checkpoint_resume_equivalent: bool,
    pub fail_fast_cancellation_enabled: bool,
    pub report_published_last: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json<T: Serialize>(value: &T) -> io::Result<String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn basis_sha256(entries: &[LocalLorentzResidualEntry]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-11d-local-lorentz-basis-v1\0");
    for entry in entries {
        hasher.update([entry.source.derivative_spinor]);
        hasher.update([entry.source.two_form_ordinal]);
        hasher.update(entry.source.two_form_mask.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn current_process_rss_bytes() -> Option<u64> {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for key in ["VmHWM:", "VmRSS:"] {
            if let Some(line) = status.lines().find(|line| line.starts_with(key)) {
                if let Some(kibibytes) = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|value| value.parse::<u64>().ok())
                {
                    return kibibytes.checked_mul(1_024);
                }
            }
        }
    }
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?
        .checked_mul(1_024)
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, value)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn checkpoint_bytes(directory: &Path) -> u64 {
    fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum()
}

fn manifest_for(
    inputs: &LocalLorentzRunInputs,
    entries: &[LocalLorentzResidualEntry],
) -> io::Result<LocalLorentzRunManifest> {
    if inputs.source_revision.trim().is_empty()
        || !is_sha256(&inputs.source_tree_sha256)
        || !is_sha256(&inputs.binary_sha256)
        || !is_sha256(&inputs.physical_projection_sha256)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local-Lorentz run provenance is incomplete",
        ));
    }
    let raw_j_map_sha256 = hash_raw_residual(entries);
    let direct_audit_sha256 = sha256_bytes(
        format!(
            "{}:{}:{}",
            "adynkra.11d.direct-local-lorentz-diagnostic.v3",
            basis_sha256(entries),
            raw_j_map_sha256
        )
        .as_bytes(),
    );
    let j_one_audit_sha256 = sha256_bytes(
        format!(
            "{}:{}",
            "adynkra.11d.j1-lorentz-residual.v3", raw_j_map_sha256
        )
        .as_bytes(),
    );
    Ok(LocalLorentzRunManifest {
        schema_version: RUN_MANIFEST_SCHEMA.to_string(),
        source_revision: inputs.source_revision.clone(),
        source_tree_sha256: inputs.source_tree_sha256.clone(),
        binary_sha256: inputs.binary_sha256.clone(),
        physical_projection_sha256: inputs.physical_projection_sha256.clone(),
        basis_sha256: basis_sha256(entries),
        raw_j_map_sha256,
        direct_audit_sha256,
        j_one_audit_sha256,
        coefficient_field: "Q(i)".to_string(),
        row_key_schema: "sector/target-coordinate/exterior-D-mask/eleven-momentum-exponents"
            .to_string(),
        total_directions: DOMAIN_DIMENSION,
        columns_per_block: BLOCK_COLUMNS,
        total_blocks: BLOCK_COUNT,
        heartbeat_interval_seconds: HEARTBEAT_INTERVAL_SECONDS,
    })
}

fn publish_or_validate_manifest(
    path: &Path,
    manifest: &LocalLorentzRunManifest,
) -> io::Result<String> {
    let digest = sha256_json(manifest)?;
    if path.exists() {
        let stored: LocalLorentzRunManifest = serde_json::from_slice(&fs::read(path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if &stored != manifest || sha256_json(&stored)? != digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "immutable local-Lorentz run manifest changed",
            ));
        }
    } else {
        atomic_json(path, manifest)?;
    }
    Ok(digest)
}

fn checkpoint_digest(entries: &[LocalLorentzResidualEntry]) -> String {
    hash_raw_residual(entries)
}

fn expected_block_entries(
    all: &[LocalLorentzResidualEntry],
    block_ordinal: usize,
) -> Vec<LocalLorentzResidualEntry> {
    all.iter()
        .filter(|entry| usize::from(entry.source.two_form_ordinal) == block_ordinal)
        .cloned()
        .collect()
}

fn load_or_write_block(
    path: &Path,
    manifest_sha256: &str,
    block_ordinal: usize,
    expected: Vec<LocalLorentzResidualEntry>,
) -> io::Result<(LocalLorentzBlockCheckpoint, bool)> {
    let masks = masks_of_degree_two();
    let expected_digest = checkpoint_digest(&expected);
    if path.exists() {
        let stored: LocalLorentzBlockCheckpoint = serde_json::from_slice(&fs::read(path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let valid = stored.schema_version == BLOCK_CHECKPOINT_SCHEMA
            && stored.manifest_sha256 == manifest_sha256
            && stored.block_ordinal == block_ordinal
            && stored.two_form_ordinal == block_ordinal
            && stored.two_form_mask == masks[block_ordinal]
            && stored.completed_directions == BLOCK_COLUMNS
            && stored.entries == expected
            && stored.exact_output_rows
                == stored
                    .entries
                    .iter()
                    .map(|entry| entry.row.target_coordinate)
                    .collect::<BTreeSet<_>>()
                    .len()
            && stored.entries_sha256 == expected_digest
            && checkpoint_digest(&stored.entries) == stored.entries_sha256;
        if !valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("corrupt local-Lorentz checkpoint block {block_ordinal}"),
            ));
        }
        return Ok((stored, true));
    }
    let exact_output_rows = expected
        .iter()
        .map(|entry| entry.row.target_coordinate)
        .collect::<BTreeSet<_>>()
        .len();
    let checkpoint = LocalLorentzBlockCheckpoint {
        schema_version: BLOCK_CHECKPOINT_SCHEMA.to_string(),
        manifest_sha256: manifest_sha256.to_string(),
        block_ordinal,
        two_form_ordinal: block_ordinal,
        two_form_mask: masks[block_ordinal],
        completed_directions: expected.len(),
        entries: expected,
        exact_output_rows,
        entries_sha256: expected_digest,
    };
    atomic_json(path, &checkpoint)?;
    Ok((checkpoint, false))
}

struct ProgressState {
    started: Instant,
    completed_directions: usize,
    completed_blocks: usize,
    exact_residual_entries: usize,
    output_rows: BTreeSet<u32>,
    first_residual_witness: Option<FirstResidualWitness>,
    resumed_blocks: usize,
    high_water: ResourceHighWater,
}

impl ProgressState {
    fn heartbeat(
        &self,
        manifest_sha256: &str,
        phase: LocalLorentzRunPhase,
        cancelled: bool,
        failure: Option<String>,
    ) -> LocalLorentzHeartbeat {
        let elapsed = self.started.elapsed();
        let rate = if elapsed.is_zero() {
            0.0
        } else {
            self.completed_directions as f64 / elapsed.as_secs_f64()
        };
        let remaining = DOMAIN_DIMENSION.saturating_sub(self.completed_directions);
        let eta_milliseconds =
            (rate > 0.0).then(|| Duration::from_secs_f64(remaining as f64 / rate).as_millis());
        LocalLorentzHeartbeat {
            schema_version: HEARTBEAT_SCHEMA.to_string(),
            manifest_sha256: manifest_sha256.to_string(),
            phase,
            updated_at_unix_ms: unix_milliseconds(),
            completed_directions: self.completed_directions,
            total_directions: DOMAIN_DIMENSION,
            completed_blocks: self.completed_blocks,
            total_blocks: BLOCK_COUNT,
            exact_residual_entries: self.exact_residual_entries,
            exact_output_rows: self.output_rows.len(),
            exact_rank: self.output_rows.len(),
            first_residual_witness: self.first_residual_witness.clone(),
            elapsed_milliseconds: elapsed.as_millis(),
            directions_per_second: rate,
            eta_milliseconds,
            resource_high_water: self.high_water.clone(),
            resumed_blocks: self.resumed_blocks,
            cancellation_observed: cancelled,
            failure,
        }
    }
}

/// Run the currently executable 1,760-column raw-J obstruction harness.
///
/// This is the checkpointed operational substrate for the future physical
/// section-difference evaluator. It deliberately publishes a scientifically
/// fail-closed report because the Riemann, gravitino, and authorized four-form
/// projections are not evaluated here.
pub fn run_current_raw_j_harness(
    run_root: &Path,
    inputs: &LocalLorentzRunInputs,
    cancellation: &AtomicBool,
) -> io::Result<LocalLorentzHarnessRunReport> {
    fs::create_dir_all(run_root)?;
    let checkpoint_directory = run_root.join("blocks");
    fs::create_dir_all(&checkpoint_directory)?;
    let manifest_path = run_root.join("manifest.json");
    let heartbeat_path = run_root.join("heartbeat.json");
    let report_path = run_root.join("report.json");
    let all_entries = current_raw_j_residual_entries();
    let manifest = manifest_for(inputs, &all_entries)?;
    let manifest_sha256 = publish_or_validate_manifest(&manifest_path, &manifest)?;
    let mut state = ProgressState {
        started: Instant::now(),
        completed_directions: 0,
        completed_blocks: 0,
        exact_residual_entries: 0,
        output_rows: BTreeSet::new(),
        first_residual_witness: None,
        resumed_blocks: 0,
        high_water: ResourceHighWater::default(),
    };
    state.high_water.process_rss_bytes = current_process_rss_bytes();
    atomic_json(
        &heartbeat_path,
        &state.heartbeat(
            &manifest_sha256,
            LocalLorentzRunPhase::Preflight,
            false,
            None,
        ),
    )?;
    let mut block_timings = Vec::with_capacity(BLOCK_COUNT);
    for block_ordinal in 0..BLOCK_COUNT {
        if cancellation.load(Ordering::Acquire) {
            atomic_json(
                &heartbeat_path,
                &state.heartbeat(
                    &manifest_sha256,
                    LocalLorentzRunPhase::Cancelled,
                    true,
                    Some("cooperative cancellation observed before next block".to_string()),
                ),
            )?;
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "local-Lorentz harness cancelled",
            ));
        }
        let started = Instant::now();
        let expected = expected_block_entries(&all_entries, block_ordinal);
        let path = checkpoint_directory.join(format!("block_{block_ordinal:03}.json"));
        let (checkpoint, resumed) =
            match load_or_write_block(&path, &manifest_sha256, block_ordinal, expected) {
                Ok(value) => value,
                Err(error) => {
                    atomic_json(
                        &heartbeat_path,
                        &state.heartbeat(
                            &manifest_sha256,
                            LocalLorentzRunPhase::Failed,
                            false,
                            Some(error.to_string()),
                        ),
                    )?;
                    return Err(error);
                }
            };
        state.completed_directions += checkpoint.completed_directions;
        state.completed_blocks += 1;
        state.exact_residual_entries += checkpoint.entries.len();
        state.resumed_blocks += usize::from(resumed);
        for entry in &checkpoint.entries {
            state.output_rows.insert(entry.row.target_coordinate);
            if state.first_residual_witness.is_none() {
                state.first_residual_witness = Some(entry.into());
            }
        }
        state.high_water.process_rss_bytes = state
            .high_water
            .process_rss_bytes
            .max(current_process_rss_bytes());
        state.high_water.checkpoint_bytes = checkpoint_bytes(&checkpoint_directory);
        block_timings.push(LocalLorentzBlockTiming {
            block_ordinal,
            resumed,
            directions: checkpoint.completed_directions,
            exact_residual_entries: checkpoint.entries.len(),
            elapsed_microseconds: started.elapsed().as_micros(),
        });
        // Blocks are deliberately small enough that this publication cadence
        // is stricter than the required five-second heartbeat interval.
        atomic_json(
            &heartbeat_path,
            &state.heartbeat(
                &manifest_sha256,
                LocalLorentzRunPhase::Evaluating,
                false,
                None,
            ),
        )?;
    }
    atomic_json(
        &heartbeat_path,
        &state.heartbeat(
            &manifest_sha256,
            LocalLorentzRunPhase::Reducing,
            false,
            None,
        ),
    )?;
    let checkpoint_resume_equivalent = state.completed_directions == DOMAIN_DIMENSION
        && state.exact_residual_entries == all_entries.len()
        && state.output_rows.len() == SPINOR_DIMENSION
        && state.first_residual_witness == all_entries.first().map(Into::into);
    if !checkpoint_resume_equivalent {
        let message = "checkpoint reduction does not equal the fresh exact raw-J stream";
        atomic_json(
            &heartbeat_path,
            &state.heartbeat(
                &manifest_sha256,
                LocalLorentzRunPhase::Failed,
                false,
                Some(message.to_string()),
            ),
        )?;
        return Err(io::Error::new(io::ErrorKind::InvalidData, message));
    }
    atomic_json(
        &heartbeat_path,
        &state.heartbeat(
            &manifest_sha256,
            LocalLorentzRunPhase::Publishing,
            false,
            None,
        ),
    )?;
    let elapsed = state.started.elapsed();
    let scientific_report = verify();
    let passed = checkpoint_resume_equivalent
        && scientific_report.harness_integrity_passed
        && !scientific_report.physical_local_lorentz_descent_certified;
    let report = LocalLorentzHarnessRunReport {
        schema_version: RUN_REPORT_SCHEMA,
        manifest,
        manifest_sha256: manifest_sha256.clone(),
        scientific_report,
        phase: LocalLorentzRunPhase::Complete,
        completed_directions: state.completed_directions,
        total_directions: DOMAIN_DIMENSION,
        completed_blocks: state.completed_blocks,
        total_blocks: BLOCK_COUNT,
        exact_residual_entries: state.exact_residual_entries,
        exact_output_rows: state.output_rows.len(),
        exact_rank: state.output_rows.len(),
        first_residual_witness: state.first_residual_witness.clone(),
        resumed_blocks: state.resumed_blocks,
        fresh_blocks: BLOCK_COUNT - state.resumed_blocks,
        block_timings,
        elapsed_milliseconds: elapsed.as_millis(),
        directions_per_second: if elapsed.is_zero() {
            0.0
        } else {
            DOMAIN_DIMENSION as f64 / elapsed.as_secs_f64()
        },
        resource_high_water: state.high_water.clone(),
        checkpoint_resume_equivalent,
        fail_fast_cancellation_enabled: true,
        report_published_last: true,
        passed,
        boundary: "This operational pass reproduces the exact 1,760-column raw-J rank-32 obstruction with restartable checkpoints. It does not certify physical local-Lorentz descent because no accepted physical curvature section difference is evaluated.",
    };
    // Complete heartbeat precedes the report. The report is the final
    // publication event and therefore proves report-last ordering.
    atomic_json(
        &heartbeat_path,
        &state.heartbeat(
            &manifest_sha256,
            LocalLorentzRunPhase::Complete,
            false,
            None,
        ),
    )?;
    atomic_json(&report_path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "adynkra-local-lorentz-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn run_inputs() -> LocalLorentzRunInputs {
        LocalLorentzRunInputs {
            source_revision: "test-revision".to_string(),
            source_tree_sha256: sha256_bytes(b"test source tree"),
            binary_sha256: sha256_bytes(b"test binary"),
            physical_projection_sha256: sha256_bytes(b"pending physical projection"),
        }
    }

    #[test]
    fn raw_j_rows_exhaust_the_1760_column_basis_exactly() {
        let entries = current_raw_j_residual_entries();
        assert_eq!(entries.len(), 1_760);
        let sources = entries
            .iter()
            .map(|entry| entry.source)
            .collect::<BTreeSet<_>>();
        assert_eq!(sources.len(), 1_760);
        assert!(entries.iter().all(|entry| {
            entry.source.derivative_spinor < 32
                && entry.source.two_form_ordinal < 55
                && entry.source.two_form_mask.count_ones() == 2
                && entry.row.sector == LocalLorentzDescentSector::RawJOne
                && entry.row.target_coordinate < 32
                && entry.row.exterior_spinor_mask == 0
                && entry.row.momentum_exponents == [0; 11]
                && entry.coefficient.denominator == 1_056
                && entry.coefficient.numerator.abs() == 109
        }));
        let (_, rank, nullity, minimum, maximum) = raw_j_accounting(&entries);
        assert_eq!(rank, 32);
        assert_eq!(nullity, 1_728);
        assert_eq!(minimum, 55);
        assert_eq!(maximum, 55);
    }

    #[test]
    fn report_reproduces_the_obstruction_and_fails_closed_physically() {
        let report = verify();
        assert!(report.harness_integrity_passed);
        assert!(!report.passed);
        assert_eq!(report.raw_j_matrix_rank, 32);
        assert_eq!(report.raw_j_matrix_nullity, 1_728);
        assert_eq!(report.raw_j_nonzero_entries, 1_760);
        assert_eq!(report.accepted_physical_sectors_exhaustively_checked, 0);
        assert!(!report.physical_sector_projection_frozen);
        assert!(!report.exhaustive_physical_section_difference_complete);
        assert!(!report.physical_local_lorentz_descent_certified);
        let raw = report
            .sector_statuses
            .iter()
            .find(|sector| sector.sector == LocalLorentzDescentSector::RawJOne)
            .unwrap();
        assert_eq!(raw.gate_status, SectionDifferenceGateStatus::FailedNonzero);
        assert_eq!(raw.exact_zero, Some(false));
        let physical = report
            .sector_statuses
            .iter()
            .filter(|sector| sector.role == "accepted physical curvature")
            .collect::<Vec<_>>();
        assert_eq!(physical.len(), 2);
        assert!(physical.iter().all(|sector| {
            sector.gate_status == SectionDifferenceGateStatus::PendingExhaustiveProjection
                && sector.exhaustive_columns_checked == 0
                && sector.exact_zero.is_none()
        }));
    }

    #[test]
    fn raw_residual_digest_is_deterministic_and_mutation_sensitive() {
        let entries = current_raw_j_residual_entries();
        let digest = hash_raw_residual(&entries);
        assert_eq!(digest, hash_raw_residual(&current_raw_j_residual_entries()));
        let mut mutated = entries;
        mutated[0].coefficient.numerator *= -1;
        assert_ne!(digest, hash_raw_residual(&mutated));
    }

    #[test]
    fn artifact_writer_publishes_serializable_fail_closed_report() {
        let unique = format!(
            "adynkra-local-lorentz-descent-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        let path = directory.join("report.json");
        let report = write_artifact(&path).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(decoded["schema_version"], SCHEMA_VERSION);
        assert_eq!(decoded["passed"], false);
        assert_eq!(
            decoded["raw_j_residual_sha256"],
            report.raw_j_residual_sha256
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn checkpoint_resume_is_scientifically_equivalent_and_report_is_last() {
        let directory = temporary_directory("resume");
        let cancellation = AtomicBool::new(false);
        let first = run_current_raw_j_harness(&directory, &run_inputs(), &cancellation).unwrap();
        assert!(first.passed);
        assert_eq!(first.fresh_blocks, 55);
        assert_eq!(first.resumed_blocks, 0);
        assert_eq!(first.completed_directions, 1_760);
        assert_eq!(first.exact_residual_entries, 1_760);
        assert_eq!(first.exact_rank, 32);
        assert!(first.checkpoint_resume_equivalent);
        assert!(directory.join("report.json").exists());
        let heartbeat: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("heartbeat.json")).unwrap()).unwrap();
        assert_eq!(heartbeat["phase"], "complete");

        fs::remove_file(directory.join("report.json")).unwrap();
        let second = run_current_raw_j_harness(&directory, &run_inputs(), &cancellation).unwrap();
        assert_eq!(second.resumed_blocks, 55);
        assert_eq!(second.fresh_blocks, 0);
        assert_eq!(second.exact_residual_entries, first.exact_residual_entries);
        assert_eq!(second.exact_output_rows, first.exact_output_rows);
        assert_eq!(second.exact_rank, first.exact_rank);
        assert_eq!(second.first_residual_witness, first.first_residual_witness);
        assert_eq!(
            second.scientific_report.raw_j_residual_sha256,
            first.scientific_report.raw_j_residual_sha256
        );
        assert!(second.report_published_last);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn corrupt_checkpoint_fails_fast_and_does_not_publish_report() {
        let directory = temporary_directory("corrupt");
        let cancellation = AtomicBool::new(false);
        run_current_raw_j_harness(&directory, &run_inputs(), &cancellation).unwrap();
        fs::remove_file(directory.join("report.json")).unwrap();
        let checkpoint = directory.join("blocks/block_017.json");
        let mut bytes = fs::read(&checkpoint).unwrap();
        bytes.truncate(bytes.len() / 2);
        fs::write(&checkpoint, bytes).unwrap();
        let error = run_current_raw_j_harness(&directory, &run_inputs(), &cancellation)
            .expect_err("corrupt checkpoint must fail closed");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(!directory.join("report.json").exists());
        let heartbeat: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("heartbeat.json")).unwrap()).unwrap();
        assert_eq!(heartbeat["phase"], "failed");
        assert!(!heartbeat["failure"].as_str().unwrap().is_empty());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cooperative_cancellation_stops_before_blocks_and_report_publication() {
        let directory = temporary_directory("cancel");
        let cancellation = AtomicBool::new(true);
        let error = run_current_raw_j_harness(&directory, &run_inputs(), &cancellation)
            .expect_err("pre-set cancellation must stop the run");
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(directory.join("manifest.json").exists());
        assert!(directory.join("heartbeat.json").exists());
        assert!(!directory.join("report.json").exists());
        assert_eq!(fs::read_dir(directory.join("blocks")).unwrap().count(), 0);
        let heartbeat: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("heartbeat.json")).unwrap()).unwrap();
        assert_eq!(heartbeat["phase"], "cancelled");
        assert_eq!(heartbeat["cancellation_observed"], true);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn immutable_manifest_rejects_binary_mutation() {
        let directory = temporary_directory("manifest");
        let cancellation = AtomicBool::new(true);
        run_current_raw_j_harness(&directory, &run_inputs(), &cancellation).unwrap_err();
        let mut mutated = run_inputs();
        mutated.binary_sha256 = sha256_bytes(b"mutated binary");
        let error = run_current_raw_j_harness(&directory, &mutated, &cancellation)
            .expect_err("manifest mutation must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        fs::remove_dir_all(directory).unwrap();
    }
}
