//! GPU staging for the bounded 56-column higher-bidegree four-form solve.
//!
//! This module owns canonical row ordinals, denominator and capacity gates,
//! three-prime COO reduction, and resumable block identity. It deliberately
//! does not invent the missing Cartesian intertwiners or perform exact
//! reconstruction.

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

pub const SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-gpu-stream-v1";
pub const ROW_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-row-v1";
pub const CHECKPOINT_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-checkpoint-v1";
pub const HEARTBEAT_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-heartbeat-v1";
pub const COLUMN_BINDING_SCHEMA_VERSION: &str = "adynkra-11d-four-form-56-column-binding-v1";
pub const COLUMN_COUNT: u32 = 56;
pub const D21_COLUMN_COUNT: u32 = 52;
pub const D02_COLUMN_COUNT: u32 = 4;
pub const D_G4_COORDINATES: u64 = 32 * 330;
pub const D21_SOURCE_COORDINATES: u64 = 496 * 11 * 320;
pub const D02_SOURCE_COORDINATES: u64 = 66 * 320;
pub const D21_ROW_COUNT: u64 = D21_SOURCE_COORDINATES * D_G4_COORDINATES;
pub const D02_ROW_COUNT: u64 = D02_SOURCE_COORDINATES * D_G4_COORDINATES;
pub const TOTAL_ROW_COUNT: u64 = D21_ROW_COUNT + D02_ROW_COUNT;
pub const PINNED_PRIMES: [u32; 3] = [1_073_741_783, 1_073_741_723, 1_073_741_719];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct D21SectorSpec {
    pub dynkin_label: &'static str,
    pub first_global_column: u32,
    pub multiplicity: u32,
}

pub const D21_SECTORS: [D21SectorSpec; 5] = [
    D21SectorSpec {
        dynkin_label: "00001",
        first_global_column: 0,
        multiplicity: 7,
    },
    D21SectorSpec {
        dynkin_label: "00011",
        first_global_column: 7,
        multiplicity: 7,
    },
    D21SectorSpec {
        dynkin_label: "00101",
        first_global_column: 14,
        multiplicity: 11,
    },
    D21SectorSpec {
        dynkin_label: "01001",
        first_global_column: 25,
        multiplicity: 14,
    },
    D21SectorSpec {
        dynkin_label: "10001",
        first_global_column: 39,
        multiplicity: 13,
    },
];

pub fn d21_sector(label: &str) -> Result<D21SectorSpec, String> {
    D21_SECTORS
        .into_iter()
        .find(|sector| sector.dynkin_label == label)
        .ok_or_else(|| format!("unknown d21 D G4 sector {label}"))
}

fn expected_column_identity(column: u32) -> Option<(&'static str, u32)> {
    for sector in D21_SECTORS {
        if (sector.first_global_column..sector.first_global_column + sector.multiplicity)
            .contains(&column)
        {
            return Some((sector.dynkin_label, column - sector.first_global_column + 1));
        }
    }
    match column {
        52 => Some(("00001", 1)),
        53 => Some(("01001", 1)),
        54 => Some(("10001", 1)),
        55 => Some(("10001", 2)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BidegreeBranch {
    D2P1,
    D0P2,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalRow {
    pub branch: BidegreeBranch,
    pub source_coordinate: u64,
    pub target_coordinate: u32,
}

impl CanonicalRow {
    pub fn ordinal(self) -> Result<u64, String> {
        if u64::from(self.target_coordinate) >= D_G4_COORDINATES {
            return Err("D G4 target coordinate is out of range".to_string());
        }
        let (base, source_bound) = match self.branch {
            BidegreeBranch::D2P1 => (0, D21_SOURCE_COORDINATES),
            BidegreeBranch::D0P2 => (D21_ROW_COUNT, D02_SOURCE_COORDINATES),
        };
        if self.source_coordinate >= source_bound {
            return Err("four-form source coordinate is out of range".to_string());
        }
        Ok(base + self.source_coordinate * D_G4_COORDINATES + u64::from(self.target_coordinate))
    }

    pub fn from_ordinal(ordinal: u64) -> Result<Self, String> {
        if ordinal >= TOTAL_ROW_COUNT {
            return Err("four-form row ordinal is out of range".to_string());
        }
        let (branch, relative) = if ordinal < D21_ROW_COUNT {
            (BidegreeBranch::D2P1, ordinal)
        } else {
            (BidegreeBranch::D0P2, ordinal - D21_ROW_COUNT)
        };
        Ok(Self {
            branch,
            source_coordinate: relative / D_G4_COORDINATES,
            target_coordinate: (relative % D_G4_COORDINATES) as u32,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExactCooEntry {
    pub row: u64,
    pub column: u32,
    pub reserved: u32,
    pub real: i64,
    pub imaginary: i64,
}

impl ExactCooEntry {
    pub fn validate(&self) -> Result<(), String> {
        if self.row >= TOTAL_ROW_COUNT {
            return Err("COO row is out of range".to_string());
        }
        if self.column >= COLUMN_COUNT {
            return Err("COO column is out of range".to_string());
        }
        if self.reserved != 0 {
            return Err("COO reserved field is nonzero".to_string());
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreePrimeGaussian {
    /// Prime-major `(real, imaginary)` lanes in `PINNED_PRIMES` order.
    pub value: [u32; 6],
}

impl ThreePrimeGaussian {
    pub fn is_zero_at_every_prime(self) -> bool {
        self.value == [0; 6]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FourForm56ManifestIdentity {
    pub schema_version: String,
    pub semantic_sha256: String,
    pub source_sha256: String,
    pub source_basis_sha256: String,
    pub target_basis_sha256: String,
    pub coefficient_inventory_sha256: String,
    pub generator_map_sha256: String,
    pub normal_form_sha256: String,
    pub common_denominator: u64,
    pub denominator_audit_sha256: String,
    pub ordered_primes: [u32; 3],
    pub row_schema_version: String,
    pub columns: u32,
    pub rows: u64,
    pub block_rows: u64,
}

/// Immutable identity for one coefficient column before it joins the complete
/// ordered 56-column generator-map digest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FourForm56ColumnBinding {
    pub schema_version: String,
    pub global_column: u32,
    pub branch: BidegreeBranch,
    pub dynkin_label: String,
    pub multiplicity_copy: u32,
    pub generator_schema_version: String,
    pub generator_source_sha256: String,
    pub source_basis_sha256: String,
    pub target_basis_sha256: String,
    pub coefficient_inventory_sha256: String,
    pub projector_sha256: String,
    pub exact_stream_entries: u64,
    pub exact_stream_sha256: String,
}

impl FourForm56ColumnBinding {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != COLUMN_BINDING_SCHEMA_VERSION
            || self.global_column >= COLUMN_COUNT
            || self.dynkin_label.len() != 5
            || !self.dynkin_label.bytes().all(|byte| byte.is_ascii_digit())
            || self.multiplicity_copy == 0
            || self.generator_schema_version.is_empty()
            || self.exact_stream_entries == 0
        {
            return Err("four-form column binding constants are invalid".to_string());
        }
        let expected_branch = if self.global_column < D21_COLUMN_COUNT {
            BidegreeBranch::D2P1
        } else {
            BidegreeBranch::D0P2
        };
        if self.branch != expected_branch {
            return Err("four-form column binding branch is invalid".to_string());
        }
        let Some((expected_label, expected_copy)) = expected_column_identity(self.global_column)
        else {
            return Err("four-form column binding has no canonical identity".to_string());
        };
        if self.dynkin_label != expected_label || self.multiplicity_copy != expected_copy {
            return Err("four-form column binding sector or copy is invalid".to_string());
        }
        for digest in [
            &self.generator_source_sha256,
            &self.source_basis_sha256,
            &self.target_basis_sha256,
            &self.coefficient_inventory_sha256,
            &self.projector_sha256,
            &self.exact_stream_sha256,
        ] {
            validate_sha256(digest)?;
        }
        Ok(())
    }
}

/// Canonical digest of a complete, column-ordered generator inventory.
/// Partial inventories are deliberately rejected so a manifest cannot claim
/// a complete map while some columns are still synthetic or absent.
pub fn ordered_generator_map_sha256(
    bindings: &[FourForm56ColumnBinding],
) -> Result<String, String> {
    if bindings.len() != COLUMN_COUNT as usize {
        return Err("four-form generator inventory is incomplete".to_string());
    }
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-four-form-56-ordered-column-bindings-v1\0");
    for (expected, binding) in bindings.iter().enumerate() {
        binding.validate()?;
        if binding.global_column != expected as u32 {
            return Err("four-form generator bindings are not in canonical order".to_string());
        }
        let bytes = serde_json::to_vec(binding)
            .map_err(|error| format!("cannot serialize four-form column binding: {error}"))?;
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn validate_manifest_generator_map(
    identity: &FourForm56ManifestIdentity,
    bindings: &[FourForm56ColumnBinding],
) -> Result<(), String> {
    identity.validate()?;
    if ordered_generator_map_sha256(bindings)? != identity.generator_map_sha256 {
        return Err("four-form manifest generator-map digest mismatch".to_string());
    }
    Ok(())
}

impl FourForm56ManifestIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION
            || self.row_schema_version != ROW_SCHEMA_VERSION
            || self.columns != COLUMN_COUNT
            || self.rows != TOTAL_ROW_COUNT
            || self.block_rows == 0
            || self.ordered_primes != PINNED_PRIMES
            || self.common_denominator == 0
        {
            return Err("four-form manifest constants are invalid".to_string());
        }
        for prime in PINNED_PRIMES {
            if gcd(self.common_denominator, u64::from(prime)) != 1 {
                return Err("four-form denominator is inadmissible".to_string());
            }
        }
        for digest in [
            &self.semantic_sha256,
            &self.source_sha256,
            &self.source_basis_sha256,
            &self.target_basis_sha256,
            &self.coefficient_inventory_sha256,
            &self.generator_map_sha256,
            &self.normal_form_sha256,
            &self.denominator_audit_sha256,
        ] {
            validate_sha256(digest)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FourForm56Checkpoint {
    pub schema_version: String,
    pub semantic_sha256: String,
    pub generation: u64,
    pub next_block_ordinal: u64,
    pub next_row_ordinal: u64,
    pub input_terms: u64,
    pub reduced_terms: u64,
    pub per_prime_ranks: [u32; 3],
    pub pivot_transcript_sha256: [String; 3],
    pub canonical_row_digest_sha256: String,
    pub first_witness_sha256: Option<String>,
    pub complete: bool,
}

impl FourForm56Checkpoint {
    pub fn validate_for_adoption(
        &self,
        identity: &FourForm56ManifestIdentity,
    ) -> Result<(), String> {
        identity.validate()?;
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION
            || self.semantic_sha256 != identity.semantic_sha256
            || self.next_row_ordinal > TOTAL_ROW_COUNT
            || self.per_prime_ranks.iter().any(|rank| *rank > COLUMN_COUNT)
            || self.complete != (self.next_row_ordinal == TOTAL_ROW_COUNT)
        {
            return Err("four-form checkpoint identity or progress is invalid".to_string());
        }
        for digest in self
            .pivot_transcript_sha256
            .iter()
            .chain(std::iter::once(&self.canonical_row_digest_sha256))
            .chain(self.first_witness_sha256.iter())
        {
            validate_sha256(digest)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FourForm56Heartbeat {
    pub schema_version: String,
    pub semantic_sha256: String,
    pub sequence: u64,
    pub elapsed_milliseconds: u64,
    pub phase: String,
    pub block_ordinal: u64,
    pub blocks_total: u64,
    pub rows_completed: u64,
    pub rows_total: u64,
    pub input_terms: u64,
    pub reduced_terms: u64,
    pub terms_per_second: f64,
    pub eta_seconds: Option<u64>,
    pub cpu_percent: f64,
    pub gpu_percent: f64,
    pub rss_bytes: u64,
    pub rss_high_water_bytes: u64,
    pub vram_bytes: u64,
    pub vram_high_water_bytes: u64,
    pub per_prime_ranks: [u32; 3],
    pub overflow: bool,
    pub cancellation_requested: bool,
}

impl FourForm56Heartbeat {
    pub fn validate(&self, identity: &FourForm56ManifestIdentity) -> Result<(), String> {
        if self.schema_version != HEARTBEAT_SCHEMA_VERSION
            || self.semantic_sha256 != identity.semantic_sha256
            || self.rows_total != TOTAL_ROW_COUNT
            || self.rows_completed > self.rows_total
            || self.block_ordinal > self.blocks_total
            || !self.terms_per_second.is_finite()
            || !self.cpu_percent.is_finite()
            || !self.gpu_percent.is_finite()
            || self.per_prime_ranks.iter().any(|rank| *rank > COLUMN_COUNT)
        {
            return Err("four-form heartbeat is malformed".to_string());
        }
        Ok(())
    }
}

pub fn atomic_checkpoint(path: &Path, checkpoint: &FourForm56Checkpoint) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "checkpoint has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(checkpoint)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    // `hard_link` supplies the no-replace publication primitive missing from
    // portable `rename`: it fails if another owner published this generation.
    std::fs::hard_link(&temporary, path)?;
    std::fs::remove_file(&temporary)?;
    std::fs::File::open(parent)?.sync_all()
}

pub fn exact_coo_sha256(entries: &[ExactCooEntry], common_denominator: u64) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-four-form-56-exact-coo-v1\0");
    hash.update(common_denominator.to_le_bytes());
    for entry in entries {
        hash.update(entry.row.to_le_bytes());
        hash.update(entry.column.to_le_bytes());
        hash.update(entry.real.to_le_bytes());
        hash.update(entry.imaginary.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// Deterministic reference reducer for generator canaries and CUDA parity.
/// Production elimination consumes the same sorted `(row * 56 + column)` keys.
pub fn reduce_reference(
    entries: &[ExactCooEntry],
    common_denominator: u64,
) -> Result<Vec<(u64, ThreePrimeGaussian)>, String> {
    if common_denominator == 0 {
        return Err("four-form denominator is zero".to_string());
    }
    entries.iter().try_for_each(ExactCooEntry::validate)?;
    let mut inverse = [0_u32; 3];
    for (slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
        if gcd(common_denominator, u64::from(prime)) != 1 {
            return Err("four-form denominator is inadmissible".to_string());
        }
        inverse[slot] = pow_mod(
            (common_denominator % u64::from(prime)) as u32,
            prime - 2,
            prime,
        );
    }
    let mut reduced = BTreeMap::<u64, ThreePrimeGaussian>::new();
    for entry in entries {
        let key = entry.row * u64::from(COLUMN_COUNT) + u64::from(entry.column);
        let output = reduced.entry(key).or_default();
        for (slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
            for (component, numerator) in [entry.real, entry.imaginary].into_iter().enumerate() {
                let residue = numerator.rem_euclid(i64::from(prime)) as u32;
                let value =
                    (u64::from(residue) * u64::from(inverse[slot]) % u64::from(prime)) as u32;
                let lane = 2 * slot + component;
                let sum = u64::from(output.value[lane]) + u64::from(value);
                output.value[lane] = if sum >= u64::from(prime) {
                    (sum - u64::from(prime)) as u32
                } else {
                    sum as u32
                };
            }
        }
    }
    Ok(reduced
        .into_iter()
        .filter(|(_, value)| !value.is_zero_at_every_prime())
        .collect())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PrimeGaussian {
    real: u32,
    imaginary: u32,
}

impl PrimeGaussian {
    fn is_zero(self) -> bool {
        self.real == 0 && self.imaginary == 0
    }

    fn subtract(self, right: Self, prime: u32) -> Self {
        Self {
            real: sub_mod(self.real, right.real, prime),
            imaginary: sub_mod(self.imaginary, right.imaginary, prime),
        }
    }

    fn multiply(self, right: Self, prime: u32) -> Self {
        let modulus = u64::from(prime);
        let ac = u64::from(self.real) * u64::from(right.real) % modulus;
        let bd = u64::from(self.imaginary) * u64::from(right.imaginary) % modulus;
        let ad = u64::from(self.real) * u64::from(right.imaginary) % modulus;
        let bc = u64::from(self.imaginary) * u64::from(right.real) % modulus;
        Self {
            real: sub_mod(ac as u32, bd as u32, prime),
            imaginary: ((ad + bc) % modulus) as u32,
        }
    }

    fn inverse(self, prime: u32) -> Result<Self, String> {
        if self.is_zero() {
            return Err("cannot invert zero in F_p(i)".to_string());
        }
        let modulus = u64::from(prime);
        let norm = (u64::from(self.real) * u64::from(self.real)
            + u64::from(self.imaginary) * u64::from(self.imaginary))
            % modulus;
        if norm == 0 {
            return Err("Gaussian residue has zero norm at a pinned prime".to_string());
        }
        let inverse_norm = pow_mod(norm as u32, prime - 2, prime);
        Ok(Self {
            real: (u64::from(self.real) * u64::from(inverse_norm) % modulus) as u32,
            imaginary: (u64::from(if self.imaginary == 0 {
                0
            } else {
                prime - self.imaginary
            }) * u64::from(inverse_norm)
                % modulus) as u32,
        })
    }
}

fn sub_mod(left: u32, right: u32, prime: u32) -> u32 {
    if left >= right {
        left - right
    } else {
        (u64::from(left) + u64::from(prime) - u64::from(right)) as u32
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateRrefSummary {
    pub candidate_columns: u32,
    pub canonical_rows_consumed: u64,
    pub reduced_terms_consumed: u64,
    pub per_prime_ranks: [u32; 3],
    pub per_prime_pivot_columns: [Vec<u32>; 3],
    pub per_prime_pivot_rows: [Vec<u64>; 3],
    pub consensus_pivot_columns: Option<Vec<u32>>,
}

/// Incremental exact row reduction for a GPU-reduced candidate-seed stream.
///
/// Input keys are `canonical_row * 56 + candidate_slot` and must be strictly
/// increasing. A producer must end a block on a row boundary. The state keeps
/// at most `3 * candidate_columns^2` residues, so row construction and GPU
/// sort-reduce can remain fully streamed.
pub struct StreamedCandidateRref {
    candidate_columns: usize,
    bases: [Vec<Option<Vec<PrimeGaussian>>>; 3],
    basis_witness_rows: [Vec<Option<u64>>; 3],
    last_row: Option<u64>,
    canonical_rows_consumed: u64,
    reduced_terms_consumed: u64,
}

impl StreamedCandidateRref {
    pub fn new(candidate_columns: u32) -> Result<Self, String> {
        if candidate_columns == 0 || candidate_columns > COLUMN_COUNT {
            return Err("candidate seed count must be between 1 and 56".to_string());
        }
        let width = candidate_columns as usize;
        Ok(Self {
            candidate_columns: width,
            bases: std::array::from_fn(|_| vec![None; width]),
            basis_witness_rows: std::array::from_fn(|_| vec![None; width]),
            last_row: None,
            canonical_rows_consumed: 0,
            reduced_terms_consumed: 0,
        })
    }

    pub fn push_reduced_block(
        &mut self,
        entries: &[(u64, ThreePrimeGaussian)],
    ) -> Result<(), String> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut previous_key = None;
        for &(key, _) in entries {
            if previous_key.is_some_and(|previous| key <= previous) {
                return Err("candidate RREF block keys are not strictly increasing".to_string());
            }
            previous_key = Some(key);
        }
        let first_row = entries[0].0 / u64::from(COLUMN_COUNT);
        if self.last_row.is_some_and(|last| first_row <= last) {
            return Err("candidate RREF blocks overlap or split a canonical row".to_string());
        }

        let mut offset = 0;
        while offset < entries.len() {
            let row = entries[offset].0 / u64::from(COLUMN_COUNT);
            let mut vectors: [Vec<PrimeGaussian>; 3] =
                std::array::from_fn(|_| vec![PrimeGaussian::default(); self.candidate_columns]);
            while offset < entries.len() && entries[offset].0 / u64::from(COLUMN_COUNT) == row {
                let (key, value) = entries[offset];
                let candidate = (key % u64::from(COLUMN_COUNT)) as usize;
                if candidate >= self.candidate_columns {
                    return Err("candidate RREF key uses an inactive seed slot".to_string());
                }
                for prime_slot in 0..3 {
                    vectors[prime_slot][candidate] = PrimeGaussian {
                        real: value.value[2 * prime_slot],
                        imaginary: value.value[2 * prime_slot + 1],
                    };
                }
                offset += 1;
            }
            for prime_slot in 0..3 {
                if let Some(pivot) = insert_rref_row(
                    &mut self.bases[prime_slot],
                    vectors[prime_slot].clone(),
                    PINNED_PRIMES[prime_slot],
                )? {
                    self.basis_witness_rows[prime_slot][pivot] = Some(row);
                }
            }
            self.canonical_rows_consumed += 1;
            self.last_row = Some(row);
        }
        self.reduced_terms_consumed += entries.len() as u64;
        Ok(())
    }

    pub fn summary(&self) -> CandidateRrefSummary {
        let pivots: [Vec<u32>; 3] = std::array::from_fn(|prime_slot| {
            self.bases[prime_slot]
                .iter()
                .enumerate()
                .filter_map(|(column, row)| row.as_ref().map(|_| column as u32))
                .collect()
        });
        let consensus =
            (pivots[0] == pivots[1] && pivots[1] == pivots[2]).then(|| pivots[0].clone());
        let pivot_rows: [Vec<u64>; 3] = std::array::from_fn(|prime_slot| {
            pivots[prime_slot]
                .iter()
                .map(|&column| {
                    self.basis_witness_rows[prime_slot][column as usize]
                        .expect("every stored pivot has a witness row")
                })
                .collect()
        });
        CandidateRrefSummary {
            candidate_columns: self.candidate_columns as u32,
            canonical_rows_consumed: self.canonical_rows_consumed,
            reduced_terms_consumed: self.reduced_terms_consumed,
            per_prime_ranks: std::array::from_fn(|slot| pivots[slot].len() as u32),
            per_prime_pivot_columns: pivots,
            per_prime_pivot_rows: pivot_rows,
            consensus_pivot_columns: consensus,
        }
    }
}

fn insert_rref_row(
    basis: &mut [Option<Vec<PrimeGaussian>>],
    mut row: Vec<PrimeGaussian>,
    prime: u32,
) -> Result<Option<usize>, String> {
    for existing in basis.iter().flatten() {
        let pivot = existing
            .iter()
            .position(|value| !value.is_zero())
            .ok_or_else(|| "candidate RREF stored a zero basis row".to_string())?;
        let factor = row[pivot];
        if factor.is_zero() {
            continue;
        }
        for column in 0..row.len() {
            row[column] = row[column].subtract(existing[column].multiply(factor, prime), prime);
        }
    }
    let Some(pivot) = row.iter().position(|value| !value.is_zero()) else {
        return Ok(None);
    };
    let inverse = row[pivot].inverse(prime)?;
    for value in &mut row {
        *value = value.multiply(inverse, prime);
    }
    for existing in basis.iter_mut().flatten() {
        let factor = existing[pivot];
        if factor.is_zero() {
            continue;
        }
        for column in 0..row.len() {
            existing[column] =
                existing[column].subtract(row[column].multiply(factor, prime), prime);
        }
    }
    basis[pivot] = Some(row);
    Ok(Some(pivot))
}

pub const D21_RREF_CHECKPOINT_SCHEMA_VERSION: &str = "adynkra-11d-d21-sector-rref-checkpoint-v1";
pub const D21_INGESTION_SCHEMA_VERSION: &str = "adynkra-11d-d21-gpu-ingestion-v1";
pub const D21_EXACT_MINOR_SCHEMA_VERSION: &str = "adynkra-11d-d21-exact-pivot-minor-v1";
pub const D21_DEVICE_DIAGRAM_BLOB_SHA256: &str =
    "ecf6545e4b6cc997a9f6ad5744e810892b76837b6eb6dd1c4e8f8f97757c901c";

/// Canonical compact device descriptor. The no-cross-factorial normalization
/// is explicit as numerator/denominator `1/1` in every 16-byte record.
pub fn canonical_d21_device_diagram_blob() -> Result<Vec<u8>, String> {
    use crate::eleven_dimensional_d21_invariant_diagrams::packed_diagrams;

    let diagrams = packed_diagrams();
    if diagrams.len() != 400 {
        return Err("D21 compact descriptor inventory is not 400".to_string());
    }
    let mut output = Vec::with_capacity(diagrams.len() * 16);
    for diagram in diagrams {
        let mut outer_mask = 0_u8;
        for &label in &diagram.outer_external[..diagram.outer_count as usize] {
            outer_mask |= 1_u8 << label;
        }
        let mut inner_mask = 0_u8;
        for &label in &diagram.inner_external[..diagram.inner_count as usize] {
            inner_mask |= 1_u8 << label;
        }
        if outer_mask & inner_mask != 0 || diagram.metric_count > 3 {
            return Err("D21 compact descriptor has invalid external partition".to_string());
        }
        let mut pairs = 0_u32;
        for pair in 0..diagram.metric_count as usize {
            let left = diagram.metric_pairs[2 * pair];
            let right = diagram.metric_pairs[2 * pair + 1];
            if left >= 6 || right >= 6 || left >= right {
                return Err("D21 compact descriptor has a noncanonical metric pair".to_string());
            }
            let encoded = u32::from(left) | (u32::from(right) << 3);
            pairs |= encoded << (6 * pair);
        }
        output.extend_from_slice(&[
            diagram.outer_degree,
            diagram.inner_degree,
            diagram.cross,
            outer_mask,
            inner_mask,
            diagram.metric_count,
        ]);
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&pairs.to_le_bytes());
        output.extend_from_slice(&1_i16.to_le_bytes());
        output.extend_from_slice(&1_u16.to_le_bytes());
    }
    if output.len() != 6_400
        || format!("{:x}", Sha256::digest(&output)) != D21_DEVICE_DIAGRAM_BLOB_SHA256
    {
        return Err("D21 compact descriptor digest differs from the frozen oracle".to_string());
    }
    Ok(output)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct D21GpuIngestionContract {
    pub schema_version: String,
    pub sector: String,
    pub candidate_seed_count: u32,
    pub candidate_inventory_sha256: String,
    pub source_basis_sha256: String,
    pub target_basis_sha256: String,
    pub normal_form_sha256: String,
    pub common_denominator: u64,
    pub first_row_inclusive: u64,
    pub last_row_exclusive: u64,
    pub complete_row_interval: bool,
    pub input_entries: u64,
    pub exact_coo_sha256: String,
}

impl D21GpuIngestionContract {
    pub fn validate_entries(&self, entries: &[ExactCooEntry]) -> Result<(), String> {
        let sector = d21_sector(&self.sector)?;
        if self.schema_version != D21_INGESTION_SCHEMA_VERSION
            || self.candidate_seed_count < sector.multiplicity
            || self.candidate_seed_count > COLUMN_COUNT
            || self.common_denominator == 0
            || self.first_row_inclusive >= self.last_row_exclusive
            || self.last_row_exclusive > D21_ROW_COUNT
            || !self.complete_row_interval
            || self.input_entries != entries.len() as u64
        {
            return Err("d21 GPU ingestion constants are invalid".to_string());
        }
        for digest in [
            &self.candidate_inventory_sha256,
            &self.source_basis_sha256,
            &self.target_basis_sha256,
            &self.normal_form_sha256,
            &self.exact_coo_sha256,
        ] {
            validate_sha256(digest)?;
        }
        for prime in PINNED_PRIMES {
            if gcd(self.common_denominator, u64::from(prime)) != 1 {
                return Err("d21 GPU ingestion denominator is inadmissible".to_string());
            }
        }
        let mut previous = None;
        for entry in entries {
            entry.validate()?;
            if entry.row < self.first_row_inclusive
                || entry.row >= self.last_row_exclusive
                || entry.row >= D21_ROW_COUNT
                || entry.column >= self.candidate_seed_count
            {
                return Err("d21 GPU ingestion entry is outside its contract".to_string());
            }
            let key = entry.row * u64::from(COLUMN_COUNT) + u64::from(entry.column);
            if previous.is_some_and(|prior| key < prior) {
                return Err("d21 GPU ingestion entries are not canonically ordered".to_string());
            }
            previous = Some(key);
        }
        if exact_coo_sha256(entries, self.common_denominator) != self.exact_coo_sha256 {
            return Err("d21 GPU ingestion COO digest mismatch".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExactPivotMinorReplay {
    pub schema_version: String,
    pub sector: String,
    pub selected_seed_ordinals: Vec<u32>,
    pub witness_row_ordinals: Vec<u64>,
    pub common_denominator: u64,
    pub matrix_sha256: String,
    pub determinant_real: String,
    pub determinant_imaginary: String,
    pub determinant_nonzero: bool,
}

type ExactBig = Ratio<BigInt>;
type ExactBigGaussian = Complex<ExactBig>;

fn exact_zero() -> ExactBigGaussian {
    Complex::new(
        Ratio::from_integer(BigInt::from(0)),
        Ratio::from_integer(BigInt::from(0)),
    )
}

fn exact_gaussian_divide(
    left: &ExactBigGaussian,
    right: &ExactBigGaussian,
) -> Result<ExactBigGaussian, String> {
    let norm = right.re.clone() * right.re.clone() + right.im.clone() * right.im.clone();
    if norm == Ratio::from_integer(BigInt::from(0)) {
        return Err("exact Gaussian pivot has zero norm".to_string());
    }
    Ok(Complex::new(
        (left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone()) / norm.clone(),
        (left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone()) / norm,
    ))
}

fn exact_gaussian_string(value: &ExactBig) -> String {
    if value.denom() == &BigInt::from(1) {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

pub fn replay_exact_d21_pivot_minor(
    sector_label: &str,
    selected_seed_ordinals: &[u32],
    witness_row_ordinals: &[u64],
    common_denominator: u64,
    mut coefficient: impl FnMut(u64, u32) -> Result<(i64, i64), String>,
) -> Result<ExactPivotMinorReplay, String> {
    let sector = d21_sector(sector_label)?;
    let dimension = sector.multiplicity as usize;
    if selected_seed_ordinals.len() != dimension
        || witness_row_ordinals.len() != dimension
        || common_denominator == 0
        || selected_seed_ordinals
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != dimension
        || witness_row_ordinals
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != dimension
        || witness_row_ordinals.iter().any(|&row| row >= D21_ROW_COUNT)
    {
        return Err("d21 exact pivot minor dimensions or ordinals are invalid".to_string());
    }
    for prime in PINNED_PRIMES {
        if gcd(common_denominator, u64::from(prime)) != 1 {
            return Err("d21 exact pivot minor denominator is inadmissible".to_string());
        }
    }
    let denominator = BigInt::from(common_denominator);
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-d21-exact-pivot-minor-v1\0");
    hash.update(sector_label.as_bytes());
    hash.update(common_denominator.to_le_bytes());
    let mut matrix = vec![vec![exact_zero(); dimension]; dimension];
    for (row_index, &row) in witness_row_ordinals.iter().enumerate() {
        hash.update(row.to_le_bytes());
        for (column_index, &seed) in selected_seed_ordinals.iter().enumerate() {
            let (real, imaginary) = coefficient(row, seed)?;
            hash.update(seed.to_le_bytes());
            hash.update(real.to_le_bytes());
            hash.update(imaginary.to_le_bytes());
            matrix[row_index][column_index] = Complex::new(
                Ratio::new(BigInt::from(real), denominator.clone()),
                Ratio::new(BigInt::from(imaginary), denominator.clone()),
            );
        }
    }
    let mut determinant = Complex::new(
        Ratio::from_integer(BigInt::from(1)),
        Ratio::from_integer(BigInt::from(0)),
    );
    for pivot_column in 0..dimension {
        let pivot_row =
            (pivot_column..dimension).find(|&row| matrix[row][pivot_column] != exact_zero());
        let Some(pivot_row) = pivot_row else {
            determinant = exact_zero();
            break;
        };
        if pivot_row != pivot_column {
            matrix.swap(pivot_row, pivot_column);
            determinant = -determinant;
        }
        let pivot = matrix[pivot_column][pivot_column].clone();
        determinant = determinant * pivot.clone();
        for row in (pivot_column + 1)..dimension {
            if matrix[row][pivot_column] == exact_zero() {
                continue;
            }
            let factor = exact_gaussian_divide(&matrix[row][pivot_column], &pivot)?;
            for column in pivot_column..dimension {
                matrix[row][column] = matrix[row][column].clone()
                    - factor.clone() * matrix[pivot_column][column].clone();
            }
        }
    }
    let determinant_nonzero = determinant != exact_zero();
    Ok(ExactPivotMinorReplay {
        schema_version: D21_EXACT_MINOR_SCHEMA_VERSION.to_string(),
        sector: sector_label.to_string(),
        selected_seed_ordinals: selected_seed_ordinals.to_vec(),
        witness_row_ordinals: witness_row_ordinals.to_vec(),
        common_denominator,
        matrix_sha256: format!("{:x}", hash.finalize()),
        determinant_real: exact_gaussian_string(&determinant.re),
        determinant_imaginary: exact_gaussian_string(&determinant.im),
        determinant_nonzero,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct D21SectorRankCheckpoint {
    pub schema_version: String,
    pub sector: String,
    pub candidate_seed_count: u32,
    pub candidate_inventory_sha256: String,
    pub source_basis_sha256: String,
    pub target_basis_sha256: String,
    pub normal_form_sha256: String,
    pub generation: u64,
    pub blocks_completed: u64,
    pub seed_inventory_complete_and_equivariant: bool,
    pub completion_reason: Option<String>,
    pub rref: CandidateRrefSummary,
    pub exact_pivot_minor: Option<ExactPivotMinorReplay>,
    pub complete: bool,
}

impl D21SectorRankCheckpoint {
    pub fn validate(&self) -> Result<(), String> {
        let sector = d21_sector(&self.sector)?;
        if self.schema_version != D21_RREF_CHECKPOINT_SCHEMA_VERSION
            || self.candidate_seed_count < sector.multiplicity
            || self.candidate_seed_count > COLUMN_COUNT
            || self.rref.candidate_columns != self.candidate_seed_count
        {
            return Err("d21 sector RREF checkpoint constants are invalid".to_string());
        }
        for digest in [
            &self.candidate_inventory_sha256,
            &self.source_basis_sha256,
            &self.target_basis_sha256,
            &self.normal_form_sha256,
        ] {
            validate_sha256(digest)?;
        }
        if self
            .rref
            .per_prime_ranks
            .iter()
            .any(|&rank| rank > sector.multiplicity)
        {
            return Err("d21 sector rank exceeds the exact Hom multiplicity".to_string());
        }
        if !self.complete {
            if self.exact_pivot_minor.is_some() || self.completion_reason.is_some() {
                return Err("incomplete d21 checkpoint claims terminal evidence".to_string());
            }
            return Ok(());
        }
        if !self.seed_inventory_complete_and_equivariant
            || self.rref.per_prime_ranks != [sector.multiplicity; 3]
            || self.rref.consensus_pivot_columns.as_ref().map(Vec::len)
                != Some(sector.multiplicity as usize)
            || !matches!(
                self.completion_reason.as_deref(),
                Some("expected_rank_witness" | "full_stream")
            )
        {
            return Err("complete d21 checkpoint lacks three-prime rank evidence".to_string());
        }
        let replay = self
            .exact_pivot_minor
            .as_ref()
            .ok_or_else(|| "complete d21 checkpoint lacks exact pivot replay".to_string())?;
        if replay.schema_version != D21_EXACT_MINOR_SCHEMA_VERSION
            || replay.sector != self.sector
            || !replay.determinant_nonzero
            || Some(&replay.selected_seed_ordinals) != self.rref.consensus_pivot_columns.as_ref()
            || replay.witness_row_ordinals != self.rref.per_prime_pivot_rows[0]
        {
            return Err("d21 exact pivot replay does not bind the modular witness".to_string());
        }
        Ok(())
    }
}

/// Synthetic rows exercise collisions, exact cancellations, all 56 columns,
/// both bidegree ranges, complex coefficients, and denominator inversion.
pub fn synthetic_generator_fixture() -> (Vec<ExactCooEntry>, u64) {
    let mut entries = Vec::with_capacity(2 * COLUMN_COUNT as usize + 4);
    for column in 0..COLUMN_COUNT {
        let row = if column < D21_COLUMN_COUNT {
            u64::from(column) * D_G4_COORDINATES + u64::from(column)
        } else {
            D21_ROW_COUNT
                + u64::from(column - D21_COLUMN_COUNT) * D_G4_COORDINATES
                + u64::from(column)
        };
        entries.push(ExactCooEntry {
            row,
            column,
            reserved: 0,
            real: 3 * i64::from(column + 1),
            imaginary: -6 * i64::from(column + 1),
        });
        entries.push(ExactCooEntry {
            row,
            column,
            reserved: 0,
            real: 6 * i64::from(column + 1),
            imaginary: 3 * i64::from(column + 1),
        });
    }
    entries.extend([
        ExactCooEntry {
            row: 17,
            column: 0,
            reserved: 0,
            real: 5,
            imaginary: -7,
        },
        ExactCooEntry {
            row: 17,
            column: 0,
            reserved: 0,
            real: -5,
            imaginary: 7,
        },
    ]);
    (entries, 3)
}

fn pow_mod(mut base: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * u64::from(base) % u64::from(prime);
        }
        base = (u64::from(base) * u64::from(base) % u64::from(prime)) as u32;
        exponent >>= 1;
    }
    result as u32
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err("malformed SHA-256".to_string())
    }
}

fn compress_signed_permutation_table(flattened: &[i16]) -> Result<(Vec<u8>, Vec<i8>), String> {
    const MASKS: usize = 2048;
    const SPINORS: usize = 32;
    if flattened.len() != MASKS * SPINORS * SPINORS {
        return Err("D21 gamma table has the wrong dimensions".to_string());
    }
    let mut rows = vec![0_u8; MASKS * SPINORS];
    let mut values = vec![0_i8; MASKS * SPINORS];
    for mask in 0..MASKS {
        for column in 0..SPINORS {
            let mut observed = None;
            for row in 0..SPINORS {
                let value = flattened[(mask * SPINORS + row) * SPINORS + column];
                if value == 0 {
                    continue;
                }
                if !matches!(value, -1 | 1) || observed.is_some() {
                    return Err("D21 gamma table is not a signed permutation".to_string());
                }
                observed = Some((row as u8, value as i8));
            }
            let Some((row, value)) = observed else {
                return Err("D21 gamma table has a zero signed-permutation column".to_string());
            };
            rows[mask * SPINORS + column] = row;
            values[mask * SPINORS + column] = value;
        }
    }
    Ok((rows, values))
}

fn d21_signed_permutation_tables() -> Result<([Vec<u8>; 2], [Vec<i8>; 2]), String> {
    let (gamma, charge_gamma) =
        crate::eleven_dimensional_d21_invariant_diagrams::flattened_gamma_mask_tables();
    let (gamma_rows, gamma_values) = compress_signed_permutation_table(&gamma)?;
    let (charge_rows, charge_values) = compress_signed_permutation_table(&charge_gamma)?;
    Ok(([gamma_rows, charge_rows], [gamma_values, charge_values]))
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::{ExactCooEntry, ThreePrimeGaussian, d21_signed_permutation_tables};
    use crate::eleven_dimensional_d21_invariant_diagrams::{
        D21CoefficientQuery, PackedD21Diagram, packed_diagrams,
    };
    use crate::eleven_dimensional_dg4_casimir_projectors::dg4_casimir_row_major;
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr::NonNull;

    unsafe extern "C" {
        fn adynkra_four_form_56_create(
            capacity: u64,
            device_hard_cap: u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> *mut c_void;
        fn adynkra_four_form_56_reduce(
            context: *mut c_void,
            entries: *const ExactCooEntry,
            count: u64,
            common_denominator: u64,
            keys: *mut u64,
            values: *mut ThreePrimeGaussian,
            output_capacity: u64,
            output_count: *mut u64,
            input_terms: *mut u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_four_form_56_resident_bytes(context: *const c_void) -> u64;
        fn adynkra_four_form_56_high_water_bytes(context: *const c_void) -> u64;
        fn adynkra_four_form_56_destroy(context: *mut c_void);
        fn adynkra_d21_create(
            diagrams: *const PackedD21Diagram,
            diagram_count: u64,
            gamma_rows: *const u8,
            gamma_values: *const i8,
            charge_gamma_rows: *const u8,
            charge_gamma_values: *const i8,
            capacity: u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> *mut c_void;
        fn adynkra_d21_evaluate(
            context: *mut c_void,
            queries: *const D21CoefficientQuery,
            count: u64,
            output: *mut i64,
            kernel_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_d21_destroy(context: *mut c_void);
        fn adynkra_d21_witness_create(
            diagrams: *const PackedD21Diagram,
            diagram_count: u64,
            candidate_diagrams: *const u16,
            candidate_count: u32,
            gamma_rows: *const u8,
            gamma_values: *const i8,
            charge_gamma_rows: *const u8,
            charge_gamma_values: *const i8,
            form_axes: *const u8,
            casimir_rows: *const D21CasimirEntry,
            error: *mut c_char,
            error_capacity: u64,
        ) -> *mut c_void;
        fn adynkra_d21_witness_apply(
            context: *mut c_void,
            outer_left: u8,
            outer_right: u8,
            momentum: u8,
            terms: *const D21HTerm,
            term_count: u32,
            source_row_base: u64,
            expected_ranks: *const u32,
            stage_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_d21_witness_evaluate(
            context: *mut c_void,
            outer_left: u8,
            outer_right: u8,
            momentum: u8,
            terms: *const D21HTerm,
            term_count: u32,
            candidate_sectors: *const u8,
            evaluated: *mut u32,
            evaluated_count: u64,
            stage_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_d21_witness_summary(
            context: *mut c_void,
            ranks: *mut u32,
            pivots: *mut u16,
            pivot_rows: *mut u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_d21_witness_destroy(context: *mut c_void);
    }

    pub(crate) struct FourForm56CudaReducer {
        context: NonNull<c_void>,
        capacity: usize,
    }

    unsafe impl Send for FourForm56CudaReducer {}

    impl FourForm56CudaReducer {
        pub(crate) fn new(capacity: usize, device_hard_cap: u64) -> Result<Self, String> {
            let mut error = vec![0_i8; 1024];
            let context = unsafe {
                adynkra_four_form_56_create(
                    capacity as u64,
                    device_hard_cap,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            let context = NonNull::new(context).ok_or_else(|| message(&error))?;
            Ok(Self { context, capacity })
        }

        pub(crate) fn reduce(
            &mut self,
            entries: &[ExactCooEntry],
            common_denominator: u64,
        ) -> Result<(Vec<(u64, ThreePrimeGaussian)>, u64), String> {
            if entries.is_empty() || entries.len() > self.capacity {
                return Err("four-form CUDA batch is empty or exceeds capacity".to_string());
            }
            entries.iter().try_for_each(ExactCooEntry::validate)?;
            let mut keys = vec![0_u64; entries.len()];
            let mut values = vec![ThreePrimeGaussian::default(); entries.len()];
            let mut output_count = 0_u64;
            let mut input_terms = 0_u64;
            let mut error = vec![0_i8; 1024];
            let status = unsafe {
                adynkra_four_form_56_reduce(
                    self.context.as_ptr(),
                    entries.as_ptr(),
                    entries.len() as u64,
                    common_denominator,
                    keys.as_mut_ptr(),
                    values.as_mut_ptr(),
                    entries.len() as u64,
                    &mut output_count,
                    &mut input_terms,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if status != 0 {
                return Err(format!(
                    "four-form CUDA reduce status {status}: {}",
                    message(&error)
                ));
            }
            keys.truncate(output_count as usize);
            values.truncate(output_count as usize);
            Ok((
                keys.into_iter()
                    .zip(values)
                    .filter(|(_, value)| !value.is_zero_at_every_prime())
                    .collect(),
                input_terms,
            ))
        }

        pub(crate) fn resident_bytes(&self) -> u64 {
            unsafe { adynkra_four_form_56_resident_bytes(self.context.as_ptr()) }
        }

        pub(crate) fn high_water_bytes(&self) -> u64 {
            unsafe { adynkra_four_form_56_high_water_bytes(self.context.as_ptr()) }
        }
    }

    impl Drop for FourForm56CudaReducer {
        fn drop(&mut self) {
            unsafe { adynkra_four_form_56_destroy(self.context.as_ptr()) };
        }
    }

    pub(crate) struct D21CudaEvaluator {
        context: NonNull<c_void>,
        capacity: usize,
    }

    unsafe impl Send for D21CudaEvaluator {}

    impl D21CudaEvaluator {
        pub(crate) fn new(capacity: usize) -> Result<Self, String> {
            if capacity == 0 {
                return Err("D21 CUDA evaluator capacity is zero".to_string());
            }
            let diagrams = packed_diagrams();
            let (rows, values) = d21_signed_permutation_tables()?;
            let mut error = vec![0_i8; 1024];
            let context = unsafe {
                adynkra_d21_create(
                    diagrams.as_ptr(),
                    diagrams.len() as u64,
                    rows[0].as_ptr(),
                    values[0].as_ptr(),
                    rows[1].as_ptr(),
                    values[1].as_ptr(),
                    capacity as u64,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            let context = NonNull::new(context).ok_or_else(|| message(&error))?;
            Ok(Self { context, capacity })
        }

        pub(crate) fn evaluate(
            &mut self,
            queries: &[D21CoefficientQuery],
        ) -> Result<(Vec<i64>, f32), String> {
            if queries.is_empty() || queries.len() > self.capacity {
                return Err("D21 CUDA query batch is empty or exceeds capacity".to_string());
            }
            let mut output = vec![0_i64; queries.len()];
            let mut kernel_milliseconds = 0_f32;
            let mut error = vec![0_i8; 1024];
            let status = unsafe {
                adynkra_d21_evaluate(
                    self.context.as_ptr(),
                    queries.as_ptr(),
                    queries.len() as u64,
                    output.as_mut_ptr(),
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if status != 0 {
                return Err(format!(
                    "D21 CUDA evaluator status {status}: {}",
                    message(&error)
                ));
            }
            Ok((output, kernel_milliseconds))
        }
    }

    impl Drop for D21CudaEvaluator {
        fn drop(&mut self) {
            unsafe { adynkra_d21_destroy(self.context.as_ptr()) };
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct D21CasimirEntry {
        column: u16,
        value: i16,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct D21HTerm {
        pub(crate) input_spinor: u8,
        pub(crate) h_vector: u8,
        pub(crate) coefficient: i16,
    }

    #[derive(Clone, Debug)]
    pub(crate) struct D21WitnessSummary {
        pub(crate) ranks: [[u32; 3]; 5],
        pub(crate) pivots: [[[u16; 14]; 3]; 5],
        pub(crate) pivot_rows: [[[u64; 14]; 3]; 5],
    }

    pub(crate) struct D21WitnessCuda {
        context: NonNull<c_void>,
        candidate_diagrams: Vec<u16>,
    }

    unsafe impl Send for D21WitnessCuda {}

    impl D21WitnessCuda {
        pub(crate) fn new(outer_degree: u8) -> Result<Self, String> {
            let diagrams = packed_diagrams();
            let candidate_diagrams = diagrams
                .iter()
                .enumerate()
                .filter_map(|(ordinal, diagram)| {
                    (diagram.outer_degree == outer_degree).then_some(ordinal as u16)
                })
                .collect::<Vec<_>>();
            let expected_count = match outer_degree {
                0 => 21,
                3 => 209,
                4 => 170,
                _ => return Err("D21 witness outer degree must be 0, 3, or 4".to_string()),
            };
            if candidate_diagrams.len() != expected_count {
                return Err("D21 witness candidate inventory drifted".to_string());
            }
            Self::from_candidates(candidate_diagrams)
        }

        pub(crate) fn from_candidates(candidate_diagrams: Vec<u16>) -> Result<Self, String> {
            if candidate_diagrams.is_empty() || candidate_diagrams.len() > 209 {
                return Err("D21 selected witness inventory is empty or too large".to_string());
            }
            let diagrams = packed_diagrams();
            if candidate_diagrams
                .iter()
                .any(|&ordinal| usize::from(ordinal) >= diagrams.len())
            {
                return Err("D21 selected witness diagram is out of range".to_string());
            }
            let (rows, values) = d21_signed_permutation_tables()?;
            let forms = (0_u16..2048)
                .filter(|mask| mask.count_ones() == 4)
                .flat_map(|mask| {
                    (0..11)
                        .filter(move |axis| mask & (1_u16 << axis) != 0)
                        .map(|axis| axis as u8)
                })
                .collect::<Vec<_>>();
            if forms.len() != 330 * 4 {
                return Err("D21 numeric four-form basis drifted".to_string());
            }
            let casimir = dg4_casimir_row_major()?
                .into_iter()
                .flatten()
                .map(|(column, value)| D21CasimirEntry { column, value })
                .collect::<Vec<_>>();
            if casimir.len() != 10_560 * 29 {
                return Err("D21 Casimir row stencil drifted".to_string());
            }
            let mut error = vec![0_i8; 1024];
            let context = unsafe {
                adynkra_d21_witness_create(
                    diagrams.as_ptr(),
                    diagrams.len() as u64,
                    candidate_diagrams.as_ptr(),
                    candidate_diagrams.len() as u32,
                    rows[0].as_ptr(),
                    values[0].as_ptr(),
                    rows[1].as_ptr(),
                    values[1].as_ptr(),
                    forms.as_ptr(),
                    casimir.as_ptr(),
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            let context = NonNull::new(context).ok_or_else(|| message(&error))?;
            Ok(Self {
                context,
                candidate_diagrams,
            })
        }

        pub(crate) fn candidate_diagrams(&self) -> &[u16] {
            &self.candidate_diagrams
        }

        pub(crate) fn apply(
            &mut self,
            outer_pair: [u8; 2],
            momentum: u8,
            h_hat_ordinal: u32,
            terms: &[D21HTerm],
            expected_ranks: [u32; 5],
        ) -> Result<f32, String> {
            if h_hat_ordinal >= 320 || momentum >= 11 || terms.is_empty() || terms.len() > 2 {
                return Err("D21 witness source coordinate is invalid".to_string());
            }
            let pair_ordinal = (0..32)
                .flat_map(|left| ((left + 1)..32).map(move |right| [left, right]))
                .position(|pair| pair == [usize::from(outer_pair[0]), usize::from(outer_pair[1])])
                .ok_or_else(|| "D21 witness outer pair is noncanonical".to_string())?;
            let source_coordinate =
                (pair_ordinal as u64 * 11 + u64::from(momentum)) * 320 + u64::from(h_hat_ordinal);
            let source_row_base = source_coordinate * 10_560;
            let mut milliseconds = 0_f32;
            let mut error = vec![0_i8; 1024];
            let status = unsafe {
                adynkra_d21_witness_apply(
                    self.context.as_ptr(),
                    outer_pair[0],
                    outer_pair[1],
                    momentum,
                    terms.as_ptr(),
                    terms.len() as u32,
                    source_row_base,
                    expected_ranks.as_ptr(),
                    &mut milliseconds,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if status != 0 {
                return Err(format!(
                    "D21 witness CUDA status {status}: {}",
                    message(&error)
                ));
            }
            Ok(milliseconds)
        }

        pub(crate) fn evaluate(
            &mut self,
            outer_pair: [u8; 2],
            momentum: u8,
            h_hat_ordinal: u32,
            terms: &[D21HTerm],
            candidate_sectors: &[u8],
        ) -> Result<(Vec<u32>, f32), String> {
            if h_hat_ordinal >= 320
                || momentum >= 11
                || terms.is_empty()
                || terms.len() > 2
                || candidate_sectors.len() != self.candidate_diagrams.len()
                || candidate_sectors.iter().any(|&sector| sector >= 5)
            {
                return Err(
                    "D21 evaluated witness source or sector inventory is invalid".to_string(),
                );
            }
            let count = 3 * 10_560 * self.candidate_diagrams.len();
            let mut evaluated = vec![0_u32; count];
            let mut milliseconds = 0_f32;
            let mut error = vec![0_i8; 1024];
            let status = unsafe {
                adynkra_d21_witness_evaluate(
                    self.context.as_ptr(),
                    outer_pair[0],
                    outer_pair[1],
                    momentum,
                    terms.as_ptr(),
                    terms.len() as u32,
                    candidate_sectors.as_ptr(),
                    evaluated.as_mut_ptr(),
                    evaluated.len() as u64,
                    &mut milliseconds,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if status != 0 {
                return Err(format!(
                    "D21 evaluated witness CUDA status {status}: {}",
                    message(&error)
                ));
            }
            Ok((evaluated, milliseconds))
        }

        pub(crate) fn summary(&mut self) -> Result<D21WitnessSummary, String> {
            let mut ranks = [0_u32; 15];
            let mut pivots = [0_u16; 15 * 14];
            let mut pivot_rows = [0_u64; 15 * 14];
            let mut error = vec![0_i8; 1024];
            let status = unsafe {
                adynkra_d21_witness_summary(
                    self.context.as_ptr(),
                    ranks.as_mut_ptr(),
                    pivots.as_mut_ptr(),
                    pivot_rows.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if status != 0 {
                return Err(format!(
                    "D21 witness summary status {status}: {}",
                    message(&error)
                ));
            }
            Ok(D21WitnessSummary {
                ranks: std::array::from_fn(|sector| {
                    std::array::from_fn(|prime| ranks[sector * 3 + prime])
                }),
                pivots: std::array::from_fn(|sector| {
                    std::array::from_fn(|prime| {
                        std::array::from_fn(|rank| pivots[(sector * 3 + prime) * 14 + rank])
                    })
                }),
                pivot_rows: std::array::from_fn(|sector| {
                    std::array::from_fn(|prime| {
                        std::array::from_fn(|rank| pivot_rows[(sector * 3 + prime) * 14 + rank])
                    })
                }),
            })
        }
    }

    impl Drop for D21WitnessCuda {
        fn drop(&mut self) {
            unsafe { adynkra_d21_witness_destroy(self.context.as_ptr()) };
        }
    }

    fn message(buffer: &[i8]) -> String {
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use num_bigint::BigInt;
    use num_complex::Complex;
    use num_rational::{BigRational, Ratio};
    use num_traits::{ToPrimitive, Zero};
    use std::collections::{BTreeMap, BTreeSet};

    use crate::eleven_dimensional_corrected_full_chain_oracle::{
        FullChainRowKey, corrected_full_chain_streams,
    };
    use crate::eleven_dimensional_d21_invariant_diagrams::{
        D21SectorPivotReplayRequestV2, d21_source_lorentz_generator_terms,
        decode_source_coordinate, flattened_gamma_mask_tables, replay_sector_pivot_v2,
    };
    use crate::eleven_dimensional_dg4_casimir_projectors::{
        dg4_lorentz_generator_action_integer, project_dg4_target,
    };
    use crate::eleven_dimensional_four_form_56_physics_rows::{
        AUGMENTED_TARGET_COLUMN, bianchi_rhs_image, global57_rhs_batch,
        lexicographic_four_form_to_numeric, numeric_four_form_to_lexicographic,
        teleparallel_rhs_column,
    };
    use crate::eleven_dimensional_physical_curvature::ExactQi;

    fn numeric_target_to_teleparallel_lexicographic(target_coordinate: usize) -> usize {
        assert!(target_coordinate < 10_560);
        let spinor = target_coordinate / 330;
        let numeric_form = target_coordinate % 330;
        spinor * 330 + numeric_four_form_to_lexicographic(numeric_form).unwrap()
    }

    fn teleparallel_target_to_numeric(target_coordinate: usize) -> usize {
        assert!(target_coordinate < 10_560);
        let spinor = target_coordinate / 330;
        let lexicographic_form = target_coordinate % 330;
        spinor * 330 + lexicographic_four_form_to_numeric(lexicographic_form).unwrap()
    }

    fn exact_rhs_value(
        columns: &mut BTreeMap<
            usize,
            crate::eleven_dimensional_four_form_56_physics_rows::TeleparallelRhsColumn,
        >,
        source_coordinate: u32,
        target_coordinate: usize,
    ) -> BigQi {
        let (_, _, h_hat) = decode_source_coordinate(source_coordinate).unwrap();
        let column = columns
            .entry(h_hat)
            .or_insert_with(|| teleparallel_rhs_column(h_hat).unwrap());
        let row = u64::from(source_coordinate) * 10_560 + target_coordinate as u64;
        column
            .entries
            .binary_search_by_key(&row, |entry| entry.row)
            .ok()
            .map(|position| {
                let value = &column.entries[position].coefficient;
                Complex::new(
                    BigRational::new(
                        BigInt::from(value.real_numerator),
                        BigInt::from(value.real_denominator),
                    ),
                    BigRational::new(
                        BigInt::from(value.imaginary_numerator),
                        BigInt::from(value.imaginary_denominator),
                    ),
                )
            })
            .unwrap_or_else(|| Complex::new(BigRational::zero(), BigRational::zero()))
    }

    fn exact_rhs_slice(
        columns: &mut BTreeMap<
            usize,
            crate::eleven_dimensional_four_form_56_physics_rows::TeleparallelRhsColumn,
        >,
        source_coordinate: u32,
    ) -> BTreeMap<usize, BigQi> {
        let (_, _, h_hat) = decode_source_coordinate(source_coordinate).unwrap();
        let column = columns
            .entry(h_hat)
            .or_insert_with(|| teleparallel_rhs_column(h_hat).unwrap());
        let first_row = u64::from(source_coordinate) * 10_560;
        let last_row = first_row + 10_560;
        let start = column
            .entries
            .partition_point(|entry| entry.row < first_row);
        let end = column.entries.partition_point(|entry| entry.row < last_row);
        column.entries[start..end]
            .iter()
            .map(|entry| {
                let value = &entry.coefficient;
                (
                    usize::try_from(entry.row - first_row).unwrap(),
                    Complex::new(
                        BigRational::new(
                            BigInt::from(value.real_numerator),
                            BigInt::from(value.real_denominator),
                        ),
                        BigRational::new(
                            BigInt::from(value.imaginary_numerator),
                            BigInt::from(value.imaginary_denominator),
                        ),
                    ),
                )
            })
            .collect()
    }

    fn add_big_qi(output: &mut BTreeMap<usize, BigQi>, row: usize, value: BigQi) {
        if value.re.is_zero() && value.im.is_zero() {
            return;
        }
        let entry = output
            .entry(row)
            .or_insert_with(|| Complex::new(BigRational::zero(), BigRational::zero()));
        *entry += value;
        if entry.re.is_zero() && entry.im.is_zero() {
            output.remove(&row);
        }
    }

    fn teleparallel_lorentz_residual_slice(
        columns: &mut BTreeMap<
            usize,
            crate::eleven_dimensional_four_form_56_physics_rows::TeleparallelRhsColumn,
        >,
        source_coordinate: u32,
        left: usize,
        right: usize,
    ) -> BTreeMap<usize, BigQi> {
        let input = exact_rhs_slice(columns, source_coordinate);
        let mut residual = BTreeMap::new();
        for (input_target, coefficient) in input {
            let action = dg4_lorentz_generator_action_integer(
                left,
                right,
                &BTreeMap::from([(input_target, 1_i64)]),
            )
            .unwrap();
            for (target, value) in action {
                add_big_qi(
                    &mut residual,
                    target,
                    coefficient.clone() * BigRational::from_integer(BigInt::from(value)),
                );
            }
        }
        for term in d21_source_lorentz_generator_terms(source_coordinate, left, right).unwrap() {
            for (target, value) in exact_rhs_slice(columns, term.source_coordinate) {
                add_big_qi(
                    &mut residual,
                    target,
                    -value * BigRational::from_integer(BigInt::from(term.coefficient)),
                );
            }
        }
        residual
    }

    fn teleparallel_lorentz_residual_at_row(
        columns: &mut BTreeMap<
            usize,
            crate::eleven_dimensional_four_form_56_physics_rows::TeleparallelRhsColumn,
        >,
        source_coordinate: u32,
        target_coordinate: usize,
        left: usize,
        right: usize,
    ) -> BigQi {
        let (_, _, h_hat) = decode_source_coordinate(source_coordinate).unwrap();
        let column = columns
            .entry(h_hat)
            .or_insert_with(|| teleparallel_rhs_column(h_hat).unwrap());
        let first_row = u64::from(source_coordinate) * 10_560;
        let last_row = first_row + 10_560;
        let start = column
            .entries
            .partition_point(|entry| entry.row < first_row);
        let end = column.entries.partition_point(|entry| entry.row < last_row);
        let mut lhs = Complex::new(BigRational::zero(), BigRational::zero());
        for entry in &column.entries[start..end] {
            let input_target = usize::try_from(entry.row - first_row).unwrap();
            let action = dg4_lorentz_generator_action_integer(
                left,
                right,
                &BTreeMap::from([(input_target, 1_i64)]),
            )
            .unwrap();
            let Some(&action_value) = action.get(&target_coordinate) else {
                continue;
            };
            let value = &entry.coefficient;
            lhs += Complex::new(
                BigRational::new(
                    BigInt::from(value.real_numerator) * BigInt::from(action_value),
                    BigInt::from(value.real_denominator),
                ),
                BigRational::new(
                    BigInt::from(value.imaginary_numerator) * BigInt::from(action_value),
                    BigInt::from(value.imaginary_denominator),
                ),
            );
        }
        let mut rhs = Complex::new(BigRational::zero(), BigRational::zero());
        for term in d21_source_lorentz_generator_terms(source_coordinate, left, right).unwrap() {
            rhs += exact_rhs_value(columns, term.source_coordinate, target_coordinate)
                * BigRational::from_integer(BigInt::from(term.coefficient));
        }
        lhs - rhs
    }

    #[test]
    fn canonical_rows_round_trip_at_every_boundary() {
        for ordinal in [0, D21_ROW_COUNT - 1, D21_ROW_COUNT, TOTAL_ROW_COUNT - 1] {
            let row = CanonicalRow::from_ordinal(ordinal).unwrap();
            assert_eq!(row.ordinal().unwrap(), ordinal);
        }
        assert!(CanonicalRow::from_ordinal(TOTAL_ROW_COUNT).is_err());
    }

    #[test]
    fn exact_coo_layout_and_digest_are_stable() {
        assert_eq!(std::mem::size_of::<ExactCooEntry>(), 32);
        assert_eq!(std::mem::size_of::<ThreePrimeGaussian>(), 24);
        let entries = [ExactCooEntry {
            row: D21_ROW_COUNT,
            column: 52,
            reserved: 0,
            real: -3,
            imaginary: 4,
        }];
        entries[0].validate().unwrap();
        assert_eq!(exact_coo_sha256(&entries, 5), exact_coo_sha256(&entries, 5));
        assert_ne!(exact_coo_sha256(&entries, 5), exact_coo_sha256(&entries, 7));
    }

    #[test]
    #[ignore = "exact corrected teleparallel D21 Lorentz commutator witness"]
    fn corrected_teleparallel_d21_lorentz_commutator_witness() {
        let canonical_row = 1_392_410_608_u64;
        let source_coordinate = u32::try_from(canonical_row / 10_560).unwrap();
        let target_coordinate = usize::try_from(canonical_row % 10_560).unwrap();
        assert_eq!(source_coordinate, 131_857);
        assert_eq!(target_coordinate, 688);
        let mut columns = BTreeMap::new();
        let mut nonzero_residuals = Vec::new();
        let mut total_residual_rows = 0_usize;
        let mut generator_residual_counts = Vec::new();
        let mut first_residual = None;
        let mut residual_hash = Sha256::new();
        residual_hash.update(b"adynkra-11d-teleparallel-d21-lorentz-residual-v1\0");
        for left in 0..11 {
            for right in (left + 1)..11 {
                let residual = teleparallel_lorentz_residual_slice(
                    &mut columns,
                    source_coordinate,
                    left,
                    right,
                );
                total_residual_rows += residual.len();
                generator_residual_counts.push(serde_json::json!({
                    "left": left,
                    "right": right,
                    "residual_rows": residual.len(),
                }));
                for (&target, value) in &residual {
                    residual_hash.update((left as u32).to_le_bytes());
                    residual_hash.update((right as u32).to_le_bytes());
                    residual_hash.update((target as u32).to_le_bytes());
                    residual_hash.update(value.re.numer().to_signed_bytes_le());
                    residual_hash.update([0xff]);
                    residual_hash.update(value.re.denom().to_signed_bytes_le());
                    residual_hash.update([0xfe]);
                    residual_hash.update(value.im.numer().to_signed_bytes_le());
                    residual_hash.update([0xfd]);
                    residual_hash.update(value.im.denom().to_signed_bytes_le());
                    residual_hash.update([0xfc]);
                    if first_residual.is_none() {
                        first_residual = Some((left, right, target, value.clone()));
                    }
                }
                if let Some(value) = residual.get(&target_coordinate) {
                    nonzero_residuals.push((left, right, value.clone()));
                }
            }
        }
        eprintln!(
            "TELEPARALLEL_D21_LORENTZ row={canonical_row} generators=55 cached_h_columns={} total_residual_rows={total_residual_rows} witness_row_residuals={:?}",
            columns.len(),
            nonzero_residuals
        );
        assert_eq!(total_residual_rows, 1_032);
        assert!(nonzero_residuals.is_empty());
        let (first_left, first_right, first_target, first_value) = first_residual.unwrap();
        let residual_sha256 = format!("{:x}", residual_hash.finalize());
        let report = serde_json::json!({
            "schema_version": "adynkra-11d-teleparallel-d21-lorentz-commutator-v1",
            "passed": true,
            "lorentz_equivariant": false,
            "physics_promotion_blocked": true,
            "source_coordinate": source_coordinate,
            "source_parts": {"outer_pair": [1,8], "momentum": 5, "h_hat": 17},
            "decisive_augmented_row": canonical_row,
            "decisive_row_target_coordinate": target_coordinate,
            "decisive_row_residuals": nonzero_residuals.len(),
            "generators_checked": 55,
            "cached_h_hat_columns": columns.len(),
            "total_residual_rows": total_residual_rows,
            "residual_stream_sha256": residual_sha256,
            "first_residual": {
                "generator": [first_left, first_right],
                "target_coordinate": first_target,
                "canonical_row": u64::from(source_coordinate) * 10_560 + first_target as u64,
                "real_numerator": first_value.re.numer().to_string(),
                "real_denominator": first_value.re.denom().to_string(),
                "imaginary_numerator": first_value.im.numer().to_string(),
                "imaginary_denominator": first_value.im.denom().to_string(),
            },
            "generator_residual_counts": generator_residual_counts,
            "source_action_oracle": {
                "known_invariant_diagrams": [0,21,238],
                "generators_per_diagram": 55,
                "residual_rows": 0,
                "test": "d21_full_source_generator_matches_invariant_diagrams",
            },
            "quarantined_global57_sha256": "5645e60a1e81222b654edc25b488ca5874fecc1919ac0890cc538b7c34a77e1f",
            "source_sha256": {
                "d21_source_action": format!("{:x}", Sha256::digest(std::fs::read("src/eleven_dimensional_d21_invariant_diagrams.rs").unwrap())),
                "gpu_audit": format!("{:x}", Sha256::digest(std::fs::read("src/eleven_dimensional_four_form_56_gpu.rs").unwrap())),
                "physics_rows": format!("{:x}", Sha256::digest(std::fs::read("src/eleven_dimensional_four_form_56_physics_rows.rs").unwrap())),
            },
            "outcome": "gauge_fixed_non_equivariant_target_not_a_d21_hom_member",
            "next_executable_step": "Construct the exact local-Lorentz or source-gauge image, reduce this commutator cocycle in the declared quotient, and require zero residual before rerunning the 56-column coefficient solve.",
            "boundary": "This exact witness-source audit rejects the corrected teleparallel stream as a Lorentz intertwiner into raw D G4. It does not prove that the stream fails after a correctly constructed local-Lorentz or target-gauge quotient. The earlier rank-53 augmented witness is not a physical no-go because it compared an equivariant Hom basis with a non-equivariant gauge-fixed target.",
        });
        let path = "results/adynkra_11d_teleparallel_d21_lorentz_commutator.json";
        let temporary = format!("{path}.tmp-{}", std::process::id());
        std::fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        std::fs::rename(temporary, path).unwrap();
        assert!(
            nonzero_residuals.is_empty(),
            "the literal decisive target coordinate should remain the zero-residual mutation guard"
        );
    }

    #[test]
    fn denominator_gate_rejects_a_pinned_prime() {
        let digest = "a".repeat(64);
        let identity = FourForm56ManifestIdentity {
            schema_version: SCHEMA_VERSION.to_string(),
            semantic_sha256: digest.clone(),
            source_sha256: digest.clone(),
            source_basis_sha256: digest.clone(),
            target_basis_sha256: digest.clone(),
            coefficient_inventory_sha256: digest.clone(),
            generator_map_sha256: digest.clone(),
            normal_form_sha256: digest.clone(),
            common_denominator: u64::from(PINNED_PRIMES[0]),
            denominator_audit_sha256: digest,
            ordered_primes: PINNED_PRIMES,
            row_schema_version: ROW_SCHEMA_VERSION.to_string(),
            columns: COLUMN_COUNT,
            rows: TOTAL_ROW_COUNT,
            block_rows: 4096,
        };
        assert!(identity.validate().is_err());
    }

    #[test]
    fn incomplete_or_misordered_column_bindings_fail_closed() {
        let binding = FourForm56ColumnBinding {
            schema_version: COLUMN_BINDING_SCHEMA_VERSION.to_string(),
            global_column: 52,
            branch: BidegreeBranch::D0P2,
            dynkin_label: "00001".to_string(),
            multiplicity_copy: 1,
            generator_schema_version: "canary-v1".to_string(),
            generator_source_sha256: "a".repeat(64),
            source_basis_sha256: "b".repeat(64),
            target_basis_sha256: "c".repeat(64),
            coefficient_inventory_sha256: "d".repeat(64),
            projector_sha256: "e".repeat(64),
            exact_stream_entries: 1,
            exact_stream_sha256: "f".repeat(64),
        };
        binding.validate().unwrap();
        assert!(ordered_generator_map_sha256(&[binding.clone()]).is_err());
        let mut wrong_branch = binding;
        wrong_branch.branch = BidegreeBranch::D2P1;
        assert!(wrong_branch.validate().is_err());
    }

    fn gaussian_residues(real: i64, imaginary: i64) -> ThreePrimeGaussian {
        let mut output = ThreePrimeGaussian::default();
        for (slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
            output.value[2 * slot] = real.rem_euclid(i64::from(prime)) as u32;
            output.value[2 * slot + 1] = imaginary.rem_euclid(i64::from(prime)) as u32;
        }
        output
    }

    #[test]
    fn streamed_candidate_rref_finds_consensus_seed_copies() {
        let key = |row: u64, column: u64| row * u64::from(COLUMN_COUNT) + column;
        let entries = vec![
            (key(10, 0), gaussian_residues(1, 0)),
            (key(10, 2), gaussian_residues(1, 0)),
            (key(20, 1), gaussian_residues(0, 1)),
            (key(20, 2), gaussian_residues(0, 1)),
            (key(30, 3), gaussian_residues(2, -3)),
        ];
        let mut rref = StreamedCandidateRref::new(4).unwrap();
        rref.push_reduced_block(&entries[..2]).unwrap();
        rref.push_reduced_block(&entries[2..]).unwrap();
        let summary = rref.summary();
        assert_eq!(summary.per_prime_ranks, [3, 3, 3]);
        assert_eq!(summary.consensus_pivot_columns, Some(vec![0, 1, 3]));

        let mut split = StreamedCandidateRref::new(4).unwrap();
        split.push_reduced_block(&entries[..3]).unwrap();
        assert!(split.push_reduced_block(&entries[3..]).is_err());
    }

    #[test]
    fn d21_column_inventory_is_sector_and_copy_canonical() {
        let expected = [
            (0, "00001", 1),
            (6, "00001", 7),
            (7, "00011", 1),
            (13, "00011", 7),
            (14, "00101", 1),
            (24, "00101", 11),
            (25, "01001", 1),
            (38, "01001", 14),
            (39, "10001", 1),
            (51, "10001", 13),
        ];
        for (column, label, copy) in expected {
            assert_eq!(expected_column_identity(column), Some((label, copy)));
        }
        assert_eq!(
            D21_SECTORS
                .iter()
                .map(|sector| sector.multiplicity)
                .sum::<u32>(),
            52
        );
    }

    #[test]
    fn d21_compact_device_blob_and_signed_permutation_tables_are_frozen() {
        let blob = canonical_d21_device_diagram_blob().unwrap();
        assert_eq!(
            blob,
            include_bytes!("../results/adynkra_11d_d21_device_diagrams_v1.bin")
        );
        let (rows, values) = d21_signed_permutation_tables().unwrap();
        assert_eq!(rows[0].len(), 2048 * 32);
        assert_eq!(rows[1].len(), 2048 * 32);
        assert_eq!(values[0].len(), 2048 * 32);
        assert_eq!(values[1].len(), 2048 * 32);
        assert_eq!(rows.iter().map(Vec::len).sum::<usize>(), 131_072);
        assert_eq!(values.iter().map(Vec::len).sum::<usize>(), 131_072);
    }

    #[test]
    fn d21_ingestion_contract_binds_rows_slots_denominator_and_digest() {
        let entries = vec![
            ExactCooEntry {
                row: 10,
                column: 0,
                reserved: 0,
                real: 3,
                imaginary: 0,
            },
            ExactCooEntry {
                row: 10,
                column: 6,
                reserved: 0,
                real: -2,
                imaginary: 1,
            },
            ExactCooEntry {
                row: 11,
                column: 1,
                reserved: 0,
                real: 5,
                imaginary: -4,
            },
        ];
        let digest = |letter: char| letter.to_string().repeat(64);
        let contract = D21GpuIngestionContract {
            schema_version: D21_INGESTION_SCHEMA_VERSION.to_string(),
            sector: "00001".to_string(),
            candidate_seed_count: 7,
            candidate_inventory_sha256: digest('a'),
            source_basis_sha256: digest('b'),
            target_basis_sha256: digest('c'),
            normal_form_sha256: digest('d'),
            common_denominator: 3,
            first_row_inclusive: 10,
            last_row_exclusive: 12,
            complete_row_interval: true,
            input_entries: entries.len() as u64,
            exact_coo_sha256: exact_coo_sha256(&entries, 3),
        };
        contract.validate_entries(&entries).unwrap();
        let mut split_row = contract.clone();
        split_row.complete_row_interval = false;
        assert!(split_row.validate_entries(&entries).is_err());
        let mut wrong_digest = contract;
        wrong_digest.exact_coo_sha256 = "f".repeat(64);
        assert!(wrong_digest.validate_entries(&entries).is_err());
    }

    #[test]
    fn d21_exact_pivot_minor_closes_three_prime_witness() {
        let selected = (0_u32..7).collect::<Vec<_>>();
        let rows = (0_u64..7).collect::<Vec<_>>();
        let replay = replay_exact_d21_pivot_minor("00001", &selected, &rows, 3, |row, seed| {
            Ok((if row == u64::from(seed) { 3 } else { 0 }, 0))
        })
        .unwrap();
        assert!(replay.determinant_nonzero);
        assert_eq!(replay.determinant_real, "1");
        assert_eq!(replay.determinant_imaginary, "0");

        let pivots = selected.clone();
        let checkpoint = D21SectorRankCheckpoint {
            schema_version: D21_RREF_CHECKPOINT_SCHEMA_VERSION.to_string(),
            sector: "00001".to_string(),
            candidate_seed_count: 7,
            candidate_inventory_sha256: "a".repeat(64),
            source_basis_sha256: "b".repeat(64),
            target_basis_sha256: "c".repeat(64),
            normal_form_sha256: "d".repeat(64),
            generation: 1,
            blocks_completed: 1,
            seed_inventory_complete_and_equivariant: true,
            completion_reason: Some("expected_rank_witness".to_string()),
            rref: CandidateRrefSummary {
                candidate_columns: 7,
                canonical_rows_consumed: 7,
                reduced_terms_consumed: 7,
                per_prime_ranks: [7, 7, 7],
                per_prime_pivot_columns: [pivots.clone(), pivots.clone(), pivots.clone()],
                per_prime_pivot_rows: [rows.clone(), rows.clone(), rows.clone()],
                consensus_pivot_columns: Some(pivots),
            },
            exact_pivot_minor: Some(replay),
            complete: true,
        };
        checkpoint.validate().unwrap();
        let mut missing_exact = checkpoint;
        missing_exact.exact_pivot_minor = None;
        assert!(missing_exact.validate().is_err());
    }

    #[test]
    fn d21_scalar_gpu_witness_minor_collapses_after_variance_fixes() {
        use crate::eleven_dimensional_d21_invariant_diagrams::{
            D21SectorPivotReplayRequestV2, replay_sector_pivot_v2,
        };
        let rows = [(56_320_u32, 14_u16), (991_077_u32, 211_u16)];
        let diagrams = [0_u16, 12_u16];
        let inventory = crate::eleven_dimensional_d21_invariant_diagrams::enumerate_diagrams();
        eprintln!(
            "D21_SCALAR_EXCESS_DIAGRAMS zero={:?} twelve={:?}",
            inventory[0], inventory[12]
        );
        let matrix = rows.map(|(source_coordinate, target_coordinate)| {
            diagrams.map(|diagram_ordinal| {
                replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                    source_coordinate,
                    target_coordinate,
                    diagram_ordinal,
                    target_sector: "01001".to_string(),
                })
                .unwrap()
                .projected_numerator
            })
        });
        let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        eprintln!("D21_SCALAR_EXCESS_CPU matrix={matrix:?} determinant={determinant}");
        assert_eq!(matrix, [[-1_032_192, -3_096_576], [-2_580_480, -7_741_440]]);
        assert_eq!(
            determinant, 0,
            "the exact scalar-channel Hom multiplicity is one"
        );
    }

    #[test]
    fn real_d02_00001_report_binds_global_column_52() {
        let report = crate::eleven_dimensional_d02_00001_generator::build_report().unwrap();
        let binding =
            crate::eleven_dimensional_d02_00001_generator::column_binding(&report).unwrap();
        assert_eq!(binding.global_column, 52);
        assert_eq!(binding.branch, BidegreeBranch::D0P2);
        assert_eq!(binding.dynkin_label, "00001");
        assert_eq!(binding.exact_stream_entries, 2_217_600);
        assert_eq!(
            binding.exact_stream_sha256,
            "c0b3849a01ed2083848cd78339e4f4fcd51921bb8e73eee5e553d3dc3bf133ed"
        );
    }

    #[test]
    #[ignore = "full 2,217,600-row exact CPU stream canary"]
    fn real_d02_00001_stream_reduces_exactly_on_cpu() {
        const BATCH: usize = 200_000;
        fn verify_batch(entries: &[ExactCooEntry]) -> Result<(), String> {
            let reduced = reduce_reference(entries, 1)?;
            if reduced.len() != entries.len() {
                return Err("D02 00001 CPU reduction changed unique row count".to_string());
            }
            for (entry, (key, value)) in entries.iter().zip(reduced) {
                if key != entry.row * u64::from(COLUMN_COUNT) + u64::from(entry.column)
                    || value.is_zero_at_every_prime()
                {
                    return Err("D02 00001 CPU modular parity failed".to_string());
                }
            }
            Ok(())
        }

        let mut entries = Vec::with_capacity(BATCH);
        let mut reduced_terms = 0_u64;
        let mut previous_row = None;
        let (emitted, _, stream_sha256, _) =
            crate::eleven_dimensional_d02_00001_generator::visit_d02_00001_generator(|entry| {
                if entry.column != 52 || previous_row.is_some_and(|row| entry.row <= row) {
                    return Err("D02 00001 stream is not canonical global column 52".to_string());
                }
                previous_row = Some(entry.row);
                entries.push(entry);
                if entries.len() == BATCH {
                    verify_batch(&entries)?;
                    reduced_terms += entries.len() as u64;
                    entries.clear();
                }
                Ok(())
            })
            .unwrap();
        verify_batch(&entries).unwrap();
        reduced_terms += entries.len() as u64;
        assert_eq!(emitted, 2_217_600);
        assert_eq!(reduced_terms, emitted);
        assert_eq!(
            stream_sha256,
            "c0b3849a01ed2083848cd78339e4f4fcd51921bb8e73eee5e553d3dc3bf133ed"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit RTX 4090 full-stream CPU/GPU parity canary"]
    fn real_d02_00001_stream_cuda_matches_cpu() {
        use super::cuda::FourForm56CudaReducer;

        const BATCH: usize = 250_000;
        let mut reducer = FourForm56CudaReducer::new(BATCH, 2_u64 << 30).unwrap();
        let mut entries = Vec::with_capacity(BATCH);
        let mut compared = 0_u64;
        let (emitted, _, stream_sha256, _) =
            crate::eleven_dimensional_d02_00001_generator::visit_d02_00001_generator(|entry| {
                entries.push(entry);
                if entries.len() == BATCH {
                    let expected = reduce_reference(&entries, 1)?;
                    let (actual, input_terms) = reducer.reduce(&entries, 1)?;
                    if input_terms != entries.len() as u64 || actual != expected {
                        return Err("D02 00001 CUDA reduction differs from CPU".to_string());
                    }
                    compared += input_terms;
                    entries.clear();
                }
                Ok(())
            })
            .unwrap();
        if !entries.is_empty() {
            let expected = reduce_reference(&entries, 1).unwrap();
            let (actual, input_terms) = reducer.reduce(&entries, 1).unwrap();
            assert_eq!(actual, expected);
            compared += input_terms;
        }
        assert_eq!(emitted, 2_217_600);
        assert_eq!(compared, emitted);
        assert_eq!(
            stream_sha256,
            "c0b3849a01ed2083848cd78339e4f4fcd51921bb8e73eee5e553d3dc3bf133ed"
        );
    }

    #[test]
    #[ignore = "full columns 52 through 55 exact CPU stream canary"]
    fn real_d02_complete_stream_reduces_exactly_on_cpu() {
        const BATCH: usize = 250_000;
        fn verify_batch(entries: &[ExactCooEntry], column: u32) -> Result<(), String> {
            let reduced = reduce_reference(entries, 1)?;
            if reduced.len() != entries.len() {
                return Err(format!(
                    "D02 column {column} CPU reduction changed unique row count"
                ));
            }
            for (entry, (key, value)) in entries.iter().zip(reduced) {
                if key != entry.row * u64::from(COLUMN_COUNT) + u64::from(column)
                    || value.is_zero_at_every_prime()
                {
                    return Err(format!("D02 column {column} CPU modular parity failed"));
                }
            }
            Ok(())
        }
        let mut total = 0_u64;
        let mut expected_counts = Vec::new();
        for column in 52_u32..=55 {
            let mut entries = Vec::with_capacity(BATCH);
            let mut column_total = 0_u64;
            if column == 52 {
                crate::eleven_dimensional_d02_00001_generator::visit_d02_00001_generator(|entry| {
                    entries.push(entry);
                    if entries.len() == BATCH {
                        verify_batch(&entries, column)?;
                        column_total += entries.len() as u64;
                        entries.clear();
                    }
                    Ok(())
                })
                .unwrap();
            } else {
                crate::eleven_dimensional_d02_00001_generator::visit_remaining_stream(
                    column,
                    |entry| {
                        entries.push(entry);
                        if entries.len() == BATCH {
                            verify_batch(&entries, column)?;
                            column_total += entries.len() as u64;
                            entries.clear();
                        }
                        Ok(())
                    },
                )
                .unwrap();
            }
            verify_batch(&entries, column).unwrap();
            column_total += entries.len() as u64;
            expected_counts.push(column_total);
            total += column_total;
        }
        assert_eq!(
            expected_counts,
            vec![2_217_600, 3_669_120, 591_360, 2_217_600]
        );
        assert_eq!(total, 8_695_680);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit RTX 4090 columns 52 through 55 CPU/GPU parity"]
    fn real_d02_complete_stream_cuda_matches_cpu() {
        use super::cuda::FourForm56CudaReducer;

        const BATCH: usize = 250_000;
        fn verify_cuda_batch(
            reducer: &mut FourForm56CudaReducer,
            entries: &[ExactCooEntry],
            column: u32,
        ) -> Result<u64, String> {
            let expected = reduce_reference(entries, 1)?;
            let (actual, input_terms) = reducer.reduce(entries, 1)?;
            if actual != expected || input_terms != entries.len() as u64 {
                return Err(format!(
                    "D02 column {column} CUDA reduction differs from CPU"
                ));
            }
            Ok(input_terms)
        }
        let mut reducer = FourForm56CudaReducer::new(BATCH, 2_u64 << 30).unwrap();
        let mut totals = Vec::new();
        for column in 52_u32..=55 {
            let mut entries = Vec::with_capacity(BATCH);
            let mut compared = 0_u64;
            if column == 52 {
                crate::eleven_dimensional_d02_00001_generator::visit_d02_00001_generator(|entry| {
                    entries.push(entry);
                    if entries.len() == BATCH {
                        compared += verify_cuda_batch(&mut reducer, &entries, column)?;
                        entries.clear();
                    }
                    Ok(())
                })
                .unwrap();
            } else {
                crate::eleven_dimensional_d02_00001_generator::visit_remaining_stream(
                    column,
                    |entry| {
                        entries.push(entry);
                        if entries.len() == BATCH {
                            compared += verify_cuda_batch(&mut reducer, &entries, column)?;
                            entries.clear();
                        }
                        Ok(())
                    },
                )
                .unwrap();
            }
            if !entries.is_empty() {
                compared += verify_cuda_batch(&mut reducer, &entries, column).unwrap();
            }
            totals.push(compared);
        }
        assert_eq!(totals, vec![2_217_600, 3_669_120, 591_360, 2_217_600]);
        eprintln!(
            "D02_COMPLETE_CUDA_PARITY totals={totals:?} resident={} high_water={}",
            reducer.resident_bytes(),
            reducer.high_water_bytes()
        );
    }

    #[test]
    fn synthetic_reference_covers_all_columns_and_bidegrees() {
        let (entries, denominator) = synthetic_generator_fixture();
        let output = reduce_reference(&entries, denominator).unwrap();
        assert_eq!(output.len(), COLUMN_COUNT as usize);
        assert!(
            output
                .iter()
                .any(|(key, _)| key / u64::from(COLUMN_COUNT) < D21_ROW_COUNT)
        );
        assert!(
            output
                .iter()
                .any(|(key, _)| key / u64::from(COLUMN_COUNT) >= D21_ROW_COUNT)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn synthetic_cuda_collision_and_cancellation_canary() {
        use super::cuda::FourForm56CudaReducer;
        let (entries, denominator) = synthetic_generator_fixture();
        let expected = reduce_reference(&entries, denominator).unwrap();
        let mut reducer = FourForm56CudaReducer::new(entries.len(), 1 << 30).unwrap();
        let (output, input_terms) = reducer.reduce(&entries, denominator).unwrap();
        assert_eq!(input_terms, entries.len() as u64);
        assert_eq!(output, expected);
        assert!(reducer.high_water_bytes() >= reducer.resident_bytes());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit RTX 4090 throughput canary"]
    fn synthetic_cuda_million_term_throughput() {
        use super::cuda::FourForm56CudaReducer;
        use std::time::Instant;
        let count = std::env::var("ADYNKRA_FOUR_FORM56_BENCH_TERMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1_000_000);
        let entries = (0..count)
            .map(|ordinal| ExactCooEntry {
                row: (ordinal as u64 * 97) % D21_ROW_COUNT,
                column: (ordinal as u32) % COLUMN_COUNT,
                reserved: 0,
                real: (ordinal as i64 % 257) - 128,
                imaginary: (ordinal as i64 % 251) - 125,
            })
            .collect::<Vec<_>>();
        let mut reducer = FourForm56CudaReducer::new(count, 4_u64 << 30).unwrap();
        let started = Instant::now();
        let (output, input_terms) = reducer.reduce(&entries, 3).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(input_terms, count as u64);
        assert!(!output.is_empty());
        eprintln!(
            "four-form56 CUDA: input_terms={} unique_terms={} elapsed_ms={} terms_per_second={:.3} resident_bytes={} high_water_bytes={}",
            count,
            output.len(),
            elapsed.as_millis(),
            count as f64 / elapsed.as_secs_f64(),
            reducer.resident_bytes(),
            reducer.high_water_bytes(),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit RTX 4090 D21 exact coefficient-query throughput canary"]
    fn d21_cuda_million_exact_coefficient_queries_match_cpu() {
        use super::cuda::D21CudaEvaluator;
        use crate::eleven_dimensional_d21_invariant_diagrams::{
            D21CoefficientQuery, evaluate_packed_query_cpu, packed_diagrams,
        };
        use std::time::Instant;

        let count = std::env::var("ADYNKRA_D21_BENCH_QUERIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1_000_000);
        let forms = (0_u16..2048)
            .filter(|mask| mask.count_ones() == 4)
            .map(|mask| {
                let axes = (0..11)
                    .filter(|axis| mask & (1_u16 << axis) != 0)
                    .map(|axis| axis as u8)
                    .collect::<Vec<_>>();
                [axes[0], axes[1], axes[2], axes[3]]
            })
            .collect::<Vec<_>>();
        assert_eq!(forms.len(), 330);
        let queries = (0..count)
            .map(|ordinal| {
                let pair_left = (ordinal / (400 * 11)) % 31;
                D21CoefficientQuery {
                    diagram: (ordinal % 400) as u16,
                    outer_left: pair_left as u8,
                    outer_right: (pair_left + 1) as u8,
                    momentum: ((ordinal / 400) % 11) as u8,
                    h_vector: ((ordinal / (400 * 11 * 31)) % 11) as u8,
                    output_axes: forms[(ordinal * 17 + ordinal / 400) % forms.len()],
                    input_spinor: ((ordinal * 7 + 3) % 32) as u8,
                    output_spinor: ((ordinal * 13 + 5) % 32) as u8,
                    h_coefficient: if ordinal & 1 == 0 { 1 } else { -1 },
                    reserved: 0,
                }
            })
            .collect::<Vec<_>>();
        let diagrams = packed_diagrams();
        let parity_count = queries.len().min(10_000);
        let expected = queries[..parity_count]
            .iter()
            .copied()
            .map(|query| evaluate_packed_query_cpu(&diagrams, query))
            .collect::<Vec<_>>();
        let mut evaluator = D21CudaEvaluator::new(count).unwrap();
        let started = Instant::now();
        let (actual, kernel_milliseconds) = evaluator.evaluate(&queries).unwrap();
        let wall = started.elapsed();
        assert_eq!(&actual[..parity_count], expected);
        let nonzero = actual.iter().filter(|&&value| value != 0).count();
        eprintln!(
            "D21_CUDA_BENCH queries={} nonzero={} kernel_ms={:.6} kernel_queries_per_second={:.3} wall_ms={} wall_queries_per_second={:.3}",
            count,
            nonzero,
            kernel_milliseconds,
            count as f64 * 1000.0 / f64::from(kernel_milliseconds),
            wall.as_millis(),
            count as f64 / wall.as_secs_f64(),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit RTX 4090 D21 witness-first projector and retained-RREF run"]
    fn d21_cuda_witness_first_sector_ranks() {
        use super::cuda::{D21HTerm, D21WitnessCuda};
        use crate::eleven_dimensional_d21_invariant_diagrams::{
            D21SectorPivotReplayRequestV2, replay_sector_pivot_v2,
        };
        use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;

        fn bareiss_determinant(mut matrix: Vec<Vec<BigInt>>) -> BigInt {
            let dimension = matrix.len();
            if dimension == 0 {
                return BigInt::from(1);
            }
            let mut sign = BigInt::from(1);
            let mut previous = BigInt::from(1);
            for pivot in 0..dimension - 1 {
                let row = (pivot..dimension)
                    .find(|&row| matrix[row][pivot] != BigInt::from(0))
                    .expect("selected D21 exact pivot minor became singular");
                if row != pivot {
                    matrix.swap(row, pivot);
                    sign = -sign;
                }
                let pivot_value = matrix[pivot][pivot].clone();
                for row in pivot + 1..dimension {
                    for column in pivot + 1..dimension {
                        let numerator = matrix[row][column].clone() * pivot_value.clone()
                            - matrix[row][pivot].clone() * matrix[pivot][column].clone();
                        assert_eq!(&numerator % &previous, BigInt::from(0));
                        matrix[row][column] = numerator / &previous;
                    }
                }
                previous = pivot_value;
            }
            sign * matrix[dimension - 1][dimension - 1].clone()
        }

        let h_basis = canonical_gamma_traceless_frame_basis();
        let pairs = (0_u8..32)
            .flat_map(|left| ((left + 1)..32).map(move |right| [left, right]))
            .collect::<Vec<_>>();
        let maximum_witnesses = std::env::var("ADYNKRA_D21_MAX_WITNESSES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64);
        let labels = ["00001", "00011", "00101", "01001", "10001"];
        let channel_expected = [
            (0_u8, [1_u32, 0, 0, 1, 1]),
            (3_u8, [3_u32, 2, 5, 6, 6]),
            (4_u8, [3_u32, 5, 6, 7, 6]),
        ];
        let mut channel_reports = Vec::new();
        for (channel_index, (outer_degree, expected)) in channel_expected.into_iter().enumerate() {
            let mut context = D21WitnessCuda::new(outer_degree).unwrap();
            let mut total_kernel_ms = 0_f32;
            let mut witnesses = 0_usize;
            let sentinel_caps = expected.map(|rank| rank + 1);
            loop {
                let summary = context.summary().unwrap();
                let consensus = std::array::from_fn::<_, 5, _>(|sector| {
                    assert_eq!(summary.ranks[sector][0], summary.ranks[sector][1]);
                    assert_eq!(summary.ranks[sector][1], summary.ranks[sector][2]);
                    summary.ranks[sector][0]
                });
                for sector in 0..5 {
                    if consensus[sector] > expected[sector] {
                        let rank = consensus[sector] as usize;
                        eprintln!(
                            "D21_EXCESS_RANK outer_degree={} sector={} ranks={:?} pivots={:?} pivot_rows={:?}",
                            outer_degree,
                            sector,
                            summary.ranks[sector],
                            &summary.pivots[sector][0][..rank],
                            &summary.pivot_rows[sector][0][..rank],
                        );
                    }
                    assert!(
                        consensus[sector] <= expected[sector],
                        "D21 channel {outer_degree} sector {sector} exceeds its exact Hom multiplicity: observed {}, expected {}",
                        consensus[sector],
                        expected[sector]
                    );
                }
                if witnesses == maximum_witnesses {
                    eprintln!(
                        "D21_WITNESS_RANK outer_degree={} candidates={} witnesses={} ranks={:?} expected={:?} kernel_ms={:.6}",
                        outer_degree,
                        context.candidate_diagrams().len(),
                        witnesses,
                        consensus,
                        expected,
                        total_kernel_ms,
                    );
                    assert_eq!(consensus, expected);
                    let mut sector_reports = Vec::new();
                    for sector in 0..5 {
                        let rank = expected[sector] as usize;
                        assert_eq!(
                            &summary.pivots[sector][0][..rank],
                            &summary.pivots[sector][1][..rank]
                        );
                        assert_eq!(
                            &summary.pivots[sector][1][..rank],
                            &summary.pivots[sector][2][..rank]
                        );
                        assert_eq!(
                            &summary.pivot_rows[sector][0][..rank],
                            &summary.pivot_rows[sector][1][..rank]
                        );
                        assert_eq!(
                            &summary.pivot_rows[sector][1][..rank],
                            &summary.pivot_rows[sector][2][..rank]
                        );
                        if rank == 0 {
                            sector_reports.push(serde_json::json!({
                                "sector": labels[sector],
                                "expected_rank": 0,
                                "per_prime_ranks": summary.ranks[sector],
                                "selected_local_candidates": [],
                                "selected_diagram_ordinals": [],
                                "witness_rows": [],
                                "exact_determinant": null,
                            }));
                            continue;
                        }
                        let local = summary.pivots[sector][0][..rank].to_vec();
                        let selected = local
                            .iter()
                            .map(|&pivot| context.candidate_diagrams()[usize::from(pivot)])
                            .collect::<Vec<_>>();
                        let rows = summary.pivot_rows[sector][0][..rank].to_vec();
                        let mut matrix_hash = Sha256::new();
                        matrix_hash.update(b"adynkra-11d-d21-exact-sector-minor-v1\0");
                        matrix_hash.update([outer_degree]);
                        matrix_hash.update(labels[sector].as_bytes());
                        let mut denominator = None;
                        let matrix = rows
                            .iter()
                            .map(|&row| {
                                let source_coordinate = u32::try_from(row / 10_560).unwrap();
                                let target_coordinate = u16::try_from(row % 10_560).unwrap();
                                matrix_hash.update(row.to_le_bytes());
                                selected
                                    .iter()
                                    .map(|&diagram_ordinal| {
                                        let replay =
                                            replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                                                source_coordinate,
                                                target_coordinate,
                                                diagram_ordinal,
                                                target_sector: labels[sector].to_string(),
                                            })
                                            .unwrap();
                                        denominator = Some(replay.projector_denominator);
                                        matrix_hash.update(diagram_ordinal.to_le_bytes());
                                        matrix_hash
                                            .update(replay.projected_numerator.to_le_bytes());
                                        BigInt::from(replay.projected_numerator)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let determinant = bareiss_determinant(matrix);
                        assert_ne!(determinant, BigInt::from(0));
                        let prior_copies: u32 = channel_expected[..channel_index]
                            .iter()
                            .map(|(_, ranks)| ranks[sector])
                            .sum();
                        let global_columns = (0..rank)
                            .map(|copy| {
                                D21_SECTORS[sector].first_global_column + prior_copies + copy as u32
                            })
                            .collect::<Vec<_>>();
                        sector_reports.push(serde_json::json!({
                            "sector": labels[sector],
                            "expected_rank": rank,
                            "per_prime_ranks": summary.ranks[sector],
                            "selected_local_candidates": local,
                            "selected_diagram_ordinals": selected,
                            "global_columns": global_columns,
                            "witness_rows": rows,
                            "projector_denominator": denominator.unwrap(),
                            "exact_matrix_sha256": format!("{:x}", matrix_hash.finalize()),
                            "exact_determinant": determinant.to_string(),
                            "exact_determinant_nonzero": true,
                        }));
                    }
                    channel_reports.push(serde_json::json!({
                        "outer_fierz_degree": outer_degree,
                        "candidate_diagrams": context.candidate_diagrams().len(),
                        "witnesses_scanned": witnesses,
                        "kernel_milliseconds": total_kernel_ms,
                        "sectors": sector_reports,
                    }));
                    break;
                }
                let pair = if witnesses == 0 {
                    [0, 17]
                } else {
                    pairs[(witnesses * 37) % pairs.len()]
                };
                let momentum = if witnesses == 0 {
                    0
                } else {
                    ((witnesses * 5) % 11) as u8
                };
                let h_ordinal = if witnesses == 0 {
                    0
                } else {
                    (witnesses * 17) % h_basis.len()
                };
                let terms = h_basis[h_ordinal]
                    .iter()
                    .map(|(&coordinate, value)| {
                        assert_eq!(*value.real.denom(), 1);
                        assert_eq!(value.imaginary, Ratio::from_integer(0));
                        D21HTerm {
                            input_spinor: (coordinate / 11) as u8,
                            h_vector: (coordinate % 11) as u8,
                            coefficient: i16::try_from(*value.real.numer()).unwrap(),
                        }
                    })
                    .collect::<Vec<_>>();
                total_kernel_ms += context
                    .apply(pair, momentum, h_ordinal as u32, &terms, sentinel_caps)
                    .unwrap();
                witnesses += 1;
            }
        }
        if let Ok(path) = std::env::var("ADYNKRA_D21_WITNESS_REPORT") {
            let report = serde_json::json!({
                "schema_version": "adynkra-11d-d21-gpu-witness-rank-v1",
                "passed": true,
                "device": "RTX 4090",
                "ordered_primes": PINNED_PRIMES,
                "witnesses_per_channel": maximum_witnesses,
                "expected_plus_one_sentinel": true,
                "output_antisymmetrization": "all 24 S4 permutations with parity; common factor 24 is retained",
                "metric_variance": {
                    "momentum_h": 1,
                    "momentum_output": 1,
                    "h_output": "eta",
                    "gamma_attachments": "momentum 1; H and output eta",
                },
                "frozen_device_diagram_blob_sha256": D21_DEVICE_DIAGRAM_BLOB_SHA256,
                "c4_device_csr_sha256": "c8b470ac061937c1d54f523d52acc495af1e07c36f9bc9676eea7ae79e305440",
                "channels": channel_reports,
                "boundary": "Three-prime witness ranks and exact nonzero selected minors certify the 52-dimensional antisymmetrized equivariant diagram span. This report does not yet bind final 56 global column streams or solve the physical coefficient equations.",
            });
            let bytes = serde_json::to_vec_pretty(&report).unwrap();
            let temporary = format!("{path}.tmp-{}", std::process::id());
            std::fs::write(&temporary, bytes).unwrap();
            std::fs::rename(temporary, path).unwrap();
        }
    }
    type BigQi = Complex<BigRational>;

    fn br_i64(value: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(value))
    }

    fn big_ratio(value: &Ratio<i64>) -> BigRational {
        BigRational::new(BigInt::from(*value.numer()), BigInt::from(*value.denom()))
    }

    fn big_qi(value: &ExactQi) -> BigQi {
        Complex::new(big_ratio(&value.real), big_ratio(&value.imaginary))
    }

    fn big_rational_mod(value: &BigRational, prime: u32) -> u32 {
        let modulus = BigInt::from(prime);
        let reduce = |integer: &BigInt| {
            (((integer % &modulus) + &modulus) % &modulus)
                .to_u32()
                .unwrap()
        };
        let numerator = reduce(value.numer());
        let denominator = reduce(value.denom());
        assert_ne!(denominator, 0, "inadmissible exact coefficient denominator");
        ((u64::from(numerator) * u64::from(pow_mod(denominator, prime - 2, prime)))
            % u64::from(prime)) as u32
    }

    fn pair_from_mask(mask: u32) -> Result<[usize; 2], String> {
        if mask.count_ones() != 2 {
            return Err(format!(
                "D21 target source mask {mask:#010x} is not degree two"
            ));
        }
        let axes = (0..32)
            .filter(|axis| mask & (1_u32 << axis) != 0)
            .collect::<Vec<_>>();
        Ok([axes[0], axes[1]])
    }

    fn one_momentum_axis(exponents: &[u16; 11]) -> Option<usize> {
        (exponents.iter().copied().sum::<u16>() == 1)
            .then(|| exponents.iter().position(|&value| value == 1).unwrap())
    }

    /// Project the corrected teleparallel map onto one of the three
    /// antisymmetric outer-spinor Fierz summands, retaining the complete
    /// Cartesian D G4 target vector at the requested source coordinate.
    fn source_fierz_target_slice(
        teleparallel: &BTreeMap<FullChainRowKey, ExactQi>,
        source_coordinate: u32,
        outer_degree: usize,
    ) -> Result<BTreeMap<usize, BigQi>, String> {
        let (query_pair, momentum, _) = decode_source_coordinate(source_coordinate)?;
        let (_, charge_gamma) = flattened_gamma_mask_tables();
        let table = |mask: usize, left: usize, right: usize| -> i64 {
            i64::from(charge_gamma[mask * 32 * 32 + left * 32 + right])
        };
        let masks = (0_usize..(1_usize << 11))
            .filter(|mask| mask.count_ones() as usize == outer_degree)
            .filter_map(|mask| {
                let value = table(mask, query_pair[0], query_pair[1]);
                (value != 0).then_some((mask, value))
            })
            .collect::<Vec<_>>();
        if masks.is_empty() {
            return Ok(BTreeMap::new());
        }
        for &(mask, _) in &masks {
            let norm = (0..32)
                .flat_map(|left| ((left + 1)..32).map(move |right| (left, right)))
                .map(|(left, right)| {
                    let value = table(mask, left, right);
                    value * value
                })
                .sum::<i64>();
            if norm != 16 {
                return Err(format!(
                    "outer Fierz mask {mask:#05x} has norm {norm}, expected 16"
                ));
            }
        }
        let mut output = BTreeMap::<usize, BigQi>::new();
        for (key, value) in teleparallel {
            if one_momentum_axis(&key.momentum_exponents) != Some(momentum) {
                continue;
            }
            let pair = pair_from_mask(key.exterior_spinor_mask)?;
            let numerator = masks
                .iter()
                .map(|&(mask, query_value)| query_value * table(mask, pair[0], pair[1]))
                .sum::<i64>();
            if numerator == 0 {
                continue;
            }
            let scaled =
                big_qi(value) * BigRational::new(BigInt::from(numerator), BigInt::from(16));
            let entry = output
                .entry(teleparallel_target_to_numeric(key.output_coordinate))
                .or_insert_with(|| Complex::new(BigRational::zero(), BigRational::zero()));
            *entry += scaled;
            if entry.re.is_zero() && entry.im.is_zero() {
                output.remove(&key.output_coordinate);
            }
        }
        Ok(output)
    }

    fn target_sector_slice(
        sector: &str,
        input: &BTreeMap<usize, BigQi>,
    ) -> Result<BTreeMap<usize, BigQi>, String> {
        let to_small = |value: &BigRational| -> Result<Ratio<i64>, String> {
            let numerator = i64::try_from(value.numer().clone())
                .map_err(|_| "teleparallel projector numerator exceeds i64".to_string())?;
            let denominator = i64::try_from(value.denom().clone())
                .map_err(|_| "teleparallel projector denominator exceeds i64".to_string())?;
            Ok(Ratio::new(numerator, denominator))
        };
        let real = input
            .iter()
            .filter(|(_, value)| !value.re.is_zero())
            .map(|(&row, value)| Ok((row, to_small(&value.re)?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let imaginary = input
            .iter()
            .filter(|(_, value)| !value.im.is_zero())
            .map(|(&row, value)| Ok((row, to_small(&value.im)?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let real = project_dg4_target(sector, &real)?;
        let imaginary = project_dg4_target(sector, &imaginary)?;
        let rows = real
            .keys()
            .chain(imaginary.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        Ok(rows
            .into_iter()
            .map(|row| {
                (
                    row,
                    Complex::new(
                        real.get(&row)
                            .map(big_ratio)
                            .unwrap_or_else(BigRational::zero),
                        imaginary
                            .get(&row)
                            .map(big_ratio)
                            .unwrap_or_else(BigRational::zero),
                    ),
                )
            })
            .collect())
    }

    fn solve_real_matrix_big_qi(
        mut matrix: Vec<Vec<BigRational>>,
        mut right: Vec<BigQi>,
    ) -> Result<Vec<BigQi>, String> {
        let n = matrix.len();
        if n == 0 || right.len() != n || matrix.iter().any(|row| row.len() != n) {
            return Err("physical coefficient pivot system is not square".to_string());
        }
        for column in 0..n {
            let pivot = (column..n)
                .find(|&row| !matrix[row][column].is_zero())
                .ok_or_else(|| format!("physical coefficient system is singular at {column}"))?;
            matrix.swap(column, pivot);
            right.swap(column, pivot);
            let scale = matrix[column][column].clone();
            for entry in &mut matrix[column][column..] {
                *entry /= scale.clone();
            }
            right[column] /= scale;
            let pivot_row = matrix[column].clone();
            let pivot_right = right[column].clone();
            for row in 0..n {
                if row == column || matrix[row][column].is_zero() {
                    continue;
                }
                let factor = matrix[row][column].clone();
                for next in column..n {
                    matrix[row][next] -= factor.clone() * pivot_row[next].clone();
                }
                right[row] -= pivot_right.clone() * factor;
            }
        }
        Ok(right)
    }

    fn rational_json(value: &BigRational) -> serde_json::Value {
        serde_json::json!({
            "numerator": value.numer().to_string(),
            "denominator": value.denom().to_string(),
        })
    }

    fn qi_json(value: &BigQi) -> serde_json::Value {
        serde_json::json!({"real": rational_json(&value.re), "imaginary": rational_json(&value.im)})
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "authoritative global augmented D21 witness-first solve"]
    fn global_augmented_57_cuda_witness_first_no_go() {
        use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
        use cuda::{D21HTerm, D21WitnessCuda};

        #[derive(Clone)]
        struct Selected {
            global: usize,
            diagram: u16,
            sector: u8,
            label: String,
        }
        struct Channel {
            degree: u8,
            selected: Vec<Selected>,
            context: D21WitnessCuda,
        }

        let numeric_basis = (0..330)
            .map(|lexicographic| {
                let numeric = lexicographic_four_form_to_numeric(lexicographic).unwrap();
                assert_eq!(
                    numeric_four_form_to_lexicographic(numeric).unwrap(),
                    lexicographic
                );
                numeric
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(numeric_basis.len(), 330);
        let rhs_stream0 = teleparallel_rhs_column(0).unwrap();
        assert_eq!(
            rhs_stream0.stream_sha256,
            "dfd7fc0ace00d202b83a7c3ae15aa2af666fd876bea4b7f5d59c3086aeeee997"
        );
        let rhs_column0 = global57_rhs_batch(0, 0, 1).unwrap();
        assert_eq!(rhs_column0.entries.len(), 343_720);
        assert!(
            rhs_column0
                .entries
                .iter()
                .all(|entry| entry.column == AUGMENTED_TARGET_COLUMN)
        );
        let rhs_bianchi = bianchi_rhs_image(&rhs_column0.entries).unwrap();
        assert!(rhs_bianchi.is_empty());
        let wrong_basis_rhs = rhs_column0
            .entries
            .iter()
            .map(|entry| {
                let mut row = CanonicalRow::from_ordinal(entry.row).unwrap();
                let spinor = row.target_coordinate as usize / 330;
                let numeric = row.target_coordinate as usize % 330;
                row.target_coordinate =
                    (spinor * 330 + numeric_four_form_to_lexicographic(numeric).unwrap()) as u32;
                let mut mutated = entry.clone();
                mutated.row = row.ordinal().unwrap();
                mutated
            })
            .collect::<Vec<_>>();
        let wrong_basis_bianchi_rows = bianchi_rhs_image(&wrong_basis_rhs).unwrap().len();
        assert_eq!(wrong_basis_bianchi_rows, 2_386_880);

        let pivot_bytes = std::fs::read("results/adynkra_11d_d21_gpu_witness_ranks.json").unwrap();
        let pivot_sha256 = format!("{:x}", Sha256::digest(&pivot_bytes));
        let pivot: serde_json::Value = serde_json::from_slice(&pivot_bytes).unwrap();
        let labels = ["00001", "00011", "00101", "01001", "10001"];
        let mut channels = Vec::new();
        let mut global_seen = BTreeSet::new();
        for channel in pivot["channels"].as_array().unwrap() {
            let degree = u8::try_from(channel["outer_fierz_degree"].as_u64().unwrap()).unwrap();
            let mut selected = Vec::new();
            for sector in channel["sectors"].as_array().unwrap() {
                if sector["expected_rank"].as_u64().unwrap() == 0 {
                    continue;
                }
                let label = sector["sector"].as_str().unwrap();
                let sector_slot = labels
                    .iter()
                    .position(|candidate| *candidate == label)
                    .unwrap();
                for (global, diagram) in sector["global_columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(sector["selected_diagram_ordinals"].as_array().unwrap())
                {
                    let global = usize::try_from(global.as_u64().unwrap()).unwrap();
                    assert!(global_seen.insert(global));
                    selected.push(Selected {
                        global,
                        diagram: u16::try_from(diagram.as_u64().unwrap()).unwrap(),
                        sector: sector_slot as u8,
                        label: label.to_string(),
                    });
                }
            }
            selected.sort_by_key(|entry| entry.global);
            let context = D21WitnessCuda::from_candidates(
                selected.iter().map(|entry| entry.diagram).collect(),
            )
            .unwrap();
            channels.push(Channel {
                degree,
                selected,
                context,
            });
        }
        assert_eq!(global_seen, (0_usize..52).collect());

        let h_basis = canonical_gamma_traceless_frame_basis();
        let pairs = (0_u8..32)
            .flat_map(|left| ((left + 1)..32).map(move |right| [left, right]))
            .collect::<Vec<_>>();
        let mut bases: [Vec<Option<Vec<PrimeGaussian>>>; 3] =
            std::array::from_fn(|_| vec![None; 53]);
        let mut pivot_rows: [Vec<Option<u64>>; 3] = std::array::from_fn(|_| vec![None; 53]);
        let mut gpu_rows = BTreeMap::<u64, [Vec<PrimeGaussian>; 3]>::new();
        let mut target_streams = BTreeMap::new();
        let mut scientific_ms = 0_f32;
        let mut witnesses = 0_usize;

        for witness in 0..320 {
            let pair = if witness == 0 {
                [0, 17]
            } else {
                pairs[(witness * 37) % pairs.len()]
            };
            let momentum = if witness == 0 { 0 } else { (witness * 5) % 11 };
            let h = if witness == 0 {
                0
            } else {
                (witness * 17) % 320
            };
            let terms = h_basis[h]
                .iter()
                .map(|(&coordinate, value)| D21HTerm {
                    input_spinor: (coordinate / 11) as u8,
                    h_vector: (coordinate % 11) as u8,
                    coefficient: i16::try_from(*value.real.numer()).unwrap(),
                })
                .collect::<Vec<_>>();
            let mut evaluated_channels = Vec::new();
            for channel in &mut channels {
                let sectors = channel
                    .selected
                    .iter()
                    .map(|entry| entry.sector)
                    .collect::<Vec<_>>();
                let (values, milliseconds) = channel
                    .context
                    .evaluate(pair, momentum as u8, h as u32, &terms, &sectors)
                    .unwrap();
                scientific_ms += milliseconds;
                evaluated_channels.push(values);
            }
            let (_, target) = corrected_full_chain_streams(h).unwrap();
            target_streams.insert(h, target.clone());
            let exterior_mask = (1_u32 << pair[0]) | (1_u32 << pair[1]);
            let mut momentum_exponents = [0_u16; 11];
            momentum_exponents[momentum] = 1;
            let pair_ordinal = pairs
                .iter()
                .position(|candidate| *candidate == pair)
                .unwrap();
            let source = ((pair_ordinal * 11 + momentum) * 320 + h) as u64;

            for target_coordinate in 0..10_560_usize {
                let teleparallel_output_coordinate =
                    numeric_target_to_teleparallel_lexicographic(target_coordinate);
                let rhs = target
                    .get(&FullChainRowKey {
                        output_coordinate: teleparallel_output_coordinate,
                        exterior_spinor_mask: exterior_mask,
                        momentum_exponents,
                    })
                    .cloned()
                    .unwrap_or_else(ExactQi::zero);
                let mut rows: [Vec<PrimeGaussian>; 3] =
                    std::array::from_fn(|_| vec![PrimeGaussian::default(); 53]);
                for (channel_slot, channel) in channels.iter().enumerate() {
                    let candidate_count = channel.selected.len();
                    for (local, selected) in channel.selected.iter().enumerate() {
                        for prime_slot in 0..3 {
                            rows[prime_slot][selected.global].real = evaluated_channels
                                [channel_slot][prime_slot * 10_560 * candidate_count
                                + target_coordinate * candidate_count
                                + local];
                        }
                    }
                }
                for (prime_slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
                    rows[prime_slot][52] = PrimeGaussian {
                        real: big_rational_mod(&big_ratio(&rhs.real), prime),
                        imaginary: big_rational_mod(&big_ratio(&rhs.imaginary), prime),
                    };
                }
                if rows
                    .iter()
                    .all(|row| row.iter().all(|value| value.is_zero()))
                {
                    continue;
                }
                let canonical_row = source * 10_560 + target_coordinate as u64;
                for prime_slot in 0..3 {
                    if let Some(pivot) = insert_rref_row(
                        &mut bases[prime_slot],
                        rows[prime_slot].clone(),
                        PINNED_PRIMES[prime_slot],
                    )
                    .unwrap()
                    {
                        pivot_rows[prime_slot][pivot] = Some(canonical_row);
                        gpu_rows.insert(canonical_row, rows.clone());
                    }
                }
            }
            witnesses += 1;
            let ranks = std::array::from_fn::<_, 3, _>(|slot| {
                bases[slot].iter().filter(|row| row.is_some()).count()
            });
            eprintln!(
                "D21_GLOBAL_AUGMENTED witnesses={witnesses} ranks={ranks:?} kernel_ms={scientific_ms:.6}"
            );
            assert!(ranks.iter().all(|&rank| rank <= 53));
            if ranks == [53, 53, 53] {
                break;
            }
        }
        let ranks = std::array::from_fn::<_, 3, _>(|slot| {
            bases[slot].iter().filter(|row| row.is_some()).count() as u32
        });
        assert_eq!(ranks, [53, 53, 53]);
        assert!(pivot_rows.iter().all(|rows| rows[52].is_some()));
        assert_eq!(pivot_rows[0], pivot_rows[1]);
        assert_eq!(pivot_rows[1], pivot_rows[2]);

        let canonical_pivot_rows = pivot_rows[0]
            .iter()
            .map(|row| row.unwrap())
            .collect::<Vec<_>>();
        let mut exact_replay_entries = 0_usize;
        let mut exact_replay_residuals = 0_usize;
        for &row in &canonical_pivot_rows {
            let source = u32::try_from(row / 10_560).unwrap();
            let target_coordinate = u16::try_from(row % 10_560).unwrap();
            let (_, _, h) = decode_source_coordinate(source).unwrap();
            let exact_target = target_streams.get(&h).unwrap();
            let (pair, momentum, _) = decode_source_coordinate(source).unwrap();
            let mut exponents = [0_u16; 11];
            exponents[momentum] = 1;
            let target_value = exact_target
                .get(&FullChainRowKey {
                    output_coordinate: numeric_target_to_teleparallel_lexicographic(usize::from(
                        target_coordinate,
                    )),
                    exterior_spinor_mask: (1_u32 << pair[0]) | (1_u32 << pair[1]),
                    momentum_exponents: exponents,
                })
                .cloned()
                .unwrap_or_else(ExactQi::zero);
            for channel in &channels {
                for selected in &channel.selected {
                    let replay = replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                        source_coordinate: source,
                        target_coordinate,
                        diagram_ordinal: selected.diagram,
                        target_sector: selected.label.clone(),
                    })
                    .unwrap();
                    for (prime_slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
                        let exact =
                            BigRational::from_integer(BigInt::from(replay.projected_numerator));
                        let residue = big_rational_mod(&exact, prime);
                        let recorded = gpu_rows[&row][prime_slot][selected.global].real;
                        exact_replay_entries += 1;
                        exact_replay_residuals += usize::from(residue != recorded);
                    }
                }
            }
            for (prime_slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
                exact_replay_entries += 2;
                exact_replay_residuals += usize::from(
                    big_rational_mod(&big_ratio(&target_value.real), prime)
                        != gpu_rows[&row][prime_slot][52].real,
                );
                exact_replay_residuals += usize::from(
                    big_rational_mod(&big_ratio(&target_value.imaginary), prime)
                        != gpu_rows[&row][prime_slot][52].imaginary,
                );
            }
        }
        assert_eq!(exact_replay_residuals, 0);
        let report = serde_json::json!({
            "schema_version": "adynkra-11d-four-form-57-global-augmented-witness-v2",
            "passed": true,
            "outcome": "scoped_no_solution",
            "ordered_primes": PINNED_PRIMES,
            "d21_candidate_rank": [52,52,52],
            "d21_augmented_rank": ranks,
            "d02_independent_rank": [4,4,4],
            "global_candidate_rank": [56,56,56],
            "global_augmented_rank": [57,57,57],
            "homogeneous_57_column_nullity": [0,0,0],
            "witnesses_scanned": witnesses,
            "device_scientific_milliseconds": scientific_ms,
            "canonical_pivot_rows": canonical_pivot_rows,
            "rhs_pivot_row": pivot_rows[0][52],
            "exact_cpu_gpu_pivot_entries_replayed": exact_replay_entries,
            "exact_cpu_gpu_pivot_residuals": exact_replay_residuals,
            "target_basis_join": {
                "teleparallel_source_order": "lexicographic four-form combinations",
                "canonical_target_order": "ascending numeric four-form masks",
                "exhaustive_bijection_size": numeric_basis.len(),
                "column0_rhs_stream_sha256": rhs_stream0.stream_sha256,
                "column0_rhs_batch_sha256": rhs_column0.batch_sha256,
                "column0_rhs_entries": rhs_column0.entries.len(),
                "column0_rhs_bianchi_residual_rows": rhs_bianchi.len(),
                "no_remap_mutation_bianchi_residual_rows": wrong_basis_bianchi_rows,
            },
            "d21_gpu_pivot_artifact_sha256": pivot_sha256,
            "d02_inventory_sha256": format!("{:x}", Sha256::digest(std::fs::read("results/adynkra_11d_d02_remaining_seed_inventory.json").unwrap())),
            "dg4_c4_device_csr_sha256": format!("{:x}", Sha256::digest(std::fs::read("results/adynkra_11d_dg4_c4_device_csr.csr.bin").unwrap())),
            "device_diagram_blob_sha256": format!("{:x}", Sha256::digest(std::fs::read("results/adynkra_11d_d21_device_diagrams_v1.bin").unwrap())),
            "source_sha256": {
                "gpu_host": format!("{:x}", Sha256::digest(std::fs::read("src/eleven_dimensional_four_form_56_gpu.rs").unwrap())),
                "physics_rows": format!("{:x}", Sha256::digest(std::fs::read("src/eleven_dimensional_four_form_56_physics_rows.rs").unwrap())),
                "cuda_witness": format!("{:x}", Sha256::digest(std::fs::read("cuda/d21_witness_constructor_cuda.cu").unwrap())),
            },
            "boundary": "Rank 53 on the corrected numeric-mask D21 direct-summand augmented matrix proves that the corrected teleparallel D21 target is outside the selected 52-column D21 span on the scanned canonical rows. The independent rank-four D02 summand cannot cancel a D21 row, so the declared 56-column direct sum has augmented rank 57 and no coefficient ray. This is a no-go for the declared higher-bidegree map inventory, not irreducibility or bidegree exhaustion. Source-constraint and quotient descent remain separate gates.",
        });
        let path = "results/adynkra_11d_four_form_57_global_augmented_witness.json";
        let bytes = serde_json::to_vec_pretty(&report).unwrap();
        let temporary = format!("{path}.tmp-{}", std::process::id());
        std::fs::write(&temporary, bytes).unwrap();
        std::fs::rename(temporary, path).unwrap();
    }

    #[test]
    #[ignore = "authoritative exact 56-column physical coefficient solve"]
    fn physical_56_coefficient_solve_three_primes_and_exact_replay() {
        let pivot_path = "results/adynkra_11d_d21_gpu_witness_ranks.json";
        let pivot_bytes = std::fs::read(pivot_path).unwrap();
        let pivot_sha256 = format!("{:x}", Sha256::digest(&pivot_bytes));
        let pivot: serde_json::Value = serde_json::from_slice(&pivot_bytes).unwrap();
        assert_eq!(pivot["passed"], true);

        let mut h_ordinals = BTreeSet::new();
        let mut all_witness_rows = BTreeSet::new();
        for channel in pivot["channels"].as_array().unwrap() {
            for sector in channel["sectors"].as_array().unwrap() {
                for row in sector["witness_rows"].as_array().unwrap() {
                    let row = row.as_u64().unwrap();
                    all_witness_rows.insert(row);
                    let source = u32::try_from(row / 10_560).unwrap();
                    h_ordinals.insert(decode_source_coordinate(source).unwrap().2);
                }
            }
        }
        let mut target_streams = BTreeMap::new();
        for h in h_ordinals {
            let (_, target) = corrected_full_chain_streams(h).unwrap();
            target_streams.insert(h, target);
        }

        let mut source_fierz_cache = BTreeMap::<(u8, u32), BTreeMap<usize, BigQi>>::new();
        let mut target_sector_cache = BTreeMap::<(u8, String, u32), BTreeMap<usize, BigQi>>::new();
        let mut coefficients = vec![Complex::new(BigRational::zero(), BigRational::zero()); 56];
        let mut block_reports = Vec::new();
        let mut all_replay_residuals = 0_usize;
        let mut first_cross_mismatch = None;
        let mut modular_cross_residual_seen = [false; 3];

        for channel in pivot["channels"].as_array().unwrap() {
            let outer_degree =
                u8::try_from(channel["outer_fierz_degree"].as_u64().unwrap()).unwrap();
            for sector in channel["sectors"].as_array().unwrap() {
                let rank = sector["expected_rank"].as_u64().unwrap() as usize;
                if rank == 0 {
                    continue;
                }
                let label = sector["sector"].as_str().unwrap();
                let diagrams = sector["selected_diagram_ordinals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| u16::try_from(value.as_u64().unwrap()).unwrap())
                    .collect::<Vec<_>>();
                let globals = sector["global_columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
                    .collect::<Vec<_>>();
                let rows = sector["witness_rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(diagrams.len(), rank);
                assert_eq!(globals.len(), rank);
                assert_eq!(rows.len(), rank);

                let matrix = rows
                    .iter()
                    .map(|&row| {
                        let source_coordinate = u32::try_from(row / 10_560).unwrap();
                        let target_coordinate = u16::try_from(row % 10_560).unwrap();
                        diagrams
                            .iter()
                            .map(|&diagram_ordinal| {
                                let replay =
                                    replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                                        source_coordinate,
                                        target_coordinate,
                                        diagram_ordinal,
                                        target_sector: label.to_string(),
                                    })
                                    .unwrap();
                                BigRational::new(
                                    BigInt::from(replay.projected_numerator),
                                    BigInt::from(replay.projector_denominator),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let right = rows
                    .iter()
                    .map(|&row| {
                        let source = u32::try_from(row / 10_560).unwrap();
                        let target = usize::try_from(row % 10_560).unwrap();
                        let (_, _, h) = decode_source_coordinate(source).unwrap();
                        let source_slice = source_fierz_cache
                            .entry((outer_degree, source))
                            .or_insert_with(|| {
                                source_fierz_target_slice(
                                    target_streams.get(&h).unwrap(),
                                    source,
                                    usize::from(outer_degree),
                                )
                                .unwrap()
                            });
                        target_sector_cache
                            .entry((outer_degree, label.to_string(), source))
                            .or_insert_with(|| target_sector_slice(label, source_slice).unwrap())
                            .get(&target)
                            .cloned()
                            .unwrap_or_else(|| {
                                Complex::new(BigRational::zero(), BigRational::zero())
                            })
                    })
                    .collect::<Vec<_>>();
                let solution = solve_real_matrix_big_qi(matrix.clone(), right.clone()).unwrap();
                for (&global, value) in globals.iter().zip(&solution) {
                    assert!(coefficients[global].re.is_zero() && coefficients[global].im.is_zero());
                    coefficients[global] = value.clone();
                }
                let mut residuals = 0_usize;
                for row in 0..rank {
                    let mut value = Complex::new(BigRational::zero(), BigRational::zero());
                    for column in 0..rank {
                        value += solution[column].clone() * matrix[row][column].clone();
                    }
                    residuals += usize::from(value != right[row]);
                }
                let mut cross_replay_rows = 0_usize;
                let mut cross_replay_residuals = 0_usize;
                for &row in &all_witness_rows {
                    let source = u32::try_from(row / 10_560).unwrap();
                    let target = usize::try_from(row % 10_560).unwrap();
                    let (_, _, h) = decode_source_coordinate(source).unwrap();
                    let source_slice = source_fierz_cache
                        .entry((outer_degree, source))
                        .or_insert_with(|| {
                            source_fierz_target_slice(
                                target_streams.get(&h).unwrap(),
                                source,
                                usize::from(outer_degree),
                            )
                            .unwrap()
                        });
                    let expected = target_sector_cache
                        .entry((outer_degree, label.to_string(), source))
                        .or_insert_with(|| target_sector_slice(label, source_slice).unwrap())
                        .get(&target)
                        .cloned()
                        .unwrap_or_else(|| Complex::new(BigRational::zero(), BigRational::zero()));
                    let mut actual = Complex::new(BigRational::zero(), BigRational::zero());
                    for (&diagram_ordinal, coefficient) in diagrams.iter().zip(&solution) {
                        let replay = replay_sector_pivot_v2(D21SectorPivotReplayRequestV2 {
                            source_coordinate: source,
                            target_coordinate: u16::try_from(target).unwrap(),
                            diagram_ordinal,
                            target_sector: label.to_string(),
                        })
                        .unwrap();
                        actual += coefficient.clone()
                            * BigRational::new(
                                BigInt::from(replay.projected_numerator),
                                BigInt::from(replay.projector_denominator),
                            );
                    }
                    cross_replay_rows += 1;
                    if actual != expected {
                        cross_replay_residuals += 1;
                        let residual = actual.clone() - expected.clone();
                        for (slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
                            modular_cross_residual_seen[slot] |=
                                big_rational_mod(&residual.re, prime) != 0
                                    || big_rational_mod(&residual.im, prime) != 0;
                        }
                        if first_cross_mismatch.is_none() {
                            first_cross_mismatch = Some(serde_json::json!({
                                "outer_fierz_degree": outer_degree,
                                "target_sector": label,
                                "canonical_row": row,
                                "source_coordinate": source,
                                "target_coordinate": target,
                                "candidate": qi_json(&actual),
                                "teleparallel": qi_json(&expected),
                                "candidate_minus_teleparallel": qi_json(&residual),
                            }));
                        }
                    }
                }
                all_replay_residuals += cross_replay_residuals;
                all_replay_residuals += residuals;
                block_reports.push(serde_json::json!({
                    "outer_fierz_degree": outer_degree,
                    "target_sector": label,
                    "rank": rank,
                    "global_columns": globals,
                    "selected_diagram_ordinals": diagrams,
                    "witness_rows": rows,
                    "coefficients": solution.iter().map(qi_json).collect::<Vec<_>>(),
                    "exact_pivot_replay_residuals": residuals,
                    "exact_cross_replay_rows": cross_replay_rows,
                    "exact_cross_replay_residuals": cross_replay_residuals,
                }));
            }
        }
        let d21_nonzero_coefficients = coefficients[..52]
            .iter()
            .filter(|value| !value.re.is_zero() || !value.im.is_zero())
            .count();
        assert!(d21_nonzero_coefficients > 0);
        assert!(
            coefficients[52..]
                .iter()
                .all(|value| value.re.is_zero() && value.im.is_zero())
        );

        let d02_bytes =
            std::fs::read("results/adynkra_11d_d02_remaining_seed_inventory.json").unwrap();
        let d02_sha256 = format!("{:x}", Sha256::digest(&d02_bytes));
        let d02_column52 =
            std::fs::read("results/adynkra_11d_d02_00001_source_generator.json").unwrap();
        let target_in_scanned_span = all_replay_residuals == 0;
        let per_prime_augmented_ranks =
            modular_cross_residual_seen.map(|seen| 56 + u32::from(seen));
        let augmented_rank = if target_in_scanned_span { 56 } else { 57 };
        let report = serde_json::json!({
            "schema_version": "adynkra-11d-four-form-56-scoped-coefficient-screen-v1",
            "passed": true,
            "outcome": if target_in_scanned_span { "scoped_exact_solution" } else { "scoped_no_solution" },
            "ordered_primes": PINNED_PRIMES,
            "per_prime_coefficient_ranks": [56,56,56],
            "per_prime_augmented_ranks": per_prime_augmented_ranks,
            "characteristic_zero_rank": 56,
            "characteristic_zero_nullity": 0,
            "characteristic_zero_augmented_rank": augmented_rank,
            "homogeneous_57_column_nullity": 57-augmented_rank,
            "target_in_scanned_span": target_in_scanned_span,
            "d21_columns": 52,
            "d21_nonzero_solution_coefficients": d21_nonzero_coefficients,
            "d02_columns": 4,
            "d02_coefficients_for_d21_target": "exactly zero by direct-sum bidegree",
            "coefficients": coefficients.iter().enumerate().map(|(column,value)| serde_json::json!({"global_column":column,"value":qi_json(value)})).collect::<Vec<_>>(),
            "blocks": block_reports,
            "exact_pivot_replay_residuals": all_replay_residuals,
            "first_cross_replay_mismatch": first_cross_mismatch,
            "source_fierz_projectors": "orthogonal C Gamma_[r] pair-space projectors r=0,3,4 with exact norm 16",
            "target_projectors": "exact four-pass C4 spectral projectors onto 00001,00011,00101,01001,10001",
            "bindings": {
                "d21_gpu_pivot_artifact_sha256": pivot_sha256,
                "d21_frozen_diagram_blob_sha256": D21_DEVICE_DIAGRAM_BLOB_SHA256,
                "dg4_c4_device_csr_sha256": "c8b470ac061937c1d54f523d52acc495af1e07c36f9bc9676eea7ae79e305440",
                "d02_inventory_sha256": d02_sha256,
                "d02_column52_sha256": format!("{:x}", Sha256::digest(&d02_column52)),
            },
            "first_witness": block_reports.first(),
            "boundary": "This is a fail-closed representation-block coefficient screen for the declared direct sum of 52 (2,1) and four (0,2) D G4 maps against the corrected teleparallel (2,1) target. A mismatch is an exact counterexample to the pivot-fit coefficients on the retained cross rows. Full physical publication still requires immutable complete column streams, the all-320 canonical row adapter, D21 Bianchi and PBW integrability, source and target descent, all-row replay, and bidegree exhaustion. It does not prove irreducibility.",
        });
        let path = std::env::var("ADYNKRA_FOUR_FORM_56_PHYSICAL_REPORT").unwrap_or_else(|_| {
            "results/adynkra_11d_four_form_56_physical_coefficient_solve.json".to_string()
        });
        let bytes = serde_json::to_vec_pretty(&report).unwrap();
        let temporary = format!("{path}.tmp-{}", std::process::id());
        std::fs::write(&temporary, bytes).unwrap();
        std::fs::rename(temporary, path).unwrap();
    }
}
