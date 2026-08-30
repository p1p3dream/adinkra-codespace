//! Streamed finite-field rank certificate for the exact invariant-supercurvature shards.
//!
//! The operator has only 321 source columns, but its exact sparse shards contain
//! millions of wide polynomial coordinates.  Each shard is already in the same
//! canonical row order.  A k-way merge therefore feeds rows directly into a
//! 321-wide Gaussian elimination without materializing or sorting the matrix.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_complete_f::{
    GaugeFixedInvariantOutputSector, SUPERFIELD_COLUMN_ENTRY_BYTES,
    SUPERFIELD_COLUMN_SHARD_MAGIC as SHARD_MAGIC,
    SUPERFIELD_COLUMN_SHARD_SCHEMA as INPUT_SHARD_SCHEMA,
    SUPERFIELD_OPERATOR_COLUMN_SCHEMA as INPUT_COLUMN_SCHEMA,
    SUPERFIELD_OPERATOR_SCHEMA as INPUT_OPERATOR_SCHEMA,
    SUPERFIELD_UNIFIED_OUTPUT_SCHEMA as INPUT_UNIFIED_OUTPUT_SCHEMA,
};

const ENTRY_BYTES: u64 = SUPERFIELD_COLUMN_ENTRY_BYTES as u64;
const FOOTER_BYTES: u64 = 40;
const SOURCE_DIMENSION: usize = 321;
pub(crate) const DEFAULT_PRIME: u32 = 1_073_741_783;

#[derive(Clone, Debug, Deserialize)]
struct InputCertificate {
    schema_version: String,
    source_dimension: usize,
    gamma_traceless_h_dimension: usize,
    scale_dimension: usize,
    unified_output_schema: String,
    column_shard_schema: String,
    total_nonzero_terms: u64,
    operator_sha256: String,
    direct_riemann_integrated: bool,
    direct_gravitino_curl_integrated: bool,
    direct_candidate_four_form_integrated: bool,
    raw_w2021_two_d_terms_are_not_gravity: bool,
    physical_target_component_adapter_complete: bool,
    columns: Vec<InputColumn>,
}

#[derive(Clone, Debug, Deserialize)]
struct InputColumn {
    ordinal: usize,
    source_coordinate: String,
    nonzero_terms: usize,
    raw_w_bianchi_residual_terms: usize,
    raw_w_candidate_residual_terms: usize,
    sha256: String,
    shard_path: Option<String>,
    shard_sha256: Option<String>,
    shard_byte_count: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RowKey {
    exterior: u32,
    momentum: [u16; 11],
    sector: u8,
    coordinate: u64,
}

#[derive(Clone, Copy, Debug)]
struct ExactEntry {
    key: RowKey,
    real_numerator: i64,
    real_denominator: i64,
    imaginary_numerator: i64,
    imaginary_denominator: i64,
}

struct ColumnCursor {
    ordinal: usize,
    reader: BufReader<File>,
    total_entries: u64,
    entries_read: u64,
    sector_filter: Option<u8>,
    previous_key: Option<RowKey>,
    current: Option<ExactEntry>,
}

fn read_array<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], String> {
    let mut bytes = [0_u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("truncated shard: {error}"))?;
    Ok(bytes)
}

fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N], String> {
    let end = cursor
        .checked_add(N)
        .ok_or_else(|| "column shard offset overflow".to_string())?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| "truncated column shard".to_string())?;
    *cursor = end;
    Ok(value.try_into().unwrap())
}

fn hash_bytes_with_length(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn resolve_shard_path(column: &InputColumn, certificate_path: &Path) -> Result<PathBuf, String> {
    let stored_path = column
        .shard_path
        .as_deref()
        .ok_or_else(|| format!("column {} has no shard path", column.ordinal))?;
    let direct = PathBuf::from(stored_path);
    if direct.exists() {
        return Ok(direct);
    }
    let relative = certificate_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(stored_path);
    if relative.exists() {
        return Ok(relative);
    }
    Err(format!(
        "column {} shard does not exist at {} or {}",
        column.ordinal,
        direct.display(),
        relative.display()
    ))
}

fn coordinate_is_valid(sector: u8, coordinate: u64) -> bool {
    match sector {
        value if value == GaugeFixedInvariantOutputSector::XTwo as u8 => coordinate < 605,
        value if value == GaugeFixedInvariantOutputSector::XFive as u8 => coordinate < 5_082,
        value if value == GaugeFixedInvariantOutputSector::JMinus as u8 => coordinate < 32,
        value if value == GaugeFixedInvariantOutputSector::W2021Raw as u8 => coordinate < 330,
        value if value == GaugeFixedInvariantOutputSector::LinearizedRiemann as u8 => {
            coordinate < 3_025
        }
        value if value == GaugeFixedInvariantOutputSector::DirectGravitinoCurl as u8 => {
            coordinate < 1_760
        }
        value if value == GaugeFixedInvariantOutputSector::DirectCandidateFourForm as u8 => {
            coordinate < 330
        }
        _ => false,
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn rational_is_canonical(numerator: i64, denominator: i64) -> bool {
    denominator > 0
        && gcd_u64(numerator.unsigned_abs(), denominator as u64) == 1
        && (numerator != 0 || denominator == 1)
}

fn prevalidate_shard(column: &InputColumn, certificate_path: &Path) -> Result<[u64; 12], String> {
    let path = resolve_shard_path(column, certificate_path)?;
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if column.shard_byte_count != Some(bytes.len() as u64) {
        return Err(format!(
            "{} byte count {} does not match certificate {:?}",
            path.display(),
            bytes.len(),
            column.shard_byte_count
        ));
    }
    let actual_file_sha = format!("{:x}", Sha256::digest(&bytes));
    if column.shard_sha256.as_deref() != Some(actual_file_sha.as_str()) {
        return Err(format!("{} file SHA-256 mismatch", path.display()));
    }

    let mut cursor = 0_usize;
    if &take::<16>(&bytes, &mut cursor)? != SHARD_MAGIC {
        return Err(format!("{} has invalid COL3 magic", path.display()));
    }
    let stored_ordinal = u64::from_le_bytes(take(&bytes, &mut cursor)?);
    if stored_ordinal != column.ordinal as u64 {
        return Err(format!("{} stores the wrong ordinal", path.display()));
    }
    let name_length = usize::try_from(u64::from_le_bytes(take(&bytes, &mut cursor)?))
        .map_err(|_| "column shard source name is too long".to_string())?;
    let name_end = cursor
        .checked_add(name_length)
        .ok_or_else(|| "column shard source-name offset overflow".to_string())?;
    let name = bytes
        .get(cursor..name_end)
        .ok_or_else(|| "truncated column shard source name".to_string())?;
    cursor = name_end;
    if name != column.source_coordinate.as_bytes() {
        return Err(format!("{} source-coordinate mismatch", path.display()));
    }
    let payload_end = bytes
        .len()
        .checked_sub(FOOTER_BYTES as usize)
        .ok_or_else(|| "truncated column shard footer".to_string())?;
    let payload_bytes = payload_end
        .checked_sub(cursor)
        .ok_or_else(|| "invalid column shard payload range".to_string())?;
    if payload_bytes % ENTRY_BYTES as usize != 0 {
        return Err(format!("{} contains a partial entry", path.display()));
    }
    let entry_count = payload_bytes / ENTRY_BYTES as usize;
    if entry_count != column.nonzero_terms {
        return Err(format!(
            "{} entry count {entry_count} does not match certificate {}",
            path.display(),
            column.nonzero_terms
        ));
    }

    let mut semantic = Sha256::new();
    semantic.update(INPUT_COLUMN_SCHEMA);
    semantic.update((column.ordinal as u64).to_le_bytes());
    hash_bytes_with_length(&mut semantic, name);
    let mut previous_key = None;
    let mut sector_counts = [0_u64; 12];
    for entry_ordinal in 0..entry_count {
        let sector = take::<1>(&bytes, &mut cursor)?[0];
        semantic.update([sector]);
        let coordinate = u64::from_le_bytes(take(&bytes, &mut cursor)?);
        semantic.update(coordinate.to_le_bytes());
        if !coordinate_is_valid(sector, coordinate) {
            return Err(format!(
                "{} entry {entry_ordinal} has invalid sector/coordinate {sector}/{coordinate}",
                path.display()
            ));
        }
        sector_counts[sector as usize] += 1;
        let exterior = u32::from_le_bytes(take(&bytes, &mut cursor)?);
        semantic.update(exterior.to_le_bytes());
        let mut momentum = [0_u16; 11];
        for exponent in &mut momentum {
            *exponent = u16::from_le_bytes(take(&bytes, &mut cursor)?);
            semantic.update(exponent.to_le_bytes());
        }
        let real_numerator = i64::from_le_bytes(take(&bytes, &mut cursor)?);
        let real_denominator = i64::from_le_bytes(take(&bytes, &mut cursor)?);
        let imaginary_numerator = i64::from_le_bytes(take(&bytes, &mut cursor)?);
        let imaginary_denominator = i64::from_le_bytes(take(&bytes, &mut cursor)?);
        for value in [
            real_numerator,
            real_denominator,
            imaginary_numerator,
            imaginary_denominator,
        ] {
            hash_bytes_with_length(&mut semantic, value.to_string().as_bytes());
        }
        if !rational_is_canonical(real_numerator, real_denominator)
            || !rational_is_canonical(imaginary_numerator, imaginary_denominator)
            || (real_numerator == 0 && imaginary_numerator == 0)
        {
            return Err(format!(
                "{} entry {entry_ordinal} has a noncanonical or zero coefficient",
                path.display()
            ));
        }
        let key = RowKey {
            exterior,
            momentum,
            sector,
            coordinate,
        };
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(format!(
                "{} is not strictly ordered at entry {entry_ordinal}",
                path.display()
            ));
        }
        previous_key = Some(key);
    }
    if cursor != payload_end {
        return Err(format!("{} payload length mismatch", path.display()));
    }
    let stored_count = u64::from_le_bytes(take(&bytes, &mut cursor)?);
    if stored_count != entry_count as u64 {
        return Err(format!("{} footer count mismatch", path.display()));
    }
    semantic.update(stored_count.to_le_bytes());
    let actual_semantic_sha = format!("{:x}", semantic.finalize());
    let stored_semantic_sha = take::<32>(&bytes, &mut cursor)?
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if cursor != bytes.len()
        || stored_semantic_sha != actual_semantic_sha
        || column.sha256 != actual_semantic_sha
    {
        return Err(format!("{} semantic SHA-256 mismatch", path.display()));
    }
    Ok(sector_counts)
}

impl ColumnCursor {
    fn open(
        column: &InputColumn,
        certificate_path: &Path,
        sector_filter: Option<u8>,
    ) -> Result<Self, String> {
        let path = resolve_shard_path(column, certificate_path)?;
        let file = File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let file_bytes = file
            .metadata()
            .map_err(|error| format!("{}: {error}", path.display()))?
            .len();
        let mut reader = BufReader::with_capacity(1 << 20, file);
        if &read_array::<16>(&mut reader)? != SHARD_MAGIC {
            return Err(format!("{} has invalid magic", path.display()));
        }
        let stored_ordinal = u64::from_le_bytes(read_array(&mut reader)?);
        if stored_ordinal != column.ordinal as u64 {
            return Err(format!(
                "{} stores ordinal {stored_ordinal}, expected {}",
                path.display(),
                column.ordinal
            ));
        }
        let name_length = u64::from_le_bytes(read_array(&mut reader)?);
        let name_length_usize = usize::try_from(name_length)
            .map_err(|_| format!("{} source name is too long", path.display()))?;
        let mut name = vec![0_u8; name_length_usize];
        reader
            .read_exact(&mut name)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if name != column.source_coordinate.as_bytes() {
            return Err(format!("{} source-coordinate mismatch", path.display()));
        }
        let header_bytes = 32_u64
            .checked_add(name_length)
            .ok_or_else(|| "shard header byte count overflow".to_string())?;
        let payload_bytes = file_bytes
            .checked_sub(header_bytes + FOOTER_BYTES)
            .ok_or_else(|| format!("{} is shorter than its header/footer", path.display()))?;
        if payload_bytes % ENTRY_BYTES != 0 {
            return Err(format!("{} has a partial entry", path.display()));
        }
        let total_entries = payload_bytes / ENTRY_BYTES;
        if total_entries != column.nonzero_terms as u64 {
            return Err(format!(
                "{} contains {total_entries} entries, certificate says {}",
                path.display(),
                column.nonzero_terms
            ));
        }
        let mut cursor = Self {
            ordinal: column.ordinal,
            reader,
            total_entries,
            entries_read: 0,
            sector_filter,
            previous_key: None,
            current: None,
        };
        cursor.advance()?;
        Ok(cursor)
    }

    fn advance(&mut self) -> Result<(), String> {
        loop {
            if self.entries_read == self.total_entries {
                self.current = None;
                return Ok(());
            }
            let sector = read_array::<1>(&mut self.reader)?[0];
            let coordinate = u64::from_le_bytes(read_array(&mut self.reader)?);
            if !coordinate_is_valid(sector, coordinate) {
                return Err(format!(
                    "column {} contains invalid invariant sector/coordinate {sector}/{coordinate}",
                    self.ordinal
                ));
            }
            let exterior = u32::from_le_bytes(read_array(&mut self.reader)?);
            let mut momentum = [0_u16; 11];
            for exponent in &mut momentum {
                *exponent = u16::from_le_bytes(read_array(&mut self.reader)?);
            }
            let real_numerator = i64::from_le_bytes(read_array(&mut self.reader)?);
            let real_denominator = i64::from_le_bytes(read_array(&mut self.reader)?);
            let imaginary_numerator = i64::from_le_bytes(read_array(&mut self.reader)?);
            let imaginary_denominator = i64::from_le_bytes(read_array(&mut self.reader)?);
            if real_denominator <= 0 || imaginary_denominator <= 0 {
                return Err(format!(
                    "column {} contains a nonpositive rational denominator",
                    self.ordinal
                ));
            }
            let key = RowKey {
                exterior,
                momentum,
                sector,
                coordinate,
            };
            if self.previous_key.is_some_and(|previous| previous >= key) {
                return Err(format!(
                    "column {} is not strictly ordered at entry {}",
                    self.ordinal, self.entries_read
                ));
            }
            self.previous_key = Some(key);
            self.entries_read += 1;
            if self.sector_filter.is_none_or(|filter| filter == sector) {
                self.current = Some(ExactEntry {
                    key,
                    real_numerator,
                    real_denominator,
                    imaginary_numerator,
                    imaginary_denominator,
                });
                return Ok(());
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Residue {
    real: u32,
    imaginary: u32,
}

impl Residue {
    fn is_zero(self) -> bool {
        self.real == 0 && self.imaginary == 0
    }

    fn add(self, other: Self, prime: u32) -> Self {
        Self {
            real: add_mod(self.real, other.real, prime),
            imaginary: add_mod(self.imaginary, other.imaginary, prime),
        }
    }

    fn negate(self, prime: u32) -> Self {
        Self {
            real: negate_mod(self.real, prime),
            imaginary: negate_mod(self.imaginary, prime),
        }
    }

    fn multiply(self, other: Self, prime: u32) -> Self {
        Self {
            real: subtract_mod(
                multiply_mod(self.real, other.real, prime),
                multiply_mod(self.imaginary, other.imaginary, prime),
                prime,
            ),
            imaginary: add_mod(
                multiply_mod(self.real, other.imaginary, prime),
                multiply_mod(self.imaginary, other.real, prime),
                prime,
            ),
        }
    }
}

fn add_mod(left: u32, right: u32, prime: u32) -> u32 {
    ((u64::from(left) + u64::from(right)) % u64::from(prime)) as u32
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

fn power_mod(mut value: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut result = 1_u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = multiply_mod(result, value, prime);
        }
        value = multiply_mod(value, value, prime);
        exponent >>= 1;
    }
    result
}

fn inverse_mod(value: u32, prime: u32) -> u32 {
    debug_assert_ne!(value, 0);
    power_mod(value, prime - 2, prime)
}

fn signed_mod(value: i64, prime: u32) -> u32 {
    let residue = value % i64::from(prime);
    if residue < 0 {
        (residue + i64::from(prime)) as u32
    } else {
        residue as u32
    }
}

fn rational_mod(
    numerator: i64,
    denominator: i64,
    prime: u32,
    inverse_cache: &mut HashMap<u32, u32>,
) -> Result<u32, String> {
    let denominator = signed_mod(denominator, prime);
    if denominator == 0 {
        return Err("exact coefficient denominator vanishes modulo the prime".to_string());
    }
    let inverse = *inverse_cache
        .entry(denominator)
        .or_insert_with(|| inverse_mod(denominator, prime));
    Ok(multiply_mod(signed_mod(numerator, prime), inverse, prime))
}

fn coefficient_mod(
    entry: ExactEntry,
    prime: u32,
    inverse_cache: &mut HashMap<u32, u32>,
) -> Result<Residue, String> {
    Ok(Residue {
        real: rational_mod(
            entry.real_numerator,
            entry.real_denominator,
            prime,
            inverse_cache,
        )?,
        imaginary: rational_mod(
            entry.imaginary_numerator,
            entry.imaginary_denominator,
            prime,
            inverse_cache,
        )?,
    })
}

fn gaussian_inverse(value: Residue, prime: u32) -> Residue {
    let norm = add_mod(
        multiply_mod(value.real, value.real, prime),
        multiply_mod(value.imaginary, value.imaginary, prime),
        prime,
    );
    let inverse_norm = inverse_mod(norm, prime);
    Residue {
        real: multiply_mod(value.real, inverse_norm, prime),
        imaginary: multiply_mod(negate_mod(value.imaginary, prime), inverse_norm, prime),
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct KernelCertificate {
    pub schema_version: &'static str,
    pub input_schema_version: String,
    pub input_certificate_sha256: String,
    pub input_operator_sha256: String,
    pub fully_validated_shards: usize,
    pub prime: u32,
    pub columns: usize,
    pub exact_input_terms: u64,
    pub merged_rows_examined: u64,
    pub shard_entries_examined: u64,
    pub rank_over_gaussian_extension: usize,
    pub nullity_upper_bound: usize,
    pub full_column_rank: bool,
    pub stopped_after_full_rank: bool,
    pub elapsed_seconds: f64,
    pub proof_boundary: &'static str,
}

fn operator_sha256(input: &InputCertificate) -> String {
    let mut hasher = Sha256::new();
    hash_bytes_with_length(&mut hasher, INPUT_OPERATOR_SCHEMA.as_bytes());
    hasher.update((input.source_dimension as u64).to_le_bytes());
    hasher.update(input.total_nonzero_terms.to_le_bytes());
    for column in &input.columns {
        hasher.update((column.ordinal as u64).to_le_bytes());
        hasher.update((column.nonzero_terms as u64).to_le_bytes());
        hasher.update((column.raw_w_bianchi_residual_terms as u64).to_le_bytes());
        hasher.update((column.raw_w_candidate_residual_terms as u64).to_le_bytes());
        hash_bytes_with_length(&mut hasher, column.source_coordinate.as_bytes());
        hash_bytes_with_length(&mut hasher, column.sha256.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn validate_input_metadata(input: &InputCertificate) -> Result<(), String> {
    if input.schema_version != INPUT_OPERATOR_SCHEMA
        || input.unified_output_schema != INPUT_UNIFIED_OUTPUT_SCHEMA
        || input.column_shard_schema != INPUT_SHARD_SCHEMA
        || !input.direct_riemann_integrated
        || !input.direct_gravitino_curl_integrated
        || !input.direct_candidate_four_form_integrated
        || !input.raw_w2021_two_d_terms_are_not_gravity
        || input.physical_target_component_adapter_complete
    {
        return Err(
            "input is not the exact fail-closed v4 direct-Riemann/direct-gravitino operator"
                .to_string(),
        );
    }
    if input.source_dimension != SOURCE_DIMENSION
        || input.gamma_traceless_h_dimension != 320
        || input.scale_dimension != 1
        || input.columns.len() != SOURCE_DIMENSION
    {
        return Err(format!(
            "expected {SOURCE_DIMENSION} source columns, found {}/{}",
            input.source_dimension,
            input.columns.len()
        ));
    }
    for (ordinal, column) in input.columns.iter().enumerate() {
        if column.ordinal != ordinal {
            return Err(format!(
                "certificate column position {ordinal} stores ordinal {}",
                column.ordinal
            ));
        }
        if !lowercase_sha256(&column.sha256)
            || column
                .shard_sha256
                .as_deref()
                .is_none_or(|digest| !lowercase_sha256(digest))
            || column.shard_byte_count.is_none()
        {
            return Err(format!(
                "column {ordinal} lacks canonical bound digests/size"
            ));
        }
    }
    if !lowercase_sha256(&input.operator_sha256) || operator_sha256(input) != input.operator_sha256
    {
        return Err("operator SHA-256 does not bind the declared v4 columns".to_string());
    }
    Ok(())
}

fn emit_progress(started: Instant, rows: u64, entries: u64, rank: usize, total_entries: u64) {
    let unix_milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let elapsed = started.elapsed().as_secs_f64();
    let rate = if elapsed == 0.0 {
        0.0
    } else {
        entries as f64 / elapsed
    };
    eprintln!(
        "{{\"event\":\"kernel_progress\",\"unix_milliseconds\":{unix_milliseconds},\"elapsed_seconds\":{elapsed:.3},\"rows_examined\":{rows},\"entries_examined\":{entries},\"total_entries\":{total_entries},\"rank\":{rank},\"columns\":{SOURCE_DIMENSION},\"entries_per_second\":{rate:.1}}}"
    );
}

fn emit_sector_progress(
    started: Instant,
    sector: u8,
    rows: u64,
    entries: u64,
    rank: usize,
    total_entries: u64,
) {
    let unix_milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let elapsed = started.elapsed().as_secs_f64();
    let rate = if elapsed == 0.0 {
        0.0
    } else {
        entries as f64 / elapsed
    };
    eprintln!(
        "{{\"event\":\"sector_kernel_progress\",\"unix_milliseconds\":{unix_milliseconds},\"elapsed_seconds\":{elapsed:.3},\"sector\":{sector},\"rows_examined\":{rows},\"entries_examined\":{entries},\"total_entries\":{total_entries},\"rank\":{rank},\"columns\":{SOURCE_DIMENSION},\"entries_per_second\":{rate:.1}}}"
    );
}

pub(crate) fn derive_streamed_kernel(
    certificate_path: &Path,
    prime: u32,
) -> Result<KernelCertificate, String> {
    if prime % 4 != 3 {
        return Err(format!(
            "prime {prime} must be 3 modulo 4 so i defines F_(p^2)"
        ));
    }
    let certificate_bytes = std::fs::read(certificate_path)
        .map_err(|error| format!("{}: {error}", certificate_path.display()))?;
    let input_certificate_sha256 = format!("{:x}", Sha256::digest(&certificate_bytes));
    let input: InputCertificate = serde_json::from_slice(&certificate_bytes)
        .map_err(|error| format!("{}: {error}", certificate_path.display()))?;
    validate_input_metadata(&input)?;

    let started = Instant::now();
    for (validated, column) in input.columns.iter().enumerate() {
        let _ = prevalidate_shard(column, certificate_path)?;
        if (validated + 1) % 32 == 0 || validated + 1 == SOURCE_DIMENSION {
            let unix_milliseconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or_default();
            eprintln!(
                "{{\"event\":\"kernel_shard_validation\",\"unix_milliseconds\":{unix_milliseconds},\"elapsed_seconds\":{:.3},\"validated\":{},\"total\":{SOURCE_DIMENSION}}}",
                started.elapsed().as_secs_f64(),
                validated + 1
            );
        }
    }
    let mut cursors = input
        .columns
        .iter()
        .map(|column| ColumnCursor::open(column, certificate_path, None))
        .collect::<Result<Vec<_>, _>>()?;
    let decoded_total = cursors
        .iter()
        .map(|cursor| cursor.total_entries)
        .sum::<u64>();
    if decoded_total != input.total_nonzero_terms {
        return Err(format!(
            "shard total {decoded_total} != certificate total {}",
            input.total_nonzero_terms
        ));
    }
    let mut heap = BinaryHeap::<Reverse<(RowKey, usize)>>::new();
    for (column, cursor) in cursors.iter().enumerate() {
        if let Some(entry) = cursor.current {
            heap.push(Reverse((entry.key, column)));
        }
    }

    let mut basis = vec![None::<Vec<Residue>>; SOURCE_DIMENSION];
    let mut row = vec![Residue::default(); SOURCE_DIMENSION];
    let mut inverse_cache = HashMap::new();
    let mut rank = 0_usize;
    let mut rows_examined = 0_u64;
    let mut entries_examined = 0_u64;
    let mut next_progress = Instant::now() + Duration::from_secs(3);
    while let Some(Reverse((key, first_column))) = heap.pop() {
        row.fill(Residue::default());
        let mut pending = Some(first_column);
        loop {
            let column = pending.take().unwrap();
            let entry = cursors[column]
                .current
                .ok_or_else(|| format!("column {column} heap/cursor mismatch"))?;
            if entry.key != key {
                return Err(format!("column {column} row-key mismatch"));
            }
            row[column] = coefficient_mod(entry, prime, &mut inverse_cache)?;
            entries_examined += 1;
            cursors[column].advance()?;
            if let Some(next) = cursors[column].current {
                heap.push(Reverse((next.key, column)));
            }
            pending = match heap.peek() {
                Some(Reverse((next_key, _))) if *next_key == key => {
                    let Reverse((_, next_column)) = heap.pop().unwrap();
                    Some(next_column)
                }
                _ => None,
            };
            if pending.is_none() {
                break;
            }
        }
        rows_examined += 1;

        loop {
            let Some(pivot) = row.iter().position(|value| !value.is_zero()) else {
                break;
            };
            if let Some(existing) = &basis[pivot] {
                let factor = row[pivot];
                for column in pivot..SOURCE_DIMENSION {
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
        if Instant::now() >= next_progress || rank == SOURCE_DIMENSION {
            emit_progress(
                started,
                rows_examined,
                entries_examined,
                rank,
                decoded_total,
            );
            next_progress = Instant::now() + Duration::from_secs(3);
        }
        if rank == SOURCE_DIMENSION {
            break;
        }
    }

    Ok(KernelCertificate {
        schema_version: "adynkra-11d-gauge-fixed-invariant-supercurvature-kernel-v2",
        input_schema_version: input.schema_version,
        input_certificate_sha256,
        input_operator_sha256: input.operator_sha256,
        fully_validated_shards: SOURCE_DIMENSION,
        prime,
        columns: SOURCE_DIMENSION,
        exact_input_terms: decoded_total,
        merged_rows_examined: rows_examined,
        shard_entries_examined: entries_examined,
        rank_over_gaussian_extension: rank,
        nullity_upper_bound: SOURCE_DIMENSION - rank,
        full_column_rank: rank == SOURCE_DIMENSION,
        stopped_after_full_rank: rank == SOURCE_DIMENSION && entries_examined < decoded_total,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        proof_boundary: "full rank modulo a good prime proves exact column independence over Q(i); modular rank deficiency alone does not prove an exact kernel, and this certificate covers only the declared gauge-fixed invariant-supercurvature operator",
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SectorKernelCertificate {
    pub schema_version: &'static str,
    pub input_schema_version: String,
    pub input_certificate_sha256: String,
    pub input_operator_sha256: String,
    pub fully_validated_shards: usize,
    pub sector_tag: u8,
    pub sector_name: &'static str,
    pub prime: u32,
    pub columns: usize,
    pub exact_input_terms: u64,
    pub exact_sector_terms: u64,
    pub merged_rows_examined: u64,
    pub sector_entries_examined: u64,
    pub rank_over_gaussian_extension: usize,
    pub nullity_upper_bound: usize,
    pub full_column_rank: bool,
    pub stopped_after_full_rank: bool,
    pub elapsed_seconds: f64,
    pub proof_boundary: &'static str,
}

fn sector_name(sector: u8) -> Result<&'static str, String> {
    match sector {
        value if value == GaugeFixedInvariantOutputSector::XTwo as u8 => Ok("X2"),
        value if value == GaugeFixedInvariantOutputSector::XFive as u8 => Ok("X5"),
        value if value == GaugeFixedInvariantOutputSector::JMinus as u8 => Ok("JMinus"),
        value if value == GaugeFixedInvariantOutputSector::W2021Raw as u8 => Ok("W2021Raw"),
        value if value == GaugeFixedInvariantOutputSector::LinearizedRiemann as u8 => {
            Ok("LinearizedRiemann")
        }
        value if value == GaugeFixedInvariantOutputSector::DirectGravitinoCurl as u8 => {
            Ok("DirectGravitinoCurl")
        }
        value if value == GaugeFixedInvariantOutputSector::DirectCandidateFourForm as u8 => {
            Ok("DirectCandidateFourForm")
        }
        _ => Err(format!(
            "unsupported invariant-supercurvature sector {sector}"
        )),
    }
}

pub(crate) fn derive_streamed_sector_kernel(
    certificate_path: &Path,
    prime: u32,
    sector: u8,
) -> Result<SectorKernelCertificate, String> {
    if prime % 4 != 3 {
        return Err(format!(
            "prime {prime} must be 3 modulo 4 so i defines F_(p^2)"
        ));
    }
    let sector_name = sector_name(sector)?;
    let certificate_bytes = std::fs::read(certificate_path)
        .map_err(|error| format!("{}: {error}", certificate_path.display()))?;
    let input_certificate_sha256 = format!("{:x}", Sha256::digest(&certificate_bytes));
    let input: InputCertificate = serde_json::from_slice(&certificate_bytes)
        .map_err(|error| format!("{}: {error}", certificate_path.display()))?;
    validate_input_metadata(&input)?;

    let started = Instant::now();
    let mut exact_sector_terms = 0_u64;
    for (validated, column) in input.columns.iter().enumerate() {
        let counts = prevalidate_shard(column, certificate_path)?;
        exact_sector_terms = exact_sector_terms
            .checked_add(counts[sector as usize])
            .ok_or_else(|| "sector term-count overflow".to_string())?;
        if (validated + 1) % 32 == 0 || validated + 1 == SOURCE_DIMENSION {
            let unix_milliseconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or_default();
            eprintln!(
                "{{\"event\":\"sector_kernel_shard_validation\",\"unix_milliseconds\":{unix_milliseconds},\"elapsed_seconds\":{:.3},\"sector\":{sector},\"validated\":{},\"total\":{SOURCE_DIMENSION}}}",
                started.elapsed().as_secs_f64(),
                validated + 1
            );
        }
    }
    if exact_sector_terms == 0 {
        return Err(format!("sector {sector} has no exact entries"));
    }

    let mut cursors = input
        .columns
        .iter()
        .map(|column| ColumnCursor::open(column, certificate_path, Some(sector)))
        .collect::<Result<Vec<_>, _>>()?;
    let decoded_total = cursors
        .iter()
        .map(|cursor| cursor.total_entries)
        .sum::<u64>();
    if decoded_total != input.total_nonzero_terms {
        return Err(format!(
            "shard total {decoded_total} != certificate total {}",
            input.total_nonzero_terms
        ));
    }
    let mut heap = BinaryHeap::<Reverse<(RowKey, usize)>>::new();
    for (column, cursor) in cursors.iter().enumerate() {
        if let Some(entry) = cursor.current {
            heap.push(Reverse((entry.key, column)));
        }
    }

    let mut basis = vec![None::<Vec<Residue>>; SOURCE_DIMENSION];
    let mut row = vec![Residue::default(); SOURCE_DIMENSION];
    let mut inverse_cache = HashMap::new();
    let mut rank = 0_usize;
    let mut rows_examined = 0_u64;
    let mut entries_examined = 0_u64;
    let mut next_progress = Instant::now() + Duration::from_secs(3);
    while let Some(Reverse((key, first_column))) = heap.pop() {
        row.fill(Residue::default());
        let mut pending = Some(first_column);
        loop {
            let column = pending.take().unwrap();
            let entry = cursors[column]
                .current
                .ok_or_else(|| format!("column {column} heap/cursor mismatch"))?;
            if entry.key != key || entry.key.sector != sector {
                return Err(format!("column {column} sector row-key mismatch"));
            }
            row[column] = coefficient_mod(entry, prime, &mut inverse_cache)?;
            entries_examined += 1;
            cursors[column].advance()?;
            if let Some(next) = cursors[column].current {
                heap.push(Reverse((next.key, column)));
            }
            pending = match heap.peek() {
                Some(Reverse((next_key, _))) if *next_key == key => {
                    let Reverse((_, next_column)) = heap.pop().unwrap();
                    Some(next_column)
                }
                _ => None,
            };
            if pending.is_none() {
                break;
            }
        }
        rows_examined += 1;

        loop {
            let Some(pivot) = row.iter().position(|value| !value.is_zero()) else {
                break;
            };
            if let Some(existing) = &basis[pivot] {
                let factor = row[pivot];
                for column in pivot..SOURCE_DIMENSION {
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
        if Instant::now() >= next_progress || rank == SOURCE_DIMENSION {
            emit_sector_progress(
                started,
                sector,
                rows_examined,
                entries_examined,
                rank,
                exact_sector_terms,
            );
            next_progress = Instant::now() + Duration::from_secs(3);
        }
        if rank == SOURCE_DIMENSION {
            break;
        }
    }
    if rank < SOURCE_DIMENSION && entries_examined != exact_sector_terms {
        return Err(format!(
            "sector {sector} heap exhausted after {entries_examined} of {exact_sector_terms} entries"
        ));
    }

    Ok(SectorKernelCertificate {
        schema_version: "adynkra-11d-gauge-fixed-invariant-supercurvature-sector-kernel-v1",
        input_schema_version: input.schema_version,
        input_certificate_sha256,
        input_operator_sha256: input.operator_sha256,
        fully_validated_shards: SOURCE_DIMENSION,
        sector_tag: sector,
        sector_name,
        prime,
        columns: SOURCE_DIMENSION,
        exact_input_terms: decoded_total,
        exact_sector_terms,
        merged_rows_examined: rows_examined,
        sector_entries_examined: entries_examined,
        rank_over_gaussian_extension: rank,
        nullity_upper_bound: SOURCE_DIMENSION - rank,
        full_column_rank: rank == SOURCE_DIMENSION,
        stopped_after_full_rank: rank == SOURCE_DIMENSION && entries_examined < exact_sector_terms,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        proof_boundary: "sector-only modular full rank proves exact independence of the declared source columns after projection to this one diagnostic sector over Q(i); it does not establish physical target completeness or the physical K quotient",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn append_entry(bytes: &mut Vec<u8>, semantic: &mut Sha256, entry: ExactEntry) {
        bytes.push(entry.key.sector);
        semantic.update([entry.key.sector]);
        bytes.extend_from_slice(&entry.key.coordinate.to_le_bytes());
        semantic.update(entry.key.coordinate.to_le_bytes());
        bytes.extend_from_slice(&entry.key.exterior.to_le_bytes());
        semantic.update(entry.key.exterior.to_le_bytes());
        for exponent in entry.key.momentum {
            bytes.extend_from_slice(&exponent.to_le_bytes());
            semantic.update(exponent.to_le_bytes());
        }
        for value in [
            entry.real_numerator,
            entry.real_denominator,
            entry.imaginary_numerator,
            entry.imaginary_denominator,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
            hash_bytes_with_length(semantic, value.to_string().as_bytes());
        }
    }

    fn write_test_shard(entries: &[ExactEntry]) -> (PathBuf, InputColumn) {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-complete-f-kernel-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("column_000.bin");
        let source = "test-source";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SHARD_MAGIC);
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source.as_bytes());
        let mut semantic = Sha256::new();
        semantic.update(INPUT_COLUMN_SCHEMA);
        semantic.update(0_u64.to_le_bytes());
        hash_bytes_with_length(&mut semantic, source.as_bytes());
        for entry in entries {
            append_entry(&mut bytes, &mut semantic, *entry);
        }
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        semantic.update((entries.len() as u64).to_le_bytes());
        let semantic_sha256 = format!("{:x}", semantic.finalize());
        bytes.extend_from_slice(&hex_to_bytes(&semantic_sha256));
        std::fs::write(&path, &bytes).unwrap();
        let column = InputColumn {
            ordinal: 0,
            source_coordinate: source.to_string(),
            nonzero_terms: entries.len(),
            raw_w_bianchi_residual_terms: 0,
            raw_w_candidate_residual_terms: 0,
            sha256: semantic_sha256,
            shard_path: Some(path.display().to_string()),
            shard_sha256: Some(format!("{:x}", Sha256::digest(&bytes))),
            shard_byte_count: Some(bytes.len() as u64),
        };
        (path, column)
    }

    fn hex_to_bytes(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn entry(exterior: u32, coordinate: u64) -> ExactEntry {
        sector_entry(
            exterior,
            GaugeFixedInvariantOutputSector::LinearizedRiemann as u8,
            coordinate,
        )
    }

    fn sector_entry(exterior: u32, sector: u8, coordinate: u64) -> ExactEntry {
        ExactEntry {
            key: RowKey {
                exterior,
                momentum: [0; 11],
                sector,
                coordinate,
            },
            real_numerator: 1,
            real_denominator: 1,
            imaginary_numerator: 0,
            imaginary_denominator: 1,
        }
    }

    fn valid_input_metadata() -> InputCertificate {
        let columns = (0..SOURCE_DIMENSION)
            .map(|ordinal| {
                let digest = format!("{:x}", Sha256::digest(format!("column-{ordinal}")));
                InputColumn {
                    ordinal,
                    source_coordinate: format!("source-{ordinal}"),
                    nonzero_terms: ordinal + 1,
                    raw_w_bianchi_residual_terms: ordinal * 2,
                    raw_w_candidate_residual_terms: ordinal * 3,
                    sha256: digest.clone(),
                    shard_path: Some(format!("column_{ordinal:03}.bin")),
                    shard_sha256: Some(digest),
                    shard_byte_count: Some(1),
                }
            })
            .collect::<Vec<_>>();
        let mut input = InputCertificate {
            schema_version: INPUT_OPERATOR_SCHEMA.to_string(),
            source_dimension: SOURCE_DIMENSION,
            gamma_traceless_h_dimension: 320,
            scale_dimension: 1,
            unified_output_schema: INPUT_UNIFIED_OUTPUT_SCHEMA.to_string(),
            column_shard_schema: INPUT_SHARD_SCHEMA.to_string(),
            total_nonzero_terms: columns
                .iter()
                .map(|column| column.nonzero_terms as u64)
                .sum(),
            operator_sha256: String::new(),
            direct_riemann_integrated: true,
            direct_gravitino_curl_integrated: true,
            direct_candidate_four_form_integrated: true,
            raw_w2021_two_d_terms_are_not_gravity: true,
            physical_target_component_adapter_complete: false,
            columns,
        };
        input.operator_sha256 = operator_sha256(&input);
        input
    }

    #[test]
    fn gaussian_inverse_is_exact() {
        for value in [
            Residue {
                real: 1,
                imaginary: 0,
            },
            Residue {
                real: 17,
                imaginary: 31,
            },
        ] {
            assert_eq!(
                value.multiply(gaussian_inverse(value, DEFAULT_PRIME), DEFAULT_PRIME),
                Residue {
                    real: 1,
                    imaginary: 0
                }
            );
        }
    }

    #[test]
    fn col3_direct_curvature_shard_is_fully_validated_and_d_masks_remain_distinct() {
        let (path, column) = write_test_shard(&[
            sector_entry(0, GaugeFixedInvariantOutputSector::W2021Raw as u8, 0),
            entry(0, 0),
            sector_entry(
                0,
                GaugeFixedInvariantOutputSector::DirectGravitinoCurl as u8,
                0,
            ),
            sector_entry(
                0,
                GaugeFixedInvariantOutputSector::DirectCandidateFourForm as u8,
                0,
            ),
            sector_entry(1, GaugeFixedInvariantOutputSector::W2021Raw as u8, 0),
            entry(1, 0),
            sector_entry(
                1,
                GaugeFixedInvariantOutputSector::DirectGravitinoCurl as u8,
                0,
            ),
            sector_entry(
                1,
                GaugeFixedInvariantOutputSector::DirectCandidateFourForm as u8,
                0,
            ),
        ]);
        let counts = prevalidate_shard(&column, Path::new("unused-certificate.json")).unwrap();
        assert_eq!(
            counts[GaugeFixedInvariantOutputSector::W2021Raw as usize],
            2
        );
        assert_eq!(
            counts[GaugeFixedInvariantOutputSector::LinearizedRiemann as usize],
            2
        );
        assert_eq!(
            counts[GaugeFixedInvariantOutputSector::DirectGravitinoCurl as usize],
            2
        );
        assert_eq!(
            counts[GaugeFixedInvariantOutputSector::DirectCandidateFourForm as usize],
            2
        );
        let mut cursor = ColumnCursor::open(
            &column,
            Path::new("unused-certificate.json"),
            Some(GaugeFixedInvariantOutputSector::LinearizedRiemann as u8),
        )
        .unwrap();
        let mut exterior_masks = Vec::new();
        while let Some(current) = cursor.current {
            exterior_masks.push(current.key.exterior);
            cursor.advance().unwrap();
        }
        assert_eq!(exterior_masks, vec![0, 1]);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn shard_prevalidation_rejects_legacy_magic_bounds_order_and_coefficients() {
        let cases = [
            vec![entry(0, 3_025)],
            vec![sector_entry(
                0,
                GaugeFixedInvariantOutputSector::DirectGravitinoCurl as u8,
                1_760,
            )],
            vec![sector_entry(
                0,
                GaugeFixedInvariantOutputSector::DirectCandidateFourForm as u8,
                330,
            )],
            vec![entry(0, 0), entry(0, 0)],
            vec![ExactEntry {
                real_numerator: 2,
                real_denominator: 2,
                ..entry(0, 0)
            }],
            vec![ExactEntry {
                real_numerator: 0,
                ..entry(0, 0)
            }],
        ];
        for entries in cases {
            let (path, column) = write_test_shard(&entries);
            assert!(prevalidate_shard(&column, Path::new("unused-certificate.json")).is_err());
            std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
        }

        let (path, mut column) = write_test_shard(&[entry(0, 0)]);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[..16].copy_from_slice(b"AD11FINVCOL2\0\0\0\0");
        std::fs::write(&path, &bytes).unwrap();
        column.shard_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        assert!(prevalidate_shard(&column, Path::new("unused-certificate.json")).is_err());
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn shard_prevalidation_rejects_footer_and_file_digest_mutations() {
        let (path, mut column) = write_test_shard(&[entry(0, 0)]);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(prevalidate_shard(&column, Path::new("unused-certificate.json")).is_err());
        column.shard_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        assert!(prevalidate_shard(&column, Path::new("unused-certificate.json")).is_err());
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn certificate_metadata_rejects_legacy_schema_ordinals_and_digest_mutation() {
        assert!(sector_name(7).is_err());
        let input = valid_input_metadata();
        validate_input_metadata(&input).unwrap();

        let mut legacy = input.clone();
        legacy.schema_version =
            "adynkra-11d-gauge-fixed-invariant-supercurvature-operator-v2".to_string();
        assert!(validate_input_metadata(&legacy).is_err());

        let mut duplicate = input.clone();
        duplicate.columns[1].ordinal = 0;
        duplicate.operator_sha256 = operator_sha256(&duplicate);
        assert!(validate_input_metadata(&duplicate).is_err());

        let mut missing = input.clone();
        missing.columns.pop();
        missing.operator_sha256 = operator_sha256(&missing);
        assert!(validate_input_metadata(&missing).is_err());

        let mut mutated = input;
        mutated.columns[0].nonzero_terms += 1;
        assert!(validate_input_metadata(&mutated).is_err());
    }
}
