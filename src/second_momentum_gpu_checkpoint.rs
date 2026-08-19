//! Durable word-boundary checkpoints for streamed second-momentum columns.
//!
//! A checkpoint is a complete commit unit: the next PBW-word ordinal, source
//! hash chain, counters, and every modular row accumulator are serialized and
//! replaced atomically. The active GPU path intentionally does not depend on
//! this module yet.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const CHECKPOINT_SCHEMA_VERSION: &str = "adynkra-11d-second-momentum-word-checkpoint-v1";
pub(crate) const CHECKPOINT_ROW_COUNT: usize = 101_376;
const IDENTITY_DIGEST_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-checkpoint-identity-v1\0";
const CHECKPOINT_DIGEST_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-checkpoint-payload-v1\0";
const RAW_CHAIN_INITIAL_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-raw-chain-initial-v1\0";
const RAW_CHAIN_WORD_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-raw-chain-word-v1\0";
static TEMPORARY_FILE_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CheckpointGaussianResidue {
    pub(crate) real: u32,
    pub(crate) imaginary: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CheckpointIdentity {
    pub(crate) backend_version: String,
    pub(crate) tranche: String,
    pub(crate) local_column_ordinal: u64,
    pub(crate) global_column_ordinal: u64,
    pub(crate) source_label: String,
    pub(crate) source_copy: u64,
    pub(crate) source_fixture_sha256: String,
    pub(crate) source_map_sha256: String,
    pub(crate) reciprocal_map_sha256: String,
    pub(crate) pbw_plan_sha256: String,
    pub(crate) pbw_word_count: u64,
    pub(crate) static_semantic_sha256: String,
    /// Sorted, duplicate-free finite-field moduli. Accumulator order is bound
    /// to this order.
    pub(crate) primes: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PrimeRowAccumulator {
    pub(crate) prime: u32,
    pub(crate) rows: Vec<CheckpointGaussianResidue>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CheckpointCounters {
    pub(crate) batches_flushed: u64,
    pub(crate) expanded_contributions: u64,
    pub(crate) reduced_key_visits: u64,
    pub(crate) nonzero_reduced_term_visits: u64,
}

impl CheckpointCounters {
    fn is_monotone_from(self, previous: Self) -> bool {
        self.batches_flushed >= previous.batches_flushed
            && self.expanded_contributions >= previous.expanded_contributions
            && self.reduced_key_visits >= previous.reduced_key_visits
            && self.nonzero_reduced_term_visits >= previous.nonzero_reduced_term_visits
    }

    fn is_zero(self) -> bool {
        self == Self::default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WordBoundaryCheckpoint {
    pub(crate) identity: CheckpointIdentity,
    pub(crate) identity_sha256: String,
    /// All words below this ordinal are included exactly once.
    pub(crate) next_word_ordinal: u64,
    pub(crate) raw_term_count: u64,
    pub(crate) raw_term_hash_chain_sha256: String,
    pub(crate) accumulators: Vec<PrimeRowAccumulator>,
    pub(crate) counters: CheckpointCounters,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointEnvelope {
    schema_version: String,
    checkpoint_semantic_sha256: String,
    checkpoint: WordBoundaryCheckpoint,
}

impl WordBoundaryCheckpoint {
    pub(crate) fn new(identity: CheckpointIdentity) -> io::Result<Self> {
        validate_identity(&identity)?;
        let identity_sha256 = identity_sha256(&identity)?;
        let raw_term_hash_chain_sha256 = initial_raw_term_hash_chain(&identity)?;
        let accumulators = identity
            .primes
            .iter()
            .copied()
            .map(|prime| PrimeRowAccumulator {
                prime,
                rows: vec![CheckpointGaussianResidue::default(); CHECKPOINT_ROW_COUNT],
            })
            .collect();
        let checkpoint = Self {
            identity,
            identity_sha256,
            next_word_ordinal: 0,
            raw_term_count: 0,
            raw_term_hash_chain_sha256,
            accumulators,
            counters: CheckpointCounters::default(),
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Commit one fully processed word to the in-memory snapshot. The caller
    /// supplies the complete post-word accumulators, making the subsequent
    /// atomic file replacement the only durable commit point.
    pub(crate) fn record_completed_word(
        &mut self,
        word_ordinal: u64,
        word_raw_term_count: u64,
        word_raw_terms_sha256: &str,
        accumulators: Vec<PrimeRowAccumulator>,
        counters: CheckpointCounters,
    ) -> io::Result<()> {
        if word_ordinal != self.next_word_ordinal {
            return Err(invalid_input(format!(
                "word ordinal {word_ordinal} does not match next ordinal {}",
                self.next_word_ordinal
            )));
        }
        if word_ordinal >= self.identity.pbw_word_count {
            return Err(invalid_input("word ordinal exceeds the bound PBW plan"));
        }
        validate_accumulators(&self.identity.primes, &accumulators)?;
        if !counters.is_monotone_from(self.counters) {
            return Err(invalid_input("checkpoint counters moved backward"));
        }
        let raw_term_count = self
            .raw_term_count
            .checked_add(word_raw_term_count)
            .ok_or_else(|| invalid_input("raw term count overflow"))?;
        let raw_term_hash_chain_sha256 = extend_raw_term_hash_chain(
            &self.raw_term_hash_chain_sha256,
            word_ordinal,
            word_raw_term_count,
            word_raw_terms_sha256,
        )?;
        self.next_word_ordinal += 1;
        self.raw_term_count = raw_term_count;
        self.raw_term_hash_chain_sha256 = raw_term_hash_chain_sha256;
        self.accumulators = accumulators;
        self.counters = counters;
        self.validate()
    }

    pub(crate) fn validate(&self) -> io::Result<()> {
        validate_identity(&self.identity)?;
        let observed_identity_sha256 = identity_sha256(&self.identity)?;
        if self.identity_sha256 != observed_identity_sha256 {
            return Err(invalid_data("checkpoint identity digest mismatch"));
        }
        if self.next_word_ordinal > self.identity.pbw_word_count {
            return Err(invalid_data("next word ordinal exceeds PBW plan length"));
        }
        validate_digest(
            "raw term hash chain",
            &self.raw_term_hash_chain_sha256,
            io::ErrorKind::InvalidData,
        )?;
        validate_accumulators(&self.identity.primes, &self.accumulators)?;
        if self.next_word_ordinal == 0 {
            if self.raw_term_count != 0
                || !self.counters.is_zero()
                || self.accumulators.iter().any(|accumulator| {
                    accumulator
                        .rows
                        .iter()
                        .any(|residue| *residue != CheckpointGaussianResidue::default())
                })
            {
                return Err(invalid_data("initial checkpoint contains completed work"));
            }
            if self.raw_term_hash_chain_sha256 != initial_raw_term_hash_chain(&self.identity)? {
                return Err(invalid_data("initial raw term hash chain mismatch"));
            }
        }
        Ok(())
    }
}

pub(crate) fn write_checkpoint_atomic(
    path: &Path,
    checkpoint: &WordBoundaryCheckpoint,
) -> io::Result<()> {
    write_checkpoint_atomic_inner(path, checkpoint, None)
}

pub(crate) fn load_checkpoint(
    path: &Path,
    expected_identity: &CheckpointIdentity,
) -> io::Result<WordBoundaryCheckpoint> {
    validate_identity(expected_identity)?;
    let envelope = read_envelope(path)?;
    if &envelope.checkpoint.identity != expected_identity {
        return Err(invalid_data(
            "checkpoint identity does not match requested column",
        ));
    }
    Ok(envelope.checkpoint)
}

pub(crate) fn identity_sha256(identity: &CheckpointIdentity) -> io::Result<String> {
    validate_identity(identity)?;
    let encoded = serde_json::to_vec(identity).map_err(invalid_json)?;
    let mut hash = Sha256::new();
    hash.update(IDENTITY_DIGEST_DOMAIN);
    hash.update(encoded);
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn initial_raw_term_hash_chain(identity: &CheckpointIdentity) -> io::Result<String> {
    let identity_digest = decode_digest(
        "identity",
        &identity_sha256(identity)?,
        io::ErrorKind::InvalidInput,
    )?;
    let mut hash = Sha256::new();
    hash.update(RAW_CHAIN_INITIAL_DOMAIN);
    hash.update(identity_digest);
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn extend_raw_term_hash_chain(
    previous_chain_sha256: &str,
    word_ordinal: u64,
    word_raw_term_count: u64,
    word_raw_terms_sha256: &str,
) -> io::Result<String> {
    let previous = decode_digest(
        "previous raw term hash chain",
        previous_chain_sha256,
        io::ErrorKind::InvalidInput,
    )?;
    let word = decode_digest(
        "word raw terms",
        word_raw_terms_sha256,
        io::ErrorKind::InvalidInput,
    )?;
    let mut hash = Sha256::new();
    hash.update(RAW_CHAIN_WORD_DOMAIN);
    hash.update(previous);
    hash.update(word_ordinal.to_le_bytes());
    hash.update(word_raw_term_count.to_le_bytes());
    hash.update(word);
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_identity(identity: &CheckpointIdentity) -> io::Result<()> {
    if identity.backend_version.is_empty()
        || identity.tranche.is_empty()
        || identity.source_label.is_empty()
        || identity.source_copy == 0
        || identity.pbw_word_count == 0
        || identity.primes.is_empty()
    {
        return Err(invalid_input(
            "checkpoint identity has an empty required field",
        ));
    }
    for (name, digest) in [
        ("source fixture", &identity.source_fixture_sha256),
        ("source map", &identity.source_map_sha256),
        ("reciprocal map", &identity.reciprocal_map_sha256),
        ("PBW plan", &identity.pbw_plan_sha256),
        ("static semantic", &identity.static_semantic_sha256),
    ] {
        validate_digest(name, digest, io::ErrorKind::InvalidInput)?;
    }
    if !identity.primes.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(invalid_input(
            "checkpoint primes must be sorted and duplicate-free",
        ));
    }
    for &prime in &identity.primes {
        if prime % 4 != 3 || !is_prime_u32(prime) {
            return Err(invalid_input(format!(
                "checkpoint modulus {prime} is not a prime congruent to 3 modulo 4"
            )));
        }
    }
    Ok(())
}

fn validate_accumulators(primes: &[u32], accumulators: &[PrimeRowAccumulator]) -> io::Result<()> {
    if accumulators.len() != primes.len() {
        return Err(invalid_data("checkpoint prime accumulator count mismatch"));
    }
    for (&prime, accumulator) in primes.iter().zip(accumulators) {
        if accumulator.prime != prime {
            return Err(invalid_data("checkpoint prime accumulator order mismatch"));
        }
        if accumulator.rows.len() != CHECKPOINT_ROW_COUNT {
            return Err(invalid_data(format!(
                "checkpoint for prime {prime} has {} rows, expected {CHECKPOINT_ROW_COUNT}",
                accumulator.rows.len()
            )));
        }
        if accumulator
            .rows
            .iter()
            .any(|residue| residue.real >= prime || residue.imaginary >= prime)
        {
            return Err(invalid_data(format!(
                "checkpoint contains a noncanonical residue modulo {prime}"
            )));
        }
    }
    Ok(())
}

fn is_prime_u32(value: u32) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3_u32;
    while u64::from(divisor) * u64::from(divisor) <= u64::from(value) {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

fn checkpoint_semantic_sha256(checkpoint: &WordBoundaryCheckpoint) -> io::Result<String> {
    checkpoint.validate()?;
    let encoded = serde_json::to_vec(checkpoint).map_err(invalid_json)?;
    let mut hash = Sha256::new();
    hash.update(CHECKPOINT_DIGEST_DOMAIN);
    hash.update(CHECKPOINT_SCHEMA_VERSION.as_bytes());
    hash.update([0]);
    hash.update(encoded);
    Ok(format!("{:x}", hash.finalize()))
}

fn read_envelope(path: &Path) -> io::Result<CheckpointEnvelope> {
    let bytes = fs::read(path)?;
    let envelope: CheckpointEnvelope = serde_json::from_slice(&bytes).map_err(invalid_json)?;
    if envelope.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(invalid_data("unsupported checkpoint schema version"));
    }
    envelope.checkpoint.validate()?;
    let observed = checkpoint_semantic_sha256(&envelope.checkpoint)?;
    if envelope.checkpoint_semantic_sha256 != observed {
        return Err(invalid_data("checkpoint semantic digest mismatch"));
    }
    Ok(envelope)
}

fn validate_transition(
    previous: &WordBoundaryCheckpoint,
    next: &WordBoundaryCheckpoint,
) -> io::Result<()> {
    if previous.identity != next.identity {
        return Err(invalid_data(
            "refusing to replace a checkpoint for another column",
        ));
    }
    if previous == next {
        return Ok(());
    }
    if next.next_word_ordinal != previous.next_word_ordinal + 1 {
        return Err(invalid_data(
            "checkpoint replacement must commit exactly one next word",
        ));
    }
    if next.raw_term_count < previous.raw_term_count
        || !next.counters.is_monotone_from(previous.counters)
    {
        return Err(invalid_data(
            "checkpoint replacement moves counters backward",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AtomicWriteFault {
    BeforeRename,
    AfterRename,
}

fn write_checkpoint_atomic_inner(
    path: &Path,
    checkpoint: &WordBoundaryCheckpoint,
    fault: Option<AtomicWriteFault>,
) -> io::Result<()> {
    checkpoint.validate()?;
    let semantic_sha256 = checkpoint_semantic_sha256(checkpoint)?;
    if path.exists() {
        let previous = read_envelope(path)?;
        validate_transition(&previous.checkpoint, checkpoint)?;
        if previous.checkpoint == *checkpoint {
            return Ok(());
        }
    }
    let envelope = CheckpointEnvelope {
        schema_version: CHECKPOINT_SCHEMA_VERSION.to_string(),
        checkpoint_semantic_sha256: semantic_sha256,
        checkpoint: checkpoint.clone(),
    };
    let bytes = serde_json::to_vec(&envelope).map_err(invalid_json)?;
    let parent = checkpoint_parent(path);
    fs::create_dir_all(parent)?;
    let temporary = temporary_path(path)?;
    let result = (|| {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        if fault == Some(AtomicWriteFault::BeforeRename) {
            return Err(io::Error::other(
                "injected checkpoint failure before rename",
            ));
        }
        fs::rename(&temporary, path)?;
        if fault == Some(AtomicWriteFault::AfterRename) {
            return Err(io::Error::other("injected checkpoint failure after rename"));
        }
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn checkpoint_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn temporary_path(path: &Path) -> io::Result<PathBuf> {
    let file_name = path
        .file_name()
        .ok_or_else(|| invalid_input("checkpoint path has no file name"))?
        .to_string_lossy();
    let nonce = TEMPORARY_FILE_NONCE.fetch_add(1, Ordering::Relaxed);
    Ok(checkpoint_parent(path).join(format!(".{file_name}.tmp.{}.{}", std::process::id(), nonce)))
}

fn decode_digest(name: &str, value: &str, kind: io::ErrorKind) -> io::Result<[u8; 32]> {
    validate_digest(name, value, kind)?;
    let mut bytes = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(bytes)
}

fn validate_digest(name: &str, value: &str, kind: io::ErrorKind) -> io::Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(io::Error::new(
            kind,
            format!("{name} digest must be 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(())
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("digest was validated before decoding"),
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_json(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_NONCE: AtomicU64 = AtomicU64::new(0);

    fn digest(label: &str) -> String {
        format!("{:x}", Sha256::digest(label.as_bytes()))
    }

    fn identity() -> CheckpointIdentity {
        CheckpointIdentity {
            backend_version: "test-gpu-backend-v1".to_string(),
            tranche: "20001".to_string(),
            local_column_ordinal: 2,
            global_column_ordinal: 6,
            source_label: "00010".to_string(),
            source_copy: 1,
            source_fixture_sha256: digest("fixture"),
            source_map_sha256: digest("map"),
            reciprocal_map_sha256: digest("reciprocal"),
            pbw_plan_sha256: digest("pbw-plan"),
            pbw_word_count: 4,
            static_semantic_sha256: digest("static"),
            primes: vec![1_073_741_783],
        }
    }

    fn completed_one_word() -> WordBoundaryCheckpoint {
        let mut checkpoint = WordBoundaryCheckpoint::new(identity()).unwrap();
        let mut accumulators = checkpoint.accumulators.clone();
        accumulators[0].rows[17] = CheckpointGaussianResidue {
            real: 41,
            imaginary: 73,
        };
        checkpoint
            .record_completed_word(
                0,
                123,
                &digest("word-zero-raw-terms"),
                accumulators,
                CheckpointCounters {
                    batches_flushed: 2,
                    expanded_contributions: 9_001,
                    reduced_key_visits: 700,
                    nonzero_reduced_term_visits: 650,
                },
            )
            .unwrap();
        checkpoint
    }

    fn test_directory(name: &str) -> PathBuf {
        let nonce = TEST_DIRECTORY_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "adynkra-word-checkpoint-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn checkpoint_round_trip_binds_identity_and_full_rows() {
        let directory = test_directory("round-trip");
        let path = directory.join("column.checkpoint.json");
        let checkpoint = completed_one_word();
        write_checkpoint_atomic(&path, &checkpoint).unwrap();
        let loaded = load_checkpoint(&path, &identity()).unwrap();
        assert_eq!(loaded, checkpoint);
        assert_eq!(loaded.accumulators[0].rows.len(), CHECKPOINT_ROW_COUNT);
        assert_eq!(loaded.accumulators[0].rows[17].imaginary, 73);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn before_rename_failure_preserves_previous_checkpoint_bytes() {
        let directory = test_directory("before-rename");
        let path = directory.join("column.checkpoint.json");
        let initial = WordBoundaryCheckpoint::new(identity()).unwrap();
        write_checkpoint_atomic(&path, &initial).unwrap();
        let before = fs::read(&path).unwrap();
        let next = completed_one_word();
        let error =
            write_checkpoint_atomic_inner(&path, &next, Some(AtomicWriteFault::BeforeRename))
                .unwrap_err();
        assert!(error.to_string().contains("before rename"));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(load_checkpoint(&path, &identity()).unwrap(), initial);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn after_rename_failure_leaves_a_complete_resumable_checkpoint() {
        let directory = test_directory("after-rename");
        let path = directory.join("column.checkpoint.json");
        write_checkpoint_atomic(&path, &WordBoundaryCheckpoint::new(identity()).unwrap()).unwrap();
        let next = completed_one_word();
        let error =
            write_checkpoint_atomic_inner(&path, &next, Some(AtomicWriteFault::AfterRename))
                .unwrap_err();
        assert!(error.to_string().contains("after rename"));
        assert_eq!(load_checkpoint(&path, &identity()).unwrap(), next);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn wrong_column_identity_is_rejected_without_replacement() {
        let directory = test_directory("identity");
        let path = directory.join("column.checkpoint.json");
        let checkpoint = completed_one_word();
        write_checkpoint_atomic(&path, &checkpoint).unwrap();
        let before = fs::read(&path).unwrap();
        let mut wrong_identity = identity();
        wrong_identity.global_column_ordinal += 1;
        assert_eq!(
            load_checkpoint(&path, &wrong_identity).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let wrong_checkpoint = WordBoundaryCheckpoint::new(wrong_identity).unwrap();
        assert_eq!(
            write_checkpoint_atomic(&path, &wrong_checkpoint)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn semantic_corruption_is_rejected() {
        let directory = test_directory("corruption");
        let path = directory.join("column.checkpoint.json");
        write_checkpoint_atomic(&path, &completed_one_word()).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["checkpoint"]["accumulators"][0]["rows"][17]["real"] = 42.into();
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = load_checkpoint(&path, &identity()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("semantic digest"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn dimensions_and_canonical_residues_are_validated() {
        let mut wrong_rows = completed_one_word();
        wrong_rows.accumulators[0].rows.pop();
        assert!(wrong_rows.validate().is_err());

        let mut wrong_residue = completed_one_word();
        wrong_residue.accumulators[0].rows[0].real = wrong_residue.accumulators[0].prime;
        assert!(wrong_residue.validate().is_err());
    }

    #[test]
    fn checkpoint_replacement_is_idempotent_and_single_word_monotone() {
        let directory = test_directory("monotone");
        let path = directory.join("column.checkpoint.json");
        let initial = WordBoundaryCheckpoint::new(identity()).unwrap();
        write_checkpoint_atomic(&path, &initial).unwrap();
        let bytes = fs::read(&path).unwrap();
        write_checkpoint_atomic(&path, &initial).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);

        let mut skipped = completed_one_word();
        skipped.next_word_ordinal = 2;
        assert!(write_checkpoint_atomic(&path, &skipped).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(directory).unwrap();
    }
}
