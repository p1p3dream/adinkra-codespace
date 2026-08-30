//! Resumable word-boundary hashing for the v3 second-momentum GPU stream.
//!
//! Only complete PBW words enter the durable chain. A word may be hashed
//! incrementally, but its in-progress SHA-256 state is deliberately neither
//! serializable nor resumable. This keeps every restart boundary aligned with
//! an exactly-once word commit.
//!
//! The v3 chain is the single authority for full-stream source and packed
//! artifact digests. Legacy checkpoint raw chains are migration inputs only
//! and must not be published as an independent digest authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io;

pub(crate) const WORD_HASH_SCHEMA_V3: &str = "adynkra-11d-second-momentum-word-hash-v3";
const INITIAL_CHAIN_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-word-chain-initial-v3\0";
const RAW_WORD_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-raw-word-v3\0";
const PACKED_WORD_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-packed-word-v3\0";
const CHAIN_LINK_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-word-chain-link-v3\0";
const STATE_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-word-chain-state-v3\0";
const SOURCE_ARTIFACT_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-source-artifact-v3\0";
const PACKED_ARTIFACT_DOMAIN: &[u8] = b"adynkra-11d-second-momentum-packed-artifact-v3\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CanonicalWordTermV3 {
    pub(crate) momentum_pair: [u8; 2],
    pub(crate) free_spinor: u8,
    pub(crate) exterior_mask: u32,
    pub(crate) coefficient: i128,
    /// Canonical packed recoupling key produced by the v3 packing schema.
    pub(crate) packed_key: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct InProgressWordHashV3 {
    identity_digest: [u8; 32],
    word_ordinal: u64,
    raw_terms: u64,
    packed_terms: u64,
    raw_hash: Sha256,
    packed_hash: Sha256,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompletedWordHashV3 {
    pub(crate) word_ordinal: u64,
    pub(crate) raw_terms: u64,
    pub(crate) packed_terms: u64,
    pub(crate) raw_word_sha256: String,
    pub(crate) packed_word_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResumableWordHashChainV3 {
    schema_version: String,
    identity_sha256: String,
    next_word_ordinal: u64,
    completed_words: u64,
    raw_terms: u64,
    packed_terms: u64,
    chain_sha256: String,
    state_semantic_sha256: String,
}

impl InProgressWordHashV3 {
    fn new(identity_digest: [u8; 32], word_ordinal: u64) -> Self {
        let mut raw_hash = Sha256::new();
        raw_hash.update(RAW_WORD_DOMAIN);
        raw_hash.update(WORD_HASH_SCHEMA_V3.as_bytes());
        raw_hash.update([0]);
        raw_hash.update(identity_digest);
        raw_hash.update(word_ordinal.to_le_bytes());

        let mut packed_hash = Sha256::new();
        packed_hash.update(PACKED_WORD_DOMAIN);
        packed_hash.update(WORD_HASH_SCHEMA_V3.as_bytes());
        packed_hash.update([0]);
        packed_hash.update(identity_digest);
        packed_hash.update(word_ordinal.to_le_bytes());
        Self {
            identity_digest,
            word_ordinal,
            raw_terms: 0,
            packed_terms: 0,
            raw_hash,
            packed_hash,
        }
    }

    pub(crate) fn push(&mut self, term: CanonicalWordTermV3) -> io::Result<()> {
        self.raw_terms = self
            .raw_terms
            .checked_add(1)
            .ok_or_else(|| invalid_input("raw word term count overflow"))?;
        self.packed_terms = self
            .packed_terms
            .checked_add(1)
            .ok_or_else(|| invalid_input("packed word term count overflow"))?;

        self.raw_hash.update(term.momentum_pair);
        self.raw_hash.update([term.free_spinor]);
        self.raw_hash.update(term.exterior_mask.to_le_bytes());
        self.raw_hash.update(term.coefficient.to_le_bytes());

        let coefficient_bits = term.coefficient as u128;
        self.packed_hash.update(term.packed_key.to_le_bytes());
        self.packed_hash
            .update((coefficient_bits as u64).to_le_bytes());
        self.packed_hash
            .update(((term.coefficient >> 64) as i64).to_le_bytes());
        Ok(())
    }

    pub(crate) fn word_ordinal(&self) -> u64 {
        self.word_ordinal
    }

    fn complete(mut self) -> CompletedWordHashV3 {
        self.raw_hash.update(self.raw_terms.to_le_bytes());
        self.packed_hash.update(self.packed_terms.to_le_bytes());
        CompletedWordHashV3 {
            word_ordinal: self.word_ordinal,
            raw_terms: self.raw_terms,
            packed_terms: self.packed_terms,
            raw_word_sha256: format!("{:x}", self.raw_hash.finalize()),
            packed_word_sha256: format!("{:x}", self.packed_hash.finalize()),
        }
    }
}

impl ResumableWordHashChainV3 {
    pub(crate) fn new(identity_sha256: &str) -> io::Result<Self> {
        let identity_digest = decode_digest("word-chain identity", identity_sha256)?;
        let mut hash = Sha256::new();
        hash.update(INITIAL_CHAIN_DOMAIN);
        hash.update(WORD_HASH_SCHEMA_V3.as_bytes());
        hash.update([0]);
        hash.update(identity_digest);
        let mut state = Self {
            schema_version: WORD_HASH_SCHEMA_V3.to_string(),
            identity_sha256: identity_sha256.to_string(),
            next_word_ordinal: 0,
            completed_words: 0,
            raw_terms: 0,
            packed_terms: 0,
            chain_sha256: format!("{:x}", hash.finalize()),
            state_semantic_sha256: String::new(),
        };
        state.refresh_state_digest()?;
        state.validate(identity_sha256)?;
        Ok(state)
    }

    pub(crate) fn start_word(&self, word_ordinal: u64) -> io::Result<InProgressWordHashV3> {
        self.validate(&self.identity_sha256)?;
        if word_ordinal != self.next_word_ordinal {
            return Err(invalid_input(format!(
                "word ordinal {word_ordinal} does not match next ordinal {}",
                self.next_word_ordinal
            )));
        }
        Ok(InProgressWordHashV3::new(
            decode_digest("word-chain identity", &self.identity_sha256)?,
            word_ordinal,
        ))
    }

    pub(crate) fn finish_word(
        &mut self,
        word: InProgressWordHashV3,
    ) -> io::Result<CompletedWordHashV3> {
        self.validate(&self.identity_sha256)?;
        if word.identity_digest != decode_digest("word-chain identity", &self.identity_sha256)? {
            return Err(invalid_input("word hasher belongs to another identity"));
        }
        if word.word_ordinal != self.next_word_ordinal {
            return Err(invalid_input(format!(
                "completed word ordinal {} does not match next ordinal {}",
                word.word_ordinal, self.next_word_ordinal
            )));
        }
        let completed = word.complete();
        let raw_digest = decode_digest("raw word", &completed.raw_word_sha256)?;
        let packed_digest = decode_digest("packed word", &completed.packed_word_sha256)?;
        let previous = decode_digest("word chain", &self.chain_sha256)?;
        let mut hash = Sha256::new();
        hash.update(CHAIN_LINK_DOMAIN);
        hash.update(WORD_HASH_SCHEMA_V3.as_bytes());
        hash.update([0]);
        hash.update(decode_digest("word-chain identity", &self.identity_sha256)?);
        hash.update(previous);
        hash.update(completed.word_ordinal.to_le_bytes());
        hash.update(completed.raw_terms.to_le_bytes());
        hash.update(completed.packed_terms.to_le_bytes());
        hash.update(raw_digest);
        hash.update(packed_digest);

        let next_word_ordinal = self
            .next_word_ordinal
            .checked_add(1)
            .ok_or_else(|| invalid_input("next word ordinal overflow"))?;
        let completed_words = self
            .completed_words
            .checked_add(1)
            .ok_or_else(|| invalid_input("completed word count overflow"))?;
        let raw_terms = self
            .raw_terms
            .checked_add(completed.raw_terms)
            .ok_or_else(|| invalid_input("raw chain term count overflow"))?;
        let packed_terms = self
            .packed_terms
            .checked_add(completed.packed_terms)
            .ok_or_else(|| invalid_input("packed chain term count overflow"))?;
        self.next_word_ordinal = next_word_ordinal;
        self.completed_words = completed_words;
        self.raw_terms = raw_terms;
        self.packed_terms = packed_terms;
        self.chain_sha256 = format!("{:x}", hash.finalize());
        self.refresh_state_digest()?;
        self.validate(&self.identity_sha256)?;
        Ok(completed)
    }

    pub(crate) fn append_word(
        &mut self,
        word_ordinal: u64,
        terms: impl IntoIterator<Item = CanonicalWordTermV3>,
    ) -> io::Result<CompletedWordHashV3> {
        let mut word = self.start_word(word_ordinal)?;
        for term in terms {
            word.push(term)?;
        }
        self.finish_word(word)
    }

    pub(crate) fn to_json(&self) -> io::Result<Vec<u8>> {
        self.validate(&self.identity_sha256)?;
        serde_json::to_vec(self).map_err(invalid_json)
    }

    pub(crate) fn restore_json(bytes: &[u8], expected_identity_sha256: &str) -> io::Result<Self> {
        let state: Self = serde_json::from_slice(bytes).map_err(invalid_json)?;
        state.validate(expected_identity_sha256)?;
        Ok(state)
    }

    pub(crate) fn source_artifact_sha256(&self) -> io::Result<String> {
        self.artifact_sha256(SOURCE_ARTIFACT_DOMAIN, self.raw_terms)
    }

    pub(crate) fn packed_artifact_sha256(&self) -> io::Result<String> {
        self.artifact_sha256(PACKED_ARTIFACT_DOMAIN, self.packed_terms)
    }

    pub(crate) fn next_word_ordinal(&self) -> u64 {
        self.next_word_ordinal
    }

    pub(crate) fn completed_words(&self) -> u64 {
        self.completed_words
    }

    pub(crate) fn raw_terms(&self) -> u64 {
        self.raw_terms
    }

    pub(crate) fn packed_terms(&self) -> u64 {
        self.packed_terms
    }

    pub(crate) fn chain_sha256(&self) -> &str {
        &self.chain_sha256
    }

    fn artifact_sha256(&self, domain: &[u8], terms: u64) -> io::Result<String> {
        self.validate(&self.identity_sha256)?;
        let mut hash = Sha256::new();
        hash.update(domain);
        hash.update(WORD_HASH_SCHEMA_V3.as_bytes());
        hash.update([0]);
        hash.update(decode_digest("word-chain identity", &self.identity_sha256)?);
        hash.update(self.next_word_ordinal.to_le_bytes());
        hash.update(self.completed_words.to_le_bytes());
        hash.update(terms.to_le_bytes());
        hash.update(decode_digest("word chain", &self.chain_sha256)?);
        Ok(format!("{:x}", hash.finalize()))
    }

    fn validate(&self, expected_identity_sha256: &str) -> io::Result<()> {
        if self.schema_version != WORD_HASH_SCHEMA_V3 {
            return Err(invalid_data("unsupported word-chain schema version"));
        }
        decode_digest("expected word-chain identity", expected_identity_sha256)?;
        if self.identity_sha256 != expected_identity_sha256 {
            return Err(invalid_data("word-chain identity mismatch"));
        }
        decode_digest("word chain", &self.chain_sha256)?;
        decode_digest("word-chain state", &self.state_semantic_sha256)?;
        if self.completed_words != self.next_word_ordinal {
            return Err(invalid_data(
                "completed word count does not match next word ordinal",
            ));
        }
        if self.raw_terms != self.packed_terms {
            return Err(invalid_data("raw and packed chain term counts differ"));
        }
        if self.state_semantic_sha256 != self.compute_state_digest()? {
            return Err(invalid_data("word-chain state semantic digest mismatch"));
        }
        Ok(())
    }

    fn refresh_state_digest(&mut self) -> io::Result<()> {
        self.state_semantic_sha256 = self.compute_state_digest()?;
        Ok(())
    }

    fn compute_state_digest(&self) -> io::Result<String> {
        let mut hash = Sha256::new();
        hash.update(STATE_DOMAIN);
        hash.update(WORD_HASH_SCHEMA_V3.as_bytes());
        hash.update([0]);
        hash.update(decode_digest("word-chain identity", &self.identity_sha256)?);
        hash.update(self.next_word_ordinal.to_le_bytes());
        hash.update(self.completed_words.to_le_bytes());
        hash.update(self.raw_terms.to_le_bytes());
        hash.update(self.packed_terms.to_le_bytes());
        hash.update(decode_digest("word chain", &self.chain_sha256)?);
        Ok(format!("{:x}", hash.finalize()))
    }
}

fn decode_digest(name: &str, value: &str) -> io::Result<[u8; 32]> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(invalid_data(format!(
            "{name} digest must be 64 lowercase hexadecimal characters"
        )));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(output)
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

    fn identity(label: &str) -> String {
        format!("{:x}", Sha256::digest(label.as_bytes()))
    }

    fn term(seed: i128) -> CanonicalWordTermV3 {
        CanonicalWordTermV3 {
            momentum_pair: [seed as u8 % 11, seed as u8 % 13],
            free_spinor: seed as u8 % 32,
            exterior_mask: (seed as u32).wrapping_mul(0x9e37_79b9),
            coefficient: seed * seed - 3 * seed - 17,
            packed_key: (seed as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
        }
    }

    fn words() -> Vec<Vec<CanonicalWordTermV3>> {
        vec![
            vec![term(1), term(2), term(3)],
            Vec::new(),
            vec![term(4)],
            vec![term(5), term(6)],
        ]
    }

    fn append_suffix(
        state: &mut ResumableWordHashChainV3,
        words: &[Vec<CanonicalWordTermV3>],
        start: usize,
    ) {
        for (ordinal, word) in words.iter().enumerate().skip(start) {
            state
                .append_word(ordinal as u64, word.iter().copied())
                .unwrap();
        }
    }

    #[test]
    fn uninterrupted_and_every_resumed_boundary_are_identical() {
        let identity = identity("column-20001-53-prime-0");
        let words = words();
        let mut uninterrupted = ResumableWordHashChainV3::new(&identity).unwrap();
        append_suffix(&mut uninterrupted, &words, 0);
        assert_eq!(uninterrupted.completed_words(), words.len() as u64);
        assert_eq!(uninterrupted.raw_terms(), uninterrupted.packed_terms());

        for boundary in 0..=words.len() {
            let mut prefix = ResumableWordHashChainV3::new(&identity).unwrap();
            for (ordinal, word) in words.iter().enumerate().take(boundary) {
                prefix
                    .append_word(ordinal as u64, word.iter().copied())
                    .unwrap();
            }
            let serialized = prefix.to_json().unwrap();
            let mut resumed =
                ResumableWordHashChainV3::restore_json(&serialized, &identity).unwrap();
            append_suffix(&mut resumed, &words, boundary);
            assert_eq!(resumed, uninterrupted, "boundary {boundary}");
            assert_eq!(
                resumed.source_artifact_sha256().unwrap(),
                uninterrupted.source_artifact_sha256().unwrap(),
                "source boundary {boundary}"
            );
            assert_eq!(
                resumed.packed_artifact_sha256().unwrap(),
                uninterrupted.packed_artifact_sha256().unwrap(),
                "packed boundary {boundary}"
            );
        }
    }

    #[test]
    fn empty_word_is_a_real_ordered_chain_link() {
        let identity = identity("empty-word");
        let mut state = ResumableWordHashChainV3::new(&identity).unwrap();
        let initial = state.chain_sha256().to_string();
        let completed = state.append_word(0, []).unwrap();
        assert_eq!(completed.raw_terms, 0);
        assert_eq!(completed.packed_terms, 0);
        assert_ne!(state.chain_sha256(), initial);
        assert_eq!(state.next_word_ordinal(), 1);
        assert_eq!(state.raw_terms(), 0);
    }

    #[test]
    fn reordered_duplicated_and_dropped_inputs_change_the_chain() {
        let identity = identity("mutation");
        let baseline_words = words();
        let mut baseline = ResumableWordHashChainV3::new(&identity).unwrap();
        append_suffix(&mut baseline, &baseline_words, 0);

        let mut reordered_terms = baseline_words.clone();
        reordered_terms[0].swap(0, 1);
        let mut reordered = ResumableWordHashChainV3::new(&identity).unwrap();
        append_suffix(&mut reordered, &reordered_terms, 0);

        let mut duplicated_terms = baseline_words.clone();
        duplicated_terms[2].push(term(4));
        let mut duplicated = ResumableWordHashChainV3::new(&identity).unwrap();
        append_suffix(&mut duplicated, &duplicated_terms, 0);

        let mut dropped_terms = baseline_words.clone();
        dropped_terms[3].pop();
        let mut dropped = ResumableWordHashChainV3::new(&identity).unwrap();
        append_suffix(&mut dropped, &dropped_terms, 0);

        let mut reordered_words = baseline_words.clone();
        reordered_words.swap(2, 3);
        let mut word_order = ResumableWordHashChainV3::new(&identity).unwrap();
        append_suffix(&mut word_order, &reordered_words, 0);

        for mutation in [reordered, duplicated, dropped, word_order] {
            assert_ne!(mutation.chain_sha256(), baseline.chain_sha256());
            assert_ne!(
                mutation.source_artifact_sha256().unwrap(),
                baseline.source_artifact_sha256().unwrap()
            );
        }
    }

    #[test]
    fn identity_and_artifact_domains_are_separated() {
        let mut left = ResumableWordHashChainV3::new(&identity("left")).unwrap();
        let mut right = ResumableWordHashChainV3::new(&identity("right")).unwrap();
        left.append_word(0, [term(7)]).unwrap();
        right.append_word(0, [term(7)]).unwrap();
        assert_ne!(left.chain_sha256(), right.chain_sha256());
        assert_ne!(
            left.source_artifact_sha256().unwrap(),
            left.packed_artifact_sha256().unwrap()
        );
        assert_ne!(
            left.source_artifact_sha256().unwrap(),
            right.source_artifact_sha256().unwrap()
        );
    }

    #[test]
    fn invalid_or_replayed_ordinals_are_rejected() {
        let identity = identity("ordinal");
        let mut state = ResumableWordHashChainV3::new(&identity).unwrap();
        assert_eq!(
            state.start_word(1).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let stale = state.start_word(0).unwrap();
        assert_eq!(stale.word_ordinal(), 0);
        state.append_word(0, [term(8)]).unwrap();
        assert_eq!(
            state.finish_word(stale).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            state.append_word(0, [term(9)]).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn restore_rejects_wrong_identity_and_corrupted_state() {
        let expected_identity = identity("restore");
        let mut state = ResumableWordHashChainV3::new(&expected_identity).unwrap();
        state.append_word(0, [term(10)]).unwrap();
        let encoded = state.to_json().unwrap();
        assert_eq!(
            ResumableWordHashChainV3::restore_json(&encoded, &identity("wrong"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        value["raw_terms"] = 99.into();
        let corrupted = serde_json::to_vec(&value).unwrap();
        assert_eq!(
            ResumableWordHashChainV3::restore_json(&corrupted, &expected_identity)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}
