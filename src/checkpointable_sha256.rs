//! Checkpointable SHA-256 with the same byte contract as `sha2::Sha256`.
//!
//! Full blocks are passed to the compressor in one batch. On x86/x86_64 it
//! runtime-dispatches to SHA-NI when the processor provides SHA, SSE2, SSSE3,
//! and SSE4.1, as the Ryzen 7950X3D does. Other processors use the portable
//! fallback. Unsafe operations are confined to the target-feature-gated SHA-NI
//! module and documented at its dispatch and pointer boundaries.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use std::fmt::Write as _;

const BLOCK_BYTES: usize = 64;
const DIGEST_BYTES: usize = 32;
const MAX_SHA256_BYTES: u64 = u64::MAX / 8;
const INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 continuation state that can be serialized between updates.
///
/// Its serialized representation intentionally matches the former
/// `{ state, total_bytes, buffer }` checkpoint shape. Only live buffered bytes
/// are emitted, so existing valid JSON checkpoints remain readable and newly
/// written checkpoints do not contain the unused fixed-buffer tail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointableSha256 {
    state: [u32; 8],
    total_bytes: u64,
    buffer: [u8; BLOCK_BYTES],
    buffer_len: u8,
}

impl CheckpointableSha256 {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            state: INITIAL_STATE,
            total_bytes: 0,
            buffer: [0; BLOCK_BYTES],
            buffer_len: 0,
        }
    }

    pub(crate) fn continuation_parts(&self) -> ([u32; 8], u64, Vec<u8>) {
        (
            self.state,
            self.total_bytes,
            self.buffer[..usize::from(self.buffer_len)].to_vec(),
        )
    }

    pub(crate) fn from_continuation_parts(
        state: [u32; 8],
        total_bytes: u64,
        buffer: Vec<u8>,
    ) -> Result<Self, String> {
        if buffer.len() >= BLOCK_BYTES {
            return Err("serialized SHA-256 state is invalid".to_string());
        }
        let mut fixed_buffer = [0_u8; BLOCK_BYTES];
        fixed_buffer[..buffer.len()].copy_from_slice(&buffer);
        let continuation = Self {
            state,
            total_bytes,
            buffer: fixed_buffer,
            buffer_len: buffer.len() as u8,
        };
        continuation.validate()?;
        Ok(continuation)
    }

    /// Adds bytes without changing the SHA-256 byte stream at checkpoint
    /// boundaries.
    pub(crate) fn update(&mut self, mut input: &[u8]) -> Result<(), String> {
        let input_len = u64::try_from(input.len())
            .map_err(|_| "SHA-256 input length does not fit u64".to_string())?;
        let new_total = self
            .total_bytes
            .checked_add(input_len)
            .ok_or_else(|| "SHA-256 input length overflow".to_string())?;
        if new_total > MAX_SHA256_BYTES {
            return Err("SHA-256 bit length overflow".to_string());
        }

        let buffered = usize::from(self.buffer_len);
        if buffered != 0 {
            let take = (BLOCK_BYTES - buffered).min(input.len());
            self.buffer[buffered..buffered + take].copy_from_slice(&input[..take]);
            self.buffer_len += take as u8;
            input = &input[take..];
            if usize::from(self.buffer_len) == BLOCK_BYTES {
                compress_full_blocks(&mut self.state, &self.buffer);
                self.buffer.fill(0);
                self.buffer_len = 0;
            }
        }

        let full_bytes = input.len() / BLOCK_BYTES * BLOCK_BYTES;
        if full_bytes != 0 {
            compress_full_blocks(&mut self.state, &input[..full_bytes]);
            input = &input[full_bytes..];
        }
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len() as u8;
        }
        self.total_bytes = new_total;
        Ok(())
    }

    /// Checks all invariants needed to continue from serialized state.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let buffered = usize::from(self.buffer_len);
        if buffered >= BLOCK_BYTES
            || self.total_bytes > MAX_SHA256_BYTES
            || self.total_bytes % BLOCK_BYTES as u64 != buffered as u64
            || self.buffer[buffered..].iter().any(|&byte| byte != 0)
        {
            return Err("serialized SHA-256 state is invalid".to_string());
        }
        Ok(())
    }

    /// Returns the digest without consuming or modifying the continuation.
    pub(crate) fn finalize_bytes(&self) -> Result<[u8; DIGEST_BYTES], String> {
        self.validate()?;
        let buffered = usize::from(self.buffer_len);
        let mut tail = [0_u8; BLOCK_BYTES * 2];
        tail[..buffered].copy_from_slice(&self.buffer[..buffered]);
        tail[buffered] = 0x80;
        let tail_len = if buffered < 56 {
            BLOCK_BYTES
        } else {
            BLOCK_BYTES * 2
        };
        let bit_length = self
            .total_bytes
            .checked_mul(8)
            .ok_or_else(|| "SHA-256 bit length overflow".to_string())?;
        tail[tail_len - 8..tail_len].copy_from_slice(&bit_length.to_be_bytes());

        let mut state = self.state;
        compress_full_blocks(&mut state, &tail[..tail_len]);
        let mut output = [0_u8; DIGEST_BYTES];
        for (chunk, word) in output.chunks_exact_mut(4).zip(state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        Ok(output)
    }

    pub(crate) fn finalize_hex(&self) -> Result<String, String> {
        let digest = self.finalize_bytes()?;
        let mut output = String::with_capacity(DIGEST_BYTES * 2);
        for byte in digest {
            // Writing to a String is infallible.
            write!(&mut output, "{byte:02x}").expect("String formatting cannot fail");
        }
        Ok(output)
    }

    /// Reports the compressor backend selected on this host.
    pub(crate) fn backend_name() -> &'static str {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if x86_sha_ni::available() {
                return "x86-sha-ni";
            }
        }
        "portable"
    }
}

impl Default for CheckpointableSha256 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize)]
struct Sha256WireRef<'a> {
    state: &'a [u32; 8],
    total_bytes: u64,
    buffer: &'a [u8],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sha256Wire {
    state: [u32; 8],
    total_bytes: u64,
    buffer: Vec<u8>,
}

impl Serialize for CheckpointableSha256 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        debug_assert!(self.validate().is_ok());
        Sha256WireRef {
            state: &self.state,
            total_bytes: self.total_bytes,
            buffer: &self.buffer[..usize::from(self.buffer_len)],
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CheckpointableSha256 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Sha256Wire::deserialize(deserializer)?;
        if wire.buffer.len() >= BLOCK_BYTES
            || wire.total_bytes > MAX_SHA256_BYTES
            || wire.total_bytes % BLOCK_BYTES as u64 != wire.buffer.len() as u64
        {
            return Err(D::Error::custom("serialized SHA-256 state is invalid"));
        }
        let mut buffer = [0_u8; BLOCK_BYTES];
        buffer[..wire.buffer.len()].copy_from_slice(&wire.buffer);
        Ok(Self {
            state: wire.state,
            total_bytes: wire.total_bytes,
            buffer,
            buffer_len: wire.buffer.len() as u8,
        })
    }
}

/// Compresses a nonempty, block-aligned byte slice in one backend dispatch.
#[inline]
fn compress_full_blocks(state: &mut [u32; 8], bytes: &[u8]) {
    debug_assert!(!bytes.is_empty());
    assert_eq!(bytes.len() % BLOCK_BYTES, 0);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if x86_sha_ni::available() {
        // SAFETY: `available` verifies every target feature enabled on
        // `compress`. `bytes` is nonempty and block-aligned by the assertions
        // above; the backend only loads within each 64-byte chunk.
        unsafe { x86_sha_ni::compress(state, bytes) };
        return;
    }
    portable_compress(state, bytes);
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn portable_compress(state: &mut [u32; 8], bytes: &[u8]) {
    debug_assert_eq!(bytes.len() % BLOCK_BYTES, 0);
    let mut schedule = [0_u32; 64];
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        for (word, chunk) in schedule[..16].iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_be_bytes(chunk.try_into().expect("four-byte SHA-256 word"));
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_sha_ni {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    pub(super) fn available() -> bool {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            std::arch::is_x86_feature_detected!("sha")
                && std::arch::is_x86_feature_detected!("sse2")
                && std::arch::is_x86_feature_detected!("ssse3")
                && std::arch::is_x86_feature_detected!("sse4.1")
        })
    }

    #[inline(always)]
    unsafe fn schedule(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
        // SAFETY: The caller has enabled SHA and SSSE3 for this function's
        // entire call graph. These operations do not access memory.
        unsafe {
            let t1 = _mm_sha256msg1_epu32(v0, v1);
            let t2 = _mm_alignr_epi8(v3, v2, 4);
            _mm_sha256msg2_epu32(_mm_add_epi32(t1, t2), v3)
        }
    }

    macro_rules! rounds4 {
        ($abef:ident, $cdgh:ident, $words:expr, $index:expr) => {{
            let constants = &super::K[$index * 4..$index * 4 + 4];
            let k = _mm_set_epi32(
                constants[3] as i32,
                constants[2] as i32,
                constants[1] as i32,
                constants[0] as i32,
            );
            let wk = _mm_add_epi32($words, k);
            $cdgh = _mm_sha256rnds2_epu32($cdgh, $abef, wk);
            $abef = _mm_sha256rnds2_epu32($abef, $cdgh, _mm_shuffle_epi32(wk, 0x0e));
        }};
    }

    macro_rules! schedule_rounds4 {
        ($abef:ident, $cdgh:ident, $w0:expr, $w1:expr, $w2:expr, $w3:expr, $w4:expr, $i:expr) => {{
            $w4 = schedule($w0, $w1, $w2, $w3);
            rounds4!($abef, $cdgh, $w4, $i);
        }};
    }

    /// # Safety
    /// The caller must verify SHA, SSE2, SSSE3, and SSE4.1 support and provide
    /// a nonempty `bytes` slice whose length is a multiple of 64.
    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    pub(super) unsafe fn compress(state: &mut [u32; 8], bytes: &[u8]) {
        debug_assert!(!bytes.is_empty());
        debug_assert_eq!(bytes.len() % super::BLOCK_BYTES, 0);

        // SAFETY: Target features are enabled on this function. `state` owns
        // eight u32 values, so each 16-byte unaligned state load and store is
        // in bounds. Each `block` is exactly 64 bytes, so its four unaligned
        // 16-byte loads are in bounds. Unaligned intrinsics impose no pointer
        // alignment requirement. All remaining intrinsics are register-only.
        unsafe {
            let mask = _mm_set_epi64x(
                0x0c0d_0e0f_0809_0a0b_u64 as i64,
                0x0405_0607_0001_0203_u64 as i64,
            );
            let state_ptr = state.as_ptr().cast::<__m128i>();
            let dcba = _mm_loadu_si128(state_ptr);
            let efgh = _mm_loadu_si128(state_ptr.add(1));
            let cdab = _mm_shuffle_epi32(dcba, 0xb1);
            let efgh = _mm_shuffle_epi32(efgh, 0x1b);
            let mut abef = _mm_alignr_epi8(cdab, efgh, 8);
            let mut cdgh = _mm_blend_epi16(efgh, cdab, 0xf0);

            for block in bytes.chunks_exact(super::BLOCK_BYTES) {
                let saved_abef = abef;
                let saved_cdgh = cdgh;
                let data = block.as_ptr().cast::<__m128i>();
                let mut w0 = _mm_shuffle_epi8(_mm_loadu_si128(data), mask);
                let mut w1 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(1)), mask);
                let mut w2 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(2)), mask);
                let mut w3 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(3)), mask);
                let mut w4;

                rounds4!(abef, cdgh, w0, 0);
                rounds4!(abef, cdgh, w1, 1);
                rounds4!(abef, cdgh, w2, 2);
                rounds4!(abef, cdgh, w3, 3);
                schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 4);
                schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 5);
                schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 6);
                schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 7);
                schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 8);
                schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 9);
                schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 10);
                schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 11);
                schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 12);
                schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 13);
                schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 14);
                schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 15);

                abef = _mm_add_epi32(abef, saved_abef);
                cdgh = _mm_add_epi32(cdgh, saved_cdgh);
            }

            let feba = _mm_shuffle_epi32(abef, 0x1b);
            let dchg = _mm_shuffle_epi32(cdgh, 0xb1);
            let dcba = _mm_blend_epi16(feba, dchg, 0xf0);
            let hgef = _mm_alignr_epi8(dchg, feba, 8);
            let state_ptr = state.as_mut_ptr().cast::<__m128i>();
            _mm_storeu_si128(state_ptr, dcba);
            _mm_storeu_si128(state_ptr.add(1), hgef);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::time::Instant;

    fn reference(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
        Sha256::digest(bytes).into()
    }

    fn deterministic_bytes(length: usize, mut state: u64) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push(state as u8);
        }
        bytes
    }

    #[test]
    fn matches_sha2_at_padding_and_block_boundaries() {
        for length in [
            0, 1, 2, 7, 31, 32, 54, 55, 56, 57, 63, 64, 65, 119, 120, 121, 127, 128, 129, 255, 256,
            257, 4095, 4096, 4097,
        ] {
            let bytes = deterministic_bytes(length, 0x243f_6a88_85a3_08d3);
            for chunk_size in [1, 2, 3, 7, 23, 63, 64, 65, 127, 1024, usize::MAX] {
                let mut observed = CheckpointableSha256::new();
                if chunk_size == usize::MAX {
                    observed.update(&bytes).unwrap();
                } else {
                    for chunk in bytes.chunks(chunk_size) {
                        observed.update(chunk).unwrap();
                    }
                }
                assert_eq!(observed.finalize_bytes().unwrap(), reference(&bytes));
                assert_eq!(observed.finalize_bytes().unwrap(), reference(&bytes));
            }
        }
    }

    #[test]
    fn randomized_chunks_and_checkpoints_match_sha2() {
        let mut rng = 0x1319_8a2e_0370_7344_u64;
        for case in 0..256 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let length = (rng as usize) % 32_768;
            let bytes = deterministic_bytes(length, rng ^ case);
            let mut observed = CheckpointableSha256::new();
            let mut offset = 0;
            while offset < bytes.len() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let take = (1 + (rng as usize % 521)).min(bytes.len() - offset);
                observed.update(&bytes[offset..offset + take]).unwrap();
                offset += take;
                if rng & 3 == 0 {
                    let checkpoint = serde_json::to_vec(&observed).unwrap();
                    observed = serde_json::from_slice(&checkpoint).unwrap();
                }
            }
            assert_eq!(observed.finalize_bytes().unwrap(), reference(&bytes));
        }
    }

    #[test]
    fn wire_shape_is_backward_compatible_and_rejects_bad_continuations() {
        let mut observed = CheckpointableSha256::new();
        observed.update(b"checkpoint").unwrap();
        let json = serde_json::to_value(&observed).unwrap();
        assert_eq!(json["total_bytes"], 10);
        assert_eq!(json["buffer"].as_array().unwrap().len(), 10);
        assert!(json.get("buffer_len").is_none());
        let restored: CheckpointableSha256 = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(restored, observed);

        let mut bad_length = json.clone();
        bad_length["total_bytes"] = 11.into();
        assert!(serde_json::from_value::<CheckpointableSha256>(bad_length).is_err());
        let mut oversized = json;
        oversized["buffer"] = serde_json::Value::Array(vec![0.into(); BLOCK_BYTES]);
        assert!(serde_json::from_value::<CheckpointableSha256>(oversized).is_err());
    }

    #[test]
    fn known_vector_and_hex_match() {
        let mut observed = CheckpointableSha256::new();
        observed.update(b"abc").unwrap();
        assert_eq!(
            observed.finalize_hex().unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn continuation_parts_roundtrip_and_continue_exactly() {
        let prefix = deterministic_bytes(137, 0x082e_fa98_ec4e_6c89);
        let suffix = deterministic_bytes(911, 0x4528_21e6_38d0_1377);
        let mut original = CheckpointableSha256::new();
        original.update(&prefix).unwrap();
        let (state, total_bytes, buffer) = original.continuation_parts();
        let mut restored =
            CheckpointableSha256::from_continuation_parts(state, total_bytes, buffer).unwrap();
        assert_eq!(restored, original);

        original.update(&suffix).unwrap();
        restored.update(&suffix).unwrap();
        let mut complete = prefix;
        complete.extend_from_slice(&suffix);
        assert_eq!(restored.finalize_bytes().unwrap(), reference(&complete));
        assert_eq!(
            restored.finalize_bytes().unwrap(),
            original.finalize_bytes().unwrap()
        );
    }

    /// Manual throughput helper. Run in release mode with:
    /// `cargo test --release checkpointable_sha256::tests::throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "manual SHA-256 throughput microbenchmark"]
    fn throughput() {
        const MIB: usize = 1024 * 1024;
        const TOTAL_MIB: usize = 256;
        let input = deterministic_bytes(MIB, 0xa409_3822_299f_31d0);

        let started = Instant::now();
        let mut checkpointable = CheckpointableSha256::new();
        for _ in 0..TOTAL_MIB {
            checkpointable.update(black_box(&input)).unwrap();
        }
        let checkpointable_digest = black_box(checkpointable.finalize_bytes().unwrap());
        let checkpointable_elapsed = started.elapsed();

        let started = Instant::now();
        let mut baseline = Sha256::new();
        for _ in 0..TOTAL_MIB {
            baseline.update(black_box(&input));
        }
        let baseline_digest: [u8; DIGEST_BYTES] = black_box(baseline.finalize().into());
        let baseline_elapsed = started.elapsed();
        assert_eq!(checkpointable_digest, baseline_digest);

        eprintln!(
            "backend={} checkpointable={:.1} MiB/s sha2={:.1} MiB/s ratio={:.3}",
            CheckpointableSha256::backend_name(),
            TOTAL_MIB as f64 / checkpointable_elapsed.as_secs_f64(),
            TOTAL_MIB as f64 / baseline_elapsed.as_secs_f64(),
            checkpointable_elapsed.as_secs_f64() / baseline_elapsed.as_secs_f64(),
        );
    }
}
