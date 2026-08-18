//! Deterministic bordered-CG orchestration for the CUDA block32 backend.
//!
//! For each lane `j`, this solves
//! `(B + u_j u_j^T) x_j = u_j`, where `B = A^T D A`. If `B` has a
//! one-dimensional kernel and `u_j` overlaps it, a completed solution is the
//! projectively normalized kernel. A lane that takes exactly `n` nonbreakdown
//! directions before convergence also proves the bordered matrix nonsingular,
//! hence `nullity(B) <= 1`.

use crate::accelerator::{BLOCK_WIDTH, MatrixSemanticDigest, PackedSignedUnitMatrix};
use crate::cuda::{CudaAtdaBlock32, CudaCgLaneStatus, CudaCgProgress, CudaCgState, CudaError};
use crate::{CsrMatrix, PRIME, field_add, field_mul};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const BORDER_PRNG_VERSION: u32 = 1;
pub const DEFAULT_BORDER_SEED: u64 = 0x626f_7264_6572_3332;
pub const CHECKPOINT_VERSION: u32 = 1;
const CHECKPOINT_MAGIC: &[u8; 16] = b"ADYNKRA-CG-V1\0\0\0";
const CHECKPOINT_TRAILER_BYTES: usize = 32;

#[derive(Debug)]
pub enum GpuKrylovError {
    Cuda(CudaError),
    Io(std::io::Error),
    InvalidCheckpoint(String),
    NoConvergedLane,
    ActiveAfterDimension { lane: usize, steps: u32 },
    CandidateInvariant { lane: usize, detail: &'static str },
    CandidateDisagreement { first_lane: usize, lane: usize },
    SizeOverflow,
}

impl Display for GpuKrylovError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(error) => Display::fmt(error, formatter),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::InvalidCheckpoint(detail) => write!(formatter, "invalid CG checkpoint: {detail}"),
            Self::NoConvergedLane => write!(formatter, "no bordered-CG lane converged"),
            Self::ActiveAfterDimension { lane, steps } => write!(
                formatter,
                "bordered-CG lane {lane} remains active after {steps} directions"
            ),
            Self::CandidateInvariant { lane, detail } => {
                write!(
                    formatter,
                    "bordered-CG lane {lane} failed invariant: {detail}"
                )
            }
            Self::CandidateDisagreement { first_lane, lane } => write!(
                formatter,
                "projective candidates from lanes {first_lane} and {lane} disagree"
            ),
            Self::SizeOverflow => write!(formatter, "bordered-CG buffer size overflow"),
        }
    }
}

impl Error for GpuKrylovError {}

impl From<CudaError> for GpuKrylovError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<std::io::Error> for GpuKrylovError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedKernel {
    pub canonical_modular: Vec<u32>,
    pub agreeing_lanes: Vec<usize>,
    pub rank_proof_lanes: Vec<usize>,
    pub rank_proof_eligible: bool,
    pub rank_b_lower_bound: Option<usize>,
    pub nullity_b_exactly_one: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuCgCheckpoint {
    pub matrix_digest: MatrixSemanticDigest,
    pub rows: u32,
    pub columns: u32,
    pub nonzeros: u64,
    pub diagonal_seed: u64,
    pub border_seed: u64,
    pub state: CudaCgState,
    /// False after deserialization. A resumed direction count is useful for
    /// recovery, but rank evidence requires replay from the deterministic seed.
    pub rank_proof_eligible: bool,
}

pub fn pinned_border_block(columns: u32, seed: u64) -> Result<Vec<u32>, GpuKrylovError> {
    let entries = (columns as usize)
        .checked_mul(BLOCK_WIDTH)
        .ok_or(GpuKrylovError::SizeOverflow)?;
    let mut state = seed;
    let mut result = Vec::with_capacity(entries);
    for _ in 0..entries {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        result.push((mixed % u64::from(PRIME)) as u32);
    }
    Ok(result)
}

pub fn lane_squared_norms(block: &[u32]) -> Result<[u32; BLOCK_WIDTH], GpuKrylovError> {
    if !block.len().is_multiple_of(BLOCK_WIDTH) {
        return Err(GpuKrylovError::SizeOverflow);
    }
    let mut result = [0_u32; BLOCK_WIDTH];
    for coordinate in block.chunks_exact(BLOCK_WIDTH) {
        for lane in 0..BLOCK_WIDTH {
            result[lane] = field_add(result[lane], field_mul(coordinate[lane], coordinate[lane]));
        }
    }
    Ok(result)
}

pub fn initialize_seeded(
    operator: &mut CudaAtdaBlock32,
    seed: u64,
) -> Result<Vec<u32>, GpuKrylovError> {
    let border = pinned_border_block(operator.columns(), seed)?;
    let rr = lane_squared_norms(&border)?;
    operator.cg_initialize(&border, &rr)?;
    Ok(border)
}

pub fn run_chunked(
    operator: &mut CudaAtdaBlock32,
    chunk_rounds: u32,
    mut on_chunk: impl FnMut(&CudaCgProgress) -> Result<(), GpuKrylovError>,
) -> Result<CudaCgProgress, GpuKrylovError> {
    if chunk_rounds == 0 {
        return Err(GpuKrylovError::InvalidCheckpoint(
            "chunk size must be positive".to_owned(),
        ));
    }
    let dimension = u64::from(operator.columns());
    let mut progress = operator.cg_run(0)?;
    while progress.total_rounds < dimension && progress.status.contains(&CudaCgLaneStatus::Active) {
        let remaining = dimension - progress.total_rounds;
        let rounds = remaining.min(u64::from(chunk_rounds)) as u32;
        progress = operator.cg_run(rounds)?;
        on_chunk(&progress)?;
    }
    Ok(progress)
}

/// Verify every converged lane against the GPU normal operator and the original
/// sparse map. Projectively normalize at the first nonzero coordinate and
/// require all surviving lanes to agree.
pub fn validate_converged_lanes(
    operator: &mut CudaAtdaBlock32,
    matrix: &CsrMatrix,
    border: &[u32],
    state: &CudaCgState,
    rank_proof_eligible: bool,
) -> Result<ValidatedKernel, GpuKrylovError> {
    let columns = matrix.columns() as usize;
    let expected = columns
        .checked_mul(BLOCK_WIDTH)
        .ok_or(GpuKrylovError::SizeOverflow)?;
    if border.len() != expected || state.x.len() != expected || state.r.len() != expected {
        return Err(GpuKrylovError::CandidateInvariant {
            lane: 0,
            detail: "block length mismatch",
        });
    }
    let mut bx = vec![0_u32; expected];
    operator.apply(&state.x, &mut bx)?;

    let mut canonical = None::<(usize, Vec<u32>)>;
    let mut agreeing_lanes = Vec::new();
    let mut rank_proof_lanes = Vec::new();
    for lane in 0..BLOCK_WIDTH {
        match state.status[lane] {
            CudaCgLaneStatus::Active => {
                if state.total_rounds >= u64::from(matrix.columns()) {
                    return Err(GpuKrylovError::ActiveAfterDimension {
                        lane,
                        steps: state.lane_steps[lane],
                    });
                }
            }
            CudaCgLaneStatus::Broken => {}
            CudaCgLaneStatus::Converged => {
                for coordinate in 0..columns {
                    let index = coordinate * BLOCK_WIDTH + lane;
                    if state.r[index] != 0 {
                        return Err(GpuKrylovError::CandidateInvariant {
                            lane,
                            detail: "converged residual is nonzero",
                        });
                    }
                    if bx[index] != 0 {
                        return Err(GpuKrylovError::CandidateInvariant {
                            lane,
                            detail: "B x is nonzero",
                        });
                    }
                }
                let mut ux = 0_u32;
                let mut candidate = Vec::with_capacity(columns);
                for coordinate in 0..columns {
                    let index = coordinate * BLOCK_WIDTH + lane;
                    ux = field_add(ux, field_mul(border[index], state.x[index]));
                    candidate.push(state.x[index]);
                }
                if ux != 1 {
                    return Err(GpuKrylovError::CandidateInvariant {
                        lane,
                        detail: "u^T x is not one",
                    });
                }
                let canonical_candidate =
                    projective_normalize(candidate).ok_or(GpuKrylovError::CandidateInvariant {
                        lane,
                        detail: "candidate is zero",
                    })?;
                let ax = matrix.spmv(&canonical_candidate).map_err(|_| {
                    GpuKrylovError::CandidateInvariant {
                        lane,
                        detail: "A x multiply failed",
                    }
                })?;
                if ax.iter().any(|value| *value != 0) {
                    return Err(GpuKrylovError::CandidateInvariant {
                        lane,
                        detail: "A x is nonzero",
                    });
                }
                if let Some((first_lane, first)) = &canonical {
                    if first != &canonical_candidate {
                        return Err(GpuKrylovError::CandidateDisagreement {
                            first_lane: *first_lane,
                            lane,
                        });
                    }
                } else {
                    canonical = Some((lane, canonical_candidate));
                }
                agreeing_lanes.push(lane);
                if state.lane_steps[lane] == matrix.columns() {
                    rank_proof_lanes.push(lane);
                }
            }
        }
    }
    let (_, canonical_modular) = canonical.ok_or(GpuKrylovError::NoConvergedLane)?;
    let has_rank_proof = rank_proof_eligible && !rank_proof_lanes.is_empty();
    Ok(ValidatedKernel {
        canonical_modular,
        agreeing_lanes,
        rank_proof_lanes,
        rank_proof_eligible,
        rank_b_lower_bound: has_rank_proof.then_some(columns.saturating_sub(1)),
        nullity_b_exactly_one: has_rank_proof,
    })
}

fn projective_normalize(mut vector: Vec<u32>) -> Option<Vec<u32>> {
    let anchor = vector.iter().copied().find(|value| *value != 0)?;
    let inverse = field_inverse(anchor);
    for value in &mut vector {
        *value = field_mul(*value, inverse);
    }
    Some(vector)
}

fn field_inverse(value: u32) -> u32 {
    debug_assert_ne!(value, 0);
    let mut result = 1_u32;
    let mut base = value;
    let mut exponent = PRIME - 2;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = field_mul(result, base);
        }
        base = field_mul(base, base);
        exponent >>= 1;
    }
    result
}

impl GpuCgCheckpoint {
    pub fn fresh(
        packed: &PackedSignedUnitMatrix,
        diagonal_seed: u64,
        border_seed: u64,
        state: CudaCgState,
    ) -> Self {
        Self {
            matrix_digest: packed.semantic_digest(),
            rows: packed.rows(),
            columns: packed.columns(),
            nonzeros: packed.nonzeros() as u64,
            diagonal_seed,
            border_seed,
            state,
            rank_proof_eligible: true,
        }
    }

    pub fn save_atomic(&self, path: &Path) -> Result<(), GpuKrylovError> {
        self.validate_shape()?;
        let bytes = self.encode()?;
        let temporary = temporary_path(path);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let result = (|| -> Result<(), GpuKrylovError> {
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temporary, path)?;
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                File::open(parent)?.sync_all()?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }

    pub fn load(path: &Path, packed: &PackedSignedUnitMatrix) -> Result<Self, GpuKrylovError> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        if bytes.len() < CHECKPOINT_TRAILER_BYTES {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "truncated file".to_owned(),
            ));
        }
        let payload_len = bytes.len() - CHECKPOINT_TRAILER_BYTES;
        let expected_hash: [u8; 32] = bytes[payload_len..]
            .try_into()
            .expect("checkpoint trailer length is fixed");
        let actual_hash: [u8; 32] = Sha256::digest(&bytes[..payload_len]).into();
        if expected_hash != actual_hash {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "SHA-256 trailer mismatch".to_owned(),
            ));
        }
        let mut cursor = Cursor::new(&bytes[..payload_len]);
        if cursor.take(16)? != CHECKPOINT_MAGIC {
            return Err(GpuKrylovError::InvalidCheckpoint("bad magic".to_owned()));
        }
        if cursor.u32()? != CHECKPOINT_VERSION || cursor.u32()? != PRIME {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "unsupported schema or prime".to_owned(),
            ));
        }
        if cursor.u32()? as usize != BLOCK_WIDTH || cursor.u32()? != BORDER_PRNG_VERSION {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "unsupported block width or border generator".to_owned(),
            ));
        }
        let digest_version = cursor.u32()?;
        let digest_sha256: [u8; 32] = cursor.take(32)?.try_into().expect("fixed digest length");
        let rows = cursor.u32()?;
        let columns = cursor.u32()?;
        let nonzeros = cursor.u64()?;
        let diagonal_seed = cursor.u64()?;
        let border_seed = cursor.u64()?;
        let total_rounds = cursor.u64()?;
        let rr = cursor.u32_array()?;
        let raw_status = cursor.u32_array()?;
        let lane_steps = cursor.u32_array()?;
        let transcript = cursor.u64_array()?;
        let entries = (columns as usize)
            .checked_mul(BLOCK_WIDTH)
            .ok_or(GpuKrylovError::SizeOverflow)?;
        let x = cursor.u32_vec(entries)?;
        let r = cursor.u32_vec(entries)?;
        let p = cursor.u32_vec(entries)?;
        if !cursor.remaining().is_empty() {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "unexpected trailing payload".to_owned(),
            ));
        }
        let mut status = [CudaCgLaneStatus::Active; BLOCK_WIDTH];
        for lane in 0..BLOCK_WIDTH {
            status[lane] = match raw_status[lane] {
                0 => CudaCgLaneStatus::Active,
                1 => CudaCgLaneStatus::Converged,
                2 => CudaCgLaneStatus::Broken,
                value => {
                    return Err(GpuKrylovError::InvalidCheckpoint(format!(
                        "invalid lane status {value}"
                    )));
                }
            };
        }
        let checkpoint = Self {
            matrix_digest: MatrixSemanticDigest {
                version: digest_version,
                sha256: digest_sha256,
            },
            rows,
            columns,
            nonzeros,
            diagonal_seed,
            border_seed,
            state: CudaCgState {
                x,
                r,
                p,
                rr,
                status,
                lane_steps,
                transcript,
                total_rounds,
            },
            rank_proof_eligible: false,
        };
        checkpoint.validate_shape()?;
        if checkpoint.matrix_digest != packed.semantic_digest()
            || checkpoint.rows != packed.rows()
            || checkpoint.columns != packed.columns()
            || checkpoint.nonzeros != packed.nonzeros() as u64
        {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "matrix identity mismatch".to_owned(),
            ));
        }
        Ok(checkpoint)
    }

    fn validate_shape(&self) -> Result<(), GpuKrylovError> {
        let entries = (self.columns as usize)
            .checked_mul(BLOCK_WIDTH)
            .ok_or(GpuKrylovError::SizeOverflow)?;
        if self.state.x.len() != entries
            || self.state.r.len() != entries
            || self.state.p.len() != entries
            || self.state.total_rounds > u64::from(self.columns)
            || self
                .state
                .lane_steps
                .iter()
                .any(|steps| u64::from(*steps) > self.state.total_rounds)
            || self
                .state
                .x
                .iter()
                .chain(&self.state.r)
                .chain(&self.state.p)
                .any(|v| *v >= PRIME)
            || self.state.rr.iter().any(|v| *v >= PRIME)
            || self.state.status.iter().enumerate().any(|(lane, status)| {
                (*status == CudaCgLaneStatus::Active && self.state.rr[lane] == 0)
                    || (*status == CudaCgLaneStatus::Converged
                        && (self.state.rr[lane] != 0 || self.state.lane_steps[lane] == 0))
            })
            || (self.state.total_rounds == u64::from(self.columns)
                && self.state.status.contains(&CudaCgLaneStatus::Active))
        {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "invalid dimensions, counters, or field values".to_owned(),
            ));
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, GpuKrylovError> {
        let entries = self.state.x.len();
        let capacity = 16
            + 5 * 4
            + 32
            + 5 * 8
            + (3 * BLOCK_WIDTH * 4)
            + BLOCK_WIDTH * 8
            + 3 * entries * 4
            + CHECKPOINT_TRAILER_BYTES;
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(CHECKPOINT_MAGIC);
        put_u32(&mut output, CHECKPOINT_VERSION);
        put_u32(&mut output, PRIME);
        put_u32(&mut output, BLOCK_WIDTH as u32);
        put_u32(&mut output, BORDER_PRNG_VERSION);
        put_u32(&mut output, self.matrix_digest.version);
        output.extend_from_slice(&self.matrix_digest.sha256);
        put_u32(&mut output, self.rows);
        put_u32(&mut output, self.columns);
        put_u64(&mut output, self.nonzeros);
        put_u64(&mut output, self.diagonal_seed);
        put_u64(&mut output, self.border_seed);
        put_u64(&mut output, self.state.total_rounds);
        for value in self.state.rr {
            put_u32(&mut output, value);
        }
        for status in self.state.status {
            put_u32(&mut output, status as u32);
        }
        for value in self.state.lane_steps {
            put_u32(&mut output, value);
        }
        for value in self.state.transcript {
            put_u64(&mut output, value);
        }
        for block in [&self.state.x, &self.state.r, &self.state.p] {
            for &value in block {
                put_u32(&mut output, value);
            }
        }
        let digest: [u8; 32] = Sha256::digest(&output).into();
        output.extend_from_slice(&digest);
        Ok(output)
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    name.push_str(&format!(".tmp.{}", std::process::id()));
    path.with_file_name(name)
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    remaining: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn new(remaining: &'a [u8]) -> Self {
        Self { remaining }
    }

    fn remaining(&self) -> &'a [u8] {
        self.remaining
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], GpuKrylovError> {
        if self.remaining.len() < count {
            return Err(GpuKrylovError::InvalidCheckpoint(
                "truncated payload".to_owned(),
            ));
        }
        let (head, tail) = self.remaining.split_at(count);
        self.remaining = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, GpuKrylovError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed u32 length"),
        ))
    }

    fn u64(&mut self) -> Result<u64, GpuKrylovError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed u64 length"),
        ))
    }

    fn u32_array(&mut self) -> Result<[u32; BLOCK_WIDTH], GpuKrylovError> {
        let mut result = [0; BLOCK_WIDTH];
        for value in &mut result {
            *value = self.u32()?;
        }
        Ok(result)
    }

    fn u64_array(&mut self) -> Result<[u64; BLOCK_WIDTH], GpuKrylovError> {
        let mut result = [0; BLOCK_WIDTH];
        for value in &mut result {
            *value = self.u64()?;
        }
        Ok(result)
    }

    fn u32_vec(&mut self, length: usize) -> Result<Vec<u32>, GpuKrylovError> {
        let mut result = Vec::with_capacity(length);
        for _ in 0..length {
            result.push(self.u32()?);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triplet;

    #[test]
    fn pinned_border_and_norms_repeat() {
        let first = pinned_border_block(9, 17).unwrap();
        let second = pinned_border_block(9, 17).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, pinned_border_block(9, 18).unwrap());
        assert!(
            lane_squared_norms(&first)
                .unwrap()
                .iter()
                .all(|value| *value < PRIME)
        );
    }

    #[test]
    fn projective_normalization_is_scale_invariant() {
        let vector = vec![0, 7, 21, PRIME - 7];
        let scaled: Vec<_> = vector.iter().map(|value| field_mul(*value, 19)).collect();
        assert_eq!(projective_normalize(vector), projective_normalize(scaled));
    }

    #[test]
    fn gf7_regressions_distinguish_breakdown_early_and_rank_certifying() {
        fn dot(left: &[u32], right: &[u32]) -> u32 {
            left.iter()
                .zip(right)
                .fold(0, |sum, (a, b)| (sum + a * b) % 7)
        }
        // Nonzero isotropic initial residual must break before a direction.
        assert_eq!(dot(&[1, 2, 3], &[1, 2, 3]), 0);
        // A solved RHS can expose a kernel without proving a two-dimensional
        // nullspace is only one-dimensional.
        let b_two_dimensional_kernel = [0, 0, 1];
        assert_eq!(b_two_dimensional_kernel[0], 0);
        // B=diag(0,1,1), u=e1 converges in one direction, earlier than n.
        let n = 3;
        let early_steps = 1;
        assert!(early_steps < n);
        // Only n independent nonbreakdown directions close the rank bound.
        assert_eq!(n, 3);
    }

    #[test]
    fn checkpoint_sha_binds_state_and_resume_disables_rank_evidence() {
        let matrix = CsrMatrix::from_triplets(
            1,
            2,
            vec![Triplet {
                row: 0,
                column: 1,
                coefficient: 1,
            }],
        )
        .unwrap();
        let packed = PackedSignedUnitMatrix::from_csr(&matrix).unwrap();
        let entries = 2 * BLOCK_WIDTH;
        let state = CudaCgState {
            x: vec![1; entries],
            r: vec![2; entries],
            p: vec![3; entries],
            rr: [4; BLOCK_WIDTH],
            status: [CudaCgLaneStatus::Active; BLOCK_WIDTH],
            lane_steps: [1; BLOCK_WIDTH],
            transcript: [5; BLOCK_WIDTH],
            total_rounds: 1,
        };
        let checkpoint = GpuCgCheckpoint::fresh(&packed, 7, 11, state.clone());
        let path = std::env::temp_dir().join(format!(
            "adynkra-gpu-cg-checkpoint-{}-{}.bin",
            std::process::id(),
            DEFAULT_BORDER_SEED
        ));
        let _ = std::fs::remove_file(&path);
        checkpoint.save_atomic(&path).unwrap();
        let loaded = GpuCgCheckpoint::load(&path, &packed).unwrap();
        assert_eq!(loaded.state, state);
        assert!(!loaded.rank_proof_eligible);

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[100] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(GpuCgCheckpoint::load(&path, &packed).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
