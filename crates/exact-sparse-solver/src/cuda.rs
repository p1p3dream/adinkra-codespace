//! Optional CUDA backend for exact packed block32 normal-operator products.
//!
//! Matrix storage, the nonzero diagonal, and two Krylov blocks stay resident on
//! the selected device. `apply_steps` therefore chains `A^T D A` products
//! without host transfers between steps.

use crate::accelerator::{BLOCK_WIDTH, MatrixSemanticDigest, PackedSignedUnitMatrix};
use std::error::Error;
use std::ffi::{CStr, c_char, c_int};
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

const ERROR_CAPACITY: usize = 1024;
pub const CUDA_ABI_VERSION: u32 = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaError {
    pub operation: &'static str,
    pub code: i32,
    pub message: String,
}

impl Display for CudaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "CUDA {} failed with code {}: {}",
            self.operation, self.code, self.message
        )
    }
}

impl Error for CudaError {}

#[repr(C)]
struct RawCudaOperator {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn adynkra_exact_cuda_abi_version() -> u32;
    fn adynkra_exact_cuda_create(
        device: c_int,
        rows: u32,
        columns: u32,
        nonzeros: u32,
        csr_offsets: *const u32,
        csr_entries: *const u32,
        transpose_offsets: *const u32,
        transpose_entries: *const u32,
        diagonal: *const u32,
        output: *mut *mut RawCudaOperator,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_destroy(operator: *mut RawCudaOperator);
    fn adynkra_exact_cuda_upload(
        operator: *mut RawCudaOperator,
        input: *const u32,
        input_entries: usize,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_apply_steps(
        operator: *mut RawCudaOperator,
        steps: u32,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_download(
        operator: *mut RawCudaOperator,
        output: *mut u32,
        output_entries: usize,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_device_name(
        operator: *mut RawCudaOperator,
        output: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_cg_initialize(
        operator: *mut RawCudaOperator,
        border: *const u32,
        block_entries: usize,
        initial_rr: *const u32,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_cg_run(
        operator: *mut RawCudaOperator,
        rounds: u32,
        total_rounds: *mut u64,
        status: *mut u32,
        lane_steps: *mut u32,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_cg_download_state(
        operator: *mut RawCudaOperator,
        x: *mut u32,
        r: *mut u32,
        p: *mut u32,
        block_entries: usize,
        rr: *mut u32,
        status: *mut u32,
        lane_steps: *mut u32,
        transcript: *mut u64,
        total_rounds: *mut u64,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn adynkra_exact_cuda_cg_upload_state(
        operator: *mut RawCudaOperator,
        border: *const u32,
        x: *const u32,
        r: *const u32,
        p: *const u32,
        block_entries: usize,
        rr: *const u32,
        status: *const u32,
        lane_steps: *const u32,
        transcript: *const u64,
        total_rounds: u64,
        message: *mut c_char,
        capacity: usize,
    ) -> c_int;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CudaCgLaneStatus {
    Active = 0,
    Converged = 1,
    Broken = 2,
}

impl TryFrom<u32> for CudaCgLaneStatus {
    type Error = CudaError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Active),
            1 => Ok(Self::Converged),
            2 => Ok(Self::Broken),
            _ => Err(rust_error(
                "cg_status",
                format!("backend returned invalid CG lane status {value}"),
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaCgProgress {
    pub total_rounds: u64,
    pub status: [CudaCgLaneStatus; BLOCK_WIDTH],
    pub lane_steps: [u32; BLOCK_WIDTH],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaCgState {
    pub x: Vec<u32>,
    pub r: Vec<u32>,
    pub p: Vec<u32>,
    pub rr: [u32; BLOCK_WIDTH],
    pub status: [CudaCgLaneStatus; BLOCK_WIDTH],
    pub lane_steps: [u32; BLOCK_WIDTH],
    pub transcript: [u64; BLOCK_WIDTH],
    pub total_rounds: u64,
}

/// Owning CUDA normal operator. The marker deliberately keeps the stream and
/// device allocations on the creating host thread unless an explicit threaded
/// ownership policy is added later.
pub struct CudaAtdaBlock32 {
    raw: NonNull<RawCudaOperator>,
    rows: u32,
    columns: u32,
    nonzeros: usize,
    device: i32,
    semantic_digest: MatrixSemanticDigest,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl CudaAtdaBlock32 {
    pub fn new(
        matrix: &PackedSignedUnitMatrix,
        diagonal: &[u32],
        device: i32,
    ) -> Result<Self, CudaError> {
        if diagonal.len() != matrix.rows() as usize {
            return Err(rust_error(
                "create",
                format!(
                    "diagonal requires length {}, got {}",
                    matrix.rows(),
                    diagonal.len()
                ),
            ));
        }
        let nonzeros = u32::try_from(matrix.nonzeros()).map_err(|_| {
            rust_error(
                "create",
                format!("{} nonzeros do not fit the CUDA ABI", matrix.nonzeros()),
            )
        })?;
        let abi = unsafe { adynkra_exact_cuda_abi_version() };
        if abi != CUDA_ABI_VERSION {
            return Err(rust_error(
                "create",
                format!("CUDA ABI version {abi}, expected {CUDA_ABI_VERSION}"),
            ));
        }

        let mut raw = std::ptr::null_mut();
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_create(
                device,
                matrix.rows(),
                matrix.columns(),
                nonzeros,
                matrix.csr_offsets().as_ptr(),
                matrix.csr_entries().as_ptr(),
                matrix.transpose_offsets().as_ptr(),
                matrix.transpose_entries().as_ptr(),
                diagonal.as_ptr(),
                &mut raw,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("create", code, &message)?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| rust_error("create", "backend returned a null operator".to_owned()))?;
        Ok(Self {
            raw,
            rows: matrix.rows(),
            columns: matrix.columns(),
            nonzeros: matrix.nonzeros(),
            device,
            semantic_digest: matrix.semantic_digest(),
            _not_send_or_sync: PhantomData,
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }

    pub fn nonzeros(&self) -> usize {
        self.nonzeros
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn semantic_digest(&self) -> MatrixSemanticDigest {
        self.semantic_digest
    }

    pub fn device_name(&self) -> Result<String, CudaError> {
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_device_name(self.raw.as_ptr(), message.as_mut_ptr(), message.len())
        };
        check("device_name", code, &message)?;
        Ok(message_string(&message))
    }

    /// Upload a canonical coordinate-major block and reset the active chain to
    /// that block. The backend intentionally avoids an O(n) validation scan.
    pub fn upload(&mut self, input: &[u32]) -> Result<(), CudaError> {
        expect_block_length("upload", self.columns, input.len())?;
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_upload(
                self.raw.as_ptr(),
                input.as_ptr(),
                input.len(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("upload", code, &message)
    }

    /// Apply the resident `A^T D A` operator repeatedly without intermediate
    /// host transfers. A zero-step call leaves the active block unchanged.
    pub fn apply_steps(&mut self, steps: u32) -> Result<(), CudaError> {
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_apply_steps(
                self.raw.as_ptr(),
                steps,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("apply_steps", code, &message)
    }

    pub fn download(&self, output: &mut [u32]) -> Result<(), CudaError> {
        expect_block_length("download", self.columns, output.len())?;
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_download(
                self.raw.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("download", code, &message)
    }

    pub fn apply(&mut self, input: &[u32], output: &mut [u32]) -> Result<(), CudaError> {
        self.upload(input)?;
        self.apply_steps(1)?;
        self.download(output)
    }

    pub fn apply_chain(
        &mut self,
        input: &[u32],
        steps: u32,
        output: &mut [u32],
    ) -> Result<(), CudaError> {
        self.upload(input)?;
        self.apply_steps(steps)?;
        self.download(output)
    }

    /// Initialize 32 independent exact CG recurrences for
    /// `(A^T D A + u_j u_j^T) x_j = u_j`.
    ///
    /// `initial_rr[j]` must equal `u_j^T u_j`. The CUDA boundary recomputes all
    /// 32 values from `border` before accepting the state.
    pub fn cg_initialize(
        &mut self,
        border: &[u32],
        initial_rr: &[u32; BLOCK_WIDTH],
    ) -> Result<(), CudaError> {
        expect_block_length("cg_initialize", self.columns, border.len())?;
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_cg_initialize(
                self.raw.as_ptr(),
                border.as_ptr(),
                border.len(),
                initial_rr.as_ptr(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("cg_initialize", code, &message)
    }

    /// Run at most `rounds` more recurrence rounds. The backend rejects work
    /// beyond the column dimension, which is also the maximum rank-proof span.
    pub fn cg_run(&mut self, rounds: u32) -> Result<CudaCgProgress, CudaError> {
        let mut total_rounds = 0_u64;
        let mut raw_status = [0_u32; BLOCK_WIDTH];
        let mut lane_steps = [0_u32; BLOCK_WIDTH];
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_cg_run(
                self.raw.as_ptr(),
                rounds,
                &mut total_rounds,
                raw_status.as_mut_ptr(),
                lane_steps.as_mut_ptr(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("cg_run", code, &message)?;
        Ok(CudaCgProgress {
            total_rounds,
            status: decode_statuses(raw_status)?,
            lane_steps,
        })
    }

    pub fn cg_download_state(&self) -> Result<CudaCgState, CudaError> {
        let entries = block_entries(self.columns, "cg_download_state")?;
        let mut state = CudaCgState {
            x: vec![0; entries],
            r: vec![0; entries],
            p: vec![0; entries],
            rr: [0; BLOCK_WIDTH],
            status: [CudaCgLaneStatus::Active; BLOCK_WIDTH],
            lane_steps: [0; BLOCK_WIDTH],
            transcript: [0; BLOCK_WIDTH],
            total_rounds: 0,
        };
        let mut raw_status = [0_u32; BLOCK_WIDTH];
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_cg_download_state(
                self.raw.as_ptr(),
                state.x.as_mut_ptr(),
                state.r.as_mut_ptr(),
                state.p.as_mut_ptr(),
                entries,
                state.rr.as_mut_ptr(),
                raw_status.as_mut_ptr(),
                state.lane_steps.as_mut_ptr(),
                state.transcript.as_mut_ptr(),
                &mut state.total_rounds,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("cg_download_state", code, &message)?;
        state.status = decode_statuses(raw_status)?;
        Ok(state)
    }

    /// Restore recurrence vectors. A restored state can recover a kernel, but
    /// its prior direction count is not standalone rank-proof evidence unless
    /// the deterministic transcript is replayed from its original seed.
    pub fn cg_upload_state(
        &mut self,
        border: &[u32],
        state: &CudaCgState,
    ) -> Result<(), CudaError> {
        let expected = block_entries(self.columns, "cg_upload_state")?;
        expect_exact_length("cg_upload_state x", expected, state.x.len())?;
        expect_exact_length("cg_upload_state r", expected, state.r.len())?;
        expect_exact_length("cg_upload_state p", expected, state.p.len())?;
        expect_exact_length("cg_upload_state border", expected, border.len())?;
        let raw_status = state.status.map(|status| status as u32);
        let mut message = error_buffer();
        let code = unsafe {
            adynkra_exact_cuda_cg_upload_state(
                self.raw.as_ptr(),
                border.as_ptr(),
                state.x.as_ptr(),
                state.r.as_ptr(),
                state.p.as_ptr(),
                expected,
                state.rr.as_ptr(),
                raw_status.as_ptr(),
                state.lane_steps.as_ptr(),
                state.transcript.as_ptr(),
                state.total_rounds,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check("cg_upload_state", code, &message)
    }
}

impl Debug for CudaAtdaBlock32 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaAtdaBlock32")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("nonzeros", &self.nonzeros)
            .field("device", &self.device)
            .field("semantic_digest", &self.semantic_digest)
            .finish_non_exhaustive()
    }
}

impl Drop for CudaAtdaBlock32 {
    fn drop(&mut self) {
        unsafe { adynkra_exact_cuda_destroy(self.raw.as_ptr()) };
    }
}

fn expect_block_length(
    operation: &'static str,
    columns: u32,
    actual: usize,
) -> Result<(), CudaError> {
    let expected = (columns as usize)
        .checked_mul(BLOCK_WIDTH)
        .ok_or_else(|| rust_error(operation, "block32 length overflow".to_owned()))?;
    if actual == expected {
        Ok(())
    } else {
        Err(rust_error(
            operation,
            format!("block requires length {expected}, got {actual}"),
        ))
    }
}

fn block_entries(columns: u32, operation: &'static str) -> Result<usize, CudaError> {
    (columns as usize)
        .checked_mul(BLOCK_WIDTH)
        .ok_or_else(|| rust_error(operation, "block32 length overflow".to_owned()))
}

fn expect_exact_length(
    operation: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), CudaError> {
    if expected == actual {
        Ok(())
    } else {
        Err(rust_error(
            operation,
            format!("requires length {expected}, got {actual}"),
        ))
    }
}

fn decode_statuses(raw: [u32; BLOCK_WIDTH]) -> Result<[CudaCgLaneStatus; BLOCK_WIDTH], CudaError> {
    let mut result = [CudaCgLaneStatus::Active; BLOCK_WIDTH];
    for (destination, value) in result.iter_mut().zip(raw) {
        *destination = CudaCgLaneStatus::try_from(value)?;
    }
    Ok(result)
}

fn error_buffer() -> Vec<c_char> {
    vec![0; ERROR_CAPACITY]
}

fn message_string(message: &[c_char]) -> String {
    unsafe { CStr::from_ptr(message.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn check(operation: &'static str, code: c_int, message: &[c_char]) -> Result<(), CudaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError {
            operation,
            code,
            message: message_string(message),
        })
    }
}

fn rust_error(operation: &'static str, message: String) -> CudaError {
    CudaError {
        operation,
        code: -10_000,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{BlockWorkspace32, PackedSignedUnitMatrix, pinned_nonzero_diagonal};
    use crate::level12::build_level12_matrix;
    use crate::{CsrMatrix, PRIME, Triplet, field_add, field_mul, field_sub};

    fn sample_matrix() -> CsrMatrix {
        CsrMatrix::from_triplets(
            5,
            7,
            vec![
                Triplet {
                    row: 0,
                    column: 0,
                    coefficient: 1,
                },
                Triplet {
                    row: 0,
                    column: 3,
                    coefficient: -1,
                },
                Triplet {
                    row: 1,
                    column: 2,
                    coefficient: 1,
                },
                Triplet {
                    row: 1,
                    column: 6,
                    coefficient: 1,
                },
                Triplet {
                    row: 2,
                    column: 0,
                    coefficient: -1,
                },
                Triplet {
                    row: 2,
                    column: 4,
                    coefficient: 1,
                },
                Triplet {
                    row: 3,
                    column: 1,
                    coefficient: -1,
                },
                Triplet {
                    row: 3,
                    column: 3,
                    coefficient: 1,
                },
                Triplet {
                    row: 3,
                    column: 5,
                    coefficient: -1,
                },
                Triplet {
                    row: 4,
                    column: 2,
                    coefficient: -1,
                },
                Triplet {
                    row: 4,
                    column: 6,
                    coefficient: 1,
                },
            ],
        )
        .unwrap()
    }

    fn input(columns: u32) -> Vec<u32> {
        (0..columns as usize * BLOCK_WIDTH)
            .map(|index| ((index as u64 * 1_000_000_007 + 97) % u64::from(PRIME)) as u32)
            .collect()
    }

    fn cpu_chain(
        packed: &PackedSignedUnitMatrix,
        diagonal: &[u32],
        input: &[u32],
        steps: u32,
    ) -> Vec<u32> {
        let mut current = input.to_vec();
        let mut next = vec![0; current.len()];
        let mut workspace = BlockWorkspace32::new(packed).unwrap();
        for _ in 0..steps {
            packed
                .apply_atda_block32(diagonal, &current, &mut next, &mut workspace)
                .unwrap();
            std::mem::swap(&mut current, &mut next);
        }
        current
    }

    #[test]
    fn cuda_small_matrix_matches_cpu_and_repeats_exactly() {
        let packed = PackedSignedUnitMatrix::from_csr(&sample_matrix()).unwrap();
        let diagonal = pinned_nonzero_diagonal(packed.rows(), 0x1234_5678_9abc_def0);
        let input = input(packed.columns());
        let reference = cpu_chain(&packed, &diagonal, &input, 4);
        let mut operator = CudaAtdaBlock32::new(&packed, &diagonal, 0).unwrap();
        assert!(!operator.device_name().unwrap().is_empty());

        let mut first = vec![0; input.len()];
        operator.apply_chain(&input, 4, &mut first).unwrap();
        assert_eq!(first, reference);
        let mut second = vec![PRIME - 1; input.len()];
        operator.apply_chain(&input, 4, &mut second).unwrap();
        assert_eq!(second, first);
    }

    fn assert_level12_parity(label: &str, seed: u64) {
        let matrix = build_level12_matrix(label).unwrap();
        let packed = PackedSignedUnitMatrix::from_csr(&matrix.raising).unwrap();
        let diagonal = pinned_nonzero_diagonal(packed.rows(), seed);
        let input = input(packed.columns());
        let reference = cpu_chain(&packed, &diagonal, &input, 1);
        let mut output = vec![0; input.len()];
        let mut operator = CudaAtdaBlock32::new(&packed, &diagonal, 0).unwrap();
        operator.apply(&input, &mut output).unwrap();
        assert_eq!(output, reference);
        let first = output.clone();
        operator.apply(&input, &mut output).unwrap();
        assert_eq!(output, first);
    }

    #[test]
    fn cuda_level12_30002_matches_cpu_reference() {
        assert_level12_parity("30002", 0xfeed_face_cafe_beef);
    }

    #[test]
    fn cuda_level12_01002_matches_cpu_reference() {
        assert_level12_parity("01002", 0x0100_0201_0002_0100);
    }

    fn bordered_diagonal_matrix() -> CsrMatrix {
        CsrMatrix::from_triplets(
            3,
            4,
            vec![
                Triplet {
                    row: 0,
                    column: 1,
                    coefficient: 1,
                },
                Triplet {
                    row: 1,
                    column: 2,
                    coefficient: 1,
                },
                Triplet {
                    row: 2,
                    column: 3,
                    coefficient: 1,
                },
            ],
        )
        .unwrap()
    }

    fn inverse(value: u32) -> u32 {
        let mut result = 1;
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

    fn cpu_cg_step(
        packed: &PackedSignedUnitMatrix,
        diagonal: &[u32],
        border: &[u32],
        state: &mut CudaCgState,
    ) {
        let mut bp = vec![0; state.p.len()];
        let mut workspace = BlockWorkspace32::new(packed).unwrap();
        packed
            .apply_atda_block32(diagonal, &state.p, &mut bp, &mut workspace)
            .unwrap();
        for lane in 0..BLOCK_WIDTH {
            if state.status[lane] != CudaCgLaneStatus::Active {
                continue;
            }
            let mut sigma = 0;
            let mut p_bp = 0;
            for coordinate in 0..packed.columns() as usize {
                let index = coordinate * BLOCK_WIDTH + lane;
                sigma = field_add(sigma, field_mul(border[index], state.p[index]));
                p_bp = field_add(p_bp, field_mul(state.p[index], bp[index]));
            }
            let p_cp = field_add(p_bp, field_mul(sigma, sigma));
            if state.rr[lane] == 0 || p_cp == 0 {
                state.status[lane] = CudaCgLaneStatus::Broken;
                continue;
            }
            let alpha = field_mul(state.rr[lane], inverse(p_cp));
            for coordinate in 0..packed.columns() as usize {
                let index = coordinate * BLOCK_WIDTH + lane;
                let cp = field_add(bp[index], field_mul(border[index], sigma));
                state.x[index] = field_add(state.x[index], field_mul(alpha, state.p[index]));
                state.r[index] = field_sub(state.r[index], field_mul(alpha, cp));
            }
            state.lane_steps[lane] += 1;
            let mut next_rr = 0;
            let mut any_nonzero = false;
            for coordinate in 0..packed.columns() as usize {
                let value = state.r[coordinate * BLOCK_WIDTH + lane];
                next_rr = field_add(next_rr, field_mul(value, value));
                any_nonzero |= value != 0;
            }
            if !any_nonzero {
                state.status[lane] = CudaCgLaneStatus::Converged;
                state.rr[lane] = 0;
            } else if next_rr == 0 {
                state.status[lane] = CudaCgLaneStatus::Broken;
                state.rr[lane] = 0;
            } else {
                let beta = field_mul(next_rr, inverse(state.rr[lane]));
                state.rr[lane] = next_rr;
                for coordinate in 0..packed.columns() as usize {
                    let index = coordinate * BLOCK_WIDTH + lane;
                    state.p[index] = field_add(state.r[index], field_mul(beta, state.p[index]));
                }
            }
        }
        state.total_rounds += 1;
    }

    fn assert_recurrence_equal(left: &CudaCgState, right: &CudaCgState) {
        assert_eq!(left.x, right.x);
        assert_eq!(left.r, right.r);
        assert_eq!(left.p, right.p);
        assert_eq!(left.rr, right.rr);
        assert_eq!(left.status, right.status);
        assert_eq!(left.lane_steps, right.lane_steps);
        assert_eq!(left.total_rounds, right.total_rounds);
    }

    #[test]
    fn cuda_bordered_cg_matches_cpu_breakdown_rank_and_restore() {
        use crate::gpu_krylov::{lane_squared_norms, pinned_border_block};

        let matrix = bordered_diagonal_matrix();
        let packed = PackedSignedUnitMatrix::from_csr(&matrix).unwrap();
        let diagonal = vec![1, 2, 3];
        let mut border = pinned_border_block(packed.columns(), 0x1234).unwrap();
        // Lane 0 is an early one-step kernel solve.
        for coordinate in 0..4 {
            border[coordinate * BLOCK_WIDTH] = u32::from(coordinate == 0);
        }
        // Lane 1 is nonzero but isotropic over the production field.
        for (coordinate, value) in [1, 1, 1, 1_268_011_823].into_iter().enumerate() {
            border[coordinate * BLOCK_WIDTH + 1] = value;
        }
        let rr = lane_squared_norms(&border).unwrap();
        assert_eq!(rr[1], 0);

        let mut gpu = CudaAtdaBlock32::new(&packed, &diagonal, 0).unwrap();
        let mut wrong_rr = rr;
        wrong_rr[0] = field_add(wrong_rr[0], 1);
        assert!(gpu.cg_initialize(&border, &wrong_rr).is_err());
        gpu.cg_initialize(&border, &rr).unwrap();
        let initial = gpu.cg_run(0).unwrap();
        assert_eq!(initial.status[1], CudaCgLaneStatus::Broken);

        let entries = border.len();
        let mut reference = CudaCgState {
            x: vec![0; entries],
            r: border.clone(),
            p: border.clone(),
            rr,
            status: initial.status,
            lane_steps: [0; BLOCK_WIDTH],
            transcript: [0; BLOCK_WIDTH],
            total_rounds: 0,
        };
        gpu.cg_run(1).unwrap();
        cpu_cg_step(&packed, &diagonal, &border, &mut reference);
        let after_one = gpu.cg_download_state().unwrap();
        assert_recurrence_equal(&after_one, &reference);
        assert_eq!(after_one.status[0], CudaCgLaneStatus::Converged);
        assert_eq!(after_one.lane_steps[0], 1);

        gpu.cg_run(1).unwrap();
        cpu_cg_step(&packed, &diagonal, &border, &mut reference);
        let checkpoint = gpu.cg_download_state().unwrap();
        assert_recurrence_equal(&checkpoint, &reference);
        gpu.cg_run(2).unwrap();
        cpu_cg_step(&packed, &diagonal, &border, &mut reference);
        cpu_cg_step(&packed, &diagonal, &border, &mut reference);
        let final_state = gpu.cg_download_state().unwrap();
        assert_recurrence_equal(&final_state, &reference);
        assert!(
            final_state
                .status
                .iter()
                .zip(final_state.lane_steps)
                .any(|(status, steps)| *status == CudaCgLaneStatus::Converged && steps == 4)
        );
        assert!(gpu.cg_run(1).is_err());

        let mut restored = CudaAtdaBlock32::new(&packed, &diagonal, 0).unwrap();
        restored.cg_upload_state(&border, &checkpoint).unwrap();
        restored.cg_run(2).unwrap();
        let restored_final = restored.cg_download_state().unwrap();
        assert_eq!(restored_final, final_state);
    }
}
