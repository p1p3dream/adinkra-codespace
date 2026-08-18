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
pub const CUDA_ABI_VERSION: u32 = 1;

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
    use crate::{CsrMatrix, PRIME, Triplet};

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
}
