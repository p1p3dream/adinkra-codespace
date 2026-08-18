//! Accelerator-ready signed-unit sparse operator boundary.
//!
//! Entries pack an index in bits 0 through 30 and a negative sign in bit 31.
//! The CPU operator is the deterministic reference for an eventual accelerator
//! implementation. Its hot `A^T D A` block loop allocates nothing and assumes
//! canonical field inputs produced by the solver.

use crate::{CsrMatrix, PRIME, field_add, field_mul, field_sub};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const SIGN_BIT: u32 = 1 << 31;
pub const INDEX_MASK: u32 = SIGN_BIT - 1;
pub const BLOCK_WIDTH: usize = 32;
pub const SEMANTIC_DIGEST_VERSION: u32 = 1;
pub const DIAGONAL_PRNG_VERSION: u32 = 1;
pub const DEFAULT_DIAGONAL_SEED: u64 = 0x6164_796e_6b72_6131;
const SEMANTIC_DIGEST_DOMAIN: &[u8] = b"adynkra-packed-signed-unit-matrix\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AcceleratorError {
    IndexUsesSignBit {
        axis: &'static str,
        value: u32,
    },
    NonUnitCoefficient {
        entry: usize,
        coefficient: i32,
    },
    DimensionMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    SizeOverflow,
}

impl Display for AcceleratorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexUsesSignBit { axis, value } => {
                write!(
                    formatter,
                    "{axis} dimension {value} uses the packed sign bit"
                )
            }
            Self::NonUnitCoefficient { entry, coefficient } => write!(
                formatter,
                "matrix entry {entry} has coefficient {coefficient}, expected +1 or -1"
            ),
            Self::DimensionMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                formatter,
                "{buffer} requires length {expected}, got {actual}"
            ),
            Self::SizeOverflow => write!(formatter, "block workspace size overflows usize"),
        }
    }
}

impl Error for AcceleratorError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixSemanticDigest {
    pub version: u32,
    pub sha256: [u8; 32],
}

impl MatrixSemanticDigest {
    pub fn sha256_hex(&self) -> String {
        self.sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

impl Display for MatrixSemanticDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "v{}:{}", self.version, self.sha256_hex())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedSignedUnitMatrix {
    rows: u32,
    columns: u32,
    csr_offsets: Vec<u32>,
    /// Packed column index and sign, in canonical CSR order.
    csr_entries: Vec<u32>,
    transpose_offsets: Vec<u32>,
    /// Packed row index and sign, in canonical CSC order.
    transpose_entries: Vec<u32>,
    semantic_digest: MatrixSemanticDigest,
}

impl PackedSignedUnitMatrix {
    pub fn from_csr(matrix: &CsrMatrix) -> Result<Self, AcceleratorError> {
        if matrix.rows() > SIGN_BIT {
            return Err(AcceleratorError::IndexUsesSignBit {
                axis: "row",
                value: matrix.rows(),
            });
        }
        if matrix.columns() > SIGN_BIT {
            return Err(AcceleratorError::IndexUsesSignBit {
                axis: "column",
                value: matrix.columns(),
            });
        }

        let mut csr_entries = Vec::with_capacity(matrix.nonzeros());
        for (entry, (&column, &coefficient)) in matrix
            .column_indices()
            .iter()
            .zip(matrix.coefficients())
            .enumerate()
        {
            csr_entries.push(pack_entry(column, coefficient, entry)?);
        }

        let mut transpose_offsets = vec![0_u32; matrix.columns() as usize + 1];
        for &packed in &csr_entries {
            transpose_offsets[unpack_index(packed) as usize + 1] += 1;
        }
        for column in 0..matrix.columns() as usize {
            transpose_offsets[column + 1] += transpose_offsets[column];
        }
        let mut next = transpose_offsets[..matrix.columns() as usize].to_vec();
        let mut transpose_entries = vec![0_u32; matrix.nonzeros()];
        for row in 0..matrix.rows() as usize {
            let start = matrix.row_offsets()[row] as usize;
            let end = matrix.row_offsets()[row + 1] as usize;
            for (entry, &packed_column) in csr_entries.iter().enumerate().take(end).skip(start) {
                let column = unpack_index(packed_column) as usize;
                let destination = next[column] as usize;
                let coefficient = matrix.coefficients()[entry];
                transpose_entries[destination] = pack_entry(row as u32, coefficient, entry)?;
                next[column] += 1;
            }
        }

        let semantic_digest = semantic_digest(
            matrix.rows(),
            matrix.columns(),
            matrix.row_offsets(),
            &csr_entries,
        );
        Ok(Self {
            rows: matrix.rows(),
            columns: matrix.columns(),
            csr_offsets: matrix.row_offsets().to_vec(),
            csr_entries,
            transpose_offsets,
            transpose_entries,
            semantic_digest,
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }

    pub fn nonzeros(&self) -> usize {
        self.csr_entries.len()
    }

    pub fn csr_offsets(&self) -> &[u32] {
        &self.csr_offsets
    }

    pub fn csr_entries(&self) -> &[u32] {
        &self.csr_entries
    }

    pub fn transpose_offsets(&self) -> &[u32] {
        &self.transpose_offsets
    }

    pub fn transpose_entries(&self) -> &[u32] {
        &self.transpose_entries
    }

    pub fn semantic_digest(&self) -> MatrixSemanticDigest {
        self.semantic_digest
    }

    /// Apply `A^T D A` to exactly 32 coordinate-major lanes.
    ///
    /// Lengths and workspace ownership are checked in constant time before the
    /// hot loop. Field elements and diagonal entries must already be canonical;
    /// this boundary intentionally performs no validation scans.
    pub fn apply_atda_block32(
        &self,
        diagonal: &[u32],
        input: &[u32],
        output: &mut [u32],
        workspace: &mut BlockWorkspace32,
    ) -> Result<(), AcceleratorError> {
        expect_length("diagonal", diagonal.len(), self.rows as usize)?;
        expect_length("input block", input.len(), block_len(self.columns)?)?;
        expect_length("output block", output.len(), block_len(self.columns)?)?;
        expect_length(
            "workspace row block",
            workspace.row_block.len(),
            block_len(self.rows)?,
        )?;

        self.apply_atda_block32_hot(diagonal, input, output, workspace);
        Ok(())
    }

    #[inline]
    fn apply_atda_block32_hot(
        &self,
        diagonal: &[u32],
        input: &[u32],
        output: &mut [u32],
        workspace: &mut BlockWorkspace32,
    ) {
        let row_block = &mut workspace.row_block;

        // Fused D(A X): each row stays resident while all signed-unit entries
        // and all 32 lanes are accumulated, then the row diagonal is applied.
        for (row, &scale) in diagonal.iter().enumerate() {
            let row_base = row * BLOCK_WIDTH;
            let mut accumulator = [0_u32; BLOCK_WIDTH];
            let start = self.csr_offsets[row] as usize;
            let end = self.csr_offsets[row + 1] as usize;
            for &packed in &self.csr_entries[start..end] {
                let input_base = unpack_index(packed) as usize * BLOCK_WIDTH;
                if is_negative(packed) {
                    for lane in 0..BLOCK_WIDTH {
                        accumulator[lane] = field_sub(accumulator[lane], input[input_base + lane]);
                    }
                } else {
                    for lane in 0..BLOCK_WIDTH {
                        accumulator[lane] = field_add(accumulator[lane], input[input_base + lane]);
                    }
                }
            }
            for lane in 0..BLOCK_WIDTH {
                row_block[row_base + lane] = field_mul(scale, accumulator[lane]);
            }
        }

        for column in 0..self.columns as usize {
            let output_base = column * BLOCK_WIDTH;
            let mut accumulator = [0_u32; BLOCK_WIDTH];
            let start = self.transpose_offsets[column] as usize;
            let end = self.transpose_offsets[column + 1] as usize;
            for &packed in &self.transpose_entries[start..end] {
                let row_base = unpack_index(packed) as usize * BLOCK_WIDTH;
                if is_negative(packed) {
                    for lane in 0..BLOCK_WIDTH {
                        accumulator[lane] =
                            field_sub(accumulator[lane], row_block[row_base + lane]);
                    }
                } else {
                    for lane in 0..BLOCK_WIDTH {
                        accumulator[lane] =
                            field_add(accumulator[lane], row_block[row_base + lane]);
                    }
                }
            }
            output[output_base..output_base + BLOCK_WIDTH].copy_from_slice(&accumulator);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockWorkspace32 {
    row_block: Vec<u32>,
}

impl BlockWorkspace32 {
    pub fn new(matrix: &PackedSignedUnitMatrix) -> Result<Self, AcceleratorError> {
        Ok(Self {
            row_block: vec![0; block_len(matrix.rows)?],
        })
    }

    pub fn row_capacity(&self) -> usize {
        self.row_block.len() / BLOCK_WIDTH
    }
}

/// Fill an existing buffer with the pinned SplitMix64 v1 diagonal sequence.
/// Every result is in `1..PRIME`, so `D` is nonsingular over the field.
pub fn fill_pinned_nonzero_diagonal(output: &mut [u32], seed: u64) {
    let mut state = seed;
    for value in output {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        *value = (mixed % u64::from(PRIME - 1) + 1) as u32;
    }
}

pub fn pinned_nonzero_diagonal(rows: u32, seed: u64) -> Vec<u32> {
    let mut diagonal = vec![0; rows as usize];
    fill_pinned_nonzero_diagonal(&mut diagonal, seed);
    diagonal
}

#[inline]
pub fn unpack_index(packed: u32) -> u32 {
    packed & INDEX_MASK
}

#[inline]
pub fn is_negative(packed: u32) -> bool {
    packed & SIGN_BIT != 0
}

fn pack_entry(index: u32, coefficient: i32, entry: usize) -> Result<u32, AcceleratorError> {
    debug_assert!(index <= INDEX_MASK);
    match coefficient {
        1 => Ok(index),
        -1 => Ok(index | SIGN_BIT),
        _ => Err(AcceleratorError::NonUnitCoefficient { entry, coefficient }),
    }
}

fn semantic_digest(
    rows: u32,
    columns: u32,
    row_offsets: &[u32],
    entries: &[u32],
) -> MatrixSemanticDigest {
    let mut hasher = Sha256::new();
    hasher.update(SEMANTIC_DIGEST_DOMAIN);
    hasher.update(SEMANTIC_DIGEST_VERSION.to_le_bytes());
    hasher.update(PRIME.to_le_bytes());
    hasher.update(rows.to_le_bytes());
    hasher.update(columns.to_le_bytes());
    hasher.update((entries.len() as u64).to_le_bytes());
    for row in 0..rows as usize {
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        for &packed in &entries[start..end] {
            hasher.update((row as u32).to_le_bytes());
            hasher.update(unpack_index(packed).to_le_bytes());
            hasher.update([if is_negative(packed) { 0xff } else { 0x01 }]);
        }
    }
    MatrixSemanticDigest {
        version: SEMANTIC_DIGEST_VERSION,
        sha256: hasher.finalize().into(),
    }
}

fn block_len(coordinates: u32) -> Result<usize, AcceleratorError> {
    (coordinates as usize)
        .checked_mul(BLOCK_WIDTH)
        .ok_or(AcceleratorError::SizeOverflow)
}

fn expect_length(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), AcceleratorError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AcceleratorError::DimensionMismatch {
            buffer,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CscMatrix, Triplet};

    fn sample() -> CsrMatrix {
        // [ 1  0 -1  0 ]
        // [ 0 -1  1  1 ]
        // [-1  0  0  1 ]
        CsrMatrix::from_triplets(
            3,
            4,
            vec![
                Triplet {
                    row: 2,
                    column: 3,
                    coefficient: 1,
                },
                Triplet {
                    row: 0,
                    column: 2,
                    coefficient: -1,
                },
                Triplet {
                    row: 1,
                    column: 3,
                    coefficient: 1,
                },
                Triplet {
                    row: 2,
                    column: 0,
                    coefficient: -1,
                },
                Triplet {
                    row: 0,
                    column: 0,
                    coefficient: 1,
                },
                Triplet {
                    row: 1,
                    column: 1,
                    coefficient: -1,
                },
                Triplet {
                    row: 1,
                    column: 2,
                    coefficient: 1,
                },
            ],
        )
        .unwrap()
    }

    fn scalar_reference(
        csr: &CsrMatrix,
        csc: &CscMatrix,
        diagonal: &[u32],
        input: &[u32],
    ) -> Vec<u32> {
        let mut row = csr.spmv(input).unwrap();
        for (value, &scale) in row.iter_mut().zip(diagonal) {
            *value = field_mul(*value, scale);
        }
        csc.transpose_spmv(&row).unwrap()
    }

    #[test]
    fn packed_layout_and_transpose_retain_indices_and_signs() {
        let packed = PackedSignedUnitMatrix::from_csr(&sample()).unwrap();
        assert_eq!(packed.csr_offsets(), &[0, 2, 5, 7]);
        assert_eq!(
            packed.csr_entries(),
            &[0, SIGN_BIT | 2, SIGN_BIT | 1, 2, 3, SIGN_BIT, 3]
        );
        assert_eq!(packed.transpose_offsets(), &[0, 2, 3, 5, 7]);
        assert_eq!(
            packed.transpose_entries(),
            &[0, SIGN_BIT | 2, SIGN_BIT | 1, SIGN_BIT, 1, 1, 2]
        );
    }

    #[test]
    fn atda_block_lanes_match_csr_and_csc_scalar_reference() {
        let csr = sample();
        let csc = csr.to_csc();
        let packed = PackedSignedUnitMatrix::from_csr(&csr).unwrap();
        let diagonal = pinned_nonzero_diagonal(packed.rows(), 0x1234_5678);
        let mut input = vec![0; packed.columns() as usize * BLOCK_WIDTH];
        for column in 0..packed.columns() as usize {
            for lane in 0..BLOCK_WIDTH {
                input[column * BLOCK_WIDTH + lane] =
                    ((column as u64 * 1_000_000_007 + lane as u64 * 998_244_353 + 17)
                        % u64::from(PRIME)) as u32;
            }
        }
        // The operator overwrites every output lane and must not depend on the
        // caller's previous Krylov block.
        let mut output = vec![PRIME - 1; input.len()];
        let mut workspace = BlockWorkspace32::new(&packed).unwrap();
        let workspace_pointer = workspace.row_block.as_ptr();
        packed
            .apply_atda_block32(&diagonal, &input, &mut output, &mut workspace)
            .unwrap();
        assert_eq!(workspace.row_block.as_ptr(), workspace_pointer);

        for lane in 0..BLOCK_WIDTH {
            let scalar_input: Vec<_> = (0..packed.columns() as usize)
                .map(|column| input[column * BLOCK_WIDTH + lane])
                .collect();
            let reference = scalar_reference(&csr, &csc, &diagonal, &scalar_input);
            for column in 0..packed.columns() as usize {
                assert_eq!(output[column * BLOCK_WIDTH + lane], reference[column]);
            }
        }

        let first = output.clone();
        packed
            .apply_atda_block32(&diagonal, &input, &mut output, &mut workspace)
            .unwrap();
        assert_eq!(output, first);
        assert_eq!(workspace.row_block.as_ptr(), workspace_pointer);
    }

    #[test]
    fn semantic_digest_is_pinned_and_triplet_order_independent() {
        let first = sample();
        let mut triplets = vec![
            Triplet {
                row: 0,
                column: 0,
                coefficient: 1,
            },
            Triplet {
                row: 0,
                column: 2,
                coefficient: -1,
            },
            Triplet {
                row: 1,
                column: 1,
                coefficient: -1,
            },
            Triplet {
                row: 1,
                column: 2,
                coefficient: 1,
            },
            Triplet {
                row: 1,
                column: 3,
                coefficient: 1,
            },
            Triplet {
                row: 2,
                column: 0,
                coefficient: -1,
            },
            Triplet {
                row: 2,
                column: 3,
                coefficient: 1,
            },
        ];
        triplets.reverse();
        let second = CsrMatrix::from_triplets(3, 4, triplets).unwrap();
        let first_digest = PackedSignedUnitMatrix::from_csr(&first)
            .unwrap()
            .semantic_digest();
        let second_digest = PackedSignedUnitMatrix::from_csr(&second)
            .unwrap()
            .semantic_digest();
        assert_eq!(first_digest, second_digest);
        assert_eq!(first_digest.version, 1);
        assert_eq!(
            first_digest.sha256_hex(),
            "3dba2d66b1c1fb2e0a8f071483078a439c1b9e56dfef468e9ba27f283b67dc04"
        );
    }

    #[test]
    fn pinned_diagonal_sequence_is_nonzero_and_stable() {
        let diagonal = pinned_nonzero_diagonal(8, 0);
        assert_eq!(
            diagonal,
            [
                60_845_732,
                1_536_941_989,
                454_736_306,
                1_417_372_829,
                1_049_532_998,
                1_135_419_169,
                1_364_172_252,
                1_596_389_427,
            ]
        );
        assert!(diagonal.iter().all(|&value| value > 0 && value < PRIME));
    }

    #[test]
    fn rejects_non_unit_coefficients() {
        let csr = CsrMatrix::from_triplets(
            1,
            1,
            vec![Triplet {
                row: 0,
                column: 0,
                coefficient: 2,
            }],
        )
        .unwrap();
        assert!(matches!(
            PackedSignedUnitMatrix::from_csr(&csr),
            Err(AcceleratorError::NonUnitCoefficient { coefficient: 2, .. })
        ));
    }
}
