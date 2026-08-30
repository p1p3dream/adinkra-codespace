//! Compact deterministic sparse matrix operations over `GF(2^31 - 1)`.
//!
//! The matrix retains signed integer coefficients. Field vectors use canonical `u32`
//! residues in `[0, PRIME)`. Block vectors use coordinate-major layout: the lanes for
//! coordinate `i` occupy `i * block_width..(i + 1) * block_width`.

pub mod accelerator;
pub mod block32;
pub mod certificate;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod elimination;
#[cfg(feature = "cuda")]
pub mod gpu_krylov;
pub mod level12;
pub mod publish;

use std::error::Error;
use std::fmt::{Display, Formatter};

/// The largest signed 32-bit prime, and a Mersenne prime.
pub const PRIME: u32 = 2_147_483_647;
const PRIME_U64: u64 = PRIME as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Triplet {
    pub row: u32,
    pub column: u32,
    pub coefficient: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SparseMatrixError {
    CoordinateOutOfBounds {
        row: u32,
        column: u32,
        rows: u32,
        columns: u32,
    },
    TooManyNonzeros(usize),
    CoefficientOverflow {
        row: u32,
        column: u32,
    },
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonCanonicalFieldElement {
        index: usize,
        value: u32,
    },
    ZeroBlockWidth,
    SizeOverflow,
}

impl Display for SparseMatrixError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoordinateOutOfBounds {
                row,
                column,
                rows,
                columns,
            } => write!(
                formatter,
                "matrix coordinate ({row}, {column}) is outside {rows}x{columns} dimensions"
            ),
            Self::TooManyNonzeros(count) => {
                write!(formatter, "{count} nonzeros do not fit in 32-bit offsets")
            }
            Self::CoefficientOverflow { row, column } => write!(
                formatter,
                "combined coefficient at ({row}, {column}) does not fit in i32"
            ),
            Self::DimensionMismatch { expected, actual } => {
                write!(formatter, "expected vector length {expected}, got {actual}")
            }
            Self::NonCanonicalFieldElement { index, value } => write!(
                formatter,
                "field element at index {index} is not canonical: {value} >= {PRIME}"
            ),
            Self::ZeroBlockWidth => write!(formatter, "block width must be nonzero"),
            Self::SizeOverflow => write!(formatter, "requested vector size overflows usize"),
        }
    }
}

impl Error for SparseMatrixError {}

/// Canonicalize a signed integer as an element of `GF(PRIME)`.
#[inline]
pub fn field_from_i64(value: i64) -> u32 {
    value.rem_euclid(i64::from(PRIME)) as u32
}

#[inline]
pub fn field_add(left: u32, right: u32) -> u32 {
    debug_assert!(left < PRIME && right < PRIME);
    let sum = u64::from(left) + u64::from(right);
    if sum >= PRIME_U64 {
        (sum - PRIME_U64) as u32
    } else {
        sum as u32
    }
}

#[inline]
pub fn field_sub(left: u32, right: u32) -> u32 {
    debug_assert!(left < PRIME && right < PRIME);
    if left >= right {
        left - right
    } else {
        (u64::from(left) + PRIME_U64 - u64::from(right)) as u32
    }
}

/// Multiply two canonical residues using the `2^31 - 1` Mersenne reduction.
#[inline]
pub fn field_mul(left: u32, right: u32) -> u32 {
    debug_assert!(left < PRIME && right < PRIME);
    let product = u64::from(left) * u64::from(right);
    let first_fold = (product & PRIME_U64) + (product >> 31);
    let second_fold = (first_fold & PRIME_U64) + (first_fold >> 31);
    if second_fold >= PRIME_U64 {
        (second_fold - PRIME_U64) as u32
    } else {
        second_fold as u32
    }
}

#[inline]
fn field_mul_signed(coefficient: i32, value: u32) -> u32 {
    match coefficient {
        0 => 0,
        1 => value,
        -1 => {
            if value == 0 {
                0
            } else {
                PRIME - value
            }
        }
        _ => field_mul(field_from_i64(i64::from(coefficient)), value),
    }
}

fn validate_field_vector(vector: &[u32]) -> Result<(), SparseMatrixError> {
    if let Some((index, &value)) = vector
        .iter()
        .enumerate()
        .find(|(_, value)| **value >= PRIME)
    {
        return Err(SparseMatrixError::NonCanonicalFieldElement { index, value });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix {
    rows: u32,
    columns: u32,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
    coefficients: Vec<i32>,
}

impl CsrMatrix {
    /// Construct canonical CSR, sorting coordinates, combining duplicates over the
    /// integers, and dropping zero coefficients.
    pub fn from_triplets(
        rows: u32,
        columns: u32,
        mut triplets: Vec<Triplet>,
    ) -> Result<Self, SparseMatrixError> {
        for entry in &triplets {
            if entry.row >= rows || entry.column >= columns {
                return Err(SparseMatrixError::CoordinateOutOfBounds {
                    row: entry.row,
                    column: entry.column,
                    rows,
                    columns,
                });
            }
        }
        triplets.sort_unstable_by_key(|entry| (entry.row, entry.column));

        let mut combined = Vec::<Triplet>::with_capacity(triplets.len());
        let mut index = 0;
        while index < triplets.len() {
            let row = triplets[index].row;
            let column = triplets[index].column;
            let mut coefficient = 0_i64;
            while index < triplets.len()
                && triplets[index].row == row
                && triplets[index].column == column
            {
                coefficient = coefficient
                    .checked_add(i64::from(triplets[index].coefficient))
                    .ok_or(SparseMatrixError::CoefficientOverflow { row, column })?;
                index += 1;
            }
            if coefficient != 0 {
                let coefficient = i32::try_from(coefficient)
                    .map_err(|_| SparseMatrixError::CoefficientOverflow { row, column })?;
                combined.push(Triplet {
                    row,
                    column,
                    coefficient,
                });
            }
        }
        if combined.len() > u32::MAX as usize {
            return Err(SparseMatrixError::TooManyNonzeros(combined.len()));
        }

        let mut row_offsets = vec![0_u32; rows as usize + 1];
        for entry in &combined {
            row_offsets[entry.row as usize + 1] += 1;
        }
        for row in 0..rows as usize {
            row_offsets[row + 1] += row_offsets[row];
        }
        let column_indices = combined.iter().map(|entry| entry.column).collect();
        let coefficients = combined.iter().map(|entry| entry.coefficient).collect();
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            coefficients,
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }

    pub fn nonzeros(&self) -> usize {
        self.coefficients.len()
    }

    pub fn row_offsets(&self) -> &[u32] {
        &self.row_offsets
    }

    pub fn column_indices(&self) -> &[u32] {
        &self.column_indices
    }

    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    /// Build a deterministic CSC index of the same matrix. Within each column,
    /// entries retain ascending row order because canonical CSR is scanned in order.
    pub fn to_csc(&self) -> CscMatrix {
        let mut column_offsets = vec![0_u32; self.columns as usize + 1];
        for &column in &self.column_indices {
            column_offsets[column as usize + 1] += 1;
        }
        for column in 0..self.columns as usize {
            column_offsets[column + 1] += column_offsets[column];
        }

        let mut next = column_offsets[..self.columns as usize].to_vec();
        let mut row_indices = vec![0_u32; self.nonzeros()];
        let mut coefficients = vec![0_i32; self.nonzeros()];
        for row in 0..self.rows as usize {
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            for index in start..end {
                let column = self.column_indices[index] as usize;
                let destination = next[column] as usize;
                row_indices[destination] = row as u32;
                coefficients[destination] = self.coefficients[index];
                next[column] += 1;
            }
        }
        CscMatrix {
            rows: self.rows,
            columns: self.columns,
            column_offsets,
            row_indices,
            coefficients,
        }
    }

    pub fn spmv(&self, input: &[u32]) -> Result<Vec<u32>, SparseMatrixError> {
        let mut output = vec![0; self.rows as usize];
        self.spmv_into(input, &mut output)?;
        Ok(output)
    }

    pub fn spmv_into(&self, input: &[u32], output: &mut [u32]) -> Result<(), SparseMatrixError> {
        expect_length(input.len(), self.columns as usize)?;
        expect_length(output.len(), self.rows as usize)?;
        validate_field_vector(input)?;
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut accumulator = 0;
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            for index in start..end {
                let value = input[self.column_indices[index] as usize];
                accumulator = field_add(
                    accumulator,
                    field_mul_signed(self.coefficients[index], value),
                );
            }
            *output_value = accumulator;
        }
        Ok(())
    }

    pub fn spmm(&self, block_width: usize, input: &[u32]) -> Result<Vec<u32>, SparseMatrixError> {
        let output_len = block_len(self.rows, block_width)?;
        let mut output = vec![0; output_len];
        self.spmm_into(block_width, input, &mut output)?;
        Ok(output)
    }

    /// Multiply a coordinate-major dense block by the sparse matrix.
    pub fn spmm_into(
        &self,
        block_width: usize,
        input: &[u32],
        output: &mut [u32],
    ) -> Result<(), SparseMatrixError> {
        let input_len = block_len(self.columns, block_width)?;
        let output_len = block_len(self.rows, block_width)?;
        expect_length(input.len(), input_len)?;
        expect_length(output.len(), output_len)?;
        validate_field_vector(input)?;
        output.fill(0);
        for row in 0..self.rows as usize {
            let row_output = &mut output[row * block_width..(row + 1) * block_width];
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            for index in start..end {
                let column = self.column_indices[index] as usize;
                let column_input = &input[column * block_width..(column + 1) * block_width];
                let coefficient = self.coefficients[index];
                for lane in 0..block_width {
                    row_output[lane] = field_add(
                        row_output[lane],
                        field_mul_signed(coefficient, column_input[lane]),
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CscMatrix {
    rows: u32,
    columns: u32,
    column_offsets: Vec<u32>,
    row_indices: Vec<u32>,
    coefficients: Vec<i32>,
}

impl CscMatrix {
    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }

    pub fn nonzeros(&self) -> usize {
        self.coefficients.len()
    }

    pub fn column_offsets(&self) -> &[u32] {
        &self.column_offsets
    }

    pub fn row_indices(&self) -> &[u32] {
        &self.row_indices
    }

    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    pub fn transpose_spmv(&self, input: &[u32]) -> Result<Vec<u32>, SparseMatrixError> {
        let mut output = vec![0; self.columns as usize];
        self.transpose_spmv_into(input, &mut output)?;
        Ok(output)
    }

    pub fn transpose_spmv_into(
        &self,
        input: &[u32],
        output: &mut [u32],
    ) -> Result<(), SparseMatrixError> {
        expect_length(input.len(), self.rows as usize)?;
        expect_length(output.len(), self.columns as usize)?;
        validate_field_vector(input)?;
        for (column, output_value) in output.iter_mut().enumerate() {
            let mut accumulator = 0;
            let start = self.column_offsets[column] as usize;
            let end = self.column_offsets[column + 1] as usize;
            for index in start..end {
                let value = input[self.row_indices[index] as usize];
                accumulator = field_add(
                    accumulator,
                    field_mul_signed(self.coefficients[index], value),
                );
            }
            *output_value = accumulator;
        }
        Ok(())
    }

    pub fn transpose_spmm(
        &self,
        block_width: usize,
        input: &[u32],
    ) -> Result<Vec<u32>, SparseMatrixError> {
        let output_len = block_len(self.columns, block_width)?;
        let mut output = vec![0; output_len];
        self.transpose_spmm_into(block_width, input, &mut output)?;
        Ok(output)
    }

    /// Multiply a coordinate-major dense block by the transpose.
    pub fn transpose_spmm_into(
        &self,
        block_width: usize,
        input: &[u32],
        output: &mut [u32],
    ) -> Result<(), SparseMatrixError> {
        let input_len = block_len(self.rows, block_width)?;
        let output_len = block_len(self.columns, block_width)?;
        expect_length(input.len(), input_len)?;
        expect_length(output.len(), output_len)?;
        validate_field_vector(input)?;
        output.fill(0);
        for column in 0..self.columns as usize {
            let column_output = &mut output[column * block_width..(column + 1) * block_width];
            let start = self.column_offsets[column] as usize;
            let end = self.column_offsets[column + 1] as usize;
            for index in start..end {
                let row = self.row_indices[index] as usize;
                let row_input = &input[row * block_width..(row + 1) * block_width];
                let coefficient = self.coefficients[index];
                for lane in 0..block_width {
                    column_output[lane] = field_add(
                        column_output[lane],
                        field_mul_signed(coefficient, row_input[lane]),
                    );
                }
            }
        }
        Ok(())
    }
}

fn expect_length(actual: usize, expected: usize) -> Result<(), SparseMatrixError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SparseMatrixError::DimensionMismatch { expected, actual })
    }
}

fn block_len(coordinates: u32, block_width: usize) -> Result<usize, SparseMatrixError> {
    if block_width == 0 {
        return Err(SparseMatrixError::ZeroBlockWidth);
    }
    (coordinates as usize)
        .checked_mul(block_width)
        .ok_or(SparseMatrixError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> CsrMatrix {
        // [ 2  0 -1  0 ]
        // [ 0  3  0  4 ]
        // [-2  0  5  0 ]
        CsrMatrix::from_triplets(
            3,
            4,
            vec![
                Triplet {
                    row: 2,
                    column: 2,
                    coefficient: 5,
                },
                Triplet {
                    row: 0,
                    column: 2,
                    coefficient: -1,
                },
                Triplet {
                    row: 1,
                    column: 3,
                    coefficient: 4,
                },
                Triplet {
                    row: 2,
                    column: 0,
                    coefficient: -2,
                },
                Triplet {
                    row: 0,
                    column: 0,
                    coefficient: 2,
                },
                Triplet {
                    row: 1,
                    column: 1,
                    coefficient: 3,
                },
            ],
        )
        .unwrap()
    }

    fn dense_spmv(matrix: &[Vec<i32>], input: &[u32]) -> Vec<u32> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(input)
                    .fold(0, |sum, (&coefficient, &value)| {
                        field_add(sum, field_mul_signed(coefficient, value))
                    })
            })
            .collect()
    }

    fn dense_transpose_spmv(matrix: &[Vec<i32>], input: &[u32]) -> Vec<u32> {
        (0..matrix[0].len())
            .map(|column| {
                matrix.iter().zip(input).fold(0, |sum, (row, &value)| {
                    field_add(sum, field_mul_signed(row[column], value))
                })
            })
            .collect()
    }

    #[test]
    fn mersenne_field_arithmetic_matches_u64_modulo() {
        let values = [0, 1, 2, 127, 1_000_000_007, PRIME - 2, PRIME - 1];
        for &left in &values {
            for &right in &values {
                assert_eq!(
                    field_mul(left, right),
                    ((u64::from(left) * u64::from(right)) % PRIME_U64) as u32
                );
                assert_eq!(
                    field_add(left, right),
                    ((u64::from(left) + u64::from(right)) % PRIME_U64) as u32
                );
            }
        }
        assert_eq!(field_from_i64(-1), PRIME - 1);
        assert_eq!(field_from_i64(i64::from(PRIME)), 0);

        let mut state = 0x9e37_79b9_u32;
        for _ in 0..10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let left = state % PRIME;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let right = state % PRIME;
            assert_eq!(
                field_mul(left, right),
                ((u64::from(left) * u64::from(right)) % PRIME_U64) as u32
            );
        }
    }

    #[test]
    fn construction_is_canonical_and_combines_duplicates() {
        let matrix = CsrMatrix::from_triplets(
            2,
            3,
            vec![
                Triplet {
                    row: 1,
                    column: 2,
                    coefficient: 4,
                },
                Triplet {
                    row: 0,
                    column: 1,
                    coefficient: 7,
                },
                Triplet {
                    row: 1,
                    column: 2,
                    coefficient: -1,
                },
                Triplet {
                    row: 0,
                    column: 1,
                    coefficient: -7,
                },
                Triplet {
                    row: 1,
                    column: 0,
                    coefficient: -2,
                },
            ],
        )
        .unwrap();
        assert_eq!(matrix.row_offsets(), &[0, 0, 2]);
        assert_eq!(matrix.column_indices(), &[0, 2]);
        assert_eq!(matrix.coefficients(), &[-2, 3]);
    }

    #[test]
    fn csr_scalar_multiply_matches_dense_reference() {
        let matrix = sample_matrix();
        let dense = vec![vec![2, 0, -1, 0], vec![0, 3, 0, 4], vec![-2, 0, 5, 0]];
        let input = [7, PRIME - 3, 11, 13];
        assert_eq!(matrix.spmv(&input).unwrap(), dense_spmv(&dense, &input));
    }

    #[test]
    fn csc_transpose_multiply_matches_dense_reference() {
        let matrix = sample_matrix();
        let csc = matrix.to_csc();
        let dense = vec![vec![2, 0, -1, 0], vec![0, 3, 0, 4], vec![-2, 0, 5, 0]];
        let input = [17, PRIME - 5, 23];
        assert_eq!(
            csc.transpose_spmv(&input).unwrap(),
            dense_transpose_spmv(&dense, &input)
        );
        assert_eq!(csc.column_offsets(), &[0, 2, 3, 5, 6]);
        assert_eq!(csc.row_indices(), &[0, 2, 1, 0, 2, 1]);
    }

    #[test]
    fn block_multiply_matches_independent_scalar_lanes() {
        let matrix = sample_matrix();
        let csc = matrix.to_csc();
        let width = 3;
        let input = [
            1, 2, 3, // coordinate 0
            4, 5, 6, // coordinate 1
            7, 8, 9, // coordinate 2
            10, 11, 12, // coordinate 3
        ];
        let block_output = matrix.spmm(width, &input).unwrap();
        for lane in 0..width {
            let scalar_input: Vec<_> = (0..4).map(|column| input[column * width + lane]).collect();
            let scalar_output = matrix.spmv(&scalar_input).unwrap();
            for row in 0..3 {
                assert_eq!(block_output[row * width + lane], scalar_output[row]);
            }
        }

        let transpose_input = [
            13, 14, 15, // coordinate 0
            16, 17, 18, // coordinate 1
            19, 20, 21, // coordinate 2
        ];
        let block_output = csc.transpose_spmm(width, &transpose_input).unwrap();
        for lane in 0..width {
            let scalar_input: Vec<_> = (0..3)
                .map(|row| transpose_input[row * width + lane])
                .collect();
            let scalar_output = csc.transpose_spmv(&scalar_input).unwrap();
            for column in 0..4 {
                assert_eq!(block_output[column * width + lane], scalar_output[column]);
            }
        }
    }

    #[test]
    fn deterministic_construction_ignores_triplet_order() {
        let first = sample_matrix();
        let mut entries = vec![
            Triplet {
                row: 0,
                column: 0,
                coefficient: 2,
            },
            Triplet {
                row: 0,
                column: 2,
                coefficient: -1,
            },
            Triplet {
                row: 1,
                column: 1,
                coefficient: 3,
            },
            Triplet {
                row: 1,
                column: 3,
                coefficient: 4,
            },
            Triplet {
                row: 2,
                column: 0,
                coefficient: -2,
            },
            Triplet {
                row: 2,
                column: 2,
                coefficient: 5,
            },
        ];
        entries.reverse();
        let second = CsrMatrix::from_triplets(3, 4, entries).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.to_csc(), second.to_csc());
    }

    #[test]
    fn deterministic_generated_sparse_matrix_matches_dense_in_both_directions() {
        const ROWS: usize = 17;
        const COLUMNS: usize = 13;
        let mut state = 0x243f_6a88_u32;
        let mut dense = vec![vec![0_i32; COLUMNS]; ROWS];
        let mut triplets = Vec::new();
        for _ in 0..180 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let row = state as usize % ROWS;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let column = state as usize % COLUMNS;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let coefficient = (state % 9) as i32 - 4;
            dense[row][column] += coefficient;
            triplets.push(Triplet {
                row: row as u32,
                column: column as u32,
                coefficient,
            });
        }
        let matrix = CsrMatrix::from_triplets(ROWS as u32, COLUMNS as u32, triplets).unwrap();

        let input: Vec<_> = (0..COLUMNS)
            .map(|index| ((index as u64 * 1_000_000_007 + 19) % PRIME_U64) as u32)
            .collect();
        assert_eq!(matrix.spmv(&input).unwrap(), dense_spmv(&dense, &input));

        let transpose_input: Vec<_> = (0..ROWS)
            .map(|index| ((index as u64 * 998_244_353 + 23) % PRIME_U64) as u32)
            .collect();
        assert_eq!(
            matrix.to_csc().transpose_spmv(&transpose_input).unwrap(),
            dense_transpose_spmv(&dense, &transpose_input)
        );
    }

    #[test]
    fn rejects_invalid_dimensions_and_field_values() {
        let matrix = sample_matrix();
        assert!(matches!(
            matrix.spmv(&[1, 2]),
            Err(SparseMatrixError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            matrix.spmv(&[1, 2, 3, PRIME]),
            Err(SparseMatrixError::NonCanonicalFieldElement { index: 3, .. })
        ));
        assert_eq!(matrix.spmm(0, &[]), Err(SparseMatrixError::ZeroBlockWidth));
    }
}
