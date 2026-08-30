//! Integer reconstruction and exact certificates for modular sparse kernels.

use crate::{CsrMatrix, PRIME, field_from_i64, field_mul, field_sub};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReconstructedRational {
    pub numerator: i64,
    pub denominator: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CertificateError {
    InvalidLabel(String),
    VectorLength {
        copy: usize,
        expected: usize,
        actual: usize,
    },
    NonCanonicalResidue {
        copy: usize,
        column: usize,
        residue: u32,
    },
    RationalReconstruction {
        copy: usize,
        column: usize,
        residue: u32,
    },
    CommonDenominatorOverflow {
        copy: usize,
    },
    ZeroVector {
        copy: usize,
    },
    IntegerCoefficientOverflow {
        copy: usize,
        column: usize,
        value: i128,
    },
    ResidualOverflow {
        copy: usize,
        row: u32,
    },
    NonzeroIntegerResidual {
        copy: usize,
        row: u32,
        residual: i128,
    },
    EncodingOverflow {
        copy: usize,
        column: usize,
        coefficient: i64,
        width_bytes: usize,
    },
    DependentIntegerKernelBasis {
        vectors: usize,
        modular_rank: usize,
    },
    CompleteModularRankOutOfBounds {
        rank: usize,
        source_columns: usize,
    },
    CharacteristicZeroRankBoundsDoNotMeet {
        source_columns: usize,
        complete_modular_rank: usize,
        independent_integer_kernels: usize,
    },
}

impl Display for CertificateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLabel(label) => write!(formatter, "invalid level-12 label {label}"),
            Self::VectorLength {
                copy,
                expected,
                actual,
            } => write!(
                formatter,
                "kernel copy {copy} has {actual} coefficients, expected {expected}"
            ),
            Self::NonCanonicalResidue {
                copy,
                column,
                residue,
            } => write!(
                formatter,
                "kernel copy {copy} column {column} has noncanonical residue {residue}"
            ),
            Self::RationalReconstruction {
                copy,
                column,
                residue,
            } => write!(
                formatter,
                "rational reconstruction failed for kernel copy {copy} column {column} residue {residue}"
            ),
            Self::CommonDenominatorOverflow { copy } => write!(
                formatter,
                "common denominator overflow for kernel copy {copy}"
            ),
            Self::ZeroVector { copy } => {
                write!(
                    formatter,
                    "kernel copy {copy} reconstructed as the zero vector"
                )
            }
            Self::IntegerCoefficientOverflow {
                copy,
                column,
                value,
            } => write!(
                formatter,
                "kernel copy {copy} column {column} coefficient {value} exceeds i64"
            ),
            Self::ResidualOverflow { copy, row } => write!(
                formatter,
                "integer residual overflow for kernel copy {copy} row {row}"
            ),
            Self::NonzeroIntegerResidual {
                copy,
                row,
                residual,
            } => write!(
                formatter,
                "kernel copy {copy} has nonzero integer residual {residual} at row {row}"
            ),
            Self::EncodingOverflow {
                copy,
                column,
                coefficient,
                width_bytes,
            } => write!(
                formatter,
                "kernel copy {copy} column {column} coefficient {coefficient} does not fit signed {width_bytes}-byte encoding"
            ),
            Self::DependentIntegerKernelBasis {
                vectors,
                modular_rank,
            } => write!(
                formatter,
                "{vectors} reconstructed integer kernels have modular rank {modular_rank} and are dependent"
            ),
            Self::CompleteModularRankOutOfBounds {
                rank,
                source_columns,
            } => write!(
                formatter,
                "complete modular rank {rank} exceeds {source_columns} source columns"
            ),
            Self::CharacteristicZeroRankBoundsDoNotMeet {
                source_columns,
                complete_modular_rank,
                independent_integer_kernels,
            } => write!(
                formatter,
                "modular rank lower bound {complete_modular_rank} and {independent_integer_kernels} independent integer kernels do not close rank-nullity for {source_columns} columns"
            ),
        }
    }
}

impl Error for CertificateError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelArtifactMetadata {
    pub copy: usize,
    pub path: String,
    pub sha256: String,
    pub bytes: usize,
    pub nonzero_coefficients: usize,
    pub maximum_absolute_coefficient: u64,
    pub integer_residual_rows_checked: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedIntegerKernel {
    pub coefficients: Vec<i64>,
    pub encoded_little_endian: Vec<u8>,
    pub metadata: KernelArtifactMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCertificateBatch {
    pub prime: u32,
    pub matrix_rows: usize,
    pub source_columns: usize,
    pub matrix_nonzeros: usize,
    pub matrix_sha256: String,
    pub coefficient_width_bytes: usize,
    pub integer_kernel_vectors_verified: usize,
    pub reconstructed_kernel_rank_mod_prime: usize,
    pub integer_kernels_independent_mod_prime: bool,
    pub kernels: Vec<CertifiedIntegerKernel>,
}

/// Deterministic characteristic-zero rank proof obtained by meeting two exact
/// bounds. A nonzero modular minor gives the lower bound on integer-matrix
/// rank, while independent exact integer kernels give the upper bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CharacteristicZeroRankCertificate {
    pub prime: u32,
    pub matrix_rows: usize,
    pub source_columns: usize,
    pub matrix_nonzeros: usize,
    pub matrix_sha256: String,
    pub complete_modular_rank: usize,
    pub complete_modular_nullity: usize,
    pub independent_integer_kernel_vectors: usize,
    pub reconstructed_kernel_rank_mod_prime: usize,
    pub modular_rank_lower_bound: usize,
    pub integer_kernel_rank_upper_bound: usize,
    pub characteristic_zero_rank: usize,
    pub characteristic_zero_nullity: usize,
    pub deterministic_characteristic_zero_rank_certified: bool,
}

impl KernelCertificateBatch {
    /// Combine this exact integer-kernel certificate with the rank from a
    /// complete modular elimination of the same matrix at `self.prime`.
    pub fn certify_characteristic_zero_rank(
        &self,
        complete_modular_rank: usize,
    ) -> Result<CharacteristicZeroRankCertificate, CertificateError> {
        if complete_modular_rank > self.source_columns {
            return Err(CertificateError::CompleteModularRankOutOfBounds {
                rank: complete_modular_rank,
                source_columns: self.source_columns,
            });
        }
        let independent_integer_kernels = self.reconstructed_kernel_rank_mod_prime;
        let integer_kernel_rank_upper_bound = self
            .source_columns
            .checked_sub(independent_integer_kernels)
            .expect("kernel rank cannot exceed source columns");
        if !self.integer_kernels_independent_mod_prime
            || self.integer_kernel_vectors_verified != independent_integer_kernels
            || complete_modular_rank != integer_kernel_rank_upper_bound
        {
            return Err(CertificateError::CharacteristicZeroRankBoundsDoNotMeet {
                source_columns: self.source_columns,
                complete_modular_rank,
                independent_integer_kernels,
            });
        }
        let complete_modular_nullity = self.source_columns - complete_modular_rank;
        Ok(CharacteristicZeroRankCertificate {
            prime: self.prime,
            matrix_rows: self.matrix_rows,
            source_columns: self.source_columns,
            matrix_nonzeros: self.matrix_nonzeros,
            matrix_sha256: self.matrix_sha256.clone(),
            complete_modular_rank,
            complete_modular_nullity,
            independent_integer_kernel_vectors: independent_integer_kernels,
            reconstructed_kernel_rank_mod_prime: self.reconstructed_kernel_rank_mod_prime,
            modular_rank_lower_bound: complete_modular_rank,
            integer_kernel_rank_upper_bound,
            characteristic_zero_rank: complete_modular_rank,
            characteristic_zero_nullity: independent_integer_kernels,
            deterministic_characteristic_zero_rank_certified: true,
        })
    }
}

/// Rationally reconstruct one canonical modular residue using the same bound
/// and extended Euclidean recurrence as the Python fixture generator.
pub fn rational_reconstruct(residue: u32, modulus: u32) -> Option<ReconstructedRational> {
    if modulus < 2 {
        return None;
    }
    let residue = i64::from(residue % modulus);
    let modulus = i64::from(modulus);
    let bound = integer_sqrt((modulus / 2) as u64) as i64;
    let (mut old_remainder, mut remainder) = (modulus, residue);
    let (mut old_denominator, mut denominator) = (0_i64, 1_i64);
    while remainder.abs() > bound {
        if remainder == 0 {
            return None;
        }
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
        (old_denominator, denominator) = (denominator, old_denominator - quotient * denominator);
    }
    if denominator == 0 {
        return None;
    }
    if denominator < 0 {
        remainder = -remainder;
        denominator = -denominator;
    }
    let divisor = gcd_i64(remainder.abs(), denominator);
    let numerator = remainder / divisor;
    let denominator = denominator / divisor;
    if numerator.abs() <= bound
        && denominator <= bound
        && (residue * denominator - numerator).rem_euclid(modulus) == 0
    {
        Some(ReconstructedRational {
            numerator,
            denominator,
        })
    } else {
        None
    }
}

/// Reconstruct, primitive-normalize, verify, and encode a complete modular
/// kernel basis. Width selection is shared across the system, matching the
/// Python generator's signed i16/i32 artifact convention.
pub fn certify_kernel_basis(
    label: &str,
    matrix: &CsrMatrix,
    modular_basis: &[Vec<u32>],
) -> Result<KernelCertificateBatch, CertificateError> {
    if label.len() != 5 || !label.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CertificateError::InvalidLabel(label.to_owned()));
    }

    let mut integer_kernels = Vec::with_capacity(modular_basis.len());
    for (index, modular) in modular_basis.iter().enumerate() {
        let copy = index + 1;
        let coefficients = primitive_integer_reconstruction(modular, copy, matrix.columns())?;
        verify_integer_kernel(matrix, &coefficients, copy)?;
        integer_kernels.push(coefficients);
    }
    let reconstructed_kernel_rank_mod_prime = modular_rank(&integer_kernels);
    if reconstructed_kernel_rank_mod_prime != integer_kernels.len() {
        return Err(CertificateError::DependentIntegerKernelBasis {
            vectors: integer_kernels.len(),
            modular_rank: reconstructed_kernel_rank_mod_prime,
        });
    }

    let maximum_absolute_coefficient = integer_kernels
        .iter()
        .flat_map(|kernel| kernel.iter())
        .map(|coefficient| coefficient.unsigned_abs())
        .max()
        .unwrap_or(0);
    let coefficient_width_bytes = if maximum_absolute_coefficient <= i16::MAX as u64 {
        2
    } else {
        4
    };

    let copies = integer_kernels.len();
    let mut kernels = Vec::with_capacity(copies);
    for (index, coefficients) in integer_kernels.into_iter().enumerate() {
        let copy = index + 1;
        let encoded_little_endian =
            encode_signed_little_endian(&coefficients, coefficient_width_bytes, copy)?;
        let suffix = if copies == 1 {
            String::new()
        } else {
            format!("_{copy}")
        };
        let path = format!(
            "data/eleven_dimensional_spinor_bridge/level12_{label}_highest_weight_kernel{suffix}.i{}le",
            coefficient_width_bytes * 8
        );
        let sha256 = sha256_hex(&encoded_little_endian);
        let metadata = KernelArtifactMetadata {
            copy,
            path,
            sha256,
            bytes: encoded_little_endian.len(),
            nonzero_coefficients: coefficients
                .iter()
                .filter(|coefficient| **coefficient != 0)
                .count(),
            maximum_absolute_coefficient: coefficients
                .iter()
                .map(|coefficient| coefficient.unsigned_abs())
                .max()
                .unwrap_or(0),
            integer_residual_rows_checked: matrix.rows() as usize,
        };
        kernels.push(CertifiedIntegerKernel {
            coefficients,
            encoded_little_endian,
            metadata,
        });
    }

    Ok(KernelCertificateBatch {
        prime: PRIME,
        matrix_rows: matrix.rows() as usize,
        source_columns: matrix.columns() as usize,
        matrix_nonzeros: matrix.nonzeros(),
        matrix_sha256: canonical_matrix_sha256(label, matrix),
        coefficient_width_bytes,
        integer_kernel_vectors_verified: kernels.len(),
        reconstructed_kernel_rank_mod_prime,
        integer_kernels_independent_mod_prime: true,
        kernels,
    })
}

fn primitive_integer_reconstruction(
    modular: &[u32],
    copy: usize,
    columns: u32,
) -> Result<Vec<i64>, CertificateError> {
    let expected = columns as usize;
    if modular.len() != expected {
        return Err(CertificateError::VectorLength {
            copy,
            expected,
            actual: modular.len(),
        });
    }
    let mut rationals = Vec::with_capacity(expected);
    for (column, &residue) in modular.iter().enumerate() {
        if residue >= PRIME {
            return Err(CertificateError::NonCanonicalResidue {
                copy,
                column,
                residue,
            });
        }
        let value = rational_reconstruct(residue, PRIME).ok_or(
            CertificateError::RationalReconstruction {
                copy,
                column,
                residue,
            },
        )?;
        rationals.push(value);
    }

    let mut common_denominator = 1_i128;
    for value in &rationals {
        let denominator = i128::from(value.denominator);
        let divisor = gcd_i128(common_denominator, denominator);
        common_denominator = common_denominator
            .checked_div(divisor)
            .and_then(|reduced| reduced.checked_mul(denominator))
            .ok_or(CertificateError::CommonDenominatorOverflow { copy })?;
    }

    let mut coefficients = Vec::with_capacity(expected);
    for value in rationals {
        let scale = common_denominator / i128::from(value.denominator);
        let coefficient = i128::from(value.numerator)
            .checked_mul(scale)
            .ok_or(CertificateError::CommonDenominatorOverflow { copy })?;
        coefficients.push(coefficient);
    }
    let divisor = coefficients
        .iter()
        .fold(0_i128, |gcd, coefficient| gcd_i128(gcd, coefficient.abs()));
    if divisor == 0 {
        return Err(CertificateError::ZeroVector { copy });
    }
    for coefficient in &mut coefficients {
        *coefficient /= divisor;
    }
    if coefficients
        .iter()
        .find(|coefficient| **coefficient != 0)
        .is_some_and(|coefficient| *coefficient < 0)
    {
        for coefficient in &mut coefficients {
            *coefficient = -*coefficient;
        }
    }

    coefficients
        .into_iter()
        .enumerate()
        .map(|(column, value)| {
            i64::try_from(value).map_err(|_| CertificateError::IntegerCoefficientOverflow {
                copy,
                column,
                value,
            })
        })
        .collect()
}

fn verify_integer_kernel(
    matrix: &CsrMatrix,
    coefficients: &[i64],
    copy: usize,
) -> Result<(), CertificateError> {
    let row_offsets = matrix.row_offsets();
    let column_indices = matrix.column_indices();
    let matrix_coefficients = matrix.coefficients();
    for row in 0..matrix.rows() {
        let start = row_offsets[row as usize] as usize;
        let end = row_offsets[row as usize + 1] as usize;
        let mut residual = 0_i128;
        for index in start..end {
            let term = i128::from(matrix_coefficients[index])
                .checked_mul(i128::from(coefficients[column_indices[index] as usize]))
                .ok_or(CertificateError::ResidualOverflow { copy, row })?;
            residual = residual
                .checked_add(term)
                .ok_or(CertificateError::ResidualOverflow { copy, row })?;
        }
        if residual != 0 {
            return Err(CertificateError::NonzeroIntegerResidual {
                copy,
                row,
                residual,
            });
        }
    }
    Ok(())
}

fn encode_signed_little_endian(
    coefficients: &[i64],
    width_bytes: usize,
    copy: usize,
) -> Result<Vec<u8>, CertificateError> {
    let mut encoded = Vec::with_capacity(coefficients.len().saturating_mul(width_bytes));
    for (column, &coefficient) in coefficients.iter().enumerate() {
        match width_bytes {
            2 => {
                let value =
                    i16::try_from(coefficient).map_err(|_| CertificateError::EncodingOverflow {
                        copy,
                        column,
                        coefficient,
                        width_bytes,
                    })?;
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            4 => {
                let value =
                    i32::try_from(coefficient).map_err(|_| CertificateError::EncodingOverflow {
                        copy,
                        column,
                        coefficient,
                        width_bytes,
                    })?;
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            _ => unreachable!("certificate width is always two or four bytes"),
        }
    }
    Ok(encoded)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_matrix_sha256(label: &str, matrix: &CsrMatrix) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"adynkra-level12-raising-csr-v1");
    hasher.update([0]);
    hasher.update(label.as_bytes());
    hasher.update([0]);
    hasher.update(matrix.rows().to_le_bytes());
    hasher.update(matrix.columns().to_le_bytes());
    hasher.update((matrix.nonzeros() as u64).to_le_bytes());
    let offsets = matrix.row_offsets();
    for row in 0..matrix.rows() as usize {
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        for index in start..end {
            hasher.update((row as u32).to_le_bytes());
            hasher.update(matrix.column_indices()[index].to_le_bytes());
            hasher.update(matrix.coefficients()[index].to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Rank the reconstructed integer vectors after exact reduction modulo the
/// solver prime. Independence modulo one prime proves independence over the
/// integers and over characteristic zero.
fn modular_rank(integer_vectors: &[Vec<i64>]) -> usize {
    let Some(first) = integer_vectors.first() else {
        return 0;
    };
    let columns = first.len();
    debug_assert!(integer_vectors.iter().all(|vector| vector.len() == columns));
    let mut rows = integer_vectors
        .iter()
        .map(|vector| {
            vector
                .iter()
                .map(|&coefficient| field_from_i64(coefficient))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut rank = 0_usize;
    for column in 0..columns {
        let Some(pivot) = (rank..rows.len()).find(|&row| rows[row][column] != 0) else {
            continue;
        };
        rows.swap(rank, pivot);
        let inverse = modular_inverse(rows[rank][column]);
        for entry in &mut rows[rank][column..] {
            *entry = field_mul(*entry, inverse);
        }
        let (pivot_rows, remaining_rows) = rows.split_at_mut(rank + 1);
        let pivot_row = &pivot_rows[rank];
        for row in remaining_rows {
            let scale = row[column];
            if scale == 0 {
                continue;
            }
            for (entry, &pivot_entry) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *entry = field_sub(*entry, field_mul(scale, pivot_entry));
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rank
}

fn modular_inverse(value: u32) -> u32 {
    debug_assert!(value != 0 && value < PRIME);
    let mut base = value;
    let mut exponent = u64::from(PRIME - 2);
    let mut result = 1_u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = field_mul(result, base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = field_mul(base, base);
        }
    }
    result
}

fn integer_sqrt(value: u64) -> u64 {
    if value < 2 {
        return value;
    }
    let mut estimate = 1_u64 << ((64 - value.leading_zeros() as u64).div_ceil(2));
    loop {
        let next = (estimate + value / estimate) / 2;
        if next >= estimate {
            return estimate;
        }
        estimate = next;
    }
}

fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.abs()
}

fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triplet;
    use crate::elimination::{EliminationBudget, EliminationOutcome, eliminate};
    use crate::level12::build_level12_matrix;
    use std::path::{Path, PathBuf};

    fn repository_root() -> PathBuf {
        if let Some(root) = std::env::var_os("ADYNKRA_REPO_ROOT") {
            return PathBuf::from(root);
        }
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest
            .ancestors()
            .find(|ancestor| {
                ancestor
                    .join("data/eleven_dimensional_spinor_bridge")
                    .is_dir()
            })
            .expect("repository data directory must be reachable from the crate")
            .to_path_buf()
    }

    #[test]
    fn rational_reconstruction_matches_small_signed_fractions() {
        assert_eq!(
            rational_reconstruct(0, PRIME),
            Some(ReconstructedRational {
                numerator: 0,
                denominator: 1
            })
        );
        assert_eq!(
            rational_reconstruct(PRIME - 1, PRIME),
            Some(ReconstructedRational {
                numerator: -1,
                denominator: 1
            })
        );
        assert_eq!(
            rational_reconstruct(PRIME.div_ceil(2), PRIME),
            Some(ReconstructedRational {
                numerator: 1,
                denominator: 2
            })
        );
    }

    #[test]
    fn signed_encodings_are_little_endian_and_width_checked() {
        assert_eq!(
            encode_signed_little_endian(&[-2, 1, i16::MAX as i64], 2, 1).unwrap(),
            [0xfe, 0xff, 1, 0, 0xff, 0x7f]
        );
        assert_eq!(
            encode_signed_little_endian(&[-32_768, 40_000], 4, 1).unwrap(),
            [0x00, 0x80, 0xff, 0xff, 0x40, 0x9c, 0x00, 0x00]
        );
        assert!(encode_signed_little_endian(&[32_768], 2, 1).is_err());
    }

    fn zero_matrix(columns: u32) -> CsrMatrix {
        CsrMatrix::from_triplets(1, columns, Vec::<Triplet>::new()).unwrap()
    }

    #[test]
    fn rejects_duplicate_reconstructed_integer_kernels() {
        let error =
            certify_kernel_basis("00000", &zero_matrix(2), &[vec![1, 0], vec![1, 0]]).unwrap_err();
        assert_eq!(
            error,
            CertificateError::DependentIntegerKernelBasis {
                vectors: 2,
                modular_rank: 1
            }
        );
    }

    #[test]
    fn rejects_distinct_vectors_with_a_linear_dependence() {
        let error = certify_kernel_basis(
            "00000",
            &zero_matrix(2),
            &[vec![1, 0], vec![0, 1], vec![1, 1]],
        )
        .unwrap_err();
        assert_eq!(
            error,
            CertificateError::DependentIntegerKernelBasis {
                vectors: 3,
                modular_rank: 2
            }
        );
    }

    #[test]
    fn level12_30002_reproduces_all_published_integer_fixtures() {
        let level12 = build_level12_matrix("30002").unwrap();
        let elimination = match eliminate(&level12.raising, EliminationBudget::unlimited()) {
            EliminationOutcome::Complete(result) => result,
            EliminationOutcome::ThresholdExceeded(threshold) => {
                panic!("unexpected elimination threshold: {threshold:?}")
            }
        };
        assert_eq!(elimination.rank, 2_889);
        assert_eq!(elimination.free_columns, [2_816, 2_829, 2_891]);

        let certificate =
            certify_kernel_basis("30002", &level12.raising, &elimination.kernel_basis).unwrap();
        assert_eq!(certificate.prime, PRIME);
        assert_eq!(certificate.matrix_rows, 6_169);
        assert_eq!(certificate.source_columns, 2_892);
        assert_eq!(certificate.matrix_nonzeros, 19_549);
        assert_eq!(certificate.matrix_sha256, level12.canonical_sha256());
        assert_eq!(certificate.coefficient_width_bytes, 2);
        assert_eq!(certificate.integer_kernel_vectors_verified, 3);
        assert_eq!(certificate.reconstructed_kernel_rank_mod_prime, 3);
        assert!(certificate.integer_kernels_independent_mod_prime);
        assert_eq!(certificate.kernels.len(), 3);

        let rank_certificate = certificate.certify_characteristic_zero_rank(2_889).unwrap();
        assert_eq!(rank_certificate.prime, PRIME);
        assert_eq!(rank_certificate.matrix_rows, 6_169);
        assert_eq!(rank_certificate.source_columns, 2_892);
        assert_eq!(rank_certificate.matrix_nonzeros, 19_549);
        assert_eq!(rank_certificate.matrix_sha256, level12.canonical_sha256());
        assert_eq!(rank_certificate.complete_modular_rank, 2_889);
        assert_eq!(rank_certificate.complete_modular_nullity, 3);
        assert_eq!(rank_certificate.independent_integer_kernel_vectors, 3);
        assert_eq!(rank_certificate.reconstructed_kernel_rank_mod_prime, 3);
        assert_eq!(rank_certificate.modular_rank_lower_bound, 2_889);
        assert_eq!(rank_certificate.integer_kernel_rank_upper_bound, 2_889);
        assert_eq!(rank_certificate.characteristic_zero_rank, 2_889);
        assert_eq!(rank_certificate.characteristic_zero_nullity, 3);
        assert!(rank_certificate.deterministic_characteristic_zero_rank_certified);
        assert!(matches!(
            certificate.certify_characteristic_zero_rank(2_888),
            Err(CertificateError::CharacteristicZeroRankBoundsDoNotMeet { .. })
        ));

        let expected = [
            (
                "bef24d4ebe642c0b6507dee9706fea06742d6ed708d14adc9dd599a8061c28c6",
                1_858,
                56,
            ),
            (
                "0e42717e2cb9449f92257a60b7a35e3c8a501dea176db02aa2d0a714b8904908",
                1_120,
                2,
            ),
            (
                "d3321a53f37acea6d3d8e4efc11022c037e0339a5e9a7cda87bdb003ca3b2945",
                2_738,
                1_760,
            ),
        ];
        let root = repository_root();
        for (index, kernel) in certificate.kernels.iter().enumerate() {
            let copy = index + 1;
            let published_path = root.join(format!(
                "data/eleven_dimensional_spinor_bridge/level12_30002_highest_weight_kernel_{copy}.i16le"
            ));
            let published_bytes = std::fs::read(&published_path).unwrap();
            assert_eq!(kernel.encoded_little_endian, published_bytes);
            assert_eq!(kernel.metadata.copy, copy);
            assert_eq!(kernel.metadata.sha256, expected[index].0);
            assert_eq!(kernel.metadata.bytes, 5_784);
            assert_eq!(kernel.metadata.nonzero_coefficients, expected[index].1);
            assert_eq!(
                kernel.metadata.maximum_absolute_coefficient,
                expected[index].2
            );
            assert_eq!(kernel.metadata.integer_residual_rows_checked, 6_169);
            assert_eq!(
                kernel.metadata.path,
                published_path
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
            );
        }
    }
}
