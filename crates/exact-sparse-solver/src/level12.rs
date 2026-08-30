//! Exact level-12 highest-weight raising matrices.
//!
//! This is a direct port of the matrix-construction portion of
//! `scripts/generate_level12_second_momentum_kernels.py`. In particular, source
//! columns are increasing 32-bit exterior masks and raising signs use the same
//! occupied-index interval parity convention.

use crate::{CsrMatrix, SparseMatrixError, Triplet};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const DEGREE: u32 = 12;
pub const CANONICAL_CSR_DIGEST_SCHEMA: &str = "adynkra-level12-raising-csr-v1";
pub const SOURCE_LABELED_DIGEST_SCHEMA: &str = "adynkra-level12-source-labeled-raising-v1";
const MEASURED_ENTRIES_PER_SOURCE_COLUMN: usize = 13;
const ROOTS: [[i16; 5]; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];

type Weight = [i16; 5];
type HalfGroups = HashMap<(u8, Weight), Vec<u16>>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Level12Error {
    UnsupportedLabel(String),
    DimensionTooLarge(usize),
    Sparse(SparseMatrixError),
}

impl Display for Level12Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLabel(label) => {
                write!(formatter, "unsupported level-12 label {label}")
            }
            Self::DimensionTooLarge(value) => {
                write!(formatter, "level-12 matrix dimension {value} exceeds u32")
            }
            Self::Sparse(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for Level12Error {}

impl From<SparseMatrixError> for Level12Error {
    fn from(value: SparseMatrixError) -> Self {
        Self::Sparse(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Level12Matrix {
    pub label: String,
    /// Canonical Python-compatible source-column order.
    pub source_masks: Vec<u32>,
    pub raising: CsrMatrix,
}

impl Level12Matrix {
    pub fn row_degree_histogram(&self) -> BTreeMap<u32, u64> {
        let mut histogram = BTreeMap::new();
        for offsets in self.raising.row_offsets().windows(2) {
            *histogram.entry(offsets[1] - offsets[0]).or_insert(0) += 1;
        }
        histogram
    }

    pub fn column_degree_histogram(&self) -> BTreeMap<u32, u64> {
        let mut degrees = vec![0_u32; self.raising.columns() as usize];
        for &column in self.raising.column_indices() {
            degrees[column as usize] += 1;
        }
        let mut histogram = BTreeMap::new();
        for degree in degrees {
            *histogram.entry(degree).or_insert(0) += 1;
        }
        histogram
    }

    /// SHA-256 of a versioned canonical coordinate stream. All integers are
    /// little-endian and entries are emitted in CSR row/column order.
    pub fn canonical_sha256(&self) -> String {
        self.canonical_digest_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// SHA-256 that binds the ordered exterior masks to the numeric CSR digest.
    /// This is the checkpoint identity for the source-labeled linear map.
    pub fn source_labeled_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(SOURCE_LABELED_DIGEST_SCHEMA.as_bytes());
        hasher.update([0]);
        hasher.update(DEGREE.to_le_bytes());
        hasher.update((self.label.len() as u64).to_le_bytes());
        hasher.update(self.label.as_bytes());
        hasher.update((self.source_masks.len() as u64).to_le_bytes());
        for &mask in &self.source_masks {
            hasher.update(mask.to_le_bytes());
        }
        hasher.update(self.canonical_digest_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn canonical_digest_bytes(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_CSR_DIGEST_SCHEMA.as_bytes());
        hasher.update([0]);
        hasher.update(self.label.as_bytes());
        hasher.update([0]);
        hasher.update(self.raising.rows().to_le_bytes());
        hasher.update(self.raising.columns().to_le_bytes());
        hasher.update((self.raising.nonzeros() as u64).to_le_bytes());
        for row in 0..self.raising.rows() as usize {
            let start = self.raising.row_offsets()[row] as usize;
            let end = self.raising.row_offsets()[row + 1] as usize;
            for index in start..end {
                hasher.update((row as u32).to_le_bytes());
                hasher.update(self.raising.column_indices()[index].to_le_bytes());
                hasher.update(self.raising.coefficients()[index].to_le_bytes());
            }
        }
        hasher.finalize().into()
    }
}

pub fn expected_multiplicity(label: &str) -> Option<usize> {
    Some(match label {
        "00000" | "00010" | "00100" | "12000" | "30100" | "31000" | "40000" => 1,
        "01100" | "02000" | "10002" | "20002" | "20100" | "30010" => 2,
        "11100" | "20010" | "30002" => 3,
        "01002" | "11010" => 4,
        "11002" => 5,
        _ => return None,
    })
}

/// Build exactly the matrix produced by the Python level-12 generator.
pub fn build_level12_matrix(label: &str) -> Result<Level12Matrix, Level12Error> {
    if expected_multiplicity(label).is_none() {
        return Err(Level12Error::UnsupportedLabel(label.to_owned()));
    }
    let weights = spinor_weights();
    let raise = raising_table(&weights);
    let left = half_groups(0, &weights);
    let right = half_groups(16, &weights);
    let target = highest_weight(label)?;
    let source_masks = weight_basis(DEGREE, target, &left, &right);

    let mut blocks = Vec::with_capacity(ROOTS.len());
    let mut row_count = 0_usize;
    for root in ROOTS {
        let output_weight = add_weights(target, root);
        let basis = weight_basis(DEGREE, output_weight, &left, &right);
        let block: HashMap<u32, u32> = basis
            .into_iter()
            .enumerate()
            .map(|(index, mask)| {
                let row = row_count
                    .checked_add(index)
                    .ok_or(Level12Error::DimensionTooLarge(usize::MAX))?;
                let row = u32::try_from(row).map_err(|_| Level12Error::DimensionTooLarge(row))?;
                Ok((mask, row))
            })
            .collect::<Result<_, Level12Error>>()?;
        row_count = row_count
            .checked_add(block.len())
            .ok_or(Level12Error::DimensionTooLarge(usize::MAX))?;
        blocks.push(block);
    }
    let rows = u32::try_from(row_count).map_err(|_| Level12Error::DimensionTooLarge(row_count))?;
    let columns = u32::try_from(source_masks.len())
        .map_err(|_| Level12Error::DimensionTooLarge(source_masks.len()))?;

    // All nineteen inventory matrices average fewer than thirteen entries per
    // source column. Reserving that measured corpus bound avoids the doubling
    // reallocations that a five-entry estimate caused on the three large labels.
    let mut entries = Vec::with_capacity(
        source_masks
            .len()
            .saturating_mul(MEASURED_ENTRIES_PER_SOURCE_COLUMN),
    );
    for (column, &mask) in source_masks.iter().enumerate() {
        for root in 0..ROOTS.len() {
            for (lower, &upper) in raise[root].iter().enumerate() {
                if upper < 0 || mask & (1_u32 << lower) == 0 || mask & (1_u32 << upper as u32) != 0
                {
                    continue;
                }
                let upper = upper as u32;
                let output = (mask ^ (1_u32 << lower)) | (1_u32 << upper);
                let low = (lower as u32).min(upper);
                let high = (lower as u32).max(upper);
                let interval = if high == low + 1 {
                    0
                } else {
                    ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
                };
                let coefficient = if (mask & interval).count_ones() % 2 == 0 {
                    1
                } else {
                    -1
                };
                let row = *blocks[root]
                    .get(&output)
                    .expect("raising output must belong to its weight block");
                entries.push(Triplet {
                    row,
                    column: column as u32,
                    coefficient,
                });
            }
        }
    }
    let raising = CsrMatrix::from_triplets(rows, columns, entries)?;
    Ok(Level12Matrix {
        label: label.to_owned(),
        source_masks,
        raising,
    })
}

fn spinor_weights() -> [Weight; 32] {
    std::array::from_fn(|index| {
        std::array::from_fn(|axis| {
            if (index >> (4 - axis)) & 1 == 0 {
                1
            } else {
                -1
            }
        })
    })
}

fn raising_table(weights: &[Weight; 32]) -> [[i8; 32]; 5] {
    let indices: HashMap<Weight, i8> = weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| (weight, index as i8))
        .collect();
    std::array::from_fn(|root| {
        std::array::from_fn(|index| {
            let target = add_weights(weights[index], ROOTS[root]);
            indices.get(&target).copied().unwrap_or(-1)
        })
    })
}

fn highest_weight(label: &str) -> Result<Weight, Level12Error> {
    if label.len() != 5 || !label.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Level12Error::UnsupportedLabel(label.to_owned()));
    }
    let digits: Vec<i16> = label.bytes().map(|byte| i16::from(byte - b'0')).collect();
    Ok(std::array::from_fn(|index| {
        2 * digits[index..4].iter().sum::<i16>() + digits[4]
    }))
}

fn add_weights(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn half_groups(offset: usize, weights: &[Weight; 32]) -> HalfGroups {
    let mut groups = HashMap::<(u8, Weight), Vec<u16>>::new();
    for mask in 0_u32..(1_u32 << 16) {
        let mut weight = [0_i16; 5];
        let mut remainder = mask;
        while remainder != 0 {
            let bit = remainder.trailing_zeros() as usize;
            remainder &= remainder - 1;
            for axis in 0..5 {
                weight[axis] += weights[offset + bit][axis];
            }
        }
        groups
            .entry((mask.count_ones() as u8, weight))
            .or_default()
            .push(mask as u16);
    }
    groups
}

fn weight_basis(degree: u32, target: Weight, left: &HalfGroups, right: &HalfGroups) -> Vec<u32> {
    let mut result = Vec::new();
    let minimum_left = degree.saturating_sub(16);
    let maximum_left = degree.min(16);
    for left_degree in minimum_left..=maximum_left {
        let right_degree = degree - left_degree;
        for (&(candidate_degree, left_weight), left_masks) in left {
            if u32::from(candidate_degree) != left_degree {
                continue;
            }
            let needed = std::array::from_fn(|axis| target[axis] - left_weight[axis]);
            if let Some(right_masks) = right.get(&(right_degree as u8, needed)) {
                for &left_mask in left_masks {
                    for &right_mask in right_masks {
                        result.push(u32::from(left_mask) | (u32::from(right_mask) << 16));
                    }
                }
            }
        }
    }
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_from_i64;
    use std::fs;
    use std::path::PathBuf;

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn read_i16_kernel(relative_path: &str) -> Vec<u32> {
        let bytes = fs::read(repository_root().join(relative_path)).unwrap();
        assert_eq!(bytes.len() % 2, 0);
        bytes
            .chunks_exact(2)
            .map(|chunk| field_from_i64(i64::from(i16::from_le_bytes([chunk[0], chunk[1]]))))
            .collect()
    }

    #[test]
    fn label_30002_matches_published_matrix_shape_and_kernel() {
        let level12 = build_level12_matrix("30002").unwrap();
        assert_eq!(level12.raising.columns(), 2_892);
        assert_eq!(level12.raising.rows(), 6_169);
        assert_eq!(level12.raising.nonzeros(), 19_549);
        assert!(
            level12
                .source_masks
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert_eq!(
            level12.canonical_sha256(),
            "0d6012f3735d4696f3f7e1fdec2e146a8d1ba08cdd8779d859f41653cdc382a6"
        );

        let kernel = read_i16_kernel(
            "data/eleven_dimensional_spinor_bridge/level12_30002_highest_weight_kernel_1.i16le",
        );
        assert_eq!(kernel.len(), level12.raising.columns() as usize);
        let residual = level12.raising.spmv(&kernel).unwrap();
        assert!(residual.iter().all(|&value| value == 0));
    }

    #[test]
    fn source_labeled_digest_binds_ordered_masks_without_changing_numeric_digest() {
        let level12 = build_level12_matrix("30002").unwrap();
        let numeric_digest = level12.canonical_sha256();
        let source_labeled_digest = level12.source_labeled_sha256();
        assert_eq!(
            source_labeled_digest,
            "4a0c2ce6b463fa6eded11aaabe2833c9f69b3dd3fba37493469609bbf48b01d5"
        );

        let mut relabeled = level12.clone();
        relabeled.source_masks.swap(0, 1);
        assert_eq!(relabeled.canonical_sha256(), numeric_digest);
        assert_ne!(relabeled.source_labeled_sha256(), source_labeled_digest);

        relabeled.source_masks.swap(0, 1);
        assert_eq!(relabeled.source_labeled_sha256(), source_labeled_digest);
    }

    #[test]
    fn rejects_labels_outside_the_published_inventory() {
        assert!(matches!(
            build_level12_matrix("99999"),
            Err(Level12Error::UnsupportedLabel(_))
        ));
    }
}
