//! Exact dense block primitives for 32-lane finite-field Krylov methods.
//!
//! Active coordinates are always the leading `0..active_width` rows and
//! columns. Inactive storage is pinned to zero. Rank profiles select original
//! row and column coordinates, so singular blocks retain an auditable partial
//! inverse on a deterministic nonsingular minor.

use crate::{PRIME, field_add, field_mul, field_sub};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const BLOCK32_DIMENSION: usize = 32;
const BLOCK32_ENTRIES: usize = BLOCK32_DIMENSION * BLOCK32_DIMENSION;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Block32Error {
    ActiveWidthOutOfRange(usize),
    ActiveEntryCount {
        width: usize,
        expected: usize,
        actual: usize,
    },
    CoordinateOutOfRange {
        row: usize,
        column: usize,
        width: usize,
    },
    NonCanonicalEntry {
        row: usize,
        column: usize,
        value: u32,
    },
    InactiveEntryNonzero {
        row: usize,
        column: usize,
        value: u32,
        width: usize,
    },
    WidthMismatch {
        left: usize,
        right: usize,
    },
    NotSymmetric,
    InvalidRankProfile(&'static str),
    SelectedMinorSingular,
    SelectedInverseInvariant,
}

impl Display for Block32Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ActiveWidthOutOfRange(width) => {
                write!(
                    formatter,
                    "active width {width} exceeds {BLOCK32_DIMENSION}"
                )
            }
            Self::ActiveEntryCount {
                width,
                expected,
                actual,
            } => write!(
                formatter,
                "active width {width} requires {expected} entries, got {actual}"
            ),
            Self::CoordinateOutOfRange { row, column, width } => write!(
                formatter,
                "coordinate ({row}, {column}) is outside active width {width}"
            ),
            Self::NonCanonicalEntry { row, column, value } => write!(
                formatter,
                "entry ({row}, {column}) is not canonical: {value} >= {PRIME}"
            ),
            Self::InactiveEntryNonzero {
                row,
                column,
                value,
                width,
            } => write!(
                formatter,
                "inactive entry ({row}, {column}) is {value} at active width {width}"
            ),
            Self::WidthMismatch { left, right } => {
                write!(formatter, "active width mismatch: {left} versus {right}")
            }
            Self::NotSymmetric => write!(formatter, "active block is not symmetric"),
            Self::InvalidRankProfile(detail) => write!(formatter, "invalid rank profile: {detail}"),
            Self::SelectedMinorSingular => write!(formatter, "selected rank minor is singular"),
            Self::SelectedInverseInvariant => {
                write!(formatter, "selected-subspace inverse invariant failed")
            }
        }
    }
}

impl Error for Block32Error {}

/// Row-major exact matrix with at most 32 active coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockMatrix32 {
    active_width: u8,
    entries: [u32; BLOCK32_ENTRIES],
}

impl BlockMatrix32 {
    pub fn zero(active_width: usize) -> Result<Self, Block32Error> {
        Ok(Self {
            active_width: checked_width(active_width)?,
            entries: [0; BLOCK32_ENTRIES],
        })
    }

    pub fn identity(active_width: usize) -> Result<Self, Block32Error> {
        let mut result = Self::zero(active_width)?;
        for coordinate in 0..active_width {
            result.entries[storage_index(coordinate, coordinate)] = 1;
        }
        Ok(result)
    }

    /// Construct from a row-major `active_width * active_width` slice.
    pub fn from_active_entries(
        active_width: usize,
        active_entries: &[u32],
    ) -> Result<Self, Block32Error> {
        checked_width(active_width)?;
        let expected = active_width
            .checked_mul(active_width)
            .ok_or(Block32Error::ActiveWidthOutOfRange(active_width))?;
        if active_entries.len() != expected {
            return Err(Block32Error::ActiveEntryCount {
                width: active_width,
                expected,
                actual: active_entries.len(),
            });
        }
        let mut result = Self::zero(active_width)?;
        for row in 0..active_width {
            for column in 0..active_width {
                let value = active_entries[row * active_width + column];
                if value >= PRIME {
                    return Err(Block32Error::NonCanonicalEntry { row, column, value });
                }
                result.entries[storage_index(row, column)] = value;
            }
        }
        Ok(result)
    }

    pub fn from_fn(
        active_width: usize,
        mut entry: impl FnMut(usize, usize) -> u32,
    ) -> Result<Self, Block32Error> {
        let mut result = Self::zero(active_width)?;
        for row in 0..active_width {
            for column in 0..active_width {
                let value = entry(row, column);
                if value >= PRIME {
                    return Err(Block32Error::NonCanonicalEntry { row, column, value });
                }
                result.entries[storage_index(row, column)] = value;
            }
        }
        Ok(result)
    }

    pub fn active_width(&self) -> usize {
        usize::from(self.active_width)
    }

    pub fn get(&self, row: usize, column: usize) -> Option<u32> {
        (row < self.active_width() && column < self.active_width())
            .then(|| self.entries[storage_index(row, column)])
    }

    pub fn set(&mut self, row: usize, column: usize, value: u32) -> Result<(), Block32Error> {
        let width = self.active_width();
        if row >= width || column >= width {
            return Err(Block32Error::CoordinateOutOfRange { row, column, width });
        }
        if value >= PRIME {
            return Err(Block32Error::NonCanonicalEntry { row, column, value });
        }
        self.entries[storage_index(row, column)] = value;
        Ok(())
    }

    pub fn active_entries_row_major(&self) -> Vec<u32> {
        let width = self.active_width();
        let mut result = Vec::with_capacity(width * width);
        for row in 0..width {
            result.extend_from_slice(
                &self.entries[storage_index(row, 0)..storage_index(row, 0) + width],
            );
        }
        result
    }

    pub fn validate(&self) -> Result<(), Block32Error> {
        let width = self.active_width();
        if width > BLOCK32_DIMENSION {
            return Err(Block32Error::ActiveWidthOutOfRange(width));
        }
        for row in 0..BLOCK32_DIMENSION {
            for column in 0..BLOCK32_DIMENSION {
                let value = self.entries[storage_index(row, column)];
                if value >= PRIME {
                    return Err(Block32Error::NonCanonicalEntry { row, column, value });
                }
                if (row >= width || column >= width) && value != 0 {
                    return Err(Block32Error::InactiveEntryNonzero {
                        row,
                        column,
                        value,
                        width,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn transpose(&self) -> Self {
        let width = self.active_width();
        let mut result = Self::zero(width).expect("an existing active width is valid");
        for row in 0..width {
            for column in 0..width {
                result.entries[storage_index(column, row)] =
                    self.entries[storage_index(row, column)];
            }
        }
        result
    }

    pub fn multiply(&self, right: &Self) -> Result<Self, Block32Error> {
        let width = self.active_width();
        if width != right.active_width() {
            return Err(Block32Error::WidthMismatch {
                left: width,
                right: right.active_width(),
            });
        }
        let mut result = Self::zero(width)?;
        for row in 0..width {
            for inner in 0..width {
                let left_value = self.entries[storage_index(row, inner)];
                if left_value == 0 {
                    continue;
                }
                for column in 0..width {
                    let right_value = right.entries[storage_index(inner, column)];
                    if right_value == 0 {
                        continue;
                    }
                    let destination = storage_index(row, column);
                    result.entries[destination] = field_add(
                        result.entries[destination],
                        field_mul(left_value, right_value),
                    );
                }
            }
        }
        Ok(result)
    }

    /// Deterministic column-first rank profile. Pivot columns increase, and the
    /// first surviving row in the current stable row order is selected.
    pub fn rank_profile(&self) -> RankProfile32 {
        let width = self.active_width();
        let mut work = self.clone();
        let mut original_rows = std::array::from_fn::<_, BLOCK32_DIMENSION, _>(|index| index as u8);
        let mut pivot_rows = [0_u8; BLOCK32_DIMENSION];
        let mut pivot_columns = [0_u8; BLOCK32_DIMENSION];
        let mut rank = 0_usize;

        for column in 0..width {
            let Some(pivot_row) =
                (rank..width).find(|&row| work.entries[storage_index(row, column)] != 0)
            else {
                continue;
            };
            move_row_to(&mut work.entries, width, pivot_row, rank);
            let selected_original_row = original_rows[pivot_row];
            original_rows.copy_within(rank..pivot_row, rank + 1);
            original_rows[rank] = selected_original_row;
            pivot_rows[rank] = original_rows[rank];
            pivot_columns[rank] = column as u8;

            let pivot = work.entries[storage_index(rank, column)];
            let inverse = field_inverse(pivot);
            for active_column in column..width {
                let index = storage_index(rank, active_column);
                work.entries[index] = field_mul(work.entries[index], inverse);
            }
            for row in (rank + 1)..width {
                let scale = work.entries[storage_index(row, column)];
                if scale == 0 {
                    continue;
                }
                for active_column in column..width {
                    let index = storage_index(row, active_column);
                    work.entries[index] = field_sub(
                        work.entries[index],
                        field_mul(scale, work.entries[storage_index(rank, active_column)]),
                    );
                }
            }
            rank += 1;
            if rank == width {
                break;
            }
        }

        RankProfile32 {
            active_width: self.active_width,
            rank: rank as u8,
            pivot_rows,
            pivot_columns,
        }
    }

    pub fn selected_subspace_inverse(&self) -> Result<SelectedSubspaceInverse32, Block32Error> {
        let profile = self.rank_profile();
        self.selected_subspace_inverse_for(profile)
    }

    pub fn selected_subspace_inverse_for(
        &self,
        profile: RankProfile32,
    ) -> Result<SelectedSubspaceInverse32, Block32Error> {
        profile.validate()?;
        if profile.active_width() != self.active_width() {
            return Err(Block32Error::WidthMismatch {
                left: self.active_width(),
                right: profile.active_width(),
            });
        }
        let rank = profile.rank();
        let mut selected = Self::zero(rank)?;
        for selected_row in 0..rank {
            for selected_column in 0..rank {
                selected.entries[storage_index(selected_row, selected_column)] = self.entries
                    [storage_index(
                        usize::from(profile.pivot_rows[selected_row]),
                        usize::from(profile.pivot_columns[selected_column]),
                    )];
            }
        }
        let inverse = invert_nonsingular(&selected)?;
        let result = SelectedSubspaceInverse32 { profile, inverse };
        result.validate_against(self)?;
        Ok(result)
    }

    /// Rank profile for a symmetric block using deterministic congruence
    /// elimination with one-coordinate and two-coordinate pivots. The returned
    /// row and column selections are identical and name a maximum-rank
    /// nonsingular principal submatrix.
    pub fn symmetric_rank_profile(&self) -> Result<RankProfile32, Block32Error> {
        if self != &self.transpose() {
            return Err(Block32Error::NotSymmetric);
        }
        let width = self.active_width();
        let mut work = self.clone();
        let mut original_coordinates =
            std::array::from_fn::<_, BLOCK32_DIMENSION, _>(|index| index as u8);
        let mut selected = [0_u8; BLOCK32_DIMENSION];
        let mut rank = 0_usize;

        while rank < width {
            if let Some(pivot) =
                (rank..width).find(|&index| work.entries[storage_index(index, index)] != 0)
            {
                move_symmetric_coordinate_to(
                    &mut work.entries,
                    width,
                    &mut original_coordinates,
                    pivot,
                    rank,
                );
                selected[rank] = original_coordinates[rank];
                let inverse = field_inverse(work.entries[storage_index(rank, rank)]);
                for row in (rank + 1)..width {
                    for column in (rank + 1)..width {
                        let correction = field_mul(
                            field_mul(work.entries[storage_index(row, rank)], inverse),
                            work.entries[storage_index(rank, column)],
                        );
                        let index = storage_index(row, column);
                        work.entries[index] = field_sub(work.entries[index], correction);
                    }
                }
                rank += 1;
                continue;
            }

            let mut pair = None;
            'search: for left in rank..width {
                for right in (left + 1)..width {
                    if work.entries[storage_index(left, right)] != 0 {
                        pair = Some((left, right));
                        break 'search;
                    }
                }
            }
            let Some((left, right)) = pair else {
                break;
            };
            move_symmetric_coordinate_to(
                &mut work.entries,
                width,
                &mut original_coordinates,
                left,
                rank,
            );
            // Since `left < right`, moving `left` earlier does not change the
            // current position of `right`.
            move_symmetric_coordinate_to(
                &mut work.entries,
                width,
                &mut original_coordinates,
                right,
                rank + 1,
            );
            selected[rank] = original_coordinates[rank];
            selected[rank + 1] = original_coordinates[rank + 1];
            let off_diagonal = work.entries[storage_index(rank, rank + 1)];
            let inverse = field_inverse(off_diagonal);
            for row in (rank + 2)..width {
                for column in (rank + 2)..width {
                    let first = field_mul(
                        field_mul(work.entries[storage_index(row, rank)], inverse),
                        work.entries[storage_index(rank + 1, column)],
                    );
                    let second = field_mul(
                        field_mul(work.entries[storage_index(row, rank + 1)], inverse),
                        work.entries[storage_index(rank, column)],
                    );
                    let index = storage_index(row, column);
                    work.entries[index] = field_sub(work.entries[index], field_add(first, second));
                }
            }
            rank += 2;
        }

        selected[..rank].sort_unstable();
        let mut profile = RankProfile32 {
            active_width: self.active_width,
            rank: rank as u8,
            pivot_rows: [0; BLOCK32_DIMENSION],
            pivot_columns: [0; BLOCK32_DIMENSION],
        };
        profile.pivot_rows[..rank].copy_from_slice(&selected[..rank]);
        profile.pivot_columns[..rank].copy_from_slice(&selected[..rank]);
        profile.validate()?;
        Ok(profile)
    }

    pub fn symmetric_selected_subspace_inverse(
        &self,
    ) -> Result<SelectedSubspaceInverse32, Block32Error> {
        self.selected_subspace_inverse_for(self.symmetric_rank_profile()?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankProfile32 {
    active_width: u8,
    rank: u8,
    pivot_rows: [u8; BLOCK32_DIMENSION],
    pivot_columns: [u8; BLOCK32_DIMENSION],
}

impl RankProfile32 {
    pub fn active_width(&self) -> usize {
        usize::from(self.active_width)
    }

    pub fn rank(&self) -> usize {
        usize::from(self.rank)
    }

    pub fn pivot_rows(&self) -> &[u8] {
        &self.pivot_rows[..self.rank()]
    }

    pub fn pivot_columns(&self) -> &[u8] {
        &self.pivot_columns[..self.rank()]
    }

    pub fn validate(&self) -> Result<(), Block32Error> {
        let width = self.active_width();
        let rank = self.rank();
        if width > BLOCK32_DIMENSION || rank > width {
            return Err(Block32Error::InvalidRankProfile(
                "rank or width exceeds the active block",
            ));
        }
        let mut seen_rows = [false; BLOCK32_DIMENSION];
        let mut previous_column = None;
        for index in 0..rank {
            let row = usize::from(self.pivot_rows[index]);
            let column = usize::from(self.pivot_columns[index]);
            if row >= width || column >= width {
                return Err(Block32Error::InvalidRankProfile(
                    "selected coordinate is outside the active block",
                ));
            }
            if seen_rows[row] {
                return Err(Block32Error::InvalidRankProfile(
                    "selected rows are not unique",
                ));
            }
            if previous_column.is_some_and(|previous| column <= previous) {
                return Err(Block32Error::InvalidRankProfile(
                    "selected columns are not strictly increasing",
                ));
            }
            seen_rows[row] = true;
            previous_column = Some(column);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedSubspaceInverse32 {
    profile: RankProfile32,
    /// Inverse of `A[pivot_rows, pivot_columns]` in local rank coordinates.
    inverse: BlockMatrix32,
}

impl SelectedSubspaceInverse32 {
    pub fn profile(&self) -> RankProfile32 {
        self.profile
    }

    pub fn inverse(&self) -> &BlockMatrix32 {
        &self.inverse
    }

    /// Embed the selected inverse into original block coordinates. If
    /// `M = A[R,C]`, the result `B` has `B[C,R] = M^-1` and zero elsewhere.
    pub fn partial_inverse(&self) -> Result<BlockMatrix32, Block32Error> {
        self.profile.validate()?;
        let mut result = BlockMatrix32::zero(self.profile.active_width())?;
        for inverse_row in 0..self.profile.rank() {
            for inverse_column in 0..self.profile.rank() {
                let row = usize::from(self.profile.pivot_columns[inverse_row]);
                let column = usize::from(self.profile.pivot_rows[inverse_column]);
                result.entries[storage_index(row, column)] =
                    self.inverse.entries[storage_index(inverse_row, inverse_column)];
            }
        }
        Ok(result)
    }

    pub fn validate_against(&self, matrix: &BlockMatrix32) -> Result<(), Block32Error> {
        self.profile.validate()?;
        if matrix.active_width() != self.profile.active_width()
            || self.inverse.active_width() != self.profile.rank()
        {
            return Err(Block32Error::SelectedInverseInvariant);
        }
        let rank = self.profile.rank();
        let mut selected = BlockMatrix32::zero(rank)?;
        for row in 0..rank {
            for column in 0..rank {
                selected.entries[storage_index(row, column)] = matrix.entries[storage_index(
                    usize::from(self.profile.pivot_rows[row]),
                    usize::from(self.profile.pivot_columns[column]),
                )];
            }
        }
        let identity = BlockMatrix32::identity(rank)?;
        if selected.multiply(&self.inverse)? != identity
            || self.inverse.multiply(&selected)? != identity
        {
            return Err(Block32Error::SelectedInverseInvariant);
        }

        let partial = self.partial_inverse()?;
        let left_projection = matrix.multiply(&partial)?;
        let right_projection = partial.multiply(matrix)?;
        for row in 0..rank {
            for column in 0..rank {
                let expected = u32::from(row == column);
                let selected_row = usize::from(self.profile.pivot_rows[row]);
                let selected_other_row = usize::from(self.profile.pivot_rows[column]);
                let selected_column = usize::from(self.profile.pivot_columns[row]);
                let selected_other_column = usize::from(self.profile.pivot_columns[column]);
                if left_projection.entries[storage_index(selected_row, selected_other_row)]
                    != expected
                    || right_projection.entries
                        [storage_index(selected_column, selected_other_column)]
                        != expected
                {
                    return Err(Block32Error::SelectedInverseInvariant);
                }
            }
        }
        Ok(())
    }
}

fn checked_width(active_width: usize) -> Result<u8, Block32Error> {
    if active_width <= BLOCK32_DIMENSION {
        Ok(active_width as u8)
    } else {
        Err(Block32Error::ActiveWidthOutOfRange(active_width))
    }
}

#[inline]
const fn storage_index(row: usize, column: usize) -> usize {
    row * BLOCK32_DIMENSION + column
}

fn swap_rows(entries: &mut [u32; BLOCK32_ENTRIES], width: usize, left: usize, right: usize) {
    if left != right {
        for column in 0..width {
            entries.swap(storage_index(left, column), storage_index(right, column));
        }
    }
}

fn move_row_to(entries: &mut [u32; BLOCK32_ENTRIES], width: usize, from: usize, to: usize) {
    debug_assert!(to <= from && from < width);
    if from == to {
        return;
    }
    let mut selected = [0_u32; BLOCK32_DIMENSION];
    for column in 0..width {
        selected[column] = entries[storage_index(from, column)];
    }
    for row in (to + 1..=from).rev() {
        for column in 0..width {
            entries[storage_index(row, column)] = entries[storage_index(row - 1, column)];
        }
    }
    for column in 0..width {
        entries[storage_index(to, column)] = selected[column];
    }
}

fn move_symmetric_coordinate_to(
    entries: &mut [u32; BLOCK32_ENTRIES],
    width: usize,
    original_coordinates: &mut [u8; BLOCK32_DIMENSION],
    from: usize,
    to: usize,
) {
    debug_assert!(to <= from && from < width);
    if from == to {
        return;
    }
    let previous_entries = *entries;
    let previous_coordinates = *original_coordinates;
    let old_coordinate = |coordinate: usize| {
        if coordinate == to {
            from
        } else if coordinate > to && coordinate <= from {
            coordinate - 1
        } else {
            coordinate
        }
    };
    for row in 0..width {
        for column in 0..width {
            entries[storage_index(row, column)] =
                previous_entries[storage_index(old_coordinate(row), old_coordinate(column))];
        }
        original_coordinates[row] = previous_coordinates[old_coordinate(row)];
    }
}

fn invert_nonsingular(matrix: &BlockMatrix32) -> Result<BlockMatrix32, Block32Error> {
    let width = matrix.active_width();
    let mut left = matrix.clone();
    let mut right = BlockMatrix32::identity(width)?;
    for column in 0..width {
        let Some(pivot_row) =
            (column..width).find(|&row| left.entries[storage_index(row, column)] != 0)
        else {
            return Err(Block32Error::SelectedMinorSingular);
        };
        swap_rows(&mut left.entries, width, column, pivot_row);
        swap_rows(&mut right.entries, width, column, pivot_row);
        let inverse = field_inverse(left.entries[storage_index(column, column)]);
        for active_column in 0..width {
            let left_index = storage_index(column, active_column);
            let right_index = storage_index(column, active_column);
            left.entries[left_index] = field_mul(left.entries[left_index], inverse);
            right.entries[right_index] = field_mul(right.entries[right_index], inverse);
        }
        for row in 0..width {
            if row == column {
                continue;
            }
            let scale = left.entries[storage_index(row, column)];
            if scale == 0 {
                continue;
            }
            for active_column in 0..width {
                let left_index = storage_index(row, active_column);
                let right_index = storage_index(row, active_column);
                left.entries[left_index] = field_sub(
                    left.entries[left_index],
                    field_mul(scale, left.entries[storage_index(column, active_column)]),
                );
                right.entries[right_index] = field_sub(
                    right.entries[right_index],
                    field_mul(scale, right.entries[storage_index(column, active_column)]),
                );
            }
        }
    }
    Ok(right)
}

fn field_inverse(value: u32) -> u32 {
    debug_assert!(value > 0 && value < PRIME);
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

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    type Dense = Vec<Vec<u32>>;

    fn dense(matrix: &BlockMatrix32) -> Dense {
        let width = matrix.active_width();
        (0..width)
            .map(|row| {
                (0..width)
                    .map(|column| matrix.get(row, column).unwrap())
                    .collect()
            })
            .collect()
    }

    fn dense_rank(mut matrix: Dense) -> usize {
        let rows = matrix.len();
        let columns = matrix.first().map_or(0, Vec::len);
        let mut rank = 0;
        for column in 0..columns {
            let Some(pivot) = (rank..rows).find(|&row| matrix[row][column] != 0) else {
                continue;
            };
            matrix.swap(rank, pivot);
            let inverse = field_inverse(matrix[rank][column]);
            for active_column in column..columns {
                matrix[rank][active_column] = field_mul(matrix[rank][active_column], inverse);
            }
            let pivot_row = matrix[rank].clone();
            for row in (rank + 1)..rows {
                let scale = matrix[row][column];
                for active_column in column..columns {
                    matrix[row][active_column] = field_sub(
                        matrix[row][active_column],
                        field_mul(scale, pivot_row[active_column]),
                    );
                }
            }
            rank += 1;
            if rank == rows {
                break;
            }
        }
        rank
    }

    fn dense_transpose(matrix: &Dense) -> Dense {
        let width = matrix.len();
        (0..width)
            .map(|column| (0..width).map(|row| matrix[row][column]).collect())
            .collect()
    }

    fn dense_multiply(left: &Dense, right: &Dense) -> Dense {
        let width = left.len();
        let mut result = vec![vec![0; width]; width];
        for row in 0..width {
            for column in 0..width {
                for inner in 0..width {
                    result[row][column] = field_add(
                        result[row][column],
                        field_mul(left[row][inner], right[inner][column]),
                    );
                }
            }
        }
        result
    }

    fn next(state: &mut u64) -> u32 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value % u64::from(PRIME)) as u32
    }

    fn known_rank_matrix(rank: usize, state: &mut u64) -> BlockMatrix32 {
        let width = BLOCK32_DIMENSION;
        let mut left = vec![vec![0_u32; rank]; width];
        let mut right = vec![vec![0_u32; width]; rank];
        for coordinate in 0..rank {
            left[coordinate][coordinate] = 1;
            right[coordinate][coordinate] = 1;
        }
        for row in rank..width {
            for column in 0..rank {
                left[row][column] = next(state);
            }
        }
        for row in 0..rank {
            for column in rank..width {
                right[row][column] = next(state);
            }
        }
        BlockMatrix32::from_fn(width, |row, column| {
            (0..rank).fold(0, |sum, inner| {
                field_add(sum, field_mul(left[row][inner], right[inner][column]))
            })
        })
        .unwrap()
    }

    fn known_rank_symmetric_matrix(rank: usize, state: &mut u64) -> BlockMatrix32 {
        let width = BLOCK32_DIMENSION;
        let mut lower = vec![vec![0_u32; width]; width];
        for row in 0..width {
            lower[row][row] = 1;
            for column in 0..row {
                lower[row][column] = next(state);
            }
        }
        BlockMatrix32::from_fn(width, |row, column| {
            (0..rank).fold(0, |sum, inner| {
                field_add(sum, field_mul(lower[row][inner], lower[column][inner]))
            })
        })
        .unwrap()
    }

    #[test]
    fn exhaustive_binary_matrices_through_active_width_four() {
        for width in 0_usize..=4 {
            let bits = width * width;
            let matrices = 1_u64 << bits;
            for mask in 0..matrices {
                let matrix = BlockMatrix32::from_fn(width, |row, column| {
                    ((mask >> (row * width + column)) & 1) as u32
                })
                .unwrap();
                matrix.validate().unwrap();
                let profile = matrix.rank_profile();
                assert_eq!(profile.rank(), dense_rank(dense(&matrix)));
                profile.validate().unwrap();
                let selected = matrix.selected_subspace_inverse_for(profile).unwrap();
                selected.validate_against(&matrix).unwrap();
                selected.partial_inverse().unwrap().validate().unwrap();
            }
        }
    }

    #[test]
    fn every_active_width_handles_zero_identity_transpose_and_multiply() {
        for width in 0..=BLOCK32_DIMENSION {
            let zero = BlockMatrix32::zero(width).unwrap();
            let identity = BlockMatrix32::identity(width).unwrap();
            assert_eq!(zero.rank_profile().rank(), 0);
            assert_eq!(identity.rank_profile().rank(), width);
            assert_eq!(identity.transpose(), identity);
            assert_eq!(identity.multiply(&identity).unwrap(), identity);
            assert_eq!(zero.multiply(&identity).unwrap(), zero);
            identity
                .selected_subspace_inverse()
                .unwrap()
                .validate_against(&identity)
                .unwrap();
        }
    }

    #[test]
    fn exhaustive_symmetric_binary_matrices_through_active_width_five() {
        for width in 0_usize..=5 {
            let bits = width * (width + 1) / 2;
            for mask in 0..(1_u64 << bits) {
                let matrix = BlockMatrix32::from_fn(width, |row, column| {
                    let (low, high) = if row <= column {
                        (row, column)
                    } else {
                        (column, row)
                    };
                    let bit = low * width - low.saturating_sub(1) * low / 2 + (high - low);
                    ((mask >> bit) & 1) as u32
                })
                .unwrap();
                let reference_rank = dense_rank(dense(&matrix));
                let profile = matrix.symmetric_rank_profile().unwrap();
                assert_eq!(profile.rank(), reference_rank);
                assert_eq!(profile.pivot_rows(), profile.pivot_columns());
                let selected = matrix.selected_subspace_inverse_for(profile).unwrap();
                selected.validate_against(&matrix).unwrap();
                assert_eq!(
                    selected.partial_inverse().unwrap(),
                    selected.partial_inverse().unwrap().transpose()
                );
            }
        }
    }

    #[test]
    fn randomized_dense_reference_covers_every_rank() {
        let mut state = 0x243f_6a88_85a3_08d3;
        for rank in 0..=BLOCK32_DIMENSION {
            for _ in 0..3 {
                let matrix = known_rank_matrix(rank, &mut state);
                let reference = dense(&matrix);
                assert_eq!(dense_rank(reference.clone()), rank);
                assert_eq!(matrix.rank_profile().rank(), rank);
                assert_eq!(dense(&matrix.transpose()), dense_transpose(&reference));
                assert_eq!(
                    dense(&matrix.multiply(&matrix.transpose()).unwrap()),
                    dense_multiply(&reference, &dense_transpose(&reference))
                );
                let selected = matrix.selected_subspace_inverse().unwrap();
                assert_eq!(selected.profile().rank(), rank);
                selected.validate_against(&matrix).unwrap();

                let symmetric = known_rank_symmetric_matrix(rank, &mut state);
                let symmetric_profile = symmetric.symmetric_rank_profile().unwrap();
                assert_eq!(symmetric_profile.rank(), rank);
                assert_eq!(
                    symmetric_profile.pivot_rows(),
                    symmetric_profile.pivot_columns()
                );
                symmetric
                    .symmetric_selected_subspace_inverse()
                    .unwrap()
                    .validate_against(&symmetric)
                    .unwrap();
            }
        }
    }

    #[test]
    fn singular_and_zero_diagonal_breakdown_cases_are_rank_revealing() {
        let swap = BlockMatrix32::from_active_entries(2, &[0, 1, 1, 0]).unwrap();
        assert_eq!(swap.rank_profile().rank(), 2);
        let symmetric_profile = swap.symmetric_rank_profile().unwrap();
        assert_eq!(symmetric_profile.pivot_rows(), &[0, 1]);
        assert_eq!(
            symmetric_profile.pivot_rows(),
            symmetric_profile.pivot_columns()
        );
        let inverse = swap.symmetric_selected_subspace_inverse().unwrap();
        assert_eq!(inverse.inverse(), &swap);
        assert_eq!(inverse.partial_inverse().unwrap(), swap);

        let duplicate = BlockMatrix32::from_active_entries(
            4,
            &[1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 0, 1, 0, 1, 0, 1],
        )
        .unwrap();
        assert_eq!(duplicate.rank_profile().rank(), 2);
        duplicate
            .selected_subspace_inverse()
            .unwrap()
            .validate_against(&duplicate)
            .unwrap();

        let zero = BlockMatrix32::zero(32).unwrap();
        let selected = zero.selected_subspace_inverse().unwrap();
        assert_eq!(selected.profile().rank(), 0);
        assert_eq!(selected.partial_inverse().unwrap(), zero);
    }

    #[test]
    fn pivot_selection_is_deterministic_in_original_coordinates() {
        let matrix = BlockMatrix32::from_active_entries(3, &[0, 1, 0, 1, 0, 0, 1, 1, 1]).unwrap();
        let first = matrix.rank_profile();
        let second = matrix.rank_profile();
        assert_eq!(first, second);
        assert_eq!(first.pivot_rows(), &[1, 0, 2]);
        assert_eq!(first.pivot_columns(), &[0, 1, 2]);
    }

    #[test]
    fn invariant_checks_reject_corrupt_storage_and_profiles() {
        assert!(matches!(
            BlockMatrix32::zero(33),
            Err(Block32Error::ActiveWidthOutOfRange(33))
        ));
        assert!(matches!(
            BlockMatrix32::from_active_entries(2, &[0; 3]),
            Err(Block32Error::ActiveEntryCount { .. })
        ));

        let mut noncanonical = BlockMatrix32::zero(1).unwrap();
        noncanonical.entries[0] = PRIME;
        assert!(matches!(
            noncanonical.validate(),
            Err(Block32Error::NonCanonicalEntry { .. })
        ));
        let mut inactive = BlockMatrix32::zero(1).unwrap();
        inactive.entries[storage_index(1, 0)] = 1;
        assert!(matches!(
            inactive.validate(),
            Err(Block32Error::InactiveEntryNonzero { .. })
        ));

        let mut setters = BlockMatrix32::zero(2).unwrap();
        setters.set(1, 0, 7).unwrap();
        assert_eq!(setters.active_entries_row_major(), [0, 0, 7, 0]);
        assert!(matches!(
            setters.set(2, 0, 1),
            Err(Block32Error::CoordinateOutOfRange { .. })
        ));
        assert!(matches!(
            setters.set(0, 0, PRIME),
            Err(Block32Error::NonCanonicalEntry { .. })
        ));

        let matrix = BlockMatrix32::identity(3).unwrap();
        let mut profile = matrix.rank_profile();
        profile.pivot_rows[1] = profile.pivot_rows[0];
        assert!(matches!(
            matrix.selected_subspace_inverse_for(profile),
            Err(Block32Error::InvalidRankProfile(_))
        ));

        let nonsymmetric = BlockMatrix32::from_active_entries(2, &[0, 1, 0, 0]).unwrap();
        assert_eq!(
            nonsymmetric.symmetric_rank_profile(),
            Err(Block32Error::NotSymmetric)
        );

        let singular = BlockMatrix32::from_active_entries(2, &[1, 0, 0, 0]).unwrap();
        let mut bad_minor = singular.rank_profile();
        bad_minor.pivot_rows[0] = 1;
        assert!(matches!(
            singular.selected_subspace_inverse_for(bad_minor),
            Err(Block32Error::SelectedMinorSingular)
        ));

        let mut corrupted_inverse = matrix.selected_subspace_inverse().unwrap();
        corrupted_inverse.inverse.entries[0] = 2;
        assert_eq!(
            corrupted_inverse.validate_against(&matrix),
            Err(Block32Error::SelectedInverseInvariant)
        );
    }
}
