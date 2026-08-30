//! Deterministic sparse row elimination over `GF(2^31 - 1)`.
//!
//! This is the exact, rank-revealing CPU path. Columns are never permuted, so
//! pivot and free-column indices remain in the matrix's original coordinates.
//! Rows are processed by increasing input width and then by original row index,
//! matching the deterministic ordering used by the level-12 Python generator.

use crate::{CsrMatrix, PRIME, field_add, field_from_i64, field_mul, field_sub};

type SparseFieldRow = Vec<(u32, u32)>;

/// Hard limits for the memory-driving parts of sparse elimination.
///
/// `max_fill_nonzeros` bounds the total number of field entries retained in
/// normalized pivot rows. Accepted rows are compacted, so this is also the
/// total retained pivot-vector capacity. `max_pivot_width` bounds every input,
/// intermediate, and normalized pivot row. A limit may be set to `usize::MAX`
/// to disable it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EliminationBudget {
    pub max_fill_nonzeros: usize,
    pub max_pivot_width: usize,
}

impl EliminationBudget {
    pub const fn unlimited() -> Self {
        Self {
            max_fill_nonzeros: usize::MAX,
            max_pivot_width: usize::MAX,
        }
    }
}

impl Default for EliminationBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// A normalized echelon row. The entry at `column` is always one and is the
/// first entry in `entries`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModularPivot {
    pub column: u32,
    pub entries: Vec<(u32, u32)>,
}

/// Complete modular rank and nullspace data in original column coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EliminationResult {
    pub rank: usize,
    pub pivots: Vec<ModularPivot>,
    pub free_columns: Vec<u32>,
    /// One dense modular vector per free column. Basis vector `i` is one at
    /// `free_columns[i]` and zero at every other free column.
    pub kernel_basis: Vec<Vec<u32>>,
    pub rows_processed: usize,
    pub row_reductions: usize,
    pub maximum_pivot_width: usize,
    /// Total pivot entries. Every retained pivot vector is compact, so this is
    /// also their total retained capacity in field-entry units.
    pub fill_nonzeros: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EliminationThresholdKind {
    PivotWidth,
    FillNonzeros,
}

/// Clean early-stop report. No partial result is presented as a rank
/// certificate because not every input row has been consumed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EliminationThreshold {
    pub kind: EliminationThresholdKind,
    pub source_row: u32,
    pub rows_processed: usize,
    pub partial_rank: usize,
    pub partial_pivot_columns: Vec<u32>,
    pub row_reductions: usize,
    pub maximum_pivot_width: usize,
    pub fill_nonzeros: usize,
    pub limit: usize,
    pub required: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EliminationOutcome {
    Complete(EliminationResult),
    ThresholdExceeded(EliminationThreshold),
}

/// Memory guard reached after row elimination but before allocating the dense
/// modular kernel basis. Rank and free columns are already final at this point,
/// but no dense vectors have been allocated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseKernelThreshold {
    pub rank: usize,
    pub free_columns: Vec<u32>,
    pub columns: usize,
    pub required_entries: usize,
    pub limit: usize,
}

/// Elimination outcome for callers that place a separate bound on the dense
/// kernel basis. This leaves the existing `eliminate` API and its row-threshold
/// semantics unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelBoundedEliminationOutcome {
    Complete(EliminationResult),
    EliminationThresholdExceeded(EliminationThreshold),
    DenseKernelThresholdExceeded(DenseKernelThreshold),
}

/// Compute a deterministic modular echelon basis and its canonical nullspace.
///
/// This routine intentionally uses the original column order. It is suitable
/// as the exact reference path and as a fast path while fill remains bounded.
/// Large fill returns `ThresholdExceeded` so the caller can checkpoint or use
/// a black-box solver without confusing partial rank with final rank.
pub fn eliminate(matrix: &CsrMatrix, budget: EliminationBudget) -> EliminationOutcome {
    match eliminate_with_kernel_entry_limit(matrix, budget, usize::MAX) {
        KernelBoundedEliminationOutcome::Complete(result) => EliminationOutcome::Complete(result),
        KernelBoundedEliminationOutcome::EliminationThresholdExceeded(threshold) => {
            EliminationOutcome::ThresholdExceeded(threshold)
        }
        KernelBoundedEliminationOutcome::DenseKernelThresholdExceeded(_) => {
            unreachable!("an unlimited dense-kernel budget cannot be exceeded")
        }
    }
}

/// Compute deterministic elimination with a separate bound on dense kernel
/// storage. `max_dense_kernel_entries` counts `u32` field entries and is checked
/// before any kernel vector is allocated.
pub fn eliminate_with_kernel_entry_limit(
    matrix: &CsrMatrix,
    budget: EliminationBudget,
    max_dense_kernel_entries: usize,
) -> KernelBoundedEliminationOutcome {
    let columns = matrix.columns() as usize;
    let row_offsets = matrix.row_offsets();
    let column_indices = matrix.column_indices();
    let coefficients = matrix.coefficients();

    let mut row_order = (0..matrix.rows())
        .map(|row| {
            let row_index = row as usize;
            let width = row_offsets[row_index + 1] - row_offsets[row_index];
            (width, row)
        })
        .collect::<Vec<_>>();
    row_order.sort_unstable();

    let mut pivots = vec![None::<SparseFieldRow>; columns];
    let mut rows_processed = 0_usize;
    let mut row_reductions = 0_usize;
    let mut maximum_pivot_width = 0_usize;
    let mut fill_nonzeros = 0_usize;
    let mut rank = 0_usize;
    let mut row = SparseFieldRow::new();
    let mut merge_buffer = SparseFieldRow::new();

    for (_, source_row) in row_order {
        let source_row_index = source_row as usize;
        let start = row_offsets[source_row_index] as usize;
        let end = row_offsets[source_row_index + 1] as usize;
        row.clear();
        reserve_exact_capacity(&mut row, end - start);
        for index in start..end {
            let value = field_from_i64(i64::from(coefficients[index]));
            if value != 0 {
                row.push((column_indices[index], value));
            }
        }

        if row.len() > budget.max_pivot_width {
            return KernelBoundedEliminationOutcome::EliminationThresholdExceeded(threshold(
                EliminationThresholdKind::PivotWidth,
                source_row,
                rows_processed,
                &pivots,
                row_reductions,
                maximum_pivot_width,
                fill_nonzeros,
                budget.max_pivot_width,
                row.len(),
            ));
        }

        loop {
            let Some(&(pivot_column, pivot_coefficient)) = row.first() else {
                rows_processed += 1;
                break;
            };
            let pivot_index = pivot_column as usize;
            if let Some(pivot) = pivots[pivot_index].as_ref() {
                row_reductions += 1;
                match subtract_scaled_into(
                    &row,
                    pivot,
                    pivot_coefficient,
                    budget.max_pivot_width,
                    &mut merge_buffer,
                ) {
                    Ok(()) => {
                        std::mem::swap(&mut row, &mut merge_buffer);
                        merge_buffer.clear();
                    }
                    Err(required) => {
                        return KernelBoundedEliminationOutcome::EliminationThresholdExceeded(
                            threshold(
                                EliminationThresholdKind::PivotWidth,
                                source_row,
                                rows_processed,
                                &pivots,
                                row_reductions,
                                maximum_pivot_width,
                                fill_nonzeros,
                                budget.max_pivot_width,
                                required,
                            ),
                        );
                    }
                }
                continue;
            }

            normalize_row(&mut row, pivot_coefficient);
            let Some(required_fill) = fill_nonzeros.checked_add(row.len()) else {
                return KernelBoundedEliminationOutcome::EliminationThresholdExceeded(threshold(
                    EliminationThresholdKind::FillNonzeros,
                    source_row,
                    rows_processed,
                    &pivots,
                    row_reductions,
                    maximum_pivot_width,
                    fill_nonzeros,
                    budget.max_fill_nonzeros,
                    usize::MAX,
                ));
            };
            if required_fill > budget.max_fill_nonzeros {
                return KernelBoundedEliminationOutcome::EliminationThresholdExceeded(threshold(
                    EliminationThresholdKind::FillNonzeros,
                    source_row,
                    rows_processed,
                    &pivots,
                    row_reductions,
                    maximum_pivot_width,
                    fill_nonzeros,
                    budget.max_fill_nonzeros,
                    required_fill,
                ));
            }
            maximum_pivot_width = maximum_pivot_width.max(row.len());
            fill_nonzeros = required_fill;
            let accepted = compact_row(std::mem::take(&mut row));
            debug_assert_eq!(accepted.len(), accepted.capacity());
            pivots[pivot_index] = Some(accepted);
            std::mem::swap(&mut row, &mut merge_buffer);
            rank += 1;
            rows_processed += 1;
            break;
        }
    }

    let free_columns = pivots
        .iter()
        .enumerate()
        .filter_map(|(column, pivot)| pivot.is_none().then_some(column as u32))
        .collect::<Vec<_>>();
    let required_kernel_entries = columns.saturating_mul(free_columns.len());
    if required_kernel_entries > max_dense_kernel_entries {
        return KernelBoundedEliminationOutcome::DenseKernelThresholdExceeded(
            DenseKernelThreshold {
                rank,
                free_columns,
                columns,
                required_entries: required_kernel_entries,
                limit: max_dense_kernel_entries,
            },
        );
    }
    let kernel_basis = build_kernel_basis(&pivots, columns, &free_columns);
    let pivots = pivots
        .into_iter()
        .enumerate()
        .filter_map(|(column, entries)| {
            entries.map(|entries| ModularPivot {
                column: column as u32,
                entries,
            })
        })
        .collect::<Vec<_>>();

    KernelBoundedEliminationOutcome::Complete(EliminationResult {
        rank,
        pivots,
        free_columns,
        kernel_basis,
        rows_processed,
        row_reductions,
        maximum_pivot_width,
        fill_nonzeros,
    })
}

fn reserve_exact_capacity(row: &mut SparseFieldRow, required: usize) {
    if row.capacity() < required {
        row.reserve_exact(required - row.len());
    }
}

fn compact_row(row: SparseFieldRow) -> SparseFieldRow {
    row.into_boxed_slice().into_vec()
}

fn normalize_row(row: &mut SparseFieldRow, leading_coefficient: u32) {
    debug_assert_ne!(leading_coefficient, 0);
    let inverse = field_inverse(leading_coefficient);
    if inverse != 1 {
        for (_, coefficient) in row {
            *coefficient = field_mul(*coefficient, inverse);
        }
    }
}

/// Return `row - scale * pivot`, preserving sorted unique column indices.
///
/// The merge stops as soon as its retained prefix proves that `width_limit`
/// cannot be met, avoiding an unbounded transient allocation before reporting
/// the configured threshold.
fn subtract_scaled_into(
    row: &SparseFieldRow,
    pivot: &SparseFieldRow,
    scale: u32,
    width_limit: usize,
    result: &mut SparseFieldRow,
) -> Result<(), usize> {
    debug_assert_ne!(scale, 0);
    debug_assert_eq!(pivot.first().map(|entry| entry.1), Some(1));
    debug_assert_eq!(
        row.first().map(|entry| entry.0),
        pivot.first().map(|entry| entry.0)
    );

    result.clear();
    let maximum_capacity = width_limit.saturating_add(1);
    let useful_capacity = row.len().saturating_add(pivot.len()).min(maximum_capacity);
    reserve_exact_capacity(result, useful_capacity);
    let mut left = 0_usize;
    let mut right = 0_usize;
    while left < row.len() || right < pivot.len() {
        if right == pivot.len() || (left < row.len() && row[left].0 < pivot[right].0) {
            result.push(row[left]);
            left += 1;
        } else if left == row.len() || pivot[right].0 < row[left].0 {
            let product = field_mul(scale, pivot[right].1);
            if product != 0 {
                result.push((pivot[right].0, field_neg(product)));
            }
            right += 1;
        } else {
            let value = field_sub(row[left].1, field_mul(scale, pivot[right].1));
            if value != 0 {
                result.push((row[left].0, value));
            }
            left += 1;
            right += 1;
        }
        if result.len() > width_limit {
            return Err(result.len());
        }
    }
    Ok(())
}

fn build_kernel_basis(
    pivots: &[Option<SparseFieldRow>],
    columns: usize,
    free_columns: &[u32],
) -> Vec<Vec<u32>> {
    free_columns
        .iter()
        .map(|&free_column| {
            let mut vector = vec![0_u32; columns];
            vector[free_column as usize] = 1;
            for pivot_column in (0..columns).rev() {
                let Some(row) = pivots[pivot_column].as_ref() else {
                    continue;
                };
                let mut sum = 0_u32;
                for &(column, coefficient) in row.iter().skip(1) {
                    sum = field_add(sum, field_mul(coefficient, vector[column as usize]));
                }
                vector[pivot_column] = field_neg(sum);
            }
            vector
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn threshold(
    kind: EliminationThresholdKind,
    source_row: u32,
    rows_processed: usize,
    pivots: &[Option<SparseFieldRow>],
    row_reductions: usize,
    maximum_pivot_width: usize,
    fill_nonzeros: usize,
    limit: usize,
    required: usize,
) -> EliminationThreshold {
    let partial_pivot_columns = pivots
        .iter()
        .enumerate()
        .filter_map(|(column, pivot)| pivot.is_some().then_some(column as u32))
        .collect::<Vec<_>>();
    EliminationThreshold {
        kind,
        source_row,
        rows_processed,
        partial_rank: partial_pivot_columns.len(),
        partial_pivot_columns,
        row_reductions,
        maximum_pivot_width,
        fill_nonzeros,
        limit,
        required,
    }
}

#[inline]
fn field_neg(value: u32) -> u32 {
    if value == 0 { 0 } else { PRIME - value }
}

fn field_inverse(value: u32) -> u32 {
    debug_assert!(value != 0 && value < PRIME);
    field_pow(value, u64::from(PRIME - 2))
}

fn field_pow(mut base: u32, mut exponent: u64) -> u32 {
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
    use super::*;
    use crate::Triplet;

    fn matrix(rows: u32, columns: u32, entries: &[(u32, u32, i32)]) -> CsrMatrix {
        CsrMatrix::from_triplets(
            rows,
            columns,
            entries
                .iter()
                .map(|&(row, column, coefficient)| Triplet {
                    row,
                    column,
                    coefficient,
                })
                .collect(),
        )
        .unwrap()
    }

    fn complete(matrix: &CsrMatrix) -> EliminationResult {
        match eliminate(matrix, EliminationBudget::unlimited()) {
            EliminationOutcome::Complete(result) => result,
            EliminationOutcome::ThresholdExceeded(threshold) => {
                panic!("unexpected threshold: {threshold:?}")
            }
        }
    }

    #[test]
    fn returns_rank_pivots_and_canonical_kernel_in_original_order() {
        let matrix = matrix(
            3,
            4,
            &[
                (0, 0, 1),
                (0, 1, 2),
                (0, 3, 1),
                (1, 1, 1),
                (1, 2, 1),
                (2, 0, 1),
                (2, 1, 3),
                (2, 2, 1),
                (2, 3, 1),
            ],
        );
        let result = complete(&matrix);

        assert_eq!(result.rank, 2);
        assert_eq!(
            result
                .pivots
                .iter()
                .map(|pivot| pivot.column)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(result.free_columns, [2, 3]);
        assert_eq!(result.kernel_basis[0], [2, PRIME - 1, 1, 0]);
        assert_eq!(result.kernel_basis[1], [PRIME - 1, 0, 0, 1]);
        for vector in &result.kernel_basis {
            assert_eq!(matrix.spmv(vector).unwrap(), vec![0; 3]);
        }
        assert!(
            result
                .pivots
                .iter()
                .all(|pivot| pivot.entries.len() == pivot.entries.capacity())
        );
        assert_eq!(
            result.fill_nonzeros,
            result
                .pivots
                .iter()
                .map(|pivot| pivot.entries.capacity())
                .sum::<usize>()
        );
    }

    #[test]
    fn normalizes_nonunit_coefficients() {
        let matrix = matrix(1, 1, &[(0, 0, 2)]);
        let result = complete(&matrix);
        assert_eq!(result.rank, 1);
        assert_eq!(result.pivots[0].entries, [(0, 1)]);
        assert!(result.free_columns.is_empty());
        assert!(result.kernel_basis.is_empty());
    }

    #[test]
    fn zero_matrix_returns_original_coordinate_basis() {
        let matrix = matrix(2, 3, &[]);
        let result = complete(&matrix);
        assert_eq!(result.rank, 0);
        assert_eq!(result.free_columns, [0, 1, 2]);
        assert_eq!(
            result.kernel_basis,
            [vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
        );
    }

    #[test]
    fn pivot_width_budget_stops_without_claiming_complete_rank() {
        let matrix = matrix(1, 3, &[(0, 0, 1), (0, 1, 1), (0, 2, 1)]);
        let outcome = eliminate(
            &matrix,
            EliminationBudget {
                max_fill_nonzeros: usize::MAX,
                max_pivot_width: 2,
            },
        );
        let EliminationOutcome::ThresholdExceeded(threshold) = outcome else {
            panic!("expected pivot-width threshold")
        };
        assert_eq!(threshold.kind, EliminationThresholdKind::PivotWidth);
        assert_eq!(threshold.source_row, 0);
        assert_eq!(threshold.rows_processed, 0);
        assert_eq!(threshold.partial_rank, 0);
        assert_eq!(threshold.limit, 2);
        assert_eq!(threshold.required, 3);
    }

    #[test]
    fn pivot_width_budget_stops_during_fill_producing_reduction() {
        let matrix = matrix(
            2,
            5,
            &[
                (0, 0, 1),
                (0, 3, 1),
                (0, 4, 1),
                (1, 0, 1),
                (1, 1, 1),
                (1, 2, 1),
            ],
        );
        let outcome = eliminate(
            &matrix,
            EliminationBudget {
                max_fill_nonzeros: usize::MAX,
                max_pivot_width: 3,
            },
        );
        let EliminationOutcome::ThresholdExceeded(threshold) = outcome else {
            panic!("expected intermediate pivot-width threshold")
        };
        assert_eq!(threshold.kind, EliminationThresholdKind::PivotWidth);
        assert_eq!(threshold.source_row, 1);
        assert_eq!(threshold.rows_processed, 1);
        assert_eq!(threshold.partial_pivot_columns, [0]);
        assert_eq!(threshold.limit, 3);
        assert_eq!(threshold.required, 4);
    }

    #[test]
    fn fill_budget_reports_deterministic_partial_progress() {
        let matrix = matrix(2, 2, &[(0, 0, 1), (1, 1, 1)]);
        let outcome = eliminate(
            &matrix,
            EliminationBudget {
                max_fill_nonzeros: 1,
                max_pivot_width: usize::MAX,
            },
        );
        let EliminationOutcome::ThresholdExceeded(threshold) = outcome else {
            panic!("expected fill threshold")
        };
        assert_eq!(threshold.kind, EliminationThresholdKind::FillNonzeros);
        assert_eq!(threshold.source_row, 1);
        assert_eq!(threshold.rows_processed, 1);
        assert_eq!(threshold.partial_rank, 1);
        assert_eq!(threshold.partial_pivot_columns, [0]);
        assert_eq!(threshold.fill_nonzeros, 1);
        assert_eq!(threshold.limit, 1);
        assert_eq!(threshold.required, 2);
    }

    #[test]
    fn dense_kernel_budget_stops_before_basis_allocation() {
        let matrix = matrix(1, 3, &[(0, 0, 1)]);
        let outcome = eliminate_with_kernel_entry_limit(&matrix, EliminationBudget::unlimited(), 5);
        let KernelBoundedEliminationOutcome::DenseKernelThresholdExceeded(threshold) = outcome
        else {
            panic!("expected dense-kernel threshold")
        };
        assert_eq!(threshold.rank, 1);
        assert_eq!(threshold.free_columns, [1, 2]);
        assert_eq!(threshold.columns, 3);
        assert_eq!(threshold.required_entries, 6);
        assert_eq!(threshold.limit, 5);

        let outcome = eliminate_with_kernel_entry_limit(&matrix, EliminationBudget::unlimited(), 6);
        let KernelBoundedEliminationOutcome::Complete(result) = outcome else {
            panic!("dense-kernel budget equal to the requirement must complete")
        };
        assert_eq!(result.kernel_basis.len(), 2);
    }

    #[test]
    fn coefficient_equal_to_prime_is_zero_in_the_field() {
        let matrix = matrix(1, 1, &[(0, 0, PRIME as i32)]);
        let result = complete(&matrix);
        assert_eq!(result.rank, 0);
        assert_eq!(result.free_columns, [0]);
        assert_eq!(result.kernel_basis, [vec![1]]);
    }
}
