use adynkra_exact_sparse::PRIME;
use adynkra_exact_sparse::certificate::certify_kernel_basis;
use adynkra_exact_sparse::elimination::{
    EliminationBudget, EliminationOutcome, EliminationThresholdKind, eliminate,
};
use adynkra_exact_sparse::level12::{Level12Matrix, build_level12_matrix, expected_multiplicity};
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Instant;

const SCHEMA_VERSION: &str = "adynkra-exact-sparse-elimination-jsonl-v1";

fn parse_limit(value: Option<String>, name: &str) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("missing {name}"))?;
    value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a nonnegative integer, got {value}"))
}

fn usize_array_json(values: &[u32]) -> String {
    let entries = values.iter().map(u32::to_string).collect::<Vec<_>>();
    format!("[{}]", entries.join(","))
}

fn string_array_json(values: &[String]) -> String {
    let entries = values
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>();
    format!("[{}]", entries.join(","))
}

fn progress_line(
    matrix: &Level12Matrix,
    matrix_sha256: &str,
    source_labeled_sha256: &str,
    build_seconds: f64,
    budget: EliminationBudget,
) {
    println!(
        "{{\"schema_version\":\"{SCHEMA_VERSION}\",\"event\":\"matrix_built\",\"dynkin_label\":\"{}\",\"rows\":{},\"columns\":{},\"nonzero_entries\":{},\"prime\":{PRIME},\"canonical_matrix_sha256\":\"{matrix_sha256}\",\"source_labeled_matrix_sha256\":\"{source_labeled_sha256}\",\"build_seconds\":{build_seconds:.6},\"max_fill_nonzeros\":{},\"max_pivot_width\":{}}}",
        matrix.label,
        matrix.raising.rows(),
        matrix.raising.columns(),
        matrix.raising.nonzeros(),
        budget.max_fill_nonzeros,
        budget.max_pivot_width
    );
    io::stdout().flush().expect("stdout flush must succeed");
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(label) = arguments.next() else {
        eprintln!("usage: level12_eliminate LABEL MAX_FILL_NONZEROS MAX_PIVOT_WIDTH");
        return ExitCode::from(2);
    };
    let max_fill_nonzeros = match parse_limit(arguments.next(), "MAX_FILL_NONZEROS") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            eprintln!("usage: level12_eliminate LABEL MAX_FILL_NONZEROS MAX_PIVOT_WIDTH");
            return ExitCode::from(2);
        }
    };
    let max_pivot_width = match parse_limit(arguments.next(), "MAX_PIVOT_WIDTH") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            eprintln!("usage: level12_eliminate LABEL MAX_FILL_NONZEROS MAX_PIVOT_WIDTH");
            return ExitCode::from(2);
        }
    };
    if arguments.next().is_some() {
        eprintln!("usage: level12_eliminate LABEL MAX_FILL_NONZEROS MAX_PIVOT_WIDTH");
        return ExitCode::from(2);
    }
    let budget = EliminationBudget {
        max_fill_nonzeros,
        max_pivot_width,
    };

    let build_started = Instant::now();
    let matrix = match build_level12_matrix(&label) {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let build_seconds = build_started.elapsed().as_secs_f64();
    let matrix_sha256 = matrix.canonical_sha256();
    let source_labeled_sha256 = matrix.source_labeled_sha256();
    progress_line(
        &matrix,
        &matrix_sha256,
        &source_labeled_sha256,
        build_seconds,
        budget,
    );

    let elimination_started = Instant::now();
    match eliminate(&matrix.raising, budget) {
        EliminationOutcome::Complete(result) => {
            let elimination_seconds = elimination_started.elapsed().as_secs_f64();
            let mut residual_rows_by_kernel = Vec::with_capacity(result.kernel_basis.len());
            let mut verification_error = None;
            for vector in &result.kernel_basis {
                match matrix.raising.spmv(vector) {
                    Ok(residual) => residual_rows_by_kernel
                        .push(residual.into_iter().filter(|value| *value != 0).count()),
                    Err(error) => {
                        verification_error = Some(error.to_string());
                        break;
                    }
                }
            }
            if let Some(error) = verification_error {
                eprintln!("modular kernel verification failed: {error}");
                return ExitCode::FAILURE;
            }
            let verified_kernel_vectors = residual_rows_by_kernel
                .iter()
                .filter(|&&rows| rows == 0)
                .count();
            let all_kernel_vectors_verified = verified_kernel_vectors == result.kernel_basis.len();
            let expected_nullity = expected_multiplicity(&label).unwrap_or(0);
            let rank_nullity_complete =
                result.rank + result.free_columns.len() == matrix.raising.columns() as usize;
            let expected_nullity_matched = result.free_columns.len() == expected_nullity;
            let reconstruction_started = Instant::now();
            let certificate =
                match certify_kernel_basis(&matrix.label, &matrix.raising, &result.kernel_basis) {
                    Ok(certificate) => certificate,
                    Err(error) => {
                        eprintln!("integer kernel certification failed: {error}");
                        return ExitCode::FAILURE;
                    }
                };
            let reconstruction_seconds = reconstruction_started.elapsed().as_secs_f64();
            let rank_certificate = match certificate.certify_characteristic_zero_rank(result.rank) {
                Ok(certificate) => certificate,
                Err(error) => {
                    eprintln!("characteristic-zero rank certification failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let kernel_sha256 = certificate
                .kernels
                .iter()
                .map(|kernel| kernel.metadata.sha256.clone())
                .collect::<Vec<_>>();
            let kernel_paths = certificate
                .kernels
                .iter()
                .map(|kernel| kernel.metadata.path.clone())
                .collect::<Vec<_>>();
            let maximum_absolute_coefficient = certificate
                .kernels
                .iter()
                .map(|kernel| kernel.metadata.maximum_absolute_coefficient)
                .max()
                .unwrap_or(0);
            println!(
                "{{\"schema_version\":\"{SCHEMA_VERSION}\",\"event\":\"elimination_complete\",\"outcome\":\"complete\",\"dynkin_label\":\"{}\",\"prime\":{PRIME},\"canonical_matrix_sha256\":\"{matrix_sha256}\",\"source_labeled_matrix_sha256\":\"{source_labeled_sha256}\",\"rank\":{},\"nullity\":{},\"expected_nullity\":{expected_nullity},\"free_columns\":{},\"pivot_count\":{},\"rows_processed\":{},\"row_reductions\":{},\"maximum_pivot_width\":{},\"fill_nonzeros\":{},\"elimination_seconds\":{elimination_seconds:.6},\"reconstruction\":\"rational_to_primitive_integer\",\"integer_reconstruction_performed\":true,\"reconstruction_seconds\":{reconstruction_seconds:.6},\"coefficient_width_bytes\":{},\"maximum_absolute_coefficient\":{maximum_absolute_coefficient},\"kernel_paths\":{},\"kernel_sha256\":{},\"kernel_vectors\":{},\"verified_kernel_vectors\":{verified_kernel_vectors},\"nonzero_residual_rows_by_kernel\":{:?},\"all_kernel_vectors_verified\":{all_kernel_vectors_verified},\"integer_residual_verified\":true,\"integer_kernels_independent_mod_prime\":{},\"characteristic_zero_rank\":{},\"characteristic_zero_nullity\":{},\"deterministic_characteristic_zero_rank_certified\":{},\"rank_nullity_complete\":{rank_nullity_complete},\"expected_nullity_matched\":{expected_nullity_matched},\"publishable_certificate\":{}}}",
                matrix.label,
                result.rank,
                result.free_columns.len(),
                usize_array_json(&result.free_columns),
                result.pivots.len(),
                result.rows_processed,
                result.row_reductions,
                result.maximum_pivot_width,
                result.fill_nonzeros,
                certificate.coefficient_width_bytes,
                string_array_json(&kernel_paths),
                string_array_json(&kernel_sha256),
                result.kernel_basis.len(),
                residual_rows_by_kernel,
                certificate.integer_kernels_independent_mod_prime,
                rank_certificate.characteristic_zero_rank,
                rank_certificate.characteristic_zero_nullity,
                rank_certificate.deterministic_characteristic_zero_rank_certified,
                rank_certificate.deterministic_characteristic_zero_rank_certified
                    && all_kernel_vectors_verified
                    && expected_nullity_matched,
            );
            if all_kernel_vectors_verified
                && rank_nullity_complete
                && expected_nullity_matched
                && rank_certificate.deterministic_characteristic_zero_rank_certified
            {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        EliminationOutcome::ThresholdExceeded(threshold) => {
            let elimination_seconds = elimination_started.elapsed().as_secs_f64();
            let threshold_kind = match threshold.kind {
                EliminationThresholdKind::PivotWidth => "pivot_width",
                EliminationThresholdKind::FillNonzeros => "fill_nonzeros",
            };
            println!(
                "{{\"schema_version\":\"{SCHEMA_VERSION}\",\"event\":\"elimination_stopped\",\"outcome\":\"threshold_exceeded\",\"dynkin_label\":\"{}\",\"prime\":{PRIME},\"threshold_kind\":\"{threshold_kind}\",\"source_row\":{},\"rows_processed\":{},\"partial_rank\":{},\"partial_pivot_count\":{},\"row_reductions\":{},\"maximum_pivot_width\":{},\"fill_nonzeros\":{},\"limit\":{},\"required\":{},\"elimination_seconds\":{elimination_seconds:.6},\"partial_rank_is_final\":false}}",
                matrix.label,
                threshold.source_row,
                threshold.rows_processed,
                threshold.partial_rank,
                threshold.partial_pivot_columns.len(),
                threshold.row_reductions,
                threshold.maximum_pivot_width,
                threshold.fill_nonzeros,
                threshold.limit,
                threshold.required,
            );
            ExitCode::from(3)
        }
    }
}
