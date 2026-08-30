use adynkra_exact_sparse::PRIME;
use adynkra_exact_sparse::certificate::{KernelCertificateBatch, certify_kernel_basis};
use adynkra_exact_sparse::elimination::{
    EliminationBudget, EliminationOutcome, EliminationResult, EliminationThresholdKind, eliminate,
};
use adynkra_exact_sparse::level12::{Level12Matrix, build_level12_matrix, expected_multiplicity};
use adynkra_exact_sparse::publish::{
    KernelOutputMetadata, PublicationArtifact, PublishedSystem, StagedOutput, StagedSystem,
    SystemSeconds, publish_staged_systems,
};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: &str = "adynkra-exact-sparse-elimination-jsonl-v1";
const USAGE: &str =
    "usage: level12_eliminate LABEL MAX_FILL_NONZEROS MAX_PIVOT_WIDTH [--publish-root ROOT]";
const CHECKPOINT: &str = "results/adynkra_11d_level12_second_momentum_kernel_generation.json";

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

struct PublicationInputs<'a> {
    matrix: &'a Level12Matrix,
    elimination: &'a EliminationResult,
    certificate: &'a KernelCertificateBatch,
    build_seconds: f64,
    elimination_seconds: f64,
    reconstruction_seconds: f64,
}

struct StagingDirectory(PathBuf);

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn publish_certificate(
    root: &Path,
    inputs: PublicationInputs<'_>,
) -> Result<PublicationArtifact, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before the Unix epoch: {error}"))?
        .as_nanos();
    let staging_parent = root.join("results/.level12-rust-staging");
    fs::create_dir_all(&staging_parent)
        .map_err(|error| format!("failed to create {}: {error}", staging_parent.display()))?;
    let staging = staging_parent.join(format!(
        "{}-{}-{nonce}",
        inputs.matrix.label,
        std::process::id()
    ));
    fs::create_dir(&staging)
        .map_err(|error| format!("failed to create {}: {error}", staging.display()))?;
    let cleanup = StagingDirectory(staging.clone());

    let mut staged_outputs = Vec::with_capacity(inputs.certificate.kernels.len());
    for kernel in &inputs.certificate.kernels {
        let path = staging.join(format!("kernel-{}.staged", kernel.metadata.copy));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
        file.write_all(&kernel.encoded_little_endian)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("failed to sync {}: {error}", path.display()))?;
        staged_outputs.push(StagedOutput {
            copy: kernel.metadata.copy,
            path,
        });
    }

    let outputs = inputs
        .certificate
        .kernels
        .iter()
        .map(|kernel| KernelOutputMetadata {
            copy: kernel.metadata.copy,
            path: kernel.metadata.path.clone(),
            sha256: kernel.metadata.sha256.clone(),
            bytes: kernel.metadata.bytes,
            nonzero_coefficients: kernel.metadata.nonzero_coefficients,
            maximum_absolute_coefficient: kernel.metadata.maximum_absolute_coefficient,
            extra_fields: BTreeMap::new(),
        })
        .collect();
    let total_seconds =
        inputs.build_seconds + inputs.elimination_seconds + inputs.reconstruction_seconds;
    let system = PublishedSystem {
        dynkin_label: inputs.matrix.label.clone(),
        exterior_degree: 12,
        source_columns: inputs.matrix.raising.columns() as usize,
        raising_rows: inputs.matrix.raising.rows() as usize,
        nonzero_entries: inputs.matrix.raising.nonzeros(),
        prime: PRIME,
        exact_modular_rank: inputs.elimination.rank,
        exact_nullity: inputs.elimination.free_columns.len(),
        free_columns: inputs
            .elimination
            .free_columns
            .iter()
            .map(|&column| column as usize)
            .collect(),
        maximum_pivot_width: inputs.elimination.maximum_pivot_width,
        coefficient_width_bytes: inputs.certificate.coefficient_width_bytes,
        outputs,
        seconds: SystemSeconds {
            matrix: inputs.build_seconds,
            echelon: inputs.elimination_seconds,
            reconstruct_and_integer_verify: inputs.reconstruction_seconds,
            total: total_seconds,
            extra_fields: BTreeMap::new(),
        },
        passed: true,
        extra_fields: BTreeMap::new(),
    };
    let published = publish_staged_systems(
        root,
        Path::new(CHECKPOINT),
        vec![StagedSystem {
            system,
            outputs: staged_outputs,
        }],
    )
    .map_err(|error| error.to_string())?;
    drop(cleanup);
    Ok(published)
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(label) = arguments.next() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let max_fill_nonzeros = match parse_limit(arguments.next(), "MAX_FILL_NONZEROS") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let max_pivot_width = match parse_limit(arguments.next(), "MAX_PIVOT_WIDTH") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let publish_root = match arguments.next() {
        None => None,
        Some(flag) if flag == "--publish-root" => match arguments.next() {
            Some(root) => Some(PathBuf::from(root)),
            None => {
                eprintln!("--publish-root requires a repository root");
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        },
        Some(argument) => {
            eprintln!("unexpected argument {argument}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    if arguments.next().is_some() {
        eprintln!("{USAGE}");
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
            let publishable_certificate = rank_certificate
                .deterministic_characteristic_zero_rank_certified
                && all_kernel_vectors_verified
                && rank_nullity_complete
                && expected_nullity_matched;
            let publication = if let Some(root) = publish_root.as_deref() {
                if !publishable_certificate {
                    eprintln!("refusing to publish an incomplete exact certificate");
                    return ExitCode::FAILURE;
                }
                match publish_certificate(
                    root,
                    PublicationInputs {
                        matrix: &matrix,
                        elimination: &result,
                        certificate: &certificate,
                        build_seconds,
                        elimination_seconds,
                        reconstruction_seconds,
                    },
                ) {
                    Ok(artifact) => Some(artifact),
                    Err(error) => {
                        eprintln!("kernel publication failed: {error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                None
            };
            let published = publication.is_some();
            let published_completed_systems = publication
                .as_ref()
                .and_then(|artifact| artifact.completed_systems)
                .map_or_else(|| "null".to_owned(), |value| value.to_string());
            let published_completed_kernel_copies = publication
                .as_ref()
                .and_then(|artifact| artifact.completed_kernel_copies)
                .map_or_else(|| "null".to_owned(), |value| value.to_string());
            let published_inventory_complete = publication
                .as_ref()
                .and_then(|artifact| artifact.inventory_complete)
                .map_or_else(|| "null".to_owned(), |value| value.to_string());
            println!(
                "{{\"schema_version\":\"{SCHEMA_VERSION}\",\"event\":\"elimination_complete\",\"outcome\":\"complete\",\"dynkin_label\":\"{}\",\"prime\":{PRIME},\"canonical_matrix_sha256\":\"{matrix_sha256}\",\"source_labeled_matrix_sha256\":\"{source_labeled_sha256}\",\"rank\":{},\"nullity\":{},\"expected_nullity\":{expected_nullity},\"free_columns\":{},\"pivot_count\":{},\"rows_processed\":{},\"row_reductions\":{},\"maximum_pivot_width\":{},\"fill_nonzeros\":{},\"elimination_seconds\":{elimination_seconds:.6},\"reconstruction\":\"rational_to_primitive_integer\",\"integer_reconstruction_performed\":true,\"reconstruction_seconds\":{reconstruction_seconds:.6},\"coefficient_width_bytes\":{},\"maximum_absolute_coefficient\":{maximum_absolute_coefficient},\"kernel_paths\":{},\"kernel_sha256\":{},\"kernel_vectors\":{},\"verified_kernel_vectors\":{verified_kernel_vectors},\"nonzero_residual_rows_by_kernel\":{:?},\"all_kernel_vectors_verified\":{all_kernel_vectors_verified},\"integer_residual_verified\":true,\"integer_kernels_independent_mod_prime\":{},\"characteristic_zero_rank\":{},\"characteristic_zero_nullity\":{},\"deterministic_characteristic_zero_rank_certified\":{},\"rank_nullity_complete\":{rank_nullity_complete},\"expected_nullity_matched\":{expected_nullity_matched},\"publishable_certificate\":{},\"published\":{published},\"published_completed_systems\":{published_completed_systems},\"published_completed_kernel_copies\":{published_completed_kernel_copies},\"published_inventory_complete\":{published_inventory_complete}}}",
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
                publishable_certificate,
            );
            if publishable_certificate {
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
