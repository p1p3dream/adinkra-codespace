use adynkra_exact_sparse::PRIME;
use adynkra_exact_sparse::certificate::certify_kernel_basis;
use adynkra_exact_sparse::level12::{build_level12_matrix, expected_multiplicity};
use adynkra_exact_sparse::publish::{
    KernelOutputMetadata, PublishedSystem, StagedOutput, StagedSystem, SystemSeconds,
    publish_staged_systems,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: level12_publish_kernel LABEL STAGED_KERNEL ROOT MATRIX_SECONDS SOLVE_SECONDS VERIFY_SECONDS";
const CHECKPOINT_RELATIVE_PATH: &str =
    "results/adynkra_11d_level12_second_momentum_kernel_generation.json";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let label = arguments.next().ok_or(USAGE)?;
    let staged_path = PathBuf::from(arguments.next().ok_or(USAGE)?);
    let root = PathBuf::from(arguments.next().ok_or(USAGE)?);
    let matrix_seconds = parse_seconds(arguments.next(), "MATRIX_SECONDS")?;
    let solve_seconds = parse_seconds(arguments.next(), "SOLVE_SECONDS")?;
    let verify_seconds = parse_seconds(arguments.next(), "VERIFY_SECONDS")?;
    if arguments.next().is_some() {
        return Err(USAGE.into());
    }
    if expected_multiplicity(&label) != Some(1) {
        return Err(format!("{label} is not a multiplicity-one level-12 system").into());
    }

    let root = fs::canonicalize(root)?;
    let matrix = build_level12_matrix(&label)?;
    let bytes = fs::read(&staged_path)?;
    let coefficients = decode_kernel(&bytes, matrix.raising.columns() as usize)?;
    let modular = coefficients
        .iter()
        .map(|coefficient| coefficient.rem_euclid(i64::from(PRIME)) as u32)
        .collect::<Vec<_>>();
    let certificate = certify_kernel_basis(&label, &matrix.raising, &[modular])?;
    let rank = matrix.raising.columns() as usize - 1;
    let rank_certificate = certificate.certify_characteristic_zero_rank(rank)?;
    if certificate.kernels[0].encoded_little_endian != bytes {
        return Err("staged kernel is not the canonical reconstructed encoding".into());
    }
    let kernel = &certificate.kernels[0];
    let free_column = kernel
        .coefficients
        .iter()
        .rposition(|coefficient| *coefficient != 0)
        .ok_or("kernel is zero")?;

    let mut extra_fields = BTreeMap::<String, Value>::new();
    extra_fields.insert("solver".to_owned(), json!("cuda-bordered-cg-v1"));
    extra_fields.insert(
        "rank_proof_rounds".to_owned(),
        json!(matrix.raising.columns()),
    );
    extra_fields.insert("rank_proof_lane_count".to_owned(), json!(32));
    extra_fields.insert(
        "rank_proof".to_owned(),
        json!("exact-n-direction-bordered-CG"),
    );
    extra_fields.insert(
        "deterministic_characteristic_zero_rank_certified".to_owned(),
        json!(rank_certificate.deterministic_characteristic_zero_rank_certified),
    );
    extra_fields.insert("maximum_pivot_width_not_applicable".to_owned(), json!(true));

    let outputs = vec![KernelOutputMetadata {
        copy: 1,
        path: kernel.metadata.path.clone(),
        sha256: kernel.metadata.sha256.clone(),
        bytes: kernel.metadata.bytes,
        nonzero_coefficients: kernel.metadata.nonzero_coefficients,
        maximum_absolute_coefficient: kernel.metadata.maximum_absolute_coefficient,
        extra_fields: BTreeMap::new(),
    }];
    let system = PublishedSystem {
        dynkin_label: label,
        exterior_degree: 12,
        source_columns: matrix.raising.columns() as usize,
        raising_rows: matrix.raising.rows() as usize,
        nonzero_entries: matrix.raising.nonzeros(),
        prime: PRIME,
        exact_modular_rank: rank,
        exact_nullity: 1,
        free_columns: vec![free_column],
        maximum_pivot_width: 0,
        coefficient_width_bytes: certificate.coefficient_width_bytes,
        outputs,
        seconds: SystemSeconds {
            matrix: matrix_seconds,
            echelon: solve_seconds,
            reconstruct_and_integer_verify: verify_seconds,
            total: matrix_seconds + solve_seconds + verify_seconds,
            extra_fields: BTreeMap::new(),
        },
        passed: true,
        extra_fields,
    };
    let artifact = publish_staged_systems(
        &root,
        Path::new(CHECKPOINT_RELATIVE_PATH),
        vec![StagedSystem {
            system,
            outputs: vec![StagedOutput {
                copy: 1,
                path: staged_path,
            }],
        }],
    )?;
    println!(
        "published completed_systems={} completed_kernel_copies={} inventory_complete={}",
        artifact.completed_systems.unwrap_or_default(),
        artifact.completed_kernel_copies.unwrap_or_default(),
        artifact.inventory_complete.unwrap_or(false)
    );
    Ok(())
}

fn parse_seconds(value: Option<String>, name: &str) -> Result<f64, Box<dyn Error>> {
    let value: f64 = value.ok_or_else(|| format!("missing {name}"))?.parse()?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be a finite nonnegative number").into());
    }
    Ok(value)
}

fn decode_kernel(bytes: &[u8], columns: usize) -> Result<Vec<i64>, Box<dyn Error>> {
    if bytes.len() == columns.checked_mul(2).ok_or("kernel size overflow")? {
        return Ok(bytes
            .chunks_exact(2)
            .map(|chunk| i64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
            .collect());
    }
    if bytes.len() == columns.checked_mul(4).ok_or("kernel size overflow")? {
        return Ok(bytes
            .chunks_exact(4)
            .map(|chunk| i64::from(i32::from_le_bytes(chunk.try_into().expect("four bytes"))))
            .collect());
    }
    Err(format!(
        "kernel byte count {} does not match {columns} columns",
        bytes.len()
    )
    .into())
}
