use adynkra_exact_sparse::PRIME;
use adynkra_exact_sparse::level12::{
    CANONICAL_CSR_DIGEST_SCHEMA, Level12Matrix, SOURCE_LABELED_DIGEST_SCHEMA, build_level12_matrix,
};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::{Duration, Instant};

fn histogram_json(histogram: &BTreeMap<u32, u64>) -> String {
    let entries: Vec<_> = histogram
        .iter()
        .map(|(degree, count)| format!("\"{degree}\":{count}"))
        .collect();
    format!("{{{}}}", entries.join(","))
}

const BENCHMARK_SCHEMA: &str = "adynkra-level12-checked-spmv-benchmark-v2";

struct BenchmarkResult {
    warmup: Duration,
    timed: Duration,
    checksum: u32,
}

fn benchmark(matrix: &Level12Matrix, iterations: usize) -> BenchmarkResult {
    let mut state = 0x9e37_79b9_u32;
    let input: Vec<_> = (0..matrix.raising.columns())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state % PRIME
        })
        .collect();
    let mut output = vec![0; matrix.raising.rows() as usize];
    let warmup_started = Instant::now();
    matrix.raising.spmv_into(&input, &mut output).unwrap();
    black_box(&output);
    let warmup = warmup_started.elapsed();

    let started = Instant::now();
    for _ in 0..iterations {
        matrix.raising.spmv_into(&input, &mut output).unwrap();
        black_box(&output);
    }
    let timed = started.elapsed();
    let checksum = output.iter().fold(0_u32, |sum, &value| {
        ((u64::from(sum) + u64::from(value)) % u64::from(PRIME)) as u32
    });
    BenchmarkResult {
        warmup,
        timed,
        checksum,
    }
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(label) = arguments.next() else {
        eprintln!("usage: level12_matrix LABEL [SPMV_ITERATIONS]");
        return ExitCode::from(2);
    };
    let iterations = match arguments.next() {
        Some(value) => match value.parse::<usize>() {
            Ok(0) | Err(_) => {
                eprintln!("SPMV_ITERATIONS must be a positive integer");
                return ExitCode::from(2);
            }
            Ok(value) => value,
        },
        None => 20,
    };
    if arguments.next().is_some() {
        eprintln!("usage: level12_matrix LABEL [SPMV_ITERATIONS]");
        return ExitCode::from(2);
    }

    let build_started = Instant::now();
    let matrix = match build_level12_matrix(&label) {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let build_elapsed = build_started.elapsed();
    let row_histogram = matrix.row_degree_histogram();
    let column_histogram = matrix.column_degree_histogram();
    let numeric_csr_hash = matrix.canonical_sha256();
    let source_labeled_hash = matrix.source_labeled_sha256();
    let benchmark = benchmark(&matrix, iterations);
    let seconds_per_spmv = benchmark.timed.as_secs_f64() / iterations as f64;
    let million_nonzeros_per_second =
        matrix.raising.nonzeros() as f64 / seconds_per_spmv / 1_000_000.0;

    println!("{{");
    println!("  \"benchmark_schema\": \"{BENCHMARK_SCHEMA}\",");
    println!("  \"dynkin_label\": \"{}\",", matrix.label);
    println!("  \"rows\": {},", matrix.raising.rows());
    println!("  \"columns\": {},", matrix.raising.columns());
    println!("  \"nonzero_entries\": {},", matrix.raising.nonzeros());
    println!(
        "  \"row_degree_histogram\": {},",
        histogram_json(&row_histogram)
    );
    println!(
        "  \"column_degree_histogram\": {},",
        histogram_json(&column_histogram)
    );
    println!("  \"numeric_csr_digest_schema\": \"{CANONICAL_CSR_DIGEST_SCHEMA}\",");
    println!("  \"canonical_matrix_sha256\": \"{numeric_csr_hash}\",");
    println!("  \"source_labeled_digest_schema\": \"{SOURCE_LABELED_DIGEST_SCHEMA}\",");
    println!("  \"source_labeled_matrix_sha256\": \"{source_labeled_hash}\",");
    println!("  \"build_seconds\": {:.6},", build_elapsed.as_secs_f64());
    println!("  \"spmv_path\": \"checked_csr_with_canonical_input_scan\",");
    println!("  \"warmup_spmv_iterations\": 1,");
    println!(
        "  \"warmup_spmv_seconds\": {:.9},",
        benchmark.warmup.as_secs_f64()
    );
    println!("  \"timed_spmv_iterations\": {iterations},");
    println!(
        "  \"timed_spmv_total_seconds\": {:.9},",
        benchmark.timed.as_secs_f64()
    );
    println!("  \"timed_spmv_seconds_per_iteration\": {seconds_per_spmv:.9},");
    println!("  \"checked_spmv_million_nonzeros_per_second\": {million_nonzeros_per_second:.3},");
    println!("  \"spmv_checksum_mod_prime\": {}", benchmark.checksum);
    println!("}}");
    ExitCode::SUCCESS
}
