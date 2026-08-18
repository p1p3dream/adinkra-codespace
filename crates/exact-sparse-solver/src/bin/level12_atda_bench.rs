use adynkra_exact_sparse::PRIME;
use adynkra_exact_sparse::accelerator::{
    BLOCK_WIDTH, BlockWorkspace32, DEFAULT_DIAGONAL_SEED, DIAGONAL_PRNG_VERSION,
    PackedSignedUnitMatrix, pinned_nonzero_diagonal,
};
use adynkra_exact_sparse::level12::{
    CANONICAL_CSR_DIGEST_SCHEMA, SOURCE_LABELED_DIGEST_SCHEMA, build_level12_matrix,
};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const BENCHMARK_SCHEMA: &str = "adynkra-packed-atda-block32-benchmark-v2";
const DIAGONAL_DIGEST_SCHEMA: &str = "adynkra-atda-diagonal-v1";
const INPUT_BLOCK_PRNG_VERSION: u32 = 1;
const INPUT_BLOCK_SEED: u32 = 0x243f_6a88;
const INPUT_BLOCK_MULTIPLIER: u32 = 1_664_525;
const INPUT_BLOCK_INCREMENT: u32 = 1_013_904_223;
const INPUT_BLOCK_DIGEST_SCHEMA: &str = "adynkra-atda-initial-block32-v1";
const OUTPUT_BLOCK_DIGEST_SCHEMA: &str = "adynkra-atda-block32-output-v1";

fn u32_payload_bytes(elements: usize) -> u64 {
    elements as u64 * std::mem::size_of::<u32>() as u64
}

fn update_field_buffer(hasher: &mut Sha256, block: &[u32]) {
    hasher.update((block.len() as u64).to_le_bytes());
    for &value in block {
        hasher.update(value.to_le_bytes());
    }
}

fn diagonal_digest(diagonal: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DIAGONAL_DIGEST_SCHEMA.as_bytes());
    hasher.update([0]);
    hasher.update(DIAGONAL_PRNG_VERSION.to_le_bytes());
    hasher.update(DEFAULT_DIAGONAL_SEED.to_le_bytes());
    update_field_buffer(&mut hasher, diagonal);
    format!("{:x}", hasher.finalize())
}

fn initial_block_digest(block: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(INPUT_BLOCK_DIGEST_SCHEMA.as_bytes());
    hasher.update([0]);
    hasher.update(INPUT_BLOCK_PRNG_VERSION.to_le_bytes());
    hasher.update(INPUT_BLOCK_SEED.to_le_bytes());
    hasher.update(INPUT_BLOCK_MULTIPLIER.to_le_bytes());
    hasher.update(INPUT_BLOCK_INCREMENT.to_le_bytes());
    update_field_buffer(&mut hasher, block);
    format!("{:x}", hasher.finalize())
}

fn output_block_digest(block: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(OUTPUT_BLOCK_DIGEST_SCHEMA.as_bytes());
    hasher.update([0]);
    update_field_buffer(&mut hasher, block);
    format!("{:x}", hasher.finalize())
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(label) = arguments.next() else {
        eprintln!("usage: level12_atda_bench LABEL [ITERATIONS]");
        return ExitCode::from(2);
    };
    let iterations = match arguments.next() {
        Some(value) => match value.parse::<usize>() {
            Ok(0) | Err(_) => {
                eprintln!("ITERATIONS must be a positive integer");
                return ExitCode::from(2);
            }
            Ok(value) => value,
        },
        None => 3,
    };
    if arguments.next().is_some() {
        eprintln!("usage: level12_atda_bench LABEL [ITERATIONS]");
        return ExitCode::from(2);
    }
    let Some(operator_chain_length) = iterations.checked_add(1) else {
        eprintln!("ITERATIONS is too large");
        return ExitCode::from(2);
    };

    let build_started = Instant::now();
    let level12 = match build_level12_matrix(&label) {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let packed = match PackedSignedUnitMatrix::from_csr(&level12.raising) {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let numeric_csr_hash = level12.canonical_sha256();
    let source_labeled_hash = level12.source_labeled_sha256();
    // The normal-operator path needs only the packed pair, not the construction
    // CSR or source masks. Release them before allocating Krylov blocks.
    drop(level12);

    let diagonal = pinned_nonzero_diagonal(packed.rows(), DEFAULT_DIAGONAL_SEED);
    let block_elements = match (packed.columns() as usize).checked_mul(BLOCK_WIDTH) {
        Some(value) => value,
        None => {
            eprintln!("block size overflow");
            return ExitCode::FAILURE;
        }
    };
    let mut state = INPUT_BLOCK_SEED;
    let mut block_a: Vec<u32> = (0..block_elements)
        .map(|_| {
            state = state
                .wrapping_mul(INPUT_BLOCK_MULTIPLIER)
                .wrapping_add(INPUT_BLOCK_INCREMENT);
            state % PRIME
        })
        .collect();
    let mut block_b = vec![0_u32; block_elements];
    let mut workspace = match BlockWorkspace32::new(&packed) {
        Ok(workspace) => workspace,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let allocated = Instant::now();
    let diagonal_sha256 = diagonal_digest(&diagonal);
    let initial_block_sha256 = initial_block_digest(&block_a);

    let estimated_matrix_payload_bytes = u32_payload_bytes(
        packed.csr_offsets().len()
            + packed.csr_entries().len()
            + packed.transpose_offsets().len()
            + packed.transpose_entries().len(),
    );
    let estimated_diagonal_payload_bytes = u32_payload_bytes(diagonal.len());
    let estimated_one_krylov_block_payload_bytes = u32_payload_bytes(block_a.len());
    let estimated_workspace_payload_bytes =
        u32_payload_bytes(workspace.row_capacity() * BLOCK_WIDTH);
    let estimated_working_payload_bytes = estimated_matrix_payload_bytes
        + estimated_diagonal_payload_bytes
        + 2 * estimated_one_krylov_block_payload_bytes
        + estimated_workspace_payload_bytes;

    // One untimed warmup faults pages, primes caches, and proves every hot-loop
    // buffer was allocated before timing starts.
    if let Err(error) = packed.apply_atda_block32(
        &diagonal,
        black_box(&block_a),
        black_box(&mut block_b),
        &mut workspace,
    ) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    let warmup_finished = Instant::now();

    // Continue the Krylov chain after warmup and ping-pong the two preallocated
    // blocks. No allocation occurs inside the timed loop.
    let timed_started = Instant::now();
    let mut current_is_a = false;
    for _ in 0..iterations {
        let result = if current_is_a {
            packed.apply_atda_block32(
                &diagonal,
                black_box(&block_a),
                black_box(&mut block_b),
                &mut workspace,
            )
        } else {
            packed.apply_atda_block32(
                &diagonal,
                black_box(&block_b),
                black_box(&mut block_a),
                &mut workspace,
            )
        };
        if let Err(error) = result {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
        current_is_a = !current_is_a;
    }
    let timed_elapsed = timed_started.elapsed();
    let current = if current_is_a { &block_a } else { &block_b };
    let output_checksum = output_block_digest(current);
    let end_to_end_seconds = build_started.elapsed().as_secs_f64();

    let seconds = timed_elapsed.as_secs_f64();
    let milliseconds_per_normal_operator = seconds * 1_000.0 / iterations as f64;
    let effective_operations =
        2.0 * packed.nonzeros() as f64 * BLOCK_WIDTH as f64 * iterations as f64;
    let effective_nnz_lane_operations_per_second = effective_operations / seconds;

    println!("{{");
    println!("  \"benchmark_schema\": \"{BENCHMARK_SCHEMA}\",");
    println!("  \"dynkin_label\": \"{label}\",");
    println!("  \"rows\": {},", packed.rows());
    println!("  \"columns\": {},", packed.columns());
    println!("  \"nonzero_entries\": {},", packed.nonzeros());
    println!("  \"prime\": {PRIME},");
    println!("  \"block_width\": {BLOCK_WIDTH},");
    println!("  \"operator_path\": \"packed_signed_unit_fused_atda_block32_no_validation_scan\",");
    println!("  \"field_input_contract\": \"canonical_residues_in_[0,prime)\",");
    println!("  \"warmup_operators\": 1,");
    println!("  \"timed_operators\": {iterations},");
    println!("  \"output_operator_chain_length\": {operator_chain_length},");
    println!(
        "  \"build_and_allocate_seconds\": {:.6},",
        allocated.duration_since(build_started).as_secs_f64()
    );
    println!(
        "  \"warmup_seconds\": {:.6},",
        warmup_finished.duration_since(allocated).as_secs_f64()
    );
    println!("  \"timed_total_seconds\": {seconds:.9},");
    println!("  \"end_to_end_seconds\": {end_to_end_seconds:.9},");
    println!("  \"milliseconds_per_normal_operator\": {milliseconds_per_normal_operator:.6},");
    println!(
        "  \"effective_nnz_lane_operations_per_second\": {effective_nnz_lane_operations_per_second:.3},"
    );
    println!(
        "  \"effective_operation_definition\": \"2 * matrix_nnz * block_width per normal operator\","
    );
    println!("  \"estimated_matrix_payload_bytes\": {estimated_matrix_payload_bytes},");
    println!("  \"estimated_diagonal_payload_bytes\": {estimated_diagonal_payload_bytes},");
    println!(
        "  \"estimated_one_krylov_block_payload_bytes\": {estimated_one_krylov_block_payload_bytes},"
    );
    println!("  \"estimated_workspace_payload_bytes\": {estimated_workspace_payload_bytes},");
    println!("  \"estimated_working_payload_bytes\": {estimated_working_payload_bytes},");
    println!(
        "  \"payload_measurement_boundary\": \"logical_u32_buffers_only; excludes capacity slack, allocator overhead, construction peak, code, and runtime\","
    );
    println!("  \"numeric_csr_digest_schema\": \"{CANONICAL_CSR_DIGEST_SCHEMA}\",");
    println!("  \"canonical_matrix_sha256\": \"{numeric_csr_hash}\",");
    println!("  \"source_labeled_digest_schema\": \"{SOURCE_LABELED_DIGEST_SCHEMA}\",");
    println!("  \"source_labeled_matrix_sha256\": \"{source_labeled_hash}\",");
    println!(
        "  \"matrix_semantic_digest\": \"{}\",",
        packed.semantic_digest()
    );
    println!("  \"diagonal_prng_version\": {DIAGONAL_PRNG_VERSION},");
    println!("  \"diagonal_seed_hex\": \"0x{DEFAULT_DIAGONAL_SEED:016x}\",");
    println!("  \"diagonal_digest_schema\": \"{DIAGONAL_DIGEST_SCHEMA}\",");
    println!("  \"diagonal_sha256\": \"{diagonal_sha256}\",");
    println!("  \"input_block_prng_version\": {INPUT_BLOCK_PRNG_VERSION},");
    println!("  \"input_block_seed_hex\": \"0x{INPUT_BLOCK_SEED:08x}\",");
    println!("  \"input_block_multiplier\": {INPUT_BLOCK_MULTIPLIER},");
    println!("  \"input_block_increment\": {INPUT_BLOCK_INCREMENT},");
    println!("  \"input_block_digest_schema\": \"{INPUT_BLOCK_DIGEST_SCHEMA}\",");
    println!("  \"initial_input_block_sha256\": \"{initial_block_sha256}\",");
    println!("  \"output_digest_schema\": \"{OUTPUT_BLOCK_DIGEST_SCHEMA}\",");
    println!("  \"output_checksum_sha256\": \"{output_checksum}\"");
    println!("}}");
    ExitCode::SUCCESS
}
