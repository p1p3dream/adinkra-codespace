#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("level12_cuda_null requires --features cuda on Linux with CUDA 12");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use adynkra_exact_sparse::accelerator::{PackedSignedUnitMatrix, pinned_nonzero_diagonal};
    use adynkra_exact_sparse::certificate::certify_kernel_basis;
    use adynkra_exact_sparse::cuda::{CudaAtdaBlock32, CudaCgLaneStatus};
    use adynkra_exact_sparse::gpu_krylov::{
        GpuCgCheckpoint, initialize_seeded, pinned_border_block, validate_converged_lanes,
    };
    use adynkra_exact_sparse::level12::build_level12_matrix;
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::time::Instant;

    let options = Options::parse()?;
    let started = Instant::now();
    let level12 = build_level12_matrix(&options.label)?;
    let packed = PackedSignedUnitMatrix::from_csr(&level12.raising)?;

    let mut diagonal_seed = options.diagonal_seed;
    let mut border_seed = options.border_seed;
    let mut restored = None;
    if options.resume {
        let path = options
            .checkpoint
            .as_ref()
            .ok_or("--resume requires --checkpoint PATH")?;
        let checkpoint = GpuCgCheckpoint::load(path, &packed)?;
        if options.diagonal_seed_explicit && options.diagonal_seed != checkpoint.diagonal_seed {
            return Err("--diagonal-seed disagrees with the checkpoint".into());
        }
        if options.border_seed_explicit && options.border_seed != checkpoint.border_seed {
            return Err("--border-seed disagrees with the checkpoint".into());
        }
        diagonal_seed = checkpoint.diagonal_seed;
        border_seed = checkpoint.border_seed;
        restored = Some(checkpoint);
    }

    let diagonal = pinned_nonzero_diagonal(packed.rows(), diagonal_seed);
    let mut operator = CudaAtdaBlock32::new(&packed, &diagonal, options.device)?;
    let device_name = operator.device_name()?;
    let border;
    let rank_proof_eligible;
    if let Some(checkpoint) = restored {
        border = pinned_border_block(packed.columns(), checkpoint.border_seed)?;
        operator.cg_upload_state(&border, &checkpoint.state)?;
        rank_proof_eligible = checkpoint.rank_proof_eligible;
    } else {
        border = initialize_seeded(&mut operator, border_seed)?;
        rank_proof_eligible = true;
    }

    let dimension = u64::from(packed.columns());
    let target_rounds = options
        .benchmark_rounds
        .map_or(dimension, |rounds| u64::from(rounds).min(dimension));
    let mut progress = operator.cg_run(0)?;
    let mut next_checkpoint = progress
        .total_rounds
        .saturating_add(u64::from(options.checkpoint_every));
    while progress.total_rounds < target_rounds
        && progress.status.contains(&CudaCgLaneStatus::Active)
    {
        let rounds =
            (target_rounds - progress.total_rounds).min(u64::from(options.chunk_rounds)) as u32;
        let chunk_started = Instant::now();
        progress = operator.cg_run(rounds)?;
        let active = progress
            .status
            .iter()
            .filter(|status| **status == CudaCgLaneStatus::Active)
            .count();
        let converged = progress
            .status
            .iter()
            .filter(|status| **status == CudaCgLaneStatus::Converged)
            .count();
        let broken = 32 - active - converged;
        println!(
            "progress rounds={}/{} active={} converged={} broken={} chunk_seconds={:.6}",
            progress.total_rounds,
            dimension,
            active,
            converged,
            broken,
            chunk_started.elapsed().as_secs_f64()
        );
        if let Some(path) = &options.checkpoint
            && (progress.total_rounds >= next_checkpoint || active == 0)
        {
            let state = operator.cg_download_state()?;
            GpuCgCheckpoint::fresh(&packed, diagonal_seed, border_seed, state).save_atomic(path)?;
            next_checkpoint = progress
                .total_rounds
                .saturating_add(u64::from(options.checkpoint_every));
            println!("checkpoint {}", path.display());
        }
    }

    let state = operator.cg_download_state()?;
    if let Some(path) = &options.checkpoint {
        GpuCgCheckpoint::fresh(&packed, diagonal_seed, border_seed, state.clone())
            .save_atomic(path)?;
    }
    if target_rounds < dimension {
        println!("{{");
        println!("  \"schema\": \"adynkra-level12-cuda-bordered-cg-benchmark-v1\",");
        println!("  \"label\": \"{}\",", options.label);
        println!("  \"device\": \"{}\",", escape_json(&device_name));
        println!("  \"columns\": {},", packed.columns());
        println!("  \"benchmark_rounds\": {},", state.total_rounds);
        println!(
            "  \"elapsed_seconds\": {:.6},",
            started.elapsed().as_secs_f64()
        );
        println!("  \"candidate_validation_skipped\": true");
        println!("}}");
        return Ok(());
    }
    let validated = validate_converged_lanes(
        &mut operator,
        &level12.raising,
        &border,
        &state,
        rank_proof_eligible,
    )?;
    let integer = certify_kernel_basis(
        &options.label,
        &level12.raising,
        std::slice::from_ref(&validated.canonical_modular),
    )?;
    let rank_certificate = validated
        .nullity_b_exactly_one
        .then(|| integer.certify_characteristic_zero_rank(packed.columns() as usize - 1))
        .transpose()?;
    let kernel = &integer.kernels[0];
    let free_column = kernel
        .coefficients
        .iter()
        .rposition(|coefficient| *coefficient != 0)
        .ok_or("reconstructed kernel is zero")?;
    if let Some(path) = &options.kernel_output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!(
            "{}.tmp.{}",
            path.extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("kernel"),
            std::process::id()
        ));
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        output.write_all(&kernel.encoded_little_endian)?;
        output.sync_all()?;
        std::fs::rename(&temporary, path)?;
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    let transcripts = state
        .transcript
        .iter()
        .map(|value| format!("\"{value:016x}\""))
        .collect::<Vec<_>>()
        .join(",");

    println!("{{");
    println!("  \"schema\": \"adynkra-level12-cuda-bordered-cg-v1\",");
    println!("  \"label\": \"{}\",", options.label);
    println!("  \"device\": \"{}\",", escape_json(&device_name));
    println!("  \"rows\": {},", packed.rows());
    println!("  \"columns\": {},", packed.columns());
    println!("  \"nonzeros\": {},", packed.nonzeros());
    println!(
        "  \"matrix_semantic_digest\": \"{}\",",
        packed.semantic_digest()
    );
    println!("  \"diagonal_seed_hex\": \"0x{diagonal_seed:016x}\",");
    println!("  \"border_seed_hex\": \"0x{border_seed:016x}\",");
    println!("  \"rounds\": {},", state.total_rounds);
    println!("  \"agreeing_lanes\": {:?},", validated.agreeing_lanes);
    println!("  \"rank_proof_lanes\": {:?},", validated.rank_proof_lanes);
    println!(
        "  \"rank_proof_eligible\": {},",
        validated.rank_proof_eligible
    );
    println!(
        "  \"exact_nullity_one\": {},",
        validated.nullity_b_exactly_one
    );
    println!(
        "  \"characteristic_zero_rank\": {},",
        rank_certificate
            .as_ref()
            .map(|certificate| certificate.characteristic_zero_rank)
            .unwrap_or(0)
    );
    println!(
        "  \"deterministic_characteristic_zero_rank_certified\": {},",
        rank_certificate.is_some()
    );
    println!("  \"gpu_Bx_zero\": true,");
    println!("  \"u_transpose_x_one\": true,");
    println!("  \"bordered_Cx_equals_u\": true,");
    println!("  \"cpu_Ax_zero\": true,");
    println!("  \"integer_Ax_zero\": true,");
    println!(
        "  \"integer_kernel_sha256\": \"{}\",",
        kernel.metadata.sha256
    );
    println!(
        "  \"integer_kernel_nonzeros\": {},",
        kernel.metadata.nonzero_coefficients
    );
    println!(
        "  \"integer_kernel_max_abs\": {},",
        kernel.metadata.maximum_absolute_coefficient
    );
    println!(
        "  \"coefficient_width_bytes\": {},",
        integer.coefficient_width_bytes
    );
    println!("  \"free_column\": {free_column},");
    if let Some(path) = &options.kernel_output {
        println!(
            "  \"kernel_output\": \"{}\",",
            escape_json(&path.display().to_string())
        );
    }
    println!("  \"rolling_transcript_u64_hex\": [{transcripts}],");
    println!(
        "  \"elapsed_seconds\": {:.6}",
        started.elapsed().as_secs_f64()
    );
    println!("}}");
    Ok(())
}

#[cfg(feature = "cuda")]
struct Options {
    label: String,
    device: i32,
    chunk_rounds: u32,
    checkpoint_every: u32,
    benchmark_rounds: Option<u32>,
    checkpoint: Option<std::path::PathBuf>,
    kernel_output: Option<std::path::PathBuf>,
    resume: bool,
    diagonal_seed: u64,
    border_seed: u64,
    diagonal_seed_explicit: bool,
    border_seed_explicit: bool,
}

#[cfg(feature = "cuda")]
impl Options {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        use adynkra_exact_sparse::accelerator::DEFAULT_DIAGONAL_SEED;
        use adynkra_exact_sparse::gpu_krylov::DEFAULT_BORDER_SEED;

        let mut arguments = std::env::args().skip(1);
        let label = arguments.next().ok_or(
            "usage: level12_cuda_null LABEL [--device N] [--chunk N] [--checkpoint PATH] [--checkpoint-every N] [--kernel-output PATH] [--resume] [--diagonal-seed U64] [--border-seed U64] [--benchmark-rounds N]",
        )?;
        let mut result = Self {
            label,
            device: 0,
            chunk_rounds: 4096,
            checkpoint_every: 65_536,
            benchmark_rounds: None,
            checkpoint: None,
            kernel_output: None,
            resume: false,
            diagonal_seed: DEFAULT_DIAGONAL_SEED,
            border_seed: DEFAULT_BORDER_SEED,
            diagonal_seed_explicit: false,
            border_seed_explicit: false,
        };
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--device" => {
                    result.device = arguments.next().ok_or("missing --device value")?.parse()?
                }
                "--chunk" => result.chunk_rounds = positive(arguments.next(), "--chunk")?,
                "--checkpoint-every" => {
                    result.checkpoint_every = positive(arguments.next(), "--checkpoint-every")?
                }
                "--benchmark-rounds" => {
                    result.benchmark_rounds =
                        Some(positive(arguments.next(), "--benchmark-rounds")?)
                }
                "--checkpoint" => {
                    result.checkpoint = Some(std::path::PathBuf::from(
                        arguments.next().ok_or("missing --checkpoint path")?,
                    ))
                }
                "--kernel-output" => {
                    result.kernel_output = Some(std::path::PathBuf::from(
                        arguments.next().ok_or("missing --kernel-output path")?,
                    ))
                }
                "--resume" => result.resume = true,
                "--diagonal-seed" => {
                    result.diagonal_seed =
                        parse_u64(&arguments.next().ok_or("missing --diagonal-seed value")?)?;
                    result.diagonal_seed_explicit = true;
                }
                "--border-seed" => {
                    result.border_seed =
                        parse_u64(&arguments.next().ok_or("missing --border-seed value")?)?;
                    result.border_seed_explicit = true;
                }
                _ => return Err(format!("unknown argument {argument}").into()),
            }
        }
        Ok(result)
    }
}

#[cfg(feature = "cuda")]
fn positive(value: Option<String>, flag: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let value: u32 = value
        .ok_or_else(|| format!("missing {flag} value"))?
        .parse()?;
    if value == 0 {
        return Err(format!("{flag} must be positive").into());
    }
    Ok(value)
}

#[cfg(feature = "cuda")]
fn parse_u64(value: &str) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)?
    } else {
        value.parse()?
    })
}

#[cfg(feature = "cuda")]
fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
