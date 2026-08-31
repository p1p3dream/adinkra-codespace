//! Three-prime device solve for the bounded teleparallel Lorentz cocycle.
//!
//! The device solves all 55 raw commutators against one canonical rank-320
//! local-Lorentz image basis, then replays every target row. Exact rational
//! reconstruction and cocycle integrability remain host proof gates.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::eleven_dimensional_corrected_teleparallel_equivariance::{
    ExactQiPublic, LocalLorentzTargetImageHandoff, local_lorentz_target_image_handoff,
};
use crate::eleven_dimensional_four_form_56_gpu::PINNED_PRIMES;
use crate::eleven_dimensional_physical_curvature::ExactQi;

const TARGET_ROWS: usize = 10_560;
const AMBIENT_COLUMNS: usize = 1_760;
const RIGHT_HAND_SIDES: usize = 55;
const EXPECTED_RANK: usize = 320;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ThreePrimeFp2 {
    /// Prime-major real and imaginary lanes.
    pub lane: [u32; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ModularCooEntry {
    pub row: u32,
    pub column: u32,
    pub value: ThreePrimeFp2,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TeleparallelLorentzGpuReport {
    pub schema_version: &'static str,
    pub source_coordinate: u32,
    pub momentum_axis: usize,
    pub image_ambient_columns: usize,
    pub image_rank: usize,
    pub right_hand_sides: usize,
    pub matrix_nonzeros: usize,
    pub rhs_nonzeros: usize,
    pub selected_csr_nonzeros: usize,
    pub matrix_sha256: String,
    pub rhs_sha256: String,
    pub pivot_map_sha256: String,
    pub csr_sha256: String,
    pub cuda_source_sha256: String,
    pub rust_source_sha256: String,
    pub handoff_source_sha256: String,
    pub build_rs_sha256: String,
    pub executable_sha256: String,
    pub exact_coordinate_sha256: String,
    pub host_name: String,
    pub gpu_name: String,
    pub driver_version: String,
    pub nvcc_version: String,
    pub command: Vec<String>,
    pub unix_time_seconds: u64,
    pub exact_handoff_milliseconds: f64,
    pub device_input_milliseconds: f64,
    pub end_to_end_milliseconds: f64,
    pub ordered_primes: [u32; 3],
    pub residual_counts: [u64; 3],
    pub first_residual_key: Option<u64>,
    pub first_residual_value: Option<[u32; 6]>,
    pub device_milliseconds: f32,
    pub resident_bytes: u64,
    pub high_water_bytes: u64,
    pub exact_coordinate_terms: usize,
    pub generator_solutions: Vec<LorentzGeneratorSolution>,
    pub modular_membership_passed: bool,
    pub exact_reconstruction_complete: bool,
    pub coherent_cocycle_integrability_complete: bool,
    pub corrected_target_zero_commutator_complete: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LorentzGeneratorCoefficient {
    pub original_d_psi_two_coordinate: usize,
    pub value: ExactQiPublic,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LorentzGeneratorSolution {
    pub generator_left: usize,
    pub generator_right: usize,
    pub exact_image_residual_entries: usize,
    pub coefficients: Vec<LorentzGeneratorCoefficient>,
}

fn power_mod(mut base: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut result = 1_u64;
    let mut factor = u64::from(base);
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * factor % u64::from(prime);
        }
        factor = factor * factor % u64::from(prime);
        exponent >>= 1;
    }
    base = result as u32;
    base
}

fn signed_mod(value: i64, prime: u32) -> u32 {
    let residue = value % i64::from(prime);
    if residue < 0 {
        (residue + i64::from(prime)) as u32
    } else {
        residue as u32
    }
}

fn rational_mod(value: &num_rational::Ratio<i64>, prime: u32) -> Result<u32, String> {
    let denominator = signed_mod(*value.denom(), prime);
    if denominator == 0 {
        return Err(format!(
            "Lorentz-descent denominator is zero modulo pinned prime {prime}"
        ));
    }
    Ok((u64::from(signed_mod(*value.numer(), prime))
        * u64::from(power_mod(denominator, prime - 2, prime))
        % u64::from(prime)) as u32)
}

fn encode(value: &ExactQi) -> Result<ThreePrimeFp2, String> {
    let mut output = ThreePrimeFp2::default();
    for (prime_index, &prime) in PINNED_PRIMES.iter().enumerate() {
        output.lane[2 * prime_index] = rational_mod(&value.real, prime)?;
        output.lane[2 * prime_index + 1] = rational_mod(&value.imaginary, prime)?;
    }
    Ok(output)
}

fn digest_hex(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_i32_slices(slices: &[&[i32]]) -> String {
    let mut digest = Sha256::new();
    for slice in slices {
        for value in *slice {
            digest.update(value.to_le_bytes());
        }
    }
    hex_bytes(&digest.finalize())
}

fn digest_coo(entries: &[ModularCooEntry]) -> String {
    let mut digest = Sha256::new();
    for entry in entries {
        digest.update(entry.row.to_le_bytes());
        digest.update(entry.column.to_le_bytes());
        for value in entry.value.lane {
            digest.update(value.to_le_bytes());
        }
    }
    hex_bytes(&digest.finalize())
}

fn digest_csr(input: &DeviceInput) -> String {
    let mut digest = Sha256::new();
    for value in &input.row_offsets {
        digest.update(value.to_le_bytes());
    }
    for value in &input.column_indices {
        digest.update(value.to_le_bytes());
    }
    for value in &input.csr_values {
        for lane in value.lane {
            digest.update(lane.to_le_bytes());
        }
    }
    hex_bytes(&digest.finalize())
}

struct DeviceInput {
    matrix: Vec<ModularCooEntry>,
    rhs: Vec<ModularCooEntry>,
    row_to_pivot: Vec<i32>,
    column_to_basis: Vec<i32>,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
    csr_values: Vec<ThreePrimeFp2>,
}

fn device_input(handoff: &LocalLorentzTargetImageHandoff) -> Result<DeviceInput, String> {
    if handoff.ambient_d_psi_two_columns != AMBIENT_COLUMNS
        || handoff.exact_image_rank != EXPECTED_RANK
        || handoff.independent_original_columns.len() != EXPECTED_RANK
        || handoff.pivot_rows.len() != EXPECTED_RANK
        || handoff.raw_commutators.len() != RIGHT_HAND_SIDES
    {
        return Err("Lorentz-descent handoff dimensions are not canonical".to_string());
    }
    if !handoff.pivot_rows.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("Lorentz-descent pivot rows are not strictly increasing".to_string());
    }

    let mut row_to_pivot = vec![-1_i32; TARGET_ROWS];
    for (pivot_ordinal, &row) in handoff.pivot_rows.iter().enumerate() {
        if row >= TARGET_ROWS || row_to_pivot[row] != -1 {
            return Err("Lorentz-descent pivot row is invalid".to_string());
        }
        row_to_pivot[row] = pivot_ordinal as i32;
    }
    let mut column_to_basis = vec![-1_i32; AMBIENT_COLUMNS];
    let mut matrix = Vec::new();
    let mut per_row = vec![Vec::<(u32, ThreePrimeFp2)>::new(); TARGET_ROWS];
    for (basis_column, column) in handoff.independent_original_columns.iter().enumerate() {
        let original = column.original_d_psi_two_coordinate;
        if original >= AMBIENT_COLUMNS || column_to_basis[original] != -1 {
            return Err("Lorentz-descent image column identity is invalid".to_string());
        }
        column_to_basis[original] = basis_column as i32;
        for (&row, value) in &column.entries {
            if row >= TARGET_ROWS {
                return Err("Lorentz-descent image row is out of range".to_string());
            }
            let encoded = encode(value)?;
            matrix.push(ModularCooEntry {
                row: row as u32,
                column: original as u32,
                value: encoded,
            });
            per_row[row].push((basis_column as u32, encoded));
        }
    }
    let mut rhs = Vec::new();
    for (generator, column) in handoff.raw_commutators.iter().enumerate() {
        for (&row, value) in &column.entries {
            if row >= TARGET_ROWS {
                return Err("Lorentz-descent commutator row is out of range".to_string());
            }
            rhs.push(ModularCooEntry {
                row: row as u32,
                column: generator as u32,
                value: encode(value)?,
            });
        }
    }
    matrix.sort_unstable_by_key(|entry| (entry.row, entry.column));
    rhs.sort_unstable_by_key(|entry| (entry.row, entry.column));
    if matrix
        .windows(2)
        .any(|pair| (pair[0].row, pair[0].column) == (pair[1].row, pair[1].column))
        || rhs
            .windows(2)
            .any(|pair| (pair[0].row, pair[0].column) == (pair[1].row, pair[1].column))
    {
        return Err("Lorentz-descent COO contains a duplicate key".to_string());
    }
    for entries in &mut per_row {
        entries.sort_unstable_by_key(|(column, _)| *column);
    }
    let mut row_offsets = Vec::with_capacity(TARGET_ROWS + 1);
    let mut column_indices = Vec::new();
    let mut csr_values = Vec::new();
    row_offsets.push(0);
    for entries in per_row {
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0)
            || entries
                .iter()
                .any(|(column, _)| *column >= EXPECTED_RANK as u32)
        {
            return Err("Lorentz-descent CSR row is invalid".to_string());
        }
        for (column, value) in entries {
            column_indices.push(column);
            csr_values.push(value);
        }
        row_offsets.push(column_indices.len() as u32);
    }
    if row_offsets.len() != TARGET_ROWS + 1
        || row_offsets[0] != 0
        || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
        || usize::try_from(*row_offsets.last().unwrap()).unwrap() != csr_values.len()
        || column_indices.len() != csr_values.len()
    {
        return Err("Lorentz-descent CSR structure is invalid".to_string());
    }
    Ok(DeviceInput {
        matrix,
        rhs,
        row_to_pivot,
        column_to_basis,
        row_offsets,
        column_indices,
        csr_values,
    })
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr::NonNull;

    unsafe extern "C" {
        fn adynkra_teleparallel_lorentz_descent_create(
            rank: u32,
            right_hand_sides: u32,
            csr_capacity: u64,
            device_hard_cap: u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> *mut c_void;
        fn adynkra_teleparallel_lorentz_descent_solve(
            context: *mut c_void,
            matrix_entries: *const ModularCooEntry,
            matrix_count: u64,
            rhs_entries: *const ModularCooEntry,
            rhs_count: u64,
            row_to_pivot: *const i32,
            column_to_basis: *const i32,
            row_offsets: *const u32,
            column_indices: *const u32,
            csr_values: *const ThreePrimeFp2,
            csr_count: u64,
            coefficients: *mut ThreePrimeFp2,
            coefficient_capacity: u64,
            residual_counts: *mut u64,
            first_residual_key: *mut u64,
            first_residual_value: *mut ThreePrimeFp2,
            device_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_teleparallel_lorentz_descent_resident_bytes(context: *const c_void) -> u64;
        fn adynkra_teleparallel_lorentz_descent_high_water_bytes(context: *const c_void) -> u64;
        fn adynkra_teleparallel_lorentz_descent_primes(output: *mut u32);
        fn adynkra_teleparallel_lorentz_descent_destroy(context: *mut c_void);
    }

    fn message(error: &[i8]) -> String {
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    pub(super) fn run(
        handoff: &LocalLorentzTargetImageHandoff,
        input: &DeviceInput,
    ) -> Result<TeleparallelLorentzGpuReport, String> {
        let mut error = vec![0_i8; 1024];
        let context = unsafe {
            adynkra_teleparallel_lorentz_descent_create(
                EXPECTED_RANK as u32,
                RIGHT_HAND_SIDES as u32,
                input.csr_values.len() as u64,
                2_u64 << 30,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        let context = NonNull::new(context).ok_or_else(|| message(&error))?;
        let mut device_primes = [0_u32; 3];
        unsafe { adynkra_teleparallel_lorentz_descent_primes(device_primes.as_mut_ptr()) };
        if device_primes != PINNED_PRIMES {
            unsafe { adynkra_teleparallel_lorentz_descent_destroy(context.as_ptr()) };
            return Err(format!(
                "Lorentz-descent host/device prime mismatch: host={PINNED_PRIMES:?}, device={device_primes:?}"
            ));
        }
        let resident_bytes =
            unsafe { adynkra_teleparallel_lorentz_descent_resident_bytes(context.as_ptr()) };
        let mut coefficients = vec![ThreePrimeFp2::default(); EXPECTED_RANK * RIGHT_HAND_SIDES];
        let mut residual_counts = [0_u64; 3];
        let mut first_residual_key = u64::MAX;
        let mut first_residual_value = ThreePrimeFp2::default();
        let mut device_milliseconds = 0.0_f32;
        let status = unsafe {
            adynkra_teleparallel_lorentz_descent_solve(
                context.as_ptr(),
                input.matrix.as_ptr(),
                input.matrix.len() as u64,
                input.rhs.as_ptr(),
                input.rhs.len() as u64,
                input.row_to_pivot.as_ptr(),
                input.column_to_basis.as_ptr(),
                input.row_offsets.as_ptr(),
                input.column_indices.as_ptr(),
                input.csr_values.as_ptr(),
                input.csr_values.len() as u64,
                coefficients.as_mut_ptr(),
                coefficients.len() as u64,
                residual_counts.as_mut_ptr(),
                &mut first_residual_key,
                &mut first_residual_value,
                &mut device_milliseconds,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        let high_water_bytes =
            unsafe { adynkra_teleparallel_lorentz_descent_high_water_bytes(context.as_ptr()) };
        unsafe { adynkra_teleparallel_lorentz_descent_destroy(context.as_ptr()) };
        if status != 0 {
            return Err(format!(
                "teleparallel Lorentz-descent CUDA status {status}: {}",
                message(&error)
            ));
        }
        let mut generator_solutions = Vec::with_capacity(RIGHT_HAND_SIDES);
        let mut exact_coordinate_terms = 0;
        for (generator, commutator) in handoff.raw_commutators.iter().enumerate() {
            let mut exact_coefficients = Vec::new();
            for (basis, image_column) in handoff.independent_original_columns.iter().enumerate() {
                let original = image_column.original_d_psi_two_coordinate;
                let exact = commutator
                    .image_coordinates
                    .get(&original)
                    .cloned()
                    .unwrap_or_else(ExactQi::zero);
                let encoded = encode(&exact)?;
                let device = coefficients[basis * RIGHT_HAND_SIDES + generator];
                if encoded != device {
                    return Err(format!(
                        "exact/device Lorentz coefficient mismatch at generator {generator}, basis {basis}"
                    ));
                }
                if !exact.is_zero() {
                    exact_coefficients.push(LorentzGeneratorCoefficient {
                        original_d_psi_two_coordinate: original,
                        value: ExactQiPublic::from(&exact),
                    });
                }
            }
            exact_coordinate_terms += exact_coefficients.len();
            generator_solutions.push(LorentzGeneratorSolution {
                generator_left: commutator.generator_left,
                generator_right: commutator.generator_right,
                exact_image_residual_entries: commutator.exact_image_residual_entries,
                coefficients: exact_coefficients,
            });
        }
        let modular_membership_passed = residual_counts == [0; 3];
        let exact_reconstruction_complete = generator_solutions
            .iter()
            .all(|solution| solution.exact_image_residual_entries == 0);
        let exact_coordinate_sha256 = digest_hex(
            &serde_json::to_vec(&generator_solutions)
                .map_err(|error| format!("serialize exact Lorentz coordinates: {error}"))?,
        );
        let executable_sha256 = std::env::current_exe()
            .ok()
            .and_then(|path| std::fs::read(path).ok())
            .map(|bytes| digest_hex(&bytes))
            .unwrap_or_default();
        let command = std::env::args().collect::<Vec<_>>();
        let host_name = std::fs::read_to_string("/etc/hostname")
            .unwrap_or_default()
            .trim()
            .to_string();
        let nvidia = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default();
        let (gpu_name, driver_version) = nvidia
            .split_once(',')
            .map(|(name, driver)| (name.trim().to_string(), driver.trim().to_string()))
            .unwrap_or_default();
        let nvcc_version = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default();
        Ok(TeleparallelLorentzGpuReport {
            schema_version: "adynkra-11d-teleparallel-lorentz-gpu-v1",
            source_coordinate: handoff.source_coordinate,
            momentum_axis: handoff.momentum_axis,
            image_ambient_columns: handoff.ambient_d_psi_two_columns,
            image_rank: handoff.exact_image_rank,
            right_hand_sides: handoff.raw_commutators.len(),
            matrix_nonzeros: input.matrix.len(),
            rhs_nonzeros: input.rhs.len(),
            selected_csr_nonzeros: input.csr_values.len(),
            matrix_sha256: digest_coo(&input.matrix),
            rhs_sha256: digest_coo(&input.rhs),
            pivot_map_sha256: digest_i32_slices(&[&input.row_to_pivot, &input.column_to_basis]),
            csr_sha256: digest_csr(input),
            cuda_source_sha256: digest_hex(include_bytes!(
                "../cuda/teleparallel_lorentz_descent_cuda.cu"
            )),
            rust_source_sha256: digest_hex(include_bytes!(
                "eleven_dimensional_teleparallel_lorentz_gpu.rs"
            )),
            handoff_source_sha256: digest_hex(include_bytes!(
                "eleven_dimensional_corrected_teleparallel_equivariance.rs"
            )),
            build_rs_sha256: digest_hex(include_bytes!("../build.rs")),
            executable_sha256,
            exact_coordinate_sha256,
            host_name,
            gpu_name,
            driver_version,
            nvcc_version,
            command,
            unix_time_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            exact_handoff_milliseconds: 0.0,
            device_input_milliseconds: 0.0,
            end_to_end_milliseconds: 0.0,
            ordered_primes: PINNED_PRIMES,
            residual_counts,
            first_residual_key: (first_residual_key != u64::MAX).then_some(first_residual_key),
            first_residual_value: (first_residual_key != u64::MAX)
                .then_some(first_residual_value.lane),
            device_milliseconds,
            resident_bytes,
            high_water_bytes,
            exact_coordinate_terms,
            generator_solutions,
            modular_membership_passed,
            exact_reconstruction_complete,
            coherent_cocycle_integrability_complete: false,
            corrected_target_zero_commutator_complete: false,
            boundary: "Three-prime membership of all 55 fixed-source raw commutators in the rank-320 local-Lorentz image is necessary but does not construct a coherent exact compensator or a corrected physical representative.",
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn run_fixed_source_cuda(
    source_coordinate: u32,
) -> Result<TeleparallelLorentzGpuReport, String> {
    let start = Instant::now();
    let handoff = local_lorentz_target_image_handoff(source_coordinate)?;
    let handoff_elapsed = start.elapsed();
    let input = device_input(&handoff)?;
    let input_elapsed = start.elapsed() - handoff_elapsed;
    let mut report = cuda::run(&handoff, &input)?;
    report.exact_handoff_milliseconds = handoff_elapsed.as_secs_f64() * 1_000.0;
    report.device_input_milliseconds = input_elapsed.as_secs_f64() * 1_000.0;
    report.end_to_end_milliseconds = start.elapsed().as_secs_f64() * 1_000.0;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modular_encoding_respects_field_operations() {
        let value = ExactQi::from_rational(-7, 12);
        let encoded = encode(&value).unwrap();
        for (prime_index, &prime) in PINNED_PRIMES.iter().enumerate() {
            assert_eq!(
                (u64::from(encoded.lane[2 * prime_index]) * 12) % u64::from(prime),
                u64::from(prime - 7)
            );
            assert_eq!(encoded.lane[2 * prime_index + 1], 0);
        }
    }

    #[test]
    fn abi_sizes_are_pinned() {
        assert_eq!(std::mem::size_of::<ThreePrimeFp2>(), 24);
        assert_eq!(std::mem::size_of::<ModularCooEntry>(), 32);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires RTX CUDA host and builds the exact all55 handoff"]
    fn fixed_source_all55_membership_cuda() {
        let report = run_fixed_source_cuda(131_857).unwrap();
        let output =
            std::path::Path::new("results/adynkra_11d_teleparallel_lorentz_all55_gpu.json");
        let temporary = output.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        std::fs::rename(&temporary, output).unwrap();
        eprintln!(
            "TELEPARALLEL_LORENTZ_GPU device_ms={} residual={:?} exact_terms={} exact_sha256={}",
            report.device_milliseconds,
            report.residual_counts,
            report.exact_coordinate_terms,
            report.exact_coordinate_sha256,
        );
        assert!(report.modular_membership_passed);
        assert_eq!(report.residual_counts, [0; 3]);
    }
}
