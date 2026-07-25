//! Exact first-derivative gauge-intertwiner census for the proposed 11D
//! spinor prepotential.
//!
//! The six maps are the complete Lorentz-equivariant ansatz
//!
//! delta Psi_alpha =
//!   sum_p c_p (C Gamma^[a1...ap])_{alpha beta}
//!                 D^beta Lambda_[a1...ap],  p = 0,...,5.
//!
//! The exterior model stores lower-index derivative components. The executed
//! map therefore converts the lowered bilinear to the mixed-index operator
//! Gamma^[p] before composition.
//!
//! The cited 11D prepotential papers motivate the spinor prepotential but do
//! not select the coefficients c_p or print an induced gauge transformation
//! for the vector-spinor target. This module constructs the six candidate
//! intertwiners without promoting any of them to a physical gauge symmetry.

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::Ratio;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::eleven_dimensional_clifford::{GaussianRational, Matrix};

const SPINOR_DIMENSION: usize = 32;

fn g(real: i64, imaginary: i64) -> GaussianRational {
    Complex::new(Ratio::from_integer(real), Ratio::from_integer(imaginary))
}

fn conjugate(value: &GaussianRational) -> GaussianRational {
    Complex::new(value.re.clone(), -value.im.clone())
}

fn sparse_matrix(matrix: &Matrix) -> BTreeMap<usize, GaussianRational> {
    let mut result = BTreeMap::new();
    for (row, values) in matrix.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            if *value != g(0, 0) {
                result.insert(row * SPINOR_DIMENSION + column, value.clone());
            }
        }
    }
    result
}

fn hermitian_dot(
    left: &BTreeMap<usize, GaussianRational>,
    right: &BTreeMap<usize, GaussianRational>,
) -> GaussianRational {
    let (smaller, larger, conjugate_smaller) = if left.len() <= right.len() {
        (left, right, true)
    } else {
        (right, left, false)
    };
    smaller.iter().fold(g(0, 0), |sum, (index, value)| {
        let Some(other) = larger.get(index) else {
            return sum;
        };
        if conjugate_smaller {
            sum + conjugate(value) * other.clone()
        } else {
            sum + conjugate(other) * value.clone()
        }
    })
}

fn transpose_mismatches(matrix: &Matrix, symmetry_sign: i64) -> usize {
    let mut mismatches = 0;
    for row in 0..SPINOR_DIMENSION {
        for column in 0..SPINOR_DIMENSION {
            mismatches += usize::from(
                matrix[column][row].clone()
                    != matrix[row][column].clone() * Ratio::from_integer(symmetry_sign),
            );
        }
    }
    mismatches
}

fn monomial_matrix_residuals(matrix: &Matrix) -> (usize, usize, usize) {
    let row_residuals = matrix
        .iter()
        .filter(|row| row.iter().filter(|value| **value != g(0, 0)).count() != 1)
        .count();
    let column_residuals = (0..SPINOR_DIMENSION)
        .filter(|column| matrix.iter().filter(|row| row[*column] != g(0, 0)).count() != 1)
        .count();
    let unit_residuals = matrix
        .iter()
        .flatten()
        .filter(|value| **value != g(0, 0))
        .filter(|value| conjugate(value) * (*value).clone() != g(1, 0))
        .count();
    (row_residuals, column_residuals, unit_residuals)
}

#[derive(Debug, Clone, Serialize)]
pub struct GaugeIntertwinerChannelReport {
    pub form_degree: usize,
    pub dynkin_label: &'static str,
    pub parameter_dimension: usize,
    pub derivative_spinor_dimension: usize,
    pub intertwiner_domain_dimension: usize,
    pub component_matrices: usize,
    pub nonzero_matrix_entries: usize,
    pub component_matrix_rank: usize,
    pub expected_spinor_index_symmetry: &'static str,
    pub transpose_residual_entries: usize,
    pub monomial_row_residuals: usize,
    pub monomial_column_residuals: usize,
    pub unit_coefficient_residuals: usize,
    pub scalar_divergence_zero_at_zero_momentum: bool,
    pub scalar_divergence_zero_at_generic_momentum: bool,
    pub physical_status: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalGaugeIntertwinerReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub proposed_prepotential: &'static str,
    pub candidate_transformation: &'static str,
    pub executed_mixed_index_transformation: &'static str,
    pub degree_zero_operator_is_identity: bool,
    pub parameter_channels: Vec<GaugeIntertwinerChannelReport>,
    pub channel_count: usize,
    pub total_parameter_components: usize,
    pub total_bilinear_matrices: usize,
    pub spinor_square_dimension: usize,
    pub pairwise_inner_products_checked: usize,
    pub diagonal_norm_residuals: usize,
    pub off_diagonal_orthogonality_residuals: usize,
    pub exact_bilinear_basis_rank: usize,
    pub leading_operator_columns: usize,
    pub first_momentum_operator_columns: usize,
    pub zero_momentum_composition_jobs: usize,
    pub first_momentum_composition_jobs: usize,
    pub total_composition_jobs: usize,
    pub zero_momentum_scalar_divergence_kernel_degrees: Vec<usize>,
    pub generic_momentum_scalar_divergence_kernel_degrees: Vec<usize>,
    pub source_selects_channel_coefficients: bool,
    pub source_prints_target_gauge_transformation: bool,
    pub gauge_for_gauge_reducibility_supplied: bool,
    pub source_quotient_condition: &'static str,
    pub target_quotient_condition: &'static str,
    pub next_exact_step: &'static str,
    pub boundary: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GaugeCompositionSpec {
    pub ordinal: usize,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: &'static str,
    pub operator_ordinal: usize,
    pub operator_label: String,
    pub operator_kind: String,
    pub contributes_zero_momentum_d17: bool,
    pub contributes_first_momentum_pd15: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMomentumGaugeCompositionManifest {
    pub schema_version: String,
    pub passed: bool,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: String,
    pub parameter_basis: Vec<Vec<usize>>,
    pub parameter_components: usize,
    pub leading_operator: crate::eleven_dimensional_level16_couplings::JointColumnSpec,
    pub exterior_degree: usize,
    pub composition_is_zero: bool,
    pub nonzero_residual_entries: u64,
    pub exact_squared_norm: String,
    pub maximum_absolute_residual_coefficient: String,
    pub raw_record_bytes: usize,
    pub raw_uncompressed_bytes: u64,
    pub raw_compressed_bytes: u64,
    pub raw_uncompressed_sha256: String,
    pub raw_compressed_sha256: String,
    pub fixture_sha256: String,
    pub source_revision: String,
    pub executable_sha256: String,
    pub host: String,
    pub process_id: u32,
    pub started_unix_seconds: u64,
    pub finished_unix_seconds: u64,
    pub elapsed_milliseconds: u128,
    pub raw_file: String,
    pub convention: String,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMomentumGaugeKernelReport {
    pub schema_version: String,
    pub passed: bool,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: String,
    pub parameter_components: usize,
    pub leading_basis: Vec<String>,
    pub columns: usize,
    pub unique_coordinate_rows: u64,
    pub total_nonzero_residual_entries: u64,
    pub exact_gram_matrix: Vec<Vec<String>>,
    pub exact_rank: usize,
    pub exact_nullity: usize,
    pub primitive_integer_kernel_basis: Vec<Vec<String>>,
    pub kernel_residuals_exactly_zero: bool,
    pub kernel_coefficient_mutation_detected: bool,
    pub individual_zero_columns: Vec<usize>,
    pub all_streams_deep_verified: bool,
    pub source_invariant_leading_dimension: usize,
    pub interpretation: String,
    pub boundary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMomentumGaugeSubsetReport {
    pub channel_mask: usize,
    pub selected_form_degrees: Vec<usize>,
    pub selected_dynkin_labels: Vec<String>,
    pub exact_constraint_rank: usize,
    pub exact_intersection_dimension: usize,
    pub primitive_integer_kernel_basis: Vec<Vec<String>>,
    pub kernel_residuals_exactly_zero: bool,
    pub has_nonzero_leading_operator: bool,
    pub scalar_factorizing_direction_survives: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMomentumGaugeSubsetClassificationReport {
    pub schema_version: String,
    pub passed: bool,
    pub leading_basis: Vec<String>,
    pub channel_dynkin_labels: Vec<String>,
    pub source_kernel_report_sha256: Vec<String>,
    pub all_source_streams_deep_verified: bool,
    pub scalar_factorizing_primitive_coordinates: Vec<String>,
    pub subset_count: usize,
    pub nonempty_subset_count: usize,
    pub individual_channel_dimensions: Vec<usize>,
    pub all_channel_intersection_dimension: usize,
    pub subsets_with_nonzero_leading_operator: usize,
    pub subsets_preserving_scalar_factorizing_direction: usize,
    pub subset_reports: Vec<ZeroMomentumGaugeSubsetReport>,
    pub interpretation: String,
    pub boundary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstMomentumGaugeFunctionalArtifact {
    pub schema_version: String,
    pub passed: bool,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: String,
    pub parameter_components: usize,
    pub operator: crate::eleven_dimensional_level16_couplings::JointColumnSpec,
    pub exterior_degree: usize,
    pub nonzero_residual_entries: u64,
    pub exact_squared_norm: String,
    pub maximum_absolute_residual_coefficient: String,
    pub functional_seeds: Vec<String>,
    pub functional_buckets_per_real_part: usize,
    pub exact_functional_values: Vec<String>,
    pub functional_image_is_nonzero: bool,
    pub fixture_sha256: String,
    pub source_revision: String,
    pub executable_sha256: String,
    pub host: String,
    pub process_id: u32,
    pub started_unix_seconds: u64,
    pub finished_unix_seconds: u64,
    pub elapsed_milliseconds: u128,
    pub convention: String,
    pub interpretation: String,
    pub boundary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstMomentumGaugeStreamFunctionalArtifact {
    pub schema_version: String,
    pub passed: bool,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: String,
    pub parameter_components: usize,
    #[serde(default)]
    pub evaluated_parameter_components: Vec<usize>,
    pub operator: crate::eleven_dimensional_level16_couplings::JointColumnSpec,
    pub exterior_degree: usize,
    pub emitted_nonzero_terms: u64,
    pub maximum_absolute_emitted_term_coefficient: String,
    pub functional_seeds: Vec<String>,
    pub functional_buckets_per_real_part: usize,
    pub exact_functional_values: Vec<String>,
    pub functional_image_is_nonzero: bool,
    pub fixture_sha256: String,
    pub source_revision: String,
    pub executable_sha256: String,
    pub host: String,
    pub process_id: u32,
    pub started_unix_seconds: u64,
    pub finished_unix_seconds: u64,
    pub elapsed_milliseconds: u128,
    pub convention: String,
    pub interpretation: String,
    pub boundary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstMomentumGaugeFunctionalMergeReport {
    pub schema_version: String,
    pub passed: bool,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: String,
    pub parameter_components: usize,
    pub evaluated_parameter_components: Vec<usize>,
    pub parameter_projection_is_complete: bool,
    pub leading_basis: Vec<String>,
    pub first_momentum_basis: Vec<String>,
    pub zero_momentum_kernel_dimension: usize,
    pub zero_momentum_kernel_basis: Vec<Vec<String>>,
    pub parameterized_columns: usize,
    pub functional_rows: usize,
    pub exact_functional_rank: usize,
    pub exact_functional_nullity: usize,
    pub functional_kernel_leading_projection_rank: usize,
    pub nonzero_leading_extension_excluded_by_functionals: bool,
    pub functional_primitive_integer_kernel_basis: Vec<Vec<String>>,
    pub functional_kernel_residuals_exactly_zero: bool,
    pub source_artifact_sha256: Vec<String>,
    pub zero_momentum_kernel_report_sha256: String,
    pub interpretation: String,
    pub boundary: String,
}

const FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS: usize = 64;
const FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS: [u64; 4] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
];

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock precedes the Unix epoch")
        .as_secs()
}

fn write_hashed<W: Write>(
    writer: &mut W,
    hasher: &mut Sha256,
    byte_count: &mut u64,
    bytes: &[u8],
) -> io::Result<()> {
    writer.write_all(bytes)?;
    hasher.update(bytes);
    *byte_count = byte_count
        .checked_add(u64::try_from(bytes.len()).unwrap())
        .expect("raw artifact byte count overflow");
    Ok(())
}

fn write_json_durable<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()
}

pub fn verify_zero_momentum_gauge_composition_artifact(
    directory: &Path,
    gauge_form_degree: usize,
    leading_ordinal: usize,
    verify_uncompressed_stream: bool,
) -> io::Result<ZeroMomentumGaugeCompositionManifest> {
    let manifest_path = directory.join("manifest.json");
    let manifest: ZeroMomentumGaugeCompositionManifest =
        serde_json::from_reader(BufReader::new(File::open(manifest_path)?))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !manifest.passed
        || manifest.gauge_form_degree != gauge_form_degree
        || manifest.leading_operator.ordinal != leading_ordinal
        || manifest.leading_operator.kind != "leading"
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "gauge-composition manifest does not match the requested job",
        ));
    }
    let raw_path = directory.join(&manifest.raw_file);
    if sha256_file(&raw_path)? != manifest.raw_compressed_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compressed gauge-composition hash mismatch",
        ));
    }
    if verify_uncompressed_stream {
        let mut decoder = zstd::stream::read::Decoder::new(BufReader::new(File::open(raw_path)?))?;
        let mut hasher = Sha256::new();
        let mut count = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = decoder.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            count += u64::try_from(read).unwrap();
        }
        if count != manifest.raw_uncompressed_bytes
            || format!("{:x}", hasher.finalize()) != manifest.raw_uncompressed_sha256
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "uncompressed gauge-composition stream failed verification",
            ));
        }
    }
    Ok(manifest)
}

pub fn build_and_write_zero_momentum_gauge_composition_artifact(
    gauge_form_degree: usize,
    leading_ordinal: usize,
    output_root: &Path,
) -> io::Result<ZeroMomentumGaugeCompositionManifest> {
    if gauge_form_degree > 5 || leading_ordinal >= 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "zero-momentum gauge job requires form degree 0..5 and leading ordinal 0..11",
        ));
    }
    let completed_root = output_root
        .join("complete")
        .join(format!("form-{gauge_form_degree}"));
    let incomplete_root = output_root
        .join("incomplete")
        .join(format!("form-{gauge_form_degree}"));
    fs::create_dir_all(&completed_root)?;
    fs::create_dir_all(&incomplete_root)?;
    let final_directory = completed_root.join(format!("column-{leading_ordinal:03}"));
    if final_directory.exists() {
        return verify_zero_momentum_gauge_composition_artifact(
            &final_directory,
            gauge_form_degree,
            leading_ordinal,
            false,
        );
    }

    let started_unix_seconds = unix_seconds();
    let timer = Instant::now();
    let temporary_directory = incomplete_root.join(format!(
        "column-{leading_ordinal:03}-{}-{started_unix_seconds}",
        std::process::id()
    ));
    fs::create_dir(&temporary_directory)?;
    let raw_name = "residual.i128le.zst";
    let raw_path = temporary_directory.join(raw_name);
    let raw_file = File::create(&raw_path)?;
    let buffered = BufWriter::new(raw_file);
    let mut encoder = zstd::stream::write::Encoder::new(buffered, 1)?;
    let mut uncompressed_hasher = Sha256::new();
    let mut uncompressed_bytes = 0_u64;
    write_hashed(
        &mut encoder,
        &mut uncompressed_hasher,
        &mut uncompressed_bytes,
        b"AGD17V3\0",
    )?;
    write_hashed(
        &mut encoder,
        &mut uncompressed_hasher,
        &mut uncompressed_bytes,
        &[u8::try_from(gauge_form_degree).unwrap()],
    )?;
    write_hashed(
        &mut encoder,
        &mut uncompressed_hasher,
        &mut uncompressed_bytes,
        &u16::try_from(leading_ordinal).unwrap().to_le_bytes(),
    )?;
    let mut exact_squared_norm = BigInt::from(0);
    let mut nonzero_residual_entries = 0_u64;
    let (leading_operator, parameter_basis, maximum, fixture_sha256) =
        crate::eleven_dimensional_level16_couplings::visit_zero_momentum_gauge_composition_components(
            gauge_form_degree,
            leading_ordinal,
            |_, _, component_residual| {
                for entry in component_residual {
                    let real = BigInt::from(entry.real);
                    let imaginary = BigInt::from(entry.imaginary);
                    exact_squared_norm += &real * &real + &imaginary * &imaginary;
                    nonzero_residual_entries += 1;
                    write_hashed(
                        &mut encoder,
                        &mut uncompressed_hasher,
                        &mut uncompressed_bytes,
                        &u16::try_from(entry.parameter_component_index)
                            .unwrap()
                            .to_le_bytes(),
                    )?;
                    write_hashed(
                        &mut encoder,
                        &mut uncompressed_hasher,
                        &mut uncompressed_bytes,
                        &entry.exterior_mask.to_le_bytes(),
                    )?;
                    write_hashed(
                        &mut encoder,
                        &mut uncompressed_hasher,
                        &mut uncompressed_bytes,
                        &entry.real.to_le_bytes(),
                    )?;
                    write_hashed(
                        &mut encoder,
                        &mut uncompressed_hasher,
                        &mut uncompressed_bytes,
                        &entry.imaginary.to_le_bytes(),
                    )?;
                }
                Ok(())
            },
        )?;
    let mut buffered = encoder.finish()?;
    buffered.flush()?;
    buffered.get_ref().sync_all()?;
    let raw_uncompressed_sha256 = format!("{:x}", uncompressed_hasher.finalize());
    let raw_compressed_sha256 = sha256_file(&raw_path)?;
    let raw_compressed_bytes = fs::metadata(&raw_path)?.len();
    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let finished_unix_seconds = unix_seconds();
    let manifest = ZeroMomentumGaugeCompositionManifest {
        schema_version: "adynkra-11d-zero-momentum-gauge-composition-v3".to_string(),
        passed: true,
        gauge_form_degree,
        parameter_dynkin_label: labels[gauge_form_degree].to_string(),
        parameter_components: parameter_basis.len(),
        parameter_basis,
        leading_operator,
        exterior_degree: 17,
        composition_is_zero: nonzero_residual_entries == 0,
        nonzero_residual_entries,
        exact_squared_norm: exact_squared_norm.to_string(),
        maximum_absolute_residual_coefficient: maximum.to_string(),
        raw_record_bytes: 38,
        raw_uncompressed_bytes: uncompressed_bytes,
        raw_compressed_bytes,
        raw_uncompressed_sha256,
        raw_compressed_sha256,
        fixture_sha256,
        source_revision: std::env::var("ADINKRA_SOURCE_REVISION")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        executable_sha256: std::env::var("ADINKRA_EXECUTABLE_SHA256")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        host: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        process_id: std::process::id(),
        started_unix_seconds,
        finished_unix_seconds,
        elapsed_milliseconds: timer.elapsed().as_millis(),
        raw_file: raw_name.to_string(),
        convention: "AGD17V3 little-endian stream: 8-byte magic, u8 form degree, u16 leading ordinal, then manifest-counted 38-byte records (u16 parameter component, u32 exterior mask, i128 real, i128 imaginary); the lowered derivative index is acted on by the mixed-index Gamma^[p] operator, equivalent to (C Gamma^[p]) acting on D^beta; parameter components are processed and written separately to bound memory; the gauge derivative is right-wedged after the sixteen operator derivatives and normalized into ascending exterior order; zstd level 1".to_string(),
        interpretation: "This artifact certifies one exact source variation A G_p. A nonzero residual excludes source invariance for this individual operator but does not decide linear combinations of the twelve leading operators.".to_string(),
    };
    write_json_durable(&temporary_directory.join("manifest.json"), &manifest)?;
    File::open(&temporary_directory)?.sync_all()?;
    fs::rename(&temporary_directory, &final_directory)?;
    File::open(&completed_root)?.sync_all()?;
    verify_zero_momentum_gauge_composition_artifact(
        &final_directory,
        gauge_form_degree,
        leading_ordinal,
        false,
    )
}

#[derive(Debug, Clone, Copy)]
struct GaugeResidualRecord {
    parameter_component_index: u16,
    exterior_mask: u32,
    real: i128,
    imaginary: i128,
}

impl GaugeResidualRecord {
    fn key(self) -> (u16, u32) {
        (self.parameter_component_index, self.exterior_mask)
    }
}

struct GaugeResidualReader {
    reader: Box<dyn Read>,
    remaining: u64,
    previous_key: Option<(u16, u32)>,
}

impl GaugeResidualReader {
    fn open(
        raw_path: &Path,
        gauge_form_degree: usize,
        leading_ordinal: usize,
        expected_records: u64,
    ) -> io::Result<Self> {
        let decoder = zstd::stream::read::Decoder::new(BufReader::new(File::open(raw_path)?))?;
        let mut reader: Box<dyn Read> = Box::new(decoder);
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        let header_contains_count = if &magic == b"AGD17V1\0" {
            true
        } else if &magic == b"AGD17V2\0" || &magic == b"AGD17V3\0" {
            false
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid zero-momentum gauge stream magic",
            ));
        };
        let mut form = [0_u8; 1];
        reader.read_exact(&mut form)?;
        let mut ordinal = [0_u8; 2];
        reader.read_exact(&mut ordinal)?;
        let stream_ordinal = usize::from(u16::from_le_bytes(ordinal));
        let stream_count = if header_contains_count {
            let mut count = [0_u8; 8];
            reader.read_exact(&mut count)?;
            u64::from_le_bytes(count)
        } else {
            expected_records
        };
        if usize::from(form[0]) != gauge_form_degree
            || stream_ordinal != leading_ordinal
            || stream_count != expected_records
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zero-momentum gauge stream header does not match its manifest",
            ));
        }
        Ok(Self {
            reader,
            remaining: stream_count,
            previous_key: None,
        })
    }

    fn next_record(&mut self) -> io::Result<Option<GaugeResidualRecord>> {
        if self.remaining == 0 {
            let mut trailing = [0_u8; 1];
            if self.reader.read(&mut trailing)? != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "zero-momentum gauge stream has trailing bytes",
                ));
            }
            return Ok(None);
        }
        let mut component = [0_u8; 2];
        let mut mask = [0_u8; 4];
        let mut real = [0_u8; 16];
        let mut imaginary = [0_u8; 16];
        self.reader.read_exact(&mut component)?;
        self.reader.read_exact(&mut mask)?;
        self.reader.read_exact(&mut real)?;
        self.reader.read_exact(&mut imaginary)?;
        let record = GaugeResidualRecord {
            parameter_component_index: u16::from_le_bytes(component),
            exterior_mask: u32::from_le_bytes(mask),
            real: i128::from_le_bytes(real),
            imaginary: i128::from_le_bytes(imaginary),
        };
        if record.real == 0 && record.imaginary == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zero-momentum gauge stream contains an explicit zero record",
            ));
        }
        if self
            .previous_key
            .is_some_and(|previous| previous >= record.key())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zero-momentum gauge stream is not strictly key-sorted",
            ));
        }
        self.previous_key = Some(record.key());
        self.remaining -= 1;
        Ok(Some(record))
    }
}

pub fn merge_zero_momentum_gauge_composition_artifacts(
    gauge_form_degree: usize,
    output_root: &Path,
    deep_verify: bool,
) -> io::Result<ZeroMomentumGaugeKernelReport> {
    if gauge_form_degree > 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "gauge form degree must lie in 0..5",
        ));
    }
    let mut manifests = Vec::with_capacity(12);
    let mut readers = Vec::with_capacity(12);
    for leading_ordinal in 0..12 {
        let directory = output_root
            .join("complete")
            .join(format!("form-{gauge_form_degree}"))
            .join(format!("column-{leading_ordinal:03}"));
        let manifest = verify_zero_momentum_gauge_composition_artifact(
            &directory,
            gauge_form_degree,
            leading_ordinal,
            deep_verify,
        )?;
        let reader = GaugeResidualReader::open(
            &directory.join(&manifest.raw_file),
            gauge_form_degree,
            leading_ordinal,
            manifest.nonzero_residual_entries,
        )?;
        manifests.push(manifest);
        readers.push(reader);
    }
    let reference_basis = manifests[0].parameter_basis.clone();
    if manifests
        .iter()
        .any(|manifest| manifest.parameter_basis != reference_basis)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "gauge parameter bases differ across leading columns",
        ));
    }

    let mut current = readers
        .iter_mut()
        .map(GaugeResidualReader::next_record)
        .collect::<io::Result<Vec<_>>>()?;
    let mut gram = vec![vec![BigInt::from(0); 12]; 12];
    let mut unique_coordinate_rows = 0_u64;
    while let Some(next_key) = current.iter().flatten().map(|record| record.key()).min() {
        let mut coordinate = [(0_i128, 0_i128); 12];
        for column in 0..12 {
            if current[column].is_some_and(|record| record.key() == next_key) {
                let record = current[column].take().unwrap();
                coordinate[column] = (record.real, record.imaginary);
                current[column] = readers[column].next_record()?;
            }
        }
        for left in 0..12 {
            let (left_real, left_imaginary) = coordinate[left];
            if left_real == 0 && left_imaginary == 0 {
                continue;
            }
            for right in left..12 {
                let (right_real, right_imaginary) = coordinate[right];
                if right_real == 0 && right_imaginary == 0 {
                    continue;
                }
                gram[left][right] += BigInt::from(left_real) * BigInt::from(right_real)
                    + BigInt::from(left_imaginary) * BigInt::from(right_imaginary);
            }
        }
        unique_coordinate_rows += 1;
    }
    for left in 0..12 {
        for right in 0..left {
            gram[left][right] = gram[right][left].clone();
        }
    }
    for (column, manifest) in manifests.iter().enumerate() {
        if gram[column][column].to_string() != manifest.exact_squared_norm {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stream norm does not match the zero-momentum manifest",
            ));
        }
    }

    let rational_gram = gram
        .iter()
        .map(|row| {
            row.iter()
                .cloned()
                .map(Ratio::from_integer)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let exact_rank =
        crate::eleven_dimensional_level16_couplings::rational_matrix_rank(&rational_gram);
    let rational_kernel =
        crate::eleven_dimensional_level16_couplings::rational_nullspace(&rational_gram);
    let primitive_kernel = rational_kernel
        .iter()
        .map(|vector| crate::eleven_dimensional_level16_couplings::primitive_bigint_vector(vector))
        .collect::<Vec<_>>();
    let kernel_residuals_exactly_zero = primitive_kernel.iter().all(|vector| {
        gram.iter().all(|row| {
            row.iter()
                .zip(vector)
                .fold(BigInt::from(0), |sum, (coefficient, value)| {
                    sum + coefficient * value
                })
                == BigInt::from(0)
        })
    });
    let kernel_coefficient_mutation_detected = primitive_kernel.iter().all(|vector| {
        (0..vector.len()).any(|index| {
            let mut mutated = vector.clone();
            mutated[index] += BigInt::from(1);
            gram.iter().any(|row| {
                row.iter()
                    .zip(&mutated)
                    .fold(BigInt::from(0), |sum, (coefficient, value)| {
                        sum + coefficient * value
                    })
                    != BigInt::from(0)
            })
        })
    });
    let primitive_integer_kernel_basis = primitive_kernel
        .iter()
        .map(|vector| vector.iter().map(ToString::to_string).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let exact_nullity = primitive_integer_kernel_basis.len();
    let individual_zero_columns = manifests
        .iter()
        .enumerate()
        .filter(|(_, manifest)| manifest.composition_is_zero)
        .map(|(column, _)| column)
        .collect::<Vec<_>>();
    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let report = ZeroMomentumGaugeKernelReport {
        schema_version: "adynkra-11d-zero-momentum-gauge-kernel-v1".to_string(),
        passed: exact_rank + exact_nullity == 12
            && kernel_residuals_exactly_zero
            && kernel_coefficient_mutation_detected,
        gauge_form_degree,
        parameter_dynkin_label: labels[gauge_form_degree].to_string(),
        parameter_components: reference_basis.len(),
        leading_basis: manifests
            .iter()
            .map(|manifest| manifest.leading_operator.label.clone())
            .collect(),
        columns: manifests.len(),
        unique_coordinate_rows,
        total_nonzero_residual_entries: manifests
            .iter()
            .map(|manifest| manifest.nonzero_residual_entries)
            .sum(),
        exact_gram_matrix: gram
            .iter()
            .map(|row| row.iter().map(ToString::to_string).collect())
            .collect(),
        exact_rank,
        exact_nullity,
        primitive_integer_kernel_basis,
        kernel_residuals_exactly_zero,
        kernel_coefficient_mutation_detected,
        individual_zero_columns,
        all_streams_deep_verified: deep_verify,
        source_invariant_leading_dimension: exact_nullity,
        interpretation: "The primitive kernel basis is the complete rational subspace of the twelve leading operators whose D17 variation vanishes for this source-gauge channel.".to_string(),
        boundary: "This source-invariance result does not supply an induced target gauge transformation and does not quotient a nonzero hook residual.".to_string(),
    };
    let merge_root = output_root.join("merge");
    fs::create_dir_all(&merge_root)?;
    let output = merge_root.join(format!("zero-momentum-form-{gauge_form_degree}.json"));
    let temporary = merge_root.join(format!(
        ".zero-momentum-form-{gauge_form_degree}.json.{}.tmp",
        std::process::id()
    ));
    write_json_durable(&temporary, &report)?;
    fs::rename(temporary, output)?;
    File::open(&merge_root)?.sync_all()?;
    Ok(report)
}

fn parse_exact_gram(report: &ZeroMomentumGaugeKernelReport) -> io::Result<Vec<Vec<Ratio<BigInt>>>> {
    if report.exact_gram_matrix.len() != 12
        || report.exact_gram_matrix.iter().any(|row| row.len() != 12)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zero-momentum gauge Gram matrix is not 12 by 12",
        ));
    }
    report
        .exact_gram_matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| {
                    value
                        .parse::<BigInt>()
                        .map(Ratio::from_integer)
                        .map_err(|error| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("invalid exact Gram entry: {error}"),
                            )
                        })
                })
                .collect::<io::Result<Vec<_>>>()
        })
        .collect()
}

fn standard_basis(dimension: usize) -> Vec<Vec<Ratio<BigInt>>> {
    (0..dimension)
        .map(|column| {
            (0..dimension)
                .map(|row| Ratio::from_integer(BigInt::from(usize::from(row == column))))
                .collect()
        })
        .collect()
}

fn exact_constraint_kernel(
    rows: &[Vec<Ratio<BigInt>>],
    dimension: usize,
) -> (usize, Vec<Vec<Ratio<BigInt>>>) {
    if rows.is_empty() {
        return (0, standard_basis(dimension));
    }
    (
        crate::eleven_dimensional_level16_couplings::rational_matrix_rank(rows),
        crate::eleven_dimensional_level16_couplings::rational_nullspace(rows),
    )
}

fn rational_dot(left: &[Ratio<BigInt>], right: &[Ratio<BigInt>]) -> Ratio<BigInt> {
    left.iter()
        .zip(right)
        .fold(Ratio::from_integer(BigInt::from(0)), |sum, (a, b)| {
            sum + a.clone() * b.clone()
        })
}

pub fn classify_zero_momentum_gauge_channel_subsets(
    output_root: &Path,
) -> io::Result<ZeroMomentumGaugeSubsetClassificationReport> {
    let mut source_reports = Vec::with_capacity(6);
    let mut source_report_sha256 = Vec::with_capacity(6);
    for gauge_form_degree in 0..=5 {
        let path = output_root
            .join("merge")
            .join(format!("zero-momentum-form-{gauge_form_degree}.json"));
        let payload = fs::read(&path)?;
        let report: ZeroMomentumGaugeKernelReport =
            serde_json::from_slice(&payload).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("failed to parse {}: {error}", path.display()),
                )
            })?;
        if !report.passed
            || report.gauge_form_degree != gauge_form_degree
            || report.columns != 12
            || report.exact_rank + report.exact_nullity != 12
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid zero-momentum gauge kernel report {}",
                    path.display()
                ),
            ));
        }
        source_report_sha256.push(sha256_file(&path)?);
        source_reports.push(report);
    }
    let leading_basis = source_reports[0].leading_basis.clone();
    if source_reports
        .iter()
        .any(|report| report.leading_basis != leading_basis)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "leading bases differ across zero-momentum gauge kernel reports",
        ));
    }
    let exact_grams = source_reports
        .iter()
        .map(parse_exact_gram)
        .collect::<io::Result<Vec<_>>>()?;

    // This is the primitive integer normalization of the exact rational
    // coordinates in adynkra_11d_level17_derivative_matrix.json.
    let scalar_factorizing = [
        70_560_i64, 10_080, -3_780, -15_120, 7_560, -498_960, -5_040, -35, 63, 0, -120, 0,
    ]
    .into_iter()
    .map(|value| Ratio::from_integer(BigInt::from(value)))
    .collect::<Vec<_>>();

    let mut subset_reports = Vec::with_capacity(64);
    for channel_mask in 0_usize..64 {
        let selected_form_degrees = (0..6)
            .filter(|degree| channel_mask & (1 << degree) != 0)
            .collect::<Vec<_>>();
        let selected_dynkin_labels = selected_form_degrees
            .iter()
            .map(|degree| source_reports[*degree].parameter_dynkin_label.clone())
            .collect::<Vec<_>>();
        let constraint_rows = selected_form_degrees
            .iter()
            .flat_map(|degree| exact_grams[*degree].iter().cloned())
            .collect::<Vec<_>>();
        let (exact_constraint_rank, rational_kernel) =
            exact_constraint_kernel(&constraint_rows, 12);
        let primitive_kernel = rational_kernel
            .iter()
            .map(|vector| {
                crate::eleven_dimensional_level16_couplings::primitive_bigint_vector(vector)
            })
            .collect::<Vec<_>>();
        let kernel_residuals_exactly_zero = rational_kernel.iter().all(|vector| {
            constraint_rows
                .iter()
                .all(|row| rational_dot(row, vector).is_zero())
        });
        let scalar_factorizing_direction_survives = constraint_rows
            .iter()
            .all(|row| rational_dot(row, &scalar_factorizing).is_zero());
        subset_reports.push(ZeroMomentumGaugeSubsetReport {
            channel_mask,
            selected_form_degrees,
            selected_dynkin_labels,
            exact_constraint_rank,
            exact_intersection_dimension: rational_kernel.len(),
            primitive_integer_kernel_basis: primitive_kernel
                .iter()
                .map(|vector| vector.iter().map(ToString::to_string).collect())
                .collect(),
            kernel_residuals_exactly_zero,
            has_nonzero_leading_operator: !rational_kernel.is_empty(),
            scalar_factorizing_direction_survives,
        });
    }

    let monotone_dimensions = subset_reports.iter().all(|smaller| {
        subset_reports.iter().all(|larger| {
            smaller.channel_mask & larger.channel_mask != smaller.channel_mask
                || larger.exact_intersection_dimension <= smaller.exact_intersection_dimension
        })
    });
    let individual_channel_dimensions = (0..6)
        .map(|degree| subset_reports[1 << degree].exact_intersection_dimension)
        .collect::<Vec<_>>();
    let individual_dimensions_match = individual_channel_dimensions
        .iter()
        .zip(&source_reports)
        .all(|(dimension, report)| *dimension == report.exact_nullity);
    let all_source_streams_deep_verified = source_reports
        .iter()
        .all(|report| report.all_streams_deep_verified);
    let all_channel_intersection_dimension = subset_reports[63].exact_intersection_dimension;
    let subsets_with_nonzero_leading_operator = subset_reports
        .iter()
        .filter(|subset| subset.has_nonzero_leading_operator)
        .count();
    let subsets_preserving_scalar_factorizing_direction = subset_reports
        .iter()
        .filter(|subset| subset.scalar_factorizing_direction_survives)
        .count();
    let passed = subset_reports.len() == 64
        && subset_reports[0].exact_intersection_dimension == 12
        && subset_reports
            .iter()
            .all(|subset| subset.exact_constraint_rank + subset.exact_intersection_dimension == 12)
        && subset_reports
            .iter()
            .all(|subset| subset.kernel_residuals_exactly_zero)
        && monotone_dimensions
        && individual_dimensions_match
        && all_source_streams_deep_verified;
    let report = ZeroMomentumGaugeSubsetClassificationReport {
        schema_version: "adynkra-11d-zero-momentum-gauge-subsets-v1".to_string(),
        passed,
        leading_basis,
        channel_dynkin_labels: source_reports
            .iter()
            .map(|source| source.parameter_dynkin_label.clone())
            .collect(),
        source_kernel_report_sha256: source_report_sha256,
        all_source_streams_deep_verified,
        scalar_factorizing_primitive_coordinates: scalar_factorizing
            .iter()
            .map(|value| value.to_integer().to_string())
            .collect(),
        subset_count: subset_reports.len(),
        nonempty_subset_count: subset_reports.len() - 1,
        individual_channel_dimensions,
        all_channel_intersection_dimension,
        subsets_with_nonzero_leading_operator,
        subsets_preserving_scalar_factorizing_direction,
        subset_reports,
        interpretation: "Each subset entry is the exact intersection of the selected source-channel kernels on the twelve leading operators.".to_string(),
        boundary: "This report classifies zero-momentum source invariance only. It does not supply a target gauge transformation or decide first-momentum compatibility.".to_string(),
    };
    let merge_root = output_root.join("merge");
    fs::create_dir_all(&merge_root)?;
    let output = merge_root.join("zero-momentum-channel-subsets.json");
    let temporary = merge_root.join(format!(
        ".zero-momentum-channel-subsets.json.{}.tmp",
        std::process::id()
    ));
    write_json_durable(&temporary, &report)?;
    fs::rename(temporary, output)?;
    File::open(&merge_root)?.sync_all()?;
    Ok(report)
}

fn splitmix64_gauge(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn first_momentum_functional_directory(
    output_root: &Path,
    gauge_form_degree: usize,
    operator_ordinal: usize,
) -> std::path::PathBuf {
    output_root
        .join("functional")
        .join("complete")
        .join(format!("form-{gauge_form_degree}"))
        .join(format!("column-{operator_ordinal:03}"))
}

pub fn verify_first_momentum_gauge_functional_artifact(
    directory: &Path,
    gauge_form_degree: usize,
    operator_ordinal: usize,
) -> io::Result<FirstMomentumGaugeFunctionalArtifact> {
    let path = directory.join("manifest.json");
    let artifact: FirstMomentumGaugeFunctionalArtifact =
        serde_json::from_reader(BufReader::new(File::open(&path)?))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let expected_spec = crate::eleven_dimensional_level16_couplings::joint_column_specs()
        .get(operator_ordinal)
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "first-momentum operator ordinal must lie in 0..56",
            )
        })?;
    if !artifact.passed
        || artifact.gauge_form_degree != gauge_form_degree
        || artifact.operator != expected_spec
        || artifact.exterior_degree != 15
        || artifact.functional_buckets_per_real_part != FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS
        || artifact.functional_seeds
            != FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS
                .iter()
                .map(|seed| format!("{seed:016x}"))
                .collect::<Vec<_>>()
        || artifact.exact_functional_values.len()
            != FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len()
                * 2
                * FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid first-momentum functional artifact {}",
                path.display()
            ),
        ));
    }
    Ok(artifact)
}

fn verify_first_momentum_gauge_functional_column(
    directory: &Path,
    gauge_form_degree: usize,
    operator_ordinal: usize,
) -> io::Result<(usize, Vec<usize>, Vec<String>)> {
    let path = directory.join("manifest.json");
    let payload = fs::read(&path)?;
    let schema_version = serde_json::from_slice::<serde_json::Value>(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    if schema_version == "adynkra-11d-first-momentum-gauge-functional-v1" {
        let artifact = verify_first_momentum_gauge_functional_artifact(
            directory,
            gauge_form_degree,
            operator_ordinal,
        )?;
        return Ok((
            artifact.parameter_components,
            (0..artifact.parameter_components).collect(),
            artifact.exact_functional_values,
        ));
    }
    if schema_version != "adynkra-11d-first-momentum-gauge-stream-functional-v2" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported first-momentum functional schema in {}",
                path.display()
            ),
        ));
    }
    let artifact: FirstMomentumGaugeStreamFunctionalArtifact = serde_json::from_slice(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let expected_spec = crate::eleven_dimensional_level16_couplings::joint_column_specs()
        .get(operator_ordinal)
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "first-momentum operator ordinal must lie in 0..56",
            )
        })?;
    let expected_parameter_components = [1, 11, 55, 165, 330, 462]
        .get(gauge_form_degree)
        .copied()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "gauge form degree must lie in 0..5",
            )
        })?;
    let evaluated_parameter_components = if artifact.evaluated_parameter_components.is_empty() {
        (0..artifact.parameter_components).collect::<Vec<_>>()
    } else {
        artifact.evaluated_parameter_components.clone()
    };
    let mut normalized_evaluated_parameter_components = evaluated_parameter_components.clone();
    normalized_evaluated_parameter_components.sort_unstable();
    normalized_evaluated_parameter_components.dedup();
    if !artifact.passed
        || artifact.gauge_form_degree != gauge_form_degree
        || artifact.parameter_components != expected_parameter_components
        || evaluated_parameter_components.is_empty()
        || normalized_evaluated_parameter_components != evaluated_parameter_components
        || evaluated_parameter_components
            .iter()
            .any(|&index| index >= expected_parameter_components)
        || artifact.operator != expected_spec
        || artifact.exterior_degree != 15
        || artifact.functional_buckets_per_real_part != FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS
        || artifact.functional_seeds
            != FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS
                .iter()
                .map(|seed| format!("{seed:016x}"))
                .collect::<Vec<_>>()
        || artifact.exact_functional_values.len()
            != FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len()
                * 2
                * FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS
        || artifact
            .exact_functional_values
            .iter()
            .any(|value| value.parse::<BigInt>().is_err())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid streamed first-momentum functional artifact {}",
                path.display()
            ),
        ));
    }
    Ok((
        artifact.parameter_components,
        evaluated_parameter_components,
        artifact.exact_functional_values,
    ))
}

pub fn build_and_write_first_momentum_gauge_functional_artifact(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    output_root: &Path,
) -> io::Result<FirstMomentumGaugeFunctionalArtifact> {
    if gauge_form_degree > 5 || operator_ordinal >= 56 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "first-momentum functional job requires form degree 0..5 and operator ordinal 0..55",
        ));
    }
    let final_directory =
        first_momentum_functional_directory(output_root, gauge_form_degree, operator_ordinal);
    if final_directory.exists() {
        return verify_first_momentum_gauge_functional_artifact(
            &final_directory,
            gauge_form_degree,
            operator_ordinal,
        );
    }
    let incomplete_root = output_root
        .join("functional")
        .join("incomplete")
        .join(format!("form-{gauge_form_degree}"));
    let completed_root = output_root
        .join("functional")
        .join("complete")
        .join(format!("form-{gauge_form_degree}"));
    fs::create_dir_all(&incomplete_root)?;
    fs::create_dir_all(&completed_root)?;
    let started_unix_seconds = unix_seconds();
    let timer = Instant::now();
    let temporary_directory = incomplete_root.join(format!(
        "column-{operator_ordinal:03}-{}-{started_unix_seconds}",
        std::process::id()
    ));
    fs::create_dir(&temporary_directory)?;

    let rows_per_seed = 2 * FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS;
    let mut exact_functional_values =
        vec![BigInt::from(0); FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len() * rows_per_seed];
    let mut exact_squared_norm = BigInt::from(0);
    let mut nonzero_residual_entries = 0_u64;
    let (operator, parameter_basis, maximum, fixture_sha256) =
        crate::eleven_dimensional_level16_couplings::visit_first_momentum_gauge_composition_components(
            gauge_form_degree,
            operator_ordinal,
            |_, _, component_residual| {
                for entry in component_residual {
                    nonzero_residual_entries = nonzero_residual_entries
                        .checked_add(1)
                        .expect("first-momentum residual record count overflow");
                    let real = BigInt::from(entry.real);
                    let imaginary = BigInt::from(entry.imaginary);
                    exact_squared_norm += &real * &real + &imaginary * &imaginary;
                    let key = u64::from(entry.exterior_mask)
                        | (u64::try_from(entry.momentum_vector_index).unwrap() << 32)
                        | (u64::try_from(entry.parameter_component_index).unwrap() << 36);
                    for (seed_index, seed) in
                        FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.iter().enumerate()
                    {
                        let hash = splitmix64_gauge(key ^ seed);
                        let bucket =
                            (hash as usize) % FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS;
                        let base = seed_index * rows_per_seed;
                        if hash >> 63 == 0 {
                            exact_functional_values[base + bucket] += &real;
                            exact_functional_values
                                [base + FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS + bucket] +=
                                &imaginary;
                        } else {
                            exact_functional_values[base + bucket] -= &real;
                            exact_functional_values
                                [base + FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS + bucket] -=
                                &imaginary;
                        }
                    }
                }
                Ok(())
            },
        )?;
    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let finished_unix_seconds = unix_seconds();
    let functional_image_is_nonzero = exact_functional_values.iter().any(|value| !value.is_zero());
    let artifact = FirstMomentumGaugeFunctionalArtifact {
        schema_version: "adynkra-11d-first-momentum-gauge-functional-v1".to_string(),
        passed: parameter_basis.len() == [1, 11, 55, 165, 330, 462][gauge_form_degree]
            && operator.ordinal == operator_ordinal
            && exact_functional_values.len()
                == FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len() * rows_per_seed,
        gauge_form_degree,
        parameter_dynkin_label: labels[gauge_form_degree].to_string(),
        parameter_components: parameter_basis.len(),
        operator,
        exterior_degree: 15,
        nonzero_residual_entries,
        exact_squared_norm: exact_squared_norm.to_string(),
        maximum_absolute_residual_coefficient: maximum.to_string(),
        functional_seeds: FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS
            .iter()
            .map(|seed| format!("{seed:016x}"))
            .collect(),
        functional_buckets_per_real_part: FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS,
        exact_functional_values: exact_functional_values
            .iter()
            .map(ToString::to_string)
            .collect(),
        functional_image_is_nonzero,
        fixture_sha256,
        source_revision: std::env::var("ADINKRA_SOURCE_REVISION")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        executable_sha256: std::env::var("ADINKRA_EXECUTABLE_SHA256")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        host: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        process_id: std::process::id(),
        started_unix_seconds,
        finished_unix_seconds,
        elapsed_milliseconds: timer.elapsed().as_millis(),
        convention: "exact signed-hash linear image of p D^15 Lambda coordinates; key packs the 15-derivative exterior mask, momentum-vector index, and gauge-parameter component; four fixed splitmix64 seeds, 64 real and 64 imaginary buckets per seed".to_string(),
        interpretation: "This artifact evaluates one of the 56 operator columns after one candidate source-gauge map and records an exact linear image of its complete first-momentum residual.".to_string(),
        boundary: "A nonzero functional value certifies a nonzero full residual. A functional kernel may be larger than the full residual kernel and cannot by itself certify a surviving operator.".to_string(),
    };
    write_json_durable(&temporary_directory.join("manifest.json"), &artifact)?;
    File::open(&temporary_directory)?.sync_all()?;
    fs::rename(&temporary_directory, &final_directory)?;
    File::open(&completed_root)?.sync_all()?;
    verify_first_momentum_gauge_functional_artifact(
        &final_directory,
        gauge_form_degree,
        operator_ordinal,
    )
}

fn build_and_write_first_momentum_gauge_stream_functional_artifact_selected(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    evaluated_parameter_components: Vec<usize>,
    output_root: &Path,
) -> io::Result<FirstMomentumGaugeStreamFunctionalArtifact> {
    if gauge_form_degree > 5 || operator_ordinal >= 56 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "streamed first-momentum functional job requires form degree 0..5 and operator ordinal 0..55",
        ));
    }
    let parameter_components = [1, 11, 55, 165, 330, 462][gauge_form_degree];
    if evaluated_parameter_components.is_empty()
        || evaluated_parameter_components
            .windows(2)
            .any(|window| window[0] >= window[1])
        || evaluated_parameter_components
            .iter()
            .any(|&index| index >= parameter_components)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "evaluated gauge-parameter components must be a nonempty sorted unique subset",
        ));
    }
    let final_directory =
        first_momentum_functional_directory(output_root, gauge_form_degree, operator_ordinal);
    if final_directory.exists() {
        verify_first_momentum_gauge_functional_column(
            &final_directory,
            gauge_form_degree,
            operator_ordinal,
        )?;
        let artifact: FirstMomentumGaugeStreamFunctionalArtifact = serde_json::from_reader(
            BufReader::new(File::open(final_directory.join("manifest.json"))?),
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let recorded_components = if artifact.evaluated_parameter_components.is_empty() {
            (0..artifact.parameter_components).collect::<Vec<_>>()
        } else {
            artifact.evaluated_parameter_components.clone()
        };
        if recorded_components != evaluated_parameter_components {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "existing streamed artifact uses a different parameter projection",
            ));
        }
        return Ok(artifact);
    }
    let incomplete_root = output_root
        .join("functional")
        .join("incomplete")
        .join(format!("form-{gauge_form_degree}"));
    let completed_root = output_root
        .join("functional")
        .join("complete")
        .join(format!("form-{gauge_form_degree}"));
    fs::create_dir_all(&incomplete_root)?;
    fs::create_dir_all(&completed_root)?;
    let started_unix_seconds = unix_seconds();
    let timer = Instant::now();
    let temporary_directory = incomplete_root.join(format!(
        "column-{operator_ordinal:03}-{}-{started_unix_seconds}",
        std::process::id()
    ));
    fs::create_dir(&temporary_directory)?;

    let rows_per_seed = 2 * FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS;
    let mut exact_functional_values =
        vec![0_i128; FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len() * rows_per_seed];
    let (operator, parameter_basis, maximum, fixture_sha256, emitted_nonzero_terms) =
        crate::eleven_dimensional_level16_couplings::visit_first_momentum_gauge_composition_terms(
            gauge_form_degree,
            operator_ordinal,
            Some(&evaluated_parameter_components),
            |entry| {
                let key = u64::from(entry.exterior_mask)
                    | (u64::try_from(entry.momentum_vector_index).unwrap() << 32)
                    | (u64::try_from(entry.parameter_component_index).unwrap() << 36);
                for (seed_index, seed) in FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.iter().enumerate() {
                    let hash = splitmix64_gauge(key ^ seed);
                    let bucket = (hash as usize) % FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS;
                    let base = seed_index * rows_per_seed;
                    if hash >> 63 == 0 {
                        exact_functional_values[base + bucket] = exact_functional_values
                            [base + bucket]
                            .checked_add(entry.real)
                            .expect("i128 overflow in streamed real functional bucket");
                        exact_functional_values
                            [base + FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS + bucket] =
                            exact_functional_values
                                [base + FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS + bucket]
                                .checked_add(entry.imaginary)
                                .expect("i128 overflow in streamed imaginary functional bucket");
                    } else {
                        exact_functional_values[base + bucket] = exact_functional_values
                            [base + bucket]
                            .checked_sub(entry.real)
                            .expect("i128 overflow in streamed real functional bucket");
                        exact_functional_values
                            [base + FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS + bucket] =
                            exact_functional_values
                                [base + FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS + bucket]
                                .checked_sub(entry.imaginary)
                                .expect("i128 overflow in streamed imaginary functional bucket");
                    }
                }
                Ok(())
            },
        )?;
    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let finished_unix_seconds = unix_seconds();
    let functional_image_is_nonzero = exact_functional_values.iter().any(|&value| value != 0);
    let artifact = FirstMomentumGaugeStreamFunctionalArtifact {
        schema_version: "adynkra-11d-first-momentum-gauge-stream-functional-v2".to_string(),
        passed: parameter_basis.len() == [1, 11, 55, 165, 330, 462][gauge_form_degree]
            && operator.ordinal == operator_ordinal
            && exact_functional_values.len()
                == FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len() * rows_per_seed,
        gauge_form_degree,
        parameter_dynkin_label: labels[gauge_form_degree].to_string(),
        parameter_components: parameter_basis.len(),
        evaluated_parameter_components,
        operator,
        exterior_degree: 15,
        emitted_nonzero_terms,
        maximum_absolute_emitted_term_coefficient: maximum.to_string(),
        functional_seeds: FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS
            .iter()
            .map(|seed| format!("{seed:016x}"))
            .collect(),
        functional_buckets_per_real_part: FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS,
        exact_functional_values: exact_functional_values
            .iter()
            .map(ToString::to_string)
            .collect(),
        functional_image_is_nonzero,
        fixture_sha256,
        source_revision: std::env::var("ADINKRA_SOURCE_REVISION")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        executable_sha256: std::env::var("ADINKRA_EXECUTABLE_SHA256")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        host: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        process_id: std::process::id(),
        started_unix_seconds,
        finished_unix_seconds,
        elapsed_milliseconds: timer.elapsed().as_millis(),
        convention: "exact signed-hash linear image of the uncombined term stream for p D^15 Lambda coordinates; linearity makes it identical to hashing the fully accumulated residual; the key packs the 15-derivative exterior mask, momentum-vector index, and gauge-parameter component; four fixed splitmix64 seeds, 64 real and 64 imaginary buckets per seed".to_string(),
        interpretation: "This artifact evaluates one of the 56 operator columns after one candidate source-gauge map and records an exact linear image of the first-momentum residual on the recorded gauge-parameter components without materializing the full residual coordinate map.".to_string(),
        boundary: "A zero leading projection in the merged functional kernel excludes a full source-invariant extension even when only a subset of gauge-parameter components was evaluated. A nonzero projection remains inconclusive. Emitted-term counts and maximum term coefficients are not residual support counts or residual norms.".to_string(),
    };
    write_json_durable(&temporary_directory.join("manifest.json"), &artifact)?;
    File::open(&temporary_directory)?.sync_all()?;
    fs::rename(&temporary_directory, &final_directory)?;
    File::open(&completed_root)?.sync_all()?;
    verify_first_momentum_gauge_functional_column(
        &final_directory,
        gauge_form_degree,
        operator_ordinal,
    )?;
    Ok(artifact)
}

pub fn build_and_write_first_momentum_gauge_stream_functional_artifact(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    output_root: &Path,
) -> io::Result<FirstMomentumGaugeStreamFunctionalArtifact> {
    let parameter_components = [1, 11, 55, 165, 330, 462]
        .get(gauge_form_degree)
        .copied()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "gauge form degree must lie in 0..5",
            )
        })?;
    build_and_write_first_momentum_gauge_stream_functional_artifact_selected(
        gauge_form_degree,
        operator_ordinal,
        (0..parameter_components).collect(),
        output_root,
    )
}

pub fn build_and_write_first_momentum_gauge_stream_prefix_functional_artifact(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    parameter_component_count: usize,
    output_root: &Path,
) -> io::Result<FirstMomentumGaugeStreamFunctionalArtifact> {
    build_and_write_first_momentum_gauge_stream_functional_artifact_selected(
        gauge_form_degree,
        operator_ordinal,
        (0..parameter_component_count).collect(),
        output_root,
    )
}

pub fn merge_first_momentum_gauge_functional_artifacts(
    gauge_form_degree: usize,
    output_root: &Path,
    zero_momentum_root: &Path,
) -> io::Result<FirstMomentumGaugeFunctionalMergeReport> {
    if gauge_form_degree > 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "gauge form degree must lie in 0..5",
        ));
    }
    let specs = crate::eleven_dimensional_level16_couplings::joint_column_specs();
    let mut functional_columns = Vec::with_capacity(56);
    let mut parameter_component_counts = Vec::with_capacity(56);
    let mut evaluated_parameter_component_sets = Vec::with_capacity(56);
    let mut source_artifact_sha256 = Vec::with_capacity(56);
    for spec in &specs {
        let directory =
            first_momentum_functional_directory(output_root, gauge_form_degree, spec.ordinal);
        let path = directory.join("manifest.json");
        let (parameter_components, evaluated_parameter_components, exact_functional_values) =
            verify_first_momentum_gauge_functional_column(
                &directory,
                gauge_form_degree,
                spec.ordinal,
            )?;
        parameter_component_counts.push(parameter_components);
        evaluated_parameter_component_sets.push(evaluated_parameter_components);
        functional_columns.push(
            exact_functional_values
                .iter()
                .map(|value| {
                    value.parse::<BigInt>().map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid functional integer: {error}"),
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()?,
        );
        source_artifact_sha256.push(sha256_file(&path)?);
    }
    let parameter_components = parameter_component_counts[0];
    if parameter_component_counts
        .iter()
        .any(|&count| count != parameter_components)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum functional artifacts use different parameter dimensions",
        ));
    }
    let evaluated_parameter_components = evaluated_parameter_component_sets[0].clone();
    if evaluated_parameter_component_sets
        .iter()
        .any(|components| *components != evaluated_parameter_components)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum functional artifacts use different parameter projections",
        ));
    }
    let functional_rows = functional_columns[0].len();
    if functional_columns
        .iter()
        .any(|column| column.len() != functional_rows)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum functional columns have different lengths",
        ));
    }

    let zero_path = zero_momentum_root
        .join("merge")
        .join(format!("zero-momentum-form-{gauge_form_degree}.json"));
    let zero_payload = fs::read(&zero_path)?;
    let zero_report: ZeroMomentumGaugeKernelReport = serde_json::from_slice(&zero_payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !zero_report.passed
        || zero_report.gauge_form_degree != gauge_form_degree
        || zero_report.leading_basis
            != specs[..12]
                .iter()
                .map(|spec| spec.label.clone())
                .collect::<Vec<_>>()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zero-momentum kernel report does not match the first-momentum operator basis",
        ));
    }
    let zero_kernel = zero_report
        .primitive_integer_kernel_basis
        .iter()
        .map(|vector| {
            vector
                .iter()
                .map(|value| {
                    value.parse::<BigInt>().map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid zero-momentum kernel integer: {error}"),
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()
        })
        .collect::<io::Result<Vec<_>>>()?;

    let mut parameterized_columns = Vec::<Vec<BigInt>>::new();
    for kernel_vector in &zero_kernel {
        let mut column = vec![BigInt::from(0); functional_rows];
        for (coefficient, source_column) in kernel_vector.iter().zip(&functional_columns[..12]) {
            for (destination, source) in column.iter_mut().zip(source_column) {
                *destination += coefficient * source;
            }
        }
        parameterized_columns.push(column);
    }
    parameterized_columns.extend(functional_columns[12..].iter().cloned());
    let matrix = (0..functional_rows)
        .map(|row| {
            parameterized_columns
                .iter()
                .map(|column| Ratio::from_integer(column[row].clone()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let exact_functional_rank =
        crate::eleven_dimensional_level16_couplings::rational_matrix_rank(&matrix);
    let rational_kernel = crate::eleven_dimensional_level16_couplings::rational_nullspace(&matrix);
    let primitive_kernel = rational_kernel
        .iter()
        .map(|vector| crate::eleven_dimensional_level16_couplings::primitive_bigint_vector(vector))
        .collect::<Vec<_>>();
    let functional_kernel_residuals_exactly_zero = rational_kernel
        .iter()
        .all(|vector| matrix.iter().all(|row| rational_dot(row, vector).is_zero()));
    let leading_projections = rational_kernel
        .iter()
        .map(|vector| vector[..zero_kernel.len()].to_vec())
        .filter(|projection| projection.iter().any(|value| !value.is_zero()))
        .collect::<Vec<_>>();
    let functional_kernel_leading_projection_rank =
        crate::eleven_dimensional_level16_couplings::rational_matrix_rank(&leading_projections);
    let nonzero_leading_extension_excluded_by_functionals =
        functional_kernel_leading_projection_rank == 0;
    let exact_functional_nullity = parameterized_columns.len() - exact_functional_rank;
    let passed = functional_columns.len() == 56
        && zero_kernel.len() == zero_report.exact_nullity
        && parameterized_columns.len() == zero_kernel.len() + 44
        && exact_functional_rank + exact_functional_nullity == parameterized_columns.len()
        && functional_kernel_residuals_exactly_zero;
    let report = FirstMomentumGaugeFunctionalMergeReport {
        schema_version: "adynkra-11d-first-momentum-gauge-functional-merge-v1".to_string(),
        passed,
        gauge_form_degree,
        parameter_dynkin_label: zero_report.parameter_dynkin_label,
        parameter_components,
        parameter_projection_is_complete: evaluated_parameter_components.len()
            == parameter_components,
        evaluated_parameter_components,
        leading_basis: specs[..12]
            .iter()
            .map(|spec| spec.label.clone())
            .collect(),
        first_momentum_basis: specs[12..]
            .iter()
            .map(|spec| spec.label.clone())
            .collect(),
        zero_momentum_kernel_dimension: zero_kernel.len(),
        zero_momentum_kernel_basis: zero_report.primitive_integer_kernel_basis,
        parameterized_columns: parameterized_columns.len(),
        functional_rows,
        exact_functional_rank,
        exact_functional_nullity,
        functional_kernel_leading_projection_rank,
        nonzero_leading_extension_excluded_by_functionals,
        functional_primitive_integer_kernel_basis: primitive_kernel
            .iter()
            .map(|vector| vector.iter().map(ToString::to_string).collect())
            .collect(),
        functional_kernel_residuals_exactly_zero,
        source_artifact_sha256,
        zero_momentum_kernel_report_sha256: sha256_file(&zero_path)?,
        interpretation: if nonzero_leading_extension_excluded_by_functionals {
            "The exact functional kernel has zero projection onto the zero-momentum leading kernel. Because the full residual kernel is contained in every exact functional kernel, no nonzero leading operator extends through first momentum for this source channel.".to_string()
        } else {
            "The exact functional kernel has a nonzero leading projection. This screen is inconclusive; additional exact functionals or the full residual are required to determine whether a leading operator extends.".to_string()
        },
        boundary: "This is an exact linear-image screen of the first-momentum source variation on the recorded gauge-parameter components. It can exclude a leading extension, but it cannot certify one unless every parameter component and the full residual are checked.".to_string(),
    };
    let merge_root = output_root.join("functional").join("merge");
    fs::create_dir_all(&merge_root)?;
    let output = merge_root.join(format!(
        "first-momentum-functional-form-{gauge_form_degree}.json"
    ));
    let temporary = merge_root.join(format!(
        ".first-momentum-functional-form-{gauge_form_degree}.json.{}.tmp",
        std::process::id()
    ));
    write_json_durable(&temporary, &report)?;
    fs::rename(temporary, output)?;
    File::open(&merge_root)?.sync_all()?;
    Ok(report)
}

pub fn gauge_composition_specs() -> Vec<GaugeCompositionSpec> {
    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let operators = crate::eleven_dimensional_level16_couplings::joint_column_specs();
    let mut specs = Vec::with_capacity(labels.len() * operators.len());
    for (degree, parameter_dynkin_label) in labels.into_iter().enumerate() {
        for operator in &operators {
            specs.push(GaugeCompositionSpec {
                ordinal: degree * operators.len() + operator.ordinal,
                gauge_form_degree: degree,
                parameter_dynkin_label,
                operator_ordinal: operator.ordinal,
                operator_label: operator.label.clone(),
                operator_kind: operator.kind.clone(),
                contributes_zero_momentum_d17: operator.kind == "leading",
                contributes_first_momentum_pd15: true,
            });
        }
    }
    specs
}

pub fn verify() -> ElevenDimensionalGaugeIntertwinerReport {
    let labels = ["00000", "10000", "01000", "00100", "00010", "00002"];
    let expected_symmetries = [
        "antisymmetric",
        "symmetric",
        "symmetric",
        "antisymmetric",
        "antisymmetric",
        "symmetric",
    ];
    let parameter_dimensions = [1, 11, 55, 165, 330, 462];
    let zero_momentum_kernel_degrees = vec![1, 2, 5];
    let generic_momentum_kernel_degrees = vec![2, 5];

    let basis = crate::eleven_dimensional_clifford::gauge_form_bilinear_basis();
    let operator_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis();
    let degree_zero_operator_is_identity =
        operator_basis[0].2.iter().enumerate().all(|(row, values)| {
            values
                .iter()
                .enumerate()
                .all(|(column, value)| *value == if row == column { g(1, 0) } else { g(0, 0) })
        });
    let sparse = basis
        .iter()
        .map(|(_, _, matrix)| sparse_matrix(matrix))
        .collect::<Vec<_>>();

    let mut diagonal_norm_residuals = 0;
    let mut off_diagonal_orthogonality_residuals = 0;
    let mut pairwise_inner_products_checked = 0;
    for left in 0..sparse.len() {
        for right in left..sparse.len() {
            let value = hermitian_dot(&sparse[left], &sparse[right]);
            pairwise_inner_products_checked += 1;
            if left == right {
                diagonal_norm_residuals += usize::from(value != g(32, 0));
            } else {
                off_diagonal_orthogonality_residuals += usize::from(value != g(0, 0));
            }
        }
    }
    let exact_bilinear_basis_rank =
        if diagonal_norm_residuals == 0 && off_diagonal_orthogonality_residuals == 0 {
            basis.len()
        } else {
            0
        };
    let composition_specs = gauge_composition_specs();
    let leading_operator_columns = composition_specs
        .iter()
        .filter(|spec| spec.gauge_form_degree == 0 && spec.operator_kind == "leading")
        .count();
    let first_momentum_operator_columns = composition_specs
        .iter()
        .filter(|spec| spec.gauge_form_degree == 0 && spec.operator_kind == "first-momentum")
        .count();
    let zero_momentum_composition_jobs = composition_specs
        .iter()
        .filter(|spec| spec.contributes_zero_momentum_d17)
        .count();
    let first_momentum_composition_jobs = composition_specs
        .iter()
        .filter(|spec| spec.contributes_first_momentum_pd15)
        .count();
    let total_composition_jobs = composition_specs.len();

    let mut parameter_channels = Vec::new();
    for degree in 0..=5 {
        let matrices = basis
            .iter()
            .filter(|(candidate_degree, _, _)| *candidate_degree == degree)
            .map(|(_, _, matrix)| matrix)
            .collect::<Vec<_>>();
        let symmetry_sign = if expected_symmetries[degree] == "symmetric" {
            1
        } else {
            -1
        };
        let transpose_residual_entries = matrices
            .iter()
            .map(|matrix| transpose_mismatches(matrix, symmetry_sign))
            .sum();
        let (monomial_row_residuals, monomial_column_residuals, unit_coefficient_residuals) =
            matrices.iter().fold((0, 0, 0), |totals, matrix| {
                let residuals = monomial_matrix_residuals(matrix);
                (
                    totals.0 + residuals.0,
                    totals.1 + residuals.1,
                    totals.2 + residuals.2,
                )
            });
        let nonzero_matrix_entries = matrices
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .flatten()
                    .filter(|value| **value != g(0, 0))
                    .count()
            })
            .sum();
        let scalar_divergence_zero_at_zero_momentum =
            zero_momentum_kernel_degrees.contains(&degree);
        let scalar_divergence_zero_at_generic_momentum =
            generic_momentum_kernel_degrees.contains(&degree);
        let passed = matrices.len() == parameter_dimensions[degree]
            && nonzero_matrix_entries == matrices.len() * SPINOR_DIMENSION
            && transpose_residual_entries == 0
            && monomial_row_residuals == 0
            && monomial_column_residuals == 0
            && unit_coefficient_residuals == 0;
        parameter_channels.push(GaugeIntertwinerChannelReport {
            form_degree: degree,
            dynkin_label: labels[degree],
            parameter_dimension: parameter_dimensions[degree],
            derivative_spinor_dimension: SPINOR_DIMENSION,
            intertwiner_domain_dimension: parameter_dimensions[degree] * SPINOR_DIMENSION,
            component_matrices: matrices.len(),
            nonzero_matrix_entries,
            component_matrix_rank: SPINOR_DIMENSION,
            expected_spinor_index_symmetry: expected_symmetries[degree],
            transpose_residual_entries,
            monomial_row_residuals,
            monomial_column_residuals,
            unit_coefficient_residuals,
            scalar_divergence_zero_at_zero_momentum,
            scalar_divergence_zero_at_generic_momentum,
            physical_status: "complete Lorentz-compatible candidate; not selected as a physical gauge symmetry",
            passed,
        });
    }

    let channel_count = parameter_channels.len();
    let total_parameter_components = parameter_channels
        .iter()
        .map(|channel| channel.parameter_dimension)
        .sum();
    let total_bilinear_matrices = parameter_channels
        .iter()
        .map(|channel| channel.component_matrices)
        .sum();
    let passed = channel_count == 6
        && total_parameter_components == SPINOR_DIMENSION * SPINOR_DIMENSION
        && total_bilinear_matrices == SPINOR_DIMENSION * SPINOR_DIMENSION
        && pairwise_inner_products_checked
            == total_bilinear_matrices * (total_bilinear_matrices + 1) / 2
        && diagonal_norm_residuals == 0
        && off_diagonal_orthogonality_residuals == 0
        && exact_bilinear_basis_rank == SPINOR_DIMENSION * SPINOR_DIMENSION
        && operator_basis.len() == SPINOR_DIMENSION * SPINOR_DIMENSION
        && degree_zero_operator_is_identity
        && leading_operator_columns == 12
        && first_momentum_operator_columns == 44
        && zero_momentum_composition_jobs == 72
        && first_momentum_composition_jobs == 336
        && total_composition_jobs == 336
        && parameter_channels.iter().all(|channel| channel.passed);

    ElevenDimensionalGaugeIntertwinerReport {
        schema_version: "adynkra-11d-spinor-gauge-intertwiners-v1",
        role: "exact census and construction of the six Lorentz-compatible first-derivative maps into the proposed spinor prepotential",
        proposed_prepotential: "unconstrained spinor superfield Psi_alpha of arXiv:2002.08502, Eq. (6.3)",
        candidate_transformation: "delta Psi_alpha = sum_{p=0}^5 c_p (C Gamma^[a1...ap])_{alpha beta} D^beta Lambda_[a1...ap]",
        executed_mixed_index_transformation: "after lowering the derivative index with C^{-1}, delta Psi_alpha = sum_{p=0}^5 c_p Gamma^[a1...ap]_alpha^gamma D_gamma Lambda_[a1...ap]",
        degree_zero_operator_is_identity,
        parameter_channels,
        channel_count,
        total_parameter_components,
        total_bilinear_matrices,
        spinor_square_dimension: SPINOR_DIMENSION * SPINOR_DIMENSION,
        pairwise_inner_products_checked,
        diagonal_norm_residuals,
        off_diagonal_orthogonality_residuals,
        exact_bilinear_basis_rank,
        leading_operator_columns,
        first_momentum_operator_columns,
        zero_momentum_composition_jobs,
        first_momentum_composition_jobs,
        total_composition_jobs,
        zero_momentum_scalar_divergence_kernel_degrees: zero_momentum_kernel_degrees,
        generic_momentum_scalar_divergence_kernel_degrees: generic_momentum_kernel_degrees,
        source_selects_channel_coefficients: false,
        source_prints_target_gauge_transformation: false,
        gauge_for_gauge_reducibility_supplied: false,
        source_quotient_condition: "a target operator A is defined on Psi modulo the candidate source gauge image only if A composed with G is zero",
        target_quotient_condition: "allowing A composed with G to equal a target gauge map requires that target map as additional input; it is not determined by the six source intertwiners",
        next_exact_step: "after the separately recorded source-invariance screens, either supply an independently derived target gauge law for A composed with G_p equals K_p or retain the hook and compute its next Bianchi map",
        boundary: "the six intertwiners are a complete Lorentz-compatible ansatz. The cited sources do not establish that any chosen linear combination is a physical gauge symmetry, provide its gauge-for-gauge complex, or justify quotienting the hook residual by its image",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_candidate_gauge_intertwiners_form_the_complete_spinor_square() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.channel_count, 6);
        assert_eq!(report.total_parameter_components, 1_024);
        assert_eq!(report.total_bilinear_matrices, 1_024);
        assert_eq!(report.exact_bilinear_basis_rank, 1_024);
        assert_eq!(report.diagonal_norm_residuals, 0);
        assert_eq!(report.off_diagonal_orthogonality_residuals, 0);
        assert_eq!(report.leading_operator_columns, 12);
        assert_eq!(report.first_momentum_operator_columns, 44);
        assert_eq!(report.zero_momentum_composition_jobs, 72);
        assert_eq!(report.first_momentum_composition_jobs, 336);
        assert_eq!(report.total_composition_jobs, 336);
    }

    #[test]
    fn scalar_divergence_kernel_is_reported_at_both_momentum_regimes() {
        let report = verify();
        assert_eq!(
            report.zero_momentum_scalar_divergence_kernel_degrees,
            vec![1, 2, 5]
        );
        assert_eq!(
            report.generic_momentum_scalar_divergence_kernel_degrees,
            vec![2, 5]
        );
        assert!(!report.source_selects_channel_coefficients);
        assert!(!report.source_prints_target_gauge_transformation);
    }

    #[test]
    fn exact_constraint_kernel_handles_empty_and_rectangular_systems() {
        let (empty_rank, empty_kernel) = exact_constraint_kernel(&[], 3);
        assert_eq!(empty_rank, 0);
        assert_eq!(empty_kernel, standard_basis(3));

        let zero = Ratio::from_integer(BigInt::from(0));
        let one = Ratio::from_integer(BigInt::from(1));
        let rows = vec![
            vec![one.clone(), zero.clone(), zero.clone()],
            vec![zero.clone(), one.clone(), zero.clone()],
        ];
        let (rank, kernel) = exact_constraint_kernel(&rows, 3);
        assert_eq!(rank, 2);
        assert_eq!(kernel, vec![vec![zero.clone(), zero, one]]);
    }

    #[test]
    fn streamed_functional_artifact_requires_a_valid_component_subset() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "adinkra-gauge-functional-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("manifest.json");
        let mut artifact = FirstMomentumGaugeStreamFunctionalArtifact {
            schema_version: "adynkra-11d-first-momentum-gauge-stream-functional-v2".to_string(),
            passed: true,
            gauge_form_degree: 1,
            parameter_dynkin_label: "10000".to_string(),
            parameter_components: 11,
            evaluated_parameter_components: vec![0, 3],
            operator: crate::eleven_dimensional_level16_couplings::joint_column_specs()[0].clone(),
            exterior_degree: 15,
            emitted_nonzero_terms: 0,
            maximum_absolute_emitted_term_coefficient: "0".to_string(),
            functional_seeds: FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS
                .iter()
                .map(|seed| format!("{seed:016x}"))
                .collect(),
            functional_buckets_per_real_part: FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS,
            exact_functional_values: vec![
                "0".to_string();
                FIRST_MOMENTUM_GAUGE_FUNCTIONAL_SEEDS.len()
                    * 2
                    * FIRST_MOMENTUM_GAUGE_FUNCTIONAL_BUCKETS
            ],
            functional_image_is_nonzero: false,
            fixture_sha256: "fixture".to_string(),
            source_revision: "test".to_string(),
            executable_sha256: "test".to_string(),
            host: "test".to_string(),
            process_id: std::process::id(),
            started_unix_seconds: 0,
            finished_unix_seconds: 0,
            elapsed_milliseconds: 0,
            convention: "test".to_string(),
            interpretation: "test".to_string(),
            boundary: "test".to_string(),
        };
        write_json_durable(&path, &artifact).unwrap();
        let (_, components, values) =
            verify_first_momentum_gauge_functional_column(&directory, 1, 0).unwrap();
        assert_eq!(components, vec![0, 3]);
        assert_eq!(values.len(), 512);

        artifact.evaluated_parameter_components = vec![3, 3];
        write_json_durable(&path, &artifact).unwrap();
        assert!(verify_first_momentum_gauge_functional_column(&directory, 1, 0).is_err());

        artifact.evaluated_parameter_components = vec![11];
        write_json_durable(&path, &artifact).unwrap();
        assert!(verify_first_momentum_gauge_functional_column(&directory, 1, 0).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
