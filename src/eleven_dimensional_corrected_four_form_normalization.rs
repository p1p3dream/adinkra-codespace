//! Exact descendant comparison for the source-variance-corrected Lambda3 ray.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use num_rational::Ratio;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_corrected_full_chain_oracle::{
    CorrectedGamma25ParityReport, corrected_full_chain_streams, verify_corrected_gamma25_parity,
};
use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, cached_linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    FormalMomentumMonomial, OrderedSuperderivativeMonomial,
};
use crate::eleven_dimensional_target_equation_complex::{TargetSector, target_sector_complex};

const H_HAT_DIMENSION: usize = 320;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct DescendantRowKey {
    output_coordinate: usize,
    monomial: OrderedSuperderivativeMonomial,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CorrectedDescendantMismatch {
    pub source_ordinal: usize,
    pub output_coordinate: usize,
    pub exterior_spinor_mask: u32,
    pub momentum_exponents: [u16; 11],
    pub candidate: String,
    pub teleparallel: String,
    pub residual: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CorrectedColumnComparison {
    pub source_ordinal: usize,
    pub candidate_nonzero_rows: usize,
    pub teleparallel_nonzero_rows: usize,
    pub common_support_rows: usize,
    pub candidate_only_support_rows: usize,
    pub teleparallel_only_support_rows: usize,
    pub universal_scale: Option<String>,
    pub exact_residual_rows: usize,
    pub first_exact_mismatch: Option<CorrectedDescendantMismatch>,
    pub candidate_stream_sha256: String,
    pub teleparallel_stream_sha256: String,
    pub residual_stream_sha256: String,
    pub passed: bool,
}

fn hash_exact_row(hasher: &mut Sha256, key: &DescendantRowKey, value: &ExactQi) {
    hasher.update((key.output_coordinate as u64).to_le_bytes());
    hasher.update(key.monomial.exterior_spinor_mask.to_le_bytes());
    for exponent in key.monomial.momentum.exponents {
        hasher.update(exponent.to_le_bytes());
    }
    hasher.update(value.real.numer().to_le_bytes());
    hasher.update(value.real.denom().to_le_bytes());
    hasher.update(value.imaginary.numer().to_le_bytes());
    hasher.update(value.imaginary.denom().to_le_bytes());
}

fn multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn divide(left: &ExactQi, right: &ExactQi) -> Option<ExactQi> {
    if right.is_zero() {
        return None;
    }
    let denominator =
        right.real.clone() * right.real.clone() + right.imaginary.clone() * right.imaginary.clone();
    Some(ExactQi {
        real: (left.real.clone() * right.real.clone()
            + left.imaginary.clone() * right.imaginary.clone())
            / denominator.clone(),
        imaginary: (left.imaginary.clone() * right.real.clone()
            - left.real.clone() * right.imaginary.clone())
            / denominator,
    })
}

fn qi_string(value: &ExactQi) -> String {
    format!("({})+({})i", value.real, value.imaginary)
}

fn basis_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"canonical-gamma-traceless-Hhat-basis-v1");
    for (ordinal, column) in canonical_gamma_traceless_frame_basis().iter().enumerate() {
        hasher.update((ordinal as u64).to_le_bytes());
        for (&coordinate, value) in column {
            hasher.update((coordinate as u64).to_le_bytes());
            for rational in [&value.real, &value.imaginary] {
                hasher.update(rational.numer().to_le_bytes());
                hasher.update(rational.denom().to_le_bytes());
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

fn teleparallel_map_sha256() -> String {
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    let mut hasher = Sha256::new();
    hasher.update(b"teleparallel-curl-to-D-F4-v1");
    hasher.update((operator.input_dimension as u64).to_le_bytes());
    hasher.update((operator.output_dimension as u64).to_le_bytes());
    for (column, entries) in operator.columns.iter().enumerate() {
        for entry in entries {
            hasher.update((column as u64).to_le_bytes());
            hasher.update((entry.row as u64).to_le_bytes());
            for rational in [&entry.coefficient.real, &entry.coefficient.imaginary] {
                hasher.update(rational.numer().to_le_bytes());
                hasher.update(rational.denom().to_le_bytes());
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

fn target_curvature_sha256() -> String {
    let curvature = &target_sector_complex(TargetSector::FourForm).curvature;
    let mut hasher = Sha256::new();
    hasher.update(b"target-four-form-curvature-v1");
    hasher.update((curvature.rows() as u64).to_le_bytes());
    hasher.update((curvature.columns() as u64).to_le_bytes());
    for column in 0..curvature.columns() {
        for (row, term) in curvature.column_terms(column) {
            hasher.update((column as u64).to_le_bytes());
            hasher.update((row as u64).to_le_bytes());
            hasher.update(term.monomial.exponents);
            hasher.update(term.real_numerator.to_le_bytes());
            hasher.update(term.real_denominator.to_le_bytes());
            hasher.update(term.imaginary_numerator.to_le_bytes());
            hasher.update(term.imaginary_denominator.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn executable_sha256() -> Result<String, String> {
    let path = std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
    let bytes =
        fs::read(&path).map_err(|error| format!("read executable {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn corrected_streams(
    source_ordinal: usize,
) -> Result<
    (
        BTreeMap<DescendantRowKey, ExactQi>,
        BTreeMap<DescendantRowKey, ExactQi>,
    ),
    String,
> {
    let (candidate, teleparallel) = corrected_full_chain_streams(source_ordinal)?;
    let convert = |input: BTreeMap<
        crate::eleven_dimensional_corrected_full_chain_oracle::FullChainRowKey,
        ExactQi,
    >| {
        input
            .into_iter()
            .map(|(key, value)| {
                (
                    DescendantRowKey {
                        output_coordinate: key.output_coordinate,
                        monomial: OrderedSuperderivativeMonomial {
                            exterior_spinor_mask: key.exterior_spinor_mask,
                            momentum: FormalMomentumMonomial {
                                exponents: key.momentum_exponents,
                            },
                        },
                    },
                    value,
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    Ok((convert(candidate), convert(teleparallel)))
}

fn forced_scale(source_ordinal: usize) -> Result<ExactQi, String> {
    let (candidate, teleparallel) = corrected_streams(source_ordinal)?;
    let (key, candidate_value) = candidate.iter().next().ok_or_else(|| {
        format!("corrected candidate column {source_ordinal} is identically zero")
    })?;
    let target_value = teleparallel.get(key).cloned().unwrap_or_else(ExactQi::zero);
    divide(&target_value, candidate_value)
        .ok_or_else(|| "corrected candidate pivot denominator is zero".to_string())
}

fn compare_corrected_column_with_scale(
    source_ordinal: usize,
    scale: &ExactQi,
) -> Result<CorrectedColumnComparison, String> {
    let (candidate, teleparallel) = corrected_streams(source_ordinal)?;
    let mut candidate_hasher = Sha256::new();
    candidate_hasher.update(b"right-C-full-chain-candidate-v1");
    candidate_hasher.update((source_ordinal as u64).to_le_bytes());
    for (key, value) in &candidate {
        hash_exact_row(&mut candidate_hasher, key, value);
    }
    let mut teleparallel_hasher = Sha256::new();
    teleparallel_hasher.update(b"right-C-full-chain-teleparallel-v1");
    teleparallel_hasher.update((source_ordinal as u64).to_le_bytes());
    for (key, value) in &teleparallel {
        hash_exact_row(&mut teleparallel_hasher, key, value);
    }
    let mut residual_hasher = Sha256::new();
    residual_hasher.update(b"right-C-full-chain-residual-v1");
    residual_hasher.update((source_ordinal as u64).to_le_bytes());
    // The lexicographically first nonzero candidate row fixes the only
    // possible universal scale. If the target is zero there, the scale is
    // forced to zero and any nonzero target row is then decisive.
    let keys = candidate
        .keys()
        .chain(teleparallel.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut common = 0;
    let mut candidate_only = 0;
    let mut teleparallel_only = 0;
    let mut residuals = 0;
    let mut first = None;
    for key in &keys {
        let c = candidate.get(key).cloned().unwrap_or_else(ExactQi::zero);
        let t = teleparallel.get(key).cloned().unwrap_or_else(ExactQi::zero);
        match (candidate.contains_key(key), teleparallel.contains_key(key)) {
            (true, true) => common += 1,
            (true, false) => candidate_only += 1,
            (false, true) => teleparallel_only += 1,
            _ => unreachable!(),
        }
        let mut residual = multiply(scale, &c);
        residual.add_assign(&t.scaled(&Ratio::from_integer(-1)));
        if !residual.is_zero() {
            residuals += 1;
            hash_exact_row(&mut residual_hasher, key, &residual);
            first.get_or_insert_with(|| CorrectedDescendantMismatch {
                source_ordinal,
                output_coordinate: key.output_coordinate,
                exterior_spinor_mask: key.monomial.exterior_spinor_mask,
                momentum_exponents: key.monomial.momentum.exponents,
                candidate: qi_string(&c),
                teleparallel: qi_string(&t),
                residual: qi_string(&residual),
            });
        }
    }
    Ok(CorrectedColumnComparison {
        source_ordinal,
        candidate_nonzero_rows: candidate.len(),
        teleparallel_nonzero_rows: teleparallel.len(),
        common_support_rows: common,
        candidate_only_support_rows: candidate_only,
        teleparallel_only_support_rows: teleparallel_only,
        universal_scale: Some(qi_string(scale)),
        exact_residual_rows: residuals,
        first_exact_mismatch: first,
        candidate_stream_sha256: format!("{:x}", candidate_hasher.finalize()),
        teleparallel_stream_sha256: format!("{:x}", teleparallel_hasher.finalize()),
        residual_stream_sha256: format!("{:x}", residual_hasher.finalize()),
        passed: residuals == 0,
    })
}

pub fn compare_corrected_column(
    source_ordinal: usize,
) -> Result<CorrectedColumnComparison, String> {
    let scale = forced_scale(0)?;
    compare_corrected_column_with_scale(source_ordinal, &scale)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedFullComparison {
    pub schema_version: &'static str,
    pub h_hat_columns_checked: usize,
    pub parallelism_width: usize,
    pub run_manifest_sha256: Option<String>,
    pub columns_resumed: usize,
    pub columns_computed: usize,
    pub elapsed_milliseconds: u64,
    pub gamma25_parity: CorrectedGamma25ParityReport,
    pub universal_scale: String,
    pub candidate_nonzero_rows: u64,
    pub teleparallel_nonzero_rows: u64,
    pub common_support_rows: u64,
    pub candidate_only_support_rows: u64,
    pub teleparallel_only_support_rows: u64,
    pub exact_residual_rows: u64,
    pub columns_with_exact_agreement: usize,
    pub first_exact_mismatch: Option<CorrectedDescendantMismatch>,
    pub candidate_stream_sha256: String,
    pub teleparallel_stream_sha256: String,
    pub residual_stream_sha256: String,
    pub every_row_verified_exactly: bool,
    pub decisive_nonproportionality_witness: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn compare_corrected_all(width: usize) -> Result<CorrectedFullComparison, String> {
    if width == 0 || width > 8 {
        return Err("parallelism width must be in 1..=8".into());
    }
    let scale = forced_scale(0)?;
    let parity = verify_corrected_gamma25_parity();
    if !parity.passed {
        return Err("right-C Gamma2/Gamma5 parity gate failed".into());
    }
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .map_err(|error| format!("build corrected descendant pool: {error}"))?;
    let columns = pool
        .install(|| {
            (0..H_HAT_DIMENSION)
                .into_par_iter()
                .map(|ordinal| compare_corrected_column_with_scale(ordinal, &scale))
                .collect::<Vec<_>>()
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    build_full_report(
        width,
        &scale,
        &columns,
        None,
        0,
        H_HAT_DIMENSION,
        started.elapsed(),
        &parity,
    )
}

fn build_full_report(
    width: usize,
    scale: &ExactQi,
    columns: &[CorrectedColumnComparison],
    manifest_sha256: Option<String>,
    resumed: usize,
    computed: usize,
    elapsed: std::time::Duration,
    parity: &CorrectedGamma25ParityReport,
) -> Result<CorrectedFullComparison, String> {
    if columns.len() != H_HAT_DIMENSION
        || columns
            .iter()
            .map(|x| x.source_ordinal)
            .ne(0..H_HAT_DIMENSION)
    {
        return Err("corrected descendant column merge is incomplete or out of order".into());
    }
    let candidate_nonzero_rows = columns
        .iter()
        .map(|x| x.candidate_nonzero_rows as u64)
        .sum();
    let teleparallel_nonzero_rows = columns
        .iter()
        .map(|x| x.teleparallel_nonzero_rows as u64)
        .sum();
    let common_support_rows = columns.iter().map(|x| x.common_support_rows as u64).sum();
    let candidate_only_support_rows = columns
        .iter()
        .map(|x| x.candidate_only_support_rows as u64)
        .sum();
    let teleparallel_only_support_rows = columns
        .iter()
        .map(|x| x.teleparallel_only_support_rows as u64)
        .sum();
    let exact_residual_rows = columns.iter().map(|x| x.exact_residual_rows as u64).sum();
    let columns_with_exact_agreement = columns.iter().filter(|x| x.passed).count();
    let first_exact_mismatch = columns.iter().find_map(|x| x.first_exact_mismatch.clone());
    let aggregate = |domain: &[u8], select: fn(&CorrectedColumnComparison) -> &str| {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        for column in columns {
            hasher.update((column.source_ordinal as u64).to_le_bytes());
            hasher.update(select(column).as_bytes());
        }
        format!("{:x}", hasher.finalize())
    };
    Ok(CorrectedFullComparison {
        schema_version: "adynkra-11d-right-c-full-chain-four-form-normalization-v1",
        h_hat_columns_checked: H_HAT_DIMENSION,
        parallelism_width: width,
        run_manifest_sha256: manifest_sha256,
        columns_resumed: resumed,
        columns_computed: computed,
        elapsed_milliseconds: elapsed.as_millis().min(u128::from(u64::MAX)) as u64,
        gamma25_parity: parity.clone(),
        universal_scale: qi_string(&scale),
        candidate_nonzero_rows,
        teleparallel_nonzero_rows,
        common_support_rows,
        candidate_only_support_rows,
        teleparallel_only_support_rows,
        exact_residual_rows,
        columns_with_exact_agreement,
        first_exact_mismatch,
        candidate_stream_sha256: aggregate(b"right-C-full-chain-candidate-columns-v1", |x| {
            &x.candidate_stream_sha256
        }),
        teleparallel_stream_sha256: aggregate(b"right-C-full-chain-teleparallel-columns-v1", |x| {
            &x.teleparallel_stream_sha256
        }),
        residual_stream_sha256: aggregate(b"right-C-full-chain-residual-columns-v1", |x| {
            &x.residual_stream_sha256
        }),
        every_row_verified_exactly: exact_residual_rows == 0,
        decisive_nonproportionality_witness: exact_residual_rows > 0,
        passed: exact_residual_rows == 0,
        boundary: "Failure rules out proportionality after composing the H slot with right C consistently through the full Eq. 40 compensator, Eq. 25 gravitino, curl, and teleparallel D F4 chains on the unrestricted canonical H_hat source. It does not rule out equality after additional source constraints and does not establish physical four-form normalization, bidegree exhaustion, target-gauge descent, or irreducibility.",
    })
}

fn atomic_json<T: Serialize>(path: &Path, value: &T, must_be_new: bool) -> Result<(), String> {
    if must_be_new && path.exists() {
        return Err(format!("immutable path already exists: {}", path.display()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let file = File::create(&temporary).map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|e| e.to_string())?;
    writer.write_all(b"\n").map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    writer.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(writer);
    fs::rename(temporary, path).map_err(|e| e.to_string())
}

fn json_hash<T: Serialize>(value: &T) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).map_err(|e| e.to_string())?)
    ))
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|x| x.as_millis())
        .unwrap_or_default()
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointPayload {
    schema_version: String,
    manifest_sha256: String,
    column: CorrectedColumnComparison,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointEnvelope {
    payload_sha256: String,
    payload: CheckpointPayload,
}

fn checkpoint_envelope(
    manifest_sha256: &str,
    column: CorrectedColumnComparison,
) -> Result<CheckpointEnvelope, String> {
    let payload = CheckpointPayload {
        schema_version: "adynkra-11d-right-c-full-chain-column-checkpoint-v1".to_string(),
        manifest_sha256: manifest_sha256.to_string(),
        column,
    };
    Ok(CheckpointEnvelope {
        payload_sha256: json_hash(&payload)?,
        payload,
    })
}

fn read_checkpoint(
    path: &Path,
    manifest_sha256: &str,
    ordinal: usize,
    scale: &str,
) -> Result<CorrectedColumnComparison, String> {
    let envelope: CheckpointEnvelope = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("read checkpoint: {error}"))?,
    )
    .map_err(|error| format!("parse checkpoint {}: {error}", path.display()))?;
    if envelope.payload.schema_version != "adynkra-11d-right-c-full-chain-column-checkpoint-v1"
        || envelope.payload.manifest_sha256 != manifest_sha256
        || envelope.payload.column.source_ordinal != ordinal
        || envelope.payload.column.universal_scale.as_deref() != Some(scale)
        || json_hash(&envelope.payload)? != envelope.payload_sha256
    {
        return Err(format!(
            "checkpoint {ordinal} failed schema, manifest, ordinal, scale, or payload-hash validation"
        ));
    }
    Ok(envelope.payload.column)
}

/// Observable report-last full scan with immutable per-column checkpoints.
pub fn write_observed_artifact(
    path: &Path,
    width: usize,
) -> Result<CorrectedFullComparison, String> {
    if width == 0 || width > 8 {
        return Err("parallelism width must be in 1..=8".into());
    }
    let started = Instant::now();
    let scale = forced_scale(0)?;
    let parity = verify_corrected_gamma25_parity();
    if !parity.passed {
        return Err("right-C Gamma2/Gamma5 parity gate failed".into());
    }
    let executable_digest = executable_sha256()?;
    let basis_digest = basis_sha256();
    let teleparallel_digest = teleparallel_map_sha256();
    let target_curvature_digest = target_curvature_sha256();
    let root = PathBuf::from(format!("{}.run", path.display()));
    let checkpoints = root.join("checkpoints");
    fs::create_dir_all(&checkpoints).map_err(|e| e.to_string())?;
    let manifest = serde_json::json!({
        "schema_version": "adynkra-11d-right-c-full-chain-four-form-manifest-v1",
        "track": "right-C-H-slot-full-Eq40-Eq25-curl-teleparallel-D-F4",
        "module_sha256": format!("{:x}", Sha256::digest(include_bytes!("eleven_dimensional_corrected_four_form_normalization.rs"))),
        "full_chain_oracle_sha256": format!("{:x}", Sha256::digest(include_bytes!("eleven_dimensional_corrected_full_chain_oracle.rs"))),
        "gamma24_module_sha256": format!("{:x}", Sha256::digest(include_bytes!("eleven_dimensional_gamma24_source_variance.rs"))),
        "executable_sha256": executable_digest,
        "source_basis_sha256": basis_digest,
        "teleparallel_map_sha256": teleparallel_digest,
        "target_curvature_sha256": target_curvature_digest,
        "source_columns": H_HAT_DIMENSION,
        "parallelism_width": width,
        "universal_scale": qi_string(&scale),
        "gamma25_columns_checked_per_degree": parity.columns_checked_per_degree,
        "gamma2_residual_rows": parity.p2_residual_rows,
        "gamma5_residual_rows": parity.p5_residual_rows,
        "gamma2_adapted_sha256": parity.p2_adapted_sha256,
        "gamma2_direct_sha256": parity.p2_direct_sha256,
        "gamma5_adapted_sha256": parity.p5_adapted_sha256,
        "gamma5_direct_sha256": parity.p5_direct_sha256,
        "row_key": "output-coordinate/exterior-spinor-mask/p[11]",
        "candidate": "right-C H-slot adapter -> unchanged full Eq40 DH/DDH solve -> DPsi3 -> target p-wedge curvature",
        "target": "right-C H-slot adapter -> unchanged Eq25 gravitino -> curl -> pinned teleparallel D-F4 map"
    });
    let manifest_hash = json_hash(&manifest)?;
    let manifest_path = root.join("manifest.json");
    if manifest_path.exists() {
        let prior: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if prior != manifest {
            return Err("immutable corrected descendant manifest mismatch".into());
        }
    } else {
        atomic_json(&manifest_path, &manifest, true)?;
    }
    let mut columns = Vec::with_capacity(H_HAT_DIMENSION);
    let mut missing = Vec::new();
    let scale_string = qi_string(&scale);
    for ordinal in 0..H_HAT_DIMENSION {
        let checkpoint = checkpoints.join(format!("column-{ordinal:06}.json"));
        if checkpoint.exists() {
            columns.push(read_checkpoint(
                &checkpoint,
                &manifest_hash,
                ordinal,
                &scale_string,
            )?);
        } else {
            missing.push(ordinal);
        }
    }
    let resumed = columns.len();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .map_err(|e| e.to_string())?;
    for chunk in missing.chunks(width) {
        let batch = pool
            .install(|| {
                chunk
                    .par_iter()
                    .map(|&ordinal| {
                        compare_corrected_column_with_scale(ordinal, &scale)
                            .map(|value| (ordinal, value))
                    })
                    .collect::<Vec<_>>()
            })
            .into_iter()
            .collect::<Result<Vec<_>, String>>()?;
        for (ordinal, value) in batch {
            let envelope = checkpoint_envelope(&manifest_hash, value.clone())?;
            atomic_json(
                &checkpoints.join(format!("column-{ordinal:06}.json")),
                &envelope,
                true,
            )?;
            columns.push(value);
        }
        columns.sort_by_key(|x| x.source_ordinal);
        let completed = columns.len();
        let elapsed = started.elapsed();
        let rate = completed as f64 / elapsed.as_secs_f64().max(0.001);
        atomic_json(
            &root.join("status.json"),
            &serde_json::json!({
                "schema_version": "adynkra-11d-right-c-full-chain-four-form-progress-v1",
                "manifest_sha256": manifest_hash,
                "phase": "compare_columns",
                "completed_columns": completed,
                "total_columns": H_HAT_DIMENSION,
                "columns_per_second": rate,
                "eta_seconds": ((H_HAT_DIMENSION-completed) as f64 / rate.max(0.001)).ceil() as u64,
                "elapsed_milliseconds": elapsed.as_millis(),
                "first_mismatch_observed": columns.iter().any(|x| x.first_exact_mismatch.is_some()),
                "updated_unix_milliseconds": unix_millis()
            }),
            false,
        )?;
    }
    columns.sort_by_key(|x| x.source_ordinal);
    let report = build_full_report(
        width,
        &scale,
        &columns,
        Some(manifest_hash.clone()),
        resumed,
        H_HAT_DIMENSION - resumed,
        started.elapsed(),
        &parity,
    )?;
    if let Some(witness) = &report.first_exact_mismatch {
        let witness_path = root.join("first-exact-mismatch.json");
        if !witness_path.exists() {
            atomic_json(&witness_path, witness, true)?;
        }
    }
    atomic_json(
        &root.join("status.json"),
        &serde_json::json!({
            "schema_version": "adynkra-11d-right-c-full-chain-four-form-progress-v1",
            "manifest_sha256": manifest_hash,
            "phase": "completed",
            "completed_columns": H_HAT_DIMENSION,
            "total_columns": H_HAT_DIMENSION,
            "elapsed_milliseconds": started.elapsed().as_millis(),
            "first_mismatch_observed": report.first_exact_mismatch.is_some(),
            "updated_unix_milliseconds": unix_millis()
        }),
        false,
    )?;
    if path.exists() {
        return Err(format!(
            "report-last path already exists: {}",
            path.display()
        ));
    }
    atomic_json(path, &report, true)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_column() -> CorrectedColumnComparison {
        CorrectedColumnComparison {
            source_ordinal: 0,
            candidate_nonzero_rows: 1,
            teleparallel_nonzero_rows: 2,
            common_support_rows: 0,
            candidate_only_support_rows: 1,
            teleparallel_only_support_rows: 2,
            universal_scale: Some("(0)+(0)i".to_string()),
            exact_residual_rows: 2,
            first_exact_mismatch: None,
            candidate_stream_sha256: "11".repeat(32),
            teleparallel_stream_sha256: "22".repeat(32),
            residual_stream_sha256: "33".repeat(32),
            passed: false,
        }
    }

    #[test]
    fn checkpoint_payload_hash_detects_mutation() {
        let root = std::env::temp_dir().join(format!(
            "adynkra-right-c-checkpoint-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        let path = root.join("column.json");
        let envelope = checkpoint_envelope("manifest", dummy_column()).unwrap();
        atomic_json(&path, &envelope, true).unwrap();
        assert_eq!(
            read_checkpoint(&path, "manifest", 0, "(0)+(0)i").unwrap(),
            dummy_column()
        );
        let mut mutated: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        mutated["payload"]["column"]["candidate_nonzero_rows"] = 9.into();
        atomic_json(&path, &mutated, false).unwrap();
        assert!(read_checkpoint(&path, "manifest", 0, "(0)+(0)i").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrected_column_zero_is_decisive() {
        let report = compare_corrected_column(0).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.candidate_nonzero_rows, 18_972);
        assert_eq!(report.teleparallel_nonzero_rows, 343_720);
        assert_eq!(report.common_support_rows, 0);
        assert_eq!(report.candidate_only_support_rows, 18_972);
        assert_eq!(report.teleparallel_only_support_rows, 343_720);
        assert_eq!(report.universal_scale.as_deref(), Some("(0)+(0)i"));
        assert_eq!(report.exact_residual_rows, 343_720);
        let witness = report.first_exact_mismatch.as_ref().unwrap();
        assert_eq!(witness.output_coordinate, 0);
        assert_eq!(witness.exterior_spinor_mask, 0x0001_0001);
        assert_eq!(
            witness.momentum_exponents,
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(witness.teleparallel, "(1/1280)+(0)i");
        assert!(!report.passed);
    }
}
