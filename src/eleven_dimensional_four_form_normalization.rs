//! Exact relative-normalization test for the Eq. (40) three-form candidate.
//!
//! The conventional solve produces `D_alpha Psi_[3]`.  Applying the independent
//! Abelian target curvature gives `D_alpha G_[4]`.  The physical teleparallel
//! identity in hep-th/0107155 Eq. (3.1g) independently maps the direct Eq. (25)
//! gravitino curl into the same 10,560-component tensor.  This module solves one
//! universal Gaussian-rational scale from the first common nonzero row and then
//! verifies every canonical polynomial row on all 320 `H_hat` basis columns.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use num_rational::Ratio;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_complete_f::visit_gauge_fixed_linearized_gravitino_curl;
use crate::eleven_dimensional_eq40_frame_composition::{
    Eq40FrameSector, visit_eq40_frame_composition,
};
use crate::eleven_dimensional_h_hat_jet::{
    LinearizedFrameSuperfields, canonical_gamma_traceless_frame_basis,
};
use crate::eleven_dimensional_physical_curvature::{
    D_F_FOUR_FORM_DIMENSION, ExactQi, GRAVITINO_CURL_DIMENSION, SPINOR_DIMENSION,
    W_FOUR_FORM_DIMENSION, cached_linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, FormalMomentumMonomial, OrderedSuperderivativeMonomial,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-eq40-four-form-normalization-v2";
pub const MAX_PARALLELISM_WIDTH: usize = 8;
const THREE_FORM_DIMENSION: usize = 165;
const H_HAT_DIMENSION: usize = 320;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct DescendantRowKey {
    output_coordinate: usize,
    monomial: OrderedSuperderivativeMonomial,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GaussianRationalValue {
    pub real_numerator: i64,
    pub real_denominator: i64,
    pub imaginary_numerator: i64,
    pub imaginary_denominator: i64,
}

impl From<&ExactQi> for GaussianRationalValue {
    fn from(value: &ExactQi) -> Self {
        Self {
            real_numerator: *value.real.numer(),
            real_denominator: *value.real.denom(),
            imaginary_numerator: *value.imaginary.numer(),
            imaginary_denominator: *value.imaginary.denom(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExactMismatchWitness {
    pub source_ordinal: usize,
    pub output_coordinate: usize,
    pub exterior_spinor_mask: u32,
    pub momentum_exponents: [u16; 11],
    pub candidate: GaussianRationalValue,
    pub teleparallel: GaussianRationalValue,
    pub scaled_residual: GaussianRationalValue,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FourFormNormalizationReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub eq40_source: &'static str,
    pub teleparallel_source: &'static str,
    pub h_hat_basis_columns: usize,
    pub h_hat_basis_columns_checked: usize,
    pub parallelism_width: usize,
    pub stream_hash_composition: &'static str,
    pub run_manifest_sha256: Option<String>,
    pub columns_resumed: usize,
    pub columns_computed: usize,
    pub d_psi_three_dimension: usize,
    pub d_four_form_dimension: usize,
    pub gravitino_curl_dimension: usize,
    pub target_curvature_dimensions: (usize, usize),
    pub teleparallel_operator_dimensions: (usize, usize),
    pub candidate_nonzero_rows: u64,
    pub teleparallel_nonzero_rows: u64,
    pub compared_row_union: u64,
    pub common_support_rows: u64,
    pub candidate_only_support_rows: u64,
    pub teleparallel_only_support_rows: u64,
    pub exact_residual_rows: u64,
    pub columns_with_exact_agreement: usize,
    pub first_exact_mismatch: Option<ExactMismatchWitness>,
    pub per_column_elapsed_milliseconds: Vec<u64>,
    pub universal_candidate_scale: Option<GaussianRationalValue>,
    pub candidate_stream_sha256: String,
    pub teleparallel_stream_sha256: String,
    pub scaled_residual_sha256: String,
    pub denominator_nonzero: bool,
    pub universal_scale_solved: bool,
    pub every_row_verified_exactly: bool,
    pub decisive_nonproportionality_witness: bool,
    pub comparison_completed: bool,
    pub passed: bool,
    pub result: String,
    pub boundary: &'static str,
}

fn add_value(
    output: &mut BTreeMap<DescendantRowKey, ExactQi>,
    key: DescendantRowKey,
    value: ExactQi,
) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(key.clone()).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&key);
    }
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

fn public_coefficient(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    }
}

fn multiply_momentum(
    left: &OrderedSuperderivativeMonomial,
    right: &ExactPolynomialCoefficient,
) -> Result<OrderedSuperderivativeMonomial, String> {
    let mut exponents = left.momentum.exponents;
    for (axis, exponent) in right.monomial.exponents.into_iter().enumerate() {
        exponents[axis] = exponents[axis]
            .checked_add(u16::from(exponent))
            .ok_or_else(|| format!("four-form normalization momentum overflow on axis {axis}"))?;
    }
    Ok(OrderedSuperderivativeMonomial {
        exterior_spinor_mask: left.exterior_spinor_mask,
        momentum: FormalMomentumMonomial { exponents },
    })
}

fn basis_input(ordinal: usize) -> Result<LinearizedFrameSuperfields, String> {
    let basis = canonical_gamma_traceless_frame_basis();
    let vector = basis.get(ordinal).ok_or_else(|| {
        format!(
            "H_hat basis ordinal {ordinal} is outside dimension {}",
            basis.len()
        )
    })?;
    Ok(LinearizedFrameSuperfields {
        h: vector
            .iter()
            .map(|(&coordinate, coefficient)| {
                (
                    coordinate,
                    CanonicalSuperPolynomial::scalar(coefficient.clone()),
                )
            })
            .collect(),
        scale: CanonicalSuperPolynomial::default(),
        lorentz_two_form: BTreeMap::new(),
    })
}

fn candidate_d_four_form(
    input: &LinearizedFrameSuperfields,
) -> Result<BTreeMap<DescendantRowKey, ExactQi>, String> {
    let curvature = &target_sector_complex(TargetSector::FourForm).curvature;
    if (curvature.rows(), curvature.columns()) != (W_FOUR_FORM_DIMENSION, THREE_FORM_DIMENSION) {
        return Err("unexpected target three-form curvature dimensions".to_string());
    }
    let mut output = BTreeMap::new();
    visit_eq40_frame_composition(input, |entry| {
        if entry.sector != Eq40FrameSector::DPsiThree {
            return Ok(());
        }
        let derivative_spinor = entry.coordinate / THREE_FORM_DIMENSION;
        let potential_component = entry.coordinate % THREE_FORM_DIMENSION;
        if derivative_spinor >= SPINOR_DIMENSION {
            return Err(format!(
                "D Psi_[3] derivative spinor {derivative_spinor} is outside dimension {SPINOR_DIMENSION}"
            ));
        }
        for (four_form_component, operator_term) in curvature.column_terms(potential_component) {
            add_value(
                &mut output,
                DescendantRowKey {
                    output_coordinate: derivative_spinor * W_FOUR_FORM_DIMENSION
                        + four_form_component,
                    monomial: multiply_momentum(&entry.monomial, &operator_term)?,
                },
                multiply(&entry.coefficient, &public_coefficient(&operator_term)),
            );
        }
        Ok(())
    })?;
    Ok(output)
}

fn teleparallel_d_four_form(
    input: &LinearizedFrameSuperfields,
) -> Result<BTreeMap<DescendantRowKey, ExactQi>, String> {
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    if (operator.output_dimension, operator.input_dimension)
        != (D_F_FOUR_FORM_DIMENSION, GRAVITINO_CURL_DIMENSION)
    {
        return Err("unexpected teleparallel D F_[4] operator dimensions".to_string());
    }
    let mut slices = BTreeMap::<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>::new();
    visit_gauge_fixed_linearized_gravitino_curl(input, |entry| {
        let slice = slices.entry(entry.monomial).or_default();
        let value = slice.entry(entry.component).or_insert_with(ExactQi::zero);
        value.add_assign(&entry.coefficient);
        if value.is_zero() {
            slice.remove(&entry.component);
        }
        Ok(())
    })?;

    let mut output = BTreeMap::new();
    for (monomial, curl) in slices {
        for (output_coordinate, coefficient) in operator.apply_sparse(&curl) {
            add_value(
                &mut output,
                DescendantRowKey {
                    output_coordinate,
                    monomial: monomial.clone(),
                },
                coefficient,
            );
        }
    }
    Ok(output)
}

fn hash_ratio(hasher: &mut Sha256, value: &Ratio<i64>) {
    hasher.update(value.numer().to_le_bytes());
    hasher.update(value.denom().to_le_bytes());
}

fn hash_row(hasher: &mut Sha256, source_ordinal: usize, key: &DescendantRowKey, value: &ExactQi) {
    hasher.update((source_ordinal as u64).to_le_bytes());
    hasher.update((key.output_coordinate as u64).to_le_bytes());
    hasher.update(key.monomial.exterior_spinor_mask.to_le_bytes());
    for exponent in key.monomial.momentum.exponents {
        hasher.update(exponent.to_le_bytes());
    }
    hash_ratio(hasher, &value.real);
    hash_ratio(hasher, &value.imaginary);
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ColumnComparison {
    source_ordinal: usize,
    candidate_nonzero_rows: u64,
    teleparallel_nonzero_rows: u64,
    compared_row_union: u64,
    common_support_rows: u64,
    candidate_only_support_rows: u64,
    teleparallel_only_support_rows: u64,
    exact_residual_rows: u64,
    exact_agreement: bool,
    elapsed_milliseconds: u64,
    first_exact_mismatch: Option<ExactMismatchWitness>,
    candidate_digest: [u8; 32],
    teleparallel_digest: [u8; 32],
    residual_digest: [u8; 32],
}

fn column_hasher(domain: &[u8], source_ordinal: usize) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA_VERSION.as_bytes());
    hasher.update(domain);
    hasher.update((source_ordinal as u64).to_le_bytes());
    hasher
}

fn digest(hasher: Sha256) -> [u8; 32] {
    hasher.finalize().into()
}

fn compare_column(
    source_ordinal: usize,
    scale: Option<&ExactQi>,
) -> Result<ColumnComparison, String> {
    let started = Instant::now();
    let input = basis_input(source_ordinal)?;
    let candidate = candidate_d_four_form(&input)?;
    let teleparallel = teleparallel_d_four_form(&input)?;
    let mut candidate_hash = column_hasher(b"candidate", source_ordinal);
    let mut teleparallel_hash = column_hasher(b"teleparallel", source_ordinal);
    let mut residual_hash = column_hasher(b"scaled-residual", source_ordinal);
    for (key, value) in &candidate {
        hash_row(&mut candidate_hash, source_ordinal, key, value);
    }
    for (key, value) in &teleparallel {
        hash_row(&mut teleparallel_hash, source_ordinal, key, value);
    }

    let keys = candidate
        .keys()
        .chain(teleparallel.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut common_support_rows = 0_u64;
    let mut candidate_only_support_rows = 0_u64;
    let mut teleparallel_only_support_rows = 0_u64;
    let mut exact_residual_rows = 0_u64;
    let mut first_exact_mismatch = None;
    for key in &keys {
        let candidate_entry = candidate.get(key);
        let teleparallel_entry = teleparallel.get(key);
        match (candidate_entry, teleparallel_entry) {
            (Some(_), Some(_)) => common_support_rows += 1,
            (Some(_), None) => candidate_only_support_rows += 1,
            (None, Some(_)) => teleparallel_only_support_rows += 1,
            (None, None) => unreachable!("row union contains neither stream"),
        }
        let candidate_value = candidate_entry.cloned().unwrap_or_else(ExactQi::zero);
        let teleparallel_value = teleparallel_entry.cloned().unwrap_or_else(ExactQi::zero);
        let mut residual = scale
            .map(|factor| multiply(factor, &candidate_value))
            .unwrap_or_else(ExactQi::zero);
        residual.add_assign(&teleparallel_value.scaled(&Ratio::from_integer(-1)));
        if !residual.is_zero() {
            exact_residual_rows += 1;
            if first_exact_mismatch.is_none() {
                first_exact_mismatch = Some(ExactMismatchWitness {
                    source_ordinal,
                    output_coordinate: key.output_coordinate,
                    exterior_spinor_mask: key.monomial.exterior_spinor_mask,
                    momentum_exponents: key.monomial.momentum.exponents,
                    candidate: GaussianRationalValue::from(&candidate_value),
                    teleparallel: GaussianRationalValue::from(&teleparallel_value),
                    scaled_residual: GaussianRationalValue::from(&residual),
                });
            }
            hash_row(&mut residual_hash, source_ordinal, key, &residual);
        }
    }

    Ok(ColumnComparison {
        source_ordinal,
        candidate_nonzero_rows: candidate.len() as u64,
        teleparallel_nonzero_rows: teleparallel.len() as u64,
        compared_row_union: keys.len() as u64,
        common_support_rows,
        candidate_only_support_rows,
        teleparallel_only_support_rows,
        exact_residual_rows,
        exact_agreement: scale.is_some() && exact_residual_rows == 0,
        elapsed_milliseconds: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        first_exact_mismatch,
        candidate_digest: digest(candidate_hash),
        teleparallel_digest: digest(teleparallel_hash),
        residual_digest: digest(residual_hash),
    })
}

fn solve_scale(source_ordinal: usize) -> Result<ExactQi, String> {
    let input = basis_input(source_ordinal)?;
    let candidate = candidate_d_four_form(&input)?;
    let teleparallel = teleparallel_d_four_form(&input)?;
    let (key, candidate_value) = candidate.iter().next().ok_or_else(|| {
        format!("H_hat basis column {source_ordinal} has no Eq. (40) candidate row")
    })?;
    let teleparallel_value = teleparallel.get(key).cloned().unwrap_or_else(ExactQi::zero);
    divide(&teleparallel_value, candidate_value)
        .ok_or_else(|| "selected Eq. (40) normalization pivot is zero".to_string())
}

fn aggregate_digest(
    domain: &[u8],
    start: usize,
    count: usize,
    columns: &[ColumnComparison],
    select: impl Fn(&ColumnComparison) -> &[u8; 32],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA_VERSION.as_bytes());
    hasher.update(b"ordered-column-digest-v1");
    hasher.update(domain);
    hasher.update((start as u64).to_le_bytes());
    hasher.update((count as u64).to_le_bytes());
    for column in columns {
        hasher.update((column.source_ordinal as u64).to_le_bytes());
        hasher.update(select(column));
    }
    format!("{:x}", hasher.finalize())
}

fn build_report_from_columns(
    start: usize,
    count: usize,
    parallelism_width: usize,
    scale: &ExactQi,
    columns: &[ColumnComparison],
) -> Result<FourFormNormalizationReport, String> {
    let candidate_nonzero_rows = columns.iter().map(|x| x.candidate_nonzero_rows).sum();
    let teleparallel_nonzero_rows = columns.iter().map(|x| x.teleparallel_nonzero_rows).sum();
    let compared_row_union = columns.iter().map(|x| x.compared_row_union).sum();
    let common_support_rows = columns.iter().map(|x| x.common_support_rows).sum();
    let candidate_only_support_rows = columns.iter().map(|x| x.candidate_only_support_rows).sum();
    let teleparallel_only_support_rows = columns
        .iter()
        .map(|x| x.teleparallel_only_support_rows)
        .sum();
    let exact_residual_rows = columns.iter().map(|x| x.exact_residual_rows).sum();
    let columns_with_exact_agreement = columns.iter().filter(|x| x.exact_agreement).count();
    let first_exact_mismatch = columns
        .iter()
        .find_map(|column| column.first_exact_mismatch.clone());
    let per_column_elapsed_milliseconds = columns
        .iter()
        .map(|column| column.elapsed_milliseconds)
        .collect();
    let universal_scale_solved = true;
    let every_row_verified_exactly = exact_residual_rows == 0;
    let decisive_nonproportionality_witness = exact_residual_rows > 0;
    let comparison_completed = every_row_verified_exactly || decisive_nonproportionality_witness;
    let passed = every_row_verified_exactly && columns_with_exact_agreement == count;
    let result = if passed {
        format!(
            "One universal exact scale maps the Eq. (40) p wedge D Psi_[3] stream to the source-fixed teleparallel D F_[4] image on all {count} checked H_hat basis columns."
        )
    } else {
        format!(
            "The exact pivot fixes a candidate scale, but {exact_residual_rows} polynomial rows disagree across {count} checked H_hat basis columns."
        )
    };

    let target_curvature = &target_sector_complex(TargetSector::FourForm).curvature;
    let teleparallel_operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    Ok(FourFormNormalizationReport {
        schema_version: SCHEMA_VERSION,
        role: "exact relative-normalization comparison between the Eq. (40) Psi_[3] candidate and the physical teleparallel four-form descendant",
        eq40_source: "Gates and Nishino, hep-th/0101037 Eqs. (39)-(40)",
        teleparallel_source: "hep-th/0107155v2 Eq. (3.1g), linearized about flat background",
        h_hat_basis_columns: H_HAT_DIMENSION,
        h_hat_basis_columns_checked: count,
        parallelism_width,
        stream_hash_composition: "SHA-256 over ordered (source ordinal, per-column SHA-256) pairs; each per-column digest hashes typed rows in BTreeMap order",
        run_manifest_sha256: None,
        columns_resumed: 0,
        columns_computed: count,
        d_psi_three_dimension: SPINOR_DIMENSION * THREE_FORM_DIMENSION,
        d_four_form_dimension: D_F_FOUR_FORM_DIMENSION,
        gravitino_curl_dimension: GRAVITINO_CURL_DIMENSION,
        target_curvature_dimensions: (target_curvature.rows(), target_curvature.columns()),
        teleparallel_operator_dimensions: (
            teleparallel_operator.output_dimension,
            teleparallel_operator.input_dimension,
        ),
        candidate_nonzero_rows,
        teleparallel_nonzero_rows,
        compared_row_union,
        common_support_rows,
        candidate_only_support_rows,
        teleparallel_only_support_rows,
        exact_residual_rows,
        columns_with_exact_agreement,
        first_exact_mismatch,
        per_column_elapsed_milliseconds,
        universal_candidate_scale: Some(GaussianRationalValue::from(scale)),
        candidate_stream_sha256: aggregate_digest(b"candidate", start, count, &columns, |x| {
            &x.candidate_digest
        }),
        teleparallel_stream_sha256: aggregate_digest(
            b"teleparallel",
            start,
            count,
            &columns,
            |x| &x.teleparallel_digest,
        ),
        scaled_residual_sha256: aggregate_digest(b"scaled-residual", start, count, &columns, |x| {
            &x.residual_digest
        }),
        denominator_nonzero: true,
        universal_scale_solved,
        every_row_verified_exactly,
        decisive_nonproportionality_witness,
        comparison_completed,
        passed,
        result,
        boundary: "Passing fixes the Eq. (40) candidate scale relative to the direct gravitino branch under the pinned linearized teleparallel identity on the unrestricted canonical H_hat source domain. Failure rules out proportionality on that domain, but does not rule out a relation after additional source constraints. Neither branch proves that every admissible H_hat-to-G4 bidegree has been enumerated, completes local-Lorentz descent, constructs physical K, or establishes off-shell closure.",
    })
}

pub fn verify_range_parallel(
    start: usize,
    count: usize,
    parallelism_width: usize,
) -> Result<FourFormNormalizationReport, String> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| "H_hat basis range overflow".to_string())?;
    if start >= H_HAT_DIMENSION || count == 0 || end > H_HAT_DIMENSION {
        return Err(format!(
            "H_hat basis range [{start},{end}) is outside 0..{H_HAT_DIMENSION}"
        ));
    }
    if parallelism_width == 0 || parallelism_width > MAX_PARALLELISM_WIDTH {
        return Err(format!(
            "parallelism width {parallelism_width} is outside 1..={MAX_PARALLELISM_WIDTH}"
        ));
    }

    // The lexicographically first nonzero candidate row fixes the only scale.
    // Its target coefficient may be zero, in which case the forced scale is
    // exactly zero and any later nonzero target row is a decisive residual.
    let scale = solve_scale(start)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism_width)
        .build()
        .map_err(|error| format!("failed to build four-form worker pool: {error}"))?;
    let column_results = pool.install(|| {
        (start..end)
            .into_par_iter()
            .map(|source_ordinal| compare_column(source_ordinal, Some(&scale)))
            .collect::<Vec<_>>()
    });
    let mut columns = column_results
        .into_iter()
        .collect::<Result<Vec<_>, String>>()?;
    columns.sort_by_key(|column| column.source_ordinal);
    if columns
        .iter()
        .map(|column| column.source_ordinal)
        .ne(start..end)
    {
        return Err("parallel four-form result lost or duplicated a source ordinal".to_string());
    }

    build_report_from_columns(start, count, parallelism_width, &scale, &columns)
}

pub fn verify_range(start: usize, count: usize) -> Result<FourFormNormalizationReport, String> {
    verify_range_parallel(start, count, 1)
}

pub fn verify_parallel(parallelism_width: usize) -> Result<FourFormNormalizationReport, String> {
    verify_range_parallel(0, H_HAT_DIMENSION, parallelism_width)
}

pub fn verify() -> Result<FourFormNormalizationReport, String> {
    verify_parallel(1)
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(temporary, path)
}

const MANIFEST_SCHEMA: &str = "adynkra-11d-four-form-run-manifest-v1";
const CHECKPOINT_SCHEMA: &str = "adynkra-11d-four-form-column-checkpoint-v1";
const STATUS_SCHEMA: &str = "adynkra-11d-four-form-progress-v2";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RunManifest {
    schema_version: String,
    track_id: String,
    source_sha256: String,
    executable_sha256: String,
    source_basis_sha256: String,
    target_curvature_sha256: String,
    teleparallel_map_sha256: String,
    convention_sha256: String,
    row_key_schema: String,
    start_column: usize,
    column_count: usize,
    parallelism_width: usize,
    universal_candidate_scale: GaussianRationalValue,
    heartbeat_interval_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ColumnCheckpointPayload {
    schema_version: String,
    manifest_sha256: String,
    column: ColumnComparison,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ColumnCheckpointEnvelope {
    payload_sha256: String,
    payload: ColumnCheckpointPayload,
}

#[derive(Clone, Debug, Serialize)]
struct ProgressSnapshot {
    schema_version: &'static str,
    manifest_sha256: String,
    state: String,
    phase: String,
    completed_columns: u64,
    total_columns: u64,
    worker_width: usize,
    elapsed_milliseconds: u64,
    current_segment_elapsed_milliseconds: u64,
    columns_per_second: f64,
    eta_seconds: Option<u64>,
    candidate_rows: u64,
    teleparallel_rows: u64,
    common_support_rows: u64,
    candidate_only_rows: u64,
    teleparallel_only_rows: u64,
    residual_rows: u64,
    first_mismatch_observed: bool,
    updated_unix_milliseconds: u128,
    error: Option<String>,
}

#[derive(Default)]
struct LiveRunState {
    completed: AtomicU64,
    candidate_rows: AtomicU64,
    teleparallel_rows: AtomicU64,
    common_support_rows: AtomicU64,
    candidate_only_rows: AtomicU64,
    teleparallel_only_rows: AtomicU64,
    residual_rows: AtomicU64,
    first_mismatch_observed: AtomicBool,
    faulted: AtomicBool,
    error: Mutex<Option<String>>,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn exact_basis_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"canonical-gamma-traceless-H-hat-basis-v1");
    for (ordinal, vector) in canonical_gamma_traceless_frame_basis().iter().enumerate() {
        hasher.update((ordinal as u64).to_le_bytes());
        for (&coordinate, value) in vector {
            hasher.update((coordinate as u64).to_le_bytes());
            hash_ratio(&mut hasher, &value.real);
            hash_ratio(&mut hasher, &value.imaginary);
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
        for (row, coefficient) in curvature.column_terms(column) {
            hasher.update((row as u64).to_le_bytes());
            hasher.update((column as u64).to_le_bytes());
            hasher.update(coefficient.monomial.exponents);
            hasher.update(coefficient.real_numerator.to_le_bytes());
            hasher.update(coefficient.real_denominator.to_le_bytes());
            hasher.update(coefficient.imaginary_numerator.to_le_bytes());
            hasher.update(coefficient.imaginary_denominator.to_le_bytes());
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
            hash_ratio(&mut hasher, &entry.coefficient.real);
            hash_ratio(&mut hasher, &entry.coefficient.imaginary);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn executable_sha256() -> Result<String, String> {
    let path = std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
    let bytes =
        fs::read(&path).map_err(|error| format!("read executable {}: {error}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn source_sha256() -> String {
    let mut hasher = Sha256::new();
    for (name, bytes) in [
        (
            "four_form_normalization",
            include_bytes!("eleven_dimensional_four_form_normalization.rs").as_slice(),
        ),
        (
            "eq40_frame_composition",
            include_bytes!("eleven_dimensional_eq40_frame_composition.rs").as_slice(),
        ),
        (
            "complete_f",
            include_bytes!("eleven_dimensional_complete_f.rs").as_slice(),
        ),
        (
            "physical_curvature",
            include_bytes!("eleven_dimensional_physical_curvature.rs").as_slice(),
        ),
        (
            "target_equation_complex",
            include_bytes!("eleven_dimensional_target_equation_complex.rs").as_slice(),
        ),
        (
            "h_hat_jet",
            include_bytes!("eleven_dimensional_h_hat_jet.rs").as_slice(),
        ),
        (
            "superderivative_normal_form",
            include_bytes!("eleven_dimensional_superderivative_normal_form.rs").as_slice(),
        ),
    ] {
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn build_manifest(
    start: usize,
    count: usize,
    width: usize,
    scale: &ExactQi,
) -> Result<RunManifest, String> {
    let convention_sha256 = sha256_bytes(
        b"hep-th/0101037:Eqs39-40;hep-th/0107155v2:Eq3.1g;alpha-major-Lambda4-lex;ordered-D-v1;p-wedge-DPsi3",
    );
    Ok(RunManifest {
        schema_version: MANIFEST_SCHEMA.to_string(),
        track_id: "relative_normalization".to_string(),
        source_sha256: source_sha256(),
        executable_sha256: executable_sha256()?,
        source_basis_sha256: exact_basis_sha256(),
        target_curvature_sha256: target_curvature_sha256(),
        teleparallel_map_sha256: teleparallel_map_sha256(),
        convention_sha256,
        row_key_schema: "source-ordinal/output-coordinate/exterior-spinor-mask/p[11]-v1"
            .to_string(),
        start_column: start,
        column_count: count,
        parallelism_width: width,
        universal_candidate_scale: GaussianRationalValue::from(scale),
        heartbeat_interval_seconds: 5,
    })
}

fn json_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| format!("serialize hash payload: {error}"))
}

fn atomic_json_new<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("immutable path already exists: {}", path.display()),
        ));
    }
    atomic_json(path, value)
}

fn write_or_validate_manifest(path: &Path, manifest: &RunManifest) -> Result<String, String> {
    let digest = json_sha256(manifest)?;
    if path.exists() {
        let bytes = fs::read(path).map_err(|error| format!("read manifest: {error}"))?;
        let prior: RunManifest = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse immutable manifest: {error}"))?;
        if prior != *manifest {
            return Err(
                "immutable run manifest does not match current execution identity".to_string(),
            );
        }
    } else {
        atomic_json_new(path, manifest).map_err(|error| format!("write manifest: {error}"))?;
    }
    Ok(digest)
}

fn checkpoint_path(root: &Path, ordinal: usize) -> PathBuf {
    root.join("checkpoints")
        .join(format!("column-{ordinal:06}.json"))
}

fn write_checkpoint(
    path: &Path,
    manifest_sha256: &str,
    column: &ColumnComparison,
) -> Result<(), String> {
    let payload = ColumnCheckpointPayload {
        schema_version: CHECKPOINT_SCHEMA.to_string(),
        manifest_sha256: manifest_sha256.to_string(),
        column: column.clone(),
    };
    let envelope = ColumnCheckpointEnvelope {
        payload_sha256: json_sha256(&payload)?,
        payload,
    };
    atomic_json_new(path, &envelope).map_err(|error| format!("write checkpoint: {error}"))
}

fn read_checkpoint(
    path: &Path,
    manifest_sha256: &str,
    expected_ordinal: usize,
) -> Result<ColumnComparison, String> {
    let bytes = fs::read(path).map_err(|error| format!("read checkpoint: {error}"))?;
    let envelope: ColumnCheckpointEnvelope = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse checkpoint {}: {error}", path.display()))?;
    if envelope.payload.schema_version != CHECKPOINT_SCHEMA
        || envelope.payload.manifest_sha256 != manifest_sha256
        || envelope.payload.column.source_ordinal != expected_ordinal
        || json_sha256(&envelope.payload)? != envelope.payload_sha256
    {
        return Err(format!(
            "checkpoint {} failed schema, identity, ordinal, or payload hash validation",
            path.display()
        ));
    }
    Ok(envelope.payload.column)
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn snapshot(
    manifest_sha256: &str,
    state: &LiveRunState,
    started: Instant,
    cumulative_elapsed_offset_milliseconds: u64,
    total: usize,
    width: usize,
    phase: &str,
    terminal_state: &str,
) -> ProgressSnapshot {
    let completed = state.completed.load(Ordering::Relaxed);
    let segment_elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let elapsed_ms = cumulative_elapsed_offset_milliseconds.saturating_add(segment_elapsed_ms);
    let rate = if elapsed_ms == 0 {
        0.0
    } else {
        completed as f64 * 1000.0 / elapsed_ms as f64
    };
    let remaining = (total as u64).saturating_sub(completed);
    ProgressSnapshot {
        schema_version: STATUS_SCHEMA,
        manifest_sha256: manifest_sha256.to_string(),
        state: terminal_state.to_string(),
        phase: phase.to_string(),
        completed_columns: completed,
        total_columns: total as u64,
        worker_width: width,
        elapsed_milliseconds: elapsed_ms,
        current_segment_elapsed_milliseconds: segment_elapsed_ms,
        columns_per_second: rate,
        eta_seconds: (rate > 0.0).then_some((remaining as f64 / rate).ceil() as u64),
        candidate_rows: state.candidate_rows.load(Ordering::Relaxed),
        teleparallel_rows: state.teleparallel_rows.load(Ordering::Relaxed),
        common_support_rows: state.common_support_rows.load(Ordering::Relaxed),
        candidate_only_rows: state.candidate_only_rows.load(Ordering::Relaxed),
        teleparallel_only_rows: state.teleparallel_only_rows.load(Ordering::Relaxed),
        residual_rows: state.residual_rows.load(Ordering::Relaxed),
        first_mismatch_observed: state.first_mismatch_observed.load(Ordering::Relaxed),
        updated_unix_milliseconds: unix_milliseconds(),
        error: state
            .error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone(),
    }
}

fn add_live_column(state: &LiveRunState, column: &ColumnComparison) {
    state.completed.fetch_add(1, Ordering::Relaxed);
    state
        .candidate_rows
        .fetch_add(column.candidate_nonzero_rows, Ordering::Relaxed);
    state
        .teleparallel_rows
        .fetch_add(column.teleparallel_nonzero_rows, Ordering::Relaxed);
    state
        .common_support_rows
        .fetch_add(column.common_support_rows, Ordering::Relaxed);
    state
        .candidate_only_rows
        .fetch_add(column.candidate_only_support_rows, Ordering::Relaxed);
    state
        .teleparallel_only_rows
        .fetch_add(column.teleparallel_only_support_rows, Ordering::Relaxed);
    state
        .residual_rows
        .fetch_add(column.exact_residual_rows, Ordering::Relaxed);
    if column.first_exact_mismatch.is_some() {
        state.first_mismatch_observed.store(true, Ordering::Relaxed);
    }
}

fn prior_cumulative_elapsed_milliseconds(
    status_path: &Path,
    manifest_path: &Path,
    report_path: &Path,
) -> u64 {
    let status_elapsed = fs::read(status_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| value.get("elapsed_milliseconds").and_then(|x| x.as_u64()))
        .unwrap_or(0);
    let filesystem_elapsed = fs::metadata(manifest_path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|manifest_time| {
            let terminal_path = if report_path.exists() {
                report_path.to_path_buf()
            } else {
                status_path.to_path_buf()
            };
            fs::metadata(terminal_path)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|terminal_time| terminal_time.duration_since(manifest_time).ok())
        })
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0);
    // The normalization pivot is solved immediately before first manifest
    // publication because its exact value is itself manifest-bound. Recover
    // that pre-manifest work from the independently persisted timing for the
    // same first source column when a final report exists.
    let pivot_elapsed = fs::read(report_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("per_column_elapsed_milliseconds")
                .and_then(|x| x.as_array())
                .and_then(|values| values.first())
                .and_then(|x| x.as_u64())
        })
        .unwrap_or(0);
    status_elapsed.max(filesystem_elapsed.saturating_add(pivot_elapsed))
}

fn run_observed(
    report_path: &Path,
    start: usize,
    count: usize,
    width: usize,
) -> Result<FourFormNormalizationReport, String> {
    if width == 0 || width > MAX_PARALLELISM_WIDTH {
        return Err(format!(
            "parallelism width {width} is outside 1..={MAX_PARALLELISM_WIDTH}"
        ));
    }
    let end = start
        .checked_add(count)
        .ok_or_else(|| "column range overflow".to_string())?;
    if start >= H_HAT_DIMENSION || count == 0 || end > H_HAT_DIMENSION {
        return Err(format!(
            "column range [{start},{end}) is outside 0..{H_HAT_DIMENSION}"
        ));
    }
    let started = Instant::now();
    let scale = solve_scale(start)?;
    let root = PathBuf::from(format!("{}.run", report_path.display()));
    fs::create_dir_all(root.join("checkpoints"))
        .map_err(|error| format!("create run root: {error}"))?;
    fs::create_dir_all(root.join("witnesses"))
        .map_err(|error| format!("create witness root: {error}"))?;
    let manifest = build_manifest(start, count, width, &scale)?;
    let manifest_path = root.join("manifest.json");
    let status_path = root.join("status.json");
    let cumulative_elapsed_offset_milliseconds =
        prior_cumulative_elapsed_milliseconds(&status_path, &manifest_path, report_path);
    let manifest_sha256 = write_or_validate_manifest(&manifest_path, &manifest)?;
    let live = Arc::new(LiveRunState::default());
    let mut resumed = Vec::new();
    let mut missing = Vec::new();
    for ordinal in start..end {
        let path = checkpoint_path(&root, ordinal);
        if path.exists() {
            let column = read_checkpoint(&path, &manifest_sha256, ordinal)?;
            add_live_column(&live, &column);
            resumed.push(column);
        } else {
            missing.push(ordinal);
        }
    }
    atomic_json(
        &status_path,
        &snapshot(
            &manifest_sha256,
            &live,
            started,
            cumulative_elapsed_offset_milliseconds,
            count,
            width,
            "prepare",
            "running",
        ),
    )
    .map_err(|error| format!("write initial status: {error}"))?;

    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let heartbeat_live = Arc::clone(&live);
    let heartbeat_path = status_path.clone();
    let heartbeat_manifest = manifest_sha256.clone();
    let heartbeat = thread::spawn(move || {
        while stop_rx.recv_timeout(Duration::from_secs(5)).is_err() {
            let value = snapshot(
                &heartbeat_manifest,
                &heartbeat_live,
                started,
                cumulative_elapsed_offset_milliseconds,
                count,
                width,
                "compare_columns",
                "running",
            );
            if let Err(error) = atomic_json(&heartbeat_path, &value) {
                heartbeat_live.faulted.store(true, Ordering::SeqCst);
                *heartbeat_live
                    .error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) =
                    Some(format!("heartbeat publication failed: {error}"));
                return;
            }
        }
    });

    let cancelled = Arc::new(AtomicBool::new(false));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .map_err(|error| format!("build worker pool: {error}"))?;
    let new_results = pool.install(|| {
        missing
            .into_par_iter()
            .map(|ordinal| {
                if cancelled.load(Ordering::SeqCst) || live.faulted.load(Ordering::SeqCst) {
                    return Err("shared fail-fast cancellation observed".to_string());
                }
                let result = compare_column(ordinal, Some(&scale));
                match result {
                    Ok(column) => {
                        if let Err(error) = write_checkpoint(
                            &checkpoint_path(&root, ordinal),
                            &manifest_sha256,
                            &column,
                        ) {
                            cancelled.store(true, Ordering::SeqCst);
                            *live.error.lock().unwrap_or_else(|e| e.into_inner()) =
                                Some(error.clone());
                            return Err(error);
                        }
                        add_live_column(&live, &column);
                        Ok(column)
                    }
                    Err(error) => {
                        cancelled.store(true, Ordering::SeqCst);
                        *live.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error.clone());
                        Err(error)
                    }
                }
            })
            .collect::<Vec<_>>()
    });
    let _ = stop_tx.send(());
    let _ = heartbeat.join();
    let mut computed = match new_results.into_iter().collect::<Result<Vec<_>, String>>() {
        Ok(value) if !live.faulted.load(Ordering::SeqCst) => value,
        Ok(_) => return Err("observability heartbeat faulted".to_string()),
        Err(error) => {
            let _ = atomic_json(
                &status_path,
                &snapshot(
                    &manifest_sha256,
                    &live,
                    started,
                    cumulative_elapsed_offset_milliseconds,
                    count,
                    width,
                    "terminal",
                    "failed",
                ),
            );
            return Err(error);
        }
    };
    let resumed_count = resumed.len();
    let computed_count = computed.len();
    resumed.append(&mut computed);
    resumed.sort_by_key(|column| column.source_ordinal);
    if resumed.iter().map(|x| x.source_ordinal).ne(start..end) {
        return Err("checkpoint merge lost or duplicated a source ordinal".to_string());
    }
    let mut report = build_report_from_columns(start, count, width, &scale, &resumed)?;
    report.run_manifest_sha256 = Some(manifest_sha256.clone());
    report.columns_resumed = resumed_count;
    report.columns_computed = computed_count;

    if let Some(witness) = &report.first_exact_mismatch {
        let witness_path = root.join("witnesses/first-exact-mismatch.json");
        if witness_path.exists() {
            let prior: ExactMismatchWitness = serde_json::from_slice(
                &fs::read(&witness_path).map_err(|error| format!("read witness: {error}"))?,
            )
            .map_err(|error| format!("parse witness: {error}"))?;
            if prior != *witness {
                return Err("immutable first mismatch witness changed".to_string());
            }
        } else {
            atomic_json_new(&witness_path, witness)
                .map_err(|error| format!("write first mismatch witness: {error}"))?;
        }
    }
    atomic_json(
        &status_path,
        &snapshot(
            &manifest_sha256,
            &live,
            started,
            cumulative_elapsed_offset_milliseconds,
            count,
            width,
            "terminal",
            "completed",
        ),
    )
    .map_err(|error| format!("write final status: {error}"))?;

    if report_path.exists() {
        let prior: serde_json::Value = serde_json::from_slice(
            &fs::read(report_path).map_err(|error| format!("read prior report: {error}"))?,
        )
        .map_err(|error| format!("parse prior report: {error}"))?;
        let current = serde_json::to_value(&report)
            .map_err(|error| format!("serialize current report: {error}"))?;
        // Operational resume counts may differ. Semantic fields must not.
        for key in [
            "columns_resumed",
            "columns_computed",
            "per_column_elapsed_milliseconds",
        ] {
            if prior.get(key).is_none() || current.get(key).is_none() {
                return Err(format!("prior report is missing required field {key}"));
            }
        }
        for key in [
            "candidate_stream_sha256",
            "teleparallel_stream_sha256",
            "scaled_residual_sha256",
            "exact_residual_rows",
            "first_exact_mismatch",
            "run_manifest_sha256",
        ] {
            if prior.get(key) != current.get(key) {
                return Err(format!("prior report disagrees on semantic field {key}"));
            }
        }
    } else {
        atomic_json_new(report_path, &report)
            .map_err(|error| format!("publish report last: {error}"))?;
    }
    Ok(report)
}

pub fn write_artifact_parallel(
    path: &Path,
    parallelism_width: usize,
) -> io::Result<FourFormNormalizationReport> {
    run_observed(path, 0, H_HAT_DIMENSION, parallelism_width)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn write_witness_artifact(path: &Path) -> io::Result<FourFormNormalizationReport> {
    run_observed(path, 0, 1, 1).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn write_artifact(path: &Path) -> io::Result<FourFormNormalizationReport> {
    write_artifact_parallel(path, 1)
}

fn exact_from_serialized(value: &GaussianRationalValue) -> Result<ExactQi, String> {
    if value.real_denominator == 0 || value.imaginary_denominator == 0 {
        return Err("serialized Gaussian rational has a zero denominator".to_string());
    }
    Ok(ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    })
}

/// Validate the immutable manifest, every column checkpoint, the canonical
/// checkpoint merge, the exact witness, and the final semantic report without
/// regenerating any scientific row.
pub fn validate_artifact(path: &Path) -> Result<serde_json::Value, String> {
    let root = PathBuf::from(format!("{}.run", path.display()));
    let manifest_path = root.join("manifest.json");
    let manifest: RunManifest = serde_json::from_slice(
        &fs::read(&manifest_path).map_err(|error| format!("read manifest: {error}"))?,
    )
    .map_err(|error| format!("parse manifest: {error}"))?;
    let manifest_sha256 = json_sha256(&manifest)?;
    let report: serde_json::Value = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("read final report: {error}"))?,
    )
    .map_err(|error| format!("parse final report: {error}"))?;
    if report.get("run_manifest_sha256").and_then(|x| x.as_str()) != Some(manifest_sha256.as_str())
    {
        return Err("final report does not bind the canonical manifest hash".to_string());
    }
    let end = manifest
        .start_column
        .checked_add(manifest.column_count)
        .ok_or_else(|| "manifest column range overflow".to_string())?;
    let mut columns = Vec::with_capacity(manifest.column_count);
    for ordinal in manifest.start_column..end {
        columns.push(read_checkpoint(
            &checkpoint_path(&root, ordinal),
            &manifest_sha256,
            ordinal,
        )?);
    }
    let scale = exact_from_serialized(&manifest.universal_candidate_scale)?;
    let replay = build_report_from_columns(
        manifest.start_column,
        manifest.column_count,
        manifest.parallelism_width,
        &scale,
        &columns,
    )?;
    let replay_value = serde_json::to_value(&replay)
        .map_err(|error| format!("serialize replay report: {error}"))?;
    for key in [
        "candidate_nonzero_rows",
        "teleparallel_nonzero_rows",
        "compared_row_union",
        "common_support_rows",
        "candidate_only_support_rows",
        "teleparallel_only_support_rows",
        "exact_residual_rows",
        "columns_with_exact_agreement",
        "first_exact_mismatch",
        "universal_candidate_scale",
        "candidate_stream_sha256",
        "teleparallel_stream_sha256",
        "scaled_residual_sha256",
        "decisive_nonproportionality_witness",
        "comparison_completed",
        "passed",
    ] {
        if report.get(key) != replay_value.get(key) {
            return Err(format!(
                "checkpoint replay disagrees with final report field {key}"
            ));
        }
    }
    let witness: ExactMismatchWitness = serde_json::from_slice(
        &fs::read(root.join("witnesses/first-exact-mismatch.json"))
            .map_err(|error| format!("read exact witness: {error}"))?,
    )
    .map_err(|error| format!("parse exact witness: {error}"))?;
    if Some(&witness) != replay.first_exact_mismatch.as_ref() {
        return Err("exact witness does not equal the minimum checkpoint witness".to_string());
    }
    let status_path = root.join("status.json");
    let recovered_elapsed =
        prior_cumulative_elapsed_milliseconds(&status_path, &manifest_path, path);
    let replay_live = LiveRunState::default();
    for column in &columns {
        add_live_column(&replay_live, column);
    }
    atomic_json(
        &status_path,
        &snapshot(
            &manifest_sha256,
            &replay_live,
            Instant::now(),
            recovered_elapsed,
            manifest.column_count,
            manifest.parallelism_width,
            "terminal",
            "completed",
        ),
    )
    .map_err(|error| format!("repair cumulative terminal status: {error}"))?;
    let status: serde_json::Value = serde_json::from_slice(
        &fs::read(&status_path).map_err(|error| format!("read status: {error}"))?,
    )
    .map_err(|error| format!("parse status: {error}"))?;
    if status.get("state").and_then(|x| x.as_str()) != Some("completed")
        || status.get("completed_columns").and_then(|x| x.as_u64())
            != Some(manifest.column_count as u64)
    {
        return Err("terminal status is not complete".to_string());
    }
    Ok(serde_json::json!({
        "schema_version": "adynkra-11d-four-form-adoption-validation-v1",
        "manifest_sha256": manifest_sha256,
        "checkpoint_count": columns.len(),
        "candidate_stream_sha256": replay.candidate_stream_sha256,
        "teleparallel_stream_sha256": replay.teleparallel_stream_sha256,
        "scaled_residual_sha256": replay.scaled_residual_sha256,
        "first_exact_mismatch": witness,
        "terminal_elapsed_milliseconds": status.get("elapsed_milliseconds"),
        "passed": true
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_h_hat_column_runs_the_typed_normalization_comparison() {
        let report = verify_range(0, 1).unwrap();
        assert_eq!(report.h_hat_basis_columns_checked, 1);
        assert_eq!(report.d_psi_three_dimension, 5_280);
        assert_eq!(report.d_four_form_dimension, 10_560);
        assert_eq!(report.gravitino_curl_dimension, 1_760);
        assert_eq!(report.target_curvature_dimensions, (330, 165));
        assert_eq!(report.teleparallel_operator_dimensions, (10_560, 1_760));
        assert!(report.candidate_nonzero_rows > 0);
        assert!(report.teleparallel_nonzero_rows > 0);
        assert!(report.universal_scale_solved);
        assert_eq!(report.common_support_rows, 0);
        assert_eq!(
            report.candidate_only_support_rows,
            report.candidate_nonzero_rows
        );
        assert_eq!(
            report.teleparallel_only_support_rows,
            report.teleparallel_nonzero_rows
        );
        assert!(report.decisive_nonproportionality_witness);
        assert!(report.comparison_completed);
        assert!(!report.passed);
    }

    #[test]
    fn invalid_basis_ranges_fail_closed() {
        assert!(verify_range(0, 0).is_err());
        assert!(verify_range(320, 1).is_err());
        assert!(verify_range(319, 2).is_err());
    }

    #[test]
    fn observed_resume_is_semantically_identical_and_rejects_mutation() {
        let unique = format!(
            "adynkra-four-form-resume-{}-{}",
            std::process::id(),
            unix_milliseconds()
        );
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).unwrap();
        let report_path = root.join("report.json");

        let first = run_observed(&report_path, 0, 1, 1).unwrap();
        assert_eq!(first.columns_computed, 1);
        assert_eq!(first.columns_resumed, 0);
        let status_path = PathBuf::from(format!("{}.run/status.json", report_path.display()));
        let first_status: serde_json::Value =
            serde_json::from_slice(&fs::read(&status_path).unwrap()).unwrap();
        let first_elapsed = first_status["elapsed_milliseconds"].as_u64().unwrap();
        let resumed = run_observed(&report_path, 0, 1, 1).unwrap();
        assert_eq!(resumed.columns_computed, 0);
        assert_eq!(resumed.columns_resumed, 1);
        let resumed_status: serde_json::Value =
            serde_json::from_slice(&fs::read(&status_path).unwrap()).unwrap();
        let resumed_elapsed = resumed_status["elapsed_milliseconds"].as_u64().unwrap();
        assert!(resumed_elapsed >= first_elapsed);
        let resumed_rate = resumed_status["columns_per_second"].as_f64().unwrap();
        let expected_rate = 1000.0 / resumed_elapsed as f64;
        assert!((resumed_rate - expected_rate).abs() < 1.0e-12);
        assert_eq!(
            first.candidate_stream_sha256,
            resumed.candidate_stream_sha256
        );
        assert_eq!(
            first.teleparallel_stream_sha256,
            resumed.teleparallel_stream_sha256
        );
        assert_eq!(first.scaled_residual_sha256, resumed.scaled_residual_sha256);
        assert_eq!(first.first_exact_mismatch, resumed.first_exact_mismatch);
        assert_eq!(validate_artifact(&report_path).unwrap()["passed"], true);

        let checkpoint =
            checkpoint_path(&PathBuf::from(format!("{}.run", report_path.display())), 0);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&checkpoint).unwrap()).unwrap();
        value["payload"]["column"]["exact_residual_rows"] = serde_json::json!(1);
        fs::write(&checkpoint, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        let error = run_observed(&report_path, 0, 1, 1).unwrap_err();
        assert!(error.contains("failed schema, identity, ordinal, or payload hash"));
    }
}
