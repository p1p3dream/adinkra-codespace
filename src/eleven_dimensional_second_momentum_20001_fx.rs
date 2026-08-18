//! Exact physical `F_X` screen for the nine `(20001)` second-momentum maps.
//!
//! The global 77-column layout is ordered by intermediate channel.  Columns 0 through 52 belong to `(00001)`, `(01001)`, the two `(10001)` paths,
//! and `(11001)`. This module owns columns 53 through 61, the nine exact
//! `(20001)` source maps. No placeholder terms are emitted for the other 68
//! columns.
//!
//! The declared slice uses the unique highest target state, parameter
//! component zero in each of the six gauge-form channels, and the complete
//! `p^2 D^13` wedge branch for those choices.  The convention-fixed physical
//! `F_X` derivative templates produce actual `X_[2]` and `X_[5]` quotient
//! coordinates.  A deterministic linear functional maps each resulting
//! degree-14 exterior mask to degree 13 by deleting its greatest occupied
//! spinor index, then sums collisions exactly.  This is a rank-lower-bound
//! functional, not a claim that the curvature itself has derivative degree 13.
//! The separate `p^3 D^11` contraction branch is not constructed here.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use num_bigint::BigInt;
use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_k_fag_solver::ExactGaussian;
use crate::eleven_dimensional_second_momentum_fx::{
    DegreeTwoMomentumMonomial, SecondMomentumFxColumnTerm, SecondMomentumFxCoverage,
    SecondMomentumFxProvenance, SecondMomentumFxSector, SecondMomentumFxSourceKind,
    SecondMomentumFxStreamingAccumulator, SecondMomentumFxStreamingCheckpoint,
    SecondMomentumGaugeBranch, SecondMomentumGaugeChannel,
};

const SCHEMA_VERSION: &str = "adynkra-11d-second-momentum-20001-fx-v1";
const MAP_CHECKPOINT_DIRECTORY: &str = "results/adynkra_11d_second_momentum_20001_checkpoints";
const MAP_AGGREGATE_ARTIFACT: &str = "results/adynkra_11d_second_momentum_20001_maps.json";
const FIRST_GLOBAL_ORDINAL: usize = 53;
const TRANCHE_COLUMNS: usize = 9;

#[derive(Clone, Copy)]
struct SourceFixture {
    dynkin_label: &'static str,
    copy: usize,
    artifact: &'static str,
    bytes: &'static [u8],
}

macro_rules! fixture {
    ($label:literal, $copy:literal, $artifact:literal) => {
        SourceFixture {
            dynkin_label: $label,
            copy: $copy,
            artifact: $artifact,
            bytes: include_bytes!(concat!(
                "../data/eleven_dimensional_spinor_bridge/",
                $artifact
            )),
        }
    };
}

fn source_fixtures() -> [SourceFixture; TRANCHE_COLUMNS] {
    [
        fixture!("10002", 1, "level12_10002_highest_weight_kernel_1.i16le"),
        fixture!("10002", 2, "level12_10002_highest_weight_kernel_2.i16le"),
        fixture!("20100", 1, "level12_20100_highest_weight_kernel_1.i16le"),
        fixture!("20100", 2, "level12_20100_highest_weight_kernel_2.i16le"),
        fixture!("20010", 1, "level12_20010_highest_weight_kernel_1.i16le"),
        fixture!("20010", 2, "level12_20010_highest_weight_kernel_2.i16le"),
        fixture!("20010", 3, "level12_20010_highest_weight_kernel_3.i16le"),
        fixture!("20002", 1, "level12_20002_highest_weight_kernel_1.i16le"),
        fixture!("20002", 2, "level12_20002_highest_weight_kernel_2.i16le"),
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentum20001ColumnSpec {
    pub global_ordinal: usize,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub intermediate_dynkin_label: String,
    pub symmetric_momentum_path: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coupled_map_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentum20001ColumnAccounting {
    pub global_ordinal: usize,
    pub requested_intermediate_pbw_words: usize,
    pub emitted_descendant_components: u64,
    pub exact_gauge_residual_terms: u64,
    pub exact_projected_fx_terms: u64,
    pub exact_composed_highest_raising_residual_terms_by_simple_root: [usize; 5],
    pub maximum_absolute_descendant_accumulator: String,
    pub estimated_descendant_payload_bytes: u64,
    pub checkpoint_hash_parity_verified: bool,
    pub elapsed_milliseconds: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentum20001FxReport {
    pub schema_version: String,
    pub role: String,
    pub global_77_column_ordinals: Vec<usize>,
    pub columns: Vec<SecondMomentum20001ColumnSpec>,
    pub highest_target_basis_ordinal: usize,
    pub highest_target_pbw_word: Vec<u8>,
    pub selected_parameter_components: Vec<usize>,
    pub selected_gauge_form_degrees: Vec<usize>,
    pub p2_d13_wedge_slice_complete: bool,
    pub p3_d11_contraction_terms_emitted: u64,
    pub actual_convention_fixed_fx_projection_used: bool,
    pub symmetric_momentum_dual_convention: String,
    pub symmetric_momentum_dual_convention_pinned: bool,
    pub degree14_to_degree13_functional: String,
    pub source_map_aggregate_sha256: String,
    pub recoupling_report_sha256: String,
    pub recoupling_20001_certificate_sha256: String,
    pub fixture_manifest_sha256: String,
    pub component_cg_manifest_sha256: String,
    pub canonical_physical_fx_functional_rows_sha256: String,
    pub observed_physical_fx_terms: u64,
    pub observed_nonzero_columns: Vec<usize>,
    pub column_accounting: Vec<SecondMomentum20001ColumnAccounting>,
    pub maximum_observed_process_rss_bytes: Option<u64>,
    pub elapsed_milliseconds: u128,
    pub harness_checkpoint: SecondMomentumFxStreamingCheckpoint,
    pub x2_rank_lower_bound_on_9_column_tranche: usize,
    pub x5_rank_lower_bound_on_9_column_tranche: usize,
    pub joint_rank_lower_bound_on_9_column_tranche: usize,
    pub joint_tranche_nullity_upper_bound: usize,
    pub declared_slice_tranche_no_go_certified: bool,
    pub all_77_columns_materialized: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub full_f_a_g_p_established: bool,
    pub passed: bool,
    pub boundary: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GaugeResidualKey {
    momentum_pair: [usize; 2],
    target_vector_weight_index: usize,
    target_spinor_weight_index: usize,
    exterior_mask: u32,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    sha256(&serde_json::to_vec(value).expect("serialize deterministic p2 (20001) payload"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    serde_json::from_reader(BufReader::new(File::open(path)?))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn observed_process_rss_bytes() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?
        .checked_mul(1_024)
}

fn abstract_checkpoint_path(source: &str) -> std::path::PathBuf {
    Path::new(MAP_CHECKPOINT_DIRECTORY).join(format!("abstract_20001_from_{source}.json"))
}

fn embedded_checkpoint_path(source: &str, copy: usize) -> std::path::PathBuf {
    Path::new(MAP_CHECKPOINT_DIRECTORY)
        .join(format!("embedded_20001_from_{source}_copy{copy}.json"))
}

fn multiply(left: &ExactGaussian, right: &ExactGaussian) -> ExactGaussian {
    ExactGaussian {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn gaussian_from_i128(value: i128) -> ExactGaussian {
    ExactGaussian {
        real: Ratio::from_integer(BigInt::from(value)),
        imaginary: Ratio::from_integer(BigInt::from(0)),
    }
}

fn gaussian_from_small_ratios(real: &Ratio<i64>, imaginary: &Ratio<i64>) -> ExactGaussian {
    ExactGaussian {
        real: Ratio::new(BigInt::from(*real.numer()), BigInt::from(*real.denom())),
        imaginary: Ratio::new(
            BigInt::from(*imaginary.numer()),
            BigInt::from(*imaginary.denom()),
        ),
    }
}

fn scale_rational(value: &ExactGaussian, numerator: i64, denominator: i64) -> ExactGaussian {
    let scale = Ratio::new(BigInt::from(numerator), BigInt::from(denominator));
    ExactGaussian {
        real: value.real.clone() * scale.clone(),
        imaginary: value.imaginary.clone() * scale,
    }
}

fn add_exact(target: &mut ExactGaussian, value: &ExactGaussian) {
    target.real += value.real.clone();
    target.imaginary += value.imaginary.clone();
}

fn right_wedge_sign(mask: u32, spinor: usize) -> Option<i64> {
    let bit = 1_u32 << spinor;
    if mask & bit != 0 {
        return None;
    }
    let greater = if spinor + 1 == 32 {
        0
    } else {
        (mask >> (spinor + 1)).count_ones()
    };
    Some(if greater % 2 == 0 { 1 } else { -1 })
}

fn degree14_to_degree13_mask(mask: u32) -> u32 {
    assert_eq!(mask.count_ones(), 14);
    mask ^ (1_u32 << (31 - mask.leading_zeros()))
}

type Weight = [i8; 5];
const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];

fn vector_weights() -> [Weight; 11] {
    let mut weights = [[0_i8; 5]; 11];
    for axis in 0..5 {
        weights[2 * axis][axis] = 2;
        weights[2 * axis + 1][axis] = -2;
    }
    weights
}

fn spinor_weights() -> [Weight; 32] {
    std::array::from_fn(|index| {
        std::array::from_fn(|axis| {
            if (index >> (4 - axis)) & 1 == 0 {
                1
            } else {
                -1
            }
        })
    })
}

fn raised_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = std::array::from_fn(|axis| weights[index][axis] + SIMPLE_ROOTS[root][axis]);
    weights.iter().position(|weight| *weight == target)
}

fn raise_vector_weight(
    weight: Weight,
    root: usize,
    weights: &[Weight; 11],
) -> Option<(usize, i64)> {
    let mut target = weight;
    if root < 4 {
        if weight[root] == 0 && weight[root + 1] == 2 {
            target[root] = 2;
            target[root + 1] = 0;
        } else if weight[root] == -2 && weight[root + 1] == 0 {
            target[root] = 0;
            target[root + 1] = -2;
        } else {
            return None;
        }
        Some((weights.iter().position(|item| *item == target).unwrap(), 1))
    } else if weight == [0; 5] {
        target[4] = 2;
        Some((weights.iter().position(|item| *item == target).unwrap(), 2))
    } else if weight[4] == -2 {
        Some((weights.iter().position(|item| *item == [0; 5]).unwrap(), 1))
    } else {
        None
    }
}

fn exterior_replacement_sign(mask: u32, first: usize, second: usize) -> i128 {
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    let interval = if high == low + 1 {
        0
    } else {
        ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
    };
    if (mask & interval).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn add_checked_i128<K: Ord>(output: &mut BTreeMap<K, i128>, key: K, value: i128) -> io::Result<()> {
    let entry = output.entry(key).or_insert(0);
    *entry = entry
        .checked_add(value)
        .ok_or_else(|| io::Error::other("exact p2 Chevalley accumulator overflow"))?;
    Ok(())
}

/// Verify the actual tensor substitution used below.  The reciprocal terms
/// define a highest target state inside `Sym^2(V) tensor (20001)`.  Replacing
/// every abstract `(20001)` descendant by the exact source-map descendant must
/// therefore remain highest.  This is not a raw coefficient dot product and
/// requires no assumed orthonormal intermediate basis.
fn composed_highest_raising_residuals(
    state: &BTreeMap<([usize; 2], usize, u32), i128>,
) -> io::Result<[usize; 5]> {
    let vectors = vector_weights();
    let spinors = spinor_weights();
    let mut residuals = [0_usize; 5];
    for root in 0..5 {
        let mut output = BTreeMap::new();
        for (&(pair, free_spinor, exterior_mask), &coefficient) in state {
            for slot in 0..2 {
                if let Some((next, factor)) =
                    raise_vector_weight(vectors[pair[slot]], root, &vectors)
                {
                    let mut next_pair = pair;
                    next_pair[slot] = next;
                    next_pair.sort_unstable();
                    add_checked_i128(
                        &mut output,
                        (next_pair, free_spinor, exterior_mask),
                        coefficient * i128::from(factor),
                    )?;
                }
            }
            if let Some(next) = raised_spinor_index(free_spinor, root, &spinors) {
                add_checked_i128(&mut output, (pair, next, exterior_mask), coefficient)?;
            }
            let mut occupied = exterior_mask;
            while occupied != 0 {
                let old = occupied.trailing_zeros() as usize;
                occupied &= occupied - 1;
                let Some(next) = raised_spinor_index(old, root, &spinors) else {
                    continue;
                };
                if exterior_mask & (1_u32 << next) != 0 {
                    continue;
                }
                let next_mask = exterior_mask ^ (1_u32 << old) ^ (1_u32 << next);
                add_checked_i128(
                    &mut output,
                    (pair, free_spinor, next_mask),
                    coefficient * exterior_replacement_sign(exterior_mask, old, next),
                )?;
            }
        }
        output.retain(|_, value| *value != 0);
        residuals[root] = output.len();
    }
    Ok(residuals)
}

fn requested_words(
    channel: &crate::eleven_dimensional_second_momentum_remaining_recouplings::RemainingRecouplingCertificate,
) -> Vec<Vec<u8>> {
    channel
        .reciprocal_terms
        .iter()
        .map(|term| term.intermediate_pbw_word_simple_roots.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn validate_map_checkpoint(
    fixture: SourceFixture,
    abstract_checkpoint: &crate::eleven_dimensional_second_momentum_20001_maps::SecondMomentum20001AbstractCheckpoint,
    embedded_checkpoint: &crate::eleven_dimensional_second_momentum_20001_maps::SecondMomentum20001EmbeddedCheckpoint,
) -> io::Result<()> {
    let certificate_sha256 = sha256_json(&abstract_checkpoint.certificate);
    if !abstract_checkpoint.passed
        || abstract_checkpoint.source_dynkin_label != fixture.dynkin_label
        || abstract_checkpoint.target_dynkin_label != "20001"
        || abstract_checkpoint.source_fixture_sha256 != sha256(fixture.bytes)
        || abstract_checkpoint.certificate_sha256 != certificate_sha256
        || !embedded_checkpoint.passed
        || embedded_checkpoint.job.source_dynkin_label != fixture.dynkin_label
        || embedded_checkpoint.job.source_copy != fixture.copy
        || embedded_checkpoint.source_fixture != fixture.artifact
        || embedded_checkpoint.source_fixture_sha256 != sha256(fixture.bytes)
        || embedded_checkpoint.abstract_certificate_sha256 != certificate_sha256
        || embedded_checkpoint
            .certificate
            .exact_raising_residual_terms_by_simple_root
            != [0; 5]
        || !embedded_checkpoint.certificate.passed
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid or mismatched exact (20001) map checkpoint",
        ));
    }
    Ok(())
}

fn build_column_gauge_residuals(
    fixture: SourceFixture,
    reciprocal: &crate::eleven_dimensional_second_momentum_remaining_recouplings::RemainingRecouplingCertificate,
    highest_target: &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
) -> io::Result<(
    Vec<BTreeMap<GaugeResidualKey, ExactGaussian>>,
    crate::eleven_dimensional_level16_couplings::SecondMomentum20001DescendantAccounting,
    [usize; 5],
)> {
    let abstract_checkpoint = read_json::<
        crate::eleven_dimensional_second_momentum_20001_maps::SecondMomentum20001AbstractCheckpoint,
    >(&abstract_checkpoint_path(fixture.dynkin_label))?;
    let embedded_checkpoint = read_json::<
        crate::eleven_dimensional_second_momentum_20001_maps::SecondMomentum20001EmbeddedCheckpoint,
    >(&embedded_checkpoint_path(
        fixture.dynkin_label,
        fixture.copy,
    ))?;
    validate_map_checkpoint(fixture, &abstract_checkpoint, &embedded_checkpoint)?;

    let words = requested_words(reciprocal);
    let word_ordinals = words
        .iter()
        .enumerate()
        .map(|(ordinal, word)| (word.clone(), ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut descendants = vec![Vec::new(); words.len()];
    let accounting = crate::eleven_dimensional_level16_couplings::
        visit_second_momentum_20001_descendant_components(
            &abstract_checkpoint.certificate,
            fixture.copy,
            fixture.artifact,
            2,
            fixture.bytes,
            &embedded_checkpoint.source_fixture_sha256,
            &embedded_checkpoint.coupled_map_sha256,
            &words,
            |entry| {
                descendants[entry.requested_word_ordinal].push((
                    entry.free_spinor_weight_index,
                    entry.exterior_mask,
                    entry.coefficient,
                ));
                Ok(())
            },
        )?;
    if descendants.iter().any(Vec::is_empty) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "requested (20001) descendant materialized no exact components",
        ));
    }

    let mut recoupled = BTreeMap::<([usize; 2], usize, u32), i128>::new();
    for term in &reciprocal.reciprocal_terms {
        let word_ordinal = word_ordinals[&term.intermediate_pbw_word_simple_roots];
        for &(free_spinor, exterior_mask, source_coefficient) in &descendants[word_ordinal] {
            let value = i128::from(source_coefficient)
                .checked_mul(i128::from(term.primitive_coefficient))
                .ok_or_else(|| io::Error::other("p2 recoupling coefficient overflow"))?;
            let entry = recoupled
                .entry((term.momentum_pair, free_spinor, exterior_mask))
                .or_insert(0);
            *entry = entry
                .checked_add(value)
                .ok_or_else(|| io::Error::other("p2 recoupling accumulator overflow"))?;
        }
    }
    recoupled.retain(|_, coefficient| *coefficient != 0);
    if recoupled.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact (20001) momentum recoupling vanished",
        ));
    }
    let raising_residuals = composed_highest_raising_residuals(&recoupled)?;
    if raising_residuals != [0; 5] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("substituted (20001) p2 target state is not highest: {raising_residuals:?}"),
        ));
    }

    let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis();
    let mut residuals_by_degree = Vec::with_capacity(6);
    for degree in 0..6 {
        let (_, _, matrix) = gauge_basis
            .iter()
            .find(|(candidate, _, _)| *candidate == degree)
            .ok_or_else(|| io::Error::other("missing parameter component zero"))?;
        let mut residuals = BTreeMap::<GaugeResidualKey, ExactGaussian>::new();
        for (&(momentum_pair, free_spinor, exterior_mask), source_coefficient) in &recoupled {
            for derivative_spinor in 0..32 {
                let Some(wedge_sign) = right_wedge_sign(exterior_mask, derivative_spinor) else {
                    continue;
                };
                let gauge = &matrix[free_spinor][derivative_spinor];
                if *gauge.re.numer() == 0 && *gauge.im.numer() == 0 {
                    continue;
                }
                let gauge_coefficient = gaussian_from_small_ratios(&gauge.re, &gauge.im);
                let source = gaussian_from_i128(
                    source_coefficient
                        .checked_mul(i128::from(wedge_sign))
                        .ok_or_else(|| io::Error::other("p2 wedge coefficient overflow"))?,
                );
                let source = multiply(&source, &gauge_coefficient);
                for target in &highest_target.raw_terms {
                    let value = scale_rational(&source, target.numerator, target.denominator);
                    let key = GaugeResidualKey {
                        momentum_pair,
                        target_vector_weight_index: target.vector_weight_index,
                        target_spinor_weight_index: target.spinor_weight_index,
                        exterior_mask: exterior_mask | (1_u32 << derivative_spinor),
                    };
                    add_exact(
                        residuals.entry(key).or_insert_with(ExactGaussian::zero),
                        &value,
                    );
                }
            }
        }
        residuals.retain(|_, value| !value.is_zero());
        residuals_by_degree.push(residuals);
    }
    Ok((residuals_by_degree, accounting, raising_residuals))
}

fn visit_projected_fx_terms<F>(
    global_ordinal: usize,
    residuals_by_degree: &[BTreeMap<GaugeResidualKey, ExactGaussian>],
    template_cache: &mut BTreeMap<
        (usize, usize),
        Vec<crate::eleven_dimensional_physical_curvature::ExactFxDerivativeTemplateEntry>,
    >,
    mut visit: F,
) -> io::Result<u64>
where
    F: FnMut(SecondMomentumFxColumnTerm) -> io::Result<()>,
{
    let mut emitted = 0_u64;
    for (degree, residuals) in residuals_by_degree.iter().enumerate() {
        for (key, source_coefficient) in residuals {
            let cache_key = (
                key.target_vector_weight_index,
                key.target_spinor_weight_index,
            );
            if !template_cache.contains_key(&cache_key) {
                let mut entries = Vec::new();
                crate::eleven_dimensional_physical_curvature::visit_exact_fx_derivative_templates(
                    cache_key.0,
                    cache_key.1,
                    |entry| entries.push(entry),
                )
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                template_cache.insert(cache_key, entries);
            }
            for template in &template_cache[&cache_key] {
                let derivative = template.derivative_spinor_weight_index;
                let Some(sign) = right_wedge_sign(key.exterior_mask, derivative) else {
                    continue;
                };
                let output_mask = key.exterior_mask | (1_u32 << derivative);
                let functional_mask = degree14_to_degree13_mask(output_mask);
                let mut value = multiply(source_coefficient, &template.coefficient);
                if sign < 0 {
                    value.real = -value.real;
                    value.imaginary = -value.imaginary;
                }
                let sector = if template.x_two_sector {
                    SecondMomentumFxSector::X2
                } else {
                    SecondMomentumFxSector::X5
                };
                if value.is_zero() {
                    continue;
                }
                visit(SecondMomentumFxColumnTerm {
                    coefficient_column: global_ordinal,
                    gauge_channel: SecondMomentumGaugeChannel::new(degree)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
                    gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
                    source_momentum: DegreeTwoMomentumMonomial::from_pair(
                        key.momentum_pair[0],
                        key.momentum_pair[1],
                    )
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
                    parameter_component: 0,
                    target_coordinate: template.output_coordinate,
                    spinor_derivative_mask: functional_mask,
                    sector,
                    coefficient: value,
                })?;
                emitted = emitted
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("projected p2 F_X term count overflow"))?;
            }
        }
    }
    Ok(emitted)
}

fn project_column_fx(
    global_ordinal: usize,
    residuals_by_degree: &[BTreeMap<GaugeResidualKey, ExactGaussian>],
    template_cache: &mut BTreeMap<
        (usize, usize),
        Vec<crate::eleven_dimensional_physical_curvature::ExactFxDerivativeTemplateEntry>,
    >,
    harness: &mut SecondMomentumFxStreamingAccumulator,
) -> io::Result<u64> {
    visit_projected_fx_terms(
        global_ordinal,
        residuals_by_degree,
        template_cache,
        |term| {
            harness
                .push(term)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        },
    )
}

pub fn construct() -> io::Result<SecondMomentum20001FxReport> {
    let started = Instant::now();
    let map_aggregate_bytes = fs::read(MAP_AGGREGATE_ARTIFACT)?;
    let map_aggregate: crate::eleven_dimensional_second_momentum_20001_maps::SecondMomentum20001Report =
        serde_json::from_slice(&map_aggregate_bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !map_aggregate.passed
        || !map_aggregate.source_to_20001_component_maps_complete
        || map_aggregate.exact_embedded_component_maps != TRANCHE_COLUMNS
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incomplete exact (20001) source-map aggregate",
        ));
    }
    let source_map_aggregate_sha256 = sha256(&map_aggregate_bytes);

    let recoupling_report =
        crate::eleven_dimensional_second_momentum_remaining_recouplings::verify();
    let reciprocal = recoupling_report
        .channels
        .iter()
        .find(|channel| channel.intermediate_dynkin_label == "20001")
        .ok_or_else(|| io::Error::other("missing exact (20001) momentum recoupling"))?;
    if !recoupling_report.passed
        || !reciprocal.passed
        || reciprocal.reciprocal_terms.is_empty()
        || !reciprocal.exact_chevalley_equivariance_verified
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "uncertified (20001) momentum recoupling",
        ));
    }

    let target_basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let target_dual = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let highest_target = target_basis
        .iter()
        .find(|state| state.pbw_word_simple_roots.is_empty())
        .ok_or_else(|| io::Error::other("missing unique highest target state"))?;
    if target_basis
        .iter()
        .filter(|state| state.pbw_word_simple_roots.is_empty())
        .count()
        != 1
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "target basis does not have a unique highest state",
        ));
    }
    let highest_target_dual = &target_dual[highest_target.ordinal];

    let fixtures = source_fixtures();
    let fixture_manifest = fixtures
        .iter()
        .enumerate()
        .map(|(local, fixture)| {
            (
                FIRST_GLOBAL_ORDINAL + local,
                fixture.dynkin_label,
                fixture.copy,
                fixture.artifact,
                sha256(fixture.bytes),
            )
        })
        .collect::<Vec<_>>();
    let fixture_manifest_sha256 = sha256_json(&fixture_manifest);
    let component_cg_manifest_sha256 = sha256_json(&(
        source_map_aggregate_sha256.as_str(),
        recoupling_report.report_sha256.as_str(),
        reciprocal.certificate_sha256.as_str(),
        highest_target.ordinal,
        vec![0_usize],
        vec![0_usize, 1, 2, 3, 4, 5],
        "degree14 mask -> delete greatest occupied spinor -> degree13 mask",
    ));

    let provenance = SecondMomentumFxProvenance {
        source_kind: SecondMomentumFxSourceKind::PhysicalRecoupledStream,
        campaign_id: "p2-d12-20001-highest-target-parameter-zero-v1".to_string(),
        representation_inventory_sha256:
            crate::eleven_dimensional_second_momentum_fx::SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256
                .to_string(),
        level12_fixture_manifest_sha256: fixture_manifest_sha256.clone(),
        component_cg_manifest_sha256: component_cg_manifest_sha256.clone(),
        coefficient_layout_sha256:
            crate::eleven_dimensional_second_momentum_fx::second_momentum_fx_coefficient_layout_sha256(),
        expected_canonical_stream_sha256: "00".repeat(32),
    };
    let coverage = SecondMomentumFxCoverage {
        all_77_component_cg_maps_complete: false,
        p2_d13_wedge_branch_complete: false,
        p3_d11_contraction_branch_complete: false,
        all_six_gauge_channels_complete: false,
        full_parameter_projection_complete: false,
        full_target_projection_complete: false,
        complete_x2_projection: false,
        complete_x5_projection: false,
        j_and_w_sectors_complete: false,
        generic_momentum_tower_complete_or_proved_sufficient: false,
    };
    let mut harness = SecondMomentumFxStreamingAccumulator::new(provenance, coverage)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut columns = Vec::new();
    let mut column_accounting = Vec::new();
    let mut template_cache = BTreeMap::new();
    let mut maximum_observed_process_rss_bytes = observed_process_rss_bytes();
    for (local_ordinal, fixture) in fixtures.into_iter().enumerate() {
        let column_started = Instant::now();
        let global_ordinal = FIRST_GLOBAL_ORDINAL + local_ordinal;
        let embedded_checkpoint = read_json::<
            crate::eleven_dimensional_second_momentum_20001_maps::SecondMomentum20001EmbeddedCheckpoint,
        >(&embedded_checkpoint_path(fixture.dynkin_label, fixture.copy))?;
        let (residuals, accounting, raising_residuals) =
            build_column_gauge_residuals(fixture, reciprocal, highest_target_dual)?;
        let exact_gauge_residual_terms = residuals.iter().map(BTreeMap::len).sum::<usize>();
        let projected_terms = project_column_fx(
            global_ordinal,
            &residuals,
            &mut template_cache,
            &mut harness,
        )?;
        if projected_terms == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("physical F_X slice vanished for column {global_ordinal}"),
            ));
        }
        columns.push(SecondMomentum20001ColumnSpec {
            global_ordinal,
            source_dynkin_label: fixture.dynkin_label.to_string(),
            source_copy: fixture.copy,
            intermediate_dynkin_label: "20001".to_string(),
            symmetric_momentum_path: 1,
            source_fixture: fixture.artifact.to_string(),
            source_fixture_sha256: sha256(fixture.bytes),
            coupled_map_sha256: embedded_checkpoint.coupled_map_sha256,
        });
        column_accounting.push(SecondMomentum20001ColumnAccounting {
            global_ordinal,
            requested_intermediate_pbw_words: accounting.requested_pbw_words,
            emitted_descendant_components: accounting.emitted_nonzero_components,
            exact_gauge_residual_terms: u64::try_from(exact_gauge_residual_terms).unwrap(),
            exact_projected_fx_terms: projected_terms,
            exact_composed_highest_raising_residual_terms_by_simple_root: raising_residuals,
            maximum_absolute_descendant_accumulator: accounting
                .maximum_absolute_checked_accumulator
                .to_string(),
            estimated_descendant_payload_bytes: accounting.estimated_payload_bytes,
            checkpoint_hash_parity_verified: accounting.checkpoint_hash_parity_verified,
            elapsed_milliseconds: column_started.elapsed().as_millis(),
        });
        eprintln!(
            "second-momentum (20001) F_X: completed column {global_ordinal} ({}/{TRANCHE_COLUMNS}), gauge residuals {}, projected terms {}, elapsed {:.1}s",
            local_ordinal + 1,
            exact_gauge_residual_terms,
            projected_terms,
            column_started.elapsed().as_secs_f64(),
        );
        maximum_observed_process_rss_bytes = maximum_observed_process_rss_bytes
            .into_iter()
            .chain(observed_process_rss_bytes())
            .max();
    }

    let expected_ordinals =
        (FIRST_GLOBAL_ORDINAL..FIRST_GLOBAL_ORDINAL + TRANCHE_COLUMNS).collect::<Vec<_>>();
    let harness_checkpoint = harness
        .finalize()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let observed_nonzero_columns = harness_checkpoint.report.observed_nonzero_columns.clone();
    if observed_nonzero_columns != expected_ordinals {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not every exact (20001) column reached the physical F_X slice",
        ));
    }
    let x2_rank = harness_checkpoint.report.global_x2.rank;
    let x5_rank = harness_checkpoint.report.global_x5.rank;
    let joint_rank = harness_checkpoint.report.global_joint.rank;
    if x2_rank > TRANCHE_COLUMNS || x5_rank > TRANCHE_COLUMNS || joint_rank > TRANCHE_COLUMNS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rank exceeds the nine physically materialized columns",
        ));
    }
    let declared_slice_tranche_no_go_certified = joint_rank == TRANCHE_COLUMNS;
    let report = SecondMomentum20001FxReport {
        schema_version: SCHEMA_VERSION.to_string(),
        role: "exact physical F_X rank-lower-bound screen on the nine (20001) p2D12 columns"
            .to_string(),
        global_77_column_ordinals: expected_ordinals,
        columns,
        highest_target_basis_ordinal: highest_target.ordinal,
        highest_target_pbw_word: highest_target.pbw_word_simple_roots.clone(),
        selected_parameter_components: vec![0],
        selected_gauge_form_degrees: (0..6).collect(),
        p2_d13_wedge_slice_complete: true,
        p3_d11_contraction_terms_emitted: 0,
        actual_convention_fixed_fx_projection_used: true,
        symmetric_momentum_dual_convention: recoupling_report.basis_convention.clone(),
        symmetric_momentum_dual_convention_pinned: true,
        degree14_to_degree13_functional:
            "delete the greatest occupied spinor index and sum all collisions exactly"
                .to_string(),
        source_map_aggregate_sha256,
        recoupling_report_sha256: recoupling_report.report_sha256,
        recoupling_20001_certificate_sha256: reciprocal.certificate_sha256.clone(),
        fixture_manifest_sha256,
        component_cg_manifest_sha256,
        canonical_physical_fx_functional_rows_sha256: harness_checkpoint
            .report
            .canonical_functional_rows_sha256
            .clone(),
        observed_physical_fx_terms: harness_checkpoint.report.observed_terms,
        observed_nonzero_columns,
        column_accounting,
        maximum_observed_process_rss_bytes,
        elapsed_milliseconds: started.elapsed().as_millis(),
        harness_checkpoint,
        x2_rank_lower_bound_on_9_column_tranche: x2_rank,
        x5_rank_lower_bound_on_9_column_tranche: x5_rank,
        joint_rank_lower_bound_on_9_column_tranche: joint_rank,
        joint_tranche_nullity_upper_bound: TRANCHE_COLUMNS - joint_rank,
        declared_slice_tranche_no_go_certified,
        all_77_columns_materialized: false,
        full_parameter_projection_complete: false,
        full_target_projection_complete: false,
        full_f_a_g_p_established: false,
        passed: true,
        boundary: "This is an exact rank lower bound for global columns 53 through 61 on the unique highest-target, parameter-zero, all-six-channel p2D13 wedge slice. Full rank nine proves a no-go for only this nine-column tranche. The other 68 columns are absent rather than synthesized. The p3D11 branch, complete parameter and target projections, J/W, the generic momentum tower, complete F, and full F A G_p remain open."
            .to_string(),
    };
    validate(&report)?;
    Ok(report)
}

pub fn validate(report: &SecondMomentum20001FxReport) -> io::Result<()> {
    crate::eleven_dimensional_second_momentum_fx::validate_second_momentum_fx_streaming_checkpoint(
        &report.harness_checkpoint,
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let expected_ordinals =
        (FIRST_GLOBAL_ORDINAL..FIRST_GLOBAL_ORDINAL + TRANCHE_COLUMNS).collect::<Vec<_>>();
    let joint_rank = report.harness_checkpoint.report.global_joint.rank;
    if report.schema_version != SCHEMA_VERSION
        || report.global_77_column_ordinals != expected_ordinals
        || report.observed_nonzero_columns != expected_ordinals
        || report.canonical_physical_fx_functional_rows_sha256
            != report
                .harness_checkpoint
                .report
                .canonical_functional_rows_sha256
        || report.columns.len() != TRANCHE_COLUMNS
        || report.column_accounting.len() != TRANCHE_COLUMNS
        || report.columns.iter().enumerate().any(|(local, column)| {
            column.global_ordinal != FIRST_GLOBAL_ORDINAL + local
                || column.intermediate_dynkin_label != "20001"
                || column.symmetric_momentum_path != 1
        })
        || report.column_accounting.iter().any(|column| {
            !column.checkpoint_hash_parity_verified
                || column.exact_composed_highest_raising_residual_terms_by_simple_root != [0; 5]
        })
        || report.highest_target_pbw_word != Vec::<u8>::new()
        || report.selected_parameter_components != [0]
        || report.selected_gauge_form_degrees != [0, 1, 2, 3, 4, 5]
        || !report.p2_d13_wedge_slice_complete
        || report.p3_d11_contraction_terms_emitted != 0
        || !report.actual_convention_fixed_fx_projection_used
        || !report.symmetric_momentum_dual_convention_pinned
        || report.symmetric_momentum_dual_convention.is_empty()
        || report.x2_rank_lower_bound_on_9_column_tranche
            != report.harness_checkpoint.report.global_x2.rank
        || report.x5_rank_lower_bound_on_9_column_tranche
            != report.harness_checkpoint.report.global_x5.rank
        || report.joint_rank_lower_bound_on_9_column_tranche != joint_rank
        || report.joint_tranche_nullity_upper_bound != TRANCHE_COLUMNS - joint_rank
        || report.declared_slice_tranche_no_go_certified != (joint_rank == TRANCHE_COLUMNS)
        || report.all_77_columns_materialized
        || report.full_parameter_projection_complete
        || report.full_target_projection_complete
        || report.full_f_a_g_p_established
        || report
            .harness_checkpoint
            .report
            .declared_slice_no_go_certified
        || report.harness_checkpoint.report.global_joint.nullity
            != crate::eleven_dimensional_second_momentum_fx::SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
                - joint_rank
        || !report.passed
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "second-momentum (20001) physical F_X report invariant failed",
        ));
    }
    Ok(())
}

pub fn write_artifact(path: &Path) -> io::Result<SecondMomentum20001FxReport> {
    let report = construct()?;
    validate(&report)?;
    if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_global_layout_owns_columns_53_through_61() {
        let fixtures = source_fixtures();
        assert_eq!(fixtures.len(), 9);
        assert_eq!(FIRST_GLOBAL_ORDINAL, 3 + 12 + 8 + 30);
        assert_eq!(FIRST_GLOBAL_ORDINAL + fixtures.len(), 62);
        assert_eq!(fixtures.iter().map(|item| item.copy).sum::<usize>(), 15);
    }

    #[test]
    fn degree14_mask_functional_is_deterministic_and_detects_mutation() {
        let mask = (0..14).fold(0_u32, |value, bit| value | (1_u32 << bit));
        let projected = degree14_to_degree13_mask(mask);
        assert_eq!(projected.count_ones(), 13);
        assert_eq!(projected, mask ^ (1_u32 << 13));
        let mutated = mask ^ (1_u32 << 12) ^ (1_u32 << 20);
        assert_ne!(degree14_to_degree13_mask(mutated), projected);
    }

    #[test]
    fn physical_stream_hash_detects_coefficient_mutation() {
        let mut terms = vec![SecondMomentumFxColumnTerm {
            coefficient_column: 53,
            gauge_channel: SecondMomentumGaugeChannel::new(0).unwrap(),
            gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
            source_momentum: DegreeTwoMomentumMonomial::from_pair(0, 0).unwrap(),
            parameter_component: 0,
            target_coordinate: 0,
            spinor_derivative_mask: (1_u32 << 13) - 1,
            sector: SecondMomentumFxSector::X2,
            coefficient: ExactGaussian::one(),
        }];
        let hash = crate::eleven_dimensional_second_momentum_fx::
            canonical_second_momentum_fx_stream_sha256(&terms);
        terms[0].coefficient.real += Ratio::from_integer(BigInt::from(1));
        assert_ne!(
            hash,
            crate::eleven_dimensional_second_momentum_fx::
                canonical_second_momentum_fx_stream_sha256(&terms)
        );
    }

    #[test]
    #[ignore = "materializes all nine exact level-12 maps and writes the physical tranche"]
    fn write_complete_physical_20001_fx_tranche() {
        let report = write_artifact(Path::new(
            "results/adynkra_11d_second_momentum_20001_fx.json",
        ))
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.observed_nonzero_columns.len(), 9);
    }
}
