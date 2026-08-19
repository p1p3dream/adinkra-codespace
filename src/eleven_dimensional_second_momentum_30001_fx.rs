//! Exact physical `F_X` screen for the fifteen `(30001)` second-momentum maps.
//!
//! The global 77-column layout is ordered by intermediate channel.  The first
//! 62 columns belong to `(00001)`, `(01001)`, the two `(10001)` paths,
//! `(11001)`, and `(20001)`.  This module owns columns 62 through 76, the
//! fifteen exact `(30001)` source maps.  No placeholder terms are emitted for
//! the other 62 columns.
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

const SCHEMA_VERSION: &str = "adynkra-11d-second-momentum-30001-fx-v1";
const MAP_CHECKPOINT_DIRECTORY: &str = "results/adynkra_11d_second_momentum_30001_checkpoints";
const MAP_AGGREGATE_ARTIFACT: &str = "results/adynkra_11d_second_momentum_30001_maps.json";
const FIRST_GLOBAL_ORDINAL: usize = 62;
const TRANCHE_COLUMNS: usize = 15;

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
        fixture!("40000", 1, "level12_40000_highest_weight_kernel.i16le"),
        fixture!("20100", 1, "level12_20100_highest_weight_kernel_1.i16le"),
        fixture!("20100", 2, "level12_20100_highest_weight_kernel_2.i16le"),
        fixture!("31000", 1, "level12_31000_highest_weight_kernel.i16le"),
        fixture!("20010", 1, "level12_20010_highest_weight_kernel_1.i16le"),
        fixture!("20010", 2, "level12_20010_highest_weight_kernel_2.i16le"),
        fixture!("20010", 3, "level12_20010_highest_weight_kernel_3.i16le"),
        fixture!("20002", 1, "level12_20002_highest_weight_kernel_1.i16le"),
        fixture!("20002", 2, "level12_20002_highest_weight_kernel_2.i16le"),
        fixture!("30100", 1, "level12_30100_highest_weight_kernel.i16le"),
        fixture!("30010", 1, "level12_30010_highest_weight_kernel_1.i16le"),
        fixture!("30010", 2, "level12_30010_highest_weight_kernel_2.i16le"),
        fixture!("30002", 1, "level12_30002_highest_weight_kernel_1.i16le"),
        fixture!("30002", 2, "level12_30002_highest_weight_kernel_2.i16le"),
        fixture!("30002", 3, "level12_30002_highest_weight_kernel_3.i16le"),
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SecondMomentum30001GpuColumnPreflight {
    pub tranche: String,
    pub local_column_ordinal: usize,
    pub global_column_ordinal: usize,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub abstract_certificate_sha256: String,
    pub source_map_sha256: String,
    pub reciprocal_map_sha256: String,
    pub pbw_plan_sha256: String,
    pub pbw_word_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SecondMomentum30001GpuColumnEvent {
    WordLoweringStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    WordStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    Term {
        requested_word_ordinal: usize,
        term: crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm,
    },
    WordEnd {
        requested_word_ordinal: usize,
        raw_terms_emitted: u64,
    },
}

struct PreparedGpuColumn {
    fixture: SourceFixture,
    abstract_certificate: crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    source_fixture_sha256: String,
    coupled_map_sha256: String,
    words: Vec<Vec<u8>>,
    reciprocal_by_word: Vec<Vec<([usize; 2], i64)>>,
    raising_residuals: [usize; 5],
    preflight: SecondMomentum30001GpuColumnPreflight,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentum30001ColumnSpec {
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
pub struct SecondMomentum30001ColumnAccounting {
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
pub struct SecondMomentum30001FxReport {
    pub schema_version: String,
    pub role: String,
    pub global_77_column_ordinals: Vec<usize>,
    pub columns: Vec<SecondMomentum30001ColumnSpec>,
    pub highest_target_basis_ordinal: usize,
    pub highest_target_pbw_word: Vec<u8>,
    pub selected_parameter_components: Vec<usize>,
    pub selected_gauge_form_degrees: Vec<usize>,
    pub p2_d13_wedge_slice_complete: bool,
    pub p3_d11_contraction_terms_emitted: u64,
    pub actual_convention_fixed_fx_projection_used: bool,
    pub degree14_to_degree13_functional: String,
    pub source_map_aggregate_sha256: String,
    pub recoupling_report_sha256: String,
    pub recoupling_30001_certificate_sha256: String,
    pub fixture_manifest_sha256: String,
    pub component_cg_manifest_sha256: String,
    pub canonical_physical_fx_functional_rows_sha256: String,
    pub observed_physical_fx_terms: u64,
    pub observed_nonzero_columns: Vec<usize>,
    pub column_accounting: Vec<SecondMomentum30001ColumnAccounting>,
    pub maximum_observed_process_rss_bytes: Option<u64>,
    pub elapsed_milliseconds: u128,
    pub harness_checkpoint: SecondMomentumFxStreamingCheckpoint,
    pub x2_rank_lower_bound_on_15_column_tranche: usize,
    pub x5_rank_lower_bound_on_15_column_tranche: usize,
    pub joint_rank_lower_bound_on_15_column_tranche: usize,
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
    sha256(&serde_json::to_vec(value).expect("serialize deterministic p2 (30001) payload"))
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
    Path::new(MAP_CHECKPOINT_DIRECTORY).join(format!("abstract_30001_from_{source}.json"))
}

fn embedded_checkpoint_path(source: &str, copy: usize) -> std::path::PathBuf {
    Path::new(MAP_CHECKPOINT_DIRECTORY)
        .join(format!("embedded_30001_from_{source}_copy{copy}.json"))
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
/// define a highest target state inside `Sym^2(V) tensor (30001)`.  Replacing
/// every abstract `(30001)` descendant by the exact source-map descendant must
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

fn pbw_plan_sha256(words: &[Vec<u8>]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-second-momentum-30001-pbw-plan-v1\0");
    hash.update((words.len() as u64).to_le_bytes());
    for (ordinal, word) in words.iter().enumerate() {
        hash.update((ordinal as u64).to_le_bytes());
        hash.update((word.len() as u64).to_le_bytes());
        hash.update(word);
    }
    format!("{:x}", hash.finalize())
}

fn validate_map_checkpoint(
    fixture: SourceFixture,
    abstract_checkpoint: &crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001AbstractCheckpoint,
    embedded_checkpoint: &crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001EmbeddedCheckpoint,
) -> io::Result<()> {
    let certificate_sha256 = sha256_json(&abstract_checkpoint.certificate);
    if !abstract_checkpoint.passed
        || abstract_checkpoint.source_dynkin_label != fixture.dynkin_label
        || abstract_checkpoint.target_dynkin_label != "30001"
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
            "invalid or mismatched exact (30001) map checkpoint",
        ));
    }
    Ok(())
}

fn prepare_gpu_column(local_ordinal: usize) -> io::Result<PreparedGpuColumn> {
    let fixtures = source_fixtures();
    let fixture = *fixtures.get(local_ordinal).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "(30001) GPU column ordinal must lie in 0..15",
        )
    })?;
    let recoupling =
        crate::eleven_dimensional_second_momentum_remaining_recouplings::verify_cached();
    let reciprocal = recoupling
        .channels
        .iter()
        .find(|channel| channel.intermediate_dynkin_label == "30001")
        .ok_or_else(|| io::Error::other("missing exact (30001) reciprocal certificate"))?;
    if reciprocal.reciprocal_raising_residual_terms_by_simple_root != [0; 5]
        || !reciprocal.exact_chevalley_equivariance_verified
        || !reciprocal.passed
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "uncertified exact (30001) reciprocal highest-weight map",
        ));
    }

    let abstract_checkpoint = read_json::<
        crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001AbstractCheckpoint,
    >(&abstract_checkpoint_path(fixture.dynkin_label))?;
    let embedded_checkpoint = read_json::<
        crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001EmbeddedCheckpoint,
    >(&embedded_checkpoint_path(
        fixture.dynkin_label,
        fixture.copy,
    ))?;
    validate_map_checkpoint(fixture, &abstract_checkpoint, &embedded_checkpoint)?;

    let words = requested_words(reciprocal);
    if words.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty exact (30001) reciprocal PBW plan",
        ));
    }
    let word_ordinals = words
        .iter()
        .enumerate()
        .map(|(ordinal, word)| (word.clone(), ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut reciprocal_by_word = vec![Vec::new(); words.len()];
    for term in &reciprocal.reciprocal_terms {
        let word_ordinal = word_ordinals[&term.intermediate_pbw_word_simple_roots];
        reciprocal_by_word[word_ordinal].push((term.momentum_pair, term.primitive_coefficient));
    }
    if reciprocal_by_word.iter().any(Vec::is_empty) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incomplete exact (30001) reciprocal PBW plan",
        ));
    }

    let source_fixture_sha256 = embedded_checkpoint.source_fixture_sha256.clone();
    let coupled_map_sha256 = embedded_checkpoint.coupled_map_sha256.clone();
    let preflight = SecondMomentum30001GpuColumnPreflight {
        tranche: "30001".to_string(),
        local_column_ordinal: local_ordinal,
        global_column_ordinal: FIRST_GLOBAL_ORDINAL + local_ordinal,
        source_dynkin_label: fixture.dynkin_label.to_string(),
        source_copy: fixture.copy,
        source_fixture: fixture.artifact.to_string(),
        source_fixture_sha256: source_fixture_sha256.clone(),
        abstract_certificate_sha256: abstract_checkpoint.certificate_sha256.clone(),
        source_map_sha256: coupled_map_sha256.clone(),
        reciprocal_map_sha256: reciprocal.certificate_sha256.clone(),
        pbw_plan_sha256: pbw_plan_sha256(&words),
        pbw_word_count: words.len(),
    };
    Ok(PreparedGpuColumn {
        fixture,
        abstract_certificate: abstract_checkpoint.certificate,
        source_fixture_sha256,
        coupled_map_sha256,
        words,
        reciprocal_by_word,
        raising_residuals: reciprocal.reciprocal_raising_residual_terms_by_simple_root,
        preflight,
    })
}

pub(crate) fn gpu_column_preflight(
    local_ordinal: usize,
) -> io::Result<SecondMomentum30001GpuColumnPreflight> {
    Ok(prepare_gpu_column(local_ordinal)?.preflight)
}

type RecoupledState = BTreeMap<([usize; 2], usize, u32), i128>;

fn build_column_recoupled(
    fixture: SourceFixture,
    reciprocal: &crate::eleven_dimensional_second_momentum_remaining_recouplings::RemainingRecouplingCertificate,
) -> io::Result<(
    RecoupledState,
    crate::eleven_dimensional_level16_couplings::SecondMomentum30001DescendantAccounting,
    [usize; 5],
)> {
    let abstract_checkpoint = read_json::<
        crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001AbstractCheckpoint,
    >(&abstract_checkpoint_path(fixture.dynkin_label))?;
    let embedded_checkpoint = read_json::<
        crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001EmbeddedCheckpoint,
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
    let mut reciprocal_by_word = vec![Vec::new(); words.len()];
    for term in &reciprocal.reciprocal_terms {
        let word_ordinal = word_ordinals[&term.intermediate_pbw_word_simple_roots];
        reciprocal_by_word[word_ordinal].push((term.momentum_pair, term.primitive_coefficient));
    }
    let mut observed_by_word = vec![false; words.len()];
    let mut recoupled = RecoupledState::new();
    let accounting = crate::eleven_dimensional_level16_couplings::
        visit_second_momentum_30001_descendant_components(
            &abstract_checkpoint.certificate,
            fixture.copy,
            fixture.artifact,
            2,
            fixture.bytes,
            &embedded_checkpoint.source_fixture_sha256,
            &embedded_checkpoint.coupled_map_sha256,
            &words,
            |entry| {
                observed_by_word[entry.requested_word_ordinal] = true;
                for &(momentum_pair, primitive_coefficient) in
                    &reciprocal_by_word[entry.requested_word_ordinal]
                {
                    let value = i128::from(entry.coefficient)
                        .checked_mul(i128::from(primitive_coefficient))
                        .ok_or_else(|| io::Error::other("p2 recoupling coefficient overflow"))?;
                    let output = recoupled
                        .entry((
                            momentum_pair,
                            entry.free_spinor_weight_index,
                            entry.exterior_mask,
                        ))
                        .or_insert(0);
                    *output = output
                        .checked_add(value)
                        .ok_or_else(|| io::Error::other("p2 recoupling accumulator overflow"))?;
                }
                Ok(())
            },
        )?;
    if observed_by_word.iter().any(|observed| !observed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "requested (30001) descendant materialized no exact components",
        ));
    }
    recoupled.retain(|_, coefficient| *coefficient != 0);
    if recoupled.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact (30001) momentum recoupling vanished",
        ));
    }
    let raising_residuals = composed_highest_raising_residuals(&recoupled)?;
    if raising_residuals != [0; 5] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("substituted (30001) p2 target state is not highest: {raising_residuals:?}"),
        ));
    }
    Ok((recoupled, accounting, raising_residuals))
}

fn build_column_gauge_residuals(
    fixture: SourceFixture,
    reciprocal: &crate::eleven_dimensional_second_momentum_remaining_recouplings::RemainingRecouplingCertificate,
    highest_target: &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
) -> io::Result<(
    Vec<BTreeMap<GaugeResidualKey, ExactGaussian>>,
    crate::eleven_dimensional_level16_couplings::SecondMomentum30001DescendantAccounting,
    [usize; 5],
)> {
    let (recoupled, accounting, raising_residuals) = build_column_recoupled(fixture, reciprocal)?;

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

pub(crate) fn visit_gpu_column_contributions<F>(
    local_ordinal: usize,
    mut visit: F,
) -> io::Result<crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput>
where
    F: FnMut(crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm) -> io::Result<()>,
{
    let prepared = prepare_gpu_column(local_ordinal)?;
    visit_prepared_gpu_column_events_from(prepared, 0, |event| match event {
        SecondMomentum30001GpuColumnEvent::Term { term, .. } => visit(term),
        SecondMomentum30001GpuColumnEvent::WordLoweringStart { .. }
        | SecondMomentum30001GpuColumnEvent::WordStart { .. }
        | SecondMomentum30001GpuColumnEvent::WordEnd { .. } => Ok(()),
    })
}

pub(crate) fn visit_gpu_column_contribution_events_from<F>(
    expected_preflight: &SecondMomentum30001GpuColumnPreflight,
    start_word_ordinal: usize,
    visit: F,
) -> io::Result<crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput>
where
    F: FnMut(SecondMomentum30001GpuColumnEvent) -> io::Result<()>,
{
    if start_word_ordinal > expected_preflight.pbw_word_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "(30001) start word ordinal exceeds the preflight PBW plan",
        ));
    }
    let prepared = prepare_gpu_column(expected_preflight.local_column_ordinal)?;
    if prepared.preflight != *expected_preflight {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "(30001) GPU column preflight identity changed before streaming",
        ));
    }
    visit_prepared_gpu_column_events_from(prepared, start_word_ordinal, visit)
}

/// Stream one verified column through an opaque lowering backend. The
/// canonical highest state is uploaded once, prefix handles stay backend
/// resident, and only terminal handles are presented to `download_terms`.
pub(crate) fn visit_gpu_column_contribution_events_from_handles<H, U, L, D, F>(
    expected_preflight: &SecondMomentum30001GpuColumnPreflight,
    start_word_ordinal: usize,
    upload_highest: U,
    lower_word: L,
    mut download_terms: D,
    mut visit: F,
) -> io::Result<crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput>
where
    U: FnOnce(
        &crate::eleven_dimensional_level16_couplings::CanonicalSparseHighest64,
    ) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    D: FnMut(&H, &mut dyn FnMut(u64, i64) -> io::Result<()>) -> io::Result<u64>,
    F: FnMut(SecondMomentum30001GpuColumnEvent) -> io::Result<()>,
{
    if start_word_ordinal > expected_preflight.pbw_word_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "(30001) start word ordinal exceeds the preflight PBW plan",
        ));
    }
    let prepared = prepare_gpu_column(expected_preflight.local_column_ordinal)?;
    if prepared.preflight != *expected_preflight {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "(30001) GPU column preflight identity changed before opaque streaming",
        ));
    }

    let mut observed_by_word = vec![false; prepared.words.len()];
    let mut emitted_terms = 0_u64;
    let mut current_word_terms = 0_u64;
    let mut current_word_components = 0_u64;
    let accounting = crate::eleven_dimensional_level16_couplings::
        visit_second_momentum_30001_descendant_handles_from(
            &prepared.abstract_certificate,
            prepared.fixture.copy,
            prepared.fixture.artifact,
            2,
            prepared.fixture.bytes,
            &prepared.source_fixture_sha256,
            &prepared.coupled_map_sha256,
            &prepared.words,
            start_word_ordinal,
            upload_highest,
            lower_word,
            |event| match event {
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::
                    WordLoweringStart { ordinal, pbw_word } => {
                        visit(SecondMomentum30001GpuColumnEvent::WordLoweringStart {
                            requested_word_ordinal: ordinal,
                            pbw_word_simple_roots: pbw_word.to_vec(),
                        })?;
                        Ok(0)
                    }
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::WordStart {
                    ordinal,
                    pbw_word,
                } => {
                    current_word_terms = 0;
                    current_word_components = 0;
                    visit(SecondMomentum30001GpuColumnEvent::WordStart {
                        requested_word_ordinal: ordinal,
                        pbw_word_simple_roots: pbw_word.to_vec(),
                    })?;
                    Ok(0)
                }
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::State {
                    ordinal,
                    state,
                } => {
                    let mut previous_key = None;
                    let reported = download_terms(state, &mut |key, descendant_coefficient| {
                        let free_spinor_weight_index = usize::try_from(key >> 32).map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "downloaded free-spinor key exceeds host range",
                            )
                        })?;
                        let exterior_mask = key as u32;
                        if free_spinor_weight_index >= 32
                            || exterior_mask.count_ones() != 12
                            || descendant_coefficient == 0
                            || previous_key.is_some_and(|previous| previous >= key)
                        {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "downloaded (30001) terminal state is not canonical",
                            ));
                        }
                        previous_key = Some(key);
                        current_word_components = current_word_components
                            .checked_add(1)
                            .ok_or_else(|| io::Error::other("descendant component count overflow"))?;
                        for &(momentum_pair, primitive_coefficient) in
                            &prepared.reciprocal_by_word[ordinal]
                        {
                            let coefficient = i128::from(descendant_coefficient)
                                .checked_mul(i128::from(primitive_coefficient))
                                .ok_or_else(|| {
                                    io::Error::other("p2 recoupling coefficient overflow")
                                })?;
                            if coefficient == 0 {
                                continue;
                            }
                            let term = crate::eleven_dimensional_second_momentum_gpu::
                                RecoupledSourceTerm {
                                    momentum_pair: [
                                        u8::try_from(momentum_pair[0]).map_err(|_| {
                                            io::Error::other(
                                                "momentum index exceeds packed GPU range",
                                            )
                                        })?,
                                        u8::try_from(momentum_pair[1]).map_err(|_| {
                                            io::Error::other(
                                                "momentum index exceeds packed GPU range",
                                            )
                                        })?,
                                    ],
                                    free_spinor: u8::try_from(free_spinor_weight_index).map_err(
                                        |_| io::Error::other("spinor index exceeds packed GPU range"),
                                    )?,
                                    exterior_mask,
                                    coefficient,
                                };
                            visit(SecondMomentum30001GpuColumnEvent::Term {
                                requested_word_ordinal: ordinal,
                                term,
                            })?;
                            current_word_terms = current_word_terms.checked_add(1).ok_or_else(|| {
                                io::Error::other("GPU word contribution count overflow")
                            })?;
                            emitted_terms = emitted_terms.checked_add(1).ok_or_else(|| {
                                io::Error::other("GPU contribution count overflow")
                            })?;
                        }
                        Ok(())
                    })?;
                    if reported != current_word_components {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "downloaded (30001) terminal count disagrees with emitted terms",
                        ));
                    }
                    Ok(reported)
                }
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::WordEnd {
                    ordinal,
                } => {
                    if current_word_components == 0 || current_word_terms == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "requested (30001) GPU word is empty",
                        ));
                    }
                    observed_by_word[ordinal] = true;
                    visit(SecondMomentum30001GpuColumnEvent::WordEnd {
                        requested_word_ordinal: ordinal,
                        raw_terms_emitted: current_word_terms,
                    })?;
                    Ok(0)
                }
            },
        )?;
    if accounting.emitted_nonzero_components == 0 && start_word_ordinal < prepared.words.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "requested (30001) opaque descendant stream is empty",
        ));
    }
    if observed_by_word
        .iter()
        .skip(start_word_ordinal)
        .any(|observed| !observed)
        || (start_word_ordinal < prepared.words.len() && emitted_terms == 0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "requested (30001) opaque GPU descendant stream is incomplete or empty",
        ));
    }
    Ok(
        crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput {
            global_ordinal: prepared.preflight.global_column_ordinal,
            source_label: prepared.preflight.source_dynkin_label,
            source_copy: prepared.preflight.source_copy,
            terms: Vec::new(),
            raising_residuals: prepared.raising_residuals,
        },
    )
}

fn visit_prepared_gpu_column_events_from<F>(
    prepared: PreparedGpuColumn,
    start_word_ordinal: usize,
    mut visit: F,
) -> io::Result<crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput>
where
    F: FnMut(SecondMomentum30001GpuColumnEvent) -> io::Result<()>,
{
    if start_word_ordinal > prepared.words.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "(30001) start word ordinal exceeds the PBW plan",
        ));
    }
    let mut observed_by_word = vec![false; prepared.words.len()];
    let mut emitted_terms = 0_u64;
    let mut current_word_terms = 0_u64;
    crate::eleven_dimensional_level16_couplings::
        visit_second_momentum_30001_descendant_events_from(
        &prepared.abstract_certificate,
        prepared.fixture.copy,
        prepared.fixture.artifact,
        2,
        prepared.fixture.bytes,
        &prepared.source_fixture_sha256,
        &prepared.coupled_map_sha256,
        &prepared.words,
        start_word_ordinal,
        |event| {
            match event {
                crate::eleven_dimensional_level16_couplings::
                    SecondMomentum30001DescendantEvent::WordLoweringStart {
                        requested_word_ordinal,
                        pbw_word_simple_roots,
                    } => {
                        visit(SecondMomentum30001GpuColumnEvent::WordLoweringStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        })?;
                    }
                crate::eleven_dimensional_level16_couplings::
                    SecondMomentum30001DescendantEvent::WordStart {
                        requested_word_ordinal,
                        pbw_word_simple_roots,
                    } => {
                        current_word_terms = 0;
                        visit(SecondMomentum30001GpuColumnEvent::WordStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        })?;
                    }
                crate::eleven_dimensional_level16_couplings::
                    SecondMomentum30001DescendantEvent::Component(entry) => {
                    for &(momentum_pair, primitive_coefficient) in
                        &prepared.reciprocal_by_word[entry.requested_word_ordinal]
                    {
                        let coefficient = i128::from(entry.coefficient)
                            .checked_mul(i128::from(primitive_coefficient))
                            .ok_or_else(|| io::Error::other("p2 recoupling coefficient overflow"))?;
                        if coefficient == 0 {
                            continue;
                        }
                        let term = crate::eleven_dimensional_second_momentum_gpu::
                            RecoupledSourceTerm {
                            momentum_pair: [
                                u8::try_from(momentum_pair[0]).map_err(|_| {
                                    io::Error::other("momentum index exceeds packed GPU range")
                                })?,
                                u8::try_from(momentum_pair[1]).map_err(|_| {
                                    io::Error::other("momentum index exceeds packed GPU range")
                                })?,
                            ],
                            free_spinor: u8::try_from(entry.free_spinor_weight_index).map_err(
                                |_| io::Error::other("spinor index exceeds packed GPU range"),
                            )?,
                            exterior_mask: entry.exterior_mask,
                            coefficient,
                        };
                        visit(SecondMomentum30001GpuColumnEvent::Term {
                            requested_word_ordinal: entry.requested_word_ordinal,
                            term,
                        })?;
                        current_word_terms = current_word_terms.checked_add(1).ok_or_else(|| {
                            io::Error::other("GPU word contribution count overflow")
                        })?;
                        emitted_terms = emitted_terms.checked_add(1).ok_or_else(|| {
                            io::Error::other("GPU contribution count overflow")
                        })?;
                    }
                }
                crate::eleven_dimensional_level16_couplings::
                    SecondMomentum30001DescendantEvent::WordEnd {
                        requested_word_ordinal,
                        emitted_nonzero_components,
                    } => {
                        if emitted_nonzero_components == 0 || current_word_terms == 0 {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "requested (30001) GPU word is empty",
                            ));
                        }
                        observed_by_word[requested_word_ordinal] = true;
                        visit(SecondMomentum30001GpuColumnEvent::WordEnd {
                            requested_word_ordinal,
                            raw_terms_emitted: current_word_terms,
                        })?;
                    }
            }
            Ok(())
        },
    )?;
    if observed_by_word
        .iter()
        .skip(start_word_ordinal)
        .any(|observed| !observed)
        || (start_word_ordinal < prepared.words.len() && emitted_terms == 0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "requested (30001) GPU descendant stream is incomplete or empty",
        ));
    }
    Ok(
        crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput {
            global_ordinal: prepared.preflight.global_column_ordinal,
            source_label: prepared.preflight.source_dynkin_label,
            source_copy: prepared.preflight.source_copy,
            terms: Vec::new(),
            raising_residuals: prepared.raising_residuals,
        },
    )
}

pub(crate) fn build_gpu_column_input(
    local_ordinal: usize,
) -> io::Result<crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput> {
    let mut terms = Vec::new();
    let mut column = visit_gpu_column_contributions(local_ordinal, |term| {
        terms.push(term);
        Ok(())
    })?;
    column.terms = terms;
    Ok(column)
}

pub fn construct() -> io::Result<SecondMomentum30001FxReport> {
    let started = Instant::now();
    let map_aggregate_bytes = fs::read(MAP_AGGREGATE_ARTIFACT)?;
    let map_aggregate: crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001Report =
        serde_json::from_slice(&map_aggregate_bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !map_aggregate.passed
        || !map_aggregate.source_to_30001_component_maps_complete
        || map_aggregate.exact_embedded_component_maps != TRANCHE_COLUMNS
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incomplete exact (30001) source-map aggregate",
        ));
    }
    let source_map_aggregate_sha256 = sha256(&map_aggregate_bytes);

    let recoupling_report =
        crate::eleven_dimensional_second_momentum_remaining_recouplings::verify();
    let reciprocal = recoupling_report
        .channels
        .iter()
        .find(|channel| channel.intermediate_dynkin_label == "30001")
        .ok_or_else(|| io::Error::other("missing exact (30001) momentum recoupling"))?;
    if !recoupling_report.passed
        || !reciprocal.passed
        || reciprocal.reciprocal_terms.is_empty()
        || !reciprocal.exact_chevalley_equivariance_verified
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "uncertified (30001) momentum recoupling",
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
        campaign_id: "p2-d12-30001-highest-target-parameter-zero-v1".to_string(),
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
            crate::eleven_dimensional_second_momentum_30001_maps::SecondMomentum30001EmbeddedCheckpoint,
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
        columns.push(SecondMomentum30001ColumnSpec {
            global_ordinal,
            source_dynkin_label: fixture.dynkin_label.to_string(),
            source_copy: fixture.copy,
            intermediate_dynkin_label: "30001".to_string(),
            symmetric_momentum_path: 1,
            source_fixture: fixture.artifact.to_string(),
            source_fixture_sha256: sha256(fixture.bytes),
            coupled_map_sha256: embedded_checkpoint.coupled_map_sha256,
        });
        column_accounting.push(SecondMomentum30001ColumnAccounting {
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
            "second-momentum (30001) F_X: completed column {global_ordinal} ({}/{TRANCHE_COLUMNS}), gauge residuals {}, projected terms {}, elapsed {:.1}s",
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
            "not every exact (30001) column reached the physical F_X slice",
        ));
    }
    let x2_rank = harness_checkpoint.report.global_x2.rank;
    let x5_rank = harness_checkpoint.report.global_x5.rank;
    let joint_rank = harness_checkpoint.report.global_joint.rank;
    if x2_rank > TRANCHE_COLUMNS || x5_rank > TRANCHE_COLUMNS || joint_rank > TRANCHE_COLUMNS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rank exceeds the fifteen physically materialized columns",
        ));
    }
    let declared_slice_tranche_no_go_certified = joint_rank == TRANCHE_COLUMNS;
    let report = SecondMomentum30001FxReport {
        schema_version: SCHEMA_VERSION.to_string(),
        role: "exact physical F_X rank-lower-bound screen on the fifteen (30001) p2D12 columns"
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
        degree14_to_degree13_functional:
            "delete the greatest occupied spinor index and sum all collisions exactly"
                .to_string(),
        source_map_aggregate_sha256,
        recoupling_report_sha256: recoupling_report.report_sha256,
        recoupling_30001_certificate_sha256: reciprocal.certificate_sha256.clone(),
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
        x2_rank_lower_bound_on_15_column_tranche: x2_rank,
        x5_rank_lower_bound_on_15_column_tranche: x5_rank,
        joint_rank_lower_bound_on_15_column_tranche: joint_rank,
        joint_tranche_nullity_upper_bound: TRANCHE_COLUMNS - joint_rank,
        declared_slice_tranche_no_go_certified,
        all_77_columns_materialized: false,
        full_parameter_projection_complete: false,
        full_target_projection_complete: false,
        full_f_a_g_p_established: false,
        passed: true,
        boundary: "This is an exact rank lower bound for global columns 62 through 76 on the unique highest-target, parameter-zero, all-six-channel p2D13 wedge slice. Full rank fifteen proves a no-go for only this fifteen-column tranche. The other 62 columns are absent rather than synthesized. The p3D11 branch, complete parameter and target projections, J/W, the generic momentum tower, complete F, and full F A G_p remain open."
            .to_string(),
    };
    validate(&report)?;
    Ok(report)
}

pub fn validate(report: &SecondMomentum30001FxReport) -> io::Result<()> {
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
                || column.intermediate_dynkin_label != "30001"
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
        || report.x2_rank_lower_bound_on_15_column_tranche
            != report.harness_checkpoint.report.global_x2.rank
        || report.x5_rank_lower_bound_on_15_column_tranche
            != report.harness_checkpoint.report.global_x5.rank
        || report.joint_rank_lower_bound_on_15_column_tranche != joint_rank
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
            "second-momentum (30001) physical F_X report invariant failed",
        ));
    }
    Ok(())
}

pub fn write_artifact(path: &Path) -> io::Result<SecondMomentum30001FxReport> {
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

    fn test_provenance(campaign: &str) -> SecondMomentumFxProvenance {
        SecondMomentumFxProvenance {
            source_kind: SecondMomentumFxSourceKind::PhysicalRecoupledStream,
            campaign_id: campaign.to_string(),
            representation_inventory_sha256:
                crate::eleven_dimensional_second_momentum_fx::SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256
                    .to_string(),
            level12_fixture_manifest_sha256: "11".repeat(32),
            component_cg_manifest_sha256: "22".repeat(32),
            coefficient_layout_sha256:
                crate::eleven_dimensional_second_momentum_fx::second_momentum_fx_coefficient_layout_sha256(),
            expected_canonical_stream_sha256: "00".repeat(32),
        }
    }

    fn test_coverage() -> SecondMomentumFxCoverage {
        SecondMomentumFxCoverage::default()
    }

    #[test]
    fn exact_global_layout_is_the_final_fifteen_columns() {
        let fixtures = source_fixtures();
        assert_eq!(fixtures.len(), 15);
        assert_eq!(FIRST_GLOBAL_ORDINAL, 3 + 12 + 8 + 30 + 9);
        assert_eq!(FIRST_GLOBAL_ORDINAL + fixtures.len(), 77);
        assert_eq!(fixtures.iter().map(|item| item.copy).sum::<usize>(), 24);
    }

    #[test]
    fn gpu_pbw_plan_digest_binds_order_and_word_boundaries() {
        let words = vec![vec![1, 2], vec![3], vec![4, 5]];
        let mut reordered = words.clone();
        reordered.swap(0, 1);
        assert_eq!(pbw_plan_sha256(&words), pbw_plan_sha256(&words));
        assert_ne!(pbw_plan_sha256(&words), pbw_plan_sha256(&reordered));
        assert_ne!(
            pbw_plan_sha256(&words),
            pbw_plan_sha256(&[vec![1], vec![2, 3], vec![4, 5]])
        );
    }

    #[test]
    fn gpu_resume_rejects_start_beyond_preflight_before_streaming() {
        let preflight = SecondMomentum30001GpuColumnPreflight {
            tranche: "30001".to_string(),
            local_column_ordinal: 0,
            global_column_ordinal: FIRST_GLOBAL_ORDINAL,
            source_dynkin_label: String::new(),
            source_copy: 0,
            source_fixture: String::new(),
            source_fixture_sha256: String::new(),
            abstract_certificate_sha256: String::new(),
            source_map_sha256: String::new(),
            reciprocal_map_sha256: String::new(),
            pbw_plan_sha256: String::new(),
            pbw_word_count: 1,
        };
        let error = visit_gpu_column_contribution_events_from(&preflight, 2, |_| {
            panic!("invalid resume ordinal must not stream events")
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let opaque_error = visit_gpu_column_contribution_events_from_handles(
            &preflight,
            2,
            |_| -> io::Result<()> { panic!("invalid resume must not upload") },
            |_, _, _| -> io::Result<()> { panic!("invalid resume must not lower") },
            |_, _| -> io::Result<u64> { panic!("invalid resume must not download") },
            |_| -> io::Result<()> { panic!("invalid resume must not stream events") },
        )
        .unwrap_err();
        assert_eq!(opaque_error.kind(), io::ErrorKind::InvalidInput);
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
            coefficient_column: 62,
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
    #[ignore = "materializes one real 30002 source map and benchmarks streaming physical F_X"]
    fn benchmark_real_30002_copy1_streaming_column_with_parity() {
        use crate::eleven_dimensional_second_momentum_fx::SecondMomentumFxTermSource;

        #[derive(Clone)]
        struct Retained {
            terms: Vec<SecondMomentumFxColumnTerm>,
            provenance: SecondMomentumFxProvenance,
        }
        impl SecondMomentumFxTermSource for Retained {
            fn provenance(&self) -> SecondMomentumFxProvenance {
                self.provenance.clone()
            }
            fn coverage(&self) -> SecondMomentumFxCoverage {
                test_coverage()
            }
            fn visit_terms(
                &self,
                visitor: &mut dyn FnMut(SecondMomentumFxColumnTerm),
            ) -> Result<(), String> {
                self.terms.iter().cloned().for_each(visitor);
                Ok(())
            }
        }

        let started = Instant::now();
        let recoupling = crate::eleven_dimensional_second_momentum_remaining_recouplings::verify();
        let reciprocal = recoupling
            .channels
            .iter()
            .find(|channel| channel.intermediate_dynkin_label == "30001")
            .unwrap();
        let target = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .unwrap();
        let fixture = source_fixtures()[12];
        let global_ordinal = FIRST_GLOBAL_ORDINAL + 12;
        let (residuals, accounting, raising) =
            build_column_gauge_residuals(fixture, reciprocal, &target).unwrap();
        assert_eq!(raising, [0; 5]);
        assert!(accounting.checkpoint_hash_parity_verified);

        let mut subset: Vec<BTreeMap<GaugeResidualKey, ExactGaussian>> =
            (0..6).map(|_| BTreeMap::new()).collect();
        let (key, value) = residuals[0].iter().next().unwrap();
        subset[0].insert(key.clone(), value.clone());
        let mut cache = BTreeMap::new();
        let mut real_terms = Vec::new();
        visit_projected_fx_terms(global_ordinal, &subset, &mut cache, |term| {
            real_terms.push(term);
            Ok(())
        })
        .unwrap();
        assert!(!real_terms.is_empty());
        let retained_digest = crate::eleven_dimensional_second_momentum_fx::
            second_momentum_fx_functional_rows_sha256(&real_terms)
            .unwrap();
        let mut retained_provenance = test_provenance("real-30002-copy1-retained-parity");
        retained_provenance.expected_canonical_stream_sha256 =
            crate::eleven_dimensional_second_momentum_fx::
                canonical_second_momentum_fx_stream_sha256(&real_terms);
        let retained =
            crate::eleven_dimensional_second_momentum_fx::evaluate_second_momentum_fx_source(
                &Retained {
                    terms: real_terms.clone(),
                    provenance: retained_provenance,
                },
            )
            .unwrap();
        let mut parity_stream = SecondMomentumFxStreamingAccumulator::new(
            test_provenance("real-30002-copy1-streaming-parity"),
            test_coverage(),
        )
        .unwrap();
        for term in real_terms {
            parity_stream.push(term).unwrap();
        }
        let parity = parity_stream.finalize().unwrap();
        assert_eq!(
            parity.report.canonical_functional_rows_sha256,
            retained_digest
        );
        assert_eq!(parity.report.global_x2.rank, retained.global_x2.rank);
        assert_eq!(parity.report.global_x5.rank, retained.global_x5.rank);
        assert_eq!(parity.report.global_joint.rank, retained.global_joint.rank);

        let mut full_stream = SecondMomentumFxStreamingAccumulator::new(
            test_provenance("real-30002-copy1-full-streaming-benchmark"),
            test_coverage(),
        )
        .unwrap();
        let emitted =
            project_column_fx(global_ordinal, &residuals, &mut cache, &mut full_stream).unwrap();
        let full = full_stream.finalize().unwrap();
        assert!(emitted > 0);
        assert_eq!(full.report.observed_nonzero_columns, [global_ordinal]);
        assert!(full.report.global_joint.rank <= 1);
        eprintln!(
            "real 30002#1 streaming benchmark: emitted {emitted}, sparse rows {}, joint rank {}, elapsed {:.1}s, rss {:?}",
            full.report.sparse_functional_coefficients,
            full.report.global_joint.rank,
            started.elapsed().as_secs_f64(),
            observed_process_rss_bytes(),
        );
    }

    #[test]
    #[ignore = "materializes all fifteen exact level-12 maps and writes the physical tranche"]
    fn write_complete_physical_30001_fx_tranche() {
        let report = write_artifact(Path::new(
            "results/adynkra_11d_second_momentum_30001_fx.json",
        ))
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.observed_nonzero_columns.len(), 15);
    }
}
