//! Exact physical `F_X` diagnostic for the four certified `(10002)` source
//! copies times the trace and symmetric-traceless `(10001)` momentum paths.
//!
//! The calculation is deliberately a slice.  It fixes the degree-zero gauge
//! parameter component, the highest `(10001)` intermediate target state, the
//! `p^2 D^13` wedge branch, and one deterministic nonzero physical quotient
//! coordinate per derivative weight in each of `X_[2]` and `X_[5]`.  The
//! physical quotient coefficients come from `apply_polynomial_fx`; they are
//! not representation labels standing in for the physical curvature map.
//! The companion `p^3 D^11` branch and the other 73 second-momentum variables
//! stay separate and incomplete.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::Ratio;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_k_fag_solver::ExactGaussian;
use crate::eleven_dimensional_second_momentum_10001_maps::{
    EmbeddedSourceMap, SecondMomentum10001MapSpec, SecondMomentum10001Path,
    SecondMomentum10001SourceTerm,
};
use crate::eleven_dimensional_second_momentum_fx::{
    DegreeTwoMomentumMonomial, ExactRankSummary, SECOND_MOMENTUM_FX_BUCKETS_PER_SEED,
    SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS, SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256,
    SecondMomentumFxColumnTerm, SecondMomentumFxCoverage, SecondMomentumFxProjectedColumn,
    SecondMomentumFxProjectedEntry, SecondMomentumFxProvenance, SecondMomentumFxSector,
    SecondMomentumFxSourceKind, SecondMomentumFxStreamingAccumulator,
    SecondMomentumFxStreamingCheckpoint, SecondMomentumGaugeBranch, SecondMomentumGaugeChannel,
    canonical_second_momentum_fx_stream_sha256, second_momentum_fx_coefficient_layout_sha256,
    second_momentum_fx_functional_assignments,
};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const LOCAL_VARIABLES: usize = 4;
const REMAINING_VARIABLES: usize = 73;
const GAUGE_FORM_DEGREE: usize = 0;
const GAUGE_PARAMETER_COMPONENT: usize = 0;
const GLOBAL_10002_10001_ORDINALS: [usize; 4] = [19, 20, 21, 22];
const STT_COEFFICIENTS: [i64; 3] = [11, -1, -2];
const PROJECTED_SECTORS: usize = 2;

type SmallGaussian = Complex<Ratio<i64>>;
type LorentzVectorSpinor = Vec<Vec<SmallGaussian>>;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

struct FxProgressState {
    phase: AtomicUsize,
    processed: AtomicU64,
    total: AtomicU64,
    columns_completed: AtomicUsize,
    stop: AtomicBool,
    started: Instant,
}

#[derive(Clone)]
struct FxProgress {
    state: Arc<FxProgressState>,
}

struct FxProgressHeartbeat {
    state: Arc<FxProgressState>,
    worker: Option<thread::JoinHandle<()>>,
}

fn progress_phase_name(phase: usize) -> &'static str {
    match phase {
        0 => "validating_and_materializing_maps",
        1 => "building_recoupling_coefficients",
        2 => "building_physical_templates",
        3 => "building_exact_columns",
        4 => "folding_functional_rows",
        5 => "finalizing_certificate",
        6 => "complete",
        _ => "unknown",
    }
}

fn emit_progress(state: &FxProgressState, event: &str) {
    let processed = state.processed.load(Ordering::Relaxed);
    let total = state.total.load(Ordering::Relaxed);
    let elapsed = state.started.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };
    let eta_seconds = if processed > 0 && processed < total && rate > 0.0 {
        Some((total - processed) as f64 / rate)
    } else {
        None
    };
    eprintln!(
        "{}",
        serde_json::json!({
            "schema_version": "adynkra-11d-second-momentum-10001-progress-v1",
            "event": event,
            "phase": progress_phase_name(state.phase.load(Ordering::Relaxed)),
            "elapsed_seconds": elapsed,
            "processed": processed,
            "total": total,
            "percent": if total == 0 { 0.0 } else { 100.0 * processed as f64 / total as f64 },
            "rate_per_second": rate,
            "eta_seconds": eta_seconds,
            "columns_completed": state.columns_completed.load(Ordering::Relaxed),
            "columns_total": LOCAL_VARIABLES,
        })
    );
}

impl FxProgress {
    fn start() -> (Self, FxProgressHeartbeat) {
        let state = Arc::new(FxProgressState {
            phase: AtomicUsize::new(0),
            processed: AtomicU64::new(0),
            total: AtomicU64::new(0),
            columns_completed: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            started: Instant::now(),
        });
        let worker_state = Arc::clone(&state);
        let worker = thread::spawn(move || {
            while !worker_state.stop.load(Ordering::Acquire) {
                thread::park_timeout(PROGRESS_INTERVAL);
                if !worker_state.stop.load(Ordering::Acquire) {
                    emit_progress(&worker_state, "heartbeat");
                }
            }
        });
        emit_progress(&state, "run_start");
        (
            Self {
                state: Arc::clone(&state),
            },
            FxProgressHeartbeat {
                state,
                worker: Some(worker),
            },
        )
    }

    fn set_phase(&self, phase: usize, total: u64) {
        self.state.processed.store(0, Ordering::Relaxed);
        self.state.total.store(total, Ordering::Relaxed);
        self.state.phase.store(phase, Ordering::Release);
        emit_progress(&self.state, "phase_start");
    }

    fn add_processed(&self, count: u64) {
        self.state.processed.fetch_add(count, Ordering::Relaxed);
    }

    fn column_complete(&self, spec: &SecondMomentum10001FxVariableSpec, term_count: usize) {
        self.state.columns_completed.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "{}",
            serde_json::json!({
                "schema_version": "adynkra-11d-second-momentum-10001-progress-v1",
                "event": "column_complete",
                "elapsed_seconds": self.state.started.elapsed().as_secs_f64(),
                "local_ordinal": spec.local_ordinal,
                "global_ordinal": spec.global_77_ordinal,
                "source_copy": spec.source_copy,
                "momentum_path": spec.momentum_path,
                "output_terms": term_count,
            })
        );
    }

    fn finish(&self) {
        self.state.phase.store(6, Ordering::Release);
        emit_progress(&self.state, "run_complete");
    }
}

impl Drop for FxProgressHeartbeat {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct SecondMomentum10001FxVariableSpec {
    pub local_ordinal: usize,
    pub global_77_ordinal: usize,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub intermediate_dynkin_label: String,
    pub momentum_path: SecondMomentum10001Path,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PhysicalPivotCoordinate {
    pub derivative_spinor_weight_index: usize,
    pub x2_quotient_coordinate: Option<usize>,
    pub x5_quotient_coordinate: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SecondMomentum10001FxReport {
    pub schema_version: String,
    pub role: String,
    pub variable_specs: Vec<SecondMomentum10001FxVariableSpec>,
    pub local_variable_count: usize,
    pub remaining_second_momentum_variables: usize,
    pub gauge_form_degree: usize,
    pub gauge_parameter_component: usize,
    pub highest_intermediate_target_basis_ordinal: usize,
    pub target_slice: String,
    pub source_nonzero_terms_by_copy: Vec<usize>,
    pub trace_nonzero_momentum_pairs: usize,
    pub symmetric_traceless_nonzero_momentum_pairs: usize,
    pub input_target_gamma_trace_residual_entries: usize,
    pub recoupled_target_gamma_trace_residual_entries: usize,
    pub symmetric_traceless_momentum_trace_residual_entries: usize,
    pub physical_pivot_coordinates: Vec<PhysicalPivotCoordinate>,
    pub physical_adapter: String,
    pub emitted_physical_fx_terms: usize,
    pub canonical_functional_rows_sha256: String,
    pub source_fixture_manifest_sha256: String,
    pub component_map_manifest_sha256: String,
    pub x2_rank: ExactRankSummary,
    pub x5_rank: ExactRankSummary,
    pub joint_rank: ExactRankSummary,
    pub typed_harness_checkpoint: SecondMomentumFxStreamingCheckpoint,
    pub coefficient_mutation_detected: bool,
    pub global_ordinal_mutation_detected: bool,
    pub stt_mutation_detected: bool,
    pub p2_d13_declared_slice_computed: bool,
    pub p3_d11_contraction_computed: bool,
    pub all_77_variables_computed: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub full_physical_f_a_g_p_established: bool,
    pub passed: bool,
    pub boundary: String,
}

#[derive(Clone)]
struct PhysicalPivotTemplate {
    derivative_spinor_weight_index: usize,
    x2: Option<(usize, ExactGaussian)>,
    x5: Option<(usize, ExactGaussian)>,
}

fn small_zero() -> SmallGaussian {
    Complex::new(Ratio::from_integer(0), Ratio::from_integer(0))
}

fn small_integer(value: i64) -> SmallGaussian {
    Complex::new(Ratio::from_integer(value), Ratio::from_integer(0))
}

fn lorentz_metric(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("serialize second-momentum 10001 F_X pin"))
    )
}

fn target_state_lorentz_from_join(
    ordinal: usize,
    join: &crate::eleven_dimensional_b5_majorana_target_join::ExactB5MajoranaTargetJoin,
) -> LorentzVectorSpinor {
    let states = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let state = &states[ordinal];
    let mut output = vec![vec![small_zero(); SPINOR_DIMENSION]; VECTOR_DIMENSION];
    for term in &state.raw_terms {
        let coefficient = Ratio::new(term.numerator, term.denominator);
        for (vector, row) in output.iter_mut().enumerate() {
            let vector_factor = &join.upper_vector_to_lorentz[vector][term.vector_weight_index];
            if vector_factor == &small_zero() {
                continue;
            }
            for (spinor, value) in row.iter_mut().enumerate() {
                let spinor_factor = &join.spinor_to_majorana[spinor][term.spinor_weight_index];
                if spinor_factor != &small_zero() {
                    *value += vector_factor.clone()
                        * spinor_factor.clone()
                        * Complex::new(coefficient.clone(), Ratio::from_integer(0));
                }
            }
        }
    }
    output
}

fn gamma_apply(gamma: &[Vec<i8>], source: &[SmallGaussian]) -> Vec<SmallGaussian> {
    gamma
        .iter()
        .map(|row| {
            row.iter()
                .zip(source)
                .filter(|(entry, _)| **entry != 0)
                .map(|(entry, value)| value.clone() * small_integer(i64::from(*entry)))
                .sum()
        })
        .collect()
}

fn add_scaled(target: &mut [SmallGaussian], source: &[SmallGaussian], scale: i64) {
    if scale == 0 {
        return;
    }
    let scale = small_integer(scale);
    for (target, source) in target.iter_mut().zip(source) {
        *target += source.clone() * scale.clone();
    }
}

fn gamma_trace(source: &LorentzVectorSpinor) -> Vec<SmallGaussian> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut output = vec![small_zero(); SPINOR_DIMENSION];
    for axis in 0..VECTOR_DIMENSION {
        let value = gamma_apply(&gammas[axis], &source[axis]);
        add_scaled(&mut output, &value, lorentz_metric(axis));
    }
    output
}

fn recoupled_lorentz(
    source: &LorentzVectorSpinor,
    path: SecondMomentum10001Path,
    momentum_a: usize,
    momentum_b: usize,
    coefficients: [i64; 3],
) -> LorentzVectorSpinor {
    let mut output = vec![vec![small_zero(); SPINOR_DIMENSION]; VECTOR_DIMENSION];
    if path == SecondMomentum10001Path::Trace {
        if momentum_a == momentum_b {
            for axis in 0..VECTOR_DIMENSION {
                add_scaled(&mut output[axis], &source[axis], lorentz_metric(momentum_a));
            }
        }
        return output;
    }

    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let [delta, gamma, trace] = coefficients;
    let source_a = &source[momentum_a];
    let source_b = &source[momentum_b];
    let gamma_a_source_b = gamma_apply(&gammas[momentum_a], source_b);
    let gamma_b_source_a = gamma_apply(&gammas[momentum_b], source_a);
    for output_axis in 0..VECTOR_DIMENSION {
        if output_axis == momentum_a {
            add_scaled(
                &mut output[output_axis],
                source_b,
                delta * lorentz_metric(momentum_a),
            );
        }
        if output_axis == momentum_b {
            add_scaled(
                &mut output[output_axis],
                source_a,
                delta * lorentz_metric(momentum_b),
            );
        }
        let first = gamma_apply(&gammas[output_axis], &gamma_a_source_b);
        let second = gamma_apply(&gammas[output_axis], &gamma_b_source_a);
        add_scaled(&mut output[output_axis], &first, gamma);
        add_scaled(&mut output[output_axis], &second, gamma);
        if momentum_a == momentum_b {
            add_scaled(
                &mut output[output_axis],
                &source[output_axis],
                trace * lorentz_metric(momentum_a),
            );
        }
    }
    output
}

fn target_coordinate(
    source: &LorentzVectorSpinor,
    join: &crate::eleven_dimensional_b5_majorana_target_join::ExactB5MajoranaTargetJoin,
    dual: &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
) -> SmallGaussian {
    let mut total = small_zero();
    for term in &dual.raw_terms {
        let mut raw_value = small_zero();
        for (axis, source_row) in source.iter().enumerate() {
            let vector_factor = &join.lorentz_to_upper_vector[term.vector_weight_index][axis];
            if vector_factor == &small_zero() {
                continue;
            }
            for (majorana, source_value) in source_row.iter().enumerate() {
                let spinor_factor = &join.majorana_to_spinor[term.spinor_weight_index][majorana];
                if spinor_factor != &small_zero() && source_value != &small_zero() {
                    raw_value +=
                        vector_factor.clone() * spinor_factor.clone() * source_value.clone();
                }
            }
        }
        let norm = if term.vector_weight_index == 10 { 2 } else { 1 };
        total += raw_value * small_integer(term.numerator * norm) / small_integer(term.denominator);
    }
    total
}

fn recoupling_coefficients(
    highest_target: usize,
    coefficients: [i64; 3],
) -> (
    BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    usize,
    usize,
    usize,
) {
    let join = crate::eleven_dimensional_b5_majorana_target_join::exact_target_join();
    let duals = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let dual = &duals[highest_target];
    let source = target_state_lorentz_from_join(highest_target, &join);
    recoupling_coefficients_from(&source, &join, dual, coefficients)
}

fn recoupling_coefficients_from(
    source: &LorentzVectorSpinor,
    join: &crate::eleven_dimensional_b5_majorana_target_join::ExactB5MajoranaTargetJoin,
    dual: &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
    coefficients: [i64; 3],
) -> (
    BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    usize,
    usize,
    usize,
) {
    let input_residual = gamma_trace(source)
        .into_iter()
        .filter(|value| value != &small_zero())
        .count();
    let mut output_residual = 0;
    let mut momentum_trace = vec![vec![small_zero(); SPINOR_DIMENSION]; VECTOR_DIMENSION];
    let mut values = BTreeMap::new();
    for path in [
        SecondMomentum10001Path::Trace,
        SecondMomentum10001Path::SymmetricTraceless,
    ] {
        for left in 0..VECTOR_DIMENSION {
            for right in left..VECTOR_DIMENSION {
                let output = recoupled_lorentz(source, path, left, right, coefficients);
                output_residual += gamma_trace(&output)
                    .into_iter()
                    .filter(|value| value != &small_zero())
                    .count();
                if path == SecondMomentum10001Path::SymmetricTraceless && left == right {
                    for axis in 0..VECTOR_DIMENSION {
                        add_scaled(
                            &mut momentum_trace[axis],
                            &output[axis],
                            lorentz_metric(left),
                        );
                    }
                }
                let coefficient = target_coordinate(&output, join, dual);
                if coefficient != small_zero() {
                    values.insert((path, [left, right]), coefficient);
                }
            }
        }
    }
    let momentum_trace_residual = momentum_trace
        .iter()
        .flatten()
        .filter(|value| *value != &small_zero())
        .count();
    (
        values,
        input_residual,
        output_residual,
        momentum_trace_residual,
    )
}

/// Exact primitive momentum-pair coefficients used by the GPU source stream
/// for either `(10001)` path. The physical target checks are rerun before the
/// integer schedule is released.
pub(crate) fn gpu_path_reciprocal_terms(
    path: crate::eleven_dimensional_second_momentum_full_inventory::MomentumPath,
) -> Result<(Vec<([u8; 2], i128)>, String), String> {
    let selected = match path {
        crate::eleven_dimensional_second_momentum_full_inventory::MomentumPath::Trace => {
            SecondMomentum10001Path::Trace
        }
        crate::eleven_dimensional_second_momentum_full_inventory::MomentumPath::SymmetricTraceless => {
            SecondMomentum10001Path::SymmetricTraceless
        }
        crate::eleven_dimensional_second_momentum_full_inventory::MomentumPath::Unique => {
            return Err("the (10001) channel requires trace or symmetric-traceless path".to_string());
        }
    };
    let (coefficients, input_residual, output_residual, momentum_trace_residual) =
        recoupling_coefficients(0, STT_COEFFICIENTS);
    if input_residual != 0 || output_residual != 0 || momentum_trace_residual != 0 {
        return Err(
            "exact (10001) momentum recoupling failed its target residual gates".to_string(),
        );
    }
    let mut terms = Vec::new();
    for ((candidate, pair), coefficient) in coefficients {
        if candidate != selected {
            continue;
        }
        if *coefficient.im.numer() != 0
            || *coefficient.re.denom() != 1
            || *coefficient.im.denom() != 1
        {
            return Err("(10001) GPU path coefficient is not a real integer".to_string());
        }
        let coefficient = i128::from(*coefficient.re.numer());
        if coefficient == 0 {
            return Err("(10001) GPU path emitted a zero coefficient".to_string());
        }
        terms.push((
            [
                u8::try_from(pair[0]).map_err(|_| "momentum pair exceeds u8")?,
                u8::try_from(pair[1]).map_err(|_| "momentum pair exceeds u8")?,
            ],
            coefficient,
        ));
    }
    if terms.is_empty() {
        return Err("(10001) GPU path has no exact momentum terms".to_string());
    }
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-second-momentum-10001-gpu-path-v1\0");
    hash.update(match selected {
        SecondMomentum10001Path::Trace => b"trace".as_slice(),
        SecondMomentum10001Path::SymmetricTraceless => b"stt".as_slice(),
    });
    hash.update((terms.len() as u64).to_le_bytes());
    for (pair, coefficient) in &terms {
        hash.update(pair);
        hash.update(coefficient.to_le_bytes());
    }
    Ok((terms, format!("{:x}", hash.finalize())))
}

fn small_to_exact(value: &SmallGaussian) -> ExactGaussian {
    ExactGaussian {
        real: Ratio::new(
            BigInt::from(*value.re.numer()),
            BigInt::from(*value.re.denom()),
        ),
        imaginary: Ratio::new(
            BigInt::from(*value.im.numer()),
            BigInt::from(*value.im.denom()),
        ),
    }
}

fn physical_to_exact(
    value: &crate::eleven_dimensional_physical_curvature::ExactQi,
) -> ExactGaussian {
    ExactGaussian {
        real: Ratio::new(
            BigInt::from(*value.real.numer()),
            BigInt::from(*value.real.denom()),
        ),
        imaginary: Ratio::new(
            BigInt::from(*value.imaginary.numer()),
            BigInt::from(*value.imaginary.denom()),
        ),
    }
}

fn multiply_exact(left: &ExactGaussian, right: &ExactGaussian) -> ExactGaussian {
    ExactGaussian {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn scale_exact(value: &ExactGaussian, scale: i128) -> ExactGaussian {
    let scale = Ratio::from_integer(BigInt::from(scale));
    ExactGaussian {
        real: value.real.clone() * scale.clone(),
        imaginary: value.imaginary.clone() * scale,
    }
}

fn physical_pivot_templates_from(
    join: &crate::eleven_dimensional_b5_majorana_target_join::ExactB5MajoranaTargetJoin,
    h: &LorentzVectorSpinor,
) -> Vec<PhysicalPivotTemplate> {
    (0..SPINOR_DIMENSION)
        .map(|derivative_weight| {
            let mut terms = Vec::new();
            for derivative_majorana in 0..SPINOR_DIMENSION {
                let derivative = &join.spinor_to_majorana[derivative_majorana][derivative_weight];
                if derivative == &small_zero() {
                    continue;
                }
                for output_vector in 0..VECTOR_DIMENSION {
                    for h_majorana in 0..SPINOR_DIMENSION {
                        let coefficient = derivative.clone() * h[output_vector][h_majorana].clone();
                        if coefficient == small_zero() {
                            continue;
                        }
                        terms.push(
                            crate::eleven_dimensional_physical_curvature::PolynomialFxDhTerm {
                                derivative_spinor: derivative_majorana,
                                h_spinor: h_majorana,
                                output_vector,
                                exterior_spinor_mask: 0,
                                momentum: crate::eleven_dimensional_physical_curvature::FxMomentumMonomial::constant(),
                                coefficient: crate::eleven_dimensional_physical_curvature::ExactQi {
                                    real: coefficient.re,
                                    imaginary: coefficient.im,
                                },
                            },
                        );
                    }
                }
            }
            let image = crate::eleven_dimensional_physical_curvature::apply_polynomial_fx(&terms);
            let x2 = image
                .x_two_11000
                .iter()
                .next()
                .map(|(key, value)| (key.quotient_coordinate, physical_to_exact(value)));
            let x5 = image
                .x_five_10002
                .iter()
                .next()
                .map(|(key, value)| (key.quotient_coordinate, physical_to_exact(value)));
            PhysicalPivotTemplate {
                derivative_spinor_weight_index: derivative_weight,
                x2,
                x5,
            }
        })
        .collect()
}

#[allow(dead_code)]
fn source_terms_by_copy(progress: &FxProgress) -> Vec<Vec<SecondMomentum10001SourceTerm>> {
    (1..=2)
        .map(|copy| {
            let mut terms = Vec::new();
            crate::eleven_dimensional_second_momentum_10001_maps::
                visit_second_momentum_10001_highest_weight_source_terms(copy, |term| {
                    terms.push(term)
                });
            progress.add_processed(1);
            terms
        })
        .collect()
}

fn variable_specs(
    map_specs: &[SecondMomentum10001MapSpec],
) -> Vec<SecondMomentum10001FxVariableSpec> {
    map_specs
        .iter()
        .zip(GLOBAL_10002_10001_ORDINALS)
        .map(
            |(spec, global_77_ordinal)| SecondMomentum10001FxVariableSpec {
                local_ordinal: spec.variable_ordinal,
                global_77_ordinal,
                source_dynkin_label: spec.source_dynkin_label.clone(),
                source_copy: spec.source_copy,
                intermediate_dynkin_label: spec.intermediate_dynkin_label.clone(),
                momentum_path: spec.momentum_path,
            },
        )
        .collect()
}

fn build_terms_for_spec(
    spec: &SecondMomentum10001FxVariableSpec,
    source_terms: &[Vec<SecondMomentum10001SourceTerm>],
    recoupling: &BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    templates: &[PhysicalPivotTemplate],
    progress: &FxProgress,
) -> Vec<SecondMomentumFxColumnTerm> {
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
    struct Key {
        column: usize,
        pair: [usize; 2],
        mask: u32,
        target: usize,
        sector: SecondMomentumFxSector,
    }
    let mut accumulated = BTreeMap::<Key, ExactGaussian>::new();
    let mut pending_progress = 0_u64;
    for source in &source_terms[spec.source_copy - 1] {
        pending_progress += 1;
        if pending_progress == 16_384 {
            progress.add_processed(pending_progress);
            pending_progress = 0;
        }
        let Some((wedge_mask, sign)) =
            crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(
                source.exterior_mask,
                source.intermediate_spinor_weight_index,
            )
        else {
            continue;
        };
        let template = &templates[source.intermediate_spinor_weight_index];
        debug_assert_eq!(
            template.derivative_spinor_weight_index,
            source.intermediate_spinor_weight_index
        );
        for (&(path, pair), path_coefficient) in recoupling {
            if path != spec.momentum_path {
                continue;
            }
            let base = scale_exact(
                &small_to_exact(path_coefficient),
                source.coefficient * i128::from(sign),
            );
            for (sector, selected) in [
                (SecondMomentumFxSector::X2, template.x2.as_ref()),
                (SecondMomentumFxSector::X5, template.x5.as_ref()),
            ] {
                let Some((target, response)) = selected else {
                    continue;
                };
                let value = multiply_exact(&base, response);
                if value.is_zero() {
                    continue;
                }
                let key = Key {
                    column: spec.global_77_ordinal,
                    pair,
                    mask: wedge_mask,
                    target: *target,
                    sector,
                };
                let entry = accumulated.entry(key).or_insert_with(ExactGaussian::zero);
                entry.real += value.real;
                entry.imaginary += value.imaginary;
            }
        }
    }
    progress.add_processed(pending_progress);
    let terms = accumulated
        .into_iter()
        .filter(|(_, coefficient)| !coefficient.is_zero())
        .map(|(key, coefficient)| SecondMomentumFxColumnTerm {
            coefficient_column: key.column,
            gauge_channel: SecondMomentumGaugeChannel::new(GAUGE_FORM_DEGREE).unwrap(),
            gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
            source_momentum: DegreeTwoMomentumMonomial::from_pair(key.pair[0], key.pair[1])
                .unwrap(),
            parameter_component: GAUGE_PARAMETER_COMPONENT,
            target_coordinate: key.target,
            spinor_derivative_mask: key.mask,
            sector: key.sector,
            coefficient,
        })
        .collect::<Vec<_>>();
    progress.column_complete(spec, terms.len());
    terms
}

/// Build independent coefficient columns concurrently, then return them in
/// canonical specification order. Each worker owns its exact accumulator, so
/// the hot loop has no shared locks. `IndexedParallelIterator::collect`
/// preserves input order, and the caller folds the completed columns into the
/// streaming certificate sequentially to keep artifact bytes deterministic.
fn build_terms_for_specs_parallel(
    specs: &[SecondMomentum10001FxVariableSpec],
    source_terms: &[Vec<SecondMomentum10001SourceTerm>],
    recoupling: &BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    templates: &[PhysicalPivotTemplate],
    progress: &FxProgress,
) -> Vec<Vec<SecondMomentumFxColumnTerm>> {
    specs
        .par_iter()
        .map(|spec| build_terms_for_spec(spec, source_terms, recoupling, templates, progress))
        .collect()
}

struct ScalarFunctionalProjection {
    values: Vec<i128>,
    observed_terms_per_path: u64,
    source_l1: u128,
}

fn scalar_projection_index(
    derivative_weight: usize,
    pair: usize,
    sector: usize,
    seed: usize,
    bucket: usize,
    pair_count: usize,
) -> usize {
    ((((derivative_weight * pair_count + pair) * PROJECTED_SECTORS + sector)
        * SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len()
        + seed)
        * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED)
        + bucket
}

fn projection_pairs(
    recoupling: &BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
) -> Result<Vec<([usize; 2], DegreeTwoMomentumMonomial)>, String> {
    let pairs_for = |path| {
        recoupling
            .keys()
            .filter_map(|(candidate, pair)| (*candidate == path).then_some(*pair))
            .collect::<Vec<_>>()
    };
    let trace = pairs_for(SecondMomentum10001Path::Trace);
    let symmetric_traceless = pairs_for(SecondMomentum10001Path::SymmetricTraceless);
    if trace.is_empty() || trace != symmetric_traceless {
        return Err("10001 momentum paths do not share one canonical pair schedule".to_string());
    }
    trace
        .into_iter()
        .map(|pair| {
            Ok((
                pair,
                DegreeTwoMomentumMonomial::from_pair(pair[0], pair[1])?,
            ))
        })
        .collect()
}

fn template_sector(
    template: &PhysicalPivotTemplate,
    sector: usize,
) -> Option<(SecondMomentumFxSector, usize)> {
    match sector {
        0 => template
            .x2
            .as_ref()
            .map(|(target, _)| (SecondMomentumFxSector::X2, *target)),
        1 => template
            .x5
            .as_ref()
            .map(|(target, _)| (SecondMomentumFxSector::X5, *target)),
        _ => None,
    }
}

fn project_source_copy_scalars(
    source_copy: usize,
    source_map: &EmbeddedSourceMap,
    pairs: &[([usize; 2], DegreeTwoMomentumMonomial)],
    templates: &[PhysicalPivotTemplate],
    progress: &FxProgress,
) -> Result<ScalarFunctionalProjection, String> {
    if source_map.copy != source_copy || source_map.components.len() != SPINOR_DIMENSION {
        return Err("invalid 10001 embedded source map".to_string());
    }
    let value_count = SPINOR_DIMENSION
        .checked_mul(pairs.len())
        .and_then(|value| value.checked_mul(PROJECTED_SECTORS))
        .and_then(|value| value.checked_mul(SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len()))
        .and_then(|value| value.checked_mul(SECOND_MOMENTUM_FX_BUCKETS_PER_SEED))
        .ok_or_else(|| "10001 projected scalar shape overflow".to_string())?;

    let components = source_map
        .components
        .par_iter()
        .map(|(&derivative_weight, component)| {
            if derivative_weight >= SPINOR_DIMENSION
                || component.masks.len() != component.coefficients.len()
            {
                return Err("invalid 10001 source component".to_string());
            }
            let mut values = vec![0_i128; value_count];
            let mut observed_terms_per_path = 0_u64;
            let mut source_l1 = 0_u128;
            let mut processed_terms = 0_u64;
            for (&exterior_mask, &coefficient) in
                component.masks.iter().zip(&component.coefficients)
            {
                if coefficient == 0 {
                    continue;
                }
                processed_terms = processed_terms
                    .checked_add(1)
                    .ok_or_else(|| "10001 processed source count overflow".to_string())?;
                source_l1 = source_l1
                    .checked_add(coefficient.unsigned_abs())
                    .ok_or_else(|| "10001 source L1 bound overflow".to_string())?;
                let Some((wedge_mask, sign)) =
                    crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(
                        exterior_mask,
                        derivative_weight,
                    )
                else {
                    continue;
                };
                let signed_source = coefficient
                    .checked_mul(sign)
                    .ok_or_else(|| "10001 signed source coefficient overflow".to_string())?;
                let template = &templates[derivative_weight];
                for (pair_ordinal, (_, monomial)) in pairs.iter().enumerate() {
                    for sector_ordinal in 0..PROJECTED_SECTORS {
                        let Some((sector, target)) = template_sector(template, sector_ordinal)
                        else {
                            continue;
                        };
                        observed_terms_per_path = observed_terms_per_path
                            .checked_add(1)
                            .ok_or_else(|| "10001 observed term count overflow".to_string())?;
                        for (seed, (bucket, hash_sign)) in
                            second_momentum_fx_functional_assignments(
                                SecondMomentumGaugeChannel::new(GAUGE_FORM_DEGREE)?,
                                SecondMomentumGaugeBranch::P2D13Wedge,
                                *monomial,
                                GAUGE_PARAMETER_COMPONENT,
                                target,
                                wedge_mask,
                                sector,
                            )
                            .into_iter()
                            .enumerate()
                        {
                            let value = signed_source
                                .checked_mul(i128::from(hash_sign))
                                .ok_or_else(|| "10001 functional sign overflow".to_string())?;
                            let index = scalar_projection_index(
                                derivative_weight,
                                pair_ordinal,
                                sector_ordinal,
                                seed,
                                bucket,
                                pairs.len(),
                            );
                            values[index] = values[index]
                                .checked_add(value)
                                .ok_or_else(|| "10001 projected bucket overflow".to_string())?;
                        }
                    }
                }
            }
            progress.add_processed(processed_terms);
            Ok(ScalarFunctionalProjection {
                values,
                observed_terms_per_path,
                source_l1,
            })
        })
        .collect::<Vec<Result<ScalarFunctionalProjection, String>>>();

    let mut merged = ScalarFunctionalProjection {
        values: vec![0_i128; value_count],
        observed_terms_per_path: 0,
        source_l1: 0,
    };
    for component in components {
        let component = component?;
        merged.observed_terms_per_path = merged
            .observed_terms_per_path
            .checked_add(component.observed_terms_per_path)
            .ok_or_else(|| "10001 merged observed term count overflow".to_string())?;
        merged.source_l1 = merged
            .source_l1
            .checked_add(component.source_l1)
            .ok_or_else(|| "10001 merged source L1 overflow".to_string())?;
        for (target, source) in merged.values.iter_mut().zip(component.values) {
            *target = target
                .checked_add(source)
                .ok_or_else(|| "10001 merged projected bucket overflow".to_string())?;
        }
    }
    if merged.source_l1 > i128::MAX as u128 {
        return Err("10001 source L1 bound exceeds signed i128".to_string());
    }
    Ok(merged)
}

fn projected_column_from_scalars(
    spec: &SecondMomentum10001FxVariableSpec,
    scalar: &ScalarFunctionalProjection,
    pairs: &[([usize; 2], DegreeTwoMomentumMonomial)],
    recoupling: &BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    templates: &[PhysicalPivotTemplate],
    progress: &FxProgress,
) -> Result<SecondMomentumFxProjectedColumn, String> {
    let mut group_schedule = pairs
        .iter()
        .enumerate()
        .flat_map(|(pair_ordinal, (_, monomial))| {
            (0..PROJECTED_SECTORS).filter_map(move |sector_ordinal| {
                templates
                    .iter()
                    .find_map(|template| template_sector(template, sector_ordinal))
                    .map(|(sector, _)| ((*monomial, sector), pair_ordinal, sector_ordinal))
            })
        })
        .collect::<Vec<_>>();
    group_schedule.sort_by_key(|entry| entry.0);
    group_schedule.dedup_by_key(|entry| entry.0);

    let mut entries = Vec::new();
    for &((monomial, sector), pair_ordinal, sector_ordinal) in &group_schedule {
        let pair = pairs[pair_ordinal].0;
        let path_coefficient = recoupling
            .get(&(spec.momentum_path, pair))
            .ok_or_else(|| "missing 10001 path coefficient".to_string())?;
        let path_coefficient = small_to_exact(path_coefficient);
        for seed in 0..SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len() {
            for bucket in 0..SECOND_MOMENTUM_FX_BUCKETS_PER_SEED {
                let mut coefficient = ExactGaussian::zero();
                for (derivative_weight, template) in templates.iter().enumerate() {
                    let selected = match sector {
                        SecondMomentumFxSector::X2 => template.x2.as_ref(),
                        SecondMomentumFxSector::X5 => template.x5.as_ref(),
                    };
                    let Some((_, response)) = selected else {
                        continue;
                    };
                    let scalar_value = scalar.values[scalar_projection_index(
                        derivative_weight,
                        pair_ordinal,
                        sector_ordinal,
                        seed,
                        bucket,
                        pairs.len(),
                    )];
                    if scalar_value == 0 {
                        continue;
                    }
                    let factor = multiply_exact(&path_coefficient, response);
                    let contribution = scale_exact(&factor, scalar_value);
                    coefficient.real += contribution.real;
                    coefficient.imaginary += contribution.imaginary;
                }
                if !coefficient.is_zero() {
                    entries.push(SecondMomentumFxProjectedEntry {
                        source_momentum: monomial,
                        sector,
                        seed_ordinal: seed,
                        bucket,
                        coefficient,
                    });
                }
            }
        }
    }
    progress.column_complete(
        spec,
        usize::try_from(scalar.observed_terms_per_path)
            .map_err(|_| "10001 observed term count exceeds usize".to_string())?,
    );
    Ok(SecondMomentumFxProjectedColumn {
        coefficient_column: spec.global_77_ordinal,
        gauge_channel: SecondMomentumGaugeChannel::new(GAUGE_FORM_DEGREE)?,
        gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
        observed_terms: scalar.observed_terms_per_path,
        observed_groups: group_schedule
            .into_iter()
            .map(|(group, _, _)| group)
            .collect(),
        entries,
    })
}

fn build_projected_columns_parallel(
    specs: &[SecondMomentum10001FxVariableSpec],
    source_maps: &[EmbeddedSourceMap; 2],
    recoupling: &BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    templates: &[PhysicalPivotTemplate],
    progress: &FxProgress,
) -> Result<Vec<SecondMomentumFxProjectedColumn>, String> {
    let pairs = projection_pairs(recoupling)?;
    let (first, second) = rayon::join(
        || project_source_copy_scalars(1, &source_maps[0], &pairs, templates, progress),
        || project_source_copy_scalars(2, &source_maps[1], &pairs, templates, progress),
    );
    let scalars = [first?, second?];
    specs
        .par_iter()
        .map(|spec| {
            projected_column_from_scalars(
                spec,
                &scalars[spec.source_copy - 1],
                &pairs,
                recoupling,
                templates,
                progress,
            )
        })
        .collect()
}

fn first_physical_term(
    spec: &SecondMomentum10001FxVariableSpec,
    source_maps: &[EmbeddedSourceMap; 2],
    recoupling: &BTreeMap<(SecondMomentum10001Path, [usize; 2]), SmallGaussian>,
    templates: &[PhysicalPivotTemplate],
) -> Option<SecondMomentumFxColumnTerm> {
    for (&spinor, component) in &source_maps[spec.source_copy - 1].components {
        for (&exterior_mask, &coefficient) in component.masks.iter().zip(&component.coefficients) {
            if coefficient == 0 {
                continue;
            }
            let source = SecondMomentum10001SourceTerm {
                source_copy: spec.source_copy,
                intermediate_spinor_weight_index: spinor,
                exterior_mask,
                coefficient,
            };
            let Some((wedge_mask, sign)) =
                crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(
                    source.exterior_mask,
                    source.intermediate_spinor_weight_index,
                )
            else {
                continue;
            };
            let template = &templates[source.intermediate_spinor_weight_index];
            for (&(path, pair), path_coefficient) in recoupling {
                if path != spec.momentum_path {
                    continue;
                }
                let base = scale_exact(
                    &small_to_exact(path_coefficient),
                    source.coefficient.checked_mul(sign)?,
                );
                for (sector, selected) in [
                    (SecondMomentumFxSector::X2, template.x2.as_ref()),
                    (SecondMomentumFxSector::X5, template.x5.as_ref()),
                ] {
                    let Some((target, response)) = selected else {
                        continue;
                    };
                    let coefficient = multiply_exact(&base, response);
                    if coefficient.is_zero() {
                        continue;
                    }
                    return Some(SecondMomentumFxColumnTerm {
                        coefficient_column: spec.global_77_ordinal,
                        gauge_channel: SecondMomentumGaugeChannel::new(GAUGE_FORM_DEGREE).ok()?,
                        gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
                        source_momentum: DegreeTwoMomentumMonomial::from_pair(pair[0], pair[1])
                            .ok()?,
                        parameter_component: GAUGE_PARAMETER_COMPONENT,
                        target_coordinate: *target,
                        spinor_derivative_mask: wedge_mask,
                        sector,
                        coefficient,
                    });
                }
            }
        }
    }
    None
}

fn local_rank(summary: &ExactRankSummary) -> ExactRankSummary {
    assert!(summary.rank <= LOCAL_VARIABLES);
    ExactRankSummary {
        equation_count: summary.equation_count,
        variable_count: LOCAL_VARIABLES,
        rank: summary.rank,
        nullity: LOCAL_VARIABLES - summary.rank,
        outcome: if summary.rank == LOCAL_VARIABLES {
            "unique".to_string()
        } else {
            "underdetermined".to_string()
        },
    }
}

fn map_manifest_sha256(
    report: &crate::eleven_dimensional_second_momentum_10001_maps::SecondMomentum10001MapReport,
) -> String {
    #[derive(Serialize)]
    struct Manifest<'a> {
        schema_version: &'a str,
        recoupling_certificate_sha256: &'a str,
        embedded_source_hashes: Vec<&'a str>,
        coupled_map_hashes: Vec<&'a str>,
        global_ordinals: [usize; 4],
    }
    sha256_json(&Manifest {
        schema_version: &report.schema_version,
        recoupling_certificate_sha256: &report.recoupling_certificate_sha256,
        embedded_source_hashes: report
            .embedded_sources
            .iter()
            .map(|source| source.fixture_sha256.as_str())
            .collect(),
        coupled_map_hashes: report
            .embedded_sources
            .iter()
            .map(|source| source.coupled_map_sha256.as_str())
            .collect(),
        global_ordinals: GLOBAL_10002_10001_ORDINALS,
    })
}

pub fn verify_second_momentum_10001_fx() -> Result<SecondMomentum10001FxReport, String> {
    let (progress, _heartbeat) = FxProgress::start();
    progress.set_phase(0, 1);
    let (map_report, source_maps) = crate::eleven_dimensional_second_momentum_10001_maps::
        verify_second_momentum_10001_maps_with_embedded_sources();
    if !map_report.passed {
        return Err("second-momentum 10001 source-map certificate failed".to_string());
    }
    progress.add_processed(1);
    let map_specs =
        crate::eleven_dimensional_second_momentum_10001_maps::second_momentum_10001_map_specs();
    let specs = variable_specs(&map_specs);
    let source_nonzero_terms_by_copy = source_maps
        .iter()
        .zip(&map_report.embedded_sources)
        .map(|(source, audit)| {
            assert_eq!(source.copy, audit.source_copy);
            audit.coupled_nonzero_terms
        })
        .collect::<Vec<_>>();
    progress.set_phase(1, 2);
    let target_join = crate::eleven_dimensional_b5_majorana_target_join::exact_target_join();
    let target_state =
        target_state_lorentz_from_join(map_report.highest_target_basis_ordinal, &target_join);
    let target_duals = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let target_dual = &target_duals[map_report.highest_target_basis_ordinal];
    let (original_recoupling, mutated_recoupling) = rayon::join(
        || {
            let result = recoupling_coefficients_from(
                &target_state,
                &target_join,
                target_dual,
                STT_COEFFICIENTS,
            );
            progress.add_processed(1);
            result
        },
        || {
            let result = recoupling_coefficients_from(
                &target_state,
                &target_join,
                target_dual,
                [11, -1, -1],
            );
            progress.add_processed(1);
            result
        },
    );
    let (recoupling, input_residual, output_residual, momentum_trace_residual) =
        original_recoupling;
    let (_, _, mutated_output_residual, mutated_momentum_trace_residual) = mutated_recoupling;
    progress.set_phase(2, SPINOR_DIMENSION as u64);
    let templates = physical_pivot_templates_from(&target_join, &target_state);
    progress.add_processed(SPINOR_DIMENSION as u64);
    let source_fixture_manifest_sha256 = sha256_json(
        &map_report
            .embedded_sources
            .iter()
            .map(|source| (&source.fixture_sha256, source.source_copy))
            .collect::<Vec<_>>(),
    );
    let component_map_manifest_sha256 = map_manifest_sha256(&map_report);
    let provenance = SecondMomentumFxProvenance {
        source_kind: SecondMomentumFxSourceKind::PhysicalRecoupledStream,
        campaign_id: "11d-p2-10002x10001-highest-target-parameter0-v1".to_string(),
        representation_inventory_sha256: SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256
            .to_string(),
        level12_fixture_manifest_sha256: source_fixture_manifest_sha256.clone(),
        component_cg_manifest_sha256: component_map_manifest_sha256.clone(),
        coefficient_layout_sha256: second_momentum_fx_coefficient_layout_sha256(),
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
    let mut harness = SecondMomentumFxStreamingAccumulator::new(provenance, coverage)?;
    let mut emitted_physical_fx_terms = 0_usize;
    let first_term = first_physical_term(&specs[0], &source_maps, &recoupling, &templates);
    let column_source_terms = source_nonzero_terms_by_copy
        .iter()
        .map(|terms| *terms as u64)
        .sum();
    progress.set_phase(3, column_source_terms);
    let projected_columns =
        build_projected_columns_parallel(&specs, &source_maps, &recoupling, &templates, &progress)?;
    progress.set_phase(4, LOCAL_VARIABLES as u64);
    for column in projected_columns {
        emitted_physical_fx_terms = emitted_physical_fx_terms
            .checked_add(
                usize::try_from(column.observed_terms)
                    .map_err(|_| "physical p2 F_X term count exceeds usize".to_string())?,
            )
            .ok_or_else(|| "physical p2 F_X term count overflow".to_string())?;
        harness.push_projected_column(column)?;
        progress.add_processed(1);
    }
    progress.set_phase(5, 1);
    let typed_harness_checkpoint = harness.finalize()?;
    progress.add_processed(1);
    let canonical_functional_rows_sha256 = typed_harness_checkpoint
        .report
        .canonical_functional_rows_sha256
        .clone();
    let coefficient_mutation_detected = first_term.as_ref().is_some_and(|first| {
        let original_hash = canonical_second_momentum_fx_stream_sha256(&[first.clone()]);
        let mut mutated = first.clone();
        mutated.coefficient.real += Ratio::from_integer(BigInt::from(1));
        canonical_second_momentum_fx_stream_sha256(&[mutated]) != original_hash
            && first.validate().is_ok()
    });
    let global_ordinal_mutation_detected = first_term.as_ref().is_some_and(|first| {
        let original_hash = canonical_second_momentum_fx_stream_sha256(&[first.clone()]);
        let mut mutated = first.clone();
        mutated.coefficient_column = 18;
        canonical_second_momentum_fx_stream_sha256(&[mutated]) != original_hash
    });
    let x2_rank = local_rank(&typed_harness_checkpoint.report.global_x2);
    let x5_rank = local_rank(&typed_harness_checkpoint.report.global_x5);
    let joint_rank = local_rank(&typed_harness_checkpoint.report.global_joint);

    let trace_nonzero_momentum_pairs = recoupling
        .keys()
        .filter(|(path, _)| *path == SecondMomentum10001Path::Trace)
        .count();
    let symmetric_traceless_nonzero_momentum_pairs = recoupling
        .keys()
        .filter(|(path, _)| *path == SecondMomentum10001Path::SymmetricTraceless)
        .count();
    let stt_mutation_detected = output_residual == 0
        && momentum_trace_residual == 0
        && (mutated_output_residual > 0 || mutated_momentum_trace_residual > 0);
    let passed = specs.len() == LOCAL_VARIABLES
        && specs
            .iter()
            .map(|spec| spec.global_77_ordinal)
            .collect::<Vec<_>>()
            == GLOBAL_10002_10001_ORDINALS
        && input_residual == 0
        && output_residual == 0
        && momentum_trace_residual == 0
        && trace_nonzero_momentum_pairs > 0
        && symmetric_traceless_nonzero_momentum_pairs > 0
        && templates.len() == SPINOR_DIMENSION
        && templates.iter().any(|template| template.x2.is_some())
        && templates.iter().any(|template| template.x5.is_some())
        && emitted_physical_fx_terms > 0
        && typed_harness_checkpoint.report.observed_nonzero_columns == GLOBAL_10002_10001_ORDINALS
        && !typed_harness_checkpoint
            .report
            .declared_slice_no_go_certified
        && !typed_harness_checkpoint.report.full_f_a_g_p_established
        && coefficient_mutation_detected
        && global_ordinal_mutation_detected
        && stt_mutation_detected;

    let report = SecondMomentum10001FxReport {
        schema_version: "adynkra-11d-second-momentum-10001-physical-fx-slice-v1".to_string(),
        role: "exact physical X2/X5 diagnostic for four 10002 x trace/STT columns on one highest-target, degree-zero-parameter p2D13 slice".to_string(),
        variable_specs: specs,
        local_variable_count: LOCAL_VARIABLES,
        remaining_second_momentum_variables: REMAINING_VARIABLES,
        gauge_form_degree: GAUGE_FORM_DEGREE,
        gauge_parameter_component: GAUGE_PARAMETER_COMPONENT,
        highest_intermediate_target_basis_ordinal: map_report.highest_target_basis_ordinal,
        target_slice: "single highest (10001) intermediate coordinate; after exact Lorentz-Majorana join and fixed physical F_X, retain the first nonzero canonical X2 and X5 quotient coordinate separately for each derivative-spinor weight".to_string(),
        source_nonzero_terms_by_copy,
        trace_nonzero_momentum_pairs,
        symmetric_traceless_nonzero_momentum_pairs,
        input_target_gamma_trace_residual_entries: input_residual,
        recoupled_target_gamma_trace_residual_entries: output_residual,
        symmetric_traceless_momentum_trace_residual_entries: momentum_trace_residual,
        physical_pivot_coordinates: templates
            .iter()
            .map(|template| PhysicalPivotCoordinate {
                derivative_spinor_weight_index: template.derivative_spinor_weight_index,
                x2_quotient_coordinate: template.x2.as_ref().map(|(coordinate, _)| *coordinate),
                x5_quotient_coordinate: template.x5.as_ref().map(|(coordinate, _)| *coordinate),
            })
            .collect(),
        physical_adapter: "exact B5-to-Lorentz-Majorana join followed by eleven_dimensional_physical_curvature::apply_polynomial_fx".to_string(),
        emitted_physical_fx_terms,
        canonical_functional_rows_sha256,
        source_fixture_manifest_sha256,
        component_map_manifest_sha256,
        x2_rank,
        x5_rank,
        joint_rank,
        typed_harness_checkpoint,
        coefficient_mutation_detected,
        global_ordinal_mutation_detected,
        stt_mutation_detected,
        p2_d13_declared_slice_computed: true,
        p3_d11_contraction_computed: false,
        all_77_variables_computed: false,
        full_parameter_projection_complete: false,
        full_target_projection_complete: false,
        full_physical_f_a_g_p_established: false,
        passed,
        boundary: "This is an exact four-variable diagnostic on one declared p2D13 slice. A saturated four-column rank excludes those four columns only on this slice. A kernel is provisional. The p3D11 contraction branch, five other gauge degrees, remaining parameter and target coordinates, the other 73 p2 variables, J/W, and the generic momentum tower are absent, so full physical F A G_p is false.".to_string(),
    };
    progress.finish();
    Ok(report)
}

pub fn write_second_momentum_10001_fx_artifact(
    path: &Path,
) -> io::Result<SecondMomentum10001FxReport> {
    let report = verify_second_momentum_10001_fx()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "second-momentum 10001 physical F_X slice did not pass",
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&report).map_err(io::Error::other)?,
    )?;
    fs::rename(temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_path_schedules_are_exact_integer_and_distinct() {
        let (trace, trace_sha) = gpu_path_reciprocal_terms(
            crate::eleven_dimensional_second_momentum_full_inventory::MomentumPath::Trace,
        )
        .unwrap();
        let (stt, stt_sha) = gpu_path_reciprocal_terms(
            crate::eleven_dimensional_second_momentum_full_inventory::MomentumPath::SymmetricTraceless,
        )
        .unwrap();
        assert_eq!(trace.len(), 11);
        assert_eq!(stt.len(), 11);
        assert_ne!(trace, stt);
        assert_ne!(trace_sha, stt_sha);
    }

    #[test]
    fn four_10002_trace_stt_columns_reach_the_physical_fx_harness() {
        let report = verify_second_momentum_10001_fx().unwrap();
        assert!(
            report.passed,
            "terms={} trace_pairs={} stt_pairs={} in_res={} out_res={} momentum_trace_res={} x2={:?} x5={:?} joint={:?} stream={} cg={} slice={} mutations={}/{}/{} pivots_x2={} pivots_x5={}",
            report.emitted_physical_fx_terms,
            report.trace_nonzero_momentum_pairs,
            report.symmetric_traceless_nonzero_momentum_pairs,
            report.input_target_gamma_trace_residual_entries,
            report.recoupled_target_gamma_trace_residual_entries,
            report.symmetric_traceless_momentum_trace_residual_entries,
            report.x2_rank,
            report.x5_rank,
            report.joint_rank,
            report
                .typed_harness_checkpoint
                .report
                .canonical_functional_rows_sha256
                == report.canonical_functional_rows_sha256,
            false,
            report
                .typed_harness_checkpoint
                .report
                .declared_slice_no_go_certified,
            report.coefficient_mutation_detected,
            report.global_ordinal_mutation_detected,
            report.stt_mutation_detected,
            report
                .physical_pivot_coordinates
                .iter()
                .filter(|pivot| pivot.x2_quotient_coordinate.is_some())
                .count(),
            report
                .physical_pivot_coordinates
                .iter()
                .filter(|pivot| pivot.x5_quotient_coordinate.is_some())
                .count(),
        );
        assert_eq!(
            report
                .variable_specs
                .iter()
                .map(|spec| spec.global_77_ordinal)
                .collect::<Vec<_>>(),
            GLOBAL_10002_10001_ORDINALS
        );
        assert_eq!(report.source_nonzero_terms_by_copy, [4_363_766, 580_279]);
        assert_eq!(report.emitted_physical_fx_terms, 109_579_140);
        assert_eq!(
            report.canonical_functional_rows_sha256,
            "64da5ae07368fbeb135342c49a2fdd14762c3d0cbc9d16e295fd8f0d78f5c34d"
        );
        assert_eq!(
            report.typed_harness_checkpoint.checkpoint_sha256,
            "4de0865665364188b16c3ffde3aba9c957ac0e1b5bf07138fde1e16d07c7700d"
        );
        assert_eq!(report.joint_rank.variable_count, 4);
        assert_eq!(report.joint_rank.rank, 4);
        assert_eq!(report.joint_rank.nullity, 0);
        assert_eq!(report.joint_rank.rank + report.joint_rank.nullity, 4);
        assert_eq!(
            report
                .typed_harness_checkpoint
                .report
                .observed_nonzero_columns,
            GLOBAL_10002_10001_ORDINALS
        );
        assert!(
            !report
                .typed_harness_checkpoint
                .report
                .declared_slice_no_go_certified
        );
        assert!(!report.full_physical_f_a_g_p_established);
    }

    #[test]
    fn physical_stream_and_recoupling_mutations_are_detected() {
        let report = verify_second_momentum_10001_fx().unwrap();
        assert!(report.coefficient_mutation_detected);
        assert!(report.global_ordinal_mutation_detected);
        assert!(report.stt_mutation_detected);
    }
}
