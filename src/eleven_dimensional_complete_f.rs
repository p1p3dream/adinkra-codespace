//! Fail-closed assembly boundary for the complete physical 11D operator `F`.
//!
//! The repository already contains exact, source-normalized operators for the
//! `X_[2]`, `X_[5]`, `J`, connection, mixed-torsion, and linearized `W`
//! sectors.  This module joins those operators at the geometry-jet level and
//! now composes the exact constrained frame from the canonical
//! `H_hat=P_320 H` representative through the complete first geometry jet.
//!
//! Once that map and the target curvature adapter are complete, the canonical
//! operator can be hashed and its exact polynomial target-side kernel can be
//! promoted to the physical `K` gate.  A pointwise or bounded-momentum
//! nullspace is not accepted as `K`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_constrained_geometry_jet::{
    ConstrainedGeometryJetSector, visit_constrained_d_delta, visit_constrained_geometry_jet,
};
use crate::eleven_dimensional_first_superspace_jet::{
    self as first_jet, FirstSuperspaceJetInput, FirstSuperspaceJetOutput,
};
use crate::eleven_dimensional_h_hat_jet::{
    LinearizedFrameJetSector, LinearizedFrameSuperfields, canonical_gamma_traceless_frame_basis,
    canonical_physical_frame_representative, visit_d_h_jet, visit_linearized_frame_jet,
};
use crate::eleven_dimensional_physical_curvature::{self as physical, ExactQi, PhysicalXImage};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial, left_multiply_d,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, ExactPolynomialMatrix, MomentumMonomial,
    PhysicalFAdapterDescriptor, PhysicalFTargetAdapter, TargetCurvatureCoordinate,
    TargetCurvatureSector, TargetSector, target_sector_complex,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-complete-physical-f-construction-v8";

/// All geometry coordinates needed by the source-fixed operators that are
/// currently executable.  These coordinates must eventually be produced by
/// one `H_hat` jet map.  Accepting them independently here keeps the existing
/// exact geometry useful without pretending that consistency has been proved.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GeometryLevelPhysicalFInput {
    /// `D_alpha H_beta{}^c`, in the ordering of `physical_curvature`.
    pub d_h: BTreeMap<usize, ExactQi>,
    /// Undifferentiated `C_{alpha beta}{}^gamma`.
    pub c_alpha_beta_gamma: BTreeMap<usize, ExactQi>,
    /// Undifferentiated `C_{alpha,b}{}^c`.
    pub c_alpha_b_c: BTreeMap<usize, ExactQi>,
    /// The consistent first superspace jet consumed by `D J`, torsion, and W.
    pub first_jet: FirstSuperspaceJetInput,
}

/// The exact geometry-level union of every implemented physical-F sector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryLevelPhysicalFOutput {
    pub x: PhysicalXImage,
    pub spinorial_connection: BTreeMap<usize, ExactQi>,
    pub j_one: BTreeMap<usize, ExactQi>,
    pub j_two: BTreeMap<usize, ExactQi>,
    pub j_plus: BTreeMap<usize, ExactQi>,
    pub j_minus: BTreeMap<usize, ExactQi>,
    pub first_jet: FirstSuperspaceJetOutput,
}

fn validate_sparse_dimension(
    name: &str,
    values: &BTreeMap<usize, ExactQi>,
    dimension: usize,
) -> Result<(), String> {
    if let Some(index) = values.keys().find(|index| **index >= dimension) {
        return Err(format!(
            "{name} coordinate {index} is outside dimension {dimension}"
        ));
    }
    Ok(())
}

/// Join every currently implemented source-fixed sector on one geometry jet.
///
/// This function is an exact linear assembly, but its input is not yet the
/// image of a proved `H_hat` jet map.  Callers must therefore retain the
/// geometry-level qualifier.
pub fn assemble_geometry_level_physical_f(
    input: &GeometryLevelPhysicalFInput,
) -> Result<GeometryLevelPhysicalFOutput, String> {
    assemble_geometry_level_physical_f_internal(input, true)
}

fn assemble_geometry_level_physical_f_internal(
    input: &GeometryLevelPhysicalFInput,
    include_w_2001: bool,
) -> Result<GeometryLevelPhysicalFOutput, String> {
    validate_sparse_dimension("d_h", &input.d_h, physical::DH_DIMENSION)?;
    validate_sparse_dimension(
        "c_alpha_beta_gamma",
        &input.c_alpha_beta_gamma,
        physical::SPINOR_ANHOLONOMY_DIMENSION,
    )?;
    validate_sparse_dimension(
        "c_alpha_b_c",
        &input.c_alpha_b_c,
        physical::C_ALPHA_VECTOR_VECTOR_DIMENSION,
    )?;

    let x = physical::apply_leading_physical_x(&input.d_h);
    let spinorial_connection = physical::apply_spinorial_connection(&input.c_alpha_b_c);
    let j_one = physical::apply_j_one(&input.c_alpha_beta_gamma, &spinorial_connection);
    let j_two = physical::cached_c_alpha_b_c_to_j_operator().apply_sparse(&input.c_alpha_b_c);
    let j_plus = physical::apply_j_plus(&j_one, &j_two);
    let j_minus = physical::apply_j_minus(&j_one, &j_two);
    let first_jet = if include_w_2001 {
        first_jet::assemble_first_superspace_jet(&input.first_jet)
    } else {
        first_jet::assemble_first_superspace_jet_2021_only(&input.first_jet)
    };

    Ok(GeometryLevelPhysicalFOutput {
        x,
        spinorial_connection,
        j_one,
        j_two,
        j_plus,
        j_minus,
        first_jet,
    })
}

#[cfg(feature = "cuda")]
fn prolong_sparse_operator(operator: &physical::SparseQiOperator) -> physical::SparseQiOperator {
    let mut columns = Vec::with_capacity(physical::SPINOR_DIMENSION * operator.input_dimension);
    for derivative in 0..physical::SPINOR_DIMENSION {
        for column in &operator.columns {
            columns.push(
                column
                    .iter()
                    .map(|entry| physical::SparseQiEntry {
                        row: derivative * operator.output_dimension + entry.row,
                        coefficient: entry.coefficient.clone(),
                    })
                    .collect(),
            );
        }
    }
    physical::SparseQiOperator {
        input_dimension: physical::SPINOR_DIMENSION * operator.input_dimension,
        output_dimension: physical::SPINOR_DIMENSION * operator.output_dimension,
        columns,
    }
}

#[cfg(feature = "cuda")]
struct GeometryLevelCudaOperators {
    x_two_gamma: exact_cuda_sparse::ExactCudaSparseOperator,
    x_two_hook: exact_cuda_sparse::ExactCudaSparseOperator,
    x_five_gamma: exact_cuda_sparse::ExactCudaSparseOperator,
    x_five_hook: exact_cuda_sparse::ExactCudaSparseOperator,
    spinorial_connection: exact_cuda_sparse::ExactCudaSparseOperator,
    j_one_anholonomy: exact_cuda_sparse::ExactCudaSparseOperator,
    j_one_connection: exact_cuda_sparse::ExactCudaSparseOperator,
    j_two: exact_cuda_sparse::ExactCudaSparseOperator,
    d_spinorial_connection: exact_cuda_sparse::ExactCudaSparseOperator,
    d_j_one_anholonomy: exact_cuda_sparse::ExactCudaSparseOperator,
    d_j_one_connection: exact_cuda_sparse::ExactCudaSparseOperator,
    d_j_two: exact_cuda_sparse::ExactCudaSparseOperator,
    bosonic_connection: exact_cuda_sparse::ExactCudaSparseOperator,
    t_from_bosonic_connection: exact_cuda_sparse::ExactCudaSparseOperator,
    w_2021_from_t: exact_cuda_sparse::ExactCudaSparseOperator,
    w_2021_from_d_j: exact_cuda_sparse::ExactCudaSparseOperator,
    w_2001_from_t: exact_cuda_sparse::ExactCudaSparseOperator,
    w_2001_from_d_j: exact_cuda_sparse::ExactCudaSparseOperator,
}

#[cfg(feature = "cuda")]
impl GeometryLevelCudaOperators {
    fn new() -> Result<Self, String> {
        Ok(Self {
            x_two_gamma: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::gamma_dh_operator(2),
                0,
            )?,
            x_two_hook: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::hook_projector_operator(2),
                0,
            )?,
            x_five_gamma: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::gamma_dh_operator(5),
                0,
            )?,
            x_five_hook: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::hook_projector_operator(5),
                0,
            )?,
            spinorial_connection: exact_cuda_sparse::ExactCudaSparseOperator::new(
                physical::cached_c_alpha_b_c_to_spinorial_connection_operator(),
                0,
            )?,
            j_one_anholonomy: exact_cuda_sparse::ExactCudaSparseOperator::new(
                physical::cached_c_alpha_beta_gamma_to_j_one_operator(),
                0,
            )?,
            j_one_connection: exact_cuda_sparse::ExactCudaSparseOperator::new(
                physical::cached_spinorial_connection_to_j_one_operator(),
                0,
            )?,
            j_two: exact_cuda_sparse::ExactCudaSparseOperator::new(
                physical::cached_c_alpha_b_c_to_j_operator(),
                0,
            )?,
            d_spinorial_connection: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &prolong_sparse_operator(
                    physical::cached_c_alpha_b_c_to_spinorial_connection_operator(),
                ),
                0,
            )?,
            d_j_one_anholonomy: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &prolong_sparse_operator(physical::cached_c_alpha_beta_gamma_to_j_one_operator()),
                0,
            )?,
            d_j_one_connection: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &prolong_sparse_operator(physical::cached_spinorial_connection_to_j_one_operator()),
                0,
            )?,
            d_j_two: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &prolong_sparse_operator(physical::cached_c_alpha_b_c_to_j_operator()),
                0,
            )?,
            bosonic_connection: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::d_spinorial_connection_to_bosonic_connection_operator(),
                0,
            )?,
            t_from_bosonic_connection: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::bosonic_connection_to_t_alpha_e_gamma_operator(),
                0,
            )?,
            w_2021_from_t: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::t_alpha_e_gamma_to_w_operator(),
                0,
            )?,
            w_2021_from_d_j: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::d_j_to_w_operator(),
                0,
            )?,
            w_2001_from_t: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::t_alpha_e_gamma_to_w_2001_operator(),
                0,
            )?,
            w_2001_from_d_j: exact_cuda_sparse::ExactCudaSparseOperator::new(
                &physical::d_j_two_to_w_2001_operator(),
                0,
            )?,
        })
    }

    fn apply_x_batch(
        &self,
        d_h: &[BTreeMap<usize, ExactQi>],
    ) -> Result<Vec<(BTreeMap<usize, ExactQi>, BTreeMap<usize, ExactQi>)>, String> {
        let (mut x_two, _) = self
            .x_two_gamma
            .apply_composed_batch(&self.x_two_hook, d_h)?;
        let (mut x_five, _) = self
            .x_five_gamma
            .apply_composed_batch(&self.x_five_hook, d_h)?;
        let normalization = num_rational::Ratio::new(1, 16);
        for values in &mut x_two {
            for value in values.values_mut() {
                *value = value.scaled(&normalization);
            }
        }
        for values in &mut x_five {
            for value in values.values_mut() {
                *value = value.times_i().scaled(&normalization);
            }
        }
        Ok(x_two.into_iter().zip(x_five).collect())
    }

    fn assemble_first_jet_batch(
        &self,
        batch: Vec<FirstSuperspaceJetInput>,
        include_w_2001: bool,
    ) -> Result<Vec<FirstSuperspaceJetOutput>, String> {
        let lane_count = batch.len();
        let mut d_c_spinor = Vec::with_capacity(lane_count);
        let mut d_c_vector = Vec::with_capacity(lane_count);
        let mut c_mixed = Vec::with_capacity(lane_count);
        for input in batch {
            d_c_spinor.push(input.d_c_alpha_beta_gamma);
            d_c_vector.push(input.d_c_alpha_b_c);
            c_mixed.push(input.c_alpha_a_gamma);
        }
        let (d_spinorial_connection, _) = self.d_spinorial_connection.apply_batch(&d_c_vector)?;
        let (mut d_j_one, _) = self.d_j_one_anholonomy.apply_batch(&d_c_spinor)?;
        let (d_j_one_connection, _) = self
            .d_j_one_connection
            .apply_batch(&d_spinorial_connection)?;
        for (target, source) in d_j_one.iter_mut().zip(d_j_one_connection) {
            merge_exact_sparse(target, source);
        }
        let (d_j_two, _) = self.d_j_two.apply_batch(&d_c_vector)?;
        let d_j_plus = d_j_one
            .iter()
            .zip(&d_j_two)
            .map(|(one, two)| physical::apply_d_j_plus(one, two))
            .collect::<Vec<_>>();
        let (bosonic_connection, _) = self
            .bosonic_connection
            .apply_batch(&d_spinorial_connection)?;
        let (mut t_alpha_a_gamma, _) = self
            .t_from_bosonic_connection
            .apply_batch(&bosonic_connection)?;
        for (target, source) in t_alpha_a_gamma.iter_mut().zip(c_mixed) {
            merge_exact_sparse(target, source);
        }
        let (mut w_2021, _) = self.w_2021_from_t.apply_batch(&t_alpha_a_gamma)?;
        let (w_2021_d_j, _) = self.w_2021_from_d_j.apply_batch(&d_j_plus)?;
        for (target, source) in w_2021.iter_mut().zip(w_2021_d_j) {
            merge_exact_sparse(target, source);
        }
        let w_2001 = if include_w_2001 {
            let (mut w_2001, _) = self.w_2001_from_t.apply_batch(&t_alpha_a_gamma)?;
            let (w_2001_d_j, _) = self.w_2001_from_d_j.apply_batch(&d_j_two)?;
            for (target, source) in w_2001.iter_mut().zip(w_2001_d_j) {
                merge_exact_sparse(target, source);
            }
            w_2001
        } else {
            vec![BTreeMap::new(); lane_count]
        };
        Ok(d_spinorial_connection
            .into_iter()
            .zip(bosonic_connection)
            .zip(d_j_one)
            .zip(d_j_two)
            .zip(d_j_plus)
            .zip(t_alpha_a_gamma)
            .zip(w_2001)
            .zip(w_2021)
            .map(
                |(
                    (
                        (
                            (
                                (((d_spinorial_connection, bosonic_connection), d_j_one), d_j_two),
                                d_j_plus,
                            ),
                            t_alpha_a_gamma,
                        ),
                        w_2001,
                    ),
                    w_2021,
                )| {
                    if include_w_2001 {
                        FirstSuperspaceJetOutput {
                            d_spinorial_connection,
                            bosonic_connection,
                            d_j_one,
                            d_j_two,
                            d_j_plus,
                            t_alpha_a_gamma,
                            w_2001,
                            w_2021,
                        }
                    } else {
                        FirstSuperspaceJetOutput {
                            d_spinorial_connection: BTreeMap::new(),
                            bosonic_connection: BTreeMap::new(),
                            d_j_one: BTreeMap::new(),
                            d_j_two: BTreeMap::new(),
                            d_j_plus: BTreeMap::new(),
                            t_alpha_a_gamma: BTreeMap::new(),
                            w_2001,
                            w_2021,
                        }
                    }
                },
            )
            .collect())
    }
}

#[cfg(feature = "cuda")]
thread_local! {
    static GEOMETRY_LEVEL_CUDA_OPERATORS: std::cell::RefCell<Option<GeometryLevelCudaOperators>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
fn merge_exact_sparse(target: &mut BTreeMap<usize, ExactQi>, source: BTreeMap<usize, ExactQi>) {
    for (index, value) in source {
        let entry = target.entry(index).or_insert_with(ExactQi::zero);
        entry.add_assign(&value);
        if entry.is_zero() {
            target.remove(&index);
        }
    }
}

#[cfg(feature = "cuda")]
fn assemble_geometry_level_physical_f_cuda_batch(
    inputs: &[GeometryLevelPhysicalFInput],
) -> Result<Vec<GeometryLevelPhysicalFOutput>, String> {
    assemble_geometry_level_physical_f_cuda_batch_internal(inputs.to_vec(), true, true)
}

#[cfg(feature = "cuda")]
fn assemble_geometry_level_physical_f_cuda_batch_for_visitor(
    inputs: Vec<GeometryLevelPhysicalFInput>,
    include_w_2001: bool,
) -> Result<Vec<GeometryLevelPhysicalFOutput>, String> {
    assemble_geometry_level_physical_f_cuda_batch_internal(inputs, include_w_2001, include_w_2001)
}

#[cfg(feature = "cuda")]
fn assemble_geometry_level_physical_f_cuda_batch_internal(
    inputs: Vec<GeometryLevelPhysicalFInput>,
    retain_compensator_images: bool,
    include_w_2001: bool,
) -> Result<Vec<GeometryLevelPhysicalFOutput>, String> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    for input in &inputs {
        validate_sparse_dimension("d_h", &input.d_h, physical::DH_DIMENSION)?;
        validate_sparse_dimension(
            "c_alpha_beta_gamma",
            &input.c_alpha_beta_gamma,
            physical::SPINOR_ANHOLONOMY_DIMENSION,
        )?;
        validate_sparse_dimension(
            "c_alpha_b_c",
            &input.c_alpha_b_c,
            physical::C_ALPHA_VECTOR_VECTOR_DIMENSION,
        )?;
    }
    let lane_count = inputs.len();
    let mut c_spinor = Vec::with_capacity(lane_count);
    let mut d_h = Vec::with_capacity(lane_count);
    let mut c_vector = Vec::with_capacity(lane_count);
    let mut first_jet_inputs = Vec::with_capacity(lane_count);
    for input in inputs {
        c_spinor.push(input.c_alpha_beta_gamma);
        d_h.push(input.d_h);
        c_vector.push(input.c_alpha_b_c);
        first_jet_inputs.push(input.first_jet);
    }
    GEOMETRY_LEVEL_CUDA_OPERATORS.with(|cached| {
        let mut cached = cached.borrow_mut();
        if cached.is_none() {
            *cached = Some(GeometryLevelCudaOperators::new()?);
        }
        let operators = cached.as_ref().unwrap();
        let (spinorial_connection, _) = operators.spinorial_connection.apply_batch(&c_vector)?;
        let (mut j_one, _) = operators.j_one_anholonomy.apply_batch(&c_spinor)?;
        let (j_one_connection, _) = operators
            .j_one_connection
            .apply_batch(&spinorial_connection)?;
        let (j_two, _) = operators.j_two.apply_batch(&c_vector)?;
        let first_jet = operators.assemble_first_jet_batch(first_jet_inputs, include_w_2001)?;
        let x = if retain_compensator_images {
            d_h.iter()
                .map(physical::apply_leading_physical_x)
                .collect::<Vec<_>>()
        } else {
            operators
                .apply_x_batch(&d_h)?
                .into_iter()
                .map(|(x_two_11000, x_five_10002)| PhysicalXImage {
                    x_two_11000,
                    x_five_10002,
                    x_two_compensators: physical::EliminatedCompensatorImage {
                        trace_image: BTreeMap::new(),
                        exterior_image: BTreeMap::new(),
                        combined_image: BTreeMap::new(),
                    },
                    x_five_compensators: physical::EliminatedCompensatorImage {
                        trace_image: BTreeMap::new(),
                        exterior_image: BTreeMap::new(),
                        combined_image: BTreeMap::new(),
                    },
                })
                .collect::<Vec<_>>()
        };
        for (target, source) in j_one.iter_mut().zip(j_one_connection) {
            merge_exact_sparse(target, source);
        }
        let outputs = spinorial_connection
            .into_iter()
            .zip(j_one)
            .zip(j_two)
            .zip(first_jet)
            .zip(x)
            .map(|((((spinorial_connection, j_one), j_two), first_jet), x)| {
                let j_minus = physical::apply_j_minus(&j_one, &j_two);
                if retain_compensator_images {
                    let j_plus = physical::apply_j_plus(&j_one, &j_two);
                    GeometryLevelPhysicalFOutput {
                        x,
                        spinorial_connection,
                        j_one,
                        j_two,
                        j_plus,
                        j_minus,
                        first_jet,
                    }
                } else {
                    GeometryLevelPhysicalFOutput {
                        x,
                        spinorial_connection: BTreeMap::new(),
                        j_one: BTreeMap::new(),
                        j_two: BTreeMap::new(),
                        j_plus: BTreeMap::new(),
                        j_minus,
                        first_jet: FirstSuperspaceJetOutput {
                            d_spinorial_connection: BTreeMap::new(),
                            bosonic_connection: BTreeMap::new(),
                            d_j_one: BTreeMap::new(),
                            d_j_two: BTreeMap::new(),
                            d_j_plus: BTreeMap::new(),
                            t_alpha_a_gamma: BTreeMap::new(),
                            w_2001: first_jet.w_2001,
                            w_2021: first_jet.w_2021,
                        },
                    }
                }
            })
            .collect();
        Ok(outputs)
    })
}

#[cfg(feature = "cuda")]
fn complete_f_cuda_enabled() -> bool {
    std::env::var("ADYNKRA_COMPLETE_F_CUDA")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
}

type PolynomialMap = BTreeMap<usize, CanonicalSuperPolynomial>;
type MonomialSlices = BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>;
type TargetOperatorColumns = Vec<Vec<(usize, ExactPolynomialCoefficient)>>;

fn index_target_operator(operator: &ExactPolynomialMatrix) -> TargetOperatorColumns {
    (0..operator.columns())
        .map(|column| operator.column_terms(column))
        .collect()
}

fn rarita_curvature_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::RaritaSchwinger).curvature)
    })
}

fn rarita_bianchi_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::RaritaSchwinger).bianchi)
    })
}

fn rarita_curvature_to_euler_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(
            &target_sector_complex(TargetSector::RaritaSchwinger).curvature_to_euler,
        )
    })
}

fn rarita_noether_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::RaritaSchwinger).noether)
    })
}

fn graviton_bianchi_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::Graviton).bianchi)
    })
}

fn graviton_curvature_to_euler_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::Graviton).curvature_to_euler)
    })
}

fn graviton_noether_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::Graviton).noether)
    })
}

fn four_form_bianchi_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::FourForm).bianchi)
    })
}

fn four_form_curvature_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::FourForm).curvature)
    })
}

fn four_form_curvature_to_euler_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::FourForm).curvature_to_euler)
    })
}

fn four_form_noether_columns() -> &'static TargetOperatorColumns {
    static COLUMNS: OnceLock<TargetOperatorColumns> = OnceLock::new();
    COLUMNS.get_or_init(|| {
        index_target_operator(&target_sector_complex(TargetSector::FourForm).noether)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub enum FrameComposedPhysicalFSector {
    XTwo,
    XFive,
    JOne,
    JTwo,
    JPlus,
    JMinus,
    MixedTorsion,
    W2001,
    W2021,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameComposedPhysicalFEntry {
    pub sector: FrameComposedPhysicalFSector,
    pub coordinate: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrameComposedPhysicalFStats {
    pub monomial_slices: usize,
    pub emitted_by_sector: BTreeMap<FrameComposedPhysicalFSector, usize>,
}

fn add_polynomial_coefficient(
    output: &mut PolynomialMap,
    coordinate: usize,
    monomial: OrderedSuperderivativeMonomial,
    coefficient: ExactQi,
) {
    if coefficient.is_zero() {
        return;
    }
    let polynomial = output.entry(coordinate).or_default();
    polynomial.add_term(monomial, coefficient);
    if polynomial.terms.is_empty() {
        output.remove(&coordinate);
    }
}

fn transpose_polynomials(input: &PolynomialMap) -> MonomialSlices {
    let mut output = MonomialSlices::new();
    for (&coordinate, polynomial) in input {
        for (monomial, coefficient) in &polynomial.terms {
            output
                .entry(monomial.clone())
                .or_default()
                .insert(coordinate, coefficient.clone());
        }
    }
    output
}

fn emit_exact_sector<F>(
    sector: FrameComposedPhysicalFSector,
    values: BTreeMap<usize, ExactQi>,
    monomial: &OrderedSuperderivativeMonomial,
    stats: &mut FrameComposedPhysicalFStats,
    emit: &mut F,
) -> Result<(), String>
where
    F: FnMut(FrameComposedPhysicalFEntry) -> Result<(), String>,
{
    for (coordinate, coefficient) in values {
        if coefficient.is_zero() {
            continue;
        }
        emit(FrameComposedPhysicalFEntry {
            sector,
            coordinate,
            monomial: monomial.clone(),
            coefficient,
        })?;
        *stats.emitted_by_sector.entry(sector).or_default() += 1;
    }
    Ok(())
}

/// Compose one exact H/scale/p=2 frame jet through every currently source-fixed
/// X, J, torsion, and W operator.
pub fn visit_frame_composed_physical_f<F>(
    frame_input: &LinearizedFrameSuperfields,
    emit: F,
) -> Result<FrameComposedPhysicalFStats, String>
where
    F: FnMut(FrameComposedPhysicalFEntry) -> Result<(), String>,
{
    #[cfg(feature = "cuda")]
    let use_cuda = complete_f_cuda_enabled();
    #[cfg(not(feature = "cuda"))]
    let use_cuda = false;
    visit_frame_composed_physical_f_internal(frame_input, use_cuda, true, emit)
}

fn visit_frame_composed_physical_f_internal<F>(
    frame_input: &LinearizedFrameSuperfields,
    use_cuda: bool,
    include_w_2001: bool,
    mut emit: F,
) -> Result<FrameComposedPhysicalFStats, String>
where
    F: FnMut(FrameComposedPhysicalFEntry) -> Result<(), String>,
{
    let mut geometry = BTreeMap::<ConstrainedGeometryJetSector, PolynomialMap>::new();
    visit_constrained_geometry_jet(frame_input, |entry| {
        add_polynomial_coefficient(
            geometry.entry(entry.sector).or_default(),
            entry.coordinate,
            entry.monomial,
            entry.coefficient,
        );
        Ok(())
    })?;
    let mut d_h = PolynomialMap::new();
    visit_linearized_frame_jet(frame_input, |entry| {
        if entry.sector == LinearizedFrameJetSector::DH {
            d_h.insert(entry.coordinate, entry.polynomial);
        }
        Ok(())
    })?;

    let d_h = transpose_polynomials(&d_h);
    let c_spinor = transpose_polynomials(
        geometry
            .get(&ConstrainedGeometryJetSector::CAlphaBetaGamma)
            .unwrap_or(&PolynomialMap::new()),
    );
    let c_vector = transpose_polynomials(
        geometry
            .get(&ConstrainedGeometryJetSector::CAlphaVectorVector)
            .unwrap_or(&PolynomialMap::new()),
    );
    let c_mixed = transpose_polynomials(
        geometry
            .get(&ConstrainedGeometryJetSector::CAlphaVectorGamma)
            .unwrap_or(&PolynomialMap::new()),
    );
    let d_c_spinor = transpose_polynomials(
        geometry
            .get(&ConstrainedGeometryJetSector::DCAlphaBetaGamma)
            .unwrap_or(&PolynomialMap::new()),
    );
    let d_c_vector = transpose_polynomials(
        geometry
            .get(&ConstrainedGeometryJetSector::DCAlphaVectorVector)
            .unwrap_or(&PolynomialMap::new()),
    );
    let monomials = d_h
        .keys()
        .chain(c_spinor.keys())
        .chain(c_vector.keys())
        .chain(c_mixed.keys())
        .chain(d_c_spinor.keys())
        .chain(d_c_vector.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let empty = BTreeMap::new();
    let mut stats = FrameComposedPhysicalFStats {
        monomial_slices: monomials.len(),
        emitted_by_sector: BTreeMap::new(),
    };
    let inputs = monomials
        .iter()
        .map(|monomial| GeometryLevelPhysicalFInput {
            d_h: d_h.get(&monomial).unwrap_or(&empty).clone(),
            c_alpha_beta_gamma: c_spinor.get(&monomial).unwrap_or(&empty).clone(),
            c_alpha_b_c: c_vector.get(&monomial).unwrap_or(&empty).clone(),
            first_jet: FirstSuperspaceJetInput {
                d_c_alpha_beta_gamma: d_c_spinor.get(&monomial).unwrap_or(&empty).clone(),
                d_c_alpha_b_c: d_c_vector.get(&monomial).unwrap_or(&empty).clone(),
                c_alpha_a_gamma: c_mixed.get(&monomial).unwrap_or(&empty).clone(),
            },
        })
        .collect::<Vec<_>>();
    let outputs = {
        #[cfg(feature = "cuda")]
        {
            if use_cuda {
                assemble_geometry_level_physical_f_cuda_batch_for_visitor(inputs, include_w_2001)?
            } else {
                inputs
                    .iter()
                    .map(|input| assemble_geometry_level_physical_f_internal(input, include_w_2001))
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = use_cuda;
            inputs
                .iter()
                .map(|input| assemble_geometry_level_physical_f_internal(input, include_w_2001))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    for (monomial, output) in monomials.into_iter().zip(outputs) {
        emit_exact_sector(
            FrameComposedPhysicalFSector::XTwo,
            output.x.x_two_11000,
            &monomial,
            &mut stats,
            &mut emit,
        )?;
        emit_exact_sector(
            FrameComposedPhysicalFSector::XFive,
            output.x.x_five_10002,
            &monomial,
            &mut stats,
            &mut emit,
        )?;
        if include_w_2001 {
            for (sector, values) in [
                (FrameComposedPhysicalFSector::JOne, output.j_one),
                (FrameComposedPhysicalFSector::JTwo, output.j_two),
                (FrameComposedPhysicalFSector::JPlus, output.j_plus),
            ] {
                emit_exact_sector(sector, values, &monomial, &mut stats, &mut emit)?;
            }
        }
        emit_exact_sector(
            FrameComposedPhysicalFSector::JMinus,
            output.j_minus,
            &monomial,
            &mut stats,
            &mut emit,
        )?;
        if include_w_2001 {
            emit_exact_sector(
                FrameComposedPhysicalFSector::MixedTorsion,
                output.first_jet.t_alpha_a_gamma,
                &monomial,
                &mut stats,
                &mut emit,
            )?;
            emit_exact_sector(
                FrameComposedPhysicalFSector::W2001,
                output.first_jet.w_2001,
                &monomial,
                &mut stats,
                &mut emit,
            )?;
        }
        emit_exact_sector(
            FrameComposedPhysicalFSector::W2021,
            output.first_jet.w_2021,
            &monomial,
            &mut stats,
            &mut emit,
        )?;
    }
    Ok(stats)
}

/// Compose the canonical `H_hat=P_320 H` representative.  Local gamma-trace
/// and p=2 Lorentz-frame coordinates are removed at the typed input boundary,
/// while the raw frame visitor remains available for explicit orbit audits.
pub fn visit_gauge_fixed_physical_f<F>(
    frame_input: &LinearizedFrameSuperfields,
    emit: F,
) -> Result<FrameComposedPhysicalFStats, String>
where
    F: FnMut(FrameComposedPhysicalFEntry) -> Result<(), String>,
{
    let representative = canonical_physical_frame_representative(frame_input)?;
    visit_frame_composed_physical_f(&representative, emit)
}

fn visit_gauge_fixed_physical_f_production<F>(
    frame_input: &LinearizedFrameSuperfields,
    emit: F,
) -> Result<FrameComposedPhysicalFStats, String>
where
    F: FnMut(FrameComposedPhysicalFEntry) -> Result<(), String>,
{
    #[cfg(feature = "cuda")]
    let use_cuda = complete_f_cuda_enabled();
    #[cfg(not(feature = "cuda"))]
    let use_cuda = false;
    let representative = canonical_physical_frame_representative(frame_input)?;
    visit_frame_composed_physical_f_internal(&representative, use_cuda, false, emit)
}

/// Coefficient-free label for one entry in the exact source-fixed frame
/// curvature stream.  Keeping the coefficient separate makes the target
/// adapter an ordinary linear map and prevents accidental double scaling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameComposedPhysicalFCoordinate {
    pub sector: FrameComposedPhysicalFSector,
    pub coordinate: usize,
    pub monomial: OrderedSuperderivativeMonomial,
}

impl FrameComposedPhysicalFEntry {
    pub fn coordinate_label(&self) -> FrameComposedPhysicalFCoordinate {
        FrameComposedPhysicalFCoordinate {
            sector: self.sector,
            coordinate: self.coordinate,
            monomial: self.monomial.clone(),
        }
    }
}

/// Conditional coordinate adapter for identifying the bosonic lowest
/// component of the modern all-real-gamma `W_[4]` superfield with a target
/// four-form. The current composed raw-W stream fails the target Bianchi test
/// on a nonzero H canary, so production persistence keeps it auxiliary and
/// never invokes this adapter as a physical identification. Spinorial
/// descendants and the X/J/T auxiliary sectors are rejected.
#[derive(Clone, Copy, Debug, Default)]
pub struct W2021FourFormTargetAdapter;

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1_usize, |value, index| value * (n - index) / (index + 1))
}

fn w_source_four_form_indices(source_ordinal: usize) -> Result<[usize; 4], String> {
    let mask = (0_u16..(1_u16 << physical::VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 4)
        .nth(source_ordinal)
        .ok_or_else(|| {
            format!(
                "W four-form coordinate {source_ordinal} is outside dimension {}",
                physical::W_FOUR_FORM_DIMENSION
            )
        })?;
    let indices = (0..physical::VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    Ok([indices[0], indices[1], indices[2], indices[3]])
}

fn target_four_form_lexicographic_ordinal(indices: [usize; 4]) -> usize {
    let mut ordinal = 0_usize;
    let mut next = 0_usize;
    for (position, value) in indices.into_iter().enumerate() {
        for candidate in next..value {
            ordinal += binomial(physical::VECTOR_DIMENSION - candidate - 1, 4 - position - 1);
        }
        next = value + 1;
    }
    ordinal
}

fn target_three_form_lexicographic_ordinal(indices: [usize; 3]) -> usize {
    let mut ordinal = 0_usize;
    let mut next = 0_usize;
    for (position, value) in indices.into_iter().enumerate() {
        for candidate in next..value {
            ordinal += binomial(physical::VECTOR_DIMENSION - candidate - 1, 3 - position - 1);
        }
        next = value + 1;
    }
    ordinal
}

fn psi_three_mask_to_target_ordinal(mask: u16) -> Result<usize, String> {
    if mask.count_ones() != 3 || mask >= (1_u16 << physical::VECTOR_DIMENSION) {
        return Err(format!(
            "Eq. (40) Psi_[3] mask {mask:#x} is not a canonical eleven-dimensional three-form"
        ));
    }
    let indices = (0..physical::VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    Ok(target_three_form_lexicographic_ordinal([
        indices[0], indices[1], indices[2],
    ]))
}

impl PhysicalFTargetAdapter for W2021FourFormTargetAdapter {
    type SourceCoordinate = FrameComposedPhysicalFCoordinate;
    type Coefficient = ExactQi;

    fn descriptor(&self) -> PhysicalFAdapterDescriptor {
        PhysicalFAdapterDescriptor {
            schema_version: "adynkra-11d-w2021-four-form-target-adapter-v1".to_string(),
            source_basis:
                "Cartesian Majorana W_[a1a2a3a4] lowest component in increasing-mask order"
                    .to_string(),
            target_basis: "Abelian four-form curvature in increasing Lorentz-index order"
                .to_string(),
            generic_formal_momentum_complete: true,
            physical_source_map_complete: false,
        }
    }

    fn apply_source_coordinate(
        &self,
        source: &Self::SourceCoordinate,
        coefficient: &Self::Coefficient,
    ) -> Result<Vec<(TargetCurvatureCoordinate, Self::Coefficient)>, String> {
        if source.sector != FrameComposedPhysicalFSector::W2021 {
            return Err(format!(
                "sector {:?} is not the W2021 four-form curvature",
                source.sector
            ));
        }
        let source_indices = w_source_four_form_indices(source.coordinate)?;
        if source.monomial.exterior_spinor_mask != 0 {
            return Err(
                "spinorial W descendants require the still-missing gravitino/Riemann component adapter"
                    .to_string(),
            );
        }
        let momentum = MomentumMonomial::try_from_u16(source.monomial.momentum.exponents)?;
        Ok(vec![(
            TargetCurvatureCoordinate {
                sector: TargetCurvatureSector::FourForm,
                component: target_four_form_lexicographic_ordinal(source_indices),
                momentum,
            },
            coefficient.clone(),
        )])
    }
}

/// One exact term in the off-shell Eq. (25) frame-to-Riemann stream.
///
/// The ordered spinor derivative remains part of the source differential
/// monomial.  The two spacetime derivatives introduced by the independent
/// target graviton-curvature operator are multiplied into its formal momentum
/// monomial.  This avoids conflating the off-shell Riemann tensor with the
/// conditional on-shell statement that `D^2 W_[4]` contains the Weyl tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectLinearizedRiemannEntry {
    pub component: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectLinearizedRiemannStats {
    pub source_frame_terms: usize,
    pub symmetric_metric_terms: usize,
    pub riemann_terms: usize,
}

/// One exact term in the source-fixed Eq. (25) vector-spinor curl.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectLinearizedGravitinoCurlEntry {
    pub component: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectLinearizedGravitinoCurlStats {
    pub fermionic_frame_terms: usize,
    pub gravitino_curl_terms: usize,
    pub euler_terms: usize,
    pub eq29_torsion_residual_terms: usize,
    pub bianchi_residual_terms: usize,
    pub noether_residual_terms: usize,
}

/// Exact closed four-form candidate obtained from the source-derived Eq. (40)
/// holonomy `Psi_[3]` through the independent Abelian target curvature map.
/// In the pinned convention,
/// `Psi_abc=(eta_cc R_ab^c-eta_bb R_ac^b+eta_aa R_bc^a)/48`, with
/// `R_ab^c=(gamma_ab)^{gamma delta}D_gamma H_delta^c`. This is the unique
/// multiplicity-one Lambda-three projection selected by the Eq. (40)
/// conventional constraint. The source does not identify that holonomy with
/// the physical component potential, so it is not claimed to be the unique
/// general `H -> A_[3]` map or a physical relative normalization of raw W.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectCandidateFourFormEntry {
    pub component: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectCandidateFourFormStats {
    pub psi_three_potential_terms: usize,
    pub four_form_curvature_terms: usize,
    pub euler_terms: usize,
    pub bianchi_residual_terms: usize,
    pub noether_residual_terms: usize,
    pub raw_w_bianchi_residual_terms: usize,
    pub raw_w_comparison_residual_terms: usize,
}

fn gauge_fixed_candidate_four_form_curvature(
    frame_input: &LinearizedFrameSuperfields,
    raw_entries: &[GaugeFixedInvariantOutputEntry],
) -> Result<
    (
        DirectCandidateFourFormStats,
        Vec<DirectCandidateFourFormEntry>,
    ),
    String,
> {
    let representative = canonical_physical_frame_representative(frame_input)?;
    let mut d_h_slices = MonomialSlices::new();
    visit_d_h_jet(&representative, |entry| {
        for (monomial, coefficient) in entry.polynomial.terms {
            let slice = d_h_slices.entry(monomial).or_default();
            let value = slice.entry(entry.coordinate).or_insert_with(ExactQi::zero);
            value.add_assign(&coefficient);
            if value.is_zero() {
                slice.remove(&entry.coordinate);
            }
        }
        Ok(())
    })?;
    d_h_slices.retain(|_, slice| !slice.is_empty());
    let mut potential = BTreeMap::new();
    for (monomial, d_h) in d_h_slices {
        for (mask, coefficient) in physical::solve_conventional_compensators(&d_h).psi_three {
            add_polynomial_map_value(
                &mut potential,
                (psi_three_mask_to_target_ordinal(mask)?, monomial.clone()),
                coefficient,
            );
        }
    }
    let curvature = apply_target_polynomial_columns(four_form_curvature_columns(), &potential)?;
    let bianchi_residual =
        apply_target_polynomial_columns(four_form_bianchi_columns(), &curvature)?;
    if !bianchi_residual.is_empty() {
        return Err(format!(
            "Eq. (40) Psi_[3] candidate four-form violates the target Bianchi identity in {} coordinates",
            bianchi_residual.len()
        ));
    }
    let euler =
        apply_target_polynomial_columns(four_form_curvature_to_euler_columns(), &curvature)?;
    let noether_residual = apply_target_polynomial_columns(four_form_noether_columns(), &euler)?;
    if !noether_residual.is_empty() {
        return Err(format!(
            "Eq. (40) Psi_[3] candidate four-form violates Noether(Euler) in {} coordinates",
            noether_residual.len()
        ));
    }

    let mut raw_w = BTreeMap::new();
    for entry in raw_entries {
        if entry.sector == GaugeFixedInvariantOutputSector::W2021Raw {
            let component = target_four_form_lexicographic_ordinal(w_source_four_form_indices(
                entry.coordinate,
            )?);
            add_polynomial_map_value(
                &mut raw_w,
                (component, entry.monomial.clone()),
                entry.coefficient.clone(),
            );
        }
    }
    let raw_w_bianchi_residual_terms =
        apply_target_polynomial_columns(four_form_bianchi_columns(), &raw_w)?;
    let mut raw_minus_candidate = raw_w.clone();
    for (key, coefficient) in &curvature {
        add_polynomial_map_value(
            &mut raw_minus_candidate,
            key.clone(),
            coefficient.scaled(&num_rational::Ratio::from_integer(-1)),
        );
    }
    let residual_bianchi =
        apply_target_polynomial_columns(four_form_bianchi_columns(), &raw_minus_candidate)?;
    if residual_bianchi != raw_w_bianchi_residual_terms {
        return Err(
            "raw W minus Eq. (40) Psi_[3] candidate changed the four-form Bianchi residual"
                .to_string(),
        );
    }
    let raw_w_comparison_residual_terms = raw_minus_candidate.len();
    let stats = DirectCandidateFourFormStats {
        psi_three_potential_terms: potential.len(),
        four_form_curvature_terms: curvature.len(),
        euler_terms: euler.len(),
        bianchi_residual_terms: 0,
        noether_residual_terms: 0,
        raw_w_bianchi_residual_terms: raw_w_bianchi_residual_terms.len(),
        raw_w_comparison_residual_terms,
    };
    let entries = curvature
        .into_iter()
        .map(
            |((component, monomial), coefficient)| DirectCandidateFourFormEntry {
                component,
                monomial,
                coefficient,
            },
        )
        .collect();
    Ok((stats, entries))
}

fn eq25_fermionic_frame_source_polynomials(
    representative: &LinearizedFrameSuperfields,
) -> Result<(PolynomialMap, PolynomialMap), String> {
    let mut d_delta = PolynomialMap::new();
    visit_constrained_d_delta(representative, |entry| {
        if entry.sector == ConstrainedGeometryJetSector::DDelta {
            add_polynomial_coefficient(
                &mut d_delta,
                entry.coordinate,
                entry.monomial,
                entry.coefficient,
            );
        }
        Ok(())
    })?;
    let mut d_scale = PolynomialMap::new();
    for derivative in 0..physical::SPINOR_DIMENSION {
        let polynomial = left_multiply_d(derivative, &representative.scale)?;
        if !polynomial.terms.is_empty() {
            d_scale.insert(derivative, polynomial);
        }
    }
    Ok((d_delta, d_scale))
}

/// Stream the direct off-shell linearized gravitino curl on the canonical
/// gauge-fixed frame section. hep-th/0101037 Eq. (25) first constructs the
/// vector-spinor from `D Delta` and `D Psi`; the independent target
/// Rarita-Schwinger curvature then forms the full curl. Every polynomial
/// coefficient is checked entrywise against the unnormalized Eq. (29)
/// anholonomy, which equals the conventional torsion by hep-th/0107155
/// Eq. (3.2f). The source bracket normalization is fixed by hep-th/0101037
/// Eqs. (7)-(8).
pub fn visit_gauge_fixed_linearized_gravitino_curl<F>(
    frame_input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<DirectLinearizedGravitinoCurlStats, String>
where
    F: FnMut(DirectLinearizedGravitinoCurlEntry) -> Result<(), String>,
{
    let representative = canonical_physical_frame_representative(frame_input)?;
    let (d_delta, d_scale) = eq25_fermionic_frame_source_polynomials(&representative)?;
    let d_delta = transpose_polynomials(&d_delta);
    let d_scale = transpose_polynomials(&d_scale);
    let monomials = d_delta
        .keys()
        .chain(d_scale.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let empty = BTreeMap::new();
    let target_curvature = rarita_curvature_columns();
    let mut curl = BTreeMap::new();
    let mut fermionic_frame_terms = 0_usize;

    for monomial in monomials {
        let source = physical::Eq25FermionicFrameInput {
            d_delta: d_delta.get(&monomial).unwrap_or(&empty).clone(),
            d_scale: d_scale.get(&monomial).unwrap_or(&empty).clone(),
        };
        let frame = physical::apply_eq25_fermionic_frame(&source)?;
        fermionic_frame_terms = fermionic_frame_terms
            .checked_add(frame.len())
            .ok_or_else(|| "direct gravitino frame term count overflow".to_string())?;
        let mut monomial_curl = BTreeMap::new();
        for (component, coefficient) in &frame {
            for (curl_component, operator_term) in &target_curvature[*component] {
                let output_monomial =
                    multiply_ordered_momentum(&monomial, &operator_term.monomial)?;
                add_polynomial_map_value(
                    &mut monomial_curl,
                    (*curl_component, output_monomial),
                    multiply_exact_qi_by_public(coefficient, &operator_term),
                );
            }
        }
        let mut monomial_eq29 = BTreeMap::new();
        for momentum_axis in 0..physical::VECTOR_DIMENSION {
            let momentum = MomentumMonomial::variable(momentum_axis);
            let output_monomial = multiply_ordered_momentum(&monomial, &momentum)?;
            for (component, coefficient) in
                physical::apply_eq29_fermionic_anholonomy(&source, momentum_axis)?
            {
                add_polynomial_map_value(
                    &mut monomial_eq29,
                    (component, output_monomial.clone()),
                    coefficient,
                );
            }
        }
        let residual_terms = polynomial_map_difference_count(&monomial_curl, &monomial_eq29);
        if residual_terms != 0 {
            return Err(format!(
                "direct gravitino curl disagrees with Eq. (29) torsion in {residual_terms} coordinates at source monomial {monomial:?}"
            ));
        }
        for (key, coefficient) in monomial_curl {
            add_polynomial_map_value(&mut curl, key, coefficient);
        }
    }
    let eq29_torsion_residual_terms = 0;
    let bianchi_residual_terms = gravitino_curl_bianchi_residual(&curl)?.len();
    if bianchi_residual_terms != 0 {
        return Err(format!(
            "direct gravitino curl violates the target Bianchi identity in {bianchi_residual_terms} coordinates"
        ));
    }
    let euler = gravitino_curl_euler_image(&curl)?;
    let noether_residual_terms =
        apply_target_polynomial_columns(rarita_noether_columns(), &euler)?.len();
    if noether_residual_terms != 0 {
        return Err(format!(
            "direct gravitino curl violates Noether(Euler) in {noether_residual_terms} coordinates"
        ));
    }
    let stats = DirectLinearizedGravitinoCurlStats {
        fermionic_frame_terms,
        gravitino_curl_terms: curl.len(),
        euler_terms: euler.len(),
        eq29_torsion_residual_terms,
        bianchi_residual_terms,
        noether_residual_terms,
    };
    for ((component, monomial), coefficient) in curl {
        emit(DirectLinearizedGravitinoCurlEntry {
            component,
            monomial,
            coefficient,
        })?;
    }
    Ok(stats)
}

fn add_polynomial_map_value(
    output: &mut BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
    key: (usize, OrderedSuperderivativeMonomial),
    coefficient: ExactQi,
) {
    if coefficient.is_zero() {
        return;
    }
    let value = output.entry(key.clone()).or_insert_with(ExactQi::zero);
    value.add_assign(&coefficient);
    if value.is_zero() {
        output.remove(&key);
    }
}

fn polynomial_map_difference_count(
    left: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
    right: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
) -> usize {
    left.keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| left.get(key) != right.get(key))
        .count()
}

fn add_polynomial_scaled(
    output: &mut PolynomialMap,
    coordinate: usize,
    polynomial: &CanonicalSuperPolynomial,
    coefficient: &ExactQi,
) {
    for (monomial, value) in &polynomial.terms {
        add_polynomial_coefficient(
            output,
            coordinate,
            monomial.clone(),
            ExactQi {
                real: value.real.clone() * coefficient.real.clone()
                    - value.imaginary.clone() * coefficient.imaginary.clone(),
                imaginary: value.real.clone() * coefficient.imaginary.clone()
                    + value.imaginary.clone() * coefficient.real.clone(),
            },
        );
    }
}

fn eq25_bosonic_frame_polynomials(
    representative: &LinearizedFrameSuperfields,
) -> Result<PolynomialMap, String> {
    let operator = physical::eq25_dh_to_bosonic_frame_operator();
    let mut frame = PolynomialMap::new();
    visit_d_h_jet(representative, |entry| {
        debug_assert_eq!(entry.sector, LinearizedFrameJetSector::DH);
        for image in &operator.columns[entry.coordinate] {
            add_polynomial_scaled(&mut frame, image.row, &entry.polynomial, &image.coefficient);
        }
        Ok(())
    })?;
    for axis in 0..physical::VECTOR_DIMENSION {
        add_polynomial_scaled(
            &mut frame,
            axis * physical::VECTOR_DIMENSION + axis,
            &representative.scale,
            &ExactQi::one(),
        );
    }
    Ok(frame)
}

fn symmetric_metric_polynomials_from_inverse_frame(frame: &PolynomialMap) -> PolynomialMap {
    let mut metric = PolynomialMap::new();
    let mut field = 0_usize;
    for left in 0..physical::VECTOR_DIMENSION {
        for right in left..physical::VECTOR_DIMENSION {
            // Eq. (25) gives the inverse frame E_a{}^m=delta_a{}^m+f_a{}^m.
            // Hence g_mn=eta_mn-f_mn-f_nm at linear order, with
            // f_mn=eta_nn f_m{}^n in Cartesian Lorentz coordinates.
            for (frame_coordinate, sign) in [
                (
                    left * physical::VECTOR_DIMENSION + right,
                    if right == 0 { 1_i64 } else { -1_i64 },
                ),
                (
                    right * physical::VECTOR_DIMENSION + left,
                    if left == 0 { 1_i64 } else { -1_i64 },
                ),
            ] {
                if let Some(polynomial) = frame.get(&frame_coordinate) {
                    add_polynomial_scaled(
                        &mut metric,
                        field,
                        polynomial,
                        &ExactQi::from_integer(sign),
                    );
                }
            }
            field += 1;
        }
    }
    debug_assert_eq!(field, 66);
    metric
}

fn multiply_ordered_momentum(
    source: &OrderedSuperderivativeMonomial,
    target: &MomentumMonomial,
) -> Result<OrderedSuperderivativeMonomial, String> {
    let mut momentum = source.momentum.clone();
    for (axis, exponent) in target.exponents.iter().copied().enumerate() {
        for _ in 0..exponent {
            momentum = momentum.multiply_variable(axis)?;
        }
    }
    Ok(OrderedSuperderivativeMonomial {
        exterior_spinor_mask: source.exterior_spinor_mask,
        momentum,
    })
}

fn riemann_from_metric_polynomials(
    metric: &PolynomialMap,
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let curvature = &target_sector_complex(TargetSector::Graviton).curvature;
    if curvature.columns() != 66 || curvature.rows() != 55 * 55 {
        return Err(format!(
            "unexpected target graviton-curvature shape {}x{}",
            curvature.rows(),
            curvature.columns()
        ));
    }
    let mut output = BTreeMap::new();
    for (&field, polynomial) in metric {
        if field >= curvature.columns() {
            return Err(format!(
                "symmetric metric coordinate {field} is outside dimension {}",
                curvature.columns()
            ));
        }
        let target_terms = curvature.column_terms(field);
        for (source_monomial, source_coefficient) in &polynomial.terms {
            for (component, target_coefficient) in &target_terms {
                let monomial =
                    multiply_ordered_momentum(source_monomial, &target_coefficient.monomial)?;
                let coefficient =
                    multiply_exact_qi_by_public(source_coefficient, target_coefficient);
                if coefficient.is_zero() {
                    continue;
                }
                let key = (*component, monomial);
                let value = output.entry(key.clone()).or_insert_with(ExactQi::zero);
                value.add_assign(&coefficient);
                if value.is_zero() {
                    output.remove(&key);
                }
            }
        }
    }
    Ok(output)
}

/// Compose the canonical `H_hat` representative through the source-fixed
/// bosonic frame in hep-th/0101037 Eq. (25), convert the inverse-frame
/// perturbation to the covariant symmetric metric, and apply the independent
/// exact target graviton-curvature operator at generic formal momentum.
///
/// The result lands in the full 1,210-dimensional algebraic Riemann
/// representation inside the target's 55-by-55 pair-pair ambient coordinates,
/// without projecting away its Ricci or scalar-curvature pieces.  No Einstein
/// equation, Ricci-flat restriction, or `D^2 W` identification is imposed.
pub fn visit_gauge_fixed_linearized_riemann<F>(
    frame_input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<DirectLinearizedRiemannStats, String>
where
    F: FnMut(DirectLinearizedRiemannEntry) -> Result<(), String>,
{
    let representative = canonical_physical_frame_representative(frame_input)?;
    let frame = eq25_bosonic_frame_polynomials(&representative)?;
    let metric = symmetric_metric_polynomials_from_inverse_frame(&frame);
    let riemann = riemann_from_metric_polynomials(&metric)?;
    let stats = DirectLinearizedRiemannStats {
        source_frame_terms: frame.values().map(|value| value.terms.len()).sum(),
        symmetric_metric_terms: metric.values().map(|value| value.terms.len()).sum(),
        riemann_terms: riemann.len(),
    };
    for ((component, monomial), coefficient) in riemann {
        emit(DirectLinearizedRiemannEntry {
            component,
            monomial,
            coefficient,
        })?;
    }
    Ok(stats)
}

/// Apply the independent target differential-Bianchi operator to the direct
/// frame curvature.  A zero map is an exact polynomial identity at generic
/// formal momentum.  Spinor exterior monomials are carried through unchanged.
pub fn direct_riemann_bianchi_residual(
    curvature: &[DirectLinearizedRiemannEntry],
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let bianchi = &target_sector_complex(TargetSector::Graviton).bianchi;
    let mut residual = BTreeMap::new();
    for entry in curvature {
        if entry.component >= bianchi.columns() {
            return Err(format!(
                "Riemann component {} is outside target dimension {}",
                entry.component,
                bianchi.columns()
            ));
        }
        for (row, operator_term) in bianchi.column_terms(entry.component) {
            let monomial = multiply_ordered_momentum(&entry.monomial, &operator_term.monomial)?;
            let coefficient = multiply_exact_qi_by_public(&entry.coefficient, &operator_term);
            if coefficient.is_zero() {
                continue;
            }
            let key = (row, monomial);
            let value = residual.entry(key.clone()).or_insert_with(ExactQi::zero);
            value.add_assign(&coefficient);
            if value.is_zero() {
                residual.remove(&key);
            }
        }
    }
    Ok(residual)
}

/// Adapt a complete fixed-momentum first-spinor descendant of W_[4] into
/// the 1,760-component target gravitino curl.
///
/// This boundary is deliberately aggregate rather than coordinate-local.
/// hep-th/0107155 Eq. (3.1g) couples every D_alpha F_bcde component to the
/// same curl tensor. The exact left inverse therefore runs only after all
/// one-spinor W entries with a common momentum monomial have been assembled.
/// It also pushes the recovered curl forward and rejects any off-image input.
/// Interpreting this source-fixed teleparallel identity as D W_2021 remains
/// conditional on the convention W_[4]|_0 = F_[4].
pub fn adapt_w2021_first_descendants(
    entries: &[FrameComposedPhysicalFEntry],
) -> Result<BTreeMap<TargetCurvatureCoordinate, ExactQi>, String> {
    let mut descendants = BTreeMap::<MomentumMonomial, BTreeMap<usize, ExactQi>>::new();
    for entry in entries {
        if entry.sector != FrameComposedPhysicalFSector::W2021 {
            return Err(format!(
                "sector {:?} is not a W2021 first descendant",
                entry.sector
            ));
        }
        let exterior = entry.monomial.exterior_spinor_mask;
        if exterior.count_ones() != 1 {
            return Err(format!(
                "W2021 first descendant requires one exterior spinor, found mask {exterior:#010x}"
            ));
        }
        let alpha = exterior.trailing_zeros() as usize;
        let four_form =
            target_four_form_lexicographic_ordinal(w_source_four_form_indices(entry.coordinate)?);
        let row = alpha
            .checked_mul(physical::W_FOUR_FORM_DIMENSION)
            .and_then(|base| base.checked_add(four_form))
            .ok_or_else(|| "D W2021 row index overflow".to_string())?;
        let momentum = MomentumMonomial::try_from_u16(entry.monomial.momentum.exponents)?;
        let tensor = descendants.entry(momentum).or_default();
        let value = tensor.entry(row).or_insert_with(ExactQi::zero);
        value.add_assign(&entry.coefficient);
        if value.is_zero() {
            tensor.remove(&row);
        }
    }

    let mut output = BTreeMap::new();
    for (momentum, descendant) in descendants {
        let recovered = physical::recover_gravitino_curl_from_linearized_d_f_four(&descendant)?;
        for (component, coefficient) in recovered {
            let coordinate = TargetCurvatureCoordinate {
                sector: TargetCurvatureSector::GravitinoCurl,
                component,
                momentum: momentum.clone(),
            };
            let value = output
                .entry(coordinate.clone())
                .or_insert_with(ExactQi::zero);
            value.add_assign(&coefficient);
            if value.is_zero() {
                output.remove(&coordinate);
            }
        }
    }
    Ok(output)
}

/// One term in an explicitly differentiated W_[4] polynomial.
///
/// `derivative_spinor` is retained independently from `monomial`: the latter
/// is the ordered normal form after D_alpha has acted on W(H_hat), and cannot
/// in general be used to reconstruct alpha because anticommutators generate
/// momentum terms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplicitW2021FirstDescendantEntry {
    pub derivative_spinor: usize,
    pub four_form_coordinate: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

/// Recover polynomial gravitino-curl coordinates from explicitly typed
/// first descendants. Each ordered H_hat differential monomial is inverted
/// independently and pushed forward through the source-fixed map again.
pub fn adapt_explicit_w2021_first_descendants(
    entries: &[ExplicitW2021FirstDescendantEntry],
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let mut descendants =
        BTreeMap::<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>::new();
    for entry in entries {
        add_explicit_w2021_descendant(&mut descendants, entry)?;
    }

    recover_grouped_w2021_first_descendants(descendants)
}

fn recover_grouped_w2021_first_descendants(
    descendants: BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>,
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let mut output = BTreeMap::new();
    for (monomial, descendant) in descendants {
        let recovered = physical::recover_gravitino_curl_from_linearized_d_f_four(&descendant)
            .map_err(|error| {
                format!(
                    "conditional D W2021 image gate failed for exterior mask {:#010x} and momentum {:?}: {error}",
                    monomial.exterior_spinor_mask, monomial.momentum.exponents
                )
            })?;
        for (component, coefficient) in recovered {
            output.insert((component, monomial.clone()), coefficient);
        }
    }
    Ok(output)
}

fn add_explicit_w2021_descendant(
    descendants: &mut BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>,
    entry: &ExplicitW2021FirstDescendantEntry,
) -> Result<(), String> {
    if entry.derivative_spinor >= physical::SPINOR_DIMENSION {
        return Err(format!(
            "W2021 derivative spinor {} is outside dimension {}",
            entry.derivative_spinor,
            physical::SPINOR_DIMENSION
        ));
    }
    let four_form = target_four_form_lexicographic_ordinal(w_source_four_form_indices(
        entry.four_form_coordinate,
    )?);
    let row = entry.derivative_spinor * physical::W_FOUR_FORM_DIMENSION + four_form;
    let tensor = descendants.entry(entry.monomial.clone()).or_default();
    let value = tensor.entry(row).or_insert_with(ExactQi::zero);
    value.add_assign(&entry.coefficient);
    if value.is_zero() {
        tensor.remove(&row);
    }
    Ok(())
}

fn visit_explicitly_differentiated_w2021<F>(
    raw_entries: &[FrameComposedPhysicalFEntry],
    mut emit: F,
) -> Result<(), String>
where
    F: FnMut(ExplicitW2021FirstDescendantEntry) -> Result<(), String>,
{
    let mut w = PolynomialMap::new();
    for entry in raw_entries {
        if entry.sector != FrameComposedPhysicalFSector::W2021 {
            continue;
        }
        add_polynomial_coefficient(
            &mut w,
            entry.coordinate,
            entry.monomial.clone(),
            entry.coefficient.clone(),
        );
    }
    for (four_form_coordinate, polynomial) in w {
        for derivative_spinor in 0..physical::SPINOR_DIMENSION {
            for (monomial, coefficient) in left_multiply_d(derivative_spinor, &polynomial)?.terms {
                emit(ExplicitW2021FirstDescendantEntry {
                    derivative_spinor,
                    four_form_coordinate,
                    monomial,
                    coefficient,
                })?;
            }
        }
    }
    Ok(())
}

fn explicitly_differentiate_w2021(
    raw_entries: &[FrameComposedPhysicalFEntry],
) -> Result<Vec<ExplicitW2021FirstDescendantEntry>, String> {
    let mut output = Vec::new();
    visit_explicitly_differentiated_w2021(raw_entries, |entry| {
        output.push(entry);
        Ok(())
    })?;
    Ok(output)
}

fn adapt_differentiated_w2021_first_descendants(
    raw_entries: &[FrameComposedPhysicalFEntry],
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let mut descendants =
        BTreeMap::<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>::new();
    visit_explicitly_differentiated_w2021(raw_entries, |entry| {
        add_explicit_w2021_descendant(&mut descendants, &entry)
    })?;
    recover_grouped_w2021_first_descendants(descendants)
}

fn multiply_exact_qi_by_public(left: &ExactQi, right: &ExactPolynomialCoefficient) -> ExactQi {
    let right_real = num_rational::Ratio::new(right.real_numerator, right.real_denominator);
    let right_imaginary =
        num_rational::Ratio::new(right.imaginary_numerator, right.imaginary_denominator);
    ExactQi {
        real: left.real.clone() * right_real.clone()
            - left.imaginary.clone() * right_imaginary.clone(),
        imaginary: left.real.clone() * right_imaginary + left.imaginary.clone() * right_real,
    }
}

/// Apply the independent target four-form Bianchi operator to adapted
/// curvature entries.  A zero result is an exact polynomial identity, not a
/// sampled-momentum check.
pub fn four_form_bianchi_residual(
    curvature: &[(TargetCurvatureCoordinate, ExactQi)],
) -> Result<BTreeMap<(usize, MomentumMonomial), ExactQi>, String> {
    let bianchi = &target_sector_complex(TargetSector::FourForm).bianchi;
    let mut residual = BTreeMap::<(usize, MomentumMonomial), ExactQi>::new();
    for (coordinate, coefficient) in curvature {
        if coordinate.sector != TargetCurvatureSector::FourForm {
            return Err("non-four-form coordinate supplied to four-form Bianchi map".to_string());
        }
        if coordinate.component >= bianchi.columns() {
            return Err(format!(
                "four-form component {} is outside target dimension {}",
                coordinate.component,
                bianchi.columns()
            ));
        }
        for (row, operator_term) in bianchi.column_terms(coordinate.component) {
            let monomial = coordinate
                .momentum
                .checked_multiply(&operator_term.monomial)?;
            let value = multiply_exact_qi_by_public(coefficient, &operator_term);
            if value.is_zero() {
                continue;
            }
            let key = (row, monomial);
            let entry = residual.entry(key.clone()).or_insert_with(ExactQi::zero);
            entry.add_assign(&value);
            if entry.is_zero() {
                residual.remove(&key);
            }
        }
    }
    Ok(residual)
}

fn apply_target_polynomial_columns(
    columns: &TargetOperatorColumns,
    input: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let mut output = BTreeMap::new();
    for ((component, monomial), coefficient) in input {
        if *component >= columns.len() {
            return Err(format!(
                "target-operator input component {component} is outside dimension {}",
                columns.len()
            ));
        }
        for (row, operator_term) in &columns[*component] {
            let output_monomial = multiply_ordered_momentum(monomial, &operator_term.monomial)?;
            let output_coefficient = multiply_exact_qi_by_public(coefficient, &operator_term);
            add_polynomial_map_value(&mut output, (*row, output_monomial), output_coefficient);
        }
    }
    Ok(output)
}

fn certify_target_curvature_stream(
    label: &str,
    curvature: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
    bianchi: &TargetOperatorColumns,
    curvature_to_euler: &TargetOperatorColumns,
    noether: &TargetOperatorColumns,
) -> Result<usize, String> {
    let bianchi_residual = apply_target_polynomial_columns(bianchi, curvature)?;
    if !bianchi_residual.is_empty() {
        return Err(format!(
            "{label} violates the target Bianchi identity in {} coordinates",
            bianchi_residual.len()
        ));
    }
    let euler = apply_target_polynomial_columns(curvature_to_euler, curvature)?;
    let noether_residual = apply_target_polynomial_columns(noether, &euler)?;
    if !noether_residual.is_empty() {
        return Err(format!(
            "{label} violates Noether(Euler) in {} coordinates",
            noether_residual.len()
        ));
    }
    Ok(euler.len())
}

fn certify_emitted_riemann_target_stream(
    entries: &[GaugeFixedInvariantOutputEntry],
) -> Result<usize, String> {
    let mut riemann = BTreeMap::new();
    for entry in entries {
        if entry.sector == GaugeFixedInvariantOutputSector::LinearizedRiemann {
            add_polynomial_map_value(
                &mut riemann,
                (entry.coordinate, entry.monomial.clone()),
                entry.coefficient.clone(),
            );
        }
    }
    certify_target_curvature_stream(
        "direct linearized Riemann stream",
        &riemann,
        graviton_bianchi_columns(),
        graviton_curvature_to_euler_columns(),
        graviton_noether_columns(),
    )
}

/// Apply the independent target Rarita-Schwinger differential-Bianchi map to
/// polynomial gravitino-curl coordinates.
pub fn gravitino_curl_bianchi_residual(
    curvature: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    apply_target_polynomial_columns(rarita_bianchi_columns(), curvature)
}

/// Retain the exact Rarita-Schwinger Euler image derived from a curl stream.
pub fn gravitino_curl_euler_image(
    curvature: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    apply_target_polynomial_columns(rarita_curvature_to_euler_columns(), curvature)
}

/// Apply Noether to the retained Euler image. Exact zero is the target
/// Noether identity; any nonzero coordinate fails closed.
pub fn gravitino_curl_noether_residual(
    curvature: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
) -> Result<BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>, String> {
    let euler = gravitino_curl_euler_image(curvature)?;
    apply_target_polynomial_columns(rarita_noether_columns(), &euler)
}

/// Stable sectors in the production invariant-supercurvature column format.
///
/// `W2021Raw` deliberately remains a raw superfield coordinate at every
/// exterior-D degree.  In particular, its two-D terms are not relabeled as
/// gravity.  `LinearizedRiemann` is emitted only by the independent direct
/// frame adapter, so the schema cannot double-emit a theta-two W term as a
/// second Riemann coordinate. `ConditionalGravitinoCurl` is present only when
/// the caller explicitly enables the pinned W_2021|_0=F_[4] convention gate.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(u8)]
pub enum GaugeFixedInvariantOutputSector {
    XTwo = 0,
    XFive = 1,
    JMinus = 5,
    W2021Raw = 8,
    LinearizedRiemann = 9,
    DirectGravitinoCurl = 10,
    DirectCandidateFourForm = 11,
    ConditionalGravitinoCurl = 12,
}

impl GaugeFixedInvariantOutputSector {
    fn tag(self) -> u8 {
        self as u8
    }

    fn from_frame(sector: FrameComposedPhysicalFSector) -> Option<Self> {
        match sector {
            FrameComposedPhysicalFSector::XTwo => Some(Self::XTwo),
            FrameComposedPhysicalFSector::XFive => Some(Self::XFive),
            FrameComposedPhysicalFSector::JMinus => Some(Self::JMinus),
            FrameComposedPhysicalFSector::W2021 => Some(Self::W2021Raw),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GaugeFixedInvariantOutputEntry {
    pub sector: GaugeFixedInvariantOutputSector,
    pub coordinate: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GaugeFixedInvariantOutputStats {
    pub raw_invariant_terms: usize,
    pub direct_riemann_terms: usize,
    pub direct_gravitino_curl_terms: usize,
    pub direct_candidate_four_form_terms: usize,
    pub candidate_four_form_raw_w_bianchi_residual_terms: usize,
    pub candidate_four_form_raw_w_comparison_residual_terms: usize,
    pub conditional_gravitino_curl_terms: usize,
    pub emitted_by_sector: BTreeMap<GaugeFixedInvariantOutputSector, usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum W2021FirstDescendantConventionGate {
    Disabled,
    /// Explicitly identify the W_2021 lowest component with the physical
    /// four-form and apply hep-th/0107155 Eq. (3.1g).
    W2021LowestIsPhysicalFourForm,
}

fn invariant_entry_cmp(
    left: &GaugeFixedInvariantOutputEntry,
    right: &GaugeFixedInvariantOutputEntry,
) -> std::cmp::Ordering {
    (&left.monomial, left.sector, left.coordinate).cmp(&(
        &right.monomial,
        right.sector,
        right.coordinate,
    ))
}

/// Stream the versioned production invariant output for one gauge-fixed source.
///
/// The raw X/J/W stream and the direct Riemann stream are merged into strict
/// `(exterior-D mask, momentum, sector, coordinate)` order.  This retains the
/// bounded-memory shard contract while preserving every exterior-D mask.
pub fn visit_gauge_fixed_invariant_supercurvature<F>(
    frame_input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<GaugeFixedInvariantOutputStats, String>
where
    F: FnMut(GaugeFixedInvariantOutputEntry) -> Result<(), String>,
{
    let mut ignore_raw_frame = |_: &FrameComposedPhysicalFEntry| Ok(());
    visit_gauge_fixed_invariant_supercurvature_base(frame_input, &mut ignore_raw_frame, &mut emit)
}

fn visit_gauge_fixed_invariant_supercurvature_base<F, G>(
    frame_input: &LinearizedFrameSuperfields,
    observe_raw_frame: &mut G,
    emit: &mut F,
) -> Result<GaugeFixedInvariantOutputStats, String>
where
    F: FnMut(GaugeFixedInvariantOutputEntry) -> Result<(), String>,
    G: FnMut(&FrameComposedPhysicalFEntry) -> Result<(), String>,
{
    let mut riemann = Vec::new();
    visit_gauge_fixed_linearized_riemann(frame_input, |entry| {
        riemann.push(GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::LinearizedRiemann,
            coordinate: entry.component,
            monomial: entry.monomial,
            coefficient: entry.coefficient,
        });
        Ok(())
    })?;
    riemann.sort_by(invariant_entry_cmp);
    let mut riemann = riemann.into_iter().peekable();
    let mut previous = None::<GaugeFixedInvariantOutputEntry>;
    let mut stats = GaugeFixedInvariantOutputStats::default();

    let mut emit_checked = |entry: GaugeFixedInvariantOutputEntry| -> Result<(), String> {
        if previous
            .as_ref()
            .is_some_and(|prior| invariant_entry_cmp(prior, &entry).is_ge())
        {
            return Err("unified invariant output is not strictly row ordered".to_string());
        }
        *stats.emitted_by_sector.entry(entry.sector).or_default() += 1;
        previous = Some(entry.clone());
        emit(entry)
    };

    visit_gauge_fixed_physical_f_production(frame_input, |entry| {
        observe_raw_frame(&entry)?;
        let Some(sector) = GaugeFixedInvariantOutputSector::from_frame(entry.sector) else {
            return Ok(());
        };
        let unified = GaugeFixedInvariantOutputEntry {
            sector,
            coordinate: entry.coordinate,
            monomial: entry.monomial,
            coefficient: entry.coefficient,
        };
        while riemann
            .peek()
            .is_some_and(|candidate| invariant_entry_cmp(candidate, &unified).is_lt())
        {
            emit_checked(riemann.next().unwrap())?;
        }
        emit_checked(unified)
    })?;
    for entry in riemann {
        emit_checked(entry)?;
    }
    stats.direct_riemann_terms = stats
        .emitted_by_sector
        .get(&GaugeFixedInvariantOutputSector::LinearizedRiemann)
        .copied()
        .unwrap_or(0);
    stats.raw_invariant_terms = stats
        .emitted_by_sector
        .iter()
        .filter(|(sector, _)| **sector != GaugeFixedInvariantOutputSector::LinearizedRiemann)
        .map(|(_, count)| *count)
        .sum();
    Ok(stats)
}

/// Transactional successor stream adding the direct source-fixed gravitino
/// curl and the separately labeled Eq. (40) `Psi_[3]` four-form candidate to
/// the frozen invariant sectors. The v4/COL3 writer consumes this visitor;
/// the frozen v3/COL2 artifacts remain schema-incompatible.
pub fn visit_gauge_fixed_invariant_supercurvature_with_direct_gravitino<F>(
    frame_input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<GaugeFixedInvariantOutputStats, String>
where
    F: FnMut(GaugeFixedInvariantOutputEntry) -> Result<(), String>,
{
    let mut entries = Vec::new();
    let mut ignore_raw_frame = |_: &FrameComposedPhysicalFEntry| Ok(());
    let mut collect_base = |entry| {
        entries.push(entry);
        Ok(())
    };
    visit_gauge_fixed_invariant_supercurvature_base(
        frame_input,
        &mut ignore_raw_frame,
        &mut collect_base,
    )?;
    certify_emitted_riemann_target_stream(&entries)?;
    let (candidate_four_form, candidate_four_form_entries) =
        gauge_fixed_candidate_four_form_curvature(frame_input, &entries)?;
    entries.extend(candidate_four_form_entries.into_iter().map(|entry| {
        GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectCandidateFourForm,
            coordinate: entry.component,
            monomial: entry.monomial,
            coefficient: entry.coefficient,
        }
    }));
    visit_gauge_fixed_linearized_gravitino_curl(frame_input, |entry| {
        entries.push(GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectGravitinoCurl,
            coordinate: entry.component,
            monomial: entry.monomial,
            coefficient: entry.coefficient,
        });
        Ok(())
    })?;
    let mut stats = emit_sorted_invariant_entries(entries, &mut emit)?;
    stats.candidate_four_form_raw_w_bianchi_residual_terms =
        candidate_four_form.raw_w_bianchi_residual_terms;
    stats.candidate_four_form_raw_w_comparison_residual_terms =
        candidate_four_form.raw_w_comparison_residual_terms;
    Ok(stats)
}

fn emit_sorted_invariant_entries<F>(
    mut entries: Vec<GaugeFixedInvariantOutputEntry>,
    emit: &mut F,
) -> Result<GaugeFixedInvariantOutputStats, String>
where
    F: FnMut(GaugeFixedInvariantOutputEntry) -> Result<(), String>,
{
    entries.sort_by(invariant_entry_cmp);
    let mut stats = GaugeFixedInvariantOutputStats::default();
    let mut previous = None;
    for entry in entries {
        if previous
            .as_ref()
            .is_some_and(|prior| invariant_entry_cmp(prior, &entry).is_ge())
        {
            return Err("unified invariant output is not strictly row ordered".to_string());
        }
        *stats.emitted_by_sector.entry(entry.sector).or_default() += 1;
        previous = Some(entry.clone());
        emit(entry)?;
    }
    stats.direct_riemann_terms = *stats
        .emitted_by_sector
        .get(&GaugeFixedInvariantOutputSector::LinearizedRiemann)
        .unwrap_or(&0);
    stats.direct_gravitino_curl_terms = *stats
        .emitted_by_sector
        .get(&GaugeFixedInvariantOutputSector::DirectGravitinoCurl)
        .unwrap_or(&0);
    stats.direct_candidate_four_form_terms = *stats
        .emitted_by_sector
        .get(&GaugeFixedInvariantOutputSector::DirectCandidateFourForm)
        .unwrap_or(&0);
    stats.conditional_gravitino_curl_terms = *stats
        .emitted_by_sector
        .get(&GaugeFixedInvariantOutputSector::ConditionalGravitinoCurl)
        .unwrap_or(&0);
    stats.raw_invariant_terms = stats
        .emitted_by_sector
        .iter()
        .filter(|(sector, _)| {
            !matches!(
                **sector,
                GaugeFixedInvariantOutputSector::LinearizedRiemann
                    | GaugeFixedInvariantOutputSector::DirectGravitinoCurl
                    | GaugeFixedInvariantOutputSector::DirectCandidateFourForm
                    | GaugeFixedInvariantOutputSector::ConditionalGravitinoCurl
            )
        })
        .map(|(_, count)| *count)
        .sum();
    Ok(stats)
}

/// Conditional production stream including the explicitly differentiated
/// W_[4] first descendant. The gate is a required typed argument, and enabled
/// inputs are rejected unless every monomial lies in the exact forward image
/// of the 1,760-component gravitino curl.
pub fn visit_gauge_fixed_invariant_supercurvature_with_first_descendant<F>(
    frame_input: &LinearizedFrameSuperfields,
    convention: W2021FirstDescendantConventionGate,
    mut emit: F,
) -> Result<GaugeFixedInvariantOutputStats, String>
where
    F: FnMut(GaugeFixedInvariantOutputEntry) -> Result<(), String>,
{
    if convention == W2021FirstDescendantConventionGate::Disabled {
        return visit_gauge_fixed_invariant_supercurvature_with_direct_gravitino(frame_input, emit);
    }

    // The enabled route must retain one column until the aggregate image gate
    // has passed. The default production route above remains bounded and
    // byte-for-byte compatible with the frozen v3 stream.
    let mut entries = Vec::new();
    let mut raw_frame_entries = Vec::new();
    let mut observe_raw_frame = |entry: &FrameComposedPhysicalFEntry| {
        if entry.sector == FrameComposedPhysicalFSector::W2021 {
            raw_frame_entries.push(entry.clone());
        }
        Ok(())
    };
    let mut collect_base = |entry| {
        entries.push(entry);
        Ok(())
    };
    visit_gauge_fixed_invariant_supercurvature_base(
        frame_input,
        &mut observe_raw_frame,
        &mut collect_base,
    )?;
    certify_emitted_riemann_target_stream(&entries)?;
    let (candidate_four_form, candidate_four_form_entries) =
        gauge_fixed_candidate_four_form_curvature(frame_input, &entries)?;
    entries.extend(candidate_four_form_entries.into_iter().map(|entry| {
        GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectCandidateFourForm,
            coordinate: entry.component,
            monomial: entry.monomial,
            coefficient: entry.coefficient,
        }
    }));
    visit_gauge_fixed_linearized_gravitino_curl(frame_input, |entry| {
        entries.push(GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectGravitinoCurl,
            coordinate: entry.component,
            monomial: entry.monomial,
            coefficient: entry.coefficient,
        });
        Ok(())
    })?;

    let recovered = adapt_differentiated_w2021_first_descendants(&raw_frame_entries)?;
    let bianchi_residual = gravitino_curl_bianchi_residual(&recovered)?;
    if !bianchi_residual.is_empty() {
        return Err(format!(
            "conditional gravitino curl violates the exact target differential Bianchi identity ({} residual coordinates)",
            bianchi_residual.len()
        ));
    }
    for ((coordinate, monomial), coefficient) in recovered {
        entries.push(GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::ConditionalGravitinoCurl,
            coordinate,
            monomial,
            coefficient,
        });
    }

    let mut stats = emit_sorted_invariant_entries(entries, &mut emit)?;
    stats.candidate_four_form_raw_w_bianchi_residual_terms =
        candidate_four_form.raw_w_bianchi_residual_terms;
    stats.candidate_four_form_raw_w_comparison_residual_terms =
        candidate_four_form.raw_w_comparison_residual_terms;
    Ok(stats)
}

pub(crate) const SUPERFIELD_OPERATOR_COLUMN_SCHEMA: &[u8] =
    b"adynkra-11d-gauge-fixed-invariant-supercurvature-column-v3\0";
pub(crate) const SUPERFIELD_OPERATOR_SCHEMA: &str =
    "adynkra-11d-gauge-fixed-invariant-supercurvature-operator-v4";
pub(crate) const SUPERFIELD_COLUMN_SHARD_SCHEMA: &str =
    "adynkra-11d-gauge-fixed-invariant-supercurvature-column-shard-v3";
pub(crate) const SUPERFIELD_UNIFIED_OUTPUT_SCHEMA: &str =
    "adynkra-11d-gauge-fixed-invariant-supercurvature-output-v2";
pub(crate) const SUPERFIELD_COLUMN_SHARD_MAGIC: &[u8; 16] = b"AD11FINVCOL3\0\0\0\0";
pub(crate) const SUPERFIELD_COLUMN_ENTRY_BYTES: usize = 67;
const FROZEN_V3_COLUMN_SHARD_MAGIC: &[u8; 16] = b"AD11FINVCOL2\0\0\0\0";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GaugeFixedSuperfieldColumnDigest {
    pub ordinal: usize,
    pub source_coordinate: String,
    pub nonzero_terms: usize,
    /// Exact target-Bianchi residual size of the auxiliary raw-W sector.
    pub raw_w_bianchi_residual_terms: usize,
    /// Exact support size of raw W minus the closed Eq. (40) candidate.
    pub raw_w_candidate_residual_terms: usize,
    pub sha256: String,
    pub shard_path: Option<String>,
    pub shard_sha256: Option<String>,
    pub shard_byte_count: Option<u64>,
    pub shard_reused: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GaugeFixedSuperfieldOperatorCertificate {
    pub schema_version: &'static str,
    pub source_basis: &'static str,
    pub source_dimension: usize,
    pub gamma_traceless_h_dimension: usize,
    pub scale_dimension: usize,
    pub output_basis: &'static str,
    pub unified_output_schema: &'static str,
    pub column_shard_schema: &'static str,
    pub column_shard_directory: String,
    pub columns: Vec<GaugeFixedSuperfieldColumnDigest>,
    pub total_nonzero_terms: u64,
    pub operator_sha256: String,
    pub direct_riemann_integrated: bool,
    pub direct_gravitino_curl_integrated: bool,
    pub direct_candidate_four_form_integrated: bool,
    pub raw_w2021_two_d_terms_are_not_gravity: bool,
    pub physical_target_component_adapter_complete: bool,
    pub exact_polynomial_kernel_derived: bool,
}

fn write_hashed<W: Write>(
    writer: &mut W,
    file_hasher: &mut Sha256,
    bytes: &[u8],
) -> io::Result<()> {
    writer.write_all(bytes)?;
    file_hasher.update(bytes);
    Ok(())
}

fn encode_invariant_entry<W: Write>(
    writer: &mut W,
    file_hasher: &mut Sha256,
    entry: &GaugeFixedInvariantOutputEntry,
) -> io::Result<()> {
    let mut record = [0_u8; SUPERFIELD_COLUMN_ENTRY_BYTES];
    let mut cursor = 0;
    record[cursor] = entry.sector.tag();
    cursor += 1;
    let coordinate = (entry.coordinate as u64).to_le_bytes();
    record[cursor..cursor + coordinate.len()].copy_from_slice(&coordinate);
    cursor += coordinate.len();
    let exterior_mask = entry.monomial.exterior_spinor_mask.to_le_bytes();
    record[cursor..cursor + exterior_mask.len()].copy_from_slice(&exterior_mask);
    cursor += exterior_mask.len();
    for exponent in entry.monomial.momentum.exponents {
        let bytes = exponent.to_le_bytes();
        record[cursor..cursor + bytes.len()].copy_from_slice(&bytes);
        cursor += bytes.len();
    }
    for value in [
        entry.coefficient.real.numer(),
        entry.coefficient.real.denom(),
        entry.coefficient.imaginary.numer(),
        entry.coefficient.imaginary.denom(),
    ] {
        let bytes = value.to_le_bytes();
        record[cursor..cursor + bytes.len()].copy_from_slice(&bytes);
        cursor += bytes.len();
    }
    debug_assert_eq!(cursor, SUPERFIELD_COLUMN_ENTRY_BYTES);
    write_hashed(writer, file_hasher, &record)
}

fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N], String> {
    let end = cursor
        .checked_add(N)
        .ok_or_else(|| "column shard offset overflow".to_string())?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| "truncated column shard".to_string())?;
    *cursor = end;
    Ok(slice.try_into().unwrap())
}

fn invariant_coordinate_is_valid(tag: u8, coordinate: u64) -> bool {
    match tag {
        value if value == GaugeFixedInvariantOutputSector::XTwo.tag() => coordinate < 605,
        value if value == GaugeFixedInvariantOutputSector::XFive.tag() => coordinate < 5_082,
        value if value == GaugeFixedInvariantOutputSector::JMinus.tag() => coordinate < 32,
        value if value == GaugeFixedInvariantOutputSector::W2021Raw.tag() => coordinate < 330,
        value if value == GaugeFixedInvariantOutputSector::LinearizedRiemann.tag() => {
            coordinate < 3_025
        }
        value if value == GaugeFixedInvariantOutputSector::DirectGravitinoCurl.tag() => {
            coordinate < 1_760
        }
        value if value == GaugeFixedInvariantOutputSector::DirectCandidateFourForm.tag() => {
            coordinate < 330
        }
        _ => false,
    }
}

fn validate_column_shard(
    path: &Path,
    ordinal: usize,
    expected_source_coordinate: &str,
) -> Result<GaugeFixedSuperfieldColumnDigest, String> {
    let mut bytes = Vec::new();
    BufReader::new(File::open(path).map_err(|error| format!("{}: {error}", path.display()))?)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mut cursor = 0_usize;
    if &take::<16>(&bytes, &mut cursor)? != SUPERFIELD_COLUMN_SHARD_MAGIC {
        return Err(format!("{} has invalid column-shard magic", path.display()));
    }
    let stored_ordinal = u64::from_le_bytes(take(&bytes, &mut cursor)?);
    if stored_ordinal != ordinal as u64 {
        return Err(format!(
            "{} stores ordinal {stored_ordinal}, expected {ordinal}",
            path.display()
        ));
    }
    let name_length = usize::try_from(u64::from_le_bytes(take(&bytes, &mut cursor)?))
        .map_err(|_| "column shard source-name length exceeds usize".to_string())?;
    let name_end = cursor
        .checked_add(name_length)
        .ok_or_else(|| "column shard source-name offset overflow".to_string())?;
    let source_coordinate = std::str::from_utf8(
        bytes
            .get(cursor..name_end)
            .ok_or_else(|| "truncated column shard source name".to_string())?,
    )
    .map_err(|error| format!("invalid column shard source name: {error}"))?;
    cursor = name_end;
    if source_coordinate != expected_source_coordinate {
        return Err(format!(
            "{} stores source {source_coordinate}, expected {expected_source_coordinate}",
            path.display()
        ));
    }
    if bytes.len() < cursor + 40 {
        return Err("truncated column shard footer".to_string());
    }
    let entry_bytes = bytes.len() - cursor - 40;
    if entry_bytes % SUPERFIELD_COLUMN_ENTRY_BYTES != 0 {
        return Err(format!(
            "column shard payload has {entry_bytes} bytes, not a multiple of {SUPERFIELD_COLUMN_ENTRY_BYTES}"
        ));
    }
    let entry_count = entry_bytes / SUPERFIELD_COLUMN_ENTRY_BYTES;
    let mut semantic = Sha256::new();
    semantic.update(SUPERFIELD_OPERATOR_COLUMN_SCHEMA);
    semantic.update((ordinal as u64).to_le_bytes());
    hash_bytes_with_length(&mut semantic, source_coordinate.as_bytes());
    let mut previous_key = None;
    let mut raw_w = BTreeMap::new();
    let mut candidate_four_form = BTreeMap::new();
    for entry_index in 0..entry_count {
        let tag = take::<1>(&bytes, &mut cursor)?[0];
        if !matches!(tag, 0 | 1 | 5 | 8 | 9 | 10 | 11) {
            return Err(format!(
                "column shard contains non-invariant sector tag {tag}"
            ));
        }
        semantic.update([tag]);
        let coordinate = u64::from_le_bytes(take(&bytes, &mut cursor)?);
        if !invariant_coordinate_is_valid(tag, coordinate) {
            return Err(format!(
                "column shard contains invalid sector/coordinate {tag}/{coordinate}"
            ));
        }
        semantic.update(coordinate.to_le_bytes());
        let exterior = u32::from_le_bytes(take(&bytes, &mut cursor)?);
        semantic.update(exterior.to_le_bytes());
        let mut momentum = [0_u16; physical::VECTOR_DIMENSION];
        for exponent in &mut momentum {
            *exponent = u16::from_le_bytes(take(&bytes, &mut cursor)?);
            semantic.update(exponent.to_le_bytes());
        }
        let key = (exterior, momentum, tag, coordinate);
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(format!(
                "column shard is not strictly row ordered at entry {entry_index}"
            ));
        }
        previous_key = Some(key);
        let mut values = [0_i64; 4];
        for value in &mut values {
            *value = i64::from_le_bytes(take(&bytes, &mut cursor)?);
        }
        if values[1] <= 0 || values[3] <= 0 {
            return Err("column shard contains a nonpositive rational denominator".to_string());
        }
        for value in values {
            hash_bytes_with_length(&mut semantic, value.to_string().as_bytes());
        }
        if tag == GaugeFixedInvariantOutputSector::W2021Raw.tag()
            || tag == GaugeFixedInvariantOutputSector::DirectCandidateFourForm.tag()
        {
            let source_coordinate = usize::try_from(coordinate)
                .map_err(|_| "column coordinate exceeds usize".to_string())?;
            let (map, target_coordinate) = if tag == GaugeFixedInvariantOutputSector::W2021Raw.tag()
            {
                (
                    &mut raw_w,
                    target_four_form_lexicographic_ordinal(w_source_four_form_indices(
                        source_coordinate,
                    )?),
                )
            } else {
                (&mut candidate_four_form, source_coordinate)
            };
            add_polynomial_map_value(
                map,
                (
                    target_coordinate,
                    OrderedSuperderivativeMonomial {
                        exterior_spinor_mask: exterior,
                        momentum: crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                            exponents: momentum,
                        },
                    },
                ),
                ExactQi {
                    real: num_rational::Ratio::new(values[0], values[1]),
                    imaginary: num_rational::Ratio::new(values[2], values[3]),
                },
            );
        }
    }
    let stored_count = u64::from_le_bytes(take(&bytes, &mut cursor)?);
    if stored_count != entry_count as u64 {
        return Err(format!(
            "column shard footer count {stored_count} != decoded {entry_count}"
        ));
    }
    semantic.update(stored_count.to_le_bytes());
    let semantic_sha256 = format!("{:x}", semantic.finalize());
    let stored_sha256 = take::<32>(&bytes, &mut cursor)?
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if semantic_sha256 != stored_sha256 {
        return Err(format!(
            "column shard semantic SHA mismatch: stored {stored_sha256}, computed {semantic_sha256}"
        ));
    }
    let raw_w_bianchi_residual_terms =
        apply_target_polynomial_columns(four_form_bianchi_columns(), &raw_w)?.len();
    let mut raw_w_candidate_residual = raw_w;
    for (key, coefficient) in candidate_four_form {
        add_polynomial_map_value(
            &mut raw_w_candidate_residual,
            key,
            coefficient.scaled(&num_rational::Ratio::from_integer(-1)),
        );
    }
    Ok(GaugeFixedSuperfieldColumnDigest {
        ordinal,
        source_coordinate: source_coordinate.to_string(),
        nonzero_terms: entry_count,
        raw_w_bianchi_residual_terms,
        raw_w_candidate_residual_terms: raw_w_candidate_residual.len(),
        sha256: semantic_sha256,
        shard_path: Some(path.display().to_string()),
        shard_sha256: Some(file_sha256),
        shard_byte_count: Some(bytes.len() as u64),
        shard_reused: true,
    })
}

fn hash_bytes_with_length(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_ratio(hasher: &mut Sha256, value: &num_rational::Ratio<i64>) {
    hash_bytes_with_length(hasher, value.numer().to_string().as_bytes());
    hash_bytes_with_length(hasher, value.denom().to_string().as_bytes());
}

fn hash_invariant_entry(hasher: &mut Sha256, entry: &GaugeFixedInvariantOutputEntry) {
    hasher.update([entry.sector.tag()]);
    hasher.update((entry.coordinate as u64).to_le_bytes());
    hasher.update(entry.monomial.exterior_spinor_mask.to_le_bytes());
    for exponent in entry.monomial.momentum.exponents {
        hasher.update(exponent.to_le_bytes());
    }
    hash_ratio(hasher, &entry.coefficient.real);
    hash_ratio(hasher, &entry.coefficient.imaginary);
}

fn source_basis_input(ordinal: usize) -> Result<(String, LinearizedFrameSuperfields), String> {
    let basis = canonical_gamma_traceless_frame_basis();
    if ordinal < basis.len() {
        let spatial_vector = ordinal / physical::SPINOR_DIMENSION + 1;
        let spatial_spinor = ordinal % physical::SPINOR_DIMENSION;
        let h = basis[ordinal]
            .iter()
            .map(|(&coordinate, coefficient)| {
                (
                    coordinate,
                    CanonicalSuperPolynomial::scalar(coefficient.clone()),
                )
            })
            .collect();
        return Ok((
            format!("h_spatial_v{spatial_vector}_spinor{spatial_spinor}"),
            LinearizedFrameSuperfields {
                h,
                ..LinearizedFrameSuperfields::default()
            },
        ));
    }
    if ordinal == basis.len() {
        return Ok((
            "scale".to_string(),
            LinearizedFrameSuperfields {
                scale: CanonicalSuperPolynomial::scalar(ExactQi::one()),
                ..LinearizedFrameSuperfields::default()
            },
        ));
    }
    Err(format!("source ordinal {ordinal} is outside 0..321"))
}

/// Hash one exact source column. Entries are sorted by the declared output
/// tuple before hashing, so callback scheduling cannot affect the digest.
pub fn gauge_fixed_superfield_column_digest(
    ordinal: usize,
) -> Result<GaugeFixedSuperfieldColumnDigest, String> {
    let (source_coordinate, input) = source_basis_input(ordinal)?;
    let mut hasher = Sha256::new();
    hasher.update(SUPERFIELD_OPERATOR_COLUMN_SCHEMA);
    hasher.update((ordinal as u64).to_le_bytes());
    hash_bytes_with_length(&mut hasher, source_coordinate.as_bytes());
    let mut nonzero_terms = 0_u64;
    let stream_stats =
        visit_gauge_fixed_invariant_supercurvature_with_direct_gravitino(&input, |entry| {
            // The unified visitor is strictly ordered by exterior-D mask,
            // momentum, stable sector tag, then sparse coordinate. Hashing at the
            // callback boundary avoids retaining millions of entries per column.
            hash_invariant_entry(&mut hasher, &entry);
            nonzero_terms = nonzero_terms
                .checked_add(1)
                .ok_or_else(|| "superfield-curvature column term count overflow".to_string())?;
            Ok(())
        })?;
    hasher.update(nonzero_terms.to_le_bytes());
    Ok(GaugeFixedSuperfieldColumnDigest {
        ordinal,
        source_coordinate,
        nonzero_terms: usize::try_from(nonzero_terms)
            .map_err(|_| "column term count exceeds usize".to_string())?,
        raw_w_bianchi_residual_terms: stream_stats.candidate_four_form_raw_w_bianchi_residual_terms,
        raw_w_candidate_residual_terms: stream_stats
            .candidate_four_form_raw_w_comparison_residual_terms,
        sha256: format!("{:x}", hasher.finalize()),
        shard_path: None,
        shard_sha256: None,
        shard_byte_count: None,
        shard_reused: false,
    })
}

fn write_or_validate_gauge_fixed_superfield_column_shard(
    ordinal: usize,
    shard_directory: &Path,
) -> Result<GaugeFixedSuperfieldColumnDigest, String> {
    let (source_coordinate, input) = source_basis_input(ordinal)?;
    fs::create_dir_all(shard_directory)
        .map_err(|error| format!("{}: {error}", shard_directory.display()))?;
    let path = shard_directory.join(format!("column_{ordinal:03}.bin"));
    if path.exists() {
        return validate_column_shard(&path, ordinal, &source_coordinate);
    }
    let temporary = shard_directory.join(format!("column_{ordinal:03}.{}.tmp", std::process::id()));
    let file =
        File::create(&temporary).map_err(|error| format!("{}: {error}", temporary.display()))?;
    let mut writer = BufWriter::new(file);
    let mut file_hasher = Sha256::new();
    write_hashed(&mut writer, &mut file_hasher, SUPERFIELD_COLUMN_SHARD_MAGIC)
        .map_err(|error| error.to_string())?;
    write_hashed(
        &mut writer,
        &mut file_hasher,
        &(ordinal as u64).to_le_bytes(),
    )
    .map_err(|error| error.to_string())?;
    write_hashed(
        &mut writer,
        &mut file_hasher,
        &(source_coordinate.len() as u64).to_le_bytes(),
    )
    .map_err(|error| error.to_string())?;
    write_hashed(&mut writer, &mut file_hasher, source_coordinate.as_bytes())
        .map_err(|error| error.to_string())?;

    let mut semantic = Sha256::new();
    semantic.update(SUPERFIELD_OPERATOR_COLUMN_SCHEMA);
    semantic.update((ordinal as u64).to_le_bytes());
    hash_bytes_with_length(&mut semantic, source_coordinate.as_bytes());
    let mut nonzero_terms = 0_u64;
    let stream_stats =
        visit_gauge_fixed_invariant_supercurvature_with_direct_gravitino(&input, |entry| {
            hash_invariant_entry(&mut semantic, &entry);
            encode_invariant_entry(&mut writer, &mut file_hasher, &entry)
                .map_err(|error| error.to_string())?;
            nonzero_terms = nonzero_terms
                .checked_add(1)
                .ok_or_else(|| "superfield-curvature column term count overflow".to_string())?;
            Ok(())
        })?;
    semantic.update(nonzero_terms.to_le_bytes());
    let semantic_digest = semantic.finalize();
    write_hashed(&mut writer, &mut file_hasher, &nonzero_terms.to_le_bytes())
        .map_err(|error| error.to_string())?;
    write_hashed(&mut writer, &mut file_hasher, semantic_digest.as_slice())
        .map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| error.to_string())?;
    fs::rename(&temporary, &path).map_err(|error| error.to_string())?;
    File::open(shard_directory)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| error.to_string())?;
    let byte_count = fs::metadata(&path)
        .map_err(|error| error.to_string())?
        .len();
    Ok(GaugeFixedSuperfieldColumnDigest {
        ordinal,
        source_coordinate,
        nonzero_terms: usize::try_from(nonzero_terms)
            .map_err(|_| "column term count exceeds usize".to_string())?,
        raw_w_bianchi_residual_terms: stream_stats.candidate_four_form_raw_w_bianchi_residual_terms,
        raw_w_candidate_residual_terms: stream_stats
            .candidate_four_form_raw_w_comparison_residual_terms,
        sha256: semantic_digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        shard_path: Some(path.display().to_string()),
        shard_sha256: Some(format!("{:x}", file_hasher.finalize())),
        shard_byte_count: Some(byte_count),
        shard_reused: false,
    })
}

pub fn build_gauge_fixed_superfield_operator_certificate<F>(
    progress: F,
) -> Result<GaugeFixedSuperfieldOperatorCertificate, String>
where
    F: FnMut(&GaugeFixedSuperfieldColumnDigest),
{
    build_gauge_fixed_superfield_operator_certificate_internal(None, progress)
}

fn build_gauge_fixed_superfield_operator_certificate_internal<F>(
    shard_directory: Option<&Path>,
    mut progress: F,
) -> Result<GaugeFixedSuperfieldOperatorCertificate, String>
where
    F: FnMut(&GaugeFixedSuperfieldColumnDigest),
{
    let default_workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(4);
    let cuda_requested = cfg!(feature = "cuda")
        && std::env::var("ADYNKRA_COMPLETE_F_CUDA")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
    // Dense lane tiling bounds each CUDA context. The measured production
    // RTX 4090 and 64 GiB host gate is eight concurrent worker sets; wider runs
    // fail closed here instead of entering device or host OOM pressure.
    let maximum_workers = if cuda_requested { 8 } else { 16 };
    let worker_count = std::env::var("ADYNKRA_COMPLETE_F_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_workers)
        .clamp(1, maximum_workers);
    let next_ordinal = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let (sender, receiver) = mpsc::channel();
    let mut completed = (0..321).map(|_| None).collect::<Vec<_>>();
    std::thread::scope(|scope| -> Result<(), String> {
        for _ in 0..worker_count {
            let sender = sender.clone();
            let next_ordinal = &next_ordinal;
            let cancelled = &cancelled;
            let shard_directory = shard_directory.map(Path::to_path_buf);
            scope.spawn(move || {
                loop {
                    if cancelled.load(Ordering::Acquire) {
                        break;
                    }
                    let ordinal = next_ordinal.fetch_add(1, Ordering::Relaxed);
                    if ordinal >= 321 {
                        break;
                    }
                    let result = if let Some(directory) = &shard_directory {
                        write_or_validate_gauge_fixed_superfield_column_shard(ordinal, directory)
                    } else {
                        gauge_fixed_superfield_column_digest(ordinal)
                    };
                    let failed = result.is_err();
                    if failed {
                        cancelled.store(true, Ordering::Release);
                    }
                    if sender.send((ordinal, result)).is_err() {
                        break;
                    }
                    if failed {
                        break;
                    }
                }
            });
        }
        drop(sender);
        for _ in 0..321 {
            let (ordinal, result) = receiver
                .recv()
                .map_err(|error| format!("complete-F worker channel closed: {error}"))?;
            let column = result?;
            progress(&column);
            completed[ordinal] = Some(column);
        }
        Ok(())
    })?;
    let columns = completed
        .into_iter()
        .enumerate()
        .map(|(ordinal, column)| {
            column.ok_or_else(|| format!("source column {ordinal} did not complete"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total_nonzero_terms = columns.iter().try_fold(0_u64, |total, column| {
        total
            .checked_add(column.nonzero_terms as u64)
            .ok_or_else(|| "superfield-curvature term count overflow".to_string())
    })?;
    let mut hasher = Sha256::new();
    hash_bytes_with_length(&mut hasher, SUPERFIELD_OPERATOR_SCHEMA.as_bytes());
    hasher.update(321_u64.to_le_bytes());
    hasher.update(total_nonzero_terms.to_le_bytes());
    for column in &columns {
        hasher.update((column.ordinal as u64).to_le_bytes());
        hasher.update((column.nonzero_terms as u64).to_le_bytes());
        hasher.update((column.raw_w_bianchi_residual_terms as u64).to_le_bytes());
        hasher.update((column.raw_w_candidate_residual_terms as u64).to_le_bytes());
        hash_bytes_with_length(&mut hasher, column.source_coordinate.as_bytes());
        hash_bytes_with_length(&mut hasher, column.sha256.as_bytes());
    }
    Ok(GaugeFixedSuperfieldOperatorCertificate {
        schema_version: SUPERFIELD_OPERATOR_SCHEMA,
        source_basis: "320 Cartesian gamma-traceless spatial-frame basis vectors followed by scale",
        source_dimension: 321,
        gamma_traceless_h_dimension: 320,
        scale_dimension: 1,
        output_basis: "linearized super-Weyl-covariant X_[2], X_[5], J^(-), auxiliary raw W_2021, direct off-shell LinearizedRiemann, direct Eq. (25) gravitino curl, and the separate conditional Eq. (40) Psi_[3] four-form candidate with canonical exterior-D and eleven-momentum monomials",
        unified_output_schema: SUPERFIELD_UNIFIED_OUTPUT_SCHEMA,
        column_shard_schema: SUPERFIELD_COLUMN_SHARD_SCHEMA,
        column_shard_directory: shard_directory
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        columns,
        total_nonzero_terms,
        operator_sha256: format!("{:x}", hasher.finalize()),
        direct_riemann_integrated: true,
        direct_gravitino_curl_integrated: true,
        direct_candidate_four_form_integrated: true,
        raw_w2021_two_d_terms_are_not_gravity: true,
        physical_target_component_adapter_complete: false,
        exact_polynomial_kernel_derived: false,
    })
}

pub fn write_gauge_fixed_superfield_operator_certificate<F>(
    path: &Path,
    progress: F,
) -> io::Result<GaugeFixedSuperfieldOperatorCertificate>
where
    F: FnMut(&GaugeFixedSuperfieldColumnDigest),
{
    let shard_directory = PathBuf::from(format!("{}.columns-v3", path.display()));
    let certificate = build_gauge_fixed_superfield_operator_certificate_internal(
        Some(&shard_directory),
        progress,
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    atomic_json(path, &certificate)?;
    Ok(certificate)
}

#[cfg(feature = "cuda")]
mod exact_cuda_sparse {
    use std::collections::BTreeMap;
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr::NonNull;

    use num_rational::Ratio;

    use crate::eleven_dimensional_physical_curvature::{ExactQi, SparseQiOperator};

    const ERROR_CAPACITY: usize = 1024;
    const MAX_DENSE_OUTPUT_BYTES_PER_CONTEXT: usize = 64 * 1024 * 1024;
    static CUDA_CONTEXT_DESTROY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CudaSparseEntry {
        row: u32,
        real: i64,
        imaginary: i64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CudaSparseInput {
        lane: u32,
        column: u32,
        real: i64,
        imaginary: i64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CudaSparseOutput {
        lane: u32,
        row: u32,
        real: i64,
        imaginary: i64,
    }

    const _: () = {
        assert!(std::mem::size_of::<CudaSparseEntry>() == 24);
        assert!(std::mem::offset_of!(CudaSparseEntry, row) == 0);
        assert!(std::mem::offset_of!(CudaSparseEntry, real) == 8);
        assert!(std::mem::offset_of!(CudaSparseEntry, imaginary) == 16);
        assert!(std::mem::size_of::<CudaSparseInput>() == 24);
        assert!(std::mem::offset_of!(CudaSparseInput, lane) == 0);
        assert!(std::mem::offset_of!(CudaSparseInput, column) == 4);
        assert!(std::mem::offset_of!(CudaSparseInput, real) == 8);
        assert!(std::mem::offset_of!(CudaSparseInput, imaginary) == 16);
        assert!(std::mem::size_of::<CudaSparseOutput>() == 24);
        assert!(std::mem::offset_of!(CudaSparseOutput, lane) == 0);
        assert!(std::mem::offset_of!(CudaSparseOutput, row) == 4);
        assert!(std::mem::offset_of!(CudaSparseOutput, real) == 8);
        assert!(std::mem::offset_of!(CudaSparseOutput, imaginary) == 16);
    };

    unsafe extern "C" {
        fn adynkra_complete_f_sparse_create(
            device: i32,
            input_dimension: u32,
            output_dimension: u32,
            offsets: *const u32,
            entries: *const CudaSparseEntry,
            entry_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn adynkra_complete_f_sparse_apply_batch(
            context: *mut c_void,
            inputs: *const CudaSparseInput,
            input_count: u32,
            lane_count: u32,
            output_real: *mut i64,
            output_imaginary: *mut i64,
            expanded_products: *mut u64,
            elapsed_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_complete_f_sparse_apply_compact_batch(
            context: *mut c_void,
            inputs: *const CudaSparseInput,
            input_count: u32,
            lane_count: u32,
            outputs: *mut CudaSparseOutput,
            output_capacity: u64,
            output_count: *mut u64,
            expanded_products: *mut u64,
            elapsed_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_complete_f_sparse_apply_composed_batch(
            first_context: *mut c_void,
            second_context: *mut c_void,
            inputs: *const CudaSparseInput,
            input_count: u32,
            lane_count: u32,
            output_real: *mut i64,
            output_imaginary: *mut i64,
            expanded_products: *mut u64,
            elapsed_milliseconds: *mut f32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn adynkra_complete_f_sparse_resident_bytes(context: *const c_void) -> u64;
        fn adynkra_complete_f_sparse_destroy(context: *mut c_void);
    }

    fn gcd(mut left: i128, mut right: i128) -> i128 {
        left = left.abs();
        right = right.abs();
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }

    fn lcm(left: i128, right: i128) -> Result<i128, String> {
        if left == 0 || right == 0 {
            return Err("zero denominator in complete-F sparse operator".to_string());
        }
        (left / gcd(left, right))
            .checked_mul(right)
            .ok_or_else(|| "complete-F denominator LCM overflow".to_string())
    }

    fn coefficient_denominator_lcm<'a>(
        values: impl Iterator<Item = &'a ExactQi>,
    ) -> Result<i64, String> {
        let mut denominator = 1_i128;
        for value in values {
            denominator = lcm(denominator, i128::from(*value.real.denom()))?;
            denominator = lcm(denominator, i128::from(*value.imaginary.denom()))?;
        }
        i64::try_from(denominator)
            .map_err(|_| "complete-F common denominator exceeds i64".to_string())
    }

    fn scaled_component(value: &Ratio<i64>, denominator: i64) -> Result<i64, String> {
        if denominator % *value.denom() != 0 {
            return Err("complete-F denominator does not clear coefficient".to_string());
        }
        i64::try_from(i128::from(*value.numer()) * i128::from(denominator / *value.denom()))
            .map_err(|_| "complete-F scaled coefficient exceeds i64".to_string())
    }

    fn unsigned_l1(real: i64, imaginary: i64) -> u128 {
        real.unsigned_abs() as u128 + imaginary.unsigned_abs() as u128
    }

    fn error_string(buffer: &[c_char]) -> String {
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
        if message.is_empty() {
            "complete-F CUDA backend failed without an error message".to_string()
        } else {
            message.into_owned()
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ExactCudaSparseStats {
        pub(crate) lane_count: usize,
        pub(crate) input_nonzeros: usize,
        pub(crate) output_nonzeros: usize,
        pub(crate) expanded_products: u64,
        pub(crate) kernel_milliseconds: f32,
        pub(crate) resident_bytes: u64,
        pub(crate) absolute_accumulation_bound: u128,
    }

    pub(crate) struct ExactCudaSparseOperator {
        context: NonNull<c_void>,
        input_dimension: usize,
        output_dimension: usize,
        operator_denominator: i64,
        column_l1: Vec<u128>,
        column_entry_l1: Vec<Vec<(usize, u128)>>,
    }

    impl ExactCudaSparseOperator {
        pub(crate) fn new(operator: &SparseQiOperator, device: i32) -> Result<Self, String> {
            if operator.columns.len() != operator.input_dimension {
                return Err("complete-F sparse operator has incomplete columns".to_string());
            }
            let operator_denominator = coefficient_denominator_lcm(
                operator
                    .columns
                    .iter()
                    .flatten()
                    .map(|entry| &entry.coefficient),
            )?;
            let mut offsets = Vec::with_capacity(operator.input_dimension + 1);
            let mut entries = Vec::new();
            let mut column_l1 = Vec::with_capacity(operator.input_dimension);
            let mut column_entry_l1 = Vec::with_capacity(operator.input_dimension);
            offsets.push(0_u32);
            for column in &operator.columns {
                let mut l1 = 0_u128;
                let mut entry_l1 = Vec::with_capacity(column.len());
                for entry in column {
                    let real = scaled_component(&entry.coefficient.real, operator_denominator)?;
                    let imaginary =
                        scaled_component(&entry.coefficient.imaginary, operator_denominator)?;
                    l1 = l1
                        .checked_add(unsigned_l1(real, imaginary))
                        .ok_or_else(|| "complete-F column norm overflow".to_string())?;
                    entry_l1.push((entry.row, unsigned_l1(real, imaginary)));
                    entries.push(CudaSparseEntry {
                        row: u32::try_from(entry.row)
                            .map_err(|_| "complete-F output row exceeds u32".to_string())?,
                        real,
                        imaginary,
                    });
                }
                column_l1.push(l1);
                column_entry_l1.push(entry_l1);
                offsets.push(
                    u32::try_from(entries.len())
                        .map_err(|_| "complete-F operator entries exceed u32".to_string())?,
                );
            }
            let mut error = [0_i8; ERROR_CAPACITY];
            let context = unsafe {
                adynkra_complete_f_sparse_create(
                    device,
                    u32::try_from(operator.input_dimension)
                        .map_err(|_| "complete-F input dimension exceeds u32".to_string())?,
                    u32::try_from(operator.output_dimension)
                        .map_err(|_| "complete-F output dimension exceeds u32".to_string())?,
                    offsets.as_ptr(),
                    entries.as_ptr(),
                    u32::try_from(entries.len())
                        .map_err(|_| "complete-F entry count exceeds u32".to_string())?,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            NonNull::new(context)
                .map(|context| Self {
                    context,
                    input_dimension: operator.input_dimension,
                    output_dimension: operator.output_dimension,
                    operator_denominator,
                    column_l1,
                    column_entry_l1,
                })
                .ok_or_else(|| error_string(&error))
        }

        pub(crate) fn apply(
            &self,
            input: &BTreeMap<usize, ExactQi>,
        ) -> Result<(BTreeMap<usize, ExactQi>, ExactCudaSparseStats), String> {
            let (mut outputs, stats) = self.apply_batch(std::slice::from_ref(input))?;
            Ok((outputs.pop().unwrap(), stats))
        }

        pub(crate) fn apply_batch(
            &self,
            batch: &[BTreeMap<usize, ExactQi>],
        ) -> Result<(Vec<BTreeMap<usize, ExactQi>>, ExactCudaSparseStats), String> {
            if batch.is_empty() {
                return Err("complete-F CUDA batch must contain at least one lane".to_string());
            }
            let bytes_per_lane = self
                .output_dimension
                .checked_mul(2 * std::mem::size_of::<i64>())
                .ok_or_else(|| "complete-F CUDA lane byte count overflow".to_string())?;
            if bytes_per_lane > MAX_DENSE_OUTPUT_BYTES_PER_CONTEXT {
                return Err(format!(
                    "complete-F CUDA lane requires {bytes_per_lane} dense bytes, exceeding the {} byte per-context cap",
                    MAX_DENSE_OUTPUT_BYTES_PER_CONTEXT
                ));
            }
            let lanes_per_chunk = if bytes_per_lane == 0 {
                batch.len()
            } else {
                MAX_DENSE_OUTPUT_BYTES_PER_CONTEXT / bytes_per_lane
            };
            if batch.len() <= lanes_per_chunk {
                return self.apply_batch_chunk(batch);
            }

            let mut outputs = Vec::with_capacity(batch.len());
            let mut stats = ExactCudaSparseStats {
                lane_count: 0,
                input_nonzeros: 0,
                output_nonzeros: 0,
                expanded_products: 0,
                kernel_milliseconds: 0.0,
                resident_bytes: 0,
                absolute_accumulation_bound: 0,
            };
            for chunk in batch.chunks(lanes_per_chunk) {
                let (mut chunk_outputs, chunk_stats) = self.apply_batch_chunk(chunk)?;
                outputs.append(&mut chunk_outputs);
                stats.lane_count = stats
                    .lane_count
                    .checked_add(chunk_stats.lane_count)
                    .ok_or_else(|| "complete-F CUDA lane count overflow".to_string())?;
                stats.input_nonzeros = stats
                    .input_nonzeros
                    .checked_add(chunk_stats.input_nonzeros)
                    .ok_or_else(|| "complete-F CUDA input count overflow".to_string())?;
                stats.output_nonzeros = stats
                    .output_nonzeros
                    .checked_add(chunk_stats.output_nonzeros)
                    .ok_or_else(|| "complete-F CUDA output count overflow".to_string())?;
                stats.expanded_products = stats
                    .expanded_products
                    .checked_add(chunk_stats.expanded_products)
                    .ok_or_else(|| "complete-F CUDA product count overflow".to_string())?;
                stats.kernel_milliseconds += chunk_stats.kernel_milliseconds;
                stats.resident_bytes = stats.resident_bytes.max(chunk_stats.resident_bytes);
                stats.absolute_accumulation_bound = stats
                    .absolute_accumulation_bound
                    .max(chunk_stats.absolute_accumulation_bound);
            }
            Ok((outputs, stats))
        }

        fn apply_batch_chunk(
            &self,
            batch: &[BTreeMap<usize, ExactQi>],
        ) -> Result<(Vec<BTreeMap<usize, ExactQi>>, ExactCudaSparseStats), String> {
            let mut inputs = Vec::new();
            let mut output_denominators = Vec::with_capacity(batch.len());
            let mut absolute_accumulation_bound = 0_u128;
            for (lane, input) in batch.iter().enumerate() {
                if let Some(column) = input.keys().find(|column| **column >= self.input_dimension) {
                    return Err(format!(
                        "complete-F CUDA input column {column} is outside {}",
                        self.input_dimension
                    ));
                }
                let input_denominator = coefficient_denominator_lcm(input.values())?;
                output_denominators.push(
                    self.operator_denominator
                        .checked_mul(input_denominator)
                        .ok_or_else(|| "complete-F output denominator overflow".to_string())?,
                );
                let mut lane_bound = 0_u128;
                for (&column, coefficient) in input {
                    let real = scaled_component(&coefficient.real, input_denominator)?;
                    let imaginary = scaled_component(&coefficient.imaginary, input_denominator)?;
                    lane_bound = lane_bound
                        .checked_add(
                            unsigned_l1(real, imaginary)
                                .checked_mul(self.column_l1[column])
                                .ok_or_else(|| {
                                    "complete-F absolute accumulation bound overflow".to_string()
                                })?,
                        )
                        .ok_or_else(|| {
                            "complete-F absolute accumulation bound overflow".to_string()
                        })?;
                    inputs.push(CudaSparseInput {
                        lane: u32::try_from(lane)
                            .map_err(|_| "complete-F CUDA lane exceeds u32".to_string())?,
                        column: u32::try_from(column)
                            .map_err(|_| "complete-F input column exceeds u32".to_string())?,
                        real,
                        imaginary,
                    });
                }
                if lane_bound > i64::MAX as u128 {
                    return Err(format!(
                        "complete-F exact CUDA lane {lane} accumulation bound {lane_bound} exceeds i64"
                    ));
                }
                absolute_accumulation_bound = absolute_accumulation_bound.max(lane_bound);
            }

            let dense_output_count = batch
                .len()
                .checked_mul(self.output_dimension)
                .ok_or_else(|| "complete-F batched output size overflow".to_string())?;
            let expanded_capacity = inputs.iter().try_fold(0_usize, |total, input| {
                total
                    .checked_add(self.column_entry_l1[input.column as usize].len())
                    .ok_or_else(|| "complete-F compact output capacity overflow".to_string())
            })?;
            let output_capacity = dense_output_count.min(expanded_capacity);
            let mut compact_outputs =
                Vec::<std::mem::MaybeUninit<CudaSparseOutput>>::with_capacity(output_capacity);
            unsafe { compact_outputs.set_len(output_capacity) };
            let mut output_count = 0_u64;
            let mut expanded_products = 0_u64;
            let mut kernel_milliseconds = 0_f32;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_complete_f_sparse_apply_compact_batch(
                    self.context.as_ptr(),
                    inputs.as_ptr(),
                    u32::try_from(inputs.len())
                        .map_err(|_| "complete-F input nonzeros exceed u32".to_string())?,
                    u32::try_from(batch.len())
                        .map_err(|_| "complete-F CUDA lane count exceeds u32".to_string())?,
                    compact_outputs.as_mut_ptr().cast::<CudaSparseOutput>(),
                    u64::try_from(output_capacity).map_err(|_| {
                        "complete-F compact output capacity exceeds u64".to_string()
                    })?,
                    &mut output_count,
                    &mut expanded_products,
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            let output_count = usize::try_from(output_count)
                .map_err(|_| "complete-F compact output count exceeds usize".to_string())?;
            if output_count > output_capacity {
                return Err("complete-F compact output count exceeds capacity".to_string());
            }
            let mut outputs = vec![BTreeMap::new(); batch.len()];
            for output in &compact_outputs[..output_count] {
                let output = unsafe { output.assume_init_ref() };
                let lane = output.lane as usize;
                let row = output.row as usize;
                if lane >= outputs.len() || row >= self.output_dimension {
                    return Err("complete-F compact CUDA output is out of range".to_string());
                }
                let value = ExactQi {
                    real: Ratio::new(output.real, output_denominators[lane]),
                    imaginary: Ratio::new(output.imaginary, output_denominators[lane]),
                };
                if value.is_zero() || outputs[lane].insert(row, value).is_some() {
                    return Err("complete-F compact CUDA output is noncanonical".to_string());
                }
            }
            let output_nonzeros = output_count;
            let resident_bytes =
                unsafe { adynkra_complete_f_sparse_resident_bytes(self.context.as_ptr()) };
            let stats = ExactCudaSparseStats {
                lane_count: batch.len(),
                input_nonzeros: inputs.len(),
                output_nonzeros,
                expanded_products,
                kernel_milliseconds,
                resident_bytes,
                absolute_accumulation_bound,
            };
            Ok((outputs, stats))
        }

        pub(crate) fn apply_composed_batch(
            &self,
            second: &Self,
            batch: &[BTreeMap<usize, ExactQi>],
        ) -> Result<(Vec<BTreeMap<usize, ExactQi>>, ExactCudaSparseStats), String> {
            if batch.is_empty() {
                return Err("complete-F CUDA batch must contain at least one lane".to_string());
            }
            if self.output_dimension != second.input_dimension {
                return Err(format!(
                    "complete-F composed dimensions do not match: {} != {}",
                    self.output_dimension, second.input_dimension
                ));
            }
            batch
                .len()
                .checked_mul(self.output_dimension)
                .ok_or_else(|| {
                    "complete-F composed intermediate output size overflow".to_string()
                })?;
            let composed_denominator = self
                .operator_denominator
                .checked_mul(second.operator_denominator)
                .ok_or_else(|| "complete-F composed operator denominator overflow".to_string())?;
            let mut inputs = Vec::new();
            let mut output_denominators = Vec::with_capacity(batch.len());
            let mut absolute_accumulation_bound = 0_u128;
            for (lane, input) in batch.iter().enumerate() {
                if let Some(column) = input.keys().find(|column| **column >= self.input_dimension) {
                    return Err(format!(
                        "complete-F CUDA input column {column} is outside {}",
                        self.input_dimension
                    ));
                }
                let input_denominator = coefficient_denominator_lcm(input.values())?;
                output_denominators.push(
                    composed_denominator
                        .checked_mul(input_denominator)
                        .ok_or_else(|| {
                            "complete-F composed output denominator overflow".to_string()
                        })?,
                );
                let mut first_lane_bound = 0_u128;
                let mut lane_bound = 0_u128;
                for (&column, coefficient) in input {
                    let real = scaled_component(&coefficient.real, input_denominator)?;
                    let imaginary = scaled_component(&coefficient.imaginary, input_denominator)?;
                    let input_l1 = unsigned_l1(real, imaginary);
                    first_lane_bound = first_lane_bound
                        .checked_add(input_l1.checked_mul(self.column_l1[column]).ok_or_else(
                            || "complete-F first composed accumulation bound overflow".to_string(),
                        )?)
                        .ok_or_else(|| {
                            "complete-F first composed accumulation bound overflow".to_string()
                        })?;
                    let mut composed_column_l1 = 0_u128;
                    for &(middle, first_l1) in &self.column_entry_l1[column] {
                        composed_column_l1 = composed_column_l1
                            .checked_add(
                                first_l1.checked_mul(second.column_l1[middle]).ok_or_else(
                                    || "complete-F composed column norm overflow".to_string(),
                                )?,
                            )
                            .ok_or_else(|| {
                                "complete-F composed column norm overflow".to_string()
                            })?;
                    }
                    lane_bound = lane_bound
                        .checked_add(input_l1.checked_mul(composed_column_l1).ok_or_else(|| {
                            "complete-F composed accumulation bound overflow".to_string()
                        })?)
                        .ok_or_else(|| {
                            "complete-F composed accumulation bound overflow".to_string()
                        })?;
                    inputs.push(CudaSparseInput {
                        lane: u32::try_from(lane)
                            .map_err(|_| "complete-F CUDA lane exceeds u32".to_string())?,
                        column: u32::try_from(column)
                            .map_err(|_| "complete-F input column exceeds u32".to_string())?,
                        real,
                        imaginary,
                    });
                }
                if first_lane_bound > i64::MAX as u128 {
                    return Err(format!(
                        "complete-F exact composed CUDA lane {lane} first-stage accumulation bound {first_lane_bound} exceeds i64"
                    ));
                }
                if lane_bound > i64::MAX as u128 {
                    return Err(format!(
                        "complete-F exact composed CUDA lane {lane} accumulation bound {lane_bound} exceeds i64"
                    ));
                }
                absolute_accumulation_bound = absolute_accumulation_bound
                    .max(first_lane_bound)
                    .max(lane_bound);
            }
            let output_count = batch
                .len()
                .checked_mul(second.output_dimension)
                .ok_or_else(|| "complete-F composed batched output size overflow".to_string())?;
            let mut output_real = vec![0_i64; output_count];
            let mut output_imaginary = vec![0_i64; output_count];
            let mut expanded_products = 0_u64;
            let mut kernel_milliseconds = 0_f32;
            let mut error = [0_i8; ERROR_CAPACITY];
            let status = unsafe {
                adynkra_complete_f_sparse_apply_composed_batch(
                    self.context.as_ptr(),
                    second.context.as_ptr(),
                    inputs.as_ptr(),
                    u32::try_from(inputs.len())
                        .map_err(|_| "complete-F input nonzeros exceed u32".to_string())?,
                    u32::try_from(batch.len())
                        .map_err(|_| "complete-F CUDA lane count exceeds u32".to_string())?,
                    output_real.as_mut_ptr(),
                    output_imaginary.as_mut_ptr(),
                    &mut expanded_products,
                    &mut kernel_milliseconds,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_string(&error));
            }
            let mut outputs = Vec::with_capacity(batch.len());
            let mut output_nonzeros = 0_usize;
            for (lane, output_denominator) in output_denominators.into_iter().enumerate() {
                let mut output = BTreeMap::new();
                let begin = lane * second.output_dimension;
                let end = begin + second.output_dimension;
                for (row, (&real, &imaginary)) in output_real[begin..end]
                    .iter()
                    .zip(&output_imaginary[begin..end])
                    .enumerate()
                {
                    if real == 0 && imaginary == 0 {
                        continue;
                    }
                    output.insert(
                        row,
                        ExactQi {
                            real: Ratio::new(real, output_denominator),
                            imaginary: Ratio::new(imaginary, output_denominator),
                        },
                    );
                }
                output_nonzeros += output.len();
                outputs.push(output);
            }
            let resident_bytes = unsafe {
                adynkra_complete_f_sparse_resident_bytes(self.context.as_ptr())
                    + adynkra_complete_f_sparse_resident_bytes(second.context.as_ptr())
            };
            Ok((
                outputs,
                ExactCudaSparseStats {
                    lane_count: batch.len(),
                    input_nonzeros: inputs.len(),
                    output_nonzeros,
                    expanded_products,
                    kernel_milliseconds,
                    resident_bytes,
                    absolute_accumulation_bound,
                },
            ))
        }
    }

    impl Drop for ExactCudaSparseOperator {
        fn drop(&mut self) {
            let _guard = CUDA_CONTEXT_DESTROY_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { adynkra_complete_f_sparse_destroy(self.context.as_ptr()) };
        }
    }
}

fn first_nonempty_column(operator: &physical::SparseQiOperator) -> usize {
    operator
        .columns
        .iter()
        .position(|column| !column.is_empty())
        .expect("source-fixed operator is nonzero")
}

fn deterministic_probe() -> GeometryLevelPhysicalFOutput {
    let c_to_j_one = physical::c_alpha_beta_gamma_to_j_one_operator();
    let c_to_omega = physical::c_alpha_b_c_to_spinorial_connection_operator();
    let c_to_j_two = physical::c_alpha_b_c_to_j_operator();
    let t_to_w = physical::t_alpha_e_gamma_to_w_operator();

    let mut input = GeometryLevelPhysicalFInput::default();
    input.d_h.insert(0, ExactQi::one());
    input
        .c_alpha_beta_gamma
        .insert(first_nonempty_column(&c_to_j_one), ExactQi::from_integer(2));
    input
        .c_alpha_b_c
        .insert(first_nonempty_column(&c_to_omega), ExactQi::from_integer(3));
    input
        .c_alpha_b_c
        .insert(first_nonempty_column(&c_to_j_two), ExactQi::from_integer(5));

    input.first_jet.d_c_alpha_beta_gamma.insert(
        3 * c_to_j_one.input_dimension + first_nonempty_column(&c_to_j_one),
        ExactQi::from_integer(7),
    );
    input.first_jet.d_c_alpha_b_c.insert(
        5 * c_to_omega.input_dimension + first_nonempty_column(&c_to_omega),
        ExactQi::from_integer(11),
    );
    input.first_jet.d_c_alpha_b_c.insert(
        7 * c_to_j_two.input_dimension + first_nonempty_column(&c_to_j_two),
        ExactQi::from_integer(13),
    );
    input
        .first_jet
        .c_alpha_a_gamma
        .insert(first_nonempty_column(&t_to_w), ExactQi::from_integer(17));

    assemble_geometry_level_physical_f(&input).expect("deterministic geometry probe is valid")
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFSectorStatus {
    pub sector: &'static str,
    pub domain: &'static str,
    pub codomain: &'static str,
    pub exact_operator_available: bool,
    pub composed_from_h_hat: bool,
    pub source: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletePhysicalFConstructionReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_hashes: Vec<&'static str>,
    pub sectors: Vec<PhysicalFSectorStatus>,
    pub geometry_level_join_implemented: bool,
    pub probe_x_two_entries: usize,
    pub probe_x_five_entries: usize,
    pub probe_j_one_entries: usize,
    pub probe_j_two_entries: usize,
    pub probe_j_plus_entries: usize,
    pub probe_j_minus_entries: usize,
    pub probe_t_entries: usize,
    pub probe_w_2001_entries: usize,
    pub probe_w_2021_entries: usize,
    pub h_hat_to_consistent_geometry_jet_implemented: bool,
    pub ordered_superderivative_normal_form_complete: bool,
    pub bounded_linearized_frame_jet_stream_implemented: bool,
    pub gamma_traceless_h_hat_input_projected: bool,
    pub local_lorentz_orbit_raw_response_is_nonzero: bool,
    pub local_lorentz_gauge_fixed_at_input: bool,
    pub local_lorentz_orbit_descends_through_j_t_w: bool,
    pub linearized_invariant_supercurvature_basis_fixed: bool,
    pub conditional_raw_w_lowest_component_adapter_implemented: bool,
    pub raw_w_identified_as_physical_four_form: bool,
    pub direct_off_shell_frame_to_riemann_adapter_implemented: bool,
    pub direct_riemann_integrated_into_321_column_operator: bool,
    pub direct_riemann_bianchi_certified: bool,
    pub direct_eq25_gravitino_potential_adapter_implemented: bool,
    pub direct_gravitino_curl_eq29_torsion_certified: bool,
    pub direct_gravitino_curl_target_bianchi_certified: bool,
    pub direct_gravitino_euler_image_computed_and_checked: bool,
    pub direct_gravitino_noether_certified: bool,
    pub direct_gravitino_stream_persisted_in_v4_col3_schema: bool,
    pub eq40_psi_three_candidate_four_form_adapter_implemented: bool,
    pub eq40_psi_three_candidate_bianchi_certified: bool,
    pub eq40_psi_three_candidate_euler_noether_certified: bool,
    pub raw_w_four_form_target_bianchi_certified: bool,
    pub psi_three_identified_as_physical_a3: bool,
    pub candidate_raw_w_relative_normalization_fixed: bool,
    pub nonzero_h_canary_raw_w_terms: usize,
    pub nonzero_h_canary_candidate_four_form_terms: usize,
    pub nonzero_h_canary_raw_minus_candidate_terms: usize,
    pub nonzero_h_canary_raw_w_bianchi_residual_terms: usize,
    pub conditional_w2021_first_descendant_adapter_implemented: bool,
    pub conditional_gravitino_curl_forward_image_gate_implemented: bool,
    pub conditional_gravitino_curl_target_bianchi_gate_implemented: bool,
    pub conditional_gravitino_curl_stream_requires_explicit_convention: bool,
    pub conditional_w2021_identification_passes_scale_canary: bool,
    pub theta_two_w_gravity_double_emitted: bool,
    pub complete_all_sector_target_curvature_adapter_implemented: bool,
    pub complete_all_sector_target_bianchi_euler_noether_composition_certified: bool,
    pub complete_physical_f_implemented: bool,
    pub complete_f_operator_sha256: Option<String>,
    pub exact_polynomial_target_kernel_derived: bool,
    pub pointwise_or_bounded_kernel_is_accepted_as_physical_k: bool,
    pub next_executable_step: &'static str,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

fn build_report() -> CompletePhysicalFConstructionReport {
    let probe = deterministic_probe();
    let geometry_level_join_implemented = !probe.x.x_two_11000.is_empty()
        && !probe.x.x_five_10002.is_empty()
        && !probe.j_one.is_empty()
        && !probe.j_two.is_empty()
        && !probe.j_plus.is_empty()
        && !probe.j_minus.is_empty()
        && !probe.first_jet.t_alpha_a_gamma.is_empty()
        && !probe.first_jet.w_2001.is_empty()
        && !probe.first_jet.w_2021.is_empty();
    let curvature = physical::verify();
    let jet = first_jet::verify();
    let local_lorentz = crate::eleven_dimensional_j1_lorentz_residual::verify();
    let target = crate::eleven_dimensional_target_equation_complex::verify();
    let derivative_normal_form = crate::eleven_dimensional_superderivative_normal_form::verify();
    let frame_jet = crate::eleven_dimensional_h_hat_jet::verify();
    let constrained = crate::eleven_dimensional_constrained_geometry_jet::verify();
    let passed = geometry_level_join_implemented
        && curvature.bounded_slice_passed
        && jet.passed
        && local_lorentz.passed
        && target.passed
        && derivative_normal_form.passed
        && frame_jet.passed
        && constrained.passed;

    CompletePhysicalFConstructionReport {
        schema_version: SCHEMA_VERSION,
        role: "exact geometry-level physical-F assembly and fail-closed route to a target-kernel-derived K",
        source_hashes: vec![
            physical::HEP_TH_0101037_SOURCE_SHA256,
            physical::ARXIV_2007_05097_SOURCE_SHA256,
            physical::HEP_TH_0107155_SOURCE_SHA256,
        ],
        sectors: vec![
            PhysicalFSectorStatus {
                sector: "X_[2]",
                domain: "D H",
                codomain: "(11000), rank 429 conventional quotient",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eqs. (39)-(40), (44)",
            },
            PhysicalFSectorStatus {
                sector: "X_[5]",
                domain: "D H",
                codomain: "(10002), rank 4290 conventional quotient",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eqs. (39)-(40), (44)",
            },
            PhysicalFSectorStatus {
                sector: "J^(1), J^(2), J^(+), J^(-)",
                domain: "spinor and mixed anholonomy plus spinorial connection",
                codomain: "spinor geometry",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eq. (44), Table 3; arXiv:2007.05097 Eqs. (2.19)-(2.22)",
            },
            PhysicalFSectorStatus {
                sector: "T and omega",
                domain: "geometry anholonomy and first jet",
                codomain: "mixed torsion and Lorentz connections",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Table 3; hep-th/0107155 Eqs. (3.2c)-(3.2e)",
            },
            PhysicalFSectorStatus {
                sector: "auxiliary raw W",
                domain: "T and D J",
                codomain: "source W_[4] coordinates, not identified with physical F_[4]",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eq. (44); arXiv:2007.05097 Eqs. (2.22)-(2.23)",
            },
            PhysicalFSectorStatus {
                sector: "direct graviton-curvature target branch",
                domain: "gauge-fixed Eq. (25) bosonic frame",
                codomain: "all 1,210 algebraic Riemann coordinates and target Bianchi",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eq. (25); repository exact graviton target complex",
            },
            PhysicalFSectorStatus {
                sector: "direct gravitino-curvature target branch",
                domain: "gauge-fixed Eq. (25) D Delta plus D Psi vector-spinor",
                codomain: "all 1,760 curl coordinates, checked Euler image, Bianchi, and Noether",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eqs. (7)-(8), (25), (29); hep-th/0107155 Eq. (3.2f); repository exact Rarita-Schwinger target complex",
            },
            PhysicalFSectorStatus {
                sector: "conditional Eq. (40) Psi_[3] four-form target branch",
                domain: "gauge-fixed D H through the multiplicity-one conventional Psi_[3] solve",
                codomain: "all 330 closed four-form coordinates, checked Euler image, Bianchi, and Noether",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eq. (40) conventional constraints plus repository exact Abelian three-form target complex; physical A_[3] identification unprinted",
            },
            PhysicalFSectorStatus {
                sector: "complete all-sector physical target curvature assembly",
                domain: "completed H_hat curvature data",
                codomain: "Riemann, four-form, gravitino curl, Bianchi, Euler, Noether",
                exact_operator_available: true,
                composed_from_h_hat: false,
                source: "repository exact 44+84|128 target equation complex",
            },
        ],
        geometry_level_join_implemented,
        probe_x_two_entries: probe.x.x_two_11000.len(),
        probe_x_five_entries: probe.x.x_five_10002.len(),
        probe_j_one_entries: probe.j_one.len(),
        probe_j_two_entries: probe.j_two.len(),
        probe_j_plus_entries: probe.j_plus.len(),
        probe_j_minus_entries: probe.j_minus.len(),
        probe_t_entries: probe.first_jet.t_alpha_a_gamma.len(),
        probe_w_2001_entries: probe.first_jet.w_2001.len(),
        probe_w_2021_entries: probe.first_jet.w_2021.len(),
        h_hat_to_consistent_geometry_jet_implemented: constrained.passed,
        ordered_superderivative_normal_form_complete: derivative_normal_form.passed,
        bounded_linearized_frame_jet_stream_implemented: frame_jet.passed,
        gamma_traceless_h_hat_input_projected: true,
        local_lorentz_orbit_raw_response_is_nonzero: true,
        local_lorentz_gauge_fixed_at_input: true,
        local_lorentz_orbit_descends_through_j_t_w: false,
        linearized_invariant_supercurvature_basis_fixed: true,
        conditional_raw_w_lowest_component_adapter_implemented: true,
        raw_w_identified_as_physical_four_form: false,
        direct_off_shell_frame_to_riemann_adapter_implemented: true,
        direct_riemann_integrated_into_321_column_operator: true,
        direct_riemann_bianchi_certified: true,
        direct_eq25_gravitino_potential_adapter_implemented: true,
        direct_gravitino_curl_eq29_torsion_certified: true,
        direct_gravitino_curl_target_bianchi_certified: true,
        direct_gravitino_euler_image_computed_and_checked: true,
        direct_gravitino_noether_certified: true,
        direct_gravitino_stream_persisted_in_v4_col3_schema: true,
        eq40_psi_three_candidate_four_form_adapter_implemented: true,
        eq40_psi_three_candidate_bianchi_certified: true,
        eq40_psi_three_candidate_euler_noether_certified: true,
        raw_w_four_form_target_bianchi_certified: false,
        psi_three_identified_as_physical_a3: false,
        candidate_raw_w_relative_normalization_fixed: false,
        nonzero_h_canary_raw_w_terms: 45_260,
        nonzero_h_canary_candidate_four_form_terms: 648,
        nonzero_h_canary_raw_minus_candidate_terms: 45_260,
        nonzero_h_canary_raw_w_bianchi_residual_terms: 312_704,
        conditional_w2021_first_descendant_adapter_implemented: true,
        conditional_gravitino_curl_forward_image_gate_implemented: true,
        conditional_gravitino_curl_target_bianchi_gate_implemented: true,
        conditional_gravitino_curl_stream_requires_explicit_convention: true,
        conditional_w2021_identification_passes_scale_canary: false,
        theta_two_w_gravity_double_emitted: false,
        complete_all_sector_target_curvature_adapter_implemented: false,
        complete_all_sector_target_bianchi_euler_noether_composition_certified: false,
        complete_physical_f_implemented: false,
        complete_f_operator_sha256: None,
        exact_polynomial_target_kernel_derived: false,
        pointwise_or_bounded_kernel_is_accepted_as_physical_k: false,
        next_executable_step: "regenerate and validate the v4/COL3 successor columns, then determine the physical A_[3] identification/relative W normalization and construct the physical target gauge quotient before deriving K",
        passed,
        result: "The exact ordered-superderivative frame composes H_hat and scale through Delta, both Eqs. (13)-(14) anholonomies, D C, both Lorentz connections, J, mixed torsion, and both W conventions. P_320 and the p=2 gauge choice are enforced at the typed input boundary. The v4/COL3 successor stream preserves the frozen X_[2], X_[5], J^(-), auxiliary raw W_2021, and direct Riemann sectors, and adds the direct Eq. (25) gravitino curl plus a separately labeled conditional Eq. (40) Psi_[3] four-form candidate. Riemann, the gravitino curl, and the candidate four-form each pass their target Bianchi and computed-Euler-to-Noether gates exactly. Raw W is never relabeled as target F_[4]. On the exact nonzero-H canary it has 45,260 terms, versus 648 candidate terms, their residual has 45,260 terms, and raw W has 312,704 nonzero Bianchi-residual terms. Therefore raw W is neither equal nor proportional to the closed candidate; the residual has the same Bianchi image entrywise. Full physical F remains incomplete because the source does not identify Psi_[3] with physical A_[3], and the target gauge quotient is not constructed.",
        boundary: "Passing this report certifies the exact gauge-fixed H_hat-to-X/J/T/W differential stream, the direct full off-shell Riemann branch, the direct kinematic gravitino curvature on the canonical local-Lorentz gauge section, and the closed Eq. (40) holonomy Psi_[3] candidate branch. It does not prove off-shell closure, raw p=2 descent, or a physical A_[3] identification. The candidate is unique only as the multiplicity-one Lambda-three conventional projection with source-fixed 1/16 coefficient; it is not claimed as the unique general H-to-A_[3] map. Euler images are computed and checked but not emitted. The v4/COL3 schema persists direct Riemann, direct gravitino curl, conditional candidate four-form, and auxiliary raw W as distinct tags; frozen v3/COL2 is rejected. The raw-W first-descendant identification remains rejected. Complete all-sector physical composition and physical K remain false until the A_[3]/W normalization and target gauge quotient are fixed; no diagnostic rank is labeled as physical K.",
    }
}

pub fn verify() -> CompletePhysicalFConstructionReport {
    static REPORT: OnceLock<CompletePhysicalFConstructionReport> = OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub fn write_artifact(path: &Path) -> io::Result<CompletePhysicalFConstructionReport> {
    let report = verify();
    atomic_json(path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_pair_ordinal(left: usize, right: usize) -> usize {
        assert!(left < right);
        let mut ordinal = 0;
        for a in 0..physical::VECTOR_DIMENSION {
            for b in (a + 1)..physical::VECTOR_DIMENSION {
                if (a, b) == (left, right) {
                    return ordinal;
                }
                ordinal += 1;
            }
        }
        unreachable!()
    }

    fn metric_field_ordinal(left: usize, right: usize) -> usize {
        assert!(left <= right);
        let mut ordinal = 0;
        for a in 0..physical::VECTOR_DIMENSION {
            for b in a..physical::VECTOR_DIMENSION {
                if (a, b) == (left, right) {
                    return ordinal;
                }
                ordinal += 1;
            }
        }
        unreachable!()
    }

    fn scalar_polynomial(value: i64) -> CanonicalSuperPolynomial {
        CanonicalSuperPolynomial::scalar(ExactQi::from_integer(value))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_matches_exact_j_two_operator() {
        let operator = physical::cached_c_alpha_b_c_to_j_operator();
        let columns = operator
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| !column.is_empty())
            .map(|(column, _)| column)
            .take(3)
            .collect::<Vec<_>>();
        assert_eq!(columns.len(), 3);
        let mut input = BTreeMap::new();
        input.insert(
            columns[0],
            ExactQi {
                real: num_rational::Ratio::new(3, 5),
                imaginary: num_rational::Ratio::new(-2, 7),
            },
        );
        input.insert(
            columns[1],
            ExactQi {
                real: num_rational::Ratio::new(-11, 13),
                imaginary: num_rational::Ratio::new(5, 9),
            },
        );
        input.insert(columns[2], ExactQi::from_rational(17, 19));

        let expected = operator.apply_sparse(&input);
        let gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(operator, 0).unwrap();
        let (actual, stats) = gpu.apply(&input).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.lane_count, 1);
        assert_eq!(stats.input_nonzeros, input.len());
        assert_eq!(stats.output_nonzeros, expected.len());
        assert!(stats.expanded_products > 0);
        assert!(stats.kernel_milliseconds >= 0.0);
        assert!(stats.resident_bytes > 0);
        assert!(stats.absolute_accumulation_bound > 0);
        eprintln!("complete-F exact CUDA sparse stats: {stats:#?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_compacts_only_after_exact_cancellation() {
        let operator = physical::SparseQiOperator {
            input_dimension: 3,
            output_dimension: 1,
            columns: vec![
                vec![physical::SparseQiEntry {
                    row: 0,
                    coefficient: ExactQi::from_integer(1),
                }],
                vec![physical::SparseQiEntry {
                    row: 0,
                    coefficient: ExactQi::from_integer(-1),
                }],
                vec![physical::SparseQiEntry {
                    row: 0,
                    coefficient: ExactQi::from_integer(1),
                }],
            ],
        };
        let batch = vec![
            BTreeMap::from([
                (0, ExactQi::from_integer(17)),
                (1, ExactQi::from_integer(17)),
            ]),
            BTreeMap::from([
                (0, ExactQi::from_integer(23)),
                (1, ExactQi::from_integer(23)),
                (2, ExactQi::from_integer(-5)),
            ]),
            BTreeMap::new(),
        ];
        let expected = batch
            .iter()
            .map(|input| operator.apply_sparse(input))
            .collect::<Vec<_>>();
        assert!(expected[0].is_empty());
        assert_eq!(
            expected[1],
            BTreeMap::from([(0, ExactQi::from_integer(-5))])
        );
        assert!(expected[2].is_empty());

        let gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&operator, 0).unwrap();
        for _ in 0..100 {
            let (actual, stats) = gpu.apply_batch(&batch).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(stats.output_nonzeros, 1);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_context_teardown_is_concurrent_safe() {
        let operator = physical::SparseQiOperator {
            input_dimension: 1,
            output_dimension: 1,
            columns: vec![vec![physical::SparseQiEntry {
                row: 0,
                coefficient: ExactQi::from_integer(3),
            }]],
        };
        std::thread::scope(|scope| {
            for lane in 0..8_i64 {
                let operator = &operator;
                scope.spawn(move || {
                    for iteration in 0..4_i64 {
                        let gpu =
                            exact_cuda_sparse::ExactCudaSparseOperator::new(operator, 0).unwrap();
                        let input =
                            BTreeMap::from([(0, ExactQi::from_integer(lane + iteration + 1))]);
                        let expected = operator.apply_sparse(&input);
                        assert_eq!(gpu.apply(&input).unwrap().0, expected);
                    }
                });
            }
        });
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_batch_matches_exact_connection_operator() {
        let operator = physical::cached_c_alpha_b_c_to_spinorial_connection_operator();
        let columns = operator
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| !column.is_empty())
            .map(|(column, _)| column)
            .take(64)
            .collect::<Vec<_>>();
        assert_eq!(columns.len(), 64);
        let batch = (0..96)
            .map(|lane| {
                columns
                    .iter()
                    .enumerate()
                    .filter(|(position, _)| (position + lane) % 5 == 0)
                    .map(|(position, &column)| {
                        (
                            column,
                            ExactQi {
                                real: num_rational::Ratio::new(
                                    i64::try_from(position + lane + 1).unwrap(),
                                    3,
                                ),
                                imaginary: num_rational::Ratio::new(
                                    i64::try_from(position).unwrap() - i64::try_from(lane).unwrap(),
                                    7,
                                ),
                            },
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let expected = batch
            .iter()
            .map(|input| operator.apply_sparse(input))
            .collect::<Vec<_>>();
        let gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(operator, 0).unwrap();
        let (actual, stats) = gpu.apply_batch(&batch).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.lane_count, batch.len());
        assert_eq!(
            stats.input_nonzeros,
            batch.iter().map(BTreeMap::len).sum::<usize>()
        );
        assert_eq!(
            stats.output_nonzeros,
            expected.iter().map(BTreeMap::len).sum::<usize>()
        );
        eprintln!("complete-F exact CUDA batch stats: {stats:#?}");

        let iterations = 200_u32;
        let cpu_started = std::time::Instant::now();
        for _ in 0..iterations {
            let output = batch
                .iter()
                .map(|input| operator.apply_sparse(input))
                .collect::<Vec<_>>();
            std::hint::black_box(output);
        }
        let cpu_elapsed = cpu_started.elapsed();
        let gpu_started = std::time::Instant::now();
        let mut gpu_kernel_milliseconds = 0_f64;
        for _ in 0..iterations {
            let (output, iteration_stats) = gpu.apply_batch(&batch).unwrap();
            gpu_kernel_milliseconds += f64::from(iteration_stats.kernel_milliseconds);
            std::hint::black_box(output);
        }
        let gpu_elapsed = gpu_started.elapsed();
        let cpu_per_batch = cpu_elapsed.as_secs_f64() / f64::from(iterations);
        let gpu_per_batch = gpu_elapsed.as_secs_f64() / f64::from(iterations);
        let kernel_per_batch = gpu_kernel_milliseconds / f64::from(iterations) / 1_000.0;
        eprintln!(
            "complete-F sparse batch benchmark: products={} CPU={:.6}ms CUDA-kernel={:.6}ms CUDA-end-to-end={:.6}ms kernel-vs-CPU={:.3}x end-to-end={:.3}x",
            stats.expanded_products,
            cpu_per_batch * 1_000.0,
            kernel_per_batch * 1_000.0,
            gpu_per_batch * 1_000.0,
            cpu_per_batch / kernel_per_batch,
            cpu_per_batch / gpu_per_batch,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_composed_batch_matches_exact_x_two_chain() {
        let first = physical::gamma_dh_operator(2);
        let second = physical::hook_projector_operator(2);
        assert_eq!(first.output_dimension, second.input_dimension);
        let columns = first
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| !column.is_empty())
            .map(|(column, _)| column)
            .take(16)
            .collect::<Vec<_>>();
        let batch = (0..8)
            .map(|lane| {
                columns
                    .iter()
                    .enumerate()
                    .filter(|(position, _)| (position + lane) % 3 == 0)
                    .map(|(position, &column)| {
                        (
                            column,
                            ExactQi {
                                real: num_rational::Ratio::new(
                                    i64::try_from(position + lane + 1).unwrap(),
                                    5,
                                ),
                                imaginary: num_rational::Ratio::new(
                                    i64::try_from(position).unwrap() - i64::try_from(lane).unwrap(),
                                    7,
                                ),
                            },
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let expected = batch
            .iter()
            .map(|input| second.apply_sparse(&first.apply_sparse(input)))
            .collect::<Vec<_>>();
        let first_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&first, 0).unwrap();
        let second_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&second, 0).unwrap();
        let (actual, stats) = first_gpu.apply_composed_batch(&second_gpu, &batch).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.lane_count, batch.len());
        assert_eq!(
            stats.output_nonzeros,
            expected.iter().map(BTreeMap::len).sum::<usize>()
        );
        assert!(stats.expanded_products > 0);
        assert!(stats.kernel_milliseconds >= 0.0);
        eprintln!("complete-F exact composed CUDA stats: {stats:#?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_composed_batch_matches_exact_x_five_chain() {
        let first = physical::gamma_dh_operator(5);
        let second = physical::hook_projector_operator(5);
        assert_eq!(first.output_dimension, second.input_dimension);
        let columns = first
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| !column.is_empty())
            .map(|(column, _)| column)
            .take(4)
            .collect::<Vec<_>>();
        let batch = (0..2)
            .map(|lane| {
                columns
                    .iter()
                    .enumerate()
                    .map(|(position, &column)| {
                        (
                            column,
                            ExactQi {
                                real: num_rational::Ratio::new(
                                    i64::try_from(position + lane + 1).unwrap(),
                                    3,
                                ),
                                imaginary: num_rational::Ratio::new(
                                    i64::try_from(position).unwrap() - i64::try_from(lane).unwrap(),
                                    5,
                                ),
                            },
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let expected = batch
            .iter()
            .map(|input| second.apply_sparse(&first.apply_sparse(input)))
            .collect::<Vec<_>>();
        let first_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&first, 0).unwrap();
        let second_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&second, 0).unwrap();
        let (actual, stats) = first_gpu.apply_composed_batch(&second_gpu, &batch).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.lane_count, batch.len());
        assert_eq!(
            stats.output_nonzeros,
            expected.iter().map(BTreeMap::len).sum::<usize>()
        );
        eprintln!("complete-F exact composed X5 CUDA stats: {stats:#?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_sparse_cuda_composed_rejects_each_stage_overflow_and_aliasing() {
        let first_large = physical::SparseQiOperator {
            input_dimension: 1,
            output_dimension: 1,
            columns: vec![vec![physical::SparseQiEntry {
                row: 0,
                coefficient: ExactQi::from_integer(i64::MAX),
            }]],
        };
        let zero_second = physical::SparseQiOperator {
            input_dimension: 1,
            output_dimension: 1,
            columns: vec![Vec::new()],
        };
        let first_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&first_large, 0).unwrap();
        let zero_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&zero_second, 0).unwrap();
        let input = vec![BTreeMap::from([(0, ExactQi::from_integer(2))])];
        let first_error = first_gpu
            .apply_composed_batch(&zero_gpu, &input)
            .unwrap_err();
        assert!(first_error.contains("first-stage accumulation bound"));

        let identity = physical::SparseQiOperator {
            input_dimension: 1,
            output_dimension: 1,
            columns: vec![vec![physical::SparseQiEntry {
                row: 0,
                coefficient: ExactQi::one(),
            }]],
        };
        let second_large = physical::SparseQiOperator {
            input_dimension: 1,
            output_dimension: 1,
            columns: vec![vec![physical::SparseQiEntry {
                row: 0,
                coefficient: ExactQi::from_integer(i64::MAX),
            }]],
        };
        let identity_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&identity, 0).unwrap();
        let second_gpu = exact_cuda_sparse::ExactCudaSparseOperator::new(&second_large, 0).unwrap();
        let second_error = identity_gpu
            .apply_composed_batch(&second_gpu, &input)
            .unwrap_err();
        assert!(second_error.contains("accumulation bound"));

        let alias_error = identity_gpu
            .apply_composed_batch(&identity_gpu, &input)
            .unwrap_err();
        assert!(alias_error.contains("invalid complete-F composed"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn complete_f_geometry_cuda_batch_matches_cpu() {
        let c_to_j_one = physical::c_alpha_beta_gamma_to_j_one_operator();
        let c_to_omega = physical::c_alpha_b_c_to_spinorial_connection_operator();
        let c_to_j_two = physical::c_alpha_b_c_to_j_operator();
        let batch = (0..24)
            .map(|lane| {
                let mut input = GeometryLevelPhysicalFInput::default();
                input
                    .d_h
                    .insert(lane % physical::DH_DIMENSION, ExactQi::one());
                input.c_alpha_beta_gamma.insert(
                    first_nonempty_column(&c_to_j_one),
                    ExactQi::from_integer(i64::try_from(lane).unwrap() + 2),
                );
                input.c_alpha_b_c.insert(
                    first_nonempty_column(&c_to_omega),
                    ExactQi::from_integer(i64::try_from(lane).unwrap() + 3),
                );
                input.c_alpha_b_c.insert(
                    first_nonempty_column(&c_to_j_two),
                    ExactQi::from_rational(i64::try_from(lane).unwrap() + 5, 11),
                );
                input
            })
            .collect::<Vec<_>>();
        let expected = batch
            .iter()
            .map(assemble_geometry_level_physical_f)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let actual = assemble_geometry_level_physical_f_cuda_batch(&batch).unwrap();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "RTX 4090 complete-F source-column benchmark"]
    fn complete_f_gpu_source_column_benchmark() {
        let (_, input) = source_basis_input(0).unwrap();
        visit_frame_composed_physical_f_internal(&input, false, true, |_| Ok(())).unwrap();
        visit_frame_composed_physical_f_internal(&input, true, true, |_| Ok(())).unwrap();

        let mut cpu_entries = Vec::new();
        let cpu_started = std::time::Instant::now();
        let cpu_stats = visit_frame_composed_physical_f_internal(&input, false, true, |entry| {
            cpu_entries.push(entry);
            Ok(())
        })
        .unwrap();
        let cpu_elapsed = cpu_started.elapsed();

        let mut gpu_entries = Vec::new();
        let gpu_started = std::time::Instant::now();
        let gpu_stats = visit_frame_composed_physical_f_internal(&input, true, true, |entry| {
            gpu_entries.push(entry);
            Ok(())
        })
        .unwrap();
        let gpu_elapsed = gpu_started.elapsed();

        assert_eq!(gpu_entries, cpu_entries);
        assert_eq!(gpu_stats, cpu_stats);
        eprintln!(
            "complete-F source column 0: CPU={:.6}s CUDA={:.6}s speedup={:.3}x entries={}",
            cpu_elapsed.as_secs_f64(),
            gpu_elapsed.as_secs_f64(),
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64(),
            cpu_entries.len()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "RTX 4090 end-to-end COL3 column benchmark"]
    fn complete_f_gpu_col3_column_zero_benchmark() {
        unsafe { std::env::remove_var("ADYNKRA_COMPLETE_F_CUDA") };
        let cpu_started = std::time::Instant::now();
        let cpu = gauge_fixed_superfield_column_digest(0).unwrap();
        let cpu_elapsed = cpu_started.elapsed();

        unsafe { std::env::set_var("ADYNKRA_COMPLETE_F_CUDA", "1") };
        let gpu_started = std::time::Instant::now();
        let gpu = gauge_fixed_superfield_column_digest(0).unwrap();
        let gpu_elapsed = gpu_started.elapsed();
        unsafe { std::env::remove_var("ADYNKRA_COMPLETE_F_CUDA") };

        assert_eq!(gpu, cpu);
        assert_eq!(gpu.nonzero_terms, 95_105);
        assert_eq!(
            gpu.sha256,
            "f2ea64e3d7c9ea35698f4d6b98680606889464caacfa9fdbc25d6fbbd7902997"
        );
        eprintln!(
            "complete-F COL3 column 0: CPU={:.6}s CUDA={:.6}s speedup={:.3}x terms={}",
            cpu_elapsed.as_secs_f64(),
            gpu_elapsed.as_secs_f64(),
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64(),
            gpu.nonzero_terms
        );
    }

    #[test]
    fn geometry_level_join_reaches_every_existing_sector() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert!(report.geometry_level_join_implemented);
        assert!(report.probe_x_two_entries > 0);
        assert!(report.probe_x_five_entries > 0);
        assert!(report.probe_j_one_entries > 0);
        assert!(report.probe_j_two_entries > 0);
        assert!(report.probe_j_plus_entries > 0);
        assert!(report.probe_j_minus_entries > 0);
        assert!(report.probe_t_entries > 0);
        assert!(report.probe_w_2001_entries > 0);
        assert!(report.probe_w_2021_entries > 0);
    }

    #[test]
    fn incomplete_h_hat_composition_cannot_claim_f_or_k() {
        let report = verify();
        assert!(report.h_hat_to_consistent_geometry_jet_implemented);
        assert!(report.ordered_superderivative_normal_form_complete);
        assert!(report.bounded_linearized_frame_jet_stream_implemented);
        assert!(!report.local_lorentz_orbit_descends_through_j_t_w);
        assert!(report.gamma_traceless_h_hat_input_projected);
        assert!(report.local_lorentz_orbit_raw_response_is_nonzero);
        assert!(report.local_lorentz_gauge_fixed_at_input);
        assert!(report.linearized_invariant_supercurvature_basis_fixed);
        assert!(report.conditional_raw_w_lowest_component_adapter_implemented);
        assert!(!report.raw_w_identified_as_physical_four_form);
        assert!(report.direct_off_shell_frame_to_riemann_adapter_implemented);
        assert!(report.direct_riemann_integrated_into_321_column_operator);
        assert!(report.direct_riemann_bianchi_certified);
        assert!(report.direct_eq25_gravitino_potential_adapter_implemented);
        assert!(report.direct_gravitino_curl_eq29_torsion_certified);
        assert!(report.direct_gravitino_curl_target_bianchi_certified);
        assert!(report.direct_gravitino_euler_image_computed_and_checked);
        assert!(report.direct_gravitino_noether_certified);
        assert!(report.direct_gravitino_stream_persisted_in_v4_col3_schema);
        assert!(report.eq40_psi_three_candidate_four_form_adapter_implemented);
        assert!(report.eq40_psi_three_candidate_bianchi_certified);
        assert!(report.eq40_psi_three_candidate_euler_noether_certified);
        assert!(!report.raw_w_four_form_target_bianchi_certified);
        assert!(!report.psi_three_identified_as_physical_a3);
        assert!(!report.candidate_raw_w_relative_normalization_fixed);
        assert_eq!(report.nonzero_h_canary_raw_w_terms, 45_260);
        assert_eq!(report.nonzero_h_canary_candidate_four_form_terms, 648);
        assert_eq!(report.nonzero_h_canary_raw_minus_candidate_terms, 45_260);
        assert_eq!(
            report.nonzero_h_canary_raw_w_bianchi_residual_terms,
            312_704
        );
        assert!(report.conditional_w2021_first_descendant_adapter_implemented);
        assert!(report.conditional_gravitino_curl_forward_image_gate_implemented);
        assert!(report.conditional_gravitino_curl_target_bianchi_gate_implemented);
        assert!(report.conditional_gravitino_curl_stream_requires_explicit_convention);
        assert!(!report.conditional_w2021_identification_passes_scale_canary);
        assert!(!report.theta_two_w_gravity_double_emitted);
        assert!(!report.complete_all_sector_target_curvature_adapter_implemented);
        assert!(!report.complete_all_sector_target_bianchi_euler_noether_composition_certified);
        assert!(!report.complete_physical_f_implemented);
        assert!(report.complete_f_operator_sha256.is_none());
        assert!(!report.exact_polynomial_target_kernel_derived);
        assert!(!report.pointwise_or_bounded_kernel_is_accepted_as_physical_k);
    }

    #[test]
    fn malformed_geometry_coordinate_fails_before_assembly() {
        let mut input = GeometryLevelPhysicalFInput::default();
        input.d_h.insert(physical::DH_DIMENSION, ExactQi::one());
        let error = assemble_geometry_level_physical_f(&input).unwrap_err();
        assert!(error.contains("outside dimension"));
    }

    #[test]
    fn w2021_lowest_component_adapts_to_the_exact_four_form_target() {
        let adapter = W2021FourFormTargetAdapter;
        let coordinate = FrameComposedPhysicalFCoordinate {
            sector: FrameComposedPhysicalFSector::W2021,
            coordinate: 17,
            monomial: OrderedSuperderivativeMonomial {
                exterior_spinor_mask: 0,
                momentum:
                    crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                        exponents: [0; physical::VECTOR_DIMENSION],
                    },
            },
        };
        let output = adapter
            .apply_source_coordinate(&coordinate, &ExactQi::from_rational(3, 5))
            .unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].0.sector, TargetCurvatureSector::FourForm);
        assert_eq!(output[0].0.component, 38);
        assert_eq!(output[0].1, ExactQi::from_rational(3, 5));

        let mut wrong = coordinate.clone();
        wrong.sector = FrameComposedPhysicalFSector::W2001;
        assert!(
            adapter
                .apply_source_coordinate(&wrong, &ExactQi::one())
                .is_err()
        );
        wrong.sector = FrameComposedPhysicalFSector::W2021;
        wrong.monomial.exterior_spinor_mask = 1;
        assert!(
            adapter
                .apply_source_coordinate(&wrong, &ExactQi::one())
                .is_err()
        );
    }

    #[test]
    fn w2021_adapter_exhaustively_permutates_mask_order_to_target_lexicographic_order() {
        let adapter = W2021FourFormTargetAdapter;
        let mut observed = BTreeSet::new();
        for source_ordinal in 0..physical::W_FOUR_FORM_DIMENSION {
            let coordinate = FrameComposedPhysicalFCoordinate {
                sector: FrameComposedPhysicalFSector::W2021,
                coordinate: source_ordinal,
                monomial: OrderedSuperderivativeMonomial {
                    exterior_spinor_mask: 0,
                    momentum: crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                        exponents: [0; physical::VECTOR_DIMENSION],
                    },
                },
            };
            let target = adapter
                .apply_source_coordinate(&coordinate, &ExactQi::one())
                .unwrap();
            assert_eq!(target.len(), 1);
            let expected = target_four_form_lexicographic_ordinal(
                w_source_four_form_indices(source_ordinal).unwrap(),
            );
            assert_eq!(target[0].0.component, expected);
            assert!(observed.insert(expected));
        }
        assert_eq!(
            observed,
            (0..physical::W_FOUR_FORM_DIMENSION).collect::<BTreeSet<_>>()
        );
        assert_eq!(
            target_four_form_lexicographic_ordinal(w_source_four_form_indices(2).unwrap()),
            8
        );
    }

    #[test]
    fn w2021_first_descendant_recovers_full_target_gravitino_curl_and_rejects_off_image() {
        let operator = physical::linearized_gravitino_curl_to_d_f_four_operator();
        let mut curl = BTreeMap::new();
        curl.insert(0, ExactQi::from_rational(3, 7));
        curl.insert(319, ExactQi::i());
        curl.insert(1_759, ExactQi::from_integer(-2));
        let descendant = operator.apply_sparse(&curl);
        let mut entries = Vec::new();
        for (row, coefficient) in descendant {
            let alpha = row / physical::W_FOUR_FORM_DIMENSION;
            let target_four_form = row % physical::W_FOUR_FORM_DIMENSION;
            let source_coordinate = (0..physical::W_FOUR_FORM_DIMENSION)
                .find(|source| {
                    target_four_form_lexicographic_ordinal(
                        w_source_four_form_indices(*source).unwrap(),
                    ) == target_four_form
                })
                .unwrap();
            entries.push(FrameComposedPhysicalFEntry {
                sector: FrameComposedPhysicalFSector::W2021,
                coordinate: source_coordinate,
                monomial: OrderedSuperderivativeMonomial {
                    exterior_spinor_mask: 1_u32 << alpha,
                    momentum: crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                        exponents: [0; physical::VECTOR_DIMENSION],
                    },
                },
                coefficient,
            });
        }
        let adapted = adapt_w2021_first_descendants(&entries).unwrap();
        let expected = curl
            .into_iter()
            .map(|(component, coefficient)| {
                (
                    TargetCurvatureCoordinate {
                        sector: TargetCurvatureSector::GravitinoCurl,
                        component,
                        momentum: MomentumMonomial::constant(),
                    },
                    coefficient,
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(adapted, expected);

        let off_image = vec![FrameComposedPhysicalFEntry {
            sector: FrameComposedPhysicalFSector::W2021,
            coordinate: 0,
            monomial: OrderedSuperderivativeMonomial {
                exterior_spinor_mask: 1,
                momentum:
                    crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                        exponents: [0; physical::VECTOR_DIMENSION],
                    },
            },
            coefficient: ExactQi::one(),
        }];
        assert!(adapt_w2021_first_descendants(&off_image).is_err());
    }

    #[test]
    fn explicit_first_descendant_round_trip_preserves_masks_and_target_bianchi() {
        let exterior_mask = (1_u32 << 2) | (1_u32 << 17);
        let seed_monomial = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: exterior_mask,
            momentum:
                crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial::constant(),
        };
        let target = target_sector_complex(TargetSector::RaritaSchwinger);
        let mut curl = BTreeMap::new();
        for (component, term) in target.curvature.column_terms(0) {
            let monomial = multiply_ordered_momentum(&seed_monomial, &term.monomial).unwrap();
            curl.insert(
                (component, monomial),
                multiply_exact_qi_by_public(&ExactQi::one(), &term),
            );
        }
        assert!(!curl.is_empty());
        assert!(
            curl.keys()
                .all(|(_, monomial)| monomial.exterior_spinor_mask == exterior_mask)
        );
        assert!(gravitino_curl_bianchi_residual(&curl).unwrap().is_empty());
        let mut bianchi_mutation = curl.clone();
        bianchi_mutation.insert((0, seed_monomial.clone()), ExactQi::one());
        assert!(
            !gravitino_curl_bianchi_residual(&bianchi_mutation)
                .unwrap()
                .is_empty()
        );

        let mut curl_by_monomial =
            BTreeMap::<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>::new();
        for ((component, monomial), coefficient) in &curl {
            curl_by_monomial
                .entry(monomial.clone())
                .or_default()
                .insert(*component, coefficient.clone());
        }
        let operator = physical::linearized_gravitino_curl_to_d_f_four_operator();
        let mut descendants = Vec::new();
        for (monomial, one_monomial_curl) in curl_by_monomial {
            for (row, coefficient) in operator.apply_sparse(&one_monomial_curl) {
                let derivative_spinor = row / physical::W_FOUR_FORM_DIMENSION;
                let target_four_form = row % physical::W_FOUR_FORM_DIMENSION;
                let source_coordinate = (0..physical::W_FOUR_FORM_DIMENSION)
                    .find(|source| {
                        target_four_form_lexicographic_ordinal(
                            w_source_four_form_indices(*source).unwrap(),
                        ) == target_four_form
                    })
                    .unwrap();
                descendants.push(ExplicitW2021FirstDescendantEntry {
                    derivative_spinor,
                    four_form_coordinate: source_coordinate,
                    monomial: monomial.clone(),
                    coefficient,
                });
            }
        }
        let recovered = adapt_explicit_w2021_first_descendants(&descendants).unwrap();
        assert_eq!(recovered, curl);

        descendants.push(ExplicitW2021FirstDescendantEntry {
            derivative_spinor: 0,
            four_form_coordinate: 0,
            monomial: seed_monomial,
            coefficient: ExactQi::one(),
        });
        assert!(adapt_explicit_w2021_first_descendants(&descendants).is_err());
    }

    #[test]
    fn explicit_differentiation_retains_alpha_separately_from_ordered_d_mask() {
        let raw = vec![FrameComposedPhysicalFEntry {
            sector: FrameComposedPhysicalFSector::W2021,
            coordinate: 0,
            monomial: OrderedSuperderivativeMonomial {
                exterior_spinor_mask: 1,
                momentum:
                    crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial::constant(),
            },
            coefficient: ExactQi::one(),
        }];
        let descendants = explicitly_differentiate_w2021(&raw).unwrap();
        let repeated = descendants
            .iter()
            .filter(|entry| entry.derivative_spinor == 0)
            .collect::<Vec<_>>();
        assert!(!repeated.is_empty());
        assert!(repeated.iter().all(|entry| {
            entry.monomial.exterior_spinor_mask == 0
                && entry.monomial.momentum.exponents.iter().sum::<u16>() == 1
        }));
        assert!(
            descendants
                .iter()
                .filter(|entry| entry.derivative_spinor == 1)
                .all(|entry| entry.monomial.exterior_spinor_mask == 3
                    || entry.monomial.exterior_spinor_mask == 0)
        );
    }

    #[test]
    fn direct_d_scale_path_matches_the_full_frame_jet_reference() {
        let input = LinearizedFrameSuperfields {
            scale: scalar_polynomial(3),
            ..LinearizedFrameSuperfields::default()
        };
        let representative = canonical_physical_frame_representative(&input).unwrap();
        let (_, direct) = eq25_fermionic_frame_source_polynomials(&representative).unwrap();
        let mut reference = PolynomialMap::new();
        visit_linearized_frame_jet(&representative, |entry| {
            if entry.sector == LinearizedFrameJetSector::DScale
                && !entry.polynomial.terms.is_empty()
            {
                reference.insert(entry.coordinate, entry.polynomial);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(direct, reference);
    }

    #[test]
    fn direct_scale_gravitino_curl_preserves_masks_and_all_target_identities() {
        let input = LinearizedFrameSuperfields {
            scale: scalar_polynomial(1),
            ..LinearizedFrameSuperfields::default()
        };
        let mut entries = Vec::new();
        let stats = visit_gauge_fixed_linearized_gravitino_curl(&input, |entry| {
            entries.push(entry);
            Ok(())
        })
        .unwrap();
        assert!(stats.fermionic_frame_terms > 0);
        assert!(stats.gravitino_curl_terms > 0);
        assert!(stats.euler_terms > 0);
        assert_eq!(stats.eq29_torsion_residual_terms, 0);
        assert_eq!(stats.bianchi_residual_terms, 0);
        assert_eq!(stats.noether_residual_terms, 0);
        assert_eq!(entries.len(), stats.gravitino_curl_terms);
        assert!(entries.iter().all(|entry| {
            entry.monomial.exterior_spinor_mask.count_ones() == 1
                && entry.monomial.momentum.exponents.iter().sum::<u16>() == 1
        }));
        let curl = entries
            .into_iter()
            .map(|entry| ((entry.component, entry.monomial), entry.coefficient))
            .collect::<BTreeMap<_, _>>();
        assert!(gravitino_curl_bianchi_residual(&curl).unwrap().is_empty());
        let euler = gravitino_curl_euler_image(&curl).unwrap();
        assert!(!euler.is_empty());
        assert!(gravitino_curl_noether_residual(&curl).unwrap().is_empty());

        // Adapter-local corruption gate: the valid C2E image is Noether
        // closed, while one fresh Euler coordinate is detected immediately.
        let fresh_monomial = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: 1_u32 << 31,
            momentum:
                crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial::constant(),
        };
        let mut mutated_euler = euler;
        assert!(!mutated_euler.contains_key(&(0, fresh_monomial.clone())));
        mutated_euler.insert((0, fresh_monomial), ExactQi::one());
        assert!(
            !apply_target_polynomial_columns(rarita_noether_columns(), &mutated_euler)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn direct_gamma_traceless_h_canary_reaches_nonzero_certified_gravitino_curl() {
        let (_, input) = source_basis_input(0).unwrap();
        assert!(input.scale.terms.is_empty());
        let representative = canonical_physical_frame_representative(&input).unwrap();
        let (d_delta, d_scale) = eq25_fermionic_frame_source_polynomials(&representative).unwrap();
        assert!(!d_delta.is_empty());
        assert!(d_scale.is_empty());
        let source_monomials = transpose_polynomials(&d_delta)
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();

        let mut entries = Vec::new();
        let stats = visit_gauge_fixed_linearized_gravitino_curl(&input, |entry| {
            entries.push(entry);
            Ok(())
        })
        .unwrap();
        assert!(stats.fermionic_frame_terms > 0);
        assert!(stats.gravitino_curl_terms > 0);
        assert!(stats.euler_terms > 0);
        assert_eq!(stats.eq29_torsion_residual_terms, 0);
        assert_eq!(stats.bianchi_residual_terms, 0);
        assert_eq!(stats.noether_residual_terms, 0);
        assert!(entries.iter().all(|entry| {
            let output_degree = entry
                .monomial
                .momentum
                .exponents
                .iter()
                .map(|value| usize::from(*value))
                .sum::<usize>();
            source_monomials.iter().any(|source| {
                source.exterior_spinor_mask == entry.monomial.exterior_spinor_mask
                    && source
                        .momentum
                        .exponents
                        .iter()
                        .map(|value| usize::from(*value))
                        .sum::<usize>()
                        + 1
                        == output_degree
            })
        }));
    }

    #[test]
    fn target_four_form_bianchi_helper_annihilates_an_exact_curvature_column() {
        let target = target_sector_complex(TargetSector::FourForm);
        let curvature = target
            .curvature
            .column_terms(0)
            .into_iter()
            .map(|(component, term)| {
                (
                    TargetCurvatureCoordinate {
                        sector: TargetCurvatureSector::FourForm,
                        component,
                        momentum: term.monomial.clone(),
                    },
                    multiply_exact_qi_by_public(&ExactQi::one(), &term),
                )
            })
            .collect::<Vec<_>>();
        assert!(!curvature.is_empty());
        assert!(four_form_bianchi_residual(&curvature).unwrap().is_empty());
    }

    #[test]
    fn eq40_numeric_three_form_order_maps_exhaustively_to_target_lex_order() {
        let numeric_masks = (0_u16..(1_u16 << physical::VECTOR_DIMENSION))
            .filter(|mask| mask.count_ones() == 3)
            .collect::<Vec<_>>();
        assert_eq!(numeric_masks.len(), 165);
        let mapping = numeric_masks
            .iter()
            .map(|mask| psi_three_mask_to_target_ordinal(*mask).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            mapping.iter().copied().collect::<BTreeSet<_>>(),
            (0..165).collect()
        );
        assert_eq!(numeric_masks[2], 0b000_0000_1101);
        assert_eq!(mapping[2], 9, "numeric ordinal 2 is A_023 in lex order");
        for (mask, target) in numeric_masks.iter().zip(&mapping) {
            let indices = (0..physical::VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            assert_eq!(
                *target,
                target_three_form_lexicographic_ordinal([indices[0], indices[1], indices[2]])
            );
        }
        let mut ordering_mutation = mapping.clone();
        ordering_mutation.swap(0, 1);
        assert_ne!(ordering_mutation, mapping);
    }

    #[test]
    fn nonzero_h_eq40_three_form_candidate_is_closed_and_preserves_degree() {
        let (_, input) = source_basis_input(0).unwrap();
        let (stats, entries) = gauge_fixed_candidate_four_form_curvature(&input, &[]).unwrap();
        assert!(stats.psi_three_potential_terms > 0);
        assert!(stats.four_form_curvature_terms > 0);
        assert!(stats.euler_terms > 0);
        assert_eq!(stats.bianchi_residual_terms, 0);
        assert_eq!(stats.noether_residual_terms, 0);
        assert_eq!(stats.raw_w_bianchi_residual_terms, 0);
        assert_eq!(stats.raw_w_comparison_residual_terms, entries.len());
        assert!(entries.iter().all(|entry| {
            entry.monomial.exterior_spinor_mask.count_ones() == 1
                && entry.monomial.momentum.exponents.iter().sum::<u16>() == 1
        }));
        let curvature = entries
            .into_iter()
            .map(|entry| ((entry.component, entry.monomial), entry.coefficient))
            .collect::<BTreeMap<_, _>>();
        assert!(
            apply_target_polynomial_columns(four_form_bianchi_columns(), &curvature)
                .unwrap()
                .is_empty()
        );
        let euler =
            apply_target_polynomial_columns(four_form_curvature_to_euler_columns(), &curvature)
                .unwrap();
        assert!(!euler.is_empty());
        assert!(
            apply_target_polynomial_columns(four_form_noether_columns(), &euler)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn source_w_and_eq40_candidate_have_exact_relationship_on_nonzero_h_canary() {
        let (_, input) = source_basis_input(0).unwrap();
        let mut raw_w = Vec::new();
        visit_gauge_fixed_physical_f(&input, |entry| {
            if entry.sector == FrameComposedPhysicalFSector::W2021 {
                raw_w.push(GaugeFixedInvariantOutputEntry {
                    sector: GaugeFixedInvariantOutputSector::W2021Raw,
                    coordinate: entry.coordinate,
                    monomial: entry.monomial,
                    coefficient: entry.coefficient,
                });
            }
            Ok(())
        })
        .unwrap();
        assert!(!raw_w.is_empty());
        let (stats, candidate) = gauge_fixed_candidate_four_form_curvature(&input, &raw_w).unwrap();
        assert_eq!(raw_w.len(), 45_260);
        assert_eq!(candidate.len(), 648);
        assert_eq!(stats.raw_w_comparison_residual_terms, 45_260);
        assert_eq!(stats.raw_w_bianchi_residual_terms, 312_704);
        assert!(!candidate.is_empty());
        assert!(stats.raw_w_comparison_residual_terms > 0);
        assert!(stats.raw_w_bianchi_residual_terms > 0);

        // The persisted raw-W tag retains source numeric-mask ordering, while
        // the candidate tag is already in target lexicographic ordering. Build
        // a minimal real shard and require adoption to reproduce the live
        // diagnostic counts exactly. This catches treating raw-W ordinals as
        // target ordinals during resume validation.
        let mut persisted = raw_w;
        persisted.extend(
            candidate
                .into_iter()
                .map(|entry| GaugeFixedInvariantOutputEntry {
                    sector: GaugeFixedInvariantOutputSector::DirectCandidateFourForm,
                    coordinate: entry.component,
                    monomial: entry.monomial,
                    coefficient: entry.coefficient,
                }),
        );
        persisted.sort_by(invariant_entry_cmp);
        let source = "h_ordering_probe";
        let ordinal = 0_usize;
        let directory = std::env::temp_dir().join(format!(
            "adynkra-col3-raw-w-ordering-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("column_000.bin");
        let mut shard = Vec::new();
        let mut file_hasher = Sha256::new();
        write_hashed(&mut shard, &mut file_hasher, SUPERFIELD_COLUMN_SHARD_MAGIC).unwrap();
        write_hashed(
            &mut shard,
            &mut file_hasher,
            &(ordinal as u64).to_le_bytes(),
        )
        .unwrap();
        write_hashed(
            &mut shard,
            &mut file_hasher,
            &(source.len() as u64).to_le_bytes(),
        )
        .unwrap();
        write_hashed(&mut shard, &mut file_hasher, source.as_bytes()).unwrap();
        let mut semantic = Sha256::new();
        semantic.update(SUPERFIELD_OPERATOR_COLUMN_SCHEMA);
        semantic.update((ordinal as u64).to_le_bytes());
        hash_bytes_with_length(&mut semantic, source.as_bytes());
        for entry in &persisted {
            hash_invariant_entry(&mut semantic, entry);
            encode_invariant_entry(&mut shard, &mut file_hasher, entry).unwrap();
        }
        let count = persisted.len() as u64;
        semantic.update(count.to_le_bytes());
        write_hashed(&mut shard, &mut file_hasher, &count.to_le_bytes()).unwrap();
        write_hashed(&mut shard, &mut file_hasher, semantic.finalize().as_slice()).unwrap();
        fs::write(&path, shard).unwrap();
        let adopted = validate_column_shard(&path, ordinal, source).unwrap();
        assert_eq!(
            adopted.raw_w_bianchi_residual_terms,
            stats.raw_w_bianchi_residual_terms
        );
        assert_eq!(
            adopted.raw_w_candidate_residual_terms,
            stats.raw_w_comparison_residual_terms
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn target_four_form_curvature_has_the_fixed_a123_signs() {
        let monomial = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: 0,
            momentum:
                crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial::constant(),
        };
        let a123 = target_three_form_lexicographic_ordinal([1, 2, 3]);
        let input = BTreeMap::from([((a123, monomial), ExactQi::one())]);
        let image = apply_target_polynomial_columns(four_form_curvature_columns(), &input).unwrap();
        let f0123 = target_four_form_lexicographic_ordinal([0, 1, 2, 3]);
        let f1234 = target_four_form_lexicographic_ordinal([1, 2, 3, 4]);
        let p0 = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: 0,
            momentum:
                crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                    exponents: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
        };
        let p4 = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: 0,
            momentum:
                crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                    exponents: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                },
        };
        assert_eq!(image.get(&(f0123, p0)), Some(&ExactQi::one()));
        assert_eq!(image.get(&(f1234, p4)), Some(&ExactQi::from_integer(-1)));
        assert_eq!(image.len(), 8);
    }

    #[test]
    fn unified_v3_shard_schema_persists_direct_curl_and_rejects_frozen_v2() {
        assert_eq!(
            SUPERFIELD_OPERATOR_SCHEMA,
            "adynkra-11d-gauge-fixed-invariant-supercurvature-operator-v4"
        );
        assert_eq!(SUPERFIELD_COLUMN_SHARD_MAGIC, b"AD11FINVCOL3\0\0\0\0");
        assert_eq!(FROZEN_V3_COLUMN_SHARD_MAGIC, b"AD11FINVCOL2\0\0\0\0");
        assert_eq!(SUPERFIELD_COLUMN_ENTRY_BYTES, 67);

        let source = "schema_probe";
        let ordinal = 7_usize;
        let exterior = (1_u32 << 3) | (1_u32 << 19);
        let entry = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectGravitinoCurl,
            coordinate: 42,
            monomial: OrderedSuperderivativeMonomial {
                exterior_spinor_mask: exterior,
                momentum:
                    crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial {
                        exponents: [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                    },
            },
            coefficient: ExactQi::from_rational(-3, 7),
        };
        assert_eq!(
            GaugeFixedInvariantOutputSector::from_frame(FrameComposedPhysicalFSector::W2021),
            Some(GaugeFixedInvariantOutputSector::W2021Raw)
        );
        assert_eq!(
            entry.sector,
            GaugeFixedInvariantOutputSector::DirectGravitinoCurl
        );

        let mut bytes = Vec::new();
        let mut file_hasher = Sha256::new();
        encode_invariant_entry(&mut bytes, &mut file_hasher, &entry).unwrap();
        assert_eq!(bytes.len(), SUPERFIELD_COLUMN_ENTRY_BYTES);
        assert_eq!(
            bytes[0],
            GaugeFixedInvariantOutputSector::DirectGravitinoCurl.tag()
        );
        assert_eq!(
            u32::from_le_bytes(bytes[9..13].try_into().unwrap()),
            exterior
        );

        let directory = std::env::temp_dir().join(format!(
            "adynkra-unified-v3-schema-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("column_007.bin");
        let mut shard = Vec::new();
        shard.extend_from_slice(SUPERFIELD_COLUMN_SHARD_MAGIC);
        shard.extend_from_slice(&(ordinal as u64).to_le_bytes());
        shard.extend_from_slice(&(source.len() as u64).to_le_bytes());
        shard.extend_from_slice(source.as_bytes());
        shard.extend_from_slice(&bytes);
        let mut semantic = Sha256::new();
        semantic.update(SUPERFIELD_OPERATOR_COLUMN_SCHEMA);
        semantic.update((ordinal as u64).to_le_bytes());
        hash_bytes_with_length(&mut semantic, source.as_bytes());
        hash_invariant_entry(&mut semantic, &entry);
        semantic.update(1_u64.to_le_bytes());
        shard.extend_from_slice(&1_u64.to_le_bytes());
        shard.extend_from_slice(&semantic.finalize());
        fs::write(&path, &shard).unwrap();
        let validated = validate_column_shard(&path, ordinal, source).unwrap();
        assert_eq!(validated.nonzero_terms, 1);

        let out_of_order = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::XTwo,
            coordinate: 0,
            monomial: entry.monomial.clone(),
            coefficient: ExactQi::one(),
        };
        let mut out_of_order_bytes = Vec::new();
        encode_invariant_entry(&mut out_of_order_bytes, &mut Sha256::new(), &out_of_order).unwrap();
        let mut unordered_shard = Vec::new();
        unordered_shard.extend_from_slice(SUPERFIELD_COLUMN_SHARD_MAGIC);
        unordered_shard.extend_from_slice(&(ordinal as u64).to_le_bytes());
        unordered_shard.extend_from_slice(&(source.len() as u64).to_le_bytes());
        unordered_shard.extend_from_slice(source.as_bytes());
        unordered_shard.extend_from_slice(&bytes);
        unordered_shard.extend_from_slice(&out_of_order_bytes);
        let mut unordered_semantic = Sha256::new();
        unordered_semantic.update(SUPERFIELD_OPERATOR_COLUMN_SCHEMA);
        unordered_semantic.update((ordinal as u64).to_le_bytes());
        hash_bytes_with_length(&mut unordered_semantic, source.as_bytes());
        hash_invariant_entry(&mut unordered_semantic, &entry);
        hash_invariant_entry(&mut unordered_semantic, &out_of_order);
        unordered_semantic.update(2_u64.to_le_bytes());
        unordered_shard.extend_from_slice(&2_u64.to_le_bytes());
        unordered_shard.extend_from_slice(&unordered_semantic.finalize());
        fs::write(&path, unordered_shard).unwrap();
        let error = validate_column_shard(&path, ordinal, source).unwrap_err();
        assert!(error.contains("not strictly row ordered"));

        let mut out_of_bounds_direct = shard.clone();
        let first_entry = 16 + 8 + 8 + source.len();
        out_of_bounds_direct[first_entry + 1..first_entry + 9]
            .copy_from_slice(&1_760_u64.to_le_bytes());
        fs::write(&path, out_of_bounds_direct).unwrap();
        let error = validate_column_shard(&path, ordinal, source).unwrap_err();
        assert!(error.contains("invalid sector/coordinate 10/1760"));

        let mut conditional_in_successor_schema = shard.clone();
        conditional_in_successor_schema[first_entry] =
            GaugeFixedInvariantOutputSector::ConditionalGravitinoCurl.tag();
        fs::write(&path, conditional_in_successor_schema).unwrap();
        let error = validate_column_shard(&path, ordinal, source).unwrap_err();
        assert!(error.contains("non-invariant sector tag 12"));

        shard[..16].copy_from_slice(FROZEN_V3_COLUMN_SHARD_MAGIC);
        fs::write(&path, &shard).unwrap();
        let error = validate_column_shard(&path, ordinal, source).unwrap_err();
        assert!(error.contains("invalid column-shard magic"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn unified_output_order_puts_curvatures_after_raw_w_at_one_monomial() {
        let monomial = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: 3,
            momentum: crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial::constant(),
        };
        let raw_w = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::W2021Raw,
            coordinate: 329,
            monomial: monomial.clone(),
            coefficient: ExactQi::one(),
        };
        let riemann = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::LinearizedRiemann,
            coordinate: 0,
            monomial: monomial.clone(),
            coefficient: ExactQi::one(),
        };
        let gravitino = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectGravitinoCurl,
            coordinate: 0,
            monomial: monomial.clone(),
            coefficient: ExactQi::one(),
        };
        let candidate_four_form = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::DirectCandidateFourForm,
            coordinate: 0,
            monomial: monomial.clone(),
            coefficient: ExactQi::one(),
        };
        let conditional = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::ConditionalGravitinoCurl,
            coordinate: 0,
            monomial,
            coefficient: ExactQi::one(),
        };
        assert!(invariant_entry_cmp(&raw_w, &riemann).is_lt());
        assert!(invariant_entry_cmp(&riemann, &gravitino).is_lt());
        assert!(invariant_entry_cmp(&gravitino, &candidate_four_form).is_lt());
        assert!(invariant_entry_cmp(&candidate_four_form, &conditional).is_lt());
        assert_eq!(raw_w.sector.tag(), 8);
        assert_eq!(riemann.sector.tag(), 9);
        assert_eq!(gravitino.sector.tag(), 10);
        assert_eq!(candidate_four_form.sector.tag(), 11);
        assert_eq!(conditional.sector.tag(), 12);
    }

    #[test]
    fn unified_scale_column_is_strictly_ordered_and_riemann_bianchi_closed() {
        let input = LinearizedFrameSuperfields {
            scale: scalar_polynomial(1),
            ..LinearizedFrameSuperfields::default()
        };
        let mut entries = Vec::new();
        let stats = visit_gauge_fixed_invariant_supercurvature(&input, |entry| {
            entries.push(entry);
            Ok(())
        })
        .unwrap();
        assert!(stats.direct_riemann_terms > 0);
        assert_eq!(stats.direct_gravitino_curl_terms, 0);
        assert_eq!(stats.conditional_gravitino_curl_terms, 0);
        assert_eq!(
            entries.len(),
            stats.raw_invariant_terms
                + stats.direct_riemann_terms
                + stats.direct_gravitino_curl_terms
                + stats.direct_candidate_four_form_terms
                + stats.conditional_gravitino_curl_terms
        );
        assert!(
            entries
                .windows(2)
                .all(|window| invariant_entry_cmp(&window[0], &window[1]).is_lt())
        );
        let riemann = entries
            .into_iter()
            .filter(|entry| entry.sector == GaugeFixedInvariantOutputSector::LinearizedRiemann)
            .map(|entry| DirectLinearizedRiemannEntry {
                component: entry.coordinate,
                monomial: entry.monomial,
                coefficient: entry.coefficient,
            })
            .collect::<Vec<_>>();
        assert_eq!(riemann.len(), stats.direct_riemann_terms);
        assert!(
            direct_riemann_bianchi_residual(&riemann)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unified_direct_scale_stream_is_strictly_ordered_and_gated() {
        let input = LinearizedFrameSuperfields {
            scale: scalar_polynomial(1),
            ..LinearizedFrameSuperfields::default()
        };
        let mut entries = Vec::new();
        let stats =
            visit_gauge_fixed_invariant_supercurvature_with_direct_gravitino(&input, |entry| {
                entries.push(entry);
                Ok(())
            })
            .unwrap();
        assert!(stats.direct_riemann_terms > 0);
        assert!(stats.direct_gravitino_curl_terms > 0);
        assert_eq!(stats.conditional_gravitino_curl_terms, 0);
        assert!(
            entries
                .windows(2)
                .all(|window| invariant_entry_cmp(&window[0], &window[1]).is_lt())
        );
        assert_eq!(
            entries
                .iter()
                .filter(|entry| {
                    entry.sector == GaugeFixedInvariantOutputSector::DirectGravitinoCurl
                })
                .count(),
            stats.direct_gravitino_curl_terms
        );
    }

    #[test]
    fn conditional_scale_canary_is_fail_closed_under_the_w2021_identification() {
        let input = LinearizedFrameSuperfields {
            scale: scalar_polynomial(1),
            ..LinearizedFrameSuperfields::default()
        };
        let error = visit_gauge_fixed_invariant_supercurvature_with_first_descendant(
            &input,
            W2021FirstDescendantConventionGate::W2021LowestIsPhysicalFourForm,
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(
            error.contains("image gate failed")
                && error.contains("exterior mask 0x00000001")
                && error.contains("330 residual coordinates"),
            "unexpected conditional canary error: {error}"
        );
    }

    #[test]
    fn unified_scale_digest_matches_v4_shard_and_resume_validation() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-unified-v2-scale-column-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let digest = gauge_fixed_superfield_column_digest(320).unwrap();
        let written =
            write_or_validate_gauge_fixed_superfield_column_shard(320, &directory).unwrap();
        let reused =
            write_or_validate_gauge_fixed_superfield_column_shard(320, &directory).unwrap();
        assert!(!written.shard_reused);
        assert!(reused.shard_reused);
        assert_eq!(digest.nonzero_terms, written.nonzero_terms);
        assert_eq!(
            digest.raw_w_bianchi_residual_terms,
            written.raw_w_bianchi_residual_terms
        );
        assert_eq!(
            digest.raw_w_candidate_residual_terms,
            written.raw_w_candidate_residual_terms
        );
        assert_eq!(
            written.raw_w_bianchi_residual_terms,
            reused.raw_w_bianchi_residual_terms
        );
        assert_eq!(
            written.raw_w_candidate_residual_terms,
            reused.raw_w_candidate_residual_terms
        );
        assert_eq!(digest.sha256, written.sha256);
        assert_eq!(written.sha256, reused.sha256);
        assert_eq!(written.shard_sha256, reused.shard_sha256);
        assert_eq!(written.shard_byte_count, reused.shard_byte_count);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn inverse_frame_to_metric_has_mostly_plus_parity_and_kills_lorentz_orbits() {
        let mut spatial_rotation = PolynomialMap::new();
        spatial_rotation.insert(1 * physical::VECTOR_DIMENSION + 2, scalar_polynomial(1));
        spatial_rotation.insert(2 * physical::VECTOR_DIMENSION + 1, scalar_polynomial(-1));
        assert!(symmetric_metric_polynomials_from_inverse_frame(&spatial_rotation).is_empty());

        // With eta=(-,+,...,+), a Lorentz boost has f_0{}^1=f_1{}^0.
        // The lowered entries have opposite signs, so the metric variation is zero.
        let mut boost = PolynomialMap::new();
        boost.insert(1, scalar_polynomial(1));
        boost.insert(physical::VECTOR_DIMENSION, scalar_polynomial(1));
        assert!(symmetric_metric_polynomials_from_inverse_frame(&boost).is_empty());

        let monomial = OrderedSuperderivativeMonomial {
            exterior_spinor_mask: 0,
            momentum: crate::eleven_dimensional_superderivative_normal_form::FormalMomentumMonomial::constant(),
        };
        let mut time_left = PolynomialMap::new();
        time_left.insert(1, scalar_polynomial(1));
        assert_eq!(
            symmetric_metric_polynomials_from_inverse_frame(&time_left)
                [&metric_field_ordinal(0, 1)]
                .terms[&monomial],
            ExactQi::from_integer(-1)
        );
        let mut time_right = PolynomialMap::new();
        time_right.insert(physical::VECTOR_DIMENSION, scalar_polynomial(1));
        assert_eq!(
            symmetric_metric_polynomials_from_inverse_frame(&time_right)
                [&metric_field_ordinal(0, 1)]
                .terms[&monomial],
            ExactQi::one()
        );
    }

    #[test]
    fn direct_frame_curvature_obeys_riemann_symmetries_and_bianchi_exactly() {
        let mut metric = PolynomialMap::new();
        metric.insert(metric_field_ordinal(1, 2), scalar_polynomial(1));
        let curvature = riemann_from_metric_polynomials(&metric).unwrap();
        assert!(!curvature.is_empty());

        for ((component, monomial), coefficient) in &curvature {
            let left_pair = component / 55;
            let right_pair = component % 55;
            assert_eq!(
                curvature.get(&(right_pair * 55 + left_pair, monomial.clone())),
                Some(coefficient),
                "pair-exchange symmetry failed at component {component}"
            );
        }

        let monomials = curvature
            .keys()
            .map(|(_, monomial)| monomial.clone())
            .collect::<BTreeSet<_>>();
        for a in 0..physical::VECTOR_DIMENSION {
            for b in (a + 1)..physical::VECTOR_DIMENSION {
                for c in (b + 1)..physical::VECTOR_DIMENSION {
                    for d in (c + 1)..physical::VECTOR_DIMENSION {
                        let ab_cd =
                            symmetric_pair_ordinal(a, b) * 55 + symmetric_pair_ordinal(c, d);
                        let ac_bd =
                            symmetric_pair_ordinal(a, c) * 55 + symmetric_pair_ordinal(b, d);
                        let ad_bc =
                            symmetric_pair_ordinal(a, d) * 55 + symmetric_pair_ordinal(b, c);
                        for monomial in &monomials {
                            let mut cyclic = curvature
                                .get(&(ab_cd, monomial.clone()))
                                .cloned()
                                .unwrap_or_else(ExactQi::zero);
                            cyclic.add_assign(
                                &curvature
                                    .get(&(ad_bc, monomial.clone()))
                                    .cloned()
                                    .unwrap_or_else(ExactQi::zero),
                            );
                            cyclic.add_assign(
                                &curvature
                                    .get(&(ac_bd, monomial.clone()))
                                    .cloned()
                                    .unwrap_or_else(ExactQi::zero)
                                    .scaled(&num_rational::Ratio::from_integer(-1)),
                            );
                            assert!(cyclic.is_zero(), "algebraic Bianchi failed at {a}{b}{c}{d}");
                        }
                    }
                }
            }
        }

        let entries = curvature
            .into_iter()
            .map(
                |((component, monomial), coefficient)| DirectLinearizedRiemannEntry {
                    component,
                    monomial,
                    coefficient,
                },
            )
            .collect::<Vec<_>>();
        assert!(
            direct_riemann_bianchi_residual(&entries)
                .unwrap()
                .is_empty()
        );

        let mut mutated = entries;
        let mutation_index = mutated
            .iter()
            .position(|entry| {
                !target_sector_complex(TargetSector::Graviton)
                    .bianchi
                    .column_terms(entry.component)
                    .is_empty()
            })
            .unwrap();
        let delta = mutated[mutation_index].coefficient.clone();
        mutated[mutation_index].coefficient.add_assign(&delta);
        assert!(
            !direct_riemann_bianchi_residual(&mutated)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn source_fixed_scale_reaches_full_direct_riemann_path_with_two_momenta() {
        let input = LinearizedFrameSuperfields {
            scale: scalar_polynomial(1),
            ..LinearizedFrameSuperfields::default()
        };
        let mut entries = Vec::new();
        let stats = visit_gauge_fixed_linearized_riemann(&input, |entry| {
            entries.push(entry);
            Ok(())
        })
        .unwrap();
        assert_eq!(stats.source_frame_terms, physical::VECTOR_DIMENSION);
        assert_eq!(stats.symmetric_metric_terms, physical::VECTOR_DIMENSION);
        assert_eq!(stats.riemann_terms, entries.len());
        assert!(!entries.is_empty());
        assert!(entries.iter().all(|entry| {
            entry.monomial.exterior_spinor_mask == 0
                && entry
                    .monomial
                    .momentum
                    .exponents
                    .iter()
                    .map(|exponent| usize::from(*exponent))
                    .sum::<usize>()
                    == 2
        }));
        assert!(
            direct_riemann_bianchi_residual(&entries)
                .unwrap()
                .is_empty()
        );

        // The target uses all 55x55 antisymmetric-pair coordinates, whose
        // algebraic-symmetry quotient has n^2(n^2-1)/12 = 1,210 components.
        // A nonzero scalar trace on this off-shell probe confirms that the
        // adapter has not silently replaced Riemann by its 1,144-dimensional
        // Weyl projection.
        assert_eq!(
            55 * 55,
            target_sector_complex(TargetSector::Graviton)
                .curvature
                .rows()
        );
        assert_eq!(
            physical::VECTOR_DIMENSION.pow(2) * (physical::VECTOR_DIMENSION.pow(2) - 1) / 12,
            1_210
        );
        let pairs = (0..physical::VECTOR_DIMENSION)
            .flat_map(|a| ((a + 1)..physical::VECTOR_DIMENSION).map(move |b| (a, b)))
            .collect::<Vec<_>>();
        let mut scalar_trace = BTreeMap::<OrderedSuperderivativeMonomial, ExactQi>::new();
        for entry in &entries {
            let left_pair = entry.component / 55;
            let right_pair = entry.component % 55;
            if left_pair != right_pair {
                continue;
            }
            let (a, b) = pairs[left_pair];
            let signature = if a == 0 || b == 0 { -2 } else { 2 };
            let value = entry
                .coefficient
                .scaled(&num_rational::Ratio::from_integer(signature));
            let coefficient = scalar_trace
                .entry(entry.monomial.clone())
                .or_insert_with(ExactQi::zero);
            coefficient.add_assign(&value);
        }
        scalar_trace.retain(|_, coefficient| !coefficient.is_zero());
        assert!(!scalar_trace.is_empty());
    }

    #[test]
    #[ignore = "one exact full superfield-curvature source column"]
    fn gauge_fixed_superfield_column_digest_is_deterministic() {
        let first = gauge_fixed_superfield_column_digest(0).unwrap();
        let second = gauge_fixed_superfield_column_digest(0).unwrap();
        assert_eq!(first, second);
        assert!(first.nonzero_terms > 0);
        eprintln!("{first:#?}");
    }

    #[test]
    #[ignore = "one exact persisted superfield-curvature source column"]
    fn gauge_fixed_superfield_column_shard_is_exact_and_resumable() {
        let directory = std::env::temp_dir().join(format!(
            "adynkra-complete-f-column-shard-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let first = write_or_validate_gauge_fixed_superfield_column_shard(0, &directory).unwrap();
        let second = write_or_validate_gauge_fixed_superfield_column_shard(0, &directory).unwrap();
        assert!(!first.shard_reused);
        assert!(second.shard_reused);
        assert_eq!(first.ordinal, second.ordinal);
        assert_eq!(first.source_coordinate, second.source_coordinate);
        assert_eq!(first.nonzero_terms, second.nonzero_terms);
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.shard_sha256, second.shard_sha256);
        assert_eq!(first.shard_byte_count, second.shard_byte_count);
        assert!(first.nonzero_terms > 0);
    }

    #[test]
    fn frame_composition_reaches_x_j_t_and_w() {
        let mut input = LinearizedFrameSuperfields {
            scale: CanonicalSuperPolynomial::scalar(ExactQi::from_integer(2)),
            ..LinearizedFrameSuperfields::default()
        };
        input.h.insert(
            3 * physical::VECTOR_DIMENSION + 7,
            CanonicalSuperPolynomial::scalar(ExactQi::one()),
        );
        let stats = visit_frame_composed_physical_f(&input, |_| Ok(())).unwrap();
        for sector in [
            FrameComposedPhysicalFSector::XTwo,
            FrameComposedPhysicalFSector::XFive,
            FrameComposedPhysicalFSector::JOne,
            FrameComposedPhysicalFSector::JTwo,
            FrameComposedPhysicalFSector::JPlus,
            FrameComposedPhysicalFSector::JMinus,
            FrameComposedPhysicalFSector::MixedTorsion,
            FrameComposedPhysicalFSector::W2001,
            FrameComposedPhysicalFSector::W2021,
        ] {
            assert!(
                stats.emitted_by_sector.get(&sector).copied().unwrap_or(0) > 0,
                "{sector:?}"
            );
        }
    }

    #[test]
    fn pure_p2_orbit_has_no_x_and_records_its_full_j_t_w_residual() {
        let mut input = LinearizedFrameSuperfields::default();
        input
            .lorentz_two_form
            .insert(0, CanonicalSuperPolynomial::scalar(ExactQi::one()));
        let stats = visit_frame_composed_physical_f(&input, |_| Ok(())).unwrap();
        assert_eq!(
            stats
                .emitted_by_sector
                .get(&FrameComposedPhysicalFSector::XTwo)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(
            stats
                .emitted_by_sector
                .get(&FrameComposedPhysicalFSector::XFive)
                .copied()
                .unwrap_or(0),
            0
        );
        eprintln!("pure-p2 exact sector counts: {:?}", stats.emitted_by_sector);

        let fixed = visit_gauge_fixed_physical_f(&input, |_| Ok(())).unwrap();
        assert_eq!(fixed.monomial_slices, 0);
        assert!(fixed.emitted_by_sector.is_empty());
    }
}
