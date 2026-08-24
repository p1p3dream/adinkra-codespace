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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_constrained_geometry_jet::{
    ConstrainedGeometryJetSector, visit_constrained_geometry_jet,
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
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, MomentumMonomial, PhysicalFAdapterDescriptor,
    PhysicalFTargetAdapter, TargetCurvatureCoordinate, TargetCurvatureSector, TargetSector,
    target_sector_complex,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-complete-physical-f-construction-v6";

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
    let first_jet = first_jet::assemble_first_superspace_jet(&input.first_jet);

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

type PolynomialMap = BTreeMap<usize, CanonicalSuperPolynomial>;
type MonomialSlices = BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>;

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
        .collect::<BTreeSet<_>>();
    let empty = BTreeMap::new();
    let mut stats = FrameComposedPhysicalFStats {
        monomial_slices: monomials.len(),
        emitted_by_sector: BTreeMap::new(),
    };
    for monomial in monomials {
        let input = GeometryLevelPhysicalFInput {
            d_h: d_h.get(&monomial).unwrap_or(&empty).clone(),
            c_alpha_beta_gamma: c_spinor.get(&monomial).unwrap_or(&empty).clone(),
            c_alpha_b_c: c_vector.get(&monomial).unwrap_or(&empty).clone(),
            first_jet: FirstSuperspaceJetInput {
                d_c_alpha_beta_gamma: d_c_spinor.get(&monomial).unwrap_or(&empty).clone(),
                d_c_alpha_b_c: d_c_vector.get(&monomial).unwrap_or(&empty).clone(),
                c_alpha_a_gamma: c_mixed.get(&monomial).unwrap_or(&empty).clone(),
            },
        };
        let output = assemble_geometry_level_physical_f(&input)?;
        for (sector, values) in [
            (FrameComposedPhysicalFSector::XTwo, output.x.x_two_11000),
            (FrameComposedPhysicalFSector::XFive, output.x.x_five_10002),
            (FrameComposedPhysicalFSector::JOne, output.j_one),
            (FrameComposedPhysicalFSector::JTwo, output.j_two),
            (FrameComposedPhysicalFSector::JPlus, output.j_plus),
            (FrameComposedPhysicalFSector::JMinus, output.j_minus),
            (
                FrameComposedPhysicalFSector::MixedTorsion,
                output.first_jet.t_alpha_a_gamma,
            ),
            (FrameComposedPhysicalFSector::W2001, output.first_jet.w_2001),
            (FrameComposedPhysicalFSector::W2021, output.first_jet.w_2021),
        ] {
            emit_exact_sector(sector, values, &monomial, &mut stats, &mut emit)?;
        }
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

/// Convention-fixed adapter for the bosonic lowest component of the modern
/// all-real-gamma `W_[4]` superfield.  `W_[4]|_{theta=0}` is already the
/// physical four-form curvature coordinate, so this bridge changes only the
/// type and momentum representation.  Spinorial descendants and the X/J/T
/// auxiliary sectors are rejected rather than silently misidentified with
/// Riemann or gravitino curvature.
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

/// Stable sectors in the production invariant-supercurvature column format.
///
/// `W2021Raw` deliberately remains a raw superfield coordinate at every
/// exterior-D degree.  In particular, its two-D terms are not relabeled as
/// gravity.  `LinearizedRiemann` is emitted only by the independent direct
/// frame adapter, so the schema cannot double-emit a theta-two W term as a
/// second Riemann coordinate.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(u8)]
pub enum GaugeFixedInvariantOutputSector {
    XTwo = 0,
    XFive = 1,
    JMinus = 5,
    W2021Raw = 8,
    LinearizedRiemann = 9,
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
    pub emitted_by_sector: BTreeMap<GaugeFixedInvariantOutputSector, usize>,
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

    visit_gauge_fixed_physical_f(frame_input, |entry| {
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
            let candidate = riemann.next().unwrap();
            emit_checked(candidate)?;
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

pub(crate) const SUPERFIELD_OPERATOR_COLUMN_SCHEMA: &[u8] =
    b"adynkra-11d-gauge-fixed-invariant-supercurvature-column-v2\0";
pub(crate) const SUPERFIELD_OPERATOR_SCHEMA: &str =
    "adynkra-11d-gauge-fixed-invariant-supercurvature-operator-v3";
pub(crate) const SUPERFIELD_COLUMN_SHARD_SCHEMA: &str =
    "adynkra-11d-gauge-fixed-invariant-supercurvature-column-shard-v2";
pub(crate) const SUPERFIELD_UNIFIED_OUTPUT_SCHEMA: &str =
    "adynkra-11d-gauge-fixed-invariant-supercurvature-output-v1";
pub(crate) const SUPERFIELD_COLUMN_SHARD_MAGIC: &[u8; 16] = b"AD11FINVCOL2\0\0\0\0";
pub(crate) const SUPERFIELD_COLUMN_ENTRY_BYTES: usize = 67;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GaugeFixedSuperfieldColumnDigest {
    pub ordinal: usize,
    pub source_coordinate: String,
    pub nonzero_terms: usize,
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
    write_hashed(writer, file_hasher, &[entry.sector.tag()])?;
    write_hashed(
        writer,
        file_hasher,
        &(entry.coordinate as u64).to_le_bytes(),
    )?;
    write_hashed(
        writer,
        file_hasher,
        &entry.monomial.exterior_spinor_mask.to_le_bytes(),
    )?;
    for exponent in entry.monomial.momentum.exponents {
        write_hashed(writer, file_hasher, &exponent.to_le_bytes())?;
    }
    for value in [
        entry.coefficient.real.numer(),
        entry.coefficient.real.denom(),
        entry.coefficient.imaginary.numer(),
        entry.coefficient.imaginary.denom(),
    ] {
        write_hashed(writer, file_hasher, &value.to_le_bytes())?;
    }
    Ok(())
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
    for entry_index in 0..entry_count {
        let tag = take::<1>(&bytes, &mut cursor)?[0];
        if !matches!(tag, 0 | 1 | 5 | 8 | 9) {
            return Err(format!(
                "column shard contains non-invariant sector tag {tag}"
            ));
        }
        semantic.update([tag]);
        let coordinate = u64::from_le_bytes(take(&bytes, &mut cursor)?);
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
    Ok(GaugeFixedSuperfieldColumnDigest {
        ordinal,
        source_coordinate: source_coordinate.to_string(),
        nonzero_terms: entry_count,
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
    visit_gauge_fixed_invariant_supercurvature(&input, |entry| {
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
    visit_gauge_fixed_invariant_supercurvature(&input, |entry| {
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
    let worker_count = std::env::var("ADYNKRA_COMPLETE_F_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_workers)
        .clamp(1, 16);
    let next_ordinal = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel();
    let mut completed = (0..321).map(|_| None).collect::<Vec<_>>();
    std::thread::scope(|scope| -> Result<(), String> {
        for _ in 0..worker_count {
            let sender = sender.clone();
            let next_ordinal = &next_ordinal;
            let shard_directory = shard_directory.map(Path::to_path_buf);
            scope.spawn(move || {
                loop {
                    let ordinal = next_ordinal.fetch_add(1, Ordering::Relaxed);
                    if ordinal >= 321 {
                        break;
                    }
                    let result = if let Some(directory) = &shard_directory {
                        write_or_validate_gauge_fixed_superfield_column_shard(ordinal, directory)
                    } else {
                        gauge_fixed_superfield_column_digest(ordinal)
                    };
                    if sender.send((ordinal, result)).is_err() {
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
        hash_bytes_with_length(&mut hasher, column.source_coordinate.as_bytes());
        hash_bytes_with_length(&mut hasher, column.sha256.as_bytes());
    }
    Ok(GaugeFixedSuperfieldOperatorCertificate {
        schema_version: SUPERFIELD_OPERATOR_SCHEMA,
        source_basis: "320 Cartesian gamma-traceless spatial-frame basis vectors followed by scale",
        source_dimension: 321,
        gamma_traceless_h_dimension: 320,
        scale_dimension: 1,
        output_basis: "linearized super-Weyl-covariant X_[2], X_[5], J^(-), raw W_2021, and direct off-shell LinearizedRiemann with canonical exterior-D and eleven-momentum monomials",
        unified_output_schema: SUPERFIELD_UNIFIED_OUTPUT_SCHEMA,
        column_shard_schema: SUPERFIELD_COLUMN_SHARD_SCHEMA,
        column_shard_directory: shard_directory
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        columns,
        total_nonzero_terms,
        operator_sha256: format!("{:x}", hasher.finalize()),
        direct_riemann_integrated: true,
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
    let shard_directory = PathBuf::from(format!("{}.columns-v2", path.display()));
    let certificate = build_gauge_fixed_superfield_operator_certificate_internal(
        Some(&shard_directory),
        progress,
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    atomic_json(path, &certificate)?;
    Ok(certificate)
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
    pub target_four_form_lowest_component_adapter_implemented: bool,
    pub direct_off_shell_frame_to_riemann_adapter_implemented: bool,
    pub direct_riemann_integrated_into_321_column_operator: bool,
    pub direct_riemann_bianchi_certified: bool,
    pub theta_two_w_gravity_double_emitted: bool,
    pub target_curvature_adapter_implemented: bool,
    pub target_bianchi_euler_noether_composition_certified: bool,
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
                sector: "W",
                domain: "T and D J",
                codomain: "linearized four-form Weyl curvature",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eq. (44); arXiv:2007.05097 Eqs. (2.22)-(2.23)",
            },
            PhysicalFSectorStatus {
                sector: "physical target curvature complex",
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
        target_four_form_lowest_component_adapter_implemented: true,
        direct_off_shell_frame_to_riemann_adapter_implemented: true,
        direct_riemann_integrated_into_321_column_operator: true,
        direct_riemann_bianchi_certified: true,
        theta_two_w_gravity_double_emitted: false,
        target_curvature_adapter_implemented: false,
        target_bianchi_euler_noether_composition_certified: false,
        complete_physical_f_implemented: false,
        complete_f_operator_sha256: None,
        exact_polynomial_target_kernel_derived: false,
        pointwise_or_bounded_kernel_is_accepted_as_physical_k: false,
        next_executable_step: "integrate the conditional first-descendant gravitino-curl adapter, add the remaining target identities and physical target gauge quotient, then derive the exact physical K kernel",
        passed,
        result: "The exact ordered-superderivative frame composes H_hat and scale through Delta, both Eqs. (13)-(14) anholonomies, D C, both Lorentz connections, J, mixed torsion, and both W conventions. P_320 and the p=2 gauge choice are enforced at the typed input boundary. The versioned 321-column invariant output now merges X_[2], X_[5], J^(-), raw W_2021, and the independent direct off-shell Riemann stream in strict exterior-D/momentum/sector/coordinate order. Raw two-D W terms retain their W tag and are never double-emitted as gravity. The direct Riemann branch satisfies the target differential Bianchi identity exactly. Full physical F remains incomplete until the conditional gravitino adapter and remaining target identities are integrated.",
        boundary: "Passing this report certifies the exact gauge-fixed H_hat-to-X/J/T/W differential stream, the direct full off-shell Riemann branch, and their versioned unified 321-column output contract. The raw p=2 lift has a nonzero J/T/W response in the current convention, so this is explicitly a canonical local-Lorentz gauge section rather than a proof that raw coordinates descend unchanged. D^2 W is retained only as a future conditional Weyl cross-check and is not used to define or emit Riemann. The v2 shard schema is consumed only by the fail-closed reader that binds every shard byte, footer, coefficient, row key, and digest before rank extraction. Physical K remains the future exact polynomial kernel after the conditional gravitino adapter, remaining target identities, and physical target gauge quotient are integrated; neither the combined diagnostic rank nor the Riemann-only rank is labeled as physical K.",
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
        assert!(report.target_four_form_lowest_component_adapter_implemented);
        assert!(report.direct_off_shell_frame_to_riemann_adapter_implemented);
        assert!(report.direct_riemann_integrated_into_321_column_operator);
        assert!(report.direct_riemann_bianchi_certified);
        assert!(!report.theta_two_w_gravity_double_emitted);
        assert!(!report.target_curvature_adapter_implemented);
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
    fn unified_v2_shard_schema_preserves_exterior_masks_and_rejects_v1() {
        assert_eq!(
            SUPERFIELD_OPERATOR_SCHEMA,
            "adynkra-11d-gauge-fixed-invariant-supercurvature-operator-v3"
        );
        assert_eq!(SUPERFIELD_COLUMN_SHARD_MAGIC, b"AD11FINVCOL2\0\0\0\0");
        assert_eq!(SUPERFIELD_COLUMN_ENTRY_BYTES, 67);

        let source = "schema_probe";
        let ordinal = 7_usize;
        let exterior = (1_u32 << 3) | (1_u32 << 19);
        let entry = GaugeFixedInvariantOutputEntry {
            sector: GaugeFixedInvariantOutputSector::W2021Raw,
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
        assert_ne!(
            entry.sector,
            GaugeFixedInvariantOutputSector::LinearizedRiemann
        );

        let mut bytes = Vec::new();
        let mut file_hasher = Sha256::new();
        encode_invariant_entry(&mut bytes, &mut file_hasher, &entry).unwrap();
        assert_eq!(bytes.len(), SUPERFIELD_COLUMN_ENTRY_BYTES);
        assert_eq!(bytes[0], GaugeFixedInvariantOutputSector::W2021Raw.tag());
        assert_eq!(
            u32::from_le_bytes(bytes[9..13].try_into().unwrap()),
            exterior
        );

        let directory = std::env::temp_dir().join(format!(
            "adynkra-unified-v2-schema-test-{}",
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

        shard[..16].copy_from_slice(b"AD11FINVCOL1\0\0\0\0");
        fs::write(&path, &shard).unwrap();
        let error = validate_column_shard(&path, ordinal, source).unwrap_err();
        assert!(error.contains("invalid column-shard magic"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn unified_output_order_puts_direct_riemann_after_raw_w_at_one_monomial() {
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
            monomial,
            coefficient: ExactQi::one(),
        };
        assert!(invariant_entry_cmp(&raw_w, &riemann).is_lt());
        assert_eq!(raw_w.sector.tag(), 8);
        assert_eq!(riemann.sector.tag(), 9);
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
        assert_eq!(
            entries.len(),
            stats.raw_invariant_terms + stats.direct_riemann_terms
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
    fn unified_scale_digest_matches_v2_shard_and_resume_validation() {
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
