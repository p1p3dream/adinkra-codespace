//! Complete exact anholonomy jet derived from one constrained frame jet.
//!
//! This is the bridge that was previously missing between the Eq. (40)
//! compensator solve and the geometry-level first-superspace-jet machinery.
//! It constructs `Delta`, checks its independently streamed derivative,
//! assembles the three linearized anholonomies in hep-th/0101037 Eqs. (13)
//! and (14), and differentiates the two connection-source anholonomies in the
//! certified ordered-superderivative algebra.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::eleven_dimensional_eq40_frame_composition::{
    Eq40FrameSector, visit_eq40_frame_composition,
};
use crate::eleven_dimensional_h_hat_jet::{
    LinearizedFrameJetSector, LinearizedFrameSuperfields, visit_linearized_frame_jet,
};
use crate::eleven_dimensional_physical_curvature::{
    self as physical, C_ALPHA_VECTOR_VECTOR_DIMENSION, D_DELTA_DIMENSION, DDH_DIMENSION,
    DELTA_DIMENSION, Eq14MixedSpinorAnholonomyInput, ExactQi, SPINOR_ANHOLONOMY_DIMENSION,
    SPINOR_DIMENSION, VECTOR_DIMENSION,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial, left_multiply_d,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-constrained-geometry-jet-v1";

type PolynomialMap = BTreeMap<usize, CanonicalSuperPolynomial>;
type MonomialSlices = BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub enum ConstrainedGeometryJetSector {
    Delta,
    DDelta,
    CAlphaBetaGamma,
    CAlphaVectorVector,
    CAlphaVectorGamma,
    DCAlphaBetaGamma,
    DCAlphaVectorVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedGeometryJetEntry {
    pub sector: ConstrainedGeometryJetSector,
    pub coordinate: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConstrainedGeometryJetStats {
    pub delta_derivative_identity_residual_terms: usize,
    pub emitted_by_sector: BTreeMap<ConstrainedGeometryJetSector, usize>,
}

fn add_coefficient(
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

fn add_polynomial(
    output: &mut PolynomialMap,
    coordinate: usize,
    polynomial: &CanonicalSuperPolynomial,
) {
    for (monomial, coefficient) in &polynomial.terms {
        add_coefficient(output, coordinate, monomial.clone(), coefficient.clone());
    }
}

fn merge_polynomial_maps(output: &mut PolynomialMap, input: &PolynomialMap) {
    for (&coordinate, polynomial) in input {
        add_polynomial(output, coordinate, polynomial);
    }
}

fn transpose(input: &PolynomialMap) -> MonomialSlices {
    let mut slices = MonomialSlices::new();
    for (&coordinate, polynomial) in input {
        for (monomial, coefficient) in &polynomial.terms {
            let slice = slices.entry(monomial.clone()).or_default();
            let entry = slice.entry(coordinate).or_insert_with(ExactQi::zero);
            entry.add_assign(coefficient);
            if entry.is_zero() {
                slice.remove(&coordinate);
            }
        }
    }
    slices.retain(|_, slice| !slice.is_empty());
    slices
}

fn from_sliced<F>(input: &PolynomialMap, mut apply: F) -> PolynomialMap
where
    F: FnMut(&BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi>,
{
    let mut output = PolynomialMap::new();
    for (monomial, slice) in transpose(input) {
        for (coordinate, coefficient) in apply(&slice) {
            add_coefficient(&mut output, coordinate, monomial.clone(), coefficient);
        }
    }
    output
}

fn add_sliced_binary<F>(left: &PolynomialMap, right: &PolynomialMap, mut apply: F) -> PolynomialMap
where
    F: FnMut(&BTreeMap<usize, ExactQi>, &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi>,
{
    let left = transpose(left);
    let right = transpose(right);
    let keys = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let empty = BTreeMap::new();
    let mut output = PolynomialMap::new();
    for monomial in keys {
        let left_slice = left.get(&monomial).unwrap_or(&empty);
        let right_slice = right.get(&monomial).unwrap_or(&empty);
        for (coordinate, coefficient) in apply(left_slice, right_slice) {
            add_coefficient(&mut output, coordinate, monomial.clone(), coefficient);
        }
    }
    output
}

fn spinor_derivative(
    input: &PolynomialMap,
    source_dimension: usize,
) -> Result<PolynomialMap, String> {
    let mut output = PolynomialMap::new();
    for (&coordinate, polynomial) in input {
        if coordinate >= source_dimension {
            return Err(format!(
                "source coordinate {coordinate} is outside dimension {source_dimension}"
            ));
        }
        for derivative in 0..SPINOR_DIMENSION {
            let derived = left_multiply_d(derivative, polynomial)?;
            add_polynomial(
                &mut output,
                derivative * source_dimension + coordinate,
                &derived,
            );
        }
    }
    Ok(output)
}

fn momentum_derivative(
    input: &PolynomialMap,
    source_dimension: usize,
) -> Result<PolynomialMap, String> {
    let mut output = PolynomialMap::new();
    for (&coordinate, polynomial) in input {
        if coordinate >= source_dimension {
            return Err(format!(
                "source coordinate {coordinate} is outside dimension {source_dimension}"
            ));
        }
        for momentum in 0..VECTOR_DIMENSION {
            let derived = polynomial.multiply_momentum(momentum)?;
            add_polynomial(
                &mut output,
                momentum * source_dimension + coordinate,
                &derived,
            );
        }
    }
    Ok(output)
}

fn form_fields(
    input: &LinearizedFrameSuperfields,
) -> Result<
    (
        BTreeMap<Eq40FrameSector, PolynomialMap>,
        BTreeMap<LinearizedFrameJetSector, PolynomialMap>,
    ),
    String,
> {
    let mut eq40 = BTreeMap::<Eq40FrameSector, PolynomialMap>::new();
    visit_eq40_frame_composition(input, |entry| {
        add_coefficient(
            eq40.entry(entry.sector).or_default(),
            entry.coordinate,
            entry.monomial,
            entry.coefficient,
        );
        Ok(())
    })?;
    let mut frame = BTreeMap::<LinearizedFrameJetSector, PolynomialMap>::new();
    visit_linearized_frame_jet(input, |entry| {
        frame
            .entry(entry.sector)
            .or_default()
            .insert(entry.coordinate, entry.polynomial);
        Ok(())
    })?;
    Ok((eq40, frame))
}

fn build_delta(
    eq40: &BTreeMap<Eq40FrameSector, PolynomialMap>,
    input: &LinearizedFrameSuperfields,
) -> PolynomialMap {
    let empty = PolynomialMap::new();
    let mut forms = vec![
        (
            1,
            eq40.get(&Eq40FrameSector::PsiOne).unwrap_or(&empty).clone(),
        ),
        (2, input.lorentz_two_form.clone()),
        (
            3,
            eq40.get(&Eq40FrameSector::PsiThree)
                .unwrap_or(&empty)
                .clone(),
        ),
        (
            4,
            eq40.get(&Eq40FrameSector::PsiFour)
                .unwrap_or(&empty)
                .clone(),
        ),
        (
            5,
            eq40.get(&Eq40FrameSector::PsiFive)
                .unwrap_or(&empty)
                .clone(),
        ),
    ];
    let mut delta = PolynomialMap::new();
    for (degree, form) in forms.drain(..) {
        let image = from_sliced(&form, |slice| {
            physical::inject_holonomy_form_into_delta(degree, slice)
        });
        merge_polynomial_maps(&mut delta, &image);
    }
    delta
}

fn build_d_delta(eq40: &BTreeMap<Eq40FrameSector, PolynomialMap>) -> PolynomialMap {
    let empty = PolynomialMap::new();
    let forms = [
        (1, Eq40FrameSector::DPsiOne),
        (2, Eq40FrameSector::DPsiTwoIndependent),
        (3, Eq40FrameSector::DPsiThree),
        (4, Eq40FrameSector::DPsiFour),
        (5, Eq40FrameSector::DPsiFive),
    ];
    let mut d_delta = PolynomialMap::new();
    for (degree, sector) in forms {
        let image = from_sliced(eq40.get(&sector).unwrap_or(&empty), |slice| {
            physical::inject_d_holonomy_form_into_d_delta(degree, slice)
        });
        merge_polynomial_maps(&mut d_delta, &image);
    }
    d_delta
}

fn add_scale_diagonal(d_delta: &PolynomialMap, d_scale: &PolynomialMap) -> PolynomialMap {
    let mut output = d_delta.clone();
    for (&derivative, polynomial) in d_scale {
        for delta in 0..SPINOR_DIMENSION {
            add_polynomial(
                &mut output,
                (derivative * SPINOR_DIMENSION + delta) * SPINOR_DIMENSION + delta,
                polynomial,
            );
        }
    }
    output
}

fn assemble_anholonomies(
    eq40: &BTreeMap<Eq40FrameSector, PolynomialMap>,
    frame: &BTreeMap<LinearizedFrameJetSector, PolynomialMap>,
    delta: &PolynomialMap,
    d_delta: &PolynomialMap,
) -> Result<(PolynomialMap, PolynomialMap, PolynomialMap), String> {
    let empty = PolynomialMap::new();
    let d_scale = frame
        .get(&LinearizedFrameJetSector::DScale)
        .unwrap_or(&empty);
    let eq26_input = add_scale_diagonal(d_delta, d_scale);
    let c_spinor = from_sliced(&eq26_input, |slice| {
        physical::eq26_spinor_anholonomy_operator().apply(slice)
    });

    let mut h_second = frame
        .get(&LinearizedFrameJetSector::DDH)
        .cloned()
        .unwrap_or_default();
    if let Some(p_h) = frame.get(&LinearizedFrameJetSector::PH) {
        for (&coordinate, polynomial) in p_h {
            h_second.insert(DDH_DIMENSION + coordinate, polynomial.clone());
        }
    }
    let eq28_h = physical::cached_eq28_h_to_c_alpha_b_c_operator();
    let mut c_vector = from_sliced(&h_second, |slice| eq28_h.apply_sparse(slice));
    merge_polynomial_maps(
        &mut c_vector,
        &from_sliced(d_scale, |slice| {
            physical::eq28_d_scalar_to_c_alpha_b_c_operator().apply_sparse(slice)
        }),
    );
    let d_psi_two = eq40
        .get(&Eq40FrameSector::DPsiTwoIndependent)
        .unwrap_or(&empty);
    merge_polynomial_maps(
        &mut c_vector,
        &add_sliced_binary(d_delta, d_psi_two, |delta_slice, psi_two_slice| {
            physical::apply_eq28_delta_sector_to_c_alpha_b_c(delta_slice, psi_two_slice)
        }),
    );

    let d_d_delta = spinor_derivative(d_delta, D_DELTA_DIMENSION)?;
    let p_delta = momentum_derivative(delta, DELTA_DIMENSION)?;
    let d_d_scale = frame
        .get(&LinearizedFrameJetSector::DDScale)
        .cloned()
        .unwrap_or_default();
    let p_scale = frame
        .get(&LinearizedFrameJetSector::PScale)
        .cloned()
        .unwrap_or_default();
    let d_d_delta = transpose(&d_d_delta);
    let p_delta = transpose(&p_delta);
    let d_d_scale = transpose(&d_d_scale);
    let p_scale = transpose(&p_scale);
    let keys = d_d_delta
        .keys()
        .chain(p_delta.keys())
        .chain(d_d_scale.keys())
        .chain(p_scale.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let empty_slice = BTreeMap::new();
    let mut c_mixed = PolynomialMap::new();
    for monomial in keys {
        let source = Eq14MixedSpinorAnholonomyInput {
            d_d_delta: d_d_delta.get(&monomial).unwrap_or(&empty_slice).clone(),
            p_delta: p_delta.get(&monomial).unwrap_or(&empty_slice).clone(),
            p_scale: p_scale.get(&monomial).unwrap_or(&empty_slice).clone(),
            d_d_scale: d_d_scale.get(&monomial).unwrap_or(&empty_slice).clone(),
        };
        for (coordinate, coefficient) in physical::apply_eq14_mixed_spinor_anholonomy(&source) {
            add_coefficient(&mut c_mixed, coordinate, monomial.clone(), coefficient);
        }
    }
    Ok((c_spinor, c_vector, c_mixed))
}

fn emit_map<F>(
    sector: ConstrainedGeometryJetSector,
    input: &PolynomialMap,
    stats: &mut ConstrainedGeometryJetStats,
    emit: &mut F,
) -> Result<(), String>
where
    F: FnMut(ConstrainedGeometryJetEntry) -> Result<(), String>,
{
    for (&coordinate, polynomial) in input {
        for (monomial, coefficient) in &polynomial.terms {
            emit(ConstrainedGeometryJetEntry {
                sector,
                coordinate,
                monomial: monomial.clone(),
                coefficient: coefficient.clone(),
            })?;
            *stats.emitted_by_sector.entry(sector).or_default() += 1;
        }
    }
    Ok(())
}

/// Stream only the exact Eq. (40) `D Delta` sector while retaining the
/// ordered-D derivative identity gate used by the complete geometry jet.
pub fn visit_constrained_d_delta<F>(
    input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<ConstrainedGeometryJetStats, String>
where
    F: FnMut(ConstrainedGeometryJetEntry) -> Result<(), String>,
{
    let (eq40, _) = form_fields(input)?;
    let delta = build_delta(&eq40, input);
    let d_delta = build_d_delta(&eq40);
    let derived_d_delta = spinor_derivative(&delta, DELTA_DIMENSION)?;
    let residual = d_delta
        .keys()
        .chain(derived_d_delta.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| d_delta.get(key) != derived_d_delta.get(key))
        .count();
    if residual != 0 {
        return Err(format!(
            "Eq. (40) differentiated Delta disagrees with ordered-D differentiation in {residual} coordinates"
        ));
    }
    let mut stats = ConstrainedGeometryJetStats {
        delta_derivative_identity_residual_terms: 0,
        emitted_by_sector: BTreeMap::new(),
    };
    emit_map(
        ConstrainedGeometryJetSector::DDelta,
        &d_delta,
        &mut stats,
        &mut emit,
    )?;
    Ok(stats)
}

/// Stream the complete linearized anholonomy jet derived from one H/scale/p=2
/// frame input.  The callback sees exact monomial coefficients and can apply
/// bounded backpressure.
pub fn visit_constrained_geometry_jet<F>(
    input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<ConstrainedGeometryJetStats, String>
where
    F: FnMut(ConstrainedGeometryJetEntry) -> Result<(), String>,
{
    let (eq40, frame) = form_fields(input)?;
    let delta = build_delta(&eq40, input);
    let d_delta = build_d_delta(&eq40);
    let derived_d_delta = spinor_derivative(&delta, DELTA_DIMENSION)?;
    let delta_derivative_identity_residual_terms = if d_delta == derived_d_delta {
        0
    } else {
        d_delta
            .keys()
            .chain(derived_d_delta.keys())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|key| d_delta.get(key) != derived_d_delta.get(key))
            .count()
    };
    if delta_derivative_identity_residual_terms != 0 {
        return Err(format!(
            "Eq. (40) differentiated Delta disagrees with ordered-D differentiation in {delta_derivative_identity_residual_terms} coordinates"
        ));
    }

    let (c_spinor, c_vector, c_mixed) = assemble_anholonomies(&eq40, &frame, &delta, &d_delta)?;
    let (d_c_spinor, d_c_vector) = rayon::join(
        || spinor_derivative(&c_spinor, SPINOR_ANHOLONOMY_DIMENSION),
        || spinor_derivative(&c_vector, C_ALPHA_VECTOR_VECTOR_DIMENSION),
    );
    let d_c_spinor = d_c_spinor?;
    let d_c_vector = d_c_vector?;

    let mut stats = ConstrainedGeometryJetStats {
        delta_derivative_identity_residual_terms,
        emitted_by_sector: BTreeMap::new(),
    };
    for (sector, map) in [
        (ConstrainedGeometryJetSector::Delta, &delta),
        (ConstrainedGeometryJetSector::DDelta, &d_delta),
        (ConstrainedGeometryJetSector::CAlphaBetaGamma, &c_spinor),
        (ConstrainedGeometryJetSector::CAlphaVectorVector, &c_vector),
        (ConstrainedGeometryJetSector::CAlphaVectorGamma, &c_mixed),
        (ConstrainedGeometryJetSector::DCAlphaBetaGamma, &d_c_spinor),
        (
            ConstrainedGeometryJetSector::DCAlphaVectorVector,
            &d_c_vector,
        ),
    ] {
        emit_map(sector, map, &mut stats, &mut emit)?;
    }
    Ok(stats)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConstrainedGeometryJetReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source: &'static str,
    pub source_sha256: &'static str,
    pub delta_terms: usize,
    pub d_delta_terms: usize,
    pub c_alpha_beta_gamma_terms: usize,
    pub c_alpha_vector_vector_terms: usize,
    pub c_alpha_vector_gamma_terms: usize,
    pub d_c_alpha_beta_gamma_terms: usize,
    pub d_c_alpha_vector_vector_terms: usize,
    pub delta_derivative_identity_residual_terms: usize,
    pub equation_13_composed: bool,
    pub equation_14_both_anholonomies_composed: bool,
    pub first_geometry_jet_composed: bool,
    pub p2_descent_proved: bool,
    pub complete_physical_f_implemented: bool,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

fn probe_input() -> LinearizedFrameSuperfields {
    let mut input = LinearizedFrameSuperfields {
        scale: CanonicalSuperPolynomial::scalar(ExactQi::from_integer(2)),
        ..LinearizedFrameSuperfields::default()
    };
    input.h.insert(
        3 * VECTOR_DIMENSION + 7,
        CanonicalSuperPolynomial::scalar(ExactQi::one()),
    );
    input.lorentz_two_form.insert(
        9,
        CanonicalSuperPolynomial::scalar(ExactQi::from_integer(3)),
    );
    input
}

pub fn verify() -> ConstrainedGeometryJetReport {
    let stats = visit_constrained_geometry_jet(&probe_input(), |_| Ok(()))
        .expect("deterministic constrained geometry probe is valid");
    let count = |sector| stats.emitted_by_sector.get(&sector).copied().unwrap_or(0);
    let equation_13_composed = count(ConstrainedGeometryJetSector::CAlphaBetaGamma) != 0;
    let equation_14_both_anholonomies_composed =
        count(ConstrainedGeometryJetSector::CAlphaVectorVector) != 0
            && count(ConstrainedGeometryJetSector::CAlphaVectorGamma) != 0;
    let first_geometry_jet_composed = count(ConstrainedGeometryJetSector::DCAlphaBetaGamma) != 0
        && count(ConstrainedGeometryJetSector::DCAlphaVectorVector) != 0;
    let passed = count(ConstrainedGeometryJetSector::Delta) != 0
        && count(ConstrainedGeometryJetSector::DDelta) != 0
        && stats.delta_derivative_identity_residual_terms == 0
        && equation_13_composed
        && equation_14_both_anholonomies_composed
        && first_geometry_jet_composed;
    ConstrainedGeometryJetReport {
        schema_version: SCHEMA_VERSION,
        role: "exact compensator-eliminated H-hat jet through the complete linearized anholonomy and first geometry jet",
        source: "Gates and Nishino, hep-th/0101037 Eqs. (1), (13), (14), (39), and (40)",
        source_sha256: physical::HEP_TH_0101037_SOURCE_SHA256,
        delta_terms: count(ConstrainedGeometryJetSector::Delta),
        d_delta_terms: count(ConstrainedGeometryJetSector::DDelta),
        c_alpha_beta_gamma_terms: count(ConstrainedGeometryJetSector::CAlphaBetaGamma),
        c_alpha_vector_vector_terms: count(ConstrainedGeometryJetSector::CAlphaVectorVector),
        c_alpha_vector_gamma_terms: count(ConstrainedGeometryJetSector::CAlphaVectorGamma),
        d_c_alpha_beta_gamma_terms: count(ConstrainedGeometryJetSector::DCAlphaBetaGamma),
        d_c_alpha_vector_vector_terms: count(ConstrainedGeometryJetSector::DCAlphaVectorVector),
        delta_derivative_identity_residual_terms: stats.delta_derivative_identity_residual_terms,
        equation_13_composed,
        equation_14_both_anholonomies_composed,
        first_geometry_jet_composed,
        p2_descent_proved: false,
        complete_physical_f_implemented: false,
        passed,
        result: "The same exact H/scale/p=2 frame jet now supplies Delta, D Delta, all three linearized anholonomies, and the D C inputs required by the first-superspace-jet curvature chain.",
        boundary: "This closes the missing H-hat-to-geometry-jet composition. The next gate must feed these outputs through J, torsion, and W and prove that the pure p=2 local-Lorentz orbit vanishes in the invariant curvature output before complete physical F or its exact target-side kernel K can be claimed.",
    }
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

pub fn write_artifact(path: &Path) -> io::Result<ConstrainedGeometryJetReport> {
    let report = verify();
    atomic_json(path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_frame_reaches_every_required_geometry_jet_sector() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.delta_derivative_identity_residual_terms, 0);
        assert!(report.equation_13_composed);
        assert!(report.equation_14_both_anholonomies_composed);
        assert!(report.first_geometry_jet_composed);
    }

    #[test]
    fn pure_scale_and_lorentz_input_remains_exact_and_callback_bounded() {
        let mut input = LinearizedFrameSuperfields {
            scale: CanonicalSuperPolynomial::scalar(ExactQi::one()),
            ..LinearizedFrameSuperfields::default()
        };
        input.lorentz_two_form.insert(
            0,
            CanonicalSuperPolynomial::scalar(ExactQi::from_integer(2)),
        );
        let stats = visit_constrained_geometry_jet(&input, |_| Ok(())).unwrap();
        assert_eq!(stats.delta_derivative_identity_residual_terms, 0);
        assert!(stats.emitted_by_sector[&ConstrainedGeometryJetSector::CAlphaVectorGamma] > 0);

        let mut calls = 0;
        let error = visit_constrained_geometry_jet(&input, |_| {
            calls += 1;
            Err("stop".to_string())
        })
        .unwrap_err();
        assert_eq!(error, "stop");
        assert_eq!(calls, 1);
    }

    #[test]
    fn emitted_coordinates_match_physical_curvature_dimensions() {
        let mut residuals = 0;
        visit_constrained_geometry_jet(&probe_input(), |entry| {
            let dimension = match entry.sector {
                ConstrainedGeometryJetSector::Delta => DELTA_DIMENSION,
                ConstrainedGeometryJetSector::DDelta => D_DELTA_DIMENSION,
                ConstrainedGeometryJetSector::CAlphaBetaGamma => SPINOR_ANHOLONOMY_DIMENSION,
                ConstrainedGeometryJetSector::CAlphaVectorVector => C_ALPHA_VECTOR_VECTOR_DIMENSION,
                ConstrainedGeometryJetSector::CAlphaVectorGamma => {
                    physical::T_ALPHA_VECTOR_SPINOR_DIMENSION
                }
                ConstrainedGeometryJetSector::DCAlphaBetaGamma => {
                    SPINOR_DIMENSION * SPINOR_ANHOLONOMY_DIMENSION
                }
                ConstrainedGeometryJetSector::DCAlphaVectorVector => {
                    SPINOR_DIMENSION * C_ALPHA_VECTOR_VECTOR_DIMENSION
                }
            };
            residuals += usize::from(entry.coordinate >= dimension);
            Ok(())
        })
        .unwrap();
        assert_eq!(residuals, 0);
    }

    #[test]
    fn narrow_d_delta_stream_matches_complete_geometry_jet_exactly() {
        let input = probe_input();
        let mut complete = Vec::new();
        visit_constrained_geometry_jet(&input, |entry| {
            if entry.sector == ConstrainedGeometryJetSector::DDelta {
                complete.push(entry);
            }
            Ok(())
        })
        .unwrap();

        let mut narrow = Vec::new();
        let stats = visit_constrained_d_delta(&input, |entry| {
            narrow.push(entry);
            Ok(())
        })
        .unwrap();

        assert_eq!(stats.delta_derivative_identity_residual_terms, 0);
        assert_eq!(
            stats.emitted_by_sector,
            BTreeMap::from([(ConstrainedGeometryJetSector::DDelta, narrow.len())])
        );
        assert_eq!(narrow, complete);
    }
}
