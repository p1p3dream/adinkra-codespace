//! Bounded exact jet stream for the linearized 11D constrained-frame fields.
//!
//! The source frame retains `H_alpha{}^c`, the scale compensator `Psi`, and
//! the pure-gauge Lorentz two-form `Psi_[2]` after the algebraic conventional
//! constraints eliminate the p=1,3,4,5 holonomy forms.  This module applies
//! the certified ordered-superderivative algebra to those remaining fields
//! and emits physical-curvature coordinate ordinals without materializing one
//! enormous dense jet.
//!
//! It is an input boundary for the next composition step.  The emitted jets
//! are not yet the source-normalized Eq. (26)/(28) anholonomies, and no claim
//! of complete physical `F` or target-kernel-derived `K` is made here.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::eleven_dimensional_physical_curvature::{ExactQi, SPINOR_DIMENSION, VECTOR_DIMENSION};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, left_multiply_d,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-linearized-frame-jet-v1";
pub const H_COMPONENT_DIMENSION: usize = SPINOR_DIMENSION * VECTOR_DIMENSION;
pub const LORENTZ_TWO_FORM_DIMENSION: usize = VECTOR_DIMENSION * (VECTOR_DIMENSION - 1) / 2;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LinearizedFrameSuperfields {
    /// `H_alpha{}^c`, ordered by spinor then Lorentz vector.
    pub h: BTreeMap<usize, CanonicalSuperPolynomial>,
    /// The scalar scale compensator `Psi`.
    pub scale: CanonicalSuperPolynomial,
    /// Independent increasing Lorentz pairs in the repository mask order.
    pub lorentz_two_form: BTreeMap<usize, CanonicalSuperPolynomial>,
}

/// Canonical representative of the gamma-traceless `H_hat=P_320 H` target.
/// The local gamma-trace and Lorentz-two-form directions are removed before
/// any differential jet is formed, so downstream operators act on the
/// declared physical frame domain rather than on a gauge-dependent lift.
pub fn canonical_physical_frame_representative(
    input: &LinearizedFrameSuperfields,
) -> Result<LinearizedFrameSuperfields, String> {
    if let Some(component) = input
        .h
        .keys()
        .find(|component| **component >= H_COMPONENT_DIMENSION)
    {
        return Err(format!(
            "H component {component} is outside dimension {H_COMPONENT_DIMENSION}"
        ));
    }
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let mut trace = vec![CanonicalSuperPolynomial::default(); SPINOR_DIMENSION];
    for (&component, polynomial) in &input.h {
        let (spinor, vector) = h_indices(component);
        let metric = if vector == 0 { -1 } else { 1 };
        for (row, trace_row) in trace.iter_mut().enumerate() {
            let integer = i64::from(gammas[vector][row][spinor]) * metric;
            if integer != 0 {
                trace_row.add_assign(&polynomial.scaled(&ExactQi::from_integer(integer)));
            }
        }
    }

    let mut h = input.h.clone();
    for vector in 0..VECTOR_DIMENSION {
        for row in 0..SPINOR_DIMENSION {
            let mut correction = CanonicalSuperPolynomial::default();
            for (column, trace_column) in trace.iter().enumerate() {
                let integer = gammas[vector][row][column];
                if integer != 0 {
                    correction.add_assign(
                        &trace_column.scaled(&ExactQi::from_rational(i64::from(integer), 11)),
                    );
                }
            }
            let coordinate = row * VECTOR_DIMENSION + vector;
            let entry = h.entry(coordinate).or_default();
            entry.add_assign(&correction.scaled(&ExactQi::from_integer(-1)));
            if entry.terms.is_empty() {
                h.remove(&coordinate);
            }
        }
    }
    Ok(LinearizedFrameSuperfields {
        h,
        scale: input.scale.clone(),
        lorentz_two_form: BTreeMap::new(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub enum LinearizedFrameJetSector {
    DH,
    DDH,
    PH,
    DScale,
    DDScale,
    PScale,
    DLorentzTwoForm,
    PLorentzTwoForm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearizedFrameJetEntry {
    pub sector: LinearizedFrameJetSector,
    /// Coordinate in the corresponding physical-curvature tensor ordering.
    pub coordinate: usize,
    pub polynomial: CanonicalSuperPolynomial,
}

fn h_indices(component: usize) -> (usize, usize) {
    (component / VECTOR_DIMENSION, component % VECTOR_DIMENSION)
}

fn dh_index(derivative: usize, h_spinor: usize, vector: usize) -> usize {
    (derivative * SPINOR_DIMENSION + h_spinor) * VECTOR_DIMENSION + vector
}

fn ddh_index(outer: usize, inner: usize, h_spinor: usize, vector: usize) -> usize {
    ((outer * SPINOR_DIMENSION + inner) * SPINOR_DIMENSION + h_spinor) * VECTOR_DIMENSION + vector
}

fn ph_index(momentum: usize, h_spinor: usize, vector: usize) -> usize {
    (momentum * SPINOR_DIMENSION + h_spinor) * VECTOR_DIMENSION + vector
}

fn emit_scalar_jet<F>(polynomial: &CanonicalSuperPolynomial, mut emit: F) -> Result<(), String>
where
    F: FnMut(LinearizedFrameJetEntry) -> Result<(), String>,
{
    if polynomial.terms.is_empty() {
        return Ok(());
    }
    let mut first = Vec::with_capacity(SPINOR_DIMENSION);
    for derivative in 0..SPINOR_DIMENSION {
        let d = left_multiply_d(derivative, polynomial)?;
        emit(LinearizedFrameJetEntry {
            sector: LinearizedFrameJetSector::DScale,
            coordinate: derivative,
            polynomial: d.clone(),
        })?;
        first.push(d);
    }
    for outer in 0..SPINOR_DIMENSION {
        for (inner, d_inner) in first.iter().enumerate() {
            emit(LinearizedFrameJetEntry {
                sector: LinearizedFrameJetSector::DDScale,
                coordinate: outer * SPINOR_DIMENSION + inner,
                polynomial: left_multiply_d(outer, d_inner)?,
            })?;
        }
    }
    for momentum in 0..VECTOR_DIMENSION {
        emit(LinearizedFrameJetEntry {
            sector: LinearizedFrameJetSector::PScale,
            coordinate: momentum,
            polynomial: polynomial.multiply_momentum(momentum)?,
        })?;
    }
    Ok(())
}

/// Stream the complete ordered first/second spinor and first momentum jets.
///
/// The callback gives bounded backpressure.  For each nonzero stored H
/// component, at most one component's `32 + 1024 + 11` derived polynomials
/// are live.  The scale jet has the same fixed shape.  Lorentz p=2 is emitted
/// through first spinor and momentum order because those are the terms needed
/// to test local-Lorentz descent through the linearized frame.
pub fn visit_linearized_frame_jet<F>(
    input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<(), String>
where
    F: FnMut(LinearizedFrameJetEntry) -> Result<(), String>,
{
    if let Some(component) = input
        .h
        .keys()
        .find(|component| **component >= H_COMPONENT_DIMENSION)
    {
        return Err(format!(
            "H component {component} is outside dimension {H_COMPONENT_DIMENSION}"
        ));
    }
    if let Some(component) = input
        .lorentz_two_form
        .keys()
        .find(|component| **component >= LORENTZ_TWO_FORM_DIMENSION)
    {
        return Err(format!(
            "Lorentz two-form component {component} is outside dimension {LORENTZ_TWO_FORM_DIMENSION}"
        ));
    }
    if let Some(component) = input
        .h
        .iter()
        .find_map(|(component, polynomial)| polynomial.terms.is_empty().then_some(component))
    {
        return Err(format!(
            "H component {component} has a zero polynomial and must be omitted"
        ));
    }
    if let Some(component) = input
        .lorentz_two_form
        .iter()
        .find_map(|(component, polynomial)| polynomial.terms.is_empty().then_some(component))
    {
        return Err(format!(
            "Lorentz two-form component {component} has a zero polynomial and must be omitted"
        ));
    }

    for (&component, polynomial) in &input.h {
        let (h_spinor, vector) = h_indices(component);
        let mut first = Vec::with_capacity(SPINOR_DIMENSION);
        for derivative in 0..SPINOR_DIMENSION {
            let d = left_multiply_d(derivative, polynomial)?;
            emit(LinearizedFrameJetEntry {
                sector: LinearizedFrameJetSector::DH,
                coordinate: dh_index(derivative, h_spinor, vector),
                polynomial: d.clone(),
            })?;
            first.push(d);
        }
        for outer in 0..SPINOR_DIMENSION {
            for (inner, d_inner) in first.iter().enumerate() {
                emit(LinearizedFrameJetEntry {
                    sector: LinearizedFrameJetSector::DDH,
                    coordinate: ddh_index(outer, inner, h_spinor, vector),
                    polynomial: left_multiply_d(outer, d_inner)?,
                })?;
            }
        }
        for momentum in 0..VECTOR_DIMENSION {
            emit(LinearizedFrameJetEntry {
                sector: LinearizedFrameJetSector::PH,
                coordinate: ph_index(momentum, h_spinor, vector),
                polynomial: polynomial.multiply_momentum(momentum)?,
            })?;
        }
    }

    emit_scalar_jet(&input.scale, &mut emit)?;

    for (&pair, polynomial) in &input.lorentz_two_form {
        for derivative in 0..SPINOR_DIMENSION {
            emit(LinearizedFrameJetEntry {
                sector: LinearizedFrameJetSector::DLorentzTwoForm,
                coordinate: derivative * LORENTZ_TWO_FORM_DIMENSION + pair,
                polynomial: left_multiply_d(derivative, polynomial)?,
            })?;
        }
        for momentum in 0..VECTOR_DIMENSION {
            emit(LinearizedFrameJetEntry {
                sector: LinearizedFrameJetSector::PLorentzTwoForm,
                coordinate: momentum * LORENTZ_TWO_FORM_DIMENSION + pair,
                polynomial: polynomial.multiply_momentum(momentum)?,
            })?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LinearizedFrameJetReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source: &'static str,
    pub source_sha256: &'static str,
    pub h_component_dimension: usize,
    pub scale_component_dimension: usize,
    pub lorentz_two_form_dimension: usize,
    pub probe_h_components: usize,
    pub probe_lorentz_components: usize,
    pub probe_d_h_entries: usize,
    pub probe_dd_h_entries: usize,
    pub probe_p_h_entries: usize,
    pub probe_d_scale_entries: usize,
    pub probe_dd_scale_entries: usize,
    pub probe_p_scale_entries: usize,
    pub probe_d_lorentz_entries: usize,
    pub probe_p_lorentz_entries: usize,
    pub callback_bounded_streaming: bool,
    pub physical_coordinate_orderings_matched: bool,
    pub ordered_superderivative_normal_form_used: bool,
    pub conventional_compensators_composed: bool,
    pub geometry_anholonomies_composed: bool,
    pub complete_physical_f_implemented: bool,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

pub fn verify() -> LinearizedFrameJetReport {
    let mut input = LinearizedFrameSuperfields {
        scale: CanonicalSuperPolynomial::scalar(ExactQi::from_integer(2)),
        ..LinearizedFrameSuperfields::default()
    };
    input.h.insert(
        7 * VECTOR_DIMENSION + 3,
        CanonicalSuperPolynomial::scalar(ExactQi::one()),
    );
    input.lorentz_two_form.insert(
        11,
        CanonicalSuperPolynomial::scalar(ExactQi::from_integer(3)),
    );

    let mut counts = BTreeMap::<LinearizedFrameJetSector, usize>::new();
    let mut coordinate_residuals = 0_usize;
    visit_linearized_frame_jet(&input, |entry| {
        *counts.entry(entry.sector).or_default() += 1;
        coordinate_residuals += match entry.sector {
            LinearizedFrameJetSector::DH => usize::from(
                entry.coordinate >= SPINOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION,
            ),
            LinearizedFrameJetSector::DDH => usize::from(
                entry.coordinate
                    >= SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION,
            ),
            LinearizedFrameJetSector::PH => usize::from(
                entry.coordinate >= VECTOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION,
            ),
            LinearizedFrameJetSector::DScale => usize::from(entry.coordinate >= SPINOR_DIMENSION),
            LinearizedFrameJetSector::DDScale => {
                usize::from(entry.coordinate >= SPINOR_DIMENSION * SPINOR_DIMENSION)
            }
            LinearizedFrameJetSector::PScale => usize::from(entry.coordinate >= VECTOR_DIMENSION),
            LinearizedFrameJetSector::DLorentzTwoForm => {
                usize::from(entry.coordinate >= SPINOR_DIMENSION * LORENTZ_TWO_FORM_DIMENSION)
            }
            LinearizedFrameJetSector::PLorentzTwoForm => {
                usize::from(entry.coordinate >= VECTOR_DIMENSION * LORENTZ_TWO_FORM_DIMENSION)
            }
        };
        Ok(())
    })
    .expect("deterministic frame jet is valid");

    let count = |sector| counts.get(&sector).copied().unwrap_or(0);
    let passed = coordinate_residuals == 0
        && count(LinearizedFrameJetSector::DH) == SPINOR_DIMENSION
        && count(LinearizedFrameJetSector::DDH) == SPINOR_DIMENSION * SPINOR_DIMENSION
        && count(LinearizedFrameJetSector::PH) == VECTOR_DIMENSION
        && count(LinearizedFrameJetSector::DScale) == SPINOR_DIMENSION
        && count(LinearizedFrameJetSector::DDScale) == SPINOR_DIMENSION * SPINOR_DIMENSION
        && count(LinearizedFrameJetSector::PScale) == VECTOR_DIMENSION
        && count(LinearizedFrameJetSector::DLorentzTwoForm) == SPINOR_DIMENSION
        && count(LinearizedFrameJetSector::PLorentzTwoForm) == VECTOR_DIMENSION;

    LinearizedFrameJetReport {
        schema_version: SCHEMA_VERSION,
        role: "bounded exact polynomial jet stream for the remaining linearized constrained-frame superfields",
        source: "Gates and Nishino, hep-th/0101037 Eqs. (1), (25)-(29), (39)-(40), and Table 3",
        source_sha256: crate::eleven_dimensional_physical_curvature::HEP_TH_0101037_SOURCE_SHA256,
        h_component_dimension: H_COMPONENT_DIMENSION,
        scale_component_dimension: 1,
        lorentz_two_form_dimension: LORENTZ_TWO_FORM_DIMENSION,
        probe_h_components: input.h.len(),
        probe_lorentz_components: input.lorentz_two_form.len(),
        probe_d_h_entries: count(LinearizedFrameJetSector::DH),
        probe_dd_h_entries: count(LinearizedFrameJetSector::DDH),
        probe_p_h_entries: count(LinearizedFrameJetSector::PH),
        probe_d_scale_entries: count(LinearizedFrameJetSector::DScale),
        probe_dd_scale_entries: count(LinearizedFrameJetSector::DDScale),
        probe_p_scale_entries: count(LinearizedFrameJetSector::PScale),
        probe_d_lorentz_entries: count(LinearizedFrameJetSector::DLorentzTwoForm),
        probe_p_lorentz_entries: count(LinearizedFrameJetSector::PLorentzTwoForm),
        callback_bounded_streaming: true,
        physical_coordinate_orderings_matched: coordinate_residuals == 0,
        ordered_superderivative_normal_form_used: true,
        conventional_compensators_composed: false,
        geometry_anholonomies_composed: false,
        complete_physical_f_implemented: false,
        passed,
        result: "The H, scale, and Lorentz p=2 fields now stream exact ordered spinor and formal-momentum jets directly in the physical-curvature coordinate orderings.",
        boundary: "This is the bounded algebraic jet boundary, not yet the differentiated Eq. (40) compensator solve or the Eq. (26)/(28) geometry composition. The next gate must eliminate p=1,3,4,5 from H, retain p=2 only as a tested gauge orbit, assemble D C and mixed anholonomy, and prove invariant descent before complete F or K can be claimed.",
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

pub fn write_artifact(path: &Path) -> io::Result<LinearizedFrameJetReport> {
    let report = verify();
    atomic_json(path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_frame_representative_removes_gamma_trace_and_lorentz_orbit() {
        let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
        let mut input = LinearizedFrameSuperfields::default();
        for vector in 0..VECTOR_DIMENSION {
            for row in 0..SPINOR_DIMENSION {
                let integer = gammas[vector][row][0];
                if integer != 0 {
                    input.h.insert(
                        row * VECTOR_DIMENSION + vector,
                        CanonicalSuperPolynomial::scalar(ExactQi::from_integer(i64::from(integer))),
                    );
                }
            }
        }
        input
            .lorentz_two_form
            .insert(0, CanonicalSuperPolynomial::scalar(ExactQi::one()));
        let representative = canonical_physical_frame_representative(&input).unwrap();
        assert!(representative.h.is_empty());
        assert!(representative.lorentz_two_form.is_empty());
    }
    use crate::eleven_dimensional_superderivative_normal_form::translation_action;

    #[test]
    fn deterministic_stream_has_the_complete_bounded_jet_shape() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.probe_d_h_entries, 32);
        assert_eq!(report.probe_dd_h_entries, 1_024);
        assert_eq!(report.probe_p_h_entries, 11);
        assert!(report.callback_bounded_streaming);
        assert!(report.physical_coordinate_orderings_matched);
    }

    #[test]
    fn streamed_second_h_jet_obeys_every_anticommutator() {
        let component = 5 * VECTOR_DIMENSION + 9;
        let base = CanonicalSuperPolynomial::scalar(ExactQi::one());
        let mut input = LinearizedFrameSuperfields::default();
        input.h.insert(component, base.clone());
        let mut dd = BTreeMap::new();
        visit_linearized_frame_jet(&input, |entry| {
            if entry.sector == LinearizedFrameJetSector::DDH {
                let vector = entry.coordinate % VECTOR_DIMENSION;
                let rest = entry.coordinate / VECTOR_DIMENSION;
                let h_spinor = rest % SPINOR_DIMENSION;
                let rest = rest / SPINOR_DIMENSION;
                let inner = rest % SPINOR_DIMENSION;
                let outer = rest / SPINOR_DIMENSION;
                if h_spinor == 5 && vector == 9 {
                    dd.insert((outer, inner), entry.polynomial);
                }
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(dd.len(), SPINOR_DIMENSION * SPINOR_DIMENSION);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                let mut actual = dd[&(alpha, beta)].clone();
                actual.add_assign(&dd[&(beta, alpha)]);
                assert_eq!(
                    actual,
                    translation_action(alpha, beta, &base).unwrap(),
                    "alpha={alpha}, beta={beta}"
                );
            }
        }
    }

    #[test]
    fn malformed_components_and_callback_failure_stop_immediately() {
        let mut empty_calls = 0;
        visit_linearized_frame_jet(&LinearizedFrameSuperfields::default(), |_| {
            empty_calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(empty_calls, 0);

        let mut invalid = LinearizedFrameSuperfields::default();
        invalid.h.insert(
            H_COMPONENT_DIMENSION,
            CanonicalSuperPolynomial::scalar(ExactQi::one()),
        );
        assert!(visit_linearized_frame_jet(&invalid, |_| Ok(())).is_err());

        let mut noncanonical = LinearizedFrameSuperfields::default();
        noncanonical
            .h
            .insert(0, CanonicalSuperPolynomial::default());
        assert!(visit_linearized_frame_jet(&noncanonical, |_| Ok(())).is_err());

        let mut valid = LinearizedFrameSuperfields::default();
        valid
            .h
            .insert(0, CanonicalSuperPolynomial::scalar(ExactQi::one()));
        let mut calls = 0;
        let error = visit_linearized_frame_jet(&valid, |_| {
            calls += 1;
            Err("stop".to_string())
        })
        .unwrap_err();
        assert_eq!(error, "stop");
        assert_eq!(calls, 1);
    }
}
