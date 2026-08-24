//! Exact polynomial composition of the constrained frame with Eq. (40).
//!
//! The algebraic compensator solve in `eleven_dimensional_physical_curvature`
//! acts on one numerical `D H` or `D D H` slice.  The frame-jet stream carries
//! coefficients in the certified ordered-superderivative normal form.  This
//! module transposes that sparse stream by normal-form monomial, applies the
//! exact solve independently to every coefficient slice, and emits the result
//! with callback backpressure.  The independent scale and Lorentz p=2 jets are
//! passed through in distinct sectors and can therefore not enter the
//! conventional p=1,3,4,5 elimination accidentally.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::eleven_dimensional_h_hat_jet::{
    LORENTZ_TWO_FORM_DIMENSION, LinearizedFrameJetSector, LinearizedFrameSuperfields,
    visit_linearized_frame_jet,
};
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, SPINOR_DIMENSION, VECTOR_DIMENSION, solve_conventional_compensators,
    solve_higher_jet_conventional_compensators,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-eq40-frame-composition-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub enum Eq40FrameSector {
    PsiOne,
    PsiThree,
    PsiFour,
    PsiFive,
    DPsiOne,
    DPsiThree,
    DPsiFour,
    DPsiFive,
    DScaleIndependent,
    DPsiTwoIndependent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Eq40FrameEntry {
    pub sector: Eq40FrameSector,
    /// Canonical form ordinal.  Differentiated sectors include the outer
    /// spinor in the existing `outer * C(11,p) + form` ordering.
    pub coordinate: usize,
    pub monomial: OrderedSuperderivativeMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Eq40FrameCompositionStats {
    pub d_h_monomial_slices: usize,
    pub d_d_h_monomial_slices: usize,
    pub emitted_by_sector: BTreeMap<Eq40FrameSector, usize>,
}

type MonomialSlices = BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>;

fn add_slice_term(
    slices: &mut MonomialSlices,
    monomial: OrderedSuperderivativeMonomial,
    coordinate: usize,
    coefficient: ExactQi,
) {
    if coefficient.is_zero() {
        return;
    }
    let slice = slices.entry(monomial.clone()).or_default();
    let entry = slice.entry(coordinate).or_insert_with(ExactQi::zero);
    entry.add_assign(&coefficient);
    if entry.is_zero() {
        slice.remove(&coordinate);
    }
    if slice.is_empty() {
        slices.remove(&monomial);
    }
}

fn collect_polynomial(
    slices: &mut MonomialSlices,
    coordinate: usize,
    polynomial: CanonicalSuperPolynomial,
) {
    for (monomial, coefficient) in polynomial.terms {
        add_slice_term(slices, monomial, coordinate, coefficient);
    }
}

fn masks_of_degree(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn form_ordinal(degree: usize, mask: u16) -> Result<usize, String> {
    masks_of_degree(degree)
        .binary_search(&mask)
        .map_err(|_| format!("degree-{degree} compensator mask {mask:#x} is not canonical"))
}

fn emit_one<F>(
    emit: &mut F,
    stats: &mut Eq40FrameCompositionStats,
    sector: Eq40FrameSector,
    coordinate: usize,
    monomial: &OrderedSuperderivativeMonomial,
    coefficient: ExactQi,
) -> Result<(), String>
where
    F: FnMut(Eq40FrameEntry) -> Result<(), String>,
{
    if coefficient.is_zero() {
        return Ok(());
    }
    emit(Eq40FrameEntry {
        sector,
        coordinate,
        monomial: monomial.clone(),
        coefficient,
    })?;
    *stats.emitted_by_sector.entry(sector).or_default() += 1;
    Ok(())
}

/// Compose the exact frame jet through the algebraic and differentiated
/// Eq. (40) solves.
///
/// Only sparse coefficient slices sharing one canonical normal-form monomial
/// are combined.  No dense `D H`, `D D H`, or compensator tensor is retained.
/// Scale and p=2 are emitted separately and never enter either solver.
pub fn visit_eq40_frame_composition<F>(
    input: &LinearizedFrameSuperfields,
    mut emit: F,
) -> Result<Eq40FrameCompositionStats, String>
where
    F: FnMut(Eq40FrameEntry) -> Result<(), String>,
{
    let mut d_h = MonomialSlices::new();
    let mut d_d_h = MonomialSlices::new();
    let mut independent_scale = MonomialSlices::new();
    let mut independent_psi_two = MonomialSlices::new();
    visit_linearized_frame_jet(input, |entry| {
        match entry.sector {
            LinearizedFrameJetSector::DH => {
                collect_polynomial(&mut d_h, entry.coordinate, entry.polynomial)
            }
            LinearizedFrameJetSector::DDH => {
                collect_polynomial(&mut d_d_h, entry.coordinate, entry.polynomial)
            }
            LinearizedFrameJetSector::DScale => {
                collect_polynomial(&mut independent_scale, entry.coordinate, entry.polynomial)
            }
            LinearizedFrameJetSector::DLorentzTwoForm => {
                collect_polynomial(&mut independent_psi_two, entry.coordinate, entry.polynomial)
            }
            LinearizedFrameJetSector::PH
            | LinearizedFrameJetSector::PScale
            | LinearizedFrameJetSector::PLorentzTwoForm
            | LinearizedFrameJetSector::DDScale => {}
        }
        Ok(())
    })?;

    let mut stats = Eq40FrameCompositionStats {
        d_h_monomial_slices: d_h.len(),
        d_d_h_monomial_slices: d_d_h.len(),
        emitted_by_sector: BTreeMap::new(),
    };

    for (monomial, slice) in &d_h {
        let solved = solve_conventional_compensators(slice);
        for (mask, value) in solved.psi_one {
            emit_one(
                &mut emit,
                &mut stats,
                Eq40FrameSector::PsiOne,
                form_ordinal(1, mask)?,
                monomial,
                value,
            )?;
        }
        for (mask, value) in solved.psi_three {
            emit_one(
                &mut emit,
                &mut stats,
                Eq40FrameSector::PsiThree,
                form_ordinal(3, mask)?,
                monomial,
                value,
            )?;
        }
        for (mask, value) in solved.psi_four {
            emit_one(
                &mut emit,
                &mut stats,
                Eq40FrameSector::PsiFour,
                form_ordinal(4, mask)?,
                monomial,
                value,
            )?;
        }
        for (mask, value) in solved.psi_five {
            emit_one(
                &mut emit,
                &mut stats,
                Eq40FrameSector::PsiFive,
                form_ordinal(5, mask)?,
                monomial,
                value,
            )?;
        }
    }

    for (monomial, slice) in &d_d_h {
        let solved = solve_higher_jet_conventional_compensators(slice);
        for (sector, values) in [
            (Eq40FrameSector::DPsiOne, solved.d_psi_one),
            (Eq40FrameSector::DPsiThree, solved.d_psi_three),
            (Eq40FrameSector::DPsiFour, solved.d_psi_four),
            (Eq40FrameSector::DPsiFive, solved.d_psi_five),
        ] {
            for (coordinate, value) in values {
                emit_one(&mut emit, &mut stats, sector, coordinate, monomial, value)?;
            }
        }
    }

    for (monomial, slice) in independent_scale {
        for (coordinate, value) in slice {
            emit_one(
                &mut emit,
                &mut stats,
                Eq40FrameSector::DScaleIndependent,
                coordinate,
                &monomial,
                value,
            )?;
        }
    }
    for (monomial, slice) in independent_psi_two {
        for (coordinate, value) in slice {
            emit_one(
                &mut emit,
                &mut stats,
                Eq40FrameSector::DPsiTwoIndependent,
                coordinate,
                &monomial,
                value,
            )?;
        }
    }
    Ok(stats)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Eq40FrameCompositionReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source: &'static str,
    pub source_sha256: &'static str,
    pub d_h_monomial_slices: usize,
    pub d_d_h_monomial_slices: usize,
    pub emitted_psi_one_terms: usize,
    pub emitted_psi_three_terms: usize,
    pub emitted_psi_four_terms: usize,
    pub emitted_psi_five_terms: usize,
    pub emitted_d_psi_one_terms: usize,
    pub emitted_d_psi_three_terms: usize,
    pub emitted_d_psi_four_terms: usize,
    pub emitted_d_psi_five_terms: usize,
    pub emitted_independent_scale_terms: usize,
    pub emitted_independent_psi_two_terms: usize,
    pub sparse_monomial_slicing: bool,
    pub conventional_compensators_composed: bool,
    pub differentiated_compensators_composed: bool,
    pub scale_and_p2_kept_independent: bool,
    pub geometry_anholonomies_composed: bool,
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
    for (spinor, vector, coefficient) in [(0, 0, 1), (5, 3, -2), (17, 8, 3), (31, 10, 1)] {
        input.h.insert(
            spinor * VECTOR_DIMENSION + vector,
            CanonicalSuperPolynomial::scalar(ExactQi::from_integer(coefficient)),
        );
    }
    input.lorentz_two_form.insert(
        7,
        CanonicalSuperPolynomial::scalar(ExactQi::from_integer(3)),
    );
    input
}

pub fn verify() -> Eq40FrameCompositionReport {
    let mut noncanonical_coordinates = 0_usize;
    let stats = visit_eq40_frame_composition(&probe_input(), |entry| {
        let dimension = match entry.sector {
            Eq40FrameSector::PsiOne => 11,
            Eq40FrameSector::PsiThree => 165,
            Eq40FrameSector::PsiFour => 330,
            Eq40FrameSector::PsiFive => 462,
            Eq40FrameSector::DPsiOne => SPINOR_DIMENSION * 11,
            Eq40FrameSector::DPsiThree => SPINOR_DIMENSION * 165,
            Eq40FrameSector::DPsiFour => SPINOR_DIMENSION * 330,
            Eq40FrameSector::DPsiFive => SPINOR_DIMENSION * 462,
            Eq40FrameSector::DScaleIndependent => SPINOR_DIMENSION,
            Eq40FrameSector::DPsiTwoIndependent => SPINOR_DIMENSION * LORENTZ_TWO_FORM_DIMENSION,
        };
        noncanonical_coordinates += usize::from(entry.coordinate >= dimension);
        Ok(())
    })
    .expect("deterministic Eq. (40) composition probe is valid");
    let count = |sector| stats.emitted_by_sector.get(&sector).copied().unwrap_or(0);
    let every_conventional_sector_nonzero = [
        Eq40FrameSector::PsiOne,
        Eq40FrameSector::PsiThree,
        Eq40FrameSector::PsiFour,
        Eq40FrameSector::PsiFive,
        Eq40FrameSector::DPsiOne,
        Eq40FrameSector::DPsiThree,
        Eq40FrameSector::DPsiFour,
        Eq40FrameSector::DPsiFive,
    ]
    .into_iter()
    .all(|sector| count(sector) != 0);
    let independent_present = count(Eq40FrameSector::DScaleIndependent) != 0
        && count(Eq40FrameSector::DPsiTwoIndependent) != 0;
    let passed = stats.d_h_monomial_slices != 0
        && stats.d_d_h_monomial_slices != 0
        && every_conventional_sector_nonzero
        && independent_present
        && noncanonical_coordinates == 0;

    Eq40FrameCompositionReport {
        schema_version: SCHEMA_VERSION,
        role: "exact ordered-polynomial composition of the constrained frame with the algebraic and differentiated Eq. (40) compensator solve",
        source: "Gates and Nishino, hep-th/0101037 Eqs. (1), (39), and (40)",
        source_sha256: crate::eleven_dimensional_physical_curvature::HEP_TH_0101037_SOURCE_SHA256,
        d_h_monomial_slices: stats.d_h_monomial_slices,
        d_d_h_monomial_slices: stats.d_d_h_monomial_slices,
        emitted_psi_one_terms: count(Eq40FrameSector::PsiOne),
        emitted_psi_three_terms: count(Eq40FrameSector::PsiThree),
        emitted_psi_four_terms: count(Eq40FrameSector::PsiFour),
        emitted_psi_five_terms: count(Eq40FrameSector::PsiFive),
        emitted_d_psi_one_terms: count(Eq40FrameSector::DPsiOne),
        emitted_d_psi_three_terms: count(Eq40FrameSector::DPsiThree),
        emitted_d_psi_four_terms: count(Eq40FrameSector::DPsiFour),
        emitted_d_psi_five_terms: count(Eq40FrameSector::DPsiFive),
        emitted_independent_scale_terms: count(Eq40FrameSector::DScaleIndependent),
        emitted_independent_psi_two_terms: count(Eq40FrameSector::DPsiTwoIndependent),
        sparse_monomial_slicing: true,
        conventional_compensators_composed: every_conventional_sector_nonzero,
        differentiated_compensators_composed: every_conventional_sector_nonzero,
        scale_and_p2_kept_independent: independent_present,
        geometry_anholonomies_composed: false,
        complete_physical_f_implemented: false,
        passed,
        result: "Every active ordered-superderivative monomial is now routed through the exact p=1,3,4,5 Eq. (40) solve, including its differentiated lift; scale and Lorentz p=2 remain explicit independent jets.",
        boundary: "This closes the frame-jet to differentiated-compensator boundary. The next gate must inject all five p-form sectors into D Delta with the audited Eq. (1) coefficients, then assemble Eqs. (26)/(28) from the same monomial slice and prove the p=2 orbit descends through J, torsion, and W.",
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

pub fn write_artifact(path: &Path) -> io::Result<Eq40FrameCompositionReport> {
    let report = verify();
    atomic_json(path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_eq40_composition_reaches_every_fixed_and_differentiated_sector() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert!(report.emitted_psi_one_terms > 0);
        assert!(report.emitted_psi_three_terms > 0);
        assert!(report.emitted_psi_four_terms > 0);
        assert!(report.emitted_psi_five_terms > 0);
        assert!(report.emitted_d_psi_one_terms > 0);
        assert!(report.emitted_d_psi_three_terms > 0);
        assert!(report.emitted_d_psi_four_terms > 0);
        assert!(report.emitted_d_psi_five_terms > 0);
    }

    #[test]
    fn independent_scale_and_p2_never_enter_conventional_solve() {
        let mut input = LinearizedFrameSuperfields {
            scale: CanonicalSuperPolynomial::scalar(ExactQi::one()),
            ..LinearizedFrameSuperfields::default()
        };
        input.lorentz_two_form.insert(
            0,
            CanonicalSuperPolynomial::scalar(ExactQi::from_integer(2)),
        );
        let mut sectors = BTreeMap::new();
        let stats = visit_eq40_frame_composition(&input, |entry| {
            *sectors.entry(entry.sector).or_insert(0_usize) += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(stats.d_h_monomial_slices, 0);
        assert_eq!(stats.d_d_h_monomial_slices, 0);
        assert_eq!(sectors.len(), 2);
        assert_eq!(sectors[&Eq40FrameSector::DScaleIndependent], 32);
        assert_eq!(sectors[&Eq40FrameSector::DPsiTwoIndependent], 32);
    }

    #[test]
    fn coefficient_slices_preserve_cancellation_and_callback_failure() {
        let mut slices = MonomialSlices::new();
        let polynomial = CanonicalSuperPolynomial::scalar(ExactQi::one());
        let (monomial, coefficient) = polynomial.terms.into_iter().next().unwrap();
        add_slice_term(&mut slices, monomial.clone(), 9, coefficient.clone());
        add_slice_term(
            &mut slices,
            monomial,
            9,
            coefficient.scaled(&num_rational::Ratio::from_integer(-1)),
        );
        assert!(slices.is_empty());

        let mut calls = 0;
        let error = visit_eq40_frame_composition(&probe_input(), |_| {
            calls += 1;
            Err("stop".to_string())
        })
        .unwrap_err();
        assert_eq!(error, "stop");
        assert_eq!(calls, 1);
    }
}
