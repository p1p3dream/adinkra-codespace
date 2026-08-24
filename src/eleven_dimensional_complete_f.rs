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
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Serialize;

use crate::eleven_dimensional_constrained_geometry_jet::{
    ConstrainedGeometryJetSector, visit_constrained_geometry_jet,
};
use crate::eleven_dimensional_first_superspace_jet::{
    self as first_jet, FirstSuperspaceJetInput, FirstSuperspaceJetOutput,
};
use crate::eleven_dimensional_h_hat_jet::{
    LinearizedFrameJetSector, LinearizedFrameSuperfields, canonical_physical_frame_representative,
    visit_linearized_frame_jet,
};
use crate::eleven_dimensional_physical_curvature::{self as physical, ExactQi, PhysicalXImage};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-complete-physical-f-construction-v4";

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
    let first_jet = first_jet::assemble_first_superspace_jet(&input.first_jet);

    Ok(GeometryLevelPhysicalFOutput {
        x,
        spinorial_connection,
        j_one,
        j_two,
        j_plus,
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
                sector: "J^(1), J^(2), J^(+)",
                domain: "spinor and mixed anholonomy plus spinorial connection",
                codomain: "spinor geometry",
                exact_operator_available: true,
                composed_from_h_hat: true,
                source: "hep-th/0101037 Eq. (44), Table 3; arXiv:2007.05097 Eq. (2.21)",
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
        target_curvature_adapter_implemented: false,
        target_bianchi_euler_noether_composition_certified: false,
        complete_physical_f_implemented: false,
        complete_f_operator_sha256: None,
        exact_polynomial_target_kernel_derived: false,
        pointwise_or_bounded_kernel_is_accepted_as_physical_k: false,
        next_executable_step: "map the exact gauge-fixed X/J/T/W stream into the typed physical target curvature complex, hash the complete polynomial operator, and solve its exact H_hat kernel",
        passed,
        result: "The exact ordered-superderivative frame now composes H_hat and scale through Delta, both Eqs. (13)-(14) anholonomies, D C, both Lorentz connections, J, mixed torsion, and both W conventions. P_320 and the p=2 gauge choice are enforced at the typed input boundary. Complete F now waits only on the target-curvature/Bianchi adapter and canonical operator hash; no fitted J counterterm or premature K is introduced.",
        boundary: "Passing this report certifies the exact gauge-fixed H_hat-to-X/J/T/W differential stream. The raw p=2 lift has a nonzero J/T/W response in the current convention, so this is explicitly a canonical local-Lorentz gauge section rather than a proof that the raw coordinates descend unchanged. Physical F still requires the target curvature/Bianchi composition. Physical K is then the exact polynomial H_hat kernel of the hashed complete F, not a numerical or bounded-slice nullspace.",
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
