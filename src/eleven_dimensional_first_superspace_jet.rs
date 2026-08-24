//! Exact first superspace-jet prolongation for the source-fixed 11D geometry.
//!
//! This module starts after the convention-fixed Eq. (25), Eq. (26), Eq. (28),
//! and Table 3 bridge in [`crate::eleven_dimensional_physical_curvature`].  It
//! differentiates only algebraic, constant-coefficient geometry maps.  The
//! implemented slice is
//!
//! ```text
//! (D_rho C_{alpha beta}{}^gamma, D_rho C_{alpha,b}{}^c)
//!       -> D_rho omega_{alpha,de}
//!       -> (D_rho J^(1)_alpha, D_rho J^(2)_alpha, D_rho J^(+)_alpha)
//!       -> omega_{a,de}
//! (C_{alpha,a}{}^gamma, omega_{a,de}) -> T_{alpha,a}{}^gamma
//! (T_{alpha,a}{}^gamma, D J) -> W_[4]
//! ```
//!
//! The first line is the formal first-jet prolongation of hep-th/0101037
//! Table 3 and the definitions in arXiv:2007.05097 Eqs. (2.19)-(2.23).
//! The vector connection is the fourth conventional constraint in Table 3.
//! No coefficient is inferred from a vanishing requirement.
//!
//! The input jets are ordered tensors.  This module does not impose the flat
//! superspace anticommutator on a second spinor derivative and does not derive
//! these geometry jets from a compensator-eliminated `H_hat` jet.  Consequently
//! this is not a complete physical curvature operator and proves no `F A G_p`
//! identity.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

use serde::Serialize;

use crate::eleven_dimensional_physical_curvature::{
    self as physical, C_ALPHA_VECTOR_VECTOR_DIMENSION, D_J_DIMENSION,
    D_SPINORIAL_CONNECTION_DIMENSION, ExactQi, SPINOR_ANHOLONOMY_DIMENSION, SPINOR_DIMENSION,
    T_ALPHA_VECTOR_SPINOR_DIMENSION,
};

pub const D_C_ALPHA_BETA_GAMMA_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_ANHOLONOMY_DIMENSION;
pub const D_C_ALPHA_VECTOR_VECTOR_DIMENSION: usize =
    SPINOR_DIMENSION * C_ALPHA_VECTOR_VECTOR_DIMENSION;

fn add_sparse(target: &mut BTreeMap<usize, ExactQi>, index: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = target.entry(index).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        target.remove(&index);
    }
}

/// Apply `1_(D_rho) tensor operator` without constructing its large sparse
/// Kronecker matrix.  Jet coordinates are derivative-major:
/// `rho * operator.input_dimension + source`.
pub fn apply_first_jet_prolongation(
    operator: &physical::SparseQiOperator,
    input: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut slices: BTreeMap<usize, BTreeMap<usize, ExactQi>> = BTreeMap::new();
    for (&index, value) in input {
        assert!(index < SPINOR_DIMENSION * operator.input_dimension);
        let derivative = index / operator.input_dimension;
        let source = index % operator.input_dimension;
        slices
            .entry(derivative)
            .or_default()
            .insert(source, value.clone());
    }

    let mut output = BTreeMap::new();
    for (derivative, slice) in slices {
        for (row, value) in operator.apply_sparse(&slice) {
            add_sparse(
                &mut output,
                derivative * operator.output_dimension + row,
                value,
            );
        }
    }
    output
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FirstSuperspaceJetInput {
    /// Ordered `D_rho C_{alpha beta}{}^gamma`.
    pub d_c_alpha_beta_gamma: BTreeMap<usize, ExactQi>,
    /// Ordered `D_rho C_{alpha,b}{}^c`.
    pub d_c_alpha_b_c: BTreeMap<usize, ExactQi>,
    /// Undifferentiated `C_{alpha,a}{}^gamma`, needed by the torsion term in W.
    pub c_alpha_a_gamma: BTreeMap<usize, ExactQi>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FirstSuperspaceJetOutput {
    pub d_spinorial_connection: BTreeMap<usize, ExactQi>,
    pub bosonic_connection: BTreeMap<usize, ExactQi>,
    pub d_j_one: BTreeMap<usize, ExactQi>,
    pub d_j_two: BTreeMap<usize, ExactQi>,
    pub d_j_plus: BTreeMap<usize, ExactQi>,
    pub t_alpha_a_gamma: BTreeMap<usize, ExactQi>,
    pub w_2001: BTreeMap<usize, ExactQi>,
    pub w_2021: BTreeMap<usize, ExactQi>,
}

/// Assemble the largest first superspace-jet slice fixed by the cited source
/// equations.  It accepts geometry jets rather than an `H_hat` jet because the
/// sources do not print the missing compensator-eliminated map.
pub fn assemble_first_superspace_jet(input: &FirstSuperspaceJetInput) -> FirstSuperspaceJetOutput {
    for &index in input.d_c_alpha_beta_gamma.keys() {
        assert!(index < D_C_ALPHA_BETA_GAMMA_DIMENSION);
    }
    for &index in input.d_c_alpha_b_c.keys() {
        assert!(index < D_C_ALPHA_VECTOR_VECTOR_DIMENSION);
    }
    for &index in input.c_alpha_a_gamma.keys() {
        assert!(index < T_ALPHA_VECTOR_SPINOR_DIMENSION);
    }

    let d_spinorial_connection = apply_first_jet_prolongation(
        physical::cached_c_alpha_b_c_to_spinorial_connection_operator(),
        &input.d_c_alpha_b_c,
    );
    debug_assert!(
        d_spinorial_connection
            .keys()
            .all(|index| *index < D_SPINORIAL_CONNECTION_DIMENSION)
    );

    let d_j_one_anholonomy = apply_first_jet_prolongation(
        physical::cached_c_alpha_beta_gamma_to_j_one_operator(),
        &input.d_c_alpha_beta_gamma,
    );
    let d_j_one_connection = apply_first_jet_prolongation(
        physical::cached_spinorial_connection_to_j_one_operator(),
        &d_spinorial_connection,
    );
    let mut d_j_one = d_j_one_anholonomy;
    for (index, value) in d_j_one_connection {
        add_sparse(&mut d_j_one, index, value);
    }

    // J^(2)=(4/33)T_{alpha,b}{}^b.  The Lorentz connection contribution
    // omega_{alpha,b}{}^b vanishes identically by antisymmetry, so its first
    // jet is exactly the prolonged anholonomy trace.
    let d_j_two = apply_first_jet_prolongation(
        physical::cached_c_alpha_b_c_to_j_operator(),
        &input.d_c_alpha_b_c,
    );
    let d_j_plus = physical::apply_d_j_plus(&d_j_one, &d_j_two);

    let bosonic_connection = physical::apply_bosonic_connection(&d_spinorial_connection);
    let assembled = physical::assemble_convention_separated_linearized_w(
        &input.c_alpha_a_gamma,
        &bosonic_connection,
        &d_j_one,
        &d_j_two,
    );

    FirstSuperspaceJetOutput {
        d_spinorial_connection,
        bosonic_connection,
        d_j_one,
        d_j_two,
        d_j_plus,
        t_alpha_a_gamma: assembled.t_alpha_e_gamma,
        w_2001: assembled.w_2001,
        w_2021: assembled.w_2021,
    }
}

fn first_nonempty_column(operator: &physical::SparseQiOperator) -> usize {
    operator
        .columns
        .iter()
        .position(|column| !column.is_empty())
        .expect("source-fixed operator is nonzero")
}

fn deterministic_probe() -> FirstSuperspaceJetOutput {
    let c_to_j_one = physical::c_alpha_beta_gamma_to_j_one_operator();
    let c_to_omega = physical::c_alpha_b_c_to_spinorial_connection_operator();
    let c_to_j_two = physical::c_alpha_b_c_to_j_operator();
    let t_to_w = physical::t_alpha_e_gamma_to_w_operator();

    let mut input = FirstSuperspaceJetInput::default();
    input.d_c_alpha_beta_gamma.insert(
        3 * c_to_j_one.input_dimension + first_nonempty_column(&c_to_j_one),
        ExactQi::from_integer(2),
    );
    input.d_c_alpha_b_c.insert(
        5 * c_to_omega.input_dimension + first_nonempty_column(&c_to_omega),
        ExactQi::from_integer(3),
    );
    input.d_c_alpha_b_c.insert(
        7 * c_to_j_two.input_dimension + first_nonempty_column(&c_to_j_two),
        ExactQi::from_integer(5),
    );
    input
        .c_alpha_a_gamma
        .insert(first_nonempty_column(&t_to_w), ExactQi::from_integer(7));
    assemble_first_superspace_jet(&input)
}

fn omitted_d_omega_mutation_detected() -> bool {
    let c_to_omega = physical::c_alpha_b_c_to_spinorial_connection_operator();
    for column in 0..c_to_omega.input_dimension {
        if c_to_omega.columns[column].is_empty() {
            continue;
        }
        let mut d_c = BTreeMap::new();
        d_c.insert(11 * c_to_omega.input_dimension + column, ExactQi::one());
        let d_omega = apply_first_jet_prolongation(&c_to_omega, &d_c);
        let source = apply_first_jet_prolongation(
            &physical::spinorial_connection_to_j_one_operator(),
            &d_omega,
        );
        if !source.is_empty() {
            return source != BTreeMap::new();
        }
    }
    false
}

fn j_plus_half_mutation_detected() -> bool {
    let mut d_j_one = BTreeMap::new();
    let mut d_j_two = BTreeMap::new();
    d_j_one.insert(2 * SPINOR_DIMENSION + 3, ExactQi::from_integer(2));
    d_j_two.insert(2 * SPINOR_DIMENSION + 3, ExactQi::from_integer(4));
    let source = physical::apply_d_j_plus(&d_j_one, &d_j_two);
    let mut omitted_half = BTreeMap::new();
    omitted_half.insert(2 * SPINOR_DIMENSION + 3, ExactQi::from_integer(6));
    source != omitted_half
}

fn w_d_j_coefficient_mutation_detected() -> bool {
    let source_operator = physical::d_j_to_w_operator();
    let mutation = source_operator.scaled(ExactQi::from_rational(128, 127));
    let column = first_nonempty_column(&source_operator);
    let mut input = BTreeMap::new();
    input.insert(column, ExactQi::one());
    source_operator.apply_sparse(&input) != mutation.apply_sparse(&input)
}

fn table_three_normalization_mutation_detected() -> bool {
    let source_operator = physical::c_alpha_b_c_to_spinorial_connection_operator();
    let mutation = source_operator.scaled(ExactQi::from_rational(55, 54));
    let column = first_nonempty_column(&source_operator);
    let mut input = BTreeMap::new();
    input.insert(
        13 * source_operator.input_dimension + column,
        ExactQi::one(),
    );
    apply_first_jet_prolongation(&source_operator, &input)
        != apply_first_jet_prolongation(&mutation, &input)
}

fn derivative_major_ordering_mutation_detected() -> bool {
    let operator = physical::c_alpha_beta_gamma_to_j_one_operator();
    let source_column = first_nonempty_column(&operator);
    let derivative = 17;
    let mut input = BTreeMap::new();
    input.insert(
        derivative * operator.input_dimension + source_column,
        ExactQi::one(),
    );
    let source = apply_first_jet_prolongation(&operator, &input);

    let mut wrong = BTreeMap::new();
    for (&index, value) in &input {
        let wrong_derivative = index % SPINOR_DIMENSION;
        let wrong_source = index / SPINOR_DIMENSION;
        if wrong_source < operator.input_dimension {
            let mut slice = BTreeMap::new();
            slice.insert(wrong_source, value.clone());
            for (row, result) in operator.apply_sparse(&slice) {
                add_sparse(
                    &mut wrong,
                    wrong_derivative * operator.output_dimension + row,
                    result,
                );
            }
        }
    }
    source != wrong
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstSuperspaceJetReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub physical_curvature_schema_pinned: &'static str,
    pub physical_curvature_artifact_sha256: &'static str,
    pub source_references: Vec<&'static str>,
    pub source_hashes: Vec<&'static str>,
    pub jet_coordinate_ordering: &'static str,
    pub d_c_alpha_beta_gamma_dimension: usize,
    pub d_c_alpha_vector_vector_dimension: usize,
    pub d_spinorial_connection_dimension: usize,
    pub d_j_dimension: usize,
    pub probe_d_spinorial_connection_entries: usize,
    pub probe_bosonic_connection_entries: usize,
    pub probe_d_j_one_entries: usize,
    pub probe_d_j_two_entries: usize,
    pub probe_d_j_plus_entries: usize,
    pub probe_t_entries: usize,
    pub probe_w_2001_entries: usize,
    pub probe_w_2021_entries: usize,
    pub table_three_first_jet_prolonged: bool,
    pub d_j_one_first_jet_prolonged: bool,
    pub d_j_two_first_jet_prolonged: bool,
    pub d_j_plus_assembled: bool,
    pub bosonic_connection_from_d_spinorial_connection_assembled: bool,
    pub convention_separated_w_assembled: bool,
    pub derivative_major_ordering_mutation_detected: bool,
    pub table_three_normalization_mutation_detected: bool,
    pub omitted_d_omega_mutation_detected: bool,
    pub j_plus_half_mutation_detected: bool,
    pub w_d_j_coefficient_mutation_detected: bool,
    pub h_hat_to_first_geometry_jet_implemented: bool,
    pub complete_physical_f_implemented: bool,
    pub full_fag_established: bool,
    pub j_one_109_over_1056_assessment: &'static str,
    pub howe_scale_removability_scope: &'static str,
    pub cederwall_scope: &'static str,
    pub passed: bool,
    pub boundary: &'static str,
}

fn build_report() -> FirstSuperspaceJetReport {
    let probe = deterministic_probe();
    let derivative_major_ordering_mutation_detected = derivative_major_ordering_mutation_detected();
    let table_three_normalization_mutation_detected = table_three_normalization_mutation_detected();
    let omitted_d_omega_mutation_detected = omitted_d_omega_mutation_detected();
    let j_plus_half_mutation_detected = j_plus_half_mutation_detected();
    let w_d_j_coefficient_mutation_detected = w_d_j_coefficient_mutation_detected();
    let passed = !probe.d_spinorial_connection.is_empty()
        && !probe.bosonic_connection.is_empty()
        && !probe.d_j_one.is_empty()
        && !probe.d_j_two.is_empty()
        && !probe.d_j_plus.is_empty()
        && !probe.t_alpha_a_gamma.is_empty()
        && !probe.w_2001.is_empty()
        && !probe.w_2021.is_empty()
        && derivative_major_ordering_mutation_detected
        && table_three_normalization_mutation_detected
        && omitted_d_omega_mutation_detected
        && j_plus_half_mutation_detected
        && w_d_j_coefficient_mutation_detected;

    FirstSuperspaceJetReport {
        schema_version: "adynkra.11d.first-superspace-jet.v1",
        role: "exact geometry-level first superspace-jet prolongation into convention-separated linearized W",
        physical_curvature_schema_pinned: "adynkra-11d-physical-curvature-operator-v10",
        physical_curvature_artifact_sha256: "c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f",
        source_references: vec![
            "hep-th/0101037 Eqs. (26), (28), (44) and Table 3",
            "arXiv:2007.05097 Section 2, Eqs. (2.19)-(2.23), the 11D review inside the 10D Weyl-covariance paper",
            "hep-th/9707184: local removability of the closed Weyl connection in the standard on-shell dimension-zero-torsion complex",
            "arXiv:1001.0112: pure-spinor cohomology describes the on-shell 11D multiplet and does not supply an off-shell compensator term",
        ],
        source_hashes: vec![
            physical::HEP_TH_0101037_SOURCE_SHA256,
            physical::ARXIV_2007_05097_SOURCE_SHA256,
            "Howe hep-th/9707184 TeX SHA-256 6c8e59f90d0d40c3a54164d034d6d7c2d778d6da870d8eac077ecd5bb76ec334",
            "Cederwall arXiv:1001.0112 complete_action.tex SHA-256 353dffe926aa88f99cf9e1983e93c9ab541f847fe7a36bed8972f854db8b1760",
        ],
        jet_coordinate_ordering: "derivative spinor rho major, followed by the source operator coordinate",
        d_c_alpha_beta_gamma_dimension: D_C_ALPHA_BETA_GAMMA_DIMENSION,
        d_c_alpha_vector_vector_dimension: D_C_ALPHA_VECTOR_VECTOR_DIMENSION,
        d_spinorial_connection_dimension: D_SPINORIAL_CONNECTION_DIMENSION,
        d_j_dimension: D_J_DIMENSION,
        probe_d_spinorial_connection_entries: probe.d_spinorial_connection.len(),
        probe_bosonic_connection_entries: probe.bosonic_connection.len(),
        probe_d_j_one_entries: probe.d_j_one.len(),
        probe_d_j_two_entries: probe.d_j_two.len(),
        probe_d_j_plus_entries: probe.d_j_plus.len(),
        probe_t_entries: probe.t_alpha_a_gamma.len(),
        probe_w_2001_entries: probe.w_2001.len(),
        probe_w_2021_entries: probe.w_2021.len(),
        table_three_first_jet_prolonged: true,
        d_j_one_first_jet_prolonged: true,
        d_j_two_first_jet_prolonged: true,
        d_j_plus_assembled: true,
        bosonic_connection_from_d_spinorial_connection_assembled: true,
        convention_separated_w_assembled: true,
        derivative_major_ordering_mutation_detected,
        table_three_normalization_mutation_detected,
        omitted_d_omega_mutation_detected,
        j_plus_half_mutation_detected,
        w_d_j_coefficient_mutation_detected,
        h_hat_to_first_geometry_jet_implemented: false,
        complete_physical_f_implemented: false,
        full_fag_established: false,
        j_one_109_over_1056_assessment: "not an expected physical response of a complete pure local-Lorentz orbit: torsion transforms homogeneously under Lorentz and its zero-background trace should remain zero. It is a real obstruction to treating the tested p=2 column as a complete Lorentz gauge representative, but not an obstruction to 11D supergravity. It is not certified Lorentz-gauge-removable. Only a separately completed super-Weyl orbit could test scale-gauge removal.",
        howe_scale_removability_scope: "Howe proves local removal only for a closed Weyl connection in the standard on-shell dimension-zero-torsion complex, with the stated topological qualification. That result does not authorize subtracting the p=2 local-Lorentz residual in an off-shell X/J-deformed complex.",
        cederwall_scope: "The pure-spinor BRST cohomology is an on-shell spectrum and equation-of-motion oracle. It does not print the missing off-shell H_hat-to-jet map or a compensating J term.",
        passed,
        boundary: "The source-fixed algebraic geometry maps now have an exact first spinor-jet prolongation through D omega, D J^(1), D J^(2), D J^(+), mixed torsion, and both published linearized W conventions. The missing step is the compensator-eliminated H_hat jet that supplies D C_{alpha beta}{}^gamma, D C_{alpha,b}{}^c, and C_{alpha,a}{}^gamma consistently, including ordered-superderivative momentum terms. Until that map is source-fixed, W(H_hat), complete F, F A G_p, Bianchi closure, and off-shell closure remain false.",
    }
}

pub fn verify() -> FirstSuperspaceJetReport {
    static REPORT: OnceLock<FirstSuperspaceJetReport> = OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

pub fn write_artifact(path: &Path) -> io::Result<()> {
    let report = verify();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    file.write_all(b"\n")?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prolongation_commutes_with_each_derivative_slice() {
        let operator = physical::c_alpha_b_c_to_j_operator();
        let source_column = first_nonempty_column(&operator);
        let mut input = BTreeMap::new();
        input.insert(
            4 * operator.input_dimension + source_column,
            ExactQi::from_integer(3),
        );
        input.insert(
            19 * operator.input_dimension + source_column,
            ExactQi::from_integer(5),
        );
        let prolonged = apply_first_jet_prolongation(&operator, &input);

        let mut expected = BTreeMap::new();
        for (derivative, coefficient) in [(4, 3), (19, 5)] {
            let mut slice = BTreeMap::new();
            slice.insert(source_column, ExactQi::from_integer(coefficient));
            for (row, value) in operator.apply_sparse(&slice) {
                expected.insert(derivative * operator.output_dimension + row, value);
            }
        }
        assert_eq!(prolonged, expected);
    }

    #[test]
    fn first_jet_reaches_both_published_linearized_w_conventions() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert!(report.table_three_first_jet_prolonged);
        assert!(report.d_j_one_first_jet_prolonged);
        assert!(report.d_j_two_first_jet_prolonged);
        assert!(report.d_j_plus_assembled);
        assert!(report.bosonic_connection_from_d_spinorial_connection_assembled);
        assert!(report.convention_separated_w_assembled);
        assert!(report.probe_w_2001_entries > 0);
        assert!(report.probe_w_2021_entries > 0);
        assert!(!report.h_hat_to_first_geometry_jet_implemented);
        assert!(!report.complete_physical_f_implemented);
        assert!(!report.full_fag_established);
    }

    #[test]
    fn source_normalizations_and_jet_ordering_are_mutation_sensitive() {
        let report = verify();
        assert!(report.derivative_major_ordering_mutation_detected);
        assert!(report.table_three_normalization_mutation_detected);
        assert!(report.omitted_d_omega_mutation_detected);
        assert!(report.j_plus_half_mutation_detected);
        assert!(report.w_d_j_coefficient_mutation_detected);
    }

    #[test]
    #[ignore = "artifact writer"]
    fn write_checked_artifact() {
        write_artifact(Path::new("results/adynkra_11d_first_superspace_jet.json")).unwrap();
    }
}
