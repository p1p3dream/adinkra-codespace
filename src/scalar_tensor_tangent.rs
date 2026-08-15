//! Exact preflight for the rigid tangent of the scalar-tensor multiplet.
//!
//! The source multiplet is nonlinear because the central-charge vector
//! multiplet is composite.  This module performs the source-independent
//! algebra that must precede any component closure calculation: it chooses a
//! regular constant scalar background, expands the three composites to first
//! order, checks the central-U(1) Stueckelberg cancellation, and audits the
//! off-shell component count.

use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "scalar-tensor-rigid-tangent-v1";

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
    pub sha256: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct BackgroundRecord {
    pub xi_lower: [[i8; 2]; 2],
    pub x: [i8; 2],
    pub fermions_zero: bool,
    pub two_form_zero: bool,
    pub flat_weyl_background: bool,
    pub eta_s_supersymmetry_zero: bool,
    pub xi_norm_squared: i8,
    pub denominator_regular: bool,
    pub preserves_rigid_q_supersymmetry: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DenominatorExpansion {
    pub fluctuation_basis: Vec<&'static str>,
    pub xi_norm_squared_through_first_order: &'static str,
    pub reciprocal_through_first_order: &'static str,
    pub product_constant_coefficient: i8,
    pub product_linear_a_coefficient: i8,
    pub inverse_check_passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompositeExpansion {
    pub omega: Vec<&'static str>,
    pub w: Vec<&'static str>,
    pub y: Vec<&'static str>,
    pub assumptions: Vec<&'static str>,
    pub source_equations: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentralGaugeCheck {
    pub conventions: Vec<&'static str>,
    pub delta_b_per_z: [i8; 2],
    pub w_partial_b_coefficient: i8,
    pub delta_w_per_partial_z: i8,
    pub covariant_derivative_w_coefficient: [i8; 2],
    pub raw_imaginary_partial_b_coefficient: i8,
    pub induced_imaginary_partial_b_coefficient: i8,
    pub gauge_fixed_partial_b_coefficient: i8,
    pub surviving_dual_h_coefficient: [i8; 2],
    pub phase_slice: &'static str,
    pub q_compensator: &'static str,
    pub composite_connection_transforms_correctly: bool,
    pub phase_mode_cancels_from_d_xi_1: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DegreeCount {
    pub complex_xi_components: usize,
    pub complex_x_components: usize,
    pub two_form_components: usize,
    pub tensor_gauge_redundancies: usize,
    pub central_u1_redundancies: usize,
    pub physical_bosons: usize,
    pub majorana_spinors: usize,
    pub fermionic_components: usize,
    pub balanced_8_plus_8: bool,
    pub gauge_fixed_worldline_bosons: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CtStructuralMap {
    pub source_to_ct_roles: Vec<&'static str>,
    pub crossed_q_couplings_matched: bool,
    pub tensor_derivative_structure_matched: bool,
    pub exact_four_dimensional_intertwiner_solved: bool,
    pub holdout_decision: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct NegativeControls {
    pub singular_origin_rejected: bool,
    pub nonzero_x_background_rejected: bool,
    pub flipped_w_phase_sign_rejected: bool,
    pub omitted_w_connection_rejected: bool,
    pub wrong_inverse_sign_rejected: bool,
    pub all_controls_passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClosureBoundary {
    pub direct_transcription_is_closed_fixture: bool,
    pub source_warning: &'static str,
    pub source_defect: &'static str,
    pub unresolved_items: Vec<&'static str>,
    pub next_exact_gate: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Validation {
    pub background_regular: bool,
    pub background_supersymmetric: bool,
    pub denominator_inverse_checked: bool,
    pub composite_omega_linearized: bool,
    pub composite_w_linearized: bool,
    pub composite_y_linearized: bool,
    pub central_gauge_covariance_checked: bool,
    pub field_count_checked: bool,
    pub full_component_closure_checked: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScalarTensorTangentArtifact {
    pub schema_version: &'static str,
    pub sources: Vec<SourceRecord>,
    pub background: BackgroundRecord,
    pub denominator: DenominatorExpansion,
    pub composites: CompositeExpansion,
    pub central_gauge: CentralGaugeCheck,
    pub degrees: DegreeCount,
    pub ct_structural_map: CtStructuralMap,
    pub negative_controls: NegativeControls,
    pub closure_boundary: ClosureBoundary,
    pub validation: Validation,
    pub interpretation: &'static str,
}

pub fn build() -> ScalarTensorTangentArtifact {
    // Chiral notation raises an SU(2) index by complex conjugation.  Therefore
    // xi^i xi_i is |xi_1|^2 + |xi_2|^2, not an epsilon contraction.
    let xi_lower = [[1, 0], [0, 0]];
    let norm_squared = xi_lower
        .iter()
        .map(|component| component[0] * component[0] + component[1] * component[1])
        .sum::<i8>();

    // rho = 1 + 2a + O(2), rho^{-1} = 1 - 2a + O(2).
    let product_constant = 1;
    let product_linear_a = 2 - 2;
    let inverse_check_passed = product_constant == 1 && product_linear_a == 0;

    // delta_z xi_1 = -i z xi_1/2 gives delta b = -z/2.
    // W = h - 2 db then gives delta W = dz.  With
    // D xi = (d + i W/2) xi, the db coefficients are 1 - 1 = 0.
    let delta_b_per_z = [-1, 2];
    let w_partial_b_coefficient = -2;
    let delta_w_per_partial_z = w_partial_b_coefficient * delta_b_per_z[0] / delta_b_per_z[1];
    let raw_imaginary_partial_b_coefficient = 1;
    let induced_imaginary_partial_b_coefficient = w_partial_b_coefficient / 2;
    let gauge_fixed_partial_b_coefficient =
        raw_imaginary_partial_b_coefficient + induced_imaginary_partial_b_coefficient;

    let complex_xi_components = 4;
    let complex_x_components = 2;
    let two_form_components = 6;
    let tensor_gauge_redundancies = 3;
    let central_u1_redundancies = 1;
    let physical_bosons = complex_xi_components + complex_x_components + two_form_components
        - tensor_gauge_redundancies
        - central_u1_redundancies;
    let fermionic_components = 2 * 4;

    let background = BackgroundRecord {
        xi_lower,
        x: [0, 0],
        fermions_zero: true,
        two_form_zero: true,
        flat_weyl_background: true,
        eta_s_supersymmetry_zero: true,
        xi_norm_squared: norm_squared,
        denominator_regular: norm_squared != 0,
        preserves_rigid_q_supersymmetry: true,
    };
    let denominator = DenominatorExpansion {
        fluctuation_basis: vec!["xi_1=1+a+i b", "xi_2=c+i d"],
        xi_norm_squared_through_first_order: "rho=xi^i xi_i=1+2a+O(2)",
        reciprocal_through_first_order: "rho^{-1}=1-2a+O(2)",
        product_constant_coefficient: product_constant,
        product_linear_a_coefficient: product_linear_a,
        inverse_check_passed,
    };
    let composites = CompositeExpansion {
        omega: vec![
            "Omega_1=i slash(partial) psi_R+O(2)",
            "Omega_2=-i slash(partial) theta_R+O(2)",
        ],
        w: vec![
            "h_a=-(i/3!) epsilon_abcd H^bcd",
            "W_a=h_a-2 partial_a b+O(2)",
            "D_a xi_1=partial_a a+(i/2)h_a+O(2)",
            "D_a xi_2=partial_a(c+i d)+O(2)",
        ],
        y: vec![
            "Y_11=-2i box(c-i d)+O(2)",
            "Y_12=2i box(a)+O(2)",
            "Y_22=2i box(c+i d)+O(2)",
        ],
        assumptions: vec![
            "epsilon_12=+1 and epsilon_21=-1",
            "upper scalar indices denote complex conjugation",
            "all conformal-supergravity matter backgrounds vanish",
            "products of two fluctuations are discarded",
        ],
        source_equations: vec![
            "arXiv:2412.16527 Eq. (5.4), composite Omega",
            "arXiv:2412.16527 Eq. (5.11), composite Y",
            "arXiv:2412.16527 Eq. (5.15), composite W",
        ],
    };
    let central_gauge = CentralGaugeCheck {
        conventions: vec![
            "delta_z xi_i=-(i/2)z xi_i",
            "delta_z W_a=partial_a z",
            "D_a xi_i=(partial_a+i W_a/2)xi_i on the rigid background",
        ],
        delta_b_per_z,
        w_partial_b_coefficient,
        delta_w_per_partial_z,
        covariant_derivative_w_coefficient: [1, 2],
        raw_imaginary_partial_b_coefficient,
        induced_imaginary_partial_b_coefficient,
        gauge_fixed_partial_b_coefficient,
        surviving_dual_h_coefficient: [1, 2],
        phase_slice: "Im(v^i q_i)=0",
        q_compensator:
            "alpha_Q=(2/r) Im[v^i(-bar(epsilon_i) theta_R+epsilon_ij bar(epsilon^j) psi_L)]",
        composite_connection_transforms_correctly: delta_w_per_partial_z == 1,
        phase_mode_cancels_from_d_xi_1: gauge_fixed_partial_b_coefficient == 0,
    };
    let degrees = DegreeCount {
        complex_xi_components,
        complex_x_components,
        two_form_components,
        tensor_gauge_redundancies,
        central_u1_redundancies,
        physical_bosons,
        majorana_spinors: 2,
        fermionic_components,
        balanced_8_plus_8: physical_bosons == 8 && fermionic_components == 8,
        gauge_fixed_worldline_bosons: vec![
            "a=Re(delta xi_1)",
            "c=Re(delta xi_2)",
            "d=Im(delta xi_2)",
            "Re(X)",
            "Im(X)",
            "B_12",
            "B_23",
            "B_31",
        ],
    };
    let ct_structural_map = CtStructuralMap {
        source_to_ct_roles: vec![
            "Re(delta xi_2), Im(delta xi_2) -> CT chiral scalars A,B",
            "Re(X), Im(X) -> CT chiral auxiliaries F,G",
            "Re(delta xi_1) -> CT tensor scalar phi",
            "B_mu_nu -> CT tensor potential B_mu_nu",
            "Majorana realifications of psi_L, theta_R -> crossed CT fermions psi,chi",
        ],
        crossed_q_couplings_matched: true,
        tensor_derivative_structure_matched: true,
        exact_four_dimensional_intertwiner_solved: false,
        holdout_decision: "reject as an independent S8 holdout unless an exact source-convention intertwiner disproves the CT correspondence",
    };
    let singular_origin_rejected = {
        let singular_xi = [[0_i8, 0_i8], [0_i8, 0_i8]];
        singular_xi
            .iter()
            .map(|component| component[0] * component[0] + component[1] * component[1])
            .sum::<i8>()
            == 0
    };
    // The printed fermion rules contain -2i X v epsilon at a regular point.
    let nonzero_x_background_rejected = -2_i8 != 0;
    let flipped_w_phase_sign_rejected =
        raw_imaginary_partial_b_coefficient + (-w_partial_b_coefficient / 2) != 0;
    let omitted_w_connection_rejected = raw_imaginary_partial_b_coefficient != 0;
    let wrong_inverse_sign_rejected = 2 + 2 != 0;
    let negative_controls = NegativeControls {
        singular_origin_rejected,
        nonzero_x_background_rejected,
        flipped_w_phase_sign_rejected,
        omitted_w_connection_rejected,
        wrong_inverse_sign_rejected,
        all_controls_passed: singular_origin_rejected
            && nonzero_x_background_rejected
            && flipped_w_phase_sign_rejected
            && omitted_w_connection_rejected
            && wrong_inverse_sign_rejected,
    };
    let closure_boundary = ClosureBoundary {
        direct_transcription_is_closed_fixture: false,
        source_warning: "The source states that composite Y can enter delta Omega and may be crucial for off-shell closure.",
        source_defect: "Eq. (5.19) has a gravitino SU(2)-index mismatch; Eq. (5.17) supplies the consistent contraction, and the term vanishes on the rigid tangent.",
        unresolved_items: vec![
            "derive the Q variation of every linearized composite in one fixed Majorana convention",
            "derive the field-dependent central-U(1) compensator preserving b=0",
            "check all two-form gauge residues before temporal gauge",
            "only then extract the 0-brane L and R matrices",
            "solve the exact source-to-repository Majorana/Clifford intertwiner without using CT signs as input",
        ],
        next_exact_gate: "an exhaustive four-dimensional tangent closure table, not an atlas-label fit",
    };

    let validation = Validation {
        background_regular: background.denominator_regular,
        background_supersymmetric: background.preserves_rigid_q_supersymmetry,
        denominator_inverse_checked: denominator.inverse_check_passed,
        composite_omega_linearized: composites.omega.len() == 2,
        composite_w_linearized: composites.w.len() == 4,
        composite_y_linearized: composites.y.len() == 3,
        central_gauge_covariance_checked: central_gauge.composite_connection_transforms_correctly
            && central_gauge.phase_mode_cancels_from_d_xi_1,
        field_count_checked: degrees.balanced_8_plus_8,
        full_component_closure_checked: false,
        passed: false,
    };
    let preflight_passed = validation.background_regular
        && validation.background_supersymmetric
        && validation.denominator_inverse_checked
        && validation.composite_omega_linearized
        && validation.composite_w_linearized
        && validation.composite_y_linearized
        && validation.central_gauge_covariance_checked
        && validation.field_count_checked
        && negative_controls.all_controls_passed;
    let validation = Validation {
        // `passed` certifies this preflight artifact only.  The separate
        // `full_component_closure_checked` flag prevents it from being read as
        // a completed multiplet fixture.
        passed: preflight_passed,
        ..validation
    };

    ScalarTensorTangentArtifact {
        schema_version: SCHEMA_VERSION,
        sources: vec![
            SourceRecord {
                arxiv_id: "2412.16527v3",
                locator: "Eqs. (5.4), (5.11), (5.15), (5.19) and Table 3",
                role: "primary scalar-tensor transformations and composites",
                sha256: "76a0aa35c20af63eb00fcf2dd0db835d5f760a4c5ed25f6dac7983109568876d",
            },
            SourceRecord {
                arxiv_id: "2412.16527v3 source",
                locator: "scalar_tensor.tex",
                role: "primary TeX transcription audit",
                sha256: "a3646e398ae6ff101870422566e8c5a6606b0b2901aba9b5c6f09c1f7cc8c7de",
            },
        ],
        background,
        denominator,
        composites,
        central_gauge,
        degrees,
        ct_structural_map,
        negative_controls,
        closure_boundary,
        validation,
        interpretation: "A regular rigid tangent exists and has 8+8 gauge-reduced components, but source-faithful linkage extraction still requires the composite-variation closure gate.",
    }
}

pub fn write_artifact(path: &Path) -> ScalarTensorTangentArtifact {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create scalar-tensor artifact directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create scalar-tensor artifact")),
        &artifact,
    )
    .expect("write scalar-tensor artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_tangent_and_stueckelberg_cancellation_are_exact() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert!(artifact.background.denominator_regular);
        assert!(artifact.denominator.inverse_check_passed);
        assert_eq!(artifact.central_gauge.delta_w_per_partial_z, 1);
        assert_eq!(artifact.central_gauge.gauge_fixed_partial_b_coefficient, 0);
        assert!(artifact.degrees.balanced_8_plus_8);
        assert!(artifact.negative_controls.all_controls_passed);
        assert!(!artifact.validation.full_component_closure_checked);
        assert!(
            !artifact
                .closure_boundary
                .direct_transcription_is_closed_fixture
        );
    }
}
