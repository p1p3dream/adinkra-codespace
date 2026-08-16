//! Work-list certificate for the 11D prepotential gauge-curvature kill gate.
//!
//! The current exact engine constructs the six candidate source gauge maps and
//! the 12 leading plus 44 first-momentum maps into the `(10001)` target.  It
//! does not yet contain a convention-fixed target curvature operator.  This
//! module records that boundary without treating source invariance as target
//! gauge covariance.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use serde::Serialize;

const SOURCE_CHANNELS: [(usize, &str, usize); 6] = [
    (0, "00000", 1),
    (1, "10000", 11),
    (2, "01000", 55),
    (3, "00100", 165),
    (4, "00010", 330),
    (5, "00002", 462),
];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SourceGaugeChannel {
    pub form_degree: usize,
    pub parameter_dynkin_label: &'static str,
    pub parameter_dimension: usize,
    pub map: &'static str,
    pub independent_parameter_domain: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PhysicalComponentComplex {
    pub potential: &'static str,
    pub gauge_law: &'static str,
    pub curvature: &'static str,
    pub curvature_invariance_identity: &'static str,
    pub bianchi_identity: &'static str,
    pub usable_as_current_superfield_target_operator: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TargetOperatorRequirement {
    pub symbol: &'static str,
    pub required_domain: &'static str,
    pub required_codomain: &'static str,
    pub required_exact_identity: &'static str,
    pub required_kernel_certificate: &'static str,
    pub convention_gate: &'static str,
    pub currently_available: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CurvatureWorkItem {
    pub ordinal: usize,
    pub gauge_form_degree: usize,
    pub parameter_dynkin_label: &'static str,
    pub parameter_dimension: usize,
    pub operator_ordinal: usize,
    pub operator_label: String,
    pub operator_kind: String,
    pub source_composition_zero_momentum_bidegree: Option<&'static str>,
    pub source_composition_first_momentum_bidegree: &'static str,
    pub curvature_output_bidegrees_if_f_is_algebraic: Vec<&'static str>,
    pub curvature_output_bidegrees_if_f_is_one_momentum: Vec<&'static str>,
    pub blocked_by_missing_target_operator: bool,
    pub blocked_by_missing_target_resolved_stream: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalPrepotentialGateReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_prepotential: &'static str,
    pub target_superfield: &'static str,
    pub target_superfield_local_gamma_trace_law: &'static str,
    pub target_gamma_trace_quotient: &'static str,
    pub source_channels: Vec<SourceGaugeChannel>,
    pub source_channel_count: usize,
    pub source_parameter_dimensions_sum: usize,
    pub leading_operator_count: usize,
    pub first_momentum_operator_count: usize,
    pub operator_count: usize,
    pub work_items: Vec<CurvatureWorkItem>,
    pub work_item_count: usize,
    pub zero_momentum_source_composition_jobs: usize,
    pub first_momentum_source_composition_jobs: usize,
    pub physical_component_complexes: Vec<PhysicalComponentComplex>,
    pub target_operator_requirement: TargetOperatorRequirement,
    pub source_channel_coefficients_can_cancel_across_irreps: bool,
    pub source_channel_support_rule: &'static str,
    pub current_composition_stream_boundary: &'static str,
    pub minimal_missing_scientific_input: &'static str,
    pub minimal_missing_code_api: &'static str,
    pub exact_execution_sequence: Vec<&'static str>,
    pub worklist_consistent_with_current_exact_engine: bool,
    pub physical_kill_gate_executed: bool,
    pub physical_kill_gate_passed: Option<bool>,
    pub status: &'static str,
    pub boundary: &'static str,
}

fn source_channels() -> Vec<SourceGaugeChannel> {
    SOURCE_CHANNELS
        .into_iter()
        .map(
            |(form_degree, parameter_dynkin_label, parameter_dimension)| SourceGaugeChannel {
                form_degree,
                parameter_dynkin_label,
                parameter_dimension,
                map: "G_p: Lambda_[p] -> Psi_alpha, delta Psi_alpha = (Gamma^[p])_alpha^beta D_beta Lambda_[p]",
                independent_parameter_domain: true,
            },
        )
        .collect()
}

fn physical_component_complexes() -> Vec<PhysicalComponentComplex> {
    vec![
        PhysicalComponentComplex {
            potential: "linearized graviton h_ab",
            gauge_law: "delta h_ab = 2 p_(a xi_b)",
            curvature: "R_abcd = 2 p_[a p_[c h_b]d]",
            curvature_invariance_identity: "R(delta h) = 0 by p_a p_b = p_b p_a",
            bianchi_identity: "p_[e R_ab]cd = 0",
            usable_as_current_superfield_target_operator: false,
        },
        PhysicalComponentComplex {
            potential: "three-form C_abc",
            gauge_law: "delta C_abc = 3 p_[a Lambda_bc]",
            curvature: "G_abcd = 4 p_[a C_bcd]",
            curvature_invariance_identity: "G(delta C) = 0 by p_[a p_b] = 0",
            bianchi_identity: "p_[a G_bcde] = 0",
            usable_as_current_superfield_target_operator: false,
        },
        PhysicalComponentComplex {
            potential: "unprojected linearized gravitino psi_a^alpha",
            gauge_law: "delta psi_a = p_a epsilon",
            curvature: "rho_ab = 2 p_[a psi_b]",
            curvature_invariance_identity: "rho(delta psi) = 2 p_[a p_b] epsilon = 0",
            bianchi_identity: "p_[a rho_bc] = 0",
            usable_as_current_superfield_target_operator: false,
        },
        PhysicalComponentComplex {
            potential: "unprojected linearized gravitino psi_a^alpha",
            gauge_law: "delta psi_a = p_a epsilon",
            curvature: "E^a = Gamma^{abc} p_b psi_c",
            curvature_invariance_identity: "E(delta psi) = Gamma^{abc} p_b p_c epsilon = 0",
            bianchi_identity: "not a curvature Bianchi identity; E is the Rarita-Schwinger equation operator",
            usable_as_current_superfield_target_operator: false,
        },
    ]
}

pub fn verify() -> ElevenDimensionalPrepotentialGateReport {
    let operators = crate::eleven_dimensional_level16_couplings::joint_column_specs();
    let source_specs = crate::eleven_dimensional_gauge::gauge_composition_specs();
    let source_channels = source_channels();
    let leading_operator_count = operators
        .iter()
        .filter(|operator| operator.kind == "leading")
        .count();
    let first_momentum_operator_count = operators
        .iter()
        .filter(|operator| operator.kind == "first-momentum")
        .count();

    let work_items = source_specs
        .iter()
        .map(|spec| {
            let parameter_dimension = SOURCE_CHANNELS[spec.gauge_form_degree].2;
            CurvatureWorkItem {
                ordinal: spec.ordinal,
                gauge_form_degree: spec.gauge_form_degree,
                parameter_dynkin_label: spec.parameter_dynkin_label,
                parameter_dimension,
                operator_ordinal: spec.operator_ordinal,
                operator_label: spec.operator_label.clone(),
                operator_kind: spec.operator_kind.clone(),
                source_composition_zero_momentum_bidegree: spec
                    .contributes_zero_momentum_d17
                    .then_some("D^17 Lambda"),
                source_composition_first_momentum_bidegree: "p D^15 Lambda",
                curvature_output_bidegrees_if_f_is_algebraic: if spec.contributes_zero_momentum_d17
                {
                    vec!["D^17 Lambda", "p D^15 Lambda"]
                } else {
                    vec!["p D^15 Lambda"]
                },
                curvature_output_bidegrees_if_f_is_one_momentum: if spec
                    .contributes_zero_momentum_d17
                {
                    vec!["p D^17 Lambda", "p^2 D^15 Lambda"]
                } else {
                    vec!["p^2 D^15 Lambda"]
                },
                blocked_by_missing_target_operator: true,
                blocked_by_missing_target_resolved_stream: true,
            }
        })
        .collect::<Vec<_>>();

    let zero_momentum_source_composition_jobs = work_items
        .iter()
        .filter(|item| item.source_composition_zero_momentum_bidegree.is_some())
        .count();
    let first_momentum_source_composition_jobs = work_items.len();
    let source_parameter_dimensions_sum = source_channels
        .iter()
        .map(|channel| channel.parameter_dimension)
        .sum();
    let worklist_consistent_with_current_exact_engine = source_channels.len() == 6
        && source_parameter_dimensions_sum == 1024
        && leading_operator_count == 12
        && first_momentum_operator_count == 44
        && operators.len() == 56
        && work_items.len() == 336
        && zero_momentum_source_composition_jobs == 72
        && first_momentum_source_composition_jobs == 336
        && work_items.iter().enumerate().all(|(ordinal, item)| {
            item.ordinal == ordinal
                && item.parameter_dynkin_label == SOURCE_CHANNELS[item.gauge_form_degree].1
        });

    ElevenDimensionalPrepotentialGateReport {
        schema_version: "adynkra-11d-prepotential-gauge-curvature-worklist-v1",
        role: "exact work-list certificate for F A G_p = 0 on the current 6 by 56 source compositions",
        source_prepotential: "unconstrained spinor superfield Psi_alpha",
        target_superfield: "gamma-traceless conformal-graviton semi-prepotential H_hat_alpha^a in (10001)",
        target_superfield_local_gamma_trace_law: "delta H_beta^b = (Gamma^b)_beta^alpha Lambda_alpha",
        target_gamma_trace_quotient: "H_hat = P_320 H; P_320 Gamma Lambda = 0",
        source_channels,
        source_channel_count: SOURCE_CHANNELS.len(),
        source_parameter_dimensions_sum,
        leading_operator_count,
        first_momentum_operator_count,
        operator_count: operators.len(),
        work_item_count: work_items.len(),
        work_items,
        zero_momentum_source_composition_jobs,
        first_momentum_source_composition_jobs,
        physical_component_complexes: physical_component_complexes(),
        target_operator_requirement: TargetOperatorRequirement {
            symbol: "F",
            required_domain: "the exact 320-component (10001) H_hat superfield, with all D and p normal-form degrees retained",
            required_codomain: "a convention-fixed torsion or curvature module with explicit SO(11) component basis",
            required_exact_identity: "F K = 0 for the independently specified physical target gauge map K",
            required_kernel_certificate: "ker(F) = im(K), or a stated weaker inclusion im(K) subset ker(F), at every tested normal-form bidegree",
            convention_gate: "the ordinary component law delta psi_a = p_a epsilon acts on an unprojected 352-component vector-spinor and cannot be silently substituted for a law on the gamma-traceless 320-component H_hat target",
            currently_available: false,
        },
        source_channel_coefficients_can_cancel_across_irreps: false,
        source_channel_support_rule: "The six Lambda_[p] are independent Lorentz-inequivalent parameter domains. Gauge invariance for arbitrary parameters requires F A G_p = 0 separately for every active p. A sum over c_p does not permit cross-degree cancellation unless a new Lorentz-covariant parameter-identification map is supplied.",
        current_composition_stream_boundary: "The public visitors emit a highest-weight target coefficient indexed by parameter component, exterior mask, and, at first momentum, momentum vector. They do not emit the 11 by 32 target vector-spinor component needed to apply a general component curvature operator.",
        minimal_missing_scientific_input: "A convention-fixed linearized target superfield gauge map K and target torsion or curvature operator F on H_hat, including the exact identity F K = 0 and the intended relation between H_hat components and h_ab, C_abc, and psi_a.",
        minimal_missing_code_api: "A target-resolved exact composition visitor that emits (target_vector_index, target_spinor_index, parameter_component, momentum_monomial, exterior_mask, Gaussian-rational coefficient), followed by an exact sparse F application API.",
        exact_execution_sequence: vec![
            "freeze K and F, their component bases, derivative order, normal-form convention, and the identity F K = 0",
            "certify F K = 0 independently and mutation-test every nonzero block of F",
            "lift all 12 leading and 44 first-momentum maps from the highest-weight output coordinate to the full 320-component target stream",
            "apply F to the 72 D^17 and 336 p D^15 source-composition jobs, retaining any new bidegrees generated by F",
            "assemble exact per-channel matrices because the six parameter domains are independent",
            "solve each active-channel condition F A G_p = 0 for the 56 operator coefficients and report the leading projection rank",
            "check target Bianchi identities and mutation-test every surviving kernel direction",
        ],
        worklist_consistent_with_current_exact_engine,
        physical_kill_gate_executed: false,
        physical_kill_gate_passed: None,
        status: "blocked_missing_target_superfield_gauge_curvature_complex",
        boundary: "This report certifies the source-channel and operator work list only. The component graviton, three-form, and gravitino complexes are physical reference complexes, not an inferred curvature operator on H_hat. No source channel is accepted or rejected by F A G_p = 0 until K, F, and the target-resolved stream exist.",
    }
}

pub fn write_json(output: &Path) -> io::Result<ElevenDimensionalPrepotentialGateReport> {
    let report = verify();
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("adynkra-11d-prepotential-gate.json"),
        std::process::id()
    ));
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, output)?;
    File::open(parent)?.sync_all()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_worklist_matches_the_current_six_by_fifty_six_engine() {
        let report = verify();
        assert!(report.worklist_consistent_with_current_exact_engine);
        assert_eq!(report.source_channel_count, 6);
        assert_eq!(report.source_parameter_dimensions_sum, 1024);
        assert_eq!(report.leading_operator_count, 12);
        assert_eq!(report.first_momentum_operator_count, 44);
        assert_eq!(report.work_item_count, 336);
        assert_eq!(report.zero_momentum_source_composition_jobs, 72);
        assert_eq!(report.first_momentum_source_composition_jobs, 336);
    }

    #[test]
    fn physical_gate_is_not_claimed_without_f() {
        let report = verify();
        assert!(!report.target_operator_requirement.currently_available);
        assert!(!report.physical_kill_gate_executed);
        assert_eq!(report.physical_kill_gate_passed, None);
        assert!(!report.source_channel_coefficients_can_cancel_across_irreps);
    }
}
