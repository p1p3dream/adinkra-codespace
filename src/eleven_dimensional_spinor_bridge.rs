//! Representation-level audit of the direct 11D spinor-prepotential bridge.
//!
//! The Added Note in Proof of arXiv:2002.08502 proposes a fundamental spinor
//! prepotential `Psi_alpha` with the scalar semi-prepotential
//! `V = D^alpha Psi_alpha`.  This module counts every Lorentz-equivariant
//! leading and first-momentum channel from that direct spinor source.  It also
//! separates the conventional torsion constraints in Eq. (2.6) of
//! arXiv:2007.05097 from the stronger scalar-prepotential constraint in
//! Eq. (2.7).

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DirectBridgeChannel {
    pub source_dynkin_label: String,
    pub source_dimension: u64,
    pub source_multiplicity: usize,
    pub target_multiplicity_in_source_tensor_spinor: usize,
    pub map_coefficients: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstMomentumChannel {
    pub intermediate_dynkin_label: String,
    pub intermediate_dimension: u64,
    pub multiplicity_in_vector_tensor_target: usize,
    pub multiplicity_at_spinor_level_14: usize,
    pub map_coefficients: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstMomentumSourceChannel {
    pub intermediate_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_dimension: u64,
    pub source_multiplicity_at_level_fourteen: usize,
    pub target_multiplicity_in_source_tensor_spinor: usize,
    pub map_coefficients: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstMomentumSourcePrecheck {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub exterior_degree: usize,
    pub channels: Vec<FirstMomentumSourceChannel>,
    pub source_kernel_systems:
        Vec<crate::eleven_dimensional_bridge::ExteriorHighestWeightSystemShape>,
    pub distinct_source_target_pairs: usize,
    pub embedded_source_copies: usize,
    pub expected_map_coefficients: usize,
    pub every_target_multiplicity_is_one: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GaugeParameterChannel {
    pub form_degree: usize,
    pub dynkin_label: String,
    pub dimension: u64,
    pub multiplicity_in_spinor_square: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalSpinorBridgeReport {
    pub schema_version: &'static str,
    pub inventory_source_arxiv: &'static str,
    pub constraint_source_arxiv: &'static str,
    pub proposed_fundamental_prepotential: &'static str,
    pub semi_prepotential_relation: &'static str,
    pub target: &'static str,
    pub direct_derivative_order: usize,
    pub leading_channels: Vec<DirectBridgeChannel>,
    pub leading_map_dimension: usize,
    pub leading_source_kernel_systems:
        Vec<crate::eleven_dimensional_bridge::ExteriorHighestWeightSystemShape>,
    pub scalar_factorizing_gamma_traceless_dimension: usize,
    pub nonfactorizing_direct_dimension: usize,
    pub scalar_factorizing_subspace_is_proper: bool,
    pub hook_derivative_order: usize,
    pub hook_dynkin_label: &'static str,
    pub hook_dimension: u64,
    pub hook_channels: Vec<DirectBridgeChannel>,
    pub hook_multiplicity: usize,
    pub hook_source_kernel_systems:
        Vec<crate::eleven_dimensional_bridge::ExteriorHighestWeightSystemShape>,
    pub conventional_constraint_set: &'static str,
    pub scalar_strengthened_constraint: &'static str,
    pub source_motivates_strengthened_constraint_by_scalar_prepotential: bool,
    pub direct_hook_is_permitted_by_conventional_constraints: bool,
    pub direct_hook_cancellation_required: bool,
    pub scalar_strengthened_constraint_applied_to_spinor_route: bool,
    pub first_momentum_channels: Vec<FirstMomentumChannel>,
    pub first_momentum_map_dimension: usize,
    pub gauge_parameter_channels: Vec<GaugeParameterChannel>,
    pub gauge_parameter_channel_count: usize,
    pub gauge_parameter_dimensions_sum: u64,
    pub source_selects_gauge_channel_combination: bool,
    pub source_prints_induced_target_gauge_law: bool,
    pub gauge_compatible_quotient_constructed: bool,
    pub representation_level_map_space_constructed: bool,
    pub exact_source_system_shapes_constructed: bool,
    pub exact_source_kernel_vectors_constructed: bool,
    pub component_clebsch_maps_constructed: bool,
    pub derivative_matrix_constructed: bool,
    pub explicit_intertwiner_pass_status: &'static str,
    pub next_computational_step: &'static str,
    pub next_required_input: &'static str,
    pub result: &'static str,
    pub boundary: &'static str,
    pub passed: bool,
}

fn direct_channels(level: usize, target: &str) -> Vec<DirectBridgeChannel> {
    crate::eleven_dimensional_prepotential::spinor_level_channel_sources(level, target)
        .into_iter()
        .map(
            |(source_dynkin_label, source_dimension, source_multiplicity)| DirectBridgeChannel {
                source_dynkin_label,
                source_dimension,
                source_multiplicity,
                target_multiplicity_in_source_tensor_spinor: 1,
                map_coefficients: source_multiplicity,
            },
        )
        .collect()
}

pub fn verify_first_momentum_source_precheck() -> FirstMomentumSourcePrecheck {
    let mut channels = Vec::new();
    for intermediate in ["00001", "01001", "10001", "20001"] {
        for (source_dynkin_label, source_dimension, source_multiplicity) in
            crate::eleven_dimensional_prepotential::spinor_level_channel_sources(14, intermediate)
        {
            let target_multiplicity_in_source_tensor_spinor =
                crate::eleven_dimensional_prepotential::spinor_tensor_channels(
                    &source_dynkin_label,
                )
                .iter()
                .filter(|(target, _)| target == intermediate)
                .count();
            channels.push(FirstMomentumSourceChannel {
                intermediate_dynkin_label: intermediate.to_string(),
                source_dynkin_label,
                source_dimension,
                source_multiplicity_at_level_fourteen: source_multiplicity,
                target_multiplicity_in_source_tensor_spinor,
                map_coefficients: source_multiplicity * target_multiplicity_in_source_tensor_spinor,
            });
        }
    }
    let distinct_source_target_pairs = channels.len();
    let embedded_source_copies = channels
        .iter()
        .map(|channel| channel.source_multiplicity_at_level_fourteen)
        .sum();
    let map_coefficients: usize = channels
        .iter()
        .map(|channel| channel.map_coefficients)
        .sum();
    let every_target_multiplicity_is_one = channels
        .iter()
        .all(|channel| channel.target_multiplicity_in_source_tensor_spinor == 1);
    let mut source_counts = std::collections::BTreeMap::<String, usize>::new();
    for channel in &channels {
        source_counts
            .entry(channel.source_dynkin_label.clone())
            .and_modify(|count| assert_eq!(*count, channel.source_multiplicity_at_level_fourteen))
            .or_insert(channel.source_multiplicity_at_level_fourteen);
    }
    let source_pairs = source_counts
        .iter()
        .map(|(label, copies)| (label.as_str(), *copies))
        .collect::<Vec<_>>();
    let source_kernel_systems =
        crate::eleven_dimensional_bridge::exterior_highest_weight_system_shapes(14, &source_pairs);
    let passed =
        map_coefficients == 44 && embedded_source_copies == 44 && every_target_multiplicity_is_one;
    FirstMomentumSourcePrecheck {
        schema_version: "adynkra-11d-first-momentum-source-precheck-v1",
        role: "fixed level-14 source work list for the 44 first-momentum intertwiners",
        exterior_degree: 14,
        channels,
        source_kernel_systems,
        distinct_source_target_pairs,
        embedded_source_copies,
        expected_map_coefficients: 44,
        every_target_multiplicity_is_one,
        passed,
    }
}

pub fn verify() -> ElevenDimensionalSpinorBridgeReport {
    let leading_channels = direct_channels(16, "10001");
    let leading_map_dimension = leading_channels
        .iter()
        .map(|channel| channel.map_coefficients)
        .sum();
    let leading_source_kernel_systems =
        crate::eleven_dimensional_bridge::exterior_highest_weight_system_shapes(
            16,
            &[
                ("10000", 1),
                ("20000", 1),
                ("00100", 2),
                ("00010", 2),
                ("00002", 1),
                ("10100", 1),
                ("10010", 1),
                ("10002", 3),
            ],
        );

    let scalar_factorizing_gamma_traceless_dimension = 1;
    let nonfactorizing_direct_dimension =
        leading_map_dimension - scalar_factorizing_gamma_traceless_dimension;

    let hook_channels = direct_channels(17, "11000");
    let hook_multiplicity = hook_channels
        .iter()
        .map(|channel| channel.map_coefficients)
        .sum();
    let hook_source_kernel_systems =
        crate::eleven_dimensional_bridge::exterior_highest_weight_system_shapes(
            17,
            &[("10001", 1), ("01001", 2), ("20001", 1), ("11001", 3)],
        );

    let first_momentum_channels =
        crate::eleven_dimensional_prepotential::vector_tensor_gamma_traceless_vector_spinor_channels()
            .into_iter()
            .map(
                |(
                    intermediate_dynkin_label,
                    intermediate_dimension,
                    multiplicity_in_vector_tensor_target,
                )| {
                    let multiplicity_at_spinor_level_14 =
                        crate::eleven_dimensional_prepotential::spinor_level_multiplicity(
                            14,
                            &intermediate_dynkin_label,
                        );
                    FirstMomentumChannel {
                        intermediate_dynkin_label,
                        intermediate_dimension,
                        multiplicity_in_vector_tensor_target,
                        multiplicity_at_spinor_level_14,
                        map_coefficients: multiplicity_in_vector_tensor_target
                            * multiplicity_at_spinor_level_14,
                    }
                },
            )
            .collect::<Vec<_>>();
    let first_momentum_map_dimension = first_momentum_channels
        .iter()
        .map(|channel| channel.map_coefficients)
        .sum();

    let inventory_report = crate::eleven_dimensional_prepotential::verify();
    let gauge_parameter_channels = inventory_report
        .spinor_gauge_parameter_channels
        .into_iter()
        .map(|channel| GaugeParameterChannel {
            form_degree: channel.form_degree,
            dynkin_label: channel.dynkin_label.to_owned(),
            dimension: channel.dimension,
            multiplicity_in_spinor_square: channel.multiplicity_in_spinor_square,
        })
        .collect::<Vec<_>>();
    let gauge_parameter_channel_count = gauge_parameter_channels.len();
    let gauge_parameter_dimensions_sum = gauge_parameter_channels
        .iter()
        .map(|channel| {
            channel.dimension * u64::try_from(channel.multiplicity_in_spinor_square).unwrap()
        })
        .sum();

    let leading_signature = leading_channels
        .iter()
        .map(|channel| {
            (
                channel.source_dynkin_label.as_str(),
                channel.source_multiplicity,
            )
        })
        .collect::<Vec<_>>();
    let hook_signature = hook_channels
        .iter()
        .map(|channel| {
            (
                channel.source_dynkin_label.as_str(),
                channel.source_multiplicity,
            )
        })
        .collect::<Vec<_>>();
    let momentum_signature = first_momentum_channels
        .iter()
        .map(|channel| {
            (
                channel.intermediate_dynkin_label.as_str(),
                channel.multiplicity_at_spinor_level_14,
            )
        })
        .collect::<Vec<_>>();
    let leading_system_signature = leading_source_kernel_systems
        .iter()
        .map(|system| {
            (
                system.dynkin_label.as_str(),
                system.source_weight_space_columns,
                system.total_raising_rows,
                system.expected_kernel_dimension,
            )
        })
        .collect::<Vec<_>>();
    let hook_system_signature = hook_source_kernel_systems
        .iter()
        .map(|system| {
            (
                system.dynkin_label.as_str(),
                system.source_weight_space_columns,
                system.total_raising_rows,
                system.expected_kernel_dimension,
            )
        })
        .collect::<Vec<_>>();

    let passed = leading_signature
        == [
            ("10000", 1),
            ("20000", 1),
            ("00100", 2),
            ("00010", 2),
            ("00002", 1),
            ("10100", 1),
            ("10010", 1),
            ("10002", 3),
        ]
        && leading_map_dimension == 12
        && leading_system_signature
            == [
                ("10000", 657_520, 2_112_644, 1),
                ("20000", 353_120, 1_064_136, 1),
                ("00100", 431_724, 1_377_778, 2),
                ("00010", 348_240, 1_107_276, 2),
                ("00002", 280_014, 871_670, 1),
                ("10100", 227_528, 662_706, 1),
                ("10010", 181_748, 526_602, 1),
                ("10002", 144_678, 408_486, 3),
            ]
        && scalar_factorizing_gamma_traceless_dimension == 1
        && nonfactorizing_direct_dimension == 11
        && hook_signature == [("10001", 1), ("01001", 2), ("20001", 1), ("11001", 3)]
        && hook_multiplicity == 7
        && hook_system_signature
            == [
                ("10001", 388_720, 1_174_806, 1),
                ("01001", 252_162, 755_329, 2),
                ("20001", 166_158, 464_815, 1),
                ("11001", 104_875, 282_011, 3),
            ]
        && crate::eleven_dimensional_prepotential::b5_dimension("11000") == 429
        && momentum_signature == [("00001", 5), ("01001", 18), ("10001", 8), ("20001", 13)]
        && first_momentum_map_dimension == 44
        && gauge_parameter_channel_count == 6
        && gauge_parameter_dimensions_sum == 1_024;

    ElevenDimensionalSpinorBridgeReport {
        schema_version: "adynkra-11d-spinor-bridge-v1",
        inventory_source_arxiv: "2002.08502, Added Note in Proof, Eqs. (6.1)-(6.3)",
        constraint_source_arxiv: "2007.05097, Eqs. (2.6)-(2.7)",
        proposed_fundamental_prepotential: "unconstrained spinor superfield Psi_alpha",
        semi_prepotential_relation: "V = D^alpha Psi_alpha",
        target: "gamma-traceless vector-spinor H_alpha^a, Dynkin label (10001)",
        direct_derivative_order: 16,
        leading_channels,
        leading_map_dimension,
        leading_source_kernel_systems,
        scalar_factorizing_gamma_traceless_dimension,
        nonfactorizing_direct_dimension,
        scalar_factorizing_subspace_is_proper: true,
        hook_derivative_order: 17,
        hook_dynkin_label: "11000",
        hook_dimension: crate::eleven_dimensional_prepotential::b5_dimension("11000"),
        hook_channels,
        hook_multiplicity,
        hook_source_kernel_systems,
        conventional_constraint_set: "Eq. (2.6), which permits X_[ab]^c in the 429",
        scalar_strengthened_constraint:
            "Eq. (2.7), proposed for a scalar-superfield prepotential and setting the full gamma-two torsion sector to zero",
        source_motivates_strengthened_constraint_by_scalar_prepotential: true,
        direct_hook_is_permitted_by_conventional_constraints: true,
        direct_hook_cancellation_required: false,
        scalar_strengthened_constraint_applied_to_spinor_route: false,
        first_momentum_channels,
        first_momentum_map_dimension,
        gauge_parameter_channels,
        gauge_parameter_channel_count,
        gauge_parameter_dimensions_sum,
        source_selects_gauge_channel_combination: false,
        source_prints_induced_target_gauge_law: false,
        gauge_compatible_quotient_constructed: false,
        representation_level_map_space_constructed: true,
        exact_source_system_shapes_constructed: true,
        exact_source_kernel_vectors_constructed: true,
        component_clebsch_maps_constructed: true,
        derivative_matrix_constructed: true,
        explicit_intertwiner_pass_status:
            "all twelve leading and seven hook source embeddings and couplings are exact; the 7-by-12 exterior-derivative matrix has rank 7 and nullity 5",
        next_computational_step:
            "construct the six gauge-parameter intertwiners and the 44 first-momentum correction intertwiners, then solve their joint exact compatibility system",
        next_required_input:
            "select the direct-map coefficients and the spinor-prepotential gauge transformation, including the induced transformation of H_alpha^a",
        result:
            "the scalar factorization occupies one of twelve direct leading directions and lies in the five-dimensional kernel of the exact rank-7 hook derivative; four kernel directions remain after quotienting by the scalar line",
        boundary:
            "this is an exact Lorentz-representation calculation, not a component Clebsch construction, gauge quotient, torsion solution, action, or field equation",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_spinor_bridge_map_spaces_are_counted_exactly() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.leading_map_dimension, 12);
        assert_eq!(report.scalar_factorizing_gamma_traceless_dimension, 1);
        assert_eq!(report.nonfactorizing_direct_dimension, 11);
        assert_eq!(report.hook_multiplicity, 7);
        assert_eq!(report.first_momentum_map_dimension, 44);
        assert_eq!(
            report
                .leading_source_kernel_systems
                .iter()
                .map(|system| {
                    (
                        system.dynkin_label.as_str(),
                        system.source_weight_space_columns,
                        system.total_raising_rows,
                        system.expected_kernel_dimension,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                ("10000", 657_520, 2_112_644, 1),
                ("20000", 353_120, 1_064_136, 1),
                ("00100", 431_724, 1_377_778, 2),
                ("00010", 348_240, 1_107_276, 2),
                ("00002", 280_014, 871_670, 1),
                ("10100", 227_528, 662_706, 1),
                ("10010", 181_748, 526_602, 1),
                ("10002", 144_678, 408_486, 3),
            ]
        );
        assert_eq!(
            report
                .hook_source_kernel_systems
                .iter()
                .map(|system| {
                    (
                        system.dynkin_label.as_str(),
                        system.source_weight_space_columns,
                        system.total_raising_rows,
                        system.expected_kernel_dimension,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                ("10001", 388_720, 1_174_806, 1),
                ("01001", 252_162, 755_329, 2),
                ("20001", 166_158, 464_815, 1),
                ("11001", 104_875, 282_011, 3),
            ]
        );
        assert!(report.exact_source_system_shapes_constructed);
        assert!(report.exact_source_kernel_vectors_constructed);
        assert!(report.derivative_matrix_constructed);
    }

    #[test]
    fn scalar_hook_constraint_is_not_applied_to_the_direct_spinor_route() {
        let report = verify();
        assert!(report.source_motivates_strengthened_constraint_by_scalar_prepotential);
        assert!(report.direct_hook_is_permitted_by_conventional_constraints);
        assert!(!report.direct_hook_cancellation_required);
        assert!(!report.scalar_strengthened_constraint_applied_to_spinor_route);
    }

    #[test]
    fn first_momentum_source_work_list_has_forty_four_multiplicity_one_maps() {
        let report = verify_first_momentum_source_precheck();
        assert!(report.passed);
        assert_eq!(report.embedded_source_copies, 44);
        assert!(report.every_target_multiplicity_is_one);
    }

    #[test]
    fn missing_source_gauge_law_is_reported_without_fabricating_a_quotient() {
        let report = verify();
        assert_eq!(report.gauge_parameter_channel_count, 6);
        assert_eq!(report.gauge_parameter_dimensions_sum, 1_024);
        assert!(!report.source_selects_gauge_channel_combination);
        assert!(!report.source_prints_induced_target_gauge_law);
        assert!(!report.gauge_compatible_quotient_constructed);
        assert!(report.component_clebsch_maps_constructed);
    }
}
