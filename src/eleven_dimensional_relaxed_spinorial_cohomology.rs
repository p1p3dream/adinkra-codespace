//! Exact representation and reduced physical-sector gates for 11D
//! spinorial deformation cohomology.
//!
//! The relaxed dimension-zero torsion contains `X_[2]` in `(11000)` and
//! `X_[5]` in `(10002)`.  Its purely spinorial Bianchi projection lands in
//! `(11001) + (10003)`.  This module certifies every multiplicity-one B5
//! incidence in that statement, but it does not invent the missing relative
//! tensor coefficients of the covariant differential.
//!
//! For `H_F^(0,4)(phys)`, Tsimpis gives a complete reduced physical-monomial
//! analysis through order `l^3`: one exact redefinition direction enters a
//! three-dimensional candidate space, one independent closure condition
//! leaves a two-dimensional kernel, and the quotient has dimension one.  We
//! execute the basis-independent rational normal form of those ranks.  The
//! normal form does not claim to reconstruct the unprinted physical-basis
//! coefficients.

use num_rational::Ratio;
use serde::Serialize;
#[cfg(test)]
use std::fs;

type Rational = Ratio<i64>;

fn r(value: i64) -> Rational {
    Ratio::from_integer(value)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RepresentationTerm {
    pub dynkin_label: String,
    pub dimension: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SpinorialIncidence {
    pub source_dynkin_label: String,
    pub target_dynkin_label: String,
    pub target_dimension: u64,
    pub multiplicity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrimarySourceBoundary {
    pub source: &'static str,
    pub source_archive_sha256: &'static str,
    pub statement_used: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReducedExactComplex {
    pub domain_dimension: usize,
    pub middle_dimension: usize,
    pub target_dimension: usize,
    pub incoming_canonical_matrix: Vec<Vec<i64>>,
    pub outgoing_canonical_matrix: Vec<Vec<i64>>,
    pub incoming_rank: usize,
    pub incoming_nullity: usize,
    pub outgoing_rank: usize,
    pub outgoing_nullity: usize,
    pub composition_residual_entries: usize,
    pub middle_kernel_dimension: usize,
    pub middle_image_dimension: usize,
    pub middle_cohomology_dimension: usize,
    pub canonical_basis_only: bool,
    pub physical_basis_coefficients_reconstructed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RelaxedSpinorialCohomologyReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub primary_sources: Vec<PrimarySourceBoundary>,
    pub structure_group: &'static str,
    pub relaxed_torsion_sources: Vec<RepresentationTerm>,
    pub relaxed_torsion_total_dimension: u64,
    pub relaxed_torsion_first_jet_dimension: u64,
    pub relaxed_bianchi_targets: Vec<RepresentationTerm>,
    pub relaxed_bianchi_target_total_dimension: u64,
    pub relaxed_torsion_projected_incidences: Vec<SpinorialIncidence>,
    pub relaxed_torsion_projected_incidence_count: usize,
    pub relaxed_torsion_spinor_products_dimensionally_complete: bool,
    pub relaxed_torsion_relative_coefficients_source_fixed: bool,
    pub relaxed_torsion_component_d_f_constructed: bool,
    pub relaxed_torsion_kernel_computed: bool,
    pub h_tau_zero_three_terms: Vec<RepresentationTerm>,
    pub h_tau_zero_three_dimension: u64,
    pub h_tau_zero_four_terms: Vec<RepresentationTerm>,
    pub h_tau_zero_four_dimension: u64,
    pub h_tau_zero_five_terms: Vec<RepresentationTerm>,
    pub h_tau_zero_five_dimension: u64,
    pub h_tau_zero_three_to_four_incidences: Vec<SpinorialIncidence>,
    pub h_tau_zero_four_to_five_incidences: Vec<SpinorialIncidence>,
    pub h_tau_representation_quotient_inventory_certified: bool,
    pub tau_zero_component_quotient_maps_constructed: bool,
    pub order_l_one_physical_cohomology_dimension: usize,
    pub order_l_two_physical_cohomology_dimension: usize,
    pub order_l_three_reduced_complex: ReducedExactComplex,
    pub order_l_three_h_f_zero_four_physical_dimension: usize,
    pub order_l_three_generator_interpretation: &'static str,
    pub full_h_f_zero_four_physical_computed: bool,
    pub full_torsion_bianchi_tower_computed: bool,
    pub finite_auxiliary_off_shell_closure_computed: bool,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

fn term(label: &str) -> RepresentationTerm {
    RepresentationTerm {
        dynkin_label: label.to_string(),
        dimension: crate::eleven_dimensional_prepotential::b5_dimension(label),
    }
}

fn incidences(sources: &[&str], targets: &[&str]) -> Vec<SpinorialIncidence> {
    let mut result = Vec::new();
    for source in sources {
        let product = crate::eleven_dimensional_prepotential::spinor_tensor_channels(source);
        for target in targets {
            let matches = product
                .iter()
                .filter(|(label, _)| label == target)
                .collect::<Vec<_>>();
            if matches.is_empty() {
                continue;
            }
            assert_eq!(matches.len(), 1, "spinor product is not multiplicity one");
            result.push(SpinorialIncidence {
                source_dynkin_label: (*source).to_string(),
                target_dynkin_label: (*target).to_string(),
                target_dimension: matches[0].1,
                multiplicity: 1,
            });
        }
    }
    result
}

fn spinor_product_dimension_complete(source: &str) -> bool {
    let source_dimension = crate::eleven_dimensional_prepotential::b5_dimension(source);
    crate::eleven_dimensional_prepotential::spinor_tensor_channels(source)
        .iter()
        .map(|(_, dimension)| *dimension)
        .sum::<u64>()
        == 32 * source_dimension
}

fn exact_rank(matrix: &[Vec<Rational>], columns: usize) -> usize {
    let mut reduced = matrix.to_vec();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..reduced.len()).find(|&row| reduced[row][column] != r(0))
        else {
            continue;
        };
        reduced.swap(pivot_row, found);
        let pivot = reduced[pivot_row][column].clone();
        for entry in &mut reduced[pivot_row] {
            *entry /= pivot.clone();
        }
        let normalized = reduced[pivot_row].clone();
        for row in 0..reduced.len() {
            if row == pivot_row || reduced[row][column] == r(0) {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..columns {
                reduced[row][index] -= factor.clone() * normalized[index].clone();
            }
        }
        pivot_row += 1;
        if pivot_row == reduced.len() {
            break;
        }
    }
    pivot_row
}

fn reduced_order_l_three_complex() -> ReducedExactComplex {
    // Any rank-one 1 -> 3 map followed by a rank-one 3 -> 1 map with zero
    // composition has this rational normal form after independent basis
    // changes preserving the complex.
    let incoming_integer = vec![vec![1], vec![0], vec![0]];
    let outgoing_integer = vec![vec![0, 1, 0]];
    let incoming = incoming_integer
        .iter()
        .map(|row| row.iter().map(|value| r(*value)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let outgoing = outgoing_integer
        .iter()
        .map(|row| row.iter().map(|value| r(*value)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let incoming_rank = exact_rank(&incoming, 1);
    let outgoing_rank = exact_rank(&outgoing, 3);
    let composition_residual_entries = (0..1)
        .filter(|&column| {
            (0..3)
                .map(|pivot| outgoing[0][pivot].clone() * incoming[pivot][column].clone())
                .fold(r(0), |sum, value| sum + value)
                != r(0)
        })
        .count();
    let middle_kernel_dimension = 3 - outgoing_rank;
    let middle_image_dimension = incoming_rank;
    ReducedExactComplex {
        domain_dimension: 1,
        middle_dimension: 3,
        target_dimension: 1,
        incoming_canonical_matrix: incoming_integer,
        outgoing_canonical_matrix: outgoing_integer,
        incoming_rank,
        incoming_nullity: 1 - incoming_rank,
        outgoing_rank,
        outgoing_nullity: middle_kernel_dimension,
        composition_residual_entries,
        middle_kernel_dimension,
        middle_image_dimension,
        middle_cohomology_dimension: middle_kernel_dimension - middle_image_dimension,
        canonical_basis_only: true,
        physical_basis_coefficients_reconstructed: false,
    }
}

fn primary_sources() -> Vec<PrimarySourceBoundary> {
    vec![
        PrimarySourceBoundary {
            source: "Howe, arXiv:hep-th/9707184",
            source_archive_sha256: "2aa95c8072e75c6d6b2592f4880a8dbbd843a0036f7805613d911b99608e9482",
            statement_used: "the standard dimension-zero torsion constraint plus Bianchi identities implies the 11D supergravity equations of motion",
        },
        PrimarySourceBoundary {
            source: "Cederwall, Nilsson, Tsimpis, arXiv:hep-th/0110069",
            source_archive_sha256: "5d232edd240298530f8216a351e282f4261a0cfe3ec4cfa02c1eaa055f8e43ea",
            statement_used: "after conventional constraints the relaxed torsion fields are (11000)+(10002), with projected spinorial Bianchi representations (11001)+(10003)",
        },
        PrimarySourceBoundary {
            source: "Tsimpis, arXiv:hep-th/0407271",
            source_archive_sha256: "013049a4da503cdee37a67447ab5adc17a30678e07430ecd43905fb0bdc47ee3",
            statement_used: "H_F^(0,4)(phys) controls deformations; it is trivial at orders l and l^2 and one-dimensional at order l^3 after one field redefinition and one closure condition",
        },
    ]
}

pub fn verify() -> RelaxedSpinorialCohomologyReport {
    let relaxed_sources = vec![term("11000"), term("10002")];
    let relaxed_targets = vec![term("11001"), term("10003")];
    let relaxed_incidences = incidences(&["11000", "10002"], &["11001", "10003"]);

    let h3 = vec![term("01001"), term("00003")];
    let h4 = vec![term("02000"), term("01002"), term("00004")];
    let h5 = vec![term("02001"), term("01003"), term("00005")];
    let h3_h4 = incidences(&["01001", "00003"], &["02000", "01002", "00004"]);
    let h4_h5 = incidences(&["02000", "01002", "00004"], &["02001", "01003", "00005"]);
    let reduced = reduced_order_l_three_complex();
    let relaxed_products_complete = ["11000", "10002"]
        .iter()
        .all(|source| spinor_product_dimension_complete(source));
    let quotient_products_complete = ["01001", "00003", "02000", "01002", "00004"]
        .iter()
        .all(|source| spinor_product_dimension_complete(source));
    let representation_inventory_certified =
        h3_h4.len() == 4 && h4_h5.len() == 5 && quotient_products_complete;
    let passed = relaxed_sources
        .iter()
        .map(|item| item.dimension)
        .sum::<u64>()
        == 4_719
        && relaxed_targets
            .iter()
            .map(|item| item.dimension)
            .sum::<u64>()
            == 47_200
        && relaxed_incidences.len() == 3
        && relaxed_products_complete
        && h3.iter().map(|item| item.dimension).sum::<u64>() == 5_632
        && h4.iter().map(|item| item.dimension).sum::<u64>() == 46_618
        && h5.iter().map(|item| item.dimension).sum::<u64>() == 313_248
        && representation_inventory_certified
        && reduced.incoming_rank == 1
        && reduced.outgoing_rank == 1
        && reduced.composition_residual_entries == 0
        && reduced.middle_kernel_dimension == 2
        && reduced.middle_cohomology_dimension == 1;

    RelaxedSpinorialCohomologyReport {
        schema_version: "adynkra-11d-relaxed-spinorial-cohomology-v1",
        role: "exact relaxed-torsion incidence gate and source-backed reduced H_F^(0,4)(phys) rank certificate",
        primary_sources: primary_sources(),
        structure_group: "Spin(1,10), with compact B5 labels used only for finite-dimensional representation bookkeeping",
        relaxed_torsion_total_dimension: relaxed_sources.iter().map(|item| item.dimension).sum(),
        relaxed_torsion_first_jet_dimension: 32
            * relaxed_sources
                .iter()
                .map(|item| item.dimension)
                .sum::<u64>(),
        relaxed_torsion_sources: relaxed_sources,
        relaxed_bianchi_target_total_dimension: relaxed_targets
            .iter()
            .map(|item| item.dimension)
            .sum(),
        relaxed_bianchi_targets: relaxed_targets,
        relaxed_torsion_projected_incidence_count: relaxed_incidences.len(),
        relaxed_torsion_projected_incidences: relaxed_incidences,
        relaxed_torsion_spinor_products_dimensionally_complete: relaxed_products_complete,
        relaxed_torsion_relative_coefficients_source_fixed: false,
        relaxed_torsion_component_d_f_constructed: false,
        relaxed_torsion_kernel_computed: false,
        h_tau_zero_three_dimension: h3.iter().map(|item| item.dimension).sum(),
        h_tau_zero_three_terms: h3,
        h_tau_zero_four_dimension: h4.iter().map(|item| item.dimension).sum(),
        h_tau_zero_four_terms: h4,
        h_tau_zero_five_dimension: h5.iter().map(|item| item.dimension).sum(),
        h_tau_zero_five_terms: h5,
        h_tau_zero_three_to_four_incidences: h3_h4,
        h_tau_zero_four_to_five_incidences: h4_h5,
        h_tau_representation_quotient_inventory_certified: representation_inventory_certified,
        tau_zero_component_quotient_maps_constructed: false,
        order_l_one_physical_cohomology_dimension: 0,
        order_l_two_physical_cohomology_dimension: 0,
        order_l_three_h_f_zero_four_physical_dimension: reduced.middle_cohomology_dimension,
        order_l_three_reduced_complex: reduced,
        order_l_three_generator_interpretation: "the source identifies one class represented by a fixed combination of G^2 terms, equivalently the local tr(R^2) shift with a global p_1(M) distinction",
        full_h_f_zero_four_physical_computed: false,
        full_torsion_bianchi_tower_computed: false,
        finite_auxiliary_off_shell_closure_computed: false,
        passed,
        result: "The relaxed-torsion and H_tau representation worklists are exact. On Tsimpis's order-l^3 physical-monomial subspace, ker(d_F) has dimension two, im(d_F) has dimension one, and H_F^(0,4)(phys) has dimension one.",
        boundary: "The order-l^3 matrices are a canonical rational normal form of source-established ranks, not the physical A/B/C coefficient matrix. The missing relaxed-torsion Clebsch coefficients, tau_0 component quotient maps, higher Bianchi identities, and unrestricted physical-field coefficient complex prevent a full cohomology or finite-auxiliary off-shell claim.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_torsion_incidence_is_exact_but_component_differential_fails_closed() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.relaxed_torsion_total_dimension, 4_719);
        assert_eq!(report.relaxed_torsion_first_jet_dimension, 151_008);
        assert_eq!(report.relaxed_bianchi_target_total_dimension, 47_200);
        assert_eq!(report.relaxed_torsion_projected_incidence_count, 3);
        assert!(!report.relaxed_torsion_relative_coefficients_source_fixed);
        assert!(!report.relaxed_torsion_component_d_f_constructed);
        assert!(!report.relaxed_torsion_kernel_computed);
    }

    #[test]
    fn h_tau_representation_inventories_and_incidences_match_the_source_complex() {
        let report = verify();
        assert_eq!(report.h_tau_zero_three_dimension, 5_632);
        assert_eq!(report.h_tau_zero_four_dimension, 46_618);
        assert_eq!(report.h_tau_zero_five_dimension, 313_248);
        assert_eq!(report.h_tau_zero_three_to_four_incidences.len(), 4);
        assert_eq!(report.h_tau_zero_four_to_five_incidences.len(), 5);
        assert!(report.h_tau_representation_quotient_inventory_certified);
        assert!(!report.tau_zero_component_quotient_maps_constructed);
    }

    #[test]
    fn order_l_three_reduced_physical_cohomology_is_one_dimensional() {
        let report = verify();
        let complex = report.order_l_three_reduced_complex;
        assert_eq!(complex.incoming_rank, 1);
        assert_eq!(complex.outgoing_rank, 1);
        assert_eq!(complex.composition_residual_entries, 0);
        assert_eq!(complex.middle_kernel_dimension, 2);
        assert_eq!(complex.middle_image_dimension, 1);
        assert_eq!(complex.middle_cohomology_dimension, 1);
        assert!(complex.canonical_basis_only);
        assert!(!complex.physical_basis_coefficients_reconstructed);
    }

    #[test]
    fn deformation_cohomology_is_not_promoted_to_off_shell_closure() {
        let report = verify();
        assert_eq!(report.order_l_three_h_f_zero_four_physical_dimension, 1);
        assert!(!report.full_h_f_zero_four_physical_computed);
        assert!(!report.full_torsion_bianchi_tower_computed);
        assert!(!report.finite_auxiliary_off_shell_closure_computed);
    }

    #[test]
    #[ignore = "writes the committed relaxed spinorial cohomology artifact"]
    fn write_artifact() {
        let report = verify();
        assert!(report.passed);
        let path = "results/adynkra_11d_relaxed_spinorial_cohomology.json";
        let temporary = format!("{path}.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        fs::rename(temporary, path).unwrap();
    }
}
