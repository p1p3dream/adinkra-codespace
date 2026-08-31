//! Exact component-field gauge complex for free eleven-dimensional supergravity.
//!
//! This module constructs the direct sum of the ordinary component gauge maps
//! `xi_a -> h_ab`, `Lambda_ab -> A_abc`, and `epsilon_alpha -> psi_a^alpha`.
//! Its codomain is the independent component-potential space, not the
//! gamma-traceless semi-prepotential `H_hat`.  In particular, this certificate
//! does not construct the missing prepotential map `K: H_Xi -> H_hat`.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_free_complex::{build as build_free_complex, NULL_MOMENTUM};
use crate::eleven_dimensional_target_equation_complex::{
    target_sector_complex, verify as verify_target_complex, TargetSector,
};

const TARGET_ARTIFACT: &str = "results/adynkra_11d_target_equation_complex.json";
const FREE_ARTIFACT: &str = "results/adynkra_11d_free_complex_validation.json";
const A3_FIBER_ARTIFACT: &str = "results/adynkra_11d_a3_curl_fiber_product.json";
const GRAVITON_RELATIVE_ARTIFACT: &str = "results/adynkra_11d_graviton_gravitino_relative.json";
const GRAVITON_ORACLE_ARTIFACT: &str = "results/adynkra_11d_graviton_relative_oracle.json";

const PINNED_ARTIFACTS: [(&str, &str); 5] = [
    (
        TARGET_ARTIFACT,
        "1aa334d1f2cbcc8a46bf2f915b5aeadf16543131241dcdcdc7257e6252d90092",
    ),
    (
        FREE_ARTIFACT,
        "d151cfc2b086a737aee7c02f85c6b2b77332451506a8a47cbe78d642876c17e0",
    ),
    (
        A3_FIBER_ARTIFACT,
        "53f078a1189555734a9c48f674a0528f620460d0ffe8cd60d461f1533b13558a",
    ),
    (
        GRAVITON_RELATIVE_ARTIFACT,
        "17f9f227491a1f12bc8449a51f19039522bde86e1a3e1de8250cd9fd01bb11a3",
    ),
    (
        GRAVITON_ORACLE_ARTIFACT,
        "b03408ee5e3bf7e7156e47636fe189d47a48dc8a20794c7eefec568cdcc8c789",
    ),
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DependencyBinding {
    pub path: String,
    pub expected_sha256: String,
    pub observed_sha256: String,
    pub passed_field: bool,
    pub matched: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FormalSectorCertificate {
    pub sector: String,
    pub gauge_dimensions: (usize, usize),
    pub curvature_dimensions: (usize, usize),
    pub bianchi_dimensions: (usize, usize),
    pub euler_dimensions: (usize, usize),
    pub curvature_after_gauge_residual_terms: usize,
    pub bianchi_after_curvature_residual_terms: usize,
    pub euler_factorization_residual_terms: usize,
    pub euler_after_gauge_residual_terms: usize,
    pub noether_after_euler_residual_terms: usize,
    pub mutation_residual_terms: usize,
    pub fk_gauge_term_mutation_potential_column: usize,
    pub fk_gauge_term_mutation_residual_terms: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NullCohomologyCertificate {
    pub sector: String,
    pub potential_dimension: usize,
    pub gauge_parameter_dimension: usize,
    pub gauge_rank: usize,
    pub raw_potential_quotient_dimension: usize,
    pub euler_rank: usize,
    pub on_shell_kernel_dimension: usize,
    pub gauge_image_lies_in_euler_kernel: bool,
    pub physical_cohomology_dimension: usize,
    pub expected_physical_cohomology_dimension: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ThreeFormReducibilityCertificate {
    pub chain_dimensions: [(usize, usize); 3],
    pub chain_ranks_at_null_momentum: [usize; 3],
    pub first_composition_residual_entries: usize,
    pub second_composition_residual_entries: usize,
    pub rank_exactness: [bool; 2],
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalComponentKReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub signature: &'static str,
    pub physical_component_k_constructed: bool,
    pub prepotential_k_into_h_hat_constructed: bool,
    pub direct_sum_parameter_basis: [&'static str; 3],
    pub direct_sum_potential_basis: [&'static str; 3],
    pub direct_sum_curvature_basis: [&'static str; 3],
    pub exact_gauge_formulas: [&'static str; 5],
    pub antisymmetrization_convention: &'static str,
    pub formula_source: &'static str,
    pub direct_sum_k_dimensions: (usize, usize),
    pub direct_sum_f_dimensions: (usize, usize),
    pub riemann_row_basis: &'static str,
    pub ambient_riemann_rows: usize,
    pub canonical_riemann_irrep_rows_not_used: usize,
    pub formal_sectors: Vec<FormalSectorCertificate>,
    pub formal_direct_sum_fk_residual_terms: usize,
    pub formal_direct_sum_bianchi_f_residual_terms: usize,
    pub formal_direct_sum_euler_k_residual_terms: usize,
    pub formal_direct_sum_euler_factorization_residual_terms: usize,
    pub null_momentum_covector: [i64; 11],
    pub null_momentum_square: i64,
    pub null_cohomology: Vec<NullCohomologyCertificate>,
    pub raw_potential_quotient_dimensions: [usize; 3],
    pub physical_bosonic_split: [usize; 2],
    pub physical_bosonic_dimension: usize,
    pub physical_fermionic_dimension: usize,
    pub three_form_reducibility: ThreeFormReducibilityCertificate,
    pub dependency_bindings: Vec<DependencyBinding>,
    pub source_sha256: BTreeMap<String, String>,
    pub passed: bool,
    pub scope_boundary: &'static str,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> io::Result<String> {
    fs::read(path).map(|bytes| sha256(&bytes))
}

fn bind_artifact(path: &str, expected_sha256: &str) -> io::Result<DependencyBinding> {
    let bytes = fs::read(path)?;
    let observed_sha256 = sha256(&bytes);
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    let passed_field = value
        .get("passed")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            value
                .get("report")
                .and_then(|report| report.get("passed"))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false);
    Ok(DependencyBinding {
        path: path.to_string(),
        expected_sha256: expected_sha256.to_string(),
        matched: observed_sha256 == expected_sha256 && passed_field,
        observed_sha256,
        passed_field,
    })
}

pub fn build_report() -> io::Result<PhysicalComponentKReport> {
    let target_report = verify_target_complex();
    let free = build_free_complex().report;
    let target_sectors = [
        target_sector_complex(TargetSector::Graviton),
        target_sector_complex(TargetSector::FourForm),
        target_sector_complex(TargetSector::RaritaSchwinger),
    ];
    let target_sector_reports = [
        &target_report.graviton,
        &target_report.four_form,
        &target_report.rarita_schwinger,
    ];

    let formal_sectors = target_sectors
        .iter()
        .zip(target_sector_reports)
        .map(|(complex, report)| {
            let fk = complex.curvature.multiply(&complex.gauge).nonzero_terms();
            let bianchi_f = complex.bianchi.multiply(&complex.curvature).nonzero_terms();
            let euler_k = complex
                .euler_lagrange
                .multiply(&complex.gauge)
                .nonzero_terms();
            // Mutate K by adding a constant unit map from parameter 0 into
            // the first potential column on which F is nonzero.  Since the
            // unmutated F K vanishes, the mutated residual is exactly that
            // nonzero formal F column.
            let (mutation_column, fk_mutation_terms) = (0..complex.curvature.columns())
                .map(|column| (column, complex.curvature.column_terms(column).len()))
                .find(|(_, terms)| *terms != 0)
                .expect("every physical curvature has a nonzero potential column");
            let passed = report.passed
                && fk == 0
                && bianchi_f == 0
                && euler_k == 0
                && report.euler_factorization_residual_terms == 0
                && report.mutation_residual_terms > 0
                && fk_mutation_terms > 0;
            FormalSectorCertificate {
                sector: report.sector.to_string(),
                gauge_dimensions: report.gauge_dimensions,
                curvature_dimensions: report.curvature_dimensions,
                bianchi_dimensions: report.bianchi_dimensions,
                euler_dimensions: report.euler_dimensions,
                curvature_after_gauge_residual_terms: fk,
                bianchi_after_curvature_residual_terms: bianchi_f,
                euler_factorization_residual_terms: report.euler_factorization_residual_terms,
                euler_after_gauge_residual_terms: euler_k,
                noether_after_euler_residual_terms: report.noether_after_euler_residual_terms,
                mutation_residual_terms: report.mutation_residual_terms,
                fk_gauge_term_mutation_potential_column: mutation_column,
                fk_gauge_term_mutation_residual_terms: fk_mutation_terms,
                passed,
            }
        })
        .collect::<Vec<_>>();

    let free_sectors = [&free.graviton, &free.three_form, &free.gravitino];
    let expected = [44, 84, 128];
    let null_cohomology = free_sectors
        .iter()
        .zip(expected)
        .map(|(sector, expected_physical)| {
            let c = &sector.cohomology;
            let raw_quotient = c.potential_dimension - c.gauge_rank;
            NullCohomologyCertificate {
                sector: c.sector.to_string(),
                potential_dimension: c.potential_dimension,
                gauge_parameter_dimension: c.gauge_parameter_dimension,
                gauge_rank: c.gauge_rank,
                raw_potential_quotient_dimension: raw_quotient,
                euler_rank: c.euler_lagrange_rank,
                on_shell_kernel_dimension: c.on_shell_kernel_dimension,
                gauge_image_lies_in_euler_kernel: c.gauge_image_lies_in_kernel,
                physical_cohomology_dimension: c.physical_cohomology_dimension,
                expected_physical_cohomology_dimension: expected_physical,
                passed: c.gauge_image_lies_in_kernel
                    && c.physical_cohomology_dimension == expected_physical,
            }
        })
        .collect::<Vec<_>>();

    let three = crate::eleven_dimensional_free_complex::three_form_complex(NULL_MOMENTUM);
    assert_eq!(three.reducibility.len(), 2);
    let d0_rank = three.reducibility[0].rank();
    let d1_rank = three.reducibility[1].rank();
    let d2_rank = three.gauge.rank();
    let first_composition = three.reducibility[1].multiply(&three.reducibility[0]);
    let second_composition = three.gauge.multiply(&three.reducibility[1]);
    let three_form_reducibility = ThreeFormReducibilityCertificate {
        chain_dimensions: [
            (
                three.reducibility[0].rows(),
                three.reducibility[0].columns(),
            ),
            (
                three.reducibility[1].rows(),
                three.reducibility[1].columns(),
            ),
            (three.gauge.rows(), three.gauge.columns()),
        ],
        chain_ranks_at_null_momentum: [d0_rank, d1_rank, d2_rank],
        first_composition_residual_entries: first_composition.nonzero_entries(),
        second_composition_residual_entries: second_composition.nonzero_entries(),
        rank_exactness: [d0_rank + d1_rank == 11, d1_rank + d2_rank == 55],
        passed: first_composition.is_zero()
            && second_composition.is_zero()
            && [d0_rank, d1_rank, d2_rank] == [1, 10, 45]
            && d0_rank + d1_rank == 11
            && d1_rank + d2_rank == 55,
    };

    let dependency_bindings = PINNED_ARTIFACTS
        .iter()
        .map(|(path, expected)| bind_artifact(path, expected))
        .collect::<io::Result<Vec<_>>>()?;
    let source_paths = [
        "src/eleven_dimensional_physical_component_k.rs",
        "src/eleven_dimensional_target_equation_complex.rs",
        "src/eleven_dimensional_free_complex.rs",
    ];
    let source_sha256 = source_paths
        .into_iter()
        .map(|path| Ok((path.to_string(), hash_file(Path::new(path))?)))
        .collect::<io::Result<BTreeMap<_, _>>>()?;

    let formal_direct_sum_fk_residual_terms = formal_sectors
        .iter()
        .map(|sector| sector.curvature_after_gauge_residual_terms)
        .sum();
    let formal_direct_sum_bianchi_f_residual_terms = formal_sectors
        .iter()
        .map(|sector| sector.bianchi_after_curvature_residual_terms)
        .sum();
    let formal_direct_sum_euler_k_residual_terms = formal_sectors
        .iter()
        .map(|sector| sector.euler_after_gauge_residual_terms)
        .sum();
    let formal_direct_sum_euler_factorization_residual_terms = formal_sectors
        .iter()
        .map(|sector| sector.euler_factorization_residual_terms)
        .sum();
    let raw_potential_quotient_dimensions = [
        null_cohomology[0].raw_potential_quotient_dimension,
        null_cohomology[1].raw_potential_quotient_dimension,
        null_cohomology[2].raw_potential_quotient_dimension,
    ];
    let derived_k_dimensions = (
        target_sectors
            .iter()
            .map(|sector| sector.gauge.rows())
            .sum(),
        target_sectors
            .iter()
            .map(|sector| sector.gauge.columns())
            .sum(),
    );
    let derived_f_dimensions = (
        target_sectors
            .iter()
            .map(|sector| sector.curvature.rows())
            .sum(),
        target_sectors
            .iter()
            .map(|sector| sector.curvature.columns())
            .sum(),
    );

    let passed = target_report.passed
        && free.passed
        && formal_sectors.iter().all(|sector| sector.passed)
        && formal_direct_sum_fk_residual_terms == 0
        && formal_direct_sum_bianchi_f_residual_terms == 0
        && formal_direct_sum_euler_k_residual_terms == 0
        && formal_direct_sum_euler_factorization_residual_terms == 0
        && null_cohomology.iter().all(|sector| sector.passed)
        && raw_potential_quotient_dimensions == [55, 120, 320]
        && free.null_momentum_square == 0
        && free.bosonic_physical_dimension == 128
        && free.fermionic_physical_dimension == 128
        && three_form_reducibility.passed
        && derived_k_dimensions == (583, 98)
        && derived_f_dimensions == (5115, 583)
        && dependency_bindings.iter().all(|binding| binding.matched);

    Ok(PhysicalComponentKReport {
        schema_version: "adynkra-11d-physical-component-k-v1",
        role: "exact direct-sum component gauge map, curvature map, and null-momentum physical cohomology",
        signature: "Spin(1,10), mostly-plus eta=(-,+,...,+)",
        physical_component_k_constructed: true,
        prepotential_k_into_h_hat_constructed: false,
        direct_sum_parameter_basis: [
            "diffeomorphism xi_a (11)",
            "Abelian two-form Lambda_ab (55)",
            "local supersymmetry epsilon_alpha (32)",
        ],
        direct_sum_potential_basis: [
            "symmetric metric perturbation h_ab (66)",
            "three-form potential A_abc (165)",
            "vector-spinor gravitino psi_a^alpha (352)",
        ],
        direct_sum_curvature_basis: [
            "ambient ordered antisymmetric-pair R_ab|cd basis (3025)",
            "four-form G_abcd (330)",
            "antisymmetric gravitino curl C_ab^alpha (1760)",
        ],
        exact_gauge_formulas: [
            "delta h_ab = p_a xi_b + p_b xi_a",
            "delta A_abc = p_a Lambda_bc - p_b Lambda_ac + p_c Lambda_ab",
            "delta psi_a^alpha = p_a epsilon^alpha",
            "delta Lambda_ab = p_a lambda_b - p_b lambda_a",
            "delta lambda_a = p_a sigma",
        ],
        antisymmetrization_convention: "stored exterior-product coefficients are unnormalized: no 1/q! factorial is inserted in p wedge omega",
        formula_source: "arXiv:0903.0259, Eq. (2) and the following free three-form reducibility discussion",
        direct_sum_k_dimensions: derived_k_dimensions,
        direct_sum_f_dimensions: derived_f_dimensions,
        riemann_row_basis: "the target_sector_complex graviton curvature uses the ambient 55 by 55 ordered pair basis; no algebraic-symmetry quotient or 1210-row projector is applied",
        ambient_riemann_rows: 3025,
        canonical_riemann_irrep_rows_not_used: 1210,
        formal_sectors,
        formal_direct_sum_fk_residual_terms,
        formal_direct_sum_bianchi_f_residual_terms,
        formal_direct_sum_euler_k_residual_terms,
        formal_direct_sum_euler_factorization_residual_terms,
        null_momentum_covector: NULL_MOMENTUM,
        null_momentum_square: free.null_momentum_square,
        null_cohomology,
        raw_potential_quotient_dimensions,
        physical_bosonic_split: [44, 84],
        physical_bosonic_dimension: free.bosonic_physical_dimension,
        physical_fermionic_dimension: free.fermionic_physical_dimension,
        three_form_reducibility,
        dependency_bindings,
        source_sha256,
        passed,
        scope_boundary: "This is the independent free component-field complex only. It constructs K from component gauge parameters into (h,A3,psi) and F from those component potentials into (R,G4,curl). It does not construct, infer, or validate the missing prepotential map K: H_Xi -> H_hat, a source superfield equation, auxiliary fields, interactions, or off-shell supersymmetry closure.",
    })
}

pub fn write_artifact(path: &Path) -> io::Result<PhysicalComponentKReport> {
    let report = build_report()?;
    let encoded = serde_json::to_vec_pretty(&report).map_err(io::Error::other)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    fs::write(&temporary, [&encoded[..], b"\n"].concat())?;
    fs::rename(&temporary, path)?;
    Ok(report)
}

fn temporary_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(".{name}.tmp-{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_k_is_exact_and_is_not_the_missing_h_hat_k() {
        let report = build_report().expect("build physical component K report");
        assert!(report.passed, "{report:#?}");
        assert!(report.physical_component_k_constructed);
        assert!(!report.prepotential_k_into_h_hat_constructed);
        assert_eq!(report.direct_sum_k_dimensions, (583, 98));
        assert_eq!(report.direct_sum_f_dimensions, (5115, 583));
        assert_eq!(report.formal_direct_sum_fk_residual_terms, 0);
        assert_eq!(report.raw_potential_quotient_dimensions, [55, 120, 320]);
        assert_eq!(report.physical_bosonic_split, [44, 84]);
        assert_eq!(report.physical_fermionic_dimension, 128);
    }
}
