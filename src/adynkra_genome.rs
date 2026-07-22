//! Exact reproduction of the six 4D, N=1 Adynkra genomes in
//! Gates and Hu, arXiv:2407.09334, Eqs. (3.6)-(3.11).
//!
//! Complexified 4D Lorentz representations are stored as `SL(2) x SL(2)`
//! Dynkin labels `[a,b]`.  The two Grassmann level parameters transform as
//! `[1,0]` and `[0,1]`.  Their exterior powers terminate at degree two:
//! `wedge^0 = [0,0]`, `wedge^1 = [1,0]` or `[0,1]`, and
//! `wedge^2 = [0,0]`.  Tensor products are decomposed with the ordinary
//! Clebsch-Gordan rule in each factor.  No Clebsch-Gordan coefficients are
//! claimed or stored.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const SOURCE_ARXIV: &str = "2407.09334";
pub const SOURCE_VERSION: u8 = 1;
pub const SOURCE_PDF_SHA256: &str =
    "64f0ae888933a8a6ff7b768c73d21656baa557ef7b089aab9f6252129ee58f81";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct LorentzIrrep {
    pub left: u8,
    pub right: u8,
}

impl LorentzIrrep {
    pub const fn new(left: u8, right: u8) -> Self {
        Self { left, right }
    }

    pub const fn dimension(self) -> usize {
        (self.left as usize + 1) * (self.right as usize + 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GenomeTerm {
    pub left_degree: u8,
    pub right_degree: u8,
    pub irrep: LorentzIrrep,
    pub multiplicity: usize,
    pub coefficient_numerator: usize,
    pub coefficient_denominator: usize,
}

#[derive(Debug, Clone, Copy)]
struct GenomeDefinition {
    id: &'static str,
    name: &'static str,
    equation: &'static str,
    seed: LorentzIrrep,
    include_right_levels: bool,
}

const DEFINITIONS: [GenomeDefinition; 6] = [
    GenomeDefinition {
        id: "chiral",
        name: "chiral supermultiplet",
        equation: "3.6",
        seed: LorentzIrrep::new(0, 0),
        include_right_levels: false,
    },
    GenomeDefinition {
        id: "two_form_gauge",
        name: "2-form gauge-field supermultiplet",
        equation: "3.7",
        seed: LorentzIrrep::new(1, 0),
        include_right_levels: false,
    },
    GenomeDefinition {
        id: "one_form_variant_gauge",
        name: "1-form variant gauge-field supermultiplet",
        equation: "3.8",
        seed: LorentzIrrep::new(0, 1),
        include_right_levels: false,
    },
    GenomeDefinition {
        id: "one_form_gauge",
        name: "1-form gauge-field supermultiplet",
        equation: "3.9",
        seed: LorentzIrrep::new(0, 0),
        include_right_levels: true,
    },
    GenomeDefinition {
        id: "matter_gravitino",
        name: "matter-gravitino supermultiplet",
        equation: "3.10",
        seed: LorentzIrrep::new(1, 0),
        include_right_levels: true,
    },
    GenomeDefinition {
        id: "supergravity",
        name: "supergravity supermultiplet",
        equation: "3.11",
        seed: LorentzIrrep::new(1, 1),
        include_right_levels: true,
    },
];

#[derive(Debug, Clone, Serialize)]
pub struct GenomeRecord {
    pub id: &'static str,
    pub name: &'static str,
    pub source_equation: &'static str,
    pub seed: LorentzIrrep,
    pub right_levels_included: bool,
    pub terms: Vec<GenomeTerm>,
    pub total_representation_dimension: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenomeArtifact {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_version: u8,
    pub source_pdf_sha256: &'static str,
    pub source_pdf_pages: &'static str,
    pub representation_convention: &'static str,
    pub coefficient_convention: &'static str,
    pub boundary: &'static str,
    pub genomes: Vec<GenomeRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenomeCheck {
    pub id: &'static str,
    pub equation: &'static str,
    pub expected_terms: usize,
    pub generated_terms: usize,
    pub expected_total_dimension: usize,
    pub generated_total_dimension: usize,
    pub exact_match: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenomeValidation {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub equations_checked: &'static str,
    pub genomes_checked: usize,
    pub terms_checked: usize,
    pub mismatched_genomes: usize,
    pub checks: Vec<GenomeCheck>,
    pub passed: bool,
}

fn factorial(degree: u8) -> usize {
    match degree {
        0 | 1 => 1,
        2 => 2,
        _ => panic!("4D N=1 spinor exterior degree exceeds two"),
    }
}

fn su2_product(a: u8, b: u8) -> Vec<u8> {
    let low = a.abs_diff(b);
    let high = a + b;
    (low..=high).step_by(2).collect()
}

pub fn tensor_product(left: LorentzIrrep, right: LorentzIrrep) -> Vec<LorentzIrrep> {
    let mut result = Vec::new();
    for a in su2_product(left.left, right.left) {
        for b in su2_product(left.right, right.right) {
            result.push(LorentzIrrep::new(a, b));
        }
    }
    result.sort_unstable();
    result
}

fn exterior_level(left_degree: u8, right_degree: u8) -> LorentzIrrep {
    let left = match left_degree {
        0 | 2 => 0,
        1 => 1,
        _ => panic!("left exterior degree exceeds two"),
    };
    let right = match right_degree {
        0 | 2 => 0,
        1 => 1,
        _ => panic!("right exterior degree exceeds two"),
    };
    LorentzIrrep::new(left, right)
}

fn generate(definition: GenomeDefinition) -> GenomeRecord {
    let right_max = if definition.include_right_levels {
        2
    } else {
        0
    };
    let mut terms = Vec::new();
    for left_degree in 0..=2 {
        for right_degree in 0..=right_max {
            let level = exterior_level(left_degree, right_degree);
            let mut multiplicities = BTreeMap::<LorentzIrrep, usize>::new();
            for irrep in tensor_product(definition.seed, level) {
                *multiplicities.entry(irrep).or_default() += 1;
            }
            for (irrep, multiplicity) in multiplicities {
                terms.push(GenomeTerm {
                    left_degree,
                    right_degree,
                    irrep,
                    multiplicity,
                    coefficient_numerator: 1,
                    coefficient_denominator: factorial(left_degree) * factorial(right_degree),
                });
            }
        }
    }
    terms.sort_by_key(|term| {
        (
            term.left_degree + term.right_degree,
            term.left_degree,
            term.right_degree,
            term.irrep,
        )
    });
    let total_representation_dimension = terms
        .iter()
        .map(|term| term.multiplicity * term.irrep.dimension())
        .sum();
    GenomeRecord {
        id: definition.id,
        name: definition.name,
        source_equation: definition.equation,
        seed: definition.seed,
        right_levels_included: definition.include_right_levels,
        terms,
        total_representation_dimension,
    }
}

pub fn artifact() -> GenomeArtifact {
    GenomeArtifact {
        schema_version: "adynkra-4d-n1-genomes-v1",
        source_arxiv: SOURCE_ARXIV,
        source_version: SOURCE_VERSION,
        source_pdf_sha256: SOURCE_PDF_SHA256,
        source_pdf_pages: "15-16 (journal-style page numbers; PDF pages 15-16)",
        representation_convention: "complexified Spin(1,3) Dynkin labels [a,b] for SL(2)_L x SL(2)_R",
        coefficient_convention: "each bidegree (p,q) carries 1/(p! q!) from the genome exponential",
        boundary: "representation content and multiplicities only; no Clebsch-Gordan coefficients or component transformation laws",
        genomes: DEFINITIONS.into_iter().map(generate).collect(),
    }
}

fn term(p: u8, q: u8, a: u8, b: u8) -> GenomeTerm {
    GenomeTerm {
        left_degree: p,
        right_degree: q,
        irrep: LorentzIrrep::new(a, b),
        multiplicity: 1,
        coefficient_numerator: 1,
        coefficient_denominator: factorial(p) * factorial(q),
    }
}

/// Independent literal transcription of the representation content in
/// Eqs. (3.6)-(3.11).  This fixture does not call the tensor-product engine.
fn published_fixture(id: &str) -> Vec<GenomeTerm> {
    let rows: &[(u8, u8, u8, u8)] = match id {
        "chiral" => &[(0, 0, 0, 0), (1, 0, 1, 0), (2, 0, 0, 0)],
        "two_form_gauge" => &[(0, 0, 1, 0), (1, 0, 0, 0), (1, 0, 2, 0), (2, 0, 1, 0)],
        "one_form_variant_gauge" => &[(0, 0, 0, 1), (1, 0, 1, 1), (2, 0, 0, 1)],
        "one_form_gauge" => &[
            (0, 0, 0, 0),
            (1, 0, 1, 0),
            (0, 1, 0, 1),
            (2, 0, 0, 0),
            (0, 2, 0, 0),
            (1, 1, 1, 1),
            (2, 1, 0, 1),
            (1, 2, 1, 0),
            (2, 2, 0, 0),
        ],
        "matter_gravitino" => &[
            (0, 0, 1, 0),
            (1, 0, 0, 0),
            (1, 0, 2, 0),
            (0, 1, 1, 1),
            (2, 0, 1, 0),
            (0, 2, 1, 0),
            (1, 1, 0, 1),
            (1, 1, 2, 1),
            (2, 1, 1, 1),
            (1, 2, 0, 0),
            (1, 2, 2, 0),
            (2, 2, 1, 0),
        ],
        "supergravity" => &[
            (0, 0, 1, 1),
            (1, 0, 0, 1),
            (1, 0, 2, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 2),
            (2, 0, 1, 1),
            (0, 2, 1, 1),
            (1, 1, 0, 0),
            (1, 1, 0, 2),
            (1, 1, 2, 0),
            (1, 1, 2, 2),
            (2, 1, 1, 0),
            (2, 1, 1, 2),
            (1, 2, 0, 1),
            (1, 2, 2, 1),
            (2, 2, 1, 1),
        ],
        _ => panic!("unknown genome fixture {id}"),
    };
    let mut terms: Vec<_> = rows.iter().map(|&(p, q, a, b)| term(p, q, a, b)).collect();
    terms.sort_by_key(|term| {
        (
            term.left_degree + term.right_degree,
            term.left_degree,
            term.right_degree,
            term.irrep,
        )
    });
    terms
}

pub fn verify() -> GenomeValidation {
    let generated = artifact();
    let mut checks = Vec::new();
    let mut terms_checked = 0;
    for genome in &generated.genomes {
        let expected = published_fixture(genome.id);
        let expected_total_dimension = expected
            .iter()
            .map(|term| term.multiplicity * term.irrep.dimension())
            .sum();
        terms_checked += expected.len();
        checks.push(GenomeCheck {
            id: genome.id,
            equation: genome.source_equation,
            expected_terms: expected.len(),
            generated_terms: genome.terms.len(),
            expected_total_dimension,
            generated_total_dimension: genome.total_representation_dimension,
            exact_match: expected == genome.terms,
        });
    }
    let mismatched_genomes = checks.iter().filter(|check| !check.exact_match).count();
    GenomeValidation {
        schema_version: "adynkra-4d-n1-genome-validation-v1",
        source_arxiv: SOURCE_ARXIV,
        equations_checked: "3.6-3.11",
        genomes_checked: checks.len(),
        terms_checked,
        mismatched_genomes,
        passed: checks.len() == DEFINITIONS.len() && mismatched_genomes == 0,
        checks,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> GenomeValidation {
    let generated = artifact();
    let validation = verify();
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    if let Some(parent) = validation_path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    fs::write(
        data_path,
        serde_json::to_string_pretty(&generated).unwrap() + "\n",
    )
    .unwrap_or_else(|error| panic!("failed to write {}: {error}", data_path.display()));
    fs::write(
        validation_path,
        serde_json::to_string_pretty(&validation).unwrap() + "\n",
    )
    .unwrap_or_else(|error| panic!("failed to write {}: {error}", validation_path.display()));
    validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn su2_tensor_products_are_exact() {
        assert_eq!(su2_product(1, 1), vec![0, 2]);
        assert_eq!(su2_product(2, 1), vec![1, 3]);
        assert_eq!(
            tensor_product(LorentzIrrep::new(1, 1), LorentzIrrep::new(1, 1)),
            vec![
                LorentzIrrep::new(0, 0),
                LorentzIrrep::new(0, 2),
                LorentzIrrep::new(2, 0),
                LorentzIrrep::new(2, 2),
            ]
        );
    }

    #[test]
    fn six_published_genomes_match_term_for_term() {
        let validation = verify();
        assert!(validation.passed);
        assert_eq!(validation.genomes_checked, 6);
        assert_eq!(validation.mismatched_genomes, 0);
        assert_eq!(validation.terms_checked, 47);
    }

    #[test]
    fn full_scalar_and_spinor_superfields_have_expected_dimensions() {
        let generated = artifact();
        let vector = generated
            .genomes
            .iter()
            .find(|genome| genome.id == "one_form_gauge")
            .unwrap();
        let matter_gravitino = generated
            .genomes
            .iter()
            .find(|genome| genome.id == "matter_gravitino")
            .unwrap();
        let supergravity = generated
            .genomes
            .iter()
            .find(|genome| genome.id == "supergravity")
            .unwrap();
        assert_eq!(vector.total_representation_dimension, 16);
        assert_eq!(matter_gravitino.total_representation_dimension, 32);
        assert_eq!(supergravity.total_representation_dimension, 64);
    }

    #[test]
    fn chirality_removes_all_right_levels() {
        for genome in artifact().genomes.iter().take(3) {
            assert!(genome.terms.iter().all(|term| term.right_degree == 0));
        }
    }
}
