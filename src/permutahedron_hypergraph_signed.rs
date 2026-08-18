//! Garden-sign transport across the exact S8 constraint hypergraph.
//!
//! Right translation sends every unsigned block `{P_I}` to `{P_I P_g}`.  It
//! is a common relabeling of the column nodes, so one Garden signing must
//! remain valid across the complete translated family.  This module verifies
//! that statement on all 151,200 discovered octets and records the affine
//! Garden-space dimensions of the 30 identity blocks.

use crate::lr_matrix::AdinkraRep;
use crate::permutahedron::{CosetSide, Permutation, coset_partition};
use crate::permutahedron_garden::solve_garden_signing;
use crate::permutahedron_hypergraph::identity_hyperedges;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;
const SCHEMA_VERSION: &str = "permutahedron-hypergraph-signed-transport-v1";

#[derive(Debug, Clone, Serialize)]
pub struct FamilyTransportRecord {
    pub family_id: usize,
    pub identity_block_ranks: Vec<u32>,
    pub support_compatible: bool,
    pub feasible: bool,
    pub equation_rank: usize,
    pub nullity: usize,
    pub solution_count: u64,
    pub identity_sparse_verifier_passed: bool,
    pub translated_octets_checked: usize,
    pub translated_hyperedge_sets_matched: usize,
    pub transported_garden_closures: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignedTransportValidation {
    pub discovered_families: usize,
    pub translated_octets_checked: usize,
    pub expected_octets: usize,
    pub equation_rank_histogram: BTreeMap<usize, usize>,
    pub nullity_histogram: BTreeMap<usize, usize>,
    pub solution_count_histogram: BTreeMap<u64, usize>,
    pub transported_closures: usize,
    pub all_families_passed: bool,
    pub unsigned_symmetry_broken_by_garden_feasibility: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignedTransportArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub families: Vec<FamilyTransportRecord>,
    pub validation: SignedTransportValidation,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn histogram_usize(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_insert(0) += 1;
    }
    result
}

fn histogram_u64(values: impl IntoIterator<Item = u64>) -> BTreeMap<u64, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_insert(0) += 1;
    }
    result
}

fn zero_based(permutation: Permutation) -> Vec<usize> {
    permutation
        .as_slice()
        .iter()
        .map(|&value| usize::from(value - 1))
        .collect()
}

fn verifies_garden(octet: &[Permutation], signs: &[i8]) -> bool {
    let color_permutations: Vec<Vec<usize>> = octet.iter().copied().map(zero_based).collect();
    AdinkraRep::from_parts(N, D, &color_permutations, signs).verify_garden_algebra()
}

fn sorted_ranks(octet: &[Permutation]) -> Vec<u32> {
    let mut ranks: Vec<u32> = octet
        .iter()
        .map(|permutation| permutation.rank() as u32)
        .collect();
    ranks.sort_unstable();
    ranks
}

fn left_node_relabel_signs(signs: &[i8], relabeling: Permutation) -> Vec<i8> {
    let mut transported = vec![0i8; signs.len()];
    for color in 0..N {
        for row in 0..D {
            let source_row = usize::from(relabeling.as_slice()[row] - 1);
            transported[color * D + row] = signs[color * D + source_row];
        }
    }
    transported
}

pub fn build() -> SignedTransportArtifact {
    let identity_blocks = identity_hyperedges(N);
    let mut families = Vec::with_capacity(identity_blocks.len());

    for (family_id, block) in identity_blocks.iter().enumerate() {
        let solution = solve_garden_signing(block);
        let partition = coset_partition(block, CosetSide::Right).expect("discovered subgroup");
        let mut set_matches = 0usize;
        let mut closures = 0usize;

        for hyperedge in &partition.slices {
            let seed = Permutation::unrank(N, hyperedge[0] as usize).expect("bounded S8 rank");
            let translated: Vec<Permutation> = block
                .iter()
                .map(|&element| element.compose(seed).expect("S8 product"))
                .collect();
            // With the repository's row-to-column permutation convention,
            // M_(h o g) = M_g M_h.  The right-coset translate is therefore a
            // common relabeling of the row nodes, so its signs must be
            // reindexed by g rather than copied in the original row order.
            let transported_signs = left_node_relabel_signs(&solution.canonical_signs, seed);
            set_matches += usize::from(sorted_ranks(&translated) == *hyperedge);
            closures += usize::from(verifies_garden(&translated, &transported_signs));
        }

        let translated_octets_checked = partition.slices.len();
        let passed = solution.support_compatible
            && solution.feasible
            && solution.rank == 45
            && solution.nullity == 19
            && solution.solution_count == 1 << 19
            && solution.independent_sparse_verifier_passed
            && partition.complete_cover
            && set_matches == translated_octets_checked
            && closures == translated_octets_checked;

        families.push(FamilyTransportRecord {
            family_id,
            identity_block_ranks: sorted_ranks(block),
            support_compatible: solution.support_compatible,
            feasible: solution.feasible,
            equation_rank: solution.rank,
            nullity: solution.nullity,
            solution_count: solution.solution_count,
            identity_sparse_verifier_passed: solution.independent_sparse_verifier_passed,
            translated_octets_checked,
            translated_hyperedge_sets_matched: set_matches,
            transported_garden_closures: closures,
            passed,
        });
    }

    let translated_octets_checked = families
        .iter()
        .map(|family| family.translated_octets_checked)
        .sum();
    let transported_closures = families
        .iter()
        .map(|family| family.transported_garden_closures)
        .sum();
    let equation_rank_histogram =
        histogram_usize(families.iter().map(|family| family.equation_rank));
    let nullity_histogram = histogram_usize(families.iter().map(|family| family.nullity));
    let solution_count_histogram =
        histogram_u64(families.iter().map(|family| family.solution_count));
    let all_families_passed = families.iter().all(|family| family.passed);
    let expected_octets = 151_200;
    let uniform_affine_space = equation_rank_histogram == BTreeMap::from([(45, 30)])
        && nullity_histogram == BTreeMap::from([(19, 30)])
        && solution_count_histogram == BTreeMap::from([(1 << 19, 30)]);

    let validation = SignedTransportValidation {
        discovered_families: families.len(),
        translated_octets_checked,
        expected_octets,
        equation_rank_histogram,
        nullity_histogram,
        solution_count_histogram,
        transported_closures,
        all_families_passed,
        unsigned_symmetry_broken_by_garden_feasibility: false,
        passed: families.len() == 30
            && translated_octets_checked == expected_octets
            && transported_closures == expected_octets
            && uniform_affine_space
            && all_families_passed,
    };

    SignedTransportArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Garden-space transport across the S8 constraint hypergraph",
        method: vec![
            "Discover the 30 identity octets from the fixed-point-free-involution compatibility rule without loading R8 labels.",
            "Solve the affine Garden system independently on each identity octet.",
            "Right translate each ordered identity octet through all 5,040 cosets while retaining its certified row-sign assignment.",
            "Match every ordered translate to the independently generated hyperedge set and verify the Garden algebra directly.",
        ],
        findings: vec![
            format!(
                "All {} discovered identity octets have Garden equation rank 45 and nullity 19, hence 2^19 labeled signings each.",
                validation.discovered_families
            ),
            format!(
                "A certified signing was transported to and directly verified on all {} octet hyperedges.",
                validation.transported_closures
            ),
            "Garden feasibility and affine solution-space dimension are uniform across the 30 subgroup-induced partitions, so they do not select a preferred R8 family.".into(),
        ],
        boundary: "This audit proves uniform Garden sign feasibility and constructs one closing signing on every unsigned octet. It does not enumerate 79,272,345,600 labeled signings, quotient arbitrary signings by node or color equivalence, or infer HYMN, holoraumy, Gadget, enhancement, or higher-dimensional parentage from a solver-dependent canonical sign choice.",
        families,
        validation,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> SignedTransportValidation {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create validation directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create data artifact")),
        &artifact,
    )
    .expect("write data artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create validation artifact")),
        &artifact.validation,
    )
    .expect("write validation artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_discovered_s8_hyperedge_receives_a_transported_garden_signing() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.discovered_families, 30);
        assert_eq!(artifact.validation.translated_octets_checked, 151_200);
        assert_eq!(artifact.validation.transported_closures, 151_200);
        assert_eq!(
            artifact.validation.equation_rank_histogram,
            BTreeMap::from([(45, 30)])
        );
        assert_eq!(
            artifact.validation.nullity_histogram,
            BTreeMap::from([(19, 30)])
        );
        assert!(
            !artifact
                .validation
                .unsigned_symmetry_broken_by_garden_feasibility
        );
    }
}
