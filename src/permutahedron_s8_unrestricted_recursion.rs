//! Exhaustive Boolean-mask audit of the aligned signed S8 recursion.
//!
//! The published construction tests eight cyclic weight-four masks on thirty
//! ordered distinct source pairs. This module removes that mask restriction,
//! adds same-source controls, and maps every exact Garden closer into the
//! complete constraint hypergraph and the fixed-R8 normalizer orbit atlas.

use crate::permutahedron::{coset_partition, CosetSide, Permutation};
use crate::permutahedron_hypergraph::identity_hyperedges;
use crate::permutahedron_s8_signed_recursion::{
    aligned_recursive_boolean_factors, aligned_recursive_permutations, build_rep, cyclic_mask,
    exact_published_match, S4_RECURSION_LABELS,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-unrestricted-recursion-v1";

#[derive(Debug, Clone, Serialize)]
pub struct ClosingRealization {
    pub first_label: &'static str,
    pub second_label: &'static str,
    pub same_source_pair: bool,
    pub source_parentage_category: &'static str,
    pub second_color_order_one_based: [usize; 4],
    pub boolean_mask_decimal: u8,
    pub boolean_mask_hex: String,
    pub boolean_mask_weight: u32,
    pub published_cyclic_mask: bool,
    pub exact_published_system_match: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClosingSupport {
    pub hyperedge_ranks: Vec<u32>,
    pub discovered_family_id: usize,
    pub family_slice_id: usize,
    pub normalizer_orbit_id: Option<u8>,
    pub closing_realizations: usize,
    pub source_parentage_categories: Vec<&'static str>,
    pub realizations: Vec<ClosingRealization>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnrestrictedValidation {
    pub source_labels: usize,
    pub ordered_pairs_checked: usize,
    pub ordered_distinct_pairs_checked: usize,
    pub ordered_same_source_pairs_checked: usize,
    pub relative_color_orders_checked_per_pair: usize,
    pub boolean_masks_checked_per_alignment: usize,
    pub candidates_checked: usize,
    pub distinct_pair_candidates_checked: usize,
    pub same_source_candidates_checked: usize,
    pub closing_candidates: usize,
    pub closing_distinct_pair_candidates: usize,
    pub closing_same_source_candidates: usize,
    pub published_cyclic_distinct_pair_closers_recovered: usize,
    pub noncyclic_closers: usize,
    pub distinct_closing_supports: usize,
    pub additional_supports_beyond_published_cyclic_scan: usize,
    pub closing_realization_count_histogram: BTreeMap<usize, usize>,
    pub family_histogram_by_support: BTreeMap<usize, usize>,
    pub normalizer_orbit_histogram_by_support: BTreeMap<u8, usize>,
    pub parentage_category_histogram_by_realization: BTreeMap<String, usize>,
    pub parentage_category_histogram_by_support_incidence: BTreeMap<String, usize>,
    pub normalizer_orbit_category_sets: BTreeMap<u8, Vec<String>>,
    pub normalizer_orbit_separates_source_parentage_categories: bool,
    pub every_closer_maps_to_the_constraint_hypergraph: bool,
    pub published_restricted_scan_reproduced_exactly: bool,
    pub audit_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnrestrictedArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub supports: Vec<ClosingSupport>,
    pub validation: UnrestrictedValidation,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn color_orders_4() -> Vec<[usize; 4]> {
    let mut output = Vec::with_capacity(24);
    for a in 0..4 {
        for b in 0..4 {
            if b == a {
                continue;
            }
            for c in 0..4 {
                if c == a || c == b {
                    continue;
                }
                for d in 0..4 {
                    if d != a && d != b && d != c {
                        output.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    output
}

fn support_ranks(permutations: &[[u8; 8]; 8]) -> Vec<u32> {
    let mut output: Vec<u32> = permutations
        .iter()
        .map(|permutation| {
            Permutation::new(permutation)
                .expect("recursive permutation")
                .rank() as u32
        })
        .collect();
    output.sort_unstable();
    output
}

fn source_parentage_category(first: usize, second: usize) -> &'static str {
    let named = |index| index < 3;
    match (named(first), named(second)) {
        (true, true) => "named-parent pair",
        (false, false) => "unstated-parent pair",
        _ => "mixed-source pair",
    }
}

pub fn build() -> UnrestrictedArtifact {
    let color_orders = color_orders_4();
    let cyclic_masks: BTreeSet<u8> = (0..8).map(|start| cyclic_mask(start).0).collect();
    assert_eq!(cyclic_masks.len(), 8);

    let mut hyperedge_index = BTreeMap::new();
    for (family_id, family) in identity_hyperedges(8).iter().enumerate() {
        let partition =
            coset_partition(family, CosetSide::Right).expect("identity block is a subgroup");
        for (slice_id, hyperedge) in partition.slices.into_iter().enumerate() {
            assert!(
                hyperedge_index
                    .insert(hyperedge, (family_id, slice_id))
                    .is_none(),
                "constraint hyperedges are globally unique"
            );
        }
    }
    let normalizer_orbit_by_slice = crate::permutahedron_s8_orbits::normalizer_orbit_assignment();

    let mut realizations_by_support = BTreeMap::<Vec<u32>, Vec<ClosingRealization>>::new();
    let mut candidates_checked = 0usize;
    let mut closing_candidates = 0usize;
    let mut closing_distinct_pair_candidates = 0usize;
    let mut closing_same_source_candidates = 0usize;
    let mut published_cyclic_distinct_pair_closers_recovered = 0usize;

    for first in 0..6 {
        for second in 0..6 {
            let same_source_pair = first == second;
            let category = source_parentage_category(first, second);
            for alignment in &color_orders {
                let permutations = aligned_recursive_permutations(first, second, *alignment);
                let ranks = support_ranks(&permutations);
                for mask in u8::MIN..=u8::MAX {
                    candidates_checked += 1;
                    let factors =
                        aligned_recursive_boolean_factors(first, second, *alignment, mask);
                    if !build_rep(&permutations, &factors).verify_garden_algebra() {
                        continue;
                    }
                    closing_candidates += 1;
                    if same_source_pair {
                        closing_same_source_candidates += 1;
                    } else {
                        closing_distinct_pair_candidates += 1;
                    }
                    let published_cyclic_mask = cyclic_masks.contains(&mask);
                    if !same_source_pair && published_cyclic_mask {
                        published_cyclic_distinct_pair_closers_recovered += 1;
                    }
                    realizations_by_support
                        .entry(ranks.clone())
                        .or_default()
                        .push(ClosingRealization {
                            first_label: S4_RECURSION_LABELS[first],
                            second_label: S4_RECURSION_LABELS[second],
                            same_source_pair,
                            source_parentage_category: category,
                            second_color_order_one_based: alignment.map(|color| color + 1),
                            boolean_mask_decimal: mask,
                            boolean_mask_hex: format!("{mask:02x}"),
                            boolean_mask_weight: mask.count_ones(),
                            published_cyclic_mask,
                            exact_published_system_match: exact_published_match(
                                &permutations,
                                &factors,
                            ),
                        });
                }
            }
        }
    }

    let mut supports = Vec::with_capacity(realizations_by_support.len());
    let mut every_closer_maps = true;
    for (hyperedge_ranks, mut realizations) in realizations_by_support {
        realizations.sort_unstable_by(|left, right| {
            (
                left.first_label,
                left.second_label,
                left.second_color_order_one_based,
                left.boolean_mask_decimal,
            )
                .cmp(&(
                    right.first_label,
                    right.second_label,
                    right.second_color_order_one_based,
                    right.boolean_mask_decimal,
                ))
        });
        let Some(&(discovered_family_id, family_slice_id)) = hyperedge_index.get(&hyperedge_ranks)
        else {
            every_closer_maps = false;
            continue;
        };
        let source_parentage_categories: Vec<&'static str> = realizations
            .iter()
            .map(|realization| realization.source_parentage_category)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        supports.push(ClosingSupport {
            hyperedge_ranks,
            discovered_family_id,
            family_slice_id,
            normalizer_orbit_id: (discovered_family_id == 0)
                .then(|| normalizer_orbit_by_slice[family_slice_id]),
            closing_realizations: realizations.len(),
            source_parentage_categories,
            realizations,
        });
    }
    supports.sort_unstable_by_key(|support| {
        (
            support.discovered_family_id,
            support.family_slice_id,
            support.hyperedge_ranks.clone(),
        )
    });

    let restricted_supports: BTreeSet<Vec<u32>> = supports
        .iter()
        .filter(|support| {
            support.realizations.iter().any(|realization| {
                !realization.same_source_pair && realization.published_cyclic_mask
            })
        })
        .map(|support| support.hyperedge_ranks.clone())
        .collect();
    let additional_supports_beyond_published_cyclic_scan =
        supports.len().saturating_sub(restricted_supports.len());
    let noncyclic_closers = supports
        .iter()
        .flat_map(|support| &support.realizations)
        .filter(|realization| !realization.published_cyclic_mask)
        .count();
    let mut closing_realization_count_histogram = BTreeMap::new();
    let mut family_histogram_by_support = BTreeMap::new();
    let mut normalizer_orbit_histogram_by_support = BTreeMap::new();
    let mut parentage_category_histogram_by_realization = BTreeMap::new();
    let mut parentage_category_histogram_by_support_incidence = BTreeMap::new();
    let mut orbit_categories = BTreeMap::<u8, BTreeSet<String>>::new();
    for support in &supports {
        *closing_realization_count_histogram
            .entry(support.closing_realizations)
            .or_default() += 1;
        *family_histogram_by_support
            .entry(support.discovered_family_id)
            .or_default() += 1;
        if let Some(orbit_id) = support.normalizer_orbit_id {
            *normalizer_orbit_histogram_by_support
                .entry(orbit_id)
                .or_default() += 1;
            let categories = orbit_categories.entry(orbit_id).or_default();
            categories.extend(
                support
                    .source_parentage_categories
                    .iter()
                    .map(|category| (*category).to_string()),
            );
        }
        for category in &support.source_parentage_categories {
            *parentage_category_histogram_by_support_incidence
                .entry((*category).to_string())
                .or_default() += 1;
        }
        for realization in &support.realizations {
            *parentage_category_histogram_by_realization
                .entry(realization.source_parentage_category.to_string())
                .or_default() += 1;
        }
    }
    let normalizer_orbit_category_sets: BTreeMap<u8, Vec<String>> = orbit_categories
        .into_iter()
        .map(|(orbit_id, categories)| (orbit_id, categories.into_iter().collect()))
        .collect();
    let normalizer_orbit_separates_source_parentage_categories = !normalizer_orbit_category_sets
        .is_empty()
        && normalizer_orbit_category_sets
            .values()
            .all(|categories| categories.len() == 1);

    let distinct_pair_candidates_checked = 30 * 24 * 256;
    let same_source_candidates_checked = 6 * 24 * 256;
    let published_restricted_scan_reproduced_exactly =
        published_cyclic_distinct_pair_closers_recovered == 24 && restricted_supports.len() == 12;
    let audit_passed = candidates_checked == 36 * 24 * 256
        && distinct_pair_candidates_checked + same_source_candidates_checked == candidates_checked
        && closing_distinct_pair_candidates + closing_same_source_candidates == closing_candidates
        && supports
            .iter()
            .map(|support| support.closing_realizations)
            .sum::<usize>()
            == closing_candidates
        && every_closer_maps
        && published_restricted_scan_reproduced_exactly;

    let validation = UnrestrictedValidation {
        source_labels: 6,
        ordered_pairs_checked: 36,
        ordered_distinct_pairs_checked: 30,
        ordered_same_source_pairs_checked: 6,
        relative_color_orders_checked_per_pair: 24,
        boolean_masks_checked_per_alignment: 256,
        candidates_checked,
        distinct_pair_candidates_checked,
        same_source_candidates_checked,
        closing_candidates,
        closing_distinct_pair_candidates,
        closing_same_source_candidates,
        published_cyclic_distinct_pair_closers_recovered,
        noncyclic_closers,
        distinct_closing_supports: supports.len(),
        additional_supports_beyond_published_cyclic_scan,
        closing_realization_count_histogram,
        family_histogram_by_support,
        normalizer_orbit_histogram_by_support,
        parentage_category_histogram_by_realization,
        parentage_category_histogram_by_support_incidence,
        normalizer_orbit_category_sets,
        normalizer_orbit_separates_source_parentage_categories,
        every_closer_maps_to_the_constraint_hypergraph: every_closer_maps,
        published_restricted_scan_reproduced_exactly,
        audit_passed,
    };
    let findings = vec![
        format!(
            "The unrestricted audit checks all {} aligned candidates, including {} same-source controls.",
            validation.candidates_checked, validation.same_source_candidates_checked
        ),
        format!(
            "Exact Garden closure retains {} candidates on {} unsigned supports.",
            validation.closing_candidates, validation.distinct_closing_supports
        ),
        format!(
            "The published cyclic restriction is reproduced with {} closers on 12 supports; unrestricted masks add {} supports.",
            validation.published_cyclic_distinct_pair_closers_recovered,
            validation.additional_supports_beyond_published_cyclic_scan
        ),
        format!(
            "Normalizer orbit category purity under unrestricted masks: {}.",
            validation.normalizer_orbit_separates_source_parentage_categories
        ),
    ];

    UnrestrictedArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Unrestricted Boolean-mask census of the aligned signed S8 recursion",
        method: vec![
            "Scan all 256 row-sign masks for every relative S4 color order and every ordered source pair.",
            "Include the six same-source ordered pairs as construction controls.",
            "Require exact eight-color Garden closure before retaining a candidate.",
            "Map every closing support to the complete constraint hypergraph and, for family zero, its normalizer-conjugacy orbit.",
            "Evaluate orbit and source-category purity at the support level rather than treating repeated signings as independent samples.",
        ],
        supports,
        validation,
        findings,
        boundary: "This is an exhaustive audit of the aligned block-recursion ansatz, not of all 2^64 Garden sign assignments and not of all possible S8 constructions. Source-parentage categories describe the provenance labels of the S4 inputs; they are not independently verified higher-dimensional labels for VM1, VM2, or VM3.",
    }
}

pub fn write_artifact(path: &Path) -> UnrestrictedValidation {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create unrestricted artifact")),
        &artifact,
    )
    .expect("write unrestricted artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrestricted_scan_reproduces_the_published_cyclic_subset() {
        let artifact = build();
        assert!(artifact.validation.audit_passed);
        assert_eq!(artifact.validation.candidates_checked, 221_184);
        assert!(
            artifact
                .validation
                .published_restricted_scan_reproduced_exactly
        );
    }
}
