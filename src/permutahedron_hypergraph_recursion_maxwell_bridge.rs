//! Map the closing signed S8 recursion candidates into the complete unsigned
//! constraint hypergraph and attach their embedded four-color Maxwell classes.

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

const SCHEMA_VERSION: &str = "permutahedron-hypergraph-recursion-maxwell-bridge-v1";

#[derive(Debug, Clone, Serialize)]
pub struct CandidateProjection {
    pub id: String,
    pub first_label: &'static str,
    pub second_label: &'static str,
    pub second_color_order_one_based: [usize; 4],
    pub start_position_one_based: usize,
    pub exact_published_system_match: Option<&'static str>,
    pub permutations: [[u8; 8]; 8],
    pub boolean_factors: [u8; 8],
    pub hyperedge_ranks: Vec<u32>,
    pub discovered_family_id: usize,
    pub family_slice_id: usize,
    pub normalizer_orbit_id: u8,
    pub ordered_chi0_signature: [i64; 2],
    pub ordered_maxwell_signature: [bool; 2],
    pub pair_has_two_named_parent_labels: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeValidation {
    pub recursive_candidates_checked: usize,
    pub closing_candidates: usize,
    pub candidates_mapped_uniquely: usize,
    pub distinct_unsigned_supports: usize,
    pub support_multiplicity_histogram: BTreeMap<usize, usize>,
    pub every_selected_support_has_two_closing_signings: bool,
    pub discovered_families_occupied: usize,
    pub family_histogram: BTreeMap<usize, usize>,
    pub all_closers_occupy_standard_family_zero: bool,
    pub normalizer_orbits_occupied: usize,
    pub candidate_normalizer_orbit_histogram: BTreeMap<u8, usize>,
    pub support_normalizer_orbit_histogram: BTreeMap<u8, usize>,
    pub named_parent_pair_orbit_ids: Vec<u8>,
    pub unstated_parent_pair_orbit_ids: Vec<u8>,
    pub normalizer_orbit_separates_source_parentage_categories: bool,
    pub every_occupied_orbit_mixes_ordered_maxwell_signatures: bool,
    pub every_closer_contains_exactly_one_maxwell_passing_block: bool,
    pub each_maxwell_signature_mixes_parentage_categories: bool,
    pub published_ct_projection_matches_control_ledger: bool,
    pub published_cv_projection_matches_control_ledger: bool,
    pub ct_and_cv_share_embedded_maxwell_signature: bool,
    pub ct_and_cv_have_distinct_unsigned_supports: bool,
    pub ct_normalizer_orbit_id: u8,
    pub cv_normalizer_orbit_id: u8,
    pub ct_and_cv_have_distinct_normalizer_orbits: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub candidates: Vec<CandidateProjection>,
    pub validation: BridgeValidation,
    pub findings: Vec<&'static str>,
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

fn ranks(permutations: &[[u8; 8]; 8]) -> Vec<u32> {
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

pub fn build() -> BridgeArtifact {
    let mut hyperedge_index = BTreeMap::new();
    for (family_id, family) in identity_hyperedges(8).iter().enumerate() {
        let partition =
            coset_partition(family, CosetSide::Right).expect("identity block is a subgroup");
        for (slice_id, hyperedge) in partition.slices.into_iter().enumerate() {
            assert!(
                hyperedge_index
                    .insert(hyperedge, (family_id, slice_id))
                    .is_none(),
                "discovered hyperedges are globally unique"
            );
        }
    }

    let maxwell = crate::maxwell_s8_subalgebra_scan::build();
    let maxwell_by_id: BTreeMap<_, _> = maxwell
        .candidates
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect();
    let controls = crate::permutahedron_hypergraph_controls::build();
    let control_by_id: BTreeMap<_, _> = controls
        .controls
        .iter()
        .map(|control| (control.id, control))
        .collect();
    let normalizer_orbit_by_slice = crate::permutahedron_s8_orbits::normalizer_orbit_assignment();

    let mut candidates = Vec::new();
    let mut recursive_candidates_checked = 0usize;
    for first in 0..6 {
        for second in 0..6 {
            if first == second {
                continue;
            }
            for alignment in color_orders_4() {
                let permutations = aligned_recursive_permutations(first, second, alignment);
                for start in 0..8 {
                    recursive_candidates_checked += 1;
                    let (mask, _) = cyclic_mask(start);
                    let factors = aligned_recursive_boolean_factors(first, second, alignment, mask);
                    let rep = build_rep(&permutations, &factors);
                    if !rep.verify_garden_algebra() {
                        continue;
                    }
                    let second_order = alignment.map(|color| color + 1);
                    let id = format!(
                        "{}->{}:order-{}{}{}{}:mask-{}",
                        S4_RECURSION_LABELS[first],
                        S4_RECURSION_LABELS[second],
                        second_order[0],
                        second_order[1],
                        second_order[2],
                        second_order[3],
                        start + 1,
                    );
                    let hyperedge_ranks = ranks(&permutations);
                    let &(discovered_family_id, family_slice_id) = hyperedge_index
                        .get(&hyperedge_ranks)
                        .expect("closing recursive support is a discovered hyperedge");
                    let maxwell_record = maxwell_by_id
                        .get(id.as_str())
                        .expect("Maxwell scan contains the same closing candidate");
                    let named = |label| matches!(label, "CM" | "TM" | "VM");
                    candidates.push(CandidateProjection {
                        id,
                        first_label: S4_RECURSION_LABELS[first],
                        second_label: S4_RECURSION_LABELS[second],
                        second_color_order_one_based: second_order,
                        start_position_one_based: start + 1,
                        exact_published_system_match: exact_published_match(
                            &permutations,
                            &factors,
                        ),
                        permutations,
                        boolean_factors: factors,
                        hyperedge_ranks,
                        discovered_family_id,
                        family_slice_id,
                        normalizer_orbit_id: normalizer_orbit_by_slice[family_slice_id],
                        ordered_chi0_signature: maxwell_record.ordered_chi0_signature,
                        ordered_maxwell_signature: maxwell_record.ordered_maxwell_signature,
                        pair_has_two_named_parent_labels: named(S4_RECURSION_LABELS[first])
                            && named(S4_RECURSION_LABELS[second]),
                    });
                }
            }
        }
    }

    let mut support_counts = BTreeMap::<Vec<u32>, usize>::new();
    for candidate in &candidates {
        *support_counts
            .entry(candidate.hyperedge_ranks.clone())
            .or_default() += 1;
    }
    let distinct_unsigned_supports = support_counts.len();
    let mut support_multiplicity_histogram = BTreeMap::new();
    for multiplicity in support_counts.values() {
        *support_multiplicity_histogram
            .entry(*multiplicity)
            .or_default() += 1;
    }
    let every_selected_support_has_two_closing_signings = support_counts
        .values()
        .all(|&multiplicity| multiplicity == 2);
    let mut candidate_normalizer_orbit_histogram = BTreeMap::new();
    for candidate in &candidates {
        *candidate_normalizer_orbit_histogram
            .entry(candidate.normalizer_orbit_id)
            .or_default() += 1;
    }
    let mut support_normalizer_orbit_histogram = BTreeMap::new();
    for support in support_counts.keys() {
        let candidate = candidates
            .iter()
            .find(|candidate| &candidate.hyperedge_ranks == support)
            .expect("selected support has a candidate");
        *support_normalizer_orbit_histogram
            .entry(candidate.normalizer_orbit_id)
            .or_default() += 1;
    }
    let normalizer_orbits_occupied = support_normalizer_orbit_histogram.len();
    let occupied_orbits: BTreeSet<u8> = candidates
        .iter()
        .map(|candidate| candidate.normalizer_orbit_id)
        .collect();
    let orbit_categories = |orbit_id| {
        candidates
            .iter()
            .filter(|candidate| candidate.normalizer_orbit_id == orbit_id)
            .map(|candidate| candidate.pair_has_two_named_parent_labels)
            .collect::<BTreeSet<_>>()
    };
    let named_parent_pair_orbit_ids: Vec<u8> = occupied_orbits
        .iter()
        .copied()
        .filter(|&orbit_id| orbit_categories(orbit_id) == BTreeSet::from([true]))
        .collect();
    let unstated_parent_pair_orbit_ids: Vec<u8> = occupied_orbits
        .iter()
        .copied()
        .filter(|&orbit_id| orbit_categories(orbit_id) == BTreeSet::from([false]))
        .collect();
    let normalizer_orbit_separates_source_parentage_categories = occupied_orbits
        .iter()
        .all(|&orbit_id| orbit_categories(orbit_id).len() == 1);
    let every_occupied_orbit_mixes_ordered_maxwell_signatures =
        occupied_orbits.iter().all(|&orbit_id| {
            candidates
                .iter()
                .filter(|candidate| candidate.normalizer_orbit_id == orbit_id)
                .map(|candidate| candidate.ordered_maxwell_signature)
                .collect::<BTreeSet<_>>()
                .len()
                == 2
        });
    let mut family_histogram = BTreeMap::new();
    for candidate in &candidates {
        *family_histogram
            .entry(candidate.discovered_family_id)
            .or_default() += 1;
    }
    let projection_for = |name| {
        candidates
            .iter()
            .find(|candidate| candidate.exact_published_system_match == Some(name))
    };
    let projection_matches_control = |name| {
        let projected = projection_for(name).expect("published recursion anchor");
        let control = control_by_id.get(name).expect("published control ledger");
        projected.discovered_family_id == control.discovered_family_id
            && projected.family_slice_id == control.family_slice_id
            && projected.hyperedge_ranks == control.hyperedge_ranks
    };
    let ct = projection_for("CT").expect("CT recursion anchor");
    let cv = projection_for("CV").expect("CV recursion anchor");
    let ct_and_cv_share_embedded_maxwell_signature =
        ct.ordered_maxwell_signature == cv.ordered_maxwell_signature;
    let ct_and_cv_have_distinct_unsigned_supports = ct.hyperedge_ranks != cv.hyperedge_ranks;
    let ct_normalizer_orbit_id = ct.normalizer_orbit_id;
    let cv_normalizer_orbit_id = cv.normalizer_orbit_id;
    let ct_and_cv_have_distinct_normalizer_orbits =
        ct_normalizer_orbit_id != cv_normalizer_orbit_id;
    let all_closers_occupy_standard_family_zero =
        family_histogram.len() == 1 && family_histogram.get(&0) == Some(&candidates.len());
    let passed = recursive_candidates_checked == 5_760
        && candidates.len() == 24
        && candidates.len() == maxwell.candidates.len()
        && distinct_unsigned_supports == 12
        && every_selected_support_has_two_closing_signings
        && support_normalizer_orbit_histogram.values().sum::<usize>() == 12
        && (1..20).contains(&normalizer_orbits_occupied)
        && named_parent_pair_orbit_ids == [7, 17]
        && unstated_parent_pair_orbit_ids == [1, 5]
        && normalizer_orbit_separates_source_parentage_categories
        && every_occupied_orbit_mixes_ordered_maxwell_signatures
        && family_histogram.values().sum::<usize>() == candidates.len()
        && all_closers_occupy_standard_family_zero
        && maxwell.every_closer_contains_exactly_one_maxwell_passing_block
        && maxwell.each_signature_mixes_parentage_categories
        && projection_matches_control("CT")
        && projection_matches_control("CV")
        && ct_and_cv_share_embedded_maxwell_signature
        && ct_and_cv_have_distinct_unsigned_supports
        && ct_normalizer_orbit_id == 17
        && cv_normalizer_orbit_id == 7
        && ct_and_cv_have_distinct_normalizer_orbits;

    BridgeArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Hypergraph projection of the signed S8 recursion with embedded Maxwell classes",
        method: vec![
            "Enumerate all 5,760 aligned signed recursion candidates and retain exact Garden closers.",
            "Map every closing unsigned support into the complete 151,200-edge constraint hypergraph.",
            "Attach the independently evaluated ordered chi0 and Maxwell signatures of both retained four-color blocks.",
            "Cross-check the CT and CV anchors against the published-control projection ledger.",
        ],
        findings: vec![
            "All 24 closing recursive candidates occupy the standard discovered family zero.",
            "They use 12 distinct unsigned supports, with exactly two closing recursive signings per support.",
            "The selected supports occupy a proper subset of the 20 normalizer-conjugacy orbits of family zero.",
            "Within this finite recursion library, normalizer orbits 7 and 17 contain the named-parent source pairs while orbits 1 and 5 contain the unstated-parent source pairs.",
            "Every occupied normalizer orbit contains both ordered embedded-Maxwell signatures.",
            "Every closer contains exactly one four-color block passing the Maxwell gate, but this is exactly the chi0=-1 block.",
            "Both ordered embedded-Maxwell signatures mix named-parent and unstated-parent source pairs.",
            "CT and CV have distinct unsigned supports but the same embedded-Maxwell signature.",
            "The bridge adds a consistency check but does not select a higher-dimensional parent or an R8 family.",
        ],
        boundary: "The Maxwell calculation applies to the two four-color blocks retained by colors one through four. The full eight-color representation is irreducible. This bridge does not promote a block-level Maxwell result to an N=8 enhancement theorem.",
        validation: BridgeValidation {
            recursive_candidates_checked,
            closing_candidates: candidates.len(),
            candidates_mapped_uniquely: candidates.len(),
            distinct_unsigned_supports,
            support_multiplicity_histogram,
            every_selected_support_has_two_closing_signings,
            discovered_families_occupied: family_histogram.len(),
            family_histogram,
            all_closers_occupy_standard_family_zero,
            normalizer_orbits_occupied,
            candidate_normalizer_orbit_histogram,
            support_normalizer_orbit_histogram,
            named_parent_pair_orbit_ids,
            unstated_parent_pair_orbit_ids,
            normalizer_orbit_separates_source_parentage_categories,
            every_occupied_orbit_mixes_ordered_maxwell_signatures,
            every_closer_contains_exactly_one_maxwell_passing_block: maxwell
                .every_closer_contains_exactly_one_maxwell_passing_block,
            each_maxwell_signature_mixes_parentage_categories: maxwell
                .each_signature_mixes_parentage_categories,
            published_ct_projection_matches_control_ledger: projection_matches_control("CT"),
            published_cv_projection_matches_control_ledger: projection_matches_control("CV"),
            ct_and_cv_share_embedded_maxwell_signature,
            ct_and_cv_have_distinct_unsigned_supports,
            ct_normalizer_orbit_id,
            cv_normalizer_orbit_id,
            ct_and_cv_have_distinct_normalizer_orbits,
            passed,
        },
        candidates,
    }
}

pub fn write_artifact(path: &Path) -> BridgeValidation {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create bridge artifact")),
        &artifact,
    )
    .expect("write bridge artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_recursive_closer_maps_into_family_zero_but_maxwell_does_not_select_parentage() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.closing_candidates, 24);
        assert_eq!(artifact.validation.distinct_unsigned_supports, 12);
        assert!(
            artifact
                .validation
                .every_selected_support_has_two_closing_signings
        );
        assert!(artifact.validation.all_closers_occupy_standard_family_zero);
        assert!(
            artifact
                .validation
                .ct_and_cv_share_embedded_maxwell_signature
        );
        assert!(
            artifact
                .validation
                .ct_and_cv_have_distinct_unsigned_supports
        );
    }
}
