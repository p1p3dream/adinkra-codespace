//! Maxwell gauge-enhancement classification of the two four-color blocks
//! retained by each closing signed S8 recursion candidate.
//!
//! This is deliberately not an N=8 enhancement test.  Colors 1-4 of the
//! recursive 8-by-8 construction are block diagonal.  Their upper and lower
//! 4-by-4 blocks reproduce the two signed S4 inputs, including the relative
//! color order of the second input.  We extract those blocks and apply the
//! validated N=4 Maxwell worldline search to each one.

use crate::chromochar::chromochar_antisym;
use crate::lr_matrix::AdinkraRep;
use crate::maxwell_phantom::WorldlineLinkage;
use crate::maxwell_worldline_search::search_worldline;
use crate::permutahedron_s8_signed_recursion::{
    aligned_recursive_boolean_factors, aligned_recursive_permutations, build_rep, cyclic_mask,
    exact_published_match, S4_RECURSION_LABELS,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct EmbeddedS4Record {
    pub source_label: &'static str,
    pub color_order_one_based: [usize; 4],
    pub garden_closes: bool,
    pub chi0: i64,
    pub charge_zero_normalized_candidates: usize,
    pub maxwell_gauge_enhancing_frames: usize,
    pub passes_maxwell_gate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClosingCandidateRecord {
    pub id: String,
    pub first_label: &'static str,
    pub second_label: &'static str,
    pub second_color_order_one_based: [usize; 4],
    pub start_position_one_based: usize,
    pub exact_published_system_match: Option<&'static str>,
    pub first_embedded_s4: EmbeddedS4Record,
    pub second_embedded_s4: EmbeddedS4Record,
    pub ordered_chi0_signature: [i64; 2],
    pub ordered_maxwell_signature: [bool; 2],
}

#[derive(Clone, Debug, Serialize)]
pub struct SignatureCount {
    pub ordered_chi0_signature: [i64; 2],
    pub ordered_maxwell_signature: [bool; 2],
    pub candidates: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MaxwellS8SubalgebraArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<&'static str>,
    pub candidates: Vec<ClosingCandidateRecord>,
    pub signature_counts: Vec<SignatureCount>,
    pub recursive_candidates_checked: usize,
    pub closing_candidates: usize,
    pub embedded_s4_blocks_checked: usize,
    pub embedded_s4_blocks_closing: usize,
    pub embedded_gate_passes_if_and_only_if_chi0_is_minus_one: bool,
    pub every_closer_pairs_opposite_chi0: bool,
    pub every_closer_contains_exactly_one_maxwell_passing_block: bool,
    pub maxwell_signature_is_exact_relabeling_of_ordered_chi0: bool,
    pub distinct_ordered_signatures: usize,
    pub each_signature_mixes_parentage_categories: bool,
    pub ct_ordered_maxwell_signature: Option<[bool; 2]>,
    pub cv_ordered_maxwell_signature: Option<[bool; 2]>,
    pub ct_and_cv_are_distinguished: bool,
    pub passed: bool,
    pub conclusion: &'static str,
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

fn extract_block(rep: &AdinkraRep, offset: usize) -> AdinkraRep {
    assert_eq!((rep.n, rep.d), (8, 8));
    assert!(offset == 0 || offset == 4);
    let mut permutations = Vec::with_capacity(4);
    let mut dashing = Vec::with_capacity(16);
    for color in 0..4 {
        let matrix = &rep.l_matrices[color];
        let mut permutation = Vec::with_capacity(4);
        for row in 0..4 {
            let source_row = offset + row;
            let target = usize::from(matrix.perm[source_row]);
            assert!((offset..offset + 4).contains(&target));
            permutation.push(target - offset);
            dashing.push(matrix.sign[source_row]);
        }
        permutations.push(permutation);
    }
    AdinkraRep::from_parts(4, 4, &permutations, &dashing)
}

fn linkage(rep: &AdinkraRep) -> WorldlineLinkage {
    assert_eq!((rep.n, rep.d), (4, 4));
    std::array::from_fn(|charge| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let matrix = &rep.l_matrices[charge];
                if usize::from(matrix.perm[row]) == column {
                    matrix.sign[row]
                } else {
                    0
                }
            })
        })
    })
}

fn classify(
    source_label: &'static str,
    color_order_one_based: [usize; 4],
    rep: &AdinkraRep,
) -> EmbeddedS4Record {
    let garden_closes = rep.verify_garden_algebra();
    let raw_chi0 = chromochar_antisym(rep, 0, 1, 2, 3);
    assert_eq!(raw_chi0 % 96, 0);
    let chi0 = raw_chi0 / 96;
    let search = search_worldline("embedded S4 block", &linkage(rep));
    EmbeddedS4Record {
        source_label,
        color_order_one_based,
        garden_closes,
        chi0,
        charge_zero_normalized_candidates: search.charge_zero_normalized_candidates,
        maxwell_gauge_enhancing_frames: search.gauge_enhancing_candidates,
        passes_maxwell_gate: search.gauge_enhancing_candidates > 0,
    }
}

pub fn build() -> MaxwellS8SubalgebraArtifact {
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
                    let first_block = extract_block(&rep, 0);
                    let second_block = extract_block(&rep, 4);
                    let first_record =
                        classify(S4_RECURSION_LABELS[first], [1, 2, 3, 4], &first_block);
                    let second_order = alignment.map(|color| color + 1);
                    let second_record =
                        classify(S4_RECURSION_LABELS[second], second_order, &second_block);
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
                    candidates.push(ClosingCandidateRecord {
                        id,
                        first_label: S4_RECURSION_LABELS[first],
                        second_label: S4_RECURSION_LABELS[second],
                        second_color_order_one_based: second_order,
                        start_position_one_based: start + 1,
                        exact_published_system_match: exact_published_match(
                            &permutations,
                            &factors,
                        ),
                        ordered_chi0_signature: [first_record.chi0, second_record.chi0],
                        ordered_maxwell_signature: [
                            first_record.passes_maxwell_gate,
                            second_record.passes_maxwell_gate,
                        ],
                        first_embedded_s4: first_record,
                        second_embedded_s4: second_record,
                    });
                }
            }
        }
    }

    let mut counts = BTreeMap::<([i64; 2], [bool; 2]), usize>::new();
    for candidate in &candidates {
        *counts
            .entry((
                candidate.ordered_chi0_signature,
                candidate.ordered_maxwell_signature,
            ))
            .or_default() += 1;
    }
    let signature_counts = counts
        .into_iter()
        .map(
            |((ordered_chi0_signature, ordered_maxwell_signature), candidates)| SignatureCount {
                ordered_chi0_signature,
                ordered_maxwell_signature,
                candidates,
            },
        )
        .collect::<Vec<_>>();
    let embedded_s4_blocks_checked = candidates.len() * 2;
    let embedded_s4_blocks_closing = candidates
        .iter()
        .flat_map(|candidate| [&candidate.first_embedded_s4, &candidate.second_embedded_s4])
        .filter(|block| block.garden_closes)
        .count();
    let embedded_gate_passes_if_and_only_if_chi0_is_minus_one = candidates
        .iter()
        .flat_map(|candidate| [&candidate.first_embedded_s4, &candidate.second_embedded_s4])
        .all(|block| block.passes_maxwell_gate == (block.chi0 == -1));
    let every_closer_pairs_opposite_chi0 = candidates
        .iter()
        .all(|candidate| candidate.ordered_chi0_signature.iter().sum::<i64>() == 0);
    let every_closer_contains_exactly_one_maxwell_passing_block =
        candidates.iter().all(|candidate| {
            candidate
                .ordered_maxwell_signature
                .iter()
                .filter(|&&passes| passes)
                .count()
                == 1
        });
    let maxwell_signature_is_exact_relabeling_of_ordered_chi0 =
        candidates.iter().all(|candidate| {
            (0..2).all(|index| {
                candidate.ordered_maxwell_signature[index]
                    == (candidate.ordered_chi0_signature[index] == -1)
            })
        });
    let named_parent_pair = |candidate: &ClosingCandidateRecord| {
        let named = |label| matches!(label, "CM" | "TM" | "VM");
        named(candidate.first_label) && named(candidate.second_label)
    };
    let each_signature_mixes_parentage_categories = signature_counts.iter().all(|signature| {
        let members = candidates.iter().filter(|candidate| {
            candidate.ordered_chi0_signature == signature.ordered_chi0_signature
                && candidate.ordered_maxwell_signature == signature.ordered_maxwell_signature
        });
        let categories = members
            .map(named_parent_pair)
            .collect::<std::collections::BTreeSet<_>>();
        categories.len() == 2
    });
    let published_signature = |name| {
        candidates
            .iter()
            .find(|candidate| candidate.exact_published_system_match == Some(name))
            .map(|candidate| candidate.ordered_maxwell_signature)
    };
    let ct_ordered_maxwell_signature = published_signature("CT");
    let cv_ordered_maxwell_signature = published_signature("CV");
    let ct_and_cv_are_distinguished = ct_ordered_maxwell_signature.is_some()
        && cv_ordered_maxwell_signature.is_some()
        && ct_ordered_maxwell_signature != cv_ordered_maxwell_signature;
    let distinct_ordered_signatures = signature_counts.len();
    let passed = recursive_candidates_checked == 30 * 24 * 8
        && candidates.len() == 24
        && embedded_s4_blocks_checked == 48
        && embedded_s4_blocks_closing == 48
        && embedded_gate_passes_if_and_only_if_chi0_is_minus_one
        && every_closer_pairs_opposite_chi0
        && every_closer_contains_exactly_one_maxwell_passing_block
        && maxwell_signature_is_exact_relabeling_of_ordered_chi0
        && distinct_ordered_signatures == 2
        && each_signature_mixes_parentage_categories
        && ct_ordered_maxwell_signature.is_some()
        && cv_ordered_maxwell_signature.is_some();
    MaxwellS8SubalgebraArtifact {
        schema_version: "maxwell-s8-subalgebra-scan-v1",
        title: "Maxwell classification of the embedded four-color blocks in the closing signed S8 recursion candidates",
        sources: vec![
            "arXiv:2304.09830v2 Eqs. (2.17)-(2.25)",
            "arXiv:0907.3605 Section 5.5 and Eq. (5.11)",
        ],
        candidates,
        signature_counts,
        recursive_candidates_checked,
        closing_candidates: 24,
        embedded_s4_blocks_checked,
        embedded_s4_blocks_closing,
        embedded_gate_passes_if_and_only_if_chi0_is_minus_one,
        every_closer_pairs_opposite_chi0,
        every_closer_contains_exactly_one_maxwell_passing_block,
        maxwell_signature_is_exact_relabeling_of_ordered_chi0,
        distinct_ordered_signatures,
        each_signature_mixes_parentage_categories,
        ct_ordered_maxwell_signature,
        cv_ordered_maxwell_signature,
        ct_and_cv_are_distinguished,
        passed,
        conclusion: "Every closer pairs one chi0=+1 block with one chi0=-1 block. The Maxwell gate selects exactly the chi0=-1 member, so it contributes no information beyond ordered chi0 on this finite library. Both ordered signatures mix named four-dimensional parents with sources having no stated parent, and the signature does not distinguish the published CT and CV anchors.",
        boundary: "Colors 1-4 are block diagonal in this recursion, so the two extracted 4-by-4 systems are genuine Garden subrepresentations of that four-color restriction. The full eight-color representation is irreducible and does not split into these blocks. No claim is made that either block alone determines higher-dimensional parentage of the complete N=8 system.",
    }
}

pub fn write_artifact(path: &Path) -> MaxwellS8SubalgebraArtifact {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create Maxwell S8 subalgebra artifact")),
        &artifact,
    )
    .expect("write Maxwell S8 subalgebra artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_closer_has_two_certified_s4_blocks() {
        let artifact = build();
        assert_eq!(artifact.recursive_candidates_checked, 5_760);
        assert_eq!(artifact.closing_candidates, 24);
        assert_eq!(artifact.embedded_s4_blocks_checked, 48);
        assert_eq!(artifact.embedded_s4_blocks_closing, 48);
        assert!(artifact.embedded_gate_passes_if_and_only_if_chi0_is_minus_one);
        assert!(artifact.every_closer_pairs_opposite_chi0);
        assert!(artifact.every_closer_contains_exactly_one_maxwell_passing_block);
        assert!(artifact.maxwell_signature_is_exact_relabeling_of_ordered_chi0);
        assert_eq!(artifact.distinct_ordered_signatures, 2);
        assert_eq!(artifact.signature_counts[0].candidates, 12);
        assert_eq!(artifact.signature_counts[1].candidates, 12);
        assert!(artifact.each_signature_mixes_parentage_categories);
        assert_eq!(artifact.ct_ordered_maxwell_signature, Some([false, true]));
        assert_eq!(artifact.cv_ordered_maxwell_signature, Some([false, true]));
        assert!(!artifact.ct_and_cv_are_distinguished);
        assert!(artifact.passed);
    }
}
