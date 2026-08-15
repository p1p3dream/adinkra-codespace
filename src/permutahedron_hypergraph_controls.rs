//! Projection of published S8 control systems onto the constraint hypergraph.
//!
//! This bridge keeps unsigned family membership, availability of Garden
//! signings, and closure of the published sign assignment separate.

use crate::lr_matrix::AdinkraRep;
use crate::permutahedron::{coset_partition, CosetSide, Permutation};
use crate::permutahedron_fixtures::{R8_DIADEM_OCTET, S8_REPRESENTATION_OCTETS};
use crate::permutahedron_garden::solve_garden_signing;
use crate::permutahedron_hypergraph::identity_hyperedges;
use crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;
const SCHEMA_VERSION: &str = "permutahedron-hypergraph-physical-controls-v1";

#[derive(Debug, Clone, Serialize)]
pub struct ControlRecord {
    pub id: &'static str,
    pub interpretation: &'static str,
    pub source: &'static str,
    pub positive_garden_control: bool,
    pub published_sign_status: &'static str,
    pub hyperedge_ranks: Vec<u32>,
    pub discovered_family_id: usize,
    pub family_slice_id: usize,
    pub unsigned_support_has_garden_signings: bool,
    pub published_or_certified_assignment_closes: bool,
    pub valid_signing_class: &'static str,
    pub published_assignment_belongs_to_valid_signing_class: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlProjectionValidation {
    pub controls: usize,
    pub positive_controls: usize,
    pub negative_controls: usize,
    pub controls_mapped_uniquely: usize,
    pub control_family_count: usize,
    pub positive_control_family_count: usize,
    pub negative_control_family_count: usize,
    pub positive_and_negative_controls_share_a_family: bool,
    pub all_unsigned_supports_are_signable: bool,
    pub printed_closure_matches_published_status: bool,
    pub family_membership_separates_published_closure: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlProjectionArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub controls: Vec<ControlRecord>,
    pub validation: ControlProjectionValidation,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn permutation(one_line: &[u8; 8]) -> Permutation {
    Permutation::new(one_line).expect("published S8 permutation")
}

fn sorted_ranks(octet: &[Permutation]) -> Vec<u32> {
    let mut ranks: Vec<u32> = octet
        .iter()
        .map(|permutation| permutation.rank() as u32)
        .collect();
    ranks.sort_unstable();
    ranks
}

fn signs_from_boolean_factors(factors: &[u8; 8]) -> Vec<i8> {
    factors
        .iter()
        .flat_map(|factor| (0..D).map(move |row| if factor & (1 << row) == 0 { 1 } else { -1 }))
        .collect()
}

fn verifies_garden(octet: &[Permutation], signs: &[i8]) -> bool {
    let color_permutations: Vec<Vec<usize>> = octet
        .iter()
        .map(|permutation| {
            permutation
                .as_slice()
                .iter()
                .map(|&value| usize::from(value - 1))
                .collect()
        })
        .collect();
    AdinkraRep::from_parts(N, D, &color_permutations, signs).verify_garden_algebra()
}

pub fn build() -> ControlProjectionArtifact {
    let families = identity_hyperedges(N);
    let mut hyperedge_index = BTreeMap::new();
    for (family_id, family) in families.iter().enumerate() {
        let partition = coset_partition(family, CosetSide::Right)
            .expect("discovered identity block is a subgroup");
        for (slice_id, hyperedge) in partition.slices.into_iter().enumerate() {
            assert!(
                hyperedge_index
                    .insert(hyperedge, (family_id, slice_id))
                    .is_none(),
                "hyperedges are unique across discovered families"
            );
        }
    }

    let labels = [
        ("CC", "chiral + chiral", false),
        ("CT", "chiral + tensor", true),
        ("CV", "chiral + vector", true),
        ("TT", "tensor + tensor", false),
        ("TV", "tensor + vector", false),
        ("VV", "vector + vector", false),
    ];
    let mut controls = Vec::with_capacity(7);
    for (sector, &(id, interpretation, positive)) in labels.iter().enumerate() {
        let octet: Vec<Permutation> = S8_REPRESENTATION_OCTETS[sector]
            .permutations
            .iter()
            .map(permutation)
            .collect();
        let ranks = sorted_ranks(&octet);
        let &(family_id, slice_id) = hyperedge_index
            .get(&ranks)
            .expect("published support is a discovered hyperedge");
        let solution = solve_garden_signing(&octet);
        let published_closes = verifies_garden(
            &octet,
            &signs_from_boolean_factors(&S8_BASE_BOOLEAN_FACTORS[sector]),
        );
        controls.push(ControlRecord {
            id,
            interpretation,
            source: "arXiv:2012.14015v7, Eqs. (5.1)-(5.6)",
            positive_garden_control: positive,
            published_sign_status: if positive {
                "published Garden closure"
            } else {
                "published nonclosure"
            },
            hyperedge_ranks: ranks,
            discovered_family_id: family_id,
            family_slice_id: slice_id,
            unsigned_support_has_garden_signings: solution.feasible,
            published_or_certified_assignment_closes: published_closes,
            valid_signing_class: "unlabeled-color-signed-graph/class-00",
            published_assignment_belongs_to_valid_signing_class: published_closes,
        });
    }

    let octet: Vec<Permutation> = R8_DIADEM_OCTET.iter().map(permutation).collect();
    let ranks = sorted_ranks(&octet);
    let &(family_id, slice_id) = hyperedge_index
        .get(&ranks)
        .expect("published O support is a discovered hyperedge");
    let solution = solve_garden_signing(&octet);
    let certified_closes = verifies_garden(&octet, &solution.canonical_signs);
    controls.push(ControlRecord {
        id: "O",
        interpretation: "R8 / Diadem(8)",
        source: "arXiv:2012.14015v7 Eq. (5.7); arXiv:2304.09830v2 Eq. (3.23)",
        positive_garden_control: true,
        published_sign_status: "published Garden closure",
        hyperedge_ranks: ranks,
        discovered_family_id: family_id,
        family_slice_id: slice_id,
        unsigned_support_has_garden_signings: solution.feasible,
        published_or_certified_assignment_closes: certified_closes,
        valid_signing_class: "unlabeled-color-signed-graph/class-00",
        published_assignment_belongs_to_valid_signing_class: certified_closes,
    });
    controls.sort_by_key(|control| control.id);

    let positive: Vec<&ControlRecord> = controls
        .iter()
        .filter(|control| control.positive_garden_control)
        .collect();
    let negative: Vec<&ControlRecord> = controls
        .iter()
        .filter(|control| !control.positive_garden_control)
        .collect();
    let control_families: BTreeSet<usize> = controls
        .iter()
        .map(|control| control.discovered_family_id)
        .collect();
    let positive_families: BTreeSet<usize> = positive
        .iter()
        .map(|control| control.discovered_family_id)
        .collect();
    let negative_families: BTreeSet<usize> = negative
        .iter()
        .map(|control| control.discovered_family_id)
        .collect();
    let share_family = !positive_families.is_disjoint(&negative_families);
    let all_supports_signable = controls
        .iter()
        .all(|control| control.unsigned_support_has_garden_signings);
    let closure_matches = controls.iter().all(|control| {
        control.published_or_certified_assignment_closes == control.positive_garden_control
    });
    let family_separates = positive_families.is_disjoint(&negative_families);
    let passed = controls.len() == 7
        && positive.len() == 3
        && negative.len() == 4
        && control_families.len() == 1
        && positive_families.len() == 1
        && negative_families.len() == 1
        && share_family
        && all_supports_signable
        && closure_matches
        && !family_separates;

    let validation = ControlProjectionValidation {
        controls: controls.len(),
        positive_controls: positive.len(),
        negative_controls: negative.len(),
        controls_mapped_uniquely: controls.len(),
        control_family_count: control_families.len(),
        positive_control_family_count: positive_families.len(),
        negative_control_family_count: negative_families.len(),
        positive_and_negative_controls_share_a_family: share_family,
        all_unsigned_supports_are_signable: all_supports_signable,
        printed_closure_matches_published_status: closure_matches,
        family_membership_separates_published_closure: family_separates,
        passed,
    };

    let shared_family = controls[0].discovered_family_id;
    ControlProjectionArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Published S8 controls projected onto the constraint hypergraph",
        method: vec![
            "Index all 151,200 discovered hyperedges by their sorted permutation ranks.",
            "Map the six published paired supports and the O / Diadem(8) support into that index.",
            "Verify Garden feasibility of every unsigned support independently.",
            "Verify the printed Boolean-factor assignment for CC, CT, CV, TT, TV, and VV directly, while using a certified Garden assignment for O.",
            "Keep unsigned family membership separate from closure of the particular published sign assignment.",
        ],
        findings: vec![
            format!(
                "O, CT, and CV all lie in discovered family {shared_family}, but CC, TT, TV, and VV lie in that same family as well."
            ),
            "Every one of the seven unsigned supports admits Garden signs, including the four supports whose published Boolean factors do not close.".into(),
            "The particular published signs close for O, CT, and CV and fail for CC, TT, TV, and VV, reproducing the control labels.".into(),
            "Subgroup-family membership cannot separate published closure or higher-dimensional parentage because the positive and negative controls share one family.".into(),
        ],
        boundary: "This bridge verifies support membership and one-dimensional Garden closure. The published closure labels are controls, not a general enhancement theorem. Distinguishing four-dimensional parentage still requires height, gauge, phantom-linkage, or direct enhancement data that are absent from the unsigned hypergraph.",
        controls,
        validation,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ControlProjectionValidation {
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
    fn positive_and_negative_published_controls_share_one_discovered_family() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.control_family_count, 1);
        assert!(
            artifact
                .validation
                .positive_and_negative_controls_share_a_family
        );
        assert!(artifact.validation.all_unsigned_supports_are_signable);
        assert!(artifact.validation.printed_closure_matches_published_status);
        assert!(
            !artifact
                .validation
                .family_membership_separates_published_closure
        );
    }
}
