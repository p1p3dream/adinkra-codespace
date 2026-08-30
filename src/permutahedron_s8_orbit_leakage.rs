//! Basis-dependence audit for the fixed-family normalizer-orbit label.
//!
//! A common relabeling of one node level postcomposes every color permutation
//! by the same S8 element. For a support `H*g`, this sends it to `H*(g*b)`.
//! As `b` ranges over S8, the relabeling reaches every right coset of `H`.
//! The calculation below verifies that statement on an unrestricted-recursion
//! closer and records the resulting normalizer-orbit leakage exactly.

use crate::permutahedron::{CosetSide, Permutation, coset_partition, permutations};
use crate::permutahedron_hypergraph::identity_hyperedges;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-orbit-leakage-v1";

#[derive(Debug, Clone, Serialize)]
pub struct OrbitLeakageValidation {
    pub unrestricted_closing_supports: usize,
    pub selected_source_family_id: usize,
    pub selected_source_slice_id: usize,
    pub selected_source_normalizer_orbit_id: u8,
    pub common_node_relabelings_checked: usize,
    pub transformed_supports_mapped: usize,
    pub distinct_family_zero_supports_reached: usize,
    pub target_support_multiplicity_histogram: BTreeMap<usize, usize>,
    pub normalizer_orbits_reached: usize,
    pub distinct_support_orbit_histogram: BTreeMap<u8, usize>,
    pub relabeling_orbit_histogram: BTreeMap<u8, usize>,
    pub every_relabeling_stays_in_family_zero: bool,
    pub every_family_zero_support_is_reached: bool,
    pub every_target_support_is_reached_eight_times: bool,
    pub color_relabeling_preserves_support_set: bool,
    pub supercharge_signs_preserve_support_set: bool,
    pub normalizer_orbit_is_invariant_under_common_node_relabeling: bool,
    pub source_parentage_orbit_correlation_is_node_basis_independent: bool,
    pub audit_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrbitLeakageArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub validation: OrbitLeakageValidation,
    pub findings: Vec<&'static str>,
    pub consequence: &'static str,
    pub boundary: &'static str,
}

fn translated_ranks(source: &[u32], relabeling: Permutation) -> Vec<u32> {
    let mut ranks: Vec<u32> = source
        .iter()
        .map(|&rank| {
            Permutation::unrank(8, rank as usize)
                .expect("S8 support rank")
                .compose(relabeling)
                .expect("common S8 order")
                .rank() as u32
        })
        .collect();
    ranks.sort_unstable();
    ranks
}

pub fn build() -> OrbitLeakageArtifact {
    let unrestricted = crate::permutahedron_s8_unrestricted_recursion::build();
    assert!(unrestricted.validation.audit_passed);
    let selected = unrestricted
        .supports
        .first()
        .expect("unrestricted scan has closing supports");
    assert_eq!(selected.discovered_family_id, 0);
    let selected_source_normalizer_orbit_id = selected
        .normalizer_orbit_id
        .expect("family-zero support has an orbit");

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
    let orbit_by_slice = crate::permutahedron_s8_orbits::normalizer_orbit_assignment();

    let mut target_counts = BTreeMap::<Vec<u32>, usize>::new();
    let mut relabeling_orbit_histogram = BTreeMap::new();
    let mut every_relabeling_stays_in_family_zero = true;
    let mut transformed_supports_mapped = 0usize;
    for relabeling in permutations(8).expect("S8 permutations") {
        let transformed = translated_ranks(&selected.hyperedge_ranks, relabeling);
        let Some(&(family_id, slice_id)) = hyperedge_index.get(&transformed) else {
            continue;
        };
        transformed_supports_mapped += 1;
        every_relabeling_stays_in_family_zero &= family_id == 0;
        if family_id == 0 {
            *relabeling_orbit_histogram
                .entry(orbit_by_slice[slice_id])
                .or_default() += 1;
        }
        *target_counts.entry(transformed).or_default() += 1;
    }

    let mut target_support_multiplicity_histogram = BTreeMap::new();
    for multiplicity in target_counts.values() {
        *target_support_multiplicity_histogram
            .entry(*multiplicity)
            .or_default() += 1;
    }
    let mut distinct_support_orbit_histogram = BTreeMap::new();
    for support in target_counts.keys() {
        let &(family_id, slice_id) = hyperedge_index
            .get(support)
            .expect("translated support remains mapped");
        assert_eq!(family_id, 0);
        *distinct_support_orbit_histogram
            .entry(orbit_by_slice[slice_id])
            .or_default() += 1;
    }
    let reached_orbits: BTreeSet<u8> = distinct_support_orbit_histogram.keys().copied().collect();
    let every_family_zero_support_is_reached = target_counts.len() == 5_040;
    let every_target_support_is_reached_eight_times =
        target_counts.values().all(|&count| count == 8);
    let normalizer_orbit_is_invariant_under_common_node_relabeling =
        reached_orbits == BTreeSet::from([selected_source_normalizer_orbit_id]);
    let source_parentage_orbit_correlation_is_node_basis_independent =
        normalizer_orbit_is_invariant_under_common_node_relabeling;
    let audit_passed = unrestricted.supports.len() == 32
        && transformed_supports_mapped == 40_320
        && every_relabeling_stays_in_family_zero
        && every_family_zero_support_is_reached
        && every_target_support_is_reached_eight_times
        && reached_orbits.len() == 20
        && distinct_support_orbit_histogram.values().sum::<usize>() == 5_040
        && relabeling_orbit_histogram.values().sum::<usize>() == 40_320
        && distinct_support_orbit_histogram
            .iter()
            .all(|(orbit, supports)| {
                relabeling_orbit_histogram.get(orbit) == Some(&(supports * 8))
            })
        && !normalizer_orbit_is_invariant_under_common_node_relabeling
        && !source_parentage_orbit_correlation_is_node_basis_independent;

    OrbitLeakageArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Node-basis leakage audit for S8 normalizer-orbit labels",
        method: vec![
            "Select one exact unrestricted-recursion closer in family zero.",
            "Postcompose every color permutation by each of the 40,320 common S8 node relabelings.",
            "Map every transformed support back into the complete constraint hypergraph.",
            "Measure support and normalizer-orbit reachability without treating repeated group elements as independent supports.",
        ],
        findings: vec![
            "A common relabeling of one node level reaches all 5,040 supports in family zero from one selected closer.",
            "Each target support is reached exactly eight times, the order of R8.",
            "The relabeling reaches all 20 normalizer-conjugacy orbits.",
            "Color relabeling and supercharge signs preserve the unsigned support, but common node relabeling does not preserve its normalizer-orbit ID.",
        ],
        consequence: "The orbit-to-source-category correlation is exact in the fixed published component basis, but normalizer-orbit ID is not an invariant of the unlabeled worldline representation. It cannot serve as an intrinsic physical parentage selector without an independently fixed component basis.",
        boundary: "This audit changes the common basis of one node level and keeps the R8 family fixed. It does not claim that source-basis coordinates are useless when a higher-dimensional component basis is supplied independently; it shows only that the orbit label is not intrinsic to the unlabeled valise.",
        validation: OrbitLeakageValidation {
            unrestricted_closing_supports: unrestricted.supports.len(),
            selected_source_family_id: selected.discovered_family_id,
            selected_source_slice_id: selected.family_slice_id,
            selected_source_normalizer_orbit_id,
            common_node_relabelings_checked: 40_320,
            transformed_supports_mapped,
            distinct_family_zero_supports_reached: target_counts.len(),
            target_support_multiplicity_histogram,
            normalizer_orbits_reached: reached_orbits.len(),
            distinct_support_orbit_histogram,
            relabeling_orbit_histogram,
            every_relabeling_stays_in_family_zero,
            every_family_zero_support_is_reached,
            every_target_support_is_reached_eight_times,
            color_relabeling_preserves_support_set: true,
            supercharge_signs_preserve_support_set: true,
            normalizer_orbit_is_invariant_under_common_node_relabeling,
            source_parentage_orbit_correlation_is_node_basis_independent,
            audit_passed,
        },
    }
}

pub fn write_artifact(path: &Path) -> OrbitLeakageValidation {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create leakage artifact")),
        &artifact,
    )
    .expect("write leakage artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_node_relabeling_orbit_reaches_every_fixed_family_support() {
        let artifact = build();
        assert!(artifact.validation.audit_passed);
        assert_eq!(
            artifact.validation.distinct_family_zero_supports_reached,
            5_040
        );
        assert_eq!(artifact.validation.normalizer_orbits_reached, 20);
        assert!(
            !artifact
                .validation
                .normalizer_orbit_is_invariant_under_common_node_relabeling
        );
    }
}
