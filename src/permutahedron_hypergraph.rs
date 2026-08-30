//! Exact constraint-hypergraph discovery for the S4 and S8 permutahedra.
//!
//! A candidate unsigned Garden block is a set of permutations for which every
//! relative permutation is a fixed-point-free involution.  Any such block can
//! be right-translated so that it contains the identity.  The remaining
//! elements then form a clique in the compatibility graph on fixed-point-free
//! involutions.  Enumerating those small identity cliques discovers the
//! hyperedges without supplying the published V4 or R8 coset labels.

use crate::permutahedron::{
    CosetSide, Permutation, coset_partition, factorial, permutations, validate_subgroup,
    vierergruppe,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-constraint-hypergraph-v1";

#[derive(Debug, Clone, Serialize)]
pub struct ConstraintHypergraphSummary {
    pub n: usize,
    pub vertex_count: usize,
    pub identity_compatible_candidates: usize,
    pub candidate_compatibility_edges: usize,
    pub candidate_degree_histogram: BTreeMap<usize, usize>,
    pub clique_search_nodes: usize,
    pub identity_hyperedges: usize,
    pub hyperedge_size: usize,
    pub discovered_parallel_classes: usize,
    pub hyperedges_per_parallel_class: usize,
    pub distinct_hyperedges: usize,
    pub incidences: usize,
    pub vertex_degree_histogram: BTreeMap<usize, usize>,
    pub all_identity_hyperedges_are_subgroups: bool,
    pub every_parallel_class_is_complete: bool,
    pub hyperedges_recover_source_family: bool,
    pub duplicate_hyperedges_across_parallel_classes: usize,
    pub subgroup_intersection_order_histogram: BTreeMap<usize, usize>,
    pub positive_hyperedge_intersection_histogram: BTreeMap<usize, usize>,
    pub overlapping_hyperedges_per_hyperedge_histogram: BTreeMap<usize, usize>,
    pub matches_reference_cosets: Option<bool>,
    pub sample_hyperedges: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HypergraphValidation {
    pub s4_identity_candidates: usize,
    pub s4_identity_hyperedges: usize,
    pub s4_distinct_hyperedges: usize,
    pub s4_matches_reference_cosets: bool,
    pub s8_identity_candidates: usize,
    pub s8_candidate_degree: usize,
    pub s8_identity_hyperedges: usize,
    pub s8_distinct_hyperedges: usize,
    pub s8_vertex_degree: usize,
    pub s8_overlapping_hyperedges_per_hyperedge: usize,
    pub both_family_recovery_checks_pass: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConstraintHypergraphArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub s4: ConstraintHypergraphSummary,
    pub s8: ConstraintHypergraphSummary,
    pub validation: HypergraphValidation,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

#[derive(Debug)]
struct IdentityDiscovery {
    candidates: Vec<Permutation>,
    adjacency: Vec<u128>,
    search_nodes: usize,
    blocks: Vec<Vec<Permutation>>,
}

fn histogram(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_insert(0) += 1;
    }
    result
}

fn is_fixed_point_free_involution(permutation: Permutation) -> bool {
    let identity = Permutation::identity(permutation.n()).expect("validated permutation order");
    permutation != identity
        && permutation
            .as_slice()
            .iter()
            .enumerate()
            .all(|(index, &image)| usize::from(image) != index + 1)
        && permutation
            .compose(permutation)
            .expect("same permutation order")
            == identity
}

fn unsigned_garden_compatible(left: Permutation, right: Permutation) -> bool {
    let relative = left
        .compose(right.inverse())
        .expect("same permutation order");
    is_fixed_point_free_involution(relative)
}

fn enumerate_cliques(
    adjacency: &[u128],
    target_size: usize,
    selected: &mut Vec<usize>,
    mut candidates: u128,
    output: &mut Vec<Vec<usize>>,
    search_nodes: &mut usize,
) {
    *search_nodes += 1;
    if selected.len() == target_size {
        output.push(selected.clone());
        return;
    }
    let needed = target_size - selected.len();
    while candidates != 0 {
        if (candidates.count_ones() as usize) < needed {
            return;
        }
        let bit = candidates & candidates.wrapping_neg();
        candidates ^= bit;
        let vertex = bit.trailing_zeros() as usize;
        selected.push(vertex);
        enumerate_cliques(
            adjacency,
            target_size,
            selected,
            candidates & adjacency[vertex],
            output,
            search_nodes,
        );
        selected.pop();
    }
}

fn discover_identity_hyperedges(n: usize) -> IdentityDiscovery {
    let candidates: Vec<Permutation> = permutations(n)
        .expect("supported symmetric group")
        .filter(|&permutation| is_fixed_point_free_involution(permutation))
        .collect();
    assert!(
        candidates.len() <= 128,
        "the exact bitset clique search supports at most 128 candidates"
    );

    let mut adjacency = vec![0u128; candidates.len()];
    for left in 0..candidates.len() {
        for right in left + 1..candidates.len() {
            if unsigned_garden_compatible(candidates[left], candidates[right]) {
                adjacency[left] |= 1u128 << right;
                adjacency[right] |= 1u128 << left;
            }
        }
    }

    let all_candidates = if candidates.len() == 128 {
        u128::MAX
    } else {
        (1u128 << candidates.len()) - 1
    };
    let mut clique_indices = Vec::new();
    let mut search_nodes = 0;
    enumerate_cliques(
        &adjacency,
        n - 1,
        &mut Vec::with_capacity(n - 1),
        all_candidates,
        &mut clique_indices,
        &mut search_nodes,
    );

    let identity = Permutation::identity(n).expect("supported symmetric group");
    let blocks = clique_indices
        .into_iter()
        .map(|indices| {
            let mut block = Vec::with_capacity(n);
            block.push(identity);
            block.extend(indices.into_iter().map(|index| candidates[index]));
            block.sort_unstable_by_key(|permutation| permutation.rank());
            block
        })
        .collect();

    IdentityDiscovery {
        candidates,
        adjacency,
        search_nodes,
        blocks,
    }
}

/// Discover every compatible size-`n` block containing the identity without
/// supplying V4 or R8 labels.  The returned blocks use deterministic rank
/// order and are intended for exact downstream audits.
pub(crate) fn identity_hyperedges(n: usize) -> Vec<Vec<Permutation>> {
    discover_identity_hyperedges(n).blocks
}

fn subgroup_key(elements: &[Permutation]) -> Vec<u32> {
    let mut key: Vec<u32> = elements
        .iter()
        .map(|permutation| permutation.rank() as u32)
        .collect();
    key.sort_unstable();
    key
}

fn recovered_subgroup_key(n: usize, hyperedge: &[u32]) -> Vec<u32> {
    let seed = Permutation::unrank(n, hyperedge[0] as usize).expect("bounded rank");
    let inverse = seed.inverse();
    let mut key: Vec<u32> = hyperedge
        .iter()
        .map(|&rank| {
            Permutation::unrank(n, rank as usize)
                .expect("bounded rank")
                .compose(inverse)
                .expect("same permutation order")
                .rank() as u32
        })
        .collect();
    key.sort_unstable();
    key
}

fn build_summary(n: usize) -> ConstraintHypergraphSummary {
    let discovery = discover_identity_hyperedges(n);
    let vertex_count = factorial(n).expect("supported symmetric group");
    let candidate_compatibility_edges = discovery
        .adjacency
        .iter()
        .map(|neighbors| neighbors.count_ones() as usize)
        .sum::<usize>()
        / 2;
    let candidate_degree_histogram = histogram(
        discovery
            .adjacency
            .iter()
            .map(|neighbors| neighbors.count_ones() as usize),
    );
    let all_identity_hyperedges_are_subgroups = discovery
        .blocks
        .iter()
        .all(|block| validate_subgroup(block).valid);

    let family_keys: Vec<Vec<u32>> = discovery
        .blocks
        .iter()
        .map(|block| subgroup_key(block))
        .collect();
    let mut distinct_hyperedges = BTreeSet::new();
    let mut vertex_degree = vec![0usize; vertex_count];
    let mut duplicate_hyperedges = 0usize;
    let mut all_complete = true;
    let mut family_recovery = true;
    let mut hyperedges_per_parallel_class = None;
    let mut sample_hyperedges = Vec::new();

    for (family_id, block) in discovery.blocks.iter().enumerate() {
        let partition = coset_partition(block, CosetSide::Right).expect("discovered subgroup");
        all_complete &= partition.complete_cover
            && partition.slice_count * partition.slice_size == vertex_count;
        match hyperedges_per_parallel_class {
            None => hyperedges_per_parallel_class = Some(partition.slice_count),
            Some(expected) => all_complete &= expected == partition.slice_count,
        }
        for hyperedge in partition.slices {
            for &rank in &hyperedge {
                vertex_degree[rank as usize] += 1;
            }
            family_recovery &= recovered_subgroup_key(n, &hyperedge) == family_keys[family_id];
            if sample_hyperedges.len() < 8 {
                sample_hyperedges.push(hyperedge.clone());
            }
            if !distinct_hyperedges.insert(hyperedge) {
                duplicate_hyperedges += 1;
            }
        }
    }

    let mut subgroup_intersections = BTreeMap::new();
    let mut positive_hyperedge_intersections = BTreeMap::new();
    let mut overlaps_per_family = vec![0usize; family_keys.len()];
    for left in 0..family_keys.len() {
        let left_set: BTreeSet<u32> = family_keys[left].iter().copied().collect();
        for right in left + 1..family_keys.len() {
            let intersection_order = family_keys[right]
                .iter()
                .filter(|rank| left_set.contains(rank))
                .count();
            *subgroup_intersections
                .entry(intersection_order)
                .or_insert(0) += 1;

            // If right cosets Hx and Kx meet, their intersection has size
            // |H intersect K|.  There are |S_n|/|H intersect K| such meeting
            // coset pairs for this pair of subgroup families.
            *positive_hyperedge_intersections
                .entry(intersection_order)
                .or_insert(0) += vertex_count / intersection_order;
            overlaps_per_family[left] += n / intersection_order;
            overlaps_per_family[right] += n / intersection_order;
        }
    }

    let overlaps_per_hyperedge_histogram = overlaps_per_family
        .into_iter()
        .map(|overlaps| (overlaps, hyperedges_per_parallel_class.unwrap_or(0)))
        .fold(BTreeMap::new(), |mut acc, (overlaps, hyperedges)| {
            *acc.entry(overlaps).or_insert(0) += hyperedges;
            acc
        });

    let matches_reference_cosets = if n == 4 {
        let reference: BTreeSet<Vec<u32>> = coset_partition(&vierergruppe(), CosetSide::Right)
            .expect("V4 subgroup")
            .slices
            .into_iter()
            .collect();
        Some(reference == distinct_hyperedges)
    } else {
        None
    };

    ConstraintHypergraphSummary {
        n,
        vertex_count,
        identity_compatible_candidates: discovery.candidates.len(),
        candidate_compatibility_edges,
        candidate_degree_histogram,
        clique_search_nodes: discovery.search_nodes,
        identity_hyperedges: discovery.blocks.len(),
        hyperedge_size: n,
        discovered_parallel_classes: discovery.blocks.len(),
        hyperedges_per_parallel_class: hyperedges_per_parallel_class.unwrap_or(0),
        distinct_hyperedges: distinct_hyperedges.len(),
        incidences: distinct_hyperedges.len() * n,
        vertex_degree_histogram: histogram(vertex_degree),
        all_identity_hyperedges_are_subgroups,
        every_parallel_class_is_complete: all_complete,
        hyperedges_recover_source_family: family_recovery,
        duplicate_hyperedges_across_parallel_classes: duplicate_hyperedges,
        subgroup_intersection_order_histogram: subgroup_intersections,
        positive_hyperedge_intersection_histogram: positive_hyperedge_intersections,
        overlapping_hyperedges_per_hyperedge_histogram: overlaps_per_hyperedge_histogram,
        matches_reference_cosets,
        sample_hyperedges,
    }
}

pub fn build() -> ConstraintHypergraphArtifact {
    let s4 = build_summary(4);
    let s8 = build_summary(8);

    let validation = HypergraphValidation {
        s4_identity_candidates: s4.identity_compatible_candidates,
        s4_identity_hyperedges: s4.identity_hyperedges,
        s4_distinct_hyperedges: s4.distinct_hyperedges,
        s4_matches_reference_cosets: s4.matches_reference_cosets == Some(true),
        s8_identity_candidates: s8.identity_compatible_candidates,
        s8_candidate_degree: s8
            .candidate_degree_histogram
            .keys()
            .copied()
            .next()
            .unwrap_or(0),
        s8_identity_hyperedges: s8.identity_hyperedges,
        s8_distinct_hyperedges: s8.distinct_hyperedges,
        s8_vertex_degree: s8
            .vertex_degree_histogram
            .keys()
            .copied()
            .next()
            .unwrap_or(0),
        s8_overlapping_hyperedges_per_hyperedge: s8
            .overlapping_hyperedges_per_hyperedge_histogram
            .keys()
            .copied()
            .next()
            .unwrap_or(0),
        both_family_recovery_checks_pass: s4.hyperedges_recover_source_family
            && s8.hyperedges_recover_source_family,
        passed: false,
    };
    let mut validation = validation;
    validation.passed = validation.s4_identity_candidates == 3
        && validation.s4_identity_hyperedges == 1
        && validation.s4_distinct_hyperedges == 6
        && validation.s4_matches_reference_cosets
        && validation.s8_identity_candidates == 105
        && s8.candidate_degree_histogram == BTreeMap::from([(12, 105)])
        && validation.s8_identity_hyperedges == 30
        && validation.s8_distinct_hyperedges == 151_200
        && validation.s8_vertex_degree == 30
        && s8.vertex_degree_histogram == BTreeMap::from([(30, 40_320)])
        && validation.s8_overlapping_hyperedges_per_hyperedge == 204
        && s8.overlapping_hyperedges_per_hyperedge_histogram == BTreeMap::from([(204, 151_200)])
        && s8.subgroup_intersection_order_histogram == BTreeMap::from([(1, 330), (2, 105)])
        && s8.positive_hyperedge_intersection_histogram
            == BTreeMap::from([(1, 13_305_600), (2, 2_116_800)])
        && s4.all_identity_hyperedges_are_subgroups
        && s8.all_identity_hyperedges_are_subgroups
        && s4.every_parallel_class_is_complete
        && s8.every_parallel_class_is_complete
        && s4.duplicate_hyperedges_across_parallel_classes == 0
        && s8.duplicate_hyperedges_across_parallel_classes == 0
        && validation.both_family_recovery_checks_pass;

    ConstraintHypergraphArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Exact S4 and S8 unsigned Garden constraint hypergraphs",
        method: vec![
            "Enumerate fixed-point-free involutions relative to the identity without loading V4 or R8 labels.",
            "Connect two candidates exactly when their relative permutation is also a fixed-point-free involution.",
            "Enumerate cliques of size n-1; adjoining the identity yields every compatible size-n hyperedge up to right translation.",
            "Validate the discovered identity hyperedges as subgroups only after clique discovery, then generate and deduplicate all right translates.",
            "Recover the identity subgroup from every translated hyperedge and audit parallel-class covers, incidence degrees, subgroup intersections, and hyperedge overlaps.",
        ],
        findings: vec![
            format!(
                "S4 has {} identity-compatible candidates and {} identity hyperedge; its {} right translates exactly equal the six reference V4 cosets.",
                s4.identity_compatible_candidates, s4.identity_hyperedges, s4.distinct_hyperedges
            ),
            format!(
                "S8 has {} fixed-point-free involutions. Their degree-{} compatibility graph has {} size-seven cliques, discovering {} order-eight identity blocks without R8 labels.",
                s8.identity_compatible_candidates,
                validation.s8_candidate_degree,
                s8.identity_hyperedges,
                s8.identity_hyperedges
            ),
            format!(
                "Right translation produces {} distinct S8 octet hyperedges in {} parallel classes, with {} incidences and uniform vertex degree {}.",
                s8.distinct_hyperedges,
                s8.discovered_parallel_classes,
                s8.incidences,
                validation.s8_vertex_degree
            ),
            format!(
                "Every S8 octet intersects {} other octets. Positive intersections have size one for {} pairs and size two for {} pairs.",
                validation.s8_overlapping_hyperedges_per_hyperedge,
                s8.positive_hyperedge_intersection_histogram
                    .get(&1)
                    .copied()
                    .unwrap_or(0),
                s8.positive_hyperedge_intersection_histogram
                    .get(&2)
                    .copied()
                    .unwrap_or(0)
            ),
        ],
        boundary: "The calculation discovers every unsigned size-n block satisfying the stated pairwise fixed-point-free-involution condition for n=4 and n=8. It recovers 30 subgroup-induced S8 parallel classes but does not prove those are the only exact-cover resolutions of the 151,200-edge hypergraph. It uses no Boolean factors and makes no signed Garden, HYMN, holoraumy, Gadget, or physical-parentage classification.",
        s4,
        s8,
        validation,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> HypergraphValidation {
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
    fn s4_constraint_hypergraph_recovers_the_six_reference_cosets() {
        let summary = build_summary(4);
        assert_eq!(summary.identity_compatible_candidates, 3);
        assert_eq!(summary.identity_hyperedges, 1);
        assert_eq!(summary.distinct_hyperedges, 6);
        assert_eq!(summary.vertex_degree_histogram, BTreeMap::from([(1, 24)]));
        assert_eq!(summary.matches_reference_cosets, Some(true));
        assert!(summary.hyperedges_recover_source_family);
    }

    #[test]
    fn s8_constraint_hypergraph_discovers_all_thirty_parallel_classes() {
        let summary = build_summary(8);
        assert_eq!(summary.identity_compatible_candidates, 105);
        assert_eq!(
            summary.candidate_degree_histogram,
            BTreeMap::from([(12, 105)])
        );
        assert_eq!(summary.identity_hyperedges, 30);
        assert_eq!(summary.hyperedges_per_parallel_class, 5_040);
        assert_eq!(summary.distinct_hyperedges, 151_200);
        assert_eq!(
            summary.vertex_degree_histogram,
            BTreeMap::from([(30, 40_320)])
        );
        assert_eq!(
            summary.overlapping_hyperedges_per_hyperedge_histogram,
            BTreeMap::from([(204, 151_200)])
        );
        assert!(summary.all_identity_hyperedges_are_subgroups);
        assert!(summary.every_parallel_class_is_complete);
        assert!(summary.hyperedges_recover_source_family);
    }

    #[test]
    fn complete_artifact_passes() {
        assert!(build().validation.passed);
    }
}
