//! Exact mixed-resolution witness for the S8 constraint hypergraph.
//!
//! The 30 discovered order-eight subgroups each induce a partition of S8.
//! This module asks whether those are the only exact covers.  It constructs a
//! local trade between two subgroup partitions and verifies the resulting
//! mixed cover vertex by vertex.

use crate::permutahedron::{coset_partition, factorial, CosetSide, Permutation};
use crate::permutahedron_hypergraph::identity_hyperedges;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-hypergraph-resolution-v1";

#[derive(Debug, Clone, Serialize)]
pub struct TradeCertificate {
    pub base_family: usize,
    pub replacement_family: usize,
    pub subgroup_intersection_order: usize,
    pub generated_subgroup_order: usize,
    pub removed_hyperedges: Vec<Vec<u32>>,
    pub added_hyperedges: Vec<Vec<u32>>,
    pub traded_vertex_count: usize,
    pub resulting_cover_hyperedges: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolutionValidation {
    pub discovered_families: usize,
    pub maximum_distinct_hyperedge_intersection: usize,
    pub trades_of_size_below_four_impossible: bool,
    pub four_for_four_trade_found: bool,
    pub removed_edges_are_disjoint: bool,
    pub added_edges_are_disjoint: bool,
    pub removed_and_added_unions_match: bool,
    pub resulting_cover_is_exact: bool,
    pub resulting_cover_differs_from_all_known_partitions: bool,
    pub minimum_mixed_trade_size_is_four: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolutionArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub family_pair_intersection_join_histogram: BTreeMap<String, usize>,
    pub order_32_family_pairs: usize,
    pub certified_elementary_four_trades: usize,
    pub certificate: TradeCertificate,
    pub validation: ResolutionValidation,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn ranks(elements: &[Permutation]) -> BTreeSet<u32> {
    elements
        .iter()
        .map(|element| element.rank() as u32)
        .collect()
}

fn intersection_order(left: &[Permutation], right: &[Permutation]) -> usize {
    let left = ranks(left);
    right
        .iter()
        .filter(|element| left.contains(&(element.rank() as u32)))
        .count()
}

fn generated_subgroup(left: &[Permutation], right: &[Permutation]) -> Vec<Permutation> {
    let identity = Permutation::identity(8).expect("S8 is supported");
    let generators: Vec<Permutation> = left.iter().chain(right).copied().collect();
    let mut seen = BTreeSet::from([identity]);
    let mut queue = VecDeque::from([identity]);
    while let Some(element) = queue.pop_front() {
        for &generator in &generators {
            let product = element
                .compose(generator)
                .expect("all generators belong to S8");
            if seen.insert(product) {
                queue.push_back(product);
            }
        }
    }
    seen.into_iter().collect()
}

fn blocks_inside(subgroup: &[Permutation], container: &BTreeSet<u32>) -> Vec<Vec<u32>> {
    coset_partition(subgroup, CosetSide::Right)
        .expect("discovered identity block is a subgroup")
        .slices
        .into_iter()
        .filter(|block| block.iter().all(|rank| container.contains(rank)))
        .collect()
}

fn disjoint_union(blocks: &[Vec<u32>]) -> (bool, BTreeSet<u32>) {
    let mut union = BTreeSet::new();
    let mut disjoint = true;
    for block in blocks {
        for &rank in block {
            disjoint &= union.insert(rank);
        }
    }
    (disjoint, union)
}

fn exact_cover(blocks: &[Vec<u32>], vertex_count: usize) -> bool {
    if blocks.len() * 8 != vertex_count {
        return false;
    }
    let mut degree = vec![0u8; vertex_count];
    for block in blocks {
        if block.len() != 8 {
            return false;
        }
        for &rank in block {
            let Some(slot) = degree.get_mut(rank as usize) else {
                return false;
            };
            *slot = slot.saturating_add(1);
        }
    }
    degree.into_iter().all(|value| value == 1)
}

fn maximum_identity_block_intersection(families: &[Vec<Permutation>]) -> usize {
    let family_ranks: Vec<BTreeSet<u32>> = families.iter().map(|family| ranks(family)).collect();
    let mut maximum = 0;
    for left in 0..family_ranks.len() {
        for right in left + 1..family_ranks.len() {
            maximum = maximum.max(
                family_ranks[left]
                    .intersection(&family_ranks[right])
                    .count(),
            );
        }
    }
    maximum
}

pub fn build() -> ResolutionArtifact {
    let families = identity_hyperedges(8);
    assert_eq!(families.len(), 30, "expected the discovered S8 families");
    let vertex_count = factorial(8).expect("S8 is supported");

    let mut pair_histogram = BTreeMap::new();
    let mut pair_data = Vec::new();
    for left in 0..families.len() {
        for right in left + 1..families.len() {
            let intersection = intersection_order(&families[left], &families[right]);
            let generated = generated_subgroup(&families[left], &families[right]);
            let join_order = generated.len();
            *pair_histogram
                .entry(format!("intersection_{intersection}_join_{join_order}"))
                .or_insert(0) += 1;
            pair_data.push((join_order, left, right, intersection, generated));
        }
    }
    pair_data.sort_unstable_by_key(|(order, left, right, _, _)| (*order, *left, *right));
    let order_32_family_pairs = pair_data
        .iter()
        .filter(|(order, _, _, _, _)| *order == 32)
        .count();

    let (join_order, base_family, replacement_family, subgroup_intersection, generated) = pair_data
        .into_iter()
        .next()
        .expect("at least two discovered families");
    let generated_ranks = ranks(&generated);
    let removed = blocks_inside(&families[base_family], &generated_ranks);
    let added = blocks_inside(&families[replacement_family], &generated_ranks);
    let (removed_disjoint, removed_union) = disjoint_union(&removed);
    let (added_disjoint, added_union) = disjoint_union(&added);

    let base_partition = coset_partition(&families[base_family], CosetSide::Right)
        .expect("discovered identity block is a subgroup")
        .slices;
    let removed_set: BTreeSet<Vec<u32>> = removed.iter().cloned().collect();
    let mut mixed_cover: Vec<Vec<u32>> = base_partition
        .iter()
        .filter(|block| !removed_set.contains(*block))
        .cloned()
        .collect();
    mixed_cover.extend(added.iter().cloned());
    mixed_cover.sort_unstable();

    let known_partitions: Vec<BTreeSet<Vec<u32>>> = families
        .iter()
        .map(|family| {
            coset_partition(family, CosetSide::Right)
                .expect("discovered identity block is a subgroup")
                .slices
                .into_iter()
                .collect()
        })
        .collect();
    let mixed_set: BTreeSet<Vec<u32>> = mixed_cover.iter().cloned().collect();
    let differs_from_all_known = known_partitions
        .iter()
        .all(|partition| partition != &mixed_set);

    let maximum_intersection = maximum_identity_block_intersection(&families);
    // Any two intersecting hyperedges can be right-translated by a shared
    // vertex to two identity hyperedges, so the identity-block maximum is the
    // global maximum.  An added octet in a t-edge trade can therefore cover at
    // most 2t vertices from the removed octets.  Since it has eight vertices,
    // t >= 4.
    let below_four_impossible = maximum_intersection == 2;
    let four_for_four =
        join_order == 32 && removed.len() == 4 && added.len() == 4 && subgroup_intersection == 2;
    let cover_is_exact = exact_cover(&mixed_cover, vertex_count);
    let union_matches = removed_union == added_union && removed_union == generated_ranks;
    let minimum_is_four = below_four_impossible
        && four_for_four
        && removed_disjoint
        && added_disjoint
        && union_matches
        && cover_is_exact
        && differs_from_all_known;

    let validation = ResolutionValidation {
        discovered_families: families.len(),
        maximum_distinct_hyperedge_intersection: maximum_intersection,
        trades_of_size_below_four_impossible: below_four_impossible,
        four_for_four_trade_found: four_for_four,
        removed_edges_are_disjoint: removed_disjoint,
        added_edges_are_disjoint: added_disjoint,
        removed_and_added_unions_match: union_matches,
        resulting_cover_is_exact: cover_is_exact,
        resulting_cover_differs_from_all_known_partitions: differs_from_all_known,
        minimum_mixed_trade_size_is_four: minimum_is_four,
        passed: minimum_is_four,
    };

    let certificate = TradeCertificate {
        base_family,
        replacement_family,
        subgroup_intersection_order: subgroup_intersection,
        generated_subgroup_order: join_order,
        removed_hyperedges: removed,
        added_hyperedges: added,
        traded_vertex_count: generated_ranks.len(),
        resulting_cover_hyperedges: mixed_cover.len(),
    };

    ResolutionArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Exact mixed-cover resolution of the S8 constraint hypergraph",
        method: vec![
            "Compute the intersection and generated-subgroup order for all 435 pairs of discovered identity octets.",
            "Choose the lexicographically first pair with minimum generated-subgroup order.",
            "Replace the base family's right cosets inside the generated subgroup by the replacement family's right cosets of the same 32 vertices.",
            "Verify the resulting 5,040-octet cover vertex by vertex and compare it with all 30 subgroup partitions.",
            "Use the exact global intersection bound of two vertices per distinct hyperedge to exclude trades of sizes one, two, and three.",
        ],
        family_pair_intersection_join_histogram: pair_histogram,
        order_32_family_pairs,
        certified_elementary_four_trades: order_32_family_pairs * (vertex_count / 32),
        findings: vec![
            format!(
                "A four-for-four trade between discovered families {base_family} and {replacement_family} replaces four octets on the same 32 vertices and produces a new exact cover."
            ),
            "The mixed cover has 5,040 octets but is not any of the 30 subgroup-induced partitions.".into(),
            "No trade using fewer than four removed octets is possible because distinct hyperedges intersect in at most two vertices.".into(),
            format!(
                "There are {order_32_family_pairs} family pairs generating order-32 subgroups, yielding {} translated elementary four-trade certificates.",
                order_32_family_pairs * (vertex_count / 32)
            ),
        ],
        boundary: "This proves that the 30 subgroup partitions are not the only exact covers and gives a minimum-size local mixed-cover trade. It does not enumerate all exact covers, count covers obtained by combining compatible trades, or attach signed or higher-dimensional physical data.",
        certificate,
        validation,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ResolutionValidation {
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
    fn s8_has_a_certified_minimum_four_edge_mixed_cover_trade() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.certificate.generated_subgroup_order, 32);
        assert_eq!(artifact.certificate.removed_hyperedges.len(), 4);
        assert_eq!(artifact.certificate.added_hyperedges.len(), 4);
        assert_eq!(artifact.certificate.resulting_cover_hyperedges, 5_040);
    }
}
