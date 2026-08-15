//! Exact identifiability audit for spectral recovery of the S8 R8 partitions.
//!
//! Every discovered R8 conjugate defines an equitable right-coset partition
//! of the adjacent-transposition Cayley graph. Left multiplication is a graph
//! automorphism and transports the standard partition to each conjugate one.
//! Consequently, no graph-only spectral filter can select the standard member
//! of this thirty-element orbit without external symmetry-breaking data.

use crate::permutahedron::{
    complete_graph, coset_partition, permutations, rana_r8, CosetSide, Permutation,
};
use crate::permutahedron_hypergraph::identity_hyperedges;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-spectral-identifiability-v1";

#[derive(Debug, Clone, Serialize)]
pub struct FamilySpectralRecord {
    pub family_id: usize,
    pub identity_block_ranks: Vec<u32>,
    pub least_conjugator_rank_from_standard: u32,
    pub right_cosets: usize,
    pub coset_size: usize,
    pub equitable_partition_violations: usize,
    pub directed_adjacency_relations_checked: usize,
    pub directed_adjacency_intertwining_violations: usize,
    pub source_cosets_mapped_bijectively: usize,
    pub partition_transport_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpectralIdentifiabilityValidation {
    pub graph_vertices: usize,
    pub graph_edges: usize,
    pub graph_degree: usize,
    pub discovered_r8_families: usize,
    pub distinct_right_coset_partitions: usize,
    pub right_cosets_per_partition: usize,
    pub all_partitions_equitable: bool,
    pub total_equitability_violations: usize,
    pub all_families_conjugate_to_standard: bool,
    pub all_conjugators_intertwine_adjacency: bool,
    pub directed_adjacency_relations_checked: usize,
    pub all_partition_transports_bijective: bool,
    pub vertex_partition_memberships_checked: usize,
    pub quotient_graphs_form_one_isomorphism_class: bool,
    pub adjacency_polynomial_filters_preserve_thirty_fold_ambiguity: bool,
    pub graph_only_standard_r8_selection_identifiable: bool,
    pub audit_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpectralIdentifiabilityArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub families: Vec<FamilySpectralRecord>,
    pub validation: SpectralIdentifiabilityValidation,
    pub findings: Vec<&'static str>,
    pub consequence: &'static str,
    pub boundary: &'static str,
}

fn sorted_ranks(elements: &[Permutation]) -> Vec<u32> {
    let mut ranks: Vec<u32> = elements
        .iter()
        .map(|permutation| permutation.rank() as u32)
        .collect();
    ranks.sort_unstable();
    ranks
}

fn partition_labels(slices: &[Vec<u32>], vertices: usize) -> Vec<u32> {
    let mut labels = vec![u32::MAX; vertices];
    for (coset_id, slice) in slices.iter().enumerate() {
        for &rank in slice {
            assert_eq!(labels[rank as usize], u32::MAX);
            labels[rank as usize] = coset_id as u32;
        }
    }
    assert!(labels.iter().all(|&label| label != u32::MAX));
    labels
}

fn equitability_violations(slices: &[Vec<u32>], labels: &[u32], adjacency: &[Vec<u32>]) -> usize {
    let mut violations = 0usize;
    for slice in slices {
        let mut reference = None;
        for &rank in slice {
            let mut profile: Vec<u32> = adjacency[rank as usize]
                .iter()
                .map(|&neighbor| labels[neighbor as usize])
                .collect();
            profile.sort_unstable();
            match &reference {
                None => reference = Some(profile),
                Some(expected) if *expected == profile => {}
                Some(_) => violations += 1,
            }
        }
    }
    violations
}

fn least_conjugator(source: &[Permutation], target: &[Permutation]) -> Option<Permutation> {
    let target_ranks: BTreeSet<usize> = target.iter().map(|element| element.rank()).collect();
    permutations(8)
        .expect("S8 permutations")
        .into_iter()
        .find(|&conjugator| {
            let inverse = conjugator.inverse();
            source.iter().all(|&element| {
                target_ranks.contains(
                    &conjugator
                        .compose(element)
                        .expect("common S8 order")
                        .compose(inverse)
                        .expect("common S8 order")
                        .rank(),
                )
            })
        })
}

pub fn build() -> SpectralIdentifiabilityArtifact {
    let graph = complete_graph(8).expect("S8 permutahedron graph");
    let families = identity_hyperedges(8);
    let standard_key = sorted_ranks(&rana_r8());
    let standard_family_id = families
        .iter()
        .position(|family| sorted_ranks(family) == standard_key)
        .expect("standard R8 is among discovered families");
    let standard = &families[standard_family_id];

    let mut adjacency = vec![Vec::with_capacity(graph.degree); graph.vertex_count];
    for &[left, right] in &graph.edges {
        adjacency[left as usize].push(right);
        adjacency[right as usize].push(left);
    }
    assert!(adjacency
        .iter()
        .all(|neighbors| neighbors.len() == graph.degree));

    let mut partitions = Vec::with_capacity(families.len());
    let mut labels = Vec::with_capacity(families.len());
    for family in &families {
        let partition = coset_partition(family, CosetSide::Right).expect("discovered R8 subgroup");
        labels.push(partition_labels(&partition.slices, graph.vertex_count));
        partitions.push(partition);
    }

    let standard_labels = &labels[standard_family_id];
    let mut records = Vec::with_capacity(families.len());
    for (family_id, family) in families.iter().enumerate() {
        let conjugator = least_conjugator(standard, family).expect("R8 conjugate");
        let partition = &partitions[family_id];
        let family_labels = &labels[family_id];
        let equitable_partition_violations =
            equitability_violations(&partition.slices, family_labels, &adjacency);

        let mut adjacency_violations = 0usize;
        let mut directed_adjacency_relations_checked = 0usize;
        let mut source_to_target = vec![usize::MAX; partitions[standard_family_id].slice_count];
        let mut partition_transport_passed = true;

        for rank in 0..graph.vertex_count {
            let source = Permutation::unrank(8, rank).expect("S8 rank");
            let mapped = conjugator.compose(source).expect("common S8 order");
            let source_coset = standard_labels[rank] as usize;
            let target_coset = family_labels[mapped.rank()] as usize;
            match source_to_target[source_coset] {
                usize::MAX => source_to_target[source_coset] = target_coset,
                expected if expected == target_coset => {}
                _ => partition_transport_passed = false,
            }

            for generator in 0..graph.degree {
                directed_adjacency_relations_checked += 1;
                let left_then_edge = mapped
                    .right_adjacent(generator)
                    .expect("adjacent generator");
                let edge_then_left = conjugator
                    .compose(
                        source
                            .right_adjacent(generator)
                            .expect("adjacent generator"),
                    )
                    .expect("common S8 order");
                if left_then_edge != edge_then_left {
                    adjacency_violations += 1;
                }
            }
        }

        let mapped_cosets: BTreeSet<usize> = source_to_target.iter().copied().collect();
        partition_transport_passed &=
            !mapped_cosets.contains(&usize::MAX) && mapped_cosets.len() == partition.slice_count;

        records.push(FamilySpectralRecord {
            family_id,
            identity_block_ranks: sorted_ranks(family),
            least_conjugator_rank_from_standard: conjugator.rank() as u32,
            right_cosets: partition.slice_count,
            coset_size: partition.slice_size,
            equitable_partition_violations,
            directed_adjacency_relations_checked,
            directed_adjacency_intertwining_violations: adjacency_violations,
            source_cosets_mapped_bijectively: mapped_cosets.len(),
            partition_transport_passed,
        });
    }

    let distinct_family_keys: BTreeSet<Vec<u32>> =
        families.iter().map(|family| sorted_ranks(family)).collect();
    let total_equitability_violations = records
        .iter()
        .map(|record| record.equitable_partition_violations)
        .sum();
    let all_partitions_equitable = total_equitability_violations == 0;
    let all_families_conjugate_to_standard = records.len() == families.len();
    let all_conjugators_intertwine_adjacency = records
        .iter()
        .all(|record| record.directed_adjacency_intertwining_violations == 0);
    let all_partition_transports_bijective = records
        .iter()
        .all(|record| record.partition_transport_passed);
    let quotient_graphs_form_one_isomorphism_class =
        all_conjugators_intertwine_adjacency && all_partition_transports_bijective;
    let adjacency_polynomial_filters_preserve_thirty_fold_ambiguity =
        quotient_graphs_form_one_isomorphism_class;
    let graph_only_standard_r8_selection_identifiable = false;
    let directed_checks = records
        .iter()
        .map(|record| record.directed_adjacency_relations_checked)
        .sum();
    let vertex_partition_memberships_checked = records.len() * graph.vertex_count;
    let audit_passed = graph.vertex_count == 40_320
        && graph.edge_count == 141_120
        && graph.degree == 7
        && families.len() == 30
        && distinct_family_keys.len() == 30
        && partitions
            .iter()
            .all(|partition| partition.slice_count == 5_040 && partition.slice_size == 8)
        && all_partitions_equitable
        && all_families_conjugate_to_standard
        && all_conjugators_intertwine_adjacency
        && all_partition_transports_bijective
        && quotient_graphs_form_one_isomorphism_class
        && adjacency_polynomial_filters_preserve_thirty_fold_ambiguity
        && !graph_only_standard_r8_selection_identifiable;

    SpectralIdentifiabilityArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Exact spectral identifiability audit for the thirty S8 R8 partitions",
        method: vec![
            "Construct all thirty R8 identity blocks from the unsigned Garden compatibility equations.",
            "Build each complete right-coset partition and test equitability against every directed adjacent-transposition relation.",
            "Find the least S8 conjugator from the standard R8 block to every discovered block.",
            "Verify that left multiplication by each conjugator intertwines every graph edge and bijectively transports all 5,040 standard cosets.",
            "Apply the automorphism argument to every polynomial in the adjacency or Laplacian operator.",
        ],
        families: records,
        validation: SpectralIdentifiabilityValidation {
            graph_vertices: graph.vertex_count,
            graph_edges: graph.edge_count,
            graph_degree: graph.degree,
            discovered_r8_families: families.len(),
            distinct_right_coset_partitions: distinct_family_keys.len(),
            right_cosets_per_partition: 5_040,
            all_partitions_equitable,
            total_equitability_violations,
            all_families_conjugate_to_standard,
            all_conjugators_intertwine_adjacency,
            directed_adjacency_relations_checked: directed_checks,
            all_partition_transports_bijective,
            vertex_partition_memberships_checked,
            quotient_graphs_form_one_isomorphism_class,
            adjacency_polynomial_filters_preserve_thirty_fold_ambiguity,
            graph_only_standard_r8_selection_identifiable,
            audit_passed,
        },
        findings: vec![
            "All thirty R8 right-coset partitions are exact equitable partitions of the S8 adjacent-transposition Cayley graph.",
            "The thirty partitions are distinct but lie in one orbit under left-regular graph automorphisms.",
            "Their quotient graphs are mutually isomorphic through the verified coset transports.",
            "Chebyshev filtering, polynomial-filtered Lanczos, and any other adjacency-polynomial method inherit this thirty-fold symmetry and cannot select the standard R8 member without an external anchor.",
            "The naive S8 clustering failure is therefore not repaired as a physical selection result by targeting interior eigenspaces.",
        ],
        consequence: "The spectral program is complete as an identifiability result. Spectral methods may recover an R8-type equitable partition up to the thirty-element conjugacy orbit, but the unlabeled Cayley graph cannot identify which conjugate is the standard or physically preferred partition. A basis, source fixture, or other external symmetry-breaking datum is required.",
        boundary: "This does not claim that polynomial-filtered Lanczos cannot numerically recover some R8-type partition. It proves that graph-only spectral data cannot canonically choose one member of the verified thirty-partition automorphism orbit, so such recovery would not solve the physical parentage-selection problem.",
    }
}

pub fn write_artifact(path: &Path) -> SpectralIdentifiabilityValidation {
    let artifact = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create spectral audit artifact")),
        &artifact,
    )
    .expect("write spectral audit artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_thirty_r8_partitions_are_one_equitable_automorphism_orbit() {
        let artifact = build();
        assert!(artifact.validation.audit_passed);
        assert_eq!(artifact.validation.discovered_r8_families, 30);
        assert!(artifact.validation.all_partitions_equitable);
        assert!(
            artifact
                .validation
                .quotient_graphs_form_one_isomorphism_class
        );
        assert!(
            !artifact
                .validation
                .graph_only_standard_r8_selection_identifiable
        );
    }
}
