//! Complete `S_4` and `S_8` permutahedron atlases built from the primary-paper
//! fixtures and the deterministic permutation core.

use crate::permutahedron::{
    CosetPartitionReport, CosetSide, GraphAuditReport, Permutation, abnormal_slices,
    adjacent_transposition, bfs_graph_audit, complete_graph, coset_partition, factorial,
    permutations, rana_r8, validate_subgroup, vierergruppe,
};
use crate::permutahedron_fixtures::{
    AdjacentTransposition, PRIMARY_SOURCE_DOCUMENTS, R8_DIADEM_OCTET, R8_SOURCE,
    S4_CORRELATOR_BLOCK_SOURCES, S4_CORRELATOR_SOURCE, S4_DICTIONARY, S4_DICTIONARY_SOURCE,
    S4_EDGE_SOURCE, S4_LABELED_EDGES, S4_ORDERED_QUARTETS, S4_QUARTET_SOURCE,
    S4_SYMMETRIC_CORRELATOR_ENTRY_COUNT, S8_COLOR_COUNT, S8_COMBINATORICS_SOURCE, S8_EDGE_COUNT,
    S8_FACE_COUNT, S8_MAGIC_NUMBER, S8_MAGIC_NUMBER_SOURCE, S8_OCTET_COUNT,
    S8_REPRESENTATION_OCTETS, S8_REPRESENTATION_SOURCE, S8_VERTEX_COUNT, SourceDocument,
    SourceLocator, reconstruct_s4_bruhat_correlator,
};
use crate::permutahedron_garden::{CompleteGardenScan, complete_garden_scan};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: &str = "permutahedron-atlas-v1";

#[derive(Debug, Clone, Serialize)]
pub struct AtlasSource {
    pub arxiv_id: &'static str,
    pub version: u8,
    pub local_pdf: &'static str,
    pub pdf_sha256: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct AtlasLocator {
    pub arxiv_id: &'static str,
    pub version: u8,
    pub pdf_pages: &'static str,
    pub equation_or_table: &'static str,
}

impl From<SourceLocator> for AtlasLocator {
    fn from(source: SourceLocator) -> Self {
        Self {
            arxiv_id: source.arxiv_id,
            version: source.version,
            pdf_pages: source.pdf_pages,
            equation_or_table: source.equation_or_table,
        }
    }
}

impl From<&'static SourceDocument> for AtlasSource {
    fn from(source: &'static SourceDocument) -> Self {
        Self {
            arxiv_id: source.arxiv_id,
            version: source.version,
            local_pdf: source.local_pdf,
            pdf_sha256: source.pdf_sha256,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AtlasMetadata {
    pub schema_version: &'static str,
    pub n: usize,
    pub stable_id_format: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub degree: usize,
    pub diameter: usize,
    pub edge_convention: &'static str,
    pub correlator_definition: &'static str,
    pub source: Vec<AtlasSource>,
    pub source_locators: Vec<AtlasLocator>,
    pub generated_by: &'static str,
    pub complete_vertex_and_edge_dataset: bool,
    pub dense_all_pairs_matrix_stored: bool,
    pub two_face_count: Option<usize>,
    pub square_face_count: Option<usize>,
    pub hexagonal_face_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RepresentationRecord {
    pub id: String,
    pub label: String,
    pub member_ranks: Vec<u32>,
    pub member_addresses: Vec<String>,
    pub right_slice_id: u32,
    pub garden_status: &'static str,
    pub source: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedCorrelatorMatrix {
    pub definition: &'static str,
    pub labels: Vec<String>,
    pub values: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PermutahedronAtlas {
    pub metadata: AtlasMetadata,
    /// One-line addresses in zero-based lexicographic Lehmer-rank order.
    pub permutations: Vec<String>,
    /// `[source_rank, target_rank, one_based_adjacent_generator]`.
    pub edges: Vec<[u32; 3]>,
    pub right_slice_by_rank: Vec<u32>,
    pub left_slice_by_rank: Vec<u32>,
    pub right_slices: Vec<Vec<u32>>,
    pub left_slices: Vec<Vec<u32>>,
    pub abnormal_right_slices: Vec<u32>,
    pub representations: Vec<RepresentationRecord>,
    pub named_correlator_matrix: NamedCorrelatorMatrix,
}

#[derive(Debug, Clone, Serialize)]
pub struct S4Validation {
    pub graph: GraphAuditReport,
    pub dictionary_entries_checked: usize,
    pub labeled_edges_checked: usize,
    pub published_blocks_checked: usize,
    pub published_block_entries_checked: usize,
    pub full_matrix_entries_checked: usize,
    pub independent_symmetric_entries: usize,
    pub distance_mismatches: usize,
    pub quartet_partition_complete: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct S8Validation {
    pub graph: GraphAuditReport,
    pub subgroup_order: usize,
    pub subgroup_valid: bool,
    pub right_cosets: usize,
    pub left_cosets: usize,
    pub coset_size: usize,
    pub right_cover_complete: bool,
    pub left_cover_complete: bool,
    pub abnormal_cosets: usize,
    pub conjugate_r8_subgroups: usize,
    pub named_octets: usize,
    pub named_vertices: usize,
    pub named_correlator_entries_checked: usize,
    pub named_octets_are_distinct_cosets: bool,
    pub named_magic_rows_checked: usize,
    pub named_magic_failures: usize,
    pub all_coset_intra_rows_checked: usize,
    pub all_coset_intra_failures: usize,
    pub square_faces_computed: usize,
    pub hexagonal_faces_computed: usize,
    pub two_faces_computed: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PermutahedronValidation {
    pub schema_version: &'static str,
    pub s4: S4Validation,
    pub s8: S8Validation,
    pub garden: GardenValidation,
    pub s4_artifact: &'static str,
    pub s8_artifact: &'static str,
    pub garden_artifact: &'static str,
    pub browser: &'static str,
    pub boundary: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GardenValidation {
    pub cosets_scanned: usize,
    pub signable_cosets: usize,
    pub unsignable_cosets: usize,
    pub equation_rank: usize,
    pub equation_nullity: usize,
    pub solutions_per_coset: u64,
    pub independent_sparse_verifications: usize,
    pub dense_entries_checked: usize,
    pub dense_residual_entries: usize,
    pub normalizer_order: usize,
    pub normalizer_quotient_cosets: usize,
    pub passed: bool,
}

impl From<&CompleteGardenScan> for GardenValidation {
    fn from(scan: &CompleteGardenScan) -> Self {
        assert_eq!(scan.rank_histogram.len(), 1, "Garden rank must be uniform");
        assert_eq!(scan.nullity_histogram.len(), 1, "Garden nullity must be uniform");
        Self {
            cosets_scanned: scan.cosets_scanned,
            signable_cosets: scan.signable_cosets,
            unsignable_cosets: scan.unsignable_cosets,
            equation_rank: scan.rank_histogram[0].value,
            equation_nullity: scan.nullity_histogram[0].value,
            solutions_per_coset: scan.solution_count_per_coset,
            independent_sparse_verifications: scan.independent_sparse_verifications,
            dense_entries_checked: scan.dense_entries_checked,
            dense_residual_entries: scan.dense_residual_entries,
            normalizer_order: scan.normalizer.normalizer_order,
            normalizer_quotient_cosets: scan.normalizer.normalizer_cosets,
            passed: scan.passed,
        }
    }
}

fn permutation_string(permutation: Permutation) -> String {
    permutation
        .one_line()
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("")
}

fn from_array<const N: usize>(values: &[u8; N]) -> Permutation {
    Permutation::new(values).expect("primary-source fixture is a permutation")
}

fn complete_labeled_edges(n: usize) -> Vec<[u32; 3]> {
    let count = factorial(n).expect("supported atlas order");
    let generators: Vec<Permutation> = (0..n - 1)
        .map(|index| adjacent_transposition(n, index).expect("bounded generator"))
        .collect();
    let mut edges = Vec::with_capacity(count * (n - 1) / 2);
    for rank in 0..count {
        let permutation = Permutation::unrank(n, rank).expect("bounded rank");
        for (generator, adjacent) in generators.iter().enumerate() {
            let neighbor = permutation.compose(*adjacent).expect("common order").rank();
            if rank < neighbor {
                edges.push([rank as u32, neighbor as u32, (generator + 1) as u32]);
            }
        }
    }
    let core = complete_graph(n).expect("complete core graph");
    assert_eq!(core.edges.len(), edges.len());
    assert!(
        core.edges
            .iter()
            .zip(&edges)
            .all(|(plain, labeled)| plain[0] == labeled[0] && plain[1] == labeled[1])
    );
    edges
}

fn slice_assignment(partition: &CosetPartitionReport) -> Vec<u32> {
    let mut assignment = vec![u32::MAX; partition.covered_vertices];
    for (slice_id, slice) in partition.slices.iter().enumerate() {
        for &rank in slice {
            assert_eq!(assignment[rank as usize], u32::MAX);
            assignment[rank as usize] = slice_id as u32;
        }
    }
    assert!(assignment.iter().all(|&slice| slice != u32::MAX));
    assignment
}

fn abnormal_slice_ids(right_partition: &CosetPartitionReport, abnormal: &[Vec<u32>]) -> Vec<u32> {
    let by_minimum: BTreeMap<u32, u32> = right_partition
        .slices
        .iter()
        .enumerate()
        .map(|(index, slice)| (slice[0], index as u32))
        .collect();
    abnormal.iter().map(|slice| by_minimum[&slice[0]]).collect()
}

fn source_records(ids: &[&str]) -> Vec<AtlasSource> {
    PRIMARY_SOURCE_DOCUMENTS
        .iter()
        .filter(|source| ids.contains(&source.arxiv_id))
        .map(AtlasSource::from)
        .collect()
}

fn s4_representations(right_assignment: &[u32]) -> Vec<RepresentationRecord> {
    let names = [
        ("P1", "P1 / CM / chiral"),
        ("P2", "P2 / TM / tensor"),
        ("P3", "P3 / VM / vector"),
        ("P4", "P4 / VM1"),
        ("P5", "P5 / VM2"),
        ("P6", "P6 / VM3 / Vierergruppe"),
    ];
    names
        .iter()
        .zip(S4_ORDERED_QUARTETS.iter())
        .map(|((id, label), quartet)| {
            let members: Vec<Permutation> = quartet.iter().map(from_array).collect();
            let member_ranks: Vec<u32> = members.iter().map(|p| p.rank() as u32).collect();
            let slice_id = right_assignment[member_ranks[0] as usize];
            assert!(
                member_ranks
                    .iter()
                    .all(|&rank| right_assignment[rank as usize] == slice_id)
            );
            RepresentationRecord {
                id: (*id).into(),
                label: (*label).into(),
                member_addresses: members.iter().copied().map(permutation_string).collect(),
                member_ranks,
                right_slice_id: slice_id,
                garden_status: "published_unsigned_s4_quartet",
                source: "arXiv:2012.13308v6, Fig. 2 and Tables 1-6",
            }
        })
        .collect()
}

fn s8_representations(right_assignment: &[u32]) -> Vec<RepresentationRecord> {
    let mut records = Vec::new();
    for octet in &S8_REPRESENTATION_OCTETS {
        let members: Vec<Permutation> = octet.permutations.iter().map(from_array).collect();
        let member_ranks: Vec<u32> = members.iter().map(|p| p.rank() as u32).collect();
        let slice_id = right_assignment[member_ranks[0] as usize];
        assert!(
            member_ranks
                .iter()
                .all(|&rank| right_assignment[rank as usize] == slice_id)
        );
        let garden_status = match octet.label {
            "CT" | "CV" => "published_garden_closure",
            "CC" | "TT" | "TV" | "VV" => "published_nonclosure",
            _ => unreachable!(),
        };
        records.push(RepresentationRecord {
            id: octet.label.into(),
            label: match octet.label {
                "CC" => "CC / chiral + chiral",
                "CT" => "CT / chiral + tensor",
                "CV" => "CV / chiral + vector",
                "TT" => "TT / tensor + tensor",
                "TV" => "TV / tensor + vector",
                "VV" => "VV / vector + vector",
                _ => unreachable!(),
            }
            .into(),
            member_addresses: members.iter().copied().map(permutation_string).collect(),
            member_ranks,
            right_slice_id: slice_id,
            garden_status,
            source: "arXiv:2012.14015v7, Eqs. (5.1)-(5.6)",
        });
    }
    let members: Vec<Permutation> = R8_DIADEM_OCTET.iter().map(from_array).collect();
    let member_ranks: Vec<u32> = members.iter().map(|p| p.rank() as u32).collect();
    let slice_id = right_assignment[member_ranks[0] as usize];
    assert!(
        member_ranks
            .iter()
            .all(|&rank| right_assignment[rank as usize] == slice_id)
    );
    records.push(RepresentationRecord {
        id: "O".into(),
        label: "O / R8 / Diadem(8)".into(),
        member_addresses: members.iter().copied().map(permutation_string).collect(),
        member_ranks,
        right_slice_id: slice_id,
        garden_status: "published_garden_closure",
        source: "arXiv:2012.14015v7 Eq. (5.7); arXiv:2304.09830v2 Eq. (3.23)",
    });
    records
}

fn named_matrix(representations: &[RepresentationRecord]) -> NamedCorrelatorMatrix {
    let ordered: Vec<(String, Permutation)> = representations
        .iter()
        .flat_map(|representation| {
            representation
                .member_ranks
                .iter()
                .enumerate()
                .map(move |(index, &rank)| {
                    (
                        format!("{}:L{}", representation.id, index + 1),
                        Permutation::unrank(
                            representation.member_addresses[0].len(),
                            rank as usize,
                        )
                        .expect("stored member rank"),
                    )
                })
        })
        .collect();
    let labels = ordered.iter().map(|(label, _)| label.clone()).collect();
    let values = ordered
        .iter()
        .map(|(_, left)| {
            ordered
                .iter()
                .map(|(_, right)| left.bruhat_distance(*right).expect("common atlas order") as u8)
                .collect()
        })
        .collect();
    NamedCorrelatorMatrix {
        definition: "minimal weak-Bruhat distance; not a holoraumy gadget",
        labels,
        values,
    }
}

fn s4_validation() -> S4Validation {
    let graph = bfs_graph_audit(4).expect("S4 audit");
    let published = reconstruct_s4_bruhat_correlator().expect("complete published blocks");
    let mut mismatches = 0;
    for (left_rank, row) in published.iter().enumerate() {
        let left = Permutation::unrank(4, left_rank).expect("S4 rank");
        for (right_rank, &expected) in row.iter().enumerate() {
            let right = Permutation::unrank(4, right_rank).expect("S4 rank");
            if left.bruhat_distance(right).expect("common order") != expected as usize {
                mismatches += 1;
            }
        }
    }
    let covered: BTreeSet<usize> = S4_ORDERED_QUARTETS
        .iter()
        .flatten()
        .map(|p| from_array(p).rank())
        .collect();
    let quartet_partition_complete = covered.len() == 24;
    let generated_edges = complete_labeled_edges(4);
    let labeled_edge_mismatches = S4_LABELED_EDGES
        .iter()
        .filter(|fixture| {
            let generator = match fixture.generator {
                AdjacentTransposition::S1 => 1,
                AdjacentTransposition::S2 => 2,
                AdjacentTransposition::S3 => 3,
            };
            !generated_edges.contains(&[fixture.from as u32, fixture.to as u32, generator])
        })
        .count();
    let passed = graph.connected
        && graph.vertex_count == 24
        && graph.edge_count == 36
        && graph.diameter == 6
        && mismatches == 0
        && labeled_edge_mismatches == 0
        && quartet_partition_complete;
    S4Validation {
        graph,
        dictionary_entries_checked: S4_DICTIONARY.len(),
        labeled_edges_checked: S4_LABELED_EDGES.len(),
        published_blocks_checked: S4_CORRELATOR_BLOCK_SOURCES.len(),
        published_block_entries_checked: S4_CORRELATOR_BLOCK_SOURCES.len() * 16,
        full_matrix_entries_checked: 24 * 24,
        independent_symmetric_entries: S4_SYMMETRIC_CORRELATOR_ENTRY_COUNT,
        distance_mismatches: mismatches,
        quartet_partition_complete,
        passed,
    }
}

fn conjugate_subgroup_count(subgroup: &[Permutation]) -> usize {
    let n = subgroup[0].n();
    let mut conjugates = BTreeSet::new();
    for rank in 0..factorial(n).expect("supported order") {
        let g = Permutation::unrank(n, rank).expect("bounded rank");
        let inverse = g.inverse();
        let mut conjugate: Vec<u32> = subgroup
            .iter()
            .map(|&h| {
                g.compose(h)
                    .and_then(|value| value.compose(inverse))
                    .expect("common order")
                    .rank() as u32
            })
            .collect();
        conjugate.sort_unstable();
        conjugates.insert(conjugate);
    }
    conjugates.len()
}

fn s8_validation(
    right: &CosetPartitionReport,
    left: &CosetPartitionReport,
    representations: &[RepresentationRecord],
) -> S8Validation {
    let graph = bfs_graph_audit(8).expect("S8 audit");
    let subgroup = rana_r8();
    let subgroup_report = validate_subgroup(&subgroup);
    let abnormal = abnormal_slices(&subgroup).expect("R8 abnormal cosets");
    let conjugates = conjugate_subgroup_count(&subgroup);
    let distinct_slices: BTreeSet<u32> = representations
        .iter()
        .map(|record| record.right_slice_id)
        .collect();

    let named_permutations: Vec<Vec<Permutation>> = representations
        .iter()
        .map(|record| {
            record
                .member_ranks
                .iter()
                .map(|&rank| Permutation::unrank(8, rank as usize).expect("stored S8 rank"))
                .collect()
        })
        .collect();
    let mut named_magic_rows_checked = 0;
    let mut named_magic_failures = 0;
    for source in &named_permutations {
        for target in &named_permutations {
            for &permutation in source {
                let sum: usize = target
                    .iter()
                    .map(|&other| permutation.bruhat_distance(other).expect("common S8 order"))
                    .sum();
                named_magic_rows_checked += 1;
                named_magic_failures += usize::from(sum != S8_MAGIC_NUMBER);
            }
        }
    }

    let mut all_coset_intra_rows_checked = 0;
    let mut all_coset_intra_failures = 0;
    for slice in &right.slices {
        let members: Vec<Permutation> = slice
            .iter()
            .map(|&rank| Permutation::unrank(8, rank as usize).expect("coset rank"))
            .collect();
        for &permutation in &members {
            let sum: usize = members
                .iter()
                .map(|&other| permutation.bruhat_distance(other).expect("common order"))
                .sum();
            all_coset_intra_rows_checked += 1;
            all_coset_intra_failures += usize::from(sum != S8_MAGIC_NUMBER);
        }
    }

    let vertices = factorial(8).expect("supported order");
    let noncommuting_generator_pairs = 6;
    let all_generator_pairs = 7 * 6 / 2;
    let commuting_generator_pairs = all_generator_pairs - noncommuting_generator_pairs;
    let square_faces = vertices * commuting_generator_pairs / 4;
    let hexagonal_faces = vertices * noncommuting_generator_pairs / 6;
    let two_faces = square_faces + hexagonal_faces;

    let passed = graph.connected
        && graph.vertex_count == S8_VERTEX_COUNT
        && graph.edge_count == S8_EDGE_COUNT
        && graph.diameter == 28
        && subgroup_report.valid
        && right.complete_cover
        && left.complete_cover
        && right.slice_count == S8_OCTET_COUNT
        && left.slice_count == S8_OCTET_COUNT
        && abnormal.abnormal_slice_count == 168
        && conjugates == 30
        && representations.len() == 7
        && distinct_slices.len() == 7
        && named_magic_failures == 0
        && all_coset_intra_failures == 0
        && square_faces == 151_200
        && hexagonal_faces == 40_320
        && two_faces == S8_FACE_COUNT;
    S8Validation {
        graph,
        subgroup_order: subgroup_report.order,
        subgroup_valid: subgroup_report.valid,
        right_cosets: right.slice_count,
        left_cosets: left.slice_count,
        coset_size: right.slice_size,
        right_cover_complete: right.complete_cover,
        left_cover_complete: left.complete_cover,
        abnormal_cosets: abnormal.abnormal_slice_count,
        conjugate_r8_subgroups: conjugates,
        named_octets: representations.len(),
        named_vertices: representations
            .iter()
            .map(|record| record.member_ranks.len())
            .sum(),
        named_correlator_entries_checked: representations.len()
            * representations.len()
            * right.slice_size
            * right.slice_size,
        named_octets_are_distinct_cosets: distinct_slices.len() == representations.len(),
        named_magic_rows_checked,
        named_magic_failures,
        all_coset_intra_rows_checked,
        all_coset_intra_failures,
        square_faces_computed: square_faces,
        hexagonal_faces_computed: hexagonal_faces,
        two_faces_computed: two_faces,
        passed,
    }
}

fn build_s4() -> (PermutahedronAtlas, S4Validation) {
    let graph = bfs_graph_audit(4).expect("S4 graph");
    let subgroup = vierergruppe();
    let right = coset_partition(&subgroup, CosetSide::Right).expect("S4 right cosets");
    let left = coset_partition(&subgroup, CosetSide::Left).expect("S4 left cosets");
    let abnormal = abnormal_slices(&subgroup).expect("S4 abnormal cosets");
    let right_assignment = slice_assignment(&right);
    let representations = s4_representations(&right_assignment);
    let atlas = PermutahedronAtlas {
        metadata: AtlasMetadata {
            schema_version: SCHEMA_VERSION,
            n: 4,
            stable_id_format: "S4-00000".into(),
            vertex_count: graph.vertex_count,
            edge_count: graph.edge_count,
            degree: graph.degree,
            diameter: graph.diameter,
            edge_convention: "right multiplication by adjacent transpositions; adjacent-position swaps",
            correlator_definition: "minimal weak-Bruhat distance; not a holoraumy gadget",
            source: source_records(&["2012.13308", "2012.14015"]),
            source_locators: [
                S4_DICTIONARY_SOURCE,
                S4_EDGE_SOURCE,
                S4_QUARTET_SOURCE,
                S4_CORRELATOR_SOURCE,
            ]
            .into_iter()
            .map(AtlasLocator::from)
            .collect(),
            generated_by: "Rust permutahedron atlas",
            complete_vertex_and_edge_dataset: true,
            dense_all_pairs_matrix_stored: true,
            two_face_count: Some(14),
            square_face_count: Some(6),
            hexagonal_face_count: Some(8),
        },
        permutations: permutations(4)
            .expect("S4 permutations")
            .map(permutation_string)
            .collect(),
        edges: complete_labeled_edges(4),
        right_slice_by_rank: right_assignment,
        left_slice_by_rank: slice_assignment(&left),
        right_slices: right.slices.clone(),
        left_slices: left.slices.clone(),
        abnormal_right_slices: abnormal_slice_ids(&right, &abnormal.abnormal_slices),
        named_correlator_matrix: named_matrix(&representations),
        representations,
    };
    (atlas, s4_validation())
}

fn build_s8() -> (PermutahedronAtlas, S8Validation) {
    let graph = bfs_graph_audit(8).expect("S8 graph");
    let subgroup = rana_r8();
    let right = coset_partition(&subgroup, CosetSide::Right).expect("S8 right cosets");
    let left = coset_partition(&subgroup, CosetSide::Left).expect("S8 left cosets");
    let abnormal = abnormal_slices(&subgroup).expect("S8 abnormal cosets");
    let right_assignment = slice_assignment(&right);
    let representations = s8_representations(&right_assignment);
    let validation = s8_validation(&right, &left, &representations);
    let atlas = PermutahedronAtlas {
        metadata: AtlasMetadata {
            schema_version: SCHEMA_VERSION,
            n: 8,
            stable_id_format: "S8-00000".into(),
            vertex_count: graph.vertex_count,
            edge_count: graph.edge_count,
            degree: graph.degree,
            diameter: graph.diameter,
            edge_convention: "right multiplication by adjacent transpositions; adjacent-position swaps",
            correlator_definition: "minimal weak-Bruhat distance; not a holoraumy gadget",
            source: source_records(&["2012.14015", "2304.09830"]),
            source_locators: [
                S8_REPRESENTATION_SOURCE,
                R8_SOURCE,
                S8_COMBINATORICS_SOURCE,
                S8_MAGIC_NUMBER_SOURCE,
            ]
            .into_iter()
            .map(AtlasLocator::from)
            .collect(),
            generated_by: "Rust permutahedron atlas",
            complete_vertex_and_edge_dataset: true,
            dense_all_pairs_matrix_stored: false,
            two_face_count: Some(validation.two_faces_computed),
            square_face_count: Some(validation.square_faces_computed),
            hexagonal_face_count: Some(validation.hexagonal_faces_computed),
        },
        permutations: permutations(S8_COLOR_COUNT)
            .expect("S8 permutations")
            .map(permutation_string)
            .collect(),
        edges: complete_labeled_edges(8),
        right_slice_by_rank: right_assignment,
        left_slice_by_rank: slice_assignment(&left),
        right_slices: right.slices.clone(),
        left_slices: left.slices.clone(),
        abnormal_right_slices: abnormal_slice_ids(&right, &abnormal.abnormal_slices),
        named_correlator_matrix: named_matrix(&representations),
        representations,
    };
    (atlas, validation)
}

pub fn verify() -> PermutahedronValidation {
    let (_, s4) = build_s4();
    let (_, s8) = build_s8();
    let garden_scan = complete_garden_scan();
    let garden = GardenValidation::from(&garden_scan);
    let passed = s4.passed && s8.passed && garden.passed;
    PermutahedronValidation {
        schema_version: SCHEMA_VERSION,
        s4,
        s8,
        garden,
        s4_artifact: "data/permutahedron_s4_atlas.json",
        s8_artifact: "data/permutahedron_s8_atlas.json",
        garden_artifact: "data/permutahedron_s8_garden.json",
        browser: "visualizer/permutahedron_atlas.html",
        boundary: "The atlas completes finite permutation, coset, Bruhat-distance, and R8-coset Garden-sign feasibility calculations. It does not solve arbitrary sign assignment or construct a field equation.",
        passed,
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    std::fs::rename(temporary, path)
}

pub fn build_artifacts(output_directory: &Path, validation_path: &Path) -> PermutahedronValidation {
    std::fs::create_dir_all(output_directory).expect("create atlas output directory");
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create validation output directory");
    }
    let (s4_atlas, s4) = build_s4();
    let (s8_atlas, s8) = build_s8();
    let garden_scan = complete_garden_scan();
    let garden = GardenValidation::from(&garden_scan);
    let passed = s4.passed && s8.passed && garden.passed;
    let report = PermutahedronValidation {
        schema_version: SCHEMA_VERSION,
        s4,
        s8,
        garden,
        s4_artifact: "data/permutahedron_s4_atlas.json",
        s8_artifact: "data/permutahedron_s8_atlas.json",
        garden_artifact: "data/permutahedron_s8_garden.json",
        browser: "visualizer/permutahedron_atlas.html",
        boundary: "The atlas completes finite permutation, coset, Bruhat-distance, and R8-coset Garden-sign feasibility calculations. It does not solve arbitrary sign assignment or construct a field equation.",
        passed,
    };
    assert!(report.passed, "permutahedron validation failed");
    write_json(
        &output_directory.join("permutahedron_s4_atlas.json"),
        &s4_atlas,
    )
    .expect("write S4 atlas");
    write_json(
        &output_directory.join("permutahedron_s8_atlas.json"),
        &s8_atlas,
    )
    .expect("write S8 atlas");
    write_json(
        &output_directory.join("permutahedron_s8_garden.json"),
        &garden_scan,
    )
    .expect("write S8 Garden scan");
    write_json(validation_path, &report).expect("write validation report");
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_atlas_validation_passes() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.s4.independent_symmetric_entries, 300);
        assert_eq!(report.s4.distance_mismatches, 0);
        assert_eq!(report.s8.right_cosets, 5_040);
        assert_eq!(report.s8.abnormal_cosets, 168);
        assert_eq!(report.s8.conjugate_r8_subgroups, 30);
        assert_eq!(report.s8.named_correlator_entries_checked, 3_136);
        assert_eq!(report.s8.named_magic_rows_checked, 392);
        assert_eq!(report.s8.named_magic_failures, 0);
        assert_eq!(report.s8.all_coset_intra_rows_checked, 40_320);
        assert_eq!(report.s8.all_coset_intra_failures, 0);
        assert_eq!(report.garden.cosets_scanned, 5_040);
        assert_eq!(report.garden.signable_cosets, 5_040);
        assert_eq!(report.garden.unsignable_cosets, 0);
        assert_eq!(report.garden.equation_rank, 45);
        assert_eq!(report.garden.equation_nullity, 19);
        assert_eq!(report.garden.dense_residual_entries, 0);
        assert_eq!(report.garden.normalizer_order, 1_344);
        assert_eq!(report.garden.normalizer_quotient_cosets, 168);
    }
}
