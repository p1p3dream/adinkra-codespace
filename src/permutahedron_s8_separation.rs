//! First finite separation probe for the `S_8` permutahedron.
//!
//! The published `S_4` construction partitions the 24 permutations into six
//! `V_4` cosets.  The published paired `N=2` examples embed two such
//! four-color sectors into an eight-color `R_8` coset.  This module tests that
//! block-decomposition mechanism across every one of the 5,040 right `R_8`
//! cosets and every `R_8`-invariant split of the eight labels into two blocks
//! of four.

use crate::permutahedron::{CosetSide, Permutation, abnormal_slices, coset_partition, rana_r8};
use crate::permutahedron_fixtures::{S4_ORDERED_QUARTETS, S8_REPRESENTATION_OCTETS};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-separation-probe-v1";
const S4_LABELS: [&str; 6] = ["P1", "P2", "P3", "P4", "P5", "P6"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct PairClass {
    pub first: u8,
    pub second: u8,
}

impl PairClass {
    fn new(first: usize, second: usize) -> Self {
        let first = first as u8 + 1;
        let second = second as u8 + 1;
        if first <= second {
            Self { first, second }
        } else {
            Self {
                first: second,
                second: first,
            }
        }
    }

    fn label(self) -> String {
        format!(
            "{}+{}",
            S4_LABELS[usize::from(self.first - 1)],
            S4_LABELS[usize::from(self.second - 1)]
        )
    }

    fn diagonal(self) -> bool {
        self.first == self.second
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PairClassCount {
    pub pair: PairClass,
    pub label: String,
    pub cosets: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PartitionRecord {
    pub id: usize,
    pub first_block: [u8; 4],
    pub second_block: [u8; 4],
    pub compatible_cosets: usize,
    pub pair_classes: Vec<PairClassCount>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecompositionRecord {
    pub partition_id: usize,
    pub pair: PairClass,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CosetRecord {
    pub coset_id: usize,
    pub representative_rank: u32,
    pub representative: Vec<u8>,
    pub left_right_coincident: bool,
    pub decompositions: Vec<DecompositionRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublishedOctetRecord {
    pub label: &'static str,
    pub coset_id: usize,
    pub standard_partition_pair: String,
    pub compatible_partition_count: usize,
    pub left_right_coincident: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub r8_cosets: usize,
    pub invariant_partitions: usize,
    pub compatible_cosets_per_partition: usize,
    pub total_compatible_incidences: usize,
    pub distinct_compatible_cosets: usize,
    pub pair_classes_per_partition: usize,
    pub diagonal_pair_incidences: usize,
    pub mixed_pair_incidences: usize,
    pub pair_diagonal_matches_left_right_coincidence: bool,
    pub published_octets_located: usize,
    pub published_pair_labels_matched: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeparationProbeArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub hypothesis: &'static str,
    pub sources: Vec<&'static str>,
    pub partitions: Vec<PartitionRecord>,
    pub compatible_cosets: Vec<CosetRecord>,
    pub compatibility_histogram: BTreeMap<usize, usize>,
    pub compatibility_histogram_by_coincidence: BTreeMap<&'static str, BTreeMap<usize, usize>>,
    pub published_octets: Vec<PublishedOctetRecord>,
    pub validation: ValidationRecord,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn mask_from_values(values: &[u8]) -> u8 {
    values
        .iter()
        .fold(0u8, |mask, &value| mask | (1 << (value - 1)))
}

fn values_from_mask(mask: u8) -> [u8; 4] {
    let values: Vec<u8> = (1..=8)
        .filter(|value| mask & (1 << (value - 1)) != 0)
        .collect();
    values.try_into().expect("four-element block")
}

fn image_mask(permutation: Permutation, mask: u8) -> u8 {
    let mut result = 0u8;
    for value in 1..=8u8 {
        if mask & (1 << (value - 1)) != 0 {
            let image = permutation.as_slice()[usize::from(value - 1)];
            result |= 1 << (image - 1);
        }
    }
    result
}

fn invariant_partitions(r8: &[Permutation]) -> Vec<([u8; 4], [u8; 4])> {
    let mut result = Vec::new();
    for mask in 0u16..=u16::from(u8::MAX) {
        let mask = mask as u8;
        if mask.count_ones() != 4 || mask & 1 == 0 {
            continue;
        }
        let complement = !mask;
        if r8.iter().all(|&element| {
            let image = image_mask(element, mask);
            image == mask || image == complement
        }) {
            result.push((values_from_mask(mask), values_from_mask(complement)));
        }
    }
    result.sort_unstable();
    result
}

fn local_permutation(permutation: Permutation, domain: &[u8; 4], codomain: &[u8; 4]) -> [u8; 4] {
    std::array::from_fn(|index| {
        let image = permutation.as_slice()[usize::from(domain[index] - 1)];
        codomain
            .iter()
            .position(|&value| value == image)
            .map(|position| position as u8 + 1)
            .expect("block-compatible image")
    })
}

fn s4_sector(permutation: [u8; 4]) -> usize {
    S4_ORDERED_QUARTETS
        .iter()
        .position(|quartet| quartet.contains(&permutation))
        .expect("S4 quartets partition S4")
}

fn pair_class_for_partition(
    permutation: Permutation,
    partition: &([u8; 4], [u8; 4]),
) -> Option<PairClass> {
    let first_mask = mask_from_values(&partition.0);
    let second_mask = mask_from_values(&partition.1);
    let first_image = image_mask(permutation, first_mask);
    let (first_target, second_target) = if first_image == first_mask {
        (&partition.0, &partition.1)
    } else if first_image == second_mask {
        (&partition.1, &partition.0)
    } else {
        return None;
    };
    let first = s4_sector(local_permutation(permutation, &partition.0, first_target));
    let second = s4_sector(local_permutation(permutation, &partition.1, second_target));
    Some(PairClass::new(first, second))
}

fn octet_ranks(permutations: &[[u8; 8]; 8]) -> Vec<u32> {
    let mut ranks: Vec<u32> = permutations
        .iter()
        .map(|entry| {
            Permutation::new(entry)
                .expect("published permutation")
                .rank() as u32
        })
        .collect();
    ranks.sort_unstable();
    ranks
}

pub fn build() -> SeparationProbeArtifact {
    let r8 = rana_r8();
    let partition = coset_partition(&r8, CosetSide::Right).expect("R8 right cosets");
    let invariant = invariant_partitions(&r8);
    assert_eq!(invariant.len(), 7);

    let abnormal = abnormal_slices(&r8).expect("left-right coincidence");
    let abnormal_representatives: HashSet<u32> =
        abnormal.representative_ranks.iter().copied().collect();

    let mut partition_counts: Vec<BTreeMap<PairClass, usize>> =
        vec![BTreeMap::new(); invariant.len()];
    let mut compatible_cosets = Vec::new();
    let mut compatibility_histogram = BTreeMap::new();
    let mut compatibility_histogram_by_coincidence: BTreeMap<&'static str, BTreeMap<usize, usize>> =
        BTreeMap::new();
    let mut diagonal_pair_incidences = 0usize;
    let mut mixed_pair_incidences = 0usize;
    let mut diagonal_match = true;

    for (coset_id, slice) in partition.slices.iter().enumerate() {
        let members: Vec<Permutation> = slice
            .iter()
            .map(|&rank| Permutation::unrank(8, rank as usize).expect("S8 rank"))
            .collect();
        let left_right_coincident = abnormal_representatives.contains(&slice[0]);
        let mut decompositions = Vec::new();
        for (partition_id, block_partition) in invariant.iter().enumerate() {
            let classes: BTreeSet<PairClass> = members
                .iter()
                .filter_map(|&member| pair_class_for_partition(member, block_partition))
                .collect();
            let compatible_members = members
                .iter()
                .filter(|&&member| pair_class_for_partition(member, block_partition).is_some())
                .count();
            assert!(compatible_members == 0 || compatible_members == members.len());
            if compatible_members == 0 {
                continue;
            }
            assert_eq!(classes.len(), 1);
            let pair = *classes.first().expect("one pair class");
            *partition_counts[partition_id].entry(pair).or_default() += 1;
            diagonal_pair_incidences += usize::from(pair.diagonal());
            mixed_pair_incidences += usize::from(!pair.diagonal());
            diagonal_match &= pair.diagonal() == left_right_coincident;
            decompositions.push(DecompositionRecord {
                partition_id,
                pair,
                label: pair.label(),
            });
        }
        *compatibility_histogram
            .entry(decompositions.len())
            .or_default() += 1;
        let coincidence_key = if left_right_coincident {
            "left_right_coincident"
        } else {
            "other"
        };
        *compatibility_histogram_by_coincidence
            .entry(coincidence_key)
            .or_default()
            .entry(decompositions.len())
            .or_default() += 1;
        if !decompositions.is_empty() {
            let representative = Permutation::unrank(8, slice[0] as usize).expect("representative");
            compatible_cosets.push(CosetRecord {
                coset_id,
                representative_rank: slice[0],
                representative: representative.one_line(),
                left_right_coincident,
                decompositions,
            });
        }
    }

    let partitions: Vec<PartitionRecord> = invariant
        .iter()
        .enumerate()
        .map(|(id, &(first_block, second_block))| {
            let pair_classes = partition_counts[id]
                .iter()
                .map(|(&pair, &cosets)| PairClassCount {
                    pair,
                    label: pair.label(),
                    cosets,
                })
                .collect::<Vec<_>>();
            PartitionRecord {
                id,
                first_block,
                second_block,
                compatible_cosets: pair_classes.iter().map(|record| record.cosets).sum(),
                pair_classes,
            }
        })
        .collect();

    let by_slice: HashMap<Vec<u32>, usize> = partition
        .slices
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, slice)| (slice, id))
        .collect();
    let compatible_by_id: HashMap<usize, &CosetRecord> = compatible_cosets
        .iter()
        .map(|record| (record.coset_id, record))
        .collect();
    let expected_pairs = ["P1+P1", "P1+P2", "P1+P3", "P2+P2", "P2+P3", "P3+P3"];
    let published_octets: Vec<PublishedOctetRecord> = S8_REPRESENTATION_OCTETS
        .iter()
        .zip(expected_pairs)
        .map(|(octet, expected)| {
            let ranks = octet_ranks(&octet.permutations);
            let coset_id = *by_slice.get(&ranks).expect("published R8 coset");
            let record = compatible_by_id
                .get(&coset_id)
                .expect("published octet is block-compatible");
            let standard = record
                .decompositions
                .iter()
                .find(|decomposition| {
                    partitions[decomposition.partition_id].first_block == [1, 2, 3, 4]
                })
                .expect("standard partition");
            assert_eq!(standard.label, expected);
            PublishedOctetRecord {
                label: octet.label,
                coset_id,
                standard_partition_pair: standard.label.clone(),
                compatible_partition_count: record.decompositions.len(),
                left_right_coincident: record.left_right_coincident,
            }
        })
        .collect();

    let total_compatible_incidences: usize =
        partitions.iter().map(|entry| entry.compatible_cosets).sum();
    let each_partition_complete = partitions.iter().all(|entry| {
        entry.compatible_cosets == 144
            && entry.pair_classes.len() == 21
            && entry
                .pair_classes
                .iter()
                .all(|class| class.cosets == if class.pair.diagonal() { 4 } else { 8 })
    });
    let validation = ValidationRecord {
        r8_cosets: partition.slice_count,
        invariant_partitions: invariant.len(),
        compatible_cosets_per_partition: 144,
        total_compatible_incidences,
        distinct_compatible_cosets: compatible_cosets.len(),
        pair_classes_per_partition: 21,
        diagonal_pair_incidences,
        mixed_pair_incidences,
        pair_diagonal_matches_left_right_coincidence: diagonal_match,
        published_octets_located: published_octets.len(),
        published_pair_labels_matched: published_octets
            .iter()
            .zip(expected_pairs)
            .filter(|(record, expected)| record.standard_partition_pair == **expected)
            .count(),
        passed: partition.slice_count == 5_040
            && invariant.len() == 7
            && each_partition_complete
            && total_compatible_incidences == 1_008
            && diagonal_pair_incidences == 168
            && mixed_pair_incidences == 840
            && diagonal_match
            && published_octets.len() == 6
            && compatibility_histogram_by_coincidence["left_right_coincident"]
                == BTreeMap::from([(0, 48), (1, 98), (3, 21), (7, 1)])
            && compatibility_histogram_by_coincidence["other"]
                == BTreeMap::from([(0, 4_088), (1, 756), (3, 28)]),
    };

    SeparationProbeArtifact {
        schema_version: SCHEMA_VERSION,
        title: "S8 separation probe from paired S4 sectors",
        hypothesis: "An R8 coset that preserves an R8-invariant 4+4 split inherits an unordered pair of the six published S4 sectors.",
        sources: vec![
            "arXiv:2012.13308, six S4 quartets",
            "arXiv:2012.14015, six paired S8 systems",
            "arXiv:2304.09830, R8 hoppers and 5,040-coset partition",
        ],
        partitions,
        compatible_cosets,
        compatibility_histogram,
        compatibility_histogram_by_coincidence,
        published_octets,
        validation,
        findings: vec![
            "Each R8-invariant 4+4 split contains 144 complete R8 cosets.".into(),
            "Those 144 cosets divide into all 21 unordered pairs of the six S4 sectors, with four diagonal and eight mixed cosets per pair class.".into(),
            "Across the seven invariant splits there are 1,008 compatible incidences: 168 diagonal and 840 mixed.".into(),
            "For every compatible incidence, a diagonal S4 pair is equivalent to left-right coset coincidence.".into(),
            "The 168 left-right coincident cosets preserve zero, one, three, or seven invariant splits with multiplicities 48, 98, 21, and 1.".into(),
            "The six published CC, CT, CV, TT, TV, and VV octets occupy P1+P1, P1+P2, P1+P3, P2+P2, P2+P3, and P3+P3 respectively in the standard split.".into(),
        ],
        boundary: "This is a complete classification of the R8 cosets that decompose through an R8-invariant 4+4 split. It does not classify the remaining cosets as physical supermultiplets, prove that the 21 pair classes are inequivalent representations, or solve the full hex separation problem.",
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ValidationRecord {
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
    fn seven_four_plus_four_partitions_are_r8_invariant() {
        let partitions = invariant_partitions(&rana_r8());
        assert_eq!(partitions.len(), 7);
        assert!(partitions.contains(&([1, 2, 3, 4], [5, 6, 7, 8])));
    }

    #[test]
    fn complete_separation_probe_passes() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.r8_cosets, 5_040);
        assert_eq!(artifact.validation.total_compatible_incidences, 1_008);
        assert_eq!(artifact.validation.diagonal_pair_incidences, 168);
        assert_eq!(artifact.validation.mixed_pair_incidences, 840);
    }
}
