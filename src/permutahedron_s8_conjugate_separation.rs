//! Exact conjugacy-orbit extension of the `S_8` paired-sector separation probe.
//!
//! The thirty conjugates of `R_8` define thirty different partitions of
//! `S_8` into octets.  They are alternative octet universes, not additional
//! labels on the original 5,040 right `R_8` cosets.  This module enumerates the
//! conjugates canonically, repeats the four-plus-four scan in every universe,
//! and separately tests whether transporting all block partitions adds any
//! decompositions to the original universe.

use crate::permutahedron::{CosetSide, Permutation, coset_partition, factorial, rana_r8};
use crate::permutahedron_fixtures::S4_ORDERED_QUARTETS;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-conjugate-separation-v1";
const S4_LABELS: [&str; 6] = ["P1", "P2", "P3", "P4", "P5", "P6"];
type SubgroupKey = [u32; 8];
type OctetKey = [u32; 8];

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

    fn diagonal(self) -> bool {
        self.first == self.second
    }

    fn label(self) -> String {
        format!(
            "{}+{}",
            S4_LABELS[usize::from(self.first - 1)],
            S4_LABELS[usize::from(self.second - 1)]
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderedPairClass {
    first: u8,
    second: u8,
}

impl OrderedPairClass {
    fn new(first: usize, second: usize) -> Self {
        Self {
            first: first as u8 + 1,
            second: second as u8 + 1,
        }
    }

    fn diagonal(self) -> bool {
        self.first == self.second
    }
}

#[derive(Debug, Clone)]
struct CanonicalSubgroup {
    key: SubgroupKey,
    elements: Vec<Permutation>,
    least_conjugator_rank: u32,
    conjugators: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubgroupRecord {
    pub id: usize,
    pub element_ranks: SubgroupKey,
    pub least_conjugator_rank: u32,
    pub least_conjugator: Vec<u8>,
    pub conjugators: usize,
    pub invariant_partition_ids: Vec<usize>,
    pub invariant_partitions: usize,
    pub right_cosets: usize,
    pub block_compatible_cosets: usize,
    pub block_incompatible_cosets: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_cosets: usize,
    pub diagonal_only_cosets: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_complement_cosets: usize,
    pub block_compatible_incidences: usize,
    pub diagonal_incidences: usize,
    pub mixed_incidences: usize,
    pub abnormal_cosets: usize,
    pub block_compatibility_histogram: BTreeMap<usize, usize>,
    pub unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram:
        BTreeMap<usize, usize>,
    pub pair_histogram: BTreeMap<String, usize>,
    pub ordered_distinct_pair_occurrences: BTreeMap<String, usize>,
    pub ordered_class_cardinality_histogram: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalizerPartitionAction {
    pub normalizer_order: usize,
    pub distinct_actions: usize,
    pub invariant_partitions: usize,
    pub orbit_size_of_each_partition: usize,
    pub globally_fixed_partitions: usize,
    pub transitive: bool,
    pub consequence: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderedPairCorrespondence {
    pub conjugate_subgroups: usize,
    pub ordered_distinct_s4_pairs: usize,
    pub relation_edges: usize,
    pub expected_edges_for_bijection: usize,
    pub subgroups_per_ordered_pair: usize,
    pub ordered_pairs_per_subgroup: usize,
    pub occurrences_per_subgroup_ordered_pair: usize,
    pub unsigned_support_relation_is_bijection: bool,
    pub conclusion: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecursiveConstructionAudit {
    pub source_equations: &'static str,
    pub ordered_distinct_sector_pairs_tested: usize,
    pub distinct_recursive_unsigned_octets: usize,
    pub conjugate_r8_families_used: usize,
    pub all_octets_in_standard_r8_family: bool,
    pub pair_to_conjugate_family_relation_is_bijection: bool,
    pub conclusion: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExactInvariants {
    pub ambient_group_order: usize,
    pub subgroup_order: usize,
    pub normalizer_order_from_conjugator_fibers: usize,
    pub conjugacy_orbit_size: usize,
    pub conjugator_multiplicity_histogram: BTreeMap<usize, usize>,
    pub global_four_plus_four_partitions: usize,
    pub invariant_subgroups_per_partition_histogram: BTreeMap<usize, usize>,
    pub subgroup_pair_intersection_order_histogram: BTreeMap<usize, usize>,
    pub distinct_coset_octets_across_all_subgroups: usize,
    pub total_coset_octets_across_all_subgroups: usize,
    pub block_compatible_octets_across_all_subgroups: usize,
    pub block_incompatible_octets_across_all_subgroups: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_octets_across_all_subgroups: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_complement_octets_across_all_subgroups:
        usize,
    pub block_compatible_incidences_across_all_subgroups: usize,
    pub diagonal_incidences_across_all_subgroups: usize,
    pub mixed_incidences_across_all_subgroups: usize,
    pub compatible_subgroup_membership_histogram_per_permutation: BTreeMap<usize, usize>,
    pub compatible_incidence_membership_histogram_per_permutation: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OriginalGapTest {
    pub original_cosets: usize,
    pub original_invariant_partitions: usize,
    pub transported_partition_union: usize,
    pub compatible_with_original_partitions: usize,
    pub compatible_with_transported_union: usize,
    pub recovered_original_gap_cosets: usize,
    pub original_gap_cosets: usize,
    pub diagonal_only_cosets_excluded_by_distinct_pair_restriction: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_cosets: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_complement_cosets: usize,
    pub original_gap_octets_equal_compatible_octets_from_other_subgroups: usize,
    pub conclusion: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub conjugate_subgroups: usize,
    pub canonical_keys_unique: bool,
    pub conjugators_per_subgroup: usize,
    pub invariant_partitions_per_subgroup: usize,
    pub invariant_subgroups_per_global_partition: usize,
    pub cosets_per_subgroup: usize,
    pub block_compatible_cosets_per_subgroup: usize,
    pub block_incompatible_cosets_per_subgroup: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_cosets_per_subgroup: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_complement_cosets_per_subgroup: usize,
    pub block_compatibility_histogram_per_subgroup: BTreeMap<usize, usize>,
    pub unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram_per_subgroup:
        BTreeMap<usize, usize>,
    pub pair_class_counts_uniform: bool,
    pub diagonal_iff_left_right_coincident: bool,
    pub all_subgroup_scans_identical: bool,
    pub all_coset_octets_family_unique: bool,
    pub recursive_ordered_distinct_sector_pairs_tested: usize,
    pub recursive_distinct_unsigned_octets: usize,
    pub recursive_conjugate_r8_families_used: usize,
    pub recursive_all_octets_in_standard_r8_family: bool,
    pub recovered_original_gap_cosets: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConjugateSeparationArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<&'static str>,
    pub method: Vec<&'static str>,
    pub subgroups: Vec<SubgroupRecord>,
    pub normalizer_partition_action: NormalizerPartitionAction,
    pub ordered_pair_correspondence: OrderedPairCorrespondence,
    pub recursive_construction_audit: RecursiveConstructionAudit,
    pub exact_invariants: ExactInvariants,
    pub original_gap_test: OriginalGapTest,
    pub validation: ValidationRecord,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn conjugate(g: Permutation, h: Permutation) -> Permutation {
    g.compose(h)
        .expect("common S8 order")
        .compose(g.inverse())
        .expect("common S8 order")
}

fn subgroup_key(elements: &[Permutation]) -> SubgroupKey {
    let mut ranks: Vec<u32> = elements.iter().map(|p| p.rank() as u32).collect();
    ranks.sort_unstable();
    ranks.try_into().expect("order-eight subgroup")
}

fn canonical_conjugates() -> Vec<CanonicalSubgroup> {
    let r8 = rana_r8();
    let mut fibers: BTreeMap<SubgroupKey, (u32, usize)> = BTreeMap::new();
    for rank in 0..factorial(8).expect("S8 order") {
        let g = Permutation::unrank(8, rank).expect("S8 rank");
        let elements: Vec<Permutation> = r8.iter().map(|&h| conjugate(g, h)).collect();
        let key = subgroup_key(&elements);
        let entry = fibers.entry(key).or_insert((rank as u32, 0));
        entry.0 = entry.0.min(rank as u32);
        entry.1 += 1;
    }
    fibers
        .into_iter()
        .map(
            |(key, (least_conjugator_rank, conjugators))| CanonicalSubgroup {
                key,
                elements: key
                    .iter()
                    .map(|&rank| Permutation::unrank(8, rank as usize).expect("S8 rank"))
                    .collect(),
                least_conjugator_rank,
                conjugators,
            },
        )
        .collect()
}

fn all_block_masks() -> Vec<u8> {
    (0u16..=u16::from(u8::MAX))
        .map(|mask| mask as u8)
        .filter(|mask| mask.count_ones() == 4 && mask & 1 != 0)
        .collect()
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

fn block_compatible(permutation: Permutation, mask: u8) -> bool {
    let image = image_mask(permutation, mask);
    image == mask || image == !mask
}

fn canonical_mask(mask: u8) -> u8 {
    if mask & 1 != 0 { mask } else { !mask }
}

fn invariant_masks(subgroup: &[Permutation], all_masks: &[u8]) -> Vec<u8> {
    all_masks
        .iter()
        .copied()
        .filter(|&mask| {
            subgroup
                .iter()
                .all(|&element| block_compatible(element, mask))
        })
        .collect()
}

fn values_from_mask(mask: u8) -> [u8; 4] {
    (1..=8u8)
        .filter(|&value| mask & (1 << (value - 1)) != 0)
        .collect::<Vec<_>>()
        .try_into()
        .expect("four-element mask")
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
        .expect("published S4 quartets partition S4")
}

fn ordered_pair_class(permutation: Permutation, mask: u8) -> Option<OrderedPairClass> {
    let first = values_from_mask(mask);
    let second = values_from_mask(!mask);
    let first_image = image_mask(permutation, mask);
    let (first_target, second_target) = if first_image == mask {
        (&first, &second)
    } else if first_image == !mask {
        (&second, &first)
    } else {
        return None;
    };
    Some(OrderedPairClass::new(
        s4_sector(local_permutation(permutation, &first, first_target)),
        s4_sector(local_permutation(permutation, &second, second_target)),
    ))
}

fn pair_class(permutation: Permutation, mask: u8) -> Option<PairClass> {
    ordered_pair_class(permutation, mask)
        .map(|pair| PairClass::new(usize::from(pair.first - 1), usize::from(pair.second - 1)))
}

fn pullback_permutation(chart: Permutation, permutation: Permutation) -> Permutation {
    chart
        .inverse()
        .compose(permutation)
        .expect("S8 product")
        .compose(chart)
        .expect("S8 product")
}

fn pullback_mask(chart: Permutation, mask: u8) -> u8 {
    canonical_mask(image_mask(chart.inverse(), mask))
}

fn octet_key(slice: &[u32]) -> OctetKey {
    slice.try_into().expect("eight-element coset")
}

fn recursively_constructed_unsigned_octet(first: usize, second: usize) -> OctetKey {
    let mut ranks = Vec::with_capacity(8);
    for (&first_permutation, &second_permutation) in S4_ORDERED_QUARTETS[first]
        .iter()
        .zip(S4_ORDERED_QUARTETS[second].iter())
    {
        let block_diagonal: Vec<u8> = first_permutation
            .into_iter()
            .chain(second_permutation.into_iter().map(|value| value + 4))
            .collect();
        let block_off_diagonal: Vec<u8> = first_permutation
            .into_iter()
            .map(|value| value + 4)
            .chain(second_permutation)
            .collect();
        ranks.push(
            Permutation::new(&block_diagonal)
                .expect("recursive block-diagonal permutation")
                .rank() as u32,
        );
        ranks.push(
            Permutation::new(&block_off_diagonal)
                .expect("recursive block-off-diagonal permutation")
                .rank() as u32,
        );
    }
    ranks.sort_unstable();
    ranks.try_into().expect("eight recursive permutations")
}

fn right_coset_subgroup_key(octet: OctetKey) -> SubgroupKey {
    let representative = Permutation::unrank(8, octet[0] as usize).expect("S8 rank");
    subgroup_key(
        &octet
            .iter()
            .map(|&rank| {
                Permutation::unrank(8, rank as usize)
                    .expect("S8 rank")
                    .compose(representative.inverse())
                    .expect("S8 product")
            })
            .collect::<Vec<_>>(),
    )
}

fn histogram(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_default() += 1;
    }
    result
}

#[derive(Debug)]
struct SubgroupScan {
    record: SubgroupRecord,
    all_octets: Vec<OctetKey>,
    block_compatible_octets: Vec<OctetKey>,
    distinct_pair_octets: Vec<OctetKey>,
    block_incidence_octets: Vec<OctetKey>,
}

fn scan_subgroup(id: usize, subgroup: &CanonicalSubgroup, all_masks: &[u8]) -> SubgroupScan {
    let invariant = invariant_masks(&subgroup.elements, all_masks);
    let partition =
        coset_partition(&subgroup.elements, CosetSide::Right).expect("conjugate R8 right cosets");
    let chart = Permutation::unrank(8, subgroup.least_conjugator_rank as usize)
        .expect("least conjugator rank");
    let mut block_compatibility_histogram = BTreeMap::new();
    let mut distinct_pair_histogram = BTreeMap::new();
    let mut pair_histogram = BTreeMap::new();
    let mut ordered_distinct_pair_occurrences = BTreeMap::new();
    let mut ordered_class_cardinality_histogram = BTreeMap::new();
    let mut block_compatible_octets = Vec::new();
    let mut distinct_pair_octets = Vec::new();
    let mut block_incidence_octets = Vec::new();
    let mut block_compatible_incidences = 0usize;
    let mut diagonal_incidences = 0usize;
    let mut mixed_incidences = 0usize;
    let mut abnormal_cosets = 0usize;

    for slice in &partition.slices {
        let members: Vec<Permutation> = slice
            .iter()
            .map(|&rank| Permutation::unrank(8, rank as usize).expect("S8 rank"))
            .collect();
        let g = members[0];
        let mut left: Vec<u32> = subgroup
            .elements
            .iter()
            .map(|&h| g.compose(h).expect("S8 product").rank() as u32)
            .collect();
        left.sort_unstable();
        let coincident = left == *slice;
        abnormal_cosets += usize::from(coincident);

        let pulled_members: Vec<Permutation> = members
            .iter()
            .map(|&member| pullback_permutation(chart, member))
            .collect();
        let mut decompositions = 0usize;
        let mut distinct_pair_decompositions = 0usize;
        for &mask in &invariant {
            let pulled_mask = pullback_mask(chart, mask);
            let classes: BTreeSet<PairClass> = pulled_members
                .iter()
                .filter_map(|&member| pair_class(member, pulled_mask))
                .collect();
            let ordered_classes: BTreeSet<OrderedPairClass> = pulled_members
                .iter()
                .filter_map(|&member| ordered_pair_class(member, pulled_mask))
                .collect();
            let compatible_members = members
                .iter()
                .filter(|&&member| block_compatible(member, mask))
                .count();
            assert!(compatible_members == 0 || compatible_members == 8);
            if compatible_members == 0 {
                continue;
            }
            assert_eq!(classes.len(), 1);
            let class = *classes.first().expect("one pair class per incidence");
            assert_eq!(class.diagonal(), coincident);
            *pair_histogram.entry(class.label()).or_default() += 1;
            *ordered_class_cardinality_histogram
                .entry(ordered_classes.len())
                .or_default() += 1;
            if class.diagonal() {
                assert_eq!(ordered_classes.len(), 1);
            } else {
                assert!(ordered_classes.len() == 1 || ordered_classes.len() == 2);
                assert!(ordered_classes.iter().all(|pair| !pair.diagonal()));
                for pair in ordered_classes {
                    let label = format!(
                        "{}->{}",
                        S4_LABELS[usize::from(pair.first - 1)],
                        S4_LABELS[usize::from(pair.second - 1)]
                    );
                    *ordered_distinct_pair_occurrences.entry(label).or_default() += 1;
                }
                distinct_pair_decompositions += 1;
            }
            block_compatible_incidences += 1;
            diagonal_incidences += usize::from(class.diagonal());
            mixed_incidences += usize::from(!class.diagonal());
            decompositions += 1;
            block_incidence_octets.push(octet_key(slice));
        }
        *block_compatibility_histogram
            .entry(decompositions)
            .or_default() += 1;
        *distinct_pair_histogram
            .entry(distinct_pair_decompositions)
            .or_default() += 1;
        if decompositions != 0 {
            block_compatible_octets.push(octet_key(slice));
        }
        if distinct_pair_decompositions != 0 {
            distinct_pair_octets.push(octet_key(slice));
        }
    }

    SubgroupScan {
        record: SubgroupRecord {
            id,
            element_ranks: subgroup.key,
            least_conjugator_rank: subgroup.least_conjugator_rank,
            least_conjugator: Permutation::unrank(8, subgroup.least_conjugator_rank as usize)
                .expect("S8 rank")
                .one_line(),
            conjugators: subgroup.conjugators,
            invariant_partition_ids: invariant
                .iter()
                .map(|mask| {
                    all_masks
                        .iter()
                        .position(|candidate| candidate == mask)
                        .unwrap()
                })
                .collect(),
            invariant_partitions: invariant.len(),
            right_cosets: partition.slice_count,
            block_compatible_cosets: block_compatible_octets.len(),
            block_incompatible_cosets: partition.slice_count - block_compatible_octets.len(),
            unsigned_noncoincident_distinct_sector_candidate_cosets: distinct_pair_octets.len(),
            diagonal_only_cosets: block_compatible_octets.len() - distinct_pair_octets.len(),
            unsigned_noncoincident_distinct_sector_candidate_complement_cosets: partition
                .slice_count
                - distinct_pair_octets.len(),
            block_compatible_incidences,
            diagonal_incidences,
            mixed_incidences,
            abnormal_cosets,
            block_compatibility_histogram,
            unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram:
                distinct_pair_histogram,
            pair_histogram,
            ordered_distinct_pair_occurrences,
            ordered_class_cardinality_histogram,
        },
        all_octets: partition
            .slices
            .iter()
            .map(|slice| octet_key(slice))
            .collect(),
        block_compatible_octets,
        distinct_pair_octets,
        block_incidence_octets,
    }
}

fn normalizer_partition_action(
    subgroup: &CanonicalSubgroup,
    all_masks: &[u8],
) -> NormalizerPartitionAction {
    let key = subgroup.key;
    let invariant = invariant_masks(&subgroup.elements, all_masks);
    let mut actions = BTreeSet::new();
    let mut orbit = BTreeSet::new();
    let mut globally_fixed = vec![true; invariant.len()];
    let mut normalizer_order = 0usize;
    for rank in 0..factorial(8).unwrap() {
        let g = Permutation::unrank(8, rank).unwrap();
        let conjugated: Vec<Permutation> = subgroup
            .elements
            .iter()
            .map(|&element| conjugate(g, element))
            .collect();
        if subgroup_key(&conjugated) != key {
            continue;
        }
        normalizer_order += 1;
        let action: Vec<u8> = invariant
            .iter()
            .map(|&mask| {
                let image = canonical_mask(image_mask(g, mask));
                invariant
                    .iter()
                    .position(|&candidate| candidate == image)
                    .expect("normalizer preserves invariant partitions") as u8
            })
            .collect();
        orbit.insert(action[0]);
        for (index, &image) in action.iter().enumerate() {
            globally_fixed[index] &= usize::from(image) == index;
        }
        actions.insert(action);
    }
    NormalizerPartitionAction {
        normalizer_order,
        distinct_actions: actions.len(),
        invariant_partitions: invariant.len(),
        orbit_size_of_each_partition: orbit.len(),
        globally_fixed_partitions: globally_fixed.into_iter().filter(|&fixed| fixed).count(),
        transitive: orbit.len() == invariant.len(),
        consequence: "The subgroup and its intrinsic automorphisms do not distinguish one of its seven four-plus-four partitions or an ordering of the two blocks.",
    }
}

pub fn build() -> ConjugateSeparationArtifact {
    let all_masks = all_block_masks();
    let subgroups = canonical_conjugates();
    assert_eq!(subgroups.len(), 30);

    let mut partition_subgroup_multiplicity = vec![0usize; all_masks.len()];
    let mut scans = Vec::with_capacity(subgroups.len());
    for (id, subgroup) in subgroups.iter().enumerate() {
        for mask in invariant_masks(&subgroup.elements, &all_masks) {
            let partition_id = all_masks
                .iter()
                .position(|candidate| *candidate == mask)
                .unwrap();
            partition_subgroup_multiplicity[partition_id] += 1;
        }
        scans.push(scan_subgroup(id, subgroup, &all_masks));
    }

    let mut all_octets = BTreeSet::new();
    let mut all_block_compatible_octets = BTreeSet::new();
    let mut all_distinct_pair_octets = BTreeSet::new();
    let mut compatible_membership = vec![0usize; factorial(8).unwrap()];
    let mut incidence_membership = vec![0usize; factorial(8).unwrap()];
    for scan in &scans {
        for &octet in &scan.all_octets {
            all_octets.insert(octet);
        }
        for &octet in &scan.block_compatible_octets {
            all_block_compatible_octets.insert(octet);
            for rank in octet {
                compatible_membership[rank as usize] += 1;
            }
        }
        for &octet in &scan.distinct_pair_octets {
            all_distinct_pair_octets.insert(octet);
        }
        for &octet in &scan.block_incidence_octets {
            for rank in octet {
                incidence_membership[rank as usize] += 1;
            }
        }
    }

    let original_key = subgroup_key(&rana_r8());
    let original_id = subgroups
        .iter()
        .position(|subgroup| subgroup.key == original_key)
        .expect("R8 occurs in its conjugacy class");
    let recursive_octets: Vec<OctetKey> = (0..6)
        .flat_map(|first| (0..6).map(move |second| (first, second)))
        .filter(|(first, second)| first != second)
        .map(|(first, second)| recursively_constructed_unsigned_octet(first, second))
        .collect();
    let recursive_distinct_octets: BTreeSet<OctetKey> = recursive_octets.iter().copied().collect();
    let recursive_subgroup_keys: BTreeSet<SubgroupKey> = recursive_octets
        .iter()
        .copied()
        .map(right_coset_subgroup_key)
        .collect();
    assert!(
        recursive_subgroup_keys
            .iter()
            .all(|key| subgroups.iter().any(|subgroup| subgroup.key == *key))
    );
    let recursive_construction_audit = RecursiveConstructionAudit {
        source_equations: "arXiv:2304.09830v2, Eqs. (2.17)-(2.19)",
        ordered_distinct_sector_pairs_tested: recursive_octets.len(),
        distinct_recursive_unsigned_octets: recursive_distinct_octets.len(),
        conjugate_r8_families_used: recursive_subgroup_keys.len(),
        all_octets_in_standard_r8_family: recursive_subgroup_keys == BTreeSet::from([original_key]),
        pair_to_conjugate_family_relation_is_bijection: recursive_subgroup_keys.len()
            == recursive_octets.len(),
        conclusion: "The paper's deterministic unsigned recursive construction gives 30 distinct octets, but all 30 are right cosets of the same standard R8 subgroup. Its ordered-pair to conjugate-family map is constant, not a 30-to-30 bijection. This resolves only the unsigned recursive construction tested here, not the broader relation left open in the paper.",
    };
    let original = &scans[original_id];
    let original_invariant = invariant_masks(&subgroups[original_id].elements, &all_masks);
    let original_partition = coset_partition(&subgroups[original_id].elements, CosetSide::Right)
        .expect("R8 right cosets");
    let mut compatible_original = BTreeSet::new();
    let mut compatible_global = BTreeSet::new();
    for slice in &original_partition.slices {
        let members: Vec<Permutation> = slice
            .iter()
            .map(|&rank| Permutation::unrank(8, rank as usize).unwrap())
            .collect();
        if original_invariant
            .iter()
            .any(|&mask| members.iter().all(|&member| block_compatible(member, mask)))
        {
            compatible_original.insert(octet_key(slice));
        }
        if all_masks
            .iter()
            .any(|&mask| members.iter().all(|&member| block_compatible(member, mask)))
        {
            compatible_global.insert(octet_key(slice));
        }
    }
    let original_gap: BTreeSet<OctetKey> = original
        .all_octets
        .iter()
        .copied()
        .filter(|octet| !compatible_original.contains(octet))
        .collect();
    let compatible_other: BTreeSet<OctetKey> = scans
        .iter()
        .enumerate()
        .filter(|(id, _)| *id != original_id)
        .flat_map(|(_, scan)| scan.block_compatible_octets.iter().copied())
        .collect();
    let gap_overlap = original_gap.intersection(&compatible_other).count();

    let mut intersection_orders = Vec::new();
    for i in 0..subgroups.len() {
        let first: HashSet<u32> = subgroups[i].key.iter().copied().collect();
        for second in &subgroups[i + 1..] {
            intersection_orders.push(
                second
                    .key
                    .iter()
                    .filter(|rank| first.contains(rank))
                    .count(),
            );
        }
    }

    let first_record = &scans[0].record;
    let scans_identical = scans.iter().all(|scan| {
        let record = &scan.record;
        record.invariant_partitions == first_record.invariant_partitions
            && record.right_cosets == first_record.right_cosets
            && record.block_compatible_cosets == first_record.block_compatible_cosets
            && record.block_incompatible_cosets == first_record.block_incompatible_cosets
            && record.unsigned_noncoincident_distinct_sector_candidate_cosets
                == first_record.unsigned_noncoincident_distinct_sector_candidate_cosets
            && record.diagonal_only_cosets == first_record.diagonal_only_cosets
            && record.unsigned_noncoincident_distinct_sector_candidate_complement_cosets
                == first_record.unsigned_noncoincident_distinct_sector_candidate_complement_cosets
            && record.block_compatible_incidences == first_record.block_compatible_incidences
            && record.diagonal_incidences == first_record.diagonal_incidences
            && record.mixed_incidences == first_record.mixed_incidences
            && record.abnormal_cosets == first_record.abnormal_cosets
            && record.block_compatibility_histogram == first_record.block_compatibility_histogram
            && record.unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram
                == first_record
                    .unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram
            && record.pair_histogram == first_record.pair_histogram
            && record.ordered_distinct_pair_occurrences
                == first_record.ordered_distinct_pair_occurrences
            && record.ordered_class_cardinality_histogram
                == first_record.ordered_class_cardinality_histogram
    });
    let expected_pair_histogram: BTreeMap<String, usize> = (0..6)
        .flat_map(|first| (first..6).map(move |second| PairClass::new(first, second)))
        .map(|pair| (pair.label(), if pair.diagonal() { 28 } else { 56 }))
        .collect();
    let pair_uniform = scans
        .iter()
        .all(|scan| scan.record.pair_histogram == expected_pair_histogram);
    let expected_ordered_occurrences: BTreeMap<String, usize> = (0..6)
        .flat_map(|first| (0..6).map(move |second| (first, second)))
        .filter(|(first, second)| first != second)
        .map(|(first, second)| (format!("{}->{}", S4_LABELS[first], S4_LABELS[second]), 28))
        .collect();
    let ordered_relation_uniform = scans
        .iter()
        .all(|scan| scan.record.ordered_distinct_pair_occurrences == expected_ordered_occurrences);
    let ordered_classes_singleton = scans.iter().all(|scan| {
        scan.record.ordered_class_cardinality_histogram == BTreeMap::from([(1, 1_008)])
    });
    let diagonal_iff = scans.iter().all(|scan| {
        scan.record.diagonal_incidences == scan.record.abnormal_cosets
            && scan.record.diagonal_incidences == 168
    });

    let total_cosets: usize = scans.iter().map(|scan| scan.record.right_cosets).sum();
    let block_compatible_octets: usize = scans
        .iter()
        .map(|scan| scan.record.block_compatible_cosets)
        .sum();
    let block_incompatible_octets: usize = scans
        .iter()
        .map(|scan| scan.record.block_incompatible_cosets)
        .sum();
    let distinct_pair_octets: usize = scans
        .iter()
        .map(|scan| {
            scan.record
                .unsigned_noncoincident_distinct_sector_candidate_cosets
        })
        .sum();
    let distinct_pair_gap_octets: usize = scans
        .iter()
        .map(|scan| {
            scan.record
                .unsigned_noncoincident_distinct_sector_candidate_complement_cosets
        })
        .sum();
    let block_compatible_incidences: usize = scans
        .iter()
        .map(|scan| scan.record.block_compatible_incidences)
        .sum();
    let diagonal_incidences: usize = scans
        .iter()
        .map(|scan| scan.record.diagonal_incidences)
        .sum();
    let mixed_incidences: usize = scans.iter().map(|scan| scan.record.mixed_incidences).sum();

    let normalizer_partition_action =
        normalizer_partition_action(&subgroups[original_id], &all_masks);
    let ordered_pair_correspondence = OrderedPairCorrespondence {
        conjugate_subgroups: subgroups.len(),
        ordered_distinct_s4_pairs: 30,
        relation_edges: subgroups.len() * 30,
        expected_edges_for_bijection: 30,
        subgroups_per_ordered_pair: 30,
        ordered_pairs_per_subgroup: 30,
        occurrences_per_subgroup_ordered_pair: 28,
        unsigned_support_relation_is_bijection: false,
        conclusion: "The tested unsigned support relation is complete, not one-to-one: every conjugate R8 subgroup realizes every ordered distinct S4 pair in its coset scan. A lexicographic matching of the two 30-element sets would be arbitrary, not a correspondence induced by this unsigned support relation. This does not settle whether additional signed or higher-dimensional data define a correspondence.",
    };

    let exact_invariants = ExactInvariants {
        ambient_group_order: factorial(8).unwrap(),
        subgroup_order: 8,
        normalizer_order_from_conjugator_fibers: subgroups[0].conjugators,
        conjugacy_orbit_size: subgroups.len(),
        conjugator_multiplicity_histogram: histogram(
            subgroups.iter().map(|subgroup| subgroup.conjugators),
        ),
        global_four_plus_four_partitions: all_masks.len(),
        invariant_subgroups_per_partition_histogram: histogram(
            partition_subgroup_multiplicity.iter().copied(),
        ),
        subgroup_pair_intersection_order_histogram: histogram(intersection_orders),
        distinct_coset_octets_across_all_subgroups: all_octets.len(),
        total_coset_octets_across_all_subgroups: total_cosets,
        block_compatible_octets_across_all_subgroups: block_compatible_octets,
        block_incompatible_octets_across_all_subgroups: block_incompatible_octets,
        unsigned_noncoincident_distinct_sector_candidate_octets_across_all_subgroups:
            distinct_pair_octets,
        unsigned_noncoincident_distinct_sector_candidate_complement_octets_across_all_subgroups:
            distinct_pair_gap_octets,
        block_compatible_incidences_across_all_subgroups: block_compatible_incidences,
        diagonal_incidences_across_all_subgroups: diagonal_incidences,
        mixed_incidences_across_all_subgroups: mixed_incidences,
        compatible_subgroup_membership_histogram_per_permutation: histogram(compatible_membership),
        compatible_incidence_membership_histogram_per_permutation: histogram(incidence_membership),
    };
    let original_gap_test = OriginalGapTest {
        original_cosets: original.record.right_cosets,
        original_invariant_partitions: original_invariant.len(),
        transported_partition_union: all_masks.len(),
        compatible_with_original_partitions: compatible_original.len(),
        compatible_with_transported_union: compatible_global.len(),
        recovered_original_gap_cosets: compatible_global.len() - compatible_original.len(),
        original_gap_cosets: original_gap.len(),
        diagonal_only_cosets_excluded_by_distinct_pair_restriction: original
            .record
            .diagonal_only_cosets,
        unsigned_noncoincident_distinct_sector_candidate_cosets: original
            .record
            .unsigned_noncoincident_distinct_sector_candidate_cosets,
        unsigned_noncoincident_distinct_sector_candidate_complement_cosets: original
            .record
            .unsigned_noncoincident_distinct_sector_candidate_complement_cosets,
        original_gap_octets_equal_compatible_octets_from_other_subgroups: gap_overlap,
        conclusion: "The 28 additional transported block partitions recover none of the 4,136 original block-incompatible R8 cosets. Enforcing the paper's distinct-pair restriction also removes 120 diagonal-only cosets, leaving 784 unsigned noncoincident distinct-sector candidates and a 4,256-coset complement. Other conjugate subgroups repartition S8 into different octets; they do not fill either gap in the fixed R8-coset universe.",
    };
    let expected_histogram = BTreeMap::from([(0, 4_136), (1, 854), (3, 49), (7, 1)]);
    let expected_distinct_histogram = BTreeMap::from([(0, 4_256), (1, 756), (3, 28)]);
    let validation = ValidationRecord {
        conjugate_subgroups: subgroups.len(),
        canonical_keys_unique: subgroups
            .iter()
            .map(|subgroup| subgroup.key)
            .collect::<BTreeSet<_>>()
            .len()
            == subgroups.len(),
        conjugators_per_subgroup: 1_344,
        invariant_partitions_per_subgroup: 7,
        invariant_subgroups_per_global_partition: 6,
        cosets_per_subgroup: 5_040,
        block_compatible_cosets_per_subgroup: 904,
        block_incompatible_cosets_per_subgroup: 4_136,
        unsigned_noncoincident_distinct_sector_candidate_cosets_per_subgroup: 784,
        unsigned_noncoincident_distinct_sector_candidate_complement_cosets_per_subgroup: 4_256,
        block_compatibility_histogram_per_subgroup: expected_histogram.clone(),
        unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram_per_subgroup:
            expected_distinct_histogram.clone(),
        pair_class_counts_uniform: pair_uniform,
        diagonal_iff_left_right_coincident: diagonal_iff,
        all_subgroup_scans_identical: scans_identical,
        all_coset_octets_family_unique: all_octets.len() == total_cosets,
        recursive_ordered_distinct_sector_pairs_tested: recursive_construction_audit
            .ordered_distinct_sector_pairs_tested,
        recursive_distinct_unsigned_octets: recursive_construction_audit
            .distinct_recursive_unsigned_octets,
        recursive_conjugate_r8_families_used: recursive_construction_audit
            .conjugate_r8_families_used,
        recursive_all_octets_in_standard_r8_family: recursive_construction_audit
            .all_octets_in_standard_r8_family,
        recovered_original_gap_cosets: original_gap_test.recovered_original_gap_cosets,
        passed: subgroups.len() == 30
            && subgroups
                .iter()
                .all(|subgroup| subgroup.conjugators == 1_344)
            && partition_subgroup_multiplicity
                .iter()
                .all(|&count| count == 6)
            && scans.iter().all(|scan| {
                scan.record.invariant_partitions == 7
                    && scan.record.right_cosets == 5_040
                    && scan.record.block_compatible_cosets == 904
                    && scan.record.block_incompatible_cosets == 4_136
                    && scan
                        .record
                        .unsigned_noncoincident_distinct_sector_candidate_cosets
                        == 784
                    && scan.record.diagonal_only_cosets == 120
                    && scan
                        .record
                        .unsigned_noncoincident_distinct_sector_candidate_complement_cosets
                        == 4_256
                    && scan.record.block_compatible_incidences == 1_008
                    && scan.record.block_compatibility_histogram == expected_histogram
                    && scan
                        .record
                        .unsigned_noncoincident_distinct_sector_candidate_multiplicity_histogram
                        == expected_distinct_histogram
            })
            && scans_identical
            && pair_uniform
            && ordered_relation_uniform
            && ordered_classes_singleton
            && diagonal_iff
            && normalizer_partition_action.normalizer_order == 1_344
            && normalizer_partition_action.distinct_actions == 168
            && normalizer_partition_action.orbit_size_of_each_partition == 7
            && normalizer_partition_action.globally_fixed_partitions == 0
            && normalizer_partition_action.transitive
            && all_octets.len() == 151_200
            && all_block_compatible_octets.len() == 27_120
            && all_distinct_pair_octets.len() == 23_520
            && block_compatible_octets == 27_120
            && block_incompatible_octets == 124_080
            && distinct_pair_octets == 23_520
            && distinct_pair_gap_octets == 127_680
            && block_compatible_incidences == 30_240
            && diagonal_incidences == 5_040
            && mixed_incidences == 25_200
            && compatible_original.len() == 904
            && compatible_global.len() == 904
            && original_gap.len() == 4_136
            && gap_overlap == 0
            && recursive_construction_audit.ordered_distinct_sector_pairs_tested == 30
            && recursive_construction_audit.distinct_recursive_unsigned_octets == 30
            && recursive_construction_audit.conjugate_r8_families_used == 1
            && recursive_construction_audit.all_octets_in_standard_r8_family
            && !recursive_construction_audit.pair_to_conjugate_family_relation_is_bijection,
    };

    ConjugateSeparationArtifact {
        schema_version: SCHEMA_VERSION,
        title: "S8 paired-sector separation across all conjugate R8 subgroups",
        sources: vec![
            "arXiv:2304.09830v2, Eqs. (2.17)-(2.19), Sec. 3.3, and conclusion: deterministic unsigned recursion, 30 conjugate R8 subgroups, and 6 x 5 ordered distinct S4 pairs, with the relation between the two counts left as a future question",
            "arXiv:1608.07864, Theorem 3 and Fig. 7: GR(8,8) permutation sets are cosets of 30 conjugates of the order-eight translation subgroup",
            "arXiv:2012.13308v6: six S4 V4-coset sectors",
        ],
        method: vec![
            "Enumerate g R8 g^-1 for every g in lexicographic S8 order and deduplicate by sorted Lehmer-rank subgroup key.",
            "For each canonical subgroup, enumerate its right cosets and seven invariant unordered four-plus-four partitions.",
            "Pull each scan back through its canonical least-rank conjugator and classify every compatible incidence by the published S4 V4-coset sectors induced on the blocks.",
            "Test the paper's proposed 30-to-30 correspondence using ordered distinct pairs only; retain diagonal block-compatible incidences separately because the conclusion of arXiv:2304.09830 excludes them from the decomposable basis.",
            "Apply the deterministic unsigned recursion in Eqs. (2.17)-(2.19) to all 30 ordered distinct S4-sector pairs and identify the unique conjugate R8 subgroup whose right-coset family contains each resulting octet.",
            "Apply the union of all transported block partitions directly to the original R8 cosets to test the stated 4,136-gap question without changing octet identity.",
        ],
        subgroups: scans.into_iter().map(|scan| scan.record).collect(),
        normalizer_partition_action,
        ordered_pair_correspondence,
        recursive_construction_audit,
        exact_invariants,
        original_gap_test,
        validation,
        findings: vec![
            "The conjugacy orbit contains 30 subgroups. Each canonical subgroup key has 1,344 conjugators, reproducing the normalizer order by orbit-stabilizer.".into(),
            "The thirty subgroups collectively preserve all 35 unordered four-plus-four partitions, with each partition preserved by exactly six subgroups.".into(),
            "Every conjugate-subgroup scan has the same 904 compatible and 4,136 incompatible cosets, with decomposition multiplicities 0:4,136, 1:854, 3:49, and 7:1.".into(),
            "Applying the paper's distinct-pair restriction gives 784 unsigned noncoincident distinct-sector candidates per subgroup and excludes 120 diagonal-only cosets; the resulting multiplicities are 0:4,256, 1:756, and 3:28.".into(),
            "The broad unsigned support relation is complete: every conjugate subgroup realizes all 30 ordered distinct S4 pairs, each 28 times, so that relation has 900 edges rather than 30.".into(),
            "The deterministic unsigned recursion in arXiv:2304.09830 Eqs. (2.17)-(2.19) produces 30 distinct octets, all in the standard R8 right-coset family. Its pair-to-family map is constant rather than bijective.".into(),
            "The normalizer acts transitively on each subgroup's seven invariant four-plus-four partitions and fixes none globally, so the subgroup itself supplies neither a distinguished split nor a block ordering.".into(),
            "The thirty right-coset families contain 151,200 distinct octet sets. A coset octet determines its subgroup, so different conjugates do not supply additional decompositions of a fixed original octet.".into(),
            "Testing all 35 transported block partitions against the original 5,040 R8 cosets leaves the compatible count at 904 and recovers zero of the 4,136-gap cosets.".into(),
        ],
        boundary: "This is an exact unsigned permutation and block-decomposition calculation. Conjugate R8 subgroups are relabelings that define different octet partitions of S8. The result neither excludes non-paired decomposition mechanisms nor supplies Boolean factors, Garden closure, HYMN, holoraumy, or a physical representation classification.",
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
    fn recursive_unsigned_distinct_pair_construction_uses_one_conjugate_family() {
        let conjugates = canonical_conjugates();
        let octets: Vec<OctetKey> = (0..6)
            .flat_map(|first| (0..6).map(move |second| (first, second)))
            .filter(|(first, second)| first != second)
            .map(|(first, second)| recursively_constructed_unsigned_octet(first, second))
            .collect();
        let subgroup_ids: BTreeSet<usize> = octets
            .iter()
            .copied()
            .map(|octet| {
                let key = right_coset_subgroup_key(octet);
                conjugates
                    .iter()
                    .position(|subgroup| subgroup.key == key)
                    .expect("recursive support must be a right coset of a conjugate R8")
            })
            .collect();
        assert_eq!(octets.iter().copied().collect::<BTreeSet<_>>().len(), 30);
        assert_eq!(subgroup_ids.len(), 1);
        assert_eq!(
            conjugates[*subgroup_ids.first().unwrap()].key,
            subgroup_key(&rana_r8())
        );
    }

    #[test]
    fn canonical_conjugate_enumeration_has_thirty_equal_fibers() {
        let conjugates = canonical_conjugates();
        assert_eq!(conjugates.len(), 30);
        assert!(
            conjugates
                .iter()
                .all(|subgroup| subgroup.conjugators == 1_344)
        );
    }

    #[test]
    fn complete_conjugate_separation_scan_passes() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.conjugate_subgroups, 30);
        assert_eq!(artifact.validation.recovered_original_gap_cosets, 0);
        assert_eq!(
            artifact
                .exact_invariants
                .distinct_coset_octets_across_all_subgroups,
            151_200
        );
        assert_eq!(
            artifact
                .exact_invariants
                .block_compatible_octets_across_all_subgroups,
            27_120
        );
        assert_eq!(
            artifact
                .exact_invariants
                .unsigned_noncoincident_distinct_sector_candidate_octets_across_all_subgroups,
            23_520
        );
        assert!(
            !artifact
                .ordered_pair_correspondence
                .unsigned_support_relation_is_bijection
        );
    }
}
