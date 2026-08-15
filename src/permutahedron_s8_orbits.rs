//! Exact normalizer-conjugacy classification of the fixed `R8` coset atlas.
//!
//! This module supplies a finite combinatorial classification, not a
//! supermultiplet classification.  The normalizer `N_S8(R8)` acts on the 5,040
//! fixed-`R8` cosets by conjugation.  Its twenty orbits cover every coset.
//! The paired-`S4` compatibility test is rederived here from block systems
//! alone, independently of the pair labels in `permutahedron_s8_separation`.

use crate::permutahedron::{
    coset_partition, factorial, permutations, rana_r8, CosetPartitionReport, CosetSide, Permutation,
};
use crate::permutahedron_garden::complete_garden_scan;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-normalizer-orbits-v1";

#[derive(Debug, Clone, Serialize)]
pub struct OrbitRecord {
    pub id: usize,
    pub representative_coset_id: usize,
    pub representative_rank: u32,
    pub representative: Vec<u8>,
    pub size: usize,
    pub coset_ids: Vec<u32>,
    pub compatible_partition_count: usize,
    pub paired_s4_compatible: bool,
    pub left_right_coincident: bool,
    pub noncoincident_block_compatible: bool,
    pub subgroup_intersection_order: usize,
    pub member_cycle_types: BTreeMap<String, usize>,
    pub induced_gl3_order: Option<usize>,
    pub induced_gl3_trace: Option<u8>,
    pub induced_gl3_characteristic_polynomial: Option<String>,
    pub exact_signature: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndependentCompatibilityAudit {
    pub unordered_four_plus_four_partitions: usize,
    pub r8_invariant_partitions: usize,
    pub compatibility_histogram: BTreeMap<usize, usize>,
    pub compatible_cosets: usize,
    pub incompatible_cosets: usize,
    pub compatible_incidences: usize,
    pub reproduces_existing_probe_counts: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignedUniformityAudit {
    pub garden_cosets_scanned: usize,
    pub garden_signable_cosets: usize,
    pub affine_rank_histogram: BTreeMap<usize, usize>,
    pub affine_nullity_histogram: BTreeMap<usize, usize>,
    pub canonical_hymn_trace_histogram: BTreeMap<i16, usize>,
    pub tested_signed_diagnostics_separate_cosets: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrbitValidation {
    pub normalizer_order: usize,
    pub normalizer_quotient_order: usize,
    pub conjugacy_orbits: usize,
    pub orbit_size_histogram: BTreeMap<usize, usize>,
    pub cosets_covered_once: usize,
    pub compatible_orbits: usize,
    pub compatible_cosets: usize,
    pub incompatible_orbits: usize,
    pub incompatible_cosets: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_orbits: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_cosets: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_complement_orbits: usize,
    pub unsigned_noncoincident_distinct_sector_candidate_complement_cosets: usize,
    pub abnormal_orbits: usize,
    pub abnormal_cosets: usize,
    pub intersection_order_histogram_by_orbit: BTreeMap<usize, usize>,
    pub intersection_order_histogram_by_coset: BTreeMap<usize, usize>,
    pub invariants_constant_on_every_orbit: bool,
    pub exact_signature_classes: usize,
    pub order_seven_gl3_classes: usize,
    pub order_seven_trace_values: Vec<u8>,
    pub order_seven_characteristic_polynomials: Vec<String>,
    pub twenty_not_thirty: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalizerOrbitArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub group_action: &'static str,
    pub independent_compatibility_audit: IndependentCompatibilityAudit,
    pub signed_uniformity_audit: SignedUniformityAudit,
    pub orbit_by_coset: Vec<u8>,
    pub orbits: Vec<OrbitRecord>,
    pub validation: OrbitValidation,
    pub findings: Vec<&'static str>,
    pub relation_to_published_thirty: &'static str,
    pub boundary: &'static str,
}

fn conjugate(g: Permutation, h: Permutation) -> Permutation {
    g.compose(h)
        .expect("common S8 order")
        .compose(g.inverse())
        .expect("common S8 order")
}

fn normalizer(subgroup: &[Permutation]) -> Vec<Permutation> {
    let ranks: HashSet<usize> = subgroup.iter().map(|element| element.rank()).collect();
    permutations(8)
        .expect("S8 permutations")
        .filter(|&g| {
            subgroup
                .iter()
                .all(|&h| ranks.contains(&conjugate(g, h).rank()))
        })
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

fn invariant_block_masks(subgroup: &[Permutation]) -> Vec<u8> {
    (0u16..=u16::from(u8::MAX))
        .map(|mask| mask as u8)
        .filter(|mask| mask.count_ones() == 4 && mask & 1 != 0)
        .filter(|&mask| {
            subgroup
                .iter()
                .all(|&element| block_compatible(element, mask))
        })
        .collect()
}

fn independent_compatibility(
    partition: &CosetPartitionReport,
    subgroup: &[Permutation],
) -> (IndependentCompatibilityAudit, Vec<usize>) {
    let invariant = invariant_block_masks(subgroup);
    let counts: Vec<usize> = partition
        .slices
        .iter()
        .map(|slice| {
            let representative =
                Permutation::unrank(8, slice[0] as usize).expect("S8 representative");
            invariant
                .iter()
                .filter(|&&mask| block_compatible(representative, mask))
                .count()
        })
        .collect();
    let compatibility_histogram = histogram(counts.iter().copied());
    let compatible_cosets = counts.iter().filter(|&&count| count != 0).count();
    let compatible_incidences = counts.iter().sum();
    let expected = BTreeMap::from([(0, 4_136), (1, 854), (3, 49), (7, 1)]);
    (
        IndependentCompatibilityAudit {
            unordered_four_plus_four_partitions: 35,
            r8_invariant_partitions: invariant.len(),
            compatibility_histogram: compatibility_histogram.clone(),
            compatible_cosets,
            incompatible_cosets: partition.slice_count - compatible_cosets,
            compatible_incidences,
            reproduces_existing_probe_counts: invariant.len() == 7
                && compatibility_histogram == expected
                && compatible_cosets == 904
                && compatible_incidences == 1_008,
        },
        counts,
    )
}

fn rank_assignment(partition: &CosetPartitionReport) -> Vec<usize> {
    let mut assignment = vec![usize::MAX; factorial(8).expect("S8 order")];
    for (coset_id, slice) in partition.slices.iter().enumerate() {
        for &rank in slice {
            assert_eq!(assignment[rank as usize], usize::MAX);
            assignment[rank as usize] = coset_id;
        }
    }
    assert!(assignment.iter().all(|&id| id != usize::MAX));
    assignment
}

/// Normalizer-conjugacy orbit ID for each right R8 coset in canonical slice
/// order. This structural projection omits the more expensive signed audits
/// performed by [`build`].
pub(crate) fn normalizer_orbit_assignment() -> Vec<u8> {
    let subgroup = rana_r8();
    let partition = coset_partition(&subgroup, CosetSide::Right).expect("R8 cosets");
    let normalizer = normalizer(&subgroup);
    let assignment = rank_assignment(&partition);
    let mut covered = vec![false; partition.slice_count];
    let mut orbit_by_coset = vec![u8::MAX; partition.slice_count];
    let mut orbit_id = 0u8;
    for seed_id in 0..partition.slice_count {
        if covered[seed_id] {
            continue;
        }
        let seed = Permutation::unrank(8, partition.slices[seed_id][0] as usize)
            .expect("S8 representative");
        let coset_ids: BTreeSet<usize> = normalizer
            .iter()
            .map(|&n| assignment[conjugate(n, seed).rank()])
            .collect();
        for coset_id in coset_ids {
            assert!(!covered[coset_id], "normalizer orbits overlap");
            covered[coset_id] = true;
            orbit_by_coset[coset_id] = orbit_id;
        }
        orbit_id += 1;
    }
    assert_eq!(orbit_id, 20);
    assert!(covered.into_iter().all(|value| value));
    orbit_by_coset
}

fn left_right_coincident(slice: &[u32], subgroup: &[Permutation]) -> bool {
    let representative = Permutation::unrank(8, slice[0] as usize).expect("S8 rank");
    let mut other: Vec<u32> = subgroup
        .iter()
        .map(|&h| representative.compose(h).expect("S8 product").rank() as u32)
        .collect();
    other.sort_unstable();
    other == slice
}

fn subgroup_intersection_order(g: Permutation, subgroup: &[Permutation]) -> usize {
    let subgroup_ranks: HashSet<usize> = subgroup.iter().map(|element| element.rank()).collect();
    subgroup
        .iter()
        .filter(|&&element| subgroup_ranks.contains(&conjugate(g, element).rank()))
        .count()
}

fn cycle_type(permutation: Permutation) -> Vec<u8> {
    let mut seen = [false; 8];
    let mut lengths = Vec::new();
    for start in 0..8 {
        if seen[start] {
            continue;
        }
        let mut length = 0u8;
        let mut current = start;
        while !seen[current] {
            seen[current] = true;
            length += 1;
            current = usize::from(permutation.as_slice()[current] - 1);
        }
        lengths.push(length);
    }
    lengths.sort_unstable_by(|left, right| right.cmp(left));
    lengths
}

fn cycle_label(permutation: Permutation) -> String {
    cycle_type(permutation)
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join("+")
}

fn cycle_type_histogram(slice: &[u32]) -> BTreeMap<String, usize> {
    let labels = slice
        .iter()
        .map(|&rank| cycle_label(Permutation::unrank(8, rank as usize).expect("S8 rank")));
    let mut result = BTreeMap::new();
    for label in labels {
        *result.entry(label).or_default() += 1;
    }
    result
}

fn subgroup_basis(subgroup: &[Permutation]) -> [Permutation; 3] {
    let identity = Permutation::identity(8).expect("S8 identity");
    let mut candidates = subgroup.to_vec();
    candidates.sort_unstable_by_key(|element| element.rank());
    let mut basis = Vec::new();
    let mut span = BTreeSet::from([identity.rank()]);
    for candidate in candidates {
        if span.contains(&candidate.rank()) {
            continue;
        }
        basis.push(candidate);
        let old: Vec<usize> = span.iter().copied().collect();
        for rank in old {
            let element = Permutation::unrank(8, rank).expect("S8 rank");
            span.insert(
                element
                    .compose(candidate)
                    .expect("abelian subgroup product")
                    .rank(),
            );
        }
        if basis.len() == 3 {
            break;
        }
    }
    assert_eq!(span.len(), 8);
    basis
        .try_into()
        .expect("rank-three elementary abelian basis")
}

fn subgroup_vectors(basis: &[Permutation; 3]) -> HashMap<usize, u8> {
    let identity = Permutation::identity(8).expect("S8 identity");
    let mut result = HashMap::new();
    for vector in 0u8..8 {
        let mut element = identity;
        for (bit, &generator) in basis.iter().enumerate() {
            if vector & (1 << bit) != 0 {
                element = element
                    .compose(generator)
                    .expect("elementary abelian product");
            }
        }
        assert!(result.insert(element.rank(), vector).is_none());
    }
    result
}

fn apply_linear(columns: [u8; 3], vector: u8) -> u8 {
    let mut result = 0u8;
    for (column, image) in columns.iter().enumerate() {
        if vector & (1 << column) != 0 {
            result ^= image;
        }
    }
    result
}

fn linear_order(columns: [u8; 3]) -> usize {
    for order in 1..=7 {
        if (0u8..8).all(|vector| {
            let mut image = vector;
            for _ in 0..order {
                image = apply_linear(columns, image);
            }
            image == vector
        }) {
            return order;
        }
    }
    panic!("GL(3,2) element order exceeds seven")
}

fn gl3_data(
    g: Permutation,
    basis: &[Permutation; 3],
    vectors: &HashMap<usize, u8>,
) -> (usize, u8, String) {
    let columns = std::array::from_fn(|column| {
        let image = conjugate(g, basis[column]);
        vectors[&image.rank()]
    });
    let entry = |row: usize, column: usize| (columns[column] >> row) & 1;
    let trace = entry(0, 0) ^ entry(1, 1) ^ entry(2, 2);
    let linear_coefficient = (entry(0, 0) & entry(1, 1))
        ^ (entry(0, 1) & entry(1, 0))
        ^ (entry(0, 0) & entry(2, 2))
        ^ (entry(0, 2) & entry(2, 0))
        ^ (entry(1, 1) & entry(2, 2))
        ^ (entry(1, 2) & entry(2, 1));
    let mut terms = vec!["x^3"];
    if trace == 1 {
        terms.push("x^2");
    }
    if linear_coefficient == 1 {
        terms.push("x");
    }
    terms.push("1");
    (linear_order(columns), trace, terms.join("+"))
}

fn orbit_signature(
    intersection: usize,
    cycle_types: &BTreeMap<String, usize>,
    gl_order: Option<usize>,
    gl_trace: Option<u8>,
    gl_polynomial: Option<&str>,
) -> String {
    let cycles = cycle_types
        .iter()
        .map(|(cycle, count)| format!("{cycle}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "intersection={intersection}|cycles={cycles}|gl_order={}|gl_trace={}|gl_char={}",
        gl_order.map_or_else(|| "none".into(), |value| value.to_string()),
        gl_trace.map_or_else(|| "none".into(), |value| value.to_string()),
        gl_polynomial.unwrap_or("none")
    )
}

fn canonical_hymn_trace(slice: &[u32], mask: u64) -> i16 {
    let mut product_map: [usize; 16] = std::array::from_fn(|index| index);
    let mut product_sign = [1i8; 16];
    for (color, &rank) in slice.iter().enumerate() {
        let permutation = Permutation::unrank(8, rank as usize).expect("S8 rank");
        let mut inverse = [0usize; 8];
        for (row, &column) in permutation.as_slice().iter().enumerate() {
            inverse[usize::from(column - 1)] = row;
        }
        let mut gamma_map = [0usize; 16];
        let mut gamma_sign = [0i8; 16];
        for row in 0..8 {
            let column = usize::from(permutation.as_slice()[row] - 1);
            let sign = if mask & (1u64 << (color * 8 + row)) != 0 {
                -1
            } else {
                1
            };
            gamma_map[row] = 8 + column;
            gamma_sign[row] = sign;
            gamma_map[8 + column] = row;
            gamma_sign[8 + column] = sign;
        }
        let old_map = product_map;
        let old_sign = product_sign;
        for row in 0..16 {
            product_map[row] = old_map[gamma_map[row]];
            product_sign[row] = gamma_sign[row] * old_sign[gamma_map[row]];
        }
        debug_assert!(inverse.iter().all(|&row| row < 8));
    }
    (0..16)
        .filter(|&row| product_map[row] == row)
        .map(|row| i16::from(product_sign[row]))
        .sum()
}

fn signed_uniformity(partition: &CosetPartitionReport) -> SignedUniformityAudit {
    let scan = complete_garden_scan();
    let affine_rank_histogram = scan
        .rank_histogram
        .iter()
        .map(|entry| (entry.value, entry.octets))
        .collect();
    let affine_nullity_histogram = scan
        .nullity_histogram
        .iter()
        .map(|entry| (entry.value, entry.octets))
        .collect();
    let canonical_hymn_trace_histogram = histogram_i16(
        partition
            .slices
            .iter()
            .zip(&scan.cosets)
            .map(|(slice, record)| {
                let mask =
                    u64::from_str_radix(&record.canonical_sign_mask, 16).expect("hex sign mask");
                canonical_hymn_trace(slice, mask)
            }),
    );
    let tested_signed_diagnostics_separate_cosets = scan.rank_histogram.len() != 1
        || scan.nullity_histogram.len() != 1
        || canonical_hymn_trace_histogram.len() != 1;
    SignedUniformityAudit {
        garden_cosets_scanned: scan.cosets_scanned,
        garden_signable_cosets: scan.signable_cosets,
        affine_rank_histogram,
        affine_nullity_histogram,
        canonical_hymn_trace_histogram,
        tested_signed_diagnostics_separate_cosets,
    }
}

fn histogram(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_default() += 1;
    }
    result
}

fn histogram_i16(values: impl IntoIterator<Item = i16>) -> BTreeMap<i16, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_default() += 1;
    }
    result
}

pub fn build() -> NormalizerOrbitArtifact {
    let subgroup = rana_r8();
    let partition = coset_partition(&subgroup, CosetSide::Right).expect("R8 cosets");
    let normalizer = normalizer(&subgroup);
    let assignment = rank_assignment(&partition);
    let (independent_compatibility_audit, compatibility_counts) =
        independent_compatibility(&partition, &subgroup);
    let signed_uniformity_audit = signed_uniformity(&partition);
    let basis = subgroup_basis(&subgroup);
    let vectors = subgroup_vectors(&basis);

    let mut covered = vec![false; partition.slice_count];
    let mut orbit_by_coset = vec![u8::MAX; partition.slice_count];
    let mut orbits = Vec::new();
    let mut invariant_constancy = true;

    for seed_id in 0..partition.slice_count {
        if covered[seed_id] {
            continue;
        }
        let seed = Permutation::unrank(8, partition.slices[seed_id][0] as usize)
            .expect("S8 representative");
        let coset_ids: BTreeSet<usize> = normalizer
            .iter()
            .map(|&n| assignment[conjugate(n, seed).rank()])
            .collect();
        let id = orbits.len();
        for &coset_id in &coset_ids {
            assert!(!covered[coset_id], "normalizer conjugacy orbits overlap");
            covered[coset_id] = true;
            orbit_by_coset[coset_id] = id as u8;
        }

        let representative_coset_id = *coset_ids.first().expect("nonempty orbit");
        let representative_rank = partition.slices[representative_coset_id][0];
        let representative =
            Permutation::unrank(8, representative_rank as usize).expect("S8 representative");
        let compatible_values: BTreeSet<usize> = coset_ids
            .iter()
            .map(|&coset_id| compatibility_counts[coset_id])
            .collect();
        let coincident_values: BTreeSet<bool> = coset_ids
            .iter()
            .map(|&coset_id| left_right_coincident(&partition.slices[coset_id], &subgroup))
            .collect();
        let intersection_values: BTreeSet<usize> = coset_ids
            .iter()
            .map(|&coset_id| {
                let g = Permutation::unrank(8, partition.slices[coset_id][0] as usize)
                    .expect("S8 representative");
                subgroup_intersection_order(g, &subgroup)
            })
            .collect();
        let cycle_values: BTreeSet<Vec<(String, usize)>> = coset_ids
            .iter()
            .map(|&coset_id| {
                cycle_type_histogram(&partition.slices[coset_id])
                    .into_iter()
                    .collect()
            })
            .collect();
        invariant_constancy &= compatible_values.len() == 1
            && coincident_values.len() == 1
            && intersection_values.len() == 1
            && cycle_values.len() == 1;

        let compatible_partition_count = *compatible_values.first().expect("one value");
        let left_right_coincident = *coincident_values.first().expect("one value");
        let subgroup_intersection_order = *intersection_values.first().expect("one value");
        let member_cycle_types: BTreeMap<String, usize> = cycle_values
            .first()
            .expect("one value")
            .iter()
            .cloned()
            .collect();
        let (gl_order, gl_trace, gl_polynomial) = if left_right_coincident {
            let (order, trace, polynomial) = gl3_data(representative, &basis, &vectors);
            (Some(order), Some(trace), Some(polynomial))
        } else {
            (None, None, None)
        };
        let exact_signature = orbit_signature(
            subgroup_intersection_order,
            &member_cycle_types,
            gl_order,
            gl_trace,
            gl_polynomial.as_deref(),
        );

        orbits.push(OrbitRecord {
            id,
            representative_coset_id,
            representative_rank,
            representative: representative.one_line(),
            size: coset_ids.len(),
            coset_ids: coset_ids.iter().map(|&value| value as u32).collect(),
            compatible_partition_count,
            paired_s4_compatible: compatible_partition_count != 0,
            left_right_coincident,
            noncoincident_block_compatible: compatible_partition_count != 0
                && !left_right_coincident,
            subgroup_intersection_order,
            member_cycle_types,
            induced_gl3_order: gl_order,
            induced_gl3_trace: gl_trace,
            induced_gl3_characteristic_polynomial: gl_polynomial,
            exact_signature,
        });
    }
    assert!(covered.iter().all(|&value| value));

    let orbit_size_histogram = histogram(orbits.iter().map(|orbit| orbit.size));
    let compatible_orbits = orbits
        .iter()
        .filter(|orbit| orbit.paired_s4_compatible)
        .count();
    let compatible_cosets = orbits
        .iter()
        .filter(|orbit| orbit.paired_s4_compatible)
        .map(|orbit| orbit.size)
        .sum();
    let abnormal_orbits = orbits
        .iter()
        .filter(|orbit| orbit.left_right_coincident)
        .count();
    let abnormal_cosets = orbits
        .iter()
        .filter(|orbit| orbit.left_right_coincident)
        .map(|orbit| orbit.size)
        .sum();
    let noncoincident_compatible_orbits = orbits
        .iter()
        .filter(|orbit| orbit.noncoincident_block_compatible)
        .count();
    let noncoincident_compatible_cosets = orbits
        .iter()
        .filter(|orbit| orbit.noncoincident_block_compatible)
        .map(|orbit| orbit.size)
        .sum();
    let intersection_order_histogram_by_orbit =
        histogram(orbits.iter().map(|orbit| orbit.subgroup_intersection_order));
    let intersection_order_histogram_by_coset = {
        let mut result = BTreeMap::new();
        for orbit in &orbits {
            *result.entry(orbit.subgroup_intersection_order).or_default() += orbit.size;
        }
        result
    };
    let signature_count = orbits
        .iter()
        .map(|orbit| orbit.exact_signature.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let order_seven: Vec<&OrbitRecord> = orbits
        .iter()
        .filter(|orbit| orbit.induced_gl3_order == Some(7))
        .collect();
    let order_seven_trace_values: Vec<u8> = order_seven
        .iter()
        .filter_map(|orbit| orbit.induced_gl3_trace)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let order_seven_characteristic_polynomials: Vec<String> = order_seven
        .iter()
        .filter_map(|orbit| orbit.induced_gl3_characteristic_polynomial.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let order_seven_class_count = order_seven.len();
    let orbit_count = orbits.len();

    let expected_orbit_sizes = BTreeMap::from([
        (1, 1),
        (21, 1),
        (24, 2),
        (28, 1),
        (42, 1),
        (56, 3),
        (84, 1),
        (112, 1),
        (168, 1),
        (224, 1),
        (336, 3),
        (448, 1),
        (672, 2),
        (1_344, 1),
    ]);
    let passed = normalizer.len() == 1_344
        && orbits.len() == 20
        && orbit_size_histogram == expected_orbit_sizes
        && orbit_by_coset.iter().all(|&id| id != u8::MAX)
        && compatible_orbits == 10
        && compatible_cosets == 904
        && orbits.len() - compatible_orbits == 10
        && partition.slice_count - compatible_cosets == 4_136
        && abnormal_orbits == 6
        && abnormal_cosets == 168
        && noncoincident_compatible_orbits == 6
        && noncoincident_compatible_cosets == 784
        && intersection_order_histogram_by_orbit == BTreeMap::from([(1, 9), (2, 5), (8, 6)])
        && intersection_order_histogram_by_coset
            == BTreeMap::from([(1, 3_696), (2, 1_176), (8, 168)])
        && invariant_constancy
        && signature_count == 20
        && order_seven_class_count == 2
        && order_seven_trace_values == vec![0, 1]
        && order_seven_characteristic_polynomials
            == vec!["x^3+x+1".to_string(), "x^3+x^2+1".to_string()]
        && independent_compatibility_audit.reproduces_existing_probe_counts
        && signed_uniformity_audit.garden_cosets_scanned == 5_040
        && signed_uniformity_audit.garden_signable_cosets == 5_040
        && signed_uniformity_audit.affine_rank_histogram == BTreeMap::from([(45, 5_040)])
        && signed_uniformity_audit.affine_nullity_histogram == BTreeMap::from([(19, 5_040)])
        && signed_uniformity_audit.canonical_hymn_trace_histogram == BTreeMap::from([(0, 5_040)])
        && !signed_uniformity_audit.tested_signed_diagnostics_separate_cosets;

    NormalizerOrbitArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Exact normalizer-conjugacy classes of the fixed R8 coset atlas",
        group_action: "N_S8(R8) acts on fixed-R8 cosets H*g by H*g -> H*(n*g*n^-1)",
        independent_compatibility_audit,
        signed_uniformity_audit,
        orbit_by_coset,
        orbits,
        validation: OrbitValidation {
            normalizer_order: normalizer.len(),
            normalizer_quotient_order: normalizer.len() / subgroup.len(),
            conjugacy_orbits: orbit_count,
            orbit_size_histogram,
            cosets_covered_once: covered.iter().filter(|&&value| value).count(),
            compatible_orbits,
            compatible_cosets,
            incompatible_orbits: 20 - compatible_orbits,
            incompatible_cosets: partition.slice_count - compatible_cosets,
            unsigned_noncoincident_distinct_sector_candidate_orbits:
                noncoincident_compatible_orbits,
            unsigned_noncoincident_distinct_sector_candidate_cosets:
                noncoincident_compatible_cosets,
            unsigned_noncoincident_distinct_sector_candidate_complement_orbits: orbit_count
                - noncoincident_compatible_orbits,
            unsigned_noncoincident_distinct_sector_candidate_complement_cosets: partition
                .slice_count
                - noncoincident_compatible_cosets,
            abnormal_orbits,
            abnormal_cosets,
            intersection_order_histogram_by_orbit,
            intersection_order_histogram_by_coset,
            invariants_constant_on_every_orbit: invariant_constancy,
            exact_signature_classes: signature_count,
            order_seven_gl3_classes: order_seven_class_count,
            order_seven_trace_values,
            order_seven_characteristic_polynomials,
            twenty_not_thirty: orbit_count == 20 && orbit_count != 30,
            passed,
        },
        findings: vec![
            "The block-system calculation independently reproduces 904 compatible and 4,136 incompatible fixed-R8 cosets.",
            "Normalizer conjugation partitions all 5,040 fixed-R8 cosets into twenty exact combinatorial classes.",
            "The 904 block-compatible cosets are ten complete orbits; the remaining 4,136 cosets are ten complete orbits.",
            "Restricting to unsigned noncoincident distinct-sector candidates leaves 784 cosets in six orbits and a complement of 4,256 cosets in fourteen orbits.",
            "Cycle-type multiset and subgroup-intersection order separate nineteen orbits. The induced GL(3,2) characteristic polynomial separates the two order-seven classes.",
            "Garden sign feasibility, affine rank, affine nullity, and the canonical Garden-signing HYMN trace are uniform across all 5,040 cosets.",
        ],
        relation_to_published_thirty: "The twenty normalizer-conjugacy orbits are not the thirty ordered distinct S4-sector pairs proposed in arXiv:2304.09830, and they are not the thirty conjugate R8 subgroups. The counts and underlying sets differ, so this calculation does not establish either proposed thirty-to-thirty correspondence.",
        boundary: "The orbit identifier is complete for the stated normalizer-conjugacy action on unsigned fixed-R8 cosets. It is not a supermultiplet type, does not determine a published Boolean-factor assignment, and does not supply physical meaning for the ten classes containing the 4,136 block-incompatible cosets.",
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> OrbitValidation {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create validation directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create orbit artifact")),
        &artifact,
    )
    .expect("write orbit artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create orbit validation")),
        &artifact.validation,
    )
    .expect("write orbit validation");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_block_test_reproduces_probe_counts() {
        let subgroup = rana_r8();
        let partition = coset_partition(&subgroup, CosetSide::Right).unwrap();
        let (audit, _) = independent_compatibility(&partition, &subgroup);
        assert!(audit.reproduces_existing_probe_counts);
        assert_eq!(audit.compatible_cosets, 904);
        assert_eq!(audit.incompatible_cosets, 4_136);
    }

    #[test]
    fn normalizer_orbits_classify_every_fixed_r8_coset() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.conjugacy_orbits, 20);
        assert_eq!(artifact.validation.cosets_covered_once, 5_040);
        assert_eq!(artifact.validation.compatible_orbits, 10);
        assert_eq!(artifact.validation.incompatible_orbits, 10);
        assert_eq!(artifact.validation.incompatible_cosets, 4_136);
        assert_eq!(
            artifact
                .validation
                .unsigned_noncoincident_distinct_sector_candidate_orbits,
            6
        );
        assert_eq!(
            artifact
                .validation
                .unsigned_noncoincident_distinct_sector_candidate_cosets,
            784
        );
        assert_eq!(
            artifact
                .validation
                .unsigned_noncoincident_distinct_sector_candidate_complement_cosets,
            4_256
        );
        assert_eq!(artifact.validation.exact_signature_classes, 20);
    }

    #[test]
    fn the_two_order_seven_classes_have_different_gf2_traces() {
        let artifact = build();
        assert_eq!(artifact.validation.order_seven_gl3_classes, 2);
        assert_eq!(artifact.validation.order_seven_trace_values, vec![0, 1]);
        assert_eq!(
            artifact.validation.order_seven_characteristic_polynomials,
            vec!["x^3+x+1", "x^3+x^2+1"]
        );
    }

    #[test]
    fn orbit_count_does_not_reproduce_the_published_thirty() {
        let artifact = build();
        assert!(artifact.validation.twenty_not_thirty);
        assert_eq!(artifact.validation.conjugacy_orbits, 20);
    }
}
