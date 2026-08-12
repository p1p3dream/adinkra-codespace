//! Exact first-Gadget orthonormal-frame census for the six S4 quartets.
//!
//! Each fixed ordered permutation quartet has 256 Garden-closing Boolean
//! assignments. This module enumerates the complete 6 x 256 library, computes
//! exact Gadget numerators, compresses candidates with identical cross-sector
//! compatibility profiles, and counts six-frames with one signing from each
//! quartet and zero pairwise Gadget.

use crate::chromochar::chi0_n4;
use crate::holoraumy::HoloraumyData;
use crate::lr_matrix::AdinkraRep;
use crate::permutahedron_s4_supersymmetry::{S4_BOOLEAN_FACTOR_QUARTETS, build_rep};
use crate::permutahedron_s8_signed_recursion::S4_RECURSION_BOOLEAN_FACTORS;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s4-gadget-frames-v1";
const SECTORS: usize = 6;
const SIGNINGS: usize = 256;
const LABELS: [&str; 6] = ["CM", "TM", "VM", "VM1", "VM2", "VM3"];

#[derive(Clone)]
struct Signing {
    factors: [u8; 4],
    rep: AdinkraRep,
    holoraumy: HoloraumyData,
    chi0: i8,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceRecord {
    pub source: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SectorRecord {
    pub label: &'static str,
    pub garden_signings: usize,
    pub vertex_switching_classes: usize,
    pub chi0_positive: usize,
    pub chi0_negative: usize,
    pub compatibility_profile_types: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrameRecord {
    pub source_description: &'static str,
    pub appendix_b_indices_one_based_when_present: [Option<usize>; 6],
    pub boolean_factors: [[u8; 4]; 6],
    pub gadget_numerators_over_24: [[i16; 6]; 6],
    pub is_orthonormal: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RepairRecord {
    pub total_boolean_bit_flips: u32,
    pub minimum_repair_frames: u128,
    pub example_boolean_factors: [[u8; 4]; 6],
    pub changed_sector_color_entries_one_based: Vec<[usize; 2]>,
    pub example_gadget_numerators_over_24: [[i16; 6]; 6],
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub raw_boolean_assignments_checked: usize,
    pub garden_closing_signings: usize,
    pub signings_per_sector: [usize; 6],
    pub exact_pairwise_gadgets_checked: usize,
    pub distinct_gadget_numerators_over_24: Vec<i16>,
    pub literal_weighted_table5_frame_is_orthonormal: bool,
    pub appendix_b_reference_frame_is_orthonormal: bool,
    pub raw_orthonormal_frames: u128,
    pub total_fixed_order_frames: u128,
    pub orthonormal_frame_fraction_numerator: u128,
    pub orthonormal_frame_fraction_denominator: u128,
    pub common_vertex_switching_orbits: u128,
    pub common_vertex_switching_action_is_free: bool,
    pub conservative_remaining_relabeling_group_bound: u128,
    pub conservative_minimum_classes_after_standard_common_relabelings: u128,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GadgetFrameArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub sectors: Vec<SectorRecord>,
    pub literal_weighted_table5_frame: FrameRecord,
    pub appendix_b_orthonormal_reference_frame: FrameRecord,
    pub nearest_orthonormal_repair_of_literal_table5: RepairRecord,
    pub validation: ValidationRecord,
    pub interpretation: &'static str,
    pub boundary: &'static str,
}

fn all_signings(sector: usize) -> Vec<Signing> {
    let mut output = Vec::with_capacity(SIGNINGS);
    for packed in 0u32..=u16::MAX.into() {
        let factors = std::array::from_fn(|color| ((packed >> (4 * color)) & 15) as u8);
        let rep = build_rep(sector, factors);
        if rep.verify_garden_algebra() {
            let chi = chi0_n4(&rep);
            output.push(Signing {
                factors,
                holoraumy: HoloraumyData::from_rep(&rep),
                rep,
                chi0: if chi > 0.0 { 1 } else { -1 },
            });
        }
    }
    output.sort_by_key(|signing| signing.factors);
    output
}

/// The first Gadget equals this integer divided by 24 at N=4, d=4.
fn gadget_numerator(left: &HoloraumyData, right: &HoloraumyData) -> i16 {
    let trace_sum: i64 = left
        .vtilde
        .iter()
        .zip(&right.vtilde)
        .map(|(left, right)| left.trace_product(right))
        .sum();
    i16::try_from(-trace_sum).expect("N4 Gadget numerator")
}

fn switch_factors(rep: &AdinkraRep, factors: [u8; 4], boson: u8, fermion: u8) -> [u8; 4] {
    std::array::from_fn(|color| {
        (0..4).fold(0u8, |factor, row| {
            let target = usize::from(rep.l_matrices[color].perm[row]);
            let flip = ((boson >> row) ^ (fermion >> target)) & 1;
            factor | ((((factors[color] >> row) & 1) ^ flip) << row)
        })
    })
}

fn switching_classes(signings: &[Signing]) -> usize {
    let index: BTreeMap<[u8; 4], usize> = signings
        .iter()
        .enumerate()
        .map(|(index, signing)| (signing.factors, index))
        .collect();
    let mut seen = vec![false; signings.len()];
    let mut classes = 0usize;
    for seed in 0..signings.len() {
        if seen[seed] {
            continue;
        }
        classes += 1;
        // Fix boson node zero to +1. This removes the simultaneous global
        // reversal of all boson and fermion switches.
        for boson_tail in 0u8..8 {
            let boson = boson_tail << 1;
            for fermion in 0u8..16 {
                let transformed =
                    switch_factors(&signings[seed].rep, signings[seed].factors, boson, fermion);
                seen[*index
                    .get(&transformed)
                    .expect("switching preserves closure")] = true;
            }
        }
    }
    classes
}

#[derive(Clone)]
struct ProfileType {
    representative: usize,
    multiplicity: u128,
    members: Vec<usize>,
}

fn profile_types(library: &[Vec<Signing>], sector: usize) -> Vec<ProfileType> {
    let mut profiles = BTreeMap::<Vec<i16>, ProfileType>::new();
    for (candidate, signing) in library[sector].iter().enumerate() {
        let mut profile = Vec::with_capacity((SECTORS - 1) * SIGNINGS);
        for (other_sector, other_signings) in library.iter().enumerate() {
            if other_sector == sector {
                continue;
            }
            profile.extend(
                other_signings
                    .iter()
                    .map(|other| gadget_numerator(&signing.holoraumy, &other.holoraumy)),
            );
        }
        let entry = profiles.entry(profile).or_insert(ProfileType {
            representative: candidate,
            multiplicity: 0,
            members: Vec::new(),
        });
        entry.multiplicity += 1;
        entry.members.push(candidate);
    }
    profiles.into_values().collect()
}

#[derive(Default)]
struct RepairSearch {
    best_cost: u32,
    count: u128,
    example: Option<[[u8; 4]; 6]>,
}

struct RepairBranch {
    chosen: Vec<(usize, usize)>,
    factors: [[u8; 4]; 6],
    cost: u32,
    count: u128,
}

fn factor_distance(left: [u8; 4], right: [u8; 4]) -> u32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left ^ right).count_ones())
        .sum()
}

fn search_repairs(
    sector: usize,
    branch: &mut RepairBranch,
    types: &[Vec<ProfileType>],
    library: &[Vec<Signing>],
    target: &[[u8; 4]; 6],
    result: &mut RepairSearch,
) {
    if sector == SECTORS {
        if result.example.is_none() || branch.cost < result.best_cost {
            result.best_cost = branch.cost;
            result.count = branch.count;
            result.example = Some(branch.factors);
        } else if branch.cost == result.best_cost {
            result.count += branch.count;
            if branch.factors < result.example.expect("repair example") {
                result.example = Some(branch.factors);
            }
        }
        return;
    }
    if result.example.is_some() && branch.cost > result.best_cost {
        return;
    }
    for (type_index, profile_type) in types[sector].iter().enumerate() {
        let representative = &library[sector][profile_type.representative];
        if !branch.chosen.iter().all(|&(other_sector, other_type)| {
            let other = &library[other_sector][types[other_sector][other_type].representative];
            gadget_numerator(&representative.holoraumy, &other.holoraumy) == 0
        }) {
            continue;
        }
        let minimum = profile_type
            .members
            .iter()
            .map(|&member| factor_distance(library[sector][member].factors, target[sector]))
            .min()
            .expect("profile member");
        let mut minimum_members: Vec<_> = profile_type
            .members
            .iter()
            .copied()
            .filter(|&member| {
                factor_distance(library[sector][member].factors, target[sector]) == minimum
            })
            .collect();
        minimum_members.sort_by_key(|&member| library[sector][member].factors);
        branch.chosen.push((sector, type_index));
        branch.factors[sector] = library[sector][minimum_members[0]].factors;
        branch.cost += minimum;
        branch.count *= minimum_members.len() as u128;
        search_repairs(sector + 1, branch, types, library, target, result);
        branch.count /= minimum_members.len() as u128;
        branch.cost -= minimum;
        branch.chosen.pop();
    }
}

fn nearest_repair(
    library: &[Vec<Signing>],
    types: &[Vec<ProfileType>],
    target: [[u8; 4]; 6],
) -> RepairRecord {
    let mut search = RepairSearch::default();
    let mut branch = RepairBranch {
        chosen: Vec::new(),
        factors: [[0; 4]; 6],
        cost: 0,
        count: 1,
    };
    search_repairs(0, &mut branch, types, library, &target, &mut search);
    let example = search.example.expect("at least one orthonormal repair");
    let frame = frame_record(library, example, "nearest orthonormal repair");
    let changed = (0..6)
        .flat_map(|sector| {
            (0..4).filter_map(move |color| {
                (target[sector][color] != example[sector][color]).then_some([sector + 1, color + 1])
            })
        })
        .collect();
    RepairRecord {
        total_boolean_bit_flips: search.best_cost,
        minimum_repair_frames: search.count,
        example_boolean_factors: example,
        changed_sector_color_entries_one_based: changed,
        example_gadget_numerators_over_24: frame.gadget_numerators_over_24,
    }
}

fn count_type_frames(
    sector: usize,
    chosen: &mut Vec<(usize, usize)>,
    types: &[Vec<ProfileType>],
    library: &[Vec<Signing>],
    weight: u128,
) -> u128 {
    if sector == SECTORS {
        return weight;
    }
    let mut total = 0u128;
    for (type_index, profile_type) in types[sector].iter().enumerate() {
        let signing = &library[sector][profile_type.representative];
        if chosen.iter().all(|&(other_sector, other_type)| {
            let other = &library[other_sector][types[other_sector][other_type].representative];
            gadget_numerator(&signing.holoraumy, &other.holoraumy) == 0
        }) {
            chosen.push((sector, type_index));
            total += count_type_frames(
                sector + 1,
                chosen,
                types,
                library,
                weight * profile_type.multiplicity,
            );
            chosen.pop();
        }
    }
    total
}

fn frame_record(
    library: &[Vec<Signing>],
    factors: [[u8; 4]; 6],
    source_description: &'static str,
) -> FrameRecord {
    let mut numerators = [[0i16; 6]; 6];
    for left in 0..6 {
        for right in 0..6 {
            let left_signing = library[left]
                .iter()
                .find(|signing| signing.factors == factors[left])
                .expect("published left signing");
            let right_signing = library[right]
                .iter()
                .find(|signing| signing.factors == factors[right])
                .expect("published right signing");
            numerators[left][right] =
                gadget_numerator(&left_signing.holoraumy, &right_signing.holoraumy);
        }
    }
    FrameRecord {
        source_description,
        appendix_b_indices_one_based_when_present: std::array::from_fn(|sector| {
            S4_BOOLEAN_FACTOR_QUARTETS[sector]
                .iter()
                .position(|candidate| *candidate == factors[sector])
                .map(|index| index + 1)
        }),
        boolean_factors: factors,
        is_orthonormal: (0..6).all(|row| {
            (0..6).all(|column| numerators[row][column] == if row == column { 24 } else { 0 })
        }),
        gadget_numerators_over_24: numerators,
    }
}

pub fn build() -> GadgetFrameArtifact {
    let library: Vec<Vec<Signing>> = (0..SECTORS).map(all_signings).collect();
    let types: Vec<Vec<ProfileType>> = (0..SECTORS)
        .map(|sector| profile_types(&library, sector))
        .collect();
    let switching: Vec<usize> = library
        .iter()
        .map(|signings| switching_classes(signings))
        .collect();
    let sectors: Vec<_> = (0..SECTORS)
        .map(|sector| SectorRecord {
            label: LABELS[sector],
            garden_signings: library[sector].len(),
            vertex_switching_classes: switching[sector],
            chi0_positive: library[sector]
                .iter()
                .filter(|signing| signing.chi0 > 0)
                .count(),
            chi0_negative: library[sector]
                .iter()
                .filter(|signing| signing.chi0 < 0)
                .count(),
            compatibility_profile_types: types[sector].len(),
        })
        .collect();
    let mut gadget_values = BTreeMap::<i16, usize>::new();
    let mut pairwise = 0usize;
    for left_sector in 0..SECTORS {
        for right_sector in (left_sector + 1)..SECTORS {
            for left in &library[left_sector] {
                for right in &library[right_sector] {
                    *gadget_values
                        .entry(gadget_numerator(&left.holoraumy, &right.holoraumy))
                        .or_default() += 1;
                    pairwise += 1;
                }
            }
        }
    }
    let raw_frames = count_type_frames(0, &mut Vec::new(), &types, &library, 1);
    let total_frames = 256u128.pow(6);
    let literal_weighted = frame_record(
        &library,
        S4_RECURSION_BOOLEAN_FACTORS,
        "literal weighted signed representatives in arXiv:2408.09342, Table 5, with 3421 substituted for the printed VM2 entry 3412",
    );
    let appendix_factors = std::array::from_fn(|sector| S4_BOOLEAN_FACTOR_QUARTETS[sector][0]);
    let orthonormal_reference = frame_record(
        &library,
        appendix_factors,
        "first Appendix-B fiducial signing for each quartet in arXiv:1701.00304",
    );
    let nearest_repair = nearest_repair(&library, &types, S4_RECURSION_BOOLEAN_FACTORS);
    let switching_action_free = library
        .iter()
        .all(|signings| switching_classes(signings) == 2);
    let switching_orbits = if switching_action_free {
        raw_frames / 128
    } else {
        0
    };
    // After vertex switching, a deliberately generous upper bound for the
    // remaining common relabelings is S4 on bosons, S4 on fermions, S4 on
    // colors, and 2^3 effective color signs. Some of these operations do not
    // preserve the fixed labeled six-sector problem, so this can only lower
    // the true number of orbits and is used as a non-uniqueness bound.
    let remaining_group_bound = 24u128.pow(3) * 8;
    let conservative_minimum = switching_orbits.div_ceil(remaining_group_bound);
    let signings_per_sector: [usize; 6] = std::array::from_fn(|sector| library[sector].len());
    let passed = signings_per_sector == [SIGNINGS; 6]
        && !literal_weighted.is_orthonormal
        && orthonormal_reference.is_orthonormal
        && nearest_repair.total_boolean_bit_flips > 0
        && pairwise == 15 * SIGNINGS * SIGNINGS
        && switching_action_free
        && raw_frames.is_multiple_of(128);

    GadgetFrameArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Exact first-Gadget orthonormal-frame census for the six S4 quartets",
        sources: vec![
            SourceRecord {
                source: "HowardTLK.v2.pdf",
                locator: "pp. 61, 64, 73, 77, 80-81",
                role: "six four-color quartets, permutahedron geometry, hopping operators, and the six-chain base case",
            },
            SourceRecord {
                source: "arXiv:1701.00304",
                locator: "Appendix B and the first Gadget",
                role: "published fiducial Boolean factors and Gadget definition",
            },
            SourceRecord {
                source: "arXiv:2408.09342",
                locator: "Eqs. (2.14)-(2.16) and the displayed 6 x 6 Gadget matrix",
                role: "weighted Table 5 representatives and the stated orthonormality to audit independently",
            },
        ],
        sectors,
        literal_weighted_table5_frame: literal_weighted,
        appendix_b_orthonormal_reference_frame: orthonormal_reference,
        nearest_orthonormal_repair_of_literal_table5: nearest_repair,
        validation: ValidationRecord {
            raw_boolean_assignments_checked: SECTORS * 65_536,
            garden_closing_signings: library.iter().map(Vec::len).sum(),
            signings_per_sector,
            exact_pairwise_gadgets_checked: pairwise,
            distinct_gadget_numerators_over_24: gadget_values.into_keys().collect(),
            literal_weighted_table5_frame_is_orthonormal: false,
            appendix_b_reference_frame_is_orthonormal: true,
            raw_orthonormal_frames: raw_frames,
            total_fixed_order_frames: total_frames,
            orthonormal_frame_fraction_numerator: 105,
            orthonormal_frame_fraction_denominator: 1_024,
            common_vertex_switching_orbits: switching_orbits,
            common_vertex_switching_action_is_free: switching_action_free,
            conservative_remaining_relabeling_group_bound: remaining_group_bound,
            conservative_minimum_classes_after_standard_common_relabelings: conservative_minimum,
            passed,
        },
        interpretation: "The census tests whether first-Gadget orthonormality selects a rare joint Boolean assignment. Counts are given before and after common vertex switching, while keeping the six quartet labels and four color labels fixed.",
        boundary: "This is the complete fixed-color, fixed-quartet-order Boolean library. It does not yet quotient common color permutations or supercharge signs, include independently reordered colors, or identify Gadget orthonormality with four-dimensional physical parentage.",
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ValidationRecord {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create Gadget-frame data")),
        &artifact,
    )
    .expect("write Gadget-frame data");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create Gadget-frame validation")),
        &artifact.validation,
    )
    .expect("write Gadget-frame validation");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_fixed_quartet_has_256_garden_signings_in_two_switching_classes() {
        for sector in 0..6 {
            let signings = all_signings(sector);
            assert_eq!(signings.len(), 256);
            assert_eq!(switching_classes(&signings), 2);
        }
    }

    #[test]
    fn complete_frame_census_passes() {
        let artifact = build();
        assert!(!artifact.literal_weighted_table5_frame.is_orthonormal);
        assert!(
            artifact
                .appendix_b_orthonormal_reference_frame
                .is_orthonormal
        );
        assert_eq!(artifact.validation.signings_per_sector, [256; 6]);
        assert!(artifact.validation.raw_orthonormal_frames > 0);
        assert_eq!(
            artifact.validation.raw_orthonormal_frames * 1_024,
            artifact.validation.total_fixed_order_frames * 105
        );
        assert_eq!(
            artifact
                .validation
                .conservative_minimum_classes_after_standard_common_relabelings,
            2_038_898
        );
        assert_eq!(
            artifact
                .nearest_orthonormal_repair_of_literal_table5
                .total_boolean_bit_flips,
            8
        );
        assert_eq!(
            artifact
                .nearest_orthonormal_repair_of_literal_table5
                .minimum_repair_frames,
            80
        );
        assert!(
            artifact
                .nearest_orthonormal_repair_of_literal_table5
                .example_gadget_numerators_over_24
                .iter()
                .enumerate()
                .all(|(row, values)| values
                    .iter()
                    .enumerate()
                    .all(|(column, &value)| value == if row == column { 24 } else { 0 }))
        );
        assert!(artifact.validation.passed);
    }

    #[test]
    fn literal_weighted_table5_frame_has_two_nonzero_cross_pairs() {
        let artifact = build();
        assert_eq!(
            artifact
                .literal_weighted_table5_frame
                .gadget_numerators_over_24,
            [
                [24, 0, 0, 0, 8, 0],
                [0, 24, -8, 0, 0, 0],
                [0, -8, 24, 0, 0, 0],
                [0, 0, 0, 24, 0, 0],
                [8, 0, 0, 0, 24, 0],
                [0, 0, 0, 0, 0, 24],
            ]
        );
    }
}
