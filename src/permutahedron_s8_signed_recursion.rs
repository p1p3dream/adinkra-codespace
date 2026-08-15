//! Exact signed recursion audit for the thirty ordered distinct S4 pairs.
//!
//! The unsigned block recursion is arXiv:2304.09830v2, Eqs. (2.17)-(2.19).
//! Section 2.2, especially Eqs. (2.20)-(2.25), supplies the Boolean rule:
//! concatenate the two four-bit words for colors 1-4, then obtain colors 5-8
//! by flipping one cyclic run of four adjacent bits. All eight possible runs
//! are tested here. Garden closure is the acceptance condition in the paper.

use crate::decompose::{antisymmetric_commutant_dim, commutant_dim};
use crate::holoraumy::{gadget, HoloraumyData};
use crate::lr_matrix::AdinkraRep;
use crate::permutahedron_fixtures::{S4_ORDERED_QUARTETS, S8_REPRESENTATION_OCTETS};
use crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-signed-recursion-v1";
const N: usize = 8;
const D: usize = 8;

/// Signed S4 representatives printed in arXiv:2304.09830v2 immediately before
/// Sec. 2.1, ordered CM, TM, VM, VM1, VM2, VM3. Bit zero controls row one.
pub const S4_RECURSION_BOOLEAN_FACTORS: [[u8; 4]; 6] = [
    [10, 12, 6, 0],
    [14, 4, 8, 2],
    [12, 10, 6, 0],
    [6, 3, 10, 0],
    [12, 9, 0, 10],
    [12, 5, 0, 6],
];

pub const S4_RECURSION_LABELS: [&str; 6] = ["CM", "TM", "VM", "VM1", "VM2", "VM3"];

/// Signed one-line words transcribed directly from the six source sets. A
/// negative entry is barred in arXiv:2304.09830v2.
pub const S4_RECURSION_SIGNED_WORDS: [[[i8; 4]; 4]; 6] = [
    [[1, -4, 2, -3], [2, 3, -1, -4], [3, -2, -4, 1], [4, 1, 3, 2]],
    [[1, -3, -4, -2], [2, 4, -3, 1], [3, 1, 2, -4], [4, -2, 1, 3]],
    [[1, 3, -2, -4], [2, -4, 1, -3], [3, -1, -4, 2], [4, 2, 3, 1]],
    [[1, -4, -3, 2], [-2, -3, 4, 1], [3, -2, 1, -4], [4, 1, 2, 3]],
    [[1, 2, -4, -3], [-2, 1, 3, -4], [3, 4, 2, 1], [4, -3, 1, -2]],
    [[1, 2, -3, -4], [-2, 1, -4, 3], [3, 4, 1, 2], [4, -3, -2, 1]],
];

#[derive(Debug, Clone, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub version: u8,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClosureRecord {
    pub sparse_garden_passed: bool,
    pub entries_checked: usize,
    pub residual_entries: usize,
    pub nonclosing_color_pairs: usize,
    pub maximum_absolute_residual: i16,
}

#[derive(Debug, Clone, Serialize)]
pub struct HymnRecord {
    pub diagonal: bool,
    pub diagonal_entries: Vec<i16>,
    pub off_diagonal_entries: usize,
    pub trace: i16,
    pub interpretation: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlipCandidateRecord {
    pub start_position_one_based: usize,
    pub flipped_positions_one_based: [usize; 4],
    pub flip_mask_decimal: u8,
    pub boolean_factors: [u8; 8],
    pub permutations: [[u8; 8]; 8],
    pub closure: ClosureRecord,
    pub hymn: HymnRecord,
    pub self_gadget: Option<f64>,
    pub commutant_dimension: Option<usize>,
    pub antisymmetric_commutant_dimension: Option<usize>,
    pub exact_published_system_match: Option<&'static str>,
    pub exact_matrix_checksum_fnv1a64: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderedPairRecord {
    pub first_index: usize,
    pub first_label: &'static str,
    pub second_index: usize,
    pub second_label: &'static str,
    pub first_boolean_factors: [u8; 4],
    pub second_boolean_factors: [u8; 4],
    pub first_input_closes: bool,
    pub second_input_closes: bool,
    pub candidates: Vec<FlipCandidateRecord>,
    pub closing_start_positions_one_based: Vec<usize>,
    pub first_closing_start_position_one_based: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClosingCandidateKey {
    pub ordered_pair: String,
    pub start_position_one_based: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub signed_s4_source_representatives_checked: usize,
    pub signed_s4_source_representatives_closing: usize,
    pub source_signed_words_match_fixtures: bool,
    pub vm2_uses_3421_not_3412: bool,
    pub ordered_distinct_pairs_checked: usize,
    pub cyclic_four_bit_masks_checked_per_pair: usize,
    pub signed_candidates_checked: usize,
    pub dense_garden_entries_checked: usize,
    pub closing_candidates: usize,
    pub nonclosing_candidates: usize,
    pub ordered_pairs_with_at_least_one_closing_mask: usize,
    pub ordered_pairs_without_a_closing_mask: usize,
    pub distinct_exact_closing_matrix_systems: usize,
    pub ct_source_anchor_matched: bool,
    pub all_closing_self_gadgets_equal_one: bool,
    pub all_closing_commutants_are_scalar: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignedRecursionArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub conventions: BTreeMap<&'static str, &'static str>,
    pub source_s4_labels: [&'static str; 6],
    pub source_s4_boolean_factors: [[u8; 4]; 6],
    pub ordered_pairs: Vec<OrderedPairRecord>,
    pub closing_candidates: Vec<ClosingCandidateKey>,
    pub closing_candidate_gadget_matrix: Vec<Vec<f64>>,
    pub validation: ValidationRecord,
    pub boundary: &'static str,
}

fn signs(factor: u8, width: usize) -> Vec<i8> {
    (0..width)
        .map(|row| if factor & (1 << row) == 0 { 1 } else { -1 })
        .collect()
}

fn s4_rep(index: usize) -> AdinkraRep {
    let permutations: Vec<Vec<usize>> = S4_ORDERED_QUARTETS[index]
        .iter()
        .map(|p| p.iter().map(|&entry| usize::from(entry - 1)).collect())
        .collect();
    let dashing: Vec<i8> = S4_RECURSION_BOOLEAN_FACTORS[index]
        .iter()
        .flat_map(|&factor| signs(factor, 4))
        .collect();
    AdinkraRep::from_parts(4, 4, &permutations, &dashing)
}

fn source_signed_words_match_fixtures() -> bool {
    (0..6).all(|sector| {
        (0..4).all(|color| {
            let word = S4_RECURSION_SIGNED_WORDS[sector][color];
            let permutation = word.map(i8::unsigned_abs);
            let factor = word.iter().enumerate().fold(0u8, |factor, (row, entry)| {
                factor | if *entry < 0 { 1 << row } else { 0 }
            });
            permutation == S4_ORDERED_QUARTETS[sector][color]
                && factor == S4_RECURSION_BOOLEAN_FACTORS[sector][color]
        })
    })
}

pub(crate) fn cyclic_mask(start: usize) -> (u8, [usize; 4]) {
    let positions = std::array::from_fn(|offset| (start + offset) % 8);
    let mask = positions
        .iter()
        .fold(0u8, |mask, &position| mask | (1 << position));
    (mask, positions.map(|position| position + 1))
}

fn recursive_permutations(first: usize, second: usize) -> [[u8; 8]; 8] {
    aligned_recursive_permutations(first, second, [0, 1, 2, 3])
}

pub(crate) fn aligned_recursive_permutations(
    first: usize,
    second: usize,
    second_color_alignment: [usize; 4],
) -> [[u8; 8]; 8] {
    std::array::from_fn(|color| {
        let local = color % 4;
        let left = S4_ORDERED_QUARTETS[first][local];
        let right = S4_ORDERED_QUARTETS[second][second_color_alignment[local]];
        if color < 4 {
            std::array::from_fn(|row| {
                if row < 4 {
                    left[row]
                } else {
                    right[row - 4] + 4
                }
            })
        } else {
            std::array::from_fn(|row| {
                if row < 4 {
                    left[row] + 4
                } else {
                    right[row - 4]
                }
            })
        }
    })
}

fn recursive_boolean_factors(first: usize, second: usize, mask: u8) -> [u8; 8] {
    aligned_recursive_boolean_factors(first, second, [0, 1, 2, 3], mask)
}

pub(crate) fn aligned_recursive_boolean_factors(
    first: usize,
    second: usize,
    second_color_alignment: [usize; 4],
    mask: u8,
) -> [u8; 8] {
    let first_four: [u8; 4] = std::array::from_fn(|color| {
        S4_RECURSION_BOOLEAN_FACTORS[first][color]
            | (S4_RECURSION_BOOLEAN_FACTORS[second][second_color_alignment[color]] << 4)
    });
    std::array::from_fn(|color| {
        if color < 4 {
            first_four[color]
        } else {
            first_four[color - 4] ^ mask
        }
    })
}

pub(crate) fn build_rep(permutations: &[[u8; 8]; 8], factors: &[u8; 8]) -> AdinkraRep {
    let color_permutations: Vec<Vec<usize>> = permutations
        .iter()
        .map(|p| p.iter().map(|&entry| usize::from(entry - 1)).collect())
        .collect();
    let dashing: Vec<i8> = factors
        .iter()
        .flat_map(|&factor| signs(factor, 8))
        .collect();
    AdinkraRep::from_parts(N, D, &color_permutations, &dashing)
}

fn multiply(left: &[Vec<i16>], right: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let n = left.len();
    let mut product = vec![vec![0i16; n]; n];
    for row in 0..n {
        for inner in 0..n {
            if left[row][inner] == 0 {
                continue;
            }
            for column in 0..n {
                product[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    product
}

fn dense_l(rep: &AdinkraRep, color: usize) -> Vec<Vec<i16>> {
    let mut matrix = vec![vec![0i16; D]; D];
    for row in 0..D {
        matrix[row][usize::from(rep.l_matrices[color].perm[row])] =
            i16::from(rep.l_matrices[color].sign[row]);
    }
    matrix
}

fn transpose(matrix: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let n = matrix.len();
    (0..n)
        .map(|row| (0..n).map(|column| matrix[column][row]).collect())
        .collect()
}

pub(crate) fn closure_record(rep: &AdinkraRep) -> ClosureRecord {
    let l: Vec<_> = (0..N).map(|color| dense_l(rep, color)).collect();
    let r: Vec<_> = l.iter().map(|matrix| transpose(matrix)).collect();
    let mut entries_checked = 0usize;
    let mut residual_entries = 0usize;
    let mut nonclosing_pairs = BTreeSet::new();
    let mut maximum = 0i16;
    for first in 0..N {
        for second in 0..N {
            for (left_family, right_family) in [(&l, &r), (&r, &l)] {
                let first_product = multiply(&left_family[first], &right_family[second]);
                let second_product = multiply(&left_family[second], &right_family[first]);
                for row in 0..D {
                    for column in 0..D {
                        let expected = if first == second && row == column {
                            2
                        } else {
                            0
                        };
                        let residual =
                            first_product[row][column] + second_product[row][column] - expected;
                        entries_checked += 1;
                        if residual != 0 {
                            residual_entries += 1;
                            nonclosing_pairs.insert([first.min(second), first.max(second)]);
                            maximum = maximum.max(residual.abs());
                        }
                    }
                }
            }
        }
    }
    ClosureRecord {
        sparse_garden_passed: rep.verify_garden_algebra(),
        entries_checked,
        residual_entries,
        nonclosing_color_pairs: nonclosing_pairs.len(),
        maximum_absolute_residual: maximum,
    }
}

pub(crate) fn hymn_record(rep: &AdinkraRep) -> HymnRecord {
    let mut product = (0..16)
        .map(|row| (0..16).map(|column| i16::from(row == column)).collect())
        .collect::<Vec<Vec<i16>>>();
    for color in 0..N {
        let l = dense_l(rep, color);
        let r = transpose(&l);
        let mut gamma = vec![vec![0i16; 16]; 16];
        for row in 0..D {
            for column in 0..D {
                gamma[row][column + D] = l[row][column];
                gamma[row + D][column] = r[row][column];
            }
        }
        product = multiply(&gamma, &product);
    }
    let diagonal_entries: Vec<i16> = (0..16).map(|index| product[index][index]).collect();
    let off_diagonal_entries = product
        .iter()
        .enumerate()
        .map(|(row, entries)| {
            entries
                .iter()
                .enumerate()
                .filter(|(column, value)| row != *column && **value != 0)
                .count()
        })
        .sum();
    let diagonal = off_diagonal_entries == 0;
    HymnRecord {
        diagonal,
        trace: diagonal_entries.iter().sum(),
        diagonal_entries,
        off_diagonal_entries,
        interpretation: if rep.verify_garden_algebra() {
            "HYMN invariant of a Garden-closing one-dimensional representation"
        } else {
            "formal matrix product only; this candidate does not satisfy Garden closure"
        },
    }
}

pub(crate) fn exact_published_match(
    permutations: &[[u8; 8]; 8],
    factors: &[u8; 8],
) -> Option<&'static str> {
    const LABELS: [&str; 6] = ["CC", "CT", "CV", "TT", "TV", "VV"];
    (0..6).find_map(|index| {
        (S8_REPRESENTATION_OCTETS[index].permutations == *permutations
            && S8_BASE_BOOLEAN_FACTORS[index] == *factors)
            .then_some(LABELS[index])
    })
}

pub(crate) fn matrix_checksum(rep: &AdinkraRep) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for matrix in &rep.l_matrices {
        for row in 0..D {
            for byte in [matrix.perm[row] as u8, matrix.sign[row] as u8] {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("{hash:016x}")
}

pub fn build() -> SignedRecursionArtifact {
    let input_reps: Vec<_> = (0..6).map(s4_rep).collect();
    let source_closing = input_reps
        .iter()
        .filter(|rep| rep.verify_garden_algebra())
        .count();
    let source_words_match = source_signed_words_match_fixtures();
    let vm2_correct = S4_ORDERED_QUARTETS[4].contains(&[3, 4, 2, 1])
        && !S4_ORDERED_QUARTETS[4].contains(&[3, 4, 1, 2])
        && S4_ORDERED_QUARTETS[5].contains(&[3, 4, 1, 2]);

    let mut pairs = Vec::with_capacity(30);
    let mut closing_reps = Vec::<(ClosingCandidateKey, AdinkraRep)>::new();
    let mut dense_checked = 0usize;
    let mut ct_anchor = false;
    for first in 0..6 {
        for second in 0..6 {
            if first == second {
                continue;
            }
            let permutations = recursive_permutations(first, second);
            let mut candidates = Vec::with_capacity(8);
            let mut closing_positions = Vec::new();
            for start in 0..8 {
                let (mask, positions) = cyclic_mask(start);
                let factors = recursive_boolean_factors(first, second, mask);
                let rep = build_rep(&permutations, &factors);
                let closure = closure_record(&rep);
                dense_checked += closure.entries_checked;
                let closes = closure.sparse_garden_passed && closure.residual_entries == 0;
                let self_gadget = closes.then(|| {
                    let holoraumy = HoloraumyData::from_rep(&rep);
                    gadget(&holoraumy, &holoraumy)
                });
                let commutant_dimension = closes.then(|| commutant_dim(&rep));
                let antisymmetric_commutant_dimension =
                    closes.then(|| antisymmetric_commutant_dim(&rep));
                let published_match = exact_published_match(&permutations, &factors);
                if first == 0
                    && second == 1
                    && start == 5
                    && factors == S8_BASE_BOOLEAN_FACTORS[1]
                    && published_match == Some("CT")
                    && closes
                {
                    ct_anchor = true;
                }
                if closes {
                    closing_positions.push(start + 1);
                    closing_reps.push((
                        ClosingCandidateKey {
                            ordered_pair: format!(
                                "{}->{}",
                                S4_RECURSION_LABELS[first], S4_RECURSION_LABELS[second]
                            ),
                            start_position_one_based: start + 1,
                        },
                        rep.clone(),
                    ));
                }
                candidates.push(FlipCandidateRecord {
                    start_position_one_based: start + 1,
                    flipped_positions_one_based: positions,
                    flip_mask_decimal: mask,
                    boolean_factors: factors,
                    permutations,
                    closure,
                    hymn: hymn_record(&rep),
                    self_gadget,
                    commutant_dimension,
                    antisymmetric_commutant_dimension,
                    exact_published_system_match: published_match,
                    exact_matrix_checksum_fnv1a64: matrix_checksum(&rep),
                });
            }
            pairs.push(OrderedPairRecord {
                first_index: first + 1,
                first_label: S4_RECURSION_LABELS[first],
                second_index: second + 1,
                second_label: S4_RECURSION_LABELS[second],
                first_boolean_factors: S4_RECURSION_BOOLEAN_FACTORS[first],
                second_boolean_factors: S4_RECURSION_BOOLEAN_FACTORS[second],
                first_input_closes: input_reps[first].verify_garden_algebra(),
                second_input_closes: input_reps[second].verify_garden_algebra(),
                first_closing_start_position_one_based: closing_positions.first().copied(),
                closing_start_positions_one_based: closing_positions,
                candidates,
            });
        }
    }

    let holoraumy: Vec<_> = closing_reps
        .iter()
        .map(|(_, rep)| HoloraumyData::from_rep(rep))
        .collect();
    let gadget_matrix = holoraumy
        .iter()
        .map(|left| holoraumy.iter().map(|right| gadget(left, right)).collect())
        .collect();
    let checksums: BTreeSet<String> = pairs
        .iter()
        .flat_map(|pair| &pair.candidates)
        .filter(|candidate| candidate.closure.sparse_garden_passed)
        .map(|candidate| candidate.exact_matrix_checksum_fnv1a64.clone())
        .collect();
    let closing_count = closing_reps.len();
    let pairs_with_closure = pairs
        .iter()
        .filter(|pair| !pair.closing_start_positions_one_based.is_empty())
        .count();
    let self_gadgets_one = pairs
        .iter()
        .flat_map(|pair| &pair.candidates)
        .filter_map(|candidate| candidate.self_gadget)
        .all(|value| (value - 1.0).abs() < 1e-12);
    let scalar_commutants = pairs
        .iter()
        .flat_map(|pair| &pair.candidates)
        .filter(|candidate| candidate.closure.sparse_garden_passed)
        .all(|candidate| {
            candidate.commutant_dimension == Some(1)
                && candidate.antisymmetric_commutant_dimension == Some(0)
        });
    let passed = source_closing == 6
        && source_words_match
        && vm2_correct
        && pairs.len() == 30
        && pairs.iter().all(|pair| pair.candidates.len() == 8)
        && dense_checked == 240 * 2 * N * N * D * D
        && ct_anchor
        && self_gadgets_one
        && scalar_commutants;

    let mut conventions = BTreeMap::new();
    conventions.insert(
        "Boolean word",
        "bit zero controls matrix row one, following Eq. (2.21)",
    );
    conventions.insert(
        "ordered pairs",
        "all 6 x 5 ordered pairs with unequal S4 labels",
    );
    conventions.insert(
        "cyclic flip",
        "the eight four-adjacent-bit runs listed after Eq. (2.23)",
    );
    conventions.insert(
        "acceptance",
        "exact bosonic and fermionic Garden closure with R_I = L_I transpose",
    );
    conventions.insert(
        "selection",
        "all closing masks are retained; first-closing order is recorded but not treated as invariant",
    );

    SignedRecursionArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Signed Boolean recursion on all thirty ordered distinct S4 pairs",
        sources: vec![
            SourceRecord {
                arxiv_id: "2304.09830",
                version: 2,
                locator: "PDF pp. 9-12, Eqs. (2.13), (2.17)-(2.25), Secs. 2.1-2.2",
                role: "six signed S4 inputs, unsigned block recursion, Boolean concatenation, cyclic four-bit flip, and Garden acceptance rule",
            },
            SourceRecord {
                arxiv_id: "2012.14015",
                version: 7,
                locator: "Eqs. (5.1)-(5.6)",
                role: "exact signed CC, CT, CV, TT, TV, and VV comparison fixtures",
            },
        ],
        conventions,
        source_s4_labels: S4_RECURSION_LABELS,
        source_s4_boolean_factors: S4_RECURSION_BOOLEAN_FACTORS,
        closing_candidates: closing_reps.into_iter().map(|(key, _)| key).collect(),
        closing_candidate_gadget_matrix: gadget_matrix,
        ordered_pairs: pairs,
        validation: ValidationRecord {
            signed_s4_source_representatives_checked: 6,
            signed_s4_source_representatives_closing: source_closing,
            source_signed_words_match_fixtures: source_words_match,
            vm2_uses_3421_not_3412: vm2_correct,
            ordered_distinct_pairs_checked: 30,
            cyclic_four_bit_masks_checked_per_pair: 8,
            signed_candidates_checked: 240,
            dense_garden_entries_checked: dense_checked,
            closing_candidates: closing_count,
            nonclosing_candidates: 240 - closing_count,
            ordered_pairs_with_at_least_one_closing_mask: pairs_with_closure,
            ordered_pairs_without_a_closing_mask: 30 - pairs_with_closure,
            distinct_exact_closing_matrix_systems: checksums.len(),
            ct_source_anchor_matched: ct_anchor,
            all_closing_self_gadgets_equal_one: self_gadgets_one,
            all_closing_commutants_are_scalar: scalar_commutants,
            passed,
        },
        boundary: "This is an exhaustive test of the paper's stated eight cyclic flip choices on the thirty ordered distinct source pairs under the printed S4 signs. It does not enumerate arbitrary Garden signings, define complete signed equivalence classes, or establish four-dimensional enhancement.",
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ValidationRecord {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create signed recursion data")),
        &artifact,
    )
    .expect("write signed recursion data");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create signed recursion validation")),
        &artifact.validation,
    )
    .expect("write signed recursion validation");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_printed_s4_inputs_close_and_vm2_fixture_is_correct() {
        for sector in 0..6 {
            for color in 0..4 {
                let word = S4_RECURSION_SIGNED_WORDS[sector][color];
                let permutation = word.map(i8::unsigned_abs);
                let factor = word.iter().enumerate().fold(0u8, |factor, (row, entry)| {
                    factor | if *entry < 0 { 1 << row } else { 0 }
                });
                assert_eq!(permutation, S4_ORDERED_QUARTETS[sector][color]);
                assert_eq!(factor, S4_RECURSION_BOOLEAN_FACTORS[sector][color]);
            }
        }
        assert!((0..6).all(|index| s4_rep(index).verify_garden_algebra()));
        assert!(S4_ORDERED_QUARTETS[4].contains(&[3, 4, 2, 1]));
        assert!(!S4_ORDERED_QUARTETS[4].contains(&[3, 4, 1, 2]));
        assert!(S4_ORDERED_QUARTETS[5].contains(&[3, 4, 1, 2]));
    }

    #[test]
    fn cyclic_masks_are_exactly_the_eight_runs_printed_in_section_2_2() {
        let actual: Vec<_> = (0..8).map(cyclic_mask).collect();
        assert_eq!(actual[0], (15, [1, 2, 3, 4]));
        assert_eq!(actual[4], (240, [5, 6, 7, 8]));
        assert_eq!(actual[5], (225, [6, 7, 8, 1]));
        assert_eq!(actual[7], (135, [8, 1, 2, 3]));
        assert_eq!(
            actual
                .iter()
                .map(|entry| entry.0)
                .collect::<BTreeSet<_>>()
                .len(),
            8
        );
        assert!(actual.iter().all(|(mask, _)| mask.count_ones() == 4));
    }

    #[test]
    fn ct_example_matches_equations_2_23_through_2_25_exactly() {
        let permutations = recursive_permutations(0, 1);
        let factors = recursive_boolean_factors(0, 1, cyclic_mask(5).0);
        assert_eq!(factors, [234, 76, 134, 32, 11, 173, 103, 193]);
        assert_eq!(permutations, S8_REPRESENTATION_OCTETS[1].permutations);
        assert_eq!(exact_published_match(&permutations, &factors), Some("CT"));
        assert!(build_rep(&permutations, &factors).verify_garden_algebra());
    }

    #[test]
    fn full_scan_counts_every_ordered_pair_and_flip() {
        let artifact = build();
        assert_eq!(artifact.ordered_pairs.len(), 30);
        assert!(artifact
            .ordered_pairs
            .iter()
            .all(|pair| pair.first_index != pair.second_index));
        assert!(artifact
            .ordered_pairs
            .iter()
            .all(|pair| pair.candidates.len() == 8));
        assert_eq!(artifact.validation.signed_candidates_checked, 240);
        assert_eq!(artifact.validation.dense_garden_entries_checked, 1_966_080);
        assert_eq!(artifact.validation.closing_candidates, 16);
        assert_eq!(
            artifact
                .validation
                .ordered_pairs_with_at_least_one_closing_mask,
            8
        );
        assert!(artifact.validation.all_closing_commutants_are_scalar);
        let actual_pairs: BTreeSet<_> = artifact
            .ordered_pairs
            .iter()
            .filter(|pair| !pair.closing_start_positions_one_based.is_empty())
            .map(|pair| {
                assert_eq!(pair.closing_start_positions_one_based, [2, 6]);
                (pair.first_label, pair.second_label)
            })
            .collect();
        let expected_pairs = BTreeSet::from([
            ("CM", "TM"),
            ("CM", "VM"),
            ("TM", "CM"),
            ("VM", "CM"),
            ("VM1", "VM2"),
            ("VM1", "VM3"),
            ("VM2", "VM1"),
            ("VM3", "VM1"),
        ]);
        assert_eq!(actual_pairs, expected_pairs);
        assert!(artifact.validation.passed);
    }

    #[test]
    fn relative_color_alignment_census_is_exact() {
        let alignments: Vec<[usize; 4]> = crate::permutahedron::permutations(4)
            .unwrap()
            .map(|permutation| {
                permutation
                    .as_slice()
                    .iter()
                    .map(|entry| usize::from(entry - 1))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        let mut closing = Vec::new();
        for first in 0..6 {
            for second in 0..6 {
                if first == second {
                    continue;
                }
                for alignment in &alignments {
                    let permutations = aligned_recursive_permutations(first, second, *alignment);
                    for start in 0..8 {
                        let factors = aligned_recursive_boolean_factors(
                            first,
                            second,
                            *alignment,
                            cyclic_mask(start).0,
                        );
                        let rep = build_rep(&permutations, &factors);
                        if rep.verify_garden_algebra() {
                            closing.push((first, second, *alignment, start));
                        }
                    }
                }
            }
        }
        let expected = vec![
            (0, 1, [0, 1, 2, 3], 1),
            (0, 1, [0, 1, 2, 3], 5),
            (0, 2, [0, 1, 2, 3], 1),
            (0, 2, [0, 1, 2, 3], 5),
            (0, 2, [1, 0, 3, 2], 3),
            (0, 2, [1, 0, 3, 2], 7),
            (1, 0, [0, 1, 2, 3], 1),
            (1, 0, [0, 1, 2, 3], 5),
            (2, 0, [0, 1, 2, 3], 1),
            (2, 0, [0, 1, 2, 3], 5),
            (2, 0, [1, 0, 3, 2], 3),
            (2, 0, [1, 0, 3, 2], 7),
            (3, 4, [0, 1, 2, 3], 1),
            (3, 4, [0, 1, 2, 3], 5),
            (3, 5, [0, 1, 2, 3], 1),
            (3, 5, [0, 1, 2, 3], 5),
            (3, 5, [2, 3, 0, 1], 3),
            (3, 5, [2, 3, 0, 1], 7),
            (4, 3, [0, 1, 2, 3], 1),
            (4, 3, [0, 1, 2, 3], 5),
            (5, 3, [0, 1, 2, 3], 1),
            (5, 3, [0, 1, 2, 3], 5),
            (5, 3, [2, 3, 0, 1], 3),
            (5, 3, [2, 3, 0, 1], 7),
        ];
        assert_eq!(closing, expected);
    }
}
