//! Garden-sign feasibility and normalizer analysis for the complete `S_8`
//! Rana-coset atlas.
//!
//! An unsigned octet specifies eight `8 x 8` permutation matrices.  A Garden
//! signing assigns one sign bit to every nonzero matrix entry and requires
//!
//! `L_I L_J^T + L_J L_I^T = 2 delta_{IJ} I`.
//!
//! For distinct colors the support condition is checked first.  When the two
//! monomial products have the same support, cancellation gives affine linear
//! equations over `GF(2)`.  There are 64 sign variables.  Gaussian elimination
//! determines feasibility, rank, nullity, and a canonical solution.  Every
//! solution reported by the linear solver is then checked by direct integer
//! matrix multiplication through the repository's independent `AdinkraRep`
//! verifier and by a dense entry-by-entry residual calculation.

use crate::lr_matrix::AdinkraRep;
use crate::permutahedron::{
    CosetSide, Permutation, abnormal_slices, coset_partition, factorial, permutations, rana_r8,
};
use crate::permutahedron_fixtures::{R8_DIADEM_OCTET, S8_REPRESENTATION_OCTETS};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

const N: usize = 8;
const D: usize = 8;
const VARIABLE_COUNT: usize = N * D;

#[derive(Debug, Clone, Serialize)]
pub struct GardenSigningResult {
    pub support_compatible: bool,
    pub feasible: bool,
    pub color_pairs_checked: usize,
    pub unique_affine_equations: usize,
    pub rank: usize,
    pub nullity: usize,
    pub solution_count: u64,
    /// Row-major signs: `signs[color * 8 + row]`.
    pub canonical_signs: Vec<i8>,
    pub independent_sparse_verifier_passed: bool,
    pub dense_entries_checked: usize,
    pub dense_residual_entries: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClosureContingency {
    pub abnormal_and_signable: usize,
    pub abnormal_and_not_signable: usize,
    pub ordinary_and_signable: usize,
    pub ordinary_and_not_signable: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistogramEntry {
    pub value: usize,
    pub octets: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedOctetFeasibility {
    pub id: String,
    pub right_slice_id: u32,
    pub abnormal: bool,
    pub published_status: String,
    pub garden_signing_exists: bool,
    pub solution_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CosetGardenRecord {
    pub right_slice_id: u32,
    pub representative_rank: u32,
    pub abnormal: bool,
    pub support_compatible: bool,
    pub garden_signing_exists: bool,
    pub equation_rank: usize,
    pub nullity: usize,
    pub solution_count: u64,
    /// The canonical 64 sign bits as sixteen hexadecimal digits.  A set bit
    /// means `-1`; a clear bit means `+1`.
    pub canonical_sign_mask: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalizerReport {
    pub ambient_group: &'static str,
    pub subgroup: &'static str,
    pub subgroup_order: usize,
    pub subgroup_is_elementary_abelian_rank_three: bool,
    pub normalizer_order: usize,
    pub normalizer_cosets: usize,
    pub abnormal_cosets: usize,
    pub abnormal_iff_representative_normalizes: bool,
    pub distinct_conjugation_automorphisms: usize,
    pub expected_gl_3_2_order: usize,
    pub quotient_identification: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompleteGardenScan {
    pub schema_version: &'static str,
    pub unsigned_object: &'static str,
    pub sign_equation: &'static str,
    pub cosets_scanned: usize,
    pub signable_cosets: usize,
    pub unsignable_cosets: usize,
    pub rank_histogram: Vec<HistogramEntry>,
    pub nullity_histogram: Vec<HistogramEntry>,
    pub solution_count_per_coset: u64,
    pub total_signings_across_labeled_cosets: u64,
    pub independent_sparse_verifications: usize,
    pub dense_entries_checked: usize,
    pub dense_residual_entries: usize,
    pub contingency: ClosureContingency,
    pub normalizer: NormalizerReport,
    pub named_octets: Vec<NamedOctetFeasibility>,
    pub cosets: Vec<CosetGardenRecord>,
    pub conclusion: &'static str,
    pub boundary: &'static str,
    pub passed: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Equation {
    coefficients: u64,
    rhs: bool,
}

fn zero_based(permutation: Permutation) -> Vec<usize> {
    permutation
        .as_slice()
        .iter()
        .map(|&value| usize::from(value - 1))
        .collect()
}

fn sign_variable(color: usize, row: usize) -> usize {
    color * D + row
}

fn affine_equations(octet: &[Permutation]) -> Option<Vec<Equation>> {
    if octet.len() != N || octet.iter().any(|p| p.n() != D) {
        return None;
    }
    let perms: Vec<Vec<usize>> = octet.iter().copied().map(zero_based).collect();
    let inverses: Vec<Vec<usize>> = perms
        .iter()
        .map(|p| {
            let mut inverse = vec![0usize; D];
            for (row, &column) in p.iter().enumerate() {
                inverse[column] = row;
            }
            inverse
        })
        .collect();
    let mut equations = BTreeSet::new();
    for i in 0..N {
        for j in i + 1..N {
            for row in 0..D {
                // (L_i L_j^T)[row, partner] is the sole nonzero entry in
                // this row of the first product.
                let partner = inverses[j][perms[i][row]];
                // The second product must have its nonzero at the same place.
                if perms[j][row] != perms[i][partner] {
                    return None;
                }
                let mut coefficients = 0u64;
                for variable in [
                    sign_variable(i, row),
                    sign_variable(j, partner),
                    sign_variable(j, row),
                    sign_variable(i, partner),
                ] {
                    coefficients ^= 1u64 << variable;
                }
                // The two signed monomials must have opposite signs.
                equations.insert(Equation {
                    coefficients,
                    rhs: true,
                });
            }
        }
    }
    Some(equations.into_iter().collect())
}

fn solve_affine(mut equations: Vec<Equation>) -> Option<(usize, Vec<bool>)> {
    let mut pivot_row = 0usize;
    let mut pivot_columns = Vec::new();
    for column in 0..VARIABLE_COUNT {
        let Some(found) = (pivot_row..equations.len())
            .find(|&row| ((equations[row].coefficients >> column) & 1) == 1)
        else {
            continue;
        };
        equations.swap(pivot_row, found);
        let pivot = equations[pivot_row];
        for row in 0..equations.len() {
            if row != pivot_row && ((equations[row].coefficients >> column) & 1) == 1 {
                equations[row].coefficients ^= pivot.coefficients;
                equations[row].rhs ^= pivot.rhs;
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
    }
    if equations
        .iter()
        .any(|equation| equation.coefficients == 0 && equation.rhs)
    {
        return None;
    }
    let rank = pivot_columns.len();
    let mut solution = vec![false; VARIABLE_COUNT];
    // Free variables are set to zero.  Reduced row echelon form then makes
    // each pivot equal to its row's right-hand side.
    for (row, &column) in pivot_columns.iter().enumerate() {
        solution[column] = equations[row].rhs;
    }
    Some((rank, solution))
}

/// Enumerate every Garden signing for one support-compatible unsigned octet.
///
/// Each signing is returned as a 64-bit mask in color-major, row-minor order.
/// A set bit means a negative matrix entry.  The S8/R8 systems used here have
/// nullity 19, so this returns 524,288 masks per octet.
pub fn garden_solution_masks(octet: &[Permutation]) -> Option<Vec<u64>> {
    let mut equations = affine_equations(octet)?;
    let mut pivot_row = 0usize;
    let mut pivot_columns = Vec::new();
    for column in 0..VARIABLE_COUNT {
        let Some(found) = (pivot_row..equations.len())
            .find(|&row| ((equations[row].coefficients >> column) & 1) == 1)
        else {
            continue;
        };
        equations.swap(pivot_row, found);
        let pivot = equations[pivot_row];
        for row in 0..equations.len() {
            if row != pivot_row && ((equations[row].coefficients >> column) & 1) == 1 {
                equations[row].coefficients ^= pivot.coefficients;
                equations[row].rhs ^= pivot.rhs;
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
    }
    if equations
        .iter()
        .any(|equation| equation.coefficients == 0 && equation.rhs)
    {
        return None;
    }

    let pivot_set: BTreeSet<usize> = pivot_columns.iter().copied().collect();
    let free_columns: Vec<usize> = (0..VARIABLE_COUNT)
        .filter(|column| !pivot_set.contains(column))
        .collect();
    let solution_count = 1usize << free_columns.len();
    let mut solutions = Vec::with_capacity(solution_count);
    for free_assignment in 0..solution_count {
        let mut mask = 0u64;
        for (bit, &column) in free_columns.iter().enumerate() {
            if (free_assignment >> bit) & 1 == 1 {
                mask |= 1u64 << column;
            }
        }
        for (row, &pivot_column) in pivot_columns.iter().enumerate() {
            let free_parity = (equations[row].coefficients & mask).count_ones() % 2 == 1;
            let pivot_value = equations[row].rhs ^ free_parity;
            if pivot_value {
                mask |= 1u64 << pivot_column;
            }
        }
        if free_assignment == 0 || free_assignment + 1 == solution_count {
            debug_assert!(equations.iter().all(|equation| {
                let lhs = (equation.coefficients & mask).count_ones() % 2 == 1;
                lhs == equation.rhs
            }));
        }
        solutions.push(mask);
    }
    Some(solutions)
}

fn dense_residual(octet: &[Permutation], signs: &[i8]) -> (usize, usize) {
    let perms: Vec<Vec<usize>> = octet.iter().copied().map(zero_based).collect();
    let mut checked = 0usize;
    let mut residuals = 0usize;
    for i in 0..N {
        for j in 0..N {
            for row in 0..D {
                for column in 0..D {
                    let mut value = 0i16;
                    for intermediate in 0..D {
                        if perms[i][row] == intermediate && perms[j][column] == intermediate {
                            value += i16::from(signs[sign_variable(i, row)])
                                * i16::from(signs[sign_variable(j, column)]);
                        }
                        if perms[j][row] == intermediate && perms[i][column] == intermediate {
                            value += i16::from(signs[sign_variable(j, row)])
                                * i16::from(signs[sign_variable(i, column)]);
                        }
                    }
                    let expected = if i == j && row == column { 2 } else { 0 };
                    checked += 1;
                    residuals += usize::from(value != expected);
                }
            }
        }
    }
    (checked, residuals)
}

pub fn solve_garden_signing(octet: &[Permutation]) -> GardenSigningResult {
    let Some(equations) = affine_equations(octet) else {
        return GardenSigningResult {
            support_compatible: false,
            feasible: false,
            color_pairs_checked: N * (N - 1) / 2,
            unique_affine_equations: 0,
            rank: 0,
            nullity: 0,
            solution_count: 0,
            canonical_signs: Vec::new(),
            independent_sparse_verifier_passed: false,
            dense_entries_checked: 0,
            dense_residual_entries: 0,
        };
    };
    let equation_count = equations.len();
    let Some((rank, bits)) = solve_affine(equations) else {
        return GardenSigningResult {
            support_compatible: true,
            feasible: false,
            color_pairs_checked: N * (N - 1) / 2,
            unique_affine_equations: equation_count,
            rank: 0,
            nullity: 0,
            solution_count: 0,
            canonical_signs: Vec::new(),
            independent_sparse_verifier_passed: false,
            dense_entries_checked: 0,
            dense_residual_entries: 0,
        };
    };
    let nullity = VARIABLE_COUNT - rank;
    let solution_count = 1u64 << nullity;
    let signs: Vec<i8> = bits
        .into_iter()
        .map(|bit| if bit { -1 } else { 1 })
        .collect();
    let color_perms: Vec<Vec<usize>> = octet.iter().copied().map(zero_based).collect();
    let representation = AdinkraRep::from_parts(N, D, &color_perms, &signs);
    let sparse_passed = representation.verify_garden_algebra();
    let (dense_entries_checked, dense_residual_entries) = dense_residual(octet, &signs);
    GardenSigningResult {
        support_compatible: true,
        feasible: true,
        color_pairs_checked: N * (N - 1) / 2,
        unique_affine_equations: equation_count,
        rank,
        nullity,
        solution_count,
        canonical_signs: signs,
        independent_sparse_verifier_passed: sparse_passed,
        dense_entries_checked,
        dense_residual_entries,
    }
}

fn conjugate(g: Permutation, h: Permutation) -> Permutation {
    g.compose(h)
        .expect("common S8 order")
        .compose(g.inverse())
        .expect("common S8 order")
}

fn normalizer_report() -> NormalizerReport {
    let subgroup = rana_r8();
    let subgroup_ranks: HashSet<usize> = subgroup.iter().map(|p| p.rank()).collect();
    let identity = Permutation::identity(N).expect("S8 identity");
    let elementary_abelian = subgroup
        .iter()
        .all(|&p| p == identity || p.compose(p).expect("S8 product") == identity)
        && subgroup.iter().all(|&p| {
            subgroup
                .iter()
                .all(|&q| p.compose(q).expect("S8 product") == q.compose(p).expect("S8 product"))
        });

    let normalizer: Vec<Permutation> = permutations(N)
        .expect("S8 permutations")
        .filter(|&g| {
            subgroup
                .iter()
                .all(|&h| subgroup_ranks.contains(&conjugate(g, h).rank()))
        })
        .collect();

    let nonidentity: Vec<Permutation> = subgroup
        .iter()
        .copied()
        .filter(|&p| p != identity)
        .collect();
    let index: BTreeMap<usize, u8> = nonidentity
        .iter()
        .enumerate()
        .map(|(i, p)| (p.rank(), i as u8))
        .collect();
    let actions: BTreeSet<[u8; 7]> = normalizer
        .iter()
        .map(|&g| {
            let mut action = [0u8; 7];
            for (i, &h) in nonidentity.iter().enumerate() {
                action[i] = index[&conjugate(g, h).rank()];
            }
            action
        })
        .collect();

    let abnormal = abnormal_slices(&subgroup).expect("R8 abnormal cosets");
    let right = coset_partition(&subgroup, CosetSide::Right).expect("R8 right cosets");
    let abnormal_minima: HashSet<u32> = abnormal.representative_ranks.iter().copied().collect();
    let normalizer_ranks: HashSet<usize> = normalizer.iter().map(|p| p.rank()).collect();
    let iff = right.slices.iter().all(|slice| {
        abnormal_minima.contains(&slice[0]) == normalizer_ranks.contains(&(slice[0] as usize))
    });

    NormalizerReport {
        ambient_group: "S8",
        subgroup: "R8",
        subgroup_order: subgroup.len(),
        subgroup_is_elementary_abelian_rank_three: elementary_abelian,
        normalizer_order: normalizer.len(),
        normalizer_cosets: normalizer.len() / subgroup.len(),
        abnormal_cosets: abnormal.abnormal_slice_count,
        abnormal_iff_representative_normalizes: iff,
        distinct_conjugation_automorphisms: actions.len(),
        expected_gl_3_2_order: (8 - 1) * (8 - 2) * (8 - 4),
        quotient_identification: "N_S8(R8)/R8 is GL(3,2); N_S8(R8) is AGL(3,2)",
    }
}

fn histogram(values: impl Iterator<Item = usize>) -> Vec<HistogramEntry> {
    let mut counts = BTreeMap::new();
    for value in values {
        *counts.entry(value).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .map(|(value, octets)| HistogramEntry { value, octets })
        .collect()
}

fn canonical_mask(signs: &[i8]) -> String {
    let mask = signs.iter().enumerate().fold(0u64, |mask, (index, &sign)| {
        mask | (u64::from(sign == -1) << index)
    });
    format!("{mask:016x}")
}

fn slice_for_octet(right: &crate::permutahedron::CosetPartitionReport, octet: &[[u8; 8]]) -> u32 {
    let first_rank = Permutation::new(&octet[0])
        .expect("published S8 permutation")
        .rank() as u32;
    right
        .slices
        .iter()
        .position(|slice| slice.binary_search(&first_rank).is_ok())
        .expect("published octet belongs to an R8 right coset") as u32
}

pub fn complete_garden_scan() -> CompleteGardenScan {
    let subgroup = rana_r8();
    let right = coset_partition(&subgroup, CosetSide::Right).expect("R8 right cosets");
    let abnormal = abnormal_slices(&subgroup).expect("R8 abnormal cosets");
    let abnormal_minima: HashSet<u32> = abnormal.representative_ranks.iter().copied().collect();
    let abnormal_ids: HashSet<u32> = right
        .slices
        .iter()
        .enumerate()
        .filter_map(|(id, slice)| abnormal_minima.contains(&slice[0]).then_some(id as u32))
        .collect();

    let mut results = Vec::with_capacity(right.slices.len());
    let mut sparse_verifications = 0usize;
    let mut dense_checked = 0usize;
    let mut dense_residuals = 0usize;
    for slice in &right.slices {
        let octet: Vec<Permutation> = slice
            .iter()
            .map(|&rank| Permutation::unrank(N, rank as usize).expect("S8 rank"))
            .collect();
        let result = solve_garden_signing(&octet);
        sparse_verifications += usize::from(result.independent_sparse_verifier_passed);
        dense_checked += result.dense_entries_checked;
        dense_residuals += result.dense_residual_entries;
        results.push(result);
    }
    let signable = results.iter().filter(|result| result.feasible).count();
    let abnormal_signable = results
        .iter()
        .enumerate()
        .filter(|(id, result)| abnormal_ids.contains(&(*id as u32)) && result.feasible)
        .count();
    let ordinary_signable = signable - abnormal_signable;

    let mut named: Vec<(&str, u32, &str)> = S8_REPRESENTATION_OCTETS
        .iter()
        .map(|octet| {
            let status = match octet.label {
                "CT" | "CV" => "published_garden_closure",
                "CC" | "TT" | "TV" | "VV" => "published_nonclosure",
                _ => unreachable!(),
            };
            (
                octet.label,
                slice_for_octet(&right, &octet.permutations),
                status,
            )
        })
        .collect();
    named.push((
        "O",
        slice_for_octet(&right, &R8_DIADEM_OCTET),
        "published_garden_closure",
    ));
    let named_octets = named
        .iter()
        .map(|&(id, slice, published_status)| {
            let result = &results[slice as usize];
            NamedOctetFeasibility {
                id: id.into(),
                right_slice_id: slice,
                abnormal: abnormal_ids.contains(&slice),
                published_status: published_status.into(),
                garden_signing_exists: result.feasible,
                solution_count: result.solution_count,
            }
        })
        .collect();

    let cosets = right
        .slices
        .iter()
        .zip(results.iter())
        .enumerate()
        .map(|(id, (slice, result))| CosetGardenRecord {
            right_slice_id: id as u32,
            representative_rank: slice[0],
            abnormal: abnormal_ids.contains(&(id as u32)),
            support_compatible: result.support_compatible,
            garden_signing_exists: result.feasible,
            equation_rank: result.rank,
            nullity: result.nullity,
            solution_count: result.solution_count,
            canonical_sign_mask: canonical_mask(&result.canonical_signs),
        })
        .collect();

    let normalizer = normalizer_report();
    let common_solution_count = results
        .first()
        .map(|result| result.solution_count)
        .unwrap_or(0);
    let passed = right.slice_count == factorial(N).expect("8 factorial") / subgroup.len()
        && signable == right.slice_count
        && sparse_verifications == right.slice_count
        && dense_residuals == 0
        && normalizer.normalizer_order == 1_344
        && normalizer.normalizer_cosets == 168
        && normalizer.abnormal_iff_representative_normalizes
        && normalizer.distinct_conjugation_automorphisms == 168;

    CompleteGardenScan {
        schema_version: "permutahedron-garden-scan-v1",
        unsigned_object: "eight 8x8 permutation matrices in each right R8 coset",
        sign_equation: "L_I L_J^T + L_J L_I^T = 2 delta_IJ I over signed monomial matrices",
        cosets_scanned: results.len(),
        signable_cosets: signable,
        unsignable_cosets: results.len() - signable,
        rank_histogram: histogram(results.iter().map(|result| result.rank)),
        nullity_histogram: histogram(results.iter().map(|result| result.nullity)),
        solution_count_per_coset: common_solution_count,
        total_signings_across_labeled_cosets: common_solution_count * results.len() as u64,
        independent_sparse_verifications: sparse_verifications,
        dense_entries_checked: dense_checked,
        dense_residual_entries: dense_residuals,
        contingency: ClosureContingency {
            abnormal_and_signable: abnormal_signable,
            abnormal_and_not_signable: abnormal_ids.len() - abnormal_signable,
            ordinary_and_signable: ordinary_signable,
            ordinary_and_not_signable: results.len() - abnormal_ids.len() - ordinary_signable,
        },
        normalizer,
        named_octets,
        cosets,
        conclusion: "Every R8 right coset admits Garden signs. Ab-normality therefore does not distinguish unsigned sign feasibility.",
        boundary: "Existence of some Garden signing is not the closure status of a particular published Boolean-factor assignment or a four-dimensional supermultiplet construction.",
        passed,
    }
}

pub fn write_complete_garden_scan(path: &Path) -> CompleteGardenScan {
    let report = complete_garden_scan();
    assert!(report.passed, "complete S8 Garden scan failed");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    let mut bytes = serde_json::to_vec(&report).expect("serialize complete Garden scan");
    bytes.push(b'\n');
    std::fs::write(path, bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permutahedron_fixtures::{R8_DIADEM_OCTET, S8_REPRESENTATION_OCTETS};

    fn from_array(values: &[u8; 8]) -> Permutation {
        Permutation::new(values).unwrap()
    }

    #[test]
    fn diadem_sign_system_has_the_expected_dashing_dimension() {
        let octet: Vec<Permutation> = R8_DIADEM_OCTET.iter().map(from_array).collect();
        let result = solve_garden_signing(&octet);
        assert!(result.support_compatible);
        assert!(result.feasible);
        assert_eq!(result.unique_affine_equations, 112);
        assert_eq!(result.rank, 45);
        assert_eq!(result.nullity, 19);
        assert_eq!(result.solution_count, 1 << 19);
        assert!(result.independent_sparse_verifier_passed);
        assert_eq!(result.dense_entries_checked, 4_096);
        assert_eq!(result.dense_residual_entries, 0);

        let mut mutated = result.canonical_signs.clone();
        mutated[0] *= -1;
        let color_perms: Vec<Vec<usize>> = octet.iter().copied().map(zero_based).collect();
        let representation = AdinkraRep::from_parts(N, D, &color_perms, &mutated);
        assert!(!representation.verify_garden_algebra());
        assert!(dense_residual(&octet, &mutated).1 > 0);

        let solution_masks = garden_solution_masks(&octet).expect("Diadem solution space");
        assert_eq!(solution_masks.len(), 1 << 19);
        assert_eq!(
            solution_masks.iter().copied().collect::<HashSet<_>>().len(),
            1 << 19
        );
        assert!(
            solution_masks.contains(
                &u64::from_str_radix(&canonical_mask(&result.canonical_signs), 16).unwrap()
            )
        );
    }

    #[test]
    fn all_seven_published_unsigned_octets_admit_some_garden_signing() {
        for fixture in &S8_REPRESENTATION_OCTETS {
            let octet: Vec<Permutation> = fixture.permutations.iter().map(from_array).collect();
            let result = solve_garden_signing(&octet);
            assert!(result.feasible, "{} unsigned octet", fixture.label);
            assert_eq!(result.rank, 45);
            assert_eq!(result.nullity, 19);
            assert!(result.independent_sparse_verifier_passed);
            assert_eq!(result.dense_residual_entries, 0);
        }
    }

    #[test]
    fn support_incompatible_octet_is_rejected() {
        let octet: Vec<Permutation> = (0..8)
            .map(|rank| Permutation::unrank(8, rank).unwrap())
            .collect();
        let result = solve_garden_signing(&octet);
        assert!(!result.support_compatible);
        assert!(!result.feasible);
    }

    #[test]
    fn normalizer_explains_the_number_168() {
        let report = normalizer_report();
        assert!(report.subgroup_is_elementary_abelian_rank_three);
        assert_eq!(report.normalizer_order, 1_344);
        assert_eq!(report.normalizer_cosets, 168);
        assert_eq!(report.abnormal_cosets, 168);
        assert!(report.abnormal_iff_representative_normalizes);
        assert_eq!(report.distinct_conjugation_automorphisms, 168);
        assert_eq!(report.expected_gl_3_2_order, 168);
    }

    #[test]
    fn complete_scan_classifies_every_r8_coset() {
        let report = complete_garden_scan();
        assert!(report.passed);
        assert_eq!(report.cosets_scanned, 5_040);
        assert_eq!(report.signable_cosets, 5_040);
        assert_eq!(report.unsignable_cosets, 0);
        assert_eq!(report.rank_histogram.len(), 1);
        assert_eq!(report.rank_histogram[0].value, 45);
        assert_eq!(report.rank_histogram[0].octets, 5_040);
        assert_eq!(report.nullity_histogram[0].value, 19);
        assert_eq!(report.solution_count_per_coset, 1 << 19);
        assert_eq!(report.contingency.abnormal_and_signable, 168);
        assert_eq!(report.contingency.ordinary_and_signable, 4_872);
        assert_eq!(report.dense_residual_entries, 0);
        assert_eq!(report.named_octets.len(), 7);
        assert!(
            report
                .named_octets
                .iter()
                .all(|item| item.garden_signing_exists)
        );
    }
}
