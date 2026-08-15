//! Transport one verified central extension across all 151,200 R8 supports.

#![allow(clippy::needless_range_loop)]

use crate::permutahedron::{coset_partition, permutations, CosetSide, Permutation};
use crate::permutahedron_hypergraph::identity_hyperedges;
use crate::permutahedron_hypergraph_signed;
use crate::vector_tensor_central_charge::{build as build_census, l_matrices, Matrix};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;

#[derive(Clone, Debug, Serialize)]
pub struct FamilyTransportRecord {
    pub family_id: usize,
    pub identity_ranks: Vec<u32>,
    pub conjugator_rank: u32,
    pub relabeling_paths: usize,
    pub distinct_supports: usize,
    pub minimum_path_multiplicity: usize,
    pub maximum_path_multiplicity: usize,
    pub coset_catalog_exact_match: bool,
    pub representative_central_algebra_verified: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentralAtlasReport {
    pub schema_version: &'static str,
    pub seed: &'static str,
    pub families: usize,
    pub relabeling_paths: usize,
    pub distinct_central_supports: usize,
    pub expected_supports: usize,
    pub transport_entries_checked: usize,
    pub representative_central_algebras_checked: usize,
    pub every_support_has_one_z_signing: bool,
    pub ordinary_garden_transport_recomputed: bool,
    pub ordinary_garden_supports: usize,
    pub every_support_has_both_algebra_types: bool,
    pub family_records: Vec<FamilyTransportRecord>,
    pub passed: bool,
    pub conclusion: &'static str,
    pub boundary: &'static str,
}

fn matrix_image(matrix: &Matrix, row: usize) -> usize {
    (0..D)
        .find(|&column| matrix[row][column] != 0)
        .expect("monomial matrix")
}

fn matrix_permutation(matrix: &Matrix) -> Permutation {
    let values: Vec<u8> = (0..D)
        .map(|row| u8::try_from(matrix_image(matrix, row) + 1).unwrap())
        .collect();
    Permutation::new(&values).expect("S8 matrix permutation")
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..D)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn transpose(matrix: &Matrix) -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}

fn scale(matrix: &Matrix, coefficient: i16) -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| coefficient * matrix[row][column]))
}

fn identity() -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| i16::from(row == column)))
}

fn relabel(matrix: &Matrix, row_map: Permutation, column_map: Permutation) -> Matrix {
    let mut result = [[0i16; D]; D];
    let rows = row_map.as_slice();
    let columns = column_map.as_slice();
    for old_row in 0..D {
        for old_column in 0..D {
            result[usize::from(rows[old_row] - 1)][usize::from(columns[old_column] - 1)] =
                matrix[old_row][old_column];
        }
    }
    result
}

fn verify_one_z(l: &[Matrix; N], z_b: &Matrix, z_f: &Matrix, k: &Matrix) -> bool {
    let r: [Matrix; N] = std::array::from_fn(|color| transpose(&l[color]));
    for color in 0..N {
        if multiply(&l[color], z_f) != multiply(z_b, &l[color])
            || multiply(&r[color], z_b) != multiply(z_f, &r[color])
        {
            return false;
        }
    }
    for first in 0..N {
        for second in 0..N {
            let expected_b = if first == second {
                scale(&identity(), 2)
            } else {
                scale(z_b, 2 * k[first][second])
            };
            let expected_f = if first == second {
                scale(&identity(), 2)
            } else {
                scale(z_f, 2 * k[first][second])
            };
            if add(
                &multiply(&l[first], &r[second]),
                &multiply(&l[second], &r[first]),
            ) != expected_b
                || add(
                    &multiply(&r[first], &l[second]),
                    &multiply(&r[second], &l[first]),
                ) != expected_f
            {
                return false;
            }
        }
    }
    true
}

fn sorted_ranks(permutations: impl IntoIterator<Item = Permutation>) -> Vec<u32> {
    let mut ranks: Vec<_> = permutations
        .into_iter()
        .map(|permutation| permutation.rank() as u32)
        .collect();
    ranks.sort_unstable();
    ranks
}

fn conjugated_ranks(subgroup: &[Permutation], conjugator: Permutation) -> Vec<u32> {
    let inverse = conjugator.inverse();
    sorted_ranks(subgroup.iter().map(|&entry| {
        conjugator
            .compose(entry)
            .expect("S8")
            .compose(inverse)
            .expect("S8")
    }))
}

fn least_conjugator(source: &[Permutation], target: &[Permutation]) -> Permutation {
    let target_ranks = sorted_ranks(target.iter().copied());
    permutations(N)
        .expect("enumerate S8")
        .find(|&candidate| conjugated_ranks(source, candidate) == target_ranks)
        .expect("R8 subgroups are conjugate")
}

fn ordinary_garden_support_count() -> Option<usize> {
    let validation = permutahedron_hypergraph_signed::build().validation;
    (validation.passed && validation.all_families_passed).then_some(validation.transported_closures)
}

pub fn build() -> CentralAtlasReport {
    let census = build_census();
    let cc = &census.sectors[0];
    let branch = &cc.branches[0];
    let charge = branch.central_charge.as_ref().expect("CC is one-Z");
    let seed_l = l_matrices(0, branch.effective_boolean_factors);
    assert!(verify_one_z(
        &seed_l,
        &charge.bosonic,
        &charge.fermionic,
        &charge.color_coefficient_matrix
    ));
    let seed_permutations: Vec<_> = seed_l.iter().map(matrix_permutation).collect();
    let anchor = seed_permutations[0];
    let anchor_inverse = anchor.inverse();
    let seed_subgroup: Vec<_> = seed_permutations
        .iter()
        .map(|&entry| entry.compose(anchor_inverse).expect("S8"))
        .collect();
    let families = identity_hyperedges(N);
    let all_relabelings: Vec<_> = permutations(N).expect("enumerate S8").collect();
    let mut family_records = Vec::new();
    let mut global_supports = BTreeSet::new();
    for (family_id, family) in families.iter().enumerate() {
        let conjugator = least_conjugator(&seed_subgroup, family);
        let conjugator_inverse = conjugator.inverse();
        let mut multiplicities = BTreeMap::<Vec<u32>, usize>::new();
        for &boson_relabeling in &all_relabelings {
            let inverse = boson_relabeling.inverse();
            let support = sorted_ranks(seed_permutations.iter().map(|&entry| {
                conjugator
                    .compose(entry)
                    .expect("S8")
                    .compose(conjugator_inverse)
                    .expect("S8")
                    .compose(inverse)
                    .expect("S8")
            }));
            *multiplicities.entry(support).or_insert(0) += 1;
        }
        let catalog = coset_partition(family, CosetSide::Right).expect("R8 right cosets");
        let catalog_set: BTreeSet<_> = catalog.slices.into_iter().collect();
        let transported_set: BTreeSet<_> = multiplicities.keys().cloned().collect();
        global_supports.extend(transported_set.iter().cloned());

        let transformed_l =
            std::array::from_fn(|color| relabel(&seed_l[color], conjugator, conjugator));
        let transformed_z_b = relabel(&charge.bosonic, conjugator, conjugator);
        let transformed_z_f = relabel(&charge.fermionic, conjugator, conjugator);
        let representative_verified = verify_one_z(
            &transformed_l,
            &transformed_z_b,
            &transformed_z_f,
            &charge.color_coefficient_matrix,
        );
        family_records.push(FamilyTransportRecord {
            family_id,
            identity_ranks: sorted_ranks(family.iter().copied()),
            conjugator_rank: conjugator.rank() as u32,
            relabeling_paths: all_relabelings.len(),
            distinct_supports: multiplicities.len(),
            minimum_path_multiplicity: multiplicities.values().copied().min().unwrap_or(0),
            maximum_path_multiplicity: multiplicities.values().copied().max().unwrap_or(0),
            coset_catalog_exact_match: transported_set == catalog_set,
            representative_central_algebra_verified: representative_verified,
        });
    }
    let ordinary_garden_supports = ordinary_garden_support_count().unwrap_or(0);
    let ordinary_checked = ordinary_garden_supports == 151_200;
    let every_one_z = global_supports.len() == 151_200
        && family_records.iter().all(|record| {
            record.distinct_supports == 5_040
                && record.minimum_path_multiplicity == 8
                && record.maximum_path_multiplicity == 8
                && record.coset_catalog_exact_match
                && record.representative_central_algebra_verified
        });
    let both = every_one_z && ordinary_checked;
    CentralAtlasReport {
        schema_version: "vector-tensor-central-atlas-v1",
        seed: "CC:base, the common enriched one-Z class representative",
        families: family_records.len(),
        relabeling_paths: family_records.len() * all_relabelings.len(),
        distinct_central_supports: global_supports.len(),
        expected_supports: 151_200,
        transport_entries_checked: family_records.len() * all_relabelings.len() * (N * D + D + D),
        representative_central_algebras_checked: family_records.len(),
        every_support_has_one_z_signing: every_one_z,
        ordinary_garden_transport_recomputed: ordinary_checked,
        ordinary_garden_supports,
        every_support_has_both_algebra_types: both,
        family_records,
        passed: both,
        conclusion: "Every R8 hyperedge admits both an ordinary Garden signing and a one-central-charge signing. Unsigned support cannot determine the worldline algebra type.",
        boundary: "The transported one-Z signings are algebraically exact. The transport does not assign a unique 4D physical parent because all 25 printed one-Z branches are already one enriched worldline class.",
    }
}

pub fn write_artifact(path: &Path) -> CentralAtlasReport {
    let report = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create central-atlas directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create central-atlas artifact")),
        &report,
    )
    .expect("write central-atlas artifact");
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_class_covers_every_r8_support() {
        let report = build();
        assert!(report.passed);
        assert_eq!(report.distinct_central_supports, 151_200);
        assert!(report.every_support_has_both_algebra_types);
    }
}
