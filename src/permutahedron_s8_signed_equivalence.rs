//! Relative-color scan and exact signed equivalence for the S8 recursion.
//!
//! The scan extends the literal Boolean recursion by all 24 relative orders
//! of the second four-color input.  Equivalence is then tested by explicit
//! witnesses.  The fixed-color nodal relation is the N=8 form of
//! L'_I = X L_I Y with signed permutation matrices X and Y.  Broader
//! quotients are reported separately rather than folded into that relation.

#![allow(clippy::needless_range_loop)] // fixed-size color and node maps are indexed explicitly

use crate::decompose::{antisymmetric_commutant_dim, commutant_dim};
use crate::lr_matrix::AdinkraRep;
use crate::permutahedron;
use crate::permutahedron_s8_signed_recursion::{
    S4_RECURSION_LABELS, aligned_recursive_boolean_factors, aligned_recursive_permutations,
    build_rep, cyclic_mask, exact_published_match, matrix_checksum,
};
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s8-signed-equivalence-v1";
const N: usize = 8;
const D: usize = 8;

#[derive(Debug, Clone, Serialize)]
pub struct SourceRecord {
    pub source: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlignmentScanRecord {
    pub first_label: &'static str,
    pub second_label: &'static str,
    pub second_color_order_one_based: [usize; 4],
    pub candidates_checked: usize,
    pub closing_start_positions_one_based: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClosingRecord {
    pub id: String,
    pub first_label: &'static str,
    pub second_label: &'static str,
    pub first_parentage: &'static str,
    pub second_parentage: &'static str,
    pub second_color_order_one_based: [usize; 4],
    pub start_position_one_based: usize,
    pub flip_mask_decimal: u8,
    pub boolean_factors: [u8; 8],
    pub permutations: [[u8; 8]; 8],
    pub exact_published_system_match: Option<&'static str>,
    pub matrix_checksum_fnv1a64: String,
    pub commutant_dimension: usize,
    pub antisymmetric_commutant_dimension: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EquivalencePolicy {
    pub fixed_color: bool,
    pub allow_supercharge_signs: bool,
    pub allow_color_permutation: bool,
    pub allow_boson_fermion_duality: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EquivalenceWitness {
    pub source_dualized: bool,
    pub boson_map_zero_based: [usize; D],
    pub fermion_map_zero_based: [usize; D],
    pub color_map_zero_based: [usize; N],
    pub boson_switches: [i8; D],
    pub fermion_switches: [i8; D],
    pub supercharge_signs: [i8; N],
    pub verified_on_all_64_edges: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassMemberRecord {
    pub id: String,
    pub witness_from_representative: EquivalenceWitness,
}

#[derive(Debug, Clone, Serialize)]
pub struct EquivalenceClassRecord {
    pub class_id: String,
    pub representative_id: String,
    pub members: Vec<ClassMemberRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EquivalenceLayerRecord {
    pub name: &'static str,
    pub interpretation: &'static str,
    pub policy: EquivalencePolicy,
    pub classes: Vec<EquivalenceClassRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub ordered_distinct_pairs: usize,
    pub relative_color_orders_per_pair: usize,
    pub cyclic_masks_per_order: usize,
    pub candidates_checked: usize,
    pub closing_candidates: usize,
    pub closing_pair_alignment_configurations: usize,
    pub ct_exact_anchor_recovered: bool,
    pub cv_exact_anchor_recovered: bool,
    pub all_closers_have_scalar_commutant: bool,
    pub every_serialized_witness_verified: bool,
    pub equivalence_class_counts_by_layer: Vec<usize>,
    pub all_closers_share_one_fixed_color_nodal_class: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignedEquivalenceArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub scan: Vec<AlignmentScanRecord>,
    pub closers: Vec<ClosingRecord>,
    pub equivalence_layers: Vec<EquivalenceLayerRecord>,
    pub validation: ValidationRecord,
    pub boundary: &'static str,
}

fn parentage(index: usize) -> &'static str {
    if index < 3 {
        "Carroll reduction of a named four-dimensional N=1 multiplet"
    } else {
        "Garden-algebra solution with no four-dimensional parent stated in the source"
    }
}

fn color_orders_4() -> Vec<[usize; 4]> {
    permutahedron::permutations(4)
        .expect("enumerate S4")
        .map(|permutation| {
            permutation
                .as_slice()
                .iter()
                .map(|entry| usize::from(entry - 1))
                .collect::<Vec<_>>()
                .try_into()
                .expect("four entries")
        })
        .collect()
}

fn color_orders_8() -> Vec<[usize; 8]> {
    permutahedron::permutations(8)
        .expect("enumerate S8")
        .map(|permutation| {
            permutation
                .as_slice()
                .iter()
                .map(|entry| usize::from(entry - 1))
                .collect::<Vec<_>>()
                .try_into()
                .expect("eight entries")
        })
        .collect()
}

fn dual(rep: &AdinkraRep) -> AdinkraRep {
    AdinkraRep {
        n: rep.n,
        d: rep.d,
        l_matrices: rep
            .l_matrices
            .iter()
            .map(|matrix| matrix.inverse())
            .collect(),
    }
}

/// Solve an affine GF(2) system and return one solution with free variables zero.
fn solve_gf2(rows: &mut [(u32, u8)], variables: usize) -> Option<Vec<u8>> {
    let mut pivot_row = 0usize;
    let mut pivot_for_column = vec![None; variables];
    for column in 0..variables {
        let pivot = (pivot_row..rows.len()).find(|&row| rows[row].0 & (1 << column) != 0);
        let Some(pivot) = pivot else { continue };
        rows.swap(pivot_row, pivot);
        let pivot_value = rows[pivot_row];
        for row in 0..rows.len() {
            if row != pivot_row && rows[row].0 & (1 << column) != 0 {
                rows[row].0 ^= pivot_value.0;
                rows[row].1 ^= pivot_value.1;
            }
        }
        pivot_for_column[column] = Some(pivot_row);
        pivot_row += 1;
    }
    if rows.iter().any(|&(mask, rhs)| mask == 0 && rhs != 0) {
        return None;
    }
    let mut solution = vec![0u8; variables];
    for column in (0..variables).rev() {
        if let Some(row) = pivot_for_column[column] {
            let (mask, rhs) = rows[row];
            let other = (0..variables)
                .filter(|&index| index != column && mask & (1 << index) != 0)
                .fold(0u8, |sum, index| sum ^ solution[index]);
            solution[column] = rhs ^ other;
        }
    }
    Some(solution)
}

fn verify_witness(source: &AdinkraRep, target: &AdinkraRep, witness: &EquivalenceWitness) -> bool {
    let source = if witness.source_dualized {
        dual(source)
    } else {
        source.clone()
    };
    for color in 0..N {
        let target_color = witness.color_map_zero_based[color];
        for boson in 0..D {
            let source_fermion = usize::from(source.l_matrices[color].perm[boson]);
            let target_boson = witness.boson_map_zero_based[boson];
            let target_fermion = usize::from(target.l_matrices[target_color].perm[target_boson]);
            if witness.fermion_map_zero_based[source_fermion] != target_fermion {
                return false;
            }
            let transformed_sign = source.l_matrices[color].sign[boson]
                * witness.boson_switches[boson]
                * witness.fermion_switches[source_fermion]
                * witness.supercharge_signs[color];
            if transformed_sign != target.l_matrices[target_color].sign[target_boson] {
                return false;
            }
        }
    }
    true
}

fn witness_for_color_map(
    source: &AdinkraRep,
    target: &AdinkraRep,
    color_map: [usize; N],
    target_root: usize,
    allow_supercharge_signs: bool,
    source_dualized: bool,
) -> Option<EquivalenceWitness> {
    let mut fermion_map = [usize::MAX; D];
    for color in 0..N {
        let source_fermion = usize::from(source.l_matrices[color].perm[0]);
        let target_fermion = usize::from(target.l_matrices[color_map[color]].perm[target_root]);
        if fermion_map[source_fermion] != usize::MAX
            && fermion_map[source_fermion] != target_fermion
        {
            return None;
        }
        fermion_map[source_fermion] = target_fermion;
    }
    if fermion_map.contains(&usize::MAX) {
        return None;
    }
    let mut seen_fermions = [false; D];
    for &image in &fermion_map {
        if seen_fermions[image] {
            return None;
        }
        seen_fermions[image] = true;
    }

    let mut inverse_target = [[usize::MAX; D]; N];
    for color in 0..N {
        for boson in 0..D {
            inverse_target[color][usize::from(target.l_matrices[color].perm[boson])] = boson;
        }
    }
    let mut boson_map = [usize::MAX; D];
    boson_map[0] = target_root;
    for boson in 1..D {
        let source_fermion = usize::from(source.l_matrices[0].perm[boson]);
        let candidate = inverse_target[color_map[0]][fermion_map[source_fermion]];
        if (1..N).any(|color| {
            let source_fermion = usize::from(source.l_matrices[color].perm[boson]);
            inverse_target[color_map[color]][fermion_map[source_fermion]] != candidate
        }) {
            return None;
        }
        boson_map[boson] = candidate;
    }
    let mut seen_bosons = [false; D];
    for &image in &boson_map {
        if image >= D || seen_bosons[image] {
            return None;
        }
        seen_bosons[image] = true;
    }

    let variables = 2 * D + if allow_supercharge_signs { N } else { 0 };
    let mut equations = Vec::with_capacity(N * D);
    for color in 0..N {
        for boson in 0..D {
            let source_fermion = usize::from(source.l_matrices[color].perm[boson]);
            let target_boson = boson_map[boson];
            if fermion_map[source_fermion]
                != usize::from(target.l_matrices[color_map[color]].perm[target_boson])
            {
                return None;
            }
            let mut coefficients = (1u32 << boson) | (1u32 << (D + source_fermion));
            if allow_supercharge_signs {
                coefficients |= 1u32 << (2 * D + color);
            }
            let rhs = u8::from(
                source.l_matrices[color].sign[boson]
                    != target.l_matrices[color_map[color]].sign[target_boson],
            );
            equations.push((coefficients, rhs));
        }
    }
    let solution = solve_gf2(&mut equations, variables)?;
    let to_sign = |bit: u8| if bit == 0 { 1 } else { -1 };
    let mut witness = EquivalenceWitness {
        source_dualized,
        boson_map_zero_based: boson_map,
        fermion_map_zero_based: fermion_map,
        color_map_zero_based: color_map,
        boson_switches: std::array::from_fn(|index| to_sign(solution[index])),
        fermion_switches: std::array::from_fn(|index| to_sign(solution[D + index])),
        supercharge_signs: std::array::from_fn(|index| {
            if allow_supercharge_signs {
                to_sign(solution[2 * D + index])
            } else {
                1
            }
        }),
        verified_on_all_64_edges: false,
    };
    let mut transformed_check = witness.clone();
    transformed_check.source_dualized = false;
    witness.verified_on_all_64_edges = verify_witness(source, target, &transformed_check);
    witness.verified_on_all_64_edges.then_some(witness)
}

fn find_witness(
    source: &AdinkraRep,
    target: &AdinkraRep,
    policy: EquivalencePolicy,
    all_color_maps: &[[usize; N]],
) -> Option<EquivalenceWitness> {
    for source_dualized in [false, true] {
        if source_dualized && !policy.allow_boson_fermion_duality {
            continue;
        }
        let transformed = if source_dualized {
            dual(source)
        } else {
            source.clone()
        };
        let identity = std::array::from_fn(|index| index);
        let color_maps: &[[usize; N]] = if policy.allow_color_permutation {
            all_color_maps
        } else {
            std::slice::from_ref(&identity)
        };
        for &color_map in color_maps {
            for target_root in 0..D {
                if let Some(witness) = witness_for_color_map(
                    &transformed,
                    target,
                    color_map,
                    target_root,
                    policy.allow_supercharge_signs,
                    source_dualized,
                ) {
                    return Some(witness);
                }
            }
        }
    }
    None
}

fn classify(
    name: &'static str,
    interpretation: &'static str,
    policy: EquivalencePolicy,
    records: &[ClosingRecord],
    reps: &[AdinkraRep],
    all_color_maps: &[[usize; N]],
) -> EquivalenceLayerRecord {
    let mut class_indices: Vec<Vec<(usize, EquivalenceWitness)>> = Vec::new();
    for member in 0..reps.len() {
        let mut assigned = false;
        for class in &mut class_indices {
            let representative = class[0].0;
            if let Some(witness) =
                find_witness(&reps[representative], &reps[member], policy, all_color_maps)
            {
                class.push((member, witness));
                assigned = true;
                break;
            }
        }
        if !assigned {
            let witness = find_witness(&reps[member], &reps[member], policy, all_color_maps)
                .expect("reflexive signed equivalence");
            class_indices.push(vec![(member, witness)]);
        }
    }
    let classes = class_indices
        .into_iter()
        .map(|mut class| {
            class.sort_by(|left, right| records[left.0].id.cmp(&records[right.0].id));
            let representative_id = records[class[0].0].id.clone();
            EquivalenceClassRecord {
                class_id: representative_id.clone(),
                representative_id,
                members: class
                    .into_iter()
                    .map(|(index, witness)| ClassMemberRecord {
                        id: records[index].id.clone(),
                        witness_from_representative: witness,
                    })
                    .collect(),
            }
        })
        .collect();
    EquivalenceLayerRecord {
        name,
        interpretation,
        policy,
        classes,
    }
}

pub fn build() -> SignedEquivalenceArtifact {
    let alignments = color_orders_4();
    let all_color_maps = color_orders_8();
    let mut scan = Vec::with_capacity(30 * 24);
    let mut closers = Vec::new();
    let mut reps = Vec::new();
    let mut candidates_checked = 0usize;

    for first in 0..6 {
        for second in 0..6 {
            if first == second {
                continue;
            }
            for &alignment in &alignments {
                let permutations = aligned_recursive_permutations(first, second, alignment);
                let mut closing_starts = Vec::new();
                for start in 0..8 {
                    candidates_checked += 1;
                    let (mask, _) = cyclic_mask(start);
                    let factors = aligned_recursive_boolean_factors(first, second, alignment, mask);
                    let rep = build_rep(&permutations, &factors);
                    if !rep.verify_garden_algebra() {
                        continue;
                    }
                    closing_starts.push(start + 1);
                    let alignment_one_based = alignment.map(|color| color + 1);
                    let id = format!(
                        "{}->{}:order-{}{}{}{}:mask-{}",
                        S4_RECURSION_LABELS[first],
                        S4_RECURSION_LABELS[second],
                        alignment_one_based[0],
                        alignment_one_based[1],
                        alignment_one_based[2],
                        alignment_one_based[3],
                        start + 1,
                    );
                    closers.push(ClosingRecord {
                        id,
                        first_label: S4_RECURSION_LABELS[first],
                        second_label: S4_RECURSION_LABELS[second],
                        first_parentage: parentage(first),
                        second_parentage: parentage(second),
                        second_color_order_one_based: alignment_one_based,
                        start_position_one_based: start + 1,
                        flip_mask_decimal: mask,
                        boolean_factors: factors,
                        permutations,
                        exact_published_system_match: exact_published_match(
                            &permutations,
                            &factors,
                        ),
                        matrix_checksum_fnv1a64: matrix_checksum(&rep),
                        commutant_dimension: commutant_dim(&rep),
                        antisymmetric_commutant_dimension: antisymmetric_commutant_dim(&rep),
                    });
                    reps.push(rep);
                }
                scan.push(AlignmentScanRecord {
                    first_label: S4_RECURSION_LABELS[first],
                    second_label: S4_RECURSION_LABELS[second],
                    second_color_order_one_based: alignment.map(|color| color + 1),
                    candidates_checked: 8,
                    closing_start_positions_one_based: closing_starts,
                });
            }
        }
    }

    let policies = [
        (
            "fixed-color nodal BC8",
            "signed boson and fermion node permutations with color labels fixed",
            EquivalencePolicy {
                fixed_color: true,
                allow_supercharge_signs: false,
                allow_color_permutation: false,
                allow_boson_fermion_duality: false,
            },
        ),
        (
            "fixed-color nodal BC8 plus supercharge signs",
            "the nodal relation enlarged by independent signs on the eight supercharges",
            EquivalencePolicy {
                fixed_color: true,
                allow_supercharge_signs: true,
                allow_color_permutation: false,
                allow_boson_fermion_duality: false,
            },
        ),
        (
            "unlabeled-color signed graph",
            "nodal equivalence, supercharge signs, and all permutations of the eight colors",
            EquivalencePolicy {
                fixed_color: false,
                allow_supercharge_signs: true,
                allow_color_permutation: true,
                allow_boson_fermion_duality: false,
            },
        ),
        (
            "unlabeled-color signed graph plus duality",
            "the preceding combinatorial quotient enlarged by interchange of boson and fermion levels",
            EquivalencePolicy {
                fixed_color: false,
                allow_supercharge_signs: true,
                allow_color_permutation: true,
                allow_boson_fermion_duality: true,
            },
        ),
    ];
    let equivalence_layers: Vec<_> = policies
        .into_iter()
        .map(|(name, interpretation, policy)| {
            classify(
                name,
                interpretation,
                policy,
                &closers,
                &reps,
                &all_color_maps,
            )
        })
        .collect();

    let ct_anchor = closers.iter().any(|record| {
        record.first_label == "CM"
            && record.second_label == "TM"
            && record.second_color_order_one_based == [1, 2, 3, 4]
            && record.start_position_one_based == 6
            && record.exact_published_system_match == Some("CT")
    });
    let cv_anchor = closers.iter().any(|record| {
        record.first_label == "CM"
            && record.second_label == "VM"
            && record.second_color_order_one_based == [2, 1, 4, 3]
            && record.start_position_one_based == 4
            && record.exact_published_system_match == Some("CV")
    });
    let scalar = closers.iter().all(|record| {
        record.commutant_dimension == 1 && record.antisymmetric_commutant_dimension == 0
    });
    let witnesses_verified = equivalence_layers.iter().all(|layer| {
        layer.classes.iter().all(|class| {
            class
                .members
                .iter()
                .all(|member| member.witness_from_representative.verified_on_all_64_edges)
        })
    });
    let class_counts: Vec<_> = equivalence_layers
        .iter()
        .map(|layer| layer.classes.len())
        .collect();
    let one_fixed_color_class = equivalence_layers[0].classes.len() == 1
        && equivalence_layers[0].classes[0].members.len() == closers.len();
    let configurations = scan
        .iter()
        .filter(|record| !record.closing_start_positions_one_based.is_empty())
        .count();
    let passed = candidates_checked == 5_760
        && closers.len() == 24
        && configurations == 12
        && ct_anchor
        && cv_anchor
        && scalar
        && witnesses_verified
        && one_fixed_color_class;

    SignedEquivalenceArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Relative-color completion and exact signed equivalence of the S8 recursion",
        sources: vec![
            SourceRecord {
                source: "arXiv:2304.09830",
                locator: "Eqs. (2.17)-(2.25) and Sec. 2.2",
                role: "unsigned block recursion, Boolean factors, cyclic flips, and Garden acceptance",
            },
            SourceRecord {
                source: "arXiv:2012.13308",
                locator: "discussion surrounding the twenty-four color orderings",
                role: "the color order is retained as explicit data rather than assumed immaterial",
            },
            SourceRecord {
                source: "arXiv:1712.07826",
                locator: "Eq. (2.33) and the surrounding discussion of node flips and flops",
                role: "fixed-color signed nodal equivalence",
            },
            SourceRecord {
                source: "HowardTLK.v2.pdf",
                locator: "pp. 61, 64, 73, 77, 80-81",
                role: "the six four-color quartets, permutahedron geometry, hopping operators, and the intended four-color base case",
            },
        ],
        scan,
        closers,
        equivalence_layers,
        validation: ValidationRecord {
            ordered_distinct_pairs: 30,
            relative_color_orders_per_pair: 24,
            cyclic_masks_per_order: 8,
            candidates_checked,
            closing_candidates: reps.len(),
            closing_pair_alignment_configurations: configurations,
            ct_exact_anchor_recovered: ct_anchor,
            cv_exact_anchor_recovered: cv_anchor,
            all_closers_have_scalar_commutant: scalar,
            every_serialized_witness_verified: witnesses_verified,
            equivalence_class_counts_by_layer: class_counts,
            all_closers_share_one_fixed_color_nodal_class: one_fixed_color_class,
            passed,
        },
        boundary: "The class IDs are canonical only within this finite scan. Fixed-color nodal equivalence, supercharge signs, color permutations, and boson-fermion duality are reported as separate layers. No layer is identified here with complete four-dimensional physical equivalence or enhancement.",
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ValidationRecord {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create signed equivalence data")),
        &artifact,
    )
    .expect("write signed equivalence data");
    serde_json::to_writer_pretty(
        BufWriter::new(
            File::create(validation_path).expect("create signed equivalence validation"),
        ),
        &artifact.validation,
    )
    .expect("write signed equivalence validation");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permutahedron_fixtures::{RepresentationOctet, S8_REPRESENTATION_OCTETS};
    use crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS;

    fn published_rep(index: usize) -> AdinkraRep {
        let RepresentationOctet { permutations, .. } = S8_REPRESENTATION_OCTETS[index];
        build_rep(&permutations, &S8_BASE_BOOLEAN_FACTORS[index])
    }

    #[test]
    fn gf2_solver_accepts_consistent_and_rejects_inconsistent_systems() {
        let mut consistent = [(0b0011, 1), (0b0110, 0)];
        let solution = solve_gf2(&mut consistent, 3).expect("solution");
        assert_eq!(solution[0] ^ solution[1], 1);
        assert_eq!(solution[1] ^ solution[2], 0);
        let mut inconsistent = [(0b0001, 0), (0b0001, 1)];
        assert!(solve_gf2(&mut inconsistent, 1).is_none());
    }

    #[test]
    fn exact_witness_is_reflexive_and_mutation_is_rejected() {
        let rep = published_rep(1);
        let colors = color_orders_8();
        let policy = EquivalencePolicy {
            fixed_color: true,
            allow_supercharge_signs: false,
            allow_color_permutation: false,
            allow_boson_fermion_duality: false,
        };
        let witness = find_witness(&rep, &rep, policy, &colors).expect("reflexive witness");
        assert!(verify_witness(&rep, &rep, &witness));
        let mut bad = witness;
        bad.boson_switches[0] *= -1;
        assert!(!verify_witness(&rep, &rep, &bad));
    }

    #[test]
    fn duality_witness_is_checked_against_the_original_source() {
        let rep = published_rep(1);
        let transformed = dual(&rep);
        let identity = std::array::from_fn(|index| index);
        let witness = (0..D)
            .find_map(|root| witness_for_color_map(&transformed, &rep, identity, root, true, true))
            .expect("duality witness");
        assert!(witness.source_dualized);
        assert!(verify_witness(&rep, &rep, &witness));
    }

    #[test]
    fn full_scan_recovers_ct_and_cv_with_expected_relative_orders() {
        let artifact = build();
        assert_eq!(artifact.validation.candidates_checked, 5_760);
        assert_eq!(artifact.validation.closing_candidates, 24);
        assert_eq!(
            artifact.validation.closing_pair_alignment_configurations,
            12
        );
        assert!(artifact.validation.ct_exact_anchor_recovered);
        assert!(artifact.validation.cv_exact_anchor_recovered);
        assert!(artifact.validation.every_serialized_witness_verified);
        assert_eq!(
            artifact.validation.equivalence_class_counts_by_layer,
            [1, 1, 1, 1]
        );
        assert!(
            artifact
                .validation
                .all_closers_share_one_fixed_color_nodal_class
        );
        assert!(artifact.validation.passed);
    }
}
