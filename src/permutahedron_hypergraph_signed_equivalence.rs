//! Signed-equivalence ledger for the 30 discovered S8 identity octets.
//!
//! Each unsigned identity block receives one deterministic Garden signing.
//! Those representatives are then classified under four explicitly separated
//! equivalence relations.  Every membership claim carries a transformation
//! witness checked directly on all 64 colored edges.

#![allow(clippy::needless_range_loop)]

use crate::lr_matrix::AdinkraRep;
use crate::permutahedron::{permutations, Permutation};
use crate::permutahedron_garden::solve_garden_signing;
use crate::permutahedron_hypergraph::identity_hyperedges;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;
const SCHEMA_VERSION: &str = "permutahedron-hypergraph-signed-equivalence-v1";

#[derive(Debug, Clone, Serialize)]
pub struct RepresentativeRecord {
    pub id: String,
    pub family_id: usize,
    pub identity_block_ranks: Vec<u32>,
    pub solver_selected_sign_mask: String,
    pub representation_sha256: String,
    pub garden_verified: bool,
    pub garden_nullity: usize,
    pub gauge_action_rank: usize,
    pub gauge_generators_verified: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EquivalencePolicy {
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
    pub representative_id: String,
    pub witness_from_class_representative: EquivalenceWitness,
}

#[derive(Debug, Clone, Serialize)]
pub struct EquivalenceClassRecord {
    pub class_id: String,
    pub class_representative_id: String,
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
pub struct SignedEquivalenceValidation {
    pub representatives: usize,
    pub all_representatives_close: bool,
    pub equivalence_class_counts_by_layer: Vec<usize>,
    pub every_serialized_witness_verified: bool,
    pub mutated_witness_rejected: bool,
    pub one_unlabeled_color_signed_class: bool,
    pub one_unlabeled_color_signed_class_with_duality: bool,
    pub garden_nullity_histogram: BTreeMap<usize, usize>,
    pub gauge_action_rank_histogram: BTreeMap<usize, usize>,
    pub all_gauge_generators_verified: bool,
    pub gauge_action_spans_every_garden_signing_space: bool,
    pub identity_support_labeled_signings_classified: u64,
    pub all_hyperedge_labeled_signings_classified: u64,
    pub all_labeled_signings_share_one_unlabeled_color_signed_class: bool,
    pub canonical_sign_mask_used_as_invariant: bool,
    pub one_dimensional_signed_data_break_the_thirty_fold_symmetry: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignedEquivalenceArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub method: Vec<&'static str>,
    pub representatives: Vec<RepresentativeRecord>,
    pub equivalence_layers: Vec<EquivalenceLayerRecord>,
    pub validation: SignedEquivalenceValidation,
    pub findings: Vec<String>,
    pub boundary: &'static str,
}

fn zero_based(permutation: Permutation) -> Vec<usize> {
    permutation
        .as_slice()
        .iter()
        .map(|&value| usize::from(value - 1))
        .collect()
}

fn build_rep(block: &[Permutation], signs: &[i8]) -> AdinkraRep {
    let color_permutations: Vec<Vec<usize>> = block.iter().copied().map(zero_based).collect();
    AdinkraRep::from_parts(N, D, &color_permutations, signs)
}

fn sign_mask(signs: &[i8]) -> String {
    let mut mask = 0u64;
    for (index, &sign) in signs.iter().enumerate() {
        if sign == -1 {
            mask |= 1u64 << index;
        }
    }
    format!("{mask:016x}")
}

fn representation_checksum(rep: &AdinkraRep) -> String {
    let mut digest = Sha256::new();
    for matrix in &rep.l_matrices {
        for &image in &matrix.perm {
            digest.update(image.to_le_bytes());
        }
        for &sign in &matrix.sign {
            digest.update(sign.to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn histogram(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut result = BTreeMap::new();
    for value in values {
        *result.entry(value).or_insert(0) += 1;
    }
    result
}

fn gauge_generators(rep: &AdinkraRep) -> Vec<u64> {
    let mut generators = Vec::with_capacity(2 * D + N);
    for boson in 0..D {
        let mut mask = 0u64;
        for color in 0..N {
            mask |= 1u64 << (color * D + boson);
        }
        generators.push(mask);
    }
    for fermion in 0..D {
        let mut mask = 0u64;
        for color in 0..N {
            for boson in 0..D {
                if usize::from(rep.l_matrices[color].perm[boson]) == fermion {
                    mask |= 1u64 << (color * D + boson);
                }
            }
        }
        generators.push(mask);
    }
    for color in 0..N {
        let mut mask = 0u64;
        for boson in 0..D {
            mask |= 1u64 << (color * D + boson);
        }
        generators.push(mask);
    }
    generators
}

fn gf2_rank_u64(vectors: &[u64]) -> usize {
    let mut basis = [0u64; 64];
    let mut rank = 0usize;
    for &input in vectors {
        let mut vector = input;
        while vector != 0 {
            let pivot = vector.ilog2() as usize;
            if basis[pivot] == 0 {
                basis[pivot] = vector;
                rank += 1;
                break;
            }
            vector ^= basis[pivot];
        }
    }
    rank
}

fn toggled_signs(signs: &[i8], mask: u64) -> Vec<i8> {
    signs
        .iter()
        .enumerate()
        .map(|(index, &sign)| {
            if mask & (1u64 << index) == 0 {
                sign
            } else {
                -sign
            }
        })
        .collect()
}

fn color_orders() -> Vec<[usize; N]> {
    permutations(N)
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
    records: &[RepresentativeRecord],
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
        .enumerate()
        .map(|(class_index, class)| {
            let representative = class[0].0;
            EquivalenceClassRecord {
                class_id: format!("{name}/class-{class_index:02}"),
                class_representative_id: records[representative].id.clone(),
                members: class
                    .into_iter()
                    .map(|(index, witness)| ClassMemberRecord {
                        representative_id: records[index].id.clone(),
                        witness_from_class_representative: witness,
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
    let blocks = identity_hyperedges(N);
    let mut representatives = Vec::with_capacity(blocks.len());
    let mut reps = Vec::with_capacity(blocks.len());
    for (family_id, block) in blocks.iter().enumerate() {
        let solution = solve_garden_signing(block);
        let rep = build_rep(block, &solution.canonical_signs);
        let generators = gauge_generators(&rep);
        let gauge_action_rank = gf2_rank_u64(&generators);
        let gauge_generators_verified = generators
            .iter()
            .filter(|&&generator| {
                build_rep(block, &toggled_signs(&solution.canonical_signs, generator))
                    .verify_garden_algebra()
            })
            .count();
        let id = format!("family-{family_id:02}");
        representatives.push(RepresentativeRecord {
            id,
            family_id,
            identity_block_ranks: block
                .iter()
                .map(|permutation| permutation.rank() as u32)
                .collect(),
            solver_selected_sign_mask: sign_mask(&solution.canonical_signs),
            representation_sha256: representation_checksum(&rep),
            garden_verified: solution.feasible
                && solution.independent_sparse_verifier_passed
                && rep.verify_garden_algebra(),
            garden_nullity: solution.nullity,
            gauge_action_rank,
            gauge_generators_verified,
        });
        reps.push(rep);
    }

    let all_color_maps = color_orders();
    let policies = [
        (
            "fixed-color-nodal",
            "signed boson and fermion node permutations with all eight color labels fixed",
            EquivalencePolicy {
                allow_supercharge_signs: false,
                allow_color_permutation: false,
                allow_boson_fermion_duality: false,
            },
        ),
        (
            "fixed-color-nodal-plus-supercharge-signs",
            "fixed-color nodal equivalence enlarged by independent signs on the eight supercharges",
            EquivalencePolicy {
                allow_supercharge_signs: true,
                allow_color_permutation: false,
                allow_boson_fermion_duality: false,
            },
        ),
        (
            "unlabeled-color-signed-graph",
            "nodal equivalence, supercharge signs, and all permutations of the eight colors",
            EquivalencePolicy {
                allow_supercharge_signs: true,
                allow_color_permutation: true,
                allow_boson_fermion_duality: false,
            },
        ),
        (
            "unlabeled-color-signed-graph-plus-duality",
            "the preceding quotient enlarged by interchange of boson and fermion levels",
            EquivalencePolicy {
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
                &representatives,
                &reps,
                &all_color_maps,
            )
        })
        .collect();

    let all_representatives_close = representatives
        .iter()
        .all(|representative| representative.garden_verified);
    let every_witness_verified = equivalence_layers.iter().all(|layer| {
        layer.classes.iter().all(|class| {
            class.members.iter().all(|member| {
                member
                    .witness_from_class_representative
                    .verified_on_all_64_edges
            })
        })
    });
    let class_counts: Vec<usize> = equivalence_layers
        .iter()
        .map(|layer| layer.classes.len())
        .collect();
    let one_unlabeled_class = equivalence_layers[2].classes.len() == 1
        && equivalence_layers[2].classes[0].members.len() == representatives.len();
    let one_unlabeled_dual_class = equivalence_layers[3].classes.len() == 1
        && equivalence_layers[3].classes[0].members.len() == representatives.len();
    let garden_nullity_histogram = histogram(
        representatives
            .iter()
            .map(|representative| representative.garden_nullity),
    );
    let gauge_action_rank_histogram = histogram(
        representatives
            .iter()
            .map(|representative| representative.gauge_action_rank),
    );
    let all_gauge_generators_verified = representatives
        .iter()
        .all(|representative| representative.gauge_generators_verified == 2 * D + N);
    let gauge_spans_every_space = all_gauge_generators_verified
        && representatives.iter().all(|representative| {
            representative.gauge_action_rank == representative.garden_nullity
        });
    let signings_per_support = 1u64 << 19;
    let identity_support_labeled_signings_classified =
        representatives.len() as u64 * signings_per_support;
    let all_hyperedge_labeled_signings_classified = 151_200u64 * signings_per_support;
    let all_labeled_signings_one_class = one_unlabeled_class && gauge_spans_every_space;

    let mut mutated = equivalence_layers[2].classes[0].members[0]
        .witness_from_class_representative
        .clone();
    mutated.boson_switches[0] *= -1;
    let mutated_witness_rejected = !verify_witness(&reps[0], &reps[0], &mutated);
    let symmetry_broken = !one_unlabeled_class;
    let passed = representatives.len() == 30
        && all_representatives_close
        && every_witness_verified
        && mutated_witness_rejected
        && one_unlabeled_class
        && one_unlabeled_dual_class
        && garden_nullity_histogram == BTreeMap::from([(19, 30)])
        && gauge_action_rank_histogram == BTreeMap::from([(19, 30)])
        && all_labeled_signings_one_class;

    let validation = SignedEquivalenceValidation {
        representatives: representatives.len(),
        all_representatives_close,
        equivalence_class_counts_by_layer: class_counts.clone(),
        every_serialized_witness_verified: every_witness_verified,
        mutated_witness_rejected,
        one_unlabeled_color_signed_class: one_unlabeled_class,
        one_unlabeled_color_signed_class_with_duality: one_unlabeled_dual_class,
        garden_nullity_histogram,
        gauge_action_rank_histogram,
        all_gauge_generators_verified,
        gauge_action_spans_every_garden_signing_space: gauge_spans_every_space,
        identity_support_labeled_signings_classified,
        all_hyperedge_labeled_signings_classified,
        all_labeled_signings_share_one_unlabeled_color_signed_class: all_labeled_signings_one_class,
        canonical_sign_mask_used_as_invariant: false,
        one_dimensional_signed_data_break_the_thirty_fold_symmetry: symmetry_broken,
        passed,
    };

    SignedEquivalenceArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Signed-equivalence ledger for the 30 discovered S8 identity octets",
        method: vec![
            "Assign one deterministic, solver-selected Garden signing to each discovered identity octet solely as an orbit representative.",
            "Classify the 30 representatives under four explicitly separated signed equivalence relations.",
            "Serialize boson, fermion, color, node-switch, supercharge-sign, and level-duality maps for every class membership.",
            "Verify every witness directly against all 64 colored edges and reject a deliberately mutated witness.",
            "Compute the GF(2) rank of the 24 node-switch and supercharge-sign generators on every support and compare it with the exact Garden nullity.",
            "Use right-translation covariance to extend the identity-support classification to every translated hyperedge without enumerating labeled signings.",
        ],
        findings: vec![
            format!(
                "The four equivalence layers contain {} classes respectively.",
                class_counts
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!(
                "After color relabeling and supercharge signs are admitted, all {} discovered identity representatives lie in one signed graph class.",
                representatives.len()
            ),
            "Boson-fermion level duality does not merge anything further in this finite ledger.".into(),
            format!(
                "On every support the gauge-action rank is 19, exactly equal to the Garden nullity, so node switches and supercharge signs act transitively on all 2^19 signings."
            ),
            format!(
                "Right-translation covariance therefore places all {all_hyperedge_labeled_signings_classified} labeled signings across the 151,200 hyperedges in one unlabeled-color signed graph class."
            ),
            "The solver-selected sign masks are recorded for reproducibility but are not used as invariants or class identifiers.".into(),
        ],
        boundary: "This ledger classifies the full affine Garden-signing spaces by an exact gauge-rank argument, without enumerating them. The single-class result is scoped to unlabeled colors, signed node relabelings, supercharge signs, and optional boson-fermion duality. It does not identify that one-dimensional quotient with four-dimensional physical equivalence, enhancement, or parentage.",
        representatives,
        equivalence_layers,
        validation,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> SignedEquivalenceValidation {
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
    fn all_thirty_representatives_share_one_unlabeled_color_signed_class() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.representatives, 30);
        assert!(artifact.validation.every_serialized_witness_verified);
        assert!(artifact.validation.mutated_witness_rejected);
        assert!(artifact.validation.one_unlabeled_color_signed_class);
        assert_eq!(
            artifact.validation.gauge_action_rank_histogram,
            BTreeMap::from([(19, 30)])
        );
        assert!(
            artifact
                .validation
                .gauge_action_spans_every_garden_signing_space
        );
        assert!(
            artifact
                .validation
                .all_labeled_signings_share_one_unlabeled_color_signed_class
        );
        assert_eq!(
            artifact
                .validation
                .all_hyperedge_labeled_signings_classified,
            79_272_345_600
        );
        assert!(
            !artifact
                .validation
                .one_dimensional_signed_data_break_the_thirty_fold_symmetry
        );
    }
}
