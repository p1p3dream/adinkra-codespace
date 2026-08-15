//! Exact signed-node/color equivalence of the 25 printed one-Z branches.

#![allow(clippy::needless_range_loop)]

use crate::permutahedron::permutations;
use crate::vector_tensor_central_charge::{build as build_census, l_matrices, Matrix};
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;

#[derive(Clone)]
struct CentralRep {
    id: String,
    l: [Matrix; N],
    z_b: Matrix,
    z_f: Matrix,
    k: Matrix,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentralEquivalenceWitness {
    pub source_id: String,
    pub target_id: String,
    pub boson_map_zero_based: [usize; D],
    pub fermion_map_zero_based: [usize; D],
    pub color_map_zero_based: [usize; N],
    pub boson_switches: [i8; D],
    pub fermion_switches: [i8; D],
    pub supercharge_signs: [i8; N],
    pub central_generator_sign: i8,
    pub l_entries_verified: usize,
    pub z_boson_entries_verified: usize,
    pub z_fermion_entries_verified: usize,
    pub k_entries_verified: usize,
    pub verified: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentralEquivalenceReport {
    pub schema_version: &'static str,
    pub policy: &'static str,
    pub central_branches: usize,
    pub equivalence_classes: usize,
    pub witnesses: Vec<CentralEquivalenceWitness>,
    pub color_permutations_considered: usize,
    pub k_compatible_color_permutations: usize,
    pub root_maps_considered: usize,
    pub l_entries_verified: usize,
    pub central_entries_verified: usize,
    pub mutated_witness_rejected: bool,
    pub passed: bool,
    pub conclusion: &'static str,
    pub boundary: &'static str,
}

fn sign(value: i16) -> i8 {
    i8::try_from(value).expect("signed monomial entry")
}

fn image(matrix: &Matrix, row: usize) -> Option<(usize, i8)> {
    let entries: Vec<_> = (0..D).filter(|&column| matrix[row][column] != 0).collect();
    (entries.len() == 1).then(|| (entries[0], sign(matrix[row][entries[0]])))
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

fn color_support_compatible(source: &CentralRep, target: &CentralRep, map: &[usize; N]) -> bool {
    (0..N).all(|color| {
        let Some((partner, _)) = image(&source.k, color) else {
            return false;
        };
        image(&target.k, map[color])
            .is_some_and(|(target_partner, _)| target_partner == map[partner])
    })
}

fn unsigned_maps(
    source: &CentralRep,
    target: &CentralRep,
    color_map: &[usize; N],
    target_root: usize,
) -> Option<([usize; D], [usize; D])> {
    let mut fermion_map = [usize::MAX; D];
    for color in 0..N {
        let (source_fermion, _) = image(&source.l[color], 0)?;
        let (target_fermion, _) = image(&target.l[color_map[color]], target_root)?;
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
    for &entry in &fermion_map {
        if seen_fermions[entry] {
            return None;
        }
        seen_fermions[entry] = true;
    }

    let mut inverse_target = [[usize::MAX; D]; N];
    for color in 0..N {
        for boson in 0..D {
            let (fermion, _) = image(&target.l[color], boson)?;
            inverse_target[color][fermion] = boson;
        }
    }
    let mut boson_map = [usize::MAX; D];
    for boson in 0..D {
        let (source_fermion, _) = image(&source.l[0], boson)?;
        let candidate = inverse_target[color_map[0]][fermion_map[source_fermion]];
        if (1..N).any(|color| {
            let (source_fermion, _) = image(&source.l[color], boson).unwrap();
            inverse_target[color_map[color]][fermion_map[source_fermion]] != candidate
        }) {
            return None;
        }
        boson_map[boson] = candidate;
    }
    if boson_map[0] != target_root {
        return None;
    }
    let mut seen_bosons = [false; D];
    for &entry in &boson_map {
        if entry >= D || seen_bosons[entry] {
            return None;
        }
        seen_bosons[entry] = true;
    }

    for boson in 0..D {
        let (source_partner, _) = image(&source.z_b, boson)?;
        let (target_partner, _) = image(&target.z_b, boson_map[boson])?;
        if boson_map[source_partner] != target_partner {
            return None;
        }
    }
    for fermion in 0..D {
        let (source_partner, _) = image(&source.z_f, fermion)?;
        let (target_partner, _) = image(&target.z_f, fermion_map[fermion])?;
        if fermion_map[source_partner] != target_partner {
            return None;
        }
    }
    Some((boson_map, fermion_map))
}

fn verify_witness(
    source: &CentralRep,
    target: &CentralRep,
    witness: &CentralEquivalenceWitness,
) -> bool {
    for color in 0..N {
        let target_color = witness.color_map_zero_based[color];
        for boson in 0..D {
            let (source_fermion, source_sign) = image(&source.l[color], boson).unwrap();
            let target_boson = witness.boson_map_zero_based[boson];
            let (target_fermion, target_sign) =
                image(&target.l[target_color], target_boson).unwrap();
            if witness.fermion_map_zero_based[source_fermion] != target_fermion
                || source_sign
                    * witness.boson_switches[boson]
                    * witness.fermion_switches[source_fermion]
                    * witness.supercharge_signs[color]
                    != target_sign
            {
                return false;
            }
        }
    }
    for boson in 0..D {
        let (partner, source_sign) = image(&source.z_b, boson).unwrap();
        let target_boson = witness.boson_map_zero_based[boson];
        let (target_partner, target_sign) = image(&target.z_b, target_boson).unwrap();
        if witness.boson_map_zero_based[partner] != target_partner
            || source_sign
                * witness.central_generator_sign
                * witness.boson_switches[boson]
                * witness.boson_switches[partner]
                != target_sign
        {
            return false;
        }
    }
    for fermion in 0..D {
        let (partner, source_sign) = image(&source.z_f, fermion).unwrap();
        let target_fermion = witness.fermion_map_zero_based[fermion];
        let (target_partner, target_sign) = image(&target.z_f, target_fermion).unwrap();
        if witness.fermion_map_zero_based[partner] != target_partner
            || source_sign
                * witness.central_generator_sign
                * witness.fermion_switches[fermion]
                * witness.fermion_switches[partner]
                != target_sign
        {
            return false;
        }
    }
    for color in 0..N {
        let (partner, source_sign) = image(&source.k, color).unwrap();
        let target_color = witness.color_map_zero_based[color];
        let (target_partner, target_sign) = image(&target.k, target_color).unwrap();
        if witness.color_map_zero_based[partner] != target_partner
            || source_sign
                * witness.central_generator_sign
                * witness.supercharge_signs[color]
                * witness.supercharge_signs[partner]
                != target_sign
        {
            return false;
        }
    }
    true
}

fn witness_for_maps(
    source: &CentralRep,
    target: &CentralRep,
    color_map: [usize; N],
    boson_map: [usize; D],
    fermion_map: [usize; D],
) -> Option<CentralEquivalenceWitness> {
    const VARIABLES: usize = 2 * D + N + 1;
    const Z_BIT: usize = 2 * D + N;
    let mut equations = Vec::new();
    for color in 0..N {
        for boson in 0..D {
            let (source_fermion, source_sign) = image(&source.l[color], boson)?;
            let (target_fermion, target_sign) =
                image(&target.l[color_map[color]], boson_map[boson])?;
            if fermion_map[source_fermion] != target_fermion {
                return None;
            }
            equations.push((
                (1 << boson) | (1 << (D + source_fermion)) | (1 << (2 * D + color)),
                u8::from(source_sign != target_sign),
            ));
        }
    }
    for boson in 0..D {
        let (partner, source_sign) = image(&source.z_b, boson)?;
        let (target_partner, target_sign) = image(&target.z_b, boson_map[boson])?;
        if boson_map[partner] != target_partner {
            return None;
        }
        equations.push((
            (1 << boson) | (1 << partner) | (1 << Z_BIT),
            u8::from(source_sign != target_sign),
        ));
    }
    for fermion in 0..D {
        let (partner, source_sign) = image(&source.z_f, fermion)?;
        let (target_partner, target_sign) = image(&target.z_f, fermion_map[fermion])?;
        if fermion_map[partner] != target_partner {
            return None;
        }
        equations.push((
            (1 << (D + fermion)) | (1 << (D + partner)) | (1 << Z_BIT),
            u8::from(source_sign != target_sign),
        ));
    }
    for color in 0..N {
        let (partner, source_sign) = image(&source.k, color)?;
        let (target_partner, target_sign) = image(&target.k, color_map[color])?;
        if color_map[partner] != target_partner {
            return None;
        }
        equations.push((
            (1 << (2 * D + color)) | (1 << (2 * D + partner)) | (1 << Z_BIT),
            u8::from(source_sign != target_sign),
        ));
    }
    let solution = solve_gf2(&mut equations, VARIABLES)?;
    let to_sign = |bit: u8| if bit == 0 { 1 } else { -1 };
    let mut witness = CentralEquivalenceWitness {
        source_id: source.id.clone(),
        target_id: target.id.clone(),
        boson_map_zero_based: boson_map,
        fermion_map_zero_based: fermion_map,
        color_map_zero_based: color_map,
        boson_switches: std::array::from_fn(|index| to_sign(solution[index])),
        fermion_switches: std::array::from_fn(|index| to_sign(solution[D + index])),
        supercharge_signs: std::array::from_fn(|index| to_sign(solution[2 * D + index])),
        central_generator_sign: to_sign(solution[Z_BIT]),
        l_entries_verified: N * D,
        z_boson_entries_verified: D,
        z_fermion_entries_verified: D,
        k_entries_verified: N,
        verified: false,
    };
    witness.verified = verify_witness(source, target, &witness);
    witness.verified.then_some(witness)
}

fn central_reps() -> Vec<CentralRep> {
    let census = build_census();
    let mut result = Vec::new();
    for (sector_index, sector) in census.sectors.iter().enumerate() {
        for branch in &sector.branches {
            let Some(charge) = &branch.central_charge else {
                continue;
            };
            let suffix = branch
                .m_mod_4
                .zip(branch.n_mod_4)
                .map(|(m, n)| format!("m{m}n{n}"))
                .unwrap_or_else(|| "base".into());
            result.push(CentralRep {
                id: format!("{}:{suffix}", sector.id),
                l: l_matrices(sector_index, branch.effective_boolean_factors),
                z_b: charge.bosonic,
                z_f: charge.fermionic,
                k: charge.color_coefficient_matrix,
            });
        }
    }
    result
}

pub fn build() -> CentralEquivalenceReport {
    let reps = central_reps();
    let source = &reps[0];
    let color_maps: Vec<[usize; N]> = permutations(N)
        .expect("enumerate S8")
        .map(|permutation| {
            permutation
                .as_slice()
                .iter()
                .map(|entry| usize::from(entry - 1))
                .collect::<Vec<_>>()
                .try_into()
                .expect("eight colors")
        })
        .collect();
    let compatible: Vec<_> = color_maps
        .iter()
        .copied()
        .filter(|map| color_support_compatible(source, source, map))
        .collect();
    let mut witnesses = Vec::new();
    let mut roots = 0usize;
    for target in &reps {
        let mut found = None;
        for &color_map in &color_maps {
            if !color_support_compatible(source, target, &color_map) {
                continue;
            }
            for target_root in 0..D {
                roots += 1;
                let Some((boson_map, fermion_map)) =
                    unsigned_maps(source, target, &color_map, target_root)
                else {
                    continue;
                };
                if let Some(witness) =
                    witness_for_maps(source, target, color_map, boson_map, fermion_map)
                {
                    found = Some(witness);
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        if let Some(witness) = found {
            witnesses.push(witness);
        }
    }
    let mut mutated = witnesses.last().cloned().expect("central witness");
    mutated.central_generator_sign *= -1;
    let mutated_witness_rejected = !verify_witness(source, reps.last().unwrap(), &mutated);
    let all_verified = witnesses.iter().all(|witness| witness.verified);
    let passed = reps.len() == 25
        && witnesses.len() == reps.len()
        && compatible.len() == 384
        && all_verified
        && mutated_witness_rejected;
    CentralEquivalenceReport {
        schema_version: "vector-tensor-central-equivalence-v1",
        policy: "independent signed boson and fermion permutations, signed color permutations, and simultaneous Z/K orientation; no boson-fermion duality",
        central_branches: reps.len(),
        equivalence_classes: usize::from(!witnesses.is_empty()),
        l_entries_verified: witnesses.len() * N * D,
        central_entries_verified: witnesses.len() * (D + D + N),
        witnesses,
        color_permutations_considered: color_maps.len(),
        k_compatible_color_permutations: compatible.len(),
        root_maps_considered: roots,
        mutated_witness_rejected,
        passed,
        conclusion: "All 25 printed one-central-charge branches are one enriched signed-node/color equivalence class.",
        boundary: "This classifies the printed branches. Full transport to all 151,200 unsigned hyperedges is a separate atlas calculation.",
    }
}

pub fn write_artifact(path: &Path) -> CentralEquivalenceReport {
    let report = build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create central-equivalence directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create central-equivalence artifact")),
        &report,
    )
    .expect("write central-equivalence artifact");
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printed_one_z_branches_form_one_class() {
        let report = build();
        assert!(
            report.passed,
            "branches={} witnesses={} colors={} compatible={} mutated={}",
            report.central_branches,
            report.witnesses.len(),
            report.color_permutations_considered,
            report.k_compatible_color_permutations,
            report.mutated_witness_rejected
        );
        assert_eq!(report.central_branches, 25);
        assert_eq!(report.equivalence_classes, 1);
    }
}
