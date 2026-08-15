//! Recover the Maxwell gauge-enhancement passer from worldline linkage data.

use crate::maxwell_phantom::{
    gauge_enhancement_gate, source_chiral_worldline, source_maxwell_worldline, WorldlineLinkage,
};
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SignedFrame {
    pub permutation_zero_based: [usize; 4],
    pub signs: [i8; 4],
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchWitness {
    pub boson_frame: SignedFrame,
    pub fermion_frame: SignedFrame,
    pub canonical_bosonic_residual_entries: usize,
    pub fermionic_residual_entries: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchResult {
    pub input_name: &'static str,
    pub signed_boson_frames: usize,
    pub signed_fermion_frames: usize,
    pub frame_pairs_examined: usize,
    pub charge_zero_normalized_candidates: usize,
    pub gauge_enhancing_candidates: usize,
    pub witnesses: Vec<SearchWitness>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MaxwellWorldlineSearchArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<&'static str>,
    pub normalization: &'static str,
    pub maxwell_source_basis: SearchResult,
    pub maxwell_scrambled_basis: SearchResult,
    pub chiral_negative_control: SearchResult,
    pub source_and_scrambled_pass_counts_agree: bool,
    pub maxwell_is_recovered: bool,
    pub chiral_is_rejected: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn permutations4() -> Vec<[usize; 4]> {
    let mut output = Vec::with_capacity(24);
    for a in 0..4 {
        for b in 0..4 {
            if b == a {
                continue;
            }
            for c in 0..4 {
                if c == a || c == b {
                    continue;
                }
                for d in 0..4 {
                    if d != a && d != b && d != c {
                        output.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    output
}

fn signed_frames() -> Vec<SignedFrame> {
    let mut output = Vec::with_capacity(384);
    for permutation_zero_based in permutations4() {
        for mask in 0_u8..16 {
            output.push(SignedFrame {
                permutation_zero_based,
                signs: std::array::from_fn(|index| if mask & (1 << index) == 0 { 1 } else { -1 }),
            });
        }
    }
    output
}

fn transform(
    input: &WorldlineLinkage,
    boson: SignedFrame,
    fermion: SignedFrame,
) -> WorldlineLinkage {
    std::array::from_fn(|charge| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                boson.signs[row]
                    * fermion.signs[column]
                    * input[charge][boson.permutation_zero_based[row]]
                        [fermion.permutation_zero_based[column]]
            })
        })
    })
}

fn charge_zero_matches(
    input: &WorldlineLinkage,
    boson: SignedFrame,
    fermion: SignedFrame,
    target: &[[i8; 4]; 4],
) -> bool {
    (0..4).all(|row| {
        (0..4).all(|column| {
            boson.signs[row]
                * fermion.signs[column]
                * input[0][boson.permutation_zero_based[row]]
                    [fermion.permutation_zero_based[column]]
                == target[row][column]
        })
    })
}

pub(crate) fn search_worldline(input_name: &'static str, input: &WorldlineLinkage) -> SearchResult {
    let frames = signed_frames();
    let target_charge_zero = source_maxwell_worldline()[0];
    let mut frame_pairs_examined = 0;
    let mut charge_zero_normalized_candidates = 0;
    let mut gauge_enhancing_candidates = 0;
    let mut witnesses = Vec::new();
    for &boson_frame in &frames {
        for &fermion_frame in &frames {
            frame_pairs_examined += 1;
            if !charge_zero_matches(input, boson_frame, fermion_frame, &target_charge_zero) {
                continue;
            }
            charge_zero_normalized_candidates += 1;
            let candidate = transform(input, boson_frame, fermion_frame);
            let gate = gauge_enhancement_gate(&candidate);
            if gate.passed() {
                gauge_enhancing_candidates += 1;
                if witnesses.len() < 16 {
                    witnesses.push(SearchWitness {
                        boson_frame,
                        fermion_frame,
                        canonical_bosonic_residual_entries: gate.canonical_bosonic_residual_entries,
                        fermionic_residual_entries: gate.fermionic_residual_entries,
                    });
                }
            }
        }
    }
    SearchResult {
        input_name,
        signed_boson_frames: frames.len(),
        signed_fermion_frames: frames.len(),
        frame_pairs_examined,
        charge_zero_normalized_candidates,
        gauge_enhancing_candidates,
        witnesses,
    }
}

fn scrambled_maxwell() -> WorldlineLinkage {
    transform(
        &source_maxwell_worldline(),
        SignedFrame {
            permutation_zero_based: [2, 0, 3, 1],
            signs: [-1, 1, -1, 1],
        },
        SignedFrame {
            permutation_zero_based: [1, 3, 0, 2],
            signs: [1, -1, 1, -1],
        },
    )
}

pub fn build() -> MaxwellWorldlineSearchArtifact {
    let maxwell_source_basis =
        search_worldline("Maxwell source basis", &source_maxwell_worldline());
    let maxwell_scrambled_basis = search_worldline(
        "Maxwell deterministically scrambled basis",
        &scrambled_maxwell(),
    );
    let chiral_negative_control =
        search_worldline("chiral negative control", &source_chiral_worldline());
    let source_and_scrambled_pass_counts_agree = maxwell_source_basis.gauge_enhancing_candidates
        == maxwell_scrambled_basis.gauge_enhancing_candidates;
    let maxwell_is_recovered = maxwell_source_basis.gauge_enhancing_candidates > 0
        && maxwell_scrambled_basis.gauge_enhancing_candidates > 0;
    let chiral_is_rejected = chiral_negative_control.gauge_enhancing_candidates == 0;
    let passed =
        source_and_scrambled_pass_counts_agree && maxwell_is_recovered && chiral_is_rejected;
    MaxwellWorldlineSearchArtifact {
        schema_version: "maxwell-worldline-search-v1",
        title: "Recovery of the Maxwell gauge-enhancement passer from worldline data",
        sources: vec![
            "arXiv:0907.3605 Section 5.5 and Eqs. (5.6), (5.8), (5.9), and (5.11)",
            "arXiv:1405.0048 Eqs. (32)-(41)",
        ],
        normalization: "All 384 signed boson frames and 384 signed fermion frames are examined. Charge zero is normalized to the fixed Majorana source frame before the gauge-enhancement gate is evaluated.",
        maxwell_source_basis,
        maxwell_scrambled_basis,
        chiral_negative_control,
        source_and_scrambled_pass_counts_agree,
        maxwell_is_recovered,
        chiral_is_rejected,
        passed,
        boundary: "This search varies signed boson and fermion frames while retaining the printed supercharge order and Majorana gamma basis. It is a four-color positive-control recovery, not an eight-color enhancement result.",
    }
}

pub fn write_artifact(path: &Path) -> MaxwellWorldlineSearchArtifact {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create Maxwell worldline-search artifact")),
        &artifact,
    )
    .expect("write Maxwell worldline-search artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_frame_inventory_is_complete() {
        let frames = signed_frames();
        assert_eq!(frames.len(), 24 * 16);
        let unique: std::collections::BTreeSet<_> = frames
            .iter()
            .map(|frame| (frame.permutation_zero_based, frame.signs))
            .collect();
        assert_eq!(unique.len(), frames.len());
    }

    #[test]
    fn maxwell_is_recovered_and_chiral_is_rejected() {
        let artifact = build();
        assert_eq!(
            artifact.maxwell_source_basis.frame_pairs_examined,
            384 * 384
        );
        assert_eq!(
            artifact
                .maxwell_source_basis
                .charge_zero_normalized_candidates,
            384
        );
        assert_eq!(artifact.maxwell_source_basis.gauge_enhancing_candidates, 8);
        assert_eq!(
            artifact.maxwell_scrambled_basis.gauge_enhancing_candidates,
            8
        );
        assert_eq!(
            artifact.chiral_negative_control.gauge_enhancing_candidates,
            0
        );
        assert!(artifact.maxwell_is_recovered);
        assert!(artifact.source_and_scrambled_pass_counts_agree);
        assert!(artifact.chiral_is_rejected);
        assert!(artifact.passed);
    }
}
