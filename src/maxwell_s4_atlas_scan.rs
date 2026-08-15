//! Apply the validated Maxwell worldline-recovery gate to all 96 published
//! fiducial signed S4 quartets.

use crate::maxwell_phantom::{gauge_enhancement_gate, WorldlineLinkage};
use crate::maxwell_worldline_search::search_worldline;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const LABELS: [&str; 6] = ["CM", "TM", "VM", "VM1", "VM2", "VM3"];

#[derive(Clone, Debug, Serialize)]
pub struct SigningScanRecord {
    pub label: &'static str,
    pub appendix_b_index: usize,
    pub boolean_factors: [u8; 4],
    pub chi0: i64,
    pub charge_zero_normalized_candidates: usize,
    pub maxwell_gauge_enhancing_frames: usize,
    pub direct_source_frame_passes: bool,
    pub passes_maxwell_gate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SectorScanRecord {
    pub label: &'static str,
    pub signings_tested: usize,
    pub passing_signings: usize,
    pub chi0_histogram: BTreeMap<i64, usize>,
    pub passing_chi0_histogram: BTreeMap<i64, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MaxwellS4AtlasArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<&'static str>,
    pub signings: Vec<SigningScanRecord>,
    pub sectors: Vec<SectorScanRecord>,
    pub signed_quartets_tested: usize,
    pub garden_closing_inputs: usize,
    pub passing_signed_quartets: usize,
    pub direct_source_frame_passers: usize,
    pub distinct_passing_chi0_values: Vec<i64>,
    pub pass_status_is_constant_within_each_chi0_class: bool,
    pub every_sector_splits_eight_and_eight: bool,
    pub known_vector_sector_has_passers: bool,
    pub passing_if_and_only_if_chi0_is_minus_one: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn linkage(matrices: &[crate::permutahedron_s4_supersymmetry::MatrixRecord]) -> WorldlineLinkage {
    assert_eq!(matrices.len(), 4);
    std::array::from_fn(|charge| matrices[charge].l)
}

pub fn build() -> MaxwellS4AtlasArtifact {
    let atlas = crate::permutahedron_s4_supersymmetry::build();
    let mut signings = Vec::with_capacity(96);
    let mut sectors = Vec::with_capacity(6);
    for (sector_index, sector) in atlas.sectors.iter().enumerate() {
        let mut passing_signings = 0;
        let mut chi0_histogram = BTreeMap::new();
        let mut passing_chi0_histogram = BTreeMap::new();
        for signing in &sector.published_fiducial_signings {
            *chi0_histogram.entry(signing.chi0).or_default() += 1;
            let input = linkage(&signing.matrices);
            let direct_source_frame_passes = gauge_enhancement_gate(&input).passed();
            let result = search_worldline("published S4 signing", &input);
            let passes = result.gauge_enhancing_candidates > 0;
            if passes {
                passing_signings += 1;
                *passing_chi0_histogram.entry(signing.chi0).or_default() += 1;
            }
            signings.push(SigningScanRecord {
                label: LABELS[sector_index],
                appendix_b_index: signing.appendix_b_index,
                boolean_factors: signing.boolean_factors,
                chi0: signing.chi0,
                charge_zero_normalized_candidates: result.charge_zero_normalized_candidates,
                maxwell_gauge_enhancing_frames: result.gauge_enhancing_candidates,
                direct_source_frame_passes,
                passes_maxwell_gate: passes,
            });
        }
        sectors.push(SectorScanRecord {
            label: LABELS[sector_index],
            signings_tested: sector.published_fiducial_signings.len(),
            passing_signings,
            chi0_histogram,
            passing_chi0_histogram,
        });
    }
    let passing_signed_quartets = signings
        .iter()
        .filter(|record| record.passes_maxwell_gate)
        .count();
    let direct_source_frame_passers = signings
        .iter()
        .filter(|record| record.direct_source_frame_passes)
        .count();
    let distinct_passing_chi0_values: Vec<_> = signings
        .iter()
        .filter(|record| record.passes_maxwell_gate)
        .map(|record| record.chi0)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut chi0_status = BTreeMap::new();
    let pass_status_is_constant_within_each_chi0_class = signings.iter().all(|record| {
        *chi0_status
            .entry(record.chi0)
            .or_insert(record.passes_maxwell_gate)
            == record.passes_maxwell_gate
    });
    let sector_passes = |label| {
        sectors
            .iter()
            .find(|sector| sector.label == label)
            .expect("named S4 sector")
            .passing_signings
    };
    let known_vector_sector_has_passers = sector_passes("VM") > 0;
    let every_sector_splits_eight_and_eight = sectors
        .iter()
        .all(|sector| sector.signings_tested == 16 && sector.passing_signings == 8);
    let passing_if_and_only_if_chi0_is_minus_one = signings
        .iter()
        .all(|record| record.passes_maxwell_gate == (record.chi0 == -1));
    let passed = atlas.validation.passed
        && signings.len() == 96
        && pass_status_is_constant_within_each_chi0_class
        && every_sector_splits_eight_and_eight
        && known_vector_sector_has_passers
        && passing_if_and_only_if_chi0_is_minus_one;
    MaxwellS4AtlasArtifact {
        schema_version: "maxwell-s4-atlas-scan-v1",
        title: "Maxwell gauge-enhancement scan of all published fiducial signed S4 quartets",
        sources: vec![
            "arXiv:1701.00304 Appendix B",
            "arXiv:0907.3605 Section 5.5 and Eq. (5.11)",
        ],
        signings,
        sectors,
        signed_quartets_tested: 96,
        garden_closing_inputs: atlas.validation.garden_representations_passed,
        passing_signed_quartets,
        direct_source_frame_passers,
        distinct_passing_chi0_values,
        pass_status_is_constant_within_each_chi0_class,
        every_sector_splits_eight_and_eight,
        known_vector_sector_has_passers,
        passing_if_and_only_if_chi0_is_minus_one,
        passed,
        boundary: "This exhausts the 96 published fiducial signings, not all rankings or all BC4 transformations. The unsigned labels CM, TM, VM, VM1, VM2, and VM3 each contain both chi0 classes in this library, so a passing signing does not assign that unsigned sector a Maxwell parent.",
    }
}

pub fn write_artifact(path: &Path) -> MaxwellS4AtlasArtifact {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create Maxwell S4 atlas artifact")),
        &artifact,
    )
    .expect("write Maxwell S4 atlas artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_published_s4_signings_are_classified() {
        let artifact = build();
        assert_eq!(artifact.signed_quartets_tested, 96);
        assert_eq!(artifact.garden_closing_inputs, 96);
        assert!(artifact.known_vector_sector_has_passers);
        assert_eq!(artifact.passing_signed_quartets, 48);
        assert_eq!(artifact.direct_source_frame_passers, 0);
        assert_eq!(artifact.distinct_passing_chi0_values, vec![-1]);
        assert!(artifact.every_sector_splits_eight_and_eight);
        assert!(artifact.passing_if_and_only_if_chi0_is_minus_one);
        assert!(artifact.pass_status_is_constant_within_each_chi0_class);
        assert!(artifact.passed);
    }
}
