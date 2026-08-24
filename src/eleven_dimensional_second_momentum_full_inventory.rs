//! Canonical data-driven layout for all 77 exact second-momentum columns.
//!
//! The original production path described the `(10001)`, `(20001)`, and
//! `(30001)` slices in separate modules. This module is the shared boundary
//! for the full inventory. It binds every global ordinal to one exact
//! level-12 source embedding and one momentum recoupling path without changing
//! the established ordering of the completed slices.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub(crate) const FULL_COLUMN_COUNT: usize = 77;
pub(crate) const SOURCE_INCIDENCE_COUNT: usize = 73;
pub(crate) const SOURCE_INTERMEDIATE_PAIR_COUNT: usize = 35;
pub(crate) const SOURCE_LABEL_COUNT: usize = 19;

const CHANNEL_ORDER: [&str; 6] = ["00001", "01001", "10001", "11001", "20001", "30001"];

// This is the canonical order inherited from the exact Lambda^12(S)
// decomposition. Filtering it by a target channel reproduces the established
// 20001 and 30001 global-column ordering.
const SOURCE_ORDER: [&str; 19] = [
    "00000", "00100", "00010", "40000", "02000", "10002", "01100", "20100", "12000", "31000",
    "20010", "01002", "20002", "30100", "11100", "30010", "11010", "30002", "11002",
];

#[derive(Clone, Copy, Debug)]
pub(crate) struct Level12FixtureRef {
    pub dynkin_label: &'static str,
    pub copy: usize,
    pub artifact: &'static str,
    pub coefficient_width_bytes: usize,
    pub bytes: &'static [u8],
}

macro_rules! fixture {
    ($label:literal, $copy:literal, $artifact:literal) => {
        Level12FixtureRef {
            dynkin_label: $label,
            copy: $copy,
            artifact: $artifact,
            coefficient_width_bytes: if $artifact.ends_with(".i32le") { 4 } else { 2 },
            bytes: include_bytes!(concat!(
                "../data/eleven_dimensional_spinor_bridge/",
                $artifact
            )),
        }
    };
}

pub(crate) fn level12_fixtures() -> Vec<Level12FixtureRef> {
    vec![
        fixture!("00000", 1, "level12_00000_highest_weight_kernel.i16le"),
        fixture!("00010", 1, "level12_00010_highest_weight_kernel.i16le"),
        fixture!("00100", 1, "level12_00100_highest_weight_kernel.i16le"),
        fixture!("01002", 1, "level12_01002_highest_weight_kernel_1.i32le"),
        fixture!("01002", 2, "level12_01002_highest_weight_kernel_2.i32le"),
        fixture!("01002", 3, "level12_01002_highest_weight_kernel_3.i32le"),
        fixture!("01002", 4, "level12_01002_highest_weight_kernel_4.i32le"),
        fixture!("01100", 1, "level12_01100_highest_weight_kernel_1.i16le"),
        fixture!("01100", 2, "level12_01100_highest_weight_kernel_2.i16le"),
        fixture!("02000", 1, "level12_02000_highest_weight_kernel_1.i16le"),
        fixture!("02000", 2, "level12_02000_highest_weight_kernel_2.i16le"),
        fixture!("10002", 1, "level12_10002_highest_weight_kernel_1.i16le"),
        fixture!("10002", 2, "level12_10002_highest_weight_kernel_2.i16le"),
        fixture!("11002", 1, "level12_11002_highest_weight_kernel_1.i16le"),
        fixture!("11002", 2, "level12_11002_highest_weight_kernel_2.i16le"),
        fixture!("11002", 3, "level12_11002_highest_weight_kernel_3.i16le"),
        fixture!("11002", 4, "level12_11002_highest_weight_kernel_4.i16le"),
        fixture!("11002", 5, "level12_11002_highest_weight_kernel_5.i16le"),
        fixture!("11010", 1, "level12_11010_highest_weight_kernel_1.i32le"),
        fixture!("11010", 2, "level12_11010_highest_weight_kernel_2.i32le"),
        fixture!("11010", 3, "level12_11010_highest_weight_kernel_3.i32le"),
        fixture!("11010", 4, "level12_11010_highest_weight_kernel_4.i32le"),
        fixture!("11100", 1, "level12_11100_highest_weight_kernel_1.i16le"),
        fixture!("11100", 2, "level12_11100_highest_weight_kernel_2.i16le"),
        fixture!("11100", 3, "level12_11100_highest_weight_kernel_3.i16le"),
        fixture!("12000", 1, "level12_12000_highest_weight_kernel.i16le"),
        fixture!("20002", 1, "level12_20002_highest_weight_kernel_1.i16le"),
        fixture!("20002", 2, "level12_20002_highest_weight_kernel_2.i16le"),
        fixture!("20010", 1, "level12_20010_highest_weight_kernel_1.i16le"),
        fixture!("20010", 2, "level12_20010_highest_weight_kernel_2.i16le"),
        fixture!("20010", 3, "level12_20010_highest_weight_kernel_3.i16le"),
        fixture!("20100", 1, "level12_20100_highest_weight_kernel_1.i16le"),
        fixture!("20100", 2, "level12_20100_highest_weight_kernel_2.i16le"),
        fixture!("30002", 1, "level12_30002_highest_weight_kernel_1.i16le"),
        fixture!("30002", 2, "level12_30002_highest_weight_kernel_2.i16le"),
        fixture!("30002", 3, "level12_30002_highest_weight_kernel_3.i16le"),
        fixture!("30010", 1, "level12_30010_highest_weight_kernel_1.i16le"),
        fixture!("30010", 2, "level12_30010_highest_weight_kernel_2.i16le"),
        fixture!("30100", 1, "level12_30100_highest_weight_kernel.i16le"),
        fixture!("31000", 1, "level12_31000_highest_weight_kernel.i16le"),
        fixture!("40000", 1, "level12_40000_highest_weight_kernel.i16le"),
    ]
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MomentumPath {
    Unique,
    Trace,
    SymmetricTraceless,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct FullColumnSpec {
    pub global_ordinal: usize,
    pub intermediate_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coefficient_width_bytes: usize,
    pub momentum_path: MomentumPath,
    pub legal_group_key: String,
}

fn source_couples_to(source: &str, target: &str) -> bool {
    crate::eleven_dimensional_prepotential::spinor_tensor_channels(source)
        .iter()
        .any(|(candidate, _)| candidate == target)
}

fn fixture_sha256(fixture: Level12FixtureRef) -> String {
    format!("{:x}", Sha256::digest(fixture.bytes))
}

fn build_full_column_specs() -> Vec<FullColumnSpec> {
    let fixtures = level12_fixtures();
    let by_source = fixtures.iter().fold(
        BTreeMap::<&str, Vec<Level12FixtureRef>>::new(),
        |mut grouped, fixture| {
            grouped
                .entry(fixture.dynkin_label)
                .or_default()
                .push(*fixture);
            grouped
        },
    );
    let mut columns = Vec::with_capacity(FULL_COLUMN_COUNT);
    for target in CHANNEL_ORDER {
        for source in SOURCE_ORDER {
            if !source_couples_to(source, target) {
                continue;
            }
            let source_fixtures = by_source
                .get(source)
                .unwrap_or_else(|| panic!("missing level-12 fixtures for {source}"));
            for fixture in source_fixtures {
                let paths: &[MomentumPath] = if target == "10001" {
                    &[MomentumPath::Trace, MomentumPath::SymmetricTraceless]
                } else {
                    &[MomentumPath::Unique]
                };
                for &momentum_path in paths {
                    let path_key = match momentum_path {
                        MomentumPath::Unique => "unique",
                        MomentumPath::Trace => "trace",
                        MomentumPath::SymmetricTraceless => "stt",
                    };
                    columns.push(FullColumnSpec {
                        global_ordinal: columns.len(),
                        intermediate_dynkin_label: target.to_string(),
                        source_dynkin_label: source.to_string(),
                        source_copy: fixture.copy,
                        source_fixture: fixture.artifact.to_string(),
                        source_fixture_sha256: fixture_sha256(*fixture),
                        coefficient_width_bytes: fixture.coefficient_width_bytes,
                        momentum_path,
                        legal_group_key: format!("{target}:{source}:{path_key}"),
                    });
                }
            }
        }
    }
    columns
}

fn canonical_full_column_specs() -> &'static [FullColumnSpec] {
    static COLUMNS: OnceLock<Vec<FullColumnSpec>> = OnceLock::new();
    COLUMNS.get_or_init(build_full_column_specs)
}

pub(crate) fn full_column_specs() -> Vec<FullColumnSpec> {
    canonical_full_column_specs().to_vec()
}

pub(crate) fn missing_49_column_specs() -> Vec<FullColumnSpec> {
    full_column_specs()
        .into_iter()
        .filter(|column| matches!(column.global_ordinal, 0..=18 | 23..=52))
        .collect()
}

/// Canonical unique-path GPU groups for the 45 missing columns whose
/// reciprocal map is already provided by the five-channel exact recoupling
/// certificate. Groups preserve global-column order and are split to the
/// production-tested maximum width of three.
pub(crate) fn missing_unique_gpu_groups() -> Vec<Vec<usize>> {
    let columns = canonical_full_column_specs();
    let mut groups = Vec::<Vec<usize>>::new();
    for column in columns.iter().filter(|column| {
        matches!(column.global_ordinal, 0..=18 | 23..=52)
            && column.momentum_path == MomentumPath::Unique
            && matches!(
                column.intermediate_dynkin_label.as_str(),
                "00001" | "01001" | "11001"
            )
    }) {
        let append = groups.last().is_some_and(|group| {
            group.len() < 3 && columns[group[0]].legal_group_key == column.legal_group_key
        });
        if append {
            groups.last_mut().unwrap().push(column.global_ordinal);
        } else {
            groups.push(vec![column.global_ordinal]);
        }
    }
    groups
}

/// Canonical production groups for all 53 columns outside the established
/// `(20001)` and `(30001)` tranches. This includes the four original
/// `(10001)` slice columns so every one of the first 53 columns can be emitted
/// in the same rankable GPU artifact format. `(10001)` path columns remain
/// singleton groups because trace and symmetric-traceless schedules have
/// distinct certified identities.
pub(crate) fn missing_gpu_groups() -> Vec<Vec<usize>> {
    let columns = canonical_full_column_specs();
    let mut groups = Vec::<Vec<usize>>::new();
    for column in columns
        .iter()
        .filter(|column| matches!(column.global_ordinal, 0..=52))
    {
        let append = groups.last().is_some_and(|group| {
            group.len() < 3 && columns[group[0]].legal_group_key == column.legal_group_key
        });
        if append {
            groups.last_mut().unwrap().push(column.global_ordinal);
        } else {
            groups.push(vec![column.global_ordinal]);
        }
    }
    groups
}

pub(crate) fn layout_sha256() -> String {
    let payload =
        serde_json::to_vec(canonical_full_column_specs()).expect("serialize full column layout");
    format!("{:x}", Sha256::digest(payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn full_layout_reproduces_the_exact_77_column_inventory() {
        let columns = full_column_specs();
        assert_eq!(columns.len(), FULL_COLUMN_COUNT);
        assert_eq!(
            columns
                .iter()
                .map(|column| column.global_ordinal)
                .collect::<Vec<_>>(),
            (0..FULL_COLUMN_COUNT).collect::<Vec<_>>()
        );
        let channel_counts = columns.iter().fold(BTreeMap::new(), |mut counts, column| {
            *counts
                .entry(column.intermediate_dynkin_label.as_str())
                .or_insert(0_usize) += 1;
            counts
        });
        assert_eq!(
            channel_counts,
            BTreeMap::from([
                ("00001", 3),
                ("01001", 12),
                ("10001", 8),
                ("11001", 30),
                ("20001", 9),
                ("30001", 15),
            ])
        );
        assert_eq!(missing_49_column_specs().len(), 49);
    }

    #[test]
    fn exact_level12_fixture_inventory_is_complete_and_canonical() {
        let fixtures = level12_fixtures();
        assert_eq!(fixtures.len(), 41);
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| fixture.dynkin_label)
                .collect::<BTreeSet<_>>()
                .len(),
            SOURCE_LABEL_COUNT
        );
        for fixture in fixtures {
            assert!(!fixture.bytes.is_empty());
            assert_eq!(fixture.bytes.len() % fixture.coefficient_width_bytes, 0);
            assert!(
                fixture
                    .artifact
                    .ends_with(if fixture.coefficient_width_bytes == 4 {
                        ".i32le"
                    } else {
                        ".i16le"
                    })
            );
        }
    }

    #[test]
    fn established_20001_and_30001_ordinals_do_not_move() {
        let columns = full_column_specs();
        let established = columns[53..]
            .iter()
            .map(|column| {
                (
                    column.global_ordinal,
                    column.intermediate_dynkin_label.as_str(),
                    column.source_dynkin_label.as_str(),
                    column.source_copy,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            established,
            vec![
                (53, "20001", "10002", 1),
                (54, "20001", "10002", 2),
                (55, "20001", "20100", 1),
                (56, "20001", "20100", 2),
                (57, "20001", "20010", 1),
                (58, "20001", "20010", 2),
                (59, "20001", "20010", 3),
                (60, "20001", "20002", 1),
                (61, "20001", "20002", 2),
                (62, "30001", "40000", 1),
                (63, "30001", "20100", 1),
                (64, "30001", "20100", 2),
                (65, "30001", "31000", 1),
                (66, "30001", "20010", 1),
                (67, "30001", "20010", 2),
                (68, "30001", "20010", 3),
                (69, "30001", "20002", 1),
                (70, "30001", "20002", 2),
                (71, "30001", "30100", 1),
                (72, "30001", "30010", 1),
                (73, "30001", "30010", 2),
                (74, "30001", "30002", 1),
                (75, "30001", "30002", 2),
                (76, "30001", "30002", 3),
            ]
        );
    }

    #[test]
    fn missing_unique_gpu_groups_cover_45_columns_in_bounded_order() {
        let groups = missing_unique_gpu_groups();
        assert!(
            groups
                .iter()
                .all(|group| !group.is_empty() && group.len() <= 3)
        );
        assert_eq!(
            groups.iter().flatten().copied().collect::<Vec<_>>(),
            (0..15).chain(23..53).collect::<Vec<_>>()
        );
        for group in groups {
            let columns = full_column_specs();
            assert!(group.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(
                group.iter().all(|ordinal| columns[*ordinal].legal_group_key
                    == columns[group[0]].legal_group_key)
            );
        }
    }

    #[test]
    fn all_non_large_tranche_gpu_groups_cover_53_columns() {
        let groups = missing_gpu_groups();
        assert_eq!(
            groups.iter().flatten().copied().collect::<Vec<_>>(),
            (0..53).collect::<Vec<_>>()
        );
        assert!(
            groups
                .iter()
                .all(|group| !group.is_empty() && group.len() <= 3)
        );
        assert_eq!(
            groups
                .iter()
                .filter(|group| group.iter().any(|ordinal| (15..=22).contains(ordinal)))
                .count(),
            8
        );
    }

    #[test]
    fn all_multiplicity_and_group_domains_are_bound() {
        let columns = full_column_specs();
        let source_intermediate_pairs = columns
            .iter()
            .map(|column| {
                (
                    column.source_dynkin_label.as_str(),
                    column.intermediate_dynkin_label.as_str(),
                )
            })
            .collect::<BTreeSet<_>>();
        let source_incidences = columns
            .iter()
            .map(|column| {
                (
                    column.source_dynkin_label.as_str(),
                    column.source_copy,
                    column.intermediate_dynkin_label.as_str(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            source_intermediate_pairs.len(),
            SOURCE_INTERMEDIATE_PAIR_COUNT
        );
        assert_eq!(source_incidences.len(), SOURCE_INCIDENCE_COUNT);
        assert!(
            columns
                .iter()
                .all(|column| !column.legal_group_key.is_empty())
        );
        assert_eq!(layout_sha256().len(), 64);
    }
}
