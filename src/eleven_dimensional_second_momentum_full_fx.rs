//! Generic exact descendant stream for the missing columns in the full
//! 77-column second-momentum inventory.
//!
//! The established `(20001)` and `(30001)` modules remain byte-compatible.
//! This module supplies the same preflight, PBW-word event, and opaque GPU
//! handle boundary for all 53 non-large-tranche columns. The `(10001)` trace and
//! symmetric-traceless paths use their separately certified integer schedules
//! on the empty PBW word.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;

use crate::eleven_dimensional_second_momentum_full_inventory::{
    FullColumnSpec, Level12FixtureRef, MomentumPath, full_column_specs, level12_fixtures,
};
use crate::eleven_dimensional_second_momentum_full_maps::{
    MissingMapJob, load_verified_checkpoints,
};
use crate::eleven_dimensional_second_momentum_gpu::{GpuFxColumnInput, RecoupledSourceTerm};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FullFxColumnPreflight {
    pub global_column_ordinal: usize,
    pub intermediate_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coefficient_width_bytes: usize,
    pub abstract_certificate_sha256: String,
    pub source_map_sha256: String,
    pub reciprocal_map_sha256: String,
    pub pbw_plan_sha256: String,
    pub pbw_word_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FullFxColumnEvent {
    WordLoweringStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    WordStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    Term {
        requested_word_ordinal: usize,
        term: RecoupledSourceTerm,
    },
    WordEnd {
        requested_word_ordinal: usize,
        raw_terms_emitted: u64,
    },
}

struct PreparedFullFxColumn {
    spec: FullColumnSpec,
    fixture: Level12FixtureRef,
    abstract_certificate: crate::eleven_dimensional_level16_couplings::AbstractCouplingCertificate,
    words: Vec<Vec<u8>>,
    reciprocal_by_word: Vec<Vec<([usize; 2], i64)>>,
    reciprocal_raising_residuals: [usize; 5],
    preflight: FullFxColumnPreflight,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn column_spec(global_ordinal: usize) -> io::Result<FullColumnSpec> {
    let spec = full_column_specs()
        .into_iter()
        .find(|column| column.global_ordinal == global_ordinal)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "full F_X global column ordinal must lie in 0..77",
            )
        })?;
    if !matches!(global_ordinal, 0..=52)
        || !matches!(
            spec.intermediate_dynkin_label.as_str(),
            "00001" | "01001" | "10001" | "11001"
        )
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "generic map-backed F_X stream supports columns 0 through 52",
        ));
    }
    Ok(spec)
}

fn fixture_for(spec: &FullColumnSpec) -> io::Result<Level12FixtureRef> {
    level12_fixtures()
        .into_iter()
        .find(|fixture| {
            fixture.dynkin_label == spec.source_dynkin_label
                && fixture.copy == spec.source_copy
                && fixture.artifact == spec.source_fixture
        })
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "missing full F_X source fixture"))
}

fn requested_words(
    reciprocal: &crate::eleven_dimensional_second_momentum_remaining_recouplings::RemainingRecouplingCertificate,
) -> Vec<Vec<u8>> {
    reciprocal
        .reciprocal_terms
        .iter()
        .filter(|term| term.primitive_coefficient != 0)
        .map(|term| term.intermediate_pbw_word_simple_roots.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn pbw_plan_sha256(target: &str, words: &[Vec<u8>]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-second-momentum-full-pbw-plan-v1\0");
    hash.update((target.len() as u64).to_le_bytes());
    hash.update(target.as_bytes());
    hash.update((words.len() as u64).to_le_bytes());
    for (ordinal, word) in words.iter().enumerate() {
        hash.update((ordinal as u64).to_le_bytes());
        hash.update((word.len() as u64).to_le_bytes());
        hash.update(word);
    }
    format!("{:x}", hash.finalize())
}

fn established_10001_preflight(global_ordinal: usize) -> io::Result<Option<FullFxColumnPreflight>> {
    if !(19..=22).contains(&global_ordinal) {
        return Ok(None);
    }
    let spec = column_spec(global_ordinal)?;
    let fixture = fixture_for(&spec)?;
    let report =
        crate::eleven_dimensional_second_momentum_10001_maps::verify_second_momentum_10001_maps();
    let audit = report
        .embedded_sources
        .iter()
        .find(|audit| audit.source_copy == spec.source_copy)
        .ok_or_else(|| io::Error::other("missing established (10001) source-map audit"))?;
    let path_matches = report.map_specs.iter().any(|candidate| {
        candidate.source_copy == spec.source_copy
            && match spec.momentum_path {
                MomentumPath::Trace => candidate.momentum_path
                    == crate::eleven_dimensional_second_momentum_10001_maps::SecondMomentum10001Path::Trace,
                MomentumPath::SymmetricTraceless => candidate.momentum_path
                    == crate::eleven_dimensional_second_momentum_10001_maps::SecondMomentum10001Path::SymmetricTraceless,
                MomentumPath::Unique => false,
            }
    });
    if !report.passed
        || !audit.passed
        || !path_matches
        || sha256(fixture.bytes) != spec.source_fixture_sha256
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "established (10001) source-map certificate failed",
        ));
    }
    let (_, reciprocal_map_sha256) =
        crate::eleven_dimensional_second_momentum_10001_fx::gpu_path_reciprocal_terms(
            spec.momentum_path,
        )
        .map_err(io::Error::other)?;
    Ok(Some(FullFxColumnPreflight {
        global_column_ordinal: global_ordinal,
        intermediate_dynkin_label: spec.intermediate_dynkin_label,
        source_dynkin_label: spec.source_dynkin_label,
        source_copy: spec.source_copy,
        source_fixture: spec.source_fixture,
        source_fixture_sha256: spec.source_fixture_sha256,
        coefficient_width_bytes: spec.coefficient_width_bytes,
        abstract_certificate_sha256: audit.abstract_coupling_sha256.clone(),
        source_map_sha256: audit.coupled_map_sha256.clone(),
        reciprocal_map_sha256,
        pbw_plan_sha256: pbw_plan_sha256("10001", &[Vec::new()]),
        pbw_word_count: 1,
    }))
}

fn prepare(global_ordinal: usize, map_directory: &Path) -> io::Result<PreparedFullFxColumn> {
    let spec = column_spec(global_ordinal)?;
    let fixture = fixture_for(&spec)?;
    if sha256(fixture.bytes) != spec.source_fixture_sha256
        || fixture.coefficient_width_bytes != spec.coefficient_width_bytes
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "full F_X source fixture identity changed",
        ));
    }
    let job = MissingMapJob {
        target_dynkin_label: spec.intermediate_dynkin_label.clone(),
        source_dynkin_label: spec.source_dynkin_label.clone(),
        source_copy: spec.source_copy,
    };
    let (abstract_checkpoint, embedded_checkpoint) =
        load_verified_checkpoints(map_directory, &job)?;
    let (words, reciprocal_by_word, reciprocal_raising_residuals, reciprocal_map_sha256) =
        if spec.intermediate_dynkin_label == "10001" {
            let (terms, digest) =
                crate::eleven_dimensional_second_momentum_10001_fx::gpu_path_reciprocal_terms(
                    spec.momentum_path,
                )
                .map_err(io::Error::other)?;
            let terms = terms
                .into_iter()
                .map(|(pair, coefficient)| {
                    let coefficient = i64::try_from(coefficient)
                        .map_err(|_| io::Error::other("(10001) path coefficient exceeds i64"))?;
                    Ok(([usize::from(pair[0]), usize::from(pair[1])], coefficient))
                })
                .collect::<io::Result<Vec<_>>>()?;
            (vec![Vec::new()], vec![terms], [0; 5], digest)
        } else {
            if spec.momentum_path != MomentumPath::Unique {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "non-10001 missing column has a non-unique momentum path",
                ));
            }
            let recouplings =
                crate::eleven_dimensional_second_momentum_remaining_recouplings::verify_cached();
            let reciprocal = recouplings
                .channels
                .iter()
                .find(|channel| channel.intermediate_dynkin_label == spec.intermediate_dynkin_label)
                .ok_or_else(|| io::Error::other("missing exact full F_X reciprocal certificate"))?;
            if !reciprocal.passed
                || !reciprocal.exact_chevalley_equivariance_verified
                || reciprocal.reciprocal_raising_residual_terms_by_simple_root != [0; 5]
                || reciprocal.reciprocal_terms.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "uncertified exact full F_X reciprocal map",
                ));
            }
            let words = requested_words(reciprocal);
            if words.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "empty full F_X reciprocal PBW plan",
                ));
            }
            let word_ordinals = words
                .iter()
                .enumerate()
                .map(|(ordinal, word)| (word.clone(), ordinal))
                .collect::<BTreeMap<_, _>>();
            let mut reciprocal_by_word = vec![Vec::new(); words.len()];
            for term in &reciprocal.reciprocal_terms {
                if term.primitive_coefficient == 0 {
                    continue;
                }
                let word_ordinal = *word_ordinals
                    .get(&term.intermediate_pbw_word_simple_roots)
                    .ok_or_else(|| io::Error::other("reciprocal term missing from PBW plan"))?;
                reciprocal_by_word[word_ordinal]
                    .push((term.momentum_pair, term.primitive_coefficient));
            }
            if reciprocal_by_word.iter().any(|terms| terms.is_empty()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "incomplete full F_X reciprocal PBW plan",
                ));
            }
            (
                words,
                reciprocal_by_word,
                reciprocal.reciprocal_raising_residual_terms_by_simple_root,
                reciprocal.certificate_sha256.clone(),
            )
        };
    let preflight = FullFxColumnPreflight {
        global_column_ordinal: spec.global_ordinal,
        intermediate_dynkin_label: spec.intermediate_dynkin_label.clone(),
        source_dynkin_label: spec.source_dynkin_label.clone(),
        source_copy: spec.source_copy,
        source_fixture: spec.source_fixture.clone(),
        source_fixture_sha256: spec.source_fixture_sha256.clone(),
        coefficient_width_bytes: spec.coefficient_width_bytes,
        abstract_certificate_sha256: abstract_checkpoint.certificate_sha256.clone(),
        source_map_sha256: embedded_checkpoint.coupled_map_sha256.clone(),
        reciprocal_map_sha256,
        pbw_plan_sha256: pbw_plan_sha256(&spec.intermediate_dynkin_label, &words),
        pbw_word_count: words.len(),
    };
    Ok(PreparedFullFxColumn {
        spec,
        fixture,
        abstract_certificate: abstract_checkpoint.certificate,
        words,
        reciprocal_by_word,
        reciprocal_raising_residuals,
        preflight,
    })
}

pub(crate) fn gpu_column_preflight(
    global_ordinal: usize,
    map_directory: &Path,
) -> io::Result<FullFxColumnPreflight> {
    if let Some(preflight) = established_10001_preflight(global_ordinal)? {
        return Ok(preflight);
    }
    if (53..=61).contains(&global_ordinal) {
        let coefficient_width_bytes = full_column_specs()[global_ordinal].coefficient_width_bytes;
        let value = crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(
            global_ordinal - 53,
        )?;
        return Ok(FullFxColumnPreflight {
            global_column_ordinal: value.global_column_ordinal,
            intermediate_dynkin_label: value.tranche,
            source_dynkin_label: value.source_dynkin_label,
            source_copy: value.source_copy,
            source_fixture: value.source_fixture,
            source_fixture_sha256: value.source_fixture_sha256,
            coefficient_width_bytes,
            abstract_certificate_sha256: value.abstract_certificate_sha256,
            source_map_sha256: value.source_map_sha256,
            reciprocal_map_sha256: value.reciprocal_map_sha256,
            pbw_plan_sha256: value.pbw_plan_sha256,
            pbw_word_count: value.pbw_word_count,
        });
    }
    if (62..=76).contains(&global_ordinal) {
        let coefficient_width_bytes = full_column_specs()[global_ordinal].coefficient_width_bytes;
        let value = crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
            global_ordinal - 62,
        )?;
        return Ok(FullFxColumnPreflight {
            global_column_ordinal: value.global_column_ordinal,
            intermediate_dynkin_label: value.tranche,
            source_dynkin_label: value.source_dynkin_label,
            source_copy: value.source_copy,
            source_fixture: value.source_fixture,
            source_fixture_sha256: value.source_fixture_sha256,
            coefficient_width_bytes,
            abstract_certificate_sha256: value.abstract_certificate_sha256,
            source_map_sha256: value.source_map_sha256,
            reciprocal_map_sha256: value.reciprocal_map_sha256,
            pbw_plan_sha256: value.pbw_plan_sha256,
            pbw_word_count: value.pbw_word_count,
        });
    }
    Ok(prepare(global_ordinal, map_directory)?.preflight)
}

pub(crate) fn visit_gpu_column_contribution_events_range_from_handles<H, U, L, D, F>(
    expected_preflight: &FullFxColumnPreflight,
    map_directory: &Path,
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    upload_highest: U,
    lower_word: L,
    mut download_terms: D,
    mut visit: F,
) -> io::Result<GpuFxColumnInput>
where
    U: FnOnce(
        &crate::eleven_dimensional_level16_couplings::CanonicalSparseHighest64,
    ) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    D: FnMut(&H, &mut dyn FnMut(u64, i64) -> io::Result<()>) -> io::Result<u64>,
    F: FnMut(FullFxColumnEvent) -> io::Result<()>,
{
    if (19..=22).contains(&expected_preflight.global_column_ordinal) {
        if established_10001_preflight(expected_preflight.global_column_ordinal)?
            != Some(expected_preflight.clone())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "established (10001) preflight identity changed",
            ));
        }
        if start_word_ordinal > end_word_ordinal_exclusive || end_word_ordinal_exclusive > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "established (10001) word range is invalid",
            ));
        }
        if start_word_ordinal == end_word_ordinal_exclusive {
            return Ok(GpuFxColumnInput {
                global_ordinal: expected_preflight.global_column_ordinal,
                source_label: expected_preflight.source_dynkin_label.clone(),
                source_copy: expected_preflight.source_copy,
                terms: Vec::new(),
                raising_residuals: [0; 5],
            });
        }
        visit(FullFxColumnEvent::WordLoweringStart {
            requested_word_ordinal: 0,
            pbw_word_simple_roots: Vec::new(),
        })?;
        let mut source_terms = Vec::new();
        crate::eleven_dimensional_second_momentum_10001_maps::
            visit_second_momentum_10001_highest_weight_source_terms(
                expected_preflight.source_copy,
                |term| source_terms.push(term),
            );
        let highest =
            crate::eleven_dimensional_level16_couplings::canonical_sparse_highest64_from_terms(
                source_terms
                    .into_iter()
                    .map(|term| {
                        Ok((
                            term.intermediate_spinor_weight_index,
                            term.exterior_mask,
                            i64::try_from(term.coefficient).map_err(|_| {
                                io::Error::other(
                                    "established (10001) source coefficient exceeds i64",
                                )
                            })?,
                        ))
                    })
                    .collect::<io::Result<Vec<_>>>()?,
            )?;
        let handle = upload_highest(&highest)?;
        visit(FullFxColumnEvent::WordStart {
            requested_word_ordinal: 0,
            pbw_word_simple_roots: Vec::new(),
        })?;
        let spec = column_spec(expected_preflight.global_column_ordinal)?;
        let (path_terms, _) =
            crate::eleven_dimensional_second_momentum_10001_fx::gpu_path_reciprocal_terms(
                spec.momentum_path,
            )
            .map_err(io::Error::other)?;
        let mut raw_terms_emitted = 0_u64;
        download_terms(&handle, &mut |key, source_coefficient| {
            let free_spinor = u8::try_from(key >> 32)
                .map_err(|_| io::Error::other("established (10001) spinor exceeds u8"))?;
            let exterior_mask = key as u32;
            for &(momentum_pair, path_coefficient) in &path_terms {
                visit(FullFxColumnEvent::Term {
                    requested_word_ordinal: 0,
                    term: RecoupledSourceTerm {
                        momentum_pair,
                        free_spinor,
                        exterior_mask,
                        coefficient: i128::from(source_coefficient)
                            .checked_mul(path_coefficient)
                            .ok_or_else(|| {
                                io::Error::other("established (10001) coefficient overflow")
                            })?,
                    },
                })?;
                raw_terms_emitted = raw_terms_emitted
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("established (10001) term count overflow"))?;
            }
            Ok(())
        })?;
        visit(FullFxColumnEvent::WordEnd {
            requested_word_ordinal: 0,
            raw_terms_emitted,
        })?;
        return Ok(GpuFxColumnInput {
            global_ordinal: expected_preflight.global_column_ordinal,
            source_label: expected_preflight.source_dynkin_label.clone(),
            source_copy: expected_preflight.source_copy,
            terms: Vec::new(),
            raising_residuals: [0; 5],
        });
    }
    if expected_preflight.intermediate_dynkin_label == "20001" {
        if !(53..=61).contains(&expected_preflight.global_column_ordinal) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "full compatibility (20001) ordinal is out of range",
            ));
        }
        let legacy = crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(
            expected_preflight.global_column_ordinal - 53,
        )?;
        if gpu_column_preflight(expected_preflight.global_column_ordinal, map_directory)?
            != *expected_preflight
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "full compatibility (20001) preflight identity changed",
            ));
        }
        return crate::eleven_dimensional_second_momentum_20001_fx::
            visit_gpu_column_contribution_events_range_from_handles(
                &legacy,
                start_word_ordinal,
                end_word_ordinal_exclusive,
                upload_highest,
                lower_word,
                download_terms,
                |event| {
                    use crate::eleven_dimensional_second_momentum_20001_fx::
                        SecondMomentum20001GpuColumnEvent as LegacyEvent;
                    visit(match event {
                        LegacyEvent::WordLoweringStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        } => FullFxColumnEvent::WordLoweringStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        },
                        LegacyEvent::WordStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        } => FullFxColumnEvent::WordStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        },
                        LegacyEvent::Term {
                            requested_word_ordinal,
                            term,
                        } => FullFxColumnEvent::Term {
                            requested_word_ordinal,
                            term,
                        },
                        LegacyEvent::WordEnd {
                            requested_word_ordinal,
                            raw_terms_emitted,
                        } => FullFxColumnEvent::WordEnd {
                            requested_word_ordinal,
                            raw_terms_emitted,
                        },
                    })
                },
            );
    }
    if expected_preflight.intermediate_dynkin_label == "30001" {
        if !(62..=76).contains(&expected_preflight.global_column_ordinal) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "full compatibility (30001) ordinal is out of range",
            ));
        }
        let legacy = crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
            expected_preflight.global_column_ordinal - 62,
        )?;
        if gpu_column_preflight(expected_preflight.global_column_ordinal, map_directory)?
            != *expected_preflight
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "full compatibility (30001) preflight identity changed",
            ));
        }
        return crate::eleven_dimensional_second_momentum_30001_fx::
            visit_gpu_column_contribution_events_range_from_handles(
                &legacy,
                start_word_ordinal,
                end_word_ordinal_exclusive,
                upload_highest,
                lower_word,
                download_terms,
                |event| {
                    use crate::eleven_dimensional_second_momentum_30001_fx::
                        SecondMomentum30001GpuColumnEvent as LegacyEvent;
                    visit(match event {
                        LegacyEvent::WordLoweringStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        } => FullFxColumnEvent::WordLoweringStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        },
                        LegacyEvent::WordStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        } => FullFxColumnEvent::WordStart {
                            requested_word_ordinal,
                            pbw_word_simple_roots,
                        },
                        LegacyEvent::Term {
                            requested_word_ordinal,
                            term,
                        } => FullFxColumnEvent::Term {
                            requested_word_ordinal,
                            term,
                        },
                        LegacyEvent::WordEnd {
                            requested_word_ordinal,
                            raw_terms_emitted,
                        } => FullFxColumnEvent::WordEnd {
                            requested_word_ordinal,
                            raw_terms_emitted,
                        },
                    })
                },
            );
    }
    if start_word_ordinal > end_word_ordinal_exclusive
        || end_word_ordinal_exclusive > expected_preflight.pbw_word_count
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "full F_X word range exceeds the preflight PBW plan",
        ));
    }
    let prepared = prepare(expected_preflight.global_column_ordinal, map_directory)?;
    if prepared.preflight != *expected_preflight {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "full F_X preflight identity changed before opaque streaming",
        ));
    }
    let mut observed_by_word = vec![false; prepared.words.len()];
    let mut emitted_terms = 0_u64;
    let mut completed_word_terms = 0_u64;
    let mut current_word_terms = 0_u64;
    let mut current_word_components = 0_u64;
    let accounting = crate::eleven_dimensional_level16_couplings::
        visit_second_momentum_descendant_handles_range(
            &prepared.spec.intermediate_dynkin_label,
            &prepared.abstract_certificate,
            prepared.fixture.copy,
            prepared.fixture.artifact,
            prepared.fixture.coefficient_width_bytes,
            prepared.fixture.bytes,
            &prepared.preflight.source_fixture_sha256,
            &prepared.preflight.source_map_sha256,
            &prepared.words,
            start_word_ordinal,
            end_word_ordinal_exclusive,
            upload_highest,
            lower_word,
            |event| match event {
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::
                    WordLoweringStart { ordinal, pbw_word } => {
                        visit(FullFxColumnEvent::WordLoweringStart {
                            requested_word_ordinal: ordinal,
                            pbw_word_simple_roots: pbw_word.to_vec(),
                        })?;
                        Ok(0)
                    }
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::WordStart {
                    ordinal,
                    pbw_word,
                } => {
                    current_word_terms = 0;
                    current_word_components = 0;
                    visit(FullFxColumnEvent::WordStart {
                        requested_word_ordinal: ordinal,
                        pbw_word_simple_roots: pbw_word.to_vec(),
                    })?;
                    Ok(0)
                }
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::State {
                    ordinal,
                    state,
                } => {
                    let mut previous_key = None;
                    let reported = download_terms(state, &mut |key, descendant_coefficient| {
                        let free_spinor = usize::try_from(key >> 32).map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "downloaded full F_X free-spinor key exceeds host range",
                            )
                        })?;
                        let exterior_mask = key as u32;
                        if free_spinor >= 32
                            || exterior_mask.count_ones() != 12
                            || descendant_coefficient == 0
                            || previous_key.is_some_and(|previous| previous >= key)
                        {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "downloaded full F_X terminal state is not canonical",
                            ));
                        }
                        previous_key = Some(key);
                        current_word_components = current_word_components.checked_add(1).ok_or_else(
                            || io::Error::other("full F_X descendant component count overflow"),
                        )?;
                        for &(momentum_pair, primitive_coefficient) in
                            &prepared.reciprocal_by_word[ordinal]
                        {
                            let coefficient = i128::from(descendant_coefficient)
                                .checked_mul(i128::from(primitive_coefficient))
                                .ok_or_else(|| {
                                    io::Error::other("full F_X recoupling coefficient overflow")
                                })?;
                            if coefficient == 0 {
                                continue;
                            }
                            let term = RecoupledSourceTerm {
                                momentum_pair: [
                                    u8::try_from(momentum_pair[0]).map_err(|_| {
                                        io::Error::other("momentum index exceeds packed range")
                                    })?,
                                    u8::try_from(momentum_pair[1]).map_err(|_| {
                                        io::Error::other("momentum index exceeds packed range")
                                    })?,
                                ],
                                free_spinor: u8::try_from(free_spinor).map_err(|_| {
                                    io::Error::other("spinor index exceeds packed range")
                                })?,
                                exterior_mask,
                                coefficient,
                            };
                            visit(FullFxColumnEvent::Term {
                                requested_word_ordinal: ordinal,
                                term,
                            })?;
                            current_word_terms = current_word_terms.checked_add(1).ok_or_else(|| {
                                io::Error::other("full F_X word contribution count overflow")
                            })?;
                            emitted_terms = emitted_terms.checked_add(1).ok_or_else(|| {
                                io::Error::other("full F_X contribution count overflow")
                            })?;
                        }
                        Ok(())
                    })?;
                    if reported != current_word_components {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "downloaded full F_X terminal count disagrees with emitted terms",
                        ));
                    }
                    Ok(reported)
                }
                crate::eleven_dimensional_level16_couplings::CoupledWordStateEvent::WordEnd {
                    ordinal,
                } => {
                    let observed = observed_by_word.get_mut(ordinal).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "full F_X word out of range")
                    })?;
                    if *observed {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "full F_X word completed more than once",
                        ));
                    }
                    *observed = true;
                    completed_word_terms = completed_word_terms
                        .checked_add(current_word_terms)
                        .ok_or_else(|| io::Error::other("full F_X completed count overflow"))?;
                    visit(FullFxColumnEvent::WordEnd {
                        requested_word_ordinal: ordinal,
                        raw_terms_emitted: current_word_terms,
                    })?;
                    Ok(0)
                }
            },
        )?;
    if accounting.requested_pbw_words != end_word_ordinal_exclusive - start_word_ordinal
        || accounting.source_dynkin_label != prepared.preflight.source_dynkin_label
        || accounting.source_copy != prepared.preflight.source_copy
        || accounting.source_fixture != prepared.preflight.source_fixture
        || accounting.source_fixture_sha256 != prepared.preflight.source_fixture_sha256
        || accounting.coupled_map_sha256 != prepared.preflight.source_map_sha256
        || !accounting.checkpoint_hash_parity_verified
        || completed_word_terms != emitted_terms
        || observed_by_word
            .iter()
            .skip(start_word_ordinal)
            .take(end_word_ordinal_exclusive - start_word_ordinal)
            .any(|observed| !observed)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "full F_X opaque descendant accounting changed or is incomplete",
        ));
    }
    Ok(GpuFxColumnInput {
        global_ordinal: prepared.preflight.global_column_ordinal,
        source_label: prepared.preflight.source_dynkin_label,
        source_copy: prepared.preflight.source_copy,
        terms: Vec::new(),
        raising_residuals: prepared.reciprocal_raising_residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_backed_boundary_is_exactly_the_non_large_tranche_columns() {
        let ordinals = full_column_specs()
            .into_iter()
            .filter(|column| column_spec(column.global_ordinal).is_ok())
            .map(|column| column.global_ordinal)
            .collect::<Vec<_>>();
        assert_eq!(ordinals.len(), 53);
        assert_eq!(ordinals[..3], [0, 1, 2]);
        assert_eq!(ordinals[3..15], (3..15).collect::<Vec<_>>());
        assert_eq!(ordinals[15..19], [15, 16, 17, 18]);
        assert_eq!(ordinals, (0..53).collect::<Vec<_>>());
    }

    #[test]
    fn map_backed_boundary_rejects_the_large_established_slices() {
        for ordinal in 53..77 {
            assert!(column_spec(ordinal).is_err());
        }
    }

    #[test]
    fn established_10001_column_streams_without_external_map_files() {
        let unused = Path::new("deliberately-absent-full-map-directory");
        let preflight = gpu_column_preflight(19, unused).unwrap();
        assert_eq!(preflight.global_column_ordinal, 19);
        assert_eq!(preflight.pbw_word_count, 1);
        let mut events = Vec::new();
        let result = visit_gpu_column_contribution_events_range_from_handles(
            &preflight,
            unused,
            0,
            1,
            |highest| {
                let mut terms = Vec::new();
                highest.visit_terms(|key, coefficient| {
                    terms.push((key, coefficient));
                    Ok(())
                })?;
                Ok(terms)
            },
            |_, _, _| panic!("the established empty-word path must not lower"),
            |terms, visit| {
                for &(key, coefficient) in terms {
                    visit(key, coefficient)?;
                }
                Ok(terms.len() as u64)
            },
            |event| {
                events.push(event);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result.global_ordinal, 19);
        assert_eq!(result.raising_residuals, [0; 5]);
        assert!(matches!(
            events.first(),
            Some(FullFxColumnEvent::WordLoweringStart {
                requested_word_ordinal: 0,
                pbw_word_simple_roots
            }) if pbw_word_simple_roots.is_empty()
        ));
        let emitted = events
            .iter()
            .filter(|event| matches!(event, FullFxColumnEvent::Term { .. }))
            .count();
        assert!(emitted > 0);
        assert!(matches!(
            events.last(),
            Some(FullFxColumnEvent::WordEnd {
                requested_word_ordinal: 0,
                raw_terms_emitted
            }) if *raw_terms_emitted == emitted as u64
        ));
    }

    #[test]
    fn unified_preflight_reproduces_established_20001_and_30001_identities() {
        let unused = Path::new("unused-for-established-columns");
        let full_20001 = gpu_column_preflight(53, unused).unwrap();
        let legacy_20001 =
            crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(0).unwrap();
        assert_eq!(full_20001.global_column_ordinal, 53);
        assert_eq!(full_20001.intermediate_dynkin_label, legacy_20001.tranche);
        assert_eq!(
            full_20001.source_fixture_sha256,
            legacy_20001.source_fixture_sha256
        );
        assert_eq!(full_20001.source_map_sha256, legacy_20001.source_map_sha256);
        assert_eq!(full_20001.pbw_plan_sha256, legacy_20001.pbw_plan_sha256);

        let full_30001 = gpu_column_preflight(76, unused).unwrap();
        let legacy_30001 =
            crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(14).unwrap();
        assert_eq!(full_30001.global_column_ordinal, 76);
        assert_eq!(full_30001.intermediate_dynkin_label, legacy_30001.tranche);
        assert_eq!(
            full_30001.source_fixture_sha256,
            legacy_30001.source_fixture_sha256
        );
        assert_eq!(full_30001.source_map_sha256, legacy_30001.source_map_sha256);
        assert_eq!(full_30001.pbw_plan_sha256, legacy_30001.pbw_plan_sha256);
    }
}
