//! Exact host boundary for grouped second-momentum GPU columns.
//!
//! This module owns group legality, proof identity, deterministic exact
//! per-lane reduction, bounded k-way union batches, and independent row
//! folding. It deliberately stops before the CUDA wrapper. Raw per-column
//! word hashes must be updated in caller order before terms enter reduction.

use std::io;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_second_momentum_gpu::{
    GPU_FX_PRIMES, GaussianResidue, RecoupledSourceTerm,
};

pub(crate) const GPU_FX_GROUP_SCHEMA: &str = "adynkra-11d-second-momentum-gpu-group-v1";
pub(crate) const MAX_PRODUCTION_GROUP_WIDTH: usize = 3;
pub(crate) const MAX_TESTED_GROUP_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum GpuFxTranche {
    #[serde(rename = "20001")]
    Two0001,
    #[serde(rename = "30001")]
    Three0001,
}

impl GpuFxTranche {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "20001" => Ok(Self::Two0001),
            "30001" => Ok(Self::Three0001),
            _ => Err("group tranche must be 20001 or 30001".to_string()),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Two0001 => "20001",
            Self::Three0001 => "30001",
        }
    }

    fn legal_local_groups(self) -> Vec<Vec<usize>> {
        match self {
            Self::Two0001 => {
                crate::eleven_dimensional_second_momentum_20001_fx::gpu_legal_local_column_groups()
            }
            Self::Three0001 => {
                crate::eleven_dimensional_second_momentum_30001_fx::gpu_legal_local_column_groups()
            }
        }
    }
}

pub(crate) fn discover_legal_cuda_column_groups(tranche: GpuFxTranche) -> Vec<Vec<usize>> {
    tranche.legal_local_groups()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupRuntimeIdentity {
    pub prime: u32,
    pub static_semantic_sha256: String,
    pub flat_plan_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupColumnIdentity {
    pub local_ordinal: usize,
    pub global_ordinal: usize,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub abstract_certificate_sha256: String,
    pub source_map_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedColumnGroup {
    pub schema_version: String,
    pub group_id: String,
    pub tranche: String,
    pub source_dynkin_label: String,
    pub ordered_local_ordinals: Vec<usize>,
    pub ordered_global_ordinals: Vec<usize>,
    pub ordered_source_copies: Vec<usize>,
    pub members: Vec<GroupColumnIdentity>,
    pub pbw_plan_sha256: String,
    pub pbw_word_count: usize,
    pub reciprocal_map_sha256: String,
    pub runtime: GroupRuntimeIdentity,
    pub active_columns: usize,
    pub singleton_fallback: bool,
}

/// Prime-independent identity for one exact source traversal. Plans with this
/// digest can share PBW lowering, exact term hashing, lane reduction, and key
/// union while retaining independent modular contraction contexts.
pub(crate) fn source_group_identity_sha256(group: &PreparedColumnGroup) -> String {
    fn update_bytes(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    let mut hash = Sha256::new();
    update_bytes(&mut hash, GPU_FX_GROUP_SCHEMA.as_bytes());
    update_bytes(&mut hash, b"source-group-v1");
    update_bytes(&mut hash, group.tranche.as_bytes());
    update_bytes(&mut hash, group.source_dynkin_label.as_bytes());
    hash.update((group.ordered_local_ordinals.len() as u64).to_le_bytes());
    for value in &group.ordered_local_ordinals {
        hash.update((*value as u64).to_le_bytes());
    }
    hash.update((group.ordered_global_ordinals.len() as u64).to_le_bytes());
    for value in &group.ordered_global_ordinals {
        hash.update((*value as u64).to_le_bytes());
    }
    hash.update((group.ordered_source_copies.len() as u64).to_le_bytes());
    for value in &group.ordered_source_copies {
        hash.update((*value as u64).to_le_bytes());
    }
    hash.update((group.members.len() as u64).to_le_bytes());
    for member in &group.members {
        hash.update((member.local_ordinal as u64).to_le_bytes());
        hash.update((member.global_ordinal as u64).to_le_bytes());
        hash.update((member.source_copy as u64).to_le_bytes());
        update_bytes(&mut hash, member.source_fixture.as_bytes());
        update_bytes(&mut hash, member.source_fixture_sha256.as_bytes());
        update_bytes(&mut hash, member.abstract_certificate_sha256.as_bytes());
        update_bytes(&mut hash, member.source_map_sha256.as_bytes());
    }
    update_bytes(&mut hash, group.pbw_plan_sha256.as_bytes());
    hash.update((group.pbw_word_count as u64).to_le_bytes());
    update_bytes(&mut hash, group.reciprocal_map_sha256.as_bytes());
    hash.update((group.active_columns as u64).to_le_bytes());
    hash.update([u8::from(group.singleton_fallback)]);
    format!("{:x}", hash.finalize())
}

pub(crate) fn multi_prime_group_identity_sha256(
    groups: &[PreparedColumnGroup],
) -> Result<String, String> {
    let first = groups
        .first()
        .ok_or_else(|| "multi-prime group has no plans".to_string())?;
    let source_identity = source_group_identity_sha256(first);
    let mut previous_prime_index = None;
    for group in groups {
        let prime_index = GPU_FX_PRIMES
            .iter()
            .position(|prime| *prime == group.runtime.prime)
            .ok_or_else(|| "multi-prime plan uses an unpinned prime".to_string())?;
        if source_group_identity_sha256(group) != source_identity
            || previous_prime_index.is_some_and(|previous| previous >= prime_index)
        {
            return Err(
                "multi-prime plans do not share one source identity or prime order".to_string(),
            );
        }
        previous_prime_index = Some(prime_index);
    }
    let mut hash = Sha256::new();
    hash.update((GPU_FX_GROUP_SCHEMA.len() as u64).to_le_bytes());
    hash.update(GPU_FX_GROUP_SCHEMA.as_bytes());
    hash.update(("multi-prime-group-v1".len() as u64).to_le_bytes());
    hash.update(b"multi-prime-group-v1");
    hash.update((source_identity.len() as u64).to_le_bytes());
    hash.update(source_identity.as_bytes());
    hash.update((groups.len() as u64).to_le_bytes());
    for group in groups {
        let prime_index = GPU_FX_PRIMES
            .iter()
            .position(|prime| *prime == group.runtime.prime)
            .expect("validated pinned prime");
        hash.update((prime_index as u64).to_le_bytes());
        hash.update(group.runtime.prime.to_le_bytes());
        hash.update((group.group_id.len() as u64).to_le_bytes());
        hash.update(group.group_id.as_bytes());
        hash.update((group.runtime.static_semantic_sha256.len() as u64).to_le_bytes());
        hash.update(group.runtime.static_semantic_sha256.as_bytes());
        hash.update((group.runtime.flat_plan_sha256.len() as u64).to_le_bytes());
        hash.update(group.runtime.flat_plan_sha256.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Clone, Debug)]
struct CommonPreflight {
    tranche: String,
    local_ordinal: usize,
    global_ordinal: usize,
    source_label: String,
    source_copy: usize,
    source_fixture: String,
    source_fixture_sha256: String,
    abstract_certificate_sha256: String,
    source_map_sha256: String,
    reciprocal_map_sha256: String,
    pbw_plan_sha256: String,
    pbw_word_count: usize,
}

/// Perform every member preflight before a CUDA context or union workspace is
/// allocated. The requested ordinals must equal one maximal legal same-label
/// group in canonical local-column order.
pub(crate) fn prepare_cuda_column_group(
    tranche: GpuFxTranche,
    local_ordinals: &[usize],
    runtime: GroupRuntimeIdentity,
) -> Result<PreparedColumnGroup, String> {
    validate_requested_group(tranche, local_ordinals, &runtime)?;
    let mut preflights = Vec::with_capacity(local_ordinals.len());
    for &local_ordinal in local_ordinals {
        let preflight = match tranche {
            GpuFxTranche::Two0001 => {
                let value =
                    crate::eleven_dimensional_second_momentum_20001_fx::gpu_column_preflight(
                        local_ordinal,
                    )
                    .map_err(|error| error.to_string())?;
                CommonPreflight {
                    tranche: value.tranche,
                    local_ordinal: value.local_column_ordinal,
                    global_ordinal: value.global_column_ordinal,
                    source_label: value.source_dynkin_label,
                    source_copy: value.source_copy,
                    source_fixture: value.source_fixture,
                    source_fixture_sha256: value.source_fixture_sha256,
                    abstract_certificate_sha256: value.abstract_certificate_sha256,
                    source_map_sha256: value.source_map_sha256,
                    reciprocal_map_sha256: value.reciprocal_map_sha256,
                    pbw_plan_sha256: value.pbw_plan_sha256,
                    pbw_word_count: value.pbw_word_count,
                }
            }
            GpuFxTranche::Three0001 => {
                let value =
                    crate::eleven_dimensional_second_momentum_30001_fx::gpu_column_preflight(
                        local_ordinal,
                    )
                    .map_err(|error| error.to_string())?;
                CommonPreflight {
                    tranche: value.tranche,
                    local_ordinal: value.local_column_ordinal,
                    global_ordinal: value.global_column_ordinal,
                    source_label: value.source_dynkin_label,
                    source_copy: value.source_copy,
                    source_fixture: value.source_fixture,
                    source_fixture_sha256: value.source_fixture_sha256,
                    abstract_certificate_sha256: value.abstract_certificate_sha256,
                    source_map_sha256: value.source_map_sha256,
                    reciprocal_map_sha256: value.reciprocal_map_sha256,
                    pbw_plan_sha256: value.pbw_plan_sha256,
                    pbw_word_count: value.pbw_word_count,
                }
            }
        };
        preflights.push(preflight);
    }
    build_group_identity(tranche, preflights, runtime)
}

/// Prepare one canonical group from the 45 unique-path columns outside the
/// original 28-column slice. Global ordinals are also used as stable worker
/// ordinals so assignments remain portable across machines.
pub(crate) fn prepare_full_cuda_column_group(
    global_ordinals: &[usize],
    map_directory: &std::path::Path,
    runtime: GroupRuntimeIdentity,
) -> Result<PreparedColumnGroup, String> {
    if global_ordinals.is_empty()
        || global_ordinals.len() > MAX_PRODUCTION_GROUP_WIDTH
        || global_ordinals.windows(2).any(|pair| pair[0] >= pair[1])
        || !crate::eleven_dimensional_second_momentum_full_inventory::missing_gpu_groups()
            .iter()
            .any(|group| group == global_ordinals)
    {
        return Err(
            "requested full-inventory columns are not one canonical width-1..3 group".to_string(),
        );
    }
    if !GPU_FX_PRIMES.contains(&runtime.prime) {
        return Err("group prime is not one of the pinned exact primes".to_string());
    }
    validate_sha256("static semantic", &runtime.static_semantic_sha256)?;
    validate_sha256("flat plan", &runtime.flat_plan_sha256)?;

    let columns = crate::eleven_dimensional_second_momentum_full_inventory::full_column_specs();
    let first_spec = &columns[global_ordinals[0]];
    let mut preflights = Vec::with_capacity(global_ordinals.len());
    for &global_ordinal in global_ordinals {
        let spec = columns
            .get(global_ordinal)
            .ok_or_else(|| "full-inventory group ordinal is out of range".to_string())?;
        if spec.legal_group_key != first_spec.legal_group_key
            || spec.intermediate_dynkin_label != first_spec.intermediate_dynkin_label
            || spec.source_dynkin_label != first_spec.source_dynkin_label
        {
            return Err("full-inventory group crossed a certified lane boundary".to_string());
        }
        let value = crate::eleven_dimensional_second_momentum_full_fx::gpu_column_preflight(
            global_ordinal,
            map_directory,
        )
        .map_err(|error| error.to_string())?;
        preflights.push(CommonPreflight {
            tranche: value.intermediate_dynkin_label,
            local_ordinal: value.global_column_ordinal,
            global_ordinal: value.global_column_ordinal,
            source_label: value.source_dynkin_label,
            source_copy: value.source_copy,
            source_fixture: value.source_fixture,
            source_fixture_sha256: value.source_fixture_sha256,
            abstract_certificate_sha256: value.abstract_certificate_sha256,
            source_map_sha256: value.source_map_sha256,
            reciprocal_map_sha256: value.reciprocal_map_sha256,
            pbw_plan_sha256: value.pbw_plan_sha256,
            pbw_word_count: value.pbw_word_count,
        });
    }
    let first = &preflights[0];
    for member in &preflights {
        if member.tranche != first.tranche
            || member.source_label != first.source_label
            || member.abstract_certificate_sha256 != first.abstract_certificate_sha256
            || member.reciprocal_map_sha256 != first.reciprocal_map_sha256
            || member.pbw_plan_sha256 != first.pbw_plan_sha256
            || member.pbw_word_count != first.pbw_word_count
        {
            return Err("full-inventory group preflight identities are incompatible".to_string());
        }
        for (label, digest) in [
            ("fixture", member.source_fixture_sha256.as_str()),
            (
                "abstract certificate",
                member.abstract_certificate_sha256.as_str(),
            ),
            ("source map", member.source_map_sha256.as_str()),
            ("reciprocal map", member.reciprocal_map_sha256.as_str()),
            ("PBW plan", member.pbw_plan_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
    }
    if preflights
        .windows(2)
        .any(|pair| pair[0].source_copy >= pair[1].source_copy)
    {
        return Err("full-inventory group source-copy order changed".to_string());
    }
    let members = preflights
        .iter()
        .map(|member| GroupColumnIdentity {
            local_ordinal: member.local_ordinal,
            global_ordinal: member.global_ordinal,
            source_copy: member.source_copy,
            source_fixture: member.source_fixture.clone(),
            source_fixture_sha256: member.source_fixture_sha256.clone(),
            abstract_certificate_sha256: member.abstract_certificate_sha256.clone(),
            source_map_sha256: member.source_map_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let mut group = PreparedColumnGroup {
        schema_version: GPU_FX_GROUP_SCHEMA.to_string(),
        group_id: String::new(),
        tranche: first.tranche.clone(),
        source_dynkin_label: first.source_label.clone(),
        ordered_local_ordinals: members.iter().map(|member| member.local_ordinal).collect(),
        ordered_global_ordinals: members.iter().map(|member| member.global_ordinal).collect(),
        ordered_source_copies: members.iter().map(|member| member.source_copy).collect(),
        members,
        pbw_plan_sha256: first.pbw_plan_sha256.clone(),
        pbw_word_count: first.pbw_word_count,
        reciprocal_map_sha256: first.reciprocal_map_sha256.clone(),
        runtime,
        active_columns: preflights.len(),
        singleton_fallback: preflights.len() == 1,
    };
    group.group_id = group_identity_sha256(&group);
    Ok(group)
}

fn validate_requested_group(
    tranche: GpuFxTranche,
    local_ordinals: &[usize],
    runtime: &GroupRuntimeIdentity,
) -> Result<(), String> {
    if local_ordinals.is_empty() || local_ordinals.len() > MAX_PRODUCTION_GROUP_WIDTH {
        return Err("group width must be in 1..=3".to_string());
    }
    if local_ordinals.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("group local ordinals must be strictly increasing".to_string());
    }
    if !tranche
        .legal_local_groups()
        .iter()
        .any(|legal| legal == local_ordinals)
    {
        return Err("requested columns are not one certified maximal same-label group".to_string());
    }
    if !GPU_FX_PRIMES.contains(&runtime.prime) {
        return Err("group prime is not one of the pinned exact primes".to_string());
    }
    validate_sha256("static semantic", &runtime.static_semantic_sha256)?;
    validate_sha256("flat plan", &runtime.flat_plan_sha256)
}

fn build_group_identity(
    tranche: GpuFxTranche,
    preflights: Vec<CommonPreflight>,
    runtime: GroupRuntimeIdentity,
) -> Result<PreparedColumnGroup, String> {
    let first = preflights
        .first()
        .ok_or_else(|| "group preflight is empty".to_string())?;
    let requested = preflights
        .iter()
        .map(|member| member.local_ordinal)
        .collect::<Vec<_>>();
    if !tranche
        .legal_local_groups()
        .iter()
        .any(|legal| legal == &requested)
    {
        return Err("preflight returned a noncanonical group lane order".to_string());
    }
    for member in &preflights {
        if member.tranche != tranche.as_str()
            || member.source_label != first.source_label
            || member.abstract_certificate_sha256 != first.abstract_certificate_sha256
            || member.reciprocal_map_sha256 != first.reciprocal_map_sha256
            || member.pbw_plan_sha256 != first.pbw_plan_sha256
            || member.pbw_word_count != first.pbw_word_count
        {
            return Err("group member preflight identities are incompatible".to_string());
        }
        for (label, digest) in [
            ("fixture", member.source_fixture_sha256.as_str()),
            (
                "abstract certificate",
                member.abstract_certificate_sha256.as_str(),
            ),
            ("source map", member.source_map_sha256.as_str()),
            ("reciprocal map", member.reciprocal_map_sha256.as_str()),
            ("PBW plan", member.pbw_plan_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
    }
    if preflights
        .windows(2)
        .any(|pair| pair[0].global_ordinal >= pair[1].global_ordinal)
        || preflights
            .windows(2)
            .any(|pair| pair[0].source_copy >= pair[1].source_copy)
    {
        return Err("group lane order or source-copy order changed".to_string());
    }

    let members = preflights
        .iter()
        .map(|member| GroupColumnIdentity {
            local_ordinal: member.local_ordinal,
            global_ordinal: member.global_ordinal,
            source_copy: member.source_copy,
            source_fixture: member.source_fixture.clone(),
            source_fixture_sha256: member.source_fixture_sha256.clone(),
            abstract_certificate_sha256: member.abstract_certificate_sha256.clone(),
            source_map_sha256: member.source_map_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let ordered_local_ordinals = members.iter().map(|member| member.local_ordinal).collect();
    let ordered_global_ordinals = members.iter().map(|member| member.global_ordinal).collect();
    let ordered_source_copies = members.iter().map(|member| member.source_copy).collect();
    let mut group = PreparedColumnGroup {
        schema_version: GPU_FX_GROUP_SCHEMA.to_string(),
        group_id: String::new(),
        tranche: tranche.as_str().to_string(),
        source_dynkin_label: first.source_label.clone(),
        ordered_local_ordinals,
        ordered_global_ordinals,
        ordered_source_copies,
        members,
        pbw_plan_sha256: first.pbw_plan_sha256.clone(),
        pbw_word_count: first.pbw_word_count,
        reciprocal_map_sha256: first.reciprocal_map_sha256.clone(),
        runtime,
        active_columns: preflights.len(),
        singleton_fallback: preflights.len() == 1,
    };
    group.group_id = group_identity_sha256(&group);
    Ok(group)
}

fn group_identity_sha256(group: &PreparedColumnGroup) -> String {
    let mut hash = Sha256::new();
    hash.update(GPU_FX_GROUP_SCHEMA.as_bytes());
    hash.update([0]);
    hash.update(group.tranche.as_bytes());
    hash.update([0]);
    hash.update(group.source_dynkin_label.as_bytes());
    hash.update([0]);
    hash.update((group.active_columns as u64).to_le_bytes());
    for member in &group.members {
        hash.update((member.local_ordinal as u64).to_le_bytes());
        hash.update((member.global_ordinal as u64).to_le_bytes());
        hash.update((member.source_copy as u64).to_le_bytes());
        for value in [
            member.source_fixture.as_str(),
            member.source_fixture_sha256.as_str(),
            member.abstract_certificate_sha256.as_str(),
            member.source_map_sha256.as_str(),
        ] {
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
    }
    hash.update(group.pbw_plan_sha256.as_bytes());
    hash.update((group.pbw_word_count as u64).to_le_bytes());
    hash.update(group.reciprocal_map_sha256.as_bytes());
    hash.update(group.runtime.prime.to_le_bytes());
    hash.update(group.runtime.static_semantic_sha256.as_bytes());
    hash.update(group.runtime.flat_plan_sha256.as_bytes());
    format!("{:x}", hash.finalize())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} SHA-256 is malformed"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReducedLaneTerm {
    pub key: u64,
    pub coefficient: i128,
}

/// Pack, sort, and exactly reduce one already bounded raw lane batch. The
/// returned keys are strictly increasing and coefficients are nonzero.
pub(crate) fn reduce_raw_lane_batch(
    raw_terms: &[RecoupledSourceTerm],
    host_hard_cap_bytes: u64,
) -> Result<Vec<ReducedLaneTerm>, String> {
    let planned_bytes = payload_bytes::<ReducedLaneTerm>(raw_terms.len())?;
    if planned_bytes > host_hard_cap_bytes {
        return Err("raw lane reduction exceeds the declared host cap".to_string());
    }
    let mut packed = Vec::with_capacity(raw_terms.len());
    for term in raw_terms {
        if term.coefficient == i128::MIN {
            return Err("i128::MIN is not supported by grouped exact reduction".to_string());
        }
        if term.coefficient == 0 {
            continue;
        }
        packed.push(ReducedLaneTerm {
            key: pack_recoupling_key(term)?,
            coefficient: term.coefficient,
        });
    }
    let actual_bytes = payload_bytes::<ReducedLaneTerm>(packed.capacity())?;
    if actual_bytes > host_hard_cap_bytes {
        return Err("raw lane allocation exceeded the declared host cap".to_string());
    }
    packed.sort_unstable_by_key(|term| term.key);
    let mut write = 0_usize;
    let mut read = 0_usize;
    while read < packed.len() {
        let key = packed[read].key;
        let mut coefficient = 0_i128;
        while read < packed.len() && packed[read].key == key {
            coefficient = coefficient
                .checked_add(packed[read].coefficient)
                .ok_or_else(|| "grouped exact lane coefficient overflow".to_string())?;
            read += 1;
        }
        if coefficient == i128::MIN {
            return Err("grouped exact reduction produced unsupported i128::MIN".to_string());
        }
        if coefficient != 0 {
            packed[write] = ReducedLaneTerm { key, coefficient };
            write += 1;
        }
    }
    packed.truncate(write);
    Ok(packed)
}

fn pack_recoupling_key(term: &RecoupledSourceTerm) -> Result<u64, String> {
    if term.momentum_pair[0] > term.momentum_pair[1]
        || term.momentum_pair[1] >= 11
        || term.free_spinor >= 32
        || term.exterior_mask.count_ones() != 12
    {
        return Err("invalid grouped recoupled source term".to_string());
    }
    let metadata = u32::from(term.momentum_pair[0])
        | (u32::from(term.momentum_pair[1]) << 4)
        | (u32::from(term.free_spinor) << 8);
    Ok((u64::from(metadata) << 32) | u64::from(term.exterior_mask))
}

fn validate_canonical_key(key: u64) -> Result<(), String> {
    let metadata = (key >> 32) as u32;
    let left = (metadata & 15) as u8;
    let right = ((metadata >> 4) & 15) as u8;
    let free = ((metadata >> 8) & 31) as u8;
    let mask = key as u32;
    if metadata >> 13 != 0 || left > right || right >= 11 || free >= 32 || mask.count_ones() != 12 {
        return Err("malformed canonical grouped recoupling key".to_string());
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnionBatchTelemetry {
    pub schema_version: String,
    pub group_id: String,
    pub batch_ordinal: u64,
    pub active_columns: usize,
    pub union_key_count: usize,
    /// Index `n` counts keys present in exactly `n` lanes. Index zero is zero.
    pub keys_by_present_lane_count: Vec<u64>,
    pub reduced_terms_per_column: Vec<usize>,
    pub key_capacity: usize,
    pub value_capacity: usize,
    /// All capacities owned by the union inputs, workspace, and returned batch.
    pub host_capacity_bytes: u64,
    pub union_milliseconds: u128,
    pub union_keys_per_second: u64,
    pub deterministic_batch_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupCudaBatchTelemetry {
    pub upload_milliseconds: f64,
    pub contract_milliseconds: f64,
    pub finalize_milliseconds: f64,
    pub download_milliseconds: f64,
    pub total_milliseconds: f64,
    pub nonzero_terms_per_column: Vec<u64>,
    pub expanded_contributions_per_column: Vec<u64>,
    pub device_resident_bytes: u64,
    pub device_high_water_bytes: u64,
    pub device_hard_cap_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupBatchObservation {
    pub event: String,
    pub group_id: String,
    pub active_columns: usize,
    pub ordered_local_ordinals: Vec<usize>,
    pub ordered_global_ordinals: Vec<usize>,
    pub ordered_source_copies: Vec<usize>,
    pub word_ordinal: usize,
    pub pbw_root: Option<u8>,
    pub raw_terms_per_column: Vec<u64>,
    pub union: UnionBatchTelemetry,
    /// Populated after the production CUDA wrapper returns.
    pub cuda: Option<GroupCudaBatchTelemetry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactUnionBatch {
    pub keys: Vec<u64>,
    /// Canonical key-major layout: `[key * active_columns + lane]`.
    pub key_major_values: Vec<i128>,
    pub telemetry: UnionBatchTelemetry,
}

pub(crate) struct ExactUnionBatcher<'a> {
    group_id: &'a str,
    lanes: Vec<&'a [ReducedLaneTerm]>,
    cursors: Vec<usize>,
    max_keys_per_batch: usize,
    host_hard_cap_bytes: u64,
    persistent_capacity_bytes: u64,
    batch_ordinal: u64,
}

impl<'a> ExactUnionBatcher<'a> {
    pub(crate) fn new(
        group_id: &'a str,
        lanes: &'a [Vec<ReducedLaneTerm>],
        max_keys_per_batch: usize,
        host_hard_cap_bytes: u64,
    ) -> Result<Self, String> {
        Self::new_from_batch_ordinal(group_id, lanes, max_keys_per_batch, host_hard_cap_bytes, 0)
    }

    pub(crate) fn new_from_batch_ordinal(
        group_id: &'a str,
        lanes: &'a [Vec<ReducedLaneTerm>],
        max_keys_per_batch: usize,
        host_hard_cap_bytes: u64,
        first_batch_ordinal: u64,
    ) -> Result<Self, String> {
        if lanes.is_empty() || lanes.len() > MAX_TESTED_GROUP_WIDTH {
            return Err("union width must be in 1..=32".to_string());
        }
        if max_keys_per_batch == 0 {
            return Err("union key batch cap must be nonzero".to_string());
        }
        for lane in lanes {
            let mut previous = None;
            for term in lane {
                validate_canonical_key(term.key)?;
                if term.coefficient == 0
                    || term.coefficient == i128::MIN
                    || previous.is_some_and(|key| key >= term.key)
                {
                    return Err("union lane is not canonical exact reduced input".to_string());
                }
                previous = Some(term.key);
            }
        }
        let lane_views = lanes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let cursors = vec![0; lanes.len()];
        let mut persistent_parts = lanes
            .iter()
            .map(|lane| payload_bytes::<ReducedLaneTerm>(lane.capacity()))
            .collect::<Result<Vec<_>, _>>()?;
        persistent_parts.push(payload_bytes::<&[ReducedLaneTerm]>(lane_views.capacity())?);
        persistent_parts.push(payload_bytes::<usize>(cursors.capacity())?);
        let persistent_capacity_bytes = sum_capacity_bytes(&persistent_parts)?;
        let planned = sum_capacity_bytes(&[
            persistent_capacity_bytes,
            union_payload_bytes(max_keys_per_batch, lanes.len())?,
            union_workspace_bytes(lanes.len())?,
        ])?;
        if planned > host_hard_cap_bytes {
            return Err("union batch capacity exceeds the declared host cap".to_string());
        }
        Ok(Self {
            group_id,
            lanes: lane_views,
            cursors,
            max_keys_per_batch,
            host_hard_cap_bytes,
            persistent_capacity_bytes,
            batch_ordinal: first_batch_ordinal,
        })
    }

    pub(crate) fn next_batch(&mut self) -> Result<Option<ExactUnionBatch>, String> {
        if self
            .lanes
            .iter()
            .zip(&self.cursors)
            .all(|(lane, cursor)| *cursor == lane.len())
        {
            return Ok(None);
        }
        let next_batch_ordinal = self
            .batch_ordinal
            .checked_add(1)
            .ok_or_else(|| "union batch ordinal overflow".to_string())?;
        let started = Instant::now();
        let width = self.lanes.len();
        let mut cursors = self.cursors.clone();
        let mut keys = Vec::with_capacity(self.max_keys_per_batch);
        let value_capacity = self
            .max_keys_per_batch
            .checked_mul(width)
            .ok_or_else(|| "union value capacity overflow".to_string())?;
        let mut values = Vec::with_capacity(value_capacity);
        let mut presence = vec![0_u64; width + 1];
        let mut reduced_terms_per_column = vec![0_usize; width];
        while keys.len() < self.max_keys_per_batch {
            let next_key = self
                .lanes
                .iter()
                .zip(&cursors)
                .filter_map(|(lane, cursor)| lane.get(*cursor).map(|term| term.key))
                .min();
            let Some(key) = next_key else { break };
            let value_start = values.len();
            values.resize(value_start + width, 0);
            let mut present_lanes = 0_usize;
            for (lane_ordinal, (lane, cursor)) in self.lanes.iter().zip(&mut cursors).enumerate() {
                if lane.get(*cursor).is_some_and(|term| term.key == key) {
                    let term = lane[*cursor];
                    values[value_start + lane_ordinal] = term.coefficient;
                    *cursor += 1;
                    present_lanes += 1;
                    reduced_terms_per_column[lane_ordinal] += 1;
                }
            }
            if present_lanes != 0 {
                keys.push(key);
                presence[present_lanes] = presence[present_lanes]
                    .checked_add(1)
                    .ok_or_else(|| "union presence histogram overflow".to_string())?;
            } else {
                values.truncate(value_start);
            }
        }
        let host_capacity_bytes = sum_capacity_bytes(&[
            self.persistent_capacity_bytes,
            payload_bytes::<u64>(keys.capacity())?,
            payload_bytes::<i128>(values.capacity())?,
            payload_bytes::<usize>(cursors.capacity())?,
            payload_bytes::<u64>(presence.capacity())?,
            payload_bytes::<usize>(reduced_terms_per_column.capacity())?,
        ])?;
        if host_capacity_bytes > self.host_hard_cap_bytes {
            return Err("union allocation exceeded the declared host cap".to_string());
        }
        if values.len() != keys.len() * width {
            return Err("union key-major shape invariant failed".to_string());
        }
        let elapsed = started.elapsed();
        let mut hash = Sha256::new();
        hash.update(GPU_FX_GROUP_SCHEMA.as_bytes());
        hash.update(b"\0union-batch-v1\0");
        hash.update(self.group_id.as_bytes());
        hash.update(self.batch_ordinal.to_le_bytes());
        hash.update((width as u64).to_le_bytes());
        for key in &keys {
            hash.update(key.to_le_bytes());
        }
        for value in &values {
            hash.update(value.to_le_bytes());
        }
        let nanos = elapsed.as_nanos();
        let union_keys_per_second = if nanos == 0 {
            0
        } else {
            u64::try_from((keys.len() as u128 * 1_000_000_000) / nanos).unwrap_or(u64::MAX)
        };
        let telemetry = UnionBatchTelemetry {
            schema_version: GPU_FX_GROUP_SCHEMA.to_string(),
            group_id: self.group_id.to_string(),
            batch_ordinal: self.batch_ordinal,
            active_columns: width,
            union_key_count: keys.len(),
            keys_by_present_lane_count: presence,
            reduced_terms_per_column,
            key_capacity: keys.capacity(),
            value_capacity: values.capacity(),
            host_capacity_bytes,
            union_milliseconds: elapsed.as_millis(),
            union_keys_per_second,
            deterministic_batch_sha256: format!("{:x}", hash.finalize()),
        };
        self.cursors = cursors;
        self.batch_ordinal = next_batch_ordinal;
        Ok(Some(ExactUnionBatch {
            keys,
            key_major_values: values,
            telemetry,
        }))
    }
}

fn payload_bytes<T>(capacity: usize) -> Result<u64, String> {
    let bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| "payload byte count overflow".to_string())?;
    u64::try_from(bytes).map_err(|_| "payload byte count exceeds u64".to_string())
}

fn sum_capacity_bytes(parts: &[u64]) -> Result<u64, String> {
    parts.iter().try_fold(0_u64, |total, part| {
        total
            .checked_add(*part)
            .ok_or_else(|| "host capacity byte count overflow".to_string())
    })
}

fn union_payload_bytes(key_capacity: usize, width: usize) -> Result<u64, String> {
    payload_bytes::<u64>(key_capacity)?
        .checked_add(payload_bytes::<i128>(
            key_capacity
                .checked_mul(width)
                .ok_or_else(|| "union value capacity overflow".to_string())?,
        )?)
        .ok_or_else(|| "union payload capacity overflow".to_string())
}

fn union_workspace_bytes(width: usize) -> Result<u64, String> {
    payload_bytes::<usize>(width)?
        .checked_add(payload_bytes::<u64>(width + 1)?)
        .and_then(|total| total.checked_add(payload_bytes::<usize>(width).ok()?))
        .ok_or_else(|| "union workspace capacity overflow".to_string())
}

/// Lane-separated modular row accumulator. Each CUDA batch result must use
/// `columns[lane][functional_row]`; no row or lane transposition is accepted.
pub(crate) struct GroupRowAccumulator {
    group_id: String,
    global_ordinals: Vec<usize>,
    prime: u32,
    static_semantic_sha256: String,
    columns: Vec<Vec<GaussianResidue>>,
    batches_folded: u64,
}

impl GroupRowAccumulator {
    pub(crate) fn new(
        group_id: &str,
        global_ordinals: &[usize],
        prime: u32,
        static_semantic_sha256: &str,
        functional_rows: usize,
    ) -> Result<Self, String> {
        if global_ordinals.is_empty()
            || global_ordinals.len() > MAX_TESTED_GROUP_WIDTH
            || functional_rows == 0
            || !GPU_FX_PRIMES.contains(&prime)
            || global_ordinals.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("invalid grouped row accumulator identity".to_string());
        }
        validate_sha256("static semantic", static_semantic_sha256)?;
        Ok(Self {
            group_id: group_id.to_string(),
            global_ordinals: global_ordinals.to_vec(),
            prime,
            static_semantic_sha256: static_semantic_sha256.to_string(),
            columns: vec![vec![GaussianResidue::zero(); functional_rows]; global_ordinals.len()],
            batches_folded: 0,
        })
    }

    pub(crate) fn fold_batch(
        &mut self,
        batch_columns: &[Vec<GaussianResidue>],
    ) -> Result<(), String> {
        if batch_columns.len() != self.columns.len()
            || batch_columns
                .iter()
                .zip(&self.columns)
                .any(|(batch, total)| batch.len() != total.len())
        {
            return Err("multi-column CUDA result shape or lane order mismatch".to_string());
        }
        if batch_columns.iter().flatten().any(|contribution| {
            contribution.real >= self.prime || contribution.imaginary >= self.prime
        }) {
            return Err("multi-column CUDA result contains a noncanonical residue".to_string());
        }
        let next_batches_folded = self
            .batches_folded
            .checked_add(1)
            .ok_or_else(|| "group row batch count overflow".to_string())?;
        for (total, batch) in self.columns.iter_mut().zip(batch_columns) {
            for (destination, contribution) in total.iter_mut().zip(batch) {
                *destination = destination.add(*contribution, self.prime);
            }
        }
        self.batches_folded = next_batches_folded;
        Ok(())
    }

    pub(crate) fn restore(
        &mut self,
        columns: Vec<Vec<GaussianResidue>>,
        batches_folded: u64,
    ) -> Result<(), String> {
        if columns.len() != self.columns.len()
            || columns
                .iter()
                .zip(&self.columns)
                .any(|(restored, allocated)| restored.len() != allocated.len())
            || columns
                .iter()
                .flatten()
                .any(|value| value.real >= self.prime || value.imaginary >= self.prime)
        {
            return Err("group checkpoint row shape or residue is invalid".to_string());
        }
        self.columns = columns;
        self.batches_folded = batches_folded;
        Ok(())
    }

    pub(crate) const fn batches_folded(&self) -> u64 {
        self.batches_folded
    }

    pub(crate) fn columns(&self) -> &[Vec<GaussianResidue>] {
        &self.columns
    }

    pub(crate) fn column_digests(&self) -> Vec<String> {
        self.columns
            .iter()
            .zip(&self.global_ordinals)
            .map(|(rows, global_ordinal)| {
                let mut hash = Sha256::new();
                hash.update(
                    crate::eleven_dimensional_second_momentum_gpu::GPU_FX_SCHEMA.as_bytes(),
                );
                hash.update(self.prime.to_le_bytes());
                hash.update((*global_ordinal as u64).to_le_bytes());
                hash.update(self.static_semantic_sha256.as_bytes());
                for row in rows {
                    hash.update(row.real.to_le_bytes());
                    hash.update(row.imaginary.to_le_bytes());
                }
                format!("{:x}", hash.finalize())
            })
            .collect()
    }
}

pub(crate) fn write_observation_jsonl<W: io::Write, T: Serialize>(
    writer: &mut W,
    observation: &T,
) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, observation).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LaneWordCompletion {
    pub lane_index: usize,
    pub local_ordinal: usize,
    pub global_ordinal: usize,
    pub source_copy: usize,
    pub word_ordinal: usize,
    pub raw_terms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GroupWordOrchestrationConfig {
    pub start_word_ordinal: usize,
    pub end_word_ordinal_exclusive: usize,
    pub first_global_batch_ordinal: u64,
    pub raw_batch_term_cap_per_lane: usize,
    pub max_union_keys_per_batch: usize,
    pub aggregate_host_payload_cap_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupWordOrchestrationReport {
    pub schema_version: String,
    pub group_id: String,
    pub start_word_ordinal: usize,
    pub next_word_ordinal: usize,
    pub first_global_batch_ordinal: u64,
    pub next_global_batch_ordinal: u64,
    pub completed_words: usize,
    pub union_batches: u64,
    pub raw_terms_per_column: Vec<u64>,
    pub aggregate_host_payload_cap_bytes: u64,
}

enum LaneSourceMessage {
    RawBatch(Vec<RecoupledSourceTerm>),
    Complete(Result<LaneWordCompletion, String>),
}

/// Run every lane for one PBW word concurrently through bounded rendezvous
/// channels. The main thread observes each lane's raw terms in original order,
/// reduces one bounded batch per live lane, constructs a canonical union, and
/// hands each union batch to `consume_union`. A word completes only after all
/// lanes report their exact identity and every queued batch is consumed;
/// `complete_word` is then invoked exactly once in ordinal order, including
/// for words whose every lane is empty.
pub(crate) fn orchestrate_group_words<R, O, C, W>(
    plan: &PreparedColumnGroup,
    config: GroupWordOrchestrationConfig,
    run_lane_word: &R,
    observe_raw_term: O,
    consume_union: C,
    complete_word: W,
) -> Result<GroupWordOrchestrationReport, String>
where
    R: Sync
        + Fn(
            usize,
            usize,
            usize,
            usize,
            usize,
            &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
        ) -> Result<LaneWordCompletion, String>,
    O: FnMut(usize, usize, &RecoupledSourceTerm) -> Result<(), String>,
    C: FnMut(usize, Vec<u64>, ExactUnionBatch) -> Result<(), String>,
    W: FnMut(usize, &[LaneWordCompletion]) -> Result<(), String>,
{
    orchestrate_group_words_with_batch_group_id(
        plan,
        &plan.group_id,
        config,
        run_lane_word,
        observe_raw_term,
        consume_union,
        complete_word,
    )
}

pub(crate) fn orchestrate_group_words_with_batch_group_id<R, O, C, W>(
    plan: &PreparedColumnGroup,
    batch_group_id: &str,
    config: GroupWordOrchestrationConfig,
    run_lane_word: &R,
    mut observe_raw_term: O,
    mut consume_union: C,
    mut complete_word: W,
) -> Result<GroupWordOrchestrationReport, String>
where
    R: Sync
        + Fn(
            usize,
            usize,
            usize,
            usize,
            usize,
            &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
        ) -> Result<LaneWordCompletion, String>,
    O: FnMut(usize, usize, &RecoupledSourceTerm) -> Result<(), String>,
    C: FnMut(usize, Vec<u64>, ExactUnionBatch) -> Result<(), String>,
    W: FnMut(usize, &[LaneWordCompletion]) -> Result<(), String>,
{
    orchestrate_group_words_with_batch_observer(
        plan,
        batch_group_id,
        config,
        run_lane_word,
        |lane, word, raw| {
            for term in raw {
                observe_raw_term(lane, word, term)?;
            }
            Ok(())
        },
        consume_union,
        complete_word,
    )
}

pub(crate) fn orchestrate_group_words_with_batch_observer<R, O, C, W>(
    plan: &PreparedColumnGroup,
    batch_group_id: &str,
    config: GroupWordOrchestrationConfig,
    run_lane_word: &R,
    mut observe_raw_batch: O,
    mut consume_union: C,
    mut complete_word: W,
) -> Result<GroupWordOrchestrationReport, String>
where
    R: Sync
        + Fn(
            usize,
            usize,
            usize,
            usize,
            usize,
            &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
        ) -> Result<LaneWordCompletion, String>,
    O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
    C: FnMut(usize, Vec<u64>, ExactUnionBatch) -> Result<(), String>,
    W: FnMut(usize, &[LaneWordCompletion]) -> Result<(), String>,
{
    if plan.active_columns == 0
        || plan.active_columns != plan.members.len()
        || plan.active_columns != plan.ordered_local_ordinals.len()
        || plan.active_columns != plan.ordered_global_ordinals.len()
        || plan.active_columns != plan.ordered_source_copies.len()
        || plan.members.iter().enumerate().any(|(lane, member)| {
            member.local_ordinal != plan.ordered_local_ordinals[lane]
                || member.global_ordinal != plan.ordered_global_ordinals[lane]
                || member.source_copy != plan.ordered_source_copies[lane]
        })
        || batch_group_id.is_empty()
        || config.start_word_ordinal > config.end_word_ordinal_exclusive
        || config.end_word_ordinal_exclusive > plan.pbw_word_count
        || config.raw_batch_term_cap_per_lane == 0
        || config.max_union_keys_per_batch == 0
    {
        return Err("invalid group word orchestration identity or range".to_string());
    }
    let raw_batch_bytes = payload_bytes::<RecoupledSourceTerm>(config.raw_batch_term_cap_per_lane)?;
    let reduced_lane_bytes = payload_bytes::<ReducedLaneTerm>(config.raw_batch_term_cap_per_lane)?;
    let concurrent_raw_payload_bytes = raw_batch_bytes
        .checked_mul(plan.active_columns as u64)
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or_else(|| "concurrent raw payload capacity overflow".to_string())?;
    if concurrent_raw_payload_bytes >= config.aggregate_host_payload_cap_bytes {
        return Err("bounded lane channels exhaust the aggregate host payload cap".to_string());
    }
    let union_host_cap = config.aggregate_host_payload_cap_bytes - concurrent_raw_payload_bytes;
    let mut global_batch_ordinal = config.first_global_batch_ordinal;
    let mut union_batches = 0_u64;
    let mut total_raw_terms = vec![0_u64; plan.active_columns];

    for word_ordinal in config.start_word_ordinal..config.end_word_ordinal_exclusive {
        let mut word_raw_terms = vec![0_u64; plan.active_columns];
        let mut word_completions = vec![None; plan.active_columns];
        std::thread::scope(|scope| -> Result<(), String> {
            let mut receivers = Vec::with_capacity(plan.active_columns);
            for lane in 0..plan.active_columns {
                let (sender, receiver) = std::sync::mpsc::sync_channel::<LaneSourceMessage>(1);
                receivers.push(receiver);
                let expected_local_ordinal = plan.ordered_local_ordinals[lane];
                let expected_global_ordinal = plan.ordered_global_ordinals[lane];
                let expected_source_copy = plan.ordered_source_copies[lane];
                scope.spawn(move || {
                    let mut raw = Vec::new();
                    if raw
                        .try_reserve_exact(config.raw_batch_term_cap_per_lane)
                        .is_err()
                    {
                        let _ = sender.send(LaneSourceMessage::Complete(Err(
                            "reserve bounded lane source batch failed".to_string(),
                        )));
                        return;
                    }
                    let result = run_lane_word(
                        lane,
                        expected_local_ordinal,
                        expected_global_ordinal,
                        expected_source_copy,
                        word_ordinal,
                        &mut |term| {
                            raw.push(term);
                            if raw.len() == config.raw_batch_term_cap_per_lane {
                                let mut replacement = Vec::new();
                                replacement
                                    .try_reserve_exact(config.raw_batch_term_cap_per_lane)
                                    .map_err(|error| {
                                        format!("reserve next bounded lane source batch: {error}")
                                    })?;
                                let full = std::mem::replace(&mut raw, replacement);
                                sender
                                    .send(LaneSourceMessage::RawBatch(full))
                                    .map_err(|_| {
                                        "group source consumer disconnected".to_string()
                                    })?;
                            }
                            Ok(())
                        },
                    );
                    if !raw.is_empty() && sender.send(LaneSourceMessage::RawBatch(raw)).is_err() {
                        return;
                    }
                    let _ = sender.send(LaneSourceMessage::Complete(result));
                });
            }

            let mut complete = vec![false; plan.active_columns];
            while complete.iter().any(|done| !done) {
                let mut raw_by_lane = (0..plan.active_columns)
                    .map(|_| Vec::<RecoupledSourceTerm>::new())
                    .collect::<Vec<_>>();
                let mut received_batch = false;
                for lane in 0..plan.active_columns {
                    if complete[lane] {
                        continue;
                    }
                    match receivers[lane]
                        .recv()
                        .map_err(|_| "group lane source ended without completion".to_string())?
                    {
                        LaneSourceMessage::RawBatch(raw) => {
                            received_batch = true;
                            raw_by_lane[lane] = raw;
                        }
                        LaneSourceMessage::Complete(result) => {
                            let completion = result?;
                            let expected = &plan.members[lane];
                            if completion.lane_index != lane
                                || completion.local_ordinal != expected.local_ordinal
                                || completion.global_ordinal != expected.global_ordinal
                                || completion.source_copy != expected.source_copy
                                || completion.word_ordinal != word_ordinal
                                || completion.raw_terms != word_raw_terms[lane]
                            {
                                return Err(
                                    "group lane completion identity or count mismatch".to_string()
                                );
                            }
                            word_completions[lane] = Some(completion);
                            complete[lane] = true;
                        }
                    }
                }
                if !received_batch {
                    continue;
                }

                let mut reduced_by_lane = Vec::with_capacity(plan.active_columns);
                let mut raw_counts = Vec::with_capacity(plan.active_columns);
                for (lane, raw) in raw_by_lane.iter().enumerate() {
                    raw_counts.push(raw.len() as u64);
                    observe_raw_batch(lane, word_ordinal, raw)?;
                    let count = u64::try_from(raw.len())
                        .map_err(|_| "raw lane batch count exceeds u64".to_string())?;
                    word_raw_terms[lane] = word_raw_terms[lane]
                        .checked_add(count)
                        .ok_or_else(|| "word raw lane count overflow".to_string())?;
                    total_raw_terms[lane] = total_raw_terms[lane]
                        .checked_add(count)
                        .ok_or_else(|| "group raw lane count overflow".to_string())?;
                    reduced_by_lane.push(reduce_raw_lane_batch(raw, reduced_lane_bytes)?);
                }
                let mut union = ExactUnionBatcher::new_from_batch_ordinal(
                    batch_group_id,
                    &reduced_by_lane,
                    config.max_union_keys_per_batch,
                    union_host_cap,
                    global_batch_ordinal,
                )?;
                while let Some(batch) = union.next_batch()? {
                    global_batch_ordinal = batch
                        .telemetry
                        .batch_ordinal
                        .checked_add(1)
                        .ok_or_else(|| "global group batch ordinal overflow".to_string())?;
                    union_batches = union_batches
                        .checked_add(1)
                        .ok_or_else(|| "group union batch count overflow".to_string())?;
                    consume_union(word_ordinal, raw_counts.clone(), batch)?;
                }
            }
            Ok(())
        })?;
        let word_completions = word_completions
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "group word ended without every lane completion".to_string())?;
        complete_word(word_ordinal, &word_completions)?;
    }

    Ok(GroupWordOrchestrationReport {
        schema_version: GPU_FX_GROUP_SCHEMA.to_string(),
        group_id: batch_group_id.to_string(),
        start_word_ordinal: config.start_word_ordinal,
        next_word_ordinal: config.end_word_ordinal_exclusive,
        first_global_batch_ordinal: config.first_global_batch_ordinal,
        next_global_batch_ordinal: global_batch_ordinal,
        completed_words: config.end_word_ordinal_exclusive - config.start_word_ordinal,
        union_batches,
        raw_terms_per_column: total_raw_terms,
        aggregate_host_payload_cap_bytes: config.aggregate_host_payload_cap_bytes,
    })
}

/// CUDA-backed contraction stage for already reduced union batches. Source
/// collection and word hashing remain tranche-owned and must complete before a
/// batch is handed here.
#[cfg(feature = "cuda")]
pub(crate) struct CudaGroupBatchExecutor {
    plan: PreparedColumnGroup,
    batch_group_id: String,
    cuda: crate::eleven_dimensional_second_momentum_gpu::CudaModularFx,
    rows: GroupRowAccumulator,
    batch_columns: Vec<Vec<GaussianResidue>>,
}

#[cfg(feature = "cuda")]
impl CudaGroupBatchExecutor {
    pub(crate) fn new(
        plan: PreparedColumnGroup,
        static_data: &crate::eleven_dimensional_second_momentum_gpu::ModularFxStaticData,
        device: i32,
        max_union_keys: usize,
        device_hard_cap_bytes: u64,
    ) -> Result<Self, String> {
        let batch_group_id = plan.group_id.clone();
        Self::new_for_batch_group(
            plan,
            batch_group_id,
            static_data,
            device,
            max_union_keys,
            device_hard_cap_bytes,
        )
    }

    pub(crate) fn new_for_batch_group(
        plan: PreparedColumnGroup,
        batch_group_id: String,
        static_data: &crate::eleven_dimensional_second_momentum_gpu::ModularFxStaticData,
        device: i32,
        max_union_keys: usize,
        device_hard_cap_bytes: u64,
    ) -> Result<Self, String> {
        if max_union_keys == 0
            || plan.runtime.prime != static_data.prime()
            || plan.runtime.static_semantic_sha256 != static_data.semantic_sha256()
        {
            return Err("group runtime and modular static identity disagree".to_string());
        }
        let mut cuda =
            crate::eleven_dimensional_second_momentum_gpu::CudaModularFx::new(static_data, device)?;
        if cuda.flat_plan_sha256() != plan.runtime.flat_plan_sha256 {
            return Err("group flat-plan identity changed during CUDA setup".to_string());
        }
        cuda.set_recoupling_hard_cap(device_hard_cap_bytes)?;
        cuda.reserve_multicol(max_union_keys, plan.active_columns)?;
        let rows = GroupRowAccumulator::new(
            &plan.group_id,
            &plan.ordered_global_ordinals,
            plan.runtime.prime,
            &plan.runtime.static_semantic_sha256,
            crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT,
        )?;
        let batch_columns = (0..plan.active_columns)
            .map(|_| {
                Vec::with_capacity(
                    crate::eleven_dimensional_second_momentum_gpu::FUNCTIONAL_ROW_COUNT,
                )
            })
            .collect();
        Ok(Self {
            plan,
            batch_group_id,
            cuda,
            rows,
            batch_columns,
        })
    }

    pub(crate) fn accumulate_batch(
        &mut self,
        batch: &ExactUnionBatch,
        word_ordinal: usize,
        pbw_root: Option<u8>,
        raw_terms_per_column: Vec<u64>,
    ) -> Result<GroupBatchObservation, String> {
        if batch.telemetry.group_id != self.batch_group_id
            || batch.telemetry.active_columns != self.plan.active_columns
            || raw_terms_per_column.len() != self.plan.active_columns
            || pbw_root.is_some_and(|root| !(1..=5).contains(&root))
        {
            return Err("group batch observation identity mismatch".to_string());
        }
        let stats = self.cuda.accumulate_reduced_multicol_into(
            &batch.keys,
            &batch.key_major_values,
            self.plan.active_columns,
            &mut self.batch_columns,
        )?;
        self.rows.fold_batch(&self.batch_columns)?;
        let width = self.plan.active_columns;
        let cuda = GroupCudaBatchTelemetry {
            upload_milliseconds: f64::from(stats.upload_milliseconds),
            contract_milliseconds: f64::from(stats.contract_milliseconds),
            finalize_milliseconds: f64::from(stats.finalize_milliseconds),
            download_milliseconds: f64::from(stats.download_milliseconds),
            total_milliseconds: f64::from(stats.total_milliseconds),
            nonzero_terms_per_column: stats.nonzero_terms[..width].to_vec(),
            expanded_contributions_per_column: stats.expanded_contributions[..width].to_vec(),
            device_resident_bytes: stats.resident_bytes,
            device_high_water_bytes: stats.buffer_high_water_bytes,
            device_hard_cap_bytes: stats.device_hard_cap_bytes,
        };
        Ok(GroupBatchObservation {
            event: "second_momentum_gpu_group_batch".to_string(),
            group_id: self.plan.group_id.clone(),
            active_columns: width,
            ordered_local_ordinals: self.plan.ordered_local_ordinals.clone(),
            ordered_global_ordinals: self.plan.ordered_global_ordinals.clone(),
            ordered_source_copies: self.plan.ordered_source_copies.clone(),
            word_ordinal,
            pbw_root,
            raw_terms_per_column,
            union: batch.telemetry.clone(),
            cuda: Some(cuda),
        })
    }

    pub(crate) const fn prime(&self) -> u32 {
        self.plan.runtime.prime
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.cuda.resident_bytes()
    }

    pub(crate) fn tighten_device_hard_cap_to_resident(&mut self) -> Result<u64, String> {
        let resident_bytes = self.resident_bytes();
        self.cuda.set_recoupling_hard_cap(resident_bytes)?;
        Ok(resident_bytes)
    }

    pub(crate) fn run_word_synchronous<R, O, B, W>(
        &mut self,
        config: GroupWordOrchestrationConfig,
        run_lane_word: &R,
        observe_raw_term: O,
        mut observe_batch: B,
        complete_word: W,
    ) -> Result<GroupWordOrchestrationReport, String>
    where
        R: Sync
            + Fn(
                usize,
                usize,
                usize,
                usize,
                usize,
                &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
            ) -> Result<LaneWordCompletion, String>,
        O: FnMut(usize, usize, &RecoupledSourceTerm) -> Result<(), String>,
        B: FnMut(&GroupBatchObservation) -> Result<(), String>,
        W: FnMut(usize, &[LaneWordCompletion]) -> Result<(), String>,
    {
        let plan = self.plan.clone();
        orchestrate_group_words(
            &plan,
            config,
            run_lane_word,
            observe_raw_term,
            |word_ordinal, raw_counts, batch| {
                let observation = self.accumulate_batch(&batch, word_ordinal, None, raw_counts)?;
                observe_batch(&observation)
            },
            complete_word,
        )
    }

    pub(crate) fn run_word_synchronous_batched<R, O, B, W>(
        &mut self,
        config: GroupWordOrchestrationConfig,
        run_lane_word: &R,
        observe_raw_batch: O,
        mut observe_batch: B,
        complete_word: W,
    ) -> Result<GroupWordOrchestrationReport, String>
    where
        R: Sync
            + Fn(
                usize,
                usize,
                usize,
                usize,
                usize,
                &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
            ) -> Result<LaneWordCompletion, String>,
        O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
        B: FnMut(&GroupBatchObservation) -> Result<(), String>,
        W: FnMut(usize, &[LaneWordCompletion]) -> Result<(), String>,
    {
        let plan = self.plan.clone();
        let batch_group_id = self.batch_group_id.clone();
        orchestrate_group_words_with_batch_observer(
            &plan,
            &batch_group_id,
            config,
            run_lane_word,
            observe_raw_batch,
            |word_ordinal, raw_counts, batch| {
                let observation = self.accumulate_batch(&batch, word_ordinal, None, raw_counts)?;
                observe_batch(&observation)
            },
            complete_word,
        )
    }

    /// Production extension point for linear functionals that consume the
    /// exact reduced union beside the established p2 contraction. Raw stream
    /// observation remains separate so transcript hashes keep original order.
    #[cfg(feature = "cuda")]
    pub(crate) fn run_word_synchronous_batched_with_union<R, O, U, B, W>(
        &mut self,
        config: GroupWordOrchestrationConfig,
        run_lane_word: &R,
        observe_raw_batch: O,
        mut observe_union: U,
        mut observe_batch: B,
        complete_word: W,
    ) -> Result<GroupWordOrchestrationReport, String>
    where
        R: Sync
            + Fn(
                usize,
                usize,
                usize,
                usize,
                usize,
                &mut dyn FnMut(RecoupledSourceTerm) -> Result<(), String>,
            ) -> Result<LaneWordCompletion, String>,
        O: FnMut(usize, usize, &[RecoupledSourceTerm]) -> Result<(), String>,
        U: FnMut(usize, &ExactUnionBatch) -> Result<(), String>,
        B: FnMut(&GroupBatchObservation) -> Result<(), String>,
        W: FnMut(usize, &[LaneWordCompletion]) -> Result<(), String>,
    {
        let plan = self.plan.clone();
        let batch_group_id = self.batch_group_id.clone();
        orchestrate_group_words_with_batch_observer(
            &plan,
            &batch_group_id,
            config,
            run_lane_word,
            observe_raw_batch,
            |word_ordinal, raw_counts, batch| {
                observe_union(word_ordinal, &batch)?;
                let observation = self.accumulate_batch(&batch, word_ordinal, None, raw_counts)?;
                observe_batch(&observation)
            },
            complete_word,
        )
    }

    pub(crate) fn final_columns(&self) -> &[Vec<GaussianResidue>] {
        self.rows.columns()
    }

    pub(crate) fn final_column_semantic_sha256(&self) -> Vec<String> {
        self.rows.column_digests()
    }

    pub(crate) fn restore_columns(
        &mut self,
        columns: Vec<Vec<GaussianResidue>>,
        batches_folded: u64,
    ) -> Result<(), String> {
        self.rows.restore(columns, batches_folded)
    }

    pub(crate) const fn batches_folded(&self) -> u64 {
        self.rows.batches_folded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(offset: u32) -> u32 {
        ((1_u32 << 12) - 1) << offset
    }

    fn term(key_seed: u8, coefficient: i128) -> RecoupledSourceTerm {
        RecoupledSourceTerm {
            momentum_pair: [key_seed % 5, key_seed % 5 + 1],
            free_spinor: key_seed % 32,
            exterior_mask: mask(u32::from(key_seed % 8)),
            coefficient,
        }
    }

    fn reduced(raw: &[RecoupledSourceTerm]) -> Vec<ReducedLaneTerm> {
        reduce_raw_lane_batch(raw, u64::MAX).unwrap()
    }

    fn runtime() -> GroupRuntimeIdentity {
        GroupRuntimeIdentity {
            prime: GPU_FX_PRIMES[0],
            static_semantic_sha256: "1".repeat(64),
            flat_plan_sha256: "2".repeat(64),
        }
    }

    fn fake_preflights() -> Vec<CommonPreflight> {
        (0..2)
            .map(|lane| CommonPreflight {
                tranche: "20001".to_string(),
                local_ordinal: lane,
                global_ordinal: 53 + lane,
                source_label: "10002".to_string(),
                source_copy: lane + 1,
                source_fixture: format!("copy-{lane}.i16le"),
                source_fixture_sha256: format!("{lane:064x}"),
                abstract_certificate_sha256: "3".repeat(64),
                source_map_sha256: format!("{:064x}", lane + 10),
                reciprocal_map_sha256: "4".repeat(64),
                pbw_plan_sha256: "5".repeat(64),
                pbw_word_count: 7,
            })
            .collect()
    }

    fn synthetic_plan(width: usize, word_count: usize) -> PreparedColumnGroup {
        let members = (0..width)
            .map(|lane| GroupColumnIdentity {
                local_ordinal: lane,
                global_ordinal: 53 + lane,
                source_copy: lane + 1,
                source_fixture: format!("copy-{lane}.i16le"),
                source_fixture_sha256: format!("{lane:064x}"),
                abstract_certificate_sha256: "3".repeat(64),
                source_map_sha256: format!("{:064x}", lane + 10),
            })
            .collect::<Vec<_>>();
        PreparedColumnGroup {
            schema_version: GPU_FX_GROUP_SCHEMA.to_string(),
            group_id: format!("synthetic-width-{width}"),
            tranche: "20001".to_string(),
            source_dynkin_label: "synthetic".to_string(),
            ordered_local_ordinals: members.iter().map(|member| member.local_ordinal).collect(),
            ordered_global_ordinals: members.iter().map(|member| member.global_ordinal).collect(),
            ordered_source_copies: members.iter().map(|member| member.source_copy).collect(),
            members,
            pbw_plan_sha256: "5".repeat(64),
            pbw_word_count: word_count,
            reciprocal_map_sha256: "4".repeat(64),
            runtime: runtime(),
            active_columns: width,
            singleton_fallback: width == 1,
        }
    }

    #[test]
    fn legal_group_discovery_matches_the_handoff_inventory() {
        assert_eq!(
            GpuFxTranche::Two0001.legal_local_groups(),
            vec![vec![0, 1], vec![2, 3], vec![4, 5, 6], vec![7, 8]]
        );
        assert_eq!(
            GpuFxTranche::Three0001.legal_local_groups(),
            vec![
                vec![0],
                vec![1, 2],
                vec![3],
                vec![4, 5, 6],
                vec![7, 8],
                vec![9],
                vec![10, 11],
                vec![12, 13, 14],
            ]
        );
    }

    #[test]
    fn requested_group_validation_rejects_reordering_subsets_and_bad_runtime() {
        let good = runtime();
        assert!(validate_requested_group(GpuFxTranche::Two0001, &[0, 1], &good).is_ok());
        assert!(validate_requested_group(GpuFxTranche::Two0001, &[1, 0], &good).is_err());
        assert!(validate_requested_group(GpuFxTranche::Two0001, &[0], &good).is_err());
        let mut bad = good;
        bad.prime = 17;
        assert!(validate_requested_group(GpuFxTranche::Two0001, &[0, 1], &bad).is_err());
    }

    #[test]
    fn group_identity_rejects_mixed_or_reordered_members_and_binds_runtime() {
        let members = fake_preflights();
        let first =
            build_group_identity(GpuFxTranche::Two0001, members.clone(), runtime()).unwrap();
        assert_eq!(first.ordered_local_ordinals, vec![0, 1]);
        assert_eq!(first.active_columns, 2);

        let mut reordered = members.clone();
        reordered.swap(0, 1);
        assert!(build_group_identity(GpuFxTranche::Two0001, reordered, runtime()).is_err());
        let mut wrong_plan = members.clone();
        wrong_plan[1].pbw_plan_sha256 = "6".repeat(64);
        assert!(build_group_identity(GpuFxTranche::Two0001, wrong_plan, runtime()).is_err());
        let mut changed_runtime = runtime();
        changed_runtime.flat_plan_sha256 = "7".repeat(64);
        let changed =
            build_group_identity(GpuFxTranche::Two0001, members, changed_runtime).unwrap();
        assert_ne!(first.group_id, changed.group_id);
    }

    #[test]
    fn multi_prime_identity_uses_canonical_prime_index_order_and_binds_runtime() {
        let members = fake_preflights();
        let plans = GPU_FX_PRIMES
            .iter()
            .enumerate()
            .map(|(index, prime)| {
                let runtime = GroupRuntimeIdentity {
                    prime: *prime,
                    static_semantic_sha256: format!("{index:064x}"),
                    flat_plan_sha256: format!("{:064x}", index + 10),
                };
                build_group_identity(GpuFxTranche::Two0001, members.clone(), runtime).unwrap()
            })
            .collect::<Vec<_>>();
        let all = multi_prime_group_identity_sha256(&plans).unwrap();
        assert_eq!(all.len(), 64);
        assert!(multi_prime_group_identity_sha256(&plans[1..]).is_ok());
        assert!(multi_prime_group_identity_sha256(&[plans[1].clone(), plans[0].clone()]).is_err());
        assert!(multi_prime_group_identity_sha256(&[plans[0].clone(), plans[0].clone()]).is_err());
        let mut changed = plans.clone();
        changed[2].runtime.flat_plan_sha256 = "f".repeat(64);
        assert_ne!(all, multi_prime_group_identity_sha256(&changed).unwrap());
    }

    #[test]
    fn word_synchronous_width_two_and_three_preserve_order_and_cross_word_cancellation() {
        for width in [2, 3] {
            let plan = synthetic_plan(width, 2);
            let config = GroupWordOrchestrationConfig {
                start_word_ordinal: 0,
                end_word_ordinal_exclusive: 2,
                first_global_batch_ordinal: 7,
                raw_batch_term_cap_per_lane: 1,
                max_union_keys_per_batch: 1,
                aggregate_host_payload_cap_bytes: 1 << 20,
            };
            let mut observed = vec![Vec::<i128>::new(); width];
            let mut lane_sums = (0..width)
                .map(|_| std::collections::BTreeMap::<u64, i128>::new())
                .collect::<Vec<_>>();
            let mut batch_ordinals = Vec::new();
            let mut completed_words = Vec::new();
            let report = orchestrate_group_words(
                &plan,
                config,
                &|lane, local_ordinal, global_ordinal, source_copy, word_ordinal, emit| {
                    let sign = if word_ordinal == 0 { 1 } else { -1 };
                    emit(term(2, sign * (lane as i128 + 1)))?;
                    emit(term(3, sign * (lane as i128 + 11)))?;
                    Ok(LaneWordCompletion {
                        lane_index: lane,
                        local_ordinal,
                        global_ordinal,
                        source_copy,
                        word_ordinal,
                        raw_terms: 2,
                    })
                },
                |lane, _, raw| {
                    observed[lane].push(raw.coefficient);
                    Ok(())
                },
                |_, _, batch| {
                    batch_ordinals.push(batch.telemetry.batch_ordinal);
                    for (key, values) in batch
                        .keys
                        .iter()
                        .zip(batch.key_major_values.chunks_exact(width))
                    {
                        for (lane, value) in values.iter().enumerate() {
                            *lane_sums[lane].entry(*key).or_default() += value;
                        }
                    }
                    Ok(())
                },
                |word_ordinal, completions| {
                    assert_eq!(completions.len(), width);
                    completed_words.push(word_ordinal);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(report.next_word_ordinal, 2);
            assert_eq!(report.raw_terms_per_column, vec![4; width]);
            assert_eq!(
                batch_ordinals,
                (7..report.next_global_batch_ordinal).collect::<Vec<_>>()
            );
            assert!(
                lane_sums
                    .iter()
                    .all(|by_key| by_key.values().all(|value| *value == 0))
            );
            assert_eq!(completed_words, vec![0, 1]);
            for (lane, coefficients) in observed.iter().enumerate() {
                assert_eq!(
                    coefficients,
                    &vec![
                        lane as i128 + 1,
                        lane as i128 + 11,
                        -(lane as i128 + 1),
                        -(lane as i128 + 11),
                    ]
                );
            }
        }
    }

    #[test]
    fn word_orchestration_resume_binds_batch_ordinal_and_rejects_swapped_lanes() {
        let plan = synthetic_plan(2, 3);
        let config = GroupWordOrchestrationConfig {
            start_word_ordinal: 1,
            end_word_ordinal_exclusive: 3,
            first_global_batch_ordinal: 19,
            raw_batch_term_cap_per_lane: 2,
            max_union_keys_per_batch: 4,
            aggregate_host_payload_cap_bytes: 1 << 20,
        };
        let mut ordinals = Vec::new();
        let mut completed_words = Vec::new();
        let report = orchestrate_group_words(
            &plan,
            config,
            &|lane, local_ordinal, global_ordinal, source_copy, word_ordinal, emit| {
                let raw_terms = if word_ordinal == 1 {
                    emit(term(word_ordinal as u8, lane as i128 + 1))?;
                    1
                } else {
                    0
                };
                Ok(LaneWordCompletion {
                    lane_index: lane,
                    local_ordinal,
                    global_ordinal,
                    source_copy,
                    word_ordinal,
                    raw_terms,
                })
            },
            |_, _, _| Ok(()),
            |_, _, batch| {
                ordinals.push(batch.telemetry.batch_ordinal);
                Ok(())
            },
            |word_ordinal, _| {
                completed_words.push(word_ordinal);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.start_word_ordinal, 1);
        assert_eq!(report.next_word_ordinal, 3);
        assert_eq!(completed_words, vec![1, 2]);
        assert_eq!(ordinals.first(), Some(&19));
        assert_eq!(report.next_global_batch_ordinal, 19 + ordinals.len() as u64);

        let swapped = orchestrate_group_words(
            &plan,
            config,
            &|lane, _, global_ordinal, source_copy, word_ordinal, _| {
                Ok(LaneWordCompletion {
                    lane_index: lane,
                    local_ordinal: 1 - lane,
                    global_ordinal,
                    source_copy,
                    word_ordinal,
                    raw_terms: 0,
                })
            },
            |_, _, _| Ok(()),
            |_, _, _| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert!(swapped.contains("identity"));
    }

    #[test]
    fn exact_lane_reduction_handles_duplicates_cancellation_and_overflow_guards() {
        let mut a = term(1, 9);
        let mut b = a;
        b.coefficient = -4;
        let mut c = a;
        c.coefficient = -5;
        assert!(reduced(&[a, b, c]).is_empty());
        a.coefficient = i128::MIN;
        assert!(reduce_raw_lane_batch(&[a], u64::MAX).is_err());
        assert!(reduce_raw_lane_batch(&[term(1, 1)], 0).is_err());
    }

    #[test]
    fn union_is_key_major_deterministic_and_tracks_lane_presence() {
        let lane0 = reduced(&[term(1, 3), term(2, -7)]);
        let lane1 = reduced(&[term(2, 11), term(3, 5)]);
        let lane2 = reduced(&[term(2, -13)]);
        let lanes = [lane0, lane1, lane2];
        let mut builder = ExactUnionBatcher::new("group", &lanes, 16, u64::MAX).unwrap();
        let batch = builder.next_batch().unwrap().unwrap();
        assert_eq!(batch.keys.len(), 3);
        assert_eq!(batch.key_major_values, vec![3, 0, 0, -7, 11, -13, 0, 5, 0]);
        assert_eq!(batch.telemetry.keys_by_present_lane_count, vec![0, 2, 0, 1]);
        assert_eq!(batch.telemetry.reduced_terms_per_column, vec![2, 2, 1]);
        assert!(builder.next_batch().unwrap().is_none());

        let mut again = ExactUnionBatcher::new("group", &lanes, 16, u64::MAX).unwrap();
        assert_eq!(
            batch.telemetry.deterministic_batch_sha256,
            again
                .next_batch()
                .unwrap()
                .unwrap()
                .telemetry
                .deterministic_batch_sha256
        );
        let mut resumed =
            ExactUnionBatcher::new_from_batch_ordinal("group", &lanes, 16, u64::MAX, 9).unwrap();
        let resumed = resumed.next_batch().unwrap().unwrap();
        assert_eq!(resumed.telemetry.batch_ordinal, 9);
        assert_ne!(
            batch.telemetry.deterministic_batch_sha256,
            resumed.telemetry.deterministic_batch_sha256
        );
    }

    #[test]
    fn cancellation_in_one_lane_keeps_other_lane_and_cross_batch_deltas() {
        let key_term = term(2, 9);
        let mut negative = key_term;
        negative.coefficient = -9;
        let lane0 = reduced(&[key_term, negative]);
        let lane1 = reduced(&[term(2, 4)]);
        let lanes = [lane0, lane1];
        let mut builder = ExactUnionBatcher::new("cancel", &lanes, 1, u64::MAX).unwrap();
        assert_eq!(
            builder.next_batch().unwrap().unwrap().key_major_values,
            vec![0, 4]
        );

        let positive = reduced(&[term(3, 12)]);
        let negative = reduced(&[term(3, -12)]);
        let positive_lanes = [positive];
        let negative_lanes = [negative];
        let mut first = ExactUnionBatcher::new("cross", &positive_lanes, 1, u64::MAX).unwrap();
        let mut second = ExactUnionBatcher::new("cross", &negative_lanes, 1, u64::MAX).unwrap();
        assert_eq!(
            first.next_batch().unwrap().unwrap().key_major_values[0]
                + second.next_batch().unwrap().unwrap().key_major_values[0],
            0
        );
    }

    #[test]
    fn union_batches_are_bounded_and_support_widths_through_32() {
        for width in [1, 2, 3, 4, 8, 15, 32] {
            let owned = (0..width)
                .map(|lane| reduced(&[term((lane % 8) as u8, lane as i128 + 1)]))
                .collect::<Vec<_>>();
            let mut builder = ExactUnionBatcher::new("width", &owned, 1, u64::MAX).unwrap();
            while let Some(batch) = builder.next_batch().unwrap() {
                assert_eq!(batch.keys.len(), 1);
                assert_eq!(batch.key_major_values.len(), width);
                assert!(batch.telemetry.host_capacity_bytes <= u64::MAX);
            }
        }
    }

    #[test]
    fn union_rejects_noncanonical_lanes_keys_and_allocation_caps() {
        let canonical = reduced(&[term(1, 1), term(2, 2)]);
        let mut reversed = canonical.clone();
        reversed.reverse();
        assert!(ExactUnionBatcher::new("bad", &[reversed], 2, u64::MAX).is_err());
        assert!(ExactUnionBatcher::new("cap", &[canonical], 2, 1).is_err());
        let malformed = vec![ReducedLaneTerm {
            key: 0,
            coefficient: 1,
        }];
        assert!(ExactUnionBatcher::new("bad", &[malformed], 1, u64::MAX).is_err());
    }

    #[test]
    fn independent_row_folding_preserves_lane_and_batch_separation() {
        let static_digest = "a".repeat(64);
        let mut rows =
            GroupRowAccumulator::new("g", &[53, 54], GPU_FX_PRIMES[0], &static_digest, 2).unwrap();
        rows.fold_batch(&[
            vec![
                GaussianResidue {
                    real: 1,
                    imaginary: 2,
                },
                GaussianResidue {
                    real: 3,
                    imaginary: 4,
                },
            ],
            vec![
                GaussianResidue {
                    real: 5,
                    imaginary: 6,
                },
                GaussianResidue {
                    real: 7,
                    imaginary: 8,
                },
            ],
        ])
        .unwrap();
        rows.fold_batch(&[
            vec![
                GaussianResidue {
                    real: 10,
                    imaginary: 20,
                },
                GaussianResidue {
                    real: 30,
                    imaginary: 40,
                },
            ],
            vec![
                GaussianResidue {
                    real: 50,
                    imaginary: 60,
                },
                GaussianResidue {
                    real: 70,
                    imaginary: 80,
                },
            ],
        ])
        .unwrap();
        assert_eq!(
            rows.columns()[0][0],
            GaussianResidue {
                real: 11,
                imaginary: 22
            }
        );
        assert_eq!(
            rows.columns()[1][0],
            GaussianResidue {
                real: 55,
                imaginary: 66
            }
        );
        assert_ne!(rows.column_digests()[0], rows.column_digests()[1]);
        assert!(
            rows.fold_batch(&[vec![GaussianResidue::zero(); 2]])
                .is_err()
        );
        let before_invalid = rows.columns().to_vec();
        let invalid_residue = GaussianResidue {
            real: GPU_FX_PRIMES[0],
            imaginary: 0,
        };
        assert!(
            rows.fold_batch(&[
                vec![GaussianResidue::zero(); 2],
                vec![GaussianResidue::zero(), invalid_residue],
            ])
            .is_err()
        );
        assert_eq!(rows.columns(), before_invalid);

        let mut same_rows_one_batch =
            GroupRowAccumulator::new("other-group", &[53], GPU_FX_PRIMES[0], &static_digest, 2)
                .unwrap();
        same_rows_one_batch
            .fold_batch(&[vec![
                GaussianResidue {
                    real: 11,
                    imaginary: 22,
                },
                GaussianResidue {
                    real: 33,
                    imaginary: 44,
                },
            ]])
            .unwrap();
        assert_eq!(
            rows.column_digests()[0],
            same_rows_one_batch.column_digests()[0]
        );
    }

    #[test]
    #[ignore = "builds the full real static functional data"]
    fn grouped_column_digest_matches_the_existing_single_column_contract() {
        let prime = GPU_FX_PRIMES[0];
        let static_data =
            crate::eleven_dimensional_second_momentum_gpu::ModularFxStaticData::build(prime)
                .unwrap();
        let input = crate::eleven_dimensional_second_momentum_gpu::GpuFxColumnInput {
            global_ordinal: 53,
            source_label: "synthetic".to_string(),
            source_copy: 1,
            terms: vec![term(1, 3)],
            raising_residuals: [0; 5],
        };
        let expected = crate::eleven_dimensional_second_momentum_gpu::accumulate_column_cpu(
            &static_data,
            &input,
        )
        .unwrap();
        let mut grouped = GroupRowAccumulator::new(
            "group",
            &[53],
            prime,
            static_data.semantic_sha256(),
            expected.rows.len(),
        )
        .unwrap();
        grouped.fold_batch(&[expected.rows]).unwrap();
        assert_eq!(grouped.column_digests(), vec![expected.semantic_sha256]);
    }

    #[test]
    fn observations_are_jsonl_and_flushable() {
        let mut output = Vec::new();
        write_observation_jsonl(&mut output, &serde_json::json!({"event": "union"})).unwrap();
        assert_eq!(output, b"{\"event\":\"union\"}\n");
    }
}
