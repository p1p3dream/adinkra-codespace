//! Typed block-sparse target gauge-image basis from the exact level-18 maps.
//!
//! The seventy-seven committed checkpoints certify nonzero equivariant maps
//! onto multiplicity-one irreducible targets.  Restricted to each certified
//! irreducible image, every block is therefore an isomorphism of dimension
//! equal to the target irrep dimension.  This module assembles those blocks
//! without reconstructing their exterior-power coefficients.
//!
//! A physical target gauge operator still requires two inputs not fixed by the
//! checkpoints: the routing of the six candidate channel coefficients into
//! the seventy-seven blocks, and the convention-fixed values of those
//! coefficients.  The APIs below keep both inputs explicit.  Their rank,
//! kernel, image-containment, and quotient results live in the direct sum of
//! certified incidence blocks.  They are not a physical superspace quotient
//! until the physical routing and coefficients are supplied.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: &str = "adynkra-11d-level18-target-quotient-basis-v1";
const CHECKPOINT_SCHEMA: &str = "adynkra-11d-level18-embedded-maps-v1-embedded-v1";
const CHANNEL_LABELS: [&str; 6] = [
    "k_0_scalar_00000",
    "k_1_vector_10000",
    "k_2_two_form_01000",
    "k_3_three_form_00100",
    "k_4_four_form_00010",
    "k_5_five_form_00002",
];
const CHANNEL_PARAMETER_DIMENSIONS: [u64; 6] = [1, 11, 55, 165, 330, 462];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct EmbeddedBlockKey {
    pub target_dynkin_label: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
}

impl EmbeddedBlockKey {
    pub fn canonical_name(&self) -> String {
        format!(
            "{}_from_{}_copy{}",
            self.target_dynkin_label, self.source_dynkin_label, self.source_copy
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
struct EmbeddedCertificate {
    source_dynkin_label: String,
    source_copy: usize,
    target_dynkin_label: String,
    exact_raising_residual_terms_by_simple_root: [usize; 5],
    shared_abstract_coupling_applied: bool,
    passed: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct EmbeddedCheckpoint {
    schema_version: String,
    job: EmbeddedBlockKey,
    source_fixture_sha256: String,
    abstract_certificate_sha256: String,
    coupled_map_sha256: String,
    target_dimension: u64,
    certified_irrep_image_rank: u64,
    certificate: EmbeddedCertificate,
    passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TypedCertifiedBlock {
    pub key: EmbeddedBlockKey,
    pub canonical_map_sha256: String,
    pub checkpoint_content_sha256: String,
    pub source_fixture_sha256: String,
    pub abstract_certificate_sha256: String,
    pub domain_offset: u64,
    pub codomain_offset: u64,
    pub certified_irrep_domain_dimension: u64,
    pub target_codomain_dimension: u64,
    pub certified_rank: u64,
    pub restricted_map_is_isomorphism: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CheckpointGaugeImageBasis {
    pub blocks: Vec<TypedCertifiedBlock>,
    pub block_count: usize,
    pub target_counts: BTreeMap<String, usize>,
    pub target_dimensions: BTreeMap<String, u64>,
    pub certified_domain_dimension: u64,
    pub target_codomain_dimension: u64,
    pub every_checkpoint_exact: bool,
}

impl CheckpointGaugeImageBasis {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let embedded = crate::eleven_dimensional_level18_embedded::verify_in(directory);
        if !embedded.all_77_embedded_maps_complete
            || embedded.exact_abstract_source_target_pairs != 34
            || embedded.exact_embedded_maps != 77
            || !embedded.every_completed_residual_is_zero
            || !embedded.every_completed_map_hash_is_unique_within_source_copy
        {
            return Err(
                "authoritative level-18 embedded-map provenance gate is incomplete".to_string(),
            );
        }
        let mut files = fs::read_dir(directory)
            .map_err(|error| format!("read checkpoint directory: {error}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("embedded_") && name.ends_with(".json"))
            })
            .collect::<Vec<_>>();
        files.sort();
        if files.len() != 77 {
            return Err(format!(
                "expected 77 embedded checkpoints, found {}",
                files.len()
            ));
        }

        let expected_dimensions = BTreeMap::from([
            ("01001".to_string(), 1_408_u64),
            ("10001".to_string(), 320_u64),
            ("11001".to_string(), 10_240_u64),
            ("20001".to_string(), 1_760_u64),
        ]);
        let mut checkpoints = Vec::with_capacity(files.len());
        let mut seen = BTreeSet::new();
        for path in files {
            let bytes =
                fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
            let checkpoint: EmbeddedCheckpoint = serde_json::from_slice(&bytes)
                .map_err(|error| format!("parse {}: {error}", path.display()))?;
            let expected_name = format!("embedded_{}.json", checkpoint.job.canonical_name());
            let actual_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            let expected_dimension = expected_dimensions
                .get(&checkpoint.job.target_dynkin_label)
                .copied()
                .ok_or_else(|| "unexpected target irrep in embedded checkpoint".to_string())?;
            let exact = checkpoint.schema_version == CHECKPOINT_SCHEMA
                && actual_name == expected_name
                && seen.insert(checkpoint.job.clone())
                && checkpoint.passed
                && checkpoint.certificate.passed
                && checkpoint.certificate.shared_abstract_coupling_applied
                && checkpoint.certificate.source_dynkin_label == checkpoint.job.source_dynkin_label
                && checkpoint.certificate.source_copy == checkpoint.job.source_copy
                && checkpoint.certificate.target_dynkin_label == checkpoint.job.target_dynkin_label
                && checkpoint
                    .certificate
                    .exact_raising_residual_terms_by_simple_root
                    == [0; 5]
                && is_sha256(&checkpoint.source_fixture_sha256)
                && is_sha256(&checkpoint.abstract_certificate_sha256)
                && is_sha256(&checkpoint.coupled_map_sha256)
                && checkpoint.target_dimension == expected_dimension
                && checkpoint.certified_irrep_image_rank == expected_dimension;
            if !exact {
                return Err(format!(
                    "embedded checkpoint failed exact provenance: {}",
                    path.display()
                ));
            }
            checkpoints.push((checkpoint, sha256(&bytes)));
        }
        checkpoints.sort_by(|left, right| left.0.job.cmp(&right.0.job));

        let mut offset = 0_u64;
        let mut target_counts = BTreeMap::new();
        let blocks = checkpoints
            .into_iter()
            .map(|(checkpoint, checkpoint_content_sha256)| {
                *target_counts
                    .entry(checkpoint.job.target_dynkin_label.clone())
                    .or_insert(0) += 1;
                let block = TypedCertifiedBlock {
                    key: checkpoint.job,
                    canonical_map_sha256: checkpoint.coupled_map_sha256,
                    checkpoint_content_sha256,
                    source_fixture_sha256: checkpoint.source_fixture_sha256,
                    abstract_certificate_sha256: checkpoint.abstract_certificate_sha256,
                    domain_offset: offset,
                    codomain_offset: offset,
                    certified_irrep_domain_dimension: checkpoint.target_dimension,
                    target_codomain_dimension: checkpoint.target_dimension,
                    certified_rank: checkpoint.certified_irrep_image_rank,
                    restricted_map_is_isomorphism: true,
                };
                offset += block.target_codomain_dimension;
                block
            })
            .collect::<Vec<_>>();
        let expected_counts = BTreeMap::from([
            ("01001".to_string(), 18_usize),
            ("10001".to_string(), 8_usize),
            ("11001".to_string(), 38_usize),
            ("20001".to_string(), 13_usize),
        ]);
        if target_counts != expected_counts || offset != 439_904 {
            return Err(
                "embedded checkpoint census or dimension total is inconsistent".to_string(),
            );
        }
        Ok(Self {
            block_count: blocks.len(),
            blocks,
            target_counts,
            target_dimensions: expected_dimensions,
            certified_domain_dimension: offset,
            target_codomain_dimension: offset,
            every_checkpoint_exact: true,
        })
    }

    pub fn block(&self, key: &EmbeddedBlockKey) -> Option<&TypedCertifiedBlock> {
        self.blocks.iter().find(|block| &block.key == key)
    }

    pub fn parameterize(
        &self,
        routing: Vec<BlockChannelLinearForm>,
        physical_routing: bool,
    ) -> Result<ParameterizedTargetGaugeImage, String> {
        if routing.len() != self.blocks.len() {
            return Err("routing must provide one linear form for each of 77 blocks".to_string());
        }
        let routed = routing
            .into_iter()
            .map(|form| (form.block_key.clone(), form))
            .collect::<BTreeMap<_, _>>();
        if routed.len() != self.blocks.len()
            || self
                .blocks
                .iter()
                .any(|block| !routed.contains_key(&block.key))
        {
            return Err("routing keys do not match the exact checkpoint basis".to_string());
        }
        Ok(ParameterizedTargetGaugeImage {
            basis: self.clone(),
            routing: self
                .blocks
                .iter()
                .map(|block| routed[&block.key].clone())
                .collect(),
            physical_routing,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExactRational {
    pub numerator: String,
    pub denominator: String,
}

impl ExactRational {
    pub fn integer(value: i64) -> Self {
        Self {
            numerator: value.to_string(),
            denominator: "1".to_string(),
        }
    }

    fn value(&self) -> Result<Ratio<BigInt>, String> {
        let numerator = self
            .numerator
            .parse::<BigInt>()
            .map_err(|error| format!("invalid exact numerator: {error}"))?;
        let denominator = self
            .denominator
            .parse::<BigInt>()
            .map_err(|error| format!("invalid exact denominator: {error}"))?;
        if denominator.is_zero() {
            return Err("exact denominator is zero".to_string());
        }
        Ok(Ratio::new(numerator, denominator))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockChannelLinearForm {
    pub block_key: EmbeddedBlockKey,
    pub channel_weights: [ExactRational; 6],
}

impl BlockChannelLinearForm {
    fn is_identically_zero(&self) -> Result<bool, String> {
        Ok(self
            .channel_weights
            .iter()
            .map(ExactRational::value)
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .all(Zero::is_zero))
    }

    fn specialize(
        &self,
        coefficients: &ChannelCoefficientSpecialization,
    ) -> Result<Ratio<BigInt>, String> {
        let channel_values = coefficients
            .values
            .iter()
            .map(ExactRational::value)
            .collect::<Result<Vec<_>, _>>()?;
        self.channel_weights.iter().zip(channel_values).try_fold(
            Ratio::from_integer(BigInt::from(0)),
            |sum, (weight, value)| Ok(sum + weight.value()? * value),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChannelCoefficientSpecialization {
    pub values: [ExactRational; 6],
    pub physical_coefficients: bool,
}

impl ChannelCoefficientSpecialization {
    pub fn integers(values: [i64; 6], physical_coefficients: bool) -> Self {
        Self {
            values: values.map(ExactRational::integer),
            physical_coefficients,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KernelBlock {
    pub block_key: EmbeddedBlockKey,
    pub domain_offset: u64,
    pub dimension: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GaugeImageAnalysis {
    pub generic_over_q_of_six_channel_coefficients: bool,
    pub physical_routing: bool,
    pub physical_coefficients: bool,
    pub rank: u64,
    pub kernel_dimension: u64,
    pub image_dimension: u64,
    pub quotient_dimension: u64,
    pub active_blocks: usize,
    pub inactive_blocks: usize,
    pub kernel_blocks: Vec<KernelBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SpecializedBlock {
    pub block_key: EmbeddedBlockKey,
    pub exact_multiplier: ExactRational,
    pub active: bool,
    pub dimension: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SpecializedTargetGaugeImage {
    pub channel_coefficients: ChannelCoefficientSpecialization,
    pub blocks: Vec<SpecializedBlock>,
    pub analysis: GaugeImageAnalysis,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetImageRequest {
    pub block_key: EmbeddedBlockKey,
    pub requested_image_dimension: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockContainmentResult {
    pub block_key: EmbeddedBlockKey,
    pub requested_image_dimension: u64,
    pub gauge_image_dimension: u64,
    pub contained: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImageContainmentResult {
    pub every_requested_image_contained: bool,
    pub blocks: Vec<BlockContainmentResult>,
}

impl SpecializedTargetGaugeImage {
    pub fn rank(&self) -> u64 {
        self.analysis.rank
    }

    pub fn kernel_dimension(&self) -> u64 {
        self.analysis.kernel_dimension
    }

    pub fn kernel_blocks(&self) -> &[KernelBlock] {
        &self.analysis.kernel_blocks
    }

    pub fn quotient_dimension(&self) -> u64 {
        self.analysis.quotient_dimension
    }

    pub fn contains_image(
        &self,
        requests: &[TargetImageRequest],
    ) -> Result<ImageContainmentResult, String> {
        let by_key = self
            .blocks
            .iter()
            .map(|block| (block.block_key.clone(), block))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        let mut blocks = Vec::with_capacity(requests.len());
        for request in requests {
            if !seen.insert(request.block_key.clone()) {
                return Err("duplicate target image request block".to_string());
            }
            let gauge = by_key
                .get(&request.block_key)
                .ok_or_else(|| "target image request references an unknown block".to_string())?;
            if request.requested_image_dimension > gauge.dimension {
                return Err("requested image dimension exceeds target block dimension".to_string());
            }
            let gauge_image_dimension = if gauge.active { gauge.dimension } else { 0 };
            blocks.push(BlockContainmentResult {
                block_key: request.block_key.clone(),
                requested_image_dimension: request.requested_image_dimension,
                gauge_image_dimension,
                contained: request.requested_image_dimension <= gauge_image_dimension,
            });
        }
        Ok(ImageContainmentResult {
            every_requested_image_contained: blocks.iter().all(|block| block.contained),
            blocks,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParameterizedTargetGaugeImage {
    pub basis: CheckpointGaugeImageBasis,
    pub routing: Vec<BlockChannelLinearForm>,
    pub physical_routing: bool,
}

impl ParameterizedTargetGaugeImage {
    pub fn generic_analysis(&self) -> Result<GaugeImageAnalysis, String> {
        let active = self
            .routing
            .iter()
            .map(|form| Ok(!form.is_identically_zero()?))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(self.analysis_from_support(&active, true, false))
    }

    pub fn specialize(
        &self,
        coefficients: ChannelCoefficientSpecialization,
    ) -> Result<SpecializedTargetGaugeImage, String> {
        let multipliers = self
            .routing
            .iter()
            .map(|form| form.specialize(&coefficients))
            .collect::<Result<Vec<_>, _>>()?;
        let active = multipliers
            .iter()
            .map(|multiplier| !multiplier.is_zero())
            .collect::<Vec<_>>();
        let analysis =
            self.analysis_from_support(&active, false, coefficients.physical_coefficients);
        let blocks = self
            .basis
            .blocks
            .iter()
            .zip(multipliers)
            .zip(active)
            .map(|((block, multiplier), active)| SpecializedBlock {
                block_key: block.key.clone(),
                exact_multiplier: ExactRational {
                    numerator: multiplier.numer().to_string(),
                    denominator: multiplier.denom().to_string(),
                },
                active,
                dimension: block.target_codomain_dimension,
            })
            .collect();
        Ok(SpecializedTargetGaugeImage {
            channel_coefficients: coefficients,
            blocks,
            analysis,
        })
    }

    fn analysis_from_support(
        &self,
        active: &[bool],
        generic: bool,
        physical_coefficients: bool,
    ) -> GaugeImageAnalysis {
        assert_eq!(active.len(), self.basis.blocks.len());
        let rank = self
            .basis
            .blocks
            .iter()
            .zip(active)
            .filter(|(_, active)| **active)
            .map(|(block, _)| block.certified_rank)
            .sum::<u64>();
        let kernel_blocks = self
            .basis
            .blocks
            .iter()
            .zip(active)
            .filter(|(_, active)| !**active)
            .map(|(block, _)| KernelBlock {
                block_key: block.key.clone(),
                domain_offset: block.domain_offset,
                dimension: block.certified_irrep_domain_dimension,
            })
            .collect::<Vec<_>>();
        let kernel_dimension = self.basis.certified_domain_dimension - rank;
        GaugeImageAnalysis {
            generic_over_q_of_six_channel_coefficients: generic,
            physical_routing: self.physical_routing,
            physical_coefficients,
            rank,
            kernel_dimension,
            image_dimension: rank,
            quotient_dimension: self.basis.target_codomain_dimension - rank,
            active_blocks: active.iter().filter(|value| **value).count(),
            inactive_blocks: active.iter().filter(|value| !**value).count(),
            kernel_blocks,
        }
    }
}

/// Deterministic nonphysical routing used only to verify the exact APIs.
///
/// Most blocks receive one channel variable.  Every seventh block receives
/// `k_0-k_1`, creating an exact special-locus rank drop at `k_0=k_1` while
/// remaining active over the generic rational-function field.
pub fn synthetic_control_routing(basis: &CheckpointGaugeImageBasis) -> Vec<BlockChannelLinearForm> {
    basis
        .blocks
        .iter()
        .enumerate()
        .map(|(ordinal, block)| {
            let mut channel_weights = std::array::from_fn(|_| ExactRational::integer(0));
            if ordinal % 7 == 0 {
                channel_weights[0] = ExactRational::integer(1);
                channel_weights[1] = ExactRational::integer(-1);
            } else {
                channel_weights[ordinal % 6] = ExactRational::integer(1);
            }
            BlockChannelLinearForm {
                block_key: block.key.clone(),
                channel_weights,
            }
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetQuotientControlReport {
    pub generic_rank: u64,
    pub generic_kernel_dimension: u64,
    pub generic_quotient_dimension: u64,
    pub special_rank: u64,
    pub special_kernel_dimension: u64,
    pub special_quotient_dimension: u64,
    pub special_inactive_blocks: usize,
    pub zero_rank: u64,
    pub zero_kernel_dimension: u64,
    pub zero_quotient_dimension: u64,
    pub positive_image_containment_passed: bool,
    pub negative_image_containment_rejected: bool,
    pub exact_cancellation_detected: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Level18TargetQuotientReport {
    pub schema_version: String,
    pub checkpoint_directory: String,
    pub exact_checkpoint_blocks: usize,
    pub target_counts: BTreeMap<String, usize>,
    pub target_dimensions: BTreeMap<String, u64>,
    pub certified_incidence_domain_dimension: u64,
    pub certified_incidence_codomain_dimension: u64,
    pub typed_basis_sha256: String,
    pub channel_labels: Vec<String>,
    pub channel_parameter_dimensions: Vec<u64>,
    pub physical_block_channel_routing_available: bool,
    pub physical_channel_coefficients_available: bool,
    pub parameterized_operator_ready_for_specialization: bool,
    pub exact_rank_api_available: bool,
    pub exact_kernel_api_available: bool,
    pub exact_image_containment_api_available: bool,
    pub exact_quotient_dimension_api_available: bool,
    pub controls: TargetQuotientControlReport,
    pub physical_target_gauge_quotient_complete: bool,
    pub passed: bool,
    pub result: String,
    pub boundary: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Level18TargetQuotientArtifact {
    pub schema_version: String,
    pub title: String,
    pub report: Level18TargetQuotientReport,
    pub basis: CheckpointGaugeImageBasis,
    pub synthetic_control_routing: Vec<BlockChannelLinearForm>,
}

pub fn verify_in(directory: &Path) -> Result<Level18TargetQuotientReport, String> {
    let basis = CheckpointGaugeImageBasis::load(directory)?;
    let typed_basis_sha256 = sha256(
        &serde_json::to_vec(&basis)
            .map_err(|error| format!("serialize typed checkpoint basis: {error}"))?,
    );
    let operator = basis.parameterize(synthetic_control_routing(&basis), false)?;
    let generic = operator.generic_analysis()?;
    let special = operator.specialize(ChannelCoefficientSpecialization::integers(
        [1, 1, 1, 1, 1, 1],
        false,
    ))?;
    let zero = operator.specialize(ChannelCoefficientSpecialization::integers(
        [0, 0, 0, 0, 0, 0],
        false,
    ))?;
    let active = special
        .blocks
        .iter()
        .find(|block| block.active)
        .expect("synthetic special routing has active blocks");
    let inactive = special
        .blocks
        .iter()
        .find(|block| !block.active)
        .expect("synthetic special routing has cancelled blocks");
    let positive = special.contains_image(&[TargetImageRequest {
        block_key: active.block_key.clone(),
        requested_image_dimension: active.dimension,
    }])?;
    let negative = special.contains_image(&[TargetImageRequest {
        block_key: inactive.block_key.clone(),
        requested_image_dimension: 1,
    }])?;
    let exact_cancellation_detected = special.analysis.rank < generic.rank
        && special.analysis.inactive_blocks > 0
        && special
            .blocks
            .iter()
            .filter(|block| !block.active)
            .all(|block| block.exact_multiplier == ExactRational::integer(0));
    let controls_passed = generic.rank == basis.target_codomain_dimension
        && generic.kernel_dimension == 0
        && generic.quotient_dimension == 0
        && special.analysis.rank + special.analysis.kernel_dimension
            == basis.certified_domain_dimension
        && special.analysis.rank + special.analysis.quotient_dimension
            == basis.target_codomain_dimension
        && zero.analysis.rank == 0
        && zero.analysis.kernel_dimension == basis.certified_domain_dimension
        && zero.analysis.quotient_dimension == basis.target_codomain_dimension
        && positive.every_requested_image_contained
        && !negative.every_requested_image_contained
        && exact_cancellation_detected;
    let controls = TargetQuotientControlReport {
        generic_rank: generic.rank,
        generic_kernel_dimension: generic.kernel_dimension,
        generic_quotient_dimension: generic.quotient_dimension,
        special_rank: special.analysis.rank,
        special_kernel_dimension: special.analysis.kernel_dimension,
        special_quotient_dimension: special.analysis.quotient_dimension,
        special_inactive_blocks: special.analysis.inactive_blocks,
        zero_rank: zero.analysis.rank,
        zero_kernel_dimension: zero.analysis.kernel_dimension,
        zero_quotient_dimension: zero.analysis.quotient_dimension,
        positive_image_containment_passed: positive.every_requested_image_contained,
        negative_image_containment_rejected: !negative.every_requested_image_contained,
        exact_cancellation_detected,
        passed: controls_passed,
    };
    Ok(Level18TargetQuotientReport {
        schema_version: SCHEMA_VERSION.to_string(),
        checkpoint_directory: directory.display().to_string(),
        exact_checkpoint_blocks: basis.block_count,
        target_counts: basis.target_counts.clone(),
        target_dimensions: basis.target_dimensions.clone(),
        certified_incidence_domain_dimension: basis.certified_domain_dimension,
        certified_incidence_codomain_dimension: basis.target_codomain_dimension,
        typed_basis_sha256,
        channel_labels: CHANNEL_LABELS.iter().map(|label| (*label).to_string()).collect(),
        channel_parameter_dimensions: CHANNEL_PARAMETER_DIMENSIONS.to_vec(),
        physical_block_channel_routing_available: false,
        physical_channel_coefficients_available: false,
        parameterized_operator_ready_for_specialization: true,
        exact_rank_api_available: true,
        exact_kernel_api_available: true,
        exact_image_containment_api_available: true,
        exact_quotient_dimension_api_available: true,
        controls,
        physical_target_gauge_quotient_complete: false,
        passed: basis.every_checkpoint_exact && controls_passed,
        result: "The seventy-seven exact checkpoint maps form a typed 439,904-dimensional direct sum of certified irreducible incidence blocks. Exact six-parameter specialization and quotient APIs are executable.".to_string(),
        boundary: "This is an exact quotient on the direct sum of certified source-target incidence blocks. The checkpoints do not determine how a convention-fixed physical K routes its six channel coefficients into those blocks, nor do the cited sources fix those coefficients. The supplied generic and special routings are synthetic controls. No physical target gauge quotient is claimed.".to_string(),
    })
}

pub fn verify() -> Result<Level18TargetQuotientReport, String> {
    verify_in(Path::new("results/eleven_dimensional_level18_embedded"))
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    serde_json::to_writer_pretty(BufWriter::new(File::create(&temporary)?), value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    fs::rename(temporary, path)
}

pub fn write_artifact(directory: &Path, output: &Path) -> io::Result<Level18TargetQuotientReport> {
    let report =
        verify_in(directory).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let basis = CheckpointGaugeImageBasis::load(directory)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let synthetic_control_routing = synthetic_control_routing(&basis);
    let basis_sha256 = sha256(
        &serde_json::to_vec(&basis)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
    );
    if basis_sha256 != report.typed_basis_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "typed basis changed during artifact construction",
        ));
    }
    let artifact = Level18TargetQuotientArtifact {
        schema_version: format!("{SCHEMA_VERSION}-artifact-v1"),
        title: "Parameterized level-18 target gauge-image quotient basis".to_string(),
        report: report.clone(),
        basis,
        synthetic_control_routing,
    };
    atomic_json(output, &artifact)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_all_77_exact_checkpoint_blocks_without_reconstruction() {
        let basis = CheckpointGaugeImageBasis::load(Path::new(
            "results/eleven_dimensional_level18_embedded",
        ))
        .unwrap();
        assert_eq!(basis.block_count, 77);
        assert_eq!(basis.certified_domain_dimension, 439_904);
        assert_eq!(basis.target_codomain_dimension, 439_904);
        assert!(basis.every_checkpoint_exact);
        assert!(basis.blocks.iter().all(|block| {
            block.certified_rank == block.target_codomain_dimension
                && block.restricted_map_is_isomorphism
        }));
    }

    #[test]
    fn generic_and_special_exact_ranks_obey_rank_nullity() {
        let basis = CheckpointGaugeImageBasis::load(Path::new(
            "results/eleven_dimensional_level18_embedded",
        ))
        .unwrap();
        let operator = basis
            .parameterize(synthetic_control_routing(&basis), false)
            .unwrap();
        let generic = operator.generic_analysis().unwrap();
        assert_eq!(generic.rank, 439_904);
        assert_eq!(generic.kernel_dimension, 0);
        assert_eq!(generic.quotient_dimension, 0);
        let special = operator
            .specialize(ChannelCoefficientSpecialization::integers(
                [1, 1, 1, 1, 1, 1],
                false,
            ))
            .unwrap();
        assert!(special.rank() < generic.rank);
        assert_eq!(special.rank() + special.kernel_dimension(), 439_904);
        assert_eq!(special.rank() + special.quotient_dimension(), 439_904);
        assert_eq!(
            special.kernel_dimension(),
            special
                .kernel_blocks()
                .iter()
                .map(|block| block.dimension)
                .sum::<u64>()
        );
    }

    #[test]
    fn exact_image_containment_accepts_active_and_rejects_cancelled_blocks() {
        let basis = CheckpointGaugeImageBasis::load(Path::new(
            "results/eleven_dimensional_level18_embedded",
        ))
        .unwrap();
        let operator = basis
            .parameterize(synthetic_control_routing(&basis), false)
            .unwrap();
        let special = operator
            .specialize(ChannelCoefficientSpecialization::integers(
                [1, 1, 1, 1, 1, 1],
                false,
            ))
            .unwrap();
        let active = special.blocks.iter().find(|block| block.active).unwrap();
        let inactive = special.blocks.iter().find(|block| !block.active).unwrap();
        assert!(
            special
                .contains_image(&[TargetImageRequest {
                    block_key: active.block_key.clone(),
                    requested_image_dimension: active.dimension,
                }])
                .unwrap()
                .every_requested_image_contained
        );
        assert!(
            !special
                .contains_image(&[TargetImageRequest {
                    block_key: inactive.block_key.clone(),
                    requested_image_dimension: 1,
                }])
                .unwrap()
                .every_requested_image_contained
        );
    }

    #[test]
    fn physical_gate_remains_false_without_routing_and_coefficients() {
        let report = verify().unwrap();
        assert!(report.passed);
        assert!(report.parameterized_operator_ready_for_specialization);
        assert!(!report.physical_block_channel_routing_available);
        assert!(!report.physical_channel_coefficients_available);
        assert!(!report.physical_target_gauge_quotient_complete);
        assert!(report.controls.passed);
    }

    #[test]
    #[ignore = "writes the parameterized target quotient artifact"]
    fn write_committed_artifact() {
        let report = write_artifact(
            Path::new("results/eleven_dimensional_level18_embedded"),
            Path::new("results/adynkra_11d_level18_target_quotient_basis.json"),
        )
        .unwrap();
        assert!(report.passed);
        assert!(!report.physical_target_gauge_quotient_complete);
    }
}
