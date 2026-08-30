//! Source-bound gate for the physical target-gauge map `K`.
//!
//! Four maps must remain distinct:
//!
//! * `G_q: Lambda_[q] -> Psi_source` are the six independent source-gauge maps;
//! * `A_j: Psi_source -> H_hat` are the seventy-seven candidate response maps;
//! * `K: Xi_target -> H_hat` is the physical target-gauge map;
//! * `F: H_hat -> curvature/equation data` is the physical target operator.
//!
//! The cited primary sources fix the gamma-traceless target projector and the
//! local gamma-trace redundancy, but do not print an exact `K` from a declared
//! target-gauge parameter domain.  This module therefore supplies a strict
//! specification and validation boundary.  It cannot manufacture the missing
//! convention.  A map reaches the existing exact quotient machinery only
//! after its provenance, bases, and induced routing have all been bound.  A
//! separate exact `F K = 0` certificate is required before promoting it into
//! the completed equation complex.

use num_bigint::BigInt;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::eleven_dimensional_level18_target_quotient::{
    BlockChannelLinearForm, ChannelCoefficientSpecialization, CheckpointGaugeImageBasis,
    SpecializedTargetGaugeImage,
};

pub const SCHEMA_VERSION: &str = "adynkra-11d-physical-k-specification-v1";
pub const AUDIT_SCHEMA_VERSION: &str = "adynkra-11d-physical-k-determination-audit-v1";
pub const ARXIV_2002_08502_PDF_SHA256: &str =
    "62587ef23aa92fd30bb7d978cc4e628275a18dd14fcb10fdf2020906638e554c";
pub const ARXIV_2007_05097_PDF_SHA256: &str =
    "197604bc6b5c9e0dfb12044d981aae467920f46554ba9371f1eb9b6389d00a73";
pub const HEP_TH_0101037_PDF_SHA256: &str =
    "3d40a1b32fa4491dee56b3e99802172d2c5039b2de198b987ce121a1bbb15cc3";

const SOURCE_CHANNEL_LABELS: [&str; 6] = ["00000", "10000", "01000", "00100", "00010", "00002"];
const SOURCE_CHANNEL_DIMENSIONS: [u64; 6] = [1, 11, 55, 165, 330, 462];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn basis_sha256(basis: &CheckpointGaugeImageBasis) -> Result<String, String> {
    serde_json::to_vec(basis)
        .map(|bytes| sha256(&bytes))
        .map_err(|error| format!("serialize exact quotient basis: {error}"))
}

fn exact_rational_is_zero(
    value: &crate::eleven_dimensional_level18_target_quotient::ExactRational,
) -> Result<bool, String> {
    let numerator = value
        .numerator
        .parse::<BigInt>()
        .map_err(|error| format!("invalid routing numerator: {error}"))?;
    let denominator = value
        .denominator
        .parse::<BigInt>()
        .map_err(|error| format!("invalid routing denominator: {error}"))?;
    if denominator.is_zero() {
        return Err("routing denominator is zero".to_string());
    }
    Ok(numerator.is_zero())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalKAuthority {
    PrintedPrimaryEquation,
    AuthorConfirmedConvention,
    ExactKernelOfCompleteF,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceReference {
    pub work: String,
    pub equation_or_section: String,
    pub pdf_page: u64,
    pub pdf_sha256: String,
    pub supports_exact_k_formula: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalKSpecification {
    pub schema_version: String,
    pub source_prepotential_symbol: String,
    pub target_gauge_parameter_symbol: String,
    pub target_gauge_parameter_domain: String,
    pub target_gauge_parameter_dimension: u64,
    pub target_superfield_symbol: String,
    pub target_superfield_dynkin_label: String,
    pub target_superfield_dimension: u64,
    pub target_map_formula: String,
    pub target_map_derivative_order: u64,
    pub authority: PhysicalKAuthority,
    pub authority_record_sha256: String,
    pub source_references: Vec<SourceReference>,
    pub typed_incidence_basis_sha256: String,
    pub induced_incidence_routing: Vec<BlockChannelLinearForm>,
    pub induced_source_channel_normalizations: ChannelCoefficientSpecialization,
    pub six_source_channels_are_independent_domains: bool,
    pub six_source_channels_identified_with_target_gauge_domain: bool,
    pub source_channels_treated_as_cancellable_scalars: bool,
    pub uses_projected_local_gamma_trace_as_nonzero_k: bool,
    pub synthetic_or_control_input: bool,
    pub complete_f_operator_sha256: String,
    pub fk_zero_certificate_sha256: String,
    pub fk_is_exactly_zero: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ValidatedPhysicalKSummary {
    pub specification_sha256: String,
    pub typed_incidence_basis_sha256: String,
    pub target_gauge_parameter_domain: String,
    pub target_gauge_parameter_dimension: u64,
    pub active_incidence_blocks: usize,
    pub gauge_image_rank: u64,
    pub target_quotient_dimension: u64,
    pub exact_fk_zero: bool,
    pub physical_target_gauge_quotient_complete: bool,
}

#[derive(Clone, Debug)]
pub struct ValidatedPhysicalK {
    pub summary: ValidatedPhysicalKSummary,
    pub quotient: SpecializedTargetGaugeImage,
}

impl PhysicalKSpecification {
    pub fn validate_against(
        &self,
        basis: &CheckpointGaugeImageBasis,
    ) -> Result<ValidatedPhysicalK, String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err("unsupported physical K specification schema".to_string());
        }
        if self.source_prepotential_symbol != "Psi_source_alpha"
            || self.target_gauge_parameter_symbol == self.source_prepotential_symbol
        {
            return Err(
                "source prepotential and target gauge parameter must be distinct typed roles"
                    .to_string(),
            );
        }
        if self.target_gauge_parameter_symbol.is_empty()
            || self.target_gauge_parameter_domain.is_empty()
            || self.target_gauge_parameter_dimension == 0
        {
            return Err("target gauge-parameter domain is not specified".to_string());
        }
        if self.target_superfield_symbol != "H_hat_alpha^a"
            || self.target_superfield_dynkin_label != "10001"
            || self.target_superfield_dimension != 320
        {
            return Err("physical K codomain must be the gamma-traceless 320 target".to_string());
        }
        if self.target_map_formula.trim().is_empty() {
            return Err("physical K formula is empty".to_string());
        }
        if self.uses_projected_local_gamma_trace_as_nonzero_k {
            return Err(
                "P_320 Gamma Lambda is exactly zero and cannot be used as a nonzero physical K"
                    .to_string(),
            );
        }
        if !self.six_source_channels_are_independent_domains
            || self.source_channels_treated_as_cancellable_scalars
        {
            return Err(
                "the six inequivalent source G_q domains cannot cancel as scalar coefficients"
                    .to_string(),
            );
        }
        for form in &self.induced_incidence_routing {
            let active_channels = form
                .channel_weights
                .iter()
                .map(exact_rational_is_zero)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .filter(|zero| !zero)
                .count();
            if active_channels > 1 {
                return Err(
                    "one incidence block mixes inequivalent source domains as cancellable scalars"
                        .to_string(),
                );
            }
        }
        if !self.six_source_channels_identified_with_target_gauge_domain {
            return Err(
                "no physical identification between the source G_q domains and the target K domain was supplied"
                    .to_string(),
            );
        }
        if self.synthetic_or_control_input {
            return Err("synthetic/control routing cannot be promoted to physical K".to_string());
        }
        if !self
            .induced_source_channel_normalizations
            .physical_coefficients
        {
            return Err("channel normalizations are not marked physical".to_string());
        }
        if self.source_references.is_empty()
            || self
                .source_references
                .iter()
                .any(|reference| !is_sha256(&reference.pdf_sha256))
        {
            return Err("physical K source provenance is incomplete".to_string());
        }
        match self.authority {
            PhysicalKAuthority::PrintedPrimaryEquation => {
                if !self
                    .source_references
                    .iter()
                    .any(|reference| reference.supports_exact_k_formula)
                {
                    return Err(
                        "no cited primary equation supports the exact K formula".to_string()
                    );
                }
            }
            PhysicalKAuthority::AuthorConfirmedConvention => {
                if !is_sha256(&self.authority_record_sha256) {
                    return Err(
                        "author-confirmed K is missing a bound confirmation record".to_string()
                    );
                }
            }
            PhysicalKAuthority::ExactKernelOfCompleteF => {
                if !is_sha256(&self.complete_f_operator_sha256) {
                    return Err("derived K is not bound to a complete F operator".to_string());
                }
            }
        }
        let any_fk_claim = !self.complete_f_operator_sha256.is_empty()
            || !self.fk_zero_certificate_sha256.is_empty()
            || self.fk_is_exactly_zero;
        if any_fk_claim
            && (!is_sha256(&self.complete_f_operator_sha256)
                || !is_sha256(&self.fk_zero_certificate_sha256)
                || !self.fk_is_exactly_zero)
        {
            return Err("partial or unbound F K = 0 claim".to_string());
        }
        let actual_basis_sha256 = basis_sha256(basis)?;
        if self.typed_incidence_basis_sha256 != actual_basis_sha256 {
            return Err(
                "physical K incidence basis digest does not match the exact 77-block basis"
                    .to_string(),
            );
        }

        let operator = basis.parameterize(self.induced_incidence_routing.clone(), true)?;
        let quotient = operator.specialize(self.induced_source_channel_normalizations.clone())?;
        let specification_sha256 = sha256(
            &serde_json::to_vec(self)
                .map_err(|error| format!("serialize physical K specification: {error}"))?,
        );
        let summary = ValidatedPhysicalKSummary {
            specification_sha256,
            typed_incidence_basis_sha256: actual_basis_sha256,
            target_gauge_parameter_domain: self.target_gauge_parameter_domain.clone(),
            target_gauge_parameter_dimension: self.target_gauge_parameter_dimension,
            active_incidence_blocks: quotient.analysis.active_blocks,
            gauge_image_rank: quotient.analysis.rank,
            target_quotient_dimension: quotient.analysis.quotient_dimension,
            exact_fk_zero: self.fk_is_exactly_zero,
            physical_target_gauge_quotient_complete: quotient.analysis.physical_routing
                && quotient.analysis.physical_coefficients,
        };
        Ok(ValidatedPhysicalK { summary, quotient })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MapRoleBoundary {
    pub symbol: &'static str,
    pub domain: &'static str,
    pub codomain: &'static str,
    pub count: usize,
    pub status: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceFixedStatement {
    pub work: &'static str,
    pub equation_or_section: &'static str,
    pub pdf_page: u64,
    pub pdf_sha256: &'static str,
    pub statement: &'static str,
    pub fixes_physical_k: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceGaugeChannelAudit {
    pub form_degree: usize,
    pub parameter_dynkin_label: &'static str,
    pub parameter_dimension: u64,
    pub map: &'static str,
    pub independent_parameter_domain: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalKDeterminationAudit {
    pub schema_version: &'static str,
    pub map_roles: Vec<MapRoleBoundary>,
    pub source_fixed_statements: Vec<SourceFixedStatement>,
    pub source_gauge_channels: Vec<SourceGaugeChannelAudit>,
    pub source_channel_count: usize,
    pub source_parameter_dimension_sum: u64,
    pub exact_incidence_block_count: usize,
    pub exact_incidence_dimension: u64,
    pub typed_incidence_basis_sha256: String,
    pub incidence_blocks_are_second_momentum_response_columns: bool,
    pub physical_identification_between_incidence_and_response_bases_available: bool,
    pub exact_k_formula_present_in_audited_sources: bool,
    pub target_gauge_parameter_domain_fixed: bool,
    pub six_source_channels_are_independent_domains: bool,
    pub six_source_channels_are_physical_k_coefficients: bool,
    pub projected_local_gamma_trace_has_nonzero_h_hat_image: bool,
    pub complete_f_available: bool,
    pub exact_fk_zero_certificate_available: bool,
    pub physical_k_specification_ready_to_accept: bool,
    pub physical_k_validated: bool,
    pub physical_target_gauge_quotient_complete: bool,
    pub required_inputs: Vec<&'static str>,
    pub questions_for_convention_owner: Vec<&'static str>,
    pub next_executable_step: &'static str,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

pub fn verify_in(checkpoint_directory: &Path) -> Result<PhysicalKDeterminationAudit, String> {
    let basis = CheckpointGaugeImageBasis::load(checkpoint_directory)?;
    let typed_incidence_basis_sha256 = basis_sha256(&basis)?;
    let source_gauge_channels = SOURCE_CHANNEL_LABELS
        .into_iter()
        .zip(SOURCE_CHANNEL_DIMENSIONS)
        .enumerate()
        .map(
            |(form_degree, (parameter_dynkin_label, parameter_dimension))| {
                SourceGaugeChannelAudit {
                    form_degree,
                    parameter_dynkin_label,
                    parameter_dimension,
                    map: "G_q: Lambda_[q] -> Psi_source_alpha",
                    independent_parameter_domain: true,
                }
            },
        )
        .collect::<Vec<_>>();
    let source_parameter_dimension_sum = source_gauge_channels
        .iter()
        .map(|channel| channel.parameter_dimension)
        .sum();
    let exact_basis = basis.block_count == 77
        && basis.certified_domain_dimension == 439_904
        && basis.target_codomain_dimension == 439_904
        && basis.every_checkpoint_exact;

    Ok(PhysicalKDeterminationAudit {
        schema_version: AUDIT_SCHEMA_VERSION,
        map_roles: vec![
            MapRoleBoundary {
                symbol: "G_q",
                domain: "Lambda_[q], q=0..5",
                codomain: "Psi_source_alpha",
                count: 6,
                status: "exact source maps available; parameter domains are independent",
            },
            MapRoleBoundary {
                symbol: "A_j",
                domain: "Psi_source_alpha",
                codomain: "H_hat_alpha^a and its declared representation-level targets",
                count: 77,
                status: "all canonical representation-level response columns computed",
            },
            MapRoleBoundary {
                symbol: "K",
                domain: "Xi_target, not yet fixed",
                codomain: "H_hat_alpha^a in (10001), dimension 320",
                count: 0,
                status: "missing physical target-gauge domain and exact map",
            },
            MapRoleBoundary {
                symbol: "F",
                domain: "H_hat_alpha^a",
                codomain: "complete curvature, Bianchi, Euler, and Noether complex",
                count: 0,
                status: "X_[2]/X_[5] slice exists; complete physical F is unfinished",
            },
        ],
        source_fixed_statements: vec![
            SourceFixedStatement {
                work: "arXiv:2002.08502",
                equation_or_section: "Eq. (6.3), Added Note in Proof",
                pdf_page: 44,
                pdf_sha256: ARXIV_2002_08502_PDF_SHA256,
                statement: "V = D^alpha Psi_alpha is proposed for an unconstrained spinor prepotential",
                fixes_physical_k: false,
            },
            SourceFixedStatement {
                work: "arXiv:2007.05097",
                equation_or_section: "Eqs. (2.1)-(2.3)",
                pdf_page: 7,
                pdf_sha256: ARXIV_2007_05097_PDF_SHA256,
                statement: "H_hat = P_320 H and delta H_beta^b = (gamma^b)_beta^alpha Lambda_alpha; P_320 gamma Lambda = 0",
                fixes_physical_k: false,
            },
            SourceFixedStatement {
                work: "arXiv:2007.05097",
                equation_or_section: "Eq. (2.7)",
                pdf_page: 9,
                pdf_sha256: ARXIV_2007_05097_PDF_SHA256,
                statement: "a scalar-factorized H(V) route would involve fifteen spinor derivatives, but the functional is not printed",
                fixes_physical_k: false,
            },
            SourceFixedStatement {
                work: "hep-th/0101037",
                equation_or_section: "Eqs. (24)-(29), (39)-(40), and (44)",
                pdf_page: 7,
                pdf_sha256: HEP_TH_0101037_PDF_SHA256,
                statement: "linearized frame, anholonomy, conventional quotient, and X/J/W definitions are printed; H remains a semi-prepotential with unknown differential constraints",
                fixes_physical_k: false,
            },
        ],
        source_gauge_channels,
        source_channel_count: 6,
        source_parameter_dimension_sum,
        exact_incidence_block_count: basis.block_count,
        exact_incidence_dimension: basis.target_codomain_dimension,
        typed_incidence_basis_sha256,
        incidence_blocks_are_second_momentum_response_columns: false,
        physical_identification_between_incidence_and_response_bases_available: false,
        exact_k_formula_present_in_audited_sources: false,
        target_gauge_parameter_domain_fixed: false,
        six_source_channels_are_independent_domains: true,
        six_source_channels_are_physical_k_coefficients: false,
        projected_local_gamma_trace_has_nonzero_h_hat_image: false,
        complete_f_available: false,
        exact_fk_zero_certificate_available: false,
        physical_k_specification_ready_to_accept: true,
        physical_k_validated: false,
        physical_target_gauge_quotient_complete: false,
        required_inputs: vec![
            "a distinct target gauge-parameter domain Xi_target with bound basis and dimension",
            "the exact convention-fixed formula K: Xi_target -> H_hat, including derivative order and normalization",
            "a source equation, author-confirmation record, or derivation as the exact kernel of complete F",
            "the induced routing into the exact 77-block incidence basis",
            "an exact complete-F digest and F K = 0 certificate",
        ],
        questions_for_convention_owner: vec![
            "What is the target gauge-parameter superfield Xi_target and its Spin(1,10) representation?",
            "What is delta H_hat before and after the P_320 gamma-trace projection?",
            "Is Xi_target independent of the source prepotential Psi_source_alpha?",
            "What derivative order, gamma structures, relative signs, and rational normalizations define K?",
            "Are any of the six independent G_q source domains identified with Xi_target, and by what exact maps?",
            "Which complete curvature/equation operator F supplies the identity F K = 0?",
        ],
        next_executable_step: "obtain the missing K convention or finish complete F and solve its exact target-side kernel, then instantiate PhysicalKSpecification and validate it against the 77-block basis",
        passed: exact_basis && source_parameter_dimension_sum == 1_024,
        result: "The exact 77-block quotient backend is ready, and the physical K input boundary is now typed and fail-closed. The audited sources do not determine K.",
        boundary: "No target gauge quotient is called physical until a distinct target parameter domain, exact K, and induced routing validate. Promotion into the equation complex additionally requires an exact F K = 0 certificate. The six G_q maps are source-side maps on inequivalent domains, not cancellable coefficients of K.",
    })
}

pub fn verify() -> Result<PhysicalKDeterminationAudit, String> {
    verify_in(Path::new("results/eleven_dimensional_level18_embedded"))
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub fn write_audit(output: &Path) -> io::Result<PhysicalKDeterminationAudit> {
    let report = verify().map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    atomic_json(output, &report)?;
    Ok(report)
}

pub fn validate_specification_file(
    specification_path: &Path,
    checkpoint_directory: &Path,
    output: &Path,
) -> io::Result<ValidatedPhysicalKSummary> {
    let bytes = fs::read(specification_path)?;
    let specification: PhysicalKSpecification = serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let basis = CheckpointGaugeImageBasis::load(checkpoint_directory)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let validated = specification
        .validate_against(&basis)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    atomic_json(output, &validated.summary)?;
    Ok(validated.summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eleven_dimensional_level18_target_quotient::{EmbeddedBlockKey, ExactRational};

    fn real_basis() -> CheckpointGaugeImageBasis {
        CheckpointGaugeImageBasis::load(Path::new("results/eleven_dimensional_level18_embedded"))
            .unwrap()
    }

    fn valid_test_specification(basis: &CheckpointGaugeImageBasis) -> PhysicalKSpecification {
        let routing = basis
            .blocks
            .iter()
            .enumerate()
            .map(|(ordinal, block)| {
                let mut channel_weights = std::array::from_fn(|_| ExactRational::integer(0));
                channel_weights[ordinal % 6] = ExactRational::integer(1);
                BlockChannelLinearForm {
                    block_key: EmbeddedBlockKey {
                        target_dynkin_label: block.key.target_dynkin_label.clone(),
                        source_dynkin_label: block.key.source_dynkin_label.clone(),
                        source_copy: block.key.source_copy,
                    },
                    channel_weights,
                }
            })
            .collect();
        PhysicalKSpecification {
            schema_version: SCHEMA_VERSION.to_string(),
            source_prepotential_symbol: "Psi_source_alpha".to_string(),
            target_gauge_parameter_symbol: "Xi_target".to_string(),
            target_gauge_parameter_domain: "test exact direct sum".to_string(),
            target_gauge_parameter_dimension: 1_024,
            target_superfield_symbol: "H_hat_alpha^a".to_string(),
            target_superfield_dynkin_label: "10001".to_string(),
            target_superfield_dimension: 320,
            target_map_formula: "test-only exact K".to_string(),
            target_map_derivative_order: 1,
            authority: PhysicalKAuthority::ExactKernelOfCompleteF,
            authority_record_sha256: sha256(b"test authority"),
            source_references: vec![SourceReference {
                work: "test".to_string(),
                equation_or_section: "test".to_string(),
                pdf_page: 1,
                pdf_sha256: sha256(b"test source"),
                supports_exact_k_formula: false,
            }],
            typed_incidence_basis_sha256: basis_sha256(basis).unwrap(),
            induced_incidence_routing: routing,
            induced_source_channel_normalizations: ChannelCoefficientSpecialization::integers(
                [1; 6], true,
            ),
            six_source_channels_are_independent_domains: true,
            six_source_channels_identified_with_target_gauge_domain: true,
            source_channels_treated_as_cancellable_scalars: false,
            uses_projected_local_gamma_trace_as_nonzero_k: false,
            synthetic_or_control_input: false,
            complete_f_operator_sha256: sha256(b"test complete F"),
            fk_zero_certificate_sha256: sha256(b"test exact FK zero"),
            fk_is_exactly_zero: true,
        }
    }

    #[test]
    fn audit_separates_g_a_k_and_f_and_stays_fail_closed() {
        let audit = verify().unwrap();
        assert!(audit.passed);
        assert_eq!(audit.map_roles.len(), 4);
        assert_eq!(audit.source_channel_count, 6);
        assert_eq!(audit.source_parameter_dimension_sum, 1_024);
        assert_eq!(audit.exact_incidence_block_count, 77);
        assert!(!audit.incidence_blocks_are_second_momentum_response_columns);
        assert!(!audit.physical_identification_between_incidence_and_response_bases_available);
        assert!(!audit.exact_k_formula_present_in_audited_sources);
        assert!(!audit.six_source_channels_are_physical_k_coefficients);
        assert!(!audit.physical_k_validated);
        assert!(!audit.physical_target_gauge_quotient_complete);
    }

    #[test]
    fn complete_bound_specification_reaches_exact_quotient_backend() {
        let basis = real_basis();
        let validated = valid_test_specification(&basis)
            .validate_against(&basis)
            .unwrap();
        assert!(validated.summary.physical_target_gauge_quotient_complete);
        assert!(validated.summary.exact_fk_zero);
        assert_eq!(validated.summary.active_incidence_blocks, 77);
        assert_eq!(validated.summary.gauge_image_rank, 439_904);
        assert_eq!(validated.summary.target_quotient_dimension, 0);
    }

    #[test]
    fn source_authorized_k_can_define_quotient_before_complete_f() {
        let basis = real_basis();
        let mut specification = valid_test_specification(&basis);
        specification.authority = PhysicalKAuthority::AuthorConfirmedConvention;
        specification.complete_f_operator_sha256.clear();
        specification.fk_zero_certificate_sha256.clear();
        specification.fk_is_exactly_zero = false;
        let validated = specification.validate_against(&basis).unwrap();
        assert!(validated.summary.physical_target_gauge_quotient_complete);
        assert!(!validated.summary.exact_fk_zero);
    }

    #[test]
    fn projected_gamma_trace_cannot_be_promoted_to_nonzero_k() {
        let basis = real_basis();
        let mut specification = valid_test_specification(&basis);
        specification.uses_projected_local_gamma_trace_as_nonzero_k = true;
        let error = specification.validate_against(&basis).unwrap_err();
        assert!(error.contains("P_320 Gamma Lambda is exactly zero"));
    }

    #[test]
    fn independent_source_channels_cannot_be_cancelled_as_scalars() {
        let basis = real_basis();
        let mut specification = valid_test_specification(&basis);
        specification.source_channels_treated_as_cancellable_scalars = true;
        let error = specification.validate_against(&basis).unwrap_err();
        assert!(error.contains("cannot cancel as scalar coefficients"));
    }

    #[test]
    fn incomplete_or_unbound_fk_proof_is_rejected() {
        let basis = real_basis();
        let mut specification = valid_test_specification(&basis);
        specification.fk_is_exactly_zero = false;
        let error = specification.validate_against(&basis).unwrap_err();
        assert!(error.contains("partial or unbound F K = 0 claim"));
    }
}
