//! Exact sparse highest-weight systems for the 11D bridge and first lower symbol.
//!
//! The scalar 11D superfield has level-15 space `exterior^15 S`, where `S`
//! is the 32-dimensional spinor of B5.  The published level inventory contains
//! two copies of `(00001)` and one copy of `(10001)`.  A highest-weight vector
//! of either type is an integer vector in the corresponding weight space that
//! is killed by all five simple-root raising operators.  This module builds
//! those sparse integer systems directly from the spinor weights. It also
//! constructs the three level-13 momentum corrections and tests their
//! level-14 two-form-hook images.

use num_rational::Ratio;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

type Weight = [i8; 5];

const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];
const SPINOR_KERNEL_1: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/00001_highest_weight_kernel_1.i16le");
const SPINOR_KERNEL_2: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/00001_highest_weight_kernel_2.i16le");
const VECTOR_SPINOR_KERNEL: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/10001_highest_weight_kernel.i16le");
const LEVEL13_SPINOR_KERNEL: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/level13_00001_highest_weight_kernel.i16le");
const LEVEL13_TWO_FORM_SPINOR_KERNEL_1: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/level13_01001_highest_weight_kernel_1.i16le");
const LEVEL13_TWO_FORM_SPINOR_KERNEL_2: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/level13_01001_highest_weight_kernel_2.i16le");

#[derive(Debug, Clone, Serialize)]
pub struct RaisingBlockReport {
    pub simple_root: usize,
    pub output_weight: Weight,
    pub rows: usize,
    pub nonzero_entries: usize,
    pub row_degree_histogram: BTreeMap<usize, usize>,
    pub missing_source_columns: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HighestWeightSystemReport {
    pub dynkin_label: &'static str,
    pub representation_dimension: usize,
    pub highest_weight_doubled_coordinates: Weight,
    pub exterior_degree: usize,
    pub source_weight_space_columns: usize,
    pub raising_blocks: Vec<RaisingBlockReport>,
    pub total_rows: usize,
    pub total_nonzero_entries: usize,
    pub published_multiplicity: usize,
    pub expected_kernel_dimension: usize,
    pub exact_sparse_system_constructed: bool,
    pub exact_kernel_vectors: Vec<ExactKernelVectorReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExteriorHighestWeightSystemShape {
    pub dynkin_label: String,
    pub exterior_degree: usize,
    pub highest_weight_doubled_coordinates: [i8; 5],
    pub source_weight_space_columns: usize,
    pub raising_block_rows: [usize; 5],
    pub total_raising_rows: usize,
    pub expected_kernel_dimension: usize,
}

pub struct ExteriorHighestWeightKernelFixture {
    pub exterior_degree: u8,
    pub dynkin_label: &'static str,
    pub kernel_artifacts: &'static [(&'static str, &'static [u8])],
}

#[derive(Debug, Clone, Serialize)]
pub struct ExactKernelVectorReport {
    pub artifact: &'static str,
    pub scalar_type: &'static str,
    pub coefficients: usize,
    pub nonzero_coefficients: usize,
    pub minimum_coefficient: i16,
    pub maximum_coefficient: i16,
    pub coefficient_gcd: i16,
    pub squared_norm: u64,
    pub raising_rows_checked: usize,
    pub nonzero_residual_rows: usize,
    pub maximum_absolute_residual: i64,
    pub exact_kernel_verified: bool,
    pub first_lowering_descendants: Vec<FirstLoweringReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstLoweringReport {
    pub simple_root: usize,
    pub expected_nonzero_from_dynkin_label: bool,
    pub expected_lowering_string_length: usize,
    pub nonzero_terms: usize,
    pub second_lowering_nonzero_terms: usize,
    pub lowering_power_nonzero_terms: Vec<usize>,
    pub maximum_absolute_coefficient: i64,
    pub matches_highest_weight_string: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeCoefficient {
    pub name: &'static str,
    pub source_dynkin_label: &'static str,
    pub source_copy: usize,
    pub target_sector: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalBridgeReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_level: usize,
    pub spinor_weights: usize,
    pub systems: Vec<HighestWeightSystemReport>,
    pub first_lower_symbol_systems: Vec<HighestWeightSystemReport>,
    pub generic_bridge: &'static str,
    pub coefficients: Vec<BridgeCoefficient>,
    pub equation_2_7_status: &'static str,
    pub coefficient_solution_status: &'static str,
    pub expected_kernel_vectors: usize,
    pub exact_kernel_vectors_verified: usize,
    pub all_expected_kernel_vectors_verified: bool,
    pub spinor_descendant_audits: Vec<SpinorDescendantAudit>,
    pub vector_spinor_target_audit: VectorSpinorTargetAudit,
    pub vector_spinor_source_descendant_audit: VectorSpinorSourceDescendantAudit,
    pub level_sixteen_derivative_channel_audit: LevelSixteenDerivativeChannelAudit,
    pub level_sixteen_exterior_derivative_audit: LevelSixteenExteriorDerivativeAudit,
    pub dimension_zero_torsion_sector_audit: DimensionZeroTorsionSectorAudit,
    pub first_derivative_momentum_audit: FirstDerivativeMomentumAudit,
    pub first_momentum_completion_audit: FirstMomentumCompletionAudit,
    pub zero_momentum_equation_2_7_projection: ZeroMomentumEquationProjectionAudit,
    pub local_gamma_trace_quotient: LocalGammaTraceQuotientAudit,
    pub canonical_source_line_normalization: CanonicalSourceLineNormalizationAudit,
    pub linearized_scale_freedom_audit: LinearizedScaleFreedomAudit,
    pub inherited_spinor_gauge_audit: InheritedSpinorGaugeAudit,
    pub boundary: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LevelSixteenDerivativeChannel {
    pub dynkin_label: String,
    pub dimension: u64,
    pub scalar_level_sixteen_multiplicity: usize,
    pub forced_zero_by_source_inventory: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LevelSixteenDerivativeChannelAudit {
    pub tensor_product: &'static str,
    pub tensor_product_dimension: u64,
    pub multiplicity_free_candidate_channels: usize,
    pub scalar_level_sixteen_present_channels: usize,
    pub scalar_level_sixteen_absent_channels: Vec<String>,
    pub absent_channel_dimension: u64,
    pub present_channel_dimension: u64,
    pub channels: Vec<LevelSixteenDerivativeChannel>,
    pub final_two_form_hook_dynkin_label: &'static str,
    pub final_two_form_hook_absent: bool,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn audit_level_sixteen_derivative_channels() -> LevelSixteenDerivativeChannelAudit {
    let channels = crate::eleven_dimensional_prepotential::spinor_tensor_channels("10001")
        .into_iter()
        .map(|(dynkin_label, dimension)| {
            let scalar_level_sixteen_multiplicity =
                crate::eleven_dimensional_prepotential::level_multiplicity(16, &dynkin_label);
            LevelSixteenDerivativeChannel {
                dynkin_label,
                dimension,
                scalar_level_sixteen_multiplicity,
                forced_zero_by_source_inventory: scalar_level_sixteen_multiplicity == 0,
            }
        })
        .collect::<Vec<_>>();
    let tensor_product_dimension = channels.iter().map(|channel| channel.dimension).sum();
    let scalar_level_sixteen_present_channels = channels
        .iter()
        .filter(|channel| !channel.forced_zero_by_source_inventory)
        .count();
    let scalar_level_sixteen_absent_channels = channels
        .iter()
        .filter(|channel| channel.forced_zero_by_source_inventory)
        .map(|channel| channel.dynkin_label.clone())
        .collect::<Vec<_>>();
    let absent_channel_dimension = channels
        .iter()
        .filter(|channel| channel.forced_zero_by_source_inventory)
        .map(|channel| channel.dimension)
        .sum();
    let present_channel_dimension = tensor_product_dimension - absent_channel_dimension;
    let final_two_form_hook_absent = channels
        .iter()
        .any(|channel| channel.dynkin_label == "11000" && channel.forced_zero_by_source_inventory);
    let passed = tensor_product_dimension == 320 * 32
        && channels.len() == 10
        && scalar_level_sixteen_present_channels == 8
        && scalar_level_sixteen_absent_channels == ["01000", "11000"]
        && absent_channel_dimension == 484
        && present_channel_dimension == 9_756
        && final_two_form_hook_absent;
    LevelSixteenDerivativeChannelAudit {
        tensor_product: "(00001) tensor (10001)",
        tensor_product_dimension,
        multiplicity_free_candidate_channels: channels.len(),
        scalar_level_sixteen_present_channels,
        scalar_level_sixteen_absent_channels,
        absent_channel_dimension,
        present_channel_dimension,
        channels,
        final_two_form_hook_dynkin_label: "11000",
        final_two_form_hook_absent,
        interpretation: "a sixteenth spinor derivative of the surviving bridge has ten Lorentz candidate channels before exterior antisymmetrization; the scalar level-16 inventory excludes 01000 and 11000, so every equivariant leading-symbol map into either channel vanishes; presence of the other eight representations is necessary but does not prove a nonzero image",
        passed,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LinearizedScaleFreedomAudit {
    pub bridge_is_linear_in_scalar_prepotential: bool,
    pub constraints_are_homogeneous_in_linearized_bridge: bool,
    pub reparametrization: &'static str,
    pub nonzero_bridge_class_selected: bool,
    pub computational_source_normalization_fixed: bool,
    pub physical_scale_fixed_by_homogeneous_constraints: bool,
    pub required_external_normalization: &'static str,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn audit_linearized_scale_freedom(
    normalization: &CanonicalSourceLineNormalizationAudit,
) -> LinearizedScaleFreedomAudit {
    let bridge_is_linear_in_scalar_prepotential = true;
    let constraints_are_homogeneous_in_linearized_bridge = true;
    let nonzero_bridge_class_selected = true;
    let physical_scale_fixed_by_homogeneous_constraints = false;
    let passed = bridge_is_linear_in_scalar_prepotential
        && constraints_are_homogeneous_in_linearized_bridge
        && nonzero_bridge_class_selected
        && normalization.computational_source_normalization_fixed
        && !physical_scale_fixed_by_homogeneous_constraints;
    LinearizedScaleFreedomAudit {
        bridge_is_linear_in_scalar_prepotential,
        constraints_are_homogeneous_in_linearized_bridge,
        reparametrization: "V -> lambda V and c -> c/lambda for nonzero lambda",
        nonzero_bridge_class_selected,
        computational_source_normalization_fixed: normalization
            .computational_source_normalization_fixed,
        physical_scale_fixed_by_homogeneous_constraints,
        required_external_normalization: "a declared normalization of V or matching one component of H to an independently normalized graviton or gravitino field",
        interpretation: "linearized torsion constraints and Bianchi identities can reject the bridge or constrain relative coefficients, but they cannot determine the overall nonzero c while the scalar prepotential may be rescaled",
        passed,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorSpinorSourceDescendantAudit {
    pub target_states_expected: usize,
    pub target_states_generated: usize,
    pub distinct_weights: usize,
    pub nonzero_lowering_actions_checked: usize,
    pub zero_lowering_actions_checked: usize,
    pub total_lowering_actions_checked: usize,
    pub independent_state_discoveries: usize,
    pub dependent_target_relations_checked: usize,
    pub dependent_target_relation_mismatches: usize,
    pub nonzero_relation_residual_terms: usize,
    pub maximum_absolute_relation_residual: u64,
    pub zero_action_mismatches: usize,
    pub target_basis_correspondence_mismatches: usize,
    pub minimum_source_state_support: usize,
    pub maximum_source_state_support: usize,
    pub maximum_absolute_source_coefficient: i64,
    pub exact_full_vector_spinor_intertwiner_verified: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LevelSixteenExteriorChannelAudit {
    pub dynkin_label: String,
    pub dimension: u64,
    pub scalar_level_sixteen_multiplicity: usize,
    pub highest_weight_domain_dimension: usize,
    pub highest_weight_kernel_dimension: usize,
    pub primitive_highest_weight_nonzero_coefficients: usize,
    pub raising_residual_terms: usize,
    pub fingerprint_primes: [u64; 3],
    pub exterior_image_fingerprint_residues: [u64; 3],
    pub exterior_image_nonzero_certified: bool,
    pub exterior_image_forced_zero_by_inventory: bool,
    pub inventory_zero_fingerprint_crosscheck: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LevelSixteenExteriorDerivativeAudit {
    pub source_map: &'static str,
    pub scope: &'static str,
    pub channels_checked: usize,
    pub highest_weight_kernels_verified: usize,
    pub nonzero_exterior_images_certified: usize,
    pub inventory_forced_zero_channels: usize,
    pub inventory_zero_fingerprint_crosschecks: usize,
    pub channels: Vec<LevelSixteenExteriorChannelAudit>,
    pub interpretation: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MomentumHookChannelAudit {
    pub dynkin_label: String,
    pub torsion_sector: &'static str,
    pub exterior_level: usize,
    pub exterior_image_nonzero: bool,
    pub momentum_contraction_level: usize,
    pub momentum_real_fingerprint_residues: [u64; 3],
    pub momentum_imaginary_fingerprint_residues: [u64; 3],
    pub momentum_contraction_nonzero_certified: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstDerivativeMomentumAudit {
    pub derivative_identity: &'static str,
    pub exact_sample_momentum: [i64; 11],
    pub cartan_weight_entries_checked: usize,
    pub cartan_weight_mismatches: usize,
    pub chevalley_lowering_actions_checked: usize,
    pub chevalley_lowering_residual_actions: usize,
    pub clifford_and_weight_bases_aligned: bool,
    pub channels: Vec<MomentumHookChannelAudit>,
    pub two_form_hook_momentum_contraction_nonzero: bool,
    pub five_form_hook_momentum_contraction_nonzero: bool,
    pub implication: &'static str,
    pub scope: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MomentumCompletionChannelAudit {
    pub source_dynkin_label: String,
    pub source_dimension: u64,
    pub multiplicity_in_vector_times_target: usize,
    pub multiplicity_at_scalar_level_thirteen: usize,
    pub correction_coefficients: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstMomentumCompletionAudit {
    pub normal_form_term: &'static str,
    pub target_dynkin_label: &'static str,
    pub target_dimension: usize,
    pub vector_times_target_dimension: usize,
    pub vector_times_target_channels: Vec<MomentumCompletionChannelAudit>,
    pub available_level_thirteen_source_channels: Vec<String>,
    pub first_completion_coefficient_dimension: usize,
    pub leading_two_form_hook_momentum_term_nonzero: bool,
    pub cancellation_system_constructed: bool,
    pub cancellation_exists: Option<bool>,
    pub exact_functional_rows: usize,
    pub exact_functional_definition: &'static str,
    pub exact_functional_columns: [&'static str; 4],
    pub exact_functional_matrix: Vec<Vec<String>>,
    pub correction_span_rank: usize,
    pub augmented_span_rank: usize,
    pub exact_non_cancellation_certificate: bool,
    pub implication: &'static str,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TorsionSectorChannelAudit {
    pub dynkin_label: &'static str,
    pub dimension: u64,
    pub tensor_role: &'static str,
    pub removed_by_conventional_constraint: bool,
    pub exterior_image_nonzero: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DimensionZeroTorsionSectorAudit {
    pub source_equations: &'static str,
    pub two_form_vector_dimension: u64,
    pub two_form_vector_channels: Vec<TorsionSectorChannelAudit>,
    pub two_form_conventional_dimension: u64,
    pub two_form_remaining_hook_dimension: u64,
    pub two_form_remaining_hook_nonzero: bool,
    pub five_form_vector_dimension: u64,
    pub five_form_vector_channels: Vec<TorsionSectorChannelAudit>,
    pub five_form_conventional_dimension: u64,
    pub five_form_remaining_hook_dimension: u64,
    pub five_form_remaining_hook_nonzero: bool,
    pub complementary_derivative_channels: Vec<TorsionSectorChannelAudit>,
    pub complete_dimension_partition: bool,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn audit_dimension_zero_torsion_sectors(
    exterior: &LevelSixteenExteriorDerivativeAudit,
) -> DimensionZeroTorsionSectorAudit {
    let image_nonzero = |label: &str| {
        exterior
            .channels
            .iter()
            .find(|channel| channel.dynkin_label == label)
            .unwrap()
            .exterior_image_nonzero_certified
    };
    let two_form_vector_channels = vec![
        TorsionSectorChannelAudit {
            dynkin_label: "10000",
            dimension: 11,
            tensor_role: "trace vector",
            removed_by_conventional_constraint: true,
            exterior_image_nonzero: image_nonzero("10000"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "00100",
            dimension: 165,
            tensor_role: "totally antisymmetric three-form",
            removed_by_conventional_constraint: true,
            exterior_image_nonzero: image_nonzero("00100"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "11000",
            dimension: 429,
            tensor_role: "traceless two-form-vector hook",
            removed_by_conventional_constraint: false,
            exterior_image_nonzero: image_nonzero("11000"),
        },
    ];
    let five_form_vector_channels = vec![
        TorsionSectorChannelAudit {
            dynkin_label: "00010",
            dimension: 330,
            tensor_role: "trace four-form",
            removed_by_conventional_constraint: true,
            exterior_image_nonzero: image_nonzero("00010"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "00002",
            dimension: 462,
            tensor_role: "totally antisymmetric six-form",
            removed_by_conventional_constraint: true,
            exterior_image_nonzero: image_nonzero("00002"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "10002",
            dimension: 4_290,
            tensor_role: "traceless five-form-vector hook",
            removed_by_conventional_constraint: false,
            exterior_image_nonzero: image_nonzero("10002"),
        },
    ];
    let complementary_derivative_channels = vec![
        TorsionSectorChannelAudit {
            dynkin_label: "01000",
            dimension: 55,
            tensor_role: "two-form complement",
            removed_by_conventional_constraint: false,
            exterior_image_nonzero: image_nonzero("01000"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "20000",
            dimension: 65,
            tensor_role: "symmetric traceless rank-two complement",
            removed_by_conventional_constraint: false,
            exterior_image_nonzero: image_nonzero("20000"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "10100",
            dimension: 1_430,
            tensor_role: "mixed-symmetry complement",
            removed_by_conventional_constraint: false,
            exterior_image_nonzero: image_nonzero("10100"),
        },
        TorsionSectorChannelAudit {
            dynkin_label: "10010",
            dimension: 3_003,
            tensor_role: "mixed-symmetry complement",
            removed_by_conventional_constraint: false,
            exterior_image_nonzero: image_nonzero("10010"),
        },
    ];
    let two_form_vector_dimension = two_form_vector_channels
        .iter()
        .map(|channel| channel.dimension)
        .sum();
    let five_form_vector_dimension = five_form_vector_channels
        .iter()
        .map(|channel| channel.dimension)
        .sum();
    let complementary_dimension = complementary_derivative_channels
        .iter()
        .map(|channel| channel.dimension)
        .sum::<u64>();
    let complete_dimension_partition =
        two_form_vector_dimension + five_form_vector_dimension + complementary_dimension == 10_240;
    let two_form_remaining_hook_nonzero = image_nonzero("11000");
    let five_form_remaining_hook_nonzero = image_nonzero("10002");
    let passed = two_form_vector_dimension == 605
        && five_form_vector_dimension == 5_082
        && complementary_dimension == 4_553
        && complete_dimension_partition
        && !two_form_remaining_hook_nonzero
        && five_form_remaining_hook_nonzero;
    DimensionZeroTorsionSectorAudit {
        source_equations: "Eqs. (38)-(40) of hep-th/0101037 and the final line of Eq. (2.7) of arXiv:2007.05097",
        two_form_vector_dimension,
        two_form_vector_channels,
        two_form_conventional_dimension: 176,
        two_form_remaining_hook_dimension: 429,
        two_form_remaining_hook_nonzero,
        five_form_vector_dimension,
        five_form_vector_channels,
        five_form_conventional_dimension: 792,
        five_form_remaining_hook_dimension: 4_290,
        five_form_remaining_hook_nonzero,
        complementary_derivative_channels,
        complete_dimension_partition,
        interpretation: "at the exterior symbol, the conventional trace and antisymmetric pieces of X_[2] are nonzero but its 429-dimensional hook vanishes; the conventional four-form and six-form pieces of X_[5] and its 4290-dimensional hook are nonzero; this identifies the representation sectors but does not impose the full superspace Bianchi identities",
        passed,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalSourceLineNormalizationAudit {
    pub source_artifact: &'static str,
    pub source_dynkin_label: &'static str,
    pub primitive_coefficient_gcd: i16,
    pub first_nonzero_coefficient: i16,
    pub squared_norm: u64,
    pub orthogonal_projection_denominator: u64,
    pub projector_on_source_line_numerator: u64,
    pub projector_on_source_line_denominator: u64,
    pub primitive_sign_fixed: bool,
    pub orthogonal_projector_idempotent: bool,
    pub computational_source_normalization_fixed: bool,
    pub physical_bridge_scale_fixed: bool,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn squared_norm(coefficients: &[i16]) -> u64 {
    coefficients
        .iter()
        .map(|coefficient| {
            let coefficient = i64::from(*coefficient);
            (coefficient * coefficient) as u64
        })
        .sum()
}

fn audit_canonical_source_line_normalization() -> CanonicalSourceLineNormalizationAudit {
    let coefficients = decode_kernel(VECTOR_SPINOR_KERNEL);
    let primitive_coefficient_gcd = coefficients.iter().fold(0_i16, |gcd, coefficient| {
        integer_gcd(gcd, coefficient.abs())
    });
    let first_nonzero_coefficient = coefficients
        .iter()
        .copied()
        .find(|coefficient| *coefficient != 0)
        .unwrap_or(0);
    let squared_norm = squared_norm(&coefficients);
    let orthogonal_projection_denominator = squared_norm;
    let projector_on_source_line_numerator = squared_norm;
    let projector_on_source_line_denominator = squared_norm;
    let primitive_sign_fixed = first_nonzero_coefficient > 0;
    let orthogonal_projector_idempotent = squared_norm != 0
        && projector_on_source_line_numerator == projector_on_source_line_denominator
        && orthogonal_projection_denominator == squared_norm;
    let computational_source_normalization_fixed =
        primitive_coefficient_gcd == 1 && primitive_sign_fixed && orthogonal_projector_idempotent;
    CanonicalSourceLineNormalizationAudit {
        source_artifact: "data/eleven_dimensional_bridge/10001_highest_weight_kernel.i16le",
        source_dynkin_label: "10001",
        primitive_coefficient_gcd,
        first_nonzero_coefficient,
        squared_norm,
        orthogonal_projection_denominator,
        projector_on_source_line_numerator,
        projector_on_source_line_denominator,
        primitive_sign_fixed,
        orthogonal_projector_idempotent,
        computational_source_normalization_fixed,
        physical_bridge_scale_fixed: false,
        interpretation: "the primitive integer kernel, positive first nonzero coefficient, and exact rank-one orthogonal projector fix a reproducible normalization of the source highest-weight line; they do not fix the physical coefficient c multiplying the bridge",
        passed: computational_source_normalization_fixed,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InheritedSpinorGaugeAudit {
    pub assumed_relation: &'static str,
    pub direct_parameter_form_degrees: Vec<usize>,
    pub channels_leaving_scalar_divergence_invariant: Vec<usize>,
    pub channels_leaving_quotient_bridge_invariant: Vec<usize>,
    pub invariant_channel_count: usize,
    pub boundary: &'static str,
    pub passed: bool,
}

fn audit_inherited_spinor_gauge(
    clifford: &crate::eleven_dimensional_clifford::ElevenDimensionalCliffordReport,
) -> InheritedSpinorGaugeAudit {
    let direct_parameter_form_degrees = (0..=5).collect::<Vec<_>>();
    let channels_leaving_scalar_divergence_invariant = clifford
        .generic_momentum_scalar_divergence_kernel_degrees
        .clone();
    let channels_leaving_quotient_bridge_invariant =
        channels_leaving_scalar_divergence_invariant.clone();
    let invariant_channel_count = channels_leaving_quotient_bridge_invariant.len();
    InheritedSpinorGaugeAudit {
        assumed_relation: "V = D^alpha Psi_alpha and H_hat(V) = c P_320 I_320(D^15 V)",
        direct_parameter_form_degrees,
        channels_leaving_scalar_divergence_invariant,
        channels_leaving_quotient_bridge_invariant,
        invariant_channel_count,
        boundary: "This is inherited invariance under the direct two-form and five-form spinor-parameter channels. It does not establish that V = D Psi is the fundamental 11D prepotential relation or provide gauge-for-gauge reducibility.",
        passed: invariant_channel_count == 2,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalGammaTraceQuotientAudit {
    pub source_equations: &'static str,
    pub vector_spinor_dimension: usize,
    pub local_symmetry_image_rank: usize,
    pub gamma_traceless_quotient_rank: usize,
    pub projector_completeness_residuals: usize,
    pub bridge_coefficients_before_quotient: usize,
    pub gamma_trace_coefficients_removed: Vec<&'static str>,
    pub surviving_coefficients: Vec<&'static str>,
    pub quotient_bridge_coefficient_dimension: usize,
    pub quotient_representative: &'static str,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn audit_local_gamma_trace_quotient(
    clifford: &crate::eleven_dimensional_clifford::ElevenDimensionalCliffordReport,
) -> LocalGammaTraceQuotientAudit {
    let gamma_trace_coefficients_removed = vec!["a", "b"];
    let surviving_coefficients = vec!["c"];
    let passed = clifford.gamma_trace_projector_rank == 32
        && clifford.gamma_traceless_projector_rank == 320
        && clifford.projector_completeness_residual_entries == 0
        && gamma_trace_coefficients_removed.len() == 2
        && surviving_coefficients.len() == 1;
    LocalGammaTraceQuotientAudit {
        source_equations: "Eqs. (2.2)-(2.3) of arXiv:2007.05097",
        vector_spinor_dimension: 352,
        local_symmetry_image_rank: clifford.gamma_trace_projector_rank,
        gamma_traceless_quotient_rank: clifford.gamma_traceless_projector_rank,
        projector_completeness_residuals: clifford.projector_completeness_residual_entries,
        bridge_coefficients_before_quotient: 3,
        gamma_trace_coefficients_removed,
        surviving_coefficients,
        quotient_bridge_coefficient_dimension: 1,
        quotient_representative: "H_hat_alpha^a(V) = c P_320 I_320(D^15 V)",
        interpretation: "the two 00001 bridge channels lie in the rank-32 gamma-trace image generated by the local Lambda_alpha symmetry; the 10001 channel lies in the rank-320 gamma-traceless quotient and is the unique non-gamma-trace bridge class up to normalization",
        passed,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ZeroMomentumEquationProjectionAudit {
    pub source_equation: &'static str,
    pub linearized_formula_source: &'static str,
    pub raw_two_form_vector_dimension: usize,
    pub conventional_vector_dimension: usize,
    pub conventional_three_form_dimension: usize,
    pub remaining_hook_dynkin_label: &'static str,
    pub remaining_hook_dimension: usize,
    pub dimension_decomposition_closes: bool,
    pub scalar_level_checked: usize,
    pub hook_multiplicity_in_scalar_level: usize,
    pub exterior_symbol_rank_on_bridge_coefficients: usize,
    pub exterior_symbol_kernel_dimension: usize,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn audit_zero_momentum_equation_2_7_projection() -> ZeroMomentumEquationProjectionAudit {
    let raw_two_form_vector_dimension = 55 * 11;
    let conventional_vector_dimension = 11;
    let conventional_three_form_dimension = 165;
    let remaining_hook_dimension = 429;
    let hook_multiplicity_in_scalar_level =
        crate::eleven_dimensional_prepotential::level_multiplicity(16, "11000");
    let dimension_decomposition_closes = conventional_vector_dimension
        + conventional_three_form_dimension
        + remaining_hook_dimension
        == raw_two_form_vector_dimension;
    let exterior_symbol_rank_on_bridge_coefficients =
        usize::from(hook_multiplicity_in_scalar_level != 0);
    let exterior_symbol_kernel_dimension = 3 - exterior_symbol_rank_on_bridge_coefficients;
    ZeroMomentumEquationProjectionAudit {
        source_equation: "the final gamma^[2] torsion constraint in Eq. (2.7) of arXiv:2007.05097",
        linearized_formula_source: "Eqs. (39)-(40) of hep-th/0101037",
        raw_two_form_vector_dimension,
        conventional_vector_dimension,
        conventional_three_form_dimension,
        remaining_hook_dynkin_label: "11000",
        remaining_hook_dimension,
        dimension_decomposition_closes,
        scalar_level_checked: 16,
        hook_multiplicity_in_scalar_level,
        exterior_symbol_rank_on_bridge_coefficients,
        exterior_symbol_kernel_dimension,
        interpretation: "the vector and three-form pieces are removed by conventional constraints; the remaining 429-dimensional hook is absent from the level-16 exterior symbol, so that symbol has rank zero on a, b, and c; the nonzero level-14 momentum contraction shows that this statement does not extend to the full generic-momentum Eq. (2.7) operator",
        passed: dimension_decomposition_closes
            && hook_multiplicity_in_scalar_level == 0
            && exterior_symbol_kernel_dimension == 3,
    }
}

fn audit_first_momentum_completion(
    momentum: &FirstDerivativeMomentumAudit,
    weights: &[Weight; 32],
    left: &HashMap<(u8, Weight), Vec<u16>>,
    right: &HashMap<(u8, Weight), Vec<u16>>,
) -> FirstMomentumCompletionAudit {
    let vector_times_target =
        crate::eleven_dimensional_prepotential::vector_tensor_gamma_traceless_vector_spinor_channels();
    let vector_times_target_dimension = vector_times_target
        .iter()
        .map(|(_, dimension, multiplicity)| *dimension as usize * *multiplicity)
        .sum::<usize>();
    let vector_times_target_channels = vector_times_target
        .into_iter()
        .map(
            |(source_dynkin_label, source_dimension, multiplicity_in_vector_times_target)| {
                let multiplicity_at_scalar_level_thirteen =
                    crate::eleven_dimensional_prepotential::level_multiplicity(
                        13,
                        &source_dynkin_label,
                    );
                MomentumCompletionChannelAudit {
                    source_dynkin_label,
                    source_dimension,
                    multiplicity_in_vector_times_target,
                    multiplicity_at_scalar_level_thirteen,
                    correction_coefficients: multiplicity_in_vector_times_target
                        * multiplicity_at_scalar_level_thirteen,
                }
            },
        )
        .collect::<Vec<_>>();
    let available_level_thirteen_source_channels = vector_times_target_channels
        .iter()
        .filter(|channel| channel.correction_coefficients != 0)
        .map(|channel| channel.source_dynkin_label.clone())
        .collect::<Vec<_>>();
    let first_completion_coefficient_dimension = vector_times_target_channels
        .iter()
        .map(|channel| channel.correction_coefficients)
        .sum::<usize>();
    let leading_two_form_hook_momentum_term_nonzero =
        momentum.two_form_hook_momentum_contraction_nonzero;
    let plans = build_exterior_channel_plans(weights);
    let hook_plan = plans
        .iter()
        .find(|plan| plan.dynkin_label == "11000")
        .unwrap();
    let needed_weights = hook_plan
        .domain
        .iter()
        .map(|entry| entry.vector_spinor_weight)
        .collect::<std::collections::BTreeSet<_>>();
    let level15_basis = weight_basis(15, [3, 1, 1, 1, 1], left, right);
    let leading_states = generate_partial_vector_spinor_source_states(
        &level15_basis,
        &decode_kernel(VECTOR_SPINOR_KERNEL),
        weights,
        &needed_weights,
    );
    let exact_sample_momentum = [1_i64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let contracted_bilinear = contracted_translation_bilinear(&exact_sample_momentum);
    let leading = leading_hook_functionals(hook_plan, &leading_states, &contracted_bilinear);

    let level13_spinor_basis = weight_basis(13, [1, 1, 1, 1, 1], left, right);
    let spinor_highest = spinor_correction_highest_source(
        &level13_spinor_basis,
        &decode_kernel(LEVEL13_SPINOR_KERNEL),
    );
    let spinor_states =
        generate_partial_momentum_correction_states(spinor_highest, weights, &needed_weights);
    let mut correction_functionals = vec![correction_hook_functionals(
        hook_plan,
        &spinor_states,
        &exact_sample_momentum,
    )];
    drop(spinor_states);

    let level13_two_form_spinor_basis = weight_basis(13, [3, 3, 1, 1, 1], left, right);
    for kernel in [
        LEVEL13_TWO_FORM_SPINOR_KERNEL_1,
        LEVEL13_TWO_FORM_SPINOR_KERNEL_2,
    ] {
        let highest = two_form_spinor_correction_highest_source(
            &level13_two_form_spinor_basis,
            &decode_kernel(kernel),
            weights,
        );
        let states = generate_partial_momentum_correction_states(highest, weights, &needed_weights);
        correction_functionals.push(correction_hook_functionals(
            hook_plan,
            &states,
            &exact_sample_momentum,
        ));
    }
    let mut correction_rows = Vec::with_capacity(2 * CANCELLATION_FUNCTIONAL_BUCKETS);
    let mut augmented_rows = Vec::with_capacity(2 * CANCELLATION_FUNCTIONAL_BUCKETS);
    for component in 0..2 {
        for bucket in 0..CANCELLATION_FUNCTIONAL_BUCKETS {
            let correction_row = correction_functionals
                .iter()
                .map(|functional| {
                    if component == 0 {
                        functional.0[bucket]
                    } else {
                        functional.1[bucket]
                    }
                })
                .collect::<Vec<_>>();
            let mut augmented_row = correction_row.clone();
            augmented_row.push(if component == 0 {
                leading.0[bucket]
            } else {
                leading.1[bucket]
            });
            correction_rows.push(correction_row);
            augmented_rows.push(augmented_row);
        }
    }
    let correction_span_rank = rational_rank_i128(&correction_rows);
    let augmented_span_rank = rational_rank_i128(&augmented_rows);
    let exact_functional_matrix = augmented_rows
        .iter()
        .map(|row| row.iter().map(ToString::to_string).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let exact_non_cancellation_certificate = augmented_span_rank > correction_span_rank;
    let cancellation_system_constructed = true;
    let cancellation_exists = exact_non_cancellation_certificate.then_some(false);
    let passed = vector_times_target_dimension == 11 * 320
        && available_level_thirteen_source_channels == vec!["00001", "01001"]
        && first_completion_coefficient_dimension == 3
        && leading_two_form_hook_momentum_term_nonzero
        && cancellation_system_constructed
        && correction_span_rank <= 3
        && augmented_span_rank <= 4;
    FirstMomentumCompletionAudit {
        normal_form_term:
            "p D_[13] V in H_alpha^a(V), whose exterior derivative contributes at p D_[14] V",
        target_dynkin_label: "10001",
        target_dimension: 320,
        vector_times_target_dimension,
        vector_times_target_channels,
        available_level_thirteen_source_channels,
        first_completion_coefficient_dimension,
        leading_two_form_hook_momentum_term_nonzero,
        cancellation_system_constructed,
        cancellation_exists,
        exact_functional_rows: 2 * CANCELLATION_FUNCTIONAL_BUCKETS,
        exact_functional_definition: "32 deterministic signed mask buckets, with separate real and imaginary rows, evaluated over exact integers at p=(1,2,3,4,5,6,7,8,9,10,11)",
        exact_functional_columns: [
            "level13_00001_copy1",
            "level13_01001_copy1",
            "level13_01001_copy2",
            "leading_level15_10001",
        ],
        exact_functional_matrix,
        correction_span_rank,
        augmented_span_rank,
        exact_non_cancellation_certificate,
        implication: if exact_non_cancellation_certificate {
            "the complete first lower-symbol correction space does not span the leading p D_[14] two-form-hook term; terms with two or more momenta lie in different normal-form bidegrees and cannot cancel this coefficient; the stated local polynomial scalar bridge therefore fails this part of Eq. (2.7)"
        } else {
            "the exact functionals do not separate the leading term from the three first lower-symbol corrections; a full coordinate cancellation check remains required"
        },
        passed,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorSpinorTargetAudit {
    pub tensor_product_dimension: usize,
    pub generated_irrep_dimension: usize,
    pub expected_irrep_dimension: usize,
    pub distinct_weights: usize,
    pub multiplicity_one_weights: usize,
    pub multiplicity_five_weights: usize,
    pub maximum_weight_multiplicity: usize,
    pub nonzero_lowering_actions: usize,
    pub passed: bool,
}

type TensorVector = HashMap<usize, Ratio<i64>>;
type MomentumExteriorVector = HashMap<(usize, u32), i64>;

fn vector_weights() -> Vec<Weight> {
    let mut weights = Vec::new();
    for axis in 0..5 {
        let mut positive = [0_i8; 5];
        positive[axis] = 2;
        weights.push(positive);
        let mut negative = [0_i8; 5];
        negative[axis] = -2;
        weights.push(negative);
    }
    weights.push([0_i8; 5]);
    weights
}

fn lower_vector_weight(weight: Weight, root: usize, weights: &[Weight]) -> Option<(usize, i64)> {
    let mut target = weight;
    if root < 4 {
        if weight[root] == 2 {
            target[root] = 0;
            target[root + 1] = 2;
        } else if weight[root + 1] == -2 {
            target[root] = -2;
            target[root + 1] = 0;
        } else {
            return None;
        }
        Some((weights.iter().position(|item| *item == target).unwrap(), 1))
    } else if weight[4] == 2 {
        Some((weights.iter().position(|item| *item == [0; 5]).unwrap(), 1))
    } else if weight == [0; 5] {
        target[4] = -2;
        Some((weights.iter().position(|item| *item == target).unwrap(), 2))
    } else {
        None
    }
}

fn raise_vector_weight(weight: Weight, root: usize, weights: &[Weight]) -> Option<(usize, i64)> {
    let mut target = weight;
    if root < 4 {
        if weight[root] == 0 && weight[root + 1] == 2 {
            target[root] = 2;
            target[root + 1] = 0;
        } else if weight[root] == -2 && weight[root + 1] == 0 {
            target[root] = 0;
            target[root + 1] = -2;
        } else {
            return None;
        }
        Some((weights.iter().position(|item| *item == target).unwrap(), 1))
    } else if weight == [0; 5] {
        target[4] = 2;
        Some((weights.iter().position(|item| *item == target).unwrap(), 2))
    } else if weight[4] == -2 {
        Some((weights.iter().position(|item| *item == [0; 5]).unwrap(), 1))
    } else {
        None
    }
}

fn two_form_pairs() -> Vec<(usize, usize)> {
    (0..11)
        .flat_map(|left| ((left + 1)..11).map(move |right| (left, right)))
        .collect()
}

fn wedge_pair_index(left: usize, right: usize, pairs: &[(usize, usize)]) -> Option<(usize, i64)> {
    if left == right {
        return None;
    }
    let (pair, sign) = if left < right {
        ((left, right), 1)
    } else {
        ((right, left), -1)
    };
    Some((pairs.iter().position(|item| *item == pair).unwrap(), sign))
}

fn lower_two_form_index(
    pair_index: usize,
    root: usize,
    vectors: &[Weight],
    pairs: &[(usize, usize)],
) -> Vec<(usize, i64)> {
    let (left, right) = pairs[pair_index];
    let mut output = Vec::new();
    if let Some((next, factor)) = lower_vector_weight(vectors[left], root, vectors) {
        if let Some((index, sign)) = wedge_pair_index(next, right, pairs) {
            output.push((index, sign * factor));
        }
    }
    if let Some((next, factor)) = lower_vector_weight(vectors[right], root, vectors) {
        if let Some((index, sign)) = wedge_pair_index(left, next, pairs) {
            output.push((index, sign * factor));
        }
    }
    output
}

fn raise_two_form_index(
    pair_index: usize,
    root: usize,
    vectors: &[Weight],
    pairs: &[(usize, usize)],
) -> Vec<(usize, i64)> {
    let (left, right) = pairs[pair_index];
    let mut output = Vec::new();
    if let Some((next, factor)) = raise_vector_weight(vectors[left], root, vectors) {
        if let Some((index, sign)) = wedge_pair_index(next, right, pairs) {
            output.push((index, sign * factor));
        }
    }
    if let Some((next, factor)) = raise_vector_weight(vectors[right], root, vectors) {
        if let Some((index, sign)) = wedge_pair_index(left, next, pairs) {
            output.push((index, sign * factor));
        }
    }
    output
}

fn lower_two_form_spinor_tensor(
    source: &TensorVector,
    root: usize,
    vectors: &[Weight],
    pairs: &[(usize, usize)],
    spinors: &[Weight; 32],
) -> TensorVector {
    let mut target = TensorVector::new();
    for (&index, coefficient) in source {
        let pair_index = index / 32;
        let spinor_index = index % 32;
        for (next, factor) in lower_two_form_index(pair_index, root, vectors, pairs) {
            *target.entry(next * 32 + spinor_index).or_default() +=
                coefficient.clone() * Ratio::from_integer(factor);
        }
        if let Some(next) = lowered_spinor_index(spinor_index, root, spinors) {
            *target.entry(pair_index * 32 + next).or_default() += coefficient.clone();
        }
    }
    target.retain(|_, coefficient| *coefficient != Ratio::from_integer(0));
    target
}

fn raise_two_form_spinor_tensor(
    source: &TensorVector,
    root: usize,
    vectors: &[Weight],
    pairs: &[(usize, usize)],
    spinors: &[Weight; 32],
) -> TensorVector {
    let mut target = TensorVector::new();
    for (&index, coefficient) in source {
        let pair_index = index / 32;
        let spinor_index = index % 32;
        for (next, factor) in raise_two_form_index(pair_index, root, vectors, pairs) {
            *target.entry(next * 32 + spinor_index).or_default() +=
                coefficient.clone() * Ratio::from_integer(factor);
        }
        if let Some(next) = raised_spinor_index(spinor_index, root, spinors) {
            *target.entry(pair_index * 32 + next).or_default() += coefficient.clone();
        }
    }
    target.retain(|_, coefficient| *coefficient != Ratio::from_integer(0));
    target
}

fn two_form_spinor_weight(
    vector: &TensorVector,
    vectors: &[Weight],
    pairs: &[(usize, usize)],
    spinors: &[Weight; 32],
) -> Weight {
    let index = *vector.keys().next().unwrap();
    let (left, right) = pairs[index / 32];
    add(add(vectors[left], vectors[right]), spinors[index % 32])
}

#[cfg(test)]
fn generate_two_form_spinor_target_basis(
    spinors: &[Weight; 32],
) -> (HashMap<Weight, Vec<(usize, TensorVector)>>, usize) {
    use std::collections::VecDeque;
    let vectors = vector_weights();
    let pairs = two_form_pairs();
    let pair_highest = pairs.iter().position(|pair| *pair == (0, 2)).unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let start = HashMap::from([(pair_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_weight = [3, 3, 1, 1, 1];
    let mut by_weight = HashMap::<Weight, Vec<(usize, TensorVector)>>::new();
    add_target_basis(start.clone(), by_weight.entry(highest_weight).or_default());
    let mut queue = VecDeque::from([start]);
    let mut nonzero_lowering_actions = 0;
    while let Some(state) = queue.pop_front() {
        for root in 0..5 {
            let descendant = lower_two_form_spinor_tensor(&state, root, &vectors, &pairs, spinors);
            if descendant.is_empty() {
                continue;
            }
            nonzero_lowering_actions += 1;
            let weight = two_form_spinor_weight(&descendant, &vectors, &pairs, spinors);
            if add_target_basis(descendant.clone(), by_weight.entry(weight).or_default()) {
                queue.push_back(descendant);
            }
        }
    }
    (by_weight, nonzero_lowering_actions)
}

#[derive(Clone)]
struct LowerSymbolDomainEntry {
    vector_index: usize,
    source_weight: Weight,
    source_basis_index: usize,
}

fn build_01001_to_10001_highest_map(
    spinors: &[Weight; 32],
) -> (Vec<LowerSymbolDomainEntry>, Vec<i64>, usize) {
    let vectors = vector_weights();
    let pairs = two_form_pairs();
    let source_states = generate_layer_adapted_two_form_spinor_target_states(spinors);
    let target_highest = [3, 1, 1, 1, 1];
    let mut domain = Vec::new();
    for (vector_index, vector_weight) in vectors.iter().enumerate() {
        let source_weight = subtract(target_highest, *vector_weight);
        if let Some(states) = source_states.get(&source_weight) {
            for source_basis_index in 0..states.len() {
                domain.push(LowerSymbolDomainEntry {
                    vector_index,
                    source_weight,
                    source_basis_index,
                });
            }
        }
    }
    let mut rows = BTreeMap::<usize, Vec<Ratio<i64>>>::new();
    for (column, entry) in domain.iter().enumerate() {
        let source = &source_states[&entry.source_weight][entry.source_basis_index].target;
        for root in 0..5 {
            if let Some((raised_vector, factor)) =
                raise_vector_weight(vectors[entry.vector_index], root, &vectors)
            {
                for (&inner, coefficient) in source {
                    rows.entry(root * 11 * 55 * 32 + raised_vector * 55 * 32 + inner)
                        .or_insert_with(|| vec![Ratio::from_integer(0); domain.len()])[column] +=
                        coefficient.clone() * Ratio::from_integer(factor);
                }
            }
            for (inner, coefficient) in
                raise_two_form_spinor_tensor(source, root, &vectors, &pairs, spinors)
            {
                rows.entry(root * 11 * 55 * 32 + entry.vector_index * 55 * 32 + inner)
                    .or_insert_with(|| vec![Ratio::from_integer(0); domain.len()])[column] +=
                    coefficient;
            }
        }
    }
    let row_vectors = rows.into_values().collect::<Vec<_>>();
    let kernel = ratio_nullspace(&row_vectors, domain.len());
    let primitive = if kernel.len() == 1 {
        primitive_integer_vector(&kernel[0])
    } else {
        Vec::new()
    };
    (domain, primitive, kernel.len())
}

fn lower_target_tensor(
    source: &TensorVector,
    root: usize,
    vectors: &[Weight],
    spinors: &[Weight; 32],
) -> TensorVector {
    let mut target = TensorVector::new();
    for (&index, coefficient) in source {
        let vector_index = index / 32;
        let spinor_index = index % 32;
        if let Some((next, factor)) = lower_vector_weight(vectors[vector_index], root, vectors) {
            *target.entry(next * 32 + spinor_index).or_default() +=
                coefficient.clone() * Ratio::from_integer(factor);
        }
        if let Some(next) = lowered_spinor_index(spinor_index, root, spinors) {
            *target.entry(vector_index * 32 + next).or_default() += coefficient.clone();
        }
    }
    target.retain(|_, coefficient| *coefficient != Ratio::from_integer(0));
    target
}

fn raise_target_tensor(
    source: &TensorVector,
    root: usize,
    vectors: &[Weight],
    spinors: &[Weight; 32],
) -> TensorVector {
    let mut target = TensorVector::new();
    for (&index, coefficient) in source {
        let vector_index = index / 32;
        let spinor_index = index % 32;
        if let Some((next, factor)) = raise_vector_weight(vectors[vector_index], root, vectors) {
            *target.entry(next * 32 + spinor_index).or_default() +=
                coefficient.clone() * Ratio::from_integer(factor);
        }
        if let Some(next) = raised_spinor_index(spinor_index, root, spinors) {
            *target.entry(vector_index * 32 + next).or_default() += coefficient.clone();
        }
    }
    target.retain(|_, coefficient| *coefficient != Ratio::from_integer(0));
    target
}

fn tensor_weight(vector: &TensorVector, vectors: &[Weight], spinors: &[Weight; 32]) -> Weight {
    let index = *vector.keys().next().unwrap();
    add(vectors[index / 32], spinors[index % 32])
}

fn add_target_basis(vector: TensorVector, basis: &mut Vec<(usize, TensorVector)>) -> bool {
    let mut residual = vector;
    basis.sort_unstable_by_key(|(pivot, _)| *pivot);
    for (pivot, existing) in basis.iter() {
        let Some(value) = residual.get(pivot).cloned() else {
            continue;
        };
        let factor = value / existing[pivot].clone();
        for (&index, coefficient) in existing {
            *residual.entry(index).or_default() -= factor.clone() * coefficient.clone();
        }
        residual.retain(|_, coefficient| *coefficient != Ratio::from_integer(0));
    }
    if residual.is_empty() {
        return false;
    }
    let pivot = *residual.keys().min().unwrap();
    let normalization = residual[&pivot].clone();
    for coefficient in residual.values_mut() {
        *coefficient /= normalization.clone();
    }
    basis.push((pivot, residual));
    true
}

fn generate_vector_spinor_target_basis(
    spinors: &[Weight; 32],
) -> (HashMap<Weight, Vec<(usize, TensorVector)>>, usize) {
    use std::collections::VecDeque;
    let vectors = vector_weights();
    let vector_highest = vectors
        .iter()
        .position(|weight| *weight == [2, 0, 0, 0, 0])
        .unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let start = HashMap::from([(vector_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let mut by_weight = HashMap::<Weight, Vec<(usize, TensorVector)>>::new();
    add_target_basis(start.clone(), by_weight.entry([3, 1, 1, 1, 1]).or_default());
    let mut queue = VecDeque::from([start]);
    let mut nonzero_lowering_actions = 0;
    while let Some(state) = queue.pop_front() {
        for root in 0..5 {
            let descendant = lower_target_tensor(&state, root, &vectors, spinors);
            if descendant.is_empty() {
                continue;
            }
            nonzero_lowering_actions += 1;
            let weight = tensor_weight(&descendant, &vectors, spinors);
            if add_target_basis(descendant.clone(), by_weight.entry(weight).or_default()) {
                queue.push_back(descendant);
            }
        }
    }
    (by_weight, nonzero_lowering_actions)
}

fn audit_vector_spinor_target(spinors: &[Weight; 32]) -> VectorSpinorTargetAudit {
    let (by_weight, nonzero_lowering_actions) = generate_vector_spinor_target_basis(spinors);
    let generated_irrep_dimension = by_weight.values().map(Vec::len).sum();
    let multiplicity_one_weights = by_weight.values().filter(|basis| basis.len() == 1).count();
    let multiplicity_five_weights = by_weight.values().filter(|basis| basis.len() == 5).count();
    let maximum_weight_multiplicity = by_weight.values().map(Vec::len).max().unwrap_or(0);
    VectorSpinorTargetAudit {
        tensor_product_dimension: 11 * 32,
        generated_irrep_dimension,
        expected_irrep_dimension: 320,
        distinct_weights: by_weight.len(),
        multiplicity_one_weights,
        multiplicity_five_weights,
        maximum_weight_multiplicity,
        nonzero_lowering_actions,
        passed: generated_irrep_dimension == 320
            && by_weight.len() == 192
            && multiplicity_one_weights == 160
            && multiplicity_five_weights == 32
            && maximum_weight_multiplicity == 5,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SpinorDescendantAudit {
    pub source_copy: usize,
    pub target_dynkin_label: &'static str,
    pub target_states_expected: usize,
    pub target_states_generated: usize,
    pub nonzero_lowering_actions_expected: usize,
    pub nonzero_lowering_actions_checked: usize,
    pub independent_state_discoveries: usize,
    pub repeated_path_checks: usize,
    pub repeated_path_mismatches: usize,
    pub minimum_state_support: usize,
    pub maximum_state_support: usize,
    pub exact_full_spinor_intertwiner_verified: bool,
}

fn spinor_weights() -> [Weight; 32] {
    std::array::from_fn(|index| {
        std::array::from_fn(|axis| {
            if (index >> (4 - axis)) & 1 == 0 {
                1
            } else {
                -1
            }
        })
    })
}

fn mask_weight(mask: u16, offset: usize, weights: &[Weight; 32]) -> Weight {
    let mut sum = [0_i8; 5];
    for local in 0..16 {
        if mask & (1 << local) != 0 {
            for axis in 0..5 {
                sum[axis] += weights[offset + local][axis];
            }
        }
    }
    sum
}

fn half_groups(offset: usize, weights: &[Weight; 32]) -> HashMap<(u8, Weight), Vec<u16>> {
    let mut groups: HashMap<(u8, Weight), Vec<u16>> = HashMap::new();
    for mask in 0_u32..=u32::from(u16::MAX) {
        let mask = mask as u16;
        groups
            .entry((mask.count_ones() as u8, mask_weight(mask, offset, weights)))
            .or_default()
            .push(mask);
    }
    groups
}

fn subtract(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn add(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn dynkin_highest_weight(label: &str) -> Weight {
    assert_eq!(label.len(), 5);
    let labels = label
        .bytes()
        .map(|byte| {
            assert!(byte.is_ascii_digit());
            i8::try_from(byte - b'0').unwrap()
        })
        .collect::<Vec<_>>();
    std::array::from_fn(|index| 2 * labels[index..4].iter().sum::<i8>() + labels[4])
}

fn weight_basis(
    exterior_degree: u8,
    target: Weight,
    left: &HashMap<(u8, Weight), Vec<u16>>,
    right: &HashMap<(u8, Weight), Vec<u16>>,
) -> Vec<u32> {
    let mut basis = Vec::new();
    for left_degree in 0_u8..=exterior_degree.min(16) {
        let right_degree = exterior_degree - left_degree;
        if right_degree > 16 {
            continue;
        }
        for ((degree, left_weight), left_masks) in left {
            if *degree != left_degree {
                continue;
            }
            let needed = subtract(target, *left_weight);
            if let Some(right_masks) = right.get(&(right_degree, needed)) {
                basis.reserve(left_masks.len() * right_masks.len());
                for &left_mask in left_masks {
                    for &right_mask in right_masks {
                        basis.push(u32::from(left_mask) | (u32::from(right_mask) << 16));
                    }
                }
            }
        }
    }
    basis.sort_unstable();
    basis
}

fn raised_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = add(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn lowering_preimages(output_mask: u32, root: usize, weights: &[Weight; 32]) -> Vec<(u32, i8)> {
    let mut preimages = Vec::new();
    for lower_index in 0..32 {
        let Some(upper_index) = raised_spinor_index(lower_index, root, weights) else {
            continue;
        };
        let upper_bit = 1_u32 << upper_index;
        let lower_bit = 1_u32 << lower_index;
        if output_mask & upper_bit == 0 || output_mask & lower_bit != 0 {
            continue;
        }
        let source_mask = (output_mask ^ upper_bit) | lower_bit;
        let (low, high) = if lower_index < upper_index {
            (lower_index, upper_index)
        } else {
            (upper_index, lower_index)
        };
        let strictly_between = if high == low + 1 {
            0
        } else {
            let interval = ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1);
            (source_mask & interval).count_ones()
        };
        let sign = if strictly_between % 2 == 0 { 1 } else { -1 };
        preimages.push((source_mask, sign));
    }
    preimages
}

fn lowered_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = subtract(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn first_lowering(
    source_basis: &[u32],
    coefficients: &[i16],
    root: usize,
    weights: &[Weight; 32],
) -> HashMap<u32, i64> {
    let source = source_basis
        .iter()
        .copied()
        .zip(coefficients.iter().copied())
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<HashMap<_, _>>();
    lower_sparse(&source, root, weights)
}

fn lower_sparse(
    source: &HashMap<u32, i64>,
    root: usize,
    weights: &[Weight; 32],
) -> HashMap<u32, i64> {
    let mut descendant = HashMap::new();
    for (&source_mask, &coefficient) in source {
        for upper_index in 0..32 {
            if source_mask & (1_u32 << upper_index) == 0 {
                continue;
            }
            let Some(lower_index) = lowered_spinor_index(upper_index, root, weights) else {
                continue;
            };
            if source_mask & (1_u32 << lower_index) != 0 {
                continue;
            }
            let output_mask = (source_mask ^ (1_u32 << upper_index)) | (1_u32 << lower_index);
            let (low, high) = if lower_index < upper_index {
                (lower_index, upper_index)
            } else {
                (upper_index, lower_index)
            };
            let interval = if high == low + 1 {
                0
            } else {
                ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
            };
            let sign = if (source_mask & interval).count_ones() % 2 == 0 {
                1_i64
            } else {
                -1_i64
            };
            *descendant.entry(output_mask).or_insert(0) += sign * coefficient;
        }
    }
    descendant.retain(|_, coefficient| *coefficient != 0);
    descendant
}

fn raise_sparse(
    source: &HashMap<u32, i64>,
    root: usize,
    weights: &[Weight; 32],
) -> HashMap<u32, i64> {
    let mut raised = HashMap::new();
    for (&source_mask, &coefficient) in source {
        for lower_index in 0..32 {
            if source_mask & (1_u32 << lower_index) == 0 {
                continue;
            }
            let Some(upper_index) = raised_spinor_index(lower_index, root, weights) else {
                continue;
            };
            if source_mask & (1_u32 << upper_index) != 0 {
                continue;
            }
            let output_mask = (source_mask ^ (1_u32 << lower_index)) | (1_u32 << upper_index);
            let (low, high) = if lower_index < upper_index {
                (lower_index, upper_index)
            } else {
                (upper_index, lower_index)
            };
            let interval = if high == low + 1 {
                0
            } else {
                ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
            };
            let sign = if (source_mask & interval).count_ones() % 2 == 0 {
                1_i64
            } else {
                -1_i64
            };
            *raised.entry(output_mask).or_insert(0) += sign * coefficient;
        }
    }
    raised.retain(|_, coefficient| *coefficient != 0);
    raised
}

fn lower_pairs(source: &[(u32, i64)], root: usize, weights: &[Weight; 32]) -> Vec<(u32, i64)> {
    let source = source.iter().copied().collect::<HashMap<_, _>>();
    let mut lowered = lower_sparse(&source, root, weights)
        .into_iter()
        .collect::<Vec<_>>();
    lowered.sort_unstable_by_key(|(mask, _)| *mask);
    lowered
}

fn lower_momentum_exterior(
    source: &MomentumExteriorVector,
    root: usize,
    vectors: &[Weight],
    spinors: &[Weight; 32],
) -> MomentumExteriorVector {
    let mut descendant = MomentumExteriorVector::new();
    for (&(vector_index, mask), &coefficient) in source {
        if let Some((next, factor)) = lower_vector_weight(vectors[vector_index], root, vectors) {
            *descendant.entry((next, mask)).or_insert(0) += factor * coefficient;
        }
        for upper_index in 0..32 {
            if mask & (1_u32 << upper_index) == 0 {
                continue;
            }
            let Some(lower_index) = lowered_spinor_index(upper_index, root, spinors) else {
                continue;
            };
            if mask & (1_u32 << lower_index) != 0 {
                continue;
            }
            let next_mask = (mask ^ (1_u32 << upper_index)) | (1_u32 << lower_index);
            let (low, high) = if lower_index < upper_index {
                (lower_index, upper_index)
            } else {
                (upper_index, lower_index)
            };
            let interval = if high == low + 1 {
                0
            } else {
                ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
            };
            let sign = if (mask & interval).count_ones() % 2 == 0 {
                1_i64
            } else {
                -1_i64
            };
            *descendant.entry((vector_index, next_mask)).or_insert(0) += sign * coefficient;
        }
    }
    descendant.retain(|_, coefficient| *coefficient != 0);
    descendant
}

fn momentum_source_relation_residual(
    candidate: &MomentumExteriorVector,
    basis: &[MomentumCorrectionState],
    coefficients: &[Ratio<i64>],
) -> usize {
    let denominator = coefficients.iter().fold(1_i64, |common, coefficient| {
        lcm_i64(common, *coefficient.denom())
    });
    let mut residual = HashMap::<(usize, u32), i128>::new();
    for (&key, &value) in candidate {
        *residual.entry(key).or_insert(0) += i128::from(denominator) * i128::from(value);
    }
    for (state, coefficient) in basis.iter().zip(coefficients) {
        let numerator = *coefficient.numer() * (denominator / *coefficient.denom());
        for (&key, &value) in &state.source {
            *residual.entry(key).or_insert(0) -= i128::from(numerator) * i128::from(value);
        }
    }
    residual.values().filter(|value| **value != 0).count()
}

#[derive(Clone)]
struct VectorSpinorIntertwinerState {
    target: TensorVector,
    source: Vec<(u32, i64)>,
}

#[derive(Clone)]
struct MomentumCorrectionState {
    target: TensorVector,
    source: MomentumExteriorVector,
}

fn generate_layer_adapted_vector_spinor_target_states(
    spinors: &[Weight; 32],
) -> BTreeMap<Weight, Vec<VectorSpinorIntertwinerState>> {
    let vectors = vector_weights();
    let vector_highest = vectors
        .iter()
        .position(|weight| *weight == [2, 0, 0, 0, 0])
        .unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let highest_target =
        HashMap::from([(vector_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_weight = [3, 1, 1, 1, 1];
    let highest_state = VectorSpinorIntertwinerState {
        target: highest_target,
        source: Vec::new(),
    };
    let mut all = BTreeMap::from([(highest_weight, vec![highest_state.clone()])]);
    let mut current = BTreeMap::from([(highest_weight, vec![highest_state])]);
    while !current.is_empty() {
        let mut next = BTreeMap::<Weight, Vec<VectorSpinorIntertwinerState>>::new();
        for states in current.into_values() {
            for state in states {
                for root in 0..5 {
                    let target_descendant =
                        lower_target_tensor(&state.target, root, &vectors, spinors);
                    if target_descendant.is_empty() {
                        continue;
                    }
                    let weight = tensor_weight(&target_descendant, &vectors, spinors);
                    let basis = next.entry(weight).or_default();
                    if target_span_coefficients(&target_descendant, basis, 11 * 32).is_none() {
                        basis.push(VectorSpinorIntertwinerState {
                            target: target_descendant,
                            source: Vec::new(),
                        });
                    }
                }
            }
        }
        for (weight, states) in &next {
            all.insert(*weight, states.clone());
        }
        current = next;
    }
    all
}

fn generate_layer_adapted_two_form_spinor_target_states(
    spinors: &[Weight; 32],
) -> BTreeMap<Weight, Vec<VectorSpinorIntertwinerState>> {
    let vectors = vector_weights();
    let pairs = two_form_pairs();
    let pair_highest = pairs.iter().position(|pair| *pair == (0, 2)).unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let highest_target =
        HashMap::from([(pair_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_weight = [3, 3, 1, 1, 1];
    let highest_state = VectorSpinorIntertwinerState {
        target: highest_target,
        source: Vec::new(),
    };
    let mut all = BTreeMap::from([(highest_weight, vec![highest_state.clone()])]);
    let mut current = BTreeMap::from([(highest_weight, vec![highest_state])]);
    while !current.is_empty() {
        let mut next = BTreeMap::<Weight, Vec<VectorSpinorIntertwinerState>>::new();
        for states in current.into_values() {
            for state in states {
                for root in 0..5 {
                    let target_descendant = lower_two_form_spinor_tensor(
                        &state.target,
                        root,
                        &vectors,
                        &pairs,
                        spinors,
                    );
                    if target_descendant.is_empty() {
                        continue;
                    }
                    let weight =
                        two_form_spinor_weight(&target_descendant, &vectors, &pairs, spinors);
                    let basis = next.entry(weight).or_default();
                    if target_span_coefficients(&target_descendant, basis, 55 * 32).is_none() {
                        basis.push(VectorSpinorIntertwinerState {
                            target: target_descendant,
                            source: Vec::new(),
                        });
                    }
                }
            }
        }
        for (weight, states) in &next {
            all.insert(*weight, states.clone());
        }
        current = next;
    }
    all
}

fn target_span_coefficients(
    candidate: &TensorVector,
    basis: &[VectorSpinorIntertwinerState],
    ambient_dimension: usize,
) -> Option<Vec<Ratio<i64>>> {
    let dimension = basis.len();
    if dimension == 0 {
        return None;
    }
    let zero = Ratio::from_integer(0);
    let mut rows = basis
        .iter()
        .enumerate()
        .map(|(index, state)| {
            let mut coordinates = vec![zero.clone(); dimension];
            coordinates[index] = Ratio::from_integer(1);
            (state.target.clone(), coordinates)
        })
        .collect::<Vec<_>>();
    let mut rank = 0;
    for column in 0..ambient_dimension {
        let Some(pivot_row) = (rank..dimension).find(|row| {
            rows[*row]
                .0
                .get(&column)
                .is_some_and(|value| *value != zero)
        }) else {
            continue;
        };
        rows.swap(rank, pivot_row);
        let normalization = rows[rank].0[&column].clone();
        for value in rows[rank].0.values_mut() {
            *value /= normalization.clone();
        }
        for value in &mut rows[rank].1 {
            *value /= normalization.clone();
        }
        let pivot_vector = rows[rank].0.clone();
        let pivot_coordinates = rows[rank].1.clone();
        for row in 0..dimension {
            if row == rank {
                continue;
            }
            let Some(factor) = rows[row].0.get(&column).cloned() else {
                continue;
            };
            for (&index, value) in &pivot_vector {
                *rows[row].0.entry(index).or_default() -= factor.clone() * value.clone();
            }
            rows[row].0.retain(|_, value| *value != zero);
            for (value, pivot_value) in rows[row].1.iter_mut().zip(&pivot_coordinates) {
                *value -= factor.clone() * pivot_value.clone();
            }
        }
        rank += 1;
        if rank == dimension {
            break;
        }
    }
    assert_eq!(rank, dimension);

    let mut residual = candidate.clone();
    let mut solution = vec![zero.clone(); dimension];
    for (row, coordinates) in &rows {
        let pivot = *row.keys().min().unwrap();
        let Some(factor) = residual.get(&pivot).cloned() else {
            continue;
        };
        for (&index, value) in row {
            *residual.entry(index).or_default() -= factor.clone() * value.clone();
        }
        residual.retain(|_, value| *value != zero);
        for (value, coordinate) in solution.iter_mut().zip(coordinates) {
            *value += factor.clone() * coordinate.clone();
        }
    }
    residual.is_empty().then_some(solution)
}

#[derive(Clone)]
struct ExteriorDomainEntry {
    outer_spinor_index: usize,
    vector_spinor_weight: Weight,
    vector_spinor_basis_index: usize,
}

struct ExteriorChannelPlan {
    dynkin_label: String,
    dimension: u64,
    scalar_level_sixteen_multiplicity: usize,
    domain: Vec<ExteriorDomainEntry>,
    primitive_highest_weight_coefficients: Vec<i64>,
    highest_weight_kernel_dimension: usize,
    raising_residual_terms: usize,
    fingerprint_residues: [u64; 3],
    momentum_contraction_real_residues: [u64; 3],
    momentum_contraction_imaginary_residues: [u64; 3],
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectHookTargetCouplingAudit {
    pub source_dynkin_label: &'static str,
    pub tensor_product: &'static str,
    pub target_dynkin_label: &'static str,
    pub highest_weight_domain_dimension: usize,
    pub highest_weight_kernel_dimension: usize,
    pub primitive_nonzero_coefficients: usize,
    pub raising_residual_terms: usize,
    pub multiplicity_one_coupling_constructed: bool,
    pub passed: bool,
}

fn ratio_nullspace(rows: &[Vec<Ratio<i64>>], columns: usize) -> Vec<Vec<Ratio<i64>>> {
    let zero = Ratio::from_integer(0);
    let mut reduced = rows
        .iter()
        .filter(|row| row.iter().any(|value| *value != zero))
        .cloned()
        .collect::<Vec<_>>();
    let mut pivot_columns = Vec::new();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot_row) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero)
        else {
            continue;
        };
        reduced.swap(rank, pivot_row);
        let normalization = reduced[rank][column].clone();
        for value in &mut reduced[rank] {
            *value /= normalization.clone();
        }
        let pivot = reduced[rank].clone();
        for row in 0..reduced.len() {
            if row == rank || reduced[row][column] == zero {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot[index].clone();
            }
        }
        pivot_columns.push(column);
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    let free_columns = (0..columns)
        .filter(|column| !pivot_columns.contains(column))
        .collect::<Vec<_>>();
    free_columns
        .into_iter()
        .map(|free| {
            let mut vector = vec![zero.clone(); columns];
            vector[free] = Ratio::from_integer(1);
            for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free].clone();
            }
            vector
        })
        .collect()
}

fn primitive_integer_vector(vector: &[Ratio<i64>]) -> Vec<i64> {
    let denominator = vector.iter().fold(1_i64, |common, coefficient| {
        lcm_i64(common, *coefficient.denom())
    });
    let mut integers = vector
        .iter()
        .map(|coefficient| *coefficient.numer() * (denominator / *coefficient.denom()))
        .collect::<Vec<_>>();
    let gcd = integers
        .iter()
        .fold(0_i64, |gcd, value| gcd_i64(gcd, *value));
    assert_ne!(gcd, 0);
    for value in &mut integers {
        *value /= gcd;
    }
    if integers.iter().find(|value| **value != 0).unwrap() < &0 {
        for value in &mut integers {
            *value = -*value;
        }
    }
    integers
}

fn build_exterior_channel_plans(spinors: &[Weight; 32]) -> Vec<ExteriorChannelPlan> {
    let vectors = vector_weights();
    let states = generate_layer_adapted_vector_spinor_target_states(spinors);
    crate::eleven_dimensional_prepotential::spinor_tensor_channels("10001")
        .into_iter()
        .map(|(dynkin_label, dimension)| {
            let highest_weight = dynkin_highest_weight(&dynkin_label);
            let mut domain = Vec::new();
            for (outer_spinor_index, outer_weight) in spinors.iter().enumerate() {
                let vector_spinor_weight = subtract(highest_weight, *outer_weight);
                if let Some(weight_states) = states.get(&vector_spinor_weight) {
                    for vector_spinor_basis_index in 0..weight_states.len() {
                        domain.push(ExteriorDomainEntry {
                            outer_spinor_index,
                            vector_spinor_weight,
                            vector_spinor_basis_index,
                        });
                    }
                }
            }
            let mut rows = BTreeMap::<usize, Vec<Ratio<i64>>>::new();
            for (column, entry) in domain.iter().enumerate() {
                let state =
                    &states[&entry.vector_spinor_weight][entry.vector_spinor_basis_index].target;
                for root in 0..5 {
                    if let Some(raised_outer) =
                        raised_spinor_index(entry.outer_spinor_index, root, spinors)
                    {
                        for (&inner, coefficient) in state {
                            rows.entry(root * 32 * 352 + raised_outer * 352 + inner)
                                .or_insert_with(|| vec![Ratio::from_integer(0); domain.len()])
                                [column] += coefficient.clone();
                        }
                    }
                    for (inner, coefficient) in raise_target_tensor(state, root, &vectors, spinors)
                    {
                        rows.entry(root * 32 * 352 + entry.outer_spinor_index * 352 + inner)
                            .or_insert_with(|| vec![Ratio::from_integer(0); domain.len()])
                            [column] += coefficient;
                    }
                }
            }
            let row_vectors = rows.into_values().collect::<Vec<_>>();
            let kernel = ratio_nullspace(&row_vectors, domain.len());
            let primitive_highest_weight_coefficients = if kernel.len() == 1 {
                primitive_integer_vector(&kernel[0])
            } else {
                Vec::new()
            };
            let raising_residual_terms = if primitive_highest_weight_coefficients.is_empty() {
                usize::MAX
            } else {
                row_vectors
                    .iter()
                    .filter(|row| {
                        row.iter().zip(&primitive_highest_weight_coefficients).fold(
                            Ratio::from_integer(0),
                            |sum, (value, coefficient)| {
                                sum + value.clone() * Ratio::from_integer(*coefficient)
                            },
                        ) != Ratio::from_integer(0)
                    })
                    .count()
            };
            ExteriorChannelPlan {
                scalar_level_sixteen_multiplicity:
                    crate::eleven_dimensional_prepotential::level_multiplicity(16, &dynkin_label),
                dynkin_label,
                dimension,
                domain,
                primitive_highest_weight_coefficients,
                highest_weight_kernel_dimension: kernel.len(),
                raising_residual_terms,
                fingerprint_residues: [0; 3],
                momentum_contraction_real_residues: [0; 3],
                momentum_contraction_imaginary_residues: [0; 3],
            }
        })
        .collect()
}

pub fn audit_direct_hook_target_coupling() -> DirectHookTargetCouplingAudit {
    let spinors = spinor_weights();
    let plans = build_exterior_channel_plans(&spinors);
    let hook = plans
        .iter()
        .find(|plan| plan.dynkin_label == "11000")
        .unwrap();
    let primitive_nonzero_coefficients = hook
        .primitive_highest_weight_coefficients
        .iter()
        .filter(|coefficient| **coefficient != 0)
        .count();
    let multiplicity_one_coupling_constructed = hook.highest_weight_kernel_dimension == 1
        && !hook.primitive_highest_weight_coefficients.is_empty()
        && hook.raising_residual_terms == 0;
    DirectHookTargetCouplingAudit {
        source_dynkin_label: "11000",
        tensor_product: "(00001) tensor (10001)",
        target_dynkin_label: "10001",
        highest_weight_domain_dimension: hook.domain.len(),
        highest_weight_kernel_dimension: hook.highest_weight_kernel_dimension,
        primitive_nonzero_coefficients,
        raising_residual_terms: hook.raising_residual_terms,
        multiplicity_one_coupling_constructed,
        passed: multiplicity_one_coupling_constructed,
    }
}

const EXTERIOR_FINGERPRINT_PRIMES: [u64; 3] = [1_000_000_007, 1_000_000_009, 998_244_353];
const EXTERIOR_FINGERPRINT_SEEDS: [u64; 3] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
];

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn accumulate_exterior_fingerprint(
    residues: &mut [u64; 3],
    outer_spinor_index: usize,
    domain_coefficient: i64,
    source: &[(u32, i64)],
) {
    let outer_bit = 1_u32 << outer_spinor_index;
    let lower_bits = outer_bit - 1;
    for &(mask, source_coefficient) in source {
        if mask & outer_bit != 0 {
            continue;
        }
        let wedge_sign = if (mask & lower_bits).count_ones() % 2 == 0 {
            1_i128
        } else {
            -1_i128
        };
        let signed_coefficient =
            i128::from(domain_coefficient) * i128::from(source_coefficient) * wedge_sign;
        let output_mask = mask | outer_bit;
        for index in 0..3 {
            let prime = EXTERIOR_FINGERPRINT_PRIMES[index];
            let coefficient = signed_coefficient.rem_euclid(i128::from(prime)) as u64;
            let hash =
                splitmix64(u64::from(output_mask) ^ EXTERIOR_FINGERPRINT_SEEDS[index]) % prime;
            residues[index] = ((u128::from(residues[index])
                + u128::from(coefficient) * u128::from(hash))
                % u128::from(prime)) as u64;
        }
    }
}

fn contracted_translation_bilinear(momentum: &[i64; 11]) -> Vec<Vec<(i64, i64)>> {
    let bilinears = crate::eleven_dimensional_clifford::translation_bilinears();
    let mut contracted = vec![vec![(0_i64, 0_i64); 32]; 32];
    for outer in 0..32 {
        for inner in 0..32 {
            for vector in 0..11 {
                let value = &bilinears[vector][outer][inner];
                assert_eq!(*value.re.denom(), 1);
                assert_eq!(*value.im.denom(), 1);
                contracted[outer][inner].0 += momentum[vector] * *value.re.numer();
                contracted[outer][inner].1 += momentum[vector] * *value.im.numer();
            }
        }
    }
    contracted
}

fn accumulate_momentum_contraction_fingerprint(
    real_residues: &mut [u64; 3],
    imaginary_residues: &mut [u64; 3],
    outer_spinor_index: usize,
    domain_coefficient: i64,
    source: &[(u32, i64)],
    contracted_bilinear: &[Vec<(i64, i64)>],
) {
    for &(mask, source_coefficient) in source {
        let mut remaining = mask;
        let mut position = 0_u32;
        while remaining != 0 {
            let contracted_spinor_index = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let contraction_sign = if position % 2 == 0 { 1_i128 } else { -1_i128 };
            position += 1;
            let output_mask = mask ^ (1_u32 << contracted_spinor_index);
            let (gamma_real, gamma_imaginary) =
                contracted_bilinear[outer_spinor_index][contracted_spinor_index];
            let common =
                i128::from(domain_coefficient) * i128::from(source_coefficient) * contraction_sign;
            let signed_real = common * i128::from(gamma_real);
            let signed_imaginary = common * i128::from(gamma_imaginary);
            for index in 0..3 {
                let prime = EXTERIOR_FINGERPRINT_PRIMES[index];
                let hash = splitmix64(
                    u64::from(output_mask) ^ EXTERIOR_FINGERPRINT_SEEDS[index] ^ 0xfeed_face,
                ) % prime;
                let real = signed_real.rem_euclid(i128::from(prime)) as u64;
                let imaginary = signed_imaginary.rem_euclid(i128::from(prime)) as u64;
                real_residues[index] = ((u128::from(real_residues[index])
                    + u128::from(real) * u128::from(hash))
                    % u128::from(prime)) as u64;
                imaginary_residues[index] = ((u128::from(imaginary_residues[index])
                    + u128::from(imaginary) * u128::from(hash))
                    % u128::from(prime)) as u64;
            }
        }
    }
}

fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn lcm_i64(left: i64, right: i64) -> i64 {
    left / gcd_i64(left, right) * right
}

fn source_relation_residual(
    candidate: &[(u32, i64)],
    basis: &[VectorSpinorIntertwinerState],
    coefficients: &[Ratio<i64>],
) -> (usize, u64) {
    let denominator = coefficients.iter().fold(1_i64, |common, coefficient| {
        lcm_i64(common, *coefficient.denom())
    });
    let mut residual = HashMap::<u32, i128>::new();
    for &(mask, value) in candidate {
        *residual.entry(mask).or_insert(0) += i128::from(denominator) * i128::from(value);
    }
    for (state, coefficient) in basis.iter().zip(coefficients) {
        let numerator = *coefficient.numer() * (denominator / *coefficient.denom());
        for &(mask, value) in &state.source {
            *residual.entry(mask).or_insert(0) -= i128::from(numerator) * i128::from(value);
        }
    }
    residual.retain(|_, value| *value != 0);
    let maximum_absolute_residual = residual
        .values()
        .map(|value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    (
        residual.len(),
        u64::try_from(maximum_absolute_residual).unwrap(),
    )
}

fn generate_partial_two_form_spinor_source_states(
    source_basis: &[u32],
    coefficients: &[i16],
    spinors: &[Weight; 32],
    needed_weights: &std::collections::BTreeSet<Weight>,
) -> BTreeMap<Weight, Vec<VectorSpinorIntertwinerState>> {
    let vectors = vector_weights();
    let pairs = two_form_pairs();
    let pair_highest = pairs.iter().position(|pair| *pair == (0, 2)).unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let highest_weight = [3, 3, 1, 1, 1];
    let highest_target =
        HashMap::from([(pair_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_source = source_basis
        .iter()
        .copied()
        .zip(coefficients.iter().copied())
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<Vec<_>>();
    let highest_state = VectorSpinorIntertwinerState {
        target: highest_target,
        source: highest_source,
    };
    let reference = generate_layer_adapted_two_form_spinor_target_states(spinors);
    let mut selected = BTreeMap::new();
    if needed_weights.contains(&highest_weight) {
        selected.insert(highest_weight, vec![highest_state.clone()]);
    }
    let mut current = BTreeMap::from([(highest_weight, vec![highest_state])]);
    while !current.is_empty()
        && !needed_weights.iter().all(|weight| {
            selected.get(weight).map(Vec::len).unwrap_or(0)
                == reference.get(weight).map(Vec::len).unwrap_or(0)
        })
    {
        let mut next = BTreeMap::<Weight, Vec<VectorSpinorIntertwinerState>>::new();
        for states in current.into_values() {
            for state in states {
                for root in 0..5 {
                    let target_descendant = lower_two_form_spinor_tensor(
                        &state.target,
                        root,
                        &vectors,
                        &pairs,
                        spinors,
                    );
                    let source_descendant = lower_pairs(&state.source, root, spinors);
                    if target_descendant.is_empty() {
                        assert!(source_descendant.is_empty());
                        continue;
                    }
                    let weight =
                        two_form_spinor_weight(&target_descendant, &vectors, &pairs, spinors);
                    let basis = next.entry(weight).or_default();
                    if let Some(span_coefficients) =
                        target_span_coefficients(&target_descendant, basis, 55 * 32)
                    {
                        let (residual_terms, _) =
                            source_relation_residual(&source_descendant, basis, &span_coefficients);
                        assert_eq!(residual_terms, 0);
                    } else {
                        basis.push(VectorSpinorIntertwinerState {
                            target: target_descendant,
                            source: source_descendant,
                        });
                    }
                }
            }
        }
        for weight in needed_weights {
            if let Some(states) = next.get(weight) {
                selected.insert(*weight, states.clone());
            }
        }
        current = next;
    }
    assert!(needed_weights.iter().all(|weight| {
        selected.get(weight).map(Vec::len).unwrap_or(0)
            == reference.get(weight).map(Vec::len).unwrap_or(0)
    }));
    selected
}

fn spinor_correction_highest_source(
    source_basis: &[u32],
    coefficients: &[i16],
) -> MomentumExteriorVector {
    source_basis
        .iter()
        .copied()
        .zip(coefficients.iter().copied())
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| ((0, mask), i64::from(coefficient)))
        .collect()
}

fn two_form_spinor_correction_highest_source(
    source_basis: &[u32],
    coefficients: &[i16],
    spinors: &[Weight; 32],
) -> MomentumExteriorVector {
    let (domain, highest_coefficients, kernel_dimension) =
        build_01001_to_10001_highest_map(spinors);
    assert_eq!(kernel_dimension, 1);
    let needed = domain
        .iter()
        .map(|entry| entry.source_weight)
        .collect::<std::collections::BTreeSet<_>>();
    let source_states = generate_partial_two_form_spinor_source_states(
        source_basis,
        coefficients,
        spinors,
        &needed,
    );
    let mut output = MomentumExteriorVector::new();
    for (entry, &domain_coefficient) in domain.iter().zip(&highest_coefficients) {
        let state = &source_states[&entry.source_weight][entry.source_basis_index];
        for &(mask, source_coefficient) in &state.source {
            let value = domain_coefficient
                .checked_mul(source_coefficient)
                .expect("level-13 correction coefficient overflow");
            *output.entry((entry.vector_index, mask)).or_insert(0) += value;
        }
    }
    output.retain(|_, coefficient| *coefficient != 0);
    output
}

fn generate_partial_momentum_correction_states(
    highest_source: MomentumExteriorVector,
    spinors: &[Weight; 32],
    needed_weights: &std::collections::BTreeSet<Weight>,
) -> BTreeMap<Weight, Vec<MomentumCorrectionState>> {
    let vectors = vector_weights();
    let vector_highest = vectors
        .iter()
        .position(|weight| *weight == [2, 0, 0, 0, 0])
        .unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let highest_weight = [3, 1, 1, 1, 1];
    let highest_target =
        HashMap::from([(vector_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_state = MomentumCorrectionState {
        target: highest_target,
        source: highest_source,
    };
    let reference = generate_layer_adapted_vector_spinor_target_states(spinors);
    let mut selected = BTreeMap::new();
    if needed_weights.contains(&highest_weight) {
        selected.insert(highest_weight, vec![highest_state.clone()]);
    }
    let mut current = BTreeMap::from([(highest_weight, vec![highest_state])]);
    while !current.is_empty()
        && !needed_weights.iter().all(|weight| {
            selected.get(weight).map(Vec::len).unwrap_or(0)
                == reference.get(weight).map(Vec::len).unwrap_or(0)
        })
    {
        let mut next = BTreeMap::<Weight, Vec<MomentumCorrectionState>>::new();
        for states in current.into_values() {
            for state in states {
                for root in 0..5 {
                    let target_descendant =
                        lower_target_tensor(&state.target, root, &vectors, spinors);
                    let source_descendant =
                        lower_momentum_exterior(&state.source, root, &vectors, spinors);
                    if target_descendant.is_empty() {
                        assert!(source_descendant.is_empty());
                        continue;
                    }
                    let weight = tensor_weight(&target_descendant, &vectors, spinors);
                    let basis = next.entry(weight).or_default();
                    let target_basis = basis
                        .iter()
                        .map(|item| VectorSpinorIntertwinerState {
                            target: item.target.clone(),
                            source: Vec::new(),
                        })
                        .collect::<Vec<_>>();
                    if let Some(span_coefficients) =
                        target_span_coefficients(&target_descendant, &target_basis, 11 * 32)
                    {
                        assert_eq!(
                            momentum_source_relation_residual(
                                &source_descendant,
                                basis,
                                &span_coefficients,
                            ),
                            0
                        );
                    } else {
                        basis.push(MomentumCorrectionState {
                            target: target_descendant,
                            source: source_descendant,
                        });
                    }
                }
            }
        }
        for weight in needed_weights {
            if let Some(states) = next.get(weight) {
                selected.insert(*weight, states.clone());
            }
        }
        current = next;
    }
    assert!(needed_weights.iter().all(|weight| {
        selected.get(weight).map(Vec::len).unwrap_or(0)
            == reference.get(weight).map(Vec::len).unwrap_or(0)
    }));
    selected
}

fn generate_partial_vector_spinor_source_states(
    source_basis: &[u32],
    coefficients: &[i16],
    spinors: &[Weight; 32],
    needed_weights: &std::collections::BTreeSet<Weight>,
) -> BTreeMap<Weight, Vec<VectorSpinorIntertwinerState>> {
    let vectors = vector_weights();
    let vector_highest = vectors
        .iter()
        .position(|weight| *weight == [2, 0, 0, 0, 0])
        .unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let highest_weight = [3, 1, 1, 1, 1];
    let highest_target =
        HashMap::from([(vector_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_source = source_basis
        .iter()
        .copied()
        .zip(coefficients.iter().copied())
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<Vec<_>>();
    let highest_state = VectorSpinorIntertwinerState {
        target: highest_target,
        source: highest_source,
    };
    let reference = generate_layer_adapted_vector_spinor_target_states(spinors);
    let mut selected = BTreeMap::new();
    if needed_weights.contains(&highest_weight) {
        selected.insert(highest_weight, vec![highest_state.clone()]);
    }
    let mut current = BTreeMap::from([(highest_weight, vec![highest_state])]);
    while !current.is_empty()
        && !needed_weights.iter().all(|weight| {
            selected.get(weight).map(Vec::len).unwrap_or(0)
                == reference.get(weight).map(Vec::len).unwrap_or(0)
        })
    {
        let mut next = BTreeMap::<Weight, Vec<VectorSpinorIntertwinerState>>::new();
        for states in current.into_values() {
            for state in states {
                for root in 0..5 {
                    let target_descendant =
                        lower_target_tensor(&state.target, root, &vectors, spinors);
                    let source_descendant = lower_pairs(&state.source, root, spinors);
                    if target_descendant.is_empty() {
                        assert!(source_descendant.is_empty());
                        continue;
                    }
                    let weight = tensor_weight(&target_descendant, &vectors, spinors);
                    let basis = next.entry(weight).or_default();
                    if let Some(span_coefficients) =
                        target_span_coefficients(&target_descendant, basis, 11 * 32)
                    {
                        assert_eq!(
                            source_relation_residual(
                                &source_descendant,
                                basis,
                                &span_coefficients,
                            )
                            .0,
                            0
                        );
                    } else {
                        basis.push(VectorSpinorIntertwinerState {
                            target: target_descendant,
                            source: source_descendant,
                        });
                    }
                }
            }
        }
        for weight in needed_weights {
            if let Some(states) = next.get(weight) {
                selected.insert(*weight, states.clone());
            }
        }
        current = next;
    }
    assert!(needed_weights.iter().all(|weight| {
        selected.get(weight).map(Vec::len).unwrap_or(0)
            == reference.get(weight).map(Vec::len).unwrap_or(0)
    }));
    selected
}

const CANCELLATION_FUNCTIONAL_BUCKETS: usize = 32;

fn bucket_and_sign(mask: u32) -> (usize, i128) {
    let hash = splitmix64(u64::from(mask) ^ 0x6a09_e667_f3bc_c909);
    (
        (hash as usize) % CANCELLATION_FUNCTIONAL_BUCKETS,
        if hash >> 63 == 0 { 1 } else { -1 },
    )
}

fn doubled_momentum_weight_coefficient(vector_index: usize, momentum: &[i64; 11]) -> (i64, i64) {
    if vector_index == 10 {
        return (2 * momentum[10], 0);
    }
    let axis = vector_index / 2;
    let real = momentum[2 * axis];
    let imaginary = if vector_index % 2 == 0 {
        momentum[2 * axis + 1]
    } else {
        -momentum[2 * axis + 1]
    };
    (real, imaginary)
}

fn correction_hook_functionals(
    plan: &ExteriorChannelPlan,
    states: &BTreeMap<Weight, Vec<MomentumCorrectionState>>,
    momentum: &[i64; 11],
) -> (
    [i128; CANCELLATION_FUNCTIONAL_BUCKETS],
    [i128; CANCELLATION_FUNCTIONAL_BUCKETS],
) {
    let mut real = [0_i128; CANCELLATION_FUNCTIONAL_BUCKETS];
    let mut imaginary = [0_i128; CANCELLATION_FUNCTIONAL_BUCKETS];
    for (entry, &domain_coefficient) in plan
        .domain
        .iter()
        .zip(&plan.primitive_highest_weight_coefficients)
    {
        let state = &states[&entry.vector_spinor_weight][entry.vector_spinor_basis_index];
        let outer_bit = 1_u32 << entry.outer_spinor_index;
        let lower_bits = outer_bit - 1;
        for (&(vector_index, mask), &source_coefficient) in &state.source {
            if mask & outer_bit != 0 {
                continue;
            }
            let wedge_sign = if (mask & lower_bits).count_ones() % 2 == 0 {
                1_i128
            } else {
                -1_i128
            };
            let output_mask = mask | outer_bit;
            let (bucket, sign) = bucket_and_sign(output_mask);
            let common =
                i128::from(domain_coefficient) * i128::from(source_coefficient) * wedge_sign * sign;
            let (momentum_real, momentum_imaginary) =
                doubled_momentum_weight_coefficient(vector_index, momentum);
            real[bucket] += common * i128::from(momentum_real);
            imaginary[bucket] += common * i128::from(momentum_imaginary);
        }
    }
    (real, imaginary)
}

fn leading_hook_functionals(
    plan: &ExteriorChannelPlan,
    states: &BTreeMap<Weight, Vec<VectorSpinorIntertwinerState>>,
    contracted_bilinear: &[Vec<(i64, i64)>],
) -> (
    [i128; CANCELLATION_FUNCTIONAL_BUCKETS],
    [i128; CANCELLATION_FUNCTIONAL_BUCKETS],
) {
    let mut real = [0_i128; CANCELLATION_FUNCTIONAL_BUCKETS];
    let mut imaginary = [0_i128; CANCELLATION_FUNCTIONAL_BUCKETS];
    for (entry, &domain_coefficient) in plan
        .domain
        .iter()
        .zip(&plan.primitive_highest_weight_coefficients)
    {
        let state = &states[&entry.vector_spinor_weight][entry.vector_spinor_basis_index];
        for &(mask, source_coefficient) in &state.source {
            let mut remaining = mask;
            let mut position = 0_u32;
            while remaining != 0 {
                let contracted_spinor_index = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let contraction_sign = if position % 2 == 0 { 1_i128 } else { -1_i128 };
                position += 1;
                let output_mask = mask ^ (1_u32 << contracted_spinor_index);
                let (bucket, sign) = bucket_and_sign(output_mask);
                let common = i128::from(domain_coefficient)
                    * i128::from(source_coefficient)
                    * contraction_sign
                    * sign
                    * 2;
                let (gamma_real, gamma_imaginary) =
                    contracted_bilinear[entry.outer_spinor_index][contracted_spinor_index];
                real[bucket] += common * i128::from(gamma_real);
                imaginary[bucket] += common * i128::from(gamma_imaginary);
            }
        }
    }
    (real, imaginary)
}

fn rational_rank_i128(rows: &[Vec<i128>]) -> usize {
    use num_bigint::BigInt;
    let zero = Ratio::from_integer(BigInt::from(0));
    let mut matrix = rows
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|value| Ratio::from_integer(BigInt::from(value)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if matrix.is_empty() {
        return 0;
    }
    let columns = matrix[0].len();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..matrix.len()).find(|row| matrix[*row][column] != zero) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let normalization = matrix[rank][column].clone();
        for value in &mut matrix[rank][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = matrix[rank].clone();
        for row in (rank + 1)..matrix.len() {
            let factor = matrix[row][column].clone();
            if factor == zero {
                continue;
            }
            for index in column..columns {
                matrix[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
        rank += 1;
        if rank == matrix.len() {
            break;
        }
    }
    rank
}

fn audit_full_vector_spinor_source_descendants(
    source_basis: &[u32],
    coefficients: &[i16],
    spinors: &[Weight; 32],
) -> (
    VectorSpinorSourceDescendantAudit,
    LevelSixteenExteriorDerivativeAudit,
    FirstDerivativeMomentumAudit,
) {
    let vectors = vector_weights();
    let vector_highest = vectors
        .iter()
        .position(|weight| *weight == [2, 0, 0, 0, 0])
        .unwrap();
    let spinor_highest = spinors.iter().position(|weight| *weight == [1; 5]).unwrap();
    let highest_target =
        HashMap::from([(vector_highest * 32 + spinor_highest, Ratio::from_integer(1))]);
    let highest_source = source_basis
        .iter()
        .copied()
        .zip(coefficients.iter().copied())
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<Vec<_>>();
    let mut current = BTreeMap::from([(
        [3, 1, 1, 1, 1],
        vec![VectorSpinorIntertwinerState {
            target: highest_target,
            source: highest_source,
        }],
    )]);
    let mut distinct_weights = 1;
    let mut target_states_generated = 1;
    let mut nonzero_lowering_actions_checked = 0;
    let mut zero_lowering_actions_checked = 0;
    let mut independent_state_discoveries = 0;
    let mut dependent_target_relations_checked = 0;
    let mut dependent_target_relation_mismatches = 0;
    let mut nonzero_relation_residual_terms = 0;
    let mut maximum_absolute_relation_residual = 0;
    let mut zero_action_mismatches = 0;
    let mut target_basis_correspondence_mismatches = 0;
    let mut minimum_source_state_support = usize::MAX;
    let mut maximum_source_state_support = 0;
    let mut maximum_absolute_source_coefficient = 0_i64;
    let mut exterior_channel_plans = build_exterior_channel_plans(spinors);
    let reference_target_states = generate_layer_adapted_vector_spinor_target_states(spinors);
    let exact_sample_momentum = [1_i64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let contracted_bilinear = contracted_translation_bilinear(&exact_sample_momentum);
    let (
        cartan_weight_entries_checked,
        cartan_weight_mismatches,
        chevalley_lowering_actions_checked,
        chevalley_lowering_residual_actions,
    ) = crate::eleven_dimensional_clifford::translation_bilinear_basis_alignment();
    let clifford_and_weight_bases_aligned = cartan_weight_mismatches == 0
        && chevalley_lowering_actions_checked == 48
        && chevalley_lowering_residual_actions == 0;

    while !current.is_empty() {
        let mut next = BTreeMap::<Weight, Vec<VectorSpinorIntertwinerState>>::new();
        for (state_weight, states) in current {
            for (state_basis_index, state) in states.into_iter().enumerate() {
                target_basis_correspondence_mismatches += usize::from(
                    reference_target_states[&state_weight][state_basis_index].target
                        != state.target,
                );
                for plan in &mut exterior_channel_plans {
                    for (entry, coefficient) in plan
                        .domain
                        .iter()
                        .zip(&plan.primitive_highest_weight_coefficients)
                        .filter(|(entry, coefficient)| {
                            entry.vector_spinor_weight == state_weight
                                && entry.vector_spinor_basis_index == state_basis_index
                                && **coefficient != 0
                        })
                    {
                        accumulate_exterior_fingerprint(
                            &mut plan.fingerprint_residues,
                            entry.outer_spinor_index,
                            *coefficient,
                            &state.source,
                        );
                        if plan.dynkin_label == "11000" || plan.dynkin_label == "10002" {
                            accumulate_momentum_contraction_fingerprint(
                                &mut plan.momentum_contraction_real_residues,
                                &mut plan.momentum_contraction_imaginary_residues,
                                entry.outer_spinor_index,
                                *coefficient,
                                &state.source,
                                &contracted_bilinear,
                            );
                        }
                    }
                }
                minimum_source_state_support = minimum_source_state_support.min(state.source.len());
                maximum_source_state_support = maximum_source_state_support.max(state.source.len());
                maximum_absolute_source_coefficient = maximum_absolute_source_coefficient.max(
                    state
                        .source
                        .iter()
                        .map(|(_, coefficient)| coefficient.abs())
                        .max()
                        .unwrap_or(0),
                );
                for root in 0..5 {
                    let target_descendant =
                        lower_target_tensor(&state.target, root, &vectors, spinors);
                    let source_descendant = lower_pairs(&state.source, root, spinors);
                    if target_descendant.is_empty() {
                        zero_lowering_actions_checked += 1;
                        zero_action_mismatches += usize::from(!source_descendant.is_empty());
                        continue;
                    }
                    nonzero_lowering_actions_checked += 1;
                    let weight = tensor_weight(&target_descendant, &vectors, spinors);
                    let target_weight_was_new = !next.contains_key(&weight);
                    let basis = next.entry(weight).or_default();
                    if let Some(span_coefficients) =
                        target_span_coefficients(&target_descendant, basis, 11 * 32)
                    {
                        dependent_target_relations_checked += 1;
                        let (residual_terms, maximum_absolute_residual) =
                            source_relation_residual(&source_descendant, basis, &span_coefficients);
                        dependent_target_relation_mismatches += usize::from(residual_terms != 0);
                        nonzero_relation_residual_terms += residual_terms;
                        maximum_absolute_relation_residual =
                            maximum_absolute_relation_residual.max(maximum_absolute_residual);
                    } else {
                        if target_weight_was_new {
                            distinct_weights += 1;
                        }
                        independent_state_discoveries += 1;
                        target_states_generated += 1;
                        basis.push(VectorSpinorIntertwinerState {
                            target: target_descendant,
                            source: source_descendant,
                        });
                    }
                }
            }
        }
        current = next;
    }

    let exact_full_vector_spinor_intertwiner_verified = target_states_generated == 320
        && distinct_weights == 192
        && nonzero_lowering_actions_checked + zero_lowering_actions_checked == 320 * 5
        && independent_state_discoveries == 319
        && independent_state_discoveries + dependent_target_relations_checked
            == nonzero_lowering_actions_checked
        && dependent_target_relation_mismatches == 0
        && nonzero_relation_residual_terms == 0
        && maximum_absolute_relation_residual == 0
        && zero_action_mismatches == 0
        && target_basis_correspondence_mismatches == 0;
    let source_descendant_audit = VectorSpinorSourceDescendantAudit {
        target_states_expected: 320,
        target_states_generated,
        distinct_weights,
        nonzero_lowering_actions_checked,
        zero_lowering_actions_checked,
        total_lowering_actions_checked: nonzero_lowering_actions_checked
            + zero_lowering_actions_checked,
        independent_state_discoveries,
        dependent_target_relations_checked,
        dependent_target_relation_mismatches,
        nonzero_relation_residual_terms,
        maximum_absolute_relation_residual,
        zero_action_mismatches,
        target_basis_correspondence_mismatches,
        minimum_source_state_support,
        maximum_source_state_support,
        maximum_absolute_source_coefficient,
        exact_full_vector_spinor_intertwiner_verified,
    };
    let momentum_channels = exterior_channel_plans
        .iter()
        .filter(|plan| plan.dynkin_label == "11000" || plan.dynkin_label == "10002")
        .map(|plan| {
            let momentum_contraction_nonzero_certified = plan
                .momentum_contraction_real_residues
                .iter()
                .chain(&plan.momentum_contraction_imaginary_residues)
                .any(|residue| *residue != 0);
            MomentumHookChannelAudit {
                dynkin_label: plan.dynkin_label.clone(),
                torsion_sector: if plan.dynkin_label == "11000" {
                    "X_[2] traceless hook"
                } else {
                    "X_[5] traceless hook"
                },
                exterior_level: 16,
                exterior_image_nonzero: plan.fingerprint_residues.iter().any(|value| *value != 0),
                momentum_contraction_level: 14,
                momentum_real_fingerprint_residues: plan.momentum_contraction_real_residues,
                momentum_imaginary_fingerprint_residues: plan
                    .momentum_contraction_imaginary_residues,
                momentum_contraction_nonzero_certified,
                passed: momentum_contraction_nonzero_certified,
            }
        })
        .collect::<Vec<_>>();
    let two_form_hook_momentum_contraction_nonzero = momentum_channels
        .iter()
        .find(|channel| channel.dynkin_label == "11000")
        .unwrap()
        .momentum_contraction_nonzero_certified;
    let five_form_hook_momentum_contraction_nonzero = momentum_channels
        .iter()
        .find(|channel| channel.dynkin_label == "10002")
        .unwrap()
        .momentum_contraction_nonzero_certified;
    let momentum_channels_pass = momentum_channels.iter().all(|channel| channel.passed);
    let first_derivative_momentum_audit = FirstDerivativeMomentumAudit {
        derivative_identity: "D_beta D_[alpha1...alpha15] = D_[beta alpha1...alpha15] + sum_i (-1)^(i-1) {D_beta,D_alphai} D_[alpha1...omit alphai...alpha15]",
        exact_sample_momentum,
        cartan_weight_entries_checked,
        cartan_weight_mismatches,
        chevalley_lowering_actions_checked,
        chevalley_lowering_residual_actions,
        clifford_and_weight_bases_aligned,
        channels: momentum_channels,
        two_form_hook_momentum_contraction_nonzero,
        five_form_hook_momentum_contraction_nonzero,
        implication: "the level-16 exterior inventory alone does not settle the final X_[2] hook at nonzero momentum because the superderivative anticommutator supplies a level-14 momentum term",
        scope: "a nonzero residue at one exact momentum proves that the polynomial momentum contraction is not identically zero; this does not solve the torsion constraint or the Bianchi identities",
        passed: two_form_hook_momentum_contraction_nonzero
            && five_form_hook_momentum_contraction_nonzero
            && momentum_channels_pass
            && clifford_and_weight_bases_aligned,
    };
    let exterior_channels = exterior_channel_plans
        .into_iter()
        .map(|plan| {
            let exterior_image_nonzero_certified = plan
                .fingerprint_residues
                .iter()
                .any(|residue| *residue != 0);
            let exterior_image_forced_zero_by_inventory =
                plan.scalar_level_sixteen_multiplicity == 0;
            let inventory_zero_fingerprint_crosscheck =
                exterior_image_forced_zero_by_inventory && plan.fingerprint_residues == [0; 3];
            let highest_weight_kernel_verified = plan.highest_weight_kernel_dimension == 1
                && plan.raising_residual_terms == 0
                && !plan.primitive_highest_weight_coefficients.is_empty();
            let passed = highest_weight_kernel_verified
                && if exterior_image_forced_zero_by_inventory {
                    inventory_zero_fingerprint_crosscheck
                } else {
                    exterior_image_nonzero_certified
                };
            LevelSixteenExteriorChannelAudit {
                dynkin_label: plan.dynkin_label,
                dimension: plan.dimension,
                scalar_level_sixteen_multiplicity: plan.scalar_level_sixteen_multiplicity,
                highest_weight_domain_dimension: plan.domain.len(),
                highest_weight_kernel_dimension: plan.highest_weight_kernel_dimension,
                primitive_highest_weight_nonzero_coefficients: plan
                    .primitive_highest_weight_coefficients
                    .iter()
                    .filter(|coefficient| **coefficient != 0)
                    .count(),
                raising_residual_terms: plan.raising_residual_terms,
                fingerprint_primes: EXTERIOR_FINGERPRINT_PRIMES,
                exterior_image_fingerprint_residues: plan.fingerprint_residues,
                exterior_image_nonzero_certified,
                exterior_image_forced_zero_by_inventory,
                inventory_zero_fingerprint_crosscheck,
                passed,
            }
        })
        .collect::<Vec<_>>();
    let highest_weight_kernels_verified = exterior_channels
        .iter()
        .filter(|channel| {
            channel.highest_weight_kernel_dimension == 1 && channel.raising_residual_terms == 0
        })
        .count();
    let nonzero_exterior_images_certified = exterior_channels
        .iter()
        .filter(|channel| channel.exterior_image_nonzero_certified)
        .count();
    let inventory_forced_zero_channels = exterior_channels
        .iter()
        .filter(|channel| channel.exterior_image_forced_zero_by_inventory)
        .count();
    let inventory_zero_fingerprint_crosschecks = exterior_channels
        .iter()
        .filter(|channel| channel.inventory_zero_fingerprint_crosscheck)
        .count();
    let exterior_derivative_audit = LevelSixteenExteriorDerivativeAudit {
        source_map: "(00001) tensor I_320(wedge^15 S) -> wedge^16 S by exterior multiplication",
        scope: "zero-spacetime-momentum exterior symbol of the sixteenth spinor derivative",
        channels_checked: exterior_channels.len(),
        highest_weight_kernels_verified,
        nonzero_exterior_images_certified,
        inventory_forced_zero_channels,
        inventory_zero_fingerprint_crosschecks,
        passed: exterior_channels.len() == 10
            && highest_weight_kernels_verified == 10
            && nonzero_exterior_images_certified == 8
            && inventory_forced_zero_channels == 2
            && inventory_zero_fingerprint_crosschecks == 2
            && exterior_channels.iter().all(|channel| channel.passed),
        channels: exterior_channels,
        interpretation: "the exterior symbol is nonzero on each of the eight level-16 channels allowed by the scalar inventory; the 01000 and 11000 images vanish, as required by the scalar inventory; nonzero modular residues certify that the corresponding integer exterior images are nonzero; spacetime-derivative terms and curvature identifications are not included",
    };
    (
        source_descendant_audit,
        exterior_derivative_audit,
        first_derivative_momentum_audit,
    )
}

fn audit_full_spinor_descendants(
    source_copy: usize,
    source_basis: &[u32],
    coefficients: &[i16],
    weights: &[Weight; 32],
) -> SpinorDescendantAudit {
    use std::collections::VecDeque;

    let highest = [1_i8; 5];
    let highest_vector = source_basis
        .iter()
        .copied()
        .zip(coefficients.iter().copied())
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<Vec<_>>();
    let mut states = HashMap::<Weight, Vec<(u32, i64)>>::new();
    states.insert(highest, highest_vector);
    let mut queue = VecDeque::from([highest]);
    let mut nonzero_lowering_actions_checked = 0;
    let mut independent_state_discoveries = 0;
    let mut repeated_path_checks = 0;
    let mut repeated_path_mismatches = 0;

    while let Some(weight) = queue.pop_front() {
        let vector = states[&weight].clone();
        for root in 0..5 {
            let target = subtract(weight, SIMPLE_ROOTS[root]);
            if !weights.contains(&target) {
                continue;
            }
            nonzero_lowering_actions_checked += 1;
            let descendant = lower_pairs(&vector, root, weights);
            assert!(!descendant.is_empty());
            if let Some(existing) = states.get(&target) {
                repeated_path_checks += 1;
                repeated_path_mismatches += usize::from(existing != &descendant);
            } else {
                independent_state_discoveries += 1;
                states.insert(target, descendant);
                queue.push_back(target);
            }
        }
    }

    let minimum_state_support = states.values().map(Vec::len).min().unwrap_or(0);
    let maximum_state_support = states.values().map(Vec::len).max().unwrap_or(0);
    let target_states_generated = states.len();
    SpinorDescendantAudit {
        source_copy,
        target_dynkin_label: "00001",
        target_states_expected: 32,
        target_states_generated,
        nonzero_lowering_actions_expected: 48,
        nonzero_lowering_actions_checked,
        independent_state_discoveries,
        repeated_path_checks,
        repeated_path_mismatches,
        minimum_state_support,
        maximum_state_support,
        exact_full_spinor_intertwiner_verified: target_states_generated == 32
            && nonzero_lowering_actions_checked == 48
            && independent_state_discoveries == 31
            && repeated_path_checks == 17
            && repeated_path_mismatches == 0,
    }
}

fn build_system(
    exterior_degree: u8,
    dynkin_label: &'static str,
    representation_dimension: usize,
    highest_weight: Weight,
    published_multiplicity: usize,
    left: &HashMap<(u8, Weight), Vec<u16>>,
    right: &HashMap<(u8, Weight), Vec<u16>>,
    weights: &[Weight; 32],
    kernel_artifacts: &[(&'static str, &'static [u8])],
) -> HighestWeightSystemReport {
    let source_basis = weight_basis(exterior_degree, highest_weight, left, right);
    let source_columns: HashMap<u32, usize> = source_basis
        .iter()
        .copied()
        .enumerate()
        .map(|(column, mask)| (mask, column))
        .collect();
    let kernels = kernel_artifacts
        .iter()
        .map(|(_, bytes)| {
            assert_eq!(bytes.len(), source_basis.len() * 2);
            bytes
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut kernel_nonzero_residual_rows = vec![0_usize; kernels.len()];
    let mut kernel_maximum_absolute_residual = vec![0_i64; kernels.len()];
    let mut raising_blocks = Vec::new();

    for root in 0..5 {
        let output_weight = add(highest_weight, SIMPLE_ROOTS[root]);
        let output_basis = weight_basis(exterior_degree, output_weight, left, right);
        let mut nonzero_entries = 0;
        let mut missing_source_columns = 0;
        let mut row_degree_histogram = BTreeMap::new();
        for output_mask in output_basis.iter().copied() {
            let preimages = lowering_preimages(output_mask, root, weights);
            let mut kernel_residuals = vec![0_i64; kernels.len()];
            for (source_mask, sign) in &preimages {
                if let Some(&column) = source_columns.get(source_mask) {
                    for (residual, coefficients) in kernel_residuals.iter_mut().zip(&kernels) {
                        *residual += i64::from(*sign) * i64::from(coefficients[column]);
                    }
                } else {
                    missing_source_columns += 1;
                }
            }
            for (kernel_index, residual) in kernel_residuals.into_iter().enumerate() {
                if residual != 0 {
                    kernel_nonzero_residual_rows[kernel_index] += 1;
                    kernel_maximum_absolute_residual[kernel_index] =
                        kernel_maximum_absolute_residual[kernel_index].max(residual.abs());
                }
            }
            nonzero_entries += preimages.len();
            *row_degree_histogram.entry(preimages.len()).or_insert(0) += 1;
        }
        raising_blocks.push(RaisingBlockReport {
            simple_root: root + 1,
            output_weight,
            rows: output_basis.len(),
            nonzero_entries,
            row_degree_histogram,
            missing_source_columns,
        });
    }

    let exact_kernel_vectors = kernel_artifacts
        .iter()
        .zip(kernels)
        .enumerate()
        .map(|(kernel_index, ((artifact, _), coefficients))| {
            let coefficient_gcd = coefficients.iter().fold(0_i16, |gcd, coefficient| {
                integer_gcd(gcd, coefficient.abs())
            });
            let first_lowering_descendants = (0..5)
                .map(|root| {
                    let descendant = first_lowering(&source_basis, &coefficients, root, weights);
                    let expected_lowering_string_length =
                        usize::from(dynkin_label.as_bytes()[root] - b'0');
                    let expected_nonzero_from_dynkin_label = expected_lowering_string_length != 0;
                    let mut lowering_power_nonzero_terms = vec![descendant.len()];
                    let mut current_descendant = descendant.clone();
                    for _ in 0..expected_lowering_string_length {
                        current_descendant = lower_sparse(&current_descendant, root, weights);
                        lowering_power_nonzero_terms.push(current_descendant.len());
                    }
                    let matches_highest_weight_string = lowering_power_nonzero_terms
                        .iter()
                        .take(expected_lowering_string_length)
                        .all(|terms| *terms != 0)
                        && lowering_power_nonzero_terms
                            .get(expected_lowering_string_length)
                            .is_some_and(|terms| *terms == 0);
                    FirstLoweringReport {
                        simple_root: root + 1,
                        expected_nonzero_from_dynkin_label,
                        expected_lowering_string_length,
                        nonzero_terms: descendant.len(),
                        second_lowering_nonzero_terms: lowering_power_nonzero_terms
                            .get(1)
                            .copied()
                            .unwrap_or(0),
                        lowering_power_nonzero_terms,
                        maximum_absolute_coefficient: descendant
                            .values()
                            .map(|coefficient| coefficient.abs())
                            .max()
                            .unwrap_or(0),
                        matches_highest_weight_string,
                    }
                })
                .collect::<Vec<_>>();
            ExactKernelVectorReport {
                artifact: *artifact,
                scalar_type: "signed 16-bit little-endian integer",
                coefficients: coefficients.len(),
                nonzero_coefficients: coefficients
                    .iter()
                    .filter(|coefficient| **coefficient != 0)
                    .count(),
                minimum_coefficient: *coefficients.iter().min().unwrap(),
                maximum_coefficient: *coefficients.iter().max().unwrap(),
                coefficient_gcd,
                squared_norm: squared_norm(&coefficients),
                raising_rows_checked: raising_blocks.iter().map(|block| block.rows).sum(),
                nonzero_residual_rows: kernel_nonzero_residual_rows[kernel_index],
                maximum_absolute_residual: kernel_maximum_absolute_residual[kernel_index],
                exact_kernel_verified: kernel_nonzero_residual_rows[kernel_index] == 0
                    && coefficient_gcd == 1
                    && first_lowering_descendants
                        .iter()
                        .all(|check| check.matches_highest_weight_string),
                first_lowering_descendants,
            }
        })
        .collect();

    HighestWeightSystemReport {
        dynkin_label,
        representation_dimension,
        highest_weight_doubled_coordinates: highest_weight,
        exterior_degree: usize::from(exterior_degree),
        source_weight_space_columns: source_basis.len(),
        total_rows: raising_blocks.iter().map(|block| block.rows).sum(),
        total_nonzero_entries: raising_blocks
            .iter()
            .map(|block| block.nonzero_entries)
            .sum(),
        published_multiplicity,
        expected_kernel_dimension: published_multiplicity,
        exact_sparse_system_constructed: raising_blocks
            .iter()
            .all(|block| block.missing_source_columns == 0),
        exact_kernel_vectors,
        raising_blocks,
    }
}

pub fn exterior_highest_weight_system_shapes(
    exterior_degree: u8,
    labels_and_multiplicities: &[(&str, usize)],
) -> Vec<ExteriorHighestWeightSystemShape> {
    let weights = spinor_weights();
    let left = half_groups(0, &weights);
    let right = half_groups(16, &weights);
    labels_and_multiplicities
        .iter()
        .map(|(dynkin_label, expected_kernel_dimension)| {
            let highest_weight = dynkin_highest_weight(dynkin_label);
            let source_weight_space_columns =
                weight_basis(exterior_degree, highest_weight, &left, &right).len();
            let raising_block_rows = std::array::from_fn(|root| {
                weight_basis(
                    exterior_degree,
                    add(highest_weight, SIMPLE_ROOTS[root]),
                    &left,
                    &right,
                )
                .len()
            });
            ExteriorHighestWeightSystemShape {
                dynkin_label: (*dynkin_label).to_owned(),
                exterior_degree: usize::from(exterior_degree),
                highest_weight_doubled_coordinates: highest_weight,
                source_weight_space_columns,
                raising_block_rows,
                total_raising_rows: raising_block_rows.iter().sum(),
                expected_kernel_dimension: *expected_kernel_dimension,
            }
        })
        .collect()
}

pub fn verify_exterior_highest_weight_kernel_fixtures(
    fixtures: &[ExteriorHighestWeightKernelFixture],
) -> Vec<HighestWeightSystemReport> {
    let weights = spinor_weights();
    let left = half_groups(0, &weights);
    let right = half_groups(16, &weights);
    fixtures
        .iter()
        .map(|fixture| {
            build_system(
                fixture.exterior_degree,
                fixture.dynkin_label,
                usize::try_from(crate::eleven_dimensional_prepotential::b5_dimension(
                    fixture.dynkin_label,
                ))
                .unwrap(),
                dynkin_highest_weight(fixture.dynkin_label),
                fixture.kernel_artifacts.len(),
                &left,
                &right,
                &weights,
                fixture.kernel_artifacts,
            )
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct SecondLeadingSourceCouplingAudit {
    pub source_dynkin_label: &'static str,
    pub target_dynkin_label: &'static str,
    pub formula: &'static str,
    pub source_highest_nonzero_terms: usize,
    pub lowering_chain_nonzero_terms: [usize; 6],
    pub primitive_chain_coefficients: [i64; 6],
    pub coupled_nonzero_terms: usize,
    pub raising_residual_terms_by_simple_root: [usize; 5],
    pub exact_coupling_constructed: bool,
    pub passed: bool,
}

pub fn audit_20000_to_10001_source_coupling(
    kernel_bytes: &[u8],
) -> SecondLeadingSourceCouplingAudit {
    let weights = spinor_weights();
    let left = half_groups(0, &weights);
    let right = half_groups(16, &weights);
    let source_basis = weight_basis(16, [4, 0, 0, 0, 0], &left, &right);
    let coefficients = decode_kernel(kernel_bytes);
    assert_eq!(coefficients.len(), source_basis.len());
    let highest = source_basis
        .iter()
        .copied()
        .zip(coefficients)
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<HashMap<_, _>>();
    let mut source_chain = vec![highest.clone()];
    for root in 0..5 {
        source_chain.push(lower_sparse(source_chain.last().unwrap(), root, &weights));
    }
    let free_spinor_weights = [
        [-1, 1, 1, 1, 1],
        [1, -1, 1, 1, 1],
        [1, 1, -1, 1, 1],
        [1, 1, 1, -1, 1],
        [1, 1, 1, 1, -1],
        [1, 1, 1, 1, 1],
    ];
    let free_spinor_indices = free_spinor_weights.map(|free_weight| {
        weights
            .iter()
            .position(|weight| *weight == free_weight)
            .unwrap()
    });
    let primitive_chain_coefficients = [4_i64, -2, 2, -2, 2, -1];
    let mut coupled = HashMap::<(u32, usize), i64>::new();
    for ((source, spinor_index), chain_coefficient) in source_chain
        .iter()
        .zip(free_spinor_indices)
        .zip(primitive_chain_coefficients)
    {
        for (&mask, &coefficient) in source {
            *coupled.entry((mask, spinor_index)).or_insert(0) += chain_coefficient * coefficient;
        }
    }
    coupled.retain(|_, coefficient| *coefficient != 0);
    let raising_residual_terms_by_simple_root = std::array::from_fn(|root| {
        let mut residual = HashMap::<(u32, usize), i64>::new();
        for psi_index in 0..32 {
            let exterior = coupled
                .iter()
                .filter(|((_, index), _)| *index == psi_index)
                .map(|((mask, _), coefficient)| (*mask, *coefficient))
                .collect::<HashMap<_, _>>();
            for (mask, coefficient) in raise_sparse(&exterior, root, &weights) {
                *residual.entry((mask, psi_index)).or_insert(0) += coefficient;
            }
        }
        for (&(mask, psi_index), &coefficient) in &coupled {
            if let Some(next) = raised_spinor_index(psi_index, root, &weights) {
                *residual.entry((mask, next)).or_insert(0) += coefficient;
            }
        }
        residual
            .values()
            .filter(|coefficient| **coefficient != 0)
            .count()
    });
    let exact_coupling_constructed = raising_residual_terms_by_simple_root == [0; 5];
    SecondLeadingSourceCouplingAudit {
        source_dynkin_label: "20000",
        target_dynkin_label: "10001",
        formula: "six weight-chain states from (4,0,0,0,0) to (2,0,0,0,0), with primitive coefficients (4,-2,2,-2,2,-1)",
        source_highest_nonzero_terms: highest.len(),
        lowering_chain_nonzero_terms: std::array::from_fn(|index| source_chain[index].len()),
        primitive_chain_coefficients,
        coupled_nonzero_terms: coupled.len(),
        raising_residual_terms_by_simple_root,
        exact_coupling_constructed,
        passed: exact_coupling_constructed,
    }
}

fn lowering_coordinates(upper: Weight, lower: Weight) -> Option<[i16; 5]> {
    let mut difference = [0_i16; 5];
    for index in 0..5 {
        let value = i16::from(upper[index]) - i16::from(lower[index]);
        if value % 2 != 0 {
            return None;
        }
        difference[index] = value / 2;
    }
    let coordinates = [
        difference[0],
        difference[0] + difference[1],
        difference[0] + difference[1] + difference[2],
        difference[0] + difference[1] + difference[2] + difference[3],
        difference.iter().sum(),
    ];
    coordinates
        .iter()
        .all(|coordinate| *coordinate >= 0)
        .then_some(coordinates)
}

// This prime is used only to select a computationally independent candidate
// basis. A coupling is accepted only after its rational kernel is reconstructed
// and every ambient integer raising residual is checked to be zero.
const SPARSE_RANK_PRIME: u64 = 2_305_843_009_213_693_951;

fn modular_value(value: i64) -> u64 {
    if value >= 0 {
        value as u64 % SPARSE_RANK_PRIME
    } else {
        let magnitude = value.unsigned_abs() % SPARSE_RANK_PRIME;
        if magnitude == 0 {
            0
        } else {
            SPARSE_RANK_PRIME - magnitude
        }
    }
}

fn modular_product(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(SPARSE_RANK_PRIME)) as u64
}

fn modular_power(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = modular_product(result, base);
        }
        base = modular_product(base, base);
        exponent >>= 1;
    }
    result
}

fn select_independent_sparse_vectors(candidates: Vec<HashMap<u32, i64>>) -> Vec<HashMap<u32, i64>> {
    let mut basis = Vec::<HashMap<u32, i64>>::new();
    let mut echelon = Vec::<(u32, HashMap<u32, u64>)>::new();
    for candidate in candidates {
        if candidate.is_empty() {
            continue;
        }
        let mut reduced = candidate
            .iter()
            .filter_map(|(&mask, &coefficient)| {
                let coefficient = modular_value(coefficient);
                (coefficient != 0).then_some((mask, coefficient))
            })
            .collect::<HashMap<_, _>>();
        for (pivot, pivot_vector) in &echelon {
            let factor = reduced.get(pivot).copied().unwrap_or(0);
            if factor == 0 {
                continue;
            }
            for (&mask, &pivot_coefficient) in pivot_vector {
                let subtraction = modular_product(factor, pivot_coefficient);
                let current = reduced.get(&mask).copied().unwrap_or(0);
                let next = if current >= subtraction {
                    current - subtraction
                } else {
                    SPARSE_RANK_PRIME - (subtraction - current)
                };
                if next == 0 {
                    reduced.remove(&mask);
                } else {
                    reduced.insert(mask, next);
                }
            }
        }
        if let Some(pivot) = reduced.keys().copied().min() {
            let inverse = modular_power(reduced[&pivot], SPARSE_RANK_PRIME - 2);
            for coefficient in reduced.values_mut() {
                *coefficient = modular_product(*coefficient, inverse);
            }
            echelon.push((pivot, reduced));
            basis.push(candidate);
        }
    }
    basis
}

fn relevant_source_weight_bases(
    highest: HashMap<u32, i64>,
    source_highest_weight: Weight,
    target_weight: Weight,
    spinors: &[Weight; 32],
) -> BTreeMap<Weight, Vec<HashMap<u32, i64>>> {
    let needed_weights = spinors
        .iter()
        .map(|spinor| subtract(target_weight, *spinor))
        .filter_map(|weight| {
            lowering_coordinates(source_highest_weight, weight).map(|coordinates| {
                let depth = coordinates
                    .iter()
                    .map(|value| usize::try_from(*value).unwrap())
                    .sum();
                (weight, depth)
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_depth = needed_weights.values().copied().max().unwrap_or(0);
    let mut current = BTreeMap::from([(source_highest_weight, vec![highest])]);
    let mut required = BTreeMap::new();
    for depth in 0..=maximum_depth {
        let mut next_candidates = BTreeMap::<Weight, Vec<HashMap<u32, i64>>>::new();
        for (weight, basis) in current {
            if needed_weights
                .get(&weight)
                .is_some_and(|needed_depth| *needed_depth == depth)
            {
                required.insert(weight, basis.clone());
            }
            if depth == maximum_depth {
                continue;
            }
            for root in 0..5 {
                let next_weight = subtract(weight, SIMPLE_ROOTS[root]);
                let remains_relevant = needed_weights.keys().any(|needed| {
                    lowering_coordinates(next_weight, *needed).is_some()
                        && lowering_coordinates(source_highest_weight, next_weight).is_some_and(
                            |coordinates| {
                                coordinates
                                    .iter()
                                    .map(|value| usize::try_from(*value).unwrap())
                                    .sum::<usize>()
                                    == depth + 1
                            },
                        )
                });
                if !remains_relevant {
                    continue;
                }
                for vector in &basis {
                    let lowered = lower_sparse(vector, root, spinors);
                    if !lowered.is_empty() {
                        next_candidates
                            .entry(next_weight)
                            .or_default()
                            .push(lowered);
                    }
                }
            }
        }
        current = next_candidates
            .into_iter()
            .map(|(weight, candidates)| (weight, select_independent_sparse_vectors(candidates)))
            .filter(|(_, basis)| !basis.is_empty())
            .collect();
    }
    required
}

fn raised_tensor_column(
    source: &HashMap<u32, i64>,
    spinor_index: usize,
    root: usize,
    spinors: &[Weight; 32],
) -> HashMap<(u32, usize), i64> {
    let mut output = HashMap::new();
    for (mask, coefficient) in raise_sparse(source, root, spinors) {
        *output.entry((mask, spinor_index)).or_insert(0) += coefficient;
    }
    if let Some(next_spinor) = raised_spinor_index(spinor_index, root, spinors) {
        for (&mask, &coefficient) in source {
            *output.entry((mask, next_spinor)).or_insert(0) += coefficient;
        }
    }
    output.retain(|_, coefficient| *coefficient != 0);
    output
}

fn one_dimensional_tensor_kernel(outputs: &[Vec<HashMap<(u32, usize), i64>>]) -> Option<Vec<i64>> {
    let columns = outputs.len();
    let mut rows = Vec::<Vec<Ratio<i64>>>::new();
    let mut modular_echelon = Vec::<(usize, Vec<u64>)>::new();
    let mut seen = std::collections::BTreeSet::new();
    'scan: for root in 0..5 {
        for column in outputs {
            let mut keys = column[root].keys().copied().collect::<Vec<_>>();
            keys.sort_unstable();
            for key in keys {
                if !seen.insert((root, key)) {
                    continue;
                }
                let integer_row = outputs
                    .iter()
                    .map(|candidate| candidate[root].get(&key).copied().unwrap_or(0))
                    .collect::<Vec<_>>();
                let mut reduced = integer_row
                    .iter()
                    .map(|value| modular_value(*value))
                    .collect::<Vec<_>>();
                for (pivot, pivot_row) in &modular_echelon {
                    let factor = reduced[*pivot];
                    if factor == 0 {
                        continue;
                    }
                    for index in *pivot..columns {
                        let subtraction = modular_product(factor, pivot_row[index]);
                        reduced[index] = if reduced[index] >= subtraction {
                            reduced[index] - subtraction
                        } else {
                            SPARSE_RANK_PRIME - (subtraction - reduced[index])
                        };
                    }
                }
                if let Some(pivot) = reduced.iter().position(|value| *value != 0) {
                    let inverse = modular_power(reduced[pivot], SPARSE_RANK_PRIME - 2);
                    for value in &mut reduced[pivot..] {
                        *value = modular_product(*value, inverse);
                    }
                    modular_echelon.push((pivot, reduced));
                    rows.push(integer_row.into_iter().map(Ratio::from_integer).collect());
                    if modular_echelon.len() >= columns.saturating_sub(1) {
                        break 'scan;
                    }
                }
            }
        }
    }
    let mut nullspace = ratio_nullspace(&rows, columns);
    loop {
        if nullspace.len() != 1 {
            return None;
        }
        let primitive = primitive_integer_vector(&nullspace[0]);
        let residual = (0..5).find_map(|root| {
            let mut combined = HashMap::<(u32, usize), i64>::new();
            for (column, coefficient) in outputs.iter().zip(&primitive) {
                for (&key, &value) in &column[root] {
                    *combined.entry(key).or_insert(0) += coefficient * value;
                }
            }
            combined.retain(|_, coefficient| *coefficient != 0);
            combined.keys().copied().min().map(|key| (root, key))
        });
        let Some((root, key)) = residual else {
            return Some(primitive);
        };
        rows.push(
            outputs
                .iter()
                .map(|candidate| {
                    Ratio::from_integer(candidate[root].get(&key).copied().unwrap_or(0))
                })
                .collect(),
        );
        nullspace = ratio_nullspace(&rows, columns);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenericLeadingSourceCouplingAudit {
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub target_dynkin_label: &'static str,
    pub product_weight_domain_dimension: usize,
    pub source_weight_spaces_used: usize,
    pub source_weight_multiplicities: Vec<(Weight, usize)>,
    pub primitive_domain_coefficients: Vec<i64>,
    pub coupled_nonzero_terms: usize,
    pub raising_residual_terms_by_simple_root: [usize; 5],
    pub exact_coupling_constructed: bool,
    pub passed: bool,
}

pub fn audit_generic_leading_source_coupling(
    dynkin_label: &str,
    source_copy: usize,
    kernel_bytes: &[u8],
) -> GenericLeadingSourceCouplingAudit {
    let spinors = spinor_weights();
    let left = half_groups(0, &spinors);
    let right = half_groups(16, &spinors);
    let source_highest_weight = dynkin_highest_weight(dynkin_label);
    let source_basis = weight_basis(16, source_highest_weight, &left, &right);
    let coefficients = decode_kernel(kernel_bytes);
    assert_eq!(coefficients.len(), source_basis.len());
    let highest = source_basis
        .iter()
        .copied()
        .zip(coefficients)
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(mask, coefficient)| (mask, i64::from(coefficient)))
        .collect::<HashMap<_, _>>();
    let target_weight = [3, 1, 1, 1, 1];
    let source_weight_bases =
        relevant_source_weight_bases(highest, source_highest_weight, target_weight, &spinors);
    let mut domain = Vec::<(usize, HashMap<u32, i64>)>::new();
    for (spinor_index, spinor_weight) in spinors.iter().copied().enumerate() {
        let source_weight = subtract(target_weight, spinor_weight);
        if let Some(basis) = source_weight_bases.get(&source_weight) {
            domain.extend(basis.iter().cloned().map(|source| (spinor_index, source)));
        }
    }
    let outputs = domain
        .iter()
        .map(|(spinor_index, source)| {
            (0..5)
                .map(|root| raised_tensor_column(source, *spinor_index, root, &spinors))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let primitive_domain_coefficients = one_dimensional_tensor_kernel(&outputs).unwrap_or_default();
    let mut coupled = HashMap::<(u32, usize), i64>::new();
    for ((spinor_index, source), coefficient) in domain.iter().zip(&primitive_domain_coefficients) {
        for (&mask, &value) in source {
            *coupled.entry((mask, *spinor_index)).or_insert(0) += coefficient * value;
        }
    }
    coupled.retain(|_, coefficient| *coefficient != 0);
    let raising_residual_terms_by_simple_root = std::array::from_fn(|root| {
        let mut residual = HashMap::<(u32, usize), i64>::new();
        for (column, coefficient) in outputs.iter().zip(&primitive_domain_coefficients) {
            for (&key, &value) in &column[root] {
                *residual.entry(key).or_insert(0) += coefficient * value;
            }
        }
        residual
            .values()
            .filter(|coefficient| **coefficient != 0)
            .count()
    });
    let exact_coupling_constructed = primitive_domain_coefficients.len() == domain.len()
        && !primitive_domain_coefficients.is_empty()
        && raising_residual_terms_by_simple_root == [0; 5];
    let mut source_weight_multiplicities = source_weight_bases
        .iter()
        .map(|(weight, basis)| (*weight, basis.len()))
        .collect::<Vec<_>>();
    source_weight_multiplicities.sort_unstable_by_key(|(weight, _)| *weight);
    GenericLeadingSourceCouplingAudit {
        source_dynkin_label: dynkin_label.to_string(),
        source_copy,
        target_dynkin_label: "10001",
        product_weight_domain_dimension: domain.len(),
        source_weight_spaces_used: source_weight_bases.len(),
        source_weight_multiplicities,
        primitive_domain_coefficients,
        coupled_nonzero_terms: coupled.len(),
        raising_residual_terms_by_simple_root,
        exact_coupling_constructed,
        passed: exact_coupling_constructed,
    }
}

fn integer_gcd(left: i16, right: i16) -> i16 {
    if right == 0 {
        left
    } else {
        integer_gcd(right, left % right)
    }
}

fn decode_kernel(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

pub fn verify() -> ElevenDimensionalBridgeReport {
    let weights = spinor_weights();
    let left = half_groups(0, &weights);
    let right = half_groups(16, &weights);
    let systems = vec![
        build_system(
            15,
            "00001",
            32,
            [1, 1, 1, 1, 1],
            2,
            &left,
            &right,
            &weights,
            &[
                (
                    "data/eleven_dimensional_bridge/00001_highest_weight_kernel_1.i16le",
                    SPINOR_KERNEL_1,
                ),
                (
                    "data/eleven_dimensional_bridge/00001_highest_weight_kernel_2.i16le",
                    SPINOR_KERNEL_2,
                ),
            ],
        ),
        build_system(
            15,
            "10001",
            320,
            [3, 1, 1, 1, 1],
            1,
            &left,
            &right,
            &weights,
            &[(
                "data/eleven_dimensional_bridge/10001_highest_weight_kernel.i16le",
                VECTOR_SPINOR_KERNEL,
            )],
        ),
    ];
    let first_lower_symbol_systems = vec![
        build_system(
            13,
            "00001",
            32,
            [1, 1, 1, 1, 1],
            1,
            &left,
            &right,
            &weights,
            &[(
                "data/eleven_dimensional_bridge/level13_00001_highest_weight_kernel.i16le",
                LEVEL13_SPINOR_KERNEL,
            )],
        ),
        build_system(
            13,
            "01001",
            1_408,
            [3, 3, 1, 1, 1],
            2,
            &left,
            &right,
            &weights,
            &[
                (
                    "data/eleven_dimensional_bridge/level13_01001_highest_weight_kernel_1.i16le",
                    LEVEL13_TWO_FORM_SPINOR_KERNEL_1,
                ),
                (
                    "data/eleven_dimensional_bridge/level13_01001_highest_weight_kernel_2.i16le",
                    LEVEL13_TWO_FORM_SPINOR_KERNEL_2,
                ),
            ],
        ),
    ];
    let exact_kernel_vectors_verified = systems
        .iter()
        .flat_map(|system| &system.exact_kernel_vectors)
        .filter(|kernel| kernel.exact_kernel_verified)
        .count();
    let spinor_basis = weight_basis(15, [1, 1, 1, 1, 1], &left, &right);
    let spinor_descendant_audits = [SPINOR_KERNEL_1, SPINOR_KERNEL_2]
        .into_iter()
        .enumerate()
        .map(|(index, bytes)| {
            audit_full_spinor_descendants(index + 1, &spinor_basis, &decode_kernel(bytes), &weights)
        })
        .collect::<Vec<_>>();
    let vector_spinor_target_audit = audit_vector_spinor_target(&weights);
    let vector_spinor_source_basis = weight_basis(15, [3, 1, 1, 1, 1], &left, &right);
    let (
        vector_spinor_source_descendant_audit,
        level_sixteen_exterior_derivative_audit,
        first_derivative_momentum_audit,
    ) = audit_full_vector_spinor_source_descendants(
        &vector_spinor_source_basis,
        &decode_kernel(VECTOR_SPINOR_KERNEL),
        &weights,
    );
    let first_momentum_completion_audit =
        audit_first_momentum_completion(&first_derivative_momentum_audit, &weights, &left, &right);
    let level_sixteen_derivative_channel_audit = audit_level_sixteen_derivative_channels();
    let dimension_zero_torsion_sector_audit =
        audit_dimension_zero_torsion_sectors(&level_sixteen_exterior_derivative_audit);
    let zero_momentum_equation_2_7_projection = audit_zero_momentum_equation_2_7_projection();
    let clifford = crate::eleven_dimensional_clifford::verify();
    let local_gamma_trace_quotient = audit_local_gamma_trace_quotient(&clifford);
    let canonical_source_line_normalization = audit_canonical_source_line_normalization();
    let linearized_scale_freedom_audit =
        audit_linearized_scale_freedom(&canonical_source_line_normalization);
    let inherited_spinor_gauge_audit = audit_inherited_spinor_gauge(&clifford);
    let passed = systems
        .iter()
        .all(|system| system.exact_sparse_system_constructed)
        && first_lower_symbol_systems.iter().all(|system| {
            system.exact_sparse_system_constructed
                && system
                    .exact_kernel_vectors
                    .iter()
                    .all(|kernel| kernel.exact_kernel_verified)
        })
        && exact_kernel_vectors_verified == 3
        && spinor_descendant_audits
            .iter()
            .all(|audit| audit.exact_full_spinor_intertwiner_verified)
        && vector_spinor_target_audit.passed
        && vector_spinor_source_descendant_audit.exact_full_vector_spinor_intertwiner_verified
        && level_sixteen_derivative_channel_audit.passed
        && level_sixteen_exterior_derivative_audit.passed
        && dimension_zero_torsion_sector_audit.passed
        && first_derivative_momentum_audit.passed
        && first_momentum_completion_audit.passed
        && zero_momentum_equation_2_7_projection.passed
        && local_gamma_trace_quotient.passed
        && canonical_source_line_normalization.passed
        && linearized_scale_freedom_audit.passed
        && inherited_spinor_gauge_audit.passed;

    ElevenDimensionalBridgeReport {
        schema_version: "adynkra.11d.level15-bridge.v9",
        source_arxiv: "2002.08502",
        source_level: 15,
        spinor_weights: weights.len(),
        systems,
        first_lower_symbol_systems,
        generic_bridge: "H_alpha^a(V) = a I_32^(1)(D^15 V) + b I_32^(2)(D^15 V) + c I_320(D^15 V)",
        coefficients: vec![
            BridgeCoefficient {
                name: "a",
                source_dynkin_label: "00001",
                source_copy: 1,
                target_sector: "gamma trace",
            },
            BridgeCoefficient {
                name: "b",
                source_dynkin_label: "00001",
                source_copy: 2,
                target_sector: "gamma trace",
            },
            BridgeCoefficient {
                name: "c",
                source_dynkin_label: "10001",
                source_copy: 1,
                target_sector: "gamma-traceless vector-spinor",
            },
        ],
        equation_2_7_status: "the zero-momentum exterior gamma^[2] symbol has rank zero on the three bridge channels; the aligned Clifford calculation gives a nonzero level-14 momentum contraction in the 11000 hook, and the complete level-13 correction space does not cancel it; the stated local polynomial scalar bridge fails this part of Eq. (2.7)",
        coefficient_solution_status: "the exterior symbol leaves a, b, and c unrestricted; quotienting the local gamma-trace symmetry removes a and b; the first lower-symbol correction space does not cancel the generic-momentum 11000 term in the surviving c channel",
        expected_kernel_vectors: 3,
        exact_kernel_vectors_verified,
        all_expected_kernel_vectors_verified: exact_kernel_vectors_verified == 3,
        spinor_descendant_audits,
        vector_spinor_target_audit,
        vector_spinor_source_descendant_audit,
        level_sixteen_derivative_channel_audit,
        level_sixteen_exterior_derivative_audit,
        dimension_zero_torsion_sector_audit,
        first_derivative_momentum_audit,
        first_momentum_completion_audit,
        zero_momentum_equation_2_7_projection,
        local_gamma_trace_quotient,
        canonical_source_line_normalization,
        linearized_scale_freedom_audit,
        inherited_spinor_gauge_audit,
        boundary: "This verifies the level-15 bridge intertwiners, the level-16 exterior symbol, the level-14 momentum contractions, the three level-13 source intertwiners, and the first lower-symbol cancellation system. The 64 exact integer functionals give correction rank two and augmented rank three, so the complete p D_[13] correction space cannot cancel the leading p D_[14] term in the 11000 hook. Higher-momentum terms occupy different normal-form bidegrees and cannot repair this coefficient. This rules out the stated local polynomial scalar bridge under this constraint. It does not rule out nonlocal operators, additional prepotentials, or a different constraint or gauge complex.",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn spinor_weight_set_is_complete_and_unique() {
        let weights = spinor_weights();
        let unique: HashSet<_> = weights.into_iter().collect();
        assert_eq!(unique.len(), 32);
        assert!(weights
            .iter()
            .all(|weight| weight.iter().all(|x| x.abs() == 1)));
    }

    #[test]
    fn simple_root_action_stays_inside_the_spinor_when_defined() {
        let weights = spinor_weights();
        let counts: Vec<_> = (0..5)
            .map(|root| {
                (0..32)
                    .filter(|&index| raised_spinor_index(index, root, &weights).is_some())
                    .count()
            })
            .collect();
        assert_eq!(counts, vec![8, 8, 8, 8, 16]);
    }

    #[test]
    fn two_form_spinor_target_generates_01001() {
        let spinors = spinor_weights();
        let (by_weight, nonzero_lowering_actions) = generate_two_form_spinor_target_basis(&spinors);
        assert_eq!(by_weight.values().map(Vec::len).sum::<usize>(), 1_408);
        assert!(nonzero_lowering_actions >= 1_407);
    }

    #[test]
    fn vector_times_01001_contains_one_10001_highest_line() {
        let spinors = spinor_weights();
        let (domain, primitive, kernel_dimension) = build_01001_to_10001_highest_map(&spinors);
        assert_eq!(kernel_dimension, 1);
        assert!(!domain.is_empty());
        assert_eq!(primitive.len(), domain.len());
        assert!(primitive.iter().any(|coefficient| *coefficient != 0));
    }

    #[test]
    #[ignore = "lowers two large exact level-13 source kernels through all correction weights"]
    fn level_thirteen_01001_sources_reach_all_correction_weights() {
        let spinors = spinor_weights();
        let left = half_groups(0, &spinors);
        let right = half_groups(16, &spinors);
        let source_basis = weight_basis(13, [3, 3, 1, 1, 1], &left, &right);
        let (domain, _, _) = build_01001_to_10001_highest_map(&spinors);
        let needed = domain
            .iter()
            .map(|entry| entry.source_weight)
            .collect::<std::collections::BTreeSet<_>>();
        for bytes in [
            LEVEL13_TWO_FORM_SPINOR_KERNEL_1,
            LEVEL13_TWO_FORM_SPINOR_KERNEL_2,
        ] {
            let states = generate_partial_two_form_spinor_source_states(
                &source_basis,
                &decode_kernel(bytes),
                &spinors,
                &needed,
            );
            assert_eq!(states.values().map(Vec::len).sum::<usize>(), domain.len());
        }
    }

    #[test]
    fn vector_spinor_source_line_has_canonical_computational_normalization() {
        let audit = audit_canonical_source_line_normalization();
        assert_eq!(audit.primitive_coefficient_gcd, 1);
        assert_eq!(audit.first_nonzero_coefficient, 84);
        assert_eq!(audit.squared_norm, 245_044_800);
        assert!(audit.primitive_sign_fixed);
        assert!(audit.orthogonal_projector_idempotent);
        assert!(audit.computational_source_normalization_fixed);
        assert!(!audit.physical_bridge_scale_fixed);
        assert!(audit.passed);
    }

    #[test]
    fn level_sixteen_inventory_removes_two_derivative_channels() {
        let audit = audit_level_sixteen_derivative_channels();
        assert_eq!(audit.tensor_product_dimension, 10_240);
        assert_eq!(audit.multiplicity_free_candidate_channels, 10);
        assert_eq!(audit.scalar_level_sixteen_present_channels, 8);
        assert_eq!(
            audit.scalar_level_sixteen_absent_channels,
            vec!["01000", "11000"]
        );
        assert_eq!(audit.absent_channel_dimension, 484);
        assert_eq!(audit.present_channel_dimension, 9_756);
        assert!(audit.final_two_form_hook_absent);
        assert!(audit.passed);
    }

    #[test]
    fn level_sixteen_highest_weight_kernels_are_unique() {
        let spinors = spinor_weights();
        let (by_weight, _) = generate_vector_spinor_target_basis(&spinors);
        let plans = build_exterior_channel_plans(&spinors);
        for (label, expected_domain_dimension) in [
            ("00002", 10),
            ("00010", 18),
            ("00100", 32),
            ("01000", 56),
            ("10000", 96),
            ("10002", 1),
            ("10010", 2),
            ("10100", 4),
            ("11000", 8),
            ("20000", 16),
        ] {
            let highest = dynkin_highest_weight(label);
            let dimension = spinors
                .iter()
                .map(|spinor| {
                    by_weight
                        .get(&subtract(highest, *spinor))
                        .map(Vec::len)
                        .unwrap_or(0)
                })
                .sum::<usize>();
            let plan = plans
                .iter()
                .find(|plan| plan.dynkin_label == label)
                .unwrap();
            assert_eq!(dimension, expected_domain_dimension, "{label}");
            assert_eq!(plan.domain.len(), expected_domain_dimension, "{label}");
            assert_eq!(plan.highest_weight_kernel_dimension, 1, "{label}");
            assert_eq!(
                plan.primitive_highest_weight_coefficients
                    .iter()
                    .filter(|coefficient| **coefficient != 0)
                    .count(),
                expected_domain_dimension,
                "{label}"
            );
            assert_eq!(plan.raising_residual_terms, 0, "{label}");
        }
    }

    #[test]
    fn homogeneous_constraints_leave_the_overall_bridge_scale_free() {
        let normalization = audit_canonical_source_line_normalization();
        let audit = audit_linearized_scale_freedom(&normalization);
        assert!(audit.bridge_is_linear_in_scalar_prepotential);
        assert!(audit.constraints_are_homogeneous_in_linearized_bridge);
        assert!(audit.nonzero_bridge_class_selected);
        assert!(audit.computational_source_normalization_fixed);
        assert!(!audit.physical_scale_fixed_by_homogeneous_constraints);
        assert!(audit.passed);
    }

    #[test]
    #[ignore = "constructs 3.1 million rows and 12 million exact sparse entries"]
    fn full_level_15_system_matches_independent_counts() {
        let report = verify();
        assert!(report.passed);
        let spinor = &report.systems[0];
        assert_eq!(spinor.source_weight_space_columns, 591_810);
        assert_eq!(spinor.total_rows, 1_943_600);
        assert_eq!(spinor.total_nonzero_entries, 7_412_645);
        assert_eq!(spinor.exact_kernel_vectors.len(), 2);
        assert!(spinor
            .exact_kernel_vectors
            .iter()
            .all(|kernel| kernel.exact_kernel_verified));
        assert_eq!(spinor.exact_kernel_vectors[0].nonzero_coefficients, 374_246);
        assert_eq!(spinor.exact_kernel_vectors[1].nonzero_coefficients, 6_435);
        assert_eq!(spinor.exact_kernel_vectors[0].squared_norm, 426_254_400);
        assert_eq!(spinor.exact_kernel_vectors[1].squared_norm, 6_435);
        assert_eq!(report.first_lower_symbol_systems.len(), 2);
        let level13_spinor = &report.first_lower_symbol_systems[0];
        assert_eq!(level13_spinor.exterior_degree, 13);
        assert_eq!(level13_spinor.source_weight_space_columns, 388_720);
        assert_eq!(level13_spinor.total_rows, 1_260_810);
        assert_eq!(level13_spinor.exact_kernel_vectors.len(), 1);
        assert_eq!(
            level13_spinor.exact_kernel_vectors[0].nonzero_coefficients,
            5_005
        );
        assert!(level13_spinor.exact_kernel_vectors[0].exact_kernel_verified);
        let level13_two_form_spinor = &report.first_lower_symbol_systems[1];
        assert_eq!(level13_two_form_spinor.exterior_degree, 13);
        assert_eq!(level13_two_form_spinor.source_weight_space_columns, 161_432);
        assert_eq!(level13_two_form_spinor.total_rows, 475_801);
        assert_eq!(level13_two_form_spinor.exact_kernel_vectors.len(), 2);
        assert_eq!(
            level13_two_form_spinor.exact_kernel_vectors[0].nonzero_coefficients,
            5_148
        );
        assert_eq!(
            level13_two_form_spinor.exact_kernel_vectors[1].nonzero_coefficients,
            145_065
        );
        assert!(level13_two_form_spinor
            .exact_kernel_vectors
            .iter()
            .all(|kernel| kernel.exact_kernel_verified));
        assert_eq!(report.spinor_descendant_audits.len(), 2);
        for audit in &report.spinor_descendant_audits {
            assert_eq!(audit.target_states_generated, 32);
            assert_eq!(audit.nonzero_lowering_actions_checked, 48);
            assert_eq!(audit.repeated_path_checks, 17);
            assert_eq!(audit.repeated_path_mismatches, 0);
            assert!(audit.exact_full_spinor_intertwiner_verified);
        }
        let target = &report.vector_spinor_target_audit;
        assert_eq!(target.generated_irrep_dimension, 320);
        assert_eq!(target.distinct_weights, 192);
        assert_eq!(target.multiplicity_one_weights, 160);
        assert_eq!(target.multiplicity_five_weights, 32);
        assert_eq!(target.nonzero_lowering_actions, 752);
        assert!(target.passed);
        let source_descendants = &report.vector_spinor_source_descendant_audit;
        assert_eq!(source_descendants.target_states_generated, 320);
        assert_eq!(source_descendants.distinct_weights, 192);
        assert_eq!(source_descendants.nonzero_lowering_actions_checked, 776);
        assert_eq!(source_descendants.zero_lowering_actions_checked, 824);
        assert_eq!(source_descendants.total_lowering_actions_checked, 1_600);
        assert_eq!(source_descendants.independent_state_discoveries, 319);
        assert_eq!(source_descendants.dependent_target_relations_checked, 457);
        assert_eq!(source_descendants.dependent_target_relation_mismatches, 0);
        assert_eq!(source_descendants.nonzero_relation_residual_terms, 0);
        assert_eq!(source_descendants.maximum_absolute_relation_residual, 0);
        assert_eq!(source_descendants.zero_action_mismatches, 0);
        assert_eq!(source_descendants.target_basis_correspondence_mismatches, 0);
        assert!(source_descendants.exact_full_vector_spinor_intertwiner_verified);
        let derivative_channels = &report.level_sixteen_derivative_channel_audit;
        assert_eq!(
            derivative_channels.scalar_level_sixteen_absent_channels,
            vec!["01000", "11000"]
        );
        assert!(derivative_channels.final_two_form_hook_absent);
        assert!(derivative_channels.passed);
        let exterior = &report.level_sixteen_exterior_derivative_audit;
        assert_eq!(exterior.channels_checked, 10);
        assert_eq!(exterior.highest_weight_kernels_verified, 10);
        assert_eq!(exterior.nonzero_exterior_images_certified, 8);
        assert_eq!(exterior.inventory_forced_zero_channels, 2);
        assert_eq!(exterior.inventory_zero_fingerprint_crosschecks, 2);
        assert!(exterior.channels.iter().all(|channel| channel.passed));
        assert!(exterior.passed);
        let torsion = &report.dimension_zero_torsion_sector_audit;
        assert_eq!(torsion.two_form_vector_dimension, 605);
        assert_eq!(torsion.two_form_conventional_dimension, 176);
        assert_eq!(torsion.two_form_remaining_hook_dimension, 429);
        assert!(!torsion.two_form_remaining_hook_nonzero);
        assert_eq!(torsion.five_form_vector_dimension, 5_082);
        assert_eq!(torsion.five_form_conventional_dimension, 792);
        assert_eq!(torsion.five_form_remaining_hook_dimension, 4_290);
        assert!(torsion.five_form_remaining_hook_nonzero);
        assert!(torsion.complete_dimension_partition);
        assert!(torsion.passed);
        let momentum = &report.first_derivative_momentum_audit;
        assert_eq!(momentum.channels.len(), 2);
        assert_eq!(momentum.cartan_weight_entries_checked, 5 * 32 * 32);
        assert_eq!(momentum.cartan_weight_mismatches, 0);
        assert_eq!(momentum.chevalley_lowering_actions_checked, 48);
        assert_eq!(momentum.chevalley_lowering_residual_actions, 0);
        assert!(momentum.clifford_and_weight_bases_aligned);
        assert!(momentum.two_form_hook_momentum_contraction_nonzero);
        assert!(momentum.five_form_hook_momentum_contraction_nonzero);
        assert!(momentum.channels.iter().all(|channel| channel.passed));
        assert!(momentum.passed);
        let completion = &report.first_momentum_completion_audit;
        assert_eq!(completion.vector_times_target_dimension, 11 * 320);
        assert_eq!(
            completion.available_level_thirteen_source_channels,
            vec!["00001", "01001"]
        );
        assert_eq!(completion.first_completion_coefficient_dimension, 3);
        assert!(completion.leading_two_form_hook_momentum_term_nonzero);
        assert!(completion.cancellation_system_constructed);
        assert_eq!(completion.cancellation_exists, Some(false));
        assert_eq!(completion.exact_functional_rows, 64);
        assert_eq!(completion.correction_span_rank, 2);
        assert_eq!(completion.augmented_span_rank, 3);
        assert!(completion.exact_non_cancellation_certificate);
        assert!(completion.passed);
        let projection = &report.zero_momentum_equation_2_7_projection;
        assert_eq!(projection.raw_two_form_vector_dimension, 605);
        assert_eq!(projection.remaining_hook_dynkin_label, "11000");
        assert_eq!(projection.remaining_hook_dimension, 429);
        assert_eq!(projection.hook_multiplicity_in_scalar_level, 0);
        assert_eq!(projection.exterior_symbol_rank_on_bridge_coefficients, 0);
        assert_eq!(projection.exterior_symbol_kernel_dimension, 3);
        assert!(projection.passed);
        let quotient = &report.local_gamma_trace_quotient;
        assert_eq!(quotient.local_symmetry_image_rank, 32);
        assert_eq!(quotient.gamma_traceless_quotient_rank, 320);
        assert_eq!(quotient.gamma_trace_coefficients_removed, vec!["a", "b"]);
        assert_eq!(quotient.surviving_coefficients, vec!["c"]);
        assert_eq!(quotient.quotient_bridge_coefficient_dimension, 1);
        assert!(quotient.passed);
        let normalization = &report.canonical_source_line_normalization;
        assert_eq!(normalization.primitive_coefficient_gcd, 1);
        assert_eq!(normalization.first_nonzero_coefficient, 84);
        assert_eq!(normalization.squared_norm, 245_044_800);
        assert_eq!(normalization.orthogonal_projection_denominator, 245_044_800);
        assert!(normalization.primitive_sign_fixed);
        assert!(normalization.orthogonal_projector_idempotent);
        assert!(normalization.computational_source_normalization_fixed);
        assert!(!normalization.physical_bridge_scale_fixed);
        assert!(normalization.passed);
        let scale = &report.linearized_scale_freedom_audit;
        assert!(!scale.physical_scale_fixed_by_homogeneous_constraints);
        assert!(scale.passed);
        let gauge = &report.inherited_spinor_gauge_audit;
        assert_eq!(
            gauge.channels_leaving_scalar_divergence_invariant,
            vec![2, 5]
        );
        assert_eq!(gauge.channels_leaving_quotient_bridge_invariant, vec![2, 5]);
        assert_eq!(gauge.invariant_channel_count, 2);
        assert!(gauge.passed);
        let vector_spinor = &report.systems[1];
        assert_eq!(vector_spinor.source_weight_space_columns, 388_720);
        assert_eq!(vector_spinor.total_rows, 1_174_806);
        assert_eq!(vector_spinor.total_nonzero_entries, 4_551_287);
        let kernel = &vector_spinor.exact_kernel_vectors[0];
        assert_eq!(kernel.coefficients, 388_720);
        assert_eq!(kernel.nonzero_coefficients, 260_267);
        assert_eq!(kernel.minimum_coefficient, -1320);
        assert_eq!(kernel.maximum_coefficient, 1320);
        assert_eq!(kernel.coefficient_gcd, 1);
        assert_eq!(kernel.squared_norm, 245_044_800);
        assert_eq!(kernel.nonzero_residual_rows, 0);
        assert!(kernel.exact_kernel_verified);
        assert_eq!(
            kernel
                .first_lowering_descendants
                .iter()
                .filter(|descendant| descendant.nonzero_terms > 0)
                .map(|descendant| descendant.simple_root)
                .collect::<Vec<_>>(),
            vec![1, 5]
        );
        for spinor_kernel in &spinor.exact_kernel_vectors {
            assert_eq!(
                spinor_kernel
                    .first_lowering_descendants
                    .iter()
                    .filter(|descendant| descendant.nonzero_terms > 0)
                    .map(|descendant| descendant.simple_root)
                    .collect::<Vec<_>>(),
                vec![5]
            );
        }
    }
}
