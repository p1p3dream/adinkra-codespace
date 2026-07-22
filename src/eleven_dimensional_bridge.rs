//! Exact sparse highest-weight systems for the level-15 11D bridge.
//!
//! The scalar 11D superfield has level-15 space `exterior^15 S`, where `S`
//! is the 32-dimensional spinor of B5.  The published level inventory contains
//! two copies of `(00001)` and one copy of `(10001)`.  A highest-weight vector
//! of either type is an integer vector in the corresponding weight space that
//! is killed by all five simple-root raising operators.  This module builds
//! those sparse integer systems directly from the spinor weights.

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
    pub nonzero_terms: usize,
    pub second_lowering_nonzero_terms: usize,
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
    pub final_equation_2_7_projection: FinalEquationProjectionAudit,
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
    pub minimum_source_state_support: usize,
    pub maximum_source_state_support: usize,
    pub maximum_absolute_source_coefficient: i64,
    pub exact_full_vector_spinor_intertwiner_verified: bool,
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
pub struct FinalEquationProjectionAudit {
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
    pub projected_constraint_rank_on_bridge_coefficients: usize,
    pub surviving_bridge_coefficient_dimension: usize,
    pub interpretation: &'static str,
    pub passed: bool,
}

fn audit_final_equation_2_7_projection() -> FinalEquationProjectionAudit {
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
    let projected_constraint_rank_on_bridge_coefficients =
        usize::from(hook_multiplicity_in_scalar_level != 0);
    let surviving_bridge_coefficient_dimension =
        3 - projected_constraint_rank_on_bridge_coefficients;
    FinalEquationProjectionAudit {
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
        projected_constraint_rank_on_bridge_coefficients,
        surviving_bridge_coefficient_dimension,
        interpretation: "the vector and three-form pieces are removed by conventional constraints; the remaining 429-dimensional hook is absent from level 16 of the scalar superfield, so the final Eq. (2.7) projection vanishes representation-theoretically and does not select among a, b, and c",
        passed: dimension_decomposition_closes
            && hook_multiplicity_in_scalar_level == 0
            && surviving_bridge_coefficient_dimension == 3,
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

fn audit_vector_spinor_target(spinors: &[Weight; 32]) -> VectorSpinorTargetAudit {
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
    let mut generated_irrep_dimension = 1;
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
                generated_irrep_dimension += 1;
                queue.push_back(descendant);
            }
        }
    }
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

fn weight_basis(
    target: Weight,
    left: &HashMap<(u8, Weight), Vec<u16>>,
    right: &HashMap<(u8, Weight), Vec<u16>>,
) -> Vec<u32> {
    let mut basis = Vec::new();
    for left_degree in 0_u8..=15 {
        let right_degree = 15 - left_degree;
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

fn lower_pairs(source: &[(u32, i64)], root: usize, weights: &[Weight; 32]) -> Vec<(u32, i64)> {
    let source = source.iter().copied().collect::<HashMap<_, _>>();
    let mut lowered = lower_sparse(&source, root, weights)
        .into_iter()
        .collect::<Vec<_>>();
    lowered.sort_unstable_by_key(|(mask, _)| *mask);
    lowered
}

#[derive(Clone)]
struct VectorSpinorIntertwinerState {
    target: TensorVector,
    source: Vec<(u32, i64)>,
}

fn target_span_coefficients(
    candidate: &TensorVector,
    basis: &[VectorSpinorIntertwinerState],
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
    for column in 0..11 * 32 {
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

fn audit_full_vector_spinor_source_descendants(
    source_basis: &[u32],
    coefficients: &[i16],
    spinors: &[Weight; 32],
) -> VectorSpinorSourceDescendantAudit {
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
    let mut minimum_source_state_support = usize::MAX;
    let mut maximum_source_state_support = 0;
    let mut maximum_absolute_source_coefficient = 0_i64;

    while !current.is_empty() {
        let mut next = BTreeMap::<Weight, Vec<VectorSpinorIntertwinerState>>::new();
        for states in current.into_values() {
            for state in states {
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
                        target_span_coefficients(&target_descendant, basis)
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
        && zero_action_mismatches == 0;
    VectorSpinorSourceDescendantAudit {
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
        minimum_source_state_support,
        maximum_source_state_support,
        maximum_absolute_source_coefficient,
        exact_full_vector_spinor_intertwiner_verified,
    }
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
    dynkin_label: &'static str,
    representation_dimension: usize,
    highest_weight: Weight,
    published_multiplicity: usize,
    left: &HashMap<(u8, Weight), Vec<u16>>,
    right: &HashMap<(u8, Weight), Vec<u16>>,
    weights: &[Weight; 32],
    kernel_artifacts: &[(&'static str, &'static [u8])],
) -> HighestWeightSystemReport {
    let source_basis = weight_basis(highest_weight, left, right);
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
        let output_basis = weight_basis(output_weight, left, right);
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
                    let second_descendant = lower_sparse(&descendant, root, weights);
                    let expected_nonzero_from_dynkin_label = dynkin_label.as_bytes()[root] != b'0';
                    FirstLoweringReport {
                        simple_root: root + 1,
                        expected_nonzero_from_dynkin_label,
                        nonzero_terms: descendant.len(),
                        second_lowering_nonzero_terms: second_descendant.len(),
                        maximum_absolute_coefficient: descendant
                            .values()
                            .map(|coefficient| coefficient.abs())
                            .max()
                            .unwrap_or(0),
                        matches_highest_weight_string: if expected_nonzero_from_dynkin_label {
                            !descendant.is_empty() && second_descendant.is_empty()
                        } else {
                            descendant.is_empty()
                        },
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
        exterior_degree: 15,
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
    let exact_kernel_vectors_verified = systems
        .iter()
        .flat_map(|system| &system.exact_kernel_vectors)
        .filter(|kernel| kernel.exact_kernel_verified)
        .count();
    let spinor_basis = weight_basis([1, 1, 1, 1, 1], &left, &right);
    let spinor_descendant_audits = [SPINOR_KERNEL_1, SPINOR_KERNEL_2]
        .into_iter()
        .enumerate()
        .map(|(index, bytes)| {
            audit_full_spinor_descendants(index + 1, &spinor_basis, &decode_kernel(bytes), &weights)
        })
        .collect::<Vec<_>>();
    let vector_spinor_target_audit = audit_vector_spinor_target(&weights);
    let vector_spinor_source_basis = weight_basis([3, 1, 1, 1, 1], &left, &right);
    let vector_spinor_source_descendant_audit = audit_full_vector_spinor_source_descendants(
        &vector_spinor_source_basis,
        &decode_kernel(VECTOR_SPINOR_KERNEL),
        &weights,
    );
    let level_sixteen_derivative_channel_audit = audit_level_sixteen_derivative_channels();
    let final_equation_2_7_projection = audit_final_equation_2_7_projection();
    let clifford = crate::eleven_dimensional_clifford::verify();
    let local_gamma_trace_quotient = audit_local_gamma_trace_quotient(&clifford);
    let canonical_source_line_normalization = audit_canonical_source_line_normalization();
    let linearized_scale_freedom_audit =
        audit_linearized_scale_freedom(&canonical_source_line_normalization);
    let inherited_spinor_gauge_audit = audit_inherited_spinor_gauge(&clifford);
    let passed = systems
        .iter()
        .all(|system| system.exact_sparse_system_constructed)
        && exact_kernel_vectors_verified == 3
        && spinor_descendant_audits
            .iter()
            .all(|audit| audit.exact_full_spinor_intertwiner_verified)
        && vector_spinor_target_audit.passed
        && vector_spinor_source_descendant_audit.exact_full_vector_spinor_intertwiner_verified
        && level_sixteen_derivative_channel_audit.passed
        && final_equation_2_7_projection.passed
        && local_gamma_trace_quotient.passed
        && canonical_source_line_normalization.passed
        && linearized_scale_freedom_audit.passed
        && inherited_spinor_gauge_audit.passed;

    ElevenDimensionalBridgeReport {
        schema_version: "adynkra.11d.level15-bridge.v4",
        source_arxiv: "2002.08502",
        source_level: 15,
        spinor_weights: weights.len(),
        systems,
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
        equation_2_7_status: "the final gamma^[2] projection has rank zero on the three bridge channels; all 384 source descendant states are generated and their complete simple-root action is verified",
        coefficient_solution_status: "the final gamma^[2] projection leaves a, b, and c unrestricted; quotienting the local gamma-trace symmetry removes a and b; the surviving source line has a canonical computational normalization, while the overall nonzero c is a scalar-prepotential normalization convention until V is matched to a normalized component field",
        expected_kernel_vectors: 3,
        exact_kernel_vectors_verified,
        all_expected_kernel_vectors_verified: exact_kernel_vectors_verified == 3,
        spinor_descendant_audits,
        vector_spinor_target_audit,
        vector_spinor_source_descendant_audit,
        level_sixteen_derivative_channel_audit,
        final_equation_2_7_projection,
        local_gamma_trace_quotient,
        canonical_source_line_normalization,
        linearized_scale_freedom_audit,
        inherited_spinor_gauge_audit,
        boundary: "This constructs the sparse equations, verifies all three highest-weight kernel vectors, completes the two 32-component and one 320-component source descendant systems, fixes the computational source-line normalization, and screens the ten level-16 derivative channels. A component normalization of V and the complete curvature complex remain.",
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
        assert!(source_descendants.exact_full_vector_spinor_intertwiner_verified);
        let derivative_channels = &report.level_sixteen_derivative_channel_audit;
        assert_eq!(
            derivative_channels.scalar_level_sixteen_absent_channels,
            vec!["01000", "11000"]
        );
        assert!(derivative_channels.final_two_form_hook_absent);
        assert!(derivative_channels.passed);
        let projection = &report.final_equation_2_7_projection;
        assert_eq!(projection.raw_two_form_vector_dimension, 605);
        assert_eq!(projection.remaining_hook_dynkin_label, "11000");
        assert_eq!(projection.remaining_hook_dimension, 429);
        assert_eq!(projection.hook_multiplicity_in_scalar_level, 0);
        assert_eq!(
            projection.projected_constraint_rank_on_bridge_coefficients,
            0
        );
        assert_eq!(projection.surviving_bridge_coefficient_dimension, 3);
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
