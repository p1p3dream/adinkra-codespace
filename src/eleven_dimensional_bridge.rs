//! Exact sparse highest-weight systems for the level-15 11D bridge.
//!
//! The scalar 11D superfield has level-15 space `exterior^15 S`, where `S`
//! is the 32-dimensional spinor of B5.  The published level inventory contains
//! two copies of `(00001)` and one copy of `(10001)`.  A highest-weight vector
//! of either type is an integer vector in the corresponding weight space that
//! is killed by all five simple-root raising operators.  This module builds
//! those sparse integer systems directly from the spinor weights.

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
    pub boundary: &'static str,
    pub passed: bool,
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
    let passed = systems
        .iter()
        .all(|system| system.exact_sparse_system_constructed)
        && exact_kernel_vectors_verified == 3
        && spinor_descendant_audits
            .iter()
            .all(|audit| audit.exact_full_spinor_intertwiner_verified);

    ElevenDimensionalBridgeReport {
        schema_version: "adynkra.11d.level15-bridge.v1",
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
        equation_2_7_status: "all three highest-weight kernel vectors are explicit and exact; their covariant descendants are still required before substituting the bridge into the torsion constraints",
        coefficient_solution_status: "not solved",
        expected_kernel_vectors: 3,
        exact_kernel_vectors_verified,
        all_expected_kernel_vectors_verified: exact_kernel_vectors_verified == 3,
        spinor_descendant_audits,
        boundary: "This constructs the exact sparse equations and verifies all three highest-weight kernel vectors over every raising row. Their covariant descendants remain to be constructed.",
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
        assert!(
            weights
                .iter()
                .all(|weight| weight.iter().all(|x| x.abs() == 1))
        );
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
    #[ignore = "constructs 3.1 million rows and 12 million exact sparse entries"]
    fn full_level_15_system_matches_independent_counts() {
        let report = verify();
        assert!(report.passed);
        let spinor = &report.systems[0];
        assert_eq!(spinor.source_weight_space_columns, 591_810);
        assert_eq!(spinor.total_rows, 1_943_600);
        assert_eq!(spinor.total_nonzero_entries, 7_412_645);
        assert_eq!(spinor.exact_kernel_vectors.len(), 2);
        assert!(
            spinor
                .exact_kernel_vectors
                .iter()
                .all(|kernel| kernel.exact_kernel_verified)
        );
        assert_eq!(spinor.exact_kernel_vectors[0].nonzero_coefficients, 374_246);
        assert_eq!(spinor.exact_kernel_vectors[1].nonzero_coefficients, 6_435);
        assert_eq!(report.spinor_descendant_audits.len(), 2);
        for audit in &report.spinor_descendant_audits {
            assert_eq!(audit.target_states_generated, 32);
            assert_eq!(audit.nonzero_lowering_actions_checked, 48);
            assert_eq!(audit.repeated_path_checks, 17);
            assert_eq!(audit.repeated_path_mismatches, 0);
            assert!(audit.exact_full_spinor_intertwiner_verified);
        }
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
