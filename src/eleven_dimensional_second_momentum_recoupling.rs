//! Exact recoupling certificate for the second-momentum `(10001)` sector.
//!
//! The certificate uses the repository's exact Cartesian B5 Clifford system
//! to distinguish the scalar trace in `Sym^2(V)` from the symmetric-traceless
//! momentum tensor.  Both give an equivariant copy of the gamma-traceless
//! vector-spinor `(10001)`.  Exhaustive identities on all 352 ambient
//! vector-spinor coordinate columns prove that each image has dimension 320
//! and that their direct sum has dimension 640.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_clifford::{GaussianRational, Matrix};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const VECTOR_SPINOR_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const SYMMETRIC_TENSOR_DIMENSION: usize = VECTOR_DIMENSION * (VECTOR_DIMENSION + 1) / 2;
const TARGET_DIMENSION: usize = 320;
const EXPECTED_DOMAIN_DIMENSION: usize = SYMMETRIC_TENSOR_DIMENSION * TARGET_DIMENSION;

const LEVEL12_10002_KERNEL_1: &[u8] = include_bytes!(
    "../data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_1.i16le"
);
const LEVEL12_10002_KERNEL_2: &[u8] = include_bytes!(
    "../data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_2.i16le"
);
const LEVEL12_10002_KERNEL_BYTES: usize = 113_516;
const LEVEL12_10002_KERNEL_SHA256: [&str; 2] = [
    "c3eb687d6d868cd08fcd90a0c741815681b3c68ca4cf3157ddd929df0aa42e28",
    "ec51e06970fcff1b7b719d7ee5c3d9a69775fdf6558334c400d1540dbd81c7a1",
];
const EXPECTED_CERTIFICATE_SHA256: &str =
    "d883b57a3151c02e25e7937400b1dc75c5fcdabd4899969aef8650ce7f37044d";

type SparseSpinor = BTreeMap<usize, GaussianRational>;
type VectorSpinor = Vec<SparseSpinor>;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RecouplingChannel {
    pub dynkin_label: &'static str,
    pub dimension: usize,
    pub multiplicity: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PinnedLevel12Fixture {
    pub artifact: &'static str,
    pub bytes: usize,
    pub sha256: String,
    pub hash_matches: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SecondMomentumRecouplingReport {
    pub schema_version: &'static str,
    pub basis_convention: &'static str,
    pub tensor_product: &'static str,
    pub channels: Vec<RecouplingChannel>,
    pub symmetric_tensor_dimension: usize,
    pub target_dimension: usize,
    pub tensor_product_dimension: usize,
    pub decomposed_dimension: usize,
    pub decomposition_dimension_residual: isize,
    pub ambient_vector_spinor_dimension: usize,
    pub gamma_trace_rank: usize,
    pub gamma_traceless_projector_rank: usize,
    pub gamma_traceless_projector_polynomial_entries_checked: usize,
    pub gamma_traceless_projector_polynomial_residual_entries: usize,
    pub clifford_entries_checked: usize,
    pub clifford_residual_entries: usize,
    pub trace_path_image_rank: usize,
    pub symmetric_traceless_path_image_rank: usize,
    pub combined_10001_image_rank: usize,
    pub multiplicity_space_rank: usize,
    pub extraction_matrix: [[i64; 2]; 2],
    pub extraction_determinant: i64,
    pub trace_path_target_gamma_entries_checked: usize,
    pub trace_path_target_gamma_residual_entries: usize,
    pub symmetric_traceless_target_gamma_entries_checked: usize,
    pub symmetric_traceless_target_gamma_residual_entries: usize,
    pub symmetric_traceless_momentum_trace_entries_checked: usize,
    pub symmetric_traceless_momentum_trace_residual_entries: usize,
    pub trace_extraction_entries_checked: usize,
    pub trace_extraction_residual_entries: usize,
    pub mixed_extraction_entries_checked: usize,
    pub mixed_extraction_residual_entries: usize,
    pub symmetric_traceless_embedding_coefficients: [i64; 3],
    pub pinned_level12_10002_fixtures: Vec<PinnedLevel12Fixture>,
    pub fixture_hashes_complete: bool,
    pub representation_inventory_sha256: String,
    pub representation_inventory_matches: bool,
    pub representation_decomposition_complete: bool,
    pub component_source_target_maps_complete: bool,
    pub certificate_sha256: String,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Serialize)]
struct CertificateHashPayload<'a> {
    channels: &'a [RecouplingChannel],
    symmetric_tensor_dimension: usize,
    target_dimension: usize,
    tensor_product_dimension: usize,
    decomposed_dimension: usize,
    gamma_trace_rank: usize,
    gamma_traceless_projector_rank: usize,
    clifford_residual_entries: usize,
    trace_path_image_rank: usize,
    symmetric_traceless_path_image_rank: usize,
    combined_10001_image_rank: usize,
    multiplicity_space_rank: usize,
    extraction_matrix: [[i64; 2]; 2],
    extraction_determinant: i64,
    trace_path_target_gamma_residual_entries: usize,
    symmetric_traceless_target_gamma_residual_entries: usize,
    symmetric_traceless_momentum_trace_residual_entries: usize,
    trace_extraction_residual_entries: usize,
    mixed_extraction_residual_entries: usize,
    symmetric_traceless_embedding_coefficients: [i64; 3],
    fixture_hashes: Vec<&'a str>,
    representation_inventory_sha256: &'a str,
    representation_inventory_matches: bool,
    representation_decomposition_complete: bool,
    component_source_target_maps_complete: bool,
}

fn z() -> GaussianRational {
    Complex::new(Ratio::from_integer(0), Ratio::from_integer(0))
}

fn integer(value: i64) -> GaussianRational {
    Complex::new(Ratio::from_integer(value), Ratio::from_integer(0))
}

fn add_scaled(target: &mut SparseSpinor, source: &SparseSpinor, scale: i64) {
    if scale == 0 {
        return;
    }
    for (&index, coefficient) in source {
        let entry = target.entry(index).or_insert_with(z);
        *entry += coefficient.clone() * integer(scale);
        if *entry == z() {
            target.remove(&index);
        }
    }
}

fn gamma_apply(gamma: &Matrix, source: &SparseSpinor) -> SparseSpinor {
    let mut result = SparseSpinor::new();
    for (&column, coefficient) in source {
        for (row, gamma_row) in gamma.iter().enumerate() {
            if gamma_row[column] == z() {
                continue;
            }
            let entry = result.entry(row).or_insert_with(z);
            *entry += gamma_row[column].clone() * coefficient.clone();
            if *entry == z() {
                result.remove(&row);
            }
        }
    }
    result
}

fn gamma_product_apply(
    gammas: &[Matrix],
    left: usize,
    right: usize,
    source: &SparseSpinor,
) -> SparseSpinor {
    gamma_apply(&gammas[left], &gamma_apply(&gammas[right], source))
}

fn basis_spinor(index: usize) -> SparseSpinor {
    BTreeMap::from([(index, integer(1))])
}

fn add_vector_spinor(target: &mut VectorSpinor, source: &VectorSpinor, scale: i64) {
    for (target_component, source_component) in target.iter_mut().zip(source) {
        add_scaled(target_component, source_component, scale);
    }
}

fn scale_vector_spinor(source: &VectorSpinor, scale: i64) -> VectorSpinor {
    let mut result = vec![SparseSpinor::new(); VECTOR_DIMENSION];
    add_vector_spinor(&mut result, source, scale);
    result
}

fn vector_spinor_residual_entries(left: &VectorSpinor, right: &VectorSpinor) -> usize {
    left.iter()
        .zip(right)
        .map(|(left_component, right_component)| {
            let mut residual = left_component.clone();
            add_scaled(&mut residual, right_component, -1);
            residual.len()
        })
        .sum()
}

fn spinor_residual_entries(left: &SparseSpinor, right: &SparseSpinor) -> usize {
    let mut residual = left.clone();
    add_scaled(&mut residual, right, -1);
    residual.len()
}

/// The numerator `Q = 11 I - Gamma_c Gamma^d` of the exact
/// gamma-traceless vector-spinor projector.
fn projector_numerator_column(
    gammas: &[Matrix],
    input_vector: usize,
    input_spinor: usize,
) -> VectorSpinor {
    let unit = basis_spinor(input_spinor);
    (0..VECTOR_DIMENSION)
        .map(|output_vector| {
            let mut component = SparseSpinor::new();
            if output_vector == input_vector {
                add_scaled(&mut component, &unit, VECTOR_DIMENSION as i64);
            }
            let gamma_product = gamma_product_apply(gammas, output_vector, input_vector, &unit);
            add_scaled(&mut component, &gamma_product, -1);
            component
        })
        .collect()
}

fn gamma_trace(gammas: &[Matrix], source: &VectorSpinor) -> SparseSpinor {
    let mut result = SparseSpinor::new();
    for (vector, component) in source.iter().enumerate() {
        add_scaled(&mut result, &gamma_apply(&gammas[vector], component), 1);
    }
    result
}

fn apply_projector_numerator(gammas: &[Matrix], source: &VectorSpinor) -> VectorSpinor {
    let trace = gamma_trace(gammas, source);
    source
        .iter()
        .enumerate()
        .map(|(vector, component)| {
            let mut result = SparseSpinor::new();
            add_scaled(&mut result, component, VECTOR_DIMENSION as i64);
            add_scaled(&mut result, &gamma_apply(&gammas[vector], &trace), -1);
            result
        })
        .collect()
}

fn trace_embedding(source: &VectorSpinor, momentum_a: usize, momentum_b: usize) -> VectorSpinor {
    if momentum_a == momentum_b {
        source.clone()
    } else {
        vec![SparseSpinor::new(); VECTOR_DIMENSION]
    }
}

/// Integer-normalized symmetric-traceless embedding
///
/// `A delta_{c(a} psi_{b)} + B Gamma_c Gamma_(a psi_{b)}
///  + C delta_{ab} psi_c`.
///
/// The certified coefficients are `(A,B,C)=(11,-1,-2)`.
fn symmetric_traceless_embedding(
    gammas: &[Matrix],
    source: &VectorSpinor,
    momentum_a: usize,
    momentum_b: usize,
    coefficients: [i64; 3],
) -> VectorSpinor {
    let [delta_coefficient, gamma_coefficient, trace_coefficient] = coefficients;
    let mut result = vec![SparseSpinor::new(); VECTOR_DIMENSION];
    for output_vector in 0..VECTOR_DIMENSION {
        if output_vector == momentum_a {
            add_scaled(
                &mut result[output_vector],
                &source[momentum_b],
                delta_coefficient,
            );
        }
        if output_vector == momentum_b {
            add_scaled(
                &mut result[output_vector],
                &source[momentum_a],
                delta_coefficient,
            );
        }
        let first = gamma_product_apply(gammas, output_vector, momentum_a, &source[momentum_b]);
        add_scaled(&mut result[output_vector], &first, gamma_coefficient);
        let second = gamma_product_apply(gammas, output_vector, momentum_b, &source[momentum_a]);
        add_scaled(&mut result[output_vector], &second, gamma_coefficient);
        if momentum_a == momentum_b {
            add_scaled(
                &mut result[output_vector],
                &source[output_vector],
                trace_coefficient,
            );
        }
    }
    result
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    sha256(&serde_json::to_vec(value).expect("serialize recoupling certificate hash payload"))
}

fn pinned_fixtures() -> Vec<PinnedLevel12Fixture> {
    [
        (
            "data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_1.i16le",
            LEVEL12_10002_KERNEL_1,
            LEVEL12_10002_KERNEL_SHA256[0],
        ),
        (
            "data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_2.i16le",
            LEVEL12_10002_KERNEL_2,
            LEVEL12_10002_KERNEL_SHA256[1],
        ),
    ]
    .into_iter()
    .map(|(artifact, bytes, expected)| {
        let actual = sha256(bytes);
        PinnedLevel12Fixture {
            artifact,
            bytes: bytes.len(),
            hash_matches: bytes.len() == LEVEL12_10002_KERNEL_BYTES && actual == expected,
            sha256: actual,
        }
    })
    .collect()
}

fn certify_with_coefficients(coefficients: [i64; 3]) -> SecondMomentumRecouplingReport {
    let gammas = crate::eleven_dimensional_clifford::gamma_matrices();
    assert_eq!(gammas.len(), VECTOR_DIMENSION);
    let inventory =
        crate::eleven_dimensional_second_momentum::second_momentum_representation_inventory();

    let channels = vec![
        RecouplingChannel {
            dynkin_label: "00001",
            dimension: 32,
            multiplicity: 1,
        },
        RecouplingChannel {
            dynkin_label: "01001",
            dimension: 1_408,
            multiplicity: 1,
        },
        RecouplingChannel {
            dynkin_label: "10001",
            dimension: TARGET_DIMENSION,
            multiplicity: 2,
        },
        RecouplingChannel {
            dynkin_label: "11001",
            dimension: 10_240,
            multiplicity: 1,
        },
        RecouplingChannel {
            dynkin_label: "20001",
            dimension: 1_760,
            multiplicity: 1,
        },
        RecouplingChannel {
            dynkin_label: "30001",
            dimension: 7_040,
            multiplicity: 1,
        },
    ];
    let decomposed_dimension = channels
        .iter()
        .map(|channel| channel.dimension * channel.multiplicity)
        .sum::<usize>();

    let mut clifford_residual_entries = 0;
    for left in 0..VECTOR_DIMENSION {
        for right in 0..VECTOR_DIMENSION {
            for spinor in 0..SPINOR_DIMENSION {
                let unit = basis_spinor(spinor);
                let mut actual = gamma_product_apply(&gammas, left, right, &unit);
                add_scaled(
                    &mut actual,
                    &gamma_product_apply(&gammas, right, left, &unit),
                    1,
                );
                let expected = if left == right {
                    scale_sparse(&unit, 2)
                } else {
                    SparseSpinor::new()
                };
                clifford_residual_entries += spinor_residual_entries(&actual, &expected);
            }
        }
    }

    let mut projector_polynomial_residual_entries = 0;
    let mut projector_trace = z();
    let mut trace_path_target_gamma_residual_entries = 0;
    let mut symmetric_traceless_target_gamma_residual_entries = 0;
    let mut symmetric_traceless_momentum_trace_residual_entries = 0;
    let mut trace_extraction_residual_entries = 0;
    let mut mixed_extraction_residual_entries = 0;

    let extraction_scalar = coefficients[0] * (VECTOR_DIMENSION as i64 + 1)
        + coefficients[1] * (VECTOR_DIMENSION as i64 + 2)
        + coefficients[2];

    for input_vector in 0..VECTOR_DIMENSION {
        for input_spinor in 0..SPINOR_DIMENSION {
            let projected = projector_numerator_column(&gammas, input_vector, input_spinor);
            projector_trace += projected[input_vector]
                .get(&input_spinor)
                .cloned()
                .unwrap_or_else(z);

            let projected_twice = apply_projector_numerator(&gammas, &projected);
            let expected_twice = scale_vector_spinor(&projected, VECTOR_DIMENSION as i64);
            projector_polynomial_residual_entries +=
                vector_spinor_residual_entries(&projected_twice, &expected_twice);

            let mut trace_trace = vec![SparseSpinor::new(); VECTOR_DIMENSION];
            let mut traceless_trace = vec![SparseSpinor::new(); VECTOR_DIMENSION];
            for momentum in 0..VECTOR_DIMENSION {
                let trace = trace_embedding(&projected, momentum, momentum);
                let traceless = symmetric_traceless_embedding(
                    &gammas,
                    &projected,
                    momentum,
                    momentum,
                    coefficients,
                );
                add_vector_spinor(&mut trace_trace, &trace, 1);
                add_vector_spinor(&mut traceless_trace, &traceless, 1);
            }
            trace_extraction_residual_entries += vector_spinor_residual_entries(
                &trace_trace,
                &scale_vector_spinor(&projected, VECTOR_DIMENSION as i64),
            );
            symmetric_traceless_momentum_trace_residual_entries += vector_spinor_residual_entries(
                &traceless_trace,
                &vec![SparseSpinor::new(); VECTOR_DIMENSION],
            );

            for momentum_a in 0..VECTOR_DIMENSION {
                for momentum_b in momentum_a..VECTOR_DIMENSION {
                    let trace = trace_embedding(&projected, momentum_a, momentum_b);
                    let traceless = symmetric_traceless_embedding(
                        &gammas,
                        &projected,
                        momentum_a,
                        momentum_b,
                        coefficients,
                    );
                    trace_path_target_gamma_residual_entries += gamma_trace(&gammas, &trace).len();
                    symmetric_traceless_target_gamma_residual_entries +=
                        gamma_trace(&gammas, &traceless).len();
                }
            }

            for momentum_b in 0..VECTOR_DIMENSION {
                let mut trace_mixed = SparseSpinor::new();
                let mut traceless_mixed = SparseSpinor::new();
                for momentum_a in 0..VECTOR_DIMENSION {
                    let trace = trace_embedding(&projected, momentum_a, momentum_b);
                    let traceless = symmetric_traceless_embedding(
                        &gammas,
                        &projected,
                        momentum_a,
                        momentum_b,
                        coefficients,
                    );
                    add_scaled(&mut trace_mixed, &trace[momentum_a], 1);
                    add_scaled(&mut traceless_mixed, &traceless[momentum_a], 1);
                }
                mixed_extraction_residual_entries +=
                    spinor_residual_entries(&trace_mixed, &projected[momentum_b]);
                mixed_extraction_residual_entries += spinor_residual_entries(
                    &traceless_mixed,
                    &scale_sparse(&projected[momentum_b], extraction_scalar),
                );
            }
        }
    }

    let gamma_traceless_projector_rank = if projector_polynomial_residual_entries == 0
        && projector_trace == integer((VECTOR_DIMENSION * TARGET_DIMENSION) as i64)
    {
        TARGET_DIMENSION
    } else {
        0
    };
    let gamma_trace_rank = VECTOR_SPINOR_DIMENSION - gamma_traceless_projector_rank;
    let extraction_matrix = [[VECTOR_DIMENSION as i64, 0], [1, extraction_scalar]];
    let extraction_determinant = extraction_matrix[0][0] * extraction_matrix[1][1];
    let multiplicity_space_rank =
        usize::from(extraction_matrix[0][0] != 0) + usize::from(extraction_determinant != 0);
    let trace_path_image_rank = gamma_traceless_projector_rank;
    let symmetric_traceless_path_image_rank = if extraction_scalar != 0
        && symmetric_traceless_target_gamma_residual_entries == 0
        && symmetric_traceless_momentum_trace_residual_entries == 0
        && mixed_extraction_residual_entries == 0
    {
        gamma_traceless_projector_rank
    } else {
        0
    };
    let combined_10001_image_rank = if multiplicity_space_rank == 2 {
        trace_path_image_rank + symmetric_traceless_path_image_rank
    } else {
        0
    };

    let fixtures = pinned_fixtures();
    let fixture_hashes_complete = fixtures.iter().all(|fixture| fixture.hash_matches);
    let representation_inventory_matches = inventory.representation_inventory_complete
        && inventory.symmetric_square_times_target_channels.len() == channels.len()
        && inventory
            .symmetric_square_times_target_channels
            .iter()
            .zip(&channels)
            .all(|(inventory_channel, channel)| {
                inventory_channel.dynkin_label == channel.dynkin_label
                    && inventory_channel.multiplicity == channel.multiplicity
            });
    let representation_decomposition_complete =
        representation_inventory_matches && decomposed_dimension == EXPECTED_DOMAIN_DIMENSION;
    let component_source_target_maps_complete = false;

    let hash_payload = CertificateHashPayload {
        channels: &channels,
        symmetric_tensor_dimension: SYMMETRIC_TENSOR_DIMENSION,
        target_dimension: TARGET_DIMENSION,
        tensor_product_dimension: EXPECTED_DOMAIN_DIMENSION,
        decomposed_dimension,
        gamma_trace_rank,
        gamma_traceless_projector_rank,
        clifford_residual_entries,
        trace_path_image_rank,
        symmetric_traceless_path_image_rank,
        combined_10001_image_rank,
        multiplicity_space_rank,
        extraction_matrix,
        extraction_determinant,
        trace_path_target_gamma_residual_entries,
        symmetric_traceless_target_gamma_residual_entries,
        symmetric_traceless_momentum_trace_residual_entries,
        trace_extraction_residual_entries,
        mixed_extraction_residual_entries,
        symmetric_traceless_embedding_coefficients: coefficients,
        fixture_hashes: fixtures
            .iter()
            .map(|fixture| fixture.sha256.as_str())
            .collect(),
        representation_inventory_sha256: &inventory.inventory_sha256,
        representation_inventory_matches,
        representation_decomposition_complete,
        component_source_target_maps_complete,
    };
    let certificate_sha256 = sha256_json(&hash_payload);

    let passed = representation_decomposition_complete
        && fixture_hashes_complete
        && clifford_residual_entries == 0
        && gamma_trace_rank == SPINOR_DIMENSION
        && gamma_traceless_projector_rank == TARGET_DIMENSION
        && projector_polynomial_residual_entries == 0
        && trace_path_target_gamma_residual_entries == 0
        && symmetric_traceless_target_gamma_residual_entries == 0
        && symmetric_traceless_momentum_trace_residual_entries == 0
        && trace_extraction_residual_entries == 0
        && mixed_extraction_residual_entries == 0
        && trace_path_image_rank == TARGET_DIMENSION
        && symmetric_traceless_path_image_rank == TARGET_DIMENSION
        && combined_10001_image_rank == 2 * TARGET_DIMENSION
        && multiplicity_space_rank == 2;

    SecondMomentumRecouplingReport {
        schema_version: "adynkra-11d-second-momentum-recoupling-v1",
        basis_convention: "exact 32-dimensional complex Euclidean B5 Clifford basis with orthonormal Cartesian vector metric",
        tensor_product: "Sym^2(10000) tensor (10001)",
        channels,
        symmetric_tensor_dimension: SYMMETRIC_TENSOR_DIMENSION,
        target_dimension: TARGET_DIMENSION,
        tensor_product_dimension: EXPECTED_DOMAIN_DIMENSION,
        decomposed_dimension,
        decomposition_dimension_residual: decomposed_dimension as isize
            - EXPECTED_DOMAIN_DIMENSION as isize,
        ambient_vector_spinor_dimension: VECTOR_SPINOR_DIMENSION,
        gamma_trace_rank,
        gamma_traceless_projector_rank,
        gamma_traceless_projector_polynomial_entries_checked: VECTOR_SPINOR_DIMENSION
            * VECTOR_SPINOR_DIMENSION,
        gamma_traceless_projector_polynomial_residual_entries:
            projector_polynomial_residual_entries,
        clifford_entries_checked: VECTOR_DIMENSION
            * VECTOR_DIMENSION
            * SPINOR_DIMENSION
            * SPINOR_DIMENSION,
        clifford_residual_entries,
        trace_path_image_rank,
        symmetric_traceless_path_image_rank,
        combined_10001_image_rank,
        multiplicity_space_rank,
        extraction_matrix,
        extraction_determinant,
        trace_path_target_gamma_entries_checked: VECTOR_SPINOR_DIMENSION
            * SYMMETRIC_TENSOR_DIMENSION
            * SPINOR_DIMENSION,
        trace_path_target_gamma_residual_entries,
        symmetric_traceless_target_gamma_entries_checked: VECTOR_SPINOR_DIMENSION
            * SYMMETRIC_TENSOR_DIMENSION
            * SPINOR_DIMENSION,
        symmetric_traceless_target_gamma_residual_entries,
        symmetric_traceless_momentum_trace_entries_checked: VECTOR_SPINOR_DIMENSION
            * VECTOR_DIMENSION
            * SPINOR_DIMENSION,
        symmetric_traceless_momentum_trace_residual_entries,
        trace_extraction_entries_checked: VECTOR_SPINOR_DIMENSION
            * VECTOR_DIMENSION
            * SPINOR_DIMENSION,
        trace_extraction_residual_entries,
        mixed_extraction_entries_checked: 2
            * VECTOR_SPINOR_DIMENSION
            * VECTOR_DIMENSION
            * SPINOR_DIMENSION,
        mixed_extraction_residual_entries,
        symmetric_traceless_embedding_coefficients: coefficients,
        pinned_level12_10002_fixtures: fixtures,
        fixture_hashes_complete,
        representation_inventory_sha256: inventory.inventory_sha256,
        representation_inventory_matches,
        representation_decomposition_complete,
        component_source_target_maps_complete,
        certificate_sha256,
        passed,
        boundary: "This certifies the B5 representation inventory by dimensions and constructs the two exact (10001) momentum recoupling paths. The level-12 10002 source kernels are hash-pinned, but no complete level-12 source-to-intermediate or intermediate-to-target Clebsch-Gordan map is claimed.",
    }
}

fn scale_sparse(source: &SparseSpinor, scale: i64) -> SparseSpinor {
    let mut result = SparseSpinor::new();
    add_scaled(&mut result, source, scale);
    result
}

pub fn verify() -> SecondMomentumRecouplingReport {
    let report = certify_with_coefficients([VECTOR_DIMENSION as i64, -1, -2]);
    assert_eq!(
        report.certificate_sha256, EXPECTED_CERTIFICATE_SHA256,
        "second-momentum recoupling certificate hash mismatch"
    );
    report
}

pub fn write_artifact(path: &Path) -> io::Result<SecondMomentumRecouplingReport> {
    let report = verify();
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "second-momentum recoupling certificate did not pass",
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let payload = serde_json::to_vec_pretty(&report).map_err(io::Error::other)?;
    fs::write(&temporary, payload)?;
    fs::rename(temporary, path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_second_momentum_recoupling_separates_both_10001_paths() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.certificate_sha256, EXPECTED_CERTIFICATE_SHA256);
        assert_eq!(report.tensor_product_dimension, 21_120);
        assert_eq!(report.decomposed_dimension, 21_120);
        assert_eq!(report.decomposition_dimension_residual, 0);
        assert_eq!(report.gamma_trace_rank, 32);
        assert_eq!(report.gamma_traceless_projector_rank, 320);
        assert_eq!(report.trace_path_image_rank, 320);
        assert_eq!(report.symmetric_traceless_path_image_rank, 320);
        assert_eq!(report.combined_10001_image_rank, 640);
        assert_eq!(report.multiplicity_space_rank, 2);
        assert_eq!(report.extraction_matrix, [[11, 0], [1, 117]]);
        assert_eq!(report.extraction_determinant, 1_287);
        assert_eq!(report.clifford_residual_entries, 0);
        assert_eq!(
            report.gamma_traceless_projector_polynomial_residual_entries,
            0
        );
        assert_eq!(report.trace_path_target_gamma_residual_entries, 0);
        assert_eq!(report.symmetric_traceless_target_gamma_residual_entries, 0);
        assert_eq!(
            report.symmetric_traceless_momentum_trace_residual_entries,
            0
        );
        assert_eq!(report.trace_extraction_residual_entries, 0);
        assert_eq!(report.mixed_extraction_residual_entries, 0);
        assert!(report.fixture_hashes_complete);
        assert!(report.representation_inventory_matches);
        assert!(report.representation_decomposition_complete);
        assert!(!report.component_source_target_maps_complete);
    }

    #[test]
    fn trace_subtraction_mutation_is_detected() {
        let good = verify();
        let mutated = certify_with_coefficients([VECTOR_DIMENSION as i64, -1, -1]);
        assert!(good.passed);
        assert!(!mutated.passed);
        assert_ne!(
            mutated.symmetric_traceless_momentum_trace_residual_entries,
            0
        );
        assert_ne!(mutated.certificate_sha256, good.certificate_sha256);
    }
}
