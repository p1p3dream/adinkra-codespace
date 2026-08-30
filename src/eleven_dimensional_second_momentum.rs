//! Exact bounded scaffold for second-momentum 11D operator columns.
//!
//! The representation-level inventory is exact. The component
//! Clebsch-Gordan fixtures remain representative and incomplete. This module
//! composes those fixtures with the six exact spinor-form gauge channels and
//! emits target-resolved `p^2 D^13 Lambda` terms in the established 320-state
//! vector-spinor basis.

use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

const MOMENTUM_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const TARGET_DIMENSION: usize = 320;
const REPRESENTATION_INVENTORY_SHA256: &str =
    "83698bf554699aa54c1f576366ca0945424675e9698a4203aadb787f232cff57";
const REPRESENTATIVE_FIXTURE_SHA256: [&str; 3] = [
    "e9116f42b5de4cd73c4704bd10afac4067eb1afe40c08daa72df96be86a83ee0",
    "067dbd24ff691d3b00ca88370db95ced55972faa445d5953952f8362aa59c31f",
    "b77022e0ef184c1642b990454de7b43dddec0237726f517e6453454d3902b410",
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExactQiCoefficient {
    pub real_numerator: i128,
    pub imaginary_numerator: i128,
    pub denominator: i128,
}

impl ExactQiCoefficient {
    pub fn new(real_numerator: i128, imaginary_numerator: i128, denominator: i128) -> Self {
        assert_ne!(denominator, 0, "Q(i) denominator must be nonzero");
        let (mut real_numerator, mut imaginary_numerator, mut denominator) =
            (real_numerator, imaginary_numerator, denominator);
        if denominator < 0 {
            real_numerator = -real_numerator;
            imaginary_numerator = -imaginary_numerator;
            denominator = -denominator;
        }
        let divisor = gcd_i128(
            gcd_i128(real_numerator.abs(), imaginary_numerator.abs()),
            denominator,
        )
        .max(1);
        Self {
            real_numerator: real_numerator / divisor,
            imaginary_numerator: imaginary_numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    pub fn is_zero(self) -> bool {
        self.real_numerator == 0 && self.imaginary_numerator == 0
    }

    fn multiply(self, other: Self) -> Self {
        let real = self
            .real_numerator
            .checked_mul(other.real_numerator)
            .and_then(|value| {
                self.imaginary_numerator
                    .checked_mul(other.imaginary_numerator)
                    .and_then(|other_value| value.checked_sub(other_value))
            })
            .expect("Q(i) real product exceeds i128");
        let imaginary = self
            .real_numerator
            .checked_mul(other.imaginary_numerator)
            .and_then(|value| {
                self.imaginary_numerator
                    .checked_mul(other.real_numerator)
                    .and_then(|other_value| value.checked_add(other_value))
            })
            .expect("Q(i) imaginary product exceeds i128");
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .expect("Q(i) denominator product exceeds i128");
        Self::new(real, imaginary, denominator)
    }

    fn add(self, other: Self) -> Self {
        let real = self
            .real_numerator
            .checked_mul(other.denominator)
            .and_then(|value| {
                other
                    .real_numerator
                    .checked_mul(self.denominator)
                    .and_then(|other_value| value.checked_add(other_value))
            })
            .expect("Q(i) real sum exceeds i128");
        let imaginary = self
            .imaginary_numerator
            .checked_mul(other.denominator)
            .and_then(|value| {
                other
                    .imaginary_numerator
                    .checked_mul(self.denominator)
                    .and_then(|other_value| value.checked_add(other_value))
            })
            .expect("Q(i) imaginary sum exceeds i128");
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .expect("Q(i) sum denominator exceeds i128");
        Self::new(real, imaginary, denominator)
    }

    fn scale_integer(self, scale: i128) -> Self {
        Self::new(
            self.real_numerator
                .checked_mul(scale)
                .expect("Q(i) real scale exceeds i128"),
            self.imaginary_numerator
                .checked_mul(scale)
                .expect("Q(i) imaginary scale exceeds i128"),
            self.denominator,
        )
    }
}

fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentumOperatorSpec {
    pub ordinal: usize,
    pub label: String,
    pub role: String,
    pub momentum_degree: usize,
    pub exterior_derivative_order: usize,
    pub target_dynkin_label: String,
    pub symmetric_momentum_channel_dynkin_label: String,
    pub symmetric_momentum_channel_copy: usize,
    pub representative_fixture_available: bool,
    pub representation_inventory_complete: bool,
    pub component_clebsch_gordan_fixture_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymmetricMomentumChannel {
    pub dynkin_label: String,
    pub multiplicity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentumRepresentationInventory {
    pub schema_version: String,
    pub symmetric_square_times_target_channels: Vec<SymmetricMomentumChannel>,
    pub source_labels: usize,
    pub source_intermediate_pairs: usize,
    pub source_copy_incidences: usize,
    pub new_operator_variables: usize,
    pub independent_symmetric_momentum_paths_for_10001: usize,
    pub inherited_old_coefficient_variables: usize,
    pub new_variables_independent_of_inherited_old_variables: bool,
    pub representation_inventory_complete: bool,
    pub component_clebsch_gordan_maps_complete: bool,
    pub inventory_sha256: String,
}

#[derive(Debug, Serialize)]
struct InventoryHashPayload<'a> {
    symmetric_square_times_target_channels: &'a [SymmetricMomentumChannel],
    source_labels: usize,
    source_intermediate_pairs: usize,
    source_copy_incidences: usize,
    new_operator_variables: usize,
    independent_symmetric_momentum_paths_for_10001: usize,
}

pub fn second_momentum_representation_inventory() -> SecondMomentumRepresentationInventory {
    let channels = vec![
        SymmetricMomentumChannel {
            dynkin_label: "00001".to_string(),
            multiplicity: 1,
        },
        SymmetricMomentumChannel {
            dynkin_label: "01001".to_string(),
            multiplicity: 1,
        },
        SymmetricMomentumChannel {
            dynkin_label: "10001".to_string(),
            multiplicity: 2,
        },
        SymmetricMomentumChannel {
            dynkin_label: "11001".to_string(),
            multiplicity: 1,
        },
        SymmetricMomentumChannel {
            dynkin_label: "20001".to_string(),
            multiplicity: 1,
        },
        SymmetricMomentumChannel {
            dynkin_label: "30001".to_string(),
            multiplicity: 1,
        },
    ];
    let inventory_sha256 = sha256_json(&InventoryHashPayload {
        symmetric_square_times_target_channels: &channels,
        source_labels: 19,
        source_intermediate_pairs: 35,
        source_copy_incidences: 73,
        new_operator_variables: 77,
        independent_symmetric_momentum_paths_for_10001: 2,
    });
    assert_eq!(
        inventory_sha256, REPRESENTATION_INVENTORY_SHA256,
        "second-momentum representation-inventory SHA-256 mismatch"
    );
    SecondMomentumRepresentationInventory {
        schema_version: "adynkra-11d-second-momentum-representation-inventory-v1".to_string(),
        symmetric_square_times_target_channels: channels,
        source_labels: 19,
        source_intermediate_pairs: 35,
        source_copy_incidences: 73,
        new_operator_variables: 77,
        independent_symmetric_momentum_paths_for_10001: 2,
        inherited_old_coefficient_variables: 49,
        new_variables_independent_of_inherited_old_variables: true,
        representation_inventory_complete: true,
        component_clebsch_gordan_maps_complete: false,
        inventory_sha256,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentumFixtureTerm {
    pub momentum_pair: [usize; 2],
    pub exterior_mask: u32,
    pub target_basis_ordinal: usize,
    pub source_spinor_weight_index: usize,
    pub coefficient: ExactQiCoefficient,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentumOperatorFixture {
    pub spec: SecondMomentumOperatorSpec,
    pub terms: Vec<SecondMomentumFixtureTerm>,
    pub fixture_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct FixtureHashPayload<'a> {
    spec: &'a SecondMomentumOperatorSpec,
    terms: &'a [SecondMomentumFixtureTerm],
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("serialize deterministic second-momentum value");
    format!("{:x}", Sha256::digest(bytes))
}

fn alternating_mask(offset: usize) -> u32 {
    (0..12).fold(0_u32, |mask, index| {
        mask | (1_u32 << ((offset + 2 * index) % SPINOR_DIMENSION))
    })
}

fn fixture(
    ordinal: usize,
    label: &str,
    symmetric_momentum_channel_dynkin_label: &str,
    symmetric_momentum_channel_copy: usize,
    terms: Vec<SecondMomentumFixtureTerm>,
) -> SecondMomentumOperatorFixture {
    let spec = SecondMomentumOperatorSpec {
        ordinal,
        label: label.to_string(),
        role: "representative exact p^2 D^12 vector-spinor operator column".to_string(),
        momentum_degree: 2,
        exterior_derivative_order: 12,
        target_dynkin_label: "10001".to_string(),
        symmetric_momentum_channel_dynkin_label: symmetric_momentum_channel_dynkin_label
            .to_string(),
        symmetric_momentum_channel_copy,
        representative_fixture_available: true,
        representation_inventory_complete: true,
        component_clebsch_gordan_fixture_complete: false,
    };
    let fixture_sha256 = sha256_json(&FixtureHashPayload {
        spec: &spec,
        terms: &terms,
    });
    assert_eq!(
        fixture_sha256, REPRESENTATIVE_FIXTURE_SHA256[ordinal],
        "representative second-momentum fixture SHA-256 mismatch"
    );
    SecondMomentumOperatorFixture {
        spec,
        terms,
        fixture_sha256,
    }
}

/// Deterministic representative columns used to establish the exact API.
///
/// These columns sample diagonal, mixed, and separated momentum pairs and
/// rational Gaussian coefficients. They are not a claim about the complete
/// level-12 source inventory.
pub fn representative_second_momentum_fixtures() -> Vec<SecondMomentumOperatorFixture> {
    vec![
        fixture(
            0,
            "representative-p00-target000",
            "00001",
            1,
            vec![
                SecondMomentumFixtureTerm {
                    momentum_pair: [0, 0],
                    exterior_mask: (1_u32 << 12) - 1,
                    target_basis_ordinal: 0,
                    source_spinor_weight_index: 0,
                    coefficient: ExactQiCoefficient::new(1, 0, 1),
                },
                SecondMomentumFixtureTerm {
                    momentum_pair: [0, 0],
                    exterior_mask: ((1_u32 << 13) - 1) ^ 1,
                    target_basis_ordinal: 0,
                    source_spinor_weight_index: 0,
                    coefficient: ExactQiCoefficient::new(-1, 1, 2),
                },
            ],
        ),
        fixture(
            1,
            "representative-p01-target001-path1",
            "10001",
            1,
            vec![
                SecondMomentumFixtureTerm {
                    momentum_pair: [0, 1],
                    exterior_mask: alternating_mask(0),
                    target_basis_ordinal: 1,
                    source_spinor_weight_index: 7,
                    coefficient: ExactQiCoefficient::new(2, -1, 3),
                },
                SecondMomentumFixtureTerm {
                    momentum_pair: [0, 1],
                    exterior_mask: alternating_mask(1),
                    target_basis_ordinal: 1,
                    source_spinor_weight_index: 7,
                    coefficient: ExactQiCoefficient::new(1, 1, 3),
                },
            ],
        ),
        fixture(
            2,
            "representative-p5-10-target319-path2",
            "10001",
            2,
            vec![
                SecondMomentumFixtureTerm {
                    momentum_pair: [5, 10],
                    exterior_mask: alternating_mask(4),
                    target_basis_ordinal: 319,
                    source_spinor_weight_index: 31,
                    coefficient: ExactQiCoefficient::new(0, 1, 2),
                },
                SecondMomentumFixtureTerm {
                    momentum_pair: [5, 10],
                    exterior_mask: alternating_mask(5),
                    target_basis_ordinal: 319,
                    source_spinor_weight_index: 31,
                    coefficient: ExactQiCoefficient::new(3, 0, 2),
                },
            ],
        ),
    ]
}

pub fn representative_second_momentum_operator_specs() -> Vec<SecondMomentumOperatorSpec> {
    representative_second_momentum_fixtures()
        .into_iter()
        .map(|fixture| fixture.spec)
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Level12SourceKernelTerm {
    pub exterior_mask: u32,
    pub coefficient: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AvailableLevel12SourceKernelFixture {
    pub dynkin_label: String,
    pub copy: usize,
    pub representative_role: String,
    pub exterior_degree: usize,
    pub source_columns: usize,
    pub coefficient_width_bytes: usize,
    pub binary_sha256: String,
    pub expected_binary_sha256: String,
    pub nonzero_coefficients: usize,
    pub maximum_absolute_coefficient: i16,
    pub terms: Vec<Level12SourceKernelTerm>,
    pub binary_contract_validated: bool,
    pub exact_raising_residuals_zero: bool,
    pub component_momentum_clebsch_gordan_map_available: bool,
}

fn decode_level12_i16_fixture(
    dynkin_label: &str,
    copy: usize,
    representative_role: &str,
    bytes: &[u8],
    expected_sha256: &str,
    expected_nonzero: usize,
    expected_maximum: i16,
) -> AvailableLevel12SourceKernelFixture {
    let binary_sha256 = format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        binary_sha256, expected_sha256,
        "level-12 source-kernel fixture SHA-256 mismatch"
    );
    assert_eq!(bytes.len() % 2, 0);
    let masks = crate::eleven_dimensional_level16_couplings::exterior_highest_weight_basis_masks(
        12,
        dynkin_label,
    );
    let coefficients = bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    assert_eq!(masks.len(), coefficients.len());
    let nonzero_coefficients = coefficients
        .iter()
        .filter(|coefficient| **coefficient != 0)
        .count();
    let maximum_absolute_coefficient = coefficients
        .iter()
        .map(|coefficient| coefficient.saturating_abs())
        .max()
        .unwrap_or(0);
    assert_eq!(nonzero_coefficients, expected_nonzero);
    assert_eq!(maximum_absolute_coefficient, expected_maximum);
    let terms = masks
        .into_iter()
        .zip(coefficients)
        .filter_map(|(exterior_mask, coefficient)| {
            (coefficient != 0).then_some(Level12SourceKernelTerm {
                exterior_mask,
                coefficient,
            })
        })
        .collect::<Vec<_>>();
    let sparse_terms = terms
        .iter()
        .map(|term| (term.exterior_mask, i64::from(term.coefficient)))
        .collect::<Vec<_>>();
    let exact_raising_residuals_zero = crate::eleven_dimensional_level16_couplings::exterior_highest_weight_raising_residuals_are_zero(&sparse_terms);
    assert!(
        exact_raising_residuals_zero,
        "level-12 source-kernel fixture has a nonzero exact raising residual"
    );
    AvailableLevel12SourceKernelFixture {
        dynkin_label: dynkin_label.to_string(),
        copy,
        representative_role: representative_role.to_string(),
        exterior_degree: 12,
        source_columns: bytes.len() / 2,
        coefficient_width_bytes: 2,
        binary_sha256,
        expected_binary_sha256: expected_sha256.to_string(),
        nonzero_coefficients,
        maximum_absolute_coefficient,
        terms,
        binary_contract_validated: true,
        exact_raising_residuals_zero,
        component_momentum_clebsch_gordan_map_available: false,
    }
}

/// Exact level-12 source kernels currently materialized by the independent
/// generator. They are consumed as source fixtures only. The momentum-channel
/// Clebsch-Gordan maps needed to turn all 77 variables into physical columns
/// remain separate and incomplete.
pub fn available_level12_source_kernel_fixtures() -> Vec<AvailableLevel12SourceKernelFixture> {
    vec![
        decode_level12_i16_fixture(
            "30002",
            1,
            "symmetric-traceless second-momentum source representative",
            include_bytes!(
                "../data/eleven_dimensional_spinor_bridge/level12_30002_highest_weight_kernel_1.i16le"
            ),
            "bef24d4ebe642c0b6507dee9706fea06742d6ed708d14adc9dd599a8061c28c6",
            1_858,
            56,
        ),
        decode_level12_i16_fixture(
            "30002",
            2,
            "symmetric-traceless second-momentum source representative",
            include_bytes!(
                "../data/eleven_dimensional_spinor_bridge/level12_30002_highest_weight_kernel_2.i16le"
            ),
            "0e42717e2cb9449f92257a60b7a35e3c8a501dea176db02aa2d0a714b8904908",
            1_120,
            2,
        ),
        decode_level12_i16_fixture(
            "30002",
            3,
            "symmetric-traceless second-momentum source representative",
            include_bytes!(
                "../data/eleven_dimensional_spinor_bridge/level12_30002_highest_weight_kernel_3.i16le"
            ),
            "d3321a53f37acea6d3d8e4efc11022c037e0339a5e9a7cda87bdb003ca3b2945",
            2_738,
            1_760,
        ),
        decode_level12_i16_fixture(
            "10002",
            1,
            "trace/STT second-momentum source representative",
            include_bytes!(
                "../data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_1.i16le"
            ),
            "c3eb687d6d868cd08fcd90a0c741815681b3c68ca4cf3157ddd929df0aa42e28",
            52_670,
            15_840,
        ),
        decode_level12_i16_fixture(
            "10002",
            2,
            "trace/STT second-momentum source representative",
            include_bytes!(
                "../data/eleven_dimensional_spinor_bridge/level12_10002_highest_weight_kernel_2.i16le"
            ),
            "ec51e06970fcff1b7b719d7ee5c3d9a69775fdf6558334c400d1540dbd81c7a1",
            3_465,
            3,
        ),
        decode_level12_i16_fixture(
            "40000",
            1,
            "highest-weight symmetric-tensor second-momentum source representative",
            include_bytes!(
                "../data/eleven_dimensional_spinor_bridge/level12_40000_highest_weight_kernel.i16le"
            ),
            "8d53deee4109315fec26ff5e928822fa18a4ca5cf687ec5a918daf44560d5518",
            5_544,
            7,
        ),
    ]
}

pub fn validate_representative_fixture(
    fixture: &SecondMomentumOperatorFixture,
) -> Result<(), String> {
    if fixture.spec.momentum_degree != 2
        || fixture.spec.exterior_derivative_order != 12
        || fixture.spec.target_dynkin_label != "10001"
        || !fixture.spec.representative_fixture_available
        || !fixture.spec.representation_inventory_complete
        || fixture.spec.component_clebsch_gordan_fixture_complete
        || fixture.spec.symmetric_momentum_channel_copy == 0
        || fixture.terms.is_empty()
    {
        return Err("second-momentum representative fixture metadata changed".to_string());
    }
    for term in &fixture.terms {
        if term.momentum_pair[0] > term.momentum_pair[1]
            || term.momentum_pair[1] >= MOMENTUM_DIMENSION
            || term.exterior_mask.count_ones() != 12
            || term.target_basis_ordinal >= TARGET_DIMENSION
            || term.source_spinor_weight_index >= SPINOR_DIMENSION
            || term.coefficient.denominator <= 0
            || term.coefficient.is_zero()
        {
            return Err("second-momentum representative fixture term is invalid".to_string());
        }
    }
    let observed = sha256_json(&FixtureHashPayload {
        spec: &fixture.spec,
        terms: &fixture.terms,
    });
    if fixture.fixture_sha256 != observed {
        return Err("second-momentum representative fixture SHA-256 mismatch".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct SecondMomentumStreamKey {
    momentum_pair: [usize; 2],
    exterior_mask: u32,
    target_basis_ordinal: usize,
    target_vector_weight_index: usize,
    target_spinor_weight_index: usize,
    parameter_component_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetResolvedSecondMomentumGaugeEntry {
    pub operator_ordinal: usize,
    pub momentum_pair: [usize; 2],
    pub exterior_mask: u32,
    pub target_basis_ordinal: usize,
    pub target_vector_weight_index: usize,
    pub target_spinor_weight_index: usize,
    pub parameter_component_index: usize,
    pub coefficient: ExactQiCoefficient,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentumTargetStreamReport {
    pub schema_version: String,
    pub role: String,
    pub operator_spec: SecondMomentumOperatorSpec,
    pub fixture_sha256: String,
    pub gauge_form_degree: usize,
    pub parameter_components_total: usize,
    pub parameter_components_selected: Vec<usize>,
    pub target_basis_ordinals_selected: Vec<usize>,
    pub entries: Vec<TargetResolvedSecondMomentumGaugeEntry>,
    pub stream_sha256: String,
    pub exact_normal_order_checked: bool,
    pub representative_fixture_only: bool,
    pub representation_inventory_complete: bool,
    pub component_clebsch_gordan_maps_complete: bool,
    pub second_momentum_new_operator_variables: usize,
    pub inherited_old_coefficient_variables: usize,
    pub new_variables_independent_of_inherited_old_variables: bool,
    pub p2_d13_wedge_stream_implemented: bool,
    pub p3_d11_contraction_stream_implemented: bool,
    pub inherited_old_column_p2_contractions_kept_separate: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub full_physical_fag_established: bool,
    pub boundary: String,
}

#[derive(Debug, Serialize)]
struct StreamHashPayload<'a> {
    operator_spec: &'a SecondMomentumOperatorSpec,
    fixture_sha256: &'a str,
    gauge_form_degree: usize,
    parameter_components_selected: &'a [usize],
    target_basis_ordinals_selected: &'a [usize],
    entries: &'a [TargetResolvedSecondMomentumGaugeEntry],
}

fn hash_second_momentum_stream(
    operator_spec: &SecondMomentumOperatorSpec,
    fixture_sha256: &str,
    gauge_form_degree: usize,
    parameter_components_selected: &[usize],
    target_basis_ordinals_selected: &[usize],
    entries: &[TargetResolvedSecondMomentumGaugeEntry],
) -> String {
    sha256_json(&StreamHashPayload {
        operator_spec,
        fixture_sha256,
        gauge_form_degree,
        parameter_components_selected,
        target_basis_ordinals_selected,
        entries,
    })
}

fn selected(selection: &[usize], value: usize) -> bool {
    selection.is_empty() || selection.binary_search(&value).is_ok()
}

/// Compose one representative `p^2 D^12` column with an exact gauge-form
/// channel and return the accumulated target-resolved `p^2 D^13 Lambda`
/// stream. Empty selections mean all components represented by the fixture.
pub fn build_representative_target_resolved_stream(
    operator_ordinal: usize,
    gauge_form_degree: usize,
    mut parameter_components_selected: Vec<usize>,
    mut target_basis_ordinals_selected: Vec<usize>,
) -> io::Result<SecondMomentumTargetStreamReport> {
    if gauge_form_degree > 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "gauge form degree must lie in 0..=5",
        ));
    }
    parameter_components_selected.sort_unstable();
    parameter_components_selected.dedup();
    target_basis_ordinals_selected.sort_unstable();
    target_basis_ordinals_selected.dedup();

    let fixture = representative_second_momentum_fixtures()
        .into_iter()
        .find(|fixture| fixture.spec.ordinal == operator_ordinal)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("representative second-momentum ordinal {operator_ordinal} is absent"),
            )
        })?;
    validate_representative_fixture(&fixture)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis()
        .into_iter()
        .filter(|(degree, _, _)| *degree == gauge_form_degree)
        .collect::<Vec<_>>();
    if parameter_components_selected
        .iter()
        .any(|component| *component >= gauge_basis.len())
        || target_basis_ordinals_selected
            .iter()
            .any(|target| *target >= TARGET_DIMENSION)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "second-momentum stream selection is outside its basis",
        ));
    }

    let dual_target_basis =
        crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let mut accumulated = BTreeMap::<SecondMomentumStreamKey, ExactQiCoefficient>::new();
    for (parameter_component_index, (_, _, matrix)) in gauge_basis.iter().enumerate() {
        if !selected(&parameter_components_selected, parameter_component_index) {
            continue;
        }
        for fixture_term in &fixture.terms {
            if !selected(
                &target_basis_ordinals_selected,
                fixture_term.target_basis_ordinal,
            ) {
                continue;
            }
            let dual_target = &dual_target_basis[fixture_term.target_basis_ordinal];
            for derivative_spinor in 0..SPINOR_DIMENSION {
                let gauge = &matrix[fixture_term.source_spinor_weight_index][derivative_spinor];
                if gauge.re.is_zero() && gauge.im.is_zero() {
                    continue;
                }
                let Some((exterior_mask, normal_order_sign)) =
                    crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(
                        fixture_term.exterior_mask,
                        derivative_spinor,
                    )
                else {
                    continue;
                };
                debug_assert_eq!(exterior_mask.count_ones(), 13);
                let gauge_coefficient = ExactQiCoefficient::new(
                    i128::from(*gauge.re.numer()) * i128::from(*gauge.im.denom()),
                    i128::from(*gauge.im.numer()) * i128::from(*gauge.re.denom()),
                    i128::from(*gauge.re.denom()) * i128::from(*gauge.im.denom()),
                );
                for target in &dual_target.raw_terms {
                    let coefficient = fixture_term
                        .coefficient
                        .multiply(gauge_coefficient)
                        .multiply(ExactQiCoefficient::new(
                            i128::from(target.numerator),
                            0,
                            i128::from(target.denominator),
                        ))
                        .scale_integer(normal_order_sign);
                    if coefficient.is_zero() {
                        continue;
                    }
                    let key = SecondMomentumStreamKey {
                        momentum_pair: fixture_term.momentum_pair,
                        exterior_mask,
                        target_basis_ordinal: fixture_term.target_basis_ordinal,
                        target_vector_weight_index: target.vector_weight_index,
                        target_spinor_weight_index: target.spinor_weight_index,
                        parameter_component_index,
                    };
                    accumulated
                        .entry(key)
                        .and_modify(|value| *value = value.add(coefficient))
                        .or_insert(coefficient);
                }
            }
        }
    }

    let entries = accumulated
        .into_iter()
        .filter_map(|(key, coefficient)| {
            (!coefficient.is_zero()).then_some(TargetResolvedSecondMomentumGaugeEntry {
                operator_ordinal,
                momentum_pair: key.momentum_pair,
                exterior_mask: key.exterior_mask,
                target_basis_ordinal: key.target_basis_ordinal,
                target_vector_weight_index: key.target_vector_weight_index,
                target_spinor_weight_index: key.target_spinor_weight_index,
                parameter_component_index: key.parameter_component_index,
                coefficient,
            })
        })
        .collect::<Vec<_>>();
    let stream_sha256 = hash_second_momentum_stream(
        &fixture.spec,
        &fixture.fixture_sha256,
        gauge_form_degree,
        &parameter_components_selected,
        &target_basis_ordinals_selected,
        &entries,
    );
    let full_parameter_projection_complete = parameter_components_selected.is_empty()
        || parameter_components_selected.len() == gauge_basis.len();
    let inventory = second_momentum_representation_inventory();
    Ok(SecondMomentumTargetStreamReport {
        schema_version: "adynkra-11d-second-momentum-representative-stream-v1".to_string(),
        role: "exact target-resolved p^2 D^13 Lambda stream for bounded representative p^2 D^12 columns".to_string(),
        operator_spec: fixture.spec,
        fixture_sha256: fixture.fixture_sha256,
        gauge_form_degree,
        parameter_components_total: gauge_basis.len(),
        parameter_components_selected,
        target_basis_ordinals_selected,
        entries,
        stream_sha256,
        exact_normal_order_checked: true,
        representative_fixture_only: true,
        representation_inventory_complete: inventory.representation_inventory_complete,
        component_clebsch_gordan_maps_complete: inventory.component_clebsch_gordan_maps_complete,
        second_momentum_new_operator_variables: inventory.new_operator_variables,
        inherited_old_coefficient_variables: inventory.inherited_old_coefficient_variables,
        new_variables_independent_of_inherited_old_variables: inventory
            .new_variables_independent_of_inherited_old_variables,
        p2_d13_wedge_stream_implemented: true,
        p3_d11_contraction_stream_implemented: false,
        inherited_old_column_p2_contractions_kept_separate: true,
        full_parameter_projection_complete,
        full_target_projection_complete: false,
        full_physical_fag_established: false,
        boundary: "The representation-level inventory has 77 new variables and is complete, but this exact stream validates only explicit representative p^2 D^13 wedge fixtures. Component Clebsch-Gordan maps and the companion p^3 D^11 contraction stream remain incomplete. Inherited old-column p^2 contractions remain separate under the prior rank-49 bounded result. This does not select physical K, complete F, exhaust parameters or targets, or establish F A G_p.".to_string(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Level12SourceKernelFixturePin {
    pub dynkin_label: String,
    pub copy: usize,
    pub representative_role: String,
    pub source_columns: usize,
    pub binary_sha256: String,
    pub nonzero_coefficients: usize,
    pub maximum_absolute_coefficient: i16,
    pub exact_raising_residuals_zero: bool,
    pub component_momentum_clebsch_gordan_map_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepresentativeSecondMomentumStreamPin {
    pub operator_ordinal: usize,
    pub fixture_sha256: String,
    pub symmetric_momentum_channel_dynkin_label: String,
    pub symmetric_momentum_channel_copy: usize,
    pub gauge_form_degree: usize,
    pub parameter_components_selected: Vec<usize>,
    pub target_basis_ordinals_selected: Vec<usize>,
    pub emitted_entries: usize,
    pub stream_sha256: String,
    pub exact_normal_order_checked: bool,
    pub synthetic_control_column: bool,
    pub component_clebsch_gordan_map: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondMomentumVerificationReport {
    pub schema_version: String,
    pub role: String,
    pub representation_inventory: SecondMomentumRepresentationInventory,
    pub available_level12_source_fixture_pins: Vec<Level12SourceKernelFixturePin>,
    pub available_level12_source_labels: Vec<String>,
    pub available_level12_kernel_copies: usize,
    pub expected_level12_source_labels: usize,
    pub expected_level12_kernel_copies: usize,
    pub level12_component_fixture_corpus_complete: bool,
    pub representative_stream_pins: Vec<RepresentativeSecondMomentumStreamPin>,
    pub representative_stream_benchmark_workload: String,
    pub wall_clock_benchmark_in_artifact: bool,
    pub p2_d13_wedge_control_stream_complete: bool,
    pub p3_d11_contraction_stream_complete: bool,
    pub component_clebsch_gordan_maps_complete: bool,
    pub inherited_old_column_p2_contractions_kept_separate: bool,
    pub generic_k_solved: bool,
    pub all_six_physical_fag_channels_checked: bool,
    pub physical_fag_established: bool,
    pub passed: bool,
    pub result: String,
    pub boundary: String,
}

/// Build the deterministic bounded second-momentum verification envelope.
pub fn verify() -> io::Result<SecondMomentumVerificationReport> {
    let inventory = second_momentum_representation_inventory();
    let source_fixtures = available_level12_source_kernel_fixtures();
    let source_fixture_pins = source_fixtures
        .iter()
        .map(|fixture| Level12SourceKernelFixturePin {
            dynkin_label: fixture.dynkin_label.clone(),
            copy: fixture.copy,
            representative_role: fixture.representative_role.clone(),
            source_columns: fixture.source_columns,
            binary_sha256: fixture.binary_sha256.clone(),
            nonzero_coefficients: fixture.nonzero_coefficients,
            maximum_absolute_coefficient: fixture.maximum_absolute_coefficient,
            exact_raising_residuals_zero: fixture.exact_raising_residuals_zero,
            component_momentum_clebsch_gordan_map_available: fixture
                .component_momentum_clebsch_gordan_map_available,
        })
        .collect::<Vec<_>>();
    let available_level12_source_labels = source_fixture_pins
        .iter()
        .map(|fixture| fixture.dynkin_label.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut representative_stream_pins = Vec::new();
    for fixture in representative_second_momentum_fixtures() {
        let target = fixture.terms[0].target_basis_ordinal;
        let stream = build_representative_target_resolved_stream(
            fixture.spec.ordinal,
            0,
            vec![0],
            vec![target],
        )?;
        representative_stream_pins.push(RepresentativeSecondMomentumStreamPin {
            operator_ordinal: stream.operator_spec.ordinal,
            fixture_sha256: stream.fixture_sha256,
            symmetric_momentum_channel_dynkin_label: stream
                .operator_spec
                .symmetric_momentum_channel_dynkin_label,
            symmetric_momentum_channel_copy: stream.operator_spec.symmetric_momentum_channel_copy,
            gauge_form_degree: stream.gauge_form_degree,
            parameter_components_selected: stream.parameter_components_selected,
            target_basis_ordinals_selected: stream.target_basis_ordinals_selected,
            emitted_entries: stream.entries.len(),
            stream_sha256: stream.stream_sha256,
            exact_normal_order_checked: stream.exact_normal_order_checked,
            synthetic_control_column: true,
            component_clebsch_gordan_map: false,
        });
    }

    let level12_component_fixture_corpus_complete = source_fixture_pins.len() == 41
        && available_level12_source_labels.len() == inventory.source_labels;
    let passed = inventory.representation_inventory_complete
        && inventory.new_operator_variables == 77
        && inventory.independent_symmetric_momentum_paths_for_10001 == 2
        && source_fixture_pins.len() == 6
        && available_level12_source_labels == ["10002", "30002", "40000"]
        && source_fixture_pins
            .iter()
            .all(|fixture| fixture.exact_raising_residuals_zero)
        && representative_stream_pins.len() == 3
        && representative_stream_pins.iter().all(|stream| {
            stream.emitted_entries > 0
                && stream.exact_normal_order_checked
                && stream.synthetic_control_column
                && !stream.component_clebsch_gordan_map
        })
        && !level12_component_fixture_corpus_complete;

    Ok(SecondMomentumVerificationReport {
        schema_version: "adynkra-11d-second-momentum-bounded-verification-v1".to_string(),
        role: "exact representation inventory, pinned available level-12 source fixtures, and synthetic target-resolved p^2 D^13 Lambda control streams".to_string(),
        representation_inventory: inventory,
        available_level12_source_fixture_pins: source_fixture_pins,
        available_level12_source_labels,
        available_level12_kernel_copies: 6,
        expected_level12_source_labels: 19,
        expected_level12_kernel_copies: 41,
        level12_component_fixture_corpus_complete,
        representative_stream_pins,
        representative_stream_benchmark_workload:
            "three degree-0, parameter-0, single-target synthetic control columns; observed debug test time is intentionally excluded from the deterministic artifact"
                .to_string(),
        wall_clock_benchmark_in_artifact: false,
        p2_d13_wedge_control_stream_complete: true,
        p3_d11_contraction_stream_complete: false,
        component_clebsch_gordan_maps_complete: false,
        inherited_old_column_p2_contractions_kept_separate: true,
        generic_k_solved: false,
        all_six_physical_fag_channels_checked: false,
        physical_fag_established: false,
        passed,
        result: "The 77-variable second-momentum representation inventory is exact. Six available level-12 source kernels are pinned and have zero exact raising residuals. Three synthetic p^2 D^13 target-stream controls are deterministic, but they are not component Clebsch-Gordan maps.".to_string(),
        boundary: "The component fixture corpus is incomplete, and the synthetic control columns are not physical operator columns. The p^3 D^11 contraction stream, physical K, complete F, full parameter and target projections, and generic polynomial F A G_p remain open.".to_string(),
    })
}

/// Atomically write the deterministic bounded verification artifact.
pub fn write_artifact(path: &Path) -> io::Result<SecondMomentumVerificationReport> {
    let report = verify()?;
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
    fn representation_inventory_is_exact_while_component_fixtures_are_bounded() {
        let inventory = second_momentum_representation_inventory();
        assert_eq!(inventory.source_labels, 19);
        assert_eq!(inventory.source_intermediate_pairs, 35);
        assert_eq!(inventory.source_copy_incidences, 73);
        assert_eq!(inventory.new_operator_variables, 77);
        assert_eq!(inventory.independent_symmetric_momentum_paths_for_10001, 2);
        assert!(inventory.representation_inventory_complete);
        assert!(!inventory.component_clebsch_gordan_maps_complete);
        assert_eq!(
            inventory
                .symmetric_square_times_target_channels
                .iter()
                .map(|channel| (channel.dynkin_label.as_str(), channel.multiplicity))
                .collect::<Vec<_>>(),
            vec![
                ("00001", 1),
                ("01001", 1),
                ("10001", 2),
                ("11001", 1),
                ("20001", 1),
                ("30001", 1),
            ]
        );
        let fixtures = representative_second_momentum_fixtures();
        assert_eq!(inventory.inventory_sha256, REPRESENTATION_INVENTORY_SHA256);
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| fixture.fixture_sha256.as_str())
                .collect::<Vec<_>>(),
            REPRESENTATIVE_FIXTURE_SHA256
        );
        assert_eq!(fixtures.len(), 3);
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| fixture.spec.ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        for fixture in &fixtures {
            validate_representative_fixture(fixture).unwrap();
            assert_eq!(fixture.spec.momentum_degree, 2);
            assert_eq!(fixture.spec.exterior_derivative_order, 12);
            assert!(fixture.spec.representation_inventory_complete);
            assert!(!fixture.spec.component_clebsch_gordan_fixture_complete);
            assert!(fixture.terms.iter().all(|term| {
                term.momentum_pair[0] <= term.momentum_pair[1]
                    && term.exterior_mask.count_ones() == 12
            }));
        }
    }

    #[test]
    fn exterior_right_wedge_normal_order_is_exact() {
        let mask = (1_u32 << 1) | (1_u32 << 4) | (1_u32 << 7);
        assert_eq!(
            crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(mask, 3),
            Some((mask | (1_u32 << 3), 1))
        );
        assert_eq!(
            crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(mask, 5),
            Some((mask | (1_u32 << 5), -1))
        );
        assert_eq!(
            crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(mask, 4),
            None
        );
        assert_eq!(
            crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(mask, 32),
            None
        );
    }

    #[test]
    fn available_level12_binary_source_fixtures_are_pinned_and_consumed() {
        let fixtures = available_level12_source_kernel_fixtures();
        assert_eq!(fixtures.len(), 6);
        assert!(fixtures.iter().all(|fixture| {
            matches!(fixture.dynkin_label.as_str(), "30002" | "10002" | "40000")
                && fixture.exterior_degree == 12
                && fixture.coefficient_width_bytes == 2
                && fixture.binary_sha256 == fixture.expected_binary_sha256
                && fixture.binary_contract_validated
                && fixture.exact_raising_residuals_zero
                && !fixture.component_momentum_clebsch_gordan_map_available
                && fixture.terms.len() == fixture.nonzero_coefficients
                && fixture
                    .terms
                    .iter()
                    .all(|term| term.exterior_mask.count_ones() == 12 && term.coefficient != 0)
        }));
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| fixture.copy)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 1, 2, 1]
        );
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| fixture.source_columns)
                .collect::<Vec<_>>(),
            vec![2_892, 2_892, 2_892, 56_758, 56_758, 9_376]
        );
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| fixture.maximum_absolute_coefficient)
                .collect::<Vec<_>>(),
            vec![56, 2, 1_760, 15_840, 3, 7]
        );
        assert_eq!(
            fixtures
                .iter()
                .filter(|fixture| fixture.representative_role.contains("trace/STT"))
                .count(),
            2
        );
    }

    #[test]
    fn fixture_hashes_fail_closed_under_mutation() {
        let mut fixture = representative_second_momentum_fixtures().remove(1);
        let original_hash = fixture.fixture_sha256.clone();
        fixture.terms[0].momentum_pair = [1, 0];
        assert_ne!(
            sha256_json(&FixtureHashPayload {
                spec: &fixture.spec,
                terms: &fixture.terms,
            }),
            original_hash
        );
        assert!(validate_representative_fixture(&fixture).is_err());
    }

    #[test]
    fn representative_target_stream_is_deterministic_and_bounded() {
        let first = build_representative_target_resolved_stream(0, 0, vec![0], vec![0]).unwrap();
        let second = build_representative_target_resolved_stream(0, 0, vec![0], vec![0]).unwrap();
        assert_eq!(first, second);
        assert!(!first.entries.is_empty());
        assert!(first.entries.iter().all(|entry| {
            entry.momentum_pair == [0, 0]
                && entry.exterior_mask.count_ones() == 13
                && entry.parameter_component_index == 0
                && entry.target_basis_ordinal == 0
                && entry.coefficient.denominator > 0
        }));
        assert!(first.exact_normal_order_checked);
        assert!(first.representative_fixture_only);
        assert!(first.representation_inventory_complete);
        assert!(!first.component_clebsch_gordan_maps_complete);
        assert_eq!(first.second_momentum_new_operator_variables, 77);
        assert_eq!(first.inherited_old_coefficient_variables, 49);
        assert!(first.new_variables_independent_of_inherited_old_variables);
        assert!(first.p2_d13_wedge_stream_implemented);
        assert!(!first.p3_d11_contraction_stream_implemented);
        assert!(first.inherited_old_column_p2_contractions_kept_separate);
        assert!(!first.full_target_projection_complete);
        assert!(!first.full_physical_fag_established);
    }

    #[test]
    fn stream_hash_detects_exact_coefficient_mutation() {
        let report = build_representative_target_resolved_stream(1, 0, vec![0], vec![1]).unwrap();
        assert!(!report.entries.is_empty());
        let mut entries = report.entries.clone();
        entries[0].coefficient.real_numerator += 1;
        assert_ne!(
            hash_second_momentum_stream(
                &report.operator_spec,
                &report.fixture_sha256,
                report.gauge_form_degree,
                &report.parameter_components_selected,
                &report.target_basis_ordinals_selected,
                &entries,
            ),
            report.stream_sha256
        );
    }

    #[test]
    fn bounded_verification_report_is_fail_closed_and_writable() {
        let report = verify().unwrap();
        assert!(report.passed);
        assert_eq!(report.representation_inventory.new_operator_variables, 77);
        assert_eq!(report.available_level12_source_fixture_pins.len(), 6);
        assert_eq!(report.representative_stream_pins.len(), 3);
        assert!(!report.level12_component_fixture_corpus_complete);
        assert!(report.p2_d13_wedge_control_stream_complete);
        assert!(!report.p3_d11_contraction_stream_complete);
        assert!(!report.component_clebsch_gordan_maps_complete);
        assert!(!report.generic_k_solved);
        assert!(!report.all_six_physical_fag_channels_checked);
        assert!(!report.physical_fag_established);
        assert!(
            report
                .representative_stream_pins
                .iter()
                .all(|stream| stream.synthetic_control_column
                    && !stream.component_clebsch_gordan_map)
        );

        let path = std::env::temp_dir().join(format!(
            "adynkra-11d-second-momentum-{}.json",
            std::process::id()
        ));
        let written = write_artifact(&path).unwrap();
        let reparsed: SecondMomentumVerificationReport =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(written, report);
        assert_eq!(reparsed, report);
        fs::remove_file(path).unwrap();
    }
}
