//! Fixed work list and representation-level gates for the 11D level-16
//! source-to-vector-spinor coupling certificates.

use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TARGET_DYNKIN_LABEL: &str = "10001";
const GOLDEN_COMMIT: &str = "89f20fc";
const SCALAR_BRIDGE_VECTOR_SPINOR_FIXTURE: &[u8] =
    include_bytes!("../data/eleven_dimensional_bridge/10001_highest_weight_kernel.i16le");
type Weight = [i8; 5];

const SIMPLE_ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];
const TARGET_WEIGHT: Weight = [3, 1, 1, 1, 1];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CouplingProblem {
    exterior_degree: u8,
    target_dynkin_label: &'static str,
    target_weight: Weight,
    schema_prefix: &'static str,
}

const LEVEL16_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 16,
    target_dynkin_label: TARGET_DYNKIN_LABEL,
    target_weight: TARGET_WEIGHT,
    schema_prefix: "adynkra-11d-level16",
};

const LEVEL17_HOOK_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 17,
    target_dynkin_label: "11000",
    target_weight: [4, 2, 0, 0, 0],
    schema_prefix: "adynkra-11d-level17-hook",
};

const FIRST_MOMENTUM_00001_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 14,
    target_dynkin_label: "00001",
    target_weight: [1, 1, 1, 1, 1],
    schema_prefix: "adynkra-11d-first-momentum-00001",
};

const FIRST_MOMENTUM_01001_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 14,
    target_dynkin_label: "01001",
    target_weight: [3, 3, 1, 1, 1],
    schema_prefix: "adynkra-11d-first-momentum-01001",
};

const FIRST_MOMENTUM_10001_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 14,
    target_dynkin_label: "10001",
    target_weight: [3, 1, 1, 1, 1],
    schema_prefix: "adynkra-11d-first-momentum-10001",
};

const FIRST_MOMENTUM_20001_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 14,
    target_dynkin_label: "20001",
    target_weight: [5, 1, 1, 1, 1],
    schema_prefix: "adynkra-11d-first-momentum-20001",
};

const SECOND_MOMENTUM_20001_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 12,
    target_dynkin_label: "20001",
    target_weight: [5, 1, 1, 1, 1],
    schema_prefix: "adynkra-11d-second-momentum-20001",
};

const SECOND_MOMENTUM_30001_PROBLEM: CouplingProblem = CouplingProblem {
    exterior_degree: 12,
    target_dynkin_label: "30001",
    target_weight: [7, 1, 1, 1, 1],
    schema_prefix: "adynkra-11d-second-momentum-30001",
};

fn second_momentum_problem(target_dynkin_label: &str) -> CouplingProblem {
    let target_dynkin_label: &'static str = match target_dynkin_label {
        "00001" => "00001",
        "01001" => "01001",
        "10001" => "10001",
        "11001" => "11001",
        "20001" => "20001",
        "30001" => "30001",
        _ => panic!("unknown second-momentum target {target_dynkin_label}"),
    };
    CouplingProblem {
        exterior_degree: 12,
        target_dynkin_label,
        target_weight: dynkin_highest_weight_for_label(target_dynkin_label),
        schema_prefix: match target_dynkin_label {
            "00001" => "adynkra-11d-second-momentum-00001",
            "01001" => "adynkra-11d-second-momentum-01001",
            "10001" => "adynkra-11d-second-momentum-10001",
            "11001" => "adynkra-11d-second-momentum-11001",
            "20001" => "adynkra-11d-second-momentum-20001",
            "30001" => "adynkra-11d-second-momentum-30001",
            _ => unreachable!(),
        },
    }
}

fn level18_problem(target_dynkin_label: &str) -> CouplingProblem {
    let target_dynkin_label: &'static str = match target_dynkin_label {
        "01001" => "01001",
        "10001" => "10001",
        "11001" => "11001",
        "20001" => "20001",
        _ => panic!("unknown level-18 target {target_dynkin_label}"),
    };
    CouplingProblem {
        exterior_degree: 18,
        target_dynkin_label,
        target_weight: dynkin_highest_weight_for_label(target_dynkin_label),
        schema_prefix: "adynkra-11d-level18-embedded",
    }
}

fn dynkin_highest_weight_for_label(label: &str) -> Weight {
    let labels = label
        .bytes()
        .map(|byte| i8::try_from(byte - b'0').expect("B5 Dynkin digit fits i8"))
        .collect::<Vec<_>>();
    assert_eq!(labels.len(), 5, "B5 Dynkin label has five digits");
    std::array::from_fn(|index| 2 * labels[index..4].iter().sum::<i8>() + labels[4])
}

fn first_momentum_problem(target_dynkin_label: &str) -> CouplingProblem {
    match target_dynkin_label {
        "00001" => FIRST_MOMENTUM_00001_PROBLEM,
        "01001" => FIRST_MOMENTUM_01001_PROBLEM,
        "10001" => FIRST_MOMENTUM_10001_PROBLEM,
        "20001" => FIRST_MOMENTUM_20001_PROBLEM,
        _ => panic!("unknown first-momentum target {target_dynkin_label}"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Level16FixtureManifestEntry {
    pub source_dynkin_label: &'static str,
    pub copy: usize,
    pub artifact: &'static str,
    pub byte_length: usize,
    pub coefficient_count: usize,
    pub signed_little_endian_bits: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorMultiplicityAudit {
    pub source_dynkin_label: &'static str,
    pub target_dynkin_label: &'static str,
    pub target_multiplicity_in_source_tensor_spinor: usize,
    pub multiplicity_one: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Level16CouplingPrecheckReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub exterior_degree: usize,
    pub spinor_dimension: usize,
    pub target_dynkin_label: &'static str,
    pub distinct_source_irreps: usize,
    pub expected_distinct_source_irreps: usize,
    pub embedded_source_copies: usize,
    pub expected_embedded_source_copies: usize,
    pub fixtures: Vec<Level16FixtureManifestEntry>,
    pub copy_counts_by_irrep: BTreeMap<&'static str, usize>,
    pub tensor_multiplicities: Vec<TensorMultiplicityAudit>,
    pub every_target_multiplicity_is_one: bool,
    pub golden_source_dynkin_label: &'static str,
    pub golden_source_copy: usize,
    pub golden_commit: &'static str,
    pub experimentally_validated_source_dynkin_label: &'static str,
    pub experimentally_validated_source_copy: usize,
    pub experimentally_validated_checkpoint_present: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Level17HookCouplingPrecheckReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub exterior_degree: usize,
    pub spinor_dimension: usize,
    pub target_dynkin_label: &'static str,
    pub distinct_source_irreps: usize,
    pub embedded_source_copies: usize,
    pub copy_counts_by_irrep: BTreeMap<&'static str, usize>,
    pub tensor_multiplicities: Vec<TensorMultiplicityAudit>,
    pub every_target_multiplicity_is_one: bool,
    pub passed: bool,
}

pub fn verify() -> Level16CouplingPrecheckReport {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut copy_counts_by_irrep = BTreeMap::new();
    for fixture in &fixtures {
        *copy_counts_by_irrep
            .entry(fixture.dynkin_label)
            .or_insert(0) += 1;
    }
    let tensor_multiplicities = copy_counts_by_irrep
        .keys()
        .copied()
        .map(|source_dynkin_label| {
            let target_multiplicity_in_source_tensor_spinor =
                crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
                    .iter()
                    .filter(|(target, _)| target == TARGET_DYNKIN_LABEL)
                    .count();
            TensorMultiplicityAudit {
                source_dynkin_label,
                target_dynkin_label: TARGET_DYNKIN_LABEL,
                target_multiplicity_in_source_tensor_spinor,
                multiplicity_one: target_multiplicity_in_source_tensor_spinor == 1,
            }
        })
        .collect::<Vec<_>>();
    let every_target_multiplicity_is_one = tensor_multiplicities
        .iter()
        .all(|audit| audit.multiplicity_one);
    let manifest = fixtures
        .iter()
        .map(|fixture| Level16FixtureManifestEntry {
            source_dynkin_label: fixture.dynkin_label,
            copy: fixture.copy,
            artifact: fixture.artifact,
            byte_length: fixture.bytes.len(),
            coefficient_count: fixture.bytes.len() / 2,
            signed_little_endian_bits: 16,
        })
        .collect::<Vec<_>>();
    let expected_counts = BTreeMap::from([
        ("00002", 1),
        ("00010", 2),
        ("00100", 2),
        ("10000", 1),
        ("10002", 3),
        ("10010", 1),
        ("10100", 1),
        ("20000", 1),
    ]);
    let fixture_encoding_valid = fixtures
        .iter()
        .all(|fixture| !fixture.bytes.is_empty() && fixture.bytes.len() % 2 == 0);
    let distinct_source_irreps = copy_counts_by_irrep.len();
    let embedded_source_copies = fixtures.len();
    let passed = distinct_source_irreps == 8
        && embedded_source_copies == 12
        && copy_counts_by_irrep == expected_counts
        && fixture_encoding_valid
        && every_target_multiplicity_is_one;
    Level16CouplingPrecheckReport {
        schema_version: "adynkra-11d-level16-coupling-precheck-v1",
        role: "fixed source manifest and multiplicity-one gate for level-16 couplings into (10001)",
        exterior_degree: 16,
        spinor_dimension: 32,
        target_dynkin_label: TARGET_DYNKIN_LABEL,
        distinct_source_irreps,
        expected_distinct_source_irreps: 8,
        embedded_source_copies,
        expected_embedded_source_copies: 12,
        fixtures: manifest,
        copy_counts_by_irrep,
        tensor_multiplicities,
        every_target_multiplicity_is_one,
        golden_source_dynkin_label: "20000",
        golden_source_copy: 1,
        golden_commit: GOLDEN_COMMIT,
        experimentally_validated_source_dynkin_label: "00100",
        experimentally_validated_source_copy: 1,
        experimentally_validated_checkpoint_present: true,
        passed,
    }
}

pub fn verify_hook_precheck() -> Level17HookCouplingPrecheckReport {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures();
    let mut copy_counts_by_irrep = BTreeMap::new();
    for fixture in &fixtures {
        *copy_counts_by_irrep
            .entry(fixture.dynkin_label)
            .or_insert(0) += 1;
    }
    let tensor_multiplicities = copy_counts_by_irrep
        .keys()
        .copied()
        .map(|source_dynkin_label| {
            let target_multiplicity_in_source_tensor_spinor =
                crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
                    .iter()
                    .filter(|(target, _)| target == "11000")
                    .count();
            TensorMultiplicityAudit {
                source_dynkin_label,
                target_dynkin_label: "11000",
                target_multiplicity_in_source_tensor_spinor,
                multiplicity_one: target_multiplicity_in_source_tensor_spinor == 1,
            }
        })
        .collect::<Vec<_>>();
    let every_target_multiplicity_is_one = tensor_multiplicities
        .iter()
        .all(|audit| audit.multiplicity_one);
    let expected = BTreeMap::from([("01001", 2), ("10001", 1), ("11001", 3), ("20001", 1)]);
    let passed =
        copy_counts_by_irrep == expected && fixtures.len() == 7 && every_target_multiplicity_is_one;
    Level17HookCouplingPrecheckReport {
        schema_version: "adynkra-11d-level17-hook-coupling-precheck-v1",
        role: "fixed source manifest and multiplicity-one gate for level-17 couplings into (11000)",
        exterior_degree: 17,
        spinor_dimension: 32,
        target_dynkin_label: "11000",
        distinct_source_irreps: copy_counts_by_irrep.len(),
        embedded_source_copies: fixtures.len(),
        copy_counts_by_irrep,
        tensor_multiplicities,
        every_target_multiplicity_is_one,
        passed,
    }
}

#[derive(Debug, Clone)]
struct DenseState {
    weight: Weight,
    pbw_word: Vec<u8>,
    coefficients: Vec<i64>,
}

#[derive(Debug)]
struct WeightSpace {
    masks: Vec<u32>,
    index: HashMap<u32, usize>,
}

#[derive(Debug)]
struct CsrExteriorAction {
    target_weight: Weight,
    target_dimension: usize,
    source_offsets: Vec<usize>,
    destination_indices: Vec<usize>,
    signs: Vec<i8>,
}

#[derive(Debug)]
struct ExteriorModel {
    exterior_degree: u8,
    spinors: [Weight; 32],
    left: HashMap<(u8, Weight), Vec<u16>>,
    right: HashMap<(u8, Weight), Vec<u16>>,
    spaces: BTreeMap<Weight, WeightSpace>,
    actions: BTreeMap<(Weight, usize, bool), Arc<CsrExteriorAction>>,
    maximum_absolute_accumulator: i128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalDomainBasisEntry {
    pub free_spinor_index: usize,
    pub free_spinor_weight: Weight,
    pub source_weight: Weight,
    pub pbw_word_simple_roots: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractCouplingCertificate {
    pub schema_version: String,
    pub role: String,
    pub source_dynkin_label: String,
    pub source_fixture_copy: usize,
    pub target_dynkin_label: String,
    pub basis_method: String,
    pub dependency_test: String,
    pub product_weight_domain_dimension: usize,
    pub source_weight_spaces_used: usize,
    pub source_weight_multiplicities: Vec<(Weight, usize)>,
    pub domain_basis: Vec<CanonicalDomainBasisEntry>,
    pub gram_matrix_rank: usize,
    pub kernel_dimension: usize,
    pub primitive_domain_coefficients: Vec<i64>,
    pub primitive_coefficient_gcd: i64,
    pub maximum_absolute_primitive_coefficient: i64,
    pub exact_raising_residual_terms_by_simple_root: [usize; 5],
    pub maximum_absolute_checked_accumulator: i128,
    pub storage_type: String,
    pub exterior_action_storage: String,
    pub csr_actions_built: usize,
    pub csr_nonzero_entries: usize,
    pub multiplicity_one: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedCouplingCertificate {
    pub schema_version: String,
    pub role: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub target_dynkin_label: String,
    pub abstract_coupling_source_copy: usize,
    pub product_weight_domain_dimension: usize,
    pub primitive_domain_coefficients: Vec<i64>,
    pub coupled_nonzero_terms: usize,
    pub exact_raising_residual_terms_by_simple_root: [usize; 5],
    pub maximum_absolute_checked_accumulator: i128,
    pub shared_abstract_coupling_applied: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllCouplingCertificateReport {
    pub schema_version: String,
    pub role: String,
    pub abstract_couplings: Vec<AbstractCouplingCertificate>,
    pub embedded_copies: Vec<EmbeddedCouplingCertificate>,
    pub distinct_source_irreps_certified: usize,
    pub embedded_source_copies_certified: usize,
    pub expected_distinct_source_irreps: usize,
    pub expected_embedded_source_copies: usize,
    pub every_residual_is_exactly_zero: bool,
    pub execution_workers: usize,
    pub memory_budget_gib: usize,
    pub estimated_memory_gib_per_worker: usize,
    pub resumed_from_atomic_checkpoints: bool,
    pub boundary: String,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstMomentumCouplingCertificateReport {
    pub schema_version: String,
    pub role: String,
    pub abstract_couplings: Vec<AbstractCouplingCertificate>,
    pub embedded_maps: Vec<EmbeddedCouplingCertificate>,
    pub source_target_pairs_certified: usize,
    pub embedded_maps_certified: usize,
    pub expected_source_target_pairs: usize,
    pub expected_embedded_maps: usize,
    pub every_residual_is_exactly_zero: bool,
    pub execution_workers: usize,
    pub memory_budget_gib: usize,
    pub estimated_memory_gib_per_worker: usize,
    pub resumed_from_atomic_checkpoints: bool,
    pub boundary: String,
    pub passed: bool,
}

#[derive(Debug, Clone)]
struct CoupledDenseState {
    total_weight: Weight,
    components: BTreeMap<usize, DenseState>,
}

#[derive(Debug, Clone)]
struct CoupledSparseState {
    components: BTreeMap<usize, Vec<(u32, i128)>>,
}

#[derive(Debug, Clone)]
struct CoupledSparseState64 {
    components: BTreeMap<usize, Vec<(u32, i64)>>,
}

pub(crate) struct CanonicalSparseHighest64 {
    state: CoupledSparseState64,
    maximum_absolute_coefficient: u64,
}

impl CanonicalSparseHighest64 {
    pub(crate) fn term_count(&self) -> usize {
        self.state.components.values().map(Vec::len).sum()
    }

    pub(crate) fn maximum_absolute_coefficient(&self) -> u64 {
        self.maximum_absolute_coefficient
    }

    pub(crate) fn visit_terms<F>(&self, mut visit: F) -> io::Result<()>
    where
        F: FnMut(u64, i64) -> io::Result<()>,
    {
        let mut previous_key = None;
        let mut observed_maximum = 0_u64;
        for (&free_spinor, values) in &self.state.components {
            if free_spinor >= 32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "canonical sparse highest has an invalid free-spinor index",
                ));
            }
            for &(mask, coefficient) in values {
                let key = (free_spinor as u64) << 32 | u64::from(mask);
                if coefficient == 0
                    || mask.count_ones() != 12
                    || previous_key.is_some_and(|previous| previous >= key)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "canonical sparse highest ordering or degree invariant failed",
                    ));
                }
                visit(key, coefficient)?;
                observed_maximum = observed_maximum.max(coefficient.unsigned_abs());
                previous_key = Some(key);
            }
        }
        if observed_maximum != self.maximum_absolute_coefficient {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "canonical sparse highest maximum-coefficient invariant failed",
            ));
        }
        Ok(())
    }
}

/// Build the accelerator boundary object from an independently certified,
/// already canonical highest-weight stream.
pub(crate) fn canonical_sparse_highest64_from_terms(
    terms: impl IntoIterator<Item = (usize, u32, i64)>,
) -> io::Result<CanonicalSparseHighest64> {
    let mut components = BTreeMap::<usize, Vec<(u32, i64)>>::new();
    let mut previous_key = None;
    let mut maximum_absolute_coefficient = 0_u64;
    for (free_spinor, mask, coefficient) in terms {
        let key = (free_spinor as u64) << 32 | u64::from(mask);
        if free_spinor >= 32
            || mask.count_ones() != 12
            || coefficient == 0
            || previous_key.is_some_and(|previous| previous >= key)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "independent highest-weight stream is not canonical",
            ));
        }
        previous_key = Some(key);
        maximum_absolute_coefficient = maximum_absolute_coefficient.max(coefficient.unsigned_abs());
        components
            .entry(free_spinor)
            .or_default()
            .push((mask, coefficient));
    }
    if components.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "independent highest-weight stream is empty",
        ));
    }
    let highest = CanonicalSparseHighest64 {
        state: CoupledSparseState64 { components },
        maximum_absolute_coefficient,
    };
    highest.visit_terms(|_, _| Ok(()))?;
    Ok(highest)
}

struct VerifiedSparseHighest64 {
    highest: CanonicalSparseHighest64,
    maximum_absolute_checked_accumulator: i128,
    source_fixture_sha256: String,
    coupled_map_sha256: String,
    estimated_payload_bytes: u64,
}

type SparseLoweredComponent = (usize, Vec<(u32, i64)>, i128);

#[derive(Debug, Clone)]
struct MomentumCoupledDenseState {
    total_weight: Weight,
    components: BTreeMap<(usize, usize), DenseState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MomentumHookEntry {
    pub momentum_vector_index: usize,
    pub free_spinor_index: usize,
    pub exterior_mask: u32,
    pub real: i128,
    pub imaginary: i128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroMomentumGaugeCompositionEntry {
    pub parameter_component_index: usize,
    pub exterior_mask: u32,
    pub real: i128,
    pub imaginary: i128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstMomentumGaugeCompositionEntry {
    pub parameter_component_index: usize,
    pub momentum_vector_index: usize,
    pub exterior_mask: u32,
    pub real: i128,
    pub imaginary: i128,
}

/// One exact term of the target-resolved adjoint composition stream.
///
/// `target_vector_weight_index` uses the B5 weight basis
/// `(+e_1,-e_1,...,+e_5,-e_5,0)`, not the Cartesian Clifford basis.  Terms
/// with identical keys must be summed by the consumer.  The rational target
/// dual-basis coefficient is multiplied into the Gaussian-integer source
/// residual before emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetResolvedGaugeCompositionEntry {
    pub target_basis_ordinal: usize,
    pub target_vector_weight_index: usize,
    pub target_spinor_weight_index: usize,
    pub parameter_component_index: usize,
    pub momentum_vector_weight_index: Option<usize>,
    pub exterior_mask: u32,
    pub real: Ratio<BigInt>,
    pub imaginary: Ratio<BigInt>,
}

/// Allocation-free primitive form of a target-resolved first-momentum term.
/// The exact coefficient is `*_numerator / denominator`. This avoids creating
/// millions of temporary `BigInt` ratios when a consumer immediately projects
/// the stream into a bounded exact accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetResolvedPrimitiveGaugeCompositionEntry {
    pub target_basis_ordinal: usize,
    pub target_vector_weight_index: usize,
    pub target_spinor_weight_index: usize,
    pub parameter_component_index: usize,
    pub momentum_vector_weight_index: Option<usize>,
    pub exterior_mask: u32,
    pub real_numerator: i128,
    pub imaginary_numerator: i128,
    pub denominator: i64,
}

/// One exact component of a requested descendant of a certified level-12
/// source-times-spinor map into `(20001)`.
///
/// The request ordinal refers to the caller-provided PBW-word list.  The
/// exterior mask is in the canonical degree-12 spinor-weight basis and the
/// free spinor index uses the same 32-state weight basis as the bridge code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SecondMomentum20001DescendantEntry {
    pub requested_word_ordinal: usize,
    pub free_spinor_weight_index: usize,
    pub exterior_mask: u32,
    pub coefficient: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SecondMomentum20001DescendantEvent {
    WordLoweringStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    WordStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    Component(SecondMomentum20001DescendantEntry),
    WordEnd {
        requested_word_ordinal: usize,
        emitted_nonzero_components: u64,
    },
}

/// Proof and resource accounting for one requested `(20001)` descendant
/// materialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SecondMomentum20001DescendantAccounting {
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coupled_map_sha256: String,
    pub requested_pbw_words: usize,
    pub emitted_nonzero_components: u64,
    pub maximum_absolute_checked_accumulator: i128,
    pub estimated_payload_bytes: u64,
    pub checkpoint_hash_parity_verified: bool,
}

/// One exact component of a requested descendant of a certified level-12
/// source-times-spinor map into `(30001)`.
///
/// The request ordinal refers to the caller-provided PBW-word list.  The
/// exterior mask is in the canonical degree-12 spinor-weight basis and the
/// free spinor index uses the same 32-state weight basis as the bridge code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SecondMomentum30001DescendantEntry {
    pub requested_word_ordinal: usize,
    pub free_spinor_weight_index: usize,
    pub exterior_mask: u32,
    pub coefficient: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SecondMomentum30001DescendantEvent {
    WordLoweringStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    WordStart {
        requested_word_ordinal: usize,
        pbw_word_simple_roots: Vec<u8>,
    },
    Component(SecondMomentum30001DescendantEntry),
    WordEnd {
        requested_word_ordinal: usize,
        emitted_nonzero_components: u64,
    },
}

/// Proof and resource accounting for one requested `(30001)` descendant
/// materialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SecondMomentum30001DescendantAccounting {
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coupled_map_sha256: String,
    pub requested_pbw_words: usize,
    pub emitted_nonzero_components: u64,
    pub maximum_absolute_checked_accumulator: i128,
    pub estimated_payload_bytes: u64,
    pub checkpoint_hash_parity_verified: bool,
}

/// Proof and output accounting for an opaque lowering backend.
///
/// Unlike the concrete CPU accounting, this intentionally omits the maximum
/// intermediate coefficient. An opaque backend need not expose intermediate
/// handles, so claiming that diagnostic without a backend proof would be
/// misleading.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OpaqueSecondMomentumDescendantAccounting {
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub source_fixture: String,
    pub source_fixture_sha256: String,
    pub coupled_map_sha256: String,
    pub requested_pbw_words: usize,
    pub emitted_nonzero_components: u64,
    pub estimated_host_payload_bytes: u64,
    pub checkpoint_hash_parity_verified: bool,
}

/// Accounting returned by the operator-major first-momentum stream.
///
/// The byte count covers all variable-length payloads owned by the exterior
/// model, its cached lowering actions, and the coupled target states. Hash
/// table allocator metadata is implementation-defined, so the count is an
/// auditable payload estimate rather than a process-RSS measurement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SharedFirstMomentumStateAccounting {
    pub operator_ordinal: usize,
    pub selected_gauge_form_degrees: Vec<usize>,
    pub coupled_state_materializations: usize,
    pub estimated_payload_bytes: u64,
    pub configured_payload_limit_bytes: u64,
    pub payload_limit_respected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JointColumnSpec {
    pub ordinal: usize,
    pub label: String,
    pub kind: String,
    pub source_dynkin_label: String,
    pub source_copy: usize,
    pub intermediate_dynkin_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointColumnFunctionalFile {
    pub schema_version: String,
    pub ordinal: usize,
    pub label: String,
    pub values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointColumnArtifactManifest {
    pub schema_version: String,
    pub passed: bool,
    pub spec: JointColumnSpec,
    pub nonzero_residual_entries: u64,
    pub maximum_absolute_residual_coefficient: String,
    pub exact_functional_values: usize,
    pub raw_record_bytes: usize,
    pub raw_uncompressed_bytes: u64,
    pub raw_compressed_bytes: u64,
    pub raw_uncompressed_sha256: String,
    pub raw_compressed_sha256: String,
    pub functional_file_sha256: String,
    pub fixture_sha256: String,
    pub source_revision: String,
    pub executable_sha256: String,
    pub host: String,
    pub process_id: u32,
    pub started_unix_seconds: u64,
    pub finished_unix_seconds: u64,
    pub elapsed_milliseconds: u128,
    pub raw_file: String,
    pub functional_file: String,
    pub convention: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointCompatibilityMatrixReport {
    pub schema_version: String,
    pub role: String,
    pub leading_basis: Vec<String>,
    pub first_momentum_basis: Vec<String>,
    pub hook_rows: usize,
    pub momentum_coordinate_rows: usize,
    pub coefficient_columns: usize,
    pub leading_columns: usize,
    pub first_momentum_columns: usize,
    pub reciprocal_couplings_verified: usize,
    pub reciprocal_coupling_intermediates: Vec<String>,
    pub reciprocal_coupling_domain_dimensions: Vec<usize>,
    pub reciprocal_coupling_kernel_dimensions: Vec<usize>,
    pub reciprocal_coupling_raising_residuals: Vec<usize>,
    pub exact_functional_rows: usize,
    pub exact_functional_matrix_rank: usize,
    pub exact_functional_nullity: usize,
    pub full_rank_certified_by_functional_minor: bool,
    pub exact_joint_nullity: Option<usize>,
    pub functional_kernel_leading_projection_rank: usize,
    pub leading_extension_excluded: bool,
    pub previous_hook_kernel_dimension: usize,
    pub previous_hook_kernel_subspace_dimension_extended: Option<usize>,
    pub scalar_factorizing_direction_extends: Option<bool>,
    pub direct_spinor_quotient_dimension_extended: Option<usize>,
    pub functional_primitive_integer_kernel_basis: Vec<Vec<String>>,
    pub functional_kernel_residuals_exactly_zero: bool,
    pub exact_functional_normal_matrix: Vec<Vec<RationalMatrixEntry>>,
    pub maximum_absolute_residual_coefficient: i128,
    pub maximum_absolute_normal_matrix_numerator: String,
    pub convention: String,
    pub interpretation: String,
    pub boundary: String,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RationalMatrixEntry {
    pub numerator: String,
    pub denominator: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level17DerivativeMatrixReport {
    pub schema_version: String,
    pub role: String,
    pub source_basis: Vec<String>,
    pub hook_basis: Vec<String>,
    pub target_hook_dynkin_label: String,
    pub target_coupling_terms: usize,
    pub target_coupling_primitive_coefficients: Vec<i64>,
    pub hook_gram_rank: usize,
    pub derivative_matrix_rank: usize,
    pub derivative_matrix_nullity: usize,
    pub matrix_rows_by_hook_columns_by_source: Vec<Vec<RationalMatrixEntry>>,
    pub primitive_integer_kernel_basis: Vec<Vec<String>>,
    pub kernel_residuals_exactly_zero: bool,
    pub kernel_coefficient_mutation_detected: bool,
    pub leading_gram_rank: usize,
    pub scalar_factorizing_coordinates: Vec<RationalMatrixEntry>,
    pub scalar_factorizing_reconstruction_residual_norm: RationalMatrixEntry,
    pub scalar_factorizing_direction_is_in_leading_span: bool,
    pub scalar_factorizing_hook_image: Vec<RationalMatrixEntry>,
    pub scalar_factorizing_hook_image_is_zero: bool,
    pub exact_reconstruction_residual_norms: Vec<RationalMatrixEntry>,
    pub every_derivative_column_is_in_hook_span: bool,
    pub maximum_absolute_checked_accumulator: i128,
    pub convention: String,
    pub interpretation: String,
    pub boundary: String,
    pub passed: bool,
}

impl ExteriorModel {
    fn new(exterior_degree: u8) -> Self {
        let spinors = spinor_weights();
        Self {
            exterior_degree,
            spinors,
            left: half_groups(0, &spinors),
            right: half_groups(16, &spinors),
            spaces: BTreeMap::new(),
            actions: BTreeMap::new(),
            maximum_absolute_accumulator: 0,
        }
    }

    fn space(&mut self, weight: Weight) -> &WeightSpace {
        self.spaces.entry(weight).or_insert_with(|| {
            let masks = weight_basis(self.exterior_degree, weight, &self.left, &self.right);
            let index = masks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, mask)| (mask, index))
                .collect();
            WeightSpace { masks, index }
        })
    }

    fn action(
        &mut self,
        source_weight: Weight,
        root: usize,
        raising: bool,
    ) -> Arc<CsrExteriorAction> {
        let key = (source_weight, root, raising);
        if let Some(action) = self.actions.get(&key) {
            return Arc::clone(action);
        }
        let target_weight = if raising {
            add(source_weight, SIMPLE_ROOTS[root])
        } else {
            subtract(source_weight, SIMPLE_ROOTS[root])
        };
        let source_masks = self.space(source_weight).masks.clone();
        let target_index = self.space(target_weight).index.clone();
        let mut source_offsets = Vec::with_capacity(source_masks.len() + 1);
        let mut destination_indices = Vec::new();
        let mut signs = Vec::new();
        source_offsets.push(0);
        for source_mask in source_masks {
            for occupied_index in 0..32 {
                if source_mask & (1_u32 << occupied_index) == 0 {
                    continue;
                }
                let replacement_index = if raising {
                    raised_spinor_index(occupied_index, root, &self.spinors)
                } else {
                    lowered_spinor_index(occupied_index, root, &self.spinors)
                };
                let Some(replacement_index) = replacement_index else {
                    continue;
                };
                if source_mask & (1_u32 << replacement_index) != 0 {
                    continue;
                }
                let output_mask =
                    (source_mask ^ (1_u32 << occupied_index)) | (1_u32 << replacement_index);
                destination_indices.push(
                    *target_index
                        .get(&output_mask)
                        .expect("exterior action left its target weight space"),
                );
                signs.push(
                    i8::try_from(exterior_replacement_sign(
                        source_mask,
                        occupied_index,
                        replacement_index,
                    ))
                    .unwrap(),
                );
            }
            source_offsets.push(destination_indices.len());
        }
        let action = Arc::new(CsrExteriorAction {
            target_weight,
            target_dimension: target_index.len(),
            source_offsets,
            destination_indices,
            signs,
        });
        self.actions.insert(key, Arc::clone(&action));
        action
    }

    fn apply_action(&mut self, source: &DenseState, root: usize, raising: bool) -> DenseState {
        let action = self.action(source.weight, root, raising);
        let mut accumulator = vec![0_i128; action.target_dimension];
        for (source_index, coefficient) in source.coefficients.iter().copied().enumerate() {
            if coefficient == 0 {
                continue;
            }
            for edge in action.source_offsets[source_index]..action.source_offsets[source_index + 1]
            {
                let destination = action.destination_indices[edge];
                accumulator[destination] = accumulator[destination]
                    .checked_add(i128::from(action.signs[edge]) * i128::from(coefficient))
                    .expect("i128 overflow in exact CSR exterior action");
                self.maximum_absolute_accumulator = self
                    .maximum_absolute_accumulator
                    .max(accumulator[destination].abs());
            }
        }
        let coefficients = accumulator
            .into_iter()
            .map(|value| {
                i64::try_from(value).expect("CSR exterior coefficient exceeds i64 storage")
            })
            .collect();
        let mut pbw_word = source.pbw_word.clone();
        if !raising {
            pbw_word.push(u8::try_from(root + 1).unwrap());
        }
        DenseState {
            weight: action.target_weight,
            pbw_word,
            coefficients,
        }
    }

    fn fixture_state(
        &mut self,
        dynkin_label: &str,
        fixture_bytes: &[u8],
        coefficient_width_bytes: usize,
    ) -> DenseState {
        let weight = dynkin_highest_weight(dynkin_label);
        let expected = self.space(weight).masks.len();
        let coefficients = decode_kernel(fixture_bytes, coefficient_width_bytes);
        assert_eq!(
            coefficients.len(),
            expected,
            "fixture length does not match canonical mask basis for {dynkin_label}"
        );
        DenseState {
            weight,
            pbw_word: Vec::new(),
            coefficients,
        }
    }

    fn lower(&mut self, source: &DenseState, root: usize) -> DenseState {
        self.apply_action(source, root, false)
    }

    fn raise_coefficients(&mut self, source: &DenseState, root: usize) -> (Weight, Vec<i64>) {
        let raised = self.apply_action(source, root, true);
        (raised.weight, raised.coefficients)
    }
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

fn add(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn subtract(left: Weight, right: Weight) -> Weight {
    std::array::from_fn(|axis| left[axis] - right[axis])
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
    let mut groups = HashMap::<(u8, Weight), Vec<u16>>::new();
    for mask in 0_u32..=u32::from(u16::MAX) {
        let mask = mask as u16;
        groups
            .entry((mask.count_ones() as u8, mask_weight(mask, offset, weights)))
            .or_default()
            .push(mask);
    }
    groups
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
            if let Some(right_masks) = right.get(&(right_degree, subtract(target, *left_weight))) {
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

/// Deterministic exterior-spinor weight basis used by exact binary kernel
/// fixtures at levels below sixteen.
pub(crate) fn exterior_highest_weight_basis_masks(
    exterior_degree: usize,
    dynkin_label: &str,
) -> Vec<u32> {
    assert!(exterior_degree <= 32);
    let weights = spinor_weights();
    let left = half_groups(0, &weights);
    let right = half_groups(16, &weights);
    weight_basis(
        u8::try_from(exterior_degree).unwrap(),
        dynkin_highest_weight(dynkin_label),
        &left,
        &right,
    )
}

/// Verify all five exact exterior-spinor raising residuals directly from a
/// sparse highest-weight fixture.
pub(crate) fn exterior_highest_weight_raising_residuals_are_zero(terms: &[(u32, i64)]) -> bool {
    let weights = spinor_weights();
    for root in 0..SIMPLE_ROOTS.len() {
        let mut residual = HashMap::<u32, i128>::new();
        for &(source_mask, coefficient) in terms {
            for occupied_index in 0..32 {
                if source_mask & (1_u32 << occupied_index) == 0 {
                    continue;
                }
                let Some(replacement_index) = raised_spinor_index(occupied_index, root, &weights)
                else {
                    continue;
                };
                if source_mask & (1_u32 << replacement_index) != 0 {
                    continue;
                }
                let output_mask =
                    (source_mask ^ (1_u32 << occupied_index)) | (1_u32 << replacement_index);
                let contribution = i128::from(coefficient)
                    * i128::from(exterior_replacement_sign(
                        source_mask,
                        occupied_index,
                        replacement_index,
                    ));
                *residual.entry(output_mask).or_insert(0) += contribution;
            }
        }
        if residual.values().any(|value| *value != 0) {
            return false;
        }
    }
    true
}

fn decode_kernel(bytes: &[u8], coefficient_width_bytes: usize) -> Vec<i64> {
    match coefficient_width_bytes {
        2 => bytes
            .chunks_exact(2)
            .map(|pair| i64::from(i16::from_le_bytes([pair[0], pair[1]])))
            .collect(),
        4 => bytes
            .chunks_exact(4)
            .map(|word| i64::from(i32::from_le_bytes([word[0], word[1], word[2], word[3]])))
            .collect(),
        _ => panic!("unsupported kernel coefficient width"),
    }
}

fn fixture_coefficient_width(artifact: &str) -> usize {
    if artifact.ends_with(".i32le") {
        4
    } else {
        assert!(artifact.ends_with(".i16le"));
        2
    }
}

fn raised_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = add(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn lowered_spinor_index(index: usize, root: usize, weights: &[Weight; 32]) -> Option<usize> {
    let target = subtract(weights[index], SIMPLE_ROOTS[root]);
    weights.iter().position(|weight| *weight == target)
}

fn exterior_replacement_sign(mask: u32, first: usize, second: usize) -> i64 {
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    let interval = if high == low + 1 {
        0
    } else {
        ((1_u32 << high) - 1) ^ ((1_u32 << (low + 1)) - 1)
    };
    if (mask & interval).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn right_wedge_sign(mask: u32, spinor_index: usize) -> Option<i128> {
    let bit = 1_u32 << spinor_index;
    if mask & bit != 0 {
        return None;
    }
    let greater_bits = if spinor_index == 31 {
        0
    } else {
        !((1_u32 << (spinor_index + 1)) - 1)
    };
    Some(if (mask & greater_bits).count_ones() % 2 == 0 {
        1
    } else {
        -1
    })
}

/// Insert one spinor derivative on the right and return its ascending exterior
/// normal form. The sign counts occupied spinor indices greater than the new
/// index, matching the convention used by the exact gauge-composition streams.
pub(crate) fn right_wedge_normal_order(mask: u32, spinor_index: usize) -> Option<(u32, i128)> {
    if spinor_index >= 32 {
        return None;
    }
    right_wedge_sign(mask, spinor_index).map(|sign| (mask | (1_u32 << spinor_index), sign))
}

pub(crate) fn right_contraction_sign(mask: u32, spinor_index: usize) -> Option<i128> {
    let bit = 1_u32 << spinor_index;
    if mask & bit == 0 {
        return None;
    }
    let greater_bits = if spinor_index == 31 {
        0
    } else {
        !((1_u32 << (spinor_index + 1)) - 1)
    };
    Some(if (mask & greater_bits).count_ones() % 2 == 0 {
        1
    } else {
        -1
    })
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

fn dot_i128(left: &[i64], right: &[i64]) -> i128 {
    assert_eq!(left.len(), right.len());
    left.iter().zip(right).fold(0_i128, |sum, (a, b)| {
        sum.checked_add(i128::from(*a) * i128::from(*b))
            .expect("i128 overflow in exact Gram entry")
    })
}

fn rational_rank_i128(rows: &[Vec<i128>]) -> usize {
    let zero = Ratio::from_integer(BigInt::zero());
    let mut matrix = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| Ratio::from_integer(BigInt::from(*value)))
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

fn exact_gram(states: &[DenseState]) -> Vec<Vec<i128>> {
    states
        .iter()
        .map(|left| {
            states
                .iter()
                .map(|right| dot_i128(&left.coefficients, &right.coefficients))
                .collect()
        })
        .collect()
}

fn select_canonical_independent_states(mut candidates: Vec<DenseState>) -> Vec<DenseState> {
    candidates.sort_by(|left, right| left.pbw_word.cmp(&right.pbw_word));
    candidates.dedup_by(|left, right| left.pbw_word == right.pbw_word);
    let mut selected = Vec::new();
    let mut rank = 0;
    for candidate in candidates {
        if candidate.coefficients.iter().all(|value| *value == 0) {
            continue;
        }
        let mut trial = selected.clone();
        trial.push(candidate.clone());
        let next_rank = rational_rank_i128(&exact_gram(&trial));
        if next_rank > rank {
            selected.push(candidate);
            rank = next_rank;
        }
    }
    selected
}

const LOW_MEMORY_EXACT_PRIME: i64 = 2_147_483_647;
const LOW_MEMORY_SECOND_PRIME: i64 = 2_147_483_629;

fn modular_inverse(value: i64) -> i64 {
    modular_inverse_at(value, LOW_MEMORY_EXACT_PRIME)
}

fn modular_inverse_at(mut value: i64, prime: i64) -> i64 {
    value = value.rem_euclid(prime);
    assert_ne!(value, 0);
    let mut exponent = prime - 2;
    let mut result = 1_i64;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = i64::try_from((i128::from(result) * i128::from(value)) % i128::from(prime))
                .unwrap();
        }
        value = i64::try_from((i128::from(value) * i128::from(value)) % i128::from(prime)).unwrap();
        exponent >>= 1;
    }
    result
}

/// Select a rationally independent subset without allocating a rational Gram
/// matrix for every candidate.  Independence modulo a prime implies
/// independence over Q.  Completeness is not inferred from the modular rank:
/// the caller still requires a nonzero characteristic-zero highest-weight
/// vector and checks every ambient raising coordinate exactly.
fn select_independent_states_low_memory(mut candidates: Vec<DenseState>) -> Vec<DenseState> {
    let candidate_count = candidates.len();
    candidates.sort_by(|left, right| left.pbw_word.cmp(&right.pbw_word));
    candidates.dedup_by(|left, right| left.pbw_word == right.pbw_word);
    let mut echelon = Vec::<(usize, Vec<(usize, i64)>)>::new();
    let mut selected = Vec::new();
    let mut exact_only = false;
    for candidate in candidates {
        if exact_only {
            let mut trial = selected.clone();
            trial.push(candidate.clone());
            if rational_rank_i128(&exact_gram(&trial)) > selected.len() {
                selected.push(candidate);
            }
            continue;
        }
        let mut row = candidate
            .coefficients
            .iter()
            .map(|value| value.rem_euclid(LOW_MEMORY_EXACT_PRIME))
            .collect::<Vec<_>>();
        for (pivot, basis) in &echelon {
            let factor = row[*pivot];
            if factor == 0 {
                continue;
            }
            for (index, basis_value) in basis {
                row[*index] = i64::try_from(
                    (i128::from(row[*index]) - i128::from(factor) * i128::from(*basis_value))
                        .rem_euclid(i128::from(LOW_MEMORY_EXACT_PRIME)),
                )
                .unwrap();
            }
        }
        let Some(pivot) = row.iter().position(|value| *value != 0) else {
            // Modular dependence does not prove rational dependence.  Check
            // the canonical trial exactly.  If this prime was unlucky, retain
            // the state and use the exact selector for the rest of the weight
            // space rather than making any further modular non-inferences.
            let mut trial = selected.clone();
            trial.push(candidate.clone());
            if rational_rank_i128(&exact_gram(&trial)) > selected.len() {
                selected.push(candidate);
                exact_only = true;
            }
            continue;
        };
        let inverse = modular_inverse(row[pivot]);
        let sparse = row
            .into_iter()
            .enumerate()
            .skip(pivot)
            .filter_map(|(index, value)| {
                (value != 0).then(|| {
                    (
                        index,
                        i64::try_from(
                            (i128::from(value) * i128::from(inverse))
                                % i128::from(LOW_MEMORY_EXACT_PRIME),
                        )
                        .unwrap(),
                    )
                })
            })
            .collect();
        echelon.push((pivot, sparse));
        selected.push(candidate);
    }
    eprintln!(
        "proof-safe modular basis selected {}/{} PBW candidates{}",
        selected.len(),
        candidate_count,
        if exact_only {
            " (completed with exact fallback)"
        } else {
            ""
        }
    );
    selected
}

fn relevant_source_bases(
    model: &mut ExteriorModel,
    highest: DenseState,
    target_weight: Weight,
) -> BTreeMap<Weight, Vec<DenseState>> {
    let source_highest_weight = highest.weight;
    let needed_weights = model
        .spinors
        .iter()
        .map(|spinor| subtract(target_weight, *spinor))
        .filter_map(|weight| {
            lowering_coordinates(source_highest_weight, weight).map(|coordinates| {
                let depth = coordinates.iter().map(|value| *value as usize).sum();
                (weight, depth)
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_depth = needed_weights.values().copied().max().unwrap_or(0);
    let mut current = BTreeMap::from([(source_highest_weight, vec![highest])]);
    let mut required = BTreeMap::new();
    for depth in 0..=maximum_depth {
        let mut next_candidates = BTreeMap::<Weight, Vec<DenseState>>::new();
        for (weight, basis) in current {
            if needed_weights.get(&weight) == Some(&depth) {
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
                                    .map(|value| *value as usize)
                                    .sum::<usize>()
                                    == depth + 1
                            },
                        )
                });
                if !remains_relevant {
                    continue;
                }
                for vector in &basis {
                    let lowered = model.lower(vector, root);
                    if lowered.coefficients.iter().any(|value| *value != 0) {
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
            .map(|(weight, candidates)| (weight, select_canonical_independent_states(candidates)))
            .filter(|(_, basis)| !basis.is_empty())
            .collect();
    }
    required
}

fn relevant_source_bases_low_memory(
    model: &mut ExteriorModel,
    highest: DenseState,
    target_weight: Weight,
) -> BTreeMap<Weight, Vec<DenseState>> {
    let source_highest_weight = highest.weight;
    let needed_weights = model
        .spinors
        .iter()
        .map(|spinor| subtract(target_weight, *spinor))
        .filter_map(|weight| {
            lowering_coordinates(source_highest_weight, weight).map(|coordinates| {
                let depth = coordinates.iter().map(|value| *value as usize).sum();
                (weight, depth)
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_depth = needed_weights.values().copied().max().unwrap_or(0);
    let mut current = BTreeMap::from([(source_highest_weight, vec![highest])]);
    let mut required = BTreeMap::new();
    for depth in 0..=maximum_depth {
        let mut next_candidates = BTreeMap::<Weight, Vec<DenseState>>::new();
        for (weight, basis) in current {
            if needed_weights.get(&weight) == Some(&depth) {
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
                                    .map(|value| *value as usize)
                                    .sum::<usize>()
                                    == depth + 1
                            },
                        )
                });
                if !remains_relevant {
                    continue;
                }
                for vector in &basis {
                    let lowered = model.lower(vector, root);
                    if lowered.coefficients.iter().any(|value| *value != 0) {
                        next_candidates
                            .entry(next_weight)
                            .or_default()
                            .push(lowered);
                    }
                }
            }
        }
        eprintln!(
            "low-memory level-18 basis depth {depth}/{maximum_depth}: {} active weights",
            next_candidates.len()
        );
        current = next_candidates
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(weight, candidates)| (weight, select_independent_states_low_memory(candidates)))
            .filter(|(_, basis)| !basis.is_empty())
            .collect();
    }
    required
}

#[derive(Debug, Clone)]
struct TensorOutput {
    components: BTreeMap<usize, Vec<i64>>,
}

fn add_component(target: &mut Vec<i64>, source: &[i64]) {
    assert_eq!(target.len(), source.len());
    for (target, source) in target.iter_mut().zip(source) {
        *target = i64::try_from(
            i128::from(*target)
                .checked_add(i128::from(*source))
                .expect("i128 overflow while combining tensor output"),
        )
        .expect("tensor output coefficient exceeds i64 storage");
    }
}

fn tensor_output(
    model: &mut ExteriorModel,
    source: &DenseState,
    spinor_index: usize,
    root: usize,
) -> TensorOutput {
    let (_, raised_source) = model.raise_coefficients(source, root);
    let mut components = BTreeMap::new();
    components.insert(spinor_index, raised_source);
    if let Some(next_spinor) = raised_spinor_index(spinor_index, root, &model.spinors) {
        let component = components
            .entry(next_spinor)
            .or_insert_with(|| vec![0; source.coefficients.len()]);
        add_component(component, &source.coefficients);
    }
    TensorOutput { components }
}

fn tensor_output_dot(left: &TensorOutput, right: &TensorOutput) -> i128 {
    left.components
        .iter()
        .filter_map(|(spinor, coefficients)| {
            right
                .components
                .get(spinor)
                .map(|other| dot_i128(coefficients, other))
        })
        .try_fold(0_i128, |sum, value| sum.checked_add(value))
        .expect("i128 overflow in tensor-output Gram entry")
}

fn bigint_nullspace(rows: &[Vec<BigInt>], columns: usize) -> Vec<Vec<Ratio<BigInt>>> {
    let zero = Ratio::from_integer(BigInt::zero());
    let mut reduced = rows
        .iter()
        .filter(|row| row.iter().any(|value| !value.is_zero()))
        .map(|row| {
            row.iter()
                .cloned()
                .map(Ratio::from_integer)
                .collect::<Vec<_>>()
        })
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
    (0..columns)
        .filter(|column| !pivot_columns.contains(column))
        .map(|free| {
            let mut vector = vec![zero.clone(); columns];
            vector[free] = Ratio::from_integer(BigInt::one());
            for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free].clone();
            }
            vector
        })
        .collect()
}

fn bigint_gcd(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = left.abs();
    right = right.abs();
    while !right.is_zero() {
        let remainder = left % &right;
        left = right;
        right = remainder;
    }
    left
}

fn bigint_lcm(left: BigInt, right: BigInt) -> BigInt {
    if left.is_zero() || right.is_zero() {
        BigInt::zero()
    } else {
        (&left / bigint_gcd(left.clone(), right.clone())) * right
    }
}

fn primitive_i64(vector: &[Ratio<BigInt>]) -> Vec<i64> {
    let denominator = vector.iter().fold(BigInt::one(), |common, coefficient| {
        bigint_lcm(common, coefficient.denom().clone())
    });
    let mut integers = vector
        .iter()
        .map(|coefficient| coefficient.numer() * (&denominator / coefficient.denom()))
        .collect::<Vec<_>>();
    let gcd = integers
        .iter()
        .fold(BigInt::zero(), |gcd, value| bigint_gcd(gcd, value.clone()));
    assert!(!gcd.is_zero());
    for value in &mut integers {
        *value /= &gcd;
    }
    if integers
        .iter()
        .find(|value| !value.is_zero())
        .unwrap()
        .is_negative()
    {
        for value in &mut integers {
            *value = -value.clone();
        }
    }
    integers
        .into_iter()
        .map(|value| {
            value
                .to_i64()
                .expect("primitive coupling coefficient exceeds i64")
        })
        .collect()
}

fn tensor_gram(outputs_by_root: &[Vec<TensorOutput>]) -> Vec<Vec<BigInt>> {
    let columns = outputs_by_root[0].len();
    (0..columns)
        .map(|left| {
            (0..columns)
                .map(|right| {
                    let value = outputs_by_root.iter().fold(0_i128, |sum, outputs| {
                        sum.checked_add(tensor_output_dot(&outputs[left], &outputs[right]))
                            .expect("i128 overflow in total tensor Gram")
                    });
                    BigInt::from(value)
                })
                .collect()
        })
        .collect()
}

fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn rational_reconstruct(residue: i128, modulus: i128) -> Option<(i128, i128)> {
    let residue = residue.rem_euclid(modulus);
    let bound = ((modulus / 2) as f64).sqrt() as i128;
    let (mut old_remainder, mut remainder) = (modulus, residue);
    let (mut old_denominator, mut denominator) = (0_i128, 1_i128);
    while remainder.abs() > bound {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
        (old_denominator, denominator) = (denominator, old_denominator - quotient * denominator);
    }
    if denominator == 0 {
        return None;
    }
    if denominator < 0 {
        remainder = -remainder;
        denominator = -denominator;
    }
    let divisor = gcd_i128(remainder, denominator);
    let numerator = remainder / divisor;
    denominator /= divisor;
    (numerator.abs() <= bound
        && denominator <= bound
        && (residue * denominator - numerator).rem_euclid(modulus) == 0)
        .then_some((numerator, denominator))
}

fn checked_lcm_i128(left: i128, right: i128) -> Option<i128> {
    if left == 0 || right == 0 {
        Some(0)
    } else {
        (left / gcd_i128(left, right))
            .checked_mul(right)
            .map(i128::abs)
    }
}

fn modular_null_vector(outputs_by_root: &[Vec<TensorOutput>], prime: i64) -> (Vec<i64>, usize) {
    let columns = outputs_by_root[0].len();
    eprintln!("low-memory modular Gram: {columns} columns modulo {prime}");
    let mut matrix = vec![vec![0_i64; columns]; columns];
    for left in 0..columns {
        for right in left..columns {
            let value = outputs_by_root.iter().fold(0_i128, |sum, outputs| {
                sum.checked_add(tensor_output_dot(&outputs[left], &outputs[right]))
                    .expect("i128 overflow in modular tensor Gram")
            });
            let value = i64::try_from(value.rem_euclid(i128::from(prime))).unwrap();
            matrix[left][right] = value;
            matrix[right][left] = value;
        }
    }
    eprintln!("low-memory modular Gram assembled");
    let mut pivots = Vec::new();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..columns).find(|row| matrix[*row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = modular_inverse_at(matrix[rank][column], prime);
        for value in &mut matrix[rank][column..] {
            *value = i64::try_from((i128::from(*value) * i128::from(inverse)) % i128::from(prime))
                .unwrap();
        }
        let pivot_row = matrix[rank].clone();
        matrix[(rank + 1)..].par_iter_mut().for_each(|row| {
            let factor = row[column];
            if factor == 0 {
                return;
            }
            for index in column..columns {
                row[index] = i64::try_from(
                    (i128::from(row[index]) - i128::from(factor) * i128::from(pivot_row[index]))
                        .rem_euclid(i128::from(prime)),
                )
                .unwrap();
            }
        });
        pivots.push(column);
        rank += 1;
        if rank % 100 == 0 || rank + 1 >= columns {
            eprintln!("low-memory modular Gram rank progress: {rank}/{columns}");
        }
    }
    let pivot_set = pivots
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let free = (0..columns)
        .filter(|column| !pivot_set.contains(column))
        .collect::<Vec<_>>();
    if free.len() != 1 {
        return (Vec::new(), rank);
    }
    let mut modular = vec![0_i64; columns];
    modular[free[0]] = 1;
    for (row, pivot) in pivots.iter().copied().enumerate().rev() {
        let sum = ((pivot + 1)..columns).fold(0_i128, |sum, column| {
            (sum + i128::from(matrix[row][column]) * i128::from(modular[column]))
                % i128::from(prime)
        });
        modular[pivot] = i64::try_from((-sum).rem_euclid(i128::from(prime))).unwrap();
    }
    (modular, rank)
}

/// Find the one-dimensional kernel over two large primes, combine the
/// projective coordinates by CRT, rationally reconstruct them, and return a
/// primitive integer vector.  Acceptance by the caller still uses the
/// characteristic-zero ambient raising residual.
fn modular_primitive_null_vector(outputs_by_root: &[Vec<TensorOutput>]) -> (Vec<i64>, usize) {
    let (first, first_rank) = modular_null_vector(outputs_by_root, LOW_MEMORY_EXACT_PRIME);
    let (second, second_rank) = modular_null_vector(outputs_by_root, LOW_MEMORY_SECOND_PRIME);
    if first_rank != second_rank || first.len() != second.len() || first.is_empty() {
        return (Vec::new(), first_rank.min(second_rank));
    }
    let modulus = i128::from(LOW_MEMORY_EXACT_PRIME) * i128::from(LOW_MEMORY_SECOND_PRIME);
    let first_inverse_mod_second =
        modular_inverse_at(LOW_MEMORY_EXACT_PRIME, LOW_MEMORY_SECOND_PRIME);
    let combined = first
        .into_iter()
        .zip(second)
        .map(|(first, second)| {
            let correction = i128::from(
                i64::try_from(
                    (i128::from(second - first) * i128::from(first_inverse_mod_second))
                        .rem_euclid(i128::from(LOW_MEMORY_SECOND_PRIME)),
                )
                .unwrap(),
            );
            (i128::from(first) + i128::from(LOW_MEMORY_EXACT_PRIME) * correction)
                .rem_euclid(modulus)
        })
        .collect::<Vec<_>>();
    let rationals = combined
        .into_iter()
        .map(|residue| rational_reconstruct(residue, modulus))
        .collect::<Option<Vec<_>>>();
    let Some(rationals) = rationals else {
        return (Vec::new(), first_rank);
    };
    let Some(denominator) = rationals
        .iter()
        .try_fold(1_i128, |common, (_, denominator)| {
            checked_lcm_i128(common, *denominator)
        })
    else {
        return (Vec::new(), first_rank);
    };
    let primitive = rationals
        .into_iter()
        .map(|(numerator, divisor)| {
            numerator
                .checked_mul(denominator / divisor)
                .and_then(|value| i64::try_from(value).ok())
        })
        .collect::<Option<Vec<_>>>();
    let Some(mut primitive) = primitive else {
        return (Vec::new(), first_rank);
    };
    let divisor = primitive
        .iter()
        .fold(0_i64, |common, value| gcd_i64(common, *value));
    if divisor == 0 {
        return (Vec::new(), first_rank);
    }
    for value in &mut primitive {
        *value /= divisor;
    }
    if primitive.iter().find(|value| **value != 0).unwrap() < &0 {
        for value in &mut primitive {
            *value = -*value;
        }
    }
    (primitive, first_rank)
}

fn exact_residual_counts(
    outputs_by_root: &[Vec<TensorOutput>],
    primitive: &[i64],
) -> ([usize; 5], i128) {
    let mut maximum = 0_i128;
    let counts = std::array::from_fn(|root| {
        let mut combined = BTreeMap::<usize, Vec<i128>>::new();
        for (output, coefficient) in outputs_by_root[root].iter().zip(primitive) {
            for (spinor, values) in &output.components {
                let destination = combined
                    .entry(*spinor)
                    .or_insert_with(|| vec![0; values.len()]);
                for (slot, value) in destination.iter_mut().zip(values) {
                    *slot = slot
                        .checked_add(i128::from(*coefficient) * i128::from(*value))
                        .expect("i128 overflow in exact raising residual");
                    maximum = maximum.max(slot.abs());
                }
            }
        }
        combined
            .values()
            .map(|values| values.iter().filter(|value| **value != 0).count())
            .sum()
    });
    (counts, maximum)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasisSelection {
    ExactCanonical,
    ModularLowMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NullspaceStrategy {
    ExactBigInt,
    ModularFirst,
}

fn exact_primitive_null_vector(
    outputs_by_root: &[Vec<TensorOutput>],
    domain_dimension: usize,
) -> (Vec<i64>, usize, usize) {
    let gram = tensor_gram(outputs_by_root);
    let nullspace = bigint_nullspace(&gram, domain_dimension);
    let primitive = if nullspace.len() == 1 {
        primitive_i64(&nullspace[0])
    } else {
        Vec::new()
    };
    (
        primitive,
        domain_dimension.saturating_sub(nullspace.len()),
        nullspace.len(),
    )
}

fn build_abstract_from_fixture_with_strategies(
    problem: CouplingProblem,
    dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    basis_selection: BasisSelection,
    nullspace_strategy: NullspaceStrategy,
) -> (AbstractCouplingCertificate, Vec<DenseState>) {
    let mut model = ExteriorModel::new(problem.exterior_degree);
    let highest = model.fixture_state(dynkin_label, fixture_bytes, coefficient_width_bytes);
    let bases = match basis_selection {
        BasisSelection::ExactCanonical => {
            relevant_source_bases(&mut model, highest, problem.target_weight)
        }
        BasisSelection::ModularLowMemory => {
            relevant_source_bases_low_memory(&mut model, highest, problem.target_weight)
        }
    };
    let mut domain = Vec::<(usize, DenseState)>::new();
    for (spinor_index, spinor_weight) in model.spinors.iter().copied().enumerate() {
        let source_weight = subtract(problem.target_weight, spinor_weight);
        if let Some(states) = bases.get(&source_weight) {
            domain.extend(states.iter().cloned().map(|state| (spinor_index, state)));
        }
    }
    let outputs_by_root = (0..5)
        .map(|root| {
            domain
                .iter()
                .map(|(spinor_index, state)| tensor_output(&mut model, state, *spinor_index, root))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (primitive, gram_matrix_rank, kernel_dimension, dependency_test) = match nullspace_strategy
    {
        NullspaceStrategy::ExactBigInt => {
            let (primitive, rank, kernel_dimension) =
                exact_primitive_null_vector(&outputs_by_root, domain.len());
            (
                primitive,
                rank,
                kernel_dimension,
                "exact characteristic-zero BigInt Gram nullspace; no modular acceptance"
                    .to_string(),
            )
        }
        NullspaceStrategy::ModularFirst => {
            let (candidate, modular_rank) = modular_primitive_null_vector(&outputs_by_root);
            let candidate_is_exact = modular_rank + 1 == domain.len()
                && candidate.len() == domain.len()
                && exact_residual_counts(&outputs_by_root, &candidate).0 == [0; 5];
            if candidate_is_exact {
                (
                        candidate,
                        modular_rank,
                        1,
                        "two-prime modular Gram nullspace with CRT and rational reconstruction; accepted only after the unchanged canonical domain has rank n-1 over both primes and the reconstructed primitive has zero characteristic-zero raising residual in every ambient coordinate"
                            .to_string(),
                    )
            } else {
                let (primitive, rank, kernel_dimension) =
                    exact_primitive_null_vector(&outputs_by_root, domain.len());
                (
                        primitive,
                        rank,
                        kernel_dimension,
                        "two-prime modular candidate was not certifiable; fell back to the exact characteristic-zero BigInt Gram nullspace on the unchanged canonical domain"
                            .to_string(),
                    )
            }
        }
    };
    let (residuals, residual_maximum) = if primitive.len() == domain.len() {
        exact_residual_counts(&outputs_by_root, &primitive)
    } else {
        ([usize::MAX; 5], 0)
    };
    let domain_basis = domain
        .iter()
        .map(|(spinor_index, state)| CanonicalDomainBasisEntry {
            free_spinor_index: *spinor_index,
            free_spinor_weight: model.spinors[*spinor_index],
            source_weight: state.weight,
            pbw_word_simple_roots: state.pbw_word.clone(),
        })
        .collect::<Vec<_>>();
    let source_weight_multiplicities = bases
        .iter()
        .map(|(weight, states)| (*weight, states.len()))
        .collect::<Vec<_>>();
    let primitive_coefficient_gcd = primitive
        .iter()
        .fold(0_i64, |gcd, value| gcd_i64(gcd, *value));
    let maximum_absolute_primitive_coefficient =
        primitive.iter().map(|value| value.abs()).max().unwrap_or(0);
    let multiplicity_one = kernel_dimension == 1;
    let passed = multiplicity_one
        && primitive.len() == domain.len()
        && residuals == [0; 5]
        && primitive_coefficient_gcd == 1;
    let csr_actions_built = model.actions.len();
    let csr_nonzero_entries = model
        .actions
        .values()
        .map(|action| action.destination_indices.len())
        .sum();
    let states = domain.into_iter().map(|(_, state)| state).collect();
    (
        AbstractCouplingCertificate {
            schema_version: format!("{}-abstract-coupling-v1", problem.schema_prefix),
            role: format!(
                "exact canonical abstract coupling into ({})",
                problem.target_dynkin_label
            ),
            source_dynkin_label: dynkin_label.to_string(),
            source_fixture_copy: fixture_copy,
            target_dynkin_label: problem.target_dynkin_label.to_string(),
            basis_method: match basis_selection {
                BasisSelection::ExactCanonical =>
                    "lexicographic PBW lowering words with exact exterior-realization Gram rank"
                        .to_string(),
                BasisSelection::ModularLowMemory =>
                    "lexicographic PBW lowering words with modular independence acceptance and exact fallback reserved for modular non-increases"
                        .to_string(),
            },
            dependency_test,
            product_weight_domain_dimension: domain_basis.len(),
            source_weight_spaces_used: bases.len(),
            source_weight_multiplicities,
            domain_basis,
            gram_matrix_rank,
            kernel_dimension,
            primitive_domain_coefficients: primitive,
            primitive_coefficient_gcd,
            maximum_absolute_primitive_coefficient,
            exact_raising_residual_terms_by_simple_root: residuals,
            maximum_absolute_checked_accumulator: model
                .maximum_absolute_accumulator
                .max(residual_maximum),
            storage_type: "i64 coefficients with checked i128 accumulation".to_string(),
            exterior_action_storage: "precomputed CSR over sorted exterior-mask weight spaces"
                .to_string(),
            csr_actions_built,
            csr_nonzero_entries,
            multiplicity_one,
            passed,
        },
        states,
    )
}

fn build_abstract_from_fixture(
    problem: CouplingProblem,
    dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> (AbstractCouplingCertificate, Vec<DenseState>) {
    build_abstract_from_fixture_with_strategies(
        problem,
        dynkin_label,
        fixture_copy,
        coefficient_width_bytes,
        fixture_bytes,
        BasisSelection::ExactCanonical,
        NullspaceStrategy::ModularFirst,
    )
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

fn state_for_word(
    model: &mut ExteriorModel,
    highest: &DenseState,
    word: &[u8],
    cache: &mut BTreeMap<Vec<u8>, DenseState>,
) -> DenseState {
    if let Some(state) = cache.get(word) {
        return state.clone();
    }
    let prefix = &word[..word.len() - 1];
    let parent = state_for_word(model, highest, prefix, cache);
    let root = usize::from(word[word.len() - 1] - 1);
    let state = model.lower(&parent, root);
    cache.insert(word.to_vec(), state.clone());
    state
}

fn add_scaled_dense_component(
    components: &mut BTreeMap<usize, DenseState>,
    spinor_index: usize,
    source: &DenseState,
    scale: i64,
    maximum: &mut i128,
) {
    if scale == 0 {
        return;
    }
    let destination = components
        .entry(spinor_index)
        .or_insert_with(|| DenseState {
            weight: source.weight,
            pbw_word: Vec::new(),
            coefficients: vec![0; source.coefficients.len()],
        });
    assert_eq!(destination.weight, source.weight);
    assert_eq!(destination.coefficients.len(), source.coefficients.len());
    for (output, input) in destination
        .coefficients
        .iter_mut()
        .zip(&source.coefficients)
    {
        let value = i128::from(*output)
            .checked_add(i128::from(scale) * i128::from(*input))
            .expect("i128 overflow while assembling a coupled state");
        *maximum = (*maximum).max(value.abs());
        *output = i64::try_from(value).expect("coupled-state coefficient exceeds i64 storage");
    }
}

fn materialize_coupled_highest(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
    coefficient_width_bytes: usize,
) -> (ExteriorModel, CoupledDenseState, i128) {
    assert_eq!(
        abstract_certificate.target_dynkin_label,
        problem.target_dynkin_label
    );
    assert!(abstract_certificate.passed);
    let mut model = ExteriorModel::new(problem.exterior_degree);
    let highest = model.fixture_state(
        &abstract_certificate.source_dynkin_label,
        fixture_bytes,
        coefficient_width_bytes,
    );
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut components = BTreeMap::new();
    let mut maximum = 0_i128;
    for (entry, coefficient) in abstract_certificate
        .domain_basis
        .iter()
        .zip(&abstract_certificate.primitive_domain_coefficients)
    {
        let state = state_for_word(
            &mut model,
            &highest,
            &entry.pbw_word_simple_roots,
            &mut cache,
        );
        assert_eq!(state.weight, entry.source_weight);
        add_scaled_dense_component(
            &mut components,
            entry.free_spinor_index,
            &state,
            *coefficient,
            &mut maximum,
        );
    }
    (
        model,
        CoupledDenseState {
            total_weight: problem.target_weight,
            components,
        },
        maximum,
    )
}

fn dense_state_payload_bytes(state: &DenseState) -> u64 {
    u64::try_from(
        state.pbw_word.capacity() * std::mem::size_of::<u8>()
            + state.coefficients.capacity() * std::mem::size_of::<i64>(),
    )
    .unwrap()
}

fn exterior_model_payload_bytes(model: &ExteriorModel) -> u64 {
    let half_groups = model
        .left
        .values()
        .chain(model.right.values())
        .map(|masks| masks.capacity() * std::mem::size_of::<u16>())
        .sum::<usize>();
    let spaces = model
        .spaces
        .values()
        .map(|space| {
            space.masks.capacity() * std::mem::size_of::<u32>()
                + space.index.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<usize>())
        })
        .sum::<usize>();
    let actions = model
        .actions
        .values()
        .map(|action| {
            action.source_offsets.capacity() * std::mem::size_of::<usize>()
                + action.destination_indices.capacity() * std::mem::size_of::<usize>()
                + action.signs.capacity() * std::mem::size_of::<i8>()
        })
        .sum::<usize>();
    u64::try_from(half_groups + spaces + actions).unwrap()
}

fn coupled_state_payload_bytes(state: &CoupledDenseState) -> u64 {
    state
        .components
        .values()
        .map(dense_state_payload_bytes)
        .sum()
}

fn momentum_coupled_state_payload_bytes(state: &MomentumCoupledDenseState) -> u64 {
    state
        .components
        .values()
        .map(dense_state_payload_bytes)
        .sum()
}

fn shared_state_payload_limit_bytes() -> u64 {
    std::env::var("ADINKRA_FX_SHARED_STATE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(24 * 1024 * 1024 * 1024)
}

fn enforce_shared_state_payload_limit(estimated_payload_bytes: u64) -> io::Result<u64> {
    let limit = shared_state_payload_limit_bytes();
    if estimated_payload_bytes > limit {
        return Err(io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!(
                "shared first-momentum operator state payload estimate {estimated_payload_bytes} exceeds configured limit {limit}"
            ),
        ));
    }
    Ok(limit)
}

fn lower_coupled_state(
    model: &mut ExteriorModel,
    source: &CoupledDenseState,
    root: usize,
    maximum: &mut i128,
) -> CoupledDenseState {
    let spinors = model.spinors;
    let mut components = BTreeMap::new();
    for (&free_spinor, exterior) in &source.components {
        if let Some(lowered_free_spinor) = lowered_spinor_index(free_spinor, root, &spinors) {
            add_scaled_dense_component(&mut components, lowered_free_spinor, exterior, 1, maximum);
        }
        let lowered_exterior = model.lower(exterior, root);
        if lowered_exterior
            .coefficients
            .iter()
            .any(|coefficient| *coefficient != 0)
        {
            add_scaled_dense_component(&mut components, free_spinor, &lowered_exterior, 1, maximum);
        }
    }
    CoupledDenseState {
        total_weight: subtract(source.total_weight, SIMPLE_ROOTS[root]),
        components,
    }
}

fn coupled_state_for_word(
    model: &mut ExteriorModel,
    highest: &CoupledDenseState,
    word: &[u8],
    cache: &mut BTreeMap<Vec<u8>, CoupledDenseState>,
    maximum: &mut i128,
) -> CoupledDenseState {
    if let Some(state) = cache.get(word) {
        return state.clone();
    }
    let prefix = &word[..word.len() - 1];
    let parent = coupled_state_for_word(model, highest, prefix, cache, maximum);
    let root = usize::from(word[word.len() - 1] - 1);
    let state = lower_coupled_state(model, &parent, root, maximum);
    cache.insert(word.to_vec(), state.clone());
    state
}

fn momentum_vector_weights() -> [Weight; 11] {
    let mut weights = [[0_i8; 5]; 11];
    for axis in 0..5 {
        weights[2 * axis][axis] = 2;
        weights[2 * axis + 1][axis] = -2;
    }
    weights
}

fn lowered_momentum_vector_index(vector_index: usize, root: usize) -> Option<(usize, i64)> {
    let weights = momentum_vector_weights();
    let weight = weights[vector_index];
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
        Some((10, 1))
    } else if weight == [0; 5] {
        target[4] = -2;
        Some((weights.iter().position(|item| *item == target).unwrap(), 2))
    } else {
        None
    }
}

fn add_scaled_momentum_dense_component(
    components: &mut BTreeMap<(usize, usize), DenseState>,
    momentum_vector_index: usize,
    free_spinor_index: usize,
    source: &DenseState,
    scale: i64,
    maximum: &mut i128,
) {
    if scale == 0 {
        return;
    }
    let destination = components
        .entry((momentum_vector_index, free_spinor_index))
        .or_insert_with(|| DenseState {
            weight: source.weight,
            pbw_word: Vec::new(),
            coefficients: vec![0; source.coefficients.len()],
        });
    assert_eq!(destination.weight, source.weight);
    assert_eq!(destination.coefficients.len(), source.coefficients.len());
    for (output, input) in destination
        .coefficients
        .iter_mut()
        .zip(&source.coefficients)
    {
        let value = i128::from(*output)
            .checked_add(i128::from(scale) * i128::from(*input))
            .expect("i128 overflow while assembling a momentum-coupled state");
        *maximum = (*maximum).max(value.abs());
        *output = i64::try_from(value).expect("momentum-coupled coefficient exceeds i64 storage");
    }
}

fn lower_momentum_coupled_state(
    model: &mut ExteriorModel,
    source: &MomentumCoupledDenseState,
    root: usize,
    maximum: &mut i128,
) -> MomentumCoupledDenseState {
    let spinors = model.spinors;
    let mut components = BTreeMap::new();
    for (&(momentum_vector, free_spinor), exterior) in &source.components {
        if let Some((lowered_vector, factor)) = lowered_momentum_vector_index(momentum_vector, root)
        {
            add_scaled_momentum_dense_component(
                &mut components,
                lowered_vector,
                free_spinor,
                exterior,
                factor,
                maximum,
            );
        }
        if let Some(lowered_free_spinor) = lowered_spinor_index(free_spinor, root, &spinors) {
            add_scaled_momentum_dense_component(
                &mut components,
                momentum_vector,
                lowered_free_spinor,
                exterior,
                1,
                maximum,
            );
        }
        let lowered_exterior = model.lower(exterior, root);
        if lowered_exterior
            .coefficients
            .iter()
            .any(|coefficient| *coefficient != 0)
        {
            add_scaled_momentum_dense_component(
                &mut components,
                momentum_vector,
                free_spinor,
                &lowered_exterior,
                1,
                maximum,
            );
        }
    }
    MomentumCoupledDenseState {
        total_weight: subtract(source.total_weight, SIMPLE_ROOTS[root]),
        components,
    }
}

fn momentum_coupled_state_for_word(
    model: &mut ExteriorModel,
    highest: &MomentumCoupledDenseState,
    word: &[u8],
    cache: &mut BTreeMap<Vec<u8>, MomentumCoupledDenseState>,
    maximum: &mut i128,
) -> MomentumCoupledDenseState {
    if let Some(state) = cache.get(word) {
        return state.clone();
    }
    let prefix = &word[..word.len() - 1];
    let parent = momentum_coupled_state_for_word(model, highest, prefix, cache, maximum);
    let root = usize::from(word[word.len() - 1] - 1);
    let state = lower_momentum_coupled_state(model, &parent, root, maximum);
    cache.insert(word.to_vec(), state.clone());
    state
}

fn dense_coupled_to_sparse(
    model: &mut ExteriorModel,
    source: &CoupledDenseState,
) -> CoupledSparseState {
    let components = source
        .components
        .iter()
        .map(|(spinor, state)| {
            let masks = model.space(state.weight).masks.clone();
            let values = masks
                .into_iter()
                .zip(&state.coefficients)
                .filter_map(|(mask, coefficient)| {
                    (*coefficient != 0).then_some((mask, i128::from(*coefficient)))
                })
                .collect();
            (*spinor, values)
        })
        .collect();
    CoupledSparseState { components }
}

fn dense_coupled_to_sparse64(
    model: &mut ExteriorModel,
    source: &CoupledDenseState,
) -> CoupledSparseState64 {
    let components = source
        .components
        .iter()
        .map(|(spinor, state)| {
            let masks = &model.space(state.weight).masks;
            let values = masks
                .iter()
                .copied()
                .zip(&state.coefficients)
                .filter_map(|(mask, coefficient)| {
                    (*coefficient != 0).then_some((mask, *coefficient))
                })
                .collect();
            (*spinor, values)
        })
        .collect();
    CoupledSparseState64 { components }
}

fn sparse64_payload_bytes(state: &CoupledSparseState64) -> u64 {
    state
        .components
        .values()
        .map(|values| values.capacity() * std::mem::size_of::<(u32, i64)>())
        .sum::<usize>() as u64
}

#[cfg(any(feature = "cuda", test))]
const CUDA_SPARSE_LOWERING_BYTES_PER_INPUT: u64 = 240;
#[cfg(any(feature = "cuda", test))]
const DEFAULT_CUDA_SPARSE_LOWERING_HOST_CAP_BYTES: u64 = 256 * 1024 * 1024;

#[cfg(any(feature = "cuda", test))]
fn cuda_sparse_lowering_estimated_host_bytes(input_terms: usize) -> Option<u64> {
    u64::try_from(input_terms)
        .ok()?
        .checked_mul(CUDA_SPARSE_LOWERING_BYTES_PER_INPUT)
}

#[cfg(feature = "cuda")]
fn cuda_sparse_lowering_fits_host_cap(input_terms: usize) -> io::Result<bool> {
    let cap = match std::env::var("ADYNKRA_GPU_FX_HOST_CAP_BYTES") {
        Ok(value) => value.parse::<u64>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid ADYNKRA_GPU_FX_HOST_CAP_BYTES={value}: {error}"),
            )
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_CUDA_SPARSE_LOWERING_HOST_CAP_BYTES,
        Err(error) => return Err(io::Error::other(error)),
    };
    let estimated = cuda_sparse_lowering_estimated_host_bytes(input_terms);
    Ok(estimated.is_some_and(|bytes| bytes <= cap))
}

enum SparseLoweringStream<'a> {
    Identity {
        values: &'a [(u32, i64)],
        index: usize,
    },
    ExteriorReplacement {
        values: &'a [(u32, i64)],
        index: usize,
        upper: usize,
        lower: usize,
    },
}

impl SparseLoweringStream<'_> {
    fn next(&mut self) -> io::Result<Option<(u32, i64)>> {
        match self {
            Self::Identity { values, index } => {
                let value = values.get(*index).copied();
                *index += usize::from(value.is_some());
                Ok(value)
            }
            Self::ExteriorReplacement {
                values,
                index,
                upper,
                lower,
            } => {
                let upper_bit = 1_u32 << *upper;
                let lower_bit = 1_u32 << *lower;
                while let Some(&(mask, coefficient)) = values.get(*index) {
                    *index += 1;
                    if mask & upper_bit == 0 || mask & lower_bit != 0 {
                        continue;
                    }
                    let output_mask = mask ^ upper_bit ^ lower_bit;
                    let sign = exterior_replacement_sign(mask, *upper, *lower);
                    let output_coefficient = coefficient.checked_mul(sign).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "sparse lowering coefficient overflow",
                        )
                    })?;
                    return Ok(Some((output_mask, output_coefficient)));
                }
                Ok(None)
            }
        }
    }
}

/// Exact CPU lowering with bounded transient storage. Each fixed exterior
/// replacement is an order-preserving stream over canonical masks, so a
/// small k-way merge directly constructs the canonical output. This avoids
/// the former materialized contribution vector, whose worst case was 13N.
fn lower_sparse_coupled_state64_bounded(
    source: &CoupledSparseState64,
    root: usize,
    maximum: &mut i128,
) -> io::Result<CoupledSparseState64> {
    if root >= SIMPLE_ROOTS.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sparse lowering simple root is out of range",
        ));
    }
    let spinors = spinor_weights();
    debug_assert!(source.components.iter().all(|(&spinor, values)| {
        spinor < spinors.len()
            && values.iter().all(|(_, coefficient)| *coefficient != 0)
            && values.windows(2).all(|pair| pair[0].0 < pair[1].0)
    }));
    let transitions = (0..spinors.len())
        .filter_map(|upper| lowered_spinor_index(upper, root, &spinors).map(|lower| (upper, lower)))
        .collect::<Vec<_>>();
    let outputs = (0..spinors.len())
        .into_par_iter()
        .map(|output_free_spinor| {
            lower_sparse_output_component(source, root, &spinors, &transitions, output_free_spinor)
        })
        .collect::<io::Result<Vec<_>>>()?;
    let mut components = BTreeMap::<usize, Vec<(u32, i64)>>::new();
    for (output_free_spinor, output, local_maximum) in outputs {
        *maximum = (*maximum).max(local_maximum);
        if !output.is_empty() {
            components.insert(output_free_spinor, output);
        }
    }
    Ok(CoupledSparseState64 { components })
}

fn lower_sparse_output_component(
    source: &CoupledSparseState64,
    root: usize,
    spinors: &[Weight; 32],
    transitions: &[(usize, usize)],
    output_free_spinor: usize,
) -> io::Result<SparseLoweredComponent> {
    let mut streams = Vec::<SparseLoweringStream<'_>>::new();
    if let Some(values) = source.components.get(&output_free_spinor) {
        streams.extend(transitions.iter().map(|&(upper, lower)| {
            SparseLoweringStream::ExteriorReplacement {
                values,
                index: 0,
                upper,
                lower,
            }
        }));
    }
    for (&input_free_spinor, values) in &source.components {
        if lowered_spinor_index(input_free_spinor, root, spinors) == Some(output_free_spinor) {
            streams.push(SparseLoweringStream::Identity { values, index: 0 });
        }
    }

    let mut heap = BinaryHeap::<Reverse<(u32, usize, i64)>>::new();
    for (stream_index, stream) in streams.iter_mut().enumerate() {
        if let Some((mask, coefficient)) = stream.next()? {
            heap.push(Reverse((mask, stream_index, coefficient)));
        }
    }
    let mut output = Vec::<(u32, i64)>::new();
    let mut local_maximum = 0_i128;
    while let Some(Reverse((mask, stream_index, coefficient))) = heap.pop() {
        if let Some((next_mask, next_coefficient)) = streams[stream_index].next()? {
            heap.push(Reverse((next_mask, stream_index, next_coefficient)));
        }
        let mut accumulated = i128::from(coefficient);
        while heap.peek().is_some_and(|entry| entry.0.0 == mask) {
            let Reverse((_, next_stream_index, next_coefficient)) = heap.pop().unwrap();
            accumulated = accumulated
                .checked_add(i128::from(next_coefficient))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "i128 sparse lowering accumulator overflow",
                    )
                })?;
            if let Some((next_mask, next_coefficient)) = streams[next_stream_index].next()? {
                heap.push(Reverse((next_mask, next_stream_index, next_coefficient)));
            }
        }
        if accumulated == 0 {
            continue;
        }
        local_maximum = local_maximum.max(accumulated.abs());
        let coefficient = i64::try_from(accumulated).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "sparse lowering result exceeds i64",
            )
        })?;
        output.push((mask, coefficient));
    }
    Ok((output_free_spinor, output, local_maximum))
}

fn lower_sparse_coupled_state64(
    source: &CoupledSparseState64,
    root: usize,
    maximum: &mut i128,
) -> io::Result<CoupledSparseState64> {
    if root >= SIMPLE_ROOTS.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sparse lowering simple root is out of range",
        ));
    }
    #[cfg(feature = "cuda")]
    if std::env::var("ADINKRA_FX_CUDA_SPARSE_LOWERING").as_deref() != Ok("0") {
        let input_terms = source
            .components
            .values()
            .try_fold(0_usize, |total, values| total.checked_add(values.len()))
            .ok_or_else(|| io::Error::other("sparse lowering input count overflow"))?;
        if !cuda_sparse_lowering_fits_host_cap(input_terms)? || input_terms == 0 {
            return lower_sparse_coupled_state64_bounded(source, root, maximum);
        }
        let entries = source
            .components
            .iter()
            .flat_map(|(&spinor, values)| {
                values.iter().map(move |&(mask, coefficient)| {
                    (((spinor as u64) << 32) | u64::from(mask), coefficient)
                })
            })
            .collect::<Vec<_>>();
        let device = std::env::var("ADINKRA_FX_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0);
        let (lowered, _) = crate::eleven_dimensional_second_momentum_gpu::lower_sparse_exact(
            &entries, root, device,
        )
        .map_err(io::Error::other)?;
        let mut components = BTreeMap::<usize, Vec<(u32, i64)>>::new();
        for (key, coefficient) in lowered {
            *maximum = (*maximum).max(i128::from(coefficient).abs());
            components
                .entry((key >> 32) as usize)
                .or_default()
                .push((key as u32, coefficient));
        }
        return Ok(CoupledSparseState64 { components });
    }
    lower_sparse_coupled_state64_bounded(source, root, maximum)
}

fn visit_independent_sparse_coupled_words<F>(
    highest: &CoupledSparseState64,
    words: &[Vec<u8>],
    maximum: &mut i128,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(usize, &CoupledSparseState64) -> io::Result<()>,
{
    visit_independent_coupled_word_handles(
        highest,
        words,
        maximum,
        &mut lower_sparse_coupled_root_word64,
        visit,
    )
}

fn visit_independent_sparse_coupled_word_events_from<F>(
    highest: &CoupledSparseState64,
    words: &[Vec<u8>],
    start_word_ordinal: usize,
    maximum: &mut i128,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(CoupledWordStateEvent<'_, CoupledSparseState64>) -> io::Result<()>,
{
    visit_independent_sparse_coupled_word_events_range(
        highest,
        words,
        start_word_ordinal,
        words.len(),
        maximum,
        visit,
    )
}

fn visit_independent_sparse_coupled_word_events_range<F>(
    highest: &CoupledSparseState64,
    words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    maximum: &mut i128,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(CoupledWordStateEvent<'_, CoupledSparseState64>) -> io::Result<()>,
{
    visit_independent_coupled_word_handle_events_range(
        highest,
        words,
        start_word_ordinal,
        end_word_ordinal_exclusive,
        maximum,
        &mut lower_sparse_coupled_root_word64,
        visit,
    )
}

fn common_root_prefix_length(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Lower a complete root-word segment without cloning its starting state.
/// Keeping this boundary word-oriented allows a persistent accelerator to
/// replace the scalar backend later without changing traversal or callbacks.
fn lower_sparse_coupled_root_word64(
    source: &CoupledSparseState64,
    roots: &[u8],
    maximum: &mut i128,
) -> io::Result<CoupledSparseState64> {
    let Some((&first, rest)) = roots.split_first() else {
        return Ok(source.clone());
    };
    if !(1..=5).contains(&first) || rest.iter().any(|root| !(1..=5).contains(root)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PBW lowering word contains an invalid simple root",
        ));
    }
    let mut state = lower_sparse_coupled_state64(source, usize::from(first - 1), maximum)?;
    for &simple_root in rest {
        state = lower_sparse_coupled_state64(&state, usize::from(simple_root - 1), maximum)?;
    }
    Ok(state)
}

/// Visit opaque backend handles in caller order while retaining at most one
/// shared-prefix handle. The next adjacent LCP is materialized once and reused.
/// When the LCP moves upward, its handle is recomputed only after the terminal
/// callback, so terminal and branch handles never accumulate into an unbounded
/// trie. No ordering assumption is made beyond the caller's canonical order;
/// the traversal exploits whatever adjacent prefixes that order provides.
///
/// `H` may be a host state or a device-resident allocation. Only `lower_word`
/// creates a new handle, and only `visit` needs to inspect a terminal handle.
fn visit_independent_coupled_word_handles<H, F, L>(
    highest: &H,
    words: &[Vec<u8>],
    maximum: &mut i128,
    lower_word: &mut L,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(usize, &H) -> io::Result<()>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
{
    visit_independent_coupled_word_handle_events_from(
        highest,
        words,
        0,
        maximum,
        lower_word,
        &mut |event| match event {
            CoupledWordStateEvent::State { ordinal, state } => visit(ordinal, state),
            CoupledWordStateEvent::WordLoweringStart { .. }
            | CoupledWordStateEvent::WordStart { .. }
            | CoupledWordStateEvent::WordEnd { .. } => Ok(()),
        },
    )
}

pub(crate) enum CoupledWordStateEvent<'a, H> {
    WordLoweringStart { ordinal: usize, pbw_word: &'a [u8] },
    WordStart { ordinal: usize, pbw_word: &'a [u8] },
    State { ordinal: usize, state: &'a H },
    WordEnd { ordinal: usize },
}

fn emit_coupled_word_state_events<H, F>(
    ordinal: usize,
    pbw_word: &[u8],
    state: &H,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<()>,
{
    visit(CoupledWordStateEvent::WordStart { ordinal, pbw_word })?;
    visit(CoupledWordStateEvent::State { ordinal, state })?;
    visit(CoupledWordStateEvent::WordEnd { ordinal })
}

pub(crate) fn visit_independent_coupled_word_handle_events_from<H, F, L>(
    highest: &H,
    words: &[Vec<u8>],
    start_word_ordinal: usize,
    maximum: &mut i128,
    lower_word: &mut L,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<()>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
{
    visit_independent_coupled_word_handle_events_range(
        highest,
        words,
        start_word_ordinal,
        words.len(),
        maximum,
        lower_word,
        visit,
    )
}

pub(crate) fn visit_independent_coupled_word_handle_events_range<H, F, L>(
    highest: &H,
    words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    maximum: &mut i128,
    lower_word: &mut L,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<()>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
{
    let range = words
        .get(start_word_ordinal..end_word_ordinal_exclusive)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "PBW word range is reversed or exceeds the requested plan",
            )
        })?;
    let mut shared_prefix = None::<(Vec<u8>, H)>;
    let mut pending_prefix_depth = 0_usize;
    for (relative_ordinal, word) in range.iter().enumerate() {
        let ordinal = start_word_ordinal + relative_ordinal;
        visit(CoupledWordStateEvent::WordLoweringStart {
            ordinal,
            pbw_word: word,
        })?;
        if pending_prefix_depth != 0 {
            debug_assert!(shared_prefix.is_none());
            let prefix_depth = std::mem::take(&mut pending_prefix_depth);
            let state = lower_word(highest, &word[..prefix_depth], maximum)?;
            shared_prefix = Some((word[..prefix_depth].to_vec(), state));
        }
        let next_lcp = range
            .get(relative_ordinal + 1)
            .map_or(0, |next| common_root_prefix_length(word, next));
        if shared_prefix
            .as_ref()
            .is_some_and(|(prefix, _)| !word.starts_with(prefix))
        {
            shared_prefix = None;
        }
        let start_depth = shared_prefix.as_ref().map_or(0, |(prefix, _)| prefix.len());

        if next_lcp > start_depth {
            let base = shared_prefix.as_ref().map_or(highest, |(_, state)| state);
            let next_shared = lower_word(base, &word[start_depth..next_lcp], maximum)?;
            drop(shared_prefix.take());
            if next_lcp == word.len() {
                emit_coupled_word_state_events(ordinal, word, &next_shared, visit)?;
            } else {
                let terminal = lower_word(&next_shared, &word[next_lcp..], maximum)?;
                emit_coupled_word_state_events(ordinal, word, &terminal, visit)?;
            }
            shared_prefix = Some((word[..next_lcp].to_vec(), next_shared));
            continue;
        }

        if start_depth == word.len() {
            let terminal = shared_prefix.as_ref().map_or(highest, |(_, state)| state);
            emit_coupled_word_state_events(ordinal, word, terminal, visit)?;
        } else {
            let base = shared_prefix.as_ref().map_or(highest, |(_, state)| state);
            let terminal = lower_word(base, &word[start_depth..], maximum)?;
            emit_coupled_word_state_events(ordinal, word, &terminal, visit)?;
        }

        if next_lcp == 0 {
            shared_prefix = None;
        } else if next_lcp < start_depth {
            drop(shared_prefix.take());
            pending_prefix_depth = next_lcp;
        } else {
            debug_assert_eq!(next_lcp, start_depth);
        }
    }
    Ok(())
}

fn build_derivative_candidate(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
    target_terms: &[crate::eleven_dimensional_bridge::DirectHookTargetCouplingTerm],
) -> (CoupledSparseState, CoupledSparseState, i128) {
    let (mut level16, highest, mut maximum) =
        materialize_coupled_highest(LEVEL16_PROBLEM, abstract_certificate, fixture_bytes, 2);
    let leading = dense_coupled_to_sparse(&mut level16, &highest);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut level17 = ExteriorModel::new(17);
    let mut accumulated = BTreeMap::<usize, (Weight, Vec<i128>)>::new();
    for term in target_terms {
        let target_state = coupled_state_for_word(
            &mut level16,
            &highest,
            &term.pbw_word_simple_roots,
            &mut cache,
            &mut maximum,
        );
        assert_eq!(target_state.total_weight, term.vector_spinor_weight);
        let outer_bit = 1_u32 << term.outer_spinor_index;
        let lower_bits = outer_bit - 1;
        for (&free_spinor, source) in &target_state.components {
            let destination_weight = add(source.weight, level16.spinors[term.outer_spinor_index]);
            assert_eq!(
                add(destination_weight, level16.spinors[free_spinor]),
                LEVEL17_HOOK_PROBLEM.target_weight
            );
            let source_masks = level16.space(source.weight).masks.clone();
            let destination_space = level17.space(destination_weight);
            let destination = accumulated
                .entry(free_spinor)
                .or_insert_with(|| (destination_weight, vec![0; destination_space.masks.len()]));
            assert_eq!(destination.0, destination_weight);
            for (mask, coefficient) in source_masks.into_iter().zip(&source.coefficients) {
                if *coefficient == 0 || mask & outer_bit != 0 {
                    continue;
                }
                let sign = if (mask & lower_bits).count_ones() % 2 == 0 {
                    1_i128
                } else {
                    -1_i128
                };
                let output_mask = mask | outer_bit;
                let output_index = destination_space.index[&output_mask];
                let value = destination.1[output_index]
                    .checked_add(
                        i128::from(term.primitive_coefficient) * i128::from(*coefficient) * sign,
                    )
                    .expect("i128 overflow in exterior derivative candidate");
                maximum = maximum.max(value.abs());
                destination.1[output_index] = value;
            }
        }
    }
    let components = accumulated
        .into_iter()
        .map(|(spinor, (weight, coefficients))| {
            let masks = level17.space(weight).masks.clone();
            let values = masks
                .into_iter()
                .zip(coefficients)
                .filter(|(_, coefficient)| *coefficient != 0)
                .collect();
            (spinor, values)
        })
        .collect();
    (CoupledSparseState { components }, leading, maximum)
}

fn build_first_momentum_correction_highest(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
    coefficient_width_bytes: usize,
    recoupling: &crate::eleven_dimensional_bridge::FirstMomentumRecouplingAudit,
) -> (ExteriorModel, MomentumCoupledDenseState, i128) {
    assert_eq!(
        recoupling.intermediate_dynkin_label,
        abstract_certificate.target_dynkin_label
    );
    assert!(recoupling.passed);
    let (mut model, intermediate_highest, mut maximum) = materialize_coupled_highest(
        problem,
        abstract_certificate,
        fixture_bytes,
        coefficient_width_bytes,
    );
    let mut intermediate_cache = BTreeMap::from([(Vec::new(), intermediate_highest.clone())]);
    let mut components = BTreeMap::new();
    for term in &recoupling.terms {
        let state = coupled_state_for_word(
            &mut model,
            &intermediate_highest,
            &term.intermediate_pbw_word_simple_roots,
            &mut intermediate_cache,
            &mut maximum,
        );
        assert_eq!(state.total_weight, term.intermediate_weight);
        for (&free_spinor, exterior) in &state.components {
            assert_eq!(
                add(
                    momentum_vector_weights()[term.momentum_vector_index],
                    add(model.spinors[free_spinor], exterior.weight),
                ),
                TARGET_WEIGHT
            );
            add_scaled_momentum_dense_component(
                &mut components,
                term.momentum_vector_index,
                free_spinor,
                exterior,
                term.primitive_coefficient,
                &mut maximum,
            );
        }
    }
    (
        model,
        MomentumCoupledDenseState {
            total_weight: TARGET_WEIGHT,
            components,
        },
        maximum,
    )
}

fn accumulate_momentum_hook_entry(
    entries: &mut HashMap<(usize, usize, u32), (i128, i128)>,
    key: (usize, usize, u32),
    real: i128,
    imaginary: i128,
    maximum: &mut i128,
) {
    if real == 0 && imaginary == 0 {
        return;
    }
    let value = entries.entry(key).or_insert((0, 0));
    value.0 = value
        .0
        .checked_add(real)
        .expect("real pD15 residual coefficient overflow");
    value.1 = value
        .1
        .checked_add(imaginary)
        .expect("imaginary pD15 residual coefficient overflow");
    *maximum = (*maximum).max(value.0.abs()).max(value.1.abs());
}

fn finalize_momentum_hook_entries(
    entries: HashMap<(usize, usize, u32), (i128, i128)>,
) -> Vec<MomentumHookEntry> {
    entries
        .into_iter()
        .filter_map(
            |((momentum_vector_index, free_spinor_index, exterior_mask), (real, imaginary))| {
                (real != 0 || imaginary != 0).then_some(MomentumHookEntry {
                    momentum_vector_index,
                    free_spinor_index,
                    exterior_mask,
                    real,
                    imaginary,
                })
            },
        )
        .collect()
}

fn build_first_momentum_correction_residual(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
    coefficient_width_bytes: usize,
    recoupling: &crate::eleven_dimensional_bridge::FirstMomentumRecouplingAudit,
    hook_terms: &[crate::eleven_dimensional_bridge::DirectHookTargetCouplingTerm],
) -> (Vec<MomentumHookEntry>, i128) {
    let (mut model, highest, mut maximum) = build_first_momentum_correction_highest(
        problem,
        abstract_certificate,
        fixture_bytes,
        coefficient_width_bytes,
        recoupling,
    );
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut entries = HashMap::new();
    for term in hook_terms {
        let state = momentum_coupled_state_for_word(
            &mut model,
            &highest,
            &term.pbw_word_simple_roots,
            &mut cache,
            &mut maximum,
        );
        assert_eq!(state.total_weight, term.vector_spinor_weight);
        let outer_bit = 1_u32 << term.outer_spinor_index;
        let lower_bits = outer_bit - 1;
        for (&(momentum_vector, free_spinor), exterior) in &state.components {
            let source_masks = model.space(exterior.weight).masks.clone();
            for (mask, source_coefficient) in source_masks.into_iter().zip(&exterior.coefficients) {
                if *source_coefficient == 0 || mask & outer_bit != 0 {
                    continue;
                }
                let wedge_sign = if (mask & lower_bits).count_ones() % 2 == 0 {
                    1_i128
                } else {
                    -1_i128
                };
                let coefficient = i128::from(term.primitive_coefficient)
                    * i128::from(*source_coefficient)
                    * wedge_sign;
                accumulate_momentum_hook_entry(
                    &mut entries,
                    (momentum_vector, free_spinor, mask | outer_bit),
                    coefficient,
                    0,
                    &mut maximum,
                );
            }
        }
    }
    (finalize_momentum_hook_entries(entries), maximum)
}

pub(crate) fn translation_weight_basis_coefficients() -> Vec<Vec<[(i64, i64); 11]>> {
    let bilinears = crate::eleven_dimensional_clifford::translation_bilinears();
    let mut coefficients = vec![vec![[(0_i64, 0_i64); 11]; 32]; 32];
    for outer in 0..32 {
        for contracted in 0..32 {
            for axis in 0..5 {
                let even = &bilinears[2 * axis][outer][contracted];
                let odd = &bilinears[2 * axis + 1][outer][contracted];
                assert_eq!(*even.re.denom(), 1);
                assert_eq!(*even.im.denom(), 1);
                assert_eq!(*odd.re.denom(), 1);
                assert_eq!(*odd.im.denom(), 1);
                let even_real = *even.re.numer();
                let even_imaginary = *even.im.numer();
                let odd_real = *odd.re.numer();
                let odd_imaginary = *odd.im.numer();
                coefficients[outer][contracted][2 * axis] =
                    (even_real + odd_imaginary, even_imaginary - odd_real);
                coefficients[outer][contracted][2 * axis + 1] =
                    (even_real - odd_imaginary, even_imaginary + odd_real);
            }
            let zero = &bilinears[10][outer][contracted];
            assert_eq!(*zero.re.denom(), 1);
            assert_eq!(*zero.im.denom(), 1);
            coefficients[outer][contracted][10] = (*zero.re.numer(), *zero.im.numer());
        }
    }
    coefficients
}

fn build_leading_anticommutator_residual(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_bytes: &[u8],
    hook_terms: &[crate::eleven_dimensional_bridge::DirectHookTargetCouplingTerm],
    translation_coefficients: &[Vec<[(i64, i64); 11]>],
) -> (Vec<MomentumHookEntry>, i128) {
    let (mut model, highest, mut maximum) =
        materialize_coupled_highest(LEVEL16_PROBLEM, abstract_certificate, fixture_bytes, 2);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut entries = HashMap::new();
    for term in hook_terms {
        let state = coupled_state_for_word(
            &mut model,
            &highest,
            &term.pbw_word_simple_roots,
            &mut cache,
            &mut maximum,
        );
        assert_eq!(state.total_weight, term.vector_spinor_weight);
        for (&free_spinor, exterior) in &state.components {
            let source_masks = model.space(exterior.weight).masks.clone();
            for (mask, source_coefficient) in source_masks.into_iter().zip(&exterior.coefficients) {
                if *source_coefficient == 0 {
                    continue;
                }
                let mut remaining = mask;
                let mut position = 0_u32;
                while remaining != 0 {
                    let contracted_spinor = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    let contraction_sign = if position % 2 == 0 { 1_i128 } else { -1_i128 };
                    position += 1;
                    let output_mask = mask ^ (1_u32 << contracted_spinor);
                    let common = i128::from(term.primitive_coefficient)
                        * i128::from(*source_coefficient)
                        * contraction_sign;
                    for momentum_vector in 0..11 {
                        let (real, imaginary) = translation_coefficients[term.outer_spinor_index]
                            [contracted_spinor][momentum_vector];
                        if real != 0 || imaginary != 0 {
                            assert_eq!(
                                momentum_vector_weights()[momentum_vector],
                                add(
                                    model.spinors[term.outer_spinor_index],
                                    model.spinors[contracted_spinor],
                                )
                            );
                        }
                        accumulate_momentum_hook_entry(
                            &mut entries,
                            (momentum_vector, free_spinor, output_mask),
                            common * i128::from(real),
                            common * i128::from(imaginary),
                            &mut maximum,
                        );
                    }
                }
            }
        }
    }
    (finalize_momentum_hook_entries(entries), maximum)
}

fn build_scalar_factorizing_candidate() -> (CoupledSparseState, i128) {
    let mut level15 = ExteriorModel::new(15);
    let highest =
        level15.fixture_state(TARGET_DYNKIN_LABEL, SCALAR_BRIDGE_VECTOR_SPINOR_FIXTURE, 2);
    assert_eq!(highest.weight, TARGET_WEIGHT);
    let source_masks = level15.space(highest.weight).masks.clone();
    let charge = crate::eleven_dimensional_clifford::spinor_charge_bilinear();
    let zero = Ratio::from_integer(0);
    let phase = charge
        .iter()
        .flat_map(|row| row.iter())
        .find(|value| value.re != zero || value.im != zero)
        .unwrap()
        .clone();
    let spinors = level15.spinors;
    let mut level16 = ExteriorModel::new(16);
    let mut accumulated = BTreeMap::<usize, (Weight, Vec<i128>)>::new();
    let mut maximum = 0_i128;
    for derivative_spinor in 0..32 {
        let derivative_bit = 1_u32 << derivative_spinor;
        let lower_bits = derivative_bit - 1;
        for free_spinor in 0..32 {
            let normalized = charge[derivative_spinor][free_spinor].clone() / phase.clone();
            assert_eq!(normalized.im, zero);
            assert_eq!(*normalized.re.denom(), 1);
            let contraction = *normalized.re.numer();
            if contraction == 0 {
                continue;
            }
            let destination_weight = add(highest.weight, spinors[derivative_spinor]);
            assert_eq!(add(destination_weight, spinors[free_spinor]), TARGET_WEIGHT);
            let destination_space = level16.space(destination_weight);
            let destination = accumulated
                .entry(free_spinor)
                .or_insert_with(|| (destination_weight, vec![0; destination_space.masks.len()]));
            for (mask, coefficient) in source_masks.iter().zip(&highest.coefficients) {
                if *coefficient == 0 || mask & derivative_bit != 0 {
                    continue;
                }
                let sign = if (mask & lower_bits).count_ones() % 2 == 0 {
                    1_i128
                } else {
                    -1_i128
                };
                let output_index = destination_space.index[&(mask | derivative_bit)];
                let value = destination.1[output_index]
                    .checked_add(i128::from(contraction) * i128::from(*coefficient) * sign)
                    .expect("i128 overflow in scalar-factorizing candidate");
                maximum = maximum.max(value.abs());
                destination.1[output_index] = value;
            }
        }
    }
    let components = accumulated
        .into_iter()
        .map(|(spinor, (weight, coefficients))| {
            let masks = level16.space(weight).masks.clone();
            (
                spinor,
                masks
                    .into_iter()
                    .zip(coefficients)
                    .filter(|(_, coefficient)| *coefficient != 0)
                    .collect(),
            )
        })
        .collect();
    (CoupledSparseState { components }, maximum)
}

fn sparse_coupled_dot(left: &CoupledSparseState, right: &CoupledSparseState) -> i128 {
    let mut total = 0_i128;
    for (spinor, left_values) in &left.components {
        let Some(right_values) = right.components.get(spinor) else {
            continue;
        };
        let mut left_index = 0;
        let mut right_index = 0;
        while left_index < left_values.len() && right_index < right_values.len() {
            let (left_mask, left_value) = left_values[left_index];
            let (right_mask, right_value) = right_values[right_index];
            match left_mask.cmp(&right_mask) {
                std::cmp::Ordering::Less => left_index += 1,
                std::cmp::Ordering::Greater => right_index += 1,
                std::cmp::Ordering::Equal => {
                    total = total
                        .checked_add(
                            left_value
                                .checked_mul(right_value)
                                .expect("i128 overflow in sparse coupled product"),
                        )
                        .expect("i128 overflow in sparse coupled dot product");
                    left_index += 1;
                    right_index += 1;
                }
            }
        }
    }
    total
}

fn solve_bigint_system(
    matrix: &[Vec<BigInt>],
    right_hand_side: &[BigInt],
) -> Option<Vec<Ratio<BigInt>>> {
    let dimension = matrix.len();
    assert_eq!(right_hand_side.len(), dimension);
    assert!(matrix.iter().all(|row| row.len() == dimension));
    let zero = Ratio::from_integer(BigInt::zero());
    let mut augmented = matrix
        .iter()
        .zip(right_hand_side)
        .map(|(row, right)| {
            let mut values = row
                .iter()
                .cloned()
                .map(Ratio::from_integer)
                .collect::<Vec<_>>();
            values.push(Ratio::from_integer(right.clone()));
            values
        })
        .collect::<Vec<_>>();
    for column in 0..dimension {
        let pivot = (column..dimension).find(|row| augmented[*row][column] != zero)?;
        augmented.swap(column, pivot);
        let normalization = augmented[column][column].clone();
        for value in &mut augmented[column][column..=dimension] {
            *value /= normalization.clone();
        }
        let pivot_row = augmented[column].clone();
        for row in 0..dimension {
            if row == column || augmented[row][column] == zero {
                continue;
            }
            let factor = augmented[row][column].clone();
            for index in column..=dimension {
                augmented[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
    }
    Some(
        augmented
            .into_iter()
            .map(|row| row[dimension].clone())
            .collect(),
    )
}

pub(crate) fn rational_matrix_rank(matrix: &[Vec<Ratio<BigInt>>]) -> usize {
    if matrix.is_empty() {
        return 0;
    }
    let zero = Ratio::from_integer(BigInt::zero());
    let mut reduced = matrix.to_vec();
    let columns = reduced[0].len();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero) else {
            continue;
        };
        reduced.swap(rank, pivot);
        let normalization = reduced[rank][column].clone();
        for value in &mut reduced[rank][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = reduced[rank].clone();
        for row in (rank + 1)..reduced.len() {
            let factor = reduced[row][column].clone();
            if factor == zero {
                continue;
            }
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    rank
}

pub(crate) fn rational_nullspace(matrix: &[Vec<Ratio<BigInt>>]) -> Vec<Vec<Ratio<BigInt>>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let columns = matrix[0].len();
    let zero = Ratio::from_integer(BigInt::zero());
    let mut reduced = matrix.to_vec();
    let mut pivot_columns = Vec::new();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..reduced.len()).find(|row| reduced[*row][column] != zero) else {
            continue;
        };
        reduced.swap(rank, pivot);
        let normalization = reduced[rank][column].clone();
        for value in &mut reduced[rank][column..] {
            *value /= normalization.clone();
        }
        let pivot_row = reduced[rank].clone();
        for row in 0..reduced.len() {
            if row == rank || reduced[row][column] == zero {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..columns {
                reduced[row][index] -= factor.clone() * pivot_row[index].clone();
            }
        }
        pivot_columns.push(column);
        rank += 1;
        if rank == reduced.len() {
            break;
        }
    }
    (0..columns)
        .filter(|column| !pivot_columns.contains(column))
        .map(|free| {
            let mut vector = vec![zero.clone(); columns];
            vector[free] = Ratio::from_integer(BigInt::one());
            for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
                vector[pivot] = -reduced[row][free].clone();
            }
            vector
        })
        .collect()
}

pub(crate) fn primitive_bigint_vector(vector: &[Ratio<BigInt>]) -> Vec<BigInt> {
    let denominator = vector.iter().fold(BigInt::one(), |common, coefficient| {
        bigint_lcm(common, coefficient.denom().clone())
    });
    let mut integers = vector
        .iter()
        .map(|coefficient| coefficient.numer() * (&denominator / coefficient.denom()))
        .collect::<Vec<_>>();
    let gcd = integers.iter().fold(BigInt::zero(), |common, value| {
        bigint_gcd(common, value.clone())
    });
    assert!(!gcd.is_zero());
    for value in &mut integers {
        *value /= &gcd;
    }
    if integers
        .iter()
        .find(|value| !value.is_zero())
        .is_some_and(BigInt::is_negative)
    {
        for value in &mut integers {
            *value = -value.clone();
        }
    }
    integers
}

fn matrix_times_integer_vector_is_zero(matrix: &[Vec<Ratio<BigInt>>], vector: &[BigInt]) -> bool {
    matrix.iter().all(|row| {
        row.iter()
            .zip(vector)
            .fold(
                Ratio::from_integer(BigInt::zero()),
                |sum, (coefficient, value)| {
                    sum + coefficient.clone() * Ratio::from_integer(value.clone())
                },
            )
            .is_zero()
    })
}

fn rational_entry(value: &Ratio<BigInt>) -> RationalMatrixEntry {
    RationalMatrixEntry {
        numerator: value.numer().to_string(),
        denominator: value.denom().to_string(),
    }
}

fn parse_rational_entry(value: &RationalMatrixEntry) -> Ratio<BigInt> {
    Ratio::new(
        BigInt::parse_bytes(value.numerator.as_bytes(), 10).unwrap(),
        BigInt::parse_bytes(value.denominator.as_bytes(), 10).unwrap(),
    )
}

const JOINT_FUNCTIONAL_BUCKETS: usize = 64;
const JOINT_FUNCTIONAL_SEEDS: [u64; 4] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
];

fn splitmix64_local(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn momentum_hook_functionals(entries: &[MomentumHookEntry]) -> Vec<i128> {
    let rows_per_seed = 2 * JOINT_FUNCTIONAL_BUCKETS;
    let mut output = vec![0_i128; JOINT_FUNCTIONAL_SEEDS.len() * rows_per_seed];
    for entry in entries {
        let key = u64::from(entry.exterior_mask)
            | (u64::try_from(entry.free_spinor_index).unwrap() << 32)
            | (u64::try_from(entry.momentum_vector_index).unwrap() << 37);
        for (seed_index, seed) in JOINT_FUNCTIONAL_SEEDS.iter().enumerate() {
            let hash = splitmix64_local(key ^ seed);
            let bucket = (hash as usize) % JOINT_FUNCTIONAL_BUCKETS;
            let sign = if hash >> 63 == 0 { 1_i128 } else { -1_i128 };
            let base = seed_index * rows_per_seed;
            output[base + bucket] = output[base + bucket]
                .checked_add(sign * entry.real)
                .expect("i128 overflow in exact real residual functional");
            output[base + JOINT_FUNCTIONAL_BUCKETS + bucket] = output
                [base + JOINT_FUNCTIONAL_BUCKETS + bucket]
                .checked_add(sign * entry.imaginary)
                .expect("i128 overflow in exact imaginary residual functional");
        }
    }
    output
}

fn rational_row_span_contains(rows: &[Vec<Ratio<BigInt>>], candidate: &[Ratio<BigInt>]) -> bool {
    if rows.is_empty() {
        return candidate.iter().all(Ratio::is_zero);
    }
    let rank = rational_matrix_rank(rows);
    let mut augmented = rows.to_vec();
    augmented.push(candidate.to_vec());
    rational_matrix_rank(&augmented) == rank
}

fn binomial_u128(n: u128, k: u128) -> u128 {
    let k = k.min(n - k);
    (1..=k).fold(1_u128, |value, index| value * (n - k + index) / index)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock precedes the Unix epoch")
        .as_secs()
}

pub fn joint_column_specs() -> Vec<JointColumnSpec> {
    let mut specs = Vec::new();
    for fixture in crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures() {
        specs.push(JointColumnSpec {
            ordinal: specs.len(),
            label: format!("{}#{}", fixture.dynkin_label, fixture.copy),
            kind: "leading".to_string(),
            source_dynkin_label: fixture.dynkin_label.to_string(),
            source_copy: fixture.copy,
            intermediate_dynkin_label: None,
        });
    }
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
    let fixtures_by_source =
        fixtures
            .iter()
            .fold(BTreeMap::<&str, Vec<_>>::new(), |mut grouped, fixture| {
                grouped
                    .entry(fixture.dynkin_label)
                    .or_default()
                    .push(*fixture);
                grouped
            });
    for ((source, intermediate), copies) in first_momentum_copy_manifest() {
        for copy in copies {
            assert!(
                fixtures_by_source[&source.as_str()]
                    .iter()
                    .any(|fixture| fixture.copy == copy)
            );
            specs.push(JointColumnSpec {
                ordinal: specs.len(),
                label: format!("{source}#{copy}->{intermediate}"),
                kind: "first-momentum".to_string(),
                source_dynkin_label: source.clone(),
                source_copy: copy,
                intermediate_dynkin_label: Some(intermediate.clone()),
            });
        }
    }
    assert_eq!(specs.len(), 56);
    specs
}

pub fn build_joint_column(
    ordinal: usize,
) -> (JointColumnSpec, Vec<MomentumHookEntry>, i128, String) {
    let specs = joint_column_specs();
    let spec = specs
        .get(ordinal)
        .unwrap_or_else(|| panic!("joint column ordinal {ordinal} is outside 0..56"))
        .clone();
    let hook_terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
    if spec.kind == "leading" {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let abstract_certificate = build_abstract_from_fixture(
            LEVEL16_PROBLEM,
            first.dynkin_label,
            first.copy,
            2,
            first.bytes,
        )
        .0;
        let translation_coefficients = translation_weight_basis_coefficients();
        let (residual, maximum) = build_leading_anticommutator_residual(
            &abstract_certificate,
            fixture.bytes,
            &hook_terms,
            &translation_coefficients,
        );
        (spec, residual, maximum, sha256_bytes(fixture.bytes))
    } else {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let intermediate = spec.intermediate_dynkin_label.as_deref().unwrap();
        let problem = first_momentum_problem(intermediate);
        let abstract_certificate = build_abstract_from_fixture(
            problem,
            first.dynkin_label,
            first.copy,
            fixture_coefficient_width(first.artifact),
            first.bytes,
        )
        .0;
        let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
        let recoupling = recouplings
            .iter()
            .find(|audit| audit.intermediate_dynkin_label == intermediate)
            .unwrap();
        let (residual, maximum) = build_first_momentum_correction_residual(
            problem,
            &abstract_certificate,
            fixture.bytes,
            fixture_coefficient_width(fixture.artifact),
            recoupling,
            &hook_terms,
        );
        (spec, residual, maximum, sha256_bytes(fixture.bytes))
    }
}

pub fn visit_zero_momentum_gauge_composition_components<F>(
    gauge_form_degree: usize,
    leading_ordinal: usize,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String)>
where
    F: FnMut(usize, &[usize], &[ZeroMomentumGaugeCompositionEntry]) -> io::Result<()>,
{
    assert!(gauge_form_degree <= 5);
    let spec = joint_column_specs()
        .get(leading_ordinal)
        .unwrap_or_else(|| panic!("leading column ordinal {leading_ordinal} is outside 0..12"))
        .clone();
    assert_eq!(
        spec.kind, "leading",
        "zero-momentum gauge composition requires a leading operator"
    );

    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let fixture = fixtures
        .iter()
        .find(|fixture| {
            fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
        })
        .unwrap();
    let first = fixtures
        .iter()
        .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
        .unwrap();
    let abstract_certificate = build_abstract_from_fixture(
        LEVEL16_PROBLEM,
        first.dynkin_label,
        first.copy,
        2,
        first.bytes,
    )
    .0;
    let (mut model, highest, mut maximum) =
        materialize_coupled_highest(LEVEL16_PROBLEM, &abstract_certificate, fixture.bytes, 2);

    let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis()
        .into_iter()
        .filter(|(degree, _, _)| *degree == gauge_form_degree)
        .collect::<Vec<_>>();
    let parameter_basis = gauge_basis
        .iter()
        .map(|(_, indices, _)| indices.clone())
        .collect::<Vec<_>>();
    for (parameter_component_index, (_, component_indices, matrix)) in
        gauge_basis.iter().enumerate()
    {
        let mut accumulated = HashMap::<u32, (i128, i128)>::new();
        for (&free_spinor, exterior) in &highest.components {
            let source_masks = model.space(exterior.weight).masks.clone();
            for derivative_spinor in 0..32 {
                let bilinear = &matrix[free_spinor][derivative_spinor];
                if bilinear.re.is_zero() && bilinear.im.is_zero() {
                    continue;
                }
                assert_eq!(*bilinear.re.denom(), 1);
                assert_eq!(*bilinear.im.denom(), 1);
                let bilinear_real = i128::from(*bilinear.re.numer());
                let bilinear_imaginary = i128::from(*bilinear.im.numer());
                let derivative_bit = 1_u32 << derivative_spinor;
                for (mask, source_coefficient) in
                    source_masks.iter().copied().zip(&exterior.coefficients)
                {
                    if *source_coefficient == 0 {
                        continue;
                    }
                    // The gauge derivative acts on Lambda after the sixteen
                    // derivatives in A. Moving it left into ascending exterior
                    // normal order crosses the occupied indices greater than beta.
                    let Some(wedge_sign) = right_wedge_sign(mask, derivative_spinor) else {
                        continue;
                    };
                    let scale = i128::from(*source_coefficient) * wedge_sign;
                    let value = accumulated.entry(mask | derivative_bit).or_insert((0, 0));
                    value.0 = value
                        .0
                        .checked_add(scale * bilinear_real)
                        .expect("i128 overflow in zero-momentum gauge composition");
                    value.1 = value
                        .1
                        .checked_add(scale * bilinear_imaginary)
                        .expect("i128 overflow in zero-momentum gauge composition");
                    maximum = maximum.max(value.0.abs()).max(value.1.abs());
                }
            }
        }
        let mut component_residual = accumulated
            .into_iter()
            .filter_map(|(exterior_mask, (real, imaginary))| {
                (real != 0 || imaginary != 0).then_some(ZeroMomentumGaugeCompositionEntry {
                    parameter_component_index,
                    exterior_mask,
                    real,
                    imaginary,
                })
            })
            .collect::<Vec<_>>();
        component_residual.sort_by_key(|entry| entry.exterior_mask);
        visit(
            parameter_component_index,
            component_indices,
            &component_residual,
        )?;
    }
    Ok((spec, parameter_basis, maximum, sha256_bytes(fixture.bytes)))
}

fn multiply_gaussian_integers(
    left_real: i128,
    left_imaginary: i128,
    right_real: i128,
    right_imaginary: i128,
) -> (i128, i128) {
    let real = left_real
        .checked_mul(right_real)
        .and_then(|value| {
            left_imaginary
                .checked_mul(right_imaginary)
                .and_then(|other| value.checked_sub(other))
        })
        .expect("i128 overflow in Gaussian-integer real product");
    let imaginary = left_real
        .checked_mul(right_imaginary)
        .and_then(|value| {
            left_imaginary
                .checked_mul(right_real)
                .and_then(|other| value.checked_add(other))
        })
        .expect("i128 overflow in Gaussian-integer imaginary product");
    (real, imaginary)
}

fn accumulate_first_momentum_gauge_entry(
    accumulated: &mut HashMap<(usize, u32), (i128, i128)>,
    momentum_vector_index: usize,
    exterior_mask: u32,
    real: i128,
    imaginary: i128,
    maximum: &mut i128,
) {
    if real == 0 && imaginary == 0 {
        return;
    }
    let value = accumulated
        .entry((momentum_vector_index, exterior_mask))
        .or_insert((0, 0));
    value.0 = value
        .0
        .checked_add(real)
        .expect("i128 overflow in first-momentum gauge real accumulation");
    value.1 = value
        .1
        .checked_add(imaginary)
        .expect("i128 overflow in first-momentum gauge imaginary accumulation");
    *maximum = (*maximum).max(value.0.abs()).max(value.1.abs());
}

pub fn visit_first_momentum_gauge_composition_components<F>(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String)>
where
    F: FnMut(usize, &[usize], &[FirstMomentumGaugeCompositionEntry]) -> io::Result<()>,
{
    assert!(gauge_form_degree <= 5);
    let spec = joint_column_specs()
        .get(operator_ordinal)
        .unwrap_or_else(|| panic!("operator column ordinal {operator_ordinal} is outside 0..56"))
        .clone();
    let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis()
        .into_iter()
        .filter(|(degree, _, _)| *degree == gauge_form_degree)
        .collect::<Vec<_>>();
    let parameter_basis = gauge_basis
        .iter()
        .map(|(_, indices, _)| indices.clone())
        .collect::<Vec<_>>();

    if spec.kind == "leading" {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let abstract_certificate = build_abstract_from_fixture(
            LEVEL16_PROBLEM,
            first.dynkin_label,
            first.copy,
            2,
            first.bytes,
        )
        .0;
        let (mut model, highest, mut maximum) =
            materialize_coupled_highest(LEVEL16_PROBLEM, &abstract_certificate, fixture.bytes, 2);
        let translation_coefficients = translation_weight_basis_coefficients();
        for (parameter_component_index, (_, component_indices, matrix)) in
            gauge_basis.iter().enumerate()
        {
            let mut accumulated = HashMap::<(usize, u32), (i128, i128)>::new();
            for (&free_spinor, exterior) in &highest.components {
                let source_masks = model.space(exterior.weight).masks.clone();
                for derivative_spinor in 0..32 {
                    let gauge = &matrix[free_spinor][derivative_spinor];
                    if gauge.re.is_zero() && gauge.im.is_zero() {
                        continue;
                    }
                    assert_eq!(*gauge.re.denom(), 1);
                    assert_eq!(*gauge.im.denom(), 1);
                    let gauge_real = i128::from(*gauge.re.numer());
                    let gauge_imaginary = i128::from(*gauge.im.numer());
                    for (mask, source_coefficient) in
                        source_masks.iter().copied().zip(&exterior.coefficients)
                    {
                        if *source_coefficient == 0 {
                            continue;
                        }
                        let mut occupied = mask;
                        while occupied != 0 {
                            let contracted_spinor = occupied.trailing_zeros() as usize;
                            occupied &= occupied - 1;
                            let contraction_sign =
                                right_contraction_sign(mask, contracted_spinor).unwrap();
                            let output_mask = mask ^ (1_u32 << contracted_spinor);
                            assert_eq!(output_mask.count_ones(), 15);
                            let source_scale = i128::from(*source_coefficient) * contraction_sign;
                            for momentum_vector_index in 0..11 {
                                let (translation_real, translation_imaginary) =
                                    translation_coefficients[contracted_spinor][derivative_spinor]
                                        [momentum_vector_index];
                                if translation_real == 0 && translation_imaginary == 0 {
                                    continue;
                                }
                                let (real, imaginary) = multiply_gaussian_integers(
                                    gauge_real,
                                    gauge_imaginary,
                                    i128::from(translation_real),
                                    i128::from(translation_imaginary),
                                );
                                accumulate_first_momentum_gauge_entry(
                                    &mut accumulated,
                                    momentum_vector_index,
                                    output_mask,
                                    source_scale
                                        .checked_mul(real)
                                        .expect("i128 overflow in leading gauge real coefficient"),
                                    source_scale.checked_mul(imaginary).expect(
                                        "i128 overflow in leading gauge imaginary coefficient",
                                    ),
                                    &mut maximum,
                                );
                            }
                        }
                    }
                }
            }
            let mut component_residual = accumulated
                .into_iter()
                .filter_map(
                    |((momentum_vector_index, exterior_mask), (real, imaginary))| {
                        (real != 0 || imaginary != 0).then_some(
                            FirstMomentumGaugeCompositionEntry {
                                parameter_component_index,
                                momentum_vector_index,
                                exterior_mask,
                                real,
                                imaginary,
                            },
                        )
                    },
                )
                .collect::<Vec<_>>();
            component_residual
                .sort_by_key(|entry| (entry.momentum_vector_index, entry.exterior_mask));
            visit(
                parameter_component_index,
                component_indices,
                &component_residual,
            )?;
        }
        Ok((spec, parameter_basis, maximum, sha256_bytes(fixture.bytes)))
    } else {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let intermediate = spec.intermediate_dynkin_label.as_deref().unwrap();
        let problem = first_momentum_problem(intermediate);
        let abstract_certificate = build_abstract_from_fixture(
            problem,
            first.dynkin_label,
            first.copy,
            fixture_coefficient_width(first.artifact),
            first.bytes,
        )
        .0;
        let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
        let recoupling = recouplings
            .iter()
            .find(|audit| audit.intermediate_dynkin_label == intermediate)
            .unwrap();
        let (mut model, highest, mut maximum) = build_first_momentum_correction_highest(
            problem,
            &abstract_certificate,
            fixture.bytes,
            fixture_coefficient_width(fixture.artifact),
            recoupling,
        );
        for (parameter_component_index, (_, component_indices, matrix)) in
            gauge_basis.iter().enumerate()
        {
            let mut accumulated = HashMap::<(usize, u32), (i128, i128)>::new();
            for (&(momentum_vector_index, free_spinor), exterior) in &highest.components {
                let source_masks = model.space(exterior.weight).masks.clone();
                for derivative_spinor in 0..32 {
                    let gauge = &matrix[free_spinor][derivative_spinor];
                    if gauge.re.is_zero() && gauge.im.is_zero() {
                        continue;
                    }
                    assert_eq!(*gauge.re.denom(), 1);
                    assert_eq!(*gauge.im.denom(), 1);
                    let gauge_real = i128::from(*gauge.re.numer());
                    let gauge_imaginary = i128::from(*gauge.im.numer());
                    for (mask, source_coefficient) in
                        source_masks.iter().copied().zip(&exterior.coefficients)
                    {
                        if *source_coefficient == 0 {
                            continue;
                        }
                        let Some(wedge_sign) = right_wedge_sign(mask, derivative_spinor) else {
                            continue;
                        };
                        let output_mask = mask | (1_u32 << derivative_spinor);
                        assert_eq!(output_mask.count_ones(), 15);
                        let scale = i128::from(*source_coefficient) * wedge_sign;
                        accumulate_first_momentum_gauge_entry(
                            &mut accumulated,
                            momentum_vector_index,
                            output_mask,
                            scale
                                .checked_mul(gauge_real)
                                .expect("i128 overflow in correction gauge real coefficient"),
                            scale
                                .checked_mul(gauge_imaginary)
                                .expect("i128 overflow in correction gauge imaginary coefficient"),
                            &mut maximum,
                        );
                    }
                }
            }
            let mut component_residual = accumulated
                .into_iter()
                .filter_map(
                    |((momentum_vector_index, exterior_mask), (real, imaginary))| {
                        (real != 0 || imaginary != 0).then_some(
                            FirstMomentumGaugeCompositionEntry {
                                parameter_component_index,
                                momentum_vector_index,
                                exterior_mask,
                                real,
                                imaginary,
                            },
                        )
                    },
                )
                .collect::<Vec<_>>();
            component_residual
                .sort_by_key(|entry| (entry.momentum_vector_index, entry.exterior_mask));
            visit(
                parameter_component_index,
                component_indices,
                &component_residual,
            )?;
        }
        Ok((spec, parameter_basis, maximum, sha256_bytes(fixture.bytes)))
    }
}

pub fn visit_first_momentum_gauge_composition_terms<F>(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    selected_parameter_components: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String, u64)>
where
    F: FnMut(FirstMomentumGaugeCompositionEntry) -> io::Result<()>,
{
    assert!(gauge_form_degree <= 5);
    let spec = joint_column_specs()
        .get(operator_ordinal)
        .unwrap_or_else(|| panic!("operator column ordinal {operator_ordinal} is outside 0..56"))
        .clone();
    let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis()
        .into_iter()
        .filter(|(degree, _, _)| *degree == gauge_form_degree)
        .collect::<Vec<_>>();
    let parameter_basis = gauge_basis
        .iter()
        .map(|(_, indices, _)| indices.clone())
        .collect::<Vec<_>>();
    let mut emitted_nonzero_terms = 0_u64;

    if spec.kind == "leading" {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let abstract_certificate = build_abstract_from_fixture(
            LEVEL16_PROBLEM,
            first.dynkin_label,
            first.copy,
            2,
            first.bytes,
        )
        .0;
        let (mut model, highest, mut maximum) =
            materialize_coupled_highest(LEVEL16_PROBLEM, &abstract_certificate, fixture.bytes, 2);
        let translation_coefficients = translation_weight_basis_coefficients();
        for (parameter_component_index, (_, _, matrix)) in gauge_basis.iter().enumerate() {
            if selected_parameter_components
                .is_some_and(|selected| !selected.contains(&parameter_component_index))
            {
                continue;
            }
            for (&free_spinor, exterior) in &highest.components {
                let source_masks = model.space(exterior.weight).masks.clone();
                for derivative_spinor in 0..32 {
                    let gauge = &matrix[free_spinor][derivative_spinor];
                    if gauge.re.is_zero() && gauge.im.is_zero() {
                        continue;
                    }
                    assert_eq!(*gauge.re.denom(), 1);
                    assert_eq!(*gauge.im.denom(), 1);
                    let gauge_real = i128::from(*gauge.re.numer());
                    let gauge_imaginary = i128::from(*gauge.im.numer());
                    for (mask, source_coefficient) in
                        source_masks.iter().copied().zip(&exterior.coefficients)
                    {
                        if *source_coefficient == 0 {
                            continue;
                        }
                        let mut occupied = mask;
                        while occupied != 0 {
                            let contracted_spinor = occupied.trailing_zeros() as usize;
                            occupied &= occupied - 1;
                            let contraction_sign =
                                right_contraction_sign(mask, contracted_spinor).unwrap();
                            let output_mask = mask ^ (1_u32 << contracted_spinor);
                            assert_eq!(output_mask.count_ones(), 15);
                            let source_scale = i128::from(*source_coefficient) * contraction_sign;
                            for momentum_vector_index in 0..11 {
                                let (translation_real, translation_imaginary) =
                                    translation_coefficients[contracted_spinor][derivative_spinor]
                                        [momentum_vector_index];
                                if translation_real == 0 && translation_imaginary == 0 {
                                    continue;
                                }
                                let (product_real, product_imaginary) = multiply_gaussian_integers(
                                    gauge_real,
                                    gauge_imaginary,
                                    i128::from(translation_real),
                                    i128::from(translation_imaginary),
                                );
                                let real = source_scale.checked_mul(product_real).expect(
                                    "i128 overflow in streamed leading gauge real coefficient",
                                );
                                let imaginary = source_scale.checked_mul(product_imaginary).expect(
                                    "i128 overflow in streamed leading gauge imaginary coefficient",
                                );
                                if real == 0 && imaginary == 0 {
                                    continue;
                                }
                                maximum = maximum.max(real.abs()).max(imaginary.abs());
                                emitted_nonzero_terms = emitted_nonzero_terms
                                    .checked_add(1)
                                    .expect("first-momentum emitted-term count overflow");
                                visit(FirstMomentumGaugeCompositionEntry {
                                    parameter_component_index,
                                    momentum_vector_index,
                                    exterior_mask: output_mask,
                                    real,
                                    imaginary,
                                })?;
                            }
                        }
                    }
                }
            }
        }
        Ok((
            spec,
            parameter_basis,
            maximum,
            sha256_bytes(fixture.bytes),
            emitted_nonzero_terms,
        ))
    } else {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let intermediate = spec.intermediate_dynkin_label.as_deref().unwrap();
        let problem = first_momentum_problem(intermediate);
        let abstract_certificate = build_abstract_from_fixture(
            problem,
            first.dynkin_label,
            first.copy,
            fixture_coefficient_width(first.artifact),
            first.bytes,
        )
        .0;
        let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
        let recoupling = recouplings
            .iter()
            .find(|audit| audit.intermediate_dynkin_label == intermediate)
            .unwrap();
        let (mut model, highest, mut maximum) = build_first_momentum_correction_highest(
            problem,
            &abstract_certificate,
            fixture.bytes,
            fixture_coefficient_width(fixture.artifact),
            recoupling,
        );
        for (parameter_component_index, (_, _, matrix)) in gauge_basis.iter().enumerate() {
            if selected_parameter_components
                .is_some_and(|selected| !selected.contains(&parameter_component_index))
            {
                continue;
            }
            for (&(momentum_vector_index, free_spinor), exterior) in &highest.components {
                let source_masks = model.space(exterior.weight).masks.clone();
                for derivative_spinor in 0..32 {
                    let gauge = &matrix[free_spinor][derivative_spinor];
                    if gauge.re.is_zero() && gauge.im.is_zero() {
                        continue;
                    }
                    assert_eq!(*gauge.re.denom(), 1);
                    assert_eq!(*gauge.im.denom(), 1);
                    let gauge_real = i128::from(*gauge.re.numer());
                    let gauge_imaginary = i128::from(*gauge.im.numer());
                    for (mask, source_coefficient) in
                        source_masks.iter().copied().zip(&exterior.coefficients)
                    {
                        if *source_coefficient == 0 {
                            continue;
                        }
                        let Some(wedge_sign) = right_wedge_sign(mask, derivative_spinor) else {
                            continue;
                        };
                        let output_mask = mask | (1_u32 << derivative_spinor);
                        assert_eq!(output_mask.count_ones(), 15);
                        let scale = i128::from(*source_coefficient) * wedge_sign;
                        let real = scale
                            .checked_mul(gauge_real)
                            .expect("i128 overflow in streamed correction gauge real coefficient");
                        let imaginary = scale.checked_mul(gauge_imaginary).expect(
                            "i128 overflow in streamed correction gauge imaginary coefficient",
                        );
                        if real == 0 && imaginary == 0 {
                            continue;
                        }
                        maximum = maximum.max(real.abs()).max(imaginary.abs());
                        emitted_nonzero_terms = emitted_nonzero_terms
                            .checked_add(1)
                            .expect("first-momentum emitted-term count overflow");
                        visit(FirstMomentumGaugeCompositionEntry {
                            parameter_component_index,
                            momentum_vector_index,
                            exterior_mask: output_mask,
                            real,
                            imaginary,
                        })?;
                    }
                }
            }
        }
        Ok((
            spec,
            parameter_basis,
            maximum,
            sha256_bytes(fixture.bytes),
            emitted_nonzero_terms,
        ))
    }
}

fn selected_index(selected: Option<&[usize]>, index: usize) -> bool {
    selected.is_none_or(|indices| indices.contains(&index))
}

fn emit_target_resolved_residual<F>(
    target_basis_ordinal: usize,
    dual_target: &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
    parameter_component_index: usize,
    momentum_vector_weight_index: Option<usize>,
    exterior_mask: u32,
    residual_real: i128,
    residual_imaginary: i128,
    emitted: &mut u64,
    visit: &mut F,
) -> io::Result<()>
where
    F: FnMut(TargetResolvedGaugeCompositionEntry) -> io::Result<()>,
{
    for target in &dual_target.raw_terms {
        let scale = Ratio::new(
            BigInt::from(target.numerator),
            BigInt::from(target.denominator),
        );
        let real = scale.clone() * Ratio::from_integer(BigInt::from(residual_real));
        let imaginary = scale * Ratio::from_integer(BigInt::from(residual_imaginary));
        if real.is_zero() && imaginary.is_zero() {
            continue;
        }
        *emitted = emitted
            .checked_add(1)
            .expect("target-resolved emitted-term count overflow");
        visit(TargetResolvedGaugeCompositionEntry {
            target_basis_ordinal,
            target_vector_weight_index: target.vector_weight_index,
            target_spinor_weight_index: target.spinor_weight_index,
            parameter_component_index,
            momentum_vector_weight_index,
            exterior_mask,
            real,
            imaginary,
        })?;
    }
    Ok(())
}

/// Stream the exact `D^17 Lambda` part of `A G_p` with the full target index.
///
/// The level-16 certificate constructs the adjoint embedding of `(10001)`.
/// This visitor lowers that embedding through all requested target states and
/// contracts it with the gauge variation.  The invariant-metric dual target
/// basis converts the result back to the raw 11 by 32 target weight basis.  A
/// single overall nonzero bridge normalization remains free, which cannot
/// affect a zero/nonzero gauge-curvature test.
pub fn visit_target_resolved_zero_momentum_gauge_composition_terms<F>(
    gauge_form_degree: usize,
    leading_ordinal: usize,
    selected_parameter_components: Option<&[usize]>,
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String, u64)>
where
    F: FnMut(TargetResolvedGaugeCompositionEntry) -> io::Result<()>,
{
    assert!(gauge_form_degree <= 5);
    let spec = joint_column_specs()
        .get(leading_ordinal)
        .unwrap_or_else(|| panic!("leading column ordinal {leading_ordinal} is outside 0..12"))
        .clone();
    assert_eq!(spec.kind, "leading");
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let fixture = fixtures
        .iter()
        .find(|fixture| {
            fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
        })
        .unwrap();
    let first = fixtures
        .iter()
        .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
        .unwrap();
    let abstract_certificate = build_abstract_from_fixture(
        LEVEL16_PROBLEM,
        first.dynkin_label,
        first.copy,
        2,
        first.bytes,
    )
    .0;
    let (mut model, highest, mut maximum) =
        materialize_coupled_highest(LEVEL16_PROBLEM, &abstract_certificate, fixture.bytes, 2);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let target_basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let dual_target_basis =
        crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis()
        .into_iter()
        .filter(|(degree, _, _)| *degree == gauge_form_degree)
        .collect::<Vec<_>>();
    let parameter_basis = gauge_basis
        .iter()
        .map(|(_, indices, _)| indices.clone())
        .collect::<Vec<_>>();
    let mut emitted = 0_u64;
    for target in &target_basis {
        if !selected_index(selected_target_basis_ordinals, target.ordinal) {
            continue;
        }
        let state = if target.pbw_word_simple_roots.is_empty() {
            highest.clone()
        } else {
            coupled_state_for_word(
                &mut model,
                &highest,
                &target.pbw_word_simple_roots,
                &mut cache,
                &mut maximum,
            )
        };
        assert_eq!(state.total_weight, target.doubled_weight);
        for (parameter_component_index, (_, _, matrix)) in gauge_basis.iter().enumerate() {
            if !selected_index(selected_parameter_components, parameter_component_index) {
                continue;
            }
            let mut accumulated = HashMap::<u32, (i128, i128)>::new();
            for (&free_spinor, exterior) in &state.components {
                let source_masks = model.space(exterior.weight).masks.clone();
                for derivative_spinor in 0..32 {
                    let bilinear = &matrix[free_spinor][derivative_spinor];
                    if bilinear.re.is_zero() && bilinear.im.is_zero() {
                        continue;
                    }
                    assert_eq!(*bilinear.re.denom(), 1);
                    assert_eq!(*bilinear.im.denom(), 1);
                    for (mask, source_coefficient) in
                        source_masks.iter().copied().zip(&exterior.coefficients)
                    {
                        if *source_coefficient == 0 {
                            continue;
                        }
                        let Some(wedge_sign) = right_wedge_sign(mask, derivative_spinor) else {
                            continue;
                        };
                        let scale = i128::from(*source_coefficient) * wedge_sign;
                        let value = accumulated
                            .entry(mask | (1_u32 << derivative_spinor))
                            .or_insert((0, 0));
                        value.0 = value
                            .0
                            .checked_add(scale * i128::from(*bilinear.re.numer()))
                            .expect("i128 overflow in target-resolved zero-momentum real part");
                        value.1 = value
                            .1
                            .checked_add(scale * i128::from(*bilinear.im.numer()))
                            .expect(
                                "i128 overflow in target-resolved zero-momentum imaginary part",
                            );
                        maximum = maximum.max(value.0.abs()).max(value.1.abs());
                    }
                }
            }
            for (exterior_mask, (real, imaginary)) in accumulated {
                if real == 0 && imaginary == 0 {
                    continue;
                }
                emit_target_resolved_residual(
                    target.ordinal,
                    &dual_target_basis[target.ordinal],
                    parameter_component_index,
                    None,
                    exterior_mask,
                    real,
                    imaginary,
                    &mut emitted,
                    &mut visit,
                )?;
            }
        }
    }
    Ok((
        spec,
        parameter_basis,
        maximum,
        sha256_bytes(fixture.bytes),
        emitted,
    ))
}

/// Stream all six exact zero-momentum gauge channels for one leading source
/// column while materializing its coupled highest state only once.
///
/// The callback receives the gauge-form degree before the typed target entry.
/// This is equivalent to six calls to
/// [`visit_target_resolved_zero_momentum_gauge_composition_terms`], but avoids
/// rebuilding the degree-independent source representation six times.
pub fn visit_target_resolved_zero_momentum_gauge_composition_terms_all_degrees<F>(
    leading_ordinal: usize,
    parameter_weights_by_degree: &[BTreeMap<usize, Complex<Ratio<i64>>>; 6],
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(
    JointColumnSpec,
    [Vec<Vec<usize>>; 6],
    i128,
    String,
    [u64; 6],
)>
where
    F: FnMut(usize, TargetResolvedGaugeCompositionEntry) -> io::Result<()>,
{
    let spec = joint_column_specs()
        .get(leading_ordinal)
        .unwrap_or_else(|| panic!("leading column ordinal {leading_ordinal} is outside 0..12"))
        .clone();
    assert_eq!(spec.kind, "leading");
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let fixture = fixtures
        .iter()
        .find(|fixture| {
            fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
        })
        .unwrap();
    let first = fixtures
        .iter()
        .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
        .unwrap();
    let abstract_certificate = build_abstract_from_fixture(
        LEVEL16_PROBLEM,
        first.dynkin_label,
        first.copy,
        2,
        first.bytes,
    )
    .0;
    let (mut model, highest, mut maximum) =
        materialize_coupled_highest(LEVEL16_PROBLEM, &abstract_certificate, fixture.bytes, 2);
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let target_basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let dual_target_basis =
        crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let all_gauge_basis = crate::eleven_dimensional_clifford::gauge_form_operator_basis();
    let gauge_basis_by_degree: [Vec<_>; 6] = std::array::from_fn(|degree| {
        all_gauge_basis
            .iter()
            .filter(|(candidate_degree, _, _)| *candidate_degree == degree)
            .collect()
    });
    let parameter_basis_by_degree = std::array::from_fn(|degree| {
        gauge_basis_by_degree[degree]
            .iter()
            .map(|(_, indices, _)| indices.clone())
            .collect::<Vec<_>>()
    });
    let combined_gauge_by_degree: [(Vec<Vec<(i128, i128)>>, i128); 6] =
        std::array::from_fn(|degree| {
            let mut combined = vec![vec![Complex::new(Ratio::zero(), Ratio::zero()); 32]; 32];
            for (&component, weight) in &parameter_weights_by_degree[degree] {
                let matrix = &gauge_basis_by_degree[degree][component].2;
                for row in 0..32 {
                    for column in 0..32 {
                        combined[row][column] += matrix[row][column].clone() * weight.clone();
                    }
                }
            }
            let mut common_denominator = 1_i64;
            let gcd = |mut left: i64, mut right: i64| {
                while right != 0 {
                    let remainder = left % right;
                    left = right;
                    right = remainder;
                }
                left.abs()
            };
            for value in combined.iter().flatten() {
                for denominator in [*value.re.denom(), *value.im.denom()] {
                    common_denominator = common_denominator
                        .checked_mul(denominator / gcd(common_denominator, denominator))
                        .expect("combined gauge-matrix denominator overflow");
                }
            }
            let integer_matrix = combined
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| {
                            (
                                i128::from(*value.re.numer())
                                    * i128::from(common_denominator / *value.re.denom()),
                                i128::from(*value.im.numer())
                                    * i128::from(common_denominator / *value.im.denom()),
                            )
                        })
                        .collect()
                })
                .collect();
            (integer_matrix, i128::from(common_denominator))
        });
    let mut emitted_by_degree = [0_u64; 6];
    for target in &target_basis {
        if !selected_index(selected_target_basis_ordinals, target.ordinal) {
            continue;
        }
        let state = if target.pbw_word_simple_roots.is_empty() {
            highest.clone()
        } else {
            coupled_state_for_word(
                &mut model,
                &highest,
                &target.pbw_word_simple_roots,
                &mut cache,
                &mut maximum,
            )
        };
        assert_eq!(state.total_weight, target.doubled_weight);
        for gauge_form_degree in 0..=5 {
            let (matrix, common_denominator) = &combined_gauge_by_degree[gauge_form_degree];
            let mut accumulated = HashMap::<u32, (i128, i128)>::new();
            for (&free_spinor, exterior) in &state.components {
                let source_masks = model.space(exterior.weight).masks.clone();
                for derivative_spinor in 0..32 {
                    let bilinear = matrix[free_spinor][derivative_spinor];
                    if bilinear == (0, 0) {
                        continue;
                    }
                    for (mask, source_coefficient) in
                        source_masks.iter().copied().zip(&exterior.coefficients)
                    {
                        if *source_coefficient == 0 {
                            continue;
                        }
                        let Some(wedge_sign) = right_wedge_sign(mask, derivative_spinor) else {
                            continue;
                        };
                        let scale = i128::from(*source_coefficient) * wedge_sign;
                        let value = accumulated
                            .entry(mask | (1_u32 << derivative_spinor))
                            .or_insert((0, 0));
                        value.0 = value
                            .0
                            .checked_add(scale * bilinear.0)
                            .expect("i128 overflow in all-degree zero-momentum real part");
                        value.1 = value
                            .1
                            .checked_add(scale * bilinear.1)
                            .expect("i128 overflow in all-degree zero-momentum imaginary part");
                        maximum = maximum.max(value.0.abs()).max(value.1.abs());
                    }
                }
            }
            for (exterior_mask, (real, imaginary)) in accumulated {
                if real == 0 && imaginary == 0 {
                    continue;
                }
                emit_target_resolved_residual(
                    target.ordinal,
                    &dual_target_basis[target.ordinal],
                    0,
                    None,
                    exterior_mask,
                    real,
                    imaginary,
                    &mut emitted_by_degree[gauge_form_degree],
                    &mut |mut entry| {
                        entry.real /= BigInt::from(*common_denominator);
                        entry.imaginary /= BigInt::from(*common_denominator);
                        visit(gauge_form_degree, entry)
                    },
                )?;
            }
        }
    }
    Ok((
        spec,
        parameter_basis_by_degree,
        maximum,
        sha256_bytes(fixture.bytes),
        emitted_by_degree,
    ))
}

/// Stream the exact `p D^15 Lambda` part of `A G_p` with the full target
/// vector-spinor index.  Both leading anticommutator terms and explicit
/// first-momentum correction columns use the same target dual-basis contract.
fn visit_target_resolved_first_momentum_gauge_composition_terms_all_degrees_raw<F>(
    operator_ordinal: usize,
    selected_gauge_form_degrees: Option<&[usize]>,
    selected_parameter_components: Option<&[usize]>,
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(
    JointColumnSpec,
    [Vec<Vec<usize>>; 6],
    i128,
    String,
    [u64; 6],
    SharedFirstMomentumStateAccounting,
)>
where
    F: FnMut(
        usize,
        usize,
        &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
        usize,
        Option<usize>,
        u32,
        i128,
        i128,
        &mut u64,
    ) -> io::Result<()>,
{
    if let Some(degrees) = selected_gauge_form_degrees {
        assert!(degrees.iter().all(|degree| *degree <= 5));
    }
    let spec = joint_column_specs()
        .get(operator_ordinal)
        .unwrap_or_else(|| panic!("operator column ordinal {operator_ordinal} is outside 0..56"))
        .clone();
    let mut gauge_basis_by_degree: [Vec<_>; 6] = std::array::from_fn(|_| Vec::new());
    for entry in crate::eleven_dimensional_clifford::gauge_form_operator_basis() {
        gauge_basis_by_degree[entry.0].push(entry);
    }
    let parameter_basis_by_degree = std::array::from_fn(|degree| {
        gauge_basis_by_degree[degree]
            .iter()
            .map(|(_, indices, _)| indices.clone())
            .collect::<Vec<_>>()
    });
    let target_basis = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states();
    let dual_target_basis =
        crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states();
    let mut emitted_by_degree = [0_u64; 6];
    let selected_degrees = (0..6)
        .filter(|degree| selected_index(selected_gauge_form_degrees, *degree))
        .collect::<Vec<_>>();

    if spec.kind == "leading" {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let abstract_certificate = build_abstract_from_fixture(
            LEVEL16_PROBLEM,
            first.dynkin_label,
            first.copy,
            2,
            first.bytes,
        )
        .0;
        let (mut model, highest, mut maximum) =
            materialize_coupled_highest(LEVEL16_PROBLEM, &abstract_certificate, fixture.bytes, 2);
        let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
        let translation = translation_weight_basis_coefficients();
        for target in &target_basis {
            if !selected_index(selected_target_basis_ordinals, target.ordinal) {
                continue;
            }
            let state = if target.pbw_word_simple_roots.is_empty() {
                highest.clone()
            } else {
                coupled_state_for_word(
                    &mut model,
                    &highest,
                    &target.pbw_word_simple_roots,
                    &mut cache,
                    &mut maximum,
                )
            };
            assert_eq!(state.total_weight, target.doubled_weight);
            for &gauge_form_degree in &selected_degrees {
                for (parameter_component_index, (_, _, matrix)) in
                    gauge_basis_by_degree[gauge_form_degree].iter().enumerate()
                {
                    if !selected_index(selected_parameter_components, parameter_component_index) {
                        continue;
                    }
                    let mut accumulated = HashMap::<(usize, u32), (i128, i128)>::new();
                    for (&free_spinor, exterior) in &state.components {
                        let source_masks = model.space(exterior.weight).masks.clone();
                        for derivative_spinor in 0..32 {
                            let gauge = &matrix[free_spinor][derivative_spinor];
                            if gauge.re.is_zero() && gauge.im.is_zero() {
                                continue;
                            }
                            let gauge_real = i128::from(*gauge.re.numer());
                            let gauge_imaginary = i128::from(*gauge.im.numer());
                            for (mask, source_coefficient) in
                                source_masks.iter().copied().zip(&exterior.coefficients)
                            {
                                if *source_coefficient == 0 {
                                    continue;
                                }
                                let mut occupied = mask;
                                while occupied != 0 {
                                    let contracted = occupied.trailing_zeros() as usize;
                                    occupied &= occupied - 1;
                                    let sign = right_contraction_sign(mask, contracted).unwrap();
                                    let output_mask = mask ^ (1_u32 << contracted);
                                    let source_scale = i128::from(*source_coefficient) * sign;
                                    for momentum in 0..11 {
                                        let (translation_real, translation_imaginary) =
                                            translation[contracted][derivative_spinor][momentum];
                                        if translation_real == 0 && translation_imaginary == 0 {
                                            continue;
                                        }
                                        let (real, imaginary) = multiply_gaussian_integers(
                                            gauge_real,
                                            gauge_imaginary,
                                            i128::from(translation_real),
                                            i128::from(translation_imaginary),
                                        );
                                        accumulate_first_momentum_gauge_entry(
                                            &mut accumulated,
                                            momentum,
                                            output_mask,
                                            source_scale * real,
                                            source_scale * imaginary,
                                            &mut maximum,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    for ((momentum, exterior_mask), (real, imaginary)) in accumulated {
                        if real == 0 && imaginary == 0 {
                            continue;
                        }
                        visit(
                            gauge_form_degree,
                            target.ordinal,
                            &dual_target_basis[target.ordinal],
                            parameter_component_index,
                            Some(momentum),
                            exterior_mask,
                            real,
                            imaginary,
                            &mut emitted_by_degree[gauge_form_degree],
                        )?;
                    }
                }
            }
        }
        let estimated_payload_bytes = exterior_model_payload_bytes(&model)
            + coupled_state_payload_bytes(&highest)
            + cache.values().map(coupled_state_payload_bytes).sum::<u64>();
        let configured_payload_limit_bytes =
            enforce_shared_state_payload_limit(estimated_payload_bytes)?;
        Ok((
            spec,
            parameter_basis_by_degree,
            maximum,
            sha256_bytes(fixture.bytes),
            emitted_by_degree,
            SharedFirstMomentumStateAccounting {
                operator_ordinal,
                selected_gauge_form_degrees: selected_degrees,
                coupled_state_materializations: 1,
                estimated_payload_bytes,
                configured_payload_limit_bytes,
                payload_limit_respected: true,
            },
        ))
    } else {
        let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
        let fixture = fixtures
            .iter()
            .find(|fixture| {
                fixture.dynkin_label == spec.source_dynkin_label && fixture.copy == spec.source_copy
            })
            .unwrap();
        let first = fixtures
            .iter()
            .find(|candidate| candidate.dynkin_label == fixture.dynkin_label && candidate.copy == 1)
            .unwrap();
        let intermediate = spec.intermediate_dynkin_label.as_deref().unwrap();
        let problem = first_momentum_problem(intermediate);
        let abstract_certificate = build_abstract_from_fixture(
            problem,
            first.dynkin_label,
            first.copy,
            fixture_coefficient_width(first.artifact),
            first.bytes,
        )
        .0;
        let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
        let recoupling = recouplings
            .iter()
            .find(|audit| audit.intermediate_dynkin_label == intermediate)
            .unwrap();
        let (mut model, highest, mut maximum) = build_first_momentum_correction_highest(
            problem,
            &abstract_certificate,
            fixture.bytes,
            fixture_coefficient_width(fixture.artifact),
            recoupling,
        );
        let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
        for target in &target_basis {
            if !selected_index(selected_target_basis_ordinals, target.ordinal) {
                continue;
            }
            let state = momentum_coupled_state_for_word(
                &mut model,
                &highest,
                &target.pbw_word_simple_roots,
                &mut cache,
                &mut maximum,
            );
            assert_eq!(state.total_weight, target.doubled_weight);
            for &gauge_form_degree in &selected_degrees {
                for (parameter_component_index, (_, _, matrix)) in
                    gauge_basis_by_degree[gauge_form_degree].iter().enumerate()
                {
                    if !selected_index(selected_parameter_components, parameter_component_index) {
                        continue;
                    }
                    let mut accumulated = HashMap::<(usize, u32), (i128, i128)>::new();
                    for (&(momentum, free_spinor), exterior) in &state.components {
                        let source_masks = model.space(exterior.weight).masks.clone();
                        for derivative_spinor in 0..32 {
                            let gauge = &matrix[free_spinor][derivative_spinor];
                            if gauge.re.is_zero() && gauge.im.is_zero() {
                                continue;
                            }
                            for (mask, source_coefficient) in
                                source_masks.iter().copied().zip(&exterior.coefficients)
                            {
                                if *source_coefficient == 0 {
                                    continue;
                                }
                                let Some(sign) = right_wedge_sign(mask, derivative_spinor) else {
                                    continue;
                                };
                                accumulate_first_momentum_gauge_entry(
                                    &mut accumulated,
                                    momentum,
                                    mask | (1_u32 << derivative_spinor),
                                    i128::from(*source_coefficient)
                                        * sign
                                        * i128::from(*gauge.re.numer()),
                                    i128::from(*source_coefficient)
                                        * sign
                                        * i128::from(*gauge.im.numer()),
                                    &mut maximum,
                                );
                            }
                        }
                    }
                    for ((momentum, exterior_mask), (real, imaginary)) in accumulated {
                        if real == 0 && imaginary == 0 {
                            continue;
                        }
                        visit(
                            gauge_form_degree,
                            target.ordinal,
                            &dual_target_basis[target.ordinal],
                            parameter_component_index,
                            Some(momentum),
                            exterior_mask,
                            real,
                            imaginary,
                            &mut emitted_by_degree[gauge_form_degree],
                        )?;
                    }
                }
            }
        }
        let estimated_payload_bytes = exterior_model_payload_bytes(&model)
            + momentum_coupled_state_payload_bytes(&highest)
            + cache
                .values()
                .map(momentum_coupled_state_payload_bytes)
                .sum::<u64>();
        let configured_payload_limit_bytes =
            enforce_shared_state_payload_limit(estimated_payload_bytes)?;
        Ok((
            spec,
            parameter_basis_by_degree,
            maximum,
            sha256_bytes(fixture.bytes),
            emitted_by_degree,
            SharedFirstMomentumStateAccounting {
                operator_ordinal,
                selected_gauge_form_degrees: selected_degrees,
                coupled_state_materializations: 1,
                estimated_payload_bytes,
                configured_payload_limit_bytes,
                payload_limit_respected: true,
            },
        ))
    }
}

fn visit_target_resolved_first_momentum_gauge_composition_terms_raw<F>(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    selected_parameter_components: Option<&[usize]>,
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String, u64)>
where
    F: FnMut(
        usize,
        &crate::eleven_dimensional_bridge::VectorSpinorTargetBasisState,
        usize,
        Option<usize>,
        u32,
        i128,
        i128,
        &mut u64,
    ) -> io::Result<()>,
{
    assert!(gauge_form_degree <= 5);
    let (spec, mut parameter_basis, maximum, fixture_sha256, emitted, _) =
        visit_target_resolved_first_momentum_gauge_composition_terms_all_degrees_raw(
            operator_ordinal,
            Some(&[gauge_form_degree]),
            selected_parameter_components,
            selected_target_basis_ordinals,
            |degree,
             target_basis_ordinal,
             dual_target,
             parameter_component_index,
             momentum_vector_weight_index,
             exterior_mask,
             residual_real,
             residual_imaginary,
             emitted| {
                debug_assert_eq!(degree, gauge_form_degree);
                visit(
                    target_basis_ordinal,
                    dual_target,
                    parameter_component_index,
                    momentum_vector_weight_index,
                    exterior_mask,
                    residual_real,
                    residual_imaginary,
                    emitted,
                )
            },
        )?;
    Ok((
        spec,
        std::mem::take(&mut parameter_basis[gauge_form_degree]),
        maximum,
        fixture_sha256,
        emitted[gauge_form_degree],
    ))
}

pub fn visit_target_resolved_first_momentum_gauge_composition_terms<F>(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    selected_parameter_components: Option<&[usize]>,
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String, u64)>
where
    F: FnMut(TargetResolvedGaugeCompositionEntry) -> io::Result<()>,
{
    visit_target_resolved_first_momentum_gauge_composition_terms_raw(
        gauge_form_degree,
        operator_ordinal,
        selected_parameter_components,
        selected_target_basis_ordinals,
        |target_basis_ordinal,
         dual_target,
         parameter_component_index,
         momentum_vector_weight_index,
         exterior_mask,
         residual_real,
         residual_imaginary,
         emitted| {
            emit_target_resolved_residual(
                target_basis_ordinal,
                dual_target,
                parameter_component_index,
                momentum_vector_weight_index,
                exterior_mask,
                residual_real,
                residual_imaginary,
                emitted,
                &mut visit,
            )
        },
    )
}

pub fn visit_target_resolved_first_momentum_gauge_composition_primitive_terms<F>(
    gauge_form_degree: usize,
    operator_ordinal: usize,
    selected_parameter_components: Option<&[usize]>,
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(JointColumnSpec, Vec<Vec<usize>>, i128, String, u64)>
where
    F: FnMut(TargetResolvedPrimitiveGaugeCompositionEntry) -> io::Result<()>,
{
    visit_target_resolved_first_momentum_gauge_composition_terms_raw(
        gauge_form_degree,
        operator_ordinal,
        selected_parameter_components,
        selected_target_basis_ordinals,
        |target_basis_ordinal,
         dual_target,
         parameter_component_index,
         momentum_vector_weight_index,
         exterior_mask,
         residual_real,
         residual_imaginary,
         emitted| {
            for target in &dual_target.raw_terms {
                let mut real_numerator = residual_real
                    .checked_mul(i128::from(target.numerator))
                    .expect("primitive target-resolved real numerator exceeds i128");
                let mut imaginary_numerator = residual_imaginary
                    .checked_mul(i128::from(target.numerator))
                    .expect("primitive target-resolved imaginary numerator exceeds i128");
                let mut denominator = target.denominator;
                if denominator < 0 {
                    denominator = -denominator;
                    real_numerator = -real_numerator;
                    imaginary_numerator = -imaginary_numerator;
                }
                if real_numerator == 0 && imaginary_numerator == 0 {
                    continue;
                }
                *emitted = emitted
                    .checked_add(1)
                    .expect("primitive target-resolved emitted-term count overflow");
                visit(TargetResolvedPrimitiveGaugeCompositionEntry {
                    target_basis_ordinal,
                    target_vector_weight_index: target.vector_weight_index,
                    target_spinor_weight_index: target.spinor_weight_index,
                    parameter_component_index,
                    momentum_vector_weight_index,
                    exterior_mask,
                    real_numerator,
                    imaginary_numerator,
                    denominator,
                })?;
            }
            Ok(())
        },
    )
}

/// Operator-major variant of the primitive first-momentum stream.
///
/// The level-14 or level-16 coupled state is materialized exactly once, then
/// reused for every selected gauge degree. This is opt-in and leaves the
/// established single-degree API and its checkpoint schema unchanged.
pub fn visit_target_resolved_first_momentum_gauge_composition_primitive_terms_shared<F>(
    operator_ordinal: usize,
    selected_gauge_form_degrees: Option<&[usize]>,
    selected_parameter_components: Option<&[usize]>,
    selected_target_basis_ordinals: Option<&[usize]>,
    mut visit: F,
) -> io::Result<(
    JointColumnSpec,
    [Vec<Vec<usize>>; 6],
    i128,
    String,
    [u64; 6],
    SharedFirstMomentumStateAccounting,
)>
where
    F: FnMut(usize, TargetResolvedPrimitiveGaugeCompositionEntry) -> io::Result<()>,
{
    visit_target_resolved_first_momentum_gauge_composition_terms_all_degrees_raw(
        operator_ordinal,
        selected_gauge_form_degrees,
        selected_parameter_components,
        selected_target_basis_ordinals,
        |gauge_form_degree,
         target_basis_ordinal,
         dual_target,
         parameter_component_index,
         momentum_vector_weight_index,
         exterior_mask,
         residual_real,
         residual_imaginary,
         emitted| {
            for target in &dual_target.raw_terms {
                let mut real_numerator = residual_real
                    .checked_mul(i128::from(target.numerator))
                    .expect("primitive target-resolved real numerator exceeds i128");
                let mut imaginary_numerator = residual_imaginary
                    .checked_mul(i128::from(target.numerator))
                    .expect("primitive target-resolved imaginary numerator exceeds i128");
                let mut denominator = target.denominator;
                if denominator < 0 {
                    denominator = -denominator;
                    real_numerator = -real_numerator;
                    imaginary_numerator = -imaginary_numerator;
                }
                if real_numerator == 0 && imaginary_numerator == 0 {
                    continue;
                }
                *emitted = emitted
                    .checked_add(1)
                    .expect("primitive target-resolved emitted-term count overflow");
                visit(
                    gauge_form_degree,
                    TargetResolvedPrimitiveGaugeCompositionEntry {
                        target_basis_ordinal,
                        target_vector_weight_index: target.vector_weight_index,
                        target_spinor_weight_index: target.spinor_weight_index,
                        parameter_component_index,
                        momentum_vector_weight_index,
                        exterior_mask,
                        real_numerator,
                        imaginary_numerator,
                        denominator,
                    },
                )?;
            }
            Ok(())
        },
    )
}

fn write_hashed<W: Write>(
    writer: &mut W,
    hasher: &mut Sha256,
    byte_count: &mut u64,
    bytes: &[u8],
) -> io::Result<()> {
    writer.write_all(bytes)?;
    hasher.update(bytes);
    *byte_count = byte_count
        .checked_add(u64::try_from(bytes.len()).unwrap())
        .expect("raw artifact byte count overflow");
    Ok(())
}

fn write_json_durable<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()
}

pub fn verify_joint_column_artifact(
    directory: &Path,
    expected_spec: &JointColumnSpec,
    verify_uncompressed_stream: bool,
) -> io::Result<JointColumnArtifactManifest> {
    let manifest_path = directory.join("manifest.json");
    let manifest: JointColumnArtifactManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !manifest.passed || manifest.spec != *expected_spec {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "column manifest is not a passing artifact for the expected column",
        ));
    }
    let raw_path = directory.join(&manifest.raw_file);
    let functional_path = directory.join(&manifest.functional_file);
    if sha256_file(&raw_path)? != manifest.raw_compressed_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compressed residual hash mismatch",
        ));
    }
    if sha256_file(&functional_path)? != manifest.functional_file_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "functional file hash mismatch",
        ));
    }
    let functional: JointColumnFunctionalFile =
        serde_json::from_reader(BufReader::new(File::open(functional_path)?))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if functional.ordinal != expected_spec.ordinal
        || functional.label != expected_spec.label
        || functional.values.len() != manifest.exact_functional_values
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "functional file does not match the column manifest",
        ));
    }
    if verify_uncompressed_stream {
        let mut decoder = zstd::stream::read::Decoder::new(BufReader::new(File::open(raw_path)?))?;
        let mut hasher = Sha256::new();
        let mut count = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = decoder.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            count += u64::try_from(read).unwrap();
        }
        if count != manifest.raw_uncompressed_bytes
            || format!("{:x}", hasher.finalize()) != manifest.raw_uncompressed_sha256
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "uncompressed residual stream failed verification",
            ));
        }
    }
    Ok(manifest)
}

pub fn build_and_write_joint_column_artifact(
    ordinal: usize,
    output_root: &Path,
) -> io::Result<JointColumnArtifactManifest> {
    let spec = joint_column_specs()
        .get(ordinal)
        .unwrap_or_else(|| panic!("joint column ordinal {ordinal} is outside 0..56"))
        .clone();
    let completed_root = output_root.join("complete");
    let incomplete_root = output_root.join("incomplete");
    fs::create_dir_all(&completed_root)?;
    fs::create_dir_all(&incomplete_root)?;
    let final_directory = completed_root.join(format!("column-{ordinal:03}"));
    if final_directory.exists() {
        return verify_joint_column_artifact(&final_directory, &spec, false);
    }

    let started_unix_seconds = unix_seconds();
    let timer = Instant::now();
    let temporary_directory = incomplete_root.join(format!(
        "column-{ordinal:03}-{}-{started_unix_seconds}",
        std::process::id()
    ));
    fs::create_dir(&temporary_directory)?;
    let (actual_spec, residual, maximum, fixture_sha256) = build_joint_column(ordinal);
    assert_eq!(actual_spec, spec);
    let functional_values = momentum_hook_functionals(&residual);

    let raw_name = "residual.i128le.zst";
    let raw_path = temporary_directory.join(raw_name);
    let raw_file = File::create(&raw_path)?;
    let buffered = BufWriter::new(raw_file);
    let mut encoder = zstd::stream::write::Encoder::new(buffered, 1)?;
    let mut uncompressed_hasher = Sha256::new();
    let mut uncompressed_bytes = 0_u64;
    write_hashed(
        &mut encoder,
        &mut uncompressed_hasher,
        &mut uncompressed_bytes,
        b"AJPD15V1",
    )?;
    write_hashed(
        &mut encoder,
        &mut uncompressed_hasher,
        &mut uncompressed_bytes,
        &u32::try_from(ordinal).unwrap().to_le_bytes(),
    )?;
    write_hashed(
        &mut encoder,
        &mut uncompressed_hasher,
        &mut uncompressed_bytes,
        &u64::try_from(residual.len()).unwrap().to_le_bytes(),
    )?;
    for entry in &residual {
        write_hashed(
            &mut encoder,
            &mut uncompressed_hasher,
            &mut uncompressed_bytes,
            &[u8::try_from(entry.momentum_vector_index).unwrap()],
        )?;
        write_hashed(
            &mut encoder,
            &mut uncompressed_hasher,
            &mut uncompressed_bytes,
            &[u8::try_from(entry.free_spinor_index).unwrap()],
        )?;
        write_hashed(
            &mut encoder,
            &mut uncompressed_hasher,
            &mut uncompressed_bytes,
            &entry.exterior_mask.to_le_bytes(),
        )?;
        write_hashed(
            &mut encoder,
            &mut uncompressed_hasher,
            &mut uncompressed_bytes,
            &entry.real.to_le_bytes(),
        )?;
        write_hashed(
            &mut encoder,
            &mut uncompressed_hasher,
            &mut uncompressed_bytes,
            &entry.imaginary.to_le_bytes(),
        )?;
    }
    let mut buffered = encoder.finish()?;
    buffered.flush()?;
    buffered.get_ref().sync_all()?;
    let raw_uncompressed_sha256 = format!("{:x}", uncompressed_hasher.finalize());
    let raw_compressed_sha256 = sha256_file(&raw_path)?;
    let raw_compressed_bytes = fs::metadata(&raw_path)?.len();

    let functional_name = "functional.json";
    let functional_path = temporary_directory.join(functional_name);
    let functional_file = JointColumnFunctionalFile {
        schema_version: "adynkra-11d-joint-column-functional-v1".to_string(),
        ordinal,
        label: spec.label.clone(),
        values: functional_values.iter().map(ToString::to_string).collect(),
    };
    write_json_durable(&functional_path, &functional_file)?;
    let functional_file_sha256 = sha256_file(&functional_path)?;

    let finished_unix_seconds = unix_seconds();
    let manifest = JointColumnArtifactManifest {
        schema_version: "adynkra-11d-joint-column-artifact-v1".to_string(),
        passed: true,
        spec,
        nonzero_residual_entries: u64::try_from(residual.len()).unwrap(),
        maximum_absolute_residual_coefficient: maximum.to_string(),
        exact_functional_values: functional_values.len(),
        raw_record_bytes: 38,
        raw_uncompressed_bytes: uncompressed_bytes,
        raw_compressed_bytes,
        raw_uncompressed_sha256,
        raw_compressed_sha256,
        functional_file_sha256,
        fixture_sha256,
        source_revision: std::env::var("ADINKRA_SOURCE_REVISION")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        executable_sha256: std::env::var("ADINKRA_EXECUTABLE_SHA256")
            .unwrap_or_else(|_| "unrecorded".to_string()),
        host: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        process_id: std::process::id(),
        started_unix_seconds,
        finished_unix_seconds,
        elapsed_milliseconds: timer.elapsed().as_millis(),
        raw_file: raw_name.to_string(),
        functional_file: functional_name.to_string(),
        convention: "AJPD15V1 little-endian stream: 8-byte magic, u32 ordinal, u64 record count, then 38-byte records (u8 momentum index, u8 free-spinor index, u32 exterior mask, i128 real, i128 imaginary); zstd level 1".to_string(),
    };
    write_json_durable(&temporary_directory.join("manifest.json"), &manifest)?;
    File::open(&temporary_directory)?.sync_all()?;
    fs::rename(&temporary_directory, &final_directory)?;
    File::open(&completed_root)?.sync_all()?;
    verify_joint_column_artifact(&final_directory, &manifest.spec, false)
}

pub fn build_joint_compatibility_matrix() -> JointCompatibilityMatrixReport {
    let hook_report = build_level17_derivative_matrix();
    assert!(hook_report.passed);
    let hook_terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
    let translation_coefficients = translation_weight_basis_coefficients();
    let mut maximum = 0_i128;
    let mut functional_columns = Vec::<Vec<i128>>::new();
    let mut leading_basis = Vec::new();

    let level16_fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut level16_abstract = BTreeMap::new();
    for fixture in &level16_fixtures {
        if fixture.copy == 1 {
            level16_abstract.insert(
                fixture.dynkin_label,
                build_abstract_from_fixture(
                    LEVEL16_PROBLEM,
                    fixture.dynkin_label,
                    fixture.copy,
                    2,
                    fixture.bytes,
                )
                .0,
            );
        }
    }
    for fixture in &level16_fixtures {
        let (residual, local_maximum) = build_leading_anticommutator_residual(
            &level16_abstract[fixture.dynkin_label],
            fixture.bytes,
            &hook_terms,
            &translation_coefficients,
        );
        maximum = maximum.max(local_maximum);
        leading_basis.push(format!("{}#{}", fixture.dynkin_label, fixture.copy));
        eprintln!(
            "built leading pD15 residual {}#{} with {} nonzero Gaussian coordinates ({}/12)",
            fixture.dynkin_label,
            fixture.copy,
            residual.len(),
            leading_basis.len()
        );
        functional_columns.push(momentum_hook_functionals(&residual));
    }

    let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
    assert_eq!(recouplings.len(), 4);
    assert!(recouplings.iter().all(|audit| audit.passed));
    let level14_fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
    let fixtures_by_source =
        level14_fixtures
            .iter()
            .fold(BTreeMap::<&str, Vec<_>>::new(), |mut fixtures, fixture| {
                fixtures
                    .entry(fixture.dynkin_label)
                    .or_default()
                    .push(*fixture);
                fixtures
            });
    let manifest = first_momentum_copy_manifest();
    let mut first_momentum_basis = Vec::new();
    for ((source_dynkin_label, intermediate_dynkin_label), copies) in manifest {
        let first_fixture = fixtures_by_source[&source_dynkin_label.as_str()]
            .iter()
            .find(|fixture| fixture.copy == 1)
            .unwrap();
        let problem = first_momentum_problem(&intermediate_dynkin_label);
        let abstract_certificate = build_abstract_from_fixture(
            problem,
            &source_dynkin_label,
            1,
            fixture_coefficient_width(first_fixture.artifact),
            first_fixture.bytes,
        )
        .0;
        let recoupling = recouplings
            .iter()
            .find(|audit| audit.intermediate_dynkin_label == intermediate_dynkin_label)
            .unwrap();
        for copy in copies {
            let fixture = fixtures_by_source[&source_dynkin_label.as_str()]
                .iter()
                .find(|fixture| fixture.copy == copy)
                .unwrap();
            let (residual, local_maximum) = build_first_momentum_correction_residual(
                problem,
                &abstract_certificate,
                fixture.bytes,
                fixture_coefficient_width(fixture.artifact),
                recoupling,
                &hook_terms,
            );
            maximum = maximum.max(local_maximum);
            first_momentum_basis.push(format!(
                "{}#{}->{}",
                source_dynkin_label, copy, intermediate_dynkin_label
            ));
            eprintln!(
                "built correction pD15 residual {}#{}->{} with {} nonzero Gaussian coordinates ({}/44)",
                source_dynkin_label,
                copy,
                intermediate_dynkin_label,
                residual.len(),
                first_momentum_basis.len()
            );
            functional_columns.push(momentum_hook_functionals(&residual));
        }
    }
    assert_eq!(leading_basis, hook_report.source_basis);
    assert_eq!(leading_basis.len(), 12);
    assert_eq!(first_momentum_basis.len(), 44);
    assert_eq!(functional_columns.len(), 56);

    let coefficient_columns = functional_columns.len();
    let mut normal_matrix =
        vec![vec![Ratio::from_integer(BigInt::zero()); coefficient_columns]; coefficient_columns];
    for left in 0..coefficient_columns {
        for right in left..coefficient_columns {
            let value = Ratio::from_integer(BigInt::from(
                functional_columns[left]
                    .iter()
                    .zip(&functional_columns[right])
                    .fold(0_i128, |sum, (left_value, right_value)| {
                        sum.checked_add(
                            left_value
                                .checked_mul(*right_value)
                                .expect("i128 overflow in functional product"),
                        )
                        .expect("i128 overflow in functional dot product")
                    }),
            ));
            normal_matrix[left][right] = value.clone();
            normal_matrix[right][left] = value;
        }
        eprintln!(
            "completed exact-functional normal-matrix row {}/{}",
            left + 1,
            coefficient_columns
        );
    }
    let hook_matrix = hook_report
        .matrix_rows_by_hook_columns_by_source
        .iter()
        .map(|row| row.iter().map(parse_rational_entry).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for left in 0..leading_basis.len() {
        for right in 0..leading_basis.len() {
            let hook_dot = hook_matrix
                .iter()
                .fold(Ratio::from_integer(BigInt::zero()), |sum, row| {
                    sum + row[left].clone() * row[right].clone()
                });
            normal_matrix[left][right] += hook_dot;
        }
    }

    let exact_functional_matrix_rank = rational_matrix_rank(&normal_matrix);
    let exact_functional_nullity = coefficient_columns - exact_functional_matrix_rank;
    let full_rank_certified_by_functional_minor =
        exact_functional_matrix_rank == coefficient_columns;
    let kernel = rational_nullspace(&normal_matrix);
    assert_eq!(kernel.len(), exact_functional_nullity);
    let primitive_integer_kernel_basis = kernel
        .iter()
        .map(|vector| primitive_bigint_vector(vector))
        .collect::<Vec<_>>();
    let functional_kernel_residuals_exactly_zero = primitive_integer_kernel_basis
        .iter()
        .all(|vector| matrix_times_integer_vector_is_zero(&normal_matrix, vector));
    let leading_projections = kernel
        .iter()
        .map(|vector| vector[..leading_basis.len()].to_vec())
        .filter(|projection| projection.iter().any(|value| !value.is_zero()))
        .collect::<Vec<_>>();
    let functional_kernel_leading_projection_rank = rational_matrix_rank(&leading_projections);

    let previous_hook_kernel = hook_report
        .primitive_integer_kernel_basis
        .iter()
        .map(|vector| {
            vector
                .iter()
                .map(|value| {
                    Ratio::from_integer(BigInt::parse_bytes(value.as_bytes(), 10).unwrap())
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let previous_hook_kernel_dimension = rational_matrix_rank(&previous_hook_kernel);
    assert_eq!(previous_hook_kernel_dimension, 5);
    let scalar_factorizing = hook_report
        .scalar_factorizing_coordinates
        .iter()
        .map(parse_rational_entry)
        .collect::<Vec<_>>();
    let exact_joint_nullity = full_rank_certified_by_functional_minor.then_some(0);
    let leading_extension_excluded = functional_kernel_leading_projection_rank == 0;
    let previous_hook_kernel_subspace_dimension_extended = leading_extension_excluded.then_some(0);
    let scalar_in_functional_kernel_projection =
        rational_row_span_contains(&leading_projections, &scalar_factorizing);
    let scalar_factorizing_direction_extends =
        (!scalar_in_functional_kernel_projection).then_some(false);
    let direct_spinor_quotient_dimension_extended = leading_extension_excluded.then_some(0);
    let maximum_absolute_normal_matrix_numerator = normal_matrix
        .iter()
        .flatten()
        .map(|value| value.numer().abs())
        .max()
        .unwrap_or_else(BigInt::zero);
    let momentum_coordinate_rows =
        usize::try_from(2_u128 * 11 * 32 * binomial_u128(32, 15)).unwrap();
    let passed = coefficient_columns == 56
        && previous_hook_kernel_dimension == 5
        && leading_extension_excluded
        && functional_kernel_residuals_exactly_zero;
    JointCompatibilityMatrixReport {
        schema_version: "adynkra-11d-joint-leading-first-momentum-v2".to_string(),
        role: "exact joint zero-momentum hook and first-momentum pD15 compatibility certificate"
            .to_string(),
        leading_basis,
        first_momentum_basis,
        hook_rows: hook_matrix.len(),
        momentum_coordinate_rows,
        coefficient_columns,
        leading_columns: 12,
        first_momentum_columns: 44,
        reciprocal_couplings_verified: recouplings.iter().filter(|audit| audit.passed).count(),
        reciprocal_coupling_intermediates: recouplings
            .iter()
            .map(|audit| audit.intermediate_dynkin_label.clone())
            .collect(),
        reciprocal_coupling_domain_dimensions: recouplings
            .iter()
            .map(|audit| audit.highest_weight_domain_dimension)
            .collect(),
        reciprocal_coupling_kernel_dimensions: recouplings
            .iter()
            .map(|audit| audit.highest_weight_kernel_dimension)
            .collect(),
        reciprocal_coupling_raising_residuals: recouplings
            .iter()
            .map(|audit| audit.raising_residual_terms)
            .collect(),
        exact_functional_rows: JOINT_FUNCTIONAL_SEEDS.len()
            * 2
            * JOINT_FUNCTIONAL_BUCKETS
            + hook_matrix.len(),
        exact_functional_matrix_rank,
        exact_functional_nullity,
        full_rank_certified_by_functional_minor,
        exact_joint_nullity,
        functional_kernel_leading_projection_rank,
        leading_extension_excluded,
        previous_hook_kernel_dimension,
        previous_hook_kernel_subspace_dimension_extended,
        scalar_factorizing_direction_extends,
        direct_spinor_quotient_dimension_extended,
        functional_primitive_integer_kernel_basis: primitive_integer_kernel_basis
            .iter()
            .map(|vector| vector.iter().map(ToString::to_string).collect())
            .collect(),
        functional_kernel_residuals_exactly_zero,
        exact_functional_normal_matrix: normal_matrix
            .iter()
            .map(|row| row.iter().map(rational_entry).collect())
            .collect(),
        maximum_absolute_residual_coefficient: maximum,
        maximum_absolute_normal_matrix_numerator: maximum_absolute_normal_matrix_numerator
            .to_string(),
        convention: "canonical sorted spinor-mask basis; complex B5 vector-weight basis split into exact real and imaginary coordinates; {D,D}=2 Gamma.p; primitive integer source, target, hook, and reciprocal couplings".to_string(),
        interpretation: if full_rank_certified_by_functional_minor {
            "the exact functional image has full column rank 56; because it is obtained by exact linear functionals of the full coordinate residual, the full joint system also has rank 56 and nullity zero, so none of the previous five leading hook-kernel dimensions extends".to_string()
        } else if leading_extension_excluded {
            format!(
                "the exact functional image has rank {exact_functional_matrix_rank} and nullity {exact_functional_nullity}, and its kernel has zero projection onto the twelve leading coefficients; every full-coordinate null vector lies in this functional kernel, so no nonzero leading hook-kernel vector extends, while the exact full-coordinate nullity remains between zero and {exact_functional_nullity}"
            )
        } else {
            format!(
                "the exact functional image has rank {exact_functional_matrix_rank} and nullity {exact_functional_nullity}; this is a rigorous lower bound on the full coordinate rank but does not certify the full kernel"
            )
        },
        boundary: "this is the direct spinor-prepotential compatibility calculation through first momentum order; it does not impose the six possible gauge-parameter maps, construct a curvature, supply an action, or derive a field equation".to_string(),
        passed,
    }
}

fn finalize_joint_functional_columns(
    leading_basis: Vec<String>,
    first_momentum_basis: Vec<String>,
    functional_columns: Vec<Vec<i128>>,
    maximum: i128,
) -> JointCompatibilityMatrixReport {
    assert_eq!(leading_basis.len(), 12);
    assert_eq!(first_momentum_basis.len(), 44);
    assert_eq!(functional_columns.len(), 56);
    assert!(
        functional_columns
            .iter()
            .all(|column| column.len()
                == JOINT_FUNCTIONAL_SEEDS.len() * 2 * JOINT_FUNCTIONAL_BUCKETS)
    );

    let coefficient_columns = functional_columns.len();
    let mut normal_matrix =
        vec![vec![Ratio::from_integer(BigInt::zero()); coefficient_columns]; coefficient_columns];
    for left in 0..coefficient_columns {
        for right in left..coefficient_columns {
            let value = Ratio::from_integer(BigInt::from(
                functional_columns[left]
                    .iter()
                    .zip(&functional_columns[right])
                    .fold(0_i128, |sum, (left_value, right_value)| {
                        sum.checked_add(
                            left_value
                                .checked_mul(*right_value)
                                .expect("i128 overflow in functional product"),
                        )
                        .expect("i128 overflow in functional dot product")
                    }),
            ));
            normal_matrix[left][right] = value.clone();
            normal_matrix[right][left] = value;
        }
    }

    let hook_report = build_level17_derivative_matrix();
    assert!(hook_report.passed);
    assert_eq!(leading_basis, hook_report.source_basis);
    let hook_matrix = hook_report
        .matrix_rows_by_hook_columns_by_source
        .iter()
        .map(|row| row.iter().map(parse_rational_entry).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for left in 0..leading_basis.len() {
        for right in 0..leading_basis.len() {
            let hook_dot = hook_matrix
                .iter()
                .fold(Ratio::from_integer(BigInt::zero()), |sum, row| {
                    sum + row[left].clone() * row[right].clone()
                });
            normal_matrix[left][right] += hook_dot;
        }
    }

    let exact_functional_matrix_rank = rational_matrix_rank(&normal_matrix);
    let exact_functional_nullity = coefficient_columns - exact_functional_matrix_rank;
    let full_rank_certified_by_functional_minor =
        exact_functional_matrix_rank == coefficient_columns;
    let kernel = rational_nullspace(&normal_matrix);
    assert_eq!(kernel.len(), exact_functional_nullity);
    let primitive_integer_kernel_basis = kernel
        .iter()
        .map(|vector| primitive_bigint_vector(vector))
        .collect::<Vec<_>>();
    let functional_kernel_residuals_exactly_zero = primitive_integer_kernel_basis
        .iter()
        .all(|vector| matrix_times_integer_vector_is_zero(&normal_matrix, vector));
    let leading_projections = kernel
        .iter()
        .map(|vector| vector[..leading_basis.len()].to_vec())
        .filter(|projection| projection.iter().any(|value| !value.is_zero()))
        .collect::<Vec<_>>();
    let functional_kernel_leading_projection_rank = rational_matrix_rank(&leading_projections);

    let previous_hook_kernel = hook_report
        .primitive_integer_kernel_basis
        .iter()
        .map(|vector| {
            vector
                .iter()
                .map(|value| {
                    Ratio::from_integer(BigInt::parse_bytes(value.as_bytes(), 10).unwrap())
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let previous_hook_kernel_dimension = rational_matrix_rank(&previous_hook_kernel);
    assert_eq!(previous_hook_kernel_dimension, 5);
    let scalar_factorizing = hook_report
        .scalar_factorizing_coordinates
        .iter()
        .map(parse_rational_entry)
        .collect::<Vec<_>>();
    let exact_joint_nullity = full_rank_certified_by_functional_minor.then_some(0);
    let leading_extension_excluded = functional_kernel_leading_projection_rank == 0;
    let previous_hook_kernel_subspace_dimension_extended = leading_extension_excluded.then_some(0);
    let scalar_in_functional_kernel_projection =
        rational_row_span_contains(&leading_projections, &scalar_factorizing);
    let scalar_factorizing_direction_extends =
        (!scalar_in_functional_kernel_projection).then_some(false);
    let direct_spinor_quotient_dimension_extended = leading_extension_excluded.then_some(0);
    let maximum_absolute_normal_matrix_numerator = normal_matrix
        .iter()
        .flatten()
        .map(|value| value.numer().abs())
        .max()
        .unwrap_or_else(BigInt::zero);
    let momentum_coordinate_rows =
        usize::try_from(2_u128 * 11 * 32 * binomial_u128(32, 15)).unwrap();
    let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
    assert_eq!(recouplings.len(), 4);
    assert!(recouplings.iter().all(|audit| audit.passed));
    let passed = coefficient_columns == 56
        && previous_hook_kernel_dimension == 5
        && leading_extension_excluded
        && functional_kernel_residuals_exactly_zero;
    JointCompatibilityMatrixReport {
        schema_version: "adynkra-11d-joint-leading-first-momentum-v2".to_string(),
        role: "exact joint zero-momentum hook and first-momentum pD15 compatibility certificate"
            .to_string(),
        leading_basis,
        first_momentum_basis,
        hook_rows: hook_matrix.len(),
        momentum_coordinate_rows,
        coefficient_columns,
        leading_columns: 12,
        first_momentum_columns: 44,
        reciprocal_couplings_verified: recouplings.iter().filter(|audit| audit.passed).count(),
        reciprocal_coupling_intermediates: recouplings
            .iter()
            .map(|audit| audit.intermediate_dynkin_label.clone())
            .collect(),
        reciprocal_coupling_domain_dimensions: recouplings
            .iter()
            .map(|audit| audit.highest_weight_domain_dimension)
            .collect(),
        reciprocal_coupling_kernel_dimensions: recouplings
            .iter()
            .map(|audit| audit.highest_weight_kernel_dimension)
            .collect(),
        reciprocal_coupling_raising_residuals: recouplings
            .iter()
            .map(|audit| audit.raising_residual_terms)
            .collect(),
        exact_functional_rows: JOINT_FUNCTIONAL_SEEDS.len()
            * 2
            * JOINT_FUNCTIONAL_BUCKETS
            + hook_matrix.len(),
        exact_functional_matrix_rank,
        exact_functional_nullity,
        full_rank_certified_by_functional_minor,
        exact_joint_nullity,
        functional_kernel_leading_projection_rank,
        leading_extension_excluded,
        previous_hook_kernel_dimension,
        previous_hook_kernel_subspace_dimension_extended,
        scalar_factorizing_direction_extends,
        direct_spinor_quotient_dimension_extended,
        functional_primitive_integer_kernel_basis: primitive_integer_kernel_basis
            .iter()
            .map(|vector| vector.iter().map(ToString::to_string).collect())
            .collect(),
        functional_kernel_residuals_exactly_zero,
        exact_functional_normal_matrix: normal_matrix
            .iter()
            .map(|row| row.iter().map(rational_entry).collect())
            .collect(),
        maximum_absolute_residual_coefficient: maximum,
        maximum_absolute_normal_matrix_numerator: maximum_absolute_normal_matrix_numerator
            .to_string(),
        convention: "canonical sorted spinor-mask basis; complex B5 vector-weight basis split into exact real and imaginary coordinates; {D,D}=2 Gamma.p; primitive integer source, target, hook, and reciprocal couplings".to_string(),
        interpretation: if full_rank_certified_by_functional_minor {
            "the exact functional image has full column rank 56; because it is obtained by exact linear functionals of the full coordinate residual, the full joint system also has rank 56 and nullity zero, so none of the previous five leading hook-kernel dimensions extends".to_string()
        } else if leading_extension_excluded {
            format!(
                "the exact functional image has rank {exact_functional_matrix_rank} and nullity {exact_functional_nullity}, and its kernel has zero projection onto the twelve leading coefficients; every full-coordinate null vector lies in this functional kernel, so no nonzero leading hook-kernel vector extends, while the exact full-coordinate nullity remains between zero and {exact_functional_nullity}"
            )
        } else {
            format!(
                "the exact functional image has rank {exact_functional_matrix_rank} and nullity {exact_functional_nullity}; this is a rigorous lower bound on the full coordinate rank but does not certify the full kernel"
            )
        },
        boundary: "this is the direct spinor-prepotential compatibility calculation through first momentum order; it does not impose the six possible gauge-parameter maps, construct a curvature, supply an action, or derive a field equation".to_string(),
        passed,
    }
}

pub fn merge_joint_column_artifacts(
    output_root: &Path,
    verify_uncompressed_streams: bool,
) -> io::Result<JointCompatibilityMatrixReport> {
    let specs = joint_column_specs();
    let mut leading_basis = Vec::new();
    let mut first_momentum_basis = Vec::new();
    let mut functional_columns = Vec::new();
    let mut maximum = 0_i128;
    for spec in &specs {
        let directory = output_root
            .join("complete")
            .join(format!("column-{:03}", spec.ordinal));
        let manifest = verify_joint_column_artifact(&directory, spec, verify_uncompressed_streams)?;
        let functional: JointColumnFunctionalFile = serde_json::from_reader(BufReader::new(
            File::open(directory.join(&manifest.functional_file))?,
        ))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        functional_columns.push(
            functional
                .values
                .iter()
                .map(|value| {
                    value.parse::<i128>().map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid i128 functional value: {error}"),
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()?,
        );
        maximum = maximum.max(
            manifest
                .maximum_absolute_residual_coefficient
                .parse::<i128>()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        );
        if spec.kind == "leading" {
            leading_basis.push(spec.label.clone());
        } else {
            first_momentum_basis.push(spec.label.clone());
        }
    }
    Ok(finalize_joint_functional_columns(
        leading_basis,
        first_momentum_basis,
        functional_columns,
        maximum,
    ))
}

pub fn build_level17_derivative_matrix() -> Level17DerivativeMatrixReport {
    let target_terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
    assert_eq!(target_terms.len(), 8);
    eprintln!("constructed the eight-term target coupling into (11000)");

    let hook_fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures();
    let mut hook_abstract = BTreeMap::new();
    for fixture in &hook_fixtures {
        if fixture.copy == 1 {
            hook_abstract.insert(
                fixture.dynkin_label,
                build_abstract_from_fixture(
                    LEVEL17_HOOK_PROBLEM,
                    fixture.dynkin_label,
                    fixture.copy,
                    2,
                    fixture.bytes,
                )
                .0,
            );
            eprintln!(
                "constructed abstract hook coupling {}",
                fixture.dynkin_label
            );
        }
    }
    let mut maximum = 0_i128;
    let mut hook_basis = Vec::new();
    let mut hook_labels = Vec::new();
    for fixture in &hook_fixtures {
        let certificate = &hook_abstract[fixture.dynkin_label];
        let (mut model, dense, local_maximum) = materialize_coupled_highest(
            LEVEL17_HOOK_PROBLEM,
            certificate,
            fixture.bytes,
            fixture_coefficient_width(fixture.artifact),
        );
        maximum = maximum.max(local_maximum);
        hook_basis.push(dense_coupled_to_sparse(&mut model, &dense));
        hook_labels.push(format!("{}#{}", fixture.dynkin_label, fixture.copy));
        eprintln!(
            "materialized hook basis vector {}#{} ({}/7)",
            fixture.dynkin_label,
            fixture.copy,
            hook_basis.len()
        );
    }
    let hook_gram_i128 = hook_basis
        .iter()
        .map(|left| {
            hook_basis
                .iter()
                .map(|right| sparse_coupled_dot(left, right))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let hook_gram_rank = rational_rank_i128(&hook_gram_i128);
    assert_eq!(hook_gram_rank, hook_basis.len());
    eprintln!("hook Gram matrix has exact rank {hook_gram_rank}");
    let hook_gram = hook_gram_i128
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| BigInt::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let level16_fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut level16_abstract = BTreeMap::new();
    for fixture in &level16_fixtures {
        if fixture.copy == 1 {
            level16_abstract.insert(
                fixture.dynkin_label,
                build_abstract_from_fixture(
                    LEVEL16_PROBLEM,
                    fixture.dynkin_label,
                    fixture.copy,
                    2,
                    fixture.bytes,
                )
                .0,
            );
            eprintln!(
                "constructed abstract level-16 coupling {}",
                fixture.dynkin_label
            );
        }
    }
    let mut source_labels = Vec::new();
    let mut columns = Vec::<Vec<Ratio<BigInt>>>::new();
    let mut residual_norms = Vec::new();
    let mut leading_basis = Vec::new();
    let (scalar_factorizing_candidate, scalar_maximum) = build_scalar_factorizing_candidate();
    maximum = maximum.max(scalar_maximum);
    eprintln!("constructed the scalar-factorizing leading candidate");
    for fixture in &level16_fixtures {
        let (candidate, leading, local_maximum) = build_derivative_candidate(
            &level16_abstract[fixture.dynkin_label],
            fixture.bytes,
            &target_terms,
        );
        maximum = maximum.max(local_maximum);
        leading_basis.push(leading);
        let overlaps = hook_basis
            .iter()
            .map(|hook| BigInt::from(sparse_coupled_dot(hook, &candidate)))
            .collect::<Vec<_>>();
        let coordinates = solve_bigint_system(&hook_gram, &overlaps)
            .expect("hook Gram matrix must be invertible");
        let candidate_norm =
            Ratio::from_integer(BigInt::from(sparse_coupled_dot(&candidate, &candidate)));
        let projected_norm = coordinates.iter().zip(&overlaps).fold(
            Ratio::from_integer(BigInt::zero()),
            |sum, (coordinate, overlap)| {
                sum + coordinate.clone() * Ratio::from_integer(overlap.clone())
            },
        );
        residual_norms.push(candidate_norm - projected_norm);
        columns.push(coordinates);
        source_labels.push(format!("{}#{}", fixture.dynkin_label, fixture.copy));
        eprintln!(
            "projected derivative column {}#{} ({}/12)",
            fixture.dynkin_label,
            fixture.copy,
            columns.len()
        );
    }
    let matrix = (0..hook_basis.len())
        .map(|row| {
            columns
                .iter()
                .map(|column| column[row].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let derivative_matrix_rank = rational_matrix_rank(&matrix);
    let derivative_matrix_nullity = source_labels.len() - derivative_matrix_rank;
    let primitive_integer_kernel_basis = rational_nullspace(&matrix)
        .iter()
        .map(|vector| primitive_bigint_vector(vector))
        .collect::<Vec<_>>();
    let kernel_residuals_exactly_zero = primitive_integer_kernel_basis
        .iter()
        .all(|vector| matrix_times_integer_vector_is_zero(&matrix, vector));
    let mut mutated_kernel_vector = primitive_integer_kernel_basis[0].clone();
    let mutated_index = mutated_kernel_vector
        .iter()
        .position(|value| !value.is_zero())
        .unwrap();
    mutated_kernel_vector[mutated_index] += BigInt::one();
    let kernel_coefficient_mutation_detected =
        !matrix_times_integer_vector_is_zero(&matrix, &mutated_kernel_vector);
    let leading_gram_i128 = leading_basis
        .iter()
        .map(|left| {
            leading_basis
                .iter()
                .map(|right| sparse_coupled_dot(left, right))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let leading_gram_rank = rational_rank_i128(&leading_gram_i128);
    let leading_gram = leading_gram_i128
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| BigInt::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scalar_overlaps = leading_basis
        .iter()
        .map(|leading| BigInt::from(sparse_coupled_dot(leading, &scalar_factorizing_candidate)))
        .collect::<Vec<_>>();
    let scalar_factorizing_coordinates = solve_bigint_system(&leading_gram, &scalar_overlaps)
        .expect("the twelve leading vectors must be independent");
    let scalar_candidate_norm = Ratio::from_integer(BigInt::from(sparse_coupled_dot(
        &scalar_factorizing_candidate,
        &scalar_factorizing_candidate,
    )));
    let scalar_projected_norm = scalar_factorizing_coordinates
        .iter()
        .zip(&scalar_overlaps)
        .fold(
            Ratio::from_integer(BigInt::zero()),
            |sum, (coordinate, overlap)| {
                sum + coordinate.clone() * Ratio::from_integer(overlap.clone())
            },
        );
    let scalar_factorizing_reconstruction_residual_norm =
        scalar_candidate_norm - scalar_projected_norm;
    let scalar_factorizing_direction_is_in_leading_span =
        scalar_factorizing_reconstruction_residual_norm.is_zero();
    let scalar_factorizing_hook_image = matrix
        .iter()
        .map(|row| {
            row.iter().zip(&scalar_factorizing_coordinates).fold(
                Ratio::from_integer(BigInt::zero()),
                |sum, (entry, coordinate)| sum + entry.clone() * coordinate.clone(),
            )
        })
        .collect::<Vec<_>>();
    let scalar_factorizing_hook_image_is_zero =
        scalar_factorizing_hook_image.iter().all(Ratio::is_zero);
    let every_derivative_column_is_in_hook_span = residual_norms.iter().all(Ratio::is_zero);
    let passed = hook_gram_rank == 7
        && source_labels.len() == 12
        && every_derivative_column_is_in_hook_span
        && primitive_integer_kernel_basis.len() == derivative_matrix_nullity
        && kernel_residuals_exactly_zero
        && kernel_coefficient_mutation_detected
        && leading_gram_rank == 12
        && scalar_factorizing_direction_is_in_leading_span
        && scalar_factorizing_hook_image_is_zero;
    Level17DerivativeMatrixReport {
        schema_version: "adynkra-11d-level17-derivative-matrix-v1".to_string(),
        role: "exact exterior-derivative map from twelve level-16 vector-spinor couplings to seven level-17 hook couplings".to_string(),
        source_basis: source_labels,
        hook_basis: hook_labels,
        target_hook_dynkin_label: "11000".to_string(),
        target_coupling_terms: target_terms.len(),
        target_coupling_primitive_coefficients: target_terms
            .iter()
            .map(|term| term.primitive_coefficient)
            .collect(),
        hook_gram_rank,
        derivative_matrix_rank,
        derivative_matrix_nullity,
        matrix_rows_by_hook_columns_by_source: matrix
            .iter()
            .map(|row| row.iter().map(rational_entry).collect())
            .collect(),
        primitive_integer_kernel_basis: primitive_integer_kernel_basis
            .iter()
            .map(|vector| vector.iter().map(ToString::to_string).collect())
            .collect(),
        kernel_residuals_exactly_zero,
        kernel_coefficient_mutation_detected,
        leading_gram_rank,
        scalar_factorizing_coordinates: scalar_factorizing_coordinates
            .iter()
            .map(rational_entry)
            .collect(),
        scalar_factorizing_reconstruction_residual_norm: rational_entry(
            &scalar_factorizing_reconstruction_residual_norm,
        ),
        scalar_factorizing_direction_is_in_leading_span,
        scalar_factorizing_hook_image: scalar_factorizing_hook_image
            .iter()
            .map(rational_entry)
            .collect(),
        scalar_factorizing_hook_image_is_zero,
        exact_reconstruction_residual_norms: residual_norms.iter().map(rational_entry).collect(),
        every_derivative_column_is_in_hook_span,
        maximum_absolute_checked_accumulator: maximum,
        convention: "canonical sorted spinor-mask exterior basis; left exterior multiplication by the seventeenth derivative; primitive integer source and target couplings".to_string(),
        interpretation: format!(
            "the exact map has rank {derivative_matrix_rank} and a {derivative_matrix_nullity}-dimensional kernel in the twelve-dimensional leading-map coefficient space"
        ),
        boundary: "this is the zero-spacetime-momentum exterior symbol in the direct spinor-prepotential representation complex; it does not select physical coefficients, define the gauge quotient, include momentum corrections, or derive a field equation".to_string(),
        passed,
    }
}

fn verify_embedded_with_abstract_and_hash(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    fixture_bytes: &[u8],
) -> (EmbeddedCouplingCertificate, String) {
    let mut model = ExteriorModel::new(problem.exterior_degree);
    let highest = model.fixture_state(
        &abstract_certificate.source_dynkin_label,
        fixture_bytes,
        fixture_coefficient_width(fixture_artifact),
    );
    let mut cache = BTreeMap::from([(Vec::new(), highest.clone())]);
    let mut domain = Vec::new();
    for entry in &abstract_certificate.domain_basis {
        let state = state_for_word(
            &mut model,
            &highest,
            &entry.pbw_word_simple_roots,
            &mut cache,
        );
        assert_eq!(state.weight, entry.source_weight);
        domain.push((entry.free_spinor_index, state));
    }
    let outputs_by_root = (0..5)
        .map(|root| {
            domain
                .iter()
                .map(|(spinor_index, state)| tensor_output(&mut model, state, *spinor_index, root))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (residuals, residual_maximum) = exact_residual_counts(
        &outputs_by_root,
        &abstract_certificate.primitive_domain_coefficients,
    );
    let mut coupled = BTreeMap::<usize, Vec<i128>>::new();
    for ((spinor_index, state), coefficient) in domain
        .iter()
        .zip(&abstract_certificate.primitive_domain_coefficients)
    {
        let destination = coupled
            .entry(*spinor_index)
            .or_insert_with(|| vec![0; state.coefficients.len()]);
        for (slot, value) in destination.iter_mut().zip(&state.coefficients) {
            *slot = slot
                .checked_add(i128::from(*coefficient) * i128::from(*value))
                .expect("i128 overflow while applying abstract coupling");
        }
    }
    let coupled_nonzero_terms = coupled
        .values()
        .map(|values| values.iter().filter(|value| **value != 0).count())
        .sum();
    let mut coupled_hasher = Sha256::new();
    coupled_hasher.update(b"adynkra-11d-canonical-embedded-map-v1\0");
    for (spinor_index, values) in &coupled {
        for (exterior_ordinal, value) in values.iter().enumerate() {
            if *value != 0 {
                coupled_hasher.update((*spinor_index as u64).to_le_bytes());
                coupled_hasher.update((exterior_ordinal as u64).to_le_bytes());
                coupled_hasher.update(value.to_le_bytes());
            }
        }
    }
    let coupled_map_sha256 = format!("{:x}", coupled_hasher.finalize());
    let passed = abstract_certificate.passed
        && domain.len() == abstract_certificate.product_weight_domain_dimension
        && residuals == [0; 5];
    let certificate = EmbeddedCouplingCertificate {
        schema_version: format!("{}-embedded-coupling-v1", problem.schema_prefix),
        role: "exact application of the shared abstract coupling to one exterior embedding"
            .to_string(),
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        target_dynkin_label: problem.target_dynkin_label.to_string(),
        abstract_coupling_source_copy: abstract_certificate.source_fixture_copy,
        product_weight_domain_dimension: domain.len(),
        primitive_domain_coefficients: abstract_certificate.primitive_domain_coefficients.clone(),
        coupled_nonzero_terms,
        exact_raising_residual_terms_by_simple_root: residuals,
        maximum_absolute_checked_accumulator: model
            .maximum_absolute_accumulator
            .max(residual_maximum),
        shared_abstract_coupling_applied: true,
        passed,
    };
    (certificate, coupled_map_sha256)
}

fn verify_embedded_with_abstract(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    fixture_bytes: &[u8],
) -> EmbeddedCouplingCertificate {
    verify_embedded_with_abstract_and_hash(
        problem,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        fixture_bytes,
    )
    .0
}

pub fn build_abstract(dynkin_label: &str) -> AbstractCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-16 source irrep {dynkin_label}"));
    build_abstract_from_fixture(
        LEVEL16_PROBLEM,
        dynkin_label,
        fixture.copy,
        2,
        fixture.bytes,
    )
    .0
}

pub fn verify_copy(dynkin_label: &str, copy: usize) -> EmbeddedCouplingCertificate {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let abstract_fixture = fixtures
        .iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-16 source irrep {dynkin_label}"));
    let fixture = fixtures
        .iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == copy)
        .unwrap_or_else(|| panic!("unknown copy {copy} for level-16 source irrep {dynkin_label}"));
    let (abstract_certificate, _) = build_abstract_from_fixture(
        LEVEL16_PROBLEM,
        dynkin_label,
        abstract_fixture.copy,
        2,
        abstract_fixture.bytes,
    );
    verify_embedded_with_abstract(
        LEVEL16_PROBLEM,
        &abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn verify_copy_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    copy: usize,
) -> EmbeddedCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
        .into_iter()
        .find(|fixture| {
            fixture.dynkin_label == abstract_certificate.source_dynkin_label && fixture.copy == copy
        })
        .unwrap_or_else(|| {
            panic!(
                "unknown copy {copy} for level-16 source irrep {}",
                abstract_certificate.source_dynkin_label
            )
        });
    verify_embedded_with_abstract(
        LEVEL16_PROBLEM,
        abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn build_hook_abstract(dynkin_label: &str) -> AbstractCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-17 hook source irrep {dynkin_label}"));
    build_abstract_from_fixture(
        LEVEL17_HOOK_PROBLEM,
        dynkin_label,
        fixture.copy,
        2,
        fixture.bytes,
    )
    .0
}

pub fn verify_hook_copy_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    copy: usize,
) -> EmbeddedCouplingCertificate {
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures()
        .into_iter()
        .find(|fixture| {
            fixture.dynkin_label == abstract_certificate.source_dynkin_label && fixture.copy == copy
        })
        .unwrap_or_else(|| {
            panic!(
                "unknown copy {copy} for level-17 hook source irrep {}",
                abstract_certificate.source_dynkin_label
            )
        });
    verify_embedded_with_abstract(
        LEVEL17_HOOK_PROBLEM,
        abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

pub fn hook_copy_manifest() -> BTreeMap<&'static str, Vec<usize>> {
    let mut copies = BTreeMap::<&str, Vec<usize>>::new();
    for fixture in crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures() {
        copies
            .entry(fixture.dynkin_label)
            .or_default()
            .push(fixture.copy);
    }
    copies
}

pub fn first_momentum_copy_manifest() -> BTreeMap<(String, String), Vec<usize>> {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures();
    let copies_by_source = fixtures.into_iter().fold(
        BTreeMap::<&str, Vec<usize>>::new(),
        |mut copies, fixture| {
            copies
                .entry(fixture.dynkin_label)
                .or_default()
                .push(fixture.copy);
            copies
        },
    );
    crate::eleven_dimensional_spinor_bridge::verify_first_momentum_source_precheck()
        .channels
        .into_iter()
        .map(|channel| {
            let copies = copies_by_source
                .get(channel.source_dynkin_label.as_str())
                .unwrap()
                .clone();
            (
                (
                    channel.source_dynkin_label,
                    channel.intermediate_dynkin_label,
                ),
                copies,
            )
        })
        .collect()
}

pub fn build_first_momentum_abstract(
    source_dynkin_label: &str,
    target_dynkin_label: &str,
) -> AbstractCouplingCertificate {
    let problem = first_momentum_problem(target_dynkin_label);
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
        .into_iter()
        .find(|fixture| fixture.dynkin_label == source_dynkin_label && fixture.copy == 1)
        .unwrap_or_else(|| panic!("unknown level-14 source irrep {source_dynkin_label}"));
    let target_multiplicity =
        crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
            .iter()
            .filter(|(target, _)| target == target_dynkin_label)
            .count();
    assert_eq!(
        target_multiplicity, 1,
        "the requested level-14 source-target coupling is not multiplicity one"
    );
    build_abstract_from_fixture(
        problem,
        source_dynkin_label,
        fixture.copy,
        fixture_coefficient_width(fixture.artifact),
        fixture.bytes,
    )
    .0
}

pub fn verify_first_momentum_copy_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    copy: usize,
) -> EmbeddedCouplingCertificate {
    let problem = first_momentum_problem(&abstract_certificate.target_dynkin_label);
    let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
        .into_iter()
        .find(|fixture| {
            fixture.dynkin_label == abstract_certificate.source_dynkin_label && fixture.copy == copy
        })
        .unwrap_or_else(|| {
            panic!(
                "unknown copy {copy} for level-14 source irrep {}",
                abstract_certificate.source_dynkin_label
            )
        });
    verify_embedded_with_abstract(
        problem,
        abstract_certificate,
        fixture.copy,
        fixture.artifact,
        fixture.bytes,
    )
}

/// Construct the unique abstract B5 intertwiner from one level-18 source
/// irrep tensored with the spinor into one of the four `(11000) tensor S`
/// target channels.
pub fn build_level18_abstract(
    source_dynkin_label: &str,
    target_dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> AbstractCouplingCertificate {
    assert!(
        crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
            .iter()
            .any(|(target, _)| target == target_dynkin_label),
        "requested level-18 source-target channel is absent"
    );
    build_abstract_from_fixture(
        level18_problem(target_dynkin_label),
        source_dynkin_label,
        fixture_copy,
        coefficient_width_bytes,
        fixture_bytes,
    )
    .0
}

/// Lower-memory construction for a large level-18 weight multiplicity.
///
/// Modular row selection is used to avoid repeated rational Gram allocations.
/// The nullspace is independently attempted over two primes, and is accepted
/// only after exact characteristic-zero residual verification.  If that proof
/// gate fails, the nullspace calculation falls back to BigInt arithmetic.
pub fn build_level18_abstract_low_memory(
    source_dynkin_label: &str,
    target_dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> AbstractCouplingCertificate {
    assert!(
        crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
            .iter()
            .any(|(target, _)| target == target_dynkin_label),
        "requested level-18 source-target channel is absent"
    );
    build_abstract_from_fixture_with_strategies(
        level18_problem(target_dynkin_label),
        source_dynkin_label,
        fixture_copy,
        coefficient_width_bytes,
        fixture_bytes,
        BasisSelection::ModularLowMemory,
        NullspaceStrategy::ModularFirst,
    )
    .0
}

/// Apply a certified abstract coupling to one exact level-18 exterior
/// embedding and verify all five highest-weight raising equations.
pub fn verify_level18_embedding_with_abstract(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> EmbeddedCouplingCertificate {
    let problem = level18_problem(&abstract_certificate.target_dynkin_label);
    let expected_width = if fixture_artifact.ends_with(".i32le") {
        4
    } else {
        2
    };
    assert_eq!(coefficient_width_bytes, expected_width);
    verify_embedded_with_abstract(
        problem,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        fixture_bytes,
    )
}

/// The level-18 verifier together with a SHA-256 digest of every nonzero
/// `(free spinor, exterior basis ordinal, exact i128 coefficient)` entry.
pub fn verify_level18_embedding_with_hash(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> (EmbeddedCouplingCertificate, String) {
    let problem = level18_problem(&abstract_certificate.target_dynkin_label);
    let expected_width = if fixture_artifact.ends_with(".i32le") {
        4
    } else {
        2
    };
    assert_eq!(coefficient_width_bytes, expected_width);
    verify_embedded_with_abstract_and_hash(
        problem,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        fixture_bytes,
    )
}

/// Construct the multiplicity-one abstract B5 intertwiner from a level-12
/// source irrep tensored with the spinor into any of the six exact
/// second-momentum channels. The returned certificate still has to be applied
/// independently to every exact exterior source copy.
pub(crate) fn build_second_momentum_abstract(
    target_dynkin_label: &str,
    source_dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> AbstractCouplingCertificate {
    let problem = second_momentum_problem(target_dynkin_label);
    assert!(
        crate::eleven_dimensional_prepotential::spinor_tensor_channels(source_dynkin_label)
            .iter()
            .any(|(target, _)| target == problem.target_dynkin_label),
        "requested level-12 source does not couple to ({target_dynkin_label})"
    );
    build_abstract_from_fixture(
        problem,
        source_dynkin_label,
        fixture_copy,
        coefficient_width_bytes,
        fixture_bytes,
    )
    .0
}

/// Apply one certified abstract second-momentum coupling to an exact level-12
/// source embedding and hash every nonzero component of the resulting map.
pub(crate) fn verify_second_momentum_embedding_with_hash(
    target_dynkin_label: &str,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> (EmbeddedCouplingCertificate, String) {
    let problem = second_momentum_problem(target_dynkin_label);
    assert_eq!(
        abstract_certificate.target_dynkin_label,
        problem.target_dynkin_label
    );
    let expected_width = if fixture_artifact.ends_with(".i32le") {
        4
    } else {
        assert!(fixture_artifact.ends_with(".i16le"));
        2
    };
    assert_eq!(coefficient_width_bytes, expected_width);
    verify_embedded_with_abstract_and_hash(
        problem,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        fixture_bytes,
    )
}

/// Construct the multiplicity-one abstract B5 intertwiner from a level-12
/// source irrep tensored with the spinor into the `(20001)` second-momentum
/// channel. The returned certificate still has to be applied independently to
/// every exact exterior source copy.
pub(crate) fn build_second_momentum_20001_abstract(
    source_dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> AbstractCouplingCertificate {
    build_second_momentum_abstract(
        "20001",
        source_dynkin_label,
        fixture_copy,
        coefficient_width_bytes,
        fixture_bytes,
    )
}

/// Apply one certified `(20001)` abstract coupling to an exact level-12
/// source embedding and hash every nonzero component of the resulting map.
pub(crate) fn verify_second_momentum_20001_embedding_with_hash(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> (EmbeddedCouplingCertificate, String) {
    verify_second_momentum_embedding_with_hash(
        "20001",
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
    )
}

fn prepare_verified_second_momentum_sparse_highest64(
    problem: CouplingProblem,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
) -> io::Result<VerifiedSparseHighest64> {
    if !abstract_certificate.passed
        || abstract_certificate.target_dynkin_label != problem.target_dynkin_label
        || abstract_certificate.exact_raising_residual_terms_by_simple_root != [0; 5]
        || abstract_certificate.kernel_dimension != 1
        || abstract_certificate.domain_basis.len()
            != abstract_certificate.primitive_domain_coefficients.len()
        || abstract_certificate.source_dynkin_label.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "uncertified level-12 source-to-({}) abstract map",
                problem.target_dynkin_label
            ),
        ));
    }
    let expected_width = fixture_coefficient_width(fixture_artifact);
    if coefficient_width_bytes != expected_width
        || fixture_copy == 0
        || requested_pbw_words.is_empty()
        || requested_pbw_words
            .iter()
            .any(|word| word.iter().any(|root| !(1..=5).contains(root)))
        || requested_pbw_words.iter().collect::<BTreeSet<_>>().len() != requested_pbw_words.len()
        || start_word_ordinal > requested_pbw_words.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid level-12 ({}) fixture, PBW plan, or start ordinal",
                problem.target_dynkin_label
            ),
        ));
    }
    let source_fixture_sha256 = sha256_bytes(fixture_bytes);
    if source_fixture_sha256 != expected_source_fixture_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "level-12 source fixture SHA-256 mismatch",
        ));
    }

    let (mut model, highest, maximum_absolute_checked_accumulator) = materialize_coupled_highest(
        problem,
        abstract_certificate,
        fixture_bytes,
        coefficient_width_bytes,
    );
    let mut coupled_hasher = Sha256::new();
    coupled_hasher.update(b"adynkra-11d-canonical-embedded-map-v1\0");
    for (spinor_index, exterior) in &highest.components {
        for (exterior_ordinal, value) in exterior.coefficients.iter().enumerate() {
            if *value == 0 {
                continue;
            }
            coupled_hasher.update((*spinor_index as u64).to_le_bytes());
            coupled_hasher.update((exterior_ordinal as u64).to_le_bytes());
            coupled_hasher.update(i128::from(*value).to_le_bytes());
        }
    }
    let coupled_map_sha256 = format!("{:x}", coupled_hasher.finalize());
    if coupled_map_sha256 != expected_coupled_map_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "reconstructed level-12 ({}) map disagrees with checkpoint hash",
                problem.target_dynkin_label
            ),
        ));
    }

    let highest_sparse = dense_coupled_to_sparse64(&mut model, &highest);
    let estimated_payload_bytes = exterior_model_payload_bytes(&model)
        + coupled_state_payload_bytes(&highest)
        + sparse64_payload_bytes(&highest_sparse);
    let maximum_absolute_coefficient = highest_sparse
        .components
        .values()
        .flatten()
        .map(|(_, coefficient)| coefficient.unsigned_abs())
        .max()
        .unwrap_or(0);
    let canonical = CanonicalSparseHighest64 {
        state: highest_sparse,
        maximum_absolute_coefficient,
    };
    canonical.visit_terms(|_, _| Ok(()))?;
    Ok(VerifiedSparseHighest64 {
        highest: canonical,
        maximum_absolute_checked_accumulator,
        source_fixture_sha256,
        coupled_map_sha256,
        estimated_payload_bytes,
    })
}

/// Generic opaque-handle descendant traversal for any of the six exact
/// second-momentum intermediate channels. This is the common execution
/// boundary used by the full 77-column inventory.
pub(crate) fn visit_second_momentum_descendant_handles_range<H, U, L, F>(
    target_dynkin_label: &str,
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    upload_highest: U,
    mut lower_word: L,
    mut visit: F,
) -> io::Result<OpaqueSecondMomentumDescendantAccounting>
where
    U: FnOnce(&CanonicalSparseHighest64) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<u64>,
{
    if start_word_ordinal > end_word_ordinal_exclusive
        || end_word_ordinal_exclusive > requested_pbw_words.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid generic second-momentum descendant PBW word range",
        ));
    }
    let problem = second_momentum_problem(target_dynkin_label);
    let verified = prepare_verified_second_momentum_sparse_highest64(
        problem,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
    )?;
    let mut maximum = verified.maximum_absolute_checked_accumulator;
    let highest = upload_highest(&verified.highest)?;
    let mut emitted_nonzero_components = 0_u64;
    visit_independent_coupled_word_handle_events_range(
        &highest,
        requested_pbw_words,
        start_word_ordinal,
        end_word_ordinal_exclusive,
        &mut maximum,
        &mut lower_word,
        &mut |event| {
            let is_state = matches!(&event, CoupledWordStateEvent::State { .. });
            let consumed = visit(event)?;
            if !is_state && consumed != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "generic opaque descendant visitor counted a boundary event",
                ));
            }
            if is_state {
                emitted_nonzero_components = emitted_nonzero_components
                    .checked_add(consumed)
                    .ok_or_else(|| io::Error::other("generic descendant term count overflow"))?;
            }
            Ok(())
        },
    )?;
    Ok(OpaqueSecondMomentumDescendantAccounting {
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        source_fixture_sha256: verified.source_fixture_sha256,
        coupled_map_sha256: verified.coupled_map_sha256,
        requested_pbw_words: end_word_ordinal_exclusive - start_word_ordinal,
        emitted_nonzero_components,
        estimated_host_payload_bytes: verified.estimated_payload_bytes,
        checkpoint_hash_parity_verified: true,
    })
}

/// Verify and upload one canonical `(20001)` highest state, then traverse the
/// requested PBW suffix using opaque backend handles. The highest state is
/// uploaded exactly once. Prefix and terminal handles remain backend-owned;
/// only the callback decides whether a terminal state is downloaded.
///
/// The callback must return the number of canonical nonzero components it
/// consumed for `State` and zero for boundary events. `WordLoweringStart` is
/// emitted before any backend lowering for that word. A completed word is
/// durable only after the subsequent `WordEnd` event.
pub(crate) fn visit_second_momentum_20001_descendant_handles_from<H, U, L, F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    upload_highest: U,
    lower_word: L,
    visit: F,
) -> io::Result<OpaqueSecondMomentumDescendantAccounting>
where
    U: FnOnce(&CanonicalSparseHighest64) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<u64>,
{
    visit_second_momentum_20001_descendant_handles_range(
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
        requested_pbw_words.len(),
        upload_highest,
        lower_word,
        visit,
    )
}

pub(crate) fn visit_second_momentum_20001_descendant_handles_range<H, U, L, F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    upload_highest: U,
    mut lower_word: L,
    mut visit: F,
) -> io::Result<OpaqueSecondMomentumDescendantAccounting>
where
    U: FnOnce(&CanonicalSparseHighest64) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<u64>,
{
    if start_word_ordinal > end_word_ordinal_exclusive
        || end_word_ordinal_exclusive > requested_pbw_words.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid (20001) descendant PBW word range",
        ));
    }
    let verified = prepare_verified_second_momentum_sparse_highest64(
        SECOND_MOMENTUM_20001_PROBLEM,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
    )?;
    let mut maximum = verified.maximum_absolute_checked_accumulator;
    let highest = upload_highest(&verified.highest)?;
    let mut emitted_nonzero_components = 0_u64;
    visit_independent_coupled_word_handle_events_range(
        &highest,
        requested_pbw_words,
        start_word_ordinal,
        end_word_ordinal_exclusive,
        &mut maximum,
        &mut lower_word,
        &mut |event| {
            let is_state = matches!(&event, CoupledWordStateEvent::State { .. });
            let consumed = visit(event)?;
            if !is_state && consumed != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "opaque descendant visitor counted a boundary event",
                ));
            }
            if is_state {
                emitted_nonzero_components = emitted_nonzero_components
                    .checked_add(consumed)
                    .ok_or_else(|| io::Error::other("(20001) descendant term count overflow"))?;
            }
            Ok(())
        },
    )?;
    Ok(OpaqueSecondMomentumDescendantAccounting {
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        source_fixture_sha256: verified.source_fixture_sha256,
        coupled_map_sha256: verified.coupled_map_sha256,
        requested_pbw_words: end_word_ordinal_exclusive - start_word_ordinal,
        emitted_nonzero_components,
        estimated_host_payload_bytes: verified.estimated_payload_bytes,
        checkpoint_hash_parity_verified: true,
    })
}

/// Materialize only the explicitly requested PBW descendants of one exact
/// level-12 source map into `(20001)`.
///
/// The caller must provide the source-fixture and embedded-map hashes pinned
/// by the completed map checkpoint.  Before any descendant is emitted, the
/// reconstructed highest state is hashed with the same canonical byte stream
/// as [`verify_second_momentum_20001_embedding_with_hash`].  A mismatch fails
/// closed.  This makes the visitor an exact continuation of the checkpointed
/// component map rather than a second, unpinned reconstruction.
pub(crate) fn visit_second_momentum_20001_descendant_components<F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    mut visit: F,
) -> io::Result<SecondMomentum20001DescendantAccounting>
where
    F: FnMut(SecondMomentum20001DescendantEntry) -> io::Result<()>,
{
    visit_second_momentum_20001_descendant_events_from(
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        0,
        |event| match event {
            SecondMomentum20001DescendantEvent::Component(entry) => visit(entry),
            SecondMomentum20001DescendantEvent::WordLoweringStart { .. }
            | SecondMomentum20001DescendantEvent::WordStart { .. }
            | SecondMomentum20001DescendantEvent::WordEnd { .. } => Ok(()),
        },
    )
}

pub(crate) fn visit_second_momentum_20001_descendant_events_from<F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    visit: F,
) -> io::Result<SecondMomentum20001DescendantAccounting>
where
    F: FnMut(SecondMomentum20001DescendantEvent) -> io::Result<()>,
{
    visit_second_momentum_20001_descendant_events_range(
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
        requested_pbw_words.len(),
        visit,
    )
}

pub(crate) fn visit_second_momentum_20001_descendant_events_range<F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    mut visit: F,
) -> io::Result<SecondMomentum20001DescendantAccounting>
where
    F: FnMut(SecondMomentum20001DescendantEvent) -> io::Result<()>,
{
    if !abstract_certificate.passed
        || abstract_certificate.target_dynkin_label
            != SECOND_MOMENTUM_20001_PROBLEM.target_dynkin_label
        || abstract_certificate.exact_raising_residual_terms_by_simple_root != [0; 5]
        || abstract_certificate.kernel_dimension != 1
        || abstract_certificate.domain_basis.len()
            != abstract_certificate.primitive_domain_coefficients.len()
        || abstract_certificate.source_dynkin_label.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "uncertified level-12 source-to-(20001) abstract map",
        ));
    }
    let expected_width = fixture_coefficient_width(fixture_artifact);
    if coefficient_width_bytes != expected_width
        || fixture_copy == 0
        || requested_pbw_words.is_empty()
        || requested_pbw_words
            .iter()
            .any(|word| word.iter().any(|root| !(1..=5).contains(root)))
        || requested_pbw_words.iter().collect::<BTreeSet<_>>().len() != requested_pbw_words.len()
        || start_word_ordinal > end_word_ordinal_exclusive
        || end_word_ordinal_exclusive > requested_pbw_words.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid level-12 (20001) fixture or requested PBW-word filter",
        ));
    }
    let source_fixture_sha256 = sha256_bytes(fixture_bytes);
    if source_fixture_sha256 != expected_source_fixture_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "level-12 source fixture SHA-256 mismatch",
        ));
    }

    let (mut model, highest, mut maximum) = materialize_coupled_highest(
        SECOND_MOMENTUM_20001_PROBLEM,
        abstract_certificate,
        fixture_bytes,
        coefficient_width_bytes,
    );
    let mut coupled_hasher = Sha256::new();
    coupled_hasher.update(b"adynkra-11d-canonical-embedded-map-v1\0");
    for (spinor_index, exterior) in &highest.components {
        for (exterior_ordinal, value) in exterior.coefficients.iter().enumerate() {
            if *value == 0 {
                continue;
            }
            coupled_hasher.update((*spinor_index as u64).to_le_bytes());
            coupled_hasher.update((exterior_ordinal as u64).to_le_bytes());
            coupled_hasher.update(i128::from(*value).to_le_bytes());
        }
    }
    let coupled_map_sha256 = format!("{:x}", coupled_hasher.finalize());
    if coupled_map_sha256 != expected_coupled_map_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "reconstructed level-12 (20001) map disagrees with checkpoint hash",
        ));
    }

    let highest_sparse = dense_coupled_to_sparse64(&mut model, &highest);
    let estimated_payload_bytes = exterior_model_payload_bytes(&model)
        + coupled_state_payload_bytes(&highest)
        + sparse64_payload_bytes(&highest_sparse);
    drop(highest);
    drop(model);
    let mut emitted_nonzero_components = 0_u64;
    let mut current_word_components = 0_u64;
    visit_independent_sparse_coupled_word_events_range(
        &highest_sparse,
        requested_pbw_words,
        start_word_ordinal,
        end_word_ordinal_exclusive,
        &mut maximum,
        &mut |event| {
            match event {
                CoupledWordStateEvent::WordLoweringStart { ordinal, pbw_word } => {
                    visit(SecondMomentum20001DescendantEvent::WordLoweringStart {
                        requested_word_ordinal: ordinal,
                        pbw_word_simple_roots: pbw_word.to_vec(),
                    })?;
                }
                CoupledWordStateEvent::WordStart { ordinal, pbw_word } => {
                    current_word_components = 0;
                    visit(SecondMomentum20001DescendantEvent::WordStart {
                        requested_word_ordinal: ordinal,
                        pbw_word_simple_roots: pbw_word.to_vec(),
                    })?;
                }
                CoupledWordStateEvent::State { ordinal, state } => {
                    for (&free_spinor_weight_index, exterior) in &state.components {
                        for &(exterior_mask, coefficient) in exterior {
                            emitted_nonzero_components =
                                emitted_nonzero_components.checked_add(1).ok_or_else(|| {
                                    io::Error::other("(20001) descendant term count overflow")
                                })?;
                            current_word_components =
                                current_word_components.checked_add(1).ok_or_else(|| {
                                    io::Error::other("(20001) word component count overflow")
                                })?;
                            visit(SecondMomentum20001DescendantEvent::Component(
                                SecondMomentum20001DescendantEntry {
                                    requested_word_ordinal: ordinal,
                                    free_spinor_weight_index,
                                    exterior_mask,
                                    coefficient,
                                },
                            ))?;
                        }
                    }
                }
                CoupledWordStateEvent::WordEnd { ordinal } => {
                    visit(SecondMomentum20001DescendantEvent::WordEnd {
                        requested_word_ordinal: ordinal,
                        emitted_nonzero_components: current_word_components,
                    })?;
                }
            }
            Ok(())
        },
    )?;
    Ok(SecondMomentum20001DescendantAccounting {
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        source_fixture_sha256,
        coupled_map_sha256,
        requested_pbw_words: end_word_ordinal_exclusive - start_word_ordinal,
        emitted_nonzero_components,
        maximum_absolute_checked_accumulator: maximum,
        estimated_payload_bytes,
        checkpoint_hash_parity_verified: true,
    })
}

/// Construct the multiplicity-one abstract B5 intertwiner from a level-12
/// source irrep tensored with the spinor into the `(30001)` second-momentum
/// channel. The returned certificate still has to be applied independently to
/// every exact exterior source copy.
pub(crate) fn build_second_momentum_30001_abstract(
    source_dynkin_label: &str,
    fixture_copy: usize,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> AbstractCouplingCertificate {
    build_second_momentum_abstract(
        "30001",
        source_dynkin_label,
        fixture_copy,
        coefficient_width_bytes,
        fixture_bytes,
    )
}

/// Apply one certified `(30001)` abstract coupling to an exact level-12
/// source embedding and hash every nonzero component of the resulting map.
pub(crate) fn verify_second_momentum_30001_embedding_with_hash(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
) -> (EmbeddedCouplingCertificate, String) {
    verify_second_momentum_embedding_with_hash(
        "30001",
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
    )
}

/// Verify and upload one canonical `(30001)` highest state, then traverse the
/// requested PBW suffix using opaque backend handles. See the `(20001)`
/// counterpart for event and component-count semantics.
pub(crate) fn visit_second_momentum_30001_descendant_handles_from<H, U, L, F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    upload_highest: U,
    lower_word: L,
    visit: F,
) -> io::Result<OpaqueSecondMomentumDescendantAccounting>
where
    U: FnOnce(&CanonicalSparseHighest64) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<u64>,
{
    visit_second_momentum_30001_descendant_handles_range(
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
        requested_pbw_words.len(),
        upload_highest,
        lower_word,
        visit,
    )
}

pub(crate) fn visit_second_momentum_30001_descendant_handles_range<H, U, L, F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    upload_highest: U,
    mut lower_word: L,
    mut visit: F,
) -> io::Result<OpaqueSecondMomentumDescendantAccounting>
where
    U: FnOnce(&CanonicalSparseHighest64) -> io::Result<H>,
    L: FnMut(&H, &[u8], &mut i128) -> io::Result<H>,
    F: FnMut(CoupledWordStateEvent<'_, H>) -> io::Result<u64>,
{
    if start_word_ordinal > end_word_ordinal_exclusive
        || end_word_ordinal_exclusive > requested_pbw_words.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid (30001) descendant PBW word range",
        ));
    }
    let verified = prepare_verified_second_momentum_sparse_highest64(
        SECOND_MOMENTUM_30001_PROBLEM,
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
    )?;
    let mut maximum = verified.maximum_absolute_checked_accumulator;
    let highest = upload_highest(&verified.highest)?;
    let mut emitted_nonzero_components = 0_u64;
    visit_independent_coupled_word_handle_events_range(
        &highest,
        requested_pbw_words,
        start_word_ordinal,
        end_word_ordinal_exclusive,
        &mut maximum,
        &mut lower_word,
        &mut |event| {
            let is_state = matches!(&event, CoupledWordStateEvent::State { .. });
            let consumed = visit(event)?;
            if !is_state && consumed != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "opaque descendant visitor counted a boundary event",
                ));
            }
            if is_state {
                emitted_nonzero_components = emitted_nonzero_components
                    .checked_add(consumed)
                    .ok_or_else(|| io::Error::other("(30001) descendant term count overflow"))?;
            }
            Ok(())
        },
    )?;
    Ok(OpaqueSecondMomentumDescendantAccounting {
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        source_fixture_sha256: verified.source_fixture_sha256,
        coupled_map_sha256: verified.coupled_map_sha256,
        requested_pbw_words: end_word_ordinal_exclusive - start_word_ordinal,
        emitted_nonzero_components,
        estimated_host_payload_bytes: verified.estimated_payload_bytes,
        checkpoint_hash_parity_verified: true,
    })
}

/// Materialize only the explicitly requested PBW descendants of one exact
/// level-12 source map into `(30001)`.
///
/// The caller must provide the source-fixture and embedded-map hashes pinned
/// by the completed map checkpoint.  Before any descendant is emitted, the
/// reconstructed highest state is hashed with the same canonical byte stream
/// as [`verify_second_momentum_30001_embedding_with_hash`].  A mismatch fails
/// closed.  This makes the visitor an exact continuation of the checkpointed
/// component map rather than a second, unpinned reconstruction.
pub(crate) fn visit_second_momentum_30001_descendant_components<F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    mut visit: F,
) -> io::Result<SecondMomentum30001DescendantAccounting>
where
    F: FnMut(SecondMomentum30001DescendantEntry) -> io::Result<()>,
{
    visit_second_momentum_30001_descendant_events_from(
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        0,
        |event| match event {
            SecondMomentum30001DescendantEvent::Component(entry) => visit(entry),
            SecondMomentum30001DescendantEvent::WordLoweringStart { .. }
            | SecondMomentum30001DescendantEvent::WordStart { .. }
            | SecondMomentum30001DescendantEvent::WordEnd { .. } => Ok(()),
        },
    )
}

pub(crate) fn visit_second_momentum_30001_descendant_events_from<F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    visit: F,
) -> io::Result<SecondMomentum30001DescendantAccounting>
where
    F: FnMut(SecondMomentum30001DescendantEvent) -> io::Result<()>,
{
    visit_second_momentum_30001_descendant_events_range(
        abstract_certificate,
        fixture_copy,
        fixture_artifact,
        coefficient_width_bytes,
        fixture_bytes,
        expected_source_fixture_sha256,
        expected_coupled_map_sha256,
        requested_pbw_words,
        start_word_ordinal,
        requested_pbw_words.len(),
        visit,
    )
}

pub(crate) fn visit_second_momentum_30001_descendant_events_range<F>(
    abstract_certificate: &AbstractCouplingCertificate,
    fixture_copy: usize,
    fixture_artifact: &str,
    coefficient_width_bytes: usize,
    fixture_bytes: &[u8],
    expected_source_fixture_sha256: &str,
    expected_coupled_map_sha256: &str,
    requested_pbw_words: &[Vec<u8>],
    start_word_ordinal: usize,
    end_word_ordinal_exclusive: usize,
    mut visit: F,
) -> io::Result<SecondMomentum30001DescendantAccounting>
where
    F: FnMut(SecondMomentum30001DescendantEvent) -> io::Result<()>,
{
    if !abstract_certificate.passed
        || abstract_certificate.target_dynkin_label
            != SECOND_MOMENTUM_30001_PROBLEM.target_dynkin_label
        || abstract_certificate.exact_raising_residual_terms_by_simple_root != [0; 5]
        || abstract_certificate.kernel_dimension != 1
        || abstract_certificate.domain_basis.len()
            != abstract_certificate.primitive_domain_coefficients.len()
        || abstract_certificate.source_dynkin_label.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "uncertified level-12 source-to-(30001) abstract map",
        ));
    }
    let expected_width = fixture_coefficient_width(fixture_artifact);
    if coefficient_width_bytes != expected_width
        || fixture_copy == 0
        || requested_pbw_words.is_empty()
        || requested_pbw_words
            .iter()
            .any(|word| word.iter().any(|root| !(1..=5).contains(root)))
        || requested_pbw_words.iter().collect::<BTreeSet<_>>().len() != requested_pbw_words.len()
        || start_word_ordinal > end_word_ordinal_exclusive
        || end_word_ordinal_exclusive > requested_pbw_words.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid level-12 (30001) fixture or requested PBW-word filter",
        ));
    }
    let source_fixture_sha256 = sha256_bytes(fixture_bytes);
    if source_fixture_sha256 != expected_source_fixture_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "level-12 source fixture SHA-256 mismatch",
        ));
    }

    let (mut model, highest, mut maximum) = materialize_coupled_highest(
        SECOND_MOMENTUM_30001_PROBLEM,
        abstract_certificate,
        fixture_bytes,
        coefficient_width_bytes,
    );
    let mut coupled_hasher = Sha256::new();
    coupled_hasher.update(b"adynkra-11d-canonical-embedded-map-v1\0");
    for (spinor_index, exterior) in &highest.components {
        for (exterior_ordinal, value) in exterior.coefficients.iter().enumerate() {
            if *value == 0 {
                continue;
            }
            coupled_hasher.update((*spinor_index as u64).to_le_bytes());
            coupled_hasher.update((exterior_ordinal as u64).to_le_bytes());
            coupled_hasher.update(i128::from(*value).to_le_bytes());
        }
    }
    let coupled_map_sha256 = format!("{:x}", coupled_hasher.finalize());
    if coupled_map_sha256 != expected_coupled_map_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "reconstructed level-12 (30001) map disagrees with checkpoint hash",
        ));
    }

    let highest_sparse = dense_coupled_to_sparse64(&mut model, &highest);
    let estimated_payload_bytes = exterior_model_payload_bytes(&model)
        + coupled_state_payload_bytes(&highest)
        + sparse64_payload_bytes(&highest_sparse);
    drop(highest);
    drop(model);
    let mut emitted_nonzero_components = 0_u64;
    let mut current_word_components = 0_u64;
    visit_independent_sparse_coupled_word_events_range(
        &highest_sparse,
        requested_pbw_words,
        start_word_ordinal,
        end_word_ordinal_exclusive,
        &mut maximum,
        &mut |event| {
            match event {
                CoupledWordStateEvent::WordLoweringStart { ordinal, pbw_word } => {
                    visit(SecondMomentum30001DescendantEvent::WordLoweringStart {
                        requested_word_ordinal: ordinal,
                        pbw_word_simple_roots: pbw_word.to_vec(),
                    })?;
                }
                CoupledWordStateEvent::WordStart { ordinal, pbw_word } => {
                    current_word_components = 0;
                    visit(SecondMomentum30001DescendantEvent::WordStart {
                        requested_word_ordinal: ordinal,
                        pbw_word_simple_roots: pbw_word.to_vec(),
                    })?;
                }
                CoupledWordStateEvent::State { ordinal, state } => {
                    for (&free_spinor_weight_index, exterior) in &state.components {
                        for &(exterior_mask, coefficient) in exterior {
                            emitted_nonzero_components =
                                emitted_nonzero_components.checked_add(1).ok_or_else(|| {
                                    io::Error::other("(30001) descendant term count overflow")
                                })?;
                            current_word_components =
                                current_word_components.checked_add(1).ok_or_else(|| {
                                    io::Error::other("(30001) word component count overflow")
                                })?;
                            visit(SecondMomentum30001DescendantEvent::Component(
                                SecondMomentum30001DescendantEntry {
                                    requested_word_ordinal: ordinal,
                                    free_spinor_weight_index,
                                    exterior_mask,
                                    coefficient,
                                },
                            ))?;
                        }
                    }
                }
                CoupledWordStateEvent::WordEnd { ordinal } => {
                    visit(SecondMomentum30001DescendantEvent::WordEnd {
                        requested_word_ordinal: ordinal,
                        emitted_nonzero_components: current_word_components,
                    })?;
                }
            }
            Ok(())
        },
    )?;
    Ok(SecondMomentum30001DescendantAccounting {
        source_dynkin_label: abstract_certificate.source_dynkin_label.clone(),
        source_copy: fixture_copy,
        source_fixture: fixture_artifact.to_string(),
        source_fixture_sha256,
        coupled_map_sha256,
        requested_pbw_words: end_word_ordinal_exclusive - start_word_ordinal,
        emitted_nonzero_components,
        maximum_absolute_checked_accumulator: maximum,
        estimated_payload_bytes,
        checkpoint_hash_parity_verified: true,
    })
}

pub fn summarize_first_momentum(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_maps: Vec<EmbeddedCouplingCertificate>,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> FirstMomentumCouplingCertificateReport {
    let source_target_pairs_certified = abstract_couplings
        .iter()
        .filter(|report| report.passed)
        .count();
    let embedded_maps_certified = embedded_maps.iter().filter(|report| report.passed).count();
    let every_residual_is_exactly_zero = embedded_maps
        .iter()
        .all(|report| report.exact_raising_residual_terms_by_simple_root == [0; 5]);
    let passed = source_target_pairs_certified == 23
        && embedded_maps_certified == 44
        && every_residual_is_exactly_zero;
    FirstMomentumCouplingCertificateReport {
        schema_version: "adynkra-11d-first-momentum-all-couplings-v1".to_string(),
        role: "exact certification of the 23 abstract and 44 embedded level-14 source intertwiners"
            .to_string(),
        abstract_couplings,
        embedded_maps,
        source_target_pairs_certified,
        embedded_maps_certified,
        expected_source_target_pairs: 23,
        expected_embedded_maps: 44,
        every_residual_is_exactly_zero,
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
        boundary: "this certifies the level-14 source intertwiners into the four first-momentum intermediate irreducibles; it does not construct the momentum target couplings, the joint compatibility matrix, a gauge quotient, or a field equation".to_string(),
        passed,
    }
}

fn summarize_problem(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
    expected_distinct_source_irreps: usize,
    expected_embedded_source_copies: usize,
    schema_version: &str,
    role: &str,
    boundary: &str,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> AllCouplingCertificateReport {
    let distinct_source_irreps_certified = abstract_couplings
        .iter()
        .filter(|report| report.passed)
        .count();
    let embedded_source_copies_certified = embedded_copies
        .iter()
        .filter(|report| report.passed)
        .count();
    let every_residual_is_exactly_zero = embedded_copies
        .iter()
        .all(|report| report.exact_raising_residual_terms_by_simple_root == [0; 5]);
    let passed = distinct_source_irreps_certified == expected_distinct_source_irreps
        && embedded_source_copies_certified == expected_embedded_source_copies
        && every_residual_is_exactly_zero;
    AllCouplingCertificateReport {
        schema_version: schema_version.to_string(),
        role: role.to_string(),
        abstract_couplings,
        embedded_copies,
        distinct_source_irreps_certified,
        embedded_source_copies_certified,
        expected_distinct_source_irreps,
        expected_embedded_source_copies,
        every_residual_is_exactly_zero,
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
        boundary: boundary.to_string(),
        passed,
    }
}

pub fn summarize_all(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> AllCouplingCertificateReport {
    summarize_problem(
        abstract_couplings,
        embedded_copies,
        8,
        12,
        "adynkra-11d-level16-all-couplings-v1",
        "exact dense certification of all level-16 source couplings into (10001)",
        "this certifies the twelve source embeddings and their couplings into the (10001) channel under the stated exterior-algebra conventions; it does not solve the full Gates-Hu prepotential problem",
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
    )
}

pub fn summarize_hooks(
    abstract_couplings: Vec<AbstractCouplingCertificate>,
    embedded_copies: Vec<EmbeddedCouplingCertificate>,
    execution_workers: usize,
    memory_budget_gib: usize,
    estimated_memory_gib_per_worker: usize,
    resumed_from_atomic_checkpoints: bool,
) -> AllCouplingCertificateReport {
    summarize_problem(
        abstract_couplings,
        embedded_copies,
        4,
        7,
        "adynkra-11d-level17-hook-all-couplings-v1",
        "exact dense certification of all level-17 source couplings into (11000)",
        "this certifies the seven source embeddings and their couplings into the (11000) hook channel; the derivative matrix is a separate calculation",
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resumed_from_atomic_checkpoints,
    )
}

#[allow(dead_code)]
pub fn verify_all() -> AllCouplingCertificateReport {
    let fixtures = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures();
    let mut grouped = BTreeMap::<&str, Vec<_>>::new();
    for fixture in &fixtures {
        grouped
            .entry(fixture.dynkin_label)
            .or_default()
            .push(*fixture);
    }
    let mut abstract_couplings = Vec::new();
    let mut embedded_copies = Vec::new();
    for (label, copies) in grouped {
        let first = copies
            .iter()
            .find(|fixture| fixture.copy == 1)
            .expect("each irrep must have copy 1");
        let (abstract_certificate, _) =
            build_abstract_from_fixture(LEVEL16_PROBLEM, label, first.copy, 2, first.bytes);
        for fixture in copies {
            embedded_copies.push(verify_embedded_with_abstract(
                LEVEL16_PROBLEM,
                &abstract_certificate,
                fixture.copy,
                fixture.artifact,
                fixture.bytes,
            ));
        }
        abstract_couplings.push(abstract_certificate);
    }
    summarize_all(abstract_couplings, embedded_copies, 1, 0, 0, false)
}

pub fn write_atomic_json<T: Serialize>(output: &Path, report: &T, passed: bool) -> io::Result<()> {
    if !passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "refusing to checkpoint a failed certificate",
        ));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = temporary_path(output);
    let payload = serde_json::to_vec_pretty(report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let parsed: serde_json::Value = serde_json::from_slice(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if parsed.get("passed").and_then(|value| value.as_bool()) != Some(true) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "serialized certificate does not contain passed=true",
        ));
    }
    {
        let mut file = File::create(&temporary)?;
        file.write_all(&payload)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temporary, output)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn temporary_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("certificate.json");
    output.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct AbstractCertificateFixtureReport {
        abstract_couplings: Vec<AbstractCouplingCertificate>,
    }

    fn assert_certificate_core_matches(
        exact: &AbstractCouplingCertificate,
        modular_first: &AbstractCouplingCertificate,
    ) {
        assert!(exact.passed);
        assert!(modular_first.passed);
        assert_eq!(exact.source_dynkin_label, modular_first.source_dynkin_label);
        assert_eq!(exact.target_dynkin_label, modular_first.target_dynkin_label);
        assert_eq!(exact.basis_method, modular_first.basis_method);
        assert_eq!(
            serde_json::to_value(&exact.domain_basis).unwrap(),
            serde_json::to_value(&modular_first.domain_basis).unwrap()
        );
        assert_eq!(
            exact.source_weight_multiplicities,
            modular_first.source_weight_multiplicities
        );
        assert_eq!(exact.gram_matrix_rank, modular_first.gram_matrix_rank);
        assert_eq!(exact.kernel_dimension, modular_first.kernel_dimension);
        assert_eq!(
            exact.primitive_domain_coefficients,
            modular_first.primitive_domain_coefficients
        );
        assert_eq!(
            exact.exact_raising_residual_terms_by_simple_root,
            modular_first.exact_raising_residual_terms_by_simple_root
        );
        assert_eq!(exact.passed, modular_first.passed);
    }

    #[test]
    fn second_momentum_problem_uses_the_existing_dynkin_weight_contract() {
        for target in ["00001", "01001", "10001", "11001", "20001", "30001"] {
            let problem = second_momentum_problem(target);
            assert_eq!(problem.exterior_degree, 12);
            assert_eq!(problem.target_dynkin_label, target);
            assert_eq!(
                problem.target_weight,
                dynkin_highest_weight_for_label(target)
            );
        }
        assert_eq!(
            second_momentum_problem("20001"),
            SECOND_MOMENTUM_20001_PROBLEM
        );
        assert_eq!(
            second_momentum_problem("30001"),
            SECOND_MOMENTUM_30001_PROBLEM
        );
        for target in ["01001", "10001", "11001", "20001"] {
            assert_eq!(
                level18_problem(target).target_weight,
                dynkin_highest_weight_for_label(target)
            );
        }
    }

    #[test]
    #[ignore = "constructs the real level-12 (20002) to (20001) map twice for wrapper parity"]
    fn second_momentum_20001_wrappers_match_the_shared_exact_engine() {
        let fixture = include_bytes!(
            "../data/eleven_dimensional_spinor_bridge/level12_20002_highest_weight_kernel_1.i16le"
        );
        let wrapped = build_second_momentum_20001_abstract("20002", 1, 2, fixture);
        let generic =
            build_abstract_from_fixture(SECOND_MOMENTUM_20001_PROBLEM, "20002", 1, 2, fixture).0;
        assert_certificate_core_matches(&generic, &wrapped);
        assert_eq!(wrapped.exact_raising_residual_terms_by_simple_root, [0; 5]);

        let (wrapped_embedded, wrapped_hash) = verify_second_momentum_20001_embedding_with_hash(
            &wrapped,
            1,
            "level12_20002_highest_weight_kernel_1.i16le",
            2,
            fixture,
        );
        let (generic_embedded, generic_hash) = verify_embedded_with_abstract_and_hash(
            SECOND_MOMENTUM_20001_PROBLEM,
            &generic,
            1,
            "level12_20002_highest_weight_kernel_1.i16le",
            fixture,
        );
        assert_eq!(wrapped_hash, generic_hash);
        assert_eq!(
            serde_json::to_value(&wrapped_embedded).unwrap(),
            serde_json::to_value(&generic_embedded).unwrap()
        );
        assert_eq!(
            wrapped_embedded.exact_raising_residual_terms_by_simple_root,
            [0; 5]
        );
        assert!(wrapped_embedded.passed);
    }

    #[test]
    fn level18_public_hash_wrapper_still_matches_the_shared_exact_engine() {
        let report: AbstractCertificateFixtureReport = serde_json::from_str(include_str!(
            "../results/adynkra_11d_first_momentum_couplings_all.json"
        ))
        .unwrap();
        let abstract_certificate = report
            .abstract_couplings
            .into_iter()
            .find(|certificate| {
                certificate.source_dynkin_label == "10002"
                    && certificate.target_dynkin_label == "20001"
            })
            .unwrap();
        let fixture = crate::eleven_dimensional_level18_momentum::level18_source_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "10002" && fixture.copy == 1)
            .unwrap();
        let public = verify_level18_embedding_with_hash(
            &abstract_certificate,
            fixture.copy,
            &fixture.artifact,
            fixture.coefficient_width_bytes,
            &fixture.bytes,
        );
        let direct = verify_embedded_with_abstract_and_hash(
            level18_problem("20001"),
            &abstract_certificate,
            fixture.copy,
            &fixture.artifact,
            &fixture.bytes,
        );
        assert_eq!(
            serde_json::to_value(&public.0).unwrap(),
            serde_json::to_value(&direct.0).unwrap()
        );
        assert_eq!(public.1, direct.1);
    }

    fn compare_exact_and_modular_first(
        problem: CouplingProblem,
        source_dynkin_label: &str,
        fixture_copy: usize,
        coefficient_width_bytes: usize,
        fixture_bytes: &[u8],
    ) {
        let exact = build_abstract_from_fixture_with_strategies(
            problem,
            source_dynkin_label,
            fixture_copy,
            coefficient_width_bytes,
            fixture_bytes,
            BasisSelection::ExactCanonical,
            NullspaceStrategy::ExactBigInt,
        )
        .0;
        let modular_first = build_abstract_from_fixture_with_strategies(
            problem,
            source_dynkin_label,
            fixture_copy,
            coefficient_width_bytes,
            fixture_bytes,
            BasisSelection::ExactCanonical,
            NullspaceStrategy::ModularFirst,
        )
        .0;
        assert_certificate_core_matches(&exact, &modular_first);
        assert!(modular_first.dependency_test.contains("two-prime modular"));
        assert!(modular_first.dependency_test.contains("accepted only"));
    }

    #[test]
    fn joint_column_manifest_is_complete_and_unique() {
        let specs = joint_column_specs();
        assert_eq!(specs.len(), 56);
        assert_eq!(
            specs.iter().map(|spec| spec.ordinal).collect::<Vec<_>>(),
            (0..56).collect::<Vec<_>>()
        );
        assert_eq!(
            specs.iter().filter(|spec| spec.kind == "leading").count(),
            12
        );
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.kind == "first-momentum")
                .count(),
            44
        );
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.label.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            56
        );
    }

    #[test]
    fn fixed_work_list_and_multiplicity_gate_pass() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.distinct_source_irreps, 8);
        assert_eq!(report.embedded_source_copies, 12);
        assert!(report.every_target_multiplicity_is_one);
        assert_eq!(
            report
                .tensor_multiplicities
                .iter()
                .filter(|audit| audit.multiplicity_one)
                .count(),
            8
        );
    }

    #[test]
    fn dense_engine_reproduces_the_committed_20000_golden_coupling() {
        let report = build_abstract("20000");
        assert!(report.passed);
        assert_eq!(report.product_weight_domain_dimension, 6);
        assert_eq!(report.kernel_dimension, 1);
        assert_eq!(report.primitive_domain_coefficients, [1, -2, 2, -2, 2, -4]);
        assert_eq!(report.exact_raising_residual_terms_by_simple_root, [0; 5]);
        assert_eq!(
            report
                .domain_basis
                .iter()
                .map(|entry| entry.pbw_word_simple_roots.clone())
                .collect::<Vec<_>>(),
            vec![
                vec![1, 2, 3, 4, 5],
                vec![1, 2, 3, 4],
                vec![1, 2, 3],
                vec![1, 2],
                vec![1],
                vec![],
            ]
        );
    }

    #[test]
    fn modular_first_matches_exact_for_light_level16_fixture() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "20000" && fixture.copy == 1)
            .unwrap();
        compare_exact_and_modular_first(
            LEVEL16_PROBLEM,
            fixture.dynkin_label,
            fixture.copy,
            2,
            fixture.bytes,
        );
    }

    #[test]
    fn modular_first_matches_exact_for_first_momentum_correction_fixture() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "00000" && fixture.copy == 1)
            .unwrap();
        compare_exact_and_modular_first(
            FIRST_MOMENTUM_00001_PROBLEM,
            fixture.dynkin_label,
            fixture.copy,
            fixture_coefficient_width(fixture.artifact),
            fixture.bytes,
        );
    }

    #[test]
    #[ignore = "proof comparison for the heavy 01002 correction fixture; run when the exact fleet is idle"]
    fn modular_first_matches_exact_for_heavy_first_momentum_fixture() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "01002" && fixture.copy == 1)
            .unwrap();
        compare_exact_and_modular_first(
            FIRST_MOMENTUM_01001_PROBLEM,
            fixture.dynkin_label,
            fixture.copy,
            fixture_coefficient_width(fixture.artifact),
            fixture.bytes,
        );
    }

    #[test]
    fn golden_gate_detects_a_primitive_coefficient_mutation() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "20000")
            .unwrap();
        let mut abstract_certificate = build_abstract("20000");
        abstract_certificate.primitive_domain_coefficients[0] += 1;
        let report = verify_embedded_with_abstract(
            LEVEL16_PROBLEM,
            &abstract_certificate,
            fixture.copy,
            fixture.artifact,
            fixture.bytes,
        );
        assert!(!report.passed);
        assert!(
            report
                .exact_raising_residual_terms_by_simple_root
                .iter()
                .any(|terms| *terms != 0)
        );
    }

    #[test]
    fn atomic_checkpoint_requires_and_preserves_a_passing_report() {
        let path = std::env::temp_dir().join(format!(
            "adinkra-level16-atomic-checkpoint-{}.json",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let passing = serde_json::json!({"schema_version": "test", "passed": true});
        write_atomic_json(&path, &passing, true).unwrap();
        let restored: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(restored["passed"], true);
        fs::remove_file(&path).unwrap();

        let failing = serde_json::json!({"schema_version": "test", "passed": false});
        assert!(write_atomic_json(&path, &failing, false).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn level17_hook_manifest_and_10001_golden_coupling_pass() {
        let precheck = verify_hook_precheck();
        assert!(precheck.passed);
        assert_eq!(precheck.distinct_source_irreps, 4);
        assert_eq!(precheck.embedded_source_copies, 7);
        let report = build_hook_abstract("10001");
        assert!(report.passed);
        assert_eq!(report.target_dynkin_label, "11000");
        assert_eq!(report.product_weight_domain_dimension, 8);
        assert_eq!(report.kernel_dimension, 1);
        assert_eq!(
            report.primitive_domain_coefficients,
            [1, -1, 1, -1, -1, 1, -1, 1]
        );
        assert_eq!(report.exact_raising_residual_terms_by_simple_root, [0; 5]);
    }

    #[test]
    fn level17_hook_golden_gate_detects_a_coefficient_mutation() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level17_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "10001")
            .unwrap();
        let mut abstract_certificate = build_hook_abstract("10001");
        abstract_certificate.primitive_domain_coefficients[0] += 1;
        let report = verify_embedded_with_abstract(
            LEVEL17_HOOK_PROBLEM,
            &abstract_certificate,
            fixture.copy,
            fixture.artifact,
            fixture.bytes,
        );
        assert!(!report.passed);
        assert!(
            report
                .exact_raising_residual_terms_by_simple_root
                .iter()
                .any(|terms| *terms != 0)
        );
    }

    #[test]
    fn first_momentum_manifest_has_twenty_three_pairs_and_forty_four_maps() {
        let manifest = first_momentum_copy_manifest();
        assert_eq!(manifest.len(), 23);
        assert_eq!(manifest.values().map(Vec::len).sum::<usize>(), 44);
        assert!(
            manifest
                .keys()
                .all(|(_, target)| ["00001", "01001", "10001", "20001"].contains(&target.as_str()))
        );
    }

    #[test]
    fn first_momentum_scalar_to_spinor_golden_coupling_passes() {
        let abstract_certificate = build_first_momentum_abstract("00000", "00001");
        assert!(abstract_certificate.passed);
        assert_eq!(abstract_certificate.kernel_dimension, 1);
        let embedded = verify_first_momentum_copy_with_abstract(&abstract_certificate, 1);
        assert!(embedded.passed);
        assert_eq!(embedded.exact_raising_residual_terms_by_simple_root, [0; 5]);
    }

    #[test]
    fn first_momentum_golden_gate_detects_a_coefficient_mutation() {
        let mut abstract_certificate = build_first_momentum_abstract("00100", "01001");
        abstract_certificate.primitive_domain_coefficients[0] += 1;
        let embedded = verify_first_momentum_copy_with_abstract(&abstract_certificate, 1);
        assert!(!embedded.passed);
        assert!(
            embedded
                .exact_raising_residual_terms_by_simple_root
                .iter()
                .any(|terms| *terms != 0)
        );
    }

    #[test]
    fn direct_hook_target_blueprint_has_eight_exact_terms() {
        let terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
        assert_eq!(terms.len(), 8);
        assert_eq!(
            terms
                .iter()
                .map(|term| term.primitive_coefficient)
                .collect::<Vec<_>>(),
            [1, -1, 1, -1, -1, 1, -1, 1]
        );
        assert!(terms.iter().all(|term| add(
            term.vector_spinor_weight,
            spinor_weights()[term.outer_spinor_index]
        ) == LEVEL17_HOOK_PROBLEM.target_weight));
    }

    #[test]
    fn scalar_factorizing_candidate_is_nonzero_in_the_direct_leading_space() {
        let (candidate, maximum) = build_scalar_factorizing_candidate();
        assert!(maximum > 0);
        assert_eq!(candidate.components.len(), 32);
        assert!(
            candidate
                .components
                .values()
                .all(|component| !component.is_empty())
        );
    }

    #[test]
    fn golden_leading_anticommutator_residual_is_nonzero() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level16_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "20000" && fixture.copy == 1)
            .unwrap();
        let abstract_certificate = build_abstract("20000");
        let hook_terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
        let translation = translation_weight_basis_coefficients();
        let (residual, maximum) = build_leading_anticommutator_residual(
            &abstract_certificate,
            fixture.bytes,
            &hook_terms,
            &translation,
        );
        assert!(!residual.is_empty());
        assert!(maximum > 0);
        assert!(
            residual
                .iter()
                .all(|entry| entry.exterior_mask.count_ones() == 15)
        );
    }

    #[test]
    fn golden_first_momentum_correction_residual_is_nonzero() {
        let fixture = crate::eleven_dimensional_spinor_bridge_kernels::level14_fixtures()
            .into_iter()
            .find(|fixture| fixture.dynkin_label == "00000" && fixture.copy == 1)
            .unwrap();
        let abstract_certificate = build_first_momentum_abstract("00000", "00001");
        let recouplings = crate::eleven_dimensional_bridge::first_momentum_recoupling_audits();
        let recoupling = recouplings
            .iter()
            .find(|audit| audit.intermediate_dynkin_label == "00001")
            .unwrap();
        let hook_terms = crate::eleven_dimensional_bridge::direct_hook_target_coupling_terms();
        let (residual, maximum) = build_first_momentum_correction_residual(
            FIRST_MOMENTUM_00001_PROBLEM,
            &abstract_certificate,
            fixture.bytes,
            fixture_coefficient_width(fixture.artifact),
            recoupling,
            &hook_terms,
        );
        assert!(!residual.is_empty());
        assert!(maximum > 0);
        assert!(
            residual
                .iter()
                .all(|entry| entry.exterior_mask.count_ones() == 15)
        );
    }

    #[test]
    fn joint_residual_functionals_are_exactly_linear() {
        let left = vec![
            MomentumHookEntry {
                momentum_vector_index: 0,
                free_spinor_index: 3,
                exterior_mask: 0x0000_7fff,
                real: 7,
                imaginary: -11,
            },
            MomentumHookEntry {
                momentum_vector_index: 10,
                free_spinor_index: 31,
                exterior_mask: 0xffff_0001,
                real: -13,
                imaginary: 17,
            },
        ];
        let right = vec![
            MomentumHookEntry {
                real: 5,
                imaginary: 19,
                ..left[0]
            },
            MomentumHookEntry {
                real: 23,
                imaginary: -29,
                ..left[1]
            },
        ];
        let sum = vec![
            MomentumHookEntry {
                real: 12,
                imaginary: 8,
                ..left[0]
            },
            MomentumHookEntry {
                real: 10,
                imaginary: -12,
                ..left[1]
            },
        ];
        let left_functionals = momentum_hook_functionals(&left);
        let right_functionals = momentum_hook_functionals(&right);
        let sum_functionals = momentum_hook_functionals(&sum);
        assert_eq!(
            sum_functionals,
            left_functionals
                .iter()
                .zip(right_functionals)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rational_coordinate_solver_and_rank_are_exact() {
        let gram = vec![
            vec![BigInt::from(2), BigInt::from(1)],
            vec![BigInt::from(1), BigInt::from(2)],
        ];
        let coordinates = solve_bigint_system(&gram, &[BigInt::from(1), BigInt::from(0)]).unwrap();
        assert_eq!(
            coordinates,
            [
                Ratio::new(BigInt::from(2), BigInt::from(3)),
                Ratio::new(BigInt::from(-1), BigInt::from(3))
            ]
        );
        let matrix = vec![
            vec![
                Ratio::from_integer(BigInt::from(1)),
                Ratio::from_integer(BigInt::from(2)),
                Ratio::from_integer(BigInt::from(3)),
            ],
            vec![
                Ratio::from_integer(BigInt::from(0)),
                Ratio::from_integer(BigInt::from(1)),
                Ratio::from_integer(BigInt::from(1)),
            ],
        ];
        assert_eq!(rational_matrix_rank(&matrix), 2);
        let kernel = rational_nullspace(&matrix);
        assert_eq!(kernel.len(), 1);
        let primitive = primitive_bigint_vector(&kernel[0]);
        assert_eq!(
            primitive,
            [BigInt::from(1), BigInt::from(1), BigInt::from(-1)]
        );
        assert!(matrix_times_integer_vector_is_zero(&matrix, &primitive));
        let mut mutated = primitive;
        mutated[0] += BigInt::one();
        assert!(!matrix_times_integer_vector_is_zero(&matrix, &mutated));
    }

    #[test]
    fn right_exterior_composition_uses_greater_index_crossings() {
        let mask = (1_u32 << 1) | (1_u32 << 4) | (1_u32 << 7);
        assert_eq!(right_wedge_sign(mask, 0), Some(-1));
        assert_eq!(right_wedge_sign(mask, 2), Some(1));
        assert_eq!(right_wedge_sign(mask, 6), Some(-1));
        assert_eq!(right_wedge_sign(mask, 8), Some(1));
        assert_eq!(right_wedge_sign(mask, 4), None);
        assert_eq!(right_wedge_sign(1_u32 << 0, 31), Some(1));
    }

    #[test]
    fn right_contraction_uses_the_indices_to_the_right() {
        let mask = (1_u32 << 1) | (1_u32 << 4) | (1_u32 << 7);
        assert_eq!(right_contraction_sign(mask, 1), Some(1));
        assert_eq!(right_contraction_sign(mask, 4), Some(-1));
        assert_eq!(right_contraction_sign(mask, 7), Some(1));
        assert_eq!(right_contraction_sign(mask, 2), None);
    }

    #[test]
    fn gaussian_integer_product_keeps_real_and_imaginary_parts_exact() {
        assert_eq!(multiply_gaussian_integers(2, 3, 5, -7), (31, 1));
    }

    fn reference_sparse_lowering_in_chunks(
        source: &CoupledSparseState64,
        root: usize,
        chunk_terms: usize,
        maximum: &mut i128,
    ) -> CoupledSparseState64 {
        assert!(chunk_terms != 0);
        let spinors = spinor_weights();
        let flattened = source
            .components
            .iter()
            .flat_map(|(&free_spinor, values)| {
                values
                    .iter()
                    .map(move |&(mask, coefficient)| (free_spinor, mask, coefficient))
            })
            .collect::<Vec<_>>();
        let mut accumulated = BTreeMap::<u64, i128>::new();
        for chunk in flattened.chunks(chunk_terms) {
            for &(free_spinor, mask, coefficient) in chunk {
                if let Some(lowered_free_spinor) = lowered_spinor_index(free_spinor, root, &spinors)
                {
                    *accumulated
                        .entry((lowered_free_spinor as u64) << 32 | u64::from(mask))
                        .or_default() += i128::from(coefficient);
                }
                let mut occupied = mask;
                while occupied != 0 {
                    let upper = occupied.trailing_zeros() as usize;
                    occupied &= occupied - 1;
                    let Some(lower) = lowered_spinor_index(upper, root, &spinors) else {
                        continue;
                    };
                    if mask & (1_u32 << lower) != 0 {
                        continue;
                    }
                    let output_mask = mask ^ (1_u32 << upper) ^ (1_u32 << lower);
                    let contribution = i128::from(coefficient)
                        * i128::from(exterior_replacement_sign(mask, upper, lower));
                    *accumulated
                        .entry((free_spinor as u64) << 32 | u64::from(output_mask))
                        .or_default() += contribution;
                }
            }
        }
        let mut components = BTreeMap::<usize, Vec<(u32, i64)>>::new();
        for (key, value) in accumulated {
            if value == 0 {
                continue;
            }
            *maximum = (*maximum).max(value.abs());
            components
                .entry((key >> 32) as usize)
                .or_default()
                .push((key as u32, i64::try_from(value).unwrap()));
        }
        CoupledSparseState64 { components }
    }

    fn synthetic_sparse_state64() -> CoupledSparseState64 {
        let components = (0..8)
            .map(|free_spinor| {
                let mut values = (0..16)
                    .flat_map(|left| ((left + 1)..16).map(move |right| (left, right)))
                    .map(|(left, right)| {
                        let mask = (1_u32 << left) | (1_u32 << right);
                        let coefficient =
                            i64::from(((free_spinor * 17 + left * 5 + right * 3) % 19) as i32 - 9);
                        (mask, coefficient)
                    })
                    .filter(|(_, coefficient)| *coefficient != 0)
                    .collect::<Vec<_>>();
                values.sort_unstable_by_key(|entry| entry.0);
                (free_spinor, values)
            })
            .collect();
        CoupledSparseState64 { components }
    }

    #[test]
    fn bounded_sparse_lowering_matches_chunked_reference_for_all_roots() {
        let source = synthetic_sparse_state64();
        for root in 0..SIMPLE_ROOTS.len() {
            let mut expected_maximum = 0;
            let expected =
                reference_sparse_lowering_in_chunks(&source, root, 7, &mut expected_maximum);
            let mut observed_maximum = 0;
            let observed =
                lower_sparse_coupled_state64_bounded(&source, root, &mut observed_maximum).unwrap();
            assert_eq!(observed.components, expected.components, "root {root}");
            assert_eq!(observed_maximum, expected_maximum, "root {root}");
        }
    }

    #[test]
    fn production_sized_sparse_lowering_routes_to_bounded_before_cuda_flattening() {
        let input_terms = 6_190_000;
        let estimated = cuda_sparse_lowering_estimated_host_bytes(input_terms).unwrap();
        assert_eq!(estimated, 1_485_600_000);
        assert!(estimated > DEFAULT_CUDA_SPARSE_LOWERING_HOST_CAP_BYTES);
    }

    #[test]
    fn bounded_sparse_lowering_preserves_cross_chunk_cancellation() {
        let root = 0;
        let spinors = spinor_weights();
        let (input_free_spinor, output_free_spinor) = (0..spinors.len())
            .find_map(|upper| {
                lowered_spinor_index(upper, root, &spinors).map(|lower| (upper, lower))
            })
            .unwrap();
        let exterior_upper = input_free_spinor;
        let exterior_lower = output_free_spinor;
        let extra = (0..32)
            .find(|index| *index != exterior_upper && *index != exterior_lower)
            .unwrap();
        let output_mask = (1_u32 << exterior_lower) | (1_u32 << extra);
        let replacement_input_mask =
            output_mask ^ (1_u32 << exterior_upper) ^ (1_u32 << exterior_lower);
        let identity_coefficient = 37_i64;
        let replacement_sign =
            exterior_replacement_sign(replacement_input_mask, exterior_upper, exterior_lower);
        let replacement_coefficient = -identity_coefficient * replacement_sign;
        let mut source = CoupledSparseState64 {
            components: BTreeMap::from([
                (input_free_spinor, vec![(output_mask, identity_coefficient)]),
                (
                    output_free_spinor,
                    vec![(replacement_input_mask, replacement_coefficient)],
                ),
            ]),
        };
        for values in source.components.values_mut() {
            values.sort_unstable_by_key(|entry| entry.0);
        }

        let mut expected_maximum = 0;
        let expected = reference_sparse_lowering_in_chunks(&source, root, 1, &mut expected_maximum);
        let mut observed_maximum = 0;
        let observed =
            lower_sparse_coupled_state64_bounded(&source, root, &mut observed_maximum).unwrap();
        assert_eq!(observed.components, expected.components);
        assert_eq!(observed_maximum, expected_maximum);
        assert!(
            observed
                .components
                .get(&output_free_spinor)
                .into_iter()
                .flatten()
                .all(|(mask, _)| *mask != output_mask)
        );
    }

    #[test]
    fn lcp_word_traversal_matches_independent_reference_and_preserves_ordinals() {
        let highest = synthetic_sparse_state64();
        let words = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 5],
            vec![1, 2, 4],
            vec![1, 5],
        ];
        let mut expected = Vec::new();
        let mut expected_maximum = 0;
        for (ordinal, word) in words.iter().enumerate() {
            let mut state = highest.clone();
            for &simple_root in word {
                state = reference_sparse_lowering_in_chunks(
                    &state,
                    usize::from(simple_root - 1),
                    5,
                    &mut expected_maximum,
                );
            }
            expected.push((ordinal, state.components));
        }

        let mut observed = Vec::new();
        let mut observed_maximum = 0;
        let mut lowered_root_segments = Vec::<Vec<u8>>::new();
        visit_independent_coupled_word_handles(
            &highest,
            &words,
            &mut observed_maximum,
            &mut |source, roots, maximum| {
                lowered_root_segments.push(roots.to_vec());
                lower_sparse_coupled_root_word64(source, roots, maximum)
            },
            &mut |ordinal, state| {
                observed.push((ordinal, state.components.clone()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(observed, expected);
        assert_eq!(observed_maximum, expected_maximum);
        assert_eq!(
            observed
                .iter()
                .map(|(ordinal, _)| *ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        let optimized_root_count = lowered_root_segments.iter().map(Vec::len).sum::<usize>();
        let independent_root_count = words.iter().map(Vec::len).sum::<usize>();
        assert!(optimized_root_count < independent_root_count);
    }

    #[test]
    fn lcp_word_traversal_accepts_non_clone_opaque_handles_and_unsorted_words() {
        struct OpaqueHandle {
            roots: Vec<u8>,
        }

        let highest = OpaqueHandle { roots: Vec::new() };
        let words = vec![vec![3, 1, 4], vec![3, 1, 5], vec![2, 5], vec![3, 2]];
        let mut visited = Vec::new();
        visit_independent_coupled_word_handles(
            &highest,
            &words,
            &mut 0,
            &mut |source, suffix, _| {
                let mut roots = source.roots.clone();
                roots.extend_from_slice(suffix);
                Ok(OpaqueHandle { roots })
            },
            &mut |ordinal, terminal| {
                visited.push((ordinal, terminal.roots.clone()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(visited, words.into_iter().enumerate().collect::<Vec<_>>());
    }

    #[test]
    fn canonical_sparse_highest_visits_strict_keys_and_rejects_bad_invariants() {
        let first_mask = (1_u32 << 12) - 1;
        let second_mask = first_mask << 1;
        let highest = CanonicalSparseHighest64 {
            state: CoupledSparseState64 {
                components: BTreeMap::from([(2, vec![(first_mask, -3), (second_mask, 5)])]),
            },
            maximum_absolute_coefficient: 5,
        };
        let mut terms = Vec::new();
        highest
            .visit_terms(|key, coefficient| {
                terms.push((key, coefficient));
                Ok(())
            })
            .unwrap();
        assert_eq!(highest.term_count(), 2);
        assert_eq!(highest.maximum_absolute_coefficient(), 5);
        assert_eq!(
            terms,
            vec![
                ((2_u64 << 32) | u64::from(first_mask), -3),
                ((2_u64 << 32) | u64::from(second_mask), 5),
            ]
        );

        let wrong_maximum = CanonicalSparseHighest64 {
            state: highest.state.clone(),
            maximum_absolute_coefficient: 4,
        };
        assert_eq!(
            wrong_maximum.visit_terms(|_, _| Ok(())).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let reversed = CanonicalSparseHighest64 {
            state: CoupledSparseState64 {
                components: BTreeMap::from([(2, vec![(second_mask, 5), (first_mask, -3)])]),
            },
            maximum_absolute_coefficient: 5,
        };
        assert_eq!(
            reversed.visit_terms(|_, _| Ok(())).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn resumed_word_events_match_uninterrupted_suffix_without_gaps() {
        struct OpaqueHandle {
            roots: Vec<u8>,
        }
        #[derive(Debug, PartialEq, Eq)]
        enum ObservedEvent {
            LoweringStart(usize, Vec<u8>),
            Start(usize),
            State(usize, Vec<u8>),
            End(usize),
        }
        fn collect(words: &[Vec<u8>], start_word_ordinal: usize) -> io::Result<Vec<ObservedEvent>> {
            collect_range(words, start_word_ordinal, words.len())
        }
        fn collect_range(
            words: &[Vec<u8>],
            start_word_ordinal: usize,
            end_word_ordinal_exclusive: usize,
        ) -> io::Result<Vec<ObservedEvent>> {
            let highest = OpaqueHandle { roots: Vec::new() };
            let mut observed = Vec::new();
            visit_independent_coupled_word_handle_events_range(
                &highest,
                words,
                start_word_ordinal,
                end_word_ordinal_exclusive,
                &mut 0,
                &mut |source, suffix, _| {
                    let mut roots = source.roots.clone();
                    roots.extend_from_slice(suffix);
                    Ok(OpaqueHandle { roots })
                },
                &mut |event| {
                    observed.push(match event {
                        CoupledWordStateEvent::WordLoweringStart { ordinal, pbw_word } => {
                            assert_eq!(pbw_word, words[ordinal]);
                            ObservedEvent::LoweringStart(ordinal, pbw_word.to_vec())
                        }
                        CoupledWordStateEvent::WordStart { ordinal, pbw_word } => {
                            assert_eq!(pbw_word, words[ordinal]);
                            ObservedEvent::Start(ordinal)
                        }
                        CoupledWordStateEvent::State { ordinal, state } => {
                            ObservedEvent::State(ordinal, state.roots.clone())
                        }
                        CoupledWordStateEvent::WordEnd { ordinal } => ObservedEvent::End(ordinal),
                    });
                    Ok(())
                },
            )?;
            Ok(observed)
        }

        let words = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 5],
            vec![1, 2, 4],
            vec![3, 1],
        ];
        let uninterrupted = collect(&words, 0).unwrap();
        assert_eq!(collect_range(&words, 1, 3).unwrap(), uninterrupted[4..12]);
        let resumed = collect(&words, 2).unwrap();
        assert_eq!(resumed, uninterrupted[8..]);
        assert_eq!(
            resumed,
            vec![
                ObservedEvent::LoweringStart(2, words[2].clone()),
                ObservedEvent::Start(2),
                ObservedEvent::State(2, words[2].clone()),
                ObservedEvent::End(2),
                ObservedEvent::LoweringStart(3, words[3].clone()),
                ObservedEvent::Start(3),
                ObservedEvent::State(3, words[3].clone()),
                ObservedEvent::End(3),
            ]
        );
        assert!(collect(&words, words.len()).unwrap().is_empty());
        let error = collect(&words, words.len() + 1).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            collect_range(&words, 3, 2).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn lowering_failure_emits_only_the_pre_lowering_boundary() {
        struct OpaqueHandle;
        let timeline = std::cell::RefCell::new(Vec::new());
        let error = visit_independent_coupled_word_handle_events_from(
            &OpaqueHandle,
            &[vec![1, 2]],
            0,
            &mut 0,
            &mut |_, _, _| {
                timeline.borrow_mut().push("backend_lower");
                Err(io::Error::other("injected lowering failure"))
            },
            &mut |event| {
                timeline.borrow_mut().push(match event {
                    CoupledWordStateEvent::WordLoweringStart { .. } => "lowering_start",
                    CoupledWordStateEvent::WordStart { .. } => "word_start",
                    CoupledWordStateEvent::State { .. } => "state",
                    CoupledWordStateEvent::WordEnd { .. } => "word_end",
                });
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(&*timeline.borrow(), &["lowering_start", "backend_lower"]);
    }

    #[test]
    fn every_backend_lower_is_inside_its_word_event_interval() {
        #[derive(Clone)]
        struct OpaqueHandle {
            roots: Vec<u8>,
        }
        let active_word = std::cell::Cell::new(false);
        let words = vec![vec![1, 2, 3, 4], vec![1, 2, 5], vec![1, 4]];
        visit_independent_coupled_word_handle_events_from(
            &OpaqueHandle { roots: Vec::new() },
            &words,
            0,
            &mut 0,
            &mut |source, suffix, _| {
                assert!(
                    active_word.get(),
                    "backend lowering preceded WordLoweringStart"
                );
                let mut roots = source.roots.clone();
                roots.extend_from_slice(suffix);
                Ok(OpaqueHandle { roots })
            },
            &mut |event| {
                match event {
                    CoupledWordStateEvent::WordLoweringStart { .. } => {
                        assert!(!active_word.replace(true));
                    }
                    CoupledWordStateEvent::WordStart { .. }
                    | CoupledWordStateEvent::State { .. } => assert!(active_word.get()),
                    CoupledWordStateEvent::WordEnd { .. } => {
                        assert!(active_word.replace(false));
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(!active_word.get());
    }

    #[test]
    fn sparse_coupled_lowering_matches_dense_csr_action() {
        let mut model = ExteriorModel::new(2);
        let mask = (1_u32 << 0) | (1_u32 << 7);
        let weight = add(model.spinors[0], model.spinors[7]);
        let ordinal = model.space(weight).index[&mask];
        let mut coefficients = vec![0_i64; model.space(weight).masks.len()];
        coefficients[ordinal] = -17;
        let dense = CoupledDenseState {
            total_weight: add(weight, model.spinors[3]),
            components: BTreeMap::from([(
                3,
                DenseState {
                    weight,
                    pbw_word: Vec::new(),
                    coefficients,
                },
            )]),
        };
        let sparse = dense_coupled_to_sparse64(&mut model, &dense);
        for root in 0..5 {
            let mut dense_maximum = 0_i128;
            let dense_lowered = lower_coupled_state(&mut model, &dense, root, &mut dense_maximum);
            let expected = dense_coupled_to_sparse64(&mut model, &dense_lowered);
            let mut sparse_maximum = 0_i128;
            let observed =
                lower_sparse_coupled_state64(&sparse, root, &mut sparse_maximum).unwrap();
            assert_eq!(observed.components, expected.components, "root {root}");
            assert_eq!(sparse_maximum, dense_maximum, "root {root}");
        }
    }
}
