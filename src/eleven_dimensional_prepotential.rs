//! SO(11) decompositions of the eleven-dimensional prepotential candidates.
//!
//! Appendix F of arXiv:2002.08502 lists levels zero through sixteen. Levels
//! seventeen through thirty-two follow by Hodge duality. This module parses
//! that published inventory, computes every B5 Weyl dimension with rational
//! arithmetic, and checks each level against `binomial(32, level)`. It then
//! derives the spinor-superfield inventory and checks the six Table 5 counts.

use num_rational::Ratio;
use serde::Serialize;
use std::collections::BTreeMap;

const LOWER_LEVEL_FIXTURES: [&str; 17] = [
    "00000",
    "00001",
    "00000 00100 00010",
    "00001 01001 00101",
    "00000 00100 00010 02000 10002 01100 00200 01002",
    "00001 01001 00101 00003 11001 02001 10101 10003 01101",
    "00000 00100 00010 02000 10002 01100 20100 00200 20010 2*01002 00004 11100 00102 02100 11010 11002 02010 10102",
    "00001 01001 00101 00003 30001 11001 02001 2*10101 10003 10011 21001 2*01101 20101 01003 12001 01011 20011 03001 00103 11101 11011",
    "00000 00100 00010 40000 02000 10002 01100 20100 31000 2*00200 2*20010 2*01002 20002 00110 00020 00004 22000 11100 00102 10200 30010 02100 13000 2*11010 30002 04000 2*11002 2*02010 10110 10020 20200 02002 2*10102 21010 10012 21002 20110 01102 20020 12010 01012 12002",
    "00001 01001 00101 00003 2*30001 11001 40001 02001 3*10101 10003 2*10011 2*21001 3*01101 00201 2*20101 01003 2*12001 31001 2*01011 20003 3*20011 00021 2*03001 00103 00111 30101 22001 2*11101 30011 10201 11003 3*11011 13001 10021 02003 10111 02011 21101 21011",
    "00000 00100 00010 40000 02000 10002 01100 2*20100 31000 2*00200 3*20010 3*01002 20002 00110 00020 30100 00004 22000 2*11100 2*00102 10200 2*30010 2*02100 13000 3*11010 40100 2*30002 04000 3*11002 21100 3*02010 2*10110 40010 00300 2*10020 20200 02002 4*10102 2*21010 2*10012 12100 31100 3*21002 01110 01020 00210 00030 2*20110 2*01102 2*20020 2*12010 00120 31010 2*01012 2*20102 22100 2*12002 31002 20012 03002 11110 22010 11020 30102 11102 11012",
    "00001 01001 00101 2*00003 2*30001 2*11001 40001 2*02001 4*10101 2*10003 2*10011 3*21001 4*01101 00201 4*20101 2*01003 3*12001 2*31001 3*01011 2*20003 4*20011 00021 2*03001 2*00103 2*00111 41001 3*30101 2*22001 4*11101 30003 2*30011 2*10201 2*11003 5*11011 02101 13001 40101 32001 2*10021 02003 10103 40003 3*10111 2*02011 3*21101 01201 21003 3*21011 01111 01021 12101 31101 20103 20111 12011",
    "00000 00100 00010 40000 2*02000 2*10002 2*01100 2*20100 12000 31000 3*00200 3*20010 4*01002 2*20002 00110 00020 30100 2*00004 2*22000 3*11100 2*00102 00012 2*10200 2*30010 2*02100 13000 4*11010 40100 3*30002 32000 04000 5*11002 2*21100 4*02010 3*10110 40010 00300 2*10020 10004 42000 3*20200 40002 2*02002 5*10102 3*21010 3*10012 12100 2*31100 5*21002 2*01110 01020 00210 00030 50002 4*20110 30200 4*01102 3*20020 11200 3*12010 00120 2*31010 00202 20004 41100 3*01012 3*20102 22100 3*12002 3*31002 02200 2*20012 00112 40200 03002 30110 3*11110 2*22010 2*11020 30004 2*30102 41002 3*11102 22002 02110 30012 02020 10202 2*11012 21110 10112 21102",
    "00001 2*01001 2*00101 2*00003 2*30001 3*11001 40001 3*02001 5*10101 3*10003 3*10011 4*21001 5*01101 2*00201 5*20101 3*01003 4*12001 3*31001 4*01011 3*20003 5*20011 00021 2*03001 3*00103 2*00111 00013 2*41001 4*30101 3*22001 6*11101 2*30003 3*30011 4*10201 3*11003 7*11011 2*02101 51001 13001 2*40101 2*32001 2*10021 2*02003 2*10103 2*40003 4*10111 3*02011 5*21101 10013 2*01201 40011 2*21003 50101 2*20201 5*21011 01103 2*01111 01021 2*12101 00203 2*31101 2*20103 2*20111 20013 2*12011 31003 30201 11201 31011 11111",
    "00000 2*00100 2*00010 40000 2*02000 10100 10010 2*10002 2*01100 3*20100 12000 31000 3*00200 01010 4*20010 5*01002 2*20002 2*00110 00020 2*30100 2*00004 2*22000 3*11100 3*00102 00012 2*10200 3*30010 3*02100 13000 5*11010 2*40100 3*30002 32000 04000 6*11002 2*21100 01200 5*02010 4*10110 2*40010 2*00300 2*10020 50100 10004 42000 3*20200 40002 2*02002 7*10102 4*21010 4*10012 2*12100 2*31100 50010 6*21002 3*01110 60100 01020 2*00210 00030 50002 5*20110 01004 30200 5*01102 3*20020 2*11200 4*12010 00120 3*31010 00202 20004 60010 10300 41100 4*01012 5*20102 2*22100 4*12002 00104 4*31002 02200 3*20012 00112 40200 2*03002 2*30110 41010 4*11110 21200 3*22010 2*11020 30004 10210 4*30102 2*41002 20300 5*11102 10030 22002 02110 2*30012 02020 2*10202 40110 3*11012 02102 31200 2*21110 40102 10112 20210 01202 2*21102 21012",
    "2*00001 10001 2*01001 20001 3*00101 2*00003 00011 3*30001 3*11001 2*40001 3*02001 6*10101 3*10003 4*10011 4*21001 50001 6*01101 3*00201 6*20101 3*01003 4*12001 3*31001 5*01011 60001 3*20003 6*20011 00021 3*03001 3*00103 3*00111 00013 70001 2*41001 5*30101 3*22001 7*11101 2*30003 4*30011 5*10201 4*11003 8*11011 3*02101 51001 10005 2*13001 3*40101 2*32001 2*10021 3*02003 3*10103 2*40003 5*10111 3*02011 6*21101 10013 3*01201 2*40011 00301 3*21003 2*50101 3*20201 01005 6*21011 2*01103 2*01111 01021 3*12101 50011 00203 3*31101 3*20103 3*20111 12003 20013 2*12011 31003 2*30201 2*11201 2*31011 10301 30111 11103 11111",
    "2*00000 10000 20000 2*00100 30000 2*00010 00002 2*40000 2*02000 10100 50000 10010 3*10002 2*01100 60000 3*20100 12000 31000 4*00200 01010 4*20010 70000 5*01002 3*20002 3*00110 2*00020 2*30100 2*00004 2*22000 3*11100 80000 3*00102 00012 3*10200 3*30010 3*02100 13000 5*11010 2*40100 4*30002 32000 2*04000 6*11002 2*21100 01200 5*02010 5*10110 2*40010 2*00300 3*10020 50100 10004 42000 4*20200 2*40002 3*02002 7*10102 4*21010 4*10012 2*12100 2*31100 50010 6*21002 3*01110 60100 01020 03100 2*00210 00006 00030 2*50002 6*20110 01004 2*30200 6*01102 4*20020 2*11200 4*12010 00120 3*31010 2*00202 2*20004 60010 10300 41100 4*01012 5*20102 2*22100 5*12002 00104 4*31002 2*02200 60002 3*20012 00112 2*40200 2*03002 00400 3*30110 01300 30020 41010 4*11110 21200 3*22010 2*11020 30004 10210 4*30102 2*41002 20300 6*11102 2*10030 2*22002 02110 2*30012 02020 3*10202 2*40110 3*11012 40020 02102 02004 31200 2*21110 40102 10112 20210 01202 3*21102 20202 21012",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct B5Dynkin {
    pub labels: [u8; 5],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InventoryTerm {
    dynkin: B5Dynkin,
    multiplicity: usize,
    dimension: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalLevelCheck {
    pub level: usize,
    pub statistics: &'static str,
    pub source_level: usize,
    pub distinct_irreps: usize,
    pub irreducible_fields_with_multiplicity: usize,
    pub component_dimension: u64,
    pub expected_exterior_power_dimension: u64,
    pub matches_exterior_power: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpinorPrepotentialLevelCheck {
    pub level: usize,
    pub statistics: &'static str,
    pub distinct_irreps: usize,
    pub irreducible_fields_with_multiplicity: usize,
    pub component_dimension: u64,
    pub expected_spinor_tensor_dimension: u64,
    pub matches_spinor_tensor_dimension: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DistinguishedRepresentationCheck {
    pub level: usize,
    pub dynkin_label: &'static str,
    pub dimension: u64,
    pub multiplicity: usize,
    pub interpretation: &'static str,
    pub expected_multiplicity: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SemiPrepotentialBridgeChannel {
    pub dynkin_label: &'static str,
    pub dimension: u64,
    pub interpretation: &'static str,
    pub source_level_15_multiplicity: usize,
    pub target_vector_spinor_multiplicity: usize,
    pub equivariant_symbol_channels: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpinorGaugeParameterChannel {
    pub form_degree: usize,
    pub dynkin_label: &'static str,
    pub dimension: u64,
    pub multiplicity_in_spinor_square: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElevenDimensionalScalarReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_location: &'static str,
    pub lorentz_algebra: &'static str,
    pub superfield: &'static str,
    pub spinor_coordinates: usize,
    pub levels_checked: usize,
    pub source_levels_transcribed: usize,
    pub upper_levels_generated_by_duality: usize,
    pub level_checks: Vec<ElevenDimensionalLevelCheck>,
    pub bosonic_irreducible_fields: usize,
    pub published_bosonic_irreducible_fields: usize,
    pub fermionic_irreducible_fields: usize,
    pub published_fermionic_irreducible_fields: usize,
    pub bosonic_component_dimension: u64,
    pub fermionic_component_dimension: u64,
    pub expected_parity_component_dimension: u64,
    pub mirror_symmetry_residuals: usize,
    pub distinguished_representations: Vec<DistinguishedRepresentationCheck>,
    pub spinor_prepotential_relation: &'static str,
    pub spinor_tensor_rule: &'static str,
    pub spinor_level_checks: Vec<SpinorPrepotentialLevelCheck>,
    pub spinor_tensor_product_dimension_residuals: usize,
    pub spinor_bosonic_irreducible_fields: usize,
    pub spinor_fermionic_irreducible_fields: usize,
    pub spinor_bosonic_component_dimension: u64,
    pub spinor_fermionic_component_dimension: u64,
    pub expected_spinor_parity_component_dimension: u64,
    pub spinor_candidate_representations: Vec<DistinguishedRepresentationCheck>,
    pub spinor_every_level_matches_tensor_dimension: bool,
    pub spinor_parity_dimensions_match: bool,
    pub spinor_table_5_checks_pass: bool,
    pub semi_prepotential_source_arxiv: &'static str,
    pub semi_prepotential_source_statement: &'static str,
    pub scalar_to_vector_spinor_derivative_order: usize,
    pub formal_spinor_to_vector_spinor_composite_derivative_order: usize,
    pub vector_spinor_dimension: u64,
    pub semi_prepotential_bridge_channels: Vec<SemiPrepotentialBridgeChannel>,
    pub semi_prepotential_equivariant_symbol_dimension: usize,
    pub semi_prepotential_target_content_present: bool,
    pub gauge_rule_source_arxiv: &'static str,
    pub gauge_rule_source_statement: &'static str,
    pub spinor_gauge_parameter_channels: Vec<SpinorGaugeParameterChannel>,
    pub spinor_gauge_parameter_channel_count: usize,
    pub spinor_gauge_parameter_dimensions_sum: u64,
    pub cited_sources_print_gauge_channel_selection: bool,
    pub every_level_matches_exterior_power: bool,
    pub aggregate_counts_match_publication: bool,
    pub parity_dimensions_match: bool,
    pub distinguished_representation_checks_pass: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

fn parse_dynkin(label: &str) -> B5Dynkin {
    assert_eq!(label.len(), 5, "B5 labels must contain five digits");
    let mut labels = [0_u8; 5];
    for (index, byte) in label.bytes().enumerate() {
        assert!(byte.is_ascii_digit(), "invalid B5 Dynkin label: {label}");
        labels[index] = byte - b'0';
    }
    B5Dynkin { labels }
}

fn parse_fixture(level: usize) -> Vec<InventoryTerm> {
    LOWER_LEVEL_FIXTURES[level]
        .split_whitespace()
        .map(|token| {
            let (multiplicity, label) = token
                .split_once('*')
                .map(|(multiplicity, label)| (multiplicity.parse::<usize>().unwrap(), label))
                .unwrap_or((1, token));
            let dynkin = parse_dynkin(label);
            InventoryTerm {
                dynkin,
                multiplicity,
                dimension: b5_weyl_dimension(dynkin),
            }
        })
        .collect()
}

/// Weyl dimension for B5 = so(11), evaluated over all positive roots.
pub fn b5_weyl_dimension(dynkin: B5Dynkin) -> u64 {
    let a = dynkin.labels.map(i128::from);
    // Coordinates are doubled so the spinor fundamental weight is integral.
    let rho = [9_i128, 7, 5, 3, 1];
    let mut lambda_plus_rho = [0_i128; 5];
    for index in 0..5 {
        lambda_plus_rho[index] = 2 * a[index..4].iter().sum::<i128>() + a[4] + rho[index];
    }

    let mut dimension = Ratio::from_integer(1_i128);
    for index in 0..5 {
        dimension *= Ratio::new(lambda_plus_rho[index], rho[index]);
    }
    for left in 0..5 {
        for right in (left + 1)..5 {
            dimension *= Ratio::new(
                lambda_plus_rho[left] - lambda_plus_rho[right],
                rho[left] - rho[right],
            );
            dimension *= Ratio::new(
                lambda_plus_rho[left] + lambda_plus_rho[right],
                rho[left] + rho[right],
            );
        }
    }
    assert_eq!(*dimension.denom(), 1, "Weyl dimension was not integral");
    u64::try_from(*dimension.numer()).unwrap()
}

fn binomial(n: usize, k: usize) -> u64 {
    let k = k.min(n - k);
    (0..k).fold(1_u64, |value, index| {
        value * u64::try_from(n - index).unwrap() / u64::try_from(index + 1).unwrap()
    })
}

fn source_level(level: usize) -> usize {
    level.min(32 - level)
}

fn inventory(level: usize) -> Vec<InventoryTerm> {
    parse_fixture(source_level(level))
}

fn multiplicity(level: usize, label: &str) -> usize {
    let target = parse_dynkin(label);
    inventory(level)
        .iter()
        .filter(|term| term.dynkin == target)
        .map(|term| term.multiplicity)
        .sum()
}

pub fn level_multiplicity(level: usize, label: &str) -> usize {
    multiplicity(level, label)
}

/// Decompose a B5 irrep tensored with the 32-dimensional minuscule spinor.
/// Each dominant `lambda + weight` occurs once.
fn tensor_with_spinor(dynkin: B5Dynkin) -> Vec<B5Dynkin> {
    let a = dynkin.labels.map(i16::from);
    let mut lambda = [0_i16; 5];
    for index in 0..5 {
        lambda[index] = 2 * a[index..4].iter().sum::<i16>() + a[4];
    }

    let mut outputs = Vec::new();
    for mask in 0_u8..32 {
        let mut weight = [0_i16; 5];
        for index in 0..5 {
            weight[index] = lambda[index] + if mask & (1 << index) == 0 { 1 } else { -1 };
        }
        let dominant = (0..4).all(|index| weight[index] >= weight[index + 1]) && weight[4] >= 0;
        let integral = (0..4).all(|index| (weight[index] - weight[index + 1]) % 2 == 0);
        if !dominant || !integral {
            continue;
        }
        let mut labels = [0_u8; 5];
        for index in 0..4 {
            labels[index] = u8::try_from((weight[index] - weight[index + 1]) / 2).unwrap();
        }
        labels[4] = u8::try_from(weight[4]).unwrap();
        outputs.push(B5Dynkin { labels });
    }
    outputs.sort();
    outputs.dedup();
    outputs
}

pub fn spinor_tensor_channels(label: &str) -> Vec<(String, u64)> {
    tensor_with_spinor(parse_dynkin(label))
        .into_iter()
        .map(|dynkin| {
            let label = dynkin.labels.iter().map(u8::to_string).collect::<String>();
            (label, b5_weyl_dimension(dynkin))
        })
        .collect()
}

/// Decompose `10000 x 10001` by the representation-ring identity
///
/// `V x (V x S - S) = (20000 + 01000 + 00000) x S - V x S`.
///
/// Here `S=00001`, `V=10000`, and `10001` is the gamma-traceless
/// vector-spinor. Each tensor product with `S` is resolved by the exact
/// minuscule-spinor rule above.
pub fn vector_tensor_gamma_traceless_vector_spinor_channels() -> Vec<(String, u64, usize)> {
    let mut multiplicities = BTreeMap::<B5Dynkin, i64>::new();
    for label in ["20000", "01000", "00000"] {
        for dynkin in tensor_with_spinor(parse_dynkin(label)) {
            *multiplicities.entry(dynkin).or_default() += 1;
        }
    }
    for dynkin in tensor_with_spinor(parse_dynkin("10000")) {
        *multiplicities.entry(dynkin).or_default() -= 1;
    }
    multiplicities
        .into_iter()
        .filter(|(_, multiplicity)| *multiplicity > 0)
        .map(|(dynkin, multiplicity)| {
            let label = dynkin.labels.iter().map(u8::to_string).collect::<String>();
            (
                label,
                b5_weyl_dimension(dynkin),
                usize::try_from(multiplicity).unwrap(),
            )
        })
        .collect()
}

fn spinor_inventory(level: usize) -> Vec<InventoryTerm> {
    let mut multiplicities = BTreeMap::<B5Dynkin, usize>::new();
    for term in inventory(level) {
        for output in tensor_with_spinor(term.dynkin) {
            *multiplicities.entry(output).or_default() += term.multiplicity;
        }
    }
    multiplicities
        .into_iter()
        .map(|(dynkin, multiplicity)| InventoryTerm {
            dynkin,
            multiplicity,
            dimension: b5_weyl_dimension(dynkin),
        })
        .collect()
}

fn spinor_multiplicity(level: usize, label: &str) -> usize {
    let target = parse_dynkin(label);
    spinor_inventory(level)
        .iter()
        .find(|term| term.dynkin == target)
        .map(|term| term.multiplicity)
        .unwrap_or(0)
}

pub fn verify() -> ElevenDimensionalScalarReport {
    let level_checks: Vec<_> = (0..=32)
        .map(|level| {
            let terms = inventory(level);
            let component_dimension = terms
                .iter()
                .map(|term| term.dimension * u64::try_from(term.multiplicity).unwrap())
                .sum();
            let expected = binomial(32, level);
            ElevenDimensionalLevelCheck {
                level,
                statistics: if level % 2 == 0 {
                    "bosonic"
                } else {
                    "fermionic"
                },
                source_level: source_level(level),
                distinct_irreps: terms.len(),
                irreducible_fields_with_multiplicity: terms
                    .iter()
                    .map(|term| term.multiplicity)
                    .sum(),
                component_dimension,
                expected_exterior_power_dimension: expected,
                matches_exterior_power: component_dimension == expected,
            }
        })
        .collect();

    let bosonic_irreducible_fields = level_checks
        .iter()
        .filter(|check| check.level % 2 == 0)
        .map(|check| check.irreducible_fields_with_multiplicity)
        .sum();
    let fermionic_irreducible_fields = level_checks
        .iter()
        .filter(|check| check.level % 2 == 1)
        .map(|check| check.irreducible_fields_with_multiplicity)
        .sum();
    let bosonic_component_dimension = level_checks
        .iter()
        .filter(|check| check.level % 2 == 0)
        .map(|check| check.component_dimension)
        .sum();
    let fermionic_component_dimension = level_checks
        .iter()
        .filter(|check| check.level % 2 == 1)
        .map(|check| check.component_dimension)
        .sum();

    let mirror_symmetry_residuals = (0..=16)
        .filter(|&level| inventory(level) != inventory(32 - level))
        .count();

    let distinguished_representations = vec![
        DistinguishedRepresentationCheck {
            level: 16,
            dynkin_label: "20000",
            dimension: b5_weyl_dimension(parse_dynkin("20000")),
            multiplicity: multiplicity(16, "20000"),
            interpretation: "conformal graviton",
            expected_multiplicity: 1,
            passed: multiplicity(16, "20000") == 1,
        },
        DistinguishedRepresentationCheck {
            level: 16,
            dynkin_label: "00100",
            dimension: b5_weyl_dimension(parse_dynkin("00100")),
            multiplicity: multiplicity(16, "00100"),
            interpretation: "three-form",
            expected_multiplicity: 2,
            passed: multiplicity(16, "00100") == 2,
        },
        DistinguishedRepresentationCheck {
            level: 15,
            dynkin_label: "10001",
            dimension: b5_weyl_dimension(parse_dynkin("10001")),
            multiplicity: multiplicity(15, "10001"),
            interpretation: "conformal gravitino",
            expected_multiplicity: 1,
            passed: multiplicity(15, "10001") == 1,
        },
        DistinguishedRepresentationCheck {
            level: 17,
            dynkin_label: "10001",
            dimension: b5_weyl_dimension(parse_dynkin("10001")),
            multiplicity: multiplicity(17, "10001"),
            interpretation: "dual conformal gravitino",
            expected_multiplicity: 1,
            passed: multiplicity(17, "10001") == 1,
        },
        DistinguishedRepresentationCheck {
            level: 16,
            dynkin_label: "01000",
            dimension: b5_weyl_dimension(parse_dynkin("01000")),
            multiplicity: multiplicity(16, "01000"),
            interpretation: "missing two-form required by the inverse-frame decomposition",
            expected_multiplicity: 0,
            passed: multiplicity(16, "01000") == 0,
        },
    ];

    let spinor_level_checks: Vec<_> = (0..=32)
        .map(|level| {
            let terms = spinor_inventory(level);
            let component_dimension = terms
                .iter()
                .map(|term| term.dimension * u64::try_from(term.multiplicity).unwrap())
                .sum();
            let expected = 32 * binomial(32, level);
            SpinorPrepotentialLevelCheck {
                level,
                statistics: if level % 2 == 0 {
                    "fermionic"
                } else {
                    "bosonic"
                },
                distinct_irreps: terms.len(),
                irreducible_fields_with_multiplicity: terms
                    .iter()
                    .map(|term| term.multiplicity)
                    .sum(),
                component_dimension,
                expected_spinor_tensor_dimension: expected,
                matches_spinor_tensor_dimension: component_dimension == expected,
            }
        })
        .collect();

    let spinor_tensor_product_dimension_residuals = (0..=32)
        .flat_map(inventory)
        .filter(|term| {
            let output_dimension: u64 = tensor_with_spinor(term.dynkin)
                .iter()
                .map(|&output| b5_weyl_dimension(output))
                .sum();
            output_dimension != 32 * term.dimension
        })
        .count();

    let spinor_bosonic_irreducible_fields = spinor_level_checks
        .iter()
        .filter(|check| check.level % 2 == 1)
        .map(|check| check.irreducible_fields_with_multiplicity)
        .sum();
    let spinor_fermionic_irreducible_fields = spinor_level_checks
        .iter()
        .filter(|check| check.level % 2 == 0)
        .map(|check| check.irreducible_fields_with_multiplicity)
        .sum();
    let spinor_bosonic_component_dimension = spinor_level_checks
        .iter()
        .filter(|check| check.level % 2 == 1)
        .map(|check| check.component_dimension)
        .sum();
    let spinor_fermionic_component_dimension = spinor_level_checks
        .iter()
        .filter(|check| check.level % 2 == 0)
        .map(|check| check.component_dimension)
        .sum();

    let spinor_candidate_representations = [
        (17, "00000", "scalar", 2),
        (17, "01000", "two-form", 5),
        (17, "20000", "conformal graviton", 2),
        (17, "00100", "three-form", 8),
        (18, "00001", "spinor", 5),
        (18, "10001", "conformal gravitino", 8),
    ]
    .into_iter()
    .map(|(level, label, interpretation, expected)| {
        let observed = spinor_multiplicity(level, label);
        DistinguishedRepresentationCheck {
            level,
            dynkin_label: label,
            dimension: b5_weyl_dimension(parse_dynkin(label)),
            multiplicity: observed,
            interpretation,
            expected_multiplicity: expected,
            passed: observed == expected,
        }
    })
    .collect::<Vec<_>>();

    let semi_prepotential_bridge_channels = [
        ("00001", "gamma trace", 1_usize),
        ("10001", "gamma-traceless vector-spinor", 1_usize),
    ]
    .into_iter()
    .map(|(label, interpretation, target_multiplicity)| {
        let source_multiplicity = multiplicity(15, label);
        SemiPrepotentialBridgeChannel {
            dynkin_label: label,
            dimension: b5_weyl_dimension(parse_dynkin(label)),
            interpretation,
            source_level_15_multiplicity: source_multiplicity,
            target_vector_spinor_multiplicity: target_multiplicity,
            equivariant_symbol_channels: source_multiplicity * target_multiplicity,
        }
    })
    .collect::<Vec<_>>();
    let semi_prepotential_equivariant_symbol_dimension = semi_prepotential_bridge_channels
        .iter()
        .map(|channel| channel.equivariant_symbol_channels)
        .sum();
    let semi_prepotential_target_content_present = semi_prepotential_bridge_channels
        .iter()
        .all(|channel| channel.source_level_15_multiplicity > 0);

    let spinor_square = tensor_with_spinor(parse_dynkin("00001"));
    let spinor_gauge_parameter_channels = [
        (0, "00000"),
        (1, "10000"),
        (2, "01000"),
        (3, "00100"),
        (4, "00010"),
        (5, "00002"),
    ]
    .into_iter()
    .map(|(form_degree, label)| {
        let dynkin = parse_dynkin(label);
        SpinorGaugeParameterChannel {
            form_degree,
            dynkin_label: label,
            dimension: b5_weyl_dimension(dynkin),
            multiplicity_in_spinor_square: spinor_square
                .iter()
                .filter(|&&channel| channel == dynkin)
                .count(),
        }
    })
    .collect::<Vec<_>>();
    let spinor_gauge_parameter_channel_count = spinor_gauge_parameter_channels.len();
    let spinor_gauge_parameter_dimensions_sum = spinor_gauge_parameter_channels
        .iter()
        .map(|channel| {
            channel.dimension * u64::try_from(channel.multiplicity_in_spinor_square).unwrap()
        })
        .sum();
    let gauge_parameter_census_passes = spinor_gauge_parameter_channel_count == 6
        && spinor_gauge_parameter_dimensions_sum == 32 * 32
        && spinor_gauge_parameter_channels
            .iter()
            .all(|channel| channel.multiplicity_in_spinor_square == 1);

    let every_level_matches_exterior_power = level_checks
        .iter()
        .all(|check| check.matches_exterior_power);
    let aggregate_counts_match_publication =
        bosonic_irreducible_fields == 1_494 && fermionic_irreducible_fields == 1_186;
    let expected_parity_component_dimension = 1_u64 << 31;
    let parity_dimensions_match = bosonic_component_dimension
        == expected_parity_component_dimension
        && fermionic_component_dimension == expected_parity_component_dimension;
    let distinguished_representation_checks_pass = distinguished_representations
        .iter()
        .all(|check| check.passed);
    let spinor_every_level_matches_tensor_dimension = spinor_level_checks
        .iter()
        .all(|check| check.matches_spinor_tensor_dimension);
    let spinor_table_5_checks_pass = spinor_candidate_representations
        .iter()
        .all(|check| check.passed);
    let expected_spinor_parity_component_dimension = 1_u64 << 36;
    let spinor_parity_dimensions_match = spinor_bosonic_component_dimension
        == expected_spinor_parity_component_dimension
        && spinor_fermionic_component_dimension == expected_spinor_parity_component_dimension;
    let passed = every_level_matches_exterior_power
        && aggregate_counts_match_publication
        && parity_dimensions_match
        && mirror_symmetry_residuals == 0
        && distinguished_representation_checks_pass
        && spinor_every_level_matches_tensor_dimension
        && spinor_parity_dimensions_match
        && spinor_tensor_product_dimension_residuals == 0
        && spinor_table_5_checks_pass
        && semi_prepotential_target_content_present
        && semi_prepotential_equivariant_symbol_dimension == 3
        && gauge_parameter_census_passes;

    ElevenDimensionalScalarReport {
        schema_version: "adynkra-11d-prepotential-bridge-v2",
        source_arxiv: "2002.08502",
        source_location: "Appendix F, Table 4, and the note in proof with Table 5",
        lorentz_algebra: "B5 = so(11)",
        superfield: "unconstrained real scalar V(x, theta)",
        spinor_coordinates: 32,
        levels_checked: 33,
        source_levels_transcribed: 17,
        upper_levels_generated_by_duality: 16,
        level_checks,
        bosonic_irreducible_fields,
        published_bosonic_irreducible_fields: 1_494,
        fermionic_irreducible_fields,
        published_fermionic_irreducible_fields: 1_186,
        bosonic_component_dimension,
        fermionic_component_dimension,
        expected_parity_component_dimension,
        mirror_symmetry_residuals,
        distinguished_representations,
        spinor_prepotential_relation: "V = D^alpha Psi_alpha",
        spinor_tensor_rule: "each scalar-superfield level tensored with the 32-dimensional minuscule B5 spinor",
        spinor_level_checks,
        spinor_tensor_product_dimension_residuals,
        spinor_bosonic_irreducible_fields,
        spinor_fermionic_irreducible_fields,
        spinor_bosonic_component_dimension,
        spinor_fermionic_component_dimension,
        expected_spinor_parity_component_dimension,
        spinor_candidate_representations,
        spinor_every_level_matches_tensor_dimension,
        spinor_parity_dimensions_match,
        spinor_table_5_checks_pass,
        semi_prepotential_source_arxiv: "2007.05097",
        semi_prepotential_source_statement: "the discussion after Eq. (2.7) requires H_beta^c(V) to involve fifteen spinor derivatives",
        scalar_to_vector_spinor_derivative_order: 15,
        formal_spinor_to_vector_spinor_composite_derivative_order: 16,
        vector_spinor_dimension: 352,
        semi_prepotential_bridge_channels,
        semi_prepotential_equivariant_symbol_dimension,
        semi_prepotential_target_content_present,
        gauge_rule_source_arxiv: "2007.05097",
        gauge_rule_source_statement: "the introduction states that a supergravity prepotential transforms as a first spinor derivative of a Lorentz-compatible parameter superfield and identifies this as a conjectural rule in the high-dimensional cases",
        spinor_gauge_parameter_channels,
        spinor_gauge_parameter_channel_count,
        spinor_gauge_parameter_dimensions_sum,
        cited_sources_print_gauge_channel_selection: false,
        every_level_matches_exterior_power,
        aggregate_counts_match_publication,
        parity_dimensions_match,
        distinguished_representation_checks_pass,
        boundary: "the source-defined fifteen-derivative bridge has three Lorentz-equivariant leading-symbol channels, and six Lorentz-compatible first-derivative gauge-parameter channels are possible; the two cited sources do not select their coefficients or print a gauge complex, so no eleven-dimensional curvature, action, or field equation is derived",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_b5_dimensions_are_reproduced() {
        let cases = [
            ("00000", 1),
            ("10000", 11),
            ("01000", 55),
            ("00100", 165),
            ("00010", 330),
            ("00001", 32),
            ("20000", 65),
            ("10001", 320),
            ("00002", 462),
        ];
        for (label, expected) in cases {
            assert_eq!(b5_weyl_dimension(parse_dynkin(label)), expected, "{label}");
        }
    }

    #[test]
    fn every_level_matches_the_exterior_power_dimension() {
        let report = verify();
        assert_eq!(report.levels_checked, 33);
        assert!(report.every_level_matches_exterior_power);
        for check in report.level_checks {
            assert_eq!(check.component_dimension, binomial(32, check.level));
        }
    }

    #[test]
    fn published_aggregate_counts_and_parity_dimensions_are_reproduced() {
        let report = verify();
        assert_eq!(report.bosonic_irreducible_fields, 1_494);
        assert_eq!(report.fermionic_irreducible_fields, 1_186);
        assert_eq!(report.bosonic_component_dimension, 1_u64 << 31);
        assert_eq!(report.fermionic_component_dimension, 1_u64 << 31);
        assert!(report.aggregate_counts_match_publication);
        assert!(report.parity_dimensions_match);
    }

    #[test]
    fn dual_levels_have_identical_inventory() {
        let report = verify();
        assert_eq!(report.mirror_symmetry_residuals, 0);
        for level in 0..=16 {
            assert_eq!(inventory(level), inventory(32 - level));
        }
    }

    #[test]
    fn level_fifteen_and_sixteen_supergravity_candidates_are_identified() {
        let report = verify();
        assert!(report.distinguished_representation_checks_pass);
        assert!(report.passed);
    }

    #[test]
    fn spinor_square_reproduces_the_six_standard_channels() {
        let outputs = tensor_with_spinor(parse_dynkin("00001"));
        let expected = ["00000", "10000", "01000", "00100", "00010", "00002"].map(parse_dynkin);
        assert_eq!(outputs.len(), expected.len());
        for channel in expected {
            assert!(outputs.contains(&channel));
        }
        let output_dimension: u64 = outputs.into_iter().map(b5_weyl_dimension).sum();
        assert_eq!(output_dimension, 32 * 32);
    }

    #[test]
    fn vector_spinor_times_spinor_has_ten_multiplicity_free_channels() {
        let channels = spinor_tensor_channels("10001");
        assert_eq!(channels.len(), 10);
        assert_eq!(
            channels.iter().map(|(_, dimension)| dimension).sum::<u64>(),
            320 * 32
        );
        assert_eq!(
            channels
                .iter()
                .map(|(label, _)| label.as_str())
                .collect::<Vec<_>>(),
            vec![
                "00002", "00010", "00100", "01000", "10000", "10002", "10010", "10100", "11000",
                "20000"
            ]
        );
    }

    #[test]
    fn vector_times_gamma_traceless_vector_spinor_is_resolved_exactly() {
        let channels = vector_tensor_gamma_traceless_vector_spinor_channels();
        assert_eq!(
            channels
                .iter()
                .map(|(label, _, multiplicity)| (label.as_str(), *multiplicity))
                .collect::<Vec<_>>(),
            vec![("00001", 1), ("01001", 1), ("10001", 1), ("20001", 1)]
        );
        assert_eq!(
            channels
                .iter()
                .map(|(_, dimension, multiplicity)| dimension * *multiplicity as u64)
                .sum::<u64>(),
            11 * 320
        );
    }

    #[test]
    fn every_spinor_prepotential_level_matches_its_tensor_dimension() {
        let report = verify();
        assert!(report.spinor_every_level_matches_tensor_dimension);
        assert_eq!(report.spinor_tensor_product_dimension_residuals, 0);
        assert_eq!(report.spinor_bosonic_component_dimension, 1_u64 << 36);
        assert_eq!(report.spinor_fermionic_component_dimension, 1_u64 << 36);
        assert!(report.spinor_parity_dimensions_match);
        for check in report.spinor_level_checks {
            assert_eq!(check.component_dimension, 32 * binomial(32, check.level));
        }
    }

    #[test]
    fn published_spinor_prepotential_candidate_counts_are_reproduced() {
        let report = verify();
        assert!(report.spinor_table_5_checks_pass);
        assert_eq!(spinor_multiplicity(17, "00000"), 2);
        assert_eq!(spinor_multiplicity(17, "01000"), 5);
        assert_eq!(spinor_multiplicity(17, "20000"), 2);
        assert_eq!(spinor_multiplicity(17, "00100"), 8);
        assert_eq!(spinor_multiplicity(18, "00001"), 5);
        assert_eq!(spinor_multiplicity(18, "10001"), 8);
    }

    #[test]
    fn fifteen_derivative_bridge_has_three_equivariant_symbol_channels() {
        let report = verify();
        assert_eq!(multiplicity(15, "00001"), 2);
        assert_eq!(multiplicity(15, "10001"), 1);
        assert_eq!(report.vector_spinor_dimension, 32 + 320);
        assert_eq!(report.scalar_to_vector_spinor_derivative_order, 15);
        assert_eq!(
            report.formal_spinor_to_vector_spinor_composite_derivative_order,
            16
        );
        assert_eq!(report.semi_prepotential_equivariant_symbol_dimension, 3);
        assert!(report.semi_prepotential_target_content_present);
    }

    #[test]
    fn spinor_prepotential_has_six_lorentz_compatible_parameter_channels() {
        let report = verify();
        assert_eq!(report.spinor_gauge_parameter_channel_count, 6);
        assert_eq!(report.spinor_gauge_parameter_dimensions_sum, 32 * 32);
        assert!(
            report
                .spinor_gauge_parameter_channels
                .iter()
                .all(|channel| channel.multiplicity_in_spinor_square == 1)
        );
        assert!(!report.cited_sources_print_gauge_channel_selection);
    }
}
