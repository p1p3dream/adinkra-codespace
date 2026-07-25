//! Signed six-representation reproduction for the `S_8` permutahedron.
//!
//! The permutation octets and Boolean factors are Eqs. (5.1)-(5.6) of
//! arXiv:2012.14015v7.  For TT, TV, and VV, Eqs. (2.4)-(2.7) also contain the
//! integer-dependent signs `m_+`, `m_-`, `n_+`, and `n_-`; all 16 residue
//! classes `(m mod 4, n mod 4)` are evaluated.

use crate::chromochar::chromochar_support_and_q;
use crate::holoraumy::{HoloraumyData, gadget};
use crate::lr_matrix::AdinkraRep;
use crate::permutahedron::{CosetSide, Permutation, abnormal_slices, coset_partition, rana_r8};
use crate::permutahedron_fixtures::S8_REPRESENTATION_OCTETS;
use crate::permutahedron_garden::garden_solution_masks;
use crate::signed_perm::SignedPerm;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;
const SCHEMA_VERSION: &str = "permutahedron-s8-supersymmetry-v1";

/// Boolean factors in Eqs. (5.1)-(5.6), ordered CC, CT, CV, TT, TV, VV.
///
/// For TT, TV, and VV these are the factors before multiplication by the
/// height-yielding sign matrix containing `m_±` and `n_±`.
pub const S8_BASE_BOOLEAN_FACTORS: [[u8; 8]; 6] = [
    [170, 204, 102, 0, 15, 105, 195, 165],
    [234, 76, 134, 32, 11, 173, 103, 193],
    [170, 204, 6, 96, 210, 180, 126, 24],
    [238, 68, 136, 34, 238, 68, 136, 34],
    [174, 196, 8, 98, 174, 196, 8, 98],
    [170, 204, 0, 102, 170, 204, 0, 102],
];

#[derive(Debug, Clone, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub version: u8,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixRecord {
    pub color: usize,
    pub permutation: [u8; 8],
    pub base_boolean_factor: u8,
    pub effective_boolean_factor: u8,
    pub diagonal_signs: [i8; 8],
    pub l: [[i8; 8]; 8],
    pub r: [[i8; 8]; 8],
}

#[derive(Debug, Clone, Serialize)]
pub struct PairResidualRecord {
    pub colors: [usize; 2],
    pub bosonic_residual_entries: usize,
    pub fermionic_residual_entries: usize,
    pub residual_l1_norm: usize,
    pub maximum_absolute_residual: i16,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClosureRecord {
    pub sparse_garden_passed: bool,
    pub bosonic_entries_checked: usize,
    pub bosonic_residual_entries: usize,
    pub fermionic_entries_checked: usize,
    pub fermionic_residual_entries: usize,
    pub nonclosing_color_pairs: usize,
    pub residual_l1_norm: usize,
    pub maximum_absolute_residual: i16,
    pub pair_residuals: Vec<PairResidualRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeRecord {
    pub color: usize,
    pub boson: usize,
    pub fermion: usize,
    pub sign: i8,
    pub style: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SquareRecord {
    pub colors: [usize; 2],
    pub bosons: [usize; 2],
    pub fermions: [usize; 2],
    pub dashed_edges: usize,
    pub sign_product: i8,
    pub odd_dashing: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdinkraRecord {
    pub graph_type: &'static str,
    pub bosons: Vec<String>,
    pub fermions: Vec<String>,
    pub edges: Vec<EdgeRecord>,
    pub two_color_squares: Vec<SquareRecord>,
    pub edge_count: usize,
    pub square_count: usize,
    pub odd_dashing_failures: usize,
    pub valid_garden_adinkra: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HymnRecord {
    pub diagonal: bool,
    pub diagonal_entries: Vec<i16>,
    pub off_diagonal_entries: usize,
    pub trace: i16,
    pub positive_eigenvalues: usize,
    pub negative_eigenvalues: usize,
    pub zero_eigenvalues: usize,
    pub class: &'static str,
    pub equation_3_3_matched: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GardenDistanceRecord {
    pub total_garden_signings: usize,
    pub published_sign_mask: String,
    pub minimum_edge_sign_flips: u32,
    pub nearest_garden_sign_mask: String,
    pub nearest_signings_at_minimum: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParameterBranchRecord {
    pub m_mod_4: Option<u8>,
    pub n_mod_4: Option<u8>,
    pub m_plus: i8,
    pub m_minus: i8,
    pub n_plus: i8,
    pub n_minus: i8,
    pub effective_boolean_factors: [u8; 8],
    pub matrices: Vec<MatrixRecord>,
    pub closure: ClosureRecord,
    pub hymn: HymnRecord,
    pub garden_distance: GardenDistanceRecord,
    pub graph: AdinkraRecord,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnsignedCoordinateRecord {
    pub right_r8_coset_id: u32,
    pub canonical_coset_ranks: Vec<u32>,
    pub abnormal_left_right_coincident: bool,
    pub normalizer_gl_3_2_action: Option<[u8; 7]>,
    pub bruhat_row_sums: Vec<u16>,
    pub bruhat_upper_triangle: Vec<u8>,
    pub bruhat_characteristic_polynomial: Vec<i128>,
    pub bruhat_frobenius_norm_squared: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SectorRecord {
    pub id: &'static str,
    pub label: &'static str,
    pub paper_status: &'static str,
    pub permutations: [[u8; 8]; 8],
    pub base_boolean_factors: [u8; 8],
    pub unsigned_coordinates: UnsignedCoordinateRecord,
    pub parameter_branches: Vec<ParameterBranchRecord>,
    pub branch_count: usize,
    pub closure_status_uniform_across_branches: bool,
    pub hymn_class_uniform_across_branches: bool,
    pub canonical_chromochar_support: usize,
    pub canonical_chromochar_q_scaled: i128,
    pub canonical_formal_gadget_row: Vec<f64>,
    pub formal_gadget_is_representation_invariant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeparationAssessment {
    pub hymn_class_matches_published_closure_on_every_branch: bool,
    pub hymn_trace_value_sets_disjoint: bool,
    pub garden_distance_zero_matches_closure: bool,
    pub unsigned_abnormality_separates_closure: bool,
    pub bruhat_characteristic_polynomial_value_sets_disjoint: bool,
    pub chromochar_q_value_sets_disjoint: bool,
    pub formal_self_gadget_value_sets_disjoint: bool,
    pub conclusion: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub sectors: usize,
    pub source_permutation_entries_checked: usize,
    pub source_boolean_factors_checked: usize,
    pub parameter_branches_checked: usize,
    pub closing_branches: usize,
    pub nonclosing_branches: usize,
    pub dense_closure_entries_checked: usize,
    pub dense_closure_residual_entries: usize,
    pub hymn_records_checked: usize,
    pub equation_3_3_matches: usize,
    pub graphs_constructed: usize,
    pub graph_edges_checked: usize,
    pub graph_squares_checked: usize,
    pub valid_garden_adinkras: usize,
    pub signed_graphs_with_nonclosure: usize,
    pub nearest_garden_solution_spaces_scanned: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct S8SupersymmetryArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub conventions: BTreeMap<&'static str, &'static str>,
    pub sectors: Vec<SectorRecord>,
    pub canonical_formal_gadget_matrix: Vec<Vec<f64>>,
    pub separation: SeparationAssessment,
    pub validation: ValidationRecord,
    pub boundary: &'static str,
}

fn parameter_signs(residue: u8) -> (i8, i8) {
    match residue % 4 {
        0 => (1, 1),
        1 => (1, -1),
        2 => (-1, -1),
        3 => (-1, 1),
        _ => unreachable!(),
    }
}

fn diagonal(factor: u8) -> [i8; 8] {
    std::array::from_fn(|row| if factor & (1 << row) == 0 { 1 } else { -1 })
}

fn effective_factors(sector: usize, m_mod_4: u8, n_mod_4: u8) -> [u8; 8] {
    let mut factors = S8_BASE_BOOLEAN_FACTORS[sector];
    if sector < 3 {
        return factors;
    }
    let (m_plus, m_minus) = parameter_signs(m_mod_4);
    let (n_plus, n_minus) = parameter_signs(n_mod_4);
    for (color, factor) in factors.iter_mut().enumerate() {
        let (m_sign, n_sign) = if color < 4 {
            (m_plus, n_plus)
        } else {
            (m_minus, n_minus)
        };
        if n_sign == -1 {
            *factor ^= 0x0f;
        }
        if m_sign == -1 {
            *factor ^= 0xf0;
        }
    }
    factors
}

fn permutation(values: &[u8; 8]) -> Permutation {
    Permutation::new(values).expect("published S8 permutation")
}

fn build_rep(sector: usize, factors: [u8; 8]) -> AdinkraRep {
    let color_perms: Vec<Vec<usize>> = S8_REPRESENTATION_OCTETS[sector]
        .permutations
        .iter()
        .map(|entry| entry.iter().map(|&value| usize::from(value - 1)).collect())
        .collect();
    let signs: Vec<i8> = factors
        .iter()
        .flat_map(|&factor| diagonal(factor))
        .collect();
    AdinkraRep::from_parts(N, D, &color_perms, &signs)
}

fn dense_signed(signed: &SignedPerm) -> [[i8; 8]; 8] {
    let mut matrix = [[0i8; 8]; 8];
    for row in 0..8 {
        matrix[row][usize::from(signed.perm[row])] = signed.sign[row];
    }
    matrix
}

fn transpose8(matrix: [[i8; 8]; 8]) -> [[i8; 8]; 8] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn matrix_records(sector: usize, effective: [u8; 8], rep: &AdinkraRep) -> Vec<MatrixRecord> {
    rep.l_matrices
        .iter()
        .enumerate()
        .map(|(color, signed)| {
            let l = dense_signed(signed);
            MatrixRecord {
                color: color + 1,
                permutation: S8_REPRESENTATION_OCTETS[sector].permutations[color],
                base_boolean_factor: S8_BASE_BOOLEAN_FACTORS[sector][color],
                effective_boolean_factor: effective[color],
                diagonal_signs: diagonal(effective[color]),
                l,
                r: transpose8(l),
            }
        })
        .collect()
}

fn multiply8(left: &[[i8; 8]; 8], right: &[[i8; 8]; 8]) -> [[i16; 8]; 8] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..8)
                .map(|inner| i16::from(left[row][inner]) * i16::from(right[inner][column]))
                .sum()
        })
    })
}

fn closure_record(rep: &AdinkraRep) -> ClosureRecord {
    let l: Vec<_> = rep.l_matrices.iter().map(dense_signed).collect();
    let r: Vec<_> = l.iter().copied().map(transpose8).collect();
    let mut bosonic_entries_checked = 0usize;
    let mut bosonic_residual_entries = 0usize;
    let mut fermionic_entries_checked = 0usize;
    let mut fermionic_residual_entries = 0usize;
    let mut pair_residuals = Vec::new();
    let mut residual_l1_norm = 0usize;
    let mut maximum_absolute_residual = 0i16;

    for i in 0..N {
        for j in 0..N {
            let lij = multiply8(&l[i], &r[j]);
            let lji = multiply8(&l[j], &r[i]);
            let rij = multiply8(&r[i], &l[j]);
            let rji = multiply8(&r[j], &l[i]);
            for row in 0..D {
                for column in 0..D {
                    let expected = if i == j && row == column { 2 } else { 0 };
                    bosonic_entries_checked += 1;
                    fermionic_entries_checked += 1;
                    bosonic_residual_entries +=
                        usize::from(lij[row][column] + lji[row][column] != expected);
                    fermionic_residual_entries +=
                        usize::from(rij[row][column] + rji[row][column] != expected);
                }
            }
        }
    }

    for i in 0..N {
        for j in (i + 1)..N {
            let lij = multiply8(&l[i], &r[j]);
            let lji = multiply8(&l[j], &r[i]);
            let rij = multiply8(&r[i], &l[j]);
            let rji = multiply8(&r[j], &l[i]);
            let mut bosonic = 0usize;
            let mut fermionic = 0usize;
            let mut pair_l1 = 0usize;
            let mut pair_max = 0i16;
            for row in 0..D {
                for column in 0..D {
                    let b = lij[row][column] + lji[row][column];
                    let f = rij[row][column] + rji[row][column];
                    bosonic += usize::from(b != 0);
                    fermionic += usize::from(f != 0);
                    pair_l1 += usize::from(b.unsigned_abs()) + usize::from(f.unsigned_abs());
                    pair_max = pair_max.max(b.abs()).max(f.abs());
                }
            }
            if bosonic + fermionic > 0 {
                residual_l1_norm += pair_l1;
                maximum_absolute_residual = maximum_absolute_residual.max(pair_max);
                pair_residuals.push(PairResidualRecord {
                    colors: [i + 1, j + 1],
                    bosonic_residual_entries: bosonic,
                    fermionic_residual_entries: fermionic,
                    residual_l1_norm: pair_l1,
                    maximum_absolute_residual: pair_max,
                });
            }
        }
    }

    ClosureRecord {
        sparse_garden_passed: rep.verify_garden_algebra(),
        bosonic_entries_checked,
        bosonic_residual_entries,
        fermionic_entries_checked,
        fermionic_residual_entries,
        nonclosing_color_pairs: pair_residuals.len(),
        residual_l1_norm,
        maximum_absolute_residual,
        pair_residuals,
    }
}

fn graph_record(rep: &AdinkraRep, closes: bool) -> AdinkraRecord {
    let mut edges = Vec::with_capacity(N * D);
    for (color, matrix) in rep.l_matrices.iter().enumerate() {
        for boson in 0..D {
            let sign = matrix.sign[boson];
            edges.push(EdgeRecord {
                color: color + 1,
                boson: boson + 1,
                fermion: usize::from(matrix.perm[boson]) + 1,
                sign,
                style: if sign == 1 { "solid" } else { "dashed" },
            });
        }
    }

    let mut squares = Vec::with_capacity(N * (N - 1) / 2 * D / 2);
    for first in 0..N {
        for second in (first + 1)..N {
            let p = &rep.l_matrices[first];
            let q = &rep.l_matrices[second];
            let q_inv = q.inverse();
            let mut seen = BTreeSet::new();
            for boson_a in 0..D {
                let fermion_a = usize::from(p.perm[boson_a]);
                let boson_b = usize::from(q_inv.perm[fermion_a]);
                let key = [boson_a.min(boson_b), boson_a.max(boson_b)];
                if !seen.insert(key) {
                    continue;
                }
                let fermion_b = usize::from(p.perm[boson_b]);
                assert_eq!(usize::from(q.perm[boson_a]), fermion_b);
                let signs = [
                    p.sign[boson_a],
                    q.sign[boson_b],
                    p.sign[boson_b],
                    q.sign[boson_a],
                ];
                let sign_product = signs.iter().product();
                let dashed_edges = signs.iter().filter(|&&sign| sign == -1).count();
                squares.push(SquareRecord {
                    colors: [first + 1, second + 1],
                    bosons: [boson_a + 1, boson_b + 1],
                    fermions: [fermion_a + 1, fermion_b + 1],
                    dashed_edges,
                    sign_product,
                    odd_dashing: dashed_edges % 2 == 1 && sign_product == -1,
                });
            }
            assert_eq!(seen.len(), D / 2);
        }
    }
    let odd_dashing_failures = squares.iter().filter(|square| !square.odd_dashing).count();
    AdinkraRecord {
        graph_type: if closes {
            "valid eight-color valise Garden Adinkra"
        } else {
            "signed eight-color valise graph with published nonclosure"
        },
        bosons: (1..=D).map(|index| format!("B{index}")).collect(),
        fermions: (1..=D).map(|index| format!("F{index}")).collect(),
        edge_count: edges.len(),
        square_count: squares.len(),
        edges,
        two_color_squares: squares,
        odd_dashing_failures,
        valid_garden_adinkra: closes && odd_dashing_failures == 0,
    }
}

fn multiply_dense(left: &[Vec<i16>], right: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let rows = left.len();
    let inner = right.len();
    let columns = right[0].len();
    let mut result = vec![vec![0i16; columns]; rows];
    for row in 0..rows {
        for column in 0..columns {
            result[row][column] = (0..inner).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    result
}

fn hymn_record(rep: &AdinkraRep, should_close: bool) -> HymnRecord {
    let mut product = vec![vec![0i16; 16]; 16];
    for (index, row) in product.iter_mut().enumerate() {
        row[index] = 1;
    }
    for signed in &rep.l_matrices {
        let l = dense_signed(signed);
        let r = transpose8(l);
        let gamma: Vec<Vec<i16>> = (0..16)
            .map(|row| {
                (0..16)
                    .map(|column| match (row < 8, column < 8) {
                        (true, false) => i16::from(l[row][column - 8]),
                        (false, true) => i16::from(r[row - 8][column]),
                        _ => 0,
                    })
                    .collect()
            })
            .collect();
        product = multiply_dense(&gamma, &product);
    }
    let off_diagonal_entries = product
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter(|&(column, &value)| row != column && value != 0)
                .count()
        })
        .sum();
    let diagonal_entries: Vec<i16> = (0..16).map(|index| product[index][index]).collect();
    let trace = diagonal_entries.iter().sum();
    let positive = diagonal_entries.iter().filter(|&&value| value > 0).count();
    let negative = diagonal_entries.iter().filter(|&&value| value < 0).count();
    let zero = diagonal_entries.iter().filter(|&&value| value == 0).count();
    let class = if off_diagonal_entries == 0
        && diagonal_entries[..8].iter().all(|&value| value == 1)
        && diagonal_entries[8..].iter().all(|&value| value == -1)
    {
        "sigma3_tensor_identity8"
    } else if off_diagonal_entries == 0 && diagonal_entries.iter().all(|&value| value == 1) {
        "identity16"
    } else {
        "other"
    };
    let equation_3_3_matched = if should_close {
        class == "sigma3_tensor_identity8"
    } else {
        class == "identity16"
    };
    HymnRecord {
        diagonal: off_diagonal_entries == 0,
        diagonal_entries,
        off_diagonal_entries,
        trace,
        positive_eigenvalues: positive,
        negative_eigenvalues: negative,
        zero_eigenvalues: zero,
        class,
        equation_3_3_matched,
    }
}

fn sign_mask(factors: [u8; 8]) -> u64 {
    factors
        .iter()
        .enumerate()
        .fold(0u64, |mask, (color, &factor)| {
            mask | (u64::from(factor) << (color * 8))
        })
}

fn garden_distance(factors: [u8; 8], solutions: &[u64]) -> GardenDistanceRecord {
    let published = sign_mask(factors);
    let mut minimum = u32::MAX;
    let mut nearest = u64::MAX;
    let mut nearest_count = 0usize;
    for &solution in solutions {
        let distance = (solution ^ published).count_ones();
        match distance.cmp(&minimum) {
            std::cmp::Ordering::Less => {
                minimum = distance;
                nearest = solution;
                nearest_count = 1;
            }
            std::cmp::Ordering::Equal => {
                nearest = nearest.min(solution);
                nearest_count += 1;
            }
            std::cmp::Ordering::Greater => {}
        }
    }
    GardenDistanceRecord {
        total_garden_signings: solutions.len(),
        published_sign_mask: format!("{published:016x}"),
        minimum_edge_sign_flips: minimum,
        nearest_garden_sign_mask: format!("{nearest:016x}"),
        nearest_signings_at_minimum: nearest_count,
    }
}

fn bruhat_matrix(octet: &[[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let permutations: Vec<_> = octet.iter().map(permutation).collect();
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            permutations[row]
                .bruhat_distance(permutations[column])
                .expect("S8 pair") as u8
        })
    })
}

fn matrix_i128_multiply(left: &[Vec<i128>], right: &[Vec<i128>]) -> Vec<Vec<i128>> {
    let n = left.len();
    let mut result = vec![vec![0i128; n]; n];
    for row in 0..n {
        for column in 0..n {
            result[row][column] = (0..n).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    result
}

fn characteristic_polynomial(matrix: [[u8; 8]; 8]) -> Vec<i128> {
    let n = 8usize;
    let a: Vec<Vec<i128>> = matrix
        .iter()
        .map(|row| row.iter().map(|&value| i128::from(value)).collect())
        .collect();
    let mut traces = vec![0i128; n + 1];
    let mut power = vec![vec![0i128; n]; n];
    for (index, row) in power.iter_mut().enumerate() {
        row[index] = 1;
    }
    for slot in traces.iter_mut().skip(1) {
        power = matrix_i128_multiply(&power, &a);
        *slot = (0..n).map(|index| power[index][index]).sum();
    }
    let mut coefficients = vec![0i128; n + 1];
    coefficients[0] = 1;
    for k in 1..=n {
        let sum: i128 = (1..=k).map(|i| coefficients[k - i] * traces[i]).sum();
        assert_eq!(sum % k as i128, 0);
        coefficients[k] = -sum / k as i128;
    }
    coefficients
}

fn normalizer_action(representative: Permutation) -> Option<[u8; 7]> {
    let subgroup = rana_r8();
    let identity = Permutation::identity(8).expect("S8 identity");
    let nonidentity: Vec<_> = subgroup
        .iter()
        .copied()
        .filter(|&entry| entry != identity)
        .collect();
    let ranks: BTreeMap<usize, u8> = nonidentity
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.rank(), index as u8 + 1))
        .collect();
    let inverse = representative.inverse();
    let mut action = [0u8; 7];
    for (index, &entry) in nonidentity.iter().enumerate() {
        let conjugate = representative
            .compose(entry)
            .expect("S8")
            .compose(inverse)
            .expect("S8");
        let &target = ranks.get(&conjugate.rank())?;
        action[index] = target;
    }
    Some(action)
}

fn unsigned_coordinates(
    sector: usize,
    right: &crate::permutahedron::CosetPartitionReport,
    abnormal_ids: &HashSet<u32>,
) -> UnsignedCoordinateRecord {
    let octet = &S8_REPRESENTATION_OCTETS[sector].permutations;
    let mut ranks: Vec<u32> = octet
        .iter()
        .map(|entry| permutation(entry).rank() as u32)
        .collect();
    ranks.sort_unstable();
    let right_r8_coset_id = right
        .slices
        .iter()
        .position(|slice| slice == &ranks)
        .expect("published octet is an R8 right coset") as u32;
    let matrix = bruhat_matrix(octet);
    let row_sums = matrix
        .iter()
        .map(|row| row.iter().map(|&value| u16::from(value)).sum())
        .collect();
    let upper = (0..8)
        .flat_map(|row| ((row + 1)..8).map(move |column| matrix[row][column]))
        .collect();
    let abnormal = abnormal_ids.contains(&right_r8_coset_id);
    let representative = Permutation::unrank(8, ranks[0] as usize).expect("S8 rank");
    UnsignedCoordinateRecord {
        right_r8_coset_id,
        canonical_coset_ranks: ranks,
        abnormal_left_right_coincident: abnormal,
        normalizer_gl_3_2_action: abnormal
            .then(|| normalizer_action(representative))
            .flatten(),
        bruhat_row_sums: row_sums,
        bruhat_upper_triangle: upper,
        bruhat_characteristic_polynomial: characteristic_polynomial(matrix),
        bruhat_frobenius_norm_squared: matrix
            .iter()
            .flatten()
            .map(|&value| u32::from(value).pow(2))
            .sum(),
    }
}

fn branch_records(sector: usize, solutions: &[u64]) -> Vec<ParameterBranchRecord> {
    let should_close = matches!(sector, 1 | 2);
    let parameters: Vec<(Option<u8>, Option<u8>)> = if sector < 3 {
        vec![(None, None)]
    } else {
        (0u8..4)
            .flat_map(|m| (0u8..4).map(move |n| (Some(m), Some(n))))
            .collect()
    };
    parameters
        .into_iter()
        .map(|(m_mod_4, n_mod_4)| {
            let m = m_mod_4.unwrap_or(0);
            let n = n_mod_4.unwrap_or(0);
            let (m_plus, m_minus) = parameter_signs(m);
            let (n_plus, n_minus) = parameter_signs(n);
            let factors = effective_factors(sector, m, n);
            let rep = build_rep(sector, factors);
            let closure = closure_record(&rep);
            let closes =
                closure.bosonic_residual_entries == 0 && closure.fermionic_residual_entries == 0;
            ParameterBranchRecord {
                m_mod_4,
                n_mod_4,
                m_plus,
                m_minus,
                n_plus,
                n_minus,
                effective_boolean_factors: factors,
                matrices: matrix_records(sector, factors, &rep),
                hymn: hymn_record(&rep, should_close),
                garden_distance: garden_distance(factors, solutions),
                graph: graph_record(&rep, closes),
                closure,
            }
        })
        .collect()
}

fn disjoint_value_sets<T: Ord + Clone>(
    sectors: &[SectorRecord],
    value: impl Fn(&SectorRecord) -> T,
) -> bool {
    let closing: BTreeSet<T> = sectors
        .iter()
        .filter(|sector| matches!(sector.id, "CT" | "CV"))
        .map(&value)
        .collect();
    let nonclosing: BTreeSet<T> = sectors
        .iter()
        .filter(|sector| !matches!(sector.id, "CT" | "CV"))
        .map(value)
        .collect();
    closing.is_disjoint(&nonclosing)
}

pub fn build() -> S8SupersymmetryArtifact {
    let subgroup = rana_r8();
    let right = coset_partition(&subgroup, CosetSide::Right).expect("R8 right cosets");
    let abnormal = abnormal_slices(&subgroup).expect("R8 abnormal cosets");
    let abnormal_minima: HashSet<u32> = abnormal.representative_ranks.into_iter().collect();
    let abnormal_ids: HashSet<u32> = right
        .slices
        .iter()
        .enumerate()
        .filter_map(|(id, slice)| abnormal_minima.contains(&slice[0]).then_some(id as u32))
        .collect();

    let canonical_reps: Vec<_> = (0..6)
        .map(|sector| build_rep(sector, effective_factors(sector, 0, 0)))
        .collect();
    let canonical_holoraumy: Vec<_> = canonical_reps.iter().map(HoloraumyData::from_rep).collect();
    let formal_gadget: Vec<Vec<f64>> = canonical_holoraumy
        .iter()
        .map(|left| {
            canonical_holoraumy
                .iter()
                .map(|right| {
                    let value = gadget(left, right);
                    if value.abs() < 1e-12 { 0.0 } else { value }
                })
                .collect()
        })
        .collect();

    let labels = [
        ("CC", "chiral + chiral", "published nonclosure"),
        ("CT", "chiral + tensor", "published Garden closure"),
        ("CV", "chiral + vector", "published Garden closure"),
        ("TT", "tensor + tensor", "published nonclosure"),
        ("TV", "tensor + vector", "published nonclosure"),
        ("VV", "vector + vector", "published nonclosure"),
    ];
    let mut sectors = Vec::with_capacity(6);
    for sector in 0..6 {
        let octet: Vec<Permutation> = S8_REPRESENTATION_OCTETS[sector]
            .permutations
            .iter()
            .map(permutation)
            .collect();
        let solutions =
            garden_solution_masks(&octet).expect("published unsigned octet is signable");
        assert_eq!(solutions.len(), 1 << 19);
        let branches = branch_records(sector, &solutions);
        let closure_values: BTreeSet<bool> = branches
            .iter()
            .map(|branch| branch.closure.sparse_garden_passed)
            .collect();
        let hymn_values: BTreeSet<&str> = branches.iter().map(|branch| branch.hymn.class).collect();
        let (support, q) = chromochar_support_and_q(&canonical_reps[sector]);
        sectors.push(SectorRecord {
            id: labels[sector].0,
            label: labels[sector].1,
            paper_status: labels[sector].2,
            permutations: S8_REPRESENTATION_OCTETS[sector].permutations,
            base_boolean_factors: S8_BASE_BOOLEAN_FACTORS[sector],
            unsigned_coordinates: unsigned_coordinates(sector, &right, &abnormal_ids),
            branch_count: branches.len(),
            closure_status_uniform_across_branches: closure_values.len() == 1,
            hymn_class_uniform_across_branches: hymn_values.len() == 1,
            parameter_branches: branches,
            canonical_chromochar_support: support,
            canonical_chromochar_q_scaled: q,
            canonical_formal_gadget_row: formal_gadget[sector].clone(),
            formal_gadget_is_representation_invariant: matches!(sector, 1 | 2),
        });
    }

    let hymn_matches = sectors.iter().all(|sector| {
        let should_close = matches!(sector.id, "CT" | "CV");
        sector.parameter_branches.iter().all(|branch| {
            branch.hymn.equation_3_3_matched
                && (branch.hymn.trace == 0) == should_close
                && branch.closure.sparse_garden_passed == should_close
        })
    });
    let garden_distance_matches = sectors.iter().all(|sector| {
        let should_close = matches!(sector.id, "CT" | "CV");
        sector
            .parameter_branches
            .iter()
            .all(|branch| (branch.garden_distance.minimum_edge_sign_flips == 0) == should_close)
    });
    let abnormality_matches = sectors.iter().all(|sector| {
        let should_close = matches!(sector.id, "CT" | "CV");
        sector.unsigned_coordinates.abnormal_left_right_coincident != should_close
    });
    let hymn_trace_disjoint = {
        let closing: BTreeSet<i16> = sectors
            .iter()
            .filter(|sector| matches!(sector.id, "CT" | "CV"))
            .flat_map(|sector| {
                sector
                    .parameter_branches
                    .iter()
                    .map(|branch| branch.hymn.trace)
            })
            .collect();
        let nonclosing: BTreeSet<i16> = sectors
            .iter()
            .filter(|sector| !matches!(sector.id, "CT" | "CV"))
            .flat_map(|sector| {
                sector
                    .parameter_branches
                    .iter()
                    .map(|branch| branch.hymn.trace)
            })
            .collect();
        closing.is_disjoint(&nonclosing)
    };
    let bruhat_disjoint = disjoint_value_sets(&sectors, |sector| {
        sector
            .unsigned_coordinates
            .bruhat_characteristic_polynomial
            .clone()
    });
    let chromochar_disjoint =
        disjoint_value_sets(&sectors, |sector| sector.canonical_chromochar_q_scaled);
    let self_gadget_disjoint = disjoint_value_sets(&sectors, |sector| {
        let value = sector.canonical_formal_gadget_row[sectors
            .iter()
            .position(|entry| entry.id == sector.id)
            .unwrap()];
        (value * 1_000_000.0).round() as i64
    });

    let parameter_branches_checked: usize = sectors.iter().map(|sector| sector.branch_count).sum();
    let closing_branches = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .filter(|branch| branch.closure.sparse_garden_passed)
        .count();
    let dense_entries_checked: usize = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .map(|branch| {
            branch.closure.bosonic_entries_checked + branch.closure.fermionic_entries_checked
        })
        .sum();
    let dense_residual_entries: usize = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .map(|branch| {
            branch.closure.bosonic_residual_entries + branch.closure.fermionic_residual_entries
        })
        .sum();
    let graphs_constructed = parameter_branches_checked;
    let graph_edges_checked = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .map(|branch| branch.graph.edge_count)
        .sum();
    let graph_squares_checked = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .map(|branch| branch.graph.square_count)
        .sum();
    let valid_garden_adinkras = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .filter(|branch| branch.graph.valid_garden_adinkra)
        .count();
    let equation_3_3_matches = sectors
        .iter()
        .flat_map(|sector| &sector.parameter_branches)
        .filter(|branch| branch.hymn.equation_3_3_matched)
        .count();
    let passed = sectors.len() == 6
        && parameter_branches_checked == 51
        && closing_branches == 2
        && equation_3_3_matches == 51
        && valid_garden_adinkras == 2
        && hymn_matches
        && garden_distance_matches
        && sectors.iter().all(|sector| {
            sector.closure_status_uniform_across_branches
                && sector.hymn_class_uniform_across_branches
                && sector
                    .unsigned_coordinates
                    .bruhat_row_sums
                    .iter()
                    .all(|&sum| sum == 112)
        });

    let mut conventions = BTreeMap::new();
    conventions.insert(
        "boolean_factor",
        "eight-bit reversed binary; bit zero controls matrix row one",
    );
    conventions.insert("signed_matrix", "L_I=S_I P_I and R_I=L_I^T");
    conventions.insert(
        "height_yielding_parameters",
        "all residue classes m mod 4 and n mod 4 are evaluated for TT, TV, and VV",
    );
    conventions.insert(
        "hymn",
        "C_hat=gamma_hat_8 ... gamma_hat_1 with gamma_hat_I=[[0,L_I],[R_I,0]]",
    );

    S8SupersymmetryArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Six signed S8 permutahedron representations",
        sources: vec![
            SourceRecord {
                arxiv_id: "2012.14015",
                version: 7,
                locator: "Eqs. (2.1)-(2.7)",
                role: "block L-matrices and m/n sign parameters",
            },
            SourceRecord {
                arxiv_id: "2012.14015",
                version: 7,
                locator: "Eqs. (3.1)-(3.5)",
                role: "HYMN definition, two HYMN classes, and closure split",
            },
            SourceRecord {
                arxiv_id: "2012.14015",
                version: 7,
                locator: "Eqs. (5.1)-(5.6)",
                role: "S8 one-line permutations and Boolean factors",
            },
        ],
        conventions,
        sectors,
        canonical_formal_gadget_matrix: formal_gadget,
        separation: SeparationAssessment {
            hymn_class_matches_published_closure_on_every_branch: hymn_matches,
            hymn_trace_value_sets_disjoint: hymn_trace_disjoint,
            garden_distance_zero_matches_closure: garden_distance_matches,
            unsigned_abnormality_separates_closure: abnormality_matches,
            bruhat_characteristic_polynomial_value_sets_disjoint: bruhat_disjoint,
            chromochar_q_value_sets_disjoint: chromochar_disjoint,
            formal_self_gadget_value_sets_disjoint: self_gadget_disjoint,
            conclusion: "The published HYMN class separates CT/CV from CC/TT/TV/VV for every m,n branch. Unsigned R8-coset abnormality does not. Zero distance to the Garden-sign affine space is an equivalent defect check, not an independent physical invariant.",
        },
        validation: ValidationRecord {
            sectors: 6,
            source_permutation_entries_checked: 6 * 8 * 8,
            source_boolean_factors_checked: 6 * 8,
            parameter_branches_checked,
            closing_branches,
            nonclosing_branches: parameter_branches_checked - closing_branches,
            dense_closure_entries_checked: dense_entries_checked,
            dense_closure_residual_entries: dense_residual_entries,
            hymn_records_checked: parameter_branches_checked,
            equation_3_3_matches,
            graphs_constructed,
            graph_edges_checked,
            graph_squares_checked,
            valid_garden_adinkras,
            signed_graphs_with_nonclosure: graphs_constructed - valid_garden_adinkras,
            nearest_garden_solution_spaces_scanned: 6,
            passed,
        },
        boundary: "CT and CV are valid Garden Adinkras for the printed signings. CC, TT, TV, and VV are stored as signed colored valise graphs with the published nonclosure, not relabeled as off-shell Adinkras. Formal gadget values for those four graphs are diagnostics and are not representation invariants.",
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ValidationRecord {
    let artifact = build();
    assert!(artifact.validation.passed);
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create validation directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create S8 signed artifact")),
        &artifact,
    )
    .expect("write S8 signed artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create S8 validation artifact")),
        &artifact.validation,
    )
    .expect("write S8 validation artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_sign_cycle_matches_equation_2_5() {
        assert_eq!(parameter_signs(0), (1, 1));
        assert_eq!(parameter_signs(1), (1, -1));
        assert_eq!(parameter_signs(2), (-1, -1));
        assert_eq!(parameter_signs(3), (-1, 1));
    }

    #[test]
    fn block_boolean_factors_match_equations_2_and_5() {
        assert_eq!(S8_BASE_BOOLEAN_FACTORS[0][0], 10 | (10 << 4));
        assert_eq!(S8_BASE_BOOLEAN_FACTORS[1][0], 10 | (14 << 4));
        assert_eq!(S8_BASE_BOOLEAN_FACTORS[2][4], 2 | (13 << 4));
        assert_eq!(S8_BASE_BOOLEAN_FACTORS[3][0], 14 | (14 << 4));
        assert_eq!(S8_BASE_BOOLEAN_FACTORS[4][0], 14 | (10 << 4));
        assert_eq!(S8_BASE_BOOLEAN_FACTORS[5][3], 6 | (6 << 4));
    }

    #[test]
    fn six_sector_reproduction_passes_every_gate() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.parameter_branches_checked, 51);
        assert_eq!(artifact.validation.closing_branches, 2);
        assert_eq!(artifact.validation.nonclosing_branches, 49);
        assert_eq!(artifact.validation.equation_3_3_matches, 51);
        assert_eq!(artifact.validation.valid_garden_adinkras, 2);
        assert!(
            artifact
                .separation
                .hymn_class_matches_published_closure_on_every_branch
        );
        assert!(!artifact.separation.unsigned_abnormality_separates_closure);
    }
}
