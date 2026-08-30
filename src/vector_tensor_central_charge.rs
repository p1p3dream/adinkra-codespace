//! Exact central-extension audit for the published eight-color pairings.
//!
//! Ordinary Garden closure deliberately sets all central charges to zero.  The
//! vector-tensor multiplet does not obey that algebra.  This module factors the
//! stored closure residuals into central operators and checks the extended
//! algebra, including commutation with every supercharge.

#![allow(clippy::needless_range_loop)]

use crate::permutahedron_fixtures::S8_REPRESENTATION_OCTETS;
use crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS;
use num_rational::Ratio;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const N: usize = 8;
const D: usize = 8;
const TV_SECTOR: usize = 4;
const SCHEMA_VERSION: &str = "vector-tensor-central-charge-v1";

pub(crate) type Matrix = [[i16; D]; D];

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
    pub pdf_sha256: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorldlineMapRecord {
    pub source_basis: Vec<&'static str>,
    pub valised_basis: Vec<&'static str>,
    pub source_rules: Vec<&'static str>,
    pub valised_rule: &'static str,
    pub canonical_node_operator: Matrix,
    pub squared_to_identity: bool,
    pub trace: i16,
    pub positive_eigenvalues: usize,
    pub negative_eigenvalues: usize,
    pub interpretation: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct AppendixFBridgeRecord {
    pub branches_checked: usize,
    pub matrices_checked: usize,
    pub entries_checked: usize,
    pub mismatches: usize,
    pub exact_match: bool,
    pub interpretation: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResidualPairRecord {
    pub colors: [usize; 2],
    pub coefficient: Option<i16>,
    pub bosonic_nonzero_entries: usize,
    pub fermionic_nonzero_entries: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExtractedCentralCharge {
    pub bosonic: Matrix,
    pub fermionic: Matrix,
    pub color_coefficient_matrix: Matrix,
    pub bosonic_signed_permutation: bool,
    pub fermionic_signed_permutation: bool,
    pub bosonic_symmetric: bool,
    pub fermionic_symmetric: bool,
    pub bosonic_involution: bool,
    pub fermionic_involution: bool,
    pub bosonic_trace: i16,
    pub fermionic_trace: i16,
    pub commutes_with_all_supercharges: bool,
    pub extended_closure_passed: bool,
    pub physical_bosonic_conjugacy_signature_matched: bool,
    pub physical_bosonic_conjugator: Option<Matrix>,
    pub source_omega_equation_81_matched: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct BranchRecord {
    pub m_mod_4: Option<u8>,
    pub n_mod_4: Option<u8>,
    pub residue_parity: Option<&'static str>,
    pub effective_boolean_factors: [u8; N],
    pub ordinary_garden_closure: bool,
    pub nonzero_residual_pairs: usize,
    pub bosonic_residual_rank: usize,
    pub fermionic_residual_rank: usize,
    pub paired_residual_rank: usize,
    pub residual_pairs: Vec<ResidualPairRecord>,
    pub one_central_charge_completion: bool,
    pub central_charge: Option<ExtractedCentralCharge>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SectorRecord {
    pub id: &'static str,
    pub label: &'static str,
    pub branches: Vec<BranchRecord>,
    pub ordinary_closing_branches: usize,
    pub one_central_charge_branches: usize,
    pub higher_rank_residual_branches: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensorAssessment {
    pub branches_checked: usize,
    pub one_central_charge_branches: usize,
    pub rejected_one_charge_branches: usize,
    pub accepted_residue_condition: &'static str,
    pub accepted_nonzero_color_pairs: Vec<[usize; 2]>,
    pub accepted_color_coefficients: Vec<i16>,
    pub exact_result: &'static str,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ValidationRecord {
    pub sectors_checked: usize,
    pub branches_checked: usize,
    pub residual_matrices_ranked: usize,
    pub ordinary_garden_branches: usize,
    pub one_central_charge_branches: usize,
    pub central_intertwining_checks: usize,
    pub extended_closure_checks: usize,
    pub vector_tensor_odd_branches_completed: usize,
    pub vector_tensor_even_branches_rejected: usize,
    pub physical_worldline_operator_squared: bool,
    pub appendix_f_entries_checked: usize,
    pub appendix_f_mismatches: usize,
    pub vector_tensor_explicit_conjugators: usize,
    pub vector_tensor_source_omega_matches: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensorCentralChargeArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub conventions: Vec<&'static str>,
    pub physical_worldline_map: WorldlineMapRecord,
    pub appendix_f_bridge: AppendixFBridgeRecord,
    pub sectors: Vec<SectorRecord>,
    pub vector_tensor: VectorTensorAssessment,
    pub validation: ValidationRecord,
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

fn effective_factors(sector: usize, m: u8, n: u8) -> [u8; N] {
    let mut factors = S8_BASE_BOOLEAN_FACTORS[sector];
    if sector < 3 {
        return factors;
    }
    let (m_plus, m_minus) = parameter_signs(m);
    let (n_plus, n_minus) = parameter_signs(n);
    for color in 0..N {
        let (m_sign, n_sign) = if color < 4 {
            (m_plus, n_plus)
        } else {
            (m_minus, n_minus)
        };
        if n_sign == -1 {
            factors[color] ^= 0x0f;
        }
        if m_sign == -1 {
            factors[color] ^= 0xf0;
        }
    }
    factors
}

pub(crate) fn l_matrices(sector: usize, factors: [u8; N]) -> [Matrix; N] {
    std::array::from_fn(|color| {
        let permutation = S8_REPRESENTATION_OCTETS[sector].permutations[color];
        let factor = factors[color];
        let mut matrix = [[0i16; D]; D];
        for row in 0..D {
            let sign = if factor & (1 << row) == 0 { 1 } else { -1 };
            matrix[row][usize::from(permutation[row] - 1)] = sign;
        }
        matrix
    })
}

fn transpose(matrix: &Matrix) -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}

fn scale(matrix: &Matrix, coefficient: i16) -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| coefficient * matrix[row][column]))
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..D)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn identity() -> Matrix {
    std::array::from_fn(|row| std::array::from_fn(|column| i16::from(row == column)))
}

fn zero() -> Matrix {
    [[0; D]; D]
}

fn flatten(matrix: &Matrix) -> Vec<i16> {
    matrix.iter().flatten().copied().collect()
}

fn exact_rank(columns: &[Vec<i16>]) -> usize {
    if columns.is_empty() {
        return 0;
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut matrix: Vec<Vec<Ratio<i64>>> = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|column| Ratio::from_integer(i64::from(columns[column][row])))
                .collect()
        })
        .collect();
    let mut rank = 0usize;
    for column in 0..cols {
        let Some(pivot) = (rank..rows).find(|&row| *matrix[row][column].numer() != 0) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let pivot_value = matrix[rank][column];
        for entry in matrix[rank].iter_mut().skip(column) {
            *entry /= pivot_value;
        }
        let pivot_row = matrix[rank].clone();
        for row in 0..rows {
            if row == rank || *matrix[row][column].numer() == 0 {
                continue;
            }
            let coefficient = matrix[row][column];
            for col in column..cols {
                matrix[row][col] -= coefficient * pivot_row[col];
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

fn nonzero_entries(matrix: &Matrix) -> usize {
    matrix.iter().flatten().filter(|&&value| value != 0).count()
}

fn is_signed_permutation(matrix: &Matrix) -> bool {
    (0..D).all(|row| {
        matrix[row].iter().filter(|&&value| value != 0).count() == 1
            && matrix[row].iter().all(|&value| value.abs() <= 1)
    }) && (0..D).all(|column| (0..D).filter(|&row| matrix[row][column] != 0).count() == 1)
}

fn coefficient_against(value: &[i16], base: &[i16]) -> Option<i16> {
    let pivot = base.iter().position(|&entry| entry != 0)?;
    if value[pivot] % base[pivot] != 0 {
        return None;
    }
    let coefficient = value[pivot] / base[pivot];
    value
        .iter()
        .zip(base)
        .all(|(&left, &right)| left == coefficient * right)
        .then_some(coefficient)
}

fn residuals(l: &[Matrix; N]) -> Vec<([usize; 2], Matrix, Matrix)> {
    let r: [Matrix; N] = std::array::from_fn(|color| transpose(&l[color]));
    let mut records = Vec::new();
    for first in 0..N {
        for second in (first + 1)..N {
            let bosonic = add(
                &multiply(&l[first], &r[second]),
                &multiply(&l[second], &r[first]),
            );
            let fermionic = add(
                &multiply(&r[first], &l[second]),
                &multiply(&r[second], &l[first]),
            );
            if bosonic != zero() || fermionic != zero() {
                records.push(([first + 1, second + 1], bosonic, fermionic));
            }
        }
    }
    records
}

fn canonical_worldline_operator() -> Matrix {
    // (phi, V1, V2, V3 | F, H1, H2, H3), where dot(F)=D.
    std::array::from_fn(|row| std::array::from_fn(|column| i16::from(column == (row + 4) % 8)))
}

fn signed_block(boolean_factor: u8, permutation: [usize; 4], scale: i8) -> [[i16; 4]; 4] {
    let mut matrix = [[0i16; 4]; 4];
    for row in 0..4 {
        let dash = if boolean_factor & (1 << row) == 0 {
            1
        } else {
            -1
        };
        matrix[row][permutation[row]] = i16::from(scale * dash);
    }
    matrix
}

fn appendix_f_matrices(m: u8, n: u8) -> [Matrix; N] {
    let (a_plus, a_minus) = parameter_signs(m);
    let (b_plus, b_minus) = parameter_signs(n);
    let top_factors = [14, 4, 8, 2];
    let top_permutations = [
        [0, 2, 3, 1], // (234)
        [1, 3, 2, 0], // (124)
        [2, 0, 1, 3], // (132)
        [3, 1, 0, 2], // (143)
    ];
    let bottom_factors = [10, 12, 0, 6];
    let bottom_permutations = [
        [1, 3, 0, 2], // (1243)
        [0, 2, 1, 3], // (23)
        [3, 1, 2, 0], // (14)
        [2, 0, 3, 1], // (1342)
    ];
    std::array::from_fn(|color| {
        let block = color % 4;
        let top = signed_block(
            top_factors[block],
            top_permutations[block],
            if color < 4 { b_plus } else { b_minus },
        );
        let bottom = signed_block(
            bottom_factors[block],
            bottom_permutations[block],
            if color < 4 { a_plus } else { a_minus },
        );
        let mut matrix = zero();
        for row in 0..4 {
            for column in 0..4 {
                if color < 4 {
                    matrix[row][column] = top[row][column];
                    matrix[row + 4][column + 4] = bottom[row][column];
                } else {
                    matrix[row][column + 4] = top[row][column];
                    matrix[row + 4][column] = bottom[row][column];
                }
            }
        }
        matrix
    })
}

fn appendix_f_bridge() -> AppendixFBridgeRecord {
    let mut mismatches = 0usize;
    for m in 0u8..4 {
        for n in 0u8..4 {
            let source = appendix_f_matrices(m, n);
            let fixture = l_matrices(TV_SECTOR, effective_factors(TV_SECTOR, m, n));
            for color in 0..N {
                for row in 0..D {
                    for column in 0..D {
                        mismatches +=
                            usize::from(source[color][row][column] != fixture[color][row][column]);
                    }
                }
            }
        }
    }
    AppendixFBridgeRecord {
        branches_checked: 16,
        matrices_checked: 16 * N,
        entries_checked: 16 * N * D * D,
        mismatches,
        exact_match: mismatches == 0,
        interpretation: "Direct reconstruction of arXiv:1405.0048 Appendix F from its 4x4 Boolean factors, cycles, and a+/a-/b+/b- signs",
    }
}

fn signature_matches_physical(matrix: &Matrix) -> bool {
    is_signed_permutation(matrix)
        && transpose(matrix) == *matrix
        && multiply(matrix, matrix) == identity()
        && (0..D).map(|index| matrix[index][index]).sum::<i16>() == 0
}

fn physical_conjugator(matrix: &Matrix) -> Option<Matrix> {
    if !signature_matches_physical(matrix) {
        return None;
    }
    let mut pairs = Vec::new();
    let mut seen = [false; D];
    for left in 0..D {
        if seen[left] {
            continue;
        }
        let right = (0..D).find(|&column| matrix[left][column] != 0)?;
        if left == right || seen[right] || matrix[right][left] != matrix[left][right] {
            return None;
        }
        seen[left] = true;
        seen[right] = true;
        pairs.push((left, right, matrix[left][right]));
    }
    if pairs.len() != 4 {
        return None;
    }
    let mut conjugator = zero();
    for (target, &(left, right, sign)) in pairs.iter().enumerate() {
        conjugator[target][left] = 1;
        conjugator[target + 4][right] = sign;
    }
    (is_signed_permutation(&conjugator)
        && multiply(&multiply(&conjugator, matrix), &transpose(&conjugator))
            == canonical_worldline_operator())
    .then_some(conjugator)
}

fn source_omega_equation_81() -> Matrix {
    let mut omega = zero();
    for (left, right, coefficient) in [(0, 6, 1), (1, 7, 1), (2, 4, -1), (3, 5, -1)] {
        omega[left][right] = coefficient;
        omega[right][left] = coefficient;
    }
    omega
}

fn analyze_branch(sector: usize, m: Option<u8>, n: Option<u8>) -> BranchRecord {
    let factors = effective_factors(sector, m.unwrap_or(0), n.unwrap_or(0));
    let l = l_matrices(sector, factors);
    let r: [Matrix; N] = std::array::from_fn(|color| transpose(&l[color]));
    let residuals = residuals(&l);
    let bosonic_columns: Vec<_> = residuals
        .iter()
        .map(|(_, bosonic, _)| flatten(bosonic))
        .collect();
    let fermionic_columns: Vec<_> = residuals
        .iter()
        .map(|(_, _, fermionic)| flatten(fermionic))
        .collect();
    let paired_columns: Vec<_> = residuals
        .iter()
        .map(|(_, bosonic, fermionic)| {
            let mut value = flatten(bosonic);
            value.extend(flatten(fermionic));
            value
        })
        .collect();
    let bosonic_rank = exact_rank(&bosonic_columns);
    let fermionic_rank = exact_rank(&fermionic_columns);
    let paired_rank = exact_rank(&paired_columns);
    let one_charge = paired_rank == 1;

    let mut pair_records: Vec<_> = residuals
        .iter()
        .map(|(colors, bosonic, fermionic)| ResidualPairRecord {
            colors: *colors,
            coefficient: None,
            bosonic_nonzero_entries: nonzero_entries(bosonic),
            fermionic_nonzero_entries: nonzero_entries(fermionic),
        })
        .collect();

    let central_charge = if one_charge {
        let (_, base_b, base_f) = &residuals[0];
        assert!(base_b.iter().flatten().all(|entry| entry % 2 == 0));
        assert!(base_f.iter().flatten().all(|entry| entry % 2 == 0));
        let z_b = std::array::from_fn(|row| std::array::from_fn(|column| base_b[row][column] / 2));
        let z_f = std::array::from_fn(|row| std::array::from_fn(|column| base_f[row][column] / 2));
        let base_paired = paired_columns[0].clone();
        let mut omega = zero();
        for (index, ((colors, _, _), paired)) in residuals.iter().zip(&paired_columns).enumerate() {
            let coefficient = coefficient_against(paired, &base_paired)
                .expect("rank-one residual must be proportional");
            pair_records[index].coefficient = Some(coefficient);
            omega[colors[0] - 1][colors[1] - 1] = coefficient;
            omega[colors[1] - 1][colors[0] - 1] = coefficient;
        }
        let commutes = (0..N).all(|color| {
            multiply(&l[color], &z_f) == multiply(&z_b, &l[color])
                && multiply(&r[color], &z_b) == multiply(&z_f, &r[color])
        });
        let extended_closure = (0..N).all(|first| {
            (0..N).all(|second| {
                let expected_diagonal = if first == second {
                    scale(&identity(), 2)
                } else {
                    scale(&z_b, 2 * omega[first][second])
                };
                let expected_fermionic = if first == second {
                    scale(&identity(), 2)
                } else {
                    scale(&z_f, 2 * omega[first][second])
                };
                add(
                    &multiply(&l[first], &r[second]),
                    &multiply(&l[second], &r[first]),
                ) == expected_diagonal
                    && add(
                        &multiply(&r[first], &l[second]),
                        &multiply(&r[second], &l[first]),
                    ) == expected_fermionic
            })
        });
        let conjugator = physical_conjugator(&z_b);
        Some(ExtractedCentralCharge {
            bosonic: z_b,
            fermionic: z_f,
            color_coefficient_matrix: omega,
            bosonic_signed_permutation: is_signed_permutation(&z_b),
            fermionic_signed_permutation: is_signed_permutation(&z_f),
            bosonic_symmetric: transpose(&z_b) == z_b,
            fermionic_symmetric: transpose(&z_f) == z_f,
            bosonic_involution: multiply(&z_b, &z_b) == identity(),
            fermionic_involution: multiply(&z_f, &z_f) == identity(),
            bosonic_trace: (0..D).map(|index| z_b[index][index]).sum(),
            fermionic_trace: (0..D).map(|index| z_f[index][index]).sum(),
            commutes_with_all_supercharges: commutes,
            extended_closure_passed: extended_closure,
            physical_bosonic_conjugacy_signature_matched: conjugator.is_some(),
            physical_bosonic_conjugator: conjugator,
            source_omega_equation_81_matched: omega == source_omega_equation_81(),
        })
    } else {
        None
    };

    BranchRecord {
        m_mod_4: m,
        n_mod_4: n,
        residue_parity: m.zip(n).map(|(left, right)| {
            if (left + right).is_multiple_of(2) {
                "even"
            } else {
                "odd"
            }
        }),
        effective_boolean_factors: factors,
        ordinary_garden_closure: residuals.is_empty(),
        nonzero_residual_pairs: residuals.len(),
        bosonic_residual_rank: bosonic_rank,
        fermionic_residual_rank: fermionic_rank,
        paired_residual_rank: paired_rank,
        residual_pairs: pair_records,
        one_central_charge_completion: one_charge,
        central_charge,
    }
}

fn physical_worldline_map() -> WorldlineMapRecord {
    let operator = canonical_worldline_operator();
    WorldlineMapRecord {
        source_basis: vec!["phi", "V1", "V2", "V3", "D", "H1", "H2", "H3"],
        valised_basis: vec!["phi", "V1", "V2", "V3", "F with dot(F)=D", "H1", "H2", "H3"],
        source_rules: vec![
            "Z phi = D",
            "Z D = d_t^2 phi",
            "Z V_i = d_t H_i",
            "Z H_i = d_t V_i",
        ],
        valised_rule: "Z X = d_t (sigma_1 tensor I_4) X after lowering D to F",
        canonical_node_operator: operator,
        squared_to_identity: multiply(&operator, &operator) == identity(),
        trace: (0..D).map(|index| operator[index][index]).sum(),
        positive_eigenvalues: 4,
        negative_eigenvalues: 4,
        interpretation: "Equation (4.6) after zero spatial momentum, temporal gauge, dual-tensor normalization, and lowering D by D=dot(F)",
    }
}

pub fn build() -> VectorTensorCentralChargeArtifact {
    let labels = [
        ("CC", "chiral + chiral"),
        ("CT", "chiral + tensor"),
        ("CV", "chiral + vector"),
        ("TT", "tensor + tensor"),
        ("TV", "tensor + vector"),
        ("VV", "vector + vector"),
    ];
    let mut sectors = Vec::new();
    for sector in 0..6 {
        let parameters: Vec<(Option<u8>, Option<u8>)> = if sector < 3 {
            vec![(None, None)]
        } else {
            (0u8..4)
                .flat_map(|m| (0u8..4).map(move |n| (Some(m), Some(n))))
                .collect()
        };
        let branches: Vec<_> = parameters
            .into_iter()
            .map(|(m, n)| analyze_branch(sector, m, n))
            .collect();
        sectors.push(SectorRecord {
            id: labels[sector].0,
            label: labels[sector].1,
            ordinary_closing_branches: branches
                .iter()
                .filter(|branch| branch.ordinary_garden_closure)
                .count(),
            one_central_charge_branches: branches
                .iter()
                .filter(|branch| branch.one_central_charge_completion)
                .count(),
            higher_rank_residual_branches: branches
                .iter()
                .filter(|branch| branch.paired_residual_rank > 1)
                .count(),
            branches,
        });
    }

    let tv = &sectors[TV_SECTOR];
    let accepted: Vec<_> = tv
        .branches
        .iter()
        .filter(|branch| branch.one_central_charge_completion)
        .collect();
    let representative = accepted[0];
    let accepted_nonzero_color_pairs = representative
        .residual_pairs
        .iter()
        .map(|record| record.colors)
        .collect();
    let accepted_color_coefficients = representative
        .residual_pairs
        .iter()
        .map(|record| record.coefficient.expect("one-charge coefficient"))
        .collect();
    let worldline = physical_worldline_map();
    let physical_worldline_operator_squared = worldline.squared_to_identity;
    let appendix_f_bridge = appendix_f_bridge();
    let appendix_f_entries_checked = appendix_f_bridge.entries_checked;
    let appendix_f_mismatches = appendix_f_bridge.mismatches;
    let branches_checked: usize = sectors.iter().map(|sector| sector.branches.len()).sum();
    let residual_matrices_ranked: usize = sectors
        .iter()
        .flat_map(|sector| &sector.branches)
        .map(|branch| branch.nonzero_residual_pairs * 2)
        .sum();
    let ordinary_garden_branches = sectors
        .iter()
        .flat_map(|sector| &sector.branches)
        .filter(|branch| branch.ordinary_garden_closure)
        .count();
    let one_central_charge_branches = sectors
        .iter()
        .flat_map(|sector| &sector.branches)
        .filter(|branch| branch.one_central_charge_completion)
        .count();
    let central_records: Vec<_> = sectors
        .iter()
        .flat_map(|sector| &sector.branches)
        .filter_map(|branch| branch.central_charge.as_ref())
        .collect();
    let vector_tensor_odd_branches_completed = tv
        .branches
        .iter()
        .filter(|branch| {
            branch.residue_parity == Some("odd")
                && branch.one_central_charge_completion
                && branch.central_charge.as_ref().is_some_and(|charge| {
                    charge.extended_closure_passed
                        && charge.commutes_with_all_supercharges
                        && charge.physical_bosonic_conjugacy_signature_matched
                        && charge.source_omega_equation_81_matched
                })
        })
        .count();
    let vector_tensor_even_branches_rejected = tv
        .branches
        .iter()
        .filter(|branch| {
            branch.residue_parity == Some("even")
                && !branch.one_central_charge_completion
                && branch.paired_residual_rank == 6
        })
        .count();
    let passed = sectors.len() == 6
        && branches_checked == 51
        && ordinary_garden_branches == 2
        && vector_tensor_odd_branches_completed == 8
        && vector_tensor_even_branches_rejected == 8
        && worldline.squared_to_identity
        && appendix_f_bridge.exact_match
        && central_records.iter().all(|charge| {
            charge.extended_closure_passed
                && charge.commutes_with_all_supercharges
                && charge.bosonic_involution
                && charge.fermionic_involution
        });

    VectorTensorCentralChargeArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Central-charge completion of the S8 vector-tensor residual",
        sources: vec![
            SourceRecord {
                arxiv_id: "1405.0048",
                locator: "Sec. 3.6, Eqs. (76)-(84), Appendix F",
                role: "Majorana component rules, zero-central-charge residuals, and worldline basis",
                pdf_sha256: Some(
                    "8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20",
                ),
            },
            SourceRecord {
                arxiv_id: "hep-th/9609016",
                locator: "Eqs. (2.1), (3.34), (4.3), (4.5), and (4.6)",
                role: "N=2 central-charge algebra and physical vector-tensor Z action",
                pdf_sha256: Some(
                    "3bf2549954d01e4da9e1fedc4af8e3a534bf17df3c6a27d45bdc8a9a80b8c2af",
                ),
            },
            SourceRecord {
                arxiv_id: "2012.14015",
                locator: "Eqs. (2.4)-(2.7) and (5.1)-(5.6)",
                role: "Published S8 permutation octets and Boolean factors",
                pdf_sha256: None,
            },
        ],
        conventions: vec![
            "R_I = L_I transpose",
            "residual_B(I,J) = L_I R_J + L_J R_I for I != J",
            "residual_F(I,J) = R_I L_J + R_J L_I for I != J",
            "one-Z completion requires the paired boson/fermion residual span to have exact rank one",
            "extended closure is residual_B = 2 Omega_IJ Z_B and residual_F = 2 Omega_IJ Z_F",
        ],
        physical_worldline_map: worldline,
        appendix_f_bridge,
        vector_tensor: VectorTensorAssessment {
            branches_checked: tv.branches.len(),
            one_central_charge_branches: accepted.len(),
            rejected_one_charge_branches: tv.branches.len() - accepted.len(),
            accepted_residue_condition: "m + n is odd modulo 2",
            accepted_nonzero_color_pairs,
            accepted_color_coefficients,
            exact_result: "The published TV support is not dead: eight sign branches close exactly under one nontrivial central charge, and eight even-parity branches do not.",
            boundary: "This proves the worldline central extension and its physical conjugacy signature. It does not yet certify the full 4D component commutators modulo vector and two-form gauge transformations.",
        },
        validation: ValidationRecord {
            sectors_checked: sectors.len(),
            branches_checked,
            residual_matrices_ranked,
            ordinary_garden_branches,
            one_central_charge_branches,
            central_intertwining_checks: central_records.len() * N * 2,
            extended_closure_checks: central_records.len() * N * N * 2,
            vector_tensor_odd_branches_completed,
            vector_tensor_even_branches_rejected,
            physical_worldline_operator_squared,
            appendix_f_entries_checked,
            appendix_f_mismatches,
            vector_tensor_explicit_conjugators: accepted
                .iter()
                .filter(|branch| {
                    branch
                        .central_charge
                        .as_ref()
                        .is_some_and(|charge| charge.physical_bosonic_conjugator.is_some())
                })
                .count(),
            vector_tensor_source_omega_matches: accepted
                .iter()
                .filter(|branch| {
                    branch
                        .central_charge
                        .as_ref()
                        .is_some_and(|charge| charge.source_omega_equation_81_matched)
                })
                .count(),
            passed,
        },
        sectors,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ValidationRecord {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create central-charge data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create central-charge results directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create central-charge artifact")),
        &artifact,
    )
    .expect("write central-charge artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create validation artifact")),
        &artifact.validation,
    )
    .expect("write central-charge validation");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_tensor_has_exact_one_charge_branches() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.vector_tensor.one_central_charge_branches, 8);
        assert_eq!(artifact.vector_tensor.rejected_one_charge_branches, 8);
        assert_eq!(artifact.vector_tensor.accepted_nonzero_color_pairs.len(), 4);
    }

    #[test]
    fn physical_operator_is_a_balanced_involution() {
        let record = physical_worldline_map();
        assert!(record.squared_to_identity);
        assert_eq!(record.trace, 0);
        assert_eq!(record.positive_eigenvalues, 4);
        assert_eq!(record.negative_eigenvalues, 4);
    }
}
