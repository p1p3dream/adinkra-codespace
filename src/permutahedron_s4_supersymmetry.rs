//! Complete six-sector supersymmetric disassembly of the `S_4` permutahedron.
//!
//! The unsigned permutation quartets are the six `V_4` cosets printed in
//! arXiv:2012.13308.  The Boolean-factor quartets are Appendix B of
//! arXiv:1701.00304.  A Boolean factor `b` acts on matrix rows through
//! `diag((-1)^bit_0(b), ..., (-1)^bit_3(b))`, and `L_I = S_I P_I`.
//!
//! This module verifies all 96 published fiducial signed quartets, emits one
//! canonical valise Adinkra for each unsigned sector, and records both the
//! exact `S_4/V_4 ~= S_3` sector label and coarser signed invariants.

use crate::chromochar::{chi0_n4, chromochar_antisym};
use crate::holoraumy::{HoloraumyData, gadget};
use crate::lr_matrix::AdinkraRep;
use crate::permutahedron::{CosetSide, Permutation, coset_partition, permutations, vierergruppe};
use crate::permutahedron_fixtures::{
    S4_INTER_CORRELATORS, S4_INTRA_CORRELATORS, S4_ORDERED_QUARTETS,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SCHEMA_VERSION: &str = "permutahedron-s4-supersymmetry-v1";

/// Appendix B of arXiv:1701.00304, in the paper's P[1] through P[6] order.
///
/// The second index selects one of the 16 published fiducial Boolean-factor
/// quartets.  The third index is the color/L-matrix index.
pub const S4_BOOLEAN_FACTOR_QUARTETS: [[[u8; 4]; 16]; 6] = [
    [
        [0, 6, 12, 10],
        [0, 12, 10, 6],
        [2, 4, 14, 8],
        [2, 14, 8, 4],
        [4, 2, 8, 14],
        [4, 8, 14, 2],
        [6, 0, 10, 12],
        [6, 10, 12, 0],
        [8, 4, 2, 14],
        [8, 14, 4, 2],
        [10, 6, 0, 12],
        [10, 12, 6, 0],
        [12, 0, 6, 10],
        [12, 10, 0, 6],
        [14, 2, 4, 8],
        [14, 8, 2, 4],
    ],
    [
        [0, 10, 6, 12],
        [0, 12, 10, 6],
        [2, 8, 4, 14],
        [2, 14, 8, 4],
        [4, 8, 14, 2],
        [4, 14, 2, 8],
        [6, 10, 12, 0],
        [6, 12, 0, 10],
        [8, 2, 14, 4],
        [8, 4, 2, 14],
        [10, 0, 12, 6],
        [10, 6, 0, 12],
        [12, 0, 6, 10],
        [12, 6, 10, 0],
        [14, 2, 4, 8],
        [14, 4, 8, 2],
    ],
    [
        [0, 6, 10, 12],
        [0, 12, 6, 10],
        [2, 4, 8, 14],
        [2, 14, 4, 8],
        [4, 2, 14, 8],
        [4, 8, 2, 14],
        [6, 0, 12, 10],
        [6, 10, 0, 12],
        [8, 4, 14, 2],
        [8, 14, 2, 4],
        [10, 6, 12, 0],
        [10, 12, 0, 6],
        [12, 0, 10, 6],
        [12, 10, 6, 0],
        [14, 2, 8, 4],
        [14, 8, 4, 2],
    ],
    [
        [0, 10, 12, 6],
        [0, 12, 6, 10],
        [2, 8, 14, 4],
        [2, 14, 4, 8],
        [4, 8, 2, 14],
        [4, 14, 8, 2],
        [6, 10, 0, 12],
        [6, 12, 10, 0],
        [8, 2, 4, 14],
        [8, 4, 14, 2],
        [10, 0, 6, 12],
        [10, 6, 12, 0],
        [12, 0, 10, 6],
        [12, 6, 0, 10],
        [14, 2, 8, 4],
        [14, 4, 2, 8],
    ],
    [
        [0, 6, 10, 12],
        [0, 10, 12, 6],
        [2, 4, 8, 14],
        [2, 8, 14, 4],
        [4, 2, 14, 8],
        [4, 14, 8, 2],
        [6, 0, 12, 10],
        [6, 12, 10, 0],
        [8, 2, 4, 14],
        [8, 14, 2, 4],
        [10, 0, 6, 12],
        [10, 12, 0, 6],
        [12, 6, 0, 10],
        [12, 10, 6, 0],
        [14, 4, 2, 8],
        [14, 8, 4, 2],
    ],
    [
        [0, 6, 12, 10],
        [0, 10, 6, 12],
        [2, 4, 14, 8],
        [2, 8, 4, 14],
        [4, 2, 8, 14],
        [4, 14, 2, 8],
        [6, 0, 10, 12],
        [6, 12, 0, 10],
        [8, 2, 14, 4],
        [8, 14, 4, 2],
        [10, 0, 12, 6],
        [10, 12, 6, 0],
        [12, 6, 10, 0],
        [12, 10, 0, 6],
        [14, 4, 8, 2],
        [14, 8, 2, 4],
    ],
];

#[derive(Debug, Clone, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixRecord {
    pub color: usize,
    pub permutation: [u8; 4],
    pub boolean_factor: u8,
    pub diagonal_signs: [i8; 4],
    pub l: [[i8; 4]; 4],
    pub r: [[i8; 4]; 4],
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
    pub ranking: &'static str,
    pub bosons: Vec<String>,
    pub fermions: Vec<String>,
    pub edges: Vec<EdgeRecord>,
    pub two_color_squares: Vec<SquareRecord>,
    pub edge_count: usize,
    pub square_count: usize,
    pub all_squares_odd: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SigningRecord {
    pub appendix_b_index: usize,
    pub boolean_factors: [u8; 4],
    pub chi0: i64,
    pub garden_sparse_passed: bool,
    pub dense_entries_checked: usize,
    pub dense_residual_entries: usize,
    pub matrices: Vec<MatrixRecord>,
    pub adinkra: AdinkraRecord,
}

#[derive(Debug, Clone, Serialize)]
pub struct SectorInvariantRecord {
    /// A six-way coordinate in the fixed one-line convention.  It is the
    /// conjugation action of this V4 coset on the ordered three nonidentity
    /// elements of V4.
    pub quotient_s3_one_line: [u8; 3],
    pub quotient_s3_cycle: String,
    pub canonical_coset_ranks: Vec<u32>,
    /// Fixed-order signature from the published quartet ordering.
    pub ordered_bruhat_upper_triangle: [u8; 6],
    pub bruhat_row_sums: [u8; 4],
    pub bruhat_eigenvalues: [i16; 4],
    pub bruhat_eigenvalue_norm_squared: u16,
    /// This distinguishes the canonical signed cis/trans representative, not
    /// all six unsigned sectors.
    pub canonical_chi0: i64,
    pub canonical_gadget_row: Vec<f64>,
    pub interpretation: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SectorRecord {
    pub id: String,
    pub ordered_permutations: [[u8; 4]; 4],
    pub ordered_ranks: [u32; 4],
    pub canonical_boolean_factors: [u8; 4],
    pub canonical_matrices: Vec<MatrixRecord>,
    pub published_fiducial_signings: Vec<SigningRecord>,
    pub adinkra: AdinkraRecord,
    pub invariants: SectorInvariantRecord,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationRecord {
    pub permutation_vertices: usize,
    pub quartet_count: usize,
    pub quartet_size: usize,
    pub vertices_covered_once: bool,
    pub v4_cosets_verified: usize,
    pub published_boolean_factor_quartets: usize,
    pub equation_5_10_matrices_matched: bool,
    pub garden_representations_checked: usize,
    pub garden_representations_passed: usize,
    pub dense_garden_entries_checked: usize,
    pub dense_garden_residual_entries: usize,
    pub adinkras_constructed: usize,
    pub adinkra_edges_checked: usize,
    pub adinkra_squares_checked: usize,
    pub odd_dashing_failures: usize,
    pub distinct_quotient_s3_labels: usize,
    pub intra_bruhat_spectra_verified: usize,
    pub intra_bruhat_norm_squared: u16,
    pub maximum_inter_bruhat_norm_squared: u16,
    pub intra_norm_exceeds_every_inter_quartet: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct S4SupersymmetryArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub conventions: BTreeMap<&'static str, &'static str>,
    pub permutation_vertices: Vec<[u8; 4]>,
    pub sectors: Vec<SectorRecord>,
    pub validation: ValidationRecord,
    pub full_bc4_orbit_note: &'static str,
    pub boundary: &'static str,
}

fn permutation(values: &[u8; 4]) -> Permutation {
    Permutation::new(values).expect("published S4 address")
}

/// Decimal Boolean factor to row signs.  Bit zero controls row one.
pub fn boolean_diagonal(factor: u8) -> [i8; 4] {
    assert!(factor < 16);
    let mut result = [1i8; 4];
    for (row, value) in result.iter_mut().enumerate() {
        if factor & (1 << row) != 0 {
            *value = -1;
        }
    }
    result
}

fn build_rep(sector: usize, factors: [u8; 4]) -> AdinkraRep {
    let perms: Vec<Vec<usize>> = S4_ORDERED_QUARTETS[sector]
        .iter()
        .map(|p| p.iter().map(|&x| usize::from(x - 1)).collect())
        .collect();
    let signs: Vec<i8> = factors
        .iter()
        .flat_map(|&factor| boolean_diagonal(factor))
        .collect();
    AdinkraRep::from_parts(4, 4, &perms, &signs)
}

fn dense(signed: &crate::signed_perm::SignedPerm) -> [[i8; 4]; 4] {
    let mut matrix = [[0i8; 4]; 4];
    for row in 0..4 {
        matrix[row][usize::from(signed.perm[row])] = signed.sign[row];
    }
    matrix
}

fn transpose(matrix: [[i8; 4]; 4]) -> [[i8; 4]; 4] {
    let mut result = [[0i8; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            result[row][column] = matrix[column][row];
        }
    }
    result
}

fn multiply(left: &[[i8; 4]; 4], right: &[[i8; 4]; 4]) -> [[i16; 4]; 4] {
    let mut result = [[0i16; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            for inner in 0..4 {
                result[row][column] +=
                    i16::from(left[row][inner]) * i16::from(right[inner][column]);
            }
        }
    }
    result
}

fn dense_garden_audit(rep: &AdinkraRep) -> (usize, usize) {
    let l: Vec<_> = rep.l_matrices.iter().map(dense).collect();
    let r: Vec<_> = l.iter().copied().map(transpose).collect();
    let mut checked = 0usize;
    let mut residual = 0usize;
    for i in 0..4 {
        for j in 0..4 {
            let left = multiply(&l[i], &r[j]);
            let right = multiply(&l[j], &r[i]);
            for row in 0..4 {
                for column in 0..4 {
                    let expected = if i == j && row == column { 2 } else { 0 };
                    checked += 1;
                    residual += usize::from(left[row][column] + right[row][column] != expected);
                }
            }
        }
    }
    (checked, residual)
}

fn signed_chi0(rep: &AdinkraRep) -> i64 {
    let raw = chromochar_antisym(rep, 0, 1, 2, 3);
    assert_eq!(raw % 96, 0);
    let exact = raw / 96;
    assert_eq!(chi0_n4(rep), exact as f64);
    exact
}

fn equation_5_10_matches() -> bool {
    let rep = build_rep(0, S4_BOOLEAN_FACTOR_QUARTETS[0][11]);
    let expected = [
        [[1, 0, 0, 0], [0, 0, 0, -1], [0, 1, 0, 0], [0, 0, -1, 0]],
        [[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, -1]],
        [[0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, -1], [1, 0, 0, 0]],
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
    ];
    rep.l_matrices.iter().map(dense).eq(expected)
}

fn matrix_records(sector: usize, factors: [u8; 4], rep: &AdinkraRep) -> Vec<MatrixRecord> {
    rep.l_matrices
        .iter()
        .enumerate()
        .map(|(color, signed)| {
            let l = dense(signed);
            MatrixRecord {
                color: color + 1,
                permutation: S4_ORDERED_QUARTETS[sector][color],
                boolean_factor: factors[color],
                diagonal_signs: boolean_diagonal(factors[color]),
                l,
                r: transpose(l),
            }
        })
        .collect()
}

fn adinkra_record(rep: &AdinkraRep) -> AdinkraRecord {
    let mut edges = Vec::with_capacity(16);
    for (color, matrix) in rep.l_matrices.iter().enumerate() {
        for boson in 0..4 {
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

    let mut squares = Vec::with_capacity(12);
    for first in 0..4 {
        for second in (first + 1)..4 {
            let p = &rep.l_matrices[first];
            let q = &rep.l_matrices[second];
            let q_inv = q.inverse();
            let mut seen = BTreeSet::new();
            for boson_a in 0..4 {
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
            assert_eq!(seen.len(), 2);
        }
    }
    let all_squares_odd = squares.iter().all(|square| square.odd_dashing);
    AdinkraRecord {
        ranking: "valise: four bosons at height 0 and four fermions at height 1",
        bosons: (1..=4).map(|index| format!("B{index}")).collect(),
        fermions: (1..=4).map(|index| format!("F{index}")).collect(),
        edge_count: edges.len(),
        square_count: squares.len(),
        edges,
        two_color_squares: squares,
        all_squares_odd,
    }
}

fn quotient_action(representative: Permutation) -> [u8; 3] {
    let v4 = vierergruppe();
    let inverse = representative.inverse();
    let mut action = [0u8; 3];
    for (source, &element) in v4[1..].iter().enumerate() {
        let conjugate = representative
            .compose(element)
            .expect("S4")
            .compose(inverse)
            .expect("S4");
        action[source] = v4[1..]
            .iter()
            .position(|&candidate| candidate == conjugate)
            .map(|position| position as u8 + 1)
            .expect("V4 is normal in S4");
    }
    action
}

fn cycle_notation_s3(action: [u8; 3]) -> String {
    match action {
        [1, 2, 3] => "()".into(),
        [2, 1, 3] => "(12)".into(),
        [3, 2, 1] => "(13)".into(),
        [1, 3, 2] => "(23)".into(),
        [2, 3, 1] => "(123)".into(),
        [3, 1, 2] => "(132)".into(),
        _ => panic!("not an S3 permutation: {action:?}"),
    }
}

fn upper_triangle(matrix: [[u8; 4]; 4]) -> [u8; 6] {
    [
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][2],
        matrix[1][3],
        matrix[2][3],
    ]
}

fn row_sums(matrix: [[u8; 4]; 4]) -> [u8; 4] {
    matrix.map(|row| row.iter().copied().sum())
}

fn bruhat_norm_squared(matrix: [[u8; 4]; 4]) -> u16 {
    matrix
        .iter()
        .flatten()
        .map(|&value| u16::from(value).pow(2))
        .sum()
}

fn determinant4(matrix: [[i16; 4]; 4]) -> i64 {
    let mut determinant = 0i64;
    for permutation in permutations(4).expect("S4") {
        let columns = permutation.as_slice();
        let sign = if permutation.inversion_count() % 2 == 0 {
            1i64
        } else {
            -1i64
        };
        let product = (0..4)
            .map(|row| i64::from(matrix[row][usize::from(columns[row] - 1)]))
            .product::<i64>();
        determinant += sign * product;
    }
    determinant
}

fn has_published_intra_spectrum(matrix: [[u8; 4]; 4]) -> bool {
    [12i16, 0, -4, -8].iter().all(|&eigenvalue| {
        let shifted = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                if row == column {
                    eigenvalue - i16::from(matrix[row][column])
                } else {
                    -i16::from(matrix[row][column])
                }
            })
        });
        determinant4(shifted) == 0
    })
}

pub fn build() -> S4SupersymmetryArtifact {
    let vertices: Vec<[u8; 4]> = permutations(4)
        .expect("S4")
        .map(|p| p.as_slice().try_into().expect("S4 width"))
        .collect();
    let v4 = vierergruppe();
    let cosets = coset_partition(&v4, CosetSide::Right).expect("V4 cosets");

    let canonical_reps: Vec<_> = (0..6)
        .map(|sector| build_rep(sector, S4_BOOLEAN_FACTOR_QUARTETS[sector][0]))
        .collect();
    let canonical_holoraumy: Vec<_> = canonical_reps.iter().map(HoloraumyData::from_rep).collect();
    let canonical_gadget: Vec<Vec<f64>> = canonical_holoraumy
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

    let mut garden_passed = 0usize;
    let mut dense_checked = 0usize;
    let mut dense_residual = 0usize;
    let mut adinkras_constructed = 0usize;
    let mut adinkra_edges_checked = 0usize;
    let mut adinkra_squares_checked = 0usize;
    let mut odd_dashing_failures = 0usize;
    let mut sectors = Vec::with_capacity(6);
    let mut quotient_labels = BTreeSet::new();
    let mut intra_bruhat_spectra_verified = 0usize;

    for sector in 0..6 {
        let quartet = S4_ORDERED_QUARTETS[sector];
        let ranks = quartet.map(|entry| permutation(&entry).rank() as u32);
        let sorted_ranks: Vec<u32> = {
            let mut ranks = ranks.to_vec();
            ranks.sort_unstable();
            ranks
        };
        assert!(cosets.slices.contains(&sorted_ranks));

        let mut signing_records = Vec::with_capacity(16);
        for (index, &factors) in S4_BOOLEAN_FACTOR_QUARTETS[sector].iter().enumerate() {
            let rep = build_rep(sector, factors);
            let sparse = rep.verify_garden_algebra();
            let (checked, residual) = dense_garden_audit(&rep);
            garden_passed += usize::from(sparse && residual == 0);
            dense_checked += checked;
            dense_residual += residual;
            let matrices = matrix_records(sector, factors, &rep);
            let adinkra = adinkra_record(&rep);
            adinkras_constructed += 1;
            adinkra_edges_checked += adinkra.edge_count;
            adinkra_squares_checked += adinkra.square_count;
            odd_dashing_failures += adinkra
                .two_color_squares
                .iter()
                .filter(|square| !square.odd_dashing)
                .count();
            signing_records.push(SigningRecord {
                appendix_b_index: index + 1,
                boolean_factors: factors,
                chi0: signed_chi0(&rep),
                garden_sparse_passed: sparse,
                dense_entries_checked: checked,
                dense_residual_entries: residual,
                matrices,
                adinkra,
            });
        }

        let representative = permutation(&quartet[0]);
        let action = quotient_action(representative);
        quotient_labels.insert(action);
        let correlator = S4_INTRA_CORRELATORS[sector].values;
        intra_bruhat_spectra_verified += usize::from(has_published_intra_spectrum(correlator));
        let canonical_factors = S4_BOOLEAN_FACTOR_QUARTETS[sector][0];
        let canonical_rep = &canonical_reps[sector];
        let adinkra = adinkra_record(canonical_rep);
        sectors.push(SectorRecord {
            id: format!("P{}", sector + 1),
            ordered_permutations: quartet,
            ordered_ranks: ranks,
            canonical_boolean_factors: canonical_factors,
            canonical_matrices: matrix_records(sector, canonical_factors, canonical_rep),
            published_fiducial_signings: signing_records,
            adinkra,
            invariants: SectorInvariantRecord {
                quotient_s3_one_line: action,
                quotient_s3_cycle: cycle_notation_s3(action),
                canonical_coset_ranks: sorted_ranks,
                ordered_bruhat_upper_triangle: upper_triangle(correlator),
                bruhat_row_sums: row_sums(correlator),
                bruhat_eigenvalues: [12, 0, -4, -8],
                bruhat_eigenvalue_norm_squared: bruhat_norm_squared(correlator),
                canonical_chi0: signed_chi0(canonical_rep),
                canonical_gadget_row: canonical_gadget[sector].clone(),
                interpretation: "The V4 coset is the exact unsigned sector. With the stored ordering of the three nonidentity V4 elements, its S4/V4 action supplies a distinct S3 coordinate for each sector. The ordered Bruhat word makes the published ordering recognizable. chi0 and the gadget row classify the selected signed representative and are not six-way unsigned labels.",
            },
        });
    }

    let mut covered = BTreeMap::<u32, usize>::new();
    for rank in S4_ORDERED_QUARTETS
        .iter()
        .flatten()
        .map(|entry| permutation(entry).rank() as u32)
    {
        *covered.entry(rank).or_default() += 1;
    }
    let vertices_covered_once =
        covered.len() == 24 && covered.values().all(|&multiplicity| multiplicity == 1);
    let equation_5_10_matrices_matched = equation_5_10_matches();
    let intra_bruhat_norm_squared = bruhat_norm_squared(S4_INTRA_CORRELATORS[0].values);
    let maximum_inter_bruhat_norm_squared = S4_INTER_CORRELATORS
        .iter()
        .map(|block| bruhat_norm_squared(block.values))
        .max()
        .expect("15 inter-quartet blocks");
    let intra_norm_exceeds_every_inter_quartet =
        intra_bruhat_norm_squared > maximum_inter_bruhat_norm_squared;
    let passed = vertices_covered_once
        && quotient_labels.len() == 6
        && equation_5_10_matrices_matched
        && intra_bruhat_spectra_verified == 6
        && intra_bruhat_norm_squared == 224
        && maximum_inter_bruhat_norm_squared == 208
        && intra_norm_exceeds_every_inter_quartet
        && garden_passed == 96
        && dense_residual == 0
        && odd_dashing_failures == 0
        && sectors.iter().all(|sector| {
            sector.published_fiducial_signings.iter().all(|signing| {
                signing.adinkra.edge_count == 16
                    && signing.adinkra.square_count == 12
                    && signing.adinkra.all_squares_odd
            })
        });

    let mut conventions = BTreeMap::new();
    conventions.insert(
        "permutation",
        "one-line p=[p(1),p(2),p(3),p(4)]; matrix P has P[row,p(row)]=1",
    );
    conventions.insert(
        "boolean_factor",
        "decimal b in 0..15; bit zero controls row one; S=diag((-1)^bit_r(b))",
    );
    conventions.insert("signed_matrix", "L_I=S_I P_I and R_I=L_I^T");
    conventions.insert("garden_algebra", "L_I R_J + L_J R_I = 2 delta_IJ I_4");
    conventions.insert(
        "dashing",
        "positive matrix entry is solid; negative matrix entry is dashed",
    );

    S4SupersymmetryArtifact {
        schema_version: SCHEMA_VERSION,
        title: "Six supersymmetric sectors of the four-color permutahedron",
        sources: vec![
            SourceRecord {
                arxiv_id: "2012.13308",
                locator: "Fig. 2, Tables 1-6, and Sec. 8",
                role: "24 vertices, six ordered quartets, Bruhat correlators",
            },
            SourceRecord {
                arxiv_id: "1701.00304",
                locator: "Eqs. (5.7)-(5.10) and Appendix B",
                role: "ordered permutation quartets, Boolean factors, signed L-matrices",
            },
            SourceRecord {
                arxiv_id: "1210.0478",
                locator: "Secs. 2-4",
                role: "BC4 signed permutations and Garden algebra organization",
            },
            SourceRecord {
                arxiv_id: "1712.07826",
                locator: "Secs. 2-4",
                role: "complete 36,864 four-color valise Adinkra library",
            },
        ],
        conventions,
        permutation_vertices: vertices,
        sectors,
        validation: ValidationRecord {
            permutation_vertices: 24,
            quartet_count: 6,
            quartet_size: 4,
            vertices_covered_once,
            v4_cosets_verified: cosets.slice_count,
            published_boolean_factor_quartets: 96,
            equation_5_10_matrices_matched,
            garden_representations_checked: 96,
            garden_representations_passed: garden_passed,
            dense_garden_entries_checked: dense_checked,
            dense_garden_residual_entries: dense_residual,
            adinkras_constructed,
            adinkra_edges_checked,
            adinkra_squares_checked,
            odd_dashing_failures,
            distinct_quotient_s3_labels: quotient_labels.len(),
            intra_bruhat_spectra_verified,
            intra_bruhat_norm_squared,
            maximum_inter_bruhat_norm_squared,
            intra_norm_exceeds_every_inter_quartet,
            passed,
        },
        full_bc4_orbit_note: "The 96 Appendix-B fiducial signings are seeds. Independent color complements and color permutations generate the full ordered 36,864-member BC4 valise library: 6 sectors x 16 fiducial signings x 16 color complements x 24 color orders.",
        boundary: "This is a complete verification of the six unsigned S4 sectors and the 96 published fiducial signed quartets. The JSON contains the matrices and Adinkra graph for every fiducial signing. It does not serialize all 36,864 color-complement and color-order variants because those are generated group orbits of the stored seeds.",
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
        BufWriter::new(File::create(data_path).expect("create S4 supersymmetry artifact")),
        &artifact,
    )
    .expect("write S4 supersymmetry artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create S4 validation artifact")),
        &artifact.validation,
    )
    .expect("write S4 validation artifact");
    artifact.validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_boolean_bit_convention_matches_eq_5_10() {
        assert_eq!(boolean_diagonal(10), [1, -1, 1, -1]);
        assert_eq!(boolean_diagonal(12), [1, 1, -1, -1]);
        assert_eq!(boolean_diagonal(6), [1, -1, -1, 1]);
        assert_eq!(boolean_diagonal(0), [1, 1, 1, 1]);
        assert!(equation_5_10_matches());
    }

    #[test]
    fn all_six_sectors_and_ninety_six_signings_verify() {
        let artifact = build();
        assert!(artifact.validation.passed);
        assert_eq!(artifact.validation.permutation_vertices, 24);
        assert_eq!(artifact.validation.quartet_count, 6);
        assert_eq!(artifact.validation.garden_representations_checked, 96);
        assert_eq!(artifact.validation.garden_representations_passed, 96);
        assert_eq!(artifact.validation.dense_garden_residual_entries, 0);
        assert_eq!(artifact.validation.adinkras_constructed, 96);
        assert_eq!(artifact.validation.adinkra_edges_checked, 1_536);
        assert_eq!(artifact.validation.adinkra_squares_checked, 1_152);
        assert_eq!(artifact.validation.odd_dashing_failures, 0);
        assert_eq!(artifact.validation.distinct_quotient_s3_labels, 6);
        assert_eq!(artifact.validation.intra_bruhat_spectra_verified, 6);
        assert_eq!(artifact.validation.intra_bruhat_norm_squared, 224);
        assert_eq!(artifact.validation.maximum_inter_bruhat_norm_squared, 208);
        assert!(artifact.validation.intra_norm_exceeds_every_inter_quartet);
    }

    #[test]
    fn every_published_signing_has_unit_chi0() {
        let artifact = build();
        for sector in &artifact.sectors {
            for signing in &sector.published_fiducial_signings {
                assert!(
                    signing.chi0.abs() == 1,
                    "{} signing {} has chi0={}",
                    sector.id,
                    signing.appendix_b_index,
                    signing.chi0
                );
            }
        }
    }

    #[test]
    fn quotient_label_is_constant_across_each_coset() {
        for quartet in S4_ORDERED_QUARTETS {
            let expected = quotient_action(permutation(&quartet[0]));
            for member in quartet {
                assert_eq!(quotient_action(permutation(&member)), expected);
            }
        }
    }
}
