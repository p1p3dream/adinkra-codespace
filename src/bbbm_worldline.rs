#![allow(dead_code)]

//! One-dimensional reduction of the BBBM component transformations.
//!
//! This module starts with Eqs. (22)-(23) of Baulieu, Berkovits, Bossard,
//! and Martin, arXiv:0705.2002v3.  It sets the eight transverse derivatives
//! and `partial_9` to zero and normalizes `partial_+ = partial_- = D`.  In the
//! abelian linearized theory the two light-cone potentials then have the same
//! gauge shift.  Their difference
//!
//!     B = A_- - A_+
//!
//! is gauge invariant; the sum is the temporal gauge direction.  Keeping
//! `A_i (i=1..8)`, `B`, and the seven antiselfdual auxiliaries gives the
//! `(9,16,7)` worldline hanging.
//!
//! The important qualification is algebraic.  Temporal gauge removes the
//! nonzero modes of the sum only after solving `D alpha = -(A_+ + A_-)/2`.
//! Likewise, lowering the seven auxiliary nodes requires `D^{-1}`.  Thus the
//! BBBM hanging is the local node-raised image of a 16|16 valise, but the
//! inverse map is not an isomorphism over the local polynomial ring `Z[D]`.
//! The zero modes must be removed or boundary data supplied before the formal
//! valise can be identified with the original potential multiplet.

use serde::Serialize;

/// Matrix dimension on each side of the reduced 16|16 multiplet.
pub const D: usize = 16;
/// Number of closing BBBM supercharges.
pub const N: usize = 9;

/// Integer polynomial of degree at most two in the worldline derivative.
/// Degree two is retained so products can be checked without truncation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DPoly {
    pub coefficient: [i16; 3],
}

impl DPoly {
    const fn monomial(degree: usize, value: i16) -> Self {
        let mut coefficient = [0; 3];
        coefficient[degree] = value;
        Self { coefficient }
    }

    fn add(self, rhs: Self) -> Self {
        let mut out = Self::default();
        for degree in 0..3 {
            out.coefficient[degree] = self.coefficient[degree] + rhs.coefficient[degree];
        }
        out
    }

    fn multiply(self, rhs: Self) -> Self {
        let mut out = Self::default();
        for left_degree in 0..3 {
            for right_degree in 0..3 {
                if self.coefficient[left_degree] == 0 || rhs.coefficient[right_degree] == 0 {
                    continue;
                }
                let degree = left_degree + right_degree;
                assert!(
                    degree < 3,
                    "unexpected derivative degree in BBBM closure check"
                );
                out.coefficient[degree] +=
                    self.coefficient[left_degree] * rhs.coefficient[right_degree];
            }
        }
        out
    }
}

type IntMatrix = [[i8; D]; D];
type PolyMatrix = [[DPoly; D]; D];

/// The gauge quotient used in the reduction, including the zero-mode caveat.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalGaugeQuotient {
    /// Gauge-invariant ninth physical boson.
    pub retained_combination: &'static str,
    /// Combination shifted by `2 D alpha`.
    pub gauge_combination: &'static str,
    /// A local compensator cannot solve temporal gauge on the potential.
    pub compensator_requires_inverse_derivative: bool,
    /// `ker D` is not removed by a gauge parameter through `D alpha`.
    pub temporal_zero_mode_survives: bool,
}

pub const TEMPORAL_GAUGE_QUOTIENT: TemporalGaugeQuotient = TemporalGaugeQuotient {
    retained_combination: "B = A_- - A_+",
    gauge_combination: "T = A_- + A_+",
    compensator_requires_inverse_derivative: true,
    temporal_zero_mode_survives: true,
};

/// Result of asking whether the node-raised BBBM hanging is locally equivalent
/// to a valise.  The forward map `g_a -> G_a = D g_a` is local.  Its inverse is
/// not a polynomial differential operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalValiseEquivalence {
    Obstructed {
        raised_auxiliary_nodes: usize,
        required_inverse: &'static str,
        unresolved_kernel: &'static str,
    },
}

pub const LOCAL_VALISE_EQUIVALENCE: LocalValiseEquivalence = LocalValiseEquivalence::Obstructed {
    raised_auxiliary_nodes: 7,
    required_inverse: "g_a = D^{-1} G_a",
    unresolved_kernel: "seven integration constants in ker D, plus the temporal gauge zero mode",
};

// Fano triples fixing the octonion convention used for the Spin(7) seven.
// They agree with the convention already used by the BBBM closure module.
const FANO: [(usize, usize, usize); 7] = [
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
];

fn octonion_structure(a: usize, b: usize, c: usize) -> i8 {
    let one_based = (a + 1, b + 1, c + 1);
    for &(x, y, z) in &FANO {
        if one_based == (x, y, z) || one_based == (y, z, x) || one_based == (z, x, y) {
            return 1;
        }
        if one_based == (y, x, z) || one_based == (x, z, y) || one_based == (z, y, x) {
            return -1;
        }
    }
    0
}

/// Coefficient of `chi_a` in the antiselfdual two-form `chi_ij`, using the
/// paper's octonionic basis `chi_{8a}=chi_a`, `chi_{ab}=C_ab^c chi_c`.
fn antiselfdual_coefficient(i: usize, j: usize, a: usize) -> i8 {
    debug_assert!(i < 8 && j < 8 && a < 7);
    match (i, j) {
        _ if i == j => 0,
        (7, j) if j < 7 => i8::from(j == a),
        (i, 7) if i < 7 => -i8::from(i == a),
        _ => octonion_structure(i, j, a),
    }
}

/// Constant formal-valise matrices in the field order
///
/// bosons:  `(A_1,...,A_8, B, g_1,...,g_7)`
/// fermions: `(psi_1,...,psi_8, eta, chi_1,...,chi_7)`.
///
/// Color zero is `delta_0`; colors one through eight are `delta_i`.  A matrix
/// entry `L[q][b][f]` is the coefficient in `delta_q boson_b`.  Fermion
/// transformations are `D R[q]`, where `R[0]=I` and `R[i]=-L[i]` for `i>0`.
/// These entries are obtained from the reduced component transformations, not
/// from the generic `[9,4]` scaffold.
pub fn formal_valise_matrices() -> ([IntMatrix; N], [IntMatrix; N]) {
    let mut l = [[[0i8; D]; D]; N];
    let mut r = [[[0i8; D]; D]; N];

    for field in 0..D {
        l[0][field][field] = 1;
        r[0][field][field] = 1;
    }

    for k in 0..8 {
        let color = k + 1;

        // delta_k A_j = -delta_kj eta - chi_kj
        for j in 0..8 {
            l[color][j][8] = -i8::from(k == j);
            for a in 0..7 {
                l[color][j][9 + a] = -antiselfdual_coefficient(k, j, a);
            }
        }

        // delta_k B = psi_k, from delta_k(A_- - A_+) = 0 - (-psi_k).
        l[color][8][k] = 1;

        // delta_k G_a = D sum_j C(k,j,a) psi_j.  Formally G_a=D g_a.
        for a in 0..7 {
            for j in 0..8 {
                l[color][9 + a][j] = antiselfdual_coefficient(k, j, a);
            }
        }

        // Encode the reduced fermion rules independently rather than filling
        // them from a Garden-algebra ansatz.
        for j in 0..8 {
            // delta_k psi_j = D(delta_kj B + C(k,j,a) g_a).
            r[color][j][8] = i8::from(k == j);
            for a in 0..7 {
                r[color][j][9 + a] = antiselfdual_coefficient(k, j, a);
            }
        }
        // delta_k eta = -D A_k.
        r[color][8][k] = -1;
        // delta_k chi_a = -D sum_j C(k,j,a) A_j.
        for a in 0..7 {
            for j in 0..8 {
                r[color][9 + a][j] = -antiselfdual_coefficient(k, j, a);
            }
        }
    }

    (l, r)
}

/// Polynomial linkage matrices for the actual `(9,16,7)` hanging.
///
/// `boson_to_fermion[q][b][f]` encodes `delta_q boson_b`; the last seven
/// bosons are `G_a`, not `g_a`.  `fermion_to_boson[q][f][b]` encodes
/// `delta_q fermion_f`.  All powers are nonnegative, so this representation is
/// local even though its inverse node-lowering map is not.
pub struct HangingLinkages {
    pub boson_to_fermion: [PolyMatrix; N],
    pub fermion_to_boson: [PolyMatrix; N],
}

pub fn hanging_linkages() -> HangingLinkages {
    let (l, r) = formal_valise_matrices();
    let mut boson_to_fermion = [[[DPoly::default(); D]; D]; N];
    let mut fermion_to_boson = [[[DPoly::default(); D]; D]; N];

    for q in 0..N {
        for b in 0..D {
            let raised = usize::from(b >= 9);
            for f in 0..D {
                boson_to_fermion[q][b][f] = DPoly::monomial(raised, i16::from(l[q][b][f]));
                fermion_to_boson[q][f][b] = DPoly::monomial(1 - raised, i16::from(r[q][f][b]));
            }
        }
    }

    HangingLinkages {
        boson_to_fermion,
        fermion_to_boson,
    }
}

fn multiply_poly_matrices(left: &PolyMatrix, right: &PolyMatrix) -> PolyMatrix {
    let mut out = [[DPoly::default(); D]; D];
    for row in 0..D {
        for col in 0..D {
            for middle in 0..D {
                out[row][col] = out[row][col].add(left[row][middle].multiply(right[middle][col]));
            }
        }
    }
    out
}

fn add_poly_matrices(left: &PolyMatrix, right: &PolyMatrix) -> PolyMatrix {
    let mut out = [[DPoly::default(); D]; D];
    for row in 0..D {
        for col in 0..D {
            out[row][col] = left[row][col].add(right[row][col]);
        }
    }
    out
}

/// Exact coefficient-by-coefficient Garden closure on the hanging.
pub fn verify_hanging_closure(linkages: &HangingLinkages) -> bool {
    for a in 0..N {
        for b in 0..N {
            let bosonic = add_poly_matrices(
                &multiply_poly_matrices(
                    &linkages.boson_to_fermion[a],
                    &linkages.fermion_to_boson[b],
                ),
                &multiply_poly_matrices(
                    &linkages.boson_to_fermion[b],
                    &linkages.fermion_to_boson[a],
                ),
            );
            let fermionic = add_poly_matrices(
                &multiply_poly_matrices(
                    &linkages.fermion_to_boson[a],
                    &linkages.boson_to_fermion[b],
                ),
                &multiply_poly_matrices(
                    &linkages.fermion_to_boson[b],
                    &linkages.boson_to_fermion[a],
                ),
            );

            for row in 0..D {
                for col in 0..D {
                    let expected = if a == b && row == col {
                        DPoly::monomial(1, 2)
                    } else {
                        DPoly::default()
                    };
                    if bosonic[row][col] != expected || fermionic[row][col] != expected {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn dense_multiply(left: &IntMatrix, right: &IntMatrix) -> IntMatrix {
    let mut out = [[0i8; D]; D];
    for row in 0..D {
        for col in 0..D {
            let mut value = 0i16;
            for middle in 0..D {
                value += i16::from(left[row][middle]) * i16::from(right[middle][col]);
            }
            out[row][col] = i8::try_from(value).expect("Clifford product coefficient fits i8");
        }
    }
    out
}

/// Doubly-even stabilizer code of the formal valise, represented as 9-bit
/// words in the actual BBBM color order `[delta_0,delta_1,...,delta_8]`.
pub fn formal_valise_stabilizer_code() -> Vec<u16> {
    let (l, _) = formal_valise_matrices();
    let mut words = Vec::new();
    for mask in 0u16..(1u16 << N) {
        if mask.count_ones() % 2 != 0 {
            continue;
        }
        let mut product = [[0i8; D]; D];
        for i in 0..D {
            product[i][i] = 1;
        }
        for (color, matrix) in l.iter().enumerate() {
            if mask & (1 << color) != 0 {
                product = dense_multiply(&product, matrix);
            }
        }
        let underlying_permutation_is_identity = (0..D).all(|row| {
            (0..D).all(|col| {
                if row == col {
                    product[row][col].unsigned_abs() == 1
                } else {
                    product[row][col] == 0
                }
            })
        });
        if underlying_permutation_is_identity {
            words.push(mask);
        }
    }
    words
}

/// Map actual BBBM colors to the scaffold convention: the eight vector charges
/// become scaffold colors 0..7 and the scalar charge becomes color 8.
fn actual_to_scaffold_colors(word: u16) -> u16 {
    let mut mapped = 0u16;
    if word & 1 != 0 {
        mapped |= 1 << 8;
    }
    for actual in 1..9 {
        if word & (1 << actual) != 0 {
            mapped |= 1 << (actual - 1);
        }
    }
    mapped
}

/// The `[9,4]` scaffold code generated in `bbbm.rs`, computed independently.
fn scaffold_code() -> Vec<u16> {
    let generators = [0b1110_0001u16, 0b1101_0010, 0b1011_0100, 0b0111_1000];
    let mut words = vec![0u16];
    for generator in generators {
        let old = words.clone();
        words.extend(old.into_iter().map(|word| word ^ generator));
    }
    words.sort_unstable();
    words
}

/// Invariant comparison to the generic scaffold.  This proves equality of the
/// underlying chromotopology code after the explicit color permutation.  It
/// does not identify temporal or auxiliary zero modes and therefore is not
/// advertised as a local field-space intertwiner.
pub fn chromotopology_matches_scaffold() -> bool {
    let mut reduced_code: Vec<u16> = formal_valise_stabilizer_code()
        .into_iter()
        .map(actual_to_scaffold_colors)
        .collect();
    reduced_code.sort_unstable();
    reduced_code == scaffold_code()
}

/// Machine-readable summary of the actual reduction and its locality boundary.
#[derive(Debug, Serialize)]
pub struct WorldlineReductionReport {
    pub source: &'static str,
    pub retained_hanging: &'static str,
    pub hanging_closes_exactly_over_polynomial_differential_operators: bool,
    pub formal_node_lowered_matrices_are_signed_permutations: bool,
    pub formal_chromotopology_matches_scaffold_after_color_permutation: bool,
    pub strict_local_valise_equivalence: bool,
    pub obstruction: &'static str,
    pub zero_modes_not_identified: bool,
}

pub fn run() -> WorldlineReductionReport {
    let (l, r) = formal_valise_matrices();
    let signed_permutations = l.iter().chain(r.iter()).all(|matrix| {
        (0..D).all(|row| matrix[row].iter().filter(|&&value| value != 0).count() == 1)
            && (0..D).all(|col| (0..D).filter(|&row| matrix[row][col] != 0).count() == 1)
    });
    WorldlineReductionReport {
        source: "BBBM Eqs. (22)-(23) after one-dimensional reduction and gauge quotient",
        retained_hanging: "(9,16,7)",
        hanging_closes_exactly_over_polynomial_differential_operators: verify_hanging_closure(
            &hanging_linkages(),
        ),
        formal_node_lowered_matrices_are_signed_permutations: signed_permutations,
        formal_chromotopology_matches_scaffold_after_color_permutation:
            chromotopology_matches_scaffold(),
        strict_local_valise_equivalence: false,
        obstruction: "node lowering requires D^{-1}; temporal gauge fixing and seven auxiliary integrations leave zero modes",
        zero_modes_not_identified: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauge_quotient_states_the_nonlocal_step_and_zero_mode() {
        assert_eq!(
            TEMPORAL_GAUGE_QUOTIENT.retained_combination,
            "B = A_- - A_+"
        );
        assert!(TEMPORAL_GAUGE_QUOTIENT.compensator_requires_inverse_derivative);
        assert!(TEMPORAL_GAUGE_QUOTIENT.temporal_zero_mode_survives);
    }

    #[test]
    fn reduced_rules_have_the_published_9_16_7_heights() {
        let h = hanging_linkages();

        // delta_0 A_i=psi_i, delta_0 B=eta, delta_0 G_a=D chi_a.
        for i in 0..9 {
            assert_eq!(h.boson_to_fermion[0][i][i], DPoly::monomial(0, 1));
        }
        for a in 0..7 {
            assert_eq!(h.boson_to_fermion[0][9 + a][9 + a], DPoly::monomial(1, 1));
            assert_eq!(h.fermion_to_boson[0][9 + a][9 + a], DPoly::monomial(0, 1));
        }

        // delta_k B=psi_k and delta_k eta=-D A_k.
        for k in 0..8 {
            assert_eq!(h.boson_to_fermion[k + 1][8][k], DPoly::monomial(0, 1));
            assert_eq!(h.fermion_to_boson[k + 1][8][k], DPoly::monomial(1, -1));
        }
    }

    #[test]
    fn formal_matrices_are_signed_permutations_and_close() {
        let (l, r) = formal_valise_matrices();
        for q in 0..N {
            for row in 0..D {
                assert_eq!(l[q][row].iter().filter(|&&v| v != 0).count(), 1);
                assert_eq!(r[q][row].iter().filter(|&&v| v != 0).count(), 1);
            }
            for col in 0..D {
                assert_eq!((0..D).filter(|&row| l[q][row][col] != 0).count(), 1);
                assert_eq!((0..D).filter(|&row| r[q][row][col] != 0).count(), 1);
            }
        }

        for a in 0..N {
            for b in 0..N {
                let lr_ab = dense_multiply(&l[a], &r[b]);
                let lr_ba = dense_multiply(&l[b], &r[a]);
                let rl_ab = dense_multiply(&r[a], &l[b]);
                let rl_ba = dense_multiply(&r[b], &l[a]);
                for row in 0..D {
                    for col in 0..D {
                        let expected = i8::from(a == b && row == col) * 2;
                        assert_eq!(lr_ab[row][col] + lr_ba[row][col], expected);
                        assert_eq!(rl_ab[row][col] + rl_ba[row][col], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn actual_hanging_closes_as_polynomial_differential_operators() {
        assert!(verify_hanging_closure(&hanging_linkages()));
    }

    #[test]
    fn lowering_the_auxiliaries_is_not_a_local_equivalence() {
        assert_eq!(
            LOCAL_VALISE_EQUIVALENCE,
            LocalValiseEquivalence::Obstructed {
                raised_auxiliary_nodes: 7,
                required_inverse: "g_a = D^{-1} G_a",
                unresolved_kernel: "seven integration constants in ker D, plus the temporal gauge zero mode",
            }
        );
    }

    #[test]
    fn formal_valise_has_the_scaffold_chromotopology_after_explicit_color_map() {
        let code = formal_valise_stabilizer_code();
        assert_eq!(code.len(), 16);
        assert!(code.iter().all(|word| word.count_ones() % 4 == 0));
        assert!(chromotopology_matches_scaffold());
    }

    #[test]
    fn report_preserves_the_locality_boundary() {
        let report = run();
        assert!(report.hanging_closes_exactly_over_polynomial_differential_operators);
        assert!(report.formal_node_lowered_matrices_are_signed_permutations);
        assert!(report.formal_chromotopology_matches_scaffold_after_color_permutation);
        assert!(!report.strict_local_valise_equivalence);
        assert!(report.zero_modes_not_identified);
    }
}
