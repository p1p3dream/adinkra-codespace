#![allow(dead_code)] // primitive-library module: much of its API surface is exercised by the test suite, not the binary main path

/// Holoraumy tensors and gadget inner products for Adinkra representations.
///
/// Computes the bosonic holoraumy matrices V_IJ = L_I * R_J and the fermionic
/// holoraumy matrices Vtilde_IJ = R_I * L_J for all color pairs I > J, then
/// evaluates the gadget inner product G[R, R'] between representations.

use crate::lr_matrix::AdinkraRep;
use crate::signed_perm::SignedPerm;

#[derive(Debug, Clone)]
pub struct HoloraumyData {
    pub n: usize,
    pub d: usize,
    /// Fermionic holoraumy matrices Vtilde_IJ = R_I * L_J, indexed by pair_index(I, J).
    pub vtilde: Vec<SignedPerm>,
    /// Bosonic holoraumy matrices V_IJ = L_I * R_J, indexed by pair_index(I, J).
    pub v: Vec<SignedPerm>,
}

/// Maps a pair (i, j) with i > j to a linear index in 0..C(N,2).
fn pair_index(i: usize, j: usize) -> usize {
    i * (i - 1) / 2 + j
}

/// Minimal Clifford module dimension via Bott periodicity.
///
/// Uses the modset formula:
///   modset = |4 - (n % 8)|
///   pow = f(modset, n)
///   dmin = 2^pow
///
/// Verified values:
///   dmin(1)=1, dmin(2)=2, dmin(3)=4, dmin(4)=4, dmin(5)=8,
///   dmin(6)=8, dmin(7)=8, dmin(8)=8, dmin(9)=16, dmin(10)=32,
///   dmin(12)=64, dmin(16)=128, dmin(32)=32768
pub fn dmin(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let rem = n % 8;
    let modset = if rem <= 4 { 4 - rem } else { rem - 4 };
    let pow = match modset {
        0 => n / 2,
        1 => (n + 1) / 2,
        2 => n / 2,
        3 => (n - 1) / 2,
        4 => (n - 2) / 2,
        _ => unreachable!(),
    };
    1usize << pow
}

impl HoloraumyData {
    /// Compute all holoraumy matrices from an Adinkra representation.
    ///
    /// For each pair I > J:
    ///   Vtilde_IJ = R_I * L_J = L_I^{-1} * L_J
    ///   V_IJ      = L_I * R_J = L_I * L_J^{-1}
    pub fn from_rep(rep: &AdinkraRep) -> Self {
        let n = rep.n;
        let d = rep.d;
        let num_pairs = n * (n - 1) / 2;
        let mut vtilde = Vec::with_capacity(num_pairs);
        let mut v = Vec::with_capacity(num_pairs);

        for i in 1..n {
            for j in 0..i {
                vtilde.push(rep.l_matrices[i].inverse().compose(&rep.l_matrices[j]));
                v.push(rep.l_matrices[i].compose(&rep.l_matrices[j].inverse()));
            }
        }

        HoloraumyData { n, d, vtilde, v }
    }
}

/// Compute the gadget inner product between two Adinkra representations.
///
/// G[R, R'] = -2 / (N * (N-1) * dmin(N)) * sum_{I>J} Tr(Vtilde^R_IJ * Vtilde^{R'}_IJ)
///
/// The negative sign is mandatory for stripped real holoraumy.
///
/// Proof that self-gadget equals d/dmin:
///   V_IJ^2 = -I_d (from Garden algebra), so Tr(V_IJ^2) = -d.
///   Self-gadget sum = C(N,2) * (-d) = -N(N-1)d/2.
///   G[R,R] = -2 / (N(N-1) * dmin) * (-N(N-1)d/2) = d / dmin.
///   For irreducible representations (d = dmin): G[R,R] = 1.
pub fn gadget(a: &HoloraumyData, b: &HoloraumyData) -> f64 {
    assert_eq!(a.n, b.n);
    assert_eq!(a.d, b.d);
    let n = a.n;
    let d = a.d;
    let _ = d; // used in the derivation; dmin handles normalization
    let mut sum: i64 = 0;
    for idx in 0..a.vtilde.len() {
        sum += a.vtilde[idx].trace_product(&b.vtilde[idx]);
    }
    let dmin_val = dmin(n);
    -2.0 * sum as f64 / (n * (n - 1) * dmin_val) as f64
}

/// Compute the full gadget matrix for a collection of representations.
///
/// Returns a symmetric matrix where mat[i][j] = G[reps[i], reps[j]].
pub fn gadget_matrix(reps: &[HoloraumyData]) -> Vec<Vec<f64>> {
    let n = reps.len();
    let mut mat = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let g = gadget(&reps[i], &reps[j]);
            mat[i][j] = g;
            mat[j][i] = g;
        }
    }
    mat
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helper constructors ---------------------------------------------------

    fn cs_n4() -> AdinkraRep {
        let l = vec![
            SignedPerm::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1]).unwrap(),
            SignedPerm::from_parts(vec![1, 0, 3, 2], vec![1, -1, 1, -1]).unwrap(),
            SignedPerm::from_parts(vec![2, 3, 0, 1], vec![1, -1, -1, 1]).unwrap(),
            SignedPerm::from_parts(vec![3, 2, 1, 0], vec![1, 1, -1, -1]).unwrap(),
        ];
        AdinkraRep { n: 4, d: 4, l_matrices: l }
    }

    fn vs_n4() -> AdinkraRep {
        let l = vec![
            SignedPerm::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1]).unwrap(),
            SignedPerm::from_parts(vec![1, 0, 3, 2], vec![1, -1, -1, 1]).unwrap(),
            SignedPerm::from_parts(vec![2, 3, 0, 1], vec![1, 1, -1, -1]).unwrap(),
            SignedPerm::from_parts(vec![3, 2, 1, 0], vec![1, -1, 1, -1]).unwrap(),
        ];
        AdinkraRep { n: 4, d: 4, l_matrices: l }
    }

    // -- dmin ------------------------------------------------------------------

    #[test]
    fn test_dmin() {
        assert_eq!(dmin(0), 1);
        assert_eq!(dmin(1), 1);
        assert_eq!(dmin(2), 2);
        assert_eq!(dmin(3), 4);
        assert_eq!(dmin(4), 4);
        assert_eq!(dmin(5), 8);
        assert_eq!(dmin(6), 8);
        assert_eq!(dmin(7), 8);
        assert_eq!(dmin(8), 8);
        assert_eq!(dmin(9), 16);
        assert_eq!(dmin(10), 32);
        assert_eq!(dmin(12), 64);
        assert_eq!(dmin(16), 128);
        assert_eq!(dmin(32), 32768);
    }

    // -- pair_index ------------------------------------------------------------

    #[test]
    fn test_pair_index() {
        // (1,0) -> 0
        assert_eq!(pair_index(1, 0), 0);
        // (2,0) -> 1, (2,1) -> 2
        assert_eq!(pair_index(2, 0), 1);
        assert_eq!(pair_index(2, 1), 2);
        // (3,0) -> 3, (3,1) -> 4, (3,2) -> 5
        assert_eq!(pair_index(3, 0), 3);
        assert_eq!(pair_index(3, 1), 4);
        assert_eq!(pair_index(3, 2), 5);
    }

    // -- V_IJ^2 = -I -----------------------------------------------------------

    #[test]
    fn v_squared_is_neg_identity_cs() {
        let rep = cs_n4();
        let holo = HoloraumyData::from_rep(&rep);
        for i in 1..rep.n {
            for j in 0..i {
                let idx = pair_index(i, j);
                let v2 = holo.v[idx].compose(&holo.v[idx]);
                assert!(
                    v2.is_neg_identity(),
                    "V_({},{})^2 should be -I for CS",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn v_squared_is_neg_identity_vs() {
        let rep = vs_n4();
        let holo = HoloraumyData::from_rep(&rep);
        for i in 1..rep.n {
            for j in 0..i {
                let idx = pair_index(i, j);
                let v2 = holo.v[idx].compose(&holo.v[idx]);
                assert!(
                    v2.is_neg_identity(),
                    "V_({},{})^2 should be -I for VS",
                    i,
                    j
                );
            }
        }
    }

    // -- gadget self-values (irreducible => 1.0) --------------------------------

    #[test]
    fn gadget_self_cs() {
        let holo = HoloraumyData::from_rep(&cs_n4());
        let g = gadget(&holo, &holo);
        assert!(
            (g - 1.0).abs() < 1e-10,
            "G[CS,CS] should be 1.0, got {}",
            g
        );
    }

    #[test]
    fn gadget_self_vs() {
        let holo = HoloraumyData::from_rep(&vs_n4());
        let g = gadget(&holo, &holo);
        assert!(
            (g - 1.0).abs() < 1e-10,
            "G[VS,VS] should be 1.0, got {}",
            g
        );
    }

    // -- gadget cross-values (known N=4 results) --------------------------------

    #[test]
    fn gadget_cs_vs() {
        let cs = HoloraumyData::from_rep(&cs_n4());
        let vs = HoloraumyData::from_rep(&vs_n4());
        let g = gadget(&cs, &vs);
        assert!(
            g.abs() < 1e-10,
            "G[CS,VS] should be 0.0, got {}",
            g
        );
    }

    // -- gadget symmetry -------------------------------------------------------

    #[test]
    fn gadget_symmetry() {
        let cs = HoloraumyData::from_rep(&cs_n4());
        let vs = HoloraumyData::from_rep(&vs_n4());

        let g_cv = gadget(&cs, &vs);
        let g_vc = gadget(&vs, &cs);
        assert!(
            (g_cv - g_vc).abs() < 1e-10,
            "gadget should be symmetric: G[CS,VS]={} != G[VS,CS]={}",
            g_cv,
            g_vc
        );
    }

    // -- gadget_matrix ---------------------------------------------------------

    #[test]
    fn gadget_matrix_n4() {
        let cs = HoloraumyData::from_rep(&cs_n4());
        let vs = HoloraumyData::from_rep(&vs_n4());
        let reps = vec![cs, vs];

        let mat = gadget_matrix(&reps);

        // Diagonal: all 1.0
        for i in 0..2 {
            assert!(
                (mat[i][i] - 1.0).abs() < 1e-10,
                "diagonal [{}][{}] should be 1.0, got {}",
                i,
                i,
                mat[i][i]
            );
        }

        // CS-VS = 0.0
        assert!(
            mat[0][1].abs() < 1e-10,
            "mat[0][1] (CS-VS) should be 0.0, got {}",
            mat[0][1]
        );

        // Symmetry
        assert!(
            (mat[0][1] - mat[1][0]).abs() < 1e-10,
            "matrix should be symmetric",
        );
    }

    // -- from_rep produces correct pair count -----------------------------------

    #[test]
    fn from_rep_pair_count() {
        let holo = HoloraumyData::from_rep(&cs_n4());
        let expected = 4 * 3 / 2; // C(4,2) = 6
        assert_eq!(holo.vtilde.len(), expected);
        assert_eq!(holo.v.len(), expected);
    }

    // -- vtilde squared is also -I ----------------------------------------------

    #[test]
    fn vtilde_squared_is_neg_identity_cs() {
        let holo = HoloraumyData::from_rep(&cs_n4());
        for idx in 0..holo.vtilde.len() {
            let vt2 = holo.vtilde[idx].compose(&holo.vtilde[idx]);
            assert!(
                vt2.is_neg_identity(),
                "Vtilde[{}]^2 should be -I for CS",
                idx
            );
        }
    }
}
