/// L/R matrix construction and Garden algebra verification for Adinkra
/// representations.
///
/// Builds L_I and R_I signed permutation matrices from a chromotopology and
/// dashing, then verifies the Garden algebra identity:
///
///   L_I * R_J + L_J * R_I = 2 * delta_IJ * I_d

use crate::signed_perm::SignedPerm;

#[derive(Debug, Clone)]
pub struct AdinkraRep {
    pub n: usize,
    pub d: usize,
    pub l_matrices: Vec<SignedPerm>,
}

/// Two signed perms A and B satisfy A + B = 0 iff they share the same
/// permutation and every sign entry is opposite.
fn compose_sum_is_zero(a: &SignedPerm, b: &SignedPerm) -> bool {
    a.perm == b.perm && a.sign.iter().zip(b.sign.iter()).all(|(&sa, &sb)| sa == -sb)
}

impl AdinkraRep {
    /// Build L-matrices from color permutations and dashing signs.
    ///
    /// `color_perms[I]` is a `Vec<usize>` of length `d`: `color_perms[I][j]`
    /// is the fermion index that boson j connects to along color I.
    ///
    /// `dashing` is a `Vec<i8>` of length `N*d`: `dashing[I*d + j]` is the
    /// sign (+1 or -1) on the edge from boson j along color I.
    pub fn from_parts(n: usize, d: usize, color_perms: &[Vec<usize>], dashing: &[i8]) -> Self {
        let mut l_matrices = Vec::with_capacity(n);
        for i in 0..n {
            let perm: Vec<u16> = color_perms[i].iter().map(|&x| x as u16).collect();
            let sign: Vec<i8> = (0..d).map(|j| dashing[i * d + j]).collect();
            l_matrices.push(SignedPerm::from_parts(perm, sign).unwrap());
        }
        AdinkraRep { n, d, l_matrices }
    }

    /// R_I is the transpose (inverse) of L_I.
    pub fn r_matrix(&self, color: usize) -> SignedPerm {
        self.l_matrices[color].inverse()
    }

    /// Verify L_I * R_J + L_J * R_I = 2 * delta_IJ * I_d for all I, J.
    ///
    /// For I == J: L_I * L_I^T must be the identity (signed perm
    /// orthogonality).
    ///
    /// For I != J: L_I * L_J^T + L_J * L_I^T must be zero. Equivalently,
    /// L_I * L_J^T must equal -(L_J * L_I^T).
    pub fn verify_garden_algebra(&self) -> bool {
        for i in 0..self.n {
            let li_lit = self.l_matrices[i].compose(&self.l_matrices[i].inverse());
            if !li_lit.is_identity() {
                return false;
            }
        }
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let li_ljt = self.l_matrices[i].compose(&self.l_matrices[j].inverse());
                let lj_lit = self.l_matrices[j].compose(&self.l_matrices[i].inverse());
                if !compose_sum_is_zero(&li_ljt, &lj_lit) {
                    return false;
                }
            }
        }
        true
    }

    /// Bosonic holoraumy: V_IJ = L_I * R_J.
    pub fn v_matrix(&self, i: usize, j: usize) -> SignedPerm {
        self.l_matrices[i].compose(&self.l_matrices[j].inverse())
    }

    /// Fermionic holoraumy: Vtilde_IJ = R_I * L_J.
    pub fn vtilde_matrix(&self, i: usize, j: usize) -> SignedPerm {
        self.l_matrices[i].inverse().compose(&self.l_matrices[j])
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// N=4 Chiral Supermultiplet (CS) from the [4,1,4] code, dashing class 0.
    ///
    /// Algebraically verified: L_I = -L_I^{-1} for I != 0 (required since
    /// L_0 = I), ensuring L_I * L_J^T + L_J * L_I^T = 2*delta_IJ*I.
    fn cs_n4() -> AdinkraRep {
        let d = 4;
        let l = vec![
            SignedPerm::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1]).unwrap(),
            SignedPerm::from_parts(vec![1, 0, 3, 2], vec![1, -1, 1, -1]).unwrap(),
            SignedPerm::from_parts(vec![2, 3, 0, 1], vec![1, -1, -1, 1]).unwrap(),
            SignedPerm::from_parts(vec![3, 2, 1, 0], vec![1, 1, -1, -1]).unwrap(),
        ];
        AdinkraRep { n: 4, d, l_matrices: l }
    }

    /// N=4 Vector Supermultiplet (VS) from the [4,1,4] code, dashing class 1.
    fn vs_n4() -> AdinkraRep {
        let d = 4;
        let l = vec![
            SignedPerm::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1]).unwrap(),
            SignedPerm::from_parts(vec![1, 0, 3, 2], vec![1, -1, -1, 1]).unwrap(),
            SignedPerm::from_parts(vec![2, 3, 0, 1], vec![1, 1, -1, -1]).unwrap(),
            SignedPerm::from_parts(vec![3, 2, 1, 0], vec![1, -1, 1, -1]).unwrap(),
        ];
        AdinkraRep { n: 4, d, l_matrices: l }
    }

    // -- Garden algebra verification ----------------------------------------

    #[test]
    fn garden_algebra_cs() {
        assert!(cs_n4().verify_garden_algebra());
    }

    #[test]
    fn garden_algebra_vs() {
        assert!(vs_n4().verify_garden_algebra());
    }

    // -- V_IJ^2 = -I for I != J --------------------------------------------

    #[test]
    fn v_squared_is_neg_identity_cs() {
        let rep = cs_n4();
        for i in 0..rep.n {
            for j in 0..rep.n {
                if i == j {
                    continue;
                }
                let v = rep.v_matrix(i, j);
                let v2 = v.compose(&v);
                assert!(
                    v2.is_neg_identity(),
                    "V_{},{} squared should be -I",
                    i,
                    j
                );
            }
        }
    }

    // -- Vtilde skew-symmetry: Vtilde_IJ^{-1} == -Vtilde_IJ ----------------

    #[test]
    fn vtilde_skew_cs() {
        let rep = cs_n4();
        for i in 0..rep.n {
            for j in 0..rep.n {
                if i == j {
                    continue;
                }
                let vt = rep.vtilde_matrix(i, j);
                assert_eq!(
                    vt.inverse(),
                    vt.negate(),
                    "Vtilde_{},{} inverse should equal its negation",
                    i,
                    j
                );
            }
        }
    }

    // -- from_parts constructor ---------------------------------------------

    #[test]
    fn from_parts_constructs_correctly() {
        let color_perms = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 3, 2],
            vec![2, 3, 0, 1],
            vec![3, 2, 1, 0],
        ];
        let dashing = vec![
            1, 1, 1, 1,       // color 0
            1, -1, 1, -1,     // color 1
            1, -1, -1, 1,     // color 2
            1, 1, -1, -1,     // color 3
        ];
        let rep = AdinkraRep::from_parts(4, 4, &color_perms, &dashing);

        let expected = cs_n4();
        assert_eq!(rep.n, expected.n);
        assert_eq!(rep.d, expected.d);
        for i in 0..4 {
            assert_eq!(
                rep.l_matrices[i], expected.l_matrices[i],
                "L_{} mismatch",
                i
            );
        }
    }
}
