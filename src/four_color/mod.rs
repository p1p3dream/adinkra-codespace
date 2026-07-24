#![allow(dead_code)]
//! Four-color hopper and G-matrix apparatus for Gates and Lee, arXiv:2408.09342.
//!
//! Goal: reproduce Tables 7 to 13 exactly (the X/Y/Z/W hopper recursion for the
//! minimal 4D N=1 chiral, vector, and tensor supermultiplets and the complex
//! linear system). Two separate solvers live here: a signed-permutation
//! p-th-root solver (`roots.rs`) for the hopper operators, and the G-matrix
//! solver (`gmatrix.rs`) for the general {-1,0,1} matrix G with G^2 = A of
//! Eq 8.2. It is the G-matrix solver, not the p-th-root solver, that replaces
//! the paper's brute-force `Select[Tuples[{-1,0,1}, n^2], ...]` search.
//!
//! Shared conventions (locked by the `cm_l_matrices_satisfy_garden_algebra`
//! test below, do NOT change without re-running it):
//!   - A signed address like [1,-4,2,-3] is the bracket <1 -4 2 -3>. Row i has
//!     its nonzero at column |addr[i]|-1 with sign(addr[i]). This is
//!     `SignedPerm`'s native row-form.
//!   - `matmul(a, b)` returns the matrix product matrix(a) * matrix(b).
//!   - Garden algebra with R_I = L_I^T: L_I L_J^T + L_J L_I^T = 2 delta_IJ I.

use crate::signed_perm::SignedPerm;

pub mod cls;
pub mod cm;
pub mod gmatrix;
pub mod gmatrix_oracle;
pub mod gmatrix_verify;
pub mod roots;
pub mod tm;
pub mod vm;

/// A general {-1,0,1} integer matrix (row-major), the domain of the G-matrix.
/// The G-matrix is NOT a signed permutation: it can have several nonzeros per row.
pub type IntMat = Vec<Vec<i32>>;

/// Dense integer matrix product.
pub fn imm(a: &IntMat, b: &IntMat) -> IntMat {
    let n = a.len();
    let mut c = vec![vec![0i32; n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k] == 0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Dense matrix of a signed permutation (for building A = (sum L) * L1^{-1}).
pub fn dense(sp: &crate::signed_perm::SignedPerm) -> IntMat {
    let d = sp.dim();
    let mut m = vec![vec![0i32; d]; d];
    for i in 0..d {
        m[i][sp.perm[i] as usize] = sp.sign[i] as i32;
    }
    m
}

/// Sum a list of dense matrices.
pub fn dsum(mats: &[IntMat]) -> IntMat {
    let n = mats[0].len();
    let mut s = vec![vec![0i32; n]; n];
    for m in mats {
        for i in 0..n {
            for j in 0..n {
                s[i][j] = s[i][j].checked_add(m[i][j]).expect("dsum i32 overflow");
            }
        }
    }
    s
}

/// Build a signed permutation from a signed address, e.g. sp(&[1,-4,2,-3]).
/// Row i has its nonzero at column |addr[i]|-1 with sign(addr[i]).
pub fn sp(addr: &[i32]) -> SignedPerm {
    let perm = addr.iter().map(|&a| (a.abs() - 1) as u16).collect();
    let sign = addr.iter().map(|&a| a.signum() as i8).collect();
    SignedPerm::from_parts(perm, sign).expect("valid signed address")
}

/// Matrix product matrix(a) * matrix(b).
///
/// `SignedPerm::compose(self, other)` yields matrix(other) * matrix(self),
/// so matrix(a) * matrix(b) == b.compose(a).
pub fn matmul(a: &SignedPerm, b: &SignedPerm) -> SignedPerm {
    b.compose(a)
}

/// p-th matrix power (p >= 0), pow(a, 0) = identity.
pub fn pow(a: &SignedPerm, p: u32) -> SignedPerm {
    let mut acc = SignedPerm::identity(a.dim());
    for _ in 0..p {
        acc = matmul(&acc, a);
    }
    acc
}

/// -M^T for a signed permutation M (used in the antisymmetry test).
pub fn neg_transpose(m: &SignedPerm) -> SignedPerm {
    m.transpose().negate()
}

/// Garden algebra check for a set of 4-color L-matrices with R_I = L_I^T:
/// requires L_I L_I^T = I and, for I != J, M = L_I L_J^T satisfies M = -M^T.
pub fn garden_ok(ls: &[SignedPerm]) -> bool {
    let n = ls.len();
    for i in 0..n {
        for j in 0..n {
            let m = matmul(&ls[i], &ls[j].transpose());
            if i == j {
                if !m.is_identity() {
                    return false;
                }
            } else if m != neg_transpose(&m) {
                return false;
            }
        }
    }
    true
}

/// Chiral L-matrices, Table 1 (CM row): <1 -4 2 -3>, <2 3 -1 -4>, <3 -2 -4 1>, <4 1 3 2>.
/// Kept here because the convention lock (Garden algebra) is calibrated on them.
pub fn cm_l_matrices() -> Vec<SignedPerm> {
    vec![
        sp(&[1, -4, 2, -3]),
        sp(&[2, 3, -1, -4]),
        sp(&[3, -2, -4, 1]),
        sp(&[4, 1, 3, 2]),
    ]
}

/// Run every submodule's report (called by the CLI or a smoke test).
pub fn verify_all() {
    for line in [
        cm::report(),
        vm::report(),
        tm::report(),
        roots::report(),
        cls::report(),
        gmatrix::report(),
        gmatrix_oracle::report(),
        gmatrix_verify::report(),
    ] {
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cm_l_matrices_satisfy_garden_algebra() {
        // This is the convention lock. If it fails, sp() or matmul() is wrong.
        assert!(
            garden_ok(&cm_l_matrices()),
            "CM L-matrices must satisfy the Garden algebra under the chosen convention"
        );
    }

    #[test]
    fn matmul_and_pow_basics() {
        let a = cm_l_matrices()[0].clone();
        assert!(pow(&a, 0).is_identity());
        assert_eq!(pow(&a, 1), a);
        // L_I L_I^T = I for an orthogonal signed permutation.
        assert!(matmul(&a, &a.transpose()).is_identity());
        // matmul agrees with repeated compose orientation.
        assert_eq!(pow(&a, 2), matmul(&a, &a));
    }

    // The two typos in arXiv:2408.09342, verified against the LaTeX e-print,
    // asserted here so the finding lives in the apparatus and not in scratch.

    #[test]
    fn printed_chiral_l1_is_a_typo_that_fails_garden() {
        // Table 7 prints Chiral L1 as <1 4 -2 -3>; that value does NOT satisfy
        // the Garden algebra. The corrected <1 -4 2 -3> does.
        let printed = vec![
            sp(&[1, 4, -2, -3]),
            sp(&[2, 3, -1, -4]),
            sp(&[3, -2, -4, 1]),
            sp(&[4, 1, 3, 2]),
        ];
        assert!(!garden_ok(&printed), "printed Table 7 Chiral L1 must fail the Garden algebra");
        assert!(garden_ok(&cm_l_matrices()), "corrected Chiral L-matrices satisfy the Garden algebra");
    }

    #[test]
    fn printed_tensor_r4_is_a_typo_that_is_not_a_permutation() {
        use crate::signed_perm::SignedPerm;
        // Table 12 prints Tensor R4 as <3 -2 4 3>: column 3 twice, column 1
        // missing, not a valid signed permutation. The corrected <3 -2 4 1> is.
        let printed = SignedPerm::from_parts(vec![2, 1, 3, 2], vec![1, -1, 1, 1]);
        assert!(printed.is_err(), "printed Table 12 Tensor R4 must be an invalid permutation");
        let corrected = SignedPerm::from_parts(vec![2, 1, 3, 0], vec![1, -1, 1, 1]);
        assert!(corrected.is_ok(), "corrected Tensor R4 <3 -2 4 1> is a valid signed permutation");
    }
}
