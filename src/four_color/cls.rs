#![allow(dead_code)]
//! Complex Linear Supermultiplet (CLS), the 12x12 non-minimal system of
//! Gates and Lee, arXiv:2408.09342 (Section 5, "4D N=1 CLS L-matrix" and
//! "4D N=1 CLS R-matrix"; Appendix C, "Diagonalized L- and R-matrix of CLS
//! Field"; Table 13, CLS [Wn] = -I12).
//!
//! This is the headline case. The paper states the CLS valise G-matrix is
//! "currently unavailable due to our current insufficient access to more
//! robust computational capacities to run the code," and that the CLS L- and
//! R-matrices "exceed the calculation RAM amount usage." Our job is the CLS
//! L/R matrices and the X/Y/Z/W hopper recursion, done exactly, in a
//! signed-permutation apparatus.
//!
//! Transcription choice (important, read before touching the matrices):
//!
//! Section 5.1 of the paper prints CLS L-matrices in the "raw" block adinkra
//! basis. In that basis SEVERAL rows carry TWO nonzero entries (for example
//! [L1] row 3 has -1 in column 3 AND -1 in column 11). Those printed matrices
//! are therefore NOT monomial (not signed permutations), so they cannot be
//! represented by `super::sp` and do not by themselves satisfy the Garden
//! algebra L_I L_J^T + L_J L_I^T = 2 delta_IJ I as printed. This matches the
//! paper's own remark that the CLS system is the one that blows up.
//!
//! Appendix C ("Diagonalized L- and R-matrix of CLS Field") gives the SAME
//! representation in a rotated basis where every L-matrix is block diagonal
//! (three 4x4 blocks) and every row has exactly one nonzero entry. Those ARE
//! genuine signed permutations and they DO satisfy the Garden algebra at
//! dim 12. We transcribe from Appendix C, cross-checked by `super::garden_ok`,
//! which is the transcription arbiter. Every matrix below is verified in the
//! inline tests. Source lines in the arXiv LaTeX e-print (AdnkWolfram.tex):
//! L1..L4 at 4183..4252, R1..R4 at 4255..4324.

use crate::signed_perm::SignedPerm;

/// CLS L-matrices L[0..4] == L_1..L_4, dim 12, from Appendix C (block
/// diagonal, three 4x4 blocks). L_1 is the identity in this basis.
///
/// Signed-address form for `super::sp`: row i entry is sign * (col+1).
pub fn cls_l_matrices() -> Vec<SignedPerm> {
    vec![
        // L_1 = I_12
        super::sp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        // L_2
        super::sp(&[2, -1, 4, -3, 6, -5, -8, 7, -10, 9, 12, -11]),
        // L_3
        super::sp(&[3, -4, -1, 2, -7, -8, 5, 6, 11, 12, -9, -10]),
        // L_4
        super::sp(&[4, 3, -2, -1, -8, 7, -6, 5, -12, 11, -10, 9]),
    ]
}

/// CLS R-matrices R[0..4] == R_1..R_4, dim 12, from Appendix C. In the Garden
/// algebra R_I = L_I^T; this function transcribes the printed R-matrices
/// independently so the identity R_I == L_I^T can be tested, not assumed.
pub fn cls_r_matrices() -> Vec<SignedPerm> {
    vec![
        // R_1 = I_12
        super::sp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        // R_2
        super::sp(&[-2, 1, -4, 3, -6, 5, 8, -7, 10, -9, -12, 11]),
        // R_3
        super::sp(&[-3, 4, 1, -2, 7, 8, -5, -6, -11, -12, 9, 10]),
        // R_4
        super::sp(&[-4, -3, 2, 1, 8, -7, 6, -5, 12, -11, 10, -9]),
    ]
}

/// One step of the hopper recursion on a 4-tuple of dim-12 signed
/// permutations, with cyclic index (n+1 wraps 4 -> 1):
///   out_n = A_{n+1} * A_n^{-1}  =  matmul(A_{n+1}, A_n^{-1}).
/// This is the shared minimal-case rule applied at dim 12.
fn hop(a: &[SignedPerm]) -> Vec<SignedPerm> {
    let m = a.len();
    (0..m)
        .map(|n| super::matmul(&a[(n + 1) % m], &a[n].inverse()))
        .collect()
}

/// X_n = L_{n+1} L_n^{-1}.
pub fn cls_x_matrices() -> Vec<SignedPerm> {
    hop(&cls_l_matrices())
}

/// Y_n = X_{n+1} X_n^{-1}.
pub fn cls_y_matrices() -> Vec<SignedPerm> {
    hop(&cls_x_matrices())
}

/// Z_n = Y_{n+1} Y_n^{-1}.
pub fn cls_z_matrices() -> Vec<SignedPerm> {
    hop(&cls_y_matrices())
}

/// W_n = Z_{n+1} Z_n^{-1}.
pub fn cls_w_matrices() -> Vec<SignedPerm> {
    hop(&cls_z_matrices())
}

/// True when every element of `ms` is +I or -I (the "collapsed" state the
/// minimal CM/VM/TM cases reach quickly).
fn all_pm_identity(ms: &[SignedPerm]) -> bool {
    ms.iter().all(|m| m.is_identity() || m.is_neg_identity())
}

/// True when every element of `ms` is exactly -I.
fn all_neg_identity(ms: &[SignedPerm]) -> bool {
    ms.iter().all(|m| m.is_neg_identity())
}

/// True when `m` is diagonal (identity permutation part) but NOT scalar, i.e.
/// its sign vector is a nontrivial mix of +1 and -1. This is exactly the
/// CLS Z-stage state: diagonal, but not +/-I12.
fn is_diagonal_not_scalar(m: &SignedPerm) -> bool {
    let is_diagonal = m.perm.iter().enumerate().all(|(i, &p)| p == i as u16);
    let has_plus = m.sign.iter().any(|&s| s == 1);
    let has_minus = m.sign.iter().any(|&s| s == -1);
    is_diagonal && has_plus && has_minus
}

/// One-line status report.
pub fn report() -> String {
    let l = cls_l_matrices();
    let garden = super::garden_ok(&l);
    let x = cls_x_matrices();
    let y = cls_y_matrices();
    let z = cls_z_matrices();
    let w = cls_w_matrices();
    let z_diag_not_scalar = z.iter().all(is_diagonal_not_scalar);
    format!(
        "cls: dim=12 garden_ok(L)={} (Appendix C block-diagonal basis) | \
         X all +/-I={} | Y all +/-I={} | Z all +/-I={} | \
         Z diagonal-but-not-scalar={} | W all -I12={} (Table 13). \
         Contrast with minimal CM/VM/TM: those collapse to +/-I by the Z (third) \
         step; CLS does NOT. At Z the permutation part is trivial but the sign \
         diagonal is a nontrivial block +/-1 pattern, so Z != +/-I12. CLS reaches \
         -I12 only at the fourth (W) step.",
        garden,
        all_pm_identity(&x),
        all_pm_identity(&y),
        all_pm_identity(&z),
        z_diag_not_scalar,
        all_neg_identity(&w),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cls_l_matrices_are_dim_12_signed_permutations() {
        let l = cls_l_matrices();
        assert_eq!(l.len(), 4);
        for m in &l {
            assert_eq!(m.dim(), 12, "each CLS L-matrix must be 12x12");
        }
    }

    #[test]
    fn cls_l_matrices_satisfy_garden_algebra_at_dim_12() {
        // The transcription arbiter. If this fails, the Appendix C transcription
        // is wrong and nothing downstream should be trusted.
        assert!(
            super::super::garden_ok(&cls_l_matrices()),
            "CLS (Appendix C, block-diagonal) L-matrices must satisfy the \
             Garden algebra L_I L_J^T + L_J L_I^T = 2 delta_IJ I at dim 12"
        );
    }

    #[test]
    fn cls_r_equals_l_transpose() {
        // Garden algebra sets R_I = L_I^T. We transcribed the printed R-matrices
        // independently, so this is a genuine cross-check of both blocks.
        let l = cls_l_matrices();
        let r = cls_r_matrices();
        for n in 0..4 {
            assert_eq!(
                r[n],
                l[n].transpose(),
                "CLS R_{} must equal L_{}^T",
                n + 1,
                n + 1
            );
        }
    }

    #[test]
    fn cls_w_is_negative_identity_for_all_n() {
        // Table 13, CLS row: [Wn]_i^k = -[I_12]_i^k for n in {1,2,3,4}.
        let w = cls_w_matrices();
        assert_eq!(w.len(), 4);
        for (n, m) in w.iter().enumerate() {
            assert!(
                m.is_neg_identity(),
                "CLS W_{} must be -I12 (Table 13), got a non -I12 matrix",
                n + 1
            );
        }
    }

    #[test]
    fn cls_recursion_does_not_collapse_early_unlike_minimal_cases() {
        // The honest contrast with the minimal CM/VM/TM cases. In those the
        // recursion collapses to +/-I by the Z (third) step. For CLS at dim 12
        // it does NOT. Observed exact behavior (verified against the printed
        // matrices, Appendix C basis):
        //   X_n: signed permutations with nontrivial permutation part (each is
        //        a product of transpositions inside the three 4-blocks). Not
        //        diagonal.
        //   Y_n: still nontrivial permutation part (double transpositions per
        //        4-block). Not diagonal.
        //   Z_n: permutation part becomes the identity, so Z_n IS diagonal, but
        //        the sign diagonal is a nontrivial block +/-1 pattern
        //        (e.g. diag = (+,+,+,+, -,-,-,-, -,-,-,-)). So Z_n is diagonal
        //        but NOT +/-I12. This is exactly where the minimal cases would
        //        have already collapsed.
        //   W_n = -I12 for all n (Table 13), reached only at the fourth step.
        let x = cls_x_matrices();
        let y = cls_y_matrices();
        let z = cls_z_matrices();
        let w = cls_w_matrices();

        assert!(
            !all_pm_identity(&x),
            "CLS X should NOT already be all +/-I (that would be early collapse)"
        );
        assert!(
            !all_pm_identity(&y),
            "CLS Y should NOT already be all +/-I"
        );
        assert!(
            !all_pm_identity(&z),
            "CLS Z should NOT already be all +/-I; this is the key contrast \
             with the minimal cases, whose Z stage collapses"
        );
        // Sharper: every Z_n is diagonal but carries a nontrivial sign pattern.
        assert!(
            z.iter().all(is_diagonal_not_scalar),
            "CLS Z_n must be diagonal-but-not-scalar (block +/-1 sign diagonal), \
             the precise reason the three-step collapse fails at dim 12"
        );
        // Only at the W stage does it settle, and there it is exactly -I12.
        assert!(
            all_neg_identity(&w),
            "CLS settles only at W = -I12 (fourth step), matching Table 13"
        );
    }
}
