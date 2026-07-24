#![allow(dead_code)]
//! Tensor (TM) supermultiplet: reproduction of Gates and Lee, arXiv:2408.09342,
//! Tables 11 and 12 (the X/Y/Z/W hopper recursion for the L- and R-matrices) and
//! the Tensor row of Table 13 (W_n = -I4).
//!
//! Conventions are inherited from the parent module (`super::sp`, `super::matmul`,
//! `super::garden_ok`), locked by the CM Garden-algebra test in mod.rs.
//!
//! The recursion (Eqs. 4.4, 4.7, 4.10, 4.12), cyclic in n with n+1 wrapping 4->1:
//!   X_n = L_{n+1} * L_n^{-1}
//!   Y_n = X_{n+1} * X_n^{-1}
//!   Z_n = Y_{n+1} * Y_n^{-1}
//!   W_n = Z_{n+1} * Z_n^{-1}
//! Each Z_n is +I or -I; each W_n = -I4. The same recursion drives the R side.
//!
//! Table 13 summary (W-matrices of the minimal 4-color supermultiplets):
//!   Chiral  [Wn]  = -I4   (verified in the cm module)
//!   Vector  [Wn]  = -I4   (verified in the vm module)
//!   Tensor  [Wn]  = -I4   (verified HERE, both L and R sides)
//!   CLS     [Wn]  = -I12  (verified in the cls module)
//! This module is authoritative for the Tensor (TM) entry only. The other three
//! rows are asserted in their own modules; we restate the full table here as a
//! documented note so the summary is self-contained.

use crate::signed_perm::SignedPerm;

// ----------------------------------------------------------------------------
// L side (Table 11)
// ----------------------------------------------------------------------------

/// Tensor L-matrices, Table 1 (TM row) / Table 11:
/// <1 -3 -4 -2>, <2 4 -3 1>, <3 1 2 -4>, <4 -2 1 3>.
pub fn tm_l_matrices() -> Vec<SignedPerm> {
    vec![
        super::sp(&[1, -3, -4, -2]),
        super::sp(&[2, 4, -3, 1]),
        super::sp(&[3, 1, 2, -4]),
        super::sp(&[4, -2, 1, 3]),
    ]
}

/// Tensor R-matrices, Table 12 (R_n row):
/// <1 -4 -2 -3>, <4 1 -3 2>, <2 3 1 -4>, and the fourth entry.
///
/// OCR note: the paper's fourth R entry is rendered as `<3 -2 4 3>`, which is
/// not a valid signed permutation (column 4 appears twice, column 1 is absent).
/// The Garden algebra plus the requirement R_I = L_I^T is the arbiter. Since the
/// paper builds R_I as the transpose of L_I (see mod.rs convention note), the
/// correct fourth R-matrix is L_4^T. Transposing L_4 = <4 -2 1 3> gives
/// <3 -2 4 1>: L_4 has row 1 -> col 4 (+), row 2 -> col 2 (-), row 3 -> col 1
/// (+), row 4 -> col 3 (+); the transpose has row 1 -> col 3 (+), row 2 -> col 2
/// (-), row 3 -> col 4 (+), row 4 -> col 1 (+), i.e. <3 -2 4 1>. This matches the
/// paper's `<3 -2 4 _>` up to the corrupted final slot, so the OCR "3" in the
/// last position should read "1". We assert garden_ok on the R set to confirm.
pub fn tm_r_matrices() -> Vec<SignedPerm> {
    vec![
        super::sp(&[1, -4, -2, -3]),
        super::sp(&[4, 1, -3, 2]),
        super::sp(&[2, 3, 1, -4]),
        super::sp(&[3, -2, 4, 1]),
    ]
}

// ----------------------------------------------------------------------------
// The X/Y/Z/W recursion, generic over any 4-matrix L set.
// ----------------------------------------------------------------------------

/// Next level of the hopper recursion: M_n = A_{n+1} * A_n^{-1}, cyclic mod 4.
fn hop(a: &[SignedPerm]) -> Vec<SignedPerm> {
    (0..a.len())
        .map(|n| {
            let next = (n + 1) % a.len();
            super::matmul(&a[next], &a[n].inverse())
        })
        .collect()
}

/// X_n = L_{n+1} L_n^{-1} for the given L set.
pub fn x_matrices(l: &[SignedPerm]) -> Vec<SignedPerm> {
    hop(l)
}

/// Y_n = X_{n+1} X_n^{-1}.
pub fn y_matrices(l: &[SignedPerm]) -> Vec<SignedPerm> {
    hop(&x_matrices(l))
}

/// Z_n = Y_{n+1} Y_n^{-1}.
pub fn z_matrices(l: &[SignedPerm]) -> Vec<SignedPerm> {
    hop(&y_matrices(l))
}

/// W_n = Z_{n+1} Z_n^{-1}.
pub fn w_matrices(l: &[SignedPerm]) -> Vec<SignedPerm> {
    hop(&z_matrices(l))
}

// Convenience wrappers for the two concrete sides.
pub fn tm_x() -> Vec<SignedPerm> { x_matrices(&tm_l_matrices()) }
pub fn tm_y() -> Vec<SignedPerm> { y_matrices(&tm_l_matrices()) }
pub fn tm_z() -> Vec<SignedPerm> { z_matrices(&tm_l_matrices()) }
pub fn tm_w() -> Vec<SignedPerm> { w_matrices(&tm_l_matrices()) }

pub fn tm_x_primed() -> Vec<SignedPerm> { x_matrices(&tm_r_matrices()) }
pub fn tm_y_primed() -> Vec<SignedPerm> { y_matrices(&tm_r_matrices()) }
pub fn tm_z_primed() -> Vec<SignedPerm> { z_matrices(&tm_r_matrices()) }
pub fn tm_w_primed() -> Vec<SignedPerm> { w_matrices(&tm_r_matrices()) }

// ----------------------------------------------------------------------------
// Report
// ----------------------------------------------------------------------------

/// Pass/fail summary for the Tensor supermultiplet reproduction.
pub fn report() -> String {
    let l = tm_l_matrices();
    let r = tm_r_matrices();
    let mut ok = true;

    ok &= super::garden_ok(&l);
    ok &= super::garden_ok(&r);

    // Z_n in {+I, -I}; W_n = -I4 for both sides.
    for side in [&l, &r] {
        for z in z_matrices(side) {
            ok &= z.is_identity() || z.is_neg_identity();
        }
        for w in w_matrices(side) {
            ok &= w.is_neg_identity();
        }
    }

    if ok {
        "tm: PASS (Garden algebra L+R; Z_n in {+I,-I}; W_n = -I4 both sides; \
         Tables 11, 12, Table 13 Tensor row reproduced)"
            .to_string()
    } else {
        "tm: FAIL (see cargo test four_color::tm for the failing assertion)".to_string()
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Paper's Table 11 X/Y/Z/W entries as signed addresses.
    fn table11() -> (Vec<SignedPerm>, Vec<SignedPerm>, Vec<SignedPerm>, Vec<SignedPerm>) {
        let x = vec![
            super::super::sp(&[-4, -3, 2, 1]),
            super::super::sp(&[-3, 4, 1, -2]),
            super::super::sp(&[-4, -3, 2, 1]),
            super::super::sp(&[3, -4, -1, 2]),
        ];
        let y = vec![
            super::super::sp(&[2, -1, 4, -3]),
            super::super::sp(&[-2, 1, -4, 3]),
            super::super::sp(&[-2, 1, -4, 3]),
            super::super::sp(&[2, -1, 4, -3]),
        ];
        let z = vec![
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[1, 2, 3, 4]),
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[1, 2, 3, 4]),
        ];
        let w = vec![
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[-1, -2, -3, -4]),
        ];
        (x, y, z, w)
    }

    /// Paper's Table 12 X'/Y'/Z'/W' entries as signed addresses.
    fn table12() -> (Vec<SignedPerm>, Vec<SignedPerm>, Vec<SignedPerm>, Vec<SignedPerm>) {
        let x = vec![
            super::super::sp(&[-2, 1, 4, -3]),
            super::super::sp(&[4, -3, 2, -1]),
            super::super::sp(&[2, -1, -4, 3]),
            super::super::sp(&[4, -3, 2, -1]),
        ];
        let y = vec![
            super::super::sp(&[3, 4, -1, -2]),
            super::super::sp(&[3, 4, -1, -2]),
            super::super::sp(&[-3, -4, 1, 2]),
            super::super::sp(&[-3, -4, 1, 2]),
        ];
        let z = vec![
            super::super::sp(&[1, 2, 3, 4]),
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[1, 2, 3, 4]),
            super::super::sp(&[-1, -2, -3, -4]),
        ];
        let w = vec![
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[-1, -2, -3, -4]),
            super::super::sp(&[-1, -2, -3, -4]),
        ];
        (x, y, z, w)
    }

    #[test]
    fn tm_l_matrices_satisfy_garden_algebra() {
        // First-order check: the transcription of Table 1 (TM row) is correct.
        assert!(
            super::super::garden_ok(&tm_l_matrices()),
            "TM L-matrices must satisfy the Garden algebra"
        );
    }

    #[test]
    fn tm_r_matrices_satisfy_garden_algebra() {
        // Confirms the OCR-corrupted fourth R entry is <3 -2 4 1> (= L_4^T).
        assert!(
            super::super::garden_ok(&tm_r_matrices()),
            "TM R-matrices must satisfy the Garden algebra"
        );
    }

    #[test]
    fn tm_r_is_transpose_of_l() {
        // R_I = L_I^T is the module convention; the OCR fix relies on it.
        let l = tm_l_matrices();
        let r = tm_r_matrices();
        for (li, ri) in l.iter().zip(r.iter()) {
            assert_eq!(*ri, li.transpose());
        }
    }

    #[test]
    fn tm_l_recursion_matches_table11() {
        let l = tm_l_matrices();
        let (x, y, z, w) = table11();
        assert_eq!(x_matrices(&l), x, "X_n must match Table 11");
        assert_eq!(y_matrices(&l), y, "Y_n must match Table 11");
        assert_eq!(z_matrices(&l), z, "Z_n must match Table 11");
        assert_eq!(w_matrices(&l), w, "W_n must match Table 11");
    }

    #[test]
    fn tm_r_recursion_matches_table12() {
        let r = tm_r_matrices();
        let (x, y, z, w) = table12();
        assert_eq!(x_matrices(&r), x, "X'_n must match Table 12");
        assert_eq!(y_matrices(&r), y, "Y'_n must match Table 12");
        assert_eq!(z_matrices(&r), z, "Z'_n must match Table 12");
        assert_eq!(w_matrices(&r), w, "W'_n must match Table 12");
    }

    #[test]
    fn tm_z_is_plus_or_minus_identity() {
        for side in [tm_l_matrices(), tm_r_matrices()] {
            for z in z_matrices(&side) {
                assert!(
                    z.is_identity() || z.is_neg_identity(),
                    "each Z_n must be +I4 or -I4"
                );
            }
        }
    }

    #[test]
    fn tm_table13_tensor_row_is_neg_identity() {
        // Table 13, Tensor row: every W_n = -I4, on both the L and R sides.
        for side in [tm_l_matrices(), tm_r_matrices()] {
            for w in w_matrices(&side) {
                assert!(w.is_neg_identity(), "Table 13 Tensor: W_n must be -I4");
            }
        }
    }

    #[test]
    fn report_says_pass() {
        assert!(report().starts_with("tm: PASS"), "{}", report());
    }
}
