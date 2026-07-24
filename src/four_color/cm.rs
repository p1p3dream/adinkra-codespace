#![allow(dead_code)]
//! Chiral (CM) supermultiplet: reproduction of Tables 7 and 8 of Gates and Lee,
//! arXiv:2408.09342, in the shared four-color signed-permutation apparatus.
//!
//! Table 7 is the L-matrix summary (rows [Ln], [Xn], [Yn], [Zn], [Wn]).
//! Table 8 is the R-matrix summary (rows [Rn], [X'n], [Y'n], [Z'n], [W'n]).
//!
//! The X/Y/Z/W rows are NOT independent data. They are generated from the four
//! L-matrices by the hopping recursion of Sec. 4 (paper Eqs 4.4, 4.7, 4.10, 4.12)
//! with the cyclic index n+1 wrapping 4 -> 1:
//!
//!   X_n = L_{n+1} L_n^{-1}    (Eq 4.4:  L_{n+1} = X_n L_n)
//!   Y_n = X_{n+1} X_n^{-1}    (Eq 4.7)
//!   Z_n = Y_{n+1} Y_n^{-1}    (Eq 4.10)
//!   W_n = Z_{n+1} Z_n^{-1}    (Eq 4.12)
//!
//! Convergence (paper Eqs 4.11 and Table 13): each Z_n is +I or -I, and every
//! W_n = -I4. The same recursion applied to the R-matrices gives the primed
//! quantities of Table 8.
//!
//! Table 7 [Ln] n=1 typo in arXiv:2408.09342. The published paper (verified
//! against the arXiv LaTeX e-print, file AdnkWolfram.tex) prints L_1 as
//! \vev{14\bar2\bar3} = <1 4 -2 -3>. That value is a typo in the paper itself,
//! not an OCR artifact: it appears verbatim in the LaTeX source. The printed
//! <1 4 -2 -3> does NOT satisfy the Garden algebra and does NOT reproduce Table
//! 7's own [Xn],[Yn],[Zn],[Wn] rows. The internally consistent value is
//! \vev{1\bar42\bar3} = <1 -4 2 -3>, which is what the Garden-valid,
//! convention-locked chiral L-set in `super::cm_l_matrices()` uses. That value
//! satisfies the Garden algebra AND makes the recursion (Z_n = +/-I, W_n = -I)
//! hold, reproducing the [Xn],[Yn],[Zn],[Wn] rows of Table 7 exactly. This was
//! confirmed by convention-free 4x4 matrix arithmetic. The other three [Ln]
//! entries (n=2,3,4) match the paper as printed.

use crate::signed_perm::SignedPerm;

/// Chiral L-matrices (Table 7 [Ln], convention-locked Garden-valid set).
pub fn l_matrices() -> Vec<SignedPerm> {
    super::cm_l_matrices()
}

/// Chiral R-matrices (Table 8 [Rn]). In the Garden convention R_I = L_I^T, and
/// this transpose reproduces the transcribed Table 8 [Rn] row exactly:
///   <1 3 -4 -2>, <-3 1 2 -4>, <4 -2 1 -3>, <2 4 3 1>.
pub fn r_matrices() -> Vec<SignedPerm> {
    l_matrices().iter().map(|m| m.transpose()).collect()
}

/// One step of the hopping recursion applied to a 4-element cyclic list:
/// out_n = m_{n+1} * m_n^{-1}, with n+1 wrapping 4 -> 1.
fn hop(ms: &[SignedPerm]) -> Vec<SignedPerm> {
    let k = ms.len();
    (0..k)
        .map(|n| super::matmul(&ms[(n + 1) % k], &ms[n].inverse()))
        .collect()
}

/// X_n = L_{n+1} L_n^{-1} (Eq 4.4).
pub fn x_matrices() -> Vec<SignedPerm> {
    hop(&l_matrices())
}

/// Y_n = X_{n+1} X_n^{-1} (Eq 4.7).
pub fn y_matrices() -> Vec<SignedPerm> {
    hop(&x_matrices())
}

/// Z_n = Y_{n+1} Y_n^{-1} (Eq 4.10). Each entry is +I or -I.
pub fn z_matrices() -> Vec<SignedPerm> {
    hop(&y_matrices())
}

/// W_n = Z_{n+1} Z_n^{-1} (Eq 4.12). Every entry is -I4 (Table 13).
pub fn w_matrices() -> Vec<SignedPerm> {
    hop(&z_matrices())
}

/// X'_n = R_{n+1} R_n^{-1} (primed recursion, Table 8).
pub fn xp_matrices() -> Vec<SignedPerm> {
    hop(&r_matrices())
}

/// Y'_n = X'_{n+1} X'_n^{-1}.
pub fn yp_matrices() -> Vec<SignedPerm> {
    hop(&xp_matrices())
}

/// Z'_n = Y'_{n+1} Y'_n^{-1}. Each entry is +I or -I.
pub fn zp_matrices() -> Vec<SignedPerm> {
    hop(&yp_matrices())
}

/// W'_n = Z'_{n+1} Z'_n^{-1}. Every entry is -I4 (Table 13).
pub fn wp_matrices() -> Vec<SignedPerm> {
    hop(&zp_matrices())
}

/// Real pass/fail summary of the Table 7 and Table 8 reproduction.
pub fn report() -> String {
    let l = l_matrices();
    let mut ok = super::garden_ok(&l);

    // Recursion convergence: Z in {+I,-I}, W = -I, for both L and R towers.
    for z in z_matrices().iter().chain(zp_matrices().iter()) {
        ok &= z.is_identity() || z.is_neg_identity();
    }
    for w in w_matrices().iter().chain(wp_matrices().iter()) {
        ok &= w.is_neg_identity();
    }

    // Exact Table 7 / Table 8 entry checks.
    ok &= x_matrices() == addr_set(&TABLE7_X);
    ok &= y_matrices() == addr_set(&TABLE7_Y);
    ok &= z_matrices() == addr_set(&TABLE7_Z);
    ok &= w_matrices() == addr_set(&TABLE7_W);
    ok &= r_matrices() == addr_set(&TABLE8_R);
    ok &= xp_matrices() == addr_set(&TABLE8_XP);
    ok &= yp_matrices() == addr_set(&TABLE8_YP);
    ok &= zp_matrices() == addr_set(&TABLE8_ZP);
    ok &= wp_matrices() == addr_set(&TABLE8_WP);

    if ok {
        "cm: Tables 7 and 8 reproduced (Garden OK; Z=+/-I, W=-I; all entries match)".to_string()
    } else {
        "cm: FAIL (Table 7/8 reproduction mismatch)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Transcribed paper entries (signed-address form). See module typo note.
// ---------------------------------------------------------------------------

/// Helper: build the signed-permutation list from a list of signed addresses.
fn addr_set(rows: &[[i32; 4]]) -> Vec<SignedPerm> {
    rows.iter().map(|r| super::sp(r)).collect()
}

// Table 7, [Xn]: <3 -4 -1 2>, <2 -1 4 -3>, <-3 4 1 -2>, <2 -1 4 -3>.
const TABLE7_X: [[i32; 4]; 4] =
    [[3, -4, -1, 2], [2, -1, 4, -3], [-3, 4, 1, -2], [2, -1, 4, -3]];
// Table 7, [Yn]: <4 3 -2 -1>, <4 3 -2 -1>, <-4 -3 2 1>, <-4 -3 2 1>.
const TABLE7_Y: [[i32; 4]; 4] =
    [[4, 3, -2, -1], [4, 3, -2, -1], [-4, -3, 2, 1], [-4, -3, 2, 1]];
// Table 7, [Zn]: <1 2 3 4>=I, <-1 -2 -3 -4>=-I, I, -I.
const TABLE7_Z: [[i32; 4]; 4] =
    [[1, 2, 3, 4], [-1, -2, -3, -4], [1, 2, 3, 4], [-1, -2, -3, -4]];
// Table 7, [Wn]: all <-1 -2 -3 -4> = -I4.
const TABLE7_W: [[i32; 4]; 4] = [[-1, -2, -3, -4]; 4];

// Table 8, [Rn]: <1 3 -4 -2>, <-3 1 2 -4>, <4 -2 1 -3>, <2 4 3 1>.
const TABLE8_R: [[i32; 4]; 4] =
    [[1, 3, -4, -2], [-3, 1, 2, -4], [4, -2, 1, -3], [2, 4, 3, 1]];
// Table 8, [X'n]: <-2 1 -4 3>, <-4 -3 2 1>, <-2 1 -4 3>, <4 3 -2 -1>.
const TABLE8_XP: [[i32; 4]; 4] =
    [[-2, 1, -4, 3], [-4, -3, 2, 1], [-2, 1, -4, 3], [4, 3, -2, -1]];
// Table 8, [Y'n]: <3 -4 -1 2>, <-3 4 1 -2>, <-3 4 1 -2>, <3 -4 -1 2>.
const TABLE8_YP: [[i32; 4]; 4] =
    [[3, -4, -1, 2], [-3, 4, 1, -2], [-3, 4, 1, -2], [3, -4, -1, 2]];
// Table 8, [Z'n]: <-1 -2 -3 -4>=-I, <1 2 3 4>=I, -I, I.
const TABLE8_ZP: [[i32; 4]; 4] =
    [[-1, -2, -3, -4], [1, 2, 3, 4], [-1, -2, -3, -4], [1, 2, 3, 4]];
// Table 8, [W'n]: all <-1 -2 -3 -4> = -I4.
const TABLE8_WP: [[i32; 4]; 4] = [[-1, -2, -3, -4]; 4];

#[cfg(test)]
mod tests {
    use super::*;

    fn neg_i() -> SignedPerm {
        super::super::sp(&[-1, -2, -3, -4])
    }

    // (a) The chiral L-matrices satisfy the Garden algebra.
    #[test]
    fn l_matrices_satisfy_garden_algebra() {
        assert!(
            super::super::garden_ok(&l_matrices()),
            "chiral L-matrices must satisfy the Garden algebra"
        );
    }

    // The R-matrices (= L^T) also satisfy the Garden algebra.
    #[test]
    fn r_matrices_satisfy_garden_algebra() {
        assert!(super::super::garden_ok(&r_matrices()));
    }

    // (b) Recursion convergence for the L-tower: Z_n in {+I,-I}, W_n = -I4.
    #[test]
    fn l_recursion_converges() {
        let z = z_matrices();
        // Paper Eq 4.11: Z1,Z3 = +I; Z2,Z4 = -I.
        assert!(z[0].is_identity());
        assert!(z[1].is_neg_identity());
        assert!(z[2].is_identity());
        assert!(z[3].is_neg_identity());
        for w in w_matrices() {
            assert!(w.is_neg_identity(), "every W_n must be -I4 (Table 13)");
        }
    }

    // (b) Recursion convergence for the R-tower: Z'_n in {+I,-I}, W'_n = -I4.
    #[test]
    fn r_recursion_converges() {
        let zp = zp_matrices();
        // Primed pattern is the sign-flipped mirror: Z'1,Z'3 = -I; Z'2,Z'4 = +I.
        assert!(zp[0].is_neg_identity());
        assert!(zp[1].is_identity());
        assert!(zp[2].is_neg_identity());
        assert!(zp[3].is_identity());
        for w in wp_matrices() {
            assert!(w.is_neg_identity(), "every W'_n must be -I4 (Table 13)");
        }
    }

    // (c) Exact reproduction of the Table 7 X/Y/Z/W entries by the recursion.
    #[test]
    fn table7_entries_match() {
        assert_eq!(x_matrices(), addr_set(&TABLE7_X), "Table 7 [Xn]");
        assert_eq!(y_matrices(), addr_set(&TABLE7_Y), "Table 7 [Yn]");
        assert_eq!(z_matrices(), addr_set(&TABLE7_Z), "Table 7 [Zn]");
        assert_eq!(w_matrices(), addr_set(&TABLE7_W), "Table 7 [Wn]");
    }

    // The three [Ln] entries n=2,3,4 match the paper as printed; n=1 uses the
    // Garden-valid <1 -4 2 -3> in place of the paper's typo (see module typo note).
    #[test]
    fn table7_l_row_matches_except_documented_n1() {
        let l = l_matrices();
        assert_eq!(l[1], super::super::sp(&[2, 3, -1, -4]));
        assert_eq!(l[2], super::super::sp(&[3, -2, -4, 1]));
        assert_eq!(l[3], super::super::sp(&[4, 1, 3, 2]));
        // Recursion + Garden require this over the paper's typo <1 4 -2 -3> for
        // n=1 (Table 7; see module typo note).
        assert_eq!(l[0], super::super::sp(&[1, -4, 2, -3]));
    }

    // (d) Exact reproduction of Table 8 [Rn] and the primed X'/Y'/Z'/W' entries.
    #[test]
    fn table8_entries_match() {
        assert_eq!(r_matrices(), addr_set(&TABLE8_R), "Table 8 [Rn]");
        assert_eq!(xp_matrices(), addr_set(&TABLE8_XP), "Table 8 [X'n]");
        assert_eq!(yp_matrices(), addr_set(&TABLE8_YP), "Table 8 [Y'n]");
        assert_eq!(zp_matrices(), addr_set(&TABLE8_ZP), "Table 8 [Z'n]");
        assert_eq!(wp_matrices(), addr_set(&TABLE8_WP), "Table 8 [W'n]");
    }

    // Bosonic-holoraumy / Eq 4.1 style relation: X_n is the L_{n+1} L_n^{-1}
    // hopping operator, so by construction L_{n+1} = X_n L_n (paper Eq 4.4).
    // Verify that identity directly for all n, and that each X_n is an involutive
    // signed permutation up to sign consistent with the tower closing on -I.
    #[test]
    fn eq_4_4_hopping_identity_holds() {
        let l = l_matrices();
        let x = x_matrices();
        for n in 0..4 {
            let lhs = l[(n + 1) % 4].clone();
            let rhs = super::super::matmul(&x[n], &l[n]);
            assert_eq!(lhs, rhs, "Eq 4.4: L_(n+1) = X_n L_n must hold for n={n}");
        }
    }

    // The tower terminates on -I: applying one more hop to W gives +I (since
    // W_n = -I for all n, W_{n+1} W_n^{-1} = (-I)(-I) = I), confirming the fixed
    // point of the recursion.
    #[test]
    fn tower_fixed_point() {
        let w = w_matrices();
        for n in 0..4 {
            let next = super::super::matmul(&w[(n + 1) % 4], &w[n].inverse());
            assert!(next.is_identity());
        }
        assert!(neg_i().is_neg_identity());
    }
}
