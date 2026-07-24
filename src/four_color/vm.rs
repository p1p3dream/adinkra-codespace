#![allow(dead_code)]
//! Vector (VM) supermultiplet: reproduction of Tables 9 and 10 of Gates and
//! Lee, arXiv:2408.09342, using the shared four-color apparatus in `super`.
//!
//! Table 9 is the L-matrix summary (holoraumy hopper recursion X -> Y -> Z -> W).
//! Table 10 is the matching R-matrix summary (primed matrices).
//!
//! Conventions are locked by `super::garden_ok` (the Garden algebra is the
//! arbiter of correct sign/transcription). Every bracket <n1 n2 n3 n4> is built
//! with `super::sp`.
//!
//! The hopper recursion (cyclic index, n+1 wraps 4 -> 1):
//!   X_n = L_{n+1} L_n^{-1}      (Eq 4.4)
//!   Y_n = X_{n+1} X_n^{-1}      (Eq 4.7)
//!   Z_n = Y_{n+1} Y_n^{-1}      (Eq 4.10)
//!   W_n = Z_{n+1} Z_n^{-1}      (Eq 4.12)
//! Each Z_n is +-I (Eq 4.11, the individual tables); Table 13 (the W-matrix
//! summary) gives each W_n = -I4.

use crate::signed_perm::SignedPerm;

// ---------------------------------------------------------------------------
// Table 9: Vector L-matrices and the reference X/Y/Z/W rows.
// ---------------------------------------------------------------------------

/// Vector L-matrices, Table 1 (VM row) / Table 9 top row:
/// <2 -4 1 -3>, <1 3 -2 -4>, <4 2 3 1>, <3 -1 -4 2>.
pub fn vm_l() -> Vec<SignedPerm> {
    vec![
        super::sp(&[2, -4, 1, -3]),
        super::sp(&[1, 3, -2, -4]),
        super::sp(&[4, 2, 3, 1]),
        super::sp(&[3, -1, -4, 2]),
    ]
}

/// Table 9, [Xn] reference row: <3 -4 -1 2>, <-4 -3 2 1>, <3 -4 -1 2>, <4 3 -2 -1>.
pub fn table9_x() -> Vec<SignedPerm> {
    vec![
        super::sp(&[3, -4, -1, 2]),
        super::sp(&[-4, -3, 2, 1]),
        super::sp(&[3, -4, -1, 2]),
        super::sp(&[4, 3, -2, -1]),
    ]
}

/// Table 9, [Yn] reference row: <2 -1 4 -3>, <-2 1 -4 3>, <-2 1 -4 3>, <2 -1 4 -3>.
pub fn table9_y() -> Vec<SignedPerm> {
    vec![
        super::sp(&[2, -1, 4, -3]),
        super::sp(&[-2, 1, -4, 3]),
        super::sp(&[-2, 1, -4, 3]),
        super::sp(&[2, -1, 4, -3]),
    ]
}

/// Table 9, [Zn] reference row: <-1 -2 -3 -4>, <1 2 3 4>, <-1 -2 -3 -4>, <1 2 3 4>.
pub fn table9_z() -> Vec<SignedPerm> {
    vec![
        super::sp(&[-1, -2, -3, -4]),
        super::sp(&[1, 2, 3, 4]),
        super::sp(&[-1, -2, -3, -4]),
        super::sp(&[1, 2, 3, 4]),
    ]
}

/// Table 9, [Wn] reference row: all <-1 -2 -3 -4>.
pub fn table9_w() -> Vec<SignedPerm> {
    vec![super::sp(&[-1, -2, -3, -4]); 4]
}

// ---------------------------------------------------------------------------
// Table 10: Vector R-matrices and the reference X'/Y'/Z'/W' rows.
// ---------------------------------------------------------------------------

/// Vector R-matrices, Table 10 top row:
/// <3 1 -4 -2>, <1 -3 2 -4>, <4 2 3 1>, <-2 4 1 -3>.
pub fn vm_r() -> Vec<SignedPerm> {
    vec![
        super::sp(&[3, 1, -4, -2]),
        super::sp(&[1, -3, 2, -4]),
        super::sp(&[4, 2, 3, 1]),
        super::sp(&[-2, 4, 1, -3]),
    ]
}

/// Table 10, [X'n] reference row: <2 -1 -4 3>, <-4 3 -2 1>, <-2 1 4 -3>, <-4 3 -2 1>.
pub fn table10_x() -> Vec<SignedPerm> {
    vec![
        super::sp(&[2, -1, -4, 3]),
        super::sp(&[-4, 3, -2, 1]),
        super::sp(&[-2, 1, 4, -3]),
        super::sp(&[-4, 3, -2, 1]),
    ]
}

/// Table 10, [Y'n] reference row: <3 4 -1 -2>, <3 4 -1 -2>, <-3 -4 1 2>, <-3 -4 1 2>.
pub fn table10_y() -> Vec<SignedPerm> {
    vec![
        super::sp(&[3, 4, -1, -2]),
        super::sp(&[3, 4, -1, -2]),
        super::sp(&[-3, -4, 1, 2]),
        super::sp(&[-3, -4, 1, 2]),
    ]
}

/// Table 10, [Z'n] reference row: <1 2 3 4>, <-1 -2 -3 -4>, <1 2 3 4>, <-1 -2 -3 -4>.
pub fn table10_z() -> Vec<SignedPerm> {
    vec![
        super::sp(&[1, 2, 3, 4]),
        super::sp(&[-1, -2, -3, -4]),
        super::sp(&[1, 2, 3, 4]),
        super::sp(&[-1, -2, -3, -4]),
    ]
}

/// Table 10, [W'n] reference row: all <-1 -2 -3 -4>.
pub fn table10_w() -> Vec<SignedPerm> {
    vec![super::sp(&[-1, -2, -3, -4]); 4]
}

// ---------------------------------------------------------------------------
// The hopper recursion.
// ---------------------------------------------------------------------------

/// One step of the hopper recursion on a length-4 cyclic list:
/// out_n = m_{n+1} * m_n^{-1}, index mod 4.
fn hop(m: &[SignedPerm]) -> Vec<SignedPerm> {
    (0..4)
        .map(|n| super::matmul(&m[(n + 1) % 4], &m[n].inverse()))
        .collect()
}

/// Compute (X, Y, Z, W) from a length-4 list of base matrices (L or R).
pub fn recursion(base: &[SignedPerm]) -> (Vec<SignedPerm>, Vec<SignedPerm>, Vec<SignedPerm>, Vec<SignedPerm>) {
    let x = hop(base);
    let y = hop(&x);
    let z = hop(&y);
    let w = hop(&z);
    (x, y, z, w)
}

pub fn vm_x() -> Vec<SignedPerm> { recursion(&vm_l()).0 }
pub fn vm_y() -> Vec<SignedPerm> { recursion(&vm_l()).1 }
pub fn vm_z() -> Vec<SignedPerm> { recursion(&vm_l()).2 }
pub fn vm_w() -> Vec<SignedPerm> { recursion(&vm_l()).3 }

pub fn vm_x_primed() -> Vec<SignedPerm> { recursion(&vm_r()).0 }
pub fn vm_y_primed() -> Vec<SignedPerm> { recursion(&vm_r()).1 }
pub fn vm_z_primed() -> Vec<SignedPerm> { recursion(&vm_r()).2 }
pub fn vm_w_primed() -> Vec<SignedPerm> { recursion(&vm_r()).3 }

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------

fn all_eq(a: &[SignedPerm], b: &[SignedPerm]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

/// Real pass/fail summary of the VM reproduction.
pub fn report() -> String {
    let l = vm_l();
    let r = vm_r();
    let mut ok = true;

    ok &= super::garden_ok(&l);
    ok &= super::garden_ok(&r);

    let (lx, ly, lz, lw) = recursion(&l);
    ok &= all_eq(&lx, &table9_x());
    ok &= all_eq(&ly, &table9_y());
    ok &= all_eq(&lz, &table9_z());
    ok &= all_eq(&lw, &table9_w());
    ok &= lz.iter().all(|z| z.is_identity() || z.is_neg_identity());
    ok &= lw.iter().all(|w| w.is_neg_identity());

    let (rx, ry, rz, rw) = recursion(&r);
    ok &= all_eq(&rx, &table10_x());
    ok &= all_eq(&ry, &table10_y());
    ok &= all_eq(&rz, &table10_z());
    ok &= all_eq(&rw, &table10_w());
    ok &= rz.iter().all(|z| z.is_identity() || z.is_neg_identity());
    ok &= rw.iter().all(|w| w.is_neg_identity());

    if ok {
        "vm: PASS Tables 9 and 10 (L,R Garden algebra + X/Y/Z/W recursion; Z_n in {+I,-I}, W_n = -I4)".to_string()
    } else {
        "vm: FAIL (see cargo test four_color::vm for the specific mismatch)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (a) Garden algebra is the transcription arbiter.
    #[test]
    fn vm_l_satisfies_garden_algebra() {
        assert!(
            super::super::garden_ok(&vm_l()),
            "VM L-matrices (Table 9) must satisfy the Garden algebra"
        );
    }

    #[test]
    fn vm_r_satisfies_garden_algebra() {
        assert!(
            super::super::garden_ok(&vm_r()),
            "VM R-matrices (Table 10) must satisfy the Garden algebra"
        );
    }

    // (b) Recursion structural asserts: Z_n = +-I (Eq 4.11), W_n = -I4 (Table 13).
    #[test]
    fn vm_l_recursion_z_and_w_shape() {
        let (_x, _y, z, w) = recursion(&vm_l());
        for zn in &z {
            assert!(
                zn.is_identity() || zn.is_neg_identity(),
                "each Z_n must be +I or -I"
            );
        }
        for wn in &w {
            assert!(wn.is_neg_identity(), "each W_n must be -I4");
        }
    }

    #[test]
    fn vm_r_recursion_z_and_w_shape() {
        let (_x, _y, z, w) = recursion(&vm_r());
        for zn in &z {
            assert!(
                zn.is_identity() || zn.is_neg_identity(),
                "each Z'_n must be +I or -I"
            );
        }
        for wn in &w {
            assert!(wn.is_neg_identity(), "each W'_n must be -I4");
        }
    }

    // (c) Table 9 exact reproduction. The reference rows are the OCR of the
    // paper's Table 9; the recursion is the arbiter for any OCR ambiguity.
    #[test]
    fn vm_l_matches_table9() {
        let (x, y, z, w) = recursion(&vm_l());
        assert!(all_eq(&x, &table9_x()), "X_n must match Table 9");
        assert!(all_eq(&y, &table9_y()), "Y_n must match Table 9");
        assert!(all_eq(&z, &table9_z()), "Z_n must match Table 9");
        assert!(all_eq(&w, &table9_w()), "W_n must match Table 9");
    }

    // (d) Table 10 exact reproduction (R / primed matrices).
    #[test]
    fn vm_r_matches_table10() {
        let (x, y, z, w) = recursion(&vm_r());
        assert!(all_eq(&x, &table10_x()), "X'_n must match Table 10");
        assert!(all_eq(&y, &table10_y()), "Y'_n must match Table 10");
        assert!(all_eq(&z, &table10_z()), "Z'_n must match Table 10");
        assert!(all_eq(&w, &table10_w()), "W'_n must match Table 10");
    }

    #[test]
    fn report_is_pass() {
        assert!(report().starts_with("vm: PASS"), "report was: {}", report());
    }
}
