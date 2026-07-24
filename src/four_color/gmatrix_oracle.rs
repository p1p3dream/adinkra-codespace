#![allow(dead_code)]
//! Faithful reproduction of the paper's OWN brute-force G-matrix computation.
//!
//! Gates and Lee (arXiv:2408.09342, Appendix A) define
//!
//!     findMatricesG[A_, p_] :=
//!         Select[Tuples[{-1,0,1}, {n,n}], MatrixPower[#, p] == A &]
//!
//! i.e. enumerate every n x n matrix whose entries lie in {-1,0,1} and keep the
//! ones whose p-th matrix power equals A. For the minimal 4D N=1 supermultiplets
//! they take p = 2 and
//!
//!     A = (L1 + L2 + L3 + L4) * L1^{-1}
//!
//! (Appendix A / B) and report obtaining 12 different G-matrices per
//! supermultiplet, then choose one. For the chiral multiplet the chosen G is
//! Eq (8.3):
//!
//!     [[ 1, 0, 1, 0],
//!      [ 0,-1, 0, 1],
//!      [ 0,-1, 0,-1],
//!      [ 1, 0,-1, 0]]
//!
//! This module is the GROUND-TRUTH ORACLE. It does the honest 3^(n^2) search
//! (fast enough at n = 4: 3^16 = 43,046,721 candidates, with row-by-row early
//! abort) so the fast structural solver in `super::roots` can be checked against
//! it. It does NOT assume the G-matrix is a signed permutation; the search runs
//! over all {-1,0,1} matrices, and (as Eq 8.3 shows and the tests confirm) the
//! solutions are in fact NOT signed permutations.

use super::IntMat;

/// The three values a G-matrix entry may take, in the paper's `Tuples`
/// enumeration order {-1, 0, 1}.
const VALS: [i32; 3] = [-1, 0, 1];

/// A = (L1 + L2 + ... + Ln) * L1^{-1}.
///
/// The L-matrices are dense matrices of orthogonal signed permutations, so
/// L1^{-1} = L1^T (transpose). `ls[0]` is L1. Panics if `ls` is empty.
pub fn a_of(ls: &[IntMat]) -> IntMat {
    assert!(!ls.is_empty(), "a_of needs at least one L-matrix");
    let n = ls[0].len();
    let sum = super::dsum(ls);
    // L1^{-1} = transpose of L1 (orthogonal signed permutation).
    let mut l1_inv = vec![vec![0i32; n]; n];
    for i in 0..n {
        for j in 0..n {
            l1_inv[j][i] = ls[0][i][j];
        }
    }
    super::imm(&sum, &l1_inv)
}

/// Faithful `findMatricesG[A, p]`: enumerate every n x n matrix with entries in
/// {-1,0,1} whose p-th power equals A, returned in the deterministic order of
/// the paper's `Tuples[{-1,0,1}, {n,n}]` scan (row-major, each entry cycling
/// through -1, 0, 1).
///
/// The search is row-by-row with early abort: as soon as a candidate row makes
/// some output entry of A^{(target)} decidable (every row it depends on is
/// placed) and that entry is wrong, the whole subtree is pruned. This keeps the
/// n = 4 case tractable in release builds while enumerating exactly the same
/// candidate set the Mathematica `Select` would.
pub fn brute_g_matrices(a: &IntMat, p: u32) -> Vec<IntMat> {
    let n = a.len();
    assert!(n > 0, "A must be non-empty");
    assert!(a.iter().all(|r| r.len() == n), "A must be square");
    assert!(p >= 1, "p must be >= 1");

    let mut out = Vec::new();
    let mut g = vec![vec![0i32; n]; n];
    // For p == 2 we can prune per placed row (the product entry (i,j) only needs
    // rows i and the rows referenced by row i). For general p we cannot decide
    // any product entry until all rows are placed, so we only check at the leaf.
    fill_row(a, p, n, &mut g, 0, &mut out);
    out
}

/// Recursively fill row `r` of `g`, cycling each of its n entries over
/// {-1,0,1} in `VALS` order, pruning with `partial_ok_p2` (only effective for
/// p == 2) and verifying the full p-th power at the leaf.
fn fill_row(a: &IntMat, p: u32, n: usize, g: &mut IntMat, r: usize, out: &mut Vec<IntMat>) {
    if r == n {
        if pth_power_eq(g, a, p) {
            out.push(g.clone());
        }
        return;
    }
    fill_entry(a, p, n, g, r, 0, out);
}

/// Cycle entry (r, c) through {-1,0,1}; after finishing a row, prune (p==2 only)
/// before descending to the next row.
fn fill_entry(
    a: &IntMat,
    p: u32,
    n: usize,
    g: &mut IntMat,
    r: usize,
    c: usize,
    out: &mut Vec<IntMat>,
) {
    if c == n {
        if p != 2 || partial_ok_p2(a, n, g, r) {
            fill_row(a, p, n, g, r + 1, out);
        }
        return;
    }
    for &v in VALS.iter() {
        g[r][c] = v;
        fill_entry(a, p, n, g, r, c + 1, out);
    }
    g[r][c] = 0;
}

/// For p == 2, once rows 0..=r are placed, an entry (i,j) of G^2 is fully
/// determined iff every k with g[i][k] != 0 has k <= r (row k is placed).
/// Check all such determined entries against A; return false on first mismatch.
fn partial_ok_p2(a: &IntMat, n: usize, g: &IntMat, r: usize) -> bool {
    for i in 0..=r {
        for j in 0..n {
            // Is (G^2)[i][j] fully determined by rows 0..=r?
            let mut ready = true;
            for k in 0..n {
                if g[i][k] != 0 && k > r {
                    ready = false;
                    break;
                }
            }
            if !ready {
                continue;
            }
            let mut s = 0i32;
            for k in 0..n {
                s += g[i][k] * g[k][j];
            }
            if s != a[i][j] {
                return false;
            }
        }
    }
    true
}

/// True iff g^p == a (dense integer p-th power).
fn pth_power_eq(g: &IntMat, a: &IntMat, p: u32) -> bool {
    let n = g.len();
    let mut acc = identity(n);
    for _ in 0..p {
        acc = super::imm(&acc, g);
    }
    acc == *a
}

fn identity(n: usize) -> IntMat {
    let mut m = vec![vec![0i32; n]; n];
    for i in 0..n {
        m[i][i] = 1;
    }
    m
}

/// True iff `m` is a signed permutation matrix: exactly one nonzero (+/-1) in
/// every row and every column.
pub fn is_signed_permutation(m: &IntMat) -> bool {
    let n = m.len();
    for i in 0..n {
        let mut nz = 0;
        for j in 0..n {
            let v = m[i][j];
            if v != 0 {
                nz += 1;
                if v != 1 && v != -1 {
                    return false;
                }
            }
        }
        if nz != 1 {
            return false;
        }
    }
    for j in 0..n {
        let mut nz = 0;
        for i in 0..n {
            if m[i][j] != 0 {
                nz += 1;
            }
        }
        if nz != 1 {
            return false;
        }
    }
    true
}

/// Eq (8.3) of the paper: the chiral G-matrix the authors select.
pub fn eq_8_3() -> IntMat {
    vec![
        vec![1, 0, 1, 0],
        vec![0, -1, 0, 1],
        vec![0, -1, 0, -1],
        vec![1, 0, -1, 0],
    ]
}

/// Dense L-matrices (as `IntMat`) for the minimal multiplets, built from the
/// e-print's verified signed addresses.
fn dense_l(addrs: &[[i32; 4]]) -> Vec<IntMat> {
    addrs.iter().map(|a| super::dense(&super::sp(a))).collect()
}

/// Chiral (CM) L-matrices, e-print CM-L row.
pub fn cm_l() -> Vec<IntMat> {
    dense_l(&[[1, -4, 2, -3], [2, 3, -1, -4], [3, -2, -4, 1], [4, 1, 3, 2]])
}

/// Vector (VM) L-matrices, e-print VM-L row.
pub fn vm_l() -> Vec<IntMat> {
    dense_l(&[[2, -4, 1, -3], [1, 3, -2, -4], [4, 2, 3, 1], [3, -1, -4, 2]])
}

/// Tensor (TM) L-matrices, e-print TM-L row.
pub fn tm_l() -> Vec<IntMat> {
    dense_l(&[[1, -3, -4, -2], [2, 4, -3, 1], [3, 1, 2, -4], [4, -2, 1, 3]])
}

/// Chiral (CM) R-matrices, e-print CM-R row (optional p=2 cross-check).
pub fn cm_r() -> Vec<IntMat> {
    dense_l(&[[1, 3, -4, -2], [-3, 1, 2, -4], [4, -2, 1, -3], [2, 4, 3, 1]])
}

/// Human-readable summary of the oracle result at n = 4, p = 2.
pub fn report() -> String {
    let mut lines = Vec::new();
    for (name, l) in [("CM", cm_l()), ("VM", vm_l()), ("TM", tm_l())] {
        let a = a_of(&l);
        let gs = brute_g_matrices(&a, 2);
        let all_square = gs.iter().all(|g| pth_power_eq(g, &a, 2));
        let any_sp = gs.iter().any(|g| is_signed_permutation(g));
        let has_83 = name != "TM" && gs.iter().any(|g| *g == eq_8_3());
        lines.push(format!(
            "gmatrix_oracle: {name} p=2 -> {} G-matrices (paper claim 12: {}); \
             all square to A: {}; signed permutations: {}; Eq 8.3 present: {}",
            gs.len(),
            if gs.len() == 12 { "MATCH" } else { "MISMATCH" },
            all_square,
            if any_sp { "some" } else { "none (general {-1,0,1})" },
            if name == "TM" {
                "n/a".to_string()
            } else {
                has_83.to_string()
            },
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The paper's own construction A = (sum L) * L1^{-1} for the chiral
    /// multiplet, and the fact that Eq 8.3 squares to it (orientation lock).
    #[test]
    fn eq_8_3_squares_to_cm_a() {
        let a = a_of(&cm_l());
        assert!(
            pth_power_eq(&eq_8_3(), &a, 2),
            "Eq 8.3 G must satisfy G^2 = A for the chiral multiplet"
        );
    }

    /// CM: the brute force returns EXACTLY 12 G-matrices, every one squares to
    /// A, Eq 8.3 is among them, and NONE is a signed permutation.
    #[test]
    fn cm_has_twelve_g_matrices_including_eq_8_3() {
        let a = a_of(&cm_l());
        let gs = brute_g_matrices(&a, 2);
        assert_eq!(gs.len(), 12, "paper claims 12 G-matrices per multiplet (CM)");
        assert!(
            gs.iter().all(|g| pth_power_eq(g, &a, 2)),
            "every returned G must square to A"
        );
        assert!(
            gs.iter().any(|g| *g == eq_8_3()),
            "Eq 8.3's chosen G must be one of the 12"
        );
        // Confirms the G-matrix is NOT a signed permutation (Eq 8.3 has two
        // nonzeros per row).
        assert!(
            gs.iter().all(|g| !is_signed_permutation(g)),
            "none of the 12 CM G-matrices is a signed permutation"
        );
    }

    /// VM: also exactly 12, all squaring to A. (VM shares the same A as CM under
    /// this construction, so Eq 8.3 appears here too.)
    #[test]
    fn vm_has_twelve_g_matrices() {
        let a = a_of(&vm_l());
        let gs = brute_g_matrices(&a, 2);
        assert_eq!(gs.len(), 12, "paper claims 12 G-matrices per multiplet (VM)");
        assert!(gs.iter().all(|g| pth_power_eq(g, &a, 2)));
        assert!(gs.iter().all(|g| !is_signed_permutation(g)));
    }

    /// TM: also exactly 12, all squaring to A, none a signed permutation.
    #[test]
    fn tm_has_twelve_g_matrices() {
        let a = a_of(&tm_l());
        let gs = brute_g_matrices(&a, 2);
        assert_eq!(gs.len(), 12, "paper claims 12 G-matrices per multiplet (TM)");
        assert!(gs.iter().all(|g| pth_power_eq(g, &a, 2)));
        assert!(gs.iter().all(|g| !is_signed_permutation(g)));
    }

    /// Optional R-side cross-check: the chiral R-matrices also yield exactly 12
    /// G-matrices at p = 2, each squaring to its A.
    #[test]
    fn cm_r_side_has_twelve_g_matrices() {
        let a = a_of(&cm_r());
        let gs = brute_g_matrices(&a, 2);
        assert_eq!(gs.len(), 12, "CM R-side also yields 12 G-matrices");
        assert!(gs.iter().all(|g| pth_power_eq(g, &a, 2)));
    }

    /// Determinism: repeated runs return the identical ordered list.
    #[test]
    fn enumeration_is_deterministic() {
        let a = a_of(&cm_l());
        let g1 = brute_g_matrices(&a, 2);
        let g2 = brute_g_matrices(&a, 2);
        assert_eq!(g1, g2, "brute force must be deterministic");
    }

    /// Sanity: the partial-abort prune does not drop valid roots. Every returned
    /// square root of I4 actually squares to I4, and the set is non-empty.
    #[test]
    fn identity_root_set_is_consistent() {
        let id = identity(4);
        let gs = brute_g_matrices(&id, 2);
        assert!(!gs.is_empty());
        assert!(gs.iter().all(|g| pth_power_eq(g, &id, 2)));
    }
}
