#![allow(dead_code)]
//! G-matrix solver for Gates and Lee, arXiv:2408.09342 (Eq 8.2 and 8.3).
//!
//! For a supermultiplet with L-matrices L1..L4, form the dense sum
//! L = L1 + L2 + L3 + L4 and define A = L * L1^{-1}. The G-matrix satisfies
//! G^2 * L1 = L, i.e. G^2 = A. G has entries in {-1, 0, 1} and is a GENERAL
//! matrix (not a signed permutation): it typically carries several nonzeros
//! per row. The same construction on the R side gives A_(R) = (sum R) * R1^{-1}.
//!
//! `g_matrices(&A)` returns ALL such G in a deterministic order. It does NOT
//! enumerate 3^(n^2) candidates. Instead it runs an entry-by-entry backtracking
//! search pruned by two brackets, each of which is a PROVABLE necessary
//! condition of G^2 = A (so completeness is preserved), and it verifies the
//! exact equation G^2 == A at every leaf:
//!
//!   1. Commutation.  G^2 = A implies G*A = G*G^2 = G^3 = G^2*G = A*G, so any
//!      solution commutes with A. G*A = A*G is LINEAR in the entries of G, so it
//!      propagates far more tightly than the quadratic equation alone. During the
//!      partial assignment we bracket each entry (G*A - A*G)[i][j] by its known
//!      partial value plus/minus the slack from still-unassigned entries; if 0
//!      falls outside the bracket the branch is dead.
//!   2. Quadratic.  The defining equation itself, (G^2)[i][j] = A[i][j],
//!      bracketed the same way over the unassigned entries.
//!
//! The commutation prune collapses the search by orders of magnitude. It makes
//! the block-diagonal CLS enumeration tractable (via the 4x4 blocks); it does
//! NOT make the full 12x12 search tractable, the full count includes cross-block
//! solutions and is combinatorially large and not enumerated (see the CLS
//! handling below). Completeness at n=4 is proven by an inline test that compares
//! the search output against a full 3^16 brute force.

use super::{IntMat, imm};
use std::io::Write;

// ---------------------------------------------------------------------------
// A = (sum of L) * L1^{-1}
// ---------------------------------------------------------------------------

/// Transpose of a dense integer matrix (used as L1^{-1} for a signed-permutation
/// L1, whose inverse equals its transpose).
fn transpose(m: &IntMat) -> IntMat {
    let n = m.len();
    let mut t = vec![vec![0i32; n]; n];
    for i in 0..n {
        for j in 0..n {
            t[j][i] = m[i][j];
        }
    }
    t
}

/// A = (L1 + ... + Lm) * L1^{-1}. Each L is a dense signed-permutation matrix,
/// so L1^{-1} is exactly the transpose of L1.
pub fn a_of(ls: &[IntMat]) -> IntMat {
    debug_assert!(!ls.is_empty());
    let sum = super::dsum(ls);
    let l1_inv = transpose(&ls[0]);
    imm(&sum, &l1_inv)
}

// ---------------------------------------------------------------------------
// g_matrices: all G in {-1,0,1}^(n x n) with G^2 == A, deterministic order
// ---------------------------------------------------------------------------

struct Search<'a> {
    n: usize,
    a: &'a IntMat,
    g: Vec<Vec<i32>>,
    set: Vec<Vec<bool>>,        // set[i][j] true once G[i][j] is fixed
    order: Vec<(usize, usize)>, // assignment order (leading-principal-submatrix spiral)
    out: Vec<IntMat>,
    cap: Option<usize>, // stop once `out` reaches this many solutions (None = all)
}

/// Leading-principal-submatrix ("spiral") assignment order: fill the k-th row and
/// k-th column of the growing top-left k x k block before enlarging it. Assigning
/// entries this way makes the commutation and quadratic equations among the
/// low-indexed rows/columns tighten as early as possible, which prunes the
/// search far harder than a plain row-major sweep. The order is a fixed
/// permutation of all (i, j), so the enumeration stays fully deterministic.
fn spiral_order(n: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(n * n);
    for k in 0..n {
        for j in 0..=k {
            order.push((k, j)); // new row, columns 0..=k
        }
        for i in 0..k {
            order.push((i, k)); // new column, rows 0..k
        }
    }
    order
}

impl<'a> Search<'a> {
    /// Bracket every commutation equation (G*A - A*G)[i][j] = 0 over the current
    /// partial assignment. `partial` accumulates the contribution of already-set
    /// entries; `slack` accumulates the maximal magnitude the still-unset entries
    /// can contribute (each unset entry is in {-1,0,1}). If 0 is outside
    /// [partial - slack, partial + slack] for any (i,j), the branch cannot lead
    /// to a commuting matrix and is pruned. This is a necessary condition of
    /// G^2 = A, so pruning here never discards a real solution.
    fn commutes_possible(&self) -> bool {
        let n = self.n;
        for i in 0..n {
            for j in 0..n {
                let mut partial = 0i64;
                let mut slack = 0i64;
                for k in 0..n {
                    // (G*A)[i][j] term: G[i][k] * A[k][j]
                    let coef = self.a[k][j] as i64;
                    if self.set[i][k] {
                        partial += self.g[i][k] as i64 * coef;
                    } else {
                        slack += coef.abs();
                    }
                    // -(A*G)[i][j] term: -A[i][k] * G[k][j]
                    let coef2 = self.a[i][k] as i64;
                    if self.set[k][j] {
                        partial -= coef2 * self.g[k][j] as i64;
                    } else {
                        slack += coef2.abs();
                    }
                }
                if partial - slack > 0 || 0 > partial + slack {
                    return false;
                }
            }
        }
        true
    }

    /// Bracket every quadratic equation (G^2)[i][j] = A[i][j] over the current
    /// partial assignment. An unset G[i][k] or G[k][j] in a product term
    /// contributes at most 1 in magnitude, so each unresolved product term adds
    /// 1 to the slack. If A[i][j] is outside [partial - slack, partial + slack]
    /// the branch is dead. Also a necessary condition, so completeness holds.
    fn square_possible(&self) -> bool {
        let n = self.n;
        for i in 0..n {
            for j in 0..n {
                let mut partial = 0i64;
                let mut slack = 0i64;
                for k in 0..n {
                    let a_known = self.set[i][k];
                    let b_known = self.set[k][j];
                    if a_known && self.g[i][k] == 0 {
                        continue; // whole product term is 0
                    }
                    if b_known && self.g[k][j] == 0 {
                        continue;
                    }
                    if a_known && b_known {
                        partial += self.g[i][k] as i64 * self.g[k][j] as i64;
                    } else {
                        slack += 1;
                    }
                }
                let target = self.a[i][j] as i64;
                if partial - slack > target || target > partial + slack {
                    return false;
                }
            }
        }
        true
    }

    /// Both bracket tests together (the full feasibility of the current partial
    /// assignment).
    fn feasible(&self) -> bool {
        self.commutes_possible() && self.square_possible()
    }

    /// Which of {-1, 0, 1} keep the partial assignment feasible if placed at
    /// (r, c). Returned in ascending order so downstream branching stays
    /// deterministic. Assumes (r, c) is currently unset; leaves it unset.
    fn feasible_values(&mut self, r: usize, c: usize) -> Vec<i32> {
        let mut ok = Vec::with_capacity(3);
        for &v in &[-1i32, 0, 1] {
            self.g[r][c] = v;
            self.set[r][c] = true;
            if self.feasible() {
                ok.push(v);
            }
            self.set[r][c] = false;
            self.g[r][c] = 0;
        }
        ok
    }

    /// Unit propagation: repeatedly fix any unset entry that has exactly one
    /// feasible value, until a fixpoint. Returns Err (with any entries it fixed
    /// left set) never; instead it returns Ok(fixed) on success or, on a dead
    /// assignment (some unset entry has zero feasible values), rolls back the
    /// entries it fixed and returns Err. Each fixing is a forced consequence of
    /// the necessary bracket conditions, so it never removes a real solution.
    fn propagate(&mut self, from_idx: usize) -> Result<Vec<(usize, usize)>, ()> {
        let mut fixed: Vec<(usize, usize)> = Vec::new();
        let mut changed = true;
        while changed {
            changed = false;
            for k in from_idx..self.order.len() {
                let (r, c) = self.order[k];
                if self.set[r][c] {
                    continue;
                }
                let vals = self.feasible_values(r, c);
                if vals.is_empty() {
                    // Dead: undo everything we fixed here.
                    for &(fr, fc) in &fixed {
                        self.set[fr][fc] = false;
                        self.g[fr][fc] = 0;
                    }
                    return Err(());
                }
                if vals.len() == 1 {
                    self.g[r][c] = vals[0];
                    self.set[r][c] = true;
                    fixed.push((r, c));
                    changed = true;
                }
            }
        }
        Ok(fixed)
    }

    fn recurse(&mut self, idx: usize) {
        if let Some(cap) = self.cap {
            if self.out.len() >= cap {
                return;
            }
        }
        // Advance past entries already fixed by propagation.
        let mut idx = idx;
        while idx < self.order.len() {
            let (r, c) = self.order[idx];
            if self.set[r][c] {
                idx += 1;
            } else {
                break;
            }
        }
        if idx == self.order.len() {
            // Leaf: verify the exact equation before accepting.
            if imm(&self.g, &self.g) == *self.a {
                self.out.push(self.g.clone());
            }
            return;
        }
        let (r, c) = self.order[idx];
        // Deterministic value order: -1, 0, 1.
        for v in self.feasible_values(r, c) {
            self.g[r][c] = v;
            self.set[r][c] = true;
            // Propagate forced entries; only recurse if still consistent.
            match self.propagate(idx + 1) {
                Ok(fixed) => {
                    self.recurse(idx + 1);
                    for &(fr, fc) in &fixed {
                        self.set[fr][fc] = false;
                        self.g[fr][fc] = 0;
                    }
                }
                Err(()) => {}
            }
            self.set[r][c] = false;
            self.g[r][c] = 0;
        }
    }
}

/// All matrices G with entries in {-1, 0, 1} and G^2 == A, in a deterministic
/// order (leading-principal-submatrix assignment order, value order -1, 0, 1).
/// Uses commutation and quadratic bracket propagation, not blind 3^(n^2)
/// enumeration.
pub fn g_matrices(a: &IntMat) -> Vec<IntMat> {
    g_matrices_capped(a, None)
}

/// Same as `g_matrices`, but stops after collecting `cap` solutions (in the same
/// deterministic order). `None` enumerates the complete set. Useful when the full
/// solution set is combinatorially large (the CLS case) but a verified sample
/// suffices.
pub fn g_matrices_capped(a: &IntMat, cap: Option<usize>) -> Vec<IntMat> {
    let n = a.len();
    let mut s = Search {
        n,
        a,
        g: vec![vec![0i32; n]; n],
        set: vec![vec![false; n]; n],
        order: spiral_order(n),
        out: Vec::new(),
        cap,
    };
    s.recurse(0);
    s.out
}

// ---------------------------------------------------------------------------
// Block-structured helpers (for A that is block-diagonal, e.g. the CLS A)
// ---------------------------------------------------------------------------

/// Connected components of the symmetric support graph of A: i ~ j when
/// A[i][j] != 0 or A[j][i] != 0. Each component is one diagonal block of A when
/// A is (up to this partition) block diagonal. Returns the components as sorted
/// index lists, in ascending order of their least index.
fn support_components(a: &IntMat) -> Vec<Vec<usize>> {
    let n = a.len();
    let mut comp = vec![usize::MAX; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        let id = comps.len();
        let mut stack = vec![start];
        comp[start] = id;
        let mut members = Vec::new();
        while let Some(u) = stack.pop() {
            members.push(u);
            for v in 0..n {
                if comp[v] == usize::MAX && (a[u][v] != 0 || a[v][u] != 0) {
                    comp[v] = id;
                    stack.push(v);
                }
            }
        }
        members.sort_unstable();
        comps.push(members);
    }
    comps
}

/// True when A restricted to the given components is block diagonal, i.e. every
/// nonzero of A lies inside a single component (no cross-component entries).
fn is_block_diagonal(a: &IntMat, comps: &[Vec<usize>]) -> bool {
    let n = a.len();
    let mut of = vec![0usize; n];
    for (c, members) in comps.iter().enumerate() {
        for &m in members {
            of[m] = c;
        }
    }
    for i in 0..n {
        for j in 0..n {
            if a[i][j] != 0 && of[i] != of[j] {
                return false;
            }
        }
    }
    true
}

/// Extract the square principal submatrix of A on the given index set.
fn submatrix(a: &IntMat, idx: &[usize]) -> IntMat {
    idx.iter()
        .map(|&i| idx.iter().map(|&j| a[i][j]).collect())
        .collect()
}

/// All block-diagonal G with G^2 == A, assembled from the per-block G-solutions,
/// when A is block diagonal with the given components. This is the complete set
/// of BLOCK-DIAGONAL solutions (a subset of all solutions; cross-block solutions
/// also exist for A whose blocks share spectra, see write_cls_artifact). Each
/// returned matrix is verified to square to A.
fn block_diagonal_g_matrices(a: &IntMat, comps: &[Vec<usize>]) -> Vec<IntMat> {
    let n = a.len();
    // Per-block solution lists.
    let per_block: Vec<Vec<IntMat>> = comps
        .iter()
        .map(|idx| g_matrices(&submatrix(a, idx)))
        .collect();
    // Cartesian product assembled into full n x n block-diagonal matrices.
    let mut out: Vec<IntMat> = Vec::new();
    let mut choice = vec![0usize; comps.len()];
    loop {
        let mut g = vec![vec![0i32; n]; n];
        for (b, idx) in comps.iter().enumerate() {
            let sub = &per_block[b][choice[b]];
            for (bi, &i) in idx.iter().enumerate() {
                for (bj, &j) in idx.iter().enumerate() {
                    g[i][j] = sub[bi][bj];
                }
            }
        }
        debug_assert!(imm(&g, &g) == *a);
        out.push(g);
        // Advance the odometer over block choices.
        let mut k = 0;
        loop {
            if k == comps.len() {
                return out;
            }
            choice[k] += 1;
            if choice[k] < per_block[k].len() {
                break;
            }
            choice[k] = 0;
            k += 1;
        }
    }
}

/// Number of block-diagonal G with G^2 == A (product of per-block counts).
fn block_diagonal_count(a: &IntMat, comps: &[Vec<usize>]) -> usize {
    comps
        .iter()
        .map(|idx| g_matrices(&submatrix(a, idx)).len())
        .product()
}

// ---------------------------------------------------------------------------
// Minimal-multiplet input matrices (transcribed from the arXiv LaTeX e-print)
// ---------------------------------------------------------------------------

fn dense_sp(addr: &[i32]) -> IntMat {
    super::dense(&super::sp(addr))
}

/// CM (chiral) L-matrices, dense.
fn cm_l() -> Vec<IntMat> {
    vec![
        dense_sp(&[1, -4, 2, -3]),
        dense_sp(&[2, 3, -1, -4]),
        dense_sp(&[3, -2, -4, 1]),
        dense_sp(&[4, 1, 3, 2]),
    ]
}
/// CM (chiral) R-matrices, dense.
fn cm_r() -> Vec<IntMat> {
    vec![
        dense_sp(&[1, 3, -4, -2]),
        dense_sp(&[-3, 1, 2, -4]),
        dense_sp(&[4, -2, 1, -3]),
        dense_sp(&[2, 4, 3, 1]),
    ]
}
/// VM (vector) L-matrices, dense.
fn vm_l() -> Vec<IntMat> {
    vec![
        dense_sp(&[2, -4, 1, -3]),
        dense_sp(&[1, 3, -2, -4]),
        dense_sp(&[4, 2, 3, 1]),
        dense_sp(&[3, -1, -4, 2]),
    ]
}
/// VM (vector) R-matrices, dense.
fn vm_r() -> Vec<IntMat> {
    vec![
        dense_sp(&[3, 1, -4, -2]),
        dense_sp(&[1, -3, 2, -4]),
        dense_sp(&[4, 2, 3, 1]),
        dense_sp(&[-2, 4, 1, -3]),
    ]
}
/// TM (tensor) L-matrices, dense.
fn tm_l() -> Vec<IntMat> {
    vec![
        dense_sp(&[1, -3, -4, -2]),
        dense_sp(&[2, 4, -3, 1]),
        dense_sp(&[3, 1, 2, -4]),
        dense_sp(&[4, -2, 1, 3]),
    ]
}
/// TM (tensor) R-matrices, dense.
fn tm_r() -> Vec<IntMat> {
    vec![
        dense_sp(&[1, -4, -2, -3]),
        dense_sp(&[4, 1, -3, 2]),
        dense_sp(&[2, 3, 1, -4]),
        dense_sp(&[3, -2, 4, 1]),
    ]
}

// ---------------------------------------------------------------------------
// CLS 12x12 A-matrices and artifact
// ---------------------------------------------------------------------------

/// CLS L-matrices as dense integer matrices (Appendix C, block-diagonal basis).
/// Reconstructed from the signed-permutation form already committed in cls.rs,
/// which `super::garden_ok` validates.
fn cls_l_dense() -> Vec<IntMat> {
    super::cls::cls_l_matrices()
        .iter()
        .map(super::dense)
        .collect()
}
/// CLS R-matrices as dense integer matrices (Appendix C).
fn cls_r_dense() -> Vec<IntMat> {
    super::cls::cls_r_matrices()
        .iter()
        .map(super::dense)
        .collect()
}

/// Verify every G in `gs` squares to `a`.
fn all_square_to(gs: &[IntMat], a: &IntMat) -> bool {
    gs.iter().all(|g| imm(g, g) == *a)
}

/// Result of the CLS G-matrix analysis for one side (L or R).
pub struct ClsSide {
    pub a: IntMat,
    pub components: Vec<Vec<usize>>,
    pub block_diagonal_count: usize, // exact, complete for the block-diagonal class
    pub cross_block_exists: bool,    // proven by an explicit construction
    pub cross_block_example: Option<IntMat>, // a concrete non block-diagonal G, verified
    pub sample: Vec<IntMat>,         // verified block-diagonal solutions
}

/// Analyze one CLS side: A = (sum L) * L1^{-1}, its diagonal blocks, the exact
/// count of block-diagonal G (product of per-block counts), whether cross-block
/// solutions also exist, and a verified sample of solutions.
///
/// Honesty note: the CLS A here is block diagonal with three 4x4 blocks that
/// share the same spectrum. The complete set of G with G^2 = A therefore does
/// NOT reduce to the block-diagonal solutions: because the blocks share a
/// spectrum, intertwining maps between blocks survive, so cross-block solutions
/// exist and in fact dominate. The full 12x12 count is combinatorially large and
/// is not enumerated in full here. What we report exactly and verify:
///   - block_diagonal_count: the exact number of BLOCK-DIAGONAL solutions
///     (a rigorous lower bound on the full count), every one verified G^2 = A;
///   - cross_block_exists / cross_block_example: a concrete non block-diagonal G
///     with G^2 = A, built explicitly from intertwiners between two blocks
///     (verified), proving the solution set is strictly larger than the
///     block-diagonal one.
fn analyze_cls_side(ls: &[IntMat]) -> ClsSide {
    let a = a_of(ls);
    let comps = support_components(&a);
    assert!(
        is_block_diagonal(&a, &comps),
        "CLS A must be block diagonal on its support components"
    );
    let bd_count = block_diagonal_count(&a, &comps);
    let sample = block_diagonal_g_matrices(&a, &comps)
        .into_iter()
        .take(64)
        .collect::<Vec<_>>();
    assert!(
        all_square_to(&sample, &a),
        "every CLS sample G must square to A"
    );

    // Cross-block existence: construct a concrete non block-diagonal solution.
    let cross_block_example = find_cross_block_solution(&a, &comps);
    if let Some(ref g) = cross_block_example {
        assert!(imm(g, g) == a, "cross-block example must square to A");
    }
    let cross_block_exists = cross_block_example.is_some();

    ClsSide {
        a,
        components: comps,
        block_diagonal_count: bd_count,
        cross_block_exists,
        cross_block_example,
        sample,
    }
}

/// All {-1, 0, 1} matrices X with X * C == D * X (intertwiners from C to D), in a
/// deterministic order. Same backtracking engine as g_matrices but with a linear
/// (not quadratic) constraint, so it is fast at these sizes. When C == D this is
/// the commutant of C. pub(super) so gmatrix_full can reuse it for the exact
/// entry alphabets of the full CLS enumeration.
pub(super) fn intertwiners(d: &IntMat, c: &IntMat) -> Vec<IntMat> {
    // We reuse the entry-backtracking idea specialized to the single linear
    // bracket (X*C - D*X)[i][j] = 0.
    let n = d.len();
    let mut g = vec![vec![0i32; n]; n];
    let mut set = vec![vec![false; n]; n];
    let order = spiral_order(n);
    let mut out = Vec::new();

    fn possible(g: &[Vec<i32>], set: &[Vec<bool>], d: &IntMat, c: &IntMat) -> bool {
        let n = d.len();
        for i in 0..n {
            for j in 0..n {
                let mut partial = 0i64;
                let mut slack = 0i64;
                for k in 0..n {
                    let coef = c[k][j] as i64; // X[i][k]*C[k][j]
                    if set[i][k] {
                        partial += g[i][k] as i64 * coef;
                    } else {
                        slack += coef.abs();
                    }
                    let coef2 = d[i][k] as i64; // -D[i][k]*X[k][j]
                    if set[k][j] {
                        partial -= coef2 * g[k][j] as i64;
                    } else {
                        slack += coef2.abs();
                    }
                }
                if partial - slack > 0 || 0 > partial + slack {
                    return false;
                }
            }
        }
        true
    }

    fn rec(
        idx: usize,
        order: &[(usize, usize)],
        g: &mut Vec<Vec<i32>>,
        set: &mut Vec<Vec<bool>>,
        d: &IntMat,
        c: &IntMat,
        out: &mut Vec<IntMat>,
    ) {
        if idx == order.len() {
            if imm(g, c) == imm(d, g) {
                out.push(g.clone());
            }
            return;
        }
        let (r, col) = order[idx];
        for v in [-1i32, 0, 1] {
            g[r][col] = v;
            set[r][col] = true;
            if possible(g, set, d, c) {
                rec(idx + 1, order, g, set, d, c, out);
            }
            set[r][col] = false;
            g[r][col] = 0;
        }
    }

    rec(0, &order, &mut g, &mut set, d, c, &mut out);
    out
}

/// Construct an explicit cross-block (non block-diagonal) G with G^2 == A, if one
/// exists, by a pure "swap" between the first two diagonal blocks. Choose
/// intertwiners X (X*B1 = B0*X) and Y (Y*B0 = B1*Y) with X*Y = B0 and Y*X = B1;
/// then the 2x2 block matrix [[0, X], [Y, 0]] squares to diag(B0, B1). Fill the
/// remaining diagonal blocks with per-block square roots. Returns the full n x n
/// G (verified G^2 == A) or None. This is fast: the intertwiner search is a per-
/// block linear problem.
fn find_cross_block_solution(a: &IntMat, comps: &[Vec<usize>]) -> Option<IntMat> {
    if comps.len() < 2 {
        return None;
    }
    let n = a.len();
    let b0 = submatrix(a, &comps[0]);
    let b1 = submatrix(a, &comps[1]);
    let x_candidates = intertwiners(&b0, &b1); // X*B1 = B0*X
    let y_candidates = intertwiners(&b1, &b0); // Y*B0 = B1*Y
    for x in &x_candidates {
        for y in &y_candidates {
            if imm(x, y) == b0 && imm(y, x) == b1 {
                // Assemble: block(0<-1)=X at (comps[0],comps[1]),
                // block(1<-0)=Y at (comps[1],comps[0]), other diagonal blocks a
                // per-block square root, off elsewhere 0.
                let mut g = vec![vec![0i32; n]; n];
                for (bi, &i) in comps[0].iter().enumerate() {
                    for (bj, &j) in comps[1].iter().enumerate() {
                        g[i][j] = x[bi][bj];
                    }
                }
                for (bi, &i) in comps[1].iter().enumerate() {
                    for (bj, &j) in comps[0].iter().enumerate() {
                        g[i][j] = y[bi][bj];
                    }
                }
                for c in comps.iter().skip(2) {
                    let sub = submatrix(a, c);
                    let roots = g_matrices(&sub);
                    if roots.is_empty() {
                        return None;
                    }
                    let root = &roots[0];
                    for (bi, &i) in c.iter().enumerate() {
                        for (bj, &j) in c.iter().enumerate() {
                            g[i][j] = root[bi][bj];
                        }
                    }
                }
                if imm(&g, &g) == *a {
                    return Some(g);
                }
            }
        }
    }
    None
}

/// Compute the CLS analysis for both sides and write an atomic JSON artifact.
/// Returns (block_diagonal_count_L, block_diagonal_count_R).
pub fn write_cls_artifact(path: &str) -> std::io::Result<(usize, usize)> {
    let l = analyze_cls_side(&cls_l_dense());
    let r = analyze_cls_side(&cls_r_dense());

    let comps_json = |c: &Vec<Vec<usize>>| -> String {
        let parts: Vec<String> = c
            .iter()
            .map(|m| {
                let cells: Vec<String> = m.iter().map(|v| v.to_string()).collect();
                format!("[{}]", cells.join(","))
            })
            .collect();
        format!("[{}]", parts.join(","))
    };
    let sample_l: Vec<&IntMat> = l.sample.iter().collect();
    let sample_r: Vec<&IntMat> = r.sample.iter().collect();
    let example_json = |e: &Option<IntMat>| -> String {
        match e {
            Some(m) => mat_json(m),
            None => "null".to_string(),
        }
    };

    let json = format!(
        "{{\n  \"source\": \"arXiv:2408.09342 Eq 8.2, CLS Appendix C block-diagonal basis\",\n  \"dim\": 12,\n  \"note\": \"A is block diagonal with three 4x4 blocks that share spectrum. The block-diagonal solutions are the COMPLETE block-diagonal subset (count = 12^3 per side) and a rigorous lower bound on the full count; every one is verified G^2 = A. Cross-block solutions also exist: cross_block_example is a concrete non block-diagonal G with G^2 = A, built from block intertwiners. The full 12x12 solution count is combinatorially large and is not enumerated in full.\",\n  \"A_L\": {},\n  \"A_R\": {},\n  \"components_L\": {},\n  \"components_R\": {},\n  \"block_diagonal_count_L\": {},\n  \"block_diagonal_count_R\": {},\n  \"cross_block_exists_L\": {},\n  \"cross_block_exists_R\": {},\n  \"cross_block_example_L\": {},\n  \"cross_block_example_R\": {},\n  \"sample_is\": \"block-diagonal solutions, each verified G^2 = A\",\n  \"sample_count_L\": {},\n  \"sample_count_R\": {},\n  \"g_sample_L\": {},\n  \"g_sample_R\": {}\n}}\n",
        mat_json(&l.a),
        mat_json(&r.a),
        comps_json(&l.components),
        comps_json(&r.components),
        l.block_diagonal_count,
        r.block_diagonal_count,
        l.cross_block_exists,
        r.cross_block_exists,
        example_json(&l.cross_block_example),
        example_json(&r.cross_block_example),
        l.sample.len(),
        r.sample.len(),
        mats_json(&sample_l),
        mats_json(&sample_r),
    );

    // Atomic write: temp file then rename.
    let tmp = format!("{path}.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(json.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok((l.block_diagonal_count, r.block_diagonal_count))
}

fn mat_json(m: &IntMat) -> String {
    let rows: Vec<String> = m
        .iter()
        .map(|r| {
            let cells: Vec<String> = r.iter().map(|v| v.to_string()).collect();
            format!("[{}]", cells.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

fn mats_json(ms: &[&IntMat]) -> String {
    let items: Vec<String> = ms.iter().map(|m| mat_json(m)).collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

/// One-line summary: minimal-multiplet G-counts and the CLS analysis.
pub fn report() -> String {
    let cm_ln = g_matrices(&a_of(&cm_l())).len();
    let vm_ln = g_matrices(&a_of(&vm_l())).len();
    let tm_ln = g_matrices(&a_of(&tm_l())).len();
    let cm_rn = g_matrices(&a_of(&cm_r())).len();
    let vm_rn = g_matrices(&a_of(&vm_r())).len();
    let tm_rn = g_matrices(&a_of(&tm_r())).len();

    let l = analyze_cls_side(&cls_l_dense());
    let r = analyze_cls_side(&cls_r_dense());

    format!(
        "gmatrix: minimal G-counts L/R CM={cm_ln}/{cm_rn} VM={vm_ln}/{vm_rn} \
         TM={tm_ln}/{tm_rn} (paper: 12 each). Solver = commutation + quadratic \
         bracket backtracking with unit propagation (complete; verified against \
         3^16 brute force at n=4). CLS dim=12: A is block diagonal (three 4x4 \
         blocks, shared spectrum). Block-diagonal G-counts A_(L)={} A_(R)={} \
         (each = 12^3, every solution verified G^2=A). Cross-block solutions \
         also exist (L={}, R={}) and dominate, so the full 12x12 count is \
         combinatorially large and not enumerated in full.",
        l.block_diagonal_count, r.block_diagonal_count, l.cross_block_exists, r.cross_block_exists,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn squares_to(g: &IntMat, a: &IntMat) -> bool {
        imm(g, g) == *a
    }

    #[test]
    fn a_of_uses_transpose_inverse_for_signed_perm_l1() {
        // For a signed permutation L1, L1 * L1^T = I. a_of builds L1^{-1} as the
        // transpose; check L1 * transpose(L1) == I on the CM L1.
        let l1 = &cm_l()[0];
        let prod = imm(l1, &transpose(l1));
        let n = l1.len();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(prod[i][j], if i == j { 1 } else { 0 });
            }
        }
    }

    #[test]
    fn cm_l_has_exactly_twelve_g_matrices() {
        let a = a_of(&cm_l());
        let gs = g_matrices(&a);
        assert_eq!(
            gs.len(),
            12,
            "paper states 12 G-matrices per supermultiplet"
        );
        for g in &gs {
            assert!(squares_to(g, &a), "each CM-L G must square to A");
        }
    }

    #[test]
    fn vm_l_has_exactly_twelve_g_matrices() {
        let a = a_of(&vm_l());
        let gs = g_matrices(&a);
        assert_eq!(gs.len(), 12);
        for g in &gs {
            assert!(squares_to(g, &a), "each VM-L G must square to A");
        }
    }

    #[test]
    fn tm_l_has_exactly_twelve_g_matrices() {
        let a = a_of(&tm_l());
        let gs = g_matrices(&a);
        assert_eq!(gs.len(), 12);
        for g in &gs {
            assert!(squares_to(g, &a), "each TM-L G must square to A");
        }
    }

    #[test]
    fn r_side_also_has_exactly_twelve_g_matrices() {
        for (name, r) in [("CM", cm_r()), ("VM", vm_r()), ("TM", tm_r())] {
            let a = a_of(&r);
            let gs = g_matrices(&a);
            assert_eq!(gs.len(), 12, "{name} R side should also give 12 G-matrices");
            for g in &gs {
                assert!(squares_to(g, &a), "{name}-R G must square to A");
            }
        }
    }

    #[test]
    fn paper_chiral_g_is_present() {
        // Eq 8.3 example G for the corrected CM: G^2 * L1 = sum L.
        let g_paper: IntMat = vec![
            vec![1, 0, 1, 0],
            vec![0, -1, 0, 1],
            vec![0, -1, 0, -1],
            vec![1, 0, -1, 0],
        ];
        let a = a_of(&cm_l());
        assert!(
            squares_to(&g_paper, &a),
            "the paper's chiral G must square to A_(CM-L)"
        );
        let gs = g_matrices(&a);
        assert!(
            gs.iter().any(|g| g == &g_paper),
            "the paper's chiral G must appear in g_matrices(A_(CM-L))"
        );
    }

    /// Reference: full 3^16 enumeration of all 4x4 sign matrices with G^2 == A.
    fn brute_force_n4(a: &IntMat) -> Vec<IntMat> {
        let vals = [-1i32, 0, 1];
        let mut out = Vec::new();
        let mut g = vec![vec![0i32; 4]; 4];
        // 16 entries, base-3 odometer.
        let total: u64 = 3u64.pow(16);
        for code in 0..total {
            let mut c = code;
            for i in 0..4 {
                for j in 0..4 {
                    g[i][j] = vals[(c % 3) as usize];
                    c /= 3;
                }
            }
            if imm(&g, &g) == *a {
                out.push(g.clone());
            }
        }
        out
    }

    #[test]
    fn completeness_matches_brute_force_at_n4() {
        // Proves the constrained search misses nothing: the pruned set equals the
        // exhaustive 3^16 set for CM-L.
        let a = a_of(&cm_l());
        let mut fast = g_matrices(&a);
        let mut brute = brute_force_n4(&a);
        fast.sort();
        brute.sort();
        assert_eq!(
            fast, brute,
            "constrained search must equal full 3^16 brute force at n=4"
        );
        assert_eq!(brute.len(), 12);
    }

    #[test]
    fn commutation_is_necessary_for_solutions() {
        // Sanity check of the prune's justification: every returned G commutes
        // with A (G*A == A*G).
        let a = a_of(&cm_l());
        for g in g_matrices(&a) {
            assert_eq!(imm(&g, &a), imm(&a, &g), "solution G must commute with A");
        }
    }

    #[test]
    fn cls_l_matrices_satisfy_garden_algebra() {
        // Guard: the dense CLS inputs come from the garden-valid signed perms.
        assert!(super::super::garden_ok(&super::super::cls::cls_l_matrices()));
    }

    #[test]
    fn cls_a_is_block_diagonal_with_three_4x4_blocks() {
        for ls in [cls_l_dense(), cls_r_dense()] {
            let a = a_of(&ls);
            let comps = support_components(&a);
            assert!(
                is_block_diagonal(&a, &comps),
                "CLS A must be block diagonal"
            );
            assert_eq!(comps.len(), 3, "CLS A has three diagonal blocks");
            for c in &comps {
                assert_eq!(c.len(), 4, "each CLS block is 4x4");
            }
        }
    }

    #[test]
    fn cls_dim12_g_search_is_nonempty_and_valid() {
        // The complete solver is exact and fast on each 4x4 block (12 each); the
        // block-diagonal assembly gives a nonempty, verified set of 12x12 G with
        // G^2 = A. This runs in well under a second, unlike a blind full-matrix
        // enumeration (the full 12x12 count is combinatorially large).
        for ls in [cls_l_dense(), cls_r_dense()] {
            let a = a_of(&ls);
            let comps = support_components(&a);
            let gs = block_diagonal_g_matrices(&a, &comps);
            assert!(!gs.is_empty(), "CLS must yield at least one G");
            assert_eq!(
                gs.len(),
                12usize.pow(3),
                "block-diagonal count = 12^3 = 1728"
            );
            assert!(
                all_square_to(&gs, &a),
                "every CLS block-diagonal G squares to A"
            );
        }
    }

    #[test]
    fn cls_has_cross_block_solutions() {
        // The three CLS blocks share a spectrum, so cross-block (non block
        // diagonal) solutions exist. We build one explicitly and verify it.
        for ls in [cls_l_dense(), cls_r_dense()] {
            let a = a_of(&ls);
            let comps = support_components(&a);
            let g = find_cross_block_solution(&a, &comps)
                .expect("CLS must admit a cross-block G (blocks share spectrum)");
            assert!(squares_to(&g, &a), "cross-block G must square to A");
            // Confirm it really is non block-diagonal.
            let mut of = vec![0usize; a.len()];
            for (ci, m) in comps.iter().enumerate() {
                for &x in m {
                    of[x] = ci;
                }
            }
            let has_off =
                (0..a.len()).any(|i| (0..a.len()).any(|j| g[i][j] != 0 && of[i] != of[j]));
            assert!(has_off, "the constructed G must have cross-block entries");
        }
    }

    #[test]
    fn cls_artifact_writes_and_reports_counts() {
        std::fs::create_dir_all("results").ok();
        let path = "results/four_color_cls_gmatrix.json";
        let (nl, nr) = write_cls_artifact(path).expect("artifact write");
        assert_eq!(nl, 12usize.pow(3), "block-diagonal count_L = 1728");
        assert_eq!(nr, 12usize.pow(3), "block-diagonal count_R = 1728");
        let contents = std::fs::read_to_string(path).expect("artifact readable");
        assert!(contents.contains("\"block_diagonal_count_L\""));
        assert!(contents.contains(&format!("\"block_diagonal_count_L\": {nl}")));
        assert!(contents.contains(&format!("\"block_diagonal_count_R\": {nr}")));
        assert!(contents.contains("\"cross_block_exists_L\": true"));
    }
}
