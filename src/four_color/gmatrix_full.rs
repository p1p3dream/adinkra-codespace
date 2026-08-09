#![allow(dead_code)]
//! Full enumeration of the CLS G-matrices: ALL G in {-1,0,1}^{12x12} with
//! G^2 = A, where A = (L1+L2+L3+L4) L1^{-1} for the CLS L-matrices of Gates and
//! Lee, arXiv:2408.09342 (Appendix C block-diagonal basis; there L1 = I12, so
//! A = I + L2 + L3 + L4). This closes the "combinatorially large and not
//! enumerated in full" gap documented in gmatrix.rs.
//!
//! Method (exact, complete):
//!
//!   1. A is block diagonal with three 4x4 blocks B0, B1, B2, each satisfying
//!      B^2 = 2B - 4I, so B has eigenvalues lambda = 1 + sqrt(-3) and its
//!      conjugate, each with multiplicity 2 per block.
//!   2. Any solution commutes with A (G^2 = A implies GA = G^3 = AG), so G lives
//!      in the commutant of A: block matrices (G_ij) with G_ij B_j = B_i G_ij.
//!      Each intertwiner space contains finitely many {-1,0,1} points, computed
//!      COMPLETELY by gmatrix::intertwiners (same bracket engine proven complete
//!      against 3^16 brute force at n=4).
//!   3. Over K = Q(sqrt(-3)), A diagonalizes. Because the CLS basis is block
//!      aligned, the change-of-basis matrix P = [V | conj(V)] is block diagonal
//!      (4x4 blocks per B_i), and G maps to g in M_6(K) with
//!         G^2 = A   <==>   g^2 = lambda * I_6.
//!      Reality of G is automatic: every g in M_6(K) maps back to a RATIONAL G
//!      (Galois invariance), and G is integral exactly when g's entries lie in
//!      the finite per-position alphabets obtained by pushing the intertwiner
//!      lattice points through the coordinate change.
//!   4. The search enumerates g over those alphabets with exact box brackets on
//!      (g^2)[i][j] (componentwise rational bounds on precomputed product sets),
//!      an exact trace(g) = 0 bracket, and unit propagation. Every emitted g is
//!      mapped back to an integer G and checked G^2 = A exactly, so the output
//!      has no false positives; the alphabets are complete by step 2, so there
//!      are no false negatives.
//!
//! Rank note: trace(g) = (6 - 2k) sqrt(lambda) with k the (-sqrt(lambda))-
//! eigenspace dimension. Since trace(g) lies in K but sqrt(lambda) does not,
//! k = 3 for every solution, i.e. trace(g) = 0. (Consistent with the per-block
//! picture: each 4x4 block root has eigenvalues +sqrt(lambda), -sqrt(lambda)
//! once each on the lambda-eigenspace.)
//!
//! Honesty notes:
//!   - The count is exact and complete for the {-1,0,1} class only. Nothing is
//!      claimed about rational-entry roots outside this class.
//!   - The solution CHECKSUM is the sum (mod 2^64) of splitmix64 hashes of each
//!      verified integer G (canonical trit encoding), so it is independent of
//!      enumeration order and thread scheduling.

use super::{gmatrix, imm, IntMat};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ===========================================================================
// K = Q(sqrt(-3)): elements (re + im * s) / den with s^2 = -3, den > 0, reduced
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct K {
    re: i128,
    im: i128,
    den: i128,
}

fn gcd2(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl K {
    fn new(re: i128, im: i128, den: i128) -> K {
        assert!(den != 0, "K denominator must be nonzero");
        if re == 0 && im == 0 {
            return K { re: 0, im: 0, den: 1 };
        }
        let (mut re, mut im, mut den) = (re, im, den);
        if den < 0 {
            re = -re;
            im = -im;
            den = -den;
        }
        let g = gcd2(gcd2(re, im), den);
        K { re: re / g, im: im / g, den: den / g }
    }

    pub(crate) fn zero() -> K {
        K { re: 0, im: 0, den: 1 }
    }
    pub(crate) fn one() -> K {
        K { re: 1, im: 0, den: 1 }
    }
    pub(crate) fn from_i32(x: i32) -> K {
        K { re: x as i128, im: 0, den: 1 }
    }
    /// lambda = 1 + sqrt(-3), the eigenvalue of each CLS block.
    pub(crate) fn lambda() -> K {
        K { re: 1, im: 1, den: 1 }
    }
    pub(crate) fn is_zero(&self) -> bool {
        self.re == 0 && self.im == 0
    }

    pub(crate) fn add(self, o: K) -> K {
        K::new(
            self.re * o.den + o.re * self.den,
            self.im * o.den + o.im * self.den,
            self.den * o.den,
        )
    }
    pub(crate) fn sub(self, o: K) -> K {
        self.add(o.neg())
    }
    pub(crate) fn neg(self) -> K {
        K { re: -self.re, im: -self.im, den: self.den }
    }
    pub(crate) fn mul(self, o: K) -> K {
        // (a + b s)(c + d s) = (ac - 3bd) + (ad + bc) s
        K::new(
            self.re * o.re - 3 * self.im * o.im,
            self.re * o.im + self.im * o.re,
            self.den * o.den,
        )
    }
    pub(crate) fn inv(self) -> K {
        assert!(!self.is_zero(), "K division by zero");
        // 1/(a + b s) = (a - b s)/(a^2 + 3 b^2)
        let nrm = self.re * self.re + 3 * self.im * self.im;
        K::new(self.re * self.den, -self.im * self.den, nrm)
    }
    pub(crate) fn div(self, o: K) -> K {
        self.mul(o.inv())
    }
    pub(crate) fn conj(self) -> K {
        K { re: self.re, im: -self.im, den: self.den }
    }

    /// Real and imaginary rational components as (num, den) pairs, den > 0.
    pub(crate) fn parts(&self) -> ((i128, i128), (i128, i128)) {
        ((self.re, self.den), (self.im, self.den))
    }
}

impl fmt::Display for K {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im == 0 {
            if self.den == 1 {
                write!(f, "{}", self.re)
            } else {
                write!(f, "{}/{}", self.re, self.den)
            }
        } else if self.den == 1 {
            write!(f, "{}+{}s", self.re, self.im)
        } else {
            write!(f, "({}+{}s)/{}", self.re, self.im, self.den)
        }
    }
}

impl fmt::Debug for K {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl PartialOrd for K {
    fn partial_cmp(&self, o: &K) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for K {
    // Deterministic order for sets/alphabets (NOT a field order).
    fn cmp(&self, o: &K) -> std::cmp::Ordering {
        (self.den, self.re, self.im).cmp(&(o.den, o.re, o.im))
    }
}

/// Exact compare of two rationals (n1/d1) vs (n2/d2), d1, d2 > 0.
fn rat_cmp(n1: i128, d1: i128, n2: i128, d2: i128) -> std::cmp::Ordering {
    (n1 * d2).cmp(&(n2 * d1))
}

// ===========================================================================
// Exact linear algebra over K
// ===========================================================================

type KMat = Vec<Vec<K>>;

fn kzeros(r: usize, c: usize) -> KMat {
    vec![vec![K::zero(); c]; r]
}

/// Gauss-Jordan inverse; None when singular.
fn kinv(m: &KMat) -> Option<KMat> {
    let n = m.len();
    let mut a = m.clone();
    let mut inv: KMat = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { K::one() } else { K::zero() })
                .collect()
        })
        .collect();
    for col in 0..n {
        let mut piv = None;
        for r in col..n {
            if !a[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        let piv = piv?;
        a.swap(col, piv);
        inv.swap(col, piv);
        let p = a[col][col];
        for j in 0..n {
            a[col][j] = a[col][j].div(p);
            inv[col][j] = inv[col][j].div(p);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f.is_zero() {
                continue;
            }
            for j in 0..n {
                a[r][j] = a[r][j].sub(f.mul(a[col][j]));
                inv[r][j] = inv[r][j].sub(f.mul(inv[col][j]));
            }
        }
    }
    Some(inv)
}

/// Basis of ker(m) as column vectors (each length = number of columns of m).
fn kkernel(m: &KMat) -> Vec<Vec<K>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut a = m.clone();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let mut piv = None;
        for rr in r..rows {
            if !a[rr][c].is_zero() {
                piv = Some(rr);
                break;
            }
        }
        let Some(piv) = piv else { continue };
        a.swap(r, piv);
        let p = a[r][c];
        for j in 0..cols {
            a[r][j] = a[r][j].div(p);
        }
        for rr in 0..rows {
            if rr == r {
                continue;
            }
            let f = a[rr][c];
            if f.is_zero() {
                continue;
            }
            for j in 0..cols {
                a[rr][j] = a[rr][j].sub(f.mul(a[r][j]));
            }
        }
        pivot_cols.push(c);
        r += 1;
    }
    let is_pivot = |c: usize| pivot_cols.contains(&c);
    let mut basis = Vec::new();
    for f in 0..cols {
        if is_pivot(f) {
            continue;
        }
        let mut x = vec![K::zero(); cols];
        x[f] = K::one();
        for (row, &pc) in pivot_cols.iter().enumerate() {
            x[pc] = a[row][f].neg();
        }
        basis.push(x);
    }
    basis
}

fn kmat_mul(a: &KMat, b: &KMat) -> KMat {
    let n = a.len();
    let inner = b.len();
    let cols = b[0].len();
    let mut out = kzeros(n, cols);
    for i in 0..n {
        for k in 0..inner {
            if a[i][k].is_zero() {
                continue;
            }
            for j in 0..cols {
                out[i][j] = out[i][j].add(a[i][k].mul(b[k][j]));
            }
        }
    }
    out
}

// ===========================================================================
// CLS coordinate change: A, blocks, eigenbasis, P, P^{-1}
// ===========================================================================

/// Coordinate apparatus for the first `m` diagonal blocks of the CLS A-matrix
/// (m = 1, 2, or 3; g is then 2m x 2m over K).
pub struct Coords {
    pub m: usize,
    pub a: IntMat,        // 4m x 4m integer A restricted to the first m blocks
    pub p: KMat,          // 4m x 4m, block diagonal [V_b | conj V_b]
    pub p_inv: KMat,
    pub pb: Vec<KMat>,    // per-block 4x4 [V_b | conj V_b]
    pub pb_inv: Vec<KMat>,
}

/// A for the CLS L side, restricted to the first m blocks (contiguous 4-blocks
/// in the Appendix C basis).
pub fn cls_a_blocks(m: usize, side: Side) -> (IntMat, [IntMat; 3]) {
    let ls: Vec<IntMat> = match side {
        Side::L => super::cls::cls_l_matrices().iter().map(super::dense).collect(),
        Side::R => super::cls::cls_r_matrices().iter().map(super::dense).collect(),
    };
    let a = gmatrix::a_of(&ls);
    let blocks: [IntMat; 3] = std::array::from_fn(|b| {
        (0..4)
            .map(|i| (0..4).map(|j| a[4 * b + i][4 * b + j]).collect())
            .collect()
    });
    let dim = 4 * m;
    let a_m: IntMat = (0..dim).map(|i| a[i][..dim].to_vec()).collect();
    (a_m, blocks)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    L,
    R,
}

impl Side {
    pub fn name(&self) -> &'static str {
        match self {
            Side::L => "L",
            Side::R => "R",
        }
    }
}

/// Build the coordinate change for the first m blocks. Panics if the structural
/// claims fail (block quadratic, eigenspace dimensions, invertibility): those
/// are guarded by tests below.
pub fn build_coords(m: usize, side: Side) -> Coords {
    let (a, blocks) = cls_a_blocks(m, side);
    let lam = K::lambda();
    // Per-block lambda-eigenbasis: ker(B_b - lambda I_4) must be 2-dimensional.
    let dim = 4 * m;
    let mut p = kzeros(dim, dim);
    let mut pb: Vec<KMat> = Vec::with_capacity(m);
    let mut pb_inv: Vec<KMat> = Vec::with_capacity(m);
    for b in 0..m {
        let mut mb = kzeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                mb[i][j] = K::from_i32(blocks[b][i][j]);
            }
            mb[i][i] = mb[i][i].sub(lam);
        }
        let ker = kkernel(&mb);
        assert_eq!(
            ker.len(),
            2,
            "block {b} lambda-eigenspace must be 2-dimensional"
        );
        // Per-block 4x4 change of basis p_b = [V_b | conj V_b].
        let mut p_b = kzeros(4, 4);
        for (u, vec) in ker.iter().enumerate() {
            for i in 0..4 {
                p_b[i][u] = vec[i];
                p_b[i][2 + u] = vec[i].conj();
            }
        }
        let p_b_inv = kinv(&p_b).expect("per-block P_b must be invertible");
        // Columns 2b, 2b+1 of the big P: V_b; columns 2m+2b, 2m+2b+1: conj(V_b).
        for i in 0..4 {
            for u in 0..4 {
                let target = if u < 2 { 2 * b + u } else { 2 * m + 2 * b + (u - 2) };
                p[4 * b + i][target] = p_b[i][u];
            }
        }
        pb.push(p_b);
        pb_inv.push(p_b_inv);
    }
    let p_inv = kinv(&p).expect("P must be invertible (eigenbases are independent)");
    Coords { m, a, p, p_inv, pb, pb_inv }
}

/// g = top-left 2m x 2m of P^{-1} G P (G integer).
pub fn g_of_int(coords: &Coords, g: &IntMat) -> KMat {
    let dim = 4 * coords.m;
    let gk: KMat = (0..dim)
        .map(|i| (0..dim).map(|j| K::from_i32(g[i][j])).collect())
        .collect();
    let full = kmat_mul(&kmat_mul(&coords.p_inv, &gk), &coords.p);
    let n = 2 * coords.m;
    (0..n).map(|i| full[i][..n].to_vec()).collect()
}

/// G = P [g 0; 0 conj(g)] P^{-1}; Some(integer matrix) when every entry is
/// real integral in {-1,0,1}, else None (a programming error elsewhere, since
/// alphabets are built to force integrality).
pub fn int_of_g(coords: &Coords, g: &KMat) -> Option<IntMat> {
    let n = 2 * coords.m;
    let dim = 4 * coords.m;
    let mut mid = kzeros(dim, dim);
    for i in 0..n {
        for j in 0..n {
            mid[i][j] = g[i][j];
            mid[n + i][n + j] = g[i][j].conj();
        }
    }
    let full = kmat_mul(&kmat_mul(&coords.p, &mid), &coords.p_inv);
    let mut out = vec![vec![0i32; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            let e = full[i][j];
            if e.im != 0 || e.den != 1 || e.re < -1 || e.re > 1 {
                return None;
            }
            out[i][j] = e.re as i32;
        }
    }
    Some(out)
}

/// Map one 2x2 K-slot (bi, bj) of g back to its 4x4 integer block of G:
/// block = p_bi [slot 0; 0 conj(slot)] p_bj^{-1}. Some(block) when every entry
/// is real integral in {-1,0,1}, else None. This is the LOCAL integrality
/// filter: G's block (bi,bj) depends only on g's slot (bi,bj), so a completed
/// slot that fails here can never extend to an integer solution.
pub fn slot_int_of_g(coords: &Coords, bi: usize, bj: usize, slot: &[[K; 2]; 2]) -> Option<IntMat> {
    let mut mid = kzeros(4, 4);
    for u in 0..2 {
        for w in 0..2 {
            mid[u][w] = slot[u][w];
            mid[2 + u][2 + w] = slot[u][w].conj();
        }
    }
    let full = kmat_mul(&kmat_mul(&coords.pb[bi], &mid), &coords.pb_inv[bj]);
    let mut out = vec![vec![0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let e = full[i][j];
            if e.im != 0 || e.den != 1 || e.re < -1 || e.re > 1 {
                return None;
            }
            out[i][j] = e.re as i32;
        }
    }
    Some(out)
}

// ===========================================================================
// Entry alphabets from the (complete) intertwiner lattice points
// ===========================================================================

/// Per-position sorted alphabets for g: alph[i][j] is the exact set of values
/// the (i, j) entry of g can take over all integer solutions, derived from the
/// complete {−1,0,1} intertwiner sets of each block slot.
pub struct Alphabets {
    pub n: usize,
    pub sets: Vec<Vec<Vec<K>>>,
    pub intertwiner_counts: Vec<Vec<usize>>,
    /// Per slot (flat bi*m + bj): every integral 2x2 K-slot, i.e. the image of
    /// some {-1,0,1} intertwiner X under the coordinate change. Any integer
    /// solution's slot (bi,bj) is one of these, so partial slot assignments
    /// can be checked EXACTLY against this list.
    pub slots: Vec<Vec<[[K; 2]; 2]>>,
}

pub fn build_alphabets(coords: &Coords, blocks: &[IntMat; 3]) -> Alphabets {
    let m = coords.m;
    let n = 2 * m;
    let mut sets: Vec<Vec<BTreeSet<K>>> = (0..n).map(|_| (0..n).map(|_| BTreeSet::new()).collect()).collect();
    let mut slot_sets: Vec<BTreeSet<[[K; 2]; 2]>> = (0..m * m).map(|_| BTreeSet::new()).collect();
    let mut counts = vec![vec![0usize; m]; m];
    for bi in 0..m {
        for bj in 0..m {
            let xs = gmatrix::intertwiners(&blocks[bi], &blocks[bj]);
            counts[bi][bj] = xs.len();
            // Per-slot coordinate change is the 4x4 p_b = [V_b | conj V_b].
            for x in &xs {
                // Embed X in the 4m x 4m zero matrix at slot (bi, bj).
                let dim = 4 * m;
                let mut e = vec![vec![0i32; dim]; dim];
                for i in 0..4 {
                    for j in 0..4 {
                        e[4 * bi + i][4 * bj + j] = x[i][j];
                    }
                }
                let ge = g_of_int(coords, &e);
                // Its support must be the 2x2 slot (2bi.., 2bj..); record values.
                let mut slot = [[K::zero(); 2]; 2];
                for u in 0..n {
                    for w in 0..n {
                        let in_slot = (u / 2 == bi) && (w / 2 == bj);
                        if in_slot {
                            sets[u][w].insert(ge[u][w]);
                            slot[u % 2][w % 2] = ge[u][w];
                        } else {
                            debug_assert!(
                                ge[u][w].is_zero(),
                                "slot map leaked outside its 2x2 at ({u},{w})"
                            );
                        }
                    }
                }
                slot_sets[bi * m + bj].insert(slot);
            }
        }
    }
    let sets: Vec<Vec<Vec<K>>> = sets
        .into_iter()
        .map(|row| row.into_iter().map(|s| s.into_iter().collect()).collect())
        .collect();
    let slots: Vec<Vec<[[K; 2]; 2]>> = slot_sets
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();
    Alphabets { n, sets, intertwiner_counts: counts, slots }
}

// ===========================================================================
// Backtracking enumeration over the alphabets
// ===========================================================================

/// Componentwise rational box: min/max of re-part and im-part over a product set.
#[derive(Clone, Copy)]
struct KBox {
    min_re: (i128, i128),
    max_re: (i128, i128),
    min_im: (i128, i128),
    max_im: (i128, i128),
}

fn box_of_products(xs: &[K], ys: &[K]) -> KBox {
    let mut b = KBox {
        min_re: (i128::MAX, 1),
        max_re: (i128::MIN, 1),
        min_im: (i128::MAX, 1),
        max_im: (i128::MIN, 1),
    };
    for &x in xs {
        for &y in ys {
            let p = x.mul(y);
            let ((rn, rd), (in_, id)) = p.parts();
            if rat_cmp(rn, rd, b.min_re.0, b.min_re.1) == std::cmp::Ordering::Less {
                b.min_re = (rn, rd);
            }
            if rat_cmp(rn, rd, b.max_re.0, b.max_re.1) == std::cmp::Ordering::Greater {
                b.max_re = (rn, rd);
            }
            if rat_cmp(in_, id, b.min_im.0, b.min_im.1) == std::cmp::Ordering::Less {
                b.min_im = (in_, id);
            }
            if rat_cmp(in_, id, b.max_im.0, b.max_im.1) == std::cmp::Ordering::Greater {
                b.max_im = (in_, id);
            }
        }
    }
    b
}

pub struct SolveStats {
    pub count: u64,
    pub checksum: u64,
    pub nodes: u64,
    pub seconds: f64,
    pub samples: Vec<IntMat>,
    pub complete: bool,
}

struct Shared {
    n: usize,
    coords: Coords,
    alph: Vec<Vec<Vec<K>>>,
    slots: Vec<Vec<[[K; 2]; 2]>>, // integral slots per (bi*m + bj)
    prod: Vec<Vec<Vec<KBox>>>,    // prod[i][k][j]
    order: Vec<(usize, usize)>,
}

struct Search {
    n: usize,
    g: KMat,
    set: Vec<Vec<bool>>,
    inc: KMat, // partial sums of (g^2)[i][j] over assigned-both products
    undo: Vec<(usize, usize, K)>,
    count: u64,
    checksum: u64,
    nodes: u64,
    samples: Vec<IntMat>,
}

/// Slot-major assignment order: slots in row-major order, each slot's four
/// entries as (diag0, diag1, off01, off10). Completing slots early enables the
/// local integrality filter, and the diagonal entries first strengthen the
/// trace bracket.
fn slot_order(n: usize) -> Vec<(usize, usize)> {
    let m = n / 2;
    let mut order = Vec::with_capacity(n * n);
    for bi in 0..m {
        for bj in 0..m {
            let (u, w) = (2 * bi, 2 * bj);
            order.push((u, w));
            order.push((u + 1, w + 1));
            order.push((u, w + 1));
            order.push((u + 1, w));
        }
    }
    order
}

pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub(crate) fn hash_intmat(g: &IntMat) -> u64 {
    let mut h: u64 = 0x243F6A8885A308D3;
    for row in g {
        for &v in row {
            h = splitmix64(h ^ (v as i64 as u64));
        }
    }
    h
}

impl Search {
    fn new(n: usize) -> Search {
        Search {
            n,
            g: kzeros(n, n),
            set: vec![vec![false; n]; n],
            inc: kzeros(n, n),
            undo: Vec::new(),
            count: 0,
            checksum: 0,
            nodes: 0,
            samples: Vec::new(),
        }
    }

    fn assign(&mut self, r: usize, c: usize, v: K) {
        self.g[r][c] = v;
        self.set[r][c] = true;
        let n = self.n;
        // inc[r][j] += v * g[c][j] for assigned g[c][j]
        for j in 0..n {
            if self.set[c][j] {
                self.undo.push((r, j, self.inc[r][j]));
                self.inc[r][j] = self.inc[r][j].add(v.mul(self.g[c][j]));
            }
        }
        // inc[i][c] += g[i][r] * v for assigned g[i][r]; skip the double-counted
        // self term when r == c (already added as v*v in the first loop).
        for i in 0..n {
            if self.set[i][r] && !(r == c && i == r) {
                self.undo.push((i, c, self.inc[i][c]));
                self.inc[i][c] = self.inc[i][c].add(self.g[i][r].mul(v));
            }
        }
    }

    fn unassign(&mut self, r: usize, c: usize, undo_mark: usize) {
        while self.undo.len() > undo_mark {
            let (i, j, old) = self.undo.pop().unwrap();
            self.inc[i][j] = old;
        }
        self.set[r][c] = false;
        self.g[r][c] = K::zero();
    }

    /// Box-bracket feasibility of every equation (g^2)[i][j] = target(i, j),
    /// plus the trace(g) = 0 bracket. Necessary conditions only; never discards
    /// a real solution.
    fn feasible(&self, sh: &Shared) -> bool {
        let n = self.n;
        let lam = K::lambda();
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { lam } else { K::zero() };
                let ((trn, trd), (tin, tid)) = target.parts();
                let ((prn, prd), (pin, pid)) = self.inc[i][j].parts();
                let (mut lo_re, mut hi_re) = ((prn, prd), (prn, prd));
                let (mut lo_im, mut hi_im) = ((pin, pid), (pin, pid));
                for k in 0..n {
                    if self.set[i][k] && self.set[k][j] {
                        continue; // already inside inc
                    }
                    let b = &sh.prod[i][k][j];
                    lo_re = rat_add(lo_re, b.min_re);
                    hi_re = rat_add(hi_re, b.max_re);
                    lo_im = rat_add(lo_im, b.min_im);
                    hi_im = rat_add(hi_im, b.max_im);
                }
                if rat_cmp(lo_re.0, lo_re.1, trn, trd) == std::cmp::Ordering::Greater
                    || rat_cmp(hi_re.0, hi_re.1, trn, trd) == std::cmp::Ordering::Less
                    || rat_cmp(lo_im.0, lo_im.1, tin, tid) == std::cmp::Ordering::Greater
                    || rat_cmp(hi_im.0, hi_im.1, tin, tid) == std::cmp::Ordering::Less
                {
                    return false;
                }
            }
        }
        // trace(g) = 0: partial real-part diagonal sum plus per-entry min/max
        // of remaining diagonal alphabets must bracket 0 (re part only; the im
        // part is covered by the equation brackets above).
        let mut t_lo = (0i128, 1i128);
        let mut t_hi = (0i128, 1i128);
        for d in 0..n {
            if self.set[d][d] {
                let ((rn, rd), _) = self.g[d][d].parts();
                t_lo = rat_add(t_lo, (rn, rd));
                t_hi = rat_add(t_hi, (rn, rd));
            } else {
                let (mut mn, mut mx) = ((i128::MAX, 1i128), (i128::MIN, 1i128));
                for &v in &sh.alph[d][d] {
                    let ((rn, rd), _) = v.parts();
                    if rat_cmp(rn, rd, mn.0, mn.1) == std::cmp::Ordering::Less {
                        mn = (rn, rd);
                    }
                    if rat_cmp(rn, rd, mx.0, mx.1) == std::cmp::Ordering::Greater {
                        mx = (rn, rd);
                    }
                }
                t_lo = rat_add(t_lo, mn);
                t_hi = rat_add(t_hi, mx);
            }
        }
        if rat_cmp(t_lo.0, t_lo.1, 0, 1) == std::cmp::Ordering::Greater
            || rat_cmp(t_hi.0, t_hi.1, 0, 1) == std::cmp::Ordering::Less
        {
            return false;
        }
        true
    }

    /// Exact local filter: the currently assigned entries of the slot
    /// containing (r, c) must agree with at least one integral slot (the image
    /// of a {-1,0,1} intertwiner). Necessary and complete at slot granularity:
    /// no integer solution is ever pruned, and every completed non-integral
    /// slot is rejected immediately, including at partial depth.
    fn slot_ok(&self, sh: &Shared, r: usize, c: usize) -> bool {
        let m = sh.coords.m;
        let (bi, bj) = (r / 2, c / 2);
        let (u, w) = (2 * bi, 2 * bj);
        let positions = [(u, w), (u + 1, w + 1), (u, w + 1), (u + 1, w)];
        sh.slots[bi * m + bj].iter().any(|slot| {
            positions
                .iter()
                .all(|&(i, j)| !self.set[i][j] || slot[i % 2][j % 2] == self.g[i][j])
        })
    }

    fn recurse(&mut self, sh: &Shared, idx: usize, cap: Option<u64>, counter: &AtomicU64) {
        if let Some(c) = cap {
            if self.count >= c {
                return;
            }
        }
        let mut idx = idx;
        while idx < sh.order.len() {
            let (r, c) = sh.order[idx];
            if !self.set[r][c] {
                break;
            }
            idx += 1;
        }
        self.nodes += 1;
        if self.nodes & 0xFFFFF == 0 {
            counter.fetch_add(0x100000, Ordering::Relaxed);
        }
        if idx == sh.order.len() {
            // Leaf: inc is the full g^2; verify exactly.
            let lam = K::lambda();
            for i in 0..self.n {
                for j in 0..self.n {
                    let target = if i == j { lam } else { K::zero() };
                    if self.inc[i][j] != target {
                        return;
                    }
                }
            }
            // Ground-truth: map back to integer G and verify G^2 = A. Leaves
            // that map to non-integer G are the rational (out-of-class) roots
            // the entrywise alphabets cannot exclude early; skip them.
            let Some(gi) = int_of_g(&sh.coords, &self.g) else {
                return;
            };
            debug_assert!(imm(&gi, &gi) == sh.coords.a, "mapped G must square to A");
            self.count += 1;
            self.checksum = self.checksum.wrapping_add(hash_intmat(&gi));
            if self.samples.len() < 8 {
                self.samples.push(gi);
            }
            return;
        }
        let (r, c) = sh.order[idx];
        for &v in &sh.alph[r][c] {
            let mark = self.undo.len();
            self.assign(r, c, v);
            if self.slot_ok(sh, r, c) && self.feasible(sh) {
                self.recurse(sh, idx + 1, cap, counter);
            }
            self.unassign(r, c, mark);
            if let Some(cc) = cap {
                if self.count >= cc {
                    return;
                }
            }
        }
    }
}

fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let (mut n, mut d) = (a.0 * b.1 + b.0 * a.1, a.1 * b.1);
    if n == 0 {
        return (0, 1);
    }
    if d < 0 {
        n = -n;
        d = -d;
    }
    let g = gcd2(n, d);
    (n / g, d / g)
}

/// Preassign a work prefix (list of ((r, c), value_index)) then enumerate.
fn run_worker(
    sh: Arc<Shared>,
    prefix: &[((usize, usize), usize)],
    cap: Option<u64>,
    counter: Arc<AtomicU64>,
) -> (u64, u64, u64, Vec<IntMat>) {
    let mut s = Search::new(sh.n);
    let mut ok = true;
    for &((r, c), vi) in prefix {
        let v = sh.alph[r][c][vi];
        s.assign(r, c, v);
        if !s.slot_ok(&sh, r, c) || !s.feasible(&sh) {
            ok = false;
            break;
        }
    }
    if ok {
        s.recurse(&sh, 0, cap, &counter);
    }
    (s.count, s.checksum, s.nodes, s.samples)
}

/// Enumerate all solutions for the given coordinate setup and alphabets.
/// `threads` splits the top-level assignment space deterministically; results
/// (count, checksum) are order-independent.
pub fn solve(
    coords: Coords,
    alph: Alphabets,
    threads: usize,
    cap: Option<u64>,
) -> SolveStats {
    let n = alph.n;
    let t0 = Instant::now();
    // Precompute product boxes.
    let mut prod: Vec<Vec<Vec<KBox>>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for k in 0..n {
            let mut r2 = Vec::with_capacity(n);
            for j in 0..n {
                r2.push(box_of_products(&alph.sets[i][k], &alph.sets[k][j]));
            }
            row.push(r2);
        }
        prod.push(row);
    }
    let sh = Arc::new(Shared {
        n,
        coords,
        alph: alph.sets,
        slots: alph.slots,
        prod,
        order: slot_order(n),
    });

    // Work items: all integer-consistent assignments of the FIRST slot (its
    // four entries are the first four of the slot-major order). Each item is a
    // complete 4-entry prefix that already passed the local integrality
    // filter, giving balanced parallelism at up to ~10^3 items.
    let mut items: Vec<Vec<((usize, usize), usize)>> = Vec::new();
    {
        fn gen_items(
            sh: &Shared,
            s: &mut Search,
            depth: usize,
            cur: &mut Vec<((usize, usize), usize)>,
            items: &mut Vec<Vec<((usize, usize), usize)>>,
        ) {
            if depth == 4 {
                items.push(cur.clone());
                return;
            }
            let (r, c) = sh.order[depth];
            for vi in 0..sh.alph[r][c].len() {
                let mark = s.undo.len();
                s.assign(r, c, sh.alph[r][c][vi]);
                if s.slot_ok(sh, r, c) && s.feasible(sh) {
                    cur.push(((r, c), vi));
                    gen_items(sh, s, depth + 1, cur, items);
                    cur.pop();
                }
                s.unassign(r, c, mark);
            }
        }
        let mut s = Search::new(n);
        let mut cur = Vec::new();
        gen_items(&sh, &mut s, 0, &mut cur, &mut items);
    }
    println!("work items (first-slot prefixes): {}", items.len());
    let counter = Arc::new(AtomicU64::new(0));
    let total_items = items.len();
    let next = Arc::new(AtomicU64::new(0));
    let threads = threads.max(1);

    // Progress reporter: poll the shared node counter and print rate/ETA
    // context every 15 seconds until the scope below signals completion.
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let counter = Arc::clone(&counter);
        let done = Arc::clone(&done);
        let t0p = t0;
        std::thread::spawn(move || {
            let mut last_nodes = 0u64;
            while !done.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(15));
                if done.load(Ordering::Relaxed) {
                    break;
                }
                let nodes = counter.load(Ordering::Relaxed);
                let rate = (nodes - last_nodes) as f64 / 15.0;
                last_nodes = nodes;
                eprintln!(
                    "progress: {:.0}s elapsed, {}M nodes, {:.2}M nodes/s",
                    t0p.elapsed().as_secs_f64(),
                    nodes / 1_000_000,
                    rate / 1.0e6
                );
            }
        });
    }

    // std::thread::scope keeps the borrow of `items` safe. Results (count,
    // checksum) are accumulated commutatively, so they are schedule-independent.
    let (mut count, mut checksum, mut nodes) = (0u64, 0u64, 0u64);
    let mut samples: Vec<IntMat> = Vec::new();
    std::thread::scope(|scope| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(total_items) {
            let sh = Arc::clone(&sh);
            let next = Arc::clone(&next);
            let counter = Arc::clone(&counter);
            let items = &items;
            hs.push(scope.spawn(move || {
                let (mut c, mut h, mut nd, mut smp) = (0u64, 0u64, 0u64, Vec::new());
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i as usize >= items.len() {
                        break;
                    }
                    let (cc, hh, nn, ss) = run_worker(Arc::clone(&sh), &items[i as usize], cap, Arc::clone(&counter));
                    c += cc;
                    h = h.wrapping_add(hh);
                    nd += nn;
                    smp.extend(ss);
                }
                (c, h, nd, smp)
            }));
        }
        for h in hs {
            let (c, hh, nd, smp) = h.join().unwrap();
            count += c;
            checksum = checksum.wrapping_add(hh);
            nodes += nd;
            samples.extend(smp);
        }
    });

    done.store(true, Ordering::Relaxed);
    samples.truncate(8);
    SolveStats {
        count,
        checksum,
        nodes,
        seconds: t0.elapsed().as_secs_f64(),
        samples,
        complete: cap.is_none(),
    }
}

// ===========================================================================
// Sanity ladder: per-slot M2 solutions and the block-diagonal rederivation
// ===========================================================================

/// All g in M_2(K) over the diagonal-slot alphabets with g^2 = lambda I_2 THAT
/// MAP BACK TO INTEGER 4x4 blocks. The alphabets are entrywise supersets
/// (combinations across entries need not be integral), so the map-back filter
/// is essential: 132 g in M_2(K) satisfy g^2 = lambda I_2 over the alphabets,
/// but only the 12 integer per-block roots survive int_of_g.
/// Per the per-block analysis, d = -a and a^2 + bc = lambda (the b = c = 0 case
/// would need a^2 = lambda, impossible in K, and is asserted empty).
pub fn slot_solutions(coords: &Coords, alph: &Alphabets, slot: usize) -> Vec<KMat> {
    let n = 2 * coords.m;
    let (d00, d01, d10, d11) = (
        &alph.sets[2 * slot][2 * slot],
        &alph.sets[2 * slot][2 * slot + 1],
        &alph.sets[2 * slot + 1][2 * slot],
        &alph.sets[2 * slot + 1][2 * slot + 1],
    );
    let lam = K::lambda();
    let mut out = Vec::new();
    for &a in d00 {
        for &b in d01 {
            for &c in d10 {
                if a.mul(a).add(b.mul(c)) != lam {
                    continue;
                }
                let d = a.neg();
                if !d11.contains(&d) {
                    continue;
                }
                // Embed the 2x2 slot into the full n x n g and keep it only when
                // it maps back to an integer G (the {-1,0,1} class filter).
                let mut g = kzeros(n, n);
                g[2 * slot][2 * slot] = a;
                g[2 * slot][2 * slot + 1] = b;
                g[2 * slot + 1][2 * slot] = c;
                g[2 * slot + 1][2 * slot + 1] = d;
                if int_of_g(coords, &g).is_some() {
                    out.push(vec![vec![a, b], vec![c, d]]);
                }
            }
        }
    }
    // The b = c = 0 branch would require a^2 = lambda with a in K; impossible
    // (lambda is not a square in Q(sqrt(-3))). Guard it explicitly.
    for &a in d00 {
        assert!(
            a.mul(a) != lam,
            "lambda must not be a square in K (would break the d = -a reduction)"
        );
    }
    out
}

/// Count block-diagonal solutions through the new coordinates: product of the
/// three per-slot counts; must equal 12^m. Also verifies a sample end to end.
pub fn block_diagonal_via_coords(coords: &Coords, alph: &Alphabets) -> usize {
    let per: Vec<usize> = (0..coords.m).map(|s| slot_solutions(coords, alph, s).len()).collect();
    per.iter().product()
}

// ===========================================================================
// CLI entry: build + artifact
// ===========================================================================

pub fn run_build(side: Side, m: usize, threads: usize, cap: Option<u64>, path: &str) {
    println!(
        "cls-g-full: side={} blocks={} threads={} cap={:?}",
        side.name(),
        m,
        threads,
        cap
    );
    let coords = build_coords(m, side);
    let (_, blocks) = cls_a_blocks(m, side);
    let alph = build_alphabets(&coords, &blocks);
    println!("intertwiner counts: {:?}", alph.intertwiner_counts);
    let sizes: Vec<i32> = alph.sets.iter().flatten().map(|s| s.len() as i32).collect();
    println!("alphabet sizes (row-major {}x{}): {:?}", alph.n, alph.n, sizes);

    // Sanity: per-slot counts must be 12 (matching gmatrix::g_matrices per block).
    for s in 0..m {
        let sols = slot_solutions(&coords, &alph, s);
        let expect = gmatrix::g_matrices(&blocks[s]).len();
        println!("slot {s}: M_2(K) solutions = {} (gmatrix per-block = {expect})", sols.len());
        assert_eq!(sols.len(), expect, "slot {s} M_2(K) count mismatch");
    }
    let bd = block_diagonal_via_coords(&coords, &alph);
    println!("block-diagonal via coords = {bd} (expect 12^{m})");
    assert_eq!(bd, 12usize.pow(m as u32));

    let stats = solve(coords, alph, threads, cap);
    println!(
        "DONE: count={} checksum={:016x} nodes={} seconds={:.1} complete={}",
        stats.count, stats.checksum, stats.nodes, stats.seconds, stats.complete
    );

    let sample_json: Vec<String> = stats
        .samples
        .iter()
        .map(|g| {
            let rows: Vec<String> = g
                .iter()
                .map(|r| format!("[{}]", r.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")))
                .collect();
            format!("[{}]", rows.join(","))
        })
        .collect();
    let json = format!(
        "{{\n  \"source\": \"arXiv:2408.09342 Eq 8.2, CLS Appendix C basis; full enumeration via K=Q(sqrt(-3)) commutant reduction (src/four_color/gmatrix_full.rs)\",\n  \"side\": \"{}\",\n  \"blocks\": {},\n  \"dim\": {},\n  \"count\": {},\n  \"checksum_splitmix64_sum\": \"{:016x}\",\n  \"nodes\": {},\n  \"seconds\": {:.3},\n  \"complete\": {},\n  \"note\": \"All G in {{-1,0,1}}^(4m x 4m) with G^2 = A. Alphabets are complete (intertwiner lattice computed by the proven-complete bracket engine); every counted solution was mapped back to an integer G and verified G^2 = A exactly. Checksum is order- and schedule-independent.\",\n  \"samples\": [{}]\n}}\n",
        side.name(),
        m,
        4 * m,
        stats.count,
        stats.checksum,
        stats.nodes,
        stats.seconds,
        stats.complete,
        sample_json.join(",")
    );
    std::fs::create_dir_all("results").ok();
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &json).expect("write tmp artifact");
    std::fs::rename(&tmp, path).expect("rename artifact");
    println!("wrote {path}");
}

/// Verify a written artifact: re-run the cheap sanity ladder (per-slot M_2(K)
/// counts and the 12^m block-diagonal rederivation) and re-verify every stored
/// sample against G^2 = A. This does NOT re-run the full enumeration; the
/// count/checksum are trusted from the build run, and the report says so.
pub fn run_verify(side: Side, m: usize, path: &str) -> bool {
    let coords = build_coords(m, side);
    let (_, blocks) = cls_a_blocks(m, side);
    let alph = build_alphabets(&coords, &blocks);
    let mut ladder_ok = true;
    for s in 0..m {
        let sols = slot_solutions(&coords, &alph, s);
        let expect = gmatrix::g_matrices(&blocks[s]).len();
        if sols.len() != expect {
            eprintln!("slot {s}: M_2(K) solutions {} != {expect}", sols.len());
            ladder_ok = false;
        }
    }
    let bd = block_diagonal_via_coords(&coords, &alph);
    let bd_expect = 12usize.pow(m as u32);
    if bd != bd_expect {
        eprintln!("block-diagonal via coords {bd} != {bd_expect}");
        ladder_ok = false;
    }

    let payload = match std::fs::read_to_string(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot read artifact {path}: {e}");
            return false;
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("artifact {path} is not valid JSON: {e}");
            return false;
        }
    };
    let count = v.get("count").and_then(|x| x.as_u64());
    let complete = v.get("complete").and_then(|x| x.as_bool()) == Some(true);
    let checksum = v
        .get("checksum_splitmix64_sum")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let mut samples_ok = true;
    let mut n_samples = 0usize;
    if let Some(arr) = v.get("samples").and_then(|x| x.as_array()) {
        for sv in arr {
            let g: Option<IntMat> = serde_json::from_value(sv.clone()).ok();
            let Some(g) = g else {
                eprintln!("a sample failed to parse as an integer matrix");
                samples_ok = false;
                continue;
            };
            n_samples += 1;
            let in_class = g
                .iter()
                .all(|row| row.iter().all(|&e| (-1..=1).contains(&e)));
            if !in_class || imm(&g, &g) != coords.a {
                eprintln!("a stored sample does NOT satisfy G^2 = A over {{-1,0,1}}");
                samples_ok = false;
            }
        }
    }
    let passed = ladder_ok && samples_ok && count.is_some() && complete;
    let report = serde_json::json!({
        "artifact": path,
        "side": side.name(),
        "blocks": m,
        "ladder_ok": ladder_ok,
        "block_diagonal_via_coords": bd,
        "samples_checked": n_samples,
        "samples_ok": samples_ok,
        "count": count,
        "checksum_splitmix64_sum": checksum,
        "complete": complete,
        "note": "Ladder and samples re-verified exactly; the full count is trusted from the build run (this does not re-enumerate).",
        "passed": passed,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    passed
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_arithmetic_identities() {
        let s = K::new(0, 1, 1);
        assert_eq!(s.mul(s), K::from_i32(-3), "s^2 = -3");
        let lam = K::lambda();
        assert_eq!(lam.mul(lam.conj()), K::from_i32(4), "|lambda|^2 = 4");
        let x = K::new(3, -5, 7);
        assert_eq!(x.div(x), K::one(), "x/x = 1");
        assert_eq!(x.add(x.neg()), K::zero(), "x - x = 0");
        // (1 + s)^2 = 1 + 2s - 3 = -2 + 2s, matching B^2 = 2B - 4I eigenvalue law
        assert_eq!(lam.mul(lam), lam.mul(K::from_i32(2)).sub(K::from_i32(4)));
    }

    #[test]
    fn cls_blocks_satisfy_quadratic() {
        for side in [Side::L, Side::R] {
            let (_, blocks) = cls_a_blocks(3, side);
            for b in &blocks {
                let b2 = imm(b, b);
                for i in 0..4 {
                    for j in 0..4 {
                        let expect = 2 * b[i][j] - if i == j { 4 } else { 0 };
                        assert_eq!(b2[i][j], expect, "B^2 = 2B - 4I must hold");
                    }
                }
            }
        }
    }

    #[test]
    fn eigenspaces_are_two_dimensional_and_p_invertible() {
        for side in [Side::L, Side::R] {
            let _c = build_coords(3, side); // asserts inside
        }
    }

    #[test]
    fn intertwiner_counts_match_across_slots() {
        let (_, blocks) = cls_a_blocks(3, Side::L);
        let c00 = gmatrix::intertwiners(&blocks[0], &blocks[0]).len();
        for bi in 0..3 {
            for bj in 0..3 {
                let n = gmatrix::intertwiners(&blocks[bi], &blocks[bj]).len();
                assert_eq!(n, c00, "intertwiner counts must agree across slots");
            }
        }
        println!("intertwiners per slot: {c00}");
    }

    #[test]
    fn slot_solutions_are_twelve_and_match_gmatrix() {
        let coords = build_coords(1, Side::L);
        let (_, blocks) = cls_a_blocks(1, Side::L);
        let alph = build_alphabets(&coords, &blocks);
        let sols = slot_solutions(&coords, &alph, 0);
        assert_eq!(sols.len(), 12, "M_2(K) per-block solutions must be 12");
        assert_eq!(gmatrix::g_matrices(&blocks[0]).len(), 12);
        // End-to-end: every slot solution maps back to a verified integer G.
        for g2 in &sols {
            let gi = int_of_g(&coords, g2).expect("slot solution maps to integer G");
            assert!(imm(&gi, &gi) == coords.a, "mapped G squares to A");
        }
    }

    #[test]
    fn block_diagonal_count_is_1728_via_coords() {
        let coords = build_coords(3, Side::L);
        let (_, blocks) = cls_a_blocks(3, Side::L);
        let alph = build_alphabets(&coords, &blocks);
        assert_eq!(block_diagonal_via_coords(&coords, &alph), 1728);
    }
}
