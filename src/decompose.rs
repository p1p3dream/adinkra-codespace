//! Irreducible decomposition of reducible Adinkra valise representations
//! (finding F8, "route (b)") and the dense-matrix gadget on the resulting
//! irreducible pieces.
//!
//! References (see REFERENCES.md):
//!   - GR(d,N) Garden algebra / L,R matrices: Doran-Faux-Gates-Hübsch-Iga-Landweber
//!     arXiv:0811.3410; Faux-Gates hep-th/0408004.
//!   - Gadget definition & normalization: Gates-Hübsch et al. arXiv:1508.07546
//!     (JHEP 11(2015)113).
//!   - k=8 D16/E8×E8 validation target (distance spectrum): Arunseangroj-Bedessem-
//!     Gates-Yerger, arXiv:2503.13797 (2025).
//!
//! # Background
//!
//! A valise representation at code-stratum `k` (N colours, d = 2^(N-k-1)) has
//! L-matrices that are d-dimensional signed permutations. For `k < N/2` the rep
//! is **reducible**: `d > dmin(N)`, and it splits into `r = d / dmin(N)` copies
//! of the irreducible GR(dmin, N) Clifford module. The flat per-stratum gadget
//! diagonal `G[R,R] = d/dmin` (see [`crate::holoraumy::gadget`]) is exactly that
//! multiplicity `r`, which is the tell that the rep is reducible.
//!
//! To recover the literature's object (the gadget on *irreducible* modules) we
//! must project each valise rep onto its irreducible summands, restrict every
//! `L_I` onto each summand, and run the gadget there. This module does that.
//!
//! # Method
//!
//! 1. **Commutant (exact).** `M` commutes with all `L_I` iff, for every colour,
//!    `M[a][b] = s_I[a]·s_I[b]·M[p_I[a]][p_I[b]]` (conjugation by a signed
//!    permutation). This couples each matrix cell `(a,b)` to exactly one other
//!    cell `(p_I[a], p_I[b])` with a sign. A signed union-find over the `d²`
//!    cells therefore yields the commutant exactly, in integers, with no
//!    `d²×d²` dense system: `dim(commutant) = #sign-consistent orbits`, and each
//!    consistent orbit is one integer basis matrix. See [`commutant_orbits`].
//!
//! 2. **Isotypic split (iterative refinement).** The commutant is closed under
//!    transpose (the `L_I` are orthogonal), so it contains symmetric elements.
//!    A symmetric commutant element's eigenspaces are L-invariant subreps. We
//!    take a generic symmetric commutant element, diagonalise it (Jacobi), split
//!    by eigenvalue clusters, and recurse on any block whose dimension still
//!    exceeds `dmin`. This converges to `r` invariant subspaces each of
//!    dimension `dmin` for the real/complex/quaternionic Schur types uniformly.
//!
//! 3. **Restriction + dense gadget.** For an orthonormal basis `B` (d×dmin) of a
//!    summand, the restricted operators `L_I|_W = Bᵀ L_I B` are dense real
//!    orthogonal matrices (no longer signed permutations), so the gadget is
//!    recomputed with a dense trace-product. The summand self-gadget is exactly
//!    `1.0` (dmin/dmin), independent of the basis `B`.
//!
//! # Well-definedness caveat (important)
//!
//! The summand **self**-gadget (1.0), the multiplicity `r`, the Garden algebra,
//! and `V²=−I` are all **basis-independent** and are hard-tested below.
//!
//! The **cross**-gadget *between two distinct summands* is **not** basis
//! independent: it depends on the relative orthonormal orientation chosen for
//! each `dmin`-dimensional summand (the `B_c B_{c'}ᵀ` cross term does not reduce
//! to projectors). The k=8 stratum is well-defined only because there
//! `d = dmin` and every rep lives in the shared standard basis with no rotation
//! freedom. A canonical cross-summand / cross-code orientation is the genuinely
//! open part of F8; the cross values reported here are computed in the basis the
//! decomposition happens to produce and must NOT be read as a basis-invariant
//! classification without that canonicalisation.

use crate::holoraumy::dmin;
use crate::lr_matrix::AdinkraRep;
use crate::signed_perm::SignedPerm;

/// Maximum `d` for which dense decomposition is attempted. Dense Jacobi plus
/// restriction is ~O(d³) time and O(d²) memory; at N=16 this admits k=8 (d=128,
/// already irreducible), k=7 (d=256), and k=6 (d=512). Smaller k (d ≥ 1024,
/// up to d=16384 for k=1) are skipped and logged — they need a sparse/iterative
/// rewrite, not a dense path.
pub const MAX_DECOMPOSE_D: usize = 512;

/// Memory budget (bytes) for the dense gadget step of a whole k-stratum. The
/// `d ≤ MAX_DECOMPOSE_D` guard alone is NOT sufficient: the gadget pairs every
/// irreducible summand against every other, so it retains one `DenseHoloraumy`
/// (C(N,2) matrices of dmin × dmin f64 ≈ 15.7 MB at N=16) per summand
/// SIMULTANEOUSLY. That scales with `num_irreps` (= reps × d/dmin), not `d`:
/// k=8 ≈ 8 GB (fits), but k=7 ≈ 36 GB and would silently OOM-kill a 64 GB box.
/// `run_decompose_k` refuses (clean skip) when the estimate exceeds this budget
/// rather than dying. Tune to the host; the real fix for larger strata is a
/// blocked/streamed (or GPU) Gram, not raising this.
pub const MAX_DECOMPOSE_GADGET_BYTES: u64 = 24 * 1024 * 1024 * 1024; // 24 GiB

/// Estimated peak bytes of retained dense holoraumy for `num_irreps` summands at
/// `n` colours (dmin = dmin(n)): num_irreps × C(n,2) × dmin² × 8.
pub fn estimated_gadget_bytes(n: usize, num_irreps: usize) -> u64 {
    let dm = dmin(n) as u64;
    let pairs = (n as u64) * (n as u64 - 1) / 2;
    (num_irreps as u64) * pairs * dm * dm * 8
}

/// Fixed seed for the deterministic PRNG so decomposition is reproducible across
/// runs and threads (each `decompose_rep` call uses a fresh, identically-seeded
/// generator).
const RNG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Max attempts to split a single reducible block before giving up (a generic
/// symmetric commutant element splits a reducible block with probability 1, so
/// this only guards against pathological numerics).
const MAX_SPLIT_ATTEMPTS: usize = 32;

// ===========================================================================
// Dense real matrix (row-major). Hand-rolled to match the project's no-external-
// linear-algebra style (cf. SignedPerm).
// ===========================================================================

/// Row-major dense real matrix: `data[r*cols + c]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        DenseMat { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// The d×d identity.
    pub fn identity(d: usize) -> Self {
        let mut m = DenseMat::zeros(d, d);
        for i in 0..d {
            m.data[i * d + i] = 1.0;
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Densify a signed permutation: row `i` has `sign[i]` at column `perm[i]`.
    pub fn from_signed_perm(sp: &SignedPerm) -> DenseMat {
        let d = sp.dim();
        let mut m = DenseMat::zeros(d, d);
        for i in 0..d {
            m.set(i, sp.perm[i] as usize, sp.sign[i] as f64);
        }
        m
    }

    /// Matrix product `self * other`.
    pub fn matmul(&self, other: &DenseMat) -> DenseMat {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = DenseMat::zeros(m, n);
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                if a == 0.0 {
                    continue;
                }
                let row_b = &other.data[p * n..p * n + n];
                let row_o = &mut out.data[i * n..i * n + n];
                for j in 0..n {
                    row_o[j] += a * row_b[j];
                }
            }
        }
        out
    }

    /// Transpose.
    pub fn transpose(&self) -> DenseMat {
        let mut out = DenseMat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Trace (sum of the diagonal); requires a square matrix.
    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "trace of non-square matrix");
        (0..self.rows).map(|i| self.get(i, i)).sum()
    }

    /// `Tr(self * other)` without forming the product:
    /// `sum_{i,j} self[i][j] * other[j][i]`.
    pub fn trace_product(&self, other: &DenseMat) -> f64 {
        assert_eq!(self.cols, other.rows, "trace_product shape mismatch");
        assert_eq!(self.rows, other.cols, "trace_product shape mismatch");
        let mut total = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                total += self.get(i, j) * other.get(j, i);
            }
        }
        total
    }

    /// Maximum absolute entrywise difference (for tolerance-based assertions).
    pub fn max_abs_diff(&self, other: &DenseMat) -> f64 {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Select a subset of columns (preserving order), producing a `rows × |cols|`
    /// matrix.
    fn select_cols(&self, cols: &[usize]) -> DenseMat {
        let mut out = DenseMat::zeros(self.rows, cols.len());
        for (j_out, &j) in cols.iter().enumerate() {
            for r in 0..self.rows {
                out.set(r, j_out, self.get(r, j));
            }
        }
        out
    }

    /// Symmetrised copy `(A + Aᵀ)/2` (square matrices only).
    fn symmetrized(&self) -> DenseMat {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut out = DenseMat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                out.set(i, j, 0.5 * (self.get(i, j) + self.get(j, i)));
            }
        }
        out
    }
}

// ===========================================================================
// Deterministic PRNG (xorshift64*). Avoids any dependency and keeps tests
// reproducible.
// ===========================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the zero state.
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in (-1.0, 1.0).
    fn next_signed(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        2.0 * u - 1.0
    }
}

// ===========================================================================
// Commutant via signed union-find over matrix cells.
// ===========================================================================

/// One sign-consistent commutant orbit: a list of `(cell_index, sign)` pairs
/// where `cell_index = a*d + b` and the commutant basis matrix takes value
/// `sign` at `(a, b)` (and `0` off the orbit).
type Orbit = Vec<(usize, i8)>;

/// Signed disjoint-set: `parent[x]` and `rel[x]` (sign of `x` relative to its
/// parent). `value(x) = rel-product-to-root * value(root)`.
struct SignedDsu {
    parent: Vec<u32>,
    rel: Vec<i8>,
    consistent: Vec<bool>,
}

impl SignedDsu {
    fn new(n: usize) -> Self {
        SignedDsu {
            parent: (0..n as u32).collect(),
            rel: vec![1i8; n],
            consistent: vec![true; n],
        }
    }

    /// Returns `(root, sign)` with `value(x) = sign * value(root)`.
    fn find(&mut self, x: usize) -> (usize, i8) {
        let mut cur = x;
        let mut sign = 1i8;
        // Walk to the root accumulating signs.
        while self.parent[cur] as usize != cur {
            sign *= self.rel[cur];
            cur = self.parent[cur] as usize;
        }
        let root = cur;
        // Path compression with corrected signs.
        let mut node = x;
        let mut s = sign;
        while self.parent[node] as usize != node {
            let next = self.parent[node] as usize;
            let next_rel = self.rel[node];
            self.parent[node] = root as u32;
            self.rel[node] = s;
            s *= next_rel; // strip this edge's sign moving toward root
            // s now equals value(next)/value(root)
            let _ = next; // (kept for clarity)
            node = self.parent[node] as usize;
            if node == root {
                break;
            }
        }
        (root, sign)
    }

    /// Enforce `value(x) = r * value(y)`. Records inconsistency if it conflicts.
    fn union(&mut self, x: usize, y: usize, r: i8) {
        let (rx, sx) = self.find(x);
        let (ry, sy) = self.find(y);
        if rx == ry {
            // Need value(x)=r*value(y): sx*val(root) = r*sy*val(root)
            // => sx == r*sy. If not, the whole component is inconsistent.
            if sx != r * sy {
                self.consistent[rx] = false;
            }
            return;
        }
        // value(rx) = sx*value(x) = sx*r*value(y) = sx*r*sy*value(ry).
        let new_rel = sx * r * sy;
        let merged = self.consistent[rx] && self.consistent[ry];
        self.parent[rx] = ry as u32;
        self.rel[rx] = new_rel;
        self.consistent[ry] = merged;
    }
}

/// Compute the commutant of `rep` (matrices commuting with every `L_I`) as a
/// list of integer basis matrices, encoded as sign-consistent orbits over the
/// `d²` matrix cells. `dim(commutant) == orbits.len()`.
pub fn commutant_orbits(rep: &AdinkraRep) -> Vec<Orbit> {
    let d = rep.d;
    let cells = d * d;
    let mut dsu = SignedDsu::new(cells);

    for l in &rep.l_matrices {
        // value(a,b) = s[a]*s[b] * value(p[a], p[b])
        for a in 0..d {
            let pa = l.perm[a] as usize;
            let sa = l.sign[a];
            for b in 0..d {
                let pb = l.perm[b] as usize;
                let r = sa * l.sign[b];
                let cell = a * d + b;
                let cell2 = pa * d + pb;
                dsu.union(cell, cell2, r);
            }
        }
    }

    // Collect sign-consistent orbits. Use a BTreeMap keyed by root so the orbit
    // order is DETERMINISTIC (ascending root cell index); this is required for
    // the fixed-seed PRNG to make the whole decomposition reproducible across
    // runs and threads. Cells within each orbit are appended in ascending order.
    use std::collections::BTreeMap;
    let mut by_root: BTreeMap<usize, Orbit> = BTreeMap::new();
    for cell in 0..cells {
        let (root, sign) = dsu.find(cell);
        if dsu.consistent[root] {
            by_root.entry(root).or_default().push((cell, sign));
        }
    }
    by_root.into_values().collect()
}

/// Build a generic symmetric commutant element `H = Σ_t α_t (B_t + B_tᵀ)` with
/// random coefficients `α_t`.
fn random_symmetric_commutant(orbits: &[Orbit], d: usize, rng: &mut Rng) -> DenseMat {
    let mut h = DenseMat::zeros(d, d);
    for orbit in orbits {
        let alpha = rng.next_signed();
        for &(cell, sign) in orbit {
            let a = cell / d;
            let b = cell % d;
            let v = alpha * sign as f64;
            // (B_t + B_tᵀ): add to (a,b) and (b,a).
            h.data[a * d + b] += v;
            h.data[b * d + a] += v;
        }
    }
    h
}

// ===========================================================================
// Symmetric eigendecomposition (cyclic Jacobi).
// ===========================================================================

/// Cyclic Jacobi eigendecomposition of a real symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-
/// orthonormal and `A ≈ V · diag(eigenvalues) · Vᵀ`. Inputs must be square and
/// (numerically) symmetric.
pub fn jacobi_eigen(a: &DenseMat) -> (Vec<f64>, DenseMat) {
    assert_eq!(a.rows, a.cols, "jacobi_eigen requires a square matrix");
    let n = a.rows;
    let mut m = a.clone();
    let mut v = DenseMat::identity(n);
    if n == 0 {
        return (vec![], v);
    }

    for _sweep in 0..100 {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m.get(p, q) * m.get(p, q);
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                // Tangent of the rotation that zeros (p,q) (Numerical Recipes /
                // Golub & Van Loan form, picking the smaller root for stability).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply the Givens rotation symmetrically: m <- Gᵀ m G.
                // Columns p, q of every row.
                for i in 0..n {
                    let mip = m.get(i, p);
                    let miq = m.get(i, q);
                    m.set(i, p, c * mip - s * miq);
                    m.set(i, q, s * mip + c * miq);
                }
                // Rows p, q of every column.
                for j in 0..n {
                    let mpj = m.get(p, j);
                    let mqj = m.get(q, j);
                    m.set(p, j, c * mpj - s * mqj);
                    m.set(q, j, s * mpj + c * mqj);
                }
                // Force exact zero (and symmetry) on the eliminated entry.
                m.set(p, q, 0.0);
                m.set(q, p, 0.0);
                // Accumulate eigenvectors: V <- V G.
                for i in 0..n {
                    let vip = v.get(i, p);
                    let viq = v.get(i, q);
                    v.set(i, p, c * vip - s * viq);
                    v.set(i, q, s * vip + c * viq);
                }
            }
        }
    }

    let eigs: Vec<f64> = (0..n).map(|i| m.get(i, i)).collect();
    (eigs, v)
}

/// Group eigenvalue indices into clusters of (numerically) equal eigenvalues.
/// Returns one `Vec<usize>` of column indices per cluster, ordered by ascending
/// eigenvalue.
fn cluster_eigenvalues(eigs: &[f64]) -> Vec<Vec<usize>> {
    let n = eigs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| eigs[i].partial_cmp(&eigs[j]).unwrap());

    let spread = eigs.iter().cloned().fold(f64::MIN, f64::max)
        - eigs.iter().cloned().fold(f64::MAX, f64::min);
    let tol = 1e-7 * (1.0 + spread.abs());

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for &i in &idx {
        match clusters.last_mut() {
            Some(last) if (eigs[i] - eigs[*last.last().unwrap()]).abs() <= tol => {
                last.push(i);
            }
            _ => clusters.push(vec![i]),
        }
    }
    clusters
}

// ===========================================================================
// Decomposition
// ===========================================================================

/// One irreducible summand: an orthonormal d×dmin basis `B` of an L-invariant
/// subspace, plus the restricted dense operators `L_I|_W = Bᵀ L_I B`.
#[derive(Debug, Clone)]
pub struct IrrepSummand {
    pub n: usize,
    pub dmin: usize,
    /// Column-orthonormal d×dmin basis of the invariant subspace.
    pub basis: DenseMat,
    /// The N restricted dense L-operators, each dmin×dmin.
    pub l_restricted: Vec<DenseMat>,
}

/// Result of decomposing one (reducible or irreducible) valise rep.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub n: usize,
    pub d: usize,
    pub dmin: usize,
    /// Dimension of the real commutant (`= #sign-consistent orbits`).
    pub commutant_dim: usize,
    /// Exactly `r = d/dmin` irreducible summands.
    pub summands: Vec<IrrepSummand>,
}

/// Restrict a signed-permutation operator onto a subspace: `Bᵀ L B`.
///
/// `basis` is the d×dmin column-orthonormal `B`; `l` is the d×d signed perm.
/// The d×d operator is never densified: `L B` is formed by permuting rows of
/// `B` with signs (O(d·dmin)), then `Bᵀ (L B)` is a dense multiply.
pub fn restrict(l: &SignedPerm, basis: &DenseMat) -> DenseMat {
    let d = basis.rows;
    let m = basis.cols;
    debug_assert_eq!(l.dim(), d, "restrict: dimension mismatch");
    // L has row i nonzero at column perm[i] with value sign[i], so
    // (L B)[i][c] = sign[i] * B[perm[i]][c].
    let mut lb = DenseMat::zeros(d, m);
    for i in 0..d {
        let pi = l.perm[i] as usize;
        let s = l.sign[i] as f64;
        for c in 0..m {
            lb.set(i, c, s * basis.get(pi, c));
        }
    }
    basis.transpose().matmul(&lb)
}

/// Decompose a valise `rep` into its `r = d/dmin` irreducible GR(dmin, N)
/// summands.
///
/// Returns `None` only when `rep.d > MAX_DECOMPOSE_D` (skipped as infeasible for
/// the dense path). Otherwise returns exactly `r` summands (`r = 1` when the rep
/// is already irreducible, `d == dmin`).
///
/// Panics if a genuinely reducible block fails to split after
/// [`MAX_SPLIT_ATTEMPTS`] (a correctness failure, not a skip) or if the final
/// summand count does not equal `r` — both are loud guards against a silently
/// wrong decomposition.
pub fn decompose_rep(rep: &AdinkraRep) -> Option<Decomposition> {
    let d = rep.d;
    let n = rep.n;
    let dm = dmin(n);

    if d > MAX_DECOMPOSE_D {
        return None;
    }
    assert!(d % dm == 0, "d={d} not a multiple of dmin={dm}");
    let r = d / dm;

    // Commutant (cheap and exact even for r=1).
    let orbits = commutant_orbits(rep);
    let commutant_dim = orbits.len();

    // Trivial case: already irreducible.
    if r == 1 {
        let basis = DenseMat::identity(d);
        let l_restricted = rep
            .l_matrices
            .iter()
            .map(DenseMat::from_signed_perm)
            .collect();
        return Some(Decomposition {
            n,
            d,
            dmin: dm,
            commutant_dim,
            summands: vec![IrrepSummand { n, dmin: dm, basis, l_restricted }],
        });
    }

    // Iterative refinement: split blocks until each has dimension dmin.
    let mut rng = Rng::new(RNG_SEED);
    let mut work: Vec<DenseMat> = vec![DenseMat::identity(d)]; // each is a d×m basis
    let mut final_bases: Vec<DenseMat> = Vec::with_capacity(r);

    while let Some(b) = work.pop() {
        let mblk = b.cols;
        if mblk == dm {
            final_bases.push(b);
            continue;
        }
        debug_assert!(mblk > dm, "block smaller than dmin: {mblk} < {dm}");

        let mut split: Option<Vec<DenseMat>> = None;
        for _ in 0..MAX_SPLIT_ATTEMPTS {
            let h = random_symmetric_commutant(&orbits, d, &mut rng);
            // restricted = Bᵀ H B  (m×m), symmetrised against numeric drift.
            let hb = h.matmul(&b);
            let restricted = b.transpose().matmul(&hb).symmetrized();
            let (eigs, evecs) = jacobi_eigen(&restricted);
            let clusters = cluster_eigenvalues(&eigs);
            // A genuine commutant element's eigenspaces are L-invariant, so each
            // cluster dimension MUST be a multiple of dmin. Reject any split that
            // violates this (a spurious numerical split from too-tight clustering)
            // and retry with a fresh generic element rather than emit an
            // undersized, non-invariant block.
            let sizes_valid = clusters.iter().all(|cl| cl.len() % dm == 0);
            if clusters.len() > 1 && sizes_valid {
                let subs = clusters
                    .iter()
                    .map(|cl| {
                        let u = evecs.select_cols(cl); // m × |cl|
                        b.matmul(&u) // d × |cl|
                    })
                    .collect();
                split = Some(subs);
                break;
            }
        }

        match split {
            Some(subs) => work.extend(subs),
            None => panic!(
                "decompose_rep: failed to split a reducible block of size {mblk} \
                 (d={d}, dmin={dm}) after {MAX_SPLIT_ATTEMPTS} attempts"
            ),
        }
    }

    assert_eq!(
        final_bases.len(),
        r,
        "decompose_rep: got {} summands, expected r=d/dmin={}",
        final_bases.len(),
        r
    );

    let summands = final_bases
        .into_iter()
        .map(|basis| {
            assert_eq!(basis.cols, dm);
            let l_restricted = rep.l_matrices.iter().map(|l| restrict(l, &basis)).collect();
            IrrepSummand { n, dmin: dm, basis, l_restricted }
        })
        .collect();

    Some(Decomposition { n, d, dmin: dm, commutant_dim, summands })
}

// ===========================================================================
// Dense holoraumy and gadget on irreducible summands.
// ===========================================================================

/// Maps a pair `(i, j)` with `i > j` to a linear index in `0..C(N,2)`.
/// Identical convention to [`crate::holoraumy`].
fn pair_index(i: usize, j: usize) -> usize {
    i * (i - 1) / 2 + j
}

/// Dense holoraumy data for one irreducible summand (mirrors
/// [`crate::holoraumy::HoloraumyData`] but over dense restricted operators).
#[derive(Debug, Clone)]
pub struct DenseHoloraumy {
    pub n: usize,
    pub d: usize, // == dmin for a summand
    /// Fermionic holoraumy, the dense densification of the signed-path
    /// `L_I.inverse().compose(L_J)`. Because the project's `SignedPerm::compose`
    /// is a right action (`X.compose(Y)` is the matrix product `Y·X`), this is
    /// the standard product `L_J · L_Iᵀ`, pair-indexed I>J.
    pub vtilde: Vec<DenseMat>,
}

impl DenseHoloraumy {
    /// Build from one irreducible summand's restricted dense L-operators, using
    /// the same I>J pair ordering as [`crate::holoraumy::HoloraumyData::from_rep`].
    pub fn from_summand(summand: &IrrepSummand) -> Self {
        let n = summand.n;
        let d = summand.dmin;
        let lt: Vec<DenseMat> = summand.l_restricted.iter().map(|l| l.transpose()).collect();
        let num_pairs = n * (n - 1) / 2;
        let mut vtilde = Vec::with_capacity(num_pairs);
        for i in 1..n {
            for j in 0..i {
                // Vtilde_IJ = L_J · L_Iᵀ  (matches signed L_I^{-1}.compose(L_J)).
                vtilde.push(summand.l_restricted[j].matmul(&lt[i]));
            }
        }
        DenseHoloraumy { n, d, vtilde }
    }
}

/// Dense analogue of [`crate::holoraumy::gadget`]. Identical normalisation
/// `-2 / (N(N-1)·dmin(N)) · Σ_{I>J} Tr(Vtilde^a_IJ · Vtilde^b_IJ)`.
pub fn dense_gadget(a: &DenseHoloraumy, b: &DenseHoloraumy) -> f64 {
    assert_eq!(a.n, b.n);
    assert_eq!(a.d, b.d);
    let n = a.n;
    let mut sum = 0.0;
    for idx in 0..a.vtilde.len() {
        sum += a.vtilde[idx].trace_product(&b.vtilde[idx]);
    }
    let dmin_val = dmin(n);
    -2.0 * sum / (n * (n - 1) * dmin_val) as f64
}

/// Symmetric gadget matrix over a collection of dense holoraumy data.
///
/// Parallelised over rows (mirrors the signed-path `gadget_stratum_matrix`).
pub fn dense_gadget_matrix(reps: &[DenseHoloraumy]) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let n = reps.len();
    (0..n)
        .into_par_iter()
        .map(|i| (0..n).map(|j| dense_gadget(&reps[i], &reps[j])).collect())
        .collect()
}

/// Verify the restricted operators of a summand satisfy the Garden algebra
/// `L_I L_Jᵀ + L_J L_Iᵀ = 2δ_IJ I` and `V_IJ² = -I`, to tolerance `tol`.
/// Returns the maximum residual found (0.0 == perfect).
pub fn summand_residual(summand: &IrrepSummand) -> f64 {
    let n = summand.n;
    let dm = summand.dmin;
    let l = &summand.l_restricted;
    let lt: Vec<DenseMat> = l.iter().map(|m| m.transpose()).collect();
    let id = DenseMat::identity(dm);
    let mut worst = 0.0f64;

    // Garden algebra.
    for i in 0..n {
        for j in 0..n {
            let mut s = l[i].matmul(&lt[j]); // L_I L_Jᵀ
            let lj_lit = l[j].matmul(&lt[i]); // L_J L_Iᵀ
            for k in 0..s.data.len() {
                s.data[k] += lj_lit.data[k];
            }
            let target = if i == j {
                // 2 I
                let mut t = id.clone();
                for v in t.data.iter_mut() {
                    *v *= 2.0;
                }
                t
            } else {
                DenseMat::zeros(dm, dm)
            };
            worst = worst.max(s.max_abs_diff(&target));
        }
    }

    // V_IJ² = -I for I>J.
    let neg_id = {
        let mut t = id.clone();
        for v in t.data.iter_mut() {
            *v = -*v;
        }
        t
    };
    for i in 1..n {
        for j in 0..i {
            let v = l[i].matmul(&lt[j]); // V_IJ = L_I L_Jᵀ
            let v2 = v.matmul(&v);
            worst = worst.max(v2.max_abs_diff(&neg_id));
        }
    }
    worst
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holoraumy::{gadget, HoloraumyData};

    // -- N=4 fixtures (identical to holoraumy.rs / lr_matrix.rs) ---------------

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

    /// Block-diagonal direct sum of two N=4 reps into an 8-dim rep.
    fn block_diag_n4(a: &AdinkraRep, b: &AdinkraRep) -> AdinkraRep {
        assert_eq!(a.n, b.n);
        let n = a.n;
        let da = a.d;
        let db = b.d;
        let d = da + db;
        let mut l_matrices = Vec::with_capacity(n);
        for i in 0..n {
            let la = &a.l_matrices[i];
            let lb = &b.l_matrices[i];
            let mut perm = vec![0u16; d];
            let mut sign = vec![0i8; d];
            for r in 0..da {
                perm[r] = la.perm[r];
                sign[r] = la.sign[r];
            }
            for r in 0..db {
                perm[da + r] = (da as u16) + lb.perm[r];
                sign[da + r] = lb.sign[r];
            }
            l_matrices.push(SignedPerm::from_parts(perm, sign).unwrap());
        }
        AdinkraRep { n, d, l_matrices }
    }

    // -- DenseMat basics -------------------------------------------------------

    #[test]
    fn dense_identity_trace() {
        for d in 0..=5 {
            assert_eq!(DenseMat::identity(d).trace(), d as f64);
        }
    }

    #[test]
    fn dense_from_signed_perm_trace_matches() {
        for rep in [cs_n4(), vs_n4()] {
            for l in &rep.l_matrices {
                let dm = DenseMat::from_signed_perm(l);
                assert_eq!(dm.trace(), l.trace() as f64);
            }
        }
    }

    #[test]
    fn dense_matmul_matches_signed_compose() {
        // SignedPerm::compose is a right action: `X.compose(Y)` is the matrix
        // product `Y·X`. So the standard product `A·B` (matmul) equals the
        // densification of `b.compose(a)`.
        let rep = cs_n4();
        for a in &rep.l_matrices {
            for b in &rep.l_matrices {
                let dense = DenseMat::from_signed_perm(a).matmul(&DenseMat::from_signed_perm(b));
                let signed = DenseMat::from_signed_perm(&b.compose(a));
                assert!(dense.max_abs_diff(&signed) < 1e-12);
            }
        }
    }

    #[test]
    fn dense_trace_product_matches_signed() {
        let rep = cs_n4();
        for a in &rep.l_matrices {
            for b in &rep.l_matrices {
                let da = DenseMat::from_signed_perm(a);
                let db = DenseMat::from_signed_perm(b);
                assert_eq!(da.trace_product(&db), a.trace_product(b) as f64);
            }
        }
    }

    #[test]
    fn jacobi_eigen_small() {
        // Symmetric 2x2 [[2,1],[1,2]] -> eigenvalues 1 and 3.
        let mut a = DenseMat::zeros(2, 2);
        a.set(0, 0, 2.0);
        a.set(1, 1, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        let (mut eigs, v) = jacobi_eigen(&a);
        eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 3.0).abs() < 1e-10);
        // V orthonormal: VᵀV = I.
        let vtv = v.transpose().matmul(&v);
        assert!(vtv.max_abs_diff(&DenseMat::identity(2)) < 1e-10);
    }

    // -- Dense gadget matches the trusted signed-perm path (normalisation) -----

    #[test]
    fn dense_gadget_matches_signed_gadget_n4() {
        for rep in [cs_n4(), vs_n4()] {
            let decomp = decompose_rep(&rep).unwrap();
            assert_eq!(decomp.summands.len(), 1, "N=4 rep is already irreducible");
            let dh = DenseHoloraumy::from_summand(&decomp.summands[0]);
            let signed = gadget(&HoloraumyData::from_rep(&rep), &HoloraumyData::from_rep(&rep));
            assert!(
                (dense_gadget(&dh, &dh) - signed).abs() < 1e-9,
                "dense self-gadget {} != signed {}",
                dense_gadget(&dh, &dh),
                signed
            );
            assert!((dense_gadget(&dh, &dh) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn dense_gadget_cs_vs_zero() {
        let cs = DenseHoloraumy::from_summand(&decompose_rep(&cs_n4()).unwrap().summands[0]);
        let vs = DenseHoloraumy::from_summand(&decompose_rep(&vs_n4()).unwrap().summands[0]);
        assert!(dense_gadget(&cs, &vs).abs() < 1e-9, "G[CS,VS] should be 0");
    }

    // -- Decomposition of an explicit reducible rep: CS ⊕ CS (d=8) -------------

    #[test]
    fn decompose_cs_plus_cs() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4());
        assert!(rep.verify_garden_algebra(), "CS⊕CS fixture must be a valid rep");

        // Pre-decomposition: full self-gadget = d/dmin = 8/4 = 2 (signed path).
        let full = HoloraumyData::from_rep(&rep);
        assert!((gadget(&full, &full) - 2.0).abs() < 1e-9, "full self-gadget should be 2");

        let decomp = decompose_rep(&rep).unwrap();
        assert_eq!(decomp.summands.len(), 2, "r = d/dmin = 8/4 = 2");

        // Each summand: valid algebra + self-gadget 1.0 (basis-independent).
        let mut self_sum = 0.0;
        for s in &decomp.summands {
            assert!(summand_residual(s) < 1e-7, "summand algebra residual too large");
            let dh = DenseHoloraumy::from_summand(s);
            let g = dense_gadget(&dh, &dh);
            assert!((g - 1.0).abs() < 1e-7, "summand self-gadget {} != 1", g);
            self_sum += g;
        }
        // Reconstruction: Σ self-gadgets = d/dmin = 2.
        assert!((self_sum - 2.0).abs() < 1e-7, "self-gadget sum {} != 2", self_sum);
    }

    // -- CS ⊕ VS (d=8): inequivalent summands separate -------------------------

    #[test]
    fn decompose_cs_plus_vs() {
        let rep = block_diag_n4(&cs_n4(), &vs_n4());
        assert!(rep.verify_garden_algebra());

        let full = HoloraumyData::from_rep(&rep);
        assert!((gadget(&full, &full) - 2.0).abs() < 1e-9);

        let decomp = decompose_rep(&rep).unwrap();
        assert_eq!(decomp.summands.len(), 2);

        // Robust, basis-independent checks: each summand is a valid irreducible
        // with self-gadget 1.0, and they reconstruct the full d/dmin.
        let mut self_sum = 0.0;
        for s in &decomp.summands {
            assert!(summand_residual(s) < 1e-7);
            let dh = DenseHoloraumy::from_summand(s);
            let g = dense_gadget(&dh, &dh);
            assert!((g - 1.0).abs() < 1e-7);
            self_sum += g;
        }
        assert!((self_sum - 2.0).abs() < 1e-7);

        // The genuine separation of INEQUIVALENT irreducibles (CS vs VS) is a
        // basis-independent fact visible in the commutant dimension: CS⊕VS has
        // commutant H⊕H (distinct blocks) which is strictly smaller than the
        // M_2(H) of CS⊕CS (same irrep twice). See commutant_dim_isotypic_vs_distinct.
        let cc = decompose_rep(&block_diag_n4(&cs_n4(), &cs_n4())).unwrap().commutant_dim;
        assert!(
            decomp.commutant_dim < cc,
            "CS⊕VS commutant {} should be smaller than CS⊕CS {}",
            decomp.commutant_dim,
            cc
        );
    }

    // -- Commutant dimension: same-irrep multiplicity grows it ----------------

    #[test]
    fn commutant_dim_isotypic_vs_distinct() {
        // CS⊕CS: 2 copies of the SAME irrep -> larger commutant (M_2(D)).
        // CS⊕VS: 2 DISTINCT irreps -> block-diagonal commutant (D ⊕ D).
        // Convention-independent: dim(CS⊕CS) == 2 * dim(CS⊕VS).
        let cc = decompose_rep(&block_diag_n4(&cs_n4(), &cs_n4())).unwrap().commutant_dim;
        let cv = decompose_rep(&block_diag_n4(&cs_n4(), &vs_n4())).unwrap().commutant_dim;
        assert_eq!(cc, 2 * cv, "dim(CS⊕CS)={cc} should be 2*dim(CS⊕VS)={cv}");
        // Single irrep baseline (quaternionic at N=4 => End = H, real dim 4).
        let single = decompose_rep(&cs_n4()).unwrap().commutant_dim;
        assert_eq!(single, 4, "single N=4 irrep commutant should be 4 (quaternionic H)");
        assert_eq!(cv, 8, "CS⊕VS commutant should be H⊕H = 8");
        assert_eq!(cc, 16, "CS⊕CS commutant should be M_2(H) = 16");
    }

    // -- Basis invariance of the self-gadget -----------------------------------

    #[test]
    fn self_gadget_basis_invariant_under_block_swap() {
        // Build CS⊕CS, decompose, and confirm every summand self-gadget is 1.0
        // regardless of which invariant subspace the refinement landed on.
        let decomp = decompose_rep(&block_diag_n4(&cs_n4(), &cs_n4())).unwrap();
        for s in &decomp.summands {
            let dh = DenseHoloraumy::from_summand(s);
            assert!((dense_gadget(&dh, &dh) - 1.0).abs() < 1e-7);
        }
    }

    // -- Scale guard -----------------------------------------------------------

    #[test]
    fn scale_guard_skips_large_d() {
        // A cheap synthetic rep with d just over the threshold: we only need the
        // guard to fire before any dense d×d work, so use identity L-matrices.
        let d = MAX_DECOMPOSE_D + 2;
        let l_matrices: Vec<SignedPerm> = (0..4).map(|_| SignedPerm::identity(d)).collect();
        let rep = AdinkraRep { n: 4, d, l_matrices };
        assert!(decompose_rep(&rep).is_none(), "decomposition must be skipped for large d");
    }

    #[test]
    fn decomposition_is_deterministic() {
        // Two independent decompositions of the same rep must produce bit-for-bit
        // identical summand bases (guards against nondeterministic orbit order
        // feeding the PRNG). This also makes the cross-summand gadget matrix
        // reproducible across runs.
        let rep = block_diag_n4(&cs_n4(), &vs_n4());
        let a = decompose_rep(&rep).unwrap();
        let b = decompose_rep(&rep).unwrap();
        assert_eq!(a.summands.len(), b.summands.len());
        for (sa, sb) in a.summands.iter().zip(b.summands.iter()) {
            assert_eq!(sa.basis.max_abs_diff(&sb.basis), 0.0, "bases differ across runs");
        }
        // And the full irreducible gadget matrix is identical.
        let ma = dense_gadget_matrix(
            &a.summands.iter().map(DenseHoloraumy::from_summand).collect::<Vec<_>>(),
        );
        let mb = dense_gadget_matrix(
            &b.summands.iter().map(DenseHoloraumy::from_summand).collect::<Vec<_>>(),
        );
        assert_eq!(ma, mb, "gadget matrix not reproducible");
    }

    #[test]
    fn restrict_identity_basis_is_densify() {
        let rep = cs_n4();
        for l in &rep.l_matrices {
            let r = restrict(l, &DenseMat::identity(4));
            assert!(r.max_abs_diff(&DenseMat::from_signed_perm(l)) < 1e-12);
        }
    }
}
