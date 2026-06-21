//! SO(1,9) Lorentz covariant-assembly machinery for the dimensional-lifting
//! pipeline.
//!
//! References (see REFERENCES.md):
//!   - Cigliano, Dahl, Gates et al., "10D Supergravity Numerical Data Sets for L &
//!     R Matrices", arXiv:2512.12157 (+ github.com/mcmulaz/Super-Sym) — the split
//!     SO(1,9) sigma relation σ^μ σ̃^ν + σ^ν σ̃^μ = 2 η^μν I_16 and the worldline
//!     L/R setup.
//!   - Octonionic SO(9) Clifford construction (Fano-plane imaginary-octonion
//!     left-multiplications) and recursive even-dim Clifford construction:
//!     standard; cf. arXiv:2205.09509 and Brink-Schwarz-Scherk 10D SYM.
//!
//! # Background
//!
//! 10D, N=1 super Yang-Mills has a single 16-component Majorana-Weyl spinor of
//! Spin(1,9). The supercovariant derivative algebra is carried by a pair of
//! 16x16 "sigma" matrices `σ^μ` and `σ̃^μ` (μ = 0..9) satisfying the split
//! Clifford / Dirac relation
//!
//! ```text
//!   σ^μ σ̃^ν + σ^ν σ̃^μ = 2 η^μν I_16 ,   η = diag(-1, +1, +1, …, +1)
//! ```
//!
//! (the mostly-plus signature used for the worldline lift; see the search note
//! at the bottom for sources). These are exactly the off-diagonal blocks of the
//! 32x32 real gamma matrices
//!
//! ```text
//!   Γ^μ = [[ 0 , σ^μ ],[ σ̃^μ , 0 ]] ,   {Γ^μ, Γ^ν} = 2 η^μν I_32 .
//! ```
//!
//! ## Explicit construction (the load-bearing part)
//!
//! We do **not** trust a closed formula. We build nine real *symmetric*
//! mutually-anticommuting 16x16 matrices `γ^1..γ^9` (a Euclidean SO(9) Clifford
//! algebra, `{γ^a, γ^b} = 2 δ^ab I_16`) by an explicit recursive Pauli
//! tensor-product construction, then assemble the SO(1,9) sigmas as
//!
//! ```text
//!   σ^0 = +I_16 ,  σ̃^0 = −I_16        (the η^00 = −1 timelike direction)
//!   σ^a = γ^a ,    σ̃^a = γ^a   (a=1..9)
//! ```
//!
//! and **prove** the split-Clifford relation numerically with
//! [`Clifford10D::verify_clifford`]. One checks directly:
//!
//! ```text
//!   σ^0 σ̃^0 + σ^0 σ̃^0 = 2(I·(−I)) = −2I = 2 η^00 I        ✓
//!   σ^0 σ̃^a + σ^a σ̃^0 = γ^a − γ^a = 0 = 2 η^0a I            ✓
//!   σ^a σ̃^b + σ^b σ̃^a = γ^a γ^b + γ^b γ^a = 2δ^ab I        ✓
//! ```
//!
//! The recursive Pauli construction of the nine SO(9) gammas uses the real
//! 2x2 building blocks
//!
//! ```text
//!   I = [[1,0],[0,1]]   X = [[0,1],[1,0]]   Z = [[1,0],[0,-1]]   E = [[0,1],[-1,0]]
//! ```
//!
//! (X, Z symmetric; E antisymmetric; I·… and the chirality element built from
//! Z's). Four tensor factors give 2^4 = 16. We list ten candidate symmetric,
//! mutually anticommuting generators (a full SO(10)/Spin(10) real Clifford set
//! is available in 16 dimensions; we take the first nine for SO(9)). The exact
//! tensor words are chosen so that all are symmetric and pairwise anticommuting,
//! which `verify_clifford` confirms to < 1e-9.
//!
//! # Covariant assembly
//!
//! A worldline N=16 representation gives sixteen `L`-matrices (and `R = Lᵀ`).
//! Their temporal linkage is `Δ^0_I = L_I`. The Lorentz lift demands the spatial
//! linkages be fixed by the sigmas:
//!
//! ```text
//!   Δ^a_I = −(σ^0 σ̃^a)_I^{\,J} Δ^0_J ,   a = 1..9 ,
//! ```
//! i.e. the spatial linkage is obtained by acting with the SO(9) gamma `σ^0 σ̃^a`
//! on the *spinor index* `I` that labels the supercharges. The representation
//! lifts to a genuine off-shell 10D object iff the assembled linkages still close
//! the (bosonic and fermionic) Clifford/Garden relations.
//!
//! VALIDATED vs EXPERIMENTAL. The 16×16 SO(1,9) sigma matrices and the split
//! Clifford relation `σ^μ σ̃^ν + σ^ν σ̃^μ = 2 η^μν I` are explicitly constructed
//! and numerically verified (`verify_clifford` == 0). The `assemble_and_check`
//! non-closure metric `e_norm`, however, is EXPERIMENTAL and NOT yet a calibrated
//! off-shell diagnostic: adversarial review showed the current residual target is
//! mis-normalized (an exactly closed input does not map to `e_norm == 0`), so DO
//! NOT interpret `e_norm` as an off-shell/on-shell certificate. It is a raw
//! assembled-residual probe pending re-derivation against a known positive and
//! negative fixture (e.g. the on-shell 10D dataset in `crate::tendim_data`).
//!
//! This module is deliberately self-contained: it carries its own minimal dense
//! matrix [`Mat`] (mirroring the API of [`crate::decompose::DenseMat`]) so it has
//! no dependence on the decomposition internals.

// ===========================================================================
// Minimal dense real matrix (row-major). Mirrors decompose::DenseMat's API so
// the rest of the crate's conventions carry over, but is kept local for
// isolation.
// ===========================================================================

/// Row-major dense real matrix: `data[r*cols + c]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// The `d`x`d` identity.
    pub fn identity(d: usize) -> Self {
        let mut m = Mat::zeros(d, d);
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

    /// Matrix product `self * other`.
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = Mat::zeros(m, n);
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
    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Trace (square matrices only).
    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "trace of non-square matrix");
        (0..self.rows).map(|i| self.get(i, i)).sum()
    }

    /// Entrywise `self + other`.
    pub fn add(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        for k in 0..out.data.len() {
            out.data[k] += other.data[k];
        }
        out
    }

    /// `self` scaled by `s`.
    pub fn scale(&self, s: f64) -> Mat {
        let mut out = self.clone();
        for v in out.data.iter_mut() {
            *v *= s;
        }
        out
    }

    /// Kronecker (tensor) product `self ⊗ other`.
    pub fn kron(&self, other: &Mat) -> Mat {
        let (ar, ac) = (self.rows, self.cols);
        let (br, bc) = (other.rows, other.cols);
        let mut out = Mat::zeros(ar * br, ac * bc);
        for i in 0..ar {
            for j in 0..ac {
                let a = self.get(i, j);
                if a == 0.0 {
                    continue;
                }
                for p in 0..br {
                    for q in 0..bc {
                        out.set(i * br + p, j * bc + q, a * other.get(p, q));
                    }
                }
            }
        }
        out
    }

    /// Maximum absolute entrywise value.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0, f64::max)
    }

    /// Maximum absolute entrywise difference `max|self - other|`.
    pub fn max_abs_diff(&self, other: &Mat) -> f64 {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Frobenius norm `sqrt(Σ a_ij²)`.
    pub fn frobenius(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

// 2x2 real building blocks for the recursive Pauli tensor construction.
fn pauli_i() -> Mat {
    Mat::identity(2)
}
fn pauli_x() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 1, 1.0);
    m.set(1, 0, 1.0);
    m
}
fn pauli_z() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 0, 1.0);
    m.set(1, 1, -1.0);
    m
}
/// Real antisymmetric block `E = [[0,1],[-1,0]]` (= i·σ_y, kept real).
fn pauli_e() -> Mat {
    let mut m = Mat::zeros(2, 2);
    m.set(0, 1, 1.0);
    m.set(1, 0, -1.0);
    m
}

/// Tensor four 2x2 blocks into one 16x16 matrix `a ⊗ b ⊗ c ⊗ d`.
fn kron4(a: &Mat, b: &Mat, c: &Mat, d: &Mat) -> Mat {
    a.kron(b).kron(c).kron(d)
}

/// The seven imaginary-octonion left-multiplication matrices `L_{e_1}..L_{e_7}`
/// (8x8, real, **antisymmetric**, mutually anticommuting, each squaring to `−I`).
///
/// Octonion multiplication is fixed by the Fano-plane triples
/// `(1,2,3),(1,4,5),(1,7,6),(2,4,6),(2,5,7),(3,4,7),(3,6,5)` (Cayley basis,
/// `e_i e_j = e_k` for a cyclic triple, `= −e_k` reversed, `e_i² = −1`). The
/// left-multiplication operator `L_{e_i}` has, in column `j` (basis vector
/// `e_j`, with `e_0 = 1`), the result of `e_i · e_j`. The resulting 8x8 matrices
/// are real and antisymmetric — the canonical construction of the seven complex
/// structures realising Cl(0,7) on R^8.
fn octonion_left_mult() -> Vec<Mat> {
    // mult[i][j] = (sign, k) for e_i * e_j = sign * e_k, indices 0..=7 with
    // e_0 = 1 the unit. Built from the Fano triples + unit + e_i² = −1.
    let triples: [(usize, usize, usize); 7] = [
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    ];
    // table[i][j] = (sign, k).
    let mut table = [[(0i32, 0usize); 8]; 8];
    for i in 0..8 {
        // e_0 * e_j = e_j ; e_i * e_0 = e_i.
        table[0][i] = (1, i);
        table[i][0] = (1, i);
    }
    for i in 1..8 {
        table[i][i] = (-1, 0); // e_i^2 = -1
    }
    for &(a, b, c) in &triples {
        // a*b=c, b*c=a, c*a=b and the reversed products negate.
        for &(x, y, z) in &[(a, b, c), (b, c, a), (c, a, b)] {
            table[x][y] = (1, z);
            table[y][x] = (-1, z);
        }
    }

    // L_{e_i} column j = e_i * e_j  => entry (k, j) = sign where e_i*e_j = sign e_k.
    let mut ls = Vec::with_capacity(7);
    for i in 1..8 {
        let mut m = Mat::zeros(8, 8);
        for j in 0..8 {
            let (sign, k) = table[i][j];
            m.set(k, j, sign as f64);
        }
        ls.push(m);
    }
    ls
}

/// Build nine real, **symmetric**, mutually-anticommuting 16x16 matrices forming
/// a Euclidean SO(9) Clifford algebra `{γ^a, γ^b} = 2 δ^ab I_16` (so each
/// `γ^a (γ^b)ᵀ + γ^b (γ^a)ᵀ = 2 δ^ab I` as well, the form needed by the SO(1,9)
/// split sigmas).
///
/// Construction (octonionic SO(9), the canonical 16-dim real symmetric Clifford
/// module): split `R^16 = R^8 ⊕ R^8`. Take the eight `c_0 = I_8` and the seven
/// antisymmetric octonion left-multiplications `c_a = L_{e_a}` (a=1..7), and set
///
/// ```text
///   γ^a = [[ 0 , c_a ],[ c_aᵀ , 0 ]]   (a = 1..8) ,
///   γ^9 = [[ I_8 , 0 ],[ 0 , −I_8 ]] .
/// ```
///
/// Each `γ^a (a≤8)` is symmetric by the block-transpose, `(γ^a)ᵀ = γ^a`; γ^9 is
/// diagonal hence symmetric. They mutually anticommute and square to `+I` because
/// the `c_a` satisfy `c_aᵀ c_b + c_bᵀ c_a = 2δ_ab I_8` (Hurwitz). Validated by
/// the unit tests and gated by [`Clifford10D::verify_clifford`].
fn so9_gammas() -> Vec<Mat> {
    let mut c: Vec<Mat> = Vec::with_capacity(8);
    c.push(Mat::identity(8)); // c_0 = I_8
    c.extend(octonion_left_mult()); // c_1..c_7

    let mut gammas: Vec<Mat> = Vec::with_capacity(9);
    // γ^a = [[0, c_a],[c_aᵀ, 0]] for a = 1..=8 (i.e. c index 0..=7).
    for ca in &c {
        let cat = ca.transpose();
        let mut g = Mat::zeros(16, 16);
        // top-right block = c_a
        for r in 0..8 {
            for col in 0..8 {
                g.set(r, 8 + col, ca.get(r, col));
            }
        }
        // bottom-left block = c_aᵀ
        for r in 0..8 {
            for col in 0..8 {
                g.set(8 + r, col, cat.get(r, col));
            }
        }
        gammas.push(g);
    }
    // γ^9 = diag(I_8, −I_8).
    let mut g9 = Mat::zeros(16, 16);
    for d in 0..8 {
        g9.set(d, d, 1.0);
        g9.set(8 + d, 8 + d, -1.0);
    }
    gammas.push(g9);

    assert_eq!(gammas.len(), 9);
    gammas
}

// ===========================================================================
// Clifford10D
// ===========================================================================

/// 10D Clifford data: `σ^μ` and `σ̃^μ` (16x16), μ = 0..9, in the mostly-plus
/// signature `η = diag(-1, +1^9)`.
pub struct Clifford10D {
    /// `σ^μ`, indexed by μ = 0..9. Each is 16x16.
    pub sigma: Vec<Mat>,
    /// `σ̃^μ` (the conjugate set), indexed by μ = 0..9. Each is 16x16.
    pub sigma_tilde: Vec<Mat>,
    /// The Minkowski metric diagonal `η_μμ` (length 10), `(-1, +1, …, +1)`.
    pub eta: [f64; 10],
}

impl Clifford10D {
    /// The spinor dimension (16).
    pub const DIM: usize = 16;

    /// Explicit construction of the SO(1,9) sigma matrices.
    pub fn build() -> Self {
        let dim = Self::DIM;
        let gammas = so9_gammas();
        assert_eq!(gammas.len(), 9, "need exactly nine SO(9) gammas");
        for g in &gammas {
            assert_eq!((g.rows, g.cols), (dim, dim));
        }

        let mut sigma = Vec::with_capacity(10);
        let mut sigma_tilde = Vec::with_capacity(10);

        // Timelike direction μ = 0 (η^00 = −1): σ^0 = +I, σ̃^0 = −I.
        sigma.push(Mat::identity(dim));
        sigma_tilde.push(Mat::identity(dim).scale(-1.0));

        // Spacelike directions a = 1..9: σ^a = σ̃^a = γ^a (real symmetric).
        for g in gammas {
            sigma.push(g.clone());
            sigma_tilde.push(g);
        }

        let mut eta = [1.0f64; 10];
        eta[0] = -1.0;

        Clifford10D { sigma, sigma_tilde, eta }
    }

    /// Maximum residual of the defining split-Clifford relation
    /// `σ^μ σ̃^ν + σ^ν σ̃^μ − 2 η^μν I_16`, maximised over all (μ, ν).
    ///
    /// This is the load-bearing self-test: a value < 1e-9 proves the explicit
    /// construction satisfies the SO(1,9) Clifford algebra.
    pub fn verify_clifford(&self) -> f64 {
        let dim = Self::DIM;
        let id = Mat::identity(dim);
        let mut worst = 0.0f64;
        for mu in 0..10 {
            for nu in 0..10 {
                // σ^μ σ̃^ν + σ^ν σ̃^μ
                let lhs = self.sigma[mu]
                    .matmul(&self.sigma_tilde[nu])
                    .add(&self.sigma[nu].matmul(&self.sigma_tilde[mu]));
                // 2 η^μν I  (η is diagonal, so η^μν = η_μμ when μ==ν else 0)
                let target = if mu == nu {
                    id.scale(2.0 * self.eta[mu])
                } else {
                    Mat::zeros(dim, dim)
                };
                worst = worst.max(lhs.max_abs_diff(&target));
            }
        }
        worst
    }

    /// The SO(9) gamma acting on the spinor index for spatial direction `a`
    /// (1..=9): `σ^0 σ̃^a`. Used to fix the spatial linkages from the temporal
    /// ones.
    pub fn spatial_generator(&self, a: usize) -> Mat {
        assert!((1..=9).contains(&a), "spatial index a must be in 1..=9");
        self.sigma[0].matmul(&self.sigma_tilde[a])
    }
}

// ===========================================================================
// Covariant assembly + non-closure report
// ===========================================================================

/// Result of an SO(1,9) covariant-assembly check on a worldline N=16 rep.
#[derive(Debug, Clone, PartialEq)]
pub struct NonClosureReport {
    /// Max residual of the *bosonic* closure `L_I R_J + L_J R_I − 2 δ_IJ I`
    /// over the assembled linkages (the Garden relation on the supercharges).
    pub max_residual_bosonic: f64,
    /// Max residual of the *fermionic* closure `R_I L_J + R_J L_I − 2 δ_IJ I`.
    pub max_residual_fermionic: f64,
    /// EXPERIMENTAL raw Frobenius norm of the assembled residual tensor. NOT a
    /// calibrated off-shell/on-shell certificate (the target normalization is not
    /// yet validated against a known fixture — do not read 0 as "off-shell").
    pub e_norm: f64,
}

/// Given the sixteen worldline `L`-matrices (each 16x16 dense), build the
/// SO(1,9) spatial linkages via the sigma machinery and test whether the
/// supercharges reassemble into a Lorentz-covariant 10D object.
///
/// Steps:
/// 1. Temporal linkage `Δ^0_I = L_I`, with `R_I = L_Iᵀ`.
/// 2. Spatial linkages `Δ^a_I = −(σ^0 σ̃^a)_I^{\,J} Δ^0_J` for a = 1..9, where
///    the spinor-index contraction `(σ^0 σ̃^a)_I^J Δ^0_J` is `Σ_J G_{IJ} L_J`
///    with `G = σ^0 σ̃^a` the SO(9) gamma.
/// 3. Bosonic / fermionic Clifford-closure residuals on the *temporal* linkages
///    (the Garden relation that must hold for any valid worldline rep), and the
///    non-closure tensor `E_IJ`, defined per supercharge pair as the symmetric
///    closure remnant assembled across all ten linkage directions:
///       `E_IJ = ½ Σ_μ η^μμ (Δ^μ_I (Δ^μ_J)ᵀ + Δ^μ_J (Δ^μ_I)ᵀ) − 2 δ_IJ I`,
///    whose stacked Frobenius norm is returned as `e_norm`.
///
/// `l` must contain exactly sixteen 16x16 matrices.
pub fn assemble_and_check(l: &[Mat]) -> NonClosureReport {
    let n = l.len();
    assert_eq!(n, 16, "worldline N=16 expects sixteen L-matrices");
    let dim = Clifford10D::DIM;
    for m in l {
        assert_eq!((m.rows, m.cols), (dim, dim), "each L must be 16x16");
    }

    let cliff = Clifford10D::build();
    let id = Mat::identity(dim);

    // R = Lᵀ.
    let r: Vec<Mat> = l.iter().map(|m| m.transpose()).collect();

    // --- Temporal linkages Δ^0_I = L_I, and spatial Δ^a_I (a=1..9) ----------
    // delta[mu] is the Vec of sixteen linkage matrices for direction mu.
    let mut delta: Vec<Vec<Mat>> = Vec::with_capacity(10);
    delta.push(l.to_vec()); // μ = 0

    for a in 1..=9 {
        let g = cliff.spatial_generator(a); // (σ^0 σ̃^a), 16x16 on spinor index I
        let mut delta_a: Vec<Mat> = Vec::with_capacity(n);
        for ii in 0..n {
            // Δ^a_I = −Σ_J G_{IJ} L_J
            let mut acc = Mat::zeros(dim, dim);
            for (jj, lj) in l.iter().enumerate() {
                let coeff = g.get(ii, jj);
                if coeff != 0.0 {
                    acc = acc.add(&lj.scale(coeff));
                }
            }
            delta_a.push(acc.scale(-1.0));
        }
        delta.push(delta_a);
    }

    // --- Bosonic & fermionic closure residuals on the temporal linkages -----
    // These are the worldline Garden relations the rep must satisfy:
    //   L_I R_J + L_J R_I = 2 δ_IJ I   (bosonic)
    //   R_I L_J + R_J L_I = 2 δ_IJ I   (fermionic)
    let mut max_bos = 0.0f64;
    let mut max_fer = 0.0f64;
    for ii in 0..n {
        for jj in 0..n {
            let target = if ii == jj { id.scale(2.0) } else { Mat::zeros(dim, dim) };
            let bos = l[ii].matmul(&r[jj]).add(&l[jj].matmul(&r[ii]));
            max_bos = max_bos.max(bos.max_abs_diff(&target));
            let fer = r[ii].matmul(&l[jj]).add(&r[jj].matmul(&l[ii]));
            max_fer = max_fer.max(fer.max_abs_diff(&target));
        }
    }

    // --- Non-closure tensor E_IJ across all ten linkage directions ----------
    // E_IJ = ½ Σ_μ η^μμ (Δ^μ_I (Δ^μ_J)ᵀ + Δ^μ_J (Δ^μ_I)ᵀ) − 2 δ_IJ I .
    // EXPERIMENTAL: this residual's normalization is NOT yet calibrated (a fully
    // closed input does not map to 0 under the current target), so the returned
    // e_norm is a raw probe, not an off-shell certificate. Re-derive against a
    // known fixture before any physical interpretation.
    let mut e_sq = 0.0f64;
    let dt: Vec<Vec<Mat>> = delta
        .iter()
        .map(|dir| dir.iter().map(|m| m.transpose()).collect())
        .collect();
    for ii in 0..n {
        for jj in 0..n {
            let mut acc = Mat::zeros(dim, dim);
            for mu in 0..10 {
                let eta = cliff.eta[mu];
                let term = delta[mu][ii]
                    .matmul(&dt[mu][jj])
                    .add(&delta[mu][jj].matmul(&dt[mu][ii]));
                acc = acc.add(&term.scale(0.5 * eta));
            }
            if ii == jj {
                acc = acc.add(&id.scale(-2.0));
            }
            let f = acc.frobenius();
            e_sq += f * f;
        }
    }

    NonClosureReport {
        max_residual_bosonic: max_bos,
        max_residual_fermionic: max_fer,
        e_norm: e_sq.sqrt(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Mat basics ---------------------------------------------------------

    #[test]
    fn kron_dim_and_identity() {
        let a = Mat::identity(2);
        let b = Mat::identity(3);
        let k = a.kron(&b);
        assert_eq!((k.rows, k.cols), (6, 6));
        assert_eq!(k, Mat::identity(6));
    }

    #[test]
    fn kron4_is_16x16() {
        let g = kron4(&pauli_x(), &pauli_z(), &pauli_i(), &pauli_e());
        assert_eq!((g.rows, g.cols), (16, 16));
    }

    // --- SO(9) gammas -------------------------------------------------------

    #[test]
    fn so9_gammas_are_symmetric_and_anticommuting() {
        let g = so9_gammas();
        assert_eq!(g.len(), 9);
        let id = Mat::identity(16);
        for (a, ga) in g.iter().enumerate() {
            // Symmetric.
            assert!(ga.max_abs_diff(&ga.transpose()) < 1e-12, "g{a} not symmetric");
            // Square to +I.
            let sq = ga.matmul(ga);
            assert!(sq.max_abs_diff(&id) < 1e-12, "g{a}^2 != I");
            for (b, gb) in g.iter().enumerate() {
                if a == b {
                    continue;
                }
                // Anticommute: ga gb + gb ga = 0.
                let ac = ga.matmul(gb).add(&gb.matmul(ga));
                assert!(ac.max_abs() < 1e-12, "g{a},g{b} do not anticommute");
            }
        }
    }

    // --- THE load-bearing test ---------------------------------------------

    #[test]
    fn verify_clifford_residual_is_tiny() {
        let cliff = Clifford10D::build();
        let res = cliff.verify_clifford();
        assert!(res < 1e-9, "SO(1,9) Clifford residual {res} too large");
    }

    #[test]
    fn sigma_shapes_and_signature() {
        let cliff = Clifford10D::build();
        assert_eq!(cliff.sigma.len(), 10);
        assert_eq!(cliff.sigma_tilde.len(), 10);
        for m in cliff.sigma.iter().chain(cliff.sigma_tilde.iter()) {
            assert_eq!((m.rows, m.cols), (16, 16));
        }
        assert_eq!(cliff.eta[0], -1.0);
        for &e in &cliff.eta[1..] {
            assert_eq!(e, 1.0);
        }
        // σ^0 = +I, σ̃^0 = −I.
        assert_eq!(cliff.sigma[0], Mat::identity(16));
        assert_eq!(cliff.sigma_tilde[0], Mat::identity(16).scale(-1.0));
    }

    #[test]
    fn spatial_generators_are_so9_clifford() {
        // σ^0 σ̃^a = (+I)(γ^a) = γ^a, so the nine spatial generators must again
        // be a Euclidean SO(9) Clifford set.
        let cliff = Clifford10D::build();
        let id = Mat::identity(16);
        for a in 1..=9 {
            let ga = cliff.spatial_generator(a);
            let sq = ga.matmul(&ga);
            assert!(sq.max_abs_diff(&id) < 1e-12, "spatial gen {a} squares != I");
            for b in (a + 1)..=9 {
                let gb = cliff.spatial_generator(b);
                let ac = ga.matmul(&gb).add(&gb.matmul(&ga));
                assert!(ac.max_abs() < 1e-12, "spatial gens {a},{b} don't anticommute");
            }
        }
    }

    // --- assemble_and_check synthetic example ------------------------------

    /// Sixteen 16x16 L-matrices built so that the WORLDLINE Garden relation
    /// `L_I L_Jᵀ + L_J L_Iᵀ = 2 δ_IJ I` holds exactly: take L_I = the I-th of a
    /// set of sixteen real orthogonal anticommuting-up-to-sign matrices.
    ///
    /// We use the ten Clifford sigmas (orthogonal, mutually anticommuting in the
    /// split sense) padded with extra structure is overkill; instead use a clean
    /// closed set: the eight matrices {I, γ^1..γ^7} and their products won't all
    /// satisfy the Garden relation. The simplest *exact* N=16 Garden set on a
    /// 16-dim space is the left-regular representation of a Clifford algebra,
    /// which is heavy to build here. For a self-contained smoke test we instead
    /// verify the API on a degenerate-but-valid single-direction set where each
    /// L_I is an orthogonal matrix: L_I = σ̃-style symmetric gammas give
    /// L_I L_Iᵀ = I (diagonal of Garden holds); cross terms are exercised only
    /// for the construction wiring, not full closure.
    fn synthetic_ls() -> Vec<Mat> {
        // Use I and the nine SO(9) gammas (all orthogonal: Gᵀ = G, G² = I ⇒
        // G Gᵀ = I), then six more orthogonal matrices (products of disjoint
        // gammas) to reach sixteen. Each is orthogonal so L_I L_Iᵀ = I, which
        // makes the diagonal bosonic closure exact (2I). Off-diagonal terms are
        // generally nonzero here — this fixture exercises the assembly wiring
        // and shape contracts, not full Lorentz closure.
        let g = so9_gammas();
        let mut ls = Vec::with_capacity(16);
        ls.push(Mat::identity(16));
        for gi in &g {
            ls.push(gi.clone());
        }
        // Six products γ^a γ^b (a<b) — orthogonal (product of orthogonals).
        let pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3)];
        for &(a, b) in &pairs {
            ls.push(g[a].matmul(&g[b]));
        }
        assert_eq!(ls.len(), 16);
        ls
    }

    #[test]
    fn assemble_and_check_runs_and_shapes() {
        let ls = synthetic_ls();
        let report = assemble_and_check(&ls);
        // Every L_I is orthogonal, so the diagonal bosonic/fermionic closure is
        // exact; the worst residual is finite and the e_norm is well-defined.
        assert!(report.max_residual_bosonic.is_finite());
        assert!(report.max_residual_fermionic.is_finite());
        assert!(report.e_norm.is_finite());
        // Sanity: the report is reproducible.
        let report2 = assemble_and_check(&ls);
        assert_eq!(report, report2);
    }

    #[test]
    fn assemble_diagonal_closure_is_exact_for_orthogonal_ls() {
        // For orthogonal L_I, L_I L_Iᵀ = I, so the I==J bosonic term is exactly
        // 2I and contributes zero residual on the diagonal. Confirm the diagonal
        // contribution is numerically clean by checking each L is orthogonal.
        let ls = synthetic_ls();
        let id = Mat::identity(16);
        for (idx, m) in ls.iter().enumerate() {
            let llt = m.matmul(&m.transpose());
            assert!(llt.max_abs_diff(&id) < 1e-9, "L{idx} not orthogonal");
        }
    }
}
