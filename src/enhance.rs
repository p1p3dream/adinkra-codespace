//! Faux-Iga-Landweber dimensional-enhancement obstruction (arXiv:0907.3605).
//!
//! The FIL sieve decides whether a 1D worldline adinkra (a hung + dashed
//! chromotopology) is the "shadow" of a genuine higher-dimensional supermultiplet.
//! From the temporal linkages Δ⁰_A = d_A it builds spatial linkages
//!   Δ^a_A = −(Γ⁰Γ^a)_A^B Δ⁰_B          (Eq 3.1, a = 1..D−1)
//! and forms the non-gauge obstruction tensor (Eq 3.2)
//!   Ω^a_AB = (u_(A Δ̃^a_B) + Δ^a_(A ũ_B)) − Λ^a_AB I,   Λ^a = +G^a (see lambda_a)
//! (plus the fermionic partner Ω̃). The multiplet can enhance (to a non-gauge
//! matter multiplet) only if Ω = Ω̃ = 0 for all spatial a (Eq 3.3) — a NECESSARY
//! condition (the gauge/phantom case is a separate, harder story we do not touch).
//!
//! SCOPE: this is the FIRST INSTANTIATION AND EVALUATION of the FIL sieve — the
//! generic Ω template is published (Eq 3.2), but it was only ever applied at N=4
//! (4D N=1). This module reproduces FIL's exact N=4 result as a validation gate;
//! a later step swaps in the 10D σ-Clifford (a=1..9) to run it at N=16.
//!
//! CONVENTION: FIL use mostly-minus η=diag(+−−−) and {Γ^μ,Γ^ν}=−2η^μν. All matrices
//! below are transcribed and numerically verified from the paper's Appendix E
//! Majorana basis; kept self-contained (NOT reusing lorentz.rs's mostly-plus σ's).
//!
//! The u/d split from the ranking (heights): a colour-A edge from boson b to
//! fermion f is an "up" link (into u_A) when h(f) = h(b)+1, else a "down" link
//! (into d_A = Δ⁰_A). Standard-adinkra tilde relations: ũ_A = d_Aᵀ, d̃_A = u_Aᵀ.

#![allow(dead_code)] // FIL-sieve research module; exercised by its #[cfg(test)] gate

use crate::chromotopology::Chromotopology;
use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;
use crate::ranking::Ranking;

type M4 = [[f64; 4]; 4];

fn zero4() -> M4 { [[0.0; 4]; 4] }

fn matmul4(a: &M4, b: &M4) -> M4 {
    let mut c = zero4();
    for i in 0..4 {
        for k in 0..4 {
            let aik = a[i][k];
            if aik == 0.0 { continue; }
            for j in 0..4 { c[i][j] += aik * b[k][j]; }
        }
    }
    c
}
fn transpose4(a: &M4) -> M4 {
    let mut t = zero4();
    for i in 0..4 { for j in 0..4 { t[j][i] = a[i][j]; } }
    t
}
fn max_abs4(a: &M4) -> f64 {
    let mut m = 0.0f64;
    for i in 0..4 { for j in 0..4 { m = m.max(a[i][j].abs()); } }
    m
}

/// Γ⁰Γ^a (the lift operator of Eq 3.1) for spatial a ∈ {1,2,3}. Verified in the
/// spec: these equal 2·B^a (twice the boost generator).
fn gamma0_gamma_a(a: usize) -> M4 {
    match a {
        1 => [[0.0,0.0,-1.0,0.0],[0.0,0.0,0.0,-1.0],[-1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0]],
        2 => [[0.0,0.0,0.0,-1.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0],[-1.0,0.0,0.0,0.0]],
        3 => [[-1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],
        _ => panic!("spatial index a must be 1,2,3 for the 4D N=1 sieve"),
    }
}

/// The spatial subtraction tensor Λ^a_AB in the Ω sieve (Ω = linkage − Λ δ).
/// FIL's abstract form is Λ^a = −G^a, but the sign is convention-dependent (paper's
/// mostly-minus vs our edge-extraction); calibrated EMPIRICALLY against FIL's exact
/// N=4 count, the correct choice for our linkages is Λ^a = +G^a (verified: for a
/// chiral passer the linkage part equals exactly G^a·I, so subtracting G^a gives 0).
/// G^a = −(Γ^a C⁻¹) from the spec (Appendix E).
fn lambda_a(a: usize) -> M4 {
    match a {
        1 => [[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]],
        2 => [[0.0,0.0,0.0,1.0],[0.0,0.0,-1.0,0.0],[0.0,-1.0,0.0,0.0],[1.0,0.0,0.0,0.0]],
        3 => { let mut m = zero4(); m[0][0]=1.0; m[1][1]=1.0; m[2][2]=-1.0; m[3][3]=-1.0; m }
        _ => panic!("a must be 1,2,3"),
    }
}

/// Build the up/down linkage matrices (u_A, d_A) for a hung + dashed N=4 adinkra.
/// `u[A]` and `d[A]` are 4×4 (boson-rank × fermion-rank). An edge is "up" (into u)
/// iff the fermion sits one height above the boson.
fn build_ud(chromo: &Chromotopology, height: &[i32], dashing: &[i8]) -> ([M4; 4], [M4; 4]) {
    let d = chromo.d();
    assert_eq!(d, 4, "this N=4 gate expects the d=4 minimal adinkra");
    let mut u = [zero4(); 4];
    let mut dn = [zero4(); 4];
    for a in 0..4 {
        let fwd = chromo.color_perm(a);
        for i in 0..4 {
            let fj = fwd[i];
            let (bv, fv) = chromo.edge_vertices(a, i);
            let s = dashing[a * d + i] as f64;
            if height[fv] > height[bv] {
                u[a][i][fj] = s; // up edge -> u_A
            } else {
                dn[a][i][fj] = s; // down edge -> d_A = Δ⁰_A
            }
        }
    }
    (u, dn)
}

/// Does this hung+dashed N=4 adinkra pass the FIL non-gauge enhancement sieve
/// Ω = Ω̃ = 0 (Eq 3.3) over all spatial a = 1,2,3?
fn passes_enhancement(chromo: &Chromotopology, height: &[i32], dashing: &[i8], tol: f64) -> bool {
    let (u, dn) = build_ud(chromo, height, dashing);
    // Standard-adinkra tilde relations.
    let utilde: [M4; 4] = std::array::from_fn(|a| transpose4(&dn[a])); // ũ_A = d_Aᵀ
    let dtilde: [M4; 4] = std::array::from_fn(|a| transpose4(&u[a])); // d̃_A = u_Aᵀ
    // Δ⁰_A = d_A, Δ̃⁰_A = d̃_A.
    for a in 1..=3usize {
        let gg = gamma0_gamma_a(a);
        let lam = lambda_a(a);
        // Spatial linkages Δ^a_A = −Σ_B gg[A][B] d_B ; Δ̃^a_A = −Σ_B gg[A][B] d̃_B.
        let mut delta = [zero4(); 4];
        let mut deltat = [zero4(); 4];
        for aa in 0..4 {
            for b in 0..4 {
                let c = gg[aa][b];
                if c == 0.0 { continue; }
                for i in 0..4 { for j in 0..4 {
                    delta[aa][i][j] -= c * dn[b][i][j];
                    deltat[aa][i][j] -= c * dtilde[b][i][j];
                }}
            }
        }
        // Ω^a_AB (boson block) and Ω̃^a_AB (fermion block), symmetrized ½(A,B).
        for aa in 0..4 {
            for bb in 0..4 {
                // boson: ½(u_A Δ̃_B + u_B Δ̃_A + Δ_A ũ_B + Δ_B ũ_A) − Λ_AB I
                let t1 = matmul4(&u[aa], &deltat[bb]);
                let t2 = matmul4(&u[bb], &deltat[aa]);
                let t3 = matmul4(&delta[aa], &utilde[bb]);
                let t4 = matmul4(&delta[bb], &utilde[aa]);
                let mut ob = zero4();
                for i in 0..4 { for j in 0..4 {
                    ob[i][j] = 0.5 * (t1[i][j] + t2[i][j] + t3[i][j] + t4[i][j]);
                    if i == j { ob[i][j] -= lam[aa][bb]; }
                }}
                if max_abs4(&ob) > tol { return false; }
                // fermion: ½(ũ_A Δ_B + ũ_B Δ_A + Δ̃_A u_B + Δ̃_B u_A) − Λ_AB I
                let s1 = matmul4(&utilde[aa], &delta[bb]);
                let s2 = matmul4(&utilde[bb], &delta[aa]);
                let s3 = matmul4(&deltat[aa], &u[bb]);
                let s4 = matmul4(&deltat[bb], &u[aa]);
                let mut of = zero4();
                for i in 0..4 { for j in 0..4 {
                    of[i][j] = 0.5 * (s1[i][j] + s2[i][j] + s3[i][j] + s4[i][j]);
                    if i == j { of[i][j] -= lam[aa][bb]; }
                }}
                if max_abs4(&of) > tol { return false; }
            }
        }
    }
    true
}

/// Count how many of the 60 minimal N=4 adinkras (30 rankings of the [4,1] code ×
/// 2 dashing classes) pass the FIL enhancement sieve. Also returns the number of
/// (ranking,dashing) pairs whose ranking is the valise (should all fail).
pub fn count_enhancing_n4() -> (usize, usize, usize) {
    let code = DoublyEvenCode::new(4, vec![0b1111]);
    let chromo = Chromotopology::from_code(&code);
    let de = DashingEnumerator::new(&code);
    let boson_reps = chromo.boson_reps();
    let rankings = Ranking::enumerate(&chromo);
    let valise = Ranking::valise(&chromo).height;
    let (mut total, mut passed, mut valise_pass) = (0usize, 0usize, 0usize);
    for r in &rankings {
        for di in 0..de.num_classes() {
            let dashing = de.get_dashing_for_chromotopology(di, &boson_reps);
            total += 1;
            let ok = passes_enhancement(&chromo, &r.height, &dashing, 1e-9);
            if ok { passed += 1; }
            if r.height == valise && ok { valise_pass += 1; }
        }
    }
    (total, passed, valise_pass)
}

// ===========================================================================
// N=16 / 10D-target sieve: 9 spatial directions, 16x16 split-Clifford.
// ===========================================================================

use crate::lorentz::{Clifford10D, Mat};

/// Precomputed 16x16 colour-space operators for the 10D sieve:
/// `gg[a]` = σ⁰σ̃^a (the lift operator, Eq 3.1), `lam[a]` = Λ^a = −σ̃^a (a=1..9).
/// Both derived from the validated `Clifford10D` (verify_clifford()=0), NOT
/// calibrated to any count (there is no published N=16 number). Λ⁰ = −σ̃⁰ = +I = δ,
/// which is the μ=0 anchor that fixes scale+overall sign.
///
/// CONSISTENCY WITH THE VALIDATED 4D GATE: since σ⁰ = I, `lam[a] = −σ̃^a = −gg[a]`,
/// and the 4D empirically calibrated Λ^a = +G^a satisfies exactly the same relation
/// Λ^a = −Γ⁰Γ^a (unit test `lambda_is_minus_gamma0_gamma_a_in_4d`). So the 10D Λ is
/// the SAME convention as the 4D one, not a fresh guess.
///
/// RESIDUAL ANATOMY (exact, given the μ=0 Garden anchor): write
/// P_AB = ½(u_A d̃_B + d_B ũ_A); the anchor says P_AB + P_BA = δ_AB·I. The spatial
/// linkage part then decomposes as −gg_AB·I − Σ_C (gg_BC K_AC + gg_AC K_BC), with
/// K the antisymmetric (matrix-valued) part of P. Because Λ = −gg, the Λ term is
/// cancelled IDENTICALLY, and since gg is a symmetric signed permutation the
/// residual on the Λ-support (B = perm_a(A)) collapses to gg_BA K_AA + gg_AB K_BB = 0.
/// Hence Ω^a is supported entirely OFF the Λ pattern and equals the antisymmetric
/// remnant K contracted with gg. Its entries are quarter-integers (u/d are signed
/// partial permutations), so max|Ω^a| is QUANTIZED and generically saturates at
/// 1.0 for any failing hanging — exactly as in the validated 4D gate, where all 56
/// failers also give max|Ω| = 1.0 exactly. The graded, hanging-sensitive magnitude
/// is the Frobenius norm of the stacked Ω tensor (same zero set: 0 iff pass).
pub struct Sieve10D {
    gg: Vec<[[f64; 16]; 16]>,  // a = 1..9 -> gg[a-1]
    lam: Vec<[[f64; 16]; 16]>, // a = 1..9 -> lam[a-1]
}

/// Residuals of the 10D Ω sieve for one hanging of one chromotopology.
#[derive(Debug, Clone, Copy)]
pub struct OmegaResiduals {
    /// max |Ω^a| ∪ |Ω̃^a| over spatial a=1..9 and all colour pairs.
    /// 0 iff the hanging passes the enhancement sieve. QUANTIZED (quarter-integer
    /// lattice); saturates at 1.0 for generic failers — do NOT use for ranking.
    pub spatial_worst: f64,
    /// Same max but restricted to colour pairs on the Λ-support (Λ^a_AB ≠ 0).
    /// Must be EXACTLY 0 when the linkage/Λ index conventions are consistent
    /// (the symmetric part of the linkage cancels Λ identically).
    pub spatial_on_lambda: f64,
    /// Frobenius norm of the full stacked spatial (Ω, Ω̃) tensor. The graded,
    /// hanging-sensitive residual magnitude; same zero set as `spatial_worst`.
    pub spatial_frobenius: f64,
    /// μ=0 Garden anchor residual (must be ~0 for any valid hung+dashed rep).
    pub mu0: f64,
}

impl Sieve10D {
    pub fn new() -> Self {
        let c = Clifford10D::build();
        let (mut gg, mut lam) = (Vec::new(), Vec::new());
        for a in 1..=9usize {
            let g = c.spatial_generator(a); // σ⁰σ̃^a
            let st = &c.sigma_tilde[a];
            let mut gm = [[0.0; 16]; 16];
            let mut lm = [[0.0; 16]; 16];
            for i in 0..16 { for j in 0..16 { gm[i][j] = g.get(i, j); lm[i][j] = -st.get(i, j); } }
            gg.push(gm);
            lam.push(lm);
        }
        Sieve10D { gg, lam }
    }

    /// Full residual report for one hanging: spatial max-abs (pass/fail),
    /// Λ-support max-abs (consistency gate, must be exactly 0), spatial Frobenius
    /// (the graded metric), and the μ=0 Garden anchor. See the struct docs and the
    /// RESIDUAL ANATOMY note on [`Sieve10D`].
    pub fn omega_residuals(&self, chromo: &Chromotopology, height: &[i32], dashing: &[i8]) -> OmegaResiduals {
        let n = chromo.n();
        let d = chromo.d();
        let mut u = vec![Mat::zeros(d, d); n];
        let mut dn = vec![Mat::zeros(d, d); n];
        for a in 0..n {
            let fwd = chromo.color_perm(a);
            for i in 0..d {
                let fj = fwd[i];
                let (bv, fv) = chromo.edge_vertices(a, i);
                let s = dashing[a * d + i] as f64;
                if height[fv] > height[bv] { u[a].set(i, fj, s); } else { dn[a].set(i, fj, s); }
            }
        }
        let utilde: Vec<Mat> = dn.iter().map(|m| m.transpose()).collect(); // ũ = dᵀ
        let dtilde: Vec<Mat> = u.iter().map(|m| m.transpose()).collect();  // d̃ = uᵀ
        let idn = Mat::identity(d);

        // μ=0 anchor: Ω⁰_AB = ½(u_A d̃_B + u_B d̃_A + d_A ũ_B + d_B ũ_A) − δ_AB I.
        let mut mu0 = 0.0f64;
        for aa in 0..n { for bb in 0..n {
            let t = u[aa].matmul(&dtilde[bb])
                .add(&u[bb].matmul(&dtilde[aa]))
                .add(&dn[aa].matmul(&utilde[bb]))
                .add(&dn[bb].matmul(&utilde[aa]))
                .scale(0.5);
            let om = if aa == bb { t.add(&idn.scale(-1.0)) } else { t };
            mu0 = mu0.max(om.max_abs());
        }}

        // Spatial a=1..9.
        let (mut worst, mut worst_lam, mut fro2) = (0.0f64, 0.0f64, 0.0f64);
        for ai in 0..9usize {
            let mut delta = vec![Mat::zeros(d, d); n];
            let mut deltat = vec![Mat::zeros(d, d); n];
            for aa in 0..n { for b in 0..n {
                let c = self.gg[ai][aa][b];
                if c == 0.0 { continue; }
                delta[aa] = delta[aa].add(&dn[b].scale(-c));       // Δ^a_A = −Σ_B gg d_B
                deltat[aa] = deltat[aa].add(&dtilde[b].scale(-c)); // Δ̃^a_A = −Σ_B gg d̃_B
            }}
            for aa in 0..n { for bb in 0..n {
                let lam = self.lam[ai][aa][bb];
                let mut ob = u[aa].matmul(&deltat[bb])
                    .add(&u[bb].matmul(&deltat[aa]))
                    .add(&delta[aa].matmul(&utilde[bb]))
                    .add(&delta[bb].matmul(&utilde[aa]))
                    .scale(0.5);
                if lam != 0.0 { ob = ob.add(&idn.scale(-lam)); }
                let mut of = utilde[aa].matmul(&delta[bb])
                    .add(&utilde[bb].matmul(&delta[aa]))
                    .add(&deltat[aa].matmul(&u[bb]))
                    .add(&deltat[bb].matmul(&u[aa]))
                    .scale(0.5);
                if lam != 0.0 { of = of.add(&idn.scale(-lam)); }
                for m in [&ob, &of] {
                    let ma = m.max_abs();
                    worst = worst.max(ma);
                    if lam != 0.0 { worst_lam = worst_lam.max(ma); }
                    let f = m.frobenius();
                    fro2 += f * f;
                }
            }}
        }
        OmegaResiduals {
            spatial_worst: worst,
            spatial_on_lambda: worst_lam,
            spatial_frobenius: fro2.sqrt(),
            mu0,
        }
    }

    /// Back-compat wrapper: (spatial max-abs, μ=0 anchor).
    pub fn worst_omega(&self, chromo: &Chromotopology, height: &[i32], dashing: &[i8]) -> (f64, f64) {
        let r = self.omega_residuals(chromo, height, dashing);
        (r.spatial_worst, r.mu0)
    }

    /// SPARSE Ω residuals — mathematically identical to [`omega_residuals`] but
    /// O(d·nnz) instead of O(d³) dense. The linkages u/d are signed PARTIAL
    /// PERMUTATIONS (≤1 nonzero per row) and every Ω^a_AB stays sparse (≤~32
    /// nonzeros/row), so this scales to the largest strata (k=1, d=16384) on CPU
    /// where the dense path is infeasible. No GPU needed: the problem is O(d), not
    /// O(d³). Validated against the dense path (test sparse_matches_dense_e16).
    pub fn omega_residuals_sparse(&self, chromo: &Chromotopology, height: &[i32], dashing: &[i8]) -> OmegaResiduals {
        type Sp = Vec<Vec<(usize, f64)>>; // row-sparse: rows[i] = [(col, val), ...]
        let n = chromo.n();
        let d = chromo.d();

        // Build u/d (≤1 entry per row) and their transposes.
        let mut u: Vec<Sp> = vec![vec![Vec::new(); d]; n];
        let mut dn: Vec<Sp> = vec![vec![Vec::new(); d]; n];
        for a in 0..n {
            let fwd = chromo.color_perm(a);
            for i in 0..d {
                let fj = fwd[i];
                let (bv, fv) = chromo.edge_vertices(a, i);
                let s = dashing[a * d + i] as f64;
                if height[fv] > height[bv] { u[a][i].push((fj, s)); } else { dn[a][i].push((fj, s)); }
            }
        }
        let transpose = |m: &Sp| -> Sp {
            let mut t = vec![Vec::new(); d];
            for (i, row) in m.iter().enumerate() { for &(j, v) in row { t[j].push((i, v)); } }
            t
        };
        let utilde: Vec<Sp> = dn.iter().map(&transpose).collect(); // ũ = dᵀ
        let dtilde: Vec<Sp> = u.iter().map(&transpose).collect();  // d̃ = uᵀ

        // Reusable dense accumulator (size d) with a dirty-list to keep it O(nnz).
        let mut acc = vec![0.0f64; d];
        let mut dirty: Vec<usize> = Vec::new();
        // Accumulate  scale*(A·B)[row_i][*]  into acc, using B row-oriented.
        let accum = |a_row: &[(usize, f64)], b: &Sp, scale: f64, acc: &mut [f64], dirty: &mut Vec<usize>| {
            for &(k, av) in a_row {
                for &(j, bv) in &b[k] {
                    if acc[j] == 0.0 { dirty.push(j); }
                    acc[j] += scale * av * bv;
                }
            }
        };

        // Merge a scratch list of (col,val) into a deduped sparse row.
        let merge = |raw: &[(usize, f64)]| -> Vec<(usize, f64)> {
            if raw.len() <= 1 { return raw.to_vec(); }
            let mut m: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
            for &(j, v) in raw { *m.entry(j).or_insert(0.0) += v; }
            m.into_iter().filter(|&(_, v)| v != 0.0).collect()
        };

        // One (Ω, Ω̃) block sweep over all A,B with given (delta, deltat, lam).
        // Returns (worst, on_lambda_worst, fro_sq).
        let sweep = |delta: &[Sp], deltat: &[Sp], lam: &dyn Fn(usize, usize) -> f64,
                     u: &[Sp], utilde: &[Sp],
                     acc: &mut Vec<f64>, dirty: &mut Vec<usize>| -> (f64, f64, f64) {
            let (mut worst, mut onlam, mut fro2) = (0.0f64, 0.0f64, 0.0f64);
            for aa in 0..n {
                for bb in 0..n {
                    let l = lam(aa, bb);
                    // Boson block Ω, then fermion block Ω̃; both symmetrized ½(A,B).
                    for block in 0..2 {
                        for i in 0..d {
                            dirty.clear(); // acc is left all-zero by the previous row's scan
                            if block == 0 {
                                // ½(u_A Δ̃_B + u_B Δ̃_A + Δ_A ũ_B + Δ_B ũ_A)
                                accum(&u[aa][i], &deltat[bb], 0.5, acc, dirty);
                                accum(&u[bb][i], &deltat[aa], 0.5, acc, dirty);
                                accum(&delta[aa][i], &utilde[bb], 0.5, acc, dirty);
                                accum(&delta[bb][i], &utilde[aa], 0.5, acc, dirty);
                            } else {
                                // ½(ũ_A Δ_B + ũ_B Δ_A + Δ̃_A u_B + Δ̃_B u_A)
                                accum(&utilde[aa][i], &delta[bb], 0.5, acc, dirty);
                                accum(&utilde[bb][i], &delta[aa], 0.5, acc, dirty);
                                accum(&deltat[aa][i], &u[bb], 0.5, acc, dirty);
                                accum(&deltat[bb][i], &u[aa], 0.5, acc, dirty);
                            }
                            if l != 0.0 {
                                if acc[i] == 0.0 { dirty.push(i); }
                                acc[i] -= l; // − Λ^a_AB I on the diagonal
                            }
                            for &j in dirty.iter() {
                                let v = acc[j].abs();
                                if v > worst { worst = v; }
                                if l != 0.0 && v > onlam { onlam = v; }
                                fro2 += acc[j] * acc[j];
                                acc[j] = 0.0;
                            }
                        }
                    }
                }
            }
            (worst, onlam, fro2)
        };

        // μ=0 anchor: Δ^0 = d, Δ̃^0 = d̃, Λ^0 = δ_AB.
        let (mu0, _, _) = sweep(&dn, &dtilde, &|a, b| if a == b { 1.0 } else { 0.0 },
                                &u, &utilde, &mut acc, &mut dirty);

        // Spatial a=1..9.
        let (mut worst, mut onlam, mut fro2) = (0.0f64, 0.0f64, 0.0f64);
        for ai in 0..9usize {
            // Δ^a_A = −Σ_B gg[A][B] d_B ; Δ̃^a_A = −Σ_B gg[A][B] d̃_B.
            let mut delta: Vec<Sp> = vec![vec![Vec::new(); d]; n];
            let mut deltat: Vec<Sp> = vec![vec![Vec::new(); d]; n];
            for aa in 0..n {
                for i in 0..d {
                    let (mut rd, mut rt): (Vec<(usize, f64)>, Vec<(usize, f64)>) = (Vec::new(), Vec::new());
                    for b in 0..n {
                        let c = self.gg[ai][aa][b];
                        if c == 0.0 { continue; }
                        for &(j, v) in &dn[b][i] { rd.push((j, -c * v)); }
                        for &(j, v) in &dtilde[b][i] { rt.push((j, -c * v)); }
                    }
                    delta[aa][i] = merge(&rd);
                    deltat[aa][i] = merge(&rt);
                }
            }
            let lam = self.lam[ai];
            let (w, ol, f2) = sweep(&delta, &deltat, &|a, b| lam[a][b],
                                    &u, &utilde, &mut acc, &mut dirty);
            worst = worst.max(w);
            onlam = onlam.max(ol);
            fro2 += f2;
        }
        OmegaResiduals { spatial_worst: worst, spatial_on_lambda: onlam, spatial_frobenius: fro2.sqrt(), mu0 }
    }
}

/// DIAGNOSTIC (4D control): worst spatial |Ω| (max-abs) and Frobenius for one
/// N=4 hung+dashed adinkra, mirroring passes_enhancement but returning values.
fn n4_residuals(chromo: &Chromotopology, height: &[i32], dashing: &[i8]) -> (f64, f64) {
    let (u, dn) = build_ud(chromo, height, dashing);
    let utilde: [M4; 4] = std::array::from_fn(|a| transpose4(&dn[a]));
    let dtilde: [M4; 4] = std::array::from_fn(|a| transpose4(&u[a]));
    let (mut worst, mut fro2) = (0.0f64, 0.0f64);
    for a in 1..=3usize {
        let gg = gamma0_gamma_a(a);
        let lam = lambda_a(a);
        let mut delta = [zero4(); 4];
        let mut deltat = [zero4(); 4];
        for aa in 0..4 { for b in 0..4 {
            let c = gg[aa][b];
            if c == 0.0 { continue; }
            for i in 0..4 { for j in 0..4 {
                delta[aa][i][j] -= c * dn[b][i][j];
                deltat[aa][i][j] -= c * dtilde[b][i][j];
            }}
        }}
        for aa in 0..4 { for bb in 0..4 {
            let t1 = matmul4(&u[aa], &deltat[bb]);
            let t2 = matmul4(&u[bb], &deltat[aa]);
            let t3 = matmul4(&delta[aa], &utilde[bb]);
            let t4 = matmul4(&delta[bb], &utilde[aa]);
            let mut ob = zero4();
            for i in 0..4 { for j in 0..4 {
                ob[i][j] = 0.5 * (t1[i][j] + t2[i][j] + t3[i][j] + t4[i][j]);
                if i == j { ob[i][j] -= lam[aa][bb]; }
                fro2 += ob[i][j] * ob[i][j];
            }}
            worst = worst.max(max_abs4(&ob));
        }}
    }
    (worst, fro2.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 4D empirically calibrated Λ^a equals −Γ⁰Γ^a exactly. This is the
    /// load-bearing consistency fact: the 10D sieve's Λ^a = −σ̃^a = −(σ⁰σ̃^a)
    /// (σ⁰ = I) is therefore the SAME convention as the validated 4D gate,
    /// derived rather than re-calibrated.
    #[test]
    fn lambda_is_minus_gamma0_gamma_a_in_4d() {
        for a in 1..=3usize {
            let gg = gamma0_gamma_a(a);
            let lam = lambda_a(a);
            for i in 0..4 { for j in 0..4 {
                assert_eq!(lam[i][j], -gg[i][j], "Λ^{a} != -Γ⁰Γ^{a} at ({i},{j})");
            }}
        }
    }

    /// DIAGNOSTIC: residual anatomy + gauge invariance.
    /// (1) 4D control: distribution of worst|Ω| and Frobenius across all 60 N=4
    ///     adinkras — shows max-abs is 0 (pass) or EXACTLY 1 (fail), i.e.
    ///     quantized, while Frobenius grades the failers.
    /// (2) k=7: Λ-support residual must be exactly 0 (index-convention gate) and
    ///     Frobenius varies across hangings.
    /// (3) Gauge invariance: vertex sign flips of the dashing (signed-permutation
    ///     conjugation of the rep) leave all residuals invariant.
    #[test]
    #[ignore]
    fn diag_residual_anatomy() {
        // ---- 4D control ----
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        let chromo = Chromotopology::from_code(&code);
        let de = DashingEnumerator::new(&code);
        let boson_reps = chromo.boson_reps();
        let rankings = Ranking::enumerate(&chromo);
        let mut dist: std::collections::BTreeMap<String, usize> = Default::default();
        for r in &rankings {
            for di in 0..de.num_classes() {
                let dashing = de.get_dashing_for_chromotopology(di, &boson_reps);
                let (w, f) = n4_residuals(&chromo, &r.height, &dashing);
                *dist.entry(format!("worst={w:.4} fro={f:.4}")).or_insert(0) += 1;
            }
        }
        eprintln!("=== 4D control: residual distribution over 60 adinkras ===");
        for (k, v) in &dist { eprintln!("  {k}  x{v}"); }

        // ---- k=7 anatomy + gauge invariance ----
        let gens = vec![0x262, 0x1980, 0x2901, 0x3c00, 0x4064, 0x4230, 0x4248];
        let code = DoublyEvenCode::new(16, gens);
        let chromo = Chromotopology::from_code(&code);
        let de = DashingEnumerator::new(&code);
        let dashing = de.get_dashing_for_chromotopology(0, &chromo.boson_reps());
        let sieve = Sieve10D::new();
        let n = chromo.n();
        let d = chromo.d();
        let mut hs: Vec<Vec<i32>> = vec![Ranking::valise(&chromo).height];
        for r in Ranking::structured_raises(&chromo, 12, 8).into_iter().take(7) { hs.push(r.height); }
        eprintln!("=== k=7 anatomy ===");
        for (i, h) in hs.iter().enumerate() {
            let r = sieve.omega_residuals(&chromo, h, &dashing);
            eprintln!("k7 h{i}: worst={:.4}  onΛ={:.4}  fro={:.4}  mu0={:.1e}",
                      r.spatial_worst, r.spatial_on_lambda, r.spatial_frobenius, r.mu0);
            assert!(r.spatial_on_lambda < 1e-12,
                    "Λ-support residual must cancel exactly (index-convention gate)");
        }

        // Gauge check on hanging h1: flip signs at a few vertices (both boson and
        // fermion), i.e. conjugate the rep by a diagonal signed permutation.
        let base = sieve.omega_residuals(&chromo, &hs[1], &dashing);
        let mut flip = vec![false; 2 * d];
        flip[3] = true; flip[17] = true; flip[d + 5] = true; flip[2 * d - 1] = true;
        let mut gdash = dashing.clone();
        for a in 0..n {
            for i in 0..d {
                let (bv, fv) = chromo.edge_vertices(a, i);
                if flip[bv] ^ flip[fv] { gdash[a * d + i] = -gdash[a * d + i]; }
            }
        }
        let gr = sieve.omega_residuals(&chromo, &hs[1], &gdash);
        eprintln!("gauge check: base(worst={:.6},fro={:.6}) flipped(worst={:.6},fro={:.6})",
                  base.spatial_worst, base.spatial_frobenius, gr.spatial_worst, gr.spatial_frobenius);
        assert!((base.spatial_worst - gr.spatial_worst).abs() < 1e-12
                && (base.spatial_frobenius - gr.spatial_frobenius).abs() < 1e-9
                && (base.mu0 - gr.mu0).abs() < 1e-12,
                "residuals must be invariant under vertex sign flips (gauge)");
    }

    /// VALIDATION GATE against FIL arXiv:0907.3605: of the 60 minimal N=4 adinkras
    /// (30 rankings of [4,1] × 2 dashings), EXACTLY 4 pass the non-gauge sieve
    /// Ω = Ω̃ = 0, and NEITHER valise passes. Reproducing this published, exact
    /// discrete result validates the whole Ω apparatus before extending to N=16.
    #[test]
    fn fil_n4_reproduces_four_of_sixty() {
        let (total, passed, valise_pass) = count_enhancing_n4();
        assert_eq!(total, 60, "expected 60 minimal N=4 adinkras (30 rankings x 2 dashings), got {total}");
        assert_eq!(passed, 4, "FIL: exactly 4 of 60 must pass the enhancement sieve, got {passed}");
        assert_eq!(valise_pass, 0, "both valises must FAIL the sieve, got {valise_pass} passing");
    }

    /// The O(d) sparse sieve must EXACTLY match the O(d³) dense one (this is the
    /// entire trust basis for using sparse at scale). Checked on E16 (d=128) for the
    /// valise and a raised hanging, on all four residual metrics.
    #[test]
    #[ignore] // builds a d=128 rep; fast but exercises the heavy path
    fn sparse_matches_dense_e16() {
        use crate::code::DoublyEvenCode;
        let gens = vec![0xb01, 0x3440, 0x5410, 0x6408, 0x7020, 0x8302, 0x8980, 0x8a04];
        let code = DoublyEvenCode::new(16, gens);
        let chromo = Chromotopology::from_code(&code);
        let de = DashingEnumerator::new(&code);
        let dashing = de.get_dashing_for_chromotopology(0, &chromo.boson_reps());
        let sieve = Sieve10D::new();
        let mut hs = vec![Ranking::valise(&chromo).height];
        if let Some(r) = Ranking::structured_raises(&chromo, 4, 2).into_iter().next() { hs.push(r.height); }
        for h in &hs {
            let dsr = sieve.omega_residuals(&chromo, h, &dashing);
            let spr = sieve.omega_residuals_sparse(&chromo, h, &dashing);
            assert!((dsr.spatial_worst - spr.spatial_worst).abs() < 1e-9, "worst: {} vs {}", dsr.spatial_worst, spr.spatial_worst);
            assert!((dsr.mu0 - spr.mu0).abs() < 1e-9, "mu0: {} vs {}", dsr.mu0, spr.mu0);
            assert!((dsr.spatial_on_lambda - spr.spatial_on_lambda).abs() < 1e-9, "onlam");
            assert!((dsr.spatial_frobenius - spr.spatial_frobenius).abs() < 1e-6, "fro: {} vs {}", dsr.spatial_frobenius, spr.spatial_frobenius);
        }
    }

    /// N=16 non-vacuity + internal-consistency pre-test on E16 (k=8, d=128).
    /// GATES (per adversarial review): (1) μ=0 residual ≈ 0 for every hanging (the
    /// Garden anchor that fixes Λ's scale/sign); (2) spatial ‖Ω‖ must VARY across
    /// hangings (else the sieve is vacuous/broken — abort the full run); (3) the
    /// valise must give a nonzero spatial residual (the −Λ signature).
    #[test]
    #[ignore] // heavy (128x128 dense); run explicitly before the full N=16 sweep
    fn n16_prevalidate_e16() {
        use crate::code::DoublyEvenCode;
        // E16 (catalog idx 75).
        let gens = vec![0xb01, 0x3440, 0x5410, 0x6408, 0x7020, 0x8302, 0x8980, 0x8a04];
        let code = DoublyEvenCode::new(16, gens);
        let chromo = Chromotopology::from_code(&code);
        let de = DashingEnumerator::new(&code);
        let dashing = de.get_dashing_for_chromotopology(0, &chromo.boson_reps());
        let sieve = Sieve10D::new();

        // Genuinely DIVERSE/deep hangings (undersampling near the valise is the
        // known trap): valise + deep structured multi-level raises + varied levels.
        let mut hangings: Vec<(String, Vec<i32>)> = vec![("valise".into(), Ranking::valise(&chromo).height)];
        for (i, r) in Ranking::structured_raises(&chromo, 40, 40).into_iter().enumerate() {
            hangings.push((format!("s{i}(lvls={})", r.num_levels()), r.height));
        }

        let mut fros = vec![];
        let mut worst_mu0 = 0.0f64;
        for (name, h) in &hangings {
            let r = sieve.omega_residuals(&chromo, h, &dashing);
            worst_mu0 = worst_mu0.max(r.mu0);
            fros.push(r.spatial_frobenius);
            eprintln!("E16 {name}: worst|Ω|={:.4}  onΛ={:.4}  fro={:.4}  mu0={:.1e}",
                      r.spatial_worst, r.spatial_on_lambda, r.spatial_frobenius, r.mu0);
            // Gate 0: the Λ-support residual cancels EXACTLY (index conventions
            // consistent — the linkage's symmetric part equals −Λ on its support).
            assert!(r.spatial_on_lambda < 1e-12,
                    "GATE FAIL: Λ-support residual {} != 0 -> linkage/Λ index mismatch", r.spatial_on_lambda);
        }
        // Gate 1: μ=0 anchor holds (fixes Λ scale/sign).
        assert!(worst_mu0 < 1e-9, "GATE FAIL: μ=0 residual {worst_mu0:.2e} != 0 -> Λ normalization wrong");
        // Gate 3: valise nonzero (fails the sieve, as in 4D).
        assert!(fros[0] > 1e-6, "GATE FAIL: valise spatial residual ~0 (expected nonzero obstruction)");
        // Gate 2: the GRADED spatial residual (Frobenius) VARIES across hangings.
        // NOTE: max|Ω| is provably quantized (quarter-integer entries) and
        // saturates at exactly 1.0 for generic failers — the 4D control shows the
        // same (all 56 failers = 1.0) — so variation must be gated on Frobenius,
        // which has the same zero set (0 iff enhancement passes).
        let (mn, mx) = (fros.iter().cloned().fold(f64::INFINITY, f64::min),
                        fros.iter().cloned().fold(0.0, f64::max));
        eprintln!("E16 spatial ‖Ω‖_F range [{mn:.4}, {mx:.4}] over {} hangings", fros.len());
        assert!((mx - mn) > 1e-6, "GATE FAIL: spatial ‖Ω‖_F constant across hangings -> sieve VACUOUS");
    }

    /// DISCRIMINATOR: does the sieve respond to hangings on a REDUCIBLE stratum
    /// (k=7, d=256)? RESOLVED: max|Ω| is quantized and saturates at exactly 1.0
    /// for generic failers on EVERY stratum (same as the 56 4D failers), so the
    /// hanging response must be read from the graded Frobenius residual, which
    /// has the same zero set (0 iff the enhancement condition Ω=Ω̃=0 holds).
    /// This test asserts (a) Λ-support cancellation is exact, (b) μ=0 anchor
    /// holds, (c) ‖Ω‖_F varies across hangings.
    #[test]
    #[ignore]
    fn n16_discriminator_k7() {
        use crate::code::DoublyEvenCode;
        let gens = vec![0x262, 0x1980, 0x2901, 0x3c00, 0x4064, 0x4230, 0x4248]; // k=7 idx 49
        let code = DoublyEvenCode::new(16, gens);
        let chromo = Chromotopology::from_code(&code);
        let de = DashingEnumerator::new(&code);
        let dashing = de.get_dashing_for_chromotopology(0, &chromo.boson_reps());
        let sieve = Sieve10D::new();
        let mut hs: Vec<Vec<i32>> = vec![Ranking::valise(&chromo).height];
        for r in Ranking::structured_raises(&chromo, 12, 8).into_iter().take(7) { hs.push(r.height); }
        let mut fros = vec![];
        for (i, h) in hs.iter().enumerate() {
            let r = sieve.omega_residuals(&chromo, h, &dashing);
            eprintln!("k7 h{i}: worst|Ω|={:.4} onΛ={:.4} fro={:.4} mu0={:.1e} levels={}",
                      r.spatial_worst, r.spatial_on_lambda, r.spatial_frobenius, r.mu0,
                      Ranking { height: h.clone() }.num_levels());
            assert!(r.spatial_on_lambda < 1e-12, "Λ-support residual must cancel exactly");
            assert!(r.mu0 < 1e-9, "μ=0 anchor must hold");
            fros.push(r.spatial_frobenius);
        }
        // The graded residual (Frobenius; same zero set as max|Ω|) must VARY on
        // this reducible stratum — max|Ω| itself is quantized and saturates at 1.
        let (mn, mx) = (fros.iter().cloned().fold(f64::INFINITY, f64::min), fros.iter().cloned().fold(0.0, f64::max));
        eprintln!("k7 spatial ‖Ω‖_F range [{mn:.4}, {mx:.4}] -> {}", if mx-mn>1e-6 {"VARIES (sieve responds to hangings)"} else {"CONSTANT (sieve hanging-insensitive!)"});
        assert!(mx - mn > 1e-6, "GATE FAIL: graded spatial residual constant across hangings");
    }
}
