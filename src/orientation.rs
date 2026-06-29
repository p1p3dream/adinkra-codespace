//! Orientation (gauge) structure of the cross-summand gadget, and why the k<8
//! off-diagonal "value spectrum" is NOT a basis-invariant classification.
//!
//! # The open question, resolved
//!
//! For k<8 a valise rep is reducible into `r = d/dmin` irreducible summands, each
//! produced by [`crate::decompose::decompose_rep`] in whatever orthonormal basis
//! `B` the eigensolver happens to land on. Replacing a summand's basis by `B·Q`
//! for any orthogonal `Q ∈ O(dmin)` sends the restricted operators
//! `L_I|_W → Qᵀ L_I|_W Q` and hence the holoraumy `Vtilde_IJ → Qᵀ Vtilde_IJ Q`.
//!
//! * The SELF gadget `Σ Tr(Vtilde_IJ Vtilde_IJ)` is conjugation-invariant (the
//!   `Q`s cancel cyclically), which is why every summand self-gadget is exactly
//!   `1.0` regardless of basis.
//! * The CROSS gadget between two summands `Σ Tr(Vtilde^a_IJ Vtilde^b_IJ)` is NOT
//!   invariant under rotating ONLY one of them: it depends on the relative
//!   orientation of the two arbitrary bases. This is the orientation freedom that
//!   made the k<8 off-diagonal values a run-specific fingerprint.
//!
//! Is there a relative orientation that turns the cross value into an invariant?
//! Yes, and it collapses the question. (The orientation is unique only up to a
//! residual division-algebra unit, but the resulting gadget VALUE is invariant,
//! which is all that matters here.) The real commutant of the full
//! rep (already computed exactly by [`crate::decompose::commutant_orbits`])
//! contains the intertwiners between equivalent summands. For summands `a`, `b`
//! and a generic commutant element `M`, the block `T = B_bᵀ M B_a` satisfies
//! `T L_I|_a = L_I|_b T` exactly. `TᵀT` commutes with `L_I|_a`, so the orthogonal
//! polar factor `U` of `T` is itself an exact intertwiner:
//!
//!   `Uᵀ L_I|_b U = L_I|_a`   for all I.
//!
//! Transporting `b` into `a`'s frame by this canonical `U` therefore makes `b`'s
//! operators IDENTICAL to `a`'s, so the canonically-aligned cross gadget of two
//! equivalent summands is exactly the self gadget, `1.0`. There is no nontrivial
//! orientation-invariant hiding in the equivalent-summand cross terms: every value
//! other than `1.0` is gauge.
//!
//! Consequence (the honest classification): the invariant content of a k<8 stratum
//! is (1) WHICH irreducible classes occur among the summands (a per-summand label,
//! e.g. via the conjugation-invariant chromocharacter of [`crate::chromochar`] or a
//! catalog match) and (2) the gadget BETWEEN inequivalent classes, which is exactly
//! the irreducible k=8 catalog already validated against Gates 2025. Inequivalent
//! summands carry no intertwiner (`T ≈ 0`), so there is no canonical identification
//! to align them within a single reducible rep; their genuine Gates gadget is the
//! one computed in the shared standard basis of the catalog. The dense per-stratum
//! "distinct off-diagonal value" counts (42403 at k=5, 58424 at k=6) are thus
//! orientation noise, not new invariants.
//!
//! This module provides the gauge action ([`gauge_conjugate`]), the canonical
//! intertwiner ([`intertwiner_orthogonal`]), and the machine-checked tests that
//! establish the statements above.

use crate::decompose::{commutant_orbits, DenseMat, IrrepSummand};
use crate::lr_matrix::AdinkraRep;

/// Local deterministic xorshift64* PRNG (kept private so this module does not
/// reach into `decompose`'s internals; same generator, independent state).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
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
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * u - 1.0
    }
}

/// A random `dim×dim` orthogonal matrix via modified Gram-Schmidt on random
/// columns. Not Haar-uniform, but a generic element of `O(dim)` — enough to
/// exercise the gauge action in tests.
pub fn random_orthogonal(dim: usize, seed: u64) -> DenseMat {
    let mut rng = Rng::new(seed);
    let mut cols: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..dim).map(|_| rng.next_signed()).collect())
        .collect();
    for i in 0..dim {
        for j in 0..i {
            let cj = cols[j].clone();
            let dot: f64 = (0..dim).map(|k| cols[i][k] * cj[k]).sum();
            for k in 0..dim {
                cols[i][k] -= dot * cj[k];
            }
        }
        let norm: f64 = (0..dim).map(|k| cols[i][k] * cols[i][k]).sum::<f64>().sqrt();
        for k in 0..dim {
            cols[i][k] /= norm;
        }
    }
    // Assemble: column i of the matrix is cols[i].
    let mut q = DenseMat::zeros(dim, dim);
    for i in 0..dim {
        for r in 0..dim {
            q.set(r, i, cols[i][r]);
        }
    }
    q
}

/// Apply a gauge (orthonormal re-basis) `Q ∈ O(dmin)` to a summand:
/// `L_I|_W → Qᵀ L_I|_W Q` and `B → B·Q`. The result is the SAME invariant
/// subspace described in a rotated internal frame; all basis-independent
/// quantities (self-gadget, Garden algebra residual, commutant type) are
/// unchanged, while the cross gadget against a fixed other summand generally is
/// not.
pub fn gauge_conjugate(s: &IrrepSummand, q: &DenseMat) -> IrrepSummand {
    assert_eq!(q.rows, s.dmin, "gauge Q must be dmin×dmin");
    assert_eq!(q.cols, s.dmin, "gauge Q must be dmin×dmin");
    let qt = q.transpose();
    // Q must be orthogonal for this to be a valid gauge (debug-only: O(dmin³)).
    debug_assert!(
        qt.matmul(q).max_abs_diff(&DenseMat::identity(s.dmin)) < 1e-9,
        "gauge_conjugate: Q is not orthogonal (QᵀQ != I)"
    );
    let l_restricted = s
        .l_restricted
        .iter()
        .map(|l| qt.matmul(l).matmul(q))
        .collect();
    IrrepSummand {
        n: s.n,
        dmin: s.dmin,
        basis: s.basis.matmul(q),
        l_restricted,
    }
}

/// Frobenius norm of a dense matrix.
fn frobenius(m: &DenseMat) -> f64 {
    m.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Build a generic real commutant element of `rep` (a `d×d` matrix commuting with
/// every `L_I`), as a `seed`-determined linear combination of the exact
/// sign-consistent commutant orbits. Generic ⇒ its off-diagonal summand blocks are
/// non-degenerate intertwiners between equivalent summands.
pub fn generic_commutant_element(rep: &AdinkraRep, seed: u64) -> DenseMat {
    let d = rep.d;
    let orbits = commutant_orbits(rep);
    let mut rng = Rng::new(seed);
    let mut m = DenseMat::zeros(d, d);
    for orbit in &orbits {
        let c = rng.next_signed();
        for &(cell, sign) in orbit {
            let (a, b) = (cell / d, cell % d);
            m.set(a, b, c * sign as f64);
        }
    }
    m
}

/// Orthogonal polar factor `U` of a square matrix `T` (the orthogonal matrix
/// closest to `T`): `T = U H`, `H = (TᵀT)^{1/2}` SPD, `U = T H^{-1}`. Returns
/// `None` if `T` is (numerically) singular, i.e. `min eig(TᵀT)` is below `tol²`
/// relative to the largest — the signal that the two summands are INEQUIVALENT and
/// no intertwiner exists.
pub fn orthogonal_polar_factor(t: &DenseMat, tol: f64) -> Option<DenseMat> {
    assert_eq!(t.rows, t.cols, "polar factor needs a square matrix");
    let s = t.transpose().matmul(t); // TᵀT, symmetric PSD
    let (eigs, v) = crate::decompose::jacobi_eigen(&s);
    let max_eig = eigs.iter().cloned().fold(0.0f64, f64::max);
    if max_eig <= 0.0 {
        return None;
    }
    let dim = t.rows;
    // H^{-1} = (TᵀT)^{-1/2} = V diag(1/sqrt(eig)) Vᵀ; reject if any eigenvalue ~0.
    let mut inv_sqrt = DenseMat::zeros(dim, dim);
    for (i, &e) in eigs.iter().enumerate() {
        if e <= tol * tol * max_eig {
            return None; // singular direction -> inequivalent (no intertwiner)
        }
        inv_sqrt.set(i, i, 1.0 / e.sqrt());
    }
    let h_inv = v.matmul(&inv_sqrt).matmul(&v.transpose()); // V D^{-1/2} Vᵀ
    Some(t.matmul(&h_inv))
}

/// An orthogonal intertwiner `U` aligning summand `b` to summand `a`
/// (`Uᵀ L_I|_b U = L_I|_a` for all I), derived from a generic commutant element.
/// Returns `None` when `a` and `b` are inequivalent irreducibles (no intertwiner).
///
/// NOT unique: the intertwiner space of two equivalent irreducibles is the
/// division algebra `D` (`dim 1, 2, 4` for `R, C, H`), so `U` is defined only up to
/// a residual `D`-unit. That residual does not change the aligned gadget value
/// (the value IS invariant; see the gauge-invariance test), only the frame.
///
/// A SINGLE generic commutant element can have a (near-)degenerate `a→b` block even
/// for equivalent summands, which would be a spurious false negative, so this
/// retries over several independent generic elements before concluding "no
/// intertwiner". For genuinely inequivalent summands every block is `0` (Schur), so
/// every try is rejected and the result is correctly `None`.
pub fn intertwiner_orthogonal(
    rep: &AdinkraRep,
    a: &IrrepSummand,
    b: &IrrepSummand,
    seed: u64,
) -> Option<DenseMat> {
    const TRIES: usize = 8;
    for t_i in 0..TRIES as u64 {
        let s = seed ^ t_i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678);
        let m = generic_commutant_element(rep, s);
        // T = B_bᵀ M B_a  (dmin×dmin); intertwines the restricted reps a -> b.
        let t = b.basis.transpose().matmul(&m).matmul(&a.basis);
        if frobenius(&t) < 1e-9 {
            continue; // degenerate block this try (or inequivalent); try another M
        }
        if let Some(u) = orthogonal_polar_factor(&t, 1e-6) {
            return Some(u);
        }
    }
    None // no non-degenerate intertwiner found over all tries -> inequivalent
}

/// Residual `max_I ‖Uᵀ L_I|_b U − L_I|_a‖_∞` — zero iff `U` exactly aligns `b` to
/// `a`. Used to certify a candidate intertwiner.
pub fn intertwiner_residual(a: &IrrepSummand, b: &IrrepSummand, u: &DenseMat) -> f64 {
    let bu = gauge_conjugate(b, u); // l_restricted = Uᵀ L_b U
    let mut worst = 0.0f64;
    for (la, lbu) in a.l_restricted.iter().zip(bu.l_restricted.iter()) {
        worst = worst.max(la.max_abs_diff(lbu));
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompose::{
        dense_gadget, decompose_rep, DenseHoloraumy,
    };
    use crate::signed_perm::SignedPerm;

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
    /// Block-diagonal direct sum (d = da+db) to build a genuinely reducible rep.
    fn block_diag(a: &AdinkraRep, b: &AdinkraRep) -> AdinkraRep {
        let n = a.n;
        let (da, db) = (a.d, b.d);
        let mut l_matrices = Vec::with_capacity(n);
        for i in 0..n {
            let (la, lb) = (&a.l_matrices[i], &b.l_matrices[i]);
            let mut perm = vec![0u16; da + db];
            let mut sign = vec![0i8; da + db];
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
        AdinkraRep { n, d: da + db, l_matrices }
    }

    fn self_gadget(s: &IrrepSummand) -> f64 {
        let dh = DenseHoloraumy::from_summand(s);
        dense_gadget(&dh, &dh)
    }
    fn cross_gadget(a: &IrrepSummand, b: &IrrepSummand) -> f64 {
        dense_gadget(&DenseHoloraumy::from_summand(a), &DenseHoloraumy::from_summand(b))
    }

    /// The raw cross gadget between two equivalent summands is GAUGE-DEPENDENT
    /// (changes under an O(dmin) rotation of one summand), while the self gadget is
    /// gauge-INVARIANT. This is the precise statement that the off-diagonal value
    /// is orientation noise.
    #[test]
    fn cross_gadget_is_gauge_dependent_self_is_not() {
        let rep = block_diag(&cs_n4(), &cs_n4()); // d=8 -> two equivalent summands
        let dec = decompose_rep(&rep).unwrap();
        let (a, b) = (&dec.summands[0], &dec.summands[1]);

        let self_before = self_gadget(b);
        let raw_before = cross_gadget(a, b);

        // Rotate b by several random gauges; self must be pinned to 1.0, and at
        // least one gauge must move the cross value by an O(1) amount.
        let mut moved = false;
        for seed in 1..8u64 {
            let q = random_orthogonal(b.dmin, seed);
            let bq = gauge_conjugate(b, &q);
            assert!((self_gadget(&bq) - self_before).abs() < 1e-9, "self gadget not gauge-invariant");
            assert!((self_before - 1.0).abs() < 1e-9, "self gadget should be 1.0");
            if (cross_gadget(a, &bq) - raw_before).abs() > 1e-3 {
                moved = true;
            }
        }
        assert!(moved, "cross gadget never moved under gauge -> not actually gauge-dependent");
    }

    /// The canonical intertwiner exists for equivalent summands, exactly aligns
    /// `b` to `a` (`Uᵀ L_b U = L_a`), and therefore makes the canonical cross
    /// gadget equal the self gadget = 1.0. So equivalent-summand cross values carry
    /// NO invariant beyond "same adinkra".
    #[test]
    fn canonical_alignment_collapses_equivalent_cross_to_one() {
        let rep = block_diag(&cs_n4(), &cs_n4());
        let dec = decompose_rep(&rep).unwrap();
        let (a, b) = (&dec.summands[0], &dec.summands[1]);

        let u = intertwiner_orthogonal(&rep, a, b, 12345)
            .expect("equivalent summands must have an intertwiner");
        let resid = intertwiner_residual(a, b, &u);
        assert!(resid < 1e-9, "U does not align b to a (residual {resid:e})");

        // Canonical cross gadget = gadget(a, U-transported b) = self gadget = 1.0.
        let b_aligned = gauge_conjugate(b, &u);
        let canon = cross_gadget(a, &b_aligned);
        assert!((canon - 1.0).abs() < 1e-9, "canonical cross gadget {canon} != 1.0");
    }

    /// The canonical cross gadget is INVARIANT under independent random gauges on
    /// each summand (the property the raw cross gadget lacks): regauge a and b
    /// independently, recompute the intertwiner from the same generic commutant
    /// element, and the aligned value is unchanged (still 1.0). This is the proof
    /// that the orientation freedom is fully removed by the canonical alignment.
    #[test]
    fn canonical_cross_gadget_is_gauge_invariant() {
        let rep = block_diag(&cs_n4(), &cs_n4());
        let dec = decompose_rep(&rep).unwrap();
        let (a0, b0) = (dec.summands[0].clone(), dec.summands[1].clone());

        let base = {
            let u = intertwiner_orthogonal(&rep, &a0, &b0, 999).unwrap();
            cross_gadget(&a0, &gauge_conjugate(&b0, &u))
        };

        for (sa, sb) in [(1u64, 2u64), (3, 4), (5, 6)] {
            let a = gauge_conjugate(&a0, &random_orthogonal(a0.dmin, sa));
            let b = gauge_conjugate(&b0, &random_orthogonal(b0.dmin, sb));
            // Intertwiner recomputed in the regauged frames.
            let u = intertwiner_orthogonal(&rep, &a, &b, 999).unwrap();
            assert!(intertwiner_residual(&a, &b, &u) < 1e-8, "regauged U misaligned");
            let canon = cross_gadget(&a, &gauge_conjugate(&b, &u));
            assert!((canon - base).abs() < 1e-7, "canonical value moved under gauge: {canon} vs {base}");
        }
    }

    /// Non-isotypic rep (CS ⊕ CS ⊕ VS, d=12): the intertwiner is an exact
    /// equivalence oracle. Exactly the ONE equivalent pair (the two CS) aligns, and
    /// its canonical cross gadget is 1.0; the two CS-vs-VS pairs have no intertwiner.
    #[test]
    fn non_isotypic_equivalence_structure_is_recovered() {
        let rep = block_diag(&block_diag(&cs_n4(), &cs_n4()), &vs_n4()); // 2 CS + 1 VS
        let dec = decompose_rep(&rep).unwrap();
        assert_eq!(dec.summands.len(), 3);
        let mut equiv_pairs = 0;
        for i in 0..3 {
            for j in (i + 1)..3 {
                let (si, sj) = (&dec.summands[i], &dec.summands[j]);
                if let Some(u) = intertwiner_orthogonal(&rep, si, sj, 7) {
                    assert!(intertwiner_residual(si, sj, &u) < 1e-8, "misaligned intertwiner");
                    let canon = cross_gadget(si, &gauge_conjugate(sj, &u));
                    assert!((canon - 1.0).abs() < 1e-8, "equivalent canonical cross {canon} != 1.0");
                    equiv_pairs += 1;
                }
            }
        }
        assert_eq!(equiv_pairs, 1, "CS⊕CS⊕VS must have exactly one equivalent (CS,CS) pair");
    }

    /// Isotypic multiplicity 3 (CS ⊕ CS ⊕ CS): all three pairs align to 1.0, and
    /// the canonical value is independent of the seed selecting the generic
    /// commutant element (residual division-algebra unit does not move the value).
    #[test]
    fn isotypic_multiplicity_three_all_align_and_seed_independent() {
        let rep = block_diag(&block_diag(&cs_n4(), &cs_n4()), &cs_n4());
        let dec = decompose_rep(&rep).unwrap();
        assert_eq!(dec.summands.len(), 3);
        let mut pairs = 0;
        for i in 0..3 {
            for j in (i + 1)..3 {
                let (si, sj) = (&dec.summands[i], &dec.summands[j]);
                let u = intertwiner_orthogonal(&rep, si, sj, 42).expect("all CS pairs equivalent");
                let canon = cross_gadget(si, &gauge_conjugate(sj, &u));
                assert!((canon - 1.0).abs() < 1e-8, "canonical cross {canon} != 1.0");
                pairs += 1;
            }
        }
        assert_eq!(pairs, 3);
        // Seed-independence of the aligned value (residual D-unit invariance).
        let (s0, s1) = (&dec.summands[0], &dec.summands[1]);
        let ua = intertwiner_orthogonal(&rep, s0, s1, 42).unwrap();
        let ub = intertwiner_orthogonal(&rep, s0, s1, 999_999).unwrap();
        let va = cross_gadget(s0, &gauge_conjugate(s1, &ua));
        let vb = cross_gadget(s0, &gauge_conjugate(s1, &ub));
        assert!((va - vb).abs() < 1e-8, "aligned value depends on seed: {va} vs {vb}");
    }

    /// Inequivalent irreducibles (CS vs VS) carry NO intertwiner: the commutant
    /// block is ~0 and `intertwiner_orthogonal` returns None. Their genuine Gates
    /// gadget is the catalog value computed in the shared standard basis, not an
    /// orientation-dependent one.
    #[test]
    fn inequivalent_summands_have_no_intertwiner() {
        let rep = block_diag(&cs_n4(), &vs_n4()); // CS ⊕ VS: inequivalent
        let dec = decompose_rep(&rep).unwrap();
        assert_eq!(dec.summands.len(), 2);
        // Identify which summand is which by self-consistency is unnecessary; any
        // ordered pair of the two distinct-class summands has no intertwiner.
        let (a, b) = (&dec.summands[0], &dec.summands[1]);
        // Try several generic commutant elements; none should yield an intertwiner.
        let mut found = false;
        for seed in 1..6u64 {
            if intertwiner_orthogonal(&rep, a, b, seed).is_some() {
                found = true;
            }
        }
        assert!(!found, "inequivalent CS/VS summands should have no intertwiner");
    }
}
