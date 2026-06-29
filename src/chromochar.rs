//! Chromocharacter tensor and the proposed off-shell "closability" invariant
//! (Route A: the N=16 generalization of the 4D N=2 χ₀ adinkra parameter).
//!
//! Background. For a valise adinkra the four-color chromocharacter trace obeys
//! (Gates et al., arXiv:1712.07826 / 1508.07546):
//!
//!   Tr(L_I R_J L_K R_L) = 4[ (n_c+n_t)(δ_IJ δ_KL − δ_IK δ_JL + δ_IL δ_JK)
//!                            + χ₀ · ε_IJKL ]
//!
//! The symmetric (δ) part is trivial closure data; the single number χ₀ is the
//! totally antisymmetric (ε) coefficient. At N=4 there is one independent ε
//! component, so χ₀ is a scalar that equals +1 for cis-adinkras and −1 for
//! trans-adinkras (the two differ by flipping the dashing of one color). χ₀ = 0
//! ("color confinement") is the literature's NECESSARY condition for two
//! off-shell multiplets to fuse into a larger off-shell multiplet (arXiv:1405.0048).
//!
//! Generalization to N=16. Take the full rank-4 totally-antisymmetric part of the
//! chromocharacter, X_{IJKL} for I<J<K<L (C(16,4)=1820 components), and the
//! S_N-invariant scalar Q = Σ X². Because X is built from TRACES of products of
//! L/R, it is conjugation-invariant (basis-independent) — unlike the cross-summand
//! gadget (see [`crate::decompose`]). Q = 0 is the "color-confined" candidate
//! value; Q > 0 is χ₀-obstructed.
//!
//! HONEST SCOPE: this is a basis-independent CLASSIFICATION COORDINATE and a
//! CONJECTURAL obstruction screen, NOT a proof of off-shell liftability. Two
//! independent caveats, both confirmed by adversarial review:
//!   * Even at N=2 the χ₀=0 ⇒ closure link is only empirical ("evidence", a
//!     condition that "appears to be required" — arXiv:1405.0048), necessary but
//!     not proven sufficient; and Gates 2025 (arXiv:2503.13797) showed the naive
//!     off-shell route for 4D N=4 Maxwell is closed.
//!   * The N=16 generalization Q=0 is NOT established by the χ₀ literature as a
//!     necessary condition for N=16 liftability — it is a candidate/heuristic
//!     coordinate. Do not state it as a "necessary obstruction" without the
//!     "conjectural" qualifier.
//! The classic no-go itself is the Siegel-Rocek counting argument under standard
//! finite-component/linear-realization assumptions, not an absolute theorem in
//! every sense.

use crate::lr_matrix::AdinkraRep;
use crate::signed_perm::SignedPerm;

/// `Tr(L_I R_J L_K R_L)` computed exactly in integers, with R = L^{-1} = L^T.
///
/// `SignedPerm::compose` is a right action (`a.compose(b)` is the matrix product
/// `b*a`), so the matrix product `L_I R_J L_K R_L` is built inside-out:
///   R_L.compose(L_K).compose(R_J).compose(L_I) = L_I R_J L_K R_L.
fn tr_lrlr(rep: &AdinkraRep, i: usize, j: usize, k: usize, l: usize) -> i64 {
    let li = &rep.l_matrices[i];
    let rj = rep.l_matrices[j].inverse();
    let lk = &rep.l_matrices[k];
    let rl = rep.l_matrices[l].inverse();
    rl.compose(lk).compose(&rj).compose(li).trace()
}

/// All 24 permutations of `[0,1,2,3]` with parity (+1 even, −1 odd).
const ALL_S4: [([usize; 4], i64); 24] = [
    ([0, 1, 2, 3], 1), ([0, 1, 3, 2], -1), ([0, 2, 1, 3], -1), ([0, 2, 3, 1], 1),
    ([0, 3, 1, 2], 1), ([0, 3, 2, 1], -1), ([1, 0, 2, 3], -1), ([1, 0, 3, 2], 1),
    ([1, 2, 0, 3], 1), ([1, 2, 3, 0], -1), ([1, 3, 0, 2], -1), ([1, 3, 2, 0], 1),
    ([2, 0, 1, 3], 1), ([2, 0, 3, 1], -1), ([2, 1, 0, 3], -1), ([2, 1, 3, 0], 1),
    ([2, 3, 0, 1], 1), ([2, 3, 1, 0], -1), ([3, 0, 1, 2], -1), ([3, 0, 2, 1], 1),
    ([3, 1, 0, 2], 1), ([3, 1, 2, 0], -1), ([3, 2, 0, 1], -1), ([3, 2, 1, 0], 1),
];

/// Un-normalized totally-antisymmetric chromocharacter for the quadruple
/// (i,j,k,l): `Σ_σ sign(σ) Tr(L_{σi} R_{σj} L_{σk} R_{σl})`.
///
/// The genuine component is this value / 24; we return the integer sum to stay
/// exact. For the single N=4 quadruple this equals `96·χ₀`.
pub fn chromochar_antisym(rep: &AdinkraRep, i: usize, j: usize, k: usize, l: usize) -> i64 {
    let idx = [i, j, k, l];
    let mut acc = 0i64;
    for (p, sgn) in ALL_S4.iter() {
        acc += sgn * tr_lrlr(rep, idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]]);
    }
    acc
}

/// Raw-trace activity: `Σ_{i<j<k<l} Σ_{σ∈S₄} Tr(L_{σi} R_{σj} L_{σk} R_{σl})²`,
/// the UNSIGNED magnitude of the chromocharacter traces. Diagnostic for the
/// experiment: if `Q = Σ X² = 0` (antisymmetric part vanishes) while this is `> 0`,
/// the individual traces are nonzero and the antisymmetric channel genuinely
/// cancels; if this is also `0`, every four-distinct-colour trace is itself zero
/// at N=16 (each `Tr(L_I R_J L_K R_L)` vanishes — either the permutation part has
/// no fixed points, or the signed/Clifford structure cancels the signed diagonal;
/// the mechanism varies by code and is not asserted here).
/// Either way confirms a vanishing X is real structure, not a broken evaluator.
pub fn chromochar_trace_activity(rep: &AdinkraRep) -> i128 {
    let n = rep.n;
    let mut acc: i128 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    let idx = [i, j, k, l];
                    for (p, _sgn) in ALL_S4.iter() {
                        let t = tr_lrlr(rep, idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]]) as i128;
                        acc += t * t;
                    }
                }
            }
        }
    }
    acc
}

/// The N=4 χ₀ scalar (single quadruple {0,1,2,3}), normalized to ±1 for valise
/// cis/trans adinkras. `χ₀ = (Σ_σ sign(σ) Tr(...)) / 96`.
pub fn chi0_n4(rep: &AdinkraRep) -> f64 {
    assert_eq!(rep.n, 4, "chi0_n4 is the N=4 specialization");
    chromochar_antisym(rep, 0, 1, 2, 3) as f64 / 96.0
}

/// The proposed basis-independent closability scalar
///   Q[R] = Σ_{I<J<K<L} (chromochar_antisym/24)²   (S_N- and conjugation-invariant).
///
/// Returned as an exact rational-free integer: Σ (acc)² (i.e. 24²·Q). Q == 0 is
/// the "color-confined" candidate; Q > 0 is χ₀-obstructed. This is a CONJECTURAL
/// classification/obstruction screen, NOT a proven necessary condition for N=16
/// liftability and NOT a certificate.
pub fn closability_q_scaled(rep: &AdinkraRep) -> i128 {
    let n = rep.n;
    let mut q: i128 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    let x = chromochar_antisym(rep, i, j, k, l) as i128;
                    q += x * x;
                }
            }
        }
    }
    q
}

/// Convenience: is the representation χ₀-"color-confined" (Q == 0)?
pub fn is_color_confined(rep: &AdinkraRep) -> bool {
    closability_q_scaled(rep) == 0
}

/// One pass over the C(n,4) antisymmetric chromocharacter components, returning
/// `(support, q_scaled)` where `support = #{(I<J<K<L) : X_IJKL != 0}` and
/// `q_scaled = Σ X² = 24²·Q`. The SUPPORT is the cheap signal Q discards: Q squares
/// X (so it is sign-blind — the N=4 cis/trans pair share Q), whereas the support
/// count and the signed X-vector can distinguish reps that Q cannot. Both are exact
/// integers and conjugation-invariant (X is a sum of traces).
pub fn chromochar_support_and_q(rep: &AdinkraRep) -> (usize, i128) {
    let n = rep.n;
    let (mut support, mut q): (usize, i128) = (0, 0);
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    let x = chromochar_antisym(rep, i, j, k, l) as i128;
                    if x != 0 {
                        support += 1;
                    }
                    q += x * x;
                }
            }
        }
    }
    (support, q)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// N=4 Chiral (CS) and Vector (VS) supermultiplets: identical fixtures to
    /// holoraumy.rs / lr_matrix.rs. CS and VS differ only by the dashing of the
    /// color-3 / color-2 edges, i.e. they are cis vs trans, so their χ₀ must have
    /// OPPOSITE signs and each be ±1.
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

    #[test]
    fn chi0_is_plus_minus_one_for_cis_trans() {
        let chi_cs = chi0_n4(&cs_n4());
        let chi_vs = chi0_n4(&vs_n4());
        // Exact literature convention (arXiv:1405.0048: chiral = +1, vector = −1),
        // independently confirmed by hand (CS accumulator = +96, VS = −96).
        assert!((chi_cs - 1.0).abs() < 1e-9, "χ₀(CS) should be +1 (chiral), got {chi_cs}");
        assert!((chi_vs + 1.0).abs() < 1e-9, "χ₀(VS) should be −1 (vector), got {chi_vs}");
    }

    #[test]
    fn chromochar_antisym_is_totally_antisymmetric() {
        // Swapping two indices flips the sign.
        let r = cs_n4();
        let a = chromochar_antisym(&r, 0, 1, 2, 3);
        let b = chromochar_antisym(&r, 1, 0, 2, 3);
        assert_eq!(a, -b, "antisym under a single transposition");
        let c = chromochar_antisym(&r, 0, 1, 2, 3);
        let d = chromochar_antisym(&r, 0, 2, 1, 3);
        assert_eq!(c, -d, "antisym under a single transposition");
    }

    #[test]
    fn q_is_nonzero_for_n4_minimal() {
        // A single minimal N=4 valise multiplet has χ₀ = ±1, hence Q > 0
        // (it is NOT color-confined on its own; confinement is about fusing
        // a cis with a trans so the ε-pieces cancel).
        assert!(closability_q_scaled(&cs_n4()) > 0);
        assert!(!is_color_confined(&cs_n4()));
    }

    /// The evaluator produces NONZERO four-colour traces at N=4 (d=4): the
    /// raw-trace activity is positive, so a zero activity at N=16 reflects real
    /// structure (the four-colour traces genuinely vanish there), not a broken
    /// evaluator.
    #[test]
    fn trace_activity_nonzero_at_n4() {
        assert!(chromochar_trace_activity(&cs_n4()) > 0, "N=4 raw-trace activity must be > 0");
        assert!(chromochar_trace_activity(&vs_n4()) > 0);
    }

    /// The must-pass gate for the Q-scan experiment: Q is basis-invariant, i.e.
    /// conjugating every L_I by a fixed signed permutation P (L_I -> P L_I P⁻¹, a
    /// change of vertex basis) leaves Q and the support exactly unchanged. If this
    /// failed, Q would not be the invariant it is claimed to be.
    #[test]
    fn q_and_support_are_conjugation_invariant() {
        // compose is a right action: a.compose(b) = b·a, so P·L·P⁻¹ is built as
        // Pinv.compose(L).compose(P).
        let p = SignedPerm::from_parts(vec![2, 0, 3, 1], vec![1, -1, -1, 1]).unwrap();
        let pinv = p.inverse();
        for rep in [cs_n4(), vs_n4()] {
            let conj_l: Vec<SignedPerm> = rep
                .l_matrices
                .iter()
                .map(|l| pinv.compose(l).compose(&p))
                .collect();
            let conj = AdinkraRep { n: rep.n, d: rep.d, l_matrices: conj_l };
            assert_eq!(
                chromochar_support_and_q(&rep),
                chromochar_support_and_q(&conj),
                "Q/support not invariant under signed-perm conjugation"
            );
        }
    }
}
