#![allow(dead_code)] // research module: exercised by the `bbbm` subcommand and tests

//! BBBM 9-of-16 partial off-shell: the 1D N=9 valise target (GR(16,9)).
//!
//! Baulieu-Berkovits-Bossard-Martin (arXiv:0705.2002, "Ten-dimensional
//! super-Yang-Mills with nine off-shell supersymmetries") close 9 of the 16
//! supersymmetries of 10D, N=1 super-Yang-Mills off-shell by adding 7
//! auxiliary scalars. Reduced to the worldline, the off-shell field content is
//!
//!     16 gauge-invariant bosonic degrees = 10 gauge-potential components
//!                                           - 1 gauge redundancy
//!                                           + 7 auxiliary scalars
//!     16 fermions (gaugino)
//!
//! under N = 9 supersymmetries. The local reduction is a `(9,16,7)` hanging.
//! Formal node lowering gives a minimal N=9 valise, but requires `D^{-1}` and
//! does not identify zero modes; see `crate::bbbm_worldline`.
//!
//! WHAT THIS MODULE DOES. It builds the minimal N=9 valise from its maximal
//! doubly-even code (the [8,4] extended Hamming code padded with a trivial
//! ninth coordinate; k = 4, d = 2^(9-1-4) = 16) through the codebase's tested
//! `Chromotopology` -> `DashingEnumerator` -> `AdinkraRep` machinery, and
//! verifies the Garden algebra
//!
//!     L_I R_J + L_J R_I = 2 delta_IJ I_16
//!
//! exactly, for every dashing class. Independently, `crate::bbbm_worldline`
//! derives the formal node-lowered matrices from BBBM Eqs. (22)-(23) and matches
//! their chromotopology to this `[9,4]` scaffold after a color permutation.
//!
//! WHAT IT DOES NOT DO. The all-dashing gadget survey is not BBBM-specific.
//! The seven discarded tensor-charge transformations are not printed in the
//! paper, and this module does not reconstruct them.

use serde::Serialize;

use crate::chromotopology::Chromotopology;
use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;
use crate::lr_matrix::AdinkraRep;

/// The maximal doubly-even code of length 9: the [8,4] extended Hamming code
/// (the same generators used as the codebase's `hamming_8_4` fixture) sitting
/// in the first 8 of 9 coordinates. Bit 8 is unset in every generator, so the
/// ninth coordinate is a free (trivial) color. All codeword weights stay
/// divisible by 4, so the code remains doubly-even; k = 4, hence d = 16.
fn n9_minimal_code() -> DoublyEvenCode {
    DoublyEvenCode::new(9, vec![0b1110_0001, 0b1101_0010, 0b1011_0100, 0b0111_1000])
}

/// Public accessor for the minimal N=9 code, used by `crate::bbbm_holoraumy`.
pub fn n9_minimal_code_pub() -> DoublyEvenCode {
    n9_minimal_code()
}

#[derive(Debug, Serialize)]
pub struct BbbmValiseReport {
    /// Number of supersymmetries closing off-shell in the BBBM sector.
    pub supercharges_n: usize,
    /// Valise module dimension (bosons = fermions = d).
    pub module_dimension_d: usize,
    pub bosons: usize,
    pub fermions: usize,
    pub raw_gauge_potential_components: usize,
    pub gauge_invariant_gauge_bosons: usize,
    pub auxiliary_bosons: usize,
    pub code_length_n: usize,
    pub code_dimension_k: usize,
    pub dashing_classes_checked: usize,
    /// True iff L_I R_J + L_J R_I = 2 delta_IJ I_16 holds exactly for every
    /// dashing class of the code.
    pub garden_algebra_verified_all_dashings: bool,
    pub is_minimal_n9_valise: bool,
    pub interpretation: String,
    pub scope_caveat: String,
}

/// Build the minimal N=9 valise and verify the Garden algebra closes exactly
/// on every dashing class. Binary, byte-reproducible pass/fail.
pub fn run() -> BbbmValiseReport {
    let code = n9_minimal_code();
    let n = code.n;
    let k = code.k();
    let chromo = Chromotopology::from_code(&code);
    let d = chromo.d();

    let de = DashingEnumerator::new(&code);
    let color_perms: Vec<Vec<usize>> = (0..n).map(|c| chromo.color_perm(c).to_vec()).collect();
    let boson_reps = chromo.boson_reps();

    let mut all_ok = true;
    let num_classes = de.num_classes();
    for di in 0..num_classes {
        let signs = de.get_dashing_for_chromotopology(di, &boson_reps);
        let rep = AdinkraRep::from_parts(n, d, &color_perms, &signs);
        if !rep.verify_garden_algebra() {
            all_ok = false;
        }
    }

    BbbmValiseReport {
        supercharges_n: n,
        module_dimension_d: d,
        bosons: d,
        fermions: d,
        raw_gauge_potential_components: 10,
        gauge_invariant_gauge_bosons: 9,
        auxiliary_bosons: 7,
        code_length_n: n,
        code_dimension_k: k,
        dashing_classes_checked: num_classes,
        garden_algebra_verified_all_dashings: all_ok,
        is_minimal_n9_valise: n == 9 && d == 16,
        interpretation:
            "Generic minimal N=9 valise scaffold GR(16,9). Garden algebra \
             L_I R_J + L_J R_I = 2 delta_IJ I_16 is verified exactly for every \
             dashing class. The formally node-lowered BBBM reduction has this [9,4] \
             chromotopology after an explicit color permutation."
                .to_string(),
        scope_caveat:
            "The local BBBM object is the (9,16,7) hanging. Reaching the formal valise requires \
             D^{-1}, temporal gauge fixing, and deletion or separate treatment of zero modes. \
             The all-dashing survey is generic rather than a classification of BBBM theories."
                .to_string(),
    }
}

// ===========================================================================
// Route A independent cross-check (Agent A2).
//
// Everything below re-derives the N=9 valise holoraumy / gadget numbers and the
// [9,4] code characterization through a SEPARATE code path from the production
// `holoraumy`/`lr_matrix`/`canonical` modules, so the two derivations can be
// compared byte-for-byte. In particular:
//   * L_I and R_I are built as DENSE 16x16 signed-integer matrices (Vec<Vec<i64>>),
//     never as SignedPerm, and products/traces use naive triple loops.
//   * The self-gadget is assembled from dense Tr(Vtilde_IJ^2) sums with the
//     holoraumy normalization spelled out inline, not via holoraumy::gadget.
//   * The code invariants (weight enumerator, |Aut|, self-orthogonality,
//     decomposability, d_min, k-maximality) are recomputed with local loops.
// ===========================================================================

#[cfg(test)]
mod crosscheck {
    use crate::chromotopology::Chromotopology;
    use crate::code::DoublyEvenCode;
    use crate::dashing::DashingEnumerator;
    use std::collections::HashSet;

    /// The same [9,4] code the production module builds. Restated locally so this
    /// cross-check does not depend on `super::n9_minimal_code`.
    fn code_9_4() -> DoublyEvenCode {
        DoublyEvenCode::new(9, vec![0b1110_0001, 0b1101_0010, 0b1011_0100, 0b0111_1000])
    }

    // ---- dense 16x16 signed-matrix primitives (fully independent path) ------

    type Mat = Vec<Vec<i64>>;

    fn zeros(d: usize) -> Mat {
        vec![vec![0i64; d]; d]
    }

    fn ident(d: usize) -> Mat {
        let mut m = zeros(d);
        for i in 0..d {
            m[i][i] = 1;
        }
        m
    }

    /// Dense L-matrix for one color: row = fermion index, col = boson index.
    /// L[perm[j]][j] = dashing_sign(color,j). This is a genuine dense monomial
    /// matrix, built with no reference to SignedPerm.
    fn l_dense(d: usize, perm: &[usize], signs_for_color: &[i8]) -> Mat {
        let mut m = zeros(d);
        for j in 0..d {
            m[perm[j]][j] = signs_for_color[j] as i64;
        }
        m
    }

    fn transpose(a: &Mat) -> Mat {
        let d = a.len();
        let mut t = zeros(d);
        for i in 0..d {
            for j in 0..d {
                t[j][i] = a[i][j];
            }
        }
        t
    }

    fn matmul(a: &Mat, b: &Mat) -> Mat {
        let d = a.len();
        let mut c = zeros(d);
        for i in 0..d {
            for k in 0..d {
                if a[i][k] == 0 {
                    continue;
                }
                let aik = a[i][k];
                for j in 0..d {
                    c[i][j] += aik * b[k][j];
                }
            }
        }
        c
    }

    fn add(a: &Mat, b: &Mat) -> Mat {
        let d = a.len();
        let mut c = zeros(d);
        for i in 0..d {
            for j in 0..d {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        c
    }

    fn scaled_ident(d: usize, s: i64) -> Mat {
        let mut m = zeros(d);
        for i in 0..d {
            m[i][i] = s;
        }
        m
    }

    fn trace(a: &Mat) -> i64 {
        (0..a.len()).map(|i| a[i][i]).sum()
    }

    /// Build all N dense L-matrices for a given dashing class of the code.
    fn dense_l_matrices(class_index: usize) -> (usize, usize, Vec<Mat>) {
        let code = code_9_4();
        let n = code.n;
        let chromo = Chromotopology::from_code(&code);
        let d = chromo.d();
        let de = DashingEnumerator::new(&code);
        let boson_reps = chromo.boson_reps();
        // signs is a flat Vec<i8> of length n*d: signs[color*d + j].
        let signs = de.get_dashing_for_chromotopology(class_index, &boson_reps);
        let mut ls = Vec::with_capacity(n);
        for color in 0..n {
            let perm = chromo.color_perm(color);
            let sc: Vec<i8> = (0..d).map(|j| signs[color * d + j]).collect();
            ls.push(l_dense(d, perm, &sc));
        }
        (n, d, ls)
    }

    // -- 1. Independent Garden algebra + V_IJ^2 = -I --------------------------

    #[test]
    fn dense_garden_algebra_all_classes() {
        let code = code_9_4();
        let num_classes = 1usize << code.k();
        assert_eq!(num_classes, 16);

        for class in 0..num_classes {
            let (n, d, ls) = dense_l_matrices(class);
            assert_eq!(d, 16);
            let rs: Vec<Mat> = ls.iter().map(transpose).collect();

            // L_I R_J + L_J R_I = 2 delta_IJ I_d, via dense multiply.
            for i in 0..n {
                for j in i..n {
                    let sum = add(&matmul(&ls[i], &rs[j]), &matmul(&ls[j], &rs[i]));
                    let expected = if i == j {
                        scaled_ident(d, 2)
                    } else {
                        zeros(d)
                    };
                    assert_eq!(
                        sum, expected,
                        "class {class}: Garden algebra failed at (I,J)=({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn dense_v_squared_is_neg_identity() {
        // V_IJ = L_I R_J with I != J must square to -I (dense path).
        for class in 0..16usize {
            let (n, d, ls) = dense_l_matrices(class);
            let rs: Vec<Mat> = ls.iter().map(transpose).collect();
            let neg_i = scaled_ident(d, -1);
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let v = matmul(&ls[i], &rs[j]);
                    let v2 = matmul(&v, &v);
                    assert_eq!(v2, neg_i, "class {class}: V_({i},{j})^2 != -I");
                }
            }
        }
    }

    // -- 2. Independent self-gadget number ------------------------------------

    /// Self-gadget from dense fermionic holoraumy traces:
    ///   Vtilde_IJ = R_I L_J,  G[R,R] = -2/(N(N-1)dmin) * sum_{I>J} Tr(Vtilde_IJ^2).
    /// Since Vtilde_IJ^2 = -I for I != J, each term is -d, giving G = d/dmin.
    fn dense_self_gadget(class: usize) -> (f64, i64) {
        let (n, _d, ls) = dense_l_matrices(class);
        let rs: Vec<Mat> = ls.iter().map(transpose).collect();
        let dmin = 16usize; // d_min(9)
        let mut sum: i64 = 0;
        for i in 1..n {
            for j in 0..i {
                let vt = matmul(&rs[i], &ls[j]); // R_I L_J
                let vt2 = matmul(&vt, &vt);
                sum += trace(&vt2);
            }
        }
        let g = -2.0 * sum as f64 / (n * (n - 1) * dmin) as f64;
        (g, sum)
    }

    #[test]
    fn dense_self_gadget_is_one() {
        for class in 0..16usize {
            let (g, sum) = dense_self_gadget(class);
            // C(9,2)=36 pairs, each Tr(Vtilde^2) = -16, so sum = -576.
            assert_eq!(sum, -576, "class {class}: raw holoraumy trace sum");
            assert!(
                (g - 1.0).abs() < 1e-12,
                "class {class}: self-gadget should be 1.0, got {g}"
            );
        }
    }

    // -- 3. [9,4] code characterization (independent recompute) ---------------

    #[test]
    fn code_9_4_characterization() {
        let code = code_9_4();
        assert!(code.is_valid());
        assert_eq!(code.n, 9);
        assert_eq!(code.k(), 4);
        assert_eq!(code.num_codewords(), 16);

        // Weight enumerator recomputed with a local loop over codewords.
        let mut we = vec![0usize; code.n + 1];
        for cw in code.codewords() {
            we[cw.count_ones() as usize] += 1;
        }
        // [8,4,4] extended Hamming enumerator, unshifted by the trivial 9th bit:
        // 1 x wt0, 14 x wt4, 1 x wt8. Ninth coordinate is 0 in every generator.
        assert_eq!(we[0], 1);
        assert_eq!(we[4], 14);
        assert_eq!(we[8], 1);
        assert_eq!(we.iter().sum::<usize>(), 16);
        // No codeword touches coordinate 8 (the trivial color).
        assert!(
            code.codewords().iter().all(|&w| w & (1 << 8) == 0),
            "9th coordinate must be unused"
        );

        // d_min = 4 as a classical code (min nonzero weight).
        assert_eq!(code.min_distance(), 4);

        // Self-orthogonal (doubly-even implies self-orthogonal): all generator
        // pairwise intersections have even parity, recomputed locally.
        let gens = &code.generators;
        for a in 0..gens.len() {
            for b in a..gens.len() {
                assert_eq!(
                    (gens[a] & gens[b]).count_ones() % 2,
                    0,
                    "generators {a},{b} not orthogonal"
                );
            }
        }
        assert!(code.is_doubly_even());

        // Structure of the support: the 8 coordinates 0..7 that carry the
        // Hamming block form a single connected component, so on its SUPPORT
        // the code is indecomposable (code::is_indecomposable ignores the
        // support-free 9th coordinate entirely and reports true).
        assert!(
            code.is_indecomposable(),
            "code::is_indecomposable measures the support (Hamming block), which is connected"
        );
        // FINDING: neither codebase decomposability routine models the
        // "code (+) trivial coordinate" split. Both code::is_indecomposable and
        // canonical::is_decomposable connect only SUPPORT columns (those set in
        // some generator) and ignore the support-free 9th coordinate, so they
        // both report the code as indecomposable. Mathematically the [9,4] code
        // is nonetheless the direct sum Hamming[8,4] (+) [1,0]-trivial: the 9th
        // coordinate is a free color of the adinkra, uncoupled from the rest.
        assert!(
            !crate::canonical::is_decomposable(&code),
            "canonical::is_decomposable also ignores the support-free 9th coordinate"
        );

        // Relation to Hamming[8,4]: projecting away coordinate 8 gives exactly
        // the [8,4] extended Hamming code (same 16 codewords).
        let hamming = DoublyEvenCode::new(8, vec![0b1110_0001, 0b1101_0010, 0b1011_0100, 0b0111_1000]);
        let proj: HashSet<u32> = code
            .codewords()
            .iter()
            .map(|&w| w & 0xFF)
            .collect();
        let ham_set: HashSet<u32> = hamming.codewords().into_iter().collect();
        assert_eq!(proj, ham_set, "projection to first 8 coords must equal Hamming[8,4]");
    }

    /// Automorphism group of the [9,4] code, recomputed by brute force over all
    /// 9! column permutations that fix the codeword set. Independent of the
    /// canonical.rs implementation.
    #[test]
    fn code_9_4_automorphism_group() {
        let code = code_9_4();
        let n = code.n;
        let cw_set: HashSet<u32> = code.codewords().into_iter().collect();

        // Heap's algorithm, local copy.
        fn perms(n: usize) -> Vec<Vec<usize>> {
            let mut out = Vec::new();
            let mut p: Vec<usize> = (0..n).collect();
            let mut c = vec![0usize; n];
            out.push(p.clone());
            let mut i = 0;
            while i < n {
                if c[i] < i {
                    if i % 2 == 0 {
                        p.swap(0, i);
                    } else {
                        p.swap(c[i], i);
                    }
                    out.push(p.clone());
                    c[i] += 1;
                    i = 0;
                } else {
                    c[i] = 0;
                    i += 1;
                }
            }
            out
        }

        fn apply(word: u32, perm: &[usize]) -> u32 {
            let mut out = 0u32;
            for (dest, &src) in perm.iter().enumerate() {
                if word & (1 << src) != 0 {
                    out |= 1 << dest;
                }
            }
            out
        }

        let count = perms(n)
            .iter()
            .filter(|perm| cw_set.iter().all(|&w| cw_set.contains(&apply(w, perm))))
            .count();

        // |Aut([9,4])| = |Aut(Hamming[8,4])| * 1! for the free 9th coord factor.
        // Aut(extended Hamming [8,4]) = AGL(3,2), order 1344. The trivial 9th
        // coordinate can only map to itself (it is the unique coord in no
        // codeword's support at weight-1 sense), so total = 1344.
        assert_eq!(count, 1344, "|Aut([9,4])| should be 1344");
    }

    // -- 4. d_min(9)=16 and k=4 maximality ------------------------------------

    #[test]
    fn dmin_9_is_16_via_bott() {
        // d_min(N) = 2^(N-k_max-1); for N=9 the maximal doubly-even self-orthogonal
        // code has k_max = 4 (no doubly-even [9,5] exists), so d = 2^(9-4-1) = 16.
        // Cross-check against the module dimension produced by the chromotopology.
        let code = code_9_4();
        let chromo = Chromotopology::from_code(&code);
        assert_eq!(chromo.d(), 16);
        assert_eq!(1usize << (code.n - code.k() - 1), 16);
    }

    #[test]
    fn k4_is_maximal_no_doubly_even_9_5() {
        // A doubly-even binary code is self-orthogonal, so k <= floor(n/2) = 4
        // for n = 9. We verify the Singleton-style bound directly: attempt to
        // extend the [9,4] code by ANY vector keeping it doubly-even, and show
        // no independent extension exists.
        let code = code_9_4();
        let existing: HashSet<u32> = code.codewords().into_iter().collect();
        let mask = (1u32 << 9) - 1;

        let mut found_extension = false;
        'outer: for v in 1u32..=mask {
            if existing.contains(&v) {
                continue; // already in the code
            }
            // New span = existing (+) coset (existing XOR v). Every new codeword
            // must have weight divisible by 4 for the extension to stay doubly-even.
            for &c in &existing {
                if (c ^ v).count_ones() % 4 != 0 {
                    continue 'outer;
                }
            }
            found_extension = true;
            break;
        }
        assert!(
            !found_extension,
            "no doubly-even extension to k=5 may exist for n=9"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbbm_n9_valise_closes_garden_algebra() {
        let r = run();
        assert_eq!(r.supercharges_n, 9, "BBBM off-shell sector has 9 supercharges");
        assert_eq!(r.module_dimension_d, 16, "d_min(9) = 16");
        assert_eq!(
            r.gauge_invariant_gauge_bosons + r.auxiliary_bosons,
            r.bosons,
            "16 bosons = 9 gauge + 7 auxiliary"
        );
        assert_eq!(r.code_dimension_k, 4, "[9,4] maximal doubly-even code");
        assert!(r.dashing_classes_checked >= 1, "at least one dashing class");
        assert!(
            r.garden_algebra_verified_all_dashings,
            "GR(16,9) Garden algebra must close exactly for every dashing class"
        );
        assert!(r.is_minimal_n9_valise);
    }

    /// Route A: assert the [9,4] code invariants directly (k=4, weight
    /// enumerator {0:1, 4:14, 8:1}, and module dimension d = 16).
    #[test]
    fn n9_minimal_code_invariants() {
        let code = n9_minimal_code();
        assert!(code.is_valid(), "[9,4] code must be a valid doubly-even code");
        assert_eq!(code.n, 9);
        assert_eq!(code.k(), 4, "k must be 4 (maximal doubly-even for n=9)");
        assert_eq!(code.num_codewords(), 16);

        let we = code.weight_enumerator();
        assert_eq!(we.len(), 10, "weight enumerator has length n+1 = 10");
        assert_eq!(we[0], 1);
        assert_eq!(we[4], 14);
        assert_eq!(we[8], 1);
        assert_eq!(we.iter().sum::<usize>(), 16);
        // Only weights 0, 4, 8 occur (doubly-even).
        for (wt, &c) in we.iter().enumerate() {
            if wt % 4 != 0 {
                assert_eq!(c, 0, "weight {wt} must be empty");
            }
        }

        // Classical minimum distance of the code is 4; the valise module
        // dimension (the adinkra d, = 2^(n-k-1)) is 16 = d_min(9).
        assert_eq!(code.min_distance(), 4);
        let chromo = Chromotopology::from_code(&code);
        assert_eq!(chromo.d(), 16, "valise module dimension d = 2^(9-4-1) = 16");
    }
}
