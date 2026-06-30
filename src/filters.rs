use std::collections::HashSet;

use crate::code::DoublyEvenCode;
use crate::ranking::Ranking;
use crate::chromotopology::Chromotopology;

#[derive(Debug, Clone)]
pub struct WorldsheetResult {
    pub p: usize,
    pub q: usize,
    pub passes: bool,
    pub left_colors: Vec<usize>,
}

/// Weight-2 obstruction check for a (p,q) chirality split.
///
/// Tests whether e_I ^ e_J (weight 2) is a codeword for every left color I
/// and right color J. This is a necessary condition for the full Gates-Hubsch
/// worldsheet extension, but not sufficient on its own. For doubly-even codes
/// (minimum weight 4), weight-2 vectors are never codewords, so nontrivial
/// splits always fail this check.
pub fn worldsheet_weight2_obstruction(code: &DoublyEvenCode, left_colors: &[usize]) -> bool {
    let codeword_set: HashSet<u32> = code.codewords().into_iter().collect();
    let n = code.n;
    let right_colors: Vec<usize> = (0..n).filter(|c| !left_colors.contains(c)).collect();

    for &i in left_colors {
        for &j in &right_colors {
            let test = (1u32 << i) ^ (1u32 << j);
            if !codeword_set.contains(&test) {
                return false;
            }
        }
    }
    true
}

/// Run all (p, N-p) splits for p = 0..=N using the first p colors as left.
pub fn worldsheet_all_splits(code: &DoublyEvenCode) -> Vec<WorldsheetResult> {
    let n = code.n;
    let mut results = Vec::new();

    for p in 0..=n {
        let left: Vec<usize> = (0..p).collect();
        let passes = worldsheet_weight2_obstruction(code, &left);
        results.push(WorldsheetResult {
            p,
            q: n - p,
            passes,
            left_colors: left,
        });
    }

    results
}

/// Real Gates-Hubsch worldsheet (bow-tie / spin-sum) predicate.
///
/// Reference: S. J. Gates Jr. and T. Hubsch, "On Dimensional Extension of
/// Supersymmetry: From Worldlines to Worldsheets" (arXiv:1104.0722),
/// Theorem 2.1 / Theorem 2.2 and Corollary 2.2.
///
/// CHIRALITY ASSIGNMENT
/// `chirality[I]` in {+1, -1} is the spin (left/right mover) assigned to color
/// I: spin(D_I) = chirality[I]. A (p, q) split has p colors with +1 and q with
/// -1.
///
/// THE CONDITION
/// A 1D (worldline) Adinkra extends to a 2D (p, q) worldsheet iff, around EVERY
/// 2-colored quadrangle (4-cycle using exactly two colors I, J), the
/// height-weighted spin sum vanishes and there is no ambidextrous bow-tie.
///
/// Following the paper, the relevant local object is
///     sigma-hat_{I,B}^A = spin(D_I) * ( [F_B] - [F_A] )
/// where [F_X] is the height (rank) of vertex X and spin(D_I) = chirality[I].
/// Traversing a 2-colored 4-cycle
///     b1 --I--> f1 --J--> b2 --I--> f2 --J--> b1
/// the directed height steps are
///     chirality[I]*([f1]-[b1]) + chirality[J]*([b2]-[f1])
///   + chirality[I]*([f2]-[b2]) + chirality[J]*([b1]-[f2]).
/// The two color-I terms contribute chirality[I]*(([f1]-[b1]) + ([f2]-[b2]))
/// and the two color-J terms chirality[J]*(([b2]-[f1]) + ([b1]-[f2])).
/// This signed sum must be zero for the quadrangle to extend consistently
/// (Theorem 2.2). When the two colors have OPPOSITE chirality and a vertex is a
/// local max/min for both colors with conflicting flow, the quadrangle is an
/// "ambidextrous bow-tie" and the extension is obstructed (Theorem 2.1); that
/// shows up here as a nonvanishing spin sum.
///
/// VALISE REDUCTION (Corollary 2.2)
/// For a valise ranking every boson has height 0 and every fermion height 1, so
/// every up-step is +1 and every down-step is -1. A 2-colored 4-cycle alternates
/// boson, fermion, boson, fermion, so the unsigned height steps are +1, -1, +1,
/// -1. The spin-weighted sum becomes
///     (chirality[I] - chirality[J]) * ([f1] - [b1] + [f2] - [b2]) ... = 0 only
/// when chirality[I] == chirality[J]. Hence on a valise ranking the predicate
/// returns true ONLY for the trivial unidextrous splits where all participating
/// colors share one chirality (the (N,0) / (0,N) extension), and returns false
/// on every nontrivial (p, q) split. This matches Corollary 2.2 and reproduces
/// the existing `worldsheet_weight2_obstruction` behavior for doubly-even codes.
///
/// Returns true iff the chirality assignment yields a consistent (p, q)
/// worldsheet extension.
pub fn worldsheet_spin_sum(
    chromo: &Chromotopology,
    ranking: &Ranking,
    chirality: &[i8],
) -> bool {
    let n = chromo.n();
    debug_assert_eq!(chirality.len(), n, "chirality must have one entry per color");
    let d = chromo.d();

    // Enumerate every 2-colored quadrangle. For colors I != J, a 4-cycle is
    //   b1 --I--> f1 --J--> b2 --I--> f2 --J--> b1
    // We anchor on each boson rank b1 and walk the alternating I/J edges. We
    // need boson-rank and fermion-rank views of both colors.
    for i in 0..n {
        // color_perm(c)[boson_rank] = fermion_rank reached by color c.
        let i_fwd = chromo.color_perm(i);
        for j in (i + 1)..n {
            // J's fermion->boson inverse (fermion_rank -> boson_rank).
            let j_inv = chromo.color_perm_inverse(j);

            for b1_rank in 0..d {
                // b1 --I--> f1
                let f1_rank = i_fwd[b1_rank];
                // f1 --J--> b2 (use J's fermion->boson inverse)
                let b2_rank = j_inv[f1_rank];
                // b2 --I--> f2
                let f2_rank = i_fwd[b2_rank];
                // f2 --J--> back to a boson; this closes iff it returns to b1.
                let closing_rank = j_inv[f2_rank];
                if closing_rank != b1_rank {
                    // Not a 4-cycle on this anchor (the two colors give a longer
                    // cycle here); the genuine quadrangles are caught from their
                    // own anchors. Skip degenerate / non-closing walks.
                    continue;
                }
                if b2_rank == b1_rank {
                    // Degenerate (the two colors share the same edge); not a
                    // genuine 4-cycle.
                    continue;
                }

                // Translate ranks to shared vertex (coset) indices via the
                // public edge accessor so heights line up. f1 / f2 are the
                // fermions reached from b1 / b2 by color I.
                let (b1, f1) = chromo.edge_vertices(i, b1_rank);
                let (b2, f2) = chromo.edge_vertices(i, b2_rank);

                let h_b1 = ranking.height[b1] as i64;
                let h_f1 = ranking.height[f1] as i64;
                let h_b2 = ranking.height[b2] as i64;
                let h_f2 = ranking.height[f2] as i64;

                let ci = chirality[i] as i64;
                let cj = chirality[j] as i64;

                // Directed, spin-weighted height steps around the quadrangle:
                //   I:  b1 -> f1   and   b2 -> f2
                //   J:  f1 -> b2   and   f2 -> b1
                let spin_sum = ci * ((h_f1 - h_b1) + (h_f2 - h_b2))
                    + cj * ((h_b2 - h_f1) + (h_b1 - h_f2));

                if spin_sum != 0 {
                    return false;
                }
            }
        }
    }
    true
}

/// For a FIXED ranking, the most balanced `(p,q)` worldsheet chirality split that
/// the Gates-Hübsch spin-sum predicate admits (maximizing `min(p,q)`). EXACT, not
/// sampled: the spin-sum condition around the 2-coloured 4-cycles of a colour pair
/// `(I,J)` depends only on the product `s_I·s_J` (`A + (s_I s_J)·B = 0` per cycle),
/// so the `2^N` chirality search reduces to a signed 2-colouring over the N
/// colours via union-find, then a subset-sum over the resulting sign-components.
///
/// Returns `(N,0)` when only the trivial unidextrous extension is consistent
/// (e.g. every valise), and `(0,0)` when NO chirality assignment is consistent
/// (the ranking admits no worldsheet extension at all). A returned `(p,q)` with
/// `p,q>0` is a genuine nontrivial `(p,q)` worldsheet supersymmetry.
pub fn max_balanced_worldsheet(chromo: &Chromotopology, ranking: &Ranking) -> (usize, usize) {
    let n = chromo.n();
    let d = chromo.d();
    let h = &ranking.height;

    // Per colour pair: is s_I*s_J = +1 (plus) and/or -1 (minus) consistent across
    // ALL its 2-coloured 4-cycles? (A + t*B = 0 with t in {+1,-1}: +1 needs A+B=0,
    // -1 needs A-B=0.) `constrained` marks pairs that have at least one cycle.
    let mut plus_ok = vec![true; n * n];
    let mut minus_ok = vec![true; n * n];
    let mut constrained = vec![false; n * n];
    for i in 0..n {
        let i_fwd = chromo.color_perm(i);
        for j in (i + 1)..n {
            let j_inv = chromo.color_perm_inverse(j);
            let idx = i * n + j;
            for b1 in 0..d {
                let f1 = i_fwd[b1];
                let b2 = j_inv[f1];
                let f2 = i_fwd[b2];
                if j_inv[f2] != b1 || b2 == b1 {
                    continue;
                }
                let (vb1, vf1) = chromo.edge_vertices(i, b1);
                let (vb2, vf2) = chromo.edge_vertices(i, b2);
                let a = (h[vf1] - h[vb1]) as i64 + (h[vf2] - h[vb2]) as i64; // colour-I steps
                let b = (h[vb2] - h[vf1]) as i64 + (h[vb1] - h[vf2]) as i64; // colour-J steps
                constrained[idx] = true;
                if a + b != 0 {
                    plus_ok[idx] = false;
                }
                if a - b != 0 {
                    minus_ok[idx] = false;
                }
            }
        }
    }

    // Signed union-find over the N colours: s_i = rel[i] * s_root(i).
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rel = vec![1i64; n];
    fn find(x: usize, parent: &mut [usize], rel: &mut [i64]) -> (usize, i64) {
        if parent[x] == x {
            return (x, 1);
        }
        let (r, s) = find(parent[x], parent, rel);
        rel[x] *= s;
        parent[x] = r;
        (r, rel[x])
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let idx = i * n + j;
            if !constrained[idx] {
                continue;
            }
            let t = match (plus_ok[idx], minus_ok[idx]) {
                (true, true) => continue,    // free pair, no constraint
                (true, false) => 1i64,       // s_i*s_j = +1
                (false, true) => -1i64,      // s_i*s_j = -1
                (false, false) => return (0, 0), // pair impossible -> no extension
            };
            let (ri, si) = find(i, &mut parent, &mut rel);
            let (rj, sj) = find(j, &mut parent, &mut rel);
            if ri == rj {
                if si * sj != t {
                    return (0, 0); // sign conflict -> infeasible
                }
            } else {
                parent[rj] = ri;
                rel[rj] = t * si * sj; // s_rj = (t*si*sj) * s_ri
            }
        }
    }

    // Each sign-component can be globally flipped for free. With its root at +1 a
    // component contributes `c_plus` to p (members with rel +1); flipped it
    // contributes `size - c_plus`. Subset-sum the reachable values of p.
    use std::collections::HashMap;
    let (mut cplus, mut csize): (HashMap<usize, usize>, HashMap<usize, usize>) =
        (HashMap::new(), HashMap::new());
    for x in 0..n {
        let (r, s) = find(x, &mut parent, &mut rel);
        *csize.entry(r).or_insert(0) += 1;
        if s == 1 {
            *cplus.entry(r).or_insert(0) += 1;
        }
    }
    let mut reach = vec![false; n + 1];
    reach[0] = true;
    for (r, &sz) in &csize {
        let a = *cplus.get(r).unwrap_or(&0);
        let b = sz - a;
        let mut next = vec![false; n + 1];
        for p in 0..=n {
            if reach[p] {
                if p + a <= n {
                    next[p + a] = true;
                }
                if p + b <= n {
                    next[p + b] = true;
                }
            }
        }
        reach = next;
    }
    // Most balanced reachable split.
    let (mut best, mut best_min) = ((n, 0usize), usize::MAX);
    for p in 0..=n {
        if reach[p] {
            let q = n - p;
            let m = p.min(q);
            if best_min == usize::MAX || m > best_min {
                best_min = m;
                best = (p, q);
            }
        }
    }
    best
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DoublyEvenCode;
    use crate::chromotopology::Chromotopology;
    use crate::ranking::Ranking;

    /// Valise ranking from a chromotopology: bosons at height 0, fermions at 1.
    fn valise_ranking(chromo: &Chromotopology) -> Ranking {
        let height = (0..chromo.num_vertices())
            .map(|v| if chromo.is_boson_vertex(v) { 0 } else { 1 })
            .collect();
        Ranking { height }
    }

    /// THE NON-VACUITY EXPERIMENT: on a VALISE every nontrivial (p,q) split fails
    /// (Cor 2.2), but does some HUNG (non-valise) ranking unlock a nontrivial
    /// worldsheet extension? Enumerate ALL N=4 [4,1] rankings (complete ground
    /// truth) x all 2^4 chirality splits and count nontrivial passes. If this is
    /// zero the whole worldsheet oracle is vacuous; it must be > 0.
    #[test]
    fn some_hung_ranking_unlocks_nontrivial_worldsheet_n4() {
        let chromo = Chromotopology::from_code(&DoublyEvenCode::new(4, vec![0b1111]));
        let rankings = Ranking::enumerate(&chromo);
        let mut nontrivial_passes = 0usize;
        let mut best: Option<(usize, usize, Vec<i32>, Vec<i8>)> = None;
        for r in &rankings {
            for bits in 0u32..(1 << 4) {
                let chir: Vec<i8> = (0..4).map(|c| if bits & (1 << c) != 0 { 1 } else { -1 }).collect();
                let p = chir.iter().filter(|&&s| s == 1).count();
                let q = 4 - p;
                if p == 0 || q == 0 {
                    continue; // trivial unidextrous
                }
                if worldsheet_spin_sum(&chromo, r, &chir) {
                    nontrivial_passes += 1;
                    if best.is_none() {
                        best = Some((p, q, r.height.clone(), chir.clone()));
                    }
                }
            }
        }
        assert!(
            nontrivial_passes > 0,
            "no hung ranking admits a nontrivial worldsheet split -> oracle vacuous"
        );
        // The producer (usable at N=16) must also reach at least one such ranking.
        let produced = Ranking::raised_samples(&chromo, 4, 200);
        let producer_hits = produced.iter().any(|r| {
            (0u32..(1 << 4)).any(|bits| {
                let chir: Vec<i8> = (0..4).map(|c| if bits & (1 << c) != 0 { 1 } else { -1 }).collect();
                let p = chir.iter().filter(|&&s| s == 1).count();
                p != 0 && p != 4 && worldsheet_spin_sum(&chromo, r, &chir)
            })
        });
        eprintln!(
            "N=4 worldsheet: {} nontrivial (ranking,split) passes over {} rankings; example {:?}; producer reaches a pass: {}",
            nontrivial_passes, rankings.len(), best, producer_hits
        );
    }

    /// The fast union-find `max_balanced_worldsheet` must agree with brute force
    /// over all 2^N splits, on EVERY N=4 ranking (valise + all hung). This is the
    /// correctness gate for the catalog oracle.
    #[test]
    fn max_balanced_matches_bruteforce_n4() {
        let chromo = Chromotopology::from_code(&DoublyEvenCode::new(4, vec![0b1111]));
        for r in &Ranking::enumerate(&chromo) {
            // Brute force: best min(p,q) over all chirality assignments that pass.
            let mut bf_best_min: i64 = -1; // -1 = nothing passes
            for bits in 0u32..(1 << 4) {
                let chir: Vec<i8> = (0..4).map(|c| if bits & (1 << c) != 0 { 1 } else { -1 }).collect();
                if worldsheet_spin_sum(&chromo, r, &chir) {
                    let p = chir.iter().filter(|&&s| s == 1).count();
                    bf_best_min = bf_best_min.max(p.min(4 - p) as i64);
                }
            }
            let (p, q) = max_balanced_worldsheet(&chromo, r);
            let fast_min: i64 = if (p, q) == (0, 0) { -1 } else { p.min(q) as i64 };
            assert_eq!(
                fast_min, bf_best_min,
                "ranking {:?}: fast min {} != brute-force min {}",
                r.height, fast_min, bf_best_min
            );
        }
    }

    #[test]
    fn spin_sum_trivial_unidextrous_passes_valise() {
        // (N,0): all colors left-moving. Every 2-colored quadrangle has
        // chirality[I] == chirality[J], so the spin sum vanishes trivially.
        let chromo = Chromotopology::from_code(&DoublyEvenCode::new(4, vec![0b1111]));
        let ranking = valise_ranking(&chromo);
        let all_left = vec![1i8; 4];
        let all_right = vec![-1i8; 4];
        assert!(worldsheet_spin_sum(&chromo, &ranking, &all_left));
        assert!(worldsheet_spin_sum(&chromo, &ranking, &all_right));
    }

    #[test]
    fn spin_sum_nontrivial_fails_valise_n4() {
        // Any nontrivial (p, q) split must fail on a valise ranking
        // (Corollary 2.2), matching the existing worldsheet_weight2_obstruction
        // behavior for doubly-even codes.
        let chromo = Chromotopology::from_code(&DoublyEvenCode::new(4, vec![0b1111]));
        let ranking = valise_ranking(&chromo);
        // (1,3)
        assert!(!worldsheet_spin_sum(&chromo, &ranking, &[1, -1, -1, -1]));
        // (2,2)
        assert!(!worldsheet_spin_sum(&chromo, &ranking, &[1, 1, -1, -1]));
        // (3,1)
        assert!(!worldsheet_spin_sum(&chromo, &ranking, &[1, 1, 1, -1]));
    }

    #[test]
    fn spin_sum_matches_weight2_obstruction_valise_n4() {
        // Free regression: on a valise ranking, worldsheet_spin_sum must agree
        // with worldsheet_weight2_obstruction on the "first p colors left"
        // assignment for every p (both reject all nontrivial splits, accept the
        // trivial ones).
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        let chromo = Chromotopology::from_code(&code);
        let ranking = valise_ranking(&chromo);
        for p in 0..=4usize {
            let left: Vec<usize> = (0..p).collect();
            let chirality: Vec<i8> = (0..4)
                .map(|c| if left.contains(&c) { 1 } else { -1 })
                .collect();
            let spin = worldsheet_spin_sum(&chromo, &ranking, &chirality);
            let weight2 = worldsheet_weight2_obstruction(&code, &left);
            assert_eq!(
                spin, weight2,
                "p={}: spin_sum ({}) disagrees with weight2 obstruction ({})",
                p, spin, weight2
            );
        }
    }

    #[test]
    fn trivial_splits_pass() {
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        // (0,4): no left colors, no cross-pairs to check
        assert!(worldsheet_weight2_obstruction(&code, &[]));
        // (4,0): no right colors, no cross-pairs to check
        assert!(worldsheet_weight2_obstruction(&code, &[0, 1, 2, 3]));
    }

    #[test]
    fn nontrivial_splits_fail_n4() {
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        assert!(!worldsheet_weight2_obstruction(&code, &[0]));
        assert!(!worldsheet_weight2_obstruction(&code, &[0, 1]));
        assert!(!worldsheet_weight2_obstruction(&code, &[0, 1, 2]));
    }

    #[test]
    fn weight2_never_doubly_even() {
        // For any doubly-even code, e_I ^ e_J has weight 2, never a codeword
        let code = DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        );
        let codewords: HashSet<u32> = code.codewords().into_iter().collect();
        for i in 0..8 {
            for j in (i + 1)..8 {
                let test = (1u32 << i) ^ (1u32 << j);
                assert!(!codewords.contains(&test));
            }
        }
    }

    #[test]
    fn all_splits_doubly_even_n4() {
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        let results = worldsheet_all_splits(&code);
        assert_eq!(results.len(), 5); // p = 0,1,2,3,4
        assert!(results[0].passes); // (0,4)
        assert!(!results[1].passes); // (1,3)
        assert!(!results[2].passes); // (2,2)
        assert!(!results[3].passes); // (3,1)
        assert!(results[4].passes); // (4,0)
    }

    #[test]
    fn all_splits_hamming_8_4() {
        let code = DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        );
        let results = worldsheet_all_splits(&code);
        for r in &results {
            if r.p == 0 || r.q == 0 {
                assert!(r.passes, "({},{}) should pass", r.p, r.q);
            } else {
                assert!(!r.passes, "({},{}) should fail for doubly-even", r.p, r.q);
            }
        }
    }
}
