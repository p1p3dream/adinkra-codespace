#![allow(dead_code)]
//! ADVERSARIAL, INDEPENDENT verification of the G-matrix computation of
//! Gates and Lee, arXiv:2408.09342 (Eq 8.2).
//!
//! The G-matrix is a {-1,0,1} matrix (several nonzeros per row, NOT a signed
//! permutation) satisfying
//!
//!     G^2 = A,   with   A = (L_1 + L_2 + L_3 + L_4) * L_1^{-1}.
//!
//! The paper reports 12 such G per minimal 4D N=1 supermultiplet (chiral CM,
//! vector VM, tensor TM). They enumerate the naive way,
//! `Select[Tuples[{-1,0,1},{n,n}], MatrixPower[#,2] == A]`, i.e. 3^(n^2)
//! matrices, which is why the 12x12 Complex Linear Supermultiplet (CLS) case
//! "exceeds the calculation RAM amount usage" and was reported unavailable.
//!
//! This module is deliberately a SECOND, INDEPENDENT implementation. It shares
//! nothing with the primary G-matrix solver except the arithmetic helpers
//! (`super::imm`, `super::dense`, ...). Every count it reports is recomputed
//! here from first principles. The design goal is to catch a wrong primary
//! solver, so nothing is tuned to match an expected answer.
//!
//! Independent algorithm (`g_matrices_alt`):
//!
//!   1. If A is block-diagonal (a permutation-free block split into square
//!      diagonal blocks with all off-diagonal blocks exactly zero), solve each
//!      diagonal block INDEPENDENTLY for its {-1,0,1} roots, then combine the
//!      per-block roots by Cartesian product into block-diagonal G. This is the
//!      "solve per block then combine" route and is what makes the 12x12 CLS
//!      case tractable (three 4x4 blocks, 12^3 combinations, instead of 3^144).
//!
//!      This route enumerates exactly the roots that are themselves
//!      block-diagonal (per-block root chosen independently, combined by
//!      Cartesian product). Every returned G is checked to square to A. Whether
//!      NON-block-diagonal {-1,0,1} roots also exist for a block-diagonal A is a
//!      separate and genuinely hard question: the naive 3^(n^2) enumeration is
//!      intractable for n >= 6 (exactly the bottleneck the paper reports), and
//!      interval constraint propagation does
//!      not tame it because the quadratic G^2 = A constraints do not prune until
//!      the matrix is nearly complete. This module therefore does NOT claim to
//!      have exhausted the cross-block space at n = 12; it reports the count of
//!      block-diagonal roots (12^3 = 1728 per CLS side), the complete
//!      block-diagonal family and a natural independent cross-check on a primary
//!      solver. (Cross-block solutions also exist; gmatrix.rs constructs one.)
//!
//!   2. If A is not block-diagonal (the dense minimal n=4 A), fall through to a
//!      complete search that assigns G COLUMN BY COLUMN with EXACT reachability
//!      pruning: after columns 0..t are fixed, every equation (i,j) with j<=t is
//!      split into a known part (over assigned columns) and a residual whose
//!      reach is exactly sum over unassigned columns k of |G[k][j]|; a partial
//!      assignment that puts A[i][j] outside [partial-reach, partial+reach] is
//!      pruned. This is a different search order and pruning scheme from a plain
//!      entrywise 3^(n^2) backtracker.
//!
//! Completeness is PROVEN independently at n=4 in the tests by comparing
//! `g_matrices_alt` against a full 3^16 brute-force enumeration.

use super::IntMat;

/// True iff G*G == A (exact integer arithmetic).
pub fn squares_to(g: &IntMat, a: &IntMat) -> bool {
    super::imm(g, g) == *a
}

/// Every G in `gs` squares to A.
pub fn verify_all_square(a: &IntMat, gs: &[IntMat]) -> bool {
    gs.iter().all(|g| squares_to(g, a))
}

/// Build A = (L_1 + L_2 + L_3 + L_4) * L_1^{-1} for a 4-color L set.
pub fn a_matrix(ls: &[crate::signed_perm::SignedPerm]) -> IntMat {
    let dense_ls: Vec<IntMat> = ls.iter().map(super::dense).collect();
    let sum = super::dsum(&dense_ls);
    let l1_inv = super::dense(&ls[0].inverse());
    super::imm(&sum, &l1_inv)
}

// ---------------------------------------------------------------------------
// Block-structure detection.
// ---------------------------------------------------------------------------

/// Detect a block-diagonal split of `a` into contiguous square diagonal blocks
/// with all off-diagonal blocks exactly zero. Returns the block boundaries as a
/// list of (start, len). Always returns at least the trivial single block.
///
/// The split is the finest one consistent with the nonzero pattern: it grows a
/// block only as far as a nonzero entry forces it to. This is a union-find style
/// pass over rows/cols coupled by nonzeros, restricted to CONTIGUOUS blocks
/// (which is all we need; the CLS A is contiguously block-diagonal).
pub fn block_split(a: &IntMat) -> Vec<(usize, usize)> {
    let n = a.len();
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < n {
        // Extend `end` until no nonzero couples a row < end to a col >= end
        // (and symmetrically), i.e. the [start, end) square has no nonzero
        // leaving it on either side.
        let mut end = start + 1;
        loop {
            let mut grew = false;
            for i in start..end {
                for j in 0..n {
                    if a[i][j] != 0 && j >= end {
                        end = j + 1;
                        grew = true;
                    }
                    if a[j][i] != 0 && j >= end {
                        end = j + 1;
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
            if end >= n {
                break;
            }
        }
        blocks.push((start, end - start));
        start = end;
    }
    blocks
}

/// Extract the square submatrix a[r..r+len][r..r+len].
fn subblock(a: &IntMat, r: usize, len: usize) -> IntMat {
    (0..len)
        .map(|i| (0..len).map(|j| a[r + i][r + j]).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Complete column-DFS root solver with exact reachability pruning.
// ---------------------------------------------------------------------------

/// All {-1,0,1} matrices G (size n x n) with G*G == a, found by a complete
/// column-by-column DFS with exact reachability pruning. Different in kind from
/// a plain entrywise 3^(n^2) backtracker: it fixes whole columns and prunes on
/// the exact residual reach of every equation touching an assigned column.
///
/// Practical for small dense n (n = 4). For larger n use `g_matrices_alt`,
/// which routes block-diagonal A through per-block solves.
pub fn g_matrices_columns(a: &IntMat) -> Vec<IntMat> {
    let n = a.len();
    let mut g = vec![vec![0i32; n]; n];
    let mut out = Vec::new();
    col_dfs(0, n, a, &mut g, &mut out);
    out
}

fn col_feasible(n: usize, a: &IntMat, g: &[Vec<i32>], set: usize) -> bool {
    // Columns 0..set are fully assigned. For each such column jj and each row i,
    // eqn (i,jj) = sum_k g[i][k]*g[k][jj] must be reachable: the part over
    // assigned columns (k < set) is exact; the part over unassigned columns
    // (k >= set) has |g[i][k]| free in {0,1}, so its reach is
    // sum_{k>=set} |g[k][jj]|.
    for jj in 0..set {
        for i in 0..n {
            let mut partial = 0i32;
            let mut reach = 0i32;
            for k in 0..n {
                let gkj = g[k][jj];
                if gkj == 0 {
                    continue;
                }
                if k < set {
                    partial += g[i][k] * gkj;
                } else {
                    reach += gkj.abs();
                }
            }
            if !(partial - reach <= a[i][jj] && a[i][jj] <= partial + reach) {
                return false;
            }
        }
    }
    true
}

fn col_dfs(j: usize, n: usize, a: &IntMat, g: &mut Vec<Vec<i32>>, out: &mut Vec<IntMat>) {
    if j == n {
        // Full matrix assigned; final exact check (belt and suspenders).
        if super::imm(g, g) == *a {
            out.push(g.clone());
        }
        return;
    }
    let total = 3usize
        .checked_pow(n as u32)
        .expect("column search space 3^n overflowed usize");
    for mut code in 0..total {
        for k in 0..n {
            g[k][j] = (code % 3) as i32 - 1;
            code /= 3;
        }
        if col_feasible(n, a, g, j + 1) {
            col_dfs(j + 1, n, a, g, out);
        }
    }
    for k in 0..n {
        g[k][j] = 0;
    }
}

// ---------------------------------------------------------------------------
// The independent solver: block route + fallback.
// ---------------------------------------------------------------------------

/// Independent complete solver for {-1,0,1} G with G*G == a.
///
/// If `a` is block-diagonal into contiguous square blocks (all off-diagonal
/// blocks zero) and there is more than one block, solve each diagonal block for
/// its {-1,0,1} roots and Cartesian-product them into block-diagonal G. If `a`
/// is a single block, run the complete column-DFS directly.
///
/// The block route returns exactly the block-diagonal roots (all of them, and
/// only block-diagonal ones). This is the complete block-diagonal family and the
/// independent cross-check on a primary solver; see the module docs for why the
/// cross-block space is not exhausted at n = 12.
pub fn g_matrices_alt(a: &IntMat) -> Vec<IntMat> {
    let n = a.len();
    let blocks = block_split(a);
    if blocks.len() <= 1 {
        return g_matrices_columns(a);
    }

    // Solve each block independently.
    let per_block: Vec<Vec<IntMat>> = blocks
        .iter()
        .map(|&(r, len)| g_matrices_columns(&subblock(a, r, len)))
        .collect();

    // Cartesian product -> block-diagonal G.
    let mut out: Vec<IntMat> = vec![vec![vec![0i32; n]; n]];
    for (bi, &(r, len)) in blocks.iter().enumerate() {
        let mut next = Vec::with_capacity(out.len() * per_block[bi].len());
        for base in &out {
            for choice in &per_block[bi] {
                let mut g = base.clone();
                for i in 0..len {
                    for j in 0..len {
                        g[r + i][r + j] = choice[i][j];
                    }
                }
                next.push(g);
            }
        }
        out = next;
    }
    // Final exact check on each (guards against any block-combination mistake).
    out.retain(|g| super::imm(g, g) == *a);
    out
}

// ---------------------------------------------------------------------------
// Reporting.
// ---------------------------------------------------------------------------

/// Human-readable summary of the independent verification.
pub fn report() -> String {
    use super::{cm_l_matrices, sp};

    let a_cm = a_matrix(&cm_l_matrices());
    let cm = g_matrices_alt(&a_cm).len();

    // VM / TM L-matrices, transcribed here independently.
    let vm_l = vec![
        sp(&[2, -4, 1, -3]),
        sp(&[1, 3, -2, -4]),
        sp(&[4, 2, 3, 1]),
        sp(&[3, -1, -4, 2]),
    ];
    let tm_l = vec![
        sp(&[1, -3, -4, -2]),
        sp(&[2, 4, -3, 1]),
        sp(&[3, 1, 2, -4]),
        sp(&[4, -2, 1, 3]),
    ];
    let vm = g_matrices_alt(&a_matrix(&vm_l)).len();
    let tm = g_matrices_alt(&a_matrix(&tm_l)).len();

    let a_l = a_matrix(&super::cls::cls_l_matrices());
    let a_r = a_matrix(&super::cls::cls_r_matrices());
    let cls_l = g_matrices_alt(&a_l).len();
    let cls_r = g_matrices_alt(&a_r).len();

    format!(
        "gmatrix_verify (independent, adversarial): \
         minimal G-counts CM={} VM={} TM={} (paper says 12 each); \
         CLS 12x12 A_(L) blocks={:?} -> G-count={}, A_(R) G-count={} \
         (12^3 = 1728 per side, from three independent 4x4 block solves; \
         the case the paper reported unavailable). All returned G square to A.",
        cm,
        vm,
        tm,
        block_split(&a_l)
            .iter()
            .map(|&(_, l)| l)
            .collect::<Vec<_>>(),
        cls_l,
        cls_r,
    )
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_perm::SignedPerm;

    fn vm_l() -> Vec<SignedPerm> {
        vec![
            super::super::sp(&[2, -4, 1, -3]),
            super::super::sp(&[1, 3, -2, -4]),
            super::super::sp(&[4, 2, 3, 1]),
            super::super::sp(&[3, -1, -4, 2]),
        ]
    }
    fn tm_l() -> Vec<SignedPerm> {
        vec![
            super::super::sp(&[1, -3, -4, -2]),
            super::super::sp(&[2, 4, -3, 1]),
            super::super::sp(&[3, 1, 2, -4]),
            super::super::sp(&[4, -2, 1, 3]),
        ]
    }

    /// Full 3^16 brute-force enumeration of {-1,0,1} 4x4 roots of A. This is the
    /// completely independent ground truth for n=4.
    fn brute_force_n4(a: &IntMat) -> Vec<IntMat> {
        let n = 4usize;
        let total = 3u64.pow((n * n) as u32); // 3^16 = 43_046_721
        let mut out = Vec::new();
        let mut g = vec![vec![0i32; n]; n];
        for mut code in 0..total {
            for i in 0..n {
                for j in 0..n {
                    g[i][j] = (code % 3) as i32 - 1;
                    code /= 3;
                }
            }
            if super::super::imm(&g, &g) == *a {
                out.push(g.clone());
            }
        }
        out
    }

    fn sorted(mut v: Vec<IntMat>) -> Vec<IntMat> {
        v.sort();
        v.dedup();
        v
    }

    #[test]
    fn cm_alt_finds_exactly_12_and_includes_paper_chiral_g() {
        let a = a_matrix(&super::super::cm_l_matrices());
        let gs = g_matrices_alt(&a);
        assert_eq!(gs.len(), 12, "CM must have exactly 12 G-matrices");
        assert!(verify_all_square(&a, &gs), "every CM G must square to A");

        // The paper's chiral G-matrix.
        let chiral: IntMat = vec![
            vec![1, 0, 1, 0],
            vec![0, -1, 0, 1],
            vec![0, -1, 0, -1],
            vec![1, 0, -1, 0],
        ];
        assert!(
            squares_to(&chiral, &a),
            "paper's chiral G must satisfy G^2 = A(CM)"
        );
        assert!(
            gs.iter().any(|g| *g == chiral),
            "g_matrices_alt must contain the paper's chiral G"
        );
    }

    #[test]
    fn vm_and_tm_alt_find_exactly_12_each() {
        for (name, ls) in [("VM", vm_l()), ("TM", tm_l())] {
            let a = a_matrix(&ls);
            let gs = g_matrices_alt(&a);
            assert_eq!(gs.len(), 12, "{name} must have exactly 12 G-matrices");
            assert!(
                verify_all_square(&a, &gs),
                "every {name} G must square to A"
            );
        }
    }

    /// Independent completeness proof at n=4: the alt solver equals the full
    /// 3^16 brute-force set, exactly, as a set.
    #[test]
    fn alt_equals_brute_force_at_n4() {
        let a = a_matrix(&super::super::cm_l_matrices());
        let alt = sorted(g_matrices_alt(&a));
        let brute = sorted(brute_force_n4(&a));
        assert_eq!(
            alt.len(),
            brute.len(),
            "alt count {} != brute-force count {}",
            alt.len(),
            brute.len()
        );
        assert_eq!(
            alt, brute,
            "g_matrices_alt set must equal the 3^16 brute-force set exactly"
        );
        // Sanity: also true for VM and TM.
        for ls in [vm_l(), tm_l()] {
            let a = a_matrix(&ls);
            assert_eq!(sorted(g_matrices_alt(&a)), sorted(brute_force_n4(&a)));
        }
    }

    /// The block-splitter must find the three 4x4 blocks of the CLS A matrices.
    #[test]
    fn cls_a_is_three_4x4_blocks() {
        let a_l = a_matrix(&super::super::cls::cls_l_matrices());
        let a_r = a_matrix(&super::super::cls::cls_r_matrices());
        assert_eq!(block_split(&a_l), vec![(0, 4), (4, 4), (8, 4)]);
        assert_eq!(block_split(&a_r), vec![(0, 4), (4, 4), (8, 4)]);
        // A satisfies the minimal polynomial A^2 = 2A - 4I (eigenvalues
        // 2 e^{+- i pi/3}); recorded as an independent structural check.
        let n = 12;
        let two_a: IntMat = a_l
            .iter()
            .map(|row| row.iter().map(|&x| 2 * x).collect())
            .collect();
        let a2 = super::super::imm(&a_l, &a_l);
        for i in 0..n {
            for j in 0..n {
                let four_i = if i == j { 4 } else { 0 };
                assert_eq!(
                    a2[i][j],
                    two_a[i][j] - four_i,
                    "A^2 = 2A - 4I must hold at ({i},{j})"
                );
            }
        }
    }

    /// The block route combines per-block roots correctly. Verified on a small,
    /// FULLY tractable block-diagonal matrix: A = diag(B, B) where B = P^2 for a
    /// 3x3 permutation P. Here the block route (per-block solve + Cartesian
    /// product) must reproduce a brute-force 3^36 ... which is too large, so we
    /// instead check the combination logic exactly: every block-diagonal
    /// combination of per-block roots squares to A, the count is the product of
    /// per-block counts, and there are no duplicates.
    ///
    /// This exercises the same combine path used for CLS. Whether NON-block
    /// diagonal {-1,0,1} roots exist for a block-diagonal A is a separate,
    /// genuinely hard question (the naive 3^(n^2) enumeration is intractable for
    /// n >= 6, which is exactly the bottleneck the paper reports); this module
    /// does not claim to have exhausted the cross-block space at n = 12. It
    /// enumerates all BLOCK-DIAGONAL roots and verifies each squares to A.
    #[test]
    fn block_route_combination_is_sound() {
        // B = P^2 with P the 3-cycle permutation; B is a permutation matrix.
        let p: IntMat = vec![vec![0, 1, 0], vec![0, 0, 1], vec![1, 0, 0]];
        let b = super::super::imm(&p, &p);
        let mut a = vec![vec![0i32; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = b[i][j];
                a[i + 3][j + 3] = b[i][j];
            }
        }
        assert_eq!(block_split(&a), vec![(0, 3), (3, 3)]);

        let per_block = g_matrices_columns(&b);
        assert!(
            !per_block.is_empty(),
            "the 3x3 block must have at least one root"
        );
        // The block route output.
        let gs = g_matrices_alt(&a);
        // Count is the product of per-block counts.
        assert_eq!(
            gs.len(),
            per_block.len() * per_block.len(),
            "block route count must be the product of per-block root counts"
        );
        // Every returned G squares to A, is block-diagonal, and unique.
        assert!(
            verify_all_square(&a, &gs),
            "every combined G must square to A"
        );
        for g in &gs {
            for i in 0..6 {
                for j in 0..6 {
                    if i / 3 != j / 3 {
                        assert_eq!(g[i][j], 0, "combined G must be block-diagonal");
                    }
                }
            }
        }
        let uniq = sorted(gs.clone());
        assert_eq!(uniq.len(), gs.len(), "block route must not emit duplicates");
    }

    /// CLS headline case. `g_matrices_alt` on the 12x12 A_(L) and A_(R) returns
    /// a count equal to the product of the per-block root counts (a
    /// Cartesian-assembly consistency check using the same per-block solver, not
    /// an independent recount), and every returned G squares to A.
    #[test]
    fn cls_alt_counts_match_direct_recompute() {
        let a_l = a_matrix(&super::super::cls::cls_l_matrices());
        let a_r = a_matrix(&super::super::cls::cls_r_matrices());

        for (name, a) in [("A_(L)", &a_l), ("A_(R)", &a_r)] {
            let gs = g_matrices_alt(a);
            assert!(
                verify_all_square(a, &gs),
                "every CLS {name} G must square to A"
            );

            // Consistency of the Cartesian assembly: the total must equal the
            // product of the per-block counts. This uses the same per-block solver,
            // so it checks the assembly, not an independent root count.
            let blocks = block_split(a);
            let per_block_counts: Vec<usize> = blocks
                .iter()
                .map(|&(r, len)| g_matrices_columns(&subblock(a, r, len)).len())
                .collect();
            let product: usize = per_block_counts.iter().product();
            assert_eq!(
                gs.len(),
                product,
                "CLS {name}: g_matrices_alt count {} must equal the per-block \
                 product recompute {}",
                gs.len(),
                product
            );
            // For the record, the expected structure is three 12-root blocks.
            assert_eq!(
                gs.len(),
                1728,
                "CLS {name}: expected 12^3 = 1728 block-diagonal G-matrices, got {}",
                gs.len()
            );
            println!(
                "CLS {name}: independent G-count = {} (= {:?} per-block product)",
                gs.len(),
                per_block_counts
            );
        }
    }

    /// If a results artifact exists in the worktree, assert our independently
    /// computed CLS counts (and matrices) match it exactly. If it does not
    /// exist, just report our counts (do not fail on absence).
    #[test]
    fn cls_matches_artifact_if_present() {
        let path = "results/four_color_cls_gmatrix.json";
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => {
                let a_l = a_matrix(&super::super::cls::cls_l_matrices());
                let a_r = a_matrix(&super::super::cls::cls_r_matrices());
                println!(
                    "no artifact at {path}; independent CLS counts: A_(L)={}, A_(R)={}",
                    g_matrices_alt(&a_l).len(),
                    g_matrices_alt(&a_r).len()
                );
                return;
            }
        };
        // Artifact present: parse it and compare as DATA, not substrings.
        let v: serde_json::Value = serde_json::from_str(&raw).expect("artifact is valid JSON");
        let a_l = a_matrix(&super::super::cls::cls_l_matrices());
        let a_r = a_matrix(&super::super::cls::cls_r_matrices());
        let my_l = g_matrices_alt(&a_l).len();
        let my_r = g_matrices_alt(&a_r).len();

        // 1. Counts compared as integers (a file containing only "1728" fails here).
        let art_l = v["block_diagonal_count_L"]
            .as_u64()
            .expect("block_diagonal_count_L") as usize;
        let art_r = v["block_diagonal_count_R"]
            .as_u64()
            .expect("block_diagonal_count_R") as usize;
        assert_eq!(
            art_l, my_l,
            "artifact A_(L) count {art_l} != independent {my_l}"
        );
        assert_eq!(
            art_r, my_r,
            "artifact A_(R) count {art_r} != independent {my_r}"
        );

        // 2. The artifact's A matrices equal our independently computed A.
        let parse_mat = |val: &serde_json::Value| -> super::super::IntMat {
            val.as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| {
                            let n = x.as_i64().unwrap();
                            assert!(
                                (i32::MIN as i64..=i32::MAX as i64).contains(&n),
                                "artifact cell {n} out of i32 range"
                            );
                            n as i32
                        })
                        .collect()
                })
                .collect()
        };
        assert_eq!(
            parse_mat(&v["A_L"]),
            a_l,
            "artifact A_L matrix != independent A_L"
        );
        assert_eq!(
            parse_mat(&v["A_R"]),
            a_r,
            "artifact A_R matrix != independent A_R"
        );

        // 3. Every sample G in the artifact actually squares to A (verify the data itself).
        let mut checked = 0usize;
        for (samples, a) in [(&v["g_sample_L"], &a_l), (&v["g_sample_R"], &a_r)] {
            for g in samples.as_array().unwrap() {
                let gm = parse_mat(g);
                assert_eq!(
                    super::super::imm(&gm, &gm),
                    *a,
                    "an artifact sample G does not square to A"
                );
                checked += 1;
            }
        }
        println!(
            "artifact cross-check OK: counts {my_l}/{my_r}, A matrices match, {checked} samples square to A"
        );
    }
}
