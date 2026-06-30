//! Adinkra height / ranking functions.
//!
//! A *ranking* (also called a "height function" or "hanging gardens" labeling)
//! assigns each vertex of a chromotopology an integer height `h(v)` such that
//! every edge connects vertices whose heights differ by exactly 1:
//!
//! ```text
//! |h(v) - h(w)| == 1   for every edge {v, w}.
//! ```
//!
//! Because the chromotopology is connected and bipartite, the heights are fully
//! determined (up to a global shift) by the choice of sign on each edge that
//! closes the BFS spanning tree. The simplest ranking is the **valise**: a
//! 2-level labeling with all bosons at height 0 and all fermions at height 1.
//! This is the only ranking previously assumed by the codebase.
//!
//! VERTEX INDEXING: heights are indexed by the chromotopology's shared vertex
//! (coset) index space, `0..num_vertices`. See
//! `Chromotopology::is_boson_vertex` for the convention. `height[v]` is the
//! height of vertex `v`.
//!
//! References (see REFERENCES.md):
//!   - Y. X. Zhang, "Adinkras for Mathematicians", arXiv:1111.6055 — the ranking
//!     / height-function / valise definitions and the counting recursion (§6);
//!     the N=4 ranking count cross-checked by `enumerate`.
//!   - Doran, Iga, Landweber, "Cubical Cohomology of Adinkras", arXiv:1207.6806 —
//!     dashings (H¹) are distinct from rankings.
//!   - Adinkra height functions / "hanging gardens" context: arXiv:2410.11137.

use crate::chromotopology::Chromotopology;

/// A height assignment over the vertices of a chromotopology.
///
/// `height[v]` is the integer height of vertex `v`, where `v` ranges over the
/// shared vertex index space `0..num_vertices`. By convention (after
/// canonicalization) the minimum height is 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ranking {
    pub height: Vec<i32>,
}

impl Ranking {
    /// The valise (2-level) ranking: every boson at height 0, every fermion at
    /// height 1.
    ///
    /// This is the canonical, always-valid ranking and is backward-compatible
    /// with the prior implicit assumption (bosons low, fermions high). Every
    /// edge joins a boson to a fermion, so `|h(boson) - h(fermion)| = 1` holds
    /// for every edge by construction.
    pub fn valise(chromo: &Chromotopology) -> Self {
        let n = chromo.num_vertices();
        let mut height = vec![0i32; n];
        for v in 0..n {
            height[v] = if chromo.is_boson_vertex(v) { 0 } else { 1 };
        }
        Ranking { height }
    }

    /// Build a ranking from raw heights, canonicalizing so the minimum is 0.
    ///
    /// A global shift of the heights does not change validity, so we normalize
    /// every ranking to have `min == 0` for stable comparison and dedup.
    pub fn from_heights(mut height: Vec<i32>) -> Self {
        if let Some(&min) = height.iter().min() {
            if min != 0 {
                for h in height.iter_mut() {
                    *h -= min;
                }
            }
        }
        Ranking { height }
    }

    /// Validate the hanging-gardens condition: `|h(v) - h(w)| == 1` on every
    /// edge of the chromotopology.
    ///
    /// Returns `Ok(())` if valid, or `Err` describing the first violating edge
    /// (or a length mismatch).
    pub fn is_valid(&self, chromo: &Chromotopology) -> Result<(), String> {
        if self.height.len() != chromo.num_vertices() {
            return Err(format!(
                "height length {} != num_vertices {}",
                self.height.len(),
                chromo.num_vertices()
            ));
        }
        let n = chromo.n();
        let d = chromo.d();
        for color in 0..n {
            for boson_rank in 0..d {
                let (b, f) = chromo.edge_vertices(color, boson_rank);
                let diff = (self.height[b] - self.height[f]).abs();
                if diff != 1 {
                    return Err(format!(
                        "edge color {} (boson rank {}): vertices {} (h={}) and {} (h={}) differ by {} != 1",
                        color, boson_rank, b, self.height[b], f, self.height[f], diff
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of distinct height levels in this ranking.
    pub fn num_levels(&self) -> usize {
        let mut levels: Vec<i32> = self.height.clone();
        levels.sort_unstable();
        levels.dedup();
        levels.len()
    }

    /// Vertices that are local minima under this ranking: every incident edge
    /// goes UP (all neighbours are higher). These "sources" are the vertices that
    /// can be raised. A vertex with no edges is not a source.
    pub fn sources(&self, adj: &[Vec<usize>]) -> Vec<usize> {
        (0..self.height.len())
            .filter(|&v| !adj[v].is_empty() && adj[v].iter().all(|&w| self.height[w] > self.height[v]))
            .collect()
    }

    /// The node-lifting / vertex-raising operation (arXiv:math-ph/0512016 §5):
    /// raise a source vertex by +2. This preserves the bipartite parity and the
    /// `|Δh| = 1` edge condition (the source sat one below all neighbours; +2 puts
    /// it one above them), turning the source into a target. Caller must ensure
    /// `v` is currently a source.
    pub fn raise_vertex(&mut self, v: usize) {
        self.height[v] += 2;
    }

    /// Produce a polynomial-size family of genuinely multi-level (non-valise)
    /// hangings by source-raising from the valise — usable at N=16 where
    /// [`enumerate`](Ranking::enumerate) is infeasible. Each "chain" repeatedly
    /// raises ONE source (rotating the starting index by the chain number for
    /// diversity) and records every intermediate canonical ranking, until no
    /// raisable source remains. Returns the deduplicated union over `chains`
    /// chains (always includes some 3+-level rankings as soon as a single source
    /// is raised out of the valise's 2 levels).
    pub fn raised_samples(chromo: &Chromotopology, chains: usize, max_out: usize) -> Vec<Ranking> {
        let adj = chromo.vertex_adjacency();
        let nv = chromo.num_vertices();
        let mut seen: std::collections::HashSet<Vec<i32>> = std::collections::HashSet::new();
        let mut out: Vec<Ranking> = Vec::new();
        for chain in 0..chains.max(1) {
            if out.len() >= max_out {
                break;
            }
            let mut cur = Ranking::valise(chromo).height;
            for step in 0..(nv * 4) {
                if out.len() >= max_out {
                    break;
                }
                let srcs = Ranking { height: cur.clone() }.sources(&adj);
                // Raising ONE rotated source at a time (not all at once, which
                // would just invert the valise) builds genuine 3+-level hangings.
                if srcs.is_empty() {
                    break;
                }
                let pick = srcs[(chain + step) % srcs.len()];
                cur[pick] += 2;
                let r = Ranking::from_heights(cur.clone());
                if seen.insert(r.height.clone()) {
                    out.push(r);
                }
            }
        }
        out
    }

    /// Enumerate ALL valid rankings of the chromotopology, modulo a global
    /// shift (each result is canonicalized so `min == 0`), deduplicated.
    ///
    /// Method: pinned-root DFS. We pin vertex 0 at height 0, then explore the
    /// connected component. Each vertex reached for the first time along a
    /// spanning-tree edge gets a binary up/down choice (parent +/- 1); every
    /// other edge to an already-assigned vertex is a *constraint* that must be
    /// satisfied (`|dh| == 1`) or the branch is pruned. Completed assignments
    /// are validated, canonicalized, and collected.
    ///
    /// WARNING: this is exponential in the number of independent cycles (up to
    /// 2^(num_vertices-1) leaves explored in the worst case). It is intended
    /// ONLY for small chromotopologies (N <= 5, i.e. a handful of vertices).
    /// Do NOT call this for N = 16 (it will not terminate in practice).
    pub fn enumerate(chromo: &Chromotopology) -> Vec<Ranking> {
        let nv = chromo.num_vertices();
        if nv == 0 {
            return vec![];
        }

        let adj = chromo.vertex_adjacency();

        // height[v] == i32::MIN sentinel means "unassigned".
        const UNASSIGNED: i32 = i32::MIN;
        let mut height = vec![UNASSIGNED; nv];
        let mut results: Vec<Ranking> = Vec::new();

        // Pin vertex 0 at height 0 (kills the global-shift gauge freedom).
        height[0] = 0;
        dfs_assign(0, &adj, &mut height, &mut results);

        // Canonicalize (min == 0) and dedup. dfs already produces consistent
        // assignments; from_heights normalizes the global shift.
        let mut canon: Vec<Ranking> = results
            .into_iter()
            .map(|r| Ranking::from_heights(r.height))
            .collect();
        canon.sort_by(|a, b| a.height.cmp(&b.height));
        canon.dedup();
        canon
    }
}

/// Recursive DFS that assigns heights to the next unassigned vertex.
///
/// Invariant: every assigned vertex already satisfies `|dh| == 1` against all
/// of its assigned neighbors. We pick the lowest-index unassigned vertex that
/// has at least one assigned neighbor (so it is connected to the frontier),
/// try both `neighbor_height +/- 1`, and recurse. When all reachable vertices
/// are assigned we record the assignment.
fn dfs_assign(
    _root: usize,
    adj: &[Vec<usize>],
    height: &mut Vec<i32>,
    results: &mut Vec<Ranking>,
) {
    const UNASSIGNED: i32 = i32::MIN;

    // Find the lowest-index unassigned vertex adjacent to an assigned one.
    let mut next: Option<(usize, i32)> = None;
    for v in 0..height.len() {
        if height[v] != UNASSIGNED {
            continue;
        }
        for &w in &adj[v] {
            if height[w] != UNASSIGNED {
                next = Some((v, height[w]));
                break;
            }
        }
        if next.is_some() {
            break;
        }
    }

    let (v, base) = match next {
        Some(pair) => pair,
        None => {
            // No frontier vertex: either fully assigned, or a disconnected
            // component remains. For a connected chromotopology the former
            // always holds. Only record if everything is assigned.
            if height.iter().all(|&h| h != UNASSIGNED) {
                results.push(Ranking {
                    height: height.clone(),
                });
            }
            return;
        }
    };

    // Try both height choices for v relative to its assigned neighbor.
    for cand in [base + 1, base - 1] {
        // Check consistency against ALL currently-assigned neighbors.
        let consistent = adj[v].iter().all(|&w| {
            height[w] == UNASSIGNED || (height[w] - cand).abs() == 1
        });
        if !consistent {
            continue;
        }
        height[v] = cand;
        dfs_assign(_root, adj, height, results);
        height[v] = UNASSIGNED;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DoublyEvenCode;

    /// N=4, k=1, code = {0000, 1111}. Chromotopology: 4 bosons + 4 fermions.
    fn chromo_n4_k1() -> Chromotopology {
        Chromotopology::from_code(&DoublyEvenCode::new(4, vec![0b1111]))
    }

    #[test]
    fn valise_is_valid() {
        let ct = chromo_n4_k1();
        let r = Ranking::valise(&ct);
        assert!(r.is_valid(&ct).is_ok(), "{}", r.is_valid(&ct).unwrap_err());
    }

    #[test]
    fn valise_has_two_levels() {
        let ct = chromo_n4_k1();
        let r = Ranking::valise(&ct);
        assert_eq!(r.num_levels(), 2);
    }

    #[test]
    fn valise_bosons_low_fermions_high() {
        let ct = chromo_n4_k1();
        let r = Ranking::valise(&ct);
        for v in 0..ct.num_vertices() {
            if ct.is_boson_vertex(v) {
                assert_eq!(r.height[v], 0, "boson {} should be at height 0", v);
            } else {
                assert_eq!(r.height[v], 1, "fermion {} should be at height 1", v);
            }
        }
    }

    #[test]
    fn from_heights_canonicalizes_min_to_zero() {
        let r = Ranking::from_heights(vec![5, 6, 5, 7]);
        assert_eq!(r.height, vec![0, 1, 0, 2]);
        // Round-trip: canonicalizing an already-canonical ranking is idempotent.
        let r2 = Ranking::from_heights(r.height.clone());
        assert_eq!(r, r2);
    }

    #[test]
    fn from_heights_handles_negative() {
        let r = Ranking::from_heights(vec![-3, -2, -3, -4]);
        assert_eq!(r.height, vec![1, 2, 1, 0]);
    }

    #[test]
    fn enumerate_contains_valise() {
        let ct = chromo_n4_k1();
        let all = Ranking::enumerate(&ct);
        let valise = Ranking::valise(&ct);
        // valise is canonical (min 0) already, so it must appear verbatim.
        assert!(
            all.contains(&valise),
            "valise ranking {:?} not found among {} enumerated rankings",
            valise.height,
            all.len()
        );
    }

    #[test]
    fn enumerate_all_valid() {
        let ct = chromo_n4_k1();
        let all = Ranking::enumerate(&ct);
        assert!(!all.is_empty(), "enumerate returned no rankings");
        for r in &all {
            assert!(
                r.is_valid(&ct).is_ok(),
                "enumerated ranking {:?} is invalid: {}",
                r.height,
                r.is_valid(&ct).unwrap_err()
            );
        }
    }

    #[test]
    fn enumerate_all_canonical_and_distinct() {
        let ct = chromo_n4_k1();
        let all = Ranking::enumerate(&ct);
        // Every result canonicalized to min 0.
        for r in &all {
            assert_eq!(*r.height.iter().min().unwrap(), 0);
        }
        // No duplicates.
        let mut seen = std::collections::HashSet::new();
        for r in &all {
            assert!(seen.insert(r.height.clone()), "duplicate ranking {:?}", r.height);
        }
    }

    #[test]
    fn enumerate_count_n4_k1() {
        // N=4 [4,1,4] chromotopology: 8 vertices, 16 edges (4 colors x 4 edges).
        // The number of valid rankings (mod global shift, canonicalized to
        // min == 0) is a fixed integer determined by the cycle structure.
        // Enumerated value (verified by a standalone replication of the coset /
        // edge construction over code {0000, 1111}): 30 distinct rankings.
        let ct = chromo_n4_k1();
        let all = Ranking::enumerate(&ct);
        assert_eq!(
            all.len(),
            30,
            "expected 30 rankings for N=4 [4,1,4], got {}",
            all.len()
        );
    }

    #[test]
    fn enumerate_includes_inverted_valise() {
        // The all-fermions-low ranking (heights swapped) is also valid and,
        // once canonicalized, distinct from the valise. Both must appear.
        let ct = chromo_n4_k1();
        let all = Ranking::enumerate(&ct);
        let inverted: Vec<i32> = (0..ct.num_vertices())
            .map(|v| if ct.is_boson_vertex(v) { 1 } else { 0 })
            .collect();
        let inverted = Ranking::from_heights(inverted);
        assert!(
            all.contains(&inverted),
            "inverted valise {:?} not found among enumerated rankings",
            inverted.height
        );
    }
}
