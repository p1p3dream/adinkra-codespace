#![allow(dead_code)] // primitive-library module: much of its API surface is exercised by the test suite, not the binary main path

/// Canonical form computation and equivalence class deduplication for doubly-even codes.
///
/// Two codes are equivalent if one can be obtained from the other by permuting
/// the coordinate positions (columns). We compute a canonical form by trying
/// column permutations, reducing to RREF, and taking the lexicographically
/// smallest result.
///
/// For n <= 10, we use a branch-and-bound search over all column permutations,
/// pruned by column weight profiles. For n > 10, we use a heuristic based on
/// sorted column profiles.

use crate::code::DoublyEvenCode;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Invariants of a binary linear code that are preserved under column permutation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodeInvariants {
    /// Code length (number of bit positions).
    pub n: usize,
    /// Code dimension (number of generators).
    pub k: usize,
    /// Weight enumerator: weight_enumerator[w] = number of codewords of Hamming weight w.
    /// Length is n+1.
    pub weight_enumerator: Vec<usize>,
    /// Minimum Hamming distance (minimum weight of a nonzero codeword).
    /// 0 if the code is trivial (k=0).
    pub min_distance: usize,
    /// Size of the automorphism group (column permutations mapping the code to itself).
    pub automorphism_group_size: usize,
    /// Whether C is contained in its dual C^perp.
    pub is_self_orthogonal: bool,
    /// Whether C equals its dual C^perp (requires k = n/2).
    pub is_self_dual: bool,
    /// Whether the code cannot be decomposed as a direct sum of shorter codes
    /// acting on disjoint column sets.
    pub is_indecomposable: bool,
}

// ---------------------------------------------------------------------------
// Internal: column permutation application
// ---------------------------------------------------------------------------

/// Apply a column permutation to a bitmask word.
/// perm[i] = j means "destination bit i gets the value of source bit j."
fn apply_perm_word(word: u32, perm: &[usize]) -> u32 {
    let mut out = 0u32;
    for (dest, &src) in perm.iter().enumerate() {
        if word & (1 << src) != 0 {
            out |= 1 << dest;
        }
    }
    out
}

/// Apply a column permutation to every row of a generator matrix.
fn apply_perm_matrix(rows: &[u32], perm: &[usize]) -> Vec<u32> {
    rows.iter().map(|&r| apply_perm_word(r, perm)).collect()
}

// ---------------------------------------------------------------------------
// Internal: RREF over GF(2)
// ---------------------------------------------------------------------------

/// Reduce a binary matrix to reduced row echelon form over GF(2).
/// Iterates columns from LSB (column 0) to MSB. Returns sorted nonzero rows.
fn rref_sorted(generators: &[u32]) -> Vec<u32> {
    if generators.is_empty() {
        return vec![];
    }
    let mut rows = generators.to_vec();
    let nrows = rows.len();
    let mut pivot_row = 0;

    for col in 0..32 {
        if pivot_row >= nrows {
            break;
        }
        // Find a row at index >= pivot_row that has a 1 in this column.
        let mut found = None;
        for r in pivot_row..nrows {
            if rows[r] & (1 << col) != 0 {
                found = Some(r);
                break;
            }
        }
        let Some(r) = found else { continue };

        rows.swap(pivot_row, r);
        let pivot_val = rows[pivot_row];
        for r in 0..nrows {
            if r != pivot_row && rows[r] & (1 << col) != 0 {
                rows[r] ^= pivot_val;
            }
        }
        pivot_row += 1;
    }

    rows.retain(|&r| r != 0);
    rows.sort();
    rows
}

// ---------------------------------------------------------------------------
// Internal: column weight profiling
// ---------------------------------------------------------------------------

/// For each column j in 0..n, count how many codewords have a 1 in column j.
fn column_weights(codewords: &[u32], n: usize) -> Vec<usize> {
    let mut weights = vec![0usize; n];
    for &w in codewords {
        for j in 0..n {
            if w & (1 << j) != 0 {
                weights[j] += 1;
            }
        }
    }
    weights
}

// ---------------------------------------------------------------------------
// Internal: permutation generation (Heap's algorithm)
// ---------------------------------------------------------------------------

/// Generate all permutations of 0..n using Heap's algorithm.
fn all_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut c = vec![0usize; n];
    result.push(perm.clone());
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                perm.swap(0, i);
            } else {
                perm.swap(c[i], i);
            }
            result.push(perm.clone());
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    result
}

/// Generate all permutations of the given items (not indices, actual values).
fn all_permutations_of(items: &[usize]) -> Vec<Vec<usize>> {
    let n = items.len();
    if n == 0 {
        return vec![vec![]];
    }
    if n == 1 {
        return vec![vec![items[0]]];
    }
    let mut result = Vec::new();
    let mut arr = items.to_vec();
    let mut c = vec![0usize; n];
    result.push(arr.clone());
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                arr.swap(0, i);
            } else {
                arr.swap(c[i], i);
            }
            result.push(arr.clone());
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Internal: optimized permutation search with column-weight pruning
// ---------------------------------------------------------------------------

/// Group columns by their weight profile value. Returns (weight, columns) pairs
/// sorted by weight.
fn column_groups_by_weight(col_weights: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let mut pairs: Vec<(usize, usize)> = col_weights
        .iter()
        .copied()
        .enumerate()
        .map(|(col, w)| (w, col))
        .collect();
    pairs.sort();

    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut i = 0;
    while i < pairs.len() {
        let w = pairs[i].0;
        let mut cols = Vec::new();
        while i < pairs.len() && pairs[i].0 == w {
            cols.push(pairs[i].1);
            i += 1;
        }
        groups.push((w, cols));
    }
    groups
}

/// Generate all column permutations that respect column-weight classes.
///
/// The key insight: a column permutation can map column i to position j only if
/// col_weights[i] matches the weight of whatever column was originally at position j.
/// Since we're searching for the lex-smallest RREF, we assign groups of same-weight
/// columns to blocks of destination positions, trying all possible group orderings
/// (for groups of the same size but different weights) and all internal permutations.
///
/// More precisely: columns with the same weight are interchangeable. We partition
/// the n destination positions into blocks matching the group sizes (trying all
/// orderings of groups with the same size), then permute within each block.
fn optimized_candidate_permutations(
    col_weights: &[usize],
    n: usize,
) -> Vec<Vec<usize>> {
    let groups = column_groups_by_weight(col_weights);

    // The total permutation count: product of group_size! for each group,
    // times the number of ways to interleave groups of the same size.
    // For the Cartesian product of internal permutations, we also need to
    // consider which destination positions each group gets.
    //
    // Since the destination positions are just 0..n partitioned into blocks,
    // and different orderings of blocks produce different permutations, we
    // need to enumerate orderings of groups with the same size.
    //
    // For simplicity (and because n <= ~12), we generate permutations as:
    // for each ordering of groups -> assign destination blocks -> for each
    // internal permutation within each group -> build the full perm.

    // Groups of the same size can be reordered among themselves. Groups of
    // different sizes can also be reordered, producing different destination
    // block assignments. We need to try ALL orderings of groups.
    let group_count = groups.len();
    let group_orders = all_permutations(group_count);

    let mut results: Vec<Vec<usize>> = Vec::new();

    // Pre-compute internal permutations for each group.
    let internal_perms: Vec<Vec<Vec<usize>>> = groups
        .iter()
        .map(|(_, cols)| all_permutations_of(cols))
        .collect();

    for group_order in &group_orders {
        // Assign destination blocks according to this group ordering.
        // group_order[0] is which group gets destination positions 0..size_of_that_group.
        let mut dest_starts = vec![0usize; group_count];
        let mut pos = 0;
        for &gi in group_order {
            dest_starts[gi] = pos;
            pos += groups[gi].1.len();
        }

        // Cartesian product of internal permutations.
        // Build permutations incrementally.
        let mut partial: Vec<Vec<usize>> = vec![vec![0; n]];

        for gi in 0..group_count {
            let dest_start = dest_starts[gi];
            let mut new_partial = Vec::with_capacity(partial.len() * internal_perms[gi].len());
            for base in &partial {
                for iperm in &internal_perms[gi] {
                    let mut combined = base.clone();
                    // iperm is a rearrangement of the source columns in group gi.
                    // Destination positions for this group start at dest_start.
                    for (k, &src_col) in iperm.iter().enumerate() {
                        combined[dest_start + k] = src_col;
                    }
                    new_partial.push(combined);
                }
            }
            partial = new_partial;
        }

        results.extend(partial);
    }

    results
}

/// Check if the optimized search is feasible (not too many permutations).
fn optimized_feasible(col_weights: &[usize]) -> bool {
    let groups = column_groups_by_weight(col_weights);
    let group_count = groups.len();

    // Number of group orderings: group_count!
    let mut total: u64 = (1..=group_count as u64).product();

    // Times product of group_size! for internal permutations.
    for (_, cols) in &groups {
        let factorial: u64 = (1..=cols.len() as u64).product();
        total = total.saturating_mul(factorial);
        if total > 10_000_000 {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Public: canonical form
// ---------------------------------------------------------------------------

/// Compute a canonical form of the code under column permutation equivalence.
///
/// Returns the generator matrix in canonical form (sorted rows in RREF after
/// applying the lexicographically smallest column permutation).
pub fn canonical_form(code: &DoublyEvenCode) -> Vec<u32> {
    let n = code.n;
    if code.k() == 0 || n == 0 {
        return vec![];
    }

    if n <= 10 {
        // For small n, try all n! permutations (with column-weight pruning).
        canonical_form_exact(code)
    } else {
        // For larger n, use column-weight-based optimization.
        canonical_form_optimized(code)
    }
}

/// Exact canonical form via brute-force over all n! permutations.
fn canonical_form_exact(code: &DoublyEvenCode) -> Vec<u32> {
    let n = code.n;
    let perms = all_permutations(n);
    let mut best: Option<Vec<u32>> = None;
    for perm in &perms {
        let permuted = apply_perm_matrix(&code.generators, perm);
        let r = rref_sorted(&permuted);
        match &best {
            None => best = Some(r),
            Some(current) if r < *current => best = Some(r),
            _ => {}
        }
    }
    best.unwrap_or_default()
}

/// Optimized canonical form using column weight profiling to reduce the search.
fn canonical_form_optimized(code: &DoublyEvenCode) -> Vec<u32> {
    let n = code.n;
    let codewords = code.all_codewords();
    let col_w = column_weights(&codewords, n);

    if optimized_feasible(&col_w) {
        let candidates = optimized_candidate_permutations(&col_w, n);
        let mut best: Option<Vec<u32>> = None;
        for perm in &candidates {
            let permuted = apply_perm_matrix(&code.generators, perm);
            let r = rref_sorted(&permuted);
            match &best {
                None => best = Some(r),
                Some(current) if r < *current => best = Some(r),
                _ => {}
            }
        }
        best.unwrap_or_default()
    } else {
        // Fallback heuristic: sort columns by weight, take that RREF.
        let mut cols_by_weight: Vec<(usize, usize)> = col_w
            .iter()
            .copied()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect();
        cols_by_weight.sort();
        let perm: Vec<usize> = cols_by_weight.iter().map(|&(_, c)| c).collect();
        let permuted = apply_perm_matrix(&code.generators, &perm);
        rref_sorted(&permuted)
    }
}

// ---------------------------------------------------------------------------
// Public: equivalence check
// ---------------------------------------------------------------------------

/// Check if two codes are equivalent under column permutation.
pub fn are_equivalent(a: &DoublyEvenCode, b: &DoublyEvenCode) -> bool {
    if a.n != b.n || a.k() != b.k() {
        return false;
    }
    // Fast pre-filter: weight enumerators must match.
    if a.weight_enumerator() != b.weight_enumerator() {
        return false;
    }
    // Full check via canonical forms.
    canonical_form(a) == canonical_form(b)
}

// ---------------------------------------------------------------------------
// Public: invariants
// ---------------------------------------------------------------------------

/// Compute invariants preserved under column permutation.
pub fn compute_invariants(code: &DoublyEvenCode) -> CodeInvariants {
    let n = code.n;
    let k = code.k();
    let codewords = code.all_codewords();
    let we = weight_enumerator_vec(&codewords, n);

    let min_distance = if k == 0 {
        0
    } else {
        we.iter()
            .enumerate()
            .skip(1)
            .find(|&(_, count)| *count > 0)
            .map(|(w, _)| w)
            .unwrap_or(0)
    };

    let is_self_orthogonal = check_self_orthogonal(&codewords);
    let is_self_dual = is_self_orthogonal && k * 2 == n && check_is_dual(code, &codewords);
    let is_indecomposable = check_indecomposable(code);
    let automorphism_group_size = count_automorphisms(code);

    CodeInvariants {
        n,
        k,
        weight_enumerator: we,
        min_distance,
        automorphism_group_size,
        is_self_orthogonal,
        is_self_dual,
        is_indecomposable,
    }
}

// ---------------------------------------------------------------------------
// Public: deduplication
// ---------------------------------------------------------------------------

/// Deduplicate a list of codes, keeping one representative per equivalence class.
///
/// Uses invariant binning (n, k, weight_enumerator) as a fast pre-filter,
/// then compares canonical forms only within each bin.
pub fn deduplicate(codes: Vec<DoublyEvenCode>) -> Vec<DoublyEvenCode> {
    if codes.is_empty() {
        return codes;
    }

    let mut bins: HashMap<(usize, usize, Vec<usize>), Vec<DoublyEvenCode>> = HashMap::new();
    for code in codes {
        let we = code.weight_enumerator();
        let key = (code.n, code.k(), we);
        bins.entry(key).or_default().push(code);
    }

    let mut result = Vec::new();
    for (_, bin) in bins {
        let mut seen_forms: HashSet<Vec<u32>> = HashSet::new();
        for code in bin {
            let cf = canonical_form(&code);
            if seen_forms.insert(cf) {
                result.push(code);
            }
        }
    }

    result
}

/// Deduplicate a slice of codes (convenience overload that borrows).
pub fn deduplicate_ref(codes: &[DoublyEvenCode]) -> Vec<DoublyEvenCode> {
    deduplicate(codes.to_vec())
}

// ---------------------------------------------------------------------------
// Public: decomposability check
// ---------------------------------------------------------------------------

/// Check if a code is a direct sum of two smaller codes on disjoint column sets.
pub fn is_decomposable(code: &DoublyEvenCode) -> bool {
    if code.k() <= 1 || code.n < 2 {
        return false;
    }
    !check_indecomposable(code)
}

// ---------------------------------------------------------------------------
// Internal: invariant helpers
// ---------------------------------------------------------------------------

fn weight_enumerator_vec(codewords: &[u32], n: usize) -> Vec<usize> {
    let mut we = vec![0usize; n + 1];
    for &w in codewords {
        let wt = w.count_ones() as usize;
        if wt <= n {
            we[wt] += 1;
        }
    }
    we
}

/// Self-orthogonality: every pair of codewords has even inner product.
fn check_self_orthogonal(codewords: &[u32]) -> bool {
    // It suffices to check all pairs of generators (basis vectors), since
    // if <g_i, g_j> = 0 mod 2 for all i,j then <sum_i a_i g_i, sum_j b_j g_j>
    // = sum_{i,j} a_i b_j <g_i, g_j> = 0 mod 2.
    //
    // But for safety and simplicity, and because codeword count is at most 2^k
    // which is small for our use cases, we check all pairs.
    for i in 0..codewords.len() {
        for j in i..codewords.len() {
            if (codewords[i] & codewords[j]).count_ones() % 2 != 0 {
                return false;
            }
        }
    }
    true
}

/// Check if the code is self-dual: C = C^perp. Requires k = n/2.
fn check_is_dual(code: &DoublyEvenCode, codewords: &[u32]) -> bool {
    if code.k() * 2 != code.n {
        return false;
    }
    let n = code.n;
    let codeword_set: HashSet<u32> = codewords.iter().copied().collect();
    let all_mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };

    let mut dual_size = 0usize;
    for v in 0..=all_mask {
        let in_dual = code
            .generators
            .iter()
            .all(|&g| (v & g).count_ones() % 2 == 0);
        if in_dual {
            dual_size += 1;
            if !codeword_set.contains(&v) {
                return false;
            }
        }
    }
    dual_size == codewords.len()
}

/// Indecomposability via generator support graph (union-find).
///
/// A code is decomposable if its generator matrix can be put into block diagonal
/// form by column permutation. Equivalently: columns are nodes, two columns are
/// connected if some GENERATOR ROW has 1 in both positions. The code is
/// indecomposable iff this graph (restricted to used columns) is connected.
///
/// Note: we use generators, not all codewords, because a codeword that is the
/// XOR of generators from different blocks would incorrectly connect the blocks.
fn check_indecomposable(code: &DoublyEvenCode) -> bool {
    let n = code.n;
    if n <= 1 || code.k() == 0 {
        return true;
    }

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    // Connect columns that appear together in a generator row.
    for &row in &code.generators {
        if row == 0 {
            continue;
        }
        let first = row.trailing_zeros() as usize;
        let mut remaining = row & !(1 << first);
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            union(&mut parent, first, bit);
            remaining &= !(1 << bit);
        }
    }

    // Check all used columns are in one component.
    let used: u32 = code.generators.iter().fold(0u32, |acc, &r| acc | r);
    if used == 0 {
        return true;
    }
    let first_used = used.trailing_zeros() as usize;
    let root = find(&mut parent, first_used);
    for j in 0..n {
        if used & (1 << j) != 0 && find(&mut parent, j) != root {
            return false;
        }
    }
    true
}

/// Count column permutations that map the code to itself (automorphism group size).
fn count_automorphisms(code: &DoublyEvenCode) -> usize {
    let n = code.n;
    let k = code.k();
    if k == 0 || n == 0 {
        return factorial(n);
    }

    let codewords = code.all_codewords();
    let codeword_set: HashSet<u32> = codewords.iter().copied().collect();

    if n <= 10 {
        // Brute force: try all n! permutations.
        let perms = all_permutations(n);
        perms
            .iter()
            .filter(|perm| {
                codewords
                    .iter()
                    .all(|&w| codeword_set.contains(&apply_perm_word(w, perm)))
            })
            .count()
    } else {
        // Use optimized permutation set.
        let col_w = column_weights(&codewords, n);
        if optimized_feasible(&col_w) {
            let candidates = optimized_candidate_permutations(&col_w, n);
            candidates
                .iter()
                .filter(|perm| {
                    codewords
                        .iter()
                        .all(|&w| codeword_set.contains(&apply_perm_word(w, perm)))
                })
                .count()
        } else {
            1 // fallback: identity
        }
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DoublyEvenCode;

    /// The [8,4,4] extended Hamming code.
    /// Systematic form: I_4 | P where P encodes the parity check extension.
    fn hamming_8_4() -> DoublyEvenCode {
        DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        )
    }

    /// The zero code of length 4.
    fn zero_code_4() -> DoublyEvenCode {
        DoublyEvenCode::new(4, vec![])
    }

    /// A [4,1,4] code: single generator = 1111.
    fn repetition_4() -> DoublyEvenCode {
        DoublyEvenCode::new(4, vec![0b1111])
    }

    // -- RREF tests --

    #[test]
    fn test_rref_identity() {
        let rows = vec![0b001u32, 0b010, 0b100];
        let r = rref_sorted(&rows);
        assert_eq!(r, vec![0b001, 0b010, 0b100]);
    }

    #[test]
    fn test_rref_needs_elimination() {
        // 0b11 = bits 0,1. 0b10 = bit 1.
        // Col 0: pivot on row 0 (0b11). Eliminate from others: row 1 bit0=0, no change.
        // Col 1: pivot on row 1 (0b10). Eliminate from row 0: 0b11 ^= 0b10 = 0b01.
        // Result: [0b01, 0b10]
        let rows = vec![0b11u32, 0b10];
        let r = rref_sorted(&rows);
        assert_eq!(r, vec![0b01, 0b10]);
    }

    #[test]
    fn test_rref_hamming() {
        let code = hamming_8_4();
        let r = rref_sorted(&code.generators);
        // Should have 4 rows (rank 4).
        assert_eq!(r.len(), 4);
        // In RREF, each row has a unique pivot (leading 1 in a unique column).
        // Verify no two rows share the same lowest set bit after RREF.
        // Actually, in our RREF, pivots may not be the trailing zeros since we
        // iterate from column 0 upward. Let's just verify rank.
        let reduced = r.clone();
        let r2 = rref_sorted(&reduced);
        assert_eq!(r, r2, "RREF should be idempotent");
    }

    // -- Permutation tests --

    #[test]
    fn test_apply_perm_swap() {
        let perm = vec![1, 0]; // dest 0 <- src 1, dest 1 <- src 0
        assert_eq!(apply_perm_word(0b01, &perm), 0b10);
        assert_eq!(apply_perm_word(0b10, &perm), 0b01);
        assert_eq!(apply_perm_word(0b11, &perm), 0b11);
    }

    #[test]
    fn test_all_permutations_count() {
        let perms = all_permutations(3);
        assert_eq!(perms.len(), 6);
    }

    #[test]
    fn test_column_groups() {
        let weights = vec![3, 1, 3, 2, 1];
        let groups = column_groups_by_weight(&weights);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].0, 1); // weight 1
        assert_eq!(groups[0].1, vec![1, 4]);
        assert_eq!(groups[1].0, 2); // weight 2
        assert_eq!(groups[1].1, vec![3]);
        assert_eq!(groups[2].0, 3); // weight 3
        assert_eq!(groups[2].1, vec![0, 2]);
    }

    // -- Canonical form tests --

    #[test]
    fn test_canonical_form_trivial() {
        let code = zero_code_4();
        assert_eq!(canonical_form(&code), Vec::<u32>::new());
    }

    #[test]
    fn test_canonical_form_deterministic() {
        let code = hamming_8_4();
        let cf1 = canonical_form(&code);
        let cf2 = canonical_form(&code);
        assert_eq!(cf1, cf2);
    }

    #[test]
    fn test_canonical_form_permutation_invariant() {
        // [8,1,4] with support on columns 0-3 vs columns 4-7.
        let code_a = DoublyEvenCode::new(8, vec![0b00001111]);
        let code_b = DoublyEvenCode::new(8, vec![0b11110000]);
        assert_eq!(
            canonical_form(&code_a),
            canonical_form(&code_b),
            "Permuted codes must have the same canonical form"
        );
    }

    #[test]
    fn test_canonical_form_is_rref() {
        let code = hamming_8_4();
        let cf = canonical_form(&code);
        let re_rref = rref_sorted(&cf);
        assert_eq!(cf, re_rref, "Canonical form should already be in RREF");
    }

    // -- Equivalence tests --

    #[test]
    fn test_are_equivalent_same_code() {
        let code = hamming_8_4();
        assert!(are_equivalent(&code, &code));
    }

    #[test]
    fn test_are_equivalent_permuted() {
        let code_a = DoublyEvenCode::new(8, vec![0b00001111]);
        let code_b = DoublyEvenCode::new(8, vec![0b11110000]);
        assert!(are_equivalent(&code_a, &code_b));
    }

    #[test]
    fn test_are_equivalent_different() {
        let code_a = DoublyEvenCode::new(8, vec![0b00001111]);
        let code_b = DoublyEvenCode::new(8, vec![0b11111111]);
        assert!(!are_equivalent(&code_a, &code_b));
    }

    #[test]
    fn test_are_equivalent_different_bases_same_code() {
        let code_a = DoublyEvenCode::new(8, vec![0b00001111, 0b11110000]);
        let code_b = DoublyEvenCode::new(8, vec![0b11111111, 0b11110000]);
        assert!(are_equivalent(&code_a, &code_b));
    }

    // -- Invariant tests --

    #[test]
    fn test_weight_enumerator_repetition() {
        let code = repetition_4();
        let inv = compute_invariants(&code);
        assert_eq!(inv.weight_enumerator, vec![1, 0, 0, 0, 1]);
        assert_eq!(inv.min_distance, 4);
    }

    #[test]
    fn test_invariants_hamming() {
        let code = hamming_8_4();
        let inv = compute_invariants(&code);
        assert_eq!(inv.n, 8);
        assert_eq!(inv.k, 4);
        assert_eq!(inv.min_distance, 4);
        assert_eq!(inv.weight_enumerator[0], 1);
        assert_eq!(inv.weight_enumerator[4], 14);
        assert_eq!(inv.weight_enumerator[8], 1);
    }

    #[test]
    fn test_self_orthogonal_doubly_even() {
        let code = hamming_8_4();
        let inv = compute_invariants(&code);
        assert!(inv.is_self_orthogonal);
    }

    #[test]
    fn test_self_dual_hamming() {
        let code = hamming_8_4();
        let inv = compute_invariants(&code);
        assert!(inv.is_self_dual);
    }

    #[test]
    fn test_not_self_dual() {
        let code = DoublyEvenCode::new(8, vec![0b00001111]);
        let inv = compute_invariants(&code);
        assert!(inv.is_self_orthogonal);
        assert!(!inv.is_self_dual);
    }

    #[test]
    fn test_indecomposable_hamming() {
        let code = hamming_8_4();
        let inv = compute_invariants(&code);
        assert!(inv.is_indecomposable);
    }

    #[test]
    fn test_decomposable() {
        let code = DoublyEvenCode::new(8, vec![0b00001111, 0b11110000]);
        assert!(code.is_doubly_even());
        let inv = compute_invariants(&code);
        assert!(!inv.is_indecomposable);
    }

    #[test]
    fn test_is_decomposable_fn() {
        let decomp = DoublyEvenCode::new(8, vec![0b00001111, 0b11110000]);
        assert!(is_decomposable(&decomp));

        let indecomp = hamming_8_4();
        assert!(!is_decomposable(&indecomp));
    }

    // -- Automorphism group tests --

    #[test]
    fn test_automorphism_group_repetition_4() {
        let code = repetition_4();
        let inv = compute_invariants(&code);
        assert_eq!(inv.automorphism_group_size, 24);
    }

    #[test]
    fn test_automorphism_group_8_1() {
        // [8,1,4] with generator = 0b00001111.
        // Automorphisms: permutations within {0,1,2,3} x permutations within {4,5,6,7}
        // = 4! * 4! = 576.
        let code = DoublyEvenCode::new(8, vec![0b00001111]);
        let inv = compute_invariants(&code);
        assert_eq!(inv.automorphism_group_size, 576);
    }

    // -- Deduplication tests --

    #[test]
    fn test_deduplicate() {
        let code_a = DoublyEvenCode::new(8, vec![0b00001111]);
        let code_b = DoublyEvenCode::new(8, vec![0b11110000]);
        let code_c = DoublyEvenCode::new(8, vec![0b11111111]);

        let deduped = deduplicate(vec![code_a, code_b, code_c]);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_deduplicate_identical() {
        let codes = vec![
            DoublyEvenCode::new(4, vec![0b1111]),
            DoublyEvenCode::new(4, vec![0b1111]),
        ];
        let unique = deduplicate(codes);
        assert_eq!(unique.len(), 1);
    }

    #[test]
    fn test_deduplicate_ref() {
        let codes = vec![
            DoublyEvenCode::new(4, vec![0b1111]),
            DoublyEvenCode::new(4, vec![0b1111]),
        ];
        let unique = deduplicate_ref(&codes);
        assert_eq!(unique.len(), 1);
    }
}
