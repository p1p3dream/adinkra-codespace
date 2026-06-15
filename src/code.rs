#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoublyEvenCode {
    pub n: usize,
    pub generators: Vec<u32>,
}

impl DoublyEvenCode {
    pub fn new(n: usize, generators: Vec<u32>) -> Self {
        Self { n, generators }
    }

    pub fn trivial(n: usize) -> Self {
        Self {
            n,
            generators: vec![],
        }
    }

    /// Validate: generators fit in n bits, are linearly independent, and the
    /// code is doubly-even (every codeword has weight divisible by 4).
    pub fn is_valid(&self) -> bool {
        if self.generators.is_empty() {
            return true;
        }
        let bit_mask = if self.n >= 32 {
            u32::MAX
        } else {
            (1u32 << self.n) - 1
        };
        for &row in &self.generators {
            if row & !bit_mask != 0 {
                return false;
            }
        }
        let r = rref(&self.generators);
        if r.len() != self.generators.len() {
            return false;
        }
        self.is_doubly_even()
    }

    /// Check that every codeword has weight divisible by 4.
    pub fn is_doubly_even(&self) -> bool {
        self.codewords()
            .iter()
            .all(|&w| w.count_ones() % 4 == 0)
    }

    pub fn k(&self) -> usize {
        self.generators.len()
    }

    pub fn num_codewords(&self) -> usize {
        1 << self.k()
    }

    /// Vertices per partition in the Adinkra quotient graph: 2^(n-k).
    pub fn quotient_size(&self) -> usize {
        1 << (self.n - self.k())
    }

    pub fn codewords(&self) -> Vec<u32> {
        let k = self.k();
        let count = 1usize << k;
        let mut words = Vec::with_capacity(count);
        for mask in 0..count {
            let mut word = 0u32;
            for (i, &g) in self.generators.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    word ^= g;
                }
            }
            words.push(word);
        }
        words
    }

    /// Backward-compatible alias for codewords().
    pub fn all_codewords(&self) -> Vec<u32> {
        self.codewords()
    }

    /// Weight enumerator: index i holds the number of codewords with Hamming weight i.
    pub fn weight_enumerator(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.n + 1];
        for cw in self.codewords() {
            counts[cw.count_ones() as usize] += 1;
        }
        counts
    }

    /// RREF the generators to remove dependence and get true dimension.
    pub fn normalize(&self) -> Self {
        let basis = rref(&self.generators);
        Self {
            n: self.n,
            generators: basis,
        }
    }

    /// Minimum distance: smallest nonzero codeword weight. Returns 0 for k=0.
    pub fn min_distance(&self) -> usize {
        if self.k() == 0 {
            return 0;
        }
        self.codewords()
            .iter()
            .filter(|&&w| w != 0)
            .map(|&w| w.count_ones() as usize)
            .min()
            .unwrap_or(0)
    }

    /// A code is indecomposable iff it cannot be written as a direct sum of
    /// two codes on disjoint coordinate sets. We check via union-find: bit
    /// positions that co-occur in any generator are merged, then the used
    /// positions must form a single connected component.
    pub fn is_indecomposable(&self) -> bool {
        if self.k() == 0 {
            return false;
        }

        let mut support_bits: Vec<usize> = Vec::new();
        for &g in &self.generators {
            for bit in 0..self.n {
                if g & (1 << bit) != 0 {
                    support_bits.push(bit);
                }
            }
        }
        support_bits.sort_unstable();
        support_bits.dedup();

        if support_bits.is_empty() {
            return false;
        }

        let max_bit = *support_bits.last().unwrap();
        let mut parent: Vec<usize> = (0..=max_bit).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while parent[r] != r {
                r = parent[r];
            }
            let mut c = x;
            while c != r {
                let next = parent[c];
                parent[c] = r;
                c = next;
            }
            r
        }

        for &g in &self.generators {
            let bits: Vec<usize> = (0..self.n).filter(|&b| g & (1 << b) != 0).collect();
            for window in bits.windows(2) {
                let ra = find(&mut parent, window[0]);
                let rb = find(&mut parent, window[1]);
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }

        let root = find(&mut parent, support_bits[0]);
        support_bits
            .iter()
            .all(|&b| find(&mut parent, b) == root)
    }
}

// ---------------------------------------------------------------------------
// GF(2) linear algebra
// ---------------------------------------------------------------------------

/// Reduced row echelon form over GF(2), pivoting from the LSB upward.
/// Returns the nonzero basis vectors sorted ascending by value.
pub fn rref(vectors: &[u32]) -> Vec<u32> {
    let mut rows: Vec<u32> = vectors.iter().copied().filter(|&v| v != 0).collect();
    let mut cur_row = 0;
    let mut pivot_col = 0u32;

    while cur_row < rows.len() && pivot_col < 32 {
        let pivot_mask = 1u32 << pivot_col;
        if let Some(idx) = (cur_row..rows.len()).find(|&i| rows[i] & pivot_mask != 0) {
            rows.swap(cur_row, idx);
            for i in 0..rows.len() {
                if i != cur_row && rows[i] & pivot_mask != 0 {
                    rows[i] ^= rows[cur_row];
                }
            }
            cur_row += 1;
        }
        pivot_col += 1;
    }

    rows.retain(|&v| v != 0);
    rows.sort_unstable();
    rows
}

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------

/// Enumerate all doubly-even codes of length n. Returns every distinct
/// doubly-even subspace (as a code with RREF generators), including the
/// trivial k=0 code. Does NOT deduplicate by column-permutation equivalence;
/// that is handled by canonical::deduplicate.
pub fn enumerate_codes(n: usize) -> Vec<DoublyEvenCode> {
    if n == 0 {
        return vec![DoublyEvenCode::new(0, vec![])];
    }

    let max_val = if n >= 32 {
        u32::MAX
    } else {
        (1u32 << n) - 1
    };
    let de_vectors: Vec<u32> = (1..=max_val)
        .filter(|v| v.count_ones() % 4 == 0)
        .collect();

    let mut results: Vec<DoublyEvenCode> = vec![DoublyEvenCode::new(n, vec![])];

    if de_vectors.is_empty() {
        return results;
    }

    fn search(
        n: usize,
        basis: &mut Vec<u32>,
        start_idx: usize,
        candidates: &[u32],
        results: &mut Vec<DoublyEvenCode>,
    ) {
        for idx in start_idx..candidates.len() {
            let v = candidates[idx];

            // Reduce v against the current basis to check linear independence.
            let mut reduced = v;
            for &b in basis.iter() {
                let lead = 31 - b.leading_zeros();
                if reduced & (1 << lead) != 0 {
                    reduced ^= b;
                }
            }
            if reduced == 0 {
                continue;
            }

            // Verify the new coset (old_span XOR reduced) is entirely weight-0-mod-4.
            let old_size = 1usize << basis.len();
            let mut all_de = true;
            for mask in 0..old_size {
                let mut word = reduced;
                for (bit, &b) in basis.iter().enumerate() {
                    if mask & (1 << bit) != 0 {
                        word ^= b;
                    }
                }
                if word.count_ones() % 4 != 0 {
                    all_de = false;
                    break;
                }
            }
            if !all_de {
                continue;
            }

            basis.push(reduced);
            results.push(DoublyEvenCode::new(n, basis.clone()));
            search(n, basis, idx + 1, candidates, results);
            basis.pop();
        }
    }

    search(n, &mut Vec::new(), 0, &de_vectors, &mut results);
    results
}

/// For each n in 1..=max_n, return (n, number of raw doubly-even subspaces).
/// To get equivalence class counts, pipe through canonical::deduplicate.
pub fn count_classes(max_n: usize) -> Vec<(usize, usize)> {
    (1..=max_n)
        .map(|n| (n, enumerate_codes(n).len()))
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The [8,4,4] extended Hamming code in systematic form [I4 | P].
    fn hamming_8_4() -> DoublyEvenCode {
        DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        )
    }

    // -- basic struct -------------------------------------------------------

    #[test]
    fn trivial_code_is_valid() {
        let c = DoublyEvenCode::trivial(4);
        assert!(c.is_valid());
        assert_eq!(c.k(), 0);
        assert_eq!(c.num_codewords(), 1);
        assert_eq!(c.codewords(), vec![0]);
    }

    #[test]
    fn single_generator_weight_4() {
        let c = DoublyEvenCode::new(4, vec![0b1111]);
        assert!(c.is_valid());
        assert_eq!(c.k(), 1);
        assert_eq!(c.num_codewords(), 2);
        let mut cw = c.codewords();
        cw.sort();
        assert_eq!(cw, vec![0, 0b1111]);
    }

    #[test]
    fn invalid_weight_not_mod4() {
        let c = DoublyEvenCode::new(4, vec![0b0011]);
        assert!(!c.is_doubly_even());
    }

    #[test]
    fn invalid_odd_overlap() {
        let c = DoublyEvenCode::new(5, vec![0b01111, 0b11110]);
        assert!(!c.is_doubly_even());
    }

    #[test]
    fn valid_two_generators() {
        let g1 = 0b1111_0000u32;
        let g2 = 0b1100_1100u32;
        let c = DoublyEvenCode::new(8, vec![g1, g2]);
        assert!(c.is_valid());
    }

    #[test]
    fn is_valid_rejects_dependent_generators() {
        // g2 = g0 XOR g1, so these are linearly dependent
        let c = DoublyEvenCode::new(8, vec![0b1111_0000, 0b1100_1100, 0b0011_1100]);
        assert!(!c.is_valid());
    }

    // -- all_codewords backward compat --------------------------------------

    #[test]
    fn all_codewords_matches_codewords() {
        let c = hamming_8_4();
        assert_eq!(c.all_codewords(), c.codewords());
    }

    // -- weight enumerator --------------------------------------------------

    #[test]
    fn weight_enumerator_trivial() {
        let we = DoublyEvenCode::trivial(4).weight_enumerator();
        assert_eq!(we[0], 1);
        assert_eq!(we.iter().sum::<usize>(), 1);
    }

    #[test]
    fn weight_enumerator_single_gen() {
        let we = DoublyEvenCode::new(4, vec![0b1111]).weight_enumerator();
        assert_eq!(we[0], 1);
        assert_eq!(we[4], 1);
        assert_eq!(we.iter().sum::<usize>(), 2);
    }

    #[test]
    fn weight_enumerator_length() {
        let we = DoublyEvenCode::new(6, vec![0b111100]).weight_enumerator();
        assert_eq!(we.len(), 7);
    }

    #[test]
    fn weight_enumerator_extended_hamming() {
        let we = hamming_8_4().weight_enumerator();
        assert_eq!(we[0], 1);
        assert_eq!(we[4], 14);
        assert_eq!(we[8], 1);
        assert_eq!(we.iter().sum::<usize>(), 16);
    }

    // -- min distance -------------------------------------------------------

    #[test]
    fn min_distance_trivial() {
        assert_eq!(DoublyEvenCode::trivial(4).min_distance(), 0);
    }

    #[test]
    fn min_distance_extended_hamming() {
        assert_eq!(hamming_8_4().min_distance(), 4);
    }

    // -- quotient size ------------------------------------------------------

    #[test]
    fn quotient_size_formula() {
        let c = DoublyEvenCode::new(8, vec![0b1111_0000, 0b1100_1100]);
        assert_eq!(c.quotient_size(), 1 << 6);
    }

    // -- decomposability ----------------------------------------------------

    #[test]
    fn indecomposable_single_generator() {
        assert!(DoublyEvenCode::new(4, vec![0b1111]).is_indecomposable());
    }

    #[test]
    fn decomposable_disjoint_supports() {
        let c = DoublyEvenCode::new(8, vec![0b0000_1111, 0b1111_0000]);
        assert!(!c.is_indecomposable());
    }

    #[test]
    fn trivial_not_indecomposable() {
        assert!(!DoublyEvenCode::trivial(4).is_indecomposable());
    }

    #[test]
    fn extended_hamming_indecomposable() {
        assert!(hamming_8_4().is_indecomposable());
    }

    // -- GF(2) linear algebra -----------------------------------------------

    #[test]
    fn rref_basic() {
        // Input: [0b110, 0b011]. Pivoting from LSB:
        //   col 0 pivot = 0b011. Eliminate bit 0 from 0b110 (no-op).
        //   col 1 pivot = 0b110. Eliminate bit 1 from 0b011 -> 0b011 XOR 0b110 = 0b101.
        // Result sorted: [0b101, 0b110].
        let r = rref(&[0b110, 0b011]);
        assert_eq!(r.len(), 2);
        // Verify span is correct: {0, 0b101, 0b110, 0b011}
        let span: BTreeSet<u32> = [0, r[0], r[1], r[0] ^ r[1]].into_iter().collect();
        assert!(span.contains(&0b110));
        assert!(span.contains(&0b011));
    }

    #[test]
    fn rref_dependent() {
        assert_eq!(rref(&[0b111, 0b110, 0b001]).len(), 2);
    }

    #[test]
    fn rref_zero_vectors() {
        assert_eq!(rref(&[0, 0, 0b101]), vec![0b101]);
    }

    #[test]
    fn rref_idempotent() {
        let r1 = rref(&[0b1101, 0b1011, 0b0110]);
        let r2 = rref(&r1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rref_preserves_span() {
        let input = [0b1101u32, 0b1011, 0b0110];
        let r = rref(&input);

        // Compute span of input
        let mut input_span = BTreeSet::new();
        for mask in 0..(1u32 << input.len()) {
            let mut w = 0u32;
            for (i, &v) in input.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    w ^= v;
                }
            }
            input_span.insert(w);
        }

        // Compute span of rref output
        let mut rref_span = BTreeSet::new();
        for mask in 0..(1u32 << r.len()) {
            let mut w = 0u32;
            for (i, &v) in r.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    w ^= v;
                }
            }
            rref_span.insert(w);
        }

        assert_eq!(input_span, rref_span);
    }

    // -- enumeration --------------------------------------------------------

    #[test]
    fn no_nontrivial_codes_n1_to_3() {
        for n in 1..=3 {
            let codes = enumerate_codes(n);
            assert_eq!(codes.len(), 1, "n={n}: expected only trivial code");
            assert_eq!(codes[0].k(), 0);
        }
    }

    #[test]
    fn n4_has_nontrivial_code() {
        let codes = enumerate_codes(4);
        let nontrivial: Vec<_> = codes.iter().filter(|c| c.k() > 0).collect();
        assert!(!nontrivial.is_empty(), "n=4 should have nontrivial codes");
    }

    #[test]
    fn all_enumerated_codes_doubly_even() {
        for n in 1..=8 {
            for code in enumerate_codes(n) {
                assert!(
                    code.is_doubly_even(),
                    "non-doubly-even code: n={}, generators={:?}",
                    n,
                    code.generators
                );
            }
        }
    }

    #[test]
    fn codewords_closed_under_xor() {
        for n in 4..=8 {
            for code in enumerate_codes(n) {
                let set: BTreeSet<u32> = code.codewords().into_iter().collect();
                for &a in &set {
                    for &b in &set {
                        assert!(set.contains(&(a ^ b)), "n={n}: {a:#b} ^ {b:#b} missing");
                    }
                }
            }
        }
    }

    #[test]
    fn weight_enumerator_only_weight_mod4() {
        for n in 4..=8 {
            for code in enumerate_codes(n) {
                let we = code.weight_enumerator();
                for (wt, &count) in we.iter().enumerate() {
                    if wt % 4 != 0 {
                        assert_eq!(count, 0, "n={n}: weight {wt} has {count} codewords");
                    }
                }
            }
        }
    }

    #[test]
    fn enumeration_monotonic() {
        let mut prev = 0;
        for n in 1..=8 {
            let count = enumerate_codes(n).len();
            assert!(count >= prev, "code count decreased at n={n}");
            prev = count;
        }
    }

    #[test]
    fn n8_has_k4_codes() {
        let codes = enumerate_codes(8);
        let has_k4 = codes.iter().any(|c| c.k() == 4);
        assert!(has_k4, "n=8 should have codes with dimension 4");
    }

    #[test]
    fn n8_includes_extended_hamming_weight_enumerator() {
        let codes = enumerate_codes(8);
        let has_hamming = codes.iter().any(|c| {
            if c.k() != 4 {
                return false;
            }
            let we = c.weight_enumerator();
            we[0] == 1 && we[4] == 14 && we[8] == 1
        });
        assert!(has_hamming, "n=8 must include a code with the [8,4,4] weight enumerator");
    }

    #[test]
    fn count_classes_consistent() {
        for (n, count) in count_classes(6) {
            assert_eq!(count, enumerate_codes(n).len());
        }
    }

    #[test]
    fn hamming_8_4_properties() {
        let code = hamming_8_4();
        assert!(code.is_valid());
        assert_eq!(code.k(), 4);
        assert_eq!(code.num_codewords(), 16);
        assert_eq!(code.min_distance(), 4);
        assert!(code.is_indecomposable());

        for i in 0..4 {
            assert_eq!(
                code.generators[i].count_ones() % 4,
                0,
                "generator {i} weight not 0 mod 4"
            );
            for j in (i + 1)..4 {
                let overlap = (code.generators[i] & code.generators[j]).count_ones();
                assert_eq!(overlap % 2, 0, "generators {i},{j} have odd overlap");
            }
        }
    }

    #[test]
    fn trivial_code_legacy() {
        let code = DoublyEvenCode::new(4, vec![]);
        assert!(code.is_doubly_even());
        assert_eq!(code.all_codewords(), vec![0]);
    }
}
