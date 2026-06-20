use crate::code::DoublyEvenCode;

/// The quotient graph I^N / C for a doubly-even code C.
///
/// This is the Adinkra chromotopology: a bipartite N-regular edge-colored graph
/// whose vertices are cosets of C in (Z/2Z)^N, partitioned into bosons
/// (even-weight canonical representative) and fermions (odd-weight canonical
/// representative). Each color I in 0..N defines a perfect matching between
/// bosons and fermions via XOR with the standard basis vector e_I.
#[derive(Debug, Clone)]
pub struct Chromotopology {
    n: usize,
    k: usize,
    num_vertices: usize,
    d: usize,
    coset_reps: Vec<u32>,
    coset_map: Vec<usize>,
    boson_indices: Vec<usize>,
    fermion_indices: Vec<usize>,
    boson_rank: Vec<usize>,
    fermion_rank: Vec<usize>,
    edge_target: Vec<Vec<usize>>,
}

impl Chromotopology {
    /// Construct the chromotopology from a doubly-even code.
    ///
    /// Panics if the bipartition is not balanced (which cannot happen for a
    /// doubly-even code, but is checked as a sanity invariant).
    pub fn from_code(code: &DoublyEvenCode) -> Self {
        let n = code.n;
        let k = code.k();
        let total = 1usize << n;
        let num_vertices = 1usize << (n - k);
        let d = num_vertices / 2;

        let codewords = code.codewords();

        // Build coset map: element -> coset index.
        // Iterate in order so the smallest unassigned element becomes the
        // canonical representative for its coset.
        let mut coset_map = vec![usize::MAX; total];
        let mut coset_reps: Vec<u32> = Vec::with_capacity(num_vertices);
        let mut coset_count = 0usize;

        for v in 0..total as u32 {
            if coset_map[v as usize] != usize::MAX {
                continue;
            }
            let idx = coset_count;
            coset_reps.push(v);
            for &c in &codewords {
                coset_map[(v ^ c) as usize] = idx;
            }
            coset_count += 1;
        }
        assert_eq!(coset_count, num_vertices);

        // Bipartition: even-weight rep = boson, odd-weight rep = fermion.
        let mut boson_indices: Vec<usize> = Vec::with_capacity(d);
        let mut fermion_indices: Vec<usize> = Vec::with_capacity(d);

        for (idx, &rep) in coset_reps.iter().enumerate() {
            if rep.count_ones() % 2 == 0 {
                boson_indices.push(idx);
            } else {
                fermion_indices.push(idx);
            }
        }
        assert_eq!(
            boson_indices.len(),
            d,
            "boson count {} != d = {}",
            boson_indices.len(),
            d
        );
        assert_eq!(
            fermion_indices.len(),
            d,
            "fermion count {} != d = {}",
            fermion_indices.len(),
            d
        );

        // Build rank arrays: coset_idx -> rank within its partition.
        let mut boson_rank = vec![usize::MAX; num_vertices];
        let mut fermion_rank = vec![usize::MAX; num_vertices];

        for (rank, &idx) in boson_indices.iter().enumerate() {
            boson_rank[idx] = rank;
        }
        for (rank, &idx) in fermion_indices.iter().enumerate() {
            fermion_rank[idx] = rank;
        }

        // Edge construction: for each color I and boson j (by rank), XOR the
        // canonical rep with e_I to find the neighbor coset. Because XOR with
        // a single basis vector always flips Hamming weight parity, the neighbor
        // is always a fermion.
        let mut edge_target = Vec::with_capacity(n);
        for color in 0..n {
            let basis = 1u32 << color;
            let mut perm = Vec::with_capacity(d);
            for &boson_idx in &boson_indices {
                let rep = coset_reps[boson_idx];
                let neighbor = rep ^ basis;
                let neighbor_coset = coset_map[neighbor as usize];
                let ferm_rank = fermion_rank[neighbor_coset];
                assert_ne!(
                    ferm_rank,
                    usize::MAX,
                    "color {}: boson rep {} XOR e_{} = {} lands in coset {} which is not a fermion",
                    color,
                    rep,
                    color,
                    neighbor,
                    neighbor_coset
                );
                perm.push(ferm_rank);
            }
            edge_target.push(perm);
        }

        Self {
            n,
            k,
            num_vertices,
            d,
            coset_reps,
            coset_map,
            boson_indices,
            fermion_indices,
            boson_rank,
            fermion_rank,
            edge_target,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// The forward permutation for a given color: `color_perm(I)[j]` is the
    /// fermion rank that boson rank `j` connects to via edge color `I`.
    pub fn color_perm(&self, color: usize) -> &[usize] {
        &self.edge_target[color]
    }

    /// The inverse permutation for a given color: `color_perm_inverse(I)[k]`
    /// is the boson rank that fermion rank `k` connects to via edge color `I`.
    pub fn color_perm_inverse(&self, color: usize) -> Vec<usize> {
        let fwd = &self.edge_target[color];
        let mut inv = vec![0usize; self.d];
        for (j, &k) in fwd.iter().enumerate() {
            inv[k] = j;
        }
        inv
    }

    /// Validate structural invariants of the chromotopology.
    ///
    /// Checks:
    /// - `num_vertices == 2 * d`
    /// - `coset_reps.len() == num_vertices`
    /// - Each color gives a bijection from bosons to fermions (all targets
    ///   in range, no duplicates)
    pub fn validate(&self) -> Result<(), String> {
        if self.num_vertices != 2 * self.d {
            return Err(format!(
                "num_vertices ({}) != 2 * d ({})",
                self.num_vertices,
                2 * self.d
            ));
        }

        if self.coset_reps.len() != self.num_vertices {
            return Err(format!(
                "coset_reps.len() ({}) != num_vertices ({})",
                self.coset_reps.len(),
                self.num_vertices
            ));
        }

        for color in 0..self.n {
            let perm = &self.edge_target[color];
            if perm.len() != self.d {
                return Err(format!(
                    "color {}: edge_target length {} != d = {}",
                    color,
                    perm.len(),
                    self.d
                ));
            }

            // Check all targets in range.
            for (j, &target) in perm.iter().enumerate() {
                if target >= self.d {
                    return Err(format!(
                        "color {}: edge_target[{}] = {} out of range (d = {})",
                        color, j, target, self.d
                    ));
                }
            }

            // Check bijectivity: no duplicate targets.
            let mut seen = vec![false; self.d];
            for (j, &target) in perm.iter().enumerate() {
                if seen[target] {
                    return Err(format!(
                        "color {}: fermion rank {} appears more than once (duplicate at boson rank {})",
                        color, target, j
                    ));
                }
                seen[target] = true;
            }
        }

        Ok(())
    }

    /// Return the coset index for an element of (Z/2Z)^N.
    pub fn coset_of(&self, v: u32) -> usize {
        self.coset_map[v as usize]
    }

    /// Return the raw F_2^N representative of the boson at a given rank.
    pub fn boson_rep(&self, rank: usize) -> u32 {
        self.coset_reps[self.boson_indices[rank]]
    }

    /// Return the raw F_2^N representatives of all bosons, indexed by rank.
    pub fn boson_reps(&self) -> Vec<u32> {
        self.boson_indices
            .iter()
            .map(|&idx| self.coset_reps[idx])
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// N=4, k=1, code = {0000, 1111}.
    fn code_n4_k1() -> DoublyEvenCode {
        DoublyEvenCode::new(4, vec![0b1111])
    }

    /// N=4, k=0 (trivial code).
    fn code_n4_trivial() -> DoublyEvenCode {
        DoublyEvenCode::trivial(4)
    }

    /// The [8,4,4] extended Hamming code in systematic form.
    fn hamming_8_4() -> DoublyEvenCode {
        DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        )
    }

    // -- Test 1: N=4, k=1 basic structure ----------------------------------

    #[test]
    fn n4_k1_dimensions() {
        let ct = Chromotopology::from_code(&code_n4_k1());
        assert_eq!(ct.n(), 4);
        assert_eq!(ct.k(), 1);
        assert_eq!(ct.num_vertices(), 8); // 2^(4-1)
        assert_eq!(ct.d(), 4);            // 2^(4-1-1)
    }

    #[test]
    fn n4_k1_validates() {
        let ct = Chromotopology::from_code(&code_n4_k1());
        assert!(ct.validate().is_ok(), "{}", ct.validate().unwrap_err());
    }

    #[test]
    fn n4_k1_color_count() {
        let ct = Chromotopology::from_code(&code_n4_k1());
        // Each of 4 colors is a bijection from 4 bosons to 4 fermions.
        for color in 0..4 {
            assert_eq!(ct.color_perm(color).len(), 4);
        }
    }

    // -- Test 2: N=4, trivial code -----------------------------------------

    #[test]
    fn n4_trivial_dimensions() {
        let ct = Chromotopology::from_code(&code_n4_trivial());
        assert_eq!(ct.n(), 4);
        assert_eq!(ct.k(), 0);
        assert_eq!(ct.num_vertices(), 16); // 2^4
        assert_eq!(ct.d(), 8);             // 2^3
    }

    #[test]
    fn n4_trivial_validates() {
        let ct = Chromotopology::from_code(&code_n4_trivial());
        assert!(ct.validate().is_ok(), "{}", ct.validate().unwrap_err());
    }

    // -- Test 3: [8,4,4] Hamming code --------------------------------------

    #[test]
    fn hamming_dimensions() {
        let ct = Chromotopology::from_code(&hamming_8_4());
        assert_eq!(ct.n(), 8);
        assert_eq!(ct.k(), 4);
        assert_eq!(ct.num_vertices(), 16); // 2^(8-4)
        assert_eq!(ct.d(), 8);             // 2^(8-4-1)
    }

    #[test]
    fn hamming_validates() {
        let ct = Chromotopology::from_code(&hamming_8_4());
        assert!(ct.validate().is_ok(), "{}", ct.validate().unwrap_err());
    }

    // -- Test 4: inverse is actually the inverse ---------------------------

    #[test]
    fn color_perm_inverse_roundtrip_n4_k1() {
        let ct = Chromotopology::from_code(&code_n4_k1());
        for color in 0..ct.n() {
            let fwd = ct.color_perm(color);
            let inv = ct.color_perm_inverse(color);
            // fwd[inv[k]] == k for all k
            for k in 0..ct.d() {
                assert_eq!(
                    fwd[inv[k]], k,
                    "color {}: fwd(inv({})) != {}",
                    color, k, k
                );
            }
            // inv[fwd[j]] == j for all j
            for j in 0..ct.d() {
                assert_eq!(
                    inv[fwd[j]], j,
                    "color {}: inv(fwd({})) != {}",
                    color, j, j
                );
            }
        }
    }

    #[test]
    fn color_perm_inverse_roundtrip_hamming() {
        let ct = Chromotopology::from_code(&hamming_8_4());
        for color in 0..ct.n() {
            let fwd = ct.color_perm(color);
            let inv = ct.color_perm_inverse(color);
            for k in 0..ct.d() {
                assert_eq!(fwd[inv[k]], k, "color {}: roundtrip failed at {}", color, k);
            }
            for j in 0..ct.d() {
                assert_eq!(inv[fwd[j]], j, "color {}: roundtrip failed at {}", color, j);
            }
        }
    }

    // -- Test 5: coset consistency -----------------------------------------

    #[test]
    fn coset_consistency_n4_k1() {
        let code = code_n4_k1();
        let ct = Chromotopology::from_code(&code);
        let codewords = code.codewords();
        let total = 1u32 << code.n;
        for v in 0..total {
            let coset_v = ct.coset_of(v);
            for &c in &codewords {
                assert_eq!(
                    ct.coset_of(v ^ c),
                    coset_v,
                    "v={}, c={}: coset({}) = {} != coset({}) = {}",
                    v,
                    c,
                    v,
                    coset_v,
                    v ^ c,
                    ct.coset_of(v ^ c)
                );
            }
        }
    }

    #[test]
    fn coset_consistency_hamming() {
        let code = hamming_8_4();
        let ct = Chromotopology::from_code(&code);
        let codewords = code.codewords();
        let total = 1u32 << code.n;
        for v in 0..total {
            let coset_v = ct.coset_of(v);
            for &c in &codewords {
                assert_eq!(
                    ct.coset_of(v ^ c),
                    coset_v,
                    "v={}, c={}: coset mismatch",
                    v,
                    c
                );
            }
        }
    }

    #[test]
    fn coset_consistency_trivial() {
        let code = code_n4_trivial();
        let ct = Chromotopology::from_code(&code);
        // Trivial code: each element is its own coset.
        let total = 1u32 << code.n;
        for v in 0..total {
            // Only codeword is 0, so coset_of(v ^ 0) = coset_of(v).
            assert_eq!(ct.coset_of(v), ct.coset_of(v ^ 0));
        }
        // All coset indices should be distinct.
        let mut seen = std::collections::HashSet::new();
        for v in 0..total {
            seen.insert(ct.coset_of(v));
        }
        assert_eq!(seen.len(), 16);
    }
}
