use crate::code::DoublyEvenCode;

// ---------------------------------------------------------------------------
// Minimal PRNG (LCG)
// ---------------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid a zero state which would stay zero forever.
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_bounded(&mut self, bound: u64) -> u64 {
        // Use high bits to avoid LCG low-bit short-cycle artifacts.
        let val = self.next();
        ((val >> 32) as u128 * bound as u128 >> 32) as u64
    }

    /// Random u32 value with exactly `n` usable bit positions (mask = (1 << n) - 1).
    fn next_bits(&mut self, n: usize) -> u32 {
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        (self.next() >> 32) as u32 & mask
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check the generator-level necessary and sufficient conditions for
/// a set of rows to generate a doubly-even code:
///   1. popcount(row) mod 4 == 0  for every row
///   2. popcount(row_i & row_j) mod 2 == 0  for every pair (i, j)
fn generators_valid(rows: &[u32]) -> bool {
    for &r in rows {
        if r.count_ones() % 4 != 0 {
            return false;
        }
    }
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            if (rows[i] & rows[j]).count_ones() % 2 != 0 {
                return false;
            }
        }
    }
    true
}

/// Check whether `vec` is linearly independent of `existing` over GF(2).
fn is_independent_of(existing: &[u32], vec: u32) -> bool {
    if vec == 0 {
        return false;
    }
    let basis = rref(existing);
    let mut reduced = vec;
    for &row in &basis {
        if row == 0 {
            continue;
        }
        let lead = 31 - row.leading_zeros();
        if reduced & (1 << lead) != 0 {
            reduced ^= row;
        }
    }
    reduced != 0
}

/// Put a set of generators into reduced row echelon form (over GF(2))
/// so we can compare code spaces for equality.
pub fn rref(rows: &[u32]) -> Vec<u32> {
    let mut m: Vec<u32> = rows.to_vec();
    let k = m.len();
    let mut pivot_row = 0;
    for col in (0..32).rev() {
        // Find a row with a 1 in this column, at or below pivot_row.
        let mut found = None;
        for r in pivot_row..k {
            if m[r] & (1 << col) != 0 {
                found = Some(r);
                break;
            }
        }
        let Some(r) = found else { continue };
        m.swap(pivot_row, r);
        // Eliminate this column from all other rows.
        for r2 in 0..k {
            if r2 != pivot_row && m[r2] & (1 << col) != 0 {
                m[r2] ^= m[pivot_row];
            }
        }
        pivot_row += 1;
        if pivot_row >= k {
            break;
        }
    }
    // Remove zero rows and sort descending for canonical form.
    m.retain(|&r| r != 0);
    m.sort_unstable_by(|a, b| b.cmp(a));
    m
}

/// Generate a random n-bit vector whose popcount is 0 mod 4.
/// Strategy: generate random bits, then flip a minimal number of bits to fix
/// the weight residue.
fn random_weight_mod4_vec(rng: &mut Rng, n: usize) -> u32 {
    if n < 4 {
        // For very short lengths, the only weight-0-mod-4 vector with weight > 0
        // needs weight 4 which requires n >= 4. Return 0 for n < 4.
        return 0;
    }
    loop {
        let v = rng.next_bits(n);
        let w = v.count_ones();
        if w % 4 == 0 && v != 0 {
            return v;
        }
        // For small n the density of valid nonzero vectors is decent,
        // so rejection sampling converges quickly.
    }
}

/// Two codes span the same subspace if their RREFs match.
fn same_code_space(a: &DoublyEvenCode, b: &DoublyEvenCode) -> bool {
    if a.n != b.n {
        return false;
    }
    rref(&a.generators) == rref(&b.generators)
}

// ---------------------------------------------------------------------------
// Baseline 1: Random valid code generator
// ---------------------------------------------------------------------------

/// Generate a random doubly-even code of length `n` with target dimension `target_k`.
///
/// Strategy: repeatedly sample random n-bit vectors of weight 0 mod 4,
/// check pairwise overlap condition with accumulated generators, verify
/// linear independence, and accumulate valid generators until we reach
/// the target dimension (or exhaust attempts).
///
/// Returns `None` if we fail to reach `target_k` generators within a
/// reasonable number of attempts.
pub fn random_valid_code(n: usize, target_k: usize, rng_seed: u64) -> Option<DoublyEvenCode> {
    let mut rng = Rng::new(rng_seed);
    let mut gens: Vec<u32> = Vec::with_capacity(target_k);
    let max_attempts = target_k * 2000;

    for _ in 0..max_attempts {
        if gens.len() >= target_k {
            break;
        }
        let candidate = random_weight_mod4_vec(&mut rng, n);
        if candidate == 0 {
            continue;
        }
        // Check pairwise even overlap with all existing generators.
        let overlap_ok = gens
            .iter()
            .all(|&g| (candidate & g).count_ones() % 2 == 0);
        if !overlap_ok {
            continue;
        }
        // Check linear independence.
        if !is_independent_of(&gens, candidate) {
            continue;
        }
        gens.push(candidate);
    }

    if gens.is_empty() {
        return None;
    }

    let code = DoublyEvenCode::new(n, gens);
    debug_assert!(code.is_doubly_even());
    Some(code)
}

/// Generate many random valid codes of length `n`.
///
/// Attempts to produce `count` distinct code spaces. Each attempt uses a
/// different seed derived from `seed`. Dimensions are sampled randomly
/// between 1 and n/2 (doubly-even codes are self-orthogonal, so k <= n/2).
pub fn random_batch(n: usize, count: usize, seed: u64) -> Vec<DoublyEvenCode> {
    let mut results: Vec<DoublyEvenCode> = Vec::new();
    let mut rng = Rng::new(seed);
    let max_k = (n / 2).max(1);

    for _ in 0..(count * 5) {
        if results.len() >= count {
            break;
        }
        let target_k = (rng.next_bounded(max_k as u64) as usize) + 1;
        let sub_seed = rng.next();
        if let Some(code) = random_valid_code(n, target_k, sub_seed) {
            // Check for duplicate code spaces.
            let is_dup = results.iter().any(|existing| same_code_space(existing, &code));
            if !is_dup {
                results.push(code);
            }
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Baseline 2: Direct-sum extension
// ---------------------------------------------------------------------------

/// Extend a code to a larger length by appending zero columns.
///
/// The existing generator bitmasks stay the same (their bits occupy the low
/// positions). The new higher-bit columns are all zero, so weights and
/// overlaps are unchanged, and the code remains doubly-even.
///
/// Panics if `new_n` < `code.n`.
pub fn direct_sum_extend(code: &DoublyEvenCode, new_n: usize) -> DoublyEvenCode {
    assert!(
        new_n >= code.n,
        "direct_sum_extend: new_n ({}) must be >= code.n ({})",
        new_n,
        code.n
    );
    DoublyEvenCode::new(new_n, code.generators.clone())
}

/// Extend all known codes with length strictly less than `target_n` to
/// `target_n` via direct-sum extension. Deduplicates by code space.
pub fn all_direct_sum_extensions(known: &[DoublyEvenCode], target_n: usize) -> Vec<DoublyEvenCode> {
    let mut results: Vec<DoublyEvenCode> = Vec::new();
    for code in known {
        if code.n >= target_n {
            continue;
        }
        let extended = direct_sum_extend(code, target_n);
        let is_dup = results
            .iter()
            .any(|existing| same_code_space(existing, &extended));
        if !is_dup {
            results.push(extended);
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Baseline 3: Evolutionary search
// ---------------------------------------------------------------------------

/// Configuration for the evolutionary search baseline.
pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub seed: u64,
}

/// Mutate a code using one of five validity-preserving operators.
///
/// 1. Column swap: permute two column positions (always valid).
/// 2. Row replacement: replace a row with XOR of two existing rows.
/// 3. Row extension: add a new random valid generator.
/// 4. Row deletion: remove a generator (reduces k).
/// 5. Column extension: add a column, setting the new bit in select rows
///    so each row keeps weight 0 mod 4 and pairwise overlaps stay even.
fn mutate(code: &DoublyEvenCode, target_n: usize, rng: &mut Rng) -> DoublyEvenCode {
    let op = rng.next_bounded(5);
    let mut gens = code.generators.clone();
    let mut n = code.n;

    match op {
        0 => {
            // Column swap: pick two distinct column indices and swap them in
            // every generator row. This is a coordinate permutation and always
            // preserves doubly-even validity.
            if n >= 2 {
                let c1 = rng.next_bounded(n as u64) as usize;
                let mut c2 = rng.next_bounded(n as u64) as usize;
                if c2 == c1 {
                    c2 = (c1 + 1) % n;
                }
                for row in gens.iter_mut() {
                    let b1 = (*row >> c1) & 1;
                    let b2 = (*row >> c2) & 1;
                    if b1 != b2 {
                        *row ^= (1 << c1) | (1 << c2);
                    }
                }
            }
        }
        1 => {
            // Row replacement: replace one row with XOR of two others.
            // This keeps the same code space (or a subspace), so validity
            // is preserved.
            if gens.len() >= 3 {
                let idx = rng.next_bounded(gens.len() as u64) as usize;
                let mut a = rng.next_bounded(gens.len() as u64) as usize;
                let mut b = rng.next_bounded(gens.len() as u64) as usize;
                while a == idx {
                    a = rng.next_bounded(gens.len() as u64) as usize;
                }
                while b == idx || b == a {
                    b = rng.next_bounded(gens.len() as u64) as usize;
                }
                let new_row = gens[a] ^ gens[b];
                if new_row != 0 {
                    gens[idx] = new_row;
                }
            }
        }
        2 => {
            // Row extension: add a new random generator that is valid with
            // respect to existing rows.
            if n >= 4 {
                let max_tries = 500;
                for _ in 0..max_tries {
                    let candidate = random_weight_mod4_vec(rng, n);
                    if candidate == 0 {
                        continue;
                    }
                    let overlap_ok = gens
                        .iter()
                        .all(|&g| (candidate & g).count_ones() % 2 == 0);
                    if overlap_ok && is_independent_of(&gens, candidate) {
                        gens.push(candidate);
                        break;
                    }
                }
            }
        }
        3 => {
            // Row deletion: remove a random generator row.
            if gens.len() > 1 {
                let idx = rng.next_bounded(gens.len() as u64) as usize;
                gens.remove(idx);
            }
        }
        4 => {
            // Column extension: add one new column (increase n by 1), setting
            // the new bit in a subset of rows that preserves validity.
            //
            // For a new column at position `n`, each row either gets a 0 or 1
            // in that position. We need:
            //   - popcount(row_i) mod 4 == 0 after flipping the new bit
            //   - popcount(row_i & row_j) mod 2 == 0 for all pairs (the new
            //     column contributes new_bit_i * new_bit_j to the overlap)
            //
            // Strategy: try a random assignment of the new column bits and
            // check validity.
            if n < target_n && n < 31 {
                let new_col = n;
                n += 1;
                // Try several random assignments.
                let mut best_gens = gens.clone();
                let mut found = false;
                for _ in 0..64 {
                    let bits = rng.next_bits(gens.len());
                    let mut trial: Vec<u32> = gens.clone();
                    for (i, row) in trial.iter_mut().enumerate() {
                        if bits & (1 << i) != 0 {
                            *row |= 1 << new_col;
                        }
                    }
                    if generators_valid(&trial) {
                        best_gens = trial;
                        found = true;
                        break;
                    }
                }
                if found {
                    gens = best_gens;
                } else {
                    // Fallback: just add the column as all zeros (always valid).
                }
            }
        }
        _ => unreachable!(),
    }

    DoublyEvenCode::new(n, gens)
}

/// Simple fitness score: prefer codes with higher dimension and length
/// closer to the target. Also reward codes not already seen.
fn fitness(code: &DoublyEvenCode, target_n: usize, seen: &[DoublyEvenCode]) -> f64 {
    let dim_score = code.k() as f64;
    let len_score = if code.n == target_n { 5.0 } else { 0.0 };
    let novelty = if seen.iter().any(|s| same_code_space(s, code)) {
        0.0
    } else {
        10.0
    };
    dim_score + len_score + novelty
}

/// Evolutionary search baseline.
///
/// Starting from an initial population of valid codes, applies random
/// mutations that preserve doubly-even validity, and selects for codes at
/// the target length that are novel (not duplicates of existing codes).
///
/// Returns all unique valid codes found at `target_n`.
pub fn evolve(
    initial_population: Vec<DoublyEvenCode>,
    target_n: usize,
    config: &EvolutionConfig,
) -> Vec<DoublyEvenCode> {
    let mut rng = Rng::new(config.seed);
    let pop_size = config.population_size.max(1);

    // Build the initial population, padding with clones if needed.
    let mut population: Vec<DoublyEvenCode> = initial_population;
    if population.is_empty() {
        // Seed with whatever random codes we can make.
        for i in 0..pop_size {
            if let Some(c) = random_valid_code(target_n, 1, config.seed.wrapping_add(i as u64)) {
                population.push(c);
            }
        }
    }
    while population.len() < pop_size {
        let idx = rng.next_bounded(population.len().max(1) as u64) as usize;
        if let Some(c) = population.get(idx) {
            population.push(c.clone());
        } else {
            break;
        }
    }
    population.truncate(pop_size);

    let mut archive: Vec<DoublyEvenCode> = Vec::new();

    for _gen in 0..config.generations {
        // Generate offspring via mutation.
        let mut offspring: Vec<DoublyEvenCode> = Vec::with_capacity(pop_size);
        for individual in &population {
            let r = (rng.next() as f64) / (u64::MAX as f64);
            if r < config.mutation_rate {
                let child = mutate(individual, target_n, &mut rng);
                if child.is_doubly_even() {
                    offspring.push(child);
                } else {
                    // Mutation produced an invalid code (shouldn't happen
                    // often, but we discard it).
                    offspring.push(individual.clone());
                }
            } else {
                offspring.push(individual.clone());
            }
        }

        // Combine parents and offspring, score, and select the top pop_size.
        let mut combined: Vec<(f64, DoublyEvenCode)> = population
            .into_iter()
            .chain(offspring)
            .map(|c| {
                let f = fitness(&c, target_n, &archive);
                (f, c)
            })
            .collect();
        combined.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(pop_size);

        // Archive novel codes at the target length.
        for (_, code) in &combined {
            if code.n == target_n
                && code.is_doubly_even()
                && !archive.iter().any(|a| same_code_space(a, code))
            {
                archive.push(code.clone());
            }
        }

        population = combined.into_iter().map(|(_, c)| c).collect();
    }

    archive
}

// ---------------------------------------------------------------------------
// Baseline 4: Exhaustive extension by one column
// ---------------------------------------------------------------------------

/// Given a code of length n, try all 2^k ways to insert one new column
/// (at position n) while preserving doubly-even validity.
///
/// For each of the 2^k binary vectors `b` (one bit per generator row),
/// we set bit n of row i to b_i and check:
///   - popcount(row_i) mod 4 == 0  for every i where b_i == 1
///     (rows with b_i == 0 are unchanged and already valid)
///   - popcount(row_i & row_j) mod 2 == 0  for every pair where both
///     b_i and b_j are 1 (the new column adds 1 to their overlap)
///
/// Returns all valid distinct extensions (deduplicated by code space).
pub fn extend_by_one_column(code: &DoublyEvenCode) -> Vec<DoublyEvenCode> {
    let k = code.k();
    let n = code.n;
    assert!(n < 32, "Cannot extend beyond 32 bits");

    if k > 20 {
        // Safety valve: 2^k would be over a million; bail out.
        return vec![direct_sum_extend(code, n + 1)];
    }

    let new_bit = 1u32 << n;
    let total = 1u64 << k;
    let mut results: Vec<DoublyEvenCode> = Vec::new();

    for choice in 0..total {
        // Build extended generators.
        let mut gens: Vec<u32> = code.generators.clone();
        for i in 0..k {
            if choice & (1 << i) != 0 {
                gens[i] |= new_bit;
            }
        }

        if generators_valid(&gens) {
            let extended = DoublyEvenCode::new(n + 1, gens);
            debug_assert!(extended.is_doubly_even());
            // Deduplicate.
            let is_dup = results
                .iter()
                .any(|existing| same_code_space(existing, &extended));
            if !is_dup {
                results.push(extended);
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DoublyEvenCode;

    /// The [8,4,4] extended Hamming code.
    fn hamming_8_4() -> DoublyEvenCode {
        // Systematic generator [I4 | P] with overall parity:
        //   row 0: cols {0,5,6,7}
        //   row 1: cols {1,4,6,7}
        //   row 2: cols {2,4,5,7}
        //   row 3: cols {3,4,5,6}
        DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        )
    }

    /// A trivial [4,1,4] repetition-like code: single generator 0b1111.
    fn trivial_4_1() -> DoublyEvenCode {
        DoublyEvenCode::new(4, vec![0b1111])
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_generators_valid_hamming() {
        let code = hamming_8_4();
        assert!(generators_valid(&code.generators));
    }

    #[test]
    fn test_generators_valid_trivial() {
        let code = trivial_4_1();
        assert!(generators_valid(&code.generators));
    }

    #[test]
    fn test_generators_invalid_weight() {
        // Weight 3 is not 0 mod 4.
        assert!(!generators_valid(&[0b111]));
    }

    #[test]
    fn test_generators_invalid_overlap() {
        // Two rows each of weight 4 but with odd overlap.
        let a = 0b11110000u32; // weight 4
        let b = 0b11100001u32; // weight 4, overlap with a = 0b11100000 = weight 3 (odd)
        assert!(!generators_valid(&[a, b]));
    }

    #[test]
    fn test_rref_deterministic() {
        let rows_a = vec![0b11101000u32, 0b01110100, 0b00111010, 0b11010001];
        let rows_b = vec![
            0b11101000u32 ^ 0b01110100, // XOR of first two
            0b01110100,
            0b00111010,
            0b11010001,
        ];
        assert_eq!(rref(&rows_a), rref(&rows_b));
    }

    #[test]
    fn test_is_independent_of() {
        let existing = vec![0b1100u32, 0b0011u32];
        // 0b1111 = 0b1100 ^ 0b0011, so it is dependent.
        assert!(!is_independent_of(&existing, 0b1111));
        // 0b1010 is independent of {0b1100, 0b0011}.
        assert!(is_independent_of(&existing, 0b1010));
    }

    // -----------------------------------------------------------------------
    // Baseline 1: Random
    // -----------------------------------------------------------------------

    #[test]
    fn test_random_valid_code_produces_valid() {
        for seed in 0..10 {
            if let Some(code) = random_valid_code(8, 2, seed) {
                assert!(code.is_doubly_even(), "seed={seed}: code is not doubly-even");
                assert_eq!(code.n, 8);
                assert!(code.k() >= 1);
                assert!(code.k() <= 2);
            }
        }
    }

    #[test]
    fn test_random_valid_code_dimension() {
        // For n=8, finding k=4 should be feasible (the Hamming code proves it).
        let code = random_valid_code(8, 4, 42);
        // May or may not reach k=4 depending on luck, but should produce something.
        assert!(code.is_some());
        let code = code.unwrap();
        assert!(code.is_doubly_even());
    }

    #[test]
    fn test_random_batch() {
        let codes = random_batch(8, 5, 123);
        for code in &codes {
            assert!(code.is_doubly_even());
            assert_eq!(code.n, 8);
        }
        // All should be distinct code spaces.
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert!(!same_code_space(&codes[i], &codes[j]));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Baseline 2: Direct-sum extension
    // -----------------------------------------------------------------------

    #[test]
    fn test_direct_sum_extend_preserves_validity() {
        let code = hamming_8_4();
        let extended = direct_sum_extend(&code, 12);
        assert_eq!(extended.n, 12);
        assert_eq!(extended.k(), 4);
        assert!(extended.is_doubly_even());
    }

    #[test]
    fn test_direct_sum_extend_same_length() {
        let code = trivial_4_1();
        let extended = direct_sum_extend(&code, 4);
        assert_eq!(extended, code);
    }

    #[test]
    #[should_panic]
    fn test_direct_sum_extend_shorter_panics() {
        let code = hamming_8_4();
        let _ = direct_sum_extend(&code, 6);
    }

    #[test]
    fn test_all_direct_sum_extensions() {
        let known = vec![trivial_4_1(), hamming_8_4()];
        let extended = all_direct_sum_extensions(&known, 10);
        for code in &extended {
            assert_eq!(code.n, 10);
            assert!(code.is_doubly_even());
        }
        // The [4,1] code extended to 10 and the [8,4] code extended to 10
        // span different subspaces, so we should get at least 2.
        assert!(extended.len() >= 2);
    }

    // -----------------------------------------------------------------------
    // Baseline 3: Evolutionary search
    // -----------------------------------------------------------------------

    #[test]
    fn test_evolve_produces_valid_codes() {
        let seed_pop = vec![trivial_4_1()];
        let config = EvolutionConfig {
            population_size: 10,
            generations: 20,
            mutation_rate: 0.8,
            seed: 99,
        };
        let results = evolve(seed_pop, 8, &config);
        for code in &results {
            assert!(code.is_doubly_even());
            assert_eq!(code.n, 8);
        }
    }

    #[test]
    fn test_evolve_from_empty() {
        let config = EvolutionConfig {
            population_size: 8,
            generations: 15,
            mutation_rate: 0.9,
            seed: 7,
        };
        let results = evolve(vec![], 8, &config);
        for code in &results {
            assert!(code.is_doubly_even());
        }
    }

    // -----------------------------------------------------------------------
    // Baseline 4: Exhaustive extension
    // -----------------------------------------------------------------------

    #[test]
    fn test_extend_by_one_column_trivial() {
        // [4,1] code with generator 0b1111 (weight 4).
        // Extending to length 5: the new bit either stays 0 (weight 4, ok)
        // or becomes 1 (weight 5, not 0 mod 4). So the only valid extension
        // is the trivial zero-column extension.
        let code = trivial_4_1();
        let extensions = extend_by_one_column(&code);
        assert!(!extensions.is_empty());
        for ext in &extensions {
            assert_eq!(ext.n, 5);
            assert!(ext.is_doubly_even());
        }
        // Should get exactly 1 extension (the zero-column one).
        assert_eq!(extensions.len(), 1);
        assert_eq!(extensions[0].generators, vec![0b1111]);
    }

    #[test]
    fn test_extend_by_one_column_hamming() {
        let code = hamming_8_4();
        let extensions = extend_by_one_column(&code);
        assert!(!extensions.is_empty());
        for ext in &extensions {
            assert_eq!(ext.n, 9);
            assert!(ext.is_doubly_even());
        }
        // The trivial extension (all zeros in new column) must always be present.
        let trivial = direct_sum_extend(&code, 9);
        assert!(extensions
            .iter()
            .any(|e| same_code_space(e, &trivial)));
    }

    #[test]
    fn test_extend_by_one_column_all_distinct() {
        let code = hamming_8_4();
        let extensions = extend_by_one_column(&code);
        for i in 0..extensions.len() {
            for j in (i + 1)..extensions.len() {
                assert!(
                    !same_code_space(&extensions[i], &extensions[j]),
                    "extensions[{i}] and extensions[{j}] are duplicates"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cross-baseline consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_direct_sum_is_subset_of_exhaustive() {
        // Every direct-sum extension should appear in the exhaustive list.
        let code = trivial_4_1();
        let ds = direct_sum_extend(&code, 5);
        let exhaustive = extend_by_one_column(&code);
        assert!(exhaustive.iter().any(|e| same_code_space(e, &ds)));
    }
}
