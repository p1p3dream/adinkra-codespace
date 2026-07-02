use crate::baselines;
use crate::canonical::{compute_invariants, is_decomposable};
use crate::code::{enumerate_codes, DoublyEvenCode};

use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Known counts from Doran, Faux, Gates et al. (arXiv:0806.0050, Table 4)
// and Robert L. Miller's database.
//
// These are the numbers of permutation-equivalence classes of doubly-even
// binary linear codes INCLUDING codes with zero columns (i.e., shorter codes
// embedded in length N). Each entry is (k, expected_count).
// ---------------------------------------------------------------------------

/// N=4: only the single [4,1,4] repetition code.
const MILLER_N4: [(usize, usize); 1] = [(1, 1)];

/// N=8: includes zero-column embeddings from N=4.
const MILLER_N8: [(usize, usize); 4] = [
    (1, 2), (2, 2), (3, 2), (4, 1),
];

/// N=12: includes zero-column embeddings from N=4 and N=8.
const MILLER_N12: [(usize, usize); 5] = [
    (1, 3), (2, 5), (3, 7), (4, 7), (5, 2),
];

/// N=16: full breakdown from Table 4.
const MILLER_N16: [(usize, usize); 8] = [
    (1, 4), (2, 10), (3, 23), (4, 38), (5, 36), (6, 23), (7, 9), (8, 2),
];

pub struct SearchConfig {
    pub target_n: usize,
    pub evo_population: usize,
    pub evo_generations: usize,
    pub random_batch_size: usize,
    pub seed: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            target_n: 16,
            evo_population: 500,
            evo_generations: 500,
            random_batch_size: 2000,
            seed: 42,
        }
    }
}

struct SearchState {
    found: Vec<DoublyEvenCode>,
    canonical_set: HashSet<Vec<u64>>,
    target_n: usize,
}

impl SearchState {
    fn new(target_n: usize) -> Self {
        Self {
            found: Vec::new(),
            canonical_set: HashSet::new(),
            target_n,
        }
    }

    fn try_add(&mut self, code: DoublyEvenCode) -> bool {
        if code.n != self.target_n {
            return false;
        }
        let code = code.normalize();
        if code.k() == 0 {
            return false;
        }
        if !code.is_doubly_even() {
            return false;
        }

        let cf = crate::nauty_canonical::exact_canonical_key(&code);
        if self.canonical_set.contains(&cf) {
            return false;
        }
        self.canonical_set.insert(cf);
        self.found.push(code);
        true
    }

    fn count(&self) -> usize {
        self.found.len()
    }

    /// Extend every code at dimension target_k-1 by trying all possible
    /// additional generators. This is exhaustive for the extension step:
    /// every valid (k+1)-dimensional code containing a known k-dimensional
    /// code as a subcode will be found.
    fn extend_from_k(&mut self, target_k: usize) -> usize {
        let n = self.target_n;
        let sources: Vec<DoublyEvenCode> = self.found.iter()
            .filter(|c| c.k() == target_k - 1)
            .cloned()
            .collect();

        let mut new_count = 0usize;
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };

        for code in &sources {
            let gens = &code.generators;
            // Try every possible new generator
            for candidate in 1..=mask {
                // Must have weight divisible by 4
                if candidate.count_ones() % 4 != 0 {
                    continue;
                }
                // Must have even overlap with every existing generator
                let overlap_ok = gens.iter().all(|&g| (candidate & g).count_ones() % 2 == 0);
                if !overlap_ok {
                    continue;
                }
                // Must be linearly independent
                let mut reduced = candidate;
                let basis = crate::code::rref(gens);
                for &row in &basis {
                    if row == 0 { continue; }
                    let lead = 31 - row.leading_zeros();
                    if reduced & (1 << lead) != 0 {
                        reduced ^= row;
                    }
                }
                if reduced == 0 {
                    continue;
                }
                // Build the extended code
                let mut new_gens = gens.clone();
                new_gens.push(candidate);
                let new_code = DoublyEvenCode::new(n, new_gens);
                if self.try_add(new_code) {
                    new_count += 1;
                }
            }
        }
        new_count
    }
}

/// Enumerate all nontrivial codes at N=4 through max_seed_n.
fn gather_seed_codes(max_seed_n: usize) -> Vec<DoublyEvenCode> {
    let mut seeds = Vec::new();
    for n in 4..=max_seed_n {
        let start = Instant::now();
        let codes = enumerate_codes(n);
        let deduped = crate::canonical::deduplicate(codes);
        eprintln!(
            "  Enumerated N={}: {} classes [{:?}]",
            n,
            deduped.len(),
            start.elapsed()
        );
        for code in deduped {
            if code.k() > 0 {
                seeds.push(code);
            }
        }
    }
    seeds
}

/// Extend a set of codes by one column, keeping the best results.
/// Returns codes at n+1 that are valid doubly-even.
fn extend_level(codes: &[DoublyEvenCode], max_keep: usize) -> Vec<DoublyEvenCode> {
    let mut next: Vec<DoublyEvenCode> = Vec::new();
    let mut seen_rref: HashSet<Vec<u32>> = HashSet::new();

    for code in codes {
        let extensions = baselines::extend_by_one_column(code);
        for ext in extensions {
            if ext.is_doubly_even() {
                let r = baselines::rref(&ext.generators);
                if !seen_rref.contains(&r) {
                    seen_rref.insert(r);
                    next.push(ext);
                }
            }
        }
    }

    // Prioritize high-k indecomposable codes
    next.sort_by(|a, b| {
        let ka = a.k();
        let kb = b.k();
        let da = if ka > 1 && !is_decomposable(a) { 1 } else { 0 };
        let db = if kb > 1 && !is_decomposable(b) { 1 } else { 0 };
        (db, kb).cmp(&(da, ka))
    });
    next.truncate(max_keep);
    next
}

/// Build codes at target_n by chaining column extensions from enumerated seeds.
fn chain_extend(target_n: usize, max_per_level: usize) -> Vec<DoublyEvenCode> {
    let start = Instant::now();

    // Enumerate exact codes up to N=8 (N=8 is the practical limit)
    let max_seed = target_n.min(8);
    eprintln!("  Enumerating seed codes N=4..{}...", max_seed);
    let seeds = gather_seed_codes(max_seed);
    eprintln!("  Total seed codes: {}", seeds.len());

    if max_seed >= target_n {
        return seeds.into_iter().filter(|c| c.n == target_n).collect();
    }

    // Group seeds by N
    let mut by_n: HashMap<usize, Vec<DoublyEvenCode>> = HashMap::new();
    for code in seeds {
        by_n.entry(code.n).or_default().push(code);
    }

    // For each level from max_seed+1 to target_n, extend the previous level
    let mut current_level: Vec<DoublyEvenCode> = Vec::new();

    // Start with all seeds at max_seed
    if let Some(top_seeds) = by_n.get(&max_seed) {
        current_level = top_seeds.clone();
    }

    // Also extend seeds from lower N levels up to max_seed to catch different paths
    for n in 4..max_seed {
        if let Some(lower_seeds) = by_n.get(&n) {
            // Extend these all the way up to max_seed
            let mut level = lower_seeds.clone();
            for extend_to in (n + 1)..=max_seed {
                level = extend_level(&level, max_per_level);
                if level.is_empty() {
                    break;
                }
                eprintln!(
                    "    Extending N={} seeds to N={}: {} codes",
                    n,
                    extend_to,
                    level.len()
                );
            }
            current_level.extend(level);
        }
    }

    // Dedup current_level by RREF
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    current_level.retain(|c| {
        let r = baselines::rref(&c.generators);
        seen.insert(r)
    });

    eprintln!(
        "  Starting chain extension from {} codes at N={} [{:?}]",
        current_level.len(),
        max_seed,
        start.elapsed()
    );

    // Now extend from max_seed+1 to target_n
    for n in (max_seed + 1)..=target_n {
        let phase_start = Instant::now();
        current_level = extend_level(&current_level, max_per_level);
        eprintln!(
            "  N={}: {} codes [{:?}]",
            n,
            current_level.len(),
            phase_start.elapsed()
        );
        if current_level.is_empty() {
            eprintln!("  Chain extension exhausted at N={}", n);
            break;
        }
    }

    current_level
}

pub fn search(config: &SearchConfig) {
    let total_start = Instant::now();
    let mut state = SearchState::new(config.target_n);

    println!("=== Doubly-Even Code Search at N={} ===", config.target_n);
    println!();

    // Phase 1: Chain extension from enumerated seeds
    // Skip for N >= 12 because N=8 enumeration takes ~2.5 min and chain
    // extension only produces trivial zero-column extensions anyway.
    if config.target_n <= 11 {
        eprintln!("Phase 1: Chain extension to N={}...", config.target_n);
        let phase1_start = Instant::now();
        let chain_codes = chain_extend(config.target_n, 500);
        let mut phase1_new = 0usize;
        for code in chain_codes {
            if state.try_add(code) {
                phase1_new += 1;
            }
        }
        eprintln!(
            "Phase 1 complete: {} new distinct classes [{:?}]",
            phase1_new,
            phase1_start.elapsed()
        );
    } else {
        eprintln!("Phase 1: Skipping chain extension for N={} (too expensive)", config.target_n);
    }

    // Phase 2: Random valid code generation
    eprintln!(
        "Phase 2: Random generation ({} attempts)...",
        config.random_batch_size
    );
    let phase2_start = Instant::now();
    let random_codes = baselines::random_batch(
        config.target_n,
        config.random_batch_size,
        config.seed,
    );
    let mut phase2_new = 0usize;
    for code in random_codes {
        if state.try_add(code) {
            phase2_new += 1;
        }
    }
    eprintln!(
        "Phase 2 complete: {} new distinct classes [{:?}]",
        phase2_new,
        phase2_start.elapsed()
    );

    // Phase 3: Additional random batches with different seeds
    // At high N, random generation is far more efficient than evolution
    // because evolution's archive-scan is O(pop * archive) per generation.
    let num_extra_batches = 8;
    for batch in 0..num_extra_batches {
        let batch_seed = config.seed.wrapping_add(1000 * (batch as u64 + 1));
        eprintln!(
            "Phase 3.{}: Random batch (seed={}, {} attempts)...",
            batch, batch_seed, config.random_batch_size
        );
        let batch_start = Instant::now();
        let codes = baselines::random_batch(
            config.target_n,
            config.random_batch_size,
            batch_seed,
        );
        let mut batch_new = 0usize;
        for code in codes {
            if state.try_add(code) {
                batch_new += 1;
            }
        }
        eprintln!(
            "  Batch {}: {} new classes [{:?}] (total: {})",
            batch,
            batch_new,
            batch_start.elapsed(),
            state.count()
        );
    }

    // Phase 4: More random batches to saturate coverage
    let num_extra_batches2 = 8;
    for batch in 0..num_extra_batches2 {
        let batch_seed = config
            .seed
            .wrapping_add(20000 + 1000 * (batch as u64 + 1));
        eprintln!(
            "Phase 4.{}: Random batch (seed={}, {} attempts)...",
            batch, batch_seed, config.random_batch_size
        );
        let batch_start = Instant::now();
        let codes = baselines::random_batch(
            config.target_n,
            config.random_batch_size,
            batch_seed,
        );
        let mut batch_new = 0usize;
        for code in codes {
            if state.try_add(code) {
                batch_new += 1;
            }
        }
        eprintln!(
            "  Batch {}: {} new classes [{:?}] (total: {})",
            batch,
            batch_new,
            batch_start.elapsed(),
            state.count()
        );
    }

    // Results
    let total_elapsed = total_start.elapsed();
    println!();
    println!("=== RESULTS: N={} ===", config.target_n);
    println!("Total unique equivalence classes found: {}", state.count());
    println!("Total time: {:?}", total_elapsed);
    println!();

    // Classify by k
    let max_k = state.found.iter().map(|c| c.k()).max().unwrap_or(0);
    println!(
        "{:>5} | {:>8} | {:>14} | {:>11}",
        "k", "Count", "Indecomposable", "d_min range"
    );
    println!("{}", "-".repeat(50));

    for k in 0..=max_k {
        let at_k: Vec<&DoublyEvenCode> = state.found.iter().filter(|c| c.k() == k).collect();
        if at_k.is_empty() {
            continue;
        }
        let indecomp = at_k
            .iter()
            .filter(|c| c.k() > 1 && !is_decomposable(c))
            .count();
        let min_d = at_k.iter().map(|c| c.min_distance()).min().unwrap_or(0);
        let max_d = at_k.iter().map(|c| c.min_distance()).max().unwrap_or(0);
        println!(
            "{:>5} | {:>8} | {:>14} | {:>3}-{:<3}",
            k,
            at_k.len(),
            indecomp,
            min_d,
            max_d
        );
    }

    println!();

    // Print the most interesting finds: highest-k indecomposable codes
    let mut interesting: Vec<&DoublyEvenCode> = state
        .found
        .iter()
        .filter(|c| c.k() > 1 && !is_decomposable(c))
        .collect();
    interesting.sort_by(|a, b| b.k().cmp(&a.k()).then(b.min_distance().cmp(&a.min_distance())));
    interesting.truncate(20);

    if !interesting.is_empty() {
        println!("Top {} indecomposable codes:", interesting.len());
        println!();
        for (i, code) in interesting.iter().enumerate() {
            let inv = compute_invariants(code);
            let we_nonzero: Vec<String> = inv
                .weight_enumerator
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .map(|(w, count)| format!("{}:{}", w, count))
                .collect();
            println!(
                "  [{}] [{},{},{}] self_dual={} WE: {}",
                i,
                code.n,
                code.k(),
                code.min_distance(),
                inv.is_self_dual,
                we_nonzero.join(" ")
            );
            // Print generators
            for (j, &row) in code.generators.iter().enumerate() {
                let bits: String = (0..code.n)
                    .map(|col| if row & (1 << col) != 0 { '1' } else { '0' })
                    .collect();
                println!("       g{}: {}", j, bits);
            }
        }
    }

    // Summary of all found codes
    println!();
    println!("=== All codes by weight enumerator ===");
    let mut we_groups: HashMap<Vec<usize>, Vec<&DoublyEvenCode>> = HashMap::new();
    for code in &state.found {
        let we = code.weight_enumerator();
        we_groups.entry(we).or_default().push(code);
    }
    let mut we_keys: Vec<Vec<usize>> = we_groups.keys().cloned().collect();
    we_keys.sort();
    for we in &we_keys {
        let codes = &we_groups[we];
        let we_str: Vec<String> = we
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(w, count)| format!("A_{}={}", w, count))
            .collect();
        let k = codes[0].k();
        let d = codes[0].min_distance();
        println!(
            "  [{},{},{}] x{} classes: {}",
            config.target_n,
            k,
            d,
            codes.len(),
            we_str.join(", ")
        );
    }
}

/// Construct all zero-column embeddings into a target length.
///
/// A doubly-even code of length m < target_n can be embedded into length target_n
/// by placing the code in the low m bit positions and leaving the upper target_n - m
/// positions as zero columns. Under column permutation equivalence, the choice of
/// positions is irrelevant, so each equivalence class at length m yields exactly one
/// equivalence class at length target_n.
///
/// Doubly-even codes exist at all lengths >= 4, not just multiples of 4.
/// For example, [6,2,4] and [7,3,4] are valid doubly-even codes without
/// zero columns. This function checks ALL sub-lengths from 4 to target_n-1.
pub fn zero_column_embeddings(target_n: usize) -> Vec<DoublyEvenCode> {
    let mut result = Vec::new();

    for m in 4..target_n {
        let start = Instant::now();
        let mut all_codes: Vec<DoublyEvenCode> = Vec::new();

        // For m < 12, enumerate_codes is tractable; for m >= 12 skip it
        if m < 12 {
            let enumerated = enumerate_codes(m);
            all_codes.extend(enumerated);
        }

        // Supplement with random sampling using multiple seeds
        let batch_count: usize = if m <= 8 { 5_000 } else { 10_000 };
        let num_seeds: usize = if m <= 8 { 5 } else { 20 };
        for seed_idx in 0..num_seeds {
            let seed = 77_777u64
                .wrapping_mul(m as u64)
                .wrapping_add(seed_idx as u64 * 7919);
            let batch = crate::baselines::random_batch(m, batch_count, seed);
            all_codes.extend(batch);
        }

        // Deduplicate using nauty canonical keys
        let mut seen_keys: HashSet<Vec<u64>> = HashSet::new();
        let mut nontrivial: Vec<DoublyEvenCode> = Vec::new();
        for code in all_codes {
            let code = code.normalize();
            if code.k() == 0 || !code.is_doubly_even() {
                continue;
            }
            let key = crate::nauty_canonical::exact_canonical_key(&code);
            if seen_keys.insert(key) {
                nontrivial.push(code);
            }
        }

        eprintln!(
            "  Zero-column embeddings from m={}: {} classes [{:?}]",
            m,
            nontrivial.len(),
            start.elapsed()
        );
        for code in nontrivial {
            result.push(DoublyEvenCode::new(target_n, code.generators));
        }
    }

    result
}

/// Saturation test: stress-test random sampling to determine whether the known
/// class count is the true total or merely a lower bound.
///
/// Runs up to `max_batches` batches of `random_batch(n, batch_size, seed)` with
/// incrementing seeds. Tracks when each new class is discovered and stops early
/// if 50 consecutive batches produce zero new classes.
pub fn saturate(n: usize, batch_size: usize, max_batches: usize) {
    let total_start = Instant::now();
    let mut state = SearchState::new(n);

    let dry_limit: usize = 50;
    let mut dry_streak: usize = 0;
    let mut last_new_batch: usize = 0;
    let mut discovery_curve: Vec<(usize, usize, usize)> = Vec::new(); // (batch, cumulative_attempts, total_classes)

    println!("=== Saturation Test: N={} ===", n);
    println!(
        "batch_size={}, max_batches={}, dry_limit={}",
        batch_size, max_batches, dry_limit
    );
    println!();

    // Phase 0: inject zero-column embeddings from shorter lengths
    eprintln!("Phase 0: Zero-column embeddings into N={}...", n);
    let phase0_start = Instant::now();
    let embeddings = zero_column_embeddings(n);
    let mut phase0_new: usize = 0;
    for code in embeddings {
        if state.try_add(code) {
            phase0_new += 1;
        }
    }
    eprintln!(
        "Phase 0 complete: {} new classes from zero-column embeddings [{:?}]",
        phase0_new,
        phase0_start.elapsed()
    );

    for batch_idx in 0..max_batches {
        let seed = 100_000u64 + (batch_idx as u64) * 7919; // distinct seed per batch
        let codes = baselines::random_batch(n, batch_size, seed);

        let mut new_this_batch: usize = 0;
        for code in codes {
            if state.try_add(code) {
                new_this_batch += 1;
            }
        }

        if new_this_batch > 0 {
            dry_streak = 0;
            last_new_batch = batch_idx;
        } else {
            dry_streak += 1;
        }

        let cumulative_attempts = (batch_idx + 1) * batch_size;
        discovery_curve.push((batch_idx, cumulative_attempts, state.count()));

        eprintln!(
            "batch {:>4} | +{:<3} new | total {:>4} | dry streak {:>3} | [{:?}]",
            batch_idx,
            new_this_batch,
            state.count(),
            dry_streak,
            total_start.elapsed()
        );

        if dry_streak >= dry_limit {
            eprintln!(
                "Stopping: {} consecutive dry batches (no new classes)",
                dry_limit
            );
            break;
        }
    }

    // --- Summary ---
    let total_batches = discovery_curve.len();
    let total_attempts = total_batches * batch_size;
    let elapsed = total_start.elapsed();

    println!();
    println!("=== SATURATION RESULTS: N={} ===", n);
    println!("Total equivalence classes found: {}", state.count());
    println!("Total attempts: {} ({} batches x {})", total_attempts, total_batches, batch_size);
    println!("Last new class found in batch: {}", last_new_batch);
    println!("Total time: {:?}", elapsed);
    println!();

    // k-distribution
    let max_k = state.found.iter().map(|c| c.k()).max().unwrap_or(0);
    println!("k-distribution:");
    println!("{:>5} | {:>8} | {:>14}", "k", "Count", "Indecomposable");
    println!("{}", "-".repeat(35));
    for k in 1..=max_k {
        let at_k: Vec<&DoublyEvenCode> = state.found.iter().filter(|c| c.k() == k).collect();
        if at_k.is_empty() {
            continue;
        }
        let indecomp = at_k
            .iter()
            .filter(|c| c.k() > 1 && !is_decomposable(c))
            .count();
        println!("{:>5} | {:>8} | {:>14}", k, at_k.len(), indecomp);
    }

    // Discovery curve (every 10th batch)
    println!();
    println!("Discovery curve (every 10th batch):");
    println!("{:>6} | {:>12} | {:>8}", "Batch", "Attempts", "Classes");
    println!("{}", "-".repeat(35));
    for &(batch, attempts, classes) in &discovery_curve {
        if batch % 10 == 0 || batch == total_batches - 1 {
            println!("{:>6} | {:>12} | {:>8}", batch, attempts, classes);
        }
    }

    // Save all discovered codes to JSON
    let output_path = format!("adinkra_codes_n{}_saturate.json", n);
    save_codes_to_json(&state.found, n, &output_path);
    println!();
    println!("All {} codes saved to {}", state.count(), output_path);
}

// ---------------------------------------------------------------------------
// Serialization: save all discovered codes with full metadata
// ---------------------------------------------------------------------------

fn save_codes_to_json(codes: &[DoublyEvenCode], n: usize, output_path: &str) {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path).expect("Failed to create output file");

    writeln!(file, "{{").unwrap();
    writeln!(file, "  \"n\": {},", n).unwrap();
    writeln!(file, "  \"total_classes\": {},", codes.len()).unwrap();
    writeln!(file, "  \"codes\": [").unwrap();

    for (i, code) in codes.iter().enumerate() {
        let inv = crate::canonical::compute_invariants(code);
        let codewords = code.all_codewords();
        let canonical_key = crate::nauty_canonical::exact_canonical_key(code);

        // Generator matrix as binary strings
        let gen_strings: Vec<String> = code.generators.iter().map(|&row| {
            (0..code.n).map(|col| if row & (1 << col) != 0 { '1' } else { '0' }).collect()
        }).collect();

        // Generator matrix as hex values
        let gen_hex: Vec<String> = code.generators.iter().map(|&row| format!("0x{:x}", row)).collect();

        // All codewords as hex
        let cw_hex: Vec<String> = codewords.iter().map(|&cw| format!("0x{:x}", cw)).collect();

        // Weight distribution: only nonzero entries
        let weight_dist: Vec<(usize, usize)> = inv.weight_enumerator.iter().enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(w, c)| (w, *c))
            .collect();

        // Number of zero columns
        let mut col_used = vec![false; code.n];
        for &cw in &codewords {
            for col in 0..code.n {
                if cw & (1 << col) != 0 {
                    col_used[col] = true;
                }
            }
        }
        let zero_columns = col_used.iter().filter(|&&u| !u).count();

        // Column weight profile (how many codewords have a 1 in each column)
        let mut col_weights: Vec<usize> = vec![0; code.n];
        for &cw in &codewords {
            for col in 0..code.n {
                if cw & (1 << col) != 0 {
                    col_weights[col] += 1;
                }
            }
        }
        col_weights.sort();

        let comma = if i + 1 < codes.len() { "," } else { "" };

        writeln!(file, "    {{").unwrap();
        writeln!(file, "      \"index\": {},", i).unwrap();
        writeln!(file, "      \"n\": {},", code.n).unwrap();
        writeln!(file, "      \"k\": {},", code.k()).unwrap();
        writeln!(file, "      \"num_codewords\": {},", codewords.len()).unwrap();
        writeln!(file, "      \"min_distance\": {},", inv.min_distance).unwrap();
        writeln!(file, "      \"is_self_orthogonal\": {},", inv.is_self_orthogonal).unwrap();
        writeln!(file, "      \"is_self_dual\": {},", inv.is_self_dual).unwrap();
        writeln!(file, "      \"is_indecomposable\": {},", inv.is_indecomposable).unwrap();
        writeln!(file, "      \"automorphism_group_size\": {},", inv.automorphism_group_size).unwrap();
        writeln!(file, "      \"zero_columns\": {},", zero_columns).unwrap();
        writeln!(file, "      \"generators_binary\": {:?},", gen_strings).unwrap();
        writeln!(file, "      \"generators_hex\": {:?},", gen_hex).unwrap();
        writeln!(file, "      \"generators_raw\": {:?},", code.generators).unwrap();
        writeln!(file, "      \"weight_distribution\": {:?},", weight_dist).unwrap();
        writeln!(file, "      \"weight_enumerator_full\": {:?},", inv.weight_enumerator).unwrap();
        writeln!(file, "      \"column_weight_profile\": {:?},", col_weights).unwrap();
        writeln!(file, "      \"all_codewords_hex\": {:?},", cw_hex).unwrap();
        writeln!(file, "      \"canonical_key\": {:?}", canonical_key).unwrap();
        writeln!(file, "    }}{}", comma).unwrap();
    }

    writeln!(file, "  ]").unwrap();
    writeln!(file, "}}").unwrap();

    eprintln!("Saved {} codes to {}", codes.len(), output_path);
}

// ---------------------------------------------------------------------------
// validate-miller: compare our counts against published literature
// ---------------------------------------------------------------------------

/// Return the known (k, count) pairs for a given N, if available.
fn miller_reference(n: usize) -> Option<&'static [(usize, usize)]> {
    match n {
        4 => Some(&MILLER_N4),
        8 => Some(&MILLER_N8),
        12 => Some(&MILLER_N12),
        16 => Some(&MILLER_N16),
        _ => None,
    }
}

/// Validate our enumeration against known counts from the literature.
///
/// Runs zero-column embeddings (shorter codes embedded into length N) plus
/// aggressive random batching to saturate coverage, then compares the per-k
/// class counts against the Doran-Faux-Gates / Miller reference values.
pub fn validate_miller(n: usize) {
    let reference = match miller_reference(n) {
        Some(r) => r,
        None => {
            eprintln!(
                "No reference data for N={}. Available: N=4, N=8, N=12, N=16.",
                n
            );
            std::process::exit(1);
        }
    };

    let total_start = Instant::now();
    let mut state = SearchState::new(n);

    println!("=== Miller/Doran-Faux-Gates Validation: N={} ===", n);
    println!();
    println!(
        "Reference: arXiv:0806.0050 Table 4 (permutation equivalence classes"
    );
    println!("of doubly-even codes, including zero-column embeddings).");
    println!();

    // Phase 0: zero-column embeddings from shorter codes
    eprintln!("Phase 0: Zero-column embeddings into N={}...", n);
    let phase0_start = Instant::now();
    let embeddings = zero_column_embeddings(n);
    let mut phase0_new: usize = 0;
    for code in embeddings {
        if state.try_add(code) {
            phase0_new += 1;
        }
    }
    eprintln!(
        "Phase 0 complete: {} new classes from zero-column embeddings [{:?}]",
        phase0_new,
        phase0_start.elapsed()
    );

    // Phase 1: aggressive random batching to saturate
    // Use large batches and many seeds to maximize coverage.
    let batch_size: usize = if n <= 8 { 2_000 } else { 10_000 };
    let num_batches: usize = if n <= 8 { 50 } else { 200 };
    let dry_limit: usize = if n <= 8 { 20 } else { 50 };
    let mut dry_streak: usize = 0;

    eprintln!(
        "Phase 1: Random batching ({} batches x {} attempts, dry limit {})...",
        num_batches, batch_size, dry_limit
    );

    for batch_idx in 0..num_batches {
        let seed = 500_000u64 + (batch_idx as u64) * 7919;
        let codes = baselines::random_batch(n, batch_size, seed);

        let mut new_this_batch: usize = 0;
        for code in codes {
            if state.try_add(code) {
                new_this_batch += 1;
            }
        }

        if new_this_batch > 0 {
            dry_streak = 0;
        } else {
            dry_streak += 1;
        }

        if batch_idx % 20 == 0 || new_this_batch > 0 {
            eprintln!(
                "  batch {:>4} | +{:<3} new | total {:>4} | dry {:>3}",
                batch_idx, new_this_batch, state.count(), dry_streak
            );
        }

        if dry_streak >= dry_limit {
            eprintln!(
                "  Saturated after {} consecutive dry batches.",
                dry_limit
            );
            break;
        }
    }

    // Phase 2: systematic extension from known codes
    // For each k where we might be short, try extending every (k-1)-code
    // by one generator. This is exhaustive: it finds every k-code that
    // contains a known (k-1)-code as a subcode.
    eprintln!("Phase 2: Systematic extension from known codes...");
    let phase2_start = Instant::now();
    let max_k = state.found.iter().map(|c| c.k()).max().unwrap_or(0);
    let mut total_extended = 0usize;
    for target_k in 2..=max_k + 1 {
        let ext_start = Instant::now();
        let new = state.extend_from_k(target_k);
        if new > 0 {
            eprintln!(
                "  k={}: +{} new classes [{:?}] (total: {})",
                target_k, new, ext_start.elapsed(), state.count()
            );
            total_extended += new;
        }
    }
    eprintln!(
        "Phase 2 complete: {} new classes from extension [{:?}]",
        total_extended, phase2_start.elapsed()
    );

    let elapsed = total_start.elapsed();
    println!("Search complete: {} classes found in {:?}", state.count(), elapsed);
    println!();

    // Build per-k counts from our results
    let mut our_counts: HashMap<usize, usize> = HashMap::new();
    for code in &state.found {
        *our_counts.entry(code.k()).or_insert(0) += 1;
    }

    // Compare against reference
    let expected_total: usize = reference.iter().map(|&(_, c)| c).sum();
    let our_total: usize = our_counts.values().sum();

    println!(
        "{:>5} | {:>10} | {:>8} | {:>8}",
        "k", "Expected", "Found", "Status"
    );
    println!("{}", "-".repeat(42));

    let mut all_pass = true;

    for &(k, expected) in reference {
        let found = our_counts.get(&k).copied().unwrap_or(0);
        let status = if found == expected {
            "PASS"
        } else if found < expected {
            all_pass = false;
            "FAIL (low)"
        } else {
            // found > expected would be very surprising; may indicate a
            // canonicalization bug or a reference error
            all_pass = false;
            "FAIL (high)"
        };
        println!(
            "{:>5} | {:>10} | {:>8} | {}",
            k, expected, found, status
        );
    }

    // Check for unexpected k values in our results not in the reference
    let ref_ks: HashSet<usize> = reference.iter().map(|&(k, _)| k).collect();
    for (&k, &count) in &our_counts {
        if !ref_ks.contains(&k) && count > 0 {
            println!(
                "{:>5} | {:>10} | {:>8} | UNEXPECTED",
                k, "-", count
            );
            all_pass = false;
        }
    }

    println!("{}", "-".repeat(42));
    println!(
        "{:>5} | {:>10} | {:>8} | {}",
        "TOTAL",
        expected_total,
        our_total,
        if our_total == expected_total { "PASS" } else { "FAIL" }
    );

    // Save all discovered codes to JSON
    let output_path = format!("adinkra_codes_n{}.json", n);
    save_codes_to_json(&state.found, n, &output_path);

    println!();
    if all_pass && our_total == expected_total {
        println!("RESULT: ALL CHECKS PASSED for N={}", n);
        println!("All {} codes saved to {}", our_total, output_path);
    } else {
        println!("RESULT: VALIDATION FAILED for N={}", n);
        println!("Found {} codes saved to {}", our_total, output_path);
        if our_total < expected_total {
            println!(
                "  Missing {} classes. Random search may need more iterations,",
                expected_total - our_total
            );
            println!("  or there may be classes unreachable by the current sampler.");
        } else if our_total > expected_total {
            println!(
                "  Found {} extra classes beyond reference. Possible canonicalization",
                our_total - expected_total
            );
            println!("  bug (two equivalent codes getting different canonical forms).");
        }
        std::process::exit(1);
    }
}
