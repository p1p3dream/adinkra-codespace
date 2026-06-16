use crate::baselines::{self, EvolutionConfig};
use crate::canonical::{compute_invariants, is_decomposable};
use crate::code::{enumerate_codes, DoublyEvenCode};

use std::collections::{HashMap, HashSet};
use std::time::Instant;

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

/// Fast canonical form for high-N codes: sort columns by codeword weight profile,
/// then RREF. This is O(2^k * n + n log n + k^2 * n) instead of the exponential
/// permutation search. It's a heuristic: two truly equivalent codes might get
/// different forms if they have identical column weight profiles but different
/// higher-order structure. The result is an upper bound on equivalence classes.
fn fast_canonical(code: &DoublyEvenCode) -> Vec<u32> {
    let n = code.n;
    if code.k() == 0 || n == 0 {
        return vec![];
    }

    let codewords = code.all_codewords();

    // Compute column weight for each position
    let mut col_weights = vec![0usize; n];
    for &cw in &codewords {
        for j in 0..n {
            if cw & (1 << j) != 0 {
                col_weights[j] += 1;
            }
        }
    }

    // Sort columns by weight (ascending)
    let mut cols_sorted: Vec<(usize, usize)> = col_weights
        .iter()
        .copied()
        .enumerate()
        .map(|(i, w)| (w, i))
        .collect();
    cols_sorted.sort();

    // Build the permutation: perm[dest] = src
    let perm: Vec<usize> = cols_sorted.iter().map(|&(_, c)| c).collect();

    // Apply permutation to generators
    let mut permuted: Vec<u32> = code
        .generators
        .iter()
        .map(|&row| {
            let mut out = 0u32;
            for (dest, &src) in perm.iter().enumerate() {
                if row & (1 << src) != 0 {
                    out |= 1 << dest;
                }
            }
            out
        })
        .collect();

    // RREF
    let k = permuted.len();
    let mut pivot_row = 0;
    for col in 0..32u32 {
        if pivot_row >= k {
            break;
        }
        let mut found = None;
        for r in pivot_row..k {
            if permuted[r] & (1 << col) != 0 {
                found = Some(r);
                break;
            }
        }
        let Some(r) = found else { continue };
        permuted.swap(pivot_row, r);
        let pivot_val = permuted[pivot_row];
        for r in 0..k {
            if r != pivot_row && permuted[r] & (1 << col) != 0 {
                permuted[r] ^= pivot_val;
            }
        }
        pivot_row += 1;
    }
    permuted.retain(|&r| r != 0);
    permuted.sort();
    permuted
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
}
