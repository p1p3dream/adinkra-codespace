/// Leave-one-N-out evaluation framework for doubly-even code generation baselines.
///
/// The experiment: enumerate all valid doubly-even codes for N=1..10, hold out one
/// value of N, and test whether baseline generators can recover the held-out codes
/// using only knowledge of codes at other values of N.

use crate::baselines;
use crate::canonical::{canonical_form, deduplicate, is_decomposable};
use crate::code::{enumerate_codes, DoublyEvenCode};

use std::collections::HashSet;
use std::time::Instant;

/// Results from evaluating a single baseline on a single held-out N.
pub struct EvalResult {
    pub held_out_n: usize,
    pub total_known_classes: usize,
    pub baseline_name: String,
    pub candidates_generated: usize,
    pub valid_candidates: usize,
    pub unique_after_dedup: usize,
    pub known_classes_recovered: usize,
    pub recovery_rate: f64,
    pub novel_candidates: usize,
    pub trivial_extensions: usize,
    pub indecomposable_novel: usize,
}

/// Collect all training codes (codes at lengths strictly below held_out_n).
fn gather_training_codes(held_out_n: usize) -> Vec<DoublyEvenCode> {
    let mut training = Vec::new();
    for n in 1..held_out_n {
        eprintln!("  Enumerating training codes for N={}...", n);
        let start = Instant::now();
        let codes = enumerate_codes(n);
        let raw_count = codes.len();
        let deduped = deduplicate(codes);
        let elapsed = start.elapsed();
        eprintln!(
            "    N={}: {} raw, {} unique classes ({:?})",
            n,
            raw_count,
            deduped.len(),
            elapsed
        );
        // Only include nontrivial codes (k > 0) as training data
        for code in deduped {
            if code.k() > 0 {
                training.push(code);
            }
        }
    }
    training
}

/// Run a single baseline and evaluate its output against ground truth.
fn run_baseline(
    baseline_name: &str,
    candidates: Vec<DoublyEvenCode>,
    ground_truth_canonical: &HashSet<Vec<u32>>,
    held_out_n: usize,
    total_known_classes: usize,
) -> EvalResult {
    let candidates_generated = candidates.len();

    // Filter to valid codes at the right length
    let valid: Vec<DoublyEvenCode> = candidates
        .into_iter()
        .filter(|c| c.n == held_out_n && c.is_doubly_even())
        .collect();
    let valid_candidates = valid.len();

    // Deduplicate by equivalence class
    let unique = deduplicate(valid);
    let unique_after_dedup = unique.len();

    // Compare against ground truth
    let mut recovered = 0usize;
    let mut novel = Vec::new();
    for code in &unique {
        let cf = canonical_form(code);
        if ground_truth_canonical.contains(&cf) {
            recovered += 1;
        } else {
            novel.push(code.clone());
        }
    }

    // Classify novel candidates
    let mut trivial_extensions = 0usize;
    let mut indecomposable_novel = 0usize;
    for code in &novel {
        if is_decomposable(code) {
            trivial_extensions += 1;
        } else {
            indecomposable_novel += 1;
        }
    }

    let recovery_rate = if total_known_classes > 0 {
        recovered as f64 / total_known_classes as f64
    } else {
        0.0
    };

    EvalResult {
        held_out_n,
        total_known_classes,
        baseline_name: baseline_name.to_string(),
        candidates_generated,
        valid_candidates,
        unique_after_dedup,
        known_classes_recovered: recovered,
        recovery_rate,
        novel_candidates: novel.len(),
        trivial_extensions,
        indecomposable_novel,
    }
}

/// Run the full leave-one-N-out evaluation for a specific held-out N.
///
/// Runs all baselines and returns one EvalResult per baseline.
pub fn evaluate_held_out(held_out_n: usize, max_candidates_per_baseline: usize) -> Vec<EvalResult> {
    eprintln!(
        "=== Evaluating held-out N={} (max {} candidates/baseline) ===",
        held_out_n, max_candidates_per_baseline
    );

    // Step 1: Enumerate ground truth at held_out_n
    eprintln!("Enumerating ground truth for N={}...", held_out_n);
    let gt_start = Instant::now();
    let ground_truth_raw = enumerate_codes(held_out_n);
    let raw_count = ground_truth_raw.len();
    let ground_truth = deduplicate(ground_truth_raw);
    let gt_elapsed = gt_start.elapsed();

    // Count only nontrivial classes (k > 0)
    let nontrivial_gt: Vec<&DoublyEvenCode> = ground_truth.iter().filter(|c| c.k() > 0).collect();
    let total_known_classes = nontrivial_gt.len();
    eprintln!(
        "Ground truth N={}: {} raw codes, {} equivalence classes ({} nontrivial) in {:?}",
        held_out_n,
        raw_count,
        ground_truth.len(),
        total_known_classes,
        gt_elapsed
    );

    // Build canonical form set for fast lookup
    let gt_canonical: HashSet<Vec<u32>> = nontrivial_gt
        .iter()
        .map(|c| canonical_form(c))
        .collect();

    // Step 2: Gather training codes
    eprintln!("Gathering training codes (all N != {})...", held_out_n);
    let train_start = Instant::now();
    let training_codes = gather_training_codes(held_out_n);
    let train_elapsed = train_start.elapsed();
    eprintln!(
        "Training set: {} nontrivial codes across all other N values ({:?})",
        training_codes.len(),
        train_elapsed
    );

    let mut results = Vec::new();

    // ---- Baseline 1: Random sampling ----
    {
        eprintln!("Running baseline: random_batch...");
        let start = Instant::now();
        let candidates = baselines::random_batch(held_out_n, max_candidates_per_baseline, 42);
        let elapsed = start.elapsed();
        eprintln!(
            "  random_batch: generated {} candidates in {:?}",
            candidates.len(),
            elapsed
        );
        let result = run_baseline(
            "random_batch",
            candidates,
            &gt_canonical,
            held_out_n,
            total_known_classes,
        );
        results.push(result);
    }

    // ---- Baseline 2: Direct-sum extension ----
    {
        eprintln!("Running baseline: direct_sum_extend...");
        let start = Instant::now();
        let candidates = baselines::all_direct_sum_extensions(&training_codes, held_out_n);
        let elapsed = start.elapsed();
        eprintln!(
            "  direct_sum_extend: generated {} candidates in {:?}",
            candidates.len(),
            elapsed
        );
        let result = run_baseline(
            "direct_sum_extend",
            candidates,
            &gt_canonical,
            held_out_n,
            total_known_classes,
        );
        results.push(result);
    }

    // ---- Baseline 3: Column extension (extend all N-1 codes by one column) ----
    {
        eprintln!("Running baseline: extend_by_one_column...");
        let start = Instant::now();
        let n_minus_1_codes: Vec<&DoublyEvenCode> = training_codes
            .iter()
            .filter(|c| c.n == held_out_n - 1)
            .collect();
        let mut candidates = Vec::new();
        for code in &n_minus_1_codes {
            let extensions = baselines::extend_by_one_column(code);
            candidates.extend(extensions);
            if candidates.len() >= max_candidates_per_baseline {
                candidates.truncate(max_candidates_per_baseline);
                break;
            }
        }
        let elapsed = start.elapsed();
        eprintln!(
            "  extend_by_one_column: generated {} candidates from {} base codes in {:?}",
            candidates.len(),
            n_minus_1_codes.len(),
            elapsed
        );
        let result = run_baseline(
            "extend_by_one_column",
            candidates,
            &gt_canonical,
            held_out_n,
            total_known_classes,
        );
        results.push(result);
    }

    // ---- Baseline 4: Evolutionary search ----
    {
        eprintln!("Running baseline: evolve...");
        let start = Instant::now();
        // Seed evolution with training codes near the target length
        let seed_pop: Vec<DoublyEvenCode> = training_codes
            .iter()
            .filter(|c| c.n >= held_out_n.saturating_sub(2) && c.n <= held_out_n)
            .cloned()
            .collect();
        let config = baselines::EvolutionConfig {
            population_size: max_candidates_per_baseline.min(200),
            generations: 100,
            mutation_rate: 0.8,
            seed: 12345,
        };
        let candidates = baselines::evolve(seed_pop, held_out_n, &config);
        let elapsed = start.elapsed();
        eprintln!(
            "  evolve: generated {} candidates in {:?}",
            candidates.len(),
            elapsed
        );
        let result = run_baseline(
            "evolve",
            candidates,
            &gt_canonical,
            held_out_n,
            total_known_classes,
        );
        results.push(result);
    }

    results
}

/// Print a summary table of evaluation results.
pub fn print_results(results: &[EvalResult]) {
    if results.is_empty() {
        println!("No results to display.");
        return;
    }

    // Header
    println!(
        "{:<8} {:<22} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8}",
        "N",
        "Baseline",
        "GenCand",
        "Valid",
        "Unique",
        "Known",
        "Recovery",
        "Novel",
        "Trivial",
        "Indecomp"
    );
    println!("{}", "-".repeat(106));

    for r in results {
        println!(
            "{:<8} {:<22} {:>8} {:>8} {:>8} {:>5}/{:<2} {:>9.1}% {:>8} {:>8} {:>8}",
            r.held_out_n,
            r.baseline_name,
            r.candidates_generated,
            r.valid_candidates,
            r.unique_after_dedup,
            r.known_classes_recovered,
            r.total_known_classes,
            r.recovery_rate * 100.0,
            r.novel_candidates,
            r.trivial_extensions,
            r.indecomposable_novel,
        );
    }
    println!();

    // Summary per N
    let mut ns: Vec<usize> = results.iter().map(|r| r.held_out_n).collect();
    ns.sort();
    ns.dedup();
    if ns.len() > 1 || results.len() > 1 {
        println!("=== Per-N Summary ===");
        for n in &ns {
            let n_results: Vec<&EvalResult> = results.iter().filter(|r| r.held_out_n == *n).collect();
            let best = n_results
                .iter()
                .max_by(|a, b| {
                    a.recovery_rate
                        .partial_cmp(&b.recovery_rate)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let total_novel: usize = n_results.iter().map(|r| r.novel_candidates).sum();
            let total_indecomp: usize = n_results.iter().map(|r| r.indecomposable_novel).sum();
            println!(
                "  N={}: best recovery = {:.1}% ({}), total novel across baselines = {}, indecomposable novel = {}",
                n,
                best.recovery_rate * 100.0,
                best.baseline_name,
                total_novel,
                total_indecomp,
            );
        }
    }
}
