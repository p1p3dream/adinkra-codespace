mod baselines;
mod canonical;
mod chromochar;
mod chromotopology;
mod code;
mod dashing;
mod decompose;
mod eval;
mod filters;
mod holoraumy;
mod lorentz;
mod lr_matrix;
mod nauty_canonical;
mod pipeline;
mod ranking;
mod search;
mod signed_perm;
mod tendim_data;

use std::time::Instant;

use canonical::{compute_invariants, deduplicate, is_decomposable};
use code::{enumerate_codes, DoublyEvenCode};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "enumerate" => cmd_enumerate(&args),
        "count" => cmd_count(&args),
        "eval" => cmd_eval(&args),
        "eval-all" => cmd_eval_all(&args),
        "invariants" => cmd_invariants(&args),
        "validate" => cmd_validate(),
        "search" => cmd_search(&args),
        "saturate" => cmd_saturate(&args),
        "validate-miller" => cmd_validate_miller(&args),
        "pipeline" => cmd_pipeline(&args),
        "pipeline-k" => cmd_pipeline_k(&args),
        "decompose-k" => cmd_decompose_k(&args),
        "help" | "--help" | "-h" => print_usage(&args[0]),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }
}

fn print_usage(prog: &str) {
    eprintln!("Usage: {} <command> [args]", prog);
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  enumerate <n>           Enumerate and print all doubly-even codes of length n");
    eprintln!("  count <max_n>           Count equivalence classes for n=1 to max_n");
    eprintln!("  eval <held_out_n>       Run leave-one-N-out evaluation for a specific N");
    eprintln!("  eval-all [max_cand]     Run evaluation for all N from 4 to 10");
    eprintln!("  invariants <n>          Print invariants for all codes of length n");
    eprintln!("  validate                Self-test: enumerate N=4..8, verify known results");
    eprintln!("  search [n] [pop] [gen]  Search for doubly-even codes at N (default 16)");
    eprintln!("  saturate [n] [batch_size] [max_batches]");
    eprintln!("                          Saturation test at N (defaults: 16, 5000, 500)");
    eprintln!("  validate-miller [n]     Compare counts against Miller/Doran-Faux-Gates");
    eprintln!("                          reference (available: N=4, N=8, N=12, N=16)");
    eprintln!("  pipeline <json>         Run the full dimensional lifting pipeline");
    eprintln!("  pipeline-k <k> [json]   Run pipeline for a single k-stratum only");
    eprintln!("  decompose-k <k> [json]  Irreducible-decompose a single k-stratum (F8 route b)");
    eprintln!("                          and compute the dense gadget on irreducible pieces");
    eprintln!("  help                    Print this help message");
}

// ---------------------------------------------------------------------------
// enumerate
// ---------------------------------------------------------------------------

fn cmd_enumerate(args: &[String]) {
    let n = parse_usize_arg(args, 2, "enumerate <n>");

    eprintln!("Enumerating all doubly-even codes of length {}...", n);
    let start = Instant::now();
    let codes = enumerate_codes(n);
    let elapsed = start.elapsed();
    eprintln!("Found {} codes (before dedup) in {:?}", codes.len(), elapsed);

    let start = Instant::now();
    let unique = deduplicate(codes);
    let elapsed = start.elapsed();
    eprintln!(
        "Deduplicated to {} equivalence classes in {:?}",
        unique.len(),
        elapsed
    );

    println!();
    println!(
        "Doubly-even codes of length {} ({} equivalence classes):",
        n,
        unique.len()
    );
    println!();

    for (i, code) in unique.iter().enumerate() {
        let d = code.min_distance();
        let decomp = if code.k() > 1 {
            if is_decomposable(code) {
                " [decomposable]"
            } else {
                " [indecomposable]"
            }
        } else {
            ""
        };
        println!(
            "  [{}] [{},{},{}]{}",
            i,
            code.n,
            code.k(),
            d,
            decomp
        );
        for (j, &row) in code.generators.iter().enumerate() {
            let bits: String = (0..n)
                .map(|col| if row & (1 << col) != 0 { '1' } else { '0' })
                .collect();
            println!("       g{}: {}", j, bits);
        }
    }
}

// ---------------------------------------------------------------------------
// count
// ---------------------------------------------------------------------------

fn cmd_count(args: &[String]) {
    let max_n = parse_usize_arg(args, 2, "count <max_n>");

    println!(
        "{:>3} | {:>12} | {:>12} | {:>10} | {:>5} | {:>13}",
        "N", "Raw codes", "Equiv classes", "Nontrivial", "Max k", "Indecomposable"
    );
    println!("{}", "-".repeat(72));

    for n in 1..=max_n {
        let start = Instant::now();
        let codes = enumerate_codes(n);
        let raw_count = codes.len();
        let unique = deduplicate(codes);
        let elapsed = start.elapsed();

        let nontrivial: Vec<&DoublyEvenCode> = unique.iter().filter(|c| c.k() > 0).collect();
        let max_k = unique.iter().map(|c| c.k()).max().unwrap_or(0);
        let indecomposable = nontrivial
            .iter()
            .filter(|c| !is_decomposable(c))
            .count();

        println!(
            "{:>3} | {:>12} | {:>12} | {:>10} | {:>5} | {:>13}",
            n,
            raw_count,
            unique.len(),
            nontrivial.len(),
            max_k,
            indecomposable
        );
        eprintln!("  N={} completed in {:?}", n, elapsed);
    }
}

// ---------------------------------------------------------------------------
// eval
// ---------------------------------------------------------------------------

fn cmd_eval(args: &[String]) {
    let held_out_n = parse_usize_arg(args, 2, "eval <held_out_n>");
    let max_cand = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    let results = eval::evaluate_held_out(held_out_n, max_cand);
    println!();
    eval::print_results(&results);
}

// ---------------------------------------------------------------------------
// eval-all
// ---------------------------------------------------------------------------

fn cmd_eval_all(args: &[String]) {
    let max_cand = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    let mut all_results = Vec::new();
    for n in 4..=10 {
        eprintln!();
        let results = eval::evaluate_held_out(n, max_cand);
        all_results.extend(results);
    }

    println!();
    println!("========== FULL EVAL-ALL SUMMARY ==========");
    println!();
    eval::print_results(&all_results);
}

// ---------------------------------------------------------------------------
// invariants
// ---------------------------------------------------------------------------

fn cmd_invariants(args: &[String]) {
    let n = parse_usize_arg(args, 2, "invariants <n>");

    eprintln!("Computing invariants for all codes of length {}...", n);
    let start = Instant::now();
    let codes = enumerate_codes(n);
    let unique = deduplicate(codes);
    let elapsed = start.elapsed();
    eprintln!(
        "Found {} equivalence classes in {:?}",
        unique.len(),
        elapsed
    );

    println!();
    println!(
        "Invariants for doubly-even codes of length {} ({} classes):",
        n,
        unique.len()
    );
    println!();

    println!(
        "{:>4} | {:>3} | {:>3} | {:>6} | {:>13} | {}",
        "Idx", "n", "k", "d_min", "Decomposable", "Weight enumerator"
    );
    println!("{}", "-".repeat(80));

    for (i, code) in unique.iter().enumerate() {
        let inv = compute_invariants(code);
        let d = code.min_distance();
        let decomp = if code.k() <= 1 {
            "n/a".to_string()
        } else if is_decomposable(code) {
            "yes".to_string()
        } else {
            "no".to_string()
        };

        // Format weight enumerator compactly: only show nonzero entries
        let we: Vec<String> = inv
            .weight_enumerator
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(w, count)| format!("{}:{}", w, count))
            .collect();

        println!(
            "{:>4} | {:>3} | {:>3} | {:>6} | {:>13} | {}",
            i,
            inv.n,
            inv.k,
            d,
            decomp,
            we.join(" ")
        );
    }
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

fn cmd_validate() {
    println!("=== Doubly-Even Code Enumeration Validation ===");
    println!();
    println!(
        "Enumerating codes for N=4 through N=8 and comparing against known results."
    );
    println!(
        "Reference: Doran et al., arXiv:0806.0050 (doubly-even codes and Adinkra graphs)."
    );
    println!();

    println!(
        "{:>3} | {:>10} | {:>12} | {:>10} | {:>5} | {:>13}",
        "N", "Raw codes", "Total classes", "Nontrivial", "Max k", "Indecomposable"
    );
    println!("{}", "-".repeat(68));

    let mut all_ok = true;

    for n in 4..=8 {
        let start = Instant::now();
        let codes = enumerate_codes(n);
        let raw_count = codes.len();
        let unique = deduplicate(codes);
        let elapsed = start.elapsed();

        let nontrivial: Vec<&DoublyEvenCode> = unique.iter().filter(|c| c.k() > 0).collect();
        let max_k = unique.iter().map(|c| c.k()).max().unwrap_or(0);
        let indecomposable = nontrivial
            .iter()
            .filter(|c| !is_decomposable(c))
            .count();

        println!(
            "{:>3} | {:>10} | {:>12} | {:>10} | {:>5} | {:>13}",
            n,
            raw_count,
            unique.len(),
            nontrivial.len(),
            max_k,
            indecomposable
        );
        eprintln!("  N={} completed in {:?}", n, elapsed);

        // Sanity checks
        // N=4: should have at least 1 nontrivial code (the [4,1,4] repetition-like code)
        if n == 4 && nontrivial.is_empty() {
            eprintln!("  FAIL: N=4 should have at least 1 nontrivial code");
            all_ok = false;
        }

        // N=8: the extended Hamming [8,4,4] code should be present
        if n == 8 {
            let has_8_4 = nontrivial.iter().any(|c| c.k() == 4 && c.min_distance() == 4);
            if has_8_4 {
                eprintln!("  OK: found [8,4,4] extended Hamming code at N=8");
            } else {
                eprintln!(
                    "  FAIL: did not find [8,4,4] extended Hamming code at N=8"
                );
                all_ok = false;
            }

            // Check: max k at N=8 should be 4 (the Hamming code)
            if max_k < 4 {
                eprintln!(
                    "  FAIL: max k at N=8 is {}, expected at least 4",
                    max_k
                );
                all_ok = false;
            }
        }

        // Every code should actually be doubly-even
        for code in &unique {
            if !code.is_doubly_even() {
                eprintln!(
                    "  FAIL: found non-doubly-even code at N={}: {:?}",
                    n, code.generators
                );
                all_ok = false;
            }
        }
    }

    println!();

    // Additional validation: verify the Hamming code directly
    println!("--- Direct Hamming [8,4,4] Verification ---");
    let hamming = DoublyEvenCode::new(
        8,
        vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
    );
    let is_de = hamming.is_doubly_even();
    let d = hamming.min_distance();
    let we = hamming.weight_enumerator();
    println!(
        "  Hamming [8,4,4]: doubly_even={}, k={}, d_min={}, weight_enum={:?}",
        is_de,
        hamming.k(),
        d,
        we
    );
    if is_de && hamming.k() == 4 && d == 4 {
        eprintln!("  OK: Hamming code validates correctly");
    } else {
        eprintln!("  FAIL: Hamming code validation failed");
        all_ok = false;
    }

    // Check expected weight enumerator for [8,4,4]:
    // A_0=1, A_4=14, A_8=1 => total 16 codewords
    let total_cw: usize = we.iter().sum();
    if total_cw == 16 && we[0] == 1 && we[4] == 14 && we[8] == 1 {
        println!(
            "  Weight enumerator: A_0={}, A_4={}, A_8={} (correct for [8,4,4])",
            we[0], we[4], we[8]
        );
    } else {
        eprintln!("  WARN: unexpected weight enumerator for Hamming code");
        all_ok = false;
    }

    println!();
    if all_ok {
        println!("All validation checks passed.");
    } else {
        println!("Some validation checks FAILED. See messages above.");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

fn cmd_search(args: &[String]) {
    let mut config = search::SearchConfig::default();

    if args.len() > 2 {
        config.target_n = args[2].parse::<usize>().unwrap_or(16);
    }
    if args.len() > 3 {
        config.evo_population = args[3].parse::<usize>().unwrap_or(500);
    }
    if args.len() > 4 {
        config.evo_generations = args[4].parse::<usize>().unwrap_or(500);
    }

    search::search(&config);
}

// ---------------------------------------------------------------------------
// saturate
// ---------------------------------------------------------------------------

fn cmd_saturate(args: &[String]) {
    let n = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(16)
    } else {
        16
    };
    let batch_size = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(5000)
    } else {
        5000
    };
    let max_batches = if args.len() > 4 {
        args[4].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    search::saturate(n, batch_size, max_batches);
}

// ---------------------------------------------------------------------------
// validate-miller
// ---------------------------------------------------------------------------

fn cmd_validate_miller(args: &[String]) {
    let n = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(16)
    } else {
        16
    };

    search::validate_miller(n);
}

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

fn cmd_pipeline(args: &[String]) {
    let json_path = if args.len() > 2 {
        args[2].as_str()
    } else {
        "adinkra_codes_n16.json"
    };

    let output = pipeline::run_pipeline(json_path);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    println!("{}", json);
}

fn cmd_pipeline_k(args: &[String]) {
    let k = parse_usize_arg(args, 2, "pipeline-k <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };

    let output = pipeline::run_pipeline_k(json_path, k);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    println!("{}", json);
}

fn cmd_decompose_k(args: &[String]) {
    let k = parse_usize_arg(args, 2, "decompose-k <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };

    let output = pipeline::run_decompose_k(json_path, k);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    println!("{}", json);
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

fn parse_usize_arg(args: &[String], index: usize, usage_hint: &str) -> usize {
    if args.len() <= index {
        eprintln!("Missing argument. Usage: {} {}", args[0], usage_hint);
        std::process::exit(1);
    }
    match args[index].parse::<usize>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!(
                "Invalid number '{}'. Usage: {} {}",
                args[index], args[0], usage_hint
            );
            std::process::exit(1);
        }
    }
}
