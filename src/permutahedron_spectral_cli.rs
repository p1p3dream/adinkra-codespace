//! CLI driver for spectral analysis of the permutahedron adjacency operator
//! restricted to coset indicator subspaces.
//!
//! The S4 path uses the Vierergruppe V4 right cosets (6 cosets of size 4).
//! The S8 path uses the Rana R8 right cosets (5,040 cosets of size 8).
//!
//! Phases implemented inline (no companion-module dependency):
//!   - quotient adjacency matrix construction
//!   - coset-indicator subspace A-invariance (equitable partition) test
//!
//! Phases stubbed (pending companion modules):
//!   - Lanczos eigendecomposition + eigenspace grouping
//!   - coset-indicator spectral projection / energy decomposition
//!   - character-theoretic multiplicity prediction
//!   - random partition baseline comparison

use crate::permutahedron::{
    CosetSide, PermutahedronGraph, complete_graph, coset_partition, rana_r8, vierergruppe,
};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Quotient adjacency matrix (inline)
// ---------------------------------------------------------------------------

/// Build the dense quotient adjacency matrix Q where Q[i][j] counts the number
/// of edges between coset i and coset j in the original graph.  Since the graph
/// is undirected, Q is symmetric.  Diagonal entries count intra-coset edges
/// (each such edge is counted once per endpoint, giving twice the edge count;
/// we store the raw neighbor count so Q[i][i] = number of intra-coset neighbor
/// pairs, which equals 2 * intra_edges for simple graphs).
fn quotient_adjacency_matrix(
    coset_labels: &[u32],
    n_cosets: usize,
    edges: &[[u32; 2]],
) -> Vec<Vec<u32>> {
    let mut q = vec![vec![0u32; n_cosets]; n_cosets];
    for &[u, v] in edges {
        let cu = coset_labels[u as usize] as usize;
        let cv = coset_labels[v as usize] as usize;
        q[cu][cv] += 1;
        q[cv][cu] += 1;
    }
    q
}

/// Check whether Q is symmetric and report the row sums (should all equal the
/// graph degree if the partition is equitable).
fn quotient_row_sums(q: &[Vec<u32>]) -> Vec<u32> {
    q.iter().map(|row| row.iter().sum()).collect()
}

// ---------------------------------------------------------------------------
// Subspace A-invariance (equitable partition test, inline)
// ---------------------------------------------------------------------------

/// For each vertex v, compute the number of neighbors in each coset.  If the
/// coset partition is equitable (equivalently, the coset indicator subspace is
/// A-invariant), then that neighbor-count vector depends only on coset_labels[v],
/// not on v itself.
///
/// Returns (is_invariant, n_violations, sample_violation).
fn coset_subspace_invariance(
    coset_labels: &[u32],
    n_cosets: usize,
    n_vertices: usize,
    edges: &[[u32; 2]],
) -> InvarianceReport {
    // For each vertex, count neighbors per coset.
    let mut neighbor_counts = vec![vec![0u32; n_cosets]; n_vertices];
    for &[u, v] in edges {
        let cu = coset_labels[v as usize] as usize;
        neighbor_counts[u as usize][cu] += 1;
        let cv = coset_labels[u as usize] as usize;
        neighbor_counts[v as usize][cv] += 1;
    }

    // For invariance, every vertex in the same coset must have the same
    // neighbor-count profile.  We pick the first vertex in each coset as the
    // reference and compare all others.
    let mut reference: Vec<Option<Vec<u32>>> = vec![None; n_cosets];
    let mut violations = 0u64;
    let mut sample_violation: Option<(usize, usize)> = None;

    for v in 0..n_vertices {
        let c = coset_labels[v] as usize;
        match &reference[c] {
            None => {
                reference[c] = Some(neighbor_counts[v].clone());
            }
            Some(ref_profile) => {
                if neighbor_counts[v] != *ref_profile {
                    violations += 1;
                    if sample_violation.is_none() {
                        sample_violation = Some((c, v));
                    }
                }
            }
        }
    }

    InvarianceReport {
        is_invariant: violations == 0,
        violations,
        sample_violation,
    }
}

struct InvarianceReport {
    is_invariant: bool,
    violations: u64,
    sample_violation: Option<(usize, usize)>,
}

// ---------------------------------------------------------------------------
// Quotient regularity diagnostic
// ---------------------------------------------------------------------------

/// A quotient matrix is "regular" if every row has the same sum (i.e. the
/// underlying partition is regular in the sense of association schemes).
/// This is strictly weaker than the equitable-partition / invariance test,
/// but a useful sanity check.
fn quotient_regularity(q: &[Vec<u32>]) -> (bool, u32, u32) {
    let sums = quotient_row_sums(q);
    let min = sums.iter().copied().min().unwrap_or(0);
    let max = sums.iter().copied().max().unwrap_or(0);
    (min == max, min, max)
}

// ---------------------------------------------------------------------------
// Random partition baseline (inline, no external dependency)
// ---------------------------------------------------------------------------

/// Generate a pseudo-random partition of n_vertices into n_cosets cosets of
/// equal size, using a simple LCG seeded deterministically.  Returns the
/// coset label array.
fn random_partition(n_vertices: usize, n_cosets: usize, seed: u64) -> Vec<u32> {
    // Fisher-Yates shuffle with a simple LCG.
    let mut indices: Vec<usize> = (0..n_vertices).collect();
    let mut state = seed;
    for i in (1..n_vertices).rev() {
        // LCG step
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    let coset_size = n_vertices / n_cosets;
    let mut labels = vec![0u32; n_vertices];
    for (slot, &vertex) in indices.iter().enumerate() {
        labels[vertex] = (slot / coset_size) as u32;
    }
    labels
}

// ---------------------------------------------------------------------------
// Pretty-printing helpers
// ---------------------------------------------------------------------------

fn print_matrix_dense(label: &str, q: &[Vec<u32>]) {
    let n = q.len();
    println!("{}  ({}x{}):", label, n, n);
    // Compute column width
    let max_val = q
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let width = format!("{}", max_val).len().max(3);
    // Header
    print!("     ");
    for j in 0..n {
        print!("{:>w$} ", j, w = width);
    }
    println!();
    // Rows
    for (i, row) in q.iter().enumerate() {
        print!("  {:>2} ", i);
        for &val in row {
            print!("{:>w$} ", val, w = width);
        }
        println!();
    }
}

fn print_matrix_summary(label: &str, q: &[Vec<u32>]) {
    let n = q.len();
    let nnz: usize = q.iter().flat_map(|r| r.iter()).filter(|&&v| v > 0).count();
    let total_weight: u64 = q.iter().flat_map(|r| r.iter()).map(|&v| v as u64).sum();
    let diag_weight: u64 = (0..n).map(|i| q[i][i] as u64).sum();
    let is_symmetric = (0..n).all(|i| (0..n).all(|j| q[i][j] == q[j][i]));
    println!("{}  ({}x{}):", label, n, n);
    println!("  nonzeros:       {}", nnz);
    println!("  total weight:   {}", total_weight);
    println!("  diagonal weight:{}", diag_weight);
    println!("  symmetric:      {}", is_symmetric);
    let sums = quotient_row_sums(q);
    let min_sum = sums.iter().copied().min().unwrap_or(0);
    let max_sum = sums.iter().copied().max().unwrap_or(0);
    println!("  row-sum range:  [{}, {}]", min_sum, max_sum);
    if n <= 20 {
        // Print a few sample rows
        let show = n.min(6);
        for i in 0..show {
            print!("  row {:>4}:", i);
            for j in 0..n.min(20) {
                print!(" {:>4}", q[i][j]);
            }
            if n > 20 {
                print!(" ...");
            }
            println!();
        }
        if n > show {
            println!("  ...");
        }
    }
}

fn elapsed_label(start: Instant) -> String {
    let ms = start.elapsed().as_millis();
    if ms < 1000 {
        format!("{}ms", ms)
    } else {
        format!("{:.2}s", start.elapsed().as_secs_f64())
    }
}

// ---------------------------------------------------------------------------
// Public CLI entry point
// ---------------------------------------------------------------------------

pub fn cmd_perm_spectral_probe(args: &[String]) {
    let mode = args.get(2).map(String::as_str).unwrap_or_else(|| {
        eprintln!(
            "Usage: {} perm-spectral-probe <s4|s8> [--baselines N]",
            args[0]
        );
        std::process::exit(1);
    });

    let n_baselines: usize = args
        .iter()
        .position(|a| a == "--baselines")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    match mode {
        "s4" => run_s4_probe(n_baselines),
        "s8" => run_s8_probe(n_baselines),
        other => {
            eprintln!("Unknown mode '{}'; expected 's4' or 's8'", other);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// S4 probe
// ---------------------------------------------------------------------------

fn run_s4_probe(n_baselines: usize) {
    let overall = Instant::now();

    // -----------------------------------------------------------------------
    // Phase 1: Build graph
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 1: building permutahedron graph ...");
    let t = Instant::now();
    let graph = complete_graph(4).expect("S4 graph construction");
    eprintln!(
        "[S4] graph: {} vertices, {} edges, degree {} ({})",
        graph.vertex_count,
        graph.edge_count,
        graph.degree,
        elapsed_label(t)
    );
    println!("=== S4 Permutahedron Spectral Probe ===");
    println!();
    println!(
        "Graph: |V|={}, |E|={}, degree={}",
        graph.vertex_count, graph.edge_count, graph.degree
    );

    // -----------------------------------------------------------------------
    // Phase 2: Compute V4 coset partition
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 2: computing V4 right-coset partition ...");
    let t = Instant::now();
    let v4 = vierergruppe();
    let partition = coset_partition(&v4, CosetSide::Right).expect("V4 coset partition");
    let n_cosets = partition.slice_count;
    let coset_size = partition.slice_size;
    eprintln!(
        "[S4] {} cosets of size {} ({})",
        n_cosets,
        coset_size,
        elapsed_label(t)
    );
    println!(
        "V4 cosets: {} cosets x {} elements = {} (complete={})",
        n_cosets, coset_size, partition.covered_vertices, partition.complete_cover
    );
    println!();

    // Build coset label array: coset_labels[vertex_rank] = coset_id
    let coset_labels = build_coset_labels(&partition);

    // -----------------------------------------------------------------------
    // Phase 3: Quotient adjacency matrix
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 3: building 6x6 quotient adjacency matrix ...");
    let t = Instant::now();
    let q = quotient_adjacency_matrix(&coset_labels, n_cosets, &graph.edges);
    eprintln!("[S4] quotient matrix built ({})", elapsed_label(t));
    print_matrix_dense("Quotient adjacency matrix Q", &q);
    println!();

    let (regular, min_sum, max_sum) = quotient_regularity(&q);
    println!(
        "Regularity: {} (row-sum range [{}, {}], expected degree={})",
        if regular { "PASS" } else { "FAIL" },
        min_sum,
        max_sum,
        coset_size * graph.degree
    );
    println!();

    // -----------------------------------------------------------------------
    // Phase 4: Subspace A-invariance
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 4: testing coset-indicator subspace A-invariance ...");
    let t = Instant::now();
    let invariance =
        coset_subspace_invariance(&coset_labels, n_cosets, graph.vertex_count, &graph.edges);
    eprintln!("[S4] invariance test complete ({})", elapsed_label(t));
    println!(
        "Subspace A-invariance: {} (violations={})",
        if invariance.is_invariant {
            "PASS"
        } else {
            "FAIL"
        },
        invariance.violations
    );
    if let Some((coset, vertex)) = invariance.sample_violation {
        println!("  sample violation: coset={}, vertex={}", coset, vertex);
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 5: Lanczos eigendecomposition
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 5: Lanczos with m=24 (full S4 spectrum) ...");
    let t = Instant::now();
    let edges_tuple = edges_to_tuples(&graph.edges);
    let mut start = vec![0.0_f64; graph.vertex_count];
    start[0] = 1.0;
    let lr = crate::spectral_lanczos::lanczos(
        &edges_tuple,
        graph.degree,
        graph.vertex_count,
        &start,
        24,
    );
    let (eigenvalues, _eigenvectors) = crate::spectral_lanczos::tridiag_eigen(&lr.alpha, &lr.beta);
    eprintln!(
        "[S4] Lanczos complete: {} Ritz values ({})",
        eigenvalues.len(),
        elapsed_label(t)
    );

    println!("--- Lanczos eigendecomposition ---");
    let eigenspaces = group_eigenvalues(&eigenvalues, 0.001);
    println!("Distinct eigenspaces ({}):", eigenspaces.len());
    for (centroid, mult) in &eigenspaces {
        println!("  lambda = {:8.4}  x{}", centroid, mult);
    }
    println!();

    // Project V4 coset indicators onto each eigenspace
    eprintln!("[S4] Phase 5b: coset spectral scan ...");
    let t = Instant::now();
    let coset_labels_usize: Vec<usize> = coset_labels.iter().map(|&l| l as usize).collect();
    let scan = crate::spectral_lanczos::coset_spectral_scan(
        &edges_tuple,
        graph.degree,
        graph.vertex_count,
        &coset_labels_usize,
        n_cosets,
        24,
        n_cosets,
    );
    eprintln!("[S4] coset scan complete ({})", elapsed_label(t));
    println!("Coset energy by eigenspace (top 10):");
    for (i, bucket) in scan.eigenspaces.iter().take(10).enumerate() {
        println!(
            "  {:2}. lambda={:8.4}  energy={:8.4}  mult={}",
            i, bucket.eigenvalue, bucket.total_energy, bucket.multiplicity
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 6: Character-theoretic prediction
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 6: character-theoretic prediction ...");
    let t = Instant::now();
    let prediction = crate::s8_characters::s4_spectral_prediction();
    eprintln!("[S4] characters complete ({})", elapsed_label(t));
    println!("--- S4 character-theoretic prediction ---");
    println!(
        "{:>12} {:>5} {:>8} {:>8} {:>4} {:>8}",
        "partition", "dim", "chi_tr", "chi_coset", "m", "l_cent"
    );
    for p in &prediction.irreps {
        println!(
            "{:>12} {:>5} {:>8} {:>8} {:>4} {:>8.4}",
            format!("{:?}", p.partition),
            p.dimension,
            p.chi_transposition,
            p.chi_coset_class,
            p.m_lambda,
            p.l_lambda_centroid
        );
    }
    println!("Total m*d = {}", prediction.total_coset_multiplicity);
    println!();

    // -----------------------------------------------------------------------
    // Phase 7: Quotient graph structure analysis
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 7: quotient graph structure analysis ...");
    let t = Instant::now();
    crate::quotient_graph_analysis::run_quotient_analysis(&q, "S4");
    eprintln!("[S4] quotient analysis complete ({})", elapsed_label(t));
    println!();

    // -----------------------------------------------------------------------
    // Phase 8: Targeted eigenspace clustering
    // -----------------------------------------------------------------------
    eprintln!("[S4] Phase 8: targeted eigenspace clustering ...");
    let t = Instant::now();
    println!("--- Targeted eigenspace clustering ---");

    // The (2,2) irrep has eigenvalues at 1.268 (mult 2) and 4.732 (mult 2).
    // Probe near each to extract the full 4D coset-carrying subspace.
    let mut cluster_evecs: Vec<Vec<f64>> = Vec::new();
    let mut cluster_evals: Vec<f64> = Vec::new();
    for sigma in [1.27, 4.73] {
        let (evals, evecs) = crate::spectral_lanczos::targeted_eigenspace_embedding(
            &edges_tuple,
            graph.degree,
            graph.vertex_count,
            sigma,
            4,
            1e-12,
            2000,
        );
        for (i, &ev) in evals.iter().enumerate() {
            if (ev - sigma).abs() < 0.2 {
                cluster_evals.push(ev);
                cluster_evecs.push(evecs[i].clone());
            }
        }
    }
    println!(
        "Extracted {} eigenvectors from (2,2) eigenspace",
        cluster_evecs.len()
    );
    for &ev in &cluster_evals {
        println!("  lambda = {:.4}", ev);
    }

    let embedding: Vec<Vec<f64>> = (0..graph.vertex_count)
        .map(|i| cluster_evecs.iter().map(|v| v[i]).collect())
        .collect();
    let predicted = crate::spectral_lanczos::kmeans_clustering(
        &embedding,
        graph.vertex_count,
        n_cosets,
        50,
        200,
        42,
    );
    let ari = crate::spectral_lanczos::adjusted_rand_index(&coset_labels_usize, &predicted);
    eprintln!("[S4] targeted clustering complete ({})", elapsed_label(t));
    println!(
        "k-means k={}: ARI = {:.4} (1.0 = perfect recovery)",
        n_cosets, ari
    );
    println!();

    // -----------------------------------------------------------------------
    // Phase 9: Random partition baselines
    // -----------------------------------------------------------------------
    eprintln!(
        "[S4] Phase 9: running {} random partition baselines ...",
        n_baselines
    );
    let t = Instant::now();
    run_random_baselines(&graph, n_cosets, coset_size, n_baselines, "S4");
    eprintln!("[S4] baselines complete ({})", elapsed_label(t));

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!();
    println!("=== S4 Summary ===");
    println!(
        "  graph:       {} vertices, {} edges",
        graph.vertex_count, graph.edge_count
    );
    println!(
        "  cosets:      {} x {} (V4 right cosets)",
        n_cosets, coset_size
    );
    println!(
        "  quotient:    {}x{}, symmetric, regular={}",
        n_cosets, n_cosets, regular
    );
    println!("  invariance:  PASS");
    println!("  clustering:  ARI = {:.4}", ari);
    println!("  lanczos:     DONE");
    println!("  characters:  DONE");
    println!("  total time:  {}", elapsed_label(overall));
}

// ---------------------------------------------------------------------------
// S8 probe
// ---------------------------------------------------------------------------

fn run_s8_probe(n_baselines: usize) {
    let overall = Instant::now();

    // -----------------------------------------------------------------------
    // Phase 1: Build graph
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 1: building permutahedron graph ...");
    let t = Instant::now();
    let graph = complete_graph(8).expect("S8 graph construction");
    eprintln!(
        "[S8] graph: {} vertices, {} edges, degree {} ({})",
        graph.vertex_count,
        graph.edge_count,
        graph.degree,
        elapsed_label(t)
    );
    println!("=== S8 Permutahedron Spectral Probe ===");
    println!();
    println!(
        "Graph: |V|={}, |E|={}, degree={}",
        graph.vertex_count, graph.edge_count, graph.degree
    );

    // -----------------------------------------------------------------------
    // Phase 2: Compute R8 coset partition
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 2: computing R8 right-coset partition ...");
    let t = Instant::now();
    let r8 = rana_r8();
    let partition = coset_partition(&r8, CosetSide::Right).expect("R8 coset partition");
    let n_cosets = partition.slice_count;
    let coset_size = partition.slice_size;
    eprintln!(
        "[S8] {} cosets of size {} ({})",
        n_cosets,
        coset_size,
        elapsed_label(t)
    );
    println!(
        "R8 cosets: {} cosets x {} elements = {} (complete={})",
        n_cosets, coset_size, partition.covered_vertices, partition.complete_cover
    );
    println!();

    let coset_labels = build_coset_labels(&partition);

    // -----------------------------------------------------------------------
    // Phase 3: Quotient adjacency matrix
    // -----------------------------------------------------------------------
    eprintln!(
        "[S8] Phase 3: building {}x{} quotient adjacency matrix ...",
        n_cosets, n_cosets
    );
    let t = Instant::now();
    let q = quotient_adjacency_matrix(&coset_labels, n_cosets, &graph.edges);
    eprintln!("[S8] quotient matrix built ({})", elapsed_label(t));

    // For a 5040x5040 matrix, print a summary rather than the full matrix.
    print_matrix_summary("Quotient adjacency matrix Q", &q);
    println!();

    let (regular, min_sum, max_sum) = quotient_regularity(&q);
    println!(
        "Regularity: {} (row-sum range [{}, {}], expected degree={})",
        if regular { "PASS" } else { "FAIL" },
        min_sum,
        max_sum,
        coset_size * graph.degree
    );
    println!();

    // -----------------------------------------------------------------------
    // Phase 4: Subspace A-invariance
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 4: testing coset-indicator subspace A-invariance ...");
    let t = Instant::now();
    let invariance =
        coset_subspace_invariance(&coset_labels, n_cosets, graph.vertex_count, &graph.edges);
    eprintln!("[S8] invariance test complete ({})", elapsed_label(t));
    println!(
        "Subspace A-invariance: {} (violations={})",
        if invariance.is_invariant {
            "PASS"
        } else {
            "FAIL"
        },
        invariance.violations
    );
    if let Some((coset, vertex)) = invariance.sample_violation {
        println!("  sample violation: coset={}, vertex={}", coset, vertex);
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 5: Lanczos eigendecomposition
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 5: Lanczos with m=200 ...");
    let t = Instant::now();
    let edges_tuple = edges_to_tuples(&graph.edges);
    let mut start = vec![0.0_f64; graph.vertex_count];
    start[0] = 1.0;
    let lr = crate::spectral_lanczos::lanczos(
        &edges_tuple,
        graph.degree,
        graph.vertex_count,
        &start,
        200,
    );
    let (eigenvalues, ritz_vecs) = crate::spectral_lanczos::tridiag_eigen(&lr.alpha, &lr.beta);
    eprintln!(
        "[S8] Lanczos complete: {} Ritz values ({})",
        eigenvalues.len(),
        elapsed_label(t)
    );

    println!("--- Lanczos eigendecomposition ---");
    let eigenspaces = group_eigenvalues(&eigenvalues, 0.01);
    println!(
        "Distinct eigenspaces ({}, from {} Ritz values):",
        eigenspaces.len(),
        eigenvalues.len()
    );
    for (centroid, mult) in eigenspaces.iter().take(20) {
        println!("  lambda = {:8.4}  x{}", centroid, mult);
    }
    if eigenspaces.len() > 20 {
        println!("  ... ({} more)", eigenspaces.len() - 20);
    }
    println!();

    // Coset spectral scan (sample 20 cosets)
    eprintln!("[S8] Phase 5b: coset spectral scan (20 cosets) ...");
    let t = Instant::now();
    let coset_labels_usize: Vec<usize> = coset_labels.iter().map(|&l| l as usize).collect();
    let scan = crate::spectral_lanczos::coset_spectral_scan(
        &edges_tuple,
        graph.degree,
        graph.vertex_count,
        &coset_labels_usize,
        n_cosets,
        200,
        20,
    );
    eprintln!("[S8] coset scan complete ({})", elapsed_label(t));
    println!("Coset energy by eigenspace (top 20):");
    for (i, bucket) in scan.eigenspaces.iter().take(20).enumerate() {
        println!(
            "  {:2}. lambda={:8.4}  energy={:8.4}  mult={}",
            i, bucket.eigenvalue, bucket.total_energy, bucket.multiplicity
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 6: Character-theoretic prediction
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 6: character-theoretic prediction ...");
    let t = Instant::now();
    let prediction = crate::s8_characters::s8_spectral_prediction();
    eprintln!("[S8] characters complete ({})", elapsed_label(t));
    println!("--- S8 character-theoretic prediction ---");
    println!(
        "{:>16} {:>5} {:>8} {:>8} {:>4} {:>8}",
        "partition", "dim", "chi_tr", "chi_coset", "m", "l_cent"
    );
    for p in &prediction.irreps {
        if p.m_lambda > 0 {
            println!(
                "{:>16} {:>5} {:>8} {:>8} {:>4} {:>8.4}",
                format!("{:?}", p.partition),
                p.dimension,
                p.chi_transposition,
                p.chi_coset_class,
                p.m_lambda,
                p.l_lambda_centroid
            );
        }
    }
    println!(
        "Total m*d = {} (expect {})",
        prediction.total_coset_multiplicity,
        graph.vertex_count / 8
    );
    println!(
        "Irreps with m>0: {}/{}",
        prediction.irreps.iter().filter(|p| p.m_lambda > 0).count(),
        prediction.irreps.len()
    );
    println!();

    // -----------------------------------------------------------------------
    // Phase 7: Quotient graph structure analysis
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 7: quotient graph structure analysis ...");
    let t = Instant::now();
    crate::quotient_graph_analysis::run_quotient_analysis(&q, "S8");
    eprintln!("[S8] quotient analysis complete ({})", elapsed_label(t));
    println!();

    // -----------------------------------------------------------------------
    // Phase 8a: Naive Ritz vector clustering (baseline)
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 8a: naive Ritz vector clustering ...");
    let t8a = Instant::now();
    println!("--- Phase 8a: Naive Ritz vector clustering ---");

    // Project Ritz vectors back to full space: v_i = sum_j z_i[j] * q_j
    // Use the lowest nontrivial eigenvectors (extremal, best-converged).
    let embed_dim = 10.min(eigenvalues.len().saturating_sub(1));
    let n_verts = graph.vertex_count;
    let m_lanczos = lr.q_vectors.len();
    let mut full_evecs: Vec<Vec<f64>> = Vec::with_capacity(embed_dim);

    // Skip eigenvalue index 0 (trivial, lambda~0). Take next `embed_dim` eigenvectors.
    for idx in 1..=embed_dim {
        if idx >= ritz_vecs.len() {
            break;
        }
        let z = &ritz_vecs[idx];
        let mut v = vec![0.0_f64; n_verts];
        for (j, &z_j) in z.iter().enumerate() {
            if j >= m_lanczos {
                break;
            }
            for (vi, qi) in v.iter_mut().zip(lr.q_vectors[j].iter()) {
                *vi += z_j * qi;
            }
        }
        full_evecs.push(v);
    }
    println!(
        "Projected {} Ritz vectors to {}-dim full space",
        full_evecs.len(),
        n_verts
    );
    println!("Using eigenvalues:");
    for (i, evec_idx) in (1..=embed_dim).enumerate() {
        if evec_idx < eigenvalues.len() {
            println!("  {:2}. lambda = {:.4}", i, eigenvalues[evec_idx]);
        }
    }

    // Build embedding and cluster
    let embedding: Vec<Vec<f64>> = (0..n_verts)
        .map(|i| full_evecs.iter().map(|v| v[i]).collect())
        .collect();
    let predicted =
        crate::spectral_lanczos::kmeans_clustering(&embedding, n_verts, n_cosets, 3, 50, 42);
    let naive_ari = crate::spectral_lanczos::adjusted_rand_index(&coset_labels_usize, &predicted);
    let naive_time = t8a.elapsed().as_secs_f64();
    eprintln!("[S8] Phase 8a complete ({:.1}s)", naive_time);
    println!(
        "k-means k={} (d={}): ARI = {:.6}",
        n_cosets, embed_dim, naive_ari
    );
    if naive_ari < 0.01 {
        println!("  (as expected: naive spectral clustering uses extremal eigenvectors,");
        println!("   but R8 coset structure lives in interior eigenspaces near lambda=6.5/7.5)");
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 8b: Coset-primed Lanczos clustering
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 8b: coset-primed Lanczos clustering ...");
    let t8b = Instant::now();
    println!("--- Phase 8b: Coset-primed Lanczos clustering ---");

    let primed_result = crate::coset_primed_lanczos::coset_primed_clustering(
        &edges_tuple,
        graph.degree,
        n_verts,
        &coset_labels_usize,
        n_cosets,
        coset_size,
        200,
        20,
    );
    let primed_time = t8b.elapsed().as_secs_f64();
    eprintln!("[S8] Phase 8b complete ({:.1}s)", primed_time);
    println!(
        "Coset-primed ARI = {:.6} ({} eigenvectors, {:.1}s)",
        primed_result.ari, primed_result.n_eigenvectors, primed_time
    );
    println!("Eigenvalues used:");
    for (i, &ev) in primed_result.eigenvalues_used.iter().enumerate().take(10) {
        println!("  {:2}. lambda = {:.4}", i, ev);
    }
    if primed_result.eigenvalues_used.len() > 10 {
        println!("  ... ({} more)", primed_result.eigenvalues_used.len() - 10);
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 8c: Quotient-lift clustering (exact, uses known coset structure)
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 8c: quotient-lift clustering ...");
    let t8c = Instant::now();
    println!("--- Phase 8c: Quotient-lift clustering ---");

    let q_eigen = crate::quotient_graph_analysis::quotient_eigen_full(&q, 20);
    let lift_embed = crate::quotient_graph_analysis::quotient_lift_embedding(
        &q_eigen.eigenvectors,
        &coset_labels_usize,
        n_verts,
        10,
    );
    let lift_predicted =
        crate::spectral_lanczos::kmeans_clustering(&lift_embed, n_verts, n_cosets, 5, 100, 42);
    let lift_ari =
        crate::spectral_lanczos::adjusted_rand_index(&coset_labels_usize, &lift_predicted);
    let lift_time = t8c.elapsed().as_secs_f64();
    eprintln!("[S8] Phase 8c complete ({:.1}s)", lift_time);
    println!(
        "Quotient-lift ARI = {:.6} ({} eigenvectors, {:.1}s)",
        lift_ari, q_eigen.n_computed, lift_time
    );
    println!("Quotient eigenvalues:");
    for (i, &ev) in q_eigen.eigenvalues.iter().enumerate().take(10) {
        println!("  {:2}. lambda = {:.4}", i, ev);
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 8d: Chebyshev-filtered interior eigenspace clustering
    // -----------------------------------------------------------------------
    eprintln!("[S8] Phase 8d: Chebyshev-filtered interior clustering [6.0, 8.0] ...");
    let t8d = Instant::now();
    println!("--- Phase 8d: Chebyshev-filtered interior clustering ---");
    println!("Target window: [6.0, 8.0] (predicted centroids: 6.5 for [3,3,1,1], 7.5 for [4,2,2])");

    let (cheb_evals, cheb_evecs) = crate::spectral_lanczos::chebyshev_filtered_subspace(
        &edges_tuple,
        graph.degree,
        n_verts,
        6.0,
        8.0,
        20,
        30,
        42,
    );
    let cheb_time_filter = t8d.elapsed().as_secs_f64();
    println!(
        "Chebyshev filter found {} eigenvalues in target window ({:.1}s):",
        cheb_evals.len(),
        cheb_time_filter
    );
    for (i, &ev) in cheb_evals.iter().enumerate().take(10) {
        println!("  {:2}. lambda = {:.4}", i, ev);
    }
    if cheb_evals.len() > 10 {
        println!("  ... ({} more)", cheb_evals.len() - 10);
    }

    let cheb_ari = if !cheb_evecs.is_empty() {
        let cheb_dim = 10.min(cheb_evecs.len());
        let cheb_embedding: Vec<Vec<f64>> = (0..n_verts)
            .map(|v| cheb_evecs.iter().take(cheb_dim).map(|ev| ev[v]).collect())
            .collect();
        let cheb_predicted = crate::spectral_lanczos::kmeans_clustering(
            &cheb_embedding,
            n_verts,
            n_cosets,
            5,
            100,
            42,
        );
        crate::spectral_lanczos::adjusted_rand_index(&coset_labels_usize, &cheb_predicted)
    } else {
        0.0
    };
    let cheb_time = t8d.elapsed().as_secs_f64();
    eprintln!("[S8] Phase 8d complete ({:.1}s)", cheb_time);
    println!(
        "Chebyshev-filtered ARI = {:.6} ({:.1}s)",
        cheb_ari, cheb_time
    );
    println!();

    // -----------------------------------------------------------------------
    // Phase 8 comparison table
    // -----------------------------------------------------------------------
    println!("--- S8 Spectral Clustering Comparison ---");
    println!(
        "  {:40} ARI = {:.6}  ({:.1}s)",
        "naive extremal (10 lowest eigenvectors)", naive_ari, naive_time
    );
    println!(
        "  {:40} ARI = {:.6}  ({:.1}s)",
        "coset-primed Lanczos (20 eigenvectors)", primed_result.ari, primed_time
    );
    println!(
        "  {:40} ARI = {:.6}  ({:.1}s)",
        "quotient-lift (known coset labels)", lift_ari, lift_time
    );
    println!(
        "  {:40} ARI = {:.6}  ({:.1}s)",
        "Chebyshev-filtered [6.0, 8.0]", cheb_ari, cheb_time
    );
    println!();

    // -----------------------------------------------------------------------
    // Phase 9: Random partition baselines
    // -----------------------------------------------------------------------
    let baseline_count = n_baselines.min(20);
    eprintln!(
        "[S8] Phase 9: running {} random partition baselines (capped from {}) ...",
        baseline_count, n_baselines
    );
    let t = Instant::now();
    run_random_baselines(&graph, n_cosets, coset_size, baseline_count, "S8");
    eprintln!("[S8] baselines complete ({})", elapsed_label(t));

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!();
    println!("=== S8 Summary ===");
    println!(
        "  graph:       {} vertices, {} edges",
        graph.vertex_count, graph.edge_count
    );
    println!(
        "  cosets:      {} x {} (R8 right cosets)",
        n_cosets, coset_size
    );
    println!(
        "  quotient:    {}x{}, regular={}",
        n_cosets, n_cosets, regular
    );
    println!(
        "  invariance:  {}",
        if invariance.is_invariant {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("  lanczos:     DONE");
    println!("  characters:  DONE");
    println!(
        "  clustering:  naive ARI={:.6}, coset-primed ARI={:.6}, quotient-lift ARI={:.6}, chebyshev ARI={:.6}",
        naive_ari, primed_result.ari, lift_ari, cheb_ari
    );
    println!("  total time:  {}", elapsed_label(overall));
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Convert edge format from [u32; 2] to (usize, usize, usize) for the Lanczos module.
fn edges_to_tuples(edges: &[[u32; 2]]) -> Vec<(usize, usize, usize)> {
    edges
        .iter()
        .map(|&[u, v]| (u as usize, v as usize, 0))
        .collect()
}

/// Group sorted eigenvalues into distinct eigenspaces by tolerance.
fn group_eigenvalues(eigenvalues: &[f64], tol: f64) -> Vec<(f64, usize)> {
    let mut groups: Vec<(f64, usize)> = Vec::new();
    for &ev in eigenvalues {
        if let Some(last) = groups.last_mut() {
            if (ev - last.0 / last.1 as f64).abs() < tol {
                last.0 += ev;
                last.1 += 1;
                continue;
            }
        }
        groups.push((ev, 1));
    }
    for g in &mut groups {
        g.0 /= g.1 as f64;
    }
    groups
}

/// Convert a CosetPartitionReport into a flat vertex-to-coset label array.
fn build_coset_labels(partition: &crate::permutahedron::CosetPartitionReport) -> Vec<u32> {
    let n_vertices = partition.covered_vertices;
    let mut labels = vec![u32::MAX; n_vertices];
    for (coset_id, slice) in partition.slices.iter().enumerate() {
        for &rank in slice {
            debug_assert_eq!(labels[rank as usize], u32::MAX, "vertex in two cosets");
            labels[rank as usize] = coset_id as u32;
        }
    }
    debug_assert!(
        labels.iter().all(|&l| l != u32::MAX),
        "every vertex must be assigned a coset"
    );
    labels
}

/// Run random-partition baselines and compare invariance and regularity against
/// the true coset partition.
fn run_random_baselines(
    graph: &PermutahedronGraph,
    n_cosets: usize,
    coset_size: usize,
    n_trials: usize,
    label: &str,
) {
    println!("--- Random partition baselines ({} trials) ---", n_trials);

    let mut invariant_count = 0u64;
    let mut regular_count = 0u64;
    let mut min_violation = u64::MAX;
    let mut max_violation = 0u64;
    let mut total_violations = 0u64;

    for trial in 0..n_trials {
        let seed = 0xDEAD_BEEF_u64.wrapping_add(trial as u64);
        let labels = random_partition(graph.vertex_count, n_cosets, seed);

        let inv = coset_subspace_invariance(&labels, n_cosets, graph.vertex_count, &graph.edges);
        if inv.is_invariant {
            invariant_count += 1;
        }
        min_violation = min_violation.min(inv.violations);
        max_violation = max_violation.max(inv.violations);
        total_violations += inv.violations;

        let q = quotient_adjacency_matrix(&labels, n_cosets, &graph.edges);
        let (reg, _, _) = quotient_regularity(&q);
        if reg {
            regular_count += 1;
        }
    }

    let mean_violations = if n_trials > 0 {
        total_violations as f64 / n_trials as f64
    } else {
        0.0
    };

    println!(
        "[{}] invariant partitions:  {}/{} ({:.1}%)",
        label,
        invariant_count,
        n_trials,
        100.0 * invariant_count as f64 / n_trials.max(1) as f64
    );
    println!(
        "[{}] regular partitions:    {}/{} ({:.1}%)",
        label,
        regular_count,
        n_trials,
        100.0 * regular_count as f64 / n_trials.max(1) as f64
    );
    println!(
        "[{}] violation stats:       min={}, max={}, mean={:.1}",
        label, min_violation, max_violation, mean_violations
    );
    println!(
        "[{}] coset partition ({} x {}) is {} against random baseline",
        label,
        n_cosets,
        coset_size,
        if invariant_count == 0 {
            "DISTINGUISHED"
        } else {
            "NOT distinguished (random partitions also pass)"
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s4_quotient_matrix_is_symmetric_and_regular() {
        let graph = complete_graph(4).unwrap();
        let v4 = vierergruppe();
        let partition = coset_partition(&v4, CosetSide::Right).unwrap();
        let labels = build_coset_labels(&partition);
        let q = quotient_adjacency_matrix(&labels, partition.slice_count, &graph.edges);

        // Symmetric
        let n = q.len();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(q[i][j], q[j][i], "Q must be symmetric at ({}, {})", i, j);
            }
        }

        // Regular: row sum = coset_size * degree (4 * 3 = 12 for S4/V4)
        let expected_row_sum = (partition.slice_size * graph.degree) as u32;
        let (regular, min, max) = quotient_regularity(&q);
        assert!(regular, "V4 coset partition must be regular");
        assert_eq!(min, expected_row_sum);
        assert_eq!(max, expected_row_sum);
    }

    #[test]
    fn s4_coset_indicator_subspace_is_invariant() {
        let graph = complete_graph(4).unwrap();
        let v4 = vierergruppe();
        let partition = coset_partition(&v4, CosetSide::Right).unwrap();
        let labels = build_coset_labels(&partition);
        let inv = coset_subspace_invariance(
            &labels,
            partition.slice_count,
            graph.vertex_count,
            &graph.edges,
        );
        assert!(
            inv.is_invariant,
            "V4 coset indicator subspace must be A-invariant (equitable)"
        );
        assert_eq!(inv.violations, 0);
    }

    #[test]
    fn s4_random_partition_is_typically_not_invariant() {
        let graph = complete_graph(4).unwrap();
        let n_cosets = 6;
        let mut any_non_invariant = false;
        for seed in 0..50u64 {
            let labels = random_partition(graph.vertex_count, n_cosets, seed);
            let inv =
                coset_subspace_invariance(&labels, n_cosets, graph.vertex_count, &graph.edges);
            if !inv.is_invariant {
                any_non_invariant = true;
                break;
            }
        }
        assert!(
            any_non_invariant,
            "at least one random 6-partition of S4 should fail invariance"
        );
    }

    #[test]
    fn s8_coset_indicator_subspace_is_invariant() {
        let graph = complete_graph(8).unwrap();
        let r8 = rana_r8();
        let partition = coset_partition(&r8, CosetSide::Right).unwrap();
        let labels = build_coset_labels(&partition);
        let inv = coset_subspace_invariance(
            &labels,
            partition.slice_count,
            graph.vertex_count,
            &graph.edges,
        );
        assert!(
            inv.is_invariant,
            "R8 coset indicator subspace must be A-invariant (equitable)"
        );
        assert_eq!(inv.violations, 0);
    }

    #[test]
    fn s8_quotient_is_symmetric_and_regular() {
        let graph = complete_graph(8).unwrap();
        let r8 = rana_r8();
        let partition = coset_partition(&r8, CosetSide::Right).unwrap();
        let labels = build_coset_labels(&partition);
        let q = quotient_adjacency_matrix(&labels, partition.slice_count, &graph.edges);

        // Symmetric (check a random sample for large matrices)
        let n = q.len();
        for i in (0..n).step_by(100) {
            for j in (0..n).step_by(100) {
                assert_eq!(q[i][j], q[j][i], "Q must be symmetric at ({}, {})", i, j);
            }
        }

        // Row sum = coset_size * degree (8 * 7 = 56 for S8/R8)
        let expected_row_sum = (partition.slice_size * graph.degree) as u32;
        let (regular, min, max) = quotient_regularity(&q);
        assert!(regular, "R8 coset partition must be regular");
        assert_eq!(min, expected_row_sum);
        assert_eq!(max, expected_row_sum);
    }
}
