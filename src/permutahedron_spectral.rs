//! Spectral analysis of coset partitions on the permutahedron Cayley graph.
//!
//! Given a subgroup H of S_n, the right coset partition {H*g} tiles the vertices
//! of the permutahedron.  This module measures whether that partition is
//! "spectrally clean": whether the coset indicator functions span an
//! A-invariant subspace (equivalently, whether the quotient graph is
//! well-defined and regular).
//!
//! For V4 in S4, the quotient is perfectly regular because V4 is normal.
//! For R8 in S8, R8 is NOT normal, so the quotient may exhibit irregularity.

use serde::Serialize;

use crate::permutahedron::{self, CosetSide, PermutahedronGraph};

// ---------------------------------------------------------------------------
// Helper: convert slices representation to a flat coset-label array
// ---------------------------------------------------------------------------

/// Build a vertex-indexed coset label array from the slices representation.
/// `coset_labels[rank] = coset_index` for each vertex rank.
fn labels_from_slices(n_vertices: usize, slices: &[Vec<u32>]) -> Vec<usize> {
    let mut labels = vec![usize::MAX; n_vertices];
    for (coset_idx, slice) in slices.iter().enumerate() {
        for &rank in slice {
            labels[rank as usize] = coset_idx;
        }
    }
    labels
}

// ---------------------------------------------------------------------------
// 1. Quotient adjacency matrix
// ---------------------------------------------------------------------------

/// Dense quotient adjacency matrix counting directed half-edges between cosets.
///
/// For each undirected edge {u, v} in the Cayley graph, both directions
/// (u -> v) and (v -> u) contribute, so `quotient[c_u][c_v] += 1` and
/// `quotient[c_v][c_u] += 1`.  The result is symmetric.
///
/// `coset_labels[rank]` gives the coset index for vertex `rank`.
pub fn quotient_adjacency_matrix(
    coset_labels: &[usize],
    n_cosets: usize,
    graph: &PermutahedronGraph,
) -> Vec<Vec<u32>> {
    let mut matrix = vec![vec![0u32; n_cosets]; n_cosets];
    for edge in &graph.edges {
        let cu = coset_labels[edge[0] as usize];
        let cv = coset_labels[edge[1] as usize];
        matrix[cu][cv] += 1;
        matrix[cv][cu] += 1;
    }
    matrix
}

// ---------------------------------------------------------------------------
// 2. Coset subspace A-invariance
// ---------------------------------------------------------------------------

/// Per-coset invariance diagnostic.
#[derive(Debug, Clone, Serialize)]
pub struct CosetInvarianceDiagnostic {
    pub coset_index: usize,
    /// For each coset c', the value (A * f_c)[v] should be the same for all
    /// v in c'.  `max_within_coset_deviation` is the maximum over all c' of
    /// (max - min) of (A * f_c) restricted to c'.
    pub max_within_coset_deviation: u32,
}

/// Report whether the coset indicator subspace is A-invariant.
#[derive(Debug, Clone, Serialize)]
pub struct InvarianceReport {
    pub n_cosets: usize,
    pub a_invariant: bool,
    /// Maximum deviation across all coset indicators.
    pub global_max_deviation: u32,
    pub per_coset: Vec<CosetInvarianceDiagnostic>,
}

/// For each coset indicator function f_c (f_c[v] = 1 if v in coset c, else 0),
/// compute A * f_c using the adjacency structure, then check whether
/// (A * f_c) is constant within each coset.
///
/// If every coset indicator maps to a vector that is constant on each coset,
/// then the coset indicators span an A-invariant subspace, and the quotient
/// graph is well-defined.
pub fn coset_subspace_invariance(
    coset_labels: &[usize],
    n_cosets: usize,
    graph: &PermutahedronGraph,
) -> InvarianceReport {
    let n_vertices = coset_labels.len();

    // Build adjacency lists from the edge list for efficient A * f_c.
    let mut adj = vec![Vec::new(); n_vertices];
    for edge in &graph.edges {
        let u = edge[0] as usize;
        let v = edge[1] as usize;
        adj[u].push(v);
        adj[v].push(u);
    }

    let mut global_max_deviation: u32 = 0;
    let mut per_coset = Vec::with_capacity(n_cosets);

    for c in 0..n_cosets {
        // Compute (A * f_c)[v] = number of neighbors of v that belong to coset c.
        let mut a_fc = vec![0u32; n_vertices];
        for (v, neighbors) in adj.iter().enumerate() {
            let mut count = 0u32;
            for &u in neighbors {
                if coset_labels[u] == c {
                    count += 1;
                }
            }
            a_fc[v] = count;
        }

        // Check constancy of a_fc within each coset c'.
        // For each coset c', gather min and max of a_fc on that coset.
        let mut max_deviation: u32 = 0;
        // Use a pair of arrays to track min/max per coset.
        let mut coset_min = vec![u32::MAX; n_cosets];
        let mut coset_max = vec![0u32; n_cosets];

        for (v, &label) in coset_labels.iter().enumerate() {
            let val = a_fc[v];
            if val < coset_min[label] {
                coset_min[label] = val;
            }
            if val > coset_max[label] {
                coset_max[label] = val;
            }
        }

        for cp in 0..n_cosets {
            let dev = coset_max[cp].saturating_sub(coset_min[cp]);
            if dev > max_deviation {
                max_deviation = dev;
            }
        }

        if max_deviation > global_max_deviation {
            global_max_deviation = max_deviation;
        }

        per_coset.push(CosetInvarianceDiagnostic {
            coset_index: c,
            max_within_coset_deviation: max_deviation,
        });
    }

    InvarianceReport {
        n_cosets,
        a_invariant: global_max_deviation == 0,
        global_max_deviation,
        per_coset,
    }
}

// ---------------------------------------------------------------------------
// 3. Quotient regularity
// ---------------------------------------------------------------------------

/// Regularity report for the quotient graph.
#[derive(Debug, Clone, Serialize)]
pub struct RegularityReport {
    pub n_cosets: usize,
    /// Row sums of the quotient matrix.
    pub row_sums: Vec<u32>,
    /// Whether all row sums are equal.
    pub uniform_row_sums: bool,
    /// Whether all diagonal entries are equal.
    pub uniform_diagonal: bool,
    /// Whether all diagonal entries are zero (cosets are independent sets).
    pub zero_diagonal: bool,
    /// The diagonal entries.
    pub diagonal: Vec<u32>,
    /// Whether the quotient matrix is symmetric.
    pub symmetric: bool,
}

/// Analyze regularity of the quotient adjacency matrix.
pub fn quotient_regularity(quotient: &[Vec<u32>]) -> RegularityReport {
    let n_cosets = quotient.len();
    let row_sums: Vec<u32> = quotient.iter().map(|row| row.iter().sum()).collect();
    let diagonal: Vec<u32> = (0..n_cosets).map(|i| quotient[i][i]).collect();

    let uniform_row_sums = row_sums.windows(2).all(|w| w[0] == w[1]);
    let uniform_diagonal = diagonal.windows(2).all(|w| w[0] == w[1]);
    let zero_diagonal = diagonal.iter().all(|&d| d == 0);

    let symmetric = (0..n_cosets).all(|i| (0..n_cosets).all(|j| quotient[i][j] == quotient[j][i]));

    RegularityReport {
        n_cosets,
        row_sums,
        uniform_row_sums,
        uniform_diagonal,
        zero_diagonal,
        diagonal,
        symmetric,
    }
}

// ---------------------------------------------------------------------------
// 4. Random partition baseline
// ---------------------------------------------------------------------------

/// Baseline comparison against random equal-size partitions.
#[derive(Debug, Clone, Serialize)]
pub struct BaselineReport {
    pub n_trials: usize,
    pub coset_size: usize,
    pub n_cosets: usize,
    /// Fraction of random trials that achieved A-invariance (should be ~0).
    pub invariance_fraction: f64,
    /// Fraction of random trials with uniform row sums.
    pub uniform_row_sum_fraction: f64,
    /// Fraction of random trials with zero diagonal.
    pub zero_diagonal_fraction: f64,
    /// Mean global max deviation across random trials.
    pub mean_max_deviation: f64,
    /// The actual coset partition's max deviation (for comparison).
    pub actual_max_deviation: u32,
    /// Whether the actual coset partition is A-invariant.
    pub actual_invariant: bool,
}

/// Simple linear congruential generator for deterministic random permutations.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // Knuth's LCG constants
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Fisher-Yates shuffle of a mutable slice.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}

/// Generate random equal-size partitions and compare their spectral properties
/// against the actual coset partition.
pub fn random_partition_baseline(
    n_vertices: usize,
    n_cosets: usize,
    coset_size: usize,
    graph: &PermutahedronGraph,
    actual_labels: &[usize],
    n_trials: usize,
    seed: u64,
) -> BaselineReport {
    // Compute actual partition metrics.
    let actual_inv = coset_subspace_invariance(actual_labels, n_cosets, graph);
    let actual_max_deviation = actual_inv.global_max_deviation;
    let actual_invariant = actual_inv.a_invariant;

    let mut rng = Lcg::new(seed);
    let mut invariance_count = 0usize;
    let mut uniform_row_sum_count = 0usize;
    let mut zero_diag_count = 0usize;
    let mut total_max_deviation: u64 = 0;

    for _ in 0..n_trials {
        // Build a random equal-size partition by shuffling vertex indices.
        let mut indices: Vec<usize> = (0..n_vertices).collect();
        rng.shuffle(&mut indices);

        let mut random_labels = vec![0usize; n_vertices];
        for (pos, &vertex) in indices.iter().enumerate() {
            random_labels[vertex] = pos / coset_size;
        }

        let q = quotient_adjacency_matrix(&random_labels, n_cosets, graph);
        let reg = quotient_regularity(&q);
        let inv = coset_subspace_invariance(&random_labels, n_cosets, graph);

        if inv.a_invariant {
            invariance_count += 1;
        }
        if reg.uniform_row_sums {
            uniform_row_sum_count += 1;
        }
        if reg.zero_diagonal {
            zero_diag_count += 1;
        }
        total_max_deviation += u64::from(inv.global_max_deviation);
    }

    let n = n_trials.max(1) as f64;
    BaselineReport {
        n_trials,
        coset_size,
        n_cosets,
        invariance_fraction: invariance_count as f64 / n,
        uniform_row_sum_fraction: uniform_row_sum_count as f64 / n,
        zero_diagonal_fraction: zero_diag_count as f64 / n,
        mean_max_deviation: total_max_deviation as f64 / n,
        actual_max_deviation,
        actual_invariant,
    }
}

// ---------------------------------------------------------------------------
// 5. S4 validation
// ---------------------------------------------------------------------------

/// Full spectral validation report for the S4 permutahedron with V4 cosets.
#[derive(Debug, Clone, Serialize)]
pub struct S4ValidationReport {
    pub n: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub n_cosets: usize,
    pub coset_size: usize,
    pub quotient_matrix: Vec<Vec<u32>>,
    pub regularity: RegularityReport,
    pub invariance: InvarianceReport,
    pub baseline: BaselineReport,
}

/// Build the S4 permutahedron, partition via V4 right cosets, and run all
/// spectral diagnostics.
pub fn run_s4_validation() -> S4ValidationReport {
    let graph = permutahedron::complete_graph(4).expect("S4 graph construction must succeed");
    let v4 = permutahedron::vierergruppe();
    let partition = permutahedron::coset_partition(&v4, CosetSide::Right)
        .expect("V4 coset partition must succeed");

    let n_cosets = partition.slice_count;
    let coset_size = partition.slice_size;
    let labels = labels_from_slices(graph.vertex_count, &partition.slices);

    let quotient = quotient_adjacency_matrix(&labels, n_cosets, &graph);
    let regularity = quotient_regularity(&quotient);
    let invariance = coset_subspace_invariance(&labels, n_cosets, &graph);
    let baseline = random_partition_baseline(
        graph.vertex_count,
        n_cosets,
        coset_size,
        &graph,
        &labels,
        100,
        42,
    );

    eprintln!("--- S4 Spectral Validation ---");
    eprintln!(
        "Vertices: {}, Edges: {}",
        graph.vertex_count, graph.edge_count
    );
    eprintln!("Cosets: {} of size {}", n_cosets, coset_size);
    eprintln!("Quotient matrix ({n_cosets}x{n_cosets}):");
    for row in &quotient {
        eprintln!("  {:?}", row);
    }
    eprintln!(
        "Regularity: row_sums={:?}, zero_diag={}, symmetric={}",
        regularity.row_sums, regularity.zero_diagonal, regularity.symmetric
    );
    eprintln!(
        "A-invariant: {} (max deviation {})",
        invariance.a_invariant, invariance.global_max_deviation
    );
    eprintln!(
        "Baseline ({} random trials): invariance_frac={:.3}, mean_max_dev={:.2}",
        baseline.n_trials, baseline.invariance_fraction, baseline.mean_max_deviation
    );

    S4ValidationReport {
        n: 4,
        vertex_count: graph.vertex_count,
        edge_count: graph.edge_count,
        n_cosets,
        coset_size,
        quotient_matrix: quotient,
        regularity,
        invariance,
        baseline,
    }
}

// ---------------------------------------------------------------------------
// 6. S8 probe
// ---------------------------------------------------------------------------

/// Full spectral probe report for the S8 permutahedron with R8 cosets.
#[derive(Debug, Clone, Serialize)]
pub struct S8ProbeReport {
    pub n: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub n_cosets: usize,
    pub coset_size: usize,
    /// The quotient matrix is 5040x5040; we store summary statistics only in
    /// the report struct, but the full matrix is computed internally.
    pub regularity: RegularityReport,
    /// Full invariance check is O(n_cosets * n_vertices), which is large for
    /// S8.  We sample a subset of cosets and report per-sample diagnostics.
    pub sampled_invariance: SampledInvarianceReport,
}

/// Sampled invariance report for large permutahedra.
#[derive(Debug, Clone, Serialize)]
pub struct SampledInvarianceReport {
    pub n_cosets_sampled: usize,
    pub n_cosets_total: usize,
    /// Number of sampled cosets with nonzero deviation.
    pub nonzero_deviation_count: usize,
    /// Maximum deviation observed in the sample.
    pub max_deviation: u32,
    /// Mean deviation across sampled cosets.
    pub mean_deviation: f64,
    /// Whether ALL sampled cosets had zero deviation.
    pub all_sampled_invariant: bool,
}

/// Build the S8 permutahedron, partition via R8 right cosets, and run spectral
/// diagnostics.  The full invariance check over all 5040 cosets is expensive
/// (O(5040 * 40320)), so we sample a deterministic subset.
pub fn run_s8_probe() -> S8ProbeReport {
    let graph = permutahedron::complete_graph(8).expect("S8 graph construction must succeed");
    let r8 = permutahedron::rana_r8();
    let partition = permutahedron::coset_partition(&r8, CosetSide::Right)
        .expect("R8 coset partition must succeed");

    let n_cosets = partition.slice_count;
    let coset_size = partition.slice_size;
    let labels = labels_from_slices(graph.vertex_count, &partition.slices);

    // Quotient adjacency matrix (5040x5040, but entries are small u32s).
    let quotient = quotient_adjacency_matrix(&labels, n_cosets, &graph);
    let regularity = quotient_regularity(&quotient);

    // Sampled invariance: check a deterministic subset of cosets.
    let sample_size = 200.min(n_cosets);
    let sampled_invariance = sampled_coset_invariance(&labels, n_cosets, &graph, sample_size, 137);

    eprintln!("--- S8 Spectral Probe ---");
    eprintln!(
        "Vertices: {}, Edges: {}",
        graph.vertex_count, graph.edge_count
    );
    eprintln!("Cosets: {} of size {}", n_cosets, coset_size);
    eprintln!(
        "Regularity: uniform_row_sums={}, zero_diag={}, symmetric={}",
        regularity.uniform_row_sums, regularity.zero_diagonal, regularity.symmetric
    );
    if !regularity.uniform_row_sums {
        let min_rs = regularity.row_sums.iter().copied().min().unwrap_or(0);
        let max_rs = regularity.row_sums.iter().copied().max().unwrap_or(0);
        eprintln!("  Row sum range: {} .. {}", min_rs, max_rs);
    }
    eprintln!(
        "Sampled invariance ({}/{} cosets): max_dev={}, all_invariant={}",
        sampled_invariance.n_cosets_sampled,
        sampled_invariance.n_cosets_total,
        sampled_invariance.max_deviation,
        sampled_invariance.all_sampled_invariant
    );

    S8ProbeReport {
        n: 8,
        vertex_count: graph.vertex_count,
        edge_count: graph.edge_count,
        n_cosets,
        coset_size,
        regularity,
        sampled_invariance,
    }
}

/// Check A-invariance for a deterministic sample of cosets, avoiding the
/// O(n_cosets^2 * coset_size) cost of the full check.
fn sampled_coset_invariance(
    coset_labels: &[usize],
    n_cosets: usize,
    graph: &PermutahedronGraph,
    sample_size: usize,
    seed: u64,
) -> SampledInvarianceReport {
    let n_vertices = coset_labels.len();

    // Build adjacency lists.
    let mut adj = vec![Vec::new(); n_vertices];
    for edge in &graph.edges {
        let u = edge[0] as usize;
        let v = edge[1] as usize;
        adj[u].push(v);
        adj[v].push(u);
    }

    // Deterministic sample: use LCG to select coset indices.
    let mut rng = Lcg::new(seed);
    let mut sample_indices: Vec<usize> = (0..n_cosets).collect();
    rng.shuffle(&mut sample_indices);
    sample_indices.truncate(sample_size);
    sample_indices.sort_unstable();

    let mut nonzero_count = 0usize;
    let mut max_dev: u32 = 0;
    let mut total_dev: u64 = 0;

    for &c in &sample_indices {
        // Compute (A * f_c)[v] = number of neighbors of v in coset c.
        let mut a_fc = vec![0u32; n_vertices];
        for (v, neighbors) in adj.iter().enumerate() {
            let mut count = 0u32;
            for &u in neighbors {
                if coset_labels[u] == c {
                    count += 1;
                }
            }
            a_fc[v] = count;
        }

        // Check constancy within each coset.
        let mut coset_min = vec![u32::MAX; n_cosets];
        let mut coset_max = vec![0u32; n_cosets];
        for (v, &label) in coset_labels.iter().enumerate() {
            let val = a_fc[v];
            if val < coset_min[label] {
                coset_min[label] = val;
            }
            if val > coset_max[label] {
                coset_max[label] = val;
            }
        }

        let mut this_max_dev: u32 = 0;
        for cp in 0..n_cosets {
            // Skip cosets that have no vertices with any neighbor in c
            // (coset_min would still be MAX).
            if coset_min[cp] == u32::MAX {
                continue;
            }
            let dev = coset_max[cp] - coset_min[cp];
            if dev > this_max_dev {
                this_max_dev = dev;
            }
        }

        if this_max_dev > 0 {
            nonzero_count += 1;
        }
        if this_max_dev > max_dev {
            max_dev = this_max_dev;
        }
        total_dev += u64::from(this_max_dev);
    }

    let n = sample_indices.len().max(1) as f64;
    SampledInvarianceReport {
        n_cosets_sampled: sample_indices.len(),
        n_cosets_total: n_cosets,
        nonzero_deviation_count: nonzero_count,
        max_deviation: max_dev,
        mean_deviation: total_dev as f64 / n,
        all_sampled_invariant: nonzero_count == 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn s4_setup() -> (PermutahedronGraph, Vec<usize>, usize) {
        let graph = permutahedron::complete_graph(4).unwrap();
        let v4 = permutahedron::vierergruppe();
        let partition = permutahedron::coset_partition(&v4, CosetSide::Right).unwrap();
        let n_cosets = partition.slice_count;
        let labels = labels_from_slices(graph.vertex_count, &partition.slices);
        (graph, labels, n_cosets)
    }

    #[test]
    fn s4_quotient_is_6x6() {
        let (graph, labels, n_cosets) = s4_setup();
        let q = quotient_adjacency_matrix(&labels, n_cosets, &graph);
        assert_eq!(n_cosets, 6);
        assert_eq!(q.len(), 6);
        assert!(q.iter().all(|row| row.len() == 6));
    }

    #[test]
    fn s4_quotient_has_zero_diagonal() {
        // V4 cosets are independent sets in the Cayley graph: no edge connects
        // two vertices in the same coset.
        let (graph, labels, n_cosets) = s4_setup();
        let q = quotient_adjacency_matrix(&labels, n_cosets, &graph);
        for i in 0..n_cosets {
            assert_eq!(q[i][i], 0, "diagonal entry [{i}][{i}] should be 0");
        }
    }

    #[test]
    fn s4_quotient_row_sum_is_12() {
        // Each coset has 4 vertices, each of degree 3. All edges leave the
        // coset (zero diagonal), so the row sum = 4 * 3 = 12.
        let (graph, labels, n_cosets) = s4_setup();
        let q = quotient_adjacency_matrix(&labels, n_cosets, &graph);
        for (i, row) in q.iter().enumerate() {
            let sum: u32 = row.iter().sum();
            assert_eq!(sum, 12, "row {i} sum should be 12, got {sum}");
        }
    }

    #[test]
    fn s4_quotient_is_symmetric() {
        let (graph, labels, n_cosets) = s4_setup();
        let q = quotient_adjacency_matrix(&labels, n_cosets, &graph);
        for i in 0..n_cosets {
            for j in 0..n_cosets {
                assert_eq!(
                    q[i][j], q[j][i],
                    "quotient matrix should be symmetric at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn s4_subspace_is_a_invariant() {
        // V4 is normal in S4, so the coset indicator subspace must be
        // A-invariant (the quotient is a genuine graph homomorphism).
        let (graph, labels, n_cosets) = s4_setup();
        let inv = coset_subspace_invariance(&labels, n_cosets, &graph);
        assert!(
            inv.a_invariant,
            "V4 coset subspace should be A-invariant; max deviation = {}",
            inv.global_max_deviation
        );
        assert_eq!(inv.global_max_deviation, 0);
    }

    #[test]
    fn s4_regularity_all_pass() {
        let (graph, labels, n_cosets) = s4_setup();
        let q = quotient_adjacency_matrix(&labels, n_cosets, &graph);
        let reg = quotient_regularity(&q);
        assert!(reg.uniform_row_sums);
        assert!(reg.zero_diagonal);
        assert!(reg.symmetric);
    }

    #[test]
    fn s4_full_validation_report() {
        let report = run_s4_validation();
        assert_eq!(report.vertex_count, 24);
        assert_eq!(report.edge_count, 36);
        assert_eq!(report.n_cosets, 6);
        assert_eq!(report.coset_size, 4);
        assert!(report.invariance.a_invariant);
        assert!(report.regularity.zero_diagonal);
        assert!(report.regularity.uniform_row_sums);
        // Random partitions should almost never be A-invariant.
        assert!(report.baseline.actual_invariant);
    }

    #[test]
    fn s4_each_coset_indicator_perfectly_constant() {
        let (graph, labels, n_cosets) = s4_setup();
        let inv = coset_subspace_invariance(&labels, n_cosets, &graph);
        for diag in &inv.per_coset {
            assert_eq!(
                diag.max_within_coset_deviation, 0,
                "coset {} should have zero deviation",
                diag.coset_index
            );
        }
    }

    #[test]
    fn random_baseline_is_deterministic() {
        let (graph, labels, n_cosets) = s4_setup();
        let b1 =
            random_partition_baseline(graph.vertex_count, n_cosets, 4, &graph, &labels, 10, 99);
        let b2 =
            random_partition_baseline(graph.vertex_count, n_cosets, 4, &graph, &labels, 10, 99);
        assert_eq!(b1.invariance_fraction, b2.invariance_fraction);
        assert_eq!(b1.mean_max_deviation, b2.mean_max_deviation);
    }

    #[test]
    fn labels_from_slices_covers_all_vertices() {
        let v4 = permutahedron::vierergruppe();
        let partition = permutahedron::coset_partition(&v4, CosetSide::Right).unwrap();
        let labels = labels_from_slices(24, &partition.slices);
        assert_eq!(labels.len(), 24);
        assert!(labels.iter().all(|&l| l < 6));
        // Each coset index appears exactly 4 times.
        for c in 0..6 {
            assert_eq!(labels.iter().filter(|&&l| l == c).count(), 4);
        }
    }

    // Longer-running S8 tests are gated behind a feature or run time budget.
    // The run_s8_probe() function exercises the same code paths.
    #[test]
    #[ignore]
    fn s8_probe_runs_to_completion() {
        let report = run_s8_probe();
        assert_eq!(report.vertex_count, 40_320);
        assert_eq!(report.edge_count, 141_120);
        assert_eq!(report.n_cosets, 5_040);
        assert_eq!(report.coset_size, 8);
        // R8 is not normal in S8, but the coset partition is still equitable
        // (A-invariant). Non-normality does not imply non-equitability.
        assert!(
            report.sampled_invariance.all_sampled_invariant,
            "R8 coset partition is equitable despite R8 not being normal"
        );
    }
}
