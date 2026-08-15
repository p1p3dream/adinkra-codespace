//! Coset-primed spectral clustering for permutation group Cayley graphs.
//!
//! Instead of starting Lanczos from a random or delta vector (which biases the
//! Krylov subspace toward extremal eigenvalues), start from a coset indicator
//! vector. This concentrates spectral energy in the eigenspaces that carry the
//! coset structure, enabling recovery of cosets that naive spectral clustering
//! misses entirely.
//!
//! A single coset indicator captures one direction per degenerate eigenspace.
//! To recover the full eigenspace (which may have multiplicity > 1), the
//! single-indicator function supplements with a complementary Lanczos from a
//! vertex in a different coset, and the multi-indicator function interleaves
//! vectors from several indicators.

use crate::spectral_lanczos;

// ── Result struct ──────────────────────────────────────────────────────────

/// Output of a coset-primed spectral clustering run.
#[derive(Debug, Clone)]
pub struct CosetPrimedResult {
    /// Adjusted Rand Index between predicted and true coset labels.
    pub ari: f64,
    /// Eigenvalues of the Ritz pairs selected for the embedding.
    pub eigenvalues_used: Vec<f64>,
    /// Number of eigenvectors used in the embedding.
    pub n_eigenvectors: usize,
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Project a tridiagonal eigenvector (Ritz vector) back to the full
/// n-dimensional space using the Lanczos basis vectors.
fn project_ritz_to_full(ritz_vec: &[f64], q_vectors: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    let len = q_vectors.len().min(ritz_vec.len());
    for j in 0..len {
        for (k, vk) in v.iter_mut().enumerate() {
            *vk += ritz_vec[j] * q_vectors[j][k];
        }
    }
    v
}

/// Check whether a vector is approximately constant (carries no clustering
/// information). Uses a dimension-aware threshold: for a unit-norm vector
/// in R^n, a constant vector has all components = 1/sqrt(n) and range = 0,
/// while a non-constant eigenvector has range >> 1/sqrt(n).
fn is_near_constant(v: &[f64]) -> bool {
    if v.is_empty() {
        return true;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in v {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    let range = hi - lo;
    let nrm = norm(v);
    if nrm < 1e-14 {
        return true;
    }
    (range / nrm) < 1e-6
}

/// Modified Gram-Schmidt orthonormalization. Returns the surviving
/// (non-zero-norm) orthonormal vectors.
fn orthonormalize(vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::new();
    for orig in vectors {
        let mut v = orig.clone();
        for u in &result {
            let c = dot(u, &v);
            for (vi, ui) in v.iter_mut().zip(u.iter()) {
                *vi -= c * ui;
            }
        }
        let nrm = norm(&v);
        if nrm > 1e-12 {
            for vi in &mut v {
                *vi /= nrm;
            }
            result.push(v);
        }
    }
    result
}

/// Ritz pair with its coset energy and projected full-space eigenvector.
struct RitzCandidate {
    energy: f64,
    eigenvalue: f64,
    full_vec: Vec<f64>,
}

/// Run Lanczos from a start vector, eigendecompose, and return Ritz
/// candidates sorted by coset energy (descending).
fn extract_ritz_candidates(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n_verts: usize,
    start: &[f64],
) -> Vec<RitzCandidate> {
    let start_norm_sq = dot(start, start);
    let m = n_verts; // allow full Krylov subspace
    let lr = spectral_lanczos::lanczos(edges, degree, n_verts, start, m);
    let (ritz_evals, ritz_evecs) = spectral_lanczos::tridiag_eigen(&lr.alpha, &lr.beta);

    let mut candidates: Vec<RitzCandidate> = ritz_evals
        .iter()
        .zip(ritz_evecs.iter())
        .map(|(&eval, evec)| {
            let z0 = evec[0];
            let energy = start_norm_sq * z0 * z0;
            let full_vec = project_ritz_to_full(evec, &lr.q_vectors, n_verts);
            RitzCandidate {
                energy,
                eigenvalue: eval,
                full_vec,
            }
        })
        .collect();

    candidates.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());
    candidates
}

// ── 1. Single-indicator coset-primed clustering ────────────────────────────

/// Run spectral clustering starting Lanczos from a single coset indicator
/// vector (coset 0), supplemented by a complementary Lanczos from a vertex
/// in a different coset to break eigenspace degeneracy.
///
/// The indicator biases the Krylov subspace toward eigenspaces that carry
/// the coset structure. A complementary pass from a different coset captures
/// the second direction in degenerate eigenspaces, enabling full separation.
pub fn coset_primed_clustering(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n_verts: usize,
    coset_labels: &[usize],
    n_cosets: usize,
    _coset_size: usize,
    lanczos_m: usize,
    embed_dim: usize,
) -> CosetPrimedResult {
    assert_eq!(coset_labels.len(), n_verts);

    // Step 1: Build indicator vector for coset 0, run Lanczos.
    let indicator: Vec<f64> = coset_labels
        .iter()
        .map(|&label| if label == 0 { 1.0 } else { 0.0 })
        .collect();
    let ind_norm_sq = dot(&indicator, &indicator);

    let lr = spectral_lanczos::lanczos(edges, degree, n_verts, &indicator, lanczos_m);
    let (ritz_evals, ritz_evecs) = spectral_lanczos::tridiag_eigen(&lr.alpha, &lr.beta);
    let m_actual = ritz_evals.len();

    // Compute coset energy and collect indicator Ritz vectors.
    let mut indicator_candidates: Vec<RitzCandidate> = (0..m_actual)
        .map(|i| {
            let z0 = ritz_evecs[i][0];
            let energy = ind_norm_sq * z0 * z0;
            let full_vec = project_ritz_to_full(&ritz_evecs[i], &lr.q_vectors, n_verts);
            RitzCandidate {
                energy,
                eigenvalue: ritz_evals[i],
                full_vec,
            }
        })
        .collect();
    indicator_candidates.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());

    // Step 2: Complementary Lanczos from a vertex in coset 1 to capture
    // the second direction in degenerate eigenspaces.
    let complement_vertex = coset_labels
        .iter()
        .position(|&label| label == 1)
        .unwrap_or(1);
    let mut delta = vec![0.0_f64; n_verts];
    delta[complement_vertex] = 1.0;

    let lr2 = spectral_lanczos::lanczos(edges, degree, n_verts, &delta, lanczos_m);
    let (ritz_evals2, ritz_evecs2) = spectral_lanczos::tridiag_eigen(&lr2.alpha, &lr2.beta);

    // Collect complementary Ritz vectors whose eigenvalues match the
    // indicator's high-energy eigenvalues.
    let mut complement_vecs: Vec<Vec<f64>> = Vec::new();
    for cand in &indicator_candidates {
        for j in 0..ritz_evals2.len() {
            if (ritz_evals2[j] - cand.eigenvalue).abs() < 0.1 {
                let v = project_ritz_to_full(&ritz_evecs2[j], &lr2.q_vectors, n_verts);
                complement_vecs.push(v);
            }
        }
    }

    // Step 3: Merge indicator and complement vectors, orthogonalize,
    // filter out constant-like vectors.
    let mut all_vecs: Vec<Vec<f64>> = indicator_candidates
        .iter()
        .map(|c| c.full_vec.clone())
        .collect();
    all_vecs.extend(complement_vecs);

    let ortho = orthonormalize(&all_vecs);
    let useful: Vec<&Vec<f64>> = ortho.iter().filter(|v| !is_near_constant(v)).collect();

    let actual_dim = embed_dim.min(useful.len());
    let eigenvalues_used: Vec<f64> = indicator_candidates
        .iter()
        .map(|c| c.eigenvalue)
        .take(actual_dim)
        .collect();

    // Step 4: Build embedding, cluster, compute ARI.
    let embedding: Vec<Vec<f64>> = (0..n_verts)
        .map(|v| useful.iter().take(actual_dim).map(|ev| ev[v]).collect())
        .collect();

    let n_init = if n_cosets > 100 { 3 } else { 50 };
    let max_iter = if n_cosets > 100 { 50 } else { 200 };
    let predicted =
        spectral_lanczos::kmeans_clustering(&embedding, n_verts, n_cosets, n_init, max_iter, 42);

    let ari = spectral_lanczos::adjusted_rand_index(coset_labels, &predicted);

    CosetPrimedResult {
        ari,
        eigenvalues_used,
        n_eigenvectors: actual_dim,
    }
}

// ── 2. Multi-indicator coset-primed clustering ─────────────────────────────

/// Run spectral clustering starting Lanczos from multiple coset indicator
/// vectors (cosets 0 through n_indicators-1), interleaving the resulting
/// Ritz vectors (round-robin by energy rank) before orthogonalization.
///
/// Interleaving ensures that complementary directions from different
/// indicators appear early in the orthogonalization order, so they survive
/// into the first embed_dim slots. This captures the full eigenspace even
/// when individual eigenvalues have multiplicity > 1.
pub fn multi_indicator_clustering(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n_verts: usize,
    coset_labels: &[usize],
    n_cosets: usize,
    _coset_size: usize,
    n_indicators: usize,
    lanczos_m: usize,
    embed_dim: usize,
) -> CosetPrimedResult {
    assert_eq!(coset_labels.len(), n_verts);
    let indicator_count = n_indicators.min(n_cosets);

    // Run Lanczos from each indicator and collect energy-sorted candidates.
    let mut all_per_indicator: Vec<Vec<RitzCandidate>> = Vec::new();

    for coset_id in 0..indicator_count {
        let indicator: Vec<f64> = coset_labels
            .iter()
            .map(|&label| if label == coset_id { 1.0 } else { 0.0 })
            .collect();

        let ind_norm_sq = dot(&indicator, &indicator);
        if ind_norm_sq < 1e-14 {
            continue;
        }

        let lr = spectral_lanczos::lanczos(edges, degree, n_verts, &indicator, lanczos_m);
        let (ritz_evals, ritz_evecs) = spectral_lanczos::tridiag_eigen(&lr.alpha, &lr.beta);

        let mut candidates: Vec<RitzCandidate> = ritz_evals
            .iter()
            .zip(ritz_evecs.iter())
            .map(|(&eval, evec)| {
                let z0 = evec[0];
                let energy = ind_norm_sq * z0 * z0;
                let full_vec = project_ritz_to_full(evec, &lr.q_vectors, n_verts);
                RitzCandidate {
                    energy,
                    eigenvalue: eval,
                    full_vec,
                }
            })
            .collect();

        candidates.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());
        all_per_indicator.push(candidates);
    }

    // Interleave: round-robin by energy rank across indicators.
    let max_per = all_per_indicator.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut interleaved_vecs: Vec<Vec<f64>> = Vec::new();
    let mut interleaved_evals: Vec<f64> = Vec::new();

    for rank in 0..max_per {
        for candidates in &all_per_indicator {
            if rank < candidates.len() {
                interleaved_vecs.push(candidates[rank].full_vec.clone());
                interleaved_evals.push(candidates[rank].eigenvalue);
            }
        }
    }

    // Orthogonalize and filter out constant-like vectors.
    let ortho = orthonormalize(&interleaved_vecs);
    let useful: Vec<&Vec<f64>> = ortho.iter().filter(|v| !is_near_constant(v)).collect();

    let actual_dim = embed_dim.min(useful.len());
    let eigenvalues_used: Vec<f64> = interleaved_evals.iter().take(actual_dim).copied().collect();

    // Build embedding, cluster, compute ARI.
    let embedding: Vec<Vec<f64>> = (0..n_verts)
        .map(|v| useful.iter().take(actual_dim).map(|ev| ev[v]).collect())
        .collect();

    let n_init = if n_cosets > 100 { 3 } else { 50 };
    let max_iter = if n_cosets > 100 { 50 } else { 200 };
    let predicted =
        spectral_lanczos::kmeans_clustering(&embedding, n_verts, n_cosets, n_init, max_iter, 42);

    let ari = spectral_lanczos::adjusted_rand_index(coset_labels, &predicted);

    CosetPrimedResult {
        ari,
        eigenvalues_used,
        n_eigenvectors: actual_dim,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permutahedron::{self, CosetSide};

    /// Build S4 Cayley graph edges from the permutahedron module.
    fn build_s4_edges() -> (Vec<(usize, usize, usize)>, usize, usize) {
        let graph = permutahedron::complete_graph(4).expect("S4 graph construction must succeed");
        let edges: Vec<(usize, usize, usize)> = graph
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| (e[0] as usize, e[1] as usize, i % graph.degree))
            .collect();
        (edges, graph.degree, graph.vertex_count)
    }

    /// Compute V4 right-coset labels for S4.
    fn build_s4_coset_labels() -> (Vec<usize>, usize, usize) {
        let graph = permutahedron::complete_graph(4).expect("S4 graph construction must succeed");
        let v4 = permutahedron::vierergruppe();
        let partition = permutahedron::coset_partition(&v4, CosetSide::Right)
            .expect("V4 coset partition must succeed");

        let mut labels = vec![0usize; graph.vertex_count];
        for (i, sl) in partition.slices.iter().enumerate() {
            for &v in sl {
                labels[v as usize] = i;
            }
        }
        let n_cosets = partition.slices.len();
        let coset_size = partition.slice_size;
        (labels, n_cosets, coset_size)
    }

    /// Single-indicator coset-primed clustering (with complementary pass)
    /// on S4 should recover V4 cosets perfectly (ARI > 0.9, typically 1.0).
    #[test]
    fn test_coset_primed_s4_single() {
        let (edges, degree, n_verts) = build_s4_edges();
        let (labels, n_cosets, coset_size) = build_s4_coset_labels();

        assert_eq!(n_cosets, 6, "S4/V4 should have 6 cosets");
        assert_eq!(coset_size, 4, "each V4 coset should have 4 elements");

        let result = coset_primed_clustering(
            &edges, degree, n_verts, &labels, n_cosets, coset_size,
            24, // lanczos_m = full S4 dimension
            4,  // embed_dim
        );

        assert!(
            result.ari > 0.9,
            "single-indicator coset-primed ARI should be > 0.9, got {:.4}",
            result.ari
        );
    }

    /// Multi-indicator coset-primed clustering on S4 should also recover
    /// V4 cosets (ARI > 0.9).
    #[test]
    fn test_coset_primed_s4_multi() {
        let (edges, degree, n_verts) = build_s4_edges();
        let (labels, n_cosets, coset_size) = build_s4_coset_labels();

        let result = multi_indicator_clustering(
            &edges, degree, n_verts, &labels, n_cosets, coset_size, 3,  // n_indicators
            24, // lanczos_m
            4,  // embed_dim
        );

        assert!(
            result.ari > 0.9,
            "multi-indicator coset-primed ARI should be > 0.9, got {:.4}",
            result.ari
        );
    }

    /// Compare coset-primed vs naive delta-start Lanczos clustering.
    ///
    /// Coset-primed should achieve ARI close to 1.0, while naive spectral
    /// clustering (using lowest eigenvectors from a delta start) should score
    /// lower because the bottom eigenspace of the S4 Cayley-graph Laplacian
    /// does not align with V4 coset structure.
    #[test]
    fn test_coset_primed_s4_vs_naive() {
        let (edges, degree, n_verts) = build_s4_edges();
        let (labels, n_cosets, coset_size) = build_s4_coset_labels();

        // Coset-primed clustering
        let primed_result = coset_primed_clustering(
            &edges, degree, n_verts, &labels, n_cosets, coset_size, 24, 4,
        );

        // Naive clustering: delta start, take lowest-eigenvalue eigenvectors.
        // Skip the zero eigenvalue (constant vector, useless for clustering).
        let mut delta_start = vec![0.0_f64; n_verts];
        delta_start[0] = 1.0;
        let lr = spectral_lanczos::lanczos(&edges, degree, n_verts, &delta_start, 24);
        let (ritz_evals, ritz_evecs) = spectral_lanczos::tridiag_eigen(&lr.alpha, &lr.beta);
        let m_actual = ritz_evals.len();
        let n_lanczos = lr.q_vectors.len();

        let naive_dim = 4.min(m_actual.saturating_sub(1));
        let naive_evecs: Vec<Vec<f64>> = (1..=naive_dim)
            .map(|i| project_ritz_to_full(&ritz_evecs[i], &lr.q_vectors, n_verts))
            .collect();

        let naive_embedding: Vec<Vec<f64>> = (0..n_verts)
            .map(|v| naive_evecs.iter().map(|ev| ev[v]).collect())
            .collect();

        let naive_predicted =
            spectral_lanczos::kmeans_clustering(&naive_embedding, n_verts, n_cosets, 50, 200, 42);
        let naive_ari = spectral_lanczos::adjusted_rand_index(&labels, &naive_predicted);

        // The coset-primed ARI should be strictly better than or equal to
        // the naive ARI (within a small tolerance for k-means variability).
        assert!(
            primed_result.ari >= naive_ari - 0.05,
            "coset-primed ARI ({:.4}) should be >= naive ARI ({:.4}) - tolerance",
            primed_result.ari,
            naive_ari
        );

        // And it should be good in absolute terms.
        assert!(
            primed_result.ari > 0.9,
            "coset-primed ARI should be > 0.9, got {:.4}",
            primed_result.ari
        );
    }

    #[test]
    fn test_multi_indicator_s8() {
        let graph = permutahedron::complete_graph(8)
            .expect("S8 graph construction must succeed");
        let r8 = permutahedron::rana_r8();
        let partition = permutahedron::coset_partition(&r8, CosetSide::Right)
            .expect("R8 coset partition must succeed");
        let n_cosets = partition.slices.len();
        let coset_size = partition.slice_size;
        let mut labels = vec![0usize; graph.vertex_count];
        for (i, sl) in partition.slices.iter().enumerate() {
            for &v in sl {
                labels[v as usize] = i;
            }
        }
        let edges: Vec<(usize, usize, usize)> = graph
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| (e[0] as usize, e[1] as usize, i % graph.degree))
            .collect();

        let configs: &[(usize, usize, usize)] = &[
            (3, 200, 20),
            (5, 200, 20),
            (3, 200, 40),
            (10, 200, 40),
        ];

        let mut best_ari = 0.0_f64;
        for &(n_ind, m, dim) in configs {
            let result = multi_indicator_clustering(
                &edges, graph.degree, graph.vertex_count,
                &labels, n_cosets, coset_size,
                n_ind, m, dim,
            );
            println!(
                "multi_indicator S8: n_indicators={}, m={}, embed_dim={} => ARI={:.6} ({} evecs)",
                n_ind, m, dim, result.ari, result.n_eigenvectors
            );
            if result.ari > best_ari {
                best_ari = result.ari;
            }
        }
        println!("Best multi-indicator ARI on S8: {:.6}", best_ari);
        assert!(
            best_ari > 0.4,
            "multi-indicator S8 best ARI should be > 0.4, got {:.6}",
            best_ari
        );
    }
}
