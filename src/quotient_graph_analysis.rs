//! Quotient graph structure analysis for the spectral pipeline.
//!
//! When a coset partition is equitable (A-invariant), the quotient graph is
//! well-defined. This module provides structural analyses of that quotient:
//!
//! - **Eigendecomposition**: dense Jacobi for n <= 100, Lanczos for n > 100
//! - **Distance-regularity check**: BFS-based intersection array computation
//! - **Orbit structure**: row-profile grouping as an upper bound on orbits
//! - **Spectral comparison**: match quotient eigenvalues against the full spectrum

use serde::Serialize;
use std::collections::BTreeMap;

// ── Report structs ─────────────────────────────────────────────────────────

/// A group of eigenvalues clustered within tolerance.
#[derive(Debug, Clone, Serialize)]
pub struct EigenBucket {
    /// Representative eigenvalue (centroid of the cluster).
    pub eigenvalue: f64,
    /// Number of eigenvalues in this cluster.
    pub multiplicity: usize,
}

/// Result of the quotient graph eigendecomposition.
#[derive(Debug, Clone, Serialize)]
pub struct QuotientEigenReport {
    /// Total number of eigenvalues found.
    pub n_eigenvalues: usize,
    /// Distinct eigenvalues grouped by tolerance, sorted ascending.
    pub distinct: Vec<EigenBucket>,
    /// Spectral gap: difference between the largest and second-largest eigenvalue.
    /// For a connected k-regular graph this measures expansion.
    pub spectral_gap: f64,
    /// Largest eigenvalue (equals degree for regular graphs).
    pub largest_eigenvalue: f64,
    /// Smallest eigenvalue.
    pub smallest_eigenvalue: f64,
    /// Whether the graph is bipartite (smallest eigenvalue = -largest eigenvalue).
    pub is_bipartite: bool,
    /// Row sum of the quotient matrix (weighted degree). Meaningful only if the
    /// graph is regular (all row sums equal).
    pub row_sum: u32,
    /// Raw sorted eigenvalues (ascending).
    pub raw_eigenvalues: Vec<f64>,
}

/// Result of the distance-regularity check on the simple (unweighted) quotient.
#[derive(Debug, Clone, Serialize)]
pub struct DistanceRegularityReport {
    /// Whether the simple quotient graph is distance-regular.
    pub is_distance_regular: bool,
    /// Graph diameter (longest shortest path).
    pub diameter: usize,
    /// Intersection array: for each distance d from 0 to diameter,
    /// the triple (c_d, a_d, b_d) counting neighbors at distances d-1, d, d+1
    /// from a reference vertex. Present only if distance-regular.
    pub intersection_array: Option<Vec<(u32, u32, u32)>>,
    /// Number of vertices sampled for the check.
    pub n_vertices_sampled: usize,
    /// Number of (u, v, d) triples that violated the constant-triple condition.
    pub violations: usize,
    /// Simple graph degree (number of nonzero entries per row).
    pub simple_degree: usize,
}

/// Row-profile-based orbit structure report.
#[derive(Debug, Clone, Serialize)]
pub struct OrbitReport {
    /// Number of distinct sorted row profiles.
    pub n_profiles: usize,
    /// Sizes of each profile class, sorted descending.
    pub profile_sizes: Vec<usize>,
    /// True if there is a single profile (all rows look the same up to
    /// column permutation), which is necessary for vertex-transitivity.
    pub is_vertex_transitive: bool,
}

/// Result of full eigenvector extraction from the quotient graph.
#[derive(Debug, Clone, Serialize)]
pub struct QuotientEigenFullReport {
    /// Eigenvalues sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors: eigenvectors[i] is the eigenvector for eigenvalues[i].
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of eigenpairs computed.
    pub n_computed: usize,
}

/// Comparison between full graph spectrum and quotient spectrum.
#[derive(Debug, Clone, Serialize)]
pub struct SpectralComparisonReport {
    /// Number of quotient eigenvalues matched in the full spectrum.
    pub matched: usize,
    /// Number of quotient eigenvalues not found in the full spectrum.
    pub unmatched: usize,
    /// The unmatched quotient eigenvalues (if any).
    pub unmatched_values: Vec<f64>,
    /// Tolerance used for matching.
    pub tolerance: f64,
}

// ── Internal linear algebra helpers ────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * *xi;
    }
}

fn scale(a: f64, v: &mut [f64]) {
    for vi in v.iter_mut() {
        *vi *= a;
    }
}

/// Dense matrix-vector product y = Q * v where Q is stored as Vec<Vec<u32>>.
fn dense_matvec(matrix: &[Vec<u32>], v: &[f64]) -> Vec<f64> {
    let n = matrix.len();
    let mut out = vec![0.0_f64; n];
    for (i, row) in matrix.iter().enumerate() {
        let mut s = 0.0_f64;
        for (j, &entry) in row.iter().enumerate() {
            if entry != 0 {
                s += entry as f64 * v[j];
            }
        }
        out[i] = s;
    }
    out
}

// ── Dense Jacobi eigensolver (for n <= 100) ────────────────────────────────

/// Jacobi rotation method for symmetric real matrices.
///
/// Returns (eigenvalues_sorted_ascending, eigenvectors) where eigenvectors[i]
/// is the eigenvector for eigenvalues[i].
fn jacobi_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![matrix[0][0]], vec![vec![1.0]]);
    }

    // Working copy of the matrix (stored flat for cache friendliness).
    let mut a: Vec<Vec<f64>> = matrix.to_vec();

    // Eigenvector accumulator starts as identity.
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_sweeps = 100;
    let eps = 1e-14;

    for _sweep in 0..max_sweeps {
        // Find the largest off-diagonal element.
        let mut max_val = 0.0_f64;
        let mut p = 0_usize;
        let mut q = 1_usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let abs_aij = a[i][j].abs();
                if abs_aij > max_val {
                    max_val = abs_aij;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < eps {
            break;
        }

        // Compute rotation parameters.
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let (cos, sin) = if (app - aqq).abs() < 1e-30 {
            // theta = pi/4
            let c = std::f64::consts::FRAC_1_SQRT_2;
            (c, c)
        } else {
            let tau = (aqq - app) / (2.0 * apq);
            // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            (c, s)
        };

        // Apply rotation to A: A <- G^T * A * G
        // Update rows and columns p, q.
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = cos * aip - sin * aiq;
            a[p][i] = a[i][p];
            a[i][q] = sin * aip + cos * aiq;
            a[q][i] = a[i][q];
        }

        let new_app = cos * cos * app - 2.0 * sin * cos * apq + sin * sin * aqq;
        let new_aqq = sin * sin * app + 2.0 * sin * cos * apq + cos * cos * aqq;
        a[p][p] = new_app;
        a[q][q] = new_aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Accumulate eigenvector rotation: V <- V * G
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos * vip - sin * viq;
            v[i][q] = sin * vip + cos * viq;
        }
    }

    // Extract eigenvalues from diagonal.
    let mut evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

    // Sort ascending and permute eigenvectors.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap());

    let sorted_evals: Vec<f64> = idx.iter().map(|&i| evals[i]).collect();
    // Eigenvectors: column i of V is the eigenvector for eigenvalue i.
    // V is stored row-major, so column i = [v[0][i], v[1][i], ...].
    let sorted_evecs: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| (0..n).map(|row| v[row][i]).collect())
        .collect();

    // Overwrite evals for the return
    evals = sorted_evals;

    (evals, sorted_evecs)
}

// ── Lanczos on adjacency matrix (for n > 100) ─────────────────────────────

/// Lanczos tridiagonalization with full reorthogonalization, operating on
/// the adjacency matrix of the quotient graph (not the Laplacian).
///
/// Returns (alpha, beta) of the tridiagonal matrix T where alpha is the
/// diagonal and beta the super/sub-diagonal.
fn adjacency_lanczos(matrix: &[Vec<u32>], m: usize) -> (Vec<f64>, Vec<f64>) {
    let n = matrix.len();
    let m = m.min(n);

    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m.saturating_sub(1));
    let mut q_vectors: Vec<Vec<f64>> = Vec::with_capacity(m);

    // Start vector: unit vector at index 0.
    let mut q = vec![0.0_f64; n];
    q[0] = 1.0;
    q_vectors.push(q);

    for j in 0..m {
        // w = A * q_j
        let mut w = dense_matvec(matrix, &q_vectors[j]);

        // alpha_j = q_j . w
        let a = dot(&q_vectors[j], &w);
        alpha.push(a);

        // w = w - alpha_j * q_j
        axpy(-a, &q_vectors[j], &mut w);

        // w = w - beta_{j-1} * q_{j-1}
        if j > 0 {
            axpy(-beta[j - 1], &q_vectors[j - 1], &mut w);
        }

        // Full reorthogonalization
        for k in 0..=j {
            let c = dot(&q_vectors[k], &w);
            axpy(-c, &q_vectors[k], &mut w);
        }

        let b = norm(&w);

        if j + 1 < m {
            beta.push(b);
            if b < 1e-14 {
                alpha.truncate(j + 1);
                beta.truncate(j);
                break;
            }
            scale(1.0 / b, &mut w);
            q_vectors.push(w);
        }
    }

    (alpha, beta)
}

/// Result of Lanczos tridiagonalization that retains the Lanczos basis vectors.
struct LanczosFullResult {
    /// Diagonal of the tridiagonal matrix.
    alpha: Vec<f64>,
    /// Sub/super-diagonal of the tridiagonal matrix.
    beta: Vec<f64>,
    /// The Lanczos basis vectors q_0, q_1, ..., q_{m-1}, each of length n.
    q_vectors: Vec<Vec<f64>>,
}

/// Lanczos tridiagonalization that retains the full Lanczos basis (Q matrix).
///
/// Identical to `adjacency_lanczos` but also returns the Lanczos vectors,
/// which are needed for projecting tridiagonal eigenvectors back to the
/// original space: full_evec = Q^T * ritz_vec.
fn adjacency_lanczos_full(matrix: &[Vec<u32>], m: usize) -> LanczosFullResult {
    let n = matrix.len();
    let m = m.min(n);

    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m.saturating_sub(1));
    let mut q_vectors: Vec<Vec<f64>> = Vec::with_capacity(m);

    // Start vector: unit vector at index 0.
    let mut q = vec![0.0_f64; n];
    q[0] = 1.0;
    q_vectors.push(q);

    for j in 0..m {
        // w = A * q_j
        let mut w = dense_matvec(matrix, &q_vectors[j]);

        // alpha_j = q_j . w
        let a = dot(&q_vectors[j], &w);
        alpha.push(a);

        // w = w - alpha_j * q_j
        axpy(-a, &q_vectors[j], &mut w);

        // w = w - beta_{j-1} * q_{j-1}
        if j > 0 {
            axpy(-beta[j - 1], &q_vectors[j - 1], &mut w);
        }

        // Full reorthogonalization
        for k in 0..=j {
            let c = dot(&q_vectors[k], &w);
            axpy(-c, &q_vectors[k], &mut w);
        }

        let b = norm(&w);

        if j + 1 < m {
            beta.push(b);
            if b < 1e-14 {
                alpha.truncate(j + 1);
                beta.truncate(j);
                break;
            }
            scale(1.0 / b, &mut w);
            q_vectors.push(w);
        }
    }

    LanczosFullResult {
        alpha,
        beta,
        q_vectors,
    }
}

// ── Tridiagonal eigensolver (QL with implicit shift) ───────────────────────
//
// Self-contained copy to keep this module independent of spectral_lanczos.
// Same algorithm as crate::spectral_lanczos::tridiag_eigen.

fn copysign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

fn tridiag_eigen(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = alpha.len();
    assert_eq!(beta.len(), n.saturating_sub(1));
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![alpha[0]], vec![vec![1.0]]);
    }

    let mut d = alpha.to_vec();
    let mut e = vec![0.0; n];
    for (i, &b) in beta.iter().enumerate() {
        e[i] = b;
    }

    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0; n];
            v[i] = 1.0;
            v
        })
        .collect();

    let max_iter_per = 100 * n;

    for l in 0..n {
        let mut iter_count = 0_usize;

        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * dd.max(1e-300) {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            assert!(
                iter_count < max_iter_per,
                "tridiag_eigen: QL iteration did not converge for index {l}"
            );
            iter_count += 1;

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = (g * g + 1.0).sqrt();
            g = d[m] - d[l] + e[l] / (g + copysign(r, g));

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            let mut converged_early = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let rr = f.hypot(g);
                e[i + 1] = rr;

                if rr.abs() < 1e-30 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    converged_early = true;
                    break;
                }

                s = f / rr;
                c = g / rr;
                g = d[i + 1] - p;
                let rr2 = (d[i] - g) * s + 2.0 * c * b;
                p = s * rr2;
                d[i + 1] = g + p;
                g = c * rr2 - b;

                for k in 0..n {
                    let t = z[i + 1][k];
                    z[i + 1][k] = s * z[i][k] + c * t;
                    z[i][k] = c * z[i][k] - s * t;
                }
            }

            if !converged_early {
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        }
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());

    let evals: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let evecs: Vec<Vec<f64>> = idx.iter().map(|&i| z[i].clone()).collect();

    (evals, evecs)
}

// ── Eigenvalue grouping ────────────────────────────────────────────────────

/// Group sorted eigenvalues into distinct buckets by tolerance.
fn group_eigenvalues(eigenvalues: &[f64], tol: f64) -> Vec<EigenBucket> {
    let mut buckets: Vec<EigenBucket> = Vec::new();
    for &ev in eigenvalues {
        if let Some(last) = buckets.last_mut() {
            if (ev - last.eigenvalue).abs() < tol {
                // Update centroid incrementally.
                let total = last.eigenvalue * last.multiplicity as f64 + ev;
                last.multiplicity += 1;
                last.eigenvalue = total / last.multiplicity as f64;
                continue;
            }
        }
        buckets.push(EigenBucket {
            eigenvalue: ev,
            multiplicity: 1,
        });
    }
    buckets
}

// ── 1. Quotient eigendecomposition ─────────────────────────────────────────

/// Compute the eigendecomposition of the quotient adjacency matrix.
///
/// For small matrices (n <= 100), uses the Jacobi rotation method (exact
/// for the given precision). For larger matrices, uses Lanczos
/// tridiagonalization with full reorthogonalization.
pub fn quotient_eigen(quotient: &[Vec<u32>]) -> QuotientEigenReport {
    let n = quotient.len();
    assert!(n > 0, "quotient matrix must be nonempty");

    let row_sum: u32 = quotient[0].iter().sum();

    let raw_eigenvalues = if n <= 100 {
        // Dense Jacobi path.
        let fmat: Vec<Vec<f64>> = quotient
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();
        let (evals, _evecs) = jacobi_eigen(&fmat);
        evals
    } else {
        // Lanczos path with m = min(n, 300) steps.
        let m = n.min(300);
        let (alpha, beta) = adjacency_lanczos(quotient, m);
        let (evals, _evecs) = tridiag_eigen(&alpha, &beta);
        evals
    };

    let tol = 0.001;
    let distinct = group_eigenvalues(&raw_eigenvalues, tol);

    let largest = *raw_eigenvalues.last().unwrap_or(&0.0);
    let smallest = *raw_eigenvalues.first().unwrap_or(&0.0);

    // Spectral gap: difference between largest and second-largest distinct eigenvalue.
    let spectral_gap = if distinct.len() >= 2 {
        let second_largest = distinct[distinct.len() - 2].eigenvalue;
        largest - second_largest
    } else {
        0.0
    };

    // Bipartite iff smallest eigenvalue = -largest eigenvalue (within tolerance).
    let is_bipartite = (smallest + largest).abs() < tol;

    QuotientEigenReport {
        n_eigenvalues: raw_eigenvalues.len(),
        distinct,
        spectral_gap,
        largest_eigenvalue: largest,
        smallest_eigenvalue: smallest,
        is_bipartite,
        row_sum,
        raw_eigenvalues,
    }
}

// ── 1b. Full eigenvector extraction ───────────────────────────────────────

/// Compute eigenvalues and eigenvectors of the quotient adjacency matrix.
///
/// For small matrices (n <= 100), uses Jacobi (all eigenpairs).
/// For larger matrices, uses Lanczos with m = min(n, max_vectors * 2) steps,
/// then projects tridiagonal eigenvectors back to the original space via
/// the Lanczos basis: full_evec = Q^T * ritz_vec.
///
/// Returns at most `max_vectors` eigenpairs (or all n if max_vectors >= n).
pub fn quotient_eigen_full(quotient: &[Vec<u32>], max_vectors: usize) -> QuotientEigenFullReport {
    let n = quotient.len();
    assert!(n > 0, "quotient matrix must be nonempty");

    let max_vectors = max_vectors.min(n);

    if n <= 100 {
        // Dense Jacobi path: returns all eigenpairs.
        let fmat: Vec<Vec<f64>> = quotient
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();
        let (evals, evecs) = jacobi_eigen(&fmat);
        let k = max_vectors.min(evals.len());
        QuotientEigenFullReport {
            eigenvalues: evals[..k].to_vec(),
            eigenvectors: evecs[..k].to_vec(),
            n_computed: k,
        }
    } else {
        // Lanczos path with projection.
        let m = n.min(max_vectors * 2);
        let lr = adjacency_lanczos_full(quotient, m);
        let (ritz_evals, ritz_vecs) = tridiag_eigen(&lr.alpha, &lr.beta);

        let k = max_vectors.min(ritz_evals.len());
        let q_dim = lr.q_vectors[0].len(); // = n (quotient dimension)
        let n_lanczos = lr.q_vectors.len(); // number of Lanczos vectors

        // Project Ritz vectors back to original space:
        // full_evec[i][j] = sum_over_l ritz_vecs[i][l] * q_vectors[l][j]
        let mut full_evecs: Vec<Vec<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            let mut evec = vec![0.0_f64; q_dim];
            for l in 0..n_lanczos {
                if l < ritz_vecs[i].len() {
                    axpy(ritz_vecs[i][l], &lr.q_vectors[l], &mut evec);
                }
            }
            // Normalize the projected eigenvector.
            let nrm = norm(&evec);
            if nrm > 1e-14 {
                scale(1.0 / nrm, &mut evec);
            }
            full_evecs.push(evec);
        }

        QuotientEigenFullReport {
            eigenvalues: ritz_evals[..k].to_vec(),
            eigenvectors: full_evecs,
            n_computed: k,
        }
    }
}

// ── 1c. Quotient lift embedding ──────────────────────────────────────────

/// Lift quotient eigenvectors to the full graph vertex space.
///
/// For each vertex v in the full graph with coset label `coset_labels[v]`,
/// its coordinate in dimension i is `quotient_eigenvectors[i][coset_labels[v]]`.
///
/// Returns an n_verts x embed_dim matrix (outer index = vertex, inner = dimension).
pub fn quotient_lift_embedding(
    quotient_eigenvectors: &[Vec<f64>],
    coset_labels: &[usize],
    n_verts: usize,
    embed_dim: usize,
) -> Vec<Vec<f64>> {
    let k = embed_dim.min(quotient_eigenvectors.len());
    let mut embedding = Vec::with_capacity(n_verts);
    for v in 0..n_verts {
        let c = coset_labels[v];
        let mut coords = Vec::with_capacity(k);
        for i in 0..k {
            coords.push(quotient_eigenvectors[i][c]);
        }
        embedding.push(coords);
    }
    embedding
}

// ── 2. Distance-regularity check ──────────────────────────────────────────

/// BFS shortest distances from a single source on the simple (unweighted)
/// adjacency derived from the quotient matrix.
fn bfs_distances(adj: &[Vec<bool>], start: usize) -> Vec<usize> {
    let n = adj.len();
    let mut dist = vec![usize::MAX; n];
    dist[start] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    while let Some(u) = queue.pop_front() {
        for (v, &connected) in adj[u].iter().enumerate() {
            if connected && dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist
}

/// Check whether the simple (unweighted) quotient graph is distance-regular.
///
/// For n <= 100, all vertices are sampled. For n > 100, a deterministic
/// sample of 50 vertices is used (which can detect violations but cannot
/// prove distance-regularity).
pub fn distance_regularity_check(quotient: &[Vec<u32>]) -> DistanceRegularityReport {
    let n = quotient.len();

    // Build simple adjacency (bool).
    let adj: Vec<Vec<bool>> = quotient
        .iter()
        .map(|row| row.iter().map(|&v| v > 0).collect())
        .collect();

    // Simple degree (should be uniform for the quotient of a regular graph).
    let simple_degree = adj[0].iter().filter(|&&b| b).count();

    // Select sample vertices.
    let sample: Vec<usize> = if n <= 100 {
        (0..n).collect()
    } else {
        // Deterministic sample: evenly spaced + LCG jitter.
        let mut s: Vec<usize> = Vec::with_capacity(50);
        let step = n / 50;
        let mut state = 0xCAFE_BABE_u64;
        for i in 0..50 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let jitter = (state >> 33) as usize % step.max(1);
            let idx = (i * step + jitter).min(n - 1);
            s.push(idx);
        }
        s.sort_unstable();
        s.dedup();
        s
    };

    // Precompute BFS distances from all sample vertices.
    let dist_maps: Vec<Vec<usize>> = sample.iter().map(|&s| bfs_distances(&adj, s)).collect();

    // Find diameter across sampled vertices.
    let mut diameter: usize = 0;
    for dm in &dist_maps {
        for &d in dm {
            if d != usize::MAX && d > diameter {
                diameter = d;
            }
        }
    }

    // For each distance d, collect intersection triples (c_d, a_d, b_d).
    // For a pair (u, v) at distance d, count neighbors of v at distances
    // d-1, d, d+1 from u.
    let mut triples_per_d: Vec<BTreeMap<(u32, u32, u32), usize>> =
        vec![BTreeMap::new(); diameter + 1];
    let mut violations = 0_usize;

    for (si, &u) in sample.iter().enumerate() {
        let du = &dist_maps[si];
        for v in 0..n {
            if u == v {
                continue;
            }
            let d = du[v];
            if d == usize::MAX {
                continue;
            }

            let mut c_d: u32 = 0; // neighbors of v at distance d-1 from u
            let mut a_d: u32 = 0; // neighbors of v at distance d from u
            let mut b_d: u32 = 0; // neighbors of v at distance d+1 from u

            for (w, &connected) in adj[v].iter().enumerate() {
                if !connected {
                    continue;
                }
                let dw = du[w];
                if dw == usize::MAX {
                    continue;
                }
                if d > 0 && dw == d - 1 {
                    c_d += 1;
                } else if dw == d {
                    a_d += 1;
                } else if dw == d + 1 {
                    b_d += 1;
                }
            }

            let triple = (c_d, a_d, b_d);
            let entry = triples_per_d[d].entry(triple).or_insert(0);
            *entry += 1;
        }
    }

    // Check: for each distance, there should be exactly one triple.
    let is_dr = triples_per_d.iter().all(|map| map.len() <= 1);
    for map in &triples_per_d {
        if map.len() > 1 {
            violations += map.values().skip(1).sum::<usize>();
        }
    }

    let intersection_array = if is_dr {
        let mut ia: Vec<(u32, u32, u32)> = Vec::with_capacity(diameter + 1);
        for (d, map) in triples_per_d.iter().enumerate() {
            if d == 0 {
                // d=0 is never populated (we skip u==v). The intersection
                // numbers are always c_0=0, a_0=0, b_0=degree.
                ia.push((0, 0, simple_degree as u32));
            } else {
                ia.push(map.keys().next().copied().unwrap_or((0, 0, 0)));
            }
        }
        Some(ia)
    } else {
        None
    };

    DistanceRegularityReport {
        is_distance_regular: is_dr,
        diameter,
        intersection_array,
        n_vertices_sampled: sample.len(),
        violations,
        simple_degree,
    }
}

// ── 3. Orbit structure ────────────────────────────────────────────────────

/// Detect orbit structure by grouping vertices with identical sorted row
/// profiles. If all vertices share the same sorted row profile, the graph
/// is a candidate for vertex-transitivity.
pub fn orbit_structure(quotient: &[Vec<u32>], n_cosets: usize) -> OrbitReport {
    assert_eq!(quotient.len(), n_cosets);

    // For each vertex, compute its sorted row profile.
    let mut profile_counts: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
    for row in quotient {
        let mut sorted_row = row.clone();
        sorted_row.sort_unstable();
        *profile_counts.entry(sorted_row).or_insert(0) += 1;
    }

    let n_profiles = profile_counts.len();
    let mut profile_sizes: Vec<usize> = profile_counts.values().copied().collect();
    profile_sizes.sort_unstable_by(|a, b| b.cmp(a)); // descending

    OrbitReport {
        n_profiles,
        profile_sizes,
        is_vertex_transitive: n_profiles == 1,
    }
}

// ── 4. Spectral comparison ────────────────────────────────────────────────

/// Compare quotient eigenvalues against the full graph spectrum.
///
/// Every eigenvalue of the quotient graph should also appear in the full
/// graph spectrum (possibly with higher multiplicity). This function checks
/// for matches within the given tolerance and reports mismatches.
///
/// `full_eigenvalues` and `quotient_eigenvalues` should both be sorted
/// ascending. `_degree` is reserved for future normalization.
pub fn spectral_comparison(
    full_eigenvalues: &[f64],
    quotient_eigenvalues: &[f64],
    _degree: usize,
) -> SpectralComparisonReport {
    let tol = 0.01;
    let mut matched = 0_usize;
    let mut unmatched_values: Vec<f64> = Vec::new();

    for &qev in quotient_eigenvalues {
        // Binary search for the closest full eigenvalue.
        let pos = full_eigenvalues.partition_point(|&x| x < qev - tol);

        let mut found = false;
        // Check a neighborhood around the insertion point.
        let lo = pos.saturating_sub(2);
        let hi = (pos + 3).min(full_eigenvalues.len());
        for i in lo..hi {
            if (full_eigenvalues[i] - qev).abs() < tol {
                found = true;
                break;
            }
        }

        if found {
            matched += 1;
        } else {
            unmatched_values.push(qev);
        }
    }

    SpectralComparisonReport {
        matched,
        unmatched: unmatched_values.len(),
        unmatched_values,
        tolerance: tol,
    }
}

// ── 5. CLI integration ────────────────────────────────────────────────────

/// Run all quotient graph analyses and print results to stdout.
pub fn run_quotient_analysis(quotient: &[Vec<u32>], label: &str) {
    let n = quotient.len();

    println!("=== Quotient Graph Analysis: {} ===", label);
    println!("Quotient size: {}x{}", n, n);
    println!();

    // 1. Eigendecomposition
    let t = std::time::Instant::now();
    let eigen = quotient_eigen(quotient);
    let eigen_ms = t.elapsed().as_millis();

    println!("--- Eigendecomposition ({} ms) ---", eigen_ms);
    println!("  eigenvalues found:  {}", eigen.n_eigenvalues);
    println!("  distinct (tol=0.001): {}", eigen.distinct.len());
    for bucket in &eigen.distinct {
        println!(
            "    lambda = {:10.4}  x{}",
            bucket.eigenvalue, bucket.multiplicity
        );
    }
    println!("  spectral gap:       {:.4}", eigen.spectral_gap);
    println!("  largest eigenvalue: {:.4}", eigen.largest_eigenvalue);
    println!("  smallest eigenvalue:{:.4}", eigen.smallest_eigenvalue);
    println!("  bipartite:          {}", eigen.is_bipartite);
    println!("  row sum (degree):   {}", eigen.row_sum);
    println!();

    // 2. Distance regularity
    let t = std::time::Instant::now();
    let dr = distance_regularity_check(quotient);
    let dr_ms = t.elapsed().as_millis();

    println!("--- Distance Regularity ({} ms) ---", dr_ms);
    println!("  simple degree:      {}", dr.simple_degree);
    println!("  diameter:           {}", dr.diameter);
    println!("  distance-regular:   {}", dr.is_distance_regular);
    println!("  vertices sampled:   {}", dr.n_vertices_sampled);
    println!("  violations:         {}", dr.violations);
    if let Some(ref ia) = dr.intersection_array {
        println!("  intersection array:");
        for (d, &(c, a, b)) in ia.iter().enumerate() {
            println!("    d={}: c={}, a={}, b={}", d, c, a, b);
        }
        // Traditional notation: {b_0, ..., b_{d-1}; c_1, ..., c_d}
        let bs: Vec<String> = ia
            .iter()
            .take(ia.len() - 1)
            .map(|&(_, _, b)| b.to_string())
            .collect();
        let cs: Vec<String> = ia.iter().skip(1).map(|&(c, _, _)| c.to_string()).collect();
        println!("  traditional: {{{};{}}}", bs.join(","), cs.join(","));
    }
    println!();

    // 3. Orbit structure
    let t = std::time::Instant::now();
    let orbits = orbit_structure(quotient, n);
    let orbit_ms = t.elapsed().as_millis();

    println!("--- Orbit Structure ({} ms) ---", orbit_ms);
    println!("  distinct profiles:  {}", orbits.n_profiles);
    println!("  vertex-transitive:  {}", orbits.is_vertex_transitive);
    if orbits.n_profiles <= 20 {
        println!("  profile sizes:      {:?}", orbits.profile_sizes);
    } else {
        println!(
            "  profile sizes (top 10): {:?}...",
            &orbits.profile_sizes[..10]
        );
    }
    println!();

    println!("=== End Quotient Graph Analysis: {} ===", label);
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The S4/V4 quotient adjacency matrix (6x6).
    /// Row sums are all 12. Zero diagonal. Symmetric.
    fn s4_quotient_matrix() -> Vec<Vec<u32>> {
        vec![
            vec![0, 8, 4, 0, 0, 0],
            vec![8, 0, 0, 0, 4, 0],
            vec![4, 0, 0, 8, 0, 0],
            vec![0, 0, 8, 0, 0, 4],
            vec![0, 4, 0, 0, 0, 8],
            vec![0, 0, 0, 4, 8, 0],
        ]
    }

    // ── Eigendecomposition tests ──────────────────────────────────────────

    #[test]
    fn s4_quotient_eigenvalues() {
        let q = s4_quotient_matrix();
        let report = quotient_eigen(&q);

        assert_eq!(report.n_eigenvalues, 6);

        // Known eigenvalues: -12, -4*sqrt(3) (x2), 4*sqrt(3) (x2), 12
        let sqrt3_4 = 4.0 * 3.0_f64.sqrt(); // 6.9282...

        let expected = vec![-12.0, -sqrt3_4, -sqrt3_4, sqrt3_4, sqrt3_4, 12.0];
        for (got, want) in report.raw_eigenvalues.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 0.01,
                "eigenvalue mismatch: got {:.4}, expected {:.4}",
                got,
                want
            );
        }

        // 4 distinct eigenvalues: -12 (x1), -4sqrt3 (x2), 4sqrt3 (x2), 12 (x1).
        assert_eq!(
            report.distinct.len(),
            4,
            "expected 4 distinct eigenvalues, got {}",
            report.distinct.len()
        );
    }

    #[test]
    fn s4_quotient_eigen_properties() {
        let q = s4_quotient_matrix();
        let report = quotient_eigen(&q);

        // Row sum = 12 = largest eigenvalue (regular graph).
        assert_eq!(report.row_sum, 12);
        assert!((report.largest_eigenvalue - 12.0).abs() < 0.01);

        // Bipartite: smallest eigenvalue = -12 = -degree.
        assert!(report.is_bipartite, "S4/V4 quotient should be bipartite");
        assert!((report.smallest_eigenvalue + 12.0).abs() < 0.01);

        // Spectral gap: 12 - 4*sqrt(3) ~ 5.072
        let expected_gap = 12.0 - 4.0 * 3.0_f64.sqrt();
        assert!(
            (report.spectral_gap - expected_gap).abs() < 0.01,
            "spectral gap: got {:.4}, expected {:.4}",
            report.spectral_gap,
            expected_gap
        );

        // 4 distinct eigenvalues with multiplicities [1, 2, 2, 1].
        assert_eq!(report.distinct.len(), 4);
        let mults: Vec<usize> = report.distinct.iter().map(|b| b.multiplicity).collect();
        assert_eq!(mults, vec![1, 2, 2, 1]);
    }

    // ── Jacobi correctness test ───────────────────────────────────────────

    #[test]
    fn jacobi_2x2() {
        // [[3, 1], [1, 3]] has eigenvalues 2 and 4.
        let m = vec![vec![3.0, 1.0], vec![1.0, 3.0]];
        let (evals, evecs) = jacobi_eigen(&m);
        assert_eq!(evals.len(), 2);
        assert!((evals[0] - 2.0).abs() < 1e-10);
        assert!((evals[1] - 4.0).abs() < 1e-10);

        // Verify eigenvectors: M * v = lambda * v.
        for (i, &lam) in evals.iter().enumerate() {
            let v = &evecs[i];
            let mv: Vec<f64> = m
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                .collect();
            for (j, (&got, want)) in mv.iter().zip(v.iter().map(|x| x * lam)).enumerate() {
                assert!(
                    (got - want).abs() < 1e-10,
                    "eigenvector {} component {}: M*v={}, lam*v={}",
                    i,
                    j,
                    got,
                    want
                );
            }
        }
    }

    #[test]
    fn jacobi_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal entries.
        let m = vec![
            vec![5.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let (evals, _) = jacobi_eigen(&m);
        assert!((evals[0] - 1.0).abs() < 1e-10);
        assert!((evals[1] - 3.0).abs() < 1e-10);
        assert!((evals[2] - 5.0).abs() < 1e-10);
    }

    // ── Distance-regularity tests ─────────────────────────────────────────

    #[test]
    fn s4_quotient_is_distance_regular() {
        let q = s4_quotient_matrix();
        let report = distance_regularity_check(&q);

        assert!(
            report.is_distance_regular,
            "S4/V4 quotient simple graph (C_6) should be distance-regular"
        );
        assert_eq!(report.diameter, 3);
        assert_eq!(report.simple_degree, 2);
        assert_eq!(report.violations, 0);
        assert_eq!(report.n_vertices_sampled, 6);

        // Intersection array for C_6: {2,1,1; 1,1,2}
        // As triples: d=0 -> (0,0,2), d=1 -> (1,0,1), d=2 -> (1,0,1), d=3 -> (2,0,0)
        let ia = report.intersection_array.as_ref().unwrap();
        assert_eq!(ia.len(), 4);
        assert_eq!(ia[0], (0, 0, 2)); // d=0: no d-1, 0 at same, 2 at d+1
        assert_eq!(ia[1], (1, 0, 1)); // d=1
        assert_eq!(ia[2], (1, 0, 1)); // d=2
        assert_eq!(ia[3], (2, 0, 0)); // d=3: antipodal, both neighbors closer
    }

    // ── Orbit structure tests ─────────────────────────────────────────────

    #[test]
    fn s4_quotient_is_vertex_transitive() {
        let q = s4_quotient_matrix();
        let report = orbit_structure(&q, 6);

        // V4 is normal in S4, so S4/V4 = S3 acts transitively on the 6 cosets.
        // All row profiles should be identical after sorting.
        assert!(
            report.is_vertex_transitive,
            "S4/V4 quotient should be vertex-transitive (single row profile)"
        );
        assert_eq!(report.n_profiles, 1);
        assert_eq!(report.profile_sizes, vec![6]);
    }

    // ── Spectral comparison tests ─────────────────────────────────────────

    #[test]
    fn spectral_comparison_exact_subset() {
        // Full spectrum contains all quotient eigenvalues plus extras.
        let full = vec![-12.0, -8.0, -6.928, -6.928, 0.0, 6.928, 6.928, 8.0, 12.0];
        let quotient = vec![-12.0, -6.928, -6.928, 6.928, 6.928, 12.0];
        let report = spectral_comparison(&full, &quotient, 12);
        assert_eq!(report.matched, 6);
        assert_eq!(report.unmatched, 0);
    }

    #[test]
    fn spectral_comparison_with_mismatch() {
        let full = vec![-12.0, -6.928, 6.928, 12.0];
        let quotient = vec![-12.0, -6.928, 0.0, 6.928, 12.0];
        let report = spectral_comparison(&full, &quotient, 12);
        assert_eq!(report.matched, 4);
        assert_eq!(report.unmatched, 1);
        assert!((report.unmatched_values[0] - 0.0).abs() < 0.01);
    }

    // ── Dense matvec test ─────────────────────────────────────────────────

    #[test]
    fn dense_matvec_identity() {
        let id = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        let v = vec![3.0, 5.0, 7.0];
        let result = dense_matvec(&id, &v);
        assert!((result[0] - 3.0).abs() < 1e-14);
        assert!((result[1] - 5.0).abs() < 1e-14);
        assert!((result[2] - 7.0).abs() < 1e-14);
    }

    #[test]
    fn dense_matvec_s4_quotient() {
        let q = s4_quotient_matrix();
        // All-ones vector: Q * [1,1,1,1,1,1] = [12,12,12,12,12,12] for regular graph.
        let ones = vec![1.0; 6];
        let result = dense_matvec(&q, &ones);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 12.0).abs() < 1e-10,
                "row {} sum should be 12, got {}",
                i,
                val
            );
        }
    }

    // ── Eigenvalue grouping test ──────────────────────────────────────────

    #[test]
    fn grouping_merges_close_values() {
        let vals = vec![1.0, 1.0005, 1.001, 3.0, 3.0002, 5.0];
        let buckets = group_eigenvalues(&vals, 0.002);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].multiplicity, 3);
        assert_eq!(buckets[1].multiplicity, 2);
        assert_eq!(buckets[2].multiplicity, 1);
    }

    // ── Full integration test on S4 quotient ──────────────────────────────

    #[test]
    fn s4_full_analysis_smoke() {
        // Verify that run_quotient_analysis completes without panics.
        let q = s4_quotient_matrix();
        run_quotient_analysis(&q, "S4/V4 test");
    }

    // ── Full eigenvector extraction tests ─────────────────────────────────

    /// Build the S4/V4 quotient matrix from the actual permutahedron graph
    /// and V4 coset partition. Returns (quotient_matrix, coset_labels, n_verts).
    fn build_s4_quotient_from_graph() -> (Vec<Vec<u32>>, Vec<usize>, usize) {
        let graph = crate::permutahedron::complete_graph(4).unwrap();
        let v4 = crate::permutahedron::vierergruppe();
        let partition =
            crate::permutahedron::coset_partition(&v4, crate::permutahedron::CosetSide::Right)
                .unwrap();
        let n_cosets = partition.slices.len();
        let n_verts = graph.vertex_count;
        let mut labels = vec![0_usize; n_verts];
        for (i, sl) in partition.slices.iter().enumerate() {
            for &v in sl {
                labels[v as usize] = i;
            }
        }
        let mut q = vec![vec![0u32; n_cosets]; n_cosets];
        for edge in &graph.edges {
            let ci = labels[edge[0] as usize];
            let cj = labels[edge[1] as usize];
            q[ci][cj] += 1;
            q[cj][ci] += 1;
        }
        (q, labels, n_verts)
    }

    #[test]
    fn test_quotient_eigen_full_s4() {
        let q = s4_quotient_matrix();
        let report = quotient_eigen_full(&q, 100);

        // Should return all 6 eigenpairs for a 6x6 matrix.
        assert_eq!(report.n_computed, 6);
        assert_eq!(report.eigenvalues.len(), 6);
        assert_eq!(report.eigenvectors.len(), 6);

        // Each eigenvector should have dimension 6.
        for (i, ev) in report.eigenvectors.iter().enumerate() {
            assert_eq!(
                ev.len(),
                6,
                "eigenvector {} should have 6 components, got {}",
                i,
                ev.len()
            );
        }

        // Known eigenvalues: -12, -4*sqrt(3), -4*sqrt(3), 4*sqrt(3), 4*sqrt(3), 12
        let sqrt3_4 = 4.0 * 3.0_f64.sqrt();
        let expected = vec![-12.0, -sqrt3_4, -sqrt3_4, sqrt3_4, sqrt3_4, 12.0];
        for (got, want) in report.eigenvalues.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 0.01,
                "eigenvalue mismatch: got {:.4}, expected {:.4}",
                got,
                want
            );
        }

        // Eigenvectors should be orthonormal: V^T * V = I within tolerance.
        let tol = 1e-8;
        for i in 0..6 {
            for j in 0..6 {
                let d: f64 = report.eigenvectors[i]
                    .iter()
                    .zip(report.eigenvectors[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let expected_val = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (d - expected_val).abs() < tol,
                    "V^T*V[{},{}] = {:.10}, expected {}",
                    i,
                    j,
                    d,
                    expected_val
                );
            }
        }

        // Verify Av = lambda*v for each eigenpair.
        let fmat: Vec<Vec<f64>> = q
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();
        for (idx, &lam) in report.eigenvalues.iter().enumerate() {
            let v = &report.eigenvectors[idx];
            // Compute A * v
            let av: Vec<f64> = fmat
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                .collect();
            for (j, (&got, &vj)) in av.iter().zip(v.iter()).enumerate() {
                assert!(
                    (got - lam * vj).abs() < 1e-8,
                    "A*v[{}] component {}: got {:.10}, expected {:.10}",
                    idx,
                    j,
                    got,
                    lam * vj
                );
            }
        }
    }

    #[test]
    fn test_quotient_eigen_full_max_vectors() {
        let q = s4_quotient_matrix();
        // Request only 3 eigenvectors.
        let report = quotient_eigen_full(&q, 3);
        assert_eq!(report.n_computed, 3);
        assert_eq!(report.eigenvalues.len(), 3);
        assert_eq!(report.eigenvectors.len(), 3);
    }

    #[test]
    fn test_quotient_lift_s4() {
        let (q, labels, n_verts) = build_s4_quotient_from_graph();
        assert_eq!(n_verts, 24);

        let report = quotient_eigen_full(&q, 100);
        let embed_dim = 4;
        let embedding = quotient_lift_embedding(&report.eigenvectors, &labels, n_verts, embed_dim);

        // Output should have 24 rows and embed_dim columns.
        assert_eq!(embedding.len(), 24);
        for (v, row) in embedding.iter().enumerate() {
            assert_eq!(
                row.len(),
                embed_dim,
                "vertex {} embedding should have {} dims, got {}",
                v,
                embed_dim,
                row.len()
            );
        }

        // All vertices in the same coset must have identical embedding coordinates.
        let n_cosets = q.len();
        for ci in 0..n_cosets {
            let members: Vec<usize> = (0..n_verts).filter(|&v| labels[v] == ci).collect();
            assert!(
                members.len() >= 2,
                "coset {} should have multiple members",
                ci
            );
            let first_embed = &embedding[members[0]];
            for &v in &members[1..] {
                assert_eq!(
                    &embedding[v], first_embed,
                    "vertex {} and {} are in coset {} but have different embeddings",
                    members[0], v, ci
                );
            }
        }

        // Vertices in different cosets must have different coordinates.
        // Pick representatives from each coset and verify they differ.
        let mut representatives: Vec<(usize, &Vec<f64>)> = Vec::new();
        for ci in 0..n_cosets {
            let rep = (0..n_verts).find(|&v| labels[v] == ci).unwrap();
            representatives.push((ci, &embedding[rep]));
        }
        for i in 0..representatives.len() {
            for j in (i + 1)..representatives.len() {
                let (ci, ei) = &representatives[i];
                let (cj, ej) = &representatives[j];
                assert_ne!(
                    ei, ej,
                    "cosets {} and {} should have different embeddings",
                    ci, cj
                );
            }
        }
    }

    #[test]
    fn test_quotient_lift_direct_s8() {
        use crate::permutahedron::{self, CosetSide};

        let graph = permutahedron::complete_graph(8)
            .expect("S8 graph construction must succeed");
        let r8 = permutahedron::rana_r8();
        let partition = permutahedron::coset_partition(&r8, CosetSide::Right)
            .expect("R8 coset partition must succeed");
        let n_cosets = partition.slices.len();
        let mut coset_labels = vec![0usize; graph.vertex_count];
        for (i, sl) in partition.slices.iter().enumerate() {
            for &v in sl {
                coset_labels[v as usize] = i;
            }
        }

        let mut q = vec![vec![0u32; n_cosets]; n_cosets];
        for &[u, v] in &graph.edges {
            let cu = coset_labels[u as usize];
            let cv = coset_labels[v as usize];
            q[cu][cv] += 1;
            q[cv][cu] += 1;
        }

        let q_eigen = quotient_eigen_full(&q, 20);
        println!("Quotient eigenvalues (first 10):");
        for (i, &ev) in q_eigen.eigenvalues.iter().enumerate().take(10) {
            println!("  {:2}. lambda = {:.4}", i, ev);
        }

        let embedding = quotient_lift_embedding(
            &q_eigen.eigenvectors, &coset_labels, graph.vertex_count, 10,
        );

        // Direct assignment: group vertices by embedding coordinates.
        // Since the embedding is coset-constant, each unique point = one coset.
        use std::collections::HashMap;
        let mut point_to_cluster: HashMap<Vec<i64>, usize> = HashMap::new();
        let mut direct_predicted = vec![0usize; graph.vertex_count];
        let mut next_id = 0usize;
        for v in 0..graph.vertex_count {
            let key: Vec<i64> = embedding[v]
                .iter()
                .map(|&x| (x * 1e8).round() as i64)
                .collect();
            let cluster = *point_to_cluster.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            direct_predicted[v] = cluster;
        }
        let direct_ari = crate::spectral_lanczos::adjusted_rand_index(
            &coset_labels, &direct_predicted,
        );
        println!("Direct assignment: {} unique points, ARI = {:.6}", next_id, direct_ari);
        assert!(
            direct_ari > 0.999,
            "direct assignment ARI should be ~1.0, got {:.6}",
            direct_ari
        );

        // Also test k-means with higher n_init
        for &n_init in &[3, 10, 20] {
            let predicted = crate::spectral_lanczos::kmeans_clustering(
                &embedding, graph.vertex_count, n_cosets, n_init, 100, 42,
            );
            let ari = crate::spectral_lanczos::adjusted_rand_index(
                &coset_labels, &predicted,
            );
            println!("k-means n_init={}: ARI = {:.6}", n_init, ari);
        }
    }
}
