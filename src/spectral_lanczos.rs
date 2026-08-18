//! Lanczos tridiagonalization and dense eigensolver for spectral analysis of
//! permutahedron Cayley graphs.
//!
//! No external linear algebra dependencies. Matrices are at most 40,320 x 40,320
//! sparse (S8 graph) but Lanczos vectors are at most m=300 long, and the
//! tridiagonal eigenproblem is at most 300 x 300.

use serde::Serialize;

// ── Result structs ──────────────────────────────────────────────────────────

/// Output of the Lanczos tridiagonalization.
#[derive(Debug, Clone, Serialize)]
pub struct LanczosResult {
    /// Diagonal entries of the tridiagonal matrix T.
    pub alpha: Vec<f64>,
    /// Super/sub-diagonal entries of T. Length = alpha.len() - 1.
    pub beta: Vec<f64>,
    /// Lanczos vectors q_0 .. q_{m-1}, each of length n.
    pub q_vectors: Vec<Vec<f64>>,
}

/// One eigenspace bucket in the coset spectral report.
#[derive(Debug, Clone, Serialize)]
pub struct EigenspaceBucket {
    /// Representative eigenvalue (centroid of the cluster).
    pub eigenvalue: f64,
    /// Total coset energy summed across all cosets in the sample.
    pub total_energy: f64,
    /// Number of individual Ritz values that fell into this bucket.
    pub multiplicity: usize,
}

/// Aggregated spectral report over a sample of cosets.
#[derive(Debug, Clone, Serialize)]
pub struct CosetSpectralReport {
    /// Eigenspace buckets sorted descending by total coset energy.
    pub eigenspaces: Vec<EigenspaceBucket>,
    /// Number of cosets sampled.
    pub n_cosets_sampled: usize,
    /// Lanczos dimension used.
    pub lanczos_m: usize,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

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

// ── 1. Sparse Laplacian matvec ──────────────────────────────────────────────

/// Compute L*v = degree*v - A*v where A is the adjacency matrix of an
/// undirected graph with edges `(src, tgt, generator_index)`.
pub fn sparse_laplacian_matvec(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    v: &[f64],
) -> Vec<f64> {
    assert_eq!(v.len(), n);
    let deg = degree as f64;
    let mut out = vec![0.0_f64; n];
    // D*v
    for i in 0..n {
        out[i] = deg * v[i];
    }
    // subtract A*v (each undirected edge contributes in both directions)
    for &(src, tgt, _gen) in edges {
        out[src] -= v[tgt];
        out[tgt] -= v[src];
    }
    out
}

// ── 2. Lanczos with full reorthogonalization ────────────────────────────────

/// Lanczos tridiagonalization with full reorthogonalization (modified
/// Gram-Schmidt against all previous Lanczos vectors).
///
/// * `edges` - undirected graph edges (src, tgt, generator)
/// * `degree` - vertex degree (uniform for Cayley graphs)
/// * `n` - number of vertices
/// * `start` - starting vector (length n, need not be normalized)
/// * `m` - number of Lanczos steps (at most n)
pub fn lanczos(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    start: &[f64],
    m: usize,
) -> LanczosResult {
    assert_eq!(start.len(), n);
    let m = m.min(n);

    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m.saturating_sub(1));
    let mut q_vectors: Vec<Vec<f64>> = Vec::with_capacity(m);

    // q_0 = start / ||start||
    let mut q = start.to_vec();
    let nrm = norm(&q);
    assert!(nrm > 1e-14, "start vector must be nonzero");
    scale(1.0 / nrm, &mut q);
    q_vectors.push(q.clone());

    for j in 0..m {
        // w = L * q_j
        let mut w = sparse_laplacian_matvec(edges, degree, n, &q_vectors[j]);

        // alpha_j = q_j . w
        let a = dot(&q_vectors[j], &w);
        alpha.push(a);

        // w = w - alpha_j * q_j
        axpy(-a, &q_vectors[j], &mut w);

        // w = w - beta_{j-1} * q_{j-1}  (if j > 0)
        if j > 0 {
            axpy(-beta[j - 1], &q_vectors[j - 1], &mut w);
        }

        // Full reorthogonalization: MGS against all previous vectors
        for k in 0..=j {
            let c = dot(&q_vectors[k], &w);
            axpy(-c, &q_vectors[k], &mut w);
        }

        let b = norm(&w);

        if j + 1 < m {
            beta.push(b);
            if b < 1e-14 {
                // Invariant subspace found; fill remaining with zeros
                // and break (the tridiagonal will be smaller)
                // Pad with a random restart direction
                // For correctness, we just stop early.
                alpha.truncate(j + 1);
                beta.truncate(j);
                break;
            }
            scale(1.0 / b, &mut w);
            q_vectors.push(w);
        }
    }

    LanczosResult {
        alpha,
        beta,
        q_vectors,
    }
}

// ── 3. Tridiagonal eigensolver (QL algorithm with implicit shift) ───────────

/// Eigendecomposition of a symmetric tridiagonal matrix using the QL
/// algorithm with implicit shifts (the approach used by LAPACK dsteqr).
///
/// T\[i\]\[i\] = alpha\[i\], T\[i\]\[i+1\] = T\[i+1\]\[i\] = beta\[i\].
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted ascending
/// and eigenvectors\[i\] is the column eigenvector for eigenvalues\[i\].
pub fn tridiag_eigen(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = alpha.len();
    assert_eq!(beta.len(), n.saturating_sub(1));
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![alpha[0]], vec![vec![1.0]]);
    }

    let mut d = alpha.to_vec();
    // Pad off-diagonal to length n (entry n-1 unused, set to 0)
    let mut e = vec![0.0; n];
    for (i, &b) in beta.iter().enumerate() {
        e[i] = b;
    }

    // Eigenvector accumulator: z[i][j] is the j-th component of eigenvector i.
    // Starts as the identity matrix.
    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0; n];
            v[i] = 1.0;
            v
        })
        .collect();

    // QL iteration: for each l from 0..n, deflate e[l] to zero.
    let max_iter_per = 100 * n;

    for l in 0..n {
        let mut iter_count = 0_usize;

        loop {
            // Find the smallest m >= l such that e[m] is negligible.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                // Convergence test: e[m] is negligible relative to neighbors
                if e[m].abs() <= 1e-15 * dd.max(1e-300) {
                    break;
                }
                m += 1;
            }

            if m == l {
                // e[l] has converged to zero; eigenvalue d[l] is isolated.
                break;
            }

            assert!(
                iter_count < max_iter_per,
                "tridiag_eigen: QL iteration did not converge for index {l}"
            );
            iter_count += 1;

            // Compute the shift from the trailing 2x2 block of the
            // unreduced submatrix d[l..=m], e[l..m-1].
            //   g = (d[l+1] - d[l]) / (2 * e[l])
            //   shift = d[m] - d[l] + e[l] / (g + sign(hypot(g,1), g))
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = (g * g + 1.0).sqrt();
            g = d[m] - d[l] + e[l] / (g + copysign(r, g));

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            // Chase the bulge from m-1 down to l.
            let mut converged_early = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let rr = f.hypot(g);
                e[i + 1] = rr;

                if rr.abs() < 1e-30 {
                    // Lucky exact cancellation: undo the damage and restart.
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

                // Accumulate eigenvector rotation.
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

    // Sort eigenvalues ascending and permute eigenvectors accordingly.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());

    let evals: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let evecs: Vec<Vec<f64>> = idx.iter().map(|&i| z[i].clone()).collect();

    (evals, evecs)
}

/// Copy of libm copysign: return `a` with the sign of `b`.
fn copysign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

// ── 4. Spectral density ─────────────────────────────────────────────────────

/// Compute spectral density of a target vector f relative to a Lanczos
/// decomposition.
///
/// Returns `(eigenvalue, energy)` pairs sorted by eigenvalue, where the energy
/// at Ritz value lambda_i is `||f||^2 * z_0[i]^2` (z_0[i] is the first
/// component of the i-th tridiagonal eigenvector).
///
/// This exploits the fact that if the Lanczos start vector was f/||f||, then
/// q_0 = f/||f|| and <q_j, f> = 0 for j > 0, so the overlap simplifies.
pub fn spectral_density(lanczos_result: &LanczosResult, target: &[f64]) -> Vec<(f64, f64)> {
    let (evals, evecs) = tridiag_eigen(&lanczos_result.alpha, &lanczos_result.beta);
    let f_norm_sq = dot(target, target);

    let mut pairs: Vec<(f64, f64)> = evals
        .iter()
        .zip(evecs.iter())
        .map(|(&lam, z)| {
            let z0 = z[0]; // first component of this eigenvector
            let energy = f_norm_sq * z0 * z0;
            (lam, energy)
        })
        .collect();

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs
}

// ── 5. Coset spectral scan ──────────────────────────────────────────────────

/// Scan the spectral density of coset indicator functions.
///
/// For each of the first `n_sample` cosets (labeled 0..n_sample in
/// `coset_labels`), build the indicator vector, run Lanczos starting from it,
/// and accumulate spectral density. Then group nearby Ritz values (tolerance
/// 0.01) into eigenspace buckets and sort by total energy descending.
pub fn coset_spectral_scan(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    coset_labels: &[usize],
    n_cosets: usize,
    lanczos_m: usize,
    n_sample: usize,
) -> CosetSpectralReport {
    assert_eq!(coset_labels.len(), n);
    let sample_count = n_sample.min(n_cosets);

    // Collect all (eigenvalue, energy) pairs across sampled cosets
    let mut all_pairs: Vec<(f64, f64)> = Vec::new();

    for coset_id in 0..sample_count {
        // Build indicator vector for this coset
        let indicator: Vec<f64> = coset_labels
            .iter()
            .map(|&label| if label == coset_id { 1.0 } else { 0.0 })
            .collect();

        // Skip empty cosets
        let ind_norm = norm(&indicator);
        if ind_norm < 1e-14 {
            continue;
        }

        let lr = lanczos(edges, degree, n, &indicator, lanczos_m);
        let density = spectral_density(&lr, &indicator);
        all_pairs.extend(density);
    }

    // Sort by eigenvalue
    all_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Group into eigenspace buckets with tolerance 0.01
    let tolerance = 0.01;
    let mut buckets: Vec<EigenspaceBucket> = Vec::new();

    for (lam, energy) in &all_pairs {
        if let Some(last) = buckets.last_mut() {
            if (*lam - last.eigenvalue / last.multiplicity as f64).abs() < tolerance {
                // Merge into existing bucket (update centroid incrementally)
                let old_total = last.eigenvalue;
                last.eigenvalue = old_total + *lam;
                last.total_energy += *energy;
                last.multiplicity += 1;
                continue;
            }
        }
        buckets.push(EigenspaceBucket {
            eigenvalue: *lam,
            total_energy: *energy,
            multiplicity: 1,
        });
    }

    // Finalize centroids
    for b in &mut buckets {
        b.eigenvalue /= b.multiplicity as f64;
    }

    // Sort by total energy descending
    buckets.sort_by(|a, b| b.total_energy.partial_cmp(&a.total_energy).unwrap());

    CosetSpectralReport {
        eigenspaces: buckets,
        n_cosets_sampled: sample_count,
        lanczos_m,
    }
}

// ── 6. Symmetric indefinite solve via Lanczos ─────────────────────────────

/// Solve A*x = b for a symmetric (possibly indefinite) operator A, using
/// Lanczos tridiagonalization followed by a spectral solve of the projected
/// system.
///
/// Runs Lanczos on A starting from b to build Q and tridiagonal T, then
/// eigendecomposes T = Z D Z^T and computes x = Q Z D^{-1} Z^T (||b|| e_1).
///
/// This is equivalent to MINRES and works for both positive definite and
/// indefinite symmetric operators, as long as A is nonsingular.
fn symmetric_solve(
    apply_a: &dyn Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    n: usize,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let b_norm = norm(b);
    if b_norm < tol {
        return vec![0.0; n];
    }

    let m = max_iter.min(n);
    let mut alpha_vec: Vec<f64> = Vec::with_capacity(m);
    let mut beta_vec: Vec<f64> = Vec::with_capacity(m);
    let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(m);

    // q_0 = b / ||b||
    let mut q = b.to_vec();
    scale(1.0 / b_norm, &mut q);
    q_vecs.push(q);

    for j in 0..m {
        let mut w = apply_a(&q_vecs[j]);
        let a = dot(&q_vecs[j], &w);
        alpha_vec.push(a);

        axpy(-a, &q_vecs[j], &mut w);
        if j > 0 {
            axpy(-beta_vec[j - 1], &q_vecs[j - 1], &mut w);
        }

        // Full reorthogonalization
        for k in 0..=j {
            let c = dot(&q_vecs[k], &w);
            axpy(-c, &q_vecs[k], &mut w);
        }

        let b_val = norm(&w);

        // Check convergence: the residual of the Lanczos-projected solve
        // is bounded by beta_j * |y_j| where y is the tridiagonal solution.
        // For efficiency, check every few steps or when beta is small.
        if b_val < 1e-14 || j + 1 >= m {
            // Lanczos has converged or reached max iterations.
            // Solve the projected system T * y = ||b|| * e_1 via eigendecomposition.
            let (evals, evecs) = tridiag_eigen(&alpha_vec, &beta_vec);
            let mut x = vec![0.0; n];

            for (i, (&lam, z)) in evals.iter().zip(evecs.iter()).enumerate() {
                if lam.abs() < 1e-14 {
                    continue; // skip near-zero eigenvalue (singular component)
                }
                // Coefficient: (z^T * (||b|| e_1)) / lam = ||b|| * z[0] / lam
                let coeff = b_norm * z[0] / lam;
                // Accumulate: x += coeff * Q * z (project Ritz vector to full space)
                for (k, &z_k) in z.iter().enumerate() {
                    if k < q_vecs.len() {
                        axpy(coeff * z_k, &q_vecs[k], &mut x);
                    }
                }
                let _ = i; // suppress unused warning
            }
            return x;
        }

        beta_vec.push(b_val);
        scale(1.0 / b_val, &mut w);
        q_vecs.push(w);

        // Early convergence check via residual estimate.
        // The tridiagonal T_{j+1} uses alpha[0..j] and beta[0..j-1].
        // beta_vec[j] (just pushed) is the coupling to q_{j+1}, NOT part of T.
        if j > 0 && j % 5 == 0 {
            let beta_for_tridiag = &beta_vec[..alpha_vec.len() - 1];
            let (evals_check, evecs_check) = tridiag_eigen(&alpha_vec, beta_for_tridiag);
            // Solve T*y = ||b||*e1 via spectral decomposition
            let mut y = vec![0.0; alpha_vec.len()];
            for (&lam, z) in evals_check.iter().zip(evecs_check.iter()) {
                if lam.abs() < 1e-14 {
                    continue;
                }
                let c = b_norm * z[0] / lam;
                for (k, &z_k) in z.iter().enumerate() {
                    y[k] += c * z_k;
                }
            }
            // Residual bound: |beta_j| * |y[j]|
            let residual_bound = b_val * y.last().unwrap_or(&0.0).abs();
            if residual_bound < tol * b_norm {
                // Converged: compute full solution x = Q * y
                let mut x = vec![0.0; n];
                for (k, &y_k) in y.iter().enumerate() {
                    if k < q_vecs.len() {
                        axpy(y_k, &q_vecs[k], &mut x);
                    }
                }
                return x;
            }
        }
    }

    vec![0.0; n]
}

// ── 7. Shift-invert Lanczos ───────────────────────────────────────────────

/// Lanczos iteration on (L - sigma*I)^{-1} to resolve eigenvalues near `sigma`.
///
/// At each Lanczos step, the matrix-vector product is replaced by a CG solve
/// of (L - sigma*I) x = q_j. The resulting Ritz values are eigenvalues of the
/// inverse operator; original Laplacian eigenvalues are recovered as
/// lambda = sigma + 1/ritz_value.
///
/// * `sigma` - spectral shift (target eigenvalue region)
/// * `start` - starting vector (length n, need not be normalized)
/// * `m` - number of Lanczos steps
/// * `cg_tol` - CG convergence tolerance (1e-10 recommended)
/// * `cg_max_iter` - maximum CG iterations per solve (1000 recommended)
pub fn shift_invert_lanczos(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    sigma: f64,
    start: &[f64],
    m: usize,
    cg_tol: f64,
    cg_max_iter: usize,
) -> LanczosResult {
    assert_eq!(start.len(), n);
    let m = m.min(n);

    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m.saturating_sub(1));
    let mut q_vectors: Vec<Vec<f64>> = Vec::with_capacity(m);

    // Closure for the shifted operator: (L - sigma*I) * v
    let shifted_matvec = |v: &[f64]| -> Vec<f64> {
        let mut result = sparse_laplacian_matvec(edges, degree, n, v);
        for i in 0..n {
            result[i] -= sigma * v[i];
        }
        result
    };

    // q_0 = start / ||start||
    let mut q = start.to_vec();
    let nrm = norm(&q);
    assert!(nrm > 1e-14, "start vector must be nonzero");
    scale(1.0 / nrm, &mut q);
    q_vectors.push(q.clone());

    for j in 0..m {
        // Solve (L - sigma*I) x = q_j via Lanczos-based symmetric solve.
        // This handles the indefinite case (sigma inside the spectrum) where
        // standard CG would fail.
        let mut w = symmetric_solve(&shifted_matvec, &q_vectors[j], n, cg_tol, cg_max_iter);

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

    LanczosResult {
        alpha,
        beta,
        q_vectors,
    }
}

// ── 8. Targeted eigenspace embedding ──────────────────────────────────────

/// Extract eigenvectors near a target eigenvalue `sigma` by shift-invert
/// Lanczos, returning full n-dimensional eigenvectors projected from the
/// Lanczos basis.
///
/// Returns `(eigenvalues, eigenvectors)` for the `n_vectors` Ritz pairs
/// whose Laplacian eigenvalues are closest to `sigma`.
pub fn targeted_eigenspace_embedding(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    sigma: f64,
    n_vectors: usize,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Run shift-invert Lanczos with extra vectors for convergence margin
    let m = (n_vectors + 20).min(n);
    // Use a delta vector at vertex 0 as the start vector. A delta function
    // on a Cayley graph has nonzero projection onto every irrep (each irrep
    // rho contributes dim(rho)^2/|G| weight at the identity), guaranteeing
    // the Krylov subspace spans all distinct eigenspaces.
    let mut start = vec![0.0; n];
    start[0] = 1.0;

    let lr = shift_invert_lanczos(edges, degree, n, sigma, &start, m, cg_tol, cg_max_iter);

    // Eigendecompose the tridiagonal
    let (ritz_values, ritz_vectors) = tridiag_eigen(&lr.alpha, &lr.beta);

    // Convert Ritz values back to Laplacian eigenvalues: lambda = sigma + 1/ritz
    let laplacian_evals: Vec<f64> = ritz_values
        .iter()
        .map(|&rv| {
            if rv.abs() < 1e-14 {
                f64::INFINITY
            } else {
                sigma + 1.0 / rv
            }
        })
        .collect();

    // Select the n_vectors Ritz pairs closest to sigma
    let actual_n = n_vectors.min(laplacian_evals.len());
    let mut indices: Vec<usize> = (0..laplacian_evals.len()).collect();
    indices.sort_by(|&a, &b| {
        let da = (laplacian_evals[a] - sigma).abs();
        let db = (laplacian_evals[b] - sigma).abs();
        da.partial_cmp(&db).unwrap()
    });
    indices.truncate(actual_n);
    indices.sort(); // restore order for determinism

    let selected_evals: Vec<f64> = indices.iter().map(|&i| laplacian_evals[i]).collect();

    // Project Ritz vectors back to full n-dimensional space:
    // v_i = sum_j z_j[i] * q_vectors[j]
    let n_lanczos = lr.q_vectors.len();
    let selected_evecs: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| {
            let z = &ritz_vectors[i]; // coefficients in Lanczos basis
            let mut v = vec![0.0; n];
            for j in 0..n_lanczos.min(z.len()) {
                axpy(z[j], &lr.q_vectors[j], &mut v);
            }
            v
        })
        .collect();

    (selected_evals, selected_evecs)
}

// ── 9. K-means clustering ─────────────────────────────────────────────────

/// Simple k-means clustering with multiple random restarts.
///
/// * `embedding` - data points, each a vector of the same dimension
/// * `n` - number of data points (must equal embedding.len())
/// * `k` - number of clusters
/// * `n_init` - number of random initializations (best inertia wins)
/// * `max_iter` - maximum iterations per initialization
/// * `seed` - deterministic RNG seed (LCG)
pub fn kmeans_clustering(
    embedding: &[Vec<f64>],
    n: usize,
    k: usize,
    n_init: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<usize> {
    assert_eq!(embedding.len(), n);
    assert!(k > 0 && k <= n);

    let dim = embedding[0].len();
    let mut best_labels = vec![0_usize; n];
    let mut best_inertia = f64::INFINITY;
    let mut rng_state = seed;

    // Simple LCG: state = state * a + c (mod 2^64)
    let lcg_next = |state: &mut u64| -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    };

    for _init in 0..n_init {
        // Pick k distinct random centers from data points
        let mut center_indices = Vec::with_capacity(k);
        let mut used = vec![false; n];
        while center_indices.len() < k {
            let idx = (lcg_next(&mut rng_state) % n as u64) as usize;
            if !used[idx] {
                used[idx] = true;
                center_indices.push(idx);
            }
        }
        let mut centers: Vec<Vec<f64>> = center_indices
            .iter()
            .map(|&i| embedding[i].clone())
            .collect();

        let mut labels = vec![0_usize; n];

        for _iter in 0..max_iter {
            // Assignment step: assign each point to nearest center
            let mut changed = false;
            for i in 0..n {
                let mut best_c = 0;
                let mut best_dist = f64::INFINITY;
                for c in 0..k {
                    let mut dist_sq = 0.0;
                    for d in 0..dim {
                        let diff = embedding[i][d] - centers[c][d];
                        dist_sq += diff * diff;
                    }
                    if dist_sq < best_dist {
                        best_dist = dist_sq;
                        best_c = c;
                    }
                }
                if labels[i] != best_c {
                    labels[i] = best_c;
                    changed = true;
                }
            }

            // Update step: recompute centers
            let mut new_centers = vec![vec![0.0; dim]; k];
            let mut counts = vec![0_usize; k];
            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                for d in 0..dim {
                    new_centers[c][d] += embedding[i][d];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        new_centers[c][d] /= counts[c] as f64;
                    }
                }
            }
            centers = new_centers;

            if !changed {
                break;
            }
        }

        // Compute inertia (sum of squared distances to assigned center)
        let mut inertia = 0.0;
        for i in 0..n {
            let c = labels[i];
            for d in 0..dim {
                let diff = embedding[i][d] - centers[c][d];
                inertia += diff * diff;
            }
        }

        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
        }
    }

    best_labels
}

// ── 10. Adjusted Rand Index ───────────────────────────────────────────────

/// Compute the Adjusted Rand Index between two label arrays.
///
/// ARI = 1.0 for perfect agreement, 0.0 for random, negative for
/// worse-than-random.
pub fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> f64 {
    let n = labels_a.len();
    assert_eq!(n, labels_b.len());
    if n == 0 {
        return 1.0;
    }

    // Determine number of classes in each labeling
    let max_a = labels_a.iter().copied().max().unwrap_or(0);
    let max_b = labels_b.iter().copied().max().unwrap_or(0);
    let na = max_a + 1;
    let nb = max_b + 1;

    // Build contingency table
    let mut contingency = vec![vec![0_i64; nb]; na];
    for i in 0..n {
        contingency[labels_a[i]][labels_b[i]] += 1;
    }

    // Row sums and column sums
    let row_sums: Vec<i64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let col_sums: Vec<i64> = (0..nb)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // Binomial coefficient C(x, 2) = x*(x-1)/2
    let comb2 = |x: i64| -> i64 { x * (x - 1) / 2 };

    // Sum of C(n_ij, 2) over all cells
    let sum_comb: i64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| comb2(v))
        .sum();

    // Sum of C(a_i, 2) and C(b_j, 2)
    let sum_comb_a: i64 = row_sums.iter().map(|&v| comb2(v)).sum();
    let sum_comb_b: i64 = col_sums.iter().map(|&v| comb2(v)).sum();

    let n_comb = comb2(n as i64);

    if n_comb == 0 {
        return 1.0;
    }

    let expected = sum_comb_a as f64 * sum_comb_b as f64 / n_comb as f64;
    let max_index = 0.5 * (sum_comb_a as f64 + sum_comb_b as f64);
    let denom = max_index - expected;

    if denom.abs() < 1e-14 {
        return 1.0; // all labels identical or trivial partition
    }

    (sum_comb as f64 - expected) / denom
}

// ── 11. Gram-Schmidt orthogonalization ────────────────────────────────────

/// Modified Gram-Schmidt orthogonalization of a set of vectors in-place.
/// Vectors that become near-zero after orthogonalization (linearly dependent
/// on preceding vectors) are zeroed out.
fn gram_schmidt(vectors: &mut [Vec<f64>]) {
    let k = vectors.len();
    for i in 0..k {
        let nrm = norm(&vectors[i]);
        if nrm < 1e-14 {
            for x in vectors[i].iter_mut() {
                *x = 0.0;
            }
            continue;
        }
        scale(1.0 / nrm, &mut vectors[i]);

        // Orthogonalize all subsequent vectors against vectors[i]
        let (left, right) = vectors.split_at_mut(i + 1);
        let vi = &left[i];
        for vj in right.iter_mut() {
            let c = dot(vi, vj);
            axpy(-c, vi, vj);
        }
    }
}

// ── 12. Dense symmetric eigensolver (Jacobi rotations) ───────────────────

/// Eigendecomposition of a small dense symmetric matrix using cyclic Jacobi
/// rotations. Suitable for projected matrices up to ~100x100.
///
/// Input: `a[i][j]` is the (i,j) entry of the symmetric matrix.
/// Returns `(eigenvalues, eigenvectors)` sorted ascending.
/// `eigenvectors[i]` is the eigenvector (as a Vec<f64>) for `eigenvalues[i]`.
fn dense_eigen_jacobi(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![a[0][0]], vec![vec![1.0]]);
    }

    // Working copy of the matrix
    let mut mat: Vec<Vec<f64>> = a.to_vec();

    // Eigenvector accumulator: v[j][i] = V_{i,j} (j-th column, i-th row).
    // Starts as the identity matrix.
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let mut col = vec![0.0; n];
            col[j] = 1.0;
            col
        })
        .collect();

    let max_sweeps = 200;

    for _sweep in 0..max_sweeps {
        // Find largest off-diagonal element
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mat[i][j].abs() > max_off {
                    max_off = mat[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-14 {
            break;
        }

        // Compute Jacobi rotation angle to zero out mat[p][q]
        let app = mat[p][p];
        let aqq = mat[q][q];
        let apq = mat[p][q];

        let (c, s) = if (app - aqq).abs() < 1e-30 {
            let angle = std::f64::consts::PI / 4.0;
            (angle.cos(), angle.sin())
        } else {
            // tau = (aqq - app) / (2 * apq), solve t^2 + 2*tau*t - 1 = 0
            // picking the smaller root for numerical stability
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            (c, t * c)
        };

        // Apply rotation to matrix: A' = G^T A G
        // where G_{pp}=c, G_{pq}=-s, G_{qp}=s, G_{qq}=c
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let mip = mat[i][p];
            let miq = mat[i][q];
            mat[i][p] = c * mip + s * miq;
            mat[p][i] = mat[i][p];
            mat[i][q] = -s * mip + c * miq;
            mat[q][i] = mat[i][q];
        }

        mat[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        mat[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;

        // Accumulate eigenvectors: V' = V * G (update columns p and q)
        for i in 0..n {
            let vip = v[p][i];
            let viq = v[q][i];
            v[p][i] = c * vip + s * viq;
            v[q][i] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues from the diagonal and sort ascending
    let evals: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap());

    let sorted_evals: Vec<f64> = idx.iter().map(|&i| evals[i]).collect();
    let sorted_evecs: Vec<Vec<f64>> = idx.iter().map(|&i| v[i].clone()).collect();

    (sorted_evals, sorted_evecs)
}

// ── 13. Chebyshev polynomial filter ──────────────────────────────────────

/// Apply a Chebyshev polynomial filter T_{poly_degree}((L - c) / e) to vector
/// `v`, where `c = (lo + hi) / 2` and `e = (hi - lo) / 2`.
///
/// This maps Laplacian eigenvalues in `[lo, hi]` to `[-1, 1]` via an affine
/// transform. Eigenvalues inside `[lo, hi]` produce bounded Chebyshev values
/// (oscillating in [-1, 1]), while eigenvalues outside produce exponentially
/// growing values. The filter therefore amplifies spectral components whose
/// eigenvalues lie outside `[lo, hi]`.
///
/// Each Chebyshev step costs one sparse Laplacian matvec. Uses the standard
/// three-term recurrence: T_0(x) = 1, T_1(x) = x, T_{j+1} = 2x T_j - T_{j-1}.
pub fn chebyshev_filter_apply(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    v: &[f64],
    lo: f64,
    hi: f64,
    poly_degree: usize,
) -> Vec<f64> {
    assert_eq!(v.len(), n);

    let width = hi - lo;
    if width.abs() < 1e-14 || poly_degree == 0 {
        return v.to_vec();
    }

    let c = (hi + lo) / 2.0;
    let e = (hi - lo) / 2.0;

    // T_1((L - c*I) / e) * v
    let lv = sparse_laplacian_matvec(edges, degree, n, v);
    let t1: Vec<f64> = (0..n).map(|i| (lv[i] - c * v[i]) / e).collect();
    if poly_degree == 1 {
        return t1;
    }

    // Three-term recurrence for T_2 .. T_{poly_degree}
    let mut t_prev = v.to_vec(); // T_0
    let mut t_curr = t1;

    for _ in 2..=poly_degree {
        let lt = sparse_laplacian_matvec(edges, degree, n, &t_curr);
        let t_next: Vec<f64> = (0..n)
            .map(|i| 2.0 * (lt[i] - c * t_curr[i]) / e - t_prev[i])
            .collect();
        t_prev = t_curr;
        t_curr = t_next;
    }

    t_curr
}

// ── 14. Chebyshev-filtered subspace iteration ────────────────────────────

/// Apply the folded-spectrum Chebyshev filter: T_m((B - c_b*I) / e_b) * v
/// where B = (L - sigma*I)^2. Each Chebyshev step costs two sparse matvecs
/// (one to form (L - sigma*I)*w and one to form (L - sigma*I) of that result).
///
/// The folded operator B has its smallest eigenvalues at the Laplacian
/// eigenvalues closest to sigma. By mapping the "unwanted" portion of B's
/// spectrum (large B-eigenvalues) into [-1, 1], the Chebyshev polynomial
/// amplifies the target components (small B-eigenvalues, i.e., L-eigenvalues
/// near sigma) and bounds everything else.
fn chebyshev_folded_apply(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    v: &[f64],
    sigma: f64,
    c_b: f64,
    e_b: f64,
    poly_degree: usize,
) -> Vec<f64> {
    if poly_degree == 0 {
        return v.to_vec();
    }

    // B * w = (L - sigma*I)^2 * w: two matvecs
    let apply_b = |w: &[f64]| -> Vec<f64> {
        let mut temp = sparse_laplacian_matvec(edges, degree, n, w);
        for i in 0..n {
            temp[i] -= sigma * w[i];
        }
        let mut result = sparse_laplacian_matvec(edges, degree, n, &temp);
        for i in 0..n {
            result[i] -= sigma * temp[i];
        }
        result
    };

    // T_1((B - c_b*I) / e_b) * v
    let bv = apply_b(v);
    let t1: Vec<f64> = (0..n).map(|i| (bv[i] - c_b * v[i]) / e_b).collect();
    if poly_degree == 1 {
        return t1;
    }

    let mut t_prev = v.to_vec(); // T_0
    let mut t_curr = t1;

    for _ in 2..=poly_degree {
        let bt = apply_b(&t_curr);
        let t_next: Vec<f64> = (0..n)
            .map(|i| 2.0 * (bt[i] - c_b * t_curr[i]) / e_b - t_prev[i])
            .collect();
        t_prev = t_curr;
        t_curr = t_next;
    }

    t_curr
}

/// Extract eigenpairs with eigenvalues in `[target_low, target_high]` using
/// Chebyshev-filtered subspace iteration with the folded-spectrum operator
/// B = (L - sigma * I)^2 where sigma = (target_low + target_high) / 2.
///
/// The folded spectrum converts the interior eigenvalue problem into an
/// extremal one: eigenvalues of L near sigma become the smallest eigenvalues
/// of B. A Chebyshev polynomial filter on B then amplifies these target
/// components while bounding everything else.
///
/// The algorithm iterates: (1) apply the Chebyshev filter to each vector,
/// (2) normalize, (3) re-orthogonalize via Gram-Schmidt. After convergence,
/// the filtered subspace is projected onto L and the small projected matrix
/// is eigendecomposed via Jacobi rotations.
///
/// # Arguments
/// * `edges` - undirected graph edges (src, tgt, generator)
/// * `degree` - vertex degree (uniform for Cayley graphs)
/// * `n` - number of vertices
/// * `target_low` - lower bound of target eigenvalue window
/// * `target_high` - upper bound of target eigenvalue window
/// * `n_vectors` - number of eigenpairs to extract
/// * `poly_degree` - Chebyshev filter polynomial degree (higher = sharper)
/// * `seed` - deterministic RNG seed
///
/// Returns `(eigenvalues, eigenvectors)` sorted ascending. Each eigenvector
/// has length n.
pub fn chebyshev_filtered_subspace(
    edges: &[(usize, usize, usize)],
    degree: usize,
    n: usize,
    target_low: f64,
    target_high: f64,
    n_vectors: usize,
    poly_degree: usize,
    seed: u64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let lo = 0.0_f64;
    let hi = 2.0 * degree as f64;

    // Folded-spectrum parameters
    let sigma = (target_low + target_high) / 2.0;
    let target_half_width = (target_high - target_low) / 2.0;
    let b_target_max = target_half_width * target_half_width;
    let b_max = {
        let d_lo = (sigma - lo) * (sigma - lo);
        let d_hi = (hi - sigma) * (hi - sigma);
        d_lo.max(d_hi)
    };

    // Mapping: [b_target_max, b_max] -> [-1, 1]
    // B-eigenvalues below b_target_max (the target) map outside [-1, 1]
    // and are amplified by the Chebyshev polynomial.
    let c_b = (b_max + b_target_max) / 2.0;
    let e_b = (b_max - b_target_max) / 2.0;

    if e_b.abs() < 1e-14 {
        return (vec![], vec![]);
    }

    // Generate random starting vectors using LCG (same generator as k-means)
    let mut rng_state = seed;
    let mut vectors: Vec<Vec<f64>> = (0..n_vectors)
        .map(|_| {
            (0..n)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (rng_state as f64) / (u64::MAX as f64) * 2.0 - 1.0
                })
                .collect()
        })
        .collect();

    // Subspace iteration: repeated filter + orthogonalize rounds
    let n_outer = 3;
    for _round in 0..n_outer {
        for v in vectors.iter_mut() {
            // Apply the folded-spectrum Chebyshev filter
            *v = chebyshev_folded_apply(edges, degree, n, v, sigma, c_b, e_b, poly_degree);

            // Normalize to prevent overflow/underflow
            let nrm = norm(v);
            if nrm > 1e-14 {
                scale(1.0 / nrm, v);
            }
        }

        // Re-orthogonalize the filtered vectors
        gram_schmidt(&mut vectors);
    }

    // Remove vectors that became zero during orthogonalization
    // (linearly dependent, i.e., the target eigenspace has fewer dimensions
    // than n_vectors)
    vectors.retain(|v| norm(v) > 0.5);
    let k = vectors.len();
    if k == 0 {
        return (vec![], vec![]);
    }

    // Project L into the filtered subspace: H[i][j] = v_i^T L v_j
    let mut h = vec![vec![0.0; k]; k];
    for j in 0..k {
        let lv = sparse_laplacian_matvec(edges, degree, n, &vectors[j]);
        for i in 0..k {
            h[i][j] = dot(&vectors[i], &lv);
        }
    }

    // Eigendecompose the small projected matrix via Jacobi rotations
    let (evals, evecs_small) = dense_eigen_jacobi(&h);

    // Project eigenvectors back to full n-dimensional space:
    // full_evec[i] = sum_j evecs_small[i][j] * vectors[j]
    let full_evecs: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut fv = vec![0.0; n];
            for j in 0..k {
                axpy(evecs_small[i][j], &vectors[j], &mut fv);
            }
            fv
        })
        .collect();

    (evals, full_evecs)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build cycle graph C_4: edges (0,1),(1,2),(2,3),(3,0), degree=2.
    /// Laplacian eigenvalues: 0, 2, 2, 4.
    #[test]
    fn test_sparse_laplacian_matvec_cycle4() {
        let edges: Vec<(usize, usize, usize)> = vec![(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)];
        let n = 4;
        let degree = 2;

        // The all-ones vector is in the kernel of the Laplacian
        let ones = vec![1.0; n];
        let result = sparse_laplacian_matvec(&edges, degree, n, &ones);
        for val in &result {
            assert!(val.abs() < 1e-14, "L * [1,1,1,1] should be zero");
        }

        // The alternating vector [1,-1,1,-1] is an eigenvector with eigenvalue 4
        let alt = vec![1.0, -1.0, 1.0, -1.0];
        let result = sparse_laplacian_matvec(&edges, degree, n, &alt);
        for (r, a) in result.iter().zip(alt.iter()) {
            assert!((r - 4.0 * a).abs() < 1e-14, "L * alt should be 4 * alt");
        }

        // [1,0,-1,0] is an eigenvector with eigenvalue 2
        let v = vec![1.0, 0.0, -1.0, 0.0];
        let result = sparse_laplacian_matvec(&edges, degree, n, &v);
        for (r, vi) in result.iter().zip(v.iter()) {
            assert!((r - 2.0 * vi).abs() < 1e-14, "L * v should be 2 * v");
        }
    }

    /// Test tridiag_eigen on a known 4x4 tridiagonal with analytically
    /// known eigenvalues (the C_4 Laplacian in tridiagonal form after
    /// Lanczos on the cycle).
    #[test]
    fn test_tridiag_eigen_4x4() {
        // A 4x4 symmetric tridiagonal:
        //   | 2  -1   0   0 |
        //   |-1   2  -1   0 |
        //   | 0  -1   2  -1 |
        //   | 0   0  -1   2 |
        // Eigenvalues: 2 - 2*cos(k*pi/4) for k=1..4
        //   k=1: 2 - sqrt(2) ~ 0.585786
        //   k=2: 2 - 0       = 2
        //   k=3: 2 + sqrt(2) ~ 3.414214
        //   k=4: 2 + 2       = 4  (wait, 2-2*cos(4*pi/4) = 2+2=4? no)
        //
        // Actually for the tridiagonal with 1s on the off-diagonal (not -1s), the
        // eigenvalues of the path Laplacian L_path = 2I - T where T has 1s are:
        // For T(n,n) with all 1s: eigenvalues of T are 2*cos(k*pi/(n+1)).
        //
        // Let's just use explicit values: alpha = [1, 2, 3, 4], beta = [0.5, 1.0, 0.5]
        // and verify against a direct dense multiplication.

        // Simple case: diagonal matrix (beta all zero)
        let alpha = vec![3.0, 1.0, 4.0, 2.0];
        let beta = vec![0.0, 0.0, 0.0];
        let (evals, _evecs) = tridiag_eigen(&alpha, &beta);
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        for (e, ex) in evals.iter().zip(expected.iter()) {
            assert!(
                (e - ex).abs() < 1e-10,
                "diagonal case: got {e}, expected {ex}"
            );
        }

        // Non-trivial case: tridiagonal with known eigenvalues
        // Path graph Laplacian (4 vertices): alpha=[2,2,2,2], beta=[-1,-1,-1]
        // but that's not quite right for a path. Let me use the standard form.
        //
        // Symmetric tridiagonal with alpha=[2,2,2,2], beta=[1,1,1].
        // This is the adjacency matrix of a path shifted by 2.
        // Eigenvalues of the path adjacency: 2*cos(k*pi/5) for k=1..4
        //   k=1: 2*cos(pi/5)   = 2*0.80902 = 1.61803
        //   k=2: 2*cos(2pi/5)  = 2*0.30902 = 0.61803
        //   k=3: 2*cos(3pi/5)  = -0.61803
        //   k=4: 2*cos(4pi/5)  = -1.61803
        // So our tridiagonal with alpha=[2,2,2,2], beta=[1,1,1] has eigenvalues:
        //   2+1.61803, 2+0.61803, 2-0.61803, 2-1.61803
        //   = 3.61803, 2.61803, 1.38197, 0.38197
        let alpha2 = vec![2.0, 2.0, 2.0, 2.0];
        let beta2 = vec![1.0, 1.0, 1.0];
        let (evals2, evecs2) = tridiag_eigen(&alpha2, &beta2);

        let expected2: Vec<f64> = (1..=4)
            .map(|k| 2.0 + 2.0 * (k as f64 * std::f64::consts::PI / 5.0).cos())
            .collect::<Vec<_>>();
        let mut expected2_sorted = expected2.clone();
        expected2_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (e, ex) in evals2.iter().zip(expected2_sorted.iter()) {
            assert!((e - ex).abs() < 1e-10, "path case: got {e}, expected {ex}");
        }

        // Verify eigenvectors: T * v_i = lambda_i * v_i
        let n = 4;
        for (i, (&lam, evec)) in evals2.iter().zip(evecs2.iter()).enumerate() {
            // Compute T * evec
            let mut tv = vec![0.0; n];
            for j in 0..n {
                tv[j] = alpha2[j] * evec[j];
                if j > 0 {
                    tv[j] += beta2[j - 1] * evec[j - 1];
                }
                if j + 1 < n {
                    tv[j] += beta2[j] * evec[j + 1];
                }
            }
            // Check T*v = lambda*v
            for j in 0..n {
                assert!(
                    (tv[j] - lam * evec[j]).abs() < 1e-10,
                    "eigenvector {i} component {j}: T*v={}, lam*v={}",
                    tv[j],
                    lam * evec[j]
                );
            }
        }
    }

    /// Build the S4 Cayley graph (adjacent transpositions) and verify that
    /// Lanczos with m=24 (full) recovers the known Laplacian spectrum.
    ///
    /// The 10 distinct eigenvalues (with multiplicities summing to 24) are:
    ///   0 (x1), 0.5858 (x3), 1.2679 (x2), 2 (x3), 2.5858 (x3),
    ///   3.4142 (x3), 4 (x3), 4.7321 (x2), 5.4142 (x3), 6 (x1)
    #[test]
    fn test_lanczos_s4_full_spectrum() {
        // Build S4 permutahedron edges with generator labels.
        // Permutations in lexicographic order of one-line notation.
        let n_group = 4_usize;
        let n_verts = 24_usize; // 4!
        let degree = 3_usize; // n-1 adjacent transpositions

        // Enumerate all permutations of [0,1,2,3] in lexicographic order
        let perms = lex_permutations(n_group);
        assert_eq!(perms.len(), n_verts);

        let perm_to_rank: std::collections::HashMap<Vec<u8>, usize> = perms
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();

        // Build edges: for each permutation, apply each generator (swap adjacent positions)
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for (rank, perm) in perms.iter().enumerate() {
            for g in 0..degree {
                let mut neighbor = perm.clone();
                neighbor.swap(g, g + 1);
                let neighbor_rank = perm_to_rank[&neighbor];
                if rank < neighbor_rank {
                    edges.push((rank, neighbor_rank, g));
                }
            }
        }
        assert_eq!(edges.len(), 36); // 24 * 3 / 2

        // Expected 10 distinct eigenvalues of the S4 Cayley-graph Laplacian.
        let expected_distinct = vec![
            0.0,
            2.0 - std::f64::consts::SQRT_2, // 0.585786...
            1.267949192431123,
            2.0,
            2.0 + 2.0 - std::f64::consts::SQRT_2, // 2.585786...
            2.0 + std::f64::consts::SQRT_2,       // 3.414214...
            4.0,
            4.732050807568877,
            4.0 + std::f64::consts::SQRT_2, // 5.414214...
            6.0,
        ];

        // --- Test 1: single delta start vector ---
        // A delta function at the identity has nonzero overlap with every
        // irrep, so its Krylov subspace has dimension = number of distinct
        // eigenvalues (10 for S4).  Lanczos terminates early.
        let mut start_delta = vec![0.0_f64; n_verts];
        start_delta[0] = 1.0;

        let lr_delta = lanczos(&edges, degree, n_verts, &start_delta, n_verts);

        // Should produce exactly 10 Lanczos vectors
        assert_eq!(
            lr_delta.alpha.len(),
            10,
            "Krylov subspace of delta should be 10-dimensional"
        );

        let (evals_delta, _) = tridiag_eigen(&lr_delta.alpha, &lr_delta.beta);

        // Each of the 10 Ritz values should match a distinct eigenvalue
        assert_eq!(evals_delta.len(), 10);
        for &exp in &expected_distinct {
            let count = evals_delta
                .iter()
                .filter(|&&e| (e - exp).abs() < 1e-6)
                .count();
            assert!(
                count == 1,
                "expected eigenvalue {exp:.6} not found (or duplicated) in delta-start spectrum"
            );
        }

        // --- Test 2: dense start vector ---
        // The minimal polynomial of the S4 Laplacian has degree 10 (=
        // number of distinct eigenvalues), so the Krylov subspace is
        // always at most 10-dimensional.  Verify that a different start
        // vector also recovers all 10 distinct eigenvalues.
        let start_full: Vec<f64> = (0..n_verts)
            .map(|i| ((i as f64 + 1.0) * 0.7).sin())
            .collect();

        let lr_full = lanczos(&edges, degree, n_verts, &start_full, n_verts);
        let (evals_full, _) = tridiag_eigen(&lr_full.alpha, &lr_full.beta);

        assert_eq!(evals_full.len(), 10, "Krylov dimension should be 10");

        for &exp in &expected_distinct {
            let count = evals_full
                .iter()
                .filter(|&&e| (e - exp).abs() < 1e-6)
                .count();
            assert_eq!(
                count, 1,
                "eigenvalue {exp:.6} not found or duplicated in full-start spectrum"
            );
        }

        // Verify the spectral density sums to ||start||^2.
        let density = spectral_density(&lr_full, &start_full);
        let total_e: f64 = density.iter().map(|(_, e)| e).sum();
        let expected_e: f64 = dot(&start_full, &start_full);
        assert!(
            (total_e - expected_e).abs() < 1e-6,
            "spectral density total {total_e} != ||f||^2 = {expected_e}"
        );
    }

    /// Test spectral_density on a small example.
    #[test]
    fn test_spectral_density_cycle4() {
        let edges: Vec<(usize, usize, usize)> = vec![(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)];
        let n = 4;
        let degree = 2;

        // Start from a specific vector
        let start = vec![1.0, 0.0, 0.0, 0.0];
        let lr = lanczos(&edges, degree, n, &start, n);
        let density = spectral_density(&lr, &start);

        // Eigenvalues of C_4 Laplacian: 0, 2, 2, 4
        // Total energy should equal ||start||^2 = 1
        let total_energy: f64 = density.iter().map(|(_lam, e)| e).sum();
        assert!(
            (total_energy - 1.0).abs() < 1e-10,
            "total energy should be 1, got {total_energy}"
        );

        // All energies should be non-negative
        for (lam, e) in &density {
            assert!(*e >= -1e-14, "negative energy {e} at eigenvalue {lam}");
        }
    }

    /// Helper: generate all permutations of [0..n) in lexicographic order.
    fn lex_permutations(n: usize) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        let mut perm: Vec<u8> = (0..n as u8).collect();
        loop {
            result.push(perm.clone());
            // Next permutation in lex order
            let mut i = perm.len().wrapping_sub(1);
            if i == usize::MAX {
                break;
            }
            while i > 0 && perm[i - 1] >= perm[i] {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            let pivot = i - 1;
            let mut j = perm.len() - 1;
            while perm[j] <= perm[pivot] {
                j -= 1;
            }
            perm.swap(pivot, j);
            perm[pivot + 1..].reverse();
        }
        result
    }

    /// Helper: build S4 Cayley graph edges and return (edges, perms, perm_to_rank).
    fn build_s4_graph() -> (
        Vec<(usize, usize, usize)>,
        Vec<Vec<u8>>,
        std::collections::HashMap<Vec<u8>, usize>,
    ) {
        let n_group = 4_usize;
        let degree = 3_usize;
        let perms = lex_permutations(n_group);
        let perm_to_rank: std::collections::HashMap<Vec<u8>, usize> = perms
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();

        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for (rank, perm) in perms.iter().enumerate() {
            for g in 0..degree {
                let mut neighbor = perm.clone();
                neighbor.swap(g, g + 1);
                let neighbor_rank = perm_to_rank[&neighbor];
                if rank < neighbor_rank {
                    edges.push((rank, neighbor_rank, g));
                }
            }
        }
        (edges, perms, perm_to_rank)
    }

    /// Helper: compute V4 = {(), (01)(23), (02)(13), (03)(12)} coset labels
    /// for all S4 permutations.
    fn v4_coset_labels(
        perms: &[Vec<u8>],
        perm_to_rank: &std::collections::HashMap<Vec<u8>, usize>,
    ) -> Vec<usize> {
        // V4 generators (acting on 0-indexed positions):
        //   (0 1)(2 3): [1,0,3,2]
        //   (0 2)(1 3): [2,3,0,1]
        //   (0 3)(1 2): [3,2,1,0]
        let v4_elements: Vec<Vec<u8>> = vec![
            vec![0, 1, 2, 3], // identity
            vec![1, 0, 3, 2], // (01)(23)
            vec![2, 3, 0, 1], // (02)(13)
            vec![3, 2, 1, 0], // (03)(12)
        ];

        let n_verts = perms.len();
        let mut labels = vec![0_usize; n_verts];
        let mut visited = vec![false; n_verts];
        let mut coset_id = 0_usize;

        for i in 0..n_verts {
            if visited[i] {
                continue;
            }
            // Compute the right coset: {v4 * perms[i]} for each v4 element
            for v4_elem in &v4_elements {
                // Compose: (v4_elem)(perms[i]) means apply perms[i] first, then v4_elem
                // Composition: result[pos] = v4_elem[perms[i][pos]]
                let composed: Vec<u8> = perms[i].iter().map(|&p| v4_elem[p as usize]).collect();
                let rank = perm_to_rank[&composed];
                labels[rank] = coset_id;
                visited[rank] = true;
            }
            coset_id += 1;
        }

        labels
    }

    #[test]
    fn test_shift_invert_s4() {
        let (edges, _perms, _perm_to_rank) = build_s4_graph();
        let n_verts = 24_usize;
        let degree = 3_usize;

        // Target eigenvalues near sigma=3.0
        // The S4 Laplacian eigenvalues near 3.0 are:
        //   2.585786... (2+sqrt(2)-2 = 2-sqrt(2)+2... actually 4-sqrt(2) ~ 2.5858)
        //   3.414214... (2+sqrt(2))
        // and the (2,2) irrep eigenvalues are ~1.268 and ~4.732
        let sigma = 3.0;
        // Use a delta vector (broad spectral content, not an eigenvector)
        let mut start = vec![0.0_f64; n_verts];
        start[0] = 1.0;

        let lr = shift_invert_lanczos(&edges, degree, n_verts, sigma, &start, 24, 1e-10, 1000);
        let (ritz_values, _ritz_vectors) = tridiag_eigen(&lr.alpha, &lr.beta);

        // Convert Ritz values back to Laplacian eigenvalues
        let laplacian_evals: Vec<f64> = ritz_values
            .iter()
            .map(|&rv| {
                if rv.abs() < 1e-14 {
                    f64::INFINITY
                } else {
                    sigma + 1.0 / rv
                }
            })
            .collect();

        // The known S4 eigenvalues
        let expected_near_3 = vec![
            2.0 + 2.0 - std::f64::consts::SQRT_2, // 2.585786
            2.0 + std::f64::consts::SQRT_2,       // 3.414214
        ];

        // Verify that eigenvalues near sigma=3.0 are recovered accurately
        for &exp in &expected_near_3 {
            let best_match = laplacian_evals
                .iter()
                .filter(|e| e.is_finite())
                .map(|&e| (e - exp).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(
                best_match < 0.01,
                "expected eigenvalue {exp:.6} not found near sigma=3.0, best distance = {best_match:.6}"
            );
        }

        // Also check that the (2,2) eigenvalues (1.268 and 4.732) appear
        let expected_22 = vec![1.267949192431123, 4.732050807568877];
        for &exp in &expected_22 {
            let best_match = laplacian_evals
                .iter()
                .filter(|e| e.is_finite())
                .map(|&e| (e - exp).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(
                best_match < 0.01,
                "expected (2,2) eigenvalue {exp:.6} not found, best distance = {best_match:.6}"
            );
        }
    }

    #[test]
    fn test_targeted_clustering_s4() {
        let (edges, perms, perm_to_rank) = build_s4_graph();
        let n_verts = 24_usize;
        let degree = 3_usize;

        // Get V4 coset labels (ground truth): 6 cosets of size 4
        let true_labels = v4_coset_labels(&perms, &perm_to_rank);

        // Verify we have 6 cosets of 4 elements each
        let n_cosets = *true_labels.iter().max().unwrap() + 1;
        assert_eq!(n_cosets, 6, "S4/V4 should have 6 cosets");
        for c in 0..n_cosets {
            let count = true_labels.iter().filter(|&&l| l == c).count();
            assert_eq!(count, 4, "each V4 coset should have 4 elements");
        }

        // The (2,2) irrep eigenvalues are at 1.268 (mult 2) and 4.732 (mult 2).
        // We need eigenvectors from BOTH to span the V4 coset indicator subspace.
        // Run two shift-invert probes: one near each eigenvalue.
        let mut all_evecs: Vec<Vec<f64>> = Vec::new();

        for sigma in [1.27, 4.73] {
            let (evals, evecs) =
                targeted_eigenspace_embedding(&edges, degree, n_verts, sigma, 3, 1e-12, 2000);
            // Select eigenvectors with eigenvalue close to the target
            for (i, &ev) in evals.iter().enumerate() {
                if (ev - sigma).abs() < 0.1 {
                    all_evecs.push(evecs[i].clone());
                }
            }
        }

        assert!(
            all_evecs.len() >= 2,
            "need at least 2 (2,2) eigenvectors, got {}",
            all_evecs.len()
        );

        // Build embedding from the selected eigenvectors
        let embedding: Vec<Vec<f64>> = (0..n_verts)
            .map(|i| all_evecs.iter().map(|v| v[i]).collect())
            .collect();

        // Cluster into 6 groups
        let predicted = kmeans_clustering(&embedding, n_verts, 6, 50, 200, 42);

        // Compute ARI: should be 1.0 or very close
        let ari = adjusted_rand_index(&true_labels, &predicted);
        assert!(
            ari > 0.9,
            "ARI should be > 0.9 for V4 coset recovery via (2,2) eigenspace, got {ari:.4}"
        );
    }

    #[test]
    fn test_ari_perfect() {
        // Two identical label arrays should give ARI = 1.0
        let labels = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let ari = adjusted_rand_index(&labels, &labels);
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "ARI of identical labels should be 1.0, got {ari}"
        );

        // Relabeled version (permuted labels) should also give 1.0
        let relabeled = vec![2, 2, 0, 0, 3, 3, 1, 1];
        let ari2 = adjusted_rand_index(&labels, &relabeled);
        assert!(
            (ari2 - 1.0).abs() < 1e-10,
            "ARI of permuted labels should be 1.0, got {ari2}"
        );
    }

    #[test]
    fn test_ari_random() {
        // Structured labels vs pseudorandom labels should give ARI near 0
        let n = 100;
        let structured: Vec<usize> = (0..n).map(|i| i / 10).collect(); // 10 groups of 10

        // Deterministic pseudorandom labels via LCG
        let mut state = 12345_u64;
        let random_labels: Vec<usize> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 32) as usize % 10
            })
            .collect();

        let ari = adjusted_rand_index(&structured, &random_labels);
        assert!(
            ari.abs() < 0.15,
            "ARI of structured vs random labels should be near 0, got {ari:.4}"
        );
    }

    #[test]
    fn test_chebyshev_filter_s4() {
        let (edges, _perms, _perm_to_rank) = build_s4_graph();
        let n_verts = 24_usize;
        let degree = 3_usize;

        // Target window [1.0, 1.5] should capture the [2,2] irrep eigenvalue
        // at 1.268 (multiplicity 2).
        let (evals, evecs) = chebyshev_filtered_subspace(
            &edges, degree, n_verts, 1.0, 1.5, // target window
            4,   // n_vectors (more than multiplicity for robustness)
            30,  // poly_degree
            42,  // seed
        );

        // Should have extracted at least 2 eigenvectors
        assert!(
            evals.len() >= 2,
            "expected at least 2 eigenvectors, got {}",
            evals.len()
        );

        // At least one eigenvalue should be close to 1.268
        let target = 1.267949192431123;
        let best_match = evals
            .iter()
            .map(|&e| (e - target).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_match < 0.05,
            "expected eigenvalue near {target:.6}, best distance = {best_match:.6}, evals = {evals:?}"
        );

        // Verify via Rayleigh quotient: v^T L v / v^T v should match eigenvalue
        for (i, (&eval, evec)) in evals.iter().zip(evecs.iter()).enumerate() {
            let lv = sparse_laplacian_matvec(&edges, degree, n_verts, evec);
            let rayleigh = dot(evec, &lv) / dot(evec, evec);
            assert!(
                (rayleigh - eval).abs() < 0.01,
                "eigenvector {i}: Rayleigh quotient {rayleigh:.6} != eigenvalue {eval:.6}"
            );
        }
    }

    #[test]
    fn test_chebyshev_filter_s4_high_window() {
        let (edges, _perms, _perm_to_rank) = build_s4_graph();
        let n_verts = 24_usize;
        let degree = 3_usize;

        // Target window [4.5, 5.0] should capture the other [2,2] irrep
        // eigenvalue at 4.732 (multiplicity 2).
        let (evals, evecs) = chebyshev_filtered_subspace(
            &edges, degree, n_verts, 4.5, 5.0, // target window
            4,   // n_vectors
            30,  // poly_degree
            123, // seed
        );

        assert!(
            evals.len() >= 2,
            "expected at least 2 eigenvectors, got {}",
            evals.len()
        );

        let target = 4.732050807568877;
        let best_match = evals
            .iter()
            .map(|&e| (e - target).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_match < 0.05,
            "expected eigenvalue near {target:.6}, best distance = {best_match:.6}, evals = {evals:?}"
        );

        for (i, (&eval, evec)) in evals.iter().zip(evecs.iter()).enumerate() {
            let lv = sparse_laplacian_matvec(&edges, degree, n_verts, evec);
            let rayleigh = dot(evec, &lv) / dot(evec, evec);
            assert!(
                (rayleigh - eval).abs() < 0.01,
                "eigenvector {i}: Rayleigh quotient {rayleigh:.6} != eigenvalue {eval:.6}"
            );
        }
    }

    fn build_s8_test_data() -> (Vec<(usize, usize, usize)>, usize, usize, Vec<usize>, usize) {
        let graph = crate::permutahedron::complete_graph(8).unwrap();
        let edges: Vec<(usize, usize, usize)> = graph
            .edges
            .iter()
            .map(|e| (e[0] as usize, e[1] as usize, 0))
            .collect();
        let r8 = crate::permutahedron::rana_r8();
        let partition =
            crate::permutahedron::coset_partition(&r8, crate::permutahedron::CosetSide::Right)
                .unwrap();
        let n_cosets = partition.slices.len();
        let mut coset_labels = vec![0usize; graph.vertex_count];
        for (i, sl) in partition.slices.iter().enumerate() {
            for &v in sl {
                coset_labels[v as usize] = i;
            }
        }
        (
            edges,
            graph.degree,
            graph.vertex_count,
            coset_labels,
            n_cosets,
        )
    }

    #[test]
    fn test_chebyshev_narrow_s8_low_window() {
        let (edges, degree, n_verts, coset_labels, n_cosets) = build_s8_test_data();

        for &n_vecs in &[20, 50, 100] {
            let (evals, evecs) =
                chebyshev_filtered_subspace(&edges, degree, n_verts, 6.3, 6.7, n_vecs, 40, 42);
            println!(
                "[6.3, 6.7] n_vectors={}: found {} eigenvalues",
                n_vecs,
                evals.len()
            );
            for (i, &ev) in evals.iter().enumerate().take(10) {
                println!("  {:2}. lambda = {:.6}", i, ev);
            }
            if !evecs.is_empty() {
                let dim = n_vecs.min(evecs.len());
                let embedding: Vec<Vec<f64>> = (0..n_verts)
                    .map(|v| evecs.iter().take(dim).map(|ev| ev[v]).collect())
                    .collect();
                let predicted = kmeans_clustering(&embedding, n_verts, n_cosets, 3, 50, 42);
                let ari = adjusted_rand_index(&coset_labels, &predicted);
                println!("  ARI = {:.6} (dim={})", ari, dim);
            }
        }
    }

    #[test]
    fn test_chebyshev_narrow_s8_high_window() {
        let (edges, degree, n_verts, coset_labels, n_cosets) = build_s8_test_data();

        for &n_vecs in &[20, 50, 100] {
            let (evals, evecs) =
                chebyshev_filtered_subspace(&edges, degree, n_verts, 7.3, 7.7, n_vecs, 40, 42);
            println!(
                "[7.3, 7.7] n_vectors={}: found {} eigenvalues",
                n_vecs,
                evals.len()
            );
            for (i, &ev) in evals.iter().enumerate().take(10) {
                println!("  {:2}. lambda = {:.6}", i, ev);
            }
            if !evecs.is_empty() {
                let dim = n_vecs.min(evecs.len());
                let embedding: Vec<Vec<f64>> = (0..n_verts)
                    .map(|v| evecs.iter().take(dim).map(|ev| ev[v]).collect())
                    .collect();
                let predicted = kmeans_clustering(&embedding, n_verts, n_cosets, 3, 50, 42);
                let ari = adjusted_rand_index(&coset_labels, &predicted);
                println!("  ARI = {:.6} (dim={})", ari, dim);
            }
        }
    }

    #[test]
    fn test_chebyshev_dual_window_s8() {
        let (edges, degree, n_verts, coset_labels, n_cosets) = build_s8_test_data();

        let n_vecs = 50;
        println!("Extracting {} vectors from [6.3, 6.7]...", n_vecs);
        let (evals_low, evecs_low) =
            chebyshev_filtered_subspace(&edges, degree, n_verts, 6.3, 6.7, n_vecs, 40, 42);
        println!("  Found {} eigenvalues in [6.3, 6.7]", evals_low.len());
        for (i, &ev) in evals_low.iter().enumerate().take(5) {
            println!("  {:2}. lambda = {:.6}", i, ev);
        }

        println!("Extracting {} vectors from [7.3, 7.7]...", n_vecs);
        let (evals_high, evecs_high) =
            chebyshev_filtered_subspace(&edges, degree, n_verts, 7.3, 7.7, n_vecs, 40, 123);
        println!("  Found {} eigenvalues in [7.3, 7.7]", evals_high.len());
        for (i, &ev) in evals_high.iter().enumerate().take(5) {
            println!("  {:2}. lambda = {:.6}", i, ev);
        }

        let n_low = evecs_low.len().min(n_vecs);
        let n_high = evecs_high.len().min(n_vecs);
        let total_dim = n_low + n_high;
        println!(
            "Combined embedding: {} + {} = {} dimensions",
            n_low, n_high, total_dim
        );

        let embedding: Vec<Vec<f64>> = (0..n_verts)
            .map(|v| {
                let mut coords = Vec::with_capacity(total_dim);
                for ev in evecs_low.iter().take(n_low) {
                    coords.push(ev[v]);
                }
                for ev in evecs_high.iter().take(n_high) {
                    coords.push(ev[v]);
                }
                coords
            })
            .collect();

        let predicted = kmeans_clustering(&embedding, n_verts, n_cosets, 3, 50, 42);
        let ari = adjusted_rand_index(&coset_labels, &predicted);
        println!("Combined dual-window ARI = {:.6} (dim={})", ari, total_dim);

        if n_low > 0 {
            let embed_low: Vec<Vec<f64>> = (0..n_verts)
                .map(|v| evecs_low.iter().take(n_low).map(|ev| ev[v]).collect())
                .collect();
            let pred_low = kmeans_clustering(&embed_low, n_verts, n_cosets, 3, 50, 42);
            let ari_low = adjusted_rand_index(&coset_labels, &pred_low);
            println!("[6.3,6.7] alone ARI = {:.6}", ari_low);
        }
        if n_high > 0 {
            let embed_high: Vec<Vec<f64>> = (0..n_verts)
                .map(|v| evecs_high.iter().take(n_high).map(|ev| ev[v]).collect())
                .collect();
            let pred_high = kmeans_clustering(&embed_high, n_verts, n_cosets, 3, 50, 42);
            let ari_high = adjusted_rand_index(&coset_labels, &pred_high);
            println!("[7.3,7.7] alone ARI = {:.6}", ari_high);
        }
    }
}
