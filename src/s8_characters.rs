//! Character values of S_n irreducible representations via the
//! Murnaghan-Nakayama rule, with spectral predictions for Cayley-graph
//! Laplacian eigenspaces carrying R_n coset information.
//!
//! Key formulas for S_n with adjacent-transposition generators s_1 .. s_{n-1}
//! and a subgroup R_n of order |R_n| whose non-identity elements all share one
//! conjugacy class:
//!
//!   m_lambda  = (1/|R_n|)(d_lambda + (|R_n|-1) * chi_lambda(coset class))
//!   l_lambda  = (n-1) - (n-1) * chi_lambda(transposition) / d_lambda
//!
//! m_lambda counts copies of the trivial R_n-rep inside the restriction of
//! lambda; l_lambda is the centroid of the Laplacian eigenvalues within the
//! lambda-isotypic block.

use serde::Serialize;

// ---------------------------------------------------------------------------
// Partition utilities
// ---------------------------------------------------------------------------

/// All integer partitions of `n`, each in weakly decreasing order, enumerated
/// in reverse lexicographic order.
pub fn partitions(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut buf = Vec::new();
    partitions_inner(n, n, &mut buf, &mut result);
    result
}

fn partitions_inner(
    remaining: usize,
    max_part: usize,
    buf: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        out.push(buf.clone());
        return;
    }
    for p in (1..=remaining.min(max_part)).rev() {
        buf.push(p);
        partitions_inner(remaining - p, p, buf, out);
        buf.pop();
    }
}

// ---------------------------------------------------------------------------
// Hook-length dimension
// ---------------------------------------------------------------------------

/// Dimension of the S_n irrep indexed by `partition`, via the hook-length
/// formula: n! / prod(hook lengths).
///
/// Each cell (i, j) of the Young diagram has hook length
///   h(i,j) = (arm length) + (leg length) + 1
///          = (lambda_i - j) + (lambda'_j - i) - 1
/// where lambda' is the conjugate partition.
pub fn hook_length_dimension(partition: &[usize]) -> u64 {
    if partition.is_empty() {
        return 1;
    }
    let n: usize = partition.iter().sum();

    // Conjugate partition: lambda'_j = number of parts >= j+1
    let width = partition[0];
    let mut conjugate = vec![0usize; width];
    for &part in partition {
        for j in 0..part {
            conjugate[j] += 1;
        }
    }

    let mut hook_prod: u64 = 1;
    for (i, &li) in partition.iter().enumerate() {
        for j in 0..li {
            let h = (li - j) as u64 + (conjugate[j] - i) as u64 - 1;
            hook_prod *= h;
        }
    }
    factorial_u64(n as u64) / hook_prod
}

fn factorial_u64(n: u64) -> u64 {
    (1..=n).product()
}

// ---------------------------------------------------------------------------
// Border strips (rim hooks)
// ---------------------------------------------------------------------------

/// Enumerate all valid border-strip (rim-hook) removals of `length` cells from
/// `partition`.  Returns `(remaining_partition, sign)` where
///   sign = (-1)^height,   height = (rows spanned) - 1.
///
/// A border strip spanning rows t..=b in the Young diagram is uniquely
/// determined by the pair (t, b): each intermediate row i contributes
/// partition\[i\] - partition\[i+1\] + 1 cells (starting at column
/// partition\[i+1\]-1), while the bottom row b contributes the remainder.
/// The bottom-row starting column is:
///
///   c_b = partition\[t\] + (b - t) - length
///
/// The strip is valid iff c_b >= 0, c_b < partition\[b\] (at least one cell
/// removed from row b), and c_b >= partition\[b+1\] (remaining partition stays
/// weakly decreasing).
pub fn border_strips(partition: &[usize], length: usize) -> Vec<(Vec<usize>, i32)> {
    let k = partition.len();
    if length == 0 || k == 0 {
        return Vec::new();
    }
    let mut result = Vec::new();

    for t in 0..k {
        for b in t..k {
            let c_b_signed = partition[t] as i64 + (b as i64 - t as i64) - length as i64;
            if c_b_signed < 0 {
                continue;
            }
            let c_b = c_b_signed as usize;

            // Remaining partition at row b must be >= row b+1 (if it exists)
            let lower = if b + 1 < k { partition[b + 1] } else { 0 };
            if c_b < lower || c_b >= partition[b] {
                continue;
            }

            // Build the remaining partition mu.
            // Rows outside [t, b] are unchanged.
            // Rows t..b-1: mu[i] = partition[i+1] - 1 (the overlap column).
            // Row b: mu[b] = c_b.
            let mut mu = partition.to_vec();
            for i in t..b {
                mu[i] = partition[i + 1] - 1;
            }
            mu[b] = c_b;

            // Drop trailing zero-width rows
            while mu.last() == Some(&0) {
                mu.pop();
            }

            let sign = if (b - t) % 2 == 0 { 1i32 } else { -1 };
            result.push((mu, sign));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Murnaghan-Nakayama character computation
// ---------------------------------------------------------------------------

/// Character chi_lambda(mu) of the S_n irrep indexed by `lambda` evaluated at
/// the conjugacy class with cycle type `mu`, via the Murnaghan-Nakayama rule.
///
/// Both `lambda` and `mu` must be partitions of the same integer n, given in
/// weakly decreasing order.  The algorithm peels off mu\[0\] as the first
/// cycle length, enumerates all border-strip removals of that length from
/// lambda, and recurses on the remaining partition and cycle type.
pub fn murnaghan_nakayama(lambda: &[usize], mu: &[usize]) -> i64 {
    // Base cases
    if mu.is_empty() {
        return if lambda.is_empty() { 1 } else { 0 };
    }
    if lambda.is_empty() {
        return 0;
    }

    let r = mu[0];
    let rest = &mu[1..];
    let mut total = 0i64;
    for (sub, sign) in border_strips(lambda, r) {
        total += sign as i64 * murnaghan_nakayama(&sub, rest);
    }
    total
}

// ---------------------------------------------------------------------------
// Spectral prediction types
// ---------------------------------------------------------------------------

/// Per-irrep prediction of Laplacian eigenspace properties.
#[derive(Debug, Clone, Serialize)]
pub struct IrrepPrediction {
    /// The partition indexing this irrep.
    pub partition: Vec<usize>,
    /// Dimension d_lambda of the irrep.
    pub dimension: u64,
    /// chi_lambda evaluated at the single-transposition conjugacy class
    /// (cycle type (2, 1^{n-2})).
    pub chi_transposition: i64,
    /// chi_lambda evaluated at the coset-diagnostic conjugacy class
    /// (cycle type (2^{n/2}) for even n).
    pub chi_coset_class: i64,
    /// Multiplicity of trivial R_n-representation inside the restriction of
    /// lambda to R_n.
    pub m_lambda: i64,
    /// Centroid of the Laplacian eigenvalues within the lambda-isotypic block:
    /// l = (n-1) - (n-1)*chi_trans/d.
    pub l_lambda_centroid: f64,
    /// True when m_lambda > 0, indicating this eigenspace block carries R_n
    /// coset information.
    pub carries_coset_info: bool,
}

/// Collected spectral predictions for all irreps of S_n.
#[derive(Debug, Clone, Serialize)]
pub struct SpectralPrediction {
    pub n: usize,
    /// |S_n| = n!.
    pub group_order: u64,
    /// |R_n|, the subgroup whose cosets we are probing.
    pub subgroup_order: u64,
    pub num_irreps: usize,
    pub irreps: Vec<IrrepPrediction>,
    /// Sum of d_lambda^2 across all irreps (should equal n! by Burnside).
    pub sum_dim_squared: u64,
    /// Sum of m_lambda * d_lambda (should equal |S_n| / |R_n|).
    pub total_coset_multiplicity: i64,
}

// ---------------------------------------------------------------------------
// Public prediction entry points
// ---------------------------------------------------------------------------

/// Spectral prediction for all 22 irreps of S8 against R8 (order 8, with
/// seven non-identity elements of cycle type (2^4)).
pub fn s8_spectral_prediction() -> SpectralPrediction {
    spectral_prediction(8, 8, &[2, 2, 2, 2])
}

/// Spectral prediction for all 5 irreps of S4 against V4 = R4 (order 4, with
/// three non-identity elements of cycle type (2^2)).
pub fn s4_spectral_prediction() -> SpectralPrediction {
    spectral_prediction(4, 4, &[2, 2])
}

fn spectral_prediction(
    n: usize,
    subgroup_order: u64,
    coset_cycle_type: &[usize],
) -> SpectralPrediction {
    let parts = partitions(n);

    // Transposition class: (2, 1, 1, ..., 1) with n-2 ones
    let mut trans_class = vec![1usize; n - 1];
    trans_class[0] = 2;

    let nontrivial = subgroup_order as i64 - 1;
    let generators = (n - 1) as f64;

    let mut irreps = Vec::new();
    let mut sum_d2 = 0u64;
    let mut total_md = 0i64;

    for p in &parts {
        let d = hook_length_dimension(p);
        let chi_t = murnaghan_nakayama(p, &trans_class);
        let chi_c = murnaghan_nakayama(p, coset_cycle_type);
        let m = (d as i64 + nontrivial * chi_c) / subgroup_order as i64;
        let l = generators - generators * chi_t as f64 / d as f64;

        sum_d2 += d * d;
        total_md += m * d as i64;

        irreps.push(IrrepPrediction {
            partition: p.clone(),
            dimension: d,
            chi_transposition: chi_t,
            chi_coset_class: chi_c,
            m_lambda: m,
            l_lambda_centroid: l,
            carries_coset_info: m > 0,
        });
    }

    SpectralPrediction {
        n,
        group_order: factorial_u64(n as u64),
        subgroup_order,
        num_irreps: parts.len(),
        irreps,
        sum_dim_squared: sum_d2,
        total_coset_multiplicity: total_md,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Partition enumeration ----

    #[test]
    fn partition_counts() {
        assert_eq!(partitions(4).len(), 5);
        assert_eq!(partitions(8).len(), 22);
    }

    #[test]
    fn partitions_of_4_enumerated() {
        let p = partitions(4);
        assert_eq!(p[0], vec![4]);
        assert_eq!(p[1], vec![3, 1]);
        assert_eq!(p[2], vec![2, 2]);
        assert_eq!(p[3], vec![2, 1, 1]);
        assert_eq!(p[4], vec![1, 1, 1, 1]);
    }

    // ---- Hook-length dimensions ----

    #[test]
    fn s4_dimensions() {
        let dims: Vec<u64> = partitions(4)
            .iter()
            .map(|p| hook_length_dimension(p))
            .collect();
        assert_eq!(dims, vec![1, 3, 2, 3, 1]);
    }

    #[test]
    fn s4_burnside_sum_of_squares() {
        let total: u64 = partitions(4)
            .iter()
            .map(|p| {
                let d = hook_length_dimension(p);
                d * d
            })
            .sum();
        assert_eq!(total, 24);
    }

    // ---- S4 characters at transposition class (2,1,1) ----

    #[test]
    fn s4_chi_at_transposition() {
        let mu = vec![2, 1, 1];
        let vals: Vec<i64> = partitions(4)
            .iter()
            .map(|p| murnaghan_nakayama(p, &mu))
            .collect();
        // (4)->1, (3,1)->1, (2,2)->0, (2,1,1)->-1, (1^4)->-1
        assert_eq!(vals, vec![1, 1, 0, -1, -1]);
    }

    #[test]
    fn s4_laplacian_trivial_zero() {
        let pred = s4_spectral_prediction();
        assert!(
            pred.irreps[0].l_lambda_centroid.abs() < 1e-12,
            "trivial rep should have eigenvalue 0"
        );
    }

    #[test]
    fn s4_laplacian_sign_max() {
        let pred = s4_spectral_prediction();
        assert!(
            (pred.irreps[4].l_lambda_centroid - 6.0).abs() < 1e-12,
            "sign rep should have eigenvalue 2*(n-1) = 6"
        );
    }

    // ---- S4 characters at double-transposition class (2,2) ----

    #[test]
    fn s4_chi_at_double_transposition() {
        let mu = vec![2, 2];
        let vals: Vec<i64> = partitions(4)
            .iter()
            .map(|p| murnaghan_nakayama(p, &mu))
            .collect();
        // (4)->1, (3,1)->-1, (2,2)->2, (2,1,1)->-1, (1^4)->1
        assert_eq!(vals, vec![1, -1, 2, -1, 1]);
    }

    // ---- V4 multiplicities for S4 ----

    #[test]
    fn s4_v4_multiplicities() {
        let pred = s4_spectral_prediction();
        let ms: Vec<i64> = pred.irreps.iter().map(|ir| ir.m_lambda).collect();
        // m = (d + 3*chi(2,2))/4
        // (4)->1, (3,1)->0, (2,2)->2, (2,1,1)->0, (1^4)->1
        assert_eq!(ms, vec![1, 0, 2, 0, 1]);
    }

    #[test]
    fn s4_total_coset_multiplicity() {
        let pred = s4_spectral_prediction();
        // sum m*d = |S4|/|V4| = 24/4 = 6
        assert_eq!(pred.total_coset_multiplicity, 6);
    }

    #[test]
    fn s4_centroid_22_irrep() {
        // For the (2,2) irrep: centroid = 3 - 3*0/2 = 3.
        // Actual eigenvalues ~1.268 and ~4.732 average to 3.
        let pred = s4_spectral_prediction();
        assert_eq!(pred.irreps[2].partition, vec![2, 2]);
        assert!(
            (pred.irreps[2].l_lambda_centroid - 3.0).abs() < 1e-12,
            "(2,2) centroid should be 3.0"
        );
    }

    // ---- S8 aggregate checks ----

    #[test]
    fn s8_burnside_sum_of_squares() {
        let pred = s8_spectral_prediction();
        assert_eq!(pred.sum_dim_squared, 40320, "sum d^2 should be 8! = 40320");
    }

    #[test]
    fn s8_total_coset_multiplicity() {
        let pred = s8_spectral_prediction();
        // sum m*d = |S8|/|R8| = 40320/8 = 5040
        assert_eq!(pred.total_coset_multiplicity, 5040);
    }

    #[test]
    fn s8_multiplicity_integrality_and_nonnegativity() {
        let pred = s8_spectral_prediction();
        for ir in &pred.irreps {
            let numerator = ir.dimension as i64 + 7 * ir.chi_coset_class;
            assert_eq!(
                numerator % 8,
                0,
                "R8 multiplicity not integral for {:?}: ({} + 7*{}) = {}",
                ir.partition,
                ir.dimension,
                ir.chi_coset_class,
                numerator
            );
            assert!(
                ir.m_lambda >= 0,
                "negative multiplicity for {:?}",
                ir.partition
            );
        }
    }

    // ---- Known individual S8 characters ----

    #[test]
    fn s8_trivial_rep_characters() {
        // Trivial rep (8): all characters = 1
        let trivial = vec![8];
        assert_eq!(murnaghan_nakayama(&trivial, &[2, 1, 1, 1, 1, 1, 1]), 1);
        assert_eq!(murnaghan_nakayama(&trivial, &[2, 2, 2, 2]), 1);
    }

    #[test]
    fn s8_sign_rep_characters() {
        // Sign rep (1^8): chi(mu) = sign of any permutation with cycle type mu.
        // Transposition is odd: sign = -1.
        // (2^4) is even (product of 4 transpositions): sign = +1.
        let sign_rep = vec![1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(murnaghan_nakayama(&sign_rep, &[2, 1, 1, 1, 1, 1, 1]), -1);
        assert_eq!(murnaghan_nakayama(&sign_rep, &[2, 2, 2, 2]), 1);
    }

    // ---- chi at identity recovers dimension ----

    #[test]
    fn chi_at_identity_equals_dimension() {
        let identity = vec![1; 4];
        for p in &partitions(4) {
            assert_eq!(
                murnaghan_nakayama(p, &identity),
                hook_length_dimension(p) as i64,
                "chi at identity should equal dimension for {:?}",
                p
            );
        }
    }

    // ---- Border-strip enumeration ----

    #[test]
    fn border_strip_single_row() {
        // [4], length 2: remove 2 from row 0, leaving [2]
        let bs = border_strips(&[4], 2);
        assert_eq!(bs, vec![(vec![2], 1)]);
    }

    #[test]
    fn border_strip_two_row() {
        // [3,1], length 2: only valid strip is top row 0, bottom row 0
        // c_b = 3 - 2 = 1 >= partition[1] = 1, yielding [1,1], sign +1
        let bs = border_strips(&[3, 1], 2);
        assert_eq!(bs, vec![(vec![1, 1], 1)]);
    }

    #[test]
    fn border_strip_multi_row() {
        // [2,2], length 2: two strips
        // (t=0,b=1): c_b=1, mu=[1,1], sign=-1
        // (t=1,b=1): c_b=0, mu=[2], sign=+1
        let bs = border_strips(&[2, 2], 2);
        assert_eq!(bs.len(), 2);
        assert!(bs.contains(&(vec![1, 1], -1)));
        assert!(bs.contains(&(vec![2], 1)));
    }
}
