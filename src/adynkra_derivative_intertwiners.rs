//! Exact fundamental-spinor Clebsch-Gordan intertwiners required by the
//! four-dimensional N=1 Adynkrafield derivative maps.
//!
//! The implementation uses the polynomial realization of the `sl(2)` irrep
//! `[n]` on homogeneous degree-`n` polynomials.  It constructs rational
//! embeddings and projections for `[n] tensor [1] = [n+1] + [n-1]`, verifies
//! equivariance and completeness, and then applies the construction to every
//! Lorentz irrep in the six genomes of arXiv:2407.09334.

use crate::adynkra_genome::{self, LorentzIrrep};
use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

type Rat = Ratio<i64>;
type Matrix = Vec<Vec<Rat>>;

#[derive(Debug, Clone, Serialize)]
pub struct FundamentalCouplingCheck {
    pub source_irrep: LorentzIrrep,
    pub derivative_chirality: &'static str,
    pub target_irreps: Vec<LorentzIrrep>,
    pub tensor_product_dimension: usize,
    pub target_dimensions_sum: usize,
    pub summands_checked: usize,
    pub equivariance_commutators_checked: usize,
    pub projection_embedding_identities: bool,
    pub cross_channels_zero: bool,
    pub complete: bool,
    pub equivariant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RepeatedIrrepCheck {
    pub genome: &'static str,
    pub total_level: u8,
    pub irrep: LorentzIrrep,
    pub bidegrees: Vec<[u8; 2]>,
    pub multiplicity: usize,
    pub left_derivative_selector: Vec<i8>,
    pub right_derivative_selector: Vec<i8>,
    pub selector_ranks: [usize; 2],
    pub combined_selector_rank: usize,
    pub channels_distinguished: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DerivativeIntertwinerReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_equations: &'static str,
    pub convention: &'static str,
    pub genome_irreps_checked: usize,
    pub fundamental_products_checked: usize,
    pub coupling_checks: Vec<FundamentalCouplingCheck>,
    pub repeated_irrep_sectors: Vec<RepeatedIrrepCheck>,
    pub repeated_sectors_expected: usize,
    pub repeated_sectors_found: usize,
    pub repeated_channels_distinguished: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

fn zeros(rows: usize, columns: usize) -> Matrix {
    vec![vec![Rat::zero(); columns]; rows]
}

fn identity(dimension: usize) -> Matrix {
    let mut result = zeros(dimension, dimension);
    for index in 0..dimension {
        result[index][index] = Rat::from_integer(1);
    }
    result
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.clone() + y.clone())
                .collect()
        })
        .collect()
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    assert_eq!(left[0].len(), right.len());
    let mut result = zeros(left.len(), right[0].len());
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot].is_zero() {
                continue;
            }
            for column in 0..right[0].len() {
                result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
            }
        }
    }
    result
}

fn rank(matrix: &Matrix) -> usize {
    let mut work = matrix.clone();
    let rows = work.len();
    let columns = work[0].len();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| !work[row][column].is_zero()) else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..rows {
            if row == pivot_row || work[row][column].is_zero() {
                continue;
            }
            let factor = work[row][column].clone();
            for next in column..columns {
                let subtraction = factor.clone() * work[pivot_row][next].clone();
                work[row][next] -= subtraction;
            }
        }
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    pivot_row
}

/// E, F, H on degree-n binary forms in the basis x^(n-k)y^k.
fn sl2_generators(n: usize) -> [Matrix; 3] {
    let dimension = n + 1;
    let mut e = zeros(dimension, dimension);
    let mut f = zeros(dimension, dimension);
    let mut h = zeros(dimension, dimension);
    for k in 0..=n {
        h[k][k] = Rat::from_integer(n as i64 - 2 * k as i64);
        if k > 0 {
            e[k - 1][k] = Rat::from_integer(k as i64);
        }
        if k < n {
            f[k + 1][k] = Rat::from_integer((n - k) as i64);
        }
    }
    [e, f, h]
}

fn tensor_generator(n: usize, generator_index: usize) -> Matrix {
    let source = sl2_generators(n);
    let fundamental = sl2_generators(1);
    let mut result = zeros(2 * (n + 1), 2 * (n + 1));
    for k in 0..=n {
        for spinor in 0..2 {
            let column = 2 * k + spinor;
            for output in 0..=n {
                result[2 * output + spinor][column] += source[generator_index][output][k].clone();
            }
            for output in 0..2 {
                result[2 * k + output][column] +=
                    fundamental[generator_index][output][spinor].clone();
            }
        }
    }
    result
}

/// Returns `(embedding, projection)` for the `[n+1]` channel.
fn upper_channel(n: usize) -> (Matrix, Matrix) {
    let mut embedding = zeros(2 * (n + 1), n + 2);
    let mut projection = zeros(n + 2, 2 * (n + 1));
    let denominator = Rat::from_integer((n + 1) as i64);
    for j in 0..=(n + 1) {
        if j <= n {
            embedding[2 * j][j] = Rat::from_integer((n + 1 - j) as i64) / denominator.clone();
        }
        if j > 0 {
            embedding[2 * (j - 1) + 1][j] = Rat::from_integer(j as i64) / denominator.clone();
        }
    }
    for k in 0..=n {
        projection[k][2 * k] = Rat::from_integer(1);
        projection[k + 1][2 * k + 1] = Rat::from_integer(1);
    }
    (embedding, projection)
}

/// Returns `(embedding, projection)` for the `[n-1]` channel.
fn lower_channel(n: usize) -> Option<(Matrix, Matrix)> {
    if n == 0 {
        return None;
    }
    let mut embedding = zeros(2 * (n + 1), n);
    let mut projection = zeros(n, 2 * (n + 1));
    let denominator = Rat::from_integer((n + 1) as i64);
    for j in 0..n {
        embedding[2 * j + 1][j] = Rat::from_integer(1) / denominator.clone();
        embedding[2 * (j + 1)][j] = Rat::from_integer(-1) / denominator.clone();
    }
    for k in 0..=n {
        if k > 0 {
            projection[k - 1][2 * k] = Rat::from_integer(-(k as i64));
        }
        if k < n {
            projection[k][2 * k + 1] = Rat::from_integer((n - k) as i64);
        }
    }
    Some((embedding, projection))
}

fn check_one_factor(n: usize) -> (bool, bool, bool, bool, usize) {
    let upper = upper_channel(n);
    let lower = lower_channel(n);
    let mut projection_embedding = multiply(&upper.1, &upper.0) == identity(n + 2);
    let mut cross_zero = true;
    let mut completeness = multiply(&upper.0, &upper.1);
    let mut equivariant = true;
    let mut commutators = 0;

    if let Some((lower_embedding, lower_projection)) = &lower {
        projection_embedding &= multiply(lower_projection, lower_embedding) == identity(n);
        cross_zero &= multiply(&upper.1, lower_embedding) == zeros(n + 2, n);
        cross_zero &= multiply(lower_projection, &upper.0) == zeros(n, n + 2);
        completeness = add(&completeness, &multiply(lower_embedding, lower_projection));
    }
    completeness = if completeness == identity(2 * (n + 1)) {
        completeness
    } else {
        return (projection_embedding, cross_zero, false, false, commutators);
    };
    let _ = completeness;

    for generator_index in 0..3 {
        let tensor = tensor_generator(n, generator_index);
        let upper_target = &sl2_generators(n + 1)[generator_index];
        equivariant &= multiply(&tensor, &upper.0) == multiply(&upper.0, upper_target);
        equivariant &= multiply(&upper.1, &tensor) == multiply(upper_target, &upper.1);
        commutators += 2;
        if let Some((lower_embedding, lower_projection)) = &lower {
            let lower_target = &sl2_generators(n - 1)[generator_index];
            equivariant &=
                multiply(&tensor, lower_embedding) == multiply(lower_embedding, lower_target);
            equivariant &=
                multiply(lower_projection, &tensor) == multiply(lower_target, lower_projection);
            commutators += 2;
        }
    }
    (
        projection_embedding,
        cross_zero,
        true,
        equivariant,
        commutators,
    )
}

fn coupling_check(irrep: LorentzIrrep, left_derivative: bool) -> FundamentalCouplingCheck {
    let active = if left_derivative {
        irrep.left as usize
    } else {
        irrep.right as usize
    };
    let (projection_embedding, cross_zero, complete, equivariant, commutators) =
        check_one_factor(active);
    let mut targets = vec![if left_derivative {
        LorentzIrrep::new(irrep.left + 1, irrep.right)
    } else {
        LorentzIrrep::new(irrep.left, irrep.right + 1)
    }];
    if active > 0 {
        targets.push(if left_derivative {
            LorentzIrrep::new(irrep.left - 1, irrep.right)
        } else {
            LorentzIrrep::new(irrep.left, irrep.right - 1)
        });
    }
    targets.sort_unstable();
    let tensor_product_dimension = 2 * irrep.dimension();
    let target_dimensions_sum = targets.iter().map(|target| target.dimension()).sum();
    FundamentalCouplingCheck {
        source_irrep: irrep,
        derivative_chirality: if left_derivative { "left" } else { "right" },
        target_irreps: targets,
        tensor_product_dimension,
        target_dimensions_sum,
        summands_checked: if active > 0 { 2 } else { 1 },
        equivariance_commutators_checked: commutators,
        projection_embedding_identities: projection_embedding,
        cross_channels_zero: cross_zero,
        complete: complete && tensor_product_dimension == target_dimensions_sum,
        equivariant,
    }
}

fn repeated_irrep_checks() -> Vec<RepeatedIrrepCheck> {
    let artifact = adynkra_genome::artifact();
    let mut checks = Vec::new();
    for genome in artifact.genomes {
        let mut groups = BTreeMap::<(u8, LorentzIrrep), Vec<[u8; 2]>>::new();
        for term in genome.terms {
            groups
                .entry((term.left_degree + term.right_degree, term.irrep))
                .or_default()
                .push([term.left_degree, term.right_degree]);
        }
        for ((total_level, irrep), mut bidegrees) in groups {
            if bidegrees.len() < 2 {
                continue;
            }
            bidegrees.sort_unstable();
            let left_selector: Vec<i8> = bidegrees
                .iter()
                .map(|degree| i8::from(*degree == [2, 0]))
                .collect();
            let right_selector: Vec<i8> = bidegrees
                .iter()
                .map(|degree| i8::from(*degree == [0, 2]))
                .collect();
            let selector_matrix = vec![
                left_selector
                    .iter()
                    .map(|&value| Rat::from_integer(value as i64))
                    .collect(),
                right_selector
                    .iter()
                    .map(|&value| Rat::from_integer(value as i64))
                    .collect(),
            ];
            let selector_ranks = [
                usize::from(left_selector.iter().any(|&value| value != 0)),
                usize::from(right_selector.iter().any(|&value| value != 0)),
            ];
            let combined_selector_rank = rank(&selector_matrix);
            let multiplicity = bidegrees.len();
            checks.push(RepeatedIrrepCheck {
                genome: genome.id,
                total_level,
                irrep,
                bidegrees,
                multiplicity,
                left_derivative_selector: left_selector,
                right_derivative_selector: right_selector,
                selector_ranks,
                combined_selector_rank,
                channels_distinguished: selector_ranks == [1, 1]
                    && combined_selector_rank == multiplicity,
            });
        }
    }
    checks
}

pub fn verify() -> DerivativeIntertwinerReport {
    let artifact = adynkra_genome::artifact();
    let irreps: BTreeSet<_> = artifact
        .genomes
        .iter()
        .flat_map(|genome| genome.terms.iter().map(|term| term.irrep))
        .collect();
    let mut coupling_checks = Vec::new();
    for irrep in irreps.iter().copied() {
        coupling_checks.push(coupling_check(irrep, true));
        coupling_checks.push(coupling_check(irrep, false));
    }
    let repeated_irrep_sectors = repeated_irrep_checks();
    let repeated_channels_distinguished = repeated_irrep_sectors
        .iter()
        .all(|check| check.channels_distinguished);
    let couplings_pass = coupling_checks.iter().all(|check| {
        check.projection_embedding_identities
            && check.cross_channels_zero
            && check.complete
            && check.equivariant
    });
    let repeated_sectors_found = repeated_irrep_sectors.len();
    DerivativeIntertwinerReport {
        schema_version: "adynkra-4d-n1-derivative-intertwiners-v1",
        source_arxiv: "2407.09334",
        source_equations: "2.25, 2.26, 2.45, 3.6-3.11",
        convention: "rational binary-form Clebsch-Gordan basis for SL(2)_L x SL(2)_R",
        genome_irreps_checked: irreps.len(),
        fundamental_products_checked: coupling_checks.len(),
        coupling_checks,
        repeated_irrep_sectors,
        repeated_sectors_expected: 3,
        repeated_sectors_found,
        repeated_channels_distinguished,
        boundary: "fundamental Clebsch-Gordan intertwiners and chirality-resolved repeated channels; component normalizations and prepotential gauge cohomology remain open",
        passed: couplings_pass && repeated_sectors_found == 3 && repeated_channels_distinguished,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fundamental_couplings_are_exact_through_the_published_range() {
        for n in 0..=3 {
            let (projection_embedding, cross_zero, complete, equivariant, _) = check_one_factor(n);
            assert!(projection_embedding);
            assert!(cross_zero);
            assert!(complete);
            assert!(equivariant);
        }
    }

    #[test]
    fn every_genome_irrep_has_complete_left_and_right_derivative_channels() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.fundamental_products_checked, 18);
    }

    #[test]
    fn all_repeated_level_irreps_are_resolved_by_bidegree_and_chirality() {
        let report = verify();
        assert_eq!(report.repeated_sectors_found, 3);
        assert!(report.repeated_channels_distinguished);
        assert!(report.repeated_irrep_sectors.iter().all(|check| {
            check.bidegrees == vec![[0, 2], [2, 0]] && check.combined_selector_rank == 2
        }));
    }
}
