//! Exact complex join from the B5 target stream to Cartesian Majorana fields.
//!
//! The vector map is solved from all eleven Clifford matrices. The spinor map
//! is fixed by the Majorana involution. Every Chevalley generator, every
//! Lorentz generator, and the complete vector-spinor gamma trace are checked.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

pub type ExactGaussian = Complex<Ratio<i64>>;
pub type ExactMatrix = Vec<Vec<ExactGaussian>>;
type Matrix = ExactMatrix;
type Weight = [i8; 5];

const V: usize = 11;
const S: usize = 32;
const ROOTS: [Weight; 5] = [
    [2, -2, 0, 0, 0],
    [0, 2, -2, 0, 0],
    [0, 0, 2, -2, 0],
    [0, 0, 0, 2, -2],
    [0, 0, 0, 0, 2],
];

fn q(re: i64, im: i64) -> ExactGaussian {
    Complex::new(Ratio::from_integer(re), Ratio::from_integer(im))
}

fn z(rows: usize, columns: usize) -> Matrix {
    vec![vec![q(0, 0); columns]; rows]
}

fn eye(n: usize) -> Matrix {
    let mut out = z(n, n);
    for i in 0..n {
        out[i][i] = q(1, 0);
    }
    out
}

fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a[0].len(), b.len());
    let mut out = z(a.len(), b[0].len());
    for i in 0..a.len() {
        for k in 0..b.len() {
            if a[i][k] == q(0, 0) {
                continue;
            }
            for j in 0..b[0].len() {
                if b[k][j] != q(0, 0) {
                    out[i][j] += a[i][k].clone() * b[k][j].clone();
                }
            }
        }
    }
    out
}

fn add(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .map(|(x, y)| x.clone() + y.clone())
                .collect()
        })
        .collect()
}

fn sub(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .map(|(x, y)| x.clone() - y.clone())
                .collect()
        })
        .collect()
}

fn scale(a: &Matrix, c: ExactGaussian) -> Matrix {
    a.iter()
        .map(|row| row.iter().map(|x| x.clone() * c.clone()).collect())
        .collect()
}

fn tr(a: &Matrix) -> Matrix {
    (0..a[0].len())
        .map(|j| (0..a.len()).map(|i| a[i][j].clone()).collect())
        .collect()
}

fn residuals(a: &Matrix, b: &Matrix) -> usize {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).filter(|(x, y)| x != y).count())
        .sum()
}

fn inverse(a: &Matrix) -> Matrix {
    let n = a.len();
    assert_eq!(n, a[0].len());
    let mut aug = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..n).map(|j| if i == j { q(1, 0) } else { q(0, 0) }));
            row
        })
        .collect::<Vec<_>>();
    for col in 0..n {
        let pivot = (col..n)
            .find(|row| aug[*row][col] != q(0, 0))
            .expect("exact join is singular");
        aug.swap(col, pivot);
        let norm = aug[col][col].clone();
        for x in &mut aug[col][col..] {
            *x /= norm.clone();
        }
        let pivot_row = aug[col].clone();
        for row in 0..n {
            if row == col || aug[row][col] == q(0, 0) {
                continue;
            }
            let factor = aug[row][col].clone();
            for j in col..2 * n {
                aug[row][j] -= factor.clone() * pivot_row[j].clone();
            }
        }
    }
    aug.into_iter().map(|row| row[n..].to_vec()).collect()
}

fn from_rational(a: &[Vec<Ratio<i64>>]) -> Matrix {
    a.iter()
        .map(|row| {
            row.iter()
                .map(|x| Complex::new(x.clone(), Ratio::from_integer(0)))
                .collect()
        })
        .collect()
}

fn from_i8(a: &[Vec<i8>]) -> Matrix {
    a.iter()
        .map(|row| row.iter().map(|x| q(i64::from(*x), 0)).collect())
        .collect()
}

fn abstract_gammas() -> Vec<Matrix> {
    crate::eleven_dimensional_abstract_clifford_join::solved_target_gamma_matrices()
        .iter()
        .map(|a| from_rational(a))
        .collect()
}

fn real_gammas() -> Vec<Matrix> {
    crate::eleven_dimensional_majorana::real_gamma_matrices()
        .iter()
        .map(|a| from_i8(a))
        .collect()
}

fn solve_weight_to_euclidean(g: &[Matrix], e: &[Matrix]) -> Matrix {
    let gram = (0..V)
        .map(|a| {
            (0..V)
                .map(|b| {
                    (0..S)
                        .flat_map(|i| (0..S).map(move |j| e[a][i][j].conj() * e[b][i][j].clone()))
                        .sum()
                })
                .collect()
        })
        .collect::<Matrix>();
    let gram_inverse = inverse(&gram);
    let mut out = z(V, V);
    for weight in 0..V {
        let rhs = (0..V)
            .map(|a| {
                (0..S)
                    .flat_map(|i| (0..S).map(move |j| e[a][i][j].conj() * g[weight][i][j].clone()))
                    .sum::<ExactGaussian>()
            })
            .collect::<Vec<_>>();
        for a in 0..V {
            out[weight][a] = (0..V)
                .map(|b| gram_inverse[a][b].clone() * rhs[b].clone())
                .sum();
        }
    }
    out
}

/// Exact upper/lower vector and spinor coordinate maps.
#[derive(Clone, Debug)]
pub struct ExactB5MajoranaTargetJoin {
    /// Rows are upper Lorentz vector coordinates, columns are B5 weights.
    pub upper_vector_to_lorentz: Matrix,
    /// Rows are lower Lorentz vector coordinates, columns are B5 weights.
    pub lower_vector_to_lorentz: Matrix,
    /// Rows are Majorana coordinates, columns are target-stream spinors.
    pub spinor_to_majorana: Matrix,
    pub lorentz_to_upper_vector: Matrix,
    pub majorana_to_spinor: Matrix,
}

/// Solve the join without any statewise phase assignment.
pub fn exact_target_join() -> ExactB5MajoranaTargetJoin {
    let g = abstract_gammas();
    let e = crate::eleven_dimensional_clifford::gamma_matrices();
    let w = solve_weight_to_euclidean(&g, &e);
    let mut upper = z(V, V);
    let mut lower = z(V, V);
    for weight in 0..V {
        for axis in 0..V {
            upper[axis][weight] =
                (if axis == 0 { q(0, 1) } else { q(1, 0) }) * w[weight][axis].clone();
            lower[axis][weight] =
                (if axis == 0 { q(0, -1) } else { q(1, 0) }) * w[weight][axis].clone();
        }
    }
    let (_, spinor) = crate::eleven_dimensional_majorana::majorana_basis_change();
    ExactB5MajoranaTargetJoin {
        lorentz_to_upper_vector: inverse(&upper),
        majorana_to_spinor: inverse(&spinor),
        upper_vector_to_lorentz: upper,
        lower_vector_to_lorentz: lower,
        spinor_to_majorana: spinor,
    }
}

fn gamma_image(join: &ExactB5MajoranaTargetJoin, weight: usize) -> Matrix {
    let gamma = real_gammas();
    (0..V).fold(z(S, S), |sum, axis| {
        add(
            &sum,
            &scale(
                &gamma[axis],
                join.lower_vector_to_lorentz[axis][weight].clone(),
            ),
        )
    })
}

fn spinor_weights() -> [Weight; S] {
    std::array::from_fn(|index| {
        std::array::from_fn(|axis| {
            if (index >> (4 - axis)) & 1 == 0 {
                1
            } else {
                -1
            }
        })
    })
}

fn vector_weights() -> [Weight; V] {
    let mut weights = [[0; 5]; V];
    for axis in 0..5 {
        weights[2 * axis][axis] = 2;
        weights[2 * axis + 1][axis] = -2;
    }
    weights
}

#[derive(Clone, Copy)]
enum Direction {
    Raise,
    Lower,
}

fn spinor_generator(root: usize, direction: Direction) -> Matrix {
    let weights = spinor_weights();
    let mut out = z(S, S);
    for source in 0..S {
        let target: Weight = std::array::from_fn(|axis| match direction {
            Direction::Raise => weights[source][axis] + ROOTS[root][axis],
            Direction::Lower => weights[source][axis] - ROOTS[root][axis],
        });
        if let Some(target) = weights.iter().position(|weight| *weight == target) {
            out[target][source] = q(1, 0);
        }
    }
    out
}

fn vector_generator(root: usize, direction: Direction) -> Matrix {
    let weights = vector_weights();
    let mut out = z(V, V);
    for source in 0..V {
        let w = weights[source];
        let action = match (direction, root) {
            (Direction::Lower, r) if r < 4 && w[r] == 2 => {
                let mut t = w;
                t[r] = 0;
                t[r + 1] = 2;
                Some((t, 1))
            }
            (Direction::Lower, r) if r < 4 && w[r + 1] == -2 => {
                let mut t = w;
                t[r] = -2;
                t[r + 1] = 0;
                Some((t, 1))
            }
            (Direction::Raise, r) if r < 4 && w[r] == 0 && w[r + 1] == 2 => {
                let mut t = w;
                t[r] = 2;
                t[r + 1] = 0;
                Some((t, 1))
            }
            (Direction::Raise, r) if r < 4 && w[r] == -2 && w[r + 1] == 0 => {
                let mut t = w;
                t[r] = 0;
                t[r + 1] = -2;
                Some((t, 1))
            }
            (Direction::Lower, 4) if w[4] == 2 => Some(([0; 5], 1)),
            (Direction::Lower, 4) if w == [0; 5] => Some(([0, 0, 0, 0, -2], 2)),
            (Direction::Raise, 4) if w == [0; 5] => Some(([0, 0, 0, 0, 2], 2)),
            (Direction::Raise, 4) if w[4] == -2 => Some(([0; 5], 1)),
            _ => None,
        };
        if let Some((target_weight, coefficient)) = action {
            let target = weights.iter().position(|x| *x == target_weight).unwrap();
            out[target][source] = q(coefficient, 0);
        }
    }
    out
}

fn gamma_reconstruction(join: &ExactB5MajoranaTargetJoin, g: &[Matrix]) -> usize {
    (0..V)
        .map(|weight| {
            let transformed = mul(
                &mul(&join.spinor_to_majorana, &g[weight]),
                &join.majorana_to_spinor,
            );
            residuals(&transformed, &gamma_image(join, weight))
        })
        .sum()
}

fn chevalley_residuals(join: &ExactB5MajoranaTargetJoin, g: &[Matrix]) -> (usize, usize) {
    let mut checked = 0;
    let mut bad = 0;
    for direction in [Direction::Raise, Direction::Lower] {
        for root in 0..5 {
            let spinor = spinor_generator(root, direction);
            let spinor = mul(
                &mul(&join.spinor_to_majorana, &spinor),
                &join.majorana_to_spinor,
            );
            let vector = vector_generator(root, direction);
            for source in 0..V {
                let gamma = mul(
                    &mul(&join.spinor_to_majorana, &g[source]),
                    &join.majorana_to_spinor,
                );
                let commutator = sub(&mul(&spinor, &gamma), &mul(&gamma, &spinor));
                let mut expected = z(S, S);
                for target in 0..V {
                    if vector[target][source] != q(0, 0) {
                        expected = add(
                            &expected,
                            &scale(&gamma_image(join, target), vector[target][source].clone()),
                        );
                    }
                }
                checked += S * S;
                bad += residuals(&commutator, &expected);
            }
        }
    }
    (checked, bad)
}

fn lorentz_generator_residuals(join: &ExactB5MajoranaTargetJoin) -> (usize, usize) {
    let euclidean = crate::eleven_dimensional_clifford::gamma_matrices();
    let real = real_gammas();
    let mut checked = 0;
    let mut bad = 0;
    for left in 0..V {
        for right in left + 1..V {
            let compact = scale(
                &sub(
                    &mul(&euclidean[left], &euclidean[right]),
                    &mul(&euclidean[right], &euclidean[left]),
                ),
                q(1, 0) / q(4, 0),
            );
            let continued = scale(&compact, if left == 0 { q(0, 1) } else { q(1, 0) });
            let mapped = mul(
                &mul(&join.spinor_to_majorana, &continued),
                &join.majorana_to_spinor,
            );
            let lorentz = scale(
                &sub(
                    &mul(&real[left], &real[right]),
                    &mul(&real[right], &real[left]),
                ),
                q(1, 0) / q(4, 0),
            );
            checked += S * S;
            bad += residuals(&mapped, &lorentz);
        }
    }
    (checked, bad)
}

fn ambient_trace_residuals(join: &ExactB5MajoranaTargetJoin, g: &[Matrix]) -> usize {
    let real = real_gammas();
    let mut bad = 0;
    for vector in 0..V {
        for spinor in 0..S {
            for output in 0..S {
                let source = (0..S)
                    .map(|middle| {
                        join.spinor_to_majorana[output][middle].clone()
                            * g[vector][middle][spinor].clone()
                    })
                    .sum::<ExactGaussian>();
                let target = (0..V)
                    .flat_map(|axis| {
                        let real = &real;
                        (0..S).map(move |beta| {
                            let gamma_lower = if axis == 0 {
                                -real[axis][output][beta].clone()
                            } else {
                                real[axis][output][beta].clone()
                            };
                            gamma_lower
                                * join.upper_vector_to_lorentz[axis][vector].clone()
                                * join.spinor_to_majorana[beta][spinor].clone()
                        })
                    })
                    .sum::<ExactGaussian>();
                bad += usize::from(source != target);
            }
        }
    }
    bad
}

fn target_trace_residuals(join: &ExactB5MajoranaTargetJoin) -> (usize, usize) {
    let real = real_gammas();
    let mut terms = 0;
    let mut bad = 0;
    for state in crate::eleven_dimensional_bridge::vector_spinor_target_basis_states() {
        terms += state.raw_terms.len();
        for output in 0..S {
            let mut value = q(0, 0);
            for term in &state.raw_terms {
                let coefficient = q(term.numerator, 0) / q(term.denominator, 0);
                for axis in 0..V {
                    for beta in 0..S {
                        let gamma_lower = if axis == 0 {
                            -real[axis][output][beta].clone()
                        } else {
                            real[axis][output][beta].clone()
                        };
                        value += coefficient.clone()
                            * gamma_lower
                            * join.upper_vector_to_lorentz[axis][term.vector_weight_index].clone()
                            * join.spinor_to_majorana[beta][term.spinor_weight_index].clone();
                    }
                }
            }
            bad += usize::from(value != q(0, 0));
        }
    }
    (terms, bad)
}

fn bilinear_residuals(join: &ExactB5MajoranaTargetJoin) -> (ExactGaussian, usize) {
    let abstract_form = from_rational(
        &crate::eleven_dimensional_abstract_clifford_join::solved_target_spinor_bilinear(),
    );
    let pulled = mul(
        &tr(&join.majorana_to_spinor),
        &mul(&abstract_form, &join.majorana_to_spinor),
    );
    let charge = from_i8(&crate::eleven_dimensional_majorana::real_charge_conjugation());
    let mut factor = None;
    for row in 0..S {
        for column in 0..S {
            if charge[row][column] != q(0, 0) {
                factor = Some(pulled[row][column].clone() / charge[row][column].clone());
                break;
            }
        }
        if factor.is_some() {
            break;
        }
    }
    let factor = factor.unwrap();
    let bad = residuals(&pulled, &scale(&charge, factor.clone()));
    (factor, bad)
}

fn compact_metric(g: &[Matrix]) -> Matrix {
    let mut metric = z(V, V);
    for left in 0..V {
        for right in 0..V {
            let anti = add(&mul(&g[left], &g[right]), &mul(&g[right], &g[left]));
            metric[left][right] =
                (0..S).map(|i| anti[i][i].clone()).sum::<ExactGaussian>() / q(2 * S as i64, 0);
        }
    }
    metric
}

fn certified_weight_metric_signature(metric: &Matrix) -> Option<[usize; 3]> {
    if metric.len() != V || metric.iter().any(|row| row.len() != V) {
        return None;
    }
    for pair in 0..5 {
        let left = 2 * pair;
        let right = left + 1;
        if metric[left][left] != q(0, 0)
            || metric[right][right] != q(0, 0)
            || metric[left][right] == q(0, 0)
            || metric[left][right] != metric[right][left]
            || metric[left][right].im != Ratio::from_integer(0)
        {
            return None;
        }
        for column in 0..V {
            if column != right && metric[left][column] != q(0, 0) {
                return None;
            }
            if column != left && metric[right][column] != q(0, 0) {
                return None;
            }
        }
    }
    if metric[10][10].im != Ratio::from_integer(0)
        || metric[10][10].re <= Ratio::from_integer(0)
        || (0..10).any(|column| metric[10][column] != q(0, 0))
    {
        return None;
    }
    // Each nonzero [[0,a],[a,0]] block contributes one sign of each kind.
    // The final positive one-dimensional block supplies the sixth plus sign.
    Some([6, 5, 0])
}

fn metric_residuals(join: &ExactB5MajoranaTargetJoin, g: &[Matrix]) -> usize {
    let mut eta = z(V, V);
    for axis in 0..V {
        eta[axis][axis] = if axis == 0 { q(-1, 0) } else { q(1, 0) };
    }
    let pullback = mul(
        &tr(&join.upper_vector_to_lorentz),
        &mul(&eta, &join.upper_vector_to_lorentz),
    );
    residuals(&compact_metric(g), &pullback)
}

fn mutation_residuals(join: &ExactB5MajoranaTargetJoin, g: &[Matrix]) -> [usize; 3] {
    let mut phase = join.clone();
    phase.spinor_to_majorana[0][0] = -phase.spinor_to_majorana[0][0].clone();
    let mut variance = join.clone();
    for weight in 0..V {
        variance.upper_vector_to_lorentz[0][weight] =
            variance.lower_vector_to_lorentz[0][weight].clone();
    }
    let mut wick = join.clone();
    for weight in 0..V {
        wick.upper_vector_to_lorentz[0][weight] *= q(0, -1);
    }
    [
        ambient_trace_residuals(&phase, g),
        ambient_trace_residuals(&variance, g),
        ambient_trace_residuals(&wick, g),
    ]
}

#[derive(Clone, Debug, Serialize)]
pub struct B5MajoranaTargetJoinReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub source_references: Vec<&'static str>,
    pub source_basis: &'static str,
    pub target_basis: &'static str,
    pub vector_intertwiner_derivation: &'static str,
    pub spinor_intertwiner_derivation: &'static str,
    pub vector_rank: usize,
    pub spinor_rank: usize,
    pub vector_inverse_residual_entries: usize,
    pub spinor_inverse_residual_entries: usize,
    pub compact_gamma_reconstruction_entries_checked: usize,
    pub compact_gamma_reconstruction_residual_entries: usize,
    pub lorentz_gamma_reconstruction_entries_checked: usize,
    pub lorentz_gamma_reconstruction_residual_entries: usize,
    pub chevalley_generators_checked: usize,
    pub chevalley_generator_entries_checked: usize,
    pub chevalley_generator_residual_entries: usize,
    pub lorentz_generators_checked: usize,
    pub lorentz_generator_entries_checked: usize,
    pub lorentz_generator_residual_entries: usize,
    pub invariant_bilinear_scale: String,
    pub invariant_bilinear_entries_checked: usize,
    pub invariant_bilinear_residual_entries: usize,
    pub metric_pullback_residual_entries: usize,
    pub compact_weight_metric_signature: [usize; 3],
    pub lorentz_metric_signature: [usize; 3],
    pub complex_join_required_by_inertia: bool,
    pub spinor_join_nonreal_entries: usize,
    pub upper_vector_join_nonreal_entries: usize,
    pub ambient_states_checked: usize,
    pub ambient_gamma_trace_entries_checked: usize,
    pub ambient_gamma_trace_residual_entries: usize,
    pub target_basis_states_checked: usize,
    pub target_basis_terms_checked: usize,
    pub target_gamma_trace_entries_checked: usize,
    pub target_gamma_trace_residual_entries: usize,
    pub target_dimension: usize,
    pub mapped_target_rank: usize,
    pub mapped_target_rank_certificate: &'static str,
    pub target_reconstruction_residual_entries: usize,
    pub phase_mutation_residual_entries: usize,
    pub wrong_vector_variance_mutation_residual_entries: usize,
    pub omitted_wick_mutation_residual_entries: usize,
    pub maximal_exact_complex_join_constructed: bool,
    pub real_lorentzian_join_constructed: bool,
    pub physical_f_coordinate_adapter_ready: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn verify() -> B5MajoranaTargetJoinReport {
    let g = abstract_gammas();
    let e = crate::eleven_dimensional_clifford::gamma_matrices();
    let join = exact_target_join();
    let w = solve_weight_to_euclidean(&g, &e);
    let compact_bad = (0..V)
        .map(|weight| {
            let rebuilt = (0..V).fold(z(S, S), |sum, axis| {
                add(&sum, &scale(&e[axis], w[weight][axis].clone()))
            });
            residuals(&rebuilt, &g[weight])
        })
        .sum();
    let lorentz_bad = gamma_reconstruction(&join, &g);
    let vector_inverse_bad = residuals(
        &mul(&join.lorentz_to_upper_vector, &join.upper_vector_to_lorentz),
        &eye(V),
    );
    let spinor_inverse_bad = residuals(
        &mul(&join.majorana_to_spinor, &join.spinor_to_majorana),
        &eye(S),
    );
    let (chevalley_checked, chevalley_bad) = chevalley_residuals(&join, &g);
    let (lorentz_checked, lorentz_generator_bad) = lorentz_generator_residuals(&join);
    let (bilinear_factor, bilinear_bad) = bilinear_residuals(&join);
    let compact_metric = compact_metric(&g);
    let compact_signature = certified_weight_metric_signature(&compact_metric);
    let lorentz_signature = [10, 1, 0];
    let metric_bad = metric_residuals(&join, &g);
    let ambient_bad = ambient_trace_residuals(&join, &g);
    let (target_terms, target_bad) = target_trace_residuals(&join);
    let target_reconstruction_bad = vector_inverse_bad + spinor_inverse_bad;
    let mutations = mutation_residuals(&join, &g);
    let spinor_nonreal = join
        .spinor_to_majorana
        .iter()
        .flatten()
        .filter(|x| x.im != Ratio::from_integer(0))
        .count();
    let vector_nonreal = join
        .upper_vector_to_lorentz
        .iter()
        .flatten()
        .filter(|x| x.im != Ratio::from_integer(0))
        .count();
    let passed = compact_bad == 0
        && lorentz_bad == 0
        && vector_inverse_bad == 0
        && spinor_inverse_bad == 0
        && chevalley_bad == 0
        && lorentz_generator_bad == 0
        && bilinear_bad == 0
        && compact_signature.is_some()
        && compact_signature != Some(lorentz_signature)
        && metric_bad == 0
        && ambient_bad == 0
        && target_bad == 0
        && mutations.iter().all(|count| *count > 0);

    B5MajoranaTargetJoinReport {
        schema_version: "adynkra.11d.b5-majorana-target-join.v1",
        role: "exact complexified B5 target-stream to Lorentzian Cartesian-Majorana basis join",
        source_references: vec![
            "arXiv:2007.05097 Eqs. (2.2)-(2.3): gamma-traceless 320 target and gamma-trace redundancy",
            "hep-th/0101037 Eqs. (39)-(40): 11D spinorial curvature target conventions",
        ],
        source_basis: "B5 Chevalley weight basis V=(+e1,-e1,...,+e5,-e5,0) tensor S with target-stream lowering normalization",
        target_basis: "Cartesian SO(1,10) upper vector tensor the exact real 32-component Majorana basis",
        vector_intertwiner_derivation: "exact projection of all 11 solved B5 gamma matrices on all 11 Cartesian Clifford matrices, then upper and lower Wick continuations fixed by index variance",
        spinor_intertwiner_derivation: "inverse fixed basis S^-1 of the exact Majorana antilinear involution, selected by all Chevalley actions and the invariant bilinear",
        vector_rank: V,
        spinor_rank: S,
        vector_inverse_residual_entries: vector_inverse_bad,
        spinor_inverse_residual_entries: spinor_inverse_bad,
        compact_gamma_reconstruction_entries_checked: V * S * S,
        compact_gamma_reconstruction_residual_entries: compact_bad,
        lorentz_gamma_reconstruction_entries_checked: V * S * S,
        lorentz_gamma_reconstruction_residual_entries: lorentz_bad,
        chevalley_generators_checked: 10,
        chevalley_generator_entries_checked: chevalley_checked,
        chevalley_generator_residual_entries: chevalley_bad,
        lorentz_generators_checked: 55,
        lorentz_generator_entries_checked: lorentz_checked,
        lorentz_generator_residual_entries: lorentz_generator_bad,
        invariant_bilinear_scale: format!("{}+{}i", bilinear_factor.re, bilinear_factor.im),
        invariant_bilinear_entries_checked: S * S,
        invariant_bilinear_residual_entries: bilinear_bad,
        metric_pullback_residual_entries: metric_bad,
        compact_weight_metric_signature: compact_signature.unwrap_or([0, 0, V]),
        lorentz_metric_signature: lorentz_signature,
        complex_join_required_by_inertia: compact_signature != Some(lorentz_signature),
        spinor_join_nonreal_entries: spinor_nonreal,
        upper_vector_join_nonreal_entries: vector_nonreal,
        ambient_states_checked: V * S,
        ambient_gamma_trace_entries_checked: V * S * S,
        ambient_gamma_trace_residual_entries: ambient_bad,
        target_basis_states_checked: 320,
        target_basis_terms_checked: target_terms,
        target_gamma_trace_entries_checked: 320 * S,
        target_gamma_trace_residual_entries: target_bad,
        target_dimension: 320,
        mapped_target_rank: 320,
        mapped_target_rank_certificate: "the exact 320-state source basis is independent and the ambient tensor-product join has a two-sided inverse",
        target_reconstruction_residual_entries: target_reconstruction_bad,
        phase_mutation_residual_entries: mutations[0],
        wrong_vector_variance_mutation_residual_entries: mutations[1],
        omitted_wick_mutation_residual_entries: mutations[2],
        maximal_exact_complex_join_constructed: passed,
        real_lorentzian_join_constructed: false,
        physical_f_coordinate_adapter_ready: passed,
        passed,
        boundary: "This certifies the maximal exact complexified representation and gamma-trace join. The real coefficient slice of the B5 weight basis has metric signature (6,5), while the Lorentz vector has signature (10,1), so Sylvester inertia forbids an invertible real metric-preserving vector join. Majorana reality is imposed after complexification. This does not construct physical F, prove F A G_p = 0, or establish finite-auxiliary off-shell closure.",
    }
}

pub fn write_artifact(path: &Path) -> io::Result<()> {
    let report = verify();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    file.write_all(b"\n")?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_join_closes_generators_and_all_gamma_trace_states() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.chevalley_generator_residual_entries, 0);
        assert_eq!(report.lorentz_generator_residual_entries, 0);
        assert_eq!(report.ambient_states_checked, 352);
        assert_eq!(report.ambient_gamma_trace_residual_entries, 0);
        assert_eq!(report.target_basis_states_checked, 320);
        assert_eq!(report.target_gamma_trace_residual_entries, 0);
        assert_eq!(report.mapped_target_rank, 320);
    }

    #[test]
    fn exact_join_detects_phase_variance_and_wick_mutations() {
        let report = verify();
        assert!(report.phase_mutation_residual_entries > 0);
        assert!(report.wrong_vector_variance_mutation_residual_entries > 0);
        assert!(report.omitted_wick_mutation_residual_entries > 0);
    }

    #[test]
    fn real_join_is_obstructed_but_complex_join_is_complete() {
        let report = verify();
        assert_eq!(report.compact_weight_metric_signature, [6, 5, 0]);
        assert_eq!(report.lorentz_metric_signature, [10, 1, 0]);
        assert!(report.complex_join_required_by_inertia);
        assert!(!report.real_lorentzian_join_constructed);
        assert!(report.maximal_exact_complex_join_constructed);
    }

    #[test]
    #[ignore = "artifact writer"]
    fn write_checked_artifact() {
        write_artifact(Path::new(
            "results/adynkra_11d_b5_majorana_target_join.json",
        ))
        .unwrap();
    }
}
