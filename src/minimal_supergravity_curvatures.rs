//! Linearized old-minimal 4D N=1 curvature complex from *Superspace*,
//! Eqs. (3.1.22), (5.4.18), (7.4.2b), and (7.5.19).

use crate::supercovariant_derivative::{apply, Derivative, GaussianRational, Polynomial};
use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

const GRASSMANN_DIMENSION: usize = 16;
type Matrix = Vec<Vec<GaussianRational>>;

#[derive(Debug, Clone, Serialize)]
pub struct MomentumCohomologyCheck {
    pub momentum: [i64; 4],
    pub momentum_class: &'static str,
    pub chiral_compensator_dimension: usize,
    pub antichiral_compensator_dimension: usize,
    pub allowed_potential_dimension: usize,
    pub gauge_rank: usize,
    pub curvature_rank: usize,
    pub curvature_kernel_dimension: usize,
    pub middle_cohomology_dimension: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MinimalScalarCurvatureReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub source_equations: &'static str,
    pub compensator: &'static str,
    pub scalar_curvature: &'static str,
    pub vector_curvature: &'static str,
    pub bianchi_identities: [&'static str; 4],
    pub convention: &'static str,
    pub prepotential_basis_inputs: usize,
    pub compensator_prepotential_basis_inputs_per_chirality: usize,
    pub nonzero_curvature_images: usize,
    pub nonzero_conjugate_curvature_images: usize,
    pub chirality_relations_checked: usize,
    pub chirality_residuals: usize,
    pub antichirality_relations_checked: usize,
    pub antichirality_residuals: usize,
    pub gauge_invariance_relations_checked: usize,
    pub gauge_invariance_residuals: usize,
    pub uncompensated_mutation_residuals: usize,
    pub vector_curvature_components: usize,
    pub nonzero_vector_curvature_images: usize,
    pub vector_gauge_relations_checked: usize,
    pub vector_gauge_residuals: usize,
    pub scalar_vector_bianchi_relations_checked: usize,
    pub scalar_vector_bianchi_residuals: usize,
    pub conjugate_scalar_vector_bianchi_relations_checked: usize,
    pub conjugate_scalar_vector_bianchi_residuals: usize,
    pub weyl_vector_bianchi_relations_checked: usize,
    pub weyl_vector_bianchi_residuals: usize,
    pub conjugate_weyl_vector_bianchi_relations_checked: usize,
    pub conjugate_weyl_vector_bianchi_residuals: usize,
    pub unconstrained_compensator_mutation_residuals: usize,
    pub momentum_fiber_cohomology_checks: Vec<MomentumCohomologyCheck>,
    pub nonzero_momentum_fibers_exact_at_potential_term: bool,
    pub chiral_only_null_cohomology_dimension: usize,
    pub euler_lagrange_source_equations: &'static str,
    pub euler_lagrange_equation: &'static str,
    pub euler_lagrange_momentum_checks: Vec<MomentumCohomologyCheck>,
    pub euler_null_bosonic_cohomology_dimension: usize,
    pub euler_null_fermionic_cohomology_dimension: usize,
    pub euler_null_classes_detected_by_chiral_weyl_bosonic: usize,
    pub euler_null_classes_detected_by_chiral_weyl_fermionic: usize,
    pub euler_null_classes_detected_by_conjugate_weyl_bosonic: usize,
    pub euler_null_classes_detected_by_conjugate_weyl_fermionic: usize,
    pub euler_non_null_fibers_have_zero_cohomology: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> GaussianRational {
    Complex::new(
        Ratio::new(real, denominator),
        Ratio::new(imaginary, denominator),
    )
}

pub(crate) fn zero_h() -> Vec<Vec<Polynomial>> {
    vec![vec![Polynomial::default(); 2]; 2]
}

pub(crate) fn d_squared(polynomial: &Polynomial) -> Polynomial {
    apply(Derivative::Left(1), &apply(Derivative::Left(0), polynomial))
}

pub(crate) fn d_bar_squared(polynomial: &Polynomial) -> Polynomial {
    apply(
        Derivative::Right(1),
        &apply(Derivative::Right(0), polynomial),
    )
}

fn spinor_metric_upper(source: usize, target: usize) -> i64 {
    match (source, target) {
        (0, 1) => 1,
        (1, 0) => -1,
        _ => 0,
    }
}

fn spinor_metric_lower(first: usize, second: usize) -> i64 {
    match (first, second) {
        (0, 1) => -1,
        (1, 0) => 1,
        _ => 0,
    }
}

fn vector_pair(index: usize) -> (usize, usize) {
    (index / 2, index % 2)
}

pub(crate) fn raised_vector_component(vector: &[Polynomial], target: usize) -> Polynomial {
    let (target_alpha, target_dotted) = vector_pair(target);
    let mut result = Polynomial::default();
    for source in 0..4 {
        let (source_alpha, source_dotted) = vector_pair(source);
        let coefficient = spinor_metric_upper(source_alpha, target_alpha)
            * spinor_metric_upper(source_dotted, target_dotted);
        if coefficient != 0 {
            result = result.add(vector[source].clone().scale(gaussian(coefficient, 0, 1)));
        }
    }
    result
}

fn raised_spacetime_derivative(polynomial: Polynomial, target: usize) -> Polynomial {
    let (target_alpha, target_dotted) = vector_pair(target);
    let mut result = Polynomial::default();
    for source in 0..4 {
        let (source_alpha, source_dotted) = vector_pair(source);
        let coefficient = spinor_metric_upper(source_alpha, target_alpha)
            * spinor_metric_upper(source_dotted, target_dotted);
        if coefficient != 0 {
            result = result.add(
                polynomial
                    .clone()
                    .spacetime_derivative(source)
                    .scale(gaussian(coefficient, 0, 1)),
            );
        }
    }
    result
}

fn apply_raised_d(alpha: usize, polynomial: &Polynomial) -> Polynomial {
    let mut result = Polynomial::default();
    for source in 0..2 {
        let coefficient = spinor_metric_upper(source, alpha);
        if coefficient != 0 {
            result = result.add(apply(Derivative::Left(source), polynomial).scale(gaussian(
                coefficient,
                0,
                1,
            )));
        }
    }
    result
}

fn apply_raised_d_bar(dotted: usize, polynomial: &Polynomial) -> Polynomial {
    let mut result = Polynomial::default();
    for source in 0..2 {
        let coefficient = spinor_metric_upper(source, dotted);
        if coefficient != 0 {
            result = result.add(apply(Derivative::Right(source), polynomial).scale(gaussian(
                coefficient,
                0,
                1,
            )));
        }
    }
    result
}

fn levi_civita_lower(a: usize, b: usize, c: usize, d: usize) -> GaussianRational {
    let (alpha, dotted_alpha) = vector_pair(a);
    let (beta, dotted_beta) = vector_pair(b);
    let (gamma, dotted_gamma) = vector_pair(c);
    let (delta, dotted_delta) = vector_pair(d);
    let first = spinor_metric_lower(alpha, delta)
        * spinor_metric_lower(beta, gamma)
        * spinor_metric_lower(dotted_alpha, dotted_beta)
        * spinor_metric_lower(dotted_gamma, dotted_delta);
    let second = spinor_metric_lower(alpha, beta)
        * spinor_metric_lower(gamma, delta)
        * spinor_metric_lower(dotted_alpha, dotted_delta)
        * spinor_metric_lower(dotted_beta, dotted_gamma);
    gaussian(0, first - second, 1)
}

pub(crate) fn vector_curvature(
    h: &[Vec<Polynomial>],
    chi: &Polynomial,
    chi_bar: &Polynomial,
    output: usize,
) -> Polynomial {
    let h_vector = vec![
        h[0][0].clone(),
        h[0][1].clone(),
        h[1][0].clone(),
        h[1][1].clone(),
    ];

    // The source term is -(2/3) D^beta Dbar^2 D_beta H_a. In the internal
    // component basis used here its coefficient is +2/3. Complete gauge-image
    // and Bianchi tests below fix this sign relative to the other three terms.
    let mut result = Polynomial::default();
    for beta in 0..2 {
        let inner = apply(Derivative::Left(beta), &h_vector[output]);
        let term = apply_raised_d(beta, &d_bar_squared(&inner));
        result = result.add(term.scale(gaussian(2, 0, 3)));
    }

    // -(1/6) epsilon_abcd partial^b [D^gamma,Dbar^dot-gamma] H^d.
    for b in 0..4 {
        for c in 0..4 {
            let (gamma, dotted_gamma) = vector_pair(c);
            for d in 0..4 {
                let epsilon = levi_civita_lower(output, b, c, d);
                if epsilon == gaussian(0, 0, 1) {
                    continue;
                }
                let h_raised = raised_vector_component(&h_vector, d);
                let d_then_dbar =
                    apply_raised_d(gamma, &apply_raised_d_bar(dotted_gamma, &h_raised));
                let dbar_then_d =
                    apply_raised_d_bar(dotted_gamma, &apply_raised_d(gamma, &h_raised));
                let commutator = d_then_dbar.add(dbar_then_d.scale(gaussian(-1, 0, 1)));
                result = result.add(
                    raised_spacetime_derivative(commutator, b).scale(epsilon * gaussian(-1, 0, 6)),
                );
            }
        }
    }

    // -(1/3) partial_a partial_b H^b + i partial_a (chi - chi_bar).
    result = result.add(
        spacetime_divergence(h)
            .spacetime_derivative(output)
            .scale(gaussian(-1, 0, 3)),
    );
    result.add(
        chi.clone()
            .add(chi_bar.clone().scale(gaussian(-1, 0, 1)))
            .spacetime_derivative(output)
            .scale(gaussian(0, 1, 1)),
    )
}

pub(crate) fn spacetime_divergence(h: &[Vec<Polynomial>]) -> Polynomial {
    // Source bispinor convention for partial_a H^a, with epsilon^(01)=1.
    h[0][0]
        .clone()
        .spacetime_derivative(3)
        .add(
            h[0][1]
                .clone()
                .spacetime_derivative(2)
                .scale(gaussian(-1, 0, 1)),
        )
        .add(
            h[1][0]
                .clone()
                .spacetime_derivative(1)
                .scale(gaussian(-1, 0, 1)),
        )
        .add(h[1][1].clone().spacetime_derivative(0))
}

pub(crate) fn scalar_curvature(h: &[Vec<Polynomial>], chi_bar: &Polynomial) -> Polynomial {
    let interior = chi_bar
        .clone()
        .add(spacetime_divergence(h).scale(gaussian(0, -1, 3)));
    d_bar_squared(&interior)
}

pub(crate) fn conjugate_scalar_curvature(h: &[Vec<Polynomial>], chi: &Polynomial) -> Polynomial {
    let interior = chi
        .clone()
        .add(spacetime_divergence(h).scale(gaussian(0, 1, 3)));
    d_squared(&interior)
}

fn raised_d(alpha_parameter: usize, input: &Polynomial) -> Polynomial {
    match alpha_parameter {
        0 => apply(Derivative::Left(1), input).scale(gaussian(-1, 0, 1)),
        1 => apply(Derivative::Left(0), input),
        _ => panic!("undotted spinor index exceeds two components"),
    }
}

pub(crate) fn gauge_image_l(alpha: usize, mask: u8) -> (Vec<Vec<Polynomial>>, Polynomial) {
    let mut h = zero_h();
    let input = Polynomial::basis(mask);
    for dotted in 0..2 {
        h[alpha][dotted] = apply(Derivative::Right(dotted), &input).scale(gaussian(-1, 0, 1));
    }
    let chi = d_bar_squared(&raised_d(alpha, &input)).scale(gaussian(1, 0, 3));
    (h, chi)
}

fn raised_d_bar(dotted_parameter: usize, input: &Polynomial) -> Polynomial {
    match dotted_parameter {
        0 => apply(Derivative::Right(1), input).scale(gaussian(-1, 0, 1)),
        1 => apply(Derivative::Right(0), input),
        _ => panic!("dotted spinor index exceeds two components"),
    }
}

pub(crate) fn gauge_image_l_bar(
    dotted: usize,
    mask: u8,
    include_compensator: bool,
) -> (Vec<Vec<Polynomial>>, Polynomial) {
    let mut h = zero_h();
    let input = Polynomial::basis(mask);
    for alpha in 0..2 {
        h[alpha][dotted] = apply(Derivative::Left(alpha), &input);
    }
    let chi_bar = if include_compensator {
        d_squared(&raised_d_bar(dotted, &input)).scale(gaussian(1, 0, 3))
    } else {
        Polynomial::default()
    };
    (h, chi_bar)
}

fn scalar_vector_bianchi_residual(
    h: &[Vec<Polynomial>],
    chi: &Polynomial,
    chi_bar: &Polynomial,
    alpha: usize,
) -> Polynomial {
    let mut divergence = Polynomial::default();
    for dotted in 0..2 {
        divergence = divergence.add(apply_raised_d_bar(
            dotted,
            &vector_curvature(h, chi, chi_bar, 2 * alpha + dotted),
        ));
    }
    // In the implemented epsilon convention, Eq. (5.4.18) is
    // Dbar^dot-alpha G_(alpha dot-alpha) = -D_alpha R.
    divergence.add(apply(
        Derivative::Left(alpha),
        &scalar_curvature(h, chi_bar),
    ))
}

fn conjugate_scalar_vector_bianchi_residual(
    h: &[Vec<Polynomial>],
    chi: &Polynomial,
    chi_bar: &Polynomial,
    dotted: usize,
) -> Polynomial {
    let mut divergence = Polynomial::default();
    for alpha in 0..2 {
        divergence = divergence.add(apply_raised_d(
            alpha,
            &vector_curvature(h, chi, chi_bar, 2 * alpha + dotted),
        ));
    }
    // Conjugate identity in the implemented epsilon convention:
    // D^alpha G_(alpha dot-alpha) = Dbar_dot-alpha Rbar.
    divergence.add(
        apply(
            Derivative::Right(dotted),
            &conjugate_scalar_curvature(h, chi),
        )
        .scale(gaussian(-1, 0, 1)),
    )
}

fn partial_lower_upper_dotted(polynomial: &Polynomial, alpha: usize, dotted: usize) -> Polynomial {
    let mut result = Polynomial::default();
    for source_dotted in 0..2 {
        let coefficient = spinor_metric_upper(source_dotted, dotted);
        if coefficient != 0 {
            result = result.add(
                polynomial
                    .clone()
                    .spacetime_derivative(2 * alpha + source_dotted)
                    .scale(gaussian(coefficient, 0, 1)),
            );
        }
    }
    result
}

fn partial_upper_undotted_lower(
    polynomial: &Polynomial,
    alpha: usize,
    dotted: usize,
) -> Polynomial {
    let mut result = Polynomial::default();
    for source_alpha in 0..2 {
        let coefficient = spinor_metric_upper(source_alpha, alpha);
        if coefficient != 0 {
            result = result.add(
                polynomial
                    .clone()
                    .spacetime_derivative(2 * source_alpha + dotted)
                    .scale(gaussian(coefficient, 0, 1)),
            );
        }
    }
    result
}

fn weyl_vector_bianchi_residual(h: &[Vec<Polynomial>], beta: usize, gamma: usize) -> Polynomial {
    let mut divergence = Polynomial::default();
    for alpha in 0..2 {
        divergence = divergence.add(apply_raised_d(
            alpha,
            &crate::prepotential_curvature::weyl_component(h, alpha + beta + gamma),
        ));
    }

    let zero = Polynomial::default();
    let g: Vec<_> = (0..4)
        .map(|output| vector_curvature(h, &zero, &zero, output))
        .collect();
    let mut symmetrized_gradient = Polynomial::default();
    for (derivative_alpha, g_alpha) in [(beta, gamma), (gamma, beta)] {
        for dotted in 0..2 {
            symmetrized_gradient = symmetrized_gradient.add(partial_lower_upper_dotted(
                &g[2 * g_alpha + dotted],
                derivative_alpha,
                dotted,
            ));
        }
    }
    // The source coefficient i/2 becomes -i/2 in the implemented epsilon
    // convention.
    divergence.add(symmetrized_gradient.scale(gaussian(0, 1, 2)))
}

fn conjugate_weyl_vector_bianchi_terms(
    h: &[Vec<Polynomial>],
    dotted_beta: usize,
    dotted_gamma: usize,
) -> (Polynomial, Polynomial) {
    let mut divergence = Polynomial::default();
    for dotted_alpha in 0..2 {
        divergence = divergence.add(apply_raised_d_bar(
            dotted_alpha,
            &crate::prepotential_curvature::conjugate_weyl_component(
                h,
                dotted_alpha + dotted_beta + dotted_gamma,
            ),
        ));
    }

    let zero = Polynomial::default();
    let g: Vec<_> = (0..4)
        .map(|output| vector_curvature(h, &zero, &zero, output))
        .collect();
    let mut symmetrized_gradient = Polynomial::default();
    for (derivative_dotted, g_dotted) in [(dotted_beta, dotted_gamma), (dotted_gamma, dotted_beta)]
    {
        for alpha in 0..2 {
            symmetrized_gradient = symmetrized_gradient.add(partial_upper_undotted_lower(
                &g[2 * alpha + g_dotted],
                alpha,
                derivative_dotted,
            ));
        }
    }
    (divergence, symmetrized_gradient)
}

fn conjugate_weyl_vector_bianchi_residual(
    h: &[Vec<Polynomial>],
    dotted_beta: usize,
    dotted_gamma: usize,
) -> Polynomial {
    let (divergence, symmetrized_gradient) =
        conjugate_weyl_vector_bianchi_terms(h, dotted_beta, dotted_gamma);
    divergence.add(symmetrized_gradient.scale(gaussian(0, -1, 2)))
}

fn evaluate_polynomial(polynomial: &Polynomial, momentum: [i64; 4]) -> Vec<GaussianRational> {
    let mut result = vec![gaussian(0, 0, 1); GRASSMANN_DIMENSION];
    for (monomial, source_coefficient) in &polynomial.0 {
        let mut coefficient = source_coefficient.clone();
        for (index, &power) in monomial.spacetime_derivatives.iter().enumerate() {
            for _ in 0..power {
                coefficient *= Ratio::from_integer(momentum[index]);
            }
        }
        result[monomial.grassmann_mask as usize] += coefficient;
    }
    result
}

fn columns_to_matrix(columns: &[Vec<GaussianRational>]) -> Matrix {
    if columns.is_empty() {
        return Vec::new();
    }
    let rows = columns[0].len();
    let mut matrix = vec![vec![gaussian(0, 0, 1); columns.len()]; rows];
    for (column, values) in columns.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            matrix[row][column] = value.clone();
        }
    }
    matrix
}

fn pivot_columns(matrix: &Matrix) -> Vec<usize> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let mut work = matrix.clone();
    let rows = work.len();
    let columns = work[0].len();
    let mut pivot_row = 0;
    let mut pivots = Vec::new();
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| work[row][column] != gaussian(0, 0, 1))
        else {
            continue;
        };
        work.swap(pivot_row, found);
        let pivot = work[pivot_row][column].clone();
        for value in &mut work[pivot_row][column..] {
            *value /= pivot.clone();
        }
        for row in 0..rows {
            if row == pivot_row || work[row][column] == gaussian(0, 0, 1) {
                continue;
            }
            let factor = work[row][column].clone();
            for next in column..columns {
                let subtraction = factor.clone() * work[pivot_row][next].clone();
                work[row][next] -= subtraction;
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    pivots
}

fn matrix_rank(matrix: &Matrix) -> usize {
    pivot_columns(matrix).len()
}

fn append_evaluated(
    output: &mut Vec<GaussianRational>,
    polynomial: &Polynomial,
    momentum: [i64; 4],
) {
    output.extend(evaluate_polynomial(polynomial, momentum));
}

fn ambient_potential_vector(
    h: &[Vec<Polynomial>],
    chi: &Polynomial,
    chi_bar: &Polynomial,
    momentum: [i64; 4],
) -> Vec<GaussianRational> {
    let mut result = Vec::with_capacity(96);
    for row in h.iter().take(2) {
        for component in row.iter().take(2) {
            append_evaluated(&mut result, component, momentum);
        }
    }
    append_evaluated(&mut result, chi, momentum);
    append_evaluated(&mut result, chi_bar, momentum);
    result
}

fn curvature_vector(
    h: &[Vec<Polynomial>],
    chi: &Polynomial,
    chi_bar: &Polynomial,
    momentum: [i64; 4],
    include_weyl: bool,
    include_conjugate_weyl: bool,
) -> Vec<GaussianRational> {
    let mut result = Vec::with_capacity(224);
    append_evaluated(&mut result, &scalar_curvature(h, chi_bar), momentum);
    append_evaluated(&mut result, &conjugate_scalar_curvature(h, chi), momentum);
    for output in 0..4 {
        append_evaluated(
            &mut result,
            &vector_curvature(h, chi, chi_bar, output),
            momentum,
        );
    }
    if include_weyl {
        for component in 0..4 {
            append_evaluated(
                &mut result,
                &crate::prepotential_curvature::weyl_component(h, component),
                momentum,
            );
        }
    }
    if include_conjugate_weyl {
        for component in 0..4 {
            append_evaluated(
                &mut result,
                &crate::prepotential_curvature::conjugate_weyl_component(h, component),
                momentum,
            );
        }
    }
    result
}

fn momentum_cohomology(
    momentum: [i64; 4],
    include_weyl: bool,
    include_conjugate_weyl: bool,
) -> MomentumCohomologyCheck {
    momentum_cohomology_filtered(momentum, include_weyl, include_conjugate_weyl, None)
}

fn momentum_cohomology_filtered(
    momentum: [i64; 4],
    include_weyl: bool,
    include_conjugate_weyl: bool,
    potential_parity: Option<u32>,
) -> MomentumCohomologyCheck {
    let chiral_columns: Vec<_> = (0..GRASSMANN_DIMENSION as u8)
        .map(|mask| evaluate_polynomial(&d_bar_squared(&Polynomial::basis(mask)), momentum))
        .collect();
    let antichiral_columns: Vec<_> = (0..GRASSMANN_DIMENSION as u8)
        .map(|mask| evaluate_polynomial(&d_squared(&Polynomial::basis(mask)), momentum))
        .collect();
    let chiral_pivots = pivot_columns(&columns_to_matrix(&chiral_columns));
    let antichiral_pivots = pivot_columns(&columns_to_matrix(&antichiral_columns));

    let mut gauge_columns = Vec::with_capacity(64);
    for chirality in 0..2 {
        for spinor in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                if potential_parity.is_some_and(|parity| parity != (mask.count_ones() + 1) % 2) {
                    continue;
                }
                let (h, chi, chi_bar) = if chirality == 0 {
                    let (h, chi) = gauge_image_l(spinor, mask);
                    (h, chi, Polynomial::default())
                } else {
                    let (h, chi_bar) = gauge_image_l_bar(spinor, mask, true);
                    (h, Polynomial::default(), chi_bar)
                };
                gauge_columns.push(ambient_potential_vector(&h, &chi, &chi_bar, momentum));
            }
        }
    }
    let gauge_rank = matrix_rank(&columns_to_matrix(&gauge_columns));

    let mut curvature_columns = Vec::with_capacity(72);
    for component in 0..4 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            if potential_parity.is_some_and(|parity| parity != mask.count_ones() % 2) {
                continue;
            }
            let mut h = zero_h();
            h[component / 2][component % 2] = Polynomial::basis(mask);
            curvature_columns.push(curvature_vector(
                &h,
                &Polynomial::default(),
                &Polynomial::default(),
                momentum,
                include_weyl,
                include_conjugate_weyl,
            ));
        }
    }
    for &mask in &chiral_pivots {
        if potential_parity.is_some_and(|parity| parity != (mask as u8).count_ones() % 2) {
            continue;
        }
        curvature_columns.push(curvature_vector(
            &zero_h(),
            &d_bar_squared(&Polynomial::basis(mask as u8)),
            &Polynomial::default(),
            momentum,
            include_weyl,
            include_conjugate_weyl,
        ));
    }
    for &mask in &antichiral_pivots {
        if potential_parity.is_some_and(|parity| parity != (mask as u8).count_ones() % 2) {
            continue;
        }
        curvature_columns.push(curvature_vector(
            &zero_h(),
            &Polynomial::default(),
            &d_squared(&Polynomial::basis(mask as u8)),
            momentum,
            include_weyl,
            include_conjugate_weyl,
        ));
    }
    let allowed_potential_dimension = curvature_columns.len();
    let curvature_rank = matrix_rank(&columns_to_matrix(&curvature_columns));
    let curvature_kernel_dimension = allowed_potential_dimension - curvature_rank;
    let middle_cohomology_dimension = curvature_kernel_dimension
        .checked_sub(gauge_rank)
        .expect("the exact gauge image must lie in the curvature kernel");
    let determinant = momentum[0] * momentum[3] - momentum[1] * momentum[2];
    let momentum_class = if momentum == [0; 4] {
        "zero"
    } else if determinant == 0 {
        "null"
    } else {
        "non-null"
    };
    MomentumCohomologyCheck {
        momentum,
        momentum_class,
        chiral_compensator_dimension: chiral_pivots.len(),
        antichiral_compensator_dimension: antichiral_pivots.len(),
        allowed_potential_dimension,
        gauge_rank,
        curvature_rank,
        curvature_kernel_dimension,
        middle_cohomology_dimension,
    }
}

pub fn verify() -> MinimalScalarCurvatureReport {
    let mut nonzero_curvature_images = 0;
    let mut nonzero_conjugate_curvature_images = 0;
    let mut chirality_relations_checked = 0;
    let mut chirality_residuals = 0;
    let mut antichirality_relations_checked = 0;
    let mut antichirality_residuals = 0;
    for input_kind in 0..5 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            let mut h = zero_h();
            let mut chi = Polynomial::default();
            let mut chi_bar = Polynomial::default();
            if input_kind < 4 {
                h[input_kind / 2][input_kind % 2] = Polynomial::basis(mask);
            } else {
                // The old-minimal compensator is chiral. Parameterize the
                // conjugate pair by chiral projection of an unconstrained
                // 16-monomial prepotential basis.
                chi = d_bar_squared(&Polynomial::basis(mask));
                chi_bar = d_squared(&Polynomial::basis(mask));
            }
            let r = scalar_curvature(&h, &chi_bar);
            let r_bar = conjugate_scalar_curvature(&h, &chi);
            if !r.0.is_empty() {
                nonzero_curvature_images += 1;
            }
            if !r_bar.0.is_empty() {
                nonzero_conjugate_curvature_images += 1;
            }
            for derivative in [Derivative::Right(0), Derivative::Right(1)] {
                chirality_relations_checked += 1;
                if !apply(derivative, &r).0.is_empty() {
                    chirality_residuals += 1;
                }
            }
            for derivative in [Derivative::Left(0), Derivative::Left(1)] {
                antichirality_relations_checked += 1;
                if !apply(derivative, &r_bar).0.is_empty() {
                    antichirality_residuals += 1;
                }
            }
        }
    }

    let mut gauge_invariance_relations_checked = 0;
    let mut gauge_invariance_residuals = 0;
    let mut uncompensated_mutation_residuals = 0;
    let mut vector_gauge_relations_checked = 0;
    let mut vector_gauge_residuals = 0;
    for chirality in 0..2 {
        for spinor in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                let (h, chi, chi_bar) = if chirality == 0 {
                    let (h, chi) = gauge_image_l(spinor, mask);
                    (h, chi, Polynomial::default())
                } else {
                    let (h, chi_bar) = gauge_image_l_bar(spinor, mask, true);
                    (h, Polynomial::default(), chi_bar)
                };
                gauge_invariance_relations_checked += 2;
                if !scalar_curvature(&h, &chi_bar).0.is_empty() {
                    gauge_invariance_residuals += 1;
                }
                if !conjugate_scalar_curvature(&h, &chi).0.is_empty() {
                    gauge_invariance_residuals += 1;
                }
                if chirality == 0 {
                    let (h_uncompensated, _) = gauge_image_l(spinor, mask);
                    if !conjugate_scalar_curvature(&h_uncompensated, &Polynomial::default())
                        .0
                        .is_empty()
                    {
                        uncompensated_mutation_residuals += 1;
                    }
                } else {
                    let (h_uncompensated, zero_chi) = gauge_image_l_bar(spinor, mask, false);
                    if !scalar_curvature(&h_uncompensated, &zero_chi).0.is_empty() {
                        uncompensated_mutation_residuals += 1;
                    }
                }
                for output in 0..4 {
                    vector_gauge_relations_checked += 1;
                    if !vector_curvature(&h, &chi, &chi_bar, output).0.is_empty() {
                        vector_gauge_residuals += 1;
                    }
                }
            }
        }
    }

    let mut nonzero_vector_curvature_images = 0;
    for input_kind in 0..6 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            let mut h = zero_h();
            let mut chi = Polynomial::default();
            let mut chi_bar = Polynomial::default();
            if input_kind < 4 {
                h[input_kind / 2][input_kind % 2] = Polynomial::basis(mask);
            } else if input_kind == 4 {
                chi = d_bar_squared(&Polynomial::basis(mask));
            } else {
                chi_bar = d_squared(&Polynomial::basis(mask));
            }
            for output in 0..4 {
                if !vector_curvature(&h, &chi, &chi_bar, output).0.is_empty() {
                    nonzero_vector_curvature_images += 1;
                }
            }
        }
    }

    let mut scalar_vector_bianchi_relations_checked = 0;
    let mut scalar_vector_bianchi_residuals = 0;
    let mut conjugate_scalar_vector_bianchi_relations_checked = 0;
    let mut conjugate_scalar_vector_bianchi_residuals = 0;
    let mut unconstrained_compensator_mutation_residuals = 0;
    for input_kind in 0..6 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            let mut h = zero_h();
            let mut chi = Polynomial::default();
            let mut chi_bar = Polynomial::default();
            if input_kind < 4 {
                h[input_kind / 2][input_kind % 2] = Polynomial::basis(mask);
            } else if input_kind == 4 {
                chi = d_bar_squared(&Polynomial::basis(mask));
            } else {
                chi_bar = d_squared(&Polynomial::basis(mask));
            }
            for alpha in 0..2 {
                scalar_vector_bianchi_relations_checked += 1;
                if !scalar_vector_bianchi_residual(&h, &chi, &chi_bar, alpha)
                    .0
                    .is_empty()
                {
                    scalar_vector_bianchi_residuals += 1;
                }
            }
            for dotted in 0..2 {
                conjugate_scalar_vector_bianchi_relations_checked += 1;
                if !conjugate_scalar_vector_bianchi_residual(&h, &chi, &chi_bar, dotted)
                    .0
                    .is_empty()
                {
                    conjugate_scalar_vector_bianchi_residuals += 1;
                }
            }

            if input_kind >= 4 {
                let (raw_chi, raw_chi_bar) = if input_kind == 4 {
                    (Polynomial::basis(mask), Polynomial::default())
                } else {
                    (Polynomial::default(), Polynomial::basis(mask))
                };
                for alpha in 0..2 {
                    if !scalar_vector_bianchi_residual(&zero_h(), &raw_chi, &raw_chi_bar, alpha)
                        .0
                        .is_empty()
                    {
                        unconstrained_compensator_mutation_residuals += 1;
                    }
                }
            }
        }
    }

    let mut weyl_vector_bianchi_relations_checked = 0;
    let mut weyl_vector_bianchi_residuals = 0;
    let mut conjugate_weyl_vector_bianchi_relations_checked = 0;
    let mut conjugate_weyl_vector_bianchi_residuals = 0;
    for input_kind in 0..4 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            let mut h = zero_h();
            h[input_kind / 2][input_kind % 2] = Polynomial::basis(mask);
            for (beta, gamma) in [(0, 0), (0, 1), (1, 1)] {
                weyl_vector_bianchi_relations_checked += 1;
                if !weyl_vector_bianchi_residual(&h, beta, gamma).0.is_empty() {
                    weyl_vector_bianchi_residuals += 1;
                }
            }
            for (dotted_beta, dotted_gamma) in [(0, 0), (0, 1), (1, 1)] {
                conjugate_weyl_vector_bianchi_relations_checked += 1;
                if !conjugate_weyl_vector_bianchi_residual(&h, dotted_beta, dotted_gamma)
                    .0
                    .is_empty()
                {
                    conjugate_weyl_vector_bianchi_residuals += 1;
                }
            }
        }
    }

    let momenta = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 2, 3, 6],
        [0, 1, 1, 0],
        [1, 2, 3, 5],
        [2, -1, 4, 3],
    ];
    let momentum_fiber_cohomology_checks: Vec<_> = momenta
        .into_iter()
        .map(|momentum| momentum_cohomology(momentum, true, true))
        .collect();
    let nonzero_momentum_fibers_exact_at_potential_term = momentum_fiber_cohomology_checks
        .iter()
        .filter(|check| check.momentum_class != "zero")
        .all(|check| check.middle_cohomology_dimension == 0);
    let chiral_only_null_cohomology_dimension =
        momentum_cohomology([1, 0, 0, 0], true, false).middle_cohomology_dimension;
    let euler_lagrange_momentum_checks: Vec<_> = momenta
        .into_iter()
        .map(|momentum| momentum_cohomology(momentum, false, false))
        .collect();
    let euler_null_bosonic_cohomology_dimension =
        momentum_cohomology_filtered([1, 0, 0, 0], false, false, Some(0))
            .middle_cohomology_dimension;
    let euler_null_fermionic_cohomology_dimension =
        momentum_cohomology_filtered([1, 0, 0, 0], false, false, Some(1))
            .middle_cohomology_dimension;
    let euler_null_classes_detected_by_chiral_weyl_bosonic = euler_null_bosonic_cohomology_dimension
        - momentum_cohomology_filtered([1, 0, 0, 0], true, false, Some(0))
            .middle_cohomology_dimension;
    let euler_null_classes_detected_by_chiral_weyl_fermionic =
        euler_null_fermionic_cohomology_dimension
            - momentum_cohomology_filtered([1, 0, 0, 0], true, false, Some(1))
                .middle_cohomology_dimension;
    let euler_null_classes_detected_by_conjugate_weyl_bosonic =
        euler_null_bosonic_cohomology_dimension
            - momentum_cohomology_filtered([1, 0, 0, 0], false, true, Some(0))
                .middle_cohomology_dimension;
    let euler_null_classes_detected_by_conjugate_weyl_fermionic =
        euler_null_fermionic_cohomology_dimension
            - momentum_cohomology_filtered([1, 0, 0, 0], false, true, Some(1))
                .middle_cohomology_dimension;
    let euler_non_null_fibers_have_zero_cohomology = euler_lagrange_momentum_checks
        .iter()
        .filter(|check| check.momentum_class == "non-null")
        .all(|check| check.middle_cohomology_dimension == 0);
    let euler_null_fibers_have_dimension_four = euler_lagrange_momentum_checks
        .iter()
        .filter(|check| check.momentum_class == "null")
        .all(|check| check.middle_cohomology_dimension == 4);

    let passed = nonzero_curvature_images > 0
        && chirality_relations_checked == 160
        && chirality_residuals == 0
        && antichirality_relations_checked == 160
        && antichirality_residuals == 0
        && gauge_invariance_relations_checked == 128
        && gauge_invariance_residuals == 0
        && uncompensated_mutation_residuals == 48
        && vector_gauge_relations_checked == 256
        && vector_gauge_residuals == 0
        && scalar_vector_bianchi_relations_checked == 192
        && scalar_vector_bianchi_residuals == 0
        && conjugate_scalar_vector_bianchi_relations_checked == 192
        && conjugate_scalar_vector_bianchi_residuals == 0
        && weyl_vector_bianchi_relations_checked == 192
        && weyl_vector_bianchi_residuals == 0
        && conjugate_weyl_vector_bianchi_relations_checked == 192
        && conjugate_weyl_vector_bianchi_residuals == 0
        && unconstrained_compensator_mutation_residuals == 52
        && nonzero_momentum_fibers_exact_at_potential_term
        && chiral_only_null_cohomology_dimension == 2
        && euler_null_bosonic_cohomology_dimension == 2
        && euler_null_fermionic_cohomology_dimension == 2
        && euler_null_classes_detected_by_chiral_weyl_bosonic == 1
        && euler_null_classes_detected_by_chiral_weyl_fermionic == 1
        && euler_null_classes_detected_by_conjugate_weyl_bosonic == 1
        && euler_null_classes_detected_by_conjugate_weyl_fermionic == 1
        && euler_non_null_fibers_have_zero_cohomology
        && euler_null_fibers_have_dimension_four
        && momentum_fiber_cohomology_checks[0].middle_cohomology_dimension == 26;
    MinimalScalarCurvatureReport {
        schema_version: "adynkra-4d-n1-old-minimal-curvature-complex-v4",
        source: "hep-th/0108200",
        source_equations: "3.1.22, 5.4.18, 5.5.45, 5.5.48, 7.4.2b, and 7.5.19",
        compensator: "the chiral conjugate pair obtained by linearizing delta phi^3 = Dbar^2 D^alpha L_alpha",
        scalar_curvature: "R = Dbar^2 (chi_bar - (i/3) partial_a H^a), with its conjugate Rbar",
        vector_curvature: "the four-term G_a expression in Superspace Eq. (7.5.19)",
        bianchi_identities: [
            "Dbar^dot-alpha G_(alpha dot-alpha) = -D_alpha R",
            "D^alpha G_(alpha dot-alpha) = Dbar_dot-alpha Rbar",
            "D^alpha W_(alpha beta gamma) = -(i/2) partial_(beta^dot-alpha G_(gamma) dot-alpha)",
            "Dbar^dot-alpha Wbar_(dot-alpha dot-beta dot-gamma) = (i/2) partial^alpha_(dot-beta G_(alpha dot-gamma))",
        ],
        convention: "phi = 1 + chi, epsilon^(01) = 1, D^2 = D_1 D_0, and Dbar^2 = Dbar_1 Dbar_0; displayed Bianchi signs are the translated internal convention",
        prepotential_basis_inputs: 64,
        compensator_prepotential_basis_inputs_per_chirality: 16,
        nonzero_curvature_images,
        nonzero_conjugate_curvature_images,
        chirality_relations_checked,
        chirality_residuals,
        antichirality_relations_checked,
        antichirality_residuals,
        gauge_invariance_relations_checked,
        gauge_invariance_residuals,
        uncompensated_mutation_residuals,
        vector_curvature_components: 4,
        nonzero_vector_curvature_images,
        vector_gauge_relations_checked,
        vector_gauge_residuals,
        scalar_vector_bianchi_relations_checked,
        scalar_vector_bianchi_residuals,
        conjugate_scalar_vector_bianchi_relations_checked,
        conjugate_scalar_vector_bianchi_residuals,
        weyl_vector_bianchi_relations_checked,
        weyl_vector_bianchi_residuals,
        conjugate_weyl_vector_bianchi_relations_checked,
        conjugate_weyl_vector_bianchi_residuals,
        unconstrained_compensator_mutation_residuals,
        momentum_fiber_cohomology_checks,
        nonzero_momentum_fibers_exact_at_potential_term,
        chiral_only_null_cohomology_dimension,
        euler_lagrange_source_equations: "Superspace Eqs. (5.5.45) and (5.5.48), with vanishing cosmological and matter sources",
        euler_lagrange_equation: "G_(alpha dot-alpha) = 0, R = 0, and Rbar = 0",
        euler_lagrange_momentum_checks,
        euler_null_bosonic_cohomology_dimension,
        euler_null_fermionic_cohomology_dimension,
        euler_null_classes_detected_by_chiral_weyl_bosonic,
        euler_null_classes_detected_by_chiral_weyl_fermionic,
        euler_null_classes_detected_by_conjugate_weyl_bosonic,
        euler_null_classes_detected_by_conjugate_weyl_fermionic,
        euler_non_null_fibers_have_zero_cohomology,
        boundary: "linearized old-minimal curvature complex, known source-free Euler-Lagrange operator, and sampled momentum-fiber cohomology; polynomial-module cohomology, helicity identification, and nonlinear equations remain open",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_curvature_pair_has_the_required_chirality() {
        let report = verify();
        assert_eq!(report.chirality_relations_checked, 160);
        assert_eq!(report.chirality_residuals, 0);
        assert_eq!(report.antichirality_relations_checked, 160);
        assert_eq!(report.antichirality_residuals, 0);
    }

    #[test]
    fn scalar_and_vector_curvatures_annihilate_the_complete_gauge_image() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.gauge_invariance_relations_checked, 128);
        assert_eq!(report.gauge_invariance_residuals, 0);
        assert_eq!(report.vector_gauge_relations_checked, 256);
        assert_eq!(report.vector_gauge_residuals, 0);
        assert_eq!(report.uncompensated_mutation_residuals, 48);
    }

    #[test]
    fn scalar_vector_bianchi_identity_holds_on_the_complete_input_basis() {
        let report = verify();
        assert_eq!(report.scalar_vector_bianchi_relations_checked, 192);
        assert_eq!(report.scalar_vector_bianchi_residuals, 0);
        assert_eq!(
            report.conjugate_scalar_vector_bianchi_relations_checked,
            192
        );
        assert_eq!(report.conjugate_scalar_vector_bianchi_residuals, 0);
    }

    #[test]
    fn weyl_vector_bianchi_identity_holds_on_the_complete_prepotential_basis() {
        let report = verify();
        assert_eq!(report.weyl_vector_bianchi_relations_checked, 192);
        assert_eq!(report.weyl_vector_bianchi_residuals, 0);
        assert_eq!(report.conjugate_weyl_vector_bianchi_relations_checked, 192);
        assert_eq!(report.conjugate_weyl_vector_bianchi_residuals, 0);
    }

    #[test]
    fn dropping_compensator_chirality_is_detected() {
        let report = verify();
        assert_eq!(report.unconstrained_compensator_mutation_residuals, 52);
    }

    #[test]
    fn momentum_fiber_cohomology_separates_zero_null_and_non_null_momenta() {
        let report = verify();
        assert!(report.nonzero_momentum_fibers_exact_at_potential_term);
        assert_eq!(report.chiral_only_null_cohomology_dimension, 2);
        assert_eq!(
            report
                .momentum_fiber_cohomology_checks
                .iter()
                .map(|check| (check.momentum_class, check.middle_cohomology_dimension))
                .collect::<Vec<_>>(),
            vec![
                ("zero", 26),
                ("null", 0),
                ("null", 0),
                ("null", 0),
                ("non-null", 0),
                ("non-null", 0),
                ("non-null", 0),
            ]
        );
    }
    #[test]
    fn known_euler_lagrange_operator_has_the_massless_parity_split() {
        let report = verify();
        assert_eq!(report.euler_null_bosonic_cohomology_dimension, 2);
        assert_eq!(report.euler_null_fermionic_cohomology_dimension, 2);
        assert_eq!(report.euler_null_classes_detected_by_chiral_weyl_bosonic, 1);
        assert_eq!(
            report.euler_null_classes_detected_by_chiral_weyl_fermionic,
            1
        );
        assert_eq!(
            report.euler_null_classes_detected_by_conjugate_weyl_bosonic,
            1
        );
        assert_eq!(
            report.euler_null_classes_detected_by_conjugate_weyl_fermionic,
            1
        );
        assert!(report.euler_non_null_fibers_have_zero_cohomology);
        assert!(report
            .euler_lagrange_momentum_checks
            .iter()
            .filter(|check| check.momentum_class == "null")
            .all(|check| check.middle_cohomology_dimension == 4));
    }
}
