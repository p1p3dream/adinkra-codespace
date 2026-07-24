//! Linearized chiral super-Weyl curvature of the 4D N=1 supergravity
//! prepotential.  The formula is Eq. (5.2.5) of *Superspace* and its gauge map
//! is Eq. (5.2.7), reproduced as Eq. (2.21) of arXiv:2407.09334.

use crate::supercovariant_derivative::{Derivative, GaussianRational, Polynomial, apply};
use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

const GRASSMANN_DIMENSION: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct PrepotentialCurvatureReport {
    pub schema_version: &'static str,
    pub sources: [&'static str; 2],
    pub source_equations: [&'static str; 2],
    pub curvature: &'static str,
    pub convention: &'static str,
    pub symmetric_weyl_components: usize,
    pub prepotential_basis_inputs: usize,
    pub gauge_parameter_basis_inputs: usize,
    pub nonzero_curvature_images: usize,
    pub nonzero_conjugate_curvature_images: usize,
    pub maximum_curvature_derivative_order: usize,
    pub chirality_relations_checked: usize,
    pub chirality_residuals: usize,
    pub antichirality_relations_checked: usize,
    pub antichirality_residuals: usize,
    pub gauge_invariance_relations_checked: usize,
    pub gauge_invariance_residuals: usize,
    pub conjugate_gauge_invariance_relations_checked: usize,
    pub conjugate_gauge_invariance_residuals: usize,
    pub boundary: &'static str,
    pub passed: bool,
}

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> GaussianRational {
    Complex::new(
        Ratio::new(real, denominator),
        Ratio::new(imaginary, denominator),
    )
}

fn minus(polynomial: Polynomial) -> Polynomial {
    polynomial.scale(gaussian(-1, 0, 1))
}

fn zero_h() -> Vec<Vec<Polynomial>> {
    vec![vec![Polynomial::default(); 2]; 2]
}

fn permutations(values: [usize; 3]) -> [[usize; 3]; 6] {
    let [a, b, c] = values;
    [
        [a, b, c],
        [a, c, b],
        [b, a, c],
        [b, c, a],
        [c, a, b],
        [c, b, a],
    ]
}

fn bar_d_squared(polynomial: &Polynomial) -> Polynomial {
    // Superspace Eq. (3.4.10): Dbar^2 = (1/2) Dbar^dot-alpha Dbar_dot-alpha.
    apply(
        Derivative::Right(1),
        &apply(Derivative::Right(0), polynomial),
    )
}

fn d_squared(polynomial: &Polynomial) -> Polynomial {
    apply(Derivative::Left(1), &apply(Derivative::Left(0), polynomial))
}

fn contracted_vector_derivative(beta: usize, h_gamma: &[Polynomial]) -> Polynomial {
    // partial_{beta dot-beta} H_gamma^{dot-beta}, with
    // H^dot0 = H_dot1 and H^dot1 = -H_dot0.
    h_gamma[1]
        .clone()
        .spacetime_derivative(2 * beta)
        .add(minus(h_gamma[0].clone().spacetime_derivative(2 * beta + 1)))
}

pub(crate) fn weyl_component(h: &[Vec<Polynomial>], number_of_one_indices: usize) -> Polynomial {
    let indices = match number_of_one_indices {
        0 => [0, 0, 0],
        1 => [0, 0, 1],
        2 => [0, 1, 1],
        3 => [1, 1, 1],
        _ => panic!("a symmetric rank-three spinor has four components"),
    };
    let mut result = Polynomial::default();
    for [alpha, beta, gamma] in permutations(indices) {
        let contracted = contracted_vector_derivative(beta, &h[gamma]);
        let d_applied = apply(Derivative::Left(alpha), &contracted);
        result = result.add(bar_d_squared(&d_applied));
    }
    // Eq. (5.2.5): -i / 3! times the unnormalized symmetrization.
    result.scale(gaussian(0, -1, 6))
}

fn contracted_conjugate_vector_derivative(
    dotted_beta: usize,
    dotted_gamma: usize,
    h: &[Vec<Polynomial>],
) -> Polynomial {
    // partial_{beta dot-beta} H^beta_dot-gamma, with
    // H^0_dot-gamma = H_1_dot-gamma and H^1_dot-gamma = -H_0_dot-gamma.
    h[1][dotted_gamma]
        .clone()
        .spacetime_derivative(dotted_beta)
        .add(minus(
            h[0][dotted_gamma]
                .clone()
                .spacetime_derivative(2 + dotted_beta),
        ))
}

pub(crate) fn conjugate_weyl_component(
    h: &[Vec<Polynomial>],
    number_of_one_indices: usize,
) -> Polynomial {
    let indices = match number_of_one_indices {
        0 => [0, 0, 0],
        1 => [0, 0, 1],
        2 => [0, 1, 1],
        3 => [1, 1, 1],
        _ => panic!("a symmetric rank-three spinor has four components"),
    };
    let mut result = Polynomial::default();
    for [dotted_alpha, dotted_beta, dotted_gamma] in permutations(indices) {
        let contracted = contracted_conjugate_vector_derivative(dotted_beta, dotted_gamma, h);
        let d_applied = apply(Derivative::Right(dotted_alpha), &contracted);
        result = result.add(d_squared(&d_applied));
    }
    // Complex conjugate of Eq. (5.2.5).
    result.scale(gaussian(0, 1, 6))
}

fn gauge_image_l(alpha: usize, mask: u8) -> Vec<Vec<Polynomial>> {
    let mut h = zero_h();
    let input = Polynomial::basis(mask);
    for dotted in 0..2 {
        h[alpha][dotted] = minus(apply(Derivative::Right(dotted), &input));
    }
    h
}

fn gauge_image_l_bar(dotted: usize, mask: u8) -> Vec<Vec<Polynomial>> {
    let mut h = zero_h();
    let input = Polynomial::basis(mask);
    for alpha in 0..2 {
        h[alpha][dotted] = apply(Derivative::Left(alpha), &input);
    }
    h
}

#[cfg(test)]
fn gauge_image_l_without_spinor_derivative(alpha: usize, mask: u8) -> Vec<Vec<Polynomial>> {
    let mut h = zero_h();
    for dotted in 0..2 {
        h[alpha][dotted] = Polynomial::basis(mask);
    }
    h
}

fn maximum_derivative_order(polynomial: &Polynomial) -> usize {
    polynomial
        .0
        .keys()
        .map(|monomial| {
            monomial
                .spacetime_derivatives
                .iter()
                .map(|&power| power as usize)
                .sum()
        })
        .max()
        .unwrap_or(0)
}

pub fn verify() -> PrepotentialCurvatureReport {
    let mut nonzero_curvature_images = 0;
    let mut nonzero_conjugate_curvature_images = 0;
    let mut maximum_curvature_derivative_order = 0;
    let mut chirality_relations_checked = 0;
    let mut chirality_residuals = 0;
    let mut antichirality_relations_checked = 0;
    let mut antichirality_residuals = 0;
    for alpha in 0..2 {
        for dotted in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                let mut h = zero_h();
                h[alpha][dotted] = Polynomial::basis(mask);
                for component in 0..4 {
                    let w = weyl_component(&h, component);
                    if !w.0.is_empty() {
                        nonzero_curvature_images += 1;
                    }
                    maximum_curvature_derivative_order =
                        maximum_curvature_derivative_order.max(maximum_derivative_order(&w));
                    for derivative in [Derivative::Right(0), Derivative::Right(1)] {
                        chirality_relations_checked += 1;
                        if !apply(derivative, &w).0.is_empty() {
                            chirality_residuals += 1;
                        }
                    }
                    let w_bar = conjugate_weyl_component(&h, component);
                    if !w_bar.0.is_empty() {
                        nonzero_conjugate_curvature_images += 1;
                    }
                    maximum_curvature_derivative_order =
                        maximum_curvature_derivative_order.max(maximum_derivative_order(&w_bar));
                    for derivative in [Derivative::Left(0), Derivative::Left(1)] {
                        antichirality_relations_checked += 1;
                        if !apply(derivative, &w_bar).0.is_empty() {
                            antichirality_residuals += 1;
                        }
                    }
                }
            }
        }
    }

    let mut gauge_invariance_relations_checked = 0;
    let mut gauge_invariance_residuals = 0;
    let mut conjugate_gauge_invariance_relations_checked = 0;
    let mut conjugate_gauge_invariance_residuals = 0;
    for chirality in 0..2 {
        for spinor in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                let h = if chirality == 0 {
                    gauge_image_l(spinor, mask)
                } else {
                    gauge_image_l_bar(spinor, mask)
                };
                for component in 0..4 {
                    gauge_invariance_relations_checked += 1;
                    if !weyl_component(&h, component).0.is_empty() {
                        gauge_invariance_residuals += 1;
                    }
                    conjugate_gauge_invariance_relations_checked += 1;
                    if !conjugate_weyl_component(&h, component).0.is_empty() {
                        conjugate_gauge_invariance_residuals += 1;
                    }
                }
            }
        }
    }

    let passed = nonzero_curvature_images > 0
        && maximum_curvature_derivative_order == 4
        && chirality_relations_checked == 512
        && chirality_residuals == 0
        && antichirality_relations_checked == 512
        && antichirality_residuals == 0
        && gauge_invariance_relations_checked == 256
        && gauge_invariance_residuals == 0
        && conjugate_gauge_invariance_relations_checked == 256
        && conjugate_gauge_invariance_residuals == 0;
    PrepotentialCurvatureReport {
        schema_version: "adynkra-4d-n1-prepotential-curvature-v2",
        sources: ["hep-th/0108200", "2407.09334"],
        source_equations: ["5.2.5 and 5.2.7", "2.21 and 2.22"],
        curvature: "the conjugate pair W_{alpha beta gamma} and Wbar_{dot-alpha dot-beta dot-gamma} from Superspace Eq. (5.2.5)",
        convention: "epsilon^(01) = 1 and Dbar^2 = Dbar_1 Dbar_0, following Superspace Eq. (3.4.10)",
        symmetric_weyl_components: 4,
        prepotential_basis_inputs: 64,
        gauge_parameter_basis_inputs: 64,
        nonzero_curvature_images,
        nonzero_conjugate_curvature_images,
        maximum_curvature_derivative_order,
        chirality_relations_checked,
        chirality_residuals,
        antichirality_relations_checked,
        antichirality_residuals,
        gauge_invariance_relations_checked,
        gauge_invariance_residuals,
        conjugate_gauge_invariance_relations_checked,
        conjugate_gauge_invariance_residuals,
        boundary: "this report covers the conformal super-Weyl curvature; the old-minimal scalar and vector curvatures and Bianchi identities are validated in the minimal-curvature artifact; cohomology and the Euler-Lagrange equation are separate",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn super_weyl_curvature_is_chiral_on_the_complete_prepotential_basis() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.chirality_relations_checked, 512);
        assert_eq!(report.chirality_residuals, 0);
        assert_eq!(report.antichirality_relations_checked, 512);
        assert_eq!(report.antichirality_residuals, 0);
    }

    #[test]
    fn super_weyl_curvature_annihilates_the_complete_gauge_image() {
        let report = verify();
        assert_eq!(report.gauge_invariance_relations_checked, 256);
        assert_eq!(report.gauge_invariance_residuals, 0);
        assert_eq!(report.conjugate_gauge_invariance_relations_checked, 256);
        assert_eq!(report.conjugate_gauge_invariance_residuals, 0);
    }

    #[test]
    fn dropping_the_spinor_derivative_from_the_gauge_map_is_detected() {
        let mut residuals = 0;
        for alpha in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                let h = gauge_image_l_without_spinor_derivative(alpha, mask);
                for component in 0..4 {
                    residuals += usize::from(!weyl_component(&h, component).0.is_empty());
                }
            }
        }
        assert!(residuals > 0);
    }
}
