//! Linearized old-minimal 4D N=1 scalar curvature and chiral compensator,
//! from *Superspace*, Eqs. (7.4.2b) and (7.5.19).

use crate::supercovariant_derivative::{Derivative, GaussianRational, Polynomial, apply};
use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

const GRASSMANN_DIMENSION: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct MinimalScalarCurvatureReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub source_equations: &'static str,
    pub compensator: &'static str,
    pub scalar_curvature: &'static str,
    pub convention: &'static str,
    pub prepotential_basis_inputs: usize,
    pub compensator_basis_inputs: usize,
    pub nonzero_curvature_images: usize,
    pub chirality_relations_checked: usize,
    pub chirality_residuals: usize,
    pub gauge_invariance_relations_checked: usize,
    pub gauge_invariance_residuals: usize,
    pub uncompensated_mutation_residuals: usize,
    pub boundary: &'static str,
    pub passed: bool,
}

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> GaussianRational {
    Complex::new(
        Ratio::new(real, denominator),
        Ratio::new(imaginary, denominator),
    )
}

fn zero_h() -> Vec<Vec<Polynomial>> {
    vec![vec![Polynomial::default(); 2]; 2]
}

fn d_squared(polynomial: &Polynomial) -> Polynomial {
    apply(Derivative::Left(1), &apply(Derivative::Left(0), polynomial))
}

fn d_bar_squared(polynomial: &Polynomial) -> Polynomial {
    apply(
        Derivative::Right(1),
        &apply(Derivative::Right(0), polynomial),
    )
}

fn spacetime_divergence(h: &[Vec<Polynomial>]) -> Polynomial {
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

fn scalar_curvature(h: &[Vec<Polynomial>], chi_bar: &Polynomial) -> Polynomial {
    let interior = chi_bar
        .clone()
        .add(spacetime_divergence(h).scale(gaussian(0, -1, 3)));
    d_bar_squared(&interior)
}

fn gauge_image_l(alpha: usize, mask: u8) -> (Vec<Vec<Polynomial>>, Polynomial) {
    let mut h = zero_h();
    let input = Polynomial::basis(mask);
    for dotted in 0..2 {
        h[alpha][dotted] = apply(Derivative::Right(dotted), &input).scale(gaussian(-1, 0, 1));
    }
    (h, Polynomial::default())
}

fn raised_d_bar(dotted_parameter: usize, input: &Polynomial) -> Polynomial {
    match dotted_parameter {
        0 => apply(Derivative::Right(1), input).scale(gaussian(-1, 0, 1)),
        1 => apply(Derivative::Right(0), input),
        _ => panic!("dotted spinor index exceeds two components"),
    }
}

fn gauge_image_l_bar(
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

pub fn verify() -> MinimalScalarCurvatureReport {
    let mut nonzero_curvature_images = 0;
    let mut chirality_relations_checked = 0;
    let mut chirality_residuals = 0;
    for input_kind in 0..5 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            let mut h = zero_h();
            let mut chi_bar = Polynomial::default();
            if input_kind < 4 {
                h[input_kind / 2][input_kind % 2] = Polynomial::basis(mask);
            } else {
                chi_bar = Polynomial::basis(mask);
            }
            let r = scalar_curvature(&h, &chi_bar);
            if !r.0.is_empty() {
                nonzero_curvature_images += 1;
            }
            for derivative in [Derivative::Right(0), Derivative::Right(1)] {
                chirality_relations_checked += 1;
                if !apply(derivative, &r).0.is_empty() {
                    chirality_residuals += 1;
                }
            }
        }
    }

    let mut gauge_invariance_relations_checked = 0;
    let mut gauge_invariance_residuals = 0;
    let mut uncompensated_mutation_residuals = 0;
    for chirality in 0..2 {
        for spinor in 0..2 {
            for mask in 0..GRASSMANN_DIMENSION as u8 {
                let (h, chi_bar) = if chirality == 0 {
                    gauge_image_l(spinor, mask)
                } else {
                    gauge_image_l_bar(spinor, mask, true)
                };
                gauge_invariance_relations_checked += 1;
                if !scalar_curvature(&h, &chi_bar).0.is_empty() {
                    gauge_invariance_residuals += 1;
                }
                if chirality == 1 {
                    let (h_uncompensated, zero_chi) = gauge_image_l_bar(spinor, mask, false);
                    if !scalar_curvature(&h_uncompensated, &zero_chi).0.is_empty() {
                        uncompensated_mutation_residuals += 1;
                    }
                }
            }
        }
    }

    let passed = nonzero_curvature_images > 0
        && chirality_relations_checked == 160
        && chirality_residuals == 0
        && gauge_invariance_relations_checked == 64
        && gauge_invariance_residuals == 0
        && uncompensated_mutation_residuals > 0;
    MinimalScalarCurvatureReport {
        schema_version: "adynkra-4d-n1-old-minimal-scalar-curvature-v1",
        source: "hep-th/0108200",
        source_equations: "7.4.2b and 7.5.19",
        compensator: "delta chi_bar = (1/3) D^2 Dbar^dot-alpha Lbar_dot-alpha",
        scalar_curvature: "R = Dbar^2 (chi_bar - (i/3) partial_a H^a)",
        convention: "phi = 1 + chi, epsilon^(01) = 1, D^2 = D_1 D_0, and Dbar^2 = Dbar_1 Dbar_0",
        prepotential_basis_inputs: 64,
        compensator_basis_inputs: 16,
        nonzero_curvature_images,
        chirality_relations_checked,
        chirality_residuals,
        gauge_invariance_relations_checked,
        gauge_invariance_residuals,
        uncompensated_mutation_residuals,
        boundary: "old-minimal scalar curvature and compensator only; G_a, Bianchi identities, cohomology, and equations of motion remain open",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_curvature_is_chiral_on_the_complete_input_basis() {
        let report = verify();
        assert_eq!(report.chirality_relations_checked, 160);
        assert_eq!(report.chirality_residuals, 0);
    }

    #[test]
    fn compensator_makes_the_scalar_curvature_gauge_invariant() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.gauge_invariance_residuals, 0);
        assert!(report.uncompensated_mutation_residuals > 0);
    }
}
