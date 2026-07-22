//! Quadratic old-minimal 4D N=1 supergravity action and its variation.
//!
//! The source action is *Superspace*, Eq. (7.2.36), equivalently the flat
//! background specialization of Eq. (7.4.1). The checks below compare its
//! polarized Hessian with the independently implemented curvature operator
//! `(G, R, Rbar)` and verify the associated Noether identities.

use crate::minimal_supergravity_curvatures::{
    conjugate_scalar_curvature, d_bar_squared, d_squared, gauge_image_l, gauge_image_l_bar,
    raised_vector_component, scalar_curvature, spacetime_divergence, vector_curvature, zero_h,
};
use crate::supercovariant_derivative::{Derivative, GaussianRational, Polynomial, apply};
use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;

const GRASSMANN_DIMENSION: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct ActionMomentumCheck {
    pub momentum: [i64; 4],
    pub potential_basis_dimension: usize,
    pub hessian_entries_checked: usize,
    pub source_variation_residuals: usize,
    pub formal_self_adjointness_residuals: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MinimalSupergravityActionReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub source_equation: &'static str,
    pub action: &'static str,
    pub euler_lagrange_operator: &'static str,
    pub convention: &'static str,
    pub quadratic_momentum_monomials: usize,
    pub momentum_interpolation_determinant: i64,
    pub momentum_interpolation_full_rank: bool,
    pub momentum_checks: Vec<ActionMomentumCheck>,
    pub total_hessian_entries_checked: usize,
    pub total_source_variation_residuals: usize,
    pub total_formal_self_adjointness_residuals: usize,
    pub compensator_projection_relations_checked: usize,
    pub compensator_projection_residuals: usize,
    pub noether_relations_checked: usize,
    pub noether_residuals: usize,
    pub source_action_yields_known_equations: bool,
    pub formally_self_adjoint: bool,
    pub gauge_noether_identity_holds: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

#[derive(Clone)]
struct Potential {
    h: Vec<Vec<Polynomial>>,
    chi: Polynomial,
    chi_bar: Polynomial,
    parity: u32,
}

#[derive(Clone)]
struct ActionData {
    potential: Potential,
    divergence: Polynomial,
    commutator_trace: Polynomial,
    kinetic_h: Vec<Polynomial>,
    vector_euler: Vec<Polynomial>,
    chi_force: Polynomial,
    chi_bar_force: Polynomial,
}

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> GaussianRational {
    Complex::new(
        Ratio::new(real, denominator),
        Ratio::new(imaginary, denominator),
    )
}

fn zero() -> GaussianRational {
    gaussian(0, 0, 1)
}

fn flatten_h(h: &[Vec<Polynomial>]) -> Vec<Polynomial> {
    vec![
        h[0][0].clone(),
        h[0][1].clone(),
        h[1][0].clone(),
        h[1][1].clone(),
    ]
}

fn d_alembertian(input: &Polynomial) -> Polynomial {
    input
        .clone()
        .spacetime_derivative(0)
        .spacetime_derivative(3)
        .add(
            input
                .clone()
                .spacetime_derivative(1)
                .spacetime_derivative(2)
                .scale(gaussian(-1, 0, 1)),
        )
}

fn kinetic_operator(input: &Polynomial) -> Polynomial {
    d_alembertian(input)
        .add(d_squared(&d_bar_squared(input)).scale(gaussian(-1, 0, 1)))
        .add(d_bar_squared(&d_squared(input)).scale(gaussian(-1, 0, 1)))
}

fn commutator_trace(h: &[Vec<Polynomial>]) -> Polynomial {
    let flat = flatten_h(h);
    let mut result = Polynomial::default();
    for alpha in 0..2 {
        for dotted in 0..2 {
            let raised = raised_vector_component(&flat, 2 * alpha + dotted);
            let d_then_d_bar = apply(
                Derivative::Right(dotted),
                &apply(Derivative::Left(alpha), &raised),
            );
            let d_bar_then_d = apply(
                Derivative::Left(alpha),
                &apply(Derivative::Right(dotted), &raised),
            );
            result = result
                .add(d_then_d_bar)
                .add(d_bar_then_d.scale(gaussian(-1, 0, 1)));
        }
    }
    result
}

fn action_data(potential: Potential) -> ActionData {
    let flat = flatten_h(&potential.h);
    let divergence = spacetime_divergence(&potential.h);
    let commutator_trace = commutator_trace(&potential.h);
    let kinetic_h = flat.iter().map(kinetic_operator).collect();
    let vector_euler = (0..4)
        .map(|output| {
            vector_curvature(&potential.h, &potential.chi, &potential.chi_bar, output)
                .scale(gaussian(-1, 0, 1))
        })
        .collect();
    let chi_force = potential
        .chi_bar
        .clone()
        .scale(gaussian(-3, 0, 1))
        .add(divergence.clone().scale(gaussian(0, 1, 1)));
    let chi_bar_force = potential
        .chi
        .clone()
        .scale(gaussian(-3, 0, 1))
        .add(divergence.clone().scale(gaussian(0, -1, 1)));
    ActionData {
        potential,
        divergence,
        commutator_trace,
        kinetic_h,
        vector_euler,
        chi_force,
        chi_bar_force,
    }
}

fn basis_potentials() -> Vec<Potential> {
    let mut basis = Vec::with_capacity(96);
    for component in 0..4 {
        for mask in 0..GRASSMANN_DIMENSION as u8 {
            let mut h = zero_h();
            h[component / 2][component % 2] = Polynomial::basis(mask);
            basis.push(Potential {
                h,
                chi: Polynomial::default(),
                chi_bar: Polynomial::default(),
                parity: mask.count_ones() % 2,
            });
        }
    }
    for mask in 0..GRASSMANN_DIMENSION as u8 {
        basis.push(Potential {
            h: zero_h(),
            chi: Polynomial::basis(mask),
            chi_bar: Polynomial::default(),
            parity: mask.count_ones() % 2,
        });
    }
    for mask in 0..GRASSMANN_DIMENSION as u8 {
        basis.push(Potential {
            h: zero_h(),
            chi: Polynomial::default(),
            chi_bar: Polynomial::basis(mask),
            parity: mask.count_ones() % 2,
        });
    }
    basis
}

fn evaluate(polynomial: &Polynomial, momentum: [i64; 4]) -> [GaussianRational; 16] {
    let mut result: [GaussianRational; 16] = std::array::from_fn(|_| zero());
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

fn multiplication_sign(left_mask: u8, right_mask: u8) -> i64 {
    let mut inversions = 0;
    for left in 0..4 {
        if left_mask & (1 << left) == 0 {
            continue;
        }
        inversions += (right_mask & ((1 << left) - 1)).count_ones();
    }
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn full_superspace_pair(
    left: &Polynomial,
    left_momentum: [i64; 4],
    right: &Polynomial,
    right_momentum: [i64; 4],
) -> GaussianRational {
    let left_values = evaluate(left, left_momentum);
    let right_values = evaluate(right, right_momentum);
    let mut result = zero();
    for left_mask in 0u8..16 {
        let right_mask = 15 ^ left_mask;
        result += left_values[left_mask as usize].clone()
            * right_values[right_mask as usize].clone()
            * Ratio::from_integer(multiplication_sign(left_mask, right_mask));
    }
    result
}

fn opposite(momentum: [i64; 4]) -> [i64; 4] {
    momentum.map(|entry| -entry)
}

fn source_hessian(left: &ActionData, right: &ActionData, momentum: [i64; 4]) -> GaussianRational {
    let minus_momentum = opposite(momentum);
    let left_h = flatten_h(&left.potential.h);
    let right_h = flatten_h(&right.potential.h);
    let left_h_raised: Vec<_> = (0..4)
        .map(|component| raised_vector_component(&left_h, component))
        .collect();
    let right_h_raised: Vec<_> = (0..4)
        .map(|component| raised_vector_component(&right_h, component))
        .collect();
    let reversed_order_sign = if left.potential.parity * right.potential.parity == 0 {
        gaussian(1, 0, 1)
    } else {
        gaussian(-1, 0, 1)
    };

    let mut result = full_superspace_pair(
        &left.potential.chi_bar,
        minus_momentum,
        &right.potential.chi,
        momentum,
    ) * gaussian(-3, 0, 1);
    result += full_superspace_pair(
        &right.potential.chi_bar,
        momentum,
        &left.potential.chi,
        minus_momentum,
    ) * gaussian(-3, 0, 1)
        * reversed_order_sign.clone();

    let left_compensator_difference = left
        .potential
        .chi
        .clone()
        .add(left.potential.chi_bar.clone().scale(gaussian(-1, 0, 1)));
    let right_compensator_difference = right
        .potential
        .chi
        .clone()
        .add(right.potential.chi_bar.clone().scale(gaussian(-1, 0, 1)));
    result += full_superspace_pair(
        &left_compensator_difference,
        minus_momentum,
        &right.divergence,
        momentum,
    ) * gaussian(0, 1, 1);
    result += full_superspace_pair(
        &right_compensator_difference,
        momentum,
        &left.divergence,
        minus_momentum,
    ) * gaussian(0, 1, 1)
        * reversed_order_sign.clone();

    for component in 0..4 {
        result += full_superspace_pair(
            &left_h_raised[component],
            minus_momentum,
            &right.kinetic_h[component],
            momentum,
        ) * gaussian(-1, 0, 2);
        result += full_superspace_pair(
            &right_h_raised[component],
            momentum,
            &left.kinetic_h[component],
            minus_momentum,
        ) * gaussian(-1, 0, 2)
            * reversed_order_sign.clone();
    }
    result += full_superspace_pair(
        &left.divergence,
        minus_momentum,
        &right.divergence,
        momentum,
    ) * gaussian(-1, 0, 2);
    result += full_superspace_pair(
        &left.commutator_trace,
        minus_momentum,
        &right.commutator_trace,
        momentum,
    ) * gaussian(1, 0, 6);
    result
}

fn euler_pair(left: &ActionData, right: &ActionData, momentum: [i64; 4]) -> GaussianRational {
    let minus_momentum = opposite(momentum);
    let left_h = flatten_h(&left.potential.h);
    let left_h_raised: Vec<_> = (0..4)
        .map(|component| raised_vector_component(&left_h, component))
        .collect();
    let mut result = zero();
    for component in 0..4 {
        result += full_superspace_pair(
            &left_h_raised[component],
            minus_momentum,
            &right.vector_euler[component],
            momentum,
        );
    }
    result += full_superspace_pair(
        &left.potential.chi,
        minus_momentum,
        &right.chi_force,
        momentum,
    );
    result += full_superspace_pair(
        &left.potential.chi_bar,
        minus_momentum,
        &right.chi_bar_force,
        momentum,
    );
    result
}

fn momentum_check(momentum: [i64; 4], basis: &[ActionData]) -> ActionMomentumCheck {
    let mut source_variation_residuals = 0;
    let mut formal_self_adjointness_residuals = 0;
    for left in basis {
        for right in basis {
            let euler = euler_pair(left, right, momentum);
            let source = source_hessian(left, right, momentum);
            if source != euler {
                source_variation_residuals += 1;
            }
            let graded_sign = if left.potential.parity * right.potential.parity == 0 {
                gaussian(1, 0, 1)
            } else {
                gaussian(-1, 0, 1)
            };
            let adjoint = euler_pair(right, left, opposite(momentum)) * graded_sign;
            if euler != adjoint {
                formal_self_adjointness_residuals += 1;
            }
        }
    }
    ActionMomentumCheck {
        momentum,
        potential_basis_dimension: basis.len(),
        hessian_entries_checked: basis.len() * basis.len(),
        source_variation_residuals,
        formal_self_adjointness_residuals,
    }
}

fn momentum_interpolation_determinant(momenta: &[[i64; 4]]) -> Ratio<i64> {
    let mut matrix: Vec<Vec<Ratio<i64>>> = momenta
        .iter()
        .map(|momentum| {
            let mut row = vec![Ratio::from_integer(1)];
            row.extend(momentum.iter().copied().map(Ratio::from_integer));
            row.extend(
                momentum
                    .iter()
                    .map(|value| Ratio::from_integer(value * value)),
            );
            for first in 0..4 {
                for second in first + 1..4 {
                    row.push(Ratio::from_integer(momentum[first] * momentum[second]));
                }
            }
            row
        })
        .collect();
    assert_eq!(matrix.len(), 15);
    assert!(matrix.iter().all(|row| row.len() == 15));
    let mut determinant = Ratio::from_integer(1);
    for column in 0..15 {
        let pivot = (column..15).find(|&row| matrix[row][column] != Ratio::from_integer(0));
        let Some(pivot) = pivot else {
            return Ratio::from_integer(0);
        };
        if pivot != column {
            matrix.swap(pivot, column);
            determinant = -determinant;
        }
        let pivot_value = matrix[column][column];
        determinant *= pivot_value;
        for row in column + 1..15 {
            let factor = matrix[row][column] / pivot_value;
            for next in column..15 {
                matrix[row][next] = matrix[row][next] - factor * matrix[column][next];
            }
        }
    }
    determinant
}

pub fn verify() -> MinimalSupergravityActionReport {
    let basis: Vec<_> = basis_potentials().into_iter().map(action_data).collect();
    // These probes span the constant, linear, and quadratic momentum
    // monomials appearing in the Hessian comparison.
    let momenta = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 2, 3, 5],
        [2, -1, 4, 3],
        [1, 2, 3, 6],
        [-2, 3, 1, 4],
    ];
    let momentum_checks: Vec<_> = momenta
        .into_iter()
        .map(|momentum| momentum_check(momentum, &basis))
        .collect();
    let interpolation_determinant = momentum_interpolation_determinant(&momenta);
    let momentum_interpolation_full_rank = interpolation_determinant != Ratio::from_integer(0);
    let total_hessian_entries_checked = momentum_checks
        .iter()
        .map(|check| check.hessian_entries_checked)
        .sum();
    let total_source_variation_residuals = momentum_checks
        .iter()
        .map(|check| check.source_variation_residuals)
        .sum();
    let total_formal_self_adjointness_residuals = momentum_checks
        .iter()
        .map(|check| check.formal_self_adjointness_residuals)
        .sum();

    let mut compensator_projection_relations_checked = 0;
    let mut compensator_projection_residuals = 0;
    for input in &basis {
        compensator_projection_relations_checked += 2;
        let chi_projection = d_bar_squared(&input.chi_force);
        let expected_chi = scalar_curvature(&input.potential.h, &input.potential.chi_bar)
            .scale(gaussian(-3, 0, 1));
        if chi_projection != expected_chi {
            compensator_projection_residuals += 1;
        }
        let chi_bar_projection = d_squared(&input.chi_bar_force);
        let expected_chi_bar = conjugate_scalar_curvature(&input.potential.h, &input.potential.chi)
            .scale(gaussian(-3, 0, 1));
        if chi_bar_projection != expected_chi_bar {
            compensator_projection_residuals += 1;
        }
    }

    let mut noether_relations_checked = 0;
    let mut noether_residuals = 0;
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
                for output in 0..4 {
                    noether_relations_checked += 1;
                    if !vector_curvature(&h, &chi, &chi_bar, output).0.is_empty() {
                        noether_residuals += 1;
                    }
                }
                noether_relations_checked += 2;
                if !scalar_curvature(&h, &chi_bar).0.is_empty() {
                    noether_residuals += 1;
                }
                if !conjugate_scalar_curvature(&h, &chi).0.is_empty() {
                    noether_residuals += 1;
                }
            }
        }
    }

    let source_action_yields_known_equations =
        total_source_variation_residuals == 0 && compensator_projection_residuals == 0;
    let formally_self_adjoint = total_formal_self_adjointness_residuals == 0;
    let gauge_noether_identity_holds = noether_residuals == 0;
    let passed = source_action_yields_known_equations
        && formally_self_adjoint
        && gauge_noether_identity_holds
        && momentum_interpolation_full_rank;
    MinimalSupergravityActionReport {
        schema_version: "adynkra-4d-n1-old-minimal-action-v1",
        source: "hep-th/0108200",
        source_equation: "Superspace Eq. (7.2.36), equivalent in flat background to Eq. (7.4.1)",
        action: "integral d^4x d^4theta [-3 chi_bar chi + i(chi-chi_bar) partial.H - (1/2) H.(box-D^2 Dbar^2-Dbar^2 D^2)H - (1/4)(partial.H)^2 + (1/12)([Dbar,D]H)^2]",
        euler_lagrange_operator: "delta S/delta H = -G; Dbar^2(delta S/delta chi) = -3R; D^2(delta S/delta chi_bar) = -3Rbar",
        convention: "phi = 1 + chi, epsilon^(01) = 1, box = partial_00 partial_11 - partial_01 partial_10, and the overall kappa^(-2) is suppressed",
        quadratic_momentum_monomials: 15,
        momentum_interpolation_determinant: *interpolation_determinant.numer(),
        momentum_interpolation_full_rank,
        momentum_checks,
        total_hessian_entries_checked,
        total_source_variation_residuals,
        total_formal_self_adjointness_residuals,
        compensator_projection_relations_checked,
        compensator_projection_residuals,
        noether_relations_checked,
        noether_residuals,
        source_action_yields_known_equations,
        formally_self_adjoint,
        gauge_noether_identity_holds,
        boundary: "quadratic linearized old-minimal action in flat 4D N=1 superspace; the full-rank interpolation design certifies the degree-two momentum-polynomial Hessian, but this is not a nonlinear or higher-dimensional action",
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_action_varies_to_the_known_curvature_equations() {
        let report = verify();
        assert_eq!(report.total_source_variation_residuals, 0);
        assert_eq!(report.compensator_projection_residuals, 0);
        assert_eq!(report.momentum_interpolation_determinant, -1440);
        assert!(report.momentum_interpolation_full_rank);
        assert!(report.source_action_yields_known_equations);
    }

    #[test]
    fn quadratic_operator_is_formally_self_adjoint() {
        let report = verify();
        assert_eq!(report.total_formal_self_adjointness_residuals, 0);
        assert!(report.formally_self_adjoint);
    }

    #[test]
    fn gauge_map_obeys_the_noether_identity() {
        let report = verify();
        assert_eq!(report.noether_relations_checked, 384);
        assert_eq!(report.noether_residuals, 0);
        assert!(report.gauge_noether_identity_holds);
        assert!(report.passed);
    }
}
