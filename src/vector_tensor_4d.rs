//! Exact 4D vector-tensor component closure for arXiv:1405.0048 Eq. (77).
//!
//! The first fixture uses the physical one-central-charge branch `(m,n)=(1,0)`
//! and checks the corrected Eq. (78) on all component fields.

#![allow(clippy::needless_range_loop)]

use crate::chiral_vector_4d::{Clifford4D, GaussianRational, Matrix4, matrix_mul, pauli};
use crate::exact_component_algebra::Polynomial;
use crate::higher_dimensional_canonical::{
    BianchiIdentity as CanonicalBianchi, CentralEntry as CanonicalCentralEntry,
    CentralGenerator as CanonicalCentralGenerator, CentralOccurrence as CanonicalCentralOccurrence,
    Component as CanonicalComponent, ComponentRole as CanonicalRole,
    DerivativeMonomial as CanonicalDerivative, GaugeTerm as CanonicalGaugeTerm,
    GaussianRational as CanonicalCoefficient, LinearTerm as CanonicalLinearTerm,
    LinkageTerm as CanonicalLinkage, LorentzRep,
    PhysicalFingerprint as CanonicalPhysicalFingerprint, Reality, Statistics,
    Supercharge as CanonicalSupercharge,
};
use num_rational::Ratio;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const DIM: usize = 4;
const SPINORS: usize = 4;
type Poly = Polynomial<Field, DIM>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
enum Field {
    Scalar,
    Vector(usize),
    TwoForm(usize, usize),
    Auxiliary,
    Lambda(usize, usize),
}

impl Field {
    fn all() -> Vec<Self> {
        let mut fields = vec![Self::Scalar];
        fields.extend((0..DIM).map(Self::Vector));
        for mu in 0..DIM {
            for nu in (mu + 1)..DIM {
                fields.push(Self::TwoForm(mu, nu));
            }
        }
        fields.push(Self::Auxiliary);
        fields.extend((0..2).flat_map(|supersymmetry| {
            (0..SPINORS).map(move |spinor| Self::Lambda(supersymmetry, spinor))
        }));
        fields
    }
}

#[derive(Clone, Copy)]
struct Charge {
    supersymmetry: usize,
    spinor: usize,
}

fn e(index: usize, other: usize) -> i64 {
    [[0, 1], [-1, 0]][index][other]
}

fn metric_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
}

fn epsilon_lower(indices: [usize; 4]) -> i64 {
    if (0..4).any(|left| ((left + 1)..4).any(|right| indices[left] == indices[right])) {
        return 0;
    }
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|&(left, right)| indices[left] > indices[right])
        .count();
    if inversions.is_multiple_of(2) { -1 } else { 1 }
}

fn epsilon_upper(indices: [usize; 4]) -> i64 {
    epsilon_lower(indices) * indices.into_iter().map(metric_sign).product::<i64>()
}

fn two_form(mu: usize, nu: usize) -> (Field, i64) {
    if mu < nu {
        (Field::TwoForm(mu, nu), 1)
    } else {
        (Field::TwoForm(nu, mu), -1)
    }
}

fn two_form_polynomial(mu: usize, nu: usize) -> Poly {
    if mu == nu {
        return Poly::default();
    }
    let (field, sign) = two_form(mu, nu);
    let mut result = Poly::default();
    result.add_scaled(&Poly::atom(field), GaussianRational::new(sign, 0));
    result
}

fn vector_strength(mu: usize, nu: usize) -> Poly {
    let mut result = Poly::atom(Field::Vector(nu)).derivative(mu);
    result.add_scaled(
        &Poly::atom(Field::Vector(mu)).derivative(nu),
        GaussianRational::new(-1, 0),
    );
    result
}

fn tensor_strength(alpha: usize, mu: usize, nu: usize) -> Poly {
    let mut result = two_form_polynomial(mu, nu).derivative(alpha);
    result.add_scaled(
        &two_form_polynomial(nu, alpha).derivative(mu),
        GaussianRational::new(1, 0),
    );
    result.add_scaled(
        &two_form_polynomial(alpha, mu).derivative(nu),
        GaussianRational::new(1, 0),
    );
    result
}

fn lower_commutator(clifford: &Clifford4D, mu: usize, nu: usize) -> Matrix4 {
    let first = matrix_mul(&clifford.gamma_down[mu], &clifford.gamma_down[nu]);
    let second = matrix_mul(&clifford.gamma_down[nu], &clifford.gamma_down[mu]);
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let mut value = first[row][column];
            value.add_assign(&second[row][column].mul(&GaussianRational::new(-1, 0)));
            value
        })
    })
}

fn add_fermion_row(
    result: &mut Poly,
    matrix: &Matrix4,
    row: usize,
    internal: usize,
    derivative: Option<usize>,
    scale: GaussianRational,
) {
    for component in 0..SPINORS {
        let mut term = Poly::atom(Field::Lambda(internal, component));
        if let Some(mu) = derivative {
            term = term.derivative(mu);
        }
        result.add_scaled(&term, matrix[row][component].mul(&scale));
    }
}

fn fermion_delta(
    charge: Charge,
    target_internal: usize,
    target_spinor: usize,
    clifford: &Clifford4D,
) -> Poly {
    let i = charge.supersymmetry;
    let a = charge.spinor;
    let j = target_internal;
    let b = target_spinor;
    let mut result = Poly::default();

    let aij = e(i, j);
    if aij != 0 {
        for mu in 0..DIM {
            for nu in (mu + 1)..DIM {
                let commutator = clifford.lower_spinors(&clifford.commutator_up(mu, nu));
                result.add_scaled(
                    &vector_strength(mu, nu),
                    commutator[a][b].mul(&GaussianRational::from_ratio(
                        Ratio::from_integer(0),
                        Ratio::new(-aij, 2),
                    )),
                );
            }
        }
        let gamma5 = clifford.lower_spinors(&clifford.gamma5);
        result.add_scaled(
            &Poly::atom(Field::Auxiliary),
            gamma5[a][b].mul(&GaussianRational::new(aij, 0)),
        );
    }

    if i == j {
        for mu in 0..DIM {
            let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
            result.add_scaled(
                &Poly::atom(Field::Scalar).derivative(mu),
                gamma[a][b].mul(&GaussianRational::new(0, 1)),
            );
            let gamma5_gamma_down =
                clifford.lower_spinors(&matrix_mul(&clifford.gamma5, &clifford.gamma_down[mu]));
            for rho in 0..DIM {
                for sigma in 0..DIM {
                    for tau in (sigma + 1)..DIM {
                        let epsilon = epsilon_upper([mu, rho, sigma, tau]);
                        if epsilon == 0 {
                            continue;
                        }
                        result.add_scaled(
                            &Poly::atom(Field::TwoForm(sigma, tau)).derivative(rho),
                            gamma5_gamma_down[a][b].mul(&GaussianRational::new(2 * epsilon, 0)),
                        );
                    }
                }
            }
        }
    }
    result
}

fn delta(charge: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let i = charge.supersymmetry;
    let a = charge.spinor;
    match field {
        Field::Scalar => Poly::atom(Field::Lambda(i, a)),
        Field::Vector(mu) => {
            let mut result = Poly::default();
            for j in 0..2 {
                let coefficient = e(i, j);
                if coefficient != 0 {
                    add_fermion_row(
                        &mut result,
                        &clifford.gamma_down[mu],
                        a,
                        j,
                        None,
                        GaussianRational::new(coefficient, 0),
                    );
                }
            }
            result
        }
        Field::TwoForm(mu, nu) => {
            let commutator = lower_commutator(clifford, mu, nu);
            let mut result = Poly::default();
            add_fermion_row(
                &mut result,
                &commutator,
                a,
                i,
                None,
                GaussianRational::from_ratio(Ratio::new(-1, 4), Ratio::from_integer(0)),
            );
            result
        }
        Field::Auxiliary => {
            let mut result = Poly::default();
            for j in 0..2 {
                let coefficient = e(i, j);
                if coefficient == 0 {
                    continue;
                }
                for mu in 0..DIM {
                    add_fermion_row(
                        &mut result,
                        &clifford.gamma5_gamma_up(mu),
                        a,
                        j,
                        Some(mu),
                        GaussianRational::new(0, coefficient),
                    );
                }
            }
            result
        }
        Field::Lambda(j, b) => fermion_delta(charge, j, b, clifford),
    }
}

fn apply_delta(charge: Charge, polynomial: &Poly, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    for (jet, coefficient) in &polynomial.0 {
        let mut transformed = delta(charge, jet.field, clifford);
        for mu in 0..DIM {
            for _ in 0..jet.derivatives[mu] {
                transformed = transformed.derivative(mu);
            }
        }
        result.add_scaled(&transformed, *coefficient);
    }
    result
}

/// The normalized derivative-valued central operator extracted from the
/// non-gauge part of the corrected Eq. (78) closure ledger.  The common
/// supercharge-pair coefficient is
/// `2 i (sigma_2)^{ij} (gamma^5)_{ab}`.
fn central_delta(field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    match field {
        Field::Scalar => {
            result.add_scaled(&Poly::atom(Field::Auxiliary), GaussianRational::new(1, 0))
        }
        Field::Auxiliary => {
            for rho in 0..DIM {
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(rho).derivative(rho),
                    GaussianRational::new(-metric_sign(rho), 0),
                );
            }
        }
        Field::Vector(nu) => {
            for alpha in 0..DIM {
                for beta in 0..DIM {
                    for rho in 0..DIM {
                        let epsilon = epsilon_lower([nu, alpha, beta, rho]);
                        if epsilon == 0 {
                            continue;
                        }
                        let raised = metric_sign(alpha) * metric_sign(beta) * metric_sign(rho);
                        result.add_scaled(
                            &tensor_strength(alpha, beta, rho),
                            GaussianRational::from_ratio(
                                Ratio::new(epsilon * raised, 3),
                                Ratio::from_integer(0),
                            ),
                        );
                    }
                }
            }
        }
        Field::TwoForm(mu, nu) => {
            for alpha in 0..DIM {
                for beta in (alpha + 1)..DIM {
                    let epsilon = epsilon_lower([mu, nu, alpha, beta]);
                    if epsilon == 0 {
                        continue;
                    }
                    let raised = metric_sign(alpha) * metric_sign(beta);
                    result.add_scaled(
                        &vector_strength(alpha, beta),
                        GaussianRational::from_ratio(
                            Ratio::new(-epsilon * raised, 2),
                            Ratio::from_integer(0),
                        ),
                    );
                }
            }
        }
        Field::Lambda(k, c) => {
            for l in 0..2 {
                let internal = pauli(2)[k][l];
                if internal.is_zero() {
                    continue;
                }
                for rho in 0..DIM {
                    let gamma5_gamma = clifford.gamma5_gamma_up(rho);
                    for d in 0..SPINORS {
                        result.add_scaled(
                            &Poly::atom(Field::Lambda(l, d)).derivative(rho),
                            internal
                                .mul(&gamma5_gamma[c][d])
                                .mul(&GaussianRational::new(-1, 0)),
                        );
                    }
                }
            }
        }
    }
    result
}

fn apply_central(polynomial: &Poly, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    for (jet, coefficient) in &polynomial.0 {
        let mut transformed = central_delta(jet.field, clifford);
        for mu in 0..DIM {
            for _ in 0..jet.derivatives[mu] {
                transformed = transformed.derivative(mu);
            }
        }
        result.add_scaled(&transformed, *coefficient);
    }
    result
}

fn central_supercharge_commutator(charge: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = apply_delta(charge, &central_delta(field, clifford), clifford);
    result.add_scaled(
        &apply_central(&delta(charge, field, clifford), clifford),
        GaussianRational::new(-1, 0),
    );
    result
}

fn remove_one_derivative(polynomial: &Poly, axis: usize) -> Option<Poly> {
    let mut result = Poly::default();
    for (jet, coefficient) in &polynomial.0 {
        if jet.derivatives[axis] == 0 {
            return None;
        }
        let mut derivatives = jet.derivatives;
        derivatives[axis] -= 1;
        result.add_term(jet.field, derivatives, *coefficient);
    }
    Some(result)
}

fn vector_commutator_gauge_parameter(charge: Charge, clifford: &Clifford4D) -> Option<Poly> {
    let parameters: Option<Vec<_>> = (0..DIM)
        .map(|nu| {
            remove_one_derivative(
                &central_supercharge_commutator(charge, Field::Vector(nu), clifford),
                nu,
            )
        })
        .collect();
    let parameters = parameters?;
    parameters
        .iter()
        .all(|parameter| parameter == &parameters[0])
        .then(|| parameters[0].clone())
}

fn insert_constraint(
    constraints: &mut BTreeMap<(usize, Field), GaussianRational>,
    key: (usize, Field),
    value: GaussianRational,
) -> bool {
    match constraints.get(&key) {
        Some(existing) => existing == &value,
        None => {
            constraints.insert(key, value);
            true
        }
    }
}

fn two_form_commutator_gauge_parameters(
    charge: Charge,
    clifford: &Clifford4D,
) -> Option<[Poly; DIM]> {
    let mut constraints = BTreeMap::new();
    let mut residuals = BTreeMap::new();
    for mu in 0..DIM {
        for nu in (mu + 1)..DIM {
            let residual = central_supercharge_commutator(charge, Field::TwoForm(mu, nu), clifford);
            for (jet, coefficient) in &residual.0 {
                let derivative_axes: Vec<_> = jet
                    .derivatives
                    .iter()
                    .enumerate()
                    .flat_map(|(axis, &count)| std::iter::repeat_n(axis, usize::from(count)))
                    .collect();
                if derivative_axes.len() != 1 {
                    return None;
                }
                let axis = derivative_axes[0];
                let (parameter_component, parameter_coefficient) = if axis == mu {
                    (nu, *coefficient)
                } else if axis == nu {
                    (mu, coefficient.mul(&GaussianRational::new(-1, 0)))
                } else {
                    return None;
                };
                if !insert_constraint(
                    &mut constraints,
                    (parameter_component, jet.field),
                    parameter_coefficient,
                ) {
                    return None;
                }
            }
            residuals.insert((mu, nu), residual);
        }
    }

    let mut parameters: [Poly; DIM] = std::array::from_fn(|_| Poly::default());
    for ((component, field), coefficient) in constraints {
        parameters[component].add_scaled(&Poly::atom(field), coefficient);
    }
    for mu in 0..DIM {
        for nu in (mu + 1)..DIM {
            let mut reconstructed = parameters[nu].derivative(mu);
            reconstructed.add_scaled(&parameters[mu].derivative(nu), GaussianRational::new(-1, 0));
            if reconstructed != residuals[&(mu, nu)] {
                return None;
            }
        }
    }
    Some(parameters)
}

type GaugePolynomial = BTreeMap<(usize, [u8; DIM]), Ratio<i64>>;

fn gauge_add_term(
    polynomial: &mut GaugePolynomial,
    component: usize,
    derivatives: [u8; DIM],
    coefficient: Ratio<i64>,
) {
    let key = (component, derivatives);
    let entry = polynomial
        .entry(key)
        .or_insert_with(|| Ratio::from_integer(0));
    *entry += coefficient;
    if *entry == Ratio::from_integer(0) {
        polynomial.remove(&key);
    }
}

fn gauge_derivative(polynomial: &GaugePolynomial, axis: usize) -> GaugePolynomial {
    let mut result = GaugePolynomial::new();
    for (&(component, mut derivatives), coefficient) in polynomial {
        derivatives[axis] += 1;
        gauge_add_term(&mut result, component, derivatives, *coefficient);
    }
    result
}

fn gauge_add_scaled(
    target: &mut GaugePolynomial,
    source: &GaugePolynomial,
    coefficient: Ratio<i64>,
) {
    for (&(component, derivatives), value) in source {
        gauge_add_term(target, component, derivatives, *value * coefficient);
    }
}

fn pure_two_form_gauge(mu: usize, nu: usize) -> GaugePolynomial {
    if mu == nu {
        return GaugePolynomial::new();
    }
    let mut result = GaugePolynomial::new();
    let mut derivative_mu = [0u8; DIM];
    derivative_mu[mu] = 1;
    gauge_add_term(&mut result, nu, derivative_mu, Ratio::from_integer(1));
    let mut derivative_nu = [0u8; DIM];
    derivative_nu[nu] = 1;
    gauge_add_term(&mut result, mu, derivative_nu, Ratio::from_integer(-1));
    result
}

fn pure_two_form_gauge_strength(alpha: usize, mu: usize, nu: usize) -> GaugePolynomial {
    let mut result = gauge_derivative(&pure_two_form_gauge(mu, nu), alpha);
    gauge_add_scaled(
        &mut result,
        &gauge_derivative(&pure_two_form_gauge(nu, alpha), mu),
        Ratio::from_integer(1),
    );
    gauge_add_scaled(
        &mut result,
        &gauge_derivative(&pure_two_form_gauge(alpha, mu), nu),
        Ratio::from_integer(1),
    );
    result
}

fn central_annihilates_two_form_gauge_orbits() -> bool {
    for nu in 0..DIM {
        let mut transformed = GaugePolynomial::new();
        for alpha in 0..DIM {
            for beta in 0..DIM {
                for rho in 0..DIM {
                    let epsilon = epsilon_lower([nu, alpha, beta, rho]);
                    if epsilon == 0 {
                        continue;
                    }
                    let raised = metric_sign(alpha) * metric_sign(beta) * metric_sign(rho);
                    gauge_add_scaled(
                        &mut transformed,
                        &pure_two_form_gauge_strength(alpha, beta, rho),
                        Ratio::new(epsilon * raised, 3),
                    );
                }
            }
        }
        if !transformed.is_empty() {
            return false;
        }
    }
    true
}

fn pure_vector_gauge(mu: usize) -> GaugePolynomial {
    let mut result = GaugePolynomial::new();
    let mut derivatives = [0u8; DIM];
    derivatives[mu] = 1;
    gauge_add_term(&mut result, 0, derivatives, Ratio::from_integer(1));
    result
}

fn pure_vector_gauge_strength(mu: usize, nu: usize) -> GaugePolynomial {
    let mut result = gauge_derivative(&pure_vector_gauge(nu), mu);
    gauge_add_scaled(
        &mut result,
        &gauge_derivative(&pure_vector_gauge(mu), nu),
        Ratio::from_integer(-1),
    );
    result
}

fn central_annihilates_vector_gauge_orbits() -> bool {
    for mu in 0..DIM {
        for nu in (mu + 1)..DIM {
            let mut transformed = GaugePolynomial::new();
            for alpha in 0..DIM {
                for beta in (alpha + 1)..DIM {
                    let epsilon = epsilon_lower([mu, nu, alpha, beta]);
                    if epsilon == 0 {
                        continue;
                    }
                    let raised = metric_sign(alpha) * metric_sign(beta);
                    gauge_add_scaled(
                        &mut transformed,
                        &pure_vector_gauge_strength(alpha, beta),
                        Ratio::new(-epsilon * raised, 2),
                    );
                }
            }
            if !transformed.is_empty() {
                return false;
            }
        }
    }
    true
}

const WORLDLINE_BOSONS: usize = 8;
type RationalMatrix8 = [[Ratio<i64>; WORLDLINE_BOSONS]; WORLDLINE_BOSONS];

fn retained_worldline_index(field: Field) -> Option<usize> {
    match field {
        Field::Scalar => Some(0),
        Field::Vector(1) => Some(1),
        Field::Vector(2) => Some(2),
        Field::Vector(3) => Some(3),
        Field::TwoForm(2, 3) => Some(5),
        Field::TwoForm(1, 3) => Some(6),
        Field::TwoForm(1, 2) => Some(7),
        _ => None,
    }
}

fn retained_worldline_field(index: usize) -> Option<Field> {
    match index {
        0 => Some(Field::Scalar),
        1 => Some(Field::Vector(1)),
        2 => Some(Field::Vector(2)),
        3 => Some(Field::Vector(3)),
        5 => Some(Field::TwoForm(2, 3)),
        6 => Some(Field::TwoForm(1, 3)),
        7 => Some(Field::TwoForm(1, 2)),
        _ => None,
    }
}

fn zero_brane_raw_central_matrix(clifford: &Clifford4D) -> Option<RationalMatrix8> {
    let zero = Ratio::from_integer(0);
    let mut matrix: RationalMatrix8 = std::array::from_fn(|_| std::array::from_fn(|_| zero));
    // Lower the auxiliary by D = partial_t F.
    matrix[0][4] = Ratio::from_integer(1);
    // ZD = -Box phi becomes partial_t(ZF) = partial_t^2 phi.
    matrix[4][0] = Ratio::from_integer(1);

    for source in [1usize, 2, 3, 5, 6, 7] {
        let field = retained_worldline_field(source)?;
        let transformed = central_delta(field, clifford);
        for (jet, coefficient) in &transformed.0 {
            if jet.derivatives[1..].iter().any(|&order| order != 0) {
                continue;
            }
            if coefficient.imag != Ratio::from_integer(0) || jet.derivatives[0] != 1 {
                return None;
            }
            let Some(target) = retained_worldline_index(jet.field) else {
                // Temporal-gauge potentials are zero on the reduced slice.
                if matches!(jet.field, Field::Vector(0) | Field::TwoForm(0, _)) {
                    continue;
                }
                return None;
            };
            matrix[source][target] += coefficient.real;
        }
    }
    Some(matrix)
}

fn normalized_zero_brane_central_matrix(clifford: &Clifford4D) -> Option<[[i16; 8]; 8]> {
    let raw = zero_brane_raw_central_matrix(clifford)?;
    let mut scales: [Ratio<i64>; WORLDLINE_BOSONS] =
        std::array::from_fn(|_| Ratio::from_integer(1));
    for tensor in 5..8 {
        let vector = (1..4).find(|&candidate| raw[candidate][tensor] != Ratio::from_integer(0))?;
        let coefficient = raw[vector][tensor];
        // The sign fixes the magnetic tensor orientation coherently with its
        // paired vector component, while the magnitude fixes normalization.
        scales[tensor] = coefficient;
    }
    let mut normalized = [[0i16; WORLDLINE_BOSONS]; WORLDLINE_BOSONS];
    for source in 0..WORLDLINE_BOSONS {
        for target in 0..WORLDLINE_BOSONS {
            let coefficient = scales[source] * raw[source][target] / scales[target];
            if coefficient == Ratio::from_integer(0) {
                continue;
            }
            if *coefficient.denom() != 1 || coefficient.numer().unsigned_abs() != 1 {
                return None;
            }
            normalized[source][target] = *coefficient.numer() as i16;
        }
    }
    Some(normalized)
}

fn matrix8_multiply(left: &[[i16; 8]; 8], right: &[[i16; 8]; 8]) -> [[i16; 8]; 8] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..8)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn matrix8_transpose(matrix: &[[i16; 8]; 8]) -> [[i16; 8]; 8] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn matrix8_identity() -> [[i16; 8]; 8] {
    std::array::from_fn(|row| std::array::from_fn(|column| i16::from(row == column)))
}

fn anticommutator(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = apply_delta(left, &delta(right, field, clifford), clifford);
    result.add_scaled(
        &apply_delta(right, &delta(left, field, clifford), clifford),
        GaussianRational::new(1, 0),
    );
    result
}

fn add_translation(
    result: &mut Poly,
    left: Charge,
    right: Charge,
    field: Field,
    clifford: &Clifford4D,
) {
    if left.supersymmetry != right.supersymmetry {
        return;
    }
    for rho in 0..DIM {
        let gamma = clifford.lower_spinors(&clifford.gamma_up[rho]);
        result.add_scaled(
            &Poly::atom(field).derivative(rho),
            gamma[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
        );
    }
}

fn central_pair_coefficient(
    left: Charge,
    right: Charge,
    clifford: &Clifford4D,
) -> GaussianRational {
    pauli(2)[left.supersymmetry][right.supersymmetry]
        .mul(&clifford.lower_spinors(&clifford.gamma5)[left.spinor][right.spinor])
        .mul(&GaussianRational::new(0, 2))
}

/// Exact potential-gauge remainder after separating pure translation and the
/// normalized central action.  Each term is stored as the derivative of an
/// explicit gauge parameter:
///
/// * `delta A_nu = partial_nu alpha`, with
///   `alpha = -2 i delta^{ij} gamma^rho A_rho
///            -2 (sigma_2)^{ij} C phi`;
/// * `delta B_mn = partial_m Lambda_n - partial_n Lambda_m`, with
///   `Lambda_n = 2 i delta^{ij} gamma^alpha B_{n alpha}
///               +i delta^{ij} gamma_n phi
///               -(sigma_2)^{ij} C A_n`.
fn explicit_gauge_residue(
    left: Charge,
    right: Charge,
    field: Field,
    clifford: &Clifford4D,
) -> Poly {
    let same_internal = left.supersymmetry == right.supersymmetry;
    let a = left.spinor;
    let b = right.spinor;
    let sigma2 = pauli(2)[left.supersymmetry][right.supersymmetry];
    let charge_conjugation = clifford.charge_conjugation[a][b];
    let mut result = Poly::default();
    match field {
        Field::Vector(nu) => {
            if same_internal {
                for rho in 0..DIM {
                    let gamma = clifford.lower_spinors(&clifford.gamma_up[rho]);
                    result.add_scaled(
                        &Poly::atom(Field::Vector(rho)).derivative(nu),
                        gamma[a][b].mul(&GaussianRational::new(0, -2)),
                    );
                }
            }
            result.add_scaled(
                &Poly::atom(Field::Scalar).derivative(nu),
                sigma2
                    .mul(&charge_conjugation)
                    .mul(&GaussianRational::new(-2, 0)),
            );
        }
        Field::TwoForm(mu, nu) => {
            if same_internal {
                for alpha in 0..DIM {
                    let gamma = clifford.lower_spinors(&clifford.gamma_up[alpha]);
                    result.add_scaled(
                        &two_form_polynomial(nu, alpha).derivative(mu),
                        gamma[a][b].mul(&GaussianRational::new(0, 2)),
                    );
                    result.add_scaled(
                        &two_form_polynomial(mu, alpha).derivative(nu),
                        gamma[a][b].mul(&GaussianRational::new(0, -2)),
                    );
                }
                let gamma_mu = clifford.lower_spinors(&clifford.gamma_down[mu]);
                let gamma_nu = clifford.lower_spinors(&clifford.gamma_down[nu]);
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(mu),
                    gamma_nu[a][b].mul(&GaussianRational::new(0, 1)),
                );
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(nu),
                    gamma_mu[a][b].mul(&GaussianRational::new(0, -1)),
                );
            }
            result.add_scaled(
                &Poly::atom(Field::Vector(nu)).derivative(mu),
                sigma2
                    .mul(&charge_conjugation)
                    .mul(&GaussianRational::new(-1, 0)),
            );
            result.add_scaled(
                &Poly::atom(Field::Vector(mu)).derivative(nu),
                sigma2
                    .mul(&charge_conjugation)
                    .mul(&GaussianRational::new(1, 0)),
            );
        }
        _ => {}
    }
    result
}

fn decomposed_closure(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    add_translation(&mut result, left, right, field, clifford);
    result.add_scaled(
        &central_delta(field, clifford),
        central_pair_coefficient(left, right, clifford),
    );
    result.add_scaled(
        &explicit_gauge_residue(left, right, field, clifford),
        GaussianRational::new(1, 0),
    );
    result
}

fn expected_closure(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let a = left.spinor;
    let b = right.spinor;
    let sigma2 = pauli(2)[left.supersymmetry][right.supersymmetry];
    let gamma5 = clifford.lower_spinors(&clifford.gamma5);
    let mut result = Poly::default();
    match field {
        Field::Scalar => {
            add_translation(&mut result, left, right, field, clifford);
            result.add_scaled(
                &Poly::atom(Field::Auxiliary),
                sigma2.mul(&gamma5[a][b]).mul(&GaussianRational::new(0, 2)),
            );
        }
        Field::Vector(nu) => {
            if left.supersymmetry == right.supersymmetry {
                for rho in 0..DIM {
                    let gamma = clifford.lower_spinors(&clifford.gamma_up[rho]);
                    result.add_scaled(
                        &vector_strength(rho, nu),
                        gamma[a][b].mul(&GaussianRational::new(0, 2)),
                    );
                }
            }
            for alpha in 0..DIM {
                for beta in 0..DIM {
                    for rho in 0..DIM {
                        let epsilon = epsilon_lower([nu, alpha, beta, rho]);
                        if epsilon == 0 {
                            continue;
                        }
                        let raised = metric_sign(alpha) * metric_sign(beta) * metric_sign(rho);
                        result.add_scaled(
                            &tensor_strength(alpha, beta, rho),
                            sigma2.mul(&gamma5[a][b]).mul(&GaussianRational::from_ratio(
                                Ratio::from_integer(0),
                                Ratio::new(2 * epsilon * raised, 3),
                            )),
                        );
                    }
                }
            }
            result.add_scaled(
                &Poly::atom(Field::Scalar).derivative(nu),
                sigma2
                    .mul(&clifford.charge_conjugation[a][b])
                    .mul(&GaussianRational::new(-2, 0)),
            );
        }
        Field::TwoForm(mu, nu) => {
            if left.supersymmetry == right.supersymmetry {
                for alpha in 0..DIM {
                    let gamma = clifford.lower_spinors(&clifford.gamma_up[alpha]);
                    result.add_scaled(
                        &tensor_strength(alpha, mu, nu),
                        gamma[a][b].mul(&GaussianRational::new(0, 2)),
                    );
                }
            }
            result.add_scaled(
                &vector_strength(mu, nu),
                sigma2
                    .mul(&clifford.charge_conjugation[a][b])
                    .mul(&GaussianRational::new(-1, 0)),
            );
            for alpha in 0..DIM {
                for beta in (alpha + 1)..DIM {
                    let epsilon = epsilon_lower([mu, nu, alpha, beta]);
                    if epsilon == 0 {
                        continue;
                    }
                    let raised = metric_sign(alpha) * metric_sign(beta);
                    result.add_scaled(
                        &vector_strength(alpha, beta),
                        sigma2
                            .mul(&gamma5[a][b])
                            .mul(&GaussianRational::new(0, -epsilon * raised)),
                    );
                }
            }
            if left.supersymmetry == right.supersymmetry {
                let gamma_mu = clifford.lower_spinors(&clifford.gamma_down[mu]);
                let gamma_nu = clifford.lower_spinors(&clifford.gamma_down[nu]);
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(nu),
                    gamma_mu[a][b].mul(&GaussianRational::new(0, -1)),
                );
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(mu),
                    gamma_nu[a][b].mul(&GaussianRational::new(0, 1)),
                );
            }
        }
        Field::Lambda(k, c) => {
            add_translation(&mut result, left, right, field, clifford);
            for l in 0..2 {
                let internal = pauli(2)[k][l];
                if internal.is_zero() || sigma2.is_zero() {
                    continue;
                }
                for rho in 0..DIM {
                    let gamma5_gamma = clifford.gamma5_gamma_up(rho);
                    for d in 0..SPINORS {
                        result.add_scaled(
                            &Poly::atom(Field::Lambda(l, d)).derivative(rho),
                            sigma2
                                .mul(&internal)
                                .mul(&gamma5[a][b])
                                .mul(&gamma5_gamma[c][d])
                                .mul(&GaussianRational::new(0, -2)),
                        );
                    }
                }
            }
        }
        Field::Auxiliary => {
            add_translation(&mut result, left, right, field, clifford);
            for rho in 0..DIM {
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(rho).derivative(rho),
                    sigma2
                        .mul(&gamma5[a][b])
                        .mul(&GaussianRational::new(0, -2 * metric_sign(rho))),
                );
            }
        }
    }
    result
}

fn field_label(field: Field) -> String {
    match field {
        Field::Scalar => "phi".into(),
        Field::Vector(mu) => format!("A_{mu}"),
        Field::TwoForm(mu, nu) => format!("B_{mu}{nu}"),
        Field::Auxiliary => "d".into(),
        Field::Lambda(internal, spinor) => format!("lambda_{}_{spinor}", internal + 1),
    }
}

fn canonical_coefficient(value: GaussianRational) -> CanonicalCoefficient {
    CanonicalCoefficient::new(
        *value.real.numer(),
        *value.real.denom(),
        *value.imag.numer(),
        *value.imag.denom(),
    )
    .expect("source Gaussian rational is normalized and finite")
}

fn canonical_lorentz(left: u8, right: u8) -> LorentzRep {
    LorentzRep {
        left_twice_spin: left,
        right_twice_spin: right,
        reality: Reality::Real,
    }
}

/// Exact source-basis fixture for the higher-dimensional canonicalizer.
///
/// This retains both gauge complexes, their curvatures and Bianchi identities,
/// and the complete one-generator central extension.  It is intentionally
/// built in this module so no term-level closure data are duplicated.
pub(crate) fn exact_canonical_fixture() -> CanonicalPhysicalFingerprint {
    let clifford = Clifford4D::build();
    let fields = Field::all();
    let charges: Vec<_> = (0..2)
        .flat_map(|supersymmetry| {
            (0..SPINORS).map(move |spinor| Charge {
                supersymmetry,
                spinor,
            })
        })
        .collect();

    let mut components = Vec::new();
    for &field in &fields {
        let (statistics, lorentz, height_twice, role, form_degree) = match field {
            Field::Scalar => (
                Statistics::Boson,
                canonical_lorentz(0, 0),
                0,
                CanonicalRole::Propagating,
                None,
            ),
            Field::Vector(_) => (
                Statistics::Boson,
                canonical_lorentz(1, 1),
                0,
                CanonicalRole::GaugePotential,
                Some(1),
            ),
            Field::TwoForm(_, _) => (
                Statistics::Boson,
                canonical_lorentz(2, 0),
                0,
                CanonicalRole::GaugePotential,
                Some(2),
            ),
            Field::Auxiliary => (
                Statistics::Boson,
                canonical_lorentz(0, 0),
                2,
                CanonicalRole::Auxiliary,
                None,
            ),
            Field::Lambda(_, _) => (
                Statistics::Fermion,
                canonical_lorentz(1, 0),
                1,
                CanonicalRole::Propagating,
                None,
            ),
        };
        components.push(CanonicalComponent {
            label: field_label(field),
            statistics,
            lorentz,
            height_twice,
            role,
            form_degree,
        });
    }
    let field_indices: BTreeMap<_, _> = fields
        .iter()
        .enumerate()
        .map(|(index, &field)| (field, index))
        .collect();

    let vector_gauge_parameter = components.len();
    components.push(CanonicalComponent {
        label: "alpha".into(),
        statistics: Statistics::Boson,
        lorentz: canonical_lorentz(0, 0),
        height_twice: -2,
        role: CanonicalRole::GaugeParameter { stage: 0 },
        form_degree: Some(0),
    });
    let mut tensor_gauge_parameters = Vec::new();
    for mu in 0..DIM {
        tensor_gauge_parameters.push(components.len());
        components.push(CanonicalComponent {
            label: format!("Lambda_{mu}"),
            statistics: Statistics::Boson,
            lorentz: canonical_lorentz(1, 1),
            height_twice: -2,
            role: CanonicalRole::GaugeParameter { stage: 0 },
            form_degree: Some(1),
        });
    }
    let tensor_reducibility_parameter = components.len();
    components.push(CanonicalComponent {
        label: "tensor_gauge_for_gauge".into(),
        statistics: Statistics::Boson,
        lorentz: canonical_lorentz(0, 0),
        height_twice: -4,
        role: CanonicalRole::GaugeParameter { stage: 1 },
        form_degree: Some(0),
    });

    let pairs: Vec<_> = (0..DIM)
        .flat_map(|mu| ((mu + 1)..DIM).map(move |nu| (mu, nu)))
        .collect();
    let triples: Vec<_> = (0..DIM)
        .flat_map(|mu| {
            ((mu + 1)..DIM).flat_map(move |nu| ((nu + 1)..DIM).map(move |rho| (mu, nu, rho)))
        })
        .collect();
    let mut vector_strength_indices = BTreeMap::new();
    for &(mu, nu) in &pairs {
        vector_strength_indices.insert((mu, nu), components.len());
        components.push(CanonicalComponent {
            label: format!("F_{mu}{nu}"),
            statistics: Statistics::Boson,
            lorentz: canonical_lorentz(2, 0),
            height_twice: 2,
            role: CanonicalRole::FieldStrength,
            form_degree: Some(2),
        });
    }
    let mut tensor_strength_indices = BTreeMap::new();
    for &(mu, nu, rho) in &triples {
        tensor_strength_indices.insert((mu, nu, rho), components.len());
        components.push(CanonicalComponent {
            label: format!("H_{mu}{nu}{rho}"),
            statistics: Statistics::Boson,
            lorentz: canonical_lorentz(1, 1),
            height_twice: 2,
            role: CanonicalRole::FieldStrength,
            form_degree: Some(3),
        });
    }

    let supercharges = charges
        .iter()
        .map(|charge| CanonicalSupercharge {
            label: format!("D{}_{}", charge.supersymmetry + 1, charge.spinor),
            lorentz: canonical_lorentz(1, 0),
            height_twice: 1,
        })
        .collect();

    let mut linkage = Vec::new();
    for (charge_index, &charge) in charges.iter().enumerate() {
        for &source_field in &fields {
            for (atom, coefficient) in delta(charge, source_field, &clifford).0 {
                linkage.push(CanonicalLinkage {
                    charge: charge_index,
                    source: field_indices[&source_field],
                    target: field_indices[&atom.field],
                    derivative: CanonicalDerivative(atom.derivatives),
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
        for &(mu, nu) in &pairs {
            for (atom, coefficient) in apply_delta(charge, &vector_strength(mu, nu), &clifford).0 {
                linkage.push(CanonicalLinkage {
                    charge: charge_index,
                    source: vector_strength_indices[&(mu, nu)],
                    target: field_indices[&atom.field],
                    derivative: CanonicalDerivative(atom.derivatives),
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
        for &(mu, nu, rho) in &triples {
            for (atom, coefficient) in
                apply_delta(charge, &tensor_strength(mu, nu, rho), &clifford).0
            {
                linkage.push(CanonicalLinkage {
                    charge: charge_index,
                    source: tensor_strength_indices[&(mu, nu, rho)],
                    target: field_indices[&atom.field],
                    derivative: CanonicalDerivative(atom.derivatives),
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
    }

    let mut gauge_complex = Vec::new();
    for mu in 0..DIM {
        gauge_complex.push(CanonicalGaugeTerm {
            parameter: vector_gauge_parameter,
            target: field_indices[&Field::Vector(mu)],
            derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == mu))),
            coefficient: CanonicalCoefficient::integer(1, 0),
        });
        gauge_complex.push(CanonicalGaugeTerm {
            parameter: tensor_reducibility_parameter,
            target: tensor_gauge_parameters[mu],
            derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == mu))),
            coefficient: CanonicalCoefficient::integer(1, 0),
        });
        for nu in (mu + 1)..DIM {
            let target = field_indices[&Field::TwoForm(mu, nu)];
            gauge_complex.push(CanonicalGaugeTerm {
                parameter: tensor_gauge_parameters[nu],
                target,
                derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == mu))),
                coefficient: CanonicalCoefficient::integer(1, 0),
            });
            gauge_complex.push(CanonicalGaugeTerm {
                parameter: tensor_gauge_parameters[mu],
                target,
                derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == nu))),
                coefficient: CanonicalCoefficient::integer(-1, 0),
            });
        }
    }

    let mut bianchi_identities = Vec::new();
    for &(mu, nu, rho) in &triples {
        bianchi_identities.push(CanonicalBianchi {
            terms: vec![
                CanonicalLinearTerm {
                    component: vector_strength_indices[&(nu, rho)],
                    derivative: CanonicalDerivative(std::array::from_fn(|axis| {
                        u8::from(axis == mu)
                    })),
                    coefficient: CanonicalCoefficient::integer(1, 0),
                },
                CanonicalLinearTerm {
                    component: vector_strength_indices[&(mu, rho)],
                    derivative: CanonicalDerivative(std::array::from_fn(|axis| {
                        u8::from(axis == nu)
                    })),
                    coefficient: CanonicalCoefficient::integer(-1, 0),
                },
                CanonicalLinearTerm {
                    component: vector_strength_indices[&(mu, nu)],
                    derivative: CanonicalDerivative(std::array::from_fn(|axis| {
                        u8::from(axis == rho)
                    })),
                    coefficient: CanonicalCoefficient::integer(1, 0),
                },
            ],
        });
    }
    bianchi_identities.push(CanonicalBianchi {
        terms: vec![
            CanonicalLinearTerm {
                component: tensor_strength_indices[&(1, 2, 3)],
                derivative: CanonicalDerivative([1, 0, 0, 0]),
                coefficient: CanonicalCoefficient::integer(1, 0),
            },
            CanonicalLinearTerm {
                component: tensor_strength_indices[&(0, 2, 3)],
                derivative: CanonicalDerivative([0, 1, 0, 0]),
                coefficient: CanonicalCoefficient::integer(-1, 0),
            },
            CanonicalLinearTerm {
                component: tensor_strength_indices[&(0, 1, 3)],
                derivative: CanonicalDerivative([0, 0, 1, 0]),
                coefficient: CanonicalCoefficient::integer(1, 0),
            },
            CanonicalLinearTerm {
                component: tensor_strength_indices[&(0, 1, 2)],
                derivative: CanonicalDerivative([0, 0, 0, 1]),
                coefficient: CanonicalCoefficient::integer(-1, 0),
            },
        ],
    });

    let central_generators = vec![CanonicalCentralGenerator {
        label: "Z".into(),
        lorentz: canonical_lorentz(0, 0),
        height_twice: 2,
    }];
    let mut central_entries = Vec::new();
    for &source_field in &fields {
        if matches!(source_field, Field::Vector(_) | Field::TwoForm(_, _)) {
            continue;
        }
        for (atom, coefficient) in central_delta(source_field, &clifford).0 {
            central_entries.push(CanonicalCentralEntry {
                generator: 0,
                source: field_indices[&source_field],
                target: field_indices[&atom.field],
                derivative: CanonicalDerivative(atom.derivatives),
                coefficient: canonical_coefficient(coefficient),
            });
        }
    }
    // Store the potential action in the independent curvature basis rather
    // than duplicating F/H as derivatives of A/B. After antisymmetric
    // aggregation each vector maps to one complementary H and each two-form
    // maps to one complementary F.
    for nu in 0..DIM {
        let (alpha, beta, rho) = triples
            .iter()
            .copied()
            .find(|&(alpha, beta, rho)| ![alpha, beta, rho].contains(&nu))
            .expect("one complementary three-form component");
        let epsilon = epsilon_lower([nu, alpha, beta, rho]);
        let raised = metric_sign(alpha) * metric_sign(beta) * metric_sign(rho);
        central_entries.push(CanonicalCentralEntry {
            generator: 0,
            source: field_indices[&Field::Vector(nu)],
            target: tensor_strength_indices[&(alpha, beta, rho)],
            derivative: CanonicalDerivative::IDENTITY,
            coefficient: CanonicalCoefficient::integer(2 * epsilon * raised, 0),
        });
    }
    for &(mu, nu) in &pairs {
        let (alpha, beta) = pairs
            .iter()
            .copied()
            .find(|&(alpha, beta)| ![alpha, beta].contains(&mu) && ![alpha, beta].contains(&nu))
            .expect("one complementary two-form component");
        let epsilon = epsilon_lower([mu, nu, alpha, beta]);
        let raised = metric_sign(alpha) * metric_sign(beta);
        central_entries.push(CanonicalCentralEntry {
            generator: 0,
            source: field_indices[&Field::TwoForm(mu, nu)],
            target: vector_strength_indices[&(alpha, beta)],
            derivative: CanonicalDerivative::IDENTITY,
            coefficient: CanonicalCoefficient::new(-epsilon * raised, 2, 0, 1)
                .expect("nonzero rational denominator"),
        });
    }
    for &(mu, nu) in &pairs {
        for (sign, derivative, vector) in [(1, mu, nu), (-1, nu, mu)] {
            let (alpha, beta, rho) = triples
                .iter()
                .copied()
                .find(|&(alpha, beta, rho)| ![alpha, beta, rho].contains(&vector))
                .expect("one complementary three-form component");
            let epsilon = epsilon_lower([vector, alpha, beta, rho]);
            let raised = metric_sign(alpha) * metric_sign(beta) * metric_sign(rho);
            central_entries.push(CanonicalCentralEntry {
                generator: 0,
                source: vector_strength_indices[&(mu, nu)],
                target: tensor_strength_indices[&(alpha, beta, rho)],
                derivative: CanonicalDerivative(std::array::from_fn(|axis| {
                    u8::from(axis == derivative)
                })),
                coefficient: CanonicalCoefficient::integer(sign * 2 * epsilon * raised, 0),
            });
        }
    }
    for &(mu, nu, rho) in &triples {
        for (derivative, first, second, orientation) in
            [(mu, nu, rho, 1), (nu, rho, mu, 1), (rho, mu, nu, 1)]
        {
            let (first, second, pair_sign) = if first < second {
                (first, second, 1)
            } else {
                (second, first, -1)
            };
            let (alpha, beta) = pairs
                .iter()
                .copied()
                .find(|&(alpha, beta)| {
                    ![alpha, beta].contains(&first) && ![alpha, beta].contains(&second)
                })
                .expect("one complementary two-form component");
            let epsilon = epsilon_lower([first, second, alpha, beta]);
            let raised = metric_sign(alpha) * metric_sign(beta);
            central_entries.push(CanonicalCentralEntry {
                generator: 0,
                source: tensor_strength_indices[&(mu, nu, rho)],
                target: vector_strength_indices[&(alpha, beta)],
                derivative: CanonicalDerivative(std::array::from_fn(|axis| {
                    u8::from(axis == derivative)
                })),
                coefficient: CanonicalCoefficient::new(
                    -orientation * pair_sign * epsilon * raised,
                    2,
                    0,
                    1,
                )
                .expect("nonzero rational denominator"),
            });
        }
    }
    let mut central_occurrences = Vec::new();
    for left in 0..charges.len() {
        for right in left..charges.len() {
            let coefficient = central_pair_coefficient(charges[left], charges[right], &clifford);
            if !coefficient.is_zero() {
                central_occurrences.push(CanonicalCentralOccurrence {
                    left_charge: left,
                    right_charge: right,
                    generator: 0,
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
    }

    CanonicalPhysicalFingerprint {
        name: "vector-tensor-exact-source-adapter".into(),
        components,
        supercharges,
        linkage,
        gauge_complex,
        bianchi_identities,
        central_generators,
        central_entries,
        central_occurrences,
    }
}

fn transform_monomial_operator(
    source: &RationalMatrix8,
    source_index_for_target: [usize; 8],
    target_scales: [Ratio<i64>; 8],
) -> RationalMatrix8 {
    std::array::from_fn(|target_source| {
        std::array::from_fn(|target_target| {
            target_scales[target_source]
                * source[source_index_for_target[target_source]]
                    [source_index_for_target[target_target]]
                / target_scales[target_target]
        })
    })
}

fn integral_matrix(source: &RationalMatrix8) -> Option<[[i16; 8]; 8]> {
    let mut result = [[0i16; 8]; 8];
    for row in 0..8 {
        for column in 0..8 {
            if *source[row][column].denom() != 1 {
                return None;
            }
            result[row][column] = i16::try_from(*source[row][column].numer()).ok()?;
        }
    }
    Some(result)
}

/// Eq. (4.6) in the 960 source, transformed through the fixed source map
/// `B_1405=B_960/6`, `d_1405=-D_960`, and the published Phi definitions.
fn source_960_bosonic_operator(
    two_form_map: Ratio<i64>,
    auxiliary_map_sign: i64,
) -> Option<[[i16; 8]; 8]> {
    let zero = Ratio::from_integer(0);
    // Source order: phi,V1,V2,V3,F,B23,B31,B12, with dot(F)=D.
    let mut source: RationalMatrix8 = std::array::from_fn(|_| std::array::from_fn(|_| zero));
    source[0][4] = Ratio::from_integer(1);
    source[4][0] = Ratio::from_integer(1);
    for (vector, tensor) in [(1, 5), (2, 6), (3, 7)] {
        source[vector][tensor] = Ratio::new(1, 3);
        source[tensor][vector] = Ratio::from_integer(3);
    }
    // Phi=(phi,2B12,2B23,2B31,A1,A2,A3,Phi8).  Since
    // B_1405=two_form_map*B_960, the tensor scale is 2*two_form_map.
    // Since d_1405=-D_960 and dot(Phi8)=d_1405, Phi8=-F.
    let indices = [0, 7, 5, 6, 1, 2, 3, 4];
    let tensor_scale = Ratio::from_integer(2) * two_form_map;
    let scales = [
        Ratio::from_integer(1),
        tensor_scale,
        tensor_scale,
        tensor_scale,
        Ratio::from_integer(1),
        Ratio::from_integer(1),
        Ratio::from_integer(1),
        Ratio::from_integer(auxiliary_map_sign),
    ];
    integral_matrix(&transform_monomial_operator(&source, indices, scales))
}

fn component_bosonic_operator_in_phi_basis(clifford: &Clifford4D) -> Option<[[i16; 8]; 8]> {
    // Raw order from zero_brane_raw_central_matrix:
    // phi,A1,A2,A3,Phi8,B23,B13,B12.
    let raw = zero_brane_raw_central_matrix(clifford)?;
    let indices = [0, 7, 5, 6, 1, 2, 3, 4];
    let scales = [
        Ratio::from_integer(1),
        Ratio::from_integer(2),
        Ratio::from_integer(2),
        Ratio::from_integer(-2),
        Ratio::from_integer(1),
        Ratio::from_integer(1),
        Ratio::from_integer(1),
        Ratio::from_integer(1),
    ];
    integral_matrix(&transform_monomial_operator(&raw, indices, scales))
}

fn component_fermionic_zero_brane_operator(clifford: &Clifford4D) -> Option<[[i16; 8]; 8]> {
    let mut result = [[0i16; 8]; 8];
    for source_internal in 0..2 {
        for source_spinor in 0..4 {
            let source = 4 * source_internal + source_spinor;
            for (jet, coefficient) in
                central_delta(Field::Lambda(source_internal, source_spinor), clifford).0
            {
                if jet.derivatives[1..].iter().any(|&order| order != 0) {
                    continue;
                }
                if jet.derivatives != [1, 0, 0, 0] || coefficient.imag != Ratio::from_integer(0) {
                    return None;
                }
                let Field::Lambda(target_internal, target_spinor) = jet.field else {
                    return None;
                };
                if *coefficient.real.denom() != 1 {
                    return None;
                }
                result[source][4 * target_internal + target_spinor] =
                    i16::try_from(*coefficient.real.numer()).ok()?;
            }
        }
    }
    Some(result)
}

fn source_960_fermionic_operator() -> [[i16; 8]; 8] {
    let epsilon_upper = [[0i16, 1], [-1, 0]];
    let temporal_spinor = [[0i16, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]];
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            epsilon_upper[row / 4][column / 4] * temporal_spinor[row % 4][column % 4]
        })
    })
}

fn weyl_phase_realification_verified() -> bool {
    // W=((1+i)/2) [[1,0,0,i],[0,1,-i,0]] maps each real Majorana
    // spinor to the two phased Weyl components. Eq. (4.6) at p_i=0 requires
    // W M = T conjugate(W), T=epsilon_lower. The phase turns the unphased
    // imaginary Weyl map into this real antisymmetric matrix.
    let half = Ratio::new(1, 2);
    let w = [
        [
            GaussianRational::from_ratio(half, half),
            GaussianRational::new(0, 0),
            GaussianRational::new(0, 0),
            GaussianRational::from_ratio(-half, half),
        ],
        [
            GaussianRational::new(0, 0),
            GaussianRational::from_ratio(half, half),
            GaussianRational::from_ratio(half, -half),
            GaussianRational::new(0, 0),
        ],
    ];
    let m = [[0i64, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]];
    let t = [
        [GaussianRational::new(0, 0), GaussianRational::new(1, 0)],
        [GaussianRational::new(-1, 0), GaussianRational::new(0, 0)],
    ];
    for row in 0..2 {
        for column in 0..4 {
            let mut left = GaussianRational::default();
            for inner in 0..4 {
                left.add_assign(&w[row][inner].mul(&GaussianRational::new(m[inner][column], 0)));
            }
            let mut right = GaussianRational::default();
            for inner in 0..2 {
                let conjugated =
                    GaussianRational::from_ratio(w[inner][column].real, -w[inner][column].imag);
                right.add_assign(&t[row][inner].mul(&conjugated));
            }
            if left != right {
                return false;
            }
        }
    }
    true
}

fn component_color_operator(clifford: &Clifford4D) -> Option<[[i16; 8]; 8]> {
    let charges: Vec<_> = (0..2)
        .flat_map(|supersymmetry| {
            (0..4).map(move |spinor| Charge {
                supersymmetry,
                spinor,
            })
        })
        .collect();
    let mut result = [[0i16; 8]; 8];
    for left in 0..8 {
        for right in 0..8 {
            let coefficient = central_pair_coefficient(charges[left], charges[right], clifford);
            // Divide by the common 2i used in the worldline closure convention.
            let real = coefficient.imag / Ratio::from_integer(2);
            let imaginary = -coefficient.real / Ratio::from_integer(2);
            if imaginary != Ratio::from_integer(0) || *real.denom() != 1 {
                return None;
            }
            result[left][right] = i16::try_from(*real.numer()).ok()?;
        }
    }
    Some(result)
}

fn negated_matrix(matrix: &[[i16; 8]; 8]) -> [[i16; 8]; 8] {
    std::array::from_fn(|row| std::array::from_fn(|column| -matrix[row][column]))
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensorCentralBridgeReport {
    pub schema_version: &'static str,
    pub source_field_map: Vec<&'static str>,
    pub zero_mode_policy: &'static str,
    pub residual_gauge_policy: &'static str,
    pub bosonic_eq46_entries_checked: usize,
    pub fermionic_eq46_entries_checked: usize,
    pub omega_entries_checked: usize,
    pub fixed_one_sixth_normalization_matched: bool,
    pub auxiliary_sign_matched: bool,
    pub gauge_orbits_preserved: bool,
    pub residual_temporal_gauge_action_exact: bool,
    pub temporal_weyl_realification_exact: bool,
    pub simultaneous_z_omega_orientation_exact: bool,
    pub narrow_zero_brane_bridge_passed: bool,
    pub repaired_eq45_reduced_to_appendix_f: bool,
    pub full_four_dimensional_source_bridge_passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidualGaugeVariation {
    vector_temporal: i64,
    vector_spatial: [i64; 3],
    two_form_temporal: [i64; 3],
    two_form_spatial: [i64; 3],
}

fn residual_gauge_variation(
    spatial_momentum: [i64; 3],
    alpha: i64,
    alpha_time_derivative: i64,
    lambda_temporal: i64,
    lambda_spatial: [i64; 3],
    lambda_spatial_time_derivative: [i64; 3],
) -> ResidualGaugeVariation {
    let [p1, p2, p3] = spatial_momentum;
    let [l1, l2, l3] = lambda_spatial;
    ResidualGaugeVariation {
        vector_temporal: alpha_time_derivative,
        vector_spatial: [p1 * alpha, p2 * alpha, p3 * alpha],
        two_form_temporal: [
            lambda_spatial_time_derivative[0] - p1 * lambda_temporal,
            lambda_spatial_time_derivative[1] - p2 * lambda_temporal,
            lambda_spatial_time_derivative[2] - p3 * lambda_temporal,
        ],
        // Ordered as B_23, B_31, B_12.
        two_form_spatial: [p2 * l3 - p3 * l2, p3 * l1 - p1 * l3, p1 * l2 - p2 * l1],
    }
}

fn residual_temporal_gauge_action_is_zero() -> bool {
    let zero = ResidualGaugeVariation {
        vector_temporal: 0,
        vector_spatial: [0; 3],
        two_form_temporal: [0; 3],
        two_form_spatial: [0; 3],
    };
    // Check a basis of the residual alpha, Lambda_0, and Lambda_i parameters.
    (0..5).all(|parameter| {
        let alpha = i64::from(parameter == 0);
        let lambda_temporal = i64::from(parameter == 1);
        let mut lambda_spatial = [0; 3];
        if parameter >= 2 {
            lambda_spatial[parameter - 2] = 1;
        }
        residual_gauge_variation([0; 3], alpha, 0, lambda_temporal, lambda_spatial, [0; 3]) == zero
    })
}

fn verify_central_bridge() -> VectorTensorCentralBridgeReport {
    let clifford = Clifford4D::build();
    let artifact = crate::vector_tensor_central_charge::build();
    let branch = artifact.sectors[4]
        .branches
        .iter()
        .find(|branch| branch.m_mod_4 == Some(1) && branch.n_mod_4 == Some(0))
        .expect("TV m=1,n=0 branch");
    let committed = branch.central_charge.as_ref().expect("one-Z branch");
    let source_bosonic = source_960_bosonic_operator(Ratio::new(1, 6), -1);
    let component_bosonic = component_bosonic_operator_in_phi_basis(&clifford);
    let source_fermionic = source_960_fermionic_operator();
    let component_fermionic = component_fermionic_zero_brane_operator(&clifford);
    let component_color = component_color_operator(&clifford);
    let fixed_one_sixth_normalization_matched = source_bosonic == Some(committed.bosonic)
        && source_960_bosonic_operator(Ratio::from_integer(1), -1) != Some(committed.bosonic);
    let auxiliary_sign_matched =
        source_960_bosonic_operator(Ratio::new(1, 6), 1) != Some(committed.bosonic);
    let simultaneous_z_omega_orientation_exact = component_bosonic
        == Some(negated_matrix(&committed.bosonic))
        && component_fermionic == Some(negated_matrix(&committed.fermionic))
        && component_color == Some(negated_matrix(&committed.color_coefficient_matrix));
    let gauge_orbits_preserved =
        central_annihilates_vector_gauge_orbits() && central_annihilates_two_form_gauge_orbits();
    let residual_temporal_gauge_action_exact = residual_temporal_gauge_action_is_zero();
    let temporal_weyl_realification_exact =
        weyl_phase_realification_verified() && source_fermionic == committed.fermionic;
    let narrow_zero_brane_bridge_passed = fixed_one_sixth_normalization_matched
        && auxiliary_sign_matched
        && gauge_orbits_preserved
        && residual_temporal_gauge_action_exact
        && temporal_weyl_realification_exact
        && simultaneous_z_omega_orientation_exact;
    VectorTensorCentralBridgeReport {
        schema_version: "vector-tensor-central-bridge-v1",
        source_field_map: vec![
            "phi_1405=phi_960",
            "A_mu_1405=V_mu_960",
            "B_mu_nu_1405=B_mu_nu_960/6",
            "d_1405=-D_960",
            "Phi=(phi,2B12,2B23,2B31,A1,A2,A3,Phi8), dot(Phi8)=d",
        ],
        zero_mode_policy: "Auxiliary lowering is defined on nonzero temporal modes, equivalently modulo the time-independent integration constant.",
        residual_gauge_policy: "Use strict zero spatial momentum and temporal gauge A0=B0i=0; residual time-independent gauge parameters act trivially on retained nodes.",
        bosonic_eq46_entries_checked: 64,
        fermionic_eq46_entries_checked: 64,
        omega_entries_checked: 64,
        fixed_one_sixth_normalization_matched,
        auxiliary_sign_matched,
        gauge_orbits_preserved,
        residual_temporal_gauge_action_exact,
        temporal_weyl_realification_exact,
        simultaneous_z_omega_orientation_exact,
        narrow_zero_brane_bridge_passed,
        repaired_eq45_reduced_to_appendix_f: false,
        full_four_dimensional_source_bridge_passed: false,
        boundary: "The exact Eq. (4.6) central bridge and its Omega orientation pass. A direct 512-entry reduction of repaired Eq. (4.5), including the coherent spatial frame, remains required for a full source-equivalence claim.",
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensor4DReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub branch: &'static str,
    pub gamma_conventions_verified: bool,
    pub fields: usize,
    pub supercharges: usize,
    pub unordered_charge_pairs: usize,
    pub component_relations_checked: usize,
    pub scalar_relations_checked: usize,
    pub vector_potential_relations_checked: usize,
    pub two_form_potential_relations_checked: usize,
    pub auxiliary_relations_checked: usize,
    pub fermion_relations_checked: usize,
    pub residual_relations: usize,
    pub residual_terms: usize,
    pub scalar_residual_relations: usize,
    pub vector_residual_relations: usize,
    pub two_form_residual_relations: usize,
    pub auxiliary_residual_relations: usize,
    pub fermion_residual_relations: usize,
    pub corrected_scalar_typo_used: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
    pub pdf_sha256: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensor4DArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub branch_coefficients: Vec<&'static str>,
    pub source_corrections: Vec<&'static str>,
    pub convention_translation: Vec<&'static str>,
    pub report: VectorTensor4DReport,
    pub central_bridge: VectorTensorCentralBridgeReport,
}

pub fn verify() -> VectorTensor4DReport {
    let clifford = Clifford4D::build();
    let fields = Field::all();
    let charges: Vec<_> = (0..2)
        .flat_map(|supersymmetry| {
            (0..SPINORS).map(move |spinor| Charge {
                supersymmetry,
                spinor,
            })
        })
        .collect();
    let mut checked = 0usize;
    let mut residual_relations = 0usize;
    let mut residual_terms = 0usize;
    let mut residual_by_kind = [0usize; 5];
    for left in 0..charges.len() {
        for right in left..charges.len() {
            for &field in &fields {
                checked += 1;
                let actual = anticommutator(charges[left], charges[right], field, &clifford);
                let expected = expected_closure(charges[left], charges[right], field, &clifford);
                if actual != expected {
                    residual_relations += 1;
                    let kind = match field {
                        Field::Scalar => 0,
                        Field::Vector(_) => 1,
                        Field::TwoForm(_, _) => 2,
                        Field::Auxiliary => 3,
                        Field::Lambda(_, _) => 4,
                    };
                    residual_by_kind[kind] += 1;
                    let mut residual = actual;
                    residual.add_scaled(&expected, GaussianRational::new(-1, 0));
                    residual_terms += residual.term_count();
                }
            }
        }
    }
    VectorTensor4DReport {
        schema_version: "vector-tensor-4d-closure-v1",
        source: "arXiv:1405.0048 Eqs. (59)-(61) and (76)-(84)",
        branch: "m=1, n=0",
        gamma_conventions_verified: clifford.verifies_source_conventions(),
        fields: fields.len(),
        supercharges: charges.len(),
        unordered_charge_pairs: charges.len() * (charges.len() + 1) / 2,
        component_relations_checked: checked,
        scalar_relations_checked: 36,
        vector_potential_relations_checked: 4 * 36,
        two_form_potential_relations_checked: 6 * 36,
        auxiliary_relations_checked: 36,
        fermion_relations_checked: 8 * 36,
        residual_relations,
        residual_terms,
        scalar_residual_relations: residual_by_kind[0],
        vector_residual_relations: residual_by_kind[1],
        two_form_residual_relations: residual_by_kind[2],
        auxiliary_residual_relations: residual_by_kind[3],
        fermion_residual_relations: residual_by_kind[4],
        corrected_scalar_typo_used: true,
        passed: clifford.verifies_source_conventions() && checked == 720 && residual_relations == 0,
        boundary: "This checks the sourced 4D transformation composition, including the printed gauge residues and extension terms. It does not independently identify those terms with the hep-th/9609016 central-coordinate operator.",
    }
}

pub fn build() -> VectorTensor4DArtifact {
    VectorTensor4DArtifact {
        schema_version: "vector-tensor-4d-artifact-v1",
        title: "Exact four-dimensional vector-tensor component closure",
        sources: vec![
            SourceRecord {
                arxiv_id: "1405.0048",
                locator: "Sec. 3.6, Eqs. (59)-(61) and (76)-(84)",
                role: "component transformations and closure ledger",
                pdf_sha256: "8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20",
            },
            SourceRecord {
                arxiv_id: "hep-th/9609016",
                locator: "Sec. 4, repaired Eq. (4.5) and Eq. (4.6)",
                role: "intrinsic central-coordinate action and source field normalization",
                pdf_sha256: "3bf2549954d01e4da9e1fedc4af8e3a534bf17df3c6a27d45bdc8a9a80b8c2af",
            },
        ],
        branch_coefficients: vec![
            "m=1, n=0",
            "a=i sigma_2",
            "b=identity_2",
            "c1=0",
            "s1=1",
            "c2_plus=0",
            "c2_minus=-2",
        ],
        source_corrections: vec![
            "Eq. (78) scalar extension uses d, not the printed uncontracted partial_mu d",
            "Eq. (80) second row is Phi_5 through Phi_8, not repeated Phi_2 through Phi_4 labels",
        ],
        convention_translation: vec![
            "epsilon_lower_0123=-1 and mostly-plus metric",
            "the dual-H and dual-F signs are converted to the stored Clifford4D epsilon convention",
            "antisymmetric tensor pairs are stored once and their summed coefficients are doubled",
        ],
        report: verify(),
        central_bridge: verify_central_bridge(),
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> VectorTensor4DReport {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create vector-tensor data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create vector-tensor result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create vector-tensor artifact")),
        &artifact,
    )
    .expect("write vector-tensor artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create vector-tensor validation")),
        &artifact.report,
    )
    .expect("write vector-tensor validation");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrected_equation_78_closes_exactly() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
    }

    #[test]
    fn closure_splits_into_translation_explicit_gauge_and_one_central_operator() {
        let clifford = Clifford4D::build();
        let charges: Vec<_> = (0..2)
            .flat_map(|supersymmetry| {
                (0..SPINORS).map(move |spinor| Charge {
                    supersymmetry,
                    spinor,
                })
            })
            .collect();
        let mut checked = 0usize;
        for left in 0..charges.len() {
            for right in left..charges.len() {
                for field in Field::all() {
                    assert_eq!(
                        expected_closure(charges[left], charges[right], field, &clifford),
                        decomposed_closure(charges[left], charges[right], field, &clifford),
                        "decomposition failed for charges ({left},{right}) on {field:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 720);
    }

    #[test]
    fn central_operator_commutes_with_all_supercharges_modulo_explicit_gauge_maps() {
        let clifford = Clifford4D::build();
        let charges: Vec<_> = (0..2)
            .flat_map(|supersymmetry| {
                (0..SPINORS).map(move |spinor| Charge {
                    supersymmetry,
                    spinor,
                })
            })
            .collect();
        let mut exact_relations = 0usize;
        let mut vector_gauge_certificates = 0usize;
        let mut two_form_gauge_certificates = 0usize;
        for (charge_index, &charge) in charges.iter().enumerate() {
            for field in Field::all() {
                if !matches!(field, Field::Vector(_) | Field::TwoForm(_, _)) {
                    let commutator = central_supercharge_commutator(charge, field, &clifford);
                    assert_eq!(
                        commutator.term_count(),
                        0,
                        "[Z,Q_{charge_index}] failed on {field:?}: {:?}",
                        commutator.0
                    );
                    exact_relations += 1;
                }
            }
            assert!(vector_commutator_gauge_parameter(charge, &clifford).is_some());
            vector_gauge_certificates += 1;
            assert!(two_form_commutator_gauge_parameters(charge, &clifford).is_some());
            two_form_gauge_certificates += 1;
        }
        assert_eq!(exact_relations, 80);
        assert_eq!(vector_gauge_certificates, 8);
        assert_eq!(two_form_gauge_certificates, 8);
    }

    #[test]
    fn central_operator_is_well_defined_on_both_gauge_orbit_spaces() {
        assert!(central_annihilates_vector_gauge_orbits());
        assert!(central_annihilates_two_form_gauge_orbits());
    }

    #[test]
    fn exact_zero_brane_normalization_reaches_the_committed_physical_node_operator() {
        let clifford = Clifford4D::build();
        let reduced = normalized_zero_brane_central_matrix(&clifford)
            .expect("solve coherent scalar/vector/tensor normalization");
        assert_eq!(matrix8_multiply(&reduced, &reduced), matrix8_identity());
        assert_eq!(reduced, matrix8_transpose(&reduced));

        let central_artifact = crate::vector_tensor_central_charge::build();
        assert_eq!(
            reduced,
            central_artifact
                .physical_worldline_map
                .canonical_node_operator
        );
        let branch = central_artifact.sectors[4]
            .branches
            .iter()
            .find(|branch| branch.m_mod_4 == Some(1) && branch.n_mod_4 == Some(0))
            .expect("TV m=1,n=0 branch");
        let central = branch.central_charge.as_ref().expect("one-Z branch");
        assert!(central.extended_closure_passed);
        assert!(central.source_omega_equation_81_matched);
        assert!(central.physical_bosonic_conjugator.is_some());
    }

    #[test]
    fn residual_temporal_gauge_freedom_acts_trivially_on_retained_nodes() {
        assert!(residual_temporal_gauge_action_is_zero());

        // A spatial-momentum mutation must be visible on retained nodes, so
        // the zero result above is an evaluated gauge action rather than a
        // boolean restatement of the temporal-gauge assumptions.
        let mutated = residual_gauge_variation([1, 0, 0], 2, 0, 0, [0, 3, 5], [0; 3]);
        assert_eq!(mutated.vector_spatial, [2, 0, 0]);
        assert_eq!(mutated.two_form_spatial, [0, -5, 3]);
        assert_ne!(mutated.vector_spatial, [0; 3]);
        assert_ne!(mutated.two_form_spatial, [0; 3]);
    }

    #[test]
    fn fixed_source_normalization_and_simultaneous_orientation_bridge_eq46() {
        let bridge = verify_central_bridge();
        assert!(bridge.fixed_one_sixth_normalization_matched);
        assert!(bridge.auxiliary_sign_matched);
        assert!(bridge.gauge_orbits_preserved);
        assert!(bridge.temporal_weyl_realification_exact);
        assert!(bridge.simultaneous_z_omega_orientation_exact);
        assert!(bridge.narrow_zero_brane_bridge_passed);
        assert!(!bridge.repaired_eq45_reduced_to_appendix_f);
        assert!(!bridge.full_four_dimensional_source_bridge_passed);
    }
}
