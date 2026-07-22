//! Exact four-dimensional N=1 supercovariant-derivative algebra in the
//! conventions of Gates and Hu, arXiv:2407.09334v1, Eq. (2.22).

use num_complex::Complex;
use num_rational::Ratio;
use serde::Serialize;
use std::collections::BTreeMap;

type GaussianRational = Complex<Ratio<i64>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Monomial {
    grassmann_mask: u8,
    spacetime_derivatives: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Polynomial(BTreeMap<Monomial, GaussianRational>);

#[derive(Debug, Clone, Copy)]
enum Derivative {
    Left(usize),
    Right(usize),
}

#[derive(Debug, Clone, Serialize)]
pub struct SupercovariantDerivativeReport {
    pub schema_version: &'static str,
    pub source_arxiv: &'static str,
    pub source_equation: &'static str,
    pub convention: &'static str,
    pub grassmann_basis_dimension: usize,
    pub symmetric_derivative_pairs: usize,
    pub monomials_checked_per_pair: usize,
    pub exact_relations_checked: usize,
    pub zero_residual_relations: usize,
    pub residual_relations: usize,
    pub same_chirality_anticommutators_zero: bool,
    pub mixed_anticommutators_are_spacetime_derivatives: bool,
    pub boundary: &'static str,
    pub passed: bool,
}

fn zero() -> GaussianRational {
    Complex::new(Ratio::from_integer(0), Ratio::from_integer(0))
}

fn one() -> GaussianRational {
    Complex::new(Ratio::from_integer(1), Ratio::from_integer(0))
}

fn i_half() -> GaussianRational {
    Complex::new(Ratio::from_integer(0), Ratio::new(1, 2))
}

fn i_unit() -> GaussianRational {
    Complex::new(Ratio::from_integer(0), Ratio::from_integer(1))
}

impl Polynomial {
    fn basis(mask: u8) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(
            Monomial {
                grassmann_mask: mask,
                spacetime_derivatives: [0; 4],
            },
            one(),
        );
        Self(terms)
    }

    fn insert(&mut self, monomial: Monomial, coefficient: GaussianRational) {
        if coefficient == zero() {
            return;
        }
        let value = self.0.entry(monomial).or_insert_with(zero);
        *value += coefficient;
        if *value == zero() {
            self.0.remove(&monomial);
        }
    }

    fn add(mut self, other: Self) -> Self {
        for (monomial, coefficient) in other.0 {
            self.insert(monomial, coefficient);
        }
        self
    }

    fn spacetime_derivative(mut self, index: usize) -> Self {
        let mut result = Self::default();
        for (mut monomial, coefficient) in std::mem::take(&mut self.0) {
            monomial.spacetime_derivatives[index] += 1;
            result.insert(monomial, coefficient);
        }
        result
    }

    fn scale(mut self, scalar: GaussianRational) -> Self {
        for coefficient in self.0.values_mut() {
            *coefficient *= scalar.clone();
        }
        self.0.retain(|_, coefficient| *coefficient != zero());
        self
    }
}

fn preceding_occupied(mask: u8, variable: usize) -> u32 {
    (mask & ((1u8 << variable) - 1)).count_ones()
}

fn grassmann_derivative(polynomial: &Polynomial, variable: usize) -> Polynomial {
    let mut result = Polynomial::default();
    for (&monomial, coefficient) in &polynomial.0 {
        if monomial.grassmann_mask & (1 << variable) == 0 {
            continue;
        }
        let sign = if preceding_occupied(monomial.grassmann_mask, variable) % 2 == 0 {
            1
        } else {
            -1
        };
        let mut output = monomial;
        output.grassmann_mask &= !(1 << variable);
        result.insert(output, coefficient.clone() * Ratio::from_integer(sign));
    }
    result
}

fn grassmann_left_multiply(polynomial: &Polynomial, variable: usize) -> Polynomial {
    let mut result = Polynomial::default();
    for (&monomial, coefficient) in &polynomial.0 {
        if monomial.grassmann_mask & (1 << variable) != 0 {
            continue;
        }
        let sign = if preceding_occupied(monomial.grassmann_mask, variable) % 2 == 0 {
            1
        } else {
            -1
        };
        let mut output = monomial;
        output.grassmann_mask |= 1 << variable;
        result.insert(output, coefficient.clone() * Ratio::from_integer(sign));
    }
    result
}

fn apply(operator: Derivative, polynomial: &Polynomial) -> Polynomial {
    match operator {
        Derivative::Left(alpha) => {
            let mut result = grassmann_derivative(polynomial, alpha);
            for dotted in 0..2 {
                let translation = grassmann_left_multiply(polynomial, 2 + dotted)
                    .spacetime_derivative(2 * alpha + dotted)
                    .scale(i_half());
                result = result.add(translation);
            }
            result
        }
        Derivative::Right(dotted) => {
            let mut result = grassmann_derivative(polynomial, 2 + dotted);
            for alpha in 0..2 {
                let translation = grassmann_left_multiply(polynomial, alpha)
                    .spacetime_derivative(2 * alpha + dotted)
                    .scale(i_half());
                result = result.add(translation);
            }
            result
        }
    }
}

fn anticommutator(first: Derivative, second: Derivative, input: &Polynomial) -> Polynomial {
    apply(first, &apply(second, input)).add(apply(second, &apply(first, input)))
}

fn expected(first: Derivative, second: Derivative, input: Polynomial) -> Polynomial {
    match (first, second) {
        (Derivative::Left(alpha), Derivative::Right(dotted))
        | (Derivative::Right(dotted), Derivative::Left(alpha)) => input
            .spacetime_derivative(2 * alpha + dotted)
            .scale(i_unit()),
        _ => Polynomial::default(),
    }
}

pub fn verify() -> SupercovariantDerivativeReport {
    let operators = [
        Derivative::Left(0),
        Derivative::Left(1),
        Derivative::Right(0),
        Derivative::Right(1),
    ];
    let mut exact_relations_checked = 0;
    let mut zero_residual_relations = 0;
    let mut same_chirality_ok = true;
    let mut mixed_ok = true;
    for first_index in 0..operators.len() {
        for second_index in first_index..operators.len() {
            let first = operators[first_index];
            let second = operators[second_index];
            for mask in 0..16 {
                let input = Polynomial::basis(mask);
                let actual = anticommutator(first, second, &input);
                let wanted = expected(first, second, input);
                exact_relations_checked += 1;
                if actual == wanted {
                    zero_residual_relations += 1;
                } else if matches!(
                    (first, second),
                    (Derivative::Left(_), Derivative::Left(_))
                        | (Derivative::Right(_), Derivative::Right(_))
                ) {
                    same_chirality_ok = false;
                } else {
                    mixed_ok = false;
                }
            }
        }
    }
    let residual_relations = exact_relations_checked - zero_residual_relations;
    SupercovariantDerivativeReport {
        schema_version: "adynkra-4d-n1-supercovariant-derivative-v1",
        source_arxiv: "2407.09334",
        source_equation: "2.22",
        convention: "D_alpha = partial_alpha + (i/2) theta_bar^dot_alpha partial_alpha_dot_alpha, with the conjugate formula for D_dot_alpha",
        grassmann_basis_dimension: 16,
        symmetric_derivative_pairs: 10,
        monomials_checked_per_pair: 16,
        exact_relations_checked,
        zero_residual_relations,
        residual_relations,
        same_chirality_anticommutators_zero: same_chirality_ok,
        mixed_anticommutators_are_spacetime_derivatives: mixed_ok,
        boundary: "exact superspace derivative algebra only; irreducible Clebsch-Gordan intertwiners and prepotential gauge cohomology remain open",
        passed: residual_relations == 0 && same_chirality_ok && mixed_ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grassmann_left_derivatives_obey_the_sign_rule() {
        let theta0_theta1 = Polynomial::basis(0b0011);
        assert_eq!(
            grassmann_derivative(&theta0_theta1, 0),
            Polynomial::basis(0b0010)
        );
        assert_eq!(
            grassmann_derivative(&theta0_theta1, 1),
            Polynomial::basis(0b0001).scale(Complex::new(
                Ratio::from_integer(-1),
                Ratio::from_integer(0),
            ))
        );
    }

    #[test]
    fn equation_2_22_closes_on_all_grassmann_monomials() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.exact_relations_checked, 160);
        assert_eq!(report.residual_relations, 0);
    }
}
