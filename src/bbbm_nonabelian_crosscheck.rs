#![allow(dead_code)]

//! Independent nonabelian cross-check of BBBM Eqs. (22)-(24).
//!
//! The checker works in a free differential associative superalgebra.  Matrix
//! multiplication is represented by concatenation of noncommuting words;
//! fermionic signs enter only through the odd Leibniz rule for a supersymmetry
//! variation.  This is stronger than testing commuting scalar fixtures and
//! avoids sharing an expression engine with the production implementation.
//!
//! The curvature and covariant derivative are
//!
//!     F_{mu nu} = partial_mu A_nu - partial_nu A_mu + [A_mu,A_nu],
//!     D_mu X    = partial_mu X + [A_mu,X].
//!
//! With `gauge(lambda) A_mu=-D_mu lambda` and
//! `gauge(lambda) X=[lambda,X]`, the operator
//! `partial_mu+gauge(A_mu)` is `F_{mu nu}` on a potential and `D_mu` on every
//! covariant field.  No field equation, integration by parts, trace identity,
//! random evaluation, or commutativity assumption is used.

use std::collections::BTreeMap;

const DIRECTIONS: usize = 10;
const PLUS: usize = 0;
const MINUS: usize = 1;

const fn transverse(i: usize) -> usize {
    i + 2
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Field {
    APlus,
    AMinus,
    A(usize),
    Eta,
    Psi(usize),
    Chi(usize),
    G(usize),
}

impl Field {
    const fn is_fermionic(self) -> bool {
        matches!(self, Self::Eta | Self::Psi(_) | Self::Chi(_))
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Atom {
    field: Field,
    derivatives: [u8; DIRECTIONS],
}

type Word = Vec<Atom>;

/// Exact sum of noncommuting differential words.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct FreeExpr(BTreeMap<Word, i64>);

impl FreeExpr {
    fn field(field: Field) -> Self {
        let atom = Atom {
            field,
            derivatives: [0; DIRECTIONS],
        };
        Self(BTreeMap::from([(vec![atom], 1)]))
    }

    fn add_term(&mut self, word: Word, coefficient: i64) {
        if coefficient == 0 {
            return;
        }
        let entry = self.0.entry(word.clone()).or_insert(0);
        *entry += coefficient;
        if *entry == 0 {
            self.0.remove(&word);
        }
    }

    fn add_scaled(&mut self, rhs: &Self, scale: i64) {
        for (word, coefficient) in &rhs.0 {
            self.add_term(word.clone(), scale * coefficient);
        }
    }

    fn scaled(mut self, scale: i64) -> Self {
        for coefficient in self.0.values_mut() {
            *coefficient *= scale;
        }
        self
    }

    fn multiply(&self, rhs: &Self) -> Self {
        let mut out = Self::default();
        for (left_word, left_coefficient) in &self.0 {
            for (right_word, right_coefficient) in &rhs.0 {
                let mut word = left_word.clone();
                word.extend(right_word.iter().cloned());
                out.add_term(word, left_coefficient * right_coefficient);
            }
        }
        out
    }

    /// Ordinary, even spacetime derivative with the full Leibniz rule.
    fn derivative(&self, direction: usize) -> Self {
        let mut out = Self::default();
        for (word, coefficient) in &self.0 {
            for varied in 0..word.len() {
                let mut differentiated = word.clone();
                differentiated[varied].derivatives[direction] += 1;
                out.add_term(differentiated, *coefficient);
            }
        }
        out
    }

    fn bracket(&self, rhs: &Self) -> Self {
        let mut out = self.multiply(rhs);
        out.add_scaled(&rhs.multiply(self), -1);
        out
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn term_count(&self) -> usize {
        self.0.len()
    }
}

fn potential(direction: usize) -> FreeExpr {
    let field = match direction {
        PLUS => Field::APlus,
        MINUS => Field::AMinus,
        2..=9 => Field::A(direction - 2),
        _ => panic!("invalid BBBM direction"),
    };
    FreeExpr::field(field)
}

fn curvature(mu: usize, nu: usize) -> FreeExpr {
    let a_mu = potential(mu);
    let a_nu = potential(nu);
    let mut out = a_nu.derivative(mu);
    out.add_scaled(&a_mu.derivative(nu), -1);
    out.add_scaled(&a_mu.bracket(&a_nu), 1);
    out
}

fn covariant_derivative(direction: usize, expression: &FreeExpr) -> FreeExpr {
    let mut out = expression.derivative(direction);
    out.add_scaled(&potential(direction).bracket(expression), 1);
    out
}

// Independent octonionic realization of the antiselfdual Spin(7) seven.
const ORIENTED_FANO_LINES: [(usize, usize, usize); 7] = [
    (0, 1, 2),
    (0, 3, 4),
    (0, 6, 5),
    (1, 3, 5),
    (1, 4, 6),
    (2, 3, 6),
    (2, 5, 4),
];

fn permutation_sign(indices: &[usize]) -> i64 {
    let inversions = indices
        .iter()
        .enumerate()
        .map(|(i, left)| {
            indices[(i + 1)..]
                .iter()
                .filter(|right| left > *right)
                .count()
        })
        .sum::<usize>();
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn associative_form(a: usize, b: usize, c: usize) -> i64 {
    if a == b || a == c || b == c || a >= 7 || b >= 7 || c >= 7 {
        return 0;
    }
    let mut query = [a, b, c];
    query.sort_unstable();
    for line in ORIENTED_FANO_LINES {
        let mut sorted_line = [line.0, line.1, line.2];
        sorted_line.sort_unstable();
        if query == sorted_line {
            return permutation_sign(&[line.0, line.1, line.2]) * permutation_sign(&[a, b, c]);
        }
    }
    0
}

fn cayley(i: usize, j: usize, k: usize, l: usize) -> i64 {
    let slots = [i, j, k, l];
    let mut distinct = slots.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != 4 {
        return 0;
    }
    if let Some(position) = slots.iter().position(|&slot| slot == 7) {
        let abc: Vec<_> = slots.iter().copied().filter(|&slot| slot != 7).collect();
        let sign = if (3 - position) % 2 == 0 { 1 } else { -1 };
        return sign * associative_form(abc[0], abc[1], abc[2]);
    }

    let complement: Vec<_> = (0..7).filter(|slot| !slots.contains(slot)).collect();
    let ordering = [
        slots[0],
        slots[1],
        slots[2],
        slots[3],
        complement[0],
        complement[1],
        complement[2],
    ];
    permutation_sign(&ordering) * associative_form(complement[0], complement[1], complement[2])
}

/// `8 P^-_{ij kl}` in the individual-index convention of BBBM Eq. (23).
fn projector8(i: usize, j: usize, k: usize, l: usize) -> i64 {
    if i == j || k == l {
        return 0;
    }
    let (i0, j0, left_sign) = if i < j { (i, j, 1) } else { (j, i, -1) };
    let (k0, l0, right_sign) = if k < l { (k, l, 1) } else { (l, k, -1) };
    let identity = i64::from(i0 == k0 && j0 == l0);
    left_sign * right_sign * (identity - cayley(i0, j0, k0, l0))
}

fn projected_pair(i: usize, j: usize, field: fn(usize) -> Field) -> FreeExpr {
    let mut out = FreeExpr::default();
    for a in 0..7 {
        out.add_scaled(&FreeExpr::field(field(a)), projector8(i, j, a, 7));
    }
    out
}

fn chi(i: usize, j: usize) -> FreeExpr {
    projected_pair(i, j, Field::Chi)
}

fn auxiliary(i: usize, j: usize) -> FreeExpr {
    projected_pair(i, j, Field::G)
}

#[derive(Clone, Copy)]
enum Charge {
    Scalar,
    Vector(usize),
}

#[derive(Clone, Copy)]
struct CheckerVariant {
    /// Deliberately wrong sign used only by the negative control.
    mutate_vector_a_plus_sign: bool,
}

impl CheckerVariant {
    const REFERENCE: Self = Self {
        mutate_vector_a_plus_sign: false,
    };
    const MUTATED: Self = Self {
        mutate_vector_a_plus_sign: true,
    };
}

fn delta_scalar(field: Field) -> FreeExpr {
    match field {
        Field::A(i) => FreeExpr::field(Field::Psi(i)),
        Field::APlus => FreeExpr::default(),
        Field::AMinus => FreeExpr::field(Field::Eta),
        Field::Psi(i) => curvature(transverse(i), PLUS).scaled(-1),
        Field::Eta => curvature(PLUS, MINUS),
        Field::Chi(a) => FreeExpr::field(Field::G(a)),
        Field::G(a) => covariant_derivative(PLUS, &FreeExpr::field(Field::Chi(a))),
    }
}

fn delta_vector(i: usize, field: Field, variant: CheckerVariant) -> FreeExpr {
    assert!(i < 8);
    match field {
        Field::A(j) => {
            let mut out = chi(i, j).scaled(-1);
            if i == j {
                out.add_scaled(&FreeExpr::field(Field::Eta), -1);
            }
            out
        }
        Field::APlus => {
            FreeExpr::field(Field::Psi(i)).scaled(if variant.mutate_vector_a_plus_sign {
                1
            } else {
                -1
            })
        }
        Field::AMinus => FreeExpr::default(),
        Field::Psi(j) => {
            let mut out = curvature(transverse(i), transverse(j));
            out.add_scaled(&auxiliary(i, j), 1);
            if i == j {
                out.add_scaled(&curvature(PLUS, MINUS), 1);
            }
            out
        }
        Field::Eta => curvature(transverse(i), MINUS),
        Field::Chi(a) => {
            let mut out = FreeExpr::default();
            for l in 0..8 {
                out.add_scaled(&curvature(transverse(l), MINUS), projector8(a, 7, i, l));
            }
            out
        }
        Field::G(a) => {
            let chi_a = FreeExpr::field(Field::Chi(a));
            let mut out = covariant_derivative(transverse(i), &chi_a);
            for l in 0..8 {
                let coefficient = projector8(a, 7, i, l);
                out.add_scaled(
                    &covariant_derivative(transverse(l), &FreeExpr::field(Field::Eta)),
                    -coefficient,
                );
                out.add_scaled(
                    &covariant_derivative(MINUS, &FreeExpr::field(Field::Psi(l))),
                    coefficient,
                );
            }
            out
        }
    }
}

fn delta_field(charge: Charge, field: Field, variant: CheckerVariant) -> FreeExpr {
    match charge {
        Charge::Scalar => delta_scalar(field),
        Charge::Vector(i) => delta_vector(i, field, variant),
    }
}

fn differentiate_multi(mut expression: FreeExpr, powers: [u8; DIRECTIONS]) -> FreeExpr {
    for (direction, power) in powers.into_iter().enumerate() {
        for _ in 0..power {
            expression = expression.derivative(direction);
        }
    }
    expression
}

/// Apply a charge as an odd derivation.  The prefix parity implements
/// `delta(XY)=delta(X)Y+(-1)^|X| X delta(Y)`.
fn apply_charge(charge: Charge, expression: &FreeExpr, variant: CheckerVariant) -> FreeExpr {
    let mut out = FreeExpr::default();
    for (word, coefficient) in &expression.0 {
        let mut odd_prefix = false;
        for position in 0..word.len() {
            let atom = &word[position];
            let varied =
                differentiate_multi(delta_field(charge, atom.field, variant), atom.derivatives);
            let prefix = FreeExpr(BTreeMap::from([(
                word[..position].to_vec(),
                if odd_prefix {
                    -*coefficient
                } else {
                    *coefficient
                },
            )]));
            let suffix = FreeExpr(BTreeMap::from([(word[(position + 1)..].to_vec(), 1)]));
            out.add_scaled(&prefix.multiply(&varied).multiply(&suffix), 1);
            odd_prefix ^= atom.field.is_fermionic();
        }
    }
    out
}

fn all_fields() -> Vec<Field> {
    let mut fields = vec![Field::APlus, Field::AMinus];
    fields.extend((0..8).map(Field::A));
    fields.push(Field::Eta);
    fields.extend((0..8).map(Field::Psi));
    fields.extend((0..7).map(Field::Chi));
    fields.extend((0..7).map(Field::G));
    fields
}

fn covariant_translation(field: Field, direction: usize) -> FreeExpr {
    match field {
        Field::APlus => curvature(direction, PLUS),
        Field::AMinus => curvature(direction, MINUS),
        Field::A(i) => curvature(direction, transverse(i)),
        _ => covariant_derivative(direction, &FreeExpr::field(field)),
    }
}

#[derive(Clone, Debug)]
struct Failure {
    relation: String,
    field: Field,
    residual_terms: usize,
}

fn closure_failures(variant: CheckerVariant) -> Vec<Failure> {
    let scalar = Charge::Scalar;
    let mut failures = Vec::new();

    for field in all_fields() {
        let mut scalar_square = apply_charge(scalar, &delta_field(scalar, field, variant), variant);
        scalar_square.add_scaled(&covariant_translation(field, PLUS), -1);
        if !scalar_square.is_zero() {
            failures.push(Failure {
                relation: "delta_0^2".to_string(),
                field,
                residual_terms: scalar_square.term_count(),
            });
        }

        for i in 0..8 {
            let vector_i = Charge::Vector(i);
            let mut mixed = apply_charge(scalar, &delta_field(vector_i, field, variant), variant);
            mixed.add_scaled(
                &apply_charge(vector_i, &delta_field(scalar, field, variant), variant),
                1,
            );
            mixed.add_scaled(&covariant_translation(field, transverse(i)), -1);
            if !mixed.is_zero() {
                failures.push(Failure {
                    relation: format!("{{delta_0,delta_{i}}}"),
                    field,
                    residual_terms: mixed.term_count(),
                });
            }

            for j in i..8 {
                let vector_j = Charge::Vector(j);
                let mut pair =
                    apply_charge(vector_i, &delta_field(vector_j, field, variant), variant);
                pair.add_scaled(
                    &apply_charge(vector_j, &delta_field(vector_i, field, variant), variant),
                    1,
                );
                if i == j {
                    pair.add_scaled(&covariant_translation(field, MINUS), -2);
                }
                if !pair.is_zero() {
                    failures.push(Failure {
                        relation: format!("{{delta_{i},delta_{j}}}"),
                        field,
                        residual_terms: pair.term_count(),
                    });
                }
            }
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_projector_has_rank_seven_normalization() {
        let pairs: Vec<_> = (0..8)
            .flat_map(|i| ((i + 1)..8).map(move |j| (i, j)))
            .collect();
        let q: Vec<Vec<_>> = pairs
            .iter()
            .map(|&(i, j)| pairs.iter().map(|&(k, l)| projector8(i, j, k, l)).collect())
            .collect();
        for row in 0..28 {
            for col in 0..28 {
                let square: i64 = (0..28).map(|middle| q[row][middle] * q[middle][col]).sum();
                assert_eq!(square, 4 * q[row][col]);
            }
        }
        assert_eq!((0..28).map(|i| q[i][i]).sum::<i64>(), 28);
    }

    #[test]
    fn curvature_is_genuinely_noncommutative() {
        let commutator = potential(PLUS).bracket(&potential(MINUS));
        assert_eq!(commutator.term_count(), 2);
        assert!(!commutator.is_zero());
        assert_ne!(curvature(PLUS, MINUS), curvature(MINUS, PLUS));
    }

    #[test]
    fn all_1485_nonabelian_component_relations_close_exactly() {
        let failures = closure_failures(CheckerVariant::REFERENCE);
        assert!(
            failures.is_empty(),
            "independent nonabelian BBBM residuals: {failures:#?}"
        );
    }

    #[test]
    fn negative_sign_mutation_is_detected() {
        let failures = closure_failures(CheckerVariant::MUTATED);
        assert!(!failures.is_empty());
        assert!(failures.iter().any(|failure| {
            failure.relation.contains("delta_0,delta_") && failure.residual_terms > 0
        }));
    }
}
