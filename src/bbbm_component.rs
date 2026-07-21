#![allow(dead_code)]

//! Exact linearized component closure for the nine BBBM supersymmetries.
//!
//! Source: L. Baulieu, N. Berkovits, G. Bossard, A. Martin,
//! "Ten-dimensional super-Yang-Mills with nine off-shell supersymmetries",
//! arXiv:0705.2002. The transformations below are Eqs. (22) and (23), after
//! the field redefinition stated immediately before Eq. (22). The closure
//! targets are Eq. (24).
//!
//! This module checks the linearized, abelian algebra. Covariant derivatives
//! become ordinary commuting derivatives and an infinitesimal gauge
//! transformation with parameter `lambda` is
//! `delta_gauge(lambda) A_mu = -partial_mu lambda`. No field equation or
//! integration by parts is used.

use std::collections::BTreeMap;

use serde::Serialize;

const N_DIRECTIONS: usize = 10;
const PLUS: usize = 0;
const MINUS: usize = 1;

const fn spatial(i: usize) -> usize {
    i + 2
}

/// Independent component fields. `Chi(a)` and `G(a)`, `a=0,...,6`, are exact
/// coordinates on the Spin(7) antiselfdual 7 inside the 28 two-forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Field {
    A(usize),
    APlus,
    AMinus,
    Psi(usize),
    Eta,
    Chi(usize),
    G(usize),
}

type Powers = [u8; N_DIRECTIONS];

/// A sparse polynomial differential expression. A key `(f,p)` denotes
/// `(partial_+)^p[0] (partial_-)^p[1] ... f` with an exact integer coefficient.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expr(BTreeMap<(Field, Powers), i64>);

impl Expr {
    fn field(field: Field) -> Self {
        let mut out = Self::default();
        out.add_term(1, [0; N_DIRECTIONS], field);
        out
    }

    fn term(coeff: i64, direction: usize, field: Field) -> Self {
        let mut powers = [0; N_DIRECTIONS];
        powers[direction] = 1;
        let mut out = Self::default();
        out.add_term(coeff, powers, field);
        out
    }

    fn add_term(&mut self, coeff: i64, powers: Powers, field: Field) {
        if coeff == 0 {
            return;
        }
        let key = (field, powers);
        let value = self.0.entry(key).or_insert(0);
        *value += coeff;
        if *value == 0 {
            self.0.remove(&key);
        }
    }

    fn add_scaled(&mut self, other: &Self, scale: i64) {
        for (&(field, powers), &coeff) in &other.0 {
            self.add_term(scale * coeff, powers, field);
        }
    }

    fn derivative(&self, direction: usize) -> Self {
        let mut out = Self::default();
        for (&(field, mut powers), &coeff) in &self.0 {
            powers[direction] = powers[direction]
                .checked_add(1)
                .expect("derivative degree overflow");
            out.add_term(coeff, powers, field);
        }
        out
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    pub fn term_count(&self) -> usize {
        self.0.len()
    }
}

fn f(mu: usize, nu: usize) -> Expr {
    // F_{mu nu} = partial_mu A_nu - partial_nu A_mu.
    let mut out = derivative_of_potential(nu, mu);
    out.add_scaled(&derivative_of_potential(mu, nu), -1);
    out
}

fn potential(mu: usize) -> Field {
    match mu {
        PLUS => Field::APlus,
        MINUS => Field::AMinus,
        2..=9 => Field::A(mu - 2),
        _ => panic!("invalid spacetime direction"),
    }
}

fn derivative_of_potential(mu: usize, direction: usize) -> Expr {
    Expr::term(1, direction, potential(mu))
}

fn all_fields() -> Vec<Field> {
    let mut fields = Vec::with_capacity(33);
    fields.push(Field::APlus);
    fields.push(Field::AMinus);
    fields.extend((0..8).map(Field::A));
    fields.push(Field::Eta);
    fields.extend((0..8).map(Field::Psi));
    fields.extend((0..7).map(Field::Chi));
    fields.extend((0..7).map(Field::G));
    fields
}

// ---------------------------------------------------------------------------
// Spin(7) antiselfdual 7.
// ---------------------------------------------------------------------------

// Fano triples use the same Cayley convention as the repository's octonion
// matrices. The associated Cayley form has eigenvalues -3 and +1 on the 7 and
// 21 in Lambda^2(R^8). Therefore Q=I-Omega is exactly 4P as a 28 by 28 pair
// operator, and exactly 8 P^-_{ij}{}^{kl} in the individual-index convention
// used in BBBM Eq. (23). This factor of two is the usual conversion between a
// sum over all ordered antisymmetric indices and a sum over canonical pairs.
const FANO: [(usize, usize, usize); 7] = [
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
];

fn permutation_sign(values: &[usize]) -> i64 {
    let mut inversions = 0;
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if values[i] > values[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn phi(a: usize, b: usize, c: usize) -> i64 {
    if a == b || a == c || b == c || a >= 7 || b >= 7 || c >= 7 {
        return 0;
    }
    for &(x, y, z) in &FANO {
        let base = [x - 1, y - 1, z - 1];
        let mut lhs = [a, b, c];
        let mut rhs = base;
        lhs.sort_unstable();
        rhs.sort_unstable();
        if lhs == rhs {
            // `base` is assigned +1. Relative permutation signs give phi_abc.
            return permutation_sign(&base) * permutation_sign(&[a, b, c]);
        }
    }
    0
}

fn omega(i: usize, j: usize, k: usize, l: usize) -> i64 {
    if i == j || i == k || i == l || j == k || j == l || k == l {
        return 0;
    }
    let indices = [i, j, k, l];
    if indices.contains(&7) {
        // Move 7 to the last slot. Omega_{abc8}=phi_{abc}.
        let position = indices.iter().position(|&x| x == 7).unwrap();
        let mut abc = Vec::with_capacity(3);
        for &x in &indices {
            if x != 7 {
                abc.push(x);
            }
        }
        let move_sign = if (3 - position) % 2 == 0 { 1 } else { -1 };
        return move_sign * phi(abc[0], abc[1], abc[2]);
    }

    // Omega_abcd = (*phi)_abcd in R^7. The complementary indices are unique.
    let mut rest = Vec::with_capacity(3);
    for x in 0..7 {
        if !indices.contains(&x) {
            rest.push(x);
        }
    }
    let mut seven = Vec::with_capacity(7);
    seven.extend(indices);
    seven.extend(rest.iter().copied());
    permutation_sign(&seven) * phi(rest[0], rest[1], rest[2])
}

fn canonical_pair(i: usize, j: usize) -> Option<((usize, usize), i64)> {
    if i == j {
        None
    } else if i < j {
        Some(((i, j), 1))
    } else {
        Some(((j, i), -1))
    }
}

/// Q_{ij,kl}=8 P^-_{ij kl}, with both antisymmetric pairs allowed in either
/// order. This is the coefficient appearing directly in BBBM Eq. (23).
fn q(i: usize, j: usize, k: usize, l: usize) -> i64 {
    let Some(((i, j), left_sign)) = canonical_pair(i, j) else {
        return 0;
    };
    let Some(((k, l), right_sign)) = canonical_pair(k, l) else {
        return 0;
    };
    let identity = i64::from(i == k && j == l);
    left_sign * right_sign * (identity - omega(i, j, k, l))
}

/// Reconstruct an antiselfdual pair component from the seven coordinates
/// T_{a,8}. The seven columns Q_{ij,a8} are independent because
/// Q_{a8,b8}=delta_ab.
fn projected_pair(i: usize, j: usize, constructor: fn(usize) -> Field) -> Expr {
    let mut out = Expr::default();
    for a in 0..7 {
        out.add_scaled(&Expr::field(constructor(a)), q(i, j, a, 7));
    }
    out
}

fn chi(i: usize, j: usize) -> Expr {
    projected_pair(i, j, Field::Chi)
}

fn g(i: usize, j: usize) -> Expr {
    projected_pair(i, j, Field::G)
}

// ---------------------------------------------------------------------------
// BBBM transformations, Eqs. (22) and (23).
// ---------------------------------------------------------------------------

fn delta0(field: Field) -> Expr {
    match field {
        Field::A(i) => Expr::field(Field::Psi(i)),
        Field::APlus => Expr::default(),
        Field::AMinus => Expr::field(Field::Eta),
        Field::Psi(i) => {
            let mut out = f(spatial(i), PLUS);
            for value in out.0.values_mut() {
                *value = -*value;
            }
            out
        }
        Field::Eta => f(PLUS, MINUS),
        Field::Chi(a) => Expr::field(Field::G(a)),
        Field::G(a) => Expr::term(1, PLUS, Field::Chi(a)),
    }
}

fn deltai(i: usize, field: Field) -> Expr {
    assert!(i < 8);
    match field {
        Field::A(j) => {
            let mut out = Expr::default();
            if i == j {
                out.add_scaled(&Expr::field(Field::Eta), -1);
            }
            out.add_scaled(&chi(i, j), -1);
            out
        }
        Field::APlus => {
            let mut out = Expr::default();
            out.add_scaled(&Expr::field(Field::Psi(i)), -1);
            out
        }
        Field::AMinus => Expr::default(),
        Field::Psi(j) => {
            let mut out = f(spatial(i), spatial(j));
            out.add_scaled(&g(i, j), 1);
            if i == j {
                out.add_scaled(&f(PLUS, MINUS), 1);
            }
            out
        }
        Field::Eta => f(spatial(i), MINUS),
        Field::Chi(a) => {
            // Coordinate a is the (a,8) pair component. Eq. (23):
            // delta_i chi_{a8}=8 P^-_{a8 i l} F_{l-}.
            let mut out = Expr::default();
            for l in 0..8 {
                out.add_scaled(&f(spatial(l), MINUS), q(a, 7, i, l));
            }
            out
        }
        Field::G(a) => {
            // Eq. (23): delta_i G_{a8}=partial_i chi_{a8}
            // -8P^-_{a8 i l}(partial_l eta-partial_- psi_l).
            let mut out = Expr::term(1, spatial(i), Field::Chi(a));
            for l in 0..8 {
                let coeff = q(a, 7, i, l);
                out.add_scaled(&Expr::term(1, spatial(l), Field::Eta), -coeff);
                out.add_scaled(&Expr::term(1, MINUS, Field::Psi(l)), coeff);
            }
            out
        }
    }
}

fn apply(operator: impl Fn(Field) -> Expr, expression: &Expr) -> Expr {
    let mut out = Expr::default();
    for (&(field, powers), &coeff) in &expression.0 {
        let variation = operator(field);
        for (&(varied_field, varied_powers), &varied_coeff) in &variation.0 {
            let mut total_powers = powers;
            for direction in 0..N_DIRECTIONS {
                total_powers[direction] = total_powers[direction]
                    .checked_add(varied_powers[direction])
                    .expect("derivative degree overflow");
            }
            out.add_term(coeff * varied_coeff, total_powers, varied_field);
        }
    }
    out
}

fn compose(left: impl Fn(Field) -> Expr, right: impl Fn(Field) -> Expr, field: Field) -> Expr {
    apply(left, &right(field))
}

fn gauge_covariant_translation(field: Field, direction: usize) -> Expr {
    match field {
        Field::A(i) => f(direction, spatial(i)),
        Field::APlus => f(direction, PLUS),
        Field::AMinus => f(direction, MINUS),
        _ => Expr::term(1, direction, field),
    }
}

/// One failed component relation, retained as an exact sparse residual.
#[derive(Clone, Debug)]
pub struct Residual {
    pub relation: String,
    pub field: Field,
    pub terms: Expr,
}

/// Check all 33 independent fields against BBBM Eq. (24).
pub fn component_closure_residuals() -> Vec<Residual> {
    let mut residuals = Vec::new();

    for field in all_fields() {
        let mut d0_squared = compose(delta0, delta0, field);
        d0_squared.add_scaled(&gauge_covariant_translation(field, PLUS), -1);
        if !d0_squared.is_zero() {
            residuals.push(Residual {
                relation: "delta_0^2".to_string(),
                field,
                terms: d0_squared,
            });
        }

        for i in 0..8 {
            let mut mixed = compose(delta0, |f| deltai(i, f), field);
            mixed.add_scaled(&compose(|f| deltai(i, f), delta0, field), 1);
            mixed.add_scaled(&gauge_covariant_translation(field, spatial(i)), -1);
            if !mixed.is_zero() {
                residuals.push(Residual {
                    relation: format!("{{delta_0,delta_{i}}}"),
                    field,
                    terms: mixed,
                });
            }

            for j in i..8 {
                // Eq. (24) uses symmetrization. We avoid fractions by checking
                // the raw anticommutator against twice its stated right side.
                let mut pair = compose(|f| deltai(i, f), |f| deltai(j, f), field);
                pair.add_scaled(&compose(|f| deltai(j, f), |f| deltai(i, f), field), 1);
                if i == j {
                    pair.add_scaled(&gauge_covariant_translation(field, MINUS), -2);
                }
                if !pair.is_zero() {
                    residuals.push(Residual {
                        relation: format!("{{delta_{i},delta_{j}}}"),
                        field,
                        terms: pair,
                    });
                }
            }
        }
    }

    residuals
}

/// Machine-readable summary of the source-derived component calculation.
#[derive(Debug, Serialize)]
pub struct ComponentClosureReport {
    pub approximation: &'static str,
    pub raw_bosonic_fields: usize,
    pub gauge_redundancies: usize,
    pub gauge_invariant_bosonic_degrees: usize,
    pub fermionic_fields: usize,
    pub independent_fields_checked: usize,
    pub charge_pair_relations_per_field: usize,
    pub exact_relations_checked: usize,
    pub residual_relations: usize,
    pub residual_terms: usize,
    pub all_printed_relations_close_modulo_gauge_without_eom: bool,
}

pub fn run() -> ComponentClosureReport {
    let residuals = component_closure_residuals();
    ComponentClosureReport {
        approximation: "linearized abelian component transformations from BBBM Eqs. (22)-(23)",
        raw_bosonic_fields: 17,
        gauge_redundancies: 1,
        gauge_invariant_bosonic_degrees: 16,
        fermionic_fields: 16,
        independent_fields_checked: 33,
        charge_pair_relations_per_field: 45,
        exact_relations_checked: 33 * 45,
        residual_relations: residuals.len(),
        residual_terms: residuals
            .iter()
            .map(|residual| residual.terms.term_count())
            .sum(),
        all_printed_relations_close_modulo_gauge_without_eom: residuals.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin7_q_is_four_times_a_rank_seven_projector() {
        let pairs: Vec<_> = (0..8)
            .flat_map(|i| ((i + 1)..8).map(move |j| (i, j)))
            .collect();
        let matrix: Vec<Vec<i64>> = pairs
            .iter()
            .map(|&(i, j)| pairs.iter().map(|&(k, l)| q(i, j, k, l)).collect())
            .collect();

        // Q^2=4Q. Its trace is 28, so P=Q/4 has rank 7.
        for row in 0..28 {
            for col in 0..28 {
                let product: i64 = (0..28).map(|mid| matrix[row][mid] * matrix[mid][col]).sum();
                assert_eq!(product, 4 * matrix[row][col]);
                assert_eq!(matrix[row][col], matrix[col][row]);
            }
        }
        assert_eq!((0..28).map(|i| matrix[i][i]).sum::<i64>(), 28);

        // The selected coordinate columns Q_{ij,a8} reconstruct every vector
        // in the image, and have Q_{a8,b8}=delta_ab.
        for a in 0..7 {
            for b in 0..7 {
                assert_eq!(q(a, 7, b, 7), i64::from(a == b));
            }
        }
    }

    #[test]
    fn all_printed_component_relations_close_without_residue() {
        let residuals = component_closure_residuals();
        assert!(
            residuals.is_empty(),
            "BBBM Eq. (24) residuals from Eqs. (22)-(23): {residuals:#?}"
        );
    }

    #[test]
    fn field_inventory_is_complete() {
        let fields = all_fields();
        assert_eq!(fields.len(), 33);
        assert_eq!(
            fields.iter().filter(|f| matches!(f, Field::A(_))).count(),
            8
        );
        assert_eq!(
            fields.iter().filter(|f| matches!(f, Field::Psi(_))).count(),
            8
        );
        assert_eq!(
            fields.iter().filter(|f| matches!(f, Field::Chi(_))).count(),
            7
        );
        assert_eq!(
            fields.iter().filter(|f| matches!(f, Field::G(_))).count(),
            7
        );
    }

    #[test]
    fn report_counts_every_charge_pair_on_every_field() {
        let report = run();
        assert_eq!(report.exact_relations_checked, 1_485);
        assert_eq!(report.residual_relations, 0);
        assert_eq!(report.residual_terms, 0);
        assert!(report.all_printed_relations_close_modulo_gauge_without_eom);
    }
}
