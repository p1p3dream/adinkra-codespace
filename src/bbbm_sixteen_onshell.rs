#![allow(dead_code)]

//! Exact on-shell closure of all sixteen 10D N=1 SYM supercharges.
//!
//! This is deliberately separate from the 33-field BBBM auxiliary multiplet.
//! The seven auxiliaries have been eliminated, leaving ten gauge potentials
//! and sixteen gaugino components.  The transformation source is Berkovits,
//! hep-th/9308128, Eqs. (1)-(5), together with Baulieu, Berkovits, Bossard,
//! and Martin, arXiv:0705.2002, Eqs. (2)-(6).
//!
//! We suppress the conventional factor of `i`, so translations are represented
//! by `partial_mu` rather than `i partial_mu`, and use the exact real chiral
//! sigma matrices constructed in the same octonionic convention as
//! `lorentz.rs`. With `A,B=0..15`,
//!
//!   Q_A A_mu = (sigma_mu)_{A beta} psi_beta,
//!   Q_A psi   = -Gamma^{rho sigma}_{. A} F_{rho sigma},
//!   Gamma^{rho sigma}=1/2(tilde_sigma^rho sigma^sigma
//!                                -tilde_sigma^sigma sigma^rho),
//!
//! where the curvature sum is over `rho<sigma`.  The algebra is
//!
//!   {Q_A,Q_B} = 2 sigma^mu_{AB}(partial_mu+gauge(A_mu)) + EOM.
//!
//! Every identity is expanded in a free differential associative
//! superalgebra.  The gaugino residual is verified as an exact matrix
//! multiple of the Dirac equation `E=sum_mu sigma^mu D_mu psi`.

use std::collections::BTreeMap;

use serde::Serialize;

const SPACETIME: usize = 10;
const SPINORS: usize = 16;
const CHARGE_PAIRS: usize = SPINORS * (SPINORS + 1) / 2;

type Matrix16 = [[i8; SPINORS]; SPINORS];

#[derive(Clone)]
struct Clifford {
    sigma: [Matrix16; SPACETIME],
    sigma_tilde: [Matrix16; SPACETIME],
    sigma_down: [Matrix16; SPACETIME],
    bivector: [[[[i8; SPINORS]; SPINORS]; SPACETIME]; SPACETIME],
}

const FANO: [(usize, usize, usize); 7] = [
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
];

fn identity16(sign: i8) -> Matrix16 {
    let mut matrix = [[0; SPINORS]; SPINORS];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = sign;
    }
    matrix
}

fn octonion_left_multiplications() -> [[[i8; 8]; 8]; 7] {
    let mut table = [[(0i8, 0usize); 8]; 8];
    for (i, slot) in table[0].iter_mut().enumerate() {
        *slot = (1, i);
    }
    for (i, row) in table.iter_mut().enumerate() {
        row[0] = (1, i);
    }
    for (i, row) in table.iter_mut().enumerate().skip(1) {
        row[i] = (-1, 0);
    }
    for &(a, b, c) in &FANO {
        for (x, y, z) in [(a, b, c), (b, c, a), (c, a, b)] {
            table[x][y] = (1, z);
            table[y][x] = (-1, z);
        }
    }

    let mut out = [[[0; 8]; 8]; 7];
    for imaginary in 1..8 {
        for basis in 0..8 {
            let (sign, result) = table[imaginary][basis];
            out[imaginary - 1][result][basis] = sign;
        }
    }
    out
}

fn so9_gammas() -> [Matrix16; 9] {
    let mut c = [[[0i8; 8]; 8]; 8];
    for (i, row) in c[0].iter_mut().enumerate() {
        row[i] = 1;
    }
    let octonions = octonion_left_multiplications();
    c[1..].copy_from_slice(&octonions);

    let mut gamma = [[[0i8; SPINORS]; SPINORS]; 9];
    for a in 0..8 {
        for row in 0..8 {
            for col in 0..8 {
                gamma[a][row][8 + col] = c[a][row][col];
                gamma[a][8 + row][col] = c[a][col][row];
            }
        }
    }
    for i in 0..8 {
        gamma[8][i][i] = 1;
        gamma[8][8 + i][8 + i] = -1;
    }
    gamma
}

fn matrix_product(left: &Matrix16, right: &Matrix16) -> [[i16; SPINORS]; SPINORS] {
    let mut out = [[0i16; SPINORS]; SPINORS];
    for row in 0..SPINORS {
        for col in 0..SPINORS {
            out[row][col] = (0..SPINORS)
                .map(|middle| i16::from(left[row][middle]) * i16::from(right[middle][col]))
                .sum();
        }
    }
    out
}

impl Clifford {
    fn build() -> Self {
        let spatial = so9_gammas();
        let mut sigma = [[[0i8; SPINORS]; SPINORS]; SPACETIME];
        let mut sigma_tilde = sigma;
        sigma[0] = identity16(1);
        sigma_tilde[0] = identity16(-1);
        sigma[1..].copy_from_slice(&spatial);
        sigma_tilde[1..].copy_from_slice(&spatial);

        let mut sigma_down = sigma;
        for row in &mut sigma_down[0] {
            for value in row {
                *value = -*value;
            }
        }

        // Stored coefficient is -Gamma^{rho sigma}, the coefficient appearing
        // in Q psi.  The raw difference is always even in this basis.
        let mut bivector = [[[[0i8; SPINORS]; SPINORS]; SPACETIME]; SPACETIME];
        for rho in 0..SPACETIME {
            for sigma_index in (rho + 1)..SPACETIME {
                let left = matrix_product(&sigma_tilde[rho], &sigma[sigma_index]);
                let right = matrix_product(&sigma_tilde[sigma_index], &sigma[rho]);
                for row in 0..SPINORS {
                    for col in 0..SPINORS {
                        let difference = left[row][col] - right[row][col];
                        assert_eq!(difference % 2, 0);
                        bivector[rho][sigma_index][row][col] =
                            i8::try_from(-difference / 2).expect("bivector coefficient fits i8");
                    }
                }
            }
        }

        Self {
            sigma,
            sigma_tilde,
            sigma_down,
            bivector,
        }
    }

    fn verifies_split_clifford(&self) -> bool {
        for mu in 0..SPACETIME {
            for nu in 0..SPACETIME {
                let first = matrix_product(&self.sigma[mu], &self.sigma_tilde[nu]);
                let second = matrix_product(&self.sigma[nu], &self.sigma_tilde[mu]);
                for row in 0..SPINORS {
                    for col in 0..SPINORS {
                        let metric = if mu == nu {
                            if mu == 0 {
                                -1
                            } else {
                                1
                            }
                        } else {
                            0
                        };
                        let expected = if row == col { 2 * metric } else { 0 };
                        if first[row][col] + second[row][col] != expected {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Field {
    Gauge(usize),
    Gaugino(usize),
}

impl Field {
    const fn odd(self) -> bool {
        matches!(self, Self::Gaugino(_))
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Jet {
    field: Field,
    derivatives: [u8; SPACETIME],
}

type Word = Vec<Jet>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Polynomial(BTreeMap<Word, i64>);

impl Polynomial {
    fn atom(field: Field) -> Self {
        Self(BTreeMap::from([(
            vec![Jet {
                field,
                derivatives: [0; SPACETIME],
            }],
            1,
        )]))
    }

    fn add_word(&mut self, word: Word, coefficient: i64) {
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
            self.add_word(word.clone(), scale * coefficient);
        }
    }

    fn scaled(mut self, scale: i64) -> Self {
        for coefficient in self.0.values_mut() {
            *coefficient *= scale;
        }
        self
    }

    fn product(&self, rhs: &Self) -> Self {
        let mut out = Self::default();
        for (left_word, left_coefficient) in &self.0 {
            for (right_word, right_coefficient) in &rhs.0 {
                let mut word = left_word.clone();
                word.extend(right_word.iter().cloned());
                out.add_word(word, left_coefficient * right_coefficient);
            }
        }
        out
    }

    fn derivative(&self, direction: usize) -> Self {
        let mut out = Self::default();
        for (word, coefficient) in &self.0 {
            for position in 0..word.len() {
                let mut differentiated = word.clone();
                differentiated[position].derivatives[direction] += 1;
                out.add_word(differentiated, *coefficient);
            }
        }
        out
    }

    fn bracket(&self, rhs: &Self) -> Self {
        let mut out = self.product(rhs);
        out.add_scaled(&rhs.product(self), -1);
        out
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn term_count(&self) -> usize {
        self.0.len()
    }
}

fn gauge_potential(mu: usize) -> Polynomial {
    Polynomial::atom(Field::Gauge(mu))
}

fn curvature(mu: usize, nu: usize) -> Polynomial {
    let a_mu = gauge_potential(mu);
    let a_nu = gauge_potential(nu);
    let mut out = a_nu.derivative(mu);
    out.add_scaled(&a_mu.derivative(nu), -1);
    out.add_scaled(&a_mu.bracket(&a_nu), 1);
    out
}

fn covariant_derivative(mu: usize, expression: &Polynomial) -> Polynomial {
    let mut out = expression.derivative(mu);
    out.add_scaled(&gauge_potential(mu).bracket(expression), 1);
    out
}

fn delta(charge: usize, field: Field, clifford: &Clifford) -> Polynomial {
    match field {
        Field::Gauge(mu) => {
            let mut out = Polynomial::default();
            for beta in 0..SPINORS {
                out.add_scaled(
                    &Polynomial::atom(Field::Gaugino(beta)),
                    i64::from(clifford.sigma_down[mu][charge][beta]),
                );
            }
            out
        }
        Field::Gaugino(alpha) => {
            let mut out = Polynomial::default();
            for rho in 0..SPACETIME {
                for sigma in (rho + 1)..SPACETIME {
                    out.add_scaled(
                        &curvature(rho, sigma),
                        i64::from(clifford.bivector[rho][sigma][alpha][charge]),
                    );
                }
            }
            out
        }
    }
}

/// Apply a supercharge as an odd derivation.
fn apply_delta(charge: usize, expression: &Polynomial, clifford: &Clifford) -> Polynomial {
    let mut out = Polynomial::default();
    for (word, coefficient) in &expression.0 {
        let mut odd_prefix = false;
        for position in 0..word.len() {
            let jet = &word[position];
            let mut varied = delta(charge, jet.field, clifford);
            for (direction, power) in jet.derivatives.iter().copied().enumerate() {
                for _ in 0..power {
                    varied = varied.derivative(direction);
                }
            }
            for (varied_word, varied_coefficient) in &varied.0 {
                let mut combined = word[..position].to_vec();
                combined.extend(varied_word.iter().cloned());
                combined.extend(word[(position + 1)..].iter().cloned());
                let sign = if odd_prefix { -1 } else { 1 };
                out.add_word(combined, sign * coefficient * varied_coefficient);
            }
            odd_prefix ^= jet.field.odd();
        }
    }
    out
}

fn anticommutator(left: usize, right: usize, field: Field, clifford: &Clifford) -> Polynomial {
    let mut out = apply_delta(left, &delta(right, field, clifford), clifford);
    out.add_scaled(
        &apply_delta(right, &delta(left, field, clifford), clifford),
        1,
    );
    out
}

/// `2 sigma^mu_AB (partial_mu + gauge(A_mu))` on one field.
fn translation_plus_gauge(
    left: usize,
    right: usize,
    field: Field,
    clifford: &Clifford,
) -> Polynomial {
    let mut out = Polynomial::default();
    for mu in 0..SPACETIME {
        let coefficient = 2 * i64::from(clifford.sigma[mu][left][right]);
        if coefficient == 0 {
            continue;
        }
        let translated = match field {
            Field::Gauge(nu) => curvature(mu, nu),
            Field::Gaugino(alpha) => {
                covariant_derivative(mu, &Polynomial::atom(Field::Gaugino(alpha)))
            }
        };
        out.add_scaled(&translated, coefficient);
    }
    out
}

fn dirac_equation(component: usize, clifford: &Clifford) -> Polynomial {
    let mut out = Polynomial::default();
    for mu in 0..SPACETIME {
        for beta in 0..SPINORS {
            out.add_scaled(
                &covariant_derivative(mu, &Polynomial::atom(Field::Gaugino(beta))),
                i64::from(clifford.sigma[mu][component][beta]),
            );
        }
    }
    out
}

/// The temporal derivative coefficient determines the exact left multiplier
/// of the Dirac equation because sigma^0 is the identity.
fn eom_multiplier_from_temporal_residual(residual: &Polynomial) -> [i64; SPINORS] {
    let mut multiplier = [0i64; SPINORS];
    for (component, slot) in multiplier.iter_mut().enumerate() {
        let mut derivatives = [0u8; SPACETIME];
        derivatives[0] = 1;
        let key = vec![Jet {
            field: Field::Gaugino(component),
            derivatives,
        }];
        *slot = residual.0.get(&key).copied().unwrap_or(0);
    }
    multiplier
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SixteenOnShellReport {
    pub source: &'static str,
    pub system: &'static str,
    pub fields_checked: usize,
    pub charge_pairs: usize,
    pub bbbm_retained_charge_pairs: usize,
    pub pairs_involving_tensor_charges: usize,
    pub identities_checked: usize,
    pub identities_involving_tensor_charges: usize,
    pub gauge_field_identities: usize,
    pub gaugino_identities: usize,
    pub gauge_residuals_after_translation_and_gauge: usize,
    /// Nonzero Dirac multipliers in this explicit spinor/charge basis.  This
    /// sparsity count is basis-dependent; exact factorization is not.
    pub gaugino_identities_with_nonzero_dirac_multiplier: usize,
    pub unfactored_gaugino_residuals: usize,
    pub residual_terms_after_dirac_factorization: usize,
    pub auxiliaries_included: bool,
    pub spin7_subspaces_identified_in_octonionic_basis: bool,
    pub individual_nu_two_form_intertwiner_implemented: bool,
    pub all_relations_close_modulo_gauge_and_dirac_equation: bool,
    pub boundary: &'static str,
}

pub fn run() -> SixteenOnShellReport {
    let clifford = Clifford::build();
    assert!(clifford.verifies_split_clifford());

    let mut gauge_residuals = 0usize;
    let mut requiring_eom = 0usize;
    let mut unfactored = 0usize;
    let mut residual_terms = 0usize;

    for left in 0..SPINORS {
        for right in left..SPINORS {
            for mu in 0..SPACETIME {
                let field = Field::Gauge(mu);
                let mut residual = anticommutator(left, right, field, &clifford);
                residual.add_scaled(&translation_plus_gauge(left, right, field, &clifford), -1);
                if !residual.is_zero() {
                    gauge_residuals += 1;
                }
            }

            for alpha in 0..SPINORS {
                let field = Field::Gaugino(alpha);
                let mut residual = anticommutator(left, right, field, &clifford);
                residual.add_scaled(&translation_plus_gauge(left, right, field, &clifford), -1);
                let multiplier = eom_multiplier_from_temporal_residual(&residual);
                if multiplier.iter().any(|&value| value != 0) {
                    requiring_eom += 1;
                }
                for (component, coefficient) in multiplier.into_iter().enumerate() {
                    residual.add_scaled(&dirac_equation(component, &clifford), -coefficient);
                }
                if !residual.is_zero() {
                    unfactored += 1;
                    residual_terms += residual.term_count();
                }
            }
        }
    }

    SixteenOnShellReport {
        source: "Berkovits hep-th/9308128 Eqs. (1)-(5); BBBM arXiv:0705.2002 Eqs. (2)-(6)",
        system: "10D N=1 super-Yang-Mills on A_mu and Psi after eliminating the seven auxiliaries",
        fields_checked: SPACETIME + SPINORS,
        charge_pairs: CHARGE_PAIRS,
        bbbm_retained_charge_pairs: 9 * 10 / 2,
        pairs_involving_tensor_charges: 9 * 7 + 7 * 8 / 2,
        identities_checked: CHARGE_PAIRS * (SPACETIME + SPINORS),
        identities_involving_tensor_charges: (9 * 7 + 7 * 8 / 2) * (SPACETIME + SPINORS),
        gauge_field_identities: CHARGE_PAIRS * SPACETIME,
        gaugino_identities: CHARGE_PAIRS * SPINORS,
        gauge_residuals_after_translation_and_gauge: gauge_residuals,
        gaugino_identities_with_nonzero_dirac_multiplier: requiring_eom,
        unfactored_gaugino_residuals: unfactored,
        residual_terms_after_dirac_factorization: residual_terms,
        auxiliaries_included: false,
        // gamma^9=diag(I_8,-I_8). In the chosen octonionic basis the first
        // block is epsilon_1: coordinate 0 is the Spin(7) singlet and 1..7
        // span its tensor seven. Coordinates 8..15 are the vector eight.
        spin7_subspaces_identified_in_octonionic_basis: true,
        individual_nu_two_form_intertwiner_implemented: false,
        all_relations_close_modulo_gauge_and_dirac_equation: gauge_residuals == 0
            && unfactored == 0
            && residual_terms == 0,
        boundary: "This is the ordinary 26-field on-shell system, not a sixteen-charge extension of BBBM's 33-field auxiliary multiplet",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_chiral_matrices_match_the_lorentz_convention() {
        assert!(Clifford::build().verifies_split_clifford());
    }

    #[test]
    fn every_gauge_potential_pair_closes_without_eom() {
        let clifford = Clifford::build();
        for left in 0..SPINORS {
            for right in left..SPINORS {
                for mu in 0..SPACETIME {
                    let field = Field::Gauge(mu);
                    let mut residual = anticommutator(left, right, field, &clifford);
                    residual.add_scaled(&translation_plus_gauge(left, right, field, &clifford), -1);
                    assert!(residual.is_zero(), "A={left}, B={right}, mu={mu}");
                }
            }
        }
    }

    #[test]
    fn every_gaugino_pair_factors_exactly_through_the_dirac_equation() {
        let report = run();
        assert_eq!(report.unfactored_gaugino_residuals, 0);
        assert_eq!(report.residual_terms_after_dirac_factorization, 0);
        assert!(report.gaugino_identities_with_nonzero_dirac_multiplier > 0);
    }

    #[test]
    fn report_counts_all_136_pairs_on_all_26_fields() {
        let report = run();
        assert_eq!(report.fields_checked, 26);
        assert_eq!(report.charge_pairs, 136);
        assert_eq!(report.bbbm_retained_charge_pairs, 45);
        assert_eq!(report.pairs_involving_tensor_charges, 91);
        assert_eq!(report.identities_checked, 3_536);
        assert_eq!(report.identities_involving_tensor_charges, 2_366);
        assert_eq!(report.gauge_field_identities, 1_360);
        assert_eq!(report.gaugino_identities, 2_176);
        assert_eq!(report.gauge_residuals_after_translation_and_gauge, 0);
        assert_eq!(
            report.gaugino_identities_with_nonzero_dirac_multiplier,
            1_120
        );
        assert!(report.all_relations_close_modulo_gauge_and_dirac_equation);
        assert!(!report.auxiliaries_included);
    }
}
