#![cfg(test)]

// Source-calibrated checks for Baulieu, Berkovits, Bossard, and Martin,
// arXiv:0705.2002v3. This file is intentionally standalone. Run it with:
//
//   rustc --edition 2024 --test src/bbbm_source_audit.rs -o /tmp/bbbm_source_audit
//   /tmp/bbbm_source_audit

mod tests {
    use std::collections::BTreeMap;

    const N_SPATIAL: usize = 8;
    const PLUS: usize = 8;
    const MINUS: usize = 9;
    const N_DIRECTIONS: usize = 10;

    // One explicit, normalized Cayley-form fixture. The paper defines Omega by
    // Eq. (13), but does not print a gamma-matrix basis or component table. The
    // following is therefore a valid chosen basis, not a basis claimed by the paper.
    const CAYLEY_TERMS: [([usize; 4], i64); 14] = [
        ([0, 1, 2, 7], 1),
        ([0, 1, 3, 6], -1),
        ([0, 1, 4, 5], 1),
        ([0, 2, 3, 5], 1),
        ([0, 2, 4, 6], 1),
        ([0, 3, 4, 7], 1),
        ([0, 5, 6, 7], -1),
        ([1, 2, 3, 4], -1),
        ([1, 2, 5, 6], 1),
        ([1, 3, 5, 7], 1),
        ([1, 4, 6, 7], 1),
        ([2, 3, 6, 7], 1),
        ([2, 4, 5, 7], -1),
        ([3, 4, 5, 6], 1),
    ];

    fn permutation_sign(values: &[usize]) -> i64 {
        let inversions = (0..values.len())
            .flat_map(|i| ((i + 1)..values.len()).map(move |j| (i, j)))
            .filter(|&(i, j)| values[i] > values[j])
            .count();
        if inversions % 2 == 0 { 1 } else { -1 }
    }

    fn omega(indices: [usize; 4]) -> i64 {
        if (0..4).any(|i| ((i + 1)..4).any(|j| indices[i] == indices[j])) {
            return 0;
        }
        let mut sorted = indices;
        sorted.sort_unstable();
        let sign = permutation_sign(&indices);
        sign * CAYLEY_TERMS
            .iter()
            .find_map(|(term, value)| (*term == sorted).then_some(*value))
            .unwrap_or(0)
    }

    fn pairs() -> Vec<(usize, usize)> {
        (0..8)
            .flat_map(|i| ((i + 1)..8).map(move |j| (i, j)))
            .collect()
    }

    // On independent coordinates X_{kl}, k<l, Eq. (14) becomes
    //
    //   P^- = (I - Omega_pair)/4.
    //
    // The paper's explicit 1/2 multiplying Omega is not discarded. It cancels
    // the factor of two from summing both ordered components X_kl and X_lk.
    fn projector_numerator() -> Vec<Vec<i64>> {
        let ps = pairs();
        ps.iter()
            .enumerate()
            .map(|(r, &(i, j))| {
                ps.iter()
                    .enumerate()
                    .map(|(c, &(k, l))| {
                        let identity = i64::from(r == c);
                        identity - omega([i, j, k, l])
                    })
                    .collect()
            })
            .collect()
    }

    fn matrix_rank(mut a: Vec<Vec<Rat>>) -> usize {
        let rows = a.len();
        let cols = a.first().map_or(0, Vec::len);
        let mut rank = 0;
        for col in 0..cols {
            let Some(pivot) = (rank..rows).find(|&r| a[r][col] != Rat::ZERO) else {
                continue;
            };
            a.swap(rank, pivot);
            let pivot_value = a[rank][col];
            for c in col..cols {
                a[rank][c] = a[rank][c] / pivot_value;
            }
            for r in 0..rows {
                if r == rank || a[r][col] == Rat::ZERO {
                    continue;
                }
                let factor = a[r][col];
                for c in col..cols {
                    a[r][c] = a[r][c] - factor * a[rank][c];
                }
            }
            rank += 1;
        }
        rank
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Rat {
        num: i64,
        den: i64,
    }

    impl Rat {
        const ZERO: Self = Self { num: 0, den: 1 };
        const ONE: Self = Self { num: 1, den: 1 };

        fn new(mut num: i64, mut den: i64) -> Self {
            assert_ne!(den, 0);
            if num == 0 {
                return Self::ZERO;
            }
            if den < 0 {
                num = -num;
                den = -den;
            }
            let divisor = gcd(num.unsigned_abs(), den as u64) as i64;
            Self {
                num: num / divisor,
                den: den / divisor,
            }
        }
    }

    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    impl std::ops::Add for Rat {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
        }
    }

    impl std::ops::Sub for Rat {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
        }
    }

    impl std::ops::Mul for Rat {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(self.num * rhs.num, self.den * rhs.den)
        }
    }

    impl std::ops::Div for Rat {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self::new(self.num * rhs.den, self.den * rhs.num)
        }
    }

    impl std::ops::Neg for Rat {
        type Output = Self;
        fn neg(self) -> Self {
            Self::new(-self.num, self.den)
        }
    }

    impl From<i64> for Rat {
        fn from(value: i64) -> Self {
            Self::new(value, 1)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum Field {
        A(u8),
        APlus,
        AMinus,
        Psi(u8),
        Eta,
        ChiSeed(u8),
        GSeed(u8),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Atom {
        field: Field,
        derivatives: [u8; N_DIRECTIONS],
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Expr(BTreeMap<Atom, Rat>);

    impl Expr {
        fn field(field: Field) -> Self {
            let mut terms = BTreeMap::new();
            terms.insert(
                Atom {
                    field,
                    derivatives: [0; N_DIRECTIONS],
                },
                Rat::ONE,
            );
            Self(terms)
        }

        fn add_scaled(&mut self, other: &Self, scale: Rat) {
            for (&atom, &coefficient) in &other.0 {
                let next = self.0.get(&atom).copied().unwrap_or(Rat::ZERO) + scale * coefficient;
                if next == Rat::ZERO {
                    self.0.remove(&atom);
                } else {
                    self.0.insert(atom, next);
                }
            }
        }

        fn scaled(&self, scale: Rat) -> Self {
            let mut result = Self::default();
            result.add_scaled(self, scale);
            result
        }

        fn derivative(&self, direction: usize) -> Self {
            let terms = self
                .0
                .iter()
                .map(|(&atom, &coefficient)| {
                    let mut next = atom;
                    next.derivatives[direction] += 1;
                    (next, coefficient)
                })
                .collect();
            Self(terms)
        }
    }

    fn sum(parts: &[(Rat, Expr)]) -> Expr {
        let mut result = Expr::default();
        for (coefficient, expression) in parts {
            result.add_scaled(expression, *coefficient);
        }
        result
    }

    fn pair_index(i: usize, j: usize) -> (usize, i64) {
        assert_ne!(i, j);
        let (ordered, sign) = if i < j { ((i, j), 1) } else { ((j, i), -1) };
        let index = pairs().iter().position(|&pair| pair == ordered).unwrap();
        (index, sign)
    }

    fn projected_seed(i: usize, j: usize, g_field: bool) -> Expr {
        if i == j {
            return Expr::default();
        }
        let (row, sign) = pair_index(i, j);
        let p4 = projector_numerator();
        let mut result = Expr::default();
        for a in 0..28 {
            let coefficient = Rat::new(sign * p4[row][a], 4);
            if coefficient != Rat::ZERO {
                let field = if g_field {
                    Field::GSeed(a as u8)
                } else {
                    Field::ChiSeed(a as u8)
                };
                result.add_scaled(&Expr::field(field), coefficient);
            }
        }
        result
    }

    fn chi(i: usize, j: usize) -> Expr {
        projected_seed(i, j, false)
    }

    fn g(i: usize, j: usize) -> Expr {
        projected_seed(i, j, true)
    }

    fn a(direction: usize) -> Expr {
        match direction {
            0..=7 => Expr::field(Field::A(direction as u8)),
            PLUS => Expr::field(Field::APlus),
            MINUS => Expr::field(Field::AMinus),
            _ => unreachable!(),
        }
    }

    fn psi(i: usize) -> Expr {
        Expr::field(Field::Psi(i as u8))
    }

    fn eta() -> Expr {
        Expr::field(Field::Eta)
    }

    fn field_strength(mu: usize, nu: usize) -> Expr {
        sum(&[
            (Rat::ONE, a(nu).derivative(mu)),
            (-Rat::ONE, a(mu).derivative(nu)),
        ])
    }

    // Charge 0 is delta_0. Charges 1 through 8 are delta_i.
    fn delta_field(charge: usize, field: Field) -> Expr {
        if charge == 0 {
            return match field {
                Field::A(i) => psi(i as usize),
                Field::APlus => Expr::default(),
                Field::AMinus => eta(),
                Field::Psi(i) => field_strength(i as usize, PLUS).scaled(-Rat::ONE),
                Field::Eta => field_strength(PLUS, MINUS),
                Field::ChiSeed(a) => Expr::field(Field::GSeed(a)),
                Field::GSeed(a) => Expr::field(Field::ChiSeed(a)).derivative(PLUS),
            };
        }

        let k = charge - 1;
        match field {
            Field::A(j) => {
                let mut result = chi(k, j as usize).scaled(-Rat::ONE);
                if k == j as usize {
                    result.add_scaled(&eta(), -Rat::ONE);
                }
                result
            }
            Field::APlus => psi(k).scaled(-Rat::ONE),
            Field::AMinus => Expr::default(),
            Field::Psi(j) => {
                let mut result = field_strength(k, j as usize);
                result.add_scaled(&g(k, j as usize), Rat::ONE);
                if k == j as usize {
                    result.add_scaled(&field_strength(PLUS, MINUS), Rat::ONE);
                }
                result
            }
            Field::Eta => field_strength(k, MINUS),
            Field::ChiSeed(seed) => {
                let (m, n) = pairs()[seed as usize];
                let mut result = Expr::default();
                if m == k {
                    result.add_scaled(&field_strength(n, MINUS), Rat::from(4));
                }
                if n == k {
                    result.add_scaled(&field_strength(m, MINUS), Rat::from(-4));
                }
                result
            }
            Field::GSeed(seed) => {
                let (m, n) = pairs()[seed as usize];
                let mut result = Expr::field(Field::ChiSeed(seed)).derivative(k);
                let v = |l: usize| {
                    sum(&[
                        (Rat::ONE, eta().derivative(l)),
                        (-Rat::ONE, psi(l).derivative(MINUS)),
                    ])
                };
                if m == k {
                    result.add_scaled(&v(n), Rat::from(-4));
                }
                if n == k {
                    result.add_scaled(&v(m), Rat::from(4));
                }
                result
            }
        }
    }

    fn delta(charge: usize, expression: &Expr) -> Expr {
        let mut result = Expr::default();
        for (atom, coefficient) in &expression.0 {
            let mut variation = delta_field(charge, atom.field);
            for direction in 0..N_DIRECTIONS {
                for _ in 0..atom.derivatives[direction] {
                    variation = variation.derivative(direction);
                }
            }
            result.add_scaled(&variation, *coefficient);
        }
        result
    }

    #[derive(Clone)]
    struct CheckedField {
        name: String,
        expression: Expr,
        gauge_direction: Option<usize>,
    }

    fn checked_fields() -> Vec<CheckedField> {
        let mut result = Vec::new();
        for mu in 0..N_DIRECTIONS {
            result.push(CheckedField {
                name: format!("A_{mu}"),
                expression: a(mu),
                gauge_direction: Some(mu),
            });
        }
        for i in 0..N_SPATIAL {
            result.push(CheckedField {
                name: format!("psi_{i}"),
                expression: psi(i),
                gauge_direction: None,
            });
        }
        result.push(CheckedField {
            name: "eta".to_string(),
            expression: eta(),
            gauge_direction: None,
        });
        for (index, &(i, j)) in pairs().iter().enumerate() {
            result.push(CheckedField {
                name: format!("chi_{i}{j}[row {index}]"),
                expression: chi(i, j),
                gauge_direction: None,
            });
            result.push(CheckedField {
                name: format!("G_{i}{j}[row {index}]"),
                expression: g(i, j),
                gauge_direction: None,
            });
        }
        result
    }

    fn translation_plus_gauge(
        field: &CheckedField,
        translation_direction: usize,
        gauge_parameter_direction: usize,
    ) -> Expr {
        let mut target = field.expression.derivative(translation_direction);
        if let Some(mu) = field.gauge_direction {
            target.add_scaled(&a(gauge_parameter_direction).derivative(mu), -Rat::ONE);
        }
        target
    }

    #[test]
    fn cayley_fixture_has_paper_normalization_and_orientation_properties() {
        assert_eq!(CAYLEY_TERMS.len(), 14);
        assert!(CAYLEY_TERMS.iter().all(|(_, value)| value.abs() == 1));

        // Self-duality fixes the relative signs between each quadruple and its
        // complement, independently of the gamma-matrix realization in Eq. (13).
        for (indices, _) in CAYLEY_TERMS {
            let complement: Vec<_> = (0..8).filter(|i| !indices.contains(i)).collect();
            let mut eight = indices.to_vec();
            eight.extend(&complement);
            let hodge_dual = permutation_sign(&eight)
                * omega([complement[0], complement[1], complement[2], complement[3]]);
            assert_eq!(
                omega(indices),
                hodge_dual,
                "Omega must be self-dual at {indices:?}"
            );
        }
    }

    #[test]
    fn equation_14_projector_is_exactly_rank_seven() {
        let p4 = projector_numerator();
        assert_eq!(p4.len(), 28);
        for row in 0..28 {
            for col in 0..28 {
                let product: i64 = (0..28).map(|k| p4[row][k] * p4[k][col]).sum();
                assert_eq!(product, 4 * p4[row][col], "P^- must be idempotent");
                assert_eq!(p4[row][col], p4[col][row], "P^- must be symmetric");
            }
        }
        let rational = p4
            .iter()
            .map(|row| row.iter().map(|&value| Rat::new(value, 4)).collect())
            .collect();
        assert_eq!(matrix_rank(rational), 7);
        assert_eq!((0..28).map(|i| p4[i][i]).sum::<i64>(), 28, "tr(P^-)=7");
    }

    #[test]
    fn equations_12_to_21_fix_charge_and_field_counts() {
        let charge_split = [1usize, 8, 7];
        assert_eq!(charge_split.iter().sum::<usize>(), 16);
        assert_eq!(charge_split[0] + charge_split[1], 9);

        // Eqs. (18)-(20) contain ten gauge-potential components and seven
        // auxiliary components. The 16|16 count follows only after the one
        // gauge redundancy is removed from the 17 raw bosonic variables.
        let gauge_potential_components = 8 + 1 + 1;
        let auxiliary_components = 7;
        let fermion_components = 8 + 1 + 7;
        assert_eq!(gauge_potential_components, 10);
        assert_eq!(gauge_potential_components + auxiliary_components, 17);
        assert_eq!(gauge_potential_components - 1 + auxiliary_components, 16);
        assert_eq!(fermion_components, 16);
    }

    #[test]
    fn equations_22_and_24_scalar_charge_close_exactly() {
        for field in checked_fields() {
            let actual = delta(0, &delta(0, &field.expression));
            let target = translation_plus_gauge(&field, PLUS, PLUS);
            assert_eq!(actual, target, "delta_0^2 failed on {}", field.name);
        }
    }

    #[test]
    fn equations_22_to_24_mixed_charges_close_exactly() {
        for i in 0..N_SPATIAL {
            for field in checked_fields() {
                let actual = sum(&[
                    (Rat::ONE, delta(0, &delta(i + 1, &field.expression))),
                    (Rat::ONE, delta(i + 1, &delta(0, &field.expression))),
                ]);
                let target = translation_plus_gauge(&field, i, i);
                assert_eq!(
                    actual, target,
                    "{{delta_0,delta_{i}}} failed on {}",
                    field.name
                );
            }
        }
    }

    #[test]
    fn equations_23_and_24_vector_charges_close_exactly() {
        for i in 0..N_SPATIAL {
            for j in 0..N_SPATIAL {
                for field in checked_fields() {
                    let actual = sum(&[
                        (Rat::ONE, delta(i + 1, &delta(j + 1, &field.expression))),
                        (Rat::ONE, delta(j + 1, &delta(i + 1, &field.expression))),
                    ]);
                    let target = if i == j {
                        translation_plus_gauge(&field, MINUS, MINUS).scaled(Rat::from(2))
                    } else {
                        Expr::default()
                    };
                    assert_eq!(
                        actual, target,
                        "{{delta_{i},delta_{j}}} failed on {}",
                        field.name
                    );
                }
            }
        }
    }
}
