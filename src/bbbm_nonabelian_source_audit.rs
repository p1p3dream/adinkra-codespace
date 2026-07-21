#![cfg(test)]

// Independent convention and identity checks for the nonabelian reading of
// Baulieu, Berkovits, Bossard, and Martin, arXiv:0705.2002v3.
//
// Source anchors in the original TeX file `twisted10df.tex`:
//   393-405: Lie-algebra-valued fields and the covariant derivative in the action
//   638-675: Eqs. (22)-(24)
//   718-727: F = dA + AA
//   737-765: graded Q, shadow gauge transformation, and closure
//   745-753: closure follows from the Bianchi identity
//   787-804: ten-dimensional horizontality condition and Bianchi identity
//   887-903: covariant derivatives and reduced-superspace constraints
//   924-938: Bianchi consequences for the component superfields
//
// Integrated test command:
//   cargo test bbbm_nonabelian_source_audit

mod tests {
    use std::collections::BTreeMap;

    const DIM: usize = 10;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Mat2([[i64; 2]; 2]);

    impl Mat2 {
        const ZERO: Self = Self([[0, 0], [0, 0]]);

        const fn new(a: i64, b: i64, c: i64, d: i64) -> Self {
            Self([[a, b], [c, d]])
        }

        fn add(self, rhs: Self) -> Self {
            let mut out = Self::ZERO;
            for i in 0..2 {
                for j in 0..2 {
                    out.0[i][j] = self.0[i][j] + rhs.0[i][j];
                }
            }
            out
        }

        fn sub(self, rhs: Self) -> Self {
            let mut out = Self::ZERO;
            for i in 0..2 {
                for j in 0..2 {
                    out.0[i][j] = self.0[i][j] - rhs.0[i][j];
                }
            }
            out
        }

        fn neg(self) -> Self {
            Self::ZERO.sub(self)
        }

        fn mul(self, rhs: Self) -> Self {
            let mut out = Self::ZERO;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        out.0[i][j] += self.0[i][k] * rhs.0[k][j];
                    }
                }
            }
            out
        }
    }

    fn commutator(x: Mat2, y: Mat2) -> Mat2 {
        x.mul(y).sub(y.mul(x))
    }

    #[derive(Clone, Debug)]
    struct Jet {
        value: Mat2,
        first: [Mat2; DIM],
        second: [[Mat2; DIM]; DIM],
    }

    impl Jet {
        fn constant(value: Mat2) -> Self {
            Self {
                value,
                first: [Mat2::ZERO; DIM],
                second: [[Mat2::ZERO; DIM]; DIM],
            }
        }
    }

    fn fixture_connection() -> [Jet; DIM] {
        std::array::from_fn(|mu| {
            let m = mu as i64 + 1;
            let mut jet = Jet::constant(Mat2::new(m, 1 - m, 2 * m + 1, -m));
            for nu in 0..DIM {
                let n = nu as i64 + 1;
                jet.first[nu] = Mat2::new(m - n, m + n, 1 - m * n, n - 2 * m);
            }
            for nu in 0..DIM {
                for rho in nu..DIM {
                    let n = nu as i64 + 1;
                    let r = rho as i64 + 1;
                    let value = Mat2::new(m + n + r, m - n - r, n * r - m, r - n);
                    jet.second[nu][rho] = value;
                    jet.second[rho][nu] = value;
                }
            }
            jet
        })
    }

    fn fixture_covariant_field() -> Jet {
        let mut jet = Jet::constant(Mat2::new(1, 2, -1, 0));
        for mu in 0..DIM {
            let m = mu as i64 + 1;
            jet.first[mu] = Mat2::new(m, 1 - m, 2 * m, -m);
        }
        for mu in 0..DIM {
            for nu in mu..DIM {
                let m = mu as i64 + 1;
                let n = nu as i64 + 1;
                let value = Mat2::new(m + n, m - n, m * n, 1 - m - n);
                jet.second[mu][nu] = value;
                jet.second[nu][mu] = value;
            }
        }
        jet
    }

    // Source convention fixed by F=dA+AA at TeX lines 718-727.
    fn field_strength(connection: &[Jet; DIM], mu: usize, nu: usize) -> Mat2 {
        connection[nu].first[mu]
            .sub(connection[mu].first[nu])
            .add(commutator(connection[mu].value, connection[nu].value))
    }

    fn partial_field_strength(connection: &[Jet; DIM], rho: usize, mu: usize, nu: usize) -> Mat2 {
        connection[nu].second[rho][mu]
            .sub(connection[mu].second[rho][nu])
            .add(commutator(connection[mu].first[rho], connection[nu].value))
            .add(commutator(connection[mu].value, connection[nu].first[rho]))
    }

    fn covariant_derivative(connection: &[Jet; DIM], mu: usize, x: &Jet) -> Mat2 {
        x.first[mu].add(commutator(connection[mu].value, x.value))
    }

    fn iterated_covariant_derivative(
        connection: &[Jet; DIM],
        mu: usize,
        nu: usize,
        x: &Jet,
    ) -> Mat2 {
        // D_mu(D_nu X), with ordinary coordinate derivatives commuting.
        let partial_mu_d_nu_x = x.second[mu][nu]
            .add(commutator(connection[nu].first[mu], x.value))
            .add(commutator(connection[nu].value, x.first[mu]));
        partial_mu_d_nu_x.add(commutator(
            connection[mu].value,
            covariant_derivative(connection, nu, x),
        ))
    }

    fn covariant_derivative_of_curvature(
        connection: &[Jet; DIM],
        rho: usize,
        mu: usize,
        nu: usize,
    ) -> Mat2 {
        partial_field_strength(connection, rho, mu, nu).add(commutator(
            connection[rho].value,
            field_strength(connection, mu, nu),
        ))
    }

    // The sign is fixed by Eq. (24): partial_mu + gauge(A_mu) must equal D_mu
    // on a covariant field and yield F_{mu nu} on A_nu.
    fn gauge_on_covariant(lambda: Mat2, x: Mat2) -> Mat2 {
        commutator(lambda, x)
    }

    fn gauge_on_connection(lambda: &Jet, connection: &Jet, nu: usize) -> Mat2 {
        lambda.first[nu]
            .add(commutator(connection.value, lambda.value))
            .neg()
    }

    #[test]
    fn equation_24_pins_covariant_derivative_and_gauge_signs() {
        let connection = fixture_connection();
        let x = fixture_covariant_field();
        for mu in 0..DIM {
            let rhs_on_x = x.first[mu].add(gauge_on_covariant(connection[mu].value, x.value));
            assert_eq!(rhs_on_x, covariant_derivative(&connection, mu, &x));

            for nu in 0..DIM {
                let rhs_on_a_nu = connection[nu].first[mu].add(gauge_on_connection(
                    &connection[mu],
                    &connection[nu],
                    nu,
                ));
                assert_eq!(
                    rhs_on_a_nu,
                    field_strength(&connection, mu, nu),
                    "partial_mu + gauge(A_mu) must give F_mu_nu"
                );
            }
        }
    }

    #[test]
    fn covariant_derivative_commutator_is_adjoint_curvature() {
        let connection = fixture_connection();
        let x = fixture_covariant_field();
        for mu in 0..DIM {
            for nu in 0..DIM {
                let lhs = iterated_covariant_derivative(&connection, mu, nu, &x)
                    .sub(iterated_covariant_derivative(&connection, nu, mu, &x));
                let rhs = commutator(field_strength(&connection, mu, nu), x.value);
                assert_eq!(lhs, rhs, "[D_{mu},D_{nu}]X=[F_{mu}{nu},X]");
            }
        }
    }

    #[test]
    fn nonabelian_bianchi_identity_is_exact() {
        let connection = fixture_connection();
        for mu in 0..DIM {
            for nu in 0..DIM {
                for rho in 0..DIM {
                    let cyclic = covariant_derivative_of_curvature(&connection, mu, nu, rho)
                        .add(covariant_derivative_of_curvature(&connection, nu, rho, mu))
                        .add(covariant_derivative_of_curvature(&connection, rho, mu, nu));
                    assert_eq!(cyclic, Mat2::ZERO, "D_[mu F_nu rho] must vanish");
                }
            }
        }
    }

    #[test]
    fn jacobi_identity_is_not_an_optional_rewrite() {
        let x = Mat2::new(0, 1, 0, 0);
        let y = Mat2::new(0, 0, 1, 0);
        let z = Mat2::new(1, 2, 3, -1);
        let jacobi = commutator(x, commutator(y, z))
            .add(commutator(y, commutator(z, x)))
            .add(commutator(z, commutator(x, y)));
        assert_eq!(jacobi, Mat2::ZERO);
    }

    #[test]
    fn abelianization_mutation_erases_a_required_equation_24_term() {
        // Constant, noncommuting A_+ and A_i isolate the nonlinear term. From
        // Eqs. (22), delta_0^2 A_i=-F_{i+}=[A_+,A_i]. Eq. (24) gives the same
        // result as gauge(A_+)A_i. Dropping [A_i,A_+] changes it to zero.
        let a_plus = Mat2::new(0, 1, 0, 0);
        let a_i = Mat2::new(0, 0, 1, 0);
        let source_required = commutator(a_plus, a_i);
        let abelianized_mutation = Mat2::ZERO;
        assert_ne!(source_required, Mat2::ZERO);
        assert_ne!(source_required, abelianized_mutation);

        let delta0_squared_a_i = commutator(a_plus, a_i);
        let equation_24_rhs = gauge_on_covariant(a_plus, a_i);
        assert_eq!(delta0_squared_a_i, equation_24_rhs);
    }

    // ---------------------------------------------------------------------
    // Independent graded-Leibniz fixture for the odd SUSY transformations.
    // ---------------------------------------------------------------------

    #[derive(Clone, Copy, Debug)]
    struct Generator {
        name: &'static str,
        parity: u8,
        delta_name: &'static str,
    }

    const GENERATORS: [Generator; 8] = [
        Generator {
            name: "A",
            parity: 0,
            delta_name: "dA",
        },
        Generator {
            name: "dA",
            parity: 1,
            delta_name: "ddA",
        },
        Generator {
            name: "psi",
            parity: 1,
            delta_name: "dpsi",
        },
        Generator {
            name: "dpsi",
            parity: 0,
            delta_name: "ddpsi",
        },
        Generator {
            name: "chi",
            parity: 1,
            delta_name: "dchi",
        },
        Generator {
            name: "dchi",
            parity: 0,
            delta_name: "ddchi",
        },
        Generator {
            name: "X",
            parity: 0,
            delta_name: "dX",
        },
        Generator {
            name: "dX",
            parity: 1,
            delta_name: "ddX",
        },
    ];

    fn generator(name: &str) -> Generator {
        *GENERATORS.iter().find(|g| g.name == name).unwrap()
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct WordExpr(BTreeMap<Vec<&'static str>, i64>);

    impl WordExpr {
        fn word(coefficient: i64, word: Vec<&'static str>) -> Self {
            let mut terms = BTreeMap::new();
            if coefficient != 0 {
                terms.insert(word, coefficient);
            }
            Self(terms)
        }

        fn add_scaled(&mut self, other: &Self, scale: i64) {
            for (word, coefficient) in &other.0 {
                let next = self.0.get(word).copied().unwrap_or(0) + scale * coefficient;
                if next == 0 {
                    self.0.remove(word);
                } else {
                    self.0.insert(word.clone(), next);
                }
            }
        }
    }

    fn odd_derivative(expression: &WordExpr, graded: bool) -> WordExpr {
        let mut result = WordExpr::default();
        for (word, coefficient) in &expression.0 {
            let mut preceding_parity = 0u8;
            for position in 0..word.len() {
                let current = generator(word[position]);
                let sign = if graded && preceding_parity == 1 {
                    -1
                } else {
                    1
                };
                let mut differentiated = word.clone();
                differentiated[position] = current.delta_name;
                result.add_scaled(&WordExpr::word(sign * coefficient, differentiated), 1);
                preceding_parity ^= current.parity;
            }
        }
        result
    }

    fn graded_bracket(x: &'static str, y: &'static str) -> WordExpr {
        let sign = if generator(x).parity * generator(y).parity == 1 {
            1
        } else {
            -1
        };
        let mut result = WordExpr::word(1, vec![x, y]);
        result.add_scaled(&WordExpr::word(sign, vec![y, x]), 1);
        result
    }

    #[test]
    fn odd_susy_operator_requires_the_graded_leibniz_sign() {
        let product = WordExpr::word(1, vec!["psi", "chi"]);
        let correct = odd_derivative(&product, true);
        let ungraded_mutation = odd_derivative(&product, false);

        let mut expected = WordExpr::word(1, vec!["dpsi", "chi"]);
        expected.add_scaled(&WordExpr::word(-1, vec!["psi", "dchi"]), 1);
        assert_eq!(correct, expected);
        assert_ne!(correct, ungraded_mutation);
    }

    #[test]
    fn odd_derivation_respects_the_graded_commutator() {
        // delta[X,psi] = [delta X,psi] + [X,delta psi] because X is even.
        let lhs = odd_derivative(&graded_bracket("X", "psi"), true);
        let mut rhs = graded_bracket("dX", "psi");
        rhs.add_scaled(&graded_bracket("X", "dpsi"), 1);
        assert_eq!(lhs, rhs);

        // delta[psi,chi]_graded = [delta psi,chi]_graded
        //                              - [psi,delta chi]_graded.
        let lhs = odd_derivative(&graded_bracket("psi", "chi"), true);
        let mut rhs = graded_bracket("dpsi", "chi");
        rhs.add_scaled(&graded_bracket("psi", "dchi"), -1);
        assert_eq!(lhs, rhs);
    }
}
