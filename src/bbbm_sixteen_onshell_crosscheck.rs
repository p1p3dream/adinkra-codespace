#![cfg(test)]
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

//! Independent coefficient-level cross-check of the full on-shell 10D,
//! N=1 super-Maxwell algebra.
//!
//! Sources:
//! * Gates, Hannon, Siew, and Stiffler, arXiv:2010.06124, Eqs. (2.2) and
//!   (2.6)-(2.10), for the 10D transformations, potential closure, and the
//!   gaugino non-closure factor multiplying its equation of motion.
//! * Baulieu, Berkovits, Bossard, and Martin, arXiv:0705.2002, Eqs. (1)-(6),
//!   for the distinction between the 16-charge on-shell system and the
//!   9-charge auxiliary system.
//!
//! This file deliberately shares no matrix or closure implementation with the
//! production BBBM modules. All arithmetic is integral.

const SPINOR: usize = 16;
const VECTOR: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
struct M8([[i32; 8]; 8]);

impl M8 {
    fn zero() -> Self {
        Self([[0; 8]; 8])
    }

    fn identity() -> Self {
        let mut out = Self::zero();
        for i in 0..8 {
            out.0[i][i] = 1;
        }
        out
    }

    fn transpose(&self) -> Self {
        let mut out = Self::zero();
        for i in 0..8 {
            for j in 0..8 {
                out.0[i][j] = self.0[j][i];
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct M16([[i32; SPINOR]; SPINOR]);

impl M16 {
    fn zero() -> Self {
        Self([[0; SPINOR]; SPINOR])
    }

    fn identity() -> Self {
        let mut out = Self::zero();
        for i in 0..SPINOR {
            out.0[i][i] = 1;
        }
        out
    }

    fn transpose(&self) -> Self {
        let mut out = Self::zero();
        for i in 0..SPINOR {
            for j in 0..SPINOR {
                out.0[i][j] = self.0[j][i];
            }
        }
        out
    }

    fn product(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..SPINOR {
            for k in 0..SPINOR {
                let left = self.0[i][k];
                if left == 0 {
                    continue;
                }
                for j in 0..SPINOR {
                    out.0[i][j] += left * rhs.0[k][j];
                }
            }
        }
        out
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..SPINOR {
            for j in 0..SPINOR {
                out.0[i][j] = self.0[i][j] + rhs.0[i][j];
            }
        }
        out
    }

    fn subtract(&self, rhs: &Self) -> Self {
        self.add(&rhs.scaled(-1))
    }

    fn scaled(&self, factor: i32) -> Self {
        let mut out = Self::zero();
        for i in 0..SPINOR {
            for j in 0..SPINOR {
                out.0[i][j] = factor * self.0[i][j];
            }
        }
        out
    }

    fn exact_half(&self) -> Self {
        let mut out = Self::zero();
        for i in 0..SPINOR {
            for j in 0..SPINOR {
                assert_eq!(self.0[i][j] % 2, 0, "nonintegral bivector entry");
                out.0[i][j] = self.0[i][j] / 2;
            }
        }
        out
    }

    fn max_abs(&self) -> i32 {
        self.0
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| value.abs())
            .max()
            .unwrap_or(0)
    }
}

fn octonion_left_multiplications() -> Vec<M8> {
    // e_a e_b=e_c for cyclic triples. Entry (k,j) of L_ei is the
    // coefficient of e_k in e_i e_j, with e_0=1.
    const FANO: [(usize, usize, usize); 7] = [
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    ];
    let mut table = [[(0i32, 0usize); 8]; 8];
    for i in 0..8 {
        table[0][i] = (1, i);
        table[i][0] = (1, i);
    }
    for i in 1..8 {
        table[i][i] = (-1, 0);
    }
    for &(a, b, c) in &FANO {
        for &(x, y, z) in &[(a, b, c), (b, c, a), (c, a, b)] {
            table[x][y] = (1, z);
            table[y][x] = (-1, z);
        }
    }

    let mut result = Vec::with_capacity(7);
    for i in 1..8 {
        let mut matrix = M8::zero();
        for j in 0..8 {
            let (sign, k) = table[i][j];
            matrix.0[k][j] = sign;
        }
        result.push(matrix);
    }
    result
}

fn euclidean_so9_gammas() -> Vec<M16> {
    let mut c = vec![M8::identity()];
    c.extend(octonion_left_multiplications());

    let mut gammas = Vec::with_capacity(9);
    for matrix in &c {
        let transpose = matrix.transpose();
        let mut gamma = M16::zero();
        for i in 0..8 {
            for j in 0..8 {
                gamma.0[i][8 + j] = matrix.0[i][j];
                gamma.0[8 + i][j] = transpose.0[i][j];
            }
        }
        gammas.push(gamma);
    }

    let mut gamma9 = M16::zero();
    for i in 0..8 {
        gamma9.0[i][i] = 1;
        gamma9.0[8 + i][8 + i] = -1;
    }
    gammas.push(gamma9);
    gammas
}

/// Real chiral 10D matrices satisfying
/// sigma^mu tilde_sigma^nu + sigma^nu tilde_sigma^mu=2 eta^{mu nu}.
#[derive(Clone)]
struct Chiral10D {
    sigma_up: Vec<M16>,
    sigma_tilde_up: Vec<M16>,
    sigma_lower: Vec<M16>,
    sigma_two: Vec<Vec<M16>>,
}

impl Chiral10D {
    fn build() -> Self {
        let identity = M16::identity();
        let spatial = euclidean_so9_gammas();

        let mut sigma_up = vec![identity.clone()];
        sigma_up.extend(spatial.iter().cloned());
        let mut sigma_tilde_up = vec![identity.scaled(-1)];
        sigma_tilde_up.extend(spatial);

        let metric = [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let sigma_lower = (0..VECTOR)
            .map(|mu| sigma_up[mu].scaled(metric[mu]))
            .collect::<Vec<_>>();

        // (sigma^{mu nu})_a^b = 1/2(sigma^mu tilde_sigma^nu
        // - sigma^nu tilde_sigma^mu)_a^b.
        let sigma_two = (0..VECTOR)
            .map(|mu| {
                (0..VECTOR)
                    .map(|nu| {
                        sigma_up[mu]
                            .product(&sigma_tilde_up[nu])
                            .subtract(&sigma_up[nu].product(&sigma_tilde_up[mu]))
                            .exact_half()
                    })
                    .collect()
            })
            .collect();

        Self {
            sigma_up,
            sigma_tilde_up,
            sigma_lower,
            sigma_two,
        }
    }

    fn clifford_max_residual(&self) -> i32 {
        let identity = M16::identity();
        let metric = [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut maximum = 0;
        for mu in 0..VECTOR {
            for nu in 0..VECTOR {
                let mut residual = self.sigma_up[mu]
                    .product(&self.sigma_tilde_up[nu])
                    .add(&self.sigma_up[nu].product(&self.sigma_tilde_up[mu]));
                if mu == nu {
                    residual = residual.subtract(&identity.scaled(2 * metric[mu]));
                }
                maximum = maximum.max(residual.max_abs());
            }
        }
        maximum
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PotentialClosureReport {
    charge_pairs: usize,
    coefficient_identities: usize,
    max_residual: i32,
}

fn potential_closure(sigma: &Chiral10D) -> PotentialClosureReport {
    // From Eq. (2.2), the coefficient of partial_mu A_nu in
    // {D_a,D_b}A_kappa is
    // (sigma_kappa)_{bc}(sigma^{mu nu})_a^c + (a<->b).
    // Eq. (2.6) requires 2 sigma^mu_ab delta_kappa,nu
    // -2 delta_mu,kappa sigma^nu_ab.
    let mut maximum = 0;
    let mut identities = 0;
    for a in 0..SPINOR {
        for b in a..SPINOR {
            for kappa in 0..VECTOR {
                for mu in 0..VECTOR {
                    for nu in 0..VECTOR {
                        let mut actual = 0;
                        for c in 0..SPINOR {
                            actual += sigma.sigma_lower[kappa].0[b][c]
                                * sigma.sigma_two[mu][nu].0[a][c]
                                + sigma.sigma_lower[kappa].0[a][c]
                                    * sigma.sigma_two[mu][nu].0[b][c];
                        }
                        let expected = 2 * sigma.sigma_up[mu].0[a][b] * i32::from(kappa == nu)
                            - 2 * i32::from(mu == kappa) * sigma.sigma_up[nu].0[a][b];
                        maximum = maximum.max((actual - expected).abs());
                        identities += 1;
                    }
                }
            }
        }
    }
    PotentialClosureReport {
        charge_pairs: 16 * 17 / 2,
        coefficient_identities: identities,
        max_residual: maximum,
    }
}

fn gaugino_closure_coefficient(
    sigma: &Chiral10D,
    sigma_two: &[Vec<M16>],
    a: usize,
    b: usize,
    c: usize,
    mu: usize,
    d: usize,
) -> i32 {
    // Coefficient of i partial_mu lambda_d in {D_a,D_b}lambda_c,
    // derived directly from Eq. (2.2).
    (0..VECTOR)
        .map(|nu| {
            sigma_two[mu][nu].0[b][c] * sigma.sigma_lower[nu].0[a][d]
                + sigma_two[mu][nu].0[a][c] * sigma.sigma_lower[nu].0[b][d]
        })
        .sum()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GauginoFactorizationReport {
    charge_pairs: usize,
    coefficient_identities: usize,
    nonzero_factor_matrices: usize,
    max_residual: i32,
    first_failure: Option<(usize, usize, usize, usize, usize, i32, i32)>,
}

fn gaugino_eom_factorization(
    sigma: &Chiral10D,
    sigma_two: &[Vec<M16>],
) -> GauginoFactorizationReport {
    // Each `(mu,d)` slot is the coefficient of D_mu Psi_d. The factorized
    // expression is K_ab,c^e E_e with
    // E_e = sum_mu (sigma^mu)_ed D_mu Psi_d.
    // The source's abelian example has D=partial. The coefficient identity is
    // unchanged by replacing that slot with the gauge-covariant derivative.
    let mut maximum = 0;
    let mut identities = 0;
    let mut nonzero_factors = 0;
    let mut first_failure = None;

    for a in 0..SPINOR {
        for b in a..SPINOR {
            // R_ab,c^(0,d)=K_ab,c^e sigma^0_ed=K_ab,c^d because sigma^0=I.
            // Thus the time-direction residual determines the unique candidate
            // factor K. The remaining nine directions are independent checks.
            let mut factor = [[0i32; SPINOR]; SPINOR];
            for c in 0..SPINOR {
                for d in 0..SPINOR {
                    let closure = gaugino_closure_coefficient(sigma, sigma_two, a, b, c, 0, d);
                    factor[c][d] = closure - 2 * sigma.sigma_up[0].0[a][b] * i32::from(c == d);
                }
            }
            if factor.iter().flatten().any(|&entry| entry != 0) {
                nonzero_factors += 1;
            }

            for mu in 0..VECTOR {
                for c in 0..SPINOR {
                    for d in 0..SPINOR {
                        let closure = gaugino_closure_coefficient(sigma, sigma_two, a, b, c, mu, d);
                        let residual = closure - 2 * sigma.sigma_up[mu].0[a][b] * i32::from(c == d);
                        let factored: i32 = (0..SPINOR)
                            .map(|e| factor[c][e] * sigma.sigma_up[mu].0[e][d])
                            .sum();
                        let difference = residual - factored;
                        maximum = maximum.max(difference.abs());
                        if difference != 0 && first_failure.is_none() {
                            first_failure = Some((a, b, mu, c, d, residual, factored));
                        }
                        identities += 1;
                    }
                }
            }
        }
    }

    GauginoFactorizationReport {
        charge_pairs: 16 * 17 / 2,
        coefficient_identities: identities,
        nonzero_factor_matrices: nonzero_factors,
        max_residual: maximum,
        first_failure,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScopeBoundary {
    system: &'static str,
    gauge_potential_components: usize,
    gaugino_components: usize,
    independent_auxiliary_components: usize,
    raw_component_fields: usize,
    supersymmetries: usize,
    closes_without_gaugino_eom: bool,
    claim_excluded: &'static str,
}

fn scope_boundary() -> ScopeBoundary {
    ScopeBoundary {
        system: "10D N=1 on-shell vector multiplet after eliminating the BBBM G auxiliaries",
        gauge_potential_components: 10,
        gaugino_components: 16,
        independent_auxiliary_components: 0,
        raw_component_fields: 26,
        supersymmetries: 16,
        closes_without_gaugino_eom: false,
        claim_excluded: "This is not a 16-charge extension of BBBM's 33-field auxiliary multiplet",
    }
}

#[test]
fn exact_real_chiral_matrices_satisfy_the_10d_clifford_relation() {
    let sigma = Chiral10D::build();
    assert_eq!(sigma.clifford_max_residual(), 0);
    for matrix in &sigma.sigma_up {
        assert_eq!(
            matrix,
            &matrix.transpose(),
            "chiral sigma must be symmetric"
        );
    }
}

#[test]
fn all_136_charge_pairs_close_on_the_potential_modulo_gauge() {
    let report = potential_closure(&Chiral10D::build());
    assert_eq!(report.charge_pairs, 136);
    assert_eq!(report.coefficient_identities, 136_000);
    assert_eq!(report.max_residual, 0, "potential closure tensor mismatch");
}

#[test]
fn all_136_gaugino_residuals_factor_exactly_through_dirac_eom() {
    let sigma = Chiral10D::build();
    let report = gaugino_eom_factorization(&sigma, &sigma.sigma_two);
    assert_eq!(report.charge_pairs, 136);
    assert_eq!(report.coefficient_identities, 348_160);
    assert_eq!(
        report.nonzero_factor_matrices, 136,
        "every charge pair has a nonzero gaugino EOM multiplier"
    );
    assert_eq!(report.max_residual, 0, "factorization failure: {report:?}");
    assert_eq!(report.first_failure, None);
}

#[test]
fn bivector_mutation_breaks_gaugino_eom_factorization() {
    let sigma = Chiral10D::build();
    let mut mutated = sigma.sigma_two.clone();
    assert_eq!(mutated[1][2].0[0][1].abs(), 1);
    mutated[1][2].0[0][1] *= -1;
    mutated[2][1] = mutated[1][2].scaled(-1);

    let report = gaugino_eom_factorization(&sigma, &mutated);
    assert!(report.max_residual > 0, "mutation was not detected");
    assert!(report.first_failure.is_some());
}

#[test]
fn report_excludes_a_sixteen_charge_auxiliary_claim() {
    let boundary = scope_boundary();
    assert_eq!(boundary.gauge_potential_components, 10);
    assert_eq!(boundary.gaugino_components, 16);
    assert_eq!(boundary.independent_auxiliary_components, 0);
    assert_eq!(boundary.raw_component_fields, 26);
    assert_eq!(boundary.supersymmetries, 16);
    assert!(!boundary.closes_without_gaugino_eom);
    assert!(boundary
        .claim_excluded
        .contains("not a 16-charge extension"));
    assert!(boundary.claim_excluded.contains("33-field auxiliary"));
}
