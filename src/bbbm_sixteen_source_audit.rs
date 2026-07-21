#![cfg(test)]
#![allow(clippy::needless_range_loop)]

// Source-boundary audit for the sixteen supersymmetry parameters surrounding
// the nine-charge auxiliary-field construction.
//
// Primary-source anchors:
//
// * N. Berkovits, hep-th/9308128, Eqs. (2)-(5): the transformations of
//   (A_mu, psi, G_a), the constraints on (epsilon, v_a), off-shell squaring
//   for one generalized transformation, and the restriction on mutually
//   closing generalized transformations. The text after Eq. (5) gives nine
//   as the maximum in this construction.
// * J. M. Evans, hep-th/9404190, Sec. 1, especially Eqs. (1.2), (1.3), and
//   (1.6): conventional transformations require v_a = M_a epsilon on a
//   restricted parameter subspace. Polarization then gives mutual off-shell
//   closure on that subspace.
// * L. Baulieu, N. Berkovits, G. Bossard, and A. Martin, arXiv:0705.2002v3,
//   Eqs. (2), (4), and (15)-(17): epsilon decomposes under Spin(7) as
//   1 + 7 + 8, and their covariant linear solution sets nu_ij = 0, leaving
//   the scalar plus vector parameters, 1 + 8 = 9.
//
// This file deliberately does not reconstruct transformations absent from
// those sources. It records the boundary between the printed nine-charge
// auxiliary-field system and ordinary ten-dimensional SYM with all sixteen
// supersymmetries closing on shell.
//
// Standalone test command:
//
//   rustc --edition 2024 --test src/bbbm_sixteen_source_audit.rs \
//     -o /tmp/bbbm_sixteen_source_audit
//   /tmp/bbbm_sixteen_source_audit

mod tests {
    const SCALAR_PARAMETERS: usize = 1;
    const VECTOR_PARAMETERS: usize = 8;
    const TENSOR_PARAMETERS: usize = 7;
    const FULL_PARAMETERS: usize = 16;
    const BBBM_PARAMETERS: usize = 9;

    const GAUGE_COMPONENTS: usize = 10;
    const GAUGINO_COMPONENTS: usize = 16;
    const AUXILIARY_COMPONENTS: usize = 7;
    const ON_SHELL_FIELDS: usize = GAUGE_COMPONENTS + GAUGINO_COMPONENTS;
    const AUXILIARY_EXTENDED_FIELDS: usize = ON_SHELL_FIELDS + AUXILIARY_COMPONENTS;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Spin7Parameter {
        Scalar,
        Vector(u8),
        AntiSelfDualTensor(u8),
    }

    fn all_parameters() -> Vec<Spin7Parameter> {
        let mut parameters = vec![Spin7Parameter::Scalar];
        parameters.extend((0..VECTOR_PARAMETERS).map(|i| Spin7Parameter::Vector(i as u8)));
        parameters
            .extend((0..TENSOR_PARAMETERS).map(|i| Spin7Parameter::AntiSelfDualTensor(i as u8)));
        parameters
    }

    fn has_printed_linear_v_solution(parameter: Spin7Parameter) -> bool {
        // arXiv:0705.2002, Eqs. (15)-(17): nu_ij is the seven-dimensional
        // antiselfdual tensor parameter, and the displayed solution imposes
        // nu_ij = 0. The supplied v(epsilon) therefore covers only omega and
        // epsilon^i.
        !matches!(parameter, Spin7Parameter::AntiSelfDualTensor(_))
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FieldSystem {
        OrdinaryOnShell26,
        AuxiliaryExtended33,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum VariationTarget {
        GaugePotential,
        Gaugino,
        Auxiliary,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct RequiredParameterData {
        epsilon: bool,
        v: bool,
    }

    fn variation_requirements(
        target: VariationTarget,
        system: FieldSystem,
    ) -> Option<RequiredParameterData> {
        match (system, target) {
            // Berkovits Eq. (2) and BBBM Eq. (2): delta A_mu uses epsilon.
            (_, VariationTarget::GaugePotential) => Some(RequiredParameterData {
                epsilon: true,
                v: false,
            }),
            // Before G_a is eliminated, delta psi contains both the ordinary
            // F_mu_nu Gamma^{mu nu} epsilon term and the G_a v_a term.
            (FieldSystem::AuxiliaryExtended33, VariationTarget::Gaugino) => {
                Some(RequiredParameterData {
                    epsilon: true,
                    v: true,
                })
            }
            // The auxiliary variation is proportional to v_a Gamma^mu D_mu psi.
            (FieldSystem::AuxiliaryExtended33, VariationTarget::Auxiliary) => {
                Some(RequiredParameterData {
                    epsilon: false,
                    v: true,
                })
            }
            // G_a = 0 leaves the ordinary gaugino variation, which uses epsilon.
            (FieldSystem::OrdinaryOnShell26, VariationTarget::Gaugino) => {
                Some(RequiredParameterData {
                    epsilon: true,
                    v: false,
                })
            }
            // G_a is not a field in the 26-component on-shell system.
            (FieldSystem::OrdinaryOnShell26, VariationTarget::Auxiliary) => None,
        }
    }

    fn variation_targets(system: FieldSystem) -> &'static [VariationTarget] {
        match system {
            FieldSystem::OrdinaryOnShell26 => {
                &[VariationTarget::GaugePotential, VariationTarget::Gaugino]
            }
            FieldSystem::AuxiliaryExtended33 => &[
                VariationTarget::GaugePotential,
                VariationTarget::Gaugino,
                VariationTarget::Auxiliary,
            ],
        }
    }

    fn variation_is_sourced(
        parameter: Spin7Parameter,
        target: VariationTarget,
        system: FieldSystem,
    ) -> bool {
        let Some(requirements) = variation_requirements(target, system) else {
            return false;
        };
        let epsilon_is_sourced = true;
        let v_is_sourced = has_printed_linear_v_solution(parameter);
        (!requirements.epsilon || epsilon_is_sourced) && (!requirements.v || v_is_sourced)
    }

    fn complete_transformation_is_sourced(parameter: Spin7Parameter, system: FieldSystem) -> bool {
        variation_targets(system)
            .iter()
            .all(|&target| variation_is_sourced(parameter, target, system))
    }

    const fn symmetric_charge_pairs(charges: usize) -> usize {
        charges * (charges + 1) / 2
    }

    const fn field_relations(charges: usize, fields: usize) -> usize {
        symmetric_charge_pairs(charges) * fields
    }

    type Matrix = Vec<Vec<i64>>;

    fn zero_matrix(rows: usize, columns: usize) -> Matrix {
        vec![vec![0; columns]; rows]
    }

    fn identity(dimension: usize) -> Matrix {
        let mut result = zero_matrix(dimension, dimension);
        for (i, row) in result.iter_mut().enumerate() {
            row[i] = 1;
        }
        result
    }

    fn diagonal_projector(dimension: usize, retained: impl IntoIterator<Item = usize>) -> Matrix {
        let mut result = zero_matrix(dimension, dimension);
        for i in retained {
            result[i][i] = 1;
        }
        result
    }

    fn matrix_add(left: &Matrix, right: &Matrix) -> Matrix {
        left.iter()
            .zip(right)
            .map(|(left_row, right_row)| {
                left_row.iter().zip(right_row).map(|(x, y)| x + y).collect()
            })
            .collect()
    }

    fn matrix_multiply(left: &Matrix, right: &Matrix) -> Matrix {
        let rows = left.len();
        let inner = right.len();
        let columns = right.first().map_or(0, Vec::len);
        assert!(left.iter().all(|row| row.len() == inner));
        let mut result = zero_matrix(rows, columns);
        for i in 0..rows {
            for k in 0..inner {
                for j in 0..columns {
                    result[i][j] += left[i][k] * right[k][j];
                }
            }
        }
        result
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Rational {
        numerator: i128,
        denominator: i128,
    }

    impl Rational {
        const ZERO: Self = Self {
            numerator: 0,
            denominator: 1,
        };

        fn new(mut numerator: i128, mut denominator: i128) -> Self {
            assert_ne!(denominator, 0);
            if numerator == 0 {
                return Self::ZERO;
            }
            if denominator < 0 {
                numerator = -numerator;
                denominator = -denominator;
            }
            let divisor = gcd(numerator.unsigned_abs(), denominator as u128) as i128;
            Self {
                numerator: numerator / divisor,
                denominator: denominator / divisor,
            }
        }
    }

    const fn gcd(mut left: u128, mut right: u128) -> u128 {
        while right != 0 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        left
    }

    impl std::ops::Sub for Rational {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(
                self.numerator * rhs.denominator - rhs.numerator * self.denominator,
                self.denominator * rhs.denominator,
            )
        }
    }

    impl std::ops::Mul for Rational {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self::new(
                self.numerator * rhs.numerator,
                self.denominator * rhs.denominator,
            )
        }
    }

    impl std::ops::Div for Rational {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            assert_ne!(rhs, Self::ZERO);
            Self::new(
                self.numerator * rhs.denominator,
                self.denominator * rhs.numerator,
            )
        }
    }

    impl From<i64> for Rational {
        fn from(value: i64) -> Self {
            Self::new(value as i128, 1)
        }
    }

    fn exact_rank(matrix: &Matrix) -> usize {
        let mut work: Vec<Vec<Rational>> = matrix
            .iter()
            .map(|row| row.iter().copied().map(Rational::from).collect())
            .collect();
        let rows = work.len();
        let columns = work.first().map_or(0, Vec::len);
        let mut rank = 0;

        for column in 0..columns {
            let Some(pivot) = (rank..rows).find(|&row| work[row][column] != Rational::ZERO) else {
                continue;
            };
            work.swap(rank, pivot);
            let pivot_value = work[rank][column];
            for j in column..columns {
                work[rank][j] = work[rank][j] / pivot_value;
            }
            for row in 0..rows {
                if row == rank || work[row][column] == Rational::ZERO {
                    continue;
                }
                let factor = work[row][column];
                for j in column..columns {
                    work[row][j] = work[row][j] - factor * work[rank][j];
                }
            }
            rank += 1;
        }
        rank
    }

    // A normalized Cayley four-form in one oriented orthonormal basis. BBBM
    // Eq. (13) defines Omega through the invariant spinor but does not select a
    // component basis. The rank statements below are basis independent.
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
        if inversions % 2 == 0 {
            1
        } else {
            -1
        }
    }

    fn omega(indices: [usize; 4]) -> i64 {
        if (0..4).any(|i| ((i + 1)..4).any(|j| indices[i] == indices[j])) {
            return 0;
        }
        let mut sorted = indices;
        sorted.sort_unstable();
        let orientation = permutation_sign(&indices);
        orientation
            * CAYLEY_TERMS
                .iter()
                .find_map(|(term, coefficient)| (*term == sorted).then_some(*coefficient))
                .unwrap_or(0)
    }

    fn two_form_pairs() -> Vec<(usize, usize)> {
        (0..8)
            .flat_map(|i| ((i + 1)..8).map(move |j| (i, j)))
            .collect()
    }

    // On the 28 independent coordinates X_kl with k<l, four times BBBM's
    // Eq. (14) antiselfdual projector is N^- = I - Omega_pair. The factor 1/2
    // printed with Omega in Eq. (14) cancels the factor two from summing the
    // ordered components X_kl and X_lk.
    fn anti_self_dual_projector_numerator() -> Matrix {
        let pairs = two_form_pairs();
        let mut result = zero_matrix(pairs.len(), pairs.len());
        for (row, &(i, j)) in pairs.iter().enumerate() {
            for (column, &(k, l)) in pairs.iter().enumerate() {
                result[row][column] = i64::from(row == column) - omega([i, j, k, l]);
            }
        }
        result
    }

    #[test]
    fn source_parameter_space_is_one_plus_eight_plus_seven() {
        let parameters = all_parameters();
        assert_eq!(parameters.len(), FULL_PARAMETERS);
        assert_eq!(
            parameters
                .iter()
                .filter(|parameter| matches!(parameter, Spin7Parameter::Scalar))
                .count(),
            SCALAR_PARAMETERS
        );
        assert_eq!(
            parameters
                .iter()
                .filter(|parameter| matches!(parameter, Spin7Parameter::Vector(_)))
                .count(),
            VECTOR_PARAMETERS
        );
        assert_eq!(
            parameters
                .iter()
                .filter(|parameter| matches!(parameter, Spin7Parameter::AntiSelfDualTensor(_)))
                .count(),
            TENSOR_PARAMETERS
        );
        assert_eq!(
            SCALAR_PARAMETERS + VECTOR_PARAMETERS + TENSOR_PARAMETERS,
            FULL_PARAMETERS
        );
    }

    #[test]
    fn equations_15_to_17_source_nine_conventional_auxiliary_transformations() {
        let parameters = all_parameters();
        let sourced: Vec<_> = parameters
            .iter()
            .copied()
            .filter(|&parameter| has_printed_linear_v_solution(parameter))
            .collect();
        let omitted: Vec<_> = parameters
            .iter()
            .copied()
            .filter(|&parameter| !has_printed_linear_v_solution(parameter))
            .collect();

        assert_eq!(sourced.len(), BBBM_PARAMETERS);
        assert!(sourced
            .iter()
            .all(|parameter| !matches!(parameter, Spin7Parameter::AntiSelfDualTensor(_))));
        assert_eq!(omitted.len(), TENSOR_PARAMETERS);
        assert!(omitted
            .iter()
            .all(|parameter| matches!(parameter, Spin7Parameter::AntiSelfDualTensor(_))));
    }

    #[test]
    fn seven_ordinary_transformations_are_fully_sourced_only_on_physical_fields() {
        let omitted: Vec<_> = all_parameters()
            .into_iter()
            .filter(|&parameter| !has_printed_linear_v_solution(parameter))
            .collect();
        assert_eq!(omitted.len(), TENSOR_PARAMETERS);

        for parameter in omitted {
            // The gauge-potential line is not the obstruction. It contains no
            // v_a in either version of the field system.
            assert!(variation_is_sourced(
                parameter,
                VariationTarget::GaugePotential,
                FieldSystem::AuxiliaryExtended33
            ));
            // The complete gaugino and auxiliary lines on the 33-component
            // system cannot be formed without a sourced v_a(parameter).
            assert!(!variation_is_sourced(
                parameter,
                VariationTarget::Gaugino,
                FieldSystem::AuxiliaryExtended33
            ));
            assert!(!variation_is_sourced(
                parameter,
                VariationTarget::Auxiliary,
                FieldSystem::AuxiliaryExtended33
            ));
            assert!(complete_transformation_is_sourced(
                parameter,
                FieldSystem::OrdinaryOnShell26
            ));
            assert!(!complete_transformation_is_sourced(
                parameter,
                FieldSystem::AuxiliaryExtended33
            ));
        }
    }

    #[test]
    fn exact_relation_counts_distinguish_sourced_and_would_be_systems() {
        assert_eq!(ON_SHELL_FIELDS, 26);
        assert_eq!(AUXILIARY_EXTENDED_FIELDS, 33);
        assert_eq!(symmetric_charge_pairs(FULL_PARAMETERS), 136);
        assert_eq!(field_relations(FULL_PARAMETERS, ON_SHELL_FIELDS), 3_536);
        assert_eq!(
            field_relations(FULL_PARAMETERS, AUXILIARY_EXTENDED_FIELDS),
            4_488
        );

        // This smaller number is the part the printed 9-of-16 auxiliary-field
        // solution actually makes available for an off-shell closure audit.
        assert_eq!(symmetric_charge_pairs(BBBM_PARAMETERS), 45);
        assert_eq!(
            field_relations(BBBM_PARAMETERS, AUXILIARY_EXTENDED_FIELDS),
            1_485
        );
    }

    #[test]
    fn spin_seven_parameter_projectors_have_exact_ranks_one_eight_and_seven() {
        // Choose a basis of 8_+ aligned with BBBM's invariant spinor zeta.
        // Coordinate 0 is the singlet omega*zeta and coordinates 1..7 are its
        // seven-dimensional complement. Coordinates 8..15 form 8_-.
        let scalar = diagonal_projector(FULL_PARAMETERS, [0]);
        let tensor = diagonal_projector(FULL_PARAMETERS, 1..8);
        let vector = diagonal_projector(FULL_PARAMETERS, 8..16);

        assert_eq!(exact_rank(&scalar), SCALAR_PARAMETERS);
        assert_eq!(exact_rank(&vector), VECTOR_PARAMETERS);
        assert_eq!(exact_rank(&tensor), TENSOR_PARAMETERS);
        assert_eq!(matrix_multiply(&scalar, &scalar), scalar);
        assert_eq!(matrix_multiply(&vector, &vector), vector);
        assert_eq!(matrix_multiply(&tensor, &tensor), tensor);
        assert_eq!(
            matrix_multiply(&scalar, &vector),
            zero_matrix(FULL_PARAMETERS, FULL_PARAMETERS)
        );
        assert_eq!(
            matrix_multiply(&scalar, &tensor),
            zero_matrix(FULL_PARAMETERS, FULL_PARAMETERS)
        );
        assert_eq!(
            matrix_multiply(&vector, &tensor),
            zero_matrix(FULL_PARAMETERS, FULL_PARAMETERS)
        );
        assert_eq!(
            matrix_add(&matrix_add(&scalar, &vector), &tensor),
            identity(16)
        );
    }

    #[test]
    fn equation_14_antiselfdual_two_form_projector_has_exact_rank_seven() {
        let numerator = anti_self_dual_projector_numerator();
        assert_eq!(numerator.len(), 28);
        assert_eq!(exact_rank(&numerator), TENSOR_PARAMETERS);

        // Since P^- = N^-/4, idempotence is the integer identity N^-N^-=4N^-.
        let squared = matrix_multiply(&numerator, &numerator);
        let four_times: Matrix = numerator
            .iter()
            .map(|row| row.iter().map(|entry| 4 * entry).collect())
            .collect();
        assert_eq!(squared, four_times);

        let complement_numerator: Matrix = identity(28)
            .iter()
            .zip(&numerator)
            .map(|(identity_row, numerator_row)| {
                identity_row
                    .iter()
                    .zip(numerator_row)
                    .map(|(identity_entry, numerator_entry)| 4 * identity_entry - numerator_entry)
                    .collect()
            })
            .collect();
        assert_eq!(exact_rank(&complement_numerator), 21);
    }
}
