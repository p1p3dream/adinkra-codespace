//! Exact target-side curvature, Bianchi, and field-equation complexes for
//! linearized eleven-dimensional supergravity.
//!
//! This module lifts the numeric symbols in
//! [`crate::eleven_dimensional_free_complex`] to polynomials in all eleven
//! formal momentum variables.  It then factors each free field equation
//! through its gauge-invariant curvature.  The fermionic factor uses the
//! repository's exact real Majorana gamma matrices.
//!
//! This is a target-side on-shell complex.  It does not construct a map from
//! an unconstrained scalar superfield, a physical prepotential, or any other
//! source into these curvatures.  It also does not establish off-shell
//! supersymmetry closure.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use num_rational::Ratio;
use serde::Serialize;

pub const VECTOR_DIMENSION: usize = 11;
pub const SPINOR_DIMENSION: usize = 32;

type Q = Ratio<i64>;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Qi {
    real: Q,
    imaginary: Q,
}

impl Qi {
    fn zero() -> Self {
        Self {
            real: Q::from_integer(0),
            imaginary: Q::from_integer(0),
        }
    }

    fn rational(numerator: i64, denominator: i64) -> Self {
        Self {
            real: Q::new(numerator, denominator),
            imaginary: Q::from_integer(0),
        }
    }

    fn from_public(value: crate::eleven_dimensional_free_complex::ExactCoefficient) -> Self {
        Self {
            real: Q::new(value.real_numerator, value.real_denominator),
            imaginary: Q::new(value.imaginary_numerator, value.imaginary_denominator),
        }
    }

    fn is_zero(&self) -> bool {
        self.real == Q::from_integer(0) && self.imaginary == Q::from_integer(0)
    }

    fn add_assign(&mut self, other: &Self) {
        self.real += other.real.clone();
        self.imaginary += other.imaginary.clone();
    }

    fn subtract(&self, other: &Self) -> Self {
        Self {
            real: self.real.clone() - other.real.clone(),
            imaginary: self.imaginary.clone() - other.imaginary.clone(),
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        Self {
            real: self.real.clone() * other.real.clone()
                - self.imaginary.clone() * other.imaginary.clone(),
            imaginary: self.real.clone() * other.imaginary.clone()
                + self.imaginary.clone() * other.real.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct MomentumMonomial {
    pub exponents: [u8; VECTOR_DIMENSION],
}

impl MomentumMonomial {
    pub fn constant() -> Self {
        Self {
            exponents: [0; VECTOR_DIMENSION],
        }
    }

    pub fn variable(axis: usize) -> Self {
        assert!(axis < VECTOR_DIMENSION);
        let mut exponents = [0; VECTOR_DIMENSION];
        exponents[axis] = 1;
        Self { exponents }
    }

    fn square(axis: usize) -> Self {
        let mut result = Self::variable(axis);
        result.exponents[axis] = 2;
        result
    }

    fn pair(left: usize, right: usize) -> Self {
        let mut result = Self::variable(left);
        result.exponents[right] += 1;
        result
    }

    fn multiply(&self, other: &Self) -> Self {
        Self {
            exponents: std::array::from_fn(|axis| self.exponents[axis] + other.exponents[axis]),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExactPolynomialCoefficient {
    pub monomial: MomentumMonomial,
    pub real_numerator: i64,
    pub real_denominator: i64,
    pub imaginary_numerator: i64,
    pub imaginary_denominator: i64,
}

/// Sparse exact matrix over `Q(i)[p_0,...,p_10]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactPolynomialMatrix {
    rows: usize,
    columns: usize,
    entries: BTreeMap<(usize, usize, MomentumMonomial), Qi>,
}

impl ExactPolynomialMatrix {
    pub fn zero(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            entries: BTreeMap::new(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn nonzero_terms(&self) -> usize {
        self.entries.len()
    }

    pub fn is_zero(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn coefficient_terms(&self, row: usize, column: usize) -> Vec<ExactPolynomialCoefficient> {
        assert!(row < self.rows && column < self.columns);
        self.entries
            .iter()
            .filter(|((entry_row, entry_column, _), _)| {
                *entry_row == row && *entry_column == column
            })
            .map(|((_, _, monomial), value)| ExactPolynomialCoefficient {
                monomial: monomial.clone(),
                real_numerator: *value.real.numer(),
                real_denominator: *value.real.denom(),
                imaginary_numerator: *value.imaginary.numer(),
                imaginary_denominator: *value.imaginary.denom(),
            })
            .collect()
    }

    fn add_term(&mut self, row: usize, column: usize, monomial: MomentumMonomial, value: Qi) {
        assert!(row < self.rows && column < self.columns);
        if value.is_zero() {
            return;
        }
        let key = (row, column, monomial);
        let entry = self.entries.entry(key.clone()).or_insert_with(Qi::zero);
        entry.add_assign(&value);
        if entry.is_zero() {
            self.entries.remove(&key);
        }
    }

    pub fn multiply(&self, right: &Self) -> Self {
        assert_eq!(self.columns, right.rows);
        let mut right_rows = vec![Vec::new(); right.rows];
        for ((row, column, monomial), coefficient) in &right.entries {
            right_rows[*row].push((*column, monomial.clone(), coefficient.clone()));
        }
        let mut output = Self::zero(self.rows, right.columns);
        for ((row, pivot, left_monomial), left_coefficient) in &self.entries {
            for (column, right_monomial, right_coefficient) in &right_rows[*pivot] {
                output.add_term(
                    *row,
                    *column,
                    left_monomial.multiply(right_monomial),
                    left_coefficient.multiply(right_coefficient),
                );
            }
        }
        output
    }

    fn subtract(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.columns), (other.rows, other.columns));
        let mut output = self.clone();
        for ((row, column, monomial), value) in &other.entries {
            output.add_term(*row, *column, monomial.clone(), Qi::zero().subtract(value));
        }
        output
    }

    fn mutate_first_term(&self) -> Self {
        let mut output = self.clone();
        let key = output.entries.keys().next().cloned().expect("nonzero map");
        output.entries.get_mut(&key).unwrap().real += Q::from_integer(1);
        if output.entries[&key].is_zero() {
            output.entries.remove(&key);
        }
        output
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum TargetSector {
    Graviton,
    FourForm,
    RaritaSchwinger,
}

impl TargetSector {
    fn name(self) -> &'static str {
        match self {
            Self::Graviton => "Pauli-Fierz graviton",
            Self::FourForm => "four-form curvature of the Abelian three-form",
            Self::RaritaSchwinger => "Rarita-Schwinger vector-spinor",
        }
    }
}

/// Formal target-side maps for one free field sector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetSectorComplex {
    pub sector: TargetSector,
    pub reducibility: Vec<ExactPolynomialMatrix>,
    pub gauge: ExactPolynomialMatrix,
    pub curvature: ExactPolynomialMatrix,
    pub bianchi: ExactPolynomialMatrix,
    pub curvature_to_euler: ExactPolynomialMatrix,
    pub euler_lagrange: ExactPolynomialMatrix,
    pub reference_euler_lagrange: Option<ExactPolynomialMatrix>,
    pub noether: ExactPolynomialMatrix,
}

fn add_public_matrix(
    target: &mut ExactPolynomialMatrix,
    source: &crate::eleven_dimensional_free_complex::SparseExactMatrix,
    monomial: MomentumMonomial,
    sign: i64,
) {
    assert_eq!(
        (target.rows, target.columns),
        (source.rows(), source.columns())
    );
    for (row, column, coefficient) in source.nonzero_coefficients() {
        let mut value = Qi::from_public(coefficient);
        if sign == -1 {
            value = Qi::zero().subtract(&value);
        }
        target.add_term(row, column, monomial.clone(), value);
    }
}

fn lift_linear(
    builder: impl Fn(
        [i64; VECTOR_DIMENSION],
    ) -> crate::eleven_dimensional_free_complex::SparseExactMatrix,
) -> ExactPolynomialMatrix {
    let mut basis = [0_i64; VECTOR_DIMENSION];
    basis[0] = 1;
    let sample = builder(basis);
    let mut output = ExactPolynomialMatrix::zero(sample.rows(), sample.columns());
    for axis in 0..VECTOR_DIMENSION {
        let mut momentum = [0_i64; VECTOR_DIMENSION];
        momentum[axis] = 1;
        add_public_matrix(
            &mut output,
            &builder(momentum),
            MomentumMonomial::variable(axis),
            1,
        );
    }
    output
}

fn lift_quadratic(
    builder: impl Fn(
        [i64; VECTOR_DIMENSION],
    ) -> crate::eleven_dimensional_free_complex::SparseExactMatrix,
) -> ExactPolynomialMatrix {
    let evaluations = (0..VECTOR_DIMENSION)
        .map(|axis| {
            let mut momentum = [0_i64; VECTOR_DIMENSION];
            momentum[axis] = 1;
            builder(momentum)
        })
        .collect::<Vec<_>>();
    let mut output = ExactPolynomialMatrix::zero(evaluations[0].rows(), evaluations[0].columns());
    for axis in 0..VECTOR_DIMENSION {
        add_public_matrix(
            &mut output,
            &evaluations[axis],
            MomentumMonomial::square(axis),
            1,
        );
    }
    for left in 0..VECTOR_DIMENSION {
        for right in (left + 1)..VECTOR_DIMENSION {
            let mut momentum = [0_i64; VECTOR_DIMENSION];
            momentum[left] = 1;
            momentum[right] = 1;
            let mixed = builder(momentum);
            let mut coefficient = ExactPolynomialMatrix::zero(mixed.rows(), mixed.columns());
            add_public_matrix(&mut coefficient, &mixed, MomentumMonomial::constant(), 1);
            add_public_matrix(
                &mut coefficient,
                &evaluations[left],
                MomentumMonomial::constant(),
                -1,
            );
            add_public_matrix(
                &mut coefficient,
                &evaluations[right],
                MomentumMonomial::constant(),
                -1,
            );
            for ((row, column, _), value) in coefficient.entries {
                output.add_term(row, column, MomentumMonomial::pair(left, right), value);
            }
        }
    }
    output
}

fn combinations(degree: usize) -> Vec<Vec<usize>> {
    fn extend(
        next: usize,
        remaining: usize,
        prefix: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(prefix.clone());
            return;
        }
        for value in next..=VECTOR_DIMENSION - remaining {
            prefix.push(value);
            extend(value + 1, remaining - 1, prefix, output);
            prefix.pop();
        }
    }
    let mut output = Vec::new();
    extend(0, degree, &mut Vec::new(), &mut output);
    output
}

fn symmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| (left..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn antisymmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn metric_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn oriented_pair(
    lookup: &BTreeMap<(usize, usize), usize>,
    left: usize,
    right: usize,
) -> Option<(usize, i64)> {
    if left < right {
        Some((lookup[&(left, right)], 1))
    } else if right < left {
        Some((lookup[&(right, left)], -1))
    } else {
        None
    }
}

fn graviton_curvature_to_euler() -> ExactPolynomialMatrix {
    let fields = symmetric_pairs();
    let pairs = antisymmetric_pairs();
    let pair_lookup = pairs
        .iter()
        .copied()
        .enumerate()
        .map(|(index, pair)| (pair, index))
        .collect::<BTreeMap<_, _>>();
    let mut output = ExactPolynomialMatrix::zero(fields.len(), pairs.len() * pairs.len());
    for (row, &(b, d)) in fields.iter().enumerate() {
        // -Ricci_bd, with Ricci_bd=eta^{ac} R_{a b|c d}.
        for a in 0..VECTOR_DIMENSION {
            let Some((left, left_sign)) = oriented_pair(&pair_lookup, a, b) else {
                continue;
            };
            let Some((right, right_sign)) = oriented_pair(&pair_lookup, a, d) else {
                continue;
            };
            output.add_term(
                row,
                left * pairs.len() + right,
                MomentumMonomial::constant(),
                Qi::rational(-metric_sign(a) * left_sign * right_sign, 1),
            );
        }
        // +(1/2) eta_bd R.  The diagonal metric makes this term vanish for
        // off-diagonal output components.
        if b == d {
            for trace_axis in 0..VECTOR_DIMENSION {
                for a in 0..VECTOR_DIMENSION {
                    let Some((left, left_sign)) = oriented_pair(&pair_lookup, a, trace_axis) else {
                        continue;
                    };
                    let coefficient = metric_sign(b)
                        * metric_sign(trace_axis)
                        * metric_sign(a)
                        * left_sign
                        * left_sign;
                    output.add_term(
                        row,
                        left * pairs.len() + left,
                        MomentumMonomial::constant(),
                        Qi::rational(coefficient, 2),
                    );
                }
            }
        }
    }
    output
}

fn four_form_curvature_to_euler() -> ExactPolynomialMatrix {
    let source = combinations(4);
    let target = combinations(3);
    let target_lookup = target
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, indices)| (indices, index))
        .collect::<BTreeMap<_, _>>();
    let mut output = ExactPolynomialMatrix::zero(target.len(), source.len());
    for (column, indices) in source.iter().enumerate() {
        for (position, &axis) in indices.iter().enumerate() {
            let mut remaining = indices.clone();
            remaining.remove(position);
            let sign = if position % 2 == 0 { 1 } else { -1 };
            output.add_term(
                target_lookup[&remaining],
                column,
                MomentumMonomial::variable(axis),
                Qi::rational(sign * metric_sign(axis), 1),
            );
        }
    }
    output
}

fn multiply_i8(left: &[Vec<i8>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            if left[row][pivot] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] +=
                    i16::from(left[row][pivot]) * i16::from(right[pivot][column]);
            }
        }
    }
    output
}

fn multiply_i16_i8(left: &[Vec<i16>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            if left[row][pivot] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += left[row][pivot] * i16::from(right[pivot][column]);
            }
        }
    }
    output
}

fn rarita_schwinger_curvature_to_euler() -> ExactPolynomialMatrix {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let pairs = antisymmetric_pairs();
    let mut output = ExactPolynomialMatrix::zero(
        VECTOR_DIMENSION * SPINOR_DIMENSION,
        pairs.len() * SPINOR_DIMENSION,
    );
    for a in 0..VECTOR_DIMENSION {
        for (pair, &(b, c)) in pairs.iter().enumerate() {
            if a == b || a == c {
                continue;
            }
            let gamma_ab = multiply_i8(&gammas[a], &gammas[b]);
            let gamma_abc = multiply_i16_i8(&gamma_ab, &gammas[c]);
            for output_spinor in 0..SPINOR_DIMENSION {
                for input_spinor in 0..SPINOR_DIMENSION {
                    let coefficient = gamma_abc[output_spinor][input_spinor];
                    if coefficient != 0 {
                        output.add_term(
                            a * SPINOR_DIMENSION + output_spinor,
                            pair * SPINOR_DIMENSION + input_spinor,
                            MomentumMonomial::constant(),
                            Qi::rational(i64::from(coefficient), 1),
                        );
                    }
                }
            }
        }
    }
    output
}

fn lift_sector(sector: TargetSector) -> TargetSectorComplex {
    use crate::eleven_dimensional_free_complex::{
        gravitino_complex, graviton_complex, three_form_complex,
    };
    match sector {
        TargetSector::Graviton => {
            let gauge = lift_linear(|p| graviton_complex(p).gauge);
            let curvature = lift_quadratic(|p| graviton_complex(p).curvature);
            let bianchi = lift_linear(|p| graviton_complex(p).bianchi);
            let curvature_to_euler = graviton_curvature_to_euler();
            let reference_euler_lagrange = lift_quadratic(|p| graviton_complex(p).euler_lagrange);
            let euler_lagrange = reference_euler_lagrange.clone();
            let noether = lift_linear(|p| graviton_complex(p).noether);
            TargetSectorComplex {
                sector,
                reducibility: Vec::new(),
                gauge,
                curvature,
                bianchi,
                curvature_to_euler,
                euler_lagrange,
                reference_euler_lagrange: Some(reference_euler_lagrange),
                noether,
            }
        }
        TargetSector::FourForm => {
            let reducibility_zero = lift_linear(|p| three_form_complex(p).reducibility[0].clone());
            let reducibility_one = lift_linear(|p| three_form_complex(p).reducibility[1].clone());
            let gauge = lift_linear(|p| three_form_complex(p).gauge);
            let curvature = lift_linear(|p| three_form_complex(p).curvature);
            let bianchi = lift_linear(|p| three_form_complex(p).bianchi);
            let curvature_to_euler = four_form_curvature_to_euler();
            let reference_euler_lagrange = lift_quadratic(|p| three_form_complex(p).euler_lagrange);
            let euler_lagrange = reference_euler_lagrange.clone();
            let noether = lift_linear(|p| three_form_complex(p).noether);
            TargetSectorComplex {
                sector,
                reducibility: vec![reducibility_zero, reducibility_one],
                gauge,
                curvature,
                bianchi,
                curvature_to_euler,
                euler_lagrange,
                reference_euler_lagrange: Some(reference_euler_lagrange),
                noether,
            }
        }
        TargetSector::RaritaSchwinger => {
            let gauge = lift_linear(|p| gravitino_complex(p).gauge);
            let curvature = lift_linear(|p| gravitino_complex(p).curvature);
            let bianchi = lift_linear(|p| gravitino_complex(p).bianchi);
            let curvature_to_euler = rarita_schwinger_curvature_to_euler();
            let euler_lagrange = curvature_to_euler.multiply(&curvature);
            let noether = lift_linear(|p| gravitino_complex(p).noether);
            TargetSectorComplex {
                sector,
                reducibility: Vec::new(),
                gauge,
                curvature,
                bianchi,
                curvature_to_euler,
                euler_lagrange,
                reference_euler_lagrange: None,
                noether,
            }
        }
    }
}

/// Build one typed exact target sector over all eleven formal momenta.
pub fn target_sector_complex(sector: TargetSector) -> &'static TargetSectorComplex {
    static GRAVITON: OnceLock<TargetSectorComplex> = OnceLock::new();
    static FOUR_FORM: OnceLock<TargetSectorComplex> = OnceLock::new();
    static RARITA_SCHWINGER: OnceLock<TargetSectorComplex> = OnceLock::new();
    match sector {
        TargetSector::Graviton => GRAVITON.get_or_init(|| lift_sector(sector)),
        TargetSector::FourForm => FOUR_FORM.get_or_init(|| lift_sector(sector)),
        TargetSector::RaritaSchwinger => RARITA_SCHWINGER.get_or_init(|| lift_sector(sector)),
    }
}

/// Typed coordinate in the future physical source-to-target curvature map.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TargetCurvatureCoordinate {
    pub sector: TargetCurvatureSector,
    pub component: usize,
    pub momentum: MomentumMonomial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub enum TargetCurvatureSector {
    LinearizedRiemann,
    FourForm,
    GravitinoCurl,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFAdapterDescriptor {
    pub schema_version: String,
    pub source_basis: String,
    pub target_basis: String,
    pub generic_formal_momentum_complete: bool,
    pub physical_source_map_complete: bool,
}

/// Adapter boundary for the future source-selected physical curvature `F`.
/// Implementations must return target curvature coordinates, never field
/// equations directly, so the Bianchi and Euler gates remain independent.
pub trait PhysicalFTargetAdapter {
    type SourceCoordinate;
    type Coefficient;

    fn descriptor(&self) -> PhysicalFAdapterDescriptor;

    fn apply_source_coordinate(
        &self,
        source: &Self::SourceCoordinate,
        coefficient: &Self::Coefficient,
    ) -> Result<Vec<(TargetCurvatureCoordinate, Self::Coefficient)>, String>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TargetSectorComplexReport {
    pub sector: &'static str,
    pub gauge_dimensions: (usize, usize),
    pub curvature_dimensions: (usize, usize),
    pub bianchi_dimensions: (usize, usize),
    pub curvature_to_euler_dimensions: (usize, usize),
    pub euler_dimensions: (usize, usize),
    pub noether_dimensions: (usize, usize),
    pub reducibility_compositions_residual_terms: usize,
    pub curvature_after_gauge_residual_terms: usize,
    pub bianchi_after_curvature_residual_terms: usize,
    pub euler_factorization_residual_terms: usize,
    pub reference_euler_symbol_compared: bool,
    pub reference_euler_agreement_residual_terms: usize,
    pub euler_after_gauge_residual_terms: usize,
    pub noether_after_euler_residual_terms: usize,
    pub mutation_residual_terms: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TargetEquationComplexReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub signature: &'static str,
    pub formal_momentum_variables: usize,
    pub graviton: TargetSectorComplexReport,
    pub four_form: TargetSectorComplexReport,
    pub rarita_schwinger: TargetSectorComplexReport,
    pub generic_formal_compositions_certified: bool,
    pub null_momentum: [i64; VECTOR_DIMENSION],
    pub null_graviton_physical_dimension: usize,
    pub null_three_form_physical_dimension: usize,
    pub null_rarita_schwinger_physical_dimension: usize,
    pub null_bosonic_dimension: usize,
    pub null_fermionic_dimension: usize,
    pub majorana_real_form_certified: bool,
    pub light_cone_susy_maps_certified: bool,
    pub light_cone_susy_closure_residual_entries: usize,
    pub physical_f_adapter_api_available: bool,
    pub physical_source_to_target_f_constructed: bool,
    pub source_to_field_equation_operator_constructed: bool,
    pub off_shell_closure_established: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn report_sector(complex: &TargetSectorComplex) -> TargetSectorComplexReport {
    let reducibility_residual = if complex.reducibility.is_empty() {
        0
    } else {
        complex.reducibility[1]
            .multiply(&complex.reducibility[0])
            .nonzero_terms()
            + complex
                .gauge
                .multiply(&complex.reducibility[1])
                .nonzero_terms()
    };
    let curvature_after_gauge = complex.curvature.multiply(&complex.gauge);
    let bianchi_after_curvature = complex.bianchi.multiply(&complex.curvature);
    let euler_from_curvature = complex.curvature_to_euler.multiply(&complex.curvature);
    let euler_factorization = euler_from_curvature.subtract(&complex.euler_lagrange);
    let reference_euler_agreement = complex
        .reference_euler_lagrange
        .as_ref()
        .map(|reference| complex.euler_lagrange.subtract(reference));
    let euler_after_gauge = complex.euler_lagrange.multiply(&complex.gauge);
    let noether_after_euler = complex.noether.multiply(&complex.euler_lagrange);
    let mutated = complex
        .curvature_to_euler
        .mutate_first_term()
        .multiply(&complex.curvature)
        .subtract(&complex.euler_lagrange);
    let mutation_residual_terms = mutated.nonzero_terms();
    let passed = reducibility_residual == 0
        && curvature_after_gauge.is_zero()
        && bianchi_after_curvature.is_zero()
        && euler_factorization.is_zero()
        && euler_after_gauge.is_zero()
        && noether_after_euler.is_zero()
        && mutation_residual_terms > 0;
    TargetSectorComplexReport {
        sector: complex.sector.name(),
        gauge_dimensions: (complex.gauge.rows, complex.gauge.columns),
        curvature_dimensions: (complex.curvature.rows, complex.curvature.columns),
        bianchi_dimensions: (complex.bianchi.rows, complex.bianchi.columns),
        curvature_to_euler_dimensions: (
            complex.curvature_to_euler.rows,
            complex.curvature_to_euler.columns,
        ),
        euler_dimensions: (complex.euler_lagrange.rows, complex.euler_lagrange.columns),
        noether_dimensions: (complex.noether.rows, complex.noether.columns),
        reducibility_compositions_residual_terms: reducibility_residual,
        curvature_after_gauge_residual_terms: curvature_after_gauge.nonzero_terms(),
        bianchi_after_curvature_residual_terms: bianchi_after_curvature.nonzero_terms(),
        euler_factorization_residual_terms: euler_factorization.nonzero_terms(),
        reference_euler_symbol_compared: reference_euler_agreement.is_some(),
        reference_euler_agreement_residual_terms: reference_euler_agreement
            .as_ref()
            .map_or(0, ExactPolynomialMatrix::nonzero_terms),
        euler_after_gauge_residual_terms: euler_after_gauge.nonzero_terms(),
        noether_after_euler_residual_terms: noether_after_euler.nonzero_terms(),
        mutation_residual_terms,
        passed,
    }
}

fn compute_report() -> TargetEquationComplexReport {
    let graviton = report_sector(target_sector_complex(TargetSector::Graviton));
    let four_form = report_sector(target_sector_complex(TargetSector::FourForm));
    let rarita_schwinger = report_sector(target_sector_complex(TargetSector::RaritaSchwinger));
    let free = crate::eleven_dimensional_free_complex::build().report;
    let majorana = crate::eleven_dimensional_majorana::verify();
    let susy = crate::eleven_dimensional_linear_susy::verify();
    let generic_formal_compositions_certified =
        graviton.passed && four_form.passed && rarita_schwinger.passed;
    let passed = generic_formal_compositions_certified
        && free.graviton.cohomology.physical_cohomology_dimension == 44
        && free.three_form.cohomology.physical_cohomology_dimension == 84
        && free.gravitino.cohomology.physical_cohomology_dimension == 128
        && majorana.passed
        && susy.passed
        && susy.bosonic_closure_residual_entries == 0
        && susy.fermionic_closure_residual_entries == 0;
    TargetEquationComplexReport {
        schema_version: "adynkra-11d-target-equation-complex-v1",
        role: "exact target-side free curvature, Bianchi, Euler-Lagrange, and Noether complex",
        signature: "Spin(1,10), mostly-plus eta=(-,+,...,+)",
        formal_momentum_variables: VECTOR_DIMENSION,
        graviton,
        four_form,
        rarita_schwinger,
        generic_formal_compositions_certified,
        null_momentum: crate::eleven_dimensional_free_complex::NULL_MOMENTUM,
        null_graviton_physical_dimension: free.graviton.cohomology.physical_cohomology_dimension,
        null_three_form_physical_dimension: free
            .three_form
            .cohomology
            .physical_cohomology_dimension,
        null_rarita_schwinger_physical_dimension: free
            .gravitino
            .cohomology
            .physical_cohomology_dimension,
        null_bosonic_dimension: free.bosonic_physical_dimension,
        null_fermionic_dimension: free.fermionic_physical_dimension,
        majorana_real_form_certified: majorana.majorana_real_form_constructed && majorana.passed,
        light_cone_susy_maps_certified: susy.linearized_susy_maps_constructed && susy.passed,
        light_cone_susy_closure_residual_entries: susy.bosonic_closure_residual_entries
            + susy.fermionic_closure_residual_entries,
        physical_f_adapter_api_available: true,
        physical_source_to_target_f_constructed: false,
        source_to_field_equation_operator_constructed: false,
        off_shell_closure_established: false,
        passed,
        boundary: "This certifies the free on-shell target complex over all eleven formal momenta and its 44+84|128 null fiber. It provides an adapter boundary for a future physical F, but no source-to-curvature map is supplied. It does not establish a superfield equation operator, finite auxiliary fields, interactions, or covariant off-shell closure.",
    }
}

pub fn verify() -> TargetEquationComplexReport {
    static REPORT: OnceLock<TargetEquationComplexReport> = OnceLock::new();
    REPORT.get_or_init(compute_report).clone()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn all_generic_formal_compositions_vanish_and_mutations_are_detected() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        for sector in [
            &report.graviton,
            &report.four_form,
            &report.rarita_schwinger,
        ] {
            assert_eq!(sector.curvature_after_gauge_residual_terms, 0);
            assert_eq!(sector.bianchi_after_curvature_residual_terms, 0);
            assert_eq!(sector.euler_factorization_residual_terms, 0);
            assert_eq!(sector.euler_after_gauge_residual_terms, 0);
            assert_eq!(sector.noether_after_euler_residual_terms, 0);
            assert!(sector.mutation_residual_terms > 0);
        }
    }

    #[test]
    fn null_fiber_and_real_light_cone_maps_match_44_plus_84_given_128() {
        let report = verify();
        assert_eq!(report.null_graviton_physical_dimension, 44);
        assert_eq!(report.null_three_form_physical_dimension, 84);
        assert_eq!(report.null_rarita_schwinger_physical_dimension, 128);
        assert_eq!(report.null_bosonic_dimension, 128);
        assert_eq!(report.null_fermionic_dimension, 128);
        assert!(report.majorana_real_form_certified);
        assert!(report.light_cone_susy_maps_certified);
        assert_eq!(report.light_cone_susy_closure_residual_entries, 0);
    }

    #[test]
    fn adapter_exists_but_source_and_off_shell_claims_fail_closed() {
        let report = verify();
        assert!(report.physical_f_adapter_api_available);
        assert!(!report.physical_source_to_target_f_constructed);
        assert!(!report.source_to_field_equation_operator_constructed);
        assert!(!report.off_shell_closure_established);
    }

    #[test]
    #[ignore = "writes the committed target-equation-complex artifact"]
    fn write_artifact() {
        let report = verify();
        assert!(report.passed);
        let path = "results/adynkra_11d_target_equation_complex.json";
        let temporary = format!("{path}.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        fs::rename(temporary, path).unwrap();
    }
}
