//! Exact momentum-symbol complexes for the free fields of eleven-dimensional
//! supergravity.
//!
//! The sourced free gauge transformations are those of arXiv:0903.0259,
//! Eqs. (1)-(2): a Pauli-Fierz field, an Abelian three-form, and a massless
//! Rarita-Schwinger field.  Antisymmetrization and symmetrization are stored
//! without factorials.  This changes only overall row normalizations.

use num_rational::Ratio;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub const VECTOR_DIMENSION: usize = 11;
pub const SPINOR_DIMENSION: usize = 32;
pub const NULL_MOMENTUM: [i64; VECTOR_DIMENSION] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExactGaussian {
    real: Ratio<i64>,
    imaginary: Ratio<i64>,
}

impl ExactGaussian {
    fn integer(real: i64, imaginary: i64) -> Self {
        Self {
            real: Ratio::from_integer(real),
            imaginary: Ratio::from_integer(imaginary),
        }
    }

    fn is_zero(self) -> bool {
        self.real == Ratio::from_integer(0) && self.imaginary == Ratio::from_integer(0)
    }

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imaginary: self.imaginary + other.imaginary,
        }
    }

    fn negated(self) -> Self {
        Self {
            real: -self.real,
            imaginary: -self.imaginary,
        }
    }

    fn subtract(self, other: Self) -> Self {
        self.add(other.negated())
    }

    fn multiply(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imaginary * other.imaginary,
            imaginary: self.real * other.imaginary + self.imaginary * other.real,
        }
    }

    fn inverse(self) -> Self {
        assert!(!self.is_zero());
        let norm = self.real * self.real + self.imaginary * self.imaginary;
        Self {
            real: self.real / norm,
            imaginary: -self.imaginary / norm,
        }
    }
}

/// An exact sparse matrix over Q(i).  Rows and columns are finite component
/// bases.  Entries are private so normalization cannot be bypassed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseExactMatrix {
    rows: usize,
    columns: usize,
    entries: BTreeMap<(usize, usize), ExactGaussian>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExactCoefficient {
    pub real_numerator: i64,
    pub real_denominator: i64,
    pub imaginary_numerator: i64,
    pub imaginary_denominator: i64,
}

impl SparseExactMatrix {
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

    pub fn nonzero_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn is_zero(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn coefficient(&self, row: usize, column: usize) -> ExactCoefficient {
        assert!(row < self.rows && column < self.columns);
        let value = self
            .entries
            .get(&(row, column))
            .copied()
            .unwrap_or_else(|| ExactGaussian::integer(0, 0));
        ExactCoefficient {
            real_numerator: *value.real.numer(),
            real_denominator: *value.real.denom(),
            imaginary_numerator: *value.imaginary.numer(),
            imaginary_denominator: *value.imaginary.denom(),
        }
    }

    fn add_exact(&mut self, row: usize, column: usize, coefficient: ExactGaussian) {
        assert!(row < self.rows && column < self.columns);
        if coefficient.is_zero() {
            return;
        }
        let key = (row, column);
        let value = self
            .entries
            .get(&key)
            .copied()
            .unwrap_or_else(|| ExactGaussian::integer(0, 0))
            .add(coefficient);
        if value.is_zero() {
            self.entries.remove(&key);
        } else {
            self.entries.insert(key, value);
        }
    }

    fn add_integer(&mut self, row: usize, column: usize, coefficient: i64) {
        self.add_exact(row, column, ExactGaussian::integer(coefficient, 0));
    }

    pub fn multiply(&self, right: &Self) -> Self {
        assert_eq!(self.columns, right.rows);
        let mut right_rows = vec![Vec::new(); right.rows];
        for (&(row, column), &coefficient) in &right.entries {
            right_rows[row].push((column, coefficient));
        }
        let mut result = Self::zero(self.rows, right.columns);
        for (&(row, pivot), &left_coefficient) in &self.entries {
            for &(column, right_coefficient) in &right_rows[pivot] {
                result.add_exact(row, column, left_coefficient.multiply(right_coefficient));
            }
        }
        result
    }

    /// Exact sparse row reduction over Q(i).
    pub fn rank(&self) -> usize {
        let mut source_rows = vec![BTreeMap::new(); self.rows];
        for (&(row, column), &coefficient) in &self.entries {
            source_rows[row].insert(column, coefficient);
        }
        let mut pivots = BTreeMap::<usize, BTreeMap<usize, ExactGaussian>>::new();
        for mut row in source_rows {
            loop {
                let Some((&column, &leading)) = row.first_key_value() else {
                    break;
                };
                if let Some(pivot) = pivots.get(&column) {
                    let terms: Vec<_> = pivot.iter().map(|(&key, &value)| (key, value)).collect();
                    for (key, value) in terms {
                        let updated = row
                            .get(&key)
                            .copied()
                            .unwrap_or_else(|| ExactGaussian::integer(0, 0))
                            .subtract(leading.multiply(value));
                        if updated.is_zero() {
                            row.remove(&key);
                        } else {
                            row.insert(key, updated);
                        }
                    }
                } else {
                    let inverse = leading.inverse();
                    for value in row.values_mut() {
                        *value = value.multiply(inverse);
                    }
                    pivots.insert(column, row);
                    break;
                }
            }
        }
        pivots.len()
    }
}

fn metric_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
}

fn raised_momentum(momentum: [i64; VECTOR_DIMENSION]) -> [i64; VECTOR_DIMENSION] {
    std::array::from_fn(|index| metric_sign(index) * momentum[index])
}

fn momentum_square(momentum: [i64; VECTOR_DIMENSION]) -> i64 {
    let raised = raised_momentum(momentum);
    (0..VECTOR_DIMENSION)
        .map(|index| momentum[index] * raised[index])
        .sum()
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

fn exterior_derivative(momentum: [i64; VECTOR_DIMENSION], degree: usize) -> SparseExactMatrix {
    let source = combinations(degree);
    let target = combinations(degree + 1);
    let target_indices: BTreeMap<_, _> = target
        .iter()
        .enumerate()
        .map(|(index, value)| (value.clone(), index))
        .collect();
    let mut result = SparseExactMatrix::zero(target.len(), source.len());
    for (column, indices) in source.iter().enumerate() {
        for axis in 0..VECTOR_DIMENSION {
            if momentum[axis] == 0 || indices.contains(&axis) {
                continue;
            }
            let mut output = indices.clone();
            output.push(axis);
            output.sort_unstable();
            let position = output.iter().position(|&value| value == axis).unwrap();
            let sign = if position % 2 == 0 { 1 } else { -1 };
            result.add_integer(target_indices[&output], column, sign * momentum[axis]);
        }
    }
    result
}

fn momentum_contraction(momentum: [i64; VECTOR_DIMENSION], degree: usize) -> SparseExactMatrix {
    let source = combinations(degree);
    let target = combinations(degree - 1);
    let target_indices: BTreeMap<_, _> = target
        .iter()
        .enumerate()
        .map(|(index, value)| (value.clone(), index))
        .collect();
    let raised = raised_momentum(momentum);
    let mut result = SparseExactMatrix::zero(target.len(), source.len());
    for (column, indices) in source.iter().enumerate() {
        for (position, &axis) in indices.iter().enumerate() {
            if raised[axis] == 0 {
                continue;
            }
            let mut output = indices.clone();
            output.remove(position);
            let sign = if position % 2 == 0 { 1 } else { -1 };
            result.add_integer(target_indices[&output], column, sign * raised[axis]);
        }
    }
    result
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

fn symmetric_index(indices: &BTreeMap<(usize, usize), usize>, left: usize, right: usize) -> usize {
    indices[&(left.min(right), left.max(right))]
}

fn graviton_gauge(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let fields = symmetric_pairs();
    let mut result = SparseExactMatrix::zero(fields.len(), VECTOR_DIMENSION);
    for (row, &(left, right)) in fields.iter().enumerate() {
        result.add_integer(row, right, momentum[left]);
        result.add_integer(row, left, momentum[right]);
    }
    result
}

fn graviton_curvature(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let symmetric = symmetric_pairs();
    let symmetric_indices: BTreeMap<_, _> = symmetric
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index))
        .collect();
    let pairs = antisymmetric_pairs();
    let mut result = SparseExactMatrix::zero(pairs.len() * pairs.len(), symmetric.len());
    for (left_index, &(a, b)) in pairs.iter().enumerate() {
        for (right_index, &(c, d)) in pairs.iter().enumerate() {
            let row = left_index * pairs.len() + right_index;
            result.add_integer(
                row,
                symmetric_index(&symmetric_indices, b, d),
                momentum[a] * momentum[c],
            );
            result.add_integer(
                row,
                symmetric_index(&symmetric_indices, b, c),
                -momentum[a] * momentum[d],
            );
            result.add_integer(
                row,
                symmetric_index(&symmetric_indices, a, d),
                -momentum[b] * momentum[c],
            );
            result.add_integer(
                row,
                symmetric_index(&symmetric_indices, a, c),
                momentum[b] * momentum[d],
            );
        }
    }
    result
}

fn graviton_bianchi(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let pairs = antisymmetric_pairs();
    let pair_indices: BTreeMap<_, _> = pairs
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index))
        .collect();
    let triples = combinations(3);
    let mut result =
        SparseExactMatrix::zero(triples.len() * pairs.len(), pairs.len() * pairs.len());
    for (triple_index, triple) in triples.iter().enumerate() {
        let (a, b, c) = (triple[0], triple[1], triple[2]);
        for right_index in 0..pairs.len() {
            let row = triple_index * pairs.len() + right_index;
            for (pair, coefficient) in [
                ((b, c), momentum[a]),
                ((a, c), -momentum[b]),
                ((a, b), momentum[c]),
            ] {
                let column = pair_indices[&pair] * pairs.len() + right_index;
                result.add_integer(row, column, coefficient);
            }
        }
    }
    result
}

fn graviton_euler(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let fields = symmetric_pairs();
    let raised = raised_momentum(momentum);
    let square = momentum_square(momentum);
    let mut result = SparseExactMatrix::zero(fields.len(), fields.len());
    for (column, &(m, n)) in fields.iter().enumerate() {
        let h = |left: usize, right: usize| -> i64 {
            i64::from((left == m && right == n) || (left == n && right == m))
        };
        let trace: i64 = (0..VECTOR_DIMENSION)
            .map(|axis| metric_sign(axis) * h(axis, axis))
            .sum();
        let double_divergence: i64 = (0..VECTOR_DIMENSION)
            .flat_map(|left| {
                (0..VECTOR_DIMENSION)
                    .map(move |right| raised[left] * raised[right] * h(left, right))
            })
            .sum();
        for (row, &(a, b)) in fields.iter().enumerate() {
            let first: i64 = (0..VECTOR_DIMENSION)
                .map(|axis| raised[axis] * (momentum[a] * h(b, axis) + momentum[b] * h(a, axis)))
                .sum();
            let eta_ab = if a == b { metric_sign(a) } else { 0 };
            let value = first
                - square * h(a, b)
                - momentum[a] * momentum[b] * trace
                - eta_ab * (double_divergence - square * trace);
            result.add_integer(row, column, value);
        }
    }
    result
}

fn graviton_noether(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let fields = symmetric_pairs();
    let indices: BTreeMap<_, _> = fields
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index))
        .collect();
    let raised = raised_momentum(momentum);
    let mut result = SparseExactMatrix::zero(VECTOR_DIMENSION, fields.len());
    for b in 0..VECTOR_DIMENSION {
        for a in 0..VECTOR_DIMENSION {
            result.add_integer(b, symmetric_index(&indices, a, b), raised[a]);
        }
    }
    result
}

fn converted_lorentz_gammas() -> Vec<Vec<Vec<ExactGaussian>>> {
    crate::eleven_dimensional_clifford::gamma_matrices()
        .into_iter()
        .enumerate()
        .map(|(axis, matrix)| {
            matrix
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| {
                            let converted = ExactGaussian {
                                real: value.re,
                                imaginary: value.im,
                            };
                            if axis == 0 {
                                ExactGaussian::integer(0, 1).multiply(converted)
                            } else {
                                converted
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn dense_gamma_multiply(
    left: &[Vec<ExactGaussian>],
    right: &[Vec<ExactGaussian>],
) -> Vec<Vec<ExactGaussian>> {
    let mut result = vec![vec![ExactGaussian::integer(0, 0); SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            if left[row][pivot].is_zero() {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                if !right[pivot][column].is_zero() {
                    result[row][column] =
                        result[row][column].add(left[row][pivot].multiply(right[pivot][column]));
                }
            }
        }
    }
    result
}

fn gravitino_gauge(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let mut result = SparseExactMatrix::zero(VECTOR_DIMENSION * SPINOR_DIMENSION, SPINOR_DIMENSION);
    for axis in 0..VECTOR_DIMENSION {
        for spinor in 0..SPINOR_DIMENSION {
            result.add_integer(axis * SPINOR_DIMENSION + spinor, spinor, momentum[axis]);
        }
    }
    result
}

fn gravitino_curvature(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let pairs = antisymmetric_pairs();
    let mut result = SparseExactMatrix::zero(
        pairs.len() * SPINOR_DIMENSION,
        VECTOR_DIMENSION * SPINOR_DIMENSION,
    );
    for (pair_index, &(a, b)) in pairs.iter().enumerate() {
        for spinor in 0..SPINOR_DIMENSION {
            let row = pair_index * SPINOR_DIMENSION + spinor;
            result.add_integer(row, b * SPINOR_DIMENSION + spinor, momentum[a]);
            result.add_integer(row, a * SPINOR_DIMENSION + spinor, -momentum[b]);
        }
    }
    result
}

fn gravitino_bianchi(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let exterior = exterior_derivative(momentum, 2);
    let mut result = SparseExactMatrix::zero(
        exterior.rows * SPINOR_DIMENSION,
        exterior.columns * SPINOR_DIMENSION,
    );
    for (&(row, column), &coefficient) in &exterior.entries {
        for spinor in 0..SPINOR_DIMENSION {
            result.add_exact(
                row * SPINOR_DIMENSION + spinor,
                column * SPINOR_DIMENSION + spinor,
                coefficient,
            );
        }
    }
    result
}

fn gravitino_euler(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let gammas = converted_lorentz_gammas();
    let mut triple_products = BTreeMap::new();
    for a in 0..VECTOR_DIMENSION {
        for b in 0..VECTOR_DIMENSION {
            for c in 0..VECTOR_DIMENSION {
                if momentum[b] != 0 && a != b && a != c && b != c {
                    triple_products.entry((a, b, c)).or_insert_with(|| {
                        dense_gamma_multiply(
                            &dense_gamma_multiply(&gammas[a], &gammas[b]),
                            &gammas[c],
                        )
                    });
                }
            }
        }
    }
    let mut result = SparseExactMatrix::zero(
        VECTOR_DIMENSION * SPINOR_DIMENSION,
        VECTOR_DIMENSION * SPINOR_DIMENSION,
    );
    for ((a, b, c), matrix) in triple_products {
        for output_spinor in 0..SPINOR_DIMENSION {
            for input_spinor in 0..SPINOR_DIMENSION {
                result.add_exact(
                    a * SPINOR_DIMENSION + output_spinor,
                    c * SPINOR_DIMENSION + input_spinor,
                    ExactGaussian::integer(momentum[b], 0)
                        .multiply(matrix[output_spinor][input_spinor]),
                );
            }
        }
    }
    result
}

fn gravitino_noether(momentum: [i64; VECTOR_DIMENSION]) -> SparseExactMatrix {
    let mut result = SparseExactMatrix::zero(SPINOR_DIMENSION, VECTOR_DIMENSION * SPINOR_DIMENSION);
    for axis in 0..VECTOR_DIMENSION {
        for spinor in 0..SPINOR_DIMENSION {
            result.add_integer(spinor, axis * SPINOR_DIMENSION + spinor, momentum[axis]);
        }
    }
    result
}

/// Gauge, curvature, Bianchi, Euler-Lagrange, and Noether symbols for one free
/// field sector at one exact momentum covector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorComplex {
    pub name: &'static str,
    pub reducibility: Vec<SparseExactMatrix>,
    pub gauge: SparseExactMatrix,
    pub curvature: SparseExactMatrix,
    pub bianchi: SparseExactMatrix,
    pub euler_lagrange: SparseExactMatrix,
    pub noether: SparseExactMatrix,
}

pub fn graviton_complex(momentum: [i64; VECTOR_DIMENSION]) -> SectorComplex {
    SectorComplex {
        name: "Pauli-Fierz graviton h_ab",
        reducibility: Vec::new(),
        gauge: graviton_gauge(momentum),
        curvature: graviton_curvature(momentum),
        bianchi: graviton_bianchi(momentum),
        euler_lagrange: graviton_euler(momentum),
        noether: graviton_noether(momentum),
    }
}

pub fn three_form_complex(momentum: [i64; VECTOR_DIMENSION]) -> SectorComplex {
    let gauge = exterior_derivative(momentum, 2);
    let curvature = exterior_derivative(momentum, 3);
    let euler_lagrange = momentum_contraction(momentum, 4).multiply(&curvature);
    SectorComplex {
        name: "Abelian three-form A_abc",
        reducibility: vec![
            exterior_derivative(momentum, 0),
            exterior_derivative(momentum, 1),
        ],
        gauge,
        curvature,
        bianchi: exterior_derivative(momentum, 4),
        euler_lagrange,
        noether: momentum_contraction(momentum, 3),
    }
}

pub fn gravitino_complex(momentum: [i64; VECTOR_DIMENSION]) -> SectorComplex {
    SectorComplex {
        name: "massless Rarita-Schwinger gravitino psi_a",
        reducibility: Vec::new(),
        gauge: gravitino_gauge(momentum),
        curvature: gravitino_curvature(momentum),
        bianchi: gravitino_bianchi(momentum),
        euler_lagrange: gravitino_euler(momentum),
        noether: gravitino_noether(momentum),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalCohomologyCensus {
    pub sector: &'static str,
    pub potential_dimension: usize,
    pub gauge_parameter_dimension: usize,
    pub gauge_rank: usize,
    pub euler_lagrange_rank: usize,
    pub on_shell_kernel_dimension: usize,
    pub gauge_image_lies_in_kernel: bool,
    pub physical_cohomology_dimension: usize,
    pub expected_physical_dimension: usize,
    pub matched: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SectorComplexReport {
    pub sector: &'static str,
    pub reducibility_dimensions: Vec<(usize, usize)>,
    pub gauge_dimensions: (usize, usize),
    pub curvature_dimensions: (usize, usize),
    pub bianchi_dimensions: (usize, usize),
    pub euler_lagrange_dimensions: (usize, usize),
    pub noether_dimensions: (usize, usize),
    pub nonzero_entries: [usize; 5],
    pub reducibility_compositions_zero: bool,
    pub curvature_after_gauge_zero: bool,
    pub bianchi_after_curvature_zero: bool,
    pub euler_lagrange_after_gauge_zero: bool,
    pub noether_after_euler_lagrange_zero: bool,
    pub cohomology: PhysicalCohomologyCensus,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ElevenDimensionalFreeComplexReport {
    pub schema_version: &'static str,
    pub spacetime_signature: &'static str,
    pub null_momentum_covector: [i64; VECTOR_DIMENSION],
    pub null_momentum_square: i64,
    pub graviton: SectorComplexReport,
    pub three_form: SectorComplexReport,
    pub gravitino: SectorComplexReport,
    pub bosonic_physical_dimension: usize,
    pub fermionic_physical_dimension: usize,
    pub expected_bosonic_split: [usize; 2],
    pub expected_fermionic_dimension: usize,
    pub exact_symbol_samples_checked: usize,
    pub all_sampled_complex_compositions_zero: bool,
    pub majorana_real_form_constructed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ElevenDimensionalFreeComplexArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub conventions: Vec<&'static str>,
    pub report: ElevenDimensionalFreeComplexReport,
}

fn report_sector(complex: &SectorComplex, expected: usize) -> SectorComplexReport {
    let reducibility_compositions_zero = if complex.reducibility.is_empty() {
        true
    } else {
        complex.reducibility[1]
            .multiply(&complex.reducibility[0])
            .is_zero()
            && complex.gauge.multiply(&complex.reducibility[1]).is_zero()
    };
    let curvature_after_gauge_zero = complex.curvature.multiply(&complex.gauge).is_zero();
    let bianchi_after_curvature_zero = complex.bianchi.multiply(&complex.curvature).is_zero();
    let euler_lagrange_after_gauge_zero = complex.euler_lagrange.multiply(&complex.gauge).is_zero();
    let noether_after_euler_lagrange_zero =
        complex.noether.multiply(&complex.euler_lagrange).is_zero();
    let gauge_rank = complex.gauge.rank();
    let euler_lagrange_rank = complex.euler_lagrange.rank();
    let on_shell_kernel_dimension = complex.euler_lagrange.columns - euler_lagrange_rank;
    let gauge_image_lies_in_kernel = euler_lagrange_after_gauge_zero;
    let physical_cohomology_dimension = on_shell_kernel_dimension - gauge_rank;
    let cohomology = PhysicalCohomologyCensus {
        sector: complex.name,
        potential_dimension: complex.gauge.rows,
        gauge_parameter_dimension: complex.gauge.columns,
        gauge_rank,
        euler_lagrange_rank,
        on_shell_kernel_dimension,
        gauge_image_lies_in_kernel,
        physical_cohomology_dimension,
        expected_physical_dimension: expected,
        matched: gauge_image_lies_in_kernel && physical_cohomology_dimension == expected,
    };
    let passed = reducibility_compositions_zero
        && curvature_after_gauge_zero
        && bianchi_after_curvature_zero
        && euler_lagrange_after_gauge_zero
        && noether_after_euler_lagrange_zero
        && cohomology.matched;
    SectorComplexReport {
        sector: complex.name,
        reducibility_dimensions: complex
            .reducibility
            .iter()
            .map(|matrix| (matrix.rows, matrix.columns))
            .collect(),
        gauge_dimensions: (complex.gauge.rows, complex.gauge.columns),
        curvature_dimensions: (complex.curvature.rows, complex.curvature.columns),
        bianchi_dimensions: (complex.bianchi.rows, complex.bianchi.columns),
        euler_lagrange_dimensions: (complex.euler_lagrange.rows, complex.euler_lagrange.columns),
        noether_dimensions: (complex.noether.rows, complex.noether.columns),
        nonzero_entries: [
            complex.gauge.nonzero_entries(),
            complex.curvature.nonzero_entries(),
            complex.bianchi.nonzero_entries(),
            complex.euler_lagrange.nonzero_entries(),
            complex.noether.nonzero_entries(),
        ],
        reducibility_compositions_zero,
        curvature_after_gauge_zero,
        bianchi_after_curvature_zero,
        euler_lagrange_after_gauge_zero,
        noether_after_euler_lagrange_zero,
        cohomology,
        passed,
    }
}

fn sampled_compositions_zero(momentum: [i64; VECTOR_DIMENSION]) -> bool {
    for complex in [
        graviton_complex(momentum),
        three_form_complex(momentum),
        gravitino_complex(momentum),
    ] {
        if !complex.curvature.multiply(&complex.gauge).is_zero()
            || !complex.bianchi.multiply(&complex.curvature).is_zero()
            || !complex.euler_lagrange.multiply(&complex.gauge).is_zero()
            || !complex.noether.multiply(&complex.euler_lagrange).is_zero()
        {
            return false;
        }
        if !complex.reducibility.is_empty()
            && (!complex.reducibility[1]
                .multiply(&complex.reducibility[0])
                .is_zero()
                || !complex.gauge.multiply(&complex.reducibility[1]).is_zero())
        {
            return false;
        }
    }
    true
}

pub fn build() -> ElevenDimensionalFreeComplexArtifact {
    let graviton = report_sector(&graviton_complex(NULL_MOMENTUM), 44);
    let three_form = report_sector(&three_form_complex(NULL_MOMENTUM), 84);
    let gravitino = report_sector(&gravitino_complex(NULL_MOMENTUM), 128);
    let samples = [
        NULL_MOMENTUM,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, -1, 3, 0, 1, 0, 0, 0, 0, 0, 0],
    ];
    let all_sampled_complex_compositions_zero = samples.into_iter().all(sampled_compositions_zero);
    let bosonic_physical_dimension = graviton.cohomology.physical_cohomology_dimension
        + three_form.cohomology.physical_cohomology_dimension;
    let fermionic_physical_dimension = gravitino.cohomology.physical_cohomology_dimension;
    let passed = graviton.passed
        && three_form.passed
        && gravitino.passed
        && bosonic_physical_dimension == 44 + 84
        && fermionic_physical_dimension == 128
        && all_sampled_complex_compositions_zero;
    ElevenDimensionalFreeComplexArtifact {
        schema_version: "adynkra-11d-free-complex-v1",
        title: "Exact free gauge-curvature-Bianchi and Euler-Lagrange complexes in 11D",
        sources: vec![
            SourceRecord {
                arxiv_id: "0903.0259",
                locator: "Eqs. (1)-(3)",
                role: "free Pauli-Fierz, Abelian three-form, and Rarita-Schwinger actions; gauge transformations and second-stage three-form reducibility",
            },
            SourceRecord {
                arxiv_id: "hep-th/0410239",
                locator: "Sec. 2, light-cone physical-field formulation",
                role: "SO(9) on-shell target carried by the graviton, three-form, and gamma-traceless vector-spinor",
            },
        ],
        conventions: vec![
            "mostly-plus eta=(-,+,...,+), covariant momentum components supplied to every builder",
            "null representative p_a=(1,1,0,...,0), hence p^a=(-1,1,0,...,0)",
            "unnormalized symmetrization and antisymmetrization; factorial changes are invertible row rescalings",
            "R_ab|cd=p_a p_c h_bd-p_a p_d h_bc-p_b p_c h_ad+p_b p_d h_ac",
            "F_4=p wedge A_3 and E_A=i_(p sharp) F_4",
            "E_psi^a=Gamma^(abc) p_b psi_c, with Gamma^0=i gamma_E^0 and Gamma^i=gamma_E^i",
        ],
        report: ElevenDimensionalFreeComplexReport {
            schema_version: "adynkra-11d-free-complex-report-v1",
            spacetime_signature: "mostly plus (-,+,+,+,+,+,+,+,+,+,+)",
            null_momentum_covector: NULL_MOMENTUM,
            null_momentum_square: momentum_square(NULL_MOMENTUM),
            graviton,
            three_form,
            gravitino,
            bosonic_physical_dimension,
            fermionic_physical_dimension,
            expected_bosonic_split: [44, 84],
            expected_fermionic_dimension: 128,
            exact_symbol_samples_checked: samples.len(),
            all_sampled_complex_compositions_zero,
            majorana_real_form_constructed: false,
            passed,
            boundary: "The bosonic complexes are exact real rational symbols. The Rarita-Schwinger rank is exact over Q(i) after Lorentzizing the repository's Euclidean B5 gamma basis and matches the complexification of the 128-state Majorana module, but this module does not construct the compatible Majorana conjugation or a supersymmetry map between the three sectors. Interactions and nonlinear equations are outside scope.",
        },
    }
}

pub fn write_artifacts(
    data_path: &Path,
    validation_path: &Path,
) -> ElevenDimensionalFreeComplexReport {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create 11D free-complex data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create 11D free-complex validation directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create 11D free-complex artifact")),
        &artifact,
    )
    .expect("write 11D free-complex artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create 11D free-complex report")),
        &artifact.report,
    )
    .expect("write 11D free-complex report");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sparse_rank_and_cancellation_work_over_q_i() {
        let mut matrix = SparseExactMatrix::zero(3, 3);
        matrix.add_exact(0, 0, ExactGaussian::integer(0, 1));
        matrix.add_integer(0, 1, 1);
        matrix.add_exact(1, 0, ExactGaussian::integer(1, 0));
        matrix.add_exact(1, 1, ExactGaussian::integer(0, -1));
        matrix.add_integer(2, 2, 2);
        assert_eq!(matrix.rank(), 2);
        let mut right = SparseExactMatrix::zero(3, 1);
        right.add_integer(0, 0, 1);
        right.add_exact(1, 0, ExactGaussian::integer(0, -1));
        assert!(matrix.multiply(&right).is_zero());
    }

    #[test]
    fn lorentzized_gamma_basis_has_mostly_plus_clifford_signature() {
        let gammas = converted_lorentz_gammas();
        for left in 0..VECTOR_DIMENSION {
            for right in 0..VECTOR_DIMENSION {
                let lr = dense_gamma_multiply(&gammas[left], &gammas[right]);
                let rl = dense_gamma_multiply(&gammas[right], &gammas[left]);
                for row in 0..SPINOR_DIMENSION {
                    for column in 0..SPINOR_DIMENSION {
                        let actual = lr[row][column].add(rl[row][column]);
                        let expected = if left == right && row == column {
                            ExactGaussian::integer(2 * metric_sign(left), 0)
                        } else {
                            ExactGaussian::integer(0, 0)
                        };
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn three_form_has_both_reducibility_stages_and_d_squared_zero() {
        let complex = three_form_complex([2, -1, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(complex.reducibility.len(), 2);
        assert_eq!(
            (
                complex.reducibility[0].rows(),
                complex.reducibility[0].columns()
            ),
            (11, 1)
        );
        assert_eq!(
            (
                complex.reducibility[1].rows(),
                complex.reducibility[1].columns()
            ),
            (55, 11)
        );
        assert!(
            complex.reducibility[1]
                .multiply(&complex.reducibility[0])
                .is_zero()
        );
        assert!(complex.gauge.multiply(&complex.reducibility[1]).is_zero());
        assert!(complex.curvature.multiply(&complex.gauge).is_zero());
        assert!(complex.bianchi.multiply(&complex.curvature).is_zero());
    }

    #[test]
    fn every_free_sector_is_an_exact_gauge_and_euler_complex() {
        for momentum in [
            NULL_MOMENTUM,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, -1, 3, 0, 1, 0, 0, 0, 0, 0, 0],
        ] {
            assert!(sampled_compositions_zero(momentum));
        }
    }

    #[test]
    fn null_fiber_recovers_44_plus_84_bosons_and_128_fermions() {
        let artifact = build();
        assert!(artifact.report.passed);
        assert_eq!(artifact.report.null_momentum_square, 0);
        assert_eq!(artifact.report.graviton.cohomology.gauge_rank, 11);
        assert_eq!(artifact.report.graviton.cohomology.euler_lagrange_rank, 11);
        assert_eq!(
            artifact
                .report
                .graviton
                .cohomology
                .physical_cohomology_dimension,
            44
        );
        assert_eq!(artifact.report.three_form.cohomology.gauge_rank, 45);
        assert_eq!(
            artifact.report.three_form.cohomology.euler_lagrange_rank,
            36
        );
        assert_eq!(
            artifact
                .report
                .three_form
                .cohomology
                .physical_cohomology_dimension,
            84
        );
        assert_eq!(artifact.report.gravitino.cohomology.gauge_rank, 32);
        assert_eq!(
            artifact.report.gravitino.cohomology.euler_lagrange_rank,
            192
        );
        assert_eq!(
            artifact
                .report
                .gravitino
                .cohomology
                .physical_cohomology_dimension,
            128
        );
        assert_eq!(artifact.report.bosonic_physical_dimension, 128);
        assert_eq!(artifact.report.fermionic_physical_dimension, 128);
        assert!(!artifact.report.majorana_real_form_constructed);
    }

    #[test]
    fn nonnull_fibers_have_no_physical_cohomology() {
        let timelike = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for complex in [
            graviton_complex(timelike),
            three_form_complex(timelike),
            gravitino_complex(timelike),
        ] {
            assert!(complex.euler_lagrange.multiply(&complex.gauge).is_zero());
            assert_eq!(
                complex.euler_lagrange.columns() - complex.euler_lagrange.rank(),
                complex.gauge.rank()
            );
        }
    }

    #[test]
    fn curvature_and_euler_sign_mutations_are_detected() {
        let mut graviton = graviton_complex(NULL_MOMENTUM);
        let key = *graviton.curvature.entries.keys().next().unwrap();
        graviton.curvature.entries.get_mut(&key).unwrap().real += Ratio::from_integer(1);
        assert!(!graviton.curvature.multiply(&graviton.gauge).is_zero());

        let mut three_form = three_form_complex(NULL_MOMENTUM);
        let key = *three_form.euler_lagrange.entries.keys().next().unwrap();
        three_form
            .euler_lagrange
            .entries
            .get_mut(&key)
            .unwrap()
            .real += Ratio::from_integer(1);
        assert!(
            !three_form
                .euler_lagrange
                .multiply(&three_form.gauge)
                .is_zero()
        );

        let mut gravitino = gravitino_complex(NULL_MOMENTUM);
        let key = *gravitino
            .euler_lagrange
            .entries
            .keys()
            .find(|(_, column)| column / SPINOR_DIMENSION < 2)
            .unwrap();
        gravitino.euler_lagrange.entries.get_mut(&key).unwrap().real += Ratio::from_integer(1);
        assert!(
            !gravitino
                .euler_lagrange
                .multiply(&gravitino.gauge)
                .is_zero()
        );
    }
}
