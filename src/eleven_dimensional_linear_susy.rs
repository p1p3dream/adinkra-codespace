//! Exact light-cone realization of the physical 11D supergravity multiplet.
//!
//! The calculation uses null momentum `p=(1,1,0,...,0)` in the mostly-plus
//! Majorana basis.  It fixes light-cone gauge and constructs the physical
//! `SO(9)` graviton 44, three-form 84, and transverse gamma-traceless
//! vector-spinor 128.  All matrices and all closure checks use exact rational
//! arithmetic.
//!
//! This is an on-shell physical-state realization.  It is not an off-shell
//! completion and does not construct the unconstrained 11D scalar-superfield
//! maps studied in the adynkra program.

use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;
use std::collections::BTreeMap;
#[cfg(test)]
use std::fs;
use std::sync::OnceLock;

pub type ExactRational = Ratio<i64>;
type Q = ExactRational;
type RationalMatrix = Vec<Vec<Q>>;
type IntegralMatrix = Vec<Vec<i8>>;

const SPINOR_DIMENSION: usize = 32;
const TRANSVERSE_DIMENSION: usize = 9;
const GRAVITON_DIMENSION: usize = 44;
const THREE_FORM_DIMENSION: usize = 84;
const BOSON_DIMENSION: usize = GRAVITON_DIMENSION + THREE_FORM_DIMENSION;
const FERMION_DIMENSION: usize = 128;
const SUPERCHARGES: usize = 32;

fn z() -> Q {
    Q::from_integer(0)
}

fn qi(value: i64) -> Q {
    Q::from_integer(value)
}

fn identity_i8(dimension: usize) -> IntegralMatrix {
    let mut result = vec![vec![0; dimension]; dimension];
    for index in 0..dimension {
        result[index][index] = 1;
    }
    result
}

fn multiply_i8(left: &IntegralMatrix, right: &IntegralMatrix) -> IntegralMatrix {
    let mut result = vec![vec![0_i8; right[0].len()]; left.len()];
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot] == 0 {
                continue;
            }
            for column in 0..right[0].len() {
                let contribution = i16::from(left[row][pivot]) * i16::from(right[pivot][column]);
                if contribution != 0 {
                    let value = i16::from(result[row][column]) + contribution;
                    result[row][column] = i8::try_from(value).expect("gamma product entry fits i8");
                }
            }
        }
    }
    result
}

fn apply_i8(matrix: &IntegralMatrix, vector: &[Q]) -> Vec<Q> {
    let mut result = vec![z(); matrix.len()];
    for row in 0..matrix.len() {
        for column in 0..vector.len() {
            if matrix[row][column] != 0 && !vector[column].is_zero() {
                result[row] += qi(i64::from(matrix[row][column])) * vector[column].clone();
            }
        }
    }
    result
}

fn gamma_product(gammas: &[IntegralMatrix], axes: &[usize]) -> IntegralMatrix {
    axes.iter()
        .fold(identity_i8(SPINOR_DIMENSION), |product, &axis| {
            multiply_i8(&product, &gammas[axis])
        })
}

/// Returns a row-major nullspace basis together with pivot and free columns.
fn nullspace(matrix: &[Vec<Q>]) -> (RationalMatrix, Vec<usize>, Vec<usize>) {
    let mut reduced = matrix.to_vec();
    let rows = reduced.len();
    let columns = reduced[0].len();
    let mut pivots = Vec::new();
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|&row| !reduced[row][column].is_zero()) else {
            continue;
        };
        reduced.swap(pivot_row, found);
        let scale = reduced[pivot_row][column].clone();
        for entry in &mut reduced[pivot_row] {
            *entry /= scale.clone();
        }
        for row in 0..rows {
            if row == pivot_row || reduced[row][column].is_zero() {
                continue;
            }
            let scale = reduced[row][column].clone();
            for index in 0..columns {
                let subtraction = scale.clone() * reduced[pivot_row][index].clone();
                reduced[row][index] -= subtraction;
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    let free = (0..columns)
        .filter(|column| !pivots.contains(column))
        .collect::<Vec<_>>();
    let mut basis = vec![vec![z(); free.len()]; columns];
    for (basis_column, &free_column) in free.iter().enumerate() {
        basis[free_column][basis_column] = qi(1);
        for (row, &pivot_column) in pivots.iter().enumerate() {
            basis[pivot_column][basis_column] = -reduced[row][free_column].clone();
        }
    }
    (basis, pivots, free)
}

fn matrix_column(matrix: &RationalMatrix, column: usize) -> Vec<Q> {
    matrix.iter().map(|row| row[column].clone()).collect()
}

fn multiply_q(left: &RationalMatrix, right: &RationalMatrix) -> RationalMatrix {
    let mut result = vec![vec![z(); right[0].len()]; left.len()];
    for row in 0..left.len() {
        for pivot in 0..right.len() {
            if left[row][pivot].is_zero() {
                continue;
            }
            for column in 0..right[0].len() {
                if !right[pivot][column].is_zero() {
                    result[row][column] += left[row][pivot].clone() * right[pivot][column].clone();
                }
            }
        }
    }
    result
}

fn integral_as_q(matrix: &IntegralMatrix) -> RationalMatrix {
    matrix
        .iter()
        .map(|row| row.iter().map(|&value| qi(i64::from(value))).collect())
        .collect()
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            output.push(current.clone());
            return;
        }
        for value in start..n {
            current.push(value);
            visit(n, k, value + 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    visit(n, k, 0, &mut Vec::new(), &mut output);
    output
}

#[derive(Clone)]
struct PhysicalBasis {
    slash_kernel: RationalMatrix,
    slash_free_rows: Vec<usize>,
    gamma_trace_kernel: RationalMatrix,
    gamma_trace_free_rows: Vec<usize>,
    fermions: RationalMatrix,
}

fn physical_basis(gammas: &[IntegralMatrix]) -> PhysicalBasis {
    let mut slash = vec![vec![z(); SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for column in 0..SPINOR_DIMENSION {
            slash[row][column] = qi(i64::from(gammas[0][row][column] + gammas[1][row][column]));
        }
    }
    let (slash_kernel, _, slash_free_rows) = nullspace(&slash);
    assert_eq!(slash_free_rows.len(), 16);

    let mut gamma_trace = vec![vec![z(); TRANSVERSE_DIMENSION * 16]; SPINOR_DIMENSION];
    for transverse in 0..TRANSVERSE_DIMENSION {
        let block = multiply_q(&integral_as_q(&gammas[transverse + 2]), &slash_kernel);
        for row in 0..SPINOR_DIMENSION {
            for column in 0..16 {
                gamma_trace[row][transverse * 16 + column] = block[row][column].clone();
            }
        }
    }
    let (gamma_trace_kernel, _, gamma_trace_free_rows) = nullspace(&gamma_trace);
    assert_eq!(gamma_trace_free_rows.len(), FERMION_DIMENSION);

    let mut fermions = vec![vec![z(); FERMION_DIMENSION]; TRANSVERSE_DIMENSION * SPINOR_DIMENSION];
    for transverse in 0..TRANSVERSE_DIMENSION {
        let block_coordinates = gamma_trace_kernel[transverse * 16..(transverse + 1) * 16].to_vec();
        let block = multiply_q(&slash_kernel, &block_coordinates);
        for row in 0..SPINOR_DIMENSION {
            fermions[transverse * SPINOR_DIMENSION + row] = block[row].clone();
        }
    }
    PhysicalBasis {
        slash_kernel,
        slash_free_rows,
        gamma_trace_kernel,
        gamma_trace_free_rows,
        fermions,
    }
}

fn fermion_coordinates(basis: &PhysicalBasis, ambient: &[Q]) -> Option<Vec<Q>> {
    let mut slash_coordinates = Vec::with_capacity(TRANSVERSE_DIMENSION * 16);
    for transverse in 0..TRANSVERSE_DIMENSION {
        for &row in &basis.slash_free_rows {
            slash_coordinates.push(ambient[transverse * SPINOR_DIMENSION + row].clone());
        }
    }
    let coordinates = basis
        .gamma_trace_free_rows
        .iter()
        .map(|&row| slash_coordinates[row].clone())
        .collect::<Vec<_>>();
    for row in 0..ambient.len() {
        let reconstructed = (0..FERMION_DIMENSION).fold(z(), |sum, column| {
            sum + basis.fermions[row][column].clone() * coordinates[column].clone()
        });
        if reconstructed != ambient[row] {
            return None;
        }
    }
    Some(coordinates)
}

#[derive(Clone)]
struct Boson {
    h: Vec<Q>,
    a: Vec<Q>,
}

fn decode_boson(coordinates: &[Q], pairs: &[Vec<usize>], triples: &[Vec<usize>]) -> Boson {
    let mut h = vec![z(); TRANSVERSE_DIMENSION * TRANSVERSE_DIMENSION];
    for (coordinate, pair) in pairs.iter().enumerate() {
        h[pair[0] * TRANSVERSE_DIMENSION + pair[1]] = coordinates[coordinate].clone();
        h[pair[1] * TRANSVERSE_DIMENSION + pair[0]] = coordinates[coordinate].clone();
    }
    let diagonal_offset = pairs.len();
    for index in 0..8 {
        h[index * TRANSVERSE_DIMENSION + index] = coordinates[diagonal_offset + index].clone();
        h[8 * TRANSVERSE_DIMENSION + 8] -= coordinates[diagonal_offset + index].clone();
    }
    let mut a = vec![z(); TRANSVERSE_DIMENSION.pow(3)];
    for (index, triple) in triples.iter().enumerate() {
        a[(triple[0] * TRANSVERSE_DIMENSION + triple[1]) * TRANSVERSE_DIMENSION + triple[2]] =
            coordinates[GRAVITON_DIMENSION + index].clone();
    }
    Boson { h, a }
}

fn permutation_sign(values: &[usize]) -> i64 {
    let inversions = (0..values.len())
        .flat_map(|left| (left + 1..values.len()).map(move |right| (left, right)))
        .filter(|&(left, right)| values[left] > values[right])
        .count();
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn a_get(a: &[Q], axes: [usize; 3]) -> Q {
    if axes.iter().any(|&axis| !(2..11).contains(&axis))
        || axes[0] == axes[1]
        || axes[0] == axes[2]
        || axes[1] == axes[2]
    {
        return z();
    }
    let transverse = axes.map(|axis| axis - 2);
    let mut sorted = transverse;
    sorted.sort_unstable();
    qi(permutation_sign(&transverse))
        * a[(sorted[0] * TRANSVERSE_DIMENSION + sorted[1]) * TRANSVERSE_DIMENSION + sorted[2]]
            .clone()
}

fn f_get(a: &[Q], axes: [usize; 4]) -> Q {
    let mut unique = axes;
    unique.sort_unstable();
    if unique.windows(2).any(|pair| pair[0] == pair[1]) {
        return z();
    }
    let momentum = |axis: usize| if axis < 2 { qi(1) } else { z() };
    let mut value = z();
    for removed in 0..4 {
        let rest = (0..4)
            .filter(|&index| index != removed)
            .map(|index| axes[index])
            .collect::<Vec<_>>();
        value += qi(if removed % 2 == 0 { 1 } else { -1 })
            * momentum(axes[removed])
            * a_get(a, [rest[0], rest[1], rest[2]]);
    }
    value
}

fn h_get(h: &[Q], left: usize, right: usize) -> Q {
    if left < 2 || right < 2 {
        z()
    } else {
        h[(left - 2) * TRANSVERSE_DIMENSION + (right - 2)].clone()
    }
}

fn encode_boson(h: &[Q], a_values: &[Q], pairs: &[Vec<usize>]) -> Option<Vec<Q>> {
    let trace = (0..TRANSVERSE_DIMENSION).fold(z(), |sum, index| {
        sum + h[index * TRANSVERSE_DIMENSION + index].clone()
    });
    if !trace.is_zero() {
        return None;
    }
    let mut coordinates = Vec::with_capacity(BOSON_DIMENSION);
    for pair in pairs {
        coordinates.push(h[pair[0] * TRANSVERSE_DIMENSION + pair[1]].clone());
    }
    for index in 0..8 {
        coordinates.push(h[index * TRANSVERSE_DIMENSION + index].clone());
    }
    coordinates.extend(a_values.iter().cloned());
    assert_eq!(coordinates.len(), BOSON_DIMENSION);
    Some(coordinates)
}

/// Immutable exact sparse linear map stored in deterministic column-major
/// order. Entries within each column are sorted by row.
#[derive(Clone, Debug)]
pub struct ExactSparseMap {
    rows: usize,
    columns: Vec<Vec<(usize, Q)>>,
}

impl ExactSparseMap {
    fn from_columns(rows: usize, columns: Vec<Vec<Q>>) -> Self {
        Self {
            rows,
            columns: columns
                .into_iter()
                .map(|column| {
                    column
                        .into_iter()
                        .enumerate()
                        .filter(|(_, value)| !value.is_zero())
                        .collect()
                })
                .collect(),
        }
    }

    pub fn row_count(&self) -> usize {
        self.rows
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn nonzero_entries(&self) -> usize {
        self.columns.iter().map(Vec::len).sum()
    }

    pub fn entries(&self) -> Vec<(usize, usize, ExactRational)> {
        self.columns
            .iter()
            .enumerate()
            .flat_map(|(column, entries)| {
                entries
                    .iter()
                    .map(move |(row, value)| (*row, column, value.clone()))
            })
            .collect()
    }

    pub fn apply(&self, vector: &[ExactRational]) -> Option<Vec<ExactRational>> {
        if vector.len() != self.column_count() {
            return None;
        }
        let mut output = vec![z(); self.rows];
        for (column, scalar) in vector.iter().enumerate() {
            if scalar.is_zero() {
                continue;
            }
            for (row, value) in &self.columns[column] {
                output[*row] += value.clone() * scalar.clone();
            }
        }
        Some(output)
    }
}

struct TransformationContext {
    gammas: Vec<IntegralMatrix>,
    charge_conjugation: IntegralMatrix,
    physical: PhysicalBasis,
    pairs: Vec<Vec<usize>>,
    triples: Vec<Vec<usize>>,
    quads: Vec<Vec<usize>>,
    transverse_pairs: Vec<Vec<IntegralMatrix>>,
    lorentz_pairs: Vec<(usize, usize, IntegralMatrix)>,
    flux_terms: Vec<Vec<IntegralMatrix>>,
}

impl TransformationContext {
    fn new() -> Self {
        let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
        let charge_conjugation = crate::eleven_dimensional_majorana::real_charge_conjugation();
        let physical = physical_basis(&gammas);
        let pairs = combinations(TRANSVERSE_DIMENSION, 2);
        let triples = combinations(TRANSVERSE_DIMENSION, 3);
        let quads = combinations(11, 4);
        let transverse_pairs = (0..TRANSVERSE_DIMENSION)
            .map(|left| {
                (0..TRANSVERSE_DIMENSION)
                    .map(|right| gamma_product(&gammas, &[left + 2, right + 2]))
                    .collect()
            })
            .collect();
        let lorentz_pairs = combinations(11, 2)
            .into_iter()
            .map(|pair| (pair[0], pair[1], gamma_product(&gammas, &pair)))
            .collect();
        let flux_terms = (0..TRANSVERSE_DIMENSION)
            .map(|transverse| {
                let axis = transverse + 2;
                quads
                    .iter()
                    .map(|quad| {
                        let gamma_quad = gamma_product(&gammas, quad);
                        let mut term = multiply_i8(&gammas[axis], &gamma_quad);
                        if let Some(position) = quad.iter().position(|&entry| entry == axis) {
                            let rest = quad
                                .iter()
                                .copied()
                                .filter(|&entry| entry != axis)
                                .collect::<Vec<_>>();
                            let gamma_rest = gamma_product(&gammas, &rest);
                            let coefficient = if position % 2 == 0 { -3 } else { 3 };
                            for row in 0..SPINOR_DIMENSION {
                                for column in 0..SPINOR_DIMENSION {
                                    term[row][column] += coefficient * gamma_rest[row][column];
                                }
                            }
                        }
                        term
                    })
                    .collect()
            })
            .collect();
        Self {
            gammas,
            charge_conjugation,
            physical,
            pairs,
            triples,
            quads,
            transverse_pairs,
            lorentz_pairs,
            flux_terms,
        }
    }

    fn boson_from_fermion(&self, charge: usize, fermion: &[Q]) -> Option<Vec<Q>> {
        let spinors = (0..TRANSVERSE_DIMENSION)
            .map(|transverse| {
                fermion[transverse * SPINOR_DIMENSION..(transverse + 1) * SPINOR_DIMENSION].to_vec()
            })
            .collect::<Vec<_>>();
        let bar = self.charge_conjugation[charge]
            .iter()
            .map(|&entry| qi(i64::from(entry)))
            .collect::<Vec<_>>();
        let bilinear = |matrix: &IntegralMatrix, spinor: &[Q]| {
            let transformed = apply_i8(matrix, spinor);
            (0..SPINOR_DIMENSION).fold(z(), |sum, index| {
                sum + bar[index].clone() * transformed[index].clone()
            })
        };
        let mut h = vec![z(); TRANSVERSE_DIMENSION * TRANSVERSE_DIMENSION];
        for left in 0..TRANSVERSE_DIMENSION {
            for right in left..TRANSVERSE_DIMENSION {
                let value = bilinear(&self.gammas[left + 2], &spinors[right])
                    + bilinear(&self.gammas[right + 2], &spinors[left]);
                h[left * TRANSVERSE_DIMENSION + right] = value.clone();
                h[right * TRANSVERSE_DIMENSION + left] = value;
            }
        }
        let a_values = self
            .triples
            .iter()
            .map(|triple| {
                let i = triple[0];
                let j = triple[1];
                let k = triple[2];
                -bilinear(&self.transverse_pairs[i][j], &spinors[k])
                    + bilinear(&self.transverse_pairs[i][k], &spinors[j])
                    - bilinear(&self.transverse_pairs[j][k], &spinors[i])
            })
            .collect::<Vec<_>>();
        encode_boson(&h, &a_values, &self.pairs)
    }

    fn fermion_from_boson(&self, charge: usize, coordinates: &[Q]) -> Option<Vec<Q>> {
        let boson = decode_boson(coordinates, &self.pairs, &self.triples);
        let mut epsilon = vec![z(); SPINOR_DIMENSION];
        epsilon[charge] = qi(1);
        let field_strengths = self
            .quads
            .iter()
            .map(|quad| f_get(&boson.a, [quad[0], quad[1], quad[2], quad[3]]))
            .collect::<Vec<_>>();
        let mut ambient = vec![z(); TRANSVERSE_DIMENSION * SPINOR_DIMENSION];
        for transverse in 0..TRANSVERSE_DIMENSION {
            let axis = transverse + 2;
            let mut output = vec![z(); SPINOR_DIMENSION];
            for (left, right, gamma_pair) in &self.lorentz_pairs {
                let derivative = (if *left < 2 { qi(1) } else { z() })
                    * h_get(&boson.h, axis, *right)
                    - (if *right < 2 { qi(1) } else { z() }) * h_get(&boson.h, axis, *left);
                if derivative.is_zero() {
                    continue;
                }
                let spinor = apply_i8(gamma_pair, &epsilon);
                for row in 0..SPINOR_DIMENSION {
                    output[row] -= qi(1) / qi(2) * derivative.clone() * spinor[row].clone();
                }
            }
            for (quad_index, field_strength) in field_strengths.iter().enumerate() {
                if field_strength.is_zero() {
                    continue;
                }
                let spinor = apply_i8(&self.flux_terms[transverse][quad_index], &epsilon);
                for row in 0..SPINOR_DIMENSION {
                    output[row] += qi(1) / qi(6) * field_strength.clone() * spinor[row].clone();
                }
            }
            ambient[transverse * SPINOR_DIMENSION..(transverse + 1) * SPINOR_DIMENSION]
                .clone_from_slice(&output);
        }
        fermion_coordinates(&self.physical, &ambient)
    }

    fn maps(&self, charge: usize) -> Option<(ExactSparseMap, ExactSparseMap)> {
        let bf_columns = (0..FERMION_DIMENSION)
            .map(|column| {
                self.boson_from_fermion(charge, &matrix_column(&self.physical.fermions, column))
            })
            .collect::<Option<Vec<_>>>()?;
        let fb_columns = (0..BOSON_DIMENSION)
            .map(|column| {
                let mut basis = vec![z(); BOSON_DIMENSION];
                basis[column] = qi(1);
                self.fermion_from_boson(charge, &basis)
            })
            .collect::<Option<Vec<_>>>()?;
        Some((
            ExactSparseMap::from_columns(BOSON_DIMENSION, bf_columns),
            ExactSparseMap::from_columns(FERMION_DIMENSION, fb_columns),
        ))
    }
}

/// All 32 exact boson-from-fermion and fermion-from-boson maps at the fixed
/// null momentum. The collection is built once and is immutable thereafter.
#[derive(Clone, Debug)]
pub struct ElevenDimensionalLinearSusyMapSet {
    bf_maps: Vec<ExactSparseMap>,
    fb_maps: Vec<ExactSparseMap>,
    translation_bilinear: IntegralMatrix,
    translation_bilinear_rank: usize,
    slash_kernel_dimension: usize,
    gamma_trace_rank: usize,
}

impl ElevenDimensionalLinearSusyMapSet {
    pub fn supercharge_count(&self) -> usize {
        self.bf_maps.len()
    }

    pub fn boson_dimension(&self) -> usize {
        BOSON_DIMENSION
    }

    pub fn fermion_dimension(&self) -> usize {
        FERMION_DIMENSION
    }

    pub fn bf_map(&self, charge: usize) -> Option<&ExactSparseMap> {
        self.bf_maps.get(charge)
    }

    pub fn fb_map(&self, charge: usize) -> Option<&ExactSparseMap> {
        self.fb_maps.get(charge)
    }

    pub fn translation_bilinear(&self) -> &[Vec<i8>] {
        &self.translation_bilinear
    }

    pub fn translation_bilinear_rank(&self) -> usize {
        self.translation_bilinear_rank
    }
}

fn build_public_map_set() -> ElevenDimensionalLinearSusyMapSet {
    let context = TransformationContext::new();
    let maps = (0..SUPERCHARGES)
        .map(|charge| context.maps(charge))
        .collect::<Option<Vec<_>>>()
        .expect("the exact component transformations preserve the physical spaces");
    let (bf_maps, fb_maps): (Vec<_>, Vec<_>) = maps.into_iter().unzip();
    let mut slash = vec![vec![0_i8; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for column in 0..SPINOR_DIMENSION {
            slash[row][column] = context.gammas[0][row][column] + context.gammas[1][row][column];
        }
    }
    let translation_bilinear = multiply_i8(&context.charge_conjugation, &slash);
    let (_, pivots, _) = nullspace(&integral_as_q(&translation_bilinear));
    ElevenDimensionalLinearSusyMapSet {
        bf_maps,
        fb_maps,
        translation_bilinear,
        translation_bilinear_rank: pivots.len(),
        slash_kernel_dimension: context.physical.slash_kernel[0].len(),
        gamma_trace_rank: TRANSVERSE_DIMENSION * 16 - context.physical.gamma_trace_kernel[0].len(),
    }
}

/// Public immutable exact map collection. This is the executable output of
/// the light-cone construction, not merely its validation report.
pub fn linear_susy_maps() -> &'static ElevenDimensionalLinearSusyMapSet {
    static MAPS: OnceLock<ElevenDimensionalLinearSusyMapSet> = OnceLock::new();
    MAPS.get_or_init(build_public_map_set)
}

fn add_composition(
    result: &mut BTreeMap<usize, Q>,
    outer: &ExactSparseMap,
    inner: &ExactSparseMap,
    column: usize,
) {
    for (pivot, inner_value) in &inner.columns[column] {
        for (row, outer_value) in &outer.columns[*pivot] {
            *result.entry(*row).or_insert_with(z) += outer_value.clone() * inner_value.clone();
        }
    }
}

fn closure_residual(
    outer_left: &ExactSparseMap,
    inner_left: &ExactSparseMap,
    outer_right: &ExactSparseMap,
    inner_right: &ExactSparseMap,
    expected_diagonal: &Q,
) -> (usize, usize) {
    assert_eq!(outer_left.rows, outer_right.rows);
    let dimension = outer_left.rows;
    let mut residuals = 0;
    for column in 0..dimension {
        let mut values = BTreeMap::new();
        add_composition(&mut values, outer_left, inner_left, column);
        add_composition(&mut values, outer_right, inner_right, column);
        for row in 0..dimension {
            let actual = values.remove(&row).unwrap_or_else(z);
            let expected = if row == column {
                expected_diagonal.clone()
            } else {
                z()
            };
            residuals += usize::from(actual != expected);
        }
    }
    (dimension * dimension, residuals)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactClosureAudit {
    pub unordered_charge_pairs_checked: usize,
    pub bosonic_entries_checked: usize,
    pub bosonic_residual_entries: usize,
    pub fermionic_entries_checked: usize,
    pub fermionic_residual_entries: usize,
}

impl ElevenDimensionalLinearSusyMapSet {
    /// Independently verifies the superalgebra directly from the public sparse
    /// maps and the exactly computed translation bilinear.
    pub fn verify_closure(&self) -> ExactClosureAudit {
        let mut audit = ExactClosureAudit {
            unordered_charge_pairs_checked: 0,
            bosonic_entries_checked: 0,
            bosonic_residual_entries: 0,
            fermionic_entries_checked: 0,
            fermionic_residual_entries: 0,
        };
        for left in 0..self.supercharge_count() {
            for right in left..self.supercharge_count() {
                let expected = qi(2 * i64::from(self.translation_bilinear[left][right]));
                let (checked, residual) = closure_residual(
                    &self.bf_maps[left],
                    &self.fb_maps[right],
                    &self.bf_maps[right],
                    &self.fb_maps[left],
                    &expected,
                );
                audit.bosonic_entries_checked += checked;
                audit.bosonic_residual_entries += residual;
                let (checked, residual) = closure_residual(
                    &self.fb_maps[left],
                    &self.bf_maps[right],
                    &self.fb_maps[right],
                    &self.bf_maps[left],
                    &expected,
                );
                audit.fermionic_entries_checked += checked;
                audit.fermionic_residual_entries += residual;
                audit.unordered_charge_pairs_checked += 1;
            }
        }
        audit
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ElevenDimensionalLinearSusyReport {
    pub schema_version: &'static str,
    pub component_transformation_source: &'static str,
    pub representation_source: &'static str,
    pub signature: &'static str,
    pub momentum: [i8; 11],
    pub momentum_norm: i8,
    pub little_group: &'static str,
    pub slash_kernel_dimension: usize,
    pub transverse_vector_spinor_dimension: usize,
    pub gamma_trace_rank: usize,
    pub physical_fermion_dimension: usize,
    pub graviton_dimension: usize,
    pub three_form_dimension: usize,
    pub physical_boson_dimension: usize,
    pub supercharge_count: usize,
    pub bf_maps_constructed: usize,
    pub fb_maps_constructed: usize,
    pub nonzero_bf_maps_at_fixed_momentum: usize,
    pub nonzero_fb_maps_at_fixed_momentum: usize,
    pub translation_bilinear_rank: usize,
    pub translation_bilinear_rank_computed_exactly: bool,
    pub public_sparse_map_api_available: bool,
    pub public_map_api: &'static str,
    pub bf_nonzero_entries: usize,
    pub fb_nonzero_entries: usize,
    pub gravity_coefficient: &'static str,
    pub flux_coefficient: &'static str,
    pub raw_gravity_to_flux_ratio: &'static str,
    pub unordered_charge_pairs_checked: usize,
    pub bosonic_closure_entries_checked: usize,
    pub bosonic_closure_residual_entries: usize,
    pub fermionic_closure_entries_checked: usize,
    pub fermionic_closure_residual_entries: usize,
    pub closure_identity: &'static str,
    pub majorana_real_form_used: bool,
    pub linearized_susy_maps_constructed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn compute_report() -> ElevenDimensionalLinearSusyReport {
    let maps = linear_susy_maps();
    let bf_nonzero_entries = maps
        .bf_maps
        .iter()
        .map(ExactSparseMap::nonzero_entries)
        .sum();
    let fb_nonzero_entries = maps
        .fb_maps
        .iter()
        .map(ExactSparseMap::nonzero_entries)
        .sum();
    let nonzero_bf_maps = maps
        .bf_maps
        .iter()
        .filter(|map| map.nonzero_entries() != 0)
        .count();
    let nonzero_fb_maps = maps
        .fb_maps
        .iter()
        .filter(|map| map.nonzero_entries() != 0)
        .count();
    let closure = maps.verify_closure();

    let passed = closure.unordered_charge_pairs_checked == SUPERCHARGES * (SUPERCHARGES + 1) / 2
        && closure.bosonic_residual_entries == 0
        && closure.fermionic_residual_entries == 0;
    ElevenDimensionalLinearSusyReport {
        schema_version: "adynkra-11d-linear-light-cone-susy-v1",
        component_transformation_source: "linearized limit of arXiv:0903.0259, equations (id6a), (tr3fcurb), and (trfullmajcurb); relative normalization validated by exact closure",
        representation_source: "physical 11D supergraviton little-group content 44+84|128",
        signature: "mostly plus (-,+,+,+,+,+,+,+,+,+,+)",
        momentum: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        momentum_norm: 0,
        little_group: "SO(9)",
        slash_kernel_dimension: maps.slash_kernel_dimension,
        transverse_vector_spinor_dimension: TRANSVERSE_DIMENSION * 16,
        gamma_trace_rank: maps.gamma_trace_rank,
        physical_fermion_dimension: maps.fermion_dimension(),
        graviton_dimension: GRAVITON_DIMENSION,
        three_form_dimension: THREE_FORM_DIMENSION,
        physical_boson_dimension: BOSON_DIMENSION,
        supercharge_count: SUPERCHARGES,
        bf_maps_constructed: maps.bf_maps.len(),
        fb_maps_constructed: maps.fb_maps.len(),
        nonzero_bf_maps_at_fixed_momentum: nonzero_bf_maps,
        nonzero_fb_maps_at_fixed_momentum: nonzero_fb_maps,
        translation_bilinear_rank: maps.translation_bilinear_rank(),
        translation_bilinear_rank_computed_exactly: true,
        public_sparse_map_api_available: true,
        public_map_api: "linear_susy_maps() exposes immutable ExactSparseMap values with deterministic entries() and exact apply()",
        bf_nonzero_entries,
        fb_nonzero_entries,
        gravity_coefficient: "-1/2",
        flux_coefficient: "1/6",
        raw_gravity_to_flux_ratio: "-3:1",
        unordered_charge_pairs_checked: closure.unordered_charge_pairs_checked,
        bosonic_closure_entries_checked: closure.bosonic_entries_checked,
        bosonic_closure_residual_entries: closure.bosonic_residual_entries,
        fermionic_closure_entries_checked: closure.fermionic_entries_checked,
        fermionic_closure_residual_entries: closure.fermionic_residual_entries,
        closure_identity: "BF_q FB_r + BF_r FB_q = FB_q BF_r + FB_r BF_q = 2 (C slash(p))_{qr} I",
        majorana_real_form_used: true,
        linearized_susy_maps_constructed: true,
        passed,
        boundary: "exact on-shell light-cone physical-state maps at one fixed nonzero null momentum. This does not provide an off-shell 11D multiplet, covariant auxiliary fields, or an irreducible scalar-superfield decomposition",
    }
}

pub fn verify() -> ElevenDimensionalLinearSusyReport {
    static REPORT: OnceLock<ElevenDimensionalLinearSusyReport> = OnceLock::new();
    REPORT.get_or_init(compute_report).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_little_group_dimensions_are_exact() {
        let context = TransformationContext::new();
        assert_eq!(context.physical.slash_kernel[0].len(), 16);
        assert_eq!(context.physical.gamma_trace_kernel[0].len(), 128);
        assert_eq!(context.physical.fermions[0].len(), 128);
        assert_eq!(GRAVITON_DIMENSION, 44);
        assert_eq!(THREE_FORM_DIMENSION, 84);
    }

    #[test]
    fn all_supercharges_close_exactly_on_44_plus_84_and_128() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.bf_maps_constructed, 32);
        assert_eq!(report.fb_maps_constructed, 32);
        assert_eq!(report.nonzero_bf_maps_at_fixed_momentum, 16);
        assert_eq!(report.nonzero_fb_maps_at_fixed_momentum, 16);
        assert_eq!(report.translation_bilinear_rank, 16);
        assert_eq!(report.unordered_charge_pairs_checked, 528);
        assert_eq!(report.bosonic_closure_residual_entries, 0);
        assert_eq!(report.fermionic_closure_residual_entries, 0);
    }

    #[test]
    fn public_sparse_maps_reproduce_report_counts_and_are_executable() {
        let maps = linear_susy_maps();
        let report = verify();
        assert_eq!(maps.supercharge_count(), 32);
        assert_eq!(
            maps.translation_bilinear_rank(),
            report.translation_bilinear_rank
        );

        let mut bf_nonzero_entries = 0;
        let mut fb_nonzero_entries = 0;
        for charge in 0..maps.supercharge_count() {
            let bf = maps.bf_map(charge).unwrap();
            let fb = maps.fb_map(charge).unwrap();
            assert_eq!((bf.row_count(), bf.column_count()), (128, 128));
            assert_eq!((fb.row_count(), fb.column_count()), (128, 128));
            bf_nonzero_entries += bf.nonzero_entries();
            fb_nonzero_entries += fb.nonzero_entries();

            let mut fermion = vec![z(); maps.fermion_dimension()];
            fermion[charge % maps.fermion_dimension()] = qi(1);
            assert_eq!(bf.apply(&fermion).unwrap().len(), maps.boson_dimension());
            let mut boson = vec![z(); maps.boson_dimension()];
            boson[charge % maps.boson_dimension()] = qi(1);
            assert_eq!(fb.apply(&boson).unwrap().len(), maps.fermion_dimension());
            assert!(bf.apply(&fermion[..127]).is_none());

            let entries = bf.entries();
            assert!(
                entries
                    .windows(2)
                    .all(|pair| { (pair[0].1, pair[0].0) < (pair[1].1, pair[1].0) })
            );
        }
        assert_eq!(bf_nonzero_entries, report.bf_nonzero_entries);
        assert_eq!(fb_nonzero_entries, report.fb_nonzero_entries);
        let closure = maps.verify_closure();
        assert_eq!(closure.unordered_charge_pairs_checked, 528);
        assert_eq!(closure.bosonic_residual_entries, 0);
        assert_eq!(closure.fermionic_residual_entries, 0);
        assert_eq!(
            closure.bosonic_entries_checked,
            report.bosonic_closure_entries_checked
        );
        assert_eq!(
            closure.fermionic_entries_checked,
            report.fermionic_closure_entries_checked
        );
    }

    #[test]
    fn report_states_the_on_shell_boundary() {
        let report = verify();
        assert!(report.linearized_susy_maps_constructed);
        assert!(report.boundary.contains("on-shell"));
        assert!(report.boundary.contains("does not provide an off-shell"));
    }

    #[test]
    #[ignore = "writes the committed exact verification artifact"]
    fn write_artifact() {
        let report = verify();
        assert!(report.passed);
        let path = "results/adynkra_11d_linear_light_cone_susy.json";
        let temporary = format!("{path}.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        fs::rename(temporary, path).unwrap();
    }
}
