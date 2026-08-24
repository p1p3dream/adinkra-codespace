//! Convention-fixed executable slice of the linearized 11D geometry.
//!
//! The implemented chain is the part fixed without choosing a conjectural
//! spinor-prepotential gauge law:
//!
//! ```text
//! D_alpha H_beta^c
//!   -> Eq. (25) bosonic frame perturbation
//!   -> Eq. (29) bosonic anholonomy
//!   -> Eq. (39) gamma-rank 2 and 5 contractions
//!   -> Eq. (40) exact conventional-compensator quotient
//!   -> the dimension-zero X_[2] and X_[5] curvature sectors.
//! ```
//!
//! Source equations are hep-th/0101037 Eqs. (25), (29), (39), (40),
//! and (44).  The target projector convention agrees with
//! arXiv:2007.05097 Eqs. (2.2)-(2.6).  Antisymmetrization has unit weight,
//! the Lorentz metric is diag(-,+,...,+), epsilon_(0...10)=+1, and raised
//! spinor gamma matrices are `Gamma_[p] C^{-1}` in the real Majorana basis.
//!
//! The sources do not fix a physical `Psi_alpha -> H_hat` operator or the
//! coefficients of the six proposed gauge channels.  This module therefore
//! does not claim a complete K/F complex or covariant off-shell closure.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const VECTOR_DIMENSION: usize = 11;
pub const SPINOR_DIMENSION: usize = 32;
pub const DH_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION;
pub const FRAME_DIMENSION: usize = VECTOR_DIMENSION * VECTOR_DIMENSION;
pub const TWO_FORM_VECTOR_DIMENSION: usize = 55 * VECTOR_DIMENSION;
pub const FIVE_FORM_VECTOR_DIMENSION: usize = 462 * VECTOR_DIMENSION;
pub const DDH_DIMENSION: usize =
    SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION;
pub const PH_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION;
pub const H_SECOND_JET_DIMENSION: usize = DDH_DIMENSION + PH_DIMENSION;
pub const C_ALPHA_VECTOR_VECTOR_DIMENSION: usize =
    SPINOR_DIMENSION * VECTOR_DIMENSION * VECTOR_DIMENSION;
pub const SPINORIAL_CONNECTION_DIMENSION: usize = SPINOR_DIMENSION * 55;
pub const D_SPINORIAL_CONNECTION_DIMENSION: usize =
    SPINOR_DIMENSION * SPINORIAL_CONNECTION_DIMENSION;
pub const BOSONIC_CONNECTION_DIMENSION: usize = VECTOR_DIMENSION * 55;
pub const T_ALPHA_VECTOR_SPINOR_DIMENSION: usize =
    SPINOR_DIMENSION * VECTOR_DIMENSION * SPINOR_DIMENSION;
pub const D_J_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION;
pub const W_FOUR_FORM_DIMENSION: usize = 330;
pub const SPINOR_ANHOLONOMY_DIMENSION: usize =
    SPINOR_DIMENSION * SPINOR_DIMENSION * SPINOR_DIMENSION;
pub const DELTA_DIMENSION: usize = SPINOR_DIMENSION * SPINOR_DIMENSION;
pub const D_DELTA_DIMENSION: usize = SPINOR_DIMENSION * DELTA_DIMENSION;
pub const DD_DELTA_DIMENSION: usize = SPINOR_DIMENSION * D_DELTA_DIMENSION;
pub const P_DELTA_DIMENSION: usize = VECTOR_DIMENSION * DELTA_DIMENSION;

pub const HEP_TH_0101037_SOURCE_SHA256: &str =
    "9405ca44a0036567cf86bfbc89de097d8b064612c314b28f31d614e4553a4453";
pub const ARXIV_2007_05097_SOURCE_SHA256: &str =
    "3a6e81c2c677cf3b68455615145510a4d8bce7db967c77c4afd3b85423535df7";

type Rational = Ratio<i64>;

fn r(value: i64) -> Rational {
    Ratio::from_integer(value)
}

fn rr(numerator: i64, denominator: i64) -> Rational {
    Ratio::new(numerator, denominator)
}

/// An exact element of Q(i).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactQi {
    pub real: Rational,
    pub imaginary: Rational,
}

impl ExactQi {
    pub fn zero() -> Self {
        Self::from_integer(0)
    }

    pub fn one() -> Self {
        Self::from_integer(1)
    }

    pub fn i() -> Self {
        Self {
            real: r(0),
            imaginary: r(1),
        }
    }

    pub fn from_integer(value: i64) -> Self {
        Self {
            real: r(value),
            imaginary: r(0),
        }
    }

    pub fn from_rational(numerator: i64, denominator: i64) -> Self {
        Self {
            real: rr(numerator, denominator),
            imaginary: r(0),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.real == r(0) && self.imaginary == r(0)
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.real += other.real.clone();
        self.imaginary += other.imaginary.clone();
    }

    pub fn scaled(&self, factor: &Rational) -> Self {
        Self {
            real: self.real.clone() * factor.clone(),
            imaginary: self.imaginary.clone() * factor.clone(),
        }
    }

    pub fn times_i(&self) -> Self {
        Self {
            real: -self.imaginary.clone(),
            imaginary: self.real.clone(),
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

fn add_sparse(target: &mut BTreeMap<usize, ExactQi>, index: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = target.entry(index).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        target.remove(&index);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseQiEntry {
    pub row: usize,
    pub coefficient: ExactQi,
}

/// Exact sparse linear operator, stored by input column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseQiOperator {
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub columns: Vec<Vec<SparseQiEntry>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Eq26CliffordBlock {
    pub gamma_rank: usize,
    pub output_lower_spinor_gamma: Vec<Vec<i16>>,
    pub input_raised_spinor_gamma: Vec<Vec<i16>>,
    pub coefficient: ExactQi,
}

/// Factored exact form of hep-th/0101037 Eq. (26).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Eq26FactoredOperator {
    pub dimension: usize,
    pub blocks: Vec<Eq26CliffordBlock>,
}

impl Eq26FactoredOperator {
    /// Input is `D_gamma Delta_delta{}^epsilon + delta_delta{}^epsilon D_gamma Psi`.
    pub fn apply(&self, input: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
        let mut output = BTreeMap::new();
        for (&index, value) in input {
            assert!(index < self.dimension);
            let epsilon = index % SPINOR_DIMENSION;
            let pair = index / SPINOR_DIMENSION;
            let delta = pair % SPINOR_DIMENSION;
            let gamma = pair / SPINOR_DIMENSION;
            for block in &self.blocks {
                let contraction = block.input_raised_spinor_gamma[gamma][delta];
                if contraction == 0 {
                    continue;
                }
                let input_factor = value
                    .multiply(&block.coefficient)
                    .scaled(&r(i64::from(contraction)));
                for alpha in 0..SPINOR_DIMENSION {
                    for beta in 0..SPINOR_DIMENSION {
                        let output_gamma = block.output_lower_spinor_gamma[alpha][beta];
                        if output_gamma != 0 {
                            let row =
                                (alpha * SPINOR_DIMENSION + beta) * SPINOR_DIMENSION + epsilon;
                            add_sparse(
                                &mut output,
                                row,
                                input_factor.scaled(&r(i64::from(output_gamma))),
                            );
                        }
                    }
                }
            }
        }
        output
    }
}

impl SparseQiOperator {
    pub fn apply_sparse(&self, input: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
        let mut output = BTreeMap::new();
        for (&column, value) in input {
            assert!(column < self.input_dimension);
            for entry in &self.columns[column] {
                add_sparse(&mut output, entry.row, entry.coefficient.multiply(value));
            }
        }
        output
    }

    pub fn nonzero_entries(&self) -> usize {
        self.columns.iter().map(Vec::len).sum()
    }

    pub fn scaled(&self, factor: ExactQi) -> Self {
        let mut result = self.clone();
        for column in &mut result.columns {
            for entry in column {
                entry.coefficient = entry.coefficient.multiply(&factor);
            }
        }
        result
    }
}

fn lorentz_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
}

fn masks_of_degree(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn form_vector_basis(degree: usize) -> Vec<(u16, usize)> {
    masks_of_degree(degree)
        .into_iter()
        .flat_map(|mask| (0..VECTOR_DIMENSION).map(move |vector| (mask, vector)))
        .collect()
}

fn dh_index(derivative_spinor: usize, h_spinor: usize, vector: usize) -> usize {
    assert!(derivative_spinor < SPINOR_DIMENSION);
    assert!(h_spinor < SPINOR_DIMENSION);
    assert!(vector < VECTOR_DIMENSION);
    (derivative_spinor * SPINOR_DIMENSION + h_spinor) * VECTOR_DIMENSION + vector
}

fn frame_index(frame_vector: usize, output_vector: usize) -> usize {
    frame_vector * VECTOR_DIMENSION + output_vector
}

fn ddh_index(
    first_derivative: usize,
    second_derivative: usize,
    h_spinor: usize,
    output_vector: usize,
) -> usize {
    (((first_derivative * SPINOR_DIMENSION + second_derivative) * SPINOR_DIMENSION + h_spinor)
        * VECTOR_DIMENSION)
        + output_vector
}

fn ph_index(momentum: usize, h_spinor: usize, output_vector: usize) -> usize {
    (momentum * SPINOR_DIMENSION + h_spinor) * VECTOR_DIMENSION + output_vector
}

fn c_alpha_b_c_index(alpha: usize, input_vector: usize, output_vector: usize) -> usize {
    (alpha * VECTOR_DIMENSION + input_vector) * VECTOR_DIMENSION + output_vector
}

fn t_alpha_e_gamma_index(alpha: usize, vector: usize, gamma: usize) -> usize {
    (alpha * VECTOR_DIMENSION + vector) * SPINOR_DIMENSION + gamma
}

fn d_j_index(derivative: usize, spinor: usize) -> usize {
    derivative * SPINOR_DIMENSION + spinor
}

fn spinorial_connection_index(spinor: usize, pair: usize) -> usize {
    spinor * 55 + pair
}

fn d_spinorial_connection_index(
    derivative_spinor: usize,
    connection_spinor: usize,
    pair: usize,
) -> usize {
    derivative_spinor * SPINORIAL_CONNECTION_DIMENSION
        + spinorial_connection_index(connection_spinor, pair)
}

fn bosonic_connection_index(vector: usize, pair: usize) -> usize {
    vector * 55 + pair
}

fn multiply_i8(left: &[Vec<i8>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut result = vec![vec![0_i16; right[0].len()]; left.len()];
    for row in 0..left.len() {
        for middle in 0..right.len() {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..right[0].len() {
                result[row][column] +=
                    i16::from(left[row][middle]) * i16::from(right[middle][column]);
            }
        }
    }
    result
}

fn real_gammas() -> &'static Vec<Vec<Vec<i8>>> {
    static GAMMAS: OnceLock<Vec<Vec<Vec<i8>>>> = OnceLock::new();
    GAMMAS.get_or_init(crate::eleven_dimensional_majorana::real_gamma_matrices)
}

fn real_charge() -> &'static Vec<Vec<i8>> {
    static CHARGE: OnceLock<Vec<Vec<i8>>> = OnceLock::new();
    CHARGE.get_or_init(crate::eleven_dimensional_majorana::real_charge_conjugation)
}

fn multiply_i16_i8(left: &[Vec<i16>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut result = vec![vec![0_i16; right[0].len()]; left.len()];
    for row in 0..left.len() {
        for middle in 0..right.len() {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..right[0].len() {
                result[row][column] += left[row][middle] * i16::from(right[middle][column]);
            }
        }
    }
    result
}

fn multiply_i8_i16(left: &[Vec<i8>], right: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let mut result = vec![vec![0_i16; right[0].len()]; left.len()];
    for row in 0..left.len() {
        for middle in 0..right.len() {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..right[0].len() {
                result[row][column] += i16::from(left[row][middle]) * right[middle][column];
            }
        }
    }
    result
}

fn gamma_product(indices: &[usize], lower_indices: bool) -> Vec<Vec<i16>> {
    let gammas = real_gammas();
    let mut result = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = 1;
    }
    for &axis in indices {
        result = multiply_i16_i8(&result, &gammas[axis]);
        if lower_indices && axis == 0 {
            for row in &mut result {
                for value in row {
                    *value = -*value;
                }
            }
        }
    }
    result
}

fn raised_gamma(indices: &[usize]) -> Vec<Vec<i16>> {
    let product = gamma_product(indices, true);
    let charge = real_charge();
    // In this primitive Majorana normalization C^2=-I, hence C^-1=-C.
    let mut charge_inverse = charge.clone();
    for row in &mut charge_inverse {
        for value in row {
            *value = -*value;
        }
    }
    // The lower bilinear is L=C Gamma.  Raising both spinor indices gives
    // C^{-1} L (C^{-1})^T = -Gamma C^{-1} because C is antisymmetric.
    let mut raised = multiply_i16_i8(&product, &charge_inverse);
    for row in &mut raised {
        for value in row {
            *value = -*value;
        }
    }
    raised
}

fn lower_spinor_gamma(indices: &[usize], lower_vector_indices: bool) -> Vec<Vec<i16>> {
    let charge = real_charge();
    let product = gamma_product(indices, lower_vector_indices);
    let product_i8 = product
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| i8::try_from(*value).unwrap())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    multiply_i8(charge, &product_i8)
}

fn raised_one_gammas() -> &'static Vec<Vec<Vec<i16>>> {
    static RAISED_ONE: OnceLock<Vec<Vec<Vec<i16>>>> = OnceLock::new();
    RAISED_ONE.get_or_init(|| {
        (0..VECTOR_DIMENSION)
            .map(|axis| raised_gamma(&[axis]))
            .collect()
    })
}

/// Spinor-index placements printed in the third term of hep-th/0101037
/// Eq. (28): `(gamma_b)^{beta delta}` and
/// `(gamma^c)_{epsilon alpha}`.  These bilinears cannot be replaced by the
/// mixed-index Clifford matrices `Gamma_b` and `Gamma^c`.
fn eq28_raised_lower_one_gammas() -> &'static Vec<Vec<Vec<i16>>> {
    static RAISED_LOWER_ONE: OnceLock<Vec<Vec<Vec<i16>>>> = OnceLock::new();
    RAISED_LOWER_ONE.get_or_init(|| {
        (0..VECTOR_DIMENSION)
            .map(|axis| lower_spinor_gamma(&[axis], true))
            .collect()
    })
}

fn eq28_lowered_upper_one_gammas() -> &'static Vec<Vec<Vec<i16>>> {
    static LOWERED_UPPER_ONE: OnceLock<Vec<Vec<Vec<i16>>>> = OnceLock::new();
    LOWERED_UPPER_ONE.get_or_init(|| {
        let charge = real_charge();
        (0..VECTOR_DIMENSION)
            .map(|axis| multiply_i16_i8(&gamma_product(&[axis], false), charge))
            .collect()
    })
}

fn build_eq26_spinor_anholonomy_operator() -> Eq26FactoredOperator {
    let mut blocks = Vec::with_capacity(55 + 462);
    for degree in [2, 5] {
        for mask in masks_of_degree(degree) {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            blocks.push(Eq26CliffordBlock {
                gamma_rank: degree,
                output_lower_spinor_gamma: lower_spinor_gamma(&indices, false),
                input_raised_spinor_gamma: raised_gamma(&indices),
                // Eq. (26) uses ordered repeated antisymmetric indices.  This
                // module stores only increasing masks, so the independent
                // rank-p coefficient includes p!.
                coefficient: if degree == 2 {
                    ExactQi::from_rational(1, 32)
                } else {
                    ExactQi::from_rational(-1, 32)
                },
            });
        }
    }
    Eq26FactoredOperator {
        dimension: SPINOR_ANHOLONOMY_DIMENSION,
        blocks,
    }
}

fn cached_eq26_spinor_anholonomy_operator() -> &'static Eq26FactoredOperator {
    static OPERATOR: OnceLock<Eq26FactoredOperator> = OnceLock::new();
    OPERATOR.get_or_init(build_eq26_spinor_anholonomy_operator)
}

pub fn eq26_spinor_anholonomy_operator() -> Eq26FactoredOperator {
    cached_eq26_spinor_anholonomy_operator().clone()
}

fn insertion_sign(mask: u16, index: usize) -> i64 {
    if (mask >> (index + 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn total_antisymmetric_part(
    degree: usize,
    input: &BTreeMap<(u16, usize), ExactQi>,
) -> BTreeMap<u16, ExactQi> {
    let mut output = BTreeMap::new();
    for (&(mask, vector), value) in input {
        if mask & (1_u16 << vector) != 0 {
            continue;
        }
        let factor = rr(
            insertion_sign(mask, vector) * lorentz_sign(vector),
            (degree + 1) as i64,
        );
        let key = mask | (1_u16 << vector);
        let entry = output.entry(key).or_insert_with(ExactQi::zero);
        entry.add_assign(&value.scaled(&factor));
        if entry.is_zero() {
            output.remove(&key);
        }
    }
    output
}

fn inject_total_antisymmetric(
    degree: usize,
    form: &BTreeMap<u16, ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let mut output = BTreeMap::new();
    for (&mask, value) in form {
        assert_eq!(mask.count_ones() as usize, degree + 1);
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) == 0 {
                continue;
            }
            let remaining = mask ^ (1_u16 << vector);
            let factor = r(insertion_sign(remaining, vector) * lorentz_sign(vector));
            output.insert((remaining, vector), value.scaled(&factor));
        }
    }
    output
}

fn mixed_trace(degree: usize, input: &BTreeMap<(u16, usize), ExactQi>) -> BTreeMap<u16, ExactQi> {
    let mut output = BTreeMap::new();
    for (&(mask, vector), value) in input {
        if mask & (1_u16 << vector) == 0 {
            continue;
        }
        let remaining = mask ^ (1_u16 << vector);
        let factor = r(insertion_sign(remaining, vector));
        let entry = output.entry(remaining).or_insert_with(ExactQi::zero);
        entry.add_assign(&value.scaled(&factor));
        if entry.is_zero() {
            output.remove(&remaining);
        }
    }
    assert!(
        output
            .keys()
            .all(|mask| mask.count_ones() as usize + 1 == degree)
    );
    output
}

fn inject_mixed_trace(
    degree: usize,
    form: &BTreeMap<u16, ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let sign = if (degree - 1) % 2 == 0 { 1 } else { -1 };
    let eigenvalue = sign * (VECTOR_DIMENSION - degree + 1) as i64;
    let mut output = BTreeMap::new();
    for (&mask, value) in form {
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) != 0 {
                continue;
            }
            let output_mask = mask | (1_u16 << vector);
            let less = (mask & ((1_u16 << vector) - 1)).count_ones();
            let insertion = if less % 2 == 0 { 1 } else { -1 };
            output.insert(
                (output_mask, vector),
                value.scaled(&rr(insertion, eigenvalue)),
            );
        }
    }
    output
}

fn delta_wedge(
    output_degree: usize,
    form: &BTreeMap<u16, ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let mut output = BTreeMap::new();
    for (&mask, value) in form {
        assert_eq!(mask.count_ones() as usize, output_degree - 1);
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) != 0 {
                continue;
            }
            let output_mask = mask | (1_u16 << vector);
            let less = (mask & ((1_u16 << vector) - 1)).count_ones();
            let sign = if less % 2 == 0 { 1 } else { -1 };
            output.insert(
                (output_mask, vector),
                value.scaled(&rr(sign, output_degree as i64)),
            );
        }
    }
    output
}

fn subtract_tensor(
    left: &mut BTreeMap<(u16, usize), ExactQi>,
    right: &BTreeMap<(u16, usize), ExactQi>,
) {
    for (&key, value) in right {
        let entry = left.entry(key).or_insert_with(ExactQi::zero);
        entry.add_assign(&value.scaled(&r(-1)));
        if entry.is_zero() {
            left.remove(&key);
        }
    }
}

fn hook_projection(
    degree: usize,
    input: &BTreeMap<(u16, usize), ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let mut output = input.clone();
    subtract_tensor(
        &mut output,
        &inject_total_antisymmetric(degree, &total_antisymmetric_part(degree, input)),
    );
    subtract_tensor(
        &mut output,
        &inject_mixed_trace(degree, &mixed_trace(degree, input)),
    );
    output
}

fn tensor_to_sparse(
    basis_lookup: &BTreeMap<(u16, usize), usize>,
    tensor: BTreeMap<(u16, usize), ExactQi>,
) -> Vec<SparseQiEntry> {
    tensor
        .into_iter()
        .map(|(key, coefficient)| SparseQiEntry {
            row: basis_lookup[&key],
            coefficient,
        })
        .collect()
}

/// Eq. (40) projection to the trace-free, non-exterior hook.
pub fn hook_projector_operator(degree: usize) -> SparseQiOperator {
    assert!(degree == 2 || degree == 5);
    let basis = form_vector_basis(degree);
    let lookup = basis
        .iter()
        .copied()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    let columns = basis
        .iter()
        .map(|&key| {
            let mut unit = BTreeMap::new();
            unit.insert(key, ExactQi::one());
            tensor_to_sparse(&lookup, hook_projection(degree, &unit))
        })
        .collect();
    SparseQiOperator {
        input_dimension: basis.len(),
        output_dimension: basis.len(),
        columns,
    }
}

/// The Eq. (39) gamma contraction `Gamma_[p]^{alpha beta} D_alpha H_beta^c`.
fn build_gamma_dh_operator(degree: usize) -> SparseQiOperator {
    assert!(degree == 2 || degree == 5);
    let masks = masks_of_degree(degree);
    let output_dimension = masks.len() * VECTOR_DIMENSION;
    let mut columns = vec![Vec::new(); DH_DIMENSION];
    for (form_ordinal, mask) in masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|index| mask & (1_u16 << index) != 0)
            .collect::<Vec<_>>();
        let gamma = raised_gamma(&indices);
        for derivative in 0..SPINOR_DIMENSION {
            for h_spinor in 0..SPINOR_DIMENSION {
                let coefficient = gamma[derivative][h_spinor];
                if coefficient == 0 {
                    continue;
                }
                for vector in 0..VECTOR_DIMENSION {
                    columns[dh_index(derivative, h_spinor, vector)].push(SparseQiEntry {
                        row: form_ordinal * VECTOR_DIMENSION + vector,
                        coefficient: ExactQi::from_integer(i64::from(coefficient)),
                    });
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: DH_DIMENSION,
        output_dimension,
        columns,
    }
}

fn cached_gamma_dh_operator(degree: usize) -> &'static SparseQiOperator {
    static GAMMA_TWO: OnceLock<SparseQiOperator> = OnceLock::new();
    static GAMMA_FIVE: OnceLock<SparseQiOperator> = OnceLock::new();
    match degree {
        2 => GAMMA_TWO.get_or_init(|| build_gamma_dh_operator(2)),
        5 => GAMMA_FIVE.get_or_init(|| build_gamma_dh_operator(5)),
        _ => panic!("gamma-DH operator is defined only for degrees 2 and 5"),
    }
}

pub fn gamma_dh_operator(degree: usize) -> SparseQiOperator {
    cached_gamma_dh_operator(degree).clone()
}

/// The `D H` part of the bosonic frame in hep-th/0101037 Eq. (25).
pub fn eq25_dh_to_bosonic_frame_operator() -> SparseQiOperator {
    let mut columns = vec![Vec::new(); DH_DIMENSION];
    for frame_vector in 0..VECTOR_DIMENSION {
        let gamma = &raised_one_gammas()[frame_vector];
        for derivative in 0..SPINOR_DIMENSION {
            for h_spinor in 0..SPINOR_DIMENSION {
                let coefficient = gamma[derivative][h_spinor];
                if coefficient == 0 {
                    continue;
                }
                for output_vector in 0..VECTOR_DIMENSION {
                    columns[dh_index(derivative, h_spinor, output_vector)].push(SparseQiEntry {
                        row: frame_index(frame_vector, output_vector),
                        coefficient: ExactQi {
                            real: r(0),
                            imaginary: rr(i64::from(coefficient), 16),
                        },
                    });
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: DH_DIMENSION,
        output_dimension: FRAME_DIMENSION,
        columns,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Eq25BosonicFrameInput {
    pub d_h: BTreeMap<usize, ExactQi>,
    pub scalar_compensator: ExactQi,
    /// Canonical lower-index two-form components by bit mask.
    pub lorentz_compensator: BTreeMap<u16, ExactQi>,
}

/// Apply the complete bosonic coefficient in Eq. (25).
pub fn apply_eq25_bosonic_frame(input: &Eq25BosonicFrameInput) -> BTreeMap<usize, ExactQi> {
    apply_eq25_with_operator(input, &eq25_dh_to_bosonic_frame_operator())
}

fn apply_eq25_with_operator(
    input: &Eq25BosonicFrameInput,
    operator: &SparseQiOperator,
) -> BTreeMap<usize, ExactQi> {
    let mut output = operator.apply_sparse(&input.d_h);
    for axis in 0..VECTOR_DIMENSION {
        add_sparse(
            &mut output,
            frame_index(axis, axis),
            input.scalar_compensator.clone(),
        );
    }
    for (&mask, value) in &input.lorentz_compensator {
        assert_eq!(mask.count_ones(), 2);
        let indices = (0..VECTOR_DIMENSION)
            .filter(|index| mask & (1_u16 << index) != 0)
            .collect::<Vec<_>>();
        let left = indices[0];
        let right = indices[1];
        add_sparse(
            &mut output,
            frame_index(left, right),
            value.scaled(&r(-lorentz_sign(right))),
        );
        add_sparse(
            &mut output,
            frame_index(right, left),
            value.scaled(&r(lorentz_sign(left))),
        );
    }
    output
}

fn frame_curl_for_momentum_axis(
    frame: &BTreeMap<usize, ExactQi>,
    momentum_axis: usize,
) -> BTreeMap<usize, ExactQi> {
    let pairs = masks_of_degree(2);
    let pair_lookup = pairs
        .iter()
        .copied()
        .enumerate()
        .map(|(index, mask)| (mask, index))
        .collect::<BTreeMap<_, _>>();
    let mut output = BTreeMap::new();
    for (&index, value) in frame {
        let a = index / VECTOR_DIMENSION;
        let c = index % VECTOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            if a == b {
                continue;
            }
            let mask = (1_u16 << a) | (1_u16 << b);
            let pair = pair_lookup[&mask];
            let (left, right) = if a < b { (a, b) } else { (b, a) };
            let factor = if momentum_axis == left && a == right {
                rr(1, 2)
            } else if momentum_axis == right && a == left {
                rr(-1, 2)
            } else {
                continue;
            };
            add_sparse(
                &mut output,
                pair * VECTOR_DIMENSION + c,
                value.scaled(&factor),
            );
        }
    }
    output
}

/// Eq. (29) bosonic anholonomy, evaluated at a single momentum basis vector.
/// This is independently equal to the unit-weight curl of Eq. (25).
pub fn apply_eq29_bosonic_anholonomy(
    input: &Eq25BosonicFrameInput,
    momentum_axis: usize,
) -> BTreeMap<usize, ExactQi> {
    assert!(momentum_axis < VECTOR_DIMENSION);
    let pairs = masks_of_degree(2);
    let mut output = BTreeMap::new();
    let raised_one = raised_one_gammas();

    for (pair, mask) in pairs.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let a = indices[0];
        let b = indices[1];
        for (&column, value) in &input.d_h {
            let vector = column % VECTOR_DIMENSION;
            let spinors = column / VECTOR_DIMENSION;
            let h_spinor = spinors % SPINOR_DIMENSION;
            let derivative = spinors / SPINOR_DIMENSION;
            let integer = if momentum_axis == b {
                -raised_one[a][derivative][h_spinor]
            } else if momentum_axis == a {
                raised_one[b][derivative][h_spinor]
            } else {
                0
            };
            if integer != 0 {
                add_sparse(
                    &mut output,
                    pair * VECTOR_DIMENSION + vector,
                    value.multiply(&ExactQi {
                        real: r(0),
                        imaginary: rr(i64::from(integer), 32),
                    }),
                );
            }
        }

        for c in 0..VECTOR_DIMENSION {
            let scalar_integer = if momentum_axis == b && c == a {
                -1
            } else if momentum_axis == a && c == b {
                1
            } else {
                0
            };
            if scalar_integer != 0 {
                add_sparse(
                    &mut output,
                    pair * VECTOR_DIMENSION + c,
                    input.scalar_compensator.scaled(&rr(scalar_integer, 2)),
                );
            }
        }

        for (&lorentz_mask, value) in &input.lorentz_compensator {
            let lorentz_indices = (0..VECTOR_DIMENSION)
                .filter(|axis| lorentz_mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            let left = lorentz_indices[0];
            let right = lorentz_indices[1];
            for c in 0..VECTOR_DIMENSION {
                let psi_a_c = if a == left && c == right {
                    value.scaled(&r(lorentz_sign(right)))
                } else if a == right && c == left {
                    value.scaled(&r(-lorentz_sign(left)))
                } else {
                    ExactQi::zero()
                };
                let psi_b_c = if b == left && c == right {
                    value.scaled(&r(lorentz_sign(right)))
                } else if b == right && c == left {
                    value.scaled(&r(-lorentz_sign(left)))
                } else {
                    ExactQi::zero()
                };
                if momentum_axis == a {
                    add_sparse(
                        &mut output,
                        pair * VECTOR_DIMENSION + c,
                        psi_b_c.scaled(&rr(-1, 2)),
                    );
                }
                if momentum_axis == b {
                    add_sparse(
                        &mut output,
                        pair * VECTOR_DIMENSION + c,
                        psi_a_c.scaled(&rr(1, 2)),
                    );
                }
            }
        }
    }
    output
}

/// The H-dependent terms of `C_{alpha,b}{}^c` in Eq. (28).
///
/// Inputs `0..DDH_DIMENSION` are ordered `D_alpha D_beta H_gamma{}^c`.
/// The remaining inputs are `partial_b H_alpha{}^c`.
pub fn eq28_h_to_c_alpha_b_c_operator() -> SparseQiOperator {
    let mut columns = vec![Vec::new(); H_SECOND_JET_DIMENSION];
    let gamma = raised_one_gammas();
    for alpha in 0..SPINOR_DIMENSION {
        for b in 0..VECTOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                for h_spinor in 0..SPINOR_DIMENSION {
                    let integer = gamma[b][beta][h_spinor];
                    if integer == 0 {
                        continue;
                    }
                    for c in 0..VECTOR_DIMENSION {
                        columns[ddh_index(alpha, beta, h_spinor, c)].push(SparseQiEntry {
                            row: c_alpha_b_c_index(alpha, b, c),
                            coefficient: ExactQi {
                                real: r(0),
                                imaginary: rr(i64::from(integer), 16),
                            },
                        });
                    }
                }
            }
            for c in 0..VECTOR_DIMENSION {
                columns[DDH_DIMENSION + ph_index(b, alpha, c)].push(SparseQiEntry {
                    row: c_alpha_b_c_index(alpha, b, c),
                    coefficient: ExactQi::from_integer(-1),
                });
            }
        }
    }
    SparseQiOperator {
        input_dimension: H_SECOND_JET_DIMENSION,
        output_dimension: C_ALPHA_VECTOR_VECTOR_DIMENSION,
        columns,
    }
}

/// The scalar-compensator terms in `C_{alpha,b}{}^c` from Eq. (28).
pub fn eq28_d_scalar_to_c_alpha_b_c_operator() -> SparseQiOperator {
    let gammas = real_gammas();
    let mut columns = vec![Vec::new(); SPINOR_DIMENSION];
    for b in 0..VECTOR_DIMENSION {
        let mut gamma_b = gammas[b].clone();
        if b == 0 {
            for row in &mut gamma_b {
                for value in row {
                    *value = -*value;
                }
            }
        }
        for c in 0..VECTOR_DIMENSION {
            let gamma_c_gamma_b = multiply_i8(&gammas[c], &gamma_b);
            for alpha in 0..SPINOR_DIMENSION {
                columns[alpha].push(SparseQiEntry {
                    row: c_alpha_b_c_index(alpha, b, c),
                    coefficient: if b == c {
                        ExactQi::one()
                    } else {
                        ExactQi::zero()
                    },
                });
                for gamma in 0..SPINOR_DIMENSION {
                    let integer = gamma_c_gamma_b[alpha][gamma];
                    if integer != 0 {
                        columns[gamma].push(SparseQiEntry {
                            row: c_alpha_b_c_index(alpha, b, c),
                            coefficient: ExactQi::from_rational(-i64::from(integer), 32),
                        });
                    }
                }
            }
        }
    }
    for column in &mut columns {
        let mut combined = BTreeMap::<usize, ExactQi>::new();
        for entry in column.drain(..) {
            add_sparse(&mut combined, entry.row, entry.coefficient);
        }
        *column = combined
            .into_iter()
            .map(|(row, coefficient)| SparseQiEntry { row, coefficient })
            .collect();
    }
    SparseQiOperator {
        input_dimension: SPINOR_DIMENSION,
        output_dimension: C_ALPHA_VECTOR_VECTOR_DIMENSION,
        columns,
    }
}

/// The connection-independent Eq. (44) trace `J_alpha=(4/33)T_{alpha b}{}^b`.
/// Lorentz antisymmetry makes the spinorial connection traceless here, so this
/// map acts directly on `C_{alpha,b}{}^c`.
pub fn c_alpha_b_c_to_j_operator() -> SparseQiOperator {
    let mut columns = vec![Vec::new(); C_ALPHA_VECTOR_VECTOR_DIMENSION];
    for alpha in 0..SPINOR_DIMENSION {
        for b in 0..VECTOR_DIMENSION {
            columns[c_alpha_b_c_index(alpha, b, b)].push(SparseQiEntry {
                row: alpha,
                coefficient: ExactQi::from_rational(4, 33),
            });
        }
    }
    SparseQiOperator {
        input_dimension: C_ALPHA_VECTOR_VECTOR_DIMENSION,
        output_dimension: SPINOR_DIMENSION,
        columns,
    }
}

pub fn apply_h_sector_j(second_jets: &BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi> {
    let c = eq28_h_to_c_alpha_b_c_operator().apply_sparse(second_jets);
    c_alpha_b_c_to_j_operator().apply_sparse(&c)
}

/// Solve the Table 3 spinorial Lorentz connection from
/// `T_{alpha,[de]}=(2/55)(Gamma_de)_alpha{}^gamma T_{gamma b}{}^b`.
pub fn c_alpha_b_c_to_spinorial_connection_operator() -> SparseQiOperator {
    cached_c_alpha_b_c_to_spinorial_connection_operator().clone()
}

fn cached_c_alpha_b_c_to_spinorial_connection_operator() -> &'static SparseQiOperator {
    static OPERATOR: OnceLock<SparseQiOperator> = OnceLock::new();
    OPERATOR.get_or_init(|| build_c_alpha_b_c_to_spinorial_connection_operator(55))
}

fn build_c_alpha_b_c_to_spinorial_connection_operator(trace_denominator: i64) -> SparseQiOperator {
    let pair_masks = masks_of_degree(2);
    let mut columns = vec![Vec::new(); C_ALPHA_VECTOR_VECTOR_DIMENSION];
    for (pair, mask) in pair_masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let d = indices[0];
        let e = indices[1];
        let gamma_de = gamma_product(&indices, true);

        // -(2/55) Gamma_de times the connection-invariant vector trace.
        for alpha in 0..SPINOR_DIMENSION {
            for gamma in 0..SPINOR_DIMENSION {
                let integer = gamma_de[alpha][gamma];
                if integer == 0 {
                    continue;
                }
                for b in 0..VECTOR_DIMENSION {
                    columns[c_alpha_b_c_index(gamma, b, b)].push(SparseQiEntry {
                        row: alpha * 55 + pair,
                        coefficient: ExactQi::from_rational(
                            -2 * i64::from(integer),
                            trace_denominator,
                        ),
                    });
                }
            }
        }

        // +C_{alpha,[de]}, after lowering the output vector index.
        for alpha in 0..SPINOR_DIMENSION {
            columns[c_alpha_b_c_index(alpha, d, e)].push(SparseQiEntry {
                row: alpha * 55 + pair,
                coefficient: ExactQi::from_rational(lorentz_sign(e), 2),
            });
            columns[c_alpha_b_c_index(alpha, e, d)].push(SparseQiEntry {
                row: alpha * 55 + pair,
                coefficient: ExactQi::from_rational(-lorentz_sign(d), 2),
            });
        }
    }
    SparseQiOperator {
        input_dimension: C_ALPHA_VECTOR_VECTOR_DIMENSION,
        output_dimension: SPINORIAL_CONNECTION_DIMENSION,
        columns,
    }
}

pub fn apply_spinorial_connection(
    c_alpha_b_c: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    cached_c_alpha_b_c_to_spinorial_connection_operator().apply_sparse(c_alpha_b_c)
}

/// Solve the vectorial Lorentz connection from the fourth conventional
/// constraint in hep-th/0101037 Table 3 and arXiv:2007.05097 Eq. (2.6).
///
/// At linear order, with the source conventions
/// `nabla_A=E_A+(1/2) omega_{A,d}{}^e M_e{}^d` and
/// `C_{alpha beta}{}^c=i Gamma^c_{alpha beta}`, the constraint is
///
/// ```text
/// Gamma_a^{alpha beta}
///   (D_alpha omega_{beta,de}+D_beta omega_{alpha,de}
///    -i Gamma^c_{alpha beta} omega_{c,de}) = 0.
/// ```
///
/// In this module's explicit `-Gamma C^{-1}` raised-spinor convention the exact
/// contraction is
/// `Gamma_a^{alpha beta} Gamma^c_{alpha beta}=-32 delta_a{}^c`,
/// hence `omega_{a,de}=(i/16) Gamma_a^{alpha beta}
/// D_alpha omega_{beta,de}`.
pub fn d_spinorial_connection_to_bosonic_connection_operator() -> SparseQiOperator {
    build_d_spinorial_connection_to_bosonic_connection_operator(16)
}

fn build_d_spinorial_connection_to_bosonic_connection_operator(
    denominator: i64,
) -> SparseQiOperator {
    let gamma = raised_one_gammas();
    let mut columns = vec![Vec::new(); D_SPINORIAL_CONNECTION_DIMENSION];
    for a in 0..VECTOR_DIMENSION {
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                let integer = gamma[a][alpha][beta];
                if integer == 0 {
                    continue;
                }
                for pair in 0..55 {
                    columns[d_spinorial_connection_index(alpha, beta, pair)].push(SparseQiEntry {
                        row: bosonic_connection_index(a, pair),
                        coefficient: ExactQi {
                            real: r(0),
                            imaginary: rr(i64::from(integer), denominator),
                        },
                    });
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: D_SPINORIAL_CONNECTION_DIMENSION,
        output_dimension: BOSONIC_CONNECTION_DIMENSION,
        columns,
    }
}

pub fn apply_bosonic_connection(
    d_spinorial_connection: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    d_spinorial_connection_to_bosonic_connection_operator().apply_sparse(d_spinorial_connection)
}

/// The connection contribution to `T_{alpha,e}{}^gamma`.
///
/// Nishino-Rajpoot hep-th/0107155v2 Eq. (3.2e) fixes
/// `T_{alpha,e}{}^gamma=C_{alpha,e}{}^gamma
/// +(1/4)omega_{e,de}(Gamma^de)_alpha{}^gamma` at linear order.
pub fn bosonic_connection_to_t_alpha_e_gamma_operator() -> SparseQiOperator {
    build_bosonic_connection_to_t_alpha_e_gamma_operator(4)
}

fn build_bosonic_connection_to_t_alpha_e_gamma_operator(denominator: i64) -> SparseQiOperator {
    let pair_masks = masks_of_degree(2);
    let mut columns = vec![Vec::new(); BOSONIC_CONNECTION_DIMENSION];
    for (pair, mask) in pair_masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma_de = gamma_product(&indices, true);
        for e in 0..VECTOR_DIMENSION {
            for alpha in 0..SPINOR_DIMENSION {
                for gamma in 0..SPINOR_DIMENSION {
                    let integer = gamma_de[alpha][gamma];
                    if integer != 0 {
                        columns[bosonic_connection_index(e, pair)].push(SparseQiEntry {
                            row: t_alpha_e_gamma_index(alpha, e, gamma),
                            coefficient: ExactQi::from_rational(i64::from(integer), denominator),
                        });
                    }
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: BOSONIC_CONNECTION_DIMENSION,
        output_dimension: T_ALPHA_VECTOR_SPINOR_DIMENSION,
        columns,
    }
}

/// Assemble the complete mixed torsion from source anholonomy and the solved
/// bosonic connection.  This API deliberately takes the anholonomy as a typed
/// input because Eqs. (24)-(29) do not print the final compensator-eliminated
/// `H_hat` expression for every term.
pub fn apply_t_alpha_e_gamma(
    c_alpha_e_gamma: &BTreeMap<usize, ExactQi>,
    bosonic_connection: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    for &index in c_alpha_e_gamma.keys() {
        assert!(index < T_ALPHA_VECTOR_SPINOR_DIMENSION);
    }
    let mut output = c_alpha_e_gamma.clone();
    for (index, value) in
        bosonic_connection_to_t_alpha_e_gamma_operator().apply_sparse(bosonic_connection)
    {
        add_sparse(&mut output, index, value);
    }
    output
}

/// The anholonomy contribution to
/// `J_alpha^(1)=(4/33)T_{alpha beta}{}^beta`.
pub fn c_alpha_beta_gamma_to_j_one_operator() -> SparseQiOperator {
    let mut columns = vec![Vec::new(); SPINOR_ANHOLONOMY_DIMENSION];
    for alpha in 0..SPINOR_DIMENSION {
        for beta in 0..SPINOR_DIMENSION {
            let column = (alpha * SPINOR_DIMENSION + beta) * SPINOR_DIMENSION + beta;
            columns[column].push(SparseQiEntry {
                row: alpha,
                coefficient: ExactQi::from_rational(4, 33),
            });
        }
    }
    SparseQiOperator {
        input_dimension: SPINOR_ANHOLONOMY_DIMENSION,
        output_dimension: SPINOR_DIMENSION,
        columns,
    }
}

/// The spinorial-connection contribution to `J^(1)` in the same source
/// Lorentz-generator convention as `apply_t_alpha_e_gamma`.  The stored
/// connection coordinate is `omega_[de]`, so the trace contracts it with
/// `Gamma^de`, including the raised-index sign on boost pairs.
pub fn spinorial_connection_to_j_one_operator() -> SparseQiOperator {
    cached_spinorial_connection_to_j_one_operator().clone()
}

fn cached_spinorial_connection_to_j_one_operator() -> &'static SparseQiOperator {
    static OPERATOR: OnceLock<SparseQiOperator> = OnceLock::new();
    OPERATOR.get_or_init(|| build_spinorial_connection_to_j_one_operator(33, false))
}

fn build_spinorial_connection_to_j_one_operator(
    denominator: i64,
    lower_gamma_indices: bool,
) -> SparseQiOperator {
    let pair_masks = masks_of_degree(2);
    let mut columns = vec![Vec::new(); SPINORIAL_CONNECTION_DIMENSION];
    for (pair, mask) in pair_masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma_de = gamma_product(&indices, lower_gamma_indices);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                let integer = gamma_de[alpha][beta];
                if integer != 0 {
                    // (4/33)*(1/4) from the traced connection action.
                    columns[spinorial_connection_index(beta, pair)].push(SparseQiEntry {
                        row: alpha,
                        coefficient: ExactQi::from_rational(i64::from(integer), denominator),
                    });
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: SPINORIAL_CONNECTION_DIMENSION,
        output_dimension: SPINOR_DIMENSION,
        columns,
    }
}

pub fn apply_j_one(
    c_alpha_beta_gamma: &BTreeMap<usize, ExactQi>,
    spinorial_connection: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = c_alpha_beta_gamma_to_j_one_operator().apply_sparse(c_alpha_beta_gamma);
    for (index, value) in
        cached_spinorial_connection_to_j_one_operator().apply_sparse(spinorial_connection)
    {
        add_sparse(&mut output, index, value);
    }
    output
}

/// Inject the derivative of the Lorentz p=2 compensator into
/// `D_alpha Delta_delta{}^epsilon` using hep-th/0101037 Eq. (24),
/// `Delta|_[2]=(1/2) Psi^[de] Gamma_de`.
///
/// The input coordinate is one independent lower component `Psi_de`.  The
/// ordered source sum removes the displayed `1/2`, and raising the vector
/// indices converts the matrix factor to `Gamma^d Gamma^e`.
pub fn inject_d_lorentz_compensator_into_d_delta(
    d_psi_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    inject_d_holonomy_form_into_d_delta(2, d_psi_two)
}

fn inject_holonomy_form(
    degree: usize,
    outer_dimension: usize,
    input: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    assert!((1..=5).contains(&degree));
    let masks = masks_of_degree(degree);
    let form_dimension = masks.len();
    let mut output = BTreeMap::new();
    for (&index, value) in input {
        assert!(index < outer_dimension * form_dimension);
        let form = index % form_dimension;
        let outer = index / form_dimension;
        let mask = masks[form];
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = gamma_product(&indices, false);
        let coefficient = if degree % 2 == 0 {
            value.clone()
        } else {
            value.times_i()
        };
        for delta in 0..SPINOR_DIMENSION {
            for epsilon in 0..SPINOR_DIMENSION {
                let integer = gamma[delta][epsilon];
                if integer != 0 {
                    add_sparse(
                        &mut output,
                        (outer * SPINOR_DIMENSION + delta) * SPINOR_DIMENSION + epsilon,
                        coefficient.scaled(&r(i64::from(integer))),
                    );
                }
            }
        }
    }
    output
}

/// Inject independent lower-index p-form coordinates into Eq. (1) `Delta`.
/// The displayed `1/p!` cancels the ordered Einstein sum for one increasing
/// stored mask.  Odd Clifford degrees carry the source's explicit factor i.
pub fn inject_holonomy_form_into_delta(
    degree: usize,
    psi: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    inject_holonomy_form(degree, 1, psi)
}

/// Derivative-major lift of [`inject_holonomy_form_into_delta`].
pub fn inject_d_holonomy_form_into_d_delta(
    degree: usize,
    d_psi: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    inject_holonomy_form(degree, SPINOR_DIMENSION, d_psi)
}

/// Apply the Delta and explicit Lorentz-compensator terms in the second line
/// of hep-th/0101037 Eq. (28) to `C_{alpha,b}{}^c`.
pub fn apply_eq28_delta_sector_to_c_alpha_b_c(
    d_delta: &BTreeMap<usize, ExactQi>,
    d_psi_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let gamma_b_raised = eq28_raised_lower_one_gammas();
    let gamma_c_lowered = eq28_lowered_upper_one_gammas();
    let mut output = BTreeMap::new();
    for (&index, value) in d_delta {
        assert!(index < SPINOR_ANHOLONOMY_DIMENSION);
        let epsilon = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let derivative = rest / SPINOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            let gamma_b_factor = i64::from(gamma_b_raised[b][derivative][delta]);
            if gamma_b_factor == 0 {
                continue;
            }
            for c in 0..VECTOR_DIMENSION {
                for alpha in 0..SPINOR_DIMENSION {
                    let gamma_c_factor = i64::from(gamma_c_lowered[c][epsilon][alpha]);
                    if gamma_c_factor != 0 {
                        add_sparse(
                            &mut output,
                            c_alpha_b_c_index(alpha, b, c),
                            value.scaled(&rr(gamma_b_factor * gamma_c_factor, 32)),
                        );
                    }
                }
            }
        }
    }

    let pair_masks = masks_of_degree(2);
    for (&index, value) in d_psi_two {
        assert!(index < SPINORIAL_CONNECTION_DIMENSION);
        let pair = index % 55;
        let alpha = index / 55;
        let mask = pair_masks[pair];
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let left = indices[0];
        let right = indices[1];
        add_sparse(
            &mut output,
            c_alpha_b_c_index(alpha, left, right),
            value.scaled(&r(-lorentz_sign(right))),
        );
        add_sparse(
            &mut output,
            c_alpha_b_c_index(alpha, right, left),
            value.scaled(&r(lorentz_sign(left))),
        );
    }
    output
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Eq14MixedSpinorAnholonomyInput {
    /// `D_alpha D_delta Delta_zeta{}^gamma`, in that major order.
    pub d_d_delta: BTreeMap<usize, ExactQi>,
    /// `partial_b Delta_alpha{}^gamma`, momentum-major.
    pub p_delta: BTreeMap<usize, ExactQi>,
    /// `partial_b Psi`.
    pub p_scale: BTreeMap<usize, ExactQi>,
    /// `D_alpha D_delta Psi`.
    pub d_d_scale: BTreeMap<usize, ExactQi>,
}

/// Assemble the first line of hep-th/0101037 Eq. (14),
/// `C_{alpha,b}{}^gamma`, from one exact constrained-frame jet.
pub fn apply_eq14_mixed_spinor_anholonomy(
    input: &Eq14MixedSpinorAnholonomyInput,
) -> BTreeMap<usize, ExactQi> {
    assert!(
        input
            .d_d_delta
            .keys()
            .all(|index| *index < DD_DELTA_DIMENSION)
    );
    assert!(input.p_delta.keys().all(|index| *index < P_DELTA_DIMENSION));
    assert!(input.p_scale.keys().all(|index| *index < VECTOR_DIMENSION));
    assert!(
        input
            .d_d_scale
            .keys()
            .all(|index| *index < SPINOR_DIMENSION * SPINOR_DIMENSION)
    );
    let gamma_b = eq28_raised_lower_one_gammas();
    let mut output = BTreeMap::new();

    for (&index, value) in &input.d_d_delta {
        let gamma = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let zeta = rest % SPINOR_DIMENSION;
        let rest = rest / SPINOR_DIMENSION;
        let delta = rest % SPINOR_DIMENSION;
        let alpha = rest / SPINOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            let integer = gamma_b[b][delta][zeta];
            if integer != 0 {
                add_sparse(
                    &mut output,
                    (alpha * VECTOR_DIMENSION + b) * SPINOR_DIMENSION + gamma,
                    value.times_i().scaled(&rr(i64::from(integer), 32)),
                );
            }
        }
    }

    for (&index, value) in &input.p_delta {
        let gamma = index % SPINOR_DIMENSION;
        let rest = index / SPINOR_DIMENSION;
        let alpha = rest % SPINOR_DIMENSION;
        let b = rest / SPINOR_DIMENSION;
        add_sparse(
            &mut output,
            (alpha * VECTOR_DIMENSION + b) * SPINOR_DIMENSION + gamma,
            value.scaled(&rr(-1, 2)),
        );
    }

    for (&b, value) in &input.p_scale {
        for alpha in 0..SPINOR_DIMENSION {
            add_sparse(
                &mut output,
                (alpha * VECTOR_DIMENSION + b) * SPINOR_DIMENSION + alpha,
                value.scaled(&rr(-1, 2)),
            );
        }
    }

    for (&index, value) in &input.d_d_scale {
        let delta = index % SPINOR_DIMENSION;
        let alpha = index / SPINOR_DIMENSION;
        for b in 0..VECTOR_DIMENSION {
            for gamma in 0..SPINOR_DIMENSION {
                let integer = gamma_b[b][delta][gamma];
                if integer != 0 {
                    add_sparse(
                        &mut output,
                        (alpha * VECTOR_DIMENSION + b) * SPINOR_DIMENSION + gamma,
                        value.times_i().scaled(&rr(i64::from(integer), 32)),
                    );
                }
            }
        }
    }
    output
}

/// Assemble `J^(1)` from the source Eq. (26), Eq. (28), and Table 3 maps for
/// a Delta jet and its explicit p=2 Lorentz coordinate.
pub fn apply_j_one_from_d_delta(
    d_delta: &BTreeMap<usize, ExactQi>,
    d_psi_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let c_spinor = eq26_spinor_anholonomy_operator().apply(d_delta);
    let c_vector = apply_eq28_delta_sector_to_c_alpha_b_c(d_delta, d_psi_two);
    let omega_spinor = apply_spinorial_connection(&c_vector);
    apply_j_one(&c_spinor, &omega_spinor)
}

/// Apply the induced `J^(1)` map to the deterministic p=2-free quotient
/// representative.  Representatives are ordered by outer spinor, then
/// Clifford degree `1,3,4,5`, then increasing Lorentz mask.  Exact exhaustive
/// annihilation of the omitted p=2 image is certified in [`verify`].
pub fn apply_induced_j_one_on_derivative_lorentz_quotient(
    p_two_free_d_delta: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    apply_j_one_from_d_delta(p_two_free_d_delta, &BTreeMap::new())
}

/// Exact arXiv:2007.05097 Eq. (2.21) basis change.
pub fn apply_j_plus(
    j_one: &BTreeMap<usize, ExactQi>,
    j_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for source in [j_one, j_two] {
        for (&index, value) in source {
            assert!(index < SPINOR_DIMENSION);
            add_sparse(&mut output, index, value.scaled(&rr(1, 2)));
        }
    }
    output
}

/// The derivative-level Eq. (2.21) basis change used by the linearized Weyl
/// curvature in Eq. (2.23).
pub fn apply_d_j_plus(
    d_j_one: &BTreeMap<usize, ExactQi>,
    d_j_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for source in [d_j_one, d_j_two] {
        for (&index, value) in source {
            assert!(index < D_J_DIMENSION);
            add_sparse(&mut output, index, value.scaled(&rr(1, 2)));
        }
    }
    output
}

/// The torsion term in arXiv:2007.05097 Eq. (2.23), in its all-real gamma convention.
pub fn t_alpha_e_gamma_to_w_operator() -> SparseQiOperator {
    let four_masks = masks_of_degree(4);
    let gammas = real_gammas();
    let mut columns = vec![Vec::new(); T_ALPHA_VECTOR_SPINOR_DIMENSION];
    for (form, mask) in four_masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma_four_lower = gamma_product(&indices, true);
        for e in 0..VECTOR_DIMENSION {
            let mixed = multiply_i8_i16(&gammas[e], &gamma_four_lower);
            for gamma in 0..SPINOR_DIMENSION {
                for alpha in 0..SPINOR_DIMENSION {
                    let integer = mixed[gamma][alpha];
                    if integer != 0 {
                        columns[t_alpha_e_gamma_index(alpha, e, gamma)].push(SparseQiEntry {
                            row: form,
                            coefficient: ExactQi::from_rational(i64::from(integer), 32),
                        });
                    }
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: T_ALPHA_VECTOR_SPINOR_DIMENSION,
        output_dimension: W_FOUR_FORM_DIMENSION,
        columns,
    }
}

/// The linearized `J^(+)` derivative in arXiv:2007.05097 Eq. (2.22).
/// Its all-real-gamma coefficient is `(i/32)(11/4)=11i/128`.
pub fn d_j_to_w_operator() -> SparseQiOperator {
    let four_masks = masks_of_degree(4);
    let mut columns = vec![Vec::new(); D_J_DIMENSION];
    for (form, mask) in four_masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = raised_gamma(&indices);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                let integer = gamma[alpha][beta];
                if integer != 0 {
                    columns[d_j_index(alpha, beta)].push(SparseQiEntry {
                        row: form,
                        coefficient: ExactQi {
                            real: r(0),
                            imaginary: rr(11 * i64::from(integer), 128),
                        },
                    });
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: D_J_DIMENSION,
        output_dimension: W_FOUR_FORM_DIMENSION,
        columns,
    }
}

pub fn apply_linearized_w(
    t_alpha_e_gamma: &BTreeMap<usize, ExactQi>,
    d_j: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = t_alpha_e_gamma_to_w_operator().apply_sparse(t_alpha_e_gamma);
    for (index, value) in d_j_to_w_operator().apply_sparse(d_j) {
        add_sparse(&mut output, index, value);
    }
    output
}

/// The torsion input operator in hep-th/0101037 Eq. (44), whose older gamma
/// convention differs by a displayed factor of `i`.
pub fn t_alpha_e_gamma_to_w_2001_operator() -> SparseQiOperator {
    t_alpha_e_gamma_to_w_operator().scaled(ExactQi::i())
}

/// The `J^(2)` derivative input in hep-th/0101037 Eq. (44), after substituting
/// `T_{alpha b}{}^b=(33/4)J^(2)_alpha`.  Its coefficient is `-11/128`.
pub fn d_j_two_to_w_2001_operator() -> SparseQiOperator {
    let four_masks = masks_of_degree(4);
    let mut columns = vec![Vec::new(); D_J_DIMENSION];
    for (form, mask) in four_masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = raised_gamma(&indices);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                let integer = gamma[alpha][beta];
                if integer != 0 {
                    columns[d_j_index(alpha, beta)].push(SparseQiEntry {
                        row: form,
                        coefficient: ExactQi::from_rational(-11 * i64::from(integer), 128),
                    });
                }
            }
        }
    }
    SparseQiOperator {
        input_dimension: D_J_DIMENSION,
        output_dimension: W_FOUR_FORM_DIMENSION,
        columns,
    }
}

pub fn apply_linearized_w_2001(
    t_alpha_e_gamma: &BTreeMap<usize, ExactQi>,
    d_j_two: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let mut output = t_alpha_e_gamma_to_w_2001_operator().apply_sparse(t_alpha_e_gamma);
    for (index, value) in d_j_two_to_w_2001_operator().apply_sparse(d_j_two) {
        add_sparse(&mut output, index, value);
    }
    output
}

/// Convention-separated output of the source-fixed linearized curvature
/// assembly.  The 2001 formula uses `D J^(2)`, while the 2021 all-real-gamma
/// formula uses `D J^(+)`; they are intentionally never conflated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConventionSeparatedLinearizedW {
    pub t_alpha_e_gamma: BTreeMap<usize, ExactQi>,
    pub d_j_plus: BTreeMap<usize, ExactQi>,
    pub w_2001: BTreeMap<usize, ExactQi>,
    pub w_2021: BTreeMap<usize, ExactQi>,
}

/// Assemble both published linearized Weyl-curvature conventions from the
/// geometrical source data.  This is the complete exact `T/J -> W` map.
/// It is not labeled `W(H_hat)` because the papers do not print the remaining
/// compensator-eliminated jets needed to build all four inputs from `H_hat`.
pub fn assemble_convention_separated_linearized_w(
    c_alpha_e_gamma: &BTreeMap<usize, ExactQi>,
    bosonic_connection: &BTreeMap<usize, ExactQi>,
    d_j_one: &BTreeMap<usize, ExactQi>,
    d_j_two: &BTreeMap<usize, ExactQi>,
) -> ConventionSeparatedLinearizedW {
    let t_alpha_e_gamma = apply_t_alpha_e_gamma(c_alpha_e_gamma, bosonic_connection);
    let d_j_plus = apply_d_j_plus(d_j_one, d_j_two);
    let w_2001 = apply_linearized_w_2001(&t_alpha_e_gamma, d_j_two);
    let w_2021 = apply_linearized_w(&t_alpha_e_gamma, &d_j_plus);
    ConventionSeparatedLinearizedW {
        t_alpha_e_gamma,
        d_j_plus,
        w_2001,
        w_2021,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EliminatedCompensatorImage {
    pub trace_image: BTreeMap<usize, ExactQi>,
    pub exterior_image: BTreeMap<usize, ExactQi>,
    pub combined_image: BTreeMap<usize, ExactQi>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalXImage {
    pub x_two_11000: BTreeMap<usize, ExactQi>,
    pub x_five_10002: BTreeMap<usize, ExactQi>,
    pub x_two_compensators: EliminatedCompensatorImage,
    pub x_five_compensators: EliminatedCompensatorImage,
}

/// Formal eleven-momentum monomial carried through the partial `F_X` map.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FxMomentumMonomial {
    pub exponents: [u16; VECTOR_DIMENSION],
}

impl FxMomentumMonomial {
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
}

/// One Cartesian-Majorana polynomial `D_alpha H_beta{}^c` term.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialFxDhTerm {
    pub derivative_spinor: usize,
    pub h_spinor: usize,
    pub output_vector: usize,
    pub exterior_spinor_mask: u32,
    pub momentum: FxMomentumMonomial,
    pub coefficient: ExactQi,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PolynomialFxOutputKey {
    pub exterior_spinor_mask: u32,
    pub momentum: FxMomentumMonomial,
    /// Coordinate in the canonical ambient form-vector ordering.  The value
    /// lies in the exact hook-projector image of rank 429 or 4290.
    pub quotient_coordinate: usize,
}

/// The partial physical curvature `F_X=(X_[2],X_[5])` on the complete
/// `429+4290` conventional quotient.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialFxImage {
    pub x_two_11000: BTreeMap<PolynomialFxOutputKey, ExactQi>,
    pub x_five_10002: BTreeMap<PolynomialFxOutputKey, ExactQi>,
}

/// Apply the exact source-normalized `F_X` to a sparse polynomial stream.
/// Momentum monomials and exterior spinor normal forms are preserved exactly;
/// no `J` or `W` component is implied by this API.
pub fn apply_polynomial_fx(terms: &[PolynomialFxDhTerm]) -> PolynomialFxImage {
    let mut grouped = BTreeMap::<(u32, FxMomentumMonomial), BTreeMap<usize, ExactQi>>::new();
    for term in terms {
        assert!(term.derivative_spinor < SPINOR_DIMENSION);
        assert!(term.h_spinor < SPINOR_DIMENSION);
        assert!(term.output_vector < VECTOR_DIMENSION);
        add_sparse(
            grouped
                .entry((term.exterior_spinor_mask, term.momentum.clone()))
                .or_default(),
            dh_index(term.derivative_spinor, term.h_spinor, term.output_vector),
            term.coefficient.clone(),
        );
    }
    let mut x_two_11000 = BTreeMap::new();
    let mut x_five_10002 = BTreeMap::new();
    for ((exterior_spinor_mask, momentum), d_h) in grouped {
        let image = apply_leading_physical_x(&d_h);
        for (quotient_coordinate, coefficient) in image.x_two_11000 {
            x_two_11000.insert(
                PolynomialFxOutputKey {
                    exterior_spinor_mask,
                    momentum: momentum.clone(),
                    quotient_coordinate,
                },
                coefficient,
            );
        }
        for (quotient_coordinate, coefficient) in image.x_five_10002 {
            x_five_10002.insert(
                PolynomialFxOutputKey {
                    exterior_spinor_mask,
                    momentum: momentum.clone(),
                    quotient_coordinate,
                },
                coefficient,
            );
        }
    }
    PolynomialFxImage {
        x_two_11000,
        x_five_10002,
    }
}

/// K/FAG harness adapter.  The exact B5-to-Cartesian-Majorana intertwiner now
/// exists, but the current harness key retains only a target-basis ordinal and
/// drops the raw vector/spinor coordinates needed to apply that intertwiner.
/// The adapter therefore rejects this lossy boundary.  The standalone
/// Cartesian API above remains fully executable.
pub struct CartesianPolynomialFxApi;

fn derivative_wedge_sign(mask: u32, derivative_weight: usize) -> i64 {
    let greater = if derivative_weight + 1 == u32::BITS as usize {
        0
    } else {
        (mask >> (derivative_weight + 1)).count_ones()
    };
    if greater % 2 == 0 { 1 } else { -1 }
}

fn cached_b5_majorana_target_join()
-> &'static crate::eleven_dimensional_b5_majorana_target_join::ExactB5MajoranaTargetJoin {
    static JOIN: OnceLock<
        crate::eleven_dimensional_b5_majorana_target_join::ExactB5MajoranaTargetJoin,
    > = OnceLock::new();
    JOIN.get_or_init(crate::eleven_dimensional_b5_majorana_target_join::exact_target_join)
}

impl crate::eleven_dimensional_k_fag_solver::PhysicalCurvaturePolynomialApi
    for CartesianPolynomialFxApi
{
    fn descriptor(&self) -> crate::eleven_dimensional_k_fag_solver::PhysicalCurvatureApiDescriptor {
        crate::eleven_dimensional_k_fag_solver::PhysicalCurvatureApiDescriptor {
            schema_version: "adynkra-11d-partial-polynomial-fx-v1".to_string(),
            provenance_sha256: vec![
                HEP_TH_0101037_SOURCE_SHA256.to_string(),
                ARXIV_2007_05097_SOURCE_SHA256.to_string(),
            ],
            accepted_target_basis: "explicit Cartesian 11D Majorana D_alpha H_beta{}^c; ordinal-only target keys are rejected".to_string(),
            target_basis_join_complete: true,
            output_is_conventional_quotient_coordinates: true,
            output_quotient_complete: true,
            derivative_normal_form_complete: true,
            generic_polynomial_action_complete: true,
            complete_physical_f: false,
        }
    }

    fn apply_term(
        &self,
        input: &crate::eleven_dimensional_k_fag_solver::TargetVariationKey,
        coefficient: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
    ) -> Result<
        Vec<(
            crate::eleven_dimensional_k_fag_solver::CurvatureVariationKey,
            crate::eleven_dimensional_k_fag_solver::ExactGaussian,
        )>,
        String,
    > {
        let vector_weight = input.target_vector_weight_index.ok_or_else(|| {
            "partial F_X requires target_vector_weight_index in TargetVariationKey".to_string()
        })?;
        let target_spinor = input.target_spinor_weight_index.ok_or_else(|| {
            "partial F_X requires target_spinor_weight_index in TargetVariationKey".to_string()
        })?;
        if vector_weight >= VECTOR_DIMENSION || target_spinor >= SPINOR_DIMENSION {
            return Err("B5 target coordinate is out of range".to_string());
        }
        let join = cached_b5_majorana_target_join();
        let momentum = FxMomentumMonomial {
            exponents: input.momentum_monomial.exponents,
        };
        let mut terms = Vec::new();
        for derivative_weight in 0..SPINOR_DIMENSION {
            if input.spinor_derivative_mask & (1_u32 << derivative_weight) != 0 {
                continue;
            }
            let wedge_sign = derivative_wedge_sign(input.spinor_derivative_mask, derivative_weight);
            let output_mask = input.spinor_derivative_mask | (1_u32 << derivative_weight);
            for derivative_majorana in 0..SPINOR_DIMENSION {
                let derivative_factor =
                    &join.spinor_to_majorana[derivative_majorana][derivative_weight];
                if derivative_factor.re == r(0) && derivative_factor.im == r(0) {
                    continue;
                }
                for h_majorana in 0..SPINOR_DIMENSION {
                    let h_factor = &join.spinor_to_majorana[h_majorana][target_spinor];
                    if h_factor.re == r(0) && h_factor.im == r(0) {
                        continue;
                    }
                    for output_vector in 0..VECTOR_DIMENSION {
                        let vector_factor =
                            &join.upper_vector_to_lorentz[output_vector][vector_weight];
                        if vector_factor.re == r(0) && vector_factor.im == r(0) {
                            continue;
                        }
                        let factor = derivative_factor.clone()
                            * h_factor.clone()
                            * vector_factor.clone()
                            * num_complex::Complex::new(r(wedge_sign), r(0));
                        terms.push(PolynomialFxDhTerm {
                            derivative_spinor: derivative_majorana,
                            h_spinor: h_majorana,
                            output_vector,
                            exterior_spinor_mask: output_mask,
                            momentum: momentum.clone(),
                            coefficient: ExactQi {
                                real: factor.re,
                                imaginary: factor.im,
                            },
                        });
                    }
                }
            }
        }
        let image = apply_polynomial_fx(&terms);
        let mut output = BTreeMap::<
            crate::eleven_dimensional_k_fag_solver::CurvatureVariationKey,
            crate::eleven_dimensional_k_fag_solver::ExactGaussian,
        >::new();
        let mut add_sector = |sector: &str, key: PolynomialFxOutputKey, value: ExactQi| {
            let small_real = num_rational::Ratio::new(
                num_bigint::BigInt::from(*value.real.numer()),
                num_bigint::BigInt::from(*value.real.denom()),
            );
            let small_imaginary = num_rational::Ratio::new(
                num_bigint::BigInt::from(*value.imaginary.numer()),
                num_bigint::BigInt::from(*value.imaginary.denom()),
            );
            let product = crate::eleven_dimensional_k_fag_solver::ExactGaussian {
                real: coefficient.real.clone() * small_real.clone()
                    - coefficient.imaginary.clone() * small_imaginary.clone(),
                imaginary: coefficient.real.clone() * small_imaginary
                    + coefficient.imaginary.clone() * small_real,
            };
            let output_key = crate::eleven_dimensional_k_fag_solver::CurvatureVariationKey {
                parameter_component: input.parameter_component,
                output_sector: sector.to_string(),
                output_coordinate: key.quotient_coordinate,
                spinor_derivative_mask: key.exterior_spinor_mask,
                spinor_derivative_order: key.exterior_spinor_mask.count_ones() as usize,
                momentum_monomial: crate::eleven_dimensional_k_fag_solver::MomentumMonomial {
                    exponents: key.momentum.exponents,
                },
            };
            let entry = output
                .entry(output_key)
                .or_insert_with(crate::eleven_dimensional_k_fag_solver::ExactGaussian::zero);
            entry.real += product.real;
            entry.imaginary += product.imaginary;
        };
        for (key, value) in image.x_two_11000 {
            add_sector("X2_11000", key, value);
        }
        for (key, value) in image.x_five_10002 {
            add_sector("X5_10002", key, value);
        }
        Ok(output
            .into_iter()
            .filter(|(_, value)| !value.is_zero())
            .collect())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConventionalCompensatorSolution {
    /// The one-form `Psi_a` fixed by `X_ab^b=0`.
    pub psi_one: BTreeMap<u16, ExactQi>,
    /// The three-form `Psi_[abc]` fixed by `X_[abc]=0`.
    pub psi_three: BTreeMap<u16, ExactQi>,
    /// The four-form `Psi_[a1...a4]` fixed by the X_[5] trace.
    pub psi_four: BTreeMap<u16, ExactQi>,
    /// The individually normalized five-form fixed by the exterior Eq. (40)
    /// constraint and the ordered-index convention of Eq. (39).
    pub psi_five: BTreeMap<u16, ExactQi>,
    /// The unique contracted epsilon image of the p=5 holonomy field.
    /// The source does not state enough summation detail to invert this image
    /// to a uniquely normalized named `Psi_[5]` field.
    pub epsilon_psi_five_image: BTreeMap<usize, ExactQi>,
}

/// The derivative lift of the Eq. (40) conventional-compensator solve.
/// Keys in the named p-form maps are `outer_spinor * C(11,p) + form`.
/// Keys in the p=5 contracted image are
/// `outer_spinor * (C(11,5)*11) + ambient_image_index`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HigherJetCompensatorSolution {
    pub d_psi_one: BTreeMap<usize, ExactQi>,
    pub d_psi_three: BTreeMap<usize, ExactQi>,
    pub d_psi_four: BTreeMap<usize, ExactQi>,
    pub d_psi_five: BTreeMap<usize, ExactQi>,
    pub d_epsilon_psi_five_image: BTreeMap<usize, ExactQi>,
}

fn epsilon_sign(indices: &[usize]) -> i64 {
    let inversions = indices
        .iter()
        .enumerate()
        .map(|(left, value)| {
            indices[left + 1..]
                .iter()
                .filter(|right| *right < value)
                .count()
        })
        .sum::<usize>();
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn hodge_five_to_six(psi_five: &BTreeMap<u16, ExactQi>) -> BTreeMap<u16, ExactQi> {
    let full = (1_u16 << VECTOR_DIMENSION) - 1;
    let mut output = BTreeMap::new();
    for (&five_mask, value) in psi_five {
        assert_eq!(five_mask.count_ones(), 5);
        let six_mask = full ^ five_mask;
        let mut ordered = (0..VECTOR_DIMENSION)
            .filter(|axis| six_mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        ordered.extend((0..VECTOR_DIMENSION).filter(|axis| five_mask & (1_u16 << axis) != 0));
        let metric = (0..VECTOR_DIMENSION)
            .filter(|axis| five_mask & (1_u16 << axis) != 0)
            .map(lorentz_sign)
            .product::<i64>();
        output.insert(six_mask, value.scaled(&r(epsilon_sign(&ordered) * metric)));
    }
    output.retain(|_, value| !value.is_zero());
    output
}

fn inverse_hodge_six_to_five(star: &BTreeMap<u16, ExactQi>) -> BTreeMap<u16, ExactQi> {
    let full = (1_u16 << VECTOR_DIMENSION) - 1;
    let mut output = BTreeMap::new();
    for (&six_mask, value) in star {
        assert_eq!(six_mask.count_ones(), 6);
        let five_mask = full ^ six_mask;
        let mut ordered = (0..VECTOR_DIMENSION)
            .filter(|axis| six_mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        ordered.extend((0..VECTOR_DIMENSION).filter(|axis| five_mask & (1_u16 << axis) != 0));
        let metric = (0..VECTOR_DIMENSION)
            .filter(|axis| five_mask & (1_u16 << axis) != 0)
            .map(lorentz_sign)
            .product::<i64>();
        output.insert(five_mask, value.scaled(&r(epsilon_sign(&ordered) * metric)));
    }
    output.retain(|_, value| !value.is_zero());
    output
}

/// Solve the p=1, p=3, and p=4 conventional compensators in the explicit
/// unit-weight convention, and return the unique p=5 contracted image.
pub fn solve_conventional_compensators(
    d_h: &BTreeMap<usize, ExactQi>,
) -> ConventionalCompensatorSolution {
    let raw_two = sparse_to_tensor(2, &cached_gamma_dh_operator(2).apply_sparse(d_h));
    let raw_five = sparse_to_tensor(5, &cached_gamma_dh_operator(5).apply_sparse(d_h));
    let psi_one = mixed_trace(2, &raw_two)
        .into_iter()
        .map(|(mask, value)| (mask, value.scaled(&rr(1, 80))))
        .collect();
    let psi_three = total_antisymmetric_part(2, &raw_two)
        .into_iter()
        .map(|(mask, value)| (mask, value.scaled(&rr(1, 16))))
        .collect();
    let psi_four = mixed_trace(5, &raw_five)
        .into_iter()
        .map(|(mask, value)| (mask, value.times_i().scaled(&rr(-15, 7))))
        .collect();
    let exterior = inject_total_antisymmetric(5, &total_antisymmetric_part(5, &raw_five))
        .into_iter()
        .map(|(key, value)| (key, value.times_i().scaled(&rr(-1, 16))))
        .collect();
    // `exterior` is the correction inserted into X, namely `-*Psi_[5]`.
    // Recover the named field from the opposite six-form image.
    let star_psi_five = total_antisymmetric_part(5, &exterior)
        .into_iter()
        .map(|(mask, value)| (mask, value.scaled(&r(-1))))
        .collect();
    let psi_five = inverse_hodge_six_to_five(&star_psi_five);
    ConventionalCompensatorSolution {
        psi_one,
        psi_three,
        psi_four,
        psi_five,
        epsilon_psi_five_image: tensor_to_indexed(5, exterior),
    }
}

/// Differentiate the exact algebraic Eq. (40) solve once.  Because every
/// projector coefficient is constant, differentiation commutes with the
/// solve.  The input ordering is the module's `D_outer D_inner H_spinor^c`
/// ordering used by Eq. (28).
pub fn solve_higher_jet_conventional_compensators(
    d_d_h: &BTreeMap<usize, ExactQi>,
) -> HigherJetCompensatorSolution {
    let mut d_psi_one = BTreeMap::new();
    let mut d_psi_three = BTreeMap::new();
    let mut d_psi_four = BTreeMap::new();
    let mut d_psi_five = BTreeMap::new();
    let mut d_epsilon_psi_five_image = BTreeMap::new();
    for outer in 0..SPINOR_DIMENSION {
        let mut slice = BTreeMap::new();
        for (&index, value) in d_d_h {
            assert!(index < DDH_DIMENSION);
            let output_vector = index % VECTOR_DIMENSION;
            let rest = index / VECTOR_DIMENSION;
            let h_spinor = rest % SPINOR_DIMENSION;
            let rest = rest / SPINOR_DIMENSION;
            let inner = rest % SPINOR_DIMENSION;
            let input_outer = rest / SPINOR_DIMENSION;
            if input_outer == outer {
                slice.insert(dh_index(inner, h_spinor, output_vector), value.clone());
            }
        }
        if slice.is_empty() {
            continue;
        }
        let solved = solve_conventional_compensators(&slice);
        for (mask, value) in solved.psi_one {
            let form = masks_of_degree(1)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            d_psi_one.insert(outer * 11 + form, value);
        }
        for (mask, value) in solved.psi_three {
            let form = masks_of_degree(3)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            d_psi_three.insert(outer * 165 + form, value);
        }
        for (mask, value) in solved.psi_four {
            let form = masks_of_degree(4)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            d_psi_four.insert(outer * 330 + form, value);
        }
        for (mask, value) in solved.psi_five {
            let form = masks_of_degree(5)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            d_psi_five.insert(outer * 462 + form, value);
        }
        for (index, value) in solved.epsilon_psi_five_image {
            d_epsilon_psi_five_image.insert(outer * FIVE_FORM_VECTOR_DIMENSION + index, value);
        }
    }
    HigherJetCompensatorSolution {
        d_psi_one,
        d_psi_three,
        d_psi_four,
        d_psi_five,
        d_epsilon_psi_five_image,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DimensionZeroTorsion {
    /// Canonical `(alpha,beta,c)` entries.  Both spinor orders are retained.
    pub components: BTreeMap<(usize, usize, usize), ExactQi>,
}

/// Reconstruct the complete dimension-zero torsion in Eq. (38).
///
/// The displayed `1/2` and `1/120` multiply ordered Einstein sums.  In the
/// unique-mask storage used here those factorials are absorbed.  The X terms
/// retain the positive signs printed in Eq. (39).  Eq. (44) below independently
/// recovers the input X tensors and therefore checks this conversion.
pub fn reconstruct_dimension_zero_torsion(image: &PhysicalXImage) -> DimensionZeroTorsion {
    let mut components = BTreeMap::new();
    for output_vector in 0..VECTOR_DIMENSION {
        let gamma = lower_spinor_gamma(&[output_vector], false);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                if gamma[alpha][beta] != 0 {
                    components.insert(
                        (alpha, beta, output_vector),
                        ExactQi::i().scaled(&r(i64::from(gamma[alpha][beta]))),
                    );
                }
            }
        }
    }
    let two_masks = masks_of_degree(2);
    for (&index, coefficient) in &image.x_two_11000 {
        let output_vector = index % VECTOR_DIMENSION;
        let mask = two_masks[index / VECTOR_DIMENSION];
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = lower_spinor_gamma(&indices, false);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                if gamma[alpha][beta] == 0 {
                    continue;
                }
                let entry = components
                    .entry((alpha, beta, output_vector))
                    .or_insert_with(ExactQi::zero);
                entry.add_assign(&coefficient.scaled(&r(i64::from(gamma[alpha][beta]))));
            }
        }
    }
    let five_masks = masks_of_degree(5);
    for (&index, coefficient) in &image.x_five_10002 {
        let output_vector = index % VECTOR_DIMENSION;
        let mask = five_masks[index / VECTOR_DIMENSION];
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let gamma = lower_spinor_gamma(&indices, false);
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                if gamma[alpha][beta] == 0 {
                    continue;
                }
                let entry = components
                    .entry((alpha, beta, output_vector))
                    .or_insert_with(ExactQi::zero);
                entry.add_assign(
                    &coefficient
                        .times_i()
                        .scaled(&r(i64::from(gamma[alpha][beta]))),
                );
            }
        }
    }
    components.retain(|_, value| !value.is_zero());
    DimensionZeroTorsion { components }
}

/// Read one component of the reconstructed dimension-zero torsion.
pub fn dimension_zero_torsion_component(
    torsion: &DimensionZeroTorsion,
    alpha: usize,
    beta: usize,
    output_vector: usize,
) -> ExactQi {
    assert!(alpha < SPINOR_DIMENSION);
    assert!(beta < SPINOR_DIMENSION);
    assert!(output_vector < VECTOR_DIMENSION);
    torsion
        .components
        .get(&(alpha, beta, output_vector))
        .cloned()
        .unwrap_or_else(ExactQi::zero)
}

/// Recover Eq. (44)'s X projection from the reconstructed dimension-zero
/// torsion.  This is an independent convention check on the `1/2`, `i/120`,
/// `1/32`, and `i/32` factors.
pub fn recover_x_from_dimension_zero_torsion(
    image: &PhysicalXImage,
    degree: usize,
) -> BTreeMap<usize, ExactQi> {
    assert!(degree == 2 || degree == 5);
    let torsion = reconstruct_dimension_zero_torsion(image);
    let masks = masks_of_degree(degree);
    let mut output = BTreeMap::new();
    for (form, mask) in masks.into_iter().enumerate() {
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let projector_gamma = raised_gamma(&indices);
        for vector in 0..VECTOR_DIMENSION {
            let mut contraction = ExactQi::zero();
            for alpha in 0..SPINOR_DIMENSION {
                for beta in 0..SPINOR_DIMENSION {
                    let integer = projector_gamma[alpha][beta];
                    if integer == 0 {
                        continue;
                    }
                    contraction.add_assign(
                        &dimension_zero_torsion_component(&torsion, alpha, beta, vector)
                            .scaled(&r(i64::from(integer))),
                    );
                }
            }
            let projected = if degree == 2 {
                contraction.scaled(&rr(1, 32))
            } else {
                contraction.times_i().scaled(&rr(1, 32))
            };
            if !projected.is_zero() {
                output.insert(form * VECTOR_DIMENSION + vector, projected);
            }
        }
    }
    output
}

fn sparse_to_tensor(
    degree: usize,
    input: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<(u16, usize), ExactQi> {
    let basis = form_vector_basis(degree);
    input
        .iter()
        .map(|(&index, value)| {
            assert!(index < basis.len());
            (basis[index], value.clone())
        })
        .collect()
}

fn tensor_to_indexed(
    degree: usize,
    input: BTreeMap<(u16, usize), ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let lookup = form_vector_basis(degree)
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    input
        .into_iter()
        .map(|(key, value)| (lookup[&key], value))
        .collect()
}

fn eliminate_one_degree(
    degree: usize,
    raw: &BTreeMap<usize, ExactQi>,
    source_factor: ExactQi,
) -> (BTreeMap<usize, ExactQi>, EliminatedCompensatorImage) {
    let tensor = sparse_to_tensor(degree, raw);
    let trace = inject_mixed_trace(degree, &mixed_trace(degree, &tensor));
    let exterior = inject_total_antisymmetric(degree, &total_antisymmetric_part(degree, &tensor));
    let physical = hook_projection(degree, &tensor)
        .into_iter()
        .map(|(key, value)| (key, value.multiply(&source_factor)))
        .collect::<BTreeMap<_, _>>();
    let trace_image = trace
        .into_iter()
        .map(|(key, value)| (key, value.multiply(&source_factor).scaled(&r(-1))))
        .collect::<BTreeMap<_, _>>();
    let exterior_image = exterior
        .into_iter()
        .map(|(key, value)| (key, value.multiply(&source_factor).scaled(&r(-1))))
        .collect::<BTreeMap<_, _>>();
    let mut combined = trace_image.clone();
    for (&key, value) in &exterior_image {
        let entry = combined.entry(key).or_insert_with(ExactQi::zero);
        entry.add_assign(value);
        if entry.is_zero() {
            combined.remove(&key);
        }
    }
    (
        tensor_to_indexed(degree, physical),
        EliminatedCompensatorImage {
            trace_image: tensor_to_indexed(degree, trace_image),
            exterior_image: tensor_to_indexed(degree, exterior_image),
            combined_image: tensor_to_indexed(degree, combined),
        },
    )
}

/// Apply the source-fixed leading dimension-zero curvature map.
///
/// The p=1,3,4,5 holonomy fields are eliminated exactly at the image level.
/// The returned compensator images are the unique trace and exterior
/// corrections fixed by Eq. (40), without assigning an unstated normalization
/// to the paper's epsilon-contracted p=5 field.
pub fn apply_leading_physical_x(d_h: &BTreeMap<usize, ExactQi>) -> PhysicalXImage {
    let raw_two = cached_gamma_dh_operator(2).apply_sparse(d_h);
    let raw_five = cached_gamma_dh_operator(5).apply_sparse(d_h);
    let (x_two_11000, x_two_compensators) =
        eliminate_one_degree(2, &raw_two, ExactQi::from_rational(1, 16));
    let (x_five_10002, x_five_compensators) = eliminate_one_degree(
        5,
        &raw_five,
        ExactQi {
            real: r(0),
            imaginary: rr(1, 16),
        },
    );
    PhysicalXImage {
        x_two_11000,
        x_five_10002,
        x_two_compensators,
        x_five_compensators,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct JOneConventionAuditRow {
    pub spinor_variance: &'static str,
    pub antisymmetrized_sum: &'static str,
    pub gamma_ab_normalization: &'static str,
    pub equation_26_d_index_order: &'static str,
    pub table_3_connection_sign: i64,
    pub lorentz_image_residual_entries: usize,
    pub preserves_existing_green_gates: bool,
    pub qualifies: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PartialFxChannelGate {
    pub gauge_form_degree: usize,
    pub leading_k_directions_tested: usize,
    pub individually_excluded_leading_ordinals: Vec<usize>,
    pub individually_zero_x2_leading_ordinals: Vec<usize>,
    pub exact_x2_functional_rank_lower_bound: usize,
    pub all_parameter_components_covered: bool,
    pub x_five_target_stream_join_complete: bool,
    pub first_momentum_fx_composition_complete: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PhysicalCurvatureOperatorReport {
    pub schema_version: &'static str,
    pub source_locators: Vec<&'static str>,
    pub source_hashes: Vec<&'static str>,
    pub lorentz_signature: &'static str,
    pub epsilon_convention: &'static str,
    pub antisymmetrization_convention: &'static str,
    pub raised_spinor_gamma_convention: &'static str,
    pub spinorial_connection_source_relation: &'static str,
    pub mixed_torsion_connection_source_relation: &'static str,
    pub j_one_connection_trace_convention: &'static str,
    pub stored_x5_to_named_psi5_relation: &'static str,
    pub dh_dimension: usize,
    pub eq25_frame_dimension: usize,
    pub eq25_dh_operator_nonzero_entries: usize,
    pub eq25_to_eq29_curl_certificate_residual_entries: usize,
    pub eq28_h_sector_operator_nonzero_entries: usize,
    pub equation_26_factored_blocks: usize,
    pub equation_26_unit_block_normalization_residual_entries: usize,
    pub equation_26_output_symmetry_residual_entries: usize,
    pub equation_26_mutation_detected: bool,
    pub equation_24_lorentz_injection_residual_entries: usize,
    pub equation_24_lorentz_injection_mutation_detected: bool,
    pub h_sector_j_operator_implemented: bool,
    pub scalar_sector_j_coefficient: &'static str,
    pub scalar_sector_j_residual_entries: usize,
    pub spinorial_connection_dimension: usize,
    pub spinorial_connection_operator_nonzero_entries: usize,
    pub spinorial_connection_constraint_residual_entries: usize,
    pub spinorial_connection_mutation_detected: bool,
    pub bosonic_connection_dimension: usize,
    pub bosonic_connection_operator_nonzero_entries: usize,
    pub bosonic_connection_constraint_residual_entries: usize,
    pub bosonic_connection_mutation_detected: bool,
    pub mixed_torsion_connection_operator_nonzero_entries: usize,
    pub mixed_torsion_connection_mutation_detected: bool,
    pub j_one_anholonomy_operator_nonzero_entries: usize,
    pub j_one_connection_operator_nonzero_entries: usize,
    pub j_one_connection_mutation_detected: bool,
    pub j_plus_basis_change_implemented: bool,
    pub convention_separated_w_assembly_implemented: bool,
    pub t_alpha_e_gamma_to_w_operator_nonzero_entries: usize,
    pub d_j_to_w_operator_nonzero_entries: usize,
    pub linearized_w_coefficients_implemented: bool,
    pub linearized_2001_w_coefficients_implemented: bool,
    pub w_2021_linear_torsion_coefficient: &'static str,
    pub w_2021_linear_j_plus_coefficient: &'static str,
    pub w_2001_linear_torsion_coefficient: &'static str,
    pub w_2001_linear_j_two_coefficient: &'static str,
    pub j_plus_from_h_hat_implemented: bool,
    pub w_coefficient_mutation_detected: bool,
    pub gamma_two_dh_operator_nonzero_entries: usize,
    pub gamma_five_dh_operator_nonzero_entries: usize,
    pub x2_ambient_dimension: usize,
    pub x2_hook_dimension: usize,
    pub x2_compensator_image_rank: usize,
    pub x5_ambient_dimension: usize,
    pub x5_hook_dimension: usize,
    pub x5_compensator_image_rank: usize,
    pub compensator_solution_residual_entries: usize,
    pub hook_idempotence_residual_entries: usize,
    pub hook_constraint_residual_entries: usize,
    pub gamma_rank_symmetry_residual_entries: usize,
    pub dimension_zero_torsion_reconstruction_implemented: bool,
    pub equation_44_x_projection_residual_entries: usize,
    pub convention_mutation_detected: bool,
    pub equation_25_bosonic_frame_implemented: bool,
    pub equation_29_bosonic_anholonomy_implemented: bool,
    pub equation_28_h_sector_implemented: bool,
    pub equation_26_spinor_anholonomy_implemented: bool,
    pub table_3_spinorial_connection_solved: bool,
    pub table_3_bosonic_connection_solved_from_d_spinorial_connection: bool,
    pub complete_t_alpha_e_gamma_from_geometry_inputs_implemented: bool,
    pub j_one_from_geometry_inputs_implemented: bool,
    pub equations_39_40_compensator_image_eliminated: bool,
    pub higher_jet_conventional_constraint_ambient_dimension: usize,
    pub higher_jet_conventional_constraint_rank: usize,
    pub higher_jet_conventional_constraint_nullity: usize,
    pub higher_jet_solve_classification: &'static str,
    pub higher_jet_lift_residual_entries: usize,
    pub higher_jet_lift_mutation_detected: bool,
    pub derivative_lorentz_quotient_kernel_equals_image: bool,
    pub derivative_lorentz_quotient_basis: &'static str,
    pub derivative_lorentz_quotient_orthogonality_residual_entries: usize,
    pub j_one_lorentz_image_probe_residual_entries: usize,
    pub induced_j_one_on_quotient_established: bool,
    pub induced_t_and_w_on_quotient_established: bool,
    pub p5_normalization_eliminated_or_fixed_by_w: bool,
    pub p5_named_normalization_parameter: &'static str,
    pub p5_named_normalization_residual_entries: usize,
    pub p5_named_normalization_mutation_detected: bool,
    pub j_one_convention_audit: Vec<JOneConventionAuditRow>,
    pub j_one_convention_audit_qualifying_rows: usize,
    pub polynomial_fx_api_implemented: bool,
    pub polynomial_fx_output_dimension: usize,
    pub polynomial_fx_preserves_all_eleven_momenta: bool,
    pub polynomial_fx_mutation_detected: bool,
    pub partial_fx_channels: Vec<PartialFxChannelGate>,
    pub leading_x2_stream_artifact_sha256: &'static str,
    pub leading_x2_all_six_channels_composed: bool,
    pub leading_x5_all_six_channels_composed: bool,
    pub first_momentum_fx_all_six_channels_composed: bool,
    pub first_momentum_fx_declared_slice_status: FirstMomentumFxDeclaredSliceStatus,
    pub individually_excluded_leading_k_ordinals_union: Vec<usize>,
    pub individually_unexcluded_leading_k_ordinals: Vec<usize>,
    pub linear_combination_survivor_space_solved: bool,
    pub leading_x2_joint_exact_rank: usize,
    pub leading_x2_joint_exact_nullity: usize,
    pub leading_x2_joint_kernel_basis: Vec<Vec<i64>>,
    pub leading_x2_joint_kernel_proved_on_exact_source_streams: bool,
    pub leading_fx_combined_kernel_proved_by_rank_sandwich: bool,
    pub leading_fx_k_solver_rank: usize,
    pub leading_fx_k_solver_nullity: usize,
    pub leading_fx_k_solver_kernel_matches_source_relations: bool,
    pub leading_fx_k_solver_mutation_detected: bool,
    pub b5_to_cartesian_majorana_intertwiner_implemented: bool,
    pub k_fag_target_key_retains_raw_vector_spinor_coordinates: bool,
    pub k_fag_solver_adapter_present: bool,
    pub k_fag_solver_refuses_missing_target_basis_join: bool,
    pub k_fag_adapter_accepts_exact_b5_target_coordinate: bool,
    pub historical_first_momentum_controls_degrees: Vec<usize>,
    pub historical_controls_are_not_fx_compositions: bool,
    pub partial_fx_a_g_p_vanishing_established: bool,
    pub dimension_zero_x_curvature_operator_implemented: bool,
    pub individual_p1_p3_p4_fields_solvable_in_fixed_convention: bool,
    pub individual_epsilon_contracted_p5_normalization_source_fixed: bool,
    pub full_equations_24_to_29_operator_implemented: bool,
    pub spin_connections_solved: bool,
    pub w_and_j_from_h_hat_implemented: bool,
    pub physical_psi_to_h_hat_k_source_fixed: bool,
    pub complete_f_from_h_hat_implemented: bool,
    pub full_f_a_g_p_test_ready: bool,
    pub covariant_off_shell_closure_established: bool,
    pub bounded_slice_passed: bool,
    pub boundary: &'static str,
}

fn hook_residuals(degree: usize) -> (usize, usize) {
    let projector = hook_projector_operator(degree);
    let mut idempotence = 0;
    let mut constraints = 0;
    for column in 0..projector.input_dimension {
        let once = projector.columns[column]
            .iter()
            .map(|entry| (entry.row, entry.coefficient.clone()))
            .collect::<BTreeMap<_, _>>();
        let twice = projector.apply_sparse(&once);
        let keys = once
            .keys()
            .chain(twice.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        idempotence += keys
            .iter()
            .filter(|key| once.get(key) != twice.get(key))
            .count();
        let tensor = sparse_to_tensor(degree, &once);
        constraints += mixed_trace(degree, &tensor).len();
        constraints += total_antisymmetric_part(degree, &tensor).len();
    }
    (idempotence, constraints)
}

fn gamma_symmetry_residuals() -> usize {
    let mut residuals = 0;
    for degree in [1, 2, 5] {
        for mask in masks_of_degree(degree) {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|index| mask & (1_u16 << index) != 0)
                .collect::<Vec<_>>();
            let gamma = raised_gamma(&indices);
            for left in 0..SPINOR_DIMENSION {
                for right in 0..SPINOR_DIMENSION {
                    residuals += usize::from(gamma[left][right] != gamma[right][left]);
                }
            }
        }
    }
    residuals
}

fn mutation_detected() -> bool {
    let gamma = gamma_dh_operator(2);
    let hook = hook_projector_operator(2);
    for column in 0..DH_DIMENSION {
        if gamma.columns[column].is_empty() {
            continue;
        }
        let mut input = BTreeMap::new();
        input.insert(column, ExactQi::one());
        let raw = gamma.apply_sparse(&input);
        let correct = hook
            .scaled(ExactQi::from_rational(1, 16))
            .apply_sparse(&raw);
        if correct.is_empty() {
            continue;
        }
        let mutated = hook
            .scaled(ExactQi::from_rational(1, 15))
            .apply_sparse(&raw);
        return correct != mutated;
    }
    false
}

fn eq25_eq29_curl_residuals() -> usize {
    let eq25_operator = eq25_dh_to_bosonic_frame_operator();
    let representative_dh_columns = eq25_operator
        .columns
        .iter()
        .enumerate()
        .filter(|(_, column)| !column.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut residuals = 0;
    for column in representative_dh_columns {
        let mut d_h = BTreeMap::new();
        d_h.insert(column, ExactQi::one());
        let input = Eq25BosonicFrameInput {
            d_h,
            scalar_compensator: ExactQi::zero(),
            lorentz_compensator: BTreeMap::new(),
        };
        let frame = apply_eq25_with_operator(&input, &eq25_operator);
        for momentum in 0..VECTOR_DIMENSION {
            let direct = apply_eq29_bosonic_anholonomy(&input, momentum);
            let curl = frame_curl_for_momentum_axis(&frame, momentum);
            let keys = direct
                .keys()
                .chain(curl.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            residuals += keys
                .iter()
                .filter(|key| direct.get(key) != curl.get(key))
                .count();
        }
    }

    let scalar_input = Eq25BosonicFrameInput {
        d_h: BTreeMap::new(),
        scalar_compensator: ExactQi::one(),
        lorentz_compensator: BTreeMap::new(),
    };
    let scalar_frame = apply_eq25_with_operator(&scalar_input, &eq25_operator);
    for momentum in 0..VECTOR_DIMENSION {
        residuals += usize::from(
            apply_eq29_bosonic_anholonomy(&scalar_input, momentum)
                != frame_curl_for_momentum_axis(&scalar_frame, momentum),
        );
    }
    for mask in masks_of_degree(2) {
        let mut lorentz = BTreeMap::new();
        lorentz.insert(mask, ExactQi::one());
        let input = Eq25BosonicFrameInput {
            d_h: BTreeMap::new(),
            scalar_compensator: ExactQi::zero(),
            lorentz_compensator: lorentz,
        };
        let frame = apply_eq25_with_operator(&input, &eq25_operator);
        for momentum in 0..VECTOR_DIMENSION {
            residuals += usize::from(
                apply_eq29_bosonic_anholonomy(&input, momentum)
                    != frame_curl_for_momentum_axis(&frame, momentum),
            );
        }
    }
    residuals
}

fn equation_44_x_projection_residuals() -> usize {
    let mut d_h = BTreeMap::new();
    d_h.insert(dh_index(0, 0, 0), ExactQi::one());
    d_h.insert(dh_index(3, 9, 7), ExactQi::i());
    let image = apply_leading_physical_x(&d_h);
    let mut residuals = 0;
    for (degree, expected) in [(2, &image.x_two_11000), (5, &image.x_five_10002)] {
        let recovered = recover_x_from_dimension_zero_torsion(&image, degree);
        let keys = recovered
            .keys()
            .chain(expected.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        residuals += keys
            .iter()
            .filter(|key| recovered.get(key) != expected.get(key))
            .count();
    }
    residuals
}

fn compensator_solution_residuals() -> usize {
    let mut d_h = BTreeMap::new();
    d_h.insert(dh_index(0, 0, 0), ExactQi::one());
    d_h.insert(dh_index(3, 9, 7), ExactQi::i());
    let image = apply_leading_physical_x(&d_h);
    let solution = solve_conventional_compensators(&d_h);
    let psi_one_image = tensor_to_indexed(2, delta_wedge(2, &solution.psi_one));
    let psi_three_image = tensor_to_indexed(
        2,
        inject_total_antisymmetric(2, &solution.psi_three)
            .into_iter()
            .map(|(key, value)| (key, value.scaled(&r(-1))))
            .collect(),
    );
    let psi_four_image = tensor_to_indexed(
        5,
        delta_wedge(5, &solution.psi_four)
            .into_iter()
            .map(|(key, value)| (key, value.scaled(&rr(1, 48))))
            .collect(),
    );
    [
        (&psi_one_image, &image.x_two_compensators.trace_image),
        (&psi_three_image, &image.x_two_compensators.exterior_image),
        (&psi_four_image, &image.x_five_compensators.trace_image),
        (
            &solution.epsilon_psi_five_image,
            &image.x_five_compensators.exterior_image,
        ),
    ]
    .into_iter()
    .map(|(actual, expected)| {
        actual
            .keys()
            .chain(expected.keys())
            .copied()
            .collect::<BTreeSet<_>>()
            .iter()
            .filter(|key| actual.get(key) != expected.get(key))
            .count()
    })
    .sum()
}

fn p5_named_normalization_probe() -> (usize, bool) {
    let mut d_h = BTreeMap::new();
    d_h.insert(dh_index(0, 0, 0), ExactQi::one());
    d_h.insert(dh_index(3, 9, 7), ExactQi::i());
    let solution = solve_conventional_compensators(&d_h);
    let image_tensor = sparse_to_tensor(5, &solution.epsilon_psi_five_image);
    let expected_inserted = total_antisymmetric_part(5, &image_tensor);
    let actual_inserted = hodge_five_to_six(&solution.psi_five)
        .into_iter()
        .map(|(mask, value)| (mask, value.scaled(&r(-1))))
        .collect::<BTreeMap<_, _>>();
    let residuals = expected_inserted
        .keys()
        .chain(actual_inserted.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .iter()
        .filter(|key| expected_inserted.get(key) != actual_inserted.get(key))
        .count();
    let sign_mutation = hodge_five_to_six(&solution.psi_five);
    (residuals, sign_mutation != expected_inserted)
}

fn higher_jet_lift_probe() -> (usize, bool) {
    let mut input = BTreeMap::new();
    input.insert(ddh_index(2, 0, 0, 0), ExactQi::one());
    input.insert(ddh_index(7, 3, 9, 7), ExactQi::i());
    let lifted = solve_higher_jet_conventional_compensators(&input);
    let mut residuals = 0;
    for outer in [2, 7] {
        let mut slice = BTreeMap::new();
        for (&index, value) in &input {
            let output_vector = index % VECTOR_DIMENSION;
            let rest = index / VECTOR_DIMENSION;
            let h_spinor = rest % SPINOR_DIMENSION;
            let rest = rest / SPINOR_DIMENSION;
            let inner = rest % SPINOR_DIMENSION;
            if rest / SPINOR_DIMENSION == outer {
                slice.insert(dh_index(inner, h_spinor, output_vector), value.clone());
            }
        }
        let expected = solve_conventional_compensators(&slice);
        for (mask, value) in expected.psi_one {
            let form = masks_of_degree(1)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            residuals += usize::from(lifted.d_psi_one.get(&(outer * 11 + form)) != Some(&value));
        }
        for (mask, value) in expected.psi_three {
            let form = masks_of_degree(3)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            residuals += usize::from(lifted.d_psi_three.get(&(outer * 165 + form)) != Some(&value));
        }
        for (mask, value) in expected.psi_four {
            let form = masks_of_degree(4)
                .iter()
                .position(|item| *item == mask)
                .unwrap();
            residuals += usize::from(lifted.d_psi_four.get(&(outer * 330 + form)) != Some(&value));
        }
        for (index, value) in expected.epsilon_psi_five_image {
            residuals += usize::from(
                lifted
                    .d_epsilon_psi_five_image
                    .get(&(outer * FIVE_FORM_VECTOR_DIMENSION + index))
                    != Some(&value),
            );
        }
    }
    let mutation_detected = lifted
        .d_psi_one
        .values()
        .chain(lifted.d_psi_three.values())
        .chain(lifted.d_psi_four.values())
        .chain(lifted.d_epsilon_psi_five_image.values())
        .next()
        .map(|value| value != &value.scaled(&rr(81, 80)))
        .unwrap_or(false);
    (residuals, mutation_detected)
}

fn derivative_lorentz_quotient_orthogonality_residuals() -> usize {
    let two = masks_of_degree(2)
        .into_iter()
        .map(|mask| {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            gamma_product(&indices, true)
        })
        .collect::<Vec<_>>();
    let quotient = [1, 3, 4, 5]
        .into_iter()
        .flat_map(|degree| {
            masks_of_degree(degree).into_iter().map(|mask| {
                let indices = (0..VECTOR_DIMENSION)
                    .filter(|axis| mask & (1_u16 << axis) != 0)
                    .collect::<Vec<_>>();
                gamma_product(&indices, true)
            })
        })
        .collect::<Vec<_>>();
    let mut residuals = 0;
    for (left_index, left) in two.iter().enumerate() {
        for (right_index, right) in two.iter().enumerate() {
            let dot = (0..SPINOR_DIMENSION)
                .flat_map(|row| {
                    (0..SPINOR_DIMENSION).map(move |column| {
                        i64::from(left[row][column]) * i64::from(right[row][column])
                    })
                })
                .sum::<i64>();
            residuals += usize::from((left_index == right_index) != (dot != 0));
        }
        for right in &quotient {
            let dot = (0..SPINOR_DIMENSION)
                .flat_map(|row| {
                    (0..SPINOR_DIMENSION).map(move |column| {
                        i64::from(left[row][column]) * i64::from(right[row][column])
                    })
                })
                .sum::<i64>();
            residuals += usize::from(dot != 0);
        }
    }
    residuals
}

fn j_one_anholonomy_for_lorentz_basis(
    derivative: usize,
    gamma_pair: &[Vec<i16>],
    swap_d_indices: bool,
) -> BTreeMap<usize, ExactQi> {
    let eq26 = cached_eq26_spinor_anholonomy_operator();
    let mut output = BTreeMap::new();
    for block in &eq26.blocks {
        for delta in 0..SPINOR_DIMENSION {
            let input_gamma = if swap_d_indices {
                block.input_raised_spinor_gamma[delta][derivative]
            } else {
                block.input_raised_spinor_gamma[derivative][delta]
            };
            if input_gamma == 0 {
                continue;
            }
            for epsilon in 0..SPINOR_DIMENSION {
                let injected = gamma_pair[delta][epsilon];
                if injected == 0 {
                    continue;
                }
                for alpha in 0..SPINOR_DIMENSION {
                    let output_gamma = block.output_lower_spinor_gamma[alpha][epsilon];
                    if output_gamma != 0 {
                        add_sparse(
                            &mut output,
                            alpha,
                            block.coefficient.scaled(&rr(
                                2 * i64::from(input_gamma)
                                    * i64::from(injected)
                                    * i64::from(output_gamma),
                                33,
                            )),
                        );
                    }
                }
            }
        }
    }
    output
}

#[derive(Clone)]
struct JOneLorentzBasisParts {
    anholonomy_direct: BTreeMap<usize, ExactQi>,
    anholonomy_swapped: BTreeMap<usize, ExactQi>,
    connection_from_delta: BTreeMap<usize, ExactQi>,
    connection_explicit: BTreeMap<usize, ExactQi>,
}

fn j_one_lorentz_basis_parts() -> Vec<JOneLorentzBasisParts> {
    let mut output = Vec::with_capacity(SPINORIAL_CONNECTION_DIMENSION);
    for derivative in 0..SPINOR_DIMENSION {
        for (pair, mask) in masks_of_degree(2).into_iter().enumerate() {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            let gamma_pair = gamma_product(&indices, true);
            let anholonomy_direct =
                j_one_anholonomy_for_lorentz_basis(derivative, &gamma_pair, false);
            let anholonomy_swapped =
                j_one_anholonomy_for_lorentz_basis(derivative, &gamma_pair, true);
            let mut d_psi_two = BTreeMap::new();
            d_psi_two.insert(spinorial_connection_index(derivative, pair), ExactQi::one());
            let d_delta = inject_d_lorentz_compensator_into_d_delta(&d_psi_two);
            let delta_c = apply_eq28_delta_sector_to_c_alpha_b_c(&d_delta, &BTreeMap::new());
            let explicit_c = apply_eq28_delta_sector_to_c_alpha_b_c(&BTreeMap::new(), &d_psi_two);
            let connection_from_delta = cached_spinorial_connection_to_j_one_operator()
                .apply_sparse(&apply_spinorial_connection(&delta_c));
            let connection_explicit = cached_spinorial_connection_to_j_one_operator()
                .apply_sparse(&apply_spinorial_connection(&explicit_c));
            output.push(JOneLorentzBasisParts {
                anholonomy_direct,
                anholonomy_swapped,
                connection_from_delta,
                connection_explicit,
            });
        }
    }
    output
}

fn add_scaled_map(
    target: &mut BTreeMap<usize, ExactQi>,
    source: &BTreeMap<usize, ExactQi>,
    factor: &Rational,
) {
    for (&index, value) in source {
        add_sparse(target, index, value.scaled(factor));
    }
}

fn build_j_one_convention_audit() -> Vec<JOneConventionAuditRow> {
    let parts = j_one_lorentz_basis_parts();
    let mut rows = Vec::with_capacity(32);
    for (spinor_variance, variance_factor) in [("C", r(1)), ("C^-1", r(-1))] {
        for (antisymmetrized_sum, sum_factor) in [("unit-weight", r(1)), ("ordered", r(2))] {
            for (gamma_ab_normalization, gamma_factor) in [("Gamma_ab", r(1)), ("2 Gamma_ab", r(2))]
            {
                for swap_d_indices in [false, true] {
                    for table_3_connection_sign in [1_i64, -1_i64] {
                        let delta_factor =
                            variance_factor.clone() * sum_factor.clone() * gamma_factor.clone();
                        let connection_factor = delta_factor.clone() * r(table_3_connection_sign);
                        let explicit_factor = r(table_3_connection_sign);
                        let mut residuals = 0;
                        for part in &parts {
                            let mut value = BTreeMap::new();
                            add_scaled_map(
                                &mut value,
                                if swap_d_indices {
                                    &part.anholonomy_swapped
                                } else {
                                    &part.anholonomy_direct
                                },
                                &delta_factor,
                            );
                            add_scaled_map(
                                &mut value,
                                &part.connection_from_delta,
                                &connection_factor,
                            );
                            add_scaled_map(&mut value, &part.connection_explicit, &explicit_factor);
                            residuals += value.len();
                        }
                        let preserves_existing_green_gates = spinor_variance == "C"
                            && antisymmetrized_sum == "unit-weight"
                            && gamma_ab_normalization == "Gamma_ab"
                            && !swap_d_indices
                            && table_3_connection_sign == 1;
                        rows.push(JOneConventionAuditRow {
                            spinor_variance,
                            antisymmetrized_sum,
                            gamma_ab_normalization,
                            equation_26_d_index_order: if swap_d_indices {
                                "D_delta Delta_gamma"
                            } else {
                                "D_gamma Delta_delta"
                            },
                            table_3_connection_sign,
                            lorentz_image_residual_entries: residuals,
                            preserves_existing_green_gates,
                            qualifies: residuals == 0 && preserves_existing_green_gates,
                        });
                    }
                }
            }
        }
    }
    rows
}

fn j_one_convention_audit() -> Vec<JOneConventionAuditRow> {
    static AUDIT: OnceLock<Vec<JOneConventionAuditRow>> = OnceLock::new();
    AUDIT.get_or_init(build_j_one_convention_audit).clone()
}

fn spinorial_connection_constraint_residuals(operator: &SparseQiOperator) -> usize {
    let gamma_pairs = masks_of_degree(2)
        .into_iter()
        .map(|mask| {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            (indices[0], indices[1], gamma_product(&indices, true))
        })
        .collect::<Vec<_>>();
    let mut residuals = 0;
    for source in 0..C_ALPHA_VECTOR_VECTOR_DIMENSION {
        let source_output_vector = source % VECTOR_DIMENSION;
        let rest = source / VECTOR_DIMENSION;
        let source_input_vector = rest % VECTOR_DIMENSION;
        let source_alpha = rest / VECTOR_DIMENSION;
        let connection = operator.columns[source]
            .iter()
            .map(|entry| (entry.row, entry.coefficient.clone()))
            .collect::<BTreeMap<_, _>>();
        for (pair, (d, e, gamma)) in gamma_pairs.iter().enumerate() {
            for alpha in 0..SPINOR_DIMENSION {
                let mut value = connection
                    .get(&(alpha * 55 + pair))
                    .cloned()
                    .unwrap_or_else(ExactQi::zero);
                if source_alpha == alpha && source_input_vector == *d && source_output_vector == *e
                {
                    value.add_assign(&ExactQi::from_rational(-lorentz_sign(*e), 2));
                }
                if source_alpha == alpha && source_input_vector == *e && source_output_vector == *d
                {
                    value.add_assign(&ExactQi::from_rational(lorentz_sign(*d), 2));
                }
                if source_input_vector == source_output_vector {
                    value.add_assign(&ExactQi::from_rational(
                        2 * i64::from(gamma[alpha][source_alpha]),
                        55,
                    ));
                }
                residuals += usize::from(!value.is_zero());
            }
        }
    }
    residuals
}

fn spinorial_connection_mutation_detected() -> bool {
    spinorial_connection_constraint_residuals(&build_c_alpha_b_c_to_spinorial_connection_operator(
        54,
    )) > 0
}

fn bosonic_connection_constraint_residuals(denominator: i64) -> usize {
    let raised = raised_one_gammas();
    let lowered = (0..VECTOR_DIMENSION)
        .map(|axis| lower_spinor_gamma(&[axis], false))
        .collect::<Vec<_>>();
    let mut contraction = vec![vec![0_i64; VECTOR_DIMENSION]; VECTOR_DIMENSION];
    for a in 0..VECTOR_DIMENSION {
        for b in 0..VECTOR_DIMENSION {
            contraction[a][b] = (0..SPINOR_DIMENSION)
                .flat_map(|alpha| {
                    let lowered = &lowered;
                    (0..SPINOR_DIMENSION).map(move |beta| {
                        i64::from(raised[a][alpha][beta]) * i64::from(lowered[b][alpha][beta])
                    })
                })
                .sum();
        }
    }
    let mut residuals = 0;
    for a in 0..VECTOR_DIMENSION {
        for alpha in 0..SPINOR_DIMENSION {
            for beta in 0..SPINOR_DIMENSION {
                let mut coefficient = rr(2 * i64::from(raised[a][alpha][beta]), 1);
                for b in 0..VECTOR_DIMENSION {
                    coefficient += rr(
                        contraction[a][b] * i64::from(raised[b][alpha][beta]),
                        denominator,
                    );
                }
                residuals += usize::from(coefficient != r(0)) * 55;
            }
        }
    }
    residuals
}

fn bosonic_connection_mutation_detected() -> bool {
    bosonic_connection_constraint_residuals(17) > 0
}

fn mixed_torsion_connection_mutation_detected() -> bool {
    bosonic_connection_to_t_alpha_e_gamma_operator()
        != build_bosonic_connection_to_t_alpha_e_gamma_operator(5)
}

fn j_one_connection_mutation_detected() -> bool {
    let source = spinorial_connection_to_j_one_operator();
    source != build_spinorial_connection_to_j_one_operator(34, false)
        && source != build_spinorial_connection_to_j_one_operator(33, true)
}

fn w_coefficient_mutation_detected() -> bool {
    let operator = d_j_to_w_operator();
    let column = operator
        .columns
        .iter()
        .find(|column| !column.is_empty())
        .expect("the D J to W operator is nonzero");
    let correct = &column[0].coefficient;
    correct != &correct.scaled(&rr(128, 127))
}

fn equation_26_probe() -> (usize, bool) {
    let operator = eq26_spinor_anholonomy_operator();
    let block = operator
        .blocks
        .iter()
        .find(|block| {
            block
                .input_raised_spinor_gamma
                .iter()
                .flatten()
                .any(|value| *value != 0)
        })
        .unwrap();
    let (gamma, delta) = (0..SPINOR_DIMENSION)
        .flat_map(|gamma| (0..SPINOR_DIMENSION).map(move |delta| (gamma, delta)))
        .find(|(gamma, delta)| block.input_raised_spinor_gamma[*gamma][*delta] != 0)
        .unwrap();
    let mut input = BTreeMap::new();
    input.insert(
        (gamma * SPINOR_DIMENSION + delta) * SPINOR_DIMENSION,
        ExactQi::one(),
    );
    let output = operator.apply(&input);
    let mut symmetry_residuals = 0;
    for (&index, value) in &output {
        let epsilon = index % SPINOR_DIMENSION;
        let pair = index / SPINOR_DIMENSION;
        let beta = pair % SPINOR_DIMENSION;
        let alpha = pair / SPINOR_DIMENSION;
        let transpose = (beta * SPINOR_DIMENSION + alpha) * SPINOR_DIMENSION + epsilon;
        symmetry_residuals += usize::from(output.get(&transpose) != Some(value));
    }
    let mut mutated = operator.clone();
    mutated.blocks[0].coefficient = ExactQi::from_rational(1, 63);
    let mutation_detected = mutated.apply(&input) != output;
    (symmetry_residuals, mutation_detected)
}

fn equation_26_block_normalization_residuals() -> usize {
    cached_eq26_spinor_anholonomy_operator()
        .blocks
        .iter()
        .filter(|block| {
            let contraction = block
                .input_raised_spinor_gamma
                .iter()
                .flatten()
                .zip(block.output_lower_spinor_gamma.iter().flatten())
                .map(|(left, right)| i64::from(*left) * i64::from(*right))
                .sum::<i64>();
            block.coefficient.scaled(&r(contraction)) != ExactQi::one()
        })
        .count()
}

fn equation_24_lorentz_injection_probe() -> (usize, bool) {
    let pair_masks = masks_of_degree(2);
    let mut residuals = 0;
    let mut mutation_detected = true;
    for mask in [(1_u16 << 1) | (1_u16 << 2), (1_u16 << 0) | (1_u16 << 1)] {
        let pair = pair_masks
            .iter()
            .position(|candidate| *candidate == mask)
            .unwrap();
        let mut input = BTreeMap::new();
        input.insert(spinorial_connection_index(0, pair), ExactQi::one());
        let actual = inject_d_lorentz_compensator_into_d_delta(&input);
        let indices = (0..VECTOR_DIMENSION)
            .filter(|axis| mask & (1_u16 << axis) != 0)
            .collect::<Vec<_>>();
        let expected_gamma = gamma_product(&indices, false);
        let old_mutated_gamma = gamma_product(&indices, true);
        let mut old_mutated = BTreeMap::new();
        for delta in 0..SPINOR_DIMENSION {
            for epsilon in 0..SPINOR_DIMENSION {
                let key = delta * SPINOR_DIMENSION + epsilon;
                let expected = ExactQi::from_integer(i64::from(expected_gamma[delta][epsilon]));
                residuals += usize::from(
                    actual.get(&key).cloned().unwrap_or_else(ExactQi::zero) != expected,
                );
                let old_value =
                    ExactQi::from_rational(i64::from(old_mutated_gamma[delta][epsilon]), 2);
                if !old_value.is_zero() {
                    old_mutated.insert(key, old_value);
                }
            }
        }
        mutation_detected &= actual != old_mutated;
    }
    (residuals, mutation_detected)
}

fn scalar_sector_j_residuals() -> usize {
    let scalar_to_c = eq28_d_scalar_to_c_alpha_b_c_operator();
    let c_to_j = c_alpha_b_c_to_j_operator();
    let mut residuals = 0;
    for spinor in 0..SPINOR_DIMENSION {
        let mut input = BTreeMap::new();
        input.insert(spinor, ExactQi::one());
        let actual = c_to_j.apply_sparse(&scalar_to_c.apply_sparse(&input));
        let expected = ExactQi::from_rational(31, 24);
        for output in 0..SPINOR_DIMENSION {
            let value = actual.get(&output).cloned().unwrap_or_else(ExactQi::zero);
            residuals += usize::from(
                value
                    != if output == spinor {
                        expected.clone()
                    } else {
                        ExactQi::zero()
                    },
            );
        }
    }
    residuals
}

fn polynomial_fx_probe() -> (bool, bool) {
    let term = PolynomialFxDhTerm {
        derivative_spinor: 0,
        h_spinor: 0,
        output_vector: 0,
        exterior_spinor_mask: 1,
        momentum: FxMomentumMonomial::variable(10),
        coefficient: ExactQi::one(),
    };
    let output = apply_polynomial_fx(std::slice::from_ref(&term));
    let preserves = output
        .x_two_11000
        .keys()
        .chain(output.x_five_10002.keys())
        .all(|key| {
            key.exterior_spinor_mask == 1 && key.momentum == FxMomentumMonomial::variable(10)
        });
    let mut mutated = term;
    mutated.coefficient = ExactQi::from_integer(2);
    let mutation_detected = output != apply_polynomial_fx(&[mutated]);
    (preserves, mutation_detected)
}

fn partial_fx_channels() -> Vec<PartialFxChannelGate> {
    let killed = [
        vec![0, 1, 2, 4, 6, 7, 8, 10],
        vec![1, 7, 8, 9, 10, 11],
        vec![7, 8, 9, 11],
        vec![7, 8, 9, 10, 11],
        vec![8, 9, 10],
        vec![9, 11],
    ];
    let ranks = [1, 4, 3, 5, 2, 2];
    killed
        .into_iter()
        .enumerate()
        .map(
            |(gauge_form_degree, individually_excluded_leading_ordinals)| {
                let individually_zero_x2_leading_ordinals = (0..12)
                    .filter(|ordinal| !individually_excluded_leading_ordinals.contains(ordinal))
                    .collect();
                PartialFxChannelGate {
                    gauge_form_degree,
                    leading_k_directions_tested: 12,
                    individually_excluded_leading_ordinals,
                    individually_zero_x2_leading_ordinals,
                    exact_x2_functional_rank_lower_bound: ranks[gauge_form_degree],
                    all_parameter_components_covered: true,
                    x_five_target_stream_join_complete: false,
                    first_momentum_fx_composition_complete: false,
                }
            },
        )
        .collect()
}

fn k_fag_adapter_refuses_missing_join() -> bool {
    use crate::eleven_dimensional_k_fag_solver::PhysicalCurvaturePolynomialApi;
    CartesianPolynomialFxApi
        .apply_term(
            &crate::eleven_dimensional_k_fag_solver::TargetVariationKey {
                parameter_component: 0,
                target_coordinate: 0,
                target_vector_weight_index: None,
                target_spinor_weight_index: None,
                spinor_derivative_mask: 0,
                spinor_derivative_order: 0,
                momentum_monomial:
                    crate::eleven_dimensional_k_fag_solver::MomentumMonomial::constant(),
            },
            &crate::eleven_dimensional_k_fag_solver::ExactGaussian::one(),
        )
        .is_err()
}

fn k_fag_adapter_accepts_exact_join() -> bool {
    use crate::eleven_dimensional_k_fag_solver::PhysicalCurvaturePolynomialApi;
    CartesianPolynomialFxApi
        .apply_term(
            &crate::eleven_dimensional_k_fag_solver::TargetVariationKey {
                parameter_component: 0,
                target_coordinate: 0,
                target_vector_weight_index: Some(0),
                target_spinor_weight_index: Some(0),
                spinor_derivative_mask: u32::MAX ^ 1,
                spinor_derivative_order: 31,
                momentum_monomial:
                    crate::eleven_dimensional_k_fag_solver::MomentumMonomial::constant(),
            },
            &crate::eleven_dimensional_k_fag_solver::ExactGaussian::one(),
        )
        .is_ok_and(|output| !output.is_empty())
}

fn leading_fx_k_solver_probe() -> (usize, usize, bool, bool) {
    use crate::eleven_dimensional_k_fag_solver::{
        ExactGaussian as SolverGaussian, ExactPolynomialSystem, MomentumMonomial,
        PolynomialConstraintKey,
    };
    let specs = crate::eleven_dimensional_k_fag_solver::recorded_12_plus_44_k_ansatz()
        .into_iter()
        .take(12)
        .collect::<Vec<_>>();
    let build = |mutated: bool| {
        let mut system = ExactPolynomialSystem::new(specs.clone(), true);
        let key = |row| PolynomialConstraintKey {
            gauge_form_degree: 6,
            parameter_component: row,
            output_sector: "leading_FX_rank_sandwich".to_string(),
            output_coordinate: row,
            spinor_derivative_mask: 0,
            spinor_derivative_order: 18,
            momentum_monomial: MomentumMonomial::constant(),
        };
        for (row, variable) in [1_usize, 7, 8, 9, 10, 11].into_iter().enumerate() {
            system.add_coefficient(key(row), variable, SolverGaussian::one());
        }
        for (variable, coefficient) in [(0, 1), (2, 18), (4, -30), (6, -54)] {
            system.add_coefficient(key(6), variable, SolverGaussian::from_integer(coefficient));
        }
        if mutated {
            system.add_coefficient(key(7), 3, SolverGaussian::one());
        }
        system.solve()
    };
    let solution = build(false);
    let mutated = build(true);
    let relations = [
        [-18, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [30, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [54, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ];
    let kernel_matches = solution.rank == 7
        && solution.nullity == relations.len()
        && relations.iter().all(|relation| {
            relation[1] == 0
                && relation[7..].iter().all(|value| *value == 0)
                && relation[0] + 18 * relation[2] - 30 * relation[4] - 54 * relation[6] == 0
        });
    (
        solution.rank,
        solution.nullity,
        kernel_matches,
        mutated.rank == solution.rank + 1 && mutated.nullity + 1 == solution.nullity,
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstMomentumFxFunctionalChannelReport {
    pub gauge_form_degree: usize,
    pub parameter_components_total: usize,
    pub parameter_components_selected: Vec<usize>,
    pub target_basis_ordinals_selected: Vec<usize>,
    pub operator_columns_composed: usize,
    pub emitted_target_terms: u64,
    pub x2_functional_rank_lower_bound: usize,
    pub x2_functional_nullity_upper_bound: usize,
    pub x5_functional_rank_lower_bound: usize,
    pub x5_functional_nullity_upper_bound: usize,
    pub joint_functional_rank_lower_bound: usize,
    pub joint_functional_nullity_upper_bound: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstMomentumFxFunctionalReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub curvature_artifact_sha256: &'static str,
    pub coefficient_space: &'static str,
    pub coefficient_variables: usize,
    pub leading_kernel_variables: usize,
    pub first_momentum_correction_variables: usize,
    pub deterministic_hash_seeds: Vec<String>,
    pub buckets_per_seed: usize,
    pub bounded_channel_concurrency: usize,
    pub operator_checkpoints_per_channel: usize,
    pub checkpoint_resume_enabled: bool,
    pub channel_reports: Vec<FirstMomentumFxFunctionalChannelReport>,
    pub all_six_channels_composed_on_declared_slice: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub global_x2_rank_lower_bound: usize,
    pub global_x2_nullity_upper_bound: usize,
    pub global_x5_rank_lower_bound: usize,
    pub global_x5_nullity_upper_bound: usize,
    pub global_joint_rank_lower_bound: usize,
    pub global_joint_nullity_upper_bound: usize,
    pub global_joint_rank_exact_by_dimension_saturation: bool,
    pub surviving_leading_projection_rank_upper_bound: usize,
    pub mutation_detected: bool,
    pub partial_fx_only: bool,
    pub full_f_a_g_p_established: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FirstMomentumFxFunctionalArtifact {
    schema_version: String,
    role: String,
    curvature_artifact_sha256: String,
    coefficient_space: String,
    coefficient_variables: usize,
    leading_kernel_variables: usize,
    first_momentum_correction_variables: usize,
    deterministic_hash_seeds: Vec<String>,
    buckets_per_seed: usize,
    bounded_channel_concurrency: usize,
    operator_checkpoints_per_channel: usize,
    checkpoint_resume_enabled: bool,
    channel_reports: Vec<FirstMomentumFxFunctionalChannelReport>,
    all_six_channels_composed_on_declared_slice: bool,
    full_parameter_projection_complete: bool,
    full_target_projection_complete: bool,
    global_x2_rank_lower_bound: usize,
    global_x2_nullity_upper_bound: usize,
    global_x5_rank_lower_bound: usize,
    global_x5_nullity_upper_bound: usize,
    global_joint_rank_lower_bound: usize,
    global_joint_nullity_upper_bound: usize,
    global_joint_rank_exact_by_dimension_saturation: bool,
    surviving_leading_projection_rank_upper_bound: usize,
    mutation_detected: bool,
    partial_fx_only: bool,
    full_f_a_g_p_established: bool,
    boundary: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FirstMomentumFxPromotionManifest {
    candidate_root: String,
    candidate_sha256: BTreeMap<String, String>,
    copied_missing: usize,
    finished_utc: String,
    passed: bool,
    production_root: String,
    promotion_id: String,
    replaced_partial: usize,
    schema_version: String,
    verified_existing: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstMomentumFxDeclaredSliceStatus {
    pub fx_input_snapshot_path: &'static str,
    pub fx_input_snapshot_sha256_expected: &'static str,
    pub fx_input_snapshot_sha256_observed: Option<String>,
    pub fx_input_snapshot_schema_version: Option<String>,
    pub fx_input_snapshot_validated: bool,
    pub artifact_path: &'static str,
    pub artifact_sha256_expected: &'static str,
    pub artifact_sha256_observed: Option<String>,
    pub artifact_schema_version: Option<String>,
    pub curvature_artifact_sha256: Option<String>,
    pub functional_report_fx_input_sha256_matches_snapshot: bool,
    pub promotion_manifest_path: &'static str,
    pub promotion_manifest_sha256_expected: &'static str,
    pub promotion_manifest_sha256_observed: Option<String>,
    pub promotion_manifest_schema_version: Option<String>,
    pub promotion_id: Option<String>,
    pub promoted_checkpoint_files: Option<usize>,
    pub promotion_verified_existing: Option<usize>,
    pub promotion_copied_missing: Option<usize>,
    pub promotion_replaced_partial: Option<usize>,
    pub report_invariants_validated: bool,
    pub checkpoint_promotion_validated: bool,
    pub qualified_zero_kernel_on_declared_slice: bool,
    pub coefficient_variables: Option<usize>,
    pub global_x2_rank_lower_bound: Option<usize>,
    pub global_x2_nullity_upper_bound: Option<usize>,
    pub global_x5_rank_lower_bound: Option<usize>,
    pub global_x5_nullity_upper_bound: Option<usize>,
    pub global_joint_rank_lower_bound: Option<usize>,
    pub global_joint_nullity_upper_bound: Option<usize>,
    pub all_six_channels_composed_on_declared_slice: Option<bool>,
    pub full_parameter_projection_complete: Option<bool>,
    pub full_target_projection_complete: Option<bool>,
    pub partial_fx_only: Option<bool>,
    pub full_f_a_g_p_established: Option<bool>,
    pub validation_error: Option<String>,
    pub current_physical_envelope_schema_version: &'static str,
    pub current_physical_envelope_artifact_paths: Vec<&'static str>,
    pub current_physical_envelope_self_hash_required: bool,
    pub boundary: &'static str,
}

const FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS: [u64; 4] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
];
const FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS: usize = 16;
const FIRST_MOMENTUM_FX_OPERATOR_COLUMNS: usize = 56;
const FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES: usize = 49;
const FIRST_MOMENTUM_FX_TARGET_BASIS_ORDINAL: usize = 319;
const FIRST_MOMENTUM_FX_FUNCTIONAL_SCHEMA: &str =
    "adynkra-11d-first-momentum-partial-fx-functional-v3";
const FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA: &str =
    "adynkra-11d-first-momentum-partial-fx-checkpoint-v4";
const FIRST_MOMENTUM_FX_CURVATURE_SHA256: &str =
    "c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f";
const FIRST_MOMENTUM_FX_INPUT_SNAPSHOT_PATH: &str =
    "results/adynkra_11d_physical_curvature_fx_input_v10.json";
const FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_PATH: &str =
    "results/adynkra_11d_first_momentum_physical_fx_functional.json";
const FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_SHA256: &str =
    "5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9";
const FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_PATH: &str =
    "results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json";
const FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SCHEMA: &str =
    "adynkra-11d-fx-shared-promotion-report-v1";
const FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SHA256: &str =
    "98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4";

fn invalid_fx_artifact(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_first_momentum_fx_functional_fields(
    report: &FirstMomentumFxFunctionalArtifact,
) -> io::Result<()> {
    let expected_seeds = FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS
        .iter()
        .map(|seed| format!("{seed:016x}"))
        .collect::<Vec<_>>();
    if report.schema_version != FIRST_MOMENTUM_FX_FUNCTIONAL_SCHEMA
        || report.role
            != "exact deterministic mask-summed functional lower bound for all-six first-momentum partial F_X A G_p on a declared target/parameter slice"
        || report.curvature_artifact_sha256 != FIRST_MOMENTUM_FX_CURVATURE_SHA256
        || report.coefficient_space
            != "five exact leading F_X-kernel coordinates plus 44 recorded first-momentum correction coordinates"
        || report.coefficient_variables != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES
        || report.leading_kernel_variables != 5
        || report.first_momentum_correction_variables != 44
        || report.leading_kernel_variables + report.first_momentum_correction_variables
            != report.coefficient_variables
        || report.deterministic_hash_seeds != expected_seeds
        || report.buckets_per_seed != FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS
        || report.bounded_channel_concurrency != 1
        || report.operator_checkpoints_per_channel != FIRST_MOMENTUM_FX_OPERATOR_COLUMNS
        || !report.checkpoint_resume_enabled
    {
        return Err(invalid_fx_artifact(
            "first-momentum F_X artifact provenance or coefficient-space invariant failed",
        ));
    }

    let expected_parameter_totals = [1, 11, 55, 165, 330, 462];
    let expected_x2_ranks = [11, 46, 34, 49, 48, 45];
    let expected_x5_ranks = [11, 46, 35, 49, 48, 45];
    if report.channel_reports.len() != 6 {
        return Err(invalid_fx_artifact(
            "first-momentum F_X artifact must contain exactly six channel reports",
        ));
    }
    for (degree, channel) in report.channel_reports.iter().enumerate() {
        let x2_rank = expected_x2_ranks[degree];
        let x5_rank = expected_x5_ranks[degree];
        if channel.gauge_form_degree != degree
            || channel.parameter_components_total != expected_parameter_totals[degree]
            || channel.parameter_components_selected != [0]
            || channel.target_basis_ordinals_selected != [FIRST_MOMENTUM_FX_TARGET_BASIS_ORDINAL]
            || channel.operator_columns_composed != FIRST_MOMENTUM_FX_OPERATOR_COLUMNS
            || channel.emitted_target_terms == 0
            || channel.x2_functional_rank_lower_bound != x2_rank
            || channel.x2_functional_nullity_upper_bound
                != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES - x2_rank
            || channel.x5_functional_rank_lower_bound != x5_rank
            || channel.x5_functional_nullity_upper_bound
                != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES - x5_rank
            || channel.joint_functional_rank_lower_bound != x5_rank
            || channel.joint_functional_nullity_upper_bound
                != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES - x5_rank
        {
            return Err(invalid_fx_artifact(format!(
                "first-momentum F_X channel {degree} invariant failed"
            )));
        }
    }

    if !report.all_six_channels_composed_on_declared_slice
        || report.full_parameter_projection_complete
        || report.full_target_projection_complete
        || report.global_x2_rank_lower_bound != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES
        || report.global_x2_nullity_upper_bound != 0
        || report.global_x5_rank_lower_bound != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES
        || report.global_x5_nullity_upper_bound != 0
        || report.global_joint_rank_lower_bound != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES
        || report.global_joint_nullity_upper_bound != 0
        || !report.global_joint_rank_exact_by_dimension_saturation
        || report.surviving_leading_projection_rank_upper_bound != 0
        || !report.mutation_detected
        || !report.partial_fx_only
        || report.full_f_a_g_p_established
        || !report.boundary.contains("full F A G_p remains false")
    {
        return Err(invalid_fx_artifact(
            "first-momentum F_X rank, nullity, mutation, or completeness boundary failed",
        ));
    }
    Ok(())
}

fn validate_first_momentum_fx_functional_bytes(
    bytes: &[u8],
) -> io::Result<FirstMomentumFxFunctionalArtifact> {
    let observed = sha256_hex(bytes);
    if observed != FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_SHA256 {
        return Err(invalid_fx_artifact(format!(
            "first-momentum F_X artifact SHA-256 mismatch: expected {FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_SHA256}, observed {observed}"
        )));
    }
    let report: FirstMomentumFxFunctionalArtifact =
        serde_json::from_slice(bytes).map_err(|error| {
            invalid_fx_artifact(format!("invalid first-momentum F_X JSON: {error}"))
        })?;
    validate_first_momentum_fx_functional_fields(&report)?;
    Ok(report)
}

fn validate_generated_first_momentum_fx_functional_report(
    report: &FirstMomentumFxFunctionalReport,
) -> io::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| invalid_fx_artifact(error.to_string()))?;
    bytes.push(b'\n');
    validate_first_momentum_fx_functional_bytes(&bytes).map(|_| ())
}

fn validate_first_momentum_fx_promotion_fields(
    manifest: &FirstMomentumFxPromotionManifest,
) -> io::Result<()> {
    if manifest.schema_version != FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SCHEMA
        || !manifest.passed
        || manifest.verified_existing != 164
        || manifest.copied_missing != 172
        || manifest.replaced_partial != 0
        || manifest.verified_existing + manifest.copied_missing != 336
        || manifest.candidate_sha256.len() != 336
        || manifest.promotion_id.is_empty()
        || manifest.finished_utc.is_empty()
        || !manifest
            .production_root
            .ends_with("results/eleven_dimensional_first_momentum_fx_checkpoints")
        || manifest.candidate_root.is_empty()
    {
        return Err(invalid_fx_artifact(
            "first-momentum F_X checkpoint-promotion manifest invariant failed",
        ));
    }
    for degree in 0..6 {
        for operator in 0..FIRST_MOMENTUM_FX_OPERATOR_COLUMNS {
            let key = format!("form-{degree}/operator-{operator:02}");
            let Some(hash) = manifest.candidate_sha256.get(&key) else {
                return Err(invalid_fx_artifact(format!(
                    "checkpoint-promotion manifest is missing {key}"
                )));
            };
            if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(invalid_fx_artifact(format!(
                    "checkpoint-promotion manifest has an invalid SHA-256 for {key}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_first_momentum_fx_promotion_bytes(
    bytes: &[u8],
) -> io::Result<FirstMomentumFxPromotionManifest> {
    let observed = sha256_hex(bytes);
    if observed != FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SHA256 {
        return Err(invalid_fx_artifact(format!(
            "first-momentum F_X promotion-manifest SHA-256 mismatch: expected {FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SHA256}, observed {observed}"
        )));
    }
    let manifest: FirstMomentumFxPromotionManifest =
        serde_json::from_slice(bytes).map_err(|error| {
            invalid_fx_artifact(format!(
                "invalid first-momentum F_X promotion-manifest JSON: {error}"
            ))
        })?;
    validate_first_momentum_fx_promotion_fields(&manifest)?;
    Ok(manifest)
}

fn validate_first_momentum_fx_input_snapshot_bytes(bytes: &[u8]) -> io::Result<String> {
    let observed = sha256_hex(bytes);
    if observed != FIRST_MOMENTUM_FX_CURVATURE_SHA256 {
        return Err(invalid_fx_artifact(format!(
            "first-momentum F_X input-snapshot SHA-256 mismatch: expected {FIRST_MOMENTUM_FX_CURVATURE_SHA256}, observed {observed}"
        )));
    }
    let document: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        invalid_fx_artifact(format!(
            "invalid first-momentum F_X input-snapshot JSON: {error}"
        ))
    })?;
    let schema = document
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_fx_artifact("F_X input snapshot has no schema_version"))?;
    let boolean = |field: &str| {
        document
            .get(field)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                invalid_fx_artifact(format!("F_X input snapshot has no boolean {field}"))
            })
    };
    if schema != "adynkra-11d-physical-curvature-operator-v10"
        || !boolean("bounded_slice_passed")?
        || boolean("first_momentum_fx_all_six_channels_composed")?
        || boolean("partial_fx_a_g_p_vanishing_established")?
        || boolean("complete_f_from_h_hat_implemented")?
        || boolean("full_f_a_g_p_test_ready")?
        || boolean("covariant_off_shell_closure_established")?
    {
        return Err(invalid_fx_artifact(
            "first-momentum F_X input-snapshot schema or fail-closed boundary failed",
        ));
    }
    Ok(schema.to_string())
}

fn first_momentum_fx_declared_slice_status() -> FirstMomentumFxDeclaredSliceStatus {
    let input_snapshot_bytes = fs::read(FIRST_MOMENTUM_FX_INPUT_SNAPSHOT_PATH);
    let fx_input_snapshot_sha256_observed = input_snapshot_bytes
        .as_ref()
        .ok()
        .map(|bytes| sha256_hex(bytes));
    let input_snapshot = input_snapshot_bytes
        .map_err(|error| {
            invalid_fx_artifact(format!(
                "cannot read first-momentum F_X input snapshot: {error}"
            ))
        })
        .and_then(|bytes| validate_first_momentum_fx_input_snapshot_bytes(&bytes));

    let artifact_bytes = fs::read(FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_PATH);
    let artifact_sha256_observed = artifact_bytes.as_ref().ok().map(|bytes| sha256_hex(bytes));
    let artifact = artifact_bytes
        .map_err(|error| {
            invalid_fx_artifact(format!("cannot read first-momentum F_X artifact: {error}"))
        })
        .and_then(|bytes| validate_first_momentum_fx_functional_bytes(&bytes));

    let promotion_bytes = fs::read(FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_PATH);
    let promotion_manifest_sha256_observed =
        promotion_bytes.as_ref().ok().map(|bytes| sha256_hex(bytes));
    let promotion = promotion_bytes
        .map_err(|error| {
            invalid_fx_artifact(format!(
                "cannot read first-momentum F_X promotion manifest: {error}"
            ))
        })
        .and_then(|bytes| validate_first_momentum_fx_promotion_bytes(&bytes));

    let mut errors = Vec::new();
    if let Err(error) = &input_snapshot {
        errors.push(error.to_string());
    }
    if let Err(error) = &artifact {
        errors.push(error.to_string());
    }
    if let Err(error) = &promotion {
        errors.push(error.to_string());
    }
    let report = artifact.as_ref().ok();
    let manifest = promotion.as_ref().ok();
    let snapshot_schema = input_snapshot.as_ref().ok();
    FirstMomentumFxDeclaredSliceStatus {
        fx_input_snapshot_path: FIRST_MOMENTUM_FX_INPUT_SNAPSHOT_PATH,
        fx_input_snapshot_sha256_expected: FIRST_MOMENTUM_FX_CURVATURE_SHA256,
        fx_input_snapshot_sha256_observed,
        fx_input_snapshot_schema_version: snapshot_schema.cloned(),
        fx_input_snapshot_validated: snapshot_schema.is_some(),
        artifact_path: FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_PATH,
        artifact_sha256_expected: FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_SHA256,
        artifact_sha256_observed,
        artifact_schema_version: report.map(|value| value.schema_version.clone()),
        curvature_artifact_sha256: report.map(|value| value.curvature_artifact_sha256.clone()),
        functional_report_fx_input_sha256_matches_snapshot: report.is_some_and(|value| {
            value.curvature_artifact_sha256 == FIRST_MOMENTUM_FX_CURVATURE_SHA256
        }),
        promotion_manifest_path: FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_PATH,
        promotion_manifest_sha256_expected: FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SHA256,
        promotion_manifest_sha256_observed,
        promotion_manifest_schema_version: manifest.map(|value| value.schema_version.clone()),
        promotion_id: manifest.map(|value| value.promotion_id.clone()),
        promoted_checkpoint_files: manifest.map(|value| value.candidate_sha256.len()),
        promotion_verified_existing: manifest.map(|value| value.verified_existing),
        promotion_copied_missing: manifest.map(|value| value.copied_missing),
        promotion_replaced_partial: manifest.map(|value| value.replaced_partial),
        report_invariants_validated: report.is_some(),
        checkpoint_promotion_validated: manifest.is_some(),
        qualified_zero_kernel_on_declared_slice: snapshot_schema.is_some()
            && report.is_some()
            && manifest.is_some(),
        coefficient_variables: report.map(|value| value.coefficient_variables),
        global_x2_rank_lower_bound: report.map(|value| value.global_x2_rank_lower_bound),
        global_x2_nullity_upper_bound: report.map(|value| value.global_x2_nullity_upper_bound),
        global_x5_rank_lower_bound: report.map(|value| value.global_x5_rank_lower_bound),
        global_x5_nullity_upper_bound: report.map(|value| value.global_x5_nullity_upper_bound),
        global_joint_rank_lower_bound: report.map(|value| value.global_joint_rank_lower_bound),
        global_joint_nullity_upper_bound: report
            .map(|value| value.global_joint_nullity_upper_bound),
        all_six_channels_composed_on_declared_slice: report
            .map(|value| value.all_six_channels_composed_on_declared_slice),
        full_parameter_projection_complete: report
            .map(|value| value.full_parameter_projection_complete),
        full_target_projection_complete: report.map(|value| value.full_target_projection_complete),
        partial_fx_only: report.map(|value| value.partial_fx_only),
        full_f_a_g_p_established: report.map(|value| value.full_f_a_g_p_established),
        validation_error: (!errors.is_empty()).then(|| errors.join("; ")),
        current_physical_envelope_schema_version: "adynkra-11d-physical-curvature-operator-v10",
        current_physical_envelope_artifact_paths: vec![
            "data/eleven_dimensional_physical_curvature.json",
            "results/adynkra_11d_physical_curvature_validation.json",
        ],
        current_physical_envelope_self_hash_required: false,
        boundary: "Qualified only for the pinned immutable v10 F_X input snapshot, deterministic target/parameter slice, and promoted 336 exact operator checkpoints. Rank saturation proves zero kernel in the recorded 5+44 coefficient space. The current physical v10 envelope reports this provenance but is not required to hash itself. This does not establish complete parameter coverage, complete target coverage, J/W completion, higher-momentum descendants, or full F A G_p.",
    }
}

type FxFunctionalRows = Vec<Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>>;
#[derive(Clone, Debug)]
struct FxDerivativeTemplateEntry {
    output_sector: &'static str,
    output_coordinate: usize,
    coefficient: crate::eleven_dimensional_k_fag_solver::ExactGaussian,
}

type FxDerivativeTemplates = Vec<Vec<FxDerivativeTemplateEntry>>;

/// One exact conventional-quotient coefficient in the derivative response of
/// a raw B5 vector-spinor target coordinate.  This is a narrow internal bridge
/// for higher-momentum functional screens; it does not widen the declared
/// physical curvature envelope beyond `F_X`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExactFxDerivativeTemplateEntry {
    pub derivative_spinor_weight_index: usize,
    pub x_two_sector: bool,
    pub output_coordinate: usize,
    pub coefficient: crate::eleven_dimensional_k_fag_solver::ExactGaussian,
}
#[derive(Clone, Debug)]
struct FxProjectedDerivativeTemplate {
    x2_rows: Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>,
    x5_rows: Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>,
}
type FxProjectedTemplates = Vec<Vec<FxProjectedDerivativeTemplate>>;
type FxResponseCache = BTreeMap<(usize, usize, usize), FxProjectedTemplates>;
type FxAggregatedSourceEntry = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    crate::eleven_dimensional_k_fag_solver::ExactGaussian,
);

fn i128_gcd(mut left: i128, mut right: i128) -> i128 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn fx_target_common_denominator(target_ordinal: usize) -> i128 {
    crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states()
        .into_iter()
        .nth(target_ordinal)
        .expect("target dual basis ordinal")
        .raw_terms
        .into_iter()
        .fold(1_i128, |common, term| {
            let denominator = i128::from(term.denominator).abs();
            common
                .checked_div(i128_gcd(common, denominator))
                .and_then(|reduced| reduced.checked_mul(denominator))
                .expect("target-dual common denominator exceeds i128")
        })
}

fn scale_primitive_ratio_to_i128(
    numerator: i128,
    denominator: i64,
    common_denominator: i128,
) -> io::Result<i128> {
    let denominator = i128::from(denominator);
    if denominator <= 0 || common_denominator % denominator != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum target coefficient denominator violates the fixed target dual basis",
        ));
    }
    numerator
        .checked_mul(common_denominator / denominator)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "first-momentum target coefficient exceeds the i128 accumulator",
            )
        })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointGaussian {
    real_numerator: String,
    real_denominator: String,
    imaginary_numerator: String,
    imaginary_denominator: String,
}

impl CheckpointGaussian {
    fn from_exact(value: &crate::eleven_dimensional_k_fag_solver::ExactGaussian) -> Self {
        Self {
            real_numerator: value.real.numer().to_string(),
            real_denominator: value.real.denom().to_string(),
            imaginary_numerator: value.imaginary.numer().to_string(),
            imaginary_denominator: value.imaginary.denom().to_string(),
        }
    }

    fn into_exact(self) -> io::Result<crate::eleven_dimensional_k_fag_solver::ExactGaussian> {
        fn integer(value: &str) -> io::Result<num_bigint::BigInt> {
            value.parse::<num_bigint::BigInt>().map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid exact checkpoint integer {value:?}: {error}"),
                )
            })
        }
        let real_denominator = integer(&self.real_denominator)?;
        let imaginary_denominator = integer(&self.imaginary_denominator)?;
        if real_denominator == num_bigint::BigInt::from(0)
            || imaginary_denominator == num_bigint::BigInt::from(0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zero denominator in exact checkpoint",
            ));
        }
        Ok(crate::eleven_dimensional_k_fag_solver::ExactGaussian {
            real: num_rational::Ratio::new(integer(&self.real_numerator)?, real_denominator),
            imaginary: num_rational::Ratio::new(
                integer(&self.imaginary_numerator)?,
                imaginary_denominator,
            ),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FirstMomentumFxOperatorCheckpoint {
    schema_version: String,
    curvature_artifact_sha256: String,
    gauge_form_degree: usize,
    target_basis_ordinal: usize,
    operator_ordinal: usize,
    parameter_components_selected: Vec<usize>,
    emitted_target_terms: u64,
    source_entries_unique: u64,
    #[serde(default)]
    source_entries_processed: u64,
    #[serde(default = "fx_checkpoint_complete_default")]
    complete: bool,
    x2_rows: Vec<Vec<CheckpointGaussian>>,
    x5_rows: Vec<Vec<CheckpointGaussian>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FirstMomentumFxChannelCheckpoint {
    schema_version: String,
    curvature_artifact_sha256: String,
    gauge_form_degree: usize,
    target_basis_ordinal: usize,
    completed_operator_ordinals: Vec<usize>,
    report: FirstMomentumFxFunctionalChannelReport,
    x2_rows: Vec<Vec<CheckpointGaussian>>,
    x5_rows: Vec<Vec<CheckpointGaussian>>,
    mutation_detected: bool,
}

#[derive(Clone, Debug, Serialize)]
struct FirstMomentumFxProgressEvent<'a> {
    unix_seconds: u64,
    pid: u32,
    event: &'a str,
    gauge_form_degree: Option<usize>,
    operator_ordinal: Option<usize>,
    source_entries_processed: Option<u64>,
    reused_checkpoint: bool,
    elapsed_seconds: Option<f64>,
    response_cache_entries: usize,
}

fn fx_checkpoint_complete_default() -> bool {
    true
}

fn encode_fx_rows(rows: &FxFunctionalRows) -> Vec<Vec<CheckpointGaussian>> {
    rows.iter()
        .map(|row| row.iter().map(CheckpointGaussian::from_exact).collect())
        .collect()
}

fn decode_fx_rows(rows: Vec<Vec<CheckpointGaussian>>) -> io::Result<FxFunctionalRows> {
    if rows.len() != FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS.len() * FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS
        || rows
            .iter()
            .any(|row| row.len() != FIRST_MOMENTUM_FX_COEFFICIENT_VARIABLES)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum F_X checkpoint has the wrong functional-row shape",
        ));
    }
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(CheckpointGaussian::into_exact)
                .collect()
        })
        .collect()
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    {
        let file = File::create(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(&temporary, path)
}

fn fx_operator_checkpoint_path(
    directory: &Path,
    gauge_form_degree: usize,
    operator_ordinal: usize,
) -> PathBuf {
    directory.join(format!(
        "form-{gauge_form_degree}/operator-{operator_ordinal:02}.json"
    ))
}

fn fx_channel_checkpoint_path(directory: &Path, gauge_form_degree: usize) -> PathBuf {
    directory.join(format!("form-{gauge_form_degree}/channel.json"))
}

fn log_fx_progress(
    checkpoint_directory: &Path,
    event: &str,
    gauge_form_degree: Option<usize>,
    operator_ordinal: Option<usize>,
    source_entries_processed: Option<u64>,
    reused_checkpoint: bool,
    elapsed_seconds: Option<f64>,
    response_cache_entries: usize,
) -> io::Result<()> {
    fs::create_dir_all(checkpoint_directory)?;
    let progress = FirstMomentumFxProgressEvent {
        unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        pid: std::process::id(),
        event,
        gauge_form_degree,
        operator_ordinal,
        source_entries_processed,
        reused_checkpoint,
        elapsed_seconds,
        response_cache_entries,
    };
    let line = serde_json::to_string(&progress)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let path = checkpoint_directory.join("progress.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")?;
    file.sync_data()?;
    eprintln!("first-momentum F_X progress: {line}");
    Ok(())
}

fn fx_response_cache_limit() -> usize {
    std::env::var("ADINKRA_FX_RESPONSE_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(1, 16))
        .unwrap_or(4)
}

fn fx_checkpoint_batch_size() -> u64 {
    std::env::var("ADINKRA_FX_CHECKPOINT_BATCH")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(|value| value.clamp(1, 4_096))
        .unwrap_or(128)
}

fn splitmix64_fx(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn functional_hash_parts(
    parameter_component: usize,
    output_coordinate: usize,
    spinor_derivative_mask: u32,
    x_two_sector: bool,
    momentum_exponents: &[u16; VECTOR_DIMENSION],
) -> u64 {
    let mut value = u64::try_from(parameter_component).unwrap();
    value ^= u64::try_from(output_coordinate).unwrap().rotate_left(11);
    value ^= u64::from(spinor_derivative_mask).rotate_left(29);
    value ^= if x_two_sector {
        0x1100_0f00_d123_0002
    } else {
        0x1000_2f00_d123_0005
    };
    for (axis, exponent) in momentum_exponents.iter().enumerate() {
        value ^= (u64::from(*exponent) + 1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15_u64.rotate_left(axis as u32));
    }
    splitmix64_fx(value)
}

fn functional_hash_key(key: &crate::eleven_dimensional_k_fag_solver::CurvatureVariationKey) -> u64 {
    functional_hash_parts(
        key.parameter_component,
        key.output_coordinate,
        key.spinor_derivative_mask,
        key.output_sector == "X2_11000",
        &key.momentum_monomial.exponents,
    )
}

fn fx_functional_specs() -> Vec<crate::eleven_dimensional_k_fag_solver::KCoefficientSpec> {
    use crate::eleven_dimensional_k_fag_solver::KCoefficientSpec;
    let mut specs = (0..5)
        .map(|ordinal| KCoefficientSpec {
            ordinal,
            label: format!("leading-FX-kernel-{ordinal}"),
            operator_kind: "leading-kernel".to_string(),
            spinor_derivative_order_before_gauge_map: 16,
            momentum_degree_before_gauge_map: 0,
            lower_symbol_status: "restricted to the exact five-dimensional leading F_X kernel"
                .to_string(),
        })
        .collect::<Vec<_>>();
    specs.extend((0..44).map(|correction| {
        let ordinal = 5 + correction;
        KCoefficientSpec {
            ordinal,
            label: format!("first-pD14-{correction:02}"),
            operator_kind: "first-momentum".to_string(),
            spinor_derivative_order_before_gauge_map: 14,
            momentum_degree_before_gauge_map: 1,
            lower_symbol_status: "recorded first correction only".to_string(),
        }
    }));
    specs
}

fn empty_fx_functional_rows() -> Vec<Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>> {
    vec![
        vec![crate::eleven_dimensional_k_fag_solver::ExactGaussian::zero(); 49];
        FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS.len() * FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS
    ]
}

fn add_scaled_solver_gaussian(
    target: &mut crate::eleven_dimensional_k_fag_solver::ExactGaussian,
    value: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
    scale: i64,
) {
    if scale == 0 {
        return;
    }
    let scale = num_rational::Ratio::from_integer(num_bigint::BigInt::from(scale));
    target.real += value.real.clone() * scale.clone();
    target.imaginary += value.imaginary.clone() * scale;
}

fn multiply_solver_gaussian(
    left: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
    right: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
) -> crate::eleven_dimensional_k_fag_solver::ExactGaussian {
    crate::eleven_dimensional_k_fag_solver::ExactGaussian {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn exact_qi_to_solver_gaussian(
    value: &ExactQi,
) -> crate::eleven_dimensional_k_fag_solver::ExactGaussian {
    crate::eleven_dimensional_k_fag_solver::ExactGaussian {
        real: num_rational::Ratio::new(
            num_bigint::BigInt::from(*value.real.numer()),
            num_bigint::BigInt::from(*value.real.denom()),
        ),
        imaginary: num_rational::Ratio::new(
            num_bigint::BigInt::from(*value.imaginary.numer()),
            num_bigint::BigInt::from(*value.imaginary.denom()),
        ),
    }
}

/// Precompute the expensive physical `F_X` image separately for every
/// derivative spinor. The exterior mask changes only the admissible
/// derivative set, wedge sign, and output mask, so it must not force a fresh
/// gamma/hook projection for every source-stream entry.
fn build_fx_derivative_templates(
    vector_weight: usize,
    target_spinor: usize,
) -> Result<FxDerivativeTemplates, String> {
    if vector_weight >= VECTOR_DIMENSION || target_spinor >= SPINOR_DIMENSION {
        return Err("B5 target coordinate is out of range".to_string());
    }
    let join = cached_b5_majorana_target_join();
    let mut templates = Vec::with_capacity(SPINOR_DIMENSION);
    for derivative_weight in 0..SPINOR_DIMENSION {
        let mut d_h = BTreeMap::<usize, ExactQi>::new();
        for derivative_majorana in 0..SPINOR_DIMENSION {
            let derivative_factor =
                &join.spinor_to_majorana[derivative_majorana][derivative_weight];
            if derivative_factor.re == r(0) && derivative_factor.im == r(0) {
                continue;
            }
            for h_majorana in 0..SPINOR_DIMENSION {
                let h_factor = &join.spinor_to_majorana[h_majorana][target_spinor];
                if h_factor.re == r(0) && h_factor.im == r(0) {
                    continue;
                }
                for output_vector in 0..VECTOR_DIMENSION {
                    let vector_factor = &join.upper_vector_to_lorentz[output_vector][vector_weight];
                    if vector_factor.re == r(0) && vector_factor.im == r(0) {
                        continue;
                    }
                    let factor =
                        derivative_factor.clone() * h_factor.clone() * vector_factor.clone();
                    add_sparse(
                        &mut d_h,
                        dh_index(derivative_majorana, h_majorana, output_vector),
                        ExactQi {
                            real: factor.re,
                            imaginary: factor.im,
                        },
                    );
                }
            }
        }
        let image = apply_leading_physical_x(&d_h);
        let mut response = Vec::with_capacity(image.x_two_11000.len() + image.x_five_10002.len());
        response.extend(
            image
                .x_two_11000
                .into_iter()
                .map(|(coordinate, coefficient)| FxDerivativeTemplateEntry {
                    output_sector: "X2_11000",
                    output_coordinate: coordinate,
                    coefficient: exact_qi_to_solver_gaussian(&coefficient),
                }),
        );
        response.extend(
            image
                .x_five_10002
                .into_iter()
                .map(|(coordinate, coefficient)| FxDerivativeTemplateEntry {
                    output_sector: "X5_10002",
                    output_coordinate: coordinate,
                    coefficient: exact_qi_to_solver_gaussian(&coefficient),
                }),
        );
        templates.push(response);
    }
    Ok(templates)
}

/// Visit the exact `F_X` derivative templates for one raw B5 target
/// vector-spinor coordinate.  All 32 derivative-spinor responses are built
/// once and emitted with their derivative index so callers can cache the
/// response without reconstructing gamma and hook projections per source
/// term.
pub(crate) fn visit_exact_fx_derivative_templates<F>(
    vector_weight: usize,
    target_spinor: usize,
    mut visit: F,
) -> Result<(), String>
where
    F: FnMut(ExactFxDerivativeTemplateEntry),
{
    for (derivative_spinor_weight_index, entries) in
        build_fx_derivative_templates(vector_weight, target_spinor)?
            .into_iter()
            .enumerate()
    {
        for entry in entries {
            visit(ExactFxDerivativeTemplateEntry {
                derivative_spinor_weight_index,
                x_two_sector: entry.output_sector == "X2_11000",
                output_coordinate: entry.output_coordinate,
                coefficient: entry.coefficient,
            });
        }
    }
    Ok(())
}

fn build_fx_projected_templates(
    vector_weight: usize,
    target_spinor: usize,
) -> Result<FxProjectedTemplates, String> {
    let templates = build_fx_derivative_templates(vector_weight, target_spinor)?;
    let row_count = FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS.len() * FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS;
    let mut projected = Vec::with_capacity(SPINOR_DIMENSION);
    for derivative_templates in templates {
        let mut by_momentum = Vec::with_capacity(VECTOR_DIMENSION);
        for momentum_axis in 0..VECTOR_DIMENSION {
            let momentum =
                crate::eleven_dimensional_k_fag_solver::MomentumMonomial::variable(momentum_axis);
            let mut x2_rows =
                vec![crate::eleven_dimensional_k_fag_solver::ExactGaussian::zero(); row_count];
            let mut x5_rows = x2_rows.clone();
            for template in &derivative_templates {
                // This functional slice deliberately sums over exterior masks
                // before applying F_X. Coordinate, sector, and momentum retain
                // independent deterministic hash projections.
                let base_hash = functional_hash_parts(
                    0,
                    template.output_coordinate,
                    0,
                    template.output_sector == "X2_11000",
                    &momentum.exponents,
                );
                let rows = if template.output_sector == "X2_11000" {
                    &mut x2_rows
                } else {
                    &mut x5_rows
                };
                for (seed_index, seed) in FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS.iter().enumerate() {
                    let hash = splitmix64_fx(base_hash ^ seed);
                    let bucket = (hash as usize) % FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS;
                    let row = seed_index * FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS + bucket;
                    let sign = if hash >> 63 == 0 { 1 } else { -1 };
                    add_scaled_solver_gaussian(
                        rows.get_mut(row).unwrap(),
                        &template.coefficient,
                        sign,
                    );
                }
            }
            by_momentum.push(FxProjectedDerivativeTemplate { x2_rows, x5_rows });
        }
        projected.push(by_momentum);
    }
    Ok(projected)
}

fn add_fx_projected_rows(
    rows: &mut FxFunctionalRows,
    operator_ordinal: usize,
    projection: &[crate::eleven_dimensional_k_fag_solver::ExactGaussian],
    source_coefficient: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
) {
    const LEADING_KERNEL: [[i64; 12]; 5] = [
        [-18, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [30, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [54, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ];
    assert_eq!(rows.len(), projection.len());
    for (row, projected_value) in rows.iter_mut().zip(projection) {
        if projected_value.is_zero() {
            continue;
        }
        let value = multiply_solver_gaussian(projected_value, source_coefficient);
        if operator_ordinal < 12 {
            for variable in 0..5 {
                add_scaled_solver_gaussian(
                    &mut row[variable],
                    &value,
                    LEADING_KERNEL[variable][operator_ordinal],
                );
            }
        } else {
            let target = &mut row[5 + operator_ordinal - 12];
            target.real += value.real;
            target.imaginary += value.imaginary;
        }
    }
}

fn add_fx_functional_value(
    rows: &mut [Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>],
    operator_ordinal: usize,
    key: &crate::eleven_dimensional_k_fag_solver::CurvatureVariationKey,
    value: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
) {
    add_fx_functional_hashed_value(rows, operator_ordinal, functional_hash_key(key), value);
}

fn add_fx_functional_hashed_value(
    rows: &mut [Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>],
    operator_ordinal: usize,
    base_hash: u64,
    value: &crate::eleven_dimensional_k_fag_solver::ExactGaussian,
) {
    const LEADING_KERNEL: [[i64; 12]; 5] = [
        [-18, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [30, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [54, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ];
    for (seed_index, seed) in FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS.iter().enumerate() {
        let hash = splitmix64_fx(base_hash ^ seed);
        let bucket = (hash as usize) % FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS;
        let row = seed_index * FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS + bucket;
        let sign = if hash >> 63 == 0 { 1 } else { -1 };
        if operator_ordinal < 12 {
            for variable in 0..5 {
                add_scaled_solver_gaussian(
                    &mut rows[row][variable],
                    value,
                    sign * LEADING_KERNEL[variable][operator_ordinal],
                );
            }
        } else {
            add_scaled_solver_gaussian(&mut rows[row][5 + operator_ordinal - 12], value, sign);
        }
    }
}

fn solve_fx_functional_rows(
    rows: &[Vec<crate::eleven_dimensional_k_fag_solver::ExactGaussian>],
    gauge_form_degree: usize,
    sector: &str,
) -> crate::eleven_dimensional_k_fag_solver::ExactCoefficientSolution {
    use crate::eleven_dimensional_k_fag_solver::{
        ExactPolynomialSystem, MomentumMonomial, PolynomialConstraintKey,
    };
    let mut system = ExactPolynomialSystem::new(fx_functional_specs(), true);
    for (row, coefficients) in rows.iter().enumerate() {
        let key = PolynomialConstraintKey {
            gauge_form_degree,
            parameter_component: row,
            output_sector: sector.to_string(),
            output_coordinate: row,
            spinor_derivative_mask: 0,
            spinor_derivative_order: 16,
            momentum_monomial: MomentumMonomial::constant(),
        };
        for (variable, coefficient) in coefficients.iter().enumerate() {
            system.add_coefficient(key.clone(), variable, coefficient.clone());
        }
    }
    system.solve()
}

/// Execute an exact, deterministic functional slice of the all-six
/// first-momentum `F_X A G_p` composition.  One target highest-weight state
/// and one gauge-parameter component per channel are selected deliberately.
/// Any rank obtained is therefore a rigorous lower bound on the full
/// composition rank, while a surviving functional kernel is not accepted as
/// a physical kernel.
fn add_fx_functional_rows(target: &mut FxFunctionalRows, source: &FxFunctionalRows) {
    assert_eq!(target.len(), source.len());
    for (target_row, source_row) in target.iter_mut().zip(source) {
        assert_eq!(target_row.len(), source_row.len());
        for (target_value, source_value) in target_row.iter_mut().zip(source_row) {
            target_value.real += source_value.real.clone();
            target_value.imaginary += source_value.imaginary.clone();
        }
    }
}

fn read_fx_json<T>(path: &Path) -> io::Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_reader(BufReader::new(File::open(path)?))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn quarantine_invalid_fx_checkpoint(path: &Path, error: &io::Error) -> io::Result<()> {
    let rejected = path.with_extension(format!("json.rejected.{}", std::process::id()));
    eprintln!(
        "rejecting invalid first-momentum F_X checkpoint {}: {error}",
        path.display()
    );
    fs::rename(path, rejected)
}

fn load_fx_operator_checkpoint(
    path: &Path,
    gauge_form_degree: usize,
    target_ordinal: usize,
    operator_ordinal: usize,
) -> io::Result<Option<(FxFunctionalRows, FxFunctionalRows, u64, u64, u64, bool)>> {
    if !path.exists() {
        return Ok(None);
    }
    let checkpoint: FirstMomentumFxOperatorCheckpoint = read_fx_json(path)?;
    if checkpoint.schema_version != FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA
        || checkpoint.curvature_artifact_sha256 != FIRST_MOMENTUM_FX_CURVATURE_SHA256
        || checkpoint.gauge_form_degree != gauge_form_degree
        || checkpoint.target_basis_ordinal != target_ordinal
        || checkpoint.operator_ordinal != operator_ordinal
        || checkpoint.parameter_components_selected != [0]
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum F_X operator checkpoint metadata mismatch",
        ));
    }
    if checkpoint.source_entries_unique > checkpoint.emitted_target_terms
        || checkpoint.source_entries_processed > checkpoint.source_entries_unique
        || (checkpoint.complete
            && checkpoint.source_entries_processed != 0
            && checkpoint.source_entries_processed != checkpoint.source_entries_unique)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum F_X operator checkpoint progress is inconsistent",
        ));
    }
    Ok(Some((
        decode_fx_rows(checkpoint.x2_rows)?,
        decode_fx_rows(checkpoint.x5_rows)?,
        checkpoint.emitted_target_terms,
        checkpoint.source_entries_unique,
        checkpoint.source_entries_processed,
        checkpoint.complete,
    )))
}

fn load_fx_channel_checkpoint(
    path: &Path,
    gauge_form_degree: usize,
    target_ordinal: usize,
) -> io::Result<
    Option<(
        FirstMomentumFxFunctionalChannelReport,
        FxFunctionalRows,
        FxFunctionalRows,
        bool,
    )>,
> {
    if !path.exists() {
        return Ok(None);
    }
    let checkpoint: FirstMomentumFxChannelCheckpoint = read_fx_json(path)?;
    if checkpoint.schema_version != FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA
        || checkpoint.curvature_artifact_sha256 != FIRST_MOMENTUM_FX_CURVATURE_SHA256
        || checkpoint.gauge_form_degree != gauge_form_degree
        || checkpoint.target_basis_ordinal != target_ordinal
        || checkpoint.completed_operator_ordinals
            != (0..FIRST_MOMENTUM_FX_OPERATOR_COLUMNS).collect::<Vec<_>>()
        || checkpoint.report.gauge_form_degree != gauge_form_degree
        || checkpoint.report.target_basis_ordinals_selected != [target_ordinal]
        || checkpoint.report.operator_columns_composed != FIRST_MOMENTUM_FX_OPERATOR_COLUMNS
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum F_X channel checkpoint metadata mismatch",
        ));
    }
    let x2_rows = decode_fx_rows(checkpoint.x2_rows)?;
    let x5_rows = decode_fx_rows(checkpoint.x5_rows)?;
    if checkpoint.mutation_detected != (x2_rows != x5_rows) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum F_X channel checkpoint mutation flag mismatch",
        ));
    }
    Ok(Some((
        checkpoint.report,
        x2_rows,
        x5_rows,
        checkpoint.mutation_detected,
    )))
}

fn build_first_momentum_fx_operator_checkpoint(
    gauge_form_degree: usize,
    target_ordinal: usize,
    operator_ordinal: usize,
    response_cache: &mut FxResponseCache,
    checkpoint_path: &Path,
    checkpoint_directory: &Path,
    resumed: Option<(FxFunctionalRows, FxFunctionalRows, u64, u64, u64, bool)>,
) -> io::Result<(
    FirstMomentumFxOperatorCheckpoint,
    FxFunctionalRows,
    FxFunctionalRows,
)> {
    let (
        mut x2_rows,
        mut x5_rows,
        resumed_emitted,
        resumed_unique,
        mut processed,
        resumed_complete,
    ) = resumed.unwrap_or_else(|| {
        (
            empty_fx_functional_rows(),
            empty_fx_functional_rows(),
            0,
            0,
            0,
            false,
        )
    });
    if resumed_complete {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "complete operator checkpoint was sent through the resume path",
        ));
    }
    let dense_entry_count =
        VECTOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION * SPINOR_DIMENSION;
    let target_common_denominator = fx_target_common_denominator(target_ordinal);
    let mut source_projection = vec![(0_i128, 0_i128); dense_entry_count];
    let (_, parameter_basis, _, _, emitted_target_terms) =
        crate::eleven_dimensional_level16_couplings::visit_target_resolved_first_momentum_gauge_composition_primitive_terms(
            gauge_form_degree,
            operator_ordinal,
            Some(&[0]),
            Some(&[target_ordinal]),
            |entry| {
                assert_eq!(entry.target_basis_ordinal, target_ordinal);
                assert_eq!(entry.parameter_component_index, 0);
                let momentum_axis = entry
                    .momentum_vector_weight_index
                    .expect("first-momentum source term");
                let scaled_real = scale_primitive_ratio_to_i128(
                    entry.real_numerator,
                    entry.denominator,
                    target_common_denominator,
                )?;
                let scaled_imaginary = scale_primitive_ratio_to_i128(
                    entry.imaginary_numerator,
                    entry.denominator,
                    target_common_denominator,
                )?;
                for derivative_weight in 0..SPINOR_DIMENSION {
                    if entry.exterior_mask & (1_u32 << derivative_weight) != 0 {
                        continue;
                    }
                    let dense_index = (((entry.target_vector_weight_index * SPINOR_DIMENSION
                        + entry.target_spinor_weight_index)
                        * VECTOR_DIMENSION
                        + momentum_axis)
                        * SPINOR_DIMENSION)
                        + derivative_weight;
                    let value = &mut source_projection[dense_index];
                    if derivative_wedge_sign(entry.exterior_mask, derivative_weight) > 0 {
                        value.0 = value.0.checked_add(scaled_real).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "first-momentum projected real source exceeds i128",
                            )
                        })?;
                        value.1 = value.1.checked_add(scaled_imaginary).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "first-momentum projected imaginary source exceeds i128",
                            )
                        })?;
                    } else {
                        value.0 = value.0.checked_sub(scaled_real).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "first-momentum projected real source exceeds i128",
                            )
                        })?;
                        value.1 = value.1.checked_sub(scaled_imaginary).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "first-momentum projected imaginary source exceeds i128",
                            )
                        })?;
                    }
                }
                Ok(())
            },
        )?;
    assert_eq!(
        parameter_basis.len(),
        [1, 11, 55, 165, 330, 462][gauge_form_degree]
    );
    let source_entries = source_projection
        .into_iter()
        .enumerate()
        .filter_map(|(dense_index, (real, imaginary))| {
            if real == 0 && imaginary == 0 {
                return None;
            }
            let derivative_weight = dense_index % SPINOR_DIMENSION;
            let rest = dense_index / SPINOR_DIMENSION;
            let momentum_axis = rest % VECTOR_DIMENSION;
            let rest = rest / VECTOR_DIMENSION;
            let target_spinor_weight_index = rest % SPINOR_DIMENSION;
            let target_vector_weight_index = rest / SPINOR_DIMENSION;
            Some((
                target_ordinal,
                target_vector_weight_index,
                target_spinor_weight_index,
                0,
                momentum_axis,
                derivative_weight,
                crate::eleven_dimensional_k_fag_solver::ExactGaussian {
                    real: num_rational::Ratio::new(
                        num_bigint::BigInt::from(real),
                        num_bigint::BigInt::from(target_common_denominator),
                    ),
                    imaginary: num_rational::Ratio::new(
                        num_bigint::BigInt::from(imaginary),
                        num_bigint::BigInt::from(target_common_denominator),
                    ),
                },
            ))
        })
        .collect::<Vec<FxAggregatedSourceEntry>>();
    let source_entries_unique = u64::try_from(source_entries.len()).unwrap();
    if (resumed_emitted != 0 && resumed_emitted != emitted_target_terms)
        || (resumed_unique != 0 && resumed_unique != source_entries_unique)
        || processed > source_entries_unique
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "first-momentum F_X aggregated source stream changed across an operator resume",
        ));
    }
    let cache_limit = fx_response_cache_limit();
    let checkpoint_batch = fx_checkpoint_batch_size();
    for (
        target_basis_ordinal,
        target_vector_weight_index,
        target_spinor_weight_index,
        parameter_component_index,
        momentum_axis,
        derivative_weight,
        coefficient,
    ) in source_entries.into_iter().skip(processed as usize)
    {
        debug_assert_eq!(parameter_component_index, 0);
        let cache_key = (
            target_basis_ordinal,
            target_vector_weight_index,
            target_spinor_weight_index,
        );
        if !response_cache.contains_key(&cache_key) {
            if response_cache.len() >= cache_limit {
                response_cache.clear();
            }
            let response = build_fx_projected_templates(
                target_vector_weight_index,
                target_spinor_weight_index,
            )
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;
            response_cache.insert(cache_key, response);
        }
        let projected = response_cache
            .get(&cache_key)
            .expect("projected derivative F_X templates cached before use");
        let projected = &projected[derivative_weight][momentum_axis];
        add_fx_projected_rows(
            &mut x2_rows,
            operator_ordinal,
            &projected.x2_rows,
            &coefficient,
        );
        add_fx_projected_rows(
            &mut x5_rows,
            operator_ordinal,
            &projected.x5_rows,
            &coefficient,
        );
        processed += 1;
        if processed < source_entries_unique && processed % checkpoint_batch == 0 {
            let partial = FirstMomentumFxOperatorCheckpoint {
                schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
                curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
                gauge_form_degree,
                target_basis_ordinal: target_ordinal,
                operator_ordinal,
                parameter_components_selected: vec![0],
                emitted_target_terms,
                source_entries_unique,
                source_entries_processed: processed,
                complete: false,
                x2_rows: encode_fx_rows(&x2_rows),
                x5_rows: encode_fx_rows(&x5_rows),
            };
            atomic_json(checkpoint_path, &partial)?;
            log_fx_progress(
                checkpoint_directory,
                "operator-checkpoint",
                Some(gauge_form_degree),
                Some(operator_ordinal),
                Some(processed),
                false,
                None,
                response_cache.len(),
            )?;
        }
    }
    let checkpoint = FirstMomentumFxOperatorCheckpoint {
        schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
        curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
        gauge_form_degree,
        target_basis_ordinal: target_ordinal,
        operator_ordinal,
        parameter_components_selected: vec![0],
        emitted_target_terms,
        source_entries_unique,
        source_entries_processed: source_entries_unique,
        complete: true,
        x2_rows: encode_fx_rows(&x2_rows),
        x5_rows: encode_fx_rows(&x5_rows),
    };
    Ok((checkpoint, x2_rows, x5_rows))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FirstMomentumFxSharedOperatorBatchReport {
    pub schema_version: String,
    pub curvature_artifact_sha256: String,
    pub operator_ordinal: usize,
    pub operator_label: String,
    pub operator_kind: String,
    pub selected_gauge_form_degrees: Vec<usize>,
    pub completed_gauge_form_degrees: Vec<usize>,
    pub reused_batch_checkpoints: Vec<usize>,
    pub exact_reference_matches: Vec<usize>,
    pub reference_checkpoints_absent: Vec<usize>,
    pub emitted_target_terms_by_degree: [u64; 6],
    pub unique_aggregated_source_entries_by_degree: [u64; 6],
    pub shared_state_accounting:
        Option<crate::eleven_dimensional_level16_couplings::SharedFirstMomentumStateAccounting>,
    pub elapsed_seconds: f64,
    pub checkpoint_schema_compatible: bool,
    pub all_available_reference_checkpoints_match: bool,
    pub reference_coverage_complete: bool,
    pub passed: bool,
    pub boundary: String,
}

fn aggregate_shared_fx_source_entry(
    source_projection: &mut [(i128, i128)],
    target_ordinal: usize,
    target_common_denominator: i128,
    entry: crate::eleven_dimensional_level16_couplings::TargetResolvedPrimitiveGaugeCompositionEntry,
) -> io::Result<()> {
    if entry.target_basis_ordinal != target_ordinal || entry.parameter_component_index != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shared first-momentum stream escaped its declared target/parameter slice",
        ));
    }
    let momentum_axis = entry
        .momentum_vector_weight_index
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing momentum axis"))?;
    let scaled_real = scale_primitive_ratio_to_i128(
        entry.real_numerator,
        entry.denominator,
        target_common_denominator,
    )?;
    let scaled_imaginary = scale_primitive_ratio_to_i128(
        entry.imaginary_numerator,
        entry.denominator,
        target_common_denominator,
    )?;
    for derivative_weight in 0..SPINOR_DIMENSION {
        if entry.exterior_mask & (1_u32 << derivative_weight) != 0 {
            continue;
        }
        let dense_index = (((entry.target_vector_weight_index * SPINOR_DIMENSION
            + entry.target_spinor_weight_index)
            * VECTOR_DIMENSION
            + momentum_axis)
            * SPINOR_DIMENSION)
            + derivative_weight;
        let value = &mut source_projection[dense_index];
        let sign = i128::from(derivative_wedge_sign(
            entry.exterior_mask,
            derivative_weight,
        ));
        value.0 = value.0.checked_add(sign * scaled_real).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "shared F_X real source exceeds i128",
            )
        })?;
        value.1 = value
            .1
            .checked_add(sign * scaled_imaginary)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "shared F_X imaginary source exceeds i128",
                )
            })?;
    }
    Ok(())
}

fn finish_shared_fx_operator_degree(
    gauge_form_degree: usize,
    target_ordinal: usize,
    operator_ordinal: usize,
    emitted_target_terms: u64,
    target_common_denominator: i128,
    source_projection: Vec<(i128, i128)>,
    response_cache: &mut FxResponseCache,
) -> io::Result<(
    FirstMomentumFxOperatorCheckpoint,
    FxFunctionalRows,
    FxFunctionalRows,
)> {
    let mut x2_rows = empty_fx_functional_rows();
    let mut x5_rows = empty_fx_functional_rows();
    let source_entries = source_projection
        .into_iter()
        .enumerate()
        .filter_map(|(dense_index, (real, imaginary))| {
            if real == 0 && imaginary == 0 {
                return None;
            }
            let derivative_weight = dense_index % SPINOR_DIMENSION;
            let rest = dense_index / SPINOR_DIMENSION;
            let momentum_axis = rest % VECTOR_DIMENSION;
            let rest = rest / VECTOR_DIMENSION;
            let target_spinor_weight_index = rest % SPINOR_DIMENSION;
            let target_vector_weight_index = rest / SPINOR_DIMENSION;
            Some((
                target_vector_weight_index,
                target_spinor_weight_index,
                momentum_axis,
                derivative_weight,
                crate::eleven_dimensional_k_fag_solver::ExactGaussian {
                    real: num_rational::Ratio::new(
                        num_bigint::BigInt::from(real),
                        num_bigint::BigInt::from(target_common_denominator),
                    ),
                    imaginary: num_rational::Ratio::new(
                        num_bigint::BigInt::from(imaginary),
                        num_bigint::BigInt::from(target_common_denominator),
                    ),
                },
            ))
        })
        .collect::<Vec<_>>();
    let source_entries_unique = u64::try_from(source_entries.len()).unwrap();
    let cache_limit = fx_response_cache_limit();
    for (
        target_vector_weight_index,
        target_spinor_weight_index,
        momentum_axis,
        derivative_weight,
        coefficient,
    ) in source_entries
    {
        let cache_key = (
            target_ordinal,
            target_vector_weight_index,
            target_spinor_weight_index,
        );
        if !response_cache.contains_key(&cache_key) {
            if response_cache.len() >= cache_limit {
                response_cache.clear();
            }
            response_cache.insert(
                cache_key,
                build_fx_projected_templates(
                    target_vector_weight_index,
                    target_spinor_weight_index,
                )
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?,
            );
        }
        let projected = &response_cache[&cache_key][derivative_weight][momentum_axis];
        add_fx_projected_rows(
            &mut x2_rows,
            operator_ordinal,
            &projected.x2_rows,
            &coefficient,
        );
        add_fx_projected_rows(
            &mut x5_rows,
            operator_ordinal,
            &projected.x5_rows,
            &coefficient,
        );
    }
    let checkpoint = FirstMomentumFxOperatorCheckpoint {
        schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
        curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
        gauge_form_degree,
        target_basis_ordinal: target_ordinal,
        operator_ordinal,
        parameter_components_selected: vec![0],
        emitted_target_terms,
        source_entries_unique,
        source_entries_processed: source_entries_unique,
        complete: true,
        x2_rows: encode_fx_rows(&x2_rows),
        x5_rows: encode_fx_rows(&x5_rows),
    };
    Ok((checkpoint, x2_rows, x5_rows))
}

/// Build one operator column for several gauge degrees while reusing its
/// materialized level-14 or level-16 coupled state. Output checkpoints use the
/// established v4 schema, so a later channel run can resume from them.
pub fn build_first_momentum_fx_shared_operator_batch_in(
    operator_ordinal: usize,
    selected_gauge_form_degrees: &[usize],
    checkpoint_directory: &Path,
    reference_checkpoint_directory: Option<&Path>,
    report_path: &Path,
) -> io::Result<FirstMomentumFxSharedOperatorBatchReport> {
    if selected_gauge_form_degrees.is_empty()
        || selected_gauge_form_degrees.iter().any(|degree| *degree > 5)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shared F_X batch requires distinct gauge degrees in 0..=5",
        ));
    }
    let selected = selected_gauge_form_degrees
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if selected.len() != selected_gauge_form_degrees.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shared F_X batch gauge degrees must be distinct",
        ));
    }
    let spec = crate::eleven_dimensional_level16_couplings::joint_column_specs()
        .get(operator_ordinal)
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "operator ordinal outside 0..56",
            )
        })?;
    let target_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
        .into_iter()
        .find(|state| state.pbw_word_simple_roots.is_empty())
        .expect("target highest-weight state")
        .ordinal;
    let started = Instant::now();
    let mut reused_batch_checkpoints = Vec::new();
    let mut missing = Vec::new();
    let mut emitted_target_terms_by_degree = [0_u64; 6];
    let mut unique_aggregated_source_entries_by_degree = [0_u64; 6];
    for &degree in &selected {
        let path = fx_operator_checkpoint_path(checkpoint_directory, degree, operator_ordinal);
        match load_fx_operator_checkpoint(&path, degree, target_ordinal, operator_ordinal) {
            Ok(Some((_, _, emitted, unique, _, true))) => {
                emitted_target_terms_by_degree[degree] = emitted;
                unique_aggregated_source_entries_by_degree[degree] = unique;
                reused_batch_checkpoints.push(degree);
            }
            Ok(_) => missing.push(degree),
            Err(error) => {
                quarantine_invalid_fx_checkpoint(&path, &error)?;
                missing.push(degree);
            }
        }
    }
    let dense_entry_count =
        VECTOR_DIMENSION * SPINOR_DIMENSION * VECTOR_DIMENSION * SPINOR_DIMENSION;
    let target_common_denominator = fx_target_common_denominator(target_ordinal);
    let mut source_projections = (0..6)
        .map(|_| Vec::<(i128, i128)>::new())
        .collect::<Vec<_>>();
    for &degree in &missing {
        source_projections[degree] = vec![(0_i128, 0_i128); dense_entry_count];
    }
    let mut shared_state_accounting = None;
    if !missing.is_empty() {
        let (_, parameter_bases, _, _, emitted, accounting) =
            crate::eleven_dimensional_level16_couplings::visit_target_resolved_first_momentum_gauge_composition_primitive_terms_shared(
                operator_ordinal,
                Some(&missing),
                Some(&[0]),
                Some(&[target_ordinal]),
                |degree, entry| {
                    aggregate_shared_fx_source_entry(
                        &mut source_projections[degree],
                        target_ordinal,
                        target_common_denominator,
                        entry,
                    )
                },
            )?;
        for &degree in &missing {
            if parameter_bases[degree].len() != [1, 11, 55, 165, 330, 462][degree] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "shared F_X gauge basis dimension mismatch",
                ));
            }
        }
        for &degree in &missing {
            emitted_target_terms_by_degree[degree] = emitted[degree];
        }
        shared_state_accounting = Some(accounting);
    }
    let mut response_cache = FxResponseCache::new();
    for degree in missing {
        let projection = std::mem::take(&mut source_projections[degree]);
        let (checkpoint, _, _) = finish_shared_fx_operator_degree(
            degree,
            target_ordinal,
            operator_ordinal,
            emitted_target_terms_by_degree[degree],
            target_common_denominator,
            projection,
            &mut response_cache,
        )?;
        unique_aggregated_source_entries_by_degree[degree] = checkpoint.source_entries_unique;
        atomic_json(
            &fx_operator_checkpoint_path(checkpoint_directory, degree, operator_ordinal),
            &checkpoint,
        )?;
    }
    let mut exact_reference_matches = Vec::new();
    let mut reference_checkpoints_absent = Vec::new();
    if let Some(reference_directory) = reference_checkpoint_directory {
        for &degree in &selected {
            let built = load_fx_operator_checkpoint(
                &fx_operator_checkpoint_path(checkpoint_directory, degree, operator_ordinal),
                degree,
                target_ordinal,
                operator_ordinal,
            )?;
            let reference_path =
                fx_operator_checkpoint_path(reference_directory, degree, operator_ordinal);
            let reference = if reference_path.exists() {
                load_fx_operator_checkpoint(
                    &reference_path,
                    degree,
                    target_ordinal,
                    operator_ordinal,
                )?
            } else {
                None
            };
            match (built, reference) {
                (Some(left), Some(right)) if left == right => exact_reference_matches.push(degree),
                (_, None) => reference_checkpoints_absent.push(degree),
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "shared F_X output disagrees with reference checkpoint for form {degree}"
                        ),
                    ));
                }
            }
        }
    }
    let completed_gauge_form_degrees = selected.clone();
    let all_available_reference_checkpoints_match = reference_checkpoint_directory.is_none()
        || exact_reference_matches.len() + reference_checkpoints_absent.len() == selected.len();
    let reference_coverage_complete = reference_checkpoint_directory.is_some()
        && reference_checkpoints_absent.is_empty()
        && exact_reference_matches.len() == selected.len();
    let report = FirstMomentumFxSharedOperatorBatchReport {
        schema_version: "adynkra-11d-first-momentum-fx-shared-operator-batch-v1".to_string(),
        curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
        operator_ordinal,
        operator_label: spec.label,
        operator_kind: spec.kind,
        selected_gauge_form_degrees: selected,
        completed_gauge_form_degrees,
        reused_batch_checkpoints,
        exact_reference_matches,
        reference_checkpoints_absent,
        emitted_target_terms_by_degree,
        unique_aggregated_source_entries_by_degree,
        shared_state_accounting,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        checkpoint_schema_compatible: true,
        all_available_reference_checkpoints_match,
        reference_coverage_complete,
        passed: all_available_reference_checkpoints_match,
        boundary: "The batch reuses one exact coupled operator state across selected gauge degrees and emits production-compatible per-form operator checkpoints. Payload accounting excludes allocator metadata and is not process RSS. All available reference checkpoints must match exactly. Reference coverage is complete only when every selected degree had a reference at build time.".to_string(),
    };
    atomic_json(report_path, &report)?;
    Ok(report)
}

fn build_first_momentum_fx_functional_channel(
    gauge_form_degree: usize,
    target_ordinal: usize,
    checkpoint_directory: &Path,
) -> io::Result<(
    FirstMomentumFxFunctionalChannelReport,
    FxFunctionalRows,
    FxFunctionalRows,
    bool,
)> {
    let channel_path = fx_channel_checkpoint_path(checkpoint_directory, gauge_form_degree);
    match load_fx_channel_checkpoint(&channel_path, gauge_form_degree, target_ordinal) {
        Ok(Some(checkpoint)) => {
            log_fx_progress(
                checkpoint_directory,
                "channel-complete",
                Some(gauge_form_degree),
                None,
                None,
                true,
                None,
                0,
            )?;
            return Ok(checkpoint);
        }
        Ok(None) => {}
        Err(error) => quarantine_invalid_fx_checkpoint(&channel_path, &error)?,
    }

    let mut x2_rows = empty_fx_functional_rows();
    let mut x5_rows = empty_fx_functional_rows();
    let mut emitted_target_terms = 0_u64;
    let mut response_cache = FxResponseCache::new();
    log_fx_progress(
        checkpoint_directory,
        "channel-start",
        Some(gauge_form_degree),
        None,
        None,
        false,
        None,
        0,
    )?;
    for operator_ordinal in 0..FIRST_MOMENTUM_FX_OPERATOR_COLUMNS {
        let operator_path =
            fx_operator_checkpoint_path(checkpoint_directory, gauge_form_degree, operator_ordinal);
        let started = Instant::now();
        let loaded = match load_fx_operator_checkpoint(
            &operator_path,
            gauge_form_degree,
            target_ordinal,
            operator_ordinal,
        ) {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                quarantine_invalid_fx_checkpoint(&operator_path, &error)?;
                None
            }
        };
        let (operator_x2, operator_x5, emitted, unique, reused) = match loaded {
            Some((x2, x5, emitted, unique, _, true)) => (x2, x5, emitted, unique, true),
            resumed => {
                let resumed_entries = resumed.as_ref().map(|value| value.4);
                log_fx_progress(
                    checkpoint_directory,
                    if resumed_entries.unwrap_or(0) == 0 {
                        "operator-start"
                    } else {
                        "operator-resume"
                    },
                    Some(gauge_form_degree),
                    Some(operator_ordinal),
                    resumed_entries,
                    false,
                    None,
                    response_cache.len(),
                )?;
                let (checkpoint, x2, x5) = build_first_momentum_fx_operator_checkpoint(
                    gauge_form_degree,
                    target_ordinal,
                    operator_ordinal,
                    &mut response_cache,
                    &operator_path,
                    checkpoint_directory,
                    resumed,
                )?;
                atomic_json(&operator_path, &checkpoint)?;
                (
                    x2,
                    x5,
                    checkpoint.emitted_target_terms,
                    checkpoint.source_entries_unique,
                    false,
                )
            }
        };
        add_fx_functional_rows(&mut x2_rows, &operator_x2);
        add_fx_functional_rows(&mut x5_rows, &operator_x5);
        emitted_target_terms += emitted;
        log_fx_progress(
            checkpoint_directory,
            "operator-complete",
            Some(gauge_form_degree),
            Some(operator_ordinal),
            Some(unique),
            reused,
            Some(started.elapsed().as_secs_f64()),
            response_cache.len(),
        )?;
    }
    let x2_solution = solve_fx_functional_rows(&x2_rows, gauge_form_degree, "X2");
    let x5_solution = solve_fx_functional_rows(&x5_rows, gauge_form_degree, "X5");
    let mut joint_rows = x2_rows.clone();
    joint_rows.extend(x5_rows.clone());
    let joint_solution = solve_fx_functional_rows(&joint_rows, gauge_form_degree, "X2+X5");
    let mutation_detected = x2_rows != x5_rows;
    let report = FirstMomentumFxFunctionalChannelReport {
        gauge_form_degree,
        parameter_components_total: [1, 11, 55, 165, 330, 462][gauge_form_degree],
        parameter_components_selected: vec![0],
        target_basis_ordinals_selected: vec![target_ordinal],
        operator_columns_composed: FIRST_MOMENTUM_FX_OPERATOR_COLUMNS,
        emitted_target_terms,
        x2_functional_rank_lower_bound: x2_solution.rank,
        x2_functional_nullity_upper_bound: x2_solution.nullity,
        x5_functional_rank_lower_bound: x5_solution.rank,
        x5_functional_nullity_upper_bound: x5_solution.nullity,
        joint_functional_rank_lower_bound: joint_solution.rank,
        joint_functional_nullity_upper_bound: joint_solution.nullity,
    };
    let checkpoint = FirstMomentumFxChannelCheckpoint {
        schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
        curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
        gauge_form_degree,
        target_basis_ordinal: target_ordinal,
        completed_operator_ordinals: (0..FIRST_MOMENTUM_FX_OPERATOR_COLUMNS).collect(),
        report: report.clone(),
        x2_rows: encode_fx_rows(&x2_rows),
        x5_rows: encode_fx_rows(&x5_rows),
        mutation_detected,
    };
    atomic_json(&channel_path, &checkpoint)?;
    log_fx_progress(
        checkpoint_directory,
        "channel-complete",
        Some(gauge_form_degree),
        None,
        None,
        false,
        None,
        response_cache.len(),
    )?;
    Ok((report, x2_rows, x5_rows, mutation_detected))
}

fn finish_first_momentum_fx_functional_report(
    mut channel_reports: Vec<FirstMomentumFxFunctionalChannelReport>,
    global_x2: FxFunctionalRows,
    global_x5: FxFunctionalRows,
    mutation_detected: bool,
) -> FirstMomentumFxFunctionalReport {
    channel_reports.sort_by_key(|report| report.gauge_form_degree);
    let global_x2_solution = solve_fx_functional_rows(&global_x2, 6, "global-X2");
    let global_x5_solution = solve_fx_functional_rows(&global_x5, 6, "global-X5");
    let mut global_joint = global_x2;
    global_joint.extend(global_x5);
    let global_joint_solution = solve_fx_functional_rows(&global_joint, 6, "global-X2+X5");
    FirstMomentumFxFunctionalReport {
        schema_version: FIRST_MOMENTUM_FX_FUNCTIONAL_SCHEMA,
        role: "exact deterministic mask-summed functional lower bound for all-six first-momentum partial F_X A G_p on a declared target/parameter slice",
        curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256,
        coefficient_space: "five exact leading F_X-kernel coordinates plus 44 recorded first-momentum correction coordinates",
        coefficient_variables: 49,
        leading_kernel_variables: 5,
        first_momentum_correction_variables: 44,
        deterministic_hash_seeds: FIRST_MOMENTUM_FX_FUNCTIONAL_SEEDS
            .iter()
            .map(|seed| format!("{seed:016x}"))
            .collect(),
        buckets_per_seed: FIRST_MOMENTUM_FX_FUNCTIONAL_BUCKETS,
        bounded_channel_concurrency: 1,
        operator_checkpoints_per_channel: FIRST_MOMENTUM_FX_OPERATOR_COLUMNS,
        checkpoint_resume_enabled: true,
        channel_reports,
        all_six_channels_composed_on_declared_slice: true,
        full_parameter_projection_complete: false,
        full_target_projection_complete: false,
        global_x2_rank_lower_bound: global_x2_solution.rank,
        global_x2_nullity_upper_bound: global_x2_solution.nullity,
        global_x5_rank_lower_bound: global_x5_solution.rank,
        global_x5_nullity_upper_bound: global_x5_solution.nullity,
        global_joint_rank_lower_bound: global_joint_solution.rank,
        global_joint_nullity_upper_bound: global_joint_solution.nullity,
        global_joint_rank_exact_by_dimension_saturation: global_joint_solution.rank == 49,
        surviving_leading_projection_rank_upper_bound: if global_joint_solution.rank == 49 {
            0
        } else {
            5
        },
        mutation_detected,
        partial_fx_only: true,
        full_f_a_g_p_established: false,
        boundary: "Each reported rank is exact for the deterministic functional slice and is a rigorous lower bound on the full partial-F_X composition. The slice sums exterior-spinor masks before applying coordinate/sector/momentum hash functionals, which preserves lower-bound validity but can lower detected rank. Dimension saturation at rank 49 proves that the recorded leading-plus-first-momentum coefficient space has zero kernel. Without saturation, functional survivors are inconclusive. The selected slice is not a complete parameter or target projection, F_X omits J/W, and the 12+44 ansatz omits higher momentum descendants; full F A G_p remains false.",
    }
}

pub fn build_first_momentum_fx_functional_report_in(
    checkpoint_directory: &Path,
) -> io::Result<FirstMomentumFxFunctionalReport> {
    let target_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
        .into_iter()
        .find(|state| state.pbw_word_simple_roots.is_empty())
        .expect("target highest-weight state")
        .ordinal;
    let mut global_x2 = Vec::new();
    let mut global_x5 = Vec::new();
    let mut channel_reports = Vec::new();
    let mut mutation_detected = false;
    log_fx_progress(
        checkpoint_directory,
        "run-start",
        None,
        None,
        None,
        false,
        None,
        0,
    )?;
    for gauge_form_degree in 0..6 {
        let (report, x2_rows, x5_rows, channel_mutation) =
            build_first_momentum_fx_functional_channel(
                gauge_form_degree,
                target_ordinal,
                checkpoint_directory,
            )?;
        mutation_detected |= channel_mutation;
        global_x2.extend(x2_rows);
        global_x5.extend(x5_rows);
        channel_reports.push(report);
    }
    let report = finish_first_momentum_fx_functional_report(
        channel_reports,
        global_x2,
        global_x5,
        mutation_detected,
    );
    log_fx_progress(
        checkpoint_directory,
        "run-complete",
        None,
        None,
        None,
        false,
        None,
        0,
    )?;
    Ok(report)
}

pub fn build_first_momentum_fx_functional_report() -> io::Result<FirstMomentumFxFunctionalReport> {
    build_first_momentum_fx_functional_report_in(Path::new(
        "results/eleven_dimensional_first_momentum_fx_checkpoints",
    ))
}

/// Merge the final first-momentum `F_X` functional artifact from a complete
/// production checkpoint tree without constructing any operator state.
///
/// All 6 x 56 operator checkpoints must already exist, carry the exact v4
/// schema and frozen curvature-input hash, contain complete progress, and
/// decode to the declared functional-row shape. Missing, partial, or corrupt
/// inputs fail before `path` is touched. This function never quarantines,
/// resumes, repairs, or computes an operator checkpoint. Its only write is the
/// final atomic artifact commit after channel/global solves and strict report
/// validation succeed.
pub fn merge_first_momentum_fx_functional_artifact_from_complete_checkpoints(
    path: &Path,
    checkpoint_directory: &Path,
) -> io::Result<()> {
    let target_ordinal = FIRST_MOMENTUM_FX_TARGET_BASIS_ORDINAL;
    let mut global_x2 = Vec::new();
    let mut global_x5 = Vec::new();
    let mut channel_reports = Vec::with_capacity(6);
    let mut mutation_detected = false;

    for gauge_form_degree in 0..6 {
        let mut x2_rows = empty_fx_functional_rows();
        let mut x5_rows = empty_fx_functional_rows();
        let mut emitted_target_terms = 0_u64;
        for operator_ordinal in 0..FIRST_MOMENTUM_FX_OPERATOR_COLUMNS {
            let operator_path = fx_operator_checkpoint_path(
                checkpoint_directory,
                gauge_form_degree,
                operator_ordinal,
            );
            let checkpoint = load_fx_operator_checkpoint(
                &operator_path,
                gauge_form_degree,
                target_ordinal,
                operator_ordinal,
            )?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "required first-momentum F_X operator checkpoint is missing: {}",
                        operator_path.display()
                    ),
                )
            })?;
            let (operator_x2, operator_x5, emitted, unique, processed, complete) = checkpoint;
            if !complete || unique == 0 || processed != unique {
                return Err(invalid_fx_artifact(format!(
                    "required first-momentum F_X operator checkpoint is not complete: {}",
                    operator_path.display()
                )));
            }
            emitted_target_terms = emitted_target_terms.checked_add(emitted).ok_or_else(|| {
                invalid_fx_artifact(format!(
                    "emitted-target-term count overflow while merging {}",
                    operator_path.display()
                ))
            })?;
            add_fx_functional_rows(&mut x2_rows, &operator_x2);
            add_fx_functional_rows(&mut x5_rows, &operator_x5);
        }

        let x2_solution = solve_fx_functional_rows(&x2_rows, gauge_form_degree, "X2");
        let x5_solution = solve_fx_functional_rows(&x5_rows, gauge_form_degree, "X5");
        let mut joint_rows = x2_rows.clone();
        joint_rows.extend(x5_rows.clone());
        let joint_solution = solve_fx_functional_rows(&joint_rows, gauge_form_degree, "X2+X5");
        mutation_detected |= x2_rows != x5_rows;
        channel_reports.push(FirstMomentumFxFunctionalChannelReport {
            gauge_form_degree,
            parameter_components_total: [1, 11, 55, 165, 330, 462][gauge_form_degree],
            parameter_components_selected: vec![0],
            target_basis_ordinals_selected: vec![target_ordinal],
            operator_columns_composed: FIRST_MOMENTUM_FX_OPERATOR_COLUMNS,
            emitted_target_terms,
            x2_functional_rank_lower_bound: x2_solution.rank,
            x2_functional_nullity_upper_bound: x2_solution.nullity,
            x5_functional_rank_lower_bound: x5_solution.rank,
            x5_functional_nullity_upper_bound: x5_solution.nullity,
            joint_functional_rank_lower_bound: joint_solution.rank,
            joint_functional_nullity_upper_bound: joint_solution.nullity,
        });
        global_x2.extend(x2_rows);
        global_x5.extend(x5_rows);
    }

    let report = finish_first_momentum_fx_functional_report(
        channel_reports,
        global_x2,
        global_x5,
        mutation_detected,
    );
    validate_generated_first_momentum_fx_functional_report(&report)?;
    atomic_json(path, &report)
}

pub fn write_first_momentum_fx_functional_artifact_with_checkpoints(
    path: &Path,
    checkpoint_directory: &Path,
) -> io::Result<()> {
    let report = build_first_momentum_fx_functional_report_in(checkpoint_directory)?;
    validate_generated_first_momentum_fx_functional_report(&report)?;
    atomic_json(path, &report)
}

/// Operator-major all-six runner. Every operator state is built once and its
/// six production-schema checkpoints are committed before the next operator
/// starts. A restart reuses all complete operator checkpoints, then the
/// established channel merger produces the final report.
pub fn write_first_momentum_fx_functional_artifact_shared_with_checkpoints(
    path: &Path,
    checkpoint_directory: &Path,
    reference_checkpoint_directory: Option<&Path>,
) -> io::Result<()> {
    let degrees = [0_usize, 1, 2, 3, 4, 5];
    for operator_ordinal in 0..FIRST_MOMENTUM_FX_OPERATOR_COLUMNS {
        let report_path = checkpoint_directory.join(format!(
            "operator-{operator_ordinal:02}-shared-benchmark.json"
        ));
        let report = build_first_momentum_fx_shared_operator_batch_in(
            operator_ordinal,
            &degrees,
            checkpoint_directory,
            reference_checkpoint_directory,
            &report_path,
        )?;
        if !report.passed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("shared F_X operator {operator_ordinal} failed its exact gate"),
            ));
        }
    }
    let report = build_first_momentum_fx_functional_report_in(checkpoint_directory)?;
    validate_generated_first_momentum_fx_functional_report(&report)?;
    atomic_json(path, &report)
}

pub fn write_first_momentum_fx_functional_artifact(path: &Path) -> io::Result<()> {
    write_first_momentum_fx_functional_artifact_with_checkpoints(
        path,
        Path::new("results/eleven_dimensional_first_momentum_fx_checkpoints"),
    )
}

pub fn verify() -> PhysicalCurvatureOperatorReport {
    let first_momentum_fx_declared_slice_status = first_momentum_fx_declared_slice_status();
    let eq25 = eq25_dh_to_bosonic_frame_operator();
    let eq26 = eq26_spinor_anholonomy_operator();
    let eq28_h = eq28_h_to_c_alpha_b_c_operator();
    let spinorial_connection = c_alpha_b_c_to_spinorial_connection_operator();
    let bosonic_connection = d_spinorial_connection_to_bosonic_connection_operator();
    let mixed_torsion_connection = bosonic_connection_to_t_alpha_e_gamma_operator();
    let j_one_anholonomy = c_alpha_beta_gamma_to_j_one_operator();
    let j_one_connection = spinorial_connection_to_j_one_operator();
    let t_to_w = t_alpha_e_gamma_to_w_operator();
    let d_j_to_w = d_j_to_w_operator();
    let gamma_two = gamma_dh_operator(2);
    let gamma_five = gamma_dh_operator(5);
    let (x2_idempotence, x2_constraints) = hook_residuals(2);
    let (x5_idempotence, x5_constraints) = hook_residuals(5);
    let gamma_rank_symmetry_residual_entries = gamma_symmetry_residuals();
    let eq25_to_eq29_curl_certificate_residual_entries = eq25_eq29_curl_residuals();
    let equation_44_x_projection_residual_entries = equation_44_x_projection_residuals();
    let compensator_solution_residual_entries = compensator_solution_residuals();
    let (p5_named_normalization_residual_entries, p5_named_normalization_mutation_detected) =
        p5_named_normalization_probe();
    let (higher_jet_lift_residual_entries, higher_jet_lift_mutation_detected) =
        higher_jet_lift_probe();
    let derivative_lorentz_quotient_orthogonality_residual_entries =
        derivative_lorentz_quotient_orthogonality_residuals();
    let j_one_convention_audit = j_one_convention_audit();
    let j_one_lorentz_image_probe_residual_entries = j_one_convention_audit
        .iter()
        .find(|row| row.preserves_existing_green_gates)
        .expect("the source convention is present in the bounded audit")
        .lorentz_image_residual_entries;
    let j_one_convention_audit_qualifying_rows = j_one_convention_audit
        .iter()
        .filter(|row| row.qualifies)
        .count();
    let (polynomial_fx_preserves_all_eleven_momenta, polynomial_fx_mutation_detected) =
        polynomial_fx_probe();
    let partial_fx_channels = partial_fx_channels();
    let k_fag_solver_refuses_missing_target_basis_join = k_fag_adapter_refuses_missing_join();
    let k_fag_adapter_accepts_exact_b5_target_coordinate = k_fag_adapter_accepts_exact_join();
    let (
        leading_fx_k_solver_rank,
        leading_fx_k_solver_nullity,
        leading_fx_k_solver_kernel_matches_source_relations,
        leading_fx_k_solver_mutation_detected,
    ) = leading_fx_k_solver_probe();
    let spinorial_connection_constraint_residual_entries =
        spinorial_connection_constraint_residuals(&spinorial_connection);
    let spinorial_connection_mutation_detected = spinorial_connection_mutation_detected();
    let bosonic_connection_constraint_residual_entries =
        bosonic_connection_constraint_residuals(16);
    let bosonic_connection_mutation_detected = bosonic_connection_mutation_detected();
    let mixed_torsion_connection_mutation_detected = mixed_torsion_connection_mutation_detected();
    let j_one_connection_mutation_detected = j_one_connection_mutation_detected();
    let w_coefficient_mutation_detected = w_coefficient_mutation_detected();
    let (equation_26_output_symmetry_residual_entries, equation_26_mutation_detected) =
        equation_26_probe();
    let equation_26_unit_block_normalization_residual_entries =
        equation_26_block_normalization_residuals();
    let (
        equation_24_lorentz_injection_residual_entries,
        equation_24_lorentz_injection_mutation_detected,
    ) = equation_24_lorentz_injection_probe();
    let scalar_sector_j_residual_entries = scalar_sector_j_residuals();
    let convention_mutation_detected = mutation_detected();
    let bounded_slice_passed = x2_idempotence + x5_idempotence == 0
        && x2_constraints + x5_constraints == 0
        && gamma_rank_symmetry_residual_entries == 0
        && eq25_to_eq29_curl_certificate_residual_entries == 0
        && equation_44_x_projection_residual_entries == 0
        && compensator_solution_residual_entries == 0
        && p5_named_normalization_residual_entries == 0
        && p5_named_normalization_mutation_detected
        && higher_jet_lift_residual_entries == 0
        && higher_jet_lift_mutation_detected
        && derivative_lorentz_quotient_orthogonality_residual_entries == 0
        && polynomial_fx_preserves_all_eleven_momenta
        && polynomial_fx_mutation_detected
        && k_fag_solver_refuses_missing_target_basis_join
        && k_fag_adapter_accepts_exact_b5_target_coordinate
        && leading_fx_k_solver_rank == 7
        && leading_fx_k_solver_nullity == 5
        && leading_fx_k_solver_kernel_matches_source_relations
        && leading_fx_k_solver_mutation_detected
        && spinorial_connection_constraint_residual_entries == 0
        && spinorial_connection_mutation_detected
        && bosonic_connection_constraint_residual_entries == 0
        && bosonic_connection_mutation_detected
        && mixed_torsion_connection_mutation_detected
        && j_one_connection_mutation_detected
        && w_coefficient_mutation_detected
        && equation_26_output_symmetry_residual_entries == 0
        && equation_26_unit_block_normalization_residual_entries == 0
        && equation_26_mutation_detected
        && equation_24_lorentz_injection_residual_entries == 0
        && equation_24_lorentz_injection_mutation_detected
        && scalar_sector_j_residual_entries == 0
        && convention_mutation_detected;
    PhysicalCurvatureOperatorReport {
        schema_version: "adynkra-11d-physical-curvature-operator-v10",
        source_locators: vec![
            "hep-th/0101037 Eq. (25): linearized vector frame",
            "hep-th/0101037 Eq. (29): vector-vector anholonomy",
            "hep-th/0101037 Eqs. (39)-(40): X_[2], X_[5], and conventional compensator elimination",
            "hep-th/0101037 Eq. (44): gauge-independent curvature definitions",
            "hep-th/0107155v2 Eqs. (3.2c)-(3.2e): conventional torsion solves the spinorial connection and fixes the positive spinor Lorentz-connection action",
            "hep-th/0106150v2 App. A Eqs. (A.5)-(A.6): ordered five-index epsilon sum and gamma/epsilon orientation",
            "arXiv:2007.05097 Eqs. (2.2)-(2.6): gamma-traceless target and conventional constraints",
            "arXiv:2007.05097 Eqs. (2.19)-(2.23): J^(1), J^(2), J^(+), and all-real-gamma W",
        ],
        source_hashes: vec![HEP_TH_0101037_SOURCE_SHA256, ARXIV_2007_05097_SOURCE_SHA256],
        lorentz_signature: "diag(-,+,+,+,+,+,+,+,+,+,+)",
        epsilon_convention: "epsilon_(0...10)=+1; canonical increasing form masks",
        antisymmetrization_convention: "unit weight, including 1/p!",
        raised_spinor_gamma_convention: "Gamma_[p]^{alpha beta}=-(Gamma_[p] C^{-1})^{alpha beta}, derived by raising both indices of C Gamma_[p]; C^{-1}=-C",
        spinorial_connection_source_relation: "omega_(alpha,de)=C_(alpha,[de])-(2/55)(Gamma_de)_alpha{}^gamma C_(gamma,b){}^b, from hep-th/0107155v2 Eq. (3.2c) and the Table 3 constraint",
        mixed_torsion_connection_source_relation: "T_(alpha,b){}^gamma=C_(alpha,b){}^gamma+(1/4)(Gamma^cd)_alpha{}^gamma omega_(b,cd), from hep-th/0107155v2 Eq. (3.2e)",
        j_one_connection_trace_convention: "stored omega_(alpha,de) is lower-index, so the J^(1) connection trace contracts it with Gamma^de; using Gamma_de is the certified boost-sign mutation",
        stored_x5_to_named_psi5_relation: "stored Eq. (39) exterior X_[5] correction = -(*Psi_[5]); named Psi_[5] is recovered by a sign flip followed by inverse Hodge duality",
        dh_dimension: DH_DIMENSION,
        eq25_frame_dimension: FRAME_DIMENSION,
        eq25_dh_operator_nonzero_entries: eq25.nonzero_entries(),
        eq25_to_eq29_curl_certificate_residual_entries,
        eq28_h_sector_operator_nonzero_entries: eq28_h.nonzero_entries(),
        equation_26_factored_blocks: eq26.blocks.len(),
        equation_26_unit_block_normalization_residual_entries,
        equation_26_output_symmetry_residual_entries,
        equation_26_mutation_detected,
        equation_24_lorentz_injection_residual_entries,
        equation_24_lorentz_injection_mutation_detected,
        h_sector_j_operator_implemented: true,
        scalar_sector_j_coefficient: "31/24",
        scalar_sector_j_residual_entries,
        spinorial_connection_dimension: SPINORIAL_CONNECTION_DIMENSION,
        spinorial_connection_operator_nonzero_entries: spinorial_connection.nonzero_entries(),
        spinorial_connection_constraint_residual_entries,
        spinorial_connection_mutation_detected,
        bosonic_connection_dimension: BOSONIC_CONNECTION_DIMENSION,
        bosonic_connection_operator_nonzero_entries: bosonic_connection.nonzero_entries(),
        bosonic_connection_constraint_residual_entries,
        bosonic_connection_mutation_detected,
        mixed_torsion_connection_operator_nonzero_entries: mixed_torsion_connection
            .nonzero_entries(),
        mixed_torsion_connection_mutation_detected,
        j_one_anholonomy_operator_nonzero_entries: j_one_anholonomy.nonzero_entries(),
        j_one_connection_operator_nonzero_entries: j_one_connection.nonzero_entries(),
        j_one_connection_mutation_detected,
        j_plus_basis_change_implemented: true,
        convention_separated_w_assembly_implemented: true,
        t_alpha_e_gamma_to_w_operator_nonzero_entries: t_to_w.nonzero_entries(),
        d_j_to_w_operator_nonzero_entries: d_j_to_w.nonzero_entries(),
        linearized_w_coefficients_implemented: true,
        linearized_2001_w_coefficients_implemented: true,
        w_2021_linear_torsion_coefficient: "1/32",
        w_2021_linear_j_plus_coefficient: "11i/128",
        w_2001_linear_torsion_coefficient: "i/32",
        w_2001_linear_j_two_coefficient: "-11/128",
        j_plus_from_h_hat_implemented: false,
        w_coefficient_mutation_detected,
        gamma_two_dh_operator_nonzero_entries: gamma_two.nonzero_entries(),
        gamma_five_dh_operator_nonzero_entries: gamma_five.nonzero_entries(),
        x2_ambient_dimension: TWO_FORM_VECTOR_DIMENSION,
        x2_hook_dimension: 429,
        x2_compensator_image_rank: 176,
        x5_ambient_dimension: FIVE_FORM_VECTOR_DIMENSION,
        x5_hook_dimension: 4_290,
        x5_compensator_image_rank: 792,
        compensator_solution_residual_entries,
        hook_idempotence_residual_entries: x2_idempotence + x5_idempotence,
        hook_constraint_residual_entries: x2_constraints + x5_constraints,
        gamma_rank_symmetry_residual_entries,
        dimension_zero_torsion_reconstruction_implemented: true,
        equation_44_x_projection_residual_entries,
        convention_mutation_detected,
        equation_25_bosonic_frame_implemented: true,
        equation_29_bosonic_anholonomy_implemented: true,
        equation_28_h_sector_implemented: true,
        equation_26_spinor_anholonomy_implemented: true,
        table_3_spinorial_connection_solved: true,
        table_3_bosonic_connection_solved_from_d_spinorial_connection: true,
        complete_t_alpha_e_gamma_from_geometry_inputs_implemented: true,
        j_one_from_geometry_inputs_implemented: true,
        equations_39_40_compensator_image_eliminated: true,
        higher_jet_conventional_constraint_ambient_dimension: SPINOR_DIMENSION * 1_023,
        higher_jet_conventional_constraint_rank: SPINOR_DIMENSION * 968,
        higher_jet_conventional_constraint_nullity: SPINOR_DIMENSION * 55,
        higher_jet_solve_classification: "family: unique p=1, p=3, p=4, and individually normalized p=5 derivative images modulo the 32x55 spinorial derivative of the Lorentz p=2 gauge compensator",
        higher_jet_lift_residual_entries,
        higher_jet_lift_mutation_detected,
        derivative_lorentz_quotient_kernel_equals_image:
            derivative_lorentz_quotient_orthogonality_residual_entries == 0,
        derivative_lorentz_quotient_basis: "lexicographic (outer spinor, Clifford degree p in [1,3,4,5], increasing Lorentz mask); the p=2 coordinates are the 1760-dimensional derivative-Lorentz image",
        derivative_lorentz_quotient_orthogonality_residual_entries,
        j_one_lorentz_image_probe_residual_entries,
        induced_j_one_on_quotient_established: j_one_lorentz_image_probe_residual_entries == 0,
        induced_t_and_w_on_quotient_established: false,
        p5_normalization_eliminated_or_fixed_by_w: true,
        p5_named_normalization_parameter: "fixed: the unique-mask Eq. (39) coefficient is -1 because (1/16)(2/15)5!=1; Psi_[5] is the exact inverse Hodge image in epsilon_(0...10)=+1, diag(-,+,...,+)",
        p5_named_normalization_residual_entries,
        p5_named_normalization_mutation_detected,
        j_one_convention_audit,
        j_one_convention_audit_qualifying_rows,
        polynomial_fx_api_implemented: true,
        polynomial_fx_output_dimension: 429 + 4_290,
        polynomial_fx_preserves_all_eleven_momenta,
        polynomial_fx_mutation_detected,
        partial_fx_channels,
        leading_x2_stream_artifact_sha256: "c3cc58b3545c9fef1e19d351c9d0f839e1f5c846db1a335acb9d6dc91af53968",
        leading_x2_all_six_channels_composed: true,
        leading_x5_all_six_channels_composed: false,
        first_momentum_fx_all_six_channels_composed: false,
        first_momentum_fx_declared_slice_status,
        individually_excluded_leading_k_ordinals_union: vec![0, 1, 2, 4, 6, 7, 8, 9, 10, 11],
        individually_unexcluded_leading_k_ordinals: vec![3, 5],
        linear_combination_survivor_space_solved: true,
        leading_x2_joint_exact_rank: 7,
        leading_x2_joint_exact_nullity: 5,
        leading_x2_joint_kernel_basis: vec![
            vec![-18, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![30, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            vec![54, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ],
        leading_x2_joint_kernel_proved_on_exact_source_streams: true,
        leading_fx_combined_kernel_proved_by_rank_sandwich: true,
        leading_fx_k_solver_rank,
        leading_fx_k_solver_nullity,
        leading_fx_k_solver_kernel_matches_source_relations,
        leading_fx_k_solver_mutation_detected,
        b5_to_cartesian_majorana_intertwiner_implemented: {
            let join =
                crate::eleven_dimensional_abstract_clifford_join::b5_cartesian_majorana_intertwiner(
                );
            join.vector_to_lorentz_cartesian.len() == VECTOR_DIMENSION
                && join.spinor_weight_to_majorana.len() == SPINOR_DIMENSION
        },
        k_fag_target_key_retains_raw_vector_spinor_coordinates: true,
        k_fag_solver_adapter_present: true,
        k_fag_solver_refuses_missing_target_basis_join,
        k_fag_adapter_accepts_exact_b5_target_coordinate,
        historical_first_momentum_controls_degrees: vec![1, 2, 5],
        historical_controls_are_not_fx_compositions: true,
        partial_fx_a_g_p_vanishing_established: false,
        dimension_zero_x_curvature_operator_implemented: true,
        individual_p1_p3_p4_fields_solvable_in_fixed_convention: true,
        individual_epsilon_contracted_p5_normalization_source_fixed: true,
        full_equations_24_to_29_operator_implemented: false,
        spin_connections_solved: true,
        w_and_j_from_h_hat_implemented: false,
        physical_psi_to_h_hat_k_source_fixed: false,
        complete_f_from_h_hat_implemented: false,
        full_f_a_g_p_test_ready: false,
        covariant_off_shell_closure_established: false,
        bounded_slice_passed,
        boundary: "The exact Eq. (25) H-sector frame, factored Eq. (26) spinor anholonomy, Eq. (28) H and scalar sectors, Eq. (29) bosonic anholonomy, both Table 3 Lorentz connections, Eq. (39)-(40) X_[2]/X_[5] compensator quotient, generic J^(1), J^(2), and J^(+) geometry maps, complete mixed-torsion assembly from anholonomy and connection inputs, and convention-separated 2001/2021 linearized W assembly are executable over Q(i). The spinor-raising sign, ordered Eq. (26) rank factorials, lower-coordinate Lorentz injection, spinorial-connection solve, Eq. (39) X signs, named Psi_[5] inverse-Hodge sign, positive Gamma omega/4 mixed-torsion term, raised Gamma^de J^(1) connection trace, and 2021 W coefficient 11i/128 follow source-fixed audits; Eq. (44) still recovers both X sectors exactly. The typed polynomial F_X API acts exactly on the full Cartesian-Majorana 429+4290 quotient and preserves all eleven formal B5 momentum exponents. The exact abstract-B5 vector and spinor intertwiners are derived from representation action and the Majorana involution, with no phase guessing. The joint leading X_[2] stream over all six channels has exact rank seven and nullity five. Five relations vanish on the complete source streams before either X sector is applied, so they give a common upper rank bound for X_[2]+X_[5]; the rank-seven X_[2] functional image saturates that bound. The resulting exact five-vector combined F_X kernel is fed to the K solver, which independently returns rank seven and nullity five and detects a one-row mutation. TargetVariationKey retains the raw B5 vector/spinor coordinates and the curvature adapter accepts them, emits both X_[2] and X_[5] quotient sectors, and rejects legacy ordinal-only terms. The all-six first-momentum F_X functional runner is implemented separately and keeps parameter/target completeness explicit. Historical first-momentum screens for p=1,2,5 remain old-ansatz negative controls, not F_X compositions. The differentiated conventional-constraint solve has rank 30,976 and a 1,760-dimensional p=2 derivative-Lorentz kernel; an exact Clifford-orthogonality certificate proves kernel=image. The Eq. (28) Delta term now follows the Appendix-A spinor metric identity C_{alpha beta} C^{gamma beta}=delta_alpha^gamma, using (gamma_b)^{beta delta}=C Gamma_b and (gamma^c)_{epsilon alpha}=Gamma^c C rather than mixed-index Clifford products. This restores one Lorentz-uniform connection trace coefficient on all 1,760 derivative p=2 columns. The resulting bounded J^(1) response is the unique Lorentz-equivariant Gamma^[2] map with coefficient 109/1056, but it remains nonzero and has not been identified with a complete constrained gauge orbit. No compensating term is added because no audited source prints one. Therefore induced J/T/W, full F A G_p, Bianchi identities, and off-shell closure remain fail-closed.",
    }
}

pub fn write_artifacts(data_path: &Path, results_path: &Path) -> io::Result<()> {
    let report = verify();
    for path in [data_path, results_path] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        serde_json::to_writer_pretty(&mut file, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finished_first_momentum_fx_slice_is_pinned_and_strictly_qualified() {
        let snapshot_bytes = fs::read(FIRST_MOMENTUM_FX_INPUT_SNAPSHOT_PATH).unwrap();
        assert_eq!(
            sha256_hex(&snapshot_bytes),
            FIRST_MOMENTUM_FX_CURVATURE_SHA256
        );
        assert_eq!(
            validate_first_momentum_fx_input_snapshot_bytes(&snapshot_bytes).unwrap(),
            "adynkra-11d-physical-curvature-operator-v10"
        );

        let artifact_bytes = fs::read(FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_PATH).unwrap();
        assert_eq!(
            sha256_hex(&artifact_bytes),
            FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_SHA256
        );
        let artifact = validate_first_momentum_fx_functional_bytes(&artifact_bytes).unwrap();
        assert_eq!(artifact.schema_version, FIRST_MOMENTUM_FX_FUNCTIONAL_SCHEMA);
        assert_eq!(
            artifact.curvature_artifact_sha256,
            FIRST_MOMENTUM_FX_CURVATURE_SHA256
        );
        assert_eq!(artifact.global_x2_rank_lower_bound, 49);
        assert_eq!(artifact.global_x2_nullity_upper_bound, 0);
        assert_eq!(artifact.global_x5_rank_lower_bound, 49);
        assert_eq!(artifact.global_x5_nullity_upper_bound, 0);
        assert_eq!(artifact.global_joint_rank_lower_bound, 49);
        assert_eq!(artifact.global_joint_nullity_upper_bound, 0);
        assert!(artifact.all_six_channels_composed_on_declared_slice);
        assert!(!artifact.full_parameter_projection_complete);
        assert!(!artifact.full_target_projection_complete);
        assert!(artifact.partial_fx_only);
        assert!(!artifact.full_f_a_g_p_established);

        let promotion_bytes = fs::read(FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_PATH).unwrap();
        assert_eq!(
            sha256_hex(&promotion_bytes),
            FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_SHA256
        );
        let promotion = validate_first_momentum_fx_promotion_bytes(&promotion_bytes).unwrap();
        assert_eq!(promotion.candidate_sha256.len(), 336);
        assert_eq!(promotion.verified_existing, 164);
        assert_eq!(promotion.copied_missing, 172);
        assert_eq!(promotion.replaced_partial, 0);

        let status = first_momentum_fx_declared_slice_status();
        assert!(status.fx_input_snapshot_validated);
        assert!(status.functional_report_fx_input_sha256_matches_snapshot);
        assert!(status.report_invariants_validated);
        assert!(status.checkpoint_promotion_validated);
        assert!(status.qualified_zero_kernel_on_declared_slice);
        assert_eq!(status.global_joint_rank_lower_bound, Some(49));
        assert_eq!(status.global_joint_nullity_upper_bound, Some(0));
        assert_eq!(status.full_parameter_projection_complete, Some(false));
        assert_eq!(status.full_target_projection_complete, Some(false));
        assert_eq!(status.full_f_a_g_p_established, Some(false));
        assert!(!status.current_physical_envelope_self_hash_required);
        assert!(status.validation_error.is_none());
    }

    #[test]
    fn first_momentum_fx_slice_mutations_fail_closed() {
        let artifact_bytes = fs::read(FIRST_MOMENTUM_FX_FUNCTIONAL_ARTIFACT_PATH).unwrap();
        let artifact: FirstMomentumFxFunctionalArtifact =
            serde_json::from_slice(&artifact_bytes).unwrap();

        let mut mutated = artifact.clone();
        mutated.schema_version.push_str("-mutated");
        assert!(validate_first_momentum_fx_functional_fields(&mutated).is_err());

        let mut mutated = artifact.clone();
        mutated.curvature_artifact_sha256.replace_range(0..1, "0");
        assert!(validate_first_momentum_fx_functional_fields(&mutated).is_err());

        let mut mutated = artifact.clone();
        mutated.global_joint_nullity_upper_bound = 1;
        assert!(validate_first_momentum_fx_functional_fields(&mutated).is_err());

        let mut mutated = artifact.clone();
        mutated.full_parameter_projection_complete = true;
        assert!(validate_first_momentum_fx_functional_fields(&mutated).is_err());

        let mut mutated = artifact;
        mutated.mutation_detected = false;
        assert!(validate_first_momentum_fx_functional_fields(&mutated).is_err());

        let promotion_bytes = fs::read(FIRST_MOMENTUM_FX_PROMOTION_MANIFEST_PATH).unwrap();
        let mut promotion: FirstMomentumFxPromotionManifest =
            serde_json::from_slice(&promotion_bytes).unwrap();
        promotion.passed = false;
        assert!(validate_first_momentum_fx_promotion_fields(&promotion).is_err());

        let snapshot_bytes = fs::read(FIRST_MOMENTUM_FX_INPUT_SNAPSHOT_PATH).unwrap();
        let mut snapshot: serde_json::Value = serde_json::from_slice(&snapshot_bytes).unwrap();
        snapshot["full_f_a_g_p_test_ready"] = serde_json::Value::Bool(true);
        let mutated_snapshot = serde_json::to_vec_pretty(&snapshot).unwrap();
        assert!(validate_first_momentum_fx_input_snapshot_bytes(&mutated_snapshot).is_err());
    }

    #[test]
    fn merge_only_first_momentum_fx_refuses_missing_checkpoints_before_writing() {
        let directory = std::env::temp_dir().join(format!(
            "adinkra-first-momentum-fx-merge-missing-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let checkpoints = directory.join("checkpoints");
        let output = directory.join("final.json");
        fs::create_dir_all(&checkpoints).unwrap();
        fs::write(&output, b"preexisting-output\n").unwrap();

        let error = merge_first_momentum_fx_functional_artifact_from_complete_checkpoints(
            &output,
            &checkpoints,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(fs::read(&output).unwrap(), b"preexisting-output\n");
        assert!(fs::read_dir(&checkpoints).unwrap().next().is_none());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn merge_only_first_momentum_fx_refuses_corrupt_checkpoints_without_quarantine() {
        let directory = std::env::temp_dir().join(format!(
            "adinkra-first-momentum-fx-merge-corrupt-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let checkpoints = directory.join("checkpoints");
        let corrupt = fx_operator_checkpoint_path(&checkpoints, 0, 0);
        let output = directory.join("final.json");
        fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
        fs::write(&corrupt, b"{}\n").unwrap();
        fs::write(&output, b"preexisting-output\n").unwrap();

        let error = merge_first_momentum_fx_functional_artifact_from_complete_checkpoints(
            &output,
            &checkpoints,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read(&output).unwrap(), b"preexisting-output\n");
        assert_eq!(fs::read(&corrupt).unwrap(), b"{}\n");
        assert!(
            fs::read_dir(corrupt.parent().unwrap())
                .unwrap()
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains("rejected"))
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn merge_only_first_momentum_fx_refuses_partial_checkpoints_before_writing() {
        let directory = std::env::temp_dir().join(format!(
            "adinkra-first-momentum-fx-merge-partial-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let checkpoints = directory.join("checkpoints");
        let partial = fx_operator_checkpoint_path(&checkpoints, 0, 0);
        let output = directory.join("final.json");
        let checkpoint = FirstMomentumFxOperatorCheckpoint {
            schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
            curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
            gauge_form_degree: 0,
            target_basis_ordinal: FIRST_MOMENTUM_FX_TARGET_BASIS_ORDINAL,
            operator_ordinal: 0,
            parameter_components_selected: vec![0],
            emitted_target_terms: 1,
            source_entries_unique: 1,
            source_entries_processed: 0,
            complete: false,
            x2_rows: encode_fx_rows(&empty_fx_functional_rows()),
            x5_rows: encode_fx_rows(&empty_fx_functional_rows()),
        };
        atomic_json(&partial, &checkpoint).unwrap();
        fs::write(&output, b"preexisting-output\n").unwrap();

        let error = merge_first_momentum_fx_functional_artifact_from_complete_checkpoints(
            &output,
            &checkpoints,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read(&output).unwrap(), b"preexisting-output\n");
        assert!(partial.exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn exact_hook_quotients_close() {
        let report = verify();
        assert!(report.bounded_slice_passed);
        assert_eq!(report.x2_hook_dimension, 429);
        assert_eq!(report.x5_hook_dimension, 4_290);
        assert_eq!(report.x2_compensator_image_rank, 176);
        assert_eq!(report.x5_compensator_image_rank, 792);
        assert_eq!(report.hook_idempotence_residual_entries, 0);
        assert_eq!(report.hook_constraint_residual_entries, 0);
        assert_eq!(report.spinorial_connection_constraint_residual_entries, 0);
        assert!(report.spinorial_connection_mutation_detected);
        assert!(report.w_coefficient_mutation_detected);
        assert!(!report.first_momentum_fx_all_six_channels_composed);
        assert!(
            report
                .first_momentum_fx_declared_slice_status
                .qualified_zero_kernel_on_declared_slice
        );
        assert_eq!(
            report
                .first_momentum_fx_declared_slice_status
                .full_parameter_projection_complete,
            Some(false)
        );
        assert_eq!(
            report
                .first_momentum_fx_declared_slice_status
                .full_target_projection_complete,
            Some(false)
        );
        assert_eq!(
            report
                .first_momentum_fx_declared_slice_status
                .full_f_a_g_p_established,
            Some(false)
        );
    }

    #[test]
    fn eq25_frame_contains_all_three_printed_terms() {
        let mut input = Eq25BosonicFrameInput {
            d_h: BTreeMap::new(),
            scalar_compensator: ExactQi::from_integer(3),
            lorentz_compensator: BTreeMap::new(),
        };
        input.lorentz_compensator.insert(0b11, ExactQi::one());
        let scalar_and_lorentz = apply_eq25_bosonic_frame(&input);
        assert_eq!(
            scalar_and_lorentz[&frame_index(2, 2)],
            ExactQi::from_integer(3)
        );
        assert_eq!(
            scalar_and_lorentz[&frame_index(0, 1)],
            ExactQi::from_integer(-1)
        );
        assert_eq!(
            scalar_and_lorentz[&frame_index(1, 0)],
            ExactQi::from_integer(-1)
        );

        input.d_h.insert(dh_index(0, 0, 4), ExactQi::one());
        let with_h = apply_eq25_bosonic_frame(&input);
        assert_ne!(with_h, scalar_and_lorentz);
    }

    #[test]
    fn eq29_is_the_exact_momentum_curl_of_eq25() {
        let mut input = Eq25BosonicFrameInput {
            d_h: BTreeMap::new(),
            scalar_compensator: ExactQi::from_integer(2),
            lorentz_compensator: BTreeMap::new(),
        };
        input.d_h.insert(dh_index(0, 0, 3), ExactQi::one());
        input.lorentz_compensator.insert(0b101, ExactQi::one());
        for momentum in 0..VECTOR_DIMENSION {
            let direct = apply_eq29_bosonic_anholonomy(&input, momentum);
            let curl = frame_curl_for_momentum_axis(&apply_eq25_bosonic_frame(&input), momentum);
            assert_eq!(direct, curl);
        }
    }

    #[test]
    fn eliminated_x_images_obey_both_eq40_constraints() {
        let mut input = BTreeMap::new();
        input.insert(dh_index(0, 0, 0), ExactQi::one());
        input.insert(dh_index(3, 9, 7), ExactQi::i());
        let image = apply_leading_physical_x(&input);
        for (degree, output) in [(2, &image.x_two_11000), (5, &image.x_five_10002)] {
            let tensor = sparse_to_tensor(degree, &output);
            assert!(mixed_trace(degree, &tensor).is_empty());
            assert!(total_antisymmetric_part(degree, &tensor).is_empty());
        }
        let recovered_five = recover_x_from_dimension_zero_torsion(&image, 5);
        assert_eq!(recovered_five, image.x_five_10002);
        assert_eq!(
            recover_x_from_dimension_zero_torsion(&image, 2),
            image.x_two_11000
        );

        let solution = solve_conventional_compensators(&input);
        let psi_one_image = tensor_to_indexed(2, delta_wedge(2, &solution.psi_one));
        assert_eq!(psi_one_image, image.x_two_compensators.trace_image);
        let psi_three_image = inject_total_antisymmetric(2, &solution.psi_three)
            .into_iter()
            .map(|(key, value)| (key, value.scaled(&r(-1))))
            .collect();
        assert_eq!(
            tensor_to_indexed(2, psi_three_image),
            image.x_two_compensators.exterior_image
        );
        let psi_four_image = delta_wedge(5, &solution.psi_four)
            .into_iter()
            .map(|(key, value)| (key, value.scaled(&rr(1, 48))))
            .collect();
        assert_eq!(
            tensor_to_indexed(5, psi_four_image),
            image.x_five_compensators.trace_image
        );
        assert_eq!(
            solution.epsilon_psi_five_image,
            image.x_five_compensators.exterior_image
        );
    }

    #[test]
    fn polynomial_fx_preserves_momentum_and_both_hook_quotients() {
        let term = PolynomialFxDhTerm {
            derivative_spinor: 0,
            h_spinor: 0,
            output_vector: 0,
            exterior_spinor_mask: 0x21,
            momentum: FxMomentumMonomial::variable(10),
            coefficient: ExactQi::one(),
        };
        let image = apply_polynomial_fx(&[term]);
        assert!(!image.x_two_11000.is_empty());
        assert!(!image.x_five_10002.is_empty());
        for key in image.x_two_11000.keys().chain(image.x_five_10002.keys()) {
            assert_eq!(key.exterior_spinor_mask, 0x21);
            assert_eq!(key.momentum, FxMomentumMonomial::variable(10));
        }
        let (preserves, mutation) = polynomial_fx_probe();
        assert!(preserves);
        assert!(mutation);
    }

    #[test]
    fn b5_majorana_fx_adapter_accepts_raw_coordinates_and_rejects_legacy_keys() {
        use crate::eleven_dimensional_k_fag_solver::PhysicalCurvaturePolynomialApi;
        let descriptor = CartesianPolynomialFxApi.descriptor();
        assert!(descriptor.target_basis_join_complete);
        assert!(!descriptor.complete_physical_f);
        assert!(k_fag_adapter_refuses_missing_join());
        assert!(k_fag_adapter_accepts_exact_join());
    }

    #[test]
    fn higher_jet_constraint_solve_is_a_lorentz_gauge_family() {
        let (residuals, mutation_detected) = higher_jet_lift_probe();
        assert_eq!(residuals, 0);
        assert!(mutation_detected);
        let report = verify();
        assert_eq!(report.higher_jet_conventional_constraint_rank, 30_976);
        assert_eq!(report.higher_jet_conventional_constraint_nullity, 1_760);
        assert_eq!(
            report.higher_jet_conventional_constraint_ambient_dimension,
            32_736
        );
        assert!(
            report
                .higher_jet_solve_classification
                .starts_with("family:")
        );
        assert_eq!(report.j_one_convention_audit.len(), 32);
        assert_eq!(
            report.induced_j_one_on_quotient_established,
            report.j_one_lorentz_image_probe_residual_entries == 0
        );
    }

    #[test]
    fn printed_coefficients_are_mutation_sensitive() {
        assert!(mutation_detected());
    }

    #[test]
    fn eq28_h_sector_j_is_the_connection_independent_trace() {
        let operator = eq28_h_to_c_alpha_b_c_operator();
        let column = operator
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        let mut input = BTreeMap::new();
        input.insert(column, ExactQi::one());
        input.insert(DDH_DIMENSION + ph_index(0, 0, 0), ExactQi::i());
        let c = operator.apply_sparse(&input);
        assert_eq!(
            apply_h_sector_j(&input),
            c_alpha_b_c_to_j_operator().apply_sparse(&c)
        );
        assert_eq!(scalar_sector_j_residuals(), 0);
    }

    #[test]
    fn eq26_factored_spinor_anholonomy_is_symmetric_and_mutation_sensitive() {
        let operator = eq26_spinor_anholonomy_operator();
        assert_eq!(operator.blocks.len(), 517);
        for block in &operator.blocks {
            let contraction = block
                .input_raised_spinor_gamma
                .iter()
                .flatten()
                .zip(block.output_lower_spinor_gamma.iter().flatten())
                .map(|(left, right)| i64::from(*left) * i64::from(*right))
                .sum::<i64>();
            assert_eq!(
                block.coefficient.scaled(&r(contraction)),
                ExactQi::one(),
                "Eq. (26) rank-{} block is not unit-normalized",
                block.gamma_rank
            );
        }
        let (residuals, mutation_detected) = equation_26_probe();
        assert_eq!(residuals, 0);
        assert!(mutation_detected);
    }

    #[test]
    fn eq24_lower_lorentz_coordinate_uses_ordered_upper_gamma_product() {
        let pair_masks = masks_of_degree(2);
        for mask in [(1_u16 << 1) | (1_u16 << 2), (1_u16 << 0) | (1_u16 << 1)] {
            let pair = pair_masks
                .iter()
                .position(|candidate| *candidate == mask)
                .unwrap();
            let mut input = BTreeMap::new();
            input.insert(spinorial_connection_index(0, pair), ExactQi::one());
            let actual = inject_d_lorentz_compensator_into_d_delta(&input);
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            let expected_gamma = gamma_product(&indices, false);
            for delta in 0..SPINOR_DIMENSION {
                for epsilon in 0..SPINOR_DIMENSION {
                    let key = delta * SPINOR_DIMENSION + epsilon;
                    let expected = ExactQi::from_integer(i64::from(expected_gamma[delta][epsilon]));
                    assert_eq!(
                        actual.get(&key).cloned().unwrap_or_else(ExactQi::zero),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn table_three_spinorial_connection_is_exact_and_mutation_sensitive() {
        let operator = c_alpha_b_c_to_spinorial_connection_operator();
        assert_eq!(spinorial_connection_constraint_residuals(&operator), 0);
        assert!(spinorial_connection_mutation_detected());
        let mut c = BTreeMap::new();
        c.insert(c_alpha_b_c_index(0, 0, 0), ExactQi::one());
        assert!(!apply_spinorial_connection(&c).is_empty());
    }

    #[test]
    fn table_three_bosonic_connection_and_mixed_torsion_are_exact() {
        let operator = d_spinorial_connection_to_bosonic_connection_operator();
        assert_eq!(bosonic_connection_constraint_residuals(16), 0);
        assert!(bosonic_connection_mutation_detected());
        assert!(mixed_torsion_connection_mutation_detected());
        let column = operator
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        let mut d_omega = BTreeMap::new();
        d_omega.insert(column, ExactQi::one());
        let omega = apply_bosonic_connection(&d_omega);
        assert!(!omega.is_empty());
        assert!(!apply_t_alpha_e_gamma(&BTreeMap::new(), &omega).is_empty());
    }

    #[test]
    fn j_one_j_plus_and_w_assemblies_keep_conventions_separate() {
        assert!(j_one_connection_mutation_detected());
        let mut c_spinor = BTreeMap::new();
        c_spinor.insert(0, ExactQi::one());
        let mut omega_spinor = BTreeMap::new();
        let omega_column = spinorial_connection_to_j_one_operator()
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        omega_spinor.insert(omega_column, ExactQi::one());
        let j_one = apply_j_one(&c_spinor, &omega_spinor);
        assert!(!j_one.is_empty());

        let mut j_two = BTreeMap::new();
        j_two.insert(0, ExactQi::from_integer(3));
        let j_plus = apply_j_plus(&j_one, &j_two);
        let mut expected = BTreeMap::new();
        for source in [&j_one, &j_two] {
            for (&index, value) in source {
                add_sparse(&mut expected, index, value.scaled(&rr(1, 2)));
            }
        }
        assert_eq!(j_plus, expected);

        let mut c_mixed = BTreeMap::new();
        let torsion_column = t_alpha_e_gamma_to_w_operator()
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        c_mixed.insert(torsion_column, ExactQi::one());
        let mut d_j_one = BTreeMap::new();
        let j_column = d_j_to_w_operator()
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        d_j_one.insert(j_column, ExactQi::one());
        let mut d_j_two = BTreeMap::new();
        d_j_two.insert(j_column, ExactQi::from_integer(2));
        let assembled = assemble_convention_separated_linearized_w(
            &c_mixed,
            &BTreeMap::new(),
            &d_j_one,
            &d_j_two,
        );
        assert_eq!(assembled.d_j_plus[&j_column], ExactQi::from_rational(3, 2));
        assert_ne!(assembled.w_2001, assembled.w_2021);
    }

    #[test]
    fn linearized_w_keeps_both_printed_terms() {
        let t_operator = t_alpha_e_gamma_to_w_operator();
        let t_column = t_operator
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        let mut torsion = BTreeMap::new();
        torsion.insert(t_column, ExactQi::one());
        let j_operator = d_j_to_w_operator();
        let j_column = j_operator
            .columns
            .iter()
            .position(|column| !column.is_empty())
            .unwrap();
        let mut d_j = BTreeMap::new();
        d_j.insert(j_column, ExactQi::one());
        let combined = apply_linearized_w(&torsion, &d_j);
        let mut expected = t_operator.apply_sparse(&torsion);
        for (index, value) in j_operator.apply_sparse(&d_j) {
            add_sparse(&mut expected, index, value);
        }
        assert_eq!(combined, expected);
        assert!(w_coefficient_mutation_detected());
        let old_combined = apply_linearized_w_2001(&torsion, &d_j);
        let mut old_expected = t_alpha_e_gamma_to_w_2001_operator().apply_sparse(&torsion);
        for (index, value) in d_j_two_to_w_2001_operator().apply_sparse(&d_j) {
            add_sparse(&mut old_expected, index, value);
        }
        assert_eq!(old_combined, old_expected);
        assert_ne!(old_combined, combined);
    }

    #[test]
    fn report_fails_closed_at_unprinted_physical_maps() {
        let report = verify();
        assert!(!report.physical_psi_to_h_hat_k_source_fixed);
        assert!(!report.complete_f_from_h_hat_implemented);
        assert!(!report.full_f_a_g_p_test_ready);
        assert!(!report.covariant_off_shell_closure_established);
        assert!(report.individual_epsilon_contracted_p5_normalization_source_fixed);
        assert!(report.p5_normalization_eliminated_or_fixed_by_w);
        assert_eq!(report.p5_named_normalization_residual_entries, 0);
        assert!(report.p5_named_normalization_mutation_detected);
        assert!(report.table_3_spinorial_connection_solved);
        assert!(report.table_3_bosonic_connection_solved_from_d_spinorial_connection);
        assert!(report.spin_connections_solved);
        assert!(report.complete_t_alpha_e_gamma_from_geometry_inputs_implemented);
        assert!(report.j_one_from_geometry_inputs_implemented);
        assert!(report.j_plus_basis_change_implemented);
        assert!(report.convention_separated_w_assembly_implemented);
        assert!(report.linearized_w_coefficients_implemented);
        assert!(report.linearized_2001_w_coefficients_implemented);
        assert!(!report.j_plus_from_h_hat_implemented);
        assert!(!report.w_and_j_from_h_hat_implemented);
        assert!(report.polynomial_fx_api_implemented);
        assert_eq!(report.polynomial_fx_output_dimension, 4_719);
        assert!(report.leading_x2_all_six_channels_composed);
        assert!(!report.leading_x5_all_six_channels_composed);
        assert!(!report.first_momentum_fx_all_six_channels_composed);
        assert_eq!(
            report.individually_unexcluded_leading_k_ordinals,
            vec![3, 5]
        );
        assert!(report.linear_combination_survivor_space_solved);
        assert_eq!(report.leading_x2_joint_exact_rank, 7);
        assert_eq!(report.leading_x2_joint_exact_nullity, 5);
        assert_eq!(report.leading_x2_joint_kernel_basis.len(), 5);
        assert!(report.leading_x2_joint_kernel_proved_on_exact_source_streams);
        assert!(report.leading_fx_combined_kernel_proved_by_rank_sandwich);
        assert_eq!(report.leading_fx_k_solver_rank, 7);
        assert_eq!(report.leading_fx_k_solver_nullity, 5);
        assert!(report.leading_fx_k_solver_kernel_matches_source_relations);
        assert!(report.leading_fx_k_solver_mutation_detected);
        assert!(report.b5_to_cartesian_majorana_intertwiner_implemented);
        assert!(report.k_fag_target_key_retains_raw_vector_spinor_coordinates);
        assert!(report.k_fag_adapter_accepts_exact_b5_target_coordinate);
        assert!(!report.partial_fx_a_g_p_vanishing_established);
    }

    #[test]
    fn first_momentum_fx_operator_checkpoint_roundtrips_exact_rows() {
        let directory = std::env::temp_dir().join(format!(
            "adinkra-first-momentum-fx-checkpoint-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = fx_operator_checkpoint_path(&directory, 2, 17);
        let mut x2_rows = empty_fx_functional_rows();
        let mut x5_rows = empty_fx_functional_rows();
        x2_rows[3][7].real =
            num_rational::Ratio::new(num_bigint::BigInt::from(-17), num_bigint::BigInt::from(19));
        x5_rows[9][41].imaginary =
            num_rational::Ratio::new(num_bigint::BigInt::from(23), num_bigint::BigInt::from(29));
        let checkpoint = FirstMomentumFxOperatorCheckpoint {
            schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
            curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
            gauge_form_degree: 2,
            target_basis_ordinal: 0,
            operator_ordinal: 17,
            parameter_components_selected: vec![0],
            emitted_target_terms: 123,
            source_entries_unique: 17,
            source_entries_processed: 17,
            complete: true,
            x2_rows: encode_fx_rows(&x2_rows),
            x5_rows: encode_fx_rows(&x5_rows),
        };
        atomic_json(&path, &checkpoint).unwrap();
        let (loaded_x2, loaded_x5, emitted, unique, processed, complete) =
            load_fx_operator_checkpoint(&path, 2, 0, 17)
                .unwrap()
                .unwrap();
        assert_eq!(loaded_x2, x2_rows);
        assert_eq!(loaded_x5, x5_rows);
        assert_eq!(emitted, 123);
        assert_eq!(unique, 17);
        assert_eq!(processed, 17);
        assert!(complete);
        assert!(load_fx_operator_checkpoint(&path, 2, 0, 18).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn first_momentum_fx_derivative_templates_match_original_exact_path() {
        use crate::eleven_dimensional_k_fag_solver::{
            CurvatureVariationKey, ExactGaussian, MomentumMonomial, PhysicalCurvaturePolynomialApi,
            TargetVariationKey,
        };

        let templates = build_fx_derivative_templates(0, 0).unwrap();
        for derivative_weight in [0_usize, 7, 31] {
            let input_mask = u32::MAX ^ (1_u32 << derivative_weight);
            let input = TargetVariationKey {
                parameter_component: 3,
                target_coordinate: 0,
                target_vector_weight_index: Some(0),
                target_spinor_weight_index: Some(0),
                spinor_derivative_mask: input_mask,
                spinor_derivative_order: 31,
                momentum_monomial: MomentumMonomial::constant(),
            };
            let actual = CartesianPolynomialFxApi
                .apply_term(&input, &ExactGaussian::one())
                .unwrap()
                .into_iter()
                .collect::<BTreeMap<_, _>>();
            let wedge_sign = derivative_wedge_sign(input_mask, derivative_weight);
            let output_mask = input_mask | (1_u32 << derivative_weight);
            let expected = templates[derivative_weight]
                .iter()
                .map(|template| {
                    let mut coefficient = template.coefficient.clone();
                    if wedge_sign < 0 {
                        coefficient.real = -coefficient.real;
                        coefficient.imaginary = -coefficient.imaginary;
                    }
                    (
                        CurvatureVariationKey {
                            parameter_component: 3,
                            output_sector: template.output_sector.to_string(),
                            output_coordinate: template.output_coordinate,
                            spinor_derivative_mask: output_mask,
                            spinor_derivative_order: 32,
                            momentum_monomial: MomentumMonomial::constant(),
                        },
                        coefficient,
                    )
                })
                .collect::<BTreeMap<_, _>>();
            assert_eq!(actual, expected, "derivative weight {derivative_weight}");
        }
    }

    #[test]
    fn first_momentum_fx_projected_templates_match_mask_summed_hash_rows() {
        let templates = build_fx_derivative_templates(0, 0).unwrap();
        let projected = build_fx_projected_templates(0, 0).unwrap();
        let derivative_weight = 7;
        let momentum_axis = 3;
        let momentum =
            crate::eleven_dimensional_k_fag_solver::MomentumMonomial::variable(momentum_axis);
        let mut expected_x2 = empty_fx_functional_rows();
        let mut expected_x5 = empty_fx_functional_rows();
        for template in &templates[derivative_weight] {
            let base_hash = functional_hash_parts(
                0,
                template.output_coordinate,
                0,
                template.output_sector == "X2_11000",
                &momentum.exponents,
            );
            if template.output_sector == "X2_11000" {
                add_fx_functional_hashed_value(
                    &mut expected_x2,
                    12,
                    base_hash,
                    &template.coefficient,
                );
            } else {
                add_fx_functional_hashed_value(
                    &mut expected_x5,
                    12,
                    base_hash,
                    &template.coefficient,
                );
            }
        }
        let actual = &projected[derivative_weight][momentum_axis];
        assert!(
            expected_x2
                .iter()
                .zip(&actual.x2_rows)
                .all(|(row, value)| row[5] == *value)
        );
        assert!(
            expected_x5
                .iter()
                .zip(&actual.x5_rows)
                .all(|(row, value)| row[5] == *value)
        );
    }

    #[test]
    fn first_momentum_fx_i128_target_scaling_is_exact() {
        let target_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .unwrap()
            .ordinal;
        let dual = crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states()
            [target_ordinal]
            .clone();
        let common = fx_target_common_denominator(target_ordinal);
        assert!(common > 0);
        for term in dual.raw_terms {
            let residual = 37_i128;
            let value = num_rational::Ratio::new(
                num_bigint::BigInt::from(term.numerator) * num_bigint::BigInt::from(residual),
                num_bigint::BigInt::from(term.denominator),
            );
            let scaled = scale_primitive_ratio_to_i128(
                i128::from(term.numerator) * residual,
                term.denominator,
                common,
            )
            .unwrap();
            let reconstructed = num_rational::Ratio::new(
                num_bigint::BigInt::from(scaled),
                num_bigint::BigInt::from(common),
            );
            assert_eq!(reconstructed, value);
        }
    }

    #[test]
    fn first_momentum_fx_checkpoint_shape_validation_fails_closed() {
        let malformed = vec![vec![CheckpointGaussian {
            real_numerator: "0".to_string(),
            real_denominator: "1".to_string(),
            imaginary_numerator: "0".to_string(),
            imaginary_denominator: "1".to_string(),
        }]];
        assert!(decode_fx_rows(malformed).is_err());
        assert_eq!(
            fx_response_cache_limit().clamp(1, 16_384),
            fx_response_cache_limit()
        );
    }

    #[test]
    fn shared_operator_batch_rejects_invalid_degree_sets() {
        let directory = std::env::temp_dir().join(format!(
            "adinkra-first-momentum-fx-shared-invalid-{}",
            std::process::id()
        ));
        let report = directory.join("report.json");
        assert!(
            build_first_momentum_fx_shared_operator_batch_in(0, &[], &directory, None, &report,)
                .is_err()
        );
        assert!(build_first_momentum_fx_shared_operator_batch_in(
            0,
            &[0, 0],
            &directory,
            None,
            &report,
        )
        .is_err());
        assert!(
            build_first_momentum_fx_shared_operator_batch_in(0, &[6], &directory, None, &report,)
                .is_err()
        );
    }

    #[test]
    fn shared_operator_batch_resumes_production_schema_checkpoints() {
        let directory = std::env::temp_dir().join(format!(
            "adinkra-first-momentum-fx-shared-resume-{}",
            std::process::id()
        ));
        let target_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .unwrap()
            .ordinal;
        for (degree, emitted, unique) in [(0, 101_u64, 7_u64), (5, 103, 11)] {
            let checkpoint = FirstMomentumFxOperatorCheckpoint {
                schema_version: FIRST_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
                curvature_artifact_sha256: FIRST_MOMENTUM_FX_CURVATURE_SHA256.to_string(),
                gauge_form_degree: degree,
                target_basis_ordinal: target_ordinal,
                operator_ordinal: 0,
                parameter_components_selected: vec![0],
                emitted_target_terms: emitted,
                source_entries_unique: unique,
                source_entries_processed: unique,
                complete: true,
                x2_rows: encode_fx_rows(&empty_fx_functional_rows()),
                x5_rows: encode_fx_rows(&empty_fx_functional_rows()),
            };
            atomic_json(
                &fx_operator_checkpoint_path(&directory, degree, 0),
                &checkpoint,
            )
            .unwrap();
        }
        let report_path = directory.join("resume-report.json");
        let report = build_first_momentum_fx_shared_operator_batch_in(
            0,
            &[0, 5],
            &directory,
            None,
            &report_path,
        )
        .unwrap();
        assert_eq!(report.reused_batch_checkpoints, [0, 5]);
        assert!(report.shared_state_accounting.is_none());
        assert_eq!(report.emitted_target_terms_by_degree[0], 101);
        assert_eq!(report.emitted_target_terms_by_degree[5], 103);
        assert_eq!(report.unique_aggregated_source_entries_by_degree[0], 7);
        assert_eq!(report.unique_aggregated_source_entries_by_degree[5], 11);
        assert!(report.passed);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "writes deterministic validation artifacts"]
    fn write_validation_artifacts() {
        write_artifacts(
            Path::new("data/eleven_dimensional_physical_curvature.json"),
            Path::new("results/adynkra_11d_physical_curvature_validation.json"),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "executes 336 exact target-resolved first-momentum jobs"]
    fn write_first_momentum_fx_functional_validation_artifact() {
        write_first_momentum_fx_functional_artifact(Path::new(
            "results/adynkra_11d_first_momentum_physical_fx_functional.json",
        ))
        .unwrap();
    }

    #[test]
    #[ignore = "executes one exact target-resolved first-momentum channel selected by ADINKRA_FX_CHANNEL"]
    fn write_first_momentum_fx_functional_channel_checkpoint() {
        let gauge_form_degree = std::env::var("ADINKRA_FX_CHANNEL")
            .expect("ADINKRA_FX_CHANNEL is required")
            .parse::<usize>()
            .expect("ADINKRA_FX_CHANNEL must be an integer");
        assert!(gauge_form_degree < 6);
        let target_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .expect("target highest-weight state")
            .ordinal;
        build_first_momentum_fx_functional_channel(
            gauge_form_degree,
            target_ordinal,
            Path::new("results/eleven_dimensional_first_momentum_fx_checkpoints"),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "executes one exact target-resolved first-momentum operator selected by ADINKRA_FX_CHANNEL and ADINKRA_FX_OPERATOR"]
    fn write_first_momentum_fx_functional_operator_checkpoint() {
        let gauge_form_degree = std::env::var("ADINKRA_FX_CHANNEL")
            .expect("ADINKRA_FX_CHANNEL is required")
            .parse::<usize>()
            .expect("ADINKRA_FX_CHANNEL must be an integer");
        let operator_ordinal = std::env::var("ADINKRA_FX_OPERATOR")
            .expect("ADINKRA_FX_OPERATOR is required")
            .parse::<usize>()
            .expect("ADINKRA_FX_OPERATOR must be an integer");
        assert!(gauge_form_degree < 6);
        assert!(operator_ordinal < FIRST_MOMENTUM_FX_OPERATOR_COLUMNS);
        let target_ordinal = crate::eleven_dimensional_bridge::vector_spinor_target_basis_states()
            .into_iter()
            .find(|state| state.pbw_word_simple_roots.is_empty())
            .expect("target highest-weight state")
            .ordinal;
        let checkpoint_directory =
            Path::new("results/eleven_dimensional_first_momentum_fx_checkpoints");
        let checkpoint_path =
            fx_operator_checkpoint_path(checkpoint_directory, gauge_form_degree, operator_ordinal);
        let loaded = match load_fx_operator_checkpoint(
            &checkpoint_path,
            gauge_form_degree,
            target_ordinal,
            operator_ordinal,
        ) {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                quarantine_invalid_fx_checkpoint(&checkpoint_path, &error).unwrap();
                None
            }
        };
        if let Some((_, _, _, unique, _, true)) = loaded.as_ref() {
            log_fx_progress(
                checkpoint_directory,
                "operator-complete",
                Some(gauge_form_degree),
                Some(operator_ordinal),
                Some(*unique),
                true,
                Some(0.0),
                0,
            )
            .unwrap();
            return;
        }
        let resumed_entries = loaded.as_ref().map(|value| value.4);
        log_fx_progress(
            checkpoint_directory,
            if resumed_entries.unwrap_or(0) == 0 {
                "operator-start"
            } else {
                "operator-resume"
            },
            Some(gauge_form_degree),
            Some(operator_ordinal),
            resumed_entries,
            false,
            None,
            0,
        )
        .unwrap();
        let started = Instant::now();
        let mut response_cache = FxResponseCache::new();
        let (checkpoint, _, _) = build_first_momentum_fx_operator_checkpoint(
            gauge_form_degree,
            target_ordinal,
            operator_ordinal,
            &mut response_cache,
            &checkpoint_path,
            checkpoint_directory,
            loaded,
        )
        .unwrap();
        atomic_json(&checkpoint_path, &checkpoint).unwrap();
        log_fx_progress(
            checkpoint_directory,
            "operator-complete",
            Some(gauge_form_degree),
            Some(operator_ordinal),
            Some(checkpoint.source_entries_unique),
            false,
            Some(started.elapsed().as_secs_f64()),
            response_cache.len(),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "executes one operator-major exact batch selected by ADINKRA_FX_OPERATOR and optional ADINKRA_FX_SHARED_DEGREES"]
    fn write_first_momentum_fx_shared_operator_batch() {
        let operator_ordinal = std::env::var("ADINKRA_FX_OPERATOR")
            .expect("ADINKRA_FX_OPERATOR is required")
            .parse::<usize>()
            .expect("ADINKRA_FX_OPERATOR must be an integer");
        let degrees = std::env::var("ADINKRA_FX_SHARED_DEGREES")
            .unwrap_or_else(|_| "0,1,2,3,4,5".to_string())
            .split(',')
            .map(|degree| {
                degree
                    .parse::<usize>()
                    .expect("invalid shared gauge degree")
            })
            .collect::<Vec<_>>();
        let checkpoint_directory = std::env::var("ADINKRA_FX_SHARED_CHECKPOINT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from("results/eleven_dimensional_first_momentum_fx_shared_checkpoints")
            });
        let report_path = checkpoint_directory.join(format!(
            "operator-{operator_ordinal:02}-shared-benchmark.json"
        ));
        let report = build_first_momentum_fx_shared_operator_batch_in(
            operator_ordinal,
            &degrees,
            &checkpoint_directory,
            Some(Path::new(
                "results/eleven_dimensional_first_momentum_fx_checkpoints",
            )),
            &report_path,
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.completed_gauge_form_degrees, degrees);
    }

    #[test]
    #[ignore = "executes all 56 exact operator-major batches across all six gauge forms"]
    fn write_first_momentum_fx_shared_all_six_artifact() {
        write_first_momentum_fx_functional_artifact_shared_with_checkpoints(
            Path::new("results/adynkra_11d_first_momentum_physical_fx_functional_shared.json"),
            Path::new("results/eleven_dimensional_first_momentum_fx_shared_checkpoints"),
            Some(Path::new(
                "results/eleven_dimensional_first_momentum_fx_checkpoints",
            )),
        )
        .unwrap();
    }

    #[test]
    fn generic_eq1_p_form_injection_preserves_p2_and_odd_i_factors() {
        let mut d_two = BTreeMap::new();
        d_two.insert(9 * 55 + 17, ExactQi::from_integer(3));
        assert_eq!(
            inject_d_holonomy_form_into_d_delta(2, &d_two),
            inject_d_lorentz_compensator_into_d_delta(&d_two)
        );

        let mut one = BTreeMap::new();
        one.insert(4, ExactQi::from_integer(2));
        let delta = inject_holonomy_form_into_delta(1, &one);
        assert!(!delta.is_empty());
        assert!(
            delta
                .values()
                .all(|value| value.real == r(0) && value.imaginary.denom() == &1)
        );
    }

    #[test]
    fn eq14_mixed_spinor_anholonomy_keeps_all_four_source_terms() {
        let mut input = Eq14MixedSpinorAnholonomyInput::default();
        input.d_d_delta.insert(0, ExactQi::one());
        input.p_delta.insert(0, ExactQi::from_integer(2));
        input.p_scale.insert(0, ExactQi::from_integer(3));
        input.d_d_scale.insert(0, ExactQi::from_integer(5));
        let complete = apply_eq14_mixed_spinor_anholonomy(&input);
        assert!(!complete.is_empty());

        let sectors = [
            Eq14MixedSpinorAnholonomyInput {
                d_d_delta: input.d_d_delta.clone(),
                ..Eq14MixedSpinorAnholonomyInput::default()
            },
            Eq14MixedSpinorAnholonomyInput {
                p_delta: input.p_delta.clone(),
                ..Eq14MixedSpinorAnholonomyInput::default()
            },
            Eq14MixedSpinorAnholonomyInput {
                p_scale: input.p_scale.clone(),
                ..Eq14MixedSpinorAnholonomyInput::default()
            },
            Eq14MixedSpinorAnholonomyInput {
                d_d_scale: input.d_d_scale.clone(),
                ..Eq14MixedSpinorAnholonomyInput::default()
            },
        ];
        let mut recombined = BTreeMap::new();
        for sector in sectors {
            for (index, value) in apply_eq14_mixed_spinor_anholonomy(&sector) {
                add_sparse(&mut recombined, index, value);
            }
        }
        assert_eq!(complete, recombined);
    }
}
