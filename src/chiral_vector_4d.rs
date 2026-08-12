//! Exact four-dimensional chiral-vector positive control from arXiv:1405.0048.
//!
//! This module retains the vector potential and verifies Eqs. (32)-(38) before
//! applying the temporal gauge and one-dimensional field map in Eqs. (40)-(41).

#![allow(clippy::needless_range_loop)]

use num_rational::Ratio;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SPINORS: usize = 4;
const DIM: usize = 4;

pub(crate) type Rat = Ratio<i64>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GaussianRational {
    pub(crate) real: Rat,
    pub(crate) imag: Rat,
}

impl GaussianRational {
    pub(crate) fn new(real: i64, imag: i64) -> Self {
        Self {
            real: Ratio::from_integer(real),
            imag: Ratio::from_integer(imag),
        }
    }

    pub(crate) fn from_ratio(real: Rat, imag: Rat) -> Self {
        Self { real, imag }
    }

    pub(crate) fn add_assign(&mut self, other: &Self) {
        self.real += other.real;
        self.imag += other.imag;
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        Self::from_ratio(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.real == Ratio::from_integer(0) && self.imag == Ratio::from_integer(0)
    }
}

pub(crate) type Matrix4 = [[GaussianRational; 4]; 4];

pub(crate) fn zero_matrix() -> Matrix4 {
    std::array::from_fn(|_| std::array::from_fn(|_| GaussianRational::default()))
}

fn identity2() -> [[GaussianRational; 2]; 2] {
    [
        [GaussianRational::new(1, 0), GaussianRational::new(0, 0)],
        [GaussianRational::new(0, 0), GaussianRational::new(1, 0)],
    ]
}

pub(crate) fn pauli(index: usize) -> [[GaussianRational; 2]; 2] {
    match index {
        1 => [
            [GaussianRational::new(0, 0), GaussianRational::new(1, 0)],
            [GaussianRational::new(1, 0), GaussianRational::new(0, 0)],
        ],
        2 => [
            [GaussianRational::new(0, 0), GaussianRational::new(0, -1)],
            [GaussianRational::new(0, 1), GaussianRational::new(0, 0)],
        ],
        3 => [
            [GaussianRational::new(1, 0), GaussianRational::new(0, 0)],
            [GaussianRational::new(0, 0), GaussianRational::new(-1, 0)],
        ],
        _ => panic!("Pauli index must be 1, 2, or 3"),
    }
}

fn kronecker(left: [[GaussianRational; 2]; 2], right: [[GaussianRational; 2]; 2]) -> Matrix4 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| left[row / 2][column / 2].mul(&right[row % 2][column % 2]))
    })
}

pub(crate) fn matrix_scale(matrix: &Matrix4, coefficient: GaussianRational) -> Matrix4 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[row][column].mul(&coefficient)))
}

pub(crate) fn matrix_add(left: &Matrix4, right: &Matrix4) -> Matrix4 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let mut value = left[row][column];
            value.add_assign(&right[row][column]);
            value
        })
    })
}

pub(crate) fn matrix_mul(left: &Matrix4, right: &Matrix4) -> Matrix4 {
    let mut result = zero_matrix();
    for row in 0..4 {
        for inner in 0..4 {
            for column in 0..4 {
                let value = left[row][inner].mul(&right[inner][column]);
                result[row][column].add_assign(&value);
            }
        }
    }
    result
}

#[derive(Clone)]
pub(crate) struct Clifford4D {
    pub(crate) gamma_up: [Matrix4; DIM],
    pub(crate) gamma_down: [Matrix4; DIM],
    pub(crate) gamma5: Matrix4,
    pub(crate) charge_conjugation: Matrix4,
}

impl Clifford4D {
    pub(crate) fn build() -> Self {
        let identity = identity2();
        let sigma1 = pauli(1);
        let sigma2 = pauli(2);
        let sigma3 = pauli(3);
        let gamma_up = [
            matrix_scale(&kronecker(sigma3, sigma2), GaussianRational::new(0, 1)),
            kronecker(identity, sigma1),
            kronecker(sigma2, sigma2),
            kronecker(identity, sigma3),
        ];
        let gamma_down = std::array::from_fn(|mu| {
            if mu == 0 {
                matrix_scale(&gamma_up[mu], GaussianRational::new(-1, 0))
            } else {
                gamma_up[mu]
            }
        });
        let gamma5 = matrix_scale(&kronecker(sigma1, sigma2), GaussianRational::new(-1, 0));
        let charge_conjugation =
            matrix_scale(&kronecker(sigma3, sigma2), GaussianRational::new(0, -1));
        Self {
            gamma_up,
            gamma_down,
            gamma5,
            charge_conjugation,
        }
    }

    pub(crate) fn lower_spinors(&self, matrix: &Matrix4) -> Matrix4 {
        matrix_mul(matrix, &self.charge_conjugation)
    }

    pub(crate) fn gamma5_gamma_up(&self, mu: usize) -> Matrix4 {
        matrix_mul(&self.gamma5, &self.gamma_up[mu])
    }

    pub(crate) fn commutator_up(&self, mu: usize, nu: usize) -> Matrix4 {
        matrix_add(
            &matrix_mul(&self.gamma_up[mu], &self.gamma_up[nu]),
            &matrix_scale(
                &matrix_mul(&self.gamma_up[nu], &self.gamma_up[mu]),
                GaussianRational::new(-1, 0),
            ),
        )
    }

    pub(crate) fn verifies_source_conventions(&self) -> bool {
        let identity = std::array::from_fn(|row| {
            std::array::from_fn(|column| GaussianRational::new(i64::from(row == column), 0))
        });
        for mu in 0..DIM {
            for nu in 0..DIM {
                let lhs = matrix_add(
                    &matrix_mul(&self.gamma_up[mu], &self.gamma_up[nu]),
                    &matrix_mul(&self.gamma_up[nu], &self.gamma_up[mu]),
                );
                let eta = if mu == nu {
                    if mu == 0 { -2 } else { 2 }
                } else {
                    0
                };
                if lhs != matrix_scale(&identity, GaussianRational::new(eta, 0)) {
                    return false;
                }
            }
        }
        matrix_mul(&self.gamma5, &self.gamma5) == identity
            && matrix_scale(&self.gamma_up[0], GaussianRational::new(-1, 0))
                == self.charge_conjugation
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
enum Field {
    ScalarA,
    PseudoscalarB,
    AuxiliaryF,
    AuxiliaryG,
    Vector(usize),
    AuxiliaryD,
    Psi(usize),
    Lambda(usize),
}

impl Field {
    fn all() -> Vec<Self> {
        let mut fields = vec![
            Self::ScalarA,
            Self::PseudoscalarB,
            Self::AuxiliaryF,
            Self::AuxiliaryG,
        ];
        fields.extend((0..DIM).map(Self::Vector));
        fields.push(Self::AuxiliaryD);
        fields.extend((0..SPINORS).map(Self::Psi));
        fields.extend((0..SPINORS).map(Self::Lambda));
        fields
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Atom {
    field: Field,
    derivatives: [u8; DIM],
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Polynomial(BTreeMap<Atom, GaussianRational>);

impl Polynomial {
    fn atom(field: Field) -> Self {
        Self(BTreeMap::from([(
            Atom {
                field,
                derivatives: [0; DIM],
            },
            GaussianRational::new(1, 0),
        )]))
    }

    fn add_term(&mut self, field: Field, derivatives: [u8; DIM], coefficient: GaussianRational) {
        if coefficient.is_zero() {
            return;
        }
        let atom = Atom { field, derivatives };
        let entry = self.0.entry(atom.clone()).or_default();
        entry.add_assign(&coefficient);
        if entry.is_zero() {
            self.0.remove(&atom);
        }
    }

    fn add_scaled(&mut self, other: &Self, coefficient: &GaussianRational) {
        for (atom, value) in &other.0 {
            self.add_term(atom.field, atom.derivatives, value.mul(coefficient));
        }
    }

    fn derivative(&self, mu: usize) -> Self {
        let mut result = Self::default();
        for (atom, coefficient) in &self.0 {
            let mut derivatives = atom.derivatives;
            derivatives[mu] += 1;
            result.add_term(atom.field, derivatives, *coefficient);
        }
        result
    }
}

#[derive(Clone, Copy)]
struct Charge {
    supersymmetry: usize,
    spinor: usize,
}

fn field_strength(mu: usize, nu: usize) -> Polynomial {
    let mut result = Polynomial::atom(Field::Vector(nu)).derivative(mu);
    result.add_scaled(
        &Polynomial::atom(Field::Vector(mu)).derivative(nu),
        &GaussianRational::new(-1, 0),
    );
    result
}

fn add_matrix_row(
    result: &mut Polynomial,
    matrix: &Matrix4,
    row: usize,
    field: fn(usize) -> Field,
    derivative: Option<usize>,
    scale: GaussianRational,
) {
    for component in 0..SPINORS {
        let mut polynomial = Polynomial::atom(field(component));
        if let Some(mu) = derivative {
            polynomial = polynomial.derivative(mu);
        }
        result.add_scaled(&polynomial, &matrix[row][component].mul(&scale));
    }
}

fn delta(charge: Charge, field: Field, clifford: &Clifford4D) -> Polynomial {
    let a = charge.spinor;
    let second = charge.supersymmetry == 1;
    let fermion = if second { Field::Lambda } else { Field::Psi };
    let other_fermion = if second { Field::Psi } else { Field::Lambda };
    match field {
        Field::ScalarA => Polynomial::atom(fermion(a)),
        Field::PseudoscalarB => {
            let mut result = Polynomial::default();
            add_matrix_row(
                &mut result,
                &clifford.gamma5,
                a,
                fermion,
                None,
                GaussianRational::new(0, 1),
            );
            result
        }
        Field::AuxiliaryF => {
            let mut result = Polynomial::default();
            for mu in 0..DIM {
                add_matrix_row(
                    &mut result,
                    &clifford.gamma_up[mu],
                    a,
                    fermion,
                    Some(mu),
                    GaussianRational::new(1, 0),
                );
            }
            result
        }
        Field::AuxiliaryG => {
            let mut result = Polynomial::default();
            for mu in 0..DIM {
                add_matrix_row(
                    &mut result,
                    &clifford.gamma5_gamma_up(mu),
                    a,
                    fermion,
                    Some(mu),
                    GaussianRational::new(0, if second { -1 } else { 1 }),
                );
            }
            result
        }
        Field::Vector(mu) => {
            let mut result = Polynomial::default();
            add_matrix_row(
                &mut result,
                &clifford.gamma_down[mu],
                a,
                other_fermion,
                None,
                GaussianRational::new(if second { -1 } else { 1 }, 0),
            );
            result
        }
        Field::AuxiliaryD => {
            let mut result = Polynomial::default();
            for mu in 0..DIM {
                add_matrix_row(
                    &mut result,
                    &clifford.gamma5_gamma_up(mu),
                    a,
                    other_fermion,
                    Some(mu),
                    GaussianRational::new(0, 1),
                );
            }
            result
        }
        Field::Psi(b) if !second => chiral_fermion_delta(a, b, false, clifford),
        Field::Lambda(b) if second => chiral_fermion_delta(a, b, true, clifford),
        Field::Lambda(b) if !second => vector_fermion_delta(a, b, false, clifford),
        Field::Psi(b) if second => vector_fermion_delta(a, b, true, clifford),
        _ => unreachable!(),
    }
}

fn chiral_fermion_delta(a: usize, b: usize, second: bool, clifford: &Clifford4D) -> Polynomial {
    let mut result = Polynomial::default();
    for mu in 0..DIM {
        let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
        result.add_scaled(
            &Polynomial::atom(Field::ScalarA).derivative(mu),
            &gamma[a][b].mul(&GaussianRational::new(0, 1)),
        );
        let gamma5_gamma = clifford.lower_spinors(&clifford.gamma5_gamma_up(mu));
        result.add_scaled(
            &Polynomial::atom(Field::PseudoscalarB).derivative(mu),
            &gamma5_gamma[a][b].mul(&GaussianRational::new(-1, 0)),
        );
    }
    result.add_scaled(
        &Polynomial::atom(Field::AuxiliaryF),
        &clifford.charge_conjugation[a][b].mul(&GaussianRational::new(0, -1)),
    );
    let gamma5 = clifford.lower_spinors(&clifford.gamma5);
    result.add_scaled(
        &Polynomial::atom(Field::AuxiliaryG),
        &gamma5[a][b].mul(&GaussianRational::new(if second { -1 } else { 1 }, 0)),
    );
    result
}

fn vector_fermion_delta(a: usize, b: usize, second: bool, clifford: &Clifford4D) -> Polynomial {
    let mut result = Polynomial::default();
    let scale = GaussianRational::from_ratio(
        Ratio::from_integer(0),
        Ratio::new(if second { 1 } else { -1 }, 2),
    );
    for mu in 0..DIM {
        for nu in (mu + 1)..DIM {
            let commutator = clifford.lower_spinors(&clifford.commutator_up(mu, nu));
            result.add_scaled(&field_strength(mu, nu), &commutator[a][b].mul(&scale));
        }
    }
    let gamma5 = clifford.lower_spinors(&clifford.gamma5);
    result.add_scaled(&Polynomial::atom(Field::AuxiliaryD), &gamma5[a][b]);
    result
}

fn apply_delta(charge: Charge, polynomial: &Polynomial, clifford: &Clifford4D) -> Polynomial {
    let mut result = Polynomial::default();
    for (atom, coefficient) in &polynomial.0 {
        let mut transformed = delta(charge, atom.field, clifford);
        for mu in 0..DIM {
            for _ in 0..atom.derivatives[mu] {
                transformed = transformed.derivative(mu);
            }
        }
        result.add_scaled(&transformed, coefficient);
    }
    result
}

fn anticommutator(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Polynomial {
    let mut result = apply_delta(left, &delta(right, field, clifford), clifford);
    result.add_scaled(
        &apply_delta(right, &delta(left, field, clifford), clifford),
        &GaussianRational::new(1, 0),
    );
    result
}

fn published_closure(
    left: Charge,
    right: Charge,
    field: Field,
    clifford: &Clifford4D,
) -> Polynomial {
    let mut result = Polynomial::default();
    if left.supersymmetry == right.supersymmetry {
        for mu in 0..DIM {
            let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
            result.add_scaled(
                &Polynomial::atom(field).derivative(mu),
                &gamma[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
            );
        }
        if let Field::Vector(nu) = field {
            result = Polynomial::default();
            for mu in 0..DIM {
                let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
                result.add_scaled(
                    &field_strength(mu, nu),
                    &gamma[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
                );
            }
        }
    }
    if let Field::Vector(nu) = field {
        let sigma2 = pauli(2);
        let ij = sigma2[left.supersymmetry][right.supersymmetry];
        let c = &clifford.charge_conjugation[left.spinor][right.spinor];
        result.add_scaled(
            &Polynomial::atom(Field::ScalarA).derivative(nu),
            &ij.mul(c).mul(&GaussianRational::new(-2, 0)),
        );
        let gamma5 = clifford.lower_spinors(&clifford.gamma5);
        result.add_scaled(
            &Polynomial::atom(Field::PseudoscalarB).derivative(nu),
            &ij.mul(&gamma5[left.spinor][right.spinor])
                .mul(&GaussianRational::new(0, -2)),
        );
    }
    result
}

fn real_unit(value: &GaussianRational) -> i8 {
    assert_eq!(value.imag, Ratio::from_integer(0));
    assert_eq!(*value.real.denom(), 1);
    i8::try_from(*value.real.numer()).expect("worldline linkage coefficient fits i8")
}

fn worldline_l_matrices(clifford: &Clifford4D) -> [[[i8; 8]; 8]; 8] {
    let mut output = [[[0_i8; 8]; 8]; 8];
    let identity = std::array::from_fn(|row| {
        std::array::from_fn(|column| GaussianRational::new(i64::from(row == column), 0))
    });
    let i_gamma5 = matrix_scale(&clifford.gamma5, GaussianRational::new(0, 1));
    let i_gamma5_gamma0 = matrix_scale(&clifford.gamma5_gamma_up(0), GaussianRational::new(0, 1));
    for spinor in 0..SPINORS {
        let first_rows = [
            (&identity, 0_usize),
            (&i_gamma5, 0),
            (&clifford.gamma_up[0], 0),
            (&i_gamma5_gamma0, 0),
            (&clifford.gamma_down[1], 4),
            (&clifford.gamma_down[2], 4),
            (&clifford.gamma_down[3], 4),
            (&i_gamma5_gamma0, 4),
        ];
        let second_rows = [
            (&identity, 4_usize, 1_i8),
            (&i_gamma5, 4, 1),
            (&clifford.gamma_up[0], 4, 1),
            (&i_gamma5_gamma0, 4, -1),
            (&clifford.gamma_down[1], 0, -1),
            (&clifford.gamma_down[2], 0, -1),
            (&clifford.gamma_down[3], 0, -1),
            (&i_gamma5_gamma0, 0, 1),
        ];
        for row in 0..8 {
            let (matrix, offset) = first_rows[row];
            for component in 0..SPINORS {
                output[spinor][row][offset + component] = real_unit(&matrix[spinor][component]);
            }
            let (matrix, offset, sign) = second_rows[row];
            for component in 0..SPINORS {
                output[4 + spinor][row][offset + component] =
                    sign * real_unit(&matrix[spinor][component]);
            }
        }
    }
    output
}

fn published_cv_l_matrices() -> [[[i8; 8]; 8]; 8] {
    use crate::permutahedron_fixtures::S8_REPRESENTATION_OCTETS;
    use crate::permutahedron_s8_signed_recursion::build_rep;
    use crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS;

    let rep = build_rep(
        &S8_REPRESENTATION_OCTETS[2].permutations,
        &S8_BASE_BOOLEAN_FACTORS[2],
    );
    std::array::from_fn(|color| {
        std::array::from_fn(|row| {
            let mut result = [0_i8; 8];
            result[usize::from(rep.l_matrices[color].perm[row])] = rep.l_matrices[color].sign[row];
            result
        })
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct ChiralVectorReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub gamma_conventions_verified: bool,
    pub component_fields: usize,
    pub charge_pairs: usize,
    pub component_relations_checked: usize,
    pub nongauge_field_relations_checked: usize,
    pub vector_potential_relations_checked: usize,
    pub residual_relations: usize,
    pub residual_terms: usize,
    pub vector_gauge_term_is_required: bool,
    pub reduced_l_matrix_entries_checked: usize,
    pub reduced_l_matrix_residual_entries: usize,
    pub exact_cv_anchor_recovered: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChiralVectorArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub report: ChiralVectorReport,
    pub reduced_l_matrices: [[[i8; 8]; 8]; 8],
    pub published_cv_l_matrices: [[[i8; 8]; 8]; 8],
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub source: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
}

pub fn verify() -> ChiralVectorReport {
    let clifford = Clifford4D::build();
    let fields = Field::all();
    let charges: Vec<_> = (0..2)
        .flat_map(|supersymmetry| {
            (0..SPINORS).map(move |spinor| Charge {
                supersymmetry,
                spinor,
            })
        })
        .collect();
    let mut checked = 0;
    let mut residual_relations = 0;
    let mut residual_terms = 0;
    for left in 0..charges.len() {
        for right in left..charges.len() {
            for &field in &fields {
                checked += 1;
                let actual = anticommutator(charges[left], charges[right], field, &clifford);
                let expected = published_closure(charges[left], charges[right], field, &clifford);
                if actual != expected {
                    residual_relations += 1;
                    let mut residual = actual;
                    residual.add_scaled(&expected, &GaussianRational::new(-1, 0));
                    residual_terms += residual.0.len();
                }
            }
        }
    }
    let gamma_verified = clifford.verifies_source_conventions();
    let gauge_required = (0..SPINORS).any(|left_spinor| {
        (0..SPINORS).any(|right_spinor| {
            (0..DIM).any(|nu| {
                !published_closure(
                    Charge {
                        supersymmetry: 0,
                        spinor: left_spinor,
                    },
                    Charge {
                        supersymmetry: 1,
                        spinor: right_spinor,
                    },
                    Field::Vector(nu),
                    &clifford,
                )
                .0
                .is_empty()
            })
        })
    });
    let reduced = worldline_l_matrices(&clifford);
    let published = published_cv_l_matrices();
    let reduced_residuals = reduced
        .iter()
        .flatten()
        .flatten()
        .zip(published.iter().flatten().flatten())
        .filter(|(left, right)| left != right)
        .count();
    ChiralVectorReport {
        schema_version: "chiral-vector-4d-closure-v1",
        source: "arXiv:1405.0048 Eqs. (32)-(41), with gamma conventions from arXiv:0902.3830 Appendix A",
        gamma_conventions_verified: gamma_verified,
        component_fields: fields.len(),
        charge_pairs: charges.len() * (charges.len() + 1) / 2,
        component_relations_checked: checked,
        nongauge_field_relations_checked: 13 * 36,
        vector_potential_relations_checked: 4 * 36,
        residual_relations,
        residual_terms,
        vector_gauge_term_is_required: gauge_required,
        reduced_l_matrix_entries_checked: 8 * 8 * 8,
        reduced_l_matrix_residual_entries: reduced_residuals,
        exact_cv_anchor_recovered: reduced_residuals == 0,
        passed: gamma_verified
            && residual_relations == 0
            && gauge_required
            && reduced_residuals == 0,
        boundary: "This reproduces the published linear abelian chiral-vector component algebra, including closure on the vector potential modulo the printed gauge term. It is not a new four-dimensional construction.",
    }
}

pub fn build() -> ChiralVectorArtifact {
    let clifford = Clifford4D::build();
    ChiralVectorArtifact {
        schema_version: "chiral-vector-4d-artifact-v1",
        title: "Exact four-dimensional chiral-vector closure and worldline reduction",
        sources: vec![
            SourceRecord {
                source: "arXiv:1405.0048",
                locator: "Eqs. (32)-(41) and Appendix B",
                role: "component transformations, closure modulo gauge, temporal gauge, reduction map, and published L matrices",
            },
            SourceRecord {
                source: "arXiv:0902.3830",
                locator: "Appendix A, Eqs. (A.3)-(A.11)",
                role: "mostly-plus Majorana gamma matrices and charge-conjugation convention",
            },
        ],
        report: verify(),
        reduced_l_matrices: worldline_l_matrices(&clifford),
        published_cv_l_matrices: published_cv_l_matrices(),
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ChiralVectorReport {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create chiral-vector data")),
        &artifact,
    )
    .expect("write chiral-vector data");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create chiral-vector validation")),
        &artifact.report,
    )
    .expect("write chiral-vector validation");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_gamma_conventions_are_exact() {
        assert!(Clifford4D::build().verifies_source_conventions());
    }

    #[test]
    fn complete_chiral_vector_component_algebra_closes() {
        let report = verify();
        assert_eq!(report.component_fields, 17);
        assert_eq!(report.charge_pairs, 36);
        assert_eq!(report.component_relations_checked, 612);
        assert_eq!(report.nongauge_field_relations_checked, 468);
        assert_eq!(report.vector_potential_relations_checked, 144);
        assert_eq!(report.residual_relations, 0);
        assert_eq!(report.residual_terms, 0);
        assert!(report.vector_gauge_term_is_required);
        assert_eq!(report.reduced_l_matrix_entries_checked, 512);
        assert_eq!(report.reduced_l_matrix_residual_entries, 0);
        assert!(report.exact_cv_anchor_recovered);
        assert!(report.passed);
    }
}
