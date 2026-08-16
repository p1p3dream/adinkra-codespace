//! Exact four-dimensional chiral-tensor positive control from arXiv:1405.0048.
//!
//! The two-form potential is retained through the component closure check.
//! Only afterward are temporal gauge and the field map of Eq. (53) applied.

#![allow(clippy::needless_range_loop)]

use crate::chiral_vector_4d::{
    Clifford4D, GaussianRational, Matrix4, matrix_mul, matrix_scale, pauli,
};
use crate::higher_dimensional_canonical::{
    BianchiIdentity as CanonicalBianchi, Component as CanonicalComponent,
    ComponentRole as CanonicalRole, DerivativeMonomial as CanonicalDerivative,
    GaugeTerm as CanonicalGaugeTerm, GaussianRational as CanonicalCoefficient,
    LinearTerm as CanonicalLinearTerm, LinkageTerm as CanonicalLinkage, LorentzRep,
    PhysicalFingerprint as CanonicalPhysicalFingerprint, Reality, Statistics,
    Supercharge as CanonicalSupercharge,
};
use crate::higher_dimensional_fingerprint::{
    DerivativeOperatorFingerprint, GaugeFingerprint, MultipletFingerprint, sha256_lines,
};
use crate::lr_matrix::AdinkraRep;
use num_rational::Ratio;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const SPINORS: usize = 4;
const DIM: usize = 4;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
enum Field {
    ScalarA,
    PseudoscalarB,
    AuxiliaryF,
    AuxiliaryG,
    TensorScalar,
    TwoForm(usize, usize),
    Psi(usize),
    Chi(usize),
}

impl Field {
    fn all() -> Vec<Self> {
        let mut fields = vec![
            Self::ScalarA,
            Self::PseudoscalarB,
            Self::AuxiliaryF,
            Self::AuxiliaryG,
            Self::TensorScalar,
        ];
        for mu in 0..DIM {
            for nu in (mu + 1)..DIM {
                fields.push(Self::TwoForm(mu, nu));
            }
        }
        fields.extend((0..SPINORS).map(Self::Psi));
        fields.extend((0..SPINORS).map(Self::Chi));
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

    fn add_scaled(&mut self, other: &Self, coefficient: GaussianRational) {
        for (atom, value) in &other.0 {
            self.add_term(atom.field, atom.derivatives, value.mul(&coefficient));
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

fn metric_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
}

fn epsilon_lower(indices: [usize; 4]) -> i64 {
    if (0..4).any(|left| ((left + 1)..4).any(|right| indices[left] == indices[right])) {
        return 0;
    }
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|&(left, right)| indices[left] > indices[right])
        .count();
    if inversions.is_multiple_of(2) { -1 } else { 1 }
}

fn two_form(mu: usize, nu: usize) -> (Field, i64) {
    if mu < nu {
        (Field::TwoForm(mu, nu), 1)
    } else {
        (Field::TwoForm(nu, mu), -1)
    }
}

fn two_form_polynomial(mu: usize, nu: usize) -> Polynomial {
    if mu == nu {
        return Polynomial::default();
    }
    let (field, sign) = two_form(mu, nu);
    let mut result = Polynomial::default();
    result.add_scaled(&Polynomial::atom(field), GaussianRational::new(sign, 0));
    result
}

fn field_strength(alpha: usize, mu: usize, nu: usize) -> Polynomial {
    let mut result = two_form_polynomial(mu, nu).derivative(alpha);
    result.add_scaled(
        &two_form_polynomial(nu, alpha).derivative(mu),
        GaussianRational::new(1, 0),
    );
    result.add_scaled(
        &two_form_polynomial(alpha, mu).derivative(nu),
        GaussianRational::new(1, 0),
    );
    result
}

fn lower_commutator(clifford: &Clifford4D, mu: usize, nu: usize) -> Matrix4 {
    let first = matrix_mul(&clifford.gamma_down[mu], &clifford.gamma_down[nu]);
    let second = matrix_mul(&clifford.gamma_down[nu], &clifford.gamma_down[mu]);
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let mut value = first[row][column];
            value.add_assign(&second[row][column].mul(&GaussianRational::new(-1, 0)));
            value
        })
    })
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
        let mut term = Polynomial::atom(field(component));
        if let Some(mu) = derivative {
            term = term.derivative(mu);
        }
        result.add_scaled(&term, matrix[row][component].mul(&scale));
    }
}

fn chiral_fermion_delta(a: usize, b: usize, second: bool, clifford: &Clifford4D) -> Polynomial {
    let mut result = Polynomial::default();
    for mu in 0..DIM {
        let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
        result.add_scaled(
            &Polynomial::atom(Field::ScalarA).derivative(mu),
            gamma[a][b].mul(&GaussianRational::new(0, if second { -1 } else { 1 })),
        );
        let gamma5_gamma = clifford.lower_spinors(&clifford.gamma5_gamma_up(mu));
        result.add_scaled(
            &Polynomial::atom(Field::PseudoscalarB).derivative(mu),
            gamma5_gamma[a][b].mul(&GaussianRational::new(-1, 0)),
        );
    }
    result.add_scaled(
        &Polynomial::atom(Field::AuxiliaryF),
        clifford.charge_conjugation[a][b].mul(&GaussianRational::new(0, -1)),
    );
    let gamma5 = clifford.lower_spinors(&clifford.gamma5);
    result.add_scaled(
        &Polynomial::atom(Field::AuxiliaryG),
        gamma5[a][b].mul(&GaussianRational::new(1, 0)),
    );
    result
}

fn tensor_fermion_delta(a: usize, b: usize, dual_sign: i64, clifford: &Clifford4D) -> Polynomial {
    let mut result = Polynomial::default();
    for mu in 0..DIM {
        let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
        result.add_scaled(
            &Polynomial::atom(Field::TensorScalar).derivative(mu),
            gamma[a][b].mul(&GaussianRational::new(0, 1)),
        );
        let gamma5_gamma = clifford.lower_spinors(&clifford.gamma5_gamma_up(mu));
        for rho in 0..DIM {
            for sigma in 0..DIM {
                for tau in (sigma + 1)..DIM {
                    let epsilon = epsilon_lower([mu, rho, sigma, tau]);
                    if epsilon == 0 {
                        continue;
                    }
                    let raised = metric_sign(rho) * metric_sign(sigma) * metric_sign(tau);
                    let coefficient = dual_sign * 2 * epsilon * raised;
                    let field = Field::TwoForm(sigma, tau);
                    result.add_scaled(
                        &Polynomial::atom(field).derivative(rho),
                        gamma5_gamma[a][b].mul(&GaussianRational::new(coefficient, 0)),
                    );
                }
            }
        }
    }
    result
}

fn delta(charge: Charge, field: Field, clifford: &Clifford4D) -> Polynomial {
    let a = charge.spinor;
    let second = charge.supersymmetry == 1;
    match field {
        Field::ScalarA => {
            let field = if second { Field::Chi(a) } else { Field::Psi(a) };
            let mut result = Polynomial::default();
            result.add_scaled(
                &Polynomial::atom(field),
                GaussianRational::new(if second { -1 } else { 1 }, 0),
            );
            result
        }
        Field::PseudoscalarB => {
            let mut result = Polynomial::default();
            add_matrix_row(
                &mut result,
                &clifford.gamma5,
                a,
                if second { Field::Chi } else { Field::Psi },
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
                    if second { Field::Chi } else { Field::Psi },
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
                    if second { Field::Chi } else { Field::Psi },
                    Some(mu),
                    GaussianRational::new(0, 1),
                );
            }
            result
        }
        Field::TensorScalar => Polynomial::atom(if second { Field::Psi(a) } else { Field::Chi(a) }),
        Field::TwoForm(mu, nu) => {
            let commutator = lower_commutator(clifford, mu, nu);
            let mut result = Polynomial::default();
            add_matrix_row(
                &mut result,
                &commutator,
                a,
                if second { Field::Psi } else { Field::Chi },
                None,
                GaussianRational::from_ratio(
                    Ratio::new(if second { 1 } else { -1 }, 4),
                    Ratio::from_integer(0),
                ),
            );
            result
        }
        Field::Psi(b) if !second => chiral_fermion_delta(a, b, false, clifford),
        Field::Chi(b) if second => chiral_fermion_delta(a, b, true, clifford),
        Field::Chi(b) if !second => tensor_fermion_delta(a, b, 1, clifford),
        Field::Psi(b) if second => tensor_fermion_delta(a, b, -1, clifford),
        _ => unreachable!(),
    }
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
        result.add_scaled(&transformed, *coefficient);
    }
    result
}

fn anticommutator(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Polynomial {
    let mut result = apply_delta(left, &delta(right, field, clifford), clifford);
    result.add_scaled(
        &apply_delta(right, &delta(left, field, clifford), clifford),
        GaussianRational::new(1, 0),
    );
    result
}

fn translation(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Polynomial {
    let mut result = Polynomial::default();
    if left.supersymmetry == right.supersymmetry {
        for mu in 0..DIM {
            let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
            result.add_scaled(
                &Polynomial::atom(field).derivative(mu),
                gamma[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
            );
        }
    }
    result
}

fn published_tensor_closure(
    left: Charge,
    right: Charge,
    mu: usize,
    nu: usize,
    clifford: &Clifford4D,
) -> Polynomial {
    let mut result = Polynomial::default();
    if left.supersymmetry == right.supersymmetry {
        for alpha in 0..DIM {
            let gamma = clifford.lower_spinors(&clifford.gamma_up[alpha]);
            result.add_scaled(
                &field_strength(alpha, mu, nu),
                gamma[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
            );
        }
    }

    let sigma1 = pauli(1)[left.supersymmetry][right.supersymmetry];
    let sigma2 = pauli(2)[left.supersymmetry][right.supersymmetry];
    let sigma3 = pauli(3)[left.supersymmetry][right.supersymmetry];
    let gamma_mu = clifford.lower_spinors(&clifford.gamma_down[mu]);
    let gamma_nu = clifford.lower_spinors(&clifford.gamma_down[nu]);
    let gamma5 = clifford.gamma5;

    let add_gauge_derivative =
        |result: &mut Polynomial, derivative_index: usize, gamma: &Matrix4, sign: i64| {
            let a = left.spinor;
            let b = right.spinor;
            result.add_scaled(
                &Polynomial::atom(Field::ScalarA).derivative(derivative_index),
                gamma[a][b]
                    .mul(&sigma1)
                    .mul(&GaussianRational::new(0, sign)),
            );
            let mut gamma_gamma5 = GaussianRational::default();
            for c in 0..SPINORS {
                gamma_gamma5.add_assign(&gamma[a][c].mul(&gamma5[b][c]));
            }
            result.add_scaled(
                &Polynomial::atom(Field::PseudoscalarB).derivative(derivative_index),
                gamma_gamma5
                    .mul(&sigma2)
                    .mul(&GaussianRational::new(0, sign)),
            );
            result.add_scaled(
                &Polynomial::atom(Field::TensorScalar).derivative(derivative_index),
                gamma[a][b]
                    .mul(&sigma3)
                    .mul(&GaussianRational::new(0, -sign)),
            );
        };
    add_gauge_derivative(&mut result, nu, &gamma_mu, 1);
    add_gauge_derivative(&mut result, mu, &gamma_nu, -1);
    result
}

fn real_unit(value: GaussianRational) -> i8 {
    assert_eq!(value.imag, Ratio::from_integer(0));
    assert_eq!(*value.real.denom(), 1);
    i8::try_from(*value.real.numer()).expect("worldline coefficient fits i8")
}

fn worldline_l_matrices(clifford: &Clifford4D) -> [[[i8; 8]; 8]; 8] {
    let mut output = [[[0_i8; 8]; 8]; 8];
    let identity: Matrix4 = std::array::from_fn(|row| {
        std::array::from_fn(|column| GaussianRational::new(i64::from(row == column), 0))
    });
    let i_gamma5 = matrix_scale(&clifford.gamma5, GaussianRational::new(0, 1));
    let i_gamma5_gamma0 = matrix_scale(&clifford.gamma5_gamma_up(0), GaussianRational::new(0, 1));
    let spatial_pairs = [(1, 2), (2, 3), (3, 1)];
    for spinor in 0..SPINORS {
        for component in 0..SPINORS {
            output[spinor][0][component] = real_unit(identity[spinor][component]);
            output[spinor][1][component] = real_unit(i_gamma5[spinor][component]);
            output[spinor][2][component] = real_unit(clifford.gamma_up[0][spinor][component]);
            output[spinor][3][component] = real_unit(i_gamma5_gamma0[spinor][component]);
            output[spinor][4][4 + component] = real_unit(identity[spinor][component]);

            output[4 + spinor][0][4 + component] = -real_unit(identity[spinor][component]);
            output[4 + spinor][1][4 + component] = real_unit(i_gamma5[spinor][component]);
            output[4 + spinor][2][4 + component] =
                real_unit(clifford.gamma_up[0][spinor][component]);
            output[4 + spinor][3][4 + component] = real_unit(i_gamma5_gamma0[spinor][component]);
            output[4 + spinor][4][component] = real_unit(identity[spinor][component]);
        }
        for (offset, &(mu, nu)) in spatial_pairs.iter().enumerate() {
            let commutator = lower_commutator(clifford, mu, nu);
            for component in 0..SPINORS {
                let half = GaussianRational::from_ratio(Ratio::new(1, 2), Ratio::from_integer(0));
                output[spinor][5 + offset][4 + component] =
                    -real_unit(commutator[spinor][component].mul(&half));
                output[4 + spinor][5 + offset][component] =
                    real_unit(commutator[spinor][component].mul(&half));
            }
        }
    }
    output
}

fn published_ct_l_matrices() -> [[[i8; 8]; 8]; 8] {
    use crate::permutahedron_fixtures::S8_REPRESENTATION_OCTETS;
    use crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS;

    let color_permutations: Vec<Vec<usize>> = S8_REPRESENTATION_OCTETS[1]
        .permutations
        .iter()
        .map(|permutation| {
            permutation
                .iter()
                .map(|&entry| usize::from(entry - 1))
                .collect()
        })
        .collect();
    let signs: Vec<i8> = S8_BASE_BOOLEAN_FACTORS[1]
        .iter()
        .flat_map(|factor| (0..8).map(move |row| if factor & (1 << row) == 0 { 1 } else { -1 }))
        .collect();
    let rep = AdinkraRep::from_parts(8, 8, &color_permutations, &signs);
    std::array::from_fn(|color| {
        std::array::from_fn(|row| {
            let mut result = [0_i8; 8];
            result[usize::from(rep.l_matrices[color].perm[row])] = rep.l_matrices[color].sign[row];
            result
        })
    })
}

fn field_label(field: Field) -> String {
    match field {
        Field::ScalarA => "A".into(),
        Field::PseudoscalarB => "B".into(),
        Field::AuxiliaryF => "F".into(),
        Field::AuxiliaryG => "G".into(),
        Field::TensorScalar => "phi".into(),
        Field::TwoForm(mu, nu) => format!("B_{mu}{nu}"),
        Field::Psi(component) => format!("psi_{component}"),
        Field::Chi(component) => format!("chi_{component}"),
    }
}

fn coefficient_label(value: GaussianRational) -> String {
    format!(
        "{}/{}+i*{}/{}",
        value.real.numer(),
        value.real.denom(),
        value.imag.numer(),
        value.imag.denom()
    )
}

fn canonical_coefficient(value: GaussianRational) -> CanonicalCoefficient {
    CanonicalCoefficient::new(
        *value.real.numer(),
        *value.real.denom(),
        *value.imag.numer(),
        *value.imag.denom(),
    )
    .expect("source Gaussian rational is normalized and finite")
}

fn canonical_lorentz(left: u8, right: u8) -> LorentzRep {
    LorentzRep {
        left_twice_spin: left,
        right_twice_spin: right,
        reality: Reality::Real,
    }
}

/// Exact source-basis adapter used by the canonical parentage engine.
pub(crate) fn exact_canonical_fixture() -> CanonicalPhysicalFingerprint {
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

    let mut components = Vec::new();
    for &field in &fields {
        let (statistics, lorentz, height_twice, role, form_degree) = match field {
            Field::ScalarA | Field::PseudoscalarB | Field::TensorScalar => (
                Statistics::Boson,
                canonical_lorentz(0, 0),
                0,
                CanonicalRole::Propagating,
                None,
            ),
            Field::AuxiliaryF | Field::AuxiliaryG => (
                Statistics::Boson,
                canonical_lorentz(0, 0),
                2,
                CanonicalRole::Auxiliary,
                None,
            ),
            Field::TwoForm(_, _) => (
                Statistics::Boson,
                canonical_lorentz(2, 0),
                0,
                CanonicalRole::GaugePotential,
                Some(2),
            ),
            Field::Psi(_) | Field::Chi(_) => (
                Statistics::Fermion,
                canonical_lorentz(1, 0),
                1,
                CanonicalRole::Propagating,
                None,
            ),
        };
        components.push(CanonicalComponent {
            label: field_label(field),
            statistics,
            lorentz,
            height_twice,
            role,
            form_degree,
        });
    }
    let field_indices: BTreeMap<_, _> = fields
        .iter()
        .enumerate()
        .map(|(index, &field)| (field, index))
        .collect();

    let mut gauge_parameters = Vec::new();
    for mu in 0..DIM {
        gauge_parameters.push(components.len());
        components.push(CanonicalComponent {
            label: format!("gauge_parameter_{mu}"),
            statistics: Statistics::Boson,
            lorentz: canonical_lorentz(1, 1),
            height_twice: -2,
            role: CanonicalRole::GaugeParameter { stage: 0 },
            form_degree: Some(1),
        });
    }
    let reducibility_parameter = components.len();
    components.push(CanonicalComponent {
        label: "gauge_for_gauge_parameter".to_owned(),
        statistics: Statistics::Boson,
        lorentz: canonical_lorentz(0, 0),
        height_twice: -4,
        role: CanonicalRole::GaugeParameter { stage: 1 },
        form_degree: Some(0),
    });

    let triples: Vec<_> = (0..DIM)
        .flat_map(|mu| {
            ((mu + 1)..DIM).flat_map(move |nu| ((nu + 1)..DIM).map(move |rho| (mu, nu, rho)))
        })
        .collect();
    let mut strength_indices = BTreeMap::new();
    for &(mu, nu, rho) in &triples {
        let index = components.len();
        strength_indices.insert((mu, nu, rho), index);
        components.push(CanonicalComponent {
            label: format!("field_strength_{mu}{nu}{rho}"),
            statistics: Statistics::Boson,
            // A real three-form is Lorentz-dual to a real vector.
            lorentz: canonical_lorentz(1, 1),
            height_twice: 2,
            role: CanonicalRole::FieldStrength,
            form_degree: Some(3),
        });
    }

    let supercharges = charges
        .iter()
        .map(|charge| CanonicalSupercharge {
            label: format!("D{}_{}", charge.supersymmetry + 1, charge.spinor),
            lorentz: canonical_lorentz(1, 0),
            height_twice: 1,
        })
        .collect();

    let mut linkage = Vec::new();
    for (charge_index, &charge) in charges.iter().enumerate() {
        for &source_field in &fields {
            let source = field_indices[&source_field];
            for (atom, coefficient) in delta(charge, source_field, &clifford).0 {
                linkage.push(CanonicalLinkage {
                    charge: charge_index,
                    source,
                    target: field_indices[&atom.field],
                    derivative: CanonicalDerivative(atom.derivatives),
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
        for &(mu, nu, rho) in &triples {
            for (atom, coefficient) in
                apply_delta(charge, &field_strength(mu, nu, rho), &clifford).0
            {
                linkage.push(CanonicalLinkage {
                    charge: charge_index,
                    source: strength_indices[&(mu, nu, rho)],
                    target: field_indices[&atom.field],
                    derivative: CanonicalDerivative(atom.derivatives),
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
    }

    let mut gauge_complex = Vec::new();
    for mu in 0..DIM {
        gauge_complex.push(CanonicalGaugeTerm {
            parameter: reducibility_parameter,
            target: gauge_parameters[mu],
            derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == mu))),
            coefficient: CanonicalCoefficient::integer(1, 0),
        });
        for nu in (mu + 1)..DIM {
            let target = field_indices[&Field::TwoForm(mu, nu)];
            gauge_complex.push(CanonicalGaugeTerm {
                parameter: gauge_parameters[nu],
                target,
                derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == mu))),
                coefficient: CanonicalCoefficient::integer(1, 0),
            });
            gauge_complex.push(CanonicalGaugeTerm {
                parameter: gauge_parameters[mu],
                target,
                derivative: CanonicalDerivative(std::array::from_fn(|axis| u8::from(axis == nu))),
                coefficient: CanonicalCoefficient::integer(-1, 0),
            });
        }
    }

    let bianchi_identities = vec![CanonicalBianchi {
        terms: vec![
            CanonicalLinearTerm {
                component: strength_indices[&(1, 2, 3)],
                derivative: CanonicalDerivative([1, 0, 0, 0]),
                coefficient: CanonicalCoefficient::integer(1, 0),
            },
            CanonicalLinearTerm {
                component: strength_indices[&(0, 2, 3)],
                derivative: CanonicalDerivative([0, 1, 0, 0]),
                coefficient: CanonicalCoefficient::integer(-1, 0),
            },
            CanonicalLinearTerm {
                component: strength_indices[&(0, 1, 3)],
                derivative: CanonicalDerivative([0, 0, 1, 0]),
                coefficient: CanonicalCoefficient::integer(1, 0),
            },
            CanonicalLinearTerm {
                component: strength_indices[&(0, 1, 2)],
                derivative: CanonicalDerivative([0, 0, 0, 1]),
                coefficient: CanonicalCoefficient::integer(-1, 0),
            },
        ],
    }];

    CanonicalPhysicalFingerprint {
        name: "chiral-tensor-exact-source-adapter".to_owned(),
        components,
        supercharges,
        linkage,
        gauge_complex,
        bianchi_identities,
        central_generators: Vec::new(),
        central_entries: Vec::new(),
        central_occurrences: Vec::new(),
    }
}

pub fn higher_dimensional_fingerprint() -> MultipletFingerprint {
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

    let mut transformation_terms = 0;
    let mut algebraic_terms = 0;
    let mut temporal_derivative_terms = 0;
    let mut spatial_derivative_terms = 0;
    let mut relations_with_spatial_derivatives = 0;
    let mut spatial_lines = Vec::new();
    for (charge_index, &charge) in charges.iter().enumerate() {
        for &target in &fields {
            let polynomial = delta(charge, target, &clifford);
            transformation_terms += polynomial.0.len();
            let mut relation_has_spatial_term = false;
            for (atom, coefficient) in &polynomial.0 {
                let temporal = atom.derivatives[0] > 0;
                let spatial = atom.derivatives[1..].iter().any(|&degree| degree > 0);
                algebraic_terms += usize::from(!temporal && !spatial);
                temporal_derivative_terms += usize::from(temporal);
                spatial_derivative_terms += usize::from(spatial);
                if spatial {
                    relation_has_spatial_term = true;
                    spatial_lines.push(format!(
                        "q={charge_index};target={};source={};d={:?};c={}",
                        field_label(target),
                        field_label(atom.field),
                        atom.derivatives,
                        coefficient_label(*coefficient)
                    ));
                }
            }
            relations_with_spatial_derivatives += usize::from(relation_has_spatial_term);
        }
    }

    let mut potential_closure_relations = 0;
    let mut nonzero_gauge_residue_relations = 0;
    let mut gauge_residue_terms = 0;
    let mut gauge_temporal_terms = 0;
    let mut gauge_spatial_terms = 0;
    let mut gauge_lines = Vec::new();
    let mut residue_source_fields = BTreeSet::new();
    for left in 0..charges.len() {
        for right in left..charges.len() {
            for mu in 0..DIM {
                for nu in (mu + 1)..DIM {
                    potential_closure_relations += 1;
                    let field = Field::TwoForm(mu, nu);
                    let mut residue =
                        published_tensor_closure(charges[left], charges[right], mu, nu, &clifford);
                    residue.add_scaled(
                        &translation(charges[left], charges[right], field, &clifford),
                        GaussianRational::new(-1, 0),
                    );
                    nonzero_gauge_residue_relations += usize::from(!residue.0.is_empty());
                    gauge_residue_terms += residue.0.len();
                    for (atom, coefficient) in &residue.0 {
                        let temporal = atom.derivatives[0] > 0;
                        let spatial = atom.derivatives[1..].iter().any(|&degree| degree > 0);
                        gauge_temporal_terms += usize::from(temporal);
                        gauge_spatial_terms += usize::from(spatial);
                        residue_source_fields.insert(field_label(atom.field));
                        gauge_lines.push(format!(
                            "pair={left},{right};target={};source={};d={:?};c={}",
                            field_label(field),
                            field_label(atom.field),
                            atom.derivatives,
                            coefficient_label(*coefficient)
                        ));
                    }
                }
            }
        }
    }

    MultipletFingerprint {
        name: "chiral-tensor",
        source: "arXiv:1405.0048 Eqs. (44)-(53)",
        raw_component_fields: fields.len(),
        worldline_bosons: 8,
        worldline_fermions: 8,
        derivative_operator: DerivativeOperatorFingerprint {
            transformation_relations: charges.len() * fields.len(),
            transformation_terms,
            algebraic_terms,
            temporal_derivative_terms,
            spatial_derivative_terms,
            relations_with_spatial_derivatives,
            canonical_spatial_operator_sha256: sha256_lines(&spatial_lines),
        },
        gauge: GaugeFingerprint {
            potential_form_degree: 2,
            potential_components_before_gauge_fixing: 6,
            temporal_gauge_components_removed: 3,
            potential_components_after_temporal_gauge: 3,
            field_strength_form_degree: 3,
            gauge_parameter_form_degree: 1,
            potential_closure_relations,
            nonzero_gauge_residue_relations,
            gauge_residue_terms,
            temporal_derivative_terms: gauge_temporal_terms,
            spatial_derivative_terms: gauge_spatial_terms,
            residue_source_fields: residue_source_fields.into_iter().collect(),
            canonical_gauge_residue_sha256: sha256_lines(&gauge_lines),
        },
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ChiralTensorReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub gamma_conventions_verified: bool,
    pub epsilon_lower_0123: i64,
    pub component_fields: usize,
    pub charge_pairs: usize,
    pub component_relations_checked: usize,
    pub nongauge_field_relations_checked: usize,
    pub two_form_potential_relations_checked: usize,
    pub residual_relations: usize,
    pub residual_terms: usize,
    pub two_form_gauge_term_is_required: bool,
    pub reduced_l_matrix_entries_checked: usize,
    pub reduced_l_matrix_residual_entries: usize,
    pub exact_ct_anchor_recovered: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub source: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChiralTensorArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub report: ChiralTensorReport,
    pub reduced_l_matrices: [[[i8; 8]; 8]; 8],
    pub published_ct_l_matrices: [[[i8; 8]; 8]; 8],
}

pub fn verify() -> ChiralTensorReport {
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
                let expected = if let Field::TwoForm(mu, nu) = field {
                    published_tensor_closure(charges[left], charges[right], mu, nu, &clifford)
                } else {
                    translation(charges[left], charges[right], field, &clifford)
                };
                if actual != expected {
                    residual_relations += 1;
                    let mut residual = actual;
                    residual.add_scaled(&expected, GaussianRational::new(-1, 0));
                    residual_terms += residual.0.len();
                }
            }
        }
    }
    let gauge_required = (0..SPINORS).any(|left_spinor| {
        (0..SPINORS).any(|right_spinor| {
            (0..DIM).any(|mu| {
                ((mu + 1)..DIM).any(|nu| {
                    !published_tensor_closure(
                        Charge {
                            supersymmetry: 0,
                            spinor: left_spinor,
                        },
                        Charge {
                            supersymmetry: 1,
                            spinor: right_spinor,
                        },
                        mu,
                        nu,
                        &clifford,
                    )
                    .0
                    .is_empty()
                })
            })
        })
    });
    let reduced = worldline_l_matrices(&clifford);
    let published = published_ct_l_matrices();
    let reduced_residuals = reduced
        .iter()
        .flatten()
        .flatten()
        .zip(published.iter().flatten().flatten())
        .filter(|(left, right)| left != right)
        .count();
    let gamma_verified = clifford.verifies_source_conventions();
    ChiralTensorReport {
        schema_version: "chiral-tensor-4d-closure-v1",
        source: "arXiv:1405.0048 Eqs. (44)-(53), with gamma conventions from arXiv:0902.3830 Appendix A",
        gamma_conventions_verified: gamma_verified,
        epsilon_lower_0123: epsilon_lower([0, 1, 2, 3]),
        component_fields: fields.len(),
        charge_pairs: charges.len() * (charges.len() + 1) / 2,
        component_relations_checked: checked,
        nongauge_field_relations_checked: 13 * 36,
        two_form_potential_relations_checked: 6 * 36,
        residual_relations,
        residual_terms,
        two_form_gauge_term_is_required: gauge_required,
        reduced_l_matrix_entries_checked: 8 * 8 * 8,
        reduced_l_matrix_residual_entries: reduced_residuals,
        exact_ct_anchor_recovered: reduced_residuals == 0,
        passed: gamma_verified
            && residual_relations == 0
            && gauge_required
            && reduced_residuals == 0,
        boundary: "This reproduces the published linear abelian chiral-tensor component algebra, including closure on the two-form potential modulo the printed gauge term. It is not a new four-dimensional construction.",
    }
}

pub fn build() -> ChiralTensorArtifact {
    let clifford = Clifford4D::build();
    ChiralTensorArtifact {
        schema_version: "chiral-tensor-4d-artifact-v1",
        title: "Exact four-dimensional chiral-tensor closure and worldline reduction",
        sources: vec![
            SourceRecord {
                source: "arXiv:1405.0048",
                locator: "Eqs. (44)-(53) and Appendix C",
                role: "component transformations, closure modulo two-form gauge transformations, temporal gauge, reduction map, and published L matrices",
            },
            SourceRecord {
                source: "arXiv:0902.3830",
                locator: "Appendix A, Eqs. (A.3)-(A.11)",
                role: "mostly-plus Majorana gamma matrices, charge conjugation, and epsilon convention",
            },
        ],
        report: verify(),
        reduced_l_matrices: worldline_l_matrices(&clifford),
        published_ct_l_matrices: published_ct_l_matrices(),
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> ChiralTensorReport {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create chiral-tensor data")),
        &artifact,
    )
    .expect("write chiral-tensor data");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create chiral-tensor validation")),
        &artifact.report,
    )
    .expect("write chiral-tensor validation");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_chiral_tensor_component_algebra_closes() {
        let report = verify();
        assert_eq!(report.epsilon_lower_0123, -1);
        assert_eq!(report.component_fields, 19);
        assert_eq!(report.charge_pairs, 36);
        assert_eq!(report.component_relations_checked, 684);
        assert_eq!(report.nongauge_field_relations_checked, 468);
        assert_eq!(report.two_form_potential_relations_checked, 216);
        assert_eq!(report.residual_relations, 0);
        assert_eq!(report.residual_terms, 0);
        assert!(report.two_form_gauge_term_is_required);
        assert_eq!(report.reduced_l_matrix_entries_checked, 512);
        assert_eq!(report.reduced_l_matrix_residual_entries, 0);
        assert!(report.exact_ct_anchor_recovered);
        assert!(report.passed);
    }
}
