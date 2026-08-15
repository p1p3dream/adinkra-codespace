//! Exact 4D vector-tensor component closure for arXiv:1405.0048 Eq. (77).
//!
//! The first fixture uses the physical one-central-charge branch `(m,n)=(1,0)`
//! and checks the corrected Eq. (78) on all component fields.

#![allow(clippy::needless_range_loop)]

use crate::chiral_vector_4d::{matrix_mul, pauli, Clifford4D, GaussianRational, Matrix4};
use crate::exact_component_algebra::Polynomial;
use num_rational::Ratio;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const DIM: usize = 4;
const SPINORS: usize = 4;
type Poly = Polynomial<Field, DIM>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
enum Field {
    Scalar,
    Vector(usize),
    TwoForm(usize, usize),
    Auxiliary,
    Lambda(usize, usize),
}

impl Field {
    fn all() -> Vec<Self> {
        let mut fields = vec![Self::Scalar];
        fields.extend((0..DIM).map(Self::Vector));
        for mu in 0..DIM {
            for nu in (mu + 1)..DIM {
                fields.push(Self::TwoForm(mu, nu));
            }
        }
        fields.push(Self::Auxiliary);
        fields.extend((0..2).flat_map(|supersymmetry| {
            (0..SPINORS).map(move |spinor| Self::Lambda(supersymmetry, spinor))
        }));
        fields
    }
}

#[derive(Clone, Copy)]
struct Charge {
    supersymmetry: usize,
    spinor: usize,
}

fn e(index: usize, other: usize) -> i64 {
    [[0, 1], [-1, 0]][index][other]
}

fn metric_sign(index: usize) -> i64 {
    if index == 0 {
        -1
    } else {
        1
    }
}

fn epsilon_lower(indices: [usize; 4]) -> i64 {
    if (0..4).any(|left| ((left + 1)..4).any(|right| indices[left] == indices[right])) {
        return 0;
    }
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|&(left, right)| indices[left] > indices[right])
        .count();
    if inversions.is_multiple_of(2) {
        -1
    } else {
        1
    }
}

fn epsilon_upper(indices: [usize; 4]) -> i64 {
    epsilon_lower(indices) * indices.into_iter().map(metric_sign).product::<i64>()
}

fn two_form(mu: usize, nu: usize) -> (Field, i64) {
    if mu < nu {
        (Field::TwoForm(mu, nu), 1)
    } else {
        (Field::TwoForm(nu, mu), -1)
    }
}

fn two_form_polynomial(mu: usize, nu: usize) -> Poly {
    if mu == nu {
        return Poly::default();
    }
    let (field, sign) = two_form(mu, nu);
    let mut result = Poly::default();
    result.add_scaled(&Poly::atom(field), GaussianRational::new(sign, 0));
    result
}

fn vector_strength(mu: usize, nu: usize) -> Poly {
    let mut result = Poly::atom(Field::Vector(nu)).derivative(mu);
    result.add_scaled(
        &Poly::atom(Field::Vector(mu)).derivative(nu),
        GaussianRational::new(-1, 0),
    );
    result
}

fn tensor_strength(alpha: usize, mu: usize, nu: usize) -> Poly {
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

fn add_fermion_row(
    result: &mut Poly,
    matrix: &Matrix4,
    row: usize,
    internal: usize,
    derivative: Option<usize>,
    scale: GaussianRational,
) {
    for component in 0..SPINORS {
        let mut term = Poly::atom(Field::Lambda(internal, component));
        if let Some(mu) = derivative {
            term = term.derivative(mu);
        }
        result.add_scaled(&term, matrix[row][component].mul(&scale));
    }
}

fn fermion_delta(
    charge: Charge,
    target_internal: usize,
    target_spinor: usize,
    clifford: &Clifford4D,
) -> Poly {
    let i = charge.supersymmetry;
    let a = charge.spinor;
    let j = target_internal;
    let b = target_spinor;
    let mut result = Poly::default();

    let aij = e(i, j);
    if aij != 0 {
        for mu in 0..DIM {
            for nu in (mu + 1)..DIM {
                let commutator = clifford.lower_spinors(&clifford.commutator_up(mu, nu));
                result.add_scaled(
                    &vector_strength(mu, nu),
                    commutator[a][b].mul(&GaussianRational::from_ratio(
                        Ratio::from_integer(0),
                        Ratio::new(-aij, 2),
                    )),
                );
            }
        }
        let gamma5 = clifford.lower_spinors(&clifford.gamma5);
        result.add_scaled(
            &Poly::atom(Field::Auxiliary),
            gamma5[a][b].mul(&GaussianRational::new(aij, 0)),
        );
    }

    if i == j {
        for mu in 0..DIM {
            let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
            result.add_scaled(
                &Poly::atom(Field::Scalar).derivative(mu),
                gamma[a][b].mul(&GaussianRational::new(0, 1)),
            );
            let gamma5_gamma_down =
                clifford.lower_spinors(&matrix_mul(&clifford.gamma5, &clifford.gamma_down[mu]));
            for rho in 0..DIM {
                for sigma in 0..DIM {
                    for tau in (sigma + 1)..DIM {
                        let epsilon = epsilon_upper([mu, rho, sigma, tau]);
                        if epsilon == 0 {
                            continue;
                        }
                        result.add_scaled(
                            &Poly::atom(Field::TwoForm(sigma, tau)).derivative(rho),
                            gamma5_gamma_down[a][b].mul(&GaussianRational::new(2 * epsilon, 0)),
                        );
                    }
                }
            }
        }
    }
    result
}

fn delta(charge: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let i = charge.supersymmetry;
    let a = charge.spinor;
    match field {
        Field::Scalar => Poly::atom(Field::Lambda(i, a)),
        Field::Vector(mu) => {
            let mut result = Poly::default();
            for j in 0..2 {
                let coefficient = e(i, j);
                if coefficient != 0 {
                    add_fermion_row(
                        &mut result,
                        &clifford.gamma_down[mu],
                        a,
                        j,
                        None,
                        GaussianRational::new(coefficient, 0),
                    );
                }
            }
            result
        }
        Field::TwoForm(mu, nu) => {
            let commutator = lower_commutator(clifford, mu, nu);
            let mut result = Poly::default();
            add_fermion_row(
                &mut result,
                &commutator,
                a,
                i,
                None,
                GaussianRational::from_ratio(Ratio::new(-1, 4), Ratio::from_integer(0)),
            );
            result
        }
        Field::Auxiliary => {
            let mut result = Poly::default();
            for j in 0..2 {
                let coefficient = e(i, j);
                if coefficient == 0 {
                    continue;
                }
                for mu in 0..DIM {
                    add_fermion_row(
                        &mut result,
                        &clifford.gamma5_gamma_up(mu),
                        a,
                        j,
                        Some(mu),
                        GaussianRational::new(0, coefficient),
                    );
                }
            }
            result
        }
        Field::Lambda(j, b) => fermion_delta(charge, j, b, clifford),
    }
}

fn apply_delta(charge: Charge, polynomial: &Poly, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    for (jet, coefficient) in &polynomial.0 {
        let mut transformed = delta(charge, jet.field, clifford);
        for mu in 0..DIM {
            for _ in 0..jet.derivatives[mu] {
                transformed = transformed.derivative(mu);
            }
        }
        result.add_scaled(&transformed, *coefficient);
    }
    result
}

fn anticommutator(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = apply_delta(left, &delta(right, field, clifford), clifford);
    result.add_scaled(
        &apply_delta(right, &delta(left, field, clifford), clifford),
        GaussianRational::new(1, 0),
    );
    result
}

fn add_translation(
    result: &mut Poly,
    left: Charge,
    right: Charge,
    field: Field,
    clifford: &Clifford4D,
) {
    if left.supersymmetry != right.supersymmetry {
        return;
    }
    for rho in 0..DIM {
        let gamma = clifford.lower_spinors(&clifford.gamma_up[rho]);
        result.add_scaled(
            &Poly::atom(field).derivative(rho),
            gamma[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
        );
    }
}

fn expected_closure(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let a = left.spinor;
    let b = right.spinor;
    let sigma2 = pauli(2)[left.supersymmetry][right.supersymmetry];
    let gamma5 = clifford.lower_spinors(&clifford.gamma5);
    let mut result = Poly::default();
    match field {
        Field::Scalar => {
            add_translation(&mut result, left, right, field, clifford);
            result.add_scaled(
                &Poly::atom(Field::Auxiliary),
                sigma2.mul(&gamma5[a][b]).mul(&GaussianRational::new(0, 2)),
            );
        }
        Field::Vector(nu) => {
            if left.supersymmetry == right.supersymmetry {
                for rho in 0..DIM {
                    let gamma = clifford.lower_spinors(&clifford.gamma_up[rho]);
                    result.add_scaled(
                        &vector_strength(rho, nu),
                        gamma[a][b].mul(&GaussianRational::new(0, 2)),
                    );
                }
            }
            for alpha in 0..DIM {
                for beta in 0..DIM {
                    for rho in 0..DIM {
                        let epsilon = epsilon_lower([nu, alpha, beta, rho]);
                        if epsilon == 0 {
                            continue;
                        }
                        let raised = metric_sign(alpha) * metric_sign(beta) * metric_sign(rho);
                        result.add_scaled(
                            &tensor_strength(alpha, beta, rho),
                            sigma2.mul(&gamma5[a][b]).mul(&GaussianRational::from_ratio(
                                Ratio::from_integer(0),
                                Ratio::new(2 * epsilon * raised, 3),
                            )),
                        );
                    }
                }
            }
            result.add_scaled(
                &Poly::atom(Field::Scalar).derivative(nu),
                sigma2
                    .mul(&clifford.charge_conjugation[a][b])
                    .mul(&GaussianRational::new(-2, 0)),
            );
        }
        Field::TwoForm(mu, nu) => {
            if left.supersymmetry == right.supersymmetry {
                for alpha in 0..DIM {
                    let gamma = clifford.lower_spinors(&clifford.gamma_up[alpha]);
                    result.add_scaled(
                        &tensor_strength(alpha, mu, nu),
                        gamma[a][b].mul(&GaussianRational::new(0, 2)),
                    );
                }
            }
            result.add_scaled(
                &vector_strength(mu, nu),
                sigma2
                    .mul(&clifford.charge_conjugation[a][b])
                    .mul(&GaussianRational::new(-1, 0)),
            );
            for alpha in 0..DIM {
                for beta in (alpha + 1)..DIM {
                    let epsilon = epsilon_lower([mu, nu, alpha, beta]);
                    if epsilon == 0 {
                        continue;
                    }
                    let raised = metric_sign(alpha) * metric_sign(beta);
                    result.add_scaled(
                        &vector_strength(alpha, beta),
                        sigma2
                            .mul(&gamma5[a][b])
                            .mul(&GaussianRational::new(0, -epsilon * raised)),
                    );
                }
            }
            if left.supersymmetry == right.supersymmetry {
                let gamma_mu = clifford.lower_spinors(&clifford.gamma_down[mu]);
                let gamma_nu = clifford.lower_spinors(&clifford.gamma_down[nu]);
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(nu),
                    gamma_mu[a][b].mul(&GaussianRational::new(0, -1)),
                );
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(mu),
                    gamma_nu[a][b].mul(&GaussianRational::new(0, 1)),
                );
            }
        }
        Field::Lambda(k, c) => {
            add_translation(&mut result, left, right, field, clifford);
            for l in 0..2 {
                let internal = pauli(2)[k][l];
                if internal.is_zero() || sigma2.is_zero() {
                    continue;
                }
                for rho in 0..DIM {
                    let gamma5_gamma = clifford.gamma5_gamma_up(rho);
                    for d in 0..SPINORS {
                        result.add_scaled(
                            &Poly::atom(Field::Lambda(l, d)).derivative(rho),
                            sigma2
                                .mul(&internal)
                                .mul(&gamma5[a][b])
                                .mul(&gamma5_gamma[c][d])
                                .mul(&GaussianRational::new(0, -2)),
                        );
                    }
                }
            }
        }
        Field::Auxiliary => {
            add_translation(&mut result, left, right, field, clifford);
            for rho in 0..DIM {
                result.add_scaled(
                    &Poly::atom(Field::Scalar).derivative(rho).derivative(rho),
                    sigma2
                        .mul(&gamma5[a][b])
                        .mul(&GaussianRational::new(0, -2 * metric_sign(rho))),
                );
            }
        }
    }
    result
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensor4DReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub branch: &'static str,
    pub gamma_conventions_verified: bool,
    pub fields: usize,
    pub supercharges: usize,
    pub unordered_charge_pairs: usize,
    pub component_relations_checked: usize,
    pub scalar_relations_checked: usize,
    pub vector_potential_relations_checked: usize,
    pub two_form_potential_relations_checked: usize,
    pub auxiliary_relations_checked: usize,
    pub fermion_relations_checked: usize,
    pub residual_relations: usize,
    pub residual_terms: usize,
    pub scalar_residual_relations: usize,
    pub vector_residual_relations: usize,
    pub two_form_residual_relations: usize,
    pub auxiliary_residual_relations: usize,
    pub fermion_residual_relations: usize,
    pub corrected_scalar_typo_used: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
    pub pdf_sha256: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorTensor4DArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub branch_coefficients: Vec<&'static str>,
    pub source_corrections: Vec<&'static str>,
    pub convention_translation: Vec<&'static str>,
    pub report: VectorTensor4DReport,
}

pub fn verify() -> VectorTensor4DReport {
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
    let mut checked = 0usize;
    let mut residual_relations = 0usize;
    let mut residual_terms = 0usize;
    let mut residual_by_kind = [0usize; 5];
    for left in 0..charges.len() {
        for right in left..charges.len() {
            for &field in &fields {
                checked += 1;
                let actual = anticommutator(charges[left], charges[right], field, &clifford);
                let expected = expected_closure(charges[left], charges[right], field, &clifford);
                if actual != expected {
                    residual_relations += 1;
                    let kind = match field {
                        Field::Scalar => 0,
                        Field::Vector(_) => 1,
                        Field::TwoForm(_, _) => 2,
                        Field::Auxiliary => 3,
                        Field::Lambda(_, _) => 4,
                    };
                    residual_by_kind[kind] += 1;
                    let mut residual = actual;
                    residual.add_scaled(&expected, GaussianRational::new(-1, 0));
                    residual_terms += residual.term_count();
                }
            }
        }
    }
    VectorTensor4DReport {
        schema_version: "vector-tensor-4d-closure-v1",
        source: "arXiv:1405.0048 Eqs. (59)-(61) and (76)-(84)",
        branch: "m=1, n=0",
        gamma_conventions_verified: clifford.verifies_source_conventions(),
        fields: fields.len(),
        supercharges: charges.len(),
        unordered_charge_pairs: charges.len() * (charges.len() + 1) / 2,
        component_relations_checked: checked,
        scalar_relations_checked: 36,
        vector_potential_relations_checked: 4 * 36,
        two_form_potential_relations_checked: 6 * 36,
        auxiliary_relations_checked: 36,
        fermion_relations_checked: 8 * 36,
        residual_relations,
        residual_terms,
        scalar_residual_relations: residual_by_kind[0],
        vector_residual_relations: residual_by_kind[1],
        two_form_residual_relations: residual_by_kind[2],
        auxiliary_residual_relations: residual_by_kind[3],
        fermion_residual_relations: residual_by_kind[4],
        corrected_scalar_typo_used: true,
        passed: clifford.verifies_source_conventions() && checked == 720 && residual_relations == 0,
        boundary: "This checks the sourced 4D transformation composition, including the printed gauge residues and extension terms. It does not independently identify those terms with the hep-th/9609016 central-coordinate operator.",
    }
}

pub fn build() -> VectorTensor4DArtifact {
    VectorTensor4DArtifact {
        schema_version: "vector-tensor-4d-artifact-v1",
        title: "Exact four-dimensional vector-tensor component closure",
        sources: vec![SourceRecord {
            arxiv_id: "1405.0048",
            locator: "Sec. 3.6, Eqs. (59)-(61) and (76)-(84)",
            role: "component transformations and closure ledger",
            pdf_sha256: "8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20",
        }],
        branch_coefficients: vec![
            "m=1, n=0",
            "a=i sigma_2",
            "b=identity_2",
            "c1=0",
            "s1=1",
            "c2_plus=0",
            "c2_minus=-2",
        ],
        source_corrections: vec![
            "Eq. (78) scalar extension uses d, not the printed uncontracted partial_mu d",
            "Eq. (80) second row is Phi_5 through Phi_8, not repeated Phi_2 through Phi_4 labels",
        ],
        convention_translation: vec![
            "epsilon_lower_0123=-1 and mostly-plus metric",
            "the dual-H and dual-F signs are converted to the stored Clifford4D epsilon convention",
            "antisymmetric tensor pairs are stored once and their summed coefficients are doubled",
        ],
        report: verify(),
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> VectorTensor4DReport {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create vector-tensor data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create vector-tensor result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create vector-tensor artifact")),
        &artifact,
    )
    .expect("write vector-tensor artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(validation_path).expect("create vector-tensor validation")),
        &artifact.report,
    )
    .expect("write vector-tensor validation");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrected_equation_78_closes_exactly() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
    }
}
