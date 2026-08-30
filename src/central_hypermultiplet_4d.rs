//! Exact rigid 4D N=2 Wess-Fayet central hypermultiplet.
//!
//! The source is arXiv:1405.0048 Sec. 3.1.  This module checks Eq. (13)
//! against the complete component algebra in Eq. (16), factors the extra
//! terms through one commuting off-shell central operator, and only then
//! applies the 0-brane map of Eqs. (23)-(25).  Appendix A is recovered
//! exactly and bridges the source fixture to the committed `CC` one-Z
//! worldline operator.

#![allow(clippy::needless_range_loop)]

use crate::chiral_vector_4d::{Clifford4D, GaussianRational, Matrix4, pauli};
use crate::exact_component_algebra::Polynomial;
use crate::higher_dimensional_canonical::{
    CentralEntry as CanonicalCentralEntry, CentralGenerator as CanonicalCentralGenerator,
    CentralOccurrence as CanonicalCentralOccurrence, Component as CanonicalComponent,
    ComponentRole as CanonicalRole, DerivativeMonomial as CanonicalDerivative,
    GaussianRational as CanonicalCoefficient, LinkageTerm as CanonicalLinkage, LorentzRep,
    PhysicalFingerprint as CanonicalPhysicalFingerprint, Reality, Statistics,
    Supercharge as CanonicalSupercharge,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const DIM: usize = 4;
const SPINORS: usize = 4;
const INTERNAL: usize = 2;
const WORLDLINE_DIMENSION: usize = 8;
const SCHEMA_VERSION: &str = "central-hypermultiplet-4d-v1";

type Poly = Polynomial<Field, DIM>;
type WorldlineMatrix = [[i16; WORLDLINE_DIMENSION]; WORLDLINE_DIMENSION];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
enum Field {
    A,
    B,
    F,
    G,
    TildeA,
    TildeB,
    TildeF,
    TildeG,
    Psi(usize, usize),
}

impl Field {
    fn all() -> Vec<Self> {
        let mut fields = vec![
            Self::A,
            Self::B,
            Self::F,
            Self::G,
            Self::TildeA,
            Self::TildeB,
            Self::TildeF,
            Self::TildeG,
        ];
        fields.extend(
            (0..INTERNAL)
                .flat_map(|internal| (0..SPINORS).map(move |spinor| Self::Psi(internal, spinor))),
        );
        fields
    }
}

#[derive(Clone, Copy, Debug)]
struct Charge {
    internal: usize,
    spinor: usize,
}

fn charges() -> Vec<Charge> {
    (0..INTERNAL)
        .flat_map(|internal| (0..SPINORS).map(move |spinor| Charge { internal, spinor }))
        .collect()
}

fn internal_identity(left: usize, right: usize) -> GaussianRational {
    GaussianRational::new(i64::from(left == right), 0)
}

fn internal_pauli(index: usize, left: usize, right: usize) -> GaussianRational {
    pauli(index)[left][right]
}

fn add_fermion_tensor_row(
    result: &mut Poly,
    internal: &[[GaussianRational; INTERNAL]; INTERNAL],
    spin: &Matrix4,
    charge: Charge,
    derivative: Option<usize>,
    scale: GaussianRational,
) {
    for target_internal in 0..INTERNAL {
        for target_spinor in 0..SPINORS {
            let mut term = Poly::atom(Field::Psi(target_internal, target_spinor));
            if let Some(axis) = derivative {
                term = term.derivative(axis);
            }
            let coefficient = internal[charge.internal][target_internal]
                .mul(&spin[charge.spinor][target_spinor])
                .mul(&scale);
            result.add_scaled(&term, coefficient);
        }
    }
}

fn identity4() -> Matrix4 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| GaussianRational::new(i64::from(row == column), 0))
    })
}

fn internal_matrix(index: usize) -> [[GaussianRational; INTERNAL]; INTERNAL] {
    if index == 0 {
        std::array::from_fn(|row| std::array::from_fn(|column| internal_identity(row, column)))
    } else {
        pauli(index)
    }
}

fn add_boson(
    result: &mut Poly,
    field: Field,
    derivative: Option<usize>,
    coefficient: GaussianRational,
) {
    let mut term = Poly::atom(field);
    if let Some(axis) = derivative {
        term = term.derivative(axis);
    }
    result.add_scaled(&term, coefficient);
}

fn fermion_delta(
    charge: Charge,
    target_internal: usize,
    target_spinor: usize,
    clifford: &Clifford4D,
) -> Poly {
    let i = charge.internal;
    let a = charge.spinor;
    let j = target_internal;
    let b = target_spinor;
    let sigma1 = internal_pauli(1, i, j);
    let sigma2 = internal_pauli(2, i, j);
    let sigma3 = internal_pauli(3, i, j);
    let delta = internal_identity(i, j);
    let mut result = Poly::default();

    for mu in 0..DIM {
        let gamma = clifford.lower_spinors(&clifford.gamma_up[mu]);
        let gamma5_gamma = clifford.lower_spinors(&clifford.gamma5_gamma_up(mu));
        add_boson(
            &mut result,
            Field::A,
            Some(mu),
            sigma3.mul(&gamma[a][b]).mul(&GaussianRational::new(0, 1)),
        );
        add_boson(
            &mut result,
            Field::B,
            Some(mu),
            delta
                .mul(&gamma5_gamma[a][b])
                .mul(&GaussianRational::new(-1, 0)),
        );
        add_boson(
            &mut result,
            Field::TildeA,
            Some(mu),
            sigma1.mul(&gamma[a][b]).mul(&GaussianRational::new(0, 1)),
        );
        add_boson(
            &mut result,
            Field::TildeB,
            Some(mu),
            sigma2
                .mul(&gamma5_gamma[a][b])
                .mul(&GaussianRational::new(0, -1)),
        );
    }

    let gamma5 = clifford.lower_spinors(&clifford.gamma5);
    add_boson(
        &mut result,
        Field::F,
        None,
        sigma3
            .mul(&clifford.charge_conjugation[a][b])
            .mul(&GaussianRational::new(0, -1)),
    );
    add_boson(&mut result, Field::G, None, delta.mul(&gamma5[a][b]));
    add_boson(
        &mut result,
        Field::TildeF,
        None,
        sigma1
            .mul(&clifford.charge_conjugation[a][b])
            .mul(&GaussianRational::new(0, -1)),
    );
    add_boson(
        &mut result,
        Field::TildeG,
        None,
        sigma2.mul(&gamma5[a][b]).mul(&GaussianRational::new(0, 1)),
    );
    result
}

/// arXiv:1405.0048 Eq. (13), in the repository's exact Clifford convention.
fn delta(charge: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let identity = identity4();
    let gamma5 = clifford.gamma5;
    match field {
        Field::A => {
            let mut result = Poly::default();
            add_fermion_tensor_row(
                &mut result,
                &internal_matrix(3),
                &identity,
                charge,
                None,
                GaussianRational::new(1, 0),
            );
            result
        }
        Field::B => {
            let mut result = Poly::default();
            add_fermion_tensor_row(
                &mut result,
                &internal_matrix(0),
                &gamma5,
                charge,
                None,
                GaussianRational::new(0, 1),
            );
            result
        }
        Field::F => {
            let mut result = Poly::default();
            for mu in 0..DIM {
                add_fermion_tensor_row(
                    &mut result,
                    &internal_matrix(3),
                    &clifford.gamma_up[mu],
                    charge,
                    Some(mu),
                    GaussianRational::new(1, 0),
                );
            }
            result
        }
        Field::G => {
            let mut result = Poly::default();
            for mu in 0..DIM {
                add_fermion_tensor_row(
                    &mut result,
                    &internal_matrix(0),
                    &clifford.gamma5_gamma_up(mu),
                    charge,
                    Some(mu),
                    GaussianRational::new(0, 1),
                );
            }
            result
        }
        Field::TildeA => {
            let mut result = Poly::default();
            add_fermion_tensor_row(
                &mut result,
                &internal_matrix(1),
                &identity,
                charge,
                None,
                GaussianRational::new(1, 0),
            );
            result
        }
        Field::TildeB => {
            let mut result = Poly::default();
            add_fermion_tensor_row(
                &mut result,
                &internal_matrix(2),
                &gamma5,
                charge,
                None,
                GaussianRational::new(-1, 0),
            );
            result
        }
        Field::TildeF => {
            let mut result = Poly::default();
            for mu in 0..DIM {
                add_fermion_tensor_row(
                    &mut result,
                    &internal_matrix(1),
                    &clifford.gamma_up[mu],
                    charge,
                    Some(mu),
                    GaussianRational::new(1, 0),
                );
            }
            result
        }
        Field::TildeG => {
            let mut result = Poly::default();
            for mu in 0..DIM {
                add_fermion_tensor_row(
                    &mut result,
                    &internal_matrix(2),
                    &clifford.gamma5_gamma_up(mu),
                    charge,
                    Some(mu),
                    GaussianRational::new(-1, 0),
                );
            }
            result
        }
        Field::Psi(internal, spinor) => fermion_delta(charge, internal, spinor, clifford),
    }
}

fn apply_delta(charge: Charge, polynomial: &Poly, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    for (jet, coefficient) in &polynomial.0 {
        let mut transformed = delta(charge, jet.field, clifford);
        for axis in 0..DIM {
            for _ in 0..jet.derivatives[axis] {
                transformed = transformed.derivative(axis);
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

fn d_alembertian(field: Field) -> Poly {
    let mut result = Poly::default();
    for axis in 0..DIM {
        let term = Poly::atom(field).derivative(axis).derivative(axis);
        result.add_scaled(
            &term,
            GaussianRational::new(if axis == 0 { -1 } else { 1 }, 0),
        );
    }
    result
}

fn translation(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    if left.internal == right.internal {
        let gamma = |mu| clifford.lower_spinors(&clifford.gamma_up[mu]);
        for mu in 0..DIM {
            result.add_scaled(
                &Poly::atom(field).derivative(mu),
                gamma(mu)[left.spinor][right.spinor].mul(&GaussianRational::new(0, 2)),
            );
        }
    }
    result
}

/// The single central operator obtained by factoring the common coefficient
/// `-2 sigma_2^{ij} C_ab` from arXiv:1405.0048 Eq. (16).
fn central(field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    match field {
        Field::A => result.add_scaled(&Poly::atom(Field::TildeF), GaussianRational::new(1, 0)),
        Field::B => result.add_scaled(&Poly::atom(Field::TildeG), GaussianRational::new(1, 0)),
        Field::F => result.add_scaled(&d_alembertian(Field::TildeA), GaussianRational::new(1, 0)),
        Field::G => result.add_scaled(&d_alembertian(Field::TildeB), GaussianRational::new(1, 0)),
        Field::TildeA => result.add_scaled(&Poly::atom(Field::F), GaussianRational::new(-1, 0)),
        Field::TildeB => result.add_scaled(&Poly::atom(Field::G), GaussianRational::new(-1, 0)),
        Field::TildeF => result.add_scaled(&d_alembertian(Field::A), GaussianRational::new(-1, 0)),
        Field::TildeG => result.add_scaled(&d_alembertian(Field::B), GaussianRational::new(-1, 0)),
        Field::Psi(internal, spinor) => {
            let sigma2 = pauli(2);
            for other_internal in 0..INTERNAL {
                for mu in 0..DIM {
                    for other_spinor in 0..SPINORS {
                        result.add_scaled(
                            &Poly::atom(Field::Psi(other_internal, other_spinor)).derivative(mu),
                            sigma2[internal][other_internal]
                                .mul(&clifford.gamma_up[mu][spinor][other_spinor])
                                .mul(&GaussianRational::new(0, 1)),
                        );
                    }
                }
            }
        }
    }
    result
}

fn central_occurrence(left: Charge, right: Charge, clifford: &Clifford4D) -> GaussianRational {
    pauli(2)[left.internal][right.internal]
        .mul(&clifford.charge_conjugation[left.spinor][right.spinor])
        .mul(&GaussianRational::new(-2, 0))
}

/// arXiv:1405.0048 Eq. (16), written as translation plus one central action.
fn published_closure(left: Charge, right: Charge, field: Field, clifford: &Clifford4D) -> Poly {
    let mut result = translation(left, right, field, clifford);
    result.add_scaled(
        &central(field, clifford),
        central_occurrence(left, right, clifford),
    );
    result
}

fn apply_central(polynomial: &Poly, clifford: &Clifford4D) -> Poly {
    let mut result = Poly::default();
    for (jet, coefficient) in &polynomial.0 {
        let mut transformed = central(jet.field, clifford);
        for axis in 0..DIM {
            for _ in 0..jet.derivatives[axis] {
                transformed = transformed.derivative(axis);
            }
        }
        result.add_scaled(&transformed, *coefficient);
    }
    result
}

fn central_square_expected(field: Field) -> Poly {
    let mut expected = d_alembertian(field);
    let original = expected.clone();
    expected = Poly::default();
    expected.add_scaled(&original, GaussianRational::new(-1, 0));
    expected
}

fn real_unit(value: GaussianRational) -> i16 {
    assert_eq!(value.imag.numer(), &0);
    assert_eq!(value.real.denom(), &1);
    i16::try_from(*value.real.numer()).expect("worldline coefficient fits i16")
}

fn worldline_fields() -> [Field; WORLDLINE_DIMENSION] {
    [
        Field::A,
        Field::B,
        Field::F,
        Field::G,
        Field::TildeA,
        Field::TildeB,
        Field::TildeF,
        Field::TildeG,
    ]
}

/// Implements Eq. (23).  Entries 2,3,6,7 denote the lowered nodes whose
/// time derivatives are F,G,F-tilde,G-tilde.
fn source_worldline_l_matrices(clifford: &Clifford4D) -> [WorldlineMatrix; WORLDLINE_DIMENSION] {
    let charges = charges();
    let nodes = worldline_fields();
    let mut matrices = [[[0_i16; WORLDLINE_DIMENSION]; WORLDLINE_DIMENSION]; WORLDLINE_DIMENSION];
    for (charge_index, &charge) in charges.iter().enumerate() {
        for (node, &field) in nodes.iter().enumerate() {
            let lowered_auxiliary =
                matches!(field, Field::F | Field::G | Field::TildeF | Field::TildeG);
            for (jet, coefficient) in delta(charge, field, clifford).0 {
                let Field::Psi(internal, spinor) = jet.field else {
                    panic!("bosonic transformation must target a fermion")
                };
                let expected_derivatives = if lowered_auxiliary {
                    [1, 0, 0, 0]
                } else {
                    [0; DIM]
                };
                if jet.derivatives == expected_derivatives {
                    matrices[charge_index][node][internal * SPINORS + spinor] =
                        real_unit(coefficient);
                } else {
                    assert!(
                        jet.derivatives[1..].iter().any(|&degree| degree != 0),
                        "unexpected temporal worldline derivative"
                    );
                }
            }
        }
    }
    matrices
}

fn committed_cc_l_matrices() -> [WorldlineMatrix; WORLDLINE_DIMENSION] {
    crate::vector_tensor_central_charge::l_matrices(
        0,
        crate::permutahedron_s8_supersymmetry::S8_BASE_BOOLEAN_FACTORS[0],
    )
}

/// Worldline form of Eqs. (17)-(21). The row convention is the same as the
/// linkage matrices: a row is the source node and a column is the target node.
/// Auxiliary nodes denote their time primitives, as in Eq. (23).
fn source_worldline_central_matrices(clifford: &Clifford4D) -> (WorldlineMatrix, WorldlineMatrix) {
    let nodes = worldline_fields();
    let node_indices: BTreeMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(index, &field)| (field, index))
        .collect();
    let mut bosonic = [[0_i16; WORLDLINE_DIMENSION]; WORLDLINE_DIMENSION];
    for (source, &field) in nodes.iter().enumerate() {
        let source_lowered = usize::from(matches!(
            field,
            Field::F | Field::G | Field::TildeF | Field::TildeG
        ));
        for (jet, coefficient) in central(field, clifford).0 {
            if jet.derivatives[1..].iter().any(|&degree| degree != 0) {
                continue;
            }
            let target_lowered = usize::from(matches!(
                jet.field,
                Field::F | Field::G | Field::TildeF | Field::TildeG
            ));
            assert_eq!(
                usize::from(jet.derivatives[0]) + target_lowered,
                source_lowered + 1,
                "central action must have height two after valising"
            );
            bosonic[source][node_indices[&jet.field]] = real_unit(coefficient);
        }
    }

    let mut fermionic = [[0_i16; WORLDLINE_DIMENSION]; WORLDLINE_DIMENSION];
    for internal in 0..INTERNAL {
        for spinor in 0..SPINORS {
            let source = internal * SPINORS + spinor;
            for (jet, coefficient) in central(Field::Psi(internal, spinor), clifford).0 {
                if jet.derivatives != [1, 0, 0, 0] {
                    continue;
                }
                let Field::Psi(target_internal, target_spinor) = jet.field else {
                    panic!("fermionic central action must target a fermion")
                };
                fermionic[source][target_internal * SPINORS + target_spinor] =
                    real_unit(coefficient);
            }
        }
    }
    (bosonic, fermionic)
}

fn negate_matrix(matrix: &WorldlineMatrix) -> WorldlineMatrix {
    std::array::from_fn(|row| std::array::from_fn(|column| -matrix[row][column]))
}

fn single_matrix_mismatch_count(left: &WorldlineMatrix, right: &WorldlineMatrix) -> usize {
    (0..8)
        .flat_map(|row| (0..8).map(move |column| (row, column)))
        .filter(|&(row, column)| left[row][column] != right[row][column])
        .count()
}

fn matrix_mismatch_count(left: &[WorldlineMatrix; 8], right: &[WorldlineMatrix; 8]) -> usize {
    (0..8)
        .flat_map(|color| {
            (0..8).flat_map(move |row| (0..8).map(move |column| (color, row, column)))
        })
        .filter(|&(color, row, column)| left[color][row][column] != right[color][row][column])
        .count()
}

fn canonical_coefficient(value: GaussianRational) -> CanonicalCoefficient {
    CanonicalCoefficient::new(
        *value.real.numer(),
        *value.real.denom(),
        *value.imag.numer(),
        *value.imag.denom(),
    )
    .expect("finite normalized source coefficient")
}

fn canonical_lorentz(left: u8, right: u8) -> LorentzRep {
    LorentzRep {
        left_twice_spin: left,
        right_twice_spin: right,
        reality: Reality::Real,
    }
}

fn field_label(field: Field) -> String {
    match field {
        Field::A => "A".into(),
        Field::B => "B".into(),
        Field::F => "F".into(),
        Field::G => "G".into(),
        Field::TildeA => "A_tilde".into(),
        Field::TildeB => "B_tilde".into(),
        Field::TildeF => "F_tilde".into(),
        Field::TildeG => "G_tilde".into(),
        Field::Psi(internal, spinor) => format!("psi_{}_{}", internal + 1, spinor),
    }
}

/// Full exact component linkage and central action for the canonical engine.
pub fn exact_canonical_fixture() -> CanonicalPhysicalFingerprint {
    let clifford = Clifford4D::build();
    let fields = Field::all();
    let charges = charges();
    let field_indices: BTreeMap<_, _> = fields
        .iter()
        .enumerate()
        .map(|(index, &field)| (field, index))
        .collect();
    let components = fields
        .iter()
        .map(|&field| {
            let (statistics, height_twice, role, lorentz) = match field {
                Field::A | Field::B | Field::TildeA | Field::TildeB => (
                    Statistics::Boson,
                    0,
                    CanonicalRole::Propagating,
                    canonical_lorentz(0, 0),
                ),
                Field::F | Field::G | Field::TildeF | Field::TildeG => (
                    Statistics::Boson,
                    2,
                    CanonicalRole::Auxiliary,
                    canonical_lorentz(0, 0),
                ),
                Field::Psi(_, _) => (
                    Statistics::Fermion,
                    1,
                    CanonicalRole::Propagating,
                    canonical_lorentz(1, 0),
                ),
            };
            CanonicalComponent {
                label: field_label(field),
                statistics,
                lorentz,
                height_twice,
                role,
                form_degree: None,
            }
        })
        .collect();
    let supercharges = charges
        .iter()
        .map(|charge| CanonicalSupercharge {
            label: format!("D{}_{}", charge.internal + 1, charge.spinor),
            lorentz: canonical_lorentz(1, 0),
            height_twice: 1,
        })
        .collect();
    let mut linkage = Vec::new();
    for (charge_index, &charge) in charges.iter().enumerate() {
        for &source_field in &fields {
            for (jet, coefficient) in delta(charge, source_field, &clifford).0 {
                linkage.push(CanonicalLinkage {
                    charge: charge_index,
                    source: field_indices[&source_field],
                    target: field_indices[&jet.field],
                    derivative: CanonicalDerivative(jet.derivatives),
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
    }
    let central_generators = vec![CanonicalCentralGenerator {
        label: "Z_off_shell".into(),
        lorentz: canonical_lorentz(0, 0),
        height_twice: 2,
    }];
    let mut central_entries = Vec::new();
    for &source_field in &fields {
        for (jet, coefficient) in central(source_field, &clifford).0 {
            central_entries.push(CanonicalCentralEntry {
                generator: 0,
                source: field_indices[&source_field],
                target: field_indices[&jet.field],
                derivative: CanonicalDerivative(jet.derivatives),
                coefficient: canonical_coefficient(coefficient),
            });
        }
    }
    let mut central_occurrences = Vec::new();
    for left in 0..charges.len() {
        for right in left..charges.len() {
            let coefficient = central_occurrence(charges[left], charges[right], &clifford);
            if !coefficient.is_zero() {
                central_occurrences.push(CanonicalCentralOccurrence {
                    left_charge: left,
                    right_charge: right,
                    generator: 0,
                    coefficient: canonical_coefficient(coefficient),
                });
            }
        }
    }
    CanonicalPhysicalFingerprint {
        name: "central-hypermultiplet-exact-source-adapter".into(),
        components,
        supercharges,
        linkage,
        gauge_complex: Vec::new(),
        bianchi_identities: Vec::new(),
        central_generators,
        central_entries,
        central_occurrences,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub arxiv_id: &'static str,
    pub locator: &'static str,
    pub role: &'static str,
    pub sha256: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompletenessFlags {
    pub exact_rigid_transformations_transcribed: bool,
    pub full_four_dimensional_component_closure_checked: bool,
    pub central_action_factored_from_published_closure: bool,
    pub central_commutator_checked: bool,
    pub central_square_checked: bool,
    pub source_zero_brane_map_checked: bool,
    pub appendix_a_matrices_checked: bool,
    pub committed_cc_one_z_bridge_checked: bool,
    pub interacting_or_gauged_central_charge_checked: bool,
    pub general_complex_central_charge_checked: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentralHypermultiplet4DReport {
    pub schema_version: &'static str,
    pub source: &'static str,
    pub fields: usize,
    pub bosonic_off_shell_components: usize,
    pub fermionic_off_shell_components: usize,
    pub supercharges: usize,
    pub unordered_charge_pairs: usize,
    pub component_relations_checked: usize,
    pub closure_residual_relations: usize,
    pub closure_residual_terms: usize,
    pub central_commutators_checked: usize,
    pub central_commutator_residuals: usize,
    pub central_square_relations_checked: usize,
    pub central_square_residuals: usize,
    pub central_occurrence_pairs: usize,
    pub zero_brane_l_entries_checked: usize,
    pub zero_brane_l_mismatches: usize,
    pub committed_bosonic_central_entries_checked: usize,
    pub committed_bosonic_central_mismatches: usize,
    pub committed_fermionic_central_entries_checked: usize,
    pub committed_fermionic_central_mismatches: usize,
    pub committed_extended_closure_passed: bool,
    pub canonical_fingerprint_sha256: String,
    pub canonical_linkage_terms: usize,
    pub canonical_central_entries: usize,
    pub completeness: CompletenessFlags,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentralHypermultiplet4DArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<SourceRecord>,
    pub report: CentralHypermultiplet4DReport,
    pub reduced_l_matrices: [WorldlineMatrix; WORLDLINE_DIMENSION],
    pub committed_cc_l_matrices: [WorldlineMatrix; WORLDLINE_DIMENSION],
    pub canonical_fingerprint: crate::higher_dimensional_canonical::CanonicalFingerprint,
}

pub fn verify() -> CentralHypermultiplet4DReport {
    let clifford = Clifford4D::build();
    let fields = Field::all();
    let charges = charges();
    let mut closure_residual_relations = 0usize;
    let mut closure_residual_terms = 0usize;
    for left in 0..charges.len() {
        for right in left..charges.len() {
            for &field in &fields {
                let actual = anticommutator(charges[left], charges[right], field, &clifford);
                let expected = published_closure(charges[left], charges[right], field, &clifford);
                if actual != expected {
                    closure_residual_relations += 1;
                    let mut residual = actual;
                    residual.add_scaled(&expected, GaussianRational::new(-1, 0));
                    closure_residual_terms += residual.term_count();
                }
            }
        }
    }

    let mut central_commutator_residuals = 0usize;
    for &charge in &charges {
        for &field in &fields {
            let left = apply_delta(charge, &central(field, &clifford), &clifford);
            let right = apply_central(&delta(charge, field, &clifford), &clifford);
            central_commutator_residuals += usize::from(left != right);
        }
    }
    let mut central_square_residuals = 0usize;
    for &field in &fields {
        let square = apply_central(&central(field, &clifford), &clifford);
        central_square_residuals += usize::from(square != central_square_expected(field));
    }

    let reduced_l_matrices = source_worldline_l_matrices(&clifford);
    let committed_l_matrices = committed_cc_l_matrices();
    let zero_brane_l_mismatches = matrix_mismatch_count(&reduced_l_matrices, &committed_l_matrices);
    let committed = crate::vector_tensor_central_charge::build();
    let cc = committed
        .sectors
        .iter()
        .find(|sector| sector.id == "CC")
        .expect("committed CC sector");
    let cc_branch = &cc.branches[0];
    let cc_central = cc_branch
        .central_charge
        .as_ref()
        .expect("committed CC one-Z operator");
    let (source_bosonic_central, source_fermionic_central) =
        source_worldline_central_matrices(&clifford);
    let committed_bosonic_central_mismatches =
        single_matrix_mismatch_count(&source_bosonic_central, &negate_matrix(&cc_central.bosonic));
    let committed_fermionic_central_mismatches = single_matrix_mismatch_count(
        &source_fermionic_central,
        &negate_matrix(&cc_central.fermionic),
    );

    let canonical_fixture = exact_canonical_fixture();
    let canonical = crate::higher_dimensional_canonical::canonicalize(
        &canonical_fixture,
        &crate::higher_dimensional_canonical::CanonicalOptions::default(),
    )
    .expect("canonicalize exact central hypermultiplet");
    let central_occurrence_pairs = canonical_fixture.central_occurrences.len();
    let completeness = CompletenessFlags {
        exact_rigid_transformations_transcribed: true,
        full_four_dimensional_component_closure_checked: closure_residual_relations == 0,
        central_action_factored_from_published_closure: central_occurrence_pairs == 4,
        central_commutator_checked: central_commutator_residuals == 0,
        central_square_checked: central_square_residuals == 0,
        source_zero_brane_map_checked: zero_brane_l_mismatches == 0,
        appendix_a_matrices_checked: zero_brane_l_mismatches == 0,
        committed_cc_one_z_bridge_checked: zero_brane_l_mismatches == 0
            && committed_bosonic_central_mismatches == 0
            && committed_fermionic_central_mismatches == 0
            && cc_branch.one_central_charge_completion
            && cc_central.extended_closure_passed
            && cc_central.commutes_with_all_supercharges,
        interacting_or_gauged_central_charge_checked: false,
        general_complex_central_charge_checked: false,
    };
    let passed = clifford.verifies_source_conventions()
        && fields.len() == 16
        && closure_residual_relations == 0
        && central_commutator_residuals == 0
        && central_square_residuals == 0
        && zero_brane_l_mismatches == 0
        && committed_bosonic_central_mismatches == 0
        && committed_fermionic_central_mismatches == 0
        && completeness.committed_cc_one_z_bridge_checked
        && canonical.central_generator_count == 1;
    CentralHypermultiplet4DReport {
        schema_version: SCHEMA_VERSION,
        source: "arXiv:1405.0048 Sec. 3.1 Eqs. (13)-(31), Appendix A",
        fields: fields.len(),
        bosonic_off_shell_components: 8,
        fermionic_off_shell_components: 8,
        supercharges: charges.len(),
        unordered_charge_pairs: charges.len() * (charges.len() + 1) / 2,
        component_relations_checked: charges.len() * (charges.len() + 1) / 2 * fields.len(),
        closure_residual_relations,
        closure_residual_terms,
        central_commutators_checked: charges.len() * fields.len(),
        central_commutator_residuals,
        central_square_relations_checked: fields.len(),
        central_square_residuals,
        central_occurrence_pairs,
        zero_brane_l_entries_checked: 8 * 8 * 8,
        zero_brane_l_mismatches,
        committed_bosonic_central_entries_checked: 8 * 8,
        committed_bosonic_central_mismatches,
        committed_fermionic_central_entries_checked: 8 * 8,
        committed_fermionic_central_mismatches,
        committed_extended_closure_passed: cc_central.extended_closure_passed,
        canonical_fingerprint_sha256: canonical.sha256,
        canonical_linkage_terms: canonical_fixture.linkage.len(),
        canonical_central_entries: canonical_fixture.central_entries.len(),
        completeness,
        passed,
        boundary: "This certifies the free rigid real-central-charge Wess-Fayet component system and its exact CC worldline bridge. It does not certify an interacting hypermultiplet, a gauged central charge, a general complex central charge, or a central-charge-free finite hypermultiplet.",
    }
}

pub fn build() -> CentralHypermultiplet4DArtifact {
    let clifford = Clifford4D::build();
    let canonical_fixture = exact_canonical_fixture();
    let canonical_fingerprint = crate::higher_dimensional_canonical::canonicalize(
        &canonical_fixture,
        &crate::higher_dimensional_canonical::CanonicalOptions::default(),
    )
    .expect("canonicalize central hypermultiplet");
    CentralHypermultiplet4DArtifact {
        schema_version: "central-hypermultiplet-4d-artifact-v1",
        title: "Exact rigid 4D N=2 Wess-Fayet hypermultiplet with one off-shell central charge",
        sources: vec![
            SourceRecord {
                arxiv_id: "1405.0048",
                locator: "Sec. 3.1 Eqs. (13)-(31), Eq. (23), Appendix A",
                role: "component transformations, closure, central symmetry, and 0-brane matrices",
                sha256: "8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20",
            },
            SourceRecord {
                arxiv_id: "hep-th/9607216",
                locator: "Secs. 1-2",
                role: "independent superspace account of the Fayet-Sohnius hypermultiplet with complex central charge",
                sha256: "d12f250efdbe423288cb4f72f70afb189e79d2903eaef369c958f02ab31c3b6d",
            },
        ],
        report: verify(),
        reduced_l_matrices: source_worldline_l_matrices(&clifford),
        committed_cc_l_matrices: committed_cc_l_matrices(),
        canonical_fingerprint,
    }
}

pub fn write_artifacts(data_path: &Path, validation_path: &Path) -> CentralHypermultiplet4DReport {
    let artifact = build();
    if let Some(parent) = data_path.parent() {
        std::fs::create_dir_all(parent).expect("create central hypermultiplet data directory");
    }
    if let Some(parent) = validation_path.parent() {
        std::fs::create_dir_all(parent).expect("create central hypermultiplet result directory");
    }
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(data_path).expect("create central hypermultiplet artifact")),
        &artifact,
    )
    .expect("write central hypermultiplet artifact");
    serde_json::to_writer_pretty(
        BufWriter::new(
            File::create(validation_path).expect("create central hypermultiplet validation"),
        ),
        &artifact.report,
    )
    .expect("write central hypermultiplet validation");
    artifact.report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_four_dimensional_closure_and_central_action_pass() {
        let report = verify();
        assert!(report.passed);
        assert_eq!(report.component_relations_checked, 576);
        assert_eq!(report.closure_residual_relations, 0);
        assert_eq!(report.central_commutator_residuals, 0);
        assert_eq!(report.central_square_residuals, 0);
        assert_eq!(report.central_occurrence_pairs, 4);
    }

    #[test]
    fn source_zero_brane_map_recovers_appendix_a_and_committed_cc() {
        let report = verify();
        assert_eq!(report.zero_brane_l_entries_checked, 512);
        assert_eq!(report.zero_brane_l_mismatches, 0);
        assert_eq!(report.committed_bosonic_central_mismatches, 0);
        assert_eq!(report.committed_fermionic_central_mismatches, 0);
        assert!(report.completeness.appendix_a_matrices_checked);
        assert!(report.completeness.committed_cc_one_z_bridge_checked);
    }

    #[test]
    fn canonical_adapter_retains_one_central_generator_and_no_gauge_complex() {
        let fixture = exact_canonical_fixture();
        let canonical = crate::higher_dimensional_canonical::canonicalize(
            &fixture,
            &crate::higher_dimensional_canonical::CanonicalOptions::default(),
        )
        .unwrap();
        assert_eq!(fixture.components.len(), 16);
        assert_eq!(fixture.supercharges.len(), 8);
        assert!(fixture.gauge_complex.is_empty());
        assert!(fixture.bianchi_identities.is_empty());
        assert_eq!(fixture.central_generators.len(), 1);
        assert_eq!(fixture.central_occurrences.len(), 4);
        assert_eq!(canonical.central_generator_count, 1);
    }

    #[test]
    fn completeness_flags_do_not_overclaim_interacting_or_complex_cases() {
        let report = verify();
        assert!(
            report
                .completeness
                .full_four_dimensional_component_closure_checked
        );
        assert!(report.completeness.central_commutator_checked);
        assert!(
            !report
                .completeness
                .interacting_or_gauged_central_charge_checked
        );
        assert!(!report.completeness.general_complex_central_charge_checked);
    }
}
