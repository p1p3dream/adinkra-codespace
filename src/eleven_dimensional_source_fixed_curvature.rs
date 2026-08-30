//! Source-fixed algebraic scaffold for the 11D semi-prepotential curvature.
//!
//! This module keeps three logically different objects separate:
//!
//! 1. the rank-320 gamma-trace projector and its rank-32 representative
//!    redundancy, fixed by arXiv:2007.05097 Eqs. (2.2)-(2.3);
//! 2. the irreducible two-form-vector and five-form-vector projections fixed
//!    by hep-th/0101037 Eqs. (39)-(40); and
//! 3. the source coefficients in the linearized field-strength definitions
//!    of hep-th/0101037 Eq. (44).
//!
//! The source does not print the fully eliminated differential operator from
//! `H_hat` to `(W, X_[2], X_[5], J)`.  In particular, the spin connections,
//! the raised-spinor gamma convention, and the complete superspace derivative
//! normal ordering still have to be joined.  The API below therefore begins
//! after the indicated Clifford contractions and fails closed at that boundary.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

use num_bigint::BigInt;
use num_rational::Ratio;
use serde::Serialize;

pub const VECTOR_DIMENSION: usize = 11;
pub const SPINOR_DIMENSION: usize = 32;
pub const AMBIENT_VECTOR_SPINOR_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
pub const GAMMA_TRACE_DIMENSION: usize = 32;
pub const GAMMA_TRACELESS_DIMENSION: usize = 320;

pub const HEP_TH_0101037_PDF_SHA256: &str =
    "3d40a1b32fa4491dee56b3e99802172d2c5039b2de198b987ce121a1bbb15cc3";
pub const HEP_TH_0101037_SOURCE_SHA256: &str =
    "9405ca44a0036567cf86bfbc89de097d8b064612c314b28f31d614e4553a4453";
pub const ARXIV_2007_05097_PDF_SHA256: &str =
    "197604bc6b5c9e0dfb12044d981aae467920f46554ba9371f1eb9b6389d00a73";

/// Exact linearized source formulas, transcribed in index-preserving ASCII.
///
/// The source labels in the submitted TeX are `eq:05`, `eq:06`, `eq:13`,
/// `eq:13A`, `eq:14`, `eq:15`, `X's`, `eq:043A`, and `eq:01Z`.  They are
/// numbered (24)-(29), (39)-(40), and (44) in the rendered paper. Irreducible
/// form projections retain normalized antisymmetrization. The vector-vector
/// brackets in Eq. (29) are the paper's unnormalized curl convention, as
/// fixed by its Eqs. (7)-(8).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct LinearizedSourceFormula {
    pub rendered_equation: &'static str,
    pub source_tex_label: &'static str,
    pub pdf_page: usize,
    pub formula_ascii: &'static str,
}

pub const LINEARIZED_SOURCE_FORMULAS: &[LinearizedSourceFormula] = &[
    LinearizedSourceFormula {
        rendered_equation: "(24)",
        source_tex_label: "eq:05",
        pdf_page: 7,
        formula_ascii: "E_alpha = D_alpha + (1/2)(Delta_alpha^gamma + Psi delta_alpha^gamma) D_gamma + H_alpha^c partial_c",
    },
    LinearizedSourceFormula {
        rendered_equation: "(25)",
        source_tex_label: "eq:06",
        pdf_page: 7,
        formula_ascii: "E_a = partial_a + (i/32)[D_beta(gamma_a Delta)^(beta gamma) + (D_beta Psi)(gamma_a)^(beta gamma)]D_gamma + [(i/16)(gamma_a)^(alpha beta)D_alpha H_beta^c + delta_a^c Psi - Psi_a^c]partial_c",
    },
    LinearizedSourceFormula {
        rendered_equation: "(26)",
        source_tex_label: "eq:13",
        pdf_page: 8,
        formula_ascii: "C_(alpha beta)^epsilon = (1/64)[(gamma^[2])_(alpha beta)(gamma_[2])^(gamma delta) - (1/60)(gamma^[5])_(alpha beta)(gamma_[5])^(gamma delta)][D_gamma Delta_delta^epsilon + delta_delta^epsilon D_gamma Psi]",
    },
    LinearizedSourceFormula {
        rendered_equation: "(27)",
        source_tex_label: "eq:13A",
        pdf_page: 8,
        formula_ascii: "C_(alpha beta)^c = i(gamma^c)_(alpha beta) - (1/32)(gamma^de)_(alpha beta)[(gamma_de)^(gamma delta)D_gamma H_delta^c + 32 delta_d^c Psi_e - 16 Psi^c_de] + (i/1920)(gamma^[5])_(alpha beta)[i(gamma_[5])^(gamma delta)D_gamma H_delta^c + 40 delta_d1^c Psi_d2...d5 - (2/15)epsilon^c_d1...d5^[5] Psi_[5]]",
    },
    LinearizedSourceFormula {
        rendered_equation: "(28)",
        source_tex_label: "eq:14",
        pdf_page: 8,
        formula_ascii: "C_alpha,b^gamma = (i/32)D_alpha D_delta(gamma_b Delta)^(delta gamma) - (1/2)partial_b Delta_alpha^gamma - (1/2)(partial_b Psi)delta_alpha^gamma + (i/32)(D_alpha D_delta Psi)(gamma_b)^(delta gamma); C_alpha,b^c = (i/16)(gamma_b)^(beta gamma)D_alpha D_beta H_gamma^c - partial_b H_alpha^c + (1/32)D_beta(gamma_b Delta gamma^c)^beta_alpha - D_alpha Psi_b^c + [(D_alpha Psi)delta_b^c - (1/32)(gamma^c gamma_b)_alpha^gamma D_gamma Psi]",
    },
    LinearizedSourceFormula {
        rendered_equation: "(29)",
        source_tex_label: "eq:15",
        pdf_page: 8,
        formula_ascii: "C_ab^gamma = (i/32)[-D_beta(gamma_[a partial_b] Delta)^(beta gamma) + (D_beta partial_[a Psi)(gamma_b])^(beta gamma)]; C_ab^c = -(i/16)(gamma_[a)^(alpha beta)partial_b]D_alpha H_beta^c - partial_[a Psi_b]^c - delta_[a^c partial_b]Psi",
    },
    LinearizedSourceFormula {
        rendered_equation: "(39)",
        source_tex_label: "X's",
        pdf_page: 11,
        formula_ascii: "X_ab^c = (1/16)[(gamma_ab)^(gamma delta)D_gamma H_delta^c + 16 delta_[a^c Psi_b] - 16 Psi_ab^c]; X_a1...a5^c = (1/16)[i(gamma_a1...a5)^(gamma delta)D_gamma H_delta^c + (1/3)delta_[a1^c Psi_a2...a5] - (2/15)epsilon^c_a1...a5^[5]Psi_[5]]",
    },
    LinearizedSourceFormula {
        rendered_equation: "(40)",
        source_tex_label: "eq:043A",
        pdf_page: 11,
        formula_ascii: "X_ab^b = 0; X_[abc] = 0; X_a1...a4a5^a5 = 0; X_[a1...a5b] = 0",
    },
    LinearizedSourceFormula {
        rendered_equation: "(44), linear part",
        source_tex_label: "eq:01Z",
        pdf_page: 14,
        formula_ascii: "W_abcd = (1/32)[i(gamma^e gamma_abcd)_gamma^alpha T_alpha,e^gamma - (1/3)(gamma_abcd)^(alpha beta)nabla_alpha T_beta,e^e]; X_ab^c = (1/32)(gamma_ab)^(alpha beta)T_alpha,beta^c; X_a1...a5^c = (i/32)(gamma_a1...a5)^(alpha beta)T_alpha,beta^c; J_alpha = (4/33)T_alpha,b^b",
    },
];

pub type BigRational = Ratio<BigInt>;

fn q(value: i64) -> BigRational {
    Ratio::from_integer(BigInt::from(value))
}

fn qr(numerator: i64, denominator: i64) -> BigRational {
    Ratio::new(BigInt::from(numerator), BigInt::from(denominator))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactGaussian {
    pub real: BigRational,
    pub imaginary: BigRational,
}

impl ExactGaussian {
    pub fn zero() -> Self {
        Self {
            real: q(0),
            imaginary: q(0),
        }
    }

    pub fn from_real(value: BigRational) -> Self {
        Self {
            real: value,
            imaginary: q(0),
        }
    }

    fn scaled(&self, factor: &BigRational) -> Self {
        Self {
            real: self.real.clone() * factor.clone(),
            imaginary: self.imaginary.clone() * factor.clone(),
        }
    }

    fn times_i(&self) -> Self {
        Self {
            real: -self.imaginary.clone(),
            imaginary: self.real.clone(),
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.real += other.real.clone();
        self.imaginary += other.imaginary.clone();
    }

    fn is_zero(&self) -> bool {
        self.real == q(0) && self.imaginary == q(0)
    }
}

/// Sparse tensor `Y_[a1...ap]^c` in a Cartesian Lorentz basis.
///
/// `form_mask` uses canonical increasing lower indices.  The vector index is
/// upper.  This convention makes the trace in Eq. (40) an ordinary mixed
/// contraction while lowering the upper index for total antisymmetrization
/// introduces the mostly-plus Lorentz metric.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormVectorTensor {
    pub degree: usize,
    pub components: BTreeMap<(u16, usize), ExactGaussian>,
}

impl FormVectorTensor {
    pub fn zero(degree: usize) -> Self {
        assert!(degree <= VECTOR_DIMENSION);
        Self {
            degree,
            components: BTreeMap::new(),
        }
    }

    pub fn add_component(&mut self, form_mask: u16, vector_index: usize, value: ExactGaussian) {
        assert_eq!(form_mask.count_ones() as usize, self.degree);
        assert!(form_mask < (1_u16 << VECTOR_DIMENSION));
        assert!(vector_index < VECTOR_DIMENSION);
        if value.is_zero() {
            return;
        }
        let entry = self
            .components
            .entry((form_mask, vector_index))
            .or_insert_with(ExactGaussian::zero);
        entry.add_assign(&value);
        if entry.is_zero() {
            self.components.remove(&(form_mask, vector_index));
        }
    }

    pub fn scaled(&self, factor: &BigRational) -> Self {
        let mut result = Self::zero(self.degree);
        for (&(mask, vector), value) in &self.components {
            result.add_component(mask, vector, value.scaled(factor));
        }
        result
    }

    pub fn times_i(&self) -> Self {
        let mut result = Self::zero(self.degree);
        for (&(mask, vector), value) in &self.components {
            result.add_component(mask, vector, value.times_i());
        }
        result
    }

    fn subtract_assign(&mut self, other: &Self) {
        assert_eq!(self.degree, other.degree);
        for (&(mask, vector), value) in &other.components {
            self.add_component(mask, vector, value.scaled(&q(-1)));
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseForm {
    pub degree: usize,
    pub components: BTreeMap<u16, ExactGaussian>,
}

impl SparseForm {
    pub fn zero(degree: usize) -> Self {
        Self {
            degree,
            components: BTreeMap::new(),
        }
    }

    fn add_component(&mut self, mask: u16, value: ExactGaussian) {
        assert_eq!(mask.count_ones() as usize, self.degree);
        if value.is_zero() {
            return;
        }
        let entry = self
            .components
            .entry(mask)
            .or_insert_with(ExactGaussian::zero);
        entry.add_assign(&value);
        if entry.is_zero() {
            self.components.remove(&mask);
        }
    }
}

fn lorentz_sign(index: usize) -> i64 {
    if index == 0 { -1 } else { 1 }
}

fn majorana_gammas() -> &'static Vec<Vec<Vec<i8>>> {
    static GAMMAS: OnceLock<Vec<Vec<Vec<i8>>>> = OnceLock::new();
    GAMMAS.get_or_init(crate::eleven_dimensional_majorana::real_gamma_matrices)
}

fn insertion_sign(mask: u16, index: usize) -> i64 {
    let greater = (mask >> (index + 1)).count_ones();
    if greater % 2 == 0 { 1 } else { -1 }
}

fn position_sign(mask_without_index: u16, index: usize) -> i64 {
    let less = (mask_without_index & ((1_u16 << index) - 1)).count_ones();
    if less % 2 == 0 { 1 } else { -1 }
}

/// The normalized total antisymmetrization `Y_[a1...ap,c]`.
pub fn total_antisymmetric_part(tensor: &FormVectorTensor) -> SparseForm {
    let mut result = SparseForm::zero(tensor.degree + 1);
    let normalization = qr(1, (tensor.degree + 1) as i64);
    for (&(mask, vector), value) in &tensor.components {
        if mask & (1_u16 << vector) != 0 {
            continue;
        }
        let sign = insertion_sign(mask, vector) * lorentz_sign(vector);
        result.add_component(
            mask | (1_u16 << vector),
            value.scaled(&(q(sign) * normalization.clone())),
        );
    }
    result
}

fn inject_total_antisymmetric(form: &SparseForm) -> FormVectorTensor {
    assert!(form.degree > 0);
    let mut result = FormVectorTensor::zero(form.degree - 1);
    for (&mask, value) in &form.components {
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) == 0 {
                continue;
            }
            let remaining = mask ^ (1_u16 << vector);
            let sign = insertion_sign(remaining, vector) * lorentz_sign(vector);
            result.add_component(remaining, vector, value.scaled(&q(sign)));
        }
    }
    result
}

/// The mixed trace `Y_[a1...a(p-1)b]^b` with the displayed lower-index order.
pub fn mixed_trace(tensor: &FormVectorTensor) -> SparseForm {
    assert!(tensor.degree > 0);
    let mut result = SparseForm::zero(tensor.degree - 1);
    for (&(mask, vector), value) in &tensor.components {
        if mask & (1_u16 << vector) == 0 {
            continue;
        }
        let remaining = mask ^ (1_u16 << vector);
        let sign = insertion_sign(remaining, vector);
        result.add_component(remaining, value.scaled(&q(sign)));
    }
    result
}

fn inject_mixed_trace(form: &SparseForm, output_degree: usize) -> FormVectorTensor {
    assert_eq!(form.degree + 1, output_degree);
    let eigenvalue_sign = if (output_degree - 1) % 2 == 0 { 1 } else { -1 };
    let eigenvalue = eigenvalue_sign * (VECTOR_DIMENSION - output_degree + 1) as i64;
    let mut result = FormVectorTensor::zero(output_degree);
    for (&mask, value) in &form.components {
        for vector in 0..VECTOR_DIMENSION {
            if mask & (1_u16 << vector) != 0 {
                continue;
            }
            let output_mask = mask | (1_u16 << vector);
            let sign = position_sign(mask, vector);
            result.add_component(output_mask, vector, value.scaled(&qr(sign, eigenvalue)));
        }
    }
    result
}

/// Project `Lambda^p V* tensor V` to its trace-free, non-antisymmetric hook.
///
/// For `p=2` this is the 429-dimensional `(11000)` sector.  For `p=5`
/// this is the 4,290-dimensional `(10002)` sector.  Applying this projector
/// to the first terms of Eq. (39) eliminates the p=1,3 and p=4,5 holonomy
/// forms because those forms occupy exactly the trace and exterior summands.
pub fn irreducible_form_vector_hook(tensor: &FormVectorTensor) -> FormVectorTensor {
    assert!(tensor.degree > 0 && tensor.degree < VECTOR_DIMENSION);
    let mut result = tensor.clone();
    result.subtract_assign(&inject_total_antisymmetric(&total_antisymmetric_part(
        tensor,
    )));
    result.subtract_assign(&inject_mixed_trace(&mixed_trace(tensor), tensor.degree));
    result
}

fn gamma_trace_injection(lambda: &[BigRational]) -> Vec<BigRational> {
    assert_eq!(lambda.len(), SPINOR_DIMENSION);
    let gammas = majorana_gammas();
    let mut result = vec![q(0); AMBIENT_VECTOR_SPINOR_DIMENSION];
    for vector in 0..VECTOR_DIMENSION {
        for row in 0..SPINOR_DIMENSION {
            for column in 0..SPINOR_DIMENSION {
                let coefficient = gammas[vector][row][column];
                if coefficient != 0 {
                    result[vector * SPINOR_DIMENSION + row] +=
                        q(i64::from(coefficient)) * lambda[column].clone();
                }
            }
        }
    }
    result
}

fn gamma_trace(field: &[BigRational]) -> Vec<BigRational> {
    assert_eq!(field.len(), AMBIENT_VECTOR_SPINOR_DIMENSION);
    let gammas = majorana_gammas();
    let mut result = vec![q(0); SPINOR_DIMENSION];
    for vector in 0..VECTOR_DIMENSION {
        let metric = lorentz_sign(vector);
        for row in 0..SPINOR_DIMENSION {
            for column in 0..SPINOR_DIMENSION {
                let coefficient = i64::from(gammas[vector][row][column]) * metric;
                if coefficient != 0 {
                    result[row] +=
                        q(coefficient) * field[vector * SPINOR_DIMENSION + column].clone();
                }
            }
        }
    }
    result
}

/// Eq. (2.2) of arXiv:2007.05097 in the explicit real Majorana basis.
pub fn project_gamma_traceless(field: &[BigRational]) -> Vec<BigRational> {
    assert_eq!(field.len(), AMBIENT_VECTOR_SPINOR_DIMENSION);
    let trace = gamma_trace(field);
    let correction = gamma_trace_injection(&trace);
    field
        .iter()
        .zip(correction)
        .map(|(value, correction)| value.clone() - correction * qr(1, 11))
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearizedCurvatureContractions {
    /// `(gamma_[2])^(gamma delta) D_gamma H_delta^c` before the `1/16`.
    pub gamma_two_d_h: FormVectorTensor,
    /// `(gamma_[5])^(gamma delta) D_gamma H_delta^c` before the `i/16`.
    pub gamma_five_d_h: FormVectorTensor,
    /// `T_{alpha b}^b`, before the `4/33` in Eq. (44).
    pub torsion_vector_trace: Vec<ExactGaussian>,
    /// `(gamma^e gamma_[4])_gamma^alpha T_{alpha e}^gamma` by four-form mask.
    pub w_torsion_contraction: BTreeMap<u16, ExactGaussian>,
    /// `(gamma_[4])^(alpha beta) nabla_alpha T_{beta e}^e` by four-form mask.
    pub w_trace_derivative_contraction: BTreeMap<u16, ExactGaussian>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearizedCurvatureImage {
    pub x_two_hook_11000: FormVectorTensor,
    pub x_five_hook_10002: FormVectorTensor,
    pub j_spinor: Vec<ExactGaussian>,
    pub w_four_form: BTreeMap<u16, ExactGaussian>,
}

/// Apply the coefficients in hep-th/0101037 Eqs. (39)-(40) and (44).
///
/// The input names make explicit that all raised-spinor Clifford contractions
/// and the construction of torsion from Eqs. (24)-(29) occur upstream.  This
/// function is therefore source-fixed but is not the still-missing complete
/// differential operator `F: H_hat -> (W,X,J)`.
pub fn apply_linearized_source_fixed_f(
    input: &LinearizedCurvatureContractions,
) -> LinearizedCurvatureImage {
    assert_eq!(input.gamma_two_d_h.degree, 2);
    assert_eq!(input.gamma_five_d_h.degree, 5);
    assert_eq!(input.torsion_vector_trace.len(), SPINOR_DIMENSION);

    let x_two_hook_11000 = irreducible_form_vector_hook(&input.gamma_two_d_h).scaled(&qr(1, 16));
    let x_five_hook_10002 = irreducible_form_vector_hook(&input.gamma_five_d_h)
        .times_i()
        .scaled(&qr(1, 16));
    let j_spinor = input
        .torsion_vector_trace
        .iter()
        .map(|value| value.scaled(&qr(4, 33)))
        .collect();

    let mut w_four_form = BTreeMap::new();
    for mask in input
        .w_torsion_contraction
        .keys()
        .chain(input.w_trace_derivative_contraction.keys())
    {
        assert_eq!(mask.count_ones(), 4);
        let mut value = input
            .w_torsion_contraction
            .get(mask)
            .cloned()
            .unwrap_or_else(ExactGaussian::zero)
            .times_i();
        let derivative = input
            .w_trace_derivative_contraction
            .get(mask)
            .cloned()
            .unwrap_or_else(ExactGaussian::zero)
            .scaled(&qr(-1, 3));
        value.add_assign(&derivative);
        value = value.scaled(&qr(1, 32));
        if !value.is_zero() {
            w_four_form.insert(*mask, value);
        }
    }

    LinearizedCurvatureImage {
        x_two_hook_11000,
        x_five_hook_10002,
        j_spinor,
        w_four_form,
    }
}

/// Typed join point for the exact target-resolved composition stream.
///
/// The current stream uses the B5 weight basis for both the target vector and
/// spinor.  A caller must perform the convention-fixed raised-spinor gamma
/// contractions before adding a term here.  Keeping the original entry in the
/// method signature prevents an unlabelled 320-coordinate vector from being
/// mistaken for Cartesian `H_alpha^a`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetResolvedContractionAccumulator {
    pub contractions: LinearizedCurvatureContractions,
    pub consumed_target_entries: u64,
}

impl TargetResolvedContractionAccumulator {
    pub fn new() -> Self {
        Self {
            contractions: LinearizedCurvatureContractions {
                gamma_two_d_h: FormVectorTensor::zero(2),
                gamma_five_d_h: FormVectorTensor::zero(5),
                torsion_vector_trace: vec![ExactGaussian::zero(); SPINOR_DIMENSION],
                w_torsion_contraction: BTreeMap::new(),
                w_trace_derivative_contraction: BTreeMap::new(),
            },
            consumed_target_entries: 0,
        }
    }

    pub fn add_gamma_contracted_x_term(
        &mut self,
        entry: &crate::eleven_dimensional_level16_couplings::TargetResolvedGaugeCompositionEntry,
        gamma_rank: usize,
        cartesian_form_mask: u16,
        cartesian_output_vector: usize,
        contraction: ExactGaussian,
    ) {
        assert!(entry.target_basis_ordinal < GAMMA_TRACELESS_DIMENSION);
        assert!(entry.target_vector_weight_index < VECTOR_DIMENSION);
        assert!(entry.target_spinor_weight_index < SPINOR_DIMENSION);
        assert_eq!(cartesian_form_mask.count_ones() as usize, gamma_rank);
        assert!(cartesian_output_vector < VECTOR_DIMENSION);

        let entry_coefficient = ExactGaussian {
            real: entry.real.clone(),
            imaginary: entry.imaginary.clone(),
        };
        let product = ExactGaussian {
            real: entry_coefficient.real.clone() * contraction.real.clone()
                - entry_coefficient.imaginary.clone() * contraction.imaginary.clone(),
            imaginary: entry_coefficient.real * contraction.imaginary
                + entry_coefficient.imaginary * contraction.real,
        };
        match gamma_rank {
            2 => self.contractions.gamma_two_d_h.add_component(
                cartesian_form_mask,
                cartesian_output_vector,
                product,
            ),
            5 => self.contractions.gamma_five_d_h.add_component(
                cartesian_form_mask,
                cartesian_output_vector,
                product,
            ),
            _ => panic!("only the source-fixed gamma ranks 2 and 5 are accepted"),
        }
        self.consumed_target_entries += 1;
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectorCertificate {
    pub ambient_dimension: usize,
    pub projector_rank: usize,
    pub gamma_trace_injection_rank: usize,
    pub clifford_residual_entries: usize,
    pub projector_idempotence_residual_entries: usize,
    pub projected_gamma_trace_residual_entries: usize,
    pub projector_times_trace_injection_residual_entries: usize,
    pub trace_left_inverse_residual_entries: usize,
    pub kernel_equals_trace_image: bool,
    pub passed: bool,
}

pub fn certify_gamma_trace_projector() -> ProjectorCertificate {
    let gammas = majorana_gammas();
    let mut clifford_residual_entries = 0;
    for a in 0..VECTOR_DIMENSION {
        for b in 0..VECTOR_DIMENSION {
            for row in 0..SPINOR_DIMENSION {
                for column in 0..SPINOR_DIMENSION {
                    let mut value = 0_i64;
                    for middle in 0..SPINOR_DIMENSION {
                        value += i64::from(gammas[a][row][middle])
                            * i64::from(gammas[b][middle][column]);
                        value += i64::from(gammas[b][row][middle])
                            * i64::from(gammas[a][middle][column]);
                    }
                    let expected = if a == b && row == column {
                        2 * lorentz_sign(a)
                    } else {
                        0
                    };
                    if value != expected {
                        clifford_residual_entries += 1;
                    }
                }
            }
        }
    }

    let mut projector_idempotence_residual_entries = 0;
    let mut projected_gamma_trace_residual_entries = 0;
    for basis in 0..AMBIENT_VECTOR_SPINOR_DIMENSION {
        let mut unit = vec![q(0); AMBIENT_VECTOR_SPINOR_DIMENSION];
        unit[basis] = q(1);
        let projected = project_gamma_traceless(&unit);
        let twice = project_gamma_traceless(&projected);
        projector_idempotence_residual_entries += projected
            .iter()
            .zip(twice)
            .filter(|(left, right)| **left != *right)
            .count();
        projected_gamma_trace_residual_entries += gamma_trace(&projected)
            .iter()
            .filter(|value| **value != q(0))
            .count();
    }

    let mut projector_times_trace_injection_residual_entries = 0;
    let mut trace_left_inverse_residual_entries = 0;
    for basis in 0..SPINOR_DIMENSION {
        let mut unit = vec![q(0); SPINOR_DIMENSION];
        unit[basis] = q(1);
        let injected = gamma_trace_injection(&unit);
        projector_times_trace_injection_residual_entries += project_gamma_traceless(&injected)
            .iter()
            .filter(|value| **value != q(0))
            .count();
        let traced = gamma_trace(&injected);
        trace_left_inverse_residual_entries += traced
            .iter()
            .enumerate()
            .filter(|(index, value)| **value != if *index == basis { q(11) } else { q(0) })
            .count();
    }

    let kernel_equals_trace_image = GAMMA_TRACE_DIMENSION + GAMMA_TRACELESS_DIMENSION
        == AMBIENT_VECTOR_SPINOR_DIMENSION
        && projector_times_trace_injection_residual_entries == 0
        && trace_left_inverse_residual_entries == 0;
    let passed = clifford_residual_entries == 0
        && projector_idempotence_residual_entries == 0
        && projected_gamma_trace_residual_entries == 0
        && projector_times_trace_injection_residual_entries == 0
        && trace_left_inverse_residual_entries == 0
        && kernel_equals_trace_image;
    ProjectorCertificate {
        ambient_dimension: AMBIENT_VECTOR_SPINOR_DIMENSION,
        projector_rank: GAMMA_TRACELESS_DIMENSION,
        gamma_trace_injection_rank: GAMMA_TRACE_DIMENSION,
        clifford_residual_entries,
        projector_idempotence_residual_entries,
        projected_gamma_trace_residual_entries,
        projector_times_trace_injection_residual_entries,
        trace_left_inverse_residual_entries,
        kernel_equals_trace_image,
        passed,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HookProjectorCertificate {
    pub form_degree: usize,
    pub ambient_dimension: usize,
    pub trace_dimension: usize,
    pub exterior_dimension: usize,
    pub hook_dimension: usize,
    pub dynkin_label: &'static str,
    pub idempotence_residual_entries: usize,
    pub trace_residual_entries: usize,
    pub total_antisymmetry_residual_entries: usize,
    pub operator_trace: String,
    pub passed: bool,
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1_usize, |value, index| value * (n - index) / (index + 1))
}

pub fn certify_hook_projector(degree: usize) -> HookProjectorCertificate {
    assert!(degree == 2 || degree == 5);
    let mut masks = Vec::new();
    for mask in 0_u16..(1_u16 << VECTOR_DIMENSION) {
        if mask.count_ones() as usize == degree {
            masks.push(mask);
        }
    }
    let ambient_dimension = masks.len() * VECTOR_DIMENSION;
    let trace_dimension = binomial(VECTOR_DIMENSION, degree - 1);
    let exterior_dimension = binomial(VECTOR_DIMENSION, degree + 1);
    let hook_dimension = ambient_dimension - trace_dimension - exterior_dimension;
    let mut idempotence_residual_entries = 0;
    let mut trace_residual_entries = 0;
    let mut total_antisymmetry_residual_entries = 0;
    let mut operator_trace = q(0);

    for &mask in &masks {
        for vector in 0..VECTOR_DIMENSION {
            let mut unit = FormVectorTensor::zero(degree);
            unit.add_component(mask, vector, ExactGaussian::from_real(q(1)));
            let projected = irreducible_form_vector_hook(&unit);
            let twice = irreducible_form_vector_hook(&projected);
            if projected != twice {
                let keys = projected
                    .components
                    .keys()
                    .chain(twice.components.keys())
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                idempotence_residual_entries += keys
                    .iter()
                    .filter(|key| projected.components.get(key) != twice.components.get(key))
                    .count();
            }
            trace_residual_entries += mixed_trace(&projected).components.len();
            total_antisymmetry_residual_entries +=
                total_antisymmetric_part(&projected).components.len();
            if let Some(diagonal) = projected.components.get(&(mask, vector)) {
                operator_trace += diagonal.real.clone();
                assert_eq!(diagonal.imaginary, q(0));
            }
        }
    }

    let dynkin_label = if degree == 2 { "11000" } else { "10002" };
    let passed = idempotence_residual_entries == 0
        && trace_residual_entries == 0
        && total_antisymmetry_residual_entries == 0
        && operator_trace == q(hook_dimension as i64);
    HookProjectorCertificate {
        form_degree: degree,
        ambient_dimension,
        trace_dimension,
        exterior_dimension,
        hook_dimension,
        dynkin_label,
        idempotence_residual_entries,
        trace_residual_entries,
        total_antisymmetry_residual_entries,
        operator_trace: operator_trace.to_string(),
        passed,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceFixedCurvatureReport {
    pub schema_version: &'static str,
    pub source_locators: Vec<&'static str>,
    pub source_hashes: Vec<&'static str>,
    pub p320_certificate: ProjectorCertificate,
    pub x2_certificate: HookProjectorCertificate,
    pub x5_certificate: HookProjectorCertificate,
    pub transcribed_source_formulas: &'static [LinearizedSourceFormula],
    pub equations_24_to_29_transcribed: bool,
    pub equations_24_to_29_transcribed_as_operator: bool,
    pub equations_39_to_40_hook_elimination_implemented: bool,
    pub equations_39_to_40_hook_quotient_projection_implemented: bool,
    pub equation_44_coefficients_implemented: bool,
    pub equation_44_linear_w_j_coefficients_implemented: bool,
    pub p1_p3_compensators_eliminated: bool,
    pub p4_p5_compensators_eliminated: bool,
    pub compensator_sectors_annihilated_by_hook_projection: bool,
    pub p2_lorentz_compensator_treated_as_physical: bool,
    pub typed_target_stream_join_available: bool,
    pub typed_precontracted_target_stream_sink_available: bool,
    pub target_weight_to_cartesian_clifford_join_implemented: bool,
    pub full_h_hat_to_torsion_map_implemented: bool,
    pub full_f_a_g_p_test_ready: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

pub fn verify() -> SourceFixedCurvatureReport {
    let p320_certificate = certify_gamma_trace_projector();
    let x2_certificate = certify_hook_projector(2);
    let x5_certificate = certify_hook_projector(5);
    let passed = p320_certificate.passed && x2_certificate.passed && x5_certificate.passed;
    SourceFixedCurvatureReport {
        schema_version: "adynkra-11d-source-fixed-curvature-scaffold-v1",
        source_locators: vec![
            "arXiv:2007.05097 Eqs. (2.2)-(2.3), PDF p. 7: P_320 and K_tr",
            "hep-th/0101037 Eqs. (24)-(29), PDF pp. 7-8: linearized frames and anholonomy",
            "hep-th/0101037 Eqs. (39)-(40), PDF p. 11: X_[2], X_[5], and conventional hook constraints",
            "hep-th/0101037 Eq. (44), PDF p. 14: W, X_[2], X_[5], and J definitions",
        ],
        source_hashes: vec![
            HEP_TH_0101037_PDF_SHA256,
            HEP_TH_0101037_SOURCE_SHA256,
            ARXIV_2007_05097_PDF_SHA256,
        ],
        p320_certificate,
        x2_certificate,
        x5_certificate,
        transcribed_source_formulas: LINEARIZED_SOURCE_FORMULAS,
        equations_24_to_29_transcribed: true,
        equations_24_to_29_transcribed_as_operator: false,
        equations_39_to_40_hook_elimination_implemented: false,
        equations_39_to_40_hook_quotient_projection_implemented: true,
        equation_44_coefficients_implemented: false,
        equation_44_linear_w_j_coefficients_implemented: true,
        p1_p3_compensators_eliminated: false,
        p4_p5_compensators_eliminated: false,
        compensator_sectors_annihilated_by_hook_projection: true,
        p2_lorentz_compensator_treated_as_physical: false,
        typed_target_stream_join_available: false,
        typed_precontracted_target_stream_sink_available: true,
        target_weight_to_cartesian_clifford_join_implemented: false,
        full_h_hat_to_torsion_map_implemented: false,
        full_f_a_g_p_test_ready: false,
        passed,
        boundary: "The exact algebraic P_320 and K_tr certificates, X_[2] and X_[5] hook quotient projectors, and the linearized W and J coefficients are implemented. The hook projection annihilates the compensator representation sectors but does not explicitly solve the p=1,3,4,5 compensator fields. Eqs. (24)-(29) are transcribed but not implemented as a differential operator. The spin-connection solve, covariant-spinor/charge-conjugation alignment, B5-weight to Lorentz Cartesian join, and momentum-aware superspace derivative join remain open. This is not a complete F: H_hat -> curvature and does not establish F A G_p = 0.",
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
    fn exact_p320_certificate_closes() {
        let certificate = certify_gamma_trace_projector();
        assert!(certificate.passed);
        assert_eq!(certificate.projector_rank, 320);
        assert_eq!(certificate.gamma_trace_injection_rank, 32);
        assert!(certificate.kernel_equals_trace_image);
    }

    #[test]
    fn exact_x2_hook_certificate_is_429() {
        let certificate = certify_hook_projector(2);
        assert!(certificate.passed);
        assert_eq!(certificate.ambient_dimension, 605);
        assert_eq!(certificate.trace_dimension, 11);
        assert_eq!(certificate.exterior_dimension, 165);
        assert_eq!(certificate.hook_dimension, 429);
        assert_eq!(certificate.operator_trace, "429");
    }

    #[test]
    fn exact_x5_hook_certificate_is_4290() {
        let certificate = certify_hook_projector(5);
        assert!(certificate.passed);
        assert_eq!(certificate.ambient_dimension, 5_082);
        assert_eq!(certificate.trace_dimension, 330);
        assert_eq!(certificate.exterior_dimension, 462);
        assert_eq!(certificate.hook_dimension, 4_290);
        assert_eq!(certificate.operator_trace, "4290");
    }

    #[test]
    fn f_application_uses_printed_linearized_coefficients() {
        let mut gamma_two_d_h = FormVectorTensor::zero(2);
        gamma_two_d_h.add_component(0b11, 0, ExactGaussian::from_real(q(16)));
        let mut gamma_five_d_h = FormVectorTensor::zero(5);
        gamma_five_d_h.add_component(0b1_1111, 0, ExactGaussian::from_real(q(16)));
        let expected_x2 = irreducible_form_vector_hook(&gamma_two_d_h).scaled(&qr(1, 16));
        let expected_x5 = irreducible_form_vector_hook(&gamma_five_d_h)
            .times_i()
            .scaled(&qr(1, 16));
        let trace = vec![ExactGaussian::from_real(q(33)); SPINOR_DIMENSION];
        let mut w_torsion = BTreeMap::new();
        w_torsion.insert(0b1111, ExactGaussian::from_real(q(32)));
        let mut w_derivative = BTreeMap::new();
        w_derivative.insert(0b1111, ExactGaussian::from_real(q(96)));
        let output = apply_linearized_source_fixed_f(&LinearizedCurvatureContractions {
            gamma_two_d_h,
            gamma_five_d_h,
            torsion_vector_trace: trace,
            w_torsion_contraction: w_torsion,
            w_trace_derivative_contraction: w_derivative,
        });
        assert_eq!(output.j_spinor[0], ExactGaussian::from_real(q(4)));
        assert_eq!(output.x_two_hook_11000, expected_x2);
        assert_eq!(output.x_five_hook_10002, expected_x5);
        assert!(!output.x_two_hook_11000.components.is_empty());
        assert!(!output.x_five_hook_10002.components.is_empty());
        assert_ne!(
            output.x_two_hook_11000,
            output.x_two_hook_11000.scaled(&q(2)),
            "mutating the printed 1/16 coefficient must be detected"
        );
        assert_eq!(
            output.w_four_form[&0b1111],
            ExactGaussian {
                real: q(-1),
                imaginary: q(1),
            }
        );
        assert!(mixed_trace(&output.x_two_hook_11000).components.is_empty());
        assert!(
            total_antisymmetric_part(&output.x_two_hook_11000)
                .components
                .is_empty()
        );
        assert!(mixed_trace(&output.x_five_hook_10002).components.is_empty());
        assert!(
            total_antisymmetric_part(&output.x_five_hook_10002)
                .components
                .is_empty()
        );
    }

    #[test]
    fn report_is_explicitly_incomplete_at_the_differential_join() {
        let report = verify();
        assert!(report.passed);
        assert!(!report.equations_39_to_40_hook_elimination_implemented);
        assert!(report.equations_39_to_40_hook_quotient_projection_implemented);
        assert!(!report.equation_44_coefficients_implemented);
        assert!(report.equation_44_linear_w_j_coefficients_implemented);
        assert!(report.compensator_sectors_annihilated_by_hook_projection);
        assert!(!report.typed_target_stream_join_available);
        assert!(report.typed_precontracted_target_stream_sink_available);
        assert!(report.equations_24_to_29_transcribed);
        assert!(!report.equations_24_to_29_transcribed_as_operator);
        assert!(!report.target_weight_to_cartesian_clifford_join_implemented);
        assert!(!report.full_h_hat_to_torsion_map_implemented);
        assert!(!report.full_f_a_g_p_test_ready);
    }

    #[test]
    fn typed_precontracted_sink_retains_target_stream_provenance() {
        let entry =
            crate::eleven_dimensional_level16_couplings::TargetResolvedGaugeCompositionEntry {
                target_basis_ordinal: 0,
                target_vector_weight_index: 0,
                target_spinor_weight_index: 0,
                parameter_component_index: 7,
                momentum_vector_weight_index: None,
                exterior_mask: 1,
                real: q(2),
                imaginary: q(3),
            };
        let mut sink = TargetResolvedContractionAccumulator::new();
        sink.add_gamma_contracted_x_term(&entry, 2, 0b11, 0, ExactGaussian::from_real(q(5)));
        assert_eq!(sink.consumed_target_entries, 1);
        assert_eq!(
            sink.contractions.gamma_two_d_h.components[&(0b11, 0)],
            ExactGaussian {
                real: q(10),
                imaginary: q(15),
            }
        );
    }

    #[test]
    #[ignore = "writes the checked-in source-fixed curvature artifacts"]
    fn write_checked_in_artifacts() {
        write_artifacts(
            Path::new("data/eleven_dimensional_source_fixed_curvature.json"),
            Path::new("results/eleven_dimensional_source_fixed_curvature_validation.json"),
        )
        .unwrap();
    }
}
