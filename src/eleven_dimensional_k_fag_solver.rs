//! Generic-polynomial coefficient solver and fail-closed F A G_p harness.
//!
//! The recorded twelve leading and forty-four first-momentum operators are a
//! bounded ansatz, not a generic-momentum proof.  This module supplies exact
//! polynomial bookkeeping over all eleven formal momentum variables, an exact
//! Gaussian-rational coefficient solver, and a channel-separated curvature
//! composition interface.  A future complete physical-curvature API can be
//! plugged into the trait below without changing the solver.
//!
//! No source currently fixes the physical `Psi -> H_hat` map.  The current
//! curvature module also reports that its full `H_hat -> (W,X,J)` operator is
//! incomplete.  Consequently this module keeps physical `F A G_p = 0` false.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::fs;

pub type BigRational = Ratio<BigInt>;

fn q(value: i64) -> BigRational {
    Ratio::from_integer(BigInt::from(value))
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

    pub fn one() -> Self {
        Self {
            real: q(1),
            imaginary: q(0),
        }
    }

    pub fn from_integer(value: i64) -> Self {
        Self {
            real: q(value),
            imaginary: q(0),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imaginary.is_zero()
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

    fn negate(&self) -> Self {
        Self {
            real: -self.real.clone(),
            imaginary: -self.imaginary.clone(),
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

    fn divide(&self, other: &Self) -> Self {
        assert!(!other.is_zero());
        let denominator = other.real.clone() * other.real.clone()
            + other.imaginary.clone() * other.imaginary.clone();
        Self {
            real: (self.real.clone() * other.real.clone()
                + self.imaginary.clone() * other.imaginary.clone())
                / denominator.clone(),
            imaginary: (self.imaginary.clone() * other.real.clone()
                - self.real.clone() * other.imaginary.clone())
                / denominator,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct MomentumMonomial {
    pub exponents: [u16; 11],
}

impl MomentumMonomial {
    pub fn constant() -> Self {
        Self { exponents: [0; 11] }
    }

    pub fn variable(axis: usize) -> Self {
        assert!(axis < 11);
        let mut exponents = [0; 11];
        exponents[axis] = 1;
        Self { exponents }
    }

    pub fn total_degree(&self) -> usize {
        self.exponents.iter().map(|value| usize::from(*value)).sum()
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut exponents = [0; 11];
        for (axis, exponent) in exponents.iter_mut().enumerate() {
            *exponent = self.exponents[axis]
                .checked_add(other.exponents[axis])
                .expect("formal momentum exponent overflow");
        }
        Self { exponents }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KCoefficientSpec {
    pub ordinal: usize,
    pub label: String,
    pub operator_kind: String,
    pub spinor_derivative_order_before_gauge_map: usize,
    pub momentum_degree_before_gauge_map: usize,
    pub lower_symbol_status: String,
}

/// The currently recorded bounded basis.  It is intentionally marked as
/// incomplete beyond first momentum.
pub fn recorded_12_plus_44_k_ansatz() -> Vec<KCoefficientSpec> {
    (0..56)
        .map(|ordinal| {
            if ordinal < 12 {
                KCoefficientSpec {
                    ordinal,
                    label: format!("leading-D16-{ordinal:02}"),
                    operator_kind: "leading".to_string(),
                    spinor_derivative_order_before_gauge_map: 16,
                    momentum_degree_before_gauge_map: 0,
                    lower_symbol_status: "requires p D^14 and all subsequent normal-form descendants"
                        .to_string(),
                }
            } else {
                KCoefficientSpec {
                    ordinal,
                    label: format!("first-pD14-{:02}", ordinal - 12),
                    operator_kind: "first-momentum".to_string(),
                    spinor_derivative_order_before_gauge_map: 14,
                    momentum_degree_before_gauge_map: 1,
                    lower_symbol_status: "recorded first correction only; p^2 D^12 and lower symbols are not exhausted"
                        .to_string(),
                }
            }
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PolynomialCoverage {
    pub formal_momentum_variables: usize,
    pub momentum_basis: String,
    pub maximum_momentum_degree_built: Option<usize>,
    pub spinor_derivative_orders_built: Vec<usize>,
    pub all_lower_symbols_through_maximum_degree_built: bool,
    pub polynomial_degree_unbounded_or_proved_sufficient: bool,
    pub generic_polynomial_complete: bool,
}

impl PolynomialCoverage {
    pub fn bounded_recorded_ansatz() -> Self {
        Self {
            formal_momentum_variables: 11,
            momentum_basis: "B5 vector weight coordinates pending the explicit Cartesian join"
                .to_string(),
            maximum_momentum_degree_built: Some(1),
            spinor_derivative_orders_built: vec![16, 14],
            all_lower_symbols_through_maximum_degree_built: true,
            polynomial_degree_unbounded_or_proved_sufficient: false,
            generic_polynomial_complete: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PolynomialConstraintKey {
    pub gauge_form_degree: usize,
    pub parameter_component: usize,
    pub output_sector: String,
    pub output_coordinate: usize,
    pub spinor_derivative_mask: u32,
    pub spinor_derivative_order: usize,
    pub momentum_monomial: MomentumMonomial,
}

#[derive(Clone, Debug)]
struct ExactEquation {
    coefficients: Vec<ExactGaussian>,
    right_hand_side: ExactGaussian,
}

#[derive(Clone, Debug)]
pub struct ExactPolynomialSystem {
    variable_specs: Vec<KCoefficientSpec>,
    equations: BTreeMap<PolynomialConstraintKey, ExactEquation>,
    projective_homogeneous: bool,
}

impl ExactPolynomialSystem {
    pub fn new(variable_specs: Vec<KCoefficientSpec>, projective_homogeneous: bool) -> Self {
        assert!(
            variable_specs
                .iter()
                .enumerate()
                .all(|(ordinal, spec)| ordinal == spec.ordinal)
        );
        Self {
            variable_specs,
            equations: BTreeMap::new(),
            projective_homogeneous,
        }
    }

    pub fn variable_count(&self) -> usize {
        self.variable_specs.len()
    }

    pub fn equation_count(&self) -> usize {
        self.equations.len()
    }

    pub fn add_coefficient(
        &mut self,
        key: PolynomialConstraintKey,
        variable: usize,
        coefficient: ExactGaussian,
    ) {
        assert!(variable < self.variable_count());
        if coefficient.is_zero() {
            return;
        }
        let variable_count = self.variable_count();
        let equation = self.equations.entry(key).or_insert_with(|| ExactEquation {
            coefficients: vec![ExactGaussian::zero(); variable_count],
            right_hand_side: ExactGaussian::zero(),
        });
        equation.coefficients[variable].add_assign(&coefficient);
    }

    pub fn set_right_hand_side(&mut self, key: PolynomialConstraintKey, value: ExactGaussian) {
        let variable_count = self.variable_count();
        self.equations
            .entry(key)
            .or_insert_with(|| ExactEquation {
                coefficients: vec![ExactGaussian::zero(); variable_count],
                right_hand_side: ExactGaussian::zero(),
            })
            .right_hand_side = value;
    }

    pub fn solve(&self) -> ExactCoefficientSolution {
        solve_exact_system(
            &self
                .equations
                .values()
                .map(|equation| {
                    let mut row = equation.coefficients.clone();
                    row.push(equation.right_hand_side.clone());
                    row
                })
                .collect::<Vec<_>>(),
            self.variable_count(),
            self.projective_homogeneous,
        )
    }
}

/// Intersect independently constructed gauge-channel constraint systems.
/// Channel labels are part of each row key, so inequivalent parameter domains
/// remain separate and cannot cancel each other.
pub fn solve_joint_channel_systems(systems: &[ExactPolynomialSystem]) -> ExactCoefficientSolution {
    assert!(!systems.is_empty());
    let reference = &systems[0].variable_specs;
    assert!(
        systems
            .iter()
            .all(|system| &system.variable_specs == reference)
    );
    let mut joined = ExactPolynomialSystem::new(reference.clone(), true);
    for system in systems {
        for (key, equation) in &system.equations {
            assert!(
                !joined.equations.contains_key(key),
                "duplicate row key across channel systems"
            );
            joined.equations.insert(key.clone(), equation.clone());
        }
    }
    joined.solve()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoefficientSolveOutcome {
    UniqueRay,
    Family,
    Zero,
    NoSolution,
}

#[derive(Clone, Debug)]
pub struct ExactCoefficientSolution {
    pub outcome: CoefficientSolveOutcome,
    pub equation_count: usize,
    pub variable_count: usize,
    pub rank: usize,
    pub nullity: usize,
    pub inconsistent: bool,
    pub particular_solution: Vec<ExactGaussian>,
    pub homogeneous_kernel_basis: Vec<Vec<ExactGaussian>>,
}

fn solve_exact_system(
    rows: &[Vec<ExactGaussian>],
    variable_count: usize,
    projective_homogeneous: bool,
) -> ExactCoefficientSolution {
    assert!(rows.iter().all(|row| row.len() == variable_count + 1));
    let mut reduced = rows.to_vec();
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;
    for column in 0..variable_count {
        let Some(found) = (pivot_row..reduced.len()).find(|&row| !reduced[row][column].is_zero())
        else {
            continue;
        };
        reduced.swap(pivot_row, found);
        let pivot = reduced[pivot_row][column].clone();
        for entry in &mut reduced[pivot_row] {
            *entry = entry.divide(&pivot);
        }
        let normalized = reduced[pivot_row].clone();
        for row in 0..reduced.len() {
            if row == pivot_row || reduced[row][column].is_zero() {
                continue;
            }
            let factor = reduced[row][column].clone();
            for index in column..=variable_count {
                reduced[row][index] =
                    reduced[row][index].subtract(&factor.multiply(&normalized[index]));
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == reduced.len() {
            break;
        }
    }
    let inconsistent = reduced.iter().any(|row| {
        row[..variable_count].iter().all(ExactGaussian::is_zero) && !row[variable_count].is_zero()
    });
    let free_columns = (0..variable_count)
        .filter(|column| !pivot_columns.contains(column))
        .collect::<Vec<_>>();
    let mut particular = vec![ExactGaussian::zero(); variable_count];
    if !inconsistent {
        for (row, &pivot) in pivot_columns.iter().enumerate() {
            particular[pivot] = reduced[row][variable_count].clone();
        }
    }
    let kernel = if inconsistent {
        Vec::new()
    } else {
        free_columns
            .iter()
            .map(|&free| {
                let mut vector = vec![ExactGaussian::zero(); variable_count];
                vector[free] = ExactGaussian::one();
                for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
                    vector[pivot] = reduced[row][free].negate();
                }
                vector
            })
            .collect::<Vec<_>>()
    };
    let nullity = if inconsistent { 0 } else { kernel.len() };
    let particular_is_zero = particular.iter().all(ExactGaussian::is_zero);
    let outcome = if inconsistent {
        CoefficientSolveOutcome::NoSolution
    } else if nullity == 0 && particular_is_zero {
        CoefficientSolveOutcome::Zero
    } else if (projective_homogeneous && nullity == 1)
        || (!projective_homogeneous && nullity == 0 && !particular_is_zero)
    {
        CoefficientSolveOutcome::UniqueRay
    } else {
        CoefficientSolveOutcome::Family
    };
    ExactCoefficientSolution {
        outcome,
        equation_count: rows.len(),
        variable_count,
        rank: pivot_columns.len(),
        nullity,
        inconsistent,
        particular_solution: particular,
        homogeneous_kernel_basis: kernel,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TargetVariationKey {
    pub parameter_component: usize,
    pub target_coordinate: usize,
    pub target_vector_weight_index: Option<usize>,
    pub target_spinor_weight_index: Option<usize>,
    pub spinor_derivative_mask: u32,
    pub spinor_derivative_order: usize,
    pub momentum_monomial: MomentumMonomial,
}

#[derive(Clone, Debug)]
pub struct PolynomialTargetOperatorFamily {
    pub coefficient_specs: Vec<KCoefficientSpec>,
    pub coverage: PolynomialCoverage,
    pub source_selected_physical_k: bool,
    pub parameter_components_total: usize,
    pub parameter_components_evaluated: Vec<usize>,
    pub terms: BTreeMap<(usize, TargetVariationKey), ExactGaussian>,
}

impl PolynomialTargetOperatorFamily {
    /// Add one exact raw target-stream record without discarding the B5
    /// vector and spinor coordinates required by physical curvature maps.
    #[allow(clippy::too_many_arguments)]
    pub fn add_raw_target_resolved_stream_term(
        &mut self,
        coefficient_ordinal: usize,
        parameter_component: usize,
        target_basis_ordinal: usize,
        target_vector_weight_index: usize,
        target_spinor_weight_index: usize,
        momentum_axis: Option<usize>,
        exterior_mask: u32,
        real: BigRational,
        imaginary: BigRational,
    ) {
        let momentum_monomial = momentum_axis
            .map(MomentumMonomial::variable)
            .unwrap_or_else(MomentumMonomial::constant);
        self.add_term(
            coefficient_ordinal,
            TargetVariationKey {
                parameter_component,
                target_coordinate: target_basis_ordinal,
                target_vector_weight_index: Some(target_vector_weight_index),
                target_spinor_weight_index: Some(target_spinor_weight_index),
                spinor_derivative_mask: exterior_mask,
                spinor_derivative_order: exterior_mask.count_ones() as usize,
                momentum_monomial,
            },
            ExactGaussian { real, imaginary },
        );
    }

    /// Add one exact record from the target-resolved stream contract.
    ///
    /// `momentum_axis=None` represents a leading `D^17 Lambda` record.
    /// `Some(axis)` represents one formal `p_axis D^15 Lambda` record.  Terms
    /// at higher momentum degree use [`add_term`] with an explicit monomial.
    pub fn add_target_resolved_stream_term(
        &mut self,
        coefficient_ordinal: usize,
        parameter_component: usize,
        target_basis_ordinal: usize,
        momentum_axis: Option<usize>,
        exterior_mask: u32,
        real: BigRational,
        imaginary: BigRational,
    ) {
        let momentum_monomial = momentum_axis
            .map(MomentumMonomial::variable)
            .unwrap_or_else(MomentumMonomial::constant);
        self.add_term(
            coefficient_ordinal,
            TargetVariationKey {
                parameter_component,
                target_coordinate: target_basis_ordinal,
                target_vector_weight_index: None,
                target_spinor_weight_index: None,
                spinor_derivative_mask: exterior_mask,
                spinor_derivative_order: exterior_mask.count_ones() as usize,
                momentum_monomial,
            },
            ExactGaussian { real, imaginary },
        );
    }

    pub fn add_term(
        &mut self,
        coefficient_ordinal: usize,
        key: TargetVariationKey,
        value: ExactGaussian,
    ) {
        assert!(coefficient_ordinal < self.coefficient_specs.len());
        if value.is_zero() {
            return;
        }
        let entry = self
            .terms
            .entry((coefficient_ordinal, key))
            .or_insert_with(ExactGaussian::zero);
        entry.add_assign(&value);
    }

    pub fn parameter_projection_complete(&self) -> bool {
        self.parameter_components_evaluated.len() == self.parameter_components_total
            && self
                .parameter_components_evaluated
                .iter()
                .copied()
                .eq(0..self.parameter_components_total)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CurvatureVariationKey {
    pub parameter_component: usize,
    pub output_sector: String,
    pub output_coordinate: usize,
    pub spinor_derivative_mask: u32,
    pub spinor_derivative_order: usize,
    pub momentum_monomial: MomentumMonomial,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalCurvatureApiDescriptor {
    pub schema_version: String,
    pub provenance_sha256: Vec<String>,
    pub accepted_target_basis: String,
    pub target_basis_join_complete: bool,
    pub output_is_conventional_quotient_coordinates: bool,
    pub output_quotient_complete: bool,
    pub derivative_normal_form_complete: bool,
    pub generic_polynomial_action_complete: bool,
    pub complete_physical_f: bool,
}

pub trait PhysicalCurvaturePolynomialApi {
    fn descriptor(&self) -> PhysicalCurvatureApiDescriptor;

    fn apply_term(
        &self,
        input: &TargetVariationKey,
        coefficient: &ExactGaussian,
    ) -> Result<Vec<(CurvatureVariationKey, ExactGaussian)>, String>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChannelFagVerdict {
    pub gauge_form_degree: usize,
    pub parameter_components_total: usize,
    pub parameter_components_evaluated: usize,
    pub parameter_projection_complete: bool,
    pub target_terms_consumed: usize,
    pub curvature_terms_emitted: usize,
    pub exact_equations: usize,
    pub coefficient_outcome: Option<CoefficientSolveOutcome>,
    pub coefficient_rank: Option<usize>,
    pub coefficient_nullity: Option<usize>,
    pub f_api_error: Option<String>,
    pub bounded_ansatz_only: bool,
    pub quotient_complete: bool,
    pub target_basis_join_complete: bool,
    pub curvature_provenance_complete: bool,
    pub generic_polynomial_complete: bool,
    pub complete_physical_f: bool,
    pub source_selected_physical_k: bool,
    pub nonzero_candidate_exists: bool,
    pub channel_composition_exhaustive: bool,
    pub physical_f_a_g_p_established: bool,
    pub boundary: String,
}

pub fn build_channel_fag_system<A: PhysicalCurvaturePolynomialApi>(
    gauge_form_degree: usize,
    family: &PolynomialTargetOperatorFamily,
    curvature: &A,
) -> (ExactPolynomialSystem, ChannelFagVerdict) {
    assert!(gauge_form_degree < 6);
    let descriptor = curvature.descriptor();
    let mut system = ExactPolynomialSystem::new(family.coefficient_specs.clone(), true);
    let mut emitted = 0;
    let mut error = None;
    for ((coefficient_ordinal, target_key), coefficient) in &family.terms {
        match curvature.apply_term(target_key, coefficient) {
            Ok(terms) => {
                emitted += terms.len();
                for (key, value) in terms {
                    system.add_coefficient(
                        PolynomialConstraintKey {
                            gauge_form_degree,
                            parameter_component: key.parameter_component,
                            output_sector: key.output_sector,
                            output_coordinate: key.output_coordinate,
                            spinor_derivative_mask: key.spinor_derivative_mask,
                            spinor_derivative_order: key.spinor_derivative_order,
                            momentum_monomial: key.momentum_monomial,
                        },
                        *coefficient_ordinal,
                        value,
                    );
                }
            }
            Err(message) => {
                error = Some(message);
                break;
            }
        }
    }
    let solution = error.is_none().then(|| system.solve());
    let nonzero_candidate_exists = solution.as_ref().is_some_and(|result| {
        matches!(
            result.outcome,
            CoefficientSolveOutcome::UniqueRay | CoefficientSolveOutcome::Family
        )
    });
    let curvature_provenance_complete = !descriptor.provenance_sha256.is_empty()
        && descriptor
            .provenance_sha256
            .iter()
            .all(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let parameter_projection_complete = family.parameter_projection_complete();
    let exhaustive = error.is_none()
        && parameter_projection_complete
        && family.coverage.generic_polynomial_complete
        && descriptor.generic_polynomial_action_complete
        && descriptor.derivative_normal_form_complete
        && descriptor.target_basis_join_complete
        && curvature_provenance_complete
        && descriptor.output_quotient_complete
        && descriptor.complete_physical_f;
    let physical_established = exhaustive
        && family.source_selected_physical_k
        && solution
            .as_ref()
            .is_some_and(|result| result.outcome == CoefficientSolveOutcome::UniqueRay);
    let verdict = ChannelFagVerdict {
        gauge_form_degree,
        parameter_components_total: family.parameter_components_total,
        parameter_components_evaluated: family.parameter_components_evaluated.len(),
        parameter_projection_complete,
        target_terms_consumed: family.terms.len(),
        curvature_terms_emitted: emitted,
        exact_equations: system.equation_count(),
        coefficient_outcome: solution.as_ref().map(|result| result.outcome),
        coefficient_rank: solution.as_ref().map(|result| result.rank),
        coefficient_nullity: solution.as_ref().map(|result| result.nullity),
        f_api_error: error,
        bounded_ansatz_only: !family.coverage.generic_polynomial_complete,
        quotient_complete: descriptor.output_quotient_complete,
        target_basis_join_complete: descriptor.target_basis_join_complete,
        curvature_provenance_complete,
        generic_polynomial_complete: family.coverage.generic_polynomial_complete
            && descriptor.generic_polynomial_action_complete,
        complete_physical_f: descriptor.complete_physical_f,
        source_selected_physical_k: family.source_selected_physical_k,
        nonzero_candidate_exists,
        channel_composition_exhaustive: exhaustive,
        physical_f_a_g_p_established: physical_established,
        boundary: "A solver outcome classifies the supplied coefficient ansatz only. Physical F A G_p remains false until the source-selected K, complete F, all parameter components, every lower symbol, all eleven momentum variables, and the curvature quotient are certified.".to_string(),
    };
    (system, verdict)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HistoricalFirstMomentumNegativeControl {
    pub gauge_form_degree: usize,
    pub local_locator: String,
    pub artifact_sha256: String,
    pub expected_sha256: String,
    pub schema_version: String,
    pub artifact_passed: bool,
    pub parameter_components: usize,
    pub evaluated_parameter_components: usize,
    pub parameter_projection_complete: bool,
    pub exact_functional_rank: usize,
    pub exact_functional_nullity: usize,
    pub leading_projection_rank: usize,
    pub excludes_nonzero_extension_of_recorded_leading_kernel: bool,
    pub functional_kernel_residuals_exactly_zero: bool,
    pub local_provenance_hash_verified: bool,
    pub accepted_as_generic_fag_proof: bool,
}

const FINAL_PHYSICAL_FX_ARTIFACT_SHA256: &str =
    "5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9";
const FINAL_PHYSICAL_FX_INPUT_SHA256: &str =
    "c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f";
const CURRENT_PHYSICAL_CURVATURE_ENVELOPE_SHA256: &str =
    "3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13";
const FINAL_PHYSICAL_FX_SCHEMA: &str = "adynkra-11d-first-momentum-partial-fx-functional-v3";
const FINAL_PHYSICAL_FX_PROMOTION_SHA256: &str =
    "98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFxCheckpointPromotionProvenance {
    pub local_locator: String,
    pub artifact_sha256: String,
    pub expected_sha256: String,
    pub schema_version: String,
    pub promotion_id: String,
    pub checkpoint_hashes: usize,
    pub verified_existing: usize,
    pub copied_missing: usize,
    pub replaced_partial: usize,
    pub passed: bool,
    pub strict_contract_validated: bool,
    pub accepted_as_full_physical_fag_proof: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFxInputSnapshotProvenance {
    pub local_locator: String,
    pub artifact_sha256: String,
    pub expected_sha256: String,
    pub schema_version: String,
    pub complete_physical_f: bool,
    pub physical_fag_ready: bool,
    pub covariant_off_shell_closure_established: bool,
    pub strict_contract_validated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFxBoundedNegativeControl {
    pub local_locator: String,
    pub artifact_sha256: String,
    pub expected_sha256: String,
    pub schema_version: String,
    pub fx_report_curvature_input_sha256: String,
    pub expected_fx_report_curvature_input_sha256: String,
    pub fx_input_snapshot: PhysicalFxInputSnapshotProvenance,
    pub gauge_form_degrees: Vec<usize>,
    pub channel_count: usize,
    pub operator_columns_per_channel: usize,
    pub coefficient_variables: usize,
    pub leading_kernel_variables: usize,
    pub first_momentum_correction_variables: usize,
    pub parameter_components_selected_per_channel: Vec<Vec<usize>>,
    pub target_basis_ordinals_selected_per_channel: Vec<Vec<usize>>,
    pub global_joint_rank: usize,
    pub global_joint_nullity: usize,
    pub global_rank_exact_by_dimension_saturation: bool,
    pub surviving_leading_projection_rank_upper_bound: usize,
    pub all_six_bounded_fx_channels_checked: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub partial_fx_only: bool,
    pub artifact_full_f_a_g_p_established: bool,
    pub kills_recorded_five_plus_forty_four_coefficient_space: bool,
    pub checkpoint_promotion: PhysicalFxCheckpointPromotionProvenance,
    pub local_provenance_hash_verified: bool,
    pub strict_contract_validated: bool,
    pub accepted_as_generic_k_proof: bool,
    pub accepted_as_full_physical_fag_proof: bool,
    pub boundary: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalFxChannelArtifact {
    gauge_form_degree: usize,
    parameter_components_total: usize,
    parameter_components_selected: Vec<usize>,
    target_basis_ordinals_selected: Vec<usize>,
    operator_columns_composed: usize,
    emitted_target_terms: u64,
    x2_functional_rank_lower_bound: usize,
    x2_functional_nullity_upper_bound: usize,
    x5_functional_rank_lower_bound: usize,
    x5_functional_nullity_upper_bound: usize,
    joint_functional_rank_lower_bound: usize,
    joint_functional_nullity_upper_bound: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalFxFunctionalArtifact {
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
    channel_reports: Vec<PhysicalFxChannelArtifact>,
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
struct PhysicalFxCheckpointPromotionArtifact {
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

fn parse_physical_fx_checkpoint_promotion(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<PhysicalFxCheckpointPromotionProvenance, String> {
    let artifact_sha256 = format!("{:x}", Sha256::digest(bytes));
    if artifact_sha256 != expected_sha256 {
        return Err(format!(
            "physical F_X checkpoint-promotion SHA-256 mismatch: got {artifact_sha256}, expected {expected_sha256}"
        ));
    }
    let artifact: PhysicalFxCheckpointPromotionArtifact = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse physical F_X checkpoint promotion: {error}"))?;
    let expected_keys = (0..6)
        .flat_map(|degree| {
            (0..56).map(move |operator| format!("form-{degree}/operator-{operator:02}"))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let actual_keys = artifact
        .candidate_sha256
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let hashes_valid = artifact.candidate_sha256.values().all(|hash| {
        hash.len() == 64
            && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            && hash.bytes().all(|byte| !byte.is_ascii_uppercase())
    });
    if artifact.schema_version != "adynkra-11d-fx-shared-promotion-report-v1"
        || !artifact.passed
        || artifact.candidate_sha256.len() != 336
        || actual_keys != expected_keys
        || !hashes_valid
        || artifact.verified_existing + artifact.copied_missing != 336
        || artifact.replaced_partial != 0
        || artifact.promotion_id.is_empty()
        || artifact.finished_utc.is_empty()
        || artifact.candidate_root.is_empty()
        || !artifact
            .production_root
            .ends_with("eleven_dimensional_first_momentum_fx_checkpoints")
    {
        return Err("physical F_X checkpoint-promotion contract changed".to_string());
    }
    Ok(PhysicalFxCheckpointPromotionProvenance {
        local_locator: "results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json"
            .to_string(),
        artifact_sha256,
        expected_sha256: expected_sha256.to_string(),
        schema_version: artifact.schema_version,
        promotion_id: artifact.promotion_id,
        checkpoint_hashes: artifact.candidate_sha256.len(),
        verified_existing: artifact.verified_existing,
        copied_missing: artifact.copied_missing,
        replaced_partial: artifact.replaced_partial,
        passed: artifact.passed,
        strict_contract_validated: true,
        accepted_as_full_physical_fag_proof: false,
    })
}

fn physical_fx_checkpoint_promotion() -> PhysicalFxCheckpointPromotionProvenance {
    parse_physical_fx_checkpoint_promotion(
        include_bytes!(
            "../results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json"
        ),
        FINAL_PHYSICAL_FX_PROMOTION_SHA256,
    )
    .expect("strictly validate final physical F_X checkpoint promotion")
}

fn parse_physical_fx_bounded_negative_control(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<PhysicalFxBoundedNegativeControl, String> {
    let artifact_sha256 = format!("{:x}", Sha256::digest(bytes));
    if artifact_sha256 != expected_sha256 {
        return Err(format!(
            "physical F_X artifact SHA-256 mismatch: got {artifact_sha256}, expected {expected_sha256}"
        ));
    }
    let artifact: PhysicalFxFunctionalArtifact = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse final physical F_X artifact: {error}"))?;
    if artifact.schema_version != FINAL_PHYSICAL_FX_SCHEMA {
        return Err(format!(
            "physical F_X schema mismatch: got {}",
            artifact.schema_version
        ));
    }
    if artifact.curvature_artifact_sha256 != FINAL_PHYSICAL_FX_INPUT_SHA256 {
        return Err("physical F_X curvature provenance mismatch".to_string());
    }
    if artifact.role
        != "exact deterministic mask-summed functional lower bound for all-six first-momentum partial F_X A G_p on a declared target/parameter slice"
        || artifact.coefficient_space
            != "five exact leading F_X-kernel coordinates plus 44 recorded first-momentum correction coordinates"
    {
        return Err("physical F_X role or coefficient-space contract changed".to_string());
    }
    if artifact.coefficient_variables != 49
        || artifact.leading_kernel_variables != 5
        || artifact.first_momentum_correction_variables != 44
        || artifact.leading_kernel_variables + artifact.first_momentum_correction_variables
            != artifact.coefficient_variables
    {
        return Err("physical F_X coefficient-space dimensions changed".to_string());
    }
    if artifact.deterministic_hash_seeds
        != [
            "243f6a8885a308d3",
            "13198a2e03707344",
            "a4093822299f31d0",
            "082efa98ec4e6c89",
        ]
        || artifact.buckets_per_seed != 16
        || artifact.bounded_channel_concurrency != 1
        || artifact.operator_checkpoints_per_channel != 56
        || !artifact.checkpoint_resume_enabled
    {
        return Err("physical F_X deterministic functional contract changed".to_string());
    }
    let expected_parameter_totals = [1_usize, 11, 55, 165, 330, 462];
    if artifact.channel_reports.len() != 6 {
        return Err("physical F_X artifact must contain exactly six channels".to_string());
    }
    for (degree, channel) in artifact.channel_reports.iter().enumerate() {
        if channel.gauge_form_degree != degree
            || channel.parameter_components_total != expected_parameter_totals[degree]
            || channel.parameter_components_selected != [0]
            || channel.target_basis_ordinals_selected != [319]
            || channel.operator_columns_composed != 56
            || channel.emitted_target_terms == 0
            || channel.x2_functional_rank_lower_bound + channel.x2_functional_nullity_upper_bound
                != 49
            || channel.x5_functional_rank_lower_bound + channel.x5_functional_nullity_upper_bound
                != 49
            || channel.joint_functional_rank_lower_bound
                + channel.joint_functional_nullity_upper_bound
                != 49
        {
            return Err(format!(
                "physical F_X channel {degree} failed its exact bounded-slice contract"
            ));
        }
    }
    let saturated_recorded_space = artifact.global_x2_rank_lower_bound == 49
        && artifact.global_x2_nullity_upper_bound == 0
        && artifact.global_x5_rank_lower_bound == 49
        && artifact.global_x5_nullity_upper_bound == 0
        && artifact.global_joint_rank_lower_bound == 49
        && artifact.global_joint_nullity_upper_bound == 0
        && artifact.global_joint_rank_exact_by_dimension_saturation
        && artifact.surviving_leading_projection_rank_upper_bound == 0;
    if !artifact.all_six_channels_composed_on_declared_slice
        || artifact.full_parameter_projection_complete
        || artifact.full_target_projection_complete
        || !artifact.mutation_detected
        || !artifact.partial_fx_only
        || artifact.full_f_a_g_p_established
        || !saturated_recorded_space
    {
        return Err("physical F_X coverage or exact negative-control verdict changed".to_string());
    }
    if artifact.boundary.is_empty() {
        return Err("physical F_X boundary statement is missing".to_string());
    }
    Ok(PhysicalFxBoundedNegativeControl {
        local_locator: "results/adynkra_11d_first_momentum_physical_fx_functional.json".to_string(),
        artifact_sha256,
        expected_sha256: expected_sha256.to_string(),
        schema_version: artifact.schema_version,
        fx_report_curvature_input_sha256: artifact.curvature_artifact_sha256,
        expected_fx_report_curvature_input_sha256: FINAL_PHYSICAL_FX_INPUT_SHA256.to_string(),
        fx_input_snapshot: physical_fx_input_snapshot(),
        gauge_form_degrees: artifact
            .channel_reports
            .iter()
            .map(|channel| channel.gauge_form_degree)
            .collect(),
        channel_count: artifact.channel_reports.len(),
        operator_columns_per_channel: artifact.operator_checkpoints_per_channel,
        coefficient_variables: artifact.coefficient_variables,
        leading_kernel_variables: artifact.leading_kernel_variables,
        first_momentum_correction_variables: artifact.first_momentum_correction_variables,
        parameter_components_selected_per_channel: artifact
            .channel_reports
            .iter()
            .map(|channel| channel.parameter_components_selected.clone())
            .collect(),
        target_basis_ordinals_selected_per_channel: artifact
            .channel_reports
            .iter()
            .map(|channel| channel.target_basis_ordinals_selected.clone())
            .collect(),
        global_joint_rank: artifact.global_joint_rank_lower_bound,
        global_joint_nullity: artifact.global_joint_nullity_upper_bound,
        global_rank_exact_by_dimension_saturation: artifact
            .global_joint_rank_exact_by_dimension_saturation,
        surviving_leading_projection_rank_upper_bound: artifact
            .surviving_leading_projection_rank_upper_bound,
        all_six_bounded_fx_channels_checked: artifact.all_six_channels_composed_on_declared_slice,
        full_parameter_projection_complete: artifact.full_parameter_projection_complete,
        full_target_projection_complete: artifact.full_target_projection_complete,
        partial_fx_only: artifact.partial_fx_only,
        artifact_full_f_a_g_p_established: artifact.full_f_a_g_p_established,
        kills_recorded_five_plus_forty_four_coefficient_space: saturated_recorded_space,
        checkpoint_promotion: physical_fx_checkpoint_promotion(),
        local_provenance_hash_verified: true,
        strict_contract_validated: true,
        accepted_as_generic_k_proof: false,
        accepted_as_full_physical_fag_proof: false,
        boundary: artifact.boundary,
    })
}

fn physical_fx_bounded_negative_control() -> PhysicalFxBoundedNegativeControl {
    parse_physical_fx_bounded_negative_control(
        include_bytes!("../results/adynkra_11d_first_momentum_physical_fx_functional.json"),
        FINAL_PHYSICAL_FX_ARTIFACT_SHA256,
    )
    .expect("strictly validate final physical F_X bounded negative control")
}

#[derive(Clone, Debug, Deserialize)]
struct HistoricalMergeArtifact {
    schema_version: String,
    passed: bool,
    gauge_form_degree: usize,
    parameter_components: usize,
    evaluated_parameter_components: Vec<usize>,
    parameter_projection_is_complete: bool,
    exact_functional_rank: usize,
    exact_functional_nullity: usize,
    functional_kernel_leading_projection_rank: usize,
    nonzero_leading_extension_excluded_by_functionals: bool,
    functional_kernel_residuals_exactly_zero: bool,
}

fn historical_first_momentum_controls() -> Vec<HistoricalFirstMomentumNegativeControl> {
    let expected = [
        (
            1,
            "f183def003a71cd08b7516ad5a666e589eff20629706bdda64bb5d0eb4e3b62c",
        ),
        (
            2,
            "9177fe087728bced2df21a984020a1d7d5c485a59e01f9ac1094673ccc32a7cd",
        ),
        (
            5,
            "281999a56b85ab59b7fa50a40c4b2f6afa645f4c2cb24fc6563d60c621b272c2",
        ),
    ];
    expected
        .into_iter()
        .map(|(degree, expected_sha256)| {
            let local_locator =
                format!("results/adynkra_11d_first_momentum_gauge_functional_p{degree}.json");
            let bytes = std::fs::read(&local_locator)
                .unwrap_or_else(|error| panic!("read {local_locator}: {error}"));
            let artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
            let artifact: HistoricalMergeArtifact = serde_json::from_slice(&bytes)
                .unwrap_or_else(|error| panic!("parse {local_locator}: {error}"));
            let complete_components = artifact.evaluated_parameter_components.len()
                == artifact.parameter_components
                && artifact
                    .evaluated_parameter_components
                    .iter()
                    .copied()
                    .eq(0..artifact.parameter_components);
            let verified = artifact_sha256 == expected_sha256
                && artifact.schema_version
                    == "adynkra-11d-first-momentum-gauge-functional-merge-v1"
                && artifact.passed
                && artifact.gauge_form_degree == degree
                && artifact.parameter_projection_is_complete
                && complete_components
                && artifact.exact_functional_rank == 42
                && artifact.exact_functional_nullity == 3
                && artifact.functional_kernel_leading_projection_rank == 0
                && artifact.nonzero_leading_extension_excluded_by_functionals
                && artifact.functional_kernel_residuals_exactly_zero;
            HistoricalFirstMomentumNegativeControl {
                gauge_form_degree: degree,
                local_locator,
                artifact_sha256,
                expected_sha256: expected_sha256.to_string(),
                schema_version: artifact.schema_version,
                artifact_passed: artifact.passed,
                parameter_components: artifact.parameter_components,
                evaluated_parameter_components: artifact.evaluated_parameter_components.len(),
                parameter_projection_complete: artifact.parameter_projection_is_complete
                    && complete_components,
                exact_functional_rank: artifact.exact_functional_rank,
                exact_functional_nullity: artifact.exact_functional_nullity,
                leading_projection_rank: artifact.functional_kernel_leading_projection_rank,
                excludes_nonzero_extension_of_recorded_leading_kernel: artifact
                    .nonzero_leading_extension_excluded_by_functionals,
                functional_kernel_residuals_exactly_zero: artifact
                    .functional_kernel_residuals_exactly_zero,
                local_provenance_hash_verified: verified,
                accepted_as_generic_fag_proof: false,
            }
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KAndFagHarnessReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub gauge_form_degrees: Vec<usize>,
    pub gauge_dynkin_labels: Vec<&'static str>,
    pub formal_momentum_variables: usize,
    pub recorded_leading_coefficients: usize,
    pub recorded_first_momentum_coefficients: usize,
    pub recorded_ansatz_coefficients: usize,
    pub recorded_ansatz_generic_polynomial_complete: bool,
    pub coefficient_outcomes_supported: Vec<CoefficientSolveOutcome>,
    pub unique_ray_control_passed: bool,
    pub family_control_passed: bool,
    pub zero_control_passed: bool,
    pub no_solution_control_passed: bool,
    pub monomial_multiplication_control_passed: bool,
    pub channel_separation_enforced: bool,
    pub quotient_aware_verdicts_implemented: bool,
    pub current_physical_curvature_schema: String,
    pub current_physical_curvature_envelope_sha256: String,
    pub current_physical_curvature_envelope_provenance_validated: bool,
    pub current_physical_curvature_complete_f: bool,
    pub current_physical_curvature_fag_ready: bool,
    pub current_physical_curvature_covariant_off_shell_closure_established: bool,
    pub historical_first_momentum_negative_controls: Vec<HistoricalFirstMomentumNegativeControl>,
    pub final_physical_fx_bounded_negative_control: PhysicalFxBoundedNegativeControl,
    pub generic_k_solved: bool,
    pub all_six_physical_fag_channels_checked: bool,
    pub physical_fag_established: bool,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Deserialize)]
struct CurrentPhysicalCurvatureStatus {
    schema_version: String,
    complete_f_from_h_hat_implemented: bool,
    full_f_a_g_p_test_ready: bool,
    covariant_off_shell_closure_established: bool,
    #[serde(default)]
    first_momentum_fx_declared_slice_status: Option<CurrentPhysicalFxDeclaredSliceStatus>,
}

#[derive(Clone, Debug, Deserialize)]
struct CurrentPhysicalFxDeclaredSliceStatus {
    fx_input_snapshot_path: String,
    fx_input_snapshot_sha256_expected: String,
    fx_input_snapshot_sha256_observed: String,
    fx_input_snapshot_schema_version: String,
    fx_input_snapshot_validated: bool,
    artifact_path: String,
    artifact_sha256_expected: String,
    artifact_sha256_observed: String,
    artifact_schema_version: String,
    curvature_artifact_sha256: String,
    functional_report_fx_input_sha256_matches_snapshot: bool,
    promotion_manifest_path: String,
    promotion_manifest_sha256_expected: String,
    promotion_manifest_sha256_observed: String,
    promotion_manifest_schema_version: String,
    promotion_id: String,
    promoted_checkpoint_files: usize,
    promotion_verified_existing: usize,
    promotion_copied_missing: usize,
    promotion_replaced_partial: usize,
    report_invariants_validated: bool,
    checkpoint_promotion_validated: bool,
    qualified_zero_kernel_on_declared_slice: bool,
    coefficient_variables: usize,
    global_x2_rank_lower_bound: usize,
    global_x2_nullity_upper_bound: usize,
    global_x5_rank_lower_bound: usize,
    global_x5_nullity_upper_bound: usize,
    global_joint_rank_lower_bound: usize,
    global_joint_nullity_upper_bound: usize,
    all_six_channels_composed_on_declared_slice: bool,
    full_parameter_projection_complete: bool,
    full_target_projection_complete: bool,
    partial_fx_only: bool,
    full_f_a_g_p_established: bool,
    validation_error: Option<String>,
    current_physical_envelope_schema_version: String,
    current_physical_envelope_artifact_paths: Vec<String>,
    current_physical_envelope_self_hash_required: bool,
    boundary: String,
}

fn parse_physical_fx_input_snapshot(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<PhysicalFxInputSnapshotProvenance, String> {
    let artifact_sha256 = format!("{:x}", Sha256::digest(bytes));
    if artifact_sha256 != expected_sha256 {
        return Err(format!(
            "physical F_X input-snapshot SHA-256 mismatch: got {artifact_sha256}, expected {expected_sha256}"
        ));
    }
    let artifact: CurrentPhysicalCurvatureStatus = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse physical F_X input snapshot: {error}"))?;
    if artifact.schema_version != "adynkra-11d-physical-curvature-operator-v10"
        || artifact.complete_f_from_h_hat_implemented
        || artifact.full_f_a_g_p_test_ready
        || artifact.covariant_off_shell_closure_established
        || artifact.first_momentum_fx_declared_slice_status.is_some()
    {
        return Err("physical F_X input-snapshot contract changed".to_string());
    }
    Ok(PhysicalFxInputSnapshotProvenance {
        local_locator: "results/adynkra_11d_physical_curvature_fx_input_v10.json".to_string(),
        artifact_sha256,
        expected_sha256: expected_sha256.to_string(),
        schema_version: artifact.schema_version,
        complete_physical_f: artifact.complete_f_from_h_hat_implemented,
        physical_fag_ready: artifact.full_f_a_g_p_test_ready,
        covariant_off_shell_closure_established: artifact.covariant_off_shell_closure_established,
        strict_contract_validated: true,
    })
}

fn physical_fx_input_snapshot() -> PhysicalFxInputSnapshotProvenance {
    parse_physical_fx_input_snapshot(
        include_bytes!("../results/adynkra_11d_physical_curvature_fx_input_v10.json"),
        FINAL_PHYSICAL_FX_INPUT_SHA256,
    )
    .expect("strictly validate frozen physical F_X input snapshot")
}

fn parse_current_physical_curvature_envelope(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<CurrentPhysicalCurvatureStatus, String> {
    let artifact_sha256 = format!("{:x}", Sha256::digest(bytes));
    if artifact_sha256 != expected_sha256 {
        return Err(format!(
            "current physical-curvature envelope SHA-256 mismatch: got {artifact_sha256}, expected {expected_sha256}"
        ));
    }
    let artifact: CurrentPhysicalCurvatureStatus = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse current physical-curvature envelope: {error}"))?;
    if artifact.schema_version != "adynkra-11d-physical-curvature-operator-v10"
        || artifact.complete_f_from_h_hat_implemented
        || artifact.full_f_a_g_p_test_ready
        || artifact.covariant_off_shell_closure_established
    {
        return Err("current physical-curvature envelope contract changed".to_string());
    }
    let status = artifact
        .first_momentum_fx_declared_slice_status
        .as_ref()
        .ok_or_else(|| "current physical-curvature envelope lacks F_X provenance".to_string())?;
    let expected_paths = [
        "data/eleven_dimensional_physical_curvature.json".to_string(),
        "results/adynkra_11d_physical_curvature_validation.json".to_string(),
    ];
    if status.fx_input_snapshot_path != "results/adynkra_11d_physical_curvature_fx_input_v10.json"
        || status.fx_input_snapshot_sha256_expected != FINAL_PHYSICAL_FX_INPUT_SHA256
        || status.fx_input_snapshot_sha256_observed != FINAL_PHYSICAL_FX_INPUT_SHA256
        || status.fx_input_snapshot_schema_version != "adynkra-11d-physical-curvature-operator-v10"
        || !status.fx_input_snapshot_validated
        || status.artifact_path != "results/adynkra_11d_first_momentum_physical_fx_functional.json"
        || status.artifact_sha256_expected != FINAL_PHYSICAL_FX_ARTIFACT_SHA256
        || status.artifact_sha256_observed != FINAL_PHYSICAL_FX_ARTIFACT_SHA256
        || status.artifact_schema_version != FINAL_PHYSICAL_FX_SCHEMA
        || status.curvature_artifact_sha256 != FINAL_PHYSICAL_FX_INPUT_SHA256
        || !status.functional_report_fx_input_sha256_matches_snapshot
        || status.promotion_manifest_path
            != "results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json"
        || status.promotion_manifest_sha256_expected != FINAL_PHYSICAL_FX_PROMOTION_SHA256
        || status.promotion_manifest_sha256_observed != FINAL_PHYSICAL_FX_PROMOTION_SHA256
        || status.promotion_manifest_schema_version != "adynkra-11d-fx-shared-promotion-report-v1"
        || status.promotion_id != "20260817T205240Z-3355777"
        || status.promoted_checkpoint_files != 336
        || status.promotion_verified_existing != 164
        || status.promotion_copied_missing != 172
        || status.promotion_replaced_partial != 0
        || !status.report_invariants_validated
        || !status.checkpoint_promotion_validated
        || !status.qualified_zero_kernel_on_declared_slice
        || status.coefficient_variables != 49
        || status.global_x2_rank_lower_bound != 49
        || status.global_x2_nullity_upper_bound != 0
        || status.global_x5_rank_lower_bound != 49
        || status.global_x5_nullity_upper_bound != 0
        || status.global_joint_rank_lower_bound != 49
        || status.global_joint_nullity_upper_bound != 0
        || !status.all_six_channels_composed_on_declared_slice
        || status.full_parameter_projection_complete
        || status.full_target_projection_complete
        || !status.partial_fx_only
        || status.full_f_a_g_p_established
        || status.validation_error.is_some()
        || status.current_physical_envelope_schema_version
            != "adynkra-11d-physical-curvature-operator-v10"
        || status.current_physical_envelope_artifact_paths != expected_paths
        || status.current_physical_envelope_self_hash_required
        || status.boundary.is_empty()
    {
        return Err("current physical-curvature F_X provenance contract changed".to_string());
    }
    Ok(artifact)
}

fn test_key(row: usize) -> PolynomialConstraintKey {
    PolynomialConstraintKey {
        gauge_form_degree: 0,
        parameter_component: 0,
        output_sector: "control".to_string(),
        output_coordinate: row,
        spinor_derivative_mask: 0,
        spinor_derivative_order: 0,
        momentum_monomial: MomentumMonomial::constant(),
    }
}

fn control_specs(count: usize) -> Vec<KCoefficientSpec> {
    (0..count)
        .map(|ordinal| KCoefficientSpec {
            ordinal,
            label: format!("x{ordinal}"),
            operator_kind: "control".to_string(),
            spinor_derivative_order_before_gauge_map: 0,
            momentum_degree_before_gauge_map: 0,
            lower_symbol_status: "complete control".to_string(),
        })
        .collect()
}

pub fn verify() -> KAndFagHarnessReport {
    let mut unique = ExactPolynomialSystem::new(control_specs(2), true);
    unique.add_coefficient(test_key(0), 0, ExactGaussian::one());
    unique.add_coefficient(test_key(0), 1, ExactGaussian::from_integer(-1));
    let unique_result = unique.solve();

    let mut family = ExactPolynomialSystem::new(control_specs(3), true);
    family.add_coefficient(test_key(0), 0, ExactGaussian::one());
    let family_result = family.solve();

    let mut zero_system = ExactPolynomialSystem::new(control_specs(2), true);
    zero_system.add_coefficient(test_key(0), 0, ExactGaussian::one());
    zero_system.add_coefficient(test_key(1), 1, ExactGaussian::one());
    let zero_result = zero_system.solve();

    let mut inconsistent = ExactPolynomialSystem::new(control_specs(1), false);
    inconsistent.set_right_hand_side(test_key(0), ExactGaussian::one());
    let inconsistent_result = inconsistent.solve();

    let mut product_expected = [0_u16; 11];
    product_expected[2] = 1;
    product_expected[7] = 1;
    let monomial_control = MomentumMonomial::variable(2)
        .multiply(&MomentumMonomial::variable(7))
        .exponents
        == product_expected;

    let physical_bytes =
        include_bytes!("../results/adynkra_11d_physical_curvature_validation.json");
    let physical_sha256 = format!("{:x}", Sha256::digest(physical_bytes));
    let physical = parse_current_physical_curvature_envelope(
        physical_bytes,
        CURRENT_PHYSICAL_CURVATURE_ENVELOPE_SHA256,
    )
    .expect("strictly validate current physical-curvature envelope");
    let controls = historical_first_momentum_controls();
    let physical_fx_control = physical_fx_bounded_negative_control();
    let unique_ray_control_passed = unique_result.outcome == CoefficientSolveOutcome::UniqueRay
        && unique_result.rank == 1
        && unique_result.nullity == 1;
    let family_control_passed = family_result.outcome == CoefficientSolveOutcome::Family
        && family_result.rank == 1
        && family_result.nullity == 2;
    let zero_control_passed = zero_result.outcome == CoefficientSolveOutcome::Zero
        && zero_result.rank == 2
        && zero_result.nullity == 0;
    let no_solution_control_passed = inconsistent_result.outcome
        == CoefficientSolveOutcome::NoSolution
        && inconsistent_result.inconsistent;
    let passed = unique_ray_control_passed
        && family_control_passed
        && zero_control_passed
        && no_solution_control_passed
        && monomial_control
        && recorded_12_plus_44_k_ansatz().len() == 56
        && physical_sha256 == CURRENT_PHYSICAL_CURVATURE_ENVELOPE_SHA256
        && physical_fx_control.fx_report_curvature_input_sha256 == FINAL_PHYSICAL_FX_INPUT_SHA256
        && physical_fx_control.fx_input_snapshot.artifact_sha256 == FINAL_PHYSICAL_FX_INPUT_SHA256
        && physical_fx_control
            .fx_input_snapshot
            .strict_contract_validated
        && controls.iter().all(|control| {
            control.local_provenance_hash_verified && !control.accepted_as_generic_fag_proof
        })
        && physical_fx_control.local_provenance_hash_verified
        && physical_fx_control.strict_contract_validated
        && physical_fx_control.kills_recorded_five_plus_forty_four_coefficient_space
        && physical_fx_control.checkpoint_promotion.passed
        && physical_fx_control
            .checkpoint_promotion
            .strict_contract_validated
        && physical_fx_control.checkpoint_promotion.checkpoint_hashes == 336
        && physical_fx_control.checkpoint_promotion.verified_existing
            + physical_fx_control.checkpoint_promotion.copied_missing
            == 336
        && physical_fx_control.checkpoint_promotion.replaced_partial == 0
        && !physical_fx_control
            .checkpoint_promotion
            .accepted_as_full_physical_fag_proof
        && !physical_fx_control.accepted_as_generic_k_proof
        && !physical_fx_control.accepted_as_full_physical_fag_proof;

    KAndFagHarnessReport {
        schema_version: "adynkra-11d-k-fag-polynomial-harness-v1",
        role: "generic-polynomial K coefficient solver and exact channel-separated physical-curvature composition contract",
        gauge_form_degrees: (0..6).collect(),
        gauge_dynkin_labels: vec!["00000", "10000", "01000", "00100", "00010", "00002"],
        formal_momentum_variables: 11,
        recorded_leading_coefficients: 12,
        recorded_first_momentum_coefficients: 44,
        recorded_ansatz_coefficients: 56,
        recorded_ansatz_generic_polynomial_complete: false,
        coefficient_outcomes_supported: vec![
            CoefficientSolveOutcome::UniqueRay,
            CoefficientSolveOutcome::Family,
            CoefficientSolveOutcome::Zero,
            CoefficientSolveOutcome::NoSolution,
        ],
        unique_ray_control_passed,
        family_control_passed,
        zero_control_passed,
        no_solution_control_passed,
        monomial_multiplication_control_passed: monomial_control,
        channel_separation_enforced: true,
        quotient_aware_verdicts_implemented: true,
        current_physical_curvature_schema: physical.schema_version,
        current_physical_curvature_envelope_sha256: physical_sha256,
        current_physical_curvature_envelope_provenance_validated: true,
        current_physical_curvature_complete_f: physical.complete_f_from_h_hat_implemented,
        current_physical_curvature_fag_ready: physical.full_f_a_g_p_test_ready,
        current_physical_curvature_covariant_off_shell_closure_established: physical
            .covariant_off_shell_closure_established,
        historical_first_momentum_negative_controls: controls,
        final_physical_fx_bounded_negative_control: physical_fx_control,
        generic_k_solved: false,
        all_six_physical_fag_channels_checked: false,
        physical_fag_established: false,
        passed,
        result: "The exact all-six physical F_X bounded slice has rank 49 and nullity zero, so it kills the recorded five-leading-kernel-plus-forty-four-first-momentum coefficient space. Generic K and complete physical F A G_p remain open.",
        boundary: "The exact rank-49 result is a negative control for the declared one-parameter, one-target, partial-F_X slice only. It does not prove a generic-polynomial identity. No physical verdict is accepted without a source-selected K, all formal momentum monomials and lower symbols required by a proved degree bound, complete parameter projection in each of six inequivalent channels, a convention-fixed target-basis join, complete F including J and W, and its declared curvature quotient.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct QuotientMockCurvature;

    impl PhysicalCurvaturePolynomialApi for QuotientMockCurvature {
        fn descriptor(&self) -> PhysicalCurvatureApiDescriptor {
            PhysicalCurvatureApiDescriptor {
                schema_version: "mock-quotient-v1".to_string(),
                provenance_sha256: vec!["0".repeat(64)],
                accepted_target_basis: "control".to_string(),
                target_basis_join_complete: true,
                output_is_conventional_quotient_coordinates: true,
                output_quotient_complete: true,
                derivative_normal_form_complete: true,
                generic_polynomial_action_complete: false,
                complete_physical_f: false,
            }
        }

        fn apply_term(
            &self,
            input: &TargetVariationKey,
            coefficient: &ExactGaussian,
        ) -> Result<Vec<(CurvatureVariationKey, ExactGaussian)>, String> {
            Ok(vec![(
                CurvatureVariationKey {
                    parameter_component: input.parameter_component,
                    output_sector: "mock-X".to_string(),
                    output_coordinate: input.target_coordinate % 2,
                    spinor_derivative_mask: input.spinor_derivative_mask,
                    spinor_derivative_order: input.spinor_derivative_order,
                    momentum_monomial: input.momentum_monomial.clone(),
                },
                coefficient.clone(),
            )])
        }
    }

    #[test]
    fn coefficient_solver_distinguishes_all_four_required_outcomes() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert!(report.unique_ray_control_passed);
        assert!(report.family_control_passed);
        assert!(report.zero_control_passed);
        assert!(report.no_solution_control_passed);

        let specs = control_specs(2);
        let mut first = ExactPolynomialSystem::new(specs.clone(), true);
        first.add_coefficient(test_key(0), 0, ExactGaussian::one());
        first.add_coefficient(test_key(0), 1, ExactGaussian::from_integer(-1));
        let mut second_key = test_key(0);
        second_key.gauge_form_degree = 1;
        let mut second = ExactPolynomialSystem::new(specs, true);
        second.add_coefficient(second_key.clone(), 0, ExactGaussian::one());
        second.add_coefficient(second_key, 1, ExactGaussian::one());
        let joint = solve_joint_channel_systems(&[first, second]);
        assert_eq!(joint.outcome, CoefficientSolveOutcome::Zero);
        assert_eq!(joint.rank, 2);
    }

    #[test]
    fn all_eleven_formal_momenta_and_lower_symbol_limits_are_explicit() {
        let report = verify();
        assert_eq!(report.formal_momentum_variables, 11);
        assert_eq!(recorded_12_plus_44_k_ansatz().len(), 56);
        let coverage = PolynomialCoverage::bounded_recorded_ansatz();
        assert_eq!(coverage.maximum_momentum_degree_built, Some(1));
        assert_eq!(coverage.spinor_derivative_orders_built, vec![16, 14]);
        assert!(!coverage.generic_polynomial_complete);
        assert!(!coverage.polynomial_degree_unbounded_or_proved_sufficient);

        let mut family = PolynomialTargetOperatorFamily {
            coefficient_specs: control_specs(1),
            coverage,
            source_selected_physical_k: false,
            parameter_components_total: 1,
            parameter_components_evaluated: vec![0],
            terms: BTreeMap::new(),
        };
        for axis in 0..11 {
            family.add_target_resolved_stream_term(0, 0, axis, Some(axis), 0x7fff, q(1), q(0));
        }
        assert_eq!(family.terms.len(), 11);
        assert!(family.terms.keys().all(|(_, key)| {
            key.spinor_derivative_order == 15 && key.momentum_monomial.total_degree() == 1
        }));
    }

    #[test]
    fn channel_harness_consumes_exact_f_api_but_refuses_a_bounded_physical_claim() {
        let specs = control_specs(2);
        let mut family = PolynomialTargetOperatorFamily {
            coefficient_specs: specs,
            coverage: PolynomialCoverage::bounded_recorded_ansatz(),
            source_selected_physical_k: false,
            parameter_components_total: 2,
            parameter_components_evaluated: vec![0, 1],
            terms: BTreeMap::new(),
        };
        for parameter in 0..2 {
            family.add_term(
                0,
                TargetVariationKey {
                    parameter_component: parameter,
                    target_coordinate: 0,
                    target_vector_weight_index: None,
                    target_spinor_weight_index: None,
                    spinor_derivative_mask: 3,
                    spinor_derivative_order: 2,
                    momentum_monomial: MomentumMonomial::variable(0),
                },
                ExactGaussian::one(),
            );
            family.add_term(
                1,
                TargetVariationKey {
                    parameter_component: parameter,
                    target_coordinate: 0,
                    target_vector_weight_index: None,
                    target_spinor_weight_index: None,
                    spinor_derivative_mask: 3,
                    spinor_derivative_order: 2,
                    momentum_monomial: MomentumMonomial::variable(0),
                },
                ExactGaussian::from_integer(-1),
            );
        }
        let (_, verdict) = build_channel_fag_system(2, &family, &QuotientMockCurvature);
        assert!(verdict.parameter_projection_complete);
        assert_eq!(
            verdict.coefficient_outcome,
            Some(CoefficientSolveOutcome::UniqueRay)
        );
        assert!(verdict.nonzero_candidate_exists);
        assert!(verdict.quotient_complete);
        assert!(verdict.target_basis_join_complete);
        assert!(verdict.curvature_provenance_complete);
        assert!(!verdict.generic_polynomial_complete);
        assert!(!verdict.complete_physical_f);
        assert!(!verdict.channel_composition_exhaustive);
        assert!(!verdict.physical_f_a_g_p_established);
    }

    #[test]
    fn historical_screens_are_negative_controls_not_generic_fag_proofs() {
        let report = verify();
        assert_eq!(
            report
                .historical_first_momentum_negative_controls
                .iter()
                .map(|control| control.gauge_form_degree)
                .collect::<Vec<_>>(),
            vec![1, 2, 5]
        );
        assert!(
            report
                .historical_first_momentum_negative_controls
                .iter()
                .all(|control| control.parameter_projection_complete
                    && control.exact_functional_rank == 42
                    && control.exact_functional_nullity == 3
                    && control.leading_projection_rank == 0
                    && control.functional_kernel_residuals_exactly_zero
                    && control.artifact_sha256 == control.expected_sha256
                    && control.local_provenance_hash_verified
                    && !control.accepted_as_generic_fag_proof)
        );
        assert!(!report.physical_fag_established);
    }

    #[test]
    fn final_physical_fx_artifact_is_an_exact_bounded_negative_control() {
        let report = verify();
        let control = &report.final_physical_fx_bounded_negative_control;
        assert_eq!(control.artifact_sha256, FINAL_PHYSICAL_FX_ARTIFACT_SHA256);
        assert_eq!(control.expected_sha256, FINAL_PHYSICAL_FX_ARTIFACT_SHA256);
        assert_eq!(control.schema_version, FINAL_PHYSICAL_FX_SCHEMA);
        assert_eq!(
            control.fx_report_curvature_input_sha256,
            FINAL_PHYSICAL_FX_INPUT_SHA256
        );
        assert_eq!(
            control.expected_fx_report_curvature_input_sha256,
            FINAL_PHYSICAL_FX_INPUT_SHA256
        );
        assert_eq!(
            control.fx_input_snapshot.artifact_sha256,
            FINAL_PHYSICAL_FX_INPUT_SHA256
        );
        assert_eq!(
            control.fx_input_snapshot.expected_sha256,
            FINAL_PHYSICAL_FX_INPUT_SHA256
        );
        assert!(control.fx_input_snapshot.strict_contract_validated);
        assert_eq!(control.gauge_form_degrees, (0..6).collect::<Vec<_>>());
        assert_eq!(control.channel_count, 6);
        assert_eq!(control.operator_columns_per_channel, 56);
        assert_eq!(control.coefficient_variables, 49);
        assert_eq!(control.leading_kernel_variables, 5);
        assert_eq!(control.first_momentum_correction_variables, 44);
        assert!(
            control
                .parameter_components_selected_per_channel
                .iter()
                .all(|components| components == &[0])
        );
        assert!(
            control
                .target_basis_ordinals_selected_per_channel
                .iter()
                .all(|targets| targets == &[319])
        );
        assert_eq!(control.global_joint_rank, 49);
        assert_eq!(control.global_joint_nullity, 0);
        assert!(control.global_rank_exact_by_dimension_saturation);
        assert_eq!(control.surviving_leading_projection_rank_upper_bound, 0);
        assert!(control.all_six_bounded_fx_channels_checked);
        assert!(!control.full_parameter_projection_complete);
        assert!(!control.full_target_projection_complete);
        assert!(control.partial_fx_only);
        assert!(!control.artifact_full_f_a_g_p_established);
        assert!(control.kills_recorded_five_plus_forty_four_coefficient_space);
        assert_eq!(
            control.checkpoint_promotion.artifact_sha256,
            FINAL_PHYSICAL_FX_PROMOTION_SHA256
        );
        assert_eq!(control.checkpoint_promotion.checkpoint_hashes, 336);
        assert_eq!(control.checkpoint_promotion.verified_existing, 164);
        assert_eq!(control.checkpoint_promotion.copied_missing, 172);
        assert_eq!(control.checkpoint_promotion.replaced_partial, 0);
        assert!(control.checkpoint_promotion.passed);
        assert!(control.checkpoint_promotion.strict_contract_validated);
        assert!(
            !control
                .checkpoint_promotion
                .accepted_as_full_physical_fag_proof
        );
        assert!(control.local_provenance_hash_verified);
        assert!(control.strict_contract_validated);
        assert!(!control.accepted_as_generic_k_proof);
        assert!(!control.accepted_as_full_physical_fag_proof);
        assert!(!report.generic_k_solved);
        assert!(!report.all_six_physical_fag_channels_checked);
        assert!(!report.physical_fag_established);
        assert_eq!(
            report.current_physical_curvature_envelope_sha256,
            CURRENT_PHYSICAL_CURVATURE_ENVELOPE_SHA256
        );
        assert!(report.current_physical_curvature_envelope_provenance_validated);
        assert!(!report.current_physical_curvature_complete_f);
        assert!(!report.current_physical_curvature_fag_ready);
        assert!(!report.current_physical_curvature_covariant_off_shell_closure_established);
    }

    #[test]
    fn final_physical_fx_artifact_mutations_fail_closed() {
        let original =
            include_bytes!("../results/adynkra_11d_first_momentum_physical_fx_functional.json");
        let mut changed_rank: serde_json::Value = serde_json::from_slice(original).unwrap();
        changed_rank["global_joint_rank_lower_bound"] = serde_json::json!(48);
        let changed_rank_bytes = serde_json::to_vec(&changed_rank).unwrap();
        let changed_rank_sha = format!("{:x}", Sha256::digest(&changed_rank_bytes));
        assert!(
            parse_physical_fx_bounded_negative_control(&changed_rank_bytes, &changed_rank_sha)
                .unwrap_err()
                .contains("coverage or exact negative-control verdict")
        );

        let mut changed_coverage: serde_json::Value = serde_json::from_slice(original).unwrap();
        changed_coverage["full_parameter_projection_complete"] = serde_json::json!(true);
        let changed_coverage_bytes = serde_json::to_vec(&changed_coverage).unwrap();
        let changed_coverage_sha = format!("{:x}", Sha256::digest(&changed_coverage_bytes));
        assert!(
            parse_physical_fx_bounded_negative_control(
                &changed_coverage_bytes,
                &changed_coverage_sha,
            )
            .is_err()
        );

        let mut changed_target: serde_json::Value = serde_json::from_slice(original).unwrap();
        changed_target["channel_reports"][3]["target_basis_ordinals_selected"] =
            serde_json::json!([318]);
        let changed_target_bytes = serde_json::to_vec(&changed_target).unwrap();
        let changed_target_sha = format!("{:x}", Sha256::digest(&changed_target_bytes));
        assert!(
            parse_physical_fx_bounded_negative_control(&changed_target_bytes, &changed_target_sha)
                .unwrap_err()
                .contains("channel 3")
        );

        let mut unknown_field: serde_json::Value = serde_json::from_slice(original).unwrap();
        unknown_field["unreviewed_coverage_claim"] = serde_json::json!(true);
        let unknown_field_bytes = serde_json::to_vec(&unknown_field).unwrap();
        let unknown_field_sha = format!("{:x}", Sha256::digest(&unknown_field_bytes));
        assert!(
            parse_physical_fx_bounded_negative_control(&unknown_field_bytes, &unknown_field_sha)
                .unwrap_err()
                .contains("unknown field")
        );

        assert!(
            parse_physical_fx_bounded_negative_control(original, &"0".repeat(64))
                .unwrap_err()
                .contains("SHA-256 mismatch")
        );

        let promotion = include_bytes!(
            "../results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json"
        );
        let mut changed_promotion: serde_json::Value = serde_json::from_slice(promotion).unwrap();
        changed_promotion["copied_missing"] = serde_json::json!(171);
        let changed_promotion_bytes = serde_json::to_vec(&changed_promotion).unwrap();
        let changed_promotion_sha = format!("{:x}", Sha256::digest(&changed_promotion_bytes));
        assert!(
            parse_physical_fx_checkpoint_promotion(
                &changed_promotion_bytes,
                &changed_promotion_sha,
            )
            .unwrap_err()
            .contains("checkpoint-promotion contract changed")
        );
        assert!(
            parse_physical_fx_checkpoint_promotion(promotion, &"0".repeat(64))
                .unwrap_err()
                .contains("checkpoint-promotion SHA-256 mismatch")
        );

        let snapshot =
            include_bytes!("../results/adynkra_11d_physical_curvature_fx_input_v10.json");
        assert!(
            parse_physical_fx_input_snapshot(snapshot, &"0".repeat(64))
                .unwrap_err()
                .contains("input-snapshot SHA-256 mismatch")
        );
        let mut enriched_snapshot: serde_json::Value = serde_json::from_slice(snapshot).unwrap();
        let current_envelope: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../results/adynkra_11d_physical_curvature_validation.json"
        ))
        .unwrap();
        enriched_snapshot["first_momentum_fx_declared_slice_status"] =
            current_envelope["first_momentum_fx_declared_slice_status"].clone();
        let enriched_snapshot_bytes = serde_json::to_vec(&enriched_snapshot).unwrap();
        let enriched_snapshot_sha = format!("{:x}", Sha256::digest(&enriched_snapshot_bytes));
        assert!(
            parse_physical_fx_input_snapshot(&enriched_snapshot_bytes, &enriched_snapshot_sha)
                .unwrap_err()
                .contains("input-snapshot contract changed")
        );

        let envelope = include_bytes!("../results/adynkra_11d_physical_curvature_validation.json");
        assert!(
            parse_current_physical_curvature_envelope(envelope, &"0".repeat(64))
                .unwrap_err()
                .contains("envelope SHA-256 mismatch")
        );
        let mut changed_envelope: serde_json::Value = serde_json::from_slice(envelope).unwrap();
        changed_envelope["first_momentum_fx_declared_slice_status"]["fx_input_snapshot_sha256_observed"] =
            serde_json::json!("0".repeat(64));
        let changed_envelope_bytes = serde_json::to_vec(&changed_envelope).unwrap();
        let changed_envelope_sha = format!("{:x}", Sha256::digest(&changed_envelope_bytes));
        assert!(
            parse_current_physical_curvature_envelope(
                &changed_envelope_bytes,
                &changed_envelope_sha,
            )
            .unwrap_err()
            .contains("F_X provenance contract changed")
        );
    }

    #[test]
    #[ignore = "writes the committed K/FAG harness artifact"]
    fn write_artifact() {
        let report = verify();
        assert!(report.passed);
        let path = "results/adynkra_11d_k_fag_polynomial_harness.json";
        let temporary = format!("{path}.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        fs::rename(temporary, path).unwrap();
    }
}
