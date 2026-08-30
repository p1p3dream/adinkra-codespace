//! Proof-safe exact functional harness for the bounded second-momentum `F_X` screen.
//!
//! This module consumes recoupled component streams. It does not construct or
//! guess any of the 77 physical operator columns. Every functional key retains
//! the full eleven-variable degree-two source momentum monomial, and the
//! `p^2 D^13` wedge and `p^3 D^11` contraction branches remain disjoint.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_k_fag_solver::{
    ExactGaussian, ExactPolynomialSystem, KCoefficientSpec, MomentumMonomial,
    PolynomialConstraintKey,
};

pub const SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS: usize = 77;
pub const SECOND_MOMENTUM_FX_GAUGE_CHANNELS: usize = 6;
pub const SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS: [u64; 4] = [
    0x5d12_0f02_13aa_0001,
    0x5d12_0f02_13aa_0002,
    0x5d12_0f02_13aa_0003,
    0x5d12_0f02_13aa_0004,
];
pub const SECOND_MOMENTUM_FX_BUCKETS_PER_SEED: usize = 32;
pub const SECOND_MOMENTUM_FX_FUNCTIONAL_ROWS: usize =
    SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len() * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;
pub const SECOND_MOMENTUM_FX_CHECKPOINT_SCHEMA: &str =
    "adynkra-11d-second-momentum-partial-fx-checkpoint-v1";
pub const SECOND_MOMENTUM_FX_REPORT_SCHEMA: &str =
    "adynkra-11d-second-momentum-partial-fx-functional-v1";
pub const SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256: &str =
    "83698bf554699aa54c1f576366ca0945424675e9698a4203aadb787f232cff57";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct DegreeTwoMomentumMonomial {
    pub exponents: [u16; 11],
}

impl DegreeTwoMomentumMonomial {
    pub fn new(exponents: [u16; 11]) -> Result<Self, String> {
        let value = Self { exponents };
        if value.total_degree() != 2 {
            return Err("second-momentum source monomial must have total degree two".to_string());
        }
        Ok(value)
    }

    pub fn from_pair(left: usize, right: usize) -> Result<Self, String> {
        if left >= 11 || right >= 11 {
            return Err("second-momentum axis must lie in 0..11".to_string());
        }
        let mut exponents = [0_u16; 11];
        exponents[left] += 1;
        exponents[right] += 1;
        Self::new(exponents)
    }

    pub fn total_degree(self) -> usize {
        self.exponents.iter().map(|value| usize::from(*value)).sum()
    }

    pub(crate) fn canonical_pair(self) -> Result<[usize; 2], String> {
        if self.total_degree() != 2 {
            return Err("second-momentum monomial does not have degree two".to_string());
        }
        let mut axes = [0_usize; 2];
        let mut next = 0;
        for (axis, exponent) in self.exponents.into_iter().enumerate() {
            for _ in 0..exponent {
                if next >= axes.len() {
                    return Err("second-momentum monomial exceeds degree two".to_string());
                }
                axes[next] = axis;
                next += 1;
            }
        }
        if next != axes.len() {
            return Err("second-momentum monomial has incomplete degree".to_string());
        }
        Ok(axes)
    }

    fn as_solver_monomial(self) -> MomentumMonomial {
        MomentumMonomial {
            exponents: self.exponents,
        }
    }

    fn with_contraction_axis(self, axis: u8) -> Result<MomentumMonomial, String> {
        if usize::from(axis) >= 11 {
            return Err("contraction momentum axis must lie in 0..11".to_string());
        }
        Ok(self
            .as_solver_monomial()
            .multiply(&MomentumMonomial::variable(usize::from(axis))))
    }

    fn compact_label(self) -> String {
        self.exponents
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct SecondMomentumGaugeChannel(u8);

impl SecondMomentumGaugeChannel {
    pub fn new(form_degree: usize) -> Result<Self, String> {
        if form_degree >= SECOND_MOMENTUM_FX_GAUGE_CHANNELS {
            return Err("11D gauge form degree must lie in 0..=5".to_string());
        }
        Ok(Self(form_degree as u8))
    }

    pub fn form_degree(self) -> usize {
        usize::from(self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SecondMomentumGaugeBranch {
    P2D13Wedge,
    P3D11Contraction { momentum_axis: u8 },
}

impl SecondMomentumGaugeBranch {
    fn derivative_order(self) -> usize {
        match self {
            Self::P2D13Wedge => 13,
            Self::P3D11Contraction { .. } => 11,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::P2D13Wedge => "p2_d13_wedge",
            Self::P3D11Contraction { .. } => "p3_d11_contraction",
        }
    }

    fn output_momentum(
        self,
        source: DegreeTwoMomentumMonomial,
    ) -> Result<MomentumMonomial, String> {
        match self {
            Self::P2D13Wedge => Ok(source.as_solver_monomial()),
            Self::P3D11Contraction { momentum_axis } => source.with_contraction_axis(momentum_axis),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecondMomentumFxSector {
    X2,
    X5,
}

impl SecondMomentumFxSector {
    fn label(self) -> &'static str {
        match self {
            Self::X2 => "X2_11000",
            Self::X5 => "X5_10002",
        }
    }
}

/// A physical consumer must provide these target-resolved terms after the
/// component Clebsch-Gordan maps and `F_X` projection have been applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecondMomentumFxColumnTerm {
    pub coefficient_column: usize,
    pub gauge_channel: SecondMomentumGaugeChannel,
    pub gauge_branch: SecondMomentumGaugeBranch,
    pub source_momentum: DegreeTwoMomentumMonomial,
    pub parameter_component: usize,
    pub target_coordinate: usize,
    pub spinor_derivative_mask: u32,
    pub sector: SecondMomentumFxSector,
    pub coefficient: ExactGaussian,
}

/// Exact row identity produced by the contraction half of right normal
/// ordering a gauge derivative through a degree-twelve source monomial.
///
/// The third momentum is intentionally stored as an axis rather than folded
/// into the degree-two source monomial. This prevents distinct Clifford
/// contractions that happen to yield the same commutative momentum monomial
/// from being merged before the physical target projection has inspected
/// them.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct P3D11ContractionRow {
    pub gauge_channel: SecondMomentumGaugeChannel,
    pub parameter_component: usize,
    pub source_momentum: DegreeTwoMomentumMonomial,
    pub contraction_axis: u8,
    pub spinor_derivative_mask: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct P3D11ContractionTerm {
    pub coefficient_column: usize,
    pub row: P3D11ContractionRow,
    pub coefficient: ExactGaussian,
}

/// Run-static exact Clifford data for the CPU `p^3 D^11` producer.
///
/// Building the 1,024 gauge-form matrices is expensive. Callers construct this
/// once and reuse it across every physical column and parameter component.
#[derive(Clone, Copy)]
struct P3D11GaugeEntry {
    derivative_spinor: u8,
    real: i64,
    imaginary: i64,
}

#[derive(Clone, Debug)]
struct P3D11FxTemplate {
    derivative_spinor: u8,
    sector: SecondMomentumFxSector,
    output_coordinate: usize,
    coefficient: ExactGaussian,
}

pub(crate) struct P3D11ExactStaticData {
    gauge_basis_by_degree: Vec<Vec<Vec<Vec<P3D11GaugeEntry>>>>,
    translation_weights: Vec<Vec<[(i64, i64); 11]>>,
    fx_templates: Vec<P3D11FxTemplate>,
}

impl P3D11ExactStaticData {
    pub(crate) fn build() -> Result<Self, String> {
        const EXPECTED_COMPONENTS: [usize; 6] = [1, 11, 55, 165, 330, 462];
        let mut gauge_basis_by_degree = (0..6).map(|_| Vec::new()).collect::<Vec<_>>();
        for (degree, _, matrix) in crate::eleven_dimensional_clifford::gauge_form_operator_basis() {
            if degree >= gauge_basis_by_degree.len() {
                return Err("p3 D11 gauge basis has an invalid form degree".to_string());
            }
            if matrix.len() != 32 || matrix.iter().any(|row| row.len() != 32) {
                return Err("p3 D11 gauge matrix is not 32 by 32".to_string());
            }
            let mut component = (0..32).map(|_| Vec::new()).collect::<Vec<_>>();
            for (free_spinor, row) in matrix.into_iter().enumerate() {
                for (derivative_spinor, value) in row.into_iter().enumerate() {
                    if *value.re.denom() != 1 || *value.im.denom() != 1 {
                        return Err("p3 D11 gauge matrix is not integral".to_string());
                    }
                    let real = *value.re.numer();
                    let imaginary = *value.im.numer();
                    if real != 0 || imaginary != 0 {
                        component[free_spinor].push(P3D11GaugeEntry {
                            derivative_spinor: u8::try_from(derivative_spinor).unwrap(),
                            real,
                            imaginary,
                        });
                    }
                }
            }
            gauge_basis_by_degree[degree].push(component);
        }
        for (degree, (basis, expected)) in gauge_basis_by_degree
            .iter()
            .zip(EXPECTED_COMPONENTS)
            .enumerate()
        {
            if basis.len() != expected {
                return Err(format!(
                    "p3 D11 gauge degree {degree} has {} components, expected {expected}",
                    basis.len()
                ));
            }
        }
        let translation_weights =
            crate::eleven_dimensional_level16_couplings::translation_weight_basis_coefficients();
        if translation_weights.len() != 32 || translation_weights.iter().any(|row| row.len() != 32)
        {
            return Err("p3 D11 translation table is not 32 by 32 by 11".to_string());
        }
        let highest_target =
            crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states()
                .into_iter()
                .find(|state| state.pbw_word_simple_roots.is_empty())
                .ok_or_else(|| "p3 D11 target dual has no highest state".to_string())?;
        let mut projected = BTreeMap::<(u8, SecondMomentumFxSector, usize), ExactGaussian>::new();
        for target in highest_target.raw_terms {
            if target.denominator == 0 {
                return Err("p3 D11 target dual has a zero denominator".to_string());
            }
            let target_coefficient = ExactGaussian {
                real: Ratio::new(
                    BigInt::from(target.numerator),
                    BigInt::from(target.denominator),
                ),
                imaginary: Ratio::from_integer(BigInt::from(0)),
            };
            let mut templates = Vec::new();
            crate::eleven_dimensional_physical_curvature::visit_exact_fx_derivative_templates(
                target.vector_weight_index,
                target.spinor_weight_index,
                |entry| templates.push(entry),
            )?;
            for template in templates {
                let sector = if template.x_two_sector {
                    SecondMomentumFxSector::X2
                } else {
                    SecondMomentumFxSector::X5
                };
                let key = (
                    u8::try_from(template.derivative_spinor_weight_index).map_err(|_| {
                        "p3 D11 F_X derivative spinor exceeds dimension 32".to_string()
                    })?,
                    sector,
                    template.output_coordinate,
                );
                let contribution =
                    multiply_exact_gaussians(&target_coefficient, &template.coefficient);
                let value = projected.entry(key).or_insert_with(ExactGaussian::zero);
                value.real += contribution.real;
                value.imaginary += contribution.imaginary;
                if value.is_zero() {
                    projected.remove(&key);
                }
            }
        }
        let fx_templates = projected
            .into_iter()
            .map(
                |((derivative_spinor, sector, output_coordinate), coefficient)| P3D11FxTemplate {
                    derivative_spinor,
                    sector,
                    output_coordinate,
                    coefficient,
                },
            )
            .collect::<Vec<_>>();
        if fx_templates.is_empty() {
            return Err("p3 D11 physical F_X templates vanished".to_string());
        }
        Ok(Self {
            gauge_basis_by_degree,
            translation_weights,
            fx_templates,
        })
    }

    fn gauge_component(
        &self,
        gauge_form_degree: usize,
        parameter_component: usize,
    ) -> Result<&[Vec<P3D11GaugeEntry>], String> {
        let basis = self
            .gauge_basis_by_degree
            .get(gauge_form_degree)
            .ok_or_else(|| "11D gauge form degree must lie in 0..=5".to_string())?;
        basis.get(parameter_component).map(Vec::as_slice).ok_or_else(|| {
            format!(
                "p3 D11 parameter component {parameter_component} is outside gauge degree {gauge_form_degree} dimension {}",
                basis.len()
            )
        })
    }
}

fn multiply_exact_gaussians(left: &ExactGaussian, right: &ExactGaussian) -> ExactGaussian {
    ExactGaussian {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn multiply_gaussian_integer_pairs(
    left_real: i128,
    left_imaginary: i128,
    right_real: i128,
    right_imaginary: i128,
) -> Result<(i128, i128), String> {
    let real = left_real
        .checked_mul(right_real)
        .and_then(|value| {
            left_imaginary
                .checked_mul(right_imaginary)
                .and_then(|other| value.checked_sub(other))
        })
        .ok_or_else(|| "p3 D11 Gaussian real product exceeds i128".to_string())?;
    let imaginary = left_real
        .checked_mul(right_imaginary)
        .and_then(|value| {
            left_imaginary
                .checked_mul(right_real)
                .and_then(|other| value.checked_add(other))
        })
        .ok_or_else(|| "p3 D11 Gaussian imaginary product exceeds i128".to_string())?;
    Ok((real, imaginary))
}

/// Produce the exact `p^3 D^11` half of
/// `D_[12] (C Gamma^[q]) D Lambda` for one recoupled physical column and one
/// gauge-form component.
///
/// This is deliberately a CPU reference producer. It reuses the certified
/// right-contraction sign and Cartesian translation weights from the level-16
/// engine. It does not perform the later target-vector-spinor or `F_X`
/// projection, and it never merges rows across contraction axes.
pub(crate) fn produce_p3_d11_contractions(
    static_data: &P3D11ExactStaticData,
    coefficient_column: usize,
    gauge_form_degree: usize,
    parameter_component: usize,
    source_terms: &[crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm],
) -> Result<Vec<P3D11ContractionTerm>, String> {
    if coefficient_column >= SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS {
        return Err("p3 D11 coefficient column must lie in 0..77".to_string());
    }
    let gauge_channel = SecondMomentumGaugeChannel::new(gauge_form_degree)?;
    let gauge_component = static_data.gauge_component(gauge_form_degree, parameter_component)?;
    let translation = &static_data.translation_weights;
    let mut accumulated = BTreeMap::<P3D11ContractionRow, ExactGaussian>::new();

    for source in source_terms {
        let left = usize::from(source.momentum_pair[0]);
        let right = usize::from(source.momentum_pair[1]);
        if left >= 11 || right >= 11 || left > right {
            return Err("p3 D11 source momentum pair is not canonical".to_string());
        }
        let free_spinor = usize::from(source.free_spinor);
        if free_spinor >= 32 || source.exterior_mask.count_ones() != 12 || source.coefficient == 0 {
            return Err("p3 D11 recoupled source term is not canonical".to_string());
        }
        let source_momentum = DegreeTwoMomentumMonomial::from_pair(left, right)?;
        for gauge in &gauge_component[free_spinor] {
            let derivative_spinor = usize::from(gauge.derivative_spinor);
            let gauge_real = i128::from(gauge.real);
            let gauge_imaginary = i128::from(gauge.imaginary);
            let mut occupied = source.exterior_mask;
            while occupied != 0 {
                let contracted_spinor = occupied.trailing_zeros() as usize;
                occupied &= occupied - 1;
                let contraction_sign =
                    crate::eleven_dimensional_level16_couplings::right_contraction_sign(
                        source.exterior_mask,
                        contracted_spinor,
                    )
                    .ok_or_else(|| "p3 D11 occupied contraction vanished".to_string())?;
                let output_mask = source.exterior_mask ^ (1_u32 << contracted_spinor);
                debug_assert_eq!(output_mask.count_ones(), 11);
                let source_scale = source
                    .coefficient
                    .checked_mul(contraction_sign)
                    .ok_or_else(|| "p3 D11 source coefficient exceeds i128".to_string())?;
                for contraction_axis in 0..11 {
                    let (translation_real, translation_imaginary) =
                        translation[contracted_spinor][derivative_spinor][contraction_axis];
                    if translation_real == 0 && translation_imaginary == 0 {
                        continue;
                    }
                    let (real, imaginary) = multiply_gaussian_integer_pairs(
                        gauge_real,
                        gauge_imaginary,
                        i128::from(translation_real),
                        i128::from(translation_imaginary),
                    )?;
                    let real = source_scale
                        .checked_mul(real)
                        .ok_or_else(|| "p3 D11 real coefficient exceeds i128".to_string())?;
                    let imaginary = source_scale
                        .checked_mul(imaginary)
                        .ok_or_else(|| "p3 D11 imaginary coefficient exceeds i128".to_string())?;
                    let row = P3D11ContractionRow {
                        gauge_channel,
                        parameter_component,
                        source_momentum,
                        contraction_axis: u8::try_from(contraction_axis).unwrap(),
                        spinor_derivative_mask: output_mask,
                    };
                    let value = accumulated
                        .entry(row.clone())
                        .or_insert_with(ExactGaussian::zero);
                    value.real += Ratio::from_integer(BigInt::from(real));
                    value.imaginary += Ratio::from_integer(BigInt::from(imaginary));
                    if value.is_zero() {
                        accumulated.remove(&row);
                    }
                }
            }
        }
    }

    Ok(accumulated
        .into_iter()
        .map(|(row, coefficient)| P3D11ContractionTerm {
            coefficient_column,
            row,
            coefficient,
        })
        .collect())
}

fn degree_twelve_to_degree_eleven_functional_mask(mask: u32) -> Result<u32, String> {
    if mask.count_ones() != 12 {
        return Err("p3 D11 target projection expected a degree-twelve mask".to_string());
    }
    let highest_spinor = 31 - mask.leading_zeros();
    Ok(mask ^ (1_u32 << highest_spinor))
}

/// Apply the exact highest vector-spinor dual and the physical `F_X`
/// derivative templates to raw contraction terms.
///
/// The derivative is right-wedged after the gauge contraction. The resulting
/// degree-twelve mask is mapped to the declared bounded degree-eleven
/// functional mask, while the contraction momentum axis remains explicit in
/// `SecondMomentumGaugeBranch`.
pub(crate) fn project_p3_d11_to_physical_fx(
    static_data: &P3D11ExactStaticData,
    contractions: &[P3D11ContractionTerm],
) -> Result<Vec<SecondMomentumFxColumnTerm>, String> {
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
    struct Key {
        coefficient_column: usize,
        gauge_channel: SecondMomentumGaugeChannel,
        parameter_component: usize,
        source_momentum: DegreeTwoMomentumMonomial,
        contraction_axis: u8,
        target_coordinate: usize,
        functional_mask: u32,
        sector: SecondMomentumFxSector,
    }

    let mut accumulated = BTreeMap::<Key, ExactGaussian>::new();
    for contraction in contractions {
        if contraction.coefficient_column >= SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
            || contraction.row.parameter_component
                >= static_data.gauge_basis_by_degree[contraction.row.gauge_channel.form_degree()]
                    .len()
            || contraction.row.source_momentum.total_degree() != 2
            || usize::from(contraction.row.contraction_axis) >= 11
            || contraction.row.spinor_derivative_mask.count_ones() != 11
            || contraction.coefficient.is_zero()
        {
            return Err("p3 D11 contraction term is not canonical".to_string());
        }
        for template in &static_data.fx_templates {
            let Some((degree_twelve_mask, sign)) =
                crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(
                    contraction.row.spinor_derivative_mask,
                    usize::from(template.derivative_spinor),
                )
            else {
                continue;
            };
            let functional_mask =
                degree_twelve_to_degree_eleven_functional_mask(degree_twelve_mask)?;
            let mut value =
                multiply_exact_gaussians(&contraction.coefficient, &template.coefficient);
            if sign < 0 {
                value.real = -value.real;
                value.imaginary = -value.imaginary;
            }
            if value.is_zero() {
                continue;
            }
            let key = Key {
                coefficient_column: contraction.coefficient_column,
                gauge_channel: contraction.row.gauge_channel,
                parameter_component: contraction.row.parameter_component,
                source_momentum: contraction.row.source_momentum,
                contraction_axis: contraction.row.contraction_axis,
                target_coordinate: template.output_coordinate,
                functional_mask,
                sector: template.sector,
            };
            let entry = accumulated.entry(key).or_insert_with(ExactGaussian::zero);
            entry.real += value.real;
            entry.imaginary += value.imaginary;
            if entry.is_zero() {
                accumulated.remove(&key);
            }
        }
    }

    accumulated
        .into_iter()
        .map(|(key, coefficient)| {
            let term = SecondMomentumFxColumnTerm {
                coefficient_column: key.coefficient_column,
                gauge_channel: key.gauge_channel,
                gauge_branch: SecondMomentumGaugeBranch::P3D11Contraction {
                    momentum_axis: key.contraction_axis,
                },
                source_momentum: key.source_momentum,
                parameter_component: key.parameter_component,
                target_coordinate: key.target_coordinate,
                spinor_derivative_mask: key.functional_mask,
                sector: key.sector,
                coefficient,
            };
            term.validate()?;
            Ok(term)
        })
        .collect()
}

impl SecondMomentumFxColumnTerm {
    pub fn validate(&self) -> Result<(), String> {
        if self.coefficient_column >= SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS {
            return Err("second-momentum coefficient column must lie in 0..77".to_string());
        }
        if self.source_momentum.total_degree() != 2 {
            return Err("functional term lost its degree-two source monomial".to_string());
        }
        if self.spinor_derivative_mask.count_ones() as usize != self.gauge_branch.derivative_order()
        {
            return Err("functional term has the wrong spinor-derivative order".to_string());
        }
        self.gauge_branch.output_momentum(self.source_momentum)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecondMomentumFxSourceKind {
    PhysicalRecoupledStream,
    SyntheticControl,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxProvenance {
    pub source_kind: SecondMomentumFxSourceKind,
    pub campaign_id: String,
    pub representation_inventory_sha256: String,
    pub level12_fixture_manifest_sha256: String,
    pub component_cg_manifest_sha256: String,
    pub coefficient_layout_sha256: String,
    pub expected_canonical_stream_sha256: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxCoverage {
    pub all_77_component_cg_maps_complete: bool,
    pub p2_d13_wedge_branch_complete: bool,
    pub p3_d11_contraction_branch_complete: bool,
    pub all_six_gauge_channels_complete: bool,
    pub full_parameter_projection_complete: bool,
    pub full_target_projection_complete: bool,
    pub complete_x2_projection: bool,
    pub complete_x5_projection: bool,
    pub j_and_w_sectors_complete: bool,
    pub generic_momentum_tower_complete_or_proved_sufficient: bool,
}

/// Object-safe stream interface for the future real component recoupling.
/// Implementations own the physical construction. The harness only hashes,
/// solves, checkpoints, and applies proof gates.
pub trait SecondMomentumFxTermSource {
    fn provenance(&self) -> SecondMomentumFxProvenance;
    fn coverage(&self) -> SecondMomentumFxCoverage;
    fn visit_terms(
        &self,
        visitor: &mut dyn FnMut(SecondMomentumFxColumnTerm),
    ) -> Result<(), String>;
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct FunctionalGroupKey {
    gauge_channel: SecondMomentumGaugeChannel,
    gauge_branch: SecondMomentumGaugeBranch,
    source_momentum: DegreeTwoMomentumMonomial,
    sector: SecondMomentumFxSector,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct FunctionalRowKey {
    group: FunctionalGroupKey,
    seed_ordinal: usize,
    bucket: usize,
}

type FunctionalRows = BTreeMap<FunctionalRowKey, Vec<ExactGaussian>>;
type SparseFunctionalRows = BTreeMap<FunctionalRowKey, BTreeMap<usize, ExactGaussian>>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactRankSummary {
    pub equation_count: usize,
    pub variable_count: usize,
    pub rank: usize,
    pub nullity: usize,
    pub outcome: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxChannelRank {
    pub gauge_form_degree: usize,
    pub observed_terms: usize,
    pub wedge_terms: usize,
    pub contraction_terms: usize,
    pub x2: ExactRankSummary,
    pub x5: ExactRankSummary,
    pub joint: ExactRankSummary,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxBranchRank {
    pub branch: String,
    pub observed_terms: usize,
    pub x2: ExactRankSummary,
    pub x5: ExactRankSummary,
    pub joint: ExactRankSummary,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxFunctionalReport {
    pub schema_version: String,
    pub role: String,
    pub provenance: SecondMomentumFxProvenance,
    pub coverage: SecondMomentumFxCoverage,
    pub formal_momentum_variables: usize,
    pub source_momentum_degree: usize,
    pub coefficient_variables: usize,
    pub gauge_channels_expected: usize,
    pub functional_seeds: Vec<String>,
    pub buckets_per_seed: usize,
    pub functional_rows_per_observed_group: usize,
    pub observed_terms: usize,
    pub observed_gauge_channels: Vec<usize>,
    pub observed_source_monomials: usize,
    pub canonical_stream_sha256: String,
    pub stream_hash_matches_expected: bool,
    pub channel_ranks: Vec<SecondMomentumFxChannelRank>,
    pub branch_ranks: Vec<SecondMomentumFxBranchRank>,
    pub global_x2: ExactRankSummary,
    pub global_x5: ExactRankSummary,
    pub global_joint: ExactRankSummary,
    pub at_least_128_joint_functional_rows: bool,
    pub rank_dimension_saturated: bool,
    pub component_cg_gate_passed: bool,
    pub parameter_projection_gate_passed: bool,
    pub target_projection_gate_passed: bool,
    pub declared_slice_no_go_certified: bool,
    pub surviving_kernel_is_physical_certificate: bool,
    pub partial_fx_only: bool,
    pub full_f_a_g_p_established: bool,
    pub boundary: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxCheckpoint {
    pub schema_version: String,
    pub report: SecondMomentumFxFunctionalReport,
    pub checkpoint_sha256: String,
}

pub const SECOND_MOMENTUM_FX_STREAMING_REPORT_SCHEMA: &str =
    "adynkra-11d-second-momentum-streaming-fx-functional-v1";
pub const SECOND_MOMENTUM_FX_STREAMING_CHECKPOINT_SCHEMA: &str =
    "adynkra-11d-second-momentum-streaming-fx-checkpoint-v1";

/// Memory-bounded exact rank report.  The digest covers the exact sparse
/// functional rows after all physical terms with identical keys have been
/// summed.  It is not a digest of an uncompressed component-term list.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxStreamingReport {
    pub schema_version: String,
    pub role: String,
    pub provenance: SecondMomentumFxProvenance,
    pub coverage: SecondMomentumFxCoverage,
    pub coefficient_variables: usize,
    pub observed_terms: u64,
    pub observed_gauge_channels: Vec<usize>,
    pub observed_source_monomials: usize,
    pub observed_nonzero_columns: Vec<usize>,
    pub functional_groups: usize,
    pub sparse_functional_coefficients: usize,
    pub canonical_functional_rows_sha256: String,
    pub per_column_functional_rows_sha256: Vec<String>,
    pub channel_ranks: Vec<SecondMomentumFxChannelRank>,
    pub branch_ranks: Vec<SecondMomentumFxBranchRank>,
    pub global_x2: ExactRankSummary,
    pub global_x5: ExactRankSummary,
    pub global_joint: ExactRankSummary,
    pub rank_dimension_saturated: bool,
    pub declared_slice_no_go_certified: bool,
    pub partial_fx_only: bool,
    pub full_f_a_g_p_established: bool,
    pub boundary: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondMomentumFxStreamingCheckpoint {
    pub schema_version: String,
    pub report: SecondMomentumFxStreamingReport,
    pub checkpoint_sha256: String,
}

/// Streaming accumulator for actual physical component terms.  It retains
/// only exact nonzero functional-row coefficients, never the component stream
/// itself and never dense 77-entry zero padding for each row.
pub struct SecondMomentumFxStreamingAccumulator {
    provenance: SecondMomentumFxProvenance,
    coverage: SecondMomentumFxCoverage,
    groups: BTreeSet<FunctionalGroupKey>,
    rows: SparseFunctionalRows,
    observed_terms: u64,
    observed_channels: BTreeSet<usize>,
    observed_monomials: BTreeSet<DegreeTwoMomentumMonomial>,
    observed_columns: BTreeSet<usize>,
    terms_by_channel: [u64; SECOND_MOMENTUM_FX_GAUGE_CHANNELS],
    wedge_terms_by_channel: [u64; SECOND_MOMENTUM_FX_GAUGE_CHANNELS],
    contraction_terms_by_channel: [u64; SECOND_MOMENTUM_FX_GAUGE_CHANNELS],
}

/// One already-summed exact functional coefficient. This is the proof-safe
/// boundary for producers that exploit linearity and project before
/// materializing a component stream.
#[derive(Clone)]
pub(crate) struct SecondMomentumFxProjectedEntry {
    pub source_momentum: DegreeTwoMomentumMonomial,
    pub sector: SecondMomentumFxSector,
    pub seed_ordinal: usize,
    pub bucket: usize,
    pub coefficient: ExactGaussian,
}

/// A complete projected column. `observed_terms` retains the semantic count
/// of nonzero component terms before projection, while `entries` contains
/// only the exact nonzero functional rows.
#[derive(Clone)]
pub(crate) struct SecondMomentumFxProjectedColumn {
    pub coefficient_column: usize,
    pub gauge_channel: SecondMomentumGaugeChannel,
    pub gauge_branch: SecondMomentumGaugeBranch,
    pub observed_terms: u64,
    pub observed_groups: Vec<(DegreeTwoMomentumMonomial, SecondMomentumFxSector)>,
    pub entries: Vec<SecondMomentumFxProjectedEntry>,
}

#[derive(Serialize)]
struct CheckpointHashPayload<'a> {
    schema_version: &'a str,
    report: &'a SecondMomentumFxFunctionalReport,
}

#[derive(Serialize)]
struct CanonicalTerm {
    coefficient_column: usize,
    gauge_form_degree: usize,
    gauge_branch: SecondMomentumGaugeBranch,
    source_momentum_exponents: [u16; 11],
    parameter_component: usize,
    target_coordinate: usize,
    spinor_derivative_mask: u32,
    sector: SecondMomentumFxSector,
    real_numerator: String,
    real_denominator: String,
    imaginary_numerator: String,
    imaginary_denominator: String,
}

fn canonical_term(term: &SecondMomentumFxColumnTerm) -> CanonicalTerm {
    CanonicalTerm {
        coefficient_column: term.coefficient_column,
        gauge_form_degree: term.gauge_channel.form_degree(),
        gauge_branch: term.gauge_branch,
        source_momentum_exponents: term.source_momentum.exponents,
        parameter_component: term.parameter_component,
        target_coordinate: term.target_coordinate,
        spinor_derivative_mask: term.spinor_derivative_mask,
        sector: term.sector,
        real_numerator: term.coefficient.real.numer().to_string(),
        real_denominator: term.coefficient.real.denom().to_string(),
        imaginary_numerator: term.coefficient.imaginary.numer().to_string(),
        imaginary_denominator: term.coefficient.imaginary.denom().to_string(),
    }
}

fn canonical_term_sort_key(term: &SecondMomentumFxColumnTerm) -> String {
    serde_json::to_string(&canonical_term(term)).expect("serialize canonical p2 F_X term")
}

pub fn canonical_second_momentum_fx_stream_sha256(terms: &[SecondMomentumFxColumnTerm]) -> String {
    let mut ordered = terms.iter().collect::<Vec<_>>();
    ordered.sort_by_cached_key(|term| canonical_term_sort_key(term));
    let canonical = ordered.into_iter().map(canonical_term).collect::<Vec<_>>();
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical).expect("serialize p2 F_X stream"))
    )
}

pub fn second_momentum_fx_coefficient_layout_sha256() -> String {
    let specs = second_momentum_coefficient_specs();
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&specs).expect("serialize p2 coefficient layout"))
    )
}

fn second_momentum_coefficient_specs() -> Vec<KCoefficientSpec> {
    (0..SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS)
        .map(|ordinal| KCoefficientSpec {
            ordinal,
            label: format!("second-p2D12-{ordinal:02}"),
            operator_kind: "second-momentum".to_string(),
            spinor_derivative_order_before_gauge_map: 12,
            momentum_degree_before_gauge_map: 2,
            lower_symbol_status:
                "bounded degree-two column; subsequent p^3 D^10 symbols are not exhausted"
                    .to_string(),
        })
        .collect()
}

fn sha256_is_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_provenance(provenance: &SecondMomentumFxProvenance) -> Result<(), String> {
    if provenance.campaign_id.trim().is_empty()
        || provenance.representation_inventory_sha256
            != SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256
        || provenance.coefficient_layout_sha256 != second_momentum_fx_coefficient_layout_sha256()
        || !sha256_is_valid(&provenance.level12_fixture_manifest_sha256)
        || !sha256_is_valid(&provenance.component_cg_manifest_sha256)
        || !sha256_is_valid(&provenance.expected_canonical_stream_sha256)
    {
        return Err("second-momentum F_X provenance is incomplete or mismatched".to_string());
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn functional_hash_parts(
    gauge_channel: SecondMomentumGaugeChannel,
    gauge_branch: SecondMomentumGaugeBranch,
    source_momentum: DegreeTwoMomentumMonomial,
    parameter_component: usize,
    target_coordinate: usize,
    spinor_derivative_mask: u32,
    sector: SecondMomentumFxSector,
) -> u64 {
    // The functional acts on the physical output key, never on the unknown
    // coefficient-column ordinal. Including the column here would manufacture
    // independence instead of proving a rank lower bound.
    let mut value = (gauge_channel.form_degree() as u64).rotate_left(9)
        ^ (parameter_component as u64).rotate_left(17)
        ^ (target_coordinate as u64).rotate_left(31)
        ^ u64::from(spinor_derivative_mask).rotate_left(43);
    value ^= match gauge_branch {
        SecondMomentumGaugeBranch::P2D13Wedge => 0x02d1_3000_0000_0001,
        SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis } => {
            0x03d1_1000_0000_0002 ^ u64::from(momentum_axis).rotate_left(53)
        }
    };
    value ^= match sector {
        SecondMomentumFxSector::X2 => 0x1100_0000_0000_0002,
        SecondMomentumFxSector::X5 => 0x1000_2000_0000_0005,
    };
    for (axis, exponent) in source_momentum.exponents.iter().enumerate() {
        value ^= (u64::from(*exponent) + 1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15_u64.rotate_left(axis as u32));
    }
    splitmix64(value)
}

fn functional_hash(term: &SecondMomentumFxColumnTerm) -> u64 {
    functional_hash_parts(
        term.gauge_channel,
        term.gauge_branch,
        term.source_momentum,
        term.parameter_component,
        term.target_coordinate,
        term.spinor_derivative_mask,
        term.sector,
    )
}

/// Return the exact deterministic bucket and sign selected by every pinned
/// functional seed.  The packed accelerator uses this same boundary, so it
/// cannot accidentally manufacture independence by hashing the column index.
pub(crate) fn second_momentum_fx_functional_assignments(
    gauge_channel: SecondMomentumGaugeChannel,
    gauge_branch: SecondMomentumGaugeBranch,
    source_momentum: DegreeTwoMomentumMonomial,
    parameter_component: usize,
    target_coordinate: usize,
    spinor_derivative_mask: u32,
    sector: SecondMomentumFxSector,
) -> [(usize, i8); SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len()] {
    let base = functional_hash_parts(
        gauge_channel,
        gauge_branch,
        source_momentum,
        parameter_component,
        target_coordinate,
        spinor_derivative_mask,
        sector,
    );
    std::array::from_fn(|seed_ordinal| {
        let hash = splitmix64(base ^ SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS[seed_ordinal]);
        (
            hash as usize % SECOND_MOMENTUM_FX_BUCKETS_PER_SEED,
            if hash >> 63 == 0 { 1 } else { -1 },
        )
    })
}

fn add_scaled(target: &mut ExactGaussian, source: &ExactGaussian, sign: i64) {
    let scale = Ratio::from_integer(BigInt::from(sign));
    target.real += source.real.clone() * scale.clone();
    target.imaginary += source.imaginary.clone() * scale;
}

fn functional_rows(terms: &[SecondMomentumFxColumnTerm]) -> FunctionalRows {
    let groups = terms
        .iter()
        .map(|term| FunctionalGroupKey {
            gauge_channel: term.gauge_channel,
            gauge_branch: term.gauge_branch,
            source_momentum: term.source_momentum,
            sector: term.sector,
        })
        .collect::<BTreeSet<_>>();
    let mut rows = FunctionalRows::new();
    for group in groups {
        for seed_ordinal in 0..SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len() {
            for bucket in 0..SECOND_MOMENTUM_FX_BUCKETS_PER_SEED {
                rows.insert(
                    FunctionalRowKey {
                        group: group.clone(),
                        seed_ordinal,
                        bucket,
                    },
                    vec![ExactGaussian::zero(); SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS],
                );
            }
        }
    }
    for term in terms {
        let base = functional_hash(term);
        for (seed_ordinal, seed) in SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.iter().enumerate() {
            let hash = splitmix64(base ^ seed);
            let bucket = hash as usize % SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;
            let sign = if hash >> 63 == 0 { 1 } else { -1 };
            let key = FunctionalRowKey {
                group: FunctionalGroupKey {
                    gauge_channel: term.gauge_channel,
                    gauge_branch: term.gauge_branch,
                    source_momentum: term.source_momentum,
                    sector: term.sector,
                },
                seed_ordinal,
                bucket,
            };
            add_scaled(
                &mut rows.get_mut(&key).expect("preallocated p2 functional row")
                    [term.coefficient_column],
                &term.coefficient,
                sign,
            );
        }
    }
    rows
}

fn add_sparse_functional_term(rows: &mut SparseFunctionalRows, term: &SecondMomentumFxColumnTerm) {
    let group = FunctionalGroupKey {
        gauge_channel: term.gauge_channel,
        gauge_branch: term.gauge_branch,
        source_momentum: term.source_momentum,
        sector: term.sector,
    };
    let base = functional_hash(term);
    for (seed_ordinal, seed) in SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.iter().enumerate() {
        let hash = splitmix64(base ^ seed);
        let bucket = hash as usize % SECOND_MOMENTUM_FX_BUCKETS_PER_SEED;
        let sign = if hash >> 63 == 0 { 1 } else { -1 };
        let row = rows
            .entry(FunctionalRowKey {
                group: group.clone(),
                seed_ordinal,
                bucket,
            })
            .or_default();
        let coefficient = row
            .entry(term.coefficient_column)
            .or_insert_with(ExactGaussian::zero);
        add_scaled(coefficient, &term.coefficient, sign);
        if coefficient.is_zero() {
            row.remove(&term.coefficient_column);
        }
    }
    rows.retain(|_, row| !row.is_empty());
}

#[derive(Serialize)]
struct CanonicalSparseFunctionalGroup {
    gauge_form_degree: usize,
    gauge_branch: SecondMomentumGaugeBranch,
    source_momentum_exponents: [u16; 11],
    sector: SecondMomentumFxSector,
}

#[derive(Serialize)]
struct CanonicalSparseFunctionalEntry {
    group: CanonicalSparseFunctionalGroup,
    seed_ordinal: usize,
    bucket: usize,
    coefficient_column: usize,
    real_numerator: String,
    real_denominator: String,
    imaginary_numerator: String,
    imaginary_denominator: String,
}

fn canonical_sparse_group(group: &FunctionalGroupKey) -> CanonicalSparseFunctionalGroup {
    CanonicalSparseFunctionalGroup {
        gauge_form_degree: group.gauge_channel.form_degree(),
        gauge_branch: group.gauge_branch,
        source_momentum_exponents: group.source_momentum.exponents,
        sector: group.sector,
    }
}

fn sparse_functional_rows_sha256(
    groups: &BTreeSet<FunctionalGroupKey>,
    rows: &SparseFunctionalRows,
    selected_column: Option<usize>,
) -> String {
    let group_manifest = groups
        .iter()
        .map(canonical_sparse_group)
        .collect::<Vec<_>>();
    let entries = rows
        .iter()
        .flat_map(|(key, coefficients)| {
            coefficients
                .iter()
                .filter_map(move |(column, coefficient)| {
                    if selected_column.is_some_and(|selected| selected != *column) {
                        return None;
                    }
                    Some(CanonicalSparseFunctionalEntry {
                        group: canonical_sparse_group(&key.group),
                        seed_ordinal: key.seed_ordinal,
                        bucket: key.bucket,
                        coefficient_column: *column,
                        real_numerator: coefficient.real.numer().to_string(),
                        real_denominator: coefficient.real.denom().to_string(),
                        imaginary_numerator: coefficient.imaginary.numer().to_string(),
                        imaginary_denominator: coefficient.imaginary.denom().to_string(),
                    })
                })
        })
        .collect::<Vec<_>>();
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(group_manifest, entries))
                .expect("serialize canonical sparse p2 functional rows")
        )
    )
}

fn solve_sparse_rows(
    rows: &SparseFunctionalRows,
    channel: Option<SecondMomentumGaugeChannel>,
    branch: Option<&str>,
    sector: Option<SecondMomentumFxSector>,
) -> ExactRankSummary {
    let selected = rows
        .iter()
        .filter(|(key, _)| row_selected(key, channel, branch, sector))
        .collect::<Vec<_>>();
    let active_columns = selected
        .iter()
        .flat_map(|(_, coefficients)| coefficients.keys().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if modular_full_column_rank(&selected, &active_columns).unwrap_or(false) {
        let rank = active_columns.len();
        return ExactRankSummary {
            equation_count: selected.len(),
            variable_count: SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS,
            rank,
            nullity: SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS - rank,
            outcome: if rank == SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS {
                "unique".to_string()
            } else {
                "family".to_string()
            },
        };
    }
    let mut basis = vec![None::<Vec<ExactGaussian>>; active_columns.len()];
    let mut rank = 0_usize;
    for (_, coefficients) in &selected {
        let mut vector = vec![ExactGaussian::zero(); active_columns.len()];
        for (column, coefficient) in *coefficients {
            let index = active_columns
                .binary_search(column)
                .expect("active sparse column is indexed");
            vector[index] = coefficient.clone();
        }
        for pivot in 0..active_columns.len() {
            if vector[pivot].is_zero() {
                continue;
            }
            if let Some(pivot_row) = &basis[pivot] {
                let factor = divide_gaussian(&vector[pivot], &pivot_row[pivot]);
                for column in pivot..active_columns.len() {
                    let product = multiply_gaussian(&factor, &pivot_row[column]);
                    vector[column].real -= product.real;
                    vector[column].imaginary -= product.imaginary;
                }
            } else {
                basis[pivot] = Some(vector);
                rank += 1;
                break;
            }
        }
        if rank == active_columns.len() {
            break;
        }
    }
    ExactRankSummary {
        equation_count: selected.len(),
        variable_count: SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS,
        rank,
        nullity: SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS - rank,
        outcome: if rank == SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS {
            "unique".to_string()
        } else {
            "family".to_string()
        },
    }
}

const RANK_PRIME: u64 = 2_147_483_647;

#[derive(Clone, Copy)]
struct ModularGaussian {
    real: u64,
    imaginary: u64,
}

impl ModularGaussian {
    const ZERO: Self = Self {
        real: 0,
        imaginary: 0,
    };

    fn is_zero(self) -> bool {
        self.real == 0 && self.imaginary == 0
    }

    fn subtract(self, right: Self) -> Self {
        Self {
            real: field_sub(self.real, right.real),
            imaginary: field_sub(self.imaginary, right.imaginary),
        }
    }

    fn multiply(self, right: Self) -> Self {
        Self {
            real: field_sub(
                field_mul(self.real, right.real),
                field_mul(self.imaginary, right.imaginary),
            ),
            imaginary: field_add(
                field_mul(self.real, right.imaginary),
                field_mul(self.imaginary, right.real),
            ),
        }
    }

    fn divide(self, right: Self) -> Self {
        debug_assert!(!right.is_zero());
        let norm = field_add(
            field_mul(right.real, right.real),
            field_mul(right.imaginary, right.imaginary),
        );
        debug_assert_ne!(norm, 0);
        let inverse_norm = field_pow(norm, RANK_PRIME - 2);
        self.multiply(Self {
            real: field_mul(right.real, inverse_norm),
            imaginary: field_mul(
                if right.imaginary == 0 {
                    0
                } else {
                    RANK_PRIME - right.imaginary
                },
                inverse_norm,
            ),
        })
    }
}

fn field_add(left: u64, right: u64) -> u64 {
    let sum = left + right;
    if sum >= RANK_PRIME {
        sum - RANK_PRIME
    } else {
        sum
    }
}

fn field_sub(left: u64, right: u64) -> u64 {
    if left >= right {
        left - right
    } else {
        left + RANK_PRIME - right
    }
}

fn field_mul(left: u64, right: u64) -> u64 {
    (left * right) % RANK_PRIME
}

fn field_pow(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = field_mul(result, base);
        }
        base = field_mul(base, base);
        exponent >>= 1;
    }
    result
}

fn bigint_mod_prime(value: &BigInt, prime: &BigInt) -> u64 {
    let residue = (value % prime)
        .to_i64()
        .expect("rank-prime residue fits i64");
    if residue < 0 {
        (residue + RANK_PRIME as i64) as u64
    } else {
        residue as u64
    }
}

fn ratio_mod_prime(value: &Ratio<BigInt>, prime: &BigInt) -> Option<u64> {
    let numerator = bigint_mod_prime(value.numer(), prime);
    let denominator = bigint_mod_prime(value.denom(), prime);
    (denominator != 0).then(|| field_mul(numerator, field_pow(denominator, RANK_PRIME - 2)))
}

fn gaussian_mod_prime(value: &ExactGaussian, prime: &BigInt) -> Option<ModularGaussian> {
    Some(ModularGaussian {
        real: ratio_mod_prime(&value.real, prime)?,
        imaginary: ratio_mod_prime(&value.imaginary, prime)?,
    })
}

/// A full-rank image over GF(p^2), with p = 3 mod 4, proves full column rank
/// over Q(i). A deficient or undefined modular image is never trusted and
/// falls through to the characteristic-zero eliminator below.
fn modular_full_column_rank(
    selected: &[(&FunctionalRowKey, &BTreeMap<usize, ExactGaussian>)],
    active_columns: &[usize],
) -> Option<bool> {
    if active_columns.is_empty() {
        return Some(false);
    }
    let prime = BigInt::from(RANK_PRIME);
    let mut basis = vec![None::<Vec<ModularGaussian>>; active_columns.len()];
    let mut rank = 0;
    for (_, coefficients) in selected {
        let mut vector = vec![ModularGaussian::ZERO; active_columns.len()];
        for (column, coefficient) in *coefficients {
            let index = active_columns
                .binary_search(column)
                .expect("active modular column is indexed");
            vector[index] = gaussian_mod_prime(coefficient, &prime)?;
        }
        for pivot in 0..active_columns.len() {
            if vector[pivot].is_zero() {
                continue;
            }
            if let Some(pivot_row) = &basis[pivot] {
                let factor = vector[pivot].divide(pivot_row[pivot]);
                for column in pivot..active_columns.len() {
                    vector[column] = vector[column].subtract(factor.multiply(pivot_row[column]));
                }
            } else {
                basis[pivot] = Some(vector);
                rank += 1;
                break;
            }
        }
        if rank == active_columns.len() {
            return Some(true);
        }
    }
    Some(false)
}

fn multiply_gaussian(left: &ExactGaussian, right: &ExactGaussian) -> ExactGaussian {
    ExactGaussian {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn divide_gaussian(left: &ExactGaussian, right: &ExactGaussian) -> ExactGaussian {
    debug_assert!(!right.is_zero());
    let denominator =
        right.real.clone() * right.real.clone() + right.imaginary.clone() * right.imaginary.clone();
    ExactGaussian {
        real: (left.real.clone() * right.real.clone()
            + left.imaginary.clone() * right.imaginary.clone())
            / denominator.clone(),
        imaginary: (left.imaginary.clone() * right.real.clone()
            - left.real.clone() * right.imaginary.clone())
            / denominator,
    }
}

impl SecondMomentumFxStreamingAccumulator {
    pub fn new(
        provenance: SecondMomentumFxProvenance,
        coverage: SecondMomentumFxCoverage,
    ) -> Result<Self, String> {
        validate_provenance(&provenance)?;
        Ok(Self {
            provenance,
            coverage,
            groups: BTreeSet::new(),
            rows: SparseFunctionalRows::new(),
            observed_terms: 0,
            observed_channels: BTreeSet::new(),
            observed_monomials: BTreeSet::new(),
            observed_columns: BTreeSet::new(),
            terms_by_channel: [0; SECOND_MOMENTUM_FX_GAUGE_CHANNELS],
            wedge_terms_by_channel: [0; SECOND_MOMENTUM_FX_GAUGE_CHANNELS],
            contraction_terms_by_channel: [0; SECOND_MOMENTUM_FX_GAUGE_CHANNELS],
        })
    }

    /// Merge a producer-side exact linear projection without replaying every
    /// component term through the functional hash. The whole column is
    /// validated before mutation, so an error cannot leave a partial fold.
    pub(crate) fn push_projected_column(
        &mut self,
        column: SecondMomentumFxProjectedColumn,
    ) -> Result<(), String> {
        if column.coefficient_column >= SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
            || column.observed_terms == 0
            || column.observed_groups.is_empty()
            || column.entries.is_empty()
            || self.observed_columns.contains(&column.coefficient_column)
        {
            return Err("invalid or duplicate projected second-momentum column".to_string());
        }
        let groups = column
            .observed_groups
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if groups.len() != column.observed_groups.len()
            || column
                .observed_groups
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err("projected second-momentum groups are not canonical".to_string());
        }
        let mut previous = None;
        let mut local_rows = Vec::with_capacity(column.entries.len());
        for entry in column.entries {
            if entry.source_momentum.total_degree() != 2
                || entry.seed_ordinal >= SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len()
                || entry.bucket >= SECOND_MOMENTUM_FX_BUCKETS_PER_SEED
                || entry.coefficient.is_zero()
                || !groups.contains(&(entry.source_momentum, entry.sector))
            {
                return Err("invalid projected second-momentum row".to_string());
            }
            let order_key = (
                entry.source_momentum,
                entry.sector,
                entry.seed_ordinal,
                entry.bucket,
            );
            if previous.is_some_and(|prior| prior >= order_key) {
                return Err("projected second-momentum rows are not canonical".to_string());
            }
            previous = Some(order_key);
            local_rows.push((
                FunctionalRowKey {
                    group: FunctionalGroupKey {
                        gauge_channel: column.gauge_channel,
                        gauge_branch: column.gauge_branch,
                        source_momentum: entry.source_momentum,
                        sector: entry.sector,
                    },
                    seed_ordinal: entry.seed_ordinal,
                    bucket: entry.bucket,
                },
                entry.coefficient,
            ));
        }
        let degree = column.gauge_channel.form_degree();
        let observed_terms = self
            .observed_terms
            .checked_add(column.observed_terms)
            .ok_or_else(|| "projected second-momentum term count overflow".to_string())?;
        let channel_terms = self.terms_by_channel[degree]
            .checked_add(column.observed_terms)
            .ok_or_else(|| "projected channel term count overflow".to_string())?;
        let wedge_terms = match column.gauge_branch {
            SecondMomentumGaugeBranch::P2D13Wedge => Some(
                self.wedge_terms_by_channel[degree]
                    .checked_add(column.observed_terms)
                    .ok_or_else(|| "projected wedge term count overflow".to_string())?,
            ),
            SecondMomentumGaugeBranch::P3D11Contraction { .. } => None,
        };
        let contraction_terms = match column.gauge_branch {
            SecondMomentumGaugeBranch::P2D13Wedge => None,
            SecondMomentumGaugeBranch::P3D11Contraction { .. } => Some(
                self.contraction_terms_by_channel[degree]
                    .checked_add(column.observed_terms)
                    .ok_or_else(|| "projected contraction term count overflow".to_string())?,
            ),
        };

        self.observed_terms = observed_terms;
        self.terms_by_channel[degree] = channel_terms;
        if let Some(value) = wedge_terms {
            self.wedge_terms_by_channel[degree] = value;
        }
        if let Some(value) = contraction_terms {
            self.contraction_terms_by_channel[degree] = value;
        }
        self.observed_channels.insert(degree);
        self.observed_columns.insert(column.coefficient_column);
        for (source_momentum, sector) in groups {
            self.observed_monomials.insert(source_momentum);
            self.groups.insert(FunctionalGroupKey {
                gauge_channel: column.gauge_channel,
                gauge_branch: column.gauge_branch,
                source_momentum,
                sector,
            });
        }
        for (key, coefficient) in local_rows {
            let replaced = self
                .rows
                .entry(key)
                .or_default()
                .insert(column.coefficient_column, coefficient);
            debug_assert!(replaced.is_none());
        }
        Ok(())
    }

    pub fn push(&mut self, term: SecondMomentumFxColumnTerm) -> Result<(), String> {
        term.validate()?;
        let degree = term.gauge_channel.form_degree();
        self.observed_terms = self
            .observed_terms
            .checked_add(1)
            .ok_or_else(|| "streaming p2 F_X term count overflow".to_string())?;
        self.terms_by_channel[degree] += 1;
        match term.gauge_branch {
            SecondMomentumGaugeBranch::P2D13Wedge => self.wedge_terms_by_channel[degree] += 1,
            SecondMomentumGaugeBranch::P3D11Contraction { .. } => {
                self.contraction_terms_by_channel[degree] += 1
            }
        }
        self.observed_channels.insert(degree);
        self.observed_monomials.insert(term.source_momentum);
        self.observed_columns.insert(term.coefficient_column);
        self.groups.insert(FunctionalGroupKey {
            gauge_channel: term.gauge_channel,
            gauge_branch: term.gauge_branch,
            source_momentum: term.source_momentum,
            sector: term.sector,
        });
        add_sparse_functional_term(&mut self.rows, &term);
        Ok(())
    }

    pub fn finalize(mut self) -> Result<SecondMomentumFxStreamingCheckpoint, String> {
        let canonical_functional_rows_sha256 =
            sparse_functional_rows_sha256(&self.groups, &self.rows, None);
        self.provenance.expected_canonical_stream_sha256 = canonical_functional_rows_sha256.clone();
        let empty_column_sha256 =
            sparse_functional_rows_sha256(&self.groups, &self.rows, Some(usize::MAX));
        let ((per_column_functional_rows_sha256, channel_ranks), (branch_ranks, globals)) =
            rayon::join(
                || {
                    rayon::join(
                        || {
                            (0..SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS)
                                .into_par_iter()
                                .map(|column| {
                                    if self.observed_columns.contains(&column) {
                                        sparse_functional_rows_sha256(
                                            &self.groups,
                                            &self.rows,
                                            Some(column),
                                        )
                                    } else {
                                        empty_column_sha256.clone()
                                    }
                                })
                                .collect::<Vec<_>>()
                        },
                        || {
                            (0..SECOND_MOMENTUM_FX_GAUGE_CHANNELS)
                                .into_par_iter()
                                .map(|degree| {
                                    let channel = SecondMomentumGaugeChannel::new(degree).unwrap();
                                    SecondMomentumFxChannelRank {
                                        gauge_form_degree: degree,
                                        observed_terms: usize::try_from(
                                            self.terms_by_channel[degree],
                                        )
                                        .unwrap(),
                                        wedge_terms: usize::try_from(
                                            self.wedge_terms_by_channel[degree],
                                        )
                                        .unwrap(),
                                        contraction_terms: usize::try_from(
                                            self.contraction_terms_by_channel[degree],
                                        )
                                        .unwrap(),
                                        x2: solve_sparse_rows(
                                            &self.rows,
                                            Some(channel),
                                            None,
                                            Some(SecondMomentumFxSector::X2),
                                        ),
                                        x5: solve_sparse_rows(
                                            &self.rows,
                                            Some(channel),
                                            None,
                                            Some(SecondMomentumFxSector::X5),
                                        ),
                                        joint: solve_sparse_rows(
                                            &self.rows,
                                            Some(channel),
                                            None,
                                            None,
                                        ),
                                    }
                                })
                                .collect::<Vec<_>>()
                        },
                    )
                },
                || {
                    rayon::join(
                        || {
                            ["p2_d13_wedge", "p3_d11_contraction"]
                                .into_par_iter()
                                .map(|branch| SecondMomentumFxBranchRank {
                                    branch: branch.to_string(),
                                    observed_terms: if branch == "p2_d13_wedge" {
                                        self.wedge_terms_by_channel.iter().sum::<u64>()
                                    } else {
                                        self.contraction_terms_by_channel.iter().sum::<u64>()
                                    } as usize,
                                    x2: solve_sparse_rows(
                                        &self.rows,
                                        None,
                                        Some(branch),
                                        Some(SecondMomentumFxSector::X2),
                                    ),
                                    x5: solve_sparse_rows(
                                        &self.rows,
                                        None,
                                        Some(branch),
                                        Some(SecondMomentumFxSector::X5),
                                    ),
                                    joint: solve_sparse_rows(&self.rows, None, Some(branch), None),
                                })
                                .collect::<Vec<_>>()
                        },
                        || {
                            let ((global_x2, global_x5), global_joint) = rayon::join(
                                || {
                                    rayon::join(
                                        || {
                                            solve_sparse_rows(
                                                &self.rows,
                                                None,
                                                None,
                                                Some(SecondMomentumFxSector::X2),
                                            )
                                        },
                                        || {
                                            solve_sparse_rows(
                                                &self.rows,
                                                None,
                                                None,
                                                Some(SecondMomentumFxSector::X5),
                                            )
                                        },
                                    )
                                },
                                || solve_sparse_rows(&self.rows, None, None, None),
                            );
                            (global_x2, global_x5, global_joint)
                        },
                    )
                },
            );
        let (global_x2, global_x5, global_joint) = globals;
        let sparse_functional_coefficients = self.rows.values().map(BTreeMap::len).sum();
        let report = SecondMomentumFxStreamingReport {
            schema_version: SECOND_MOMENTUM_FX_STREAMING_REPORT_SCHEMA.to_string(),
            role: "memory-bounded exact 77-column second-momentum F_X functional rank screen"
                .to_string(),
            provenance: self.provenance,
            coverage: self.coverage,
            coefficient_variables: SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS,
            observed_terms: self.observed_terms,
            observed_gauge_channels: self.observed_channels.into_iter().collect(),
            observed_source_monomials: self.observed_monomials.len(),
            observed_nonzero_columns: self.observed_columns.into_iter().collect(),
            functional_groups: self.groups.len(),
            sparse_functional_coefficients,
            canonical_functional_rows_sha256,
            per_column_functional_rows_sha256,
            channel_ranks,
            branch_ranks,
            global_x2,
            global_x5,
            global_joint: global_joint.clone(),
            rank_dimension_saturated: global_joint.rank == SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS,
            declared_slice_no_go_certified: false,
            partial_fx_only: true,
            full_f_a_g_p_established: false,
            boundary: "The sparse functional rows are exact and give rigorous rank lower bounds for physically emitted columns. Missing columns are absent inputs, not synthetic streams. A saturated tranche rank can exclude that tranche on the declared slice. This streaming screen does not by itself certify all 77 component maps, complete parameters or targets, J/W, generic momentum sufficiency, or full F A G_p."
                .to_string(),
        };
        validate_second_momentum_fx_streaming_report(&report)?;
        let checkpoint_sha256 = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&(SECOND_MOMENTUM_FX_STREAMING_CHECKPOINT_SCHEMA, &report,))
                    .expect("serialize streaming p2 F_X checkpoint")
            )
        );
        Ok(SecondMomentumFxStreamingCheckpoint {
            schema_version: SECOND_MOMENTUM_FX_STREAMING_CHECKPOINT_SCHEMA.to_string(),
            report,
            checkpoint_sha256,
        })
    }
}

pub fn validate_second_momentum_fx_streaming_report(
    report: &SecondMomentumFxStreamingReport,
) -> Result<(), String> {
    validate_provenance(&report.provenance)?;
    if report.schema_version != SECOND_MOMENTUM_FX_STREAMING_REPORT_SCHEMA
        || report.coefficient_variables != SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
        || report.per_column_functional_rows_sha256.len() != SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
        || report.canonical_functional_rows_sha256
            != report.provenance.expected_canonical_stream_sha256
        || report.global_joint.rank + report.global_joint.nullity
            != SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
        || report.rank_dimension_saturated
            != (report.global_joint.rank == SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS)
        || report.declared_slice_no_go_certified
        || !report.partial_fx_only
        || report.full_f_a_g_p_established
    {
        return Err("streaming second-momentum F_X report invariant failed".to_string());
    }
    Ok(())
}

pub fn validate_second_momentum_fx_streaming_checkpoint(
    checkpoint: &SecondMomentumFxStreamingCheckpoint,
) -> Result<(), String> {
    validate_second_momentum_fx_streaming_report(&checkpoint.report)?;
    let expected = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                SECOND_MOMENTUM_FX_STREAMING_CHECKPOINT_SCHEMA,
                &checkpoint.report,
            ))
            .expect("serialize streaming p2 F_X checkpoint")
        )
    );
    if checkpoint.schema_version != SECOND_MOMENTUM_FX_STREAMING_CHECKPOINT_SCHEMA
        || checkpoint.checkpoint_sha256 != expected
    {
        return Err("streaming second-momentum F_X checkpoint hash mismatch".to_string());
    }
    Ok(())
}

pub fn second_momentum_fx_functional_rows_sha256(
    terms: &[SecondMomentumFxColumnTerm],
) -> Result<String, String> {
    let mut rows = SparseFunctionalRows::new();
    let mut groups = BTreeSet::new();
    for term in terms {
        term.validate()?;
        groups.insert(FunctionalGroupKey {
            gauge_channel: term.gauge_channel,
            gauge_branch: term.gauge_branch,
            source_momentum: term.source_momentum,
            sector: term.sector,
        });
        add_sparse_functional_term(&mut rows, term);
    }
    Ok(sparse_functional_rows_sha256(&groups, &rows, None))
}

fn row_selected(
    key: &FunctionalRowKey,
    channel: Option<SecondMomentumGaugeChannel>,
    branch: Option<&str>,
    sector: Option<SecondMomentumFxSector>,
) -> bool {
    channel.is_none_or(|value| key.group.gauge_channel == value)
        && branch.is_none_or(|value| key.group.gauge_branch.label() == value)
        && sector.is_none_or(|value| key.group.sector == value)
}

fn solve_rows(
    rows: &FunctionalRows,
    channel: Option<SecondMomentumGaugeChannel>,
    branch: Option<&str>,
    sector: Option<SecondMomentumFxSector>,
) -> ExactRankSummary {
    let mut system = ExactPolynomialSystem::new(second_momentum_coefficient_specs(), true);
    for (key, coefficients) in rows {
        if !row_selected(key, channel, branch, sector) {
            continue;
        }
        let output_momentum = key
            .group
            .gauge_branch
            .output_momentum(key.group.source_momentum)
            .expect("validated p2 functional output momentum");
        let polynomial_key = PolynomialConstraintKey {
            gauge_form_degree: key.group.gauge_channel.form_degree(),
            parameter_component: key.seed_ordinal * SECOND_MOMENTUM_FX_BUCKETS_PER_SEED
                + key.bucket,
            output_sector: format!(
                "{}:{}:source-p2-{}",
                key.group.sector.label(),
                key.group.gauge_branch.label(),
                key.group.source_momentum.compact_label()
            ),
            output_coordinate: key.bucket,
            spinor_derivative_mask: 0,
            spinor_derivative_order: key.group.gauge_branch.derivative_order(),
            momentum_monomial: output_momentum,
        };
        system.set_right_hand_side(polynomial_key.clone(), ExactGaussian::zero());
        for (column, coefficient) in coefficients.iter().enumerate() {
            system.add_coefficient(polynomial_key.clone(), column, coefficient.clone());
        }
    }
    let solution = system.solve();
    ExactRankSummary {
        equation_count: solution.equation_count,
        variable_count: solution.variable_count,
        rank: solution.rank,
        nullity: solution.nullity,
        outcome: format!("{:?}", solution.outcome).to_ascii_lowercase(),
    }
}

pub fn evaluate_second_momentum_fx_source(
    source: &dyn SecondMomentumFxTermSource,
) -> Result<SecondMomentumFxFunctionalReport, String> {
    let provenance = source.provenance();
    validate_provenance(&provenance)?;
    let coverage = source.coverage();
    let mut terms = Vec::new();
    source.visit_terms(&mut |term| terms.push(term))?;
    for term in &terms {
        term.validate()?;
    }
    let canonical_stream_sha256 = canonical_second_momentum_fx_stream_sha256(&terms);
    let stream_hash_matches_expected =
        canonical_stream_sha256 == provenance.expected_canonical_stream_sha256;
    let rows = functional_rows(&terms);
    let observed_channels = terms
        .iter()
        .map(|term| term.gauge_channel.form_degree())
        .collect::<BTreeSet<_>>();
    let observed_source_monomials = terms
        .iter()
        .map(|term| term.source_momentum)
        .collect::<BTreeSet<_>>()
        .len();

    let channel_ranks = (0..SECOND_MOMENTUM_FX_GAUGE_CHANNELS)
        .map(|degree| {
            let channel = SecondMomentumGaugeChannel::new(degree).unwrap();
            let selected = terms
                .iter()
                .filter(|term| term.gauge_channel == channel)
                .collect::<Vec<_>>();
            SecondMomentumFxChannelRank {
                gauge_form_degree: degree,
                observed_terms: selected.len(),
                wedge_terms: selected
                    .iter()
                    .filter(|term| {
                        matches!(term.gauge_branch, SecondMomentumGaugeBranch::P2D13Wedge)
                    })
                    .count(),
                contraction_terms: selected
                    .iter()
                    .filter(|term| {
                        matches!(
                            term.gauge_branch,
                            SecondMomentumGaugeBranch::P3D11Contraction { .. }
                        )
                    })
                    .count(),
                x2: solve_rows(&rows, Some(channel), None, Some(SecondMomentumFxSector::X2)),
                x5: solve_rows(&rows, Some(channel), None, Some(SecondMomentumFxSector::X5)),
                joint: solve_rows(&rows, Some(channel), None, None),
            }
        })
        .collect::<Vec<_>>();
    let branch_ranks = ["p2_d13_wedge", "p3_d11_contraction"]
        .into_iter()
        .map(|branch| SecondMomentumFxBranchRank {
            branch: branch.to_string(),
            observed_terms: terms
                .iter()
                .filter(|term| term.gauge_branch.label() == branch)
                .count(),
            x2: solve_rows(&rows, None, Some(branch), Some(SecondMomentumFxSector::X2)),
            x5: solve_rows(&rows, None, Some(branch), Some(SecondMomentumFxSector::X5)),
            joint: solve_rows(&rows, None, Some(branch), None),
        })
        .collect::<Vec<_>>();
    let global_x2 = solve_rows(&rows, None, None, Some(SecondMomentumFxSector::X2));
    let global_x5 = solve_rows(&rows, None, None, Some(SecondMomentumFxSector::X5));
    let global_joint = solve_rows(&rows, None, None, None);
    let observed_all_six = observed_channels.len() == SECOND_MOMENTUM_FX_GAUGE_CHANNELS;
    let observed_both_branches = branch_ranks.iter().all(|branch| branch.observed_terms > 0);
    let component_cg_gate_passed = provenance.source_kind
        == SecondMomentumFxSourceKind::PhysicalRecoupledStream
        && coverage.all_77_component_cg_maps_complete
        && observed_both_branches
        && observed_all_six
        && stream_hash_matches_expected;
    let parameter_projection_gate_passed =
        component_cg_gate_passed && coverage.full_parameter_projection_complete;
    let target_projection_gate_passed =
        component_cg_gate_passed && coverage.full_target_projection_complete;
    let at_least_128_joint_functional_rows =
        global_joint.equation_count >= SECOND_MOMENTUM_FX_FUNCTIONAL_ROWS;
    let rank_dimension_saturated =
        global_joint.rank == SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS && global_joint.nullity == 0;
    let declared_slice_no_go_certified = component_cg_gate_passed
        && coverage.p2_d13_wedge_branch_complete
        && coverage.p3_d11_contraction_branch_complete
        && coverage.all_six_gauge_channels_complete
        && coverage.complete_x2_projection
        && coverage.complete_x5_projection
        && parameter_projection_gate_passed
        && target_projection_gate_passed
        && at_least_128_joint_functional_rows
        && rank_dimension_saturated;

    Ok(SecondMomentumFxFunctionalReport {
        schema_version: SECOND_MOMENTUM_FX_REPORT_SCHEMA.to_string(),
        role: "exact bounded degree-two F_X functional rank screen".to_string(),
        provenance,
        coverage,
        formal_momentum_variables: 11,
        source_momentum_degree: 2,
        coefficient_variables: SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS,
        gauge_channels_expected: SECOND_MOMENTUM_FX_GAUGE_CHANNELS,
        functional_seeds: SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS
            .iter()
            .map(|seed| format!("{seed:016x}"))
            .collect(),
        buckets_per_seed: SECOND_MOMENTUM_FX_BUCKETS_PER_SEED,
        functional_rows_per_observed_group: SECOND_MOMENTUM_FX_FUNCTIONAL_ROWS,
        observed_terms: terms.len(),
        observed_gauge_channels: observed_channels.into_iter().collect(),
        observed_source_monomials,
        canonical_stream_sha256,
        stream_hash_matches_expected,
        channel_ranks,
        branch_ranks,
        global_x2,
        global_x5,
        global_joint,
        at_least_128_joint_functional_rows,
        rank_dimension_saturated,
        component_cg_gate_passed,
        parameter_projection_gate_passed,
        target_projection_gate_passed,
        declared_slice_no_go_certified,
        surviving_kernel_is_physical_certificate: false,
        partial_fx_only: true,
        full_f_a_g_p_established: false,
        boundary: "A saturated rank is a no-go only after the pinned physical component-CG, all-parameter, all-target, both-branch, and six-channel gates pass. A surviving functional kernel is provisional. This bounded F_X screen omits J/W and the generic momentum tower, so it never establishes full F A G_p.".to_string(),
    })
}

fn checkpoint_hash(report: &SecondMomentumFxFunctionalReport) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&CheckpointHashPayload {
                schema_version: SECOND_MOMENTUM_FX_CHECKPOINT_SCHEMA,
                report,
            })
            .expect("serialize second-momentum F_X checkpoint payload")
        )
    )
}

pub fn second_momentum_fx_checkpoint(
    report: SecondMomentumFxFunctionalReport,
) -> Result<SecondMomentumFxCheckpoint, String> {
    validate_second_momentum_fx_report(&report)?;
    let checkpoint_sha256 = checkpoint_hash(&report);
    Ok(SecondMomentumFxCheckpoint {
        schema_version: SECOND_MOMENTUM_FX_CHECKPOINT_SCHEMA.to_string(),
        report,
        checkpoint_sha256,
    })
}

pub fn validate_second_momentum_fx_report(
    report: &SecondMomentumFxFunctionalReport,
) -> Result<(), String> {
    validate_provenance(&report.provenance)?;
    let rank_summaries = report
        .channel_ranks
        .iter()
        .flat_map(|channel| [&channel.x2, &channel.x5, &channel.joint])
        .chain(
            report
                .branch_ranks
                .iter()
                .flat_map(|branch| [&branch.x2, &branch.x5, &branch.joint]),
        )
        .chain([&report.global_x2, &report.global_x5, &report.global_joint]);
    if rank_summaries.into_iter().any(|summary| {
        summary.variable_count != SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
            || summary.rank + summary.nullity != SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
    }) {
        return Err("second-momentum F_X exact-rank summary is inconsistent".to_string());
    }
    let observed_channels = report
        .channel_ranks
        .iter()
        .filter(|channel| channel.observed_terms > 0)
        .map(|channel| channel.gauge_form_degree)
        .collect::<Vec<_>>();
    let observed_all_six = observed_channels == [0, 1, 2, 3, 4, 5];
    let observed_both_branches = report
        .branch_ranks
        .iter()
        .all(|branch| branch.observed_terms > 0);
    let expected_component_cg_gate = report.provenance.source_kind
        == SecondMomentumFxSourceKind::PhysicalRecoupledStream
        && report.coverage.all_77_component_cg_maps_complete
        && observed_both_branches
        && observed_all_six
        && report.stream_hash_matches_expected;
    let expected_parameter_gate =
        expected_component_cg_gate && report.coverage.full_parameter_projection_complete;
    let expected_target_gate =
        expected_component_cg_gate && report.coverage.full_target_projection_complete;
    let expected_row_gate =
        report.global_joint.equation_count >= SECOND_MOMENTUM_FX_FUNCTIONAL_ROWS;
    let expected_rank_saturation = report.global_joint.rank
        == SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
        && report.global_joint.nullity == 0;
    let expected_declared_no_go = expected_component_cg_gate
        && report.coverage.p2_d13_wedge_branch_complete
        && report.coverage.p3_d11_contraction_branch_complete
        && report.coverage.all_six_gauge_channels_complete
        && report.coverage.complete_x2_projection
        && report.coverage.complete_x5_projection
        && expected_parameter_gate
        && expected_target_gate
        && expected_row_gate
        && expected_rank_saturation;
    if report.schema_version != SECOND_MOMENTUM_FX_REPORT_SCHEMA
        || report.formal_momentum_variables != 11
        || report.source_momentum_degree != 2
        || report.coefficient_variables != SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS
        || report.gauge_channels_expected != SECOND_MOMENTUM_FX_GAUGE_CHANNELS
        || report.functional_seeds.len() != SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len()
        || report.buckets_per_seed != SECOND_MOMENTUM_FX_BUCKETS_PER_SEED
        || report.functional_rows_per_observed_group != SECOND_MOMENTUM_FX_FUNCTIONAL_ROWS
        || report.channel_ranks.len() != SECOND_MOMENTUM_FX_GAUGE_CHANNELS
        || report.branch_ranks.len() != 2
        || report.observed_gauge_channels != observed_channels
        || report.stream_hash_matches_expected
            != (report.canonical_stream_sha256
                == report.provenance.expected_canonical_stream_sha256)
        || report.component_cg_gate_passed != expected_component_cg_gate
        || report.parameter_projection_gate_passed != expected_parameter_gate
        || report.target_projection_gate_passed != expected_target_gate
        || report.at_least_128_joint_functional_rows != expected_row_gate
        || report.rank_dimension_saturated != expected_rank_saturation
        || report.declared_slice_no_go_certified != expected_declared_no_go
        || !report.partial_fx_only
        || report.full_f_a_g_p_established
        || report.surviving_kernel_is_physical_certificate
    {
        return Err("second-momentum F_X report invariant failed".to_string());
    }
    Ok(())
}

pub fn validate_second_momentum_fx_checkpoint(
    checkpoint: &SecondMomentumFxCheckpoint,
) -> Result<(), String> {
    if checkpoint.schema_version != SECOND_MOMENTUM_FX_CHECKPOINT_SCHEMA {
        return Err("checkpoint is not a second-momentum F_X v1 checkpoint".to_string());
    }
    validate_second_momentum_fx_report(&checkpoint.report)?;
    if checkpoint.checkpoint_sha256 != checkpoint_hash(&checkpoint.report) {
        return Err("second-momentum F_X checkpoint SHA-256 mismatch".to_string());
    }
    Ok(())
}

pub fn write_second_momentum_fx_checkpoint(
    path: &Path,
    checkpoint: &SecondMomentumFxCheckpoint,
) -> io::Result<()> {
    validate_second_momentum_fx_checkpoint(checkpoint)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    {
        let file = File::create(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, checkpoint)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(temporary, path)
}

pub fn read_second_momentum_fx_checkpoint(path: &Path) -> io::Result<SecondMomentumFxCheckpoint> {
    let checkpoint: SecondMomentumFxCheckpoint =
        serde_json::from_reader(BufReader::new(File::open(path)?))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    validate_second_momentum_fx_checkpoint(&checkpoint)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn p3_static_data() -> &'static P3D11ExactStaticData {
        static DATA: OnceLock<P3D11ExactStaticData> = OnceLock::new();
        DATA.get_or_init(|| P3D11ExactStaticData::build().unwrap())
    }

    #[test]
    fn p3_d11_producer_has_golden_translation_normalization_and_degree() {
        let static_data = p3_static_data();
        let contracted = 0;
        let free_spinor = 15;
        let axis = 0;
        let source_mask = 0x0000_0fff;
        assert_eq!(static_data.translation_weights[0][15][0], (0, -2));
        assert_eq!(
            crate::eleven_dimensional_level16_couplings::right_contraction_sign(
                source_mask,
                contracted,
            ),
            Some(-1)
        );
        let source = crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm {
            momentum_pair: [2, 7],
            free_spinor: u8::try_from(free_spinor).unwrap(),
            exterior_mask: source_mask,
            coefficient: 7,
        };
        let output = produce_p3_d11_contractions(static_data, 4, 0, 0, &[source]).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().all(|term| {
            term.coefficient_column == 4
                && term.row.gauge_channel.form_degree() == 0
                && term.row.parameter_component == 0
                && term.row.source_momentum.total_degree() == 2
                && term.row.spinor_derivative_mask.count_ones() == 11
                && SecondMomentumGaugeBranch::P3D11Contraction {
                    momentum_axis: term.row.contraction_axis,
                }
                .output_momentum(term.row.source_momentum)
                .unwrap()
                .total_degree()
                    == 3
        }));
        let row = P3D11ContractionRow {
            gauge_channel: SecondMomentumGaugeChannel::new(0).unwrap(),
            parameter_component: 0,
            source_momentum: DegreeTwoMomentumMonomial::from_pair(2, 7).unwrap(),
            contraction_axis: u8::try_from(axis).unwrap(),
            spinor_derivative_mask: source_mask ^ (1_u32 << contracted),
        };
        let actual = &output
            .iter()
            .find(|term| term.row == row)
            .unwrap()
            .coefficient;
        assert_eq!(
            actual,
            &ExactGaussian {
                real: Ratio::from_integer(BigInt::from(0)),
                imaginary: Ratio::from_integer(BigInt::from(14)),
            }
        );
    }

    #[test]
    fn p3_d11_producer_keeps_contraction_axis_in_the_row_and_rejects_bad_input() {
        let static_data = p3_static_data();
        let source_mask = 0x0000_0fff;
        let base = crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm {
            momentum_pair: [1, 3],
            free_spinor: 15,
            exterior_mask: source_mask,
            coefficient: 1,
        };
        let mut second = base;
        second.momentum_pair = [1, 4];
        let output = produce_p3_d11_contractions(static_data, 0, 0, 0, &[base, second]).unwrap();
        assert!(output.iter().any(|term| {
            term.row.source_momentum == DegreeTwoMomentumMonomial::from_pair(1, 3).unwrap()
                && term.row.contraction_axis == 0
                && term.row.spinor_derivative_mask == 0x0000_0ffe
        }));
        assert!(output.iter().any(|term| {
            term.row.source_momentum == DegreeTwoMomentumMonomial::from_pair(1, 4).unwrap()
                && term.row.contraction_axis == 0
                && term.row.spinor_derivative_mask == 0x0000_0ffe
        }));
        assert!(produce_p3_d11_contractions(static_data, 77, 0, 0, &[base]).is_err());
        assert!(produce_p3_d11_contractions(static_data, 0, 6, 0, &[base]).is_err());
        assert!(produce_p3_d11_contractions(static_data, 0, 0, 1, &[base]).is_err());
        let mut bad_mask = base;
        bad_mask.exterior_mask &= bad_mask.exterior_mask - 1;
        assert!(produce_p3_d11_contractions(static_data, 0, 0, 0, &[bad_mask]).is_err());
        let mut overflowing = base;
        overflowing.coefficient = i128::MIN;
        assert!(produce_p3_d11_contractions(static_data, 0, 0, 0, &[overflowing]).is_err());
        assert!(multiply_gaussian_integer_pairs(i128::MAX, 0, 2, 0).is_err());
    }

    #[test]
    fn p3_d11_nontrivial_gauge_matches_direct_matrix_orientation_and_complex_signs() {
        let static_data = p3_static_data();
        let source = crate::eleven_dimensional_second_momentum_gpu::RecoupledSourceTerm {
            momentum_pair: [3, 8],
            free_spinor: 15,
            exterior_mask: 0x0000_0fff,
            coefficient: 5,
        };
        let actual = produce_p3_d11_contractions(static_data, 6, 1, 0, &[source]).unwrap();
        let actual = actual
            .into_iter()
            .map(|term| (term.row, term.coefficient))
            .collect::<BTreeMap<_, _>>();

        let (_, _, gauge) = crate::eleven_dimensional_clifford::gauge_form_operator_basis()
            .into_iter()
            .find(|(degree, _, _)| *degree == 1)
            .unwrap();
        let mut expected = BTreeMap::<P3D11ContractionRow, ExactGaussian>::new();
        for derivative_spinor in 0..32 {
            let gauge_value = &gauge[usize::from(source.free_spinor)][derivative_spinor];
            if *gauge_value.re.numer() == 0 && *gauge_value.im.numer() == 0 {
                continue;
            }
            let mut occupied = source.exterior_mask;
            while occupied != 0 {
                let contracted_spinor = occupied.trailing_zeros() as usize;
                occupied &= occupied - 1;
                let sign = crate::eleven_dimensional_level16_couplings::right_contraction_sign(
                    source.exterior_mask,
                    contracted_spinor,
                )
                .unwrap();
                for contraction_axis in 0..11 {
                    let (translation_real, translation_imaginary) = static_data.translation_weights
                        [contracted_spinor][derivative_spinor][contraction_axis];
                    if translation_real == 0 && translation_imaginary == 0 {
                        continue;
                    }
                    let (real, imaginary) = multiply_gaussian_integer_pairs(
                        i128::from(*gauge_value.re.numer()),
                        i128::from(*gauge_value.im.numer()),
                        i128::from(translation_real),
                        i128::from(translation_imaginary),
                    )
                    .unwrap();
                    let scale = source.coefficient * sign;
                    let row = P3D11ContractionRow {
                        gauge_channel: SecondMomentumGaugeChannel::new(1).unwrap(),
                        parameter_component: 0,
                        source_momentum: DegreeTwoMomentumMonomial::from_pair(3, 8).unwrap(),
                        contraction_axis: contraction_axis as u8,
                        spinor_derivative_mask: source.exterior_mask ^ (1_u32 << contracted_spinor),
                    };
                    let value = expected
                        .entry(row.clone())
                        .or_insert_with(ExactGaussian::zero);
                    value.real += Ratio::from_integer(BigInt::from(scale * real));
                    value.imaginary += Ratio::from_integer(BigInt::from(scale * imaginary));
                    if value.is_zero() {
                        expected.remove(&row);
                    }
                }
            }
        }
        assert!(!expected.is_empty());
        assert_eq!(actual, expected);
    }

    fn sample_p3_contraction(momentum_axis: u8) -> P3D11ContractionTerm {
        P3D11ContractionTerm {
            coefficient_column: 9,
            row: P3D11ContractionRow {
                gauge_channel: SecondMomentumGaugeChannel::new(0).unwrap(),
                parameter_component: 0,
                source_momentum: DegreeTwoMomentumMonomial::from_pair(2, 7).unwrap(),
                contraction_axis: momentum_axis,
                spinor_derivative_mask: 0x0000_0ffe,
            },
            coefficient: ExactGaussian {
                real: Ratio::from_integer(BigInt::from(3)),
                imaginary: Ratio::from_integer(BigInt::from(2)),
            },
        }
    }

    #[test]
    fn p3_d11_physical_fx_projection_matches_direct_exact_reference() {
        let contraction = sample_p3_contraction(4);
        let actual =
            project_p3_d11_to_physical_fx(p3_static_data(), &[contraction.clone()]).unwrap();
        assert!(!actual.is_empty());
        let mut actual_by_key = BTreeMap::new();
        for term in actual {
            assert_eq!(term.coefficient_column, contraction.coefficient_column);
            assert_eq!(term.gauge_channel, contraction.row.gauge_channel);
            assert_eq!(
                term.parameter_component,
                contraction.row.parameter_component
            );
            assert_eq!(term.source_momentum, contraction.row.source_momentum);
            assert_eq!(term.spinor_derivative_mask.count_ones(), 11);
            assert_eq!(
                term.gauge_branch,
                SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 4 }
            );
            assert!(
                actual_by_key
                    .insert(
                        (
                            term.target_coordinate,
                            term.spinor_derivative_mask,
                            term.sector
                        ),
                        term.coefficient,
                    )
                    .is_none()
            );
        }

        let highest_target =
            crate::eleven_dimensional_bridge::vector_spinor_target_dual_basis_states()
                .into_iter()
                .find(|state| state.pbw_word_simple_roots.is_empty())
                .unwrap();
        let mut expected = BTreeMap::<(usize, u32, SecondMomentumFxSector), ExactGaussian>::new();
        for target in highest_target.raw_terms {
            let target_coefficient = ExactGaussian {
                real: Ratio::new(
                    BigInt::from(target.numerator),
                    BigInt::from(target.denominator),
                ),
                imaginary: Ratio::from_integer(BigInt::from(0)),
            };
            let mut templates = Vec::new();
            crate::eleven_dimensional_physical_curvature::visit_exact_fx_derivative_templates(
                target.vector_weight_index,
                target.spinor_weight_index,
                |entry| templates.push(entry),
            )
            .unwrap();
            for template in templates {
                let Some((degree_twelve_mask, sign)) =
                    crate::eleven_dimensional_level16_couplings::right_wedge_normal_order(
                        contraction.row.spinor_derivative_mask,
                        template.derivative_spinor_weight_index,
                    )
                else {
                    continue;
                };
                let functional_mask =
                    degree_twelve_to_degree_eleven_functional_mask(degree_twelve_mask).unwrap();
                let sector = if template.x_two_sector {
                    SecondMomentumFxSector::X2
                } else {
                    SecondMomentumFxSector::X5
                };
                let target_template =
                    multiply_exact_gaussians(&target_coefficient, &template.coefficient);
                let mut value =
                    multiply_exact_gaussians(&contraction.coefficient, &target_template);
                if sign < 0 {
                    value.real = -value.real;
                    value.imaginary = -value.imaginary;
                }
                let key = (template.output_coordinate, functional_mask, sector);
                let entry = expected.entry(key).or_insert_with(ExactGaussian::zero);
                entry.real += value.real;
                entry.imaginary += value.imaginary;
                if entry.is_zero() {
                    expected.remove(&key);
                }
            }
        }
        assert_eq!(actual_by_key, expected);
        assert!(
            actual_by_key
                .keys()
                .any(|(_, _, sector)| *sector == SecondMomentumFxSector::X2)
        );
        assert!(
            actual_by_key
                .keys()
                .any(|(_, _, sector)| *sector == SecondMomentumFxSector::X5)
        );
    }

    #[test]
    fn p3_d11_physical_fx_projection_detects_axis_sign_and_degree_mutations() {
        let static_data = p3_static_data();
        let baseline = sample_p3_contraction(4);
        let baseline_output =
            project_p3_d11_to_physical_fx(static_data, &[baseline.clone()]).unwrap();

        let axis_mutation = sample_p3_contraction(5);
        let axis_output = project_p3_d11_to_physical_fx(static_data, &[axis_mutation]).unwrap();
        let without_branch = |terms: &[SecondMomentumFxColumnTerm]| {
            terms
                .iter()
                .map(|term| {
                    (
                        term.target_coordinate,
                        term.spinor_derivative_mask,
                        term.sector,
                        term.coefficient.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            without_branch(&baseline_output),
            without_branch(&axis_output)
        );
        assert!(baseline_output.iter().all(|term| {
            term.gauge_branch == SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 4 }
        }));
        assert!(axis_output.iter().all(|term| {
            term.gauge_branch == SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 5 }
        }));
        let first = &baseline_output[0];
        assert_ne!(
            second_momentum_fx_functional_assignments(
                first.gauge_channel,
                SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 4 },
                first.source_momentum,
                first.parameter_component,
                first.target_coordinate,
                first.spinor_derivative_mask,
                first.sector,
            ),
            second_momentum_fx_functional_assignments(
                first.gauge_channel,
                SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 5 },
                first.source_momentum,
                first.parameter_component,
                first.target_coordinate,
                first.spinor_derivative_mask,
                first.sector,
            )
        );

        let mut sign_mutation = baseline.clone();
        sign_mutation.coefficient.real = -sign_mutation.coefficient.real;
        sign_mutation.coefficient.imaginary = -sign_mutation.coefficient.imaginary;
        let sign_output = project_p3_d11_to_physical_fx(static_data, &[sign_mutation]).unwrap();
        assert_eq!(baseline_output.len(), sign_output.len());
        for (positive, negative) in baseline_output.iter().zip(&sign_output) {
            assert_eq!(
                positive.coefficient.real,
                -negative.coefficient.real.clone()
            );
            assert_eq!(
                positive.coefficient.imaginary,
                -negative.coefficient.imaginary.clone()
            );
        }

        let mut degree_mutation = baseline;
        degree_mutation.row.spinor_derivative_mask &=
            degree_mutation.row.spinor_derivative_mask - 1;
        assert!(project_p3_d11_to_physical_fx(static_data, &[degree_mutation]).is_err());
    }

    #[test]
    fn p3_d11_projection_axis_is_part_of_combined_accumulator_key() {
        let positive = sample_p3_contraction(4);
        let mut negative_other_axis = sample_p3_contraction(5);
        negative_other_axis.coefficient.real = -negative_other_axis.coefficient.real;
        negative_other_axis.coefficient.imaginary = -negative_other_axis.coefficient.imaginary;
        let positive_count = project_p3_d11_to_physical_fx(p3_static_data(), &[positive.clone()])
            .unwrap()
            .len();
        let negative_count =
            project_p3_d11_to_physical_fx(p3_static_data(), &[negative_other_axis.clone()])
                .unwrap()
                .len();
        let combined =
            project_p3_d11_to_physical_fx(p3_static_data(), &[positive, negative_other_axis])
                .unwrap();
        assert_eq!(combined.len(), positive_count + negative_count);
        assert!(combined.iter().any(|term| {
            term.gauge_branch == SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 4 }
        }));
        assert!(combined.iter().any(|term| {
            term.gauge_branch == SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 5 }
        }));
    }

    #[test]
    fn p3_d11_functional_identity_covers_every_physical_row_field() {
        fn assignments(
            term: &SecondMomentumFxColumnTerm,
        ) -> [(usize, i8); SECOND_MOMENTUM_FX_FUNCTIONAL_SEEDS.len()] {
            second_momentum_fx_functional_assignments(
                term.gauge_channel,
                term.gauge_branch,
                term.source_momentum,
                term.parameter_component,
                term.target_coordinate,
                term.spinor_derivative_mask,
                term.sector,
            )
        }

        let baseline = project_p3_d11_to_physical_fx(p3_static_data(), &[sample_p3_contraction(4)])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let expected = assignments(&baseline);
        let mut mutations = Vec::new();
        let mut value = baseline.clone();
        value.gauge_channel = SecondMomentumGaugeChannel::new(1).unwrap();
        mutations.push(value);
        let mut value = baseline.clone();
        value.gauge_branch = SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 5 };
        mutations.push(value);
        let mut value = baseline.clone();
        value.source_momentum = DegreeTwoMomentumMonomial::from_pair(2, 8).unwrap();
        mutations.push(value);
        let mut value = baseline.clone();
        value.parameter_component += 1;
        mutations.push(value);
        let mut value = baseline.clone();
        value.target_coordinate += 1;
        mutations.push(value);
        let mut value = baseline.clone();
        let removed = value.spinor_derivative_mask.trailing_zeros();
        let replacement = (0..32)
            .find(|spinor| value.spinor_derivative_mask & (1_u32 << spinor) == 0)
            .unwrap();
        value.spinor_derivative_mask ^= 1_u32 << removed;
        value.spinor_derivative_mask |= 1_u32 << replacement;
        mutations.push(value);
        let mut value = baseline.clone();
        value.sector = match value.sector {
            SecondMomentumFxSector::X2 => SecondMomentumFxSector::X5,
            SecondMomentumFxSector::X5 => SecondMomentumFxSector::X2,
        };
        mutations.push(value);
        for mutation in mutations {
            assert_ne!(assignments(&mutation), expected);
        }

        let mut column_only = baseline;
        column_only.coefficient_column += 1;
        assert_eq!(assignments(&column_only), expected);
    }

    #[derive(Clone)]
    struct SyntheticSource {
        terms: Vec<SecondMomentumFxColumnTerm>,
        provenance: SecondMomentumFxProvenance,
        coverage: SecondMomentumFxCoverage,
    }

    impl SecondMomentumFxTermSource for SyntheticSource {
        fn provenance(&self) -> SecondMomentumFxProvenance {
            self.provenance.clone()
        }

        fn coverage(&self) -> SecondMomentumFxCoverage {
            self.coverage.clone()
        }

        fn visit_terms(
            &self,
            visitor: &mut dyn FnMut(SecondMomentumFxColumnTerm),
        ) -> Result<(), String> {
            for term in &self.terms {
                visitor(term.clone());
            }
            Ok(())
        }
    }

    fn mask(degree: usize, offset: usize) -> u32 {
        (0..degree).fold(0_u32, |value, bit| value | (1_u32 << ((bit + offset) % 32)))
    }

    fn synthetic_terms() -> Vec<SecondMomentumFxColumnTerm> {
        let pairs = (0..11)
            .flat_map(|left| (left..11).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let mut terms = Vec::new();
        for column in 0..SECOND_MOMENTUM_FX_COEFFICIENT_COLUMNS {
            for sector in [SecondMomentumFxSector::X2, SecondMomentumFxSector::X5] {
                let branch = if column % 2 == 0 {
                    SecondMomentumGaugeBranch::P2D13Wedge
                } else {
                    SecondMomentumGaugeBranch::P3D11Contraction {
                        momentum_axis: (column % 11) as u8,
                    }
                };
                terms.push(SecondMomentumFxColumnTerm {
                    coefficient_column: column,
                    gauge_channel: SecondMomentumGaugeChannel::new(column % 6).unwrap(),
                    gauge_branch: branch,
                    source_momentum: DegreeTwoMomentumMonomial::from_pair(
                        pairs[column % pairs.len()].0,
                        pairs[column % pairs.len()].1,
                    )
                    .unwrap(),
                    parameter_component: column * 17
                        + usize::from(sector == SecondMomentumFxSector::X5),
                    target_coordinate: column * 31
                        + usize::from(sector == SecondMomentumFxSector::X5),
                    spinor_derivative_mask: mask(branch.derivative_order(), column),
                    sector,
                    coefficient: ExactGaussian::from_integer((column % 7 + 1) as i64),
                });
            }
        }
        terms
    }

    fn synthetic_source(terms: Vec<SecondMomentumFxColumnTerm>) -> SyntheticSource {
        let stream = canonical_second_momentum_fx_stream_sha256(&terms);
        SyntheticSource {
            terms,
            provenance: SecondMomentumFxProvenance {
                source_kind: SecondMomentumFxSourceKind::SyntheticControl,
                campaign_id: "synthetic-p2-fx-rank-control-v1".to_string(),
                representation_inventory_sha256: SECOND_MOMENTUM_REPRESENTATION_INVENTORY_SHA256
                    .to_string(),
                level12_fixture_manifest_sha256: "11".repeat(32),
                component_cg_manifest_sha256: "22".repeat(32),
                coefficient_layout_sha256: second_momentum_fx_coefficient_layout_sha256(),
                expected_canonical_stream_sha256: stream,
            },
            coverage: SecondMomentumFxCoverage {
                all_77_component_cg_maps_complete: true,
                p2_d13_wedge_branch_complete: true,
                p3_d11_contraction_branch_complete: true,
                all_six_gauge_channels_complete: true,
                full_parameter_projection_complete: true,
                full_target_projection_complete: true,
                complete_x2_projection: true,
                complete_x5_projection: true,
                j_and_w_sectors_complete: false,
                generic_momentum_tower_complete_or_proved_sufficient: false,
            },
        }
    }

    fn legacy_comparator_stream_sha256(terms: &[SecondMomentumFxColumnTerm]) -> String {
        let mut ordered = terms.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|term| canonical_term_sort_key(term));
        let canonical = ordered.into_iter().map(canonical_term).collect::<Vec<_>>();
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&canonical).unwrap())
        )
    }

    #[test]
    fn cached_canonical_sort_matches_legacy_comparator_semantics() {
        let original = synthetic_terms();
        let mut shuffled = original.clone();
        shuffled.reverse();
        shuffled.rotate_left(37);
        let mut mutated = shuffled.clone();
        mutated[11].coefficient.real += Ratio::from_integer(BigInt::from(1));

        for terms in [&original, &shuffled, &mutated] {
            assert_eq!(
                canonical_second_momentum_fx_stream_sha256(terms),
                legacy_comparator_stream_sha256(terms)
            );
        }
        assert_ne!(
            canonical_second_momentum_fx_stream_sha256(&shuffled),
            canonical_second_momentum_fx_stream_sha256(&mutated)
        );
    }

    #[test]
    fn degree_two_momentum_is_explicit_and_contraction_is_degree_three() {
        assert!(DegreeTwoMomentumMonomial::new([0; 11]).is_err());
        let p2 = DegreeTwoMomentumMonomial::from_pair(3, 3).unwrap();
        assert_eq!(p2.exponents[3], 2);
        let p3 = SecondMomentumGaugeBranch::P3D11Contraction { momentum_axis: 7 }
            .output_momentum(p2)
            .unwrap();
        assert_eq!(p3.total_degree(), 3);
        assert_eq!(p3.exponents[3], 2);
        assert_eq!(p3.exponents[7], 1);
    }

    #[test]
    fn synthetic_control_saturates_all_77_columns_but_cannot_claim_physics() {
        let report = evaluate_second_momentum_fx_source(&synthetic_source(synthetic_terms()))
            .expect("evaluate exact synthetic p2 F_X control");
        assert!(report.stream_hash_matches_expected);
        assert_eq!(report.observed_gauge_channels, vec![0, 1, 2, 3, 4, 5]);
        assert!(report.at_least_128_joint_functional_rows);
        assert_eq!(report.global_x2.rank, 77);
        assert_eq!(report.global_x5.rank, 77);
        assert_eq!(report.global_joint.rank, 77);
        assert_eq!(report.global_joint.nullity, 0);
        assert!(report.rank_dimension_saturated);
        assert!(!report.component_cg_gate_passed);
        assert!(!report.declared_slice_no_go_certified);
        assert!(!report.surviving_kernel_is_physical_certificate);
        assert!(!report.full_f_a_g_p_established);
    }

    #[test]
    fn streaming_sparse_rows_match_retained_synthetic_ranks_and_digest() {
        let terms = synthetic_terms();
        let source = synthetic_source(terms.clone());
        let retained = evaluate_second_momentum_fx_source(&source).unwrap();
        let expected_functional_digest = second_momentum_fx_functional_rows_sha256(&terms).unwrap();
        let mut streaming = SecondMomentumFxStreamingAccumulator::new(
            source.provenance.clone(),
            source.coverage.clone(),
        )
        .unwrap();
        for term in terms {
            streaming.push(term).unwrap();
        }
        let checkpoint = streaming.finalize().unwrap();
        validate_second_momentum_fx_streaming_checkpoint(&checkpoint).unwrap();
        assert_eq!(
            checkpoint.report.canonical_functional_rows_sha256,
            expected_functional_digest
        );
        assert_eq!(checkpoint.report.global_x2.rank, retained.global_x2.rank);
        assert_eq!(checkpoint.report.global_x5.rank, retained.global_x5.rank);
        assert_eq!(
            checkpoint.report.global_joint.rank,
            retained.global_joint.rank
        );
        assert_eq!(checkpoint.report.global_joint.nullity, 0);
        assert_eq!(
            checkpoint.report.observed_nonzero_columns,
            (0..77).collect::<Vec<_>>()
        );
    }

    #[test]
    fn streaming_functional_digest_detects_exact_mutation() {
        let terms = synthetic_terms();
        let original = second_momentum_fx_functional_rows_sha256(&terms).unwrap();
        let mut mutated = terms;
        mutated[0].coefficient.real += Ratio::from_integer(BigInt::from(1));
        assert_ne!(
            original,
            second_momentum_fx_functional_rows_sha256(&mutated).unwrap()
        );
    }

    #[test]
    fn modular_full_rank_fast_path_and_exact_fallback_agree() {
        let channel = SecondMomentumGaugeChannel::new(0).unwrap();
        let group = FunctionalGroupKey {
            gauge_channel: channel,
            gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
            source_momentum: DegreeTwoMomentumMonomial::from_pair(0, 0).unwrap(),
            sector: SecondMomentumFxSector::X2,
        };
        let row_key = |bucket| FunctionalRowKey {
            group: group.clone(),
            seed_ordinal: 0,
            bucket,
        };
        let mut full = SparseFunctionalRows::new();
        full.insert(
            row_key(0),
            BTreeMap::from([(0, ExactGaussian::from_integer(1))]),
        );
        full.insert(
            row_key(1),
            BTreeMap::from([(1, ExactGaussian::from_integer(1))]),
        );
        assert_eq!(solve_sparse_rows(&full, None, None, None).rank, 2);

        let mut deficient = SparseFunctionalRows::new();
        deficient.insert(
            row_key(0),
            BTreeMap::from([
                (0, ExactGaussian::from_integer(1)),
                (1, ExactGaussian::from_integer(1)),
            ]),
        );
        deficient.insert(
            row_key(1),
            BTreeMap::from([
                (0, ExactGaussian::from_integer(2)),
                (1, ExactGaussian::from_integer(2)),
            ]),
        );
        assert_eq!(solve_sparse_rows(&deficient, None, None, None).rank, 1);

        let noninvertible_mod_prime = ExactGaussian {
            real: Ratio::new(BigInt::from(1), BigInt::from(RANK_PRIME)),
            imaginary: Ratio::from_integer(BigInt::from(0)),
        };
        let singular_modular_image = SparseFunctionalRows::from([(
            row_key(0),
            BTreeMap::from([(0, noninvertible_mod_prime)]),
        )]);
        assert_eq!(
            solve_sparse_rows(&singular_modular_image, None, None, None).rank,
            1
        );
    }

    #[test]
    fn projected_column_validation_is_transactional() {
        let source = synthetic_source(synthetic_terms());
        let mut accumulator = SecondMomentumFxStreamingAccumulator::new(
            source.provenance.clone(),
            source.coverage.clone(),
        )
        .unwrap();
        let monomial = DegreeTwoMomentumMonomial::from_pair(0, 0).unwrap();
        let valid_entry = SecondMomentumFxProjectedEntry {
            source_momentum: monomial,
            sector: SecondMomentumFxSector::X2,
            seed_ordinal: 0,
            bucket: 0,
            coefficient: ExactGaussian::from_integer(1),
        };
        let invalid = SecondMomentumFxProjectedColumn {
            coefficient_column: 0,
            gauge_channel: SecondMomentumGaugeChannel::new(0).unwrap(),
            gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
            observed_terms: 2,
            observed_groups: vec![(monomial, SecondMomentumFxSector::X2)],
            entries: vec![valid_entry.clone(), valid_entry.clone()],
        };
        assert!(accumulator.push_projected_column(invalid).is_err());
        assert_eq!(accumulator.observed_terms, 0);
        assert!(accumulator.rows.is_empty());
        assert!(accumulator.observed_columns.is_empty());

        let valid = SecondMomentumFxProjectedColumn {
            coefficient_column: 0,
            gauge_channel: SecondMomentumGaugeChannel::new(0).unwrap(),
            gauge_branch: SecondMomentumGaugeBranch::P2D13Wedge,
            observed_terms: 1,
            observed_groups: vec![(monomial, SecondMomentumFxSector::X2)],
            entries: vec![valid_entry],
        };
        accumulator.push_projected_column(valid.clone()).unwrap();
        let rows = accumulator.rows.clone();
        assert!(accumulator.push_projected_column(valid).is_err());
        assert_eq!(accumulator.observed_terms, 1);
        assert_eq!(accumulator.rows, rows);
        assert_eq!(accumulator.observed_columns, BTreeSet::from([0]));
    }

    #[test]
    fn coefficient_mutation_is_detected_by_pinned_stream_hash() {
        let original = synthetic_source(synthetic_terms());
        let mut mutated = original.clone();
        mutated.terms[0].coefficient = ExactGaussian::from_integer(31337);
        let report = evaluate_second_momentum_fx_source(&mutated).unwrap();
        assert!(!report.stream_hash_matches_expected);
        assert!(!report.component_cg_gate_passed);
        assert!(!report.declared_slice_no_go_certified);
    }

    #[test]
    fn checkpoint_rejects_first_momentum_v4_schema_and_mutation() {
        let report =
            evaluate_second_momentum_fx_source(&synthetic_source(synthetic_terms())).unwrap();
        let checkpoint = second_momentum_fx_checkpoint(report).unwrap();
        validate_second_momentum_fx_checkpoint(&checkpoint).unwrap();

        let mut wrong_schema = checkpoint.clone();
        wrong_schema.schema_version =
            "adynkra-11d-first-momentum-partial-fx-checkpoint-v4".to_string();
        assert!(validate_second_momentum_fx_checkpoint(&wrong_schema).is_err());

        let mut mutated = checkpoint;
        mutated.report.global_joint.rank -= 1;
        assert!(validate_second_momentum_fx_checkpoint(&mutated).is_err());
    }

    #[test]
    fn canonical_stream_hash_is_order_independent_and_monomial_sensitive() {
        let terms = synthetic_terms();
        let original = canonical_second_momentum_fx_stream_sha256(&terms);
        let mut reversed = terms.clone();
        reversed.reverse();
        assert_eq!(
            original,
            canonical_second_momentum_fx_stream_sha256(&reversed)
        );

        reversed[0].source_momentum = DegreeTwoMomentumMonomial::from_pair(9, 10).unwrap();
        assert_ne!(
            original,
            canonical_second_momentum_fx_stream_sha256(&reversed)
        );
    }
}
