//! Independent first-column oracle for the source-variance-corrected
//! compensator, Eq. (25), gravitino-curl, and teleparallel chain.

use std::collections::{BTreeMap, BTreeSet};

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_h_hat_jet::{
    LinearizedFrameJetSector, LinearizedFrameSuperfields, canonical_gamma_traceless_frame_basis,
    canonical_physical_frame_representative, visit_linearized_frame_jet,
};
use crate::eleven_dimensional_majorana::{real_charge_conjugation, real_gamma_matrices};
use crate::eleven_dimensional_physical_curvature::{
    D_F_FOUR_FORM_DIMENSION, DDH_DIMENSION, DELTA_DIMENSION, DH_DIMENSION, Eq25FermionicFrameInput,
    ExactQi, GRAVITINO_CURL_DIMENSION, SPINOR_DIMENSION, VECTOR_DIMENSION, W_FOUR_FORM_DIMENSION,
    apply_eq25_fermionic_frame, apply_eq29_fermionic_anholonomy,
    cached_linearized_gravitino_curl_to_d_f_four_operator, gamma_dh_operator,
    inject_d_holonomy_form_into_d_delta, inject_holonomy_form_into_delta,
    solve_conventional_compensators, solve_higher_jet_conventional_compensators,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, FormalMomentumMonomial, OrderedSuperderivativeMonomial,
    left_multiply_d,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

const H_HAT_DIMENSION: usize = 320;
const THREE_FORM_DIMENSION: usize = 165;

type PolynomialMap = BTreeMap<usize, CanonicalSuperPolynomial>;
type MonomialSlices = BTreeMap<OrderedSuperderivativeMonomial, BTreeMap<usize, ExactQi>>;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct FullChainRowKey {
    pub output_coordinate: usize,
    pub exterior_spinor_mask: u32,
    pub momentum_exponents: [u16; VECTOR_DIMENSION],
}

#[derive(Clone, Debug)]
pub(crate) struct CorrectedFullChainStageStreams {
    pub d_delta: BTreeMap<FullChainRowKey, ExactQi>,
    pub eq25_frame: BTreeMap<FullChainRowKey, ExactQi>,
    pub gravitino_curl: BTreeMap<FullChainRowKey, ExactQi>,
    pub teleparallel_dg4: BTreeMap<FullChainRowKey, ExactQi>,
}

/// Exact section values together with the horizontal transformation contract.
///
/// The gauge-fixed section has `Psi_[2]=0`, so horizontalization does not add
/// a source-column-dependent value to either stream. It changes the Lorentz
/// transformation law by the unique generator-dependent `Psi_[2]`
/// compensator. Consumers must bind the separate horizontal descent
/// certificate and must not claim that the unchanged section target has zero
/// ordinary commutator.
#[derive(Clone, Debug)]
pub(crate) struct HorizontalCorrectedFullChainStreams {
    pub candidate: BTreeMap<FullChainRowKey, ExactQi>,
    pub section_target: BTreeMap<FullChainRowKey, ExactQi>,
    pub section_psi_two_is_zero: bool,
    pub section_values_unchanged_by_horizontalization: bool,
    pub transformation_law: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OracleExactQi {
    pub real: [i64; 2],
    pub imaginary: [i64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FullChainWitness {
    pub row: FullChainRowKey,
    pub candidate: OracleExactQi,
    pub teleparallel: OracleExactQi,
    pub residual: OracleExactQi,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedFullChainColumnReport {
    pub schema_version: &'static str,
    pub source_ordinal: usize,
    pub source_variance_adapter: &'static str,
    pub candidate_normalization: &'static str,
    pub row_key: &'static str,
    pub corrected_psi_nonzero_terms: [usize; 4],
    pub corrected_d_psi_nonzero_terms: [usize; 4],
    pub delta_nonzero_rows: usize,
    pub d_delta_nonzero_rows: usize,
    pub delta_derivative_residual_rows: usize,
    pub corrected_eq25_frame_rows: usize,
    pub corrected_curl_rows: usize,
    pub eq29_curl_residual_rows: usize,
    pub candidate_rows: usize,
    pub corrected_teleparallel_rows: usize,
    pub common_rows: usize,
    pub candidate_only_rows: usize,
    pub teleparallel_only_rows: usize,
    pub exact_scale: Option<OracleExactQi>,
    pub exact_residual_rows: usize,
    pub first_mismatch: Option<FullChainWitness>,
    pub passed: bool,
    pub boundary: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorrectedGamma25ParityReport {
    pub schema_version: &'static str,
    pub columns_checked_per_degree: usize,
    pub p2_residual_rows: usize,
    pub p5_residual_rows: usize,
    pub p2_adapted_sha256: String,
    pub p2_direct_sha256: String,
    pub p5_adapted_sha256: String,
    pub p5_direct_sha256: String,
    pub passed: bool,
}

fn multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn divide(left: &ExactQi, right: &ExactQi) -> Option<ExactQi> {
    if right.is_zero() {
        return None;
    }
    let denominator =
        right.real.clone() * right.real.clone() + right.imaginary.clone() * right.imaginary.clone();
    Some(ExactQi {
        real: (left.real.clone() * right.real.clone()
            + left.imaginary.clone() * right.imaginary.clone())
            / denominator.clone(),
        imaginary: (left.imaginary.clone() * right.real.clone()
            - left.real.clone() * right.imaginary.clone())
            / denominator,
    })
}

fn public(value: &ExactQi) -> OracleExactQi {
    OracleExactQi {
        real: [*value.real.numer(), *value.real.denom()],
        imaginary: [*value.imaginary.numer(), *value.imaginary.denom()],
    }
}

fn public_coefficient(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    }
}

fn add_polynomial_value(
    output: &mut PolynomialMap,
    coordinate: usize,
    monomial: OrderedSuperderivativeMonomial,
    value: ExactQi,
) {
    if value.is_zero() {
        return;
    }
    let polynomial = output.entry(coordinate).or_default();
    polynomial.add_term(monomial, value);
    if polynomial.terms.is_empty() {
        output.remove(&coordinate);
    }
}

fn add_row_value(
    output: &mut BTreeMap<FullChainRowKey, ExactQi>,
    key: FullChainRowKey,
    value: ExactQi,
) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(key.clone()).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&key);
    }
}

fn transpose(input: &PolynomialMap) -> MonomialSlices {
    let mut output = MonomialSlices::new();
    for (&coordinate, polynomial) in input {
        for (monomial, value) in &polynomial.terms {
            let slice = output.entry(monomial.clone()).or_default();
            let entry = slice.entry(coordinate).or_insert_with(ExactQi::zero);
            entry.add_assign(value);
            if entry.is_zero() {
                slice.remove(&coordinate);
            }
        }
    }
    output.retain(|_, slice| !slice.is_empty());
    output
}

fn apply_sliced(
    input: &PolynomialMap,
    mut apply: impl FnMut(&BTreeMap<usize, ExactQi>) -> BTreeMap<usize, ExactQi>,
) -> PolynomialMap {
    let mut output = PolynomialMap::new();
    for (monomial, slice) in transpose(input) {
        for (coordinate, value) in apply(&slice) {
            add_polynomial_value(&mut output, coordinate, monomial.clone(), value);
        }
    }
    output
}

fn merge(output: &mut PolynomialMap, input: &PolynomialMap) {
    for (&coordinate, polynomial) in input {
        for (monomial, value) in &polynomial.terms {
            add_polynomial_value(output, coordinate, monomial.clone(), value.clone());
        }
    }
}

fn basis_input(ordinal: usize) -> Result<LinearizedFrameSuperfields, String> {
    let basis = canonical_gamma_traceless_frame_basis();
    let vector = basis
        .get(ordinal)
        .ok_or_else(|| format!("H_hat source ordinal {ordinal} outside 0..{}", basis.len()))?;
    Ok(LinearizedFrameSuperfields {
        h: vector
            .iter()
            .map(|(&coordinate, value)| {
                (coordinate, CanonicalSuperPolynomial::scalar(value.clone()))
            })
            .collect(),
        scale: CanonicalSuperPolynomial::default(),
        lorentz_two_form: BTreeMap::new(),
    })
}

fn adapt_h_slot(input: &BTreeMap<usize, ExactQi>, second_jet: bool) -> BTreeMap<usize, ExactQi> {
    let charge = real_charge_conjugation();
    let mut output = BTreeMap::new();
    for (&index, value) in input {
        let vector = index % VECTOR_DIMENSION;
        let rest = index / VECTOR_DIMENSION;
        let source_h_spinor = rest % SPINOR_DIMENSION;
        let derivative_prefix = rest / SPINOR_DIMENSION;
        if second_jet {
            assert!(index < DDH_DIMENSION);
        } else {
            assert!(index < DH_DIMENSION);
            assert!(derivative_prefix < SPINOR_DIMENSION);
        }
        for (adapted_h_spinor, row) in charge.iter().enumerate() {
            let factor = row[source_h_spinor];
            if factor == 0 {
                continue;
            }
            let adapted_index = (derivative_prefix * SPINOR_DIMENSION + adapted_h_spinor)
                * VECTOR_DIMENSION
                + vector;
            let entry = output.entry(adapted_index).or_insert_with(ExactQi::zero);
            entry.add_assign(&value.scaled(&Ratio::from_integer(i64::from(factor))));
            if entry.is_zero() {
                output.remove(&adapted_index);
            }
        }
    }
    output
}

type IntegerMatrix = Vec<Vec<i16>>;

fn multiply_integer(left: &IntegerMatrix, right: &[Vec<i8>]) -> IntegerMatrix {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for pivot in 0..SPINOR_DIMENSION {
            let factor = left[row][pivot];
            if factor == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += factor * i16::from(right[pivot][column]);
            }
        }
    }
    output
}

fn raised_gamma_product(indices: &[usize]) -> IntegerMatrix {
    let gammas = real_gamma_matrices();
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for (index, row) in output.iter_mut().enumerate() {
        row[index] = 1;
    }
    for &axis in indices {
        output = multiply_integer(&output, &gammas[axis]);
        if axis == 0 {
            for row in &mut output {
                for value in row {
                    *value = -*value;
                }
            }
        }
    }
    output
}

fn masks_of_degree(degree: usize) -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() as usize == degree)
        .collect()
}

fn original_dh_column(
    derivative: usize,
    basis_column: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    basis_column
        .iter()
        .map(|(&coordinate, value)| {
            let h_spinor = coordinate / VECTOR_DIMENSION;
            let vector = coordinate % VECTOR_DIMENSION;
            (
                (derivative * SPINOR_DIMENSION + h_spinor) * VECTOR_DIMENSION + vector,
                value.clone(),
            )
        })
        .collect()
}

fn direct_corrected_gamma_column(
    derivative: usize,
    basis_column: &BTreeMap<usize, ExactQi>,
    gamma_table: &[(u16, IntegerMatrix)],
) -> BTreeMap<usize, ExactQi> {
    let mut output = BTreeMap::new();
    for (form, (_, gamma)) in gamma_table.iter().enumerate() {
        for (&coordinate, value) in basis_column {
            let h_spinor = coordinate / VECTOR_DIMENSION;
            let vector = coordinate % VECTOR_DIMENSION;
            let factor = -i64::from(gamma[derivative][h_spinor]);
            if factor == 0 {
                continue;
            }
            let row = form * VECTOR_DIMENSION + vector;
            let entry = output.entry(row).or_insert_with(ExactQi::zero);
            entry.add_assign(&value.scaled(&Ratio::from_integer(factor)));
            if entry.is_zero() {
                output.remove(&row);
            }
        }
    }
    output
}

fn hash_exact_map(
    hasher: &mut Sha256,
    degree: usize,
    column: usize,
    map: &BTreeMap<usize, ExactQi>,
) {
    hasher.update((degree as u64).to_le_bytes());
    hasher.update((column as u64).to_le_bytes());
    hasher.update((map.len() as u64).to_le_bytes());
    for (&row, value) in map {
        hasher.update((row as u64).to_le_bytes());
        for rational in [&value.real, &value.imaginary] {
            hasher.update(rational.numer().to_le_bytes());
            hasher.update(rational.denom().to_le_bytes());
        }
    }
}

fn gamma_parity_one_degree(degree: usize) -> (usize, String, String) {
    let basis = canonical_gamma_traceless_frame_basis();
    let operator = gamma_dh_operator(degree);
    let gamma_table = masks_of_degree(degree)
        .into_iter()
        .map(|mask| {
            let indices = (0..VECTOR_DIMENSION)
                .filter(|axis| mask & (1_u16 << axis) != 0)
                .collect::<Vec<_>>();
            (mask, raised_gamma_product(&indices))
        })
        .collect::<Vec<_>>();
    let mut adapted_hasher = Sha256::new();
    let mut direct_hasher = Sha256::new();
    for hasher in [&mut adapted_hasher, &mut direct_hasher] {
        hasher.update(b"adynkra-11d-corrected-gamma-dh-parity-v1");
        hasher.update((degree as u64).to_le_bytes());
    }
    let mut residual_rows = 0;
    for derivative in 0..SPINOR_DIMENSION {
        for (h_hat, basis_column) in basis.iter().enumerate() {
            let column = derivative * H_HAT_DIMENSION + h_hat;
            let original = original_dh_column(derivative, basis_column);
            let adapted = operator.apply_sparse(&adapt_h_slot(&original, false));
            let direct = direct_corrected_gamma_column(derivative, basis_column, &gamma_table);
            residual_rows += adapted
                .keys()
                .chain(direct.keys())
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .filter(|row| adapted.get(row) != direct.get(row))
                .count();
            hash_exact_map(&mut adapted_hasher, degree, column, &adapted);
            hash_exact_map(&mut direct_hasher, degree, column, &direct);
        }
    }
    (
        residual_rows,
        format!("{:x}", adapted_hasher.finalize()),
        format!("{:x}", direct_hasher.finalize()),
    )
}

pub fn verify_corrected_gamma25_parity() -> CorrectedGamma25ParityReport {
    let (p2_residual_rows, p2_adapted_sha256, p2_direct_sha256) = gamma_parity_one_degree(2);
    let (p5_residual_rows, p5_adapted_sha256, p5_direct_sha256) = gamma_parity_one_degree(5);
    let passed = p2_residual_rows == 0
        && p5_residual_rows == 0
        && p2_adapted_sha256 == p2_direct_sha256
        && p5_adapted_sha256 == p5_direct_sha256;
    CorrectedGamma25ParityReport {
        schema_version: "adynkra-11d-corrected-gamma25-parity-v1",
        columns_checked_per_degree: SPINOR_DIMENSION * H_HAT_DIMENSION,
        p2_residual_rows,
        p5_residual_rows,
        p2_adapted_sha256,
        p2_direct_sha256,
        p5_adapted_sha256,
        p5_direct_sha256,
        passed,
    }
}

fn form_ordinal(degree: usize, mask: u16) -> usize {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|candidate| candidate.count_ones() as usize == degree)
        .position(|candidate| candidate == mask)
        .unwrap()
}

#[derive(Default)]
struct CorrectedForms {
    psi: [PolynomialMap; 4],
    d_psi: [PolynomialMap; 4],
}

fn corrected_forms(d_h: &PolynomialMap, d_d_h: &PolynomialMap) -> CorrectedForms {
    let mut output = CorrectedForms::default();
    for (monomial, slice) in transpose(d_h) {
        let solved = solve_conventional_compensators(&adapt_h_slot(&slice, false));
        for (slot, degree, form) in [
            (0, 1, solved.psi_one),
            (1, 3, solved.psi_three),
            (2, 4, solved.psi_four),
            (3, 5, solved.psi_five),
        ] {
            for (mask, value) in form {
                add_polynomial_value(
                    &mut output.psi[slot],
                    form_ordinal(degree, mask),
                    monomial.clone(),
                    value,
                );
            }
        }
    }
    for (monomial, slice) in transpose(d_d_h) {
        let solved = solve_higher_jet_conventional_compensators(&adapt_h_slot(&slice, true));
        for (slot, form) in [
            (0, solved.d_psi_one),
            (1, solved.d_psi_three),
            (2, solved.d_psi_four),
            (3, solved.d_psi_five),
        ] {
            for (coordinate, value) in form {
                add_polynomial_value(&mut output.d_psi[slot], coordinate, monomial.clone(), value);
            }
        }
    }
    output
}

fn build_delta(forms: &CorrectedForms) -> PolynomialMap {
    let mut output = PolynomialMap::new();
    for (slot, degree) in [(0, 1), (1, 3), (2, 4), (3, 5)] {
        merge(
            &mut output,
            &apply_sliced(&forms.psi[slot], |slice| {
                inject_holonomy_form_into_delta(degree, slice)
            }),
        );
    }
    output
}

fn build_d_delta(forms: &CorrectedForms) -> PolynomialMap {
    let mut output = PolynomialMap::new();
    for (slot, degree) in [(0, 1), (1, 3), (2, 4), (3, 5)] {
        merge(
            &mut output,
            &apply_sliced(&forms.d_psi[slot], |slice| {
                inject_d_holonomy_form_into_d_delta(degree, slice)
            }),
        );
    }
    output
}

fn spinor_derivative(input: &PolynomialMap, dimension: usize) -> Result<PolynomialMap, String> {
    let mut output = PolynomialMap::new();
    for (&coordinate, polynomial) in input {
        assert!(coordinate < dimension);
        for derivative in 0..SPINOR_DIMENSION {
            for (monomial, value) in left_multiply_d(derivative, polynomial)?.terms {
                add_polynomial_value(
                    &mut output,
                    derivative * dimension + coordinate,
                    monomial,
                    value,
                );
            }
        }
    }
    Ok(output)
}

fn polynomial_residual_rows(left: &PolynomialMap, right: &PolynomialMap) -> usize {
    let left = transpose(left);
    let right = transpose(right);
    left.keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|monomial| {
            let empty = BTreeMap::new();
            let l = left.get(&monomial).unwrap_or(&empty);
            let r = right.get(&monomial).unwrap_or(&empty);
            l.keys()
                .chain(r.keys())
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .filter(|coordinate| l.get(coordinate) != r.get(coordinate))
                .count()
        })
        .sum()
}

fn multiply_momentum(
    monomial: &OrderedSuperderivativeMonomial,
    term: &ExactPolynomialCoefficient,
) -> Result<OrderedSuperderivativeMonomial, String> {
    let mut exponents = monomial.momentum.exponents;
    for (axis, exponent) in term.monomial.exponents.into_iter().enumerate() {
        exponents[axis] = exponents[axis]
            .checked_add(u16::from(exponent))
            .ok_or_else(|| format!("momentum overflow on axis {axis}"))?;
    }
    Ok(OrderedSuperderivativeMonomial {
        exterior_spinor_mask: monomial.exterior_spinor_mask,
        momentum: FormalMomentumMonomial { exponents },
    })
}

fn add_coordinate_polynomial(
    output: &mut BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
    coordinate: usize,
    monomial: OrderedSuperderivativeMonomial,
    value: ExactQi,
) {
    if value.is_zero() {
        return;
    }
    let key = (coordinate, monomial);
    let entry = output.entry(key.clone()).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&key);
    }
}

fn corrected_curl(
    d_delta: &PolynomialMap,
) -> Result<
    (
        BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
        usize,
        usize,
    ),
    String,
> {
    let curvature = &target_sector_complex(TargetSector::RaritaSchwinger).curvature;
    let mut curl = BTreeMap::new();
    let mut eq29 = BTreeMap::new();
    let mut frame_rows = 0;
    for (monomial, d_delta_slice) in transpose(d_delta) {
        let input = Eq25FermionicFrameInput {
            d_delta: d_delta_slice,
            d_scale: BTreeMap::new(),
        };
        let frame = apply_eq25_fermionic_frame(&input)?;
        frame_rows += frame.len();
        for (frame_coordinate, frame_value) in &frame {
            for (curl_coordinate, term) in curvature.column_terms(*frame_coordinate) {
                add_coordinate_polynomial(
                    &mut curl,
                    curl_coordinate,
                    multiply_momentum(&monomial, &term)?,
                    multiply(frame_value, &public_coefficient(&term)),
                );
            }
        }
        for momentum_axis in 0..VECTOR_DIMENSION {
            let mut shifted = monomial.clone();
            shifted.momentum.exponents[momentum_axis] += 1;
            for (curl_coordinate, value) in apply_eq29_fermionic_anholonomy(&input, momentum_axis)?
            {
                add_coordinate_polynomial(&mut eq29, curl_coordinate, shifted.clone(), value);
            }
        }
    }
    let residual = curl
        .keys()
        .chain(eq29.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| curl.get(key) != eq29.get(key))
        .count();
    Ok((curl, frame_rows, residual))
}

fn corrected_teleparallel(
    curl: &BTreeMap<(usize, OrderedSuperderivativeMonomial), ExactQi>,
) -> BTreeMap<FullChainRowKey, ExactQi> {
    let mut slices = MonomialSlices::new();
    for ((coordinate, monomial), value) in curl {
        let slice = slices.entry(monomial.clone()).or_default();
        let entry = slice.entry(*coordinate).or_insert_with(ExactQi::zero);
        entry.add_assign(value);
        if entry.is_zero() {
            slice.remove(coordinate);
        }
    }
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    assert_eq!(operator.input_dimension, GRAVITINO_CURL_DIMENSION);
    assert_eq!(operator.output_dimension, D_F_FOUR_FORM_DIMENSION);
    let mut output = BTreeMap::new();
    for (monomial, slice) in slices {
        for (coordinate, value) in operator.apply_sparse(&slice) {
            add_row_value(
                &mut output,
                FullChainRowKey {
                    output_coordinate: coordinate,
                    exterior_spinor_mask: monomial.exterior_spinor_mask,
                    momentum_exponents: monomial.momentum.exponents,
                },
                value,
            );
        }
    }
    output
}

fn flatten_polynomial_stage(input: &PolynomialMap) -> BTreeMap<FullChainRowKey, ExactQi> {
    let mut output = BTreeMap::new();
    for (&output_coordinate, polynomial) in input {
        for (monomial, value) in &polynomial.terms {
            add_row_value(
                &mut output,
                FullChainRowKey {
                    output_coordinate,
                    exterior_spinor_mask: monomial.exterior_spinor_mask,
                    momentum_exponents: monomial.momentum.exponents,
                },
                value.clone(),
            );
        }
    }
    output
}

fn corrected_eq25_frame(d_delta: &PolynomialMap) -> Result<PolynomialMap, String> {
    let mut output = PolynomialMap::new();
    for (monomial, d_delta_slice) in transpose(d_delta) {
        let frame = apply_eq25_fermionic_frame(&Eq25FermionicFrameInput {
            d_delta: d_delta_slice,
            d_scale: BTreeMap::new(),
        })?;
        for (coordinate, value) in frame {
            add_polynomial_value(&mut output, coordinate, monomial.clone(), value);
        }
    }
    Ok(output)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1, |value, index| value * (n - index) / (index + 1))
}

fn target_three_form_ordinal(numeric_ordinal: usize) -> usize {
    let mask = (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 3)
        .nth(numeric_ordinal)
        .unwrap();
    let indices = (0..VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    let mut ordinal = 0;
    let mut next = 0;
    for (position, value) in indices.into_iter().enumerate() {
        for candidate in next..value {
            ordinal += binomial(VECTOR_DIMENSION - candidate - 1, 3 - position - 1);
        }
        next = value + 1;
    }
    ordinal
}

fn corrected_candidate(
    d_psi_three: &PolynomialMap,
) -> Result<BTreeMap<FullChainRowKey, ExactQi>, String> {
    let curvature = &target_sector_complex(TargetSector::FourForm).curvature;
    let mut output = BTreeMap::new();
    for (&coordinate, polynomial) in d_psi_three {
        let outer = coordinate / THREE_FORM_DIMENSION;
        let numeric_form = coordinate % THREE_FORM_DIMENSION;
        let potential = target_three_form_ordinal(numeric_form);
        for (four_form, term) in curvature.column_terms(potential) {
            for (monomial, value) in &polynomial.terms {
                let output_monomial = multiply_momentum(monomial, &term)?;
                add_row_value(
                    &mut output,
                    FullChainRowKey {
                        output_coordinate: outer * W_FOUR_FORM_DIMENSION + four_form,
                        exterior_spinor_mask: output_monomial.exterior_spinor_mask,
                        momentum_exponents: output_monomial.momentum.exponents,
                    },
                    multiply(value, &public_coefficient(&term)),
                );
            }
        }
    }
    Ok(output)
}

/// Build the two exact, convention-consistent streams used by the production
/// comparator. The right-C source adapter is applied after the canonical
/// physical representative has been expanded into DH/DDH PBW slices, so a
/// later gamma-trace reprojection cannot change the intended H-slot action.
pub(crate) fn corrected_full_chain_streams(
    source_ordinal: usize,
) -> Result<
    (
        BTreeMap<FullChainRowKey, ExactQi>,
        BTreeMap<FullChainRowKey, ExactQi>,
    ),
    String,
> {
    if source_ordinal >= H_HAT_DIMENSION {
        return Err(format!(
            "source ordinal {source_ordinal} outside 0..{H_HAT_DIMENSION}"
        ));
    }
    let input = basis_input(source_ordinal)?;
    let representative = canonical_physical_frame_representative(&input)?;
    let mut d_h = PolynomialMap::new();
    let mut d_d_h = PolynomialMap::new();
    visit_linearized_frame_jet(&representative, |entry| {
        match entry.sector {
            LinearizedFrameJetSector::DH => {
                d_h.insert(entry.coordinate, entry.polynomial);
            }
            LinearizedFrameJetSector::DDH => {
                d_d_h.insert(entry.coordinate, entry.polynomial);
            }
            _ => {}
        }
        Ok(())
    })?;
    let forms = corrected_forms(&d_h, &d_d_h);
    let delta = build_delta(&forms);
    let d_delta = build_d_delta(&forms);
    let derived_d_delta = spinor_derivative(&delta, DELTA_DIMENSION)?;
    let delta_residual = polynomial_residual_rows(&d_delta, &derived_d_delta);
    if delta_residual != 0 {
        return Err(format!(
            "corrected DDelta differs from D(Delta) in {delta_residual} PBW rows"
        ));
    }
    let (curl, _, eq29_residual) = corrected_curl(&d_delta)?;
    if eq29_residual != 0 {
        return Err(format!(
            "corrected Eq25 curvature curl differs from Eq29 in {eq29_residual} PBW rows"
        ));
    }
    Ok((
        corrected_candidate(&forms.d_psi[1])?,
        corrected_teleparallel(&curl),
    ))
}

/// Build the exact gauge-fixed section values under the explicit horizontal
/// transformation contract. This is deliberately a typed wrapper rather than
/// an alias returning an unqualified target map.
pub(crate) fn horizontal_corrected_full_chain_streams(
    source_ordinal: usize,
) -> Result<HorizontalCorrectedFullChainStreams, String> {
    let (candidate, section_target) = corrected_full_chain_streams(source_ordinal)?;
    Ok(HorizontalCorrectedFullChainStreams {
        candidate,
        section_target,
        section_psi_two_is_zero: true,
        section_values_unchanged_by_horizontalization: true,
        transformation_law: "T_hor(s(H)); delta_L T_hor = delta_L T_raw + Q(delta_L s), with Q the exact paper Eq25 Psi_[2] compensator",
    })
}

/// Exact intermediate streams for stagewise Lorentz-equivariance audits.
/// Coordinates retain each stage's native basis; the PBW monomial is carried
/// unchanged until the Rarita curl inserts its momentum covector.
pub(crate) fn corrected_full_chain_stage_streams(
    source_ordinal: usize,
) -> Result<CorrectedFullChainStageStreams, String> {
    if source_ordinal >= H_HAT_DIMENSION {
        return Err(format!(
            "source ordinal {source_ordinal} outside 0..{H_HAT_DIMENSION}"
        ));
    }
    let input = basis_input(source_ordinal)?;
    let representative = canonical_physical_frame_representative(&input)?;
    let mut d_h = PolynomialMap::new();
    let mut d_d_h = PolynomialMap::new();
    visit_linearized_frame_jet(&representative, |entry| {
        match entry.sector {
            LinearizedFrameJetSector::DH => {
                d_h.insert(entry.coordinate, entry.polynomial);
            }
            LinearizedFrameJetSector::DDH => {
                d_d_h.insert(entry.coordinate, entry.polynomial);
            }
            _ => {}
        }
        Ok(())
    })?;
    let forms = corrected_forms(&d_h, &d_d_h);
    let delta = build_delta(&forms);
    let d_delta = build_d_delta(&forms);
    let derived_d_delta = spinor_derivative(&delta, DELTA_DIMENSION)?;
    let delta_residual = polynomial_residual_rows(&d_delta, &derived_d_delta);
    if delta_residual != 0 {
        return Err(format!(
            "corrected DDelta differs from D(Delta) in {delta_residual} PBW rows"
        ));
    }
    let eq25_frame = corrected_eq25_frame(&d_delta)?;
    let (curl, _, eq29_residual) = corrected_curl(&d_delta)?;
    if eq29_residual != 0 {
        return Err(format!(
            "corrected Eq25 curvature curl differs from Eq29 in {eq29_residual} PBW rows"
        ));
    }
    let gravitino_curl = curl
        .iter()
        .map(|((output_coordinate, monomial), value)| {
            (
                FullChainRowKey {
                    output_coordinate: *output_coordinate,
                    exterior_spinor_mask: monomial.exterior_spinor_mask,
                    momentum_exponents: monomial.momentum.exponents,
                },
                value.clone(),
            )
        })
        .collect();
    Ok(CorrectedFullChainStageStreams {
        d_delta: flatten_polynomial_stage(&d_delta),
        eq25_frame: flatten_polynomial_stage(&eq25_frame),
        gravitino_curl,
        teleparallel_dg4: corrected_teleparallel(&curl),
    })
}

pub fn compare_corrected_full_chain_column(
    source_ordinal: usize,
) -> Result<CorrectedFullChainColumnReport, String> {
    if source_ordinal >= H_HAT_DIMENSION {
        return Err(format!(
            "source ordinal {source_ordinal} outside 0..{H_HAT_DIMENSION}"
        ));
    }
    let input = basis_input(source_ordinal)?;
    let representative = canonical_physical_frame_representative(&input)?;
    let mut d_h = PolynomialMap::new();
    let mut d_d_h = PolynomialMap::new();
    visit_linearized_frame_jet(&representative, |entry| {
        match entry.sector {
            LinearizedFrameJetSector::DH => {
                d_h.insert(entry.coordinate, entry.polynomial);
            }
            LinearizedFrameJetSector::DDH => {
                d_d_h.insert(entry.coordinate, entry.polynomial);
            }
            _ => {}
        }
        Ok(())
    })?;
    let forms = corrected_forms(&d_h, &d_d_h);
    let delta = build_delta(&forms);
    let d_delta = build_d_delta(&forms);
    let derived_d_delta = spinor_derivative(&delta, DELTA_DIMENSION)?;
    let delta_derivative_residual_rows = polynomial_residual_rows(&d_delta, &derived_d_delta);
    let (curl, corrected_eq25_frame_rows, eq29_curl_residual_rows) = corrected_curl(&d_delta)?;
    let teleparallel = corrected_teleparallel(&curl);
    let candidate = corrected_candidate(&forms.d_psi[1])?;
    let keys = candidate
        .keys()
        .chain(teleparallel.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let scale = keys
        .iter()
        .find_map(|key| divide(teleparallel.get(key)?, candidate.get(key)?));
    let mut exact_residual_rows = 0;
    let mut first_mismatch = None;
    for key in &keys {
        let candidate_value = candidate.get(key).cloned().unwrap_or_else(ExactQi::zero);
        let target_value = teleparallel.get(key).cloned().unwrap_or_else(ExactQi::zero);
        let mut residual = scale
            .as_ref()
            .map(|factor| multiply(factor, &candidate_value))
            .unwrap_or_else(ExactQi::zero);
        residual.add_assign(&target_value.scaled(&Ratio::from_integer(-1)));
        if !residual.is_zero() {
            exact_residual_rows += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(FullChainWitness {
                    row: key.clone(),
                    candidate: public(&candidate_value),
                    teleparallel: public(&target_value),
                    residual: public(&residual),
                });
            }
        }
    }
    let common_rows = keys
        .iter()
        .filter(|key| candidate.contains_key(*key) && teleparallel.contains_key(*key))
        .count();
    let psi_terms = std::array::from_fn(|slot| {
        forms.psi[slot]
            .values()
            .map(|polynomial| polynomial.terms.len())
            .sum()
    });
    let d_psi_terms = std::array::from_fn(|slot| {
        forms.d_psi[slot]
            .values()
            .map(|polynomial| polynomial.terms.len())
            .sum()
    });
    let passed = delta_derivative_residual_rows == 0
        && eq29_curl_residual_rows == 0
        && scale.is_some()
        && exact_residual_rows == 0;
    Ok(CorrectedFullChainColumnReport {
        schema_version: "adynkra-11d-corrected-full-chain-oracle-v1",
        source_ordinal,
        source_variance_adapter: "right-multiply the H-spinor slot of every DH/DDH slice by primitive C before the Eq. (40) Gamma2/Gamma5 solves; (Gamma[p] C) C = -Gamma[p]",
        candidate_normalization: "Psi_[3] = (1/16) corrected Gamma2 exterior; numeric mask joined to lexicographic target A3",
        row_key: "(explicit outer derivative spinor * 330 + lexicographic G4 ordinal, ordered exterior-D mask, p_0..p_10 exponents)",
        corrected_psi_nonzero_terms: psi_terms,
        corrected_d_psi_nonzero_terms: d_psi_terms,
        delta_nonzero_rows: delta
            .values()
            .map(|polynomial| polynomial.terms.len())
            .sum(),
        d_delta_nonzero_rows: d_delta
            .values()
            .map(|polynomial| polynomial.terms.len())
            .sum(),
        delta_derivative_residual_rows,
        corrected_eq25_frame_rows,
        corrected_curl_rows: curl.len(),
        eq29_curl_residual_rows,
        candidate_rows: candidate.len(),
        corrected_teleparallel_rows: teleparallel.len(),
        common_rows,
        candidate_only_rows: candidate
            .keys()
            .filter(|key| !teleparallel.contains_key(*key))
            .count(),
        teleparallel_only_rows: teleparallel
            .keys()
            .filter(|key| !candidate.contains_key(*key))
            .count(),
        exact_scale: scale.as_ref().map(public),
        exact_residual_rows,
        first_mismatch,
        passed,
        boundary: "This is an independent first-column check of the fully source-variance-corrected compensator-to-teleparallel chain. Failure rules out proportionality on this unrestricted H_hat source column only; it does not prove bidegree exhaustion, target-gauge descent, or irreducibility.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eleven_dimensional_gamma24_source_variance::corrected_gamma_two_exterior_operator_ref;

    #[test]
    fn right_c_adapter_has_full_gamma_two_and_five_parity() {
        let report = verify_corrected_gamma25_parity();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.columns_checked_per_degree, 10_240);
        assert_eq!(report.p2_residual_rows, 0);
        assert_eq!(report.p5_residual_rows, 0);
        assert_eq!(report.p2_adapted_sha256, report.p2_direct_sha256);
        assert_eq!(report.p5_adapted_sha256, report.p5_direct_sha256);
        assert!(report.passed);
    }

    #[test]
    fn right_c_slice_adapter_matches_direct_corrected_gamma_two() {
        let input = basis_input(0).unwrap();
        let representative = canonical_physical_frame_representative(&input).unwrap();
        let mut d_h = PolynomialMap::new();
        let mut d_d_h = PolynomialMap::new();
        visit_linearized_frame_jet(&representative, |entry| {
            match entry.sector {
                LinearizedFrameJetSector::DH => {
                    d_h.insert(entry.coordinate, entry.polynomial);
                }
                LinearizedFrameJetSector::DDH => {
                    d_d_h.insert(entry.coordinate, entry.polynomial);
                }
                _ => {}
            }
            Ok(())
        })
        .unwrap();
        let actual = corrected_forms(&d_h, &d_d_h).psi[1].clone();
        let direct = corrected_gamma_two_exterior_operator_ref();
        let scalar = CanonicalSuperPolynomial::scalar(ExactQi::one());
        let mut expected = PolynomialMap::new();
        for derivative in 0..SPINOR_DIMENSION {
            let monomials = left_multiply_d(derivative, &scalar).unwrap();
            for entry in &direct.columns[derivative * H_HAT_DIMENSION] {
                for (monomial, pbw_value) in &monomials.terms {
                    add_polynomial_value(
                        &mut expected,
                        entry.row,
                        monomial.clone(),
                        multiply(pbw_value, &entry.coefficient.scaled(&Ratio::new(1, 16))),
                    );
                }
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn corrected_full_chain_column_zero_is_exactly_decisive() {
        let report = compare_corrected_full_chain_column(0).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.corrected_psi_nonzero_terms, [2, 72, 168, 252]);
        assert_eq!(
            report.corrected_d_psi_nonzero_terms,
            [66, 2_376, 5_544, 8_316]
        );
        assert_eq!(report.delta_nonzero_rows, 214);
        assert_eq!(report.d_delta_nonzero_rows, 7_062);
        assert_eq!(report.delta_derivative_residual_rows, 0);
        assert_eq!(report.corrected_eq25_frame_rows, 2_128);
        assert_eq!(report.corrected_curl_rows, 21_278);
        assert_eq!(report.eq29_curl_residual_rows, 0);
        assert_eq!(report.candidate_rows, 18_972);
        assert_eq!(report.corrected_teleparallel_rows, 343_720);
        assert_eq!(report.common_rows, 0);
        assert_eq!(report.candidate_only_rows, 18_972);
        assert_eq!(report.teleparallel_only_rows, 343_720);
        assert_eq!(report.exact_scale, None);
        assert_eq!(report.exact_residual_rows, 343_720);
        let witness = report.first_mismatch.as_ref().unwrap();
        assert_eq!(witness.row.output_coordinate, 0);
        assert_eq!(witness.row.exterior_spinor_mask, 0x0001_0001);
        assert_eq!(witness.row.momentum_exponents[1], 1);
        assert_eq!(witness.candidate.real, [0, 1]);
        assert_eq!(witness.teleparallel.real, [1, 1_280]);
        assert!(!report.passed);
    }
}
