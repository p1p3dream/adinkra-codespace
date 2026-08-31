//! Exact component-level graviton/gravitino relative-normalization canary.
//!
//! This module uses the flat linearization of hep-th/0101037 Eq. (41), not
//! the gauge-fixed Eq. (25) source construction.  With
//! `T_{alpha beta}{}^c=i(Gamma^c)_{alpha beta}` and the inverse frame
//! `e_a{}^m=delta_a{}^m+f_a{}^m`, Eq. (41) gives
//! `D_alpha f_a{}^m=-i(C Gamma^m)_{alpha gamma} psi_a{}^gamma`.
//! Since `h_mn=-eta_nn f_m{}^n-eta_mm f_n{}^m`, the inferred metric law is
//! `D_alpha h_mn=i[(C Gamma_n)_{alpha gamma}psi_m{}^gamma
//!                  +(C Gamma_m)_{alpha gamma}psi_n{}^gamma]`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_four_form_56_gpu::PINNED_PRIMES;
use crate::eleven_dimensional_majorana::{real_charge_conjugation, real_gamma_matrices};
use crate::eleven_dimensional_physical_curvature::ExactQi;
use crate::eleven_dimensional_target_equation_complex::{TargetSector, target_sector_complex};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const METRIC_DIMENSION: usize = 66;
const PAIR_DIMENSION: usize = 55;
const RIEMANN_AMBIENT_DIMENSION: usize = PAIR_DIMENSION * PAIR_DIMENSION;
const PSI_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const CURL_DIMENSION: usize = PAIR_DIMENSION * SPINOR_DIMENSION;
const D_RIEMANN_DIMENSION: usize = SPINOR_DIMENSION * RIEMANN_AMBIENT_DIMENSION;
const HEP_TH_0101037_SHA256: &str =
    "3d40a1b32fa4491dee56b3e99802172d2c5039b2de198b987ce121a1bbb15cc3";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct GravitonGravitinoAxisReport {
    pub momentum_axis: usize,
    pub d_riemann_rank_by_prime: [usize; 3],
    pub curl_rank_by_prime: [usize; 3],
    pub stacked_rank_by_prime: [usize; 3],
    pub common_gauge_kernel_dimension: usize,
    pub exact_curl_factor_replay_residual_entries: usize,
    pub exact_d_riemann_gauge_residual_entries: usize,
    pub exact_curl_gauge_residual_entries: usize,
    pub omitted_symmetric_term_mutation_difference_entries: usize,
    pub first_d_riemann_coordinate: usize,
    pub first_d_riemann_value: String,
    pub d_riemann_stream_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct GravitonGravitinoRelativeReport {
    pub schema_version: &'static str,
    pub source_pdf_sha256: &'static str,
    pub source_equation: &'static str,
    pub inferred_flat_metric_formula: &'static str,
    pub inference_steps: [&'static str; 3],
    pub pinned_primes: [u32; 3],
    pub axes: Vec<GravitonGravitinoAxisReport>,
    pub characteristic_zero_common_rank: usize,
    pub characteristic_zero_common_kernel_dimension: usize,
    pub normalization_canary_passed: bool,
    pub charge_square_minus_identity_residual_entries: usize,
    pub lower_c_gamma_symmetry_residual_entries: usize,
    pub no_c_mutation_difference_entries: usize,
    pub right_c_mutation_difference_entries: usize,
    pub spinor_basis_conversion: &'static str,
    pub source_sha256: BTreeMap<String, String>,
    pub passed: bool,
    pub scope_boundary: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModQi {
    real: u32,
    imaginary: u32,
}

impl ModQi {
    const ZERO: Self = Self {
        real: 0,
        imaginary: 0,
    };

    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    fn add(self, other: Self, prime: u32) -> Self {
        Self {
            real: ((u64::from(self.real) + u64::from(other.real)) % u64::from(prime)) as u32,
            imaginary: ((u64::from(self.imaginary) + u64::from(other.imaginary)) % u64::from(prime))
                as u32,
        }
    }

    fn subtract(self, other: Self, prime: u32) -> Self {
        Self {
            real: ((u64::from(self.real) + u64::from(prime) - u64::from(other.real))
                % u64::from(prime)) as u32,
            imaginary: ((u64::from(self.imaginary) + u64::from(prime) - u64::from(other.imaginary))
                % u64::from(prime)) as u32,
        }
    }

    fn multiply(self, other: Self, prime: u32) -> Self {
        let modulus = u64::from(prime);
        let real = (u64::from(self.real) * u64::from(other.real) + modulus
            - u64::from(self.imaginary) * u64::from(other.imaginary) % modulus)
            % modulus;
        let imaginary = (u64::from(self.real) * u64::from(other.imaginary)
            + u64::from(self.imaginary) * u64::from(other.real))
            % modulus;
        Self {
            real: real as u32,
            imaginary: imaginary as u32,
        }
    }

    fn inverse(self, prime: u32) -> Self {
        let norm = ((u64::from(self.real) * u64::from(self.real)
            + u64::from(self.imaginary) * u64::from(self.imaginary))
            % u64::from(prime)) as u32;
        assert_ne!(norm, 0);
        let inverse_norm = pow_mod(norm, prime - 2, prime);
        Self {
            real: (u64::from(self.real) * u64::from(inverse_norm) % u64::from(prime)) as u32,
            imaginary: (u64::from(prime - self.imaginary) * u64::from(inverse_norm)
                % u64::from(prime)) as u32,
        }
    }
}

fn pow_mod(mut base: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut output = 1_u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = (u64::from(output) * u64::from(base) % u64::from(prime)) as u32;
        }
        base = (u64::from(base) * u64::from(base) % u64::from(prime)) as u32;
        exponent >>= 1;
    }
    output
}

fn ratio_mod(value: &Ratio<i64>, prime: u32) -> Result<u32, String> {
    let denominator = value.denom().rem_euclid(i64::from(prime)) as u32;
    if denominator == 0 {
        return Err("graviton-relative denominator is inadmissible at a pinned prime".to_string());
    }
    let numerator = value.numer().rem_euclid(i64::from(prime)) as u32;
    Ok(
        (u64::from(numerator) * u64::from(pow_mod(denominator, prime - 2, prime))
            % u64::from(prime)) as u32,
    )
}

fn qi_mod(value: &ExactQi, prime: u32) -> Result<ModQi, String> {
    Ok(ModQi {
        real: ratio_mod(&value.real, prime)?,
        imaginary: ratio_mod(&value.imaginary, prime)?,
    })
}

fn add_exact(output: &mut BTreeMap<usize, ExactQi>, row: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(row).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&row);
    }
}

fn exact_multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn exact_string(value: &ExactQi) -> String {
    format!("{}+({})i", value.real, value.imaginary)
}

fn symmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| (left..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn antisymmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn metric_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn lower_bilinear_gammas() -> Vec<Vec<Vec<i16>>> {
    let charge = real_charge_conjugation();
    real_gamma_matrices()
        .iter()
        .map(|gamma| {
            let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
            for alpha in 0..SPINOR_DIMENSION {
                for pivot in 0..SPINOR_DIMENSION {
                    for gamma_index in 0..SPINOR_DIMENSION {
                        output[alpha][gamma_index] +=
                            i16::from(charge[alpha][pivot]) * i16::from(gamma[pivot][gamma_index]);
                    }
                }
            }
            output
        })
        .collect()
}

fn spinor_conversion_audit() -> (usize, usize, usize, usize) {
    let charge = real_charge_conjugation();
    let gammas = real_gamma_matrices();
    let lower = lower_bilinear_gammas();
    let mut charge_square_residual = 0;
    for row in 0..SPINOR_DIMENSION {
        for column in 0..SPINOR_DIMENSION {
            let value = (0..SPINOR_DIMENSION)
                .map(|pivot| i16::from(charge[row][pivot]) * i16::from(charge[pivot][column]))
                .sum::<i16>();
            let expected = if row == column { -1 } else { 0 };
            charge_square_residual += usize::from(value != expected);
        }
    }
    let mut lower_symmetry_residual = 0;
    let mut no_c_difference = 0;
    let mut right_c_difference = 0;
    for axis in 0..VECTOR_DIMENSION {
        for alpha in 0..SPINOR_DIMENSION {
            for gamma_index in 0..SPINOR_DIMENSION {
                lower_symmetry_residual +=
                    usize::from(lower[axis][alpha][gamma_index] != lower[axis][gamma_index][alpha]);
                no_c_difference += usize::from(
                    lower[axis][alpha][gamma_index] != i16::from(gammas[axis][alpha][gamma_index]),
                );
                let right_c = (0..SPINOR_DIMENSION)
                    .map(|pivot| {
                        i16::from(gammas[axis][alpha][pivot])
                            * i16::from(charge[pivot][gamma_index])
                    })
                    .sum::<i16>();
                right_c_difference += usize::from(lower[axis][alpha][gamma_index] != right_c);
            }
        }
    }
    (
        charge_square_residual,
        lower_symmetry_residual,
        no_c_difference,
        right_c_difference,
    )
}

fn fixed_momentum_terms(sector: TargetSector, momentum_axis: usize) -> Vec<Vec<(usize, ExactQi)>> {
    let operator = &target_sector_complex(sector).curvature;
    (0..operator.columns())
        .map(|column| {
            operator
                .column_terms(column)
                .into_iter()
                .filter(|(_, term)| {
                    term.monomial.exponents[momentum_axis]
                        == if sector == TargetSector::Graviton {
                            2
                        } else {
                            1
                        }
                        && term
                            .monomial
                            .exponents
                            .iter()
                            .enumerate()
                            .all(|(axis, &power)| axis == momentum_axis || power == 0)
                })
                .map(|(row, term)| {
                    (
                        row,
                        ExactQi {
                            real: Ratio::new(term.real_numerator, term.real_denominator),
                            imaginary: Ratio::new(
                                term.imaginary_numerator,
                                term.imaginary_denominator,
                            ),
                        },
                    )
                })
                .collect()
        })
        .collect()
}

fn d_riemann_columns(
    momentum_axis: usize,
    omit_symmetric_term: bool,
) -> Vec<BTreeMap<usize, ExactQi>> {
    let gammas = lower_bilinear_gammas();
    let metric_pairs = symmetric_pairs();
    let riemann = fixed_momentum_terms(TargetSector::Graviton, momentum_axis);
    let mut columns = vec![BTreeMap::new(); PSI_DIMENSION];
    for psi_vector in 0..VECTOR_DIMENSION {
        for psi_spinor in 0..SPINOR_DIMENSION {
            let column = psi_vector * SPINOR_DIMENSION + psi_spinor;
            for (metric_coordinate, &(left, right)) in metric_pairs.iter().enumerate() {
                for derivative_spinor in 0..SPINOR_DIMENSION {
                    let mut integer = 0_i64;
                    if psi_vector == left {
                        integer += metric_sign(right)
                            * i64::from(gammas[right][derivative_spinor][psi_spinor]);
                    }
                    if !omit_symmetric_term && psi_vector == right {
                        integer += metric_sign(left)
                            * i64::from(gammas[left][derivative_spinor][psi_spinor]);
                    }
                    if integer == 0 {
                        continue;
                    }
                    let metric_value = ExactQi {
                        real: Ratio::from_integer(0),
                        imaginary: Ratio::from_integer(integer),
                    };
                    for (riemann_row, curvature_value) in &riemann[metric_coordinate] {
                        add_exact(
                            &mut columns[column],
                            derivative_spinor * RIEMANN_AMBIENT_DIMENSION + riemann_row,
                            exact_multiply(&metric_value, curvature_value),
                        );
                    }
                }
            }
        }
    }
    columns
}

fn curl_columns(momentum_axis: usize) -> Vec<BTreeMap<usize, ExactQi>> {
    fixed_momentum_terms(TargetSector::RaritaSchwinger, momentum_axis)
        .into_iter()
        .map(|entries| entries.into_iter().collect())
        .collect()
}

fn sparse_column_rank(columns: &[BTreeMap<usize, ExactQi>], prime: u32) -> Result<usize, String> {
    let mut pivots = BTreeMap::<usize, BTreeMap<usize, ModQi>>::new();
    for column in columns {
        let mut reduced = column
            .iter()
            .map(|(&row, value)| Ok((row, qi_mod(value, prime)?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        reduced.retain(|_, value| !value.is_zero());
        loop {
            let Some((&row, &value)) = reduced.first_key_value() else {
                break;
            };
            let Some(pivot) = pivots.get(&row) else {
                let inverse = value.inverse(prime);
                for entry in reduced.values_mut() {
                    *entry = entry.multiply(inverse, prime);
                }
                pivots.insert(row, reduced);
                break;
            };
            let terms = pivot
                .iter()
                .map(|(&target, &coefficient)| (target, value.multiply(coefficient, prime)))
                .collect::<Vec<_>>();
            for (target, subtraction) in terms {
                let next = reduced
                    .get(&target)
                    .copied()
                    .unwrap_or(ModQi::ZERO)
                    .subtract(subtraction, prime);
                if next.is_zero() {
                    reduced.remove(&target);
                } else {
                    reduced.insert(target, next);
                }
            }
        }
    }
    Ok(pivots.len())
}

fn exact_map_residual(left: &BTreeMap<usize, ExactQi>, right: &BTreeMap<usize, ExactQi>) -> usize {
    left.keys()
        .chain(right.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|row| left.get(row) != right.get(row))
        .count()
}

fn stream_hash(columns: &[BTreeMap<usize, ExactQi>]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-component-d-riemann-v1\0");
    for (column, entries) in columns.iter().enumerate() {
        hash.update((column as u64).to_le_bytes());
        hash.update((entries.len() as u64).to_le_bytes());
        for (&row, value) in entries {
            hash.update((row as u64).to_le_bytes());
            hash.update(value.real.numer().to_le_bytes());
            hash.update(value.real.denom().to_le_bytes());
            hash.update(value.imaginary.numer().to_le_bytes());
            hash.update(value.imaginary.denom().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn axis_report(momentum_axis: usize) -> Result<GravitonGravitinoAxisReport, String> {
    let d_riemann = d_riemann_columns(momentum_axis, false);
    let curl = curl_columns(momentum_axis);
    let mutated = d_riemann_columns(momentum_axis, true);
    let stacked = d_riemann
        .iter()
        .zip(&curl)
        .map(|(left, right)| {
            let mut output = left.clone();
            for (&row, value) in right {
                add_exact(&mut output, D_RIEMANN_DIMENSION + row, value.clone());
            }
            output
        })
        .collect::<Vec<_>>();
    let d_riemann_rank_by_prime =
        std::array::from_fn(|slot| sparse_column_rank(&d_riemann, PINNED_PRIMES[slot]).unwrap());
    let curl_rank_by_prime =
        std::array::from_fn(|slot| sparse_column_rank(&curl, PINNED_PRIMES[slot]).unwrap());
    let stacked_rank_by_prime =
        std::array::from_fn(|slot| sparse_column_rank(&stacked, PINNED_PRIMES[slot]).unwrap());

    let mut exact_d_riemann_gauge_residual_entries = 0;
    let mut exact_curl_gauge_residual_entries = 0;
    let mutation_difference = d_riemann
        .iter()
        .zip(&mutated)
        .map(|(correct, wrong)| exact_map_residual(correct, wrong))
        .sum();
    for spinor in 0..SPINOR_DIMENSION {
        let gauge_column = momentum_axis * SPINOR_DIMENSION + spinor;
        exact_d_riemann_gauge_residual_entries += d_riemann[gauge_column].len();
        exact_curl_gauge_residual_entries += curl[gauge_column].len();
    }

    let pair_lookup = antisymmetric_pairs()
        .into_iter()
        .enumerate()
        .map(|(ordinal, pair)| (pair, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut factor_residual = 0;
    for source in 0..PSI_DIMENSION {
        let mut section_d_riemann = BTreeMap::new();
        for (&curl_row, value) in &curl[source] {
            let pair = curl_row / SPINOR_DIMENSION;
            let spinor = curl_row % SPINOR_DIMENSION;
            let (left, right) = antisymmetric_pairs()[pair];
            let (section_vector, sign) = if left == momentum_axis {
                (right, 1_i64)
            } else if right == momentum_axis {
                (left, -1_i64)
            } else {
                return Err("fixed-momentum curl escaped the momentum-containing pairs".to_string());
            };
            debug_assert_eq!(pair_lookup[&(left, right)], pair);
            for (&row, coefficient) in &d_riemann[section_vector * SPINOR_DIMENSION + spinor] {
                add_exact(
                    &mut section_d_riemann,
                    row,
                    exact_multiply(value, &coefficient.scaled(&Ratio::from_integer(sign))),
                );
            }
        }
        factor_residual += exact_map_residual(&d_riemann[source], &section_d_riemann);
    }
    let (first_d_riemann_coordinate, first_d_riemann_value) = d_riemann
        .iter()
        .find_map(|column| column.first_key_value())
        .map(|(&row, value)| (row, exact_string(value)))
        .ok_or_else(|| "component D Riemann operator is zero".to_string())?;
    Ok(GravitonGravitinoAxisReport {
        momentum_axis,
        d_riemann_rank_by_prime,
        curl_rank_by_prime,
        stacked_rank_by_prime,
        common_gauge_kernel_dimension: PSI_DIMENSION - stacked_rank_by_prime[0],
        exact_curl_factor_replay_residual_entries: factor_residual,
        exact_d_riemann_gauge_residual_entries,
        exact_curl_gauge_residual_entries,
        omitted_symmetric_term_mutation_difference_entries: mutation_difference,
        first_d_riemann_coordinate,
        first_d_riemann_value,
        d_riemann_stream_sha256: stream_hash(&d_riemann),
    })
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(crate) fn build_report() -> Result<GravitonGravitinoRelativeReport, String> {
    let axes = (0..VECTOR_DIMENSION)
        .map(axis_report)
        .collect::<Result<Vec<_>, _>>()?;
    let source_paths = [
        "src/eleven_dimensional_graviton_gravitino_relative.rs",
        "src/eleven_dimensional_majorana.rs",
        "src/eleven_dimensional_target_equation_complex.rs",
        "src/eleven_dimensional_free_complex.rs",
    ];
    let source_sha256 = source_paths
        .into_iter()
        .map(|path| Ok((path.to_string(), file_sha256(Path::new(path))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let normalization_canary_passed = axes.first().is_some_and(|axis| {
        axis.first_d_riemann_coordinate == 0 && axis.first_d_riemann_value == "0+(2)i"
    });
    let (
        charge_square_minus_identity_residual_entries,
        lower_c_gamma_symmetry_residual_entries,
        no_c_mutation_difference_entries,
        right_c_mutation_difference_entries,
    ) = spinor_conversion_audit();
    let passed = axes.iter().all(|axis| {
        axis.d_riemann_rank_by_prime == [320; 3]
            && axis.curl_rank_by_prime == [320; 3]
            && axis.stacked_rank_by_prime == [320; 3]
            && axis.common_gauge_kernel_dimension == 32
            && axis.exact_curl_factor_replay_residual_entries == 0
            && axis.exact_d_riemann_gauge_residual_entries == 0
            && axis.exact_curl_gauge_residual_entries == 0
            && axis.omitted_symmetric_term_mutation_difference_entries > 0
    }) && normalization_canary_passed
        && charge_square_minus_identity_residual_entries == 0
        && lower_c_gamma_symmetry_residual_entries == 0
        && no_c_mutation_difference_entries > 0
        && right_c_mutation_difference_entries > 0;
    Ok(GravitonGravitinoRelativeReport {
        schema_version: "adynkra-11d-graviton-gravitino-relative-v1",
        source_pdf_sha256: HEP_TH_0101037_SHA256,
        source_equation: "hep-th/0101037 Eq.(41): delta_Q e_a{}^m=-epsilon^beta[T_{beta a}{}^b+psi_a{}^gamma T_{beta gamma}{}^b]e_b{}^m",
        inferred_flat_metric_formula: "D_alpha h_mn=i[(C Gamma_n)_{alpha gamma}psi_m{}^gamma+(C Gamma_m)_{alpha gamma}psi_n{}^gamma]",
        inference_steps: [
            "Flat conventional torsion is T_{alpha beta}{}^c=i(Gamma^c)_{alpha beta} and T_{alpha a}{}^b=0.",
            "Eq.(41) therefore gives D_alpha f_a{}^m=-i(C Gamma^m)_{alpha gamma}psi_a{}^gamma for e_a{}^m=delta_a{}^m+f_a{}^m.",
            "The covariant metric perturbation is h_mn=-eta_nn f_m{}^n-eta_mm f_n{}^m, which gives the recorded symmetric formula with both vector metric factors.",
        ],
        pinned_primes: PINNED_PRIMES,
        axes,
        characteristic_zero_common_rank: 320,
        characteristic_zero_common_kernel_dimension: 32,
        normalization_canary_passed,
        charge_square_minus_identity_residual_entries,
        lower_c_gamma_symmetry_residual_entries,
        no_c_mutation_difference_entries,
        right_c_mutation_difference_entries,
        spinor_basis_conversion: "The curl stores the upper Majorana spinor coordinate gamma unchanged. Eq.(41) lowers the derivative spinor with the primitive charge form, so its coefficient is (C Gamma_m)_{alpha gamma}. Both C and Gamma use the same real Majorana basis; C^2=-I and C Gamma_m is symmetric. Omitting C or multiplying C on the right changes the exact tensor and is rejected.",
        source_sha256,
        passed,
        scope_boundary: "This certifies the independent linearized component graviton/gravitino relative-curvature normalization and common gauge kernel. It does not invoke Eq25 or Hhat, identify the Hhat physical source, include nonlinear four-form terms, or prove irreducibility.",
    })
}

pub(crate) fn write_report(path: &Path) -> Result<GravitonGravitinoRelativeReport, String> {
    let report = build_report()?;
    if !report.passed {
        return Err("graviton/gravitino relative report failed a scientific gate".to_string());
    }
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize graviton-relative report: {error}"))?;
    let scratch = path.with_extension("json.tmp");
    fs::write(&scratch, bytes).map_err(|error| format!("write {}: {error}", scratch.display()))?;
    fs::rename(&scratch, path).map_err(|error| {
        format!(
            "rename {} to {}: {error}",
            scratch.display(),
            path.display()
        )
    })?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p0_component_graviton_gravitino_canary() {
        let report = axis_report(0).unwrap();
        eprintln!("GRAVITON_GRAVITINO_P0 {report:?}");
        assert_eq!(report.d_riemann_rank_by_prime, [320; 3]);
        assert_eq!(report.curl_rank_by_prime, [320; 3]);
        assert_eq!(report.stacked_rank_by_prime, [320; 3]);
        assert_eq!(report.common_gauge_kernel_dimension, 32);
        assert_eq!(report.exact_curl_factor_replay_residual_entries, 0);
        assert!(report.omitted_symmetric_term_mutation_difference_entries > 0);
    }

    #[test]
    #[ignore = "writes the authoritative report-last artifact"]
    fn write_authoritative_report() {
        let path = Path::new("results/adynkra_11d_graviton_gravitino_relative.json");
        let report = write_report(path).unwrap();
        assert!(report.passed);
        eprintln!(
            "GRAVITON_GRAVITINO_ARTIFACT_SHA {}",
            file_sha256(path).unwrap()
        );
    }
}
