//! Exact ordinary-on-shell relative normalization of the graviton and A3 fibers.
//!
//! The common source is the independent component gravitino curl
//! `C_ab{}^gamma`.  No Eq. (25), `Hhat`, Eq. (40), or conjectural source map is
//! used.  The graviton leg is evaluated in two independent ways:
//!
//! 1. hep-th/0101037 Eq. (41), specialized to the ordinary on-shell torsions,
//!    gives the covariant metric descendant
//!    `D_alpha h_mn = i[(gamma_m psi_n)_alpha+(gamma_n psi_m)_alpha]`;
//! 2. hep-th/0107155 Eqs. (3.1f), (3.2b), and (3.2f) give the same descendant
//!    after the spin connection is curled into the repository's doubled
//!    linearized Riemann convention.
//!
//! The A3 leg is the independently certified Eq. (3.1g) map.  Because both
//! legs use the same typed curl coordinates, their relative target
//! normalization is fixed on the ordinary on-shell component fiber.  Eq. (41)
//! has extra J and X terms off shell, so this module makes no off-shell source
//! normalization claim.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use num_rational::Ratio;
use num_traits::Zero;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_majorana::{real_charge_conjugation, real_gamma_matrices};
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const SYMMETRIC_DIMENSION: usize = 66;
const PAIR_DIMENSION: usize = 55;
const RIEMANN_AMBIENT_DIMENSION: usize = PAIR_DIMENSION * PAIR_DIMENSION;
const FOUR_FORM_DIMENSION: usize = 330;
const FRAME_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const CURL_DIMENSION: usize = PAIR_DIMENSION * SPINOR_DIMENSION;
const PRIME: u32 = 1_073_741_783;
const HEP_TH_0101037_PDF_SHA256: &str =
    "3d40a1b32fa4491dee56b3e99802172d2c5039b2de198b987ce121a1bbb15cc3";
const HEP_TH_0107155_PDF_SHA256: &str =
    "71ccd43c2dea3df8fb9708c016595463cca2674bccad1872c955fc2c8647f25e";

type Sparse = BTreeMap<usize, ExactQi>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct GravitonRelativeOracleReport {
    pub schema_version: &'static str,
    pub momentum_axis: usize,
    pub component_frame_dimension: usize,
    pub component_curl_dimension: usize,
    pub component_curl_rank: usize,
    pub eq41_metric_descendant_dimensions: (usize, usize),
    pub graviton_riemann_descendant_dimensions: (usize, usize),
    pub eq31f_spin_connection_descendant_dimensions: (usize, usize),
    pub eq31f_eq41_all_row_residual_entries: usize,
    pub eq41_charge_row_adapter_residual_entries: usize,
    pub eq41_charge_row_adapter_mutation_residual_entries: usize,
    pub eq31f_local_connection_curl_residual_entries: usize,
    pub graviton_bianchi_residual_entries: usize,
    pub graviton_normalization_mutation_residual_entries: usize,
    pub graviton_descendant_rank: usize,
    pub eq31g_descendant_rank: usize,
    pub eq31g_bianchi_residual_entries: usize,
    pub eq31f_printed_curl_coefficient: &'static str,
    pub eq31g_printed_curl_coefficient: &'static str,
    pub printed_coefficient_ratio_eq31g_over_eq31f: &'static str,
    pub repository_riemann_is_twice_conventional: bool,
    pub common_curl_basis_sha256: String,
    pub eq41_riemann_stream_sha256: String,
    pub eq31f_riemann_stream_sha256: String,
    pub eq31g_dg4_stream_sha256: String,
    pub source_pdf_sha256: BTreeMap<&'static str, &'static str>,
    pub oracle_source_sha256: String,
    pub ordinary_on_shell_relative_normalization_fixed: bool,
    pub off_shell_hhat_source_normalization_fixed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn combinations(degree: usize) -> Vec<Vec<usize>> {
    fn extend(
        next: usize,
        remaining: usize,
        prefix: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(prefix.clone());
            return;
        }
        for value in next..=VECTOR_DIMENSION - remaining {
            prefix.push(value);
            extend(value + 1, remaining - 1, prefix, output);
            prefix.pop();
        }
    }
    let mut output = Vec::new();
    extend(0, degree, &mut Vec::new(), &mut output);
    output
}

fn metric_sign(axis: usize) -> i64 {
    if axis == 0 { -1 } else { 1 }
}

fn exact(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
    }
}

fn multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn add(output: &mut Sparse, row: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(row).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&row);
    }
}

fn fixed_coefficient(coefficient: &ExactPolynomialCoefficient, momentum_axis: usize) -> ExactQi {
    let is_selected_monomial = coefficient
        .monomial
        .exponents
        .iter()
        .enumerate()
        .all(|(axis, &power)| axis == momentum_axis || power == 0);
    if is_selected_monomial {
        exact(coefficient)
    } else {
        ExactQi::zero()
    }
}

fn frame_to_curl_columns(momentum_axis: usize) -> Vec<Sparse> {
    let curvature = &target_sector_complex(TargetSector::RaritaSchwinger).curvature;
    (0..FRAME_DIMENSION)
        .map(|column| {
            let mut output = Sparse::new();
            for (row, coefficient) in curvature.column_terms(column) {
                add(
                    &mut output,
                    row,
                    fixed_coefficient(&coefficient, momentum_axis),
                );
            }
            output
        })
        .collect()
}

fn symmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| (left..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn lowered_gamma_entry(gammas: &[Vec<Vec<i8>>], axis: usize, alpha: usize, gamma: usize) -> i64 {
    metric_sign(axis) * i64::from(gammas[axis][alpha][gamma])
}

/// Eq. (41) is literal in a lower spinor-derivative row, with matrix
/// `C Gamma_m`.  The repository D G4 convention uses the corresponding mixed
/// row.  Since the primitive Majorana charge matrix obeys `C^2=-I`, its
/// inverse is `-C`; applying it must recover `Gamma_m` entrywise.
fn charge_row_adapter_residuals() -> (usize, usize) {
    let gammas = real_gamma_matrices();
    let charge = real_charge_conjugation();
    let mut correct_residuals = 0_usize;
    let mut wrong_sign_residuals = 0_usize;
    for gamma in &gammas {
        for alpha in 0..SPINOR_DIMENSION {
            for input in 0..SPINOR_DIMENSION {
                let mut correct = 0_i64;
                let mut wrong = 0_i64;
                for lower_row in 0..SPINOR_DIMENSION {
                    let mut lower_lower = 0_i64;
                    for pivot in 0..SPINOR_DIMENSION {
                        lower_lower +=
                            i64::from(charge[lower_row][pivot]) * i64::from(gamma[pivot][input]);
                    }
                    correct += -i64::from(charge[alpha][lower_row]) * lower_lower;
                    wrong += i64::from(charge[alpha][lower_row]) * lower_lower;
                }
                correct_residuals += usize::from(correct != i64::from(gamma[alpha][input]));
                wrong_sign_residuals += usize::from(wrong != i64::from(gamma[alpha][input]));
            }
        }
    }
    (correct_residuals, wrong_sign_residuals)
}

/// Eq. (41) is printed for the inverse elfbein.  Inverting it and forming the
/// covariant metric gives the plus-i symmetric formula used here.
fn eq41_metric_descendant_column(frame_column: usize) -> Sparse {
    let vector = frame_column / SPINOR_DIMENSION;
    let gamma = frame_column % SPINOR_DIMENSION;
    let gammas = real_gamma_matrices();
    let symmetric = symmetric_pairs();
    let mut output = Sparse::new();
    for alpha in 0..SPINOR_DIMENSION {
        for (field, &(left, right)) in symmetric.iter().enumerate() {
            let mut coefficient = 0_i64;
            if right == vector {
                coefficient += lowered_gamma_entry(&gammas, left, alpha, gamma);
            }
            if left == vector {
                coefficient += lowered_gamma_entry(&gammas, right, alpha, gamma);
            }
            if coefficient != 0 {
                add(
                    &mut output,
                    alpha * SYMMETRIC_DIMENSION + field,
                    ExactQi::i().scaled(&Ratio::from_integer(coefficient)),
                );
            }
        }
    }
    output
}

fn apply_graviton_curvature(metric_descendant: &Sparse, momentum_axis: usize) -> Sparse {
    let curvature = &target_sector_complex(TargetSector::Graviton).curvature;
    let mut output = Sparse::new();
    for (&coordinate, value) in metric_descendant {
        let alpha = coordinate / SYMMETRIC_DIMENSION;
        let field = coordinate % SYMMETRIC_DIMENSION;
        for (row, coefficient) in curvature.column_terms(field) {
            let coefficient = fixed_coefficient(&coefficient, momentum_axis);
            add(
                &mut output,
                alpha * RIEMANN_AMBIENT_DIMENSION + row,
                multiply(value, &coefficient),
            );
        }
    }
    output
}

fn oriented_pair_value(
    curl: &Sparse,
    pair_indices: &BTreeMap<(usize, usize), usize>,
    left: usize,
    right: usize,
    spinor: usize,
) -> ExactQi {
    if left == right {
        return ExactQi::zero();
    }
    let (pair, sign) = if left < right {
        ((left, right), 1_i64)
    } else {
        ((right, left), -1_i64)
    };
    curl.get(&(pair_indices[&pair] * SPINOR_DIMENSION + spinor))
        .cloned()
        .unwrap_or_else(ExactQi::zero)
        .scaled(&Ratio::from_integer(sign))
}

/// Curl the gamma term in Eq. (3.1f) through Eq. (3.2b).  The repository
/// Riemann is twice the conventional curvature, cancelling the 1/2 in the
/// spin connection.  This deliberately uses the unsimplified six-term form,
/// rather than assuming the curl Bianchi identity during construction.
fn eq31f_riemann_descendant(curl: &Sparse, momentum_axis: usize) -> Sparse {
    let pairs = combinations(2);
    let pair_indices = pairs
        .iter()
        .enumerate()
        .map(|(index, pair)| ((pair[0], pair[1]), index))
        .collect::<BTreeMap<_, _>>();
    let gammas = real_gamma_matrices();
    let mut output = Sparse::new();
    for alpha in 0..SPINOR_DIMENSION {
        for (left_pair, ab) in pairs.iter().enumerate() {
            let (a, b) = (ab[0], ab[1]);
            for (right_pair, cd) in pairs.iter().enumerate() {
                let (c, d) = (cd[0], cd[1]);
                for gamma in 0..SPINOR_DIMENSION {
                    let mut term = ExactQi::zero();
                    for (momentum, gamma_axis, q_left, q_right, sign) in [
                        (a, d, b, c, -1_i64),
                        (a, c, b, d, 1_i64),
                        (a, b, c, d, 1_i64),
                        (b, d, a, c, 1_i64),
                        (b, c, a, d, -1_i64),
                        (b, a, c, d, -1_i64),
                    ] {
                        if momentum != momentum_axis {
                            continue;
                        }
                        let gamma_entry = lowered_gamma_entry(&gammas, gamma_axis, alpha, gamma);
                        if gamma_entry == 0 {
                            continue;
                        }
                        let q = oriented_pair_value(curl, &pair_indices, q_left, q_right, gamma);
                        term.add_assign(
                            &q.times_i().scaled(&Ratio::from_integer(sign * gamma_entry)),
                        );
                    }
                    add(
                        &mut output,
                        alpha * RIEMANN_AMBIENT_DIMENSION + left_pair * PAIR_DIMENSION + right_pair,
                        term,
                    );
                }
            }
        }
    }
    output
}

fn sparse_difference_entries(left: &Sparse, right: &Sparse) -> usize {
    left.keys()
        .chain(right.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|row| left.get(row) != right.get(row))
        .count()
}

fn local_connection_curl_residual(momentum_axis: usize) -> usize {
    let pairs = combinations(2);
    let mut residual = 0_usize;
    // Eq. (3.1f)'s E_[b C_{alpha c]}{}^d term induces
    // D omega_bcd=p_b L_cd.  Its outer curl vanishes for every L_cd.
    for connection_pair in &pairs {
        for ab in &pairs {
            for cd in &pairs {
                let omega_at_b = i64::from(ab[1] == momentum_axis)
                    * i64::from(cd.as_slice() == connection_pair.as_slice());
                let omega_at_a = i64::from(ab[0] == momentum_axis)
                    * i64::from(cd.as_slice() == connection_pair.as_slice());
                let value = 2
                    * (i64::from(ab[0] == momentum_axis) * omega_at_b
                        - i64::from(ab[1] == momentum_axis) * omega_at_a);
                residual += usize::from(value != 0);
            }
        }
    }
    residual
}

fn apply_target_bianchi(
    columns: &[Sparse],
    sector: TargetSector,
    component_dimension: usize,
    momentum_axis: usize,
) -> usize {
    let bianchi = &target_sector_complex(sector).bianchi;
    let mut residual = 0_usize;
    for column in columns {
        let mut output = Sparse::new();
        for (&coordinate, value) in column {
            let derivative = coordinate / component_dimension;
            let component = coordinate % component_dimension;
            for (row, coefficient) in bianchi.column_terms(component) {
                let coefficient = fixed_coefficient(&coefficient, momentum_axis);
                add(
                    &mut output,
                    derivative * bianchi.rows() + row,
                    multiply(value, &coefficient),
                );
            }
        }
        residual += output.len();
    }
    residual
}

fn mod_inverse(value: u32) -> u32 {
    let mut base = u64::from(value);
    let mut exponent = PRIME - 2;
    let modulus = u64::from(PRIME);
    let mut output = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = output * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    output as u32
}

fn rational_mod(value: &Ratio<i64>) -> Result<u32, String> {
    let denominator = value.denom().rem_euclid(i64::from(PRIME)) as u32;
    if denominator == 0 {
        return Err("inadmissible denominator in graviton-relative rank".to_string());
    }
    let numerator = value.numer().rem_euclid(i64::from(PRIME)) as u32;
    Ok((u64::from(numerator) * u64::from(mod_inverse(denominator)) % u64::from(PRIME)) as u32)
}

fn modular_rank(columns: &[Sparse], imaginary_channel: bool) -> Result<usize, String> {
    let mut pivots = BTreeMap::<usize, BTreeMap<usize, u32>>::new();
    for column in columns {
        let mut reduced = BTreeMap::new();
        for (&row, value) in column {
            let selected = if imaginary_channel {
                if !value.real.is_zero() {
                    return Err("expected pure-imaginary graviton descendant".to_string());
                }
                &value.imaginary
            } else {
                if !value.imaginary.is_zero() {
                    return Err("expected real curl or D G4 descendant".to_string());
                }
                &value.real
            };
            let residue = rational_mod(selected)?;
            if residue != 0 {
                reduced.insert(row, residue);
            }
        }
        loop {
            let Some((&row, &value)) = reduced.first_key_value() else {
                break;
            };
            let Some(pivot) = pivots.get(&row) else {
                let inverse = mod_inverse(value);
                for entry in reduced.values_mut() {
                    *entry = (u64::from(*entry) * u64::from(inverse) % u64::from(PRIME)) as u32;
                }
                pivots.insert(row, reduced);
                break;
            };
            let factor = value;
            for (&target, &coefficient) in pivot {
                let subtraction = u64::from(factor) * u64::from(coefficient) % u64::from(PRIME);
                let current = reduced.get(&target).copied().unwrap_or(0);
                let next = (u64::from(current) + u64::from(PRIME) - subtraction) % u64::from(PRIME);
                if next == 0 {
                    reduced.remove(&target);
                } else {
                    reduced.insert(target, next as u32);
                }
            }
        }
    }
    Ok(pivots.len())
}

fn hash_columns(label: &[u8], columns: &[Sparse]) -> String {
    let mut hash = Sha256::new();
    hash.update(label);
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

pub(crate) fn build_graviton_relative_oracle_report() -> Result<GravitonRelativeOracleReport, String>
{
    let momentum_axis = 0;
    let curl_columns = frame_to_curl_columns(momentum_axis);
    let metric_columns = (0..FRAME_DIMENSION)
        .map(eq41_metric_descendant_column)
        .collect::<Vec<_>>();
    let eq41_riemann_columns = metric_columns
        .iter()
        .map(|column| apply_graviton_curvature(column, momentum_axis))
        .collect::<Vec<_>>();
    let eq31f_riemann_columns = curl_columns
        .iter()
        .map(|column| eq31f_riemann_descendant(column, momentum_axis))
        .collect::<Vec<_>>();
    let parity_residual = eq41_riemann_columns
        .iter()
        .zip(&eq31f_riemann_columns)
        .map(|(left, right)| sparse_difference_entries(left, right))
        .sum();

    let eq31g = linearized_gravitino_curl_to_d_f_four_operator();
    let dg4_columns = curl_columns
        .iter()
        .map(|column| eq31g.apply_sparse(column))
        .collect::<Vec<_>>();
    let curl_rank = modular_rank(&curl_columns, false)?;
    let graviton_rank = modular_rank(&eq41_riemann_columns, true)?;
    let dg4_rank = modular_rank(&dg4_columns, false)?;
    let graviton_bianchi = apply_target_bianchi(
        &eq41_riemann_columns,
        TargetSector::Graviton,
        RIEMANN_AMBIENT_DIMENSION,
        momentum_axis,
    );
    let dg4_bianchi = apply_target_bianchi(
        &dg4_columns,
        TargetSector::FourForm,
        FOUR_FORM_DIMENSION,
        momentum_axis,
    );
    let mutated = eq41_riemann_columns
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|(&row, value)| (row, value.scaled(&Ratio::from_integer(-1))))
                .collect::<Sparse>()
        })
        .collect::<Vec<_>>();
    let mutation_residual = mutated
        .iter()
        .zip(&eq31f_riemann_columns)
        .map(|(left, right)| sparse_difference_entries(left, right))
        .sum();
    let connection_residual = local_connection_curl_residual(momentum_axis);
    let (charge_adapter_residual, charge_adapter_mutation_residual) =
        charge_row_adapter_residuals();

    let passed = curl_rank == 320
        && graviton_rank == 320
        && dg4_rank == 320
        && parity_residual == 0
        && charge_adapter_residual == 0
        && charge_adapter_mutation_residual > 0
        && connection_residual == 0
        && graviton_bianchi == 0
        && dg4_bianchi == 0
        && mutation_residual > 0;
    let source_pdf_sha256 = BTreeMap::from([
        ("hep-th/0101037v4", HEP_TH_0101037_PDF_SHA256),
        ("hep-th/0107155v2", HEP_TH_0107155_PDF_SHA256),
    ]);

    Ok(GravitonRelativeOracleReport {
        schema_version: "adynkra-11d-graviton-relative-oracle-v1",
        momentum_axis,
        component_frame_dimension: FRAME_DIMENSION,
        component_curl_dimension: CURL_DIMENSION,
        component_curl_rank: curl_rank,
        eq41_metric_descendant_dimensions: (
            SPINOR_DIMENSION * SYMMETRIC_DIMENSION,
            FRAME_DIMENSION,
        ),
        graviton_riemann_descendant_dimensions: (
            SPINOR_DIMENSION * RIEMANN_AMBIENT_DIMENSION,
            FRAME_DIMENSION,
        ),
        eq31f_spin_connection_descendant_dimensions: (
            SPINOR_DIMENSION * RIEMANN_AMBIENT_DIMENSION,
            CURL_DIMENSION,
        ),
        eq31f_eq41_all_row_residual_entries: parity_residual,
        eq41_charge_row_adapter_residual_entries: charge_adapter_residual,
        eq41_charge_row_adapter_mutation_residual_entries: charge_adapter_mutation_residual,
        eq31f_local_connection_curl_residual_entries: connection_residual,
        graviton_bianchi_residual_entries: graviton_bianchi,
        graviton_normalization_mutation_residual_entries: mutation_residual,
        graviton_descendant_rank: graviton_rank,
        eq31g_descendant_rank: dg4_rank,
        eq31g_bianchi_residual_entries: dg4_bianchi,
        eq31f_printed_curl_coefficient: "-i",
        eq31g_printed_curl_coefficient: "-1/8",
        printed_coefficient_ratio_eq31g_over_eq31f: "-i/8",
        repository_riemann_is_twice_conventional: true,
        common_curl_basis_sha256: hash_columns(b"component-curl-p0-v1", &curl_columns),
        eq41_riemann_stream_sha256: hash_columns(
            b"eq41-metric-to-repo-riemann-p0-v1",
            &eq41_riemann_columns,
        ),
        eq31f_riemann_stream_sha256: hash_columns(
            b"eq31f-spin-connection-to-repo-riemann-p0-v1",
            &eq31f_riemann_columns,
        ),
        eq31g_dg4_stream_sha256: hash_columns(b"eq31g-dg4-p0-v1", &dg4_columns),
        source_pdf_sha256,
        oracle_source_sha256: format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "eleven_dimensional_graviton_relative_oracle.rs"
            ))
        ),
        ordinary_on_shell_relative_normalization_fixed: passed,
        off_shell_hhat_source_normalization_fixed: false,
        passed,
        boundary: "Passing fixes the ordinary on-shell component target normalization of the Eq3.1f/Eq41 graviton descendant relative to the Eq3.1g A3 descendant because both use the identical independent gravitino curl. It does not normalize an off-shell Hhat source, identify Eq40 Psi_[3] with A3, or include the J and X corrections in Eq41.",
    })
}

pub(crate) fn write_artifact(path: &Path) -> io::Result<GravitonRelativeOracleReport> {
    let report = build_graviton_relative_oracle_report()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "graviton-relative oracle failed an exact gate",
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    {
        let file = File::create(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq41_and_eq31f_fix_the_same_graviton_descendant() {
        let report = build_graviton_relative_oracle_report().unwrap();
        eprintln!("GRAVITON_RELATIVE_ORACLE {report:#?}");
        assert!(report.passed);
        assert_eq!(report.eq31f_eq41_all_row_residual_entries, 0);
        assert_eq!(report.eq31f_local_connection_curl_residual_entries, 0);
        assert_eq!(report.graviton_descendant_rank, 320);
        assert_eq!(report.eq31g_descendant_rank, 320);
        assert!(report.graviton_normalization_mutation_residual_entries > 0);
        assert!(!report.off_shell_hhat_source_normalization_fixed);
    }
}
