//! Exact fixed-momentum fiber product of the independent `D A_[3] -> D G_[4]`
//! map and the linearized Eq. (3.1g) gravitino-curl descendant.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use num_traits::{One, Zero};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_four_form_56_gpu::PINNED_PRIMES;
use crate::eleven_dimensional_h_hat_jet::canonical_gamma_traceless_frame_basis;
use crate::eleven_dimensional_independent_a3_adapter::{A3_DIMENSION, d_g4_column};
use crate::eleven_dimensional_physical_curvature::{
    ExactQi, cached_linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_target_equation_complex::{TargetSector, target_sector_complex};

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const FOUR_FORM_DIMENSION: usize = 330;
const CURL_DIMENSION: usize = 55 * SPINOR_DIMENSION;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct FixedMomentumFiberRank {
    pub momentum_axis: usize,
    pub da3_image_rank: usize,
    pub curl_image_rank: usize,
    pub outside_rows: usize,
    pub outside_nonzeros: usize,
    pub outside_rank_by_prime: [usize; 3],
    pub intersection_dimension_by_prime: [usize; 3],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct FixedMomentumExactPairing {
    pub momentum_axis: usize,
    pub paired_maps: usize,
    pub curl_basis_rank_by_prime: [usize; 3],
    pub outside_support_residual_entries: usize,
    pub exact_forward_replay_residual_entries: usize,
    pub first_curl_coordinate: usize,
    pub first_curl_value: String,
    pub first_dg4_coordinate: usize,
    pub first_dg4_value: String,
    pub first_da3_coordinate: usize,
    pub first_da3_value: String,
    pub paired_stream_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct A3CurlFiberProductReport {
    pub schema_version: &'static str,
    pub pinned_primes: [u32; 3],
    pub momentum_fibers: Vec<FixedMomentumFiberRank>,
    pub exact_pairings: Vec<FixedMomentumExactPairing>,
    pub characteristic_zero_intersection_dimension: usize,
    pub denominator_admissible_at_all_pinned_primes: bool,
    pub normalization_canary_passed: bool,
    pub source_sha256: BTreeMap<String, String>,
    pub passed: bool,
    pub scope_boundary: &'static str,
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

fn exact_real_mod(value: &ExactQi, prime: u32) -> Result<u32, String> {
    if !value.imaginary.is_zero() {
        return Err("Eq. (3.1g) fiber matrix unexpectedly has an imaginary entry".to_string());
    }
    let numerator = value.real.numer().rem_euclid(i64::from(prime)) as u32;
    let denominator = value.real.denom().rem_euclid(i64::from(prime)) as u32;
    if denominator == 0 {
        return Err("Eq. (3.1g) denominator is inadmissible at a pinned prime".to_string());
    }
    Ok(
        (u64::from(numerator) * u64::from(pow_mod(denominator, prime - 2, prime))
            % u64::from(prime)) as u32,
    )
}

fn sub_mod(left: u32, right: u32, prime: u32) -> u32 {
    if left >= right {
        left - right
    } else {
        (u64::from(left) + u64::from(prime) - u64::from(right)) as u32
    }
}

fn add_to(row: &mut BTreeMap<usize, u32>, column: usize, value: u32, prime: u32) {
    if value == 0 {
        return;
    }
    let sum =
        (u64::from(row.get(&column).copied().unwrap_or(0)) + u64::from(value)) % u64::from(prime);
    if sum == 0 {
        row.remove(&column);
    } else {
        row.insert(column, sum as u32);
    }
}

fn sparse_rank(mut rows: Vec<BTreeMap<usize, u32>>, columns: usize, prime: u32) -> usize {
    rows.sort_by_key(BTreeMap::len);
    let mut pivots = vec![None::<BTreeMap<usize, u32>>; columns];
    let mut rank = 0;
    for mut row in rows {
        loop {
            let Some((&pivot, &value)) = row.first_key_value() else {
                break;
            };
            let Some(existing) = pivots[pivot].as_ref() else {
                let inverse = pow_mod(value, prime - 2, prime);
                for coefficient in row.values_mut() {
                    *coefficient =
                        (u64::from(*coefficient) * u64::from(inverse) % u64::from(prime)) as u32;
                }
                pivots[pivot] = Some(row);
                rank += 1;
                break;
            };
            let terms = existing
                .iter()
                .map(|(&column, &coefficient)| {
                    (
                        column,
                        (u64::from(value) * u64::from(coefficient) % u64::from(prime)) as u32,
                    )
                })
                .collect::<Vec<_>>();
            for (column, product) in terms {
                let current = row.get(&column).copied().unwrap_or(0);
                let next = sub_mod(current, product, prime);
                if next == 0 {
                    row.remove(&column);
                } else {
                    row.insert(column, next);
                }
            }
        }
    }
    rank
}

fn exact_multiply(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn exact_add(output: &mut BTreeMap<usize, ExactQi>, row: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(row).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&row);
    }
}

fn exact_hash_map(hash: &mut Sha256, label: u64, map: &BTreeMap<usize, ExactQi>) {
    hash.update(label.to_le_bytes());
    hash.update((map.len() as u64).to_le_bytes());
    for (&row, value) in map {
        hash.update((row as u64).to_le_bytes());
        hash.update(value.real.numer().to_le_bytes());
        hash.update(value.real.denom().to_le_bytes());
        hash.update(value.imaginary.numer().to_le_bytes());
        hash.update(value.imaginary.denom().to_le_bytes());
    }
}

fn exact_string(value: &ExactQi) -> String {
    format!("{}+({})i", value.real, value.imaginary)
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn fixed_momentum_curl_from_frame(
    momentum_axis: usize,
    frame: &BTreeMap<usize, ExactQi>,
) -> BTreeMap<usize, ExactQi> {
    let curvature = &target_sector_complex(TargetSector::RaritaSchwinger).curvature;
    let mut output = BTreeMap::new();
    for (&h_coordinate, value) in frame {
        let spinor = h_coordinate / VECTOR_DIMENSION;
        let vector = h_coordinate % VECTOR_DIMENSION;
        let frame_coordinate = vector * SPINOR_DIMENSION + spinor;
        let lowered = if vector == 0 {
            value.scaled(&num_rational::Ratio::from_integer(-1))
        } else {
            value.clone()
        };
        for (row, term) in curvature.column_terms(frame_coordinate) {
            if term.monomial.exponents[momentum_axis] != 1
                || term
                    .monomial
                    .exponents
                    .iter()
                    .enumerate()
                    .any(|(axis, &power)| axis != momentum_axis && power != 0)
            {
                continue;
            }
            let coefficient = ExactQi {
                real: num_rational::Ratio::new(term.real_numerator, term.real_denominator),
                imaginary: num_rational::Ratio::new(
                    term.imaginary_numerator,
                    term.imaginary_denominator,
                ),
            };
            exact_add(&mut output, row, exact_multiply(&lowered, &coefficient));
        }
    }
    output
}

fn modular_column_rank(
    columns: &[BTreeMap<usize, ExactQi>],
    rows: usize,
    prime: u32,
) -> Result<usize, String> {
    let mut matrix_rows = vec![BTreeMap::new(); rows];
    for (column, entries) in columns.iter().enumerate() {
        for (&row, value) in entries {
            add_to(
                &mut matrix_rows[row],
                column,
                exact_real_mod(value, prime)?,
                prime,
            );
        }
    }
    Ok(sparse_rank(matrix_rows, columns.len(), prime))
}

pub(crate) fn fixed_momentum_exact_pairing(
    momentum_axis: usize,
) -> Result<FixedMomentumExactPairing, String> {
    if momentum_axis >= VECTOR_DIMENSION {
        return Err("fiber-product momentum axis is outside 0..11".to_string());
    }
    let four_forms = combinations(4);
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    let curl_columns = canonical_gamma_traceless_frame_basis()
        .iter()
        .map(|frame| fixed_momentum_curl_from_frame(momentum_axis, frame))
        .collect::<Vec<_>>();
    let curl_basis_rank_by_prime = std::array::from_fn(|slot| {
        modular_column_rank(&curl_columns, CURL_DIMENSION, PINNED_PRIMES[slot]).unwrap()
    });

    let mut da3_inverse = BTreeMap::<usize, (usize, ExactQi)>::new();
    for derivative in 0..SPINOR_DIMENSION {
        for a3 in 0..A3_DIMENSION {
            for term in d_g4_column(derivative, a3)? {
                if term.momentum_exponents[momentum_axis] != 1
                    || term
                        .momentum_exponents
                        .iter()
                        .enumerate()
                        .any(|(axis, &power)| axis != momentum_axis && power != 0)
                {
                    continue;
                }
                let coefficient = ExactQi {
                    real: num_rational::Ratio::new(term.real_numerator, term.real_denominator),
                    imaginary: num_rational::Ratio::new(
                        term.imaginary_numerator,
                        term.imaginary_denominator,
                    ),
                };
                if da3_inverse
                    .insert(
                        term.target_coordinate,
                        (derivative * A3_DIMENSION + a3, coefficient),
                    )
                    .is_some()
                {
                    return Err("fixed-momentum D A3 image has a duplicate target row".to_string());
                }
            }
        }
    }
    if da3_inverse.len() != SPINOR_DIMENSION * 120 {
        return Err("fixed-momentum D A3 image rank is not 3,840".to_string());
    }

    let mut outside_support_residual_entries = 0;
    let mut exact_forward_replay_residual_entries = 0;
    let mut first_curl = None;
    let mut first_dg4 = None;
    let mut first_da3 = None;
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-a3-curl-fiber-pair-v1\0");
    hash.update((momentum_axis as u64).to_le_bytes());
    for (ordinal, curl) in curl_columns.iter().enumerate() {
        let descendant = operator.apply_sparse(curl);
        let mut da3 = BTreeMap::new();
        for (&row, value) in &descendant {
            let form = row % FOUR_FORM_DIMENSION;
            if !four_forms[form].contains(&momentum_axis) {
                outside_support_residual_entries += 1;
                continue;
            }
            let Some(&(coordinate, ref coefficient)) = da3_inverse.get(&row) else {
                outside_support_residual_entries += 1;
                continue;
            };
            if !coefficient.imaginary.is_zero() || coefficient.real.is_zero() {
                return Err(
                    "fixed-momentum D A3 inverse coefficient is not real nonzero".to_string(),
                );
            }
            exact_add(
                &mut da3,
                coordinate,
                value.scaled(&(num_rational::Ratio::from_integer(1) / coefficient.real.clone())),
            );
        }
        let mut replay = BTreeMap::new();
        for (&coordinate, value) in &da3 {
            let derivative = coordinate / A3_DIMENSION;
            let a3 = coordinate % A3_DIMENSION;
            for term in d_g4_column(derivative, a3)? {
                if term.momentum_exponents[momentum_axis] != 1
                    || term
                        .momentum_exponents
                        .iter()
                        .enumerate()
                        .any(|(axis, &power)| axis != momentum_axis && power != 0)
                {
                    continue;
                }
                let coefficient = ExactQi {
                    real: num_rational::Ratio::new(term.real_numerator, term.real_denominator),
                    imaginary: num_rational::Ratio::new(
                        term.imaginary_numerator,
                        term.imaginary_denominator,
                    ),
                };
                exact_add(
                    &mut replay,
                    term.target_coordinate,
                    exact_multiply(value, &coefficient),
                );
            }
        }
        exact_forward_replay_residual_entries += descendant
            .keys()
            .chain(replay.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|row| descendant.get(row) != replay.get(row))
            .count();
        exact_hash_map(&mut hash, ordinal as u64, curl);
        exact_hash_map(&mut hash, ordinal as u64 | (1_u64 << 63), &da3);
        if first_curl.is_none() {
            first_curl = curl
                .first_key_value()
                .map(|(&coordinate, value)| (coordinate, exact_string(value)));
            first_dg4 = descendant
                .first_key_value()
                .map(|(&coordinate, value)| (coordinate, exact_string(value)));
            first_da3 = da3
                .first_key_value()
                .map(|(&coordinate, value)| (coordinate, exact_string(value)));
        }
    }
    let (first_curl_coordinate, first_curl_value) = first_curl
        .ok_or_else(|| "canonical physical curl basis starts with a zero map".to_string())?;
    let (first_dg4_coordinate, first_dg4_value) = first_dg4
        .ok_or_else(|| "canonical physical DG4 descendant starts with a zero map".to_string())?;
    let (first_da3_coordinate, first_da3_value) =
        first_da3.ok_or_else(|| "canonical paired DA3 map starts with a zero map".to_string())?;
    Ok(FixedMomentumExactPairing {
        momentum_axis,
        paired_maps: curl_columns.len(),
        curl_basis_rank_by_prime,
        outside_support_residual_entries,
        exact_forward_replay_residual_entries,
        first_curl_coordinate,
        first_curl_value,
        first_dg4_coordinate,
        first_dg4_value,
        first_da3_coordinate,
        first_da3_value,
        paired_stream_sha256: format!("{:x}", hash.finalize()),
    })
}

pub(crate) fn fixed_momentum_fiber_rank(
    momentum_axis: usize,
) -> Result<FixedMomentumFiberRank, String> {
    if momentum_axis >= VECTOR_DIMENSION {
        return Err("fiber-product momentum axis is outside 0..11".to_string());
    }
    let four_forms = combinations(4);
    let outside_ordinals = four_forms
        .iter()
        .enumerate()
        .filter_map(|(ordinal, axes)| (!axes.contains(&momentum_axis)).then_some(ordinal))
        .collect::<Vec<_>>();
    let outside_lookup = outside_ordinals
        .iter()
        .enumerate()
        .map(|(row, &ordinal)| (ordinal, row))
        .collect::<BTreeMap<_, _>>();
    let outside_rows = SPINOR_DIMENSION * outside_ordinals.len();
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    let outside_nonzeros = operator
        .columns
        .iter()
        .flat_map(|column| column.iter())
        .filter(|entry| outside_lookup.contains_key(&(entry.row % FOUR_FORM_DIMENSION)))
        .count();
    let mut ranks = [0_usize; 3];
    for (prime_slot, prime) in PINNED_PRIMES.into_iter().enumerate() {
        let mut rows = vec![BTreeMap::new(); outside_rows];
        for (column, entries) in operator.columns.iter().enumerate() {
            for entry in entries {
                let spinor = entry.row / FOUR_FORM_DIMENSION;
                let form = entry.row % FOUR_FORM_DIMENSION;
                let Some(&outside_form) = outside_lookup.get(&form) else {
                    continue;
                };
                let row = spinor * outside_ordinals.len() + outside_form;
                add_to(
                    &mut rows[row],
                    column,
                    exact_real_mod(&entry.coefficient, prime)?,
                    prime,
                );
            }
        }
        ranks[prime_slot] = sparse_rank(rows, CURL_DIMENSION, prime);
    }
    Ok(FixedMomentumFiberRank {
        momentum_axis,
        da3_image_rank: SPINOR_DIMENSION * 120,
        curl_image_rank: CURL_DIMENSION,
        outside_rows,
        outside_nonzeros,
        outside_rank_by_prime: ranks,
        intersection_dimension_by_prime: ranks.map(|rank| CURL_DIMENSION - rank),
    })
}

pub(crate) fn build_report() -> Result<A3CurlFiberProductReport, String> {
    let momentum_fibers = (0..VECTOR_DIMENSION)
        .map(fixed_momentum_fiber_rank)
        .collect::<Result<Vec<_>, _>>()?;
    let exact_pairings = (0..VECTOR_DIMENSION)
        .map(fixed_momentum_exact_pairing)
        .collect::<Result<Vec<_>, _>>()?;
    let source_paths = [
        "src/eleven_dimensional_a3_curl_fiber_product.rs",
        "src/eleven_dimensional_independent_a3_adapter.rs",
        "src/eleven_dimensional_physical_curvature.rs",
        "src/eleven_dimensional_h_hat_jet.rs",
        "src/eleven_dimensional_target_equation_complex.rs",
    ];
    let source_sha256 = source_paths
        .into_iter()
        .map(|path| Ok((path.to_string(), file_sha256(Path::new(path))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let normalization_canary_passed = exact_pairings.first().is_some_and(|pairing| {
        pairing.first_curl_coordinate == 0
            && pairing.first_curl_value == "1+(0)i"
            && pairing.first_dg4_coordinate == 330
            && pairing.first_dg4_value == "-1/2+(0)i"
            && pairing.first_da3_coordinate == 210
            && pairing.first_da3_value == "-1/2+(0)i"
    });
    let passed = momentum_fibers.iter().all(|fiber| {
        fiber.da3_image_rank == 3_840
            && fiber.curl_image_rank == 1_760
            && fiber.outside_rows == 6_720
            && fiber.outside_nonzeros == 40_320
            && fiber.outside_rank_by_prime == [1_440; 3]
            && fiber.intersection_dimension_by_prime == [320; 3]
    }) && exact_pairings.iter().all(|pairing| {
        pairing.paired_maps == 320
            && pairing.curl_basis_rank_by_prime == [320; 3]
            && pairing.outside_support_residual_entries == 0
            && pairing.exact_forward_replay_residual_entries == 0
    }) && normalization_canary_passed;
    Ok(A3CurlFiberProductReport {
        schema_version: "adynkra-11d-a3-curl-fiber-product-v1",
        pinned_primes: PINNED_PRIMES,
        momentum_fibers,
        exact_pairings,
        characteristic_zero_intersection_dimension: 320,
        denominator_admissible_at_all_pinned_primes: true,
        normalization_canary_passed,
        source_sha256,
        passed,
        scope_boundary: "This certifies the independent fixed-momentum DA3/Eq3.1g-curl fiber product and its exact characteristic-zero paired basis. It does not identify an Hhat source map, invoke Eq25, or prove irreducibility.",
    })
}

pub(crate) fn write_report(path: &Path) -> Result<A3CurlFiberProductReport, String> {
    let report = build_report()?;
    if !report.passed {
        return Err("A3/curl fiber product report failed its scientific gates".to_string());
    }
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize A3/curl fiber product report: {error}"))?;
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
    fn fixed_momentum_a3_curl_fiber_rank_three_primes() {
        for momentum in 0..VECTOR_DIMENSION {
            let report = fixed_momentum_fiber_rank(momentum).unwrap();
            eprintln!("A3_CURL_FIBER {report:?}");
            assert_eq!(report.outside_rows, SPINOR_DIMENSION * 210);
            assert_eq!(
                report.outside_rank_by_prime[0],
                report.outside_rank_by_prime[1]
            );
            assert_eq!(
                report.outside_rank_by_prime[1],
                report.outside_rank_by_prime[2]
            );
        }
    }

    #[test]
    fn fixed_momentum_exact_pairing_replays_over_qi() {
        for momentum in 0..VECTOR_DIMENSION {
            let report = fixed_momentum_exact_pairing(momentum).unwrap();
            eprintln!("A3_CURL_EXACT_PAIR {report:?}");
            assert_eq!(report.paired_maps, 320);
            assert_eq!(report.curl_basis_rank_by_prime, [320; 3]);
            assert_eq!(report.outside_support_residual_entries, 0);
            assert_eq!(report.exact_forward_replay_residual_entries, 0);
        }
    }

    #[test]
    #[ignore = "writes the authoritative report-last artifact"]
    fn write_authoritative_a3_curl_fiber_product_report() {
        let path = Path::new("results/adynkra_11d_a3_curl_fiber_product.json");
        let report = write_report(path).unwrap();
        assert!(report.passed);
        eprintln!("A3_CURL_FIBER_ARTIFACT_SHA {}", file_sha256(path).unwrap());
    }
}
