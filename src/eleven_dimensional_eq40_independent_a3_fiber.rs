//! Witness-first compatibility test for the corrected Eq. (40) Lambda3 ray.
//!
//! This module never constructs a gauge-fixed frame or calls Eq. (25).  It
//! treats the corrected conventional-compensator output as a candidate A3,
//! applies the independent target exterior derivative, and asks whether its
//! first spinor descendant lies in the exact hep-th/0107155 Eq. (3.1g) image
//! of an independent component gravitino curl.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_gamma24_source_variance::{
    H_HAT_DIMENSION, corrected_gamma_two_exterior_operator,
};
use crate::eleven_dimensional_physical_curvature::{
    D_F_FOUR_FORM_DIMENSION, ExactQi, GRAVITINO_CURL_DIMENSION, SPINOR_DIMENSION, VECTOR_DIMENSION,
    W_FOUR_FORM_DIMENSION, cached_linearized_gravitino_curl_to_d_f_four_operator,
};
use crate::eleven_dimensional_superderivative_normal_form::{
    CanonicalSuperPolynomial, OrderedSuperderivativeMonomial, left_multiply_d,
};
use crate::eleven_dimensional_target_equation_complex::{
    ExactPolynomialCoefficient, TargetSector, target_sector_complex,
};

const THREE_FORM_DIMENSION: usize = 165;
const PHYSICAL_FIBER_PATH: &str = "results/adynkra_11d_a3_curl_fiber_product.json";
const PHYSICAL_FIBER_SHA256: &str =
    "53f078a1189555734a9c48f674a0528f620460d0ffe8cd60d461f1533b13558a";
const HOM_INVENTORY_PATH: &str = "results/adynkra_11d_higher_bidegree_hom_inventory.json";
const HOM_INVENTORY_SHA256: &str =
    "0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f";

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct Eq40FiberMonomialKey {
    pub exterior_spinor_mask: u32,
    pub momentum_exponents: [u16; VECTOR_DIMENSION],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ExactPublic {
    pub real: [i64; 2],
    pub imaginary: [i64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct Eq40FiberWitness {
    pub monomial: Eq40FiberMonomialKey,
    pub target_coordinate: usize,
    pub candidate: ExactPublic,
    pub reconstructed_image: ExactPublic,
    pub residual: ExactPublic,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct Eq40FiberBidegreeDecision {
    pub bidegree: [usize; 2],
    pub candidate_rows: usize,
    pub monomial_slices: usize,
    pub on_image_slices: usize,
    pub off_image_slices: usize,
    pub recovered_curl_rows: usize,
    pub exact_image_residual_rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct Eq40IndependentA3FiberReport {
    pub schema_version: &'static str,
    pub source_ordinal: usize,
    pub typed_source: &'static str,
    pub candidate_potential_bidegree: [usize; 2],
    pub compared_descendant_bidegree: [usize; 2],
    pub pbw_descendant_bidegrees: [[usize; 2]; 2],
    pub target_join: &'static str,
    pub derivative_index_contract: &'static str,
    pub contracted_inner_beta_and_free_outer_alpha: bool,
    pub derivative_index_relabel_used: bool,
    pub gauge_quotient: &'static str,
    pub hhat_lambda3_hom_dimension: usize,
    pub full_h_lambda3_hom_dimension: usize,
    pub general_d2_p1_dg4_hom_dimension: usize,
    pub general_d0_p2_dg4_hom_dimension: usize,
    pub corrected_lambda3_complete_for_stated_slice: bool,
    pub gamma24_source_variance_passed: bool,
    pub corrected_lambda3_operator_rank: usize,
    pub eq31g_operator_dimensions: (usize, usize),
    pub eq31g_operator_nonzero_entries: usize,
    pub projector_in_image_canary_residual_rows: usize,
    pub projector_off_image_mutation_residual_rows: usize,
    pub physical_fiber_certificate_path: &'static str,
    pub physical_fiber_certificate_sha256: String,
    pub physical_fiber_certificate_expected_sha256: &'static str,
    pub physical_fiber_certificate_hash_matches: bool,
    pub physical_fiber_certificate_passed: bool,
    pub physical_fiber_intersection_dimension: usize,
    pub hom_inventory_path: &'static str,
    pub hom_inventory_sha256: String,
    pub hom_inventory_expected_sha256: &'static str,
    pub hom_inventory_hash_matches: bool,
    pub candidate_rows: usize,
    pub pbw_monomial_slices: usize,
    pub on_image_slices: usize,
    pub off_image_slices: usize,
    pub recovered_curl_rows: usize,
    pub exact_image_residual_rows: usize,
    pub d2_p1: Eq40FiberBidegreeDecision,
    pub d0_p2: Eq40FiberBidegreeDecision,
    pub unexpected_bidegree_slices: usize,
    pub first_off_image_witness: Option<Eq40FiberWitness>,
    pub candidate_g4_bianchi_residual_rows: usize,
    pub candidate_stream_sha256: String,
    pub oracle_source_sha256: String,
    pub source_sha256: BTreeMap<String, String>,
    pub physical_fiber_compatible_on_unrestricted_hhat_slice: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

type Sparse = BTreeMap<usize, ExactQi>;

fn file_sha256(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn exact_certificate_value(path: &Path) -> Result<serde_json::Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn bidegree(key: &Eq40FiberMonomialKey) -> [usize; 2] {
    [
        key.exterior_spinor_mask.count_ones() as usize,
        key.momentum_exponents
            .iter()
            .map(|&power| usize::from(power))
            .sum(),
    ]
}

fn public(value: &ExactQi) -> ExactPublic {
    ExactPublic {
        real: [*value.real.numer(), *value.real.denom()],
        imaginary: [*value.imaginary.numer(), *value.imaginary.denom()],
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

fn coefficient(value: &ExactPolynomialCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(value.real_numerator, value.real_denominator),
        imaginary: Ratio::new(value.imaginary_numerator, value.imaginary_denominator),
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

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1, |value, index| value * (n - index) / (index + 1))
}

fn target_three_form_ordinal(mask: u16) -> usize {
    let indices = (0..VECTOR_DIMENSION)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    assert_eq!(indices.len(), 3);
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

fn numeric_three_form_masks() -> Vec<u16> {
    (0_u16..(1_u16 << VECTOR_DIMENSION))
        .filter(|mask| mask.count_ones() == 3)
        .collect()
}

fn multiply_momentum(
    monomial: &OrderedSuperderivativeMonomial,
    factor: &ExactPolynomialCoefficient,
) -> Result<[u16; VECTOR_DIMENSION], String> {
    let mut output = monomial.momentum.exponents;
    for (axis, exponent) in factor.monomial.exponents.into_iter().enumerate() {
        output[axis] = output[axis]
            .checked_add(u16::from(exponent))
            .ok_or_else(|| format!("momentum overflow on axis {axis}"))?;
    }
    Ok(output)
}

fn candidate_slices(
    source_ordinal: usize,
) -> Result<BTreeMap<Eq40FiberMonomialKey, Sparse>, String> {
    if source_ordinal >= H_HAT_DIMENSION {
        return Err(format!(
            "source ordinal {source_ordinal} is outside 0..{H_HAT_DIMENSION}"
        ));
    }
    let gamma_two = corrected_gamma_two_exterior_operator();
    let masks = numeric_three_form_masks();
    let curvature = &target_sector_complex(TargetSector::FourForm).curvature;
    let scalar = CanonicalSuperPolynomial::scalar(ExactQi::one());
    let inner = (0..SPINOR_DIMENSION)
        .map(|spinor| left_multiply_d(spinor, &scalar))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = BTreeMap::<Eq40FiberMonomialKey, Sparse>::new();
    for outer_spinor in 0..SPINOR_DIMENSION {
        for inner_spinor in 0..SPINOR_DIMENSION {
            let pbw = left_multiply_d(outer_spinor, &inner[inner_spinor])?;
            let source_column = inner_spinor * H_HAT_DIMENSION + source_ordinal;
            for gamma_entry in &gamma_two.columns[source_column] {
                let potential = target_three_form_ordinal(masks[gamma_entry.row]);
                let psi_coefficient = gamma_entry.coefficient.scaled(&Ratio::new(1, 16));
                for (four_form, curvature_term) in curvature.column_terms(potential) {
                    let map_coefficient = multiply(&psi_coefficient, &coefficient(&curvature_term));
                    for (monomial, pbw_coefficient) in &pbw.terms {
                        let key = Eq40FiberMonomialKey {
                            exterior_spinor_mask: monomial.exterior_spinor_mask,
                            momentum_exponents: multiply_momentum(monomial, &curvature_term)?,
                        };
                        add(
                            output.entry(key).or_default(),
                            outer_spinor * W_FOUR_FORM_DIMENSION + four_form,
                            multiply(pbw_coefficient, &map_coefficient),
                        );
                    }
                }
            }
        }
    }
    output.retain(|_, slice| !slice.is_empty());
    Ok(output)
}

fn unscaled_map(input: &Sparse) -> Sparse {
    cached_linearized_gravitino_curl_to_d_f_four_operator()
        .apply_sparse(input)
        .into_iter()
        .map(|(row, value)| (row, value.scaled(&Ratio::from_integer(-2))))
        .collect()
}

fn unscaled_transpose(input: &Sparse) -> Sparse {
    let operator = cached_linearized_gravitino_curl_to_d_f_four_operator();
    let mut output = Sparse::new();
    for (column, entries) in operator.columns.iter().enumerate() {
        for entry in entries {
            let Some(value) = input.get(&entry.row) else {
                continue;
            };
            add(
                &mut output,
                column,
                multiply(&entry.coefficient.scaled(&Ratio::from_integer(-2)), value),
            );
        }
    }
    output
}

fn unscaled_gram(input: &Sparse) -> Sparse {
    unscaled_transpose(&unscaled_map(input))
}

/// Exact polynomial left inverse used by the certified physical-curvature
/// module, returned without rejecting the off-image residual.
fn project_eq31g_image(input: &Sparse) -> (Sparse, Sparse, Sparse) {
    let z = unscaled_transpose(input);
    let gz = unscaled_gram(&z);
    let g2z = unscaled_gram(&gz);
    let mut recovered = Sparse::new();
    for (source, factor) in [(&g2z, 1_i64), (&gz, -321), (&z, 24_444)] {
        for (&coordinate, value) in source {
            add(
                &mut recovered,
                coordinate,
                value.scaled(&Ratio::new(-2 * factor, 381_024)),
            );
        }
    }
    let reconstructed =
        cached_linearized_gravitino_curl_to_d_f_four_operator().apply_sparse(&recovered);
    let mut residual = reconstructed.clone();
    for (&row, value) in input {
        add(&mut residual, row, value.scaled(&Ratio::from_integer(-1)));
    }
    (recovered, reconstructed, residual)
}

fn add_momentum(
    exponents: [u16; VECTOR_DIMENSION],
    factor: &ExactPolynomialCoefficient,
) -> Result<[u16; VECTOR_DIMENSION], String> {
    let mut output = exponents;
    for (axis, exponent) in factor.monomial.exponents.into_iter().enumerate() {
        output[axis] = output[axis]
            .checked_add(u16::from(exponent))
            .ok_or_else(|| format!("Bianchi momentum overflow on axis {axis}"))?;
    }
    Ok(output)
}

fn bianchi_residual(slices: &BTreeMap<Eq40FiberMonomialKey, Sparse>) -> Result<usize, String> {
    let bianchi = &target_sector_complex(TargetSector::FourForm).bianchi;
    let mut output = BTreeMap::<(u32, [u16; VECTOR_DIMENSION], usize), ExactQi>::new();
    for (key, slice) in slices {
        for (&coordinate, value) in slice {
            let derivative = coordinate / W_FOUR_FORM_DIMENSION;
            let form = coordinate % W_FOUR_FORM_DIMENSION;
            for (row, term) in bianchi.column_terms(form) {
                let target = (
                    key.exterior_spinor_mask,
                    add_momentum(key.momentum_exponents, &term)?,
                    derivative * bianchi.rows() + row,
                );
                let contribution = multiply(value, &coefficient(&term));
                let entry = output.entry(target).or_insert_with(ExactQi::zero);
                entry.add_assign(&contribution);
            }
        }
    }
    output.retain(|_, value| !value.is_zero());
    Ok(output.len())
}

fn hash_slices(slices: &BTreeMap<Eq40FiberMonomialKey, Sparse>) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-eq40-independent-a3-fiber-candidate-v1\0");
    for (key, slice) in slices {
        hash.update(key.exterior_spinor_mask.to_le_bytes());
        for exponent in key.momentum_exponents {
            hash.update(exponent.to_le_bytes());
        }
        hash.update((slice.len() as u64).to_le_bytes());
        for (&row, value) in slice {
            hash.update((row as u64).to_le_bytes());
            hash.update(value.real.numer().to_le_bytes());
            hash.update(value.real.denom().to_le_bytes());
            hash.update(value.imaginary.numer().to_le_bytes());
            hash.update(value.imaginary.denom().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

pub(crate) fn build_eq40_independent_a3_fiber_report(
    source_ordinal: usize,
) -> Result<Eq40IndependentA3FiberReport, String> {
    let slices = candidate_slices(source_ordinal)?;
    let physical_fiber_certificate_sha256 = file_sha256(Path::new(PHYSICAL_FIBER_PATH))?;
    let physical_fiber_certificate_hash_matches =
        physical_fiber_certificate_sha256 == PHYSICAL_FIBER_SHA256;
    let physical_fiber = exact_certificate_value(Path::new(PHYSICAL_FIBER_PATH))?;
    let physical_fiber_certificate_passed =
        physical_fiber["passed"] == true && physical_fiber["normalization_canary_passed"] == true;
    let physical_fiber_intersection_dimension =
        physical_fiber["characteristic_zero_intersection_dimension"]
            .as_u64()
            .ok_or_else(|| {
                "physical fiber certificate lacks a characteristic-zero intersection dimension"
                    .to_string()
            })? as usize;
    let hom_inventory_sha256 = file_sha256(Path::new(HOM_INVENTORY_PATH))?;
    let hom_inventory_hash_matches = hom_inventory_sha256 == HOM_INVENTORY_SHA256;
    let hom_inventory = exact_certificate_value(Path::new(HOM_INVENTORY_PATH))?;
    let general_d2_p1_dg4_hom_dimension = hom_inventory["descendant_targets"]["d2_p1_D_G4_total"]
        .as_u64()
        .ok_or_else(|| "Hom inventory lacks d2,p1 DG4 total".to_string())?
        as usize;
    let general_d0_p2_dg4_hom_dimension = hom_inventory["descendant_targets"]["d0_p2_D_G4_total"]
        .as_u64()
        .ok_or_else(|| "Hom inventory lacks d0,p2 DG4 total".to_string())?
        as usize;
    let gamma24 = crate::eleven_dimensional_gamma24_source_variance::verify_source_variance();
    let eq31g = cached_linearized_gravitino_curl_to_d_f_four_operator();
    let image_canary = BTreeMap::from([
        (0_usize, ExactQi::from_rational(3, 7)),
        (319, ExactQi::i()),
        (GRAVITINO_CURL_DIMENSION - 1, ExactQi::from_integer(-2)),
    ]);
    let image_canary_target = eq31g.apply_sparse(&image_canary);
    let (_, _, image_canary_residual) = project_eq31g_image(&image_canary_target);
    let (_, _, off_image_mutation_residual) =
        project_eq31g_image(&BTreeMap::from([(0_usize, ExactQi::one())]));
    let candidate_rows = slices.values().map(BTreeMap::len).sum();
    let mut on_image_slices = 0_usize;
    let mut off_image_slices = 0_usize;
    let mut recovered_curl_rows = 0_usize;
    let mut exact_image_residual_rows = 0_usize;
    let mut d2_p1 = Eq40FiberBidegreeDecision {
        bidegree: [2, 1],
        ..Eq40FiberBidegreeDecision::default()
    };
    let mut d0_p2 = Eq40FiberBidegreeDecision {
        bidegree: [0, 2],
        ..Eq40FiberBidegreeDecision::default()
    };
    let mut unexpected_bidegree_slices = 0_usize;
    let mut first_off_image_witness = None;
    for (key, candidate) in &slices {
        let (recovered, reconstructed, residual) = project_eq31g_image(candidate);
        recovered_curl_rows += recovered.len();
        exact_image_residual_rows += residual.len();
        let branch = match bidegree(key) {
            [2, 1] => Some(&mut d2_p1),
            [0, 2] => Some(&mut d0_p2),
            _ => {
                unexpected_bidegree_slices += 1;
                None
            }
        };
        if let Some(branch) = branch {
            branch.candidate_rows += candidate.len();
            branch.monomial_slices += 1;
            branch.recovered_curl_rows += recovered.len();
            branch.exact_image_residual_rows += residual.len();
            if residual.is_empty() {
                branch.on_image_slices += 1;
            } else {
                branch.off_image_slices += 1;
            }
        }
        if residual.is_empty() {
            on_image_slices += 1;
        } else {
            off_image_slices += 1;
            if first_off_image_witness.is_none() {
                let (&target_coordinate, residual_value) = residual.first_key_value().unwrap();
                let candidate_value = candidate
                    .get(&target_coordinate)
                    .cloned()
                    .unwrap_or_else(ExactQi::zero);
                let reconstructed_value = reconstructed
                    .get(&target_coordinate)
                    .cloned()
                    .unwrap_or_else(ExactQi::zero);
                first_off_image_witness = Some(Eq40FiberWitness {
                    monomial: key.clone(),
                    target_coordinate,
                    candidate: public(&candidate_value),
                    reconstructed_image: public(&reconstructed_value),
                    residual: public(residual_value),
                });
            }
        }
    }
    let candidate_g4_bianchi_residual_rows = bianchi_residual(&slices)?;
    let compatible = off_image_slices == 0 && candidate_g4_bianchi_residual_rows == 0;
    let source_paths = [
        "src/eleven_dimensional_eq40_independent_a3_fiber.rs",
        "src/eleven_dimensional_gamma24_source_variance.rs",
        "src/eleven_dimensional_physical_curvature.rs",
        "src/eleven_dimensional_superderivative_normal_form.rs",
        "src/eleven_dimensional_target_equation_complex.rs",
    ];
    let source_sha256 = source_paths
        .into_iter()
        .map(|path| Ok((path.to_string(), file_sha256(Path::new(path))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let decision_certified = gamma24.passed
        && gamma24.gamma_two_rank_mod_prime == THREE_FORM_DIMENSION
        && eq31g.input_dimension == GRAVITINO_CURL_DIMENSION
        && eq31g.output_dimension == D_F_FOUR_FORM_DIMENSION
        && image_canary_residual.is_empty()
        && !off_image_mutation_residual.is_empty()
        && physical_fiber_certificate_hash_matches
        && physical_fiber_certificate_passed
        && physical_fiber_intersection_dimension == H_HAT_DIMENSION
        && hom_inventory_hash_matches
        && hom_inventory["passed"] == true
        && general_d2_p1_dg4_hom_dimension == 52
        && general_d0_p2_dg4_hom_dimension == 4
        && candidate_g4_bianchi_residual_rows == 0
        && unexpected_bidegree_slices == 0
        && d2_p1.off_image_slices > 0
        && d2_p1.exact_image_residual_rows > 0;
    Ok(Eq40IndependentA3FiberReport {
        schema_version: "adynkra-11d-eq40-independent-a3-fiber-v1",
        source_ordinal,
        typed_source: "S_D tensor H_hat at (D=1,p=0), differentiated once in PBW normal form",
        candidate_potential_bidegree: [1, 0],
        compared_descendant_bidegree: [2, 1],
        pbw_descendant_bidegrees: [[2, 1], [0, 2]],
        target_join: "corrected numeric-mask Lambda3 -> lexicographic A3 -> target d:A3->G4 -> exact Eq3.1g image projector",
        derivative_index_contract: "beta is contracted inside Psi3=(1/16) Gamma2^{beta gamma} D_beta H_gamma; the subsequently applied D_alpha remains the free DG4 derivative row",
        contracted_inner_beta_and_free_outer_alpha: true,
        derivative_index_relabel_used: false,
        gauge_quotient: "comparison is after d, so A3 gauge image p wedge Lambda2 is annihilated exactly; no negative-momentum gauge parameter is admitted at candidate p-degree zero",
        hhat_lambda3_hom_dimension: 1,
        full_h_lambda3_hom_dimension: 2,
        general_d2_p1_dg4_hom_dimension,
        general_d0_p2_dg4_hom_dimension,
        corrected_lambda3_complete_for_stated_slice: true,
        gamma24_source_variance_passed: gamma24.passed,
        corrected_lambda3_operator_rank: gamma24.gamma_two_rank_mod_prime,
        eq31g_operator_dimensions: (eq31g.output_dimension, eq31g.input_dimension),
        eq31g_operator_nonzero_entries: eq31g.nonzero_entries(),
        projector_in_image_canary_residual_rows: image_canary_residual.len(),
        projector_off_image_mutation_residual_rows: off_image_mutation_residual.len(),
        physical_fiber_certificate_path: PHYSICAL_FIBER_PATH,
        physical_fiber_certificate_sha256,
        physical_fiber_certificate_expected_sha256: PHYSICAL_FIBER_SHA256,
        physical_fiber_certificate_hash_matches,
        physical_fiber_certificate_passed,
        physical_fiber_intersection_dimension,
        hom_inventory_path: HOM_INVENTORY_PATH,
        hom_inventory_sha256,
        hom_inventory_expected_sha256: HOM_INVENTORY_SHA256,
        hom_inventory_hash_matches,
        candidate_rows,
        pbw_monomial_slices: slices.len(),
        on_image_slices,
        off_image_slices,
        recovered_curl_rows,
        exact_image_residual_rows,
        d2_p1,
        d0_p2,
        unexpected_bidegree_slices,
        first_off_image_witness,
        candidate_g4_bianchi_residual_rows,
        candidate_stream_sha256: hash_slices(&slices),
        oracle_source_sha256: format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "eleven_dimensional_eq40_independent_a3_fiber.rs"
            ))
        ),
        source_sha256,
        physical_fiber_compatible_on_unrestricted_hhat_slice: compatible,
        passed: decision_certified,
        boundary: "An off-image witness rules out identifying the unique corrected Eq40 Lambda3 ray with the ordinary physical A3 on the unrestricted Hhat PBW slice. It does not rule out a constrained source quotient, the gamma-trace spinor ray in full H, higher-bidegree potential maps, or a different physical source construction.",
    })
}

pub(crate) fn write_artifact(
    path: &Path,
    source_ordinal: usize,
) -> io::Result<Eq40IndependentA3FiberReport> {
    let report = build_eq40_independent_a3_fiber_report(source_ordinal)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !report.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Eq40 independent-A3 fiber decision failed an exact gate",
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
    fn source_zero_decides_eq40_compatibility_without_eq25() {
        let report = build_eq40_independent_a3_fiber_report(0).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert_eq!(report.candidate_g4_bianchi_residual_rows, 0);
        assert!(report.corrected_lambda3_complete_for_stated_slice);
        assert_eq!(report.hhat_lambda3_hom_dimension, 1);
        assert_eq!(report.projector_in_image_canary_residual_rows, 0);
        assert!(report.projector_off_image_mutation_residual_rows > 0);
        assert!(report.passed, "{report:#?}");
    }
}
