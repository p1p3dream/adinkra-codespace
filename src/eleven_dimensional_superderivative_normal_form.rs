//! Exact ordered normal form for flat eleven-dimensional superderivatives.
//!
//! In the Cartesian Majorana basis used by the physical-curvature modules,
//! the flat algebra is fixed here as
//!
//! `D_alpha D_beta + D_beta D_alpha = i (C Gamma^a)_{alpha beta} p_a`.
//!
//! Odd derivatives are stored in strictly descending spinor-index order as a
//! 32-bit exterior mask.  The eleven commuting momentum variables are stored
//! as an exact exponent vector.  Left multiplication performs every required
//! anticommutator while inserting the new derivative into that canonical
//! order.  This is the missing algebraic normal form needed before the
//! constrained `H_hat` first jet can be composed into the complete physical
//! curvature operator.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use num_rational::Ratio;
use serde::Serialize;

use crate::eleven_dimensional_physical_curvature::{ExactQi, SPINOR_DIMENSION, VECTOR_DIMENSION};

pub const SCHEMA_VERSION: &str = "adynkra-11d-superderivative-normal-form-v1";

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FormalMomentumMonomial {
    pub exponents: [u16; VECTOR_DIMENSION],
}

impl FormalMomentumMonomial {
    pub fn constant() -> Self {
        Self {
            exponents: [0; VECTOR_DIMENSION],
        }
    }

    pub fn multiply_variable(&self, axis: usize) -> Result<Self, String> {
        if axis >= VECTOR_DIMENSION {
            return Err(format!(
                "momentum axis {axis} is outside 0..{VECTOR_DIMENSION}"
            ));
        }
        let mut exponents = self.exponents;
        exponents[axis] = exponents[axis]
            .checked_add(1)
            .ok_or_else(|| format!("momentum exponent overflow on axis {axis}"))?;
        Ok(Self { exponents })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OrderedSuperderivativeMonomial {
    /// Canonical product is `D_31 ... D_1 D_0`, restricted to set bits.
    pub exterior_spinor_mask: u32,
    pub momentum: FormalMomentumMonomial,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CanonicalSuperPolynomial {
    pub terms: BTreeMap<OrderedSuperderivativeMonomial, ExactQi>,
}

fn product(left: &ExactQi, right: &ExactQi) -> ExactQi {
    ExactQi {
        real: left.real.clone() * right.real.clone()
            - left.imaginary.clone() * right.imaginary.clone(),
        imaginary: left.real.clone() * right.imaginary.clone()
            + left.imaginary.clone() * right.real.clone(),
    }
}

fn rational(value: i64) -> Ratio<i64> {
    Ratio::from_integer(value)
}

fn scale_integer(value: &ExactQi, factor: i64) -> ExactQi {
    value.scaled(&rational(factor))
}

impl CanonicalSuperPolynomial {
    pub fn scalar(coefficient: ExactQi) -> Self {
        let mut result = Self::default();
        result.add_term(
            OrderedSuperderivativeMonomial {
                exterior_spinor_mask: 0,
                momentum: FormalMomentumMonomial::constant(),
            },
            coefficient,
        );
        result
    }

    pub fn add_term(&mut self, key: OrderedSuperderivativeMonomial, value: ExactQi) {
        if value.is_zero() {
            return;
        }
        let entry = self.terms.entry(key.clone()).or_insert_with(ExactQi::zero);
        entry.add_assign(&value);
        if entry.is_zero() {
            self.terms.remove(&key);
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        for (key, value) in &other.terms {
            self.add_term(key.clone(), value.clone());
        }
    }

    pub fn scaled(&self, coefficient: &ExactQi) -> Self {
        let mut result = Self::default();
        for (key, value) in &self.terms {
            result.add_term(key.clone(), product(value, coefficient));
        }
        result
    }

    pub fn multiply_momentum(&self, axis: usize) -> Result<Self, String> {
        let mut result = Self::default();
        for (key, value) in &self.terms {
            result.add_term(
                OrderedSuperderivativeMonomial {
                    exterior_spinor_mask: key.exterior_spinor_mask,
                    momentum: key.momentum.multiply_variable(axis)?,
                },
                value.clone(),
            );
        }
        Ok(result)
    }
}

fn majorana_translation_bilinears() -> Vec<Vec<Vec<i64>>> {
    let charge = crate::eleven_dimensional_majorana::real_charge_conjugation();
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    gammas
        .iter()
        .map(|gamma| {
            let mut bilinear = vec![vec![0_i64; SPINOR_DIMENSION]; SPINOR_DIMENSION];
            for row in 0..SPINOR_DIMENSION {
                for pivot in 0..SPINOR_DIMENSION {
                    if charge[row][pivot] == 0 {
                        continue;
                    }
                    for column in 0..SPINOR_DIMENSION {
                        bilinear[row][column] +=
                            i64::from(charge[row][pivot]) * i64::from(gamma[pivot][column]);
                    }
                }
            }
            bilinear
        })
        .collect()
}

fn translation_action_with(
    bilinears: &[Vec<Vec<i64>>],
    alpha: usize,
    beta: usize,
    polynomial: &CanonicalSuperPolynomial,
) -> Result<CanonicalSuperPolynomial, String> {
    let mut result = CanonicalSuperPolynomial::default();
    for axis in 0..VECTOR_DIMENSION {
        let entry = bilinears[axis][alpha][beta];
        if entry == 0 {
            continue;
        }
        let momentum = polynomial.multiply_momentum(axis)?;
        result.add_assign(&momentum.scaled(&scale_integer(&ExactQi::i(), entry)));
    }
    Ok(result)
}

/// Apply the exact right-hand side of `{D_alpha,D_beta}`.
pub fn translation_action(
    alpha: usize,
    beta: usize,
    polynomial: &CanonicalSuperPolynomial,
) -> Result<CanonicalSuperPolynomial, String> {
    if alpha >= SPINOR_DIMENSION || beta >= SPINOR_DIMENSION {
        return Err("spinor derivative is outside 0..32".to_string());
    }
    translation_action_with(&majorana_translation_bilinears(), alpha, beta, polynomial)
}

fn left_multiply_with(
    bilinears: &[Vec<Vec<i64>>],
    alpha: usize,
    polynomial: &CanonicalSuperPolynomial,
) -> Result<CanonicalSuperPolynomial, String> {
    if alpha >= SPINOR_DIMENSION {
        return Err(format!(
            "spinor derivative {alpha} is outside 0..{SPINOR_DIMENSION}"
        ));
    }
    let alpha_bit = 1_u32 << alpha;
    let mut result = CanonicalSuperPolynomial::default();
    for (key, coefficient) in &polynomial.terms {
        let mask = key.exterior_spinor_mask;

        // Every larger derivative crossed on the way to canonical descending
        // order contributes one Clifford contraction.
        let mut larger = mask & !((alpha_bit << 1).wrapping_sub(1));
        let mut crossed = 0_u32;
        while larger != 0 {
            let beta = 31 - larger.leading_zeros() as usize;
            larger &= !(1_u32 << beta);
            for axis in 0..VECTOR_DIMENSION {
                let entry = bilinears[axis][alpha][beta];
                if entry == 0 {
                    continue;
                }
                let sign = if crossed % 2 == 0 { 1 } else { -1 };
                let translated = product(coefficient, &scale_integer(&ExactQi::i(), sign * entry));
                result.add_term(
                    OrderedSuperderivativeMonomial {
                        exterior_spinor_mask: mask ^ (1_u32 << beta),
                        momentum: key.momentum.multiply_variable(axis)?,
                    },
                    translated,
                );
            }
            crossed += 1;
        }

        let greater_count = if alpha + 1 == u32::BITS as usize {
            0
        } else {
            (mask >> (alpha + 1)).count_ones()
        };
        if mask & alpha_bit == 0 {
            let sign = if greater_count % 2 == 0 { 1 } else { -1 };
            result.add_term(
                OrderedSuperderivativeMonomial {
                    exterior_spinor_mask: mask | alpha_bit,
                    momentum: key.momentum.clone(),
                },
                scale_integer(coefficient, sign),
            );
        } else {
            // 2 D_alpha^2 = {D_alpha,D_alpha}.
            for axis in 0..VECTOR_DIMENSION {
                let entry = bilinears[axis][alpha][alpha];
                if entry == 0 {
                    continue;
                }
                let sign = if greater_count % 2 == 0 { 1 } else { -1 };
                let half_i = ExactQi::i().scaled(&Ratio::new(sign * entry, 2));
                result.add_term(
                    OrderedSuperderivativeMonomial {
                        exterior_spinor_mask: mask ^ alpha_bit,
                        momentum: key.momentum.multiply_variable(axis)?,
                    },
                    product(coefficient, &half_i),
                );
            }
        }
    }
    Ok(result)
}

/// Left-multiply by `D_alpha` and return the unique ordered normal form.
pub fn left_multiply_d(
    alpha: usize,
    polynomial: &CanonicalSuperPolynomial,
) -> Result<CanonicalSuperPolynomial, String> {
    left_multiply_with(&majorana_translation_bilinears(), alpha, polynomial)
}

fn anticommutator_with(
    bilinears: &[Vec<Vec<i64>>],
    alpha: usize,
    beta: usize,
    polynomial: &CanonicalSuperPolynomial,
) -> Result<CanonicalSuperPolynomial, String> {
    let mut result = left_multiply_with(
        bilinears,
        alpha,
        &left_multiply_with(bilinears, beta, polynomial)?,
    )?;
    result.add_assign(&left_multiply_with(
        bilinears,
        beta,
        &left_multiply_with(bilinears, alpha, polynomial)?,
    )?);
    Ok(result)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SuperderivativeNormalFormReport {
    pub schema_version: &'static str,
    pub role: &'static str,
    pub basis: &'static str,
    pub relation: &'static str,
    pub vector_dimension: usize,
    pub spinor_dimension: usize,
    pub symmetric_bilinear_entries_checked: usize,
    pub symmetric_bilinear_residual_entries: usize,
    pub constant_anticommutator_pairs_checked: usize,
    pub constant_anticommutator_residual_pairs: usize,
    pub degree_one_overlap_triples_checked: usize,
    pub degree_one_overlap_residual_triples: usize,
    pub formal_momentum_axes_reached: usize,
    pub mutation_rejected: bool,
    pub ordered_superderivative_normal_form_complete: bool,
    pub h_hat_first_jet_composed: bool,
    pub complete_physical_f_implemented: bool,
    pub passed: bool,
    pub result: &'static str,
    pub boundary: &'static str,
}

fn build_report() -> SuperderivativeNormalFormReport {
    let bilinears = majorana_translation_bilinears();
    let symmetric_bilinear_residual_entries = (0..VECTOR_DIMENSION)
        .flat_map(|axis| {
            let bilinears = &bilinears;
            (0..SPINOR_DIMENSION).flat_map(move |alpha| {
                (0..SPINOR_DIMENSION).map(move |beta| {
                    usize::from(bilinears[axis][alpha][beta] != bilinears[axis][beta][alpha])
                })
            })
        })
        .sum();

    let scalar = CanonicalSuperPolynomial::scalar(ExactQi::one());
    let mut constant_anticommutator_residual_pairs = 0;
    let mut degree_one_overlap_residual_triples = 0;
    let mut reached_axes = [false; VECTOR_DIMENSION];
    for alpha in 0..SPINOR_DIMENSION {
        for beta in 0..SPINOR_DIMENSION {
            let expected = translation_action_with(&bilinears, alpha, beta, &scalar)
                .expect("constant momentum exponents cannot overflow");
            let actual = anticommutator_with(&bilinears, alpha, beta, &scalar)
                .expect("constant momentum exponents cannot overflow");
            constant_anticommutator_residual_pairs += usize::from(actual != expected);
            for key in expected.terms.keys() {
                for (axis, exponent) in key.momentum.exponents.iter().enumerate() {
                    reached_axes[axis] |= *exponent != 0;
                }
            }
            for gamma in 0..SPINOR_DIMENSION {
                let degree_one = left_multiply_with(&bilinears, gamma, &scalar)
                    .expect("constant momentum exponents cannot overflow");
                let actual = anticommutator_with(&bilinears, alpha, beta, &degree_one)
                    .expect("degree-one momentum exponents cannot overflow");
                let expected = translation_action_with(&bilinears, alpha, beta, &degree_one)
                    .expect("degree-one momentum exponents cannot overflow");
                degree_one_overlap_residual_triples += usize::from(actual != expected);
            }
        }
    }

    let nonzero_pair = (0..SPINOR_DIMENSION)
        .flat_map(|alpha| (0..SPINOR_DIMENSION).map(move |beta| (alpha, beta)))
        .find(|&(alpha, beta)| (0..VECTOR_DIMENSION).any(|axis| bilinears[axis][alpha][beta] != 0))
        .expect("11D translation bilinear is nonzero");
    let actual = anticommutator_with(&bilinears, nonzero_pair.0, nonzero_pair.1, &scalar)
        .expect("constant momentum exponents cannot overflow");
    let mutated = translation_action_with(&bilinears, nonzero_pair.0, nonzero_pair.1, &scalar)
        .expect("constant momentum exponents cannot overflow")
        .scaled(&ExactQi::from_integer(2));
    let mutation_rejected = actual != mutated;

    let formal_momentum_axes_reached = reached_axes.into_iter().filter(|reached| *reached).count();
    let passed = symmetric_bilinear_residual_entries == 0
        && constant_anticommutator_residual_pairs == 0
        && degree_one_overlap_residual_triples == 0
        && formal_momentum_axes_reached == VECTOR_DIMENSION
        && mutation_rejected;

    SuperderivativeNormalFormReport {
        schema_version: SCHEMA_VERSION,
        role: "exact canonical ordered-superderivative and formal-momentum algebra for the complete H_hat jet",
        basis: "real 32-component Majorana spinors; descending spinor-index exterior monomials; eleven commuting Lorentz momentum variables",
        relation: "D_alpha D_beta + D_beta D_alpha = i (C Gamma^a)_{alpha beta} p_a",
        vector_dimension: VECTOR_DIMENSION,
        spinor_dimension: SPINOR_DIMENSION,
        symmetric_bilinear_entries_checked: VECTOR_DIMENSION
            * SPINOR_DIMENSION
            * SPINOR_DIMENSION,
        symmetric_bilinear_residual_entries,
        constant_anticommutator_pairs_checked: SPINOR_DIMENSION * SPINOR_DIMENSION,
        constant_anticommutator_residual_pairs,
        degree_one_overlap_triples_checked: SPINOR_DIMENSION
            * SPINOR_DIMENSION
            * SPINOR_DIMENSION,
        degree_one_overlap_residual_triples,
        formal_momentum_axes_reached,
        mutation_rejected,
        ordered_superderivative_normal_form_complete: passed,
        h_hat_first_jet_composed: false,
        complete_physical_f_implemented: false,
        passed,
        result: "All 1,024 spinor anticommutators and all 32,768 degree-one overlap ambiguities reduce exactly to the same eleven-momentum normal form.",
        boundary: "This certifies the flat ordered-D algebra required by the full constrained-frame calculation. It does not yet derive D C, mixed anholonomy, J, torsion, or W from one compensator-eliminated H_hat jet, and therefore does not by itself complete physical F or determine physical K.",
    }
}

pub fn verify() -> SuperderivativeNormalFormReport {
    static REPORT: OnceLock<SuperderivativeNormalFormReport> = OnceLock::new();
    REPORT.get_or_init(build_report).clone()
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub fn write_artifact(path: &Path) -> io::Result<SuperderivativeNormalFormReport> {
    let report = verify();
    atomic_json(path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_cartesian_majorana_anticommutator_and_overlap_gate_passes() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.constant_anticommutator_pairs_checked, 1_024);
        assert_eq!(report.constant_anticommutator_residual_pairs, 0);
        assert_eq!(report.degree_one_overlap_triples_checked, 32_768);
        assert_eq!(report.degree_one_overlap_residual_triples, 0);
        assert_eq!(report.formal_momentum_axes_reached, 11);
    }

    #[test]
    fn repeated_derivative_produces_exact_half_anticommutator() {
        let scalar = CanonicalSuperPolynomial::scalar(ExactQi::one());
        for alpha in 0..SPINOR_DIMENSION {
            let first = left_multiply_d(alpha, &scalar).unwrap();
            let square = left_multiply_d(alpha, &first).unwrap();
            let expected = translation_action(alpha, alpha, &scalar)
                .unwrap()
                .scaled(&ExactQi::from_rational(1, 2));
            assert_eq!(square, expected, "alpha={alpha}");
        }
    }

    #[test]
    fn left_multiplication_preserves_existing_momentum_and_checks_overflow() {
        let mut polynomial = CanonicalSuperPolynomial::default();
        let mut exponents = [0_u16; VECTOR_DIMENSION];
        exponents[3] = 7;
        polynomial.add_term(
            OrderedSuperderivativeMonomial {
                exterior_spinor_mask: (1_u32 << 7) | (1_u32 << 2),
                momentum: FormalMomentumMonomial { exponents },
            },
            ExactQi::from_integer(3),
        );
        let image = left_multiply_d(4, &polynomial).unwrap();
        assert!(image.terms.keys().all(|key| key.momentum.exponents[3] >= 7));

        let mut overflow = [0_u16; VECTOR_DIMENSION];
        overflow[0] = u16::MAX;
        assert!(FormalMomentumMonomial {
            exponents: overflow
        }
        .multiply_variable(0)
        .is_err());
    }

    #[test]
    fn invalid_spinor_and_mutated_factor_fail_closed() {
        let scalar = CanonicalSuperPolynomial::scalar(ExactQi::one());
        assert!(left_multiply_d(32, &scalar).is_err());
        assert!(translation_action(0, 32, &scalar).is_err());
        assert!(verify().mutation_rejected);
        assert!(!verify().h_hat_first_jet_composed);
        assert!(!verify().complete_physical_f_implemented);
    }
}
