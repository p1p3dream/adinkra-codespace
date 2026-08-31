//! Exact Chevalley cocycle gates for horizontal local-Lorentz descent.
//!
//! The canonical vertical coefficient space at fixed momentum is the
//! rank-320 quotient of `D Psi_[2]` selected by the exact Eq. (25)-to-DG4
//! image reducer. A section cocycle must obey the Chevalley bracket before a
//! coherent zero-cochain or streamed quotient solve is attempted.

use std::collections::BTreeMap;

use num_rational::Ratio;
use serde::Serialize;

use crate::eleven_dimensional_corrected_teleparallel_equivariance::{
    ExactQiPublic, LocalLorentzTargetImageHandoff, canonical_psi2_quotient_action,
    local_lorentz_target_image_handoff,
};
use crate::eleven_dimensional_d21_invariant_diagrams::{
    d21_source_lorentz_generator_terms, decode_source_coordinate,
};
use crate::eleven_dimensional_physical_curvature::ExactQi;

const VECTOR_DIMENSION: usize = 11;
const D21_SOURCE_DIMENSION: u64 = 496 * 11 * 320;
const AMBIENT_PSI2_DIMENSION: usize = 32 * 55;
const CANONICAL_PSI2_QUOTIENT_RANK: usize = 320;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ChevalleyCocycleWitness {
    pub source_coordinate: u32,
    pub momentum_axis: usize,
    pub generator_x: [usize; 2],
    pub generator_y: [usize; 2],
    pub bracket_generator: [usize; 2],
    pub bracket_coefficient: i64,
    pub canonical_quotient_rank: usize,
    pub residual_entries: usize,
    pub first_residual_coordinate: Option<usize>,
    pub first_residual: Option<ExactQiPublic>,
    pub omitted_bracket_factor_mutation_entries: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct HorizontalZeroCochainContract {
    pub source_bidegree: &'static str,
    pub source_dimension: u64,
    pub vertical_bidegree: &'static str,
    pub ambient_vertical_dimension_per_momentum: usize,
    pub canonical_vertical_rank_per_momentum: usize,
    pub full_zero_cochain_coefficients: u64,
    pub eliminated_solve_unknowns: usize,
    pub elimination: &'static str,
    pub ordinary_equivariance_requires_zero_cochain: bool,
}

fn add_scaled(
    output: &mut BTreeMap<usize, ExactQi>,
    input: &BTreeMap<usize, ExactQi>,
    integer: i64,
) {
    if integer == 0 {
        return;
    }
    let factor = Ratio::from_integer(integer);
    for (&coordinate, value) in input {
        let entry = output.entry(coordinate).or_insert_with(ExactQi::zero);
        entry.add_assign(&value.scaled(&factor));
        if entry.is_zero() {
            output.remove(&coordinate);
        }
    }
}

fn cocycle_coordinates(
    handoff: &LocalLorentzTargetImageHandoff,
    left: usize,
    right: usize,
) -> Result<BTreeMap<usize, ExactQi>, String> {
    handoff
        .raw_commutators
        .iter()
        .find(|column| column.generator_left == left && column.generator_right == right)
        .map(|column| column.image_coordinates.clone())
        .ok_or_else(|| format!("missing cocycle generator ({left},{right})"))
}

fn cached_handoff<'a>(
    cache: &'a mut BTreeMap<u32, LocalLorentzTargetImageHandoff>,
    source: u32,
) -> Result<&'a LocalLorentzTargetImageHandoff, String> {
    if !cache.contains_key(&source) {
        cache.insert(source, local_lorentz_target_image_handoff(source)?);
    }
    Ok(&cache[&source])
}

/// Evaluate one nontrivial bracket that preserves the witness momentum:
/// `[T_01,T_12]=2 T_02` in the repository's twice-generator normalization.
/// The source and quotient actions are both applied explicitly.
pub(crate) fn witness_chevalley_cocycle_gate(
    source_coordinate: u32,
) -> Result<ChevalleyCocycleWitness, String> {
    let (_, momentum_axis, _) = decode_source_coordinate(source_coordinate)?;
    if momentum_axis == 0 || momentum_axis == 1 || momentum_axis == 2 {
        return Err("fixed-momentum bracket canary requires momentum outside axes 0,1,2".into());
    }
    let x = (0, 1);
    let y = (1, 2);
    let bracket = (0, 2);
    let bracket_coefficient = 2_i64;
    let mut cache = BTreeMap::new();
    let root = cached_handoff(&mut cache, source_coordinate)?.clone();
    if root.exact_image_rank != CANONICAL_PSI2_QUOTIENT_RANK {
        return Err("vertical quotient rank drifted from 320".to_string());
    }
    let c_x = cocycle_coordinates(&root, x.0, x.1)?;
    let c_y = cocycle_coordinates(&root, y.0, y.1)?;
    let c_bracket = cocycle_coordinates(&root, bracket.0, bracket.1)?;
    let (rho_x_c_y, rho_x_remainder) =
        canonical_psi2_quotient_action(momentum_axis, &c_y, x.0, x.1)?;
    let (rho_y_c_x, rho_y_remainder) =
        canonical_psi2_quotient_action(momentum_axis, &c_x, y.0, y.1)?;
    if !rho_x_remainder.is_empty() || !rho_y_remainder.is_empty() {
        return Err("induced canonical Psi2 quotient action left a residual".to_string());
    }

    let mut c_y_rho_x = BTreeMap::new();
    for term in d21_source_lorentz_generator_terms(source_coordinate, x.0, x.1)? {
        let next = cached_handoff(&mut cache, term.source_coordinate)?;
        add_scaled(
            &mut c_y_rho_x,
            &cocycle_coordinates(next, y.0, y.1)?,
            term.coefficient,
        );
    }
    let mut c_x_rho_y = BTreeMap::new();
    for term in d21_source_lorentz_generator_terms(source_coordinate, y.0, y.1)? {
        let next = cached_handoff(&mut cache, term.source_coordinate)?;
        add_scaled(
            &mut c_x_rho_y,
            &cocycle_coordinates(next, x.0, x.1)?,
            term.coefficient,
        );
    }

    let residual_with_factor = |factor: i64| {
        let mut residual = BTreeMap::new();
        add_scaled(&mut residual, &rho_x_c_y, 1);
        add_scaled(&mut residual, &c_y_rho_x, -1);
        add_scaled(&mut residual, &rho_y_c_x, -1);
        add_scaled(&mut residual, &c_x_rho_y, 1);
        add_scaled(&mut residual, &c_bracket, -factor);
        residual
    };
    let residual = residual_with_factor(bracket_coefficient);
    let mutation = residual_with_factor(1);
    let first = residual.first_key_value();
    Ok(ChevalleyCocycleWitness {
        source_coordinate,
        momentum_axis,
        generator_x: [x.0, x.1],
        generator_y: [y.0, y.1],
        bracket_generator: [bracket.0, bracket.1],
        bracket_coefficient,
        canonical_quotient_rank: root.exact_image_rank,
        residual_entries: residual.len(),
        first_residual_coordinate: first.map(|(&coordinate, _)| coordinate),
        first_residual: first.map(|(_, value)| ExactQiPublic::from(value)),
        omitted_bracket_factor_mutation_entries: mutation.len(),
        passed: residual.is_empty() && !mutation.is_empty(),
    })
}

pub(crate) fn horizontal_zero_cochain_contract() -> HorizontalZeroCochainContract {
    HorizontalZeroCochainContract {
        source_bidegree: "D2P1: Lambda^2 S tensor V* tensor Hhat",
        source_dimension: D21_SOURCE_DIMENSION,
        vertical_bidegree: "V* tensor (D_beta Psi^[de] / ker Q), rank 11*320",
        ambient_vertical_dimension_per_momentum: AMBIENT_PSI2_DIMENSION,
        canonical_vertical_rank_per_momentum: CANONICAL_PSI2_QUOTIENT_RANK,
        full_zero_cochain_coefficients: D21_SOURCE_DIMENSION * 320,
        eliminated_solve_unknowns: 52,
        elimination: "stream each source row through the canonical Q reducer; solve the invariant D21 Hom coefficients, then replay T-F to recover canonical Psi2 coordinates",
        ordinary_equivariance_requires_zero_cochain: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_cocycle_obeys_nontrivial_chevalley_bracket() {
        let report = witness_chevalley_cocycle_gate(131_857).unwrap();
        eprintln!("CHEVALLEY_COCYCLE {report:#?}");
        assert!(report.passed);
        assert_eq!(report.residual_entries, 0);
        assert!(report.omitted_bracket_factor_mutation_entries > 0);
    }

    #[test]
    fn zero_cochain_contract_eliminates_streamed_vertical_unknowns() {
        let contract = horizontal_zero_cochain_contract();
        assert_eq!(contract.source_dimension, 1_745_920);
        assert_eq!(contract.full_zero_cochain_coefficients, 558_694_400);
        assert_eq!(contract.eliminated_solve_unknowns, 52);
        assert!(contract.ordinary_equivariance_requires_zero_cochain);
    }
}
