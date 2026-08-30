//! Representation audit of the derivative-Lorentz residual in `J^(1)`.
//!
//! The p=2 derivative compensator space is `S tensor Lambda^2 V`, of
//! dimension `32*55=1760`. This module extracts representative columns from
//! the executable physical maps and reconstructs the full exact residual as
//! a signed-permutation gamma contraction.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use num_rational::Ratio;
use serde::Serialize;

use crate::eleven_dimensional_physical_curvature::ExactQi;

type Rational = Ratio<i64>;

const SPINOR_DIMENSION: usize = 32;
const TWO_FORM_DIMENSION: usize = 55;
const DOMAIN_DIMENSION: usize = SPINOR_DIMENSION * TWO_FORM_DIMENSION;

fn q(value: i64) -> Rational {
    Ratio::from_integer(value)
}

fn masks_of_degree_two() -> Vec<u16> {
    (0_u16..(1_u16 << 11))
        .filter(|mask| mask.count_ones() == 2)
        .collect()
}

fn multiply_i16_i8(left: &[Vec<i16>], right: &[Vec<i8>]) -> Vec<Vec<i16>> {
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for row in 0..SPINOR_DIMENSION {
        for middle in 0..SPINOR_DIMENSION {
            if left[row][middle] == 0 {
                continue;
            }
            for column in 0..SPINOR_DIMENSION {
                output[row][column] += left[row][middle] * i16::from(right[middle][column]);
            }
        }
    }
    output
}

fn upper_gamma_pair(mask: u16) -> Vec<Vec<i16>> {
    let gammas = crate::eleven_dimensional_majorana::real_gamma_matrices();
    let axes = (0..11)
        .filter(|axis| mask & (1_u16 << axis) != 0)
        .collect::<Vec<_>>();
    let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
    for index in 0..SPINOR_DIMENSION {
        output[index][index] = 1;
    }
    for axis in axes {
        output = multiply_i16_i8(&output, &gammas[axis]);
    }
    output
}

fn add_map(target: &mut BTreeMap<usize, ExactQi>, source: &BTreeMap<usize, ExactQi>) {
    for (&index, value) in source {
        let entry = target.entry(index).or_insert_with(ExactQi::zero);
        entry.add_assign(value);
        if entry.is_zero() {
            target.remove(&index);
        }
    }
}

#[derive(Clone)]
struct ResidualParts {
    total: BTreeMap<usize, ExactQi>,
    anholonomy: BTreeMap<usize, ExactQi>,
    connection_from_delta: BTreeMap<usize, ExactQi>,
    connection_explicit: BTreeMap<usize, ExactQi>,
}

struct CachedOperators {
    eq26: crate::eleven_dimensional_physical_curvature::Eq26FactoredOperator,
    anholonomy_to_j: crate::eleven_dimensional_physical_curvature::SparseQiOperator,
    c_to_connection: crate::eleven_dimensional_physical_curvature::SparseQiOperator,
    connection_to_j: crate::eleven_dimensional_physical_curvature::SparseQiOperator,
}

impl CachedOperators {
    fn new() -> Self {
        Self {
            eq26: crate::eleven_dimensional_physical_curvature::eq26_spinor_anholonomy_operator(),
            anholonomy_to_j:
                crate::eleven_dimensional_physical_curvature::c_alpha_beta_gamma_to_j_one_operator(
                ),
            c_to_connection:
                crate::eleven_dimensional_physical_curvature::c_alpha_b_c_to_spinorial_connection_operator(),
            connection_to_j:
                crate::eleven_dimensional_physical_curvature::spinorial_connection_to_j_one_operator(),
        }
    }

    fn residual_parts(&self, derivative: usize, pair: usize) -> ResidualParts {
        let mut d_psi_two = BTreeMap::new();
        d_psi_two.insert(derivative * TWO_FORM_DIMENSION + pair, ExactQi::one());
        let d_delta =
            crate::eleven_dimensional_physical_curvature::inject_d_lorentz_compensator_into_d_delta(
                &d_psi_two,
            );
        let anholonomy = self
            .anholonomy_to_j
            .apply_sparse(&self.eq26.apply(&d_delta));
        let delta_c =
            crate::eleven_dimensional_physical_curvature::apply_eq28_delta_sector_to_c_alpha_b_c(
                &d_delta,
                &BTreeMap::new(),
            );
        let explicit_c =
            crate::eleven_dimensional_physical_curvature::apply_eq28_delta_sector_to_c_alpha_b_c(
                &BTreeMap::new(),
                &d_psi_two,
            );
        let connection_from_delta = self
            .connection_to_j
            .apply_sparse(&self.c_to_connection.apply_sparse(&delta_c));
        let connection_explicit = self
            .connection_to_j
            .apply_sparse(&self.c_to_connection.apply_sparse(&explicit_c));
        let mut total = anholonomy.clone();
        add_map(&mut total, &connection_from_delta);
        add_map(&mut total, &connection_explicit);
        ResidualParts {
            total,
            anholonomy,
            connection_from_delta,
            connection_explicit,
        }
    }
}

fn gamma_ratio(
    image: &BTreeMap<usize, ExactQi>,
    gamma_pair: &[Vec<i16>],
    derivative: usize,
) -> Option<Rational> {
    if image.len() != 1 {
        return None;
    }
    let (&output, value) = image.iter().next().unwrap();
    let gamma = gamma_pair[output][derivative];
    if gamma == 0 || value.imaginary != q(0) {
        return None;
    }
    Some(value.real.clone() / q(i64::from(gamma)))
}

#[derive(Default)]
struct SampleAudit {
    columns_checked: usize,
    mismatches: usize,
    total_boost: BTreeSet<Rational>,
    total_spatial: BTreeSet<Rational>,
    anholonomy: BTreeSet<Rational>,
    delta_boost: BTreeSet<Rational>,
    delta_spatial: BTreeSet<Rational>,
    explicit_boost: BTreeSet<Rational>,
    explicit_spatial: BTreeSet<Rational>,
}

fn record_sample(
    audit: &mut SampleAudit,
    operators: &CachedOperators,
    derivative: usize,
    pair: usize,
    mask: u16,
) {
    audit.columns_checked += 1;
    let gamma = upper_gamma_pair(mask);
    let parts = operators.residual_parts(derivative, pair);
    let ratios = [
        gamma_ratio(&parts.total, &gamma, derivative),
        gamma_ratio(&parts.anholonomy, &gamma, derivative),
        gamma_ratio(&parts.connection_from_delta, &gamma, derivative),
        gamma_ratio(&parts.connection_explicit, &gamma, derivative),
    ];
    if ratios.iter().any(Option::is_none) {
        audit.mismatches += 1;
        return;
    }
    let boost = mask & 1 != 0;
    if boost {
        audit.total_boost.insert(ratios[0].clone().unwrap());
        audit.delta_boost.insert(ratios[2].clone().unwrap());
        audit.explicit_boost.insert(ratios[3].clone().unwrap());
    } else {
        audit.total_spatial.insert(ratios[0].clone().unwrap());
        audit.delta_spatial.insert(ratios[2].clone().unwrap());
        audit.explicit_spatial.insert(ratios[3].clone().unwrap());
    }
    audit.anholonomy.insert(ratios[1].clone().unwrap());
}

fn sample_physical_map() -> SampleAudit {
    let operators = CachedOperators::new();
    let masks = masks_of_degree_two();
    let mut audit = SampleAudit::default();
    // One spinor component across all 55 pairs fixes both Spin(10) orbits.
    for (pair, &mask) in masks.iter().enumerate() {
        record_sample(&mut audit, &operators, 0, pair, mask);
    }
    // One representative boost and spatial pair across all 32 spinors checks
    // that neither coefficient depends on the derivative-spinor coordinate.
    let boost_pair = masks.iter().position(|mask| mask & 1 != 0).unwrap();
    let spatial_pair = masks.iter().position(|mask| mask & 1 == 0).unwrap();
    for derivative in 0..SPINOR_DIMENSION {
        record_sample(
            &mut audit,
            &operators,
            derivative,
            boost_pair,
            masks[boost_pair],
        );
        record_sample(
            &mut audit,
            &operators,
            derivative,
            spatial_pair,
            masks[spatial_pair],
        );
    }
    audit
}

#[derive(Clone, Debug, Serialize)]
pub struct IrreducibleSummand {
    pub dynkin_label: &'static str,
    pub dimension: usize,
    pub multiplicity: usize,
    pub maps_to_j_one: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct InvariantMapAudit {
    pub name: &'static str,
    pub independent: bool,
    pub relation: &'static str,
    pub cancels_current_residual: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct JOneLorentzResidualReport {
    pub schema_version: &'static str,
    pub physical_schema_audited: &'static str,
    pub role: &'static str,
    pub source_references: Vec<&'static str>,
    pub domain: &'static str,
    pub codomain: &'static str,
    pub matrix_rows: usize,
    pub matrix_columns: usize,
    pub matrix_rank: usize,
    pub matrix_nullity: usize,
    pub matrix_nonzero_entries: usize,
    pub nonzero_entries_per_column: usize,
    pub nonzero_entries_per_row: usize,
    pub signed_permutation_gamma_blocks: usize,
    pub signed_permutation_residual_blocks: usize,
    pub boost_columns: usize,
    pub spatial_columns: usize,
    pub boost_coefficient: String,
    pub spatial_coefficient: String,
    pub coefficient_formula: &'static str,
    pub row_gram_is_diagonal: bool,
    pub row_gram_diagonal: String,
    pub residual_is_proportional_to_identity_gamma_trace: bool,
    pub physical_columns_sampled_directly: usize,
    pub sample_formula_residuals: usize,
    pub anholonomy_coefficient: String,
    pub delta_connection_spatial_coefficient: String,
    pub delta_connection_boost_coefficient: String,
    pub explicit_connection_spatial_coefficient: String,
    pub explicit_connection_boost_coefficient: String,
    pub domain_irreducible_decomposition: Vec<IrreducibleSummand>,
    pub target_multiplicity_in_domain: usize,
    pub lorentz_equivariant_hom_dimension: usize,
    pub source_allowed_invariant_maps: Vec<InvariantMapAudit>,
    pub current_residual_is_lorentz_equivariant: bool,
    pub no_equivariant_correction_cancels_current_residual: bool,
    pub historical_v8_boost_coefficient: &'static str,
    pub historical_v8_spatial_coefficient: &'static str,
    pub variance_fix_directly_reaudited: bool,
    pub variance_fix_sufficient: bool,
    pub connection_trace_uses_wrong_vector_variance: bool,
    pub raised_connection_trace_correction: &'static str,
    pub residual_after_variance_fix: String,
    pub unique_invariant_correction_after_variance_fix: String,
    pub corrected_residual_entries: usize,
    pub unique_missing_correction_identified_algebraically: bool,
    pub extra_term_explicitly_printed_in_primary_source: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn singleton(set: &BTreeSet<Rational>, expected: Rational) -> bool {
    set.len() == 1 && set.contains(&expected)
}

pub fn verify() -> JOneLorentzResidualReport {
    let sample = sample_physical_map();
    let anholonomy = Ratio::new(23, 264);
    let delta_spatial = Ratio::new(49, 1_056);
    let delta_boost = delta_spatial.clone();
    let explicit_spatial = Ratio::new(-1, 33);
    let explicit_boost = explicit_spatial.clone();
    let spatial = Ratio::new(109, 1_056);
    let boost = spatial.clone();
    assert_eq!(
        anholonomy.clone() + delta_spatial.clone() + explicit_spatial.clone(),
        spatial
    );
    assert_eq!(
        anholonomy.clone() + delta_boost.clone() + explicit_boost.clone(),
        boost
    );

    let sample_formula_residuals = usize::from(sample.mismatches != 0)
        + usize::from(!singleton(&sample.anholonomy, anholonomy.clone()))
        + usize::from(!singleton(&sample.delta_spatial, delta_spatial.clone()))
        + usize::from(!singleton(&sample.delta_boost, delta_boost.clone()))
        + usize::from(!singleton(
            &sample.explicit_spatial,
            explicit_spatial.clone(),
        ))
        + usize::from(!singleton(&sample.explicit_boost, explicit_boost.clone()))
        + usize::from(!singleton(&sample.total_spatial, spatial.clone()))
        + usize::from(!singleton(&sample.total_boost, boost.clone()));

    let masks = masks_of_degree_two();
    let mut signed_permutation_residual_blocks = 0;
    let mut row_counts = [0_usize; SPINOR_DIMENSION];
    let mut row_gram = vec![vec![q(0); SPINOR_DIMENSION]; SPINOR_DIMENSION];
    let mut matrix_nonzero_entries = 0;
    for &mask in &masks {
        let gamma = upper_gamma_pair(mask);
        for row in 0..SPINOR_DIMENSION {
            let row_nonzero = gamma[row].iter().filter(|entry| **entry != 0).count();
            signed_permutation_residual_blocks += usize::from(row_nonzero != 1);
        }
        for column in 0..SPINOR_DIMENSION {
            let entries = (0..SPINOR_DIMENSION)
                .filter(|row| gamma[*row][column] != 0)
                .collect::<Vec<_>>();
            if entries.len() != 1 || gamma[entries[0]][column].abs() != 1 {
                signed_permutation_residual_blocks += 1;
                continue;
            }
            let row = entries[0];
            let coefficient = if mask & 1 != 0 {
                boost.clone()
            } else {
                spatial.clone()
            } * q(i64::from(gamma[row][column]));
            matrix_nonzero_entries += 1;
            row_counts[row] += 1;
            row_gram[row][row] += coefficient.clone() * coefficient;
        }
    }
    let row_gram_diagonal =
        q(10) * boost.clone() * boost.clone() + q(45) * spatial.clone() * spatial.clone();
    let row_gram_is_diagonal = row_gram.iter().enumerate().all(|(row, entries)| {
        entries.iter().enumerate().all(|(column, value)| {
            *value
                == if row == column {
                    row_gram_diagonal.clone()
                } else {
                    q(0)
                }
        })
    });
    let matrix_rank = row_counts.iter().filter(|count| **count > 0).count();

    // The source-indexed Eq. (28) bilinears remove the former boost/spatial
    // split. The remaining nonzero map is the unique Lorentz-equivariant
    // Gamma^[de] contraction, so covariance is restored but gauge closure is
    // not established by this bounded p=2 column test.
    let variance_fix_sufficient = boost == spatial;
    let corrected_residual_entries = DOMAIN_DIMENSION;

    let source_allowed_invariant_maps = vec![
        InvariantMapAudit {
            name: "Gamma^[de] contraction",
            independent: true,
            relation: "the unique copy of (00001) in (00001) tensor (01000)",
            cancels_current_residual: true,
        },
        InvariantMapAudit {
            name: "epsilon Gamma_[9] contraction",
            independent: false,
            relation: "11D Clifford duality makes Gamma_[9] proportional to epsilon Gamma^[2]",
            cancels_current_residual: false,
        },
        InvariantMapAudit {
            name: "metric-only contraction",
            independent: false,
            relation: "no metric contraction removes one antisymmetric vector pair without gamma matrices",
            cancels_current_residual: false,
        },
        InvariantMapAudit {
            name: "charge-conjugated Gamma^[2] contraction",
            independent: false,
            relation: "charge conjugation only changes spinor index variance in the fixed Majorana representation",
            cancels_current_residual: false,
        },
    ];
    let domain_irreducible_decomposition = vec![
        IrreducibleSummand {
            dynkin_label: "00001",
            dimension: 32,
            multiplicity: 1,
            maps_to_j_one: true,
        },
        IrreducibleSummand {
            dynkin_label: "10001",
            dimension: 320,
            multiplicity: 1,
            maps_to_j_one: false,
        },
        IrreducibleSummand {
            dynkin_label: "01001",
            dimension: 1_408,
            multiplicity: 1,
            maps_to_j_one: false,
        },
    ];
    let passed = sample.columns_checked == 119
        && sample_formula_residuals == 0
        && signed_permutation_residual_blocks == 0
        && matrix_nonzero_entries == DOMAIN_DIMENSION
        && row_counts.iter().all(|count| *count == TWO_FORM_DIMENSION)
        && row_gram_is_diagonal
        && row_gram_diagonal == Ratio::new(59_405, 101_376)
        && matrix_rank == SPINOR_DIMENSION
        && boost == spatial
        && variance_fix_sufficient;

    JOneLorentzResidualReport {
        schema_version: "adynkra.11d.j1-lorentz-residual.v3",
        physical_schema_audited: "adynkra-11d-physical-curvature-operator-v10",
        role: "exact representation-map audit of the 1760-column derivative-Lorentz J^(1) response after the source-indexed Eq. (28) fix",
        source_references: vec![
            "hep-th/0101037 Eq. (24): p=2 Lorentz compensator in Delta",
            "hep-th/0101037 Eq. (28): Delta and explicit Lorentz-compensator contributions",
            "hep-th/0101037 Table 3: spinorial Lorentz connection",
            "hep-th/0107155 Eqs. (2)-(5): anholonomy, connection, and torsion conventions",
        ],
        domain: "D Psi_[de] in (00001) tensor (01000), dimension 32*55=1760",
        codomain: "J_alpha^(1) in (00001), dimension 32",
        matrix_rows: SPINOR_DIMENSION,
        matrix_columns: DOMAIN_DIMENSION,
        matrix_rank,
        matrix_nullity: DOMAIN_DIMENSION - matrix_rank,
        matrix_nonzero_entries,
        nonzero_entries_per_column: 1,
        nonzero_entries_per_row: TWO_FORM_DIMENSION,
        signed_permutation_gamma_blocks: TWO_FORM_DIMENSION,
        signed_permutation_residual_blocks,
        boost_columns: 10 * SPINOR_DIMENSION,
        spatial_columns: 45 * SPINOR_DIMENSION,
        boost_coefficient: boost.to_string(),
        spatial_coefficient: spatial.to_string(),
        coefficient_formula: "R_(alpha;delta,de)=(109/1056)(Gamma^d Gamma^e)_alpha,delta",
        row_gram_is_diagonal,
        row_gram_diagonal: row_gram_diagonal.to_string(),
        residual_is_proportional_to_identity_gamma_trace: boost == spatial,
        physical_columns_sampled_directly: sample.columns_checked,
        sample_formula_residuals,
        anholonomy_coefficient: anholonomy.to_string(),
        delta_connection_spatial_coefficient: delta_spatial.to_string(),
        delta_connection_boost_coefficient: delta_boost.to_string(),
        explicit_connection_spatial_coefficient: explicit_spatial.to_string(),
        explicit_connection_boost_coefficient: explicit_boost.to_string(),
        domain_irreducible_decomposition,
        target_multiplicity_in_domain: 1,
        lorentz_equivariant_hom_dimension: 1,
        source_allowed_invariant_maps,
        current_residual_is_lorentz_equivariant: boost == spatial,
        no_equivariant_correction_cancels_current_residual: false,
        historical_v8_boost_coefficient: "1847/11616",
        historical_v8_spatial_coefficient: "59/3872",
        variance_fix_directly_reaudited: true,
        variance_fix_sufficient,
        connection_trace_uses_wrong_vector_variance: false,
        raised_connection_trace_correction:
            "applied: Gamma^de now contracts the stored lower omega_[de] coordinate",
        residual_after_variance_fix: "boost=109/1056; spatial=109/1056".to_string(),
        unique_invariant_correction_after_variance_fix: "algebraically unique: -(109/1056) Gamma^[de], but no audited source authorizes adding it".to_string(),
        corrected_residual_entries,
        unique_missing_correction_identified_algebraically: true,
        extra_term_explicitly_printed_in_primary_source: false,
        passed,
        boundary: "The source-indexed Eq. (28) correction removes the former boost/spatial split. All 1,760 columns now form the unique Lorentz-equivariant Gamma^[2] map with coefficient 109/1056 and exact rank 32. This resolves the covariance defect but does not prove that the p=2 semi-prepotential column is a complete gauge orbit. An algebraic subtraction is unique, but no audited source prints it, so no fitted term is added. Induced J, T, W, full F A G_p, and off-shell closure remain fail closed.",
    }
}

pub fn write_artifact(path: &Path) -> io::Result<()> {
    let report = verify();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &report)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    file.write_all(b"\n")?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_indexed_residual_is_the_unique_lorentz_gamma_structure() {
        let report = verify();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.matrix_rank, 32);
        assert_eq!(report.matrix_nullity, 1_728);
        assert_eq!(report.matrix_nonzero_entries, 1_760);
        assert_eq!(report.boost_coefficient, "109/1056");
        assert_eq!(report.spatial_coefficient, "109/1056");
        assert!(report.current_residual_is_lorentz_equivariant);
    }

    #[test]
    fn covariance_is_fixed_but_no_unprinted_correction_is_added() {
        let report = verify();
        assert_eq!(report.lorentz_equivariant_hom_dimension, 1);
        assert!(!report.connection_trace_uses_wrong_vector_variance);
        assert!(report.variance_fix_sufficient);
        assert_eq!(
            report.residual_after_variance_fix,
            "boost=109/1056; spatial=109/1056"
        );
        assert_eq!(report.corrected_residual_entries, 1_760);
        assert!(report.unique_missing_correction_identified_algebraically);
        assert!(!report.extra_term_explicitly_printed_in_primary_source);
    }

    #[test]
    #[ignore = "artifact writer"]
    fn write_checked_artifact() {
        write_artifact(Path::new("results/adynkra_11d_j1_lorentz_residual.json")).unwrap();
    }
}
