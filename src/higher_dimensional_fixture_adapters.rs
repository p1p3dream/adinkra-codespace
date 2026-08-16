//! Exact canonical adapters for the sourced chiral-vector, chiral-tensor, and
//! vector-tensor positive controls.

use crate::higher_dimensional_canonical::{
    CanonicalError, CanonicalFingerprint, CanonicalOptions, ComponentRole, PhysicalFingerprint,
    canonicalize,
};
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AdaptedFixtureSummary {
    pub name: String,
    pub canonical: CanonicalFingerprint,
    pub components: usize,
    pub supercharges: usize,
    pub linkage_terms: usize,
    pub gauge_arrows: usize,
    pub bianchi_relations: usize,
    pub maximum_reducibility_stage: Option<u8>,
    pub imaginary_linkage_coefficients: usize,
    pub fractional_linkage_coefficients: usize,
    pub central_generators: usize,
    pub central_entries: usize,
    pub central_occurrences: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PositiveControlAdapters {
    pub schema_version: &'static str,
    pub chiral_vector: AdaptedFixtureSummary,
    pub chiral_tensor: AdaptedFixtureSummary,
    pub vector_tensor: AdaptedFixtureSummary,
    pub central_hypermultiplet: AdaptedFixtureSummary,
    pub fingerprints_distinct: bool,
    pub validation_passed: bool,
    pub vector_tensor_boundary: &'static str,
}

/// Full exact chiral-vector source fixture, including `F=dA` Bianchi data.
pub fn chiral_vector_fixture() -> PhysicalFingerprint {
    crate::chiral_vector_4d::exact_canonical_fixture()
}

/// Full exact chiral-tensor source fixture, including the reducible gauge
/// complex and `dH=0` Bianchi data.
pub fn chiral_tensor_fixture() -> PhysicalFingerprint {
    crate::chiral_tensor_4d::exact_canonical_fixture()
}

/// Full exact vector-tensor fixture, including both reducible gauge complexes,
/// both curvature systems, and the intrinsic central generator.
pub fn vector_tensor_fixture() -> PhysicalFingerprint {
    crate::vector_tensor_4d::exact_canonical_fixture()
}

/// Independent gauge-free holdout with one intrinsic central generator.
pub fn central_hypermultiplet_fixture() -> PhysicalFingerprint {
    crate::central_hypermultiplet_4d::exact_canonical_fixture()
}

fn summary(fixture: &PhysicalFingerprint) -> Result<AdaptedFixtureSummary, CanonicalError> {
    let maximum_reducibility_stage = fixture
        .components
        .iter()
        .filter_map(|component| match component.role {
            ComponentRole::GaugeParameter { stage } => Some(stage),
            _ => None,
        })
        .max();
    let imaginary_linkage_coefficients = fixture
        .linkage
        .iter()
        .filter(|term| term.coefficient.imaginary().numerator() != 0)
        .count();
    let fractional_linkage_coefficients = fixture
        .linkage
        .iter()
        .filter(|term| {
            term.coefficient.real().denominator() != 1
                || term.coefficient.imaginary().denominator() != 1
        })
        .count();
    Ok(AdaptedFixtureSummary {
        name: fixture.name.clone(),
        canonical: canonicalize(fixture, &CanonicalOptions::default())?,
        components: fixture.components.len(),
        supercharges: fixture.supercharges.len(),
        linkage_terms: fixture.linkage.len(),
        gauge_arrows: fixture.gauge_complex.len(),
        bianchi_relations: fixture.bianchi_identities.len(),
        maximum_reducibility_stage,
        imaginary_linkage_coefficients,
        fractional_linkage_coefficients,
        central_generators: fixture.central_generators.len(),
        central_entries: fixture.central_entries.len(),
        central_occurrences: fixture.central_occurrences.len(),
    })
}

pub fn build() -> Result<PositiveControlAdapters, CanonicalError> {
    let chiral_vector = summary(&chiral_vector_fixture())?;
    let chiral_tensor = summary(&chiral_tensor_fixture())?;
    let vector_tensor = summary(&vector_tensor_fixture())?;
    let central_hypermultiplet = summary(&central_hypermultiplet_fixture())?;
    let hashes = [
        &chiral_vector.canonical.sha256,
        &chiral_tensor.canonical.sha256,
        &vector_tensor.canonical.sha256,
        &central_hypermultiplet.canonical.sha256,
    ];
    let fingerprints_distinct = (0..hashes.len())
        .all(|left| ((left + 1)..hashes.len()).all(|right| hashes[left] != hashes[right]));
    let validation_passed = fingerprints_distinct
        && chiral_vector.supercharges == 8
        && chiral_tensor.supercharges == 8
        && chiral_vector.maximum_reducibility_stage == Some(0)
        && chiral_tensor.maximum_reducibility_stage == Some(1)
        && chiral_vector.bianchi_relations == 4
        && chiral_tensor.bianchi_relations == 1
        && vector_tensor.supercharges == 8
        && vector_tensor.maximum_reducibility_stage == Some(1)
        && vector_tensor.bianchi_relations == 5
        && vector_tensor.central_generators == 1
        && vector_tensor.central_entries > 0
        && vector_tensor.central_occurrences > 0
        && central_hypermultiplet.supercharges == 8
        && central_hypermultiplet.maximum_reducibility_stage.is_none()
        && central_hypermultiplet.gauge_arrows == 0
        && central_hypermultiplet.central_generators == 1
        && central_hypermultiplet.central_entries > 0
        && central_hypermultiplet.central_occurrences == 4
        && chiral_vector.imaginary_linkage_coefficients > 0
        && chiral_tensor.imaginary_linkage_coefficients > 0
        && chiral_tensor.fractional_linkage_coefficients > 0;
    Ok(PositiveControlAdapters {
        schema_version: "higher-dimensional-fixture-adapters-v2",
        chiral_vector,
        chiral_tensor,
        vector_tensor,
        central_hypermultiplet,
        fingerprints_distinct,
        validation_passed,
        vector_tensor_boundary: "The exact 1405 component fixture and the fixed 960 Eq. (4.6) bosonic, fermionic, and simultaneous central/color zero-brane bridge are canonicalized. A direct repaired Eq. (4.5) reduction to all 512 Appendix F entries remains open.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::higher_dimensional_canonical::{BasisAction, DerivativeMonomial, GaussianRational};

    fn rephase_component(fixture: &mut PhysicalFingerprint, component: usize) {
        for term in &mut fixture.linkage {
            let occurrences =
                usize::from(term.source == component) + usize::from(term.target == component);
            if occurrences % 2 == 1 {
                term.coefficient = term.coefficient.negated().unwrap();
            }
        }
        for term in &mut fixture.gauge_complex {
            let occurrences =
                usize::from(term.parameter == component) + usize::from(term.target == component);
            if occurrences % 2 == 1 {
                term.coefficient = term.coefficient.negated().unwrap();
            }
        }
        for identity in &mut fixture.bianchi_identities {
            for term in &mut identity.terms {
                if term.component == component {
                    term.coefficient = term.coefficient.negated().unwrap();
                }
            }
        }
        for entry in &mut fixture.central_entries {
            let occurrences =
                usize::from(entry.source == component) + usize::from(entry.target == component);
            if occurrences % 2 == 1 {
                entry.coefficient = entry.coefficient.negated().unwrap();
            }
        }
    }

    fn component_sign_action(fixture: &PhysicalFingerprint, component: usize) -> BasisAction {
        let mut action = BasisAction::identity(fixture);
        action.component_signs[component] = -1;
        action
    }

    #[test]
    fn exact_positive_controls_retain_different_physics() {
        let artifact = build().unwrap();
        assert!(artifact.validation_passed);
        assert!(artifact.fingerprints_distinct);
        assert_eq!(artifact.chiral_vector.components, 24);
        assert_eq!(artifact.chiral_tensor.components, 28);
        assert_eq!(artifact.chiral_vector.gauge_arrows, 4);
        assert_eq!(artifact.chiral_tensor.gauge_arrows, 16);
        assert_eq!(artifact.chiral_vector.linkage_terms, 424);
        assert_eq!(artifact.chiral_tensor.linkage_terms, 440);
        assert_eq!(artifact.vector_tensor.components, 36);
        assert_eq!(artifact.vector_tensor.linkage_terms, 544);
        assert_eq!(artifact.vector_tensor.gauge_arrows, 20);
        assert_eq!(artifact.vector_tensor.bianchi_relations, 5);
        assert_eq!(artifact.vector_tensor.central_generators, 1);
        assert_eq!(artifact.vector_tensor.central_entries, 71);
        assert_eq!(artifact.vector_tensor.central_occurrences, 4);
        assert_eq!(artifact.central_hypermultiplet.components, 16);
        assert_eq!(artifact.central_hypermultiplet.gauge_arrows, 0);
        assert_eq!(artifact.central_hypermultiplet.central_generators, 1);
        assert_eq!(artifact.central_hypermultiplet.central_occurrences, 4);
    }

    #[test]
    fn explicit_field_rephasing_is_canonically_invariant() {
        for fixture in [
            chiral_vector_fixture(),
            chiral_tensor_fixture(),
            vector_tensor_fixture(),
            central_hypermultiplet_fixture(),
        ] {
            let component = 0;
            assert_eq!(
                fixture.components[component].role,
                ComponentRole::Propagating
            );
            let action = component_sign_action(&fixture, component);
            let options = CanonicalOptions {
                generators: vec![action],
                max_group_order: 4,
            };
            let baseline = canonicalize(&fixture, &options).unwrap();
            let mut rephased = fixture.clone();
            rephase_component(&mut rephased, component);
            let transformed = canonicalize(&rephased, &options).unwrap();
            assert_eq!(baseline.sha256, transformed.sha256);
        }
    }

    #[test]
    fn derivative_and_gauge_reducibility_mutations_are_detected() {
        let vector = chiral_vector_fixture();
        let vector_hash = canonicalize(&vector, &CanonicalOptions::default())
            .unwrap()
            .sha256;
        let mut derivative_mutation = vector.clone();
        let term = derivative_mutation
            .linkage
            .iter_mut()
            .find(|term| term.derivative.spatial_order() > 0)
            .unwrap();
        term.derivative = DerivativeMonomial::TEMPORAL;
        assert_ne!(
            vector_hash,
            canonicalize(&derivative_mutation, &CanonicalOptions::default())
                .unwrap()
                .sha256
        );

        let tensor = chiral_tensor_fixture();
        let tensor_hash = canonicalize(&tensor, &CanonicalOptions::default())
            .unwrap()
            .sha256;
        let mut reducibility_mutation = tensor.clone();
        let components = reducibility_mutation.components.clone();
        reducibility_mutation.gauge_complex.retain(|term| {
            components[term.parameter].role != (ComponentRole::GaugeParameter { stage: 1 })
        });
        assert_ne!(
            tensor_hash,
            canonicalize(&reducibility_mutation, &CanonicalOptions::default())
                .unwrap()
                .sha256
        );
    }

    #[test]
    fn bianchi_relative_coefficient_mutation_is_detected() {
        let tensor = chiral_tensor_fixture();
        let baseline = canonicalize(&tensor, &CanonicalOptions::default()).unwrap();
        let mut mutation = tensor.clone();
        mutation.bianchi_identities[0].terms[1].coefficient =
            GaussianRational::new(-3, 2, 1, 5).unwrap();
        let changed = canonicalize(&mutation, &CanonicalOptions::default()).unwrap();
        assert_ne!(baseline.sha256, changed.sha256);
    }

    #[test]
    fn every_linkage_and_gauge_arrow_respects_exact_height() {
        for fixture in [
            chiral_vector_fixture(),
            chiral_tensor_fixture(),
            vector_tensor_fixture(),
        ] {
            for term in &fixture.linkage {
                let derivative_height = 2 * i16::from(term.derivative.0.iter().sum::<u8>());
                assert_eq!(
                    fixture.components[term.source].height_twice
                        + fixture.supercharges[term.charge].height_twice,
                    fixture.components[term.target].height_twice + derivative_height
                );
            }
            for term in &fixture.gauge_complex {
                let derivative_height = 2 * i16::from(term.derivative.0.iter().sum::<u8>());
                assert_eq!(
                    fixture.components[term.parameter].height_twice + derivative_height,
                    fixture.components[term.target].height_twice
                );
            }
            for identity in &fixture.bianchi_identities {
                let heights: Vec<_> = identity
                    .terms
                    .iter()
                    .map(|term| {
                        fixture.components[term.component].height_twice
                            + 2 * i16::from(term.derivative.0.iter().sum::<u8>())
                    })
                    .collect();
                assert!(heights.iter().all(|height| *height == heights[0]));
            }
        }
    }

    #[test]
    fn adapters_are_gated_by_the_existing_exact_closure_fixtures() {
        assert!(crate::chiral_vector_4d::verify().passed);
        assert!(crate::chiral_tensor_4d::verify().passed);
    }

    #[test]
    fn adapted_source_linkage_reproduces_existing_operator_censuses() {
        let cases = [
            (
                chiral_vector_fixture(),
                crate::chiral_vector_4d::higher_dimensional_fingerprint(),
            ),
            (
                chiral_tensor_fixture(),
                crate::chiral_tensor_4d::higher_dimensional_fingerprint(),
            ),
        ];
        for (fixture, existing) in cases {
            let source_terms: Vec<_> = fixture
                .linkage
                .iter()
                .filter(|term| term.source < existing.raw_component_fields)
                .collect();
            assert_eq!(
                source_terms.len(),
                existing.derivative_operator.transformation_terms
            );
            assert_eq!(
                source_terms
                    .iter()
                    .filter(|term| term.derivative.0 == [0; 4])
                    .count(),
                existing.derivative_operator.algebraic_terms
            );
            assert_eq!(
                source_terms
                    .iter()
                    .filter(|term| term.derivative.temporal_order() > 0)
                    .count(),
                existing.derivative_operator.temporal_derivative_terms
            );
            assert_eq!(
                source_terms
                    .iter()
                    .filter(|term| term.derivative.spatial_order() > 0)
                    .count(),
                existing.derivative_operator.spatial_derivative_terms
            );
        }
    }
}
