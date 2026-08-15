//! Exact sparse jet polynomials for sourced component-algebra checks.

use crate::chiral_vector_4d::GaussianRational;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Jet<F, const DIM: usize> {
    pub(crate) field: F,
    pub(crate) derivatives: [u8; DIM],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Polynomial<F: Ord, const DIM: usize>(
    pub(crate) BTreeMap<Jet<F, DIM>, GaussianRational>,
);

impl<F: Copy + Ord, const DIM: usize> Default for Polynomial<F, DIM> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<F: Copy + Ord, const DIM: usize> Polynomial<F, DIM> {
    pub(crate) fn atom(field: F) -> Self {
        Self(BTreeMap::from([(
            Jet {
                field,
                derivatives: [0; DIM],
            },
            GaussianRational::new(1, 0),
        )]))
    }

    pub(crate) fn add_term(
        &mut self,
        field: F,
        derivatives: [u8; DIM],
        coefficient: GaussianRational,
    ) {
        if coefficient.is_zero() {
            return;
        }
        let jet = Jet { field, derivatives };
        let entry = self.0.entry(jet.clone()).or_default();
        entry.add_assign(&coefficient);
        if entry.is_zero() {
            self.0.remove(&jet);
        }
    }

    pub(crate) fn add_scaled(&mut self, other: &Self, coefficient: GaussianRational) {
        for (jet, value) in &other.0 {
            self.add_term(jet.field, jet.derivatives, value.mul(&coefficient));
        }
    }

    pub(crate) fn derivative(&self, axis: usize) -> Self {
        let mut result = Self::default();
        for (jet, coefficient) in &self.0 {
            let mut derivatives = jet.derivatives;
            derivatives[axis] += 1;
            result.add_term(jet.field, derivatives, *coefficient);
        }
        result
    }

    pub(crate) fn term_count(&self) -> usize {
        self.0.len()
    }
}
