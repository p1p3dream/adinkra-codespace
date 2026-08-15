//! Exact canonical fingerprints for higher-dimensional component multiplets.
//!
//! A fingerprint retains data that ordinary worldline adjacency forgets:
//! Lorentz type, engineering height, temporal and spatial derivative linkage,
//! the complete gauge complex, Bianchi relations, and central operators.  The
//! canonicalizer takes explicit generators for the admissible finite basis
//! group.  It enumerates that group exactly and chooses the lexicographically
//! least normalized presentation.  Human-readable labels and provenance are
//! deliberately excluded from the invariant.  Every linkage coefficient is a
//! normalized exact element of `Q(i)`; no floating-point arithmetic enters a
//! fingerprint.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{self, Write};

/// A normalized exact rational number.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, CanonicalError> {
        if denominator == 0 {
            return Err(CanonicalError::ZeroDenominator);
        }
        if numerator == 0 {
            return Ok(Self {
                numerator: 0,
                denominator: 1,
            });
        }
        let mut numerator = numerator;
        let mut denominator = denominator;
        if denominator < 0 {
            numerator = numerator
                .checked_neg()
                .ok_or(CanonicalError::IntegerOverflow)?;
            denominator = denominator
                .checked_neg()
                .ok_or(CanonicalError::IntegerOverflow)?;
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs()) as i64;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn integer(value: i64) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    pub fn numerator(self) -> i64 {
        self.numerator
    }

    pub fn denominator(self) -> i64 {
        self.denominator
    }

    fn is_zero(self) -> bool {
        self.numerator == 0
    }

    fn add(self, other: Self) -> Result<Self, CanonicalError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or(CanonicalError::IntegerOverflow)?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or(CanonicalError::IntegerOverflow)?;
        let numerator = left
            .checked_add(right)
            .ok_or(CanonicalError::IntegerOverflow)?;
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .ok_or(CanonicalError::IntegerOverflow)?;
        Self::new(numerator, denominator)
    }

    fn sub(self, other: Self) -> Result<Self, CanonicalError> {
        self.add(other.neg()?)
    }

    fn mul(self, other: Self) -> Result<Self, CanonicalError> {
        Self::new(
            self.numerator
                .checked_mul(other.numerator)
                .ok_or(CanonicalError::IntegerOverflow)?,
            self.denominator
                .checked_mul(other.denominator)
                .ok_or(CanonicalError::IntegerOverflow)?,
        )
    }

    fn div(self, other: Self) -> Result<Self, CanonicalError> {
        if other.is_zero() {
            return Err(CanonicalError::ZeroDenominator);
        }
        Self::new(
            self.numerator
                .checked_mul(other.denominator)
                .ok_or(CanonicalError::IntegerOverflow)?,
            self.denominator
                .checked_mul(other.numerator)
                .ok_or(CanonicalError::IntegerOverflow)?,
        )
    }

    fn neg(self) -> Result<Self, CanonicalError> {
        Self::new(
            self.numerator
                .checked_neg()
                .ok_or(CanonicalError::IntegerOverflow)?,
            self.denominator,
        )
    }

    fn encoding(self) -> String {
        format!("{}/{}", self.numerator, self.denominator)
    }
}

/// An exact element of the Gaussian-rational field `Q(i)`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct GaussianRational {
    real: Rational,
    imaginary: Rational,
}

impl GaussianRational {
    pub fn new(
        real_numerator: i64,
        real_denominator: i64,
        imaginary_numerator: i64,
        imaginary_denominator: i64,
    ) -> Result<Self, CanonicalError> {
        Ok(Self {
            real: Rational::new(real_numerator, real_denominator)?,
            imaginary: Rational::new(imaginary_numerator, imaginary_denominator)?,
        })
    }

    pub const fn integer(real: i64, imaginary: i64) -> Self {
        Self {
            real: Rational::integer(real),
            imaginary: Rational::integer(imaginary),
        }
    }

    pub fn real(self) -> Rational {
        self.real
    }

    pub fn imaginary(self) -> Rational {
        self.imaginary
    }

    pub fn is_zero(self) -> bool {
        self.real.is_zero() && self.imaginary.is_zero()
    }

    pub fn add_exact(self, other: Self) -> Result<Self, CanonicalError> {
        Ok(Self {
            real: self.real.add(other.real)?,
            imaginary: self.imaginary.add(other.imaginary)?,
        })
    }

    pub fn multiply(self, other: Self) -> Result<Self, CanonicalError> {
        let real = self
            .real
            .mul(other.real)?
            .sub(self.imaginary.mul(other.imaginary)?)?;
        let imaginary = self
            .real
            .mul(other.imaginary)?
            .add(self.imaginary.mul(other.real)?)?;
        Ok(Self { real, imaginary })
    }

    pub fn divide(self, other: Self) -> Result<Self, CanonicalError> {
        if other.is_zero() {
            return Err(CanonicalError::ZeroDenominator);
        }
        // (a+ib)/(c+id) = ((ac+bd)+i(bc-ad))/(c^2+d^2).
        let norm = other
            .real
            .mul(other.real)?
            .add(other.imaginary.mul(other.imaginary)?)?;
        let real = self
            .real
            .mul(other.real)?
            .add(self.imaginary.mul(other.imaginary)?)?
            .div(norm)?;
        let imaginary = self
            .imaginary
            .mul(other.real)?
            .sub(self.real.mul(other.imaginary)?)?
            .div(norm)?;
        Ok(Self { real, imaginary })
    }

    pub fn negated(self) -> Result<Self, CanonicalError> {
        Ok(Self {
            real: self.real.neg()?,
            imaginary: self.imaginary.neg()?,
        })
    }

    fn times_sign(self, sign: i8) -> Result<Self, CanonicalError> {
        match sign {
            1 => Ok(self),
            -1 => self.negated(),
            _ => Err(CanonicalError::InvalidSign("coefficient multiplier")),
        }
    }

    fn encoding(self) -> String {
        format!("({},{})", self.real.encoding(), self.imaginary.encoding())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub enum Statistics {
    Boson,
    Fermion,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub enum Reality {
    Real,
    Complex,
    PseudoReal,
}

/// Four-dimensional Lorentz type `(2j_L, 2j_R)`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct LorentzRep {
    pub left_twice_spin: u8,
    pub right_twice_spin: u8,
    pub reality: Reality,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub enum ComponentRole {
    Propagating,
    Auxiliary,
    GaugePotential,
    FieldStrength,
    /// Gauge-for-gauge stage zero is an ordinary gauge parameter.
    GaugeParameter {
        stage: u8,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Component {
    /// Documentation only.  It is not part of the canonical invariant.
    pub label: String,
    pub statistics: Statistics,
    pub lorentz: LorentzRep,
    /// Twice the engineering height, so half-integral heights remain exact.
    pub height_twice: i16,
    pub role: ComponentRole,
    pub form_degree: Option<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Supercharge {
    /// Documentation only.  It is not part of the canonical invariant.
    pub label: String,
    pub lorentz: LorentzRep,
    pub height_twice: i16,
}

/// A commuting derivative monomial in `(d_t, d_x, d_y, d_z)`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DerivativeMonomial(pub [u8; 4]);

impl DerivativeMonomial {
    pub const IDENTITY: Self = Self([0, 0, 0, 0]);
    pub const TEMPORAL: Self = Self([1, 0, 0, 0]);

    pub const fn spatial(axis: usize) -> Self {
        let mut powers = [0; 4];
        powers[axis + 1] = 1;
        Self(powers)
    }

    pub fn temporal_order(self) -> u8 {
        self.0[0]
    }

    pub fn spatial_order(self) -> u16 {
        self.0[1..].iter().map(|&power| u16::from(power)).sum()
    }
}

/// One exact term in `Q_charge component_source`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LinkageTerm {
    pub charge: usize,
    pub source: usize,
    pub target: usize,
    pub derivative: DerivativeMonomial,
    pub coefficient: GaussianRational,
}

/// One exact arrow in the gauge complex.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GaugeTerm {
    pub parameter: usize,
    pub target: usize,
    pub derivative: DerivativeMonomial,
    pub coefficient: GaussianRational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LinearTerm {
    pub component: usize,
    pub derivative: DerivativeMonomial,
    pub coefficient: GaussianRational,
}

/// A homogeneous linear Bianchi relation, understood up to nonzero scale.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BianchiIdentity {
    pub terms: Vec<LinearTerm>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CentralGenerator {
    /// Documentation only.  It is not part of the canonical invariant.
    pub label: String,
    pub lorentz: LorentzRep,
    pub height_twice: i16,
}

/// One matrix entry in the action of a central generator on components.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CentralEntry {
    pub generator: usize,
    pub source: usize,
    pub target: usize,
    pub derivative: DerivativeMonomial,
    pub coefficient: GaussianRational,
}

/// A central term appearing in one unordered supercharge anticommutator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CentralOccurrence {
    pub left_charge: usize,
    pub right_charge: usize,
    pub generator: usize,
    pub coefficient: GaussianRational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalFingerprint {
    /// Documentation only.  It is not part of the canonical invariant.
    pub name: String,
    pub components: Vec<Component>,
    pub supercharges: Vec<Supercharge>,
    pub linkage: Vec<LinkageTerm>,
    pub gauge_complex: Vec<GaugeTerm>,
    pub bianchi_identities: Vec<BianchiIdentity>,
    pub central_generators: Vec<CentralGenerator>,
    pub central_entries: Vec<CentralEntry>,
    pub central_occurrences: Vec<CentralOccurrence>,
}

/// A signed permutation generator, with every map directed old index to new.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct BasisAction {
    pub component_permutation: Vec<usize>,
    pub component_signs: Vec<i8>,
    pub charge_permutation: Vec<usize>,
    pub charge_signs: Vec<i8>,
    pub central_permutation: Vec<usize>,
    pub central_signs: Vec<i8>,
}

impl BasisAction {
    pub fn identity(fingerprint: &PhysicalFingerprint) -> Self {
        Self {
            component_permutation: (0..fingerprint.components.len()).collect(),
            component_signs: vec![1; fingerprint.components.len()],
            charge_permutation: (0..fingerprint.supercharges.len()).collect(),
            charge_signs: vec![1; fingerprint.supercharges.len()],
            central_permutation: (0..fingerprint.central_generators.len()).collect(),
            central_signs: vec![1; fingerprint.central_generators.len()],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CanonicalOptions {
    /// Generators of the finite physical basis group.
    pub generators: Vec<BasisAction>,
    /// Hard failure limit, never a truncation of the canonical orbit.
    pub max_group_order: usize,
}

impl Default for CanonicalOptions {
    fn default() -> Self {
        Self {
            generators: Vec::new(),
            max_group_order: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CanonicalFingerprint {
    pub schema_version: &'static str,
    pub sha256: String,
    pub canonical_text: String,
    pub group_order: usize,
    pub component_count: usize,
    pub supercharge_count: usize,
    pub central_generator_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalError {
    EmptyBianchiIdentity(usize),
    IndexOutOfRange(&'static str, usize),
    InvalidGaugeParameter(usize),
    InvalidPermutation(&'static str),
    InvalidSign(&'static str),
    IncompatibleRelabeling(&'static str, usize, usize),
    ZeroCoefficient(&'static str, usize),
    GroupLimitExceeded(usize),
    IntegerOverflow,
    ZeroDenominator,
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CanonicalError {}

/// Canonicalize under the exact finite group generated by `options`.
pub fn canonicalize(
    fingerprint: &PhysicalFingerprint,
    options: &CanonicalOptions,
) -> Result<CanonicalFingerprint, CanonicalError> {
    validate_fingerprint(fingerprint)?;
    let group = generated_group(fingerprint, options)?;
    let mut best: Option<String> = None;
    for action in &group {
        let encoded = encode_action(fingerprint, action)?;
        if best.as_ref().is_none_or(|current| encoded < *current) {
            best = Some(encoded);
        }
    }
    let canonical_text = best.expect("the generated group contains the identity");
    let mut hasher = Sha256::new();
    hasher.update(canonical_text.as_bytes());
    Ok(CanonicalFingerprint {
        schema_version: "higher-dimensional-canonical-v2",
        sha256: format!("{:x}", hasher.finalize()),
        canonical_text,
        group_order: group.len(),
        component_count: fingerprint.components.len(),
        supercharge_count: fingerprint.supercharges.len(),
        central_generator_count: fingerprint.central_generators.len(),
    })
}

fn component_signature(
    component: &Component,
) -> (Statistics, LorentzRep, i16, ComponentRole, Option<u8>) {
    (
        component.statistics,
        component.lorentz,
        component.height_twice,
        component.role,
        component.form_degree,
    )
}

fn charge_signature(charge: &Supercharge) -> (LorentzRep, i16) {
    (charge.lorentz, charge.height_twice)
}

fn central_signature(generator: &CentralGenerator) -> (LorentzRep, i16) {
    (generator.lorentz, generator.height_twice)
}

fn check_index(index: usize, len: usize, kind: &'static str) -> Result<(), CanonicalError> {
    if index < len {
        Ok(())
    } else {
        Err(CanonicalError::IndexOutOfRange(kind, index))
    }
}

fn validate_fingerprint(f: &PhysicalFingerprint) -> Result<(), CanonicalError> {
    for (index, term) in f.linkage.iter().enumerate() {
        check_index(term.charge, f.supercharges.len(), "linkage charge")?;
        check_index(term.source, f.components.len(), "linkage source")?;
        check_index(term.target, f.components.len(), "linkage target")?;
        if term.coefficient.is_zero() {
            return Err(CanonicalError::ZeroCoefficient("linkage", index));
        }
    }
    for (index, term) in f.gauge_complex.iter().enumerate() {
        check_index(term.parameter, f.components.len(), "gauge parameter")?;
        check_index(term.target, f.components.len(), "gauge target")?;
        if !matches!(
            f.components[term.parameter].role,
            ComponentRole::GaugeParameter { .. }
        ) {
            return Err(CanonicalError::InvalidGaugeParameter(term.parameter));
        }
        if term.coefficient.is_zero() {
            return Err(CanonicalError::ZeroCoefficient("gauge complex", index));
        }
    }
    for (identity_index, identity) in f.bianchi_identities.iter().enumerate() {
        if identity.terms.is_empty() {
            return Err(CanonicalError::EmptyBianchiIdentity(identity_index));
        }
        for (term_index, term) in identity.terms.iter().enumerate() {
            check_index(term.component, f.components.len(), "Bianchi component")?;
            if term.coefficient.is_zero() {
                return Err(CanonicalError::ZeroCoefficient("Bianchi", term_index));
            }
        }
    }
    for (index, entry) in f.central_entries.iter().enumerate() {
        check_index(
            entry.generator,
            f.central_generators.len(),
            "central generator",
        )?;
        check_index(entry.source, f.components.len(), "central source")?;
        check_index(entry.target, f.components.len(), "central target")?;
        if entry.coefficient.is_zero() {
            return Err(CanonicalError::ZeroCoefficient("central entry", index));
        }
    }
    for (index, occurrence) in f.central_occurrences.iter().enumerate() {
        check_index(
            occurrence.left_charge,
            f.supercharges.len(),
            "central left charge",
        )?;
        check_index(
            occurrence.right_charge,
            f.supercharges.len(),
            "central right charge",
        )?;
        check_index(
            occurrence.generator,
            f.central_generators.len(),
            "central occurrence",
        )?;
        if occurrence.coefficient.is_zero() {
            return Err(CanonicalError::ZeroCoefficient("central occurrence", index));
        }
    }
    Ok(())
}

fn validate_signed_permutation(
    permutation: &[usize],
    signs: &[i8],
    kind: &'static str,
) -> Result<(), CanonicalError> {
    if permutation.len() != signs.len() {
        return Err(CanonicalError::InvalidPermutation(kind));
    }
    let mut seen = vec![false; permutation.len()];
    for &target in permutation {
        if target >= permutation.len() || seen[target] {
            return Err(CanonicalError::InvalidPermutation(kind));
        }
        seen[target] = true;
    }
    if signs.iter().any(|&sign| sign != 1 && sign != -1) {
        return Err(CanonicalError::InvalidSign(kind));
    }
    Ok(())
}

fn validate_action(f: &PhysicalFingerprint, action: &BasisAction) -> Result<(), CanonicalError> {
    validate_signed_permutation(
        &action.component_permutation,
        &action.component_signs,
        "component",
    )?;
    validate_signed_permutation(
        &action.charge_permutation,
        &action.charge_signs,
        "supercharge",
    )?;
    validate_signed_permutation(
        &action.central_permutation,
        &action.central_signs,
        "central generator",
    )?;
    if action.component_permutation.len() != f.components.len() {
        return Err(CanonicalError::InvalidPermutation("component"));
    }
    if action.charge_permutation.len() != f.supercharges.len() {
        return Err(CanonicalError::InvalidPermutation("supercharge"));
    }
    if action.central_permutation.len() != f.central_generators.len() {
        return Err(CanonicalError::InvalidPermutation("central generator"));
    }
    for (old, &new) in action.component_permutation.iter().enumerate() {
        if component_signature(&f.components[old]) != component_signature(&f.components[new]) {
            return Err(CanonicalError::IncompatibleRelabeling(
                "component",
                old,
                new,
            ));
        }
    }
    for (old, &new) in action.charge_permutation.iter().enumerate() {
        if charge_signature(&f.supercharges[old]) != charge_signature(&f.supercharges[new]) {
            return Err(CanonicalError::IncompatibleRelabeling(
                "supercharge",
                old,
                new,
            ));
        }
    }
    for (old, &new) in action.central_permutation.iter().enumerate() {
        if central_signature(&f.central_generators[old])
            != central_signature(&f.central_generators[new])
        {
            return Err(CanonicalError::IncompatibleRelabeling(
                "central generator",
                old,
                new,
            ));
        }
    }
    Ok(())
}

fn compose(first: &BasisAction, second: &BasisAction) -> BasisAction {
    fn sector(
        first_permutation: &[usize],
        first_signs: &[i8],
        second_permutation: &[usize],
        second_signs: &[i8],
    ) -> (Vec<usize>, Vec<i8>) {
        let permutation = first_permutation
            .iter()
            .map(|&middle| second_permutation[middle])
            .collect();
        let signs = first_permutation
            .iter()
            .enumerate()
            .map(|(old, &middle)| first_signs[old] * second_signs[middle])
            .collect();
        (permutation, signs)
    }
    let (component_permutation, component_signs) = sector(
        &first.component_permutation,
        &first.component_signs,
        &second.component_permutation,
        &second.component_signs,
    );
    let (charge_permutation, charge_signs) = sector(
        &first.charge_permutation,
        &first.charge_signs,
        &second.charge_permutation,
        &second.charge_signs,
    );
    let (central_permutation, central_signs) = sector(
        &first.central_permutation,
        &first.central_signs,
        &second.central_permutation,
        &second.central_signs,
    );
    BasisAction {
        component_permutation,
        component_signs,
        charge_permutation,
        charge_signs,
        central_permutation,
        central_signs,
    }
}

fn generated_group(
    f: &PhysicalFingerprint,
    options: &CanonicalOptions,
) -> Result<Vec<BasisAction>, CanonicalError> {
    if options.max_group_order == 0 {
        return Err(CanonicalError::GroupLimitExceeded(0));
    }
    for generator in &options.generators {
        validate_action(f, generator)?;
    }
    let identity = BasisAction::identity(f);
    let mut group = vec![identity.clone()];
    let mut indices = HashMap::from([(identity, 0usize)]);
    let mut queue = VecDeque::from([0usize]);
    while let Some(index) = queue.pop_front() {
        for generator in &options.generators {
            let candidate = compose(&group[index], generator);
            if !indices.contains_key(&candidate) {
                if group.len() == options.max_group_order {
                    return Err(CanonicalError::GroupLimitExceeded(options.max_group_order));
                }
                let next_index = group.len();
                indices.insert(candidate.clone(), next_index);
                group.push(candidate);
                queue.push_back(next_index);
            }
        }
    }
    Ok(group)
}

fn sign_product(values: &[i8]) -> i8 {
    values.iter().copied().product()
}

fn add_sparse<K: Ord>(
    map: &mut BTreeMap<K, GaussianRational>,
    key: K,
    coefficient: GaussianRational,
) -> Result<(), CanonicalError> {
    let new_value = map
        .get(&key)
        .copied()
        .unwrap_or(GaussianRational::integer(0, 0))
        .add_exact(coefficient)?;
    if new_value.is_zero() {
        map.remove(&key);
    } else {
        map.insert(key, new_value);
    }
    Ok(())
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn normalized_relation(
    relation: &BianchiIdentity,
    action: &BasisAction,
) -> Result<Vec<(usize, DerivativeMonomial, GaussianRational)>, CanonicalError> {
    let mut terms = BTreeMap::new();
    for term in &relation.terms {
        let coefficient = term
            .coefficient
            .times_sign(action.component_signs[term.component])?;
        add_sparse(
            &mut terms,
            (
                action.component_permutation[term.component],
                term.derivative,
            ),
            coefficient,
        )?;
    }
    if terms.is_empty() {
        return Err(CanonicalError::EmptyBianchiIdentity(0));
    }
    // Division by the first nonzero coefficient fixes the full Q(i)^* scale,
    // not merely a real sign or an integer gcd.
    let pivot = *terms.values().next().expect("nonempty relation");
    terms
        .into_iter()
        .map(|((component, derivative), coefficient)| {
            Ok((component, derivative, coefficient.divide(pivot)?))
        })
        .collect::<Result<Vec<_>, CanonicalError>>()
}

fn encode_action(f: &PhysicalFingerprint, action: &BasisAction) -> Result<String, CanonicalError> {
    let mut out = String::new();
    writeln!(out, "HDCF2").expect("write to String");

    let mut components = vec![None; f.components.len()];
    for (old, component) in f.components.iter().enumerate() {
        components[action.component_permutation[old]] = Some(component_signature(component));
    }
    for signature in components.into_iter().flatten() {
        writeln!(out, "F|{:?}", signature).expect("write to String");
    }

    let mut charges = vec![None; f.supercharges.len()];
    for (old, charge) in f.supercharges.iter().enumerate() {
        charges[action.charge_permutation[old]] = Some(charge_signature(charge));
    }
    for signature in charges.into_iter().flatten() {
        writeln!(out, "Q|{:?}", signature).expect("write to String");
    }

    let mut central_generators = vec![None; f.central_generators.len()];
    for (old, generator) in f.central_generators.iter().enumerate() {
        central_generators[action.central_permutation[old]] = Some(central_signature(generator));
    }
    for signature in central_generators.into_iter().flatten() {
        writeln!(out, "Z|{:?}", signature).expect("write to String");
    }

    let mut linkage = BTreeMap::new();
    for term in &f.linkage {
        let coefficient = term.coefficient.times_sign(sign_product(&[
            action.charge_signs[term.charge],
            action.component_signs[term.source],
            action.component_signs[term.target],
        ]))?;
        add_sparse(
            &mut linkage,
            (
                action.charge_permutation[term.charge],
                action.component_permutation[term.source],
                action.component_permutation[term.target],
                term.derivative,
            ),
            coefficient,
        )?;
    }
    for (key, coefficient) in linkage {
        writeln!(out, "L|{:?}|{}", key, coefficient.encoding()).expect("write to String");
    }

    let mut gauge = BTreeMap::new();
    for term in &f.gauge_complex {
        let coefficient = term.coefficient.times_sign(sign_product(&[
            action.component_signs[term.parameter],
            action.component_signs[term.target],
        ]))?;
        add_sparse(
            &mut gauge,
            (
                action.component_permutation[term.parameter],
                action.component_permutation[term.target],
                term.derivative,
            ),
            coefficient,
        )?;
    }
    for (key, coefficient) in gauge {
        writeln!(out, "G|{:?}|{}", key, coefficient.encoding()).expect("write to String");
    }

    let mut bianchi = f
        .bianchi_identities
        .iter()
        .map(|relation| normalized_relation(relation, action))
        .collect::<Result<Vec<_>, _>>()?;
    bianchi.sort();
    for relation in bianchi {
        write!(out, "B").expect("write to String");
        for (component, derivative, coefficient) in relation {
            write!(
                out,
                "|{component}:{:?}:{}",
                derivative,
                coefficient.encoding()
            )
            .expect("write to String");
        }
        writeln!(out).expect("write to String");
    }

    let mut central_entries = BTreeMap::new();
    for entry in &f.central_entries {
        let coefficient = entry.coefficient.times_sign(sign_product(&[
            action.central_signs[entry.generator],
            action.component_signs[entry.source],
            action.component_signs[entry.target],
        ]))?;
        add_sparse(
            &mut central_entries,
            (
                action.central_permutation[entry.generator],
                action.component_permutation[entry.source],
                action.component_permutation[entry.target],
                entry.derivative,
            ),
            coefficient,
        )?;
    }
    for (key, coefficient) in central_entries {
        writeln!(out, "E|{:?}|{}", key, coefficient.encoding()).expect("write to String");
    }

    let mut occurrences = BTreeMap::new();
    for occurrence in &f.central_occurrences {
        let (left, right) = (
            action.charge_permutation[occurrence.left_charge],
            action.charge_permutation[occurrence.right_charge],
        );
        let charges = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        let coefficient = occurrence.coefficient.times_sign(sign_product(&[
            action.charge_signs[occurrence.left_charge],
            action.charge_signs[occurrence.right_charge],
            action.central_signs[occurrence.generator],
        ]))?;
        add_sparse(
            &mut occurrences,
            (
                charges.0,
                charges.1,
                action.central_permutation[occurrence.generator],
            ),
            coefficient,
        )?;
    }
    for (key, coefficient) in occurrences {
        writeln!(out, "O|{:?}|{}", key, coefficient.encoding()).expect("write to String");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn g(real: i64, imaginary: i64) -> GaussianRational {
        GaussianRational::integer(real, imaginary)
    }

    fn scalar() -> LorentzRep {
        LorentzRep {
            left_twice_spin: 0,
            right_twice_spin: 0,
            reality: Reality::Real,
        }
    }

    fn spinor() -> LorentzRep {
        LorentzRep {
            left_twice_spin: 1,
            right_twice_spin: 0,
            reality: Reality::Complex,
        }
    }

    fn component(label: &str, statistics: Statistics, role: ComponentRole) -> Component {
        Component {
            label: label.to_owned(),
            statistics,
            lorentz: if statistics == Statistics::Boson {
                scalar()
            } else {
                spinor()
            },
            height_twice: if statistics == Statistics::Boson {
                0
            } else {
                1
            },
            role,
            form_degree: None,
        }
    }

    fn fixture() -> PhysicalFingerprint {
        PhysicalFingerprint {
            name: "fixture".to_owned(),
            components: vec![
                component("a", Statistics::Boson, ComponentRole::Propagating),
                component("b", Statistics::Boson, ComponentRole::Propagating),
                component("psi", Statistics::Fermion, ComponentRole::Propagating),
                component(
                    "lambda",
                    Statistics::Boson,
                    ComponentRole::GaugeParameter { stage: 0 },
                ),
                component("f", Statistics::Boson, ComponentRole::FieldStrength),
            ],
            supercharges: vec![
                Supercharge {
                    label: "q1".to_owned(),
                    lorentz: spinor(),
                    height_twice: 1,
                },
                Supercharge {
                    label: "q2".to_owned(),
                    lorentz: spinor(),
                    height_twice: 1,
                },
            ],
            linkage: vec![
                LinkageTerm {
                    charge: 0,
                    source: 0,
                    target: 2,
                    derivative: DerivativeMonomial::IDENTITY,
                    coefficient: g(1, 0),
                },
                LinkageTerm {
                    charge: 1,
                    source: 1,
                    target: 2,
                    derivative: DerivativeMonomial::TEMPORAL,
                    coefficient: g(-2, 0),
                },
            ],
            gauge_complex: vec![GaugeTerm {
                parameter: 3,
                target: 0,
                derivative: DerivativeMonomial::spatial(0),
                coefficient: g(1, 0),
            }],
            bianchi_identities: vec![BianchiIdentity {
                terms: vec![
                    LinearTerm {
                        component: 4,
                        derivative: DerivativeMonomial::TEMPORAL,
                        coefficient: g(2, 0),
                    },
                    LinearTerm {
                        component: 0,
                        derivative: DerivativeMonomial::spatial(2),
                        coefficient: g(-4, 0),
                    },
                ],
            }],
            central_generators: vec![CentralGenerator {
                label: "z".to_owned(),
                lorentz: scalar(),
                height_twice: 2,
            }],
            central_entries: vec![CentralEntry {
                generator: 0,
                source: 0,
                target: 1,
                derivative: DerivativeMonomial::IDENTITY,
                coefficient: g(3, 0),
            }],
            central_occurrences: vec![CentralOccurrence {
                left_charge: 0,
                right_charge: 1,
                generator: 0,
                coefficient: g(1, 0),
            }],
        }
    }

    fn swap_and_flip_generator(f: &PhysicalFingerprint) -> BasisAction {
        let mut action = BasisAction::identity(f);
        action.component_permutation.swap(0, 1);
        action.component_signs[0] = -1;
        action.component_signs[2] = -1;
        action.charge_permutation.swap(0, 1);
        action.charge_signs[1] = -1;
        action.central_signs[0] = -1;
        action
    }

    fn apply_fixture_action(f: &PhysicalFingerprint, action: &BasisAction) -> PhysicalFingerprint {
        let mut transformed = f.clone();
        transformed.name = "renamed presentation".to_owned();
        let mut components = vec![f.components[0].clone(); f.components.len()];
        for (old, item) in f.components.iter().enumerate() {
            components[action.component_permutation[old]] = item.clone();
        }
        for (index, item) in components.iter_mut().enumerate() {
            item.label = format!("field_{index}");
        }
        transformed.components = components;
        let mut charges = vec![f.supercharges[0].clone(); f.supercharges.len()];
        for (old, item) in f.supercharges.iter().enumerate() {
            charges[action.charge_permutation[old]] = item.clone();
        }
        transformed.supercharges = charges;
        let mut central = vec![f.central_generators[0].clone(); f.central_generators.len()];
        for (old, item) in f.central_generators.iter().enumerate() {
            central[action.central_permutation[old]] = item.clone();
        }
        transformed.central_generators = central;

        for term in &mut transformed.linkage {
            term.coefficient = term
                .coefficient
                .times_sign(sign_product(&[
                    action.charge_signs[term.charge],
                    action.component_signs[term.source],
                    action.component_signs[term.target],
                ]))
                .unwrap();
            term.charge = action.charge_permutation[term.charge];
            term.source = action.component_permutation[term.source];
            term.target = action.component_permutation[term.target];
        }
        for term in &mut transformed.gauge_complex {
            term.coefficient = term
                .coefficient
                .times_sign(sign_product(&[
                    action.component_signs[term.parameter],
                    action.component_signs[term.target],
                ]))
                .unwrap();
            term.parameter = action.component_permutation[term.parameter];
            term.target = action.component_permutation[term.target];
        }
        for identity in &mut transformed.bianchi_identities {
            for term in &mut identity.terms {
                term.coefficient = term
                    .coefficient
                    .times_sign(action.component_signs[term.component])
                    .unwrap();
                term.component = action.component_permutation[term.component];
            }
        }
        for entry in &mut transformed.central_entries {
            entry.coefficient = entry
                .coefficient
                .times_sign(sign_product(&[
                    action.central_signs[entry.generator],
                    action.component_signs[entry.source],
                    action.component_signs[entry.target],
                ]))
                .unwrap();
            entry.generator = action.central_permutation[entry.generator];
            entry.source = action.component_permutation[entry.source];
            entry.target = action.component_permutation[entry.target];
        }
        for occurrence in &mut transformed.central_occurrences {
            occurrence.coefficient = occurrence
                .coefficient
                .times_sign(sign_product(&[
                    action.charge_signs[occurrence.left_charge],
                    action.charge_signs[occurrence.right_charge],
                    action.central_signs[occurrence.generator],
                ]))
                .unwrap();
            occurrence.left_charge = action.charge_permutation[occurrence.left_charge];
            occurrence.right_charge = action.charge_permutation[occurrence.right_charge];
            occurrence.generator = action.central_permutation[occurrence.generator];
        }
        transformed
    }

    #[test]
    fn signed_relabeling_has_one_exact_canonical_fingerprint() {
        let fixture = fixture();
        let generator = swap_and_flip_generator(&fixture);
        let transformed = apply_fixture_action(&fixture, &generator);
        let options = CanonicalOptions {
            generators: vec![generator.clone()],
            max_group_order: 64,
        };
        let first = canonicalize(&fixture, &options).unwrap();
        let second = canonicalize(&transformed, &options).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.canonical_text, second.canonical_text);
        assert!(first.group_order > 1);
    }

    #[test]
    fn labels_and_relation_scale_are_not_physical() {
        let fixture = fixture();
        let mut renamed = fixture.clone();
        renamed.name = "unrelated prose".to_owned();
        renamed.components[0].label = "alpha".to_owned();
        let gaussian_scale = GaussianRational::new(-7, 2, 3, 5).unwrap();
        for term in &mut renamed.bianchi_identities[0].terms {
            term.coefficient = term.coefficient.multiply(gaussian_scale).unwrap();
        }
        assert_eq!(
            canonicalize(&fixture, &CanonicalOptions::default())
                .unwrap()
                .sha256,
            canonicalize(&renamed, &CanonicalOptions::default())
                .unwrap()
                .sha256
        );
    }

    #[test]
    fn physical_mutations_change_the_fingerprint() {
        let fixture = fixture();
        let baseline = canonicalize(&fixture, &CanonicalOptions::default())
            .unwrap()
            .sha256;
        let mut mutations = Vec::new();

        let mut lorentz = fixture.clone();
        lorentz.components[0].lorentz.left_twice_spin = 2;
        mutations.push(lorentz);

        let mut height = fixture.clone();
        height.components[0].height_twice += 2;
        mutations.push(height);

        let mut spatial = fixture.clone();
        spatial.linkage[1].derivative = DerivativeMonomial::spatial(1);
        mutations.push(spatial);

        let mut gauge = fixture.clone();
        gauge.gauge_complex[0].derivative = DerivativeMonomial::TEMPORAL;
        mutations.push(gauge);

        let mut bianchi = fixture.clone();
        bianchi.bianchi_identities[0].terms[1].coefficient = g(-6, 1);
        mutations.push(bianchi);

        let mut central = fixture.clone();
        central.central_entries[0].coefficient = g(5, -2);
        mutations.push(central);

        for mutation in mutations {
            assert_ne!(
                baseline,
                canonicalize(&mutation, &CanonicalOptions::default())
                    .unwrap()
                    .sha256
            );
        }
    }

    #[test]
    fn reducibility_stage_is_retained() {
        let fixture = fixture();
        let baseline = canonicalize(&fixture, &CanonicalOptions::default()).unwrap();
        let mut mutation = fixture.clone();
        mutation.components[3].role = ComponentRole::GaugeParameter { stage: 1 };
        let changed = canonicalize(&mutation, &CanonicalOptions::default()).unwrap();
        assert_ne!(baseline.sha256, changed.sha256);
    }

    #[test]
    fn invalid_and_truncated_groups_fail_closed() {
        let fixture = fixture();
        let mut incompatible = BasisAction::identity(&fixture);
        incompatible.component_permutation.swap(0, 2);
        assert!(matches!(
            canonicalize(
                &fixture,
                &CanonicalOptions {
                    generators: vec![incompatible],
                    max_group_order: 10,
                }
            ),
            Err(CanonicalError::IncompatibleRelabeling("component", _, _))
        ));

        let generator = swap_and_flip_generator(&fixture);
        assert!(matches!(
            canonicalize(
                &fixture,
                &CanonicalOptions {
                    generators: vec![generator],
                    max_group_order: 1,
                }
            ),
            Err(CanonicalError::GroupLimitExceeded(1))
        ));
    }

    #[test]
    fn gaussian_rationals_are_normalized_exactly_and_serialize_deterministically() {
        let value = GaussianRational::new(2, -4, -6, -8).unwrap();
        assert_eq!(value.real(), Rational::new(-1, 2).unwrap());
        assert_eq!(value.imaginary(), Rational::new(3, 4).unwrap());
        assert_eq!(value.encoding(), "(-1/2,3/4)");
        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            r#"{"real":{"numerator":-1,"denominator":2},"imaginary":{"numerator":3,"denominator":4}}"#
        );

        let multiplier = GaussianRational::new(5, 7, -2, 3).unwrap();
        assert_eq!(
            value
                .multiply(multiplier)
                .unwrap()
                .divide(multiplier)
                .unwrap(),
            value
        );
        assert_eq!(
            GaussianRational::new(1, 0, 0, 1),
            Err(CanonicalError::ZeroDenominator)
        );
    }

    #[test]
    fn rational_duplicate_terms_combine_before_hashing() {
        let fixture = fixture();
        let baseline = canonicalize(&fixture, &CanonicalOptions::default()).unwrap();
        let mut split = fixture.clone();
        split.linkage.remove(0);
        for coefficient in [
            GaussianRational::new(1, 2, 1, 3).unwrap(),
            GaussianRational::new(1, 2, -1, 3).unwrap(),
        ] {
            split.linkage.push(LinkageTerm {
                charge: 0,
                source: 0,
                target: 2,
                derivative: DerivativeMonomial::IDENTITY,
                coefficient,
            });
        }
        assert_eq!(
            baseline.sha256,
            canonicalize(&split, &CanonicalOptions::default())
                .unwrap()
                .sha256
        );
    }
}
