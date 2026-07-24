//! Focused exact analysis of the Siegel-Rocek auxiliary-pair premise.
//!
//! This module does three deliberately narrow things:
//! 1. audits the component-count arithmetic with and without paired auxiliary
//!    spinors;
//! 2. constructs the exact 0-brane, on-shell 10D SYM seed from real SO(9)
//!    gamma matrices and measures its fermionic non-closure;
//! 3. tests whether that seed can occur inside either minimal N=16 adinkra as
//!    a literal coordinate subblock, then computes the exact linear solution
//!    space after arbitrary real field mixing is allowed.
//!
//! A positive linear-space dimension is not an off-shell completion. It is the
//! input to the next, quadratic orthonormal-frame and 10D oxidation gates.

use std::collections::{BTreeMap, HashMap};
use std::fs;

use serde::{Deserialize, Serialize};

use crate::chromotopology::Chromotopology;
use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;
use crate::lr_matrix::AdinkraRep;
use crate::signed_perm::SignedPerm;

const N: usize = 16;
const MIN_D: usize = 128;
const VECTOR_BOSONS: usize = 9;
const SPINOR_FERMIONS: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct ArithmeticAudit {
    pub physical_fermions: usize,
    pub paired_auxiliary_quantum: usize,
    pub unpaired_auxiliary_quantum: usize,
    pub minimal_off_shell_fermions: usize,
    pub paired_equation_has_solution: bool,
    pub unpaired_minimal_spinor_units: usize,
    pub unpaired_total_fermions: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SymSeedAudit {
    pub charges: usize,
    pub bosons: usize,
    pub fermions: usize,
    pub so9_clifford_verified: bool,
    pub bosonic_garden_worst_entry: i32,
    pub fermionic_garden_worst_entry: i32,
    pub diagonal_fermionic_remnant_rank: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopologyAttack {
    pub catalog_index: usize,
    pub code_dimension: usize,
    pub adinkra_dimension: usize,
    pub indecomposable: bool,
    pub dashings_tested: usize,
    pub all_dashings_garden_verified: bool,
    /// Largest number of distinct boson coordinate vectors having exactly the
    /// same set of sixteen colored fermion neighbors.
    pub max_equal_neighbor_sets: usize,
    /// A literal SYM coordinate embedding needs nine such bosons.
    pub literal_coordinate_embedding_possible: bool,
    /// Dimension of the exact homogeneous linear intertwiner space after the
    /// nine bosons are allowed to be arbitrary real vectors.
    pub dashings_with_nonzero_mixed_basis_space: usize,
    pub max_mixed_basis_linear_dimension: usize,
    pub best_dashing_indices: Vec<usize>,
    /// Result after relaxing Q A_i so distinct full fermions may project to the
    /// same physical gaugino component. This is the first auxiliary-spinor gate.
    pub auxiliary_projection: AuxiliaryProjectionAttack,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuxiliaryProjectionAttack {
    /// Support compatibility ignores all dashing signs. A false value excludes
    /// the entire topology for this coordinate-quotient ansatz.
    pub support_compatible: bool,
    pub support_obstruction_is_dashing_independent: bool,
    /// Signed dashings examined until the first exact witness was found.
    pub signed_dashings_examined: usize,
    pub dashings_with_witness: usize,
    pub minimum_exercised_auxiliary_fermion_directions: Option<usize>,
    pub witness: Option<AuxiliaryProjectionWitness>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuxiliaryProjectionWitness {
    pub dashing_index: usize,
    /// Coordinate boson selected for each SO(9) vector index i=0..8.
    pub boson_by_vector: Vec<usize>,
    /// Number of full 128-dimensional fermion coordinates reached by Q on the
    /// nine selected gauge-field coordinates.
    pub reached_full_fermions: usize,
    /// reached_full_fermions - 16. These independent differences lie in the
    /// kernel of the on-shell physical-fermion projection.
    pub exercised_auxiliary_fermion_directions: usize,
    /// Number of full fermion coordinates projecting to each physical beta.
    pub physical_fiber_sizes: Vec<usize>,
    pub uniform_fiber_size: Option<usize>,
    /// Sparse projection certificate: (full fermion, physical beta, sign).
    pub projection: Vec<ProjectionEntry>,
    /// Optional extension that also reproduces Q acting on the physical
    /// gaugino after choosing a coordinate section of the fermion quotient.
    pub two_sided_temporal_seed: Option<TwoSidedSeedWitness>,
    /// Exact integer linear system for the natural uniform section
    /// J_F = (1/8) P_F^T. Solving for P_B tests Q lambda without restricting
    /// the boson projection to coordinate values.
    pub uniform_section_system: Option<UniformSectionSystem>,
    /// Signed-permutation data needed for an independent joint P_B/J_F search.
    pub joint_section_data: Option<JointSectionData>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectionEntry {
    pub full_fermion: usize,
    pub physical_beta: usize,
    pub sign: i8,
}

#[derive(Debug, Clone, Serialize)]
pub struct TwoSidedSeedWitness {
    pub fermion_section: Vec<FermionSectionEntry>,
    pub boson_projection: Vec<BosonProjectionEntry>,
    pub both_linkages_verified: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FermionSectionEntry {
    pub physical_beta: usize,
    pub full_fermion: usize,
    /// J_F e_beta = sign * e_full_fermion.
    pub sign: i8,
}

#[derive(Debug, Clone, Serialize)]
pub struct BosonProjectionEntry {
    pub full_boson: usize,
    pub physical_vector: usize,
    pub sign: i8,
}

#[derive(Debug, Clone, Serialize)]
pub struct UniformSectionSystem {
    pub full_boson_variables: usize,
    pub equations: usize,
    pub output_components: usize,
    pub fiber_size: usize,
    pub right_inverse_verified: bool,
    /// Each row encodes P_B * input = target after multiplying the uniform
    /// section equations by 8, so every coefficient is an exact integer.
    pub rows: Vec<UniformSectionRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UniformSectionRow {
    pub input: Vec<SparseIntegerEntry>,
    pub target: Vec<i8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SparseIntegerEntry {
    pub index: usize,
    pub coefficient: i8,
}

#[derive(Debug, Clone, Serialize)]
pub struct JointSectionData {
    pub l_perm: Vec<Vec<u16>>,
    pub l_sign: Vec<Vec<i8>>,
    pub gamma_perm: Vec<Vec<u16>>,
    pub gamma_sign: Vec<Vec<i8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HoleAttackReport {
    pub scope: String,
    pub arithmetic: ArithmeticAudit,
    pub sym_seed: SymSeedAudit,
    pub topologies: Vec<TopologyAttack>,
    pub conclusion: String,
}

#[derive(Debug, Deserialize)]
struct Catalog {
    codes: Vec<CodeEntry>,
}

#[derive(Debug, Deserialize)]
struct CodeEntry {
    #[serde(default)]
    index: usize,
    k: usize,
    generators_raw: Vec<u32>,
    #[serde(default)]
    is_indecomposable: bool,
}

pub fn run(catalog_path: &str) -> HoleAttackReport {
    let arithmetic = arithmetic_audit();
    let gamma = so9_gamma();
    let sym_seed = audit_sym_seed(&gamma);

    let text = fs::read_to_string(catalog_path)
        .unwrap_or_else(|e| panic!("failed to read N=16 catalog {catalog_path}: {e}"));
    let catalog: Catalog = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse N=16 catalog {catalog_path}: {e}"));

    let mut topologies = Vec::new();
    for (position, entry) in catalog.codes.iter().enumerate().filter(|(_, e)| e.k == 8) {
        let code = DoublyEvenCode::new(N, entry.generators_raw.clone());
        assert!(code.is_valid(), "catalog code {position} is invalid");
        let chromo = Chromotopology::from_code(&code);
        assert_eq!(chromo.d(), MIN_D);
        let de = DashingEnumerator::new(&code);
        let dashing_count = std::env::var("ADINKRA_SR_DASHINGS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| de.num_classes())
            .min(de.num_classes());
        let perms: Vec<Vec<usize>> = (0..N).map(|i| chromo.color_perm(i).to_vec()).collect();
        let mut all_garden = true;
        let mut max_linear = 0usize;
        let mut nonzero_linear = 0usize;
        let mut best_dashings = Vec::new();
        let mut first_rep = None;
        let mut projection_positive = 0usize;
        let mut projection_signed_tested = 0usize;
        let mut projection_min_aux = None::<usize>;
        let mut projection_witness = None;
        let mut projection_support_possible = None::<bool>;
        for dashing_index in 0..dashing_count {
            let signs = de.get_dashing_for_chromotopology(dashing_index, &chromo.boson_reps());
            let rep = AdinkraRep::from_parts(N, MIN_D, &perms, &signs);
            all_garden &= rep.verify_garden_algebra();
            let linear = mixed_basis_linear_space(&rep, &gamma);
            if linear.free_components > 0 {
                nonzero_linear += 1;
            }
            if linear.free_components > max_linear {
                max_linear = linear.free_components;
                best_dashings.clear();
                best_dashings.push(dashing_index);
            } else if linear.free_components == max_linear {
                best_dashings.push(dashing_index);
            }
            if first_rep.is_none() {
                first_rep = Some(rep.clone());
            }
            if projection_support_possible.is_none() {
                projection_support_possible = Some(
                    auxiliary_projection_witness_mode(&rep, &gamma, dashing_index, false).is_some(),
                );
            }
            if projection_support_possible == Some(true) && projection_witness.is_none() {
                projection_signed_tested += 1;
                if let Some(witness) =
                    auxiliary_projection_witness_with_two_sided(&rep, &gamma, dashing_index)
                {
                    projection_positive += 1;
                    let aux = witness.exercised_auxiliary_fermion_directions;
                    if projection_min_aux.map_or(true, |best| aux < best) {
                        projection_min_aux = Some(aux);
                        projection_witness = Some(witness);
                    }
                }
            }
        }
        let first_rep = first_rep.unwrap();
        if max_linear == 0 {
            best_dashings.clear();
        }
        let max_equal_neighbor_sets = max_equal_neighbor_sets(&first_rep);
        topologies.push(TopologyAttack {
            catalog_index: if entry.index == 0 {
                position
            } else {
                entry.index
            },
            code_dimension: entry.k,
            adinkra_dimension: first_rep.d,
            indecomposable: entry.is_indecomposable,
            dashings_tested: dashing_count,
            all_dashings_garden_verified: all_garden,
            max_equal_neighbor_sets,
            literal_coordinate_embedding_possible: max_equal_neighbor_sets >= VECTOR_BOSONS,
            dashings_with_nonzero_mixed_basis_space: nonzero_linear,
            max_mixed_basis_linear_dimension: max_linear,
            best_dashing_indices: best_dashings,
            auxiliary_projection: AuxiliaryProjectionAttack {
                support_compatible: projection_support_possible.unwrap_or(false),
                support_obstruction_is_dashing_independent: projection_support_possible
                    == Some(false),
                signed_dashings_examined: projection_signed_tested,
                dashings_with_witness: projection_positive,
                minimum_exercised_auxiliary_fermion_directions: projection_min_aux,
                witness: projection_witness,
            },
        });
    }

    assert_eq!(
        topologies.len(),
        2,
        "expected the two self-dual length-16 codes"
    );

    let any_linear = topologies
        .iter()
        .any(|t| t.max_mixed_basis_linear_dimension > 0);
    let any_projection = topologies
        .iter()
        .any(|t| t.auxiliary_projection.dashings_with_witness > 0);
    let any_two_sided = topologies.iter().any(|t| {
        t.auxiliary_projection
            .witness
            .as_ref()
            .and_then(|w| w.two_sided_temporal_seed.as_ref())
            .is_some()
    });
    let conclusion = if any_two_sided {
        "Allowing auxiliary fermion directions produces an exact two-sided temporal SYM seed quotient inside the minimal module. The next gate is nonvalise gauge-covariant 10D oxidation."
    } else if any_projection {
        "A quotient-and-section relaxation produces an exact Q A_i witness with a 112-dimensional kernel. The kernel dimension follows from the rank-16 quotient and is not by itself evidence for seven Lorentz-covariant auxiliary spinors. No coordinate right-inverse for this fixed witness also reproduces Q lambda; this command emits the joint system for a separate non-coordinate search before any 10D oxidation test."
    } else if any_linear {
        "The literal monomial-subblock ansatz is exactly excluded, but at least one dashing leaves a nonzero exact linear intertwiner space after field mixing. The next gate is the quadratic orthonormal-frame problem, not another 145-class scan."
    } else {
        "All dashings of both minimal adinkra topologies fail the arbitrary-field-basis linear intertwiner equations for the fixed SO(9) seed and coordinate supercharge identification. A completion must relax at least one part of that ansatz."
    };

    let full_dashing_coverage = topologies.iter().all(|t| t.dashings_tested == 256);
    let coverage = if full_dashing_coverage {
        "every dashing"
    } else {
        "the explicitly reported dashing budget"
    };
    HoleAttackReport {
        scope: format!(
            "Minimal m=1, d=128 kinematic test over both self-dual chromotopologies and {coverage}, with a fixed SO(9) seed and coordinate identification of the 16 supercharges. The auxiliary-projection gate stops after its first signed witness because existence, not dashing classification, is the question. No action, Lorentz oxidation, or gauge-covariant 10D closure is claimed."
        ),
        arithmetic,
        sym_seed,
        topologies,
        conclusion: conclusion.to_string(),
    }
}

/// Search for an exact quotient realization of the conventional gauge-field
/// transformation after auxiliary fermions are allowed.
///
/// For coordinate bosons b_i, Q_a b_i is one signed full-fermion coordinate.
/// We seek a rank-16 projection P_F such that
///
///   P_F L_a^T b_i = gamma_i[a,beta] e_beta.
///
/// Different full fermion coordinates may project to the same e_beta; their
/// signed differences are precisely the auxiliary directions hidden by the
/// on-shell quotient.
fn auxiliary_projection_witness(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    dashing_index: usize,
) -> Option<AuxiliaryProjectionWitness> {
    auxiliary_projection_witness_mode(rep, gamma, dashing_index, true)
}

fn auxiliary_projection_witness_with_two_sided(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    dashing_index: usize,
) -> Option<AuxiliaryProjectionWitness> {
    let mut witness = auxiliary_projection_witness(rep, gamma, dashing_index)?;
    witness.two_sided_temporal_seed = extend_to_two_sided_seed(rep, gamma, &witness);
    witness.uniform_section_system = uniform_section_system(rep, gamma, &witness);
    witness.joint_section_data = Some(JointSectionData {
        l_perm: rep.l_matrices.iter().map(|l| l.perm.clone()).collect(),
        l_sign: rep.l_matrices.iter().map(|l| l.sign.clone()).collect(),
        gamma_perm: gamma.iter().map(|g| g.perm.clone()).collect(),
        gamma_sign: gamma.iter().map(|g| g.sign.clone()).collect(),
    });
    Some(witness)
}

fn auxiliary_projection_witness_mode(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    dashing_index: usize,
    enforce_signs: bool,
) -> Option<AuxiliaryProjectionWitness> {
    assert_eq!(
        rep.d, 128,
        "u128 CSP bitsets require the minimal d=128 module"
    );
    // required[i][b][f] is 0 if Q on coordinate boson b in vector slot i
    // never reaches full fermion f. Otherwise it is +/- (physical_beta + 1).
    let mut required = vec![vec![vec![0i8; rep.d]; rep.d]; VECTOR_BOSONS];
    for (i, g) in gamma.iter().enumerate() {
        for boson in 0..rep.d {
            for charge in 0..N {
                let full_f = rep.l_matrices[charge].perm[boson] as usize;
                let physical_beta = g.perm[charge] as usize;
                let projection_sign = if enforce_signs {
                    rep.l_matrices[charge].sign[boson] * g.sign[charge]
                } else {
                    1
                };
                let encoded = projection_sign * (physical_beta as i8 + 1);
                assert!(required[i][boson][full_f] == 0 || required[i][boson][full_f] == encoded);
                required[i][boson][full_f] = encoded;
            }
        }
    }

    // Index which bosons assign each full fermion and signed physical value.
    let mut assigned_any = vec![vec![0u128; rep.d]; VECTOR_BOSONS];
    let mut by_value = vec![vec![[0u128; 33]; rep.d]; VECTOR_BOSONS];
    for j in 0..VECTOR_BOSONS {
        for boson in 0..rep.d {
            let bit = 1u128 << boson;
            for charge in 0..N {
                let f = rep.l_matrices[charge].perm[boson] as usize;
                let v = required[j][boson][f];
                debug_assert_ne!(v, 0);
                assigned_any[j][f] |= bit;
                by_value[j][f][(v + 16) as usize] |= bit;
            }
        }
    }

    // compat[i][b][j] is the exact domain bitset of coordinate bosons in slot
    // j compatible with choosing coordinate boson b in slot i.
    let mut compat = vec![vec![vec![u128::MAX; VECTOR_BOSONS]; rep.d]; VECTOR_BOSONS];
    for i in 0..VECTOR_BOSONS {
        for boson in 0..rep.d {
            for j in 0..VECTOR_BOSONS {
                if i == j {
                    continue;
                }
                let mut allowed = u128::MAX;
                for charge in 0..N {
                    let f = rep.l_matrices[charge].perm[boson] as usize;
                    let v = required[i][boson][f];
                    debug_assert_ne!(v, 0);
                    allowed &= !assigned_any[j][f] | by_value[j][f][(v + 16) as usize];
                }
                // Distinct coordinate bosons make the physical injection rank 9.
                allowed &= !(1u128 << boson);
                compat[i][boson][j] = allowed;
            }
        }
    }

    let mut domains = [u128::MAX; VECTOR_BOSONS];
    let mut chosen = vec![usize::MAX; VECTOR_BOSONS];
    if !projection_backtrack(0, &mut domains, &compat, &mut chosen) {
        return None;
    }

    let mut merged = vec![0i8; rep.d];
    for i in 0..VECTOR_BOSONS {
        for f in 0..rep.d {
            let v = required[i][chosen[i]][f];
            if v != 0 {
                assert!(merged[f] == 0 || merged[f] == v);
                merged[f] = v;
            }
        }
    }

    let projection: Vec<ProjectionEntry> = merged
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, encoded)| *encoded != 0)
        .map(|(full_fermion, encoded)| ProjectionEntry {
            full_fermion,
            physical_beta: encoded.unsigned_abs() as usize - 1,
            sign: encoded.signum(),
        })
        .collect();
    let reached = projection.len();
    debug_assert!(reached >= SPINOR_FERMIONS);
    let mut fiber_sizes = vec![0usize; SPINOR_FERMIONS];
    for entry in &projection {
        fiber_sizes[entry.physical_beta] += 1;
    }
    let uniform_fiber_size = fiber_sizes
        .first()
        .copied()
        .filter(|first| fiber_sizes.iter().all(|size| size == first));
    let witness = AuxiliaryProjectionWitness {
        dashing_index,
        boson_by_vector: chosen,
        reached_full_fermions: reached,
        exercised_auxiliary_fermion_directions: reached - SPINOR_FERMIONS,
        physical_fiber_sizes: fiber_sizes,
        uniform_fiber_size,
        projection,
        two_sided_temporal_seed: None,
        uniform_section_system: None,
        joint_section_data: None,
    };
    if enforce_signs {
        debug_assert!(verify_auxiliary_projection_witness(rep, gamma, &witness));
    }
    Some(witness)
}

const UNASSIGNED_PROJECTION: i8 = i8::MIN;

#[derive(Debug, Clone)]
struct SectionCandidate {
    full_fermion: usize,
    section_sign: i8,
    /// UNASSIGNED_PROJECTION means untouched, 0 means explicitly projected to
    /// zero, and +/- (i+1) means a signed physical vector component.
    required_boson_projection: Vec<i8>,
}

fn extend_to_two_sided_seed(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    one_sided: &AuxiliaryProjectionWitness,
) -> Option<TwoSidedSeedWitness> {
    let mut fermion_projection = vec![0i8; rep.d];
    let mut fibers = vec![Vec::<(usize, i8)>::new(); SPINOR_FERMIONS];
    for entry in &one_sided.projection {
        fermion_projection[entry.full_fermion] = entry.sign * (entry.physical_beta as i8 + 1);
        fibers[entry.physical_beta].push((entry.full_fermion, entry.sign));
    }
    if fibers.iter().any(Vec::is_empty) {
        return None;
    }

    let inverses: Vec<SignedPerm> = rep.l_matrices.iter().map(SignedPerm::inverse).collect();
    let mut candidates = vec![Vec::<SectionCandidate>::new(); SPINOR_FERMIONS];
    for beta in 0..SPINOR_FERMIONS {
        for &(full_f, section_sign) in &fibers[beta] {
            // Multiplying by section_sign makes P_F J_F e_beta = e_beta.
            let mut required = vec![UNASSIGNED_PROJECTION; rep.d];
            let mut valid = true;
            for charge in 0..N {
                let full_b = inverses[charge].perm[full_f] as usize;
                let full_coefficient = section_sign * inverses[charge].sign[full_f];
                let desired = gamma.iter().enumerate().find_map(|(i, g)| {
                    (g.perm[charge] as usize == beta).then_some(g.sign[charge] * (i as i8 + 1))
                });
                let encoded = desired.map_or(0, |v| full_coefficient * v);
                if required[full_b] != UNASSIGNED_PROJECTION && required[full_b] != encoded {
                    valid = false;
                    break;
                }
                required[full_b] = encoded;
            }
            if valid {
                candidates[beta].push(SectionCandidate {
                    full_fermion: full_f,
                    section_sign,
                    required_boson_projection: required,
                });
            }
        }
    }

    let mut merged = vec![UNASSIGNED_PROJECTION; rep.d];
    // The selected gauge-field coordinates must project back to their physical
    // SO(9) vector basis.
    for (i, &full_b) in one_sided.boson_by_vector.iter().enumerate() {
        let encoded = i as i8 + 1;
        if merged[full_b] != UNASSIGNED_PROJECTION && merged[full_b] != encoded {
            return None;
        }
        merged[full_b] = encoded;
    }

    let mut chosen = vec![None::<SectionCandidate>; SPINOR_FERMIONS];
    if !section_backtrack(0, &candidates, &mut merged, &mut chosen) {
        return None;
    }

    let fermion_section: Vec<FermionSectionEntry> = chosen
        .iter()
        .enumerate()
        .map(|(physical_beta, candidate)| {
            let candidate = candidate.as_ref().unwrap();
            FermionSectionEntry {
                physical_beta,
                full_fermion: candidate.full_fermion,
                sign: candidate.section_sign,
            }
        })
        .collect();
    let boson_projection: Vec<BosonProjectionEntry> = merged
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, encoded)| *encoded != UNASSIGNED_PROJECTION && *encoded != 0)
        .map(|(full_boson, encoded)| BosonProjectionEntry {
            full_boson,
            physical_vector: encoded.unsigned_abs() as usize - 1,
            sign: encoded.signum(),
        })
        .collect();
    let verified = verify_two_sided_seed(
        rep,
        gamma,
        one_sided,
        &fermion_projection,
        &fermion_section,
        &boson_projection,
    );
    Some(TwoSidedSeedWitness {
        fermion_section,
        boson_projection,
        both_linkages_verified: verified,
    })
}

fn section_backtrack(
    assigned: u32,
    candidates: &[Vec<SectionCandidate>],
    merged: &mut [i8],
    chosen: &mut [Option<SectionCandidate>],
) -> bool {
    if assigned.count_ones() as usize == SPINOR_FERMIONS {
        return true;
    }
    let beta = (0..SPINOR_FERMIONS)
        .filter(|&b| assigned & (1 << b) == 0)
        .min_by_key(|&b| {
            candidates[b]
                .iter()
                .filter(|candidate| projection_requirements_compatible(merged, candidate))
                .count()
        })
        .unwrap();
    for candidate in &candidates[beta] {
        if !projection_requirements_compatible(merged, candidate) {
            continue;
        }
        let mut changed = Vec::new();
        for (idx, &want) in candidate.required_boson_projection.iter().enumerate() {
            if want != UNASSIGNED_PROJECTION && merged[idx] == UNASSIGNED_PROJECTION {
                merged[idx] = want;
                changed.push(idx);
            }
        }
        chosen[beta] = Some(candidate.clone());
        if section_backtrack(assigned | (1 << beta), candidates, merged, chosen) {
            return true;
        }
        chosen[beta] = None;
        for idx in changed {
            merged[idx] = UNASSIGNED_PROJECTION;
        }
    }
    false
}

fn projection_requirements_compatible(merged: &[i8], candidate: &SectionCandidate) -> bool {
    merged
        .iter()
        .zip(&candidate.required_boson_projection)
        .all(|(&have, &want)| {
            want == UNASSIGNED_PROJECTION || have == UNASSIGNED_PROJECTION || have == want
        })
}

fn verify_two_sided_seed(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    one_sided: &AuxiliaryProjectionWitness,
    fermion_projection: &[i8],
    fermion_section: &[FermionSectionEntry],
    boson_projection_entries: &[BosonProjectionEntry],
) -> bool {
    if !verify_auxiliary_projection_witness(rep, gamma, one_sided) {
        return false;
    }
    let mut boson_projection = vec![0i8; rep.d];
    for entry in boson_projection_entries {
        boson_projection[entry.full_boson] = entry.sign * (entry.physical_vector as i8 + 1);
    }
    for (i, &full_b) in one_sided.boson_by_vector.iter().enumerate() {
        if boson_projection[full_b] != i as i8 + 1 {
            return false;
        }
    }
    let inverses: Vec<SignedPerm> = rep.l_matrices.iter().map(SignedPerm::inverse).collect();
    for section in fermion_section {
        let encoded_pf = fermion_projection[section.full_fermion];
        if encoded_pf != section.sign * (section.physical_beta as i8 + 1) {
            return false;
        }
        for charge in 0..N {
            let full_b = inverses[charge].perm[section.full_fermion] as usize;
            let coefficient = section.sign * inverses[charge].sign[section.full_fermion];
            let actual = boson_projection[full_b];
            let desired = gamma.iter().enumerate().find_map(|(i, g)| {
                (g.perm[charge] as usize == section.physical_beta)
                    .then_some(g.sign[charge] * (i as i8 + 1))
            });
            match desired {
                Some(want) => {
                    if actual == 0 || coefficient * actual != want {
                        return false;
                    }
                }
                None => {
                    if actual != 0 {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn uniform_section_system(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    witness: &AuxiliaryProjectionWitness,
) -> Option<UniformSectionSystem> {
    let mut fibers = vec![Vec::<(usize, i8)>::new(); SPINOR_FERMIONS];
    for entry in &witness.projection {
        fibers[entry.physical_beta].push((entry.full_fermion, entry.sign));
    }
    let fiber_size = fibers.first()?.len();
    if fiber_size == 0
        || fiber_size > i8::MAX as usize
        || fibers.iter().any(|fiber| fiber.len() != fiber_size)
    {
        return None;
    }

    // P_F * ((1/s) P_F^T) = I because every physical beta has s signed
    // coordinate preimages and every sign squares to +1.
    let right_inverse_verified = fibers.iter().all(|fiber| {
        fiber
            .iter()
            .map(|(_, sign)| i32::from(*sign) * i32::from(*sign))
            .sum::<i32>()
            == fiber_size as i32
    });
    let scale = fiber_size as i8;
    let mut rows = Vec::with_capacity(VECTOR_BOSONS + N * SPINOR_FERMIONS);

    // P_B J_B = I_9, scaled by the same fiber size as the Q-lambda rows.
    for (i, &full_b) in witness.boson_by_vector.iter().enumerate() {
        let mut target = vec![0i8; VECTOR_BOSONS];
        target[i] = scale;
        rows.push(UniformSectionRow {
            input: vec![SparseIntegerEntry {
                index: full_b,
                coefficient: scale,
            }],
            target,
        });
    }

    let inverses: Vec<SignedPerm> = rep.l_matrices.iter().map(SignedPerm::inverse).collect();
    for charge in 0..N {
        for (beta, fiber) in fibers.iter().enumerate() {
            // 8 * L_a J_F e_beta is an integer combination of full bosons.
            let mut input_dense = vec![0i16; rep.d];
            for &(full_f, projection_sign) in fiber {
                let full_b = inverses[charge].perm[full_f] as usize;
                let l_sign = inverses[charge].sign[full_f];
                input_dense[full_b] += i16::from(projection_sign * l_sign);
            }
            let input = input_dense
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, coefficient)| *coefficient != 0)
                .map(|(index, coefficient)| SparseIntegerEntry {
                    index,
                    coefficient: i8::try_from(coefficient).unwrap(),
                })
                .collect();

            let mut target = vec![0i8; VECTOR_BOSONS];
            for (i, g) in gamma.iter().enumerate() {
                if g.perm[charge] as usize == beta {
                    target[i] = scale * g.sign[charge];
                }
            }
            rows.push(UniformSectionRow { input, target });
        }
    }

    Some(UniformSectionSystem {
        full_boson_variables: rep.d,
        equations: rows.len(),
        output_components: VECTOR_BOSONS,
        fiber_size,
        right_inverse_verified,
        rows,
    })
}

fn projection_backtrack(
    assigned: u16,
    domains: &mut [u128; VECTOR_BOSONS],
    compat: &[Vec<Vec<u128>>],
    chosen: &mut [usize],
) -> bool {
    if assigned.count_ones() as usize == VECTOR_BOSONS {
        return true;
    }

    // Minimum-remaining-values ordering makes both witnesses and exhaustive
    // failures fast without sampling.
    let vector = (0..VECTOR_BOSONS)
        .filter(|&i| assigned & (1 << i) == 0)
        .min_by_key(|&i| domains[i].count_ones())
        .unwrap();
    let mut options = domains[vector];
    while options != 0 {
        let boson = options.trailing_zeros() as usize;
        options &= options - 1;
        let saved = *domains;
        let mut viable = true;
        for j in 0..VECTOR_BOSONS {
            if j != vector && assigned & (1 << j) == 0 {
                domains[j] &= compat[vector][boson][j];
                if domains[j] == 0 {
                    viable = false;
                    break;
                }
            }
        }
        chosen[vector] = boson;
        if viable && projection_backtrack(assigned | (1 << vector), domains, compat, chosen) {
            return true;
        }
        *domains = saved;
    }
    chosen[vector] = usize::MAX;
    false
}

fn verify_auxiliary_projection_witness(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    witness: &AuxiliaryProjectionWitness,
) -> bool {
    if witness.boson_by_vector.len() != VECTOR_BOSONS {
        return false;
    }
    let mut projection = vec![0i8; rep.d];
    for entry in &witness.projection {
        if entry.full_fermion >= rep.d
            || entry.physical_beta >= SPINOR_FERMIONS
            || !matches!(entry.sign, -1 | 1)
        {
            return false;
        }
        let encoded = entry.sign * (entry.physical_beta as i8 + 1);
        if projection[entry.full_fermion] != 0 && projection[entry.full_fermion] != encoded {
            return false;
        }
        projection[entry.full_fermion] = encoded;
    }
    let mut distinct_bosons = witness.boson_by_vector.clone();
    distinct_bosons.sort_unstable();
    distinct_bosons.dedup();
    if distinct_bosons.len() != VECTOR_BOSONS {
        return false;
    }
    for (i, g) in gamma.iter().enumerate() {
        let boson = witness.boson_by_vector[i];
        if boson >= rep.d {
            return false;
        }
        for charge in 0..N {
            let full_f = rep.l_matrices[charge].perm[boson] as usize;
            let encoded = projection[full_f];
            if encoded == 0 {
                return false;
            }
            let projected_beta = encoded.unsigned_abs() as usize - 1;
            let projected_sign = rep.l_matrices[charge].sign[boson] * encoded.signum();
            if projected_beta != g.perm[charge] as usize || projected_sign != g.sign[charge] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
fn auxiliary_projection_from_bosons(
    rep: &AdinkraRep,
    gamma: &[SignedPerm],
    dashing_index: usize,
    boson_by_vector: Vec<usize>,
) -> Option<AuxiliaryProjectionWitness> {
    if boson_by_vector.len() != VECTOR_BOSONS {
        return None;
    }
    let mut merged = vec![0i8; rep.d];
    for (i, g) in gamma.iter().enumerate() {
        let boson = boson_by_vector[i];
        if boson >= rep.d {
            return None;
        }
        for charge in 0..N {
            let full_f = rep.l_matrices[charge].perm[boson] as usize;
            let beta = g.perm[charge] as usize;
            let sign = rep.l_matrices[charge].sign[boson] * g.sign[charge];
            let encoded = sign * (beta as i8 + 1);
            if merged[full_f] != 0 && merged[full_f] != encoded {
                return None;
            }
            merged[full_f] = encoded;
        }
    }
    let projection: Vec<ProjectionEntry> = merged
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, encoded)| *encoded != 0)
        .map(|(full_fermion, encoded)| ProjectionEntry {
            full_fermion,
            physical_beta: encoded.unsigned_abs() as usize - 1,
            sign: encoded.signum(),
        })
        .collect();
    let mut fiber_sizes = vec![0usize; SPINOR_FERMIONS];
    for entry in &projection {
        fiber_sizes[entry.physical_beta] += 1;
    }
    let uniform_fiber_size = fiber_sizes
        .first()
        .copied()
        .filter(|first| fiber_sizes.iter().all(|size| size == first));
    let reached = projection.len();
    if reached < SPINOR_FERMIONS {
        return None;
    }
    let witness = AuxiliaryProjectionWitness {
        dashing_index,
        boson_by_vector,
        reached_full_fermions: reached,
        exercised_auxiliary_fermion_directions: reached - SPINOR_FERMIONS,
        physical_fiber_sizes: fiber_sizes,
        uniform_fiber_size,
        projection,
        two_sided_temporal_seed: None,
        uniform_section_system: None,
        joint_section_data: None,
    };
    verify_auxiliary_projection_witness(rep, gamma, &witness).then_some(witness)
}

fn arithmetic_audit() -> ArithmeticAudit {
    // 16 + 32n = 128m is impossible for every m,n because the right side and
    // the auxiliary contribution are 0 mod 32 while 16 is not.
    let paired_has_solution = SPINOR_FERMIONS % 32 == 0;
    let unpaired_units = (MIN_D - SPINOR_FERMIONS) / SPINOR_FERMIONS;
    ArithmeticAudit {
        physical_fermions: SPINOR_FERMIONS,
        paired_auxiliary_quantum: 32,
        unpaired_auxiliary_quantum: 16,
        minimal_off_shell_fermions: MIN_D,
        paired_equation_has_solution: paired_has_solution,
        unpaired_minimal_spinor_units: unpaired_units,
        unpaired_total_fermions: SPINOR_FERMIONS + SPINOR_FERMIONS * unpaired_units,
    }
}

/// Nine real symmetric 16x16 gamma matrices for Cl(9,0).
///
/// The first seven are built from left multiplication by the seven imaginary
/// octonion units. Two standard block generators complete Cl(9,0).
fn so9_gamma() -> Vec<SignedPerm> {
    let triples = [
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    ];
    let mut product = HashMap::<(usize, usize), (i8, usize)>::new();
    for a in 0..8 {
        product.insert((0, a), (1, a));
        product.insert((a, 0), (1, a));
    }
    for a in 1..8 {
        product.insert((a, a), (-1, 0));
    }
    for (a, b, c) in triples {
        for (x, y, z) in [(a, b, c), (b, c, a), (c, a, b)] {
            product.insert((x, y), (1, z));
            product.insert((y, x), (-1, z));
        }
    }

    let mut gamma = Vec::with_capacity(9);
    for a in 1..8 {
        let mut dense = vec![vec![0i8; 16]; 16];
        for b in 0..8 {
            let (s, c) = product[&(a, b)];
            dense[c][8 + b] = s;
            dense[8 + c][b] = -s;
        }
        gamma.push(dense_to_signed_perm(&dense));
    }

    let mut g8 = vec![vec![0i8; 16]; 16];
    let mut g9 = vec![vec![0i8; 16]; 16];
    for i in 0..8 {
        g8[i][8 + i] = 1;
        g8[8 + i][i] = 1;
        g9[i][i] = 1;
        g9[8 + i][8 + i] = -1;
    }
    gamma.push(dense_to_signed_perm(&g8));
    gamma.push(dense_to_signed_perm(&g9));
    gamma
}

fn dense_to_signed_perm(m: &[Vec<i8>]) -> SignedPerm {
    let d = m.len();
    let mut perm = vec![0u16; d];
    let mut sign = vec![0i8; d];
    for (r, row) in m.iter().enumerate() {
        let nz: Vec<(usize, i8)> = row
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| *v != 0)
            .collect();
        assert_eq!(nz.len(), 1);
        perm[r] = nz[0].0 as u16;
        sign[r] = nz[0].1;
    }
    SignedPerm::from_parts(perm, sign).unwrap()
}

fn dense(p: &SignedPerm) -> Vec<Vec<i32>> {
    let mut out = vec![vec![0i32; p.dim()]; p.dim()];
    for r in 0..p.dim() {
        out[r][p.perm[r] as usize] = p.sign[r] as i32;
    }
    out
}

fn matmul(a: &[Vec<i32>], b: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut out = vec![vec![0i32; b[0].len()]; a.len()];
    for i in 0..a.len() {
        for k in 0..b.len() {
            if a[i][k] == 0 {
                continue;
            }
            for j in 0..b[0].len() {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut out = vec![vec![0i32; a.len()]; a[0].len()];
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[j][i] = v;
        }
    }
    out
}

fn audit_sym_seed(gamma: &[SignedPerm]) -> SymSeedAudit {
    let gd: Vec<Vec<Vec<i32>>> = gamma.iter().map(dense).collect();
    let mut clifford_verified = true;
    for i in 0..9 {
        for j in 0..9 {
            let ij = matmul(&gd[i], &gd[j]);
            let ji = matmul(&gd[j], &gd[i]);
            for r in 0..16 {
                for c in 0..16 {
                    let target = if i == j && r == c { 2 } else { 0 };
                    clifford_verified &= ij[r][c] + ji[r][c] == target;
                }
            }
        }
    }

    let mut l = Vec::with_capacity(16);
    for charge in 0..16 {
        let mut m = vec![vec![0i32; 16]; 9];
        for i in 0..9 {
            m[i].copy_from_slice(&gd[i][charge]);
        }
        l.push(m);
    }
    let r: Vec<Vec<Vec<i32>>> = l.iter().map(|m| transpose(m)).collect();

    let mut bosonic_worst = 0i32;
    let mut fermionic_worst = 0i32;
    let mut diagonal_ranks = Vec::with_capacity(16);
    for a in 0..16 {
        for b in 0..16 {
            let ab_b = matmul(&l[a], &r[b]);
            let ba_b = matmul(&l[b], &r[a]);
            for i in 0..9 {
                for j in 0..9 {
                    let target = if a == b && i == j { 2 } else { 0 };
                    bosonic_worst = bosonic_worst.max((ab_b[i][j] + ba_b[i][j] - target).abs());
                }
            }
            let ab_f = matmul(&r[a], &l[b]);
            let ba_f = matmul(&r[b], &l[a]);
            for i in 0..16 {
                for j in 0..16 {
                    let target = if a == b && i == j { 2 } else { 0 };
                    fermionic_worst = fermionic_worst.max((ab_f[i][j] + ba_f[i][j] - target).abs());
                }
            }
        }
        // L_a has nine orthonormal rows, so I - L_a^T L_a is an exact
        // rank-seven coordinate projector in this monomial SO(9) basis.
        let gram = matmul(&r[a], &l[a]);
        let rank = (0..16).filter(|&i| gram[i][i] == 0).count();
        diagonal_ranks.push(rank);
    }

    SymSeedAudit {
        charges: 16,
        bosons: 9,
        fermions: 16,
        so9_clifford_verified: clifford_verified,
        bosonic_garden_worst_entry: bosonic_worst,
        fermionic_garden_worst_entry: fermionic_worst,
        diagonal_fermionic_remnant_rank: diagonal_ranks,
    }
}

fn max_equal_neighbor_sets(rep: &AdinkraRep) -> usize {
    let mut counts = BTreeMap::<Vec<u16>, usize>::new();
    for b in 0..rep.d {
        let mut neighbors: Vec<u16> = rep.l_matrices.iter().map(|l| l.perm[b]).collect();
        neighbors.sort_unstable();
        *counts.entry(neighbors).or_default() += 1;
    }
    counts.values().copied().max().unwrap_or(0)
}

#[derive(Debug)]
struct LinearSpace {
    free_components: usize,
    #[allow(dead_code)]
    zero_components: usize,
}

fn mixed_basis_linear_space(rep: &AdinkraRep, gamma: &[SignedPerm]) -> LinearSpace {
    let variables = VECTOR_BOSONS * rep.d;
    let mut dsu = SignedDsu::new(variables);
    let inverses: Vec<SignedPerm> = rep.l_matrices.iter().map(SignedPerm::inverse).collect();

    // gamma_i[charge,beta] is the desired coefficient in
    // b_i^T L_charge f_beta. For each beta, all nine expressions
    // sign*L_charge^T*b_i must define the same physical fermion vector.
    for beta in 0..SPINOR_FERMIONS {
        let mut occurrences = Vec::<(usize, usize, i8)>::new();
        for (i, g) in gamma.iter().enumerate() {
            for charge in 0..N {
                if g.perm[charge] as usize == beta {
                    occurrences.push((i, charge, g.sign[charge]));
                }
            }
        }
        assert_eq!(occurrences.len(), VECTOR_BOSONS);
        let (i0, a0, s0) = occurrences[0];
        for &(i1, a1, s1) in &occurrences[1..] {
            for f in 0..rep.d {
                let r0 = inverses[a0].perm[f] as usize;
                let r1 = inverses[a1].perm[f] as usize;
                let c0 = s0 * rep.l_matrices[a0].sign[r0];
                let c1 = s1 * rep.l_matrices[a1].sign[r1];
                dsu.union(i0 * rep.d + r0, i1 * rep.d + r1, c0 * c1);
            }
        }
    }

    dsu.component_counts()
}

/// Signed union-find for homogeneous equalities x = +/- y. A contradictory
/// cycle does not invalidate the whole system; it forces that component to 0.
struct SignedDsu {
    parent: Vec<usize>,
    size: Vec<usize>,
    sign_to_parent: Vec<i8>,
    forced_zero: Vec<bool>,
}

impl SignedDsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            sign_to_parent: vec![1; n],
            forced_zero: vec![false; n],
        }
    }

    /// Returns (root, s) with x = s * root.
    fn find(&mut self, x: usize) -> (usize, i8) {
        if self.parent[x] == x {
            return (x, 1);
        }
        let p = self.parent[x];
        let edge = self.sign_to_parent[x];
        let (root, up) = self.find(p);
        self.parent[x] = root;
        self.sign_to_parent[x] = edge * up;
        (root, self.sign_to_parent[x])
    }

    /// Add x = relation * y.
    fn union(&mut self, x: usize, y: usize, relation: i8) {
        debug_assert!(relation == 1 || relation == -1);
        let (rx, sx) = self.find(x);
        let (ry, sy) = self.find(y);
        if rx == ry {
            if sx != relation * sy {
                self.forced_zero[rx] = true;
            }
            return;
        }

        // If rx is attached below ry, rx = t*ry follows from
        // sx*rx = relation*sy*ry.
        let t = sx * relation * sy;
        if self.size[rx] <= self.size[ry] {
            self.parent[rx] = ry;
            self.sign_to_parent[rx] = t;
            self.size[ry] += self.size[rx];
            self.forced_zero[ry] |= self.forced_zero[rx];
        } else {
            self.parent[ry] = rx;
            self.sign_to_parent[ry] = t;
            self.size[rx] += self.size[ry];
            self.forced_zero[rx] |= self.forced_zero[ry];
        }
    }

    fn component_counts(&mut self) -> LinearSpace {
        for i in 0..self.parent.len() {
            self.find(i);
        }
        let mut free = 0;
        let mut zero = 0;
        for i in 0..self.parent.len() {
            if self.parent[i] == i {
                if self.forced_zero[i] {
                    zero += 1;
                } else {
                    free += 1;
                }
            }
        }
        LinearSpace {
            free_components: free,
            zero_components: zero,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpaired_arithmetic_has_minimal_dimension_128() {
        let a = arithmetic_audit();
        assert!(!a.paired_equation_has_solution);
        assert_eq!(a.unpaired_minimal_spinor_units, 7);
        assert_eq!(a.unpaired_total_fermions, 128);
    }

    #[test]
    fn exact_sym_seed_closes_only_on_bosons() {
        let gamma = so9_gamma();
        let a = audit_sym_seed(&gamma);
        assert!(a.so9_clifford_verified);
        assert_eq!(a.bosonic_garden_worst_entry, 0);
        assert!(a.fermionic_garden_worst_entry > 0);
        assert!(a.diagonal_fermionic_remnant_rank.iter().all(|&r| r == 7));
    }

    #[test]
    fn signed_dsu_contradiction_forces_only_one_component_zero() {
        let mut d = SignedDsu::new(3);
        d.union(0, 1, 1);
        d.union(1, 0, -1);
        let c = d.component_counts();
        assert_eq!(c.zero_components, 1);
        assert_eq!(c.free_components, 1);
    }

    #[test]
    fn every_minimal_dashing_fails_the_fixed_seed_intertwiner() {
        let gamma = so9_gamma();
        let codes = [
            DoublyEvenCode::new(
                16,
                vec![2817, 13376, 21520, 25608, 28704, 33538, 35200, 35332],
            ),
            DoublyEvenCode::new(
                16,
                vec![12353, 12802, 13568, 14464, 28704, 45072, 56900, 61000],
            ),
        ];
        for code in codes {
            let chromo = Chromotopology::from_code(&code);
            let de = DashingEnumerator::new(&code);
            let perms: Vec<Vec<usize>> = (0..16).map(|i| chromo.color_perm(i).to_vec()).collect();
            assert!(
                max_equal_neighbor_sets(&AdinkraRep::from_parts(
                    16,
                    128,
                    &perms,
                    &de.get_dashing_for_chromotopology(0, &chromo.boson_reps()),
                )) < 9
            );
            for class in 0..de.num_classes() {
                let rep = AdinkraRep::from_parts(
                    16,
                    128,
                    &perms,
                    &de.get_dashing_for_chromotopology(class, &chromo.boson_reps()),
                );
                assert!(rep.verify_garden_algebra());
                assert_eq!(
                    mixed_basis_linear_space(&rep, &gamma).free_components,
                    0,
                    "dashing {class} unexpectedly admitted an intertwiner"
                );
            }
        }
    }

    #[test]
    fn e8xe8_dashing_zero_has_rank16_quotient_with_112_kernel() {
        let gamma = so9_gamma();
        let code = DoublyEvenCode::new(
            16,
            vec![2817, 13376, 21520, 25608, 28704, 33538, 35200, 35332],
        );
        let chromo = Chromotopology::from_code(&code);
        let de = DashingEnumerator::new(&code);
        let perms: Vec<Vec<usize>> = (0..16).map(|i| chromo.color_perm(i).to_vec()).collect();
        let rep = AdinkraRep::from_parts(
            16,
            128,
            &perms,
            &de.get_dashing_for_chromotopology(0, &chromo.boson_reps()),
        );
        let witness = auxiliary_projection_from_bosons(
            &rep,
            &gamma,
            0,
            vec![48, 39, 22, 64, 106, 79, 25, 125, 83],
        )
        .unwrap();
        assert!(verify_auxiliary_projection_witness(&rep, &gamma, &witness));
        assert_eq!(witness.reached_full_fermions, 128);
        assert_eq!(witness.exercised_auxiliary_fermion_directions, 112);
        assert_eq!(witness.uniform_fiber_size, Some(8));
        assert!(witness.physical_fiber_sizes.iter().all(|&size| size == 8));
        let uniform = uniform_section_system(&rep, &gamma, &witness).unwrap();
        assert!(uniform.right_inverse_verified);
        assert_eq!(uniform.equations, 265);
        assert_eq!(uniform.full_boson_variables, 128);
        assert_eq!(uniform.output_components, 9);
    }
}
