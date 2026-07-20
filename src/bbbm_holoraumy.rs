#![allow(dead_code)] // research module: exercised by the `bbbm-holoraumy` subcommand and tests

//! Holoraumy / gadget invariants of the minimal N=9 BBBM valise GR(16,9).
//!
//! Route A. Reuse the codebase's tested machinery to characterize the holoraumy
//! structure of the N=9 valise built in `crate::bbbm`:
//!
//!   * bosonic holoraumy    V_IJ    = L_I R_J   = L_I L_J^{-1}
//!   * fermionic holoraumy  Vtilde_IJ = R_I L_J = L_I^{-1} L_J
//!
//! over the C(9,2)=36 color pairs, for each of the 2^k = 16 dashing classes.
//!
//! We report, exactly:
//!   1. Every V_IJ and Vtilde_IJ is traceless (Tr = 0), the antisymmetric
//!      "holoraumy content" (V_IJ = -V_JI, V_IJ^2 = -I).
//!   2. The self-gadget G[R,R] = d/dmin = 16/16 = 1 (irreducible), matching the
//!      structural expectation in `crate::holoraumy`.
//!   3. Dashing dependence: the self-gadget is dashing-INVARIANT (always 1), but
//!      the cross-gadget between distinct dashing classes is dashing-DEPENDENT;
//!      we summarize the set of distinct cross-gadget values.

use crate::bbbm::n9_minimal_code_pub;
use crate::chromotopology::Chromotopology;
use crate::dashing::DashingEnumerator;
use crate::holoraumy::{dmin, gadget, HoloraumyData};
use crate::lr_matrix::AdinkraRep;

/// Build every dashing-class AdinkraRep of the minimal N=9 valise.
pub fn n9_valise_reps() -> Vec<AdinkraRep> {
    let code = n9_minimal_code_pub();
    let n = code.n;
    let chromo = Chromotopology::from_code(&code);
    let d = chromo.d();
    let de = DashingEnumerator::new(&code);
    let color_perms: Vec<Vec<usize>> = (0..n).map(|c| chromo.color_perm(c).to_vec()).collect();
    let boson_reps = chromo.boson_reps();

    (0..de.num_classes())
        .map(|di| {
            let signs = de.get_dashing_for_chromotopology(di, &boson_reps);
            AdinkraRep::from_parts(n, d, &color_perms, &signs)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct BbbmHoloraumyReport {
    pub n: usize,
    pub d: usize,
    pub dmin: usize,
    pub num_pairs: usize,
    pub num_dashings: usize,
    /// Every V_IJ (I>J) is traceless for every dashing class.
    pub all_v_traceless: bool,
    /// Every Vtilde_IJ (I>J) is traceless for every dashing class.
    pub all_vtilde_traceless: bool,
    /// V_IJ^2 = -I for every pair and dashing (bosonic holoraumy squares to -I).
    pub all_v_square_neg_identity: bool,
    /// V_IJ = -V_JI (antisymmetry) for every pair and dashing.
    pub v_antisymmetric: bool,
    /// Self-gadget G[R,R] for every dashing class (expected d/dmin = 1).
    pub self_gadget: f64,
    /// Self-gadget is identical (= d/dmin) across all 16 dashing classes.
    pub self_gadget_dashing_invariant: bool,
    /// Distinct off-diagonal cross-gadget values over all class pairs, sorted.
    pub distinct_cross_gadget_values: Vec<f64>,
    /// True iff at least two distinct cross-gadget values occur => the gadget is
    /// dashing-DEPENDENT across classes.
    pub cross_gadget_dashing_dependent: bool,
}

/// Round to a clean rational-ish key so f64 jitter does not fragment the set.
fn key(x: f64) -> i64 {
    (x * 1e9).round() as i64
}

pub fn compute() -> BbbmHoloraumyReport {
    let reps = n9_valise_reps();
    let n = reps[0].n;
    let d = reps[0].d;
    let num_pairs = n * (n - 1) / 2;

    let mut all_v_traceless = true;
    let mut all_vtilde_traceless = true;
    let mut all_v_square_neg_identity = true;
    let mut v_antisymmetric = true;

    for rep in &reps {
        let holo = HoloraumyData::from_rep(rep);
        for m in &holo.v {
            if m.trace() != 0 {
                all_v_traceless = false;
            }
            if !m.compose(m).is_neg_identity() {
                all_v_square_neg_identity = false;
            }
        }
        for m in &holo.vtilde {
            if m.trace() != 0 {
                all_vtilde_traceless = false;
            }
        }
        // Antisymmetry V_IJ = -V_JI, i.e. V_IJ * V_JI = I with opposite signs.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let vij = rep.v_matrix(i, j);
                let vji = rep.v_matrix(j, i);
                // V_JI = V_IJ^{-1}; antisymmetry means V_IJ^{-1} = -V_IJ.
                if vji != vij.inverse() || vij.inverse() != vij.negate() {
                    v_antisymmetric = false;
                }
            }
        }
    }

    // Self-gadget for each class, and cross-gadget over all class pairs.
    let holos: Vec<HoloraumyData> = reps.iter().map(HoloraumyData::from_rep).collect();
    let self_gadget = gadget(&holos[0], &holos[0]);
    let self_gadget_dashing_invariant = holos
        .iter()
        .all(|h| (gadget(h, h) - self_gadget).abs() < 1e-9);

    use std::collections::BTreeSet;
    let mut cross: BTreeSet<i64> = BTreeSet::new();
    for a in 0..holos.len() {
        for b in (a + 1)..holos.len() {
            cross.insert(key(gadget(&holos[a], &holos[b])));
        }
    }
    let distinct_cross_gadget_values: Vec<f64> =
        cross.iter().map(|&k| k as f64 / 1e9).collect();

    BbbmHoloraumyReport {
        n,
        d,
        dmin: dmin(n),
        num_pairs,
        num_dashings: reps.len(),
        all_v_traceless,
        all_vtilde_traceless,
        all_v_square_neg_identity,
        v_antisymmetric,
        self_gadget,
        self_gadget_dashing_invariant,
        cross_gadget_dashing_dependent: distinct_cross_gadget_values.len() > 1,
        distinct_cross_gadget_values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbbm_n9_holoraumy_invariants() {
        let r = compute();

        // Structural sanity: minimal N=9 valise GR(16,9), 36 pairs, 16 dashings.
        assert_eq!(r.n, 9);
        assert_eq!(r.d, 16);
        assert_eq!(r.dmin, 16, "d_min(9) = 16");
        assert_eq!(r.num_pairs, 36, "C(9,2) = 36 color pairs");
        assert_eq!(r.num_dashings, 16, "2^k = 2^4 = 16 dashing classes");

        // (1) Holoraumy content: traceless + antisymmetric + squares to -I.
        assert!(r.all_v_traceless, "every V_IJ must be traceless");
        assert!(r.all_vtilde_traceless, "every Vtilde_IJ must be traceless");
        assert!(
            r.all_v_square_neg_identity,
            "every V_IJ must square to -I (Garden algebra)"
        );
        assert!(r.v_antisymmetric, "V_IJ = -V_JI antisymmetry");

        // (2) Self-gadget equals d/dmin = 16/16 = 1 (irreducible representation).
        assert!(
            (r.self_gadget - 1.0).abs() < 1e-9,
            "G[R,R] = d/dmin = 1, got {}",
            r.self_gadget
        );

        // (3) Dashing dependence.
        assert!(
            r.self_gadget_dashing_invariant,
            "self-gadget is dashing-invariant (always d/dmin)"
        );
        assert!(
            r.cross_gadget_dashing_dependent,
            "cross-gadget between distinct dashing classes is dashing-dependent"
        );
    }
}
