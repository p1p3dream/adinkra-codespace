#![allow(dead_code)] // research module: exercised by the `bbbm` subcommand and tests

//! BBBM 9-of-16 partial off-shell: the 1D N=9 valise target (GR(16,9)).
//!
//! Baulieu-Berkovits-Bossard-Martin (arXiv:0705.2002, "Ten-dimensional
//! super-Yang-Mills with nine off-shell supersymmetries") close 9 of the 16
//! supersymmetries of 10D, N=1 super-Yang-Mills off-shell by adding 7
//! auxiliary scalars. Reduced to the worldline, the off-shell field content is
//!
//!     16 bosons  = 9 gauge components + 7 auxiliary scalars
//!     16 fermions (gaugino)
//!
//! under N = 9 supersymmetries. That is exactly the minimal N=9 valise,
//! GR(16,9), since d_min(9) = 16.
//!
//! WHAT THIS MODULE DOES. It builds the minimal N=9 valise from its maximal
//! doubly-even code (the [8,4] extended Hamming code padded with a trivial
//! ninth coordinate; k = 4, d = 2^(9-1-4) = 16) through the codebase's tested
//! `Chromotopology` -> `DashingEnumerator` -> `AdinkraRep` machinery, and
//! verifies the Garden algebra
//!
//!     L_I R_J + L_J R_I = 2 delta_IJ I_16
//!
//! exactly, for every dashing class. The minimal N=9 valise is unique up to
//! adinkra equivalence, so BBBM's off-shell content must realize THIS object;
//! establishing that it exists and closes is the first, byte-reproducible step.
//!
//! WHAT IT DOES NOT DO (honest scope). It does not reduce the specific BBBM
//! SUSY transformation rules of arXiv:0705.2002 (that needs the paper's
//! explicit variations), and it does not yet compute the non-closure functions
//! of the remaining 7 supercharges -- the off-shell-sector "equation of
//! motion" -- which is the intended follow-on computation.

use serde::Serialize;

use crate::chromotopology::Chromotopology;
use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;
use crate::lr_matrix::AdinkraRep;

/// The maximal doubly-even code of length 9: the [8,4] extended Hamming code
/// (the same generators used as the codebase's `hamming_8_4` fixture) sitting
/// in the first 8 of 9 coordinates. Bit 8 is unset in every generator, so the
/// ninth coordinate is a free (trivial) color. All codeword weights stay
/// divisible by 4, so the code remains doubly-even; k = 4, hence d = 16.
fn n9_minimal_code() -> DoublyEvenCode {
    DoublyEvenCode::new(9, vec![0b1110_0001, 0b1101_0010, 0b1011_0100, 0b0111_1000])
}

#[derive(Debug, Serialize)]
pub struct BbbmValiseReport {
    /// Number of supersymmetries closing off-shell in the BBBM sector.
    pub supercharges_n: usize,
    /// Valise module dimension (bosons = fermions = d).
    pub module_dimension_d: usize,
    pub bosons: usize,
    pub fermions: usize,
    pub gauge_bosons: usize,
    pub auxiliary_bosons: usize,
    pub code_length_n: usize,
    pub code_dimension_k: usize,
    pub dashing_classes_checked: usize,
    /// True iff L_I R_J + L_J R_I = 2 delta_IJ I_16 holds exactly for every
    /// dashing class of the code.
    pub garden_algebra_verified_all_dashings: bool,
    pub is_minimal_n9_valise: bool,
    pub interpretation: String,
    pub scope_caveat: String,
}

/// Build the minimal N=9 valise and verify the Garden algebra closes exactly
/// on every dashing class. Binary, byte-reproducible pass/fail.
pub fn run() -> BbbmValiseReport {
    let code = n9_minimal_code();
    let n = code.n;
    let k = code.k();
    let chromo = Chromotopology::from_code(&code);
    let d = chromo.d();

    let de = DashingEnumerator::new(&code);
    let color_perms: Vec<Vec<usize>> = (0..n).map(|c| chromo.color_perm(c).to_vec()).collect();
    let boson_reps = chromo.boson_reps();

    let mut all_ok = true;
    let num_classes = de.num_classes();
    for di in 0..num_classes {
        let signs = de.get_dashing_for_chromotopology(di, &boson_reps);
        let rep = AdinkraRep::from_parts(n, d, &color_perms, &signs);
        if !rep.verify_garden_algebra() {
            all_ok = false;
        }
    }

    BbbmValiseReport {
        supercharges_n: n,
        module_dimension_d: d,
        bosons: d,
        fermions: d,
        gauge_bosons: 9,
        auxiliary_bosons: 7,
        code_length_n: n,
        code_dimension_k: k,
        dashing_classes_checked: num_classes,
        garden_algebra_verified_all_dashings: all_ok,
        is_minimal_n9_valise: n == 9 && d == 16,
        interpretation:
            "Minimal N=9 valise GR(16,9): the 1D worldline shadow of the BBBM off-shell content \
             (9 gauge + 7 auxiliary bosons | 16 fermions). Garden algebra \
             L_I R_J + L_J R_I = 2 delta_IJ I_16 verified exactly for every dashing class."
                .to_string(),
        scope_caveat:
            "Builds the canonical minimal N=9 valise (unique up to adinkra equivalence) that \
             BBBM's off-shell content must realize. Does NOT reduce the specific BBBM \
             transformation rules (arXiv:0705.2002), nor compute the non-closure of the \
             remaining 7 supercharges (the off-shell-sector equation of motion)."
                .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbbm_n9_valise_closes_garden_algebra() {
        let r = run();
        assert_eq!(r.supercharges_n, 9, "BBBM off-shell sector has 9 supercharges");
        assert_eq!(r.module_dimension_d, 16, "d_min(9) = 16");
        assert_eq!(
            r.gauge_bosons + r.auxiliary_bosons,
            r.bosons,
            "16 bosons = 9 gauge + 7 auxiliary"
        );
        assert_eq!(r.code_dimension_k, 4, "[9,4] maximal doubly-even code");
        assert!(r.dashing_classes_checked >= 1, "at least one dashing class");
        assert!(
            r.garden_algebra_verified_all_dashings,
            "GR(16,9) Garden algebra must close exactly for every dashing class"
        );
        assert!(r.is_minimal_n9_valise);
    }
}
