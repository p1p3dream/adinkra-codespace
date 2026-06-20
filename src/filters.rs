use std::collections::HashSet;

use crate::code::DoublyEvenCode;

#[derive(Debug, Clone)]
pub struct WorldsheetResult {
    pub p: usize,
    pub q: usize,
    pub passes: bool,
    pub left_colors: Vec<usize>,
}

/// Weight-2 obstruction check for a (p,q) chirality split.
///
/// Tests whether e_I ^ e_J (weight 2) is a codeword for every left color I
/// and right color J. This is a necessary condition for the full Gates-Hubsch
/// worldsheet extension, but not sufficient on its own. For doubly-even codes
/// (minimum weight 4), weight-2 vectors are never codewords, so nontrivial
/// splits always fail this check.
pub fn worldsheet_weight2_obstruction(code: &DoublyEvenCode, left_colors: &[usize]) -> bool {
    let codeword_set: HashSet<u32> = code.codewords().into_iter().collect();
    let n = code.n;
    let right_colors: Vec<usize> = (0..n).filter(|c| !left_colors.contains(c)).collect();

    for &i in left_colors {
        for &j in &right_colors {
            let test = (1u32 << i) ^ (1u32 << j);
            if !codeword_set.contains(&test) {
                return false;
            }
        }
    }
    true
}

/// Run all (p, N-p) splits for p = 0..=N using the first p colors as left.
pub fn worldsheet_all_splits(code: &DoublyEvenCode) -> Vec<WorldsheetResult> {
    let n = code.n;
    let mut results = Vec::new();

    for p in 0..=n {
        let left: Vec<usize> = (0..p).collect();
        let passes = worldsheet_weight2_obstruction(code, &left);
        results.push(WorldsheetResult {
            p,
            q: n - p,
            passes,
            left_colors: left,
        });
    }

    results
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DoublyEvenCode;

    #[test]
    fn trivial_splits_pass() {
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        // (0,4): no left colors, no cross-pairs to check
        assert!(worldsheet_weight2_obstruction(&code, &[]));
        // (4,0): no right colors, no cross-pairs to check
        assert!(worldsheet_weight2_obstruction(&code, &[0, 1, 2, 3]));
    }

    #[test]
    fn nontrivial_splits_fail_n4() {
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        assert!(!worldsheet_weight2_obstruction(&code, &[0]));
        assert!(!worldsheet_weight2_obstruction(&code, &[0, 1]));
        assert!(!worldsheet_weight2_obstruction(&code, &[0, 1, 2]));
    }

    #[test]
    fn weight2_never_doubly_even() {
        // For any doubly-even code, e_I ^ e_J has weight 2, never a codeword
        let code = DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        );
        let codewords: HashSet<u32> = code.codewords().into_iter().collect();
        for i in 0..8 {
            for j in (i + 1)..8 {
                let test = (1u32 << i) ^ (1u32 << j);
                assert!(!codewords.contains(&test));
            }
        }
    }

    #[test]
    fn all_splits_doubly_even_n4() {
        let code = DoublyEvenCode::new(4, vec![0b1111]);
        let results = worldsheet_all_splits(&code);
        assert_eq!(results.len(), 5); // p = 0,1,2,3,4
        assert!(results[0].passes); // (0,4)
        assert!(!results[1].passes); // (1,3)
        assert!(!results[2].passes); // (2,2)
        assert!(!results[3].passes); // (3,1)
        assert!(results[4].passes); // (4,0)
    }

    #[test]
    fn all_splits_hamming_8_4() {
        let code = DoublyEvenCode::new(
            8,
            vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
        );
        let results = worldsheet_all_splits(&code);
        for r in &results {
            if r.p == 0 || r.q == 0 {
                assert!(r.passes, "({},{}) should pass", r.p, r.q);
            } else {
                assert!(!r.passes, "({},{}) should fail for doubly-even", r.p, r.q);
            }
        }
    }
}
