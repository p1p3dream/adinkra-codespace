#![allow(dead_code)] // primitive-library module: much of its API surface is exercised by the test suite, not the binary main path

/// Signed permutation algebra for Adinkra color-twist representations.
///
/// A signed permutation of dimension d is a pair (perm, sign) encoding a
/// {+1, -1}-monomial matrix: row i has its sole nonzero entry sign[i] at
/// column perm[i].

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignedPerm {
    pub perm: Vec<u16>,
    pub sign: Vec<i8>,
}

impl SignedPerm {
    /// Dimension of the permutation (length of the perm/sign vectors).
    pub fn dim(&self) -> usize {
        self.perm.len()
    }

    /// The d-dimensional identity: trivial permutation, all signs +1.
    pub fn identity(d: usize) -> Self {
        Self {
            perm: (0..d as u16).collect(),
            sign: vec![1i8; d],
        }
    }

    /// True when this element equals the identity permutation.
    pub fn is_identity(&self) -> bool {
        self.perm.iter().enumerate().all(|(i, &p)| p == i as u16) && self.sign.iter().all(|&s| s == 1)
    }

    /// Composition of signed permutations. With the convention that row `i` has
    /// its sole nonzero at column `perm[i]`, this is a RIGHT action: the returned
    /// element's matrix equals the product `other * self` (NOT `self * other`).
    /// Verified by the `compose_manual_3_element` test.
    ///
    /// Result row i: the nonzero entry is at column `self.perm[other.perm[i]]`
    /// with value `self.sign[other.perm[i]] * other.sign[i]`.
    pub fn compose(&self, other: &SignedPerm) -> SignedPerm {
        let d = self.dim();
        let mut perm = vec![0u16; d];
        let mut sign = vec![0i8; d];
        for i in 0..d {
            let j = other.perm[i] as usize;
            perm[i] = self.perm[j];
            sign[i] = self.sign[j] * other.sign[i];
        }
        SignedPerm { perm, sign }
    }

    /// Matrix inverse (equivalently, matrix transpose for orthogonal
    /// signed-permutation matrices).
    ///
    /// If A has its nonzero in row i at column perm[i] with value sign[i],
    /// then A^{-1} has its nonzero in row perm[i] at column i with the same
    /// sign value.
    pub fn inverse(&self) -> SignedPerm {
        let d = self.dim();
        let mut inv_perm = vec![0u16; d];
        let mut inv_sign = vec![0i8; d];
        for i in 0..d {
            let target = self.perm[i] as usize;
            inv_perm[target] = i as u16;
            inv_sign[target] = self.sign[i];
        }
        SignedPerm {
            perm: inv_perm,
            sign: inv_sign,
        }
    }

    /// Trace of the matrix product self * other, computed without forming the
    /// full product.
    ///
    /// `Tr(A * B) = sum over {i : B.perm[A.perm[i]] == i} of A.sign[i] * B.sign[A.perm[i]]`
    ///
    /// Only the fixed points of the composed permutation contribute.
    pub fn trace_product(&self, other: &SignedPerm) -> i64 {
        let d = self.dim();
        let mut total = 0i64;
        for i in 0..d {
            let j = self.perm[i] as usize;
            if other.perm[j] == i as u16 {
                total += (self.sign[i] as i64) * (other.sign[j] as i64);
            }
        }
        total
    }

    /// Trace of this matrix: sum of diagonal entries.
    ///
    /// Only indices where perm[i] == i contribute (fixed points), each
    /// contributing sign[i].
    pub fn trace(&self) -> i64 {
        let mut total = 0i64;
        for i in 0..self.dim() {
            if self.perm[i] == i as u16 {
                total += self.sign[i] as i64;
            }
        }
        total
    }

    /// Negate: flip all signs (equivalent to multiplying the matrix by -1).
    pub fn negate(&self) -> SignedPerm {
        SignedPerm {
            perm: self.perm.clone(),
            sign: self.sign.iter().map(|&s| -s).collect(),
        }
    }

    /// Construct from raw parts with validation.
    ///
    /// Checks that `perm` is a valid bijection on {0..d-1} and all sign
    /// entries are exactly +1 or -1.
    pub fn from_parts(perm: Vec<u16>, sign: Vec<i8>) -> Result<Self, String> {
        if perm.len() != sign.len() {
            return Err(format!(
                "perm length ({}) != sign length ({})",
                perm.len(),
                sign.len()
            ));
        }
        let d = perm.len();

        // Validate signs
        for (i, &s) in sign.iter().enumerate() {
            if s != 1 && s != -1 {
                return Err(format!("sign[{}] = {} (must be +1 or -1)", i, s));
            }
        }

        // Validate perm is a bijection: every value in 0..d must appear exactly once
        let mut seen = vec![false; d];
        for (i, &p) in perm.iter().enumerate() {
            let p = p as usize;
            if p >= d {
                return Err(format!("perm[{}] = {} (out of range 0..{})", i, p, d));
            }
            if seen[p] {
                return Err(format!("perm has duplicate value {}", p));
            }
            seen[p] = true;
        }

        Ok(Self { perm, sign })
    }

    /// Multiplicative order: smallest positive m such that self^m = identity.
    ///
    /// Uses repeated composition. For any finite signed permutation this is
    /// bounded by 2 * lcm(cycle lengths), so it always terminates.
    pub fn order(&self) -> usize {
        let id = SignedPerm::identity(self.dim());
        let mut power = self.clone();
        let mut m = 1usize;
        loop {
            if power == id {
                return m;
            }
            power = self.compose(&power);
            m += 1;
        }
    }

    /// True when this element equals -I (identity permutation, all signs -1).
    pub fn is_neg_identity(&self) -> bool {
        self.perm.iter().enumerate().all(|(i, &p)| p == i as u16) && self.sign.iter().all(|&s| s == -1)
    }

    /// Transpose of the signed-permutation matrix.
    ///
    /// For orthogonal matrices (which all signed-permutation matrices are),
    /// transpose == inverse. This is an alias for `inverse()`.
    pub fn transpose(&self) -> SignedPerm {
        self.inverse()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A specific 3-element signed permutation for manual testing.
    /// perm = [2, 0, 1], sign = [1, -1, 1]
    ///
    /// As a matrix:
    ///   [0  0  1]
    ///   [-1 0  0]
    ///   [0  1  0]
    fn sample_3() -> SignedPerm {
        SignedPerm {
            perm: vec![2, 0, 1],
            sign: vec![1, -1, 1],
        }
    }

    /// Another 3-element signed permutation for associativity/composition tests.
    /// perm = [1, 2, 0], sign = [-1, 1, -1]
    ///
    /// As a matrix:
    ///   [0 -1  0]
    ///   [0  0  1]
    ///   [-1 0  0]
    fn sample_3b() -> SignedPerm {
        SignedPerm {
            perm: vec![1, 2, 0],
            sign: vec![-1, 1, -1],
        }
    }

    /// Third 3-element signed permutation for triple-associativity.
    /// perm = [0, 2, 1], sign = [1, 1, -1]
    fn sample_3c() -> SignedPerm {
        SignedPerm {
            perm: vec![0, 2, 1],
            sign: vec![1, 1, -1],
        }
    }

    // -- identity ----------------------------------------------------------

    #[test]
    fn identity_is_identity() {
        for d in 0..=5 {
            let id = SignedPerm::identity(d);
            assert!(id.is_identity(), "identity({}) should be identity", d);
            assert_eq!(id.dim(), d);
        }
    }

    #[test]
    fn identity_not_neg_identity() {
        let id = SignedPerm::identity(3);
        assert!(!id.is_neg_identity());
    }

    // -- compose with identity ---------------------------------------------

    #[test]
    fn compose_identity_left() {
        let a = sample_3();
        let id = SignedPerm::identity(3);
        assert_eq!(id.compose(&a), a);
    }

    #[test]
    fn compose_identity_right() {
        let a = sample_3();
        let id = SignedPerm::identity(3);
        assert_eq!(a.compose(&id), a);
    }

    // -- inverse -----------------------------------------------------------

    #[test]
    fn compose_with_inverse_gives_identity() {
        let a = sample_3();
        let inv = a.inverse();
        let product = a.compose(&inv);
        assert!(product.is_identity(), "A * A^-1 should be identity");

        let product2 = inv.compose(&a);
        assert!(product2.is_identity(), "A^-1 * A should be identity");
    }

    #[test]
    fn inverse_involution() {
        let a = sample_3();
        assert_eq!(a.inverse().inverse(), a, "(A^-1)^-1 should equal A");
    }

    #[test]
    fn transpose_equals_inverse() {
        let a = sample_3();
        assert_eq!(a.transpose(), a.inverse());
    }

    // -- associativity -----------------------------------------------------

    #[test]
    fn compose_associative() {
        let a = sample_3();
        let b = sample_3b();
        let c = sample_3c();

        let ab_c = a.compose(&b).compose(&c);
        let a_bc = a.compose(&b.compose(&c));
        assert_eq!(ab_c, a_bc, "(A*B)*C should equal A*(B*C)");
    }

    // -- specific 3-element manual computation -----------------------------

    #[test]
    fn compose_manual_3_element() {
        // A = sample_3 = self: perm=[2,0,1], sign=[1,-1,1]
        // B = sample_3b = other: perm=[1,2,0], sign=[-1,1,-1]
        //
        // compose is a RIGHT action: A.compose(B) has matrix B*A. Its parts are
        //   result.perm[i] = A.perm[B.perm[i]]
        //   result.sign[i] = A.sign[B.perm[i]] * B.sign[i]
        //
        // i=0: B.perm[0]=1, A.perm[1]=0, A.sign[1]=-1, B.sign[0]=-1 => perm=0, sign=1
        // i=1: B.perm[1]=2, A.perm[2]=1, A.sign[2]=1,  B.sign[1]=1  => perm=1, sign=1
        // i=2: B.perm[2]=0, A.perm[0]=2, A.sign[0]=1,  B.sign[2]=-1 => perm=2, sign=-1

        let ab = sample_3().compose(&sample_3b());
        assert_eq!(ab.perm, vec![0, 1, 2]);
        assert_eq!(ab.sign, vec![1, 1, -1]);
    }

    // -- trace -------------------------------------------------------------

    #[test]
    fn trace_identity() {
        for d in 1..=5 {
            let id = SignedPerm::identity(d);
            assert_eq!(id.trace(), d as i64, "Tr(I_{}) should be {}", d, d);
        }
    }

    #[test]
    fn trace_negated_identity() {
        for d in 1..=5 {
            let neg = SignedPerm::identity(d).negate();
            assert_eq!(
                neg.trace(),
                -(d as i64),
                "Tr(-I_{}) should be -{}",
                d,
                d
            );
        }
    }

    #[test]
    fn trace_sample_3() {
        // perm=[2,0,1], sign=[1,-1,1]. No fixed points, so trace = 0.
        assert_eq!(sample_3().trace(), 0);
    }

    // -- trace_product -----------------------------------------------------

    #[test]
    fn trace_product_with_identity_equals_trace() {
        let a = sample_3();
        let id = SignedPerm::identity(3);
        assert_eq!(
            a.trace_product(&id),
            a.trace(),
            "Tr(A * I) should equal Tr(A)"
        );
    }

    #[test]
    fn trace_product_manual_2_element() {
        // A: perm=[1,0], sign=[1,-1]
        //   Matrix: [[0, 1], [-1, 0]]
        // B: perm=[0,1], sign=[-1, 1]
        //   Matrix: [[-1, 0], [0, 1]]
        //
        // A*B = [[0, 1], [1, 0]]
        // Tr(A*B) = 0 + 0 = 0
        //
        // Using the formula: Tr(A*B) = sum over {i : B.perm[A.perm[i]] == i} of A.sign[i]*B.sign[A.perm[i]]
        //   i=0: A.perm[0]=1, B.perm[1]=1, 1 != 0 => skip
        //   i=1: A.perm[1]=0, B.perm[0]=0, 0 != 1 => skip
        // Tr = 0

        let a = SignedPerm {
            perm: vec![1, 0],
            sign: vec![1, -1],
        };
        let b = SignedPerm {
            perm: vec![0, 1],
            sign: vec![-1, 1],
        };

        assert_eq!(a.trace_product(&b), 0);

        // Verify against explicit composition trace
        let ab = a.compose(&b);
        assert_eq!(ab.trace(), 0);
    }

    #[test]
    fn trace_product_another_2_element() {
        // A: perm=[0,1], sign=[1,-1]   (diagonal matrix diag(1,-1))
        // B: perm=[0,1], sign=[-1,1]   (diagonal matrix diag(-1,1))
        //
        // A*B = diag(-1, -1)
        // Tr(A*B) = -2
        //
        // Formula: both are identity permutation, so both indices are fixed points.
        //   i=0: A.perm[0]=0, B.perm[0]=0 == 0 => A.sign[0]*B.sign[0] = 1*(-1) = -1
        //   i=1: A.perm[1]=1, B.perm[1]=1 == 1 => A.sign[1]*B.sign[1] = (-1)*1 = -1
        // Tr = -2

        let a = SignedPerm {
            perm: vec![0, 1],
            sign: vec![1, -1],
        };
        let b = SignedPerm {
            perm: vec![0, 1],
            sign: vec![-1, 1],
        };

        assert_eq!(a.trace_product(&b), -2);
        assert_eq!(a.compose(&b).trace(), -2);
    }

    #[test]
    fn trace_product_cyclic_property() {
        // Tr(A*B) == Tr(B*A) for any A, B
        let a = sample_3();
        let b = sample_3b();
        assert_eq!(
            a.trace_product(&b),
            b.trace_product(&a),
            "Tr(A*B) should equal Tr(B*A)"
        );
    }

    #[test]
    fn trace_product_matches_explicit_composition() {
        let a = sample_3();
        let b = sample_3b();
        assert_eq!(
            a.trace_product(&b),
            a.compose(&b).trace(),
            "trace_product should match compose-then-trace"
        );
    }

    // -- negate ------------------------------------------------------------

    #[test]
    fn negate_twice_is_original() {
        let a = sample_3();
        assert_eq!(a.negate().negate(), a);
    }

    #[test]
    fn negate_identity_is_neg_identity() {
        let neg = SignedPerm::identity(4).negate();
        assert!(neg.is_neg_identity());
    }

    // -- order -------------------------------------------------------------

    #[test]
    fn order_of_identity() {
        assert_eq!(SignedPerm::identity(3).order(), 1);
    }

    #[test]
    fn order_of_negated_identity() {
        assert_eq!(SignedPerm::identity(3).negate().order(), 2);
    }

    #[test]
    fn order_of_sample_3() {
        // perm = [2,0,1] is a 3-cycle. sign = [1,-1,1].
        // Applying 3 times cycles perm back to identity, but signs may
        // accumulate. The order divides lcm(3, sign-order) which is at most 6.
        let a = sample_3();
        let m = a.order();
        assert!(m > 0);

        // Verify: a^m == identity
        let mut power = SignedPerm::identity(3);
        for _ in 0..m {
            power = a.compose(&power);
        }
        assert!(power.is_identity(), "a^order should be identity");
    }

    // -- is_neg_identity ---------------------------------------------------

    #[test]
    fn neg_identity_recognized() {
        let neg = SignedPerm::identity(4).negate();
        assert!(neg.is_neg_identity());
        assert!(!neg.is_identity());
    }

    #[test]
    fn identity_is_not_neg_identity() {
        assert!(!SignedPerm::identity(4).is_neg_identity());
    }

    #[test]
    fn non_trivial_not_neg_identity() {
        assert!(!sample_3().is_neg_identity());
    }

    // -- from_parts validation ---------------------------------------------

    #[test]
    fn from_parts_valid() {
        let sp = SignedPerm::from_parts(vec![1, 0, 2], vec![1, -1, 1]);
        assert!(sp.is_ok());
    }

    #[test]
    fn from_parts_rejects_non_bijection() {
        let sp = SignedPerm::from_parts(vec![0, 0, 2], vec![1, 1, 1]);
        assert!(sp.is_err(), "duplicate perm value should be rejected");
    }

    #[test]
    fn from_parts_rejects_out_of_range() {
        let sp = SignedPerm::from_parts(vec![0, 3, 1], vec![1, 1, 1]);
        assert!(sp.is_err(), "perm value out of range should be rejected");
    }

    #[test]
    fn from_parts_rejects_invalid_sign() {
        let sp = SignedPerm::from_parts(vec![0, 1, 2], vec![1, 0, 1]);
        assert!(sp.is_err(), "sign value 0 should be rejected");
    }

    #[test]
    fn from_parts_rejects_sign_value_2() {
        let sp = SignedPerm::from_parts(vec![0, 1, 2], vec![1, 2, -1]);
        assert!(sp.is_err(), "sign value 2 should be rejected");
    }

    #[test]
    fn from_parts_rejects_mismatched_lengths() {
        let sp = SignedPerm::from_parts(vec![0, 1], vec![1, -1, 1]);
        assert!(sp.is_err(), "mismatched lengths should be rejected");
    }

    // -- dim-0 edge case ---------------------------------------------------

    #[test]
    fn identity_dim_0() {
        let id = SignedPerm::identity(0);
        assert_eq!(id.dim(), 0);
        assert!(id.is_identity());
        assert_eq!(id.trace(), 0);
        assert_eq!(id.order(), 1);
    }
}
