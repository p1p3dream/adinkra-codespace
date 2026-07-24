#![allow(dead_code)]
//! Exact signed-permutation p-th-root solver, for the HOPPER operators.
//!
//! The four-color hopper operators X, Y, Z, W (Tables 7-13) are signed
//! permutations (elements of the hyperoctahedral group B_d), so their p-th
//! roots can be found structurally from the signed-cycle decomposition,
//! working on cycle structure instead of scanning the 3^(d^2) matrix grid.
//! (Finding ALL roots is output-sensitive and can be exponential when the
//! root set is large; this is not polynomial in general.) This is a DIFFERENT
//! object from Gates and Lee's G-matrix
//! (arXiv:2408.09342, Eq 8.2/8.3): that G-matrix is a general {-1,0,1} matrix
//! with several nonzeros per row, NOT a signed permutation, and it is solved
//! separately in `gmatrix.rs`. This module does not produce the G-matrix; the
//! paper's brute-force `Select[Tuples[{-1,0,1},{n,n}], ...]` search is what
//! `gmatrix.rs` replaces, not this solver.
//!
//! Convention. Every product here goes through `super::matmul` / `super::pow`,
//! so "p-th root of A" means a signed permutation G with matrix(G)^p = matrix(A),
//! i.e. `super::pow(&G, p) == *a`. Under this project's `compose` convention the
//! underlying permutation composes as ordinary function composition of `perm`
//! (`super::matmul(a,a).perm[i] == a.perm[a.perm[i]]`) and the p-th power's sign at
//! index i is the product of `sign` along the forward orbit i, perm[i], ...,
//! perm^{p-1}[i]:
//!     A.perm[i] = pi_G^p(i),   A.sign[i] = prod_{t=0}^{p-1} g[ pi_G^t(i) ].
//! The structural routines work on that data directly and are proved correct
//! against `brute_pth_roots` for every element of dimension d <= 4 and every p in
//! 1..=6.
//!
//! Two-stage structure of a root G.
//!   1. PERMUTATION LAYER. pi_G must be a p-th root of pi_A in the plain symmetric
//!      group. Raising a length-M cycle to the p-th power yields gcd(M,p) cycles of
//!      length M/gcd(M,p). So the length-L cycles of pi_A are bundled into groups;
//!      a group of `b` equal-length orbits merges into one root-cycle of length
//!      M = b*L, legal iff gcd(M, p) == b. We enumerate every set partition of each
//!      equal-length class and every inequivalent interleaving of a group's orbits.
//!   2. SIGN LAYER. Given a root-cycle's index layout (length M), the signs
//!      g_0..g_{M-1} are unknown in {+1,-1}. The induced target signs are
//!      A.sign[o_k] = prod_{t=0}^{p-1} g_{k+t mod M}, a small linear system over
//!      GF(2). We enumerate the 2^M sign vectors of that ONE cycle and keep those
//!      reproducing A's signs on it, then take the Cartesian product across cycles.
//!      M is bounded by d, so each cycle's sign search is at most 2^M, far smaller
//!      than the 3^(d^2) whole-matrix grid. The total is output-sensitive (the root
//!      set can be exponential), so this is not polynomial in general; it never
//!      touches the 3^(d^2) grid.

use crate::signed_perm::SignedPerm;
use std::collections::BTreeSet;

/// One orbit of a signed permutation: the ordered orbit indices and the cycle sign
/// (product of the per-index signs around the orbit). Emitted starting at the
/// smallest index for canonical order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Orbit {
    indices: Vec<usize>,
    cycle_sign: i8,
}

impl Orbit {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Decompose a signed permutation into its orbits.
fn orbits(a: &SignedPerm) -> Vec<Orbit> {
    let d = a.dim();
    let mut seen = vec![false; d];
    let mut out = Vec::new();
    for start in 0..d {
        if seen[start] {
            continue;
        }
        let mut indices = Vec::new();
        let mut sign_prod: i8 = 1;
        let mut cur = start;
        loop {
            seen[cur] = true;
            indices.push(cur);
            sign_prod *= a.sign[cur];
            cur = a.perm[cur] as usize;
            if cur == start {
                break;
            }
        }
        out.push(Orbit {
            indices,
            cycle_sign: sign_prod,
        });
    }
    out
}

/// All signed permutations G with `super::pow(&G, p) == *a`, deduplicated and in a
/// deterministic sorted order (by (perm, sign)).
pub fn pth_roots(a: &SignedPerm, p: u32) -> Vec<SignedPerm> {
    assert!(p >= 1, "p must be >= 1");
    let d = a.dim();
    if d == 0 {
        return vec![SignedPerm::identity(0)];
    }

    let orbs = orbits(a);

    // Group orbit indices by their length. (Sign does NOT affect the permutation
    // layer; it is resolved per root-cycle in the sign layer.)
    let mut by_len: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, o) in orbs.iter().enumerate() {
        by_len.entry(o.len()).or_default().push(i);
    }

    // For each length class, enumerate all valid sets of root-cycle LAYOUTS (index
    // sequences). Each option is a Vec<Vec<usize>>: the root-cycles for that class.
    let mut per_class_layouts: Vec<Vec<Vec<Vec<usize>>>> = Vec::new();
    for (&l, members) in &by_len {
        let opts = class_layout_options(&orbs, members, l, p);
        if opts.is_empty() {
            return Vec::new(); // some class has no permutation-layer root
        }
        per_class_layouts.push(opts);
    }

    // Cartesian product across classes -> full permutation layouts (list of all
    // root-cycles). Then solve signs per layout.
    let mut layouts: Vec<Vec<Vec<usize>>> = vec![Vec::new()];
    for opts in &per_class_layouts {
        let mut next = Vec::new();
        for acc in &layouts {
            for opt in opts {
                let mut merged = acc.clone();
                merged.extend(opt.iter().cloned());
                next.push(merged);
            }
        }
        layouts = next;
    }

    let mut roots: BTreeSet<(Vec<u16>, Vec<i8>)> = BTreeSet::new();
    for layout in &layouts {
        // Sign layer: for each root-cycle independently, find all sign vectors that
        // reproduce A's signs on that cycle's indices; Cartesian product across
        // cycles, assembling the full G each time.
        for g in solve_signs(a, d, layout, p) {
            debug_assert_eq!(super::pow(&g, p), *a, "assembled a non-root");
            roots.insert((g.perm.clone(), g.sign.clone()));
        }
    }

    roots
        .into_iter()
        .map(|(perm, sign)| SignedPerm::from_parts(perm, sign).unwrap())
        .collect()
}

/// Enumerate every valid set of root-cycle index layouts for one length-L class.
///
/// `members` index into `orbs`. A root bundles these orbits into groups; a group of
/// `b` orbits becomes one root-cycle of length b*L, legal iff gcd(b*L, p) == b. For
/// each legal partition we enumerate the inequivalent interleavings of each group.
fn class_layout_options(
    orbs: &[Orbit],
    members: &[usize],
    l: usize,
    p: u32,
) -> Vec<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    for partition in set_partitions(members.len()) {
        let mut group_layouts: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut ok = true;
        for group in &partition {
            let b = group.len();
            let m = b * l;
            if gcd(m as u32, p) != b as u32 {
                ok = false;
                break;
            }
            let orbit_ids: Vec<usize> = group.iter().map(|&pos| members[pos]).collect();
            let weaves = interleave_orbits(orbs, &orbit_ids, p);
            group_layouts.push(weaves);
        }
        if !ok {
            continue;
        }
        // Cartesian product over groups: each group contributes one root-cycle.
        let mut acc: Vec<Vec<Vec<usize>>> = vec![Vec::new()];
        for gl in &group_layouts {
            let mut next = Vec::new();
            for a in &acc {
                for one in gl {
                    let mut v = a.clone();
                    v.push(one.clone());
                    next.push(v);
                }
            }
            acc = next;
        }
        out.extend(acc);
    }
    out
}

/// Given a full permutation layout (all root-cycles as index sequences), find every
/// sign assignment making `super::pow(&G, p) == *a`.
///
/// Each root-cycle is independent. For a cycle with index sequence o_0..o_{M-1}
/// (pi_G(o_k) = o_{(k+1) mod M}), a sign vector g_0..g_{M-1} induces
/// A.sign[o_k] = prod_{t=0}^{p-1} g_{(k+t) mod M}. We enumerate all 2^M vectors and
/// keep those matching a.sign on this cycle's indices. The full solution set is the
/// Cartesian product over cycles.
fn solve_signs(a: &SignedPerm, d: usize, layout: &[Vec<usize>], p: u32) -> Vec<SignedPerm> {
    // Per-cycle: list of (index -> sign) partial assignments that are valid.
    let mut per_cycle: Vec<Vec<Vec<(usize, i8)>>> = Vec::new();
    for cyc in layout {
        let m = cyc.len();
        let target: Vec<i8> = cyc.iter().map(|&idx| a.sign[idx]).collect();
        let mut valid = Vec::new();
        for bits in 0u32..(1u32 << m) {
            let g: Vec<i8> = (0..m)
                .map(|k| if (bits >> k) & 1 == 1 { -1 } else { 1 })
                .collect();
            // Check induced signs.
            let mut good = true;
            for k in 0..m {
                let mut prod = 1i8;
                for t in 0..(p as usize) {
                    prod *= g[(k + t) % m];
                }
                if prod != target[k] {
                    good = false;
                    break;
                }
            }
            if good {
                valid.push(cyc.iter().cloned().zip(g).collect::<Vec<_>>());
            }
        }
        if valid.is_empty() {
            return Vec::new();
        }
        per_cycle.push(valid);
    }

    // Cartesian product over cycles, building the full signed permutation each time.
    let mut results = Vec::new();
    let mut acc: Vec<Vec<(usize, i8)>> = vec![Vec::new()];
    for choices in &per_cycle {
        let mut next = Vec::new();
        for a_partial in &acc {
            for one in choices {
                let mut v = a_partial.clone();
                v.extend(one.iter().cloned());
                next.push(v);
            }
        }
        acc = next;
    }
    for full in acc {
        let mut perm = vec![0u16; d];
        let mut sign = vec![1i8; d];
        // Permutation from layout.
        for cyc in layout {
            let m = cyc.len();
            for k in 0..m {
                perm[cyc[k]] = cyc[(k + 1) % m] as u16;
            }
        }
        for (idx, s) in full {
            sign[idx] = s;
        }
        results.push(SignedPerm::from_parts(perm, sign).unwrap());
    }
    results
}

/// Every interleaving of `b` disjoint orbits (each length L) into a single
/// length-(b*L) root-cycle G whose p-th power recovers the b original orbits.
///
/// The root-cycle is an index sequence w_0..w_{M-1} with pi_G(w_k)=w_{(k+1) mod M}.
/// Raising to the p-th power strides by p, so pi_G^p maps root-position k to k+p.
/// With gcd(M,p)=b the +p action on {0..M-1} splits into exactly b sub-cycles of
/// length L (computed here). Each sub-cycle lists root-positions in the pi_A
/// successor order, so we drop one target orbit onto each sub-cycle: assign the b
/// orbits to the b sub-cycles (every bijection) and choose each orbit's starting
/// rotation (L choices), writing the orbit's indices onto the sub-cycle's positions
/// in order. Then pi_A(orbit[j]) = orbit[j+1] equals moving +p in root-positions,
/// exactly as required. We over-generate (rotations/anchoring overlap); the final
/// BTreeSet in `pth_roots` deduplicates, so the result set is exact.
fn interleave_orbits(orbs: &[Orbit], orbit_ids: &[usize], p: u32) -> Vec<Vec<usize>> {
    let b = orbit_ids.len();
    let l = orbs[orbit_ids[0]].len();
    let m = b * l;
    let p_stride = (p as usize) % m; // pi_G^p strides by p on root-positions

    // Sub-cycles of the +p_stride action on {0..M-1}.
    let mut seen = vec![false; m];
    let mut subcycles: Vec<Vec<usize>> = Vec::new();
    for s in 0..m {
        if seen[s] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut cur = s;
        while !seen[cur] {
            seen[cur] = true;
            cyc.push(cur);
            cur = (cur + p_stride) % m;
        }
        subcycles.push(cyc);
    }
    debug_assert_eq!(subcycles.len(), b);

    let rotations: Vec<Vec<Vec<usize>>> = orbit_ids
        .iter()
        .map(|&id| all_rotations(&orbs[id].indices))
        .collect();

    let mut out = Vec::new();
    let ids: Vec<usize> = (0..b).collect();
    for assignment in permutations(&ids) {
        // assignment[c] = which orbit (position into orbit_ids) sits on sub-cycle c.
        let rot_ranges: Vec<usize> = std::iter::repeat(l).take(b).collect();
        for rot in mixed_radix(&rot_ranges) {
            let mut weave = vec![usize::MAX; m];
            for c in 0..b {
                let orbit_seq = &rotations[assignment[c]][rot[c]];
                for (j, &pos) in subcycles[c].iter().enumerate() {
                    weave[pos] = orbit_seq[j];
                }
            }
            out.push(weave);
        }
    }
    out
}

/// All L rotations of an orbit (rotation k starts at position k).
fn all_rotations(orbit: &[usize]) -> Vec<Vec<usize>> {
    let l = orbit.len();
    (0..l)
        .map(|k| (0..l).map(|j| orbit[(k + j) % l]).collect())
        .collect()
}

/// Mixed-radix enumeration: all tuples with 0 <= t_i < radix[i].
fn mixed_radix(radix: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &r in radix {
        let mut next = Vec::new();
        for prefix in &out {
            for v in 0..r {
                let mut p = prefix.clone();
                p.push(v);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

/// All permutations of a slice (by value).
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            let mut v = vec![head];
            v.append(&mut tail);
            out.push(v);
        }
    }
    out
}

/// All set partitions of {0..n-1}, each a Vec of groups (Vec of members).
fn set_partitions(n: usize) -> Vec<Vec<Vec<usize>>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut parts: Vec<Vec<Vec<usize>>> = vec![vec![vec![0]]];
    for elem in 1..n {
        let mut next = Vec::new();
        for part in &parts {
            for g in 0..part.len() {
                let mut np = part.clone();
                np[g].push(elem);
                next.push(np);
            }
            let mut np = part.clone();
            np.push(vec![elem]);
            next.push(np);
        }
        parts = next;
    }
    parts
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Reference oracle. Enumerate ALL 2^d * d! signed permutations of dimension d and
/// keep those whose p-th power equals `a`, in the same sorted order as
/// `pth_roots`. Tractable only for small d; exists to PROVE `pth_roots` correct.
pub fn brute_pth_roots(a: &SignedPerm, p: u32) -> Vec<SignedPerm> {
    assert!(p >= 1, "p must be >= 1");
    let d = a.dim();
    let mut roots: BTreeSet<(Vec<u16>, Vec<i8>)> = BTreeSet::new();
    for perm in permutations(&(0..d).collect::<Vec<_>>()) {
        let perm16: Vec<u16> = perm.iter().map(|&x| x as u16).collect();
        for sign_bits in 0u32..(1u32 << d) {
            let sign: Vec<i8> = (0..d)
                .map(|i| if (sign_bits >> i) & 1 == 1 { -1 } else { 1 })
                .collect();
            let g = SignedPerm::from_parts(perm16.clone(), sign).unwrap();
            if super::pow(&g, p) == *a {
                roots.insert((g.perm.clone(), g.sign.clone()));
            }
        }
    }
    roots
        .into_iter()
        .map(|(perm, sign)| SignedPerm::from_parts(perm, sign).unwrap())
        .collect()
}

/// Status report. Genuinely verifies (not just asserts) that the structural
/// solver equals brute force on a few concrete cases before reporting PASS; the
/// full d<=4, p<=6 set-equality proof lives in the tests below.
pub fn report() -> String {
    use std::collections::HashSet;
    let cases: [(SignedPerm, u32); 3] = [
        (SignedPerm::identity(4), 4),
        (SignedPerm::identity(3), 2),
        (SignedPerm::from_parts(vec![1, 2, 0], vec![1, -1, 1]).unwrap(), 3),
    ];
    for (a, p) in &cases {
        let structural: HashSet<SignedPerm> = pth_roots(a, *p).into_iter().collect();
        let brute: HashSet<SignedPerm> = brute_pth_roots(a, *p).into_iter().collect();
        if structural != brute {
            return format!("roots: FAIL structural != brute (d={}, p={})", a.dim(), p);
        }
        if !structural.iter().all(|g| super::pow(g, *p) == *a) {
            return "roots: FAIL a returned root does not power back to A".to_string();
        }
    }
    "roots: PASS structural roots == brute force on checked cases; full d<=4, p<=6 proof in tests"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enumerate every signed permutation of dimension d.
    fn all_signed_perms(d: usize) -> Vec<SignedPerm> {
        let mut out = Vec::new();
        for perm in permutations(&(0..d).collect::<Vec<_>>()) {
            let perm16: Vec<u16> = perm.iter().map(|&x| x as u16).collect();
            for sign_bits in 0u32..(1u32 << d) {
                let sign: Vec<i8> = (0..d)
                    .map(|i| if (sign_bits >> i) & 1 == 1 { -1 } else { 1 })
                    .collect();
                out.push(SignedPerm::from_parts(perm16.clone(), sign).unwrap());
            }
        }
        out
    }

    fn cmp_sp(x: &SignedPerm, y: &SignedPerm) -> std::cmp::Ordering {
        (x.perm.clone(), x.sign.clone()).cmp(&(y.perm.clone(), y.sign.clone()))
    }

    /// THE correctness proof: structural roots == brute-force roots as sets, for
    /// every element of dimension d <= 4 and every p in 1..=6.
    #[test]
    fn structural_equals_brute_all_small() {
        for d in 1..=4usize {
            for a in all_signed_perms(d) {
                for p in 1..=6u32 {
                    let mut structural = pth_roots(&a, p);
                    let mut brute = brute_pth_roots(&a, p);
                    structural.sort_by(cmp_sp);
                    brute.sort_by(cmp_sp);
                    assert_eq!(
                        structural, brute,
                        "mismatch at d={} p={} a.perm={:?} a.sign={:?}",
                        d, p, a.perm, a.sign
                    );
                    for g in &structural {
                        assert_eq!(super::super::pow(g, p), a);
                    }
                }
            }
        }
    }

    /// Paper Eqs 4.16 to 4.17: X1 = L2 * L1^{-1} satisfies X1^4 = I, X1^2 = -I, and
    /// X1 is a 4th root of I and a square root of -I.
    #[test]
    fn chiral_holoraumy_roots() {
        let ls = super::super::cm_l_matrices();
        let x1 = super::super::matmul(&ls[1], &ls[0].inverse());
        assert!(super::super::pow(&x1, 4).is_identity(), "X1^4 must be I");
        assert!(super::super::pow(&x1, 2).is_neg_identity(), "X1^2 must be -I");

        let d = x1.dim();
        let id = SignedPerm::identity(d);
        let neg_id = id.negate();

        let fourth_roots_of_id = pth_roots(&id, 4);
        assert!(
            fourth_roots_of_id.contains(&x1),
            "X1 must be a 4th root of the identity"
        );
        let sqrt_neg_id = pth_roots(&neg_id, 2);
        assert!(sqrt_neg_id.contains(&x1), "X1 must be a square root of -I");

        // Cross-check both sets against brute force at d=4.
        let mut a = fourth_roots_of_id.clone();
        let mut b = brute_pth_roots(&id, 4);
        a.sort_by(cmp_sp);
        b.sort_by(cmp_sp);
        assert_eq!(a, b);
        let mut a = sqrt_neg_id.clone();
        let mut b = brute_pth_roots(&neg_id, 2);
        a.sort_by(cmp_sp);
        b.sort_by(cmp_sp);
        assert_eq!(a, b);
    }

    /// Scaling: a 12-dimensional signed permutation is rooted quickly, WITHOUT
    /// touching the 3^144 grid and WITHOUT calling brute force.
    #[test]
    fn scales_to_dimension_twelve() {
        // (0 1 2)(3 4 5 6)(7 8)(9)(10)(11) with assorted signs.
        let perm: Vec<u16> = vec![1, 2, 0, 4, 5, 6, 3, 8, 7, 9, 10, 11];
        let sign: Vec<i8> = vec![1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1];
        let a = SignedPerm::from_parts(perm, sign).unwrap();

        // Non-vacuity guard: p=1 always returns exactly {a}, so the solver is
        // actually producing a concrete root here (a zero-returning bug fails).
        assert_eq!(pth_roots(&a, 1), vec![a.clone()], "p=1 must return exactly {{a}} at d=12");
        for p in [2u32, 3, 4, 6] {
            let roots = pth_roots(&a, p);
            for g in &roots {
                assert_eq!(super::super::pow(g, p), a, "bad root at d=12, p={}", p);
            }
        }
    }

    /// p == 1 returns exactly the element itself.
    #[test]
    fn first_root_is_self() {
        for d in 0..=4usize {
            for a in all_signed_perms(d) {
                let roots = pth_roots(&a, 1);
                assert_eq!(roots.len(), 1);
                assert_eq!(roots[0], a);
            }
        }
    }

    /// Dimension 0 edge case.
    #[test]
    fn dimension_zero() {
        let a = SignedPerm::identity(0);
        assert_eq!(pth_roots(&a, 3), vec![a.clone()]);
    }
}
