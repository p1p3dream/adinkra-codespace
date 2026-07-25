//! Deterministic permutation and permutahedron primitives for `S_n`, `1 <= n <= 8`.
//!
//! A permutation `p` is stored in one-line notation, with `p[i - 1] = p(i)`.
//! Composition is functional: `p.compose(q)` is `p \circ q`, so it applies `q`
//! first and then `p`. The permutahedron edges used here are right
//! multiplication by the adjacent transpositions `s_i = (i,i+1)`. In one-line
//! notation, `p \circ s_i` is equivalently obtained by swapping the entries in
//! adjacent positions `i` and `i+1`.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fmt;

pub const MAX_N: usize = 8;
const FACTORIALS: [usize; MAX_N + 1] = [1, 1, 2, 6, 24, 120, 720, 5_040, 40_320];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermutationError {
    OrderOutOfRange(usize),
    ValueOutOfRange {
        position: usize,
        value: u8,
        n: usize,
    },
    DuplicateValue(u8),
    RankOutOfRange {
        rank: usize,
        n: usize,
    },
    OrderMismatch {
        left: usize,
        right: usize,
    },
    GeneratorOutOfRange {
        generator: usize,
        n: usize,
    },
    EmptySubgroup,
    InvalidSubgroup(String),
}

impl fmt::Display for PermutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrderOutOfRange(n) => write!(f, "permutation order {n} is outside 1..={MAX_N}"),
            Self::ValueOutOfRange { position, value, n } => write!(
                f,
                "entry {value} at zero-based position {position} is outside 1..={n}"
            ),
            Self::DuplicateValue(value) => write!(f, "entry {value} occurs more than once"),
            Self::RankOutOfRange { rank, n } => {
                write!(f, "rank {rank} is outside 0..{} for S_{n}", FACTORIALS[*n])
            }
            Self::OrderMismatch { left, right } => {
                write!(f, "permutation orders differ: {left} and {right}")
            }
            Self::GeneratorOutOfRange { generator, n } => write!(
                f,
                "adjacent generator {generator} is outside 0..{} for S_{n}",
                n.saturating_sub(1)
            ),
            Self::EmptySubgroup => write!(f, "a subgroup cannot be empty"),
            Self::InvalidSubgroup(reason) => write!(f, "invalid subgroup: {reason}"),
        }
    }
}

impl std::error::Error for PermutationError {}

/// Return `n!` for `0 <= n <= 8`.
pub fn factorial(n: usize) -> Result<usize, PermutationError> {
    FACTORIALS
        .get(n)
        .copied()
        .ok_or(PermutationError::OrderOutOfRange(n))
}

/// A validated one-line permutation of at most eight objects.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Permutation {
    n: u8,
    image: [u8; MAX_N],
}

impl fmt::Debug for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Permutation")
            .field(&self.as_slice())
            .finish()
    }
}

impl<'de> Deserialize<'de> for Permutation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            n: u8,
            image: [u8; MAX_N],
        }

        let wire = Wire::deserialize(deserializer)?;
        let n = usize::from(wire.n);
        if !(1..=MAX_N).contains(&n) {
            return Err(serde::de::Error::custom("permutation order outside 1..=8"));
        }
        if wire.image[n..].iter().any(|&x| x != 0) {
            return Err(serde::de::Error::custom(
                "unused permutation storage must contain zeros",
            ));
        }
        Self::new(&wire.image[..n]).map_err(serde::de::Error::custom)
    }
}

impl Permutation {
    /// Validate and construct a permutation from one-line notation.
    pub fn new(one_line: &[u8]) -> Result<Self, PermutationError> {
        let n = one_line.len();
        if !(1..=MAX_N).contains(&n) {
            return Err(PermutationError::OrderOutOfRange(n));
        }
        let mut seen = [false; MAX_N + 1];
        let mut image = [0u8; MAX_N];
        for (position, &value) in one_line.iter().enumerate() {
            let value_index = usize::from(value);
            if value_index == 0 || value_index > n {
                return Err(PermutationError::ValueOutOfRange { position, value, n });
            }
            if seen[value_index] {
                return Err(PermutationError::DuplicateValue(value));
            }
            seen[value_index] = true;
            image[position] = value;
        }
        Ok(Self { n: n as u8, image })
    }

    pub fn identity(n: usize) -> Result<Self, PermutationError> {
        if !(1..=MAX_N).contains(&n) {
            return Err(PermutationError::OrderOutOfRange(n));
        }
        let mut image = [0u8; MAX_N];
        for (i, slot) in image[..n].iter_mut().enumerate() {
            *slot = (i + 1) as u8;
        }
        Ok(Self { n: n as u8, image })
    }

    pub fn n(self) -> usize {
        usize::from(self.n)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.image[..self.n()]
    }

    pub fn one_line(self) -> Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Zero-based lexicographic Lehmer rank.
    pub fn rank(self) -> usize {
        let n = self.n();
        let mut rank = 0usize;
        for i in 0..n {
            let smaller_to_right = self.image[i + 1..n]
                .iter()
                .filter(|&&value| value < self.image[i])
                .count();
            rank += smaller_to_right * FACTORIALS[n - i - 1];
        }
        rank
    }

    /// Invert a zero-based lexicographic Lehmer rank.
    pub fn unrank(n: usize, mut rank: usize) -> Result<Self, PermutationError> {
        if !(1..=MAX_N).contains(&n) {
            return Err(PermutationError::OrderOutOfRange(n));
        }
        if rank >= FACTORIALS[n] {
            return Err(PermutationError::RankOutOfRange { rank, n });
        }
        let mut available: Vec<u8> = (1..=n as u8).collect();
        let mut image = [0u8; MAX_N];
        for (i, slot) in image[..n].iter_mut().enumerate() {
            let block = FACTORIALS[n - i - 1];
            let digit = rank / block;
            rank %= block;
            *slot = available.remove(digit);
        }
        Ok(Self { n: n as u8, image })
    }

    /// Functional composition `self \circ rhs`.
    pub fn compose(self, rhs: Self) -> Result<Self, PermutationError> {
        if self.n != rhs.n {
            return Err(PermutationError::OrderMismatch {
                left: self.n(),
                right: rhs.n(),
            });
        }
        let n = self.n();
        let mut image = [0u8; MAX_N];
        for (i, slot) in image[..n].iter_mut().enumerate() {
            *slot = self.image[usize::from(rhs.image[i] - 1)];
        }
        Ok(Self { n: self.n, image })
    }

    pub fn inverse(self) -> Self {
        let n = self.n();
        let mut image = [0u8; MAX_N];
        for i in 0..n {
            image[usize::from(self.image[i] - 1)] = (i + 1) as u8;
        }
        Self { n: self.n, image }
    }

    /// Right multiply by `s_i=(i+1,i+2)`, using a zero-based generator index.
    /// This is exactly an adjacent-position swap in one-line notation.
    pub fn right_adjacent(self, i: usize) -> Result<Self, PermutationError> {
        let n = self.n();
        if i + 1 >= n {
            return Err(PermutationError::GeneratorOutOfRange { generator: i, n });
        }
        let mut result = self;
        result.image.swap(i, i + 1);
        Ok(result)
    }

    pub fn adjacent_neighbors(self) -> impl Iterator<Item = Self> {
        let n = self.n();
        (0..n.saturating_sub(1)).map(move |i| {
            let mut result = self;
            result.image.swap(i, i + 1);
            result
        })
    }

    pub fn inversion_count(self) -> usize {
        let n = self.n();
        let mut inversions = 0usize;
        for i in 0..n {
            for j in i + 1..n {
                inversions += usize::from(self.image[i] > self.image[j]);
            }
        }
        inversions
    }

    /// Kendall tau distance, equal to graph distance in the weak-Bruhat
    /// permutahedron generated by right adjacent transpositions.
    pub fn kendall_distance(self, rhs: Self) -> Result<usize, PermutationError> {
        Ok(self.inverse().compose(rhs)?.inversion_count())
    }

    pub fn bruhat_distance(self, rhs: Self) -> Result<usize, PermutationError> {
        self.kendall_distance(rhs)
    }
}

/// Iterate over `S_n` in lexicographic one-line order.
pub fn permutations(n: usize) -> Result<impl Iterator<Item = Permutation>, PermutationError> {
    let count = factorial(n)?;
    if n == 0 {
        return Err(PermutationError::OrderOutOfRange(n));
    }
    Ok((0..count)
        .map(move |rank| Permutation::unrank(n, rank).expect("rank is bounded by n factorial")))
}

/// Return the adjacent transposition `s_i=(i+1,i+2)` for a zero-based index.
pub fn adjacent_transposition(n: usize, i: usize) -> Result<Permutation, PermutationError> {
    Permutation::identity(n)?.right_adjacent(i)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutahedronGraph {
    pub n: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub degree: usize,
    /// Undirected edges as ordered pairs of lexicographic vertex ranks.
    pub edges: Vec<[u32; 2]>,
    pub edge_convention: String,
}

/// Generate the complete undirected 1-skeleton without storing adjacency lists.
pub fn complete_graph(n: usize) -> Result<PermutahedronGraph, PermutationError> {
    if !(1..=MAX_N).contains(&n) {
        return Err(PermutationError::OrderOutOfRange(n));
    }
    let vertex_count = FACTORIALS[n];
    let degree = n.saturating_sub(1);
    let edge_count = vertex_count * degree / 2;
    let mut edges = Vec::with_capacity(edge_count);
    for rank in 0..vertex_count {
        let p = Permutation::unrank(n, rank)?;
        for q in p.adjacent_neighbors() {
            let neighbor_rank = q.rank();
            if rank < neighbor_rank {
                edges.push([rank as u32, neighbor_rank as u32]);
            }
        }
    }
    debug_assert_eq!(edges.len(), edge_count);
    Ok(PermutahedronGraph {
        n,
        vertex_count,
        edge_count,
        degree,
        edges,
        edge_convention:
            "right multiplication by adjacent transpositions; equivalently adjacent-position swaps"
                .into(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphAuditReport {
    pub n: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub degree: usize,
    pub connected: bool,
    pub reached_vertices: usize,
    pub diameter: usize,
    pub bfs_source_rank: usize,
}

/// Audit connectivity and diameter with one BFS from the identity.
///
/// The adjacent-transposition Cayley graph is vertex-transitive, so the
/// eccentricity of the identity is its diameter. This avoids an all-pairs BFS.
pub fn bfs_graph_audit(n: usize) -> Result<GraphAuditReport, PermutationError> {
    if !(1..=MAX_N).contains(&n) {
        return Err(PermutationError::OrderOutOfRange(n));
    }
    let vertex_count = FACTORIALS[n];
    let degree = n.saturating_sub(1);
    let edge_count = vertex_count * degree / 2;
    let source = Permutation::identity(n)?.rank();
    let mut distances = vec![usize::MAX; vertex_count];
    distances[source] = 0;
    let mut queue = VecDeque::with_capacity(vertex_count);
    queue.push_back(source);
    while let Some(rank) = queue.pop_front() {
        let distance = distances[rank];
        let p = Permutation::unrank(n, rank)?;
        for q in p.adjacent_neighbors() {
            let next = q.rank();
            if distances[next] == usize::MAX {
                distances[next] = distance + 1;
                queue.push_back(next);
            }
        }
    }
    let reached_vertices = distances.iter().filter(|&&d| d != usize::MAX).count();
    let diameter = distances
        .iter()
        .copied()
        .filter(|&d| d != usize::MAX)
        .max()
        .unwrap_or(0);
    Ok(GraphAuditReport {
        n,
        vertex_count,
        edge_count,
        degree,
        connected: reached_vertices == vertex_count,
        reached_vertices,
        diameter,
        bfs_source_rank: source,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubgroupValidationReport {
    pub n: Option<usize>,
    pub order: usize,
    pub valid: bool,
    pub nonempty: bool,
    pub common_order: bool,
    pub unique: bool,
    pub identity_present: bool,
    pub closed: bool,
    pub inverses_present: bool,
    pub reason: Option<String>,
}

/// Validate the subgroup axioms by exact finite enumeration.
pub fn validate_subgroup(elements: &[Permutation]) -> SubgroupValidationReport {
    let mut report = SubgroupValidationReport {
        n: elements.first().map(|p| p.n()),
        order: elements.len(),
        valid: false,
        nonempty: !elements.is_empty(),
        common_order: false,
        unique: false,
        identity_present: false,
        closed: false,
        inverses_present: false,
        reason: None,
    };
    if elements.is_empty() {
        report.reason = Some("subgroup is empty".into());
        return report;
    }
    let n = elements[0].n();
    report.common_order = elements.iter().all(|p| p.n() == n);
    if !report.common_order {
        report.reason = Some("elements have different permutation orders".into());
        return report;
    }
    let ranks: HashSet<usize> = elements.iter().map(|p| p.rank()).collect();
    report.unique = ranks.len() == elements.len();
    if !report.unique {
        report.reason = Some("subgroup contains duplicate elements".into());
        return report;
    }
    let identity_rank = Permutation::identity(n).expect("validated order").rank();
    report.identity_present = ranks.contains(&identity_rank);
    if !report.identity_present {
        report.reason = Some("identity is absent".into());
        return report;
    }
    report.inverses_present = elements.iter().all(|p| ranks.contains(&p.inverse().rank()));
    if !report.inverses_present {
        report.reason = Some("an inverse is absent".into());
        return report;
    }
    report.closed = elements.iter().all(|&p| {
        elements
            .iter()
            .all(|&q| ranks.contains(&p.compose(q).expect("common order was established").rank()))
    });
    if !report.closed {
        report.reason = Some("set is not closed under composition".into());
        return report;
    }
    report.valid = true;
    report
}

fn require_subgroup(elements: &[Permutation]) -> Result<usize, PermutationError> {
    let report = validate_subgroup(elements);
    if report.valid {
        Ok(report.n.expect("a valid subgroup is nonempty"))
    } else if elements.is_empty() {
        Err(PermutationError::EmptySubgroup)
    } else {
        Err(PermutationError::InvalidSubgroup(
            report.reason.unwrap_or_else(|| "unknown failure".into()),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CosetSide {
    /// `H*g = {h \circ g : h in H}`.
    Right,
    /// `g*H = {g \circ h : h in H}`.
    Left,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CosetPartitionReport {
    pub n: usize,
    pub subgroup_order: usize,
    pub side: CosetSide,
    pub slice_count: usize,
    pub slice_size: usize,
    pub covered_vertices: usize,
    pub complete_cover: bool,
    /// Each slice and its elements are ordered by lexicographic Lehmer rank.
    pub slices: Vec<Vec<u32>>,
}

/// Partition `S_n` into deterministic cosets. Seeds are the first uncovered
/// lexicographic ranks; elements within each coset are sorted by rank.
pub fn coset_partition(
    subgroup: &[Permutation],
    side: CosetSide,
) -> Result<CosetPartitionReport, PermutationError> {
    let n = require_subgroup(subgroup)?;
    let vertex_count = FACTORIALS[n];
    let mut covered = vec![false; vertex_count];
    let mut slices = Vec::with_capacity(vertex_count / subgroup.len());
    for seed_rank in 0..vertex_count {
        if covered[seed_rank] {
            continue;
        }
        let seed = Permutation::unrank(n, seed_rank)?;
        let mut slice: Vec<u32> = subgroup
            .iter()
            .map(|&h| match side {
                CosetSide::Right => h.compose(seed),
                CosetSide::Left => seed.compose(h),
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|p| p.rank() as u32)
            .collect();
        slice.sort_unstable();
        slice.dedup();
        if slice.len() != subgroup.len() {
            return Err(PermutationError::InvalidSubgroup(
                "coset has the wrong cardinality".into(),
            ));
        }
        for &rank in &slice {
            covered[rank as usize] = true;
        }
        slices.push(slice);
    }
    let covered_vertices = covered.iter().filter(|&&is_covered| is_covered).count();
    let complete_cover =
        covered_vertices == vertex_count && slices.len() * subgroup.len() == vertex_count;
    Ok(CosetPartitionReport {
        n,
        subgroup_order: subgroup.len(),
        side,
        slice_count: slices.len(),
        slice_size: subgroup.len(),
        covered_vertices,
        complete_cover,
        slices,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbnormalSliceReport {
    pub n: usize,
    pub subgroup_order: usize,
    pub right_slice_count: usize,
    pub abnormal_slice_count: usize,
    pub complete_cover: bool,
    /// Right cosets `H*g` that equal their corresponding left cosets `g*H`.
    pub abnormal_slices: Vec<Vec<u32>>,
    /// Lexicographically least representative rank for each abnormal slice.
    pub representative_ranks: Vec<u32>,
}

/// Detect the paper's "ab-normal" slices: right cosets `H*g` that are equal,
/// as sets, to the left cosets `g*H` for the same representative. Under the
/// subgroup axioms this equality is independent of the representative chosen
/// from the slice, so testing its canonical least-rank representative is
/// equivalent to testing every element of the slice.
pub fn abnormal_slices(subgroup: &[Permutation]) -> Result<AbnormalSliceReport, PermutationError> {
    let n = require_subgroup(subgroup)?;
    let right = coset_partition(subgroup, CosetSide::Right)?;
    let mut found = Vec::new();
    let mut representatives = Vec::new();
    for slice in &right.slices {
        let representative_rank = slice[0];
        let g = Permutation::unrank(n, representative_rank as usize)?;
        let mut left: Vec<u32> = subgroup
            .iter()
            .map(|&h| g.compose(h).expect("orders agree").rank() as u32)
            .collect();
        left.sort_unstable();
        if left == *slice {
            representatives.push(representative_rank);
            found.push(slice.clone());
        }
    }
    Ok(AbnormalSliceReport {
        n,
        subgroup_order: subgroup.len(),
        right_slice_count: right.slice_count,
        abnormal_slice_count: found.len(),
        complete_cover: right.complete_cover,
        abnormal_slices: found,
        representative_ranks: representatives,
    })
}

/// Klein's Vierergruppe in the one-line convention used by the papers.
pub fn vierergruppe() -> Vec<Permutation> {
    [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]]
        .iter()
        .map(|row| Permutation::new(row).expect("constant is a permutation"))
        .collect()
}

/// The order-eight Rana subgroup `R_8` printed in arXiv:2304.09830, Eq. (3.23).
pub fn rana_r8() -> Vec<Permutation> {
    [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 1, 4, 3, 6, 5, 8, 7],
        [3, 4, 1, 2, 7, 8, 5, 6],
        [4, 3, 2, 1, 8, 7, 6, 5],
        [5, 6, 7, 8, 1, 2, 3, 4],
        [6, 5, 8, 7, 2, 1, 4, 3],
        [7, 8, 5, 6, 3, 4, 1, 2],
        [8, 7, 6, 5, 4, 3, 2, 1],
    ]
    .iter()
    .map(|row| Permutation::new(row).expect("constant is a permutation"))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn rejects_invalid_one_line_data() {
        assert!(matches!(
            Permutation::new(&[]),
            Err(PermutationError::OrderOutOfRange(0))
        ));
        assert!(matches!(
            Permutation::new(&[1, 1]),
            Err(PermutationError::DuplicateValue(1))
        ));
        assert!(matches!(
            Permutation::new(&[1, 3]),
            Err(PermutationError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn rank_unrank_round_trip_through_s8() {
        for n in 1..=8 {
            for rank in 0..FACTORIALS[n] {
                let p = Permutation::unrank(n, rank).unwrap();
                assert_eq!(p.rank(), rank);
            }
        }
    }

    #[test]
    fn composition_convention_and_adjacent_right_action_are_explicit() {
        let p = Permutation::new(&[2, 3, 1, 4]).unwrap();
        let q = Permutation::new(&[2, 1, 4, 3]).unwrap();
        assert_eq!(p.compose(q).unwrap().as_slice(), &[3, 2, 4, 1]);
        assert_eq!(p.right_adjacent(1).unwrap().as_slice(), &[2, 1, 3, 4]);
        let s_1 = adjacent_transposition(4, 1).unwrap();
        assert_eq!(p.compose(s_1).unwrap(), p.right_adjacent(1).unwrap());
        assert_eq!(
            p.compose(p.inverse()).unwrap(),
            Permutation::identity(4).unwrap()
        );
    }

    #[test]
    fn kendall_and_weak_bruhat_distance_agree() {
        let identity = Permutation::identity(4).unwrap();
        let reverse = Permutation::new(&[4, 3, 2, 1]).unwrap();
        assert_eq!(identity.kendall_distance(reverse).unwrap(), 6);
        assert_eq!(identity.bruhat_distance(reverse).unwrap(), 6);
        assert_eq!(reverse.kendall_distance(identity).unwrap(), 6);
    }

    #[test]
    fn s4_graph_has_published_size_degree_and_diameter() {
        let graph = complete_graph(4).unwrap();
        assert_eq!(graph.vertex_count, 24);
        assert_eq!(graph.edge_count, 36);
        assert_eq!(graph.edges.len(), 36);
        assert_eq!(graph.degree, 3);
        assert!(graph.edges.iter().all(|edge| edge[0] < edge[1]));

        let audit = bfs_graph_audit(4).unwrap();
        assert!(audit.connected);
        assert_eq!(audit.reached_vertices, 24);
        assert_eq!(audit.diameter, 6);
    }

    #[test]
    fn s8_graph_has_complete_expected_size() {
        let graph = complete_graph(8).unwrap();
        assert_eq!(graph.vertex_count, 40_320);
        assert_eq!(graph.edge_count, 141_120);
        assert_eq!(graph.edges.len(), 141_120);
        assert_eq!(graph.degree, 7);
        let distinct: BTreeSet<[u32; 2]> = graph.edges.iter().copied().collect();
        assert_eq!(distinct.len(), graph.edges.len());
    }

    #[test]
    fn subgroup_validation_accepts_v_and_r8() {
        let v = vierergruppe();
        let r8 = rana_r8();
        assert!(validate_subgroup(&v).valid);
        assert!(validate_subgroup(&r8).valid);
        assert_eq!(validate_subgroup(&v).order, 4);
        assert_eq!(validate_subgroup(&r8).order, 8);
    }

    #[test]
    fn v_cosets_are_all_abnormal_because_v_is_normal() {
        let v = vierergruppe();
        let right = coset_partition(&v, CosetSide::Right).unwrap();
        let left = coset_partition(&v, CosetSide::Left).unwrap();
        assert_eq!(right.slice_count, 6);
        assert_eq!(right.slices, left.slices);
        let abnormal = abnormal_slices(&v).unwrap();
        assert_eq!(abnormal.abnormal_slice_count, 6);
    }

    #[test]
    fn r8_partitions_s8_and_has_168_abnormal_slices() {
        let r8 = rana_r8();
        let right = coset_partition(&r8, CosetSide::Right).unwrap();
        let left = coset_partition(&r8, CosetSide::Left).unwrap();
        assert_eq!(right.slice_count, 5_040);
        assert_eq!(left.slice_count, 5_040);
        assert_eq!(right.slice_size, 8);
        assert_eq!(left.slice_size, 8);
        assert_eq!(right.covered_vertices, 40_320);
        assert_eq!(left.covered_vertices, 40_320);
        assert!(right.complete_cover);
        assert!(left.complete_cover);
        assert_ne!(right.slices, left.slices);

        let abnormal = abnormal_slices(&r8).unwrap();
        assert!(abnormal.complete_cover);
        assert_eq!(abnormal.right_slice_count, 5_040);
        assert_eq!(abnormal.abnormal_slice_count, 168);
        assert!(
            abnormal
                .abnormal_slices
                .iter()
                .all(|slice| slice.len() == 8)
        );
    }

    #[test]
    fn reports_serialize_deterministically() {
        let report = bfs_graph_audit(4).unwrap();
        let first = serde_json::to_string(&report).unwrap();
        let second = serde_json::to_string(&report).unwrap();
        assert_eq!(first, second);
    }
}
